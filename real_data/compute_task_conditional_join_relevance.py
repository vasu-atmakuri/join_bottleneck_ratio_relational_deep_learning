#!/usr/bin/env python3
"""
Task-conditional relevance of every foreign-key join that aggregates into a task's entity table (RelBench v1 entity
classification and regression tasks; recommendation tasks are excluded because their labels are entity sets).

For each task and each one-hop join (child table -> entity table):
  1. take up to --max-train-rows rows of the task's training split (entity id, timestamp, label);
  2. read only the child rows that point to those entities and are older than the validation timestamp
     (the training timestamps are all earlier than it), streaming row groups with pyarrow;
  3. stream those child rows in batches, discretize every feature column (quantile bins, categories or hashed
     buckets, fixed on the first batch) and accumulate, per entity, the normalized histogram of each feature plus
     log(1 + number of children); memory stays bounded by the number of entities, not the number of child rows;
  4. relevance = 3-fold cross-validated predictive skill of those aggregated join features alone for the label:
     classification 2 * (AUROC - 0.5) clipped at 0 (logistic regression), regression R^2 clipped at 0 (ridge).

Outputs (results/relbench_task_conditional/)
  join_task_relevance.csv / .jsonl   one row per task x join
  extraction_log.txt
Datasets are downloaded one at a time and deleted afterwards. Finished tasks are skipped on restart.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute  # noqa: F401
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "real_data"))
import extract_relbench_join_statistics as ex  # noqa: E402

OUT = ROOT / "results" / "relbench_task_conditional"
BASELINES = ROOT / "real_data" / "published_relbench_baseline_results.csv"
MAX_FEATURES = 20
RNG = np.random.default_rng(20261106)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "extraction_log.txt", "a") as f:
        f.write(line + "\n")


def relevance(X, y, classification):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    pred = np.zeros(len(y))
    if classification:
        if len(np.unique(y)) < 2:
            return float("nan"), float("nan")
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
            m = LogisticRegression(max_iter=500, C=0.1).fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        auc = roc_auc_score(y, pred)
        return max(0.0, 2 * (auc - 0.5)), auc
    yt = (y - y.mean()) / (y.std() + 1e-12)
    for tr, te in KFold(3, shuffle=True, random_state=0).split(X):
        pred[te] = Ridge(alpha=10.0).fit(X[tr], yt[tr]).predict(X[te])
    r2 = r2_score(yt, pred)
    return max(0.0, r2), r2


class StreamingEncoder:
    """Discretizes a column consistently across row batches: numeric and time columns use quantile edges taken from
    the first batch; low-cardinality categorical columns keep the categories of the first batch (others -> 'other');
    high-cardinality and text columns are hashed into N_BINS buckets. Missing values are code 0."""

    def __init__(self):
        self.kind, self.edges, self.mapping = None, None, None

    def encode(self, arr):
        n = len(arr)
        missing = np.asarray(arr.is_null().to_numpy(zero_copy_only=False), dtype=bool)
        codes = np.zeros(n, dtype=np.int16)
        valid = np.nonzero(~missing)[0]
        t = arr.type
        if pa.types.is_timestamp(t) or pa.types.is_date(t):
            arr = pa.compute.cast(arr, pa.int64(), safe=False) if pa.types.is_timestamp(t) else \
                pa.compute.cast(pa.compute.cast(arr, pa.date32()), pa.int32())
            t = arr.type
        numeric = pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t) or pa.types.is_decimal(t)
        if self.kind is None:
            self.kind = "numeric" if numeric else "categorical"
        if len(valid) == 0:
            return codes
        if self.kind == "numeric":
            vals = np.asarray(pa.compute.cast(arr, pa.float64(), safe=False).to_numpy(zero_copy_only=False),
                              dtype=float)[valid]
            if self.edges is None:
                fin = vals[np.isfinite(vals)]
                self.edges = (np.unique(np.quantile(fin, np.linspace(0, 1, ex.N_BINS + 1)[1:-1]))
                              if len(fin) else np.array([0.0]))
            codes[valid] = np.searchsorted(self.edges, vals, side="right") + 1
        else:
            s = pd.Series(arr.to_pylist(), dtype="object").iloc[valid].astype(str)
            if self.mapping is None:
                vc = s.value_counts()
                self.mapping = ({v: i + 2 for i, v in enumerate(vc.index)} if len(vc) <= ex.TOP_CATEGORIES
                                else "hash")
            if self.mapping == "hash":
                codes[valid] = (pd.util.hash_pandas_object(s, index=False).values % ex.N_BINS).astype(np.int16) + 1
            else:
                codes[valid] = s.map(self.mapping).fillna(1).astype(np.int16).values
        return codes


def join_features(path, fk_col, time_col, feats, entity_ids, cutoff, batch_rows=200_000):
    """Per-entity normalized histograms of the child features, streamed in row batches (bounded memory)."""
    dset = pads.dataset(path, format="parquet")
    fk_type = dset.schema.field(fk_col).type
    try:
        keys = pa.array(entity_ids).cast(fk_type)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        keys = pa.array([str(e) for e in entity_ids]).cast(fk_type)
    flt = pads.field(fk_col).isin(keys)
    if time_col:
        ts = pa.scalar(pd.Timestamp(cutoff).to_pydatetime(), type=pa.timestamp("us"))
        flt = flt & (pads.field(time_col).cast(pa.timestamp("us")) < ts)
    encoders = {c: StreamingEncoder() for c in feats}
    parts = {c: [] for c in feats}
    count_parts, n_rows = [], 0
    for batch in dset.to_batches(columns=[fk_col] + feats, filter=flt, batch_size=batch_rows):
        if batch.num_rows == 0:
            continue
        k = pd.Series(batch.column(0).to_pylist()).astype(str).values
        n_rows += batch.num_rows
        count_parts.append(pd.Series(k).value_counts())
        for i, c in enumerate(feats, start=1):
            codes = encoders[c].encode(batch.column(i))
            parts[c].append(pd.DataFrame({"k": k, "v": codes}).groupby(["k", "v"]).size())
        del batch
    if not count_parts:
        return pd.DataFrame(), 0
    counts = pd.concat(count_parts).groupby(level=0).sum()
    blocks = []
    for c in feats:
        h = pd.concat(parts[c]).groupby(level=[0, 1]).sum().unstack(fill_value=0).astype(np.float32)
        h = h.div(h.sum(axis=1), axis=0)
        h.columns = [f"{c}={v}" for v in h.columns]
        blocks.append(h)
    agg = pd.concat(blocks, axis=1).reindex(counts.index).fillna(0.0) if blocks else pd.DataFrame(index=counts.index)
    agg["log_child_count"] = np.log1p(counts.values)
    return agg, n_rows


def process_dataset(name, repo, tasks_wanted, args, done):
    t0 = time.time()
    ddir = ex.download(repo, name, ["manifest.yaml", "db/*"] + [f"tasks/{t}/*" for t in tasks_wanted])
    man = yaml.safe_load(open(ddir / "manifest.yaml"))
    tables = man["tables"]
    val_ts = man["val_timestamp"]
    log(f"{name}: downloaded in {time.time() - t0:.0f}s")
    rows = []
    for task in tasks_wanted:
        if (name, task) in done:
            continue
        tdir = ddir / "tasks" / task
        if not (tdir / "manifest.yaml").exists():
            for repo2 in ex.REPOS:  # task hosted in the other repository
                try:
                    ex.download(repo2, name, [f"tasks/{task}/*"])
                except Exception:
                    pass
                cand = ex.CACHE / repo2.replace("/", "__") / name / "tasks" / task
                if (cand / "manifest.yaml").exists():
                    tdir = cand
                    break
        if not (tdir / "manifest.yaml").exists():
            log(f"  {task}: task files not found")
            continue
        tm = yaml.safe_load(open(tdir / "manifest.yaml"))
        entity, ecol, tcol, ycol = tm.get("entity_table"), tm.get("entity_col"), tm.get("time_col"), tm.get("target_col")
        train = pd.read_parquet(tdir / "train.parquet", columns=[c for c in (ecol, tcol, ycol) if c])
        if len(train) > args.max_train_rows:
            train = train.sample(args.max_train_rows, random_state=0)
        classification = tm["task_type"] == "binary_classification"
        y = train[ycol].astype(float).values
        ent = train[ecol].astype(str).values
        for child, spec in tables.items():
            for fk_col, parent in (spec.get("fkeys") or {}).items():
                if parent != entity:
                    continue
                path = ddir / "db" / f"{child}.parquet"
                cols = pq.ParquetFile(path).schema_arrow.names
                feats = [c for c in cols if c != spec.get("pkey") and c not in (spec.get("fkeys") or {})
                         and c != spec.get("time_col")][:MAX_FEATURES]
                # features of the child's other parents reach the entity through one-to-one lookups
                ts = time.time()
                try:
                    agg, n_child = join_features(path, fk_col, spec.get("time_col"), feats,
                                                 list(pd.unique(train[ecol])), val_ts)
                    X = agg.reindex(ent).fillna(0.0).values
                    rel, skill = relevance(X, y, classification)
                except Exception as e:
                    log(f"  {task} {child}.{fk_col}: FAILED {e!r}")
                    continue
                row = {"dataset": name, "task": task, "task_type": tm["task_type"], "entity_table": entity,
                       "child_table": child, "fk_column": fk_col, "n_train_rows": len(train),
                       "n_child_rows_used": int(n_child), "n_aggregated_features": int(X.shape[1]),
                       "relevance": rel, "skill_raw": skill}
                rows.append(row)
                log(f"  {task} {child}.{fk_col}: relevance {rel:.3f} (raw {skill:.3f}), {n_child} child rows, "
                    f"{X.shape[1]} features ({time.time() - ts:.0f}s)")
        with open(OUT / "join_task_relevance.jsonl", "a") as f:
            for r in [r for r in rows if r["task"] == task]:
                f.write(json.dumps(r, default=float) + "\n")
            if not any(r["task"] == task for r in rows):
                f.write(json.dumps({"dataset": name, "task": task, "no_joins": True}) + "\n")
    if not args.keep_data:
        shutil.rmtree(ddir, ignore_errors=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-train-rows", type=int, default=20_000)
    ap.add_argument("--keep-data", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ex.OUT = OUT  # extraction helpers log next to these results
    base = pd.read_csv(BASELINES)
    base = base[base.task_type != "recommendation"]
    jsonl = OUT / "join_task_relevance.jsonl"
    done = set()
    if jsonl.exists():
        for line in jsonl.open():
            r = json.loads(line)
            done.add((r["dataset"], r["task"]))
    from huggingface_hub import HfApi
    found = ex.list_datasets(HfApi())
    for name, g in base.groupby("dataset"):
        wanted = [t for t in g.task if (name, t) not in done]
        if not wanted:
            log(f"{name}: all tasks done")
            continue
        try:
            process_dataset(name, found.get(name, ex.REPOS[0]), wanted, args, done)
        except Exception as e:
            import traceback
            log(f"{name}: FAILED {e!r}\n{traceback.format_exc()}")
    rows = [json.loads(l) for l in jsonl.open()] if jsonl.exists() else []
    pd.DataFrame([r for r in rows if not r.get("no_joins")]).to_csv(OUT / "join_task_relevance.csv", index=False)
    log("TASK_CONDITIONAL_EXTRACTION_COMPLETE")


if __name__ == "__main__":
    main()
