#!/usr/bin/env python3
"""
Extracts foreign-key join statistics and per-task Join Bottleneck Ratio (JBR) features from the RelBench databases.

Datasets are downloaded one at a time from the RelBench Hugging Face repositories (database parquet files, the
dataset manifest and the task manifests only; no label files), processed column by column with pyarrow so memory
stays low, and deleted afterwards unless --keep-data is given.

Per foreign key (child table -> parent table)
  K_mean_linked       mean number of child rows per parent that has at least one child
  K_mean_all          mean over all parent rows (zeros included); K_median, K_p90, K_max, K_cv, zero_child_fraction
  D                   number of child feature columns (all columns except primary key, foreign keys, time column)
  H_row_marginal      sum over child features of the plug-in entropy (bits) of the discretized feature
  H_row_conditional   the same with the shuffle-corrected mutual information with the parent subtracted per feature
                      (an estimate of sum_f H(f | parent))
  missing_rate        mean fraction of missing values over child features
  JBR_d{64,128,256}   K_mean_linked * H_row_conditional / d
  temporal statistics K_mean_linked and H_row_conditional before the validation timestamp and after the test
                      timestamp when the child table has a time column (used for the temporal-shift analysis)

Discretization: numeric and time columns use up to 32 quantile bins; categorical columns with at most 32 distinct
values keep them; higher-cardinality categorical and text columns are hashed into 32 buckets; missing values form
their own category. Each feature therefore contributes at most log2(33) bits. Entropies are estimated on a uniform row sample (default 200,000 rows) with the Miller-Madow correction.

Per task (entity classification / regression; recommendation tasks use the source entity table)
  JBR of the joins within one and two hops of the entity table (max, sum, mean, entropy-weighted mean),
  and schema-level baselines: tables and joins within two hops, maximum join depth, schema size, average
  clustering coefficient and degree of the schema graph.

Outputs (results/relbench_join_statistics/)
  foreign_key_join_statistics.csv/.jsonl, task_join_features.csv/.jsonl, dataset_schema_summary.json,
  extraction_log.txt (finished datasets are skipped on restart)
Usage
  python real_data/extract_relbench_join_statistics.py                 # all datasets in both RelBench repositories
  python real_data/extract_relbench_join_statistics.py --datasets rel-f1 rel-trial
"""

import argparse
import json
import math
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "relbench_join_statistics"
CACHE = ROOT / "data" / "relbench_download_cache"
REPOS = ("stanford-star/relbench-v1", "stanford-star/relbench-v2-extra")
EMBED_DIMS = (64, 128, 256)
N_BINS = 32
TOP_CATEGORIES = 31
RNG = np.random.default_rng(20261105)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "extraction_log.txt", "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- entropy estimation
def miller_madow_entropy(codes):
    """Entropy in bits of an integer-coded sample, with the Miller-Madow bias correction."""
    if len(codes) == 0:
        return 0.0
    codes = np.asarray(codes)
    if codes.min() < 0:
        codes = codes - codes.min()
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    h = -np.sum(p * np.log2(p))
    return float(h + (len(counts) - 1) / (2 * counts.sum() * math.log(2)))


def discretize(arr):
    """Map a pyarrow array to integer codes (quantile bins or top categories; missing is its own code)."""
    n = len(arr)
    missing = np.asarray(arr.is_null().to_numpy(zero_copy_only=False), dtype=bool)
    codes = np.zeros(n, dtype=np.int64)
    t = arr.type
    valid_idx = np.nonzero(~missing)[0]
    if len(valid_idx) == 0:
        return codes, float(missing.mean())
    if pa.types.is_timestamp(t) or pa.types.is_date(t):
        arr = pc.cast(arr, pa.int64()) if pa.types.is_timestamp(t) else pc.cast(pc.cast(arr, pa.date32()), pa.int32())
        t = arr.type
    if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t) or pa.types.is_decimal(t):
        vals = np.asarray(pc.cast(arr, pa.float64(), safe=False).to_numpy(zero_copy_only=False), dtype=float)[valid_idx]
        finite = np.isfinite(vals)
        uniq = np.unique(vals[finite])
        if len(uniq) <= N_BINS:
            c = np.searchsorted(uniq, vals)
        else:
            edges = np.unique(np.quantile(vals[finite], np.linspace(0, 1, N_BINS + 1)[1:-1]))
            c = np.searchsorted(edges, vals, side="right")
        codes[valid_idx] = c + 1
    else:
        s = pd.Series(arr.to_pylist(), dtype="object").iloc[valid_idx].astype(str)
        if s.nunique() <= TOP_CATEGORIES + 1:
            mapping = {v: i + 1 for i, v in enumerate(s.value_counts().index)}
            codes[valid_idx] = s.map(mapping).astype(np.int64).values
        else:
            # high-cardinality categorical or free text: hash values into N_BINS buckets, so the column
            # contributes its diversity at the same resolution as a numeric column
            codes[valid_idx] = (pd.util.hash_pandas_object(s, index=False).values % N_BINS).astype(np.int64) + 1
    return codes, float(missing.mean())


def conditional_entropy(codes, parent_codes):
    """H(f) - shuffle-corrected I(f; parent), floored at zero."""
    h = miller_madow_entropy(codes)
    if parent_codes is None or len(codes) < 50:
        return h, h
    joint = codes * (parent_codes.max() + 1) + parent_codes
    _, joint = np.unique(joint, return_inverse=True)
    i_plug = h + miller_madow_entropy(parent_codes) - miller_madow_entropy(joint)
    perm = RNG.permutation(len(parent_codes))
    joint_s = codes * (parent_codes.max() + 1) + parent_codes[perm]
    _, joint_s = np.unique(joint_s, return_inverse=True)
    i_shuf = h + miller_madow_entropy(parent_codes) - miller_madow_entropy(joint_s)
    return h, max(0.0, h - max(0.0, i_plug - i_shuf))


# ---------------------------------------------------------------- data access
def list_datasets(api):
    found = {}
    for repo in REPOS:
        try:
            files = api.list_repo_files(repo, repo_type="dataset")
        except Exception as e:
            log(f"could not list {repo}: {e!r}")
            continue
        for f in files:
            parts = f.split("/")
            if len(parts) == 2 and parts[1] == "manifest.yaml":
                found.setdefault(parts[0], repo)
    return found


def download(repo, name, patterns):
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=CACHE / repo.replace("/", "__"),
                             allow_patterns=[f"{name}/{p}" for p in patterns])
    return Path(path) / name


def task_manifests(api, name):
    """Task manifests may live in either repository."""
    from huggingface_hub import hf_hub_download
    out = {}
    for repo in REPOS:
        try:
            files = api.list_repo_files(repo, repo_type="dataset")
        except Exception:
            continue
        for f in files:
            p = f.split("/")
            if len(p) == 4 and p[0] == name and p[1] == "tasks" and p[3] == "manifest.yaml" and p[2] not in out:
                local = hf_hub_download(repo, f, repo_type="dataset", local_dir=CACHE / repo.replace("/", "__"))
                out[p[2]] = yaml.safe_load(open(local))
    return out


def sample_rows(pf, n_rows, columns, k):
    """Uniform row sample of the given columns (reads only the needed row groups)."""
    if n_rows <= k:
        return pf.read(columns=columns)
    idx = np.sort(RNG.choice(n_rows, size=k, replace=False))
    pieces, start = [], 0
    for rg in range(pf.num_row_groups):
        m = pf.metadata.row_group(rg).num_rows
        sel = idx[(idx >= start) & (idx < start + m)] - start
        if len(sel):
            pieces.append(pf.read_row_group(rg, columns=columns).take(pa.array(sel)))
        start += m
    return pa.concat_tables(pieces)


# ---------------------------------------------------------------- per dataset
def process_dataset(name, repo, api, args):
    t0 = time.time()
    ddir = download(repo, name, ["manifest.yaml", "db/*"])
    man = yaml.safe_load(open(ddir / "manifest.yaml"))
    tables = man.get("tables", {})
    val_ts, test_ts = pd.Timestamp(man["val_timestamp"]), pd.Timestamp(man["test_timestamp"])
    files = {t: ddir / "db" / f"{t}.parquet" for t in tables}
    n_rows = {t: pq.ParquetFile(p).metadata.num_rows for t, p in files.items() if p.exists()}
    log(f"{name}: {len(tables)} tables, rows {n_rows}, downloaded in {time.time() - t0:.0f}s")

    fk_rows = []
    for child, spec in tables.items():
        if child not in n_rows:
            continue
        pf = pq.ParquetFile(files[child])
        cols = pf.schema_arrow.names
        fkeys = spec.get("fkeys") or {}
        tcol = spec.get("time_col")
        feats = [c for c in cols if c != spec.get("pkey") and c not in fkeys and c != tcol]
        for fk_col, parent in fkeys.items():
            if parent not in n_rows:
                continue
            ts = time.time()
            fk = pf.read(columns=[fk_col]).column(0)
            vc = pc.value_counts(pc.drop_null(fk))
            counts = np.asarray(vc.field("counts").to_numpy(), dtype=float)
            n_parent = n_rows[parent]
            zero_frac = max(0.0, 1 - len(counts) / n_parent) if n_parent else float("nan")
            k_all = np.concatenate([counts, np.zeros(max(0, n_parent - len(counts)))])
            sample_cols = [fk_col] + feats + ([tcol] if tcol else [])
            smp = sample_rows(pf, n_rows[child], sample_cols, args.sample_rows)
            pcodes = None
            if smp.num_rows:
                pv = pd.Series(smp.column(fk_col).to_pylist(), dtype="object").astype(str)  # missing keys become "None"
                pcodes = pd.factorize(pv)[0].astype(np.int64)
            h_marg, h_cond, miss = 0.0, 0.0, []
            per_feature = {}
            for c in feats:
                try:
                    codes, m = discretize(smp.column(c))
                    hm, hc = conditional_entropy(codes, pcodes)
                except Exception as e:  # an unusual column type must not stop the dataset
                    log(f"    skipped feature {child}.{c}: {e!r}")
                    continue
                h_marg += hm
                h_cond += hc
                miss.append(m)
                per_feature[c] = round(hc, 4)
            row = {"dataset": name, "repository": repo, "child_table": child, "fk_column": fk_col,
                   "parent_table": parent, "child_rows": n_rows[child], "parent_rows": n_parent,
                   "K_mean_linked": float(counts.mean()) if len(counts) else 0.0,
                   "K_mean_all": float(k_all.mean()) if len(k_all) else 0.0,
                   "K_median": float(np.median(counts)) if len(counts) else 0.0,
                   "K_p90": float(np.quantile(counts, .9)) if len(counts) else 0.0,
                   "K_max": float(counts.max()) if len(counts) else 0.0,
                   "K_cv": float(counts.std() / counts.mean()) if len(counts) and counts.mean() else 0.0,
                   "zero_child_fraction": zero_frac, "D": len(feats),
                   "H_row_marginal": h_marg, "H_row_conditional": h_cond,
                   "missing_rate": float(np.mean(miss)) if miss else 0.0,
                   "child_has_time_column": bool(tcol), "feature_conditional_entropy_bits": json.dumps(per_feature)}
            for d in EMBED_DIMS:
                row[f"JBR_d{d}"] = row["K_mean_linked"] * h_cond / d
            if tcol:
                full = pf.read(columns=[fk_col, tcol])
                full_t = pd.to_datetime(full.column(tcol).to_pandas(), errors="coerce")
                full_fk = full.column(fk_col).to_pandas()
                tvals = pd.to_datetime(pd.Series(smp.column(tcol).to_pylist()), errors="coerce")
                for period, mask, full_mask in (("train_period", tvals < val_ts, full_t < val_ts),
                                                ("test_period", tvals >= test_ts, full_t >= test_ts)):
                    mask = mask.values
                    if mask.sum() < 100:
                        continue
                    sub_fk = pd.Series(smp.column(fk_col).to_pylist())[mask]
                    # exact children per linked parent within the period, from the full key and time columns
                    vc_p = full_fk[full_mask.values].value_counts()
                    row[f"K_mean_linked_{period}"] = float(vc_p.mean()) if len(vc_p) else 0.0
                    hc_p = 0.0
                    pc_p = pd.factorize(sub_fk.astype(str))[0].astype(np.int64)
                    for c in feats:
                        try:
                            codes, _ = discretize(smp.column(c).filter(pa.array(mask)))
                            hc_p += conditional_entropy(codes, pc_p)[1]
                        except Exception:
                            continue
                    row[f"H_row_conditional_{period}"] = hc_p
            fk_rows.append(row)
            log(f"  {child}.{fk_col} -> {parent}: K_mean {row['K_mean_linked']:.1f}, D {len(feats)}, "
                f"H_cond {h_cond:.1f} bits, JBR_d128 {row['JBR_d128']:.3f} ({time.time() - ts:.0f}s)")
            del fk, vc, smp
            if tcol:
                del full, full_t, full_fk

    # own-feature entropy of every table (used for information that reaches a join through one-to-one lookups)
    table_rows = []
    for t, spec in tables.items():
        if t not in n_rows:
            continue
        pf = pq.ParquetFile(files[t])
        fkeys = spec.get("fkeys") or {}
        feats = [c for c in pf.schema_arrow.names
                 if c != spec.get("pkey") and c not in fkeys and c != spec.get("time_col")]
        h = 0.0
        if feats and n_rows[t]:
            smp = sample_rows(pf, n_rows[t], feats, args.sample_rows)
            for c in feats:
                try:
                    h += miller_madow_entropy(discretize(smp.column(c))[0])
                except Exception as e:
                    log(f"    skipped table feature {t}.{c}: {e!r}")
        table_rows.append({"dataset": name, "table": t, "rows": n_rows[t], "D": len(feats),
                           "H_row_marginal": h, "foreign_keys": json.dumps(fkeys)})

    tasks = task_manifests(api, name)
    if not args.keep_data:
        shutil.rmtree(ddir / "db", ignore_errors=True)
    return man, fk_rows, tasks, table_rows


# ---------------------------------------------------------------- per task features
def schema_graph(man):
    adj = {t: set() for t in man["tables"]}
    edges = []
    for child, spec in man["tables"].items():
        for _, parent in (spec.get("fkeys") or {}).items():
            if parent in adj:
                adj[child].add(parent)
                adj[parent].add(child)
                edges.append((child, parent))
    return adj, edges


def task_features(name, man, fk_df, tasks):
    adj, edges = schema_graph(man)
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(adj)
    g.add_edges_from(edges)
    rows = []
    fk_d = fk_df[fk_df.dataset == name]
    for tname, tm in tasks.items():
        entity = tm.get("entity_table") or tm.get("src_entity_table")
        if entity not in adj:
            continue
        dist = {entity: 0}
        q = deque([entity])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        one_hop = fk_d[fk_d.parent_table == entity]
        two_hop_parents = set(one_hop.child_table)
        two_hop = fk_d[fk_d.parent_table.isin(two_hop_parents) & (fk_d.child_table != entity)]
        joins = pd.concat([one_hop.assign(hop=1), two_hop.assign(hop=2)])
        row = {"dataset": name, "task": tname, "task_type": tm.get("task_type"), "entity_table": entity,
               "n_joins_one_hop": len(one_hop), "n_joins_two_hop": len(joins),
               "tables_within_two_hops": sum(1 for v in dist.values() if v <= 2),
               "max_join_depth": max(dist.values()), "schema_tables": len(adj), "schema_joins": len(edges),
               "schema_avg_clustering": nx.average_clustering(g), "schema_avg_degree": 2 * len(edges) / max(1, len(adj)),
               "entity_degree": len(adj[entity])}
        for d in EMBED_DIMS:
            j1, j2 = one_hop[f"JBR_d{d}"], joins[f"JBR_d{d}"]
            w = joins.H_row_conditional
            row.update({f"jbr_max_one_hop_d{d}": j1.max() if len(j1) else 0.0,
                        f"jbr_sum_one_hop_d{d}": j1.sum(),
                        f"jbr_max_two_hop_d{d}": j2.max() if len(j2) else 0.0,
                        f"jbr_sum_two_hop_d{d}": j2.sum(),
                        f"jbr_mean_two_hop_d{d}": j2.mean() if len(j2) else 0.0,
                        f"jbr_entropy_weighted_d{d}": float((j2 * w).sum() / w.sum()) if len(j2) and w.sum() > 0 else 0.0})
        row["missing_rate_two_hop"] = joins.missing_rate.mean() if len(joins) else 0.0
        if "K_mean_linked_train_period" in joins and "K_mean_linked_test_period" in joins:
            ratio = (joins.K_mean_linked_test_period / joins.K_mean_linked_train_period).replace([np.inf, -np.inf], np.nan)
            row["cardinality_shift_log2"] = float(np.nanmean(np.abs(np.log2(ratio)))) if ratio.notna().any() else np.nan
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--sample-rows", type=int, default=200_000)
    ap.add_argument("--keep-data", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi
    api = HfApi()
    found = list_datasets(api)
    log(f"datasets found: {found}")
    names = args.datasets or sorted(found)
    fk_jsonl, task_jsonl = OUT / "foreign_key_join_statistics.jsonl", OUT / "task_join_features.jsonl"
    done = {json.loads(l)["dataset"] for l in open(fk_jsonl)} if fk_jsonl.exists() else set()
    summary_path = OUT / "dataset_schema_summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    for name in names:
        if name in done:
            log(f"{name}: already processed, skipping")
            continue
        if name not in found:
            log(f"{name}: not found in the RelBench repositories")
            continue
        try:
            man, fk_rows, tasks, table_rows = process_dataset(name, found[name], api, args)
        except Exception as e:
            import traceback
            log(f"{name}: FAILED {e!r}\n{traceback.format_exc()}")
            continue
        t_rows = task_features(name, man, pd.DataFrame(fk_rows), tasks) if fk_rows else []
        with open(fk_jsonl, "a") as f:
            for r in fk_rows:
                f.write(json.dumps(r, default=float) + "\n")
        with open(OUT / "table_feature_statistics.jsonl", "a") as f:
            for r in table_rows:
                f.write(json.dumps(r, default=float) + "\n")
        with open(task_jsonl, "a") as f:
            for r in t_rows:
                f.write(json.dumps(r, default=float) + "\n")
        summary[name] = {"repository": found[name], "tables": len(man["tables"]), "foreign_keys": len(fk_rows),
                         "tasks": sorted(tasks), "val_timestamp": str(man["val_timestamp"]),
                         "test_timestamp": str(man["test_timestamp"])}
        json.dump(summary, open(summary_path, "w"), indent=2)
        log(f"{name}: {len(fk_rows)} foreign keys, {len(t_rows)} tasks written")
    for src, dst in ((fk_jsonl, "foreign_key_join_statistics.csv"), (task_jsonl, "task_join_features.csv"),
                     (OUT / "table_feature_statistics.jsonl", "table_feature_statistics.csv")):
        if src.exists():
            pd.DataFrame([json.loads(l) for l in open(src)]).to_csv(OUT / dst, index=False)
    log("RELBENCH_EXTRACTION_COMPLETE")


if __name__ == "__main__":
    main()
