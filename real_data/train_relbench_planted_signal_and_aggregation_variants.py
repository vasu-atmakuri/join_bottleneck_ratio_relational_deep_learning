#!/usr/bin/env python3
"""
Two experiment families on real RelBench databases, using the same relational deep learning pipeline as
train_relbench_aggregation_interventions.py (per-table ResNet tabular encoders, relative-time encoding, heterogeneous
message passing on temporally sampled subgraphs, MLP head).

Aggregations
  sum, mean        GraphSAGE layer (neighbor aggregation plus a root weight on the node's own embedding)
  attention_root   GATv2 neighbor attention (one head, the receiving node is the query) plus a root weight on the
                   node's own embedding; without the root term the receiving node's own features are lost after the
                   first layer, which confounds a comparison with GraphSAGE
  sum_attention    GraphSAGE sum layer plus GATv2 attention messages (keeps moments and adds selection)

Family 1, real_tasks_attention_variants
  The 12 entity tasks of the intervention study with attention_root and sum_attention, widths 32 and 128, seeds 0, 1,
  two layers. Compared against the sum and mean runs of the intervention study at the same widths and seeds.

Family 2, planted_signal
  Real tables, real cardinalities and real child features, with a label that is known to need one child row. For a
  join child -> entity, one categorical child column c is chosen (the one with the most distinct values; values are
  hashed into at most 1000 buckets) and every entity gets a planted categorical column `planted_query`, a value of c.
  Seeds (entity, time) come from a RelBench entity task on that entity table; only seeds with between 1 and 64
  admissible children over the join are kept, so the first-hop sampler (fanout 64) always sees all of them.
    parent_query  y = 1 if some admissible child has c == planted_query of the entity (needs the parent's query)
    fixed_query   y = 1 if some admissible child has c in V*, a fixed set of the most common values chosen for class
                  balance (a count over fixed categories, the same for every entity)
  Both labels use the same graph, so inputs are identical. Factors: aggregation {sum, mean, attention_root,
  sum_attention} x layers {1, 2} at d = 128, plus d = 32 for sum and attention_root with one layer; seeds 0, 1.
  With two layers each child first receives a message from its parent, so pooling can become query-aware; with one
  layer the parent must recover the answer from the pooled child set. Test predictions are saved with the number of
  admissible children of every seed, so accuracy can be analyzed as a function of K.

Results: results/relbench_planted_signal/runs.jsonl, predictions/*.npz, progress.log (resumable)
Usage:   python real_data/train_relbench_planted_signal_and_aggregation_variants.py --workers 2 --threads-per-worker 4
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "results" / "relbench_planted_signal"
CACHE = ROOT / "data" / "relbench_graph_cache"
RESULTS = OUT / "runs.jsonl"

REAL_TASKS = {
    "rel-f1": ["driver-dnf", "driver-top3", "driver-position"],
    "rel-trial": ["study-outcome", "study-adverse", "site-success"],
    "rel-stack": ["user-engagement", "user-badge", "post-votes"],
    "rel-avito": ["user-visits", "user-clicks", "ad-ctr"],
}
# (dataset, RelBench task that supplies seed entities and timestamps, child table, preferred foreign key column,
#  training seeds). The rel-trial join failed during label construction and is not run.
PLANTED_JOINS = [
    ("rel-f1", "driver-dnf", "results", "driverId", [0, 1]),
    ("rel-stack", "user-engagement", "badges", "UserId", [0, 1, 2]),
    ("rel-avito", "user-visits", "SearchInfo", "UserID", [0, 1, 2]),
    ("rel-avito", "user-visits", "VisitStream", "UserID", [0, 1]),
    ("rel-stack", "user-engagement", "posts", "OwnerUserId", [0, 1, 2]),
    ("rel-stack", "user-engagement", "comments", "UserId", [0, 1, 2]),
]
# Longer-training check on original tasks: three times the step budget, three seeds, d = 128.
LONGER_TRAINING_TASKS = {"rel-f1": ["driver-dnf", "driver-top3"], "rel-stack": ["user-engagement", "user-badge"],
                         "rel-avito": ["user-visits"], "rel-trial": ["study-outcome"]}
LONGER_TRAINING_EPOCHS = 12
LEARNING_RATE_BY_WIDTH = {32: 5e-3, 128: 2e-3, 512: 1e-3}
CONFIG = {"epochs": 4, "max_steps_per_epoch": 60, "batch_size": 256}
FANOUTS = {1: [64], 2: [64, 32]}
MAX_SEEDS = {"train": 20000, "val": 5000, "test": 5000}
MAX_CHILDREN = 64
MAX_BUCKETS = 1000


def run_key(r):
    return "|".join(str(r[k]) for k in ("family", "dataset", "task", "label", "aggregation", "layers", "width", "seed"))


def all_jobs():
    jobs = []
    for ds, tasks in REAL_TASKS.items():
        for t in tasks:
            for a in ["attention_root", "sum_attention"]:
                for w in [32, 128]:
                    for s in [0, 1]:
                        jobs.append({"family": "real_tasks_attention_variants", "dataset": ds, "task": t,
                                     "label": "original", "aggregation": a, "layers": 2, "width": w, "seed": s})
    for ds, tasks in LONGER_TRAINING_TASKS.items():
        for t in tasks:
            for a in ["sum", "mean", "attention_root", "sum_attention"]:
                for s in [0, 1, 2]:
                    jobs.append({"family": "real_tasks_longer_training", "dataset": ds, "task": t, "label": "original",
                                 "aggregation": a, "layers": 2, "width": 128, "seed": s,
                                 "epochs": LONGER_TRAINING_EPOCHS})
    for ds, t, child, _, seeds in PLANTED_JOINS:
        for label in ["parent_query", "fixed_query"]:
            for a in ["sum", "mean", "attention_root", "sum_attention"]:
                for layers in [1, 2]:
                    for s in seeds:
                        jobs.append({"family": "planted_signal", "dataset": ds, "task": f"{t}|{child}", "label": label,
                                     "aggregation": a, "layers": layers, "width": 128, "seed": s})
            for a in ["sum", "attention_root"]:
                for s in seeds:
                    jobs.append({"family": "planted_signal", "dataset": ds, "task": f"{t}|{child}", "label": label,
                                 "aggregation": a, "layers": 1, "width": 32, "seed": s})
    for j in jobs:
        j["learning_rate"] = LEARNING_RATE_BY_WIDTH[j["width"]]
    return jobs


# ----------------------------------------------------------------------------------------------------------- model
def build_model(data, col_stats_dict, entity_table, aggregation, width, layers):
    import torch
    from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder
    from torch_geometric.nn import MLP, GATv2Conv, HeteroConv, LayerNorm, SAGEConv

    key = {nt: f"nt_{i}" for i, nt in enumerate(data.node_types)}  # safe ModuleDict keys

    class RelationalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = HeteroEncoder(width, {nt: data[nt].tf.col_names_dict for nt in data.node_types},
                                         col_stats_dict)
            self.temporal = HeteroTemporalEncoder([nt for nt in data.node_types if "time" in data[nt]], width)
            self.sage, self.gat, self.root, self.norms = (torch.nn.ModuleList() for _ in range(4))
            for _ in range(layers):
                if aggregation in ("sum", "mean", "sum_attention"):
                    agg = "mean" if aggregation == "mean" else "sum"
                    self.sage.append(HeteroConv({et: SAGEConv((width, width), width, aggr=agg)
                                                 for et in data.edge_types}, aggr="sum"))
                if aggregation in ("attention_root", "sum_attention"):
                    self.gat.append(HeteroConv({et: GATv2Conv((width, width), width, heads=1, concat=False,
                                                              add_self_loops=False) for et in data.edge_types},
                                               aggr="sum"))
                if aggregation == "attention_root":
                    self.root.append(torch.nn.ModuleDict({key[nt]: torch.nn.Linear(width, width)
                                                          for nt in data.node_types}))
                self.norms.append(torch.nn.ModuleDict({key[nt]: LayerNorm(width, mode="node")
                                                       for nt in data.node_types}))
            self.head = MLP(width, out_channels=1, norm="batch_norm", num_layers=1)

        def forward(self, batch):
            seed_time = batch[entity_table].seed_time
            x = self.encoder(batch.tf_dict)
            for nt, rel in self.temporal(seed_time, batch.time_dict, batch.batch_dict).items():
                x[nt] = x[nt] + rel
            for i in range(layers):
                out = {}
                parts = []
                if len(self.sage):
                    parts.append(self.sage[i](x, batch.edge_index_dict))
                if len(self.gat):
                    parts.append(self.gat[i](x, batch.edge_index_dict))
                for p in parts:
                    for nt, v in p.items():
                        out[nt] = out[nt] + v if nt in out else v
                if len(self.root):
                    for nt in x:
                        r = self.root[i][key[nt]](x[nt])
                        out[nt] = out[nt] + r if nt in out else r
                # node types that receive no messages in this layer keep their current embedding
                x = {nt: self.norms[i][key[nt]](out.get(nt, x[nt])).relu() for nt in x}
            return self.head(x[entity_table][: seed_time.size(0)])

    return RelationalModel()


# ------------------------------------------------------------------------------------------------------ data: real
def prepare_real(dataset_name, task_name, layers):
    import torch_frame
    from relbench import load_dataset
    from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from temporal_subgraph_sampler import TemporalNodeLoader, TemporalSubgraphSampler

    ds = load_dataset(dataset_name)
    task = ds.load_task(task_name)
    db = ds.get_db()
    stypes = get_stype_proposal(db)
    for table, cols in stypes.items():
        for c in [c for c, s in cols.items() if s in (torch_frame.text_embedded, torch_frame.text_tokenized)]:
            del cols[c]
    data, col_stats = make_pkey_fkey_graph(db, stypes, cache_dir=str(CACHE / dataset_name),
                                           remove_columns=task.hidden_columns())
    sampler = TemporalSubgraphSampler(data, FANOUTS[layers])
    loaders, tables = {}, {}
    for split in ("train", "val", "test"):
        table = task.get_table(split)
        tables[split] = table
        inp = get_node_train_table_input(table, task)
        loaders[split] = TemporalNodeLoader(sampler, task.entity_table, inp.nodes[1].numpy(), inp.time.numpy(),
                                            inp.target, CONFIG["batch_size"], shuffle=split == "train")
    from relbench.base import TaskType
    classification = task.task_type == TaskType.BINARY_CLASSIFICATION
    metric = "roc_auc" if classification else "nmae"
    train_y = tables["train"].df[task.target_col].to_numpy(dtype=float)

    def evaluate(split, pred):
        return float(task.evaluate(pred, tables[split])[metric] if split == "val" else task.evaluate(pred)[metric])

    return {"entity": task.entity_table, "data": data, "col_stats": col_stats, "loaders": loaders,
            "classification": classification, "metric": metric, "train_y": train_y, "evaluate": evaluate,
            "test_children": None}


# --------------------------------------------------------------------------------------------------- data: planted
def _unix(series):
    from relbench.modeling.utils import to_unix_time
    return to_unix_time(series)


def build_planted_database(dataset_name, seed_task, child_table, preferred_fk, log):
    """Returns db with planted columns, seed tables per split, label arrays and metadata (deterministic)."""
    import numpy as np
    import pandas as pd
    from relbench import load_dataset

    rng = np.random.default_rng(20261109)
    ds = load_dataset(dataset_name)
    task = ds.load_task(seed_task)
    db = ds.get_db()
    entity = task.entity_table
    child = db.table_dict[child_table]
    fks = [c for c, t in child.fkey_col_to_pkey_table.items() if t == entity]
    fk = preferred_fk if preferred_fk in fks else fks[0]
    # choose the child column with the most distinct values among non-key, non-time columns
    excluded = set(child.fkey_col_to_pkey_table) | {child.pkey_col, child.time_col}
    sample = child.df.sample(min(len(child.df), 500000), random_state=0)
    card = {c: int(sample[c].nunique()) for c in child.df.columns if c not in excluded}
    card = {c: n for c, n in card.items() if n >= 8}
    if not card:
        raise RuntimeError(f"no categorical child column with at least 8 values in {child_table}")
    # prefer discrete columns (strings, categories, integers); continuous columns only if nothing else qualifies
    discrete = {c: n for c, n in card.items() if not pd.api.types.is_float_dtype(child.df[c])}
    col = max(discrete or card, key=(discrete or card).get)
    raw = child.df[col].astype(str)
    if raw.nunique() > MAX_BUCKETS:
        codes = pd.util.hash_pandas_object(raw, index=False).values % MAX_BUCKETS
        values = pd.Series(["b%d" % v for v in codes], index=child.df.index)
    else:
        values = raw
    child.df[col] = values.values
    log(f"{dataset_name}: join {child_table}.{fk} -> {entity}, planted column {col} "
        f"({values.nunique()} values after bucketing)")

    # seeds from the RelBench task, capped per split
    seeds = {}
    for split in ("train", "val", "test"):
        t = task.get_table(split, mask_input_cols=False) if split == "test" else task.get_table(split)
        df = t.df[[task.entity_col, task.time_col]].rename(columns={task.entity_col: "entity", task.time_col: "time"})
        df = df.sample(min(len(df), 4 * MAX_SEEDS[split]), random_state=1).reset_index(drop=True)
        df["t"] = _unix(df.time)
        seeds[split] = df
    entities = np.unique(np.concatenate([s.entity.values for s in seeds.values()]))
    ch = pd.DataFrame({"entity": child.df[fk].values, "value": values.values,
                       "t": _unix(child.df[child.time_col]) if child.time_col else np.zeros(len(child.df), np.int64)})
    ch = ch[ch.entity.isin(entities)].dropna(subset=["entity"])
    ch["entity"] = ch.entity.astype(np.int64)

    # planted query per entity: half a value the entity has (at some time), half a value it never has
    global_vals = values.value_counts()
    gv, gp = global_vals.index.values, (global_vals / global_vals.sum()).values
    own = ch.groupby("entity").value.agg(lambda s: s.unique())
    n_parent = len(db.table_dict[entity].df)
    query = rng.choice(gv, size=n_parent, p=gp).astype(object)
    for e, vals in own.items():
        if rng.random() < 0.5:
            query[e] = vals[rng.integers(len(vals))]
        else:
            for _ in range(20):
                q = gv[rng.choice(len(gv), p=gp)]
                if q not in set(vals):
                    query[e] = q
                    break
    db.table_dict[entity].df["planted_query"] = query

    # labels per seed from admissible children
    def label_table(seed_df, fixed_value):
        m = seed_df.reset_index().rename(columns={"index": "row"}).merge(ch, on="entity", suffixes=("", "_c"))
        m = m[m.t_c <= m.t]
        m["match_parent"] = m.value.values == query[m.entity.values]
        m["match_fixed"] = np.isin(m.value.values, list(fixed_value))
        g = m.groupby("row").agg(n=("value", "size"), parent=("match_parent", "any"), fixed=("match_fixed", "any"))
        out = seed_df.join(g, how="inner")
        return out[(out.n >= 1) & (out.n <= MAX_CHILDREN)]

    # fixed set: the most common values, added in order of frequency until the train positive rate is closest to 0.5
    # (the label is then a count over a fixed set of categories, the same for every entity)
    probe = seeds["train"].head(5000)
    best, best_gap = [gv[0]], 1.0
    for k in range(1, min(len(gv), 200) + 1):
        lt = label_table(probe, gv[:k])
        if len(lt):
            gap = abs(lt.fixed.mean() - 0.5)
            if gap < best_gap:
                best, best_gap = list(gv[:k]), gap
            if lt.fixed.mean() > 0.5:
                break
    tables = {}
    for split, df in seeds.items():
        lt = label_table(df, best).head(MAX_SEEDS[split])
        tables[split] = lt.reset_index(drop=True)
        log(f"  {split}: {len(lt)} seeds, children per seed median {lt.n.median():.0f} max {lt.n.max()}, "
            f"positive rate parent_query {lt.parent.mean():.3f}, fixed_query {lt.fixed.mean():.3f}")
    meta = {"entity": entity, "fk": fk, "column": col, "n_values": int(values.nunique()), "fixed_values": len(best),
            "test_seeds": int(len(tables["test"])),
            "train_positive_rate": float(tables["train"].parent.mean()),
            "test_positive_rate": float(tables["test"].parent.mean())}
    return db, tables, meta


def prepare_planted(dataset_name, task_spec, label, layers, log, _cache={}):
    import numpy as np
    import torch
    import torch_frame
    from relbench.modeling.graph import make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal
    from sklearn.metrics import roc_auc_score
    from temporal_subgraph_sampler import TemporalNodeLoader, TemporalSubgraphSampler

    seed_task, child_table = task_spec.split("|")
    if (dataset_name, task_spec) not in _cache:
        pref = {(d, t, c): fk for d, t, c, fk, _ in PLANTED_JOINS}[(dataset_name, seed_task, child_table)]
        db, tables, meta = build_planted_database(dataset_name, seed_task, child_table, pref, log)
        stypes = get_stype_proposal(db)
        for table, cols in stypes.items():
            for c in [c for c, s in cols.items() if s in (torch_frame.text_embedded, torch_frame.text_tokenized)]:
                del cols[c]
        stypes[child_table][meta["column"]] = torch_frame.categorical
        stypes[meta["entity"]]["planted_query"] = torch_frame.categorical
        data, col_stats = make_pkey_fkey_graph(db, stypes, cache_dir=str(CACHE / "planted_signal" / f"{dataset_name}__{child_table}"))
        _cache[(dataset_name, task_spec)] = (data, col_stats, tables, meta)
    data, col_stats, tables, meta = _cache[(dataset_name, task_spec)]
    sampler = TemporalSubgraphSampler(data, FANOUTS[layers])
    loaders = {}
    col = "parent" if label == "parent_query" else "fixed"
    for split, t in tables.items():
        y = torch.from_numpy(t[col].to_numpy(dtype=np.float32))
        loaders[split] = TemporalNodeLoader(sampler, meta["entity"], t.entity.to_numpy(np.int64), t.t.to_numpy(np.int64),
                                            y, CONFIG["batch_size"], shuffle=split == "train")

    def evaluate(split, pred):
        return float(roc_auc_score(tables[split][col].to_numpy(), pred))

    return {"entity": meta["entity"], "data": data, "col_stats": col_stats, "loaders": loaders,
            "classification": True, "metric": "roc_auc", "train_y": tables["train"][col].to_numpy(dtype=float),
            "evaluate": evaluate, "test_children": tables["test"].n.to_numpy(), "meta": meta,
            "test_labels": tables["test"][col].to_numpy()}


# ----------------------------------------------------------------------------------------------------------- train
def train_one(job, prep):
    import numpy as np
    import torch

    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    model = build_model(prep["data"], prep["col_stats"], prep["entity"], job["aggregation"], job["width"],
                        job["layers"])
    opt = torch.optim.Adam(model.parameters(), lr=job["learning_rate"])
    cls = prep["classification"]
    loss_fn = torch.nn.BCEWithLogitsLoss() if cls else torch.nn.L1Loss()
    lo, hi = np.percentile(prep["train_y"], 1), np.percentile(prep["train_y"], 99)

    def predict(split):
        model.eval()
        out = []
        with torch.no_grad():
            for batch in prep["loaders"][split]:
                p = model(batch).view(-1)
                out.append((torch.sigmoid(p) if cls else p.clamp(lo, hi)).numpy())
        model.train()
        return np.concatenate(out)

    better = (lambda a, b: a > b) if cls else (lambda a, b: a < b)
    best, best_epoch, t0 = None, -1, time.time()
    for epoch in range(job.get("epochs", CONFIG["epochs"])):
        for step, batch in enumerate(prep["loaders"]["train"]):
            if step >= CONFIG["max_steps_per_epoch"]:
                break
            opt.zero_grad()
            loss = loss_fn(model(batch).view(-1), batch[prep["entity"]].y.float())
            loss.backward()
            opt.step()
        val = prep["evaluate"]("val", predict("val"))
        if best is None or better(val, best["val"]):
            best = {"val": val, "state": copy.deepcopy(model.state_dict())}
            best_epoch = epoch
    model.load_state_dict(best["state"])
    test_pred = predict("test")
    test = prep["evaluate"]("test", test_pred)
    out = {"metric": prep["metric"], "val_metric": best["val"], "test_metric": test, "best_epoch": best_epoch,
           "n_parameters": sum(p.numel() for p in model.parameters()), "seconds": round(time.time() - t0, 1)}
    if prep.get("test_children") is not None:
        (OUT / "predictions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / "predictions" / (run_key(job).replace("|", "__") + ".npz"), pred=test_pred,
                            label=prep["test_labels"], children=prep["test_children"])
        out.update({f"planted_{k}": v for k, v in prep["meta"].items()})
    return out


def worker(jobs, threads, log_path):
    import torch
    torch.set_num_threads(threads)

    def log(msg):
        with open(log_path, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    groups = {}
    for j in jobs:
        groups.setdefault((j["family"], j["dataset"], j["task"], j["label"], j["layers"]), []).append(j)
    for (fam, ds, task, label, layers), js in groups.items():
        try:
            prep = prepare_real(ds, task, layers) if fam.startswith("real_tasks") else \
                prepare_planted(ds, task, label, layers, log)
        except Exception as e:
            import traceback
            log(f"ERROR preparing {fam} {ds}/{task} {label} L{layers}: {e!r}\n{traceback.format_exc()}")
            continue
        for j in js:
            try:
                out = {**j, **train_one(j, prep), "status": "ok"}
            except Exception as e:
                out = {**j, "status": "error", "error": repr(e)}
            with open(RESULTS, "a") as f:
                f.write(json.dumps(out, default=str) + "\n")
            msg = (f"{out['metric']} test {out['test_metric']:.4f} val {out['val_metric']:.4f} ({out['seconds']}s)"
                   if out["status"] == "ok" else f"ERROR {out['error']}")
            log(f"{fam} {ds}/{task} {label} {j['aggregation']} L{layers} d{j['width']} s{j['seed']}: {msg}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads-per-worker", type=int, default=4)
    ap.add_argument("--families", nargs="*")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    jobs = all_jobs()
    if args.families:
        jobs = [j for j in jobs if j["family"] in args.families]
    if args.datasets:
        jobs = [j for j in jobs if j["dataset"] in args.datasets]
    if args.list:
        from collections import Counter
        for k, v in Counter(j["family"] for j in jobs).items():
            print(f"{k:32s} {v}")
        print(f"{'total':32s} {len(jobs)}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        done = {run_key(json.loads(line)) for line in RESULTS.open() if '"ok"' in line}
    todo = [j for j in jobs if run_key(j) not in done]
    log = OUT / "progress.log"
    with open(log, "a") as f:
        f.write(f"{len(jobs)} runs in plan, {len(jobs) - len(todo)} finished, {len(todo)} to run\n")
    # planted-signal runs first (the new evidence), then real-task variants; one dataset per worker at a time
    order = {"planted_signal": 0, "real_tasks_attention_variants": 1, "real_tasks_longer_training": 2}
    todo.sort(key=lambda j: (order[j["family"]], j["dataset"], j["task"], j["label"], j["layers"]))
    datasets = sorted({j["dataset"] for j in todo})
    groups = [[] for _ in range(args.workers)]
    for i, ds in enumerate(datasets):
        groups[i % args.workers] += [j for j in todo if j["dataset"] == ds]
    procs = [mp.get_context("spawn").Process(target=worker, args=(g, args.threads_per_worker, log))
             for g in groups if g]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    n_ok = sum(1 for line in RESULTS.open() if '"ok"' in line) if RESULTS.exists() else 0
    with open(log, "a") as f:
        f.write(f"{n_ok}/{len(jobs)} runs finished successfully\n")
        if n_ok >= len(jobs):
            f.write("ALL_PLANTED_SIGNAL_AND_VARIANT_RUNS_COMPLETE\n")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
