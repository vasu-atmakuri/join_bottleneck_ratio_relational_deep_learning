#!/usr/bin/env python3
"""
Controlled interventions on real RelBench tasks: the same relational deep learning model is trained while only the
aggregation function and the embedding width change, so differences between runs are caused by aggregation alone.

Model (the standard RelBench RDL pipeline): per-table ResNet tabular encoder, relative-time encoding, two layers of
heterogeneous message passing on temporally sampled subgraphs (fanout [64, 32], see temporal_subgraph_sampler.py),
MLP head.
  aggregation  sum | mean   (GraphSAGE)    or attention (GATv2, one head, parent node as query)
  width d      32 | 128 | 512
  learning rate 5e-3, 2e-3, 1e-3 for d = 32, 128, 512 (wider models need smaller steps, as in the synthetic study)
  seeds        0, 1
  training     4 epochs of at most 60 mini-batches of 256 seed entities; the checkpoint with the best validation
               metric is evaluated on the test split
Text columns are left out for every configuration (no text embedding model is downloaded); this is identical across
conditions, so comparisons between aggregators and widths are unaffected.

Tasks: 12 entity tasks on rel-f1, rel-trial, rel-stack and rel-avito.
Results: results/relbench_interventions/intervention_runs.jsonl (resumable, one line per run)
Usage:   python real_data/train_relbench_aggregation_interventions.py --workers 2 --threads-per-worker 4
"""

import argparse
import copy
import sys
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "results" / "relbench_interventions"
CACHE = ROOT / "data" / "relbench_graph_cache"
RESULTS = OUT / "intervention_runs.jsonl"

TASKS = {
    "rel-f1": ["driver-dnf", "driver-top3", "driver-position"],
    "rel-trial": ["study-outcome", "study-adverse", "site-success"],
    "rel-stack": ["user-engagement", "user-badge", "post-votes"],
    "rel-avito": ["user-visits", "user-clicks", "ad-ctr"],
}
AGGREGATIONS = ["sum", "mean", "attention"]
WIDTHS = [32, 128, 512]
LEARNING_RATE_BY_WIDTH = {32: 5e-3, 128: 2e-3, 512: 1e-3}  # smaller steps for wider models
SEEDS = [0, 1]
CONFIG = {"epochs": 4, "max_steps_per_epoch": 60, "batch_size": 256, "num_neighbors": [64, 32], "num_layers": 2}


def run_key(r):
    return "|".join(str(r[k]) for k in ("dataset", "task", "aggregation", "width", "learning_rate", "seed"))


def all_jobs():
    return [{"dataset": ds, "task": t, "aggregation": a, "width": w, "learning_rate": lr, "seed": s}
            for ds, tasks in TASKS.items() for t in tasks for a in AGGREGATIONS for w in WIDTHS
            for lr in [LEARNING_RATE_BY_WIDTH[w]] for s in SEEDS]


def build_model(data, col_stats_dict, entity_table, aggregation, width, out_channels):
    import torch
    from relbench.modeling.nn import HeteroEncoder, HeteroGAT, HeteroGraphSAGE, HeteroTemporalEncoder
    from torch_geometric.nn import MLP

    class RelationalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = HeteroEncoder(width, {nt: data[nt].tf.col_names_dict for nt in data.node_types},
                                         col_stats_dict)
            self.temporal = HeteroTemporalEncoder([nt for nt in data.node_types if "time" in data[nt]], width)
            if aggregation == "attention":
                self.gnn = HeteroGAT(data.node_types, data.edge_types, width, heads=1, num_layers=CONFIG["num_layers"])
            else:
                self.gnn = HeteroGraphSAGE(data.node_types, data.edge_types, width, aggr=aggregation,
                                           num_layers=CONFIG["num_layers"])
            self.head = MLP(width, out_channels=out_channels, norm="batch_norm", num_layers=1)

        def forward(self, batch):
            seed_time = batch[entity_table].seed_time
            x = self.encoder(batch.tf_dict)
            for nt, rel in self.temporal(seed_time, batch.time_dict, batch.batch_dict).items():
                x[nt] = x[nt] + rel
            for conv, norms in zip(self.gnn.convs, self.gnn.norms):
                out = conv(x, batch.edge_index_dict)
                # node types that receive no messages in this layer keep their current embedding
                x = {nt: norms[nt](out.get(nt, x[nt])).relu() for nt in x}
            return self.head(x[entity_table][: seed_time.size(0)])

    return RelationalModel()


def prepare(dataset_name, task_name):
    """Graph and loaders for one task (graph materialization is cached on disk per dataset)."""
    import torch_frame
    from relbench import load_dataset
    from relbench.modeling.graph import get_node_train_table_input, make_pkey_fkey_graph
    from relbench.modeling.utils import get_stype_proposal

    ds = load_dataset(dataset_name)
    task = ds.load_task(task_name)
    db = ds.get_db()
    stypes = get_stype_proposal(db)
    for table, cols in stypes.items():
        for c in [c for c, s in cols.items() if s in (torch_frame.text_embedded, torch_frame.text_tokenized)]:
            del cols[c]
    data, col_stats = make_pkey_fkey_graph(db, stypes, cache_dir=str(CACHE / dataset_name),
                                           remove_columns=task.hidden_columns())
    from temporal_subgraph_sampler import TemporalNodeLoader, TemporalSubgraphSampler
    sampler = TemporalSubgraphSampler(data, CONFIG["num_neighbors"])
    loaders, tables = {}, {}
    for split in ("train", "val", "test"):
        table = task.get_table(split)
        tables[split] = table
        inp = get_node_train_table_input(table, task)
        loaders[split] = TemporalNodeLoader(sampler, task.entity_table, inp.nodes[1].numpy(), inp.time.numpy(),
                                            inp.target, CONFIG["batch_size"], shuffle=split == "train")
    return task, data, col_stats, loaders, tables


def train_one(job, task, data, col_stats, loaders, tables):
    import numpy as np
    import torch
    from relbench.base import TaskType

    torch.manual_seed(job["seed"])
    np.random.seed(job["seed"])
    classification = task.task_type == TaskType.BINARY_CLASSIFICATION
    model = build_model(data, col_stats, task.entity_table, job["aggregation"], job["width"], 1)
    opt = torch.optim.Adam(model.parameters(), lr=job["learning_rate"])
    loss_fn = torch.nn.BCEWithLogitsLoss() if classification else torch.nn.L1Loss()
    train_y = tables["train"].df[task.target_col].to_numpy(dtype=float)
    lo, hi = np.percentile(train_y, 1), np.percentile(train_y, 99)

    def predict(split):
        model.eval()
        out = []
        with torch.no_grad():
            for batch in loaders[split]:
                p = model(batch).view(-1)
                p = torch.sigmoid(p) if classification else p.clamp(lo, hi)
                out.append(p.numpy())
        model.train()
        return np.concatenate(out)

    metric = "roc_auc" if classification else "mae"
    better = (lambda a, b: a > b) if classification else (lambda a, b: a < b)
    best, best_epoch, t0 = None, -1, time.time()
    for epoch in range(CONFIG["epochs"]):
        for step, batch in enumerate(loaders["train"]):
            if step >= CONFIG["max_steps_per_epoch"]:
                break
            opt.zero_grad()
            pred = model(batch).view(-1)
            loss = loss_fn(pred, batch[task.entity_table].y.float())
            loss.backward()
            opt.step()
        val = task.evaluate(predict("val"), tables["val"])[metric]
        if best is None or better(val, best["val"]):
            best = {"val": float(val), "state": copy.deepcopy(model.state_dict())}
            best_epoch = epoch
    model.load_state_dict(best["state"])
    test = task.evaluate(predict("test"))[metric]
    return {"metric": metric, "val_metric": best["val"], "test_metric": float(test), "best_epoch": best_epoch,
            "n_parameters": sum(p.numel() for p in model.parameters()), "seconds": round(time.time() - t0, 1)}


def worker(dataset_jobs, threads, results_path, log_path):
    import torch
    torch.set_num_threads(threads)
    by_task = {}
    for j in dataset_jobs:
        by_task.setdefault((j["dataset"], j["task"]), []).append(j)
    for (ds, t), jobs in by_task.items():
        try:
            prepared = prepare(ds, t)
        except Exception as e:
            with open(log_path, "a") as f:
                f.write(f"ERROR preparing {ds}/{t}: {e!r}\n")
            continue
        for j in jobs:
            try:
                out = {**j, **train_one(j, *prepared), "status": "ok"}
            except Exception as e:
                out = {**j, "status": "error", "error": repr(e)}
            with open(results_path, "a") as f:
                f.write(json.dumps(out) + "\n")
            with open(log_path, "a") as f:
                msg = (f"{out['metric']} test {out['test_metric']:.4f} val {out['val_metric']:.4f} ({out['seconds']}s)"
                       if out["status"] == "ok" else f"ERROR {out['error']}")
                f.write(f"[{time.strftime('%H:%M:%S')}] {ds}/{t} {j['aggregation']} d{j['width']} "
                        f"lr{j['learning_rate']} s{j['seed']}: {msg}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads-per-worker", type=int, default=4)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    jobs = all_jobs()
    if args.datasets:
        jobs = [j for j in jobs if j["dataset"] in args.datasets]
    if args.list:
        print(f"{len(jobs)} runs over {len({(j['dataset'], j['task']) for j in jobs})} tasks")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        done = {run_key(json.loads(l)) for l in RESULTS.open() if '"ok"' in l}
    todo = [j for j in jobs if run_key(j) not in done]
    log = OUT / "progress.log"
    with open(log, "a") as f:
        f.write(f"{len(jobs)} runs in plan, {len(jobs) - len(todo)} finished, {len(todo)} to run\n")
    # one worker per group of datasets, so each dataset's graph is built once per worker
    groups = [[] for _ in range(args.workers)]
    for i, ds in enumerate(sorted({j["dataset"] for j in todo})):
        groups[i % args.workers] += [j for j in todo if j["dataset"] == ds]
    procs = [mp.get_context("spawn").Process(target=worker, args=(g, args.threads_per_worker, RESULTS, log))
             for g in groups if g]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    n_ok = sum(1 for l in RESULTS.open() if '"ok"' in l) if RESULTS.exists() else 0
    with open(log, "a") as f:
        f.write(f"{n_ok}/{len(jobs)} runs finished successfully\n")
        if n_ok == len(jobs):
            f.write("ALL_INTERVENTION_RUNS_COMPLETE\n")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
