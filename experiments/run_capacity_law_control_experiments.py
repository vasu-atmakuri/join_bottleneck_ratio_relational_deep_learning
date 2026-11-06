#!/usr/bin/env python3
"""
Control experiments for the aggregation capacity law. One JSON line per run is appended to
results/capacity_law_controls/control_training_runs.jsonl; finished runs are skipped on restart.

  embedding_dimension_sweep  7 (K, D) pairs x d in {4,...,512} x 2 targets x DeepSets sum x 3 seeds       = 336
  training_budget            10 scenarios (JBR 0.25 to 8) x budget {0.5, 1, 2, 4} x rare_pattern x sum x 3 = 120
  entropy_vs_dimension       6 (K, D) pairs x bit probability {0.5, 0.25, 0.11, 0.05} x sum x 3          = 72
  irrelevant_features        relevant D {4, 8} x K {4, 16, 64} x irrelevant {0, 8, 24} x {sum, attention} x 3 = 108
  variable_cardinality       {fixed, poisson, lognormal} x K {4, 16, 64} x {sum, attention} x 3           = 54
  missing_values             missing {0, 0.1, 0.3} x 6 (K, D) pairs x sum x 3                              = 54
  distribution_shift         test shift {none, 2x cardinality, bit probability 0.3} x 6 pairs x {sum, attention} x 3 = 108
  width_and_optimization     4 (K, D) pairs x d {16,...,512} with encoder width max(128, d) x learning rate
                             {3e-4, 1e-3, 3e-3} x layer norm {off, on} x DeepSets sum x 3                  = 360

Usage: python experiments/run_capacity_law_control_experiments.py --workers 3 --threads-per-worker 2 [--list]
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results" / "capacity_law_controls" / "control_training_runs.jsonl"
SEEDS = [0, 1, 2]
SIX_PAIRS = [(4, 16), (8, 16), (16, 8), (32, 8), (8, 32), (64, 4)]


def J(experiment, K, D, d, target, family, variant, seed, heads=1, opts=None, test_opts=None, budget=1.0):
    return {"experiment": experiment, "K": K, "D": D, "d": d, "target": target, "family": family, "variant": variant,
            "seed": seed, "heads": heads, "opts": opts or {}, "test_opts": test_opts, "budget": budget}


def key(j):
    base = [j[k] for k in ("experiment", "K", "D", "d", "target", "family", "variant", "seed", "heads",
                           "opts", "test_opts", "budget")]
    if j.get("model_options") or j.get("lr"):
        base += [j.get("model_options"), j.get("lr")]
    return json.dumps(base, sort_keys=True)


def all_jobs():
    jobs = []
    for K, D in [(4, 32), (8, 16), (8, 32), (16, 8), (16, 16), (32, 8), (64, 4)]:
        for d in [4, 8, 16, 32, 64, 128, 256, 512]:
            for t in ["rare_pattern", "distributional"]:
                for s in SEEDS:
                    jobs.append(J("embedding_dimension_sweep", K, D, d, t, "deepsets", "sum", s))
    for K, D in [(2, 16), (4, 16), (8, 8), (4, 32), (16, 8), (8, 32), (32, 8), (16, 32), (32, 16), (64, 16)]:
        for b in [0.5, 1.0, 2.0, 4.0]:
            for s in SEEDS:
                jobs.append(J("training_budget", K, D, 128, "rare_pattern", "deepsets", "sum", s, budget=b))
    for K, D in [(4, 16), (8, 16), (16, 16), (8, 32), (32, 16), (64, 16)]:
        for p in [0.5, 0.25, 0.11, 0.05]:
            for s in SEEDS:
                jobs.append(J("entropy_vs_dimension", K, D, 128, "rare_pattern", "deepsets", "sum", s,
                              opts={"bit_prob": p}))
    for D in [4, 8]:
        for K in [4, 16, 64]:
            for irr in [0, 8, 24]:
                for fam, var in [("deepsets", "sum"), ("gnn", "AttnAgg")]:
                    for s in SEEDS:
                        jobs.append(J("irrelevant_features", K, D, 128, "rare_pattern", fam, var, s,
                                      opts={"n_irrelevant": irr}))
    for dist in ["fixed", "poisson", "lognormal"]:
        for K in [4, 16, 64]:
            for fam, var in [("deepsets", "sum"), ("gnn", "AttnAgg")]:
                for s in SEEDS:
                    jobs.append(J("variable_cardinality", K, 8, 128, "rare_pattern", fam, var, s,
                                  opts={"k_distribution": dist}))
    for miss in [0.0, 0.1, 0.3]:
        for K, D in SIX_PAIRS:
            for s in SEEDS:
                jobs.append(J("missing_values", K, D, 128, "rare_pattern", "deepsets", "sum", s,
                              opts={"missing_rate": miss}))
    for name, topts in [("none", None), ("cardinality_x2", {"k_scale": 2.0}), ("bit_prob_0.3", {"bit_prob": 0.3})]:
        for K, D in SIX_PAIRS:
            for fam, var in [("deepsets", "sum"), ("gnn", "AttnAgg")]:
                for s in SEEDS:
                    j = J("distribution_shift", K, D, 128, "rare_pattern", fam, var, s, test_opts=topts)
                    j["shift"] = name
                    jobs.append(j)
    # encoder width matched to d (hidden = max(128, d)), learning rate and layer normalization
    for K, D in [(4, 32), (8, 16), (16, 8), (8, 32)]:
        for d in [16, 64, 128, 256, 512]:
            for lr in [3e-4, 1e-3, 3e-3]:
                for ln in [False, True]:
                    for s in SEEDS:
                        j = J("width_and_optimization", K, D, d, "rare_pattern", "deepsets", "sum", s)
                        j["model_options"] = {"hidden": max(128, d), "layer_norm": ln}
                        j["lr"] = lr
                        jobs.append(j)
    return jobs


def _init(threads):
    import torch
    torch.set_num_threads(threads)


def _run(j):
    from relational_aggregation_bottleneck.training import TRAIN_CONFIG, train_and_evaluate
    from relational_aggregation_bottleneck.variant_training import train_and_evaluate_variant
    t0 = time.time()
    try:
        cfg = None
        if j["budget"] != 1.0:
            cfg = {"max_steps": int(TRAIN_CONFIG["max_steps"] * j["budget"]),
                   "min_steps": max(1000, int(TRAIN_CONFIG["min_steps"] * j["budget"]))}
        if j.get("lr"):
            cfg = dict(cfg or {}, learning_rate=j["lr"])
        if j["experiment"] in ("embedding_dimension_sweep", "training_budget", "width_and_optimization"):
            out = train_and_evaluate(j["K"], j["D"], j["d"], j["target"], j["family"], j["variant"], j["seed"],
                                     heads=j["heads"], config=cfg, model_options=j.get("model_options"))
        else:
            out = train_and_evaluate_variant(j["K"], j["D"], j["d"], j["target"], j["family"], j["variant"],
                                             j["seed"], heads=j["heads"], opts=j["opts"], test_opts=j["test_opts"],
                                             config=cfg)
        return {**j, **out, "status": "ok", "wall_seconds": round(time.time() - t0, 2)}
    except Exception as e:
        return {**j, "status": "error", "error": repr(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--experiments", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    jobs = all_jobs()
    if args.experiments:
        jobs = [j for j in jobs if j["experiment"] in args.experiments]
    if args.list:
        from collections import Counter
        for k, v in Counter(j["experiment"] for j in jobs).items():
            print(f"{k:28s} {v}")
        print(f"{'total':28s} {len(jobs)}")
        return
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        for line in RESULTS.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                done.add(key(r))
    todo = [j for j in jobs if key(j) not in done]
    todo.sort(key=lambda j: (j["budget"], j["K"] * j["D"]))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(jobs)} runs in plan, {len(jobs) - len(todo)} finished, {len(todo)} to run", flush=True)
    t0, n, errors = time.time(), 0, 0
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.threads_per_worker,),
                                      maxtasksperchild=25) as pool, RESULTS.open("a") as f:
        for r in pool.imap_unordered(_run, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            errors += r["status"] != "ok"
            el = time.time() - t0
            status = f"AUROC {r['test_auroc']:.3f}" if r["status"] == "ok" else f"ERROR {r['error']}"
            print(f"[{n}/{len(todo)}] {r['experiment']} K{r['K']} D{r['D']} d{r['d']} {r['variant']} "
                  f"{json.dumps(r['opts'])} s{r['seed']}: {status} | {el / 3600:.2f} h elapsed", flush=True)
    print(f"Finished {n} runs with {errors} errors.", flush=True)
    if errors == 0 and not args.limit:
        print("ALL_CONTROL_RUNS_COMPLETE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
