#!/usr/bin/env python3
"""
Learning-rate-tuned replication of the core synthetic experiments.

The width-and-optimization control showed that the fixed learning rate of the first suite (1e-3) is too high for wide
encoders and depresses accuracy in exactly the regime the capacity law describes. This suite adds learning rates 1e-4
and 3e-4 to every core configuration; the analysis selects, for each configuration and seed, the learning rate with the
best validation AUROC among {1e-4, 3e-4, 1e-3} (the 1e-3 runs come from the first suite and from this suite's width
sweep). Encoders are at least as wide as the aggregated vector (hidden = max(128, d)).

  main_deepsets_lr        25 base scenarios x 3 targets x DeepSets {sum, mean} x 3 seeds x lr {1e-4, 3e-4}   = 900
  gnn_architecture_lr     11 scenarios x 3 targets x {SumAgg, MeanAgg, AttnAgg-h4} x 3 seeds x 2 lr          = 594
  extended_lr             63 scenarios x rare_pattern x {SumAgg, AttnAgg-h1, AttnAgg-h4} x 3 seeds x 2 lr     = 1134
  width_sweep_lr          7 (K, D) x d {4,...,512} x 2 targets x sum x 3 seeds x lr {1e-4, 3e-4, 1e-3}        = 1008

Results: results/learning_rate_tuned_core/core_training_runs.jsonl (resumable).
Usage:   python experiments/run_learning_rate_tuned_core_experiments.py --workers 3 --threads-per-worker 2
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

from relational_aggregation_bottleneck.scenarios import (  # noqa: E402
    TARGETS, architecture_scenarios, base_scenarios, extreme_scenarios,
)

RESULTS = ROOT / "results" / "learning_rate_tuned_core" / "core_training_runs.jsonl"
SEEDS = [0, 1, 2]
NEW_LRS = [1e-4, 3e-4]


def J(experiment, K, D, d, target, family, variant, seed, lr, heads=1):
    return {"experiment": experiment, "scenario": f"K{K}_D{D}_d{d}", "K": K, "D": D, "d": d, "target": target,
            "family": family, "variant": variant, "heads": heads, "seed": seed, "lr": lr,
            "model_options": {"hidden": max(128, d)} if family == "deepsets" else {}}


def key(j):
    return "|".join(str(j[k]) for k in ("experiment", "scenario", "target", "family", "variant", "heads", "seed", "lr"))


def all_jobs():
    jobs = []
    for sc in base_scenarios():
        for t in TARGETS:
            for a in ["sum", "mean"]:
                for s in SEEDS:
                    for lr in NEW_LRS:
                        jobs.append(J("main_deepsets_lr", sc.K, sc.D, sc.d, t, "deepsets", a, s, lr))
    for sc in architecture_scenarios():
        for t in TARGETS:
            for a in ["SumAgg", "MeanAgg", "AttnAgg"]:
                for s in SEEDS:
                    for lr in NEW_LRS:
                        jobs.append(J("gnn_architecture_lr", sc.K, sc.D, sc.d, t, "gnn", a, s, lr,
                                      heads=4 if a == "AttnAgg" else 1))
    for sc in base_scenarios() + extreme_scenarios():
        for a, h in [("SumAgg", 1), ("AttnAgg", 1), ("AttnAgg", 4)]:
            for s in SEEDS:
                for lr in NEW_LRS:
                    jobs.append(J("extended_lr", sc.K, sc.D, sc.d, "rare_pattern", "gnn", a, s, lr, heads=h))
    for K, D in [(4, 32), (8, 16), (8, 32), (16, 8), (16, 16), (32, 8), (64, 4)]:
        for d in [4, 8, 16, 32, 64, 128, 256, 512]:
            for t in ["rare_pattern", "distributional"]:
                for s in SEEDS:
                    for lr in NEW_LRS + [1e-3]:
                        jobs.append(J("width_sweep_lr", K, D, d, t, "deepsets", "sum", s, lr))
    return jobs


def _init(threads):
    import torch
    torch.set_num_threads(threads)


def _run(j):
    from relational_aggregation_bottleneck.training import train_and_evaluate
    t0 = time.time()
    try:
        out = train_and_evaluate(j["K"], j["D"], j["d"], j["target"], j["family"], j["variant"], j["seed"],
                                 heads=j["heads"], config={"learning_rate": j["lr"]},
                                 model_options=j["model_options"] or None)
        return {**j, **out, "status": "ok", "wall_seconds": round(time.time() - t0, 2)}
    except Exception as e:
        return {**j, "status": "error", "error": repr(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--experiments", nargs="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    jobs = all_jobs()
    if args.experiments:
        jobs = [j for j in jobs if j["experiment"] in args.experiments]
    if args.list:
        from collections import Counter
        for k, v in Counter(j["experiment"] for j in jobs).items():
            print(f"{k:24s} {v}")
        print(f"{'total':24s} {len(jobs)}")
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
    order = ["main_deepsets_lr", "gnn_architecture_lr", "width_sweep_lr", "extended_lr"]
    todo.sort(key=lambda j: (order.index(j["experiment"]), j["K"] * j["D"] * max(1, j["d"] // 128)))
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
            st = f"AUROC {r['test_auroc']:.3f}" if r["status"] == "ok" else f"ERROR {r['error']}"
            print(f"[{n}/{len(todo)}] {r['experiment']} {r['scenario']} {r['target']} {r['variant']} h{r['heads']} "
                  f"lr{r['lr']} s{r['seed']}: {st} | {el / 3600:.2f} h elapsed, about "
                  f"{el / n * (len(todo) - n) / 3600:.1f} h left", flush=True)
    print(f"Finished {n} runs with {errors} errors.", flush=True)
    if errors == 0:
        print("ALL_TUNED_CORE_RUNS_COMPLETE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
