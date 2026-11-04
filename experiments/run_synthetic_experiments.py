#!/usr/bin/env python3
"""
Runs every synthetic Join Bottleneck Ratio experiment and appends one JSON line per finished training run to
results/synthetic_runs/synthetic_training_runs.jsonl. Finished runs are skipped on restart, so the script can be
stopped and resumed at any time.

Experiments
  main_deepsets         25 base scenarios x 3 targets x DeepSets {sum, mean, max} x 3 seeds        = 675 runs
  gnn_architecture      11 scenarios x 3 targets x {SumAgg, MeanAgg, AttnAgg (4 heads)} x 3 seeds = 297 runs
  attention_head_sweep  5 highest-JBR base scenarios x rare_pattern x {1, 4, 8, 16} heads x 3 seeds = 60 runs
  extended_attention    63 scenarios (25 base + 38 extreme, JBR up to 4096) x rare_pattern
                        x AttnAgg {1, 4, 8, 16} heads x 3 seeds                                  = 756 runs
  extended_sum          the same 63 scenarios x rare_pattern x SumAgg x 3 seeds                   = 189 runs
  denormalized          25 base scenarios x 3 targets x denormalized MLP x 3 seeds                = 225 runs

Usage
  python experiments/run_synthetic_experiments.py --workers 3 --threads-per-worker 2
  python experiments/run_synthetic_experiments.py --pilot          # 12 quick runs to check the setup
  python experiments/run_synthetic_experiments.py --list           # print run counts and exit
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
    DEEPSETS_AGGREGATORS, GNN_ARCHITECTURES, HEAD_COUNTS, SEEDS, TARGETS, architecture_scenarios,
    base_scenarios, extreme_scenarios, head_sweep_scenarios,
)

RESULTS_PATH = ROOT / "results" / "synthetic_runs" / "synthetic_training_runs.jsonl"


def job(experiment, sc, target, family, variant, seed, heads=1):
    return {"experiment": experiment, "scenario": sc.name, "K": sc.K, "D": sc.D, "d": sc.d, "target": target,
            "family": family, "variant": variant, "heads": heads, "seed": seed}


def job_key(j):
    return "|".join(str(j[k]) for k in ("experiment", "scenario", "target", "family", "variant", "heads", "seed"))


def all_jobs():
    jobs = []
    for sc in base_scenarios():
        for t in TARGETS:
            for a in DEEPSETS_AGGREGATORS:
                for s in SEEDS:
                    jobs.append(job("main_deepsets", sc, t, "deepsets", a, s))
    for sc in architecture_scenarios():
        for t in TARGETS:
            for a in GNN_ARCHITECTURES:
                for s in SEEDS:
                    jobs.append(job("gnn_architecture", sc, t, "gnn", a, s, heads=4 if a == "AttnAgg" else 1))
    for sc in head_sweep_scenarios():
        for h in HEAD_COUNTS:
            for s in SEEDS:
                jobs.append(job("attention_head_sweep", sc, "rare_pattern", "gnn", "AttnAgg", s, heads=h))
    for sc in base_scenarios() + extreme_scenarios():
        for s in SEEDS:
            jobs.append(job("extended_sum", sc, "rare_pattern", "gnn", "SumAgg", s))
            for h in HEAD_COUNTS:
                jobs.append(job("extended_attention", sc, "rare_pattern", "gnn", "AttnAgg", s, heads=h))
    for sc in base_scenarios():
        for t in TARGETS:
            for s in SEEDS:
                jobs.append(job("denormalized", sc, t, "denormalized", "mlp", s))
    return jobs


def pilot_jobs():
    picks = [j for j in all_jobs() if j["seed"] == 0 and j["experiment"] in ("main_deepsets", "extended_attention")]
    chosen = []
    for name in ("K2_D2_d128", "K16_D8_d128", "K64_D32_d128"):
        chosen += [j for j in picks if j["scenario"] == name and j["experiment"] == "main_deepsets"
                   and j["variant"] == "sum"]
    chosen += [j for j in picks if j["experiment"] == "extended_attention" and j["heads"] == 4][::20][:3]
    return [dict(j, experiment="pilot_" + j["experiment"]) for j in chosen]


def _init_worker(threads):
    import torch
    torch.set_num_threads(threads)


def _run(j):
    from relational_aggregation_bottleneck.training import train_and_evaluate
    t0 = time.time()
    try:
        out = train_and_evaluate(j["K"], j["D"], j["d"], j["target"], j["family"], j["variant"], j["seed"],
                                 heads=j["heads"])
        return {**j, **out, "status": "ok", "wall_seconds": round(time.time() - t0, 2)}
    except Exception as e:  # keep the queue running; failed runs are retried on the next start
        return {**j, "status": "error", "error": repr(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--experiments", nargs="*", default=None, help="subset of experiment names")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--results", default=str(RESULTS_PATH))
    args = ap.parse_args()

    jobs = pilot_jobs() if args.pilot else all_jobs()
    if args.experiments:
        jobs = [j for j in jobs if j["experiment"] in args.experiments]
    if args.list:
        from collections import Counter
        for k, v in Counter(j["experiment"] for j in jobs).items():
            print(f"{k:24s} {v}")
        print(f"{'total':24s} {len(jobs)}")
        return

    path = Path(args.results)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        for line in path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                done.add(job_key(r))
    todo = [j for j in jobs if job_key(j) not in done]
    # cheap runs first inside each experiment so early results arrive quickly
    order = ["main_deepsets", "gnn_architecture", "attention_head_sweep", "denormalized", "extended_sum",
             "extended_attention"]
    todo.sort(key=lambda j: (order.index(j["experiment"].removeprefix("pilot_")), j["K"] * j["D"]))
    print(f"{len(jobs)} runs in plan, {len(jobs) - len(todo)} already finished, {len(todo)} to run "
          f"with {args.workers} workers x {args.threads_per_worker} threads", flush=True)
    if not todo:
        print("ALL_RUNS_COMPLETE", flush=True)
        return

    t0, finished, errors = time.time(), 0, 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.threads_per_worker,),
                  maxtasksperchild=25) as pool, path.open("a") as f:
        for r in pool.imap_unordered(_run, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            finished += 1
            errors += r["status"] != "ok"
            elapsed = time.time() - t0
            eta = elapsed / finished * (len(todo) - finished)
            msg = (f"[{finished}/{len(todo)}] {r['experiment']} {r['scenario']} {r['target']} {r['variant']}"
                   f" h{r['heads']} s{r['seed']}: ")
            msg += (f"test AUROC {r['test_auroc']:.3f} ({r['seconds']:.0f}s)" if r["status"] == "ok"
                    else f"ERROR {r['error']}")
            print(f"{msg} | elapsed {elapsed / 3600:.2f} h, remaining about {eta / 3600:.2f} h", flush=True)
    print(f"Finished {finished} runs with {errors} errors.", flush=True)
    if errors == 0 and not args.pilot:
        print("ALL_RUNS_COMPLETE", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
