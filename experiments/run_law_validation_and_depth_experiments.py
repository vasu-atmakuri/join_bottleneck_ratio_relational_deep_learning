#!/usr/bin/env python3
"""
Two follow-up experiment families for the capacity law.

out_of_sample_validation
  36 configurations never used for fitting (K in {3, 6, 12, 24, 48, 96}, D in {3, 6, 12, 24},
  d in {24, 48, 96, 192}), DeepSets sum, rare_pattern and distributional targets, 2 seeds, learning rate selected on
  validation from {3e-4, 1e-3} (plus 1e-4 when d >= 192). Predicted AUROC for every configuration is written by
  analysis/preregister_capacity_law_predictions.py BEFORE these runs and compared afterwards.

out_of_sample_validation_gnn
  The same 36 configurations for one-layer message passing with sum aggregation (rare_pattern, 3 seeds); predictions
  from analysis/preregister_capacity_law_predictions_gnn.py are frozen before these runs.

two_layer_depth
  Two message-passing layers on the parent-child graph, as in relational deep learning where each foreign key is used
  in both directions. With the downward (parent -> child) message, children can compare themselves with the parent
  before pooling; without it, layer 1 is a child-only transformation of equal depth (control for depth).
    11 architecture scenarios x 2 set targets x {SumAgg, MeanAgg with downward, SumAgg without downward}
       x 3 seeds x lr {3e-4, 1e-3}                                                        = 396
    extreme scenarios with nominal JBR 32 to 256 x rare_pattern x SumAgg with downward x 3 seeds x 2 lr

Results: results/law_validation_and_depth/runs.jsonl (resumable, one line per run)
Usage:   python experiments/run_law_validation_and_depth_experiments.py --workers 3 --threads-per-worker 2
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relational_aggregation_bottleneck.scenarios import architecture_scenarios, extreme_scenarios  # noqa: E402

RESULTS = ROOT / "results" / "law_validation_and_depth" / "runs.jsonl"
SET_TARGETS = ["rare_pattern", "distributional"]


def out_of_sample_configurations():
    """36 (K, D, d) spread evenly over nominal log2 JBR from -3 to 5, deterministic."""
    from math import log2
    pool = [(K, D, d) for K, D, d in product([3, 6, 12, 24, 48, 96], [3, 6, 12, 24], [24, 48, 96, 192])
            if -3 <= log2(K * D / d) <= 5]
    pool.sort(key=lambda c: (log2(c[0] * c[1] / c[2]), c[2], c[0]))
    step = len(pool) / 36
    return [pool[int(i * step)] for i in range(36)]


def J(experiment, K, D, d, target, family, variant, seed, lr, options=None):
    return {"experiment": experiment, "scenario": f"K{K}_D{D}_d{d}", "K": K, "D": D, "d": d, "target": target,
            "family": family, "variant": variant, "heads": 1, "seed": seed, "lr": lr,
            "model_options": options or {}}


def key(j):
    return "|".join(str(j[k]) for k in ("experiment", "scenario", "target", "family", "variant", "seed", "lr")) + \
        "|" + json.dumps(j["model_options"], sort_keys=True)


def all_jobs():
    jobs = []
    for K, D, d in out_of_sample_configurations():
        lrs = [3e-4, 1e-3] + ([1e-4] if d >= 192 else [])
        for t, s, lr in product(SET_TARGETS, [0, 1, 2], lrs):
            jobs.append(J("out_of_sample_validation", K, D, d, t, "deepsets", "sum", s, lr, {"hidden": max(128, d)}))
    for K, D, d in out_of_sample_configurations():
        lrs = [3e-4, 1e-3] + ([1e-4] if d >= 192 else [])
        for s, lr in product([0, 1, 2], lrs):
            jobs.append(J("out_of_sample_validation_gnn", K, D, d, "rare_pattern", "gnn", "SumAgg", s, lr))
    variants = [("SumAgg", True), ("MeanAgg", True), ("SumAgg", False)]
    for sc in architecture_scenarios():
        for t, (v, down), s, lr in product(SET_TARGETS, variants, [0, 1, 2], [3e-4, 1e-3]):
            jobs.append(J("two_layer_depth", sc.K, sc.D, sc.d, t, "gnn_two_layer", v, s, lr, {"downward": down}))
    for sc in extreme_scenarios():
        if sc.nominal_jbr <= 256:
            for s, lr in product([0, 1, 2], [3e-4, 1e-3]):
                jobs.append(J("two_layer_depth_extended", sc.K, sc.D, sc.d, "rare_pattern", "gnn_two_layer", "SumAgg",
                              s, lr, {"downward": True}))
    return jobs


def _init(threads):
    import torch
    torch.set_num_threads(threads)


def _run(j):
    from relational_aggregation_bottleneck.training import train_and_evaluate
    t0 = time.time()
    try:
        out = train_and_evaluate(j["K"], j["D"], j["d"], j["target"], j["family"], j["variant"], j["seed"],
                                 config={"learning_rate": j["lr"]}, model_options=j["model_options"] or None)
        return {**j, **out, "status": "ok", "wall_seconds": round(time.time() - t0, 2)}
    except Exception as e:
        return {**j, "status": "error", "error": repr(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    jobs = all_jobs()
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
    order = ["out_of_sample_validation", "two_layer_depth", "two_layer_depth_extended", "out_of_sample_validation_gnn"]
    todo.sort(key=lambda j: (order.index(j["experiment"]), j["K"] * j["D"]))
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
            print(f"[{n}/{len(todo)}] {r['experiment']} {r['scenario']} {r['target']} {r['variant']} "
                  f"{r['model_options']} lr{r['lr']} s{r['seed']}: {st} | {el / 3600:.2f} h elapsed, about "
                  f"{el / n * (len(todo) - n) / 3600:.1f} h left", flush=True)
    print(f"Finished {n} runs with {errors} errors.", flush=True)
    if errors == 0:
        print("ALL_LAW_VALIDATION_AND_DEPTH_RUNS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
