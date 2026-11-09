#!/usr/bin/env python3
"""
Pre-registered predictions for the out-of-sample validation of the capacity law.

Three predictors are fitted on the existing learning-rate-tuned DeepSets sum runs (core grid and width sweep, every
configuration with K, D and d powers of two) and frozen; they then predict test AUROC for the 36 new configurations of
run_law_validation_and_depth_experiments.py, none of which was used for fitting:
  nominal_jbr     sigmoid of log2(K * H_row / d)
  multiset_jbr    sigmoid of log2(H_set / d), H_set = log2 C(2^H_row + K - 1, K)
  corrected_law   sigmoid of log2 K + a_H log2 H_row - a_d log2 d with fitted exponents
This script must be run, and its output committed, before the out-of-sample runs; the commit time is the
pre-registration record.

Output: results/law_validation_and_depth/preregistered_predictions.csv and preregistered_predictor_fits.json
"""

import json
import sys
import time
from math import lgamma, log
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "experiments"))
from analyze_synthetic_phase_transition import fit_sigmoid, sigmoid  # noqa: E402
from fit_generalized_join_bottleneck_ratio import fit as fit_free, predictor  # noqa: E402
from run_law_validation_and_depth_experiments import out_of_sample_configurations  # noqa: E402

OUT = ROOT / "results" / "law_validation_and_depth"


def log2_multiset_bits(h_row, K):
    M = 2.0 ** h_row
    return (lgamma(M + K) - lgamma(K + 1) - lgamma(M)) / log(2) if M < 1e15 else K * h_row - lgamma(K + 1) / log(2)


def main():
    sel = pd.read_csv(ROOT / "results" / "learning_rate_tuned_analysis" / "selected_runs.csv")
    train = sel[sel.family_group.isin(["main_deepsets", "width_sweep"]) & (sel.model == "sum")]
    new = pd.DataFrame(out_of_sample_configurations(), columns=["K", "D", "d"])
    new["row_entropy_bits"] = new.D.astype(float)
    new["log2_nominal_jbr"] = np.log2(new.K * new.D / new.d)
    new["log2_multiset_jbr"] = [np.log2(log2_multiset_bits(h, k) / d) for h, k, d in
                                zip(new.row_entropy_bits, new.K, new.d)]
    fits, rows = {}, []
    for target in ["rare_pattern", "distributional"]:
        g = train[train.target == target].groupby(["K", "D", "d"]).agg(
            auroc=("test_auroc", "mean"), row_entropy_bits=("row_entropy_bits", "mean")).reset_index()
        g["x_nominal"] = np.log2(g.K * g.row_entropy_bits / g.d)
        g["x_multiset"] = [np.log2(log2_multiset_bits(h, k) / d) for h, k, d in zip(g.row_entropy_bits, g.K, g.d)]
        f_nom = fit_sigmoid(g.x_nominal, g.auroc)
        f_set = fit_sigmoid(g.x_multiset, g.auroc)
        p = fit_free(g, "free", g.auroc.values)
        yhat = sigmoid(predictor(p[4:], g, "free"), *p[:4])
        r2_law = float(1 - ((g.auroc - yhat) ** 2).sum() / ((g.auroc - g.auroc.mean()) ** 2).sum())
        fits[target] = {"n_training_configurations": len(g), "corrected_law_in_sample_r2": r2_law,
                        "nominal_jbr": {k: float(v) for k, v in f_nom.items() if np.isscalar(v)},
                        "multiset_jbr": {k: float(v) for k, v in f_set.items() if np.isscalar(v)},
                        "corrected_law": dict(zip(["floor", "ceiling", "slope", "midpoint", "exponent_H",
                                                   "exponent_d"], map(float, p)))}
        out = new.copy()
        out["target"] = target
        out["pred_nominal_jbr"] = sigmoid(out.log2_nominal_jbr, f_nom["floor"], f_nom["ceiling"], f_nom["slope"],
                                          f_nom["midpoint_log2_jbr"])
        out["pred_multiset_jbr"] = sigmoid(out.log2_multiset_jbr, f_set["floor"], f_set["ceiling"], f_set["slope"],
                                           f_set["midpoint_log2_jbr"])
        out["pred_corrected_law"] = sigmoid(predictor(p[4:], out, "free"), *p[:4])
        rows.append(out)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).round(4).to_csv(OUT / "preregistered_predictions.csv", index=False)
    fits["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(OUT / "preregistered_predictor_fits.json", "w") as f:
        json.dump(fits, f, indent=2)
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
