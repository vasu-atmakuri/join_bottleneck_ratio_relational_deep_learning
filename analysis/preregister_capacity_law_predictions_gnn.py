#!/usr/bin/env python3
"""
Pre-registered predictions for the out-of-sample test of the capacity law on one-layer message passing with sum
aggregation (rare_pattern target), run before the out_of_sample_validation_gnn runs.

Fitting data: learning-rate-selected SumAgg runs of the architecture subset (d = 128) and the extended range
(d = 4 to 128), rare_pattern target. Predictors, frozen here:
  nominal_jbr              sigmoid of log2(K * H_row / d)
  transferred_law          exponents a_H, a_d taken unchanged from the DeepSets fit
                           (preregistered_predictor_fits.json), only the sigmoid refitted on message-passing data
  refitted_law             exponents a_H, a_d refitted on the message-passing data

Output: results/law_validation_and_depth/preregistered_predictions_gnn.csv, preregistered_predictor_fits_gnn.json
"""

import json
import sys
import time
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


def main():
    sel = pd.read_csv(ROOT / "results" / "learning_rate_tuned_analysis" / "selected_runs.csv")
    g = sel[sel.family_group.isin(["gnn_architecture", "extended"]) & (sel.model == "SumAgg")
            & (sel.target == "rare_pattern")]
    g = g.groupby(["K", "D", "d"]).agg(auroc=("test_auroc", "mean"),
                                         row_entropy_bits=("row_entropy_bits", "mean")).reset_index()
    ds = json.load(open(OUT / "preregistered_predictor_fits.json"))["rare_pattern"]["corrected_law"]
    a_h, a_d = ds["exponent_H"], ds["exponent_d"]
    x_nom = np.log2(g.K * g.row_entropy_bits / g.d)
    f_nom = fit_sigmoid(x_nom, g.auroc)
    x_tr = predictor([a_h, a_d], g, "free")
    f_tr = fit_sigmoid(x_tr, g.auroc)
    p = fit_free(g, "free", g.auroc.values)

    def r2(y, yhat):
        return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())

    sg = lambda x, f: sigmoid(x, f["floor"], f["ceiling"], f["slope"], f["midpoint_log2_jbr"])  # noqa: E731
    fits = {"n_training_configurations": len(g),
            "nominal_jbr": {"in_sample_r2": r2(g.auroc, sg(x_nom, f_nom)),
                            **{k: float(v) for k, v in f_nom.items() if np.isscalar(v)}},
            "transferred_law": {"exponent_H": a_h, "exponent_d": a_d, "in_sample_r2": r2(g.auroc, sg(x_tr, f_tr)),
                                **{k: float(v) for k, v in f_tr.items() if np.isscalar(v)}},
            "refitted_law": {"in_sample_r2": r2(g.auroc, sigmoid(predictor(p[4:], g, "free"), *p[:4])),
                             **dict(zip(["floor", "ceiling", "slope", "midpoint", "exponent_H", "exponent_d"],
                                        map(float, p)))},
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    new = pd.DataFrame(out_of_sample_configurations(), columns=["K", "D", "d"])
    new["row_entropy_bits"] = new.D.astype(float)
    new["target"] = "rare_pattern"
    new["log2_nominal_jbr"] = np.log2(new.K * new.D / new.d)
    new["pred_nominal_jbr"] = sg(new.log2_nominal_jbr, f_nom)
    new["pred_transferred_law"] = sg(predictor([a_h, a_d], new, "free"), f_tr)
    new["pred_refitted_law"] = sigmoid(predictor(p[4:], new, "free"), *p[:4])
    new.round(4).to_csv(OUT / "preregistered_predictions_gnn.csv", index=False)
    with open(OUT / "preregistered_predictor_fits_gnn.json", "w") as f:
        json.dump(fits, f, indent=2)
    print(json.dumps(fits, indent=2))


if __name__ == "__main__":
    main()
