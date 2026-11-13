#!/usr/bin/env python3
"""
Analysis of experiments/run_review_robustness_experiments.py.

task_families / distributions: configuration means over seeds, compared with the frozen rare-pattern predictions of
  analysis/preregister_capacity_law_predictions.py (the 12 configurations are a subset of the 36 out-of-sample ones), and
  with a sigmoid refitted per family on the corrected predictor and on log2 JBR (post hoc, in-sample).
  For the non-uniform distributions the frozen law is also evaluated with the measured whole-row entropy in place of
  the summed per-column entropy.
query_conditioning: AUROC per scenario and child input.
attention_margins: attention mass on the matching child and score margin against log K.
width_budget: best validation learning rate per width, seed and budget; AUROC by width.

Output: results/review_robustness/analysis/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_synthetic_phase_transition import fit_sigmoid, sigmoid  # noqa: E402
from fit_generalized_join_bottleneck_ratio import predictor  # noqa: E402

RUNS = ROOT / "results" / "review_robustness" / "runs.jsonl"
OUT = ROOT / "results" / "review_robustness" / "analysis"
LV = ROOT / "results" / "law_validation_and_depth"


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([json.loads(line) for line in open(RUNS)])
    df = df[df.status == "ok"].copy()
    fits = json.load(open(LV / "preregistered_predictor_fits.json"))["rare_pattern"]
    law = fits["corrected_law"]
    nom = fits["nominal_jbr"]
    pred = pd.read_csv(LV / "preregistered_predictions.csv")
    pred = pred[pred.target == "rare_pattern"][["K", "D", "d", "pred_corrected_law", "pred_nominal_jbr"]]
    # same protocol as the new runs: learning rate 1e-3, seeds 0 and 1
    lv = pd.DataFrame([json.loads(line) for line in open(LV / "runs.jsonl")])
    lv = lv[(lv.status == "ok") & (lv.experiment == "out_of_sample_validation") & (lv.target == "rare_pattern")
            & (lv.lr == 1e-3) & lv.seed.isin([0, 1])]
    obs_uniform = lv.groupby(["K", "D", "d"]).test_auroc.mean().rename("auroc_exact_match_uniform").reset_index()

    def law_at(g, h_col):
        x = predictor([law["exponent_H"], law["exponent_d"]], g.assign(row_entropy_bits=g[h_col]), "free")
        return sigmoid(x, law["floor"], law["ceiling"], law["slope"], law["midpoint"]), x

    rows, cells = [], []
    for exp, col in (("task_families", "task"), ("distributions", "dist")):
        sub = df[df.experiment == exp]
        for name, g in sub.groupby(col):
            c = g.groupby(["K", "D", "d"]).agg(auroc=("test_auroc", "mean"), seed_sd=("test_auroc", "std"),
                                               row_entropy_bits=("row_entropy_bits", "mean"),
                                               joint_row_entropy_bits=("joint_row_entropy_bits", "mean"),
                                               n_seeds=("seed", "nunique")).reset_index()
            c = c.merge(pred, on=["K", "D", "d"]).merge(obs_uniform, on=["K", "D", "d"])
            c["pred_law_plugin_entropy"], x_law = law_at(c, "row_entropy_bits")
            c["pred_law_joint_entropy"], x_joint = law_at(c, "joint_row_entropy_bits")
            x_nom = np.log2(c.K * c.row_entropy_bits / c.d)
            f_law, f_nom = fit_sigmoid(x_law, c.auroc), fit_sigmoid(x_nom, c.auroc)
            sg = lambda x, f: sigmoid(x, f["floor"], f["ceiling"], f["slope"], f["midpoint_log2_jbr"])  # noqa: E731
            row = {"experiment": exp, "variant": name, "n_configurations": len(c),
                   "mean_auroc": c.auroc.mean(), "mean_auroc_exact_match_uniform": c.auroc_exact_match_uniform.mean(),
                   "frozen_law_r2": r2(c.auroc, c.pred_law_plugin_entropy),
                   "frozen_law_mae": float((c.auroc - c.pred_law_plugin_entropy).abs().mean()),
                   "frozen_nominal_r2": r2(c.auroc, c.pred_nominal_jbr),
                   "spearman_law": float(spearmanr(-x_law, c.auroc)[0]),
                   "spearman_nominal": float(spearmanr(-x_nom, c.auroc)[0]),
                   "refit_sigmoid_r2_law_x": f_law["r2"] if f_law else np.nan,
                   "refit_sigmoid_r2_nominal_x": f_nom["r2"] if f_nom else np.nan,
                   "refit_midpoint_law_x": f_law["midpoint_log2_jbr"] if f_law else np.nan,
                   "frozen_law_midpoint": law["midpoint"],
                   "mean_plugin_entropy": c.row_entropy_bits.mean(), "mean_joint_entropy": c.joint_row_entropy_bits.mean()}
            if exp == "distributions":
                row["frozen_law_joint_entropy_r2"] = r2(c.auroc, c.pred_law_joint_entropy)
            rows.append(row)
            cells.append(c.assign(experiment=exp, variant=name))
    ref = obs_uniform.merge(pred, on=["K", "D", "d"])
    ref = ref[ref.set_index(["K", "D", "d"]).index.isin(pd.concat(cells).set_index(["K", "D", "d"]).index)]
    x_ref = predictor([law["exponent_H"], law["exponent_d"]], ref.assign(row_entropy_bits=ref.D.astype(float)), "free")
    x_nref = np.log2(ref.K * ref.D / ref.d)
    f_law, f_nom = fit_sigmoid(x_ref, ref.auroc_exact_match_uniform), fit_sigmoid(x_nref, ref.auroc_exact_match_uniform)
    rows.insert(0, {"experiment": "reference", "variant": "exact match, uniform bits, same protocol",
                    "n_configurations": len(ref), "mean_auroc": ref.auroc_exact_match_uniform.mean(),
                    "frozen_law_r2": r2(ref.auroc_exact_match_uniform, ref.pred_corrected_law),
                    "frozen_law_mae": float((ref.auroc_exact_match_uniform - ref.pred_corrected_law).abs().mean()),
                    "frozen_nominal_r2": r2(ref.auroc_exact_match_uniform, ref.pred_nominal_jbr),
                    "spearman_law": float(spearmanr(-x_ref, ref.auroc_exact_match_uniform)[0]),
                    "spearman_nominal": float(spearmanr(-x_nref, ref.auroc_exact_match_uniform)[0]),
                    "refit_sigmoid_r2_law_x": f_law["r2"], "refit_sigmoid_r2_nominal_x": f_nom["r2"],
                    "refit_midpoint_law_x": f_law["midpoint_log2_jbr"]})
    fam = pd.DataFrame(rows)
    fam.to_csv(OUT / "task_family_and_distribution_tests.csv", index=False)
    pd.concat(cells).to_csv(OUT / "task_family_and_distribution_cells.csv", index=False)

    q = df[df.experiment == "query_conditioning"]
    qc = q.groupby(["K", "D", "d", "model"]).test_auroc.mean().unstack()
    qc.insert(0, "nominal_jbr", [k * D / d for k, D, d in qc.index])
    qc = qc.sort_values("nominal_jbr")
    qc.to_csv(OUT / "query_conditioning.csv")
    params = q.groupby("model").n_parameters.median()

    a = df[df.experiment == "attention_margins"]
    am = a.groupby("K")[["test_auroc", "attention_mass_on_match_mean", "attention_mass_on_match_median",
                         "score_margin_median", "share_match_is_argmax"]].mean()
    am["log_K"] = np.log(am.index)
    am.to_csv(OUT / "attention_margins.csv")

    w = df[df.experiment == "width_budget"]
    best = w.loc[w.groupby(["d", "seed", "budget"]).val_auroc.idxmax()]
    wb = best.groupby(["d", "budget"]).agg(auroc=("test_auroc", "mean"), lr=("lr", lambda s: list(s)),
                                           steps=("steps_run", "mean"), params=("n_parameters", "first")).unstack()
    wb.to_csv(OUT / "width_budget.csv")
    with open(OUT / "summary.json", "w") as f:
        json.dump({"families": fam.round(4).to_dict("records"), "query_parameters": params.to_dict()}, f, indent=2,
                  default=float)
    pd.set_option("display.width", 250)
    print(fam.round(3).to_string(), qc.round(3).to_string(), params, am.round(3).to_string(), wb.to_string(), sep="\n\n")


if __name__ == "__main__":
    main()
