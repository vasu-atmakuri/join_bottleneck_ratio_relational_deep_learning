#!/usr/bin/env python3
"""
Additional analyses of existing runs, requested in review. Nothing here is refitted on the out-of-sample runs unless the
output says "post hoc".

  exponent_uncertainty     bootstrap over configurations (1000 resamples) of a_H, a_d and the sigmoid midpoint of the
                           corrected law, on the 76 DeepSets-sum fitting configurations and on the 63 one-layer
                           message-passing configurations
  alternative_predictors   post hoc: predictors fitted on the same 76 fitting configurations and evaluated on the 36
                           out-of-sample configurations: linear and quadratic regression on (log K, log D, log d),
                           gradient boosting and k-nearest neighbours on the same inputs, isotonic, piecewise-linear and
                           five-parameter logistic curves in the corrected predictor; plus run-level R^2 and residual
                           correlations of the frozen corrected law with log K, log D and log d
  table1_cross_validation  leave-one-configuration-out R^2 of the sigmoid in log2 JBR for every Table I fit, against
                           isotonic and piecewise-linear curves, and bootstrap midpoint intervals over configurations
  irrelevant_entropy       which information measure orders the irrelevant-column control best
  learning_rate_selection  shares of the selected learning rate by model family and width
  run_manifest             one row per executed training run of every experiment family, with counts
  planted_bins             test entities, positive rate and bootstrap AUROC interval per child-count bin

Output: results/review_response_analysis/
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_synthetic_phase_transition import fit_sigmoid, sigmoid  # noqa: E402
from fit_generalized_join_bottleneck_ratio import fit as fit_free, predictor  # noqa: E402

OUT = ROOT / "results" / "review_response_analysis"
LV = ROOT / "results" / "law_validation_and_depth"
RNG = np.random.default_rng(20261113)


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def selected():
    return pd.read_csv(ROOT / "results" / "learning_rate_tuned_analysis" / "selected_runs.csv")


def fitting_configs(target):
    sel = selected()
    t = sel[sel.family_group.isin(["main_deepsets", "width_sweep"]) & (sel.model == "sum") & (sel.target == target)]
    return t.groupby(["K", "D", "d"]).agg(auroc=("test_auroc", "mean"),
                                          row_entropy_bits=("row_entropy_bits", "mean")).reset_index()


def gnn_configs():
    sel = selected()
    t = sel[sel.family_group.isin(["gnn_architecture", "extended"]) & (sel.model == "SumAgg")
            & (sel.target == "rare_pattern")]
    return t.groupby(["K", "D", "d"]).agg(auroc=("test_auroc", "mean"),
                                          row_entropy_bits=("row_entropy_bits", "mean")).reset_index()


def exponent_uncertainty(n_boot=1000):
    rows = []
    sets = {"deepsets_sum_rare_pattern": fitting_configs("rare_pattern"),
            "deepsets_sum_distributional": fitting_configs("distributional"),
            "gnn_sum_rare_pattern": gnn_configs()}
    for name, g in sets.items():
        y = g.auroc.values
        p = fit_free(g, "free", y)
        boots = []
        for _ in range(n_boot):
            idx = RNG.integers(0, len(g), len(g))
            boots.append(fit_free(g.iloc[idx].reset_index(drop=True), "free", y[idx], start=p))
        b = np.array(boots)
        q = lambda j: [float(np.quantile(b[:, j], .025)), float(np.quantile(b[:, j], .975))]  # noqa: E731
        rows.append({"fit": name, "n_configurations": len(g), "exponent_H": p[4], "exponent_H_ci95": q(4),
                     "exponent_d": p[5], "exponent_d_ci95": q(5), "midpoint": p[3], "midpoint_ci95": q(3),
                     "share_exponent_d_below_0.5": float((b[:, 5] < 0.5).mean())})
    return pd.DataFrame(rows)


def oos_runs():
    df = pd.DataFrame([json.loads(line) for line in open(LV / "runs.jsonl")])
    oos = df[(df.status == "ok") & (df.experiment == "out_of_sample_validation")]
    return oos.loc[oos.groupby(["scenario", "target", "seed"]).val_auroc.idxmax()]


def five_pl(x, a, b, s, m, g):
    return a + (b - a) / (1 + np.exp(np.clip(s * (x - m), -50, 50))) ** g


def piecewise(x, lo, hi, x1, x2):
    return np.where(x <= x1, hi, np.where(x >= x2, lo, hi + (lo - hi) * (x - x1) / max(x2 - x1, 1e-6)))


def fit_curve(f, x, y, p0, bounds):
    best = None
    for shift in np.linspace(-3, 3, 7):
        q = list(p0)
        q[3 if f is five_pl else 2] = q[3 if f is five_pl else 2] + shift
        if f is piecewise:
            q[3] = q[2] + 2
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = curve_fit(f, x, y, p0=q, bounds=bounds, maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        sse = float(((f(x, *p) - y) ** 2).sum())
        if best is None or sse < best[1]:
            best = (p, sse)
    return best[0]


def alternative_predictors():
    pred = pd.read_csv(LV / "preregistered_predictions.csv")
    fits = json.load(open(LV / "preregistered_predictor_fits.json"))
    obs = pd.read_csv(LV / "out_of_sample_by_configuration.csv")[["K", "D", "d", "target", "observed"]]
    runs = oos_runs()
    rows, resid = [], []
    for target in ["rare_pattern", "distributional"]:
        tr = fitting_configs(target)
        te = pred[pred.target == target].merge(obs, on=["K", "D", "d", "target"])
        law = fits[target]["corrected_law"]
        xtr = predictor([law["exponent_H"], law["exponent_d"]], tr, "free")
        xte = predictor([law["exponent_H"], law["exponent_d"]], te, "free")
        Ztr = np.log2(tr[["K", "row_entropy_bits", "d"]].values.astype(float))
        Zte = np.log2(te[["K", "row_entropy_bits", "d"]].values.astype(float))
        out = {"frozen corrected law (pre-registered)": te.pred_corrected_law.values,
               "frozen original JBR (pre-registered)": te.pred_nominal_jbr.values}
        out["linear regression on log K, log D, log d"] = LinearRegression().fit(Ztr, tr.auroc).predict(Zte)
        poly = PolynomialFeatures(2)
        out["quadratic regression with interactions"] = LinearRegression().fit(
            poly.fit_transform(Ztr), tr.auroc).predict(poly.transform(Zte))
        out["gradient boosting on log K, log D, log d"] = GradientBoostingRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.05, random_state=0).fit(Ztr, tr.auroc).predict(Zte)
        out["5-nearest neighbours on log K, log D, log d"] = KNeighborsRegressor(5).fit(Ztr, tr.auroc).predict(Zte)
        out["isotonic in corrected predictor"] = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
            xtr, tr.auroc).predict(xte)
        p = fit_curve(piecewise, xtr, tr.auroc.values, [0.5, 1.0, 5.0, 8.0], ([0.3, 0.6, -10, -10], [0.8, 1.0, 30, 30]))
        out["piecewise linear in corrected predictor"] = piecewise(xte, *p)
        p = fit_curve(five_pl, xtr, tr.auroc.values, [0.5, 1.0, 1.0, 6.0, 1.0],
                      ([0.3, 0.6, 0.05, -20, 0.1], [0.8, 1.0, 20, 30, 10]))
        out["five-parameter logistic in corrected predictor"] = five_pl(xte, *p)
        for name, yhat in out.items():
            rows.append({"target": target, "predictor": name, "r2": r2(te.observed, yhat),
                         "mae": float(np.abs(te.observed - yhat).mean())})
        # run-level R^2 of the frozen predictions
        rr = runs[runs.target == target].merge(te[["K", "D", "d", "pred_corrected_law", "pred_nominal_jbr"]],
                                                on=["K", "D", "d"])
        rows.append({"target": target, "predictor": "frozen corrected law, run level", "r2":
                     r2(rr.test_auroc, rr.pred_corrected_law), "mae": float((rr.test_auroc - rr.pred_corrected_law).abs()
                                                                          .mean()), "n_runs": len(rr)})
        rows.append({"target": target, "predictor": "frozen original JBR, run level", "r2":
                     r2(rr.test_auroc, rr.pred_nominal_jbr), "mae": float((rr.test_auroc - rr.pred_nominal_jbr).abs()
                                                                        .mean()), "n_runs": len(rr)})
        e = te.observed - te.pred_corrected_law
        for v, col in (("log2 K", "K"), ("log2 D", "D"), ("log2 d", "d")):
            resid.append({"target": target, "variable": v,
                          "pearson_r_with_residual": float(np.corrcoef(np.log2(te[col]), e)[0, 1])})
        te.assign(residual=e).to_csv(OUT / f"oos_residuals_{target}.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(resid)


def table1_cross_validation():
    sel = selected()
    groups = {("DeepSets", "sum"): ("main_deepsets", "sum"), ("DeepSets", "mean"): ("main_deepsets", "mean"),
              ("GNN 1 layer", "sum"): ("gnn_architecture", "SumAgg"),
              ("GNN 1 layer", "mean"): ("gnn_architecture", "MeanAgg"),
              ("Extended", "sum"): ("extended", "SumAgg"), ("Extended", "attn 1h"): ("extended", "AttnAgg-h1"),
              ("Extended", "attn 4h"): ("extended", "AttnAgg-h4")}
    rows = []
    for (fam, mdl), (fg, m) in groups.items():
        for target in ["rare_pattern", "distributional"]:
            g = sel[(sel.family_group == fg) & (sel.model == m) & (sel.target == target)]
            if g.empty:
                continue
            c = g.groupby(["K", "D", "d"]).agg(auroc=("test_auroc", "mean"), x=("log2_jbr", "mean")).reset_index()
            x, y = c.x.values, c.auroc.values
            loo = {"sigmoid": [], "isotonic": [], "piecewise": []}
            for i in range(len(c)):
                mk = np.arange(len(c)) != i
                f = fit_sigmoid(x[mk], y[mk])
                loo["sigmoid"].append(sigmoid(x[i], f["floor"], f["ceiling"], f["slope"], f["midpoint_log2_jbr"]))
                loo["isotonic"].append(IsotonicRegression(increasing=False, out_of_bounds="clip").fit(
                    x[mk], y[mk]).predict([x[i]])[0])
                p = fit_curve(piecewise, x[mk], y[mk], [0.5, 1.0, 0.0, 2.0], ([0.3, 0.6, -10, -10], [0.8, 1.0, 15, 15]))
                loo["piecewise"].append(float(piecewise(np.array([x[i]]), *p)[0]))
            f = fit_sigmoid(x, y)
            mids = []
            for _ in range(500):
                idx = RNG.integers(0, len(c), len(c))
                fb = fit_sigmoid(x[idx], y[idx])
                if fb is not None:
                    mids.append(fb["midpoint_log2_jbr"])
            rows.append({"family": fam, "model": mdl, "target": target, "n_configurations": len(c),
                         "in_sample_r2": f["r2"], "floor": f["floor"], "ceiling": f["ceiling"],
                         "midpoint": f["midpoint_log2_jbr"],
                         "midpoint_ci95_over_configurations": [float(np.quantile(mids, .025)),
                                                               float(np.quantile(mids, .975))],
                         **{f"loo_r2_{k}": r2(y, v) for k, v in loo.items()}})
    return pd.DataFrame(rows)


def irrelevant_entropy():
    df = pd.DataFrame([json.loads(line) for line in open(ROOT / "results" / "capacity_law_controls" /
                                                         "control_training_runs.jsonl")])
    df = df[(df.status == "ok") & (df.experiment == "irrelevant_features")].copy()
    df["n_irrelevant"] = df.opts.apply(lambda o: (o or {}).get("n_irrelevant", 0))
    g = df.groupby(["K", "D", "d", "target", "variant", "heads", "n_irrelevant"]).test_auroc.mean().reset_index()
    g["H_relevant"] = g.D.astype(float)
    g["H_total"] = g.D + g.n_irrelevant
    rows = []
    for (t, v, h), s in g.groupby(["target", "variant", "heads"]):
        for name, col in (("task-relevant entropy K*H_rel/d", s.K * s.H_relevant / s.d),
                          ("total entropy K*H_total/d", s.K * s.H_total / s.d),
                          ("irrelevant column count", s.n_irrelevant)):
            from scipy.stats import spearmanr
            rho = spearmanr(col, s.test_auroc)[0] if col.nunique() > 1 else np.nan
            rows.append({"target": t, "variant": v, "heads": h, "measure": name, "spearman_with_auroc": rho,
                         "n_cells": len(s)})
    g.to_csv(OUT / "irrelevant_columns_cells.csv", index=False)
    return pd.DataFrame(rows)


def learning_rate_selection():
    sel = selected()
    sel["width_bin"] = pd.cut(sel.d, [0, 32, 128, 1024], labels=["d<=32", "32<d<=128", "d>128"])
    a = sel.groupby(["family_group", "model"]).lr.value_counts(normalize=True).unstack(fill_value=0)
    b = sel[sel.family_group == "width_sweep"].groupby("width_bin", observed=True).lr.value_counts(
        normalize=True).unstack(fill_value=0)
    return a, b


def run_manifest():
    """Every executed run with its family, configuration, seed, learning rate and metrics."""
    src = {
        "first_suite_fixed_lr": ROOT / "results" / "synthetic_runs" / "synthetic_training_runs.jsonl",
        "learning_rate_tuned_suite": ROOT / "results" / "learning_rate_tuned_core" / "core_training_runs.jsonl",
        "capacity_law_controls": ROOT / "results" / "capacity_law_controls" / "control_training_runs.jsonl",
        "law_validation_and_depth": LV / "runs.jsonl",
        "review_robustness": ROOT / "results" / "review_robustness" / "runs.jsonl",
        "relbench_interventions": ROOT / "results" / "relbench_interventions" / "intervention_runs.jsonl",
        "relbench_planted_and_variants": ROOT / "results" / "relbench_planted_signal" / "runs.jsonl",
    }
    rows = []
    for suite, p in src.items():
        if not p.exists():
            continue
        for i, line in enumerate(open(p)):
            r = json.loads(line)
            rows.append({"suite": suite, "line": i, "experiment": r.get("experiment") or r.get("family"),
                         "status": r.get("status"), "K": r.get("K"), "D": r.get("D"), "d": r.get("d", r.get("width")),
                         "target": r.get("target") or r.get("task") or r.get("label"),
                         "model": r.get("variant") or r.get("model") or r.get("aggregation"),
                         "heads": r.get("heads"), "seed": r.get("seed"), "lr": r.get("lr", r.get("learning_rate")),
                         "dataset": r.get("dataset"), "test_metric": r.get("test_auroc", r.get("test_metric"))})
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "run_manifest.csv", index=False)
    counts = m.groupby(["suite", "experiment", "status"]).size().rename("runs").reset_index()
    counts.to_csv(OUT / "run_manifest_counts.csv", index=False)
    syn = m[m.suite.isin(["first_suite_fixed_lr", "learning_rate_tuned_suite", "capacity_law_controls",
                          "law_validation_and_depth"]) & (m.status == "ok")]
    return counts, {"synthetic_completed_original_suites": int(len(syn)),
                    "synthetic_configurations_KDd": int(syn[["K", "D", "d"]].drop_duplicates().shape[0]),
                    "review_robustness_completed": int(((m.suite == "review_robustness") & (m.status == "ok")).sum()),
                    "relbench_completed": int((m.suite.str.startswith("relbench") & (m.status == "ok")).sum()),
                    "relbench_failed_and_rerun_or_excluded": int((m.suite.str.startswith("relbench")
                                                                 & (m.status != "ok")).sum())}


def planted_bins():
    p = ROOT / "results" / "relbench_planted_signal" / "analysis" / "planted_signal_by_children.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    counts, totals = run_manifest()
    summary["run_totals"] = totals
    print(counts.to_string(), totals, sep="\n")
    a, b = learning_rate_selection()
    a.to_csv(OUT / "learning_rate_selection_by_model.csv")
    b.to_csv(OUT / "learning_rate_selection_by_width.csv")
    print(a.round(2), b.round(2), sep="\n")
    irr = irrelevant_entropy()
    irr.to_csv(OUT / "irrelevant_columns_measures.csv", index=False)
    print(irr.round(3).to_string())
    alt, res = alternative_predictors()
    alt.to_csv(OUT / "out_of_sample_alternative_predictors.csv", index=False)
    res.to_csv(OUT / "out_of_sample_residual_correlations.csv", index=False)
    print(alt.round(3).to_string(), res.round(3).to_string(), sep="\n")
    t1 = table1_cross_validation()
    t1.to_csv(OUT / "table1_cross_validation.csv", index=False)
    print(t1.round(3).to_string())
    ex = exponent_uncertainty()
    ex.to_csv(OUT / "exponent_uncertainty.csv", index=False)
    print(ex.round(3).to_string())
    summary.update(exponents=ex.to_dict("records"), alternatives=alt.to_dict("records"),
                   residuals=res.to_dict("records"), table1=t1.to_dict("records"))
    with open(OUT / "review_response_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == "__main__":
    main()
