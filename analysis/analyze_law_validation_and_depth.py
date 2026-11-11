#!/usr/bin/env python3
"""
Analysis of run_law_validation_and_depth_experiments.py.

Out-of-sample validation: for every new configuration and seed the learning rate with the best validation AUROC is
selected; mean test AUROC per configuration is compared with the frozen predictions in
preregistered_predictions.csv (R^2, Spearman, mean absolute error, bootstrap CI of the R^2 difference between the
corrected law and the nominal ratio, resampling configurations).

Depth: sigmoid fits of AUROC on log2 JBR for two-layer models with and without the parent -> child message, compared
with the one-layer GNN fits of the learning-rate-tuned study (midpoint shift, bootstrap CI over seeds).

Outputs (results/law_validation_and_depth/): out_of_sample_scores.csv, out_of_sample_by_configuration.csv,
depth_fits.csv, law_validation_and_depth_summary.json, figures/{out_of_sample_validation,depth}.pdf|png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_synthetic_phase_transition import fit_sigmoid  # noqa: E402

D = ROOT / "results" / "law_validation_and_depth"
RNG = np.random.default_rng(20261109)
PRED = {"pred_nominal_jbr": "nominal JBR", "pred_multiset_jbr": "multiset JBR", "pred_corrected_law": "corrected law"}


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def load():
    df = pd.DataFrame([json.loads(line) for line in open(D / "runs.jsonl")])
    df = df[df.status == "ok"].copy()
    df["downward"] = df.model_options.apply(lambda o: o.get("downward"))
    return df


def out_of_sample(df):
    oos = df[df.experiment == "out_of_sample_validation"]
    sel = oos.loc[oos.groupby(["scenario", "target", "seed"]).val_auroc.idxmax()]
    obs = sel.groupby(["K", "D", "d", "target"]).test_auroc.agg(observed="mean", seed_sd="std").reset_index()
    m = obs.merge(pd.read_csv(D / "preregistered_predictions.csv"), on=["K", "D", "d", "target"])
    m.to_csv(D / "out_of_sample_by_configuration.csv", index=False)
    rows = []
    for t, g in m.groupby("target"):
        for p, name in PRED.items():
            rows.append({"target": t, "predictor": name, "n_configurations": len(g), "r2": r2(g.observed, g[p]),
                         "spearman": float(spearmanr(g.observed, g[p])[0]),
                         "mean_absolute_error": float((g.observed - g[p]).abs().mean())})
        boots = []
        for _ in range(2000):
            b = g.iloc[RNG.integers(0, len(g), len(g))]
            boots.append(r2(b.observed, b.pred_corrected_law) - r2(b.observed, b.pred_nominal_jbr))
        rows.append({"target": t, "predictor": "corrected minus nominal (R^2)", "n_configurations": len(g),
                     "r2": r2(g.observed, g.pred_corrected_law) - r2(g.observed, g.pred_nominal_jbr),
                     "r2_ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]})
    scores = pd.DataFrame(rows)
    scores.to_csv(D / "out_of_sample_scores.csv", index=False)
    return m, scores


def out_of_sample_gnn(df):
    """Pre-registered test on one-layer message passing (sum aggregation, rare_pattern)."""
    g = df[df.experiment == "out_of_sample_validation_gnn"]
    if not len(g):
        return pd.DataFrame()
    sel = g.loc[g.groupby(["scenario", "seed"]).val_auroc.idxmax()]
    obs = sel.groupby(["K", "D", "d"]).test_auroc.agg(observed="mean", seed_sd="std").reset_index()
    m = obs.merge(pd.read_csv(D / "preregistered_predictions_gnn.csv"), on=["K", "D", "d"])
    m.to_csv(D / "out_of_sample_gnn_by_configuration.csv", index=False)
    rows = []
    for p, name in (("pred_nominal_jbr", "nominal JBR"), ("pred_transferred_law", "law with DeepSets exponents"),
                    ("pred_refitted_law", "law with refitted exponents")):
        rows.append({"predictor": name, "n_configurations": len(m), "r2": r2(m.observed, m[p]),
                     "spearman": float(spearmanr(m.observed, m[p])[0]),
                     "mean_absolute_error": float((m.observed - m[p]).abs().mean())})
    boots = []
    for _ in range(2000):
        b = m.iloc[RNG.integers(0, len(m), len(m))]
        boots.append(r2(b.observed, b.pred_transferred_law) - r2(b.observed, b.pred_nominal_jbr))
    rows.append({"predictor": "transferred minus nominal (R^2)", "n_configurations": len(m),
                 "r2": r2(m.observed, m.pred_transferred_law) - r2(m.observed, m.pred_nominal_jbr),
                 "r2_ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]})
    out = pd.DataFrame(rows)
    out.to_csv(D / "out_of_sample_gnn_scores.csv", index=False)
    return out


def depth(df):
    dp = df[df.experiment.isin(["two_layer_depth", "two_layer_depth_extended"])].copy()
    keys = ["experiment", "scenario", "target", "variant", "downward", "seed"]
    sel = dp.loc[dp.groupby(keys).val_auroc.idxmax()].copy()
    sel["log2_jbr"] = np.log2(sel.jbr_empirical)
    sel["model"] = sel.variant + np.where(sel.downward, " two-layer", " two-layer, no downward")
    one = pd.read_csv(ROOT / "results" / "learning_rate_tuned_analysis" / "selected_runs.csv")
    one = one[(one.family_group == "gnn_architecture") & one.model.isin(["SumAgg", "MeanAgg"])
              & (one.target != "mean_based")].copy()
    one["model"] = one.model + " one-layer"
    both = pd.concat([sel[sel.experiment == "two_layer_depth"], one], ignore_index=True)
    rows = []
    for (t, m), g in both.groupby(["target", "model"]):
        s = g.groupby("scenario").agg(x=("log2_jbr", "mean"), y=("test_auroc", "mean")).reset_index()
        f = fit_sigmoid(s.x, s.y)
        mids = []
        for _ in range(300):
            b = g.groupby("scenario").sample(frac=1.0, replace=True, random_state=int(RNG.integers(1e9)))
            sb = b.groupby("scenario").agg(x=("log2_jbr", "mean"), y=("test_auroc", "mean"))
            fb = fit_sigmoid(sb.x, sb.y)
            mids.append(fb["midpoint_log2_jbr"] if fb else np.nan)
        rows.append({"target": t, "model": m, "r2": f["r2"], "midpoint_log2_jbr": f["midpoint_log2_jbr"],
                     "midpoint_ci95": [float(np.nanquantile(mids, .025)), float(np.nanquantile(mids, .975))],
                     "auroc_jbr_at_least_8": float(s[s.x >= 3].y.mean())})
    fits = pd.DataFrame(rows)
    fits.to_csv(D / "depth_fits.csv", index=False)
    ext = sel[sel.experiment == "two_layer_depth_extended"]
    return both, fits, ext


def figures(m, both, ext):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    (D / "figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharex=True, sharey=True)
    for ax, (p, name) in zip(axes, [("pred_nominal_jbr", "nominal JBR"), ("pred_corrected_law", "corrected law")]):
        for t, mk in (("rare_pattern", "o"), ("distributional", "s")):
            g = m[m.target == t]
            ax.plot(g[p], g.observed, mk, ms=2.6, alpha=.8, label=t.replace("_", " "))
        ax.plot([.45, 1.01], [.45, 1.01], "k--", lw=.6)
        sc = [r2(m[m.target == t].observed, m[m.target == t][p]) for t in ("rare_pattern", "distributional")]
        ax.set_title(f"{name}\n$R^2$ = {sc[0]:.2f} / {sc[1]:.2f}", fontsize=7)
        ax.set_xlabel("pre-registered prediction")
    axes[0].set_ylabel("observed test AUROC")
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper left")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(D / "figures" / f"out_of_sample_validation.{e}", bbox_inches="tight", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    style = {"SumAgg one-layer": ("#1f77b4", "o", ":"), "SumAgg two-layer, no downward": ("#7f7f7f", "v", "--"),
             "SumAgg two-layer": ("#d62728", "s", "-"), "MeanAgg two-layer": ("#ff7f0e", "^", "-")}
    g0 = both[both.target == "rare_pattern"]
    for mdl, (c, mk, ls) in style.items():
        s = g0[g0.model == mdl].groupby("scenario").agg(x=("log2_jbr", "mean"), y=("test_auroc", "mean")) \
            .sort_values("x")
        ax.plot(s.x, s.y, marker=mk, ls=ls, ms=2.6, color=c, lw=.8, label=mdl.replace("SumAgg", "sum")
                .replace("MeanAgg", "mean"))
    if len(ext):
        s = ext.groupby("scenario").agg(x=("log2_jbr", "mean"), y=("test_auroc", "mean")).sort_values("x")
        ax.plot(s.x, s.y, "s", ms=2.2, color="#d62728", alpha=.45, label="sum two-layer, extended range")
    ax.axvline(0, color="k", lw=.5, ls="--")
    ax.set_xlabel(r"$\log_2$ JBR")
    ax.set_ylabel("test AUROC (rare pattern)")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(D / "figures" / f"depth.{e}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    df = load()
    m, scores = out_of_sample(df)
    gnn = out_of_sample_gnn(df)
    both, fits, ext = depth(df)
    figures(m, both, ext)
    summary = {"out_of_sample": scores.round(4).to_dict("records"),
               "out_of_sample_gnn": gnn.round(4).to_dict("records"), "depth": fits.round(4).to_dict("records"),
               "extended_two_layer_auroc_by_log2_jbr": ext.assign(x=np.log2(ext.jbr_empirical).round())
               .groupby("x").test_auroc.mean().round(3).to_dict()}
    with open(D / "law_validation_and_depth_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
