#!/usr/bin/env python3
"""
Analysis of the synthetic Join Bottleneck Ratio experiments.

Reads results/synthetic_runs/synthetic_training_runs.jsonl and writes to results/synthetic_analysis/:
  scenario_level_auroc.csv             mean and standard deviation of test AUROC per scenario and model
  sigmoid_fits_main_deepsets.csv       sigmoid fit per target x aggregator, AUROC vs log2 JBR
  predictor_comparison.csv             sigmoid R^2 using different difficulty predictors (row JBR, multiset JBR,
                                       log2 K, log2 D, log2 K*D) for every target x aggregator
  regime_summary.csv                   mean AUROC in the sub-critical, critical and super-critical regimes
  iso_jbr_consistency.csv              coefficient of variation across the five JBR = 1 decompositions
  architecture_fits.csv                sigmoid fits for SumAgg, MeanAgg, AttnAgg
  attention_head_sweep.csv             AUROC by head count on the highest-JBR scenarios
  capacity_multiplier.csv              midpoint shift of attention (per head count) over SumAgg, extended range
  embedding_dimension_control.csv      AUROC at fixed (K, D) for different d (capacity versus sample difficulty)
  denormalization_gap.csv              denormalized MLP minus DeepSets sum AUROC per scenario
  synthetic_analysis_summary.json      headline numbers with bootstrap confidence intervals
  figures/*.pdf, figures/*.png

Sigmoid model:  AUROC(x) = floor + (ceiling - floor) / (1 + exp(slope * (x - midpoint))),  x = log2 JBR.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "synthetic_runs" / "synthetic_training_runs.jsonl"
OUT = ROOT / "results" / "synthetic_analysis"
FIG = OUT / "figures"
RNG = np.random.default_rng(20261104)
N_BOOT = 10_000
MIDPOINT_BOUNDS = (-8.0, 16.0)


def sigmoid(x, floor, ceiling, slope, midpoint):
    return floor + (ceiling - floor) / (1 + np.exp(np.clip(slope * (x - midpoint), -50, 50)))


def fit_sigmoid(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return None
    best = None
    for m0 in np.linspace(x.min(), x.max(), 7):
        for s0 in (0.5, 1.5, 4.0):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p, _ = curve_fit(sigmoid, x, y, p0=[0.5, min(1.0, y.max()), s0, m0],
                                     bounds=([0.3, 0.4, 0.01, MIDPOINT_BOUNDS[0]], [1.0, 1.0, 20.0, MIDPOINT_BOUNDS[1]]),
                                     maxfev=20000)
            except (RuntimeError, ValueError):
                continue
            sse = float(np.sum((y - sigmoid(x, *p)) ** 2))
            if best is None or sse < best[1]:
                best = (p, sse)
    if best is None:
        return None
    p, sse = best
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"floor": p[0], "ceiling": p[1], "slope": p[2], "midpoint_log2_jbr": p[3],
            "amplitude": p[1] - p[0], "r2": 1 - sse / sst if sst > 0 else float("nan"), "n_points": int(len(x)),
            "midpoint_inside_probed_range": bool(x.min() <= p[3] <= x.max()),
            "probed_range": [float(x.min()), float(x.max())]}


def bootstrap_ci(values, stat=np.mean, n=N_BOOT, alpha=0.05):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return [float("nan"), float("nan")]
    try:
        from scipy.stats import bootstrap
        res = bootstrap((values,), stat, n_resamples=n, method="BCa", random_state=RNG)
        return [float(res.confidence_interval.low), float(res.confidence_interval.high)]
    except Exception:
        boots = [stat(RNG.choice(values, len(values))) for _ in range(n)]
        return [float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))]


def load_runs():
    rows = [json.loads(l) for l in RUNS.open() if l.strip()]
    df = pd.DataFrame([r for r in rows if r.get("status") == "ok"])
    df = df.drop_duplicates(subset=["experiment", "scenario", "target", "family", "variant", "heads", "seed"],
                            keep="last")
    df["log2_jbr"] = np.log2(df["jbr_empirical"])
    df["log2_jbr_multiset"] = np.log2(df["jbr_multiset"])
    df["log2_K"] = np.log2(df["K"])
    df["log2_D"] = np.log2(df["D"])
    df["log2_KD"] = np.log2(df["K"] * df["D"])
    df["model"] = np.where(df["variant"] == "AttnAgg", "AttnAgg-h" + df["heads"].astype(str), df["variant"])
    return df


def scenario_means(df, keys):
    cols = ["log2_jbr", "log2_jbr_multiset", "log2_K", "log2_D", "log2_KD", "jbr_empirical", "jbr_multiset", "K", "D", "d"]
    agg = df.groupby(keys + ["scenario"]).agg(
        auroc_mean=("test_auroc", "mean"), auroc_sd=("test_auroc", "std"), n_seeds=("seed", "nunique"),
        **{c: (c, "mean") for c in cols}).reset_index()
    return agg


def regime(jbr):
    return np.where(jbr < 0.5, "sub-critical (JBR < 0.5)", np.where(jbr <= 2, "critical (0.5 <= JBR <= 2)",
                                                                    "super-critical (JBR > 2)"))


def main():
    if not RUNS.exists():
        sys.exit(f"No results at {RUNS}")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    summary = {"n_successful_runs": int(len(df)), "runs_by_experiment": df["experiment"].value_counts().to_dict()}
    scenario_means(df, ["experiment", "target", "model"]).to_csv(OUT / "scenario_level_auroc.csv", index=False)

    # ---------------- main DeepSets experiment ----------------
    main_df = df[df.experiment == "main_deepsets"]
    if len(main_df):
        sm = scenario_means(main_df, ["target", "variant"])
        fits, comp = [], []
        for (t, a), g in sm.groupby(["target", "variant"]):
            f = fit_sigmoid(g.log2_jbr, g.auroc_mean)
            if f:
                fits.append({"target": t, "aggregator": a, **f})
            for pred in ["log2_jbr", "log2_jbr_multiset", "log2_K", "log2_D", "log2_KD"]:
                fp = fit_sigmoid(g[pred], g.auroc_mean)
                comp.append({"target": t, "aggregator": a, "predictor": pred,
                             "r2": fp["r2"] if fp else float("nan"),
                             "spearman_rho": float(pd.Series(g[pred].values).corr(pd.Series(g.auroc_mean.values), method="spearman"))})
        fits = pd.DataFrame(fits)
        fits.to_csv(OUT / "sigmoid_fits_main_deepsets.csv", index=False)
        comp = pd.DataFrame(comp)
        comp.to_csv(OUT / "predictor_comparison.csv", index=False)
        if len(fits):
            summary["main_sigmoid_r2_mean"] = float(fits.r2.mean())
            summary["main_sigmoid_r2_range"] = [float(fits.r2.min()), float(fits.r2.max())]
            summary["main_sigmoid_r2_ci95_bca"] = bootstrap_ci(fits.r2.values)
            summary["main_midpoints_log2_jbr"] = {f"{r.target}/{r.aggregator}": round(r.midpoint_log2_jbr, 3)
                                                  for r in fits.itertuples()}
        else:
            fits = pd.DataFrame(columns=["target", "aggregator", "floor", "ceiling", "slope", "midpoint_log2_jbr", "r2"])
        summary["predictor_mean_r2"] = comp.groupby("predictor").r2.mean().round(4).to_dict()

        main_df = main_df.assign(regime=regime(main_df.jbr_empirical))
        reg = main_df.groupby(["regime"]).test_auroc.agg(["mean", "std", "count"]).reset_index()
        reg_t = main_df.groupby(["target", "variant", "regime"]).test_auroc.agg(["mean", "std", "count"]).reset_index()
        pd.concat([reg.assign(target="all", variant="all"), reg_t]).to_csv(OUT / "regime_summary.csv", index=False)
        summary["regime_mean_auroc"] = {r.regime: [round(r.mean, 4), round(r.std, 4), int(r.count)] for r in reg.itertuples()}

        iso = sm[np.isclose(sm.jbr_empirical, 1.0, atol=0.05) & sm.scenario.isin(
            [f"K{k}_D{dd}_d128" for k, dd in [(4, 32), (8, 16), (16, 8), (32, 4), (64, 2)]])]
        iso_cv = iso.groupby(["target", "variant"]).auroc_mean.agg(
            lambda v: float(np.std(v, ddof=1) / np.mean(v))).rename("coefficient_of_variation").reset_index()
        iso_cv["auroc_min"] = iso.groupby(["target", "variant"]).auroc_mean.min().values
        iso_cv["auroc_max"] = iso.groupby(["target", "variant"]).auroc_mean.max().values
        iso_cv.to_csv(OUT / "iso_jbr_consistency.csv", index=False)
        summary["iso_jbr_cv_max"] = float(iso_cv.coefficient_of_variation.max()) if len(iso_cv) else None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
        for ax, t in zip(axes, ["mean_based", "rare_pattern", "distributional"]):
            for a, mk in zip(["sum", "mean", "max"], ["o", "s", "^"]):
                g = sm[(sm.target == t) & (sm.variant == a)].sort_values("log2_jbr")
                ax.errorbar(g.log2_jbr, g.auroc_mean, yerr=g.auroc_sd, fmt=mk, ms=4, capsize=2, label=a, alpha=.8)
                f = fits[(fits.target == t) & (fits.aggregator == a)]
                if len(f):
                    xs = np.linspace(g.log2_jbr.min(), g.log2_jbr.max(), 200)
                    p = f.iloc[0]
                    ax.plot(xs, sigmoid(xs, p.floor, p.ceiling, p.slope, p.midpoint_log2_jbr), lw=1)
            ax.axvline(0, ls="--", c="gray", lw=.8)
            ax.set_title(t.replace("_", " "))
            ax.set_xlabel(r"$\log_2$ JBR")
        axes[0].set_ylabel("test AUROC")
        axes[0].legend(title="DeepSets", fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(FIG / f"synthetic_phase_transition.{ext}", dpi=200)
        plt.close(fig)

    # ---------------- GNN architecture comparison ----------------
    arch = df[df.experiment == "gnn_architecture"]
    if len(arch):
        sm = scenario_means(arch, ["target", "variant"])
        rows = []
        for (t, a), g in sm.groupby(["target", "variant"]):
            f = fit_sigmoid(g.log2_jbr, g.auroc_mean)
            sub = g[g.jbr_empirical < 0.5].auroc_mean.mean()
            sup = g[g.jbr_empirical > 2].auroc_mean.mean()
            rows.append({"target": t, "architecture": a, "subcritical_auroc": sub, "supercritical_auroc": sup,
                         **(f or {})})
        pd.DataFrame(rows).to_csv(OUT / "architecture_fits.csv", index=False)
        summary["architecture"] = rows

    # ---------------- head-count sweep ----------------
    hs = df[df.experiment == "attention_head_sweep"]
    if len(hs):
        t = hs.groupby(["scenario", "heads"]).test_auroc.agg(["mean", "std"]).reset_index()
        t.to_csv(OUT / "attention_head_sweep.csv", index=False)
        summary["head_sweep_mean_auroc"] = hs.groupby("heads").test_auroc.mean().round(4).to_dict()

    # ---------------- extended range: capacity multiplier ----------------
    ext = df[df.experiment.isin(["extended_sum", "extended_attention"])]
    if len(ext):
        sm = scenario_means(ext, ["model"])
        fits = {m: fit_sigmoid(g.log2_jbr, g.auroc_mean) for m, g in sm.groupby("model")}
        base = fits.get("SumAgg")
        rows = []
        for m, f in fits.items():
            if not f:
                continue
            row = {"model": m, **f}
            if base and m != "SumAgg":
                shift = f["midpoint_log2_jbr"] - base["midpoint_log2_jbr"]
                # bootstrap over seeds: resample seed-level runs within each scenario
                boots = []
                for _ in range(300):
                    def resample(model):
                        g = ext[ext.model == model]
                        pick = g.groupby("scenario").sample(frac=1.0, replace=True,
                                                            random_state=int(RNG.integers(1e9)))
                        m_ = pick.groupby("scenario").agg(x=("log2_jbr", "mean"), y=("test_auroc", "mean"))
                        fb = fit_sigmoid(m_.x, m_.y)
                        return fb["midpoint_log2_jbr"] if fb else np.nan
                    boots.append(resample(m) - resample("SumAgg"))
                boots = np.array(boots)
                row.update(midpoint_shift_log2=shift, capacity_multiplier=2 ** shift,
                           shift_ci95=[float(np.nanquantile(boots, .025)), float(np.nanquantile(boots, .975))])
                row["multiplier_ci95"] = [2 ** row["shift_ci95"][0], 2 ** row["shift_ci95"][1]]
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "capacity_multiplier.csv", index=False)
        summary["capacity_multiplier"] = rows

        # embedding-dimension control: same (K, D), different d
        ctrl = sm[sm.model.isin(["SumAgg"])].copy()
        dup = ctrl.groupby(["K", "D"]).d.transform("nunique") > 1
        ctrl[dup].sort_values(["K", "D", "d"]).to_csv(OUT / "embedding_dimension_control.csv", index=False)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for m, g in sm.groupby("model"):
            g = g.sort_values("log2_jbr")
            ax.plot(g.log2_jbr, g.auroc_mean, "o", ms=3, alpha=.6, label=m)
            f = fits.get(m)
            if f:
                xs = np.linspace(g.log2_jbr.min(), g.log2_jbr.max(), 300)
                ax.plot(xs, sigmoid(xs, f["floor"], f["ceiling"], f["slope"], f["midpoint_log2_jbr"]), lw=1)
        ax.set_xlabel(r"$\log_2$ JBR")
        ax.set_ylabel("test AUROC (rare pattern)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        for e in ("pdf", "png"):
            fig.savefig(FIG / f"extended_capacity_multiplier.{e}", dpi=200)
        plt.close(fig)

    # ---------------- denormalized baseline ----------------
    den = df[df.experiment == "denormalized"]
    if len(den) and len(main_df):
        a = scenario_means(den, ["target"]).set_index(["target", "scenario"]).auroc_mean
        b = scenario_means(main_df[main_df.variant == "sum"], ["target"]).set_index(["target", "scenario"])
        gap = pd.DataFrame({"denormalized_auroc": a, "deepsets_sum_auroc": b.auroc_mean, "jbr": b.jbr_empirical}).dropna()
        gap["gap"] = gap.denormalized_auroc - gap.deepsets_sum_auroc
        gap.reset_index().to_csv(OUT / "denormalization_gap.csv", index=False)
        summary["denormalized_minus_sum_by_target"] = gap.groupby(level=0).gap.mean().round(4).to_dict()

    with open(OUT / "synthetic_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
