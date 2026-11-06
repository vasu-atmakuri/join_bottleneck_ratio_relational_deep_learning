#!/usr/bin/env python3
"""
Analysis of the learning-rate-tuned synthetic experiments.

For every configuration (experiment family, scenario, target, model, heads) and seed, the learning rate with the best
validation AUROC among {1e-4, 3e-4, 1e-3} is selected; test AUROC is reported for that choice only. The 1e-3 runs of
the core families come from the first suite (results/synthetic_runs), the others from
results/learning_rate_tuned_core.

Outputs (results/learning_rate_tuned_analysis/)
  selected_runs.csv                     one row per configuration and seed after learning-rate selection
  learning_rate_selection.csv           how often each learning rate was selected, by family and model
  sigmoid_fits.csv                      AUROC vs log2 JBR per family x target x model (tuned)
  tuned_vs_fixed_comparison.csv         the same fits at the fixed learning rate 1e-3, for comparison
  capacity_multiplier.csv               attention midpoint shift over SumAgg on the extended range, bootstrap CI
  iso_jbr_consistency.csv               coefficient of variation over the five JBR = 1 decompositions
  width_exponent.csv                    free-exponent fit log2 X = log2 K + a_H log2 H - a_d log2 d on the width sweep
  width_sweep_table.csv                 mean AUROC by (K, D) and d after learning-rate selection
  tuned_summary.json, figures/*.pdf|png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_synthetic_phase_transition import bootstrap_ci, fit_sigmoid, sigmoid  # noqa: E402
from fit_generalized_join_bottleneck_ratio import evaluate as fit_exponents  # noqa: E402

FIRST = ROOT / "results" / "synthetic_runs" / "synthetic_training_runs.jsonl"
TUNED = ROOT / "results" / "learning_rate_tuned_core" / "core_training_runs.jsonl"
OUT = ROOT / "results" / "learning_rate_tuned_analysis"
RNG = np.random.default_rng(20261106)
FAMILY_OF = {"main_deepsets": "main_deepsets", "main_deepsets_lr": "main_deepsets",
             "gnn_architecture": "gnn_architecture", "gnn_architecture_lr": "gnn_architecture",
             "extended_sum": "extended", "extended_attention": "extended", "extended_lr": "extended",
             "width_sweep_lr": "width_sweep"}


def load():
    rows = []
    for path, default_lr in ((FIRST, 1e-3), (TUNED, None)):
        if not path.exists():
            continue
        for line in path.open():
            r = json.loads(line)
            if r.get("status") != "ok" or r["experiment"] not in FAMILY_OF:
                continue
            if r["experiment"] in ("extended_attention",) and r["heads"] not in (1, 4):
                continue
            r["lr"] = r.get("lr", default_lr)
            r["family_group"] = FAMILY_OF[r["experiment"]]
            rows.append(r)
    df = pd.DataFrame(rows)
    df["model"] = np.where(df.variant == "AttnAgg", "AttnAgg-h" + df.heads.astype(str), df.variant)
    df["log2_jbr"] = np.log2(df.jbr_empirical)
    return df


def select(df):
    keys = ["family_group", "scenario", "K", "D", "d", "target", "model", "seed"]
    df = df.drop_duplicates(subset=keys + ["lr"], keep="last")
    idx = df.groupby(keys).val_auroc.idxmax()
    return df.loc[idx].reset_index(drop=True)


def scen(df, keys):
    return df.groupby(keys + ["scenario"]).agg(auroc=("test_auroc", "mean"), sd=("test_auroc", "std"),
                                               log2_jbr=("log2_jbr", "mean"), K=("K", "first"), D=("D", "first"),
                                               d=("d", "first"), H=("row_entropy_bits", "mean")).reset_index()


def fits(sel, label):
    out = []
    for fam in ["main_deepsets", "gnn_architecture", "extended"]:
        sm = scen(sel[sel.family_group == fam], ["target", "model"])
        for (t, m), g in sm.groupby(["target", "model"]):
            f = fit_sigmoid(g.log2_jbr, g.auroc)
            if f:
                out.append({"protocol": label, "family": fam, "target": t, "model": m, **f,
                            "auroc_jbr_below_0.5": g[g.log2_jbr < -1].auroc.mean(),
                            "auroc_jbr_above_2": g[g.log2_jbr > 1].auroc.mean()})
    return pd.DataFrame(out)


def multiplier(sel):
    ext = sel[sel.family_group == "extended"]
    base = scen(ext[ext.model == "SumAgg"], [])
    fb = fit_sigmoid(base.log2_jbr, base.auroc)
    rows = []
    for m in sorted(ext.model.unique()):
        if m == "SumAgg":
            continue
        sm = scen(ext[ext.model == m], [])
        f = fit_sigmoid(sm.log2_jbr, sm.auroc)
        boots = []
        for _ in range(300):
            def mid(model):
                g = ext[ext.model == model].groupby("scenario").sample(frac=1.0, replace=True,
                                                                       random_state=int(RNG.integers(1e9)))
                s = g.groupby("scenario").agg(x=("log2_jbr", "mean"), y=("test_auroc", "mean"))
                r = fit_sigmoid(s.x, s.y)
                return r["midpoint_log2_jbr"] if r else np.nan
            boots.append(mid(m) - mid("SumAgg"))
        shift = f["midpoint_log2_jbr"] - fb["midpoint_log2_jbr"]
        lo, hi = np.nanquantile(boots, [.025, .975])
        rows.append({"model": m, "sumagg_midpoint": fb["midpoint_log2_jbr"], "model_midpoint": f["midpoint_log2_jbr"],
                     "shift_log2": shift, "multiplier": 2 ** shift, "shift_ci95": [lo, hi],
                     "multiplier_ci95": [2 ** lo, 2 ** hi], "model_r2": f["r2"], "sumagg_r2": fb["r2"]})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    sel = select(df)
    sel.to_csv(OUT / "selected_runs.csv", index=False)
    lr_tab = sel.groupby(["family_group", "model", "lr"]).size().rename("selected").reset_index()
    lr_tab.to_csv(OUT / "learning_rate_selection.csv", index=False)

    fixed = df[np.isclose(df.lr, 1e-3)].drop_duplicates(
        subset=["family_group", "scenario", "target", "model", "seed"], keep="first")
    f_tuned, f_fixed = fits(sel, "tuned"), fits(fixed[fixed.family_group != "width_sweep"], "fixed_1e-3")
    f_tuned.to_csv(OUT / "sigmoid_fits.csv", index=False)
    pd.concat([f_tuned, f_fixed]).to_csv(OUT / "tuned_vs_fixed_comparison.csv", index=False)

    mult = multiplier(sel)
    mult.to_csv(OUT / "capacity_multiplier.csv", index=False)

    main_sel = sel[sel.family_group == "main_deepsets"]
    iso = scen(main_sel[main_sel.scenario.isin(["K4_D32_d128", "K8_D16_d128", "K16_D8_d128", "K32_D4_d128",
                                                 "K64_D2_d128"])], ["target", "model"])
    iso_cv = iso.groupby(["target", "model"]).auroc.agg(lambda v: float(np.std(v, ddof=1) / np.mean(v))) \
        .rename("coefficient_of_variation").reset_index()
    iso_cv.to_csv(OUT / "iso_jbr_consistency.csv", index=False)

    ws = sel[sel.family_group == "width_sweep"]
    width_rows, wtab = [], []
    for t, g in ws.groupby("target"):
        sm = g.groupby(["K", "D", "d"]).agg(auroc=("test_auroc", "mean"), row_entropy_bits=("row_entropy_bits", "mean")) \
            .reset_index()
        wtab.append(sm.assign(target=t))
        res, _ = fit_exponents(sm, "free")
        width_rows.append({"target": t, **res})
    pd.DataFrame(width_rows).to_csv(OUT / "width_exponent.csv", index=False)
    if wtab:
        pd.concat(wtab).pivot_table(index=["target", "K", "D"], columns="d", values="auroc").round(3) \
            .to_csv(OUT / "width_sweep_table.csv")

    set_fits = f_tuned[(f_tuned.target != "mean_based") & f_tuned.model.isin(["sum", "mean", "SumAgg", "MeanAgg"])]
    summary = {
        "runs_considered": int(len(df)), "configurations_after_selection": int(len(sel)),
        "learning_rate_selected_share": sel.lr.value_counts(normalize=True).round(3).to_dict(),
        "set_target_sum_mean_fits": set_fits[["family", "target", "model", "r2", "midpoint_log2_jbr", "amplitude"]]
        .round(3).to_dict("records"),
        "set_target_r2_mean": float(set_fits.r2.mean()) if len(set_fits) else None,
        "set_target_r2_ci95": bootstrap_ci(set_fits.r2.values) if len(set_fits) > 2 else None,
        "capacity_multiplier": mult.round(3).to_dict("records"),
        "iso_jbr_cv": iso_cv.round(4).to_dict("records"),
        "width_exponent": pd.DataFrame(width_rows).round(3).to_dict("records") if width_rows else [],
    }
    with open(OUT / "tuned_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, t in zip(axes, ["rare_pattern", "distributional"]):
        sm = scen(main_sel[main_sel.target == t], ["model"])
        for m, mk in zip(["sum", "mean"], ["o", "s"]):
            g = sm[sm.model == m].sort_values("log2_jbr")
            ax.errorbar(g.log2_jbr, g.auroc, yerr=g.sd, fmt=mk, ms=4, capsize=2, alpha=.8, label=f"DeepSets {m}")
            f = fit_sigmoid(g.log2_jbr, g.auroc)
            if f:
                xs = np.linspace(g.log2_jbr.min(), g.log2_jbr.max(), 200)
                ax.plot(xs, sigmoid(xs, f["floor"], f["ceiling"], f["slope"], f["midpoint_log2_jbr"]), lw=1)
        ax.axvline(0, ls="--", c="gray", lw=.8)
        ax.set_title(t.replace("_", " "))
        ax.set_xlabel(r"$\log_2$ JBR")
    axes[0].set_ylabel("test AUROC (learning rate tuned)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"tuned_phase_transition.{e}", dpi=200)
    plt.close(fig)
    if wtab:
        w = pd.concat(wtab)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
        for ax, t in zip(axes, ["rare_pattern", "distributional"]):
            for (K, D), g in w[w.target == t].groupby(["K", "D"]):
                ax.plot(np.log2(g.d), g.auroc, "o-", ms=3, label=f"K={K}, D={D}")
            ax.set_xlabel(r"$\log_2 d$")
            ax.set_title(t.replace("_", " "))
        axes[0].set_ylabel("test AUROC (learning rate tuned)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        for e in ("pdf", "png"):
            fig.savefig(OUT / "figures" / f"tuned_width_sweep.{e}", dpi=200)
    print(json.dumps(summary, indent=2, default=float)[:5000])


if __name__ == "__main__":
    main()
