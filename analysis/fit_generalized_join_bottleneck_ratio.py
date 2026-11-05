#!/usr/bin/env python3
"""
Tests whether a generalized Join Bottleneck Ratio explains aggregation performance better than JBR = K * H / d.

Generalized form (exponents estimated from data):
    log2 GJBR = a_K * log2 K + a_H * log2 H_row - a_d * log2 d      (a_K = 1 fixes the scale; slope absorbs it)
which reduces to the original JBR when a_H = a_d = 1. A single-parameter variant, JBR_beta = K^(1 + beta) * H / d,
isolates the extra effect of cardinality.

For every model family (DeepSets sum and mean, GNN SumAgg and MeanAgg, extended-range SumAgg and AttnAgg) and the
two set targets, AUROC is fitted with a four-parameter sigmoid of the predictor. Reported per fit:
  in-sample R^2, leave-one-scenario-out cross-validated R^2, iso-JBR coefficient of variation under the predictor,
  exponent estimates with 95% bootstrap intervals over scenarios.

Outputs (results/synthetic_analysis/)
  generalized_jbr_fits.csv, generalized_jbr_summary.json, figures/generalized_jbr_collapse.{pdf,png}
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "synthetic_runs" / "synthetic_training_runs.jsonl"
OUT = ROOT / "results" / "synthetic_analysis"
RNG = np.random.default_rng(20261105)
N_BOOT = 200


def sigmoid(x, floor, ceiling, slope, mid):
    return floor + (ceiling - floor) / (1 + np.exp(np.clip(slope * (x - mid), -50, 50)))


def predictor(theta_exp, df, form):
    lk, lh, ld = np.log2(df.K.values), np.log2(df.row_entropy_bits.values), np.log2(df.d.values)
    if form == "jbr":
        return lk + lh - ld
    if form == "jbr_beta":
        return (1 + theta_exp[0]) * lk + lh - ld
    if form == "free":
        return lk + theta_exp[0] * lh - theta_exp[1] * ld
    raise ValueError(form)


N_EXP = {"jbr": 0, "jbr_beta": 1, "free": 2}


def fit(df, form, y, start=None):
    """Least-squares sigmoid fit; multi-start unless a warm start (previous solution) is given."""
    n = N_EXP[form]
    best = None
    starts = ([list(start)] if start is not None else
              [[0.5, 1.0, 2.0, mid0] + list(e0) for mid0 in np.linspace(-2, 8, 6)
               for e0 in ([[]] if n == 0 else [[0.0], [0.5]] if n == 1 else [[1.0, 1.0], [0.5, 0.5]])])
    for p0 in starts:
        if True:
            lo = [0.3, 0.6, 0.05, -20] + [-1.0] * n
            hi = [0.8, 1.0, 20.0, 30] + [4.0] * n

            def resid(p):
                return sigmoid(predictor(p[4:], df, form), *p[:4]) - y
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=4000)
                except ValueError:
                    continue
            if best is None or r.cost < best.cost:
                best = r
    return best.x


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


def evaluate(df, form):
    y = df.auroc.values
    p = fit(df, form, y)
    yhat = sigmoid(predictor(p[4:], df, form), *p[:4])
    loo = np.empty(len(df))
    for i in range(len(df)):
        m = np.ones(len(df), bool)
        m[i] = False
        pi = fit(df[m], form, y[m], start=p)
        loo[i] = sigmoid(predictor(pi[4:], df.iloc[[i]], form), *pi[:4])[0]
    res = {"form": form, "n_scenarios": len(df), "r2": r2(y, yhat), "loo_cv_r2": r2(y, loo),
           "floor": p[0], "ceiling": p[1], "slope": p[2], "midpoint": p[3]}
    if form == "jbr_beta":
        res["beta"] = p[4]
    if form == "free":
        res["exponent_H"], res["exponent_d"] = p[4], p[5]
    if N_EXP[form]:
        boots = []
        for _ in range(N_BOOT):
            idx = RNG.integers(0, len(df), len(df))
            boots.append(fit(df.iloc[idx], form, y[idx], start=p)[4:])
        boots = np.array(boots)
        names = ["beta"] if form == "jbr_beta" else ["exponent_H", "exponent_d"]
        for j, nme in enumerate(names):
            res[f"{nme}_ci95"] = [float(np.quantile(boots[:, j], .025)), float(np.quantile(boots[:, j], .975))]
    return res, p


def main():
    rows = [json.loads(l) for l in open(RUNS) if l.strip()]
    runs = pd.DataFrame([r for r in rows if r.get("status") == "ok"])
    runs["model"] = np.where(runs.variant == "AttnAgg", "AttnAgg-h" + runs.heads.astype(str), runs.variant)
    groups = {
        "deepsets_sum": runs[(runs.experiment == "main_deepsets") & (runs.variant == "sum")],
        "deepsets_mean": runs[(runs.experiment == "main_deepsets") & (runs.variant == "mean")],
        "gnn_SumAgg": runs[(runs.experiment == "gnn_architecture") & (runs.variant == "SumAgg")],
        "gnn_MeanAgg": runs[(runs.experiment == "gnn_architecture") & (runs.variant == "MeanAgg")],
        "extended_SumAgg": runs[runs.experiment == "extended_sum"],
        "extended_AttnAgg-h1": runs[(runs.experiment == "extended_attention") & (runs.heads == 1)],
    }
    out, collapse = [], {}
    for gname, g in groups.items():
        for target in sorted(g.target.unique()):
            if target == "mean_based":
                continue
            sm = (g[g.target == target].groupby("scenario")
                  .agg(auroc=("test_auroc", "mean"), K=("K", "first"), D=("D", "first"), d=("d", "first"),
                       row_entropy_bits=("row_entropy_bits", "mean")).reset_index())
            for form in ("jbr", "jbr_beta", "free"):
                if form == "free" and sm.d.nunique() == 1:
                    continue  # d is not identifiable without variation in d
                res, p = evaluate(sm, form)
                x = predictor(p[4:], sm, form)
                iso = sm[np.isclose(sm.K * sm.row_entropy_bits / sm.d, 1.0, atol=0.05)]
                res.update(group=gname, target=target,
                           iso_jbr_auroc_cv=float(iso.auroc.std(ddof=1) / iso.auroc.mean()) if len(iso) > 2 else None)
                out.append(res)
                collapse[(gname, target, form)] = (x, sm.auroc.values, p)
                print(f"{gname:20s} {target:15s} {form:9s} R2 {res['r2']:.3f}  LOO-CV R2 {res['loo_cv_r2']:.3f}  "
                      + (f"beta {res['beta']:.2f} {np.round(res['beta_ci95'], 2)}" if form == "jbr_beta" else "")
                      + (f"a_H {res['exponent_H']:.2f} {np.round(res['exponent_H_ci95'], 2)} a_d {res['exponent_d']:.2f} "
                         f"{np.round(res['exponent_d_ci95'], 2)}" if form == "free" else ""), flush=True)
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "generalized_jbr_fits.csv", index=False)
    with open(OUT / "generalized_jbr_summary.json", "w") as f:
        json.dump(df.round(4).to_dict("records"), f, indent=2, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, form, title in zip(axes, ("jbr", "jbr_beta"), ("JBR = K H / d", r"JBR$_\beta$ = K$^{1+\beta}$ H / d")):
        for (gname, target, f_), (x, y, p) in collapse.items():
            if f_ != form or not gname.startswith("deepsets_sum"):
                continue
            ax.plot(x, y, "o", ms=4, alpha=.75, label=target.replace("_", " "))
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, sigmoid(xs, *p[:4]), lw=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$\log_2$ predictor")
    axes[0].set_ylabel("test AUROC (DeepSets sum)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"generalized_jbr_collapse.{e}", dpi=200)
    print("GENERALIZED_JBR_DONE")


if __name__ == "__main__":
    main()
