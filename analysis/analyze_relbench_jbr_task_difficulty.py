#!/usr/bin/env python3
"""
Relates per-task Join Bottleneck Ratio features on the RelBench databases to published model results.

Inputs
  results/relbench_join_statistics/{foreign_key_join_statistics,table_feature_statistics,task_join_features}.csv
  real_data/published_relbench_baseline_results.csv   (LightGBM, RDL, RelGNN, RelGT on the 30 RelBench v1 tasks)

Effective JBR of a join. Message passing on the relational graph lets each child row carry, besides its own features,
the features of the other parents it points to (one-to-one lookups). The effective row entropy of a join is therefore
  H_eff = H_row_conditional(child features) + sum over the child's other foreign keys of H_row(other parent table)
and JBR_eff = K_mean_linked * H_eff / d. Both the plain and the effective variants are reported.

Task predictors (d = 128 unless stated): maximum, sum and entropy-weighted mean of JBR over the joins that aggregate
into the task's entity table (one hop) and into those tables (two hops), on a log2(1 + JBR) scale.
Baseline predictors: maximum join depth, tables within two hops, joins within two hops, schema size, average
clustering coefficient and degree of the schema graph.

Outcomes (higher = harder or more room for better aggregation)
  rdl_difficulty         within-task-type percentile of the RDL error (100 - AUROC, MAE / global-median MAE, -MAP)
  relational_gain        how much RDL improves on LightGBM (relative)
  relgnn_gain, relgt_gain  how much RelGNN and the RelGT transformer improve on RDL (relative)

Statistics: Spearman correlation over tasks with exact permutation p-values (10,000 permutations, shuffling the
predictor within databases), within-database Kendall concordance on task pairs of the same type, leave-one-database-out
cross-validated R^2 of a univariate linear model, Holm correction over the predictor x outcome family.

Outputs (results/relbench_jbr_analysis/): task_predictors_and_outcomes.csv, correlations.csv,
within_database_concordance.csv, leave_one_database_out.csv, relbench_jbr_summary.json,
figures/relbench_jbr_vs_outcomes.{pdf,png}, foreign_key_jbr_distribution.csv
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "results" / "relbench_join_statistics"
OUT = ROOT / "results" / "relbench_jbr_analysis"
RNG = np.random.default_rng(20261105)
N_PERM = 10_000
D = 128

JBR_PREDICTORS = ["jbr_max_one_hop", "jbr_entropy_weighted", "jbr_sum_two_hop",
                  "jbr_eff_max_one_hop", "jbr_eff_entropy_weighted", "jbr_eff_sum_two_hop"]
BASELINE_PREDICTORS = ["max_join_depth", "tables_within_two_hops", "n_joins_two_hop", "schema_tables",
                       "schema_avg_clustering", "schema_avg_degree"]
PRIMARY = ("jbr_eff_entropy_weighted", "rdl_difficulty")
OUTCOMES = ["rdl_difficulty", "relational_gain", "relgnn_gain", "relgt_gain"]


def effective_join_table(fk, tables):
    h_table = {(r.dataset, r.table): r.H_row_marginal for r in tables.itertuples()}
    fk = fk.copy()
    extra = []
    for r in fk.itertuples():
        others = fk[(fk.dataset == r.dataset) & (fk.child_table == r.child_table) & (fk.fk_column != r.fk_column)]
        extra.append(sum(h_table.get((r.dataset, p), 0.0) for p in others.parent_table))
    fk["H_lookup"] = extra
    fk["H_effective"] = fk.H_row_conditional + fk.H_lookup
    for d in (64, 128, 256):
        fk[f"JBR_eff_d{d}"] = fk.K_mean_linked * fk.H_effective / d
    return fk


def task_predictors(tasks, fk):
    rows = []
    for t in tasks.itertuples():
        f = fk[fk.dataset == t.dataset]
        one = f[f.parent_table == t.entity_table]
        two = pd.concat([one, f[f.parent_table.isin(set(one.child_table)) & (f.child_table != t.entity_table)]])
        rec = {"dataset": t.dataset, "task": t.task}
        for tag, col, hcol in (("jbr", f"JBR_d{D}", "H_row_conditional"), ("jbr_eff", f"JBR_eff_d{D}", "H_effective")):
            w = two[hcol]
            rec[f"{tag}_max_one_hop"] = np.log2(1 + (one[col].max() if len(one) else 0.0))
            rec[f"{tag}_sum_two_hop"] = np.log2(1 + two[col].sum())
            rec[f"{tag}_entropy_weighted"] = np.log2(1 + (float((two[col] * w).sum() / w.sum()) if w.sum() > 0 else 0.0))
        rows.append(rec)
    return pd.DataFrame(rows)


def outcomes(b):
    b = b.copy()
    err = np.where(b.metric == "auroc", 100 - b.rdl, np.where(b.metric == "mae", b.rdl / b.global_median, -b.rdl))
    b["rdl_error"] = err
    b["rdl_difficulty"] = b.groupby("task_type").rdl_error.rank(pct=True)
    rel = lambda new, old: np.where(b.metric == "auroc", (new - old) / old,  # noqa: E731
                                    np.where(b.metric == "mae", (old - new) / old, np.log(new / old)))
    b["relational_gain"] = rel(b.rdl, b.lightgbm)
    b["relgnn_gain"] = rel(b.relgnn, b.rdl)
    b["relgt_gain"] = np.where(b.relgt.notna(), rel(b.relgt, b.rdl), np.nan)
    for o in ("relational_gain", "relgnn_gain", "relgt_gain"):
        b[o] = b.groupby("task_type")[o].rank(pct=True)  # comparable across metrics
    return b


def perm_spearman(x, y, groups):
    rho = spearmanr(x, y).statistic
    if not np.isfinite(rho):
        return rho, np.nan
    count = 0
    xs = np.asarray(x, float)
    for _ in range(N_PERM):
        xp = xs.copy()
        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            xp[idx] = xp[RNG.permutation(idx)]
        r = spearmanr(xp, y).statistic
        count += abs(r) >= abs(rho) - 1e-12
    return rho, (count + 1) / (N_PERM + 1)


def concordance(df, pred, out):
    conc = disc = ties = 0
    for (_, _), g in df.groupby(["dataset", "task_type"]):
        for (_, a), (_, b) in combinations(g.iterrows(), 2):
            dx, dy = a[pred] - b[pred], a[out] - b[out]
            if abs(dx) < 1e-12 or abs(dy) < 1e-12:
                ties += 1
            elif dx * dy > 0:
                conc += 1
            else:
                disc += 1
    n = conc + disc
    return conc, disc, ties, (conc / n if n else np.nan)


def lodo_r2(df, pred, out):
    pred_y = np.empty(len(df))
    for ds in df.dataset.unique():
        tr, te = df.dataset != ds, df.dataset == ds
        a, b = np.polyfit(df.loc[tr, pred], df.loc[tr, out], 1) if df.loc[tr, pred].nunique() > 1 else (0, df.loc[tr, out].mean())
        pred_y[te.values] = a * df.loc[te, pred] + b
    y = df[out].values
    return 1 - np.sum((y - pred_y) ** 2) / np.sum((y - y.mean()) ** 2)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fk = pd.read_csv(STATS / "foreign_key_join_statistics.csv")
    tables = pd.read_csv(STATS / "table_feature_statistics.csv")
    tasks = pd.read_csv(STATS / "task_join_features.csv")
    base = pd.read_csv(ROOT / "real_data" / "published_relbench_baseline_results.csv")
    fk = effective_join_table(fk, tables)
    fk[["dataset", "child_table", "fk_column", "parent_table", "K_mean_linked", "H_row_conditional", "H_lookup",
        "H_effective", "JBR_d128", "JBR_eff_d128"]].to_csv(OUT / "foreign_key_jbr_distribution.csv", index=False)

    df = (base.merge(tasks[["dataset", "task"] + BASELINE_PREDICTORS], on=["dataset", "task"], how="left")
              .merge(task_predictors(tasks, fk), on=["dataset", "task"], how="left"))
    df = outcomes(df)
    df.to_csv(OUT / "task_predictors_and_outcomes.csv", index=False)
    missing = df[df[PRIMARY[0]].isna()][["dataset", "task"]].values.tolist()

    corr = []
    for out in OUTCOMES:
        sub = df.dropna(subset=[out])
        for pred in JBR_PREDICTORS + BASELINE_PREDICTORS:
            s = sub.dropna(subset=[pred])
            rho, p = perm_spearman(s[pred].values, s[out].values, s.dataset.values)
            c, dsc, ties, rate = concordance(s, pred, out)
            corr.append({"outcome": out, "predictor": pred, "n_tasks": len(s), "spearman_rho": rho,
                         "permutation_p": p, "concordant_pairs": c, "discordant_pairs": dsc, "tied_pairs": ties,
                         "concordance_rate": rate, "lodo_cv_r2": lodo_r2(s, pred, out),
                         "is_jbr": pred in JBR_PREDICTORS})
    corr = pd.DataFrame(corr)
    corr["p_holm"] = holm(corr.permutation_p.fillna(1).values)
    corr.to_csv(OUT / "correlations.csv", index=False)

    # within-database Kendall tau on the primary pair, with a within-database permutation test
    s = df.dropna(subset=list(PRIMARY))
    pairs = [(a, b) for (_, _), g in s.groupby(["dataset", "task_type"]) for a, b in combinations(g.index, 2)]
    def tau_of(x):
        dx = np.array([x[a] - x[b] for a, b in pairs])
        dy = np.array([s.loc[a, PRIMARY[1]] - s.loc[b, PRIMARY[1]] for a, b in pairs])
        keep = (np.abs(dx) > 1e-12) & (np.abs(dy) > 1e-12)
        return np.mean(np.sign(dx[keep]) == np.sign(dy[keep])), keep.sum()
    rate, n_nontied = tau_of(s[PRIMARY[0]])
    perm = []
    for _ in range(N_PERM):
        x = s[PRIMARY[0]].copy()
        for ds, g in s.groupby("dataset"):
            x.loc[g.index] = RNG.permutation(x.loc[g.index].values)
        perm.append(tau_of(x)[0])
    p_conc = (np.sum(np.array(perm) >= rate) + 1) / (N_PERM + 1)
    summary = {
        "foreign_keys": int(len(fk)), "databases": int(fk.dataset.nunique()),
        "fk_supercritical_d128": int((fk.JBR_d128 > 1).sum()), "fk_supercritical_eff_d128": int((fk.JBR_eff_d128 > 1).sum()),
        "fk_supercritical_d256": int((fk.JBR_d128 / 2 > 1).sum()),
        "tasks_with_published_results": int(len(df)), "tasks_missing_predictors": missing,
        "primary": {"predictor": PRIMARY[0], "outcome": PRIMARY[1],
                    "within_database_concordance": rate, "non_tied_pairs": int(n_nontied),
                    "permutation_p_one_sided": float(p_conc),
                    **corr[(corr.predictor == PRIMARY[0]) & (corr.outcome == PRIMARY[1])].iloc[0][
                        ["spearman_rho", "permutation_p", "p_holm", "lodo_cv_r2"]].to_dict()},
        "best_lodo_r2_by_outcome": {o: corr[corr.outcome == o].sort_values("lodo_cv_r2", ascending=False)
                                    .head(3)[["predictor", "lodo_cv_r2", "spearman_rho"]].round(3).to_dict("records")
                                    for o in OUTCOMES},
    }
    with open(OUT / "relbench_jbr_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, out in zip(axes, ["rdl_difficulty", "relgnn_gain", "relgt_gain"]):
        for tt, mk in zip(["binary_classification", "regression", "recommendation"], ["o", "s", "^"]):
            g = df[df.task_type == tt]
            ax.scatter(g[PRIMARY[0]], g[out], marker=mk, s=22, alpha=.8, label=tt.replace("_", " "))
        ax.set_xlabel(r"$\log_2(1 + \mathrm{JBR}_{\mathrm{eff}})$, d = 128")
        ax.set_ylabel(out.replace("_", " ") + " (percentile)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    for e in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"relbench_jbr_vs_outcomes.{e}", dpi=200)
    print(json.dumps(summary, indent=2, default=float)[:4000])


if __name__ == "__main__":
    main()
