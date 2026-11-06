#!/usr/bin/env python3
"""
Tests whether task-conditional JBR predicts RelBench task outcomes where task-agnostic JBR does not.

Per task (21 entity classification and regression tasks), from the one-hop joins into the entity table:
  tc_jbr_max                 max over joins of log2(1 + JBR_eff * relevance)
  tc_jbr_relevance_weighted  log2(1 + relevance-weighted mean of JBR_eff)
  relevant_bits_at_risk      log2(1 + sum over joins of relevance * max(0, JBR_eff - 1))
  relevance_of_widest_join   relevance of the join with the largest JBR_eff
  total_join_relevance       sum of relevance over joins
compared with the task-agnostic predictors of analyze_relbench_jbr_task_difficulty.py on the same tasks, with the same
outcomes and statistics (Spearman with within-database permutation p-values, within-database concordance,
leave-one-database-out R^2, Holm correction).

Inputs: results/relbench_task_conditional/join_task_relevance.csv,
        results/relbench_jbr_analysis/{foreign_key_jbr_distribution,task_predictors_and_outcomes}.csv
Outputs: results/relbench_task_conditional/{task_conditional_predictors,task_conditional_correlations}.csv,
         task_conditional_summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_relbench_jbr_task_difficulty import concordance, holm, lodo_r2, perm_spearman  # noqa: E402

TC = ROOT / "results" / "relbench_task_conditional"
JA = ROOT / "results" / "relbench_jbr_analysis"
OUTCOMES = ["rdl_difficulty", "relational_gain", "relgnn_gain", "relgt_gain"]
TC_PRED = ["tc_jbr_max", "tc_jbr_relevance_weighted", "relevant_bits_at_risk", "relevance_of_widest_join",
           "total_join_relevance"]
AGNOSTIC = ["jbr_eff_max_one_hop", "jbr_eff_entropy_weighted", "max_join_depth"]


def main():
    rel = pd.read_csv(TC / "join_task_relevance.csv")
    fk = pd.read_csv(JA / "foreign_key_jbr_distribution.csv")
    tasks = pd.read_csv(JA / "task_predictors_and_outcomes.csv")
    tasks = tasks[tasks.task_type != "recommendation"]
    j = rel.merge(fk[["dataset", "child_table", "fk_column", "JBR_eff_d128"]],
                  on=["dataset", "child_table", "fk_column"], how="left")
    rows = []
    for (ds, task), g in j.groupby(["dataset", "task"]):
        r = g.relevance.fillna(0).values
        x = g.JBR_eff_d128.fillna(0).values
        rows.append({"dataset": ds, "task": task,
                     "tc_jbr_max": np.log2(1 + (x * r).max()),
                     "tc_jbr_relevance_weighted": np.log2(1 + (x * r).sum() / r.sum()) if r.sum() > 0 else 0.0,
                     "relevant_bits_at_risk": np.log2(1 + (r * np.maximum(0, x - 1)).sum()),
                     "relevance_of_widest_join": r[np.argmax(x)],
                     "total_join_relevance": r.sum()})
    pred = pd.DataFrame(rows)
    df = tasks.merge(pred, on=["dataset", "task"], how="left")
    df.to_csv(TC / "task_conditional_predictors.csv", index=False)
    out = []
    for o in OUTCOMES:
        sub = df.dropna(subset=[o])
        for p in TC_PRED + AGNOSTIC:
            s = sub.dropna(subset=[p])
            if len(s) < 6:
                continue
            rho, pval = perm_spearman(s[p].values, s[o].values, s.dataset.values)
            c, dsc, ties, rate = concordance(s, p, o)
            out.append({"outcome": o, "predictor": p, "task_conditional": p in TC_PRED, "n_tasks": len(s),
                        "spearman_rho": rho, "permutation_p": pval, "concordant_pairs": c, "discordant_pairs": dsc,
                        "tied_pairs": ties, "concordance_rate": rate, "lodo_cv_r2": lodo_r2(s, p, o)})
    corr = pd.DataFrame(out)
    corr["p_holm"] = holm(corr.permutation_p.fillna(1).values)
    corr.to_csv(TC / "task_conditional_correlations.csv", index=False)
    summary = {"tasks": int(df.tc_jbr_max.notna().sum()),
               "joins_scored": int(len(rel)),
               "best_by_outcome": {o: corr[corr.outcome == o].sort_values("lodo_cv_r2", ascending=False).head(4)
                                   [["predictor", "spearman_rho", "permutation_p", "p_holm", "lodo_cv_r2",
                                     "concordant_pairs", "discordant_pairs"]].round(3).to_dict("records")
                                   for o in OUTCOMES}}
    json.dump(summary, open(TC / "task_conditional_summary.json", "w"), indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
