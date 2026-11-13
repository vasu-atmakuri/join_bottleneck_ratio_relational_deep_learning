#!/usr/bin/env python3
"""Paired bootstrap over configurations (the same resample for both models) of the midpoint shift behind each
capacity multiplier: attention vs sum on the extended range, two-layer vs one-layer sum (with and without the
parent-to-child message). Output: results/review_response_analysis/paired_midpoint_shifts.csv"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_synthetic_phase_transition import fit_sigmoid  # noqa: E402
rng = np.random.default_rng(11)
sel = pd.read_csv(ROOT / "results/learning_rate_tuned_analysis/selected_runs.csv")
lv = pd.DataFrame([json.loads(l) for l in open(ROOT / "results/law_validation_and_depth/runs.jsonl")])
lv = lv[lv.experiment == "two_layer_depth"].copy()
lv["down"] = lv.model_options.apply(lambda o: o.get("downward"))
lv = lv.loc[lv.groupby(["scenario", "target", "variant", "down", "seed"]).val_auroc.idxmax()]
lv["x"] = np.log2(lv.jbr_empirical)
def cfg(g, x="log2_jbr"):
    return g.groupby(["K", "D", "d"]).agg(auroc=("test_auroc", "mean"), x=(x, "mean"))
pairs = []
ext = sel[sel.family_group == "extended"]
base = cfg(ext[ext.model == "SumAgg"])
for m in ["AttnAgg-h1", "AttnAgg-h4"]:
    pairs.append(("extended rare_pattern", m, "SumAgg", base, cfg(ext[ext.model == m])))
gnn = sel[sel.family_group == "gnn_architecture"]
for t in ["rare_pattern", "distributional"]:
    one = cfg(gnn[(gnn.model == "SumAgg") & (gnn.target == t)])
    for lab, v, d in (("two-layer sum, message", "SumAgg", True), ("two-layer sum, no message", "SumAgg", False)):
        two = cfg(lv[(lv.target == t) & (lv.variant == v) & (lv.down == d)], "x")
        pairs.append((f"architecture subset {t}", lab, "one-layer sum", one, two))
rows = []
for name, m, ref, a, b in pairs:
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner").reset_index()
    def shift(d):
        fa = fit_sigmoid(d.x_a, d.auroc_a); fb = fit_sigmoid(d.x_b, d.auroc_b)
        return fb["midpoint_log2_jbr"] - fa["midpoint_log2_jbr"]
    s0 = shift(j)
    bs = [shift(j.iloc[rng.integers(0, len(j), len(j))]) for _ in range(300)]
    lo, hi = np.quantile(bs, [.025, .975])
    rows.append({"comparison": name, "model": m, "reference": ref, "n_configurations": len(j), "midpoint_shift": s0,
                 "multiplier": 2 ** s0, "multiplier_ci95_over_configurations": [2 ** lo, 2 ** hi],
                 "share_shift_positive": float(np.mean(np.array(bs) > 0))})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/review_response_analysis/paired_midpoint_shifts.csv", index=False)
print(out.round(3).to_string())
