#!/usr/bin/env python3
"""Configuration-level paired AUROC differences behind the mechanism claims: for every configuration where either
model is in its declining range (0.55 < AUROC < 0.95), the difference of configuration means; reported are
the mean difference with a bootstrap interval over configurations and the number of configurations where the
mechanism wins. Output: results/review_response_analysis/paired_configuration_differences.csv"""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(5)
sel = pd.read_csv(ROOT / "results/learning_rate_tuned_analysis/selected_runs.csv")
lv = pd.DataFrame([json.loads(l) for l in open(ROOT / "results/law_validation_and_depth/runs.jsonl")])
lv = lv[lv.experiment == "two_layer_depth"].copy()
lv["down"] = lv.model_options.apply(lambda o: o.get("downward"))
lv = lv.loc[lv.groupby(["scenario", "target", "variant", "down", "seed"]).val_auroc.idxmax()]
c = lambda g: g.groupby(["K", "D", "d"]).test_auroc.mean()  # noqa: E731
ext, gnn = sel[sel.family_group == "extended"], sel[sel.family_group == "gnn_architecture"]
cases = [("attention 1 head vs sum, extended", c(ext[ext.model == "SumAgg"]), c(ext[ext.model == "AttnAgg-h1"])),
         ("attention 4 heads vs sum, extended", c(ext[ext.model == "SumAgg"]), c(ext[ext.model == "AttnAgg-h4"]))]
for t in ["rare_pattern", "distributional"]:
    one = c(gnn[(gnn.model == "SumAgg") & (gnn.target == t)])
    cases.append((f"two-layer sum with message vs one-layer sum, {t}", one,
                  c(lv[(lv.target == t) & (lv.variant == "SumAgg") & (lv.down == True)])))  # noqa: E712
    cases.append((f"two-layer sum without message vs one-layer sum, {t}", one,
                  c(lv[(lv.target == t) & (lv.variant == "SumAgg") & (lv.down == False)])))  # noqa: E712
rows = []
for name, ref, new in cases:
    j = pd.concat([ref.rename("ref"), new.rename("new")], axis=1, join="inner")
    j = j[((j.ref > 0.55) & (j.ref < 0.95)) | ((j.new > 0.55) & (j.new < 0.95))]
    d = (j.new - j.ref).values
    b = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    rows.append({"comparison": name, "configurations_in_declining_range": len(d), "mean_auroc_difference": d.mean(),
                 "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))], "mechanism_wins": int((d > 0).sum())})
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results/review_response_analysis/paired_configuration_differences.csv", index=False)
print(out.round(3).to_string())
