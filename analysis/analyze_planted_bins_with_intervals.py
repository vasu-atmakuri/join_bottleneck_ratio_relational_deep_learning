#!/usr/bin/env python3
"""Planted query labels: per child-count bin, test entities, positive rate and AUROC (mean over three seeds) with a
bootstrap interval over test entities (200 resamples, the same entity resample applied to every seed).
Output: results/review_response_analysis/planted_bins_with_intervals.csv"""
import re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results" / "relbench_planted_signal" / "predictions"
BINS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 64)]
rng = np.random.default_rng(7)
groups = {}
for f in sorted(P.glob("planted_signal__*.npz")):
    _, ds, task, ent, label, agg, layers, width, seed = f.stem.split("__")
    if width != "128" or ds not in ("rel-stack", "rel-avito"):
        continue
    groups.setdefault((ds, ent, label, agg, int(layers)), []).append(np.load(f))
rows = []
for (ds, ent, label, agg, L), zs in groups.items():
    ch = zs[0]["children"]; y = zs[0]["label"].astype(int)
    for lo, hi in BINS:
        m = (ch >= lo) & (ch <= hi)
        idx = np.nonzero(m)[0]
        if len(idx) < 20 or y[idx].min() == y[idx].max():
            continue
        point = np.mean([roc_auc_score(y[idx], z["pred"][idx]) for z in zs])
        boots = []
        for _ in range(200):
            b = rng.choice(idx, len(idx))
            if y[b].min() == y[b].max():
                continue
            boots.append(np.mean([roc_auc_score(y[b], z["pred"][b]) for z in zs]))
        rows.append({"dataset": ds, "entity_join": ent, "label": label, "aggregation": agg, "layers": L,
                     "k_bin": f"{lo}-{hi}", "test_entities": len(idx), "positive_rate": y[idx].mean(),
                     "auroc_mean_3_seeds": point, "ci95_low": np.quantile(boots, .025),
                     "ci95_high": np.quantile(boots, .975), "n_seeds": len(zs)})
out = pd.DataFrame(rows)
(ROOT / "results" / "review_response_analysis").mkdir(parents=True, exist_ok=True)
out.to_csv(ROOT / "results" / "review_response_analysis" / "planted_bins_with_intervals.csv", index=False)
print(out[out.label == "parent_query"].round(3).to_string())
