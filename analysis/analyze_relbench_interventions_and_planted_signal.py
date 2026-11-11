#!/usr/bin/env python3
"""
Analysis of the controlled RelBench experiments.

Real tasks (train_relbench_aggregation_interventions.py and the real_tasks_attention_variants family of
train_relbench_planted_signal_and_aggregation_variants.py): test metric per task, aggregation and width (mean over
seeds); paired differences against sum aggregation at equal task, width and seed (AUROC points for classification,
NMAE for regression, signed so that positive means better than sum), with bootstrap intervals over tasks.

Planted signal: test AUROC per join, label, layers, aggregation and width; AUROC within bins of the number of
admissible children K of each test seed (from the saved predictions), which traces accuracy against K on real tables.

Outputs (results/relbench_planted_signal/analysis/): real_task_table.csv, real_task_differences_vs_sum.csv,
planted_signal_table.csv, planted_signal_by_children.csv, summary.json, figures/*.pdf|png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
INT = ROOT / "results" / "relbench_interventions" / "intervention_runs.jsonl"
PS = ROOT / "results" / "relbench_planted_signal"
OUT = PS / "analysis"
RNG = np.random.default_rng(20261110)
K_BINS = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 64)]


def read(path):
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.open() if line.strip()]
    df = pd.DataFrame(rows)
    return df[df.status == "ok"].copy() if len(df) else df


def real_tasks():
    a = read(INT)
    a["layers"] = 2
    b = read(PS / "runs.jsonl")
    if len(b):
        b = b[b.family == "real_tasks_attention_variants"]
    df = pd.concat([a, b], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset", "task", "aggregation", "width", "seed"], keep="last")
    table = df.groupby(["dataset", "task", "metric", "width", "aggregation"]).test_metric.agg(["mean", "std", "count"]) \
        .reset_index()
    table.to_csv(OUT / "real_task_table.csv", index=False)
    base = df[df.aggregation == "sum"][["dataset", "task", "width", "seed", "test_metric"]] \
        .rename(columns={"test_metric": "sum_metric"})
    m = df[df.aggregation != "sum"].merge(base, on=["dataset", "task", "width", "seed"])
    sign = np.where(m.metric == "roc_auc", 100.0, -1.0)  # AUROC points; NMAE lower is better
    m["gain_vs_sum"] = sign * (m.test_metric - m.sum_metric)
    per_task = m.groupby(["aggregation", "metric", "dataset", "task", "width"]).gain_vs_sum.mean().reset_index()
    rows = []
    for (agg, metric), g in per_task.groupby(["aggregation", "metric"]):
        t = g.groupby(["dataset", "task"]).gain_vs_sum.mean()
        boots = [t.sample(frac=1, replace=True, random_state=int(RNG.integers(1e9))).mean() for _ in range(2000)]
        rows.append({"aggregation": agg, "metric": metric, "n_tasks": len(t), "mean_gain_vs_sum": t.mean(),
                     "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                     "tasks_better_than_sum": int((t > 0).sum())})
    diffs = pd.DataFrame(rows)
    per_task.to_csv(OUT / "real_task_differences_vs_sum.csv", index=False)
    return table, diffs


def planted():
    df = read(PS / "runs.jsonl")
    if not len(df):
        return pd.DataFrame(), pd.DataFrame()
    df = df[df.family == "planted_signal"].copy()
    df["join"] = df.dataset + "/" + df.task.str.split("|").str[1]
    table = df.groupby(["join", "label", "layers", "width", "aggregation"]).test_metric \
        .agg(["mean", "std", "count"]).reset_index()
    table.to_csv(OUT / "planted_signal_table.csv", index=False)
    rows = []
    for r in df.itertuples():
        f = PS / "predictions" / ("__".join(str(getattr(r, k)) for k in
                                           ("family", "dataset", "task", "label", "aggregation", "layers", "width",
                                            "seed")).replace("|", "__") + ".npz")
        if not f.exists():
            continue
        z = np.load(f)
        for lo, hi in K_BINS:
            msk = (z["children"] >= lo) & (z["children"] <= hi)
            y = z["label"][msk]
            if msk.sum() >= 30 and 0 < y.mean() < 1:
                rows.append({"join": r.join, "test_seeds": len(z["label"]), "label": r.label, "layers": r.layers, "width": r.width,
                             "aggregation": r.aggregation, "seed": r.seed, "k_bin": f"{lo}-{hi}", "k_mid": (lo + hi) / 2,
                             "n": int(msk.sum()), "auroc": roc_auc_score(y, z["pred"][msk]),
                             "planted_values": getattr(r, "planted_n_values", np.nan)})
    byk = pd.DataFrame(rows)
    if len(byk):
        byk = byk.groupby(["join", "label", "layers", "width", "aggregation", "k_bin", "k_mid"]) \
            .agg(auroc=("auroc", "mean"), n=("n", "mean"), test_seeds=("test_seeds", "first"),
                 planted_values=("planted_values", "first")).reset_index()
        byk["log2_capped_jbr"] = np.log2(byk.k_mid * np.log2(byk.planted_values.astype(float)) / byk.width)
    byk.to_csv(OUT / "planted_signal_by_children.csv", index=False)
    return table, byk


def inclusion(byk):
    """Applies the inclusion rule fixed in results/relbench_planted_signal/inclusion_rule.json to every planted join."""
    import re
    rule = json.load(open(PS / "inclusion_rule.json"))["criteria"]
    rates, cur = {}, None
    for line in (PS / "progress.log").open():
        m = re.search(r"(rel-[a-z0-9]+): join (\w+)\.\w+ -> \w+", line)
        if m:
            cur = f"{m.group(1)}/{m.group(2)}"
            continue
        m = re.search(r"(train|test): (\d+) seeds.*positive rate parent_query ([0-9.]+)", line)
        if m and cur and (cur, m.group(1)) not in rates:
            rates[(cur, m.group(1))] = (int(m.group(2)), float(m.group(3)))
    rows = []
    for j in sorted(byk["join"].unique()):
        small = byk[(byk["join"] == j) & (byk.label == "parent_query") & (byk.width == 128) & (byk.k_bin == "1-4")]
        best_small = float(small.auroc.max()) if len(small) else float("nan")
        test_n, test_rate = rates.get((j, "test"), (0, float("nan")))
        _, train_rate = rates.get((j, "train"), (0, float("nan")))
        r = {"join": j, "test_seeds": test_n, "train_positive_rate": train_rate, "test_positive_rate": test_rate,
             "best_auroc_k_at_most_4": best_small}
        r["included"] = bool(test_n >= rule["minimum_test_seeds"]
                             and abs(train_rate - test_rate)
                             <= rule["maximum_abs_difference_in_parent_query_positive_rate_between_train_and_test"]
                             and best_small >= 0.90)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "planted_join_inclusion.csv", index=False)
    return out


def figures(byk):
    if not len(byk):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    dss = sorted(inclusion(byk).query("included")["join"])
    fig, axes = plt.subplots(1, len(dss), figsize=(7.16, 1.9), sharey=True, squeeze=False)
    col = {"sum": "#1f77b4", "mean": "#ff7f0e", "attention_root": "#2ca02c", "sum_attention": "#9467bd"}
    for ax, ds in zip(axes[0], dss):
        g = byk[(byk["join"] == ds) & (byk.label == "parent_query") & (byk.width == 128)]
        for (agg, layers), h in g.groupby(["aggregation", "layers"]):
            h = h.sort_values("k_mid")
            ax.plot(np.log2(h.k_mid), h.auroc, marker="o" if layers == 1 else "s", ms=2.5,
                    ls="-" if layers == 1 else "--", color=col[agg], label=f"{agg.replace('_root', '')}, {layers} layer")
        ax.set_title(ds)
        ax.set_xlabel(r"$\log_2$ admissible children $K$")
    axes[0][0].set_ylabel("test AUROC, parent-query label")
    axes[0][-1].legend(frameon=False, fontsize=5, loc="lower left")
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / "figures" / f"planted_signal_by_children.{e}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, diffs = real_tasks()
    ptable, byk = planted()
    figures(byk)
    summary = {"real_task_gain_vs_sum": diffs.round(3).to_dict("records")}
    if len(ptable):
        summary["planted_signal"] = ptable[ptable.width == 128].pivot_table(
            index=["join", "label", "layers"], columns="aggregation", values="mean").round(3).reset_index() \
            .to_dict("records")
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
