#!/usr/bin/env python3
"""
Publication figures for the manuscript, built only from result files produced by the experiment and analysis scripts.

Inputs
  results/learning_rate_tuned_analysis/selected_runs.csv        (analyze_learning_rate_tuned_results.py)
  results/learning_rate_tuned_analysis/capacity_multiplier.csv
  results/capacity_law_controls/control_training_runs.jsonl     (run_capacity_law_control_experiments.py)
  results/relbench_jbr_analysis/foreign_key_jbr_distribution.csv (analyze_relbench_jbr_task_difficulty.py)
  results/relbench_interventions/intervention_runs.jsonl        (optional; train_relbench_aggregation_interventions.py)
Outputs
  paper/figures/{phase_transition,capacity_multiplier,width_collapse,relbench_jbr_distribution,controls}.pdf
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_synthetic_phase_transition import fit_sigmoid, sigmoid  # noqa: E402
from fit_generalized_join_bottleneck_ratio import fit as fit_free, predictor  # noqa: E402

RES = ROOT / "results"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper" / "figures"
COL = {"sum": "#1f77b4", "mean": "#ff7f0e", "SumAgg": "#1f77b4", "MeanAgg": "#ff7f0e", "AttnAgg-h1": "#2ca02c",
       "AttnAgg-h4": "#9467bd", "max": "#7f7f7f"}
LABEL = {"sum": "DeepSets sum", "mean": "DeepSets mean", "max": "DeepSets max", "SumAgg": "GNN sum",
         "MeanAgg": "GNN mean", "AttnAgg-h1": "GNN attention (1 head)", "AttnAgg-h4": "GNN attention (4 heads)"}
ONE_COL, TWO_COL = 3.5, 7.16

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "lines.linewidth": 1.0})


def scenario_means(df):
    return df.groupby("scenario").agg(x=("log2_jbr", "mean"), y=("test_auroc", "mean"), sd=("test_auroc", "std"),
                                      K=("K", "first"), D=("D", "first"), d=("d", "first"),
                                      H=("row_entropy_bits", "mean")).reset_index().sort_values("x")


def draw_fit(ax, s, model, marker="o"):
    ax.errorbar(s.x, s.y, yerr=s.sd, fmt=marker, ms=2.8, capsize=1.5, elinewidth=.6, color=COL[model], alpha=.85,
                label=LABEL[model])
    f = fit_sigmoid(s.x, s.y)
    if f:
        xs = np.linspace(s.x.min(), s.x.max(), 300)
        ax.plot(xs, sigmoid(xs, f["floor"], f["ceiling"], f["slope"], f["midpoint_log2_jbr"]), color=COL[model])
        ax.axvline(f["midpoint_log2_jbr"], color=COL[model], ls=":", lw=.7)
    return f


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def phase_transition(sel):
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL, 2.1), sharey=True)
    panels = [("main_deepsets", "rare_pattern", ["sum", "mean", "max"], "(a) DeepSets, rare pattern"),
              ("main_deepsets", "distributional", ["sum", "mean", "max"], "(b) DeepSets, distributional"),
              ("gnn_architecture", "rare_pattern", ["SumAgg", "MeanAgg", "AttnAgg-h4"], "(c) GNN, rare pattern")]
    for ax, (fam, tgt, models, title) in zip(axes, panels):
        for m, mk in zip(models, ["o", "s", "^"]):
            g = sel[(sel.family_group == fam) & (sel.target == tgt) & (sel.model == m)]
            if m == "max":
                s = scenario_means(g)
                ax.plot(s.x, s.y, "^", ms=2.5, color=COL[m], alpha=.6, label=LABEL[m])
            else:
                draw_fit(ax, scenario_means(g), m, mk)
        ax.axvline(0, color="k", lw=.5, ls="--")
        ax.set_title(title)
        ax.set_xlabel(r"$\log_2$ JBR")
        ax.set_ylim(0.45, 1.02)
    axes[0].set_ylabel("test AUROC")
    for ax in axes:
        ax.legend(loc="lower left", frameon=False)
    fig.tight_layout(w_pad=0.6)
    save(fig, "phase_transition")


def capacity_multiplier(sel, mult):
    ext = sel[sel.family_group == "extended"]
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.2))
    for m, mk in zip(["SumAgg", "AttnAgg-h1", "AttnAgg-h4"], ["o", "^", "D"]):
        draw_fit(ax, scenario_means(ext[ext.model == m]), m, mk)
    txt = "\n".join(f"{LABEL[r.model]}: {r.multiplier:.2f}x" for r in mult.itertuples())
    ax.text(0.98, 0.70, "capacity multiplier vs GNN sum\n" + txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=6, bbox=dict(fc="white", ec="0.7", lw=.4, pad=2))
    ax.axvline(0, color="k", lw=.5, ls="--")
    ax.set_xlabel(r"$\log_2$ JBR")
    ax.set_ylabel("test AUROC")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, "capacity_multiplier")


def width_collapse(sel):
    ws = sel[sel.family_group == "width_sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(ONE_COL, 1.95), sharey=True)
    g = ws[ws.target == "rare_pattern"].groupby(["K", "D", "d"]).agg(
        auroc=("test_auroc", "mean"), row_entropy_bits=("row_entropy_bits", "mean")).reset_index()
    cmap = plt.get_cmap("viridis")
    keys = sorted({(k, dd) for k, dd in zip(g.K, g.D)})
    p = fit_free(g, "free", g.auroc.values)
    x_gen = predictor(p[4:], g, "free")
    for i, (K, D) in enumerate(keys):
        m = (g.K == K) & (g.D == D)
        c = cmap(i / max(1, len(keys) - 1))
        axes[0].plot(np.log2(g[m].d), g[m].auroc, "o-", ms=2.2, color=c, label=f"K={K}, D={D}")
        axes[1].plot(x_gen[m.values], g[m].auroc, "o", ms=2.2, color=c)
    xs = np.linspace(x_gen.min(), x_gen.max(), 200)
    axes[1].plot(xs, sigmoid(xs, *p[:4]), "k-", lw=.8)
    axes[0].set_xlabel(r"$\log_2 d$")
    axes[1].set_xlabel(r"$\log_2 K + %.2f\,\log_2 H - %.2f\,\log_2 d$" % (p[4], p[5]), fontsize=6.5)
    axes[0].set_ylabel("test AUROC")
    axes[0].set_title("(a) raw width sweep")
    axes[1].set_title("(b) fitted collapse")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=5.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(w_pad=0.4)
    save(fig, "width_collapse")


def relbench_distribution(fk):
    order = fk.groupby("dataset").JBR_eff_d128.median().sort_values().index
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
    rng = np.random.default_rng(0)
    for i, ds in enumerate(order):
        g = fk[fk.dataset == ds]
        jit = rng.uniform(-.15, .15, len(g))
        ax.plot(np.log2(g.JBR_d128.clip(lower=2 ** -12)), i + jit - .18, "o", ms=2.3, color="#1f77b4", alpha=.75,
                label="JBR" if i == 0 else None)
        ax.plot(np.log2(g.JBR_eff_d128.clip(lower=2 ** -12)), i + jit + .18, "s", ms=2.3, color="#d62728",
                alpha=.75, label=r"JBR$_{\mathrm{eff}}$" if i == 0 else None)
    ax.axvspan(0.47, 1.37, color="gray", alpha=.18, lw=0, label="measured midpoints (sum, mean)")
    ax.axvline(3.44, color="#2ca02c", ls=":", lw=.9, label="attention midpoint")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel(r"$\log_2$ JBR at $d=128$ (one point per foreign key; zero-entropy joins at $-12$)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=5.5)
    fig.tight_layout()
    save(fig, "relbench_jbr_distribution")


def controls(ctrl):
    ctrl = ctrl[ctrl.status == "ok"].copy()
    fig, axes = plt.subplots(1, 4, figsize=(TWO_COL, 1.9), sharey=True)

    def lines(ax, sub, xcol, title, xlabel, by=("K", "D"), variant=None, xticks=None):
        if variant is not None:
            sub = sub[sub.variant == variant]
        cmap = plt.get_cmap("viridis")
        grp = sub.groupby(list(by))
        n = len(grp)
        for i, (key, g) in enumerate(grp):
            s = g.groupby(xcol).test_auroc.agg(["mean", "std"]).reset_index()
            c = cmap(i / max(1, n - 1))
            ax.errorbar(s[xcol].astype(float), s["mean"], yerr=s["std"], fmt="o-", ms=2, capsize=1, elinewidth=.5,
                        color=c)
            if s["mean"].iloc[0] > 0.56:
                ax.text(s[xcol].astype(float).iloc[0], s["mean"].iloc[0], f"K={key[0]},D={key[1]} ", fontsize=4.6,
                        color=c, ha="right", va="center")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if xticks is not None:
            ax.set_xticks(xticks)

    irr = ctrl[ctrl.experiment == "irrelevant_features"]
    lines(axes[0], irr, "opt_n_irrelevant", "(a) irrelevant features, sum", "added irrelevant columns",
          variant="sum", xticks=[0, 8, 24])
    mis = ctrl[ctrl.experiment == "missing_values"]
    lines(axes[1], mis, "opt_missing_rate", "(b) missing values, sum", "missing rate", xticks=[0, .1, .3])
    bud = ctrl[ctrl.experiment == "training_budget"]
    lines(axes[2], bud, "budget", "(c) training budget, sum", "budget multiplier", xticks=[.5, 1, 2, 4])
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks([.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    sh = ctrl[ctrl.experiment == "distribution_shift"]
    pivot = sh.groupby(["variant", "K", "D", "shift"]).test_auroc.mean().unstack("shift").reset_index()
    pivot["KD"] = pivot.K * pivot.D
    configs = pivot[pivot.variant == "sum"].sort_values(["KD", "K"])[["K", "D"]].values.tolist()
    xpos = {tuple(c): i for i, c in enumerate(configs)}
    for v, mk in (("sum", "o"), ("AttnAgg", "^")):
        p = pivot[pivot.variant == v].copy()
        p["x"] = [xpos[(k, dd)] for k, dd in zip(p.K, p.D)]
        p = p.sort_values("x")
        for sft, ls in (("none", "-"), ("cardinality_x2", "--")):
            axes[3].plot(p.x, p[sft], ls, marker=mk, ms=2.2, color="#1f77b4" if v == "sum" else "#2ca02c",
                         label=f"{'sum' if v == 'sum' else 'attention'}, {'no shift' if sft == 'none' else 'K x2 at test'}")
    axes[3].set_xticks(range(len(configs)), [f"{k},{dd}" for k, dd in configs], rotation=45, fontsize=5.5)
    axes[3].set_title("(d) test-time cardinality shift")
    axes[3].set_xlabel("training (K, D), d = 128")
    axes[0].set_ylabel("test AUROC")
    for ax in axes[:2]:
        ax.set_xlim(left=ax.get_xlim()[0] - 0.4 * (ax.get_xlim()[1] - ax.get_xlim()[0]))
    axes[2].set_xlim(0.11, 4.6)
    axes[3].legend(frameon=False, fontsize=4.8, loc="lower left")
    fig.tight_layout(w_pad=0.4)
    save(fig, "controls")


def interventions(path):
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.open() if '"ok"' in line]
    if not rows:
        return
    df = pd.DataFrame(rows)
    cls = df[df.metric == "roc_auc"]
    if cls.empty:
        return
    piv = cls.groupby(["dataset", "task", "width", "aggregation"]).test_metric.mean().unstack("aggregation")
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.2))
    for (ds, task), g in piv.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        if "attention" in g and "sum" in g:
            ax.plot(np.log2(g.index), 100 * (g["attention"] - g["sum"]), "o-", ms=2.4, label=f"{ds}/{task}")
    ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel(r"$\log_2 d$")
    ax.set_ylabel("attention minus sum (AUROC points)")
    ax.legend(frameon=False, fontsize=5)
    fig.tight_layout()
    save(fig, "relbench_interventions")


def planted_by_children(path):
    """Parent-query label on real tables: test AUROC within bins of admissible children K (joins with at least 1000
    test seeds on which the label is learnable at small K; see the manuscript for the excluded joins)."""
    if not path.exists():
        return
    b = pd.read_csv(path)
    b = b[(b.label == "parent_query") & (b.width == 128)]
    joins = [("rel-stack/badges", "(a) stack: users <- badges"), ("rel-avito/SearchInfo", "(b) avito: users <- searches")]
    col = {"sum": "#1f77b4", "mean": "#ff7f0e", "attention_root": "#2ca02c", "sum_attention": "#9467bd"}
    name = {"sum": "sum", "mean": "mean", "attention_root": "attention", "sum_attention": "sum + attention"}
    fig, axes = plt.subplots(1, 2, figsize=(ONE_COL, 1.95), sharey=True)
    for ax, (j, title) in zip(axes, joins):
        g = b[b["join"] == j]
        for (agg, layers), h in g.groupby(["aggregation", "layers"]):
            h = h.sort_values("k_mid")
            ax.plot(np.log2(h.k_mid), h.auroc, marker="o" if layers == 1 else "s", ms=2.3,
                    ls="-" if layers == 1 else "--", color=col[agg], lw=.8,
                    label=f"{name[agg]}, {layers} layer{'s' if layers == 2 else ''}")
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(r"$\log_2 K$")
    axes[0].set_ylabel("test AUROC, query label")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=5.3, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(w_pad=0.4)
    save(fig, "planted_signal_by_children")


def main():
    sel = pd.read_csv(RES / "learning_rate_tuned_analysis" / "selected_runs.csv")
    mult = pd.read_csv(RES / "learning_rate_tuned_analysis" / "capacity_multiplier.csv")
    ctrl = pd.DataFrame([json.loads(line) for line in open(RES / "capacity_law_controls" / "control_training_runs.jsonl")])
    fk = pd.read_csv(RES / "relbench_jbr_analysis" / "foreign_key_jbr_distribution.csv")
    phase_transition(sel)
    capacity_multiplier(sel, mult)
    relbench_distribution(fk)
    controls(ctrl)
    planted_by_children(RES / "relbench_planted_signal" / "analysis" / "planted_signal_by_children.csv")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
