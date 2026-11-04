"""
Scenario grids for the synthetic Join Bottleneck Ratio (JBR) experiments.

A scenario is one parent-child foreign-key join with child cardinality K (children per parent), child
feature dimensionality D (binary features, 1 bit of entropy each) and parent embedding dimension d.
The nominal information demand at the join is K * D bits and the nominal JBR is

    JBR = K * H_row / d,   H_row = sum of per-feature conditional entropies of a child row (bits)

which equals K * D / d for uniform binary features. The empirical JBR used in every analysis is computed
from the sampled data (plug-in entropy of each child feature given the parent).
"""

from dataclasses import dataclass
from itertools import product
from math import log2

K_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512]
D_GRID = [2, 4, 8, 16, 32]
D_EMBED_BASE = 128


@dataclass(frozen=True)
class Scenario:
    K: int
    D: int
    d: int = D_EMBED_BASE

    @property
    def nominal_jbr(self):
        return self.K * self.D / self.d

    @property
    def name(self):
        return f"K{self.K}_D{self.D}_d{self.d}"


def _pick_spread(candidates, n):
    """Pick n distinct candidates spread across the list (deterministic)."""
    if n >= len(candidates):
        return list(candidates)
    if n == 1:
        return [candidates[len(candidates) // 2]]
    idx = sorted({round(i * (len(candidates) - 1) / (n - 1)) for i in range(n)})
    extra = [i for i in range(len(candidates)) if i not in idx]
    while len(idx) < n:
        idx.append(extra.pop(len(extra) // 2))
    return [candidates[i] for i in sorted(idx)]


def _balanced(pool, total, fixed_quota=None):
    """Distribute `total` scenarios as evenly as possible over log2-JBR levels."""
    by_level = {}
    for sc in pool:
        by_level.setdefault(round(log2(sc.nominal_jbr)), []).append(sc)
    quota = {lvl: 0 for lvl in by_level}
    quota.update(fixed_quota or {})
    remaining = total - sum(quota.values())
    while remaining > 0:
        open_levels = [l for l in sorted(by_level) if quota[l] < len(by_level[l]) and l not in (fixed_quota or {})]
        if not open_levels:
            break
        lvl = min(open_levels, key=lambda l: (quota[l], abs(l)))
        quota[lvl] += 1
        remaining -= 1
    chosen = []
    for lvl in sorted(by_level):
        chosen += _pick_spread(sorted(by_level[lvl], key=lambda sc: (sc.d, sc.K)), quota[lvl])
    return chosen


def base_scenarios():
    """25 scenarios at d = 128 spanning nominal JBR 1/32 to 16, with five decompositions at JBR = 1."""
    pool = [Scenario(K, D) for K, D in product(K_GRID, D_GRID) if 1 / 32 <= K * D / D_EMBED_BASE <= 16]
    chosen = _balanced(pool, 25, fixed_quota={0: 5})  # five iso-JBR decompositions at JBR = 1
    assert len(chosen) == 25, len(chosen)
    return sorted(chosen, key=lambda sc: (sc.nominal_jbr, sc.K))


def iso_jbr_scenarios():
    return [s for s in base_scenarios() if s.nominal_jbr == 1]


def architecture_scenarios():
    """11 scenarios from the base grid covering the full JBR range, for the GNN architecture comparison."""
    base = base_scenarios()
    step = (len(base) - 1) / 10
    return [base[round(i * step)] for i in range(11)]


def extreme_scenarios():
    """38 scenarios with nominal JBR 32 to 4096, reached by reducing the embedding dimension d."""
    pool = [Scenario(K, D, d) for K, D, d in product(K_GRID, D_GRID, [4, 8, 16, 32, 64, 128]) if 32 <= K * D / d <= 4096]
    chosen = _balanced(pool, 38)
    assert len(chosen) == 38, len(chosen)
    return sorted(chosen, key=lambda sc: (sc.nominal_jbr, sc.d, sc.K))


def head_sweep_scenarios():
    """5 high-JBR base scenarios for the attention head-count sweep."""
    return sorted(base_scenarios(), key=lambda s: -s.nominal_jbr)[:5]


TARGETS = ["mean_based", "rare_pattern", "distributional"]
DEEPSETS_AGGREGATORS = ["sum", "mean", "max"]
GNN_ARCHITECTURES = ["SumAgg", "MeanAgg", "AttnAgg"]
HEAD_COUNTS = [1, 4, 8, 16]
SEEDS = [0, 1, 2]
