"""
Controlled variants of the synthetic parent-child data, used to test what the Join Bottleneck Ratio does and does not
capture. Every variant keeps the rare_pattern / distributional / mean_based targets of data_generation.py and adds:

  bit_prob        child bits are Bernoulli(bit_prob) instead of uniform, lowering the entropy per feature to h(bit_prob)
                  while keeping the feature count D fixed (separates entropy H from dimensionality D)
  n_irrelevant    extra child features drawn uniformly that the target does not use (raises task-agnostic entropy
                  without changing the task-relevant information)
  missing_rate    each child feature value is independently missing (encoded 0 in the -1/+1 encoding)
  k_distribution  "fixed" (exactly K children), "poisson" or "lognormal" (sigma 1) child counts with mean K, at least
                  2 children, at most 4K; padded children are masked out of every aggregator
  k_scale         multiplies the child count (used for test-time cardinality shift)

Children are returned as int8 in {-1, 0, +1} with a boolean mask [n, K_max].
"""

import math

import numpy as np

DEFAULTS = {"bit_prob": 0.5, "n_irrelevant": 0, "missing_rate": 0.0, "k_distribution": "fixed", "k_scale": 1.0}


def options(opts=None):
    o = dict(DEFAULTS)
    o.update(opts or {})
    return o


def k_max(K, opts):
    o = options(opts)
    k = int(round(K * o["k_scale"]))
    return k if o["k_distribution"] == "fixed" else 4 * k


def _child_counts(rng, n, K, o):
    k = max(2, int(round(K * o["k_scale"])))
    if o["k_distribution"] == "fixed":
        return np.full(n, k)
    if o["k_distribution"] == "poisson":
        c = rng.poisson(k, n)
    elif o["k_distribution"] == "lognormal":
        c = np.round(rng.lognormal(math.log(k) - 0.5, 1.0, n)).astype(int)
    else:
        raise ValueError(o["k_distribution"])
    return np.clip(c, 2, 4 * k)


def sample_variant_batch(rng, n, K, D, target, w, opts=None):
    """Return (parent int8 [n, D + n_irr], children int8 [n, K_max, D + n_irr], mask bool [n, K_max], labels)."""
    o = options(opts)
    km = k_max(K, o)
    counts = _child_counts(rng, n, K, o)
    mask = np.arange(km)[None, :] < counts[:, None]
    p = o["bit_prob"]
    query = (rng.random((n, D)) < p).astype(np.int8)
    rel = (rng.random((n, km, D)) < p).astype(np.int8)
    if target == "mean_based":
        m = mask[:, :, None]
        mean = (rel * m).sum(1) / counts[:, None]
        score = (mean - p) @ w + 1e-9 * rng.standard_normal(n)
        labels = (score > 0).astype(np.float32)
    else:
        clash = np.all(rel == query[:, None, :], axis=2)
        while clash.any():  # move rows off the query (resampling keeps the bit distribution)
            rows, cols = np.nonzero(clash)
            rel[rows, cols] = (rng.random((len(rows), D)) < p).astype(np.int8)
            clash = np.all(rel == query[:, None, :], axis=2)
        labels = (rng.random(n) < 0.5).astype(np.float32)
        planted = labels.astype(int) + (1 if target == "distributional" else 0)
        for i in range(n):
            if planted[i]:
                pos = rng.choice(counts[i], size=min(planted[i], counts[i]), replace=False)
                rel[i, pos] = query[i]
    irr = o["n_irrelevant"]
    if irr:
        rel = np.concatenate([rel, (rng.random((n, km, irr)) < 0.5).astype(np.int8)], axis=2)
        query = np.concatenate([query, (rng.random((n, irr)) < 0.5).astype(np.int8)], axis=1)
    children = (2 * rel - 1).astype(np.int8)
    if o["missing_rate"] > 0:
        children[rng.random(children.shape) < o["missing_rate"]] = 0
    children[~mask] = 0
    parent = (2 * query - 1).astype(np.int8)
    return parent, children, mask, labels


def binary_entropy(p):
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def variant_information(K, D, d, opts=None):
    """Nominal information quantities of a variant (bits)."""
    o = options(opts)
    k_mean = K * o["k_scale"]
    h_rel = D * binary_entropy(o["bit_prob"])
    h_all = h_rel + o["n_irrelevant"]
    miss = o["missing_rate"]
    observed = (1 - miss) if miss else 1.0
    return {"K_mean": k_mean, "H_row_relevant": h_rel, "H_row_total": h_all,
            "jbr_task_agnostic": k_mean * h_all / d, "jbr_task_conditional": k_mean * h_rel / d,
            "jbr_task_conditional_observed": k_mean * h_rel * observed / d,
            "jbr_count_based": k_mean * (D + o["n_irrelevant"]) / d}
