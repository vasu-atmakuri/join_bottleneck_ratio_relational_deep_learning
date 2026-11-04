"""
Synthetic parent-child relational data for the Join Bottleneck Ratio experiments.

Each parent row p has its own features, a query pattern q_p in {0,1}^D, and exactly K child rows, each a
vector of D binary features drawn uniformly at random (1 bit of entropy per feature). Three binary targets
are defined over the multiset of child rows:

  mean_based      1[w . (mean_c x_c - 1/2) > 0] for a fixed random direction w per scenario. Needs only the
                  first moment of the child rows (D numbers), so it is aggregation-friendly.
  rare_pattern    1 if exactly one child row equals the parent query q_p, 0 if none does. The parent
                  representation must retain which row patterns are present in the child set.
  distributional  1 if the parent query appears exactly twice among the children, 0 if exactly once. The
                  parent representation must retain multiplicities of row patterns (the empirical
                  distribution of child rows).

Non-planted child rows never equal q_p, so for rare_pattern and distributional the two classes differ only in
the planted rows and cannot be separated from marginal feature statistics.

Training data are generated on the fly (a fresh sample for every mini-batch), so results reflect the
capacity of the aggregation channel rather than the size of a fixed training set. Validation and test sets are
fixed samples drawn from separate random streams.

Information demand at the join, measured from sampled data:
  H_row   = sum over child features of the plug-in entropy of the feature (bits per child row)
  JBR     = K * H_row / d                          (Eq. 2 with H summed over the D child features)
  H_set   = log2 C(M + K - 1, K), M = 2 ** H_row    (entropy of an unordered multiset of K rows)
  JBR_set = H_set / d                              (multiset-corrected JBR)
"""

from math import lgamma, log

import numpy as np

TARGET_TYPES = ("mean_based", "rare_pattern", "distributional")


def _scenario_rng(K, D, d, target, seed, stream):
    return np.random.default_rng([K, D, d, TARGET_TYPES.index(target), seed, stream])


def scenario_direction(K, D, d, target, seed):
    """Fixed readout direction w for the mean_based target."""
    return _scenario_rng(K, D, d, target, seed, 99).standard_normal(D)


def sample_batch(rng, n, K, D, target, w=None):
    """Return (parent [n, D] uint8, children [n, K, D] uint8, labels [n] float32)."""
    query = rng.integers(0, 2, size=(n, D), dtype=np.uint8)
    children = rng.integers(0, 2, size=(n, K, D), dtype=np.uint8)
    if target == "mean_based":
        score = (children.mean(axis=1) - 0.5) @ w + 1e-9 * rng.standard_normal(n)
        return query, children, (score > 0).astype(np.float32)

    # Rows equal to the query are moved off it by flipping one random bit.
    clash = np.all(children == query[:, None, :], axis=2)
    if clash.any():
        rows, cols = np.nonzero(clash)
        bits = rng.integers(0, D, size=len(rows))
        children[rows, cols, bits] ^= 1
    labels = (rng.random(n) < 0.5).astype(np.float32)
    n_planted = labels.astype(int) + (1 if target == "distributional" else 0)
    n_planted = np.minimum(n_planted, K)
    for copies in (1, 2):
        idx = np.nonzero(n_planted >= copies)[0]
        children[idx, copies - 1] = query[idx]
    # random positions for the planted rows
    perm = np.argsort(rng.random((n, K)), axis=1)
    children = np.take_along_axis(children, perm[:, :, None], axis=1)
    return query, children, labels


def fixed_split(K, D, d, target, seed, n, stream):
    rng = _scenario_rng(K, D, d, target, seed, stream)
    return sample_batch(rng, n, K, D, target, scenario_direction(K, D, d, target, seed))


def _plug_in_entropy_bits(column):
    p = column.mean()
    if p <= 0 or p >= 1:
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def _log2_multiset_count(M, K):
    """log2 of the number of multisets of size K drawn from M distinct values (M may be non-integer)."""
    return (lgamma(M + K) - lgamma(K + 1) - lgamma(M)) / log(2)


def join_statistics(children, d):
    n, K, D = children.shape
    flat = children.reshape(-1, D)
    h_row = float(sum(_plug_in_entropy_bits(flat[:, j]) for j in range(D)))
    h_set = _log2_multiset_count(2.0 ** h_row, K)
    return {
        "child_cardinality_K": K,
        "child_feature_dim_D": D,
        "embedding_dim_d": d,
        "row_entropy_bits": h_row,
        "mean_feature_entropy_bits": h_row / D,
        "multiset_entropy_bits": h_set,
        "jbr_empirical": K * h_row / d,
        "jbr_multiset": h_set / d,
        "jbr_nominal": K * D / d,
    }
