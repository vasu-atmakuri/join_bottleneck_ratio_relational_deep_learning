"""
Aggregation models for one parent-child foreign-key join.

All models compress the K child rows of a parent into a single vector of width d (the embedding
dimension that defines the JBR channel capacity) before combining it with the parent's own features.
Child-row encoders never see the parent, as in relational message passing.

DeepSetsAggregator   phi (3-layer MLP, output d) -> sum | mean | max over children -> concat parent encoding
                     -> rho (MLP) -> logit
GNNAggregator        one heterogeneous message-passing layer on the bipartite parent-child graph:
                     h_p' = ReLU(W_self h_p + AGG_c W_msg h_c), with AGG in {SumAgg, MeanAgg, AttnAgg};
                     AttnAgg is multi-head attention with the parent embedding as query (total width d).
TwoLayerGNNAggregator two message-passing layers on the same bipartite graph, as in relational deep learning where
                     every foreign key is used in both directions:
                       layer 1 (parent -> child):  h_c' = ReLU(W_c h_c + W_pc h_p)       (omitted when downward=False)
                       layer 2 (child -> parent):  h_p' = ReLU(W_self h_p + AGG_c W_msg h_c')
                     With the downward message each child can compare itself with its parent before pooling, so the
                     pooled vector can carry query-dependent evidence instead of a summary of the whole set. With
                     downward=False layer 1 is a child-only transformation of equal depth (control for depth).
DenormalizedMLP      oracle-style baseline that bypasses aggregation: the parent features and all child rows
                     (sorted lexicographically, so the input is permutation invariant) are concatenated and
                     fed to an MLP.
"""

import torch
from torch import nn

HIDDEN = 128


def mlp(sizes, final_activation=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def pool(h, kind, mask=None):
    """Permutation-invariant pooling over children; `mask` [B, K] marks real (non-padded) children."""
    if mask is None:
        if kind == "sum":
            return h.sum(dim=1)
        if kind == "mean":
            return h.mean(dim=1)
        if kind == "max":
            return h.max(dim=1).values
        raise ValueError(kind)
    m = mask.unsqueeze(-1).to(h.dtype)
    if kind == "sum":
        return (h * m).sum(dim=1)
    if kind == "mean":
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
    if kind == "max":
        return h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
    raise ValueError(kind)


class DeepSetsAggregator(nn.Module):
    """hidden: width of the encoder and decoder MLPs (default 128). With hidden < d the aggregated vector of a sum or
    mean aggregator spans at most `hidden` dimensions, so width-sweep experiments set hidden >= d.
    layer_norm: normalize the aggregated vector before the decoder."""

    def __init__(self, D, d, aggregator, hidden=None, layer_norm=False):
        super().__init__()
        h = hidden or HIDDEN
        self.aggregator = aggregator
        self.phi = mlp([D, h, h, d])
        self.parent_encoder = mlp([D, h, h], final_activation=True)
        self.norm = nn.LayerNorm(d) if layer_norm else nn.Identity()
        self.rho = mlp([d + h, h, h, 1])

    def forward(self, parent, children, mask=None):
        agg = self.norm(pool(self.phi(children), self.aggregator, mask))
        return self.rho(torch.cat([agg, self.parent_encoder(parent)], dim=-1)).squeeze(-1)


class GNNAggregator(nn.Module):
    def __init__(self, D, d, architecture, heads=1):
        super().__init__()
        self.architecture = architecture
        self.d = d
        self.child_encoder = mlp([D, HIDDEN, d], final_activation=True)
        self.parent_encoder = mlp([D, HIDDEN, d], final_activation=True)
        self.w_self = nn.Linear(d, d)
        if architecture == "AttnAgg":
            self.heads = max(1, min(heads, d))
            self.head_dim = max(1, d // self.heads)
            inner = self.heads * self.head_dim
            self.q = nn.Linear(d, inner, bias=False)
            self.k = nn.Linear(d, inner, bias=False)
            self.v = nn.Linear(d, inner, bias=False)
            self.out = nn.Linear(inner, d)
        else:
            self.heads = 0
            self.w_msg = nn.Linear(d, d)
        self.head = mlp([d, HIDDEN, 1])

    def aggregate(self, hp, hc, mask=None):
        if self.architecture == "SumAgg":
            return pool(self.w_msg(hc), "sum", mask)
        if self.architecture == "MeanAgg":
            return pool(self.w_msg(hc), "mean", mask)
        B, K, _ = hc.shape
        q = self.q(hp).view(B, self.heads, 1, self.head_dim)
        k = self.k(hc).view(B, K, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(hc).view(B, K, self.heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / self.head_dim ** 0.5
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        att = torch.softmax(scores, dim=-1)
        return self.out((att @ v).reshape(B, self.heads * self.head_dim))

    def forward(self, parent, children, mask=None):
        hp = self.parent_encoder(parent)
        hc = self.child_encoder(children)
        h = torch.relu(self.w_self(hp) + self.aggregate(hp, hc, mask))
        return self.head(h).squeeze(-1)


class TwoLayerGNNAggregator(nn.Module):
    def __init__(self, D, d, architecture, downward=True):
        super().__init__()
        self.architecture, self.downward, self.heads = architecture, downward, 0
        self.child_encoder = mlp([D, HIDDEN, d], final_activation=True)
        self.parent_encoder = mlp([D, HIDDEN, d], final_activation=True)
        self.w_child = nn.Linear(d, d)
        self.w_parent_to_child = nn.Linear(d, d, bias=False)
        self.w_self = nn.Linear(d, d)
        self.w_msg = nn.Linear(d, d)
        self.head = mlp([d, HIDDEN, 1])

    def forward(self, parent, children, mask=None):
        hp = self.parent_encoder(parent)
        hc = self.child_encoder(children)
        down = self.w_parent_to_child(hp).unsqueeze(1) if self.downward else 0.0
        hc = torch.relu(self.w_child(hc) + down)
        kind = "sum" if self.architecture == "SumAgg" else "mean"
        h = torch.relu(self.w_self(hp) + pool(self.w_msg(hc), kind, mask))
        return self.head(h).squeeze(-1)


class DenormalizedMLP(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.net = mlp([D + K * D, 256, HIDDEN, 1])
        self.D = D

    def forward(self, parent, children):
        B, K, D = children.shape
        weights = 2.0 ** torch.arange(D - 1, -1, -1, dtype=children.dtype, device=children.device)
        keys = (children * weights).sum(-1) if D <= 52 else children[..., 0]
        order = keys.argsort(dim=1)
        ordered = torch.gather(children, 1, order.unsqueeze(-1).expand(-1, -1, D))
        return self.net(torch.cat([parent, ordered.reshape(B, K * D)], dim=-1)).squeeze(-1)


def build_model(family, variant, K, D, d, heads=1, **options):
    if family == "deepsets":
        return DeepSetsAggregator(D, d, variant, hidden=options.get("hidden"),
                                  layer_norm=options.get("layer_norm", False))
    if family == "gnn":
        return GNNAggregator(D, d, variant, heads=heads)
    if family == "gnn_two_layer":
        return TwoLayerGNNAggregator(D, d, variant, downward=options.get("downward", True))
    if family == "denormalized":
        return DenormalizedMLP(K, D)
    raise ValueError(family)
