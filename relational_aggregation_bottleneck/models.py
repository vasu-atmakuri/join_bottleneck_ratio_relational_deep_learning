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


def pool(h, kind):
    if kind == "sum":
        return h.sum(dim=1)
    if kind == "mean":
        return h.mean(dim=1)
    if kind == "max":
        return h.max(dim=1).values
    raise ValueError(kind)


class DeepSetsAggregator(nn.Module):
    def __init__(self, D, d, aggregator):
        super().__init__()
        self.aggregator = aggregator
        self.phi = mlp([D, HIDDEN, HIDDEN, d])
        self.parent_encoder = mlp([D, HIDDEN, HIDDEN], final_activation=True)
        self.rho = mlp([d + HIDDEN, HIDDEN, HIDDEN, 1])

    def forward(self, parent, children):
        agg = pool(self.phi(children), self.aggregator)
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

    def aggregate(self, hp, hc):
        if self.architecture == "SumAgg":
            return self.w_msg(hc).sum(dim=1)
        if self.architecture == "MeanAgg":
            return self.w_msg(hc).mean(dim=1)
        B, K, _ = hc.shape
        q = self.q(hp).view(B, self.heads, 1, self.head_dim)
        k = self.k(hc).view(B, K, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(hc).view(B, K, self.heads, self.head_dim).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-1, -2) / self.head_dim ** 0.5, dim=-1)
        return self.out((att @ v).reshape(B, self.heads * self.head_dim))

    def forward(self, parent, children):
        hp = self.parent_encoder(parent)
        hc = self.child_encoder(children)
        h = torch.relu(self.w_self(hp) + self.aggregate(hp, hc))
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


def build_model(family, variant, K, D, d, heads=1):
    if family == "deepsets":
        return DeepSetsAggregator(D, d, variant)
    if family == "gnn":
        return GNNAggregator(D, d, variant, heads=heads)
    if family == "denormalized":
        return DenormalizedMLP(K, D)
    raise ValueError(family)
