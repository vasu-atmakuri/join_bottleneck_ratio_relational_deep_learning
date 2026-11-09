"""
Temporal neighbor sampling for heterogeneous relational graphs in plain NumPy (no compiled extensions).

Semantics follow relational deep learning practice: every seed entity gets its own sampled computation tree
(disjoint subgraphs); a neighbor is admissible only if its timestamp is not later than the seed time (nodes without a
time attribute are always admissible); at each hop at most `fanout` admissible neighbors are drawn uniformly per target
node and edge type (all of them when there are fewer). Messages flow along the graph's edge types (child -> parent and
parent -> child), exactly as in the heterogeneous message-passing layers that consume the batch.
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData

TIME_SHIFT = 1 << 40  # composite key: destination id * TIME_SHIFT + (time - min time)


class TemporalSubgraphSampler:
    def __init__(self, data, fanouts, seed=0):
        self.data = data
        self.fanouts = list(fanouts)
        self.rng = np.random.default_rng(seed)
        self.index = {}
        times = [data[nt].time.numpy() for nt in data.node_types if "time" in data[nt]]
        self.tmin = int(min(t.min() for t in times)) if times else 0
        for et in data.edge_types:
            src, _, dst = et
            s, d = data[et].edge_index.numpy()
            st = data[src].time.numpy().astype(np.int64) - self.tmin if "time" in data[src] else None
            order = np.lexsort((st[s], d)) if st is not None else np.argsort(d, kind="stable")
            d_sorted, s_sorted = d[order].astype(np.int64), s[order].astype(np.int64)
            n_dst = data[dst].num_nodes
            rowptr = np.searchsorted(d_sorted, np.arange(n_dst + 1))
            key = d_sorted * TIME_SHIFT + st[s_sorted] if st is not None else None
            self.index[et] = (rowptr, s_sorted, key)

    def _neighbors(self, et, targets, seed_times, fanout):
        rowptr, s_sorted, key = self.index[et]
        start, end = rowptr[targets], rowptr[targets + 1]
        if key is not None:
            q = targets.astype(np.int64) * TIME_SHIFT + np.clip(seed_times - self.tmin, 0, TIME_SHIFT - 1)
            end = np.clip(np.searchsorted(key, q, side="right"), start, end)
        cnt = end - start
        take = np.minimum(cnt, fanout)
        total = int(take.sum())
        if total == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        rep = np.repeat(np.arange(len(targets)), take)
        within = np.arange(total) - np.repeat(np.cumsum(take) - take, take)
        big = (cnt > fanout)[rep]
        rnd = (self.rng.random(total) * cnt[rep]).astype(np.int64)
        pos = start[rep] + np.where(big, rnd, within)
        return s_sorted[pos], rep

    def sample(self, entity, seed_nodes, seed_times):
        seed_nodes = np.asarray(seed_nodes, np.int64)
        seed_times = np.asarray(seed_times, np.int64)
        orig = {entity: [seed_nodes]}
        bidx = {entity: [np.arange(len(seed_nodes))]}
        count = {entity: len(seed_nodes)}
        edges = {}
        frontier = {entity: (seed_nodes, np.arange(len(seed_nodes)), np.arange(len(seed_nodes)))}
        for fanout in self.fanouts:
            new = {}
            for et in self.data.edge_types:
                src, _, dst = et
                if dst not in frontier:
                    continue
                t_orig, t_batch, t_local = frontier[dst]
                nb, rep = self._neighbors(et, t_orig, seed_times[t_batch], fanout)
                if len(nb) == 0:
                    continue
                start = count.get(src, 0)
                local = start + np.arange(len(nb))
                count[src] = start + len(nb)
                orig.setdefault(src, []).append(nb)
                bidx.setdefault(src, []).append(t_batch[rep])
                edges.setdefault(et, []).append(np.stack([local, t_local[rep]]))
                prev = new.get(src)
                cur = (nb, t_batch[rep], local)
                new[src] = cur if prev is None else tuple(np.concatenate([a, b]) for a, b in zip(prev, cur))
            frontier = new
        batch = HeteroData()
        for nt, parts in orig.items():
            ids = torch.from_numpy(np.concatenate(parts))
            batch[nt].tf = self.data[nt].tf[ids]
            batch[nt].n_id = ids
            batch[nt].batch = torch.from_numpy(np.concatenate(bidx[nt]))
            batch[nt].num_nodes = len(ids)
            if "time" in self.data[nt]:
                batch[nt].time = self.data[nt].time[ids]
        for et, parts in edges.items():
            batch[et].edge_index = torch.from_numpy(np.concatenate(parts, axis=1))
        batch[entity].seed_time = torch.from_numpy(seed_times)
        return batch


class TemporalNodeLoader:
    """Iterates over a task table in mini-batches of seed entities and yields sampled subgraphs."""

    def __init__(self, sampler, entity, nodes, times, targets, batch_size, shuffle, seed=0):
        self.sampler, self.entity = sampler, entity
        self.nodes, self.times, self.targets = np.asarray(nodes), np.asarray(times), targets
        self.batch_size, self.shuffle = batch_size, shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (len(self.nodes) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = self.rng.permutation(len(self.nodes)) if self.shuffle else np.arange(len(self.nodes))
        for i in range(0, len(order), self.batch_size):
            idx = order[i:i + self.batch_size]
            b = self.sampler.sample(self.entity, self.nodes[idx], self.times[idx])
            if self.targets is not None:
                b[self.entity].y = self.targets[torch.from_numpy(idx)]
            yield b
