#!/usr/bin/env python3
"""
Robustness experiments that test the capacity law and the mechanism claims outside the original task family.

Every run uses the online training protocol of relational_aggregation_bottleneck/training.py (fresh mini-batch per
step, best-validation checkpoint, 8000 steps at most) unless stated otherwise.

task_families      DeepSets sum on 12 of the 36 out-of-sample configurations, three task families that are not exact
                   whole-row matches:
                     partial_match     a child matches when it equals the parent query on the first ceil(D/2)
                                       columns (a predicate on a subset of columns); 1 match vs none
                     pair_conjunction  the parent carries two query rows; label 1 if both occur among the children, 0
                                       if exactly one does (an interaction between two child records)
                     latest_match      every child has a timestamp; exactly one child equals the query, label 1 if it
                                       is the most recent child, 0 otherwise (temporal ordering, top-1 selection)
distributions      DeepSets sum, rare_pattern, the same 12 configurations, child rows that are not uniform bits:
                     correlated        each bit copies the previous bit with probability 0.8 (marginals stay uniform,
                                       joint entropy falls well below the summed plug-in entropy)
                     zipf              rows drawn from a vocabulary of 4096 patterns with Zipf(1.1) frequencies
                     duplicates        each parent's children are drawn from max(2, K/4) distinct rows
query_conditioning one-layer message passing with sum aggregation on the 6 highest-JBR architecture scenarios,
                   rare_pattern, four child inputs: plain (baseline), query_concat (the child encoder also receives the
                   parent query), oracle_match (the child encoder also receives the exact-match bit 1[x_c = q_p]),
                   and the two-layer model with the parent-to-child message
attention_margins  one-layer single-head attention, rare_pattern, D = 8, d = 32, K from 8 to 512; on the test set the
                   attention mass on the matching child and the score margin to the best non-matching child are logged
width_budget       DeepSets sum, rare_pattern, K = 32, D = 8, d from 16 to 256 with hidden = max(128, d), at the standard
                   budget with learning rates down to 3e-5, and at a four times longer budget (32000 steps, no early
                   stop before 16000); d = 512 is covered by the original width sweep

Seeds 0 and 1, learning rate 1e-3; width_budget uses 3e-4 at every width plus, at the standard budget, 1e-4 (d >= 128)
and 3e-5 (d >= 256); the best validation rate is taken per width, seed and budget.
Results: results/review_robustness/runs.jsonl (resumable, one line per run)
Usage:   python experiments/run_review_robustness_experiments.py --workers 2
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

RESULTS = ROOT / "results" / "review_robustness" / "runs.jsonl"
TASKS = ["partial_match", "pair_conjunction", "latest_match"]
DISTS = ["correlated", "zipf", "duplicates"]
CHILD_INPUTS = ["plain", "query_concat", "oracle_match", "two_layer_downward"]


# ----------------------------------------------------------------------------------------------------------- data
def _vocab(D, seed):
    rng = np.random.default_rng([D, seed, 4096])
    V = min(4096, 2 ** D)
    if 2 ** D <= 4096:
        vocab = ((np.arange(2 ** D)[:, None] >> np.arange(D)[None, :]) & 1).astype(np.uint8)
        vocab = vocab[rng.permutation(len(vocab))]
    else:
        vocab = rng.integers(0, 2, size=(V, D), dtype=np.uint8)
    w = 1.0 / np.arange(1, V + 1) ** 1.1
    return vocab, w / w.sum()


def _rows(rng, shape, D, dist, vocab=None):
    n, K = shape
    if dist in ("uniform", "duplicates"):
        return rng.integers(0, 2, size=(n, K, D), dtype=np.uint8)
    if dist == "correlated":
        x = np.empty((n, K, D), dtype=np.uint8)
        x[..., 0] = rng.integers(0, 2, size=(n, K))
        for j in range(1, D):
            copy = rng.random((n, K)) < 0.8
            x[..., j] = np.where(copy, x[..., j - 1], rng.integers(0, 2, size=(n, K)))
        return x
    if dist == "zipf":
        v, p = vocab
        return v[rng.choice(len(v), size=(n, K), p=p)]
    raise ValueError(dist)


def _query(rng, n, D, dist, vocab=None):
    return _rows(rng, (n, 1), D, dist, vocab)[:, 0]


def _move_off(rng, children, query, cols, D, dist, vocab):
    """Resample child rows that match the query on `cols` until none does."""
    for _ in range(200):
        clash = np.all(children[:, :, cols] == query[:, None, cols], axis=2)
        if not clash.any():
            return children
        r, c = np.nonzero(clash)
        children[r, c] = _rows(rng, (len(r), 1), D, dist, vocab)[:, 0]
    clash = np.all(children[:, :, cols] == query[:, None, cols], axis=2)
    r, c = np.nonzero(clash)
    children[r, c, cols[0]] ^= 1
    return children


def sample(rng, n, K, D, task, dist="uniform", vocab=None):
    """Return parent [n, Dp] float, children [n, K, Dc] float (both in [-1, 1]), labels [n], match [n, K] bool."""
    labels = (rng.random(n) < 0.5).astype(np.float32)
    query = _query(rng, n, D, dist, vocab)
    if dist == "duplicates":
        m = max(2, K // 4)
        pool = rng.integers(0, 2, size=(n, m, D), dtype=np.uint8)
        pool = _move_off(rng, pool, query, list(range(D)), D, "uniform", None)
        children = np.take_along_axis(pool, rng.integers(0, m, size=(n, K))[:, :, None], axis=1)
    else:
        children = _rows(rng, (n, K), D, dist, vocab)
    extra_parent = None
    times = None
    if task in ("rare_pattern", "partial_match"):
        cols = list(range(D)) if task == "rare_pattern" else list(range(math.ceil(D / 2)))
        if dist != "duplicates" or task != "rare_pattern":
            children = _move_off(rng, children, query, cols, D, dist if dist != "duplicates" else "uniform", vocab)
        idx = np.nonzero(labels == 1)[0]
        pos = rng.integers(0, K, size=n)
        if task == "rare_pattern":
            children[idx, pos[idx]] = query[idx]
        else:
            children[idx, pos[idx], :len(cols)] = query[idx][:, cols]
        match = np.all(children[:, :, cols] == query[:, None, cols], axis=2)
    elif task == "pair_conjunction":
        q2 = _query(rng, n, D, dist, vocab)
        same = np.all(q2 == query, axis=1)
        q2[same, 0] ^= 1
        for _ in range(20):
            children = _move_off(rng, children, query, list(range(D)), D, dist, vocab)
            children = _move_off(rng, children, q2, list(range(D)), D, dist, vocab)
            if not np.all(children == query[:, None], axis=2).any():
                break
        # class 1: both queries present; class 0: exactly one of them present (chosen at random)
        perm = np.argsort(rng.random((n, K)), axis=1)
        p1, p2 = perm[:, 0], perm[:, min(1, K - 1)]
        one_is_first = rng.random(n) < 0.5
        both = labels == 1
        r = np.arange(n)
        children[r[both], p1[both]] = query[both]
        children[r[both], p2[both]] = q2[both]
        sel = ~both & one_is_first
        children[r[sel], p1[sel]] = query[sel]
        sel = ~both & ~one_is_first
        children[r[sel], p1[sel]] = q2[sel]
        match = np.all(children == query[:, None], axis=2) | np.all(children == q2[:, None], axis=2)
        extra_parent = q2
    elif task == "latest_match":
        children = _move_off(rng, children, query, list(range(D)), D, dist, vocab)
        times = rng.random((n, K)).astype(np.float32)
        latest = times.argmax(axis=1)
        other = (latest + rng.integers(1, K, size=n)) % K
        pos = np.where(labels == 1, latest, other)
        children[np.arange(n), pos] = query
        match = np.zeros((n, K), bool)
        match[np.arange(n), pos] = True
    else:
        raise ValueError(task)
    parent = query.astype(np.float32) * 2 - 1
    if extra_parent is not None:
        parent = np.concatenate([parent, extra_parent.astype(np.float32) * 2 - 1], axis=1)
    ch = children.astype(np.float32) * 2 - 1
    if times is not None:
        ch = np.concatenate([ch, (times[:, :, None] * 2 - 1)], axis=2)
        parent = np.concatenate([parent, np.zeros((n, 1), np.float32)], axis=1)
    return parent, ch, labels, match, children


def plug_in_row_entropy(children_bits):
    flat = children_bits.reshape(-1, children_bits.shape[-1]).astype(float)
    p = flat.mean(0)
    h = -(p * np.log2(np.clip(p, 1e-12, 1)) + (1 - p) * np.log2(np.clip(1 - p, 1e-12, 1)))
    return float(np.nansum(np.where((p > 0) & (p < 1), h, 0)))


def joint_row_entropy(children_bits):
    """Plug-in entropy of whole rows (lower bound on the true joint entropy for large D)."""
    flat = children_bits.reshape(-1, children_bits.shape[-1])
    _, counts = np.unique(flat, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------------------------------------- models
def build(j, Dp, Dc):
    import torch
    from torch import nn
    from relational_aggregation_bottleneck.models import GNNAggregator, HIDDEN, mlp, pool

    d = j["d"]

    class DeepSets(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            h = hidden
            self.phi = mlp([Dc, h, h, d])
            self.parent_encoder = mlp([Dp, h, h], final_activation=True)
            self.rho = mlp([d + h, h, h, 1])

        def forward(self, p, c, extra=None):
            return self.rho(torch.cat([pool(self.phi(c), "sum"), self.parent_encoder(p)], -1)).squeeze(-1)

    class OneLayer(nn.Module):
        """GNNAggregator with an optional query-aware child input."""

        def __init__(self, child_input):
            super().__init__()
            self.child_input = child_input
            extra = {"plain": 0, "query_concat": Dp, "oracle_match": 1}[child_input]
            self.child_encoder = mlp([Dc + extra, HIDDEN, d], final_activation=True)
            self.parent_encoder = mlp([Dp, HIDDEN, d], final_activation=True)
            self.w_self = nn.Linear(d, d)
            self.w_msg = nn.Linear(d, d)
            self.head = mlp([d, HIDDEN, 1])
            self.heads = 0

        def forward(self, p, c, extra=None):
            if self.child_input == "query_concat":
                c = torch.cat([c, p.unsqueeze(1).expand(-1, c.shape[1], -1)], -1)
            elif self.child_input == "oracle_match":
                c = torch.cat([c, extra.unsqueeze(-1)], -1)
            hp, hc = self.parent_encoder(p), self.child_encoder(c)
            return self.head(torch.relu(self.w_self(hp) + pool(self.w_msg(hc), "sum"))).squeeze(-1)

    class TwoLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.child_encoder = mlp([Dc, HIDDEN, d], final_activation=True)
            self.parent_encoder = mlp([Dp, HIDDEN, d], final_activation=True)
            self.w_child = nn.Linear(d, d)
            self.w_parent_to_child = nn.Linear(d, d, bias=False)
            self.w_self = nn.Linear(d, d)
            self.w_msg = nn.Linear(d, d)
            self.head = mlp([d, HIDDEN, 1])
            self.heads = 0

        def forward(self, p, c, extra=None):
            hp, hc = self.parent_encoder(p), self.child_encoder(c)
            hc = torch.relu(self.w_child(hc) + self.w_parent_to_child(hp).unsqueeze(1))
            return self.head(torch.relu(self.w_self(hp) + pool(self.w_msg(hc), "sum"))).squeeze(-1)

    class Attn(GNNAggregator):
        def forward(self, p, c, extra=None):
            return super().forward(p, c)

    m = j["model"]
    if m == "deepsets_sum":
        return DeepSets(max(HIDDEN, d))
    if m == "attention":
        return Attn(Dc, d, "AttnAgg", heads=1)
    if m == "two_layer_downward":
        return TwoLayer()
    return OneLayer(m)


# ---------------------------------------------------------------------------------------------------- training
def run(j):
    import torch
    from sklearn.metrics import roc_auc_score

    torch.set_num_threads(1)
    t0 = time.time()
    K, D, d, seed = j["K"], j["D"], j["d"], j["seed"]
    task, dist = j["task"], j["dist"]
    vocab = _vocab(D, seed) if dist == "zipf" else None
    tag = [K, D, d, seed, TASKS.index(task) if task in TASKS else 9, (DISTS + ["uniform"]).index(dist)]
    rng_tr, rng_va, rng_te = (np.random.default_rng(tag + [s]) for s in (0, 1, 2))
    val, test = sample(rng_va, 1000, K, D, task, dist, vocab), sample(rng_te, 2000, K, D, task, dist, vocab)
    Dp, Dc = val[0].shape[1], val[1].shape[2]
    torch.manual_seed(seed)
    model = build(j, Dp, Dc)
    long = j.get("budget") == "long"
    max_steps, min_steps, patience = (32000, 16000, 40) if long else (8000, 4000, 10)
    opt = torch.optim.Adam(model.parameters(), lr=j["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    batch = int(max(16, min(128, 8192 // K)))
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731

    def scores(s):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(s[0]), 4 * batch):
                sl = slice(i, i + 4 * batch)
                out.append(model(T(s[0][sl]), T(s[1][sl]), T(s[3][sl].astype(np.float32) * 2 - 1)).numpy())
        model.train()
        return np.concatenate(out)

    best = {"val_auroc": -1.0, "test_auroc": float("nan"), "best_step": 0}
    stale, state = 0, None
    for step in range(1, max_steps + 1):
        p, c, y, mt, _ = sample(rng_tr, batch, K, D, task, dist, vocab)
        loss = loss_fn(model(T(p), T(c), T(mt.astype(np.float32) * 2 - 1)), T(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            v = float(roc_auc_score(val[2], scores(val)))
            if v > best["val_auroc"] + 1e-4:
                best.update(val_auroc=v, best_step=step, test_auroc=float(roc_auc_score(test[2], scores(test))),
                            train_loss=float(loss.item()))
                stale = 0
                if j["model"] == "attention":
                    state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                stale += 1
            if best["val_auroc"] >= 0.9995 or (stale >= patience and step >= min_steps):
                break
    out = {**j, **best, "steps_run": step, "n_parameters": sum(q.numel() for q in model.parameters()),
           "row_entropy_bits": plug_in_row_entropy(test[4]), "joint_row_entropy_bits": joint_row_entropy(test[4]),
           "status": "ok", "seconds": round(time.time() - t0, 1)}
    if j["model"] == "attention" and state is not None:
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pp, cc = T(test[0]), T(test[1])
            hp, hc = model.parent_encoder(pp), model.child_encoder(cc)
            q = model.q(hp)[:, None, :]
            k = model.k(hc)
            s = (q * k).sum(-1) / model.head_dim ** 0.5
            att = torch.softmax(s, -1).numpy()
            s = s.numpy()
        mt = test[3]
        pos = test[2] == 1
        mass = (att * mt).sum(1)[pos]
        margin = (np.where(mt, s, -np.inf).max(1) - np.where(mt, -np.inf, s).max(1))[pos]
        out.update(attention_mass_on_match_mean=float(mass.mean()), attention_mass_on_match_median=float(
            np.median(mass)), score_margin_mean=float(margin.mean()), score_margin_median=float(np.median(margin)),
            share_match_is_argmax=float((margin > 0).mean()), log_K=float(np.log(K)))
    return out


# ------------------------------------------------------------------------------------------------------- plan
def twelve_configurations():
    from run_law_validation_and_depth_experiments import out_of_sample_configurations
    return out_of_sample_configurations()[1::3]


def J(experiment, K, D, d, task, dist, model, seed, lr, budget="standard"):
    return {"experiment": experiment, "K": K, "D": D, "d": d, "task": task, "dist": dist, "model": model,
            "seed": seed, "lr": lr, "budget": budget}


def key(j):
    return "|".join(str(j[k]) for k in ("experiment", "K", "D", "d", "task", "dist", "model", "seed", "lr", "budget"))


def all_jobs():
    from relational_aggregation_bottleneck.scenarios import architecture_scenarios
    jobs = []
    for (K, D, d), t, s in product(twelve_configurations(), TASKS, [0, 1]):
        jobs.append(J("task_families", K, D, d, t, "uniform", "deepsets_sum", s, 1e-3))
    for (K, D, d), dist, s in product(twelve_configurations(), DISTS, [0, 1]):
        jobs.append(J("distributions", K, D, d, "rare_pattern", dist, "deepsets_sum", s, 1e-3))
    for sc in sorted(architecture_scenarios(), key=lambda x: -x.nominal_jbr)[:6]:
        for m, s in product(CHILD_INPUTS, [0, 1]):
            jobs.append(J("query_conditioning", sc.K, sc.D, sc.d, "rare_pattern", "uniform", m, s, 1e-3))
    for K in [8, 32, 128, 512]:
        for s in [0, 1]:
            jobs.append(J("attention_margins", K, 8, 32, "rare_pattern", "uniform", "attention", s, 1e-3))
    for d, s in product([16, 32, 64, 128, 256], [0, 1]):
        jobs.append(J("width_budget", 32, 8, d, "rare_pattern", "uniform", "deepsets_sum", s, 3e-4))
        jobs.append(J("width_budget", 32, 8, d, "rare_pattern", "uniform", "deepsets_sum", s, 3e-4, "long"))
    # wide models most often select the smallest rate of the original grid, so the grid is extended downwards
    for d, s in product([128, 256], [0, 1]):
        for lr in ([1e-4] if d == 128 else [1e-4, 3e-5]):
            jobs.append(J("width_budget", 32, 8, d, "rare_pattern", "uniform", "deepsets_sum", s, lr))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    jobs = all_jobs()
    if args.list:
        from collections import Counter
        for k, v in Counter(j["experiment"] for j in jobs).items():
            print(f"{k:22s} {v}")
        print("total", len(jobs))
        return
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        done = {key(json.loads(line)) for line in RESULTS.open() if '"status": "ok"' in line}
    todo = [j for j in jobs if key(j) not in done]
    todo.sort(key=lambda j: (j["budget"] == "long", j["K"]))
    print(f"{len(jobs)} runs in plan, {len(todo)} to run", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool, RESULTS.open("a") as f:
        for n, r in enumerate(pool.imap_unordered(_safe, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            st = f"AUROC {r['test_auroc']:.3f}" if r["status"] == "ok" else r["error"]
            print(f"[{n}/{len(todo)}] {r['experiment']} K{r['K']} D{r['D']} d{r['d']} {r['task']} {r['dist']} "
                  f"{r['model']} {r['budget']} s{r['seed']}: {st} | {(time.time() - t0) / 60:.1f} min", flush=True)
    print("REVIEW_ROBUSTNESS_COMPLETE", flush=True)


def _safe(j):
    try:
        return run(j)
    except Exception as e:  # noqa: BLE001
        return {**j, "status": "error", "error": repr(e)}


if __name__ == "__main__":
    main()
