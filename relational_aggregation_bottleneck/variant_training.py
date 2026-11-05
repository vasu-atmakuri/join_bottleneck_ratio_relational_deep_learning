"""
Online training for the controlled data variants (masks, irrelevant features, missing values, test-time shift).
Mirrors training.py: fresh mini-batch per step, early stopping on validation AUROC, test AUROC at the best checkpoint.
`test_opts` lets the test set be drawn from a shifted distribution while training and validation use `opts`.
"""

import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data_generation import TARGET_TYPES
from .data_variants import k_max, options, sample_variant_batch, variant_information
from .models import build_model
from .training import TRAIN_CONFIG


def _rng(K, D, d, target, seed, stream, tag):
    return np.random.default_rng([K, D, d, TARGET_TYPES.index(target), seed, stream, tag])


def _tensors(batch):
    p, c, m, y = batch
    return torch.from_numpy(p).float(), torch.from_numpy(c).float(), torch.from_numpy(m), y


def _scores(model, data, bs):
    p, c, m, _ = data
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(p), bs):
            out.append(model(p[i:i + bs], c[i:i + bs], m[i:i + bs]).numpy())
    model.train()
    return np.concatenate(out)


def _auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else float("nan")


def train_and_evaluate_variant(K, D, d, target, family, variant, seed, heads=1, opts=None, test_opts=None,
                               config=None, tag=1):
    cfg = dict(TRAIN_CONFIG, **(config or {}))
    o, to = options(opts), options(test_opts or opts)
    torch.manual_seed(seed)
    w = _rng(K, D, d, target, seed, 99, tag).standard_normal(D)
    val = _tensors(sample_variant_batch(_rng(K, D, d, target, seed, 1, tag), cfg["n_validation"], K, D, target, w, o))
    test = _tensors(sample_variant_batch(_rng(K, D, d, target, seed, 2, tag), cfg["n_test"], K, D, target, w, to))
    train_rng = _rng(K, D, d, target, seed, 0, tag)
    km = k_max(K, o)
    batch = int(max(16, min(cfg["batch_size"], cfg["child_rows_per_batch_cap"] // km)))
    d_in = D + o["n_irrelevant"]
    model = build_model(family, variant, km, d_in, d, heads=heads)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best = {"val_auroc": -1.0, "best_step": 0, "test_auroc": float("nan")}
    stale, t0, step = 0, time.time(), 0
    for step in range(1, cfg["max_steps"] + 1):
        p, c, m, y = _tensors(sample_variant_batch(train_rng, batch, K, D, target, w, o))
        loss = loss_fn(model(p, c, m), torch.from_numpy(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg["eval_every"] == 0:
            v = _auroc(val[3], _scores(model, val, 4 * batch))
            if v > best["val_auroc"] + 1e-4:
                best.update(val_auroc=v, best_step=step, test_auroc=_auroc(test[3], _scores(model, test, 4 * batch)))
                stale = 0
            else:
                stale += 1
            if best["val_auroc"] >= 0.9995 or (stale >= cfg["patience_evals"] and step >= cfg["min_steps"]):
                break
    info = variant_information(K, D, d, o)
    best.update(steps_run=step, seconds=round(time.time() - t0, 2), batch_size=batch,
                effective_heads=getattr(model, "heads", 0), **info,
                **{f"opt_{k}": v for k, v in o.items()}, **{f"test_opt_{k}": v for k, v in to.items()})
    return best
