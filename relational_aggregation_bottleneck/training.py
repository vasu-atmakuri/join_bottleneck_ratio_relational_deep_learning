"""
Online training with a fresh synthetic mini-batch at every step. The model is evaluated on a fixed validation
set at regular intervals; the reported test AUROC is the one at the best validation checkpoint.
"""

import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data_generation import _scenario_rng, fixed_split, join_statistics, sample_batch, scenario_direction
from .models import build_model

TRAIN_CONFIG = {
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 128,
    "max_steps": 8000,
    "min_steps": 4000,  # no early stop before this, so slow starts out of the 0.5 plateau are not cut off
    "eval_every": 200,
    "patience_evals": 10,
    "n_validation": 1000,
    "n_test": 2000,
    "child_rows_per_batch_cap": 8_192,  # smaller batches for very large K keep memory bounded
}


def _signed(x):
    return torch.from_numpy(x).float() * 2 - 1  # bits encoded as -1 / +1


def _auroc(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else float("nan")


def _scores(model, parent, children, batch):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(parent), batch):
            out.append(model(_signed(parent[i:i + batch]), _signed(children[i:i + batch])).numpy())
    model.train()
    return np.concatenate(out)


def train_and_evaluate(K, D, d, target, family, variant, seed, heads=1, config=None):
    cfg = dict(TRAIN_CONFIG, **(config or {}))
    torch.manual_seed(seed)
    w = scenario_direction(K, D, d, target, seed)
    val = fixed_split(K, D, d, target, seed, cfg["n_validation"], stream=1)
    test = fixed_split(K, D, d, target, seed, cfg["n_test"], stream=2)
    train_rng = _scenario_rng(K, D, d, target, seed, 0)
    batch = int(max(16, min(cfg["batch_size"], cfg["child_rows_per_batch_cap"] // K)))
    eval_batch = 4 * batch

    model = build_model(family, variant, K, D, d, heads=heads)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best = {"val_auroc": -1.0, "best_step": 0, "test_auroc": float("nan")}
    stale, t0, step, running = 0, time.time(), 0, []
    for step in range(1, cfg["max_steps"] + 1):
        p, c, y = sample_batch(train_rng, batch, K, D, target, w)
        loss = loss_fn(model(_signed(p), _signed(c)), torch.from_numpy(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        running.append(loss.item())
        if step % cfg["eval_every"] == 0:
            v = _auroc(val[2], _scores(model, val[0], val[1], eval_batch))
            if v > best["val_auroc"] + 1e-4:
                best.update(val_auroc=v, best_step=step,
                            test_auroc=_auroc(test[2], _scores(model, test[0], test[1], eval_batch)),
                            train_loss=float(np.mean(running[-cfg["eval_every"]:])))
                stale = 0
            else:
                stale += 1
            if best["val_auroc"] >= 0.9995 or (stale >= cfg["patience_evals"] and step >= cfg["min_steps"]):
                break
    stats = join_statistics(test[1], d)
    best.update(steps_run=step, seconds=round(time.time() - t0, 2), batch_size=batch,
                n_parameters=sum(p.numel() for p in model.parameters()),
                effective_heads=getattr(model, "heads", 0), **stats)
    return best
