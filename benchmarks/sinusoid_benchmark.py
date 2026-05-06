"""
benchmarks/sinusoid_benchmark.py
================================
Sinusoid 5-shot regression benchmark following the evaluation protocol
from Finn et al. 2017 (MAML, ICML).

Task distribution:
    y = A * sin(x + phi),  x in [-5, 5]
    A   ~ Uniform(0.1, 5.0)
    phi ~ Uniform(0, pi)

Metric: Mean Squared Error on 100 query points after k inner gradient
steps at test time (averaged over 500 test tasks).

Conditions compared:
    1. Pretrain-only       : no adaptation at test time
    2. MAML-OLD            : deepcopy + load_state_dict (broken 2nd order)
    3. MAML-Functional     : torch.func.functional_call (correct 2nd order)
    4. FOMAML              : functional, first_order=True

Usage:
    python benchmarks/sinusoid_benchmark.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call
import numpy as np
import copy
import argparse

from maml_functional import functional_forward, maml_inner_loop


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Hyperparameters (match Finn et al. 2017) ─────────────────────────────────
N_EPOCHS    = 2000
META_BATCH  = 25
META_LR     = 0.001
INNER_LR    = 0.01
INNER_STEPS = 1      # 1-step MAML at train time
EVAL_STEPS  = 10     # 10-step adaptation at test time
N_SUPPORT   = 5
N_QUERY     = 100
N_EVAL      = 500


# ── Model (identical to MAML paper: 2 hidden, 40 units, ReLU) ────────────────
class SinusoidNet(nn.Module):
    def __init__(self, hidden: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Task generator ────────────────────────────────────────────────────────────
def sample_sinusoid_task(
    n_support: int = N_SUPPORT,
    n_query: int = N_QUERY,
):
    A   = np.random.uniform(0.1, 5.0)
    phi = np.random.uniform(0, np.pi)

    def sample(n: int):
        x = torch.FloatTensor(n, 1).uniform_(-5, 5)
        y = A * torch.sin(x + phi)
        return x, y

    return sample(n_support), sample(n_query)


# ── OLD inner loop (deepcopy-based, for comparison only) ─────────────────────
def _old_inner_loop(
    model: nn.Module,
    sx: torch.Tensor,
    sy: torch.Tensor,
    lr: float,
    steps: int,
) -> None:
    """In-place adaptation via deepcopy + load_state_dict (graph-breaking)."""
    fast = copy.deepcopy(model.state_dict())
    for _ in range(steps):
        model.load_state_dict(fast)
        pred  = model(sx)
        loss  = F.mse_loss(pred, sy)
        grads = torch.autograd.grad(loss, model.parameters(), allow_unused=True)
        with torch.no_grad():
            for (name, _), g in zip(model.named_parameters(), grads):
                if g is not None:
                    fast[name] = fast[name] - lr * g
    model.load_state_dict(fast)


# ── Training loops ────────────────────────────────────────────────────────────
def train_pretrain(
    model: nn.Module,
    lr: float = META_LR,
    n_epochs: int = N_EPOCHS,
    meta_batch: int = META_BATCH,
    log_every: int = 500,
) -> list:
    opt    = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(n_epochs):
        opt.zero_grad()
        total = torch.zeros(())
        for _ in range(meta_batch):
            (sx, sy), _ = sample_sinusoid_task()
            total = total + F.mse_loss(model(sx), sy)
        loss = total / meta_batch
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (epoch + 1) % log_every == 0:
            print(f"  [pretrain]  epoch {epoch+1:4d}  loss={losses[-1]:.4f}")
    return losses


def train_maml(
    model: nn.Module,
    inner_lr: float = INNER_LR,
    inner_steps: int = INNER_STEPS,
    meta_lr: float = META_LR,
    n_epochs: int = N_EPOCHS,
    meta_batch: int = META_BATCH,
    first_order: bool = False,
    use_old: bool = False,
    log_every: int = 500,
) -> list:
    opt    = optim.Adam(model.parameters(), lr=meta_lr)
    losses = []
    tag    = "old" if use_old else ("fomaml" if first_order else "maml")

    for epoch in range(n_epochs):
        tasks = [sample_sinusoid_task() for _ in range(meta_batch)]
        opt.zero_grad()

        if use_old:
            orig  = copy.deepcopy(model.state_dict())
            total = 0.0
            for (sx, sy), (qx, qy) in tasks:
                _old_inner_loop(model, sx, sy, inner_lr, inner_steps)
                model.load_state_dict(orig)
                l = F.mse_loss(model(qx), qy)
                l.backward()
                total += l.item()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(total / meta_batch)
        else:
            params = dict(model.named_parameters())
            total  = torch.zeros(())
            for (sx, sy), (qx, qy) in tasks:
                adapted = maml_inner_loop(model, params, sx, sy,
                                          inner_lr, inner_steps, first_order)
                total = total + F.mse_loss(functional_forward(model, adapted, qx), qy)
            loss = total / meta_batch
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        if (epoch + 1) % log_every == 0:
            print(f"  [{tag}]  epoch {epoch+1:4d}  loss={losses[-1]:.4f}")

    return losses


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    inner_lr: float = INNER_LR,
    inner_steps: int = EVAL_STEPS,
    n_tasks: int = N_EVAL,
    n_support: int = N_SUPPORT,
    use_old: bool = False,
    first_order: bool = False,
) -> tuple:
    """
    Evaluate mean query MSE after `inner_steps` adaptation steps.
    Returns (mean_mse, standard_error).
    """
    base  = copy.deepcopy(model)
    mses  = []

    for _ in range(n_tasks):
        m = copy.deepcopy(base)
        (sx, sy), (qx, qy) = sample_sinusoid_task(n_support)

        if use_old:
            _old_inner_loop(m, sx, sy, inner_lr, inner_steps)
            with torch.no_grad():
                pred = m(qx)
        else:
            params  = dict(m.named_parameters())
            adapted = maml_inner_loop(m, params, sx, sy,
                                       inner_lr, inner_steps, first_order)
            with torch.no_grad():
                pred = functional_forward(m, adapted, qx)

        mses.append(F.mse_loss(pred, qy).item())

    mean = float(np.mean(mses))
    se   = float(np.std(mses) / np.sqrt(n_tasks))
    return mean, se


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SinusoidRegression MAML benchmark")
    parser.add_argument("--epochs",     type=int,   default=N_EPOCHS)
    parser.add_argument("--meta_batch", type=int,   default=META_BATCH)
    parser.add_argument("--inner_lr",   type=float, default=INNER_LR)
    parser.add_argument("--eval_steps", type=int,   default=EVAL_STEPS)
    parser.add_argument("--seed",       type=int,   default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    conditions = [
        ("Pretrain-only",              dict(mode="pretrain")),
        ("MAML-OLD (broken 2nd order)", dict(mode="maml",  use_old=True)),
        ("MAML-Functional (2nd order)", dict(mode="maml",  first_order=False)),
        ("FOMAML (1st order)",          dict(mode="maml",  first_order=True)),
    ]

    results = {}
    for name, kwargs in conditions:
        print(f"\n{'='*55}")
        print(f" {name}")
        print(f"{'='*55}")
        model = SinusoidNet()
        if kwargs["mode"] == "pretrain":
            train_pretrain(model, n_epochs=args.epochs)
            mse, se = evaluate(model, inner_steps=0)
        else:
            train_maml(model,
                       inner_lr=args.inner_lr,
                       n_epochs=args.epochs,
                       meta_batch=args.meta_batch,
                       first_order=kwargs.get("first_order", False),
                       use_old=kwargs.get("use_old", False))
            mse, se = evaluate(model,
                               inner_lr=args.inner_lr,
                               inner_steps=args.eval_steps,
                               use_old=kwargs.get("use_old", False),
                               first_order=kwargs.get("first_order", False))
        results[name] = (mse, se)
        print(f"  Test MSE = {mse:.4f} ± {se:.4f}")

    print(f"\n{'='*55}")
    print(" FINAL RESULTS (5-shot, 500 test tasks)")
    print(f"{'='*55}")
    print(f"  {'Method':<40}  {'MSE':>8}  {'±SE':>8}")
    print(f"  {'-'*40}  {'-'*8}  {'-'*8}")
    for name, (mse, se) in results.items():
        print(f"  {name:<40}  {mse:8.4f}  {se:8.4f}")


if __name__ == "__main__":
    main()
