"""
Test suite for maml_functional.py — true MAML via torch.func.functional_call.
Run with: pytest tests/test_maml_functional.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from maml_functional import (
    functional_forward,
    maml_inner_loop,
    maml_outer_loss,
    MetaLearningEngineFunctional,
)
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer


# ── Helpers ──────────────────────────────────────────────────────────────────

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


@pytest.fixture
def net():
    torch.manual_seed(42)
    return SmallNet()


@pytest.fixture
def cfg():
    return DNAXConfig(
        input_dim=16, hidden_dims=[32, 32], output_dim=8,
        dropout_rate=0.0, spectral_norm=True, verbose=False,
        seed=42, use_second_order=True, hessian_free=False,
        inner_steps=3, inner_lr=0.01, meta_learning_rate=0.001,
        meta_batch_size=4, self_refine_interval=5,
        gradient_clip=1.0, weight_decay=1e-5, max_recursion_depth=5,
        improvement_threshold=0.01,
    )


def make_tasks(n=4, in_dim=16, out_dim=8):
    tasks = []
    for _ in range(n):
        tp = torch.randn(in_dim, out_dim) * 0.15
        sx = torch.randn(6, in_dim);  sy = sx @ tp
        qx = torch.randn(4, in_dim);  qy = qx @ tp
        tasks.append(((sx, sy), (qx, qy)))
    return tasks


# ── T17: functional_forward shape ────────────────────────────────────────────

def test_T17_functional_forward_shape(net):
    params = dict(net.named_parameters())
    x      = torch.randn(5, 16)
    out    = functional_forward(net, params, x)
    assert out.shape == (5, 8)


# ── T18: inner_loop mutates params (adaptation is non-trivial) ───────────────

def test_T18_inner_loop_adapts(net):
    torch.manual_seed(0)
    params  = dict(net.named_parameters())
    orig    = {k: v.clone() for k, v in params.items()}
    sx, sy  = torch.randn(6, 16), torch.randn(6, 8)
    adapted = maml_inner_loop(net, params, sx, sy,
                               inner_lr=0.01, inner_steps=3, first_order=False)
    delta   = sum((adapted[k] - orig[k]).abs().sum().item() for k in orig)
    assert delta > 0, f"inner_loop zero adaptation (delta={delta})"


# ── T19: 2nd order — adapted params HAVE grad_fn (graph preserved) ───────────

def test_T19_second_order_graph_intact(net):
    params  = dict(net.named_parameters())
    sx, sy  = torch.randn(6, 16), torch.randn(6, 8)
    adapted = maml_inner_loop(net, params, sx, sy,
                               inner_lr=0.01, inner_steps=3, first_order=False)
    has_grad_fn = any(v.grad_fn is not None for v in adapted.values())
    assert has_grad_fn, "2nd order: adapted params have no grad_fn — graph broken"


# ── T20: FOMAML — adapted params are DETACHED (no grad_fn) ──────────────────

def test_T20_fomaml_detached(net):
    params  = dict(net.named_parameters())
    sx, sy  = torch.randn(6, 16), torch.randn(6, 8)
    adapted = maml_inner_loop(net, params, sx, sy,
                               inner_lr=0.01, inner_steps=3, first_order=True)
    has_grad_fn = any(v.grad_fn is not None for v in adapted.values())
    assert not has_grad_fn, "FOMAML: adapted params should be detached"


# ── T21: outer loss gradients flow to ORIGINAL meta-params ───────────────────

def test_T21_outer_grad_flows_to_meta_params(net):
    params = dict(net.named_parameters())
    tasks  = make_tasks(4)
    loss   = maml_outer_loss(net, params, tasks,
                              inner_lr=0.01, inner_steps=3, first_order=False)
    loss.backward()
    n_with_grad = sum(1 for p in net.parameters() if p.grad is not None)
    assert n_with_grad == len(list(net.parameters())), \
        f"Only {n_with_grad}/{len(list(net.parameters()))} params received gradient"
    grad_norms = [p.grad.norm().item() for p in net.parameters()]
    assert all(np.isfinite(g) for g in grad_norms), "NaN/Inf in gradients"


# ── T22: MetaLearningEngineFunctional.meta_update() finite + changes weights ─

def test_T22_engine_meta_update(cfg):
    torch.manual_seed(1)
    model  = DeepNeuralAutoExplorer(cfg)
    engine = MetaLearningEngineFunctional(model, cfg)
    before = {k: v.clone() for k, v in model.state_dict().items()
              if v.dtype == torch.float32}
    tasks  = make_tasks(4)
    loss   = engine.meta_update(tasks)
    assert np.isfinite(loss), f"non-finite outer loss: {loss}"
    delta  = sum((model.state_dict()[k] - before[k]).abs().sum().item()
                 for k in before)
    assert delta > 0, "Weights unchanged after meta_update"


# ── T23: deepcopy-based inner_loop BREAKS the graph (regression guard) ───────

def test_T23_old_inner_loop_breaks_graph(net):
    """
    Confirm that the OLD deepcopy+load_state_dict pattern severs grad_fn.
    This test must PASS (i.e., grad_fn IS None after old method).
    It serves as a regression guard documenting why we switched.
    """
    (sx, sy), _ = make_tasks(1)[0]
    fast = copy.deepcopy(net.state_dict())
    net.load_state_dict(fast)
    pred  = net(sx)
    loss  = F.mse_loss(pred, sy)
    grads = torch.autograd.grad(loss, net.parameters(),
                                 create_graph=True, allow_unused=True)
    with torch.no_grad():
        for (name, _), g in zip(net.named_parameters(), grads):
            if g is not None:
                fast[name] = fast[name] - 0.01 * g
    net.load_state_dict(fast)
    # After load_state_dict, params are new leaves — grad_fn must be None
    assert all(p.grad_fn is None for p in net.parameters()), \
        "Unexpected: old method preserved grad_fn — test assumptions changed"
