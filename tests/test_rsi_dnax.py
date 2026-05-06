"""
Test suite for RSI-DNAX core module.
Run with: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import copy
import torch
import torch.nn.functional as F
import numpy as np
from rsi_dnax_core import (
    DNAXConfig, DeepNeuralAutoExplorer, ResidualExplorationBlock,
    MetaLearningEngine, CurriculumManager,
    UncertaintyDrivenExploration, StochasticExploration,
    RSI_DNAX_Driver, count_parameters,
)


@pytest.fixture
def cfg():
    return DNAXConfig(
        input_dim=16, hidden_dims=[32, 32], output_dim=8,
        activation="gelu", dropout_rate=0.0, spectral_norm=True,
        verbose=False, seed=42, use_second_order=False, hessian_free=False,
        inner_steps=2, meta_batch_size=4, exploration_noise_scale=0.01,
        self_refine_interval=5,
    )


@pytest.fixture
def model(cfg):
    torch.manual_seed(42)
    return DeepNeuralAutoExplorer(cfg)


# ── Instantiation ────────────────────────────────────────────────────────────

def test_model_instantiates(cfg):
    m = DeepNeuralAutoExplorer(cfg)
    assert len(m.layers) == 3          # 16→32, 32→32, 32→8
    assert len(m.layer_uncertainties) == 3
    assert count_parameters(m) > 0


# ── Forward shapes ───────────────────────────────────────────────────────────

def test_explore_output_shapes(model):
    x = torch.randn(5, 16)
    pred, unc = model.explore(x, perturb=False)
    assert pred.shape == (5, 8)
    assert unc.shape  == (5, 1)


def test_explore_perturb_records_history(model):
    x = torch.randn(5, 16)
    model.explore(x, perturb=True)
    assert len(model.exploration_history) == 1
    entry = model.exploration_history[0]
    float(entry["uncertainty"])   # must not raise


# ── Residual block ───────────────────────────────────────────────────────────

def test_block_same_dim():
    b = ResidualExplorationBlock(32, 32, dropout=0.0)
    out, unc = b(torch.randn(4, 32))
    assert out.shape == (4, 32)
    assert unc.shape == (4, 1)


def test_block_dim_change():
    b = ResidualExplorationBlock(16, 32, dropout=0.0)
    out, unc = b(torch.randn(4, 16))
    assert out.shape == (4, 32)


def test_exploration_weight_affects_output():
    """exploration_weight must actually gate the perturbation output."""
    torch.manual_seed(0)
    b = ResidualExplorationBlock(32, 32, dropout=0.0)
    x = torch.randn(4, 32)
    # With exploration_weight = -100 → sigmoid ≈ 0 → no perturbation
    with torch.no_grad():
        b.exploration_weight.fill_(-100.0)
    out_no_perturb, _ = b(x, perturb=True, perturb_scale=1.0)
    # With exploration_weight = +100 → sigmoid ≈ 1 → full perturbation
    results = []
    for _ in range(20):
        with torch.no_grad():
            b.exploration_weight.fill_(100.0)
        out_full, _ = b(x, perturb=True, perturb_scale=1.0)
        results.append((out_full - out_no_perturb).abs().mean().item())
    assert max(results) > 1e-3, "exploration_weight has no effect on perturbed output"


def test_uncertainty_scale_affects_uncertainty():
    """uncertainty_scale must modulate the returned uncertainty value."""
    b = ResidualExplorationBlock(32, 32, dropout=0.0)
    x = torch.randn(4, 32)
    with torch.no_grad():
        b.uncertainty_scale.fill_(0.01)
    _, unc_small = b(x)
    with torch.no_grad():
        b.uncertainty_scale.fill_(100.0)
    _, unc_large = b(x)
    assert unc_large.mean() > unc_small.mean(), "uncertainty_scale has no effect"


# ── sample_uncertain mode preservation ──────────────────────────────────────

def test_sample_uncertain_restores_eval(model):
    model.eval()
    model.sample_uncertain(torch.randn(3, 16), n_samples=3)
    assert not model.training


def test_sample_uncertain_restores_train(model):
    model.train()
    model.sample_uncertain(torch.randn(3, 16), n_samples=3)
    assert model.training


# ── MetaLearningEngine ───────────────────────────────────────────────────────

def _make_tasks(cfg, n=4):
    tasks = []
    for _ in range(n):
        tp = torch.randn(cfg.input_dim, cfg.output_dim) * 0.1
        sx = torch.randn(6, cfg.input_dim);  sy = sx @ tp
        qx = torch.randn(4, cfg.input_dim);  qy = qx @ tp
        tasks.append(((sx, sy), (qx, qy)))
    return tasks


def test_inner_loop_modifies_params(cfg, model):
    engine = MetaLearningEngine(model, cfg)
    orig = copy.deepcopy(model.state_dict())
    sx, sy = torch.randn(6, 16), torch.randn(6, 8)
    qx, qy = torch.randn(4, 16), torch.randn(4, 8)
    adapted, q_loss = engine.inner_loop((sx, sy), (qx, qy))
    delta = sum((adapted[k] - orig[k]).abs().sum().item()
                for k in orig if orig[k].dtype == torch.float32)
    assert delta > 0, f"inner_loop produced zero param change (delta={delta})"
    assert isinstance(q_loss, float)


def test_meta_update_finite_loss(cfg, model):
    engine = MetaLearningEngine(model, cfg)
    loss = engine.meta_update(_make_tasks(cfg))
    assert np.isfinite(loss)


def test_meta_update_changes_weights(cfg, model):
    engine = MetaLearningEngine(model, cfg)
    before = {k: v.clone() for k, v in model.state_dict().items() if v.dtype == torch.float32}
    engine.meta_update(_make_tasks(cfg))
    after = model.state_dict()
    delta = sum((after[k] - before[k]).abs().sum().item() for k in before)
    assert delta > 0


# ── CurriculumManager ────────────────────────────────────────────────────────

def test_curriculum_stage_history_bounded(cfg):
    cm = CurriculumManager(cfg)
    for i in range(1000):
        cm.update_difficulty(float(i % 2))
    assert len(cm.stage_history) <= 500


def test_curriculum_difficulty_clamped(cfg):
    cm = CurriculumManager(cfg)
    for _ in range(500):
        cm.update_difficulty(1.0)
    assert cm.current_difficulty <= cfg.max_difficulty


# ── End-to-end smoke ─────────────────────────────────────────────────────────

def test_smoke_10_epochs(cfg):
    torch.manual_seed(42); np.random.seed(42)
    driver = RSI_DNAX_Driver(cfg)
    losses = []
    for _ in range(10):
        log = driver.train_step()
        assert np.isfinite(log["meta_loss"])
        assert np.isfinite(log["val_loss"])
        losses.append(log["meta_loss"])
    # Loss should not explode
    assert max(losses) < 1e6
