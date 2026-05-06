"""
maml_functional.py
==================
True MAML / FOMAML inner loop using torch.func.functional_call.

Key differences from the original deepcopy-based implementation:

  OLD (broken 2nd order):
    fast = copy.deepcopy(model.state_dict())   # cuts the graph
    model.load_state_dict(fast)                 # new leaf tensors, no grad_fn
    → outer loss cannot differentiate through the adaptation path

  NEW (correct 2nd order):
    adapted = {k: v - lr * grad for ...}        # arithmetic on tensors
    functional_call(model, adapted, x)          # stateless forward
    → adapted params retain grad_fn, 2nd order gradients flow correctly

Usage
-----
    from maml_functional import (
        functional_forward,
        maml_inner_loop,
        maml_outer_loss,
        MetaLearningEngineFunctional,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.func import functional_call
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np
import copy


# ============================================================================
# FUNCTIONAL UTILITIES
# ============================================================================

def functional_forward(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Stateless forward pass through `model` using `params` instead of
    model.parameters().  Gradients flow through `params`.

    Args:
        model:  nn.Module used only for its computational graph structure.
        params: dict mapping parameter name -> tensor (may have grad_fn).
        x:      input tensor.

    Returns:
        model output with gradient path intact through `params`.
    """
    return functional_call(model, params, (x,))


def maml_inner_loop(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    first_order: bool = False,
    loss_fn=F.mse_loss,
) -> Dict[str, torch.Tensor]:
    """
    MAML inner adaptation loop — pure functional, zero deepcopy.

    After k steps the returned `adapted_params` satisfy:
        theta' = theta - lr * grad_theta L_support(f_theta)
    where each step builds on the previous via tensor arithmetic, so
    grad_fn is preserved for correct 2nd-order (meta) gradients.

    Args:
        model:       nn.Module (graph structure only — weights NOT mutated).
        params:      meta-parameters {name: tensor}.
        support_x:   task support inputs  [N_s, input_dim].
        support_y:   task support targets [N_s, output_dim].
        inner_lr:    per-step learning rate.
        inner_steps: number of gradient steps.
        first_order: FOMAML mode — detach gradients after each step.
                     Faster but ignores curvature of the adaptation path.
        loss_fn:     task loss function (default: MSE).

    Returns:
        adapted_params: {name: tensor} — retains grad_fn unless first_order.
    """
    adapted = {k: v.clone() for k, v in params.items()}

    for _ in range(inner_steps):
        pred  = functional_forward(model, adapted, support_x)
        loss  = loss_fn(pred, support_y)

        grads = torch.autograd.grad(
            loss,
            list(adapted.values()),
            create_graph=not first_order,
            retain_graph=not first_order,
            allow_unused=True,
        )

        if first_order:
            adapted = {
                k: v - inner_lr * (g.detach() if g is not None else torch.zeros_like(v))
                for (k, v), g in zip(adapted.items(), grads)
            }
        else:
            adapted = {
                k: v - inner_lr * (g if g is not None else torch.zeros_like(v))
                for (k, v), g in zip(adapted.items(), grads)
            }

    return adapted


def maml_outer_loss(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    tasks: List[Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]],
    inner_lr: float,
    inner_steps: int,
    first_order: bool = False,
    loss_fn=F.mse_loss,
) -> torch.Tensor:
    """
    Outer MAML objective: adapt on support set, evaluate on query set.

    Gradient of the returned loss with respect to `params` flows correctly
    through the adaptation path (2nd order) unless first_order=True.

    Args:
        model:        nn.Module.
        params:       meta-parameters {name: tensor}.
        tasks:        list of ((support_x, support_y), (query_x, query_y)).
        inner_lr:     inner-loop step size.
        inner_steps:  inner-loop steps.
        first_order:  FOMAML flag.
        loss_fn:      task loss (default: MSE).

    Returns:
        scalar outer loss (mean over tasks).
    """
    total_loss = torch.zeros((), requires_grad=False)

    for (sx, sy), (qx, qy) in tasks:
        adapted  = maml_inner_loop(model, params, sx, sy,
                                   inner_lr, inner_steps, first_order, loss_fn)
        q_pred   = functional_forward(model, adapted, qx)
        task_loss = loss_fn(q_pred, qy)
        total_loss = total_loss + task_loss

    return total_loss / len(tasks)


# ============================================================================
# DROP-IN REPLACEMENT: MetaLearningEngineFunctional
# ============================================================================

class MetaLearningEngineFunctional:
    """
    Drop-in replacement for MetaLearningEngine that uses the functional
    (torch.func) MAML implementation instead of deepcopy+load_state_dict.

    API-compatible with MetaLearningEngine:
        engine = MetaLearningEngineFunctional(model, config)
        loss   = engine.meta_update(task_batch)
        result = engine.recursive_self_improve(task_batch)

    Key improvements:
        - inner_loop() preserves the computation graph (no load_state_dict)
        - use_second_order=True now actually computes 2nd-order gradients
        - No deepcopy inside the inner loop (only one deepcopy per outer step
          to save original params for the outer optimizer)
        - first_order mode available via config.hessian_free=True
    """

    def __init__(self, model: nn.Module, config):
        self.model  = model
        self.config = config
        self.outer_optimizer = optim.Adam(
            model.parameters(),
            lr=config.meta_learning_rate,
            weight_decay=config.weight_decay,
        )
        self.meta_losses_history: deque = deque(maxlen=1000)
        self.improvement_counter: int   = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inner_loop(
        self,
        task_support: Tuple[torch.Tensor, torch.Tensor],
        task_query:   Tuple[torch.Tensor, torch.Tensor],
        perturb: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Adapt to a single task using the functional inner loop.

        Returns:
            (adapted_params, query_loss_float)
        """
        sx, sy = task_support
        qx, qy = task_query

        params = dict(self.model.named_parameters())
        adapted = maml_inner_loop(
            self.model, params, sx, sy,
            inner_lr    = self.config.inner_lr,
            inner_steps = self.config.inner_steps,
            first_order = self.config.hessian_free,
        )

        with torch.no_grad():
            q_pred     = functional_forward(self.model, adapted, qx)
            query_loss = F.mse_loss(q_pred, qy).item()

        return adapted, query_loss

    def meta_update(
        self,
        task_batch: List[Tuple[
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ]],
        recursion_depth: int = 0,
    ) -> float:
        """
        Single outer MAML update over `task_batch`.

        Gradient flows through adapted params back to meta-params correctly.
        """
        if recursion_depth >= self.config.max_recursion_depth:
            return 0.0

        self.model.train()
        params = dict(self.model.named_parameters())

        self.outer_optimizer.zero_grad()
        outer_loss = maml_outer_loss(
            self.model, params, task_batch,
            inner_lr    = self.config.inner_lr,
            inner_steps = self.config.inner_steps,
            first_order = self.config.hessian_free,
        )
        outer_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.outer_optimizer.step()

        loss_val = outer_loss.item()
        self.meta_losses_history.append(loss_val)
        return loss_val

    def recursive_self_improve(
        self,
        task_batch: List,
        recursion_depth: int = 0,
        improvement_log: Optional[List[dict]] = None,
    ) -> dict:
        """Recursive self-improvement loop (same logic as original)."""
        if improvement_log is None:
            improvement_log = []

        result = {
            "recursion_depth": recursion_depth,
            "meta_loss": 0.0,
            "improvements": [],
            "completed": False,
        }

        if recursion_depth >= self.config.max_recursion_depth:
            result["completed"] = True
            result["message"]   = "Max recursion depth reached."
            return result

        meta_loss = self.meta_update(task_batch, recursion_depth)
        result["meta_loss"] = meta_loss

        improvement_rate = 0.0
        if len(self.meta_losses_history) >= self.config.self_refine_interval:
            recent = list(self.meta_losses_history)[-self.config.self_refine_interval:]
            old    = list(self.meta_losses_history)[:self.config.self_refine_interval]
            old_mean    = np.mean(old)
            recent_mean = np.mean(recent)
            improvement_rate = (old_mean - recent_mean) / (old_mean + 1e-8)

            if improvement_rate > self.config.improvement_threshold:
                result["improvements"].append({
                    "type": "meta_learning",
                    "rate": improvement_rate,
                    "depth": recursion_depth,
                })

        should_recurse = (
            improvement_rate > self.config.improvement_threshold and
            recursion_depth  < self.config.max_recursion_depth
        )

        if should_recurse:
            child = self.recursive_self_improve(
                task_batch, recursion_depth + 1, improvement_log
            )
            result["improvements"].extend(child["improvements"])
            result["completed"] = child["completed"]

        improvement_log.append(result)
        return result
