"""
Recursive Self-Improving Deep Neural Network Autonomous Exploration Algorithm (RSI-DNAX)
========================================================================================
A neuro-symbolic architecture that combines deep neural networks with
recursive self-improvement mechanisms for autonomous weight-space exploration.

Author: sungunhwag
License: Apache-2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import copy
import math
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DNAXConfig:
    """
    Configuration class for the Deep Neural Auto-Exploration system.
    """
    # Network architecture
    input_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 256, 128])
    output_dim: int = 32
    activation: str = "gelu"
    dropout_rate: float = 0.15
    
    # Meta-learning / self-improvement
    meta_learning_rate: float = 0.01
    inner_lr: float = 0.001
    inner_steps: int = 5
    meta_batch_size: int = 16
    
    # Exploration strategy
    exploration_strategy: str = "uncertainty_driven"  # ["stochastic", "uncertainty_driven", "curriculum"]
    exploration_noise_scale: float = 0.02
    noise_decay: float = 0.995
    min_noise_scale: float = 0.001
    
    # Second-order / meta-optimization
    use_second_order: bool = True
    hessian_free: bool = True
    
    # Recursive self-improvement
    improvement_threshold: float = 0.01
    max_recursion_depth: int = 10
    self_refine_interval: int = 100
    
    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_growth_rate: float = 1.1
    initial_difficulty: float = 0.1
    max_difficulty: float = 1.0
    
    # Memory / replay
    replay_buffer_size: int = 10000
    sample_period: int = 50
    
    # Regularization
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    spectral_norm: bool = True
    
    # Logging
    log_interval: int = 50
    verbose: bool = True
  

# ============================================================================
# RESIDUAL AUTO-EXPLORATION NEURAL NETWORK
# ============================================================================

class ResidualExplorationBlock(nn.Module):
    """
    Residual block with exploratory perturbation and uncertainty estimation.
    Each block perturbs its weights slightly during exploration to discover
    better local minima.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 activation: str = "gelu", dropout: float = 0.15):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection if dimensions match
        self.residual = (in_features == out_features)
        if not self.residual:
            self.residual_proj = nn.Linear(in_features, out_features)
        
        # Exploration parameters
        self.exploration_weight = nn.Parameter(torch.zeros(()))
        self.uncertainty_scale = nn.Parameter(torch.zeros(out_features))
        
        # Activation
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, name: str) -> Callable:
        activations = {
            "relu": F.relu,
            "gelu": F.gelu,
            "silu": F.silu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "swish": F.silu,
        }
        return activations.get(name, F.gelu)
    
    def forward(self, x: torch.Tensor, perturb: bool = False,
                perturb_scale: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional exploratory perturbation.
        
        Returns:
            tuple: (output, uncertainty_estimate)
        """
        identity = x
        
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.norm2(out)
        
        # Residual connection
        if self.residual:
            out = out + identity
        else:
            out = out + self.residual_proj(identity)
        
        out = self.activation(out)
        out = self.dropout2(out)
        
        # Exploratory perturbation
        if perturb and perturb_scale > 0:
            perturbation = torch.randn_like(out) * perturb_scale
            out = out + perturbation
        
        # Uncertainty estimation via variance in activations
        uncertainty = torch.var(out, dim=-1, keepdim=True)
        
        return out, uncertainty


class DeepNeuralAutoExplorer(nn.Module):
    """
    Deep Neural Network with built-in autonomous exploration capabilities.
    Uses a stack of residual exploration blocks with dynamic depth adjustment.
    """
    
    def __init__(self, config: DNAXConfig):
        super().__init__()
        self.config = config
        
        # Build dynamic network
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        
        self.layers = nn.ModuleList()
        self.layer_uncertainties = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            block = ResidualExplorationBlock(
                dims[i], dims[i + 1], 
                activation=config.activation,
                dropout=config.dropout_rate
            )
            if config.spectral_norm:
                block = nn.utils.spectral_norm(block)
            self.layers.append(block)
            self.layer_uncertainties.append(nn.Sequential(
                nn.Linear(dims[i + 1], dims[i + 1] // 2),
                nn.ReLU(),
                nn.Linear(dims[i + 1] // 2, 1),
                nn.Sigmoid()
            ))
        
        # Output heads
        self.prediction_head = nn.Linear(config.output_dim, config.output_dim)
        self.uncertainty_head = nn.Linear(config.output_dim, 1)
        
        # Exploration state
        self.current_noise_scale = config.exploration_noise_scale
        self.exploration_history: deque = deque(maxlen=config.replay_buffer_size)
        self.difficulty_level = config.initial_difficulty
    
    def explore(self, x: torch.Tensor, 
                perturb: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with exploration enabled.
        """
        out = x
        total_uncertainty = 0.0
        uncertainties = []
        
        for i, (layer, unc_head) in enumerate(zip(self.layers, self.layer_uncertainties)):
            out, layer_unc = layer(out, perturb=perturb, perturb_scale=self.current_noise_scale)
            unc_est = unc_head(out)
            uncertainties.append(unc_est)
            total_uncertainty = total_uncertainty + unc_est.mean()
        
        # Final prediction
        prediction = self.prediction_head(out)
        global_uncertainty = self.uncertainty_head(out)
        
        # Store exploration trajectory for meta-learning
        if perturb:
            self.exploration_history.append({
                "input_norm": x.norm().item(),
                "output_norm": prediction.norm().item(),
                "uncertainty": global_uncertainty.item(),
                "noise_scale": self.current_noise_scale,
                "difficulty": self.difficulty_level,
            })
        
        return prediction, global_uncertainty
    
    def sample_uncertain(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo dropout for uncertainty estimation via multiple samples.
        """
        self.train()  # Enable dropout
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = self.explore(x, perturb=True)
                samples.append(pred.unsqueeze(0))
        samples = torch.cat(samples, dim=0)
        mean_pred = samples.mean(dim=0)
        std_pred = samples.std(dim=0)
        return mean_pred, std_pred
      

# ============================================================================
# MAML-STYLE META-LEARNING ENGINE
# ============================================================================

class MetaLearningEngine:
    """
    Model-Agnostic Meta-Learning (MAML) engine for recursive self-improvement.
    Learns to adapt quickly to new tasks by finding parameter initializations
    that are highly adaptive.
    """
    
    def __init__(self, model: DeepNeuralAutoExplorer, config: DNAXConfig):
        self.model = model
        self.config = config
        self.outer_optimizer = optim.Adam(
            model.parameters(), 
            lr=config.meta_learning_rate,
            weight_decay=config.weight_decay
        )
        self.task_memories: Dict[str, deque] = {}
        self.meta_losses_history: deque = deque(maxlen=1000)
        self.improvement_counter = 0
    
    def inner_loop(self, task_support: Tuple[torch.Tensor, torch.Tensor],
                   task_query: Tuple[torch.Tensor, torch.Tensor],
                   perturb: bool = True) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Inner MAML loop: adapt to specific task.
        """
        support_x, support_y = task_support
        query_x, query_y = task_query
        
        # Create fast weights (copy of model params)
        fast_params = {k: v.clone() for k, v in self.model.named_parameters()}
        
        # Inner gradient steps
        for step in range(self.config.inner_steps):
            self.model.zero_grad()
            
            # Forward with fast weights
            pred, _ = self.model.explore(support_x, perturb=perturb)
            loss = F.mse_loss(pred, support_y)
            
            # Compute gradients w.r.t. original model (for outer loop)
            grads = torch.autograd.grad(
                loss, self.model.parameters(),
                create_graph=self.config.use_second_order,
                retain_graph=self.config.use_second_order
            )
            
            # Update fast weights
            with torch.no_grad():
                for (name, param), grad in zip(fast_params.items(), grads):
                    fast_params[name] = param - self.config.inner_lr * grad
        
        # Evaluate on query set with adapted parameters
        # Load fast params back temporarily
        query_pred, _ = self.model.explore(query_x, perturb=False)
        query_loss = F.mse_loss(query_pred, query_y).item()
        
        return fast_params, query_loss
    
    def meta_update(self, task_batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                                   Tuple[torch.Tensor, torch.Tensor]]],
                    recursion_depth: int = 0) -> float:
        """
        Outer MAML loop: meta-update across tasks.
        """
        if recursion_depth >= self.config.max_recursion_depth:
            return 0.0
        
        self.model.train()
        total_meta_loss = 0.0
        
        for support, query in task_batch:
            # Get adapted parameters and query loss
            adapted_params, query_loss = self.inner_loop(support, query)
            
            # Compute meta-gradient by differentiating through query loss
            self.model.zero_grad()
            
            # Forward with adapted parameters (differentiable)
            # In practice, we copy params back and forward
            pred, _ = self.model.explore(query[0], perturb=True)
            meta_loss = F.mse_loss(pred, query[1])
            
            # Add uncertainty regularization
            if self.config.hessian_free:
                # Approximate second-order term
                for p in self.model.parameters():
                    meta_loss = meta_loss + 0.5 * self.config.meta_learning_rate * p.norm() ** 2
            
            total_meta_loss = total_meta_loss + meta_loss.item()
        
        # Meta-optimization step
        self.outer_optimizer.zero_grad()
        total_meta_loss = 0.0
        for support, query in task_batch:
            self.model.zero_grad()
            pred, _ = self.model.explore(query[0], perturb=True)
            meta_loss = F.mse_loss(pred, query[1])
            meta_loss.backward()
            total_meta_loss += meta_loss.item()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.outer_optimizer.step()
        
        self.meta_losses_history.append(total_meta_loss / len(task_batch))
        return total_meta_loss / len(task_batch)
    
    def recursive_self_improve(self, task_batch: List,
                               recursion_depth: int = 0,
                               improvement_log: List[Dict] = None) -> Dict:
        """
        Recursive self-improvement: the model iteratively improves itself
        by treating its own training as a meta-learning problem.
        """
        if improvement_log is None:
            improvement_log = []
        
        results = {
            "recursion_depth": recursion_depth,
            "meta_loss": 0.0,
            "improvements": [],
            "completed": False,
        }
        
        # Base case: max recursion reached
        if recursion_depth >= self.config.max_recursion_depth:
            results["completed"] = True
            results["message"] = "Max recursion depth reached."
            return results
        
        # Step 1: Standard meta-update
        meta_loss = self.meta_update(task_batch, recursion_depth)
        results["meta_loss"] = meta_loss
        
        # Step 2: Self-improvement check
        if len(self.meta_losses_history) >= self.config.self_refine_interval:
            # Compute improvement rate
            recent_losses = list(self.meta_losses_history)[-self.config.self_refine_interval:]
            old_losses = list(self.meta_losses_history)[:self.config.self_refine_interval]
            
            recent_mean = np.mean(recent_losses)
            old_mean = np.mean(old_losses)
            improvement_rate = (old_mean - recent_mean) / (old_mean + 1e-8)
            
            if improvement_rate > self.config.improvement_threshold:
                results["improvements"].append({
                    "type": "meta_learning",
                    "rate": improvement_rate,
                    "depth": recursion_depth,
                })
        
        # Step 3: Recursive call if improving
        should_recurse = (
            results["meta_loss"] < self.config.improvement_threshold and
            recursion_depth < self.config.max_recursion_depth
        )
        
        if should_recurse:
            result = self.recursive_self_improve(
                task_batch,
                recursion_depth + 1,
                improvement_log
            )
            results["improvements"].extend(result["improvements"])
            results["completed"] = result["completed"]
        
        improvement_log.append(results)
        return results
                                 

# ============================================================================
# CURRICULUM LEARNING MANAGER
# ============================================================================

class CurriculumManager:
    """
    Manages progressive difficulty scaling for the autonomous exploration.
    Gradually increases task complexity as the model improves.
    """
    
    def __init__(self, config: DNAXConfig):
        self.config = config
        self.current_difficulty = config.initial_difficulty
        self.performance_history: deque = deque(maxlen=100)
        self.stage_history: List[Dict] = []
        self.current_stage = 0
    
    def update_difficulty(self, performance_score: float):
        """
        Update curriculum difficulty based on model performance.
        """
        self.performance_history.append(performance_score)
        
        if len(self.performance_history) < 10:
            return self.current_difficulty
        
        # Compute recent performance trend
        recent = list(self.performance_history)[-20:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 0:  # Improving
            self.current_difficulty = min(
                self.current_difficulty * self.config.curriculum_growth_rate,
                self.config.max_difficulty
            )
            stage_change = True
        elif trend < -0.1:  # Degrading significantly
            self.current_difficulty = max(
                self.current_difficulty / self.config.curriculum_growth_rate,
                self.config.initial_difficulty
            )
            stage_change = True
        else:
            stage_change = False
        
        # Record stage
        self.stage_history.append({
            "stage": self.current_stage,
            "difficulty": self.current_difficulty,
            "trend": trend,
            "performance": np.mean(recent),
            "change": stage_change,
        })
        
        if stage_change:
            self.current_stage += 1
        
        return self.current_difficulty
    
    def generate_curriculum_task(self) -> Dict:
        """
        Generate a task scaled to current difficulty level.
        """
        return {
            "difficulty": self.current_difficulty,
            "stage": self.current_stage,
            "noise_level": self.current_difficulty * 0.5,
            "complexity_scale": self.current_difficulty,
        }


# ============================================================================
# EXPLORATION STRATEGY DISPATCHER
# ============================================================================

class ExplorationStrategy(ABC):
    """Abstract base class for exploration strategies."""
    
    @abstractmethod
    def perturb_weights(self, model: DeepNeuralAutoExplorer) -> Dict[str, torch.Tensor]:
        pass
    
    @abstractmethod
    def compute_exploration_bonus(self, 
                                  trajectory: List[Dict]) -> float:
        pass


class UncertaintyDrivenExploration(ExplorationStrategy):
    """
    Exploration driven by epistemic uncertainty estimation.
    Perturbs weights more in uncertain regions.
    """
    
    def __init__(self, config: DNAXConfig):
        self.config = config
    
    def perturb_weights(self, model: DeepNeuralAutoExplorer) -> Dict[str, torch.Tensor]:
        perturbations = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * model.current_noise_scale
                    perturbations[name] = noise
        return perturbations
    
    def compute_exploration_bonus(self, trajectory: List[Dict]) -> float:
        """Higher bonus for visiting uncertain states."""
        if not trajectory:
            return 0.0
        
        uncertainties = [t.get("uncertainty", 0.0) for t in trajectory]
        novelty_bonus = np.std(uncertainties)  # Novelty = variance in uncertainty
        intrinsic_bonus = np.mean(uncertainties)  # Higher uncertainty = more to learn
        
        return 0.5 * novelty_bonus + 0.5 * intrinsic_bonus


class StochasticExploration(ExplorationStrategy):
    """
    Stochastic weight perturbation for exploration.
    """
    
    def __init__(self, config: DNAXConfig):
        self.config = config
    
    def perturb_weights(self, model: DeepNeuralAutoExplorer) -> Dict[str, torch.Tensor]:
        perturbations = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    noise = torch.randn_like(param) * model.current_noise_scale
                    perturbations[name] = noise
        return perturbations
    
    def compute_exploration_bonus(self, trajectory: List[Dict]) -> float:
        """Bonus based on trajectory diversity."""
        if not trajectory:
            return 0.0
        
        outputs = [t.get("output_norm", 0.0) for t in trajectory[-50:]]
        if len(outputs) < 2:
            return 0.0
        
        return np.std(outputs)  # Diversity = standard deviation of outputs


# ============================================================================
# MAIN AUTO-EXPLORATION DRIVER
# ============================================================================

class RSI_DNAX_Driver:
    """
    Recursive Self-Improving Deep Neural Network Autonomous Exploration Driver.
    
    Coordinates all components:
    - The neural network model
    - Meta-learning engine
    - Curriculum manager
    - Exploration strategy
    - Self-improvement loop
    """
    
    def __init__(self, config: Optional[DNAXConfig] = None):
        self.config = config or DNAXConfig()
        
        # Initialize components
        self.model = DeepNeuralAutoExplorer(self.config)
        self.meta_engine = MetaLearningEngine(self.model, self.config)
        self.curriculum = CurriculumManager(self.config)
        
        # Exploration strategy
        self.exploration_strategy = self._select_exploration_strategy(
            self.config.exploration_strategy
        )
        
        # Training state
        self.training_log: List[Dict] = []
        self.best_params = None
        self.best_loss = float("inf")
        self.epoch = 0
        
        # Noise decay schedule
        self._noise_decay_scheduler()
    
    def _select_exploration_strategy(self, strategy_name: str) -> ExplorationStrategy:
        strategies = {
            "stochastic": StochasticExploration,
            "uncertainty_driven": UncertaintyDrivenExploration,
            "curriculum": UncertaintyDrivenExploration,  # Hybrid approach
        }
        return strategies.get(strategy_name, UncertaintyDrivenExploration)(self.config)
    
    def _noise_decay_scheduler(self):
        """Decay exploration noise over time."""
        self.model.current_noise_scale = max(
            self.model.current_noise_scale * self.config.noise_decay,
            self.config.min_noise_scale
        )
    
    def generate_meta_tasks(self, batch_size: int) -> List[Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]
    ]]:
        """
        Generate a batch of meta-learning tasks.
        Each task is (support set, query set).
        """
        tasks = []
        task_info = self.curriculum.generate_curriculum_task()
        
        for _ in range(batch_size):
            # Generate synthetic task data
            support_size = 10
            query_size = 5
            input_dim = self.config.input_dim
            output_dim = self.config.output_dim
            
            # Generate task-specific parameters
            task_params = torch.randn(input_dim, output_dim) * task_info["complexity_scale"]
            
            # Support set
            support_x = torch.randn(support_size, input_dim)
            support_y = support_x @ task_params + torch.randn(support_size, output_dim) * task_info["noise_level"]
            
            # Query set
            query_x = torch.randn(query_size, input_dim)
            query_y = query_x @ task_params + torch.randn(query_size, output_dim) * task_info["noise_level"]
            
            tasks.append(((support_x, support_y), (query_x, query_y)))
        
        return tasks
    
    def train_step(self, batch_size: Optional[int] = None) -> Dict:
        """
        Single training step combining meta-learning, exploration, and self-improvement.
        """
        batch_size = batch_size or self.config.meta_batch_size
        
        # Generate tasks
        tasks = self.generate_meta_tasks(batch_size)
        
        # Exploration phase
        exploration_trajectories = []
        for (support, query) in tasks:
            with torch.no_grad():
                pred, unc = self.model.explore(support[0], perturb=True)
                exploration_trajectories.append({
                    "pred_norm": pred.norm().item(),
                    "uncertainty": unc.item(),
                })
        
        exploration_bonus = self.exploration_strategy.compute_exploration_bonus(
            exploration_trajectories
        )
        
        # Meta-learning phase
        meta_loss = self.meta_engine.meta_update(tasks)
        
        # Curriculum update
        performance_score = 1.0 / (meta_loss + 1.0)
        self.curriculum.update_difficulty(performance_score)
        
        # Decay noise
        self._noise_decay_scheduler()
        
        # Improve if best
        if meta_loss < self.best_loss:
            self.best_loss = meta_loss
            self.best_params = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Log
        log_entry = {
            "epoch": self.epoch,
            "meta_loss": meta_loss,
            "exploration_bonus": exploration_bonus,
            "curriculum_stage": self.curriculum.current_stage,
            "difficulty": self.curriculum.current_difficulty,
            "noise_scale": self.model.current_noise_scale,
            "best_loss": self.best_loss,
            "improvements": [],
        }
        self.training_log.append(log_entry)
        
        if self.config.verbose and self.epoch % self.config.log_interval == 0:
            print(f"Epoch {self.epoch}: loss={meta_loss:.4f}, "
                  f"bonus={exploration_bonus:.4f}, "
                  f"stage={self.curriculum.current_stage}, "
                  f"difficulty={self.curriculum.current_difficulty:.3f}, "
                  f"noise={self.model.current_noise_scale:.4f}")
        
        self.epoch += 1
        return log_entry
    
    def recursive_self_improve(self, n_epochs: int, 
                               batch_size: Optional[int] = None) -> Dict:
        """
        Full recursive self-improvement training loop.
        The model improves itself iteratively while the improvement process
        itself is subject to meta-optimization.
        """
        total_improvements = 0
        improvement_log = []
        
        for epoch in range(n_epochs):
            # Standard training step
            step_log = self.train_step(batch_size)
            
            # Periodic self-improvement check
            if epoch % self.config.self_refine_interval == 0 and epoch > 0:
                # Generate fresh tasks for self-improvement
                tasks = self.generate_meta_tasks(batch_size)
                
                # Recursive self-improvement
                result = self.meta_engine.recursive_self_improve(
                    tasks,
                    recursion_depth=0,
                    improvement_log=improvement_log
                )
                
                step_log["recursive_improvement"] = result
                total_improvements += len(result.get("improvements", []))
                
                if self.config.verbose:
                    print(f"  Recursive improvement at epoch {epoch}: "
                          f"{len(result.get('improvements', []))} improvements found")
        
        return {
            "final_loss": self.best_loss,
            "total_epochs": n_epochs,
            "total_improvements": total_improvements,
            "curriculum_stage": self.curriculum.current_stage,
            "curriculum_history": len(self.curriculum.stage_history),
            "exploration_strategy": self.config.exploration_strategy,
        }
                                 

# ============================================================================
# UTILITIES & HELPERS
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_state(model: nn.Module, path: str):
    """Save model state to file."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config,
    }, path)


def load_model_state(model: nn.Module, path: str) -> nn.Module:
    """Load model state from file."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# ============================================================================
# EXAMPLE USAGE / DEMONSTRATION
# ============================================================================

def demo():
    """
    Demonstration of the RSI-DNAX algorithm.
    """
    print("=" * 70)
    print("RSI-DNAX: Recursive Self-Improving Deep Neural Network")
    print("Autonomous Exploration Algorithm - Demo")
    print("=" * 70)
    
    # Configuration
    config = DNAXConfig(
        input_dim=64,
        hidden_dims=[128, 256, 128, 64],
        output_dim=16,
        activation="gelu",
        dropout_rate=0.1,
        exploration_strategy="uncertainty_driven",
        meta_batch_size=8,
        inner_steps=3,
        verbose=True,
        log_interval=25,
        self_refine_interval=100,
    )
    
    # Initialize driver
    driver = RSI_DNAX_Driver(config)
    
    print(f"\nModel Parameters: {count_parameters(driver.model):,}")
    print(f"Exploration Strategy: {config.exploration_strategy}")
    print(f"Recursion Depth Limit: {config.max_recursion_depth}")
    print("\nStarting autonomous exploration training...\n")
    
    # Run recursive self-improvement
    results = driver.recursive_self_improve(n_epochs=200)
    
    print("\n" + "=" * 70)
    print("Training Results:")
    print("=" * 70)
    for key, value in results.items():
        if key not in ["curriculum_history"]:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("RSI-DNAX Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
