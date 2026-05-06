# DeepNeural-AutoExploration

**Recursive Self-Improving Deep Neural Network Autonomous Exploration Algorithm (RSI-DNAX)**

A PyTorch-based implementation of a neuro-symbolic deep learning architecture that combines:
- **Recursive self-improvement** through meta-learning
- **Autonomous weight-space exploration** via uncertainty-driven perturbations
- **MAML-style meta-optimization** for fast adaptation
- **Curriculum learning** for progressive difficulty scaling

---

## Overview

RSI-DNAX is designed to autonomously explore the weight space of deep neural networks, discovering better local minima through iterative self-improvement. The algorithm treats its own training process as a meta-learning problem, recursively optimizing how it learns.

### Key Components

| Component | Description |
|-----------|-------------|
| `DeepNeuralAutoExplorer` | Deep residual network with built-in exploratory perturbation and uncertainty estimation |
| `MetaLearningEngine` | MAML-style meta-learner with recursive self-improvement capability |
| `CurriculumManager` | Progressive difficulty scaling based on model performance trends |
| `RSI_DNAX_Driver` | Main coordination class that ties everything together |
| `ExplorationStrategy` | Pluggable exploration strategies (uncertainty-driven, stochastic) |

---

## Architecture Diagram

```
+----------------------+     +------------------------+
|  RSI_DNAX_Driver     |     |   MetaLearningEngine    |
|  (Coordinator)       |<--->|   (MAML + RSI)          |
+----------+-----------+     +------------------------+
           |
           v
+----------------------+
| DeepNeuralAutoExplorer|
| (Residual Network +    |
|  Uncertainty Heads)    |
+----------+-----------+
           |
           +-----> CurriculumManager
           |
           +-----> ExplorationStrategy
                    (Uncertainty/Stochastic)
```

---

## Installation

```bash
git clone https://github.com/sunghunkwag/DeepNeural-AutoExploration.git
cd DeepNeural-AutoExploration
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- NumPy >= 1.20

```txt
torch>=2.0.0
numpy>=1.20.0
```

---


## Quick Usage

### Basic Demo

```python
from rsi_dnax_core import RSI_DNAX_Driver, DNAXConfig

# Configure the algorithm
config = DNAXConfig(
    input_dim=64,
    hidden_dims=[128, 256, 128, 64],
    output_dim=16,
    activation="gelu",
    exploration_strategy="uncertainty_driven",
    meta_batch_size=8,
    verbose=True,
)

# Initialize and train
driver = RSI_DNAX_Driver(config)
results = driver.recursive_self_improve(n_epochs=500)

print(f"Final loss: {results['final_loss']:.4f}")
print(f"Total improvements: {results['total_improvements']}")
```

### Custom Training Loop

```python
for step in range(1000):
    log = driver.train_step(batch_size=16)
    
    # Access training metrics
    print(f"Step {step}: loss={log['meta_loss']:.4f}, "
          f"bonus={log['exploration_bonus']:.4f}")
```

---

## Configuration

The `DNAXConfig` dataclass provides full control over all algorithmic parameters:

```python
@dataclass
class DNAXConfig:
    # Network architecture
    input_dim: int = 128
    hidden_dims: List[int] = [256, 512, 256, 128]
    output_dim: int = 32
    activation: str = "gelu"  # [relu, gelu, silu, tanh]
    dropout_rate: float = 0.15
    
    # Meta-learning
    meta_learning_rate: float = 0.01
    inner_lr: float = 0.001
    inner_steps: int = 5
    meta_batch_size: int = 16
    
    # Exploration
    exploration_strategy: str = "uncertainty_driven"
    exploration_noise_scale: float = 0.02
    noise_decay: float = 0.995
    min_noise_scale: float = 0.001
    
    # Self-improvement
    improvement_threshold: float = 0.01
    max_recursion_depth: int = 10
    self_refine_interval: int = 100
    use_second_order: bool = True
    hessian_free: bool = True
    
    # Curriculum learning
    curriculum_enabled: bool = True
    curriculum_growth_rate: float = 1.1
    initial_difficulty: float = 0.1
    max_difficulty: float = 1.0
```

---

## Algorithmic Details

### 1. Recursive Self-Improvement Loop

The core innovation is that the training process itself is treated as a meta-learning objective:

```python
# Pseudo-code for the recursive self-improvement loop
def recursive_self_improve(task_batch, depth=0):
    if depth >= MAX_DEPTH:
        return
    
    # Step 1: Standard meta-update (outer loop)
    meta_loss = meta_update(task_batch)
    
    # Step 2: Check improvement rate
    improvement_rate = compute_improvement_rate(history)
    
    # Step 3: Recurse if improving
    if improvement_rate > threshold:
        recursive_self_improve(fresh_tasks, depth + 1)
```

### 2. MAML-Style Meta-Learning

The inner loop adapts to specific tasks using fast gradient descent on a support set, then the outer loop optimizes the initialization to minimize query set loss.

### 3. Uncertainty-Driven Exploration

Each layer estimates its own epistemic uncertainty. The exploration strategy uses this to guide weight perturbations toward uncertain regions of the parameter space.

### 4. Curriculum Learning

Difficulty is scaled progressively based on a polynomial trend analysis of the performance history. The model is never overwhelmed, and plateaus trigger difficulty reduction.

---

## Project Structure

```
DeepNeural-AutoExploration/
├── rsi_dnax_core.py      # Core algorithm implementation
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore patterns
└── LICENSE                # Apache 2.0 License
```

---

## Research Notes

This implementation explores the intersection of:
- **Meta-learning** (finding better parameter initializations)
- **Neuro-symbolic reasoning** (embedding uncertainty estimation in the architecture)
- **Recursive self-improvement** (iteratively improving the learning process itself)
- **Autonomous exploration** (discovering new solutions without supervision)

---

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Author

**sungunhwag**

Recursive Self-Improving AI Architectures Research

---

## Citation

If you use this code in your research, please consider citing:

```bibtex
@misc{sunghunkwag2025rsidnax,
  author = {sungunhwag},
  title = {RSI-DNAX: Recursive Self-Improving Deep Neural Network Autonomous Exploration Algorithm},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/sunghunkwag/DeepNeural-AutoExploration}
}
```
