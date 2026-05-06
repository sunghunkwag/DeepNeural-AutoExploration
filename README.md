# DeepNeural-AutoExploration

**Recursive Self-Improving Deep Neural Network Autonomous Exploration Algorithm (RSI-DNAX)**

A PyTorch implementation combining:
- **Recursive self-improvement** through meta-learning
- **Autonomous weight-space exploration** via uncertainty-driven perturbations
- **True MAML** (torch.func.functional_call) for correct 2nd-order meta-gradients
- **Curriculum learning** for progressive difficulty scaling

---

## Benchmark Results

Sinusoid 5-shot regression benchmark ([Finn et al. 2017](https://arxiv.org/abs/1703.03400) protocol),
evaluated on 500 test tasks after 2000 training epochs (seed=42).

| Method | Test MSE | ±SE | Notes |
|---|---:|---:|---|
| Pretrain-only | 2.9441 | 0.1281 | No adaptation at test time |
| MAML-OLD *(broken 2nd order)* | 3.7935 | 0.1893 | `deepcopy + load_state_dict` severs graph |
| **MAML-Functional** *(ours, 2nd order ✓)* | **1.0902** | **0.0485** | `torch.func.functional_call` |
| FOMAML *(ours, 1st order ✓)* | 1.2078 | 0.0600 | `functional_call`, first_order=True |

> **MAML-Functional achieves 63% lower MSE than the deepcopy-based baseline.**  
> See [`results/sinusoid_5shot.md`](results/sinusoid_5shot.md) for full analysis.

---

## Overview

RSI-DNAX autonomously explores the weight space of deep neural networks,
discovering better local minima through iterative self-improvement.
The algorithm treats its own training process as a meta-learning problem,
recursively optimizing how it learns.

### Key Components

| Component | File | Description |
|-----------|------|--------------|
| `DeepNeuralAutoExplorer` | `rsi_dnax_core.py` | Deep residual network with exploratory perturbation and uncertainty estimation |
| `MetaLearningEngine` | `rsi_dnax_core.py` | MAML-style meta-learner with recursive self-improvement |
| `MetaLearningEngineFunctional` | `maml_functional.py` | Drop-in replacement using `torch.func` (correct 2nd order) |
| `CurriculumManager` | `rsi_dnax_core.py` | Progressive difficulty scaling from performance trends |
| `RSI_DNAX_Driver` | `rsi_dnax_core.py` | Main coordination class |
| `ExplorationStrategy` | `rsi_dnax_core.py` | Pluggable strategies (uncertainty-driven, stochastic) |

---

## Architecture

```
+----------------------+     +-----------------------------+
|  RSI_DNAX_Driver     |     | MetaLearningEngineFunctional|
|  (Coordinator)       |<--->| (torch.func MAML + RSI)     |
+----------+-----------+     +-----------------------------+
           |
           v
+-------------------------+
| DeepNeuralAutoExplorer  |
| ResidualExplorationBlock|  <- exploration_weight (learnable gate)
| uncertainty_head        |  <- uncertainty_scale  (learnable modulator)
+----------+--------------+
           |
           +-----> CurriculumManager
           +-----> ExplorationStrategy
```

---

## Installation

```bash
git clone https://github.com/sunghunkwag/DeepNeural-AutoExploration.git
cd DeepNeural-AutoExploration
pip install torch numpy
```

**Requirements:** Python >= 3.9, PyTorch >= 2.0, NumPy >= 1.20

---

## Quick Usage

### With Functional MAML (recommended)

```python
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer
from maml_functional import MetaLearningEngineFunctional

config = DNAXConfig(
    input_dim=64,
    hidden_dims=[128, 256, 128, 64],
    output_dim=16,
    use_second_order=True,   # now actually works correctly
    hessian_free=False,      # True = FOMAML (faster)
    verbose=True,
)

model  = DeepNeuralAutoExplorer(config)
engine = MetaLearningEngineFunctional(model, config)

# Single meta-update step
loss = engine.meta_update(task_batch)
```

### Full Driver

```python
from rsi_dnax_core import RSI_DNAX_Driver, DNAXConfig

driver  = RSI_DNAX_Driver(DNAXConfig(input_dim=64, output_dim=16, verbose=True))
results = driver.recursive_self_improve(n_epochs=500)
print(f"Final loss: {results['final_loss']:.4f}")
```

---

## Project Structure

```
DeepNeural-AutoExploration/
├── rsi_dnax_core.py          # Core architecture + original MetaLearningEngine
├── maml_functional.py        # True MAML via torch.func.functional_call
├── benchmarks/
│   └── sinusoid_benchmark.py # Reproducible 5-shot sinusoid benchmark
├── results/
│   └── sinusoid_5shot.md     # Benchmark results + analysis
├── tests/
│   ├── test_rsi_dnax.py      # 16 unit tests (pytest)
│   └── test_maml_functional.py # 7 MAML-specific tests (T17-T23)
├── requirements.txt
└── LICENSE                   # Apache 2.0
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
# Expected: 23 passed
```

---

## Reproduce Benchmark

```bash
python benchmarks/sinusoid_benchmark.py --seed 42
```

---

## Research Notes

This implementation explores:
- **Meta-learning** — finding better parameter initializations via MAML
- **Correct 2nd-order gradients** — `torch.func.functional_call` preserves
  the computation graph through inner-loop adaptation
- **Neuro-symbolic uncertainty** — per-layer epistemic uncertainty estimation
  with learnable scaling parameters that receive proper gradients
- **Recursive self-improvement** — iteratively improving the learning process

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## Author

**sunghunkwag** — Recursive Self-Improving AI Architectures Research

---

## Citation

```bibtex
@misc{sunghunkwag2026rsidnax,
  author    = {sunghunkwag},
  title     = {RSI-DNAX: Recursive Self-Improving Deep Neural Network
               Autonomous Exploration Algorithm},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/sunghunkwag/DeepNeural-AutoExploration}
}
```
