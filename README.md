# DeepNeural-AutoExploration

PyTorch research implementation for autonomous neural network exploration with MAML-style meta-learning, functional inner-loop adaptation, uncertainty-driven perturbations, and curriculum scheduling.

## Key Files

- `rsi_dnax_core.py` — core model, exploration blocks, curriculum manager, and driver
- `maml_functional.py` — functional MAML/FOMAML implementation using `torch.func.functional_call`
- `benchmarks/sinusoid_benchmark.py` — sinusoid 5-shot regression benchmark
- `results/sinusoid_5shot.md` — benchmark summary
- `tests/test_rsi_dnax.py` — 15 core unit tests
- `tests/test_maml_functional.py` — 7 functional MAML tests

## Installation

```bash
git clone https://github.com/sunghunkwag/DeepNeural-AutoExploration.git
cd DeepNeural-AutoExploration
pip install -r requirements.txt
pip install pytest
```

## Running Tests

```bash
pytest tests/ -v
# Expected: 22 passed
```

## Functional MAML Usage

```python
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer
from maml_functional import MetaLearningEngineFunctional

config = DNAXConfig(
    input_dim=64,
    hidden_dims=[128, 256, 128, 64],
    output_dim=16,
    use_second_order=True,
    hessian_free=False,
    verbose=True,
)

model = DeepNeuralAutoExplorer(config)
engine = MetaLearningEngineFunctional(model, config)
loss = engine.meta_update(task_batch)
```

`DeepNeuralAutoExplorer.forward()` now provides a deterministic entry point for `torch.func.functional_call`, while stochastic exploration remains available through `explore()`.

## Benchmark

```bash
python benchmarks/sinusoid_benchmark.py --seed 42
```

## License

Apache License 2.0. See `LICENSE` for details.
