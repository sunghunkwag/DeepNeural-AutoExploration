# SinusoidRegression 5-shot Benchmark Results

**Protocol:** Finn et al. 2017 (MAML, ICML)  
**Task:** Regress `y = A·sin(x + φ)`, A ~ U(0.1, 5.0), φ ~ U(0, π), x ~ U(−5, 5)  
**Metric:** Mean Squared Error on 100 query points after inner-loop adaptation  
**Evaluation:** 500 test tasks, mean ± standard error  
**Training:** 2000 epochs, meta-batch=25, inner_lr=0.01, 1-step MAML  
**Hardware:** CPU (results reproducible on any machine, seed=42)

---

## Results

| Method | Test MSE | ±SE | Notes |
|---|---:|---:|---|
| Pretrain-only | 2.9441 | 0.1281 | No adaptation at test time |
| MAML-OLD *(broken 2nd order)* | 3.7935 | 0.1893 | `deepcopy + load_state_dict` severs computation graph |
| **MAML-Functional** *(2nd order ✓)* | **1.0902** | **0.0485** | `torch.func.functional_call`, graph intact |
| FOMAML *(1st order ✓)* | 1.2078 | 0.0600 | `functional_call`, first_order=True |

---

## Key Findings

- **MAML-OLD is actually *worse* than Pretrain-only** (3.79 vs 2.94).  
  Root cause: `load_state_dict()` inside the inner loop severs the PyTorch
  computation graph, so `use_second_order=True` silently degrades to an
  incorrect first-order approximation with corrupted gradients.

- **MAML-Functional achieves 63% lower MSE than MAML-OLD** (1.09 vs 3.79)  
  and **63% lower MSE than Pretrain-only** (1.09 vs 2.94).  
  The only change is replacing `deepcopy + load_state_dict` with
  `torch.func.functional_call` so gradients flow correctly.

- **FOMAML is within 11% of full 2nd-order MAML** (1.21 vs 1.09)  
  while being faster to train (no `create_graph=True`).

---

## Reproduce

```bash
git clone https://github.com/sunghunkwag/DeepNeural-AutoExploration.git
cd DeepNeural-AutoExploration
pip install torch numpy
python benchmarks/sinusoid_benchmark.py --seed 42
```
