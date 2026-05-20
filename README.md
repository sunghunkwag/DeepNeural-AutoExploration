# DeepNeural-AutoExploration

An experimental, CPU-runnable research scaffold for bounded closed-loop learning and validation-gated recursive self-improvement.

---

## 🚀 Overview

`DeepNeural-AutoExploration` is a structured sandbox designed to inspect and evaluate bounded neural architecture mutations:
- **Multiprocessing Candidate Evaluation**: Evaluates mutated proposals concurrently in process-isolated workers.
- **Robust Fault Tolerance**: Intercepts training, mathematical instability (NaNs), and dimensional mismatch failures to gracefully roll back failed candidates.
- **Evidence-Guided Deep Search**: Automatically schedules search depth and mutations using representation/gradient probes and rollback histories.
- **Meta-Meta-Meta Controller**: Self-diagnoses the search process and proposes optimized next-run configurations.

---

## 📊 Benchmarks & Snapshot

Selected same-shape ARC-AGI-1 results:

| Benchmark | Mode/seed | Baseline | Evolved | Delta |
| --- | --- | ---: | ---: | ---: |
| ARC cell accuracy | full/42 | `0.668` | `1.000` | `+0.332` |
| ARC exact-grid accuracy | full/42 | `0.000` | `1.000` | `+1.000` |
| ARC exact-grid accuracy | quick/42 | `0.000` | `0.400` | `+0.400` |

### Bounded Claims
- **Supported**: Bounded model candidate mutation, training, probing, evaluation, and rollback on validation tasks.
- **Unsupported**: Not AGI, not a proof of open-ended technological singularity, and not an official ARC leaderboard solver.

---

## 🛠 Setup & Run

### Installation
```bash
git clone https://github.com/sunghunkwag/DeepNeural-AutoExploration.git
cd DeepNeural-AutoExploration
pip install -r requirements.txt pytest
```

### Run Benchmarks

Run core regressions:
```bash
pytest -q
```

Run exploration loops:
```bash
# Autonomous neural exploration loop
python benchmarks/autonomous_neural_exploration_loop.py --mode smoke --seed 42

# Recursive self-improvement (RSI)
python benchmarks/recursive_self_improvement_benchmark.py --mode smoke --seed 42

# Long-horizon neural exploration
python benchmarks/long_horizon_autonomous_exploration.py --mode full --seed 42 --runs 10
```

---

## 📂 Repository Structure

- `neural_search/`: Bounded architecture mutations, weight inheritance, task families, and activation/gradient probes.
- `failure_residue/`: Structured failure residue extraction and leak refusal checks.
- `meta_rsi/`: Orchestrates experiment plans, registry updates, and evaluator repairs.
- `world_model_v2/`: Object-centric state representations, causal graphing, and rollouts.
- `benchmarks/` & `tests/`: Bounded loop execution runners and regression tests.

---

## 📄 License

Apache License 2.0.
