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

- **Supported**: Bounded model candidate mutation, training, probing, evaluation, and rollback on validation tasks. The repository allows the user to inspect loops that generate bounded operator programs.
- **Unsupported**: This repository is **not AGI**, not human-level intelligence, and not a technological singularity system. It does not prove open-ended autonomous recursive self-improvement. It is a research scaffold for testing **code-level** mutations and operator programs under verification rules.

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

## 📖 Documentation

For detailed analysis, reports, and plans, refer to:
- [External RSI Evidence Report](docs/rsi_external_evidence_report.md)
- [Research-Goal Generation Integration](docs/research_goal_generation_integration.md)
- [Next Architecture Upgrade Plan](docs/next_architecture_upgrade_plan.md)
- [Deep Neural Autonomous Exploration Upgrade](docs/deep_neural_autonomous_exploration_upgrade.md)
- [Long-Horizon Autonomous Exploration Report](docs/long_horizon_autonomous_exploration_report.md)
- [Long-Horizon Seed-42 Failure Analysis](docs/long_horizon_failure_analysis_seed42.md)
- [Long-Horizon Repair & Memory Ablation Report](docs/long_horizon_repair_and_ablation_report.md)
- [ARC Seed-44 Failure Residue Report](docs/arc_seed44_failure_residue_report.md)
- [Interaction Operating Scaffold](docs/interaction_operating_scaffold.md)
- [Interaction Residue Layer](docs/interaction_residue_layer.md)

---

## 📄 License

Apache License 2.0.
