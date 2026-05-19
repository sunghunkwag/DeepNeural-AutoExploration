# DeepNeural-AutoExploration

DeepNeural-AutoExploration is an experimental, CPU-runnable research scaffold for bounded closed-loop learning and validation-gated code-level operator/program recursive self-improvement mechanics.

It is not AGI, not human-level intelligence, not a technological singularity system, and not a proof of open-ended autonomous recursive self-improvement. The repository is meant to make bounded improvement loops inspectable: generate bounded operator programs, validate them on non-test splits, reject or roll back failures, freeze accepted state, and report held-out benchmark results.

## Current Snapshot

The strongest current external result is on the public ARC-AGI-1 same-shape subset:

| Benchmark | Mode/seed | Baseline | Evolved | Delta |
| --- | --- | ---: | ---: | ---: |
| ARC cell accuracy | full/42 | `0.6680555556` | `1.0000000000` | `+0.3319444444` |
| ARC exact-grid accuracy | full/42 | `0.0` | `1.000` | `+1.000` |
| ARC exact-grid accuracy | quick/42 | `0.0` | `0.4` | `+0.4` |

Across full seeds `42, 43, 44`, the latest oriented-gap exact-repair extension raises mean held-out cell accuracy from the previous `0.8753159041` to `0.9305936819` and mean exact-grid accuracy from `0.3333333333` to `0.4583333333`. Seed `44` still has `0.0` exact-grid accuracy, so this is a bounded benchmark gain, not a solved ARC system.

The HumanEval adapter is a template-based harness check, not a general code-generation result. See [docs/rsi_external_evidence_report.md](docs/rsi_external_evidence_report.md) for protocol details, limitations, and commands.

Current full regression check:

```bash
python -m pytest -q
# 259 passed
```

The autonomous neural exploration loop now includes an evidence-guided deep search layer and a post-freeze meta-meta-meta search controller. The deep search layer uses failure residues, representation and gradient probes, validation/hidden-validation trends, rollback reasons, architecture complexity, and module contribution signals to select bounded mutation depth and mutation families. The meta-meta-meta layer diagnoses the search process itself, proposes bounded next-run search/evaluator/curriculum config changes, runs the next experiment, attributes which config changes likely helped or harmed, evaluates decision quality, records bounded JSON config memory, and emits explicit next-run trust/revise/revert recommendations. Candidate selection remains validation/hidden-validation gated, and held-out test labels are evaluated only after candidate freeze.

## Main Components

- `rsi_dnax_core.py`: original DNAX exploration model.
- `maml_functional.py`: functional MAML inner/outer loop.
- `cognitive_core.py`: task inference, memory, world model, planner, and bounded self-improvement controller.
- `adaptation_operators.py`: executable adaptation operators and operator genomes.
- `operator_dsl.py`: bounded operator-program DSL, including ARC support-only symbolic primitives.
- `rsi_candidate_generator.py`: deterministic candidate mutation and recombination.
- `candidate_sandbox.py`: isolated validation execution and failure capture.
- `self_model.py`: candidate-effect prediction on allowed evidence.
- `failure_grammar.py`: rejected-candidate rules reused in later generations.
- `evaluator_evolution.py`: probationary evaluator mutation under adversarial checks.
- `research_goal_controller.py`: benchmark-evidence to research-goal generation and meta-meta goal-policy search.
- `neural_search/`: explicit architecture genomes, deterministic bounded mutations, weight inheritance, harder multi-domain task families, representation/gradient probes, deep search depth scheduling, mutation credit assignment, mutation policies, search history, post-freeze search-process diagnosis, bounded next-run config mutation, and validation-only neural candidate evaluation.
- `failure_residue/`: structured failure residue extraction with held-out leakage refusal and missing-module inference.
- `world_model_v2/`: object-centric state, causal graph, intervention, counterfactual rollout, and uncertainty decomposition utilities.
- `meta_rsi/`: higher-level bounded orchestration over residues, neural search candidates, search-space registry updates, experiment plans, evaluator repair proposals, and audit reports.
- `orchestrator_core/`: typed artifact routing and integration scaffold.
- `interaction_*`: bounded interaction-residue simulation and evaluation tools.

## Benchmarks

Core regression:

```bash
python -m pytest -q
```

ARC external adapter:

```bash
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 42 --arc-data-dir ../external_arc_agi --no-download
python benchmarks/arc_external_rsi_benchmark.py --mode full --seed 42 --arc-data-dir ../external_arc_agi --no-download
```

Code-level RSI:

```bash
python benchmarks/code_level_rsi_benchmark.py --mode smoke --seed 42
python benchmarks/code_level_rsi_benchmark.py --mode quick --seed 42
```

Recursive operator-level RSI:

```bash
python benchmarks/recursive_self_improvement_benchmark.py --mode smoke --seed 42
python benchmarks/recursive_self_improvement_benchmark.py --mode quick --seed 42
```

Research-goal generation:

```bash
python benchmarks/research_goal_generation_benchmark.py --mode smoke --seed 42
```

Multi-domain neural exploration:

```bash
python benchmarks/deep_neural_exploration_benchmark.py --mode smoke --seed 42
python benchmarks/deep_neural_exploration_benchmark.py --mode quick --seed 42
```

Autonomous neural exploration loop:

```bash
python benchmarks/autonomous_neural_exploration_loop.py --mode smoke --seed 42
python benchmarks/autonomous_neural_exploration_loop.py --mode quick --seed 42
```

Meta-meta-meta search-process control:

```bash
python benchmarks/meta_meta_meta_search_benchmark.py --mode smoke --seed 42
```

The meta-meta-meta benchmark writes first-run and second-run outputs, the applied next-run config, strict process-improvement components, config-change attribution, meta-decision quality, estimated counterfactual reports, a bounded meta-config memory JSON file, and a manifest with anti-cheat checks.

Long-horizon autonomous neural exploration and memory ablation:

```bash
python benchmarks/long_horizon_autonomous_exploration.py --mode full --seed 42 --runs 10
python benchmarks/memory_ablation_long_horizon.py --mode full --seeds 42,43,44 --runs 10
python benchmarks/external_neural_adapter_benchmark.py --mode full --seed 42
```

The long-horizon repair path reports rollback-risk budgets, rollback causes, residue lifecycle state, evaluator calibration repairs, memory-guided versus memory-disabled comparison, and conservative external heldout-after-freeze acceptance. Current multi-seed evidence is mixed: evaluator-overfit risk improved on average, but rollback rate, residue resolution, hidden-validation robustness, and external heldout-after-freeze did not satisfy the success criteria.

Other runnable entry points:

```bash
python benchmarks/agi_scaffold_benchmark.py --mode quick --seed 42
python benchmarks/eie_instrument_evolution_benchmark.py --mode smoke --seed 42
python benchmarks/orchestrator_integration_smoke.py --mode smoke --seed 42
python benchmarks/interaction_residue_benchmark.py --mode smoke --seed 42
python examples/next_closed_loop_demo.py
```

## Generated Results Policy

Benchmark JSON and manifest outputs are generated artifacts. They are written under `results/` locally, but `results/` is ignored by Git so the repository does not accumulate large run logs. Keep durable evidence as concise tables in `docs/` or attach full artifacts to GitHub Releases when needed.

## Installation

```bash
git clone https://github.com/sunghunkwag/DeepNeural-AutoExploration.git
cd DeepNeural-AutoExploration
pip install -r requirements.txt
pip install pytest
```

## Repository Layout

- `benchmarks/`: executable benchmark runners.
- `tests/`: regression and anti-cheat tests.
- `docs/`: evidence reports, architecture notes, and integration notes.
- `examples/`: small runnable demos.
- `scripts/`: result verification helpers.
- `orchestrator_core/`: typed integration primitives.
- `results/`: local generated output only, ignored by Git.

## Supported Claims

- Bounded operator/program candidates can be generated, validated, accepted or rejected, rolled back, reused, and evaluated on held-out tasks.
- Bounded neural architecture candidates can be mutated, trained across procedural task families, probed for representation and gradient failures, selected on validation/hidden-validation evidence, rolled back, frozen, and then evaluated on held-out transfer splits.
- The ARC adapter shows measured held-out improvement on a disclosed public same-shape subset.
- The repository includes anti-cheat checks for split leakage, query-target storage, hidden-validation acceptance, deterministic replay, dead code, and control policies.
- The research-goal controller can generate auditable next objectives from benchmark evidence.

## Unsupported Claims

- This is not AGI or quasi-AGI.
- This does not demonstrate human-level intelligence.
- This does not prove open-ended autonomous recursive self-improvement.
- The ARC result is not an official ARC leaderboard score.
- The HumanEval template harness does not prove general coding ability.
- The autonomous neural exploration benchmark uses small procedural task families; it is a research-loop signal, not a general intelligence result.

## Documentation

- [External RSI evidence report](docs/rsi_external_evidence_report.md)
- [Research-goal generation integration](docs/research_goal_generation_integration.md)
- [Next architecture upgrade plan](docs/next_architecture_upgrade_plan.md)
- [Deep neural autonomous exploration upgrade](docs/deep_neural_autonomous_exploration_upgrade.md)
- [Long-horizon autonomous exploration report](docs/long_horizon_autonomous_exploration_report.md)
- [Long-horizon seed-42 failure analysis](docs/long_horizon_failure_analysis_seed42.md)
- [Long-horizon repair and memory ablation report](docs/long_horizon_repair_and_ablation_report.md)
- [ARC seed-44 failure residue report](docs/arc_seed44_failure_residue_report.md)
- [Interaction operating scaffold](docs/interaction_operating_scaffold.md)
- [Interaction residue layer](docs/interaction_residue_layer.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
