# DeepNeural-AutoExploration

DeepNeural-AutoExploration is now an **experimental AGI-oriented scaffold** for CPU-runnable closed-loop learning experiments. It is **not finished AGI**, not human-level intelligence, and not a proof of autonomous recursive self-improvement. The repository combines the original RSI-DNAX / functional MAML prototype with a new cognitive loop that can infer task context, adapt quickly, store episodic summaries, learn latent transition dynamics, plan over those dynamics, score intrinsic objectives, and validate or roll back controlled self-improvement proposals.

## What this repository currently supports

Supported claims:

- It provides a **closed-loop cognitive prototype** around a MAML-based adaptation module.
- It includes a **world-model planning prototype** trained on synthetic latent transitions.
- It includes a bounded, similarity-based episodic memory that refuses records marked as containing query/test targets.
- It includes procedural benchmarks beyond sinusoid regression and explicit anti-cheat tests.
- It exposes AGI-relevant metrics such as adaptation improvement, OOD gap, world-model error, memory precision, planning success, rollback count, and validation-vs-test gap.

Unsupported claims:

- This is not AGI.
- This does not demonstrate human-level intelligence.
- This does not prove true recursive self-improvement.
- This is not a technological singularity system.
- This is not claimed to be state-of-the-art.

## Architecture

### Original RSI-DNAX components preserved

- `rsi_dnax_core.py` — `DeepNeuralAutoExplorer`, residual exploration blocks, uncertainty-driven perturbation, curriculum manager, and the original driver.
- `maml_functional.py` — `MetaLearningEngineFunctional`, `maml_inner_loop`, `maml_outer_loss`, and `functional_forward` using `torch.func.functional_call`.
- `benchmarks/sinusoid_benchmark.py` — original sinusoid 5-shot regression benchmark.
- Existing tests for `DeepNeuralAutoExplorer`, the original meta-learning engine, and functional MAML remain part of the pytest suite.

### New cognitive scaffold modules

- `cognitive_core.py`
  - `TaskInferenceModule`: infers support/context embeddings without using query targets.
  - `EpisodicMemory`: bounded memory for observations, actions, rewards/losses, task embeddings, and outcomes; rejects records marked as containing query targets.
  - `WorldModel`: learns latent transition dynamics and exposes prediction loss as a self-supervised signal.
  - `Planner`: deterministic short-horizon search over latent states using the world model.
  - `IntrinsicObjectiveSystem`: computes uncertainty reduction, prediction-error reduction, novelty, and improvement rate without optimizing final test scores.
  - `SelfImprovementController`: proposes controlled hyperparameter/operator changes, evaluates candidates on validation tasks, accepts only measured improvements, and rolls back rejected candidates.
  - `CognitiveCore`: coordinates task inference, memory retrieval/storage, MAML adaptation, and validation-gated self-improvement.
- `task_suite.py`
  - Procedural task generation for function adaptation, sequence prediction, latent dynamics, memory retrieval, and planning tasks.
  - Helpers for task-ID disjointness and non-constant function checks.
- `benchmarks/agi_scaffold_benchmark.py`
  - Quick/full benchmark entry point with JSON output and AGI-relevant aggregate metrics.
- `examples/closed_loop_demo.py`
  - Minimal inspectable demo of the full vertical slice.

## Cognitive loop

The intended experimental loop is:

```text
observe task/context
-> infer task embedding from support examples
-> retrieve relevant episodic memories
-> adapt DeepNeuralAutoExplorer with functional MAML
-> train/predict consequences with a latent WorldModel
-> plan/select exploration action over latent states
-> evaluate validation outcome
-> store non-query-leaking memory
-> propose a self-improvement candidate
-> validate candidate on validation tasks
-> accept or reject with rollback
```

This turns the repository from only a regression-adaptation prototype into a small closed-loop generalization testbed.

## Procedural task families

`task_suite.py` procedurally generates non-identical task instances:

### Function family adaptation

- sinusoid
- polynomial
- piecewise linear
- noisy affine
- compositional functions

### Sequence prediction

- repeating patterns
- switching rules
- delayed dependency tasks
- nonstationary transitions

### Latent dynamics

- 1D/2D-style latent transitions
- hidden rule/intervention shifts
- prediction under intervention

### Memory retrieval

- store/query associations
- few-shot recall
- distractor robustness

### Planning

- tiny grid/latent navigation
- short-horizon goal reaching

Train, validation, and test tasks are generated as separate task objects with separate IDs. OOD mode changes the evaluation distribution rather than reusing training instances.

## Anti-cheat safeguards

The repository includes tests and code paths intended to catch common benchmark leakage or fake-improvement failures:

- Train/eval task IDs must not overlap.
- Task inference uses support/context examples only.
- Episodic memory rejects records marked as containing query targets.
- Procedural benchmark functions must not return constant outputs.
- OOD sampling must differ from the training distribution.
- Self-improvement cannot accept a candidate without validation tasks.
- Rejected self-improvement candidates must roll back previous state.
- Deterministic mode must be reproducible.
- Stochastic exploration must vary outputs.
- Planner tests fail if the world model is ignored.
- Memory retrieval tests fail if query similarity is ignored.
- World-model loss must decrease on a simple learnable dynamics task.

## Installation

```bash
git clone https://github.com/sunghunkwag/DeepNeural-AutoExploration.git
cd DeepNeural-AutoExploration
pip install -r requirements.txt
pip install pytest
```

## Running tests

```bash
pytest -q
# Expected after this upgrade: 33 passed
```

## Running the original sinusoid benchmark

```bash
python benchmarks/sinusoid_benchmark.py --seed 42
```

## Running the closed-loop demo

```bash
python examples/closed_loop_demo.py
```

The demo prints transparent logs for task sampling, support-only task inference, functional MAML adaptation, memory retrieval, world-model training, latent planning, and validation-gated self-improvement acceptance/rejection.

## Running the AGI-scaffold benchmark

Quick mode is intended for CPU smoke testing:

```bash
python benchmarks/agi_scaffold_benchmark.py --mode quick --seed 42
```

Full mode runs more tasks and multiple seeds, reports mean and standard error, and writes JSON results:

```bash
python benchmarks/agi_scaffold_benchmark.py --mode full --seed 42
```

By default, results are written to:

- `results/agi_scaffold_quick_seed<seed>.json`
- `results/agi_scaffold_full_seed<seed>.json`

## Reported AGI-relevant metrics

The new benchmark reports more than final MSE:

- adaptation improvement after k steps
- out-of-distribution generalization gap
- task inference proxy loss
- world-model prediction error
- world-model error reduction
- memory retrieval precision
- planning success rate
- exploration efficiency
- self-improvement acceptance rate
- rollback count
- validation-vs-test gap
- baseline-to-adapted test delta
- seed variance in full mode through standard error

## Minimal usage example

```python
from cognitive_core import CognitiveCore, WorldModel
from rsi_dnax_core import DNAXConfig, DeepNeuralAutoExplorer
from task_suite import ProceduralTaskSuite

suite = ProceduralTaskSuite(seed=42)
task = suite.sample_function_task("sinusoid", split="train")
config = DNAXConfig(input_dim=1, hidden_dims=[24, 24], output_dim=1, dropout_rate=0.0, spectral_norm=False)
model = DeepNeuralAutoExplorer(config)
core = CognitiveCore(model, config, WorldModel(latent_dim=2, action_dim=2))
log = core.adapt_and_evaluate(task)
print(log["adaptation_improvement"], log["query_loss_after"])
```

## Current limitations and next upgrade path

Current limitations:

- Task inference is feature-based rather than a learned amortized inference network.
- The world model is small and trained on synthetic dynamics only.
- Planning is exhaustive short-horizon search, not scalable model-predictive control.
- Self-improvement currently modifies a small hyperparameter/operator state rather than synthesizing new architectures.
- Metrics are scaffold diagnostics, not evidence of general intelligence.

Recommended next upgrades:

1. Replace hand-built task embeddings with a learned context encoder trained across all procedural families.
2. Couple world-model uncertainty to exploration action selection.
3. Add held-out compositional task families and longer-horizon nonstationary sequence tasks.
4. Make self-improvement candidates include architecture/operator mutations with stricter statistical tests.
5. Add persistent experiment manifests that cryptographically record train/validation/test task IDs and seeds.

## License

Apache License 2.0. See `LICENSE` for details.

## Next architecture upgrade: connected learned closed loop

This upgrade moves the repository further toward an **AGI-oriented prototype** and **closed-loop learning scaffold** without claiming AGI, human-level intelligence, true recursive self-improvement, or state-of-the-art results.  The important change is that the new modules are connected in the execution path rather than only logged as separate components.

### New learned context encoder

- `learned_task_encoder.py` adds `LearnedTaskEncoder`, a CPU-runnable DeepSets-style **learned context encoder**.
- It encodes support pairs `[support_x_i, support_y_i]`, aggregates pair features with permutation-invariant mean/max pooling, and emits `z_task`.
- It intentionally has no `query_y` argument in the encoding API.
- `TaskConditionedRegressor`, `train_task_encoder`, and `evaluate_task_encoder` expose training loss and validation reconstruction metrics on procedural function families.
- The learned encoder augments the older `TaskInferenceModule`; the older hand-crafted module remains available for compatibility and ablations.

### Model-based update selection

- `model_based_controller.py` adds `ModelBasedController` and typed `UpdateAction` candidates.
- The controller receives `z_task`, a memory summary, and candidate update operators, calls a learned `WorldModel`, ranks candidate updates by predicted improvement, and logs predicted-vs-actual improvement error after execution.
- Candidate actions include inner-loop learning rate, inner steps, memory retrieval `k`, first-order/second-order flag metadata, planner horizon, and exploration noise scale.
- Benchmarks compare the learned world-model controller against a deliberately wrong world model so the benchmark can detect whether planning is doing useful work.

### Memory-conditioned adaptation

- `CognitiveCore` now exposes `memory_conditioned_inner_lr` and `adapt_memory_conditioned`.
- Retrieved non-test episodes influence the inner-loop learning rate through similarity-weighted reward summaries.
- Relevant memory can improve adaptation; corrupted or irrelevant memory is tested to avoid treating insertion order or arbitrary logs as transfer.
- `EpisodicMemory` continues to reject records marked as containing query targets and enforces bounded capacity.

### Controlled operator mutation

- `operator_mutation.py` adds controlled mutation candidates beyond tiny scalar tweaks:
  - inner learning-rate scale changes
  - inner-step changes
  - exploration policy switches
  - memory retrieval `k` changes
  - learned-vs-hand-crafted encoder selection metadata
  - task-encoder on/off metadata
  - planner horizon changes
  - first-order vs second-order MAML metadata
  - residual width multiplier metadata for small models
  - uncertainty and intrinsic objective weighting coefficients
- `OperatorMutationController` snapshots state, evaluates candidates only on validation tasks, accepts only consistent improvements above threshold, rolls back rejected candidates, and preserves accepted/rejected mutation logs.
- Final test tasks are forbidden for mutation acceptance.

### Experiment manifests and anti-cheat checks

- `experiment_manifest.py` writes benchmark/demo manifests with timestamp, git commit, command, seed list, train/validation/test task IDs, task families, OOD flags, accepted/rejected mutations, metric summary, config hash, and anti-cheat checks.
- Manifest validation checks disjoint splits, explicit seeds, no test-split mutation acceptance, no query targets stored in memory, config/task IDs in result metadata, and OOD distribution flags.
- The manifest is intended to make leakage visible rather than to claim the system is impossible to fool.

### Next benchmark commands

```bash
python benchmarks/next_agi_scaffold_benchmark.py --mode smoke --seed 42
python benchmarks/next_agi_scaffold_benchmark.py --mode quick --seed 42
python benchmarks/next_agi_scaffold_benchmark.py --mode full --seed 42
```

Modes:

- `smoke`: very fast CPU sanity check.
- `quick`: at least two seeds with ablations.
- `full`: five seeds with mean, standard error, per-seed results, JSON output, and manifest output.

Required ablations in the next benchmark include no adaptation, functional MAML only, MAML plus learned task encoder, MAML plus memory, MAML plus world-model controller, full loop, full loop without self-improvement, full loop with shuffled memory, and full loop with wrong world model.

### Next demo command

```bash
python examples/next_closed_loop_demo.py
```

The demo prints a transparent vertical slice:

```text
support/context
-> learned task encoder
-> task latent
-> memory retrieval
-> memory-conditioned adaptation
-> world-model prediction of candidate updates/actions
-> controller selects update/action
-> validation evaluates outcome
-> self-improvement accepts/rejects mutation
-> manifest records the run
```

### Supported and unsupported claims

Supported claims:

- This is an **AGI-oriented prototype** for studying closed-loop learning mechanics.
- It includes a learned support-only context encoder.
- It includes model-based update selection using a learned world model.
- It includes memory-conditioned adaptation.
- It includes controlled operator mutation with validation-gated self-improvement candidates.
- It includes manifests and anti-cheat tests intended to expose leakage and ablation failures.

Unsupported claims:

- This is not AGI.
- This does not prove general intelligence.
- This does not demonstrate human-level intelligence.
- This does not achieve true recursive self-improvement.
- This is not a singularity system.
- This is not claimed to be state of the art.

### Current limitations

- The learned task encoder is small and trained with a reconstruction proxy, not a broad task-language or multimodal objective.
- The world model predicts compact latent improvement proxies rather than rich environment dynamics.
- Operator mutations are metadata/config-level and small-model-safe; they are not unconstrained architecture search.
- Benchmarks are CPU-sized and procedural, so they test mechanics and leakage resistance rather than broad real-world intelligence.
- Memory transfer is similarity/reward conditioned and can still be brittle under large distribution shifts.

### Next recommended upgrade path

- Train the task encoder and world model jointly over longer task curricula.
- Add uncertainty-aware ensembles for world-model planning and mutation risk estimation.
- Add richer memory consolidation with train/validation/test access policies enforced at the data-structure level.
- Extend procedural tasks toward compositional sequence/action problems where planning horizon and memory retrieval are both necessary.
- Add stronger statistical acceptance tests for self-improvement across more seeds and task families.
