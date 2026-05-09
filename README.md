# DeepNeural-AutoExploration

DeepNeural-AutoExploration is an **experimental, CPU-runnable research scaffold** for studying bounded closed-loop learning and operator-level recursive self-improvement mechanics. It is **not AGI**, not human-level intelligence, not a technological singularity system, and not a proof of true autonomous recursive self-improvement. The repository combines the original RSI-DNAX / functional MAML prototype with learned context inference, non-leaking episodic memory, learned controller-based operator selection, executable validation-gated operator mutations, rollback, ablations, and manifests.

## What this repository currently supports

Supported claims:

- It provides a **closed-loop cognitive prototype** around a MAML-based adaptation module.
- It includes a **world-model planning prototype** trained on synthetic latent transitions.
- It includes a bounded, similarity-based episodic memory that refuses records marked as containing query/test targets.
- It includes procedural benchmarks beyond sinusoid regression and explicit anti-cheat tests.
- It exposes AGI-relevant metrics such as adaptation improvement, OOD gap, world-model error, memory precision, planning success, rollback count, and validation-vs-test gap.
- It can propose, validate, accept, reject, roll back, and reuse **bounded executable adaptation operators** under validation-only and anti-cheat constraints.

Unsupported claims:

- This is not AGI.
- This does not demonstrate human-level intelligence.
- This does not prove true recursive self-improvement.
- This does not demonstrate autonomous open-ended recursive self-improvement.
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
- `benchmarks/recursive_self_improvement_benchmark.py`
  - Smoke/quick/full recursive loop benchmark with train-only operator traces, validation-only operator mutation acceptance, frozen OOD test evaluation, ablations, JSON output, and manifest output.
- `adaptation_operators.py`
  - `OperatorGenome` and executable `AdaptationOperator` rules: functional MAML, Reptile-style update, memory-gated gradient scaling, and world-model-predicted learning-rate schedule.
- `examples/closed_loop_demo.py`
  - Minimal inspectable demo of the full vertical slice.

## Cognitive loop

The intended experimental loop is:

```text
observe task/context
-> infer task embedding from support examples
-> retrieve relevant episodic memories
-> select an executable adaptation operator from the OperatorGenome
-> adapt DeepNeuralAutoExplorer with functional MAML, Reptile-style update, memory-gated scaling, or world-model LR scheduling
-> train/predict consequences with a latent WorldModel
-> plan/select exploration action over latent states
-> evaluate validation outcome
-> store non-query-leaking memory
-> propose an executable operator mutation
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
- Operator-level self-improvement cannot accept a candidate without validation tasks.
- Rejected self-improvement candidates must roll back previous state.
- Accepted operator mutations must alter runtime adaptation behavior, not only metadata.
- A dead-code guard fails the recursive benchmark if all operators produce identical behavior.
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
# Expected after this upgrade: 47 passed
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

## Running the bounded recursive self-improvement benchmark

Smoke mode is intended for fast CPU sanity checks:

```bash
python benchmarks/recursive_self_improvement_benchmark.py --mode smoke --seed 42
```

Quick and full modes increase seeds, generations, and task counts:

```bash
python benchmarks/recursive_self_improvement_benchmark.py --mode quick --seed 42
python benchmarks/recursive_self_improvement_benchmark.py --mode full --seed 42
```

By default, results are written to:

- `results/recursive_self_improvement_<mode>_seed<seed>.json`
- `results/recursive_self_improvement_<mode>_seed<seed>.manifest.json`

The recursive benchmark loop is:

```text
sample train/validation/test task families
-> infer support-only task context
-> retrieve train-only non-leaking memory
-> collect train traces from executable operator genes
-> train learned meta-controller and world model on allowed operator traces
-> evaluate executable operator mutation candidates on validation tasks only
-> accept statistically consistent validation improvements
-> roll back rejected mutations
-> reuse accepted operator genes in later generations
-> freeze accepted state
-> evaluate held-out OOD test tasks
-> write JSON result and anti-cheat manifest
```

The benchmark reports ablations for no adaptation, functional MAML, learned task encoder only, memory-conditioned adaptation, world-model controller, learned meta-controller, random/fixed/wrong controllers, self-improvement with and without rollback, full recursive loop, wrong-world-model full loop, shuffled-memory full loop, and no-mutation full loop. Operator-level metrics include improvement velocity, operator reuse success, controller prediction error reduction across generations, OOD transfer after accepted mutations, accepted mutation quality, accepted operator count, and dead-code detection.

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
- Self-improvement candidates are bounded and executable, but still limited to small adaptation-operator mutations and CPU-sized procedural tasks.
- Metrics are scaffold diagnostics, not evidence of general intelligence.
- The recursive benchmark tests controlled mechanics and leakage resistance; it does not demonstrate open-ended self-improvement.

Recommended next upgrades:

1. Replace hand-built task embeddings with a learned context encoder trained across all procedural families.
2. Couple world-model uncertainty to exploration action selection.
3. Add held-out compositional task families and longer-horizon nonstationary sequence tasks.
4. Make self-improvement candidates include architecture/operator mutations with stricter statistical tests.
5. Add richer operator-genome search spaces while preserving validation-only acceptance and manifest checks.

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

### Executable operator genome

- `adaptation_operators.py` adds `OperatorGenome`, `OperatorGene`, and executable `AdaptationOperator` rules.
- Current operators are:
  - functional MAML inner-loop adaptation
  - Reptile-style support update and interpolation
  - memory-gated gradient scaling using non-test episodic rewards
  - world-model-predicted learning-rate scheduling
- The recursive benchmark mutates operator genes, evaluates candidates only on validation tasks, accepts only consistent improvements above threshold, rolls back rejected operator genes, and reuses accepted operators in later generations.
- `operator_mutation.py` remains available for bounded config-level mutation experiments, but the recursive benchmark now exercises executable operator-level mutations.
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

### Recursive benchmark commands

```bash
python benchmarks/recursive_self_improvement_benchmark.py --mode smoke --seed 42
python benchmarks/recursive_self_improvement_benchmark.py --mode quick --seed 42
python benchmarks/recursive_self_improvement_benchmark.py --mode full --seed 42
```

Supported claim for this benchmark: the system can run a bounded operator-level recursive loop that trains controllers on allowed traces, evaluates executable adaptation-operator candidates on validation splits, accepts or rejects them with a small statistical rule, rolls back failures, reuses accepted operators, freezes accepted state, evaluates held-out OOD test tasks, and records JSON/manifests with anti-cheat checks. Unsupported claim: autonomous open-ended recursive self-improvement.

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
- It includes a bounded recursive loop where accepted operator mutations change runtime adaptation behavior instead of only metadata.
- It includes manifests and anti-cheat tests intended to expose leakage and ablation failures.

Unsupported claims:

- This is not AGI.
- This does not prove general intelligence.
- This does not demonstrate human-level intelligence.
- This does not achieve true recursive self-improvement.
- This does not demonstrate autonomous open-ended recursive self-improvement.
- This is not a singularity system.
- This is not claimed to be state of the art.

### Current limitations

- The learned task encoder is small and trained with a reconstruction proxy, not a broad task-language or multimodal objective.
- The world model predicts compact latent improvement proxies rather than rich environment dynamics.
- Operator mutations are executable but bounded; they are not unconstrained architecture search or open-ended code synthesis.
- Benchmarks are CPU-sized and procedural, so they test mechanics and leakage resistance rather than broad real-world intelligence.
- Memory transfer is similarity/reward conditioned and can still be brittle under large distribution shifts.

### Next recommended upgrade path

- Train the task encoder and world model jointly over longer task curricula.
- Add uncertainty-aware ensembles for world-model planning and mutation risk estimation.
- Add richer memory consolidation with train/validation/test access policies enforced at the data-structure level.
- Extend procedural tasks toward compositional sequence/action problems where planning horizon and memory retrieval are both necessary.
- Add stronger statistical acceptance tests for self-improvement across more seeds and task families.
