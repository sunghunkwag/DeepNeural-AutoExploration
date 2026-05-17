# DeepNeural-AutoExploration

DeepNeural-AutoExploration is an **experimental, CPU-runnable research scaffold** for studying bounded closed-loop learning and code-level/operator-program recursive self-improvement mechanics. It is **not AGI**, not human-level intelligence, not a technological singularity system, and not a proof of true autonomous or open-ended recursive self-improvement. The repository combines the original RSI-DNAX / functional MAML prototype with learned context inference, non-leaking episodic memory, learned controller-based operator selection, executable validation-gated operator mutations, a bounded operator-program DSL, sandboxed candidate evaluation, a candidate-effect self-model, reusable failure grammar, probationary evaluator evolution, rollback, ablations, and manifests.

## What this repository currently supports

Supported claims:

- It provides a **closed-loop cognitive prototype** around a MAML-based adaptation module.
- It includes a **world-model planning prototype** trained on synthetic latent transitions.
- It includes a bounded, similarity-based episodic memory that refuses records marked as containing query/test targets.
- It includes procedural benchmarks beyond sinusoid regression and explicit anti-cheat tests.
- It exposes AGI-relevant metrics such as adaptation improvement, OOD gap, world-model error, memory precision, planning success, rollback count, and validation-vs-test gap.
- It can generate bounded operator programs, validate them, accept robust improvements, reject/roll back failures, reuse accepted candidates, and record manifests under validation-only and anti-cheat constraints.
- It can learn a bounded self-model over validation-only candidate outcomes, compress failures into future-generation rewrite rules, and evolve acceptance evaluators only after adversarial checks.
- It now uses hidden OOD validation gates, robust generalization scoring, and guarded learned-selector deployment so accepted RSI changes must survive transfer-oriented checks rather than only improving visible validation loss.
- It includes a bounded interaction scaffold for micro-turn simulation, JSONL trace replay, interaction residue extraction, evaluator-evolution bridge records, CPU-runnable interaction benchmarks, and manifest-backed anti-cheat controls.

Unsupported claims:

- This is not AGI.
- This does not demonstrate human-level intelligence.
- This does not prove true recursive self-improvement.
- This does not demonstrate autonomous open-ended recursive self-improvement.
- This does not demonstrate open-ended autonomous recursive self-improvement.
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
- `operator_dsl.py`
  - A bounded operator-program DSL with gradient, memory, world-model, objective, support-proxy, and schedule primitives that compile into executable adaptation operators.
- `rsi_candidate_generator.py`
  - Deterministic candidate generation through parent mutation, recombination, parameter perturbation, primitive insertion/deletion/replacement, schedule mutation, memory-gate mutation, world-model-gate mutation, and support-proxy mutations.
- `candidate_sandbox.py`
  - Isolated local candidate compilation/evaluation with timeout, exception capture, NaN/inf rejection, exploding-loss rejection, and serialized candidate evidence.
- `self_model.py`
  - A small CPU-runnable model that predicts candidate effects on validation improvement, hidden-validation transfer proxy, runtime cost, instability risk, future candidate quality, controller prediction error, and validation-to-test proxy gap using allowed train/validation evidence only.
- `failure_grammar.py`
  - Compresses rejected candidates into reusable rules such as validation-only overfit, OOD-collapse proxy, memory-gate brittleness, wrong-world-model sensitivity, no behavior difference, exploding/NaN loss, deterministic replay failure, dead-code candidate, and high-validation-gain poor-transfer.
- `evaluator_evolution.py`
  - Evolves bounded evaluator candidates for OOD-transfer penalty, validation-to-test gap penalty, runtime cost penalty, instability penalty, novelty bonus, failure-rule penalty, and self-model uncertainty penalty under probation and adversarial checks.
- `benchmarks/code_level_rsi_benchmark.py`
  - Smoke/quick/full benchmark for bounded code-level RSI mechanics using synthesized operator programs, validation-only acceptance, self-model-guided candidate ranking, failure-grammar candidate rewriting, probationary evaluator evolution, frozen OOD test evaluation, reuse tracking, and manifests.
- `interaction_residue_layer.py`
  - Core micro-turn events, baseline interaction policy, background task bus, evaluator, residues, and evaluator-compatible decision export.
- `interaction_stream_runtime.py`
  - Strictly ordered interaction streams, JSONL recording, deterministic replay, and trace divergence reporting.
- `interaction_adapters.py`
  - Dependency-light text delta, synthetic audio activity, symbolic visual event, and 200 ms-aligned combined micro-turn adapters.
- `background_reasoning_simulator.py`
  - Local deterministic simulator for delayed, missing, or corrupted background results.
- `interaction_policies.py`
  - Baseline, no-op, random, always-interrupt, always-listen, no-visual, and no-background policy variants for controls and ablations.
- `interaction_evaluator_bridge.py`
  - Converts interaction residues into evaluator-evolution-compatible decision records with no-op and random controls.
- `interaction_manifest.py`
  - Records interaction benchmark provenance, required metrics, control policy scores, replay status, config hash, and anti-cheat checks.
- `benchmarks/interaction_residue_benchmark.py`
  - Smoke/quick/full CPU-runnable benchmark for turn-taking, proactive visual events, background delegation, mixed streams, and adversarial controls.
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
-> synthesize bounded operator-program candidates from the DSL
-> rank candidates with a train/validation-only self-model
-> rewrite candidates with reusable failure-grammar rules
-> compile and sandbox candidate programs
-> validate candidate on validation tasks
-> evolve probationary evaluator candidates under adversarial checks
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
- Synthesized operator programs must compile into executable operators and change runtime behavior before they can be accepted.
- Candidate programs are evaluated for acceptance only on validation and hidden-control validation pools, never on held-out test tasks.
- Candidate acceptance now combines visible validation improvement, hidden OOD validation improvement, worst-hidden regression penalties, control margins, and selector guards before deployment.
- Candidate failures, NaN/inf losses, exploding losses, deterministic replay failures, and missing runtime differences are rejected and logged.
- Accepted operator programs must be reusable by later generations.
- Self-model records marked as held-out test split are rejected.
- Failure-grammar rules must alter future generated candidates to count as active.
- Evaluator candidates cannot affect acceptance unless they pass checks against old evaluator behavior, hidden validation, wrong-world-model control, shuffled-memory control, no-op control, and random control.
- Interaction traces must have strictly increasing timestamps for streaming and replay.
- Interaction recorders reject hidden labels and held-out expected answers unless explicitly marked validation-only.
- Interaction benchmarks reject events carrying expected policy actions and compare baseline behavior against no-op, random, always-interrupt, and always-silence controls.
- Interaction benchmark manifests fail if replay fails, controls are missing, required metrics are absent, all control policies score identically, or intentionally bad controls produce zero residues.
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
# Expected after this upgrade: 159 passed
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

## Running the bounded interaction scaffold

The interaction residue demo runs a deterministic micro-turn trace and prints selected actions, evaluation metrics, residue records, evaluator decision records, and final interaction state:

```bash
python examples/interaction_residue_demo.py
```

The CPU-runnable interaction benchmark supports smoke, quick, and full modes:

```bash
python benchmarks/interaction_residue_benchmark.py --mode smoke --seed 42
python benchmarks/interaction_residue_benchmark.py --mode quick --seed 42
python benchmarks/interaction_residue_benchmark.py --mode full --seed 42
```

By default, results are written to:

- `results/interaction_residue_<mode>_seed<seed>.json`
- `results/interaction_residue_<mode>_seed<seed>.manifest.json`

The benchmark reports interaction quality, bad interrupts, missed visual changes, late background results, premature answers, idle question handling, backchannel overuse, residue counts, total residue severity, background delivery rate, deterministic replay status, and control policy scores.

The interaction scaffold is documented in `docs/interaction_operating_scaffold.md`.

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
sample train/validation/hidden-validation/test task families
-> infer support-only task context
-> retrieve train-only non-leaking memory
-> collect train traces from executable operator genes
-> train learned meta-controller and world model on allowed operator traces
-> evaluate executable operator mutation candidates on visible validation and hidden OOD validation tasks
-> accept only statistically consistent robust improvements
-> roll back rejected mutations
-> reuse accepted operator genes in later generations
-> freeze accepted state
-> evaluate held-out OOD test tasks
-> write JSON result and anti-cheat manifest
```

The benchmark reports ablations for no adaptation, functional MAML, learned task encoder only, memory-conditioned adaptation, world-model controller, learned meta-controller, random/fixed/wrong controllers, self-improvement with and without rollback, full recursive loop, wrong-world-model full loop, shuffled-memory full loop, and no-mutation full loop. Operator-level metrics include improvement velocity, operator reuse success, controller prediction error reduction across generations, OOD transfer after accepted mutations, accepted mutation quality, accepted operator count, selector guard fallback count, and dead-code detection.

## Running the bounded code-level RSI benchmark

Smoke mode is intended for fast CPU sanity checks:

```bash
python benchmarks/code_level_rsi_benchmark.py --mode smoke --seed 42
```

Quick and full modes increase seeds, generations, and task counts:

```bash
python benchmarks/code_level_rsi_benchmark.py --mode quick --seed 42
python benchmarks/code_level_rsi_benchmark.py --mode full --seed 42
```

By default, results are written to:

- `results/code_level_rsi_<mode>_seed<seed>.json`
- `results/code_level_rsi_<mode>_seed<seed>.manifest.json`

The code-level benchmark loop is:

```text
initialize accepted operator programs
-> sample train/validation/hidden-validation/test tasks
-> collect train traces from accepted executable programs
-> train learned meta-controller and world model on train traces
-> generate bounded DSL candidate programs from accepted parents
-> compile candidates into executable adaptation operators
-> evaluate candidates in an isolated local sandbox on visible validation and hidden OOD validation tasks
-> reject invalid, unstable, leaking, dead-code, weak, or hidden-regressing candidates
-> accept candidates only when robust generalization score and control gates pass
-> reuse accepted candidate programs in later generations
-> freeze accepted state
-> evaluate held-out OOD test tasks
-> write JSON result and anti-cheat manifest
```

The code-level benchmark reports candidate count, compile count, failed compile count, validation rejected count, accepted program count, rollback count, accepted program reuse count, improvement velocity, improvement acceleration, robust mutation quality, candidate survival rate, validation-to-test gap, OOD transfer after accepted programs, controller prediction error, controller prediction error reduction across generations, world-model prediction error, world-model selection effect, memory-gate effect, selector guard fallback count, wrong-world-model degradation, shuffled-memory degradation, full loop versus no synthesis, full loop versus fixed operators, full loop versus random candidate generator, full loop versus no rollback, test-leak trap status, and dead-code detector status.

The next-layer metrics include self-model prediction error, self-model error reduction, failure-rule count, failure-rule reuse count, candidate quality after failure rules, evaluator candidate count, accepted evaluator count, probation evaluator count, evaluator overfit detector, candidate quality per compute, future candidate quality prediction error, OOD transfer after evaluator evolution, full loop versus no self-model, full loop versus no failure grammar, and full loop versus no evaluator evolution.

Supported claim for this benchmark: the system can generate bounded operator programs, compile and execute them, rank them with a train/validation-only self-model, rewrite them using compressed failure rules, validate them under sandboxed validation-only and hidden OOD controls, evolve probationary evaluators under adversarial checks, reject/roll back failures, accept robust improvements, reuse accepted candidates, evaluate frozen accepted programs on held-out OOD tests, and write audit manifests. Unsupported claim: broad intelligence or autonomous open-ended recursive self-improvement.

## Guarded RSI generalization upgrade

The current RSI upgrade adds real executable behavior and stricter acceptance rather than only wrapping the benchmark surface:

- `operator_dsl.py` includes `support_proxy_step`, an executable support-proxy gradient primitive that can change runtime adaptation behavior.
- `rsi_candidate_generator.py` can synthesize support-proxy mutation and recombination candidates from accepted parents.
- `benchmarks/code_level_rsi_benchmark.py` scores candidates with `robust_generalization_score`, combining visible validation gain, hidden OOD validation gain, and worst-hidden regression penalties.
- `benchmarks/recursive_self_improvement_benchmark.py` adds hidden OOD validation splits to recursive mutation acceptance and ablation evaluation.
- Learned selector deployment is guarded by a required relative improvement over active/fixed selectors; otherwise the benchmark falls back instead of deploying a learned selector that only looks good in-sample.
- `evaluator_evolution.py` can place evaluator candidates on probation when adversarial checks pass and total robust score improves, even if the ranking is unchanged.

Local verification used for this upgrade:

```bash
python -m pytest -q
# 159 passed

python -m pytest tests/test_code_level_rsi.py tests/test_recursive_self_improvement.py tests/test_operator_genome.py -q
# 31 passed

python scripts/verify_code_level_rsi_smoke.py results/code_level_rsi_smoke_actual_upgrade_seed42.json
python scripts/verify_recursive_smoke.py results/recursive_self_improvement_smoke_guarded_seed42.json
```

Observed quick-mode behavior is intentionally conservative: the recursive operator-level benchmark moved from negative to positive OOD transfer in the local comparison, while the code-level benchmark stopped accepting generated programs that failed the hidden OOD gate. That means the repository now favors rejecting weak self-improvement candidates over accepting visible-validation gains that do not generalize.

## External ARC-AGI RSI benchmark adapter

The repository now includes a real external benchmark adapter for the code-level/operator-program RSI loop:

```bash
python benchmarks/arc_external_rsi_benchmark.py --mode smoke --seed 42 --arc-data-dir ../external_arc_agi --no-download
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 42 --arc-data-dir ../external_arc_agi --no-download
```

If `--no-download` is omitted and the ARC data directory is missing, the benchmark downloads the public ARC-AGI-1 repository from `https://github.com/fchollet/ARC-AGI`.

Scope and limits:

- This is an ARC-AGI-1 public-data adapter, not an official private ARC leaderboard run.
- It uses a disclosed same-shape cell-prediction subset because the current benchmark only evaluates fixed-grid cell labels.
- ARC task IDs are split into train, validation, hidden validation, and held-out test groups.
- Candidate programs are generated by the existing `OperatorProgram` RSI loop, evaluated through the existing sandbox and hidden-validation acceptance path, and scored on held-out ARC task IDs only after candidate decisions are frozen.
- The ARC path enables classification loss and ARC-specific candidate generation, including support-only symbolic color, local-pattern, feature-lookup, position-color, and clipped long-horizon adaptation programs. These are executable DSL programs, not post-hoc result wrappers.
- Final ARC evaluation uses support cross-validation: for each held-out task, accepted programs are selected by leave-one-demonstration-out accuracy on that task's public training pairs, then applied to the held-out test input without using test labels.

Seed-42 measurements on the same-shape ARC subset:

| Mode | Filtered ARC tasks | Held-out test tasks | Accepted programs | Baseline cell accuracy | Evolved cell accuracy | Cell accuracy delta | Exact task accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | `154` | `4` | `6` | `0.5911805556` | `0.8768518519` | `+0.2856712963` | `0.0 -> 0.25` |
| quick | `169` | `5` | `9` | `0.6445714286` | `0.7469795918` | `+0.1024081633` | `0.0 -> 0.0` |
| full | `185` | `8` | `11` | `0.6680555556` | `0.7673611111` | `+0.0993055556` | `0.0 -> 0.0` |

Interpretation: the operator/program RSI loop now shows a larger held-out cell-accuracy improvement on the public ARC-AGI-1 same-shape subset, including full mode, and smoke mode reaches nonzero exact-grid success on held-out tasks. It is still not an official ARC leaderboard result and does not yet show exact-grid success in quick/full mode. In some modes symbolic programs can trade cross-entropy loss for better discrete grids, so cell and exact-grid accuracy are the primary proof metrics for this adapter.

Multi-seed ARC measurements over seeds `42, 43, 44`:

| Mode | Mean baseline cell accuracy | Mean evolved cell accuracy | Mean cell delta | Mean exact-grid delta |
| --- | ---: | ---: | ---: | ---: |
| smoke | `0.3991898148` | `0.8106172840` | `+0.4114274691` | `+0.3333333333` |
| quick | `0.4002898028` | `0.8260590239` | `+0.4257692211` | `+0.0666666667` |
| full | `0.3883530773` | `0.8115196078` | `+0.4231665305` | `+0.0416666667` |

The evidence report in `docs/rsi_external_evidence_report.md` records the commands, seeds, sources, and limitations.

## External HumanEval coding benchmark adapter

The repository also includes a narrow external coding benchmark adapter:

```bash
python benchmarks/humaneval_template_rsi_benchmark.py --mode smoke --humaneval-dir ../external_human_eval
python benchmarks/humaneval_template_rsi_benchmark.py --mode quick --humaneval-dir ../external_human_eval
python benchmarks/humaneval_template_rsi_benchmark.py --mode full --humaneval-dir ../external_human_eval
```

It uses OpenAI HumanEval from `https://github.com/openai/human-eval`. The baseline is a no-op completion, and the evolved side is a deterministic public-task template library. Canonical solutions are ignored and never executed. This benchmark demonstrates external coding-harness integration and functional-correctness measurement; it is not an LLM code-generation result and must not be read as evidence of general coding intelligence.

Measured HumanEval pass@1:

| Mode | Tasks | Baseline pass@1 | Evolved pass@1 | Delta |
| --- | ---: | ---: | ---: | ---: |
| smoke | `20` | `0.0` | `1.0` | `+1.0` |
| quick | `40` | `0.0` | `1.0` | `+1.0` |
| full | `164` | `0.0` | `1.0` | `+1.0` |

## Epistemic Instrument Evolution (EIE)

Epistemic Instrument Evolution means bounded mutation of the instruments used to observe failures, evaluate candidates, define experiments, generate operators, and version the problem space. In this repository, EIE is implemented as an explicit core layer with failure residue ledgers, observation channels, evaluator specs, experiment units, operator-generator specs, mutation contracts, problem-space version graphs, deterministic diagnosis, validation-only acceptance, rollback, and manifest checks.

Supported claim:

This repository implements a bounded Epistemic Instrument Evolution scaffold: failure residues can trigger controlled mutations to observation channels, evaluators, experiment units, operator generators, and problem-space versions under validation-only and anti-cheat controls.

Unsupported claim:

This does not demonstrate AGI, human-level intelligence, open-ended autonomous recursive self-improvement, or a technological singularity system.

Smoke command:

```bash
python benchmarks/eie_instrument_evolution_benchmark.py --mode smoke --seed 42
```

Verification command:

```bash
python scripts/verify_eie_smoke.py results/eie_instrument_evolution_smoke_seed42.json
```

## RSI Orchestrator Core MVP

Supported claim:

This repository includes a bounded typed integration scaffold that routes candidate artifacts through a system graph, converts failed evaluations into EIE failure residues, records accepted/rejected lineage, and stores decisions in a structural memory layer.

Unsupported claim:

This does not demonstrate AGI, ASI, autonomous open-ended recursive self-improvement, or a technological singularity system.

Minimal executable check:

```bash
python benchmarks/orchestrator_integration_smoke.py --mode smoke --seed 42
python scripts/verify_orchestrator_smoke.py results/orchestrator_integration_smoke_seed42.json
```

Full validation:

```bash
python benchmarks/orchestrator_integration_smoke.py --mode full --seed 42
python scripts/verify_orchestrator_smoke.py results/orchestrator_integration_full_seed42.json
```

Full validation is required before treating an integration change as complete. Minimal executable checks only prove that the pipeline runs; they do not prove the integration is robust.

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
- Self-improvement candidates are bounded and executable, but still limited to a small DSL of adaptation-operator programs and CPU-sized procedural tasks.
- The self-model and evaluator evolution are small supervised/probationary components trained from benchmark traces, not general self-understanding or autonomous research ability.
- The interaction scaffold uses synthetic scalar and symbolic streams; it does not use microphones, webcams, external services, or live multimodal models.
- Metrics are scaffold diagnostics, not evidence of general intelligence.
- The recursive benchmark tests controlled mechanics and leakage resistance; it does not demonstrate open-ended self-improvement.

Recommended next upgrades:

1. Replace hand-built task embeddings with a learned context encoder trained across all procedural families.
2. Couple world-model uncertainty to exploration action selection.
3. Add held-out compositional task families and longer-horizon nonstationary sequence tasks.
4. Expand the bounded operator-program DSL and add stricter multi-seed statistical tests.
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

### Bounded code-level/operator-program synthesis

- `operator_dsl.py` defines `OperatorProgram`, `PrimitiveStep`, `ProgramGenome`, and a compiler from bounded DSL programs into executable `AdaptationOperator`-compatible objects.
- Primitive categories include gradient updates, memory gates, world-model gates, support-only objective signals, and learning-rate schedules.
- `rsi_candidate_generator.py` mutates and recombines accepted programs without using held-out test results.
- `candidate_sandbox.py` evaluates generated programs in isolated temporary candidate directories and rejects exceptions, timeouts, NaN/inf losses, exploding losses, deterministic replay failures, and candidates that do not change runtime behavior.
- `benchmarks/code_level_rsi_benchmark.py` evaluates whether accepted generated programs are reused across later generations and whether later generations improve candidate discovery speed or quality.

### Self-model, failure grammar, and evaluator evolution

- `self_model.py` predicts bounded candidate effects from validation-allowed records and logs predicted-versus-actual effects per generation.
- `failure_grammar.py` turns rejected candidates into rewrite rules that modify future generated candidates.
- `evaluator_evolution.py` mutates evaluator scoring rules, keeps accepted evaluator candidates probationary, and rejects evaluators that fail adversarial checks.
- These mechanisms are part of the benchmark execution path, but remain bounded and audit-oriented. They do not make the system open-ended or autonomous.

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

### Code-level RSI benchmark commands

```bash
python benchmarks/code_level_rsi_benchmark.py --mode smoke --seed 42
python benchmarks/code_level_rsi_benchmark.py --mode quick --seed 42
python benchmarks/code_level_rsi_benchmark.py --mode full --seed 42
python scripts/verify_code_level_rsi_smoke.py results/code_level_rsi_smoke_seed42.json
```

Supported claim for this benchmark: the system can run a bounded code-level/operator-program loop that synthesizes executable adaptation programs from a DSL, evaluates them in a sandbox on validation splits, accepts or rejects them with statistical and anti-cheat checks, rolls back failures, reuses accepted programs, freezes accepted state, evaluates held-out OOD test tasks, and records JSON/manifests. Unsupported claim: autonomous open-ended recursive self-improvement.

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
- It includes a bounded code-level/operator-program loop where generated programs can change the adaptation algorithm itself within a fixed DSL.
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
- Operator-program synthesis is executable but bounded; it is not arbitrary code generation, unconstrained architecture search, or open-ended code synthesis.
- Benchmarks are CPU-sized and procedural, so they test mechanics and leakage resistance rather than broad real-world intelligence.
- Memory transfer is similarity/reward conditioned and can still be brittle under large distribution shifts.

### Next recommended upgrade path

- Train the task encoder and world model jointly over longer task curricula.
- Add uncertainty-aware ensembles for world-model planning and mutation risk estimation.
- Add richer memory consolidation with train/validation/test access policies enforced at the data-structure level.
- Extend procedural tasks toward compositional sequence/action problems where planning horizon and memory retrieval are both necessary.
- Add stronger statistical acceptance tests for self-improvement across more seeds and task families.
