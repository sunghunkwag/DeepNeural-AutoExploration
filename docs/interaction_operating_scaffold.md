# Interaction Operating Scaffold

This document describes the bounded interaction scaffold added around the existing interaction residue layer. The system is a CPU-runnable research scaffold for micro-turn simulation, trace replay, interaction residue extraction, evaluator-evolution bridge checks, and benchmark controls.

## Architecture

The scaffold is organized into small modules:

- `interaction_stream_runtime.py` records strictly ordered `MicroTurnEvent` streams, writes JSONL traces, and replays traces through `InteractionResidueLayer`.
- `interaction_adapters.py` converts incremental text, scalar speech activity, and symbolic visual events into 200 ms-aligned micro-turn packets.
- `background_reasoning_simulator.py` simulates slower background work delegated through `BackgroundAgentBus`.
- `interaction_policies.py` provides the baseline wrapper, no-op, random, always-interrupt, always-listen, no-visual, and no-background policies.
- `interaction_evaluator_bridge.py` converts `InteractionResidue` objects into evaluator-evolution-compatible decision records with no-op and random controls.
- `interaction_manifest.py` records benchmark provenance, metrics, controls, and anti-cheat results.
- `benchmarks/interaction_residue_benchmark.py` runs the CPU-runnable benchmark and writes JSON plus manifest files.

## Data Flow

```text
stream adapters
-> InteractionStream
-> InteractionResidueLayer
-> InteractionEvaluator
-> InteractionResidue
-> InteractionEvaluatorBridge
-> evaluator-evolution decision records
```

The recorder can store events, actions, state snapshots, evaluator reports, and background-task state. It rejects hidden labels and held-out expected answers unless the record is explicitly marked validation-only.

## Benchmark Tasks

The benchmark includes these task families:

- Turn-taking tasks: continuing user speech should lead to listen or limited backchannel actions. Bad interrupts are penalized.
- Proactive visual event tasks: symbolic bug, safety, anomaly, participant, or tool events should trigger an interrupt only when confidence and speech state justify it.
- Background delegation tasks: high-uncertainty questions should be delegated, then delivered when the simulated result is ready.
- Mixed stream tasks: text, audio activity, visual events, and background-result readiness are combined in one trace.
- Adversarial controls: shuffled timestamps, non-monotonic timestamps, corrupted visual labels, missing or delayed background results, random actions, no-op actions, always-interrupt actions, and always-silence actions are exercised.

## Anti-Cheat Rules

The manifest validation fails if:

- test labels are used for generation
- event timestamps are non-monotonic
- deterministic replay fails
- no-op or random controls are missing
- required benchmark metrics are missing
- all policies receive identical scores
- intentionally bad controls produce zero residues across all tasks

The benchmark also rejects events that carry `expected_action` labels. Policy decisions are made from observable micro-turn fields, not copied answer keys.

## Supported Claims

The scaffold supports careful claims about:

- a bounded interaction scaffold
- micro-turn simulation
- interaction residue detection
- evaluator-evolution bridge records
- CPU-runnable benchmark execution
- deterministic JSONL replay under fixed inputs
- local anti-cheat checks for controls and leakage

Each supported claim is backed by tests, benchmark output, or manifest validation.

## Unsupported Claims

The scaffold does not support claims about:

- general-purpose machine intelligence
- person-equivalent capability
- reproducing any private lab architecture
- a deployed live multimodal model
- unbounded autonomous self-improvement
- microphone, webcam, browser, external service, GPU, or network-dependent interaction

## Run Commands

```bash
python examples/interaction_residue_demo.py
python benchmarks/interaction_residue_benchmark.py --mode smoke --seed 42
python -m pytest -q tests/test_interaction_residue_layer.py
python -m pytest -q tests/test_interaction_stream_runtime.py
python -m pytest -q tests/test_interaction_adapters.py
python -m pytest -q tests/test_background_reasoning_simulator.py
python -m pytest -q tests/test_interaction_residue_benchmark.py
python -m pytest -q tests/test_interaction_manifest.py
python -m pytest -q tests/test_interaction_policies.py
python -m pytest -q tests/test_interaction_evaluator_bridge.py
```

## Expected Outputs

The benchmark writes:

```text
results/interaction_residue_<mode>_seed<seed>.json
results/interaction_residue_<mode>_seed<seed>.manifest.json
```

For example:

```text
results/interaction_residue_smoke_seed42.json
results/interaction_residue_smoke_seed42.manifest.json
```

The JSON output includes interaction quality, residue counts, background-delivery rate, deterministic replay status, and control policy scores. The manifest records command, timestamp, seed, mode, git commit when available, event/action/residue counts, task families, label-use status, replay status, anti-cheat checks, control scores, and config hash.
