# Interaction Residue Layer

This repository now includes an English-only integration layer that connects a micro-turn interaction loop to the existing bounded RSI scaffold.

The implementation is intentionally small, deterministic, and CPU-runnable. It does not claim to reproduce a private model, private weights, or a closed architecture. Its purpose is to make interaction failures explicit enough that the existing bounded mechanisms can score them, transform them into evaluator decisions, and route them through failure-oriented improvement logic.

For the stream runtime, adapters, simulator, benchmark, manifest checks, and policy controls, see [Interaction Operating Scaffold](interaction_operating_scaffold.md).

## Implemented files

- `interaction_residue_layer.py` contains the core layer.
- `tests/test_interaction_residue_layer.py` contains executable integration tests.
- `examples/interaction_residue_demo.py` prints a deterministic end-to-end trace.

## Core loop

```text
MicroTurnEvent
-> InteractionPolicy
-> InteractionAction
-> InteractionEvaluator
-> InteractionResidue
-> evaluator-evolution-compatible decision record
```

## Why this is not a shortcut

The layer contains real executable components:

1. `MicroTurnEvent` represents a 200 ms-style observation packet.
2. `InteractionPolicy` chooses listen, backchannel, interrupt, answer, delegate, or deliver-background-result actions.
3. `BackgroundAgentBus` separates fast interaction from slower background reasoning.
4. `InteractionEvaluator` detects bad interrupts, missed visual changes, late background results, idle answers to questions, premature answers, and backchannel overuse.
5. `InteractionResidue` exports failure evidence as structured records.
6. `InteractionResidue.to_evaluator_decision()` converts interaction failures into decision dictionaries compatible with the repository's evaluator-evolution path.

The tests verify policy behavior, failure detection, background delegation, decision export, and compatibility with `EvaluatorGenome` and `score_decision`.

## Architectural role

The new layer should be viewed as an observation/action shell around the bounded RSI loop:

```text
real-time-style interaction stream
-> observable interaction residue
-> evaluator scoring
-> candidate/evaluator pressure
-> bounded validation-only improvement mechanics
```

This preserves the repository's evidence boundary: it is a scaffold for studying closed-loop mechanics, not a claim of general-purpose machine intelligence and not an unbounded self-improvement claim.

## Demo command

```bash
python examples/interaction_residue_demo.py
```

The demo prints JSON containing selected actions, trace metrics, exported evaluator decisions, and final interaction state.
