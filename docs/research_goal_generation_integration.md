# Research Goal Generation Integration

Date: 2026-05-18

This note records the sibling-repository mechanisms inspected for bounded
autonomous research-goal generation and the parts ported into this repository.

## Source Mechanisms Inspected

- `Cognitive-Core-Engine-Test/agi_modules/goal_generator.py`
  - Competence-gap, frontier, creative, and skill-derived goal creation.
  - Useful idea ported: goals must be tied to capability gaps or newly observed
    skills rather than static hardcoded tasks.
- `Cognitive-Core-Engine-Test/agi_modules/transfer_engine.py`
  - Structural transfer records with similarity scores, transfer history, and
    rollback hooks.
  - Useful idea ported: successful mechanisms should become transfer hypotheses
    with explicit similarity and expected-gain fields.
- `Cognitive-Core-Engine-Test/cognitive_core_engine/core/orchestrator.py`
  - Rule proposals, critic evaluation, L1/L2 updates, and skill-derived goals
    inside a recursive cycle.
  - Useful idea ported: the goal loop should distinguish candidate generation,
    acceptance, and later verification.
- `ast-grammar-induction-prototype/rsi_new_engine.py`
  - Level-1 search over system configuration and Level-2 search over the
    meta-search configuration.
  - Useful idea ported: mutate and score the goal-generation policy itself.
- `rsi-bench/rsi_bench/axes/axis4_meta_adaptation.py`
  - Meta-adaptation metrics: adaptation gain, recovery time, shock resilience,
    transfer efficiency.
  - Useful idea ported: transfer is only a hypothesis until measured after a
    distribution shift.
- `rsi-bench/rsi_bench/axes/axis6_goal_generation.py`
  - Goal novelty, feasibility, solve rate, complexity growth, alignment, and
    curriculum monotonicity.
  - Useful idea ported: generated goals need separate audit metrics.
- `SSM-MetaRL-TestCompute` and `MetaRL-Agent-Framework`
  - Meta-RL adaptation and task-distribution protocols.
  - Useful idea ported indirectly: require held-out target splits before
    treating transfer as evidence.

## Ported Components

- `research_goal_controller.py`
  - Normalizes benchmark result JSON into `BenchmarkEvidence`.
  - Generates `ResearchGoal` records from ARC exact-grid gaps, ARC cell-accuracy
    frontiers, candidate-generator rejection residues, template-coding
    limitations, and cross-domain operator-transfer opportunities.
  - Generates `TransferHypothesis` records from successful measured mechanisms.
  - Runs a bounded `MetaMetaGoalController` that mutates `GoalPolicyConfig` and
    accepts a new policy only if its audit score improves.
- `benchmarks/research_goal_generation_benchmark.py`
  - Reads explicit benchmark result files.
  - Writes a JSON report and manifest with anti-cheat notes.
  - Reports generated-goal count, feasible-goal rate, transfer-goal count,
    ARC exact-gap goals, cross-domain goals, and meta-meta score delta.
- `tests/test_research_goal_controller.py`
  - Verifies ARC evidence normalization, exact-repair tag detection, gap goals,
    transfer goals, cross-domain goals, meta-meta scoring, and benchmark output.

## Supported Claim

The repository can now generate bounded, auditable research goals from its own
benchmark evidence and can mutate the goal-generation policy itself under a
small meta-meta scoring loop.

## Unsupported Claim

This does not demonstrate AGI, quasi-AGI, human-level intelligence, or
open-ended autonomous recursive self-improvement. It does not improve ARC or
coding scores by itself. Generated goals and transfer hypotheses must be
implemented and re-benchmarked before they count as capability gains.

## Example Command

```bash
python benchmarks/research_goal_generation_benchmark.py --mode smoke --seed 42 --evidence results/arc_external_rsi_exact_repair_guarded_quick_seed42.json results/arc_external_rsi_exact_repair_guarded_full_seed42.json
```

