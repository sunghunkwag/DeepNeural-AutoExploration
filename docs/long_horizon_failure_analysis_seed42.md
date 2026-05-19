# Long-Horizon Failure Analysis: Seed 42

Source artifacts:

- `results/long_horizon_autonomous_exploration_full_seed42.json`
- `results/external_neural_adapter_full_seed42.json`

## Observed Failure Signals

The seed 42 full long-horizon result was not a clean success. The process trends exposed rollback, residue, robustness, and evaluator calibration failures:

| Signal | Start | End | Delta | Direction |
| --- | ---: | ---: | ---: | --- |
| Rollback rate | 0.8750 | 1.0000 | +0.1250 | worse |
| Residue resolution | 0.1429 | 0.0000 | -0.1429 | worse |
| Hidden-validation robustness | 1.0000 | 0.9361 | -0.0639 | worse |
| Evaluator overfit risk | 0.0000 | 0.5119 | +0.5119 | worse |
| Task-family balance | 0.9376 | 0.9147 | -0.0229 | worse |
| Transfer regression risk | 0.1791 | 0.1552 | -0.0239 | better |

External adapter validation and hidden-validation improved, but heldout-after-freeze regressed:

| External metric | Delta |
| --- | ---: |
| Validation | +0.0071 |
| Hidden validation | +0.0119 |
| Heldout after freeze | -0.0060 |

## Root Causes

Rollback concentration was broad rather than isolated. The largest rollback contributors were `widen_hidden_dim`, `add_residual_block`, `remove_residual_block`, `add_memory_gate`, `wider_residual_stack`, `add_causal_bottleneck`, and `object_binding_adapter`.

Task-family rollback pressure concentrated most heavily in `sequence_prediction`, `object_state_transition`, and `memory_retrieval`, which means the process was not just failing on one synthetic family.

Config changes preceding rollback increases included:

- `search_strategy_config.mutation_penalties.widen_hidden_dim`
- `evaluator_objective_config.rollback_risk_weight`
- `recommended_candidate_count`
- `recommended_generations`

Residue resolution failed for persistent `evaluator-selection failure` and `gradient bottleneck` residues. `architecture-representation bottleneck` reduced but did not eliminate the overall residue-resolution failure.

Evaluator overfit risk rose from 0.0000 to 0.5119. The suspicious weight trajectory was not that hidden validation was ignored entirely; the system did raise hidden-validation weight and lower validation weight, but the changes were not strong or early enough to prevent validation-only selection pressure:

- `validation_loss_weight`: 0.30 -> 0.15
- `hidden_validation_loss_weight`: 0.35 -> 0.59
- `rollback_risk_weight`: 0.14 -> 0.20
- `transfer_score_weight`: 0.18 -> 0.24

Memory guidance also showed mixed behavior. It avoided many patterns later marked helpful, and it reused `search_strategy_config.exploration_temperature` even though that pattern later appeared harmful. This indicates memory trust needed rollback-aware filtering instead of simple helped/harmed replay.

## Repairs Implemented

The repair work adds:

- Rollback-risk budget enforcement.
- Stronger rollback penalties by mutation method.
- Candidate-count and generation reduction when rollback exceeds budget.
- Validation-hidden mismatch guard in candidate selection.
- Evaluator objective rollback when validation improves but hidden validation worsens.
- Residue lifecycle tracking across runs: appeared, persisted, reduced, resolved, reappeared.
- Memory ablation benchmark comparing memory-guided and memory-disabled runs.
- External adapter generalization diagnostics and conservative heldout-after-freeze acceptance.

## Recommended Next Experiment

Run multi-seed repair validation and memory ablation. The repair should only be considered successful if multi-seed averages improve rollback rate and evaluator overfit risk, while residue resolution, hidden-validation robustness, and external heldout-after-freeze do not worsen on average.
