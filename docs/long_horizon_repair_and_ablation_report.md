# Long-Horizon Repair and Memory Ablation Report

## Commands

```bash
python -m pytest -q
python benchmarks/long_horizon_autonomous_exploration.py --mode full --seed 42 --runs 10
python benchmarks/long_horizon_autonomous_exploration.py --mode full --seed 43 --runs 10
python benchmarks/long_horizon_autonomous_exploration.py --mode full --seed 44 --runs 10
python benchmarks/memory_ablation_long_horizon.py --mode full --seeds 42,43,44 --runs 10
python benchmarks/external_neural_adapter_benchmark.py --mode full --seed 42
python benchmarks/external_neural_adapter_benchmark.py --mode full --seed 43
python benchmarks/external_neural_adapter_benchmark.py --mode full --seed 44
```

Full pytest result:

```text
259 passed in 83.32s
```

## Long-Horizon Results

| Seed | Rollback delta | Residue-resolution delta | Hidden-robustness delta | Evaluator-overfit delta | Transfer-risk delta | Meta-quality delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | +0.0000 | +0.0000 | +0.0000 | +0.0220 | +0.0317 | -0.3889 |
| 43 | +0.0000 | +0.0000 | +0.0000 | -0.0098 | +0.0204 | -0.6667 |
| 44 | +0.0000 | +0.0000 | +0.0000 | -0.0035 | +0.0638 | -0.4167 |
| Mean | +0.0000 | +0.0000 | +0.0000 | +0.0029 | +0.0387 | -0.4907 |

Positive rollback, evaluator-overfit, and transfer-risk deltas are bad. Negative residue-resolution, hidden-robustness, and meta-quality deltas are bad.

## Memory Ablation

Memory-guided versus memory-disabled comparison:

| Metric | Mean delta, positive means memory better |
| --- | ---: |
| Rollback rate | +0.0000 |
| Residue resolution | +0.0000 |
| Hidden-validation robustness | +0.0000 |
| Evaluator overfit risk | +0.0000 |
| Transfer regression risk | +0.0000 |
| External heldout-after-freeze delta | +0.0000 |
| Meta-decision quality | +0.0196 |
| Process improvement delta | +0.0000 |

Memory guidance was slightly helpful for meta-decision quality, but neutral on the major process metrics. It should not be credited as solving rollback, residue resolution, or generalization.

## External Adapter Results

| Seed | Validation delta | Hidden-validation delta | Heldout-after-freeze delta | Acceptance |
| ---: | ---: | ---: | ---: | --- |
| 42 | +0.0000 | +0.0000 | +0.0000 | failed_generalization |
| 43 | +0.0000 | +0.0000 | +0.0000 | failed_generalization |
| 44 | +0.0000 | +0.0000 | +0.0000 | failed_generalization |
| Mean | +0.0000 | +0.0000 | +0.0000 | failed_generalization |

The external adapter repair blocks heldout-after-freeze regression by rejecting all externally suspicious candidates before freeze. This is a safety improvement, not a performance improvement.

## Repair Summary

Implemented rollback repair:

- Rollback-risk budget per run.
- Candidate-count and generation reduction when rollback exceeds budget.
- Stronger mutation penalties for rollback-heavy methods.
- Rollback-coupled validation-hidden mismatch guard.
- Rollback causes by method, task family, and config change.

Implemented residue repair:

- Residue lifecycle states: appeared, persisted, reduced, resolved, reappeared.
- Per-residue associated mutation methods and module families.
- Repair success/failure by mutation method.
- Unresolved-by-current-module-family marking when targeted attempts harm or repeatedly fail.

Implemented evaluator overfit repair:

- Validation-hidden mismatch guard.
- Higher hidden-validation, rollback-risk, task-family regression, and transfer/generalization penalties.
- Selection-score calibration fields.
- Suspicious evaluator mutation reporting.

Implemented external generalization repair:

- Train/validation/hidden/heldout gap table.
- Feature distribution shift estimate.
- Candidate complexity versus heldout delta.
- Mutation method versus heldout delta.
- Conservative external acceptance rule requiring non-negative heldout-after-freeze delta.

## Conservative Success Judgment

The repair is still a partial failure, but the failure mode changed.

Compared with the exposed seed-42 failure signals, rollback and external heldout damage are now controlled. Under strict within-run multi-seed trend criteria, it still does not satisfy all success criteria:

- Rollback rate no longer worsens on average, but it also does not strictly decrease: +0.0000.
- Residue resolution does not worsen: +0.0000.
- Hidden-validation robustness does not worsen: +0.0000.
- Evaluator-overfit risk is nearly controlled but still worsens slightly on average: +0.0029.
- External heldout-after-freeze no longer worsens: +0.0000.
- Transfer regression risk worsens on average: +0.0387.
- Meta-decision quality still declines because the final unevaluated decision has no next-run quality observation.

## Output Paths

Long-horizon outputs:

- `results/long_horizon_autonomous_exploration_full_seed42.json`
- `results/long_horizon_autonomous_exploration_full_seed43.json`
- `results/long_horizon_autonomous_exploration_full_seed44.json`

Long-horizon manifests:

- `results/long_horizon_autonomous_exploration_full_seed42.manifest.json`
- `results/long_horizon_autonomous_exploration_full_seed43.manifest.json`
- `results/long_horizon_autonomous_exploration_full_seed44.manifest.json`

Memory ablation:

- `results/memory_ablation_long_horizon_full.json`
- `results/memory_ablation_long_horizon_full.manifest.json`

External adapter:

- `results/external_neural_adapter_full_seed42.json`
- `results/external_neural_adapter_full_seed43.json`
- `results/external_neural_adapter_full_seed44.json`
- `results/external_neural_adapter_full_seed42.manifest.json`
- `results/external_neural_adapter_full_seed43.manifest.json`
- `results/external_neural_adapter_full_seed44.manifest.json`

## Remaining Bottlenecks

The dominant remaining bottleneck has shifted from rollback to transfer/generalization safety and evaluator calibration. The controller now avoids repeating the worst rollback-heavy method path, but the accepted `repair_bottleneck`-style candidates can still leave small validation-hidden mismatch and transfer-risk increases.

## Next Experiment

Run a targeted transfer-safe mutation-policy repair:

- Add transfer-risk-aware probation for accepted hidden-validation candidates.
- Require task-family transfer non-regression before accepting `repair_bottleneck` in high-shift runs.
- Add a memory-disabled baseline specifically for transfer-risk and external-adapter guard strictness.
- Compare against memory-disabled and mutation-policy-disabled controls over seeds 42, 43, and 44.
