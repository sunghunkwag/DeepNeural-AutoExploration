# ARC Seed-44 Failure Residue Report

This report documents how seed 44 should be treated by the upgraded scaffold:
as a failure case to dissect, not as a solved ARC result.

The repository README currently states that the full ARC seeds `42, 43, 44`
improve mean held-out cell accuracy, while seed `44` still has `0.0`
exact-grid accuracy. This document does not change benchmark numbers and does
not claim a rerun. It defines a residue taxonomy and a validation-only analysis
path for future runs.

## Why Seed 44 Remains a Failure Case

Seed 44 can retain useful cell-level accuracy while still failing exact-grid
success. That gap matters because ARC exact-grid scoring is brittle: a single
wrong object placement, color choice, shape transform, or selector decision can
make the entire task fail.

The existing same-shape adapter is intentionally scoped. It filters to public
ARC-AGI-1 tasks whose input/output grids have the same shape and runs bounded
operator-program candidates through validation and hidden-validation gates.
That setup is useful evidence for a scaffold, but it does not cover variable
output shapes, broad object binding, or a complete primitive library.

## Failure Taxonomy

Seed-44-like failures should be converted into structured residues using these
categories:

- object-binding failure: object identity, grouping, or relations are not
  represented robustly enough.
- variable-shape failure: the candidate path assumes same-shape or aligned-cell
  behavior when the task needs shape construction.
- color-rule ambiguity: support examples admit competing recoloring rules.
- insufficient support evidence: the support set is too sparse or conflicting
  for a stable rule choice.
- primitive-library blind spot: the needed transformation family is absent from
  the bounded primitive or neural module library.
- evaluator-selection failure: validation selection does not predict hidden
  validation or held-out behavior after decisions are frozen.
- architecture-representation bottleneck: the neural candidate lacks memory,
  bottleneck, width, or residual structure needed to represent the pattern.

## Anti-Leakage Rule

Residues used for candidate generation must come from train, validation, or
hidden-validation evidence only. The residue extractor refuses traces marked as
using held-out labels or traces whose candidate-selection split is `test`.
Held-out outputs may be reported only after candidate decisions are frozen.

## What This Enables

The taxonomy turns failed ARC traces into auditable next experiments:

- propose object-centric modules for repeated object-binding failures;
- propose variable-shape adapters when same-shape assumptions break;
- expand symbolic primitive families only when residues justify them;
- repair evaluator selection when hidden-validation regressions recur;
- route architecture bottlenecks into bounded neural architecture search.

This is failure analysis for a CPU-runnable research scaffold. It is not a
claim that ARC is solved, and it is not evidence of open-ended recursive
self-improvement.
