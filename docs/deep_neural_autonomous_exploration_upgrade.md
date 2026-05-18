# Deep Neural Autonomous Exploration Upgrade

This repository remains a bounded, CPU-runnable research scaffold. The upgrade
adds executable components for neural architecture exploration, failure residue
analysis, object-centric world modeling, and a higher-level orchestration loop.
It does not claim AGI, human-level intelligence, technological singularity, or
open-ended autonomous recursive self-improvement.

## Terms

Deep neural autonomous exploration means bounded search over neural
architectures or modules, with deterministic seeds, validation gates, rollback,
and audit records.

Recursive self-improvement means a system proposes changes to part of its own
candidate space or evaluator process, then accepts only changes that pass
predefined validation checks. In this repository the loop is bounded and
scaffolded.

Code-level operator-program RSI is the existing DSL-based path that mutates
bounded operator programs. It does not execute arbitrary generated Python.

Symbolic ARC repair is the support-only primitive path used by the ARC adapter.
It is scoped to disclosed public same-shape ARC-AGI-1 tasks in the current
benchmark.

Neural architecture search is the new `neural_search/` path. It represents
architectures as genomes and mutates residual depth, width, memory gates, and
causal bottlenecks.

## What This Repo Now Supports

- explicit neural architecture genomes;
- deterministic bounded architecture mutations;
- shape-compatible weight inheritance;
- validation-only neural candidate sandboxing;
- structured failure residues with held-out leakage refusal;
- ARC failure taxonomy records for seed-44-style failures;
- object-centric state, causal graph, do-style interventions, and
  counterfactual rollout utilities;
- a meta-RSI orchestrator that turns validation failure residues into neural
  candidates, registry updates, experiment plans, and audit reports.

## What It Still Does Not Support

- no AGI or human-level intelligence claim;
- no solved ARC claim;
- no official ARC leaderboard claim;
- no open-ended autonomous RSI;
- no arbitrary source-code generation or execution for candidates;
- no use of held-out test labels for candidate generation, selection,
  architecture mutation, primitive creation, or evaluator repair;
- no benchmark result changes without rerunning and recording the command,
  seed, mode, output path, and metrics.

## Audit Requirements

Accepted neural architecture candidates record parent id, mutation method,
validation metrics, hidden-validation metrics when available, held-out-label
usage, rollback status, random seed, and config hash. Rejected candidates are
kept in the audit trail with rejection reasons.
