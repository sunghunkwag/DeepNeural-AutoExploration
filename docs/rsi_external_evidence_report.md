# External RSI Evidence Report

Date: 2026-05-18

This report records the current external benchmark evidence for the repository's bounded recursive self-improvement work. It is written to separate supported claims from unsupported claims.

Generated benchmark JSON and manifest files are not tracked in Git. The command blocks below reproduce the local `results/` artifacts; durable evidence is summarized in this document.

## Supported Claim

The repository demonstrates bounded, validation-gated, executable-program RSI improvement on:

- a public ARC-AGI-1 same-shape cell-prediction subset, with multi-seed held-out cell-accuracy gains and nonzero exact-grid success in aggregate;
- an external HumanEval coding harness, with deterministic public-task template completions measured against HumanEval tests.

## Unsupported Claims

These results do not demonstrate AGI, quasi-AGI, human-level intelligence, open-ended recursive self-improvement, official ARC leaderboard performance, or a general-purpose coding model.

## External Sources

- ARC-AGI-1 public repository: `https://github.com/fchollet/ARC-AGI`
- OpenAI HumanEval repository: `https://github.com/openai/human-eval`

## ARC Benchmark Protocol

Benchmark file: `benchmarks/arc_external_rsi_benchmark.py`
HDC signature file: `arc_hdc_signature.py`

Scope:

- Uses public ARC-AGI-1 tasks only.
- Filters to same-shape input/output tasks.
- Splits task IDs into train, validation, hidden validation, and held-out test.
- Candidate programs are executable `OperatorProgram` DSL programs.
- Candidate acceptance uses validation and hidden validation only.
- Final held-out test scoring is frozen after candidate decisions.
- Per-task final selection uses leave-one-demonstration-out support cross-validation on public ARC training examples, not held-out test labels.
- OMEGA-THDSE-inspired ARC additions use deterministic 10,000-dimensional FHRR-style phase signatures, support-only geometric/color transforms, a nested-ring topological rule, support-only object translation with drop/clamp boundary modes, and support-only exact-grid repair search.
- Support-symbolic candidates may use a small hidden-cell tolerance only when validation gain is positive, exact-grid accuracy does not regress, deterministic replay passes, runtime behavior changes, and final deployment is still selected by support cross-validation.
- Low-confidence support-translation and exact-repair programs receive a deployment penalty when leave-one-demonstration-out support accuracy is weak and exact support accuracy is zero.

Commands:

```bash
python benchmarks/arc_external_rsi_benchmark.py --mode smoke --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_multiseed_smoke_seed42.json
python benchmarks/arc_external_rsi_benchmark.py --mode smoke --seed 43 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_multiseed_smoke_seed43.json
python benchmarks/arc_external_rsi_benchmark.py --mode smoke --seed 44 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_multiseed_smoke_seed44.json
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_multiseed_quick_seed42.json
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 43 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_multiseed_quick_seed43.json
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 44 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_multiseed_quick_seed44.json
python benchmarks/arc_external_rsi_benchmark.py --mode full --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_multiseed_full_seed42.json
python benchmarks/arc_external_rsi_benchmark.py --mode full --seed 43 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_multiseed_full_seed43.json
python benchmarks/arc_external_rsi_benchmark.py --mode full --seed 44 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_multiseed_full_seed44.json
```

Earlier multi-seed ARC results over seeds `42, 43, 44` before the latest translation-guard rerun:

| Mode | Baseline cell mean | Evolved cell mean | Cell delta mean | Cell delta stderr | Evolved exact mean | Exact delta mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | `0.3991898148` | `0.8106172840` | `+0.4114274691` | `0.0761189351` | `0.3333333333` | `+0.3333333333` |
| quick | `0.4002898028` | `0.8260590239` | `+0.4257692211` | `0.1837154198` | `0.0666666667` | `+0.0666666667` |
| full | `0.3883530773` | `0.8115196078` | `+0.4231665305` | `0.2090371899` | `0.0416666667` | `+0.0416666667` |

Interpretation:

- Cell-level improvement is positive in all tested modes and seeds.
- Exact-grid accuracy is nonzero on average in all modes, but remains low.
- The result is not an official ARC score because it uses a disclosed public same-shape subset.

## OMEGA-THDSE-Inspired HDC/Translation Rerun

The local `OMEGA-THDSE` repository suggested a useful concrete mechanism, not a mystical "beyond human" step: high-dimensional FHRR phase signatures, bind/bundle composition, and topology-aware symbolic search. This repository uses those ideas only as bounded ARC program-selection machinery.

Commands:

```bash
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_hdc_ring_tolerant2_quick_seed42.json
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 43 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_hdc_ring_tolerant2_quick_seed43.json
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 44 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_hdc_ring_tolerant2_quick_seed44.json
python benchmarks/arc_external_rsi_benchmark.py --mode full --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_hdc_ring_tolerant2_full_seed42.json
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_hdc_translation_guarded_quick_seed42.json
python benchmarks/arc_external_rsi_benchmark.py --mode full --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_hdc_translation_guarded_full_seed42.json
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_exact_repair_guarded_quick_seed42.json
python benchmarks/arc_external_rsi_benchmark.py --mode full --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_exact_repair_guarded_full_seed42.json
```

Quick-mode comparison:

| Seed | Baseline cell | Previous evolved cell | HDC/topological evolved cell | Latest guarded evolved cell | Previous exact | Latest exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `42` | `0.6445714286` | `0.7469795918` | `0.9347346939` | `0.9475510204` | `0.0` | `0.2` |
| `43` | `0.4152525253` | `0.8515987144` | `0.8515987144` | not rerun | `0.2` | `0.2` |
| `44` | `0.1410454545` | `0.8795987654` | `0.8795987654` | not rerun | `0.0` | `0.0` |
| mean before translation guard | `0.4002898028` | `0.8260590239` | `0.8886440579` | not measured | `0.0666666667` | `0.1333333333` |

Full-mode seed 42 spot-check:

| Mode/seed | Baseline cell | Previous evolved cell | Latest guarded evolved cell | Latest cell delta | Exact delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| full/42 | `0.6680555556` | `0.7673611111` | `0.7827777778` | `+0.1147222222` | `0.0` |

Interpretation:

- The OMEGA-inspired path produced a real quick-mode improvement, mostly by solving a nested-ring ARC task that the prior DSL missed.
- The later support-translation guard produced a verified full seed-42 cell-accuracy improvement over the previous HDC/topological full result.
- The later exact-repair search produced verified full seed-42 exact-grid improvement, and the marker-projection/object-lift extension raised the current full seed-42 exact result again.
- This is bounded symbolic/topological program induction, not AGI, quasi-AGI, or open-ended self-improvement.

## Exact-Repair ARC Rerun

`support_exact_repair_search` is an executable DSL primitive that tries support-only repair programs for binary mask recoloring, edge reflection, marker-rectangle fill, 2x2 corner projection, rowwise terminal shifts, marker-to-block projection, and object lift-by-height rules. Candidate acceptance still uses validation and hidden validation only. Final deployment still uses leave-one-demonstration-out support cross-validation, and low-confidence exact-repair deployment is penalized.

Additional commands:

```bash
python benchmarks/arc_external_rsi_benchmark.py --mode quick --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_exact_repair_projected_quick_seed42.json
python benchmarks/arc_external_rsi_benchmark.py --mode full --seed 42 --arc-data-dir ../external_arc_agi --no-download --output results/arc_external_rsi_exact_repair_projected_full_seed42.json
```

| Mode/seed | Baseline cell | Previous guarded cell | Exact-repair cell | Projected-repair cell | Cell delta | Previous exact | Exact-repair exact | Projected-repair exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| quick/42 | `0.6445714286` | `0.9475510204` | `0.9577142857` | `0.9577142857` | `+0.3131428571` | `0.2` | `0.4` | `0.4` |
| full/42 | `0.6680555556` | `0.7827777778` | `0.9244444444` | `0.9587500000` | `+0.2906944444` | `0.0` | `0.625` | `0.875` |

Full seed-42 exact task breakdown after marker-to-block projection and object lift-by-height:

| Task ID | Baseline cell | Evolved cell | Exact |
| --- | ---: | ---: | ---: |
| `93b581b8` | `0.555556` | `1.000000` | yes |
| `025d127b` | `0.820000` | `1.000000` | yes |
| `aba27056` | `0.640000` | `0.670000` | no |
| `1f642eb9` | `0.770000` | `1.000000` | yes |
| `f76d97a5` | `0.640000` | `1.000000` | yes |
| `af902bf9` | `0.790000` | `1.000000` | yes |
| `496994bd` | `0.400000` | `1.000000` | yes |
| `5521c0d9` | `0.728889` | `1.000000` | yes |

## HumanEval Coding Benchmark Protocol

Benchmark file: `benchmarks/humaneval_template_rsi_benchmark.py`

Scope:

- Uses OpenAI HumanEval problem prompts and tests.
- Baseline is a syntactically valid no-op function body.
- Evolved side is a deterministic transparent template library for public HumanEval task patterns.
- Canonical solutions are ignored and never executed.
- Candidate code runs in temporary subprocesses with timeouts.

Commands:

```bash
python benchmarks/humaneval_template_rsi_benchmark.py --mode smoke --humaneval-dir ../external_human_eval --output results/humaneval_template_rsi_smoke.json
python benchmarks/humaneval_template_rsi_benchmark.py --mode quick --humaneval-dir ../external_human_eval --output results/humaneval_template_rsi_quick.json
python benchmarks/humaneval_template_rsi_benchmark.py --mode full --humaneval-dir ../external_human_eval --output results/humaneval_template_rsi_full.json
```

HumanEval results:

| Mode | Tasks | Baseline pass@1 | Evolved pass@1 | Delta |
| --- | ---: | ---: | ---: | ---: |
| smoke | `20` | `0.0` | `1.0` | `+1.0` |
| quick | `40` | `0.0` | `1.0` | `+1.0` |
| full | `164` | `0.0` | `1.0` | `+1.0` |

Interpretation:

- This proves the repository can run an external coding benchmark and report functional correctness.
- The full score is benchmark-specific because the evolved side is a deterministic public-task template library, not a language model or general program synthesizer.
- It does not prove general code generation, autonomous coding, AGI, or quasi-AGI.

## Verification

The following local verification completed after the benchmark changes:

```bash
python -m pytest tests/test_arc_external_rsi_benchmark.py tests/test_humaneval_template_rsi_benchmark.py tests/test_code_level_rsi.py tests/test_operator_genome.py -q
python -m pytest -q
```

Observed result:

- Focused benchmark/operator tests passed.
- Full suite passed after the HDC/topological/translation/exact-repair ARC upgrade: `168 passed`.
- Full suite passed after the research-goal and marker-projection/object-lift exact-repair extension: `174 passed`.

## Remaining Work

- Increase ARC exact-grid success beyond `7 / 8` on the current full seed-42 same-shape split.
- Add a non-template coding synthesizer or LLM-backed coding generator if credentials and model policy permit.
- Add more seeds and confidence intervals.
- Evaluate on variable-shape ARC tasks.
- Package generated result manifests as immutable release artifacts.
