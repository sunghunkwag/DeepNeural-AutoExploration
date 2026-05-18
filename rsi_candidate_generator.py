"""Bounded candidate generation for code-level RSI experiments."""

from __future__ import annotations

import random
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from operator_dsl import (
    MEMORY_PRIMITIVES,
    OBJECTIVE_PRIMITIVES,
    SCHEDULE_PRIMITIVES,
    WORLD_MODEL_PRIMITIVES,
    OperatorProgram,
    PrimitiveStep,
)


@dataclass(frozen=True)
class CandidateGenerationRecord:
    candidate: OperatorProgram
    method: str
    reason: str
    parent_program_ids: List[str]
    generation: int
    used_test_results: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "method": self.method,
            "reason": self.reason,
            "parent_program_ids": list(self.parent_program_ids),
            "generation": self.generation,
            "used_test_results": self.used_test_results,
        }


@dataclass
class CandidateGeneratorConfig:
    max_candidates: int = 6
    max_program_length: int = 9
    include_arc_candidates: bool = False
    lr_perturbations: Sequence[float] = (0.7, 0.85, 1.15, 1.3)
    mutation_methods: Sequence[str] = (
        "parameter_perturbation",
        "primitive_insertion",
        "primitive_replacement",
        "primitive_deletion",
        "schedule_mutation",
        "memory_gate_mutation",
        "world_model_gate_mutation",
        "recombination",
    )

    def to_dict(self) -> Dict[str, object]:
        return {
            "max_candidates": self.max_candidates,
            "max_program_length": self.max_program_length,
            "include_arc_candidates": self.include_arc_candidates,
            "lr_perturbations": list(self.lr_perturbations),
            "mutation_methods": list(self.mutation_methods),
        }


class CandidateGenerator:
    """Generate bounded executable operator programs from accepted programs.

    The generator consumes accepted programs and training/validation trace
    summaries only.  It never receives held-out test scores or test task IDs.
    """

    def __init__(self, seed: int = 0, config: CandidateGeneratorConfig | None = None):
        self.seed = seed
        self.config = config or CandidateGeneratorConfig()
        self.rng = random.Random(seed)
        self.records: List[CandidateGenerationRecord] = []

    def generate(
        self,
        accepted_programs: Sequence[OperatorProgram],
        *,
        generation: int,
        trace_summary: Dict[str, float] | None = None,
        random_baseline: bool = False,
    ) -> List[CandidateGenerationRecord]:
        if not accepted_programs:
            raise ValueError("candidate generation requires accepted parent programs")
        self.records = []
        parents = list(accepted_programs)
        trace_summary = dict(trace_summary or {})
        templates = self._template_candidates(parents, generation, trace_summary, random_baseline=random_baseline)
        seen = set()
        for candidate, method, reason, parent_ids in templates:
            if candidate.source_hash in seen:
                continue
            seen.add(candidate.source_hash)
            record = CandidateGenerationRecord(candidate, method, reason, parent_ids, generation, used_test_results=False)
            self.records.append(record)
            if len(self.records) >= self.config.max_candidates:
                break
        return list(self.records)

    def _template_candidates(
        self,
        parents: Sequence[OperatorProgram],
        generation: int,
        trace_summary: Dict[str, float],
        *,
        random_baseline: bool,
    ) -> List[tuple[OperatorProgram, str, str, List[str]]]:
        active = parents[-1]
        best = max(parents, key=lambda p: float(trace_summary.get(p.program_id, 0.0)))
        other = parents[0]
        if other.program_id == best.program_id and len(parents) > 1:
            other = parents[1]
        out: List[tuple[OperatorProgram, str, str, List[str]]] = []

        if random_baseline:
            for idx in range(self.config.max_candidates):
                parent = self.rng.choice(parents)
                sequence = self._random_sequence()
                params = dict(parent.parameters)
                params["lr_scale"] = self.rng.choice(list(self.config.lr_perturbations))
                out.append(
                    (
                        self._program(f"g{generation}_random_{idx}", sequence, params, parent, generation),
                        "random_candidate_generator",
                        "random bounded program baseline generated without validation/test outcomes",
                        [parent.program_id],
                    )
                )
            return out

        if self.config.include_arc_candidates:
            out.append(
                (
                    self._program(
                        f"g{generation}_support_exact_repair_search",
                        (PrimitiveStep("support_exact_repair_search"),),
                        self._perturbed_params(best),
                        best,
                        generation,
                    ),
                    "support_exact_repair_search_mutation",
                    "search support-only exact-grid repair programs before held-out ARC scoring",
                    [best.program_id],
                )
            )
            out.append(
                (
                    self._program(
                        f"g{generation}_support_geometric_transform_mapping",
                        (PrimitiveStep("support_geometric_transform_mapping"),),
                        self._perturbed_params(best),
                        best,
                        generation,
                    ),
                    "hdc_geometric_rule_mutation",
                    "use support-only geometric/color transform search with a 10000-dimensional FHRR tie-breaker",
                    [best.program_id],
                )
            )
            out.append(
                (
                    self._program(
                        f"g{generation}_support_nested_ring_reversal_mapping",
                        (PrimitiveStep("support_nested_ring_reversal_mapping"),),
                        self._perturbed_params(best),
                        best,
                        generation,
                    ),
                    "hdc_topological_ring_rule_mutation",
                    "reverse outer-to-inner nested ARC ring colors using support-only topology and 10000-dimensional HDC scoring",
                    [best.program_id],
                )
            )
            out.append(
                (
                    self._program(
                        f"g{generation}_support_translation_mapping",
                        (PrimitiveStep("support_translation_mapping"),),
                        self._perturbed_params(best),
                        best,
                        generation,
                    ),
                    "hdc_translation_rule_mutation",
                    "infer a support-only row/column translation of non-background ARC objects with 10000-dimensional HDC tie-breaking",
                    [best.program_id],
                )
            )
            out.append(
                (
                    self._program(
                        f"g{generation}_support_local_pattern_mapping",
                        (PrimitiveStep("support_local_pattern_mapping"),),
                        self._perturbed_params(best),
                        best,
                        generation,
                    ),
                    "symbolic_local_pattern_rule_mutation",
                    "induce a support-only local-neighborhood rule for same-shape ARC classification tasks",
                    [best.program_id],
                )
            )
            out.append(
                (
                    self._program(
                        f"g{generation}_support_feature_lookup_mapping",
                        (PrimitiveStep("support_feature_lookup_mapping", {"mode": "feature"}),),
                        self._perturbed_params(best),
                        best,
                        generation,
                    ),
                    "symbolic_feature_lookup_rule_mutation",
                    "induce a support-only full-feature lookup rule for repeated ARC cell patterns",
                    [best.program_id],
                )
            )
            out.append(
                (
                    self._program(
                        f"g{generation}_support_position_color_mapping",
                        (PrimitiveStep("support_feature_lookup_mapping", {"mode": "position_color"}),),
                        self._perturbed_params(best),
                        best,
                        generation,
                    ),
                    "symbolic_position_color_rule_mutation",
                    "induce a support-only position-and-color rule for aligned same-shape ARC grids",
                    [best.program_id],
                )
            )
            out.append(
                (
                    self._program(
                        f"g{generation}_support_color_mapping",
                        (PrimitiveStep("support_color_mapping"),),
                        self._perturbed_params(best),
                        best,
                        generation,
                    ),
                    "symbolic_support_rule_mutation",
                    "induce a support-only input-color to output-color rule for same-shape classification tasks",
                    [best.program_id],
                )
            )
            out.append(
                (
                    self._program(
                        f"g{generation}_low_lr_multi_step",
                        (
                            PrimitiveStep("constant_lr"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                        ),
                        self._perturbed_params(best, lr_scale=0.2),
                        best,
                        generation,
                    ),
                    "multi_step_lr_mutation",
                    "try a lower learning-rate repeated support adaptation schedule selected only through validation traces",
                    [best.program_id],
                )
            )
            out.append(
                (
                    self._program(
                        f"g{generation}_clipped_long_adaptation",
                        (
                            PrimitiveStep("gradient_norm_clipping", {"max_norm": 1.0}),
                            PrimitiveStep("constant_lr"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                            PrimitiveStep("maml_step"),
                        ),
                        self._perturbed_params(best, lr_scale=4.0, clip_norm=1.0),
                        best,
                        generation,
                    ),
                    "adaptive_horizon_lr_mutation",
                    "combine online step-size adaptation with gradient clipping and a longer support-only horizon",
                    [best.program_id],
                )
            )
        out.append(
            (
                self._program(
                    f"g{generation}_support_proxy_clip",
                    (
                        PrimitiveStep("gradient_norm_clipping", {"max_norm": 1.25}),
                        PrimitiveStep("support_loss_weighting", {"weight": 0.7}),
                        PrimitiveStep("constant_lr"),
                        PrimitiveStep("support_proxy_step"),
                        PrimitiveStep("decayed_lr", {"decay": 0.65}),
                        PrimitiveStep("maml_step"),
                    ),
                    self._perturbed_params(best, lr_scale=0.65, decay=0.65, clip_norm=1.25, support_weight=0.7),
                    best,
                    generation,
                ),
                "support_proxy_mutation",
                "optimize through a support holdout step plus a conservative clipped MAML step for OOD robustness",
                [best.program_id],
            )
        )
        out.append(
            (
                self._program(
                    f"g{generation}_maml_clip_decay",
                    (
                        PrimitiveStep("gradient_norm_clipping", {"max_norm": 2.0}),
                        PrimitiveStep("constant_lr"),
                        PrimitiveStep("maml_step"),
                        PrimitiveStep("decayed_lr", {"decay": 0.75}),
                        PrimitiveStep("variable_lr_step", {"decay": 0.75}),
                    ),
                    self._perturbed_params(best, lr_scale=0.9, decay=0.75),
                    best,
                    generation,
                ),
                "primitive_insertion",
                "insert gradient clipping and a second decayed support-only update into the best train-trace parent",
                [best.program_id],
            )
        )
        out.append(
            (
                self._program(
                    f"g{generation}_reptile_memory_gate",
                    (
                        PrimitiveStep("memory_retrieval_k", {"k": 2}),
                        PrimitiveStep("memory_reward_gate"),
                        PrimitiveStep("memory_similarity_gate"),
                        PrimitiveStep("memory_gated_lr", {"strength": 0.75}),
                        PrimitiveStep("maml_step"),
                        PrimitiveStep("reptile_interpolation", {"mix": 0.65}),
                    ),
                    self._perturbed_params(active, memory_k=2, memory_gate_strength=0.75, reptile_mix=0.65),
                    active,
                    generation,
                ),
                "memory_gate_mutation",
                "combine Reptile interpolation with non-test memory reward and similarity gates",
                [active.program_id],
            )
        )
        out.append(
            (
                self._program(
                    f"g{generation}_world_selected_lr",
                    (
                        PrimitiveStep("world_model_lr_score"),
                        PrimitiveStep("world_model_selected_lr"),
                        PrimitiveStep("predicted_improvement_gate"),
                        PrimitiveStep("variable_lr_step", {"decay": 0.85}),
                        PrimitiveStep("prediction_error_reduction"),
                    ),
                    self._perturbed_params(best, lr_scale=1.0, decay=0.85),
                    best,
                    generation,
                ),
                "world_model_gate_mutation",
                "let the trained world model select the learning-rate schedule before support-only adaptation",
                [best.program_id],
            )
        )
        out.append(
            (
                self._program(
                    f"g{generation}_memory_uncertainty",
                    (
                        PrimitiveStep("memory_retrieval_k", {"k": 3}),
                        PrimitiveStep("memory_reward_gate"),
                        PrimitiveStep("memory_gated_lr", {"strength": 0.55}),
                        PrimitiveStep("uncertainty_penalty"),
                        PrimitiveStep("gradient_norm_clipping", {"max_norm": 1.5}),
                        PrimitiveStep("maml_step"),
                    ),
                    self._perturbed_params(active, memory_k=3, memory_gate_strength=0.55, clip_norm=1.5),
                    active,
                    generation,
                ),
                "objective_mutation",
                "add an uncertainty penalty to a memory-gated adaptation operator",
                [active.program_id],
            )
        )
        out.append(
            (
                self._program(
                    f"g{generation}_support_proxy_recombine",
                    self._recombine_sequences(best, other),
                    self._perturbed_params(best, lr_scale=0.85, support_weight=1.1),
                    best,
                    generation,
                    extra_parent=other,
                ),
                "recombination",
                "recombine two accepted parents and add support-only validation proxy loss",
                [best.program_id, other.program_id],
            )
        )
        out.append(
            (
                self._program(
                    f"g{generation}_noise_novelty_schedule",
                    (
                        PrimitiveStep("novelty_bonus"),
                        PrimitiveStep("gradient_noise_injection", {"scale": 0.001}),
                        PrimitiveStep("increasing_lr", {"growth": 1.05}),
                        PrimitiveStep("maml_step"),
                        PrimitiveStep("decayed_lr", {"decay": 0.7}),
                        PrimitiveStep("variable_lr_step", {"decay": 0.7}),
                    ),
                    self._perturbed_params(best, lr_scale=0.8, noise_scale=0.001, growth=1.05),
                    best,
                    generation,
                ),
                "schedule_mutation",
                "test a bounded novelty and noise schedule under validation-only acceptance",
                [best.program_id],
            )
        )
        return out

    def _program(
        self,
        stem: str,
        sequence: Sequence[PrimitiveStep],
        params: Dict[str, float | int | bool | str],
        parent: OperatorProgram,
        generation: int,
        *,
        extra_parent: OperatorProgram | None = None,
    ) -> OperatorProgram:
        sequence = tuple(sequence[: self.config.max_program_length])
        lineage = parent.program_id if extra_parent is None else f"{parent.program_id}+{extra_parent.program_id}"
        payload = json.dumps({"seed": self.seed, "lineage": lineage, "sequence": [step.to_dict() for step in sequence], "params": params}, sort_keys=True).encode("utf-8")
        suffix = int(hashlib.sha256(payload).hexdigest()[:8], 16) % 100000
        program_id = f"{stem}_{suffix:05d}"
        return OperatorProgram(program_id, sequence, params, parent.program_id, generation)

    def _perturbed_params(self, parent: OperatorProgram, **updates: float | int | bool | str) -> Dict[str, float | int | bool | str]:
        params = dict(parent.parameters)
        params.update(updates)
        return params

    def _recombine_sequences(self, left: OperatorProgram, right: OperatorProgram) -> Sequence[PrimitiveStep]:
        left_steps = list(left.primitive_sequence)
        right_steps = list(right.primitive_sequence)
        midpoint = max(1, len(left_steps) // 2)
        seq = left_steps[:midpoint] + [PrimitiveStep("validation_proxy_loss"), PrimitiveStep("support_loss_weighting", {"weight": 1.1})] + right_steps[-2:]
        if not any(step.name in {"maml_step", "variable_lr_step"} for step in seq):
            seq.append(PrimitiveStep("maml_step"))
        return seq[: self.config.max_program_length]

    def _random_sequence(self) -> Sequence[PrimitiveStep]:
        memory = self.rng.choice([PrimitiveStep("memory_disabled_control"), PrimitiveStep("memory_reward_gate"), PrimitiveStep("memory_similarity_gate")])
        schedule = PrimitiveStep(self.rng.choice(list(SCHEDULE_PRIMITIVES)))
        objective = PrimitiveStep(self.rng.choice(list(OBJECTIVE_PRIMITIVES)))
        world = PrimitiveStep(self.rng.choice(list(WORLD_MODEL_PRIMITIVES)))
        grad = self.rng.choice([PrimitiveStep("maml_step"), PrimitiveStep("variable_lr_step"), PrimitiveStep("reptile_interpolation")])
        seq = [memory, schedule, objective, world, grad]
        if grad.name == "reptile_interpolation":
            seq.insert(-1, PrimitiveStep("maml_step"))
        return seq
