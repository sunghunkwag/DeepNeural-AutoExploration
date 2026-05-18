"""Evidence-driven research goal generation for bounded RSI experiments.

The controller in this module ports three concrete ideas from sibling
repositories into this repository's existing benchmark-driven architecture:

* competence-gap goal creation from Cognitive-Core-Engine-Test;
* structural transfer hypotheses from its transfer engine;
* meta-meta search over the goal-policy configuration from
  ast-grammar-induction-prototype.

The implementation is intentionally bounded. It creates auditable research
goals from benchmark result files and never claims that goal generation itself
improves held-out benchmark scores.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _stable_id(prefix: str, payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _numeric_metrics(payload: Mapping[str, object]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[str(key)] = float(value)
    return metrics


def _infer_mode(source_path: str) -> str:
    lower = source_path.lower()
    for mode in ("smoke", "quick", "full"):
        if mode in lower:
            return mode
    return "unknown"


def _infer_seed(source_path: str) -> int:
    match = re.search(r"seed(\d+)", source_path.lower())
    return int(match.group(1)) if match else 0


def _metric(metrics: Mapping[str, float], *names: str) -> float | None:
    for name in names:
        if name in metrics:
            return float(metrics[name])
    return None


def _tokens(*values: str) -> Tuple[str, ...]:
    joined = " ".join(v.lower() for v in values if v)
    raw = re.split(r"[^a-z0-9]+", joined)
    return tuple(sorted({token for token in raw if token}))


@dataclass(frozen=True)
class BenchmarkEvidence:
    """Normalized benchmark result used as goal-generation evidence."""

    evidence_id: str
    benchmark: str
    mode: str
    seed: int
    metrics: Dict[str, float]
    source_path: str = ""
    scope: str = ""
    mechanism_tags: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()

    @classmethod
    def from_result(cls, result: Mapping[str, object], source_path: str = "") -> "BenchmarkEvidence":
        config = result.get("config", {})
        config_map = config if isinstance(config, Mapping) else {}
        metric_map = result.get("metrics", {})
        metrics = _numeric_metrics(result)
        if isinstance(metric_map, Mapping):
            metrics.update(_numeric_metrics(metric_map))

        benchmark = str(result.get("benchmark") or config_map.get("benchmark") or "unknown")
        mode = str(config_map.get("mode") or result.get("mode") or _infer_mode(source_path))
        seed = int(result.get("seed") or config_map.get("seed") or _infer_seed(source_path))
        scope = str(config_map.get("scope") or result.get("scope") or "")
        tags = infer_mechanism_tags(benchmark, scope, source_path, metrics)
        limitations = infer_limitations(benchmark, scope, source_path, tags)
        evidence_id = _stable_id(
            "evidence",
            {
                "benchmark": benchmark,
                "mode": mode,
                "seed": seed,
                "source_path": source_path,
                "metric_keys": sorted(metrics),
                "tags": tags,
            },
        )
        return cls(
            evidence_id=evidence_id,
            benchmark=benchmark,
            mode=mode,
            seed=seed,
            metrics=metrics,
            source_path=source_path,
            scope=scope,
            mechanism_tags=tags,
            limitations=limitations,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def infer_mechanism_tags(
    benchmark: str,
    scope: str,
    source_path: str,
    metrics: Mapping[str, float],
) -> Tuple[str, ...]:
    lower = " ".join([benchmark, scope, source_path]).lower()
    tags = set(_tokens(benchmark, scope, source_path))
    if "arc" in lower:
        tags.add("arc")
    if "humaneval" in lower:
        tags.add("coding")
    if "template" in lower:
        tags.add("template")
    if "exact_repair" in lower or "exact-repair" in lower:
        tags.add("exact_repair")
    if "translation" in lower:
        tags.add("translation")
    if "hdc" in lower or "topological" in lower:
        tags.add("hdc_topology")
    if _metric(metrics, "cell_accuracy_delta") is not None:
        tags.add("cell_accuracy")
    if _metric(metrics, "exact_task_accuracy_delta") is not None:
        tags.add("exact_grid")
    if _metric(metrics, "baseline_to_evolved_loss_delta") is not None:
        tags.add("loss_improvement")
    return tuple(sorted(tags))


def infer_limitations(
    benchmark: str,
    scope: str,
    source_path: str,
    tags: Sequence[str],
) -> Tuple[str, ...]:
    lower = " ".join([benchmark, scope, source_path, " ".join(tags)]).lower()
    limitations: List[str] = []
    if "arc" in lower and "same-shape" in lower:
        limitations.append("same_shape_arc_only")
    if "arc" in lower:
        limitations.append("not_official_arc_leaderboard")
    if "template" in lower or "humaneval_template" in lower:
        limitations.append("template_coding_harness")
    return tuple(limitations)


@dataclass(frozen=True)
class ResearchGoal:
    """A proposed research objective grounded in benchmark evidence."""

    goal_id: str
    title: str
    domain: str
    target_metric: str
    current_value: float
    target_value: float
    priority: float
    feasible: bool
    strategy: str
    source_evidence_ids: Tuple[str, ...]
    proposed_actions: Tuple[str, ...]
    rationale: str
    expected_transfer_tags: Tuple[str, ...] = ()
    parent_goal_id: str | None = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TransferHypothesis:
    """A bounded transfer proposal from one measured mechanism to another target."""

    hypothesis_id: str
    source_evidence_id: str
    target_domain: str
    mechanism_tags: Tuple[str, ...]
    similarity: float
    expected_gain: float
    rationale: str

    def to_goal(self, config: "GoalPolicyConfig") -> ResearchGoal:
        target_value = _clamp(self.expected_gain + config.transfer_gain_floor, 0.0, 1.0)
        return ResearchGoal(
            goal_id=_stable_id("goal", ["transfer", self.hypothesis_id, self.target_domain]),
            title=f"Transfer {', '.join(self.mechanism_tags[:3])} to {self.target_domain}",
            domain=self.target_domain,
            target_metric="transfer_validated_delta",
            current_value=0.0,
            target_value=target_value,
            priority=_clamp(config.transfer_weight * self.similarity + 0.5 * self.expected_gain),
            feasible=self.similarity >= config.transfer_threshold,
            strategy="transfer",
            source_evidence_ids=(self.source_evidence_id,),
            proposed_actions=(
                "Create a validation-only target split for the transfer domain.",
                "Reuse the source mechanism only as a candidate generator, not as a final scorer.",
                "Accept transfer only if held-out target metrics improve after selection is frozen.",
            ),
            rationale=self.rationale,
            expected_transfer_tags=self.mechanism_tags,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GoalPolicyConfig:
    """Searchable policy parameters for evidence-to-goal generation."""

    frontier_weight: float = 0.35
    gap_weight: float = 0.40
    creative_weight: float = 0.15
    transfer_weight: float = 0.35
    exact_target: float = 0.90
    cell_target: float = 0.98
    coding_generalization_target: float = 1.0
    min_exact_gap: float = 0.05
    min_cell_gap: float = 0.02
    transfer_threshold: float = 0.45
    transfer_gain_floor: float = 0.05
    max_goals: int = 12

    def normalized(self) -> "GoalPolicyConfig":
        total = self.frontier_weight + self.gap_weight + self.creative_weight + self.transfer_weight
        if total <= 0:
            return replace(self, frontier_weight=0.25, gap_weight=0.40, creative_weight=0.15, transfer_weight=0.20)
        return replace(
            self,
            frontier_weight=self.frontier_weight / total,
            gap_weight=self.gap_weight / total,
            creative_weight=self.creative_weight / total,
            transfer_weight=self.transfer_weight / total,
            exact_target=_clamp(self.exact_target, 0.0, 1.0),
            cell_target=_clamp(self.cell_target, 0.0, 1.0),
            coding_generalization_target=_clamp(self.coding_generalization_target, 0.0, 1.0),
            transfer_threshold=_clamp(self.transfer_threshold, 0.0, 1.0),
            max_goals=max(1, int(self.max_goals)),
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self.normalized())


@dataclass(frozen=True)
class MetaMetaConfig:
    """Controls the bounded meta-meta search over GoalPolicyConfig."""

    rounds: int = 4
    candidates_per_round: int = 6
    mutation_scale: float = 0.12
    seed: int = 42


@dataclass(frozen=True)
class MetaMetaResult:
    initial_score: float
    final_score: float
    accepted_updates: Tuple[Dict[str, object], ...]
    initial_config: Dict[str, object]
    final_config: Dict[str, object]
    goals: Tuple[ResearchGoal, ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "score_delta": self.final_score - self.initial_score,
            "accepted_updates": list(self.accepted_updates),
            "initial_config": self.initial_config,
            "final_config": self.final_config,
            "goals": [goal.to_dict() for goal in self.goals],
        }


class TransferPlanner:
    """Creates structural transfer hypotheses from successful benchmark mechanisms."""

    def __init__(self, config: GoalPolicyConfig | None = None) -> None:
        self.config = (config or GoalPolicyConfig()).normalized()

    def structural_similarity(self, source: BenchmarkEvidence, target_tokens: Iterable[str]) -> float:
        source_tokens = set(source.mechanism_tags)
        target = set(target_tokens)
        if not source_tokens or not target:
            return 0.0
        overlap = len(source_tokens & target) / len(source_tokens | target)
        domain_bonus = 0.2 if ("arc" in source_tokens and "arc" in target) else 0.0
        exact_bonus = 0.2 if ("exact_grid" in source_tokens and "exact_grid" in target) else 0.0
        return _clamp(overlap + domain_bonus + exact_bonus)

    def plan(self, evidence: Sequence[BenchmarkEvidence]) -> List[TransferHypothesis]:
        hypotheses: List[TransferHypothesis] = []
        for item in evidence:
            exact_delta = _metric(item.metrics, "exact_task_accuracy_delta") or 0.0
            cell_delta = _metric(item.metrics, "cell_accuracy_delta") or 0.0
            loss_delta = _metric(item.metrics, "baseline_to_evolved_loss_delta") or 0.0
            gain = max(exact_delta, cell_delta * 0.5, min(loss_delta, 1.0) * 0.25)
            if gain <= 0:
                continue
            tags = tuple(tag for tag in item.mechanism_tags if tag in {"arc", "exact_repair", "exact_grid", "hdc_topology", "translation", "cell_accuracy", "coding"})
            if "arc" in item.mechanism_tags:
                hypotheses.append(
                    self._hypothesis(
                        item,
                        "arc_agi1_same_shape_unseen_seeds",
                        ("arc", "exact_grid", "cell_accuracy", "exact_repair"),
                        gain,
                        "Positive ARC deltas should be checked across seeds before being treated as stable.",
                    )
                )
                hypotheses.append(
                    self._hypothesis(
                        item,
                        "arc_agi1_variable_shape_adapter",
                        ("arc", "exact_grid", "translation", "hdc_topology"),
                        gain * 0.65,
                        "Same-shape repair mechanisms are plausible candidate generators for variable-shape adapters, but need a separate validation split.",
                    )
                )
            if "coding" in item.mechanism_tags or "template" in item.mechanism_tags:
                hypotheses.append(
                    self._hypothesis(
                        item,
                        "non_template_coding_synthesis",
                        ("coding", "exact_grid", "loss_improvement"),
                        max(gain, 0.15),
                        "Template-driven coding evidence should transfer only into a non-template generator benchmark after the template limitation is removed.",
                    )
                )
            for target in evidence:
                if target.evidence_id == item.evidence_id:
                    continue
                if set(item.mechanism_tags) & set(target.mechanism_tags):
                    target_exact = _metric(target.metrics, "evolved_exact_task_accuracy") or 0.0
                    source_exact = _metric(item.metrics, "evolved_exact_task_accuracy") or 0.0
                    if source_exact > target_exact:
                        hypotheses.append(
                            self._hypothesis(
                                item,
                                f"{target.benchmark}:{target.mode}:seed{target.seed}",
                                target.mechanism_tags,
                                min(gain, source_exact - target_exact),
                                "A stronger source run and a weaker target run share mechanism tags, so transfer should be tested explicitly.",
                            )
                        )
        return _dedupe_hypotheses(hypotheses)

    def _hypothesis(
        self,
        source: BenchmarkEvidence,
        target_domain: str,
        target_tags: Sequence[str],
        gain: float,
        rationale: str,
    ) -> TransferHypothesis:
        sim = self.structural_similarity(source, target_tags)
        return TransferHypothesis(
            hypothesis_id=_stable_id("transfer", [source.evidence_id, target_domain, target_tags]),
            source_evidence_id=source.evidence_id,
            target_domain=target_domain,
            mechanism_tags=tuple(sorted(set(source.mechanism_tags) & set(target_tags))) or tuple(sorted(set(target_tags))),
            similarity=sim,
            expected_gain=_clamp(gain),
            rationale=rationale,
        )


def _dedupe_hypotheses(hypotheses: Sequence[TransferHypothesis]) -> List[TransferHypothesis]:
    best_by_key: Dict[Tuple[str, str], TransferHypothesis] = {}
    for hyp in hypotheses:
        key = (hyp.source_evidence_id, hyp.target_domain)
        current = best_by_key.get(key)
        if current is None or (hyp.similarity, hyp.expected_gain) > (current.similarity, current.expected_gain):
            best_by_key[key] = hyp
    return sorted(best_by_key.values(), key=lambda h: (h.similarity, h.expected_gain), reverse=True)


class EvidenceGoalGenerator:
    """Generate research goals from benchmark metrics and limitations."""

    def __init__(self, config: GoalPolicyConfig | None = None) -> None:
        self.config = (config or GoalPolicyConfig()).normalized()
        self.transfer_planner = TransferPlanner(self.config)

    def generate(self, evidence: Sequence[BenchmarkEvidence]) -> List[ResearchGoal]:
        goals: List[ResearchGoal] = []
        for item in evidence:
            if "arc" in item.mechanism_tags:
                goals.extend(self._arc_goals(item))
            if "coding" in item.mechanism_tags or "template" in item.mechanism_tags:
                goals.extend(self._coding_goals(item))
        for hypothesis in self.transfer_planner.plan(evidence):
            if hypothesis.similarity >= self.config.transfer_threshold:
                goals.append(hypothesis.to_goal(self.config))
        goals.extend(self._creative_goals(evidence))
        return self._rank_and_dedupe(goals)

    def _arc_goals(self, item: BenchmarkEvidence) -> List[ResearchGoal]:
        metrics = item.metrics
        goals: List[ResearchGoal] = []
        exact = _metric(metrics, "evolved_exact_task_accuracy")
        cell = _metric(metrics, "evolved_cell_accuracy")
        exact_delta = _metric(metrics, "exact_task_accuracy_delta") or 0.0
        cell_delta = _metric(metrics, "cell_accuracy_delta") or 0.0
        if exact is not None and self.config.exact_target - exact >= self.config.min_exact_gap:
            gap = self.config.exact_target - exact
            goals.append(
                ResearchGoal(
                    goal_id=_stable_id("goal", [item.evidence_id, "arc_exact", self.config.exact_target]),
                    title=f"Raise ARC exact-grid accuracy for {item.mode} seed {item.seed}",
                    domain="arc_agi1_exact_grid",
                    target_metric="evolved_exact_task_accuracy",
                    current_value=exact,
                    target_value=self.config.exact_target,
                    priority=_clamp(self.config.gap_weight + gap * 0.6 + exact_delta * 0.2),
                    feasible=exact_delta > 0.0 or cell_delta > 0.05,
                    strategy="gap",
                    source_evidence_ids=(item.evidence_id,),
                    proposed_actions=(
                        "Cluster unresolved ARC held-out tasks by failed transform family.",
                        "Add support-only exact-grid candidate primitives for the largest unresolved clusters.",
                        "Require leave-one-support-out exact-grid validation before final deployment.",
                    ),
                    rationale="The current run has positive bounded RSI evidence but still leaves exact-grid accuracy below the configured target.",
                    expected_transfer_tags=("arc", "exact_grid", "exact_repair"),
                )
            )
        if cell is not None and self.config.cell_target - cell >= self.config.min_cell_gap:
            gap = self.config.cell_target - cell
            goals.append(
                ResearchGoal(
                    goal_id=_stable_id("goal", [item.evidence_id, "arc_cell", self.config.cell_target]),
                    title=f"Improve ARC cell accuracy for {item.mode} seed {item.seed}",
                    domain="arc_agi1_cell_accuracy",
                    target_metric="evolved_cell_accuracy",
                    current_value=cell,
                    target_value=self.config.cell_target,
                    priority=_clamp(self.config.frontier_weight + gap * 0.5 + cell_delta * 0.25),
                    feasible=cell_delta > 0.0,
                    strategy="frontier",
                    source_evidence_ids=(item.evidence_id,),
                    proposed_actions=(
                        "Expand candidate search around support-validated symbolic programs.",
                        "Track cell gains and exact-grid regressions as separate acceptance terms.",
                        "Retain deployment penalties for low-confidence symbolic programs.",
                    ),
                    rationale="Cell accuracy remains below target, so there is still a measurable bounded optimization frontier.",
                    expected_transfer_tags=("arc", "cell_accuracy"),
                )
            )
        if exact is not None and cell is not None and cell - exact > 0.25:
            goals.append(
                ResearchGoal(
                    goal_id=_stable_id("goal", [item.evidence_id, "cell_to_exact"]),
                    title=f"Convert ARC cell gains into exact grids for {item.mode} seed {item.seed}",
                    domain="arc_agi1_cell_to_exact_alignment",
                    target_metric="exact_alignment_from_cell_gains",
                    current_value=exact,
                    target_value=min(1.0, exact + 0.25),
                    priority=_clamp(self.config.frontier_weight + (cell - exact) * 0.35),
                    feasible=cell > 0.75,
                    strategy="frontier",
                    source_evidence_ids=(item.evidence_id,),
                    proposed_actions=(
                        "Penalize single-cell residual errors on support cross-validation when exact-grid candidates are available.",
                        "Add residual repair primitives that target boundary, row, column, and connected-component leftovers.",
                        "Report exact-grid success separately from mean cell accuracy.",
                    ),
                    rationale="High cell accuracy with lower exact-grid accuracy indicates residual structural errors rather than complete failure.",
                    expected_transfer_tags=("arc", "exact_grid"),
                )
            )
        accepted = _metric(metrics, "accepted_program_count")
        candidates = _metric(metrics, "candidate_count")
        rejected = _metric(metrics, "validation_rejected_count") or 0.0
        if candidates and accepted is not None and rejected > 0:
            acceptance_rate = accepted / max(candidates, 1.0)
            goals.append(
                ResearchGoal(
                    goal_id=_stable_id("goal", [item.evidence_id, "candidate_precision"]),
                    title=f"Improve candidate precision for {item.mode} seed {item.seed}",
                    domain="operator_program_search",
                    target_metric="accepted_program_rate",
                    current_value=acceptance_rate,
                    target_value=min(1.0, acceptance_rate + 0.10),
                    priority=_clamp(self.config.gap_weight * 0.7 + rejected / max(candidates, 1.0)),
                    feasible=True,
                    strategy="gap",
                    source_evidence_ids=(item.evidence_id,),
                    proposed_actions=(
                        "Use rejected candidate failure residues to bias the next candidate generator batch.",
                        "Increase diversity only in primitives that previously changed runtime behavior.",
                        "Keep validation and hidden-validation acceptance separate from final test scoring.",
                    ),
                    rationale="Rejected validation candidates provide direct information for improving the candidate generator.",
                    expected_transfer_tags=("operator_program", "failure_grammar"),
                )
            )
        return goals

    def _coding_goals(self, item: BenchmarkEvidence) -> List[ResearchGoal]:
        if "template_coding_harness" not in item.limitations and "template" not in item.mechanism_tags:
            return []
        return [
            ResearchGoal(
                goal_id=_stable_id("goal", [item.evidence_id, "non_template_coding"]),
                title="Replace template coding success with non-template synthesis evidence",
                domain="external_coding_generalization",
                target_metric="non_template_coding_generator_validated",
                current_value=0.0,
                target_value=self.config.coding_generalization_target,
                priority=_clamp(self.config.gap_weight + 0.25),
                feasible=True,
                strategy="gap",
                source_evidence_ids=(item.evidence_id,),
                proposed_actions=(
                    "Add a non-template program synthesizer baseline and evolved candidate path.",
                    "Evaluate on held-out coding tasks after candidate decisions are frozen.",
                    "Report template and non-template coding evidence as separate metrics.",
                ),
                rationale="Template-driven HumanEval success verifies harness integration, but it is not evidence of general coding synthesis.",
                expected_transfer_tags=("coding", "operator_program"),
            )
        ]

    def _creative_goals(self, evidence: Sequence[BenchmarkEvidence]) -> List[ResearchGoal]:
        tags = set(tag for item in evidence for tag in item.mechanism_tags)
        if not {"arc", "coding"} <= tags:
            return []
        evidence_ids = tuple(item.evidence_id for item in evidence if "arc" in item.mechanism_tags or "coding" in item.mechanism_tags)
        return [
            ResearchGoal(
                goal_id=_stable_id("goal", ["cross_domain", sorted(evidence_ids)]),
                title="Build a cross-domain operator-program transfer protocol",
                domain="cross_domain_operator_transfer",
                target_metric="cross_domain_transfer_protocol_coverage",
                current_value=0.0,
                target_value=1.0,
                priority=_clamp(self.config.creative_weight + 0.35),
                feasible=True,
                strategy="creative",
                source_evidence_ids=evidence_ids,
                proposed_actions=(
                    "Define shared operator-program features for ARC repair and coding synthesis.",
                    "Train transfer selection only on validation traces from both domains.",
                    "Score final cross-domain transfer on held-out ARC and held-out coding tasks.",
                ),
                rationale="The repository already has bounded DSL programs in multiple external adapters; the next useful goal is to test transfer across those adapters.",
                expected_transfer_tags=("operator_program", "arc", "coding"),
            )
        ]

    def _rank_and_dedupe(self, goals: Sequence[ResearchGoal]) -> List[ResearchGoal]:
        best_by_id: Dict[str, ResearchGoal] = {}
        for goal in goals:
            current = best_by_id.get(goal.goal_id)
            if current is None or goal.priority > current.priority:
                best_by_id[goal.goal_id] = goal
        ranked = sorted(best_by_id.values(), key=lambda g: (g.priority, g.feasible), reverse=True)
        return ranked[: self.config.max_goals]


def score_goal_set(goals: Sequence[ResearchGoal], evidence: Sequence[BenchmarkEvidence]) -> float:
    if not goals:
        return 0.0
    priorities = [goal.priority for goal in goals]
    feasible_rate = sum(1 for goal in goals if goal.feasible) / len(goals)
    transfer_rate = sum(1 for goal in goals if goal.strategy == "transfer") / len(goals)
    domains = {goal.domain for goal in goals}
    evidence_domains = {
        "arc" if "arc" in item.mechanism_tags else "coding" if "coding" in item.mechanism_tags else item.benchmark
        for item in evidence
    }
    coverage = len(domains) / max(1, len(evidence_domains) + 3)
    priority_score = mean(priorities)
    count_penalty = max(0.0, (len(goals) - 12) / 12.0)
    return _clamp(0.45 * priority_score + 0.25 * feasible_rate + 0.20 * coverage + 0.10 * transfer_rate - 0.10 * count_penalty)


class MetaMetaGoalController:
    """Bounded meta-meta search over the goal-generation policy."""

    def __init__(self, meta_config: MetaMetaConfig | None = None) -> None:
        self.meta_config = meta_config or MetaMetaConfig()

    def improve(
        self,
        initial_config: GoalPolicyConfig,
        evidence: Sequence[BenchmarkEvidence],
    ) -> MetaMetaResult:
        rng = random.Random(self.meta_config.seed)
        best_config = initial_config.normalized()
        best_goals = EvidenceGoalGenerator(best_config).generate(evidence)
        best_score = score_goal_set(best_goals, evidence)
        initial_score = best_score
        accepted: List[Dict[str, object]] = []

        for round_idx in range(max(1, self.meta_config.rounds)):
            candidates = [self._calibrated_config(best_config, evidence, round_idx)]
            for _ in range(max(1, self.meta_config.candidates_per_round)):
                candidates.append(self._mutate_config(best_config, rng))
            for candidate in candidates:
                candidate = candidate.normalized()
                goals = EvidenceGoalGenerator(candidate).generate(evidence)
                score = score_goal_set(goals, evidence)
                if score > best_score + 1e-9:
                    accepted.append(
                        {
                            "round": round_idx,
                            "previous_score": best_score,
                            "new_score": score,
                            "config": candidate.to_dict(),
                            "accepted": True,
                        }
                    )
                    best_config = candidate
                    best_score = score
                    best_goals = goals

        return MetaMetaResult(
            initial_score=initial_score,
            final_score=best_score,
            accepted_updates=tuple(accepted),
            initial_config=initial_config.normalized().to_dict(),
            final_config=best_config.to_dict(),
            goals=tuple(best_goals),
        )

    def _mutate_config(self, config: GoalPolicyConfig, rng: random.Random) -> GoalPolicyConfig:
        scale = self.meta_config.mutation_scale

        def jitter(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
            return _clamp(value + rng.uniform(-scale, scale), lo, hi)

        return replace(
            config,
            frontier_weight=max(0.0, config.frontier_weight + rng.uniform(-scale, scale)),
            gap_weight=max(0.0, config.gap_weight + rng.uniform(-scale, scale)),
            creative_weight=max(0.0, config.creative_weight + rng.uniform(-scale, scale)),
            transfer_weight=max(0.0, config.transfer_weight + rng.uniform(-scale, scale)),
            exact_target=jitter(config.exact_target, 0.30, 1.0),
            cell_target=jitter(config.cell_target, 0.40, 1.0),
            transfer_threshold=jitter(config.transfer_threshold, 0.10, 0.90),
        )

    def _calibrated_config(
        self,
        config: GoalPolicyConfig,
        evidence: Sequence[BenchmarkEvidence],
        round_idx: int,
    ) -> GoalPolicyConfig:
        exact_values = [
            value
            for item in evidence
            for value in [_metric(item.metrics, "evolved_exact_task_accuracy")]
            if value is not None
        ]
        cell_values = [
            value
            for item in evidence
            for value in [_metric(item.metrics, "evolved_cell_accuracy")]
            if value is not None
        ]
        positive_transfer = any(
            (_metric(item.metrics, "exact_task_accuracy_delta") or 0.0) > 0.0
            or (_metric(item.metrics, "cell_accuracy_delta") or 0.0) > 0.0
            for item in evidence
        )
        exact_target = config.exact_target
        if exact_values:
            exact_target = max(exact_target, min(0.98, max(exact_values) + 0.10 + 0.02 * round_idx))
        cell_target = config.cell_target
        if cell_values:
            cell_target = max(cell_target, min(0.995, max(cell_values) + 0.03))
        return replace(
            config,
            gap_weight=max(config.gap_weight, 0.42),
            transfer_weight=max(config.transfer_weight, 0.30 if positive_transfer else config.transfer_weight),
            exact_target=exact_target,
            cell_target=cell_target,
            max_goals=max(config.max_goals, 10),
        ).normalized()


def load_benchmark_evidence(paths: Sequence[str | Path]) -> List[BenchmarkEvidence]:
    evidence: List[BenchmarkEvidence] = []
    for path_like in paths:
        path = Path(path_like)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"benchmark evidence must be a JSON object: {path}")
        evidence.append(BenchmarkEvidence.from_result(payload, source_path=str(path)))
    return evidence


def build_goal_generation_report(
    evidence_paths: Sequence[str | Path],
    *,
    policy_config: GoalPolicyConfig | None = None,
    meta_config: MetaMetaConfig | None = None,
) -> Dict[str, object]:
    evidence = load_benchmark_evidence(evidence_paths)
    initial_config = (policy_config or GoalPolicyConfig()).normalized()
    controller = MetaMetaGoalController(meta_config)
    meta_result = controller.improve(initial_config, evidence)
    planner = TransferPlanner(GoalPolicyConfig(**meta_result.final_config))
    hypotheses = planner.plan(evidence)
    goals = list(meta_result.goals)
    metrics = summarize_goals(goals, evidence, meta_result)
    return {
        "benchmark": "research_goal_generation_controller",
        "evidence_count": len(evidence),
        "evidence": [item.to_dict() for item in evidence],
        "generated_goals": [goal.to_dict() for goal in goals],
        "transfer_hypotheses": [hyp.to_dict() for hyp in hypotheses],
        "meta_meta": meta_result.to_dict(),
        "metrics": metrics,
    }


def summarize_goals(
    goals: Sequence[ResearchGoal],
    evidence: Sequence[BenchmarkEvidence],
    meta_result: MetaMetaResult,
) -> Dict[str, float]:
    total = len(goals)
    feasible = sum(1 for goal in goals if goal.feasible)
    transfer = sum(1 for goal in goals if goal.strategy == "transfer")
    arc_exact = sum(1 for goal in goals if goal.domain == "arc_agi1_exact_grid")
    cross_domain = sum(1 for goal in goals if goal.domain == "cross_domain_operator_transfer")
    priorities = [goal.priority for goal in goals]
    return {
        "generated_goal_count": float(total),
        "feasible_goal_rate": feasible / max(1, total),
        "transfer_goal_count": float(transfer),
        "arc_exact_gap_goal_count": float(arc_exact),
        "cross_domain_goal_count": float(cross_domain),
        "mean_goal_priority": mean(priorities) if priorities else 0.0,
        "external_evidence_count": float(len(evidence)),
        "meta_config_score_delta": float(meta_result.final_score - meta_result.initial_score),
        "meta_config_final_score": float(meta_result.final_score),
    }

