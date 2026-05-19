"""Track failure-residue lifecycle across repeated runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Dict, Mapping, Sequence


@dataclass(frozen=True)
class ResidueLifecycleEntry:
    failure_type: str
    status: str
    previous_count: int
    current_count: int
    associated_mutation_methods: list[str] = field(default_factory=list)
    associated_module_families: list[str] = field(default_factory=list)
    attempted_repair_outcome: str = "unknown"
    next_recommended_repair: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "failure_type": self.failure_type,
            "status": self.status,
            "previous_count": self.previous_count,
            "current_count": self.current_count,
            "associated_mutation_methods": list(self.associated_mutation_methods),
            "associated_module_families": list(self.associated_module_families),
            "attempted_repair_outcome": self.attempted_repair_outcome,
            "next_recommended_repair": self.next_recommended_repair,
        }


def build_residue_lifecycle_report(
    previous_run: Mapping[str, object] | None,
    current_run: Mapping[str, object],
    attribution_report: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    previous_counts = _residue_counts(previous_run or {})
    current_counts = _residue_counts(current_run)
    methods_by_failure = _methods_by_failure(current_run)
    modules_by_failure = _modules_by_failure(current_run)
    outcomes = _repair_outcomes(attribution_report)
    entries: list[ResidueLifecycleEntry] = []
    for failure_type in sorted(set(previous_counts) | set(current_counts)):
        prev = previous_counts.get(failure_type, 0)
        cur = current_counts.get(failure_type, 0)
        if prev == 0 and cur > 0:
            status = "appeared"
        elif prev > 0 and cur == 0:
            status = "resolved"
        elif cur < prev:
            status = "reduced"
        elif cur > prev:
            status = "reappeared" if prev == 0 else "persisted"
        else:
            status = "persisted" if cur > 0 else "absent"
        outcome = outcomes.get(failure_type, "unknown")
        entries.append(
            ResidueLifecycleEntry(
                failure_type=failure_type,
                status=status,
                previous_count=prev,
                current_count=cur,
                associated_mutation_methods=sorted(methods_by_failure.get(failure_type, set())),
                associated_module_families=sorted(modules_by_failure.get(failure_type, set())),
                attempted_repair_outcome=outcome,
                next_recommended_repair=_next_repair(failure_type, status, outcome),
            )
        )
    unresolved = [
        entry.failure_type
        for entry in entries
        if entry.status in {"persisted", "reappeared"} and entry.attempted_repair_outcome == "harmed"
    ]
    return {
        "residue_lifecycle": [entry.to_dict() for entry in entries],
        "residue_lifecycle_report": [entry.to_dict() for entry in entries],
        "unresolved_residue_types": unresolved,
        "residue_repair_success_by_method": _repair_by_method(attribution_report, helped=True),
        "residue_repair_failure_by_method": _repair_by_method(attribution_report, helped=False),
        "next_residue_repair_plan": [
            entry.next_recommended_repair
            for entry in entries
            if entry.next_recommended_repair
        ],
    }


def build_residue_lifecycle_sequence_report(
    runs: Sequence[Mapping[str, object]],
    attribution_reports: Sequence[Mapping[str, object]] = (),
) -> Dict[str, object]:
    """Summarize residue lifecycle across a long-horizon run sequence."""

    counts_by_run = [_residue_counts(run) for run in runs]
    all_failure_types = sorted({failure for counts in counts_by_run for failure in counts})
    methods_by_failure: Dict[str, set[str]] = defaultdict(set)
    modules_by_failure: Dict[str, set[str]] = defaultdict(set)
    for run in runs:
        for failure, methods in _methods_by_failure(run).items():
            methods_by_failure[failure].update(methods)
        for failure, modules in _modules_by_failure(run).items():
            modules_by_failure[failure].update(modules)
    outcomes = _sequence_repair_outcomes(attribution_reports)
    entries: list[Dict[str, object]] = []
    unresolved: list[str] = []
    for failure_type in all_failure_types:
        series = [counts.get(failure_type, 0) for counts in counts_by_run]
        transitions = _status_sequence(series)
        final_status = transitions[-1] if transitions else ("absent" if not series or series[-1] == 0 else "appeared")
        outcome = outcomes.get(failure_type, "unknown")
        repeated_failed_attempts = outcome == "harmed" and sum(1 for status in transitions if status in {"persisted", "reappeared"}) >= 1
        entry = {
            "failure_type": failure_type,
            "status": final_status,
            "status_sequence": transitions,
            "count_by_run": series,
            "associated_mutation_methods": sorted(methods_by_failure.get(failure_type, set())),
            "associated_module_families": sorted(modules_by_failure.get(failure_type, set())),
            "attempted_repair_outcome": outcome,
            "unresolved_by_current_module_family": bool(repeated_failed_attempts),
            "next_recommended_repair": _next_repair(failure_type, final_status, "harmed" if repeated_failed_attempts else outcome),
        }
        if series and series[-1] > 0 and (final_status in {"persisted", "reappeared"} or repeated_failed_attempts):
            unresolved.append(failure_type)
        entries.append(entry)
    success_by_method = Counter()
    failure_by_method = Counter()
    for report in attribution_reports:
        success_by_method.update(_repair_by_method(report, helped=True))
        failure_by_method.update(_repair_by_method(report, helped=False))
    return {
        "residue_lifecycle_report": entries,
        "unresolved_residue_types": sorted(set(unresolved)),
        "residue_repair_success_by_method": dict(success_by_method),
        "residue_repair_failure_by_method": dict(failure_by_method),
        "next_residue_repair_plan": [
            str(entry["next_recommended_repair"])
            for entry in entries
            if entry.get("next_recommended_repair")
        ],
    }


def _residue_counts(run: Mapping[str, object]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    raw = run.get("failure_residues", [])
    if not isinstance(raw, list):
        return counts
    for item in raw:
        if isinstance(item, Mapping):
            failure = str(item.get("failure_type", ""))
            if failure:
                counts[failure] = counts.get(failure, 0) + 1
    return counts


def _methods_by_failure(run: Mapping[str, object]) -> Dict[str, set[str]]:
    failures = sorted(_residue_counts(run))
    decisions = run.get("architecture_decisions", [])
    out = {failure: set() for failure in failures}
    if not isinstance(decisions, list):
        return out
    for decision in decisions:
        if isinstance(decision, Mapping):
            method = str(decision.get("mutation_method", ""))
            if method:
                for failure in failures:
                    out[failure].add(method)
    return out


def _modules_by_failure(run: Mapping[str, object]) -> Dict[str, set[str]]:
    out: Dict[str, set[str]] = {}
    raw = run.get("failure_residues", [])
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        failure = str(item.get("failure_type", ""))
        module = str(item.get("proposed_operator_or_module_family", ""))
        if failure and module:
            out.setdefault(failure, set()).add(module)
    return out


def _repair_outcomes(report: Mapping[str, object] | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not isinstance(report, Mapping):
        return out
    changes = report.get("changes", [])
    if not isinstance(changes, list):
        return out
    for change in changes:
        if not isinstance(change, Mapping):
            continue
        effect = change.get("observed_effect", {})
        if not isinstance(effect, Mapping):
            continue
        failures = effect.get("target_failure_types", [])
        if not isinstance(failures, list):
            continue
        label = "helped" if change.get("helped") else "harmed" if change.get("harmed") else "neutral"
        for failure in failures:
            out[str(failure)] = label
    return out


def _repair_by_method(report: Mapping[str, object] | None, *, helped: bool) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(report, Mapping):
        return out
    changes = report.get("changes", [])
    if not isinstance(changes, list):
        return out
    for change in changes:
        if not isinstance(change, Mapping) or bool(change.get("helped")) is not helped:
            continue
        effect = change.get("observed_effect", {})
        method = effect.get("mutation_method", "") if isinstance(effect, Mapping) else ""
        if method:
            out[str(method)] = out.get(str(method), 0) + 1
    return out


def _next_repair(failure_type: str, status: str, outcome: str) -> str:
    if status == "resolved":
        return ""
    if outcome == "harmed":
        return f"mark {failure_type} unresolved by current module family and propose a new bounded module hypothesis"
    if status in {"persisted", "reappeared"}:
        return f"target {failure_type} only under rollback budget with hidden-validation confirmation"
    if status == "appeared":
        return f"collect validation residue evidence for {failure_type} before increasing search breadth"
    return ""


def _status_sequence(series: Sequence[int]) -> list[str]:
    statuses: list[str] = []
    seen_positive = False
    for index, count in enumerate(series):
        prev = series[index - 1] if index else 0
        if prev == 0 and count > 0:
            status = "reappeared" if seen_positive else "appeared"
        elif prev > 0 and count == 0:
            status = "resolved"
        elif count > prev:
            status = "persisted"
        elif 0 < count < prev:
            status = "reduced"
        elif count > 0:
            status = "persisted"
        else:
            status = "absent"
        if count > 0:
            seen_positive = True
        statuses.append(status)
    return statuses


def _sequence_repair_outcomes(reports: Sequence[Mapping[str, object]]) -> Dict[str, str]:
    votes: Dict[str, Counter[str]] = defaultdict(Counter)
    for report in reports:
        for failure, label in _repair_outcomes(report).items():
            votes[failure][label] += 1
    out: Dict[str, str] = {}
    for failure, counter in votes.items():
        if counter.get("harmed", 0) > 0:
            out[failure] = "harmed"
        elif counter.get("helped", 0) > 0:
            out[failure] = "helped"
        elif counter:
            out[failure] = counter.most_common(1)[0][0]
    return out
