"""Bounded self-model for code-level RSI candidate effects.

The self-model predicts candidate operator-program effects from allowed
train/validation evidence only.  Held-out test decisions are not valid training
inputs.  OOD-transfer-like targets are estimated from hidden validation pools in
the benchmark; final test results may be logged for audit after frozen
decisions, but this module refuses records marked as coming from the test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from operator_dsl import OperatorProgram, program_action_vector


SELF_MODEL_TARGETS = (
    "validation_improvement",
    "ood_transfer",
    "runtime_cost",
    "instability_risk",
    "future_candidate_quality",
    "controller_prediction_error",
    "validation_to_test_gap",
)


@dataclass(frozen=True)
class SelfModelRecord:
    program_id: str
    generation: int
    feature_vector: List[float]
    actual_effects: Dict[str, float]
    split: str = "validation"

    def to_dict(self) -> Dict[str, object]:
        return {
            "program_id": self.program_id,
            "generation": self.generation,
            "feature_vector": [float(x) for x in self.feature_vector],
            "actual_effects": {key: float(value) for key, value in self.actual_effects.items()},
            "split": self.split,
        }


@dataclass
class SelfModelPrediction:
    program_id: str
    generation: int
    predicted_effects: Dict[str, float]
    actual_effects: Dict[str, float] | None = None
    absolute_errors: Dict[str, float] = field(default_factory=dict)
    used_for_selection: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "program_id": self.program_id,
            "generation": self.generation,
            "predicted_effects": {key: float(value) for key, value in self.predicted_effects.items()},
            "actual_effects": None if self.actual_effects is None else {key: float(value) for key, value in self.actual_effects.items()},
            "absolute_errors": {key: float(value) for key, value in self.absolute_errors.items()},
            "used_for_selection": self.used_for_selection,
        }


class CandidateSelfModel:
    """Small CPU-runnable regressor over bounded operator-program features."""

    def __init__(self, feature_dim: int = 12, hidden_dim: int = 32, seed: int = 0):
        torch.manual_seed(seed)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.target_names = list(SELF_MODEL_TARGETS)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, len(self.target_names)),
        )
        self.records: List[SelfModelRecord] = []
        self.prediction_log: List[SelfModelPrediction] = []
        self.train_loss_history: List[float] = []

    def features_for_program(self, program: OperatorProgram, generation: int, failure_rule_count: int = 0, evaluator_pressure: float = 0.0) -> torch.Tensor:
        action = program_action_vector(program).float().flatten()
        primitive_names = {step.name for step in program.primitive_sequence}
        extras = torch.tensor(
            [
                float(generation),
                float(len(program.primitive_sequence)),
                float(failure_rule_count),
                float(evaluator_pressure),
                1.0 if any("memory" in name for name in primitive_names) else 0.0,
                1.0 if any("world_model" in name or "predicted" in name for name in primitive_names) else 0.0,
            ],
            dtype=torch.float32,
        )
        vector = torch.cat([action, extras])
        if vector.numel() >= self.feature_dim:
            return vector[: self.feature_dim]
        return torch.cat([vector, torch.zeros(self.feature_dim - vector.numel())])

    def predict(self, program: OperatorProgram, generation: int, failure_rule_count: int = 0, evaluator_pressure: float = 0.0, *, used_for_selection: bool = False) -> SelfModelPrediction:
        features = self.features_for_program(program, generation, failure_rule_count, evaluator_pressure)
        with torch.no_grad():
            raw = self.net(features.reshape(1, -1)).reshape(-1)
        effects = {name: float(raw[i]) for i, name in enumerate(self.target_names)}
        prediction = SelfModelPrediction(program.program_id, generation, effects, used_for_selection=used_for_selection)
        self.prediction_log.append(prediction)
        return prediction

    def add_record(self, record: SelfModelRecord) -> None:
        if record.split == "test":
            raise ValueError("self-model cannot train on held-out test split records")
        missing = set(self.target_names).difference(record.actual_effects)
        if missing:
            raise ValueError(f"self-model record missing targets: {sorted(missing)}")
        self.records.append(record)

    def fit(self, steps: int = 30, lr: float = 0.01) -> Dict[str, float]:
        if not self.records:
            return {"initial_loss": 0.0, "final_loss": 0.0, "record_count": 0.0}
        x = torch.tensor([record.feature_vector for record in self.records], dtype=torch.float32)
        y = torch.tensor([[record.actual_effects[name] for name in self.target_names] for record in self.records], dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        with torch.no_grad():
            initial = float(F.mse_loss(self.net(x), y))
        for _ in range(max(1, int(steps))):
            opt.zero_grad()
            loss = F.mse_loss(self.net(x), y)
            loss.backward()
            opt.step()
            self.train_loss_history.append(float(loss.detach()))
        with torch.no_grad():
            final = float(F.mse_loss(self.net(x), y))
        return {"initial_loss": initial, "final_loss": final, "record_count": float(len(self.records))}

    def record_actual(self, program: OperatorProgram, generation: int, actual_effects: Dict[str, float], *, split: str = "validation") -> Dict[str, float]:
        record = SelfModelRecord(program.program_id, generation, self.features_for_program(program, generation).tolist(), _complete_effects(actual_effects), split=split)
        self.add_record(record)
        errors = {}
        for prediction in reversed(self.prediction_log):
            if prediction.program_id == program.program_id and prediction.generation == generation and prediction.actual_effects is None:
                prediction.actual_effects = record.actual_effects
                prediction.absolute_errors = {name: abs(prediction.predicted_effects.get(name, 0.0) - record.actual_effects[name]) for name in self.target_names}
                errors = prediction.absolute_errors
                break
        return errors

    def mean_prediction_error(self) -> float:
        errors = [mean(item.absolute_errors.values()) for item in self.prediction_log if item.absolute_errors]
        return float(mean(errors)) if errors else 0.0

    def future_quality_error(self) -> float:
        vals = [item.absolute_errors["future_candidate_quality"] for item in self.prediction_log if "future_candidate_quality" in item.absolute_errors]
        return float(mean(vals)) if vals else 0.0

    def selection_score(self, prediction: SelfModelPrediction) -> float:
        effects = prediction.predicted_effects
        return float(
            effects.get("validation_improvement", 0.0)
            + 0.4 * effects.get("ood_transfer", 0.0)
            + 0.2 * effects.get("future_candidate_quality", 0.0)
            - 0.05 * effects.get("runtime_cost", 0.0)
            - 0.5 * effects.get("instability_risk", 0.0)
            - 0.1 * abs(effects.get("validation_to_test_gap", 0.0))
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_names": list(self.target_names),
            "record_count": len(self.records),
            "records": [record.to_dict() for record in self.records],
            "prediction_log": [prediction.to_dict() for prediction in self.prediction_log],
            "train_loss_history": [float(x) for x in self.train_loss_history],
        }


def _complete_effects(values: Dict[str, float]) -> Dict[str, float]:
    complete = {name: float(values.get(name, 0.0)) for name in SELF_MODEL_TARGETS}
    return complete

