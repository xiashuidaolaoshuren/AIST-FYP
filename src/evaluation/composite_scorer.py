"""Lightweight per-task composite scorer for hallucination detection.

The scorer uses pre-trained logistic parameters saved as JSON and computes
probabilities from per-sample signals. This keeps inference dependency-light and
lets us iterate model training offline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FEATURES = [
    "max_contradict_prob",
    "avg_coverage_score_all",
    "low_confidence_ratio",
    "avg_contradict_prob_low_conf",
    "avg_support_prob_low_conf",
    "contradictory_count",
    "num_claims",
    "low_coverage_ratio",
    "low_confidence_count",
]


@dataclass
class TaskCoefficients:
    intercept: float
    coefficients: dict[str, float]
    threshold: float = 0.5
    imputer_statistics: dict[str, float] | None = None
    scaler_mean: dict[str, float] | None = None
    scaler_scale: dict[str, float] | None = None


class CompositeScorer:
    """Task-scoped logistic scorer loaded from a JSON artifact."""

    def __init__(self, task_models: dict[str, TaskCoefficients]):
        self.task_models = task_models

    @classmethod
    def from_json(cls, path: str | Path) -> "CompositeScorer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_models = payload.get("task_models", {})
        if not isinstance(raw_models, dict):
            raise ValueError("Invalid composite scorer format: missing task_models map")

        task_models: dict[str, TaskCoefficients] = {}
        for task, model in raw_models.items():
            if not isinstance(model, dict):
                continue
            coefficients = model.get("coefficients", {})
            if not isinstance(coefficients, dict):
                coefficients = {}
            task_models[str(task)] = TaskCoefficients(
                intercept=float(model.get("intercept", 0.0)),
                coefficients={str(k): float(v) for k, v in coefficients.items()},
                threshold=float(model.get("threshold", 0.5)),
                imputer_statistics={
                    str(k): float(v)
                    for k, v in (model.get("imputer_statistics", {}) or {}).items()
                },
                scaler_mean={
                    str(k): float(v)
                    for k, v in (model.get("scaler_mean", {}) or {}).items()
                },
                scaler_scale={
                    str(k): float(v)
                    for k, v in (model.get("scaler_scale", {}) or {}).items()
                },
            )

        if not task_models:
            raise ValueError("No valid task models loaded from composite scorer JSON")
        return cls(task_models)

    def to_json(self, path: str | Path) -> None:
        payload = {
            "task_models": {
                task: {
                    "intercept": model.intercept,
                    "coefficients": model.coefficients,
                    "threshold": model.threshold,
                    "imputer_statistics": model.imputer_statistics or {},
                    "scaler_mean": model.scaler_mean or {},
                    "scaler_scale": model.scaler_scale or {},
                }
                for task, model in self.task_models.items()
            },
            "features": FEATURES,
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _sigmoid(logit: float) -> float:
        # Numerically stable logistic transform.
        if logit >= 0:
            z = math.exp(-logit)
            return 1.0 / (1.0 + z)
        z = math.exp(logit)
        return z / (1.0 + z)

    def score_sample(self, task_type: str, sample_signals: dict[str, Any]) -> float | None:
        model = self.task_models.get(task_type)
        if model is None:
            return None

        logit = model.intercept
        for feature in FEATURES:
            weight = model.coefficients.get(feature, 0.0)
            raw_value = sample_signals.get(feature)
            value = self._safe_float(raw_value)

            if raw_value is None and model.imputer_statistics and feature in model.imputer_statistics:
                value = float(model.imputer_statistics[feature])

            if model.scaler_mean and model.scaler_scale and feature in model.scaler_mean:
                mean = float(model.scaler_mean[feature])
                scale = float(model.scaler_scale.get(feature, 1.0))
                if scale != 0:
                    value = (value - mean) / scale

            logit += weight * value

        return self._sigmoid(logit)

    def predict_sample(self, task_type: str, sample_signals: dict[str, Any]) -> dict[str, Any] | None:
        score = self.score_sample(task_type, sample_signals)
        if score is None:
            return None

        threshold = self.task_models[task_type].threshold
        return {
            "score": score,
            "threshold": threshold,
            "predicted_hallucination": bool(score >= threshold),
        }
