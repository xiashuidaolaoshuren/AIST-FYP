"""LettuceDetect-backed detector adapter for the verifier NLI interface.

This adapter maps token-level hallucination probabilities from LettuceDetect to
the existing NLI-shaped signal contract used by the rule-based aggregator:
    - contradiction ~= hallucination probability
    - entailment ~= 1 - contradiction
    - neutral = 0.0
"""

from __future__ import annotations

from typing import Dict, Any

from src.utils.config import Config
from src.utils.logger import setup_logger


class LettuceDetectDetector:
    """Adapter that exposes a drop-in NLI-like interface for LettuceDetect."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(__name__)

        nli_cfg = getattr(getattr(config, "verification", None), "nli", None)
        self.model_name = getattr(
            nli_cfg,
            "model_name",
            "KRLabsOrg/lettucedect-base-modernbert-en-v1",
        )
        self.device = getattr(nli_cfg, "device", "cuda")
        self._nli_batch_size = max(1, int(getattr(nli_cfg, "batch_size", 32)))

        try:
            from lettucedetect.models.inference import HallucinationDetector  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "LettuceDetect backend requested but package is unavailable. "
                "Install with 'pip install lettucedetect'."
            ) from exc

        init_kwargs: dict[str, Any] = {
            "method": "transformer",
            "model_path": self.model_name,
        }
        if isinstance(self.device, str) and self.device.strip():
            init_kwargs["device"] = self.device

        self.detector = HallucinationDetector(**init_kwargs)
        self.logger.info(
            "LettuceDetectDetector initialized (model=%s, device=%s, batch_size=%s)",
            self.model_name,
            self.device,
            self._nli_batch_size,
        )

    @staticmethod
    def _coerce_probability(prediction_output: Any) -> float:
        """Extract a robust hallucination probability from heterogeneous outputs."""
        if prediction_output is None:
            return 0.5

        if isinstance(prediction_output, (int, float)):
            return float(max(0.0, min(1.0, prediction_output)))

        if isinstance(prediction_output, dict):
            for key in (
                "hallucination_probability",
                "probability",
                "score",
                "confidence",
            ):
                value = prediction_output.get(key)
                if isinstance(value, (int, float)):
                    return float(max(0.0, min(1.0, value)))

            spans = prediction_output.get("spans")
            if isinstance(spans, list):
                return LettuceDetectDetector._coerce_probability(spans)

        if isinstance(prediction_output, list):
            max_prob = 0.0
            found = False
            for item in prediction_output:
                if isinstance(item, dict):
                    for key in (
                        "hallucination_probability",
                        "probability",
                        "score",
                        "confidence",
                    ):
                        value = item.get(key)
                        if isinstance(value, (int, float)):
                            found = True
                            max_prob = max(max_prob, float(value))
                            break
                elif isinstance(item, (int, float)):
                    found = True
                    max_prob = max(max_prob, float(item))
            if found:
                return float(max(0.0, min(1.0, max_prob)))

        return 0.5

    @staticmethod
    def _to_nli_scores(hallucination_prob: float) -> Dict[str, float]:
        contradiction = float(max(0.0, min(1.0, hallucination_prob)))
        entailment = float(max(0.0, min(1.0, 1.0 - contradiction)))
        return {
            "entailment": entailment,
            "neutral": 0.0,
            "contradiction": contradiction,
        }

    def detect(self, claim_text: str, evidence_text: str) -> Dict[str, float]:
        """Run LettuceDetect for one claim/evidence pair using NLI-shaped output."""
        if not claim_text or not claim_text.strip():
            raise ValueError("claim_text cannot be empty")
        if not evidence_text or not evidence_text.strip():
            raise ValueError("evidence_text cannot be empty")

        try:
            output = self.detector.predict(
                context=[evidence_text],
                question="",
                answer=claim_text,
                output_format="spans",
            )
            hallucination_prob = self._coerce_probability(output)
            scores = self._to_nli_scores(hallucination_prob)
            self.logger.debug(
                "LettuceDetect scores - entailment: %.3f, contradiction: %.3f",
                scores["entailment"],
                scores["contradiction"],
            )
            return scores
        except Exception as exc:
            self.logger.warning("LettuceDetect inference failed; returning neutral fallback: %s", exc)
            return {
                "entailment": 0.33,
                "neutral": 0.34,
                "contradiction": 0.33,
            }

    def detect_batch(self, claim_texts: list[str], evidence_texts: list[str]) -> list[Dict[str, float]]:
        """Run batch inference via iterative calls to maintain interface parity."""
        if len(claim_texts) != len(evidence_texts):
            raise ValueError("claim_texts and evidence_texts must have equal length")

        outputs: list[Dict[str, float]] = []
        for claim_text, evidence_text in zip(claim_texts, evidence_texts):
            outputs.append(self.detect(claim_text=claim_text, evidence_text=evidence_text))
        return outputs
