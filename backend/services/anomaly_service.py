"""Analysis service: loads each model once, the first time an order needs it, and analyzes
single orders and batches with backend/pipeline/analyzer.py.

Rule violations and unfinished orders need no model, so they are analyzed even when a
model file is missing; a complete order then fails with a ModelLoadError naming the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.core.errors import InputError
from backend.ml.markov.model import MarkovPathModel, load_markov_model
from backend.ml.timing.model import TimingModel, load_timing_model
from backend.pipeline.analyzer import analyze_trace

SUPPORTED_VERDICTS: List[str] = ["NORMAL", "RULE_VIOLATION", "STATISTICAL_ANOMALY", "VALID_INCOMPLETE"]


@dataclass(frozen=True)
class OrderInput:
    """One order sent for analysis. Only event_sequence is required; timestamps and amount come together."""

    event_sequence: Union[str, List[str]]
    order_id: Optional[str] = None
    event_timestamps: Optional[List[str]] = None
    order_amount: Optional[float] = None


class AnomalyService:

    def __init__(self, markov_model_path: Path, timing_model_path: Path) -> None:
        self._markov_path = Path(markov_model_path)
        self._timing_path = Path(timing_model_path)
        self._markov: Optional[MarkovPathModel] = None
        self._timing: Optional[TimingModel] = None

    def markov_model(self) -> MarkovPathModel:
        """The Markov chain, loaded on first use. Raises ModelLoadError if the file is missing or invalid."""
        if self._markov is None:
            self._markov = load_markov_model(self._markov_path)
        return self._markov

    def timing_model(self) -> TimingModel:
        """The timing-and-amount model, loaded on first use. Raises ModelLoadError if missing or invalid."""
        if self._timing is None:
            self._timing = load_timing_model(self._timing_path)
        return self._timing

    def analyze_order(self, order: OrderInput) -> Dict[str, Any]:
        """Analyze one order; the result also echoes its order_id, timestamps and amount."""
        result = analyze_trace(order.event_sequence, order.event_timestamps, order.order_amount, models=self)
        return {
            "order_id": order.order_id,
            "event_timestamps": None if order.event_timestamps is None else list(order.event_timestamps),
            "order_amount": None if order.order_amount is None else float(order.order_amount),
            **result,
        }

    def analyze_batch(self, orders: Sequence[OrderInput]) -> Dict[str, Any]:
        """Analyze every order (models are loaded at most once) and count the verdicts.

        Raises:
            InputError: If the batch is empty.
        """
        if not orders:
            raise InputError("at least one order", 0, "traces")
        results = [self.analyze_order(order) for order in orders]
        counts = {verdict: sum(r["verdict"] == verdict for r in results) for verdict in SUPPORTED_VERDICTS}
        anomalies = counts["RULE_VIOLATION"] + counts["STATISTICAL_ANOMALY"]
        return {
            "results": results,
            "summary": {
                "total_orders": len(results),
                "normal_count": counts["NORMAL"],
                "rule_violation_count": counts["RULE_VIOLATION"],
                "statistical_anomaly_count": counts["STATISTICAL_ANOMALY"],
                "valid_incomplete_count": counts["VALID_INCOMPLETE"],
                "total_anomalies": anomalies,
                "anomaly_rate": anomalies / len(results),
            },
        }

    def readiness(self) -> Dict[str, Any]:
        """Load both models (raising ModelLoadError if either fails) and describe them."""
        markov, timing = self.markov_model(), self.timing_model()
        return {
            "service_status": "OPERATIONAL",
            "markov_model": {"path": str(self._markov_path), "model_name": "markov_chain",
                             "training_sample_count": markov.training_sample_count},
            "timing_model": {"path": str(self._timing_path), "model_name": timing.detector_name,
                             "training_sample_count": timing.training_sample_count},
            "supported_verdicts": list(SUPPORTED_VERDICTS),
        }
