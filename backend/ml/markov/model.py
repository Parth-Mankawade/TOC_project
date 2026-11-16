"""Markov chain path model: how likely is a legal order path under normal behaviour.

The FSM (backend/fsm/transitions.py) says which transitions are allowed. This
model learns how often each allowed transition is taken in normal orders:

    P(event | state) = (count(state, event) + alpha) / (count(state) + alpha * k)

where k is the number of events allowed in that state. alpha > 0 (Laplace
smoothing) gives allowed-but-never-seen transitions a small non-zero
probability instead of zero.

The probability of a whole path is the product of its step probabilities, so a
path that repeats rare steps (for example many FAIL_PAYMENT/RETRY_PAYMENT
loops) becomes very unlikely, while a rare but ordinary path (for example
FAIL_PAYMENT then CANCEL_ORDER) stays reasonably likely.

Decision rule (set on validation orders the model was not fitted on):
    threshold = THRESHOLD_PERCENTILE-th percentile of log10 P(path) over the
    normal validation orders. A path is an anomaly if log10 P(path) < threshold,
    i.e. it is rarer than all but the rarest 1% of normal orders' paths.

    anomaly_score = threshold - log10 P(path)
    anomaly_score > 0 means anomaly. Each +1 means 10x less likely than the
    threshold.

This is a probabilistic finite automaton: the FSM says which transitions are
allowed, the learned probabilities say how likely each allowed one is.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from backend.core.errors import FeatureError, ModelLoadError, TrainError
from backend.fsm.transitions import TRANSITION_TABLE, get_allowed_events
from backend.fsm.validator import validate_trace

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "markov_chain.json"

MODEL_NAME = "MarkovChain"
DEFAULT_ALPHA = 1.0
THRESHOLD_PERCENTILE = 1.0     # flag paths rarer than 99% of normal validation orders' paths
MIN_VALIDATION_ORDERS = 100

# Probabilities in a saved model are rounded; per-state sums may drift by this much.
_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class MarkovPathModel:
    """Transition probabilities for every FSM transition, plus the decision threshold."""

    probabilities: Dict[Tuple[str, str], float]
    counts: Dict[Tuple[str, str], int]
    alpha: float
    training_sample_count: int
    validation_sample_count: int
    threshold_log10_probability: float


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _valid_complete(trace: Sequence[str], what: str) -> Dict[str, Any]:
    result = validate_trace(list(trace))
    if not (result["is_valid"] and result["is_complete"]):
        raise TrainError(
            "valid, complete normal orders",
            f"{list(trace)} (is_valid={result['is_valid']}, is_complete={result['is_complete']})",
            what,
        )
    return result


def _log10_path(probabilities: Dict[Tuple[str, str], float], history: Sequence[Dict[str, Any]]) -> float:
    return sum(math.log10(probabilities[(step["from_state"], step["event"])]) for step in history)


def fit_markov_model(
    traces: Sequence[Sequence[str]],
    validation_traces: Sequence[Sequence[str]],
    alpha: float = DEFAULT_ALPHA,
) -> MarkovPathModel:
    """Estimate transition probabilities from normal training orders, then set the
    threshold from normal validation orders.

    Raises:
        TrainError: No training orders, alpha not positive, fewer than MIN_VALIDATION_ORDERS
            validation orders, or any order that is not FSM-valid and complete.
    """
    if not traces:
        raise TrainError("at least one training order", 0, "fit_markov_model")
    if not alpha > 0:
        raise TrainError("a positive smoothing alpha", alpha, "fit_markov_model")
    if len(validation_traces) < MIN_VALIDATION_ORDERS:
        raise TrainError(f"at least {MIN_VALIDATION_ORDERS} normal validation orders", len(validation_traces),
                         "fit_markov_model")

    counts: Counter = Counter()
    for idx, trace in enumerate(traces):
        for step in _valid_complete(trace, f"training order {idx}")["history"]:
            counts[(step["from_state"], step["event"])] += 1

    probabilities: Dict[Tuple[str, str], float] = {}
    for state in sorted({state for state, _ in TRANSITION_TABLE}):
        allowed = get_allowed_events(state)
        state_total = sum(counts[(state, event)] for event in allowed)
        denominator = state_total + alpha * len(allowed)
        for event in allowed:
            probabilities[(state, event)] = (counts[(state, event)] + alpha) / denominator

    validation_logs = [
        _log10_path(probabilities, _valid_complete(trace, f"validation order {idx}")["history"])
        for idx, trace in enumerate(validation_traces)
    ]
    return MarkovPathModel(
        probabilities=probabilities,
        counts={key: counts[key] for key in probabilities},
        alpha=float(alpha),
        training_sample_count=len(traces),
        validation_sample_count=len(validation_traces),
        threshold_log10_probability=float(np.percentile(validation_logs, THRESHOLD_PERCENTILE)),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_trace(model: MarkovPathModel, events: Sequence[str]) -> Dict[str, Any]:
    """Score an FSM-valid trace.

    Returns:
        Dictionary with:
          - prediction ("NORMAL" or "ANOMALY") and is_ml_anomaly
          - anomaly_score: threshold - log10 P(path); > 0 means anomaly
          - path_log10_probability: log10 P(path)
          - transitions: one entry per step with its probability

    Raises:
        FeatureError: If the trace breaks an FSM rule (it has no probability).
    """
    result = validate_trace(list(events))
    if not result["is_valid"]:
        raise FeatureError("an FSM-valid order", list(events), result["error"]["message"])

    transitions: List[Dict[str, Any]] = []
    path_log10_probability = 0.0
    for step in result["history"]:
        probability = model.probabilities[(step["from_state"], step["event"])]
        path_log10_probability += math.log10(probability)
        transitions.append({
            "step": step["step"],
            "from_state": step["from_state"],
            "event": step["event"],
            "to_state": step["to_state"],
            "probability": probability,
        })

    anomaly_score = model.threshold_log10_probability - path_log10_probability
    is_anomaly = anomaly_score > 0.0

    return {
        "prediction": "ANOMALY" if is_anomaly else "NORMAL",
        "is_ml_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "path_log10_probability": path_log10_probability,
        "transitions": transitions,
    }


def describe_probability(log10_probability: float) -> str:
    """Plain-language frequency, e.g. '79.1% of orders' or 'about 1 in 27,300 orders'."""
    probability = 10 ** log10_probability
    if probability >= 0.01:
        return f"{probability:.1%} of orders"
    return f"about 1 in {round(1 / probability):,} orders"


# ---------------------------------------------------------------------------
# Persistence (plain JSON so the learned probabilities can be read directly)
# ---------------------------------------------------------------------------

def model_to_dict(model: MarkovPathModel) -> Dict[str, Any]:
    """Serialise the model into a JSON-compatible dictionary."""
    return {
        "model_name": MODEL_NAME,
        "smoothing_alpha": model.alpha,
        "training_sample_count": model.training_sample_count,
        "validation_sample_count": model.validation_sample_count,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold_rule": "percentile of log10 P(path) over normal validation orders",
        "threshold_log10_probability": model.threshold_log10_probability,
        "transitions": [
            {
                "from_state": state,
                "event": event,
                "to_state": TRANSITION_TABLE[(state, event)],
                "count": model.counts[(state, event)],
                "probability": model.probabilities[(state, event)],
            }
            for (state, event) in sorted(model.probabilities)
        ],
    }


def model_from_dict(data: Dict[str, Any], source: str = "markov model") -> MarkovPathModel:
    """Rebuild a model from model_to_dict() output, checking it matches the FSM.

    Raises:
        ModelLoadError: Missing fields, transitions that differ from the FSM table,
            or invalid probabilities.
    """
    required = ["model_name", "smoothing_alpha", "training_sample_count", "validation_sample_count",
                "threshold_log10_probability", "transitions"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ModelLoadError(f"fields {required}", f"missing {missing}", source)
    if data["model_name"] != MODEL_NAME:
        raise ModelLoadError(f"model_name '{MODEL_NAME}'", data["model_name"], source)

    probabilities: Dict[Tuple[str, str], float] = {}
    counts: Dict[Tuple[str, str], int] = {}
    for entry in data["transitions"]:
        key = (entry["from_state"], entry["event"])
        if key not in TRANSITION_TABLE or TRANSITION_TABLE[key] != entry["to_state"]:
            raise ModelLoadError("only transitions from the FSM table",
                                 f"{key[0]} --{key[1]}--> {entry['to_state']}", source)
        probability = float(entry["probability"])
        if not 0.0 < probability <= 1.0:
            raise ModelLoadError("probabilities in (0, 1]", probability, f"{source}: {key}")
        probabilities[key] = probability
        counts[key] = int(entry["count"])

    if set(probabilities) != set(TRANSITION_TABLE):
        missing_keys = sorted(set(TRANSITION_TABLE) - set(probabilities))
        raise ModelLoadError(f"transitions for all {len(TRANSITION_TABLE)} FSM transitions",
                             f"missing {missing_keys}", source)

    for state in sorted({state for state, _ in TRANSITION_TABLE}):
        state_sum = sum(p for (s, _), p in probabilities.items() if s == state)
        if abs(state_sum - 1.0) > _SUM_TOLERANCE:
            raise ModelLoadError(f"probabilities out of {state} that sum to 1", state_sum, source)

    return MarkovPathModel(
        probabilities=probabilities,
        counts=counts,
        alpha=float(data["smoothing_alpha"]),
        training_sample_count=int(data["training_sample_count"]),
        validation_sample_count=int(data["validation_sample_count"]),
        threshold_log10_probability=float(data["threshold_log10_probability"]),
    )


def save_markov_model(model: MarkovPathModel, path: Optional[Union[str, Path]] = None) -> Path:
    """Write the model to JSON and return the path."""
    target = Path(path) if path else DEFAULT_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(model_to_dict(model), fh, indent=2)
        fh.write("\n")
    return target


def load_markov_model(path: Optional[Union[str, Path]] = None) -> MarkovPathModel:
    """Load and validate a model saved by save_markov_model().

    Raises:
        ModelLoadError: Missing file, invalid JSON, or a model that does not match the FSM.
    """
    target = Path(path) if path else DEFAULT_MODEL_PATH
    if not target.is_file():
        raise ModelLoadError("a saved Markov model (run: python -m backend.ml.pipeline)", "no file", str(target))
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelLoadError("valid JSON", str(exc), str(target)) from exc
    return model_from_dict(data, str(target))
