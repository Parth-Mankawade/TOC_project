"""Alternatives that were measured and NOT adopted. Nothing here is used by the app.

They stay in the repository only so that backend/ml/evaluation/benchmark.py can
reproduce the numbers behind each decision (see IMPROVEMENTS.md):

  two refund inputs          the cluster detector also sees the z-score of the wait
                             before a refund, and the amount z-score of refunded orders
  retry-aware Markov chain   the state also remembers how many payments have failed
  detector ensemble          rank average of Isolation Forest, k-means and LOF
  stacking                   logistic regression over the four check levels, which
                             needs labelled anomalies
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backend.core.errors import TrainError
from backend.fsm.transitions import get_allowed_events
from backend.fsm.validator import validate_trace
from backend.ml.timing import model as timing_model
from backend.ml.timing.features import FeatureStats, TimedOrder, order_features, step_durations

RETRY_CAP = 2   # states: no failed payment yet, one, two or more


def _inputs_with_refund(stats: FeatureStats, orders: Sequence[TimedOrder]) -> np.ndarray:
    """MODEL_INPUTS plus refund_step_z and refunded_amount_z (both 0 for an order that was not refunded)."""
    rows = []
    for events, timestamps, amount in orders:
        features = order_features(stats, events, timestamps, amount)
        refunded = events[-1] == "REFUND_ORDER"
        last = step_durations(events, timestamps)[-1] if refunded else None
        rows.append(list(features.values()) + [stats.step(last).z(last.seconds) if refunded else 0.0,
                                               features["amount_z"] if refunded else 0.0])
    return np.asarray(rows, dtype=np.float64)


def refund_input_detectors(stats: FeatureStats, train: Sequence[TimedOrder]
                           ) -> Dict[str, Callable[[Sequence[TimedOrder]], np.ndarray]]:
    """Every detector candidate fitted on the nine inputs: {name: raw-score function}."""
    scaler = StandardScaler().fit(_inputs_with_refund(stats, train))
    X = scaler.transform(_inputs_with_refund(stats, train))
    scorers = {}
    for name in timing_model.DETECTOR_NAMES:
        detector, _ = timing_model._fit_detector(name, X)
        scorers[name] = (lambda orders, name=name, detector=detector: timing_model._raw_scores(
            name, detector, scaler.transform(_inputs_with_refund(stats, orders))))
    return scorers


def _retry_aware_steps(events: Sequence[str]) -> List[Tuple[Tuple[str, int], str]]:
    """((state, failed payments so far capped at RETRY_CAP), event) for every step."""
    result = validate_trace(list(events))
    if not result["is_valid"]:
        raise TrainError("an FSM-valid order", list(events), "retry-aware Markov chain")
    steps, failures = [], 0
    for step in result["history"]:
        steps.append(((step["from_state"], min(failures, RETRY_CAP)), step["event"]))
        failures += step["event"] == "FAIL_PAYMENT"
    return steps


def fit_retry_aware_path(traces: Sequence[Sequence[str]], alpha: float = 1.0) -> Callable[[Sequence[str]], float]:
    """A Markov chain over (state, failed payments so far): returns log10 P(path).

    With this state a first payment attempt and an attempt after a failure get
    different probabilities, which the plain chain cannot do.
    """
    counts: Counter = Counter()
    for trace in traces:
        counts.update(_retry_aware_steps(trace))

    def probability(state: Tuple[str, int], event: str) -> float:
        allowed = get_allowed_events(state[0])
        total = sum(counts[(state, e)] for e in allowed)
        return (counts[(state, event)] + alpha) / (total + alpha * len(allowed))

    def log10_path(events: Sequence[str]) -> float:
        return sum(math.log10(probability(state, event)) for state, event in _retry_aware_steps(events))

    return log10_path


def rank_average(normal_scores: Dict[str, np.ndarray], scores: Dict[str, np.ndarray]) -> np.ndarray:
    """Ensemble score: for each detector, -log10 of the share of normal validation
    orders scoring at least as high; summed over the detectors."""
    total = np.zeros(len(next(iter(scores.values()))))
    for name, values in scores.items():
        reference = np.sort(normal_scores[name])
        at_least = len(reference) - np.searchsorted(reference, values, side="left")
        total += -np.log10((at_least + 1) / (len(reference) + 1))
    return total


def fit_stack(levels: np.ndarray, labels: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Logistic regression over the check levels (one column per check); returns a scorer."""
    if len(set(labels.tolist())) != 2:
        raise TrainError("normal and anomalous orders to fit the stacked model", sorted(set(labels.tolist())),
                         "fit_stack")
    scaler = StandardScaler().fit(levels)
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(levels), labels)
    return lambda rows: model.decision_function(scaler.transform(rows))
