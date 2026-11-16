"""EVALUATE checkpoint: measure every model once on the held-out test split.

Positive class: STATISTICAL_ANOMALY (legal, complete but suspicious orders); the
test split's normal orders are the negatives. A rule violation is rejected by
the FSM by definition (that is what a rule violation is), so violations are
reported separately and not as a detection rate.

Headline metric: F1, one number that is high only when most anomalies are caught
AND most alarms are real. Precision, recall, false-alarm rate, ROC-AUC (for
models that give a score) and the number caught per anomaly type are reported too.

Models (fitted on the train split; thresholds and the timing model's choice come
from the validation split; the test split is used only here):
  always_normal     trivial baseline: never flags anything (F1 = 0)
  range_rule        non-ML baseline: flag an order if its amount, its number of
                    failed payments or any step duration is outside the min-max
                    seen in normal training orders
  markov_chain      event-order model
  timing_<name>     each timing-and-amount candidate
  hybrid            markov_chain OR the chosen timing candidate: what the API uses
  random_forest     supervised comparison, trained on normal training orders plus
                    the training rows of SEEN_TYPES only, so the other anomaly
                    types are new to it at test time
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from backend.core.errors import EvaluateError
from backend.fsm.validator import validate_trace
from backend.ml.dataset.audit import (
    MAX_SINGLE_MEASUREMENT_RECALL,
    SEPARABILITY_FALSE_ALARM_BUDGET,
    separability_report,
)
from backend.ml.dataset.generator import ANOMALY_TYPES, SPLITS
from backend.ml.dataset.loader import OrderRecord, select
from backend.ml.markov.model import MarkovPathModel, score_trace
from backend.ml.timing.features import order_features, step_durations, step_name
from backend.ml.timing.model import DETECTOR_NAMES, TimingModel, anomaly_scores

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

BASELINES = ["always_normal", "range_rule"]
MODEL_ORDER = (BASELINES + ["markov_chain"] + [f"timing_{name}" for name in DETECTOR_NAMES]
               + ["hybrid", "random_forest"])
SEEN_TYPES = ["CARD_TESTING", "FAST_FULFILMENT", "HIGH_VALUE_QUICK_REFUND"]
RF_TREES = 200
RF_RANDOM_STATE = 42
RF_DECISION_PROBABILITY = 0.5
DECIMALS = 4  # rounding of reported metrics


def _timed(orders: Sequence[OrderRecord]) -> List[Tuple[List[str], List[Any], float]]:
    return [(o.events, o.timestamps, o.amount) for o in orders]


# ---------------------------------------------------------------------------
# Non-ML baseline
# ---------------------------------------------------------------------------

def _range_observations(order: OrderRecord) -> List[Tuple[str, float]]:
    observations = [("order_amount", order.amount), ("payment_failures", float(order.events.count("FAIL_PAYMENT")))]
    for s in step_durations(order.events, order.timestamps):
        observations.append((step_name(s.from_event, s.to_event), s.seconds))
    return observations


def fit_range_rule(train_normals: Sequence[OrderRecord]) -> Dict[str, Tuple[float, float]]:
    """Min and max of every measurement over normal training orders."""
    ranges: Dict[str, Tuple[float, float]] = {}
    for order in train_normals:
        for name, value in _range_observations(order):
            low, high = ranges[name] if name in ranges else (value, value)
            ranges[name] = (min(low, value), max(high, value))
    return ranges


def range_rule_flags(ranges: Dict[str, Tuple[float, float]], order: OrderRecord) -> bool:
    for name, value in _range_observations(order):
        if name not in ranges:
            raise EvaluateError("a measurement seen in normal training orders", name, order.trace_id)
        low, high = ranges[name]
        if not low <= value <= high:
            return True
    return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(name: str, test: Sequence[OrderRecord], flags: Sequence[bool],
             scores: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    y_true = [int(o.is_anomaly) for o in test]
    y_pred = [int(f) for f in flags]
    normal_flags = [f for o, f in zip(test, flags) if not o.is_anomaly]
    caught_by_type = {
        t: {"caught": sum(1 for o, f in zip(test, flags) if o.anomaly_type == t and f),
            "total": sum(1 for o in test if o.anomaly_type == t)}
        for t in ANOMALY_TYPES
    }
    return {
        "model": name,
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), DECIMALS),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), DECIMALS),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), DECIMALS),
        "false_alarms": int(sum(normal_flags)),
        "normal_orders": len(normal_flags),
        "false_alarm_rate": round(sum(normal_flags) / len(normal_flags), DECIMALS),
        "anomalies_caught": sum(c["caught"] for c in caught_by_type.values()),
        "anomalous_orders": sum(c["total"] for c in caught_by_type.values()),
        "roc_auc": None if scores is None else round(float(roc_auc_score(y_true, scores)), DECIMALS),
        "caught_by_type": caught_by_type,
    }


def _rf_features(timing: TimingModel, markov: MarkovPathModel, order: OrderRecord) -> List[float]:
    inputs = order_features(timing.stats, order.events, order.timestamps, order.amount)
    return list(inputs.values()) + [score_trace(markov, order.events)["path_log10_probability"],
                                    float(len(order.events))]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def evaluate(orders: Sequence[OrderRecord], markov: MarkovPathModel, timing_models: Dict[str, TimingModel],
             selection: Dict[str, Any]) -> Dict[str, Any]:
    """Score every model on the test split and build the report.

    Raises:
        EvaluateError: If the test split lacks normal or anomalous orders, or the
            chosen timing model is not among the fitted candidates.
    """
    test = [o for o in orders if o.split == "test" and o.category in ("NORMAL", "STATISTICAL_ANOMALY")]
    n_anomalous = sum(o.is_anomaly for o in test)
    n_normal = len(test) - n_anomalous
    if n_anomalous == 0 or n_normal == 0:
        raise EvaluateError("normal and anomalous orders in the test split",
                            f"{n_normal} normal, {n_anomalous} anomalous", "test split")
    chosen = selection["chosen"]
    if chosen not in timing_models or set(timing_models) != set(DETECTOR_NAMES):
        raise EvaluateError(f"fitted candidates {DETECTOR_NAMES} including the chosen one", sorted(timing_models),
                            f"chosen={chosen}")

    train_normals = select(orders, "NORMAL", "train")
    timed_test = _timed(test)

    ranges = fit_range_rule(train_normals)
    path = [score_trace(markov, o.events) for o in test]
    timing_scores = {name: [float(s) for s in anomaly_scores(model, timed_test)]
                     for name, model in timing_models.items()}

    models = [
        _metrics("always_normal", test, [False] * len(test)),
        _metrics("range_rule", test, [range_rule_flags(ranges, o) for o in test]),
        _metrics("markov_chain", test, [p["is_ml_anomaly"] for p in path], [p["anomaly_score"] for p in path]),
    ]
    for name in DETECTOR_NAMES:
        scores = timing_scores[name]
        models.append(_metrics(f"timing_{name}", test, [s > 0 for s in scores], scores))
    models.append(_metrics("hybrid", test, [p["is_ml_anomaly"] or s > 0 for p, s in zip(path, timing_scores[chosen])]))

    chosen_model = timing_models[chosen]
    rf_train = train_normals + [o for o in select(orders, "STATISTICAL_ANOMALY", "train")
                                if o.anomaly_type in SEEN_TYPES]
    forest = RandomForestClassifier(n_estimators=RF_TREES, random_state=RF_RANDOM_STATE, class_weight="balanced")
    forest.fit([_rf_features(chosen_model, markov, o) for o in rf_train], [int(o.is_anomaly) for o in rf_train])
    rf_probability = forest.predict_proba([_rf_features(chosen_model, markov, o) for o in test])[:, 1]
    rf = _metrics("random_forest", test, [p >= RF_DECISION_PROBABILITY for p in rf_probability],
                  [float(p) for p in rf_probability])
    rf["trained_on_types"] = list(SEEN_TYPES)
    rf["never_seen_types"] = [t for t in ANOMALY_TYPES if t not in SEEN_TYPES]
    models.append(rf)

    violations = select(orders, "RULE_VIOLATION", "test")
    fsm_errors = [validate_trace(o.events)["error"] for o in violations]
    by_name = {m["model"]: m for m in models}
    return {
        "dataset": {
            "orders_by_split": {split: dict(sorted(Counter(o.category for o in orders if o.split == split).items()))
                                for split in SPLITS},
            "anomaly_types": list(ANOMALY_TYPES),
            "test_anomaly_share": round(n_anomalous / len(test), DECIMALS),
        },
        "separability": {
            "false_alarm_budget": SEPARABILITY_FALSE_ALARM_BUDGET,
            "trivial_if_recall_at_least": MAX_SINGLE_MEASUREMENT_RECALL,
            "by_type": {t: dict(r, recall=round(r["recall"], DECIMALS))
                        for t, r in separability_report(orders).items()},
        },
        "model_selection": selection,
        "headline": {
            "metric": "f1",
            "model": "hybrid",
            "hybrid": by_name["hybrid"]["f1"],
            "always_normal": by_name["always_normal"]["f1"],
            "range_rule": by_name["range_rule"]["f1"],
        },
        "test": {"models": models},
        "rule_violations": {
            "orders": len(violations),
            "rejected_by_fsm": sum(1 for e in fsm_errors if e is not None),
            "by_type": dict(sorted(Counter(o.anomaly_type for o in violations).items())),
            "by_fsm_error": dict(sorted(Counter(e["error_type"] for e in fsm_errors if e is not None).items())),
            "note": "Rejected by definition: this checks the generated violations, it is not a detection rate.",
        },
    }
