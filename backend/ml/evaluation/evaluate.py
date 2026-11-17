"""EVALUATE checkpoint: measure every model once on the held-out test split.

Positive class: STATISTICAL_ANOMALY (legal, complete but suspicious orders); the
test split's normal orders are the negatives. A rule violation is rejected by
the FSM by definition (that is what a rule violation is), so violations are
reported separately and not as a detection rate.

Headline metric: F1, one number that is high only when most anomalies are caught
AND most alarms are real. Precision, recall, false-alarm rate, ROC-AUC and PR-AUC
(for models that give a score) and the number caught per anomaly type are
reported too. PR-AUC (average precision) is the more telling of the two here:
with about 5% anomalies, ROC-AUC stays high even when most alarms are false.

Everything is fitted on the train split; thresholds, the timing model's choice
and the hybrid's threshold come from the validation split; the test split is
used only here. Models:

  always_normal      trivial baseline: never flags anything (F1 = 0)
  range_rule         untuned non-ML baseline: flag an order if its amount, its
                     number of failed payments or any step duration is outside the
                     min-max seen in normal training orders
  tuned_range_rule   the same idea made fair: every measurement gets a band of
                     equal tail probability among normal training orders, and the
                     band width is set on normal validation orders so that the
                     rule flags the same share of them as the hybrid does
  zscore_or_markov   no fitted detector: the sum of squared z-scores (flagging
                     its top 1% of normal validation orders) OR the Markov chain
                     (flagging its rarest 1%)
  markov_chain       the path check alone, flagging its rarest 1%
  timing_<name>      each cluster-check candidate alone, flagging its top 1%
  trend_check        the trend check alone, flagging its top 1%
  deviation_check    the deviation check alone, flagging its top 1%
  hybrid             what the API uses: the four checks under one threshold
                     (backend/ml/hybrid.py)
  random_forest      supervised comparison, trained on normal training orders plus
                     the training rows of SEEN_TYPES only, so the other anomaly
                     types are new to it at test time
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from backend.core.errors import EvaluateError
from backend.fsm.validator import validate_trace
from backend.ml import hybrid
from backend.ml.dataset.audit import (
    MAX_SINGLE_MEASUREMENT_RECALL,
    SEPARABILITY_FALSE_ALARM_BUDGET,
    separability_report,
)
from backend.ml.dataset.generator import ANOMALY_TYPES, SPLITS
from backend.ml.dataset.loader import OrderRecord, select
from backend.ml.markov.model import MarkovPathModel, score_trace
from backend.ml.timing.features import FeatureStats, order_features, step_durations, step_name, timing_summary
from backend.ml.timing.model import DETECTOR_NAMES, SCALE_PERCENTILE, TimingModel, anomaly_scores, detector_scores

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

BASELINES = ["always_normal", "range_rule", "tuned_range_rule", "zscore_or_markov"]
MODEL_ORDER = (BASELINES + ["markov_chain"] + [f"timing_{name}" for name in DETECTOR_NAMES]
               + ["trend_check", "deviation_check", "hybrid", "random_forest"])
SEEN_TYPES = ["CARD_TESTING", "FAST_FULFILMENT", "HIGH_VALUE_QUICK_REFUND"]
RF_TREES = 200
RF_RANDOM_STATE = 42
RF_DECISION_PROBABILITY = 0.5
DECIMALS = 4  # rounding of reported metrics


def timed(orders: Sequence[OrderRecord]) -> List[Tuple[List[str], List[Any], float]]:
    return [(o.events, o.timestamps, o.amount) for o in orders]


# ---------------------------------------------------------------------------
# Non-ML baselines
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


def fit_tuned_range_rule(train_normals: Sequence[OrderRecord]) -> Callable[[OrderRecord], float]:
    """A score for the tuned range rule: how far outside the normal training values
    an order's most extreme measurement lies.

    For one measurement, the tail share is the share of normal training values at
    least as extreme as the order's value, on its nearer side. The score is
    -log10 of the smallest tail share over the order's measurements. Flagging
    score > t is the same as flagging any measurement outside a band that leaves
    a 10**-t share of normal training values on each side.

    Raises:
        EvaluateError (from the returned function): A measurement no normal training order has.
    """
    pools: Dict[str, List[float]] = {}
    for order in train_normals:
        for name, value in _range_observations(order):
            pools.setdefault(name, []).append(value)
    sorted_pools = {name: np.sort(values) for name, values in pools.items()}

    def score(order: OrderRecord) -> float:
        smallest = 1.0
        for name, value in _range_observations(order):
            if name not in sorted_pools:
                raise EvaluateError("a measurement seen in normal training orders", name, order.trace_id)
            pool = sorted_pools[name]
            at_or_below = int(np.searchsorted(pool, value, side="right"))
            at_or_above = len(pool) - int(np.searchsorted(pool, value, side="left"))
            smallest = min(smallest, (min(at_or_below, at_or_above) + 1) / (len(pool) + 1))
        return -float(np.log10(smallest))

    return score


def sum_squared_z(stats: FeatureStats, orders: Sequence[OrderRecord]) -> np.ndarray:
    return np.asarray([timing_summary(stats, o.events, o.timestamps, o.amount)["sum_squared_z"] for o in orders])


def cutoff(normal_scores: Sequence[float], flagged_share: float) -> float:
    """The score that `flagged_share` of the given normal orders exceed."""
    return float(np.percentile(normal_scores, 100.0 * (1.0 - flagged_share)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(name: str, rows: Sequence[OrderRecord], flags: Sequence[bool],
            scores: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Precision, recall, F1, false alarms and per-type counts of one model on the given
    normal and anomalous orders; ROC-AUC and PR-AUC too when it gives a score."""
    y_true = [int(o.is_anomaly) for o in rows]
    y_pred = [int(f) for f in flags]
    normal_flags = [bool(f) for o, f in zip(rows, flags) if not o.is_anomaly]
    caught_by_type = {
        t: {"caught": sum(1 for o, f in zip(rows, flags) if o.anomaly_type == t and f),
            "total": sum(1 for o in rows if o.anomaly_type == t)}
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
        "pr_auc": None if scores is None else round(float(average_precision_score(y_true, scores)), DECIMALS),
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

    `markov` and the chosen timing model must be the calibrated pair the API uses
    (hybrid.calibrate); the other timing candidates are scored on their cluster check only.

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
    timing = timing_models[chosen]

    train_normals = select(orders, "NORMAL", "train")
    validation_normals = select(orders, "NORMAL", "validation")
    timed_test, timed_validation = timed(test), timed(validation_normals)
    own_share = 1.0 - SCALE_PERCENTILE / 100.0     # a single check flags its top 1% of normal validation orders
    budget = float(np.mean(hybrid.hybrid_scores(markov, timing, timed_validation) > 0))

    ranges = fit_range_rule(train_normals)
    tuned_score = fit_tuned_range_rule(train_normals)
    tuned_test = [tuned_score(o) for o in test]
    tuned_cut = cutoff([tuned_score(o) for o in validation_normals], budget)
    z_test = sum_squared_z(timing.stats, test)
    z_cut = cutoff(sum_squared_z(timing.stats, validation_normals), own_share)

    levels = hybrid.levels(markov, timing, timed_test)
    path = [score_trace(markov, o.events) for o in test]
    hybrid_score = hybrid.hybrid_scores(markov, timing, timed_test)
    timing_score = anomaly_scores(timing, timed_test)

    models = [
        metrics("always_normal", test, [False] * len(test)),
        metrics("range_rule", test, [range_rule_flags(ranges, o) for o in test]),
        metrics("tuned_range_rule", test, [s > tuned_cut for s in tuned_test], tuned_test),
        metrics("zscore_or_markov", test, [z > z_cut or level > 1 for z, level in zip(z_test, levels["path"])]),
        metrics("markov_chain", test, levels["path"] > 1, levels["path"]),
    ]
    for name in DETECTOR_NAMES:
        scores = detector_scores(timing_models[name], timed_test)
        models.append(metrics(f"timing_{name}", test, scores > 0, scores))
    models.append(metrics("trend_check", test, levels["trend"] > 1, levels["trend"]))
    models.append(metrics("deviation_check", test, levels["deviation"] > 1, levels["deviation"]))
    # The same two decisions the analyzer makes: the Markov model's and the timing model's.
    models.append(metrics("hybrid", test, [p["is_ml_anomaly"] or s > 0 for p, s in zip(path, timing_score)],
                          hybrid_score))

    rf_train = train_normals + [o for o in select(orders, "STATISTICAL_ANOMALY", "train")
                                if o.anomaly_type in SEEN_TYPES]
    forest = RandomForestClassifier(n_estimators=RF_TREES, random_state=RF_RANDOM_STATE, class_weight="balanced")
    forest.fit([_rf_features(timing, markov, o) for o in rf_train], [int(o.is_anomaly) for o in rf_train])
    rf_probability = forest.predict_proba([_rf_features(timing, markov, o) for o in test])[:, 1]
    rf = metrics("random_forest", test, [p >= RF_DECISION_PROBABILITY for p in rf_probability],
                 [float(p) for p in rf_probability])
    rf["trained_on_types"] = list(SEEN_TYPES)
    rf["never_seen_types"] = [t for t in ANOMALY_TYPES if t not in SEEN_TYPES]
    models.append(rf)

    validation_levels = hybrid.levels(markov, timing, timed_validation)
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
        "hybrid": {
            "checks": list(hybrid.CHECKS),
            "false_alarm_budget": hybrid.FALSE_ALARM_BUDGET,
            "threshold_level": timing.threshold,
            "validation_normal_orders": len(validation_normals),
            "validation_flag_rate": round(budget, DECIMALS),
            "validation_flag_rate_by_check": {
                check: round(float(np.mean(validation_levels[check] > timing.threshold)), DECIMALS)
                for check in hybrid.CHECKS
            },
        },
        "headline": {
            "metric": "f1",
            "model": "hybrid",
            "hybrid": by_name["hybrid"]["f1"],
            "always_normal": by_name["always_normal"]["f1"],
            "range_rule": by_name["range_rule"]["f1"],
            "tuned_range_rule": by_name["tuned_range_rule"]["f1"],
            "zscore_or_markov": by_name["zscore_or_markov"]["f1"],
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
