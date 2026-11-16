"""Data audit: can a single measurement pick out an anomaly type?

If one raw measurement (amount, number of failed payments, one step's
duration, ...) catches nearly all orders of a type while flagging almost no
normal orders, then the type is trivially separable and a plain threshold
would do the ML's job. The generator's tests require that this never happens,
and the evaluation report shows the result.

For each anomaly type and each measurement, the threshold (on either side) is
chosen so that at most SEPARABILITY_FALSE_ALARM_BUDGET of all normal orders are
flagged; the best share of that type caught this way is reported.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from backend.core.errors import EvaluateError
from backend.fsm.reachability import legal_step_kinds
from backend.ml.dataset.loader import OrderRecord

SEPARABILITY_FALSE_ALARM_BUDGET = 0.01  # at most 1% of normal orders flagged
MAX_SINGLE_MEASUREMENT_RECALL = 0.90    # a type is "trivially separable" if one measurement catches >= 90%

STEP_MEASUREMENTS = [f"{a} -> {b}" for a, b in legal_step_kinds()]
MEASUREMENTS = (["order_amount", "payment_failures", "event_count", "refunded", "cancelled"]
                + STEP_MEASUREMENTS + ["shortest_step", "longest_step"])


def raw_measurements(order: OrderRecord) -> Dict[str, float]:
    """Plain facts about one valid order, one value per name in MEASUREMENTS.

    A step that does not occur in the order is NaN, and NaN never counts as flagged.

    Raises:
        EvaluateError: If the order contains a step no valid order can contain.
    """
    m: Dict[str, float] = {name: math.nan for name in STEP_MEASUREMENTS}
    m.update({
        "order_amount": order.amount,
        "payment_failures": float(order.events.count("FAIL_PAYMENT")),
        "event_count": float(len(order.events)),
        "refunded": float(order.events[-1] == "REFUND_ORDER"),
        "cancelled": float(order.events[-1] == "CANCEL_ORDER"),
    })
    durations: List[float] = []
    for i in range(1, len(order.events)):
        seconds = (order.timestamps[i] - order.timestamps[i - 1]).total_seconds()
        durations.append(seconds)
        name = f"{order.events[i - 1]} -> {order.events[i]}"
        if name not in m:
            raise EvaluateError("a step that valid orders can contain", name, order.trace_id)
        m[name] = seconds if math.isnan(m[name]) else min(m[name], seconds)
    m["shortest_step"] = min(durations) if durations else math.nan
    m["longest_step"] = max(durations) if durations else math.nan
    return m


def _best_recall(normal: Sequence[float], anomalous: Sequence[float], budget: float) -> tuple[float, str]:
    """Best share of `anomalous` above (or below) a threshold that flags at most `budget` of `normal`."""
    allowed = int(budget * len(normal))
    best = (0.0, "high")
    for sign, side in ((1.0, "high"), (-1.0, "low")):
        ranked = sorted((sign * v for v in normal if not math.isnan(v)), reverse=True)
        threshold = ranked[allowed] if allowed < len(ranked) else -math.inf
        caught = sum(1 for v in anomalous if not math.isnan(v) and sign * v > threshold)
        recall = caught / len(anomalous)
        if recall > best[0]:
            best = (recall, side)
    return best


def separability_report(orders: Sequence[OrderRecord],
                        budget: float = SEPARABILITY_FALSE_ALARM_BUDGET) -> Dict[str, Dict[str, Any]]:
    """For each anomaly type: the single measurement that catches most of it within the budget.

    Raises:
        EvaluateError: If there are no normal orders or no anomalies to compare.
    """
    normal = [raw_measurements(o) for o in orders if o.category == "NORMAL"]
    types = sorted({o.anomaly_type for o in orders if o.category == "STATISTICAL_ANOMALY"})
    if not normal or not types:
        raise EvaluateError("normal orders and anomalies to compare",
                            f"{len(normal)} normal, {len(types)} anomaly types", "separability_report")
    report: Dict[str, Dict[str, Any]] = {}
    for anomaly_type in types:
        rows = [raw_measurements(o) for o in orders if o.anomaly_type == anomaly_type]
        best: Dict[str, Any] = {"measurement": MEASUREMENTS[0], "side": "high", "recall": -1.0, "anomalies": len(rows)}
        for name in MEASUREMENTS:
            recall, side = _best_recall([r[name] for r in normal], [r[name] for r in rows], budget)
            if recall > best["recall"]:
                best.update(measurement=name, side=side, recall=recall)
        report[anomaly_type] = best
    return report
