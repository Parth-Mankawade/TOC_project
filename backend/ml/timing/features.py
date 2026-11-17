"""Turn one order into the timing/amount model's inputs (FEATURES checkpoint).

Training (normal training orders only) learns, for the order amount and for
every legal kind of step (e.g. PACK_ITEMS -> SHIP_ITEMS), the mean and standard
deviation of ln(value). A value's z-score is how many standard deviations its
logarithm lies from the training mean; its percentile is the share of normal
orders expected below it.

Each order becomes MODEL_INPUTS, numbers that exist for every order:
  amount_z          how unusual the amount is
  slowest_step_z    z-score of the step that was slowest compared with its usual time
  fastest_step_z    z-score of the step that was fastest compared with its usual time (negative = faster)
  mean_step_z       average z-score over all steps: is the whole order unusually quick or slow?
  payment_failures  number of FAIL_PAYMENT events
  refunded          1 if the order ended in a refund, else 0
  cancelled         1 if the order ended in a cancellation, else 0

An order with a single event has no steps; its step inputs are NO_STEPS_Z
(0: no timing deviation) by definition.

Two further scores need no fitted detector, only the z-scores (timing_summary):
  trend      |sum of the step z-scores| / sqrt(number of steps). Under normal
             behaviour the step z-scores are independent with mean 0 and variance
             1, so this is about 0.7 on average whatever the order's length, and
             it grows when every step is a little fast (or a little slow) even
             though no single step stands out.
  deviation  -log10 of the chi-square tail probability of (sum of squared step
             z-scores + squared amount z-score), with one degree of freedom per
             value. It grows when one value, or several, lie far from normal in
             either direction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import math
from statistics import NormalDist
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from scipy.stats import chi2

from backend.core.errors import FeatureError, TrainError
from backend.fsm.reachability import legal_step_kinds
from backend.fsm.validator import validate_trace

MODEL_INPUTS = ["amount_z", "slowest_step_z", "fastest_step_z", "mean_step_z",
                "payment_failures", "refunded", "cancelled"]
SMALLEST_TAIL_PROBABILITY = 1e-300   # keeps -log10 finite for values the model calls impossible
NO_STEPS_Z = 0.0
NORMAL_BAND = (2.5, 97.5)       # percentiles: a value outside the middle 95% of normal orders is "unusual"
MIN_TRAINING_ORDERS = 100
MIN_SAMPLES_PER_STEP = 5        # every legal kind of step must appear at least this often in training

_STANDARD_NORMAL = NormalDist()

TimedOrder = Tuple[Sequence[str], Sequence[datetime], float]  # (events, timestamps, amount)


class StepDuration(NamedTuple):
    step: int            # 1-based index of the later event
    from_event: str
    to_event: str
    seconds: float


def step_name(from_event: str, to_event: str) -> str:
    return f"{from_event} -> {to_event}"


def format_duration(seconds: float) -> str:
    """Readable duration: '45 s', '12.5 min', '8.0 h', '3.1 days'."""
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def step_durations(events: Sequence[str], timestamps: Sequence[datetime]) -> List[StepDuration]:
    """Durations between consecutive events.

    Raises:
        FeatureError: If the counts differ or the timestamps do not strictly increase.
    """
    if len(events) != len(timestamps):
        raise FeatureError(f"{len(events)} timestamps (one per event)", f"{len(timestamps)} timestamps",
                           "event_timestamps")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise FeatureError("strictly increasing timestamps", [t.isoformat() for t in timestamps],
                           "event_timestamps")
    return [
        StepDuration(i + 1, events[i - 1], events[i], (timestamps[i] - timestamps[i - 1]).total_seconds())
        for i in range(1, len(events))
    ]


@dataclass(frozen=True)
class LogNormalStats:
    """Mean and standard deviation of ln(value) over normal training orders."""

    mean_ln: float
    std_ln: float
    count: int

    @classmethod
    def fit(cls, values: Sequence[float]) -> "LogNormalStats":
        if len(values) < 2 or any(not v > 0 for v in values):
            raise TrainError("at least 2 positive values", list(values)[:10], "LogNormalStats.fit")
        logs = [math.log(v) for v in values]
        mean = sum(logs) / len(logs)
        std = math.sqrt(sum((x - mean) ** 2 for x in logs) / len(logs))
        if std == 0.0:
            raise TrainError("values that are not all equal", list(values)[:10], "LogNormalStats.fit")
        return cls(mean, std, len(values))

    def z(self, value: float) -> float:
        return (math.log(value) - self.mean_ln) / self.std_ln

    def percentile(self, value: float) -> float:
        return 100.0 * _STANDARD_NORMAL.cdf(self.z(value))

    def quantile(self, percentile: float) -> float:
        return math.exp(self.mean_ln + self.std_ln * _STANDARD_NORMAL.inv_cdf(percentile / 100.0))


@dataclass
class FeatureStats:
    amount: LogNormalStats
    steps: Dict[str, LogNormalStats]      # key: step_name(from, to)
    failure_counts: Dict[int, int]        # payment failures per normal training order -> how many orders
    training_sample_count: int

    def step(self, s: StepDuration) -> LogNormalStats:
        name = step_name(s.from_event, s.to_event)
        if name not in self.steps:
            raise FeatureError(f"a step kind the model learned ({len(self.steps)} kinds)", name, f"step {s.step}")
        return self.steps[name]

    def failures_percentile(self, failures: int) -> float:
        below = sum(n for k, n in self.failure_counts.items() if k < failures)
        equal = sum(n for k, n in self.failure_counts.items() if k == failures)
        return 100.0 * (below + 0.5 * equal) / self.training_sample_count

    def failures_band(self) -> Tuple[int, int]:
        inside = [k for k in self.failure_counts if NORMAL_BAND[0] <= self.failures_percentile(k) <= NORMAL_BAND[1]]
        return min(inside), max(inside)


def _check_order(events: Sequence[str], timestamps: Sequence[datetime], amount: float) -> List[StepDuration]:
    if not (isinstance(amount, (int, float)) and math.isfinite(amount) and amount > 0):
        raise FeatureError("a positive order amount", amount, "order_amount")
    result = validate_trace(list(events))
    if not (result["is_valid"] and result["is_complete"]):
        raise FeatureError("an FSM-valid, complete order", list(events), "event_sequence")
    return step_durations(events, timestamps)


def order_features(stats: FeatureStats, events: Sequence[str], timestamps: Sequence[datetime],
                   amount: float) -> Dict[str, float]:
    """The model inputs of one valid, complete order, in MODEL_INPUTS order.

    Raises:
        FeatureError: Bad amount, invalid or unfinished order, bad timestamps, or a step kind
            the statistics do not cover.
    """
    steps = _check_order(events, timestamps, amount)
    zs = [stats.step(s).z(s.seconds) for s in steps]
    return {
        "amount_z": stats.amount.z(amount),
        "slowest_step_z": max(zs) if zs else NO_STEPS_Z,
        "fastest_step_z": min(zs) if zs else NO_STEPS_Z,
        "mean_step_z": sum(zs) / len(zs) if zs else NO_STEPS_Z,
        "payment_failures": float(events.count("FAIL_PAYMENT")),
        "refunded": float(events[-1] == "REFUND_ORDER"),
        "cancelled": float(events[-1] == "CANCEL_ORDER"),
    }


def timing_summary(stats: FeatureStats, events: Sequence[str], timestamps: Sequence[datetime],
                   amount: float) -> Dict[str, float]:
    """The trend and deviation scores of one valid, complete order (see the module docstring).

    Returns:
        trend, deviation, and what they are made of: step_count, trend_z (signed:
        negative = faster than normal overall) and sum_squared_z.

    Raises:
        FeatureError: As order_features().
    """
    steps = _check_order(events, timestamps, amount)
    zs = [stats.step(s).z(s.seconds) for s in steps]
    trend_z = sum(zs) / math.sqrt(len(zs)) if zs else NO_STEPS_Z
    sum_squared_z = sum(z * z for z in zs) + stats.amount.z(amount) ** 2
    tail = max(float(chi2.sf(sum_squared_z, len(zs) + 1)), SMALLEST_TAIL_PROBABILITY)
    return {
        "trend": abs(trend_z),
        "deviation": -math.log10(tail),
        "step_count": float(len(zs)),
        "trend_z": trend_z,
        "sum_squared_z": sum_squared_z,
    }


def _row(name: str, unit: str, value: float, low: float, high: float, percentile: float,
         step: Optional[int] = None) -> Dict[str, Any]:
    return {
        "name": name,
        "step": step,
        "unit": unit,
        "value": float(value),
        "normal_low": float(low),
        "normal_high": float(high),
        "percentile": float(percentile),
        "is_unusual": not (NORMAL_BAND[0] <= percentile <= NORMAL_BAND[1]),
    }


def measurements(stats: FeatureStats, events: Sequence[str], timestamps: Sequence[datetime],
                 amount: float) -> List[Dict[str, Any]]:
    """The amount, the number of failed payments and every step duration, each with the
    normal range (middle 95% of normal orders) and its percentile among normal orders."""
    steps = _check_order(events, timestamps, amount)
    low, high = NORMAL_BAND
    failures = events.count("FAIL_PAYMENT")
    f_low, f_high = stats.failures_band()
    rows = [
        _row("order_amount", "currency", amount, stats.amount.quantile(low), stats.amount.quantile(high),
             stats.amount.percentile(amount)),
        _row("payment_failures", "count", failures, f_low, f_high, stats.failures_percentile(failures)),
    ]
    for s in steps:
        st = stats.step(s)
        rows.append(_row(step_name(s.from_event, s.to_event), "seconds", s.seconds,
                         st.quantile(low), st.quantile(high), st.percentile(s.seconds), step=s.step))
    return rows


def fit_feature_stats(orders: Sequence[TimedOrder]) -> FeatureStats:
    """Learn amount, step-duration and failure-count statistics from normal training orders.

    Raises:
        TrainError: Too few orders, an invalid or unfinished order, a non-positive amount,
            or a legal kind of step seen fewer than MIN_SAMPLES_PER_STEP times.
    """
    if len(orders) < MIN_TRAINING_ORDERS:
        raise TrainError(f"at least {MIN_TRAINING_ORDERS} normal training orders", len(orders), "fit_feature_stats")
    durations: Dict[str, List[float]] = {step_name(a, b): [] for a, b in legal_step_kinds()}
    amounts: List[float] = []
    failures: Counter = Counter()
    for i, (events, timestamps, amount) in enumerate(orders):
        try:
            steps = _check_order(events, timestamps, amount)
        except FeatureError as exc:
            raise TrainError("valid, complete normal orders with positive amounts", list(events),
                             f"training order {i}: {exc}") from exc
        for s in steps:
            durations[step_name(s.from_event, s.to_event)].append(s.seconds)
        amounts.append(float(amount))
        failures[events.count("FAIL_PAYMENT")] += 1
    thin = {name: len(v) for name, v in durations.items() if len(v) < MIN_SAMPLES_PER_STEP}
    if thin:
        raise TrainError(f"at least {MIN_SAMPLES_PER_STEP} examples of every legal step kind",
                         ", ".join(f"{name}: {n}" for name, n in thin.items()), "training orders")
    return FeatureStats(
        amount=LogNormalStats.fit(amounts),
        steps={name: LogNormalStats.fit(v) for name, v in durations.items()},
        failure_counts=dict(sorted(failures.items())),
        training_sample_count=len(orders),
    )
