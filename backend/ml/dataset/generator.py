"""Synthetic order event log (GENERATE checkpoint).

One row = one order (one case of an event log):
  event_sequence    events in order, e.g. CONFIRM_PAYMENT;PACK_ITEMS;...
  event_timestamps  one UTC timestamp per event
  order_amount      order value

Every rule that produces data is a named parameter below, with the assumption
behind it. The label of every row is set by the same function that built it.

Three kinds of order:

NORMAL
  A random walk over the workflow automaton using the branch probabilities
  below, log-normal waiting times between events, and a log-normal amount.

STATISTICAL_ANOMALY: legal and complete, but suspicious. Five types, each built
so that no single measurement gives it away (checked by
backend/tests/test_dataset_generator.py with backend/ml/dataset/audit.py):
  CARD_TESTING              2-6 failed payments retried a bit faster than people do, small amount
  REPEATED_PAYMENT_FAILURE  2-6 failed payments at normal human pace and a normal amount
  FAST_FULFILMENT           packing, shipping and delivery all about 2 standard deviations faster than normal
  STALLED_DELIVERY          delivery takes about 9 days instead of about 3
  HIGH_VALUE_QUICK_REFUND   a high amount refunded about 30 minutes after payment

RULE_VIOLATION
  A fulfilled order broken by one named edit that the workflow rules forbid
  (VIOLATION_TYPES). Each edit is invalid by construction; the FSM check
  afterwards only confirms the label, and a mismatch raises GenerateError.

Splits are assigned here, before any model sees the data:
  train       normal orders fit the unsupervised models; the anomaly rows are
              used only by the supervised Random Forest comparison
  validation  chooses between candidate models and sets decision thresholds
  test        used once, for the final report

The whole pipeline (generate, train, evaluate) runs with:
    .venv\\Scripts\\python.exe -m backend.ml.pipeline
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from backend.core.errors import GenerateError
from backend.fsm.states import ALL_EVENTS, INITIAL_STATE, TERMINAL_STATES
from backend.fsm.transitions import TRANSITION_TABLE
from backend.fsm.validator import validate_trace

RANDOM_SEED = 42

MINUTE = 60.0
HOUR = 60 * MINUTE
DAY = 24 * HOUR

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_CSV = DATA_DIR / "orders_event_log.csv"

CSV_COLUMNS = [
    "trace_id",
    "split",
    "anomaly_category",
    "anomaly_type",
    "event_sequence",
    "event_timestamps",
    "order_amount",
    "is_rule_valid",
    "is_complete",
]

# ---------------------------------------------------------------------------
# Size of each split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSize:
    normal: int
    per_anomaly_type: int
    rule_violations: int


SPLITS = ("train", "validation", "test")

# Assumption: about 5% of completed orders are anomalous (5 types x 50 = 250 per
# 5,000 normal test orders). Rare enough to look like a real review queue, and
# still 50 test examples per type to measure each type's detection rate.
SPLIT_SIZES: Dict[str, SplitSize] = {
    "train": SplitSize(normal=4_000, per_anomaly_type=100, rule_violations=0),
    "validation": SplitSize(normal=2_000, per_anomaly_type=20, rule_violations=0),
    "test": SplitSize(normal=5_000, per_anomaly_type=50, rule_violations=500),
}

# Orders start at a random moment in this window.
START_WINDOW_BEGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)
START_WINDOW_DAYS = 300

# ---------------------------------------------------------------------------
# Normal behaviour
# ---------------------------------------------------------------------------

Branch = List[Tuple[str, float]]

# Assumption: 1 in 10 first card payments is declined; 1 in 20 customers leaves before paying.
FIRST_PAYMENT: Branch = [("CONFIRM_PAYMENT", 0.85), ("FAIL_PAYMENT", 0.10), ("CANCEL_ORDER", 0.05)]
# After a decline most people try again; a few are approved on the bank's re-check; some give up.
AFTER_DECLINE: Branch = [("RETRY_PAYMENT", 0.75), ("CONFIRM_PAYMENT", 0.08), ("CANCEL_ORDER", 0.17)]
# A card declined once is often declined again, so 2-3 failures also happen in ordinary orders.
RETRY_ATTEMPT: Branch = [("CONFIRM_PAYMENT", 0.57), ("FAIL_PAYMENT", 0.35), ("CANCEL_ORDER", 0.08)]
# Most paid orders are fulfilled; a few customers cancel and are refunded before shipping.
AFTER_PAYMENT: Branch = [("PACK_ITEMS", 0.95), ("REFUND_ORDER", 0.05)]
AFTER_PACKING: Branch = [("SHIP_ITEMS", 0.97), ("REFUND_ORDER", 0.03)]
AFTER_SHIPPING: Branch = [("DELIVER_ITEMS", 1.0)]

# Waiting time before an event, given the previous event: (median seconds, log-normal sigma).
# A log-normal spread means times are always positive and occasionally much longer than usual.
NORMAL_GAPS: Dict[Tuple[str, str], Tuple[float, float]] = {
    # Payment, done by a person: re-entering card details takes about a minute.
    ("FAIL_PAYMENT", "RETRY_PAYMENT"): (75.0, 0.8),
    ("FAIL_PAYMENT", "CONFIRM_PAYMENT"): (3 * MINUTE, 0.8),   # bank approves on a re-check
    ("FAIL_PAYMENT", "CANCEL_ORDER"): (10 * MINUTE, 1.0),
    ("RETRY_PAYMENT", "CONFIRM_PAYMENT"): (45.0, 0.7),
    ("RETRY_PAYMENT", "FAIL_PAYMENT"): (45.0, 0.7),
    ("RETRY_PAYMENT", "CANCEL_ORDER"): (5 * MINUTE, 1.0),
    # Fulfilment: packed within a working day, shipped the same day, delivered in about 3 days.
    ("CONFIRM_PAYMENT", "PACK_ITEMS"): (8 * HOUR, 0.7),
    ("CONFIRM_PAYMENT", "REFUND_ORDER"): (3 * HOUR, 1.2),     # change of mind: minutes to a day
    ("PACK_ITEMS", "SHIP_ITEMS"): (5 * HOUR, 0.7),
    ("PACK_ITEMS", "REFUND_ORDER"): (10 * HOUR, 0.9),
    ("SHIP_ITEMS", "DELIVER_ITEMS"): (3 * DAY, 0.4),
}

# Median order value and log-normal spread: 90% of orders between about 320 and 4,500.
NORMAL_AMOUNT: Tuple[float, float] = (1_200.0, 0.8)

FULFILMENT = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]

# ---------------------------------------------------------------------------
# Anomalies: each is legal and complete, and overlaps normal orders on every
# single measurement; only the combination is unusual.
# ---------------------------------------------------------------------------

ANOMALY_TYPES = [
    "CARD_TESTING",
    "REPEATED_PAYMENT_FAILURE",
    "FAST_FULFILMENT",
    "STALLED_DELIVERY",
    "HIGH_VALUE_QUICK_REFUND",
]

# Card testing: a script tries stolen card numbers on a cheap order. It paces
# itself close to human speed to avoid simple speed limits.
CARD_TESTING_FAILURES = (2, 6)  # failed payments, inclusive
CARD_TESTING_GAPS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("FAIL_PAYMENT", "RETRY_PAYMENT"): (45.0, 0.5),    # people: 75 s
    ("RETRY_PAYMENT", "FAIL_PAYMENT"): (35.0, 0.5),    # people: 45 s
    ("RETRY_PAYMENT", "CONFIRM_PAYMENT"): (35.0, 0.5),
    ("FAIL_PAYMENT", "CANCEL_ORDER"): (45.0, 0.5),     # people: 10 min
}
CARD_TESTING_CANCEL_SHARE = 0.5  # half give up; half find a working card and get the order fulfilled
CARD_TESTING_AMOUNT: Tuple[float, float] = (250.0, 0.6)

# Repeated failure: a person (or an account takeover) keeps failing at normal pace.
REPEATED_FAILURE_FAILURES = (2, 6)
REPEATED_FAILURE_CANCEL_SHARE = 0.4

# Fast fulfilment: steps marked done faster than a real warehouse and courier work,
# e.g. a seller faking fulfilment to release payment early. Each median is about
# 2 normal standard deviations faster than the normal median (8 h -> 2 h,
# 5 h -> 1.2 h, 3 days -> 1.35 days), so each step alone still looks possible.
FAST_FULFILMENT_GAPS: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("CONFIRM_PAYMENT", "PACK_ITEMS"): (2 * HOUR, 0.5),
    ("PACK_ITEMS", "SHIP_ITEMS"): (1.2 * HOUR, 0.5),
    ("SHIP_ITEMS", "DELIVER_ITEMS"): (1.35 * DAY, 0.3),
}

# Stalled delivery: the parcel sits with the courier; overlaps the slow end of normal deliveries.
STALLED_DELIVERY_GAP: Tuple[float, float] = (9 * DAY, 0.35)

# High-value quick refund: a large order paid and refunded soon after (refund abuse).
HIGH_VALUE_REFUND_AMOUNT: Tuple[float, float] = (4_500.0, 0.35)   # top ~5% of normal amounts
HIGH_VALUE_REFUND_GAP: Tuple[float, float] = (30 * MINUTE, 0.8)   # normal refunds: about 3 h
HIGH_VALUE_REFUND_AFTER_PACKING_SHARE = 0.3

# ---------------------------------------------------------------------------
# Rule violations
# ---------------------------------------------------------------------------

VIOLATION_TYPES = [
    "MISSING_PAYMENT",   # packed and shipped without a payment
    "SKIPPED_STEP",      # packing or shipping left out
    "WRONG_ORDER",       # two adjacent steps swapped
    "DUPLICATE_EVENT",   # payment, packing or shipping logged twice
    "EVENT_AFTER_END",   # something happens after delivery
    "UNKNOWN_EVENT",     # an event outside the workflow's alphabet
]
UNKNOWN_EVENTS = [
    "HACK_CHECKOUT", "BYPASS_GATEWAY", "INJECT_DISCOUNT", "SKIP_VERIFICATION",
    "ADMIN_FORCE_SHIP", "MANUAL_CREDIT", "SPOOF_DELIVERY",
]
INSERTED_EVENT_GAP = (1.0, 120.0)        # seconds (uniform): an extra event logged shortly after another
EVENT_AFTER_END_GAP = (1 * HOUR, 2 * DAY)  # seconds (uniform)

_KNOWN_EVENTS = sorted(ALL_EVENTS)


@dataclass
class GeneratedOrder:
    events: List[str]
    gaps: List[float]  # seconds before each event; gaps[0] is not used
    amount: float
    category: str
    anomaly_type: str


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _lognormal(rng: random.Random, median: float, sigma: float) -> float:
    return rng.lognormvariate(math.log(median), sigma)


def _choose(rng: random.Random, options: Branch) -> str:
    r = rng.random()
    cumulative = 0.0
    for event, p in options:
        cumulative += p
        if r < cumulative:
            return event
    return options[-1][0]  # r is within floating-point rounding of 1.0


def _gap(rng: random.Random, previous: str, event: str,
         table: Mapping[Tuple[str, str], Tuple[float, float]]) -> float:
    if (previous, event) in table:
        return _lognormal(rng, *table[(previous, event)])
    return _lognormal(rng, *NORMAL_GAPS[(previous, event)])


def _gaps(rng: random.Random, events: Sequence[str],
          table: Mapping[Tuple[str, str], Tuple[float, float]] = NORMAL_GAPS) -> List[float]:
    return [0.0] + [_gap(rng, a, b, table) for a, b in zip(events, events[1:])]


def check_label(events: Sequence[str], category: str) -> Dict[str, Any]:
    """Confirm that the FSM agrees with the label a generator gave an order.

    Raises:
        GenerateError: A RULE_VIOLATION the FSM accepts, or any other label on an
            order the FSM rejects or that is not complete.
    """
    result = validate_trace(list(events))
    if category == "RULE_VIOLATION":
        if result["is_valid"]:
            raise GenerateError("an order the FSM rejects (label RULE_VIOLATION)", "an FSM-valid order", list(events))
    elif not (result["is_valid"] and result["is_complete"]):
        raise GenerateError(
            f"a valid, complete order (label {category})",
            f"is_valid={result['is_valid']}, is_complete={result['is_complete']}",
            list(events),
        )
    return result


# ---------------------------------------------------------------------------
# Normal orders
# ---------------------------------------------------------------------------

def _normal_events(rng: random.Random) -> List[str]:
    branches = {"PAYMENT_FAILED": AFTER_DECLINE, "PAID": AFTER_PAYMENT,
                "PACKED": AFTER_PACKING, "SHIPPED": AFTER_SHIPPING}
    events: List[str] = []
    state = INITIAL_STATE.value
    failures = 0
    while state not in TERMINAL_STATES:
        if state == INITIAL_STATE.value:
            options = FIRST_PAYMENT if failures == 0 else RETRY_ATTEMPT
        else:
            options = branches[state]
        event = _choose(rng, options)
        events.append(event)
        failures += event == "FAIL_PAYMENT"
        state = TRANSITION_TABLE[(state, event)]
    return events


def generate_normal_order(rng: random.Random) -> GeneratedOrder:
    events = _normal_events(rng)
    return GeneratedOrder(events, _gaps(rng, events), _lognormal(rng, *NORMAL_AMOUNT), "NORMAL", "NONE")


# ---------------------------------------------------------------------------
# Statistical anomalies
# ---------------------------------------------------------------------------

def _failure_loop(rng: random.Random, failures: Tuple[int, int], cancel_share: float) -> List[str]:
    """k x (FAIL, RETRY), then either give up after the last failure or pay and get the order fulfilled."""
    events = ["FAIL_PAYMENT", "RETRY_PAYMENT"] * rng.randint(*failures)
    if rng.random() < cancel_share:
        events[-1] = "CANCEL_ORDER"
    else:
        events += FULFILMENT
    return events


def _card_testing(rng: random.Random) -> GeneratedOrder:
    events = _failure_loop(rng, CARD_TESTING_FAILURES, CARD_TESTING_CANCEL_SHARE)
    return GeneratedOrder(events, _gaps(rng, events, CARD_TESTING_GAPS),
                          _lognormal(rng, *CARD_TESTING_AMOUNT), "STATISTICAL_ANOMALY", "CARD_TESTING")


def _repeated_payment_failure(rng: random.Random) -> GeneratedOrder:
    events = _failure_loop(rng, REPEATED_FAILURE_FAILURES, REPEATED_FAILURE_CANCEL_SHARE)
    return GeneratedOrder(events, _gaps(rng, events), _lognormal(rng, *NORMAL_AMOUNT),
                          "STATISTICAL_ANOMALY", "REPEATED_PAYMENT_FAILURE")


def _fast_fulfilment(rng: random.Random) -> GeneratedOrder:
    events = list(FULFILMENT)
    return GeneratedOrder(events, _gaps(rng, events, FAST_FULFILMENT_GAPS), _lognormal(rng, *NORMAL_AMOUNT),
                          "STATISTICAL_ANOMALY", "FAST_FULFILMENT")


def _stalled_delivery(rng: random.Random) -> GeneratedOrder:
    events = list(FULFILMENT)
    table = {("SHIP_ITEMS", "DELIVER_ITEMS"): STALLED_DELIVERY_GAP}
    return GeneratedOrder(events, _gaps(rng, events, table), _lognormal(rng, *NORMAL_AMOUNT),
                          "STATISTICAL_ANOMALY", "STALLED_DELIVERY")


def _high_value_quick_refund(rng: random.Random) -> GeneratedOrder:
    if rng.random() < HIGH_VALUE_REFUND_AFTER_PACKING_SHARE:
        events = ["CONFIRM_PAYMENT", "PACK_ITEMS", "REFUND_ORDER"]
    else:
        events = ["CONFIRM_PAYMENT", "REFUND_ORDER"]
    table = {(events[-2], "REFUND_ORDER"): HIGH_VALUE_REFUND_GAP}
    return GeneratedOrder(events, _gaps(rng, events, table), _lognormal(rng, *HIGH_VALUE_REFUND_AMOUNT),
                          "STATISTICAL_ANOMALY", "HIGH_VALUE_QUICK_REFUND")


ANOMALY_GENERATORS: Dict[str, Callable[[random.Random], GeneratedOrder]] = {
    "CARD_TESTING": _card_testing,
    "REPEATED_PAYMENT_FAILURE": _repeated_payment_failure,
    "FAST_FULFILMENT": _fast_fulfilment,
    "STALLED_DELIVERY": _stalled_delivery,
    "HIGH_VALUE_QUICK_REFUND": _high_value_quick_refund,
}

# ---------------------------------------------------------------------------
# Rule violations: one forbidden edit of a fulfilled order
# ---------------------------------------------------------------------------

def _inserted_gap(rng: random.Random) -> float:
    return rng.uniform(*INSERTED_EVENT_GAP)


def _remove(events: List[str], gaps: List[float], i: int) -> None:
    """Drop event i; the time before it is added to the next event so timestamps keep increasing."""
    if i + 1 < len(events):
        gaps[i + 1] += gaps[i]
    del events[i], gaps[i]
    gaps[0] = 0.0


def generate_rule_violation(rng: random.Random) -> GeneratedOrder:
    events = list(FULFILMENT)
    gaps = _gaps(rng, events)
    violation = rng.choice(VIOLATION_TYPES)
    if violation == "MISSING_PAYMENT":
        _remove(events, gaps, 0)                       # PACK_ITEMS straight from ORDER_CREATED
    elif violation == "SKIPPED_STEP":
        _remove(events, gaps, rng.choice([1, 2]))      # no packing, or no shipping
    elif violation == "WRONG_ORDER":
        i = rng.choice([0, 1, 2])                      # swap events i and i+1
        events[i], events[i + 1] = events[i + 1], events[i]
    elif violation == "DUPLICATE_EVENT":
        i = rng.choice([0, 1, 2])                      # CONFIRM, PACK or SHIP logged twice
        events.insert(i + 1, events[i])
        gaps.insert(i + 1, _inserted_gap(rng))
    elif violation == "EVENT_AFTER_END":
        events.append(rng.choice(_KNOWN_EVENTS))
        gaps.append(rng.uniform(*EVENT_AFTER_END_GAP))
    else:  # UNKNOWN_EVENT, before the last event so the order has not ended yet
        i = rng.randrange(len(events))
        events.insert(i, rng.choice(UNKNOWN_EVENTS))
        gaps.insert(i, _inserted_gap(rng))
        gaps[0] = 0.0
    return GeneratedOrder(events, gaps, _lognormal(rng, *NORMAL_AMOUNT), "RULE_VIOLATION", violation)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _timestamps(rng: random.Random, gaps: Sequence[float]) -> List[datetime]:
    """Whole-second UTC timestamps; every gap is at least 1 second so times strictly increase."""
    t = START_WINDOW_BEGIN + timedelta(seconds=round(rng.uniform(0, START_WINDOW_DAYS * DAY)))
    stamps = [t]
    for gap in gaps[1:]:
        t = t + timedelta(seconds=max(1, round(gap)))
        stamps.append(t)
    return stamps


def format_timestamp(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_sizes(sizes: Mapping[str, SplitSize]) -> None:
    if set(sizes) != set(SPLITS):
        raise GenerateError(f"split sizes for exactly {list(SPLITS)}", sorted(sizes), "sizes")
    for split, size in sizes.items():
        for field in ("normal", "per_anomaly_type", "rule_violations"):
            value = getattr(size, field)
            if not isinstance(value, int) or value < 0:
                raise GenerateError("a whole number >= 0", value, f"sizes[{split!r}].{field}")


def _check_branches() -> None:
    for name, branch in [("FIRST_PAYMENT", FIRST_PAYMENT), ("AFTER_DECLINE", AFTER_DECLINE),
                         ("RETRY_ATTEMPT", RETRY_ATTEMPT), ("AFTER_PAYMENT", AFTER_PAYMENT),
                         ("AFTER_PACKING", AFTER_PACKING), ("AFTER_SHIPPING", AFTER_SHIPPING)]:
        total = sum(p for _, p in branch)
        if abs(total - 1.0) > 1e-9:
            raise GenerateError("branch probabilities that sum to 1", total, name)


def generate_orders(seed: int = RANDOM_SEED,
                    sizes: Mapping[str, SplitSize] = SPLIT_SIZES) -> List[Tuple[str, GeneratedOrder, List[datetime]]]:
    """All orders as (split, order, timestamps), in a fixed order for a given seed."""
    _check_sizes(sizes)
    _check_branches()
    rng = random.Random(seed)
    plan: List[Tuple[str, GeneratedOrder]] = []
    for split in SPLITS:
        size = sizes[split]
        plan += [(split, generate_normal_order(rng)) for _ in range(size.normal)]
        for anomaly_type in ANOMALY_TYPES:
            make = ANOMALY_GENERATORS[anomaly_type]
            plan += [(split, make(rng)) for _ in range(size.per_anomaly_type)]
        plan += [(split, generate_rule_violation(rng)) for _ in range(size.rule_violations)]
    return [(split, order, _timestamps(rng, order.gaps)) for split, order in plan]


def generate_dataset(output_path: Optional[Union[str, Path]] = None, seed: int = RANDOM_SEED,
                     sizes: Mapping[str, SplitSize] = SPLIT_SIZES) -> Path:
    """Generate every order, confirm each label with the FSM, and write the CSV."""
    target = Path(output_path) if output_path else OUTPUT_CSV
    rows = []
    for idx, (split, order, stamps) in enumerate(generate_orders(seed, sizes), start=1):
        check = check_label(order.events, order.category)
        rows.append({
            "trace_id": f"ORD_{idx:05d}",
            "split": split,
            "anomaly_category": order.category,
            "anomaly_type": order.anomaly_type,
            "event_sequence": ";".join(order.events),
            "event_timestamps": ";".join(format_timestamp(t) for t in stamps),
            "order_amount": f"{order.amount:.2f}",
            "is_rule_valid": check["is_valid"],
            "is_complete": check["is_complete"],
        })
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return target
