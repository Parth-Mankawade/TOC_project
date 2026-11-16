"""Read the order event log CSV into typed records (LOAD checkpoint).

Every row is checked; the first problem stops loading with a DataLoadError that
names the row. Nothing is skipped or repaired.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Union

from backend.core.errors import DataLoadError
from backend.fsm.parsing import parse_timestamp
from backend.fsm.validator import validate_trace
from backend.ml.dataset.generator import ANOMALY_TYPES, CSV_COLUMNS, OUTPUT_CSV, SPLITS, VIOLATION_TYPES

CATEGORY_TYPES: Dict[str, List[str]] = {
    "NORMAL": ["NONE"],
    "STATISTICAL_ANOMALY": ANOMALY_TYPES,
    "RULE_VIOLATION": VIOLATION_TYPES,
}

# Demo stream for the dashboard: test-split orders only, in a fixed shuffled order.
DEMO_NORMAL = 55
DEMO_IN_PROGRESS = 5          # normal orders cut off after their first DEMO_IN_PROGRESS_EVENTS events
DEMO_IN_PROGRESS_EVENTS = 2
DEMO_PER_ANOMALY_TYPE = 4
DEMO_RULE_VIOLATIONS = 20
DEMO_SHUFFLE_SEED = 7


@dataclass(frozen=True)
class OrderRecord:
    trace_id: str
    split: str
    category: str
    anomaly_type: str
    events: List[str]
    timestamps: List[datetime]
    amount: float

    @property
    def is_anomaly(self) -> bool:
        return self.category == "STATISTICAL_ANOMALY"


def _bool(value: str, column: str, trace_id: str) -> bool:
    if value not in ("True", "False"):
        raise DataLoadError(f"'True' or 'False' in {column}", value, trace_id)
    return value == "True"


def _amount(value: str, trace_id: str) -> float:
    try:
        amount = float(value)
    except ValueError:
        raise DataLoadError("a positive number in order_amount", value, trace_id) from None
    if not (math.isfinite(amount) and amount > 0):
        raise DataLoadError("a positive number in order_amount", value, trace_id)
    return amount


def _record(row: Dict[str, str]) -> OrderRecord:
    trace_id = row["trace_id"]
    if row["split"] not in SPLITS:
        raise DataLoadError(f"split in {list(SPLITS)}", row["split"], trace_id)
    category = row["anomaly_category"]
    if category not in CATEGORY_TYPES:
        raise DataLoadError(f"anomaly_category in {list(CATEGORY_TYPES)}", category, trace_id)
    if row["anomaly_type"] not in CATEGORY_TYPES[category]:
        raise DataLoadError(f"an anomaly_type of {category}: {CATEGORY_TYPES[category]}", row["anomaly_type"], trace_id)

    events = row["event_sequence"].split(";")
    raw_stamps = row["event_timestamps"].split(";")
    if not all(events):
        raise DataLoadError("event names separated by ';'", row["event_sequence"], trace_id)
    if len(raw_stamps) != len(events):
        raise DataLoadError(f"as many timestamps as events ({len(events)})", f"{len(raw_stamps)} timestamps", trace_id)
    try:
        stamps = [parse_timestamp(t) for t in raw_stamps]
    except ValueError as exc:
        raise DataLoadError("ISO-8601 timestamps with a timezone", row["event_timestamps"], f"{trace_id} ({exc})") from exc
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise DataLoadError("strictly increasing timestamps", row["event_timestamps"], trace_id)

    fsm = validate_trace(events)
    for column, actual in (("is_rule_valid", fsm["is_valid"]), ("is_complete", fsm["is_complete"])):
        stored = _bool(row[column], column, trace_id)
        if stored != actual:
            raise DataLoadError(f"{column}={actual} (the FSM's result)", f"{column}={stored}", trace_id)

    return OrderRecord(trace_id, row["split"], category, row["anomaly_type"], events, stamps,
                       _amount(row["order_amount"], trace_id))


def load_orders(path: Optional[Union[str, Path]] = None) -> List[OrderRecord]:
    """Read and check every row of the event log.

    Raises:
        DataLoadError: Missing file, wrong columns, no rows, or any malformed row.
    """
    target = Path(path) if path else OUTPUT_CSV
    if not target.is_file():
        raise DataLoadError("an existing CSV file (run: python -m backend.ml.pipeline)", "no file", str(target))
    with open(target, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != CSV_COLUMNS:
            raise DataLoadError(f"columns {CSV_COLUMNS}", reader.fieldnames, str(target))
        orders = [_record(row) for row in reader]
    if not orders:
        raise DataLoadError("at least one order", 0, str(target))
    seen: set[str] = set()
    for o in orders:
        if o.trace_id in seen:
            raise DataLoadError("unique trace_id values", f"{o.trace_id} twice", str(target))
        seen.add(o.trace_id)
    return orders


def select(orders: Sequence[OrderRecord], category: Optional[str] = None,
           split: Optional[str] = None) -> List[OrderRecord]:
    """Orders matching a category and/or split (None means any)."""
    return [o for o in orders
            if (category is None or o.category == category) and (split is None or o.split == split)]


def _take(orders: Sequence[OrderRecord], n: int, what: str) -> List[OrderRecord]:
    if len(orders) < n:
        raise DataLoadError(f"at least {n} test-split orders of kind {what}", len(orders), "demo sample")
    return list(orders[:n])


def _in_progress(order: OrderRecord) -> OrderRecord:
    k = DEMO_IN_PROGRESS_EVENTS
    return OrderRecord(order.trace_id + "-LIVE", order.split, "IN_PROGRESS", "NONE",
                       order.events[:k], order.timestamps[:k], order.amount)


def demo_sample(orders: Sequence[OrderRecord]) -> List[OrderRecord]:
    """The dashboard's demo stream: test-split orders in a fixed shuffled order.

    DEMO_NORMAL normal orders, DEMO_IN_PROGRESS unfinished ones, DEMO_PER_ANOMALY_TYPE
    of each anomaly type and DEMO_RULE_VIOLATIONS rule violations.

    Raises:
        DataLoadError: If the test split has too few orders of any kind.
    """
    test = select(orders, split="test")
    normal = select(test, "NORMAL")
    picked = _take(normal, DEMO_NORMAL, "NORMAL")
    longer = [o for o in normal[DEMO_NORMAL:] if len(o.events) > DEMO_IN_PROGRESS_EVENTS + 1]
    picked += [_in_progress(o) for o in _take(longer, DEMO_IN_PROGRESS, "NORMAL (for in-progress orders)")]
    picked += _take(select(test, "RULE_VIOLATION"), DEMO_RULE_VIOLATIONS, "RULE_VIOLATION")
    for anomaly_type in ANOMALY_TYPES:
        picked += _take([o for o in test if o.anomaly_type == anomaly_type], DEMO_PER_ANOMALY_TYPE, anomaly_type)
    random.Random(DEMO_SHUFFLE_SEED).shuffle(picked)
    return picked


def example_orders(orders: Sequence[OrderRecord]) -> List[OrderRecord]:
    """One real test-split order of every kind, for the Analyze page's example picker.

    Normal: one delivered, one cancelled and one refunded order. Then the first
    order of every anomaly type and every rule-violation type, and one unfinished order.

    Raises:
        DataLoadError: If the test split lacks any of these kinds.
    """
    test = select(orders, split="test")
    normal = select(test, "NORMAL")
    examples: List[OrderRecord] = []
    for last_event in ("DELIVER_ITEMS", "CANCEL_ORDER", "REFUND_ORDER"):
        examples += _take([o for o in normal if o.events[-1] == last_event], 1, f"NORMAL ending {last_event}")
    for kind in ANOMALY_TYPES + VIOLATION_TYPES:
        examples += _take([o for o in test if o.anomaly_type == kind], 1, kind)
    longer = [o for o in normal if len(o.events) > DEMO_IN_PROGRESS_EVENTS + 1]
    examples += [_in_progress(o) for o in _take(longer, 1, "NORMAL (for an in-progress order)")]
    return examples
