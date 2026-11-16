"""GENERATE checkpoint: the synthetic event log (backend/ml/dataset/generator.py).

Checks that every row follows the rule that produced it, that the data is
reproducible, and that no anomaly type can be picked out by one measurement.
The data is generated fresh into a temporary folder, so these tests need no
saved files.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from statistics import median
import tempfile
import unittest

from backend.core.errors import GenerateError
from backend.fsm.transitions import TRANSITION_TABLE
from backend.fsm.validator import validate_trace
from backend.ml.dataset import generator as gen
from backend.ml.dataset.audit import (
    MAX_SINGLE_MEASUREMENT_RECALL,
    SEPARABILITY_FALSE_ALARM_BUDGET,
    separability_report,
)
from backend.ml.dataset.loader import OrderRecord, load_orders

HAPPY = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]
SMALL = {
    "train": gen.SplitSize(normal=60, per_anomaly_type=4, rule_violations=0),
    "validation": gen.SplitSize(normal=20, per_anomaly_type=2, rule_violations=0),
    "test": gen.SplitSize(normal=30, per_anomaly_type=3, rule_violations=12),
}


def _gaps(orders: list[OrderRecord], a: str, b: str) -> list[float]:
    """Every duration (seconds) of the step a -> b across the given orders."""
    out = []
    for o in orders:
        for i in range(1, len(o.events)):
            if (o.events[i - 1], o.events[i]) == (a, b):
                out.append((o.timestamps[i] - o.timestamps[i - 1]).total_seconds())
    return out


def _close(value: float, target: float, tolerance: float) -> bool:
    return abs(value - target) <= tolerance * target


class TestReproducibility(unittest.TestCase):

    def _lines(self, seed: int) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            path = gen.generate_dataset(Path(tmp) / "orders.csv", seed=seed, sizes=SMALL)
            return path.read_text(encoding="utf-8").splitlines()

    def test_same_seed_gives_identical_file(self) -> None:
        self.assertEqual(self._lines(7), self._lines(7))

    def test_different_seed_gives_different_file(self) -> None:
        self.assertNotEqual(self._lines(7), self._lines(8))

    def test_header_is_the_documented_column_list(self) -> None:
        self.assertEqual(self._lines(7)[0].split(","), gen.CSV_COLUMNS)


class TestDefaultDataset(unittest.TestCase):
    """The full default dataset, generated once for all checks below."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = gen.generate_dataset(Path(cls._tmp.name) / "orders.csv")
        cls.orders = load_orders(cls.path)
        cls.normal = [o for o in cls.orders if o.category == "NORMAL"]
        cls.by_type = {t: [o for o in cls.orders if o.anomaly_type == t] for t in gen.ANOMALY_TYPES}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # --- counts and labels -------------------------------------------------

    def test_row_counts_match_the_split_sizes(self) -> None:
        counts = Counter((o.category, o.split) for o in self.orders)
        for split, size in gen.SPLIT_SIZES.items():
            self.assertEqual(counts[("NORMAL", split)], size.normal, split)
            self.assertEqual(counts[("RULE_VIOLATION", split)], size.rule_violations, split)
            for anomaly_type in gen.ANOMALY_TYPES:
                n = sum(1 for o in self.by_type[anomaly_type] if o.split == split)
                self.assertEqual(n, size.per_anomaly_type, (split, anomaly_type))

    def test_anomalies_are_about_five_percent_of_validation_and_test(self) -> None:
        # Stated assumption in the generator: anomalies are rare (~5% of completed orders).
        for split in ("validation", "test"):
            rows = [o for o in self.orders if o.split == split and o.category != "RULE_VIOLATION"]
            share = sum(o.is_anomaly for o in rows) / len(rows)
            self.assertTrue(0.03 <= share <= 0.07, (split, share))

    def test_trace_ids_are_unique(self) -> None:
        self.assertEqual(len({o.trace_id for o in self.orders}), len(self.orders))

    def test_every_label_agrees_with_the_fsm(self) -> None:
        for o in self.orders:
            result = validate_trace(o.events)
            if o.category == "RULE_VIOLATION":
                self.assertFalse(result["is_valid"], o.trace_id)
            else:
                self.assertTrue(result["is_valid"] and result["is_complete"], o.trace_id)

    def test_amounts_are_positive_and_finite(self) -> None:
        for o in self.orders:
            self.assertTrue(math.isfinite(o.amount) and o.amount > 0, o.trace_id)

    # --- normal behaviour follows its stated rules --------------------------

    def test_normal_training_orders_use_every_transition_and_step_kind(self) -> None:
        train = [o for o in self.normal if o.split == "train"]
        used = Counter()
        for o in train:
            for step in validate_trace(o.events)["history"]:
                used[(step["from_state"], step["event"])] += 1
        self.assertEqual(set(used), set(TRANSITION_TABLE))
        for (a, b) in gen.NORMAL_GAPS:
            self.assertGreaterEqual(len(_gaps(train, a, b)), 5, (a, b))

    def test_normal_payment_failures_follow_the_branch_probabilities(self) -> None:
        failed = sum(o.events[0] == "FAIL_PAYMENT" for o in self.normal) / len(self.normal)
        self.assertTrue(_close(failed, dict(gen.FIRST_PAYMENT)["FAIL_PAYMENT"], 0.15), failed)
        # Ordinary customers sometimes fail 3 times, so 3 failures alone is not proof of fraud.
        self.assertGreater(sum(o.events.count("FAIL_PAYMENT") >= 3 for o in self.normal), 20)

    def test_normal_outcomes_and_times_follow_their_rules(self) -> None:
        delivered = sum(o.events[-1] == "DELIVER_ITEMS" for o in self.normal) / len(self.normal)
        self.assertTrue(0.80 <= delivered <= 0.90, delivered)
        for (a, b), (target, _sigma) in gen.NORMAL_GAPS.items():
            values = _gaps(self.normal, a, b)
            self.assertTrue(_close(median(values), target, 0.25), (a, b, median(values), target))
        self.assertTrue(_close(median(o.amount for o in self.normal), gen.NORMAL_AMOUNT[0], 0.05))

    # --- each anomaly type follows its rule ----------------------------------

    def test_card_testing(self) -> None:
        rows = self.by_type["CARD_TESTING"]
        low, high = gen.CARD_TESTING_FAILURES
        self.assertTrue(all(low <= o.events.count("FAIL_PAYMENT") <= high for o in rows))
        for (a, b), (target, _sigma) in gen.CARD_TESTING_GAPS.items():
            values = _gaps(rows, a, b)
            self.assertTrue(values and _close(median(values), target, 0.25), (a, b))
        self.assertTrue(_close(median(o.amount for o in rows), gen.CARD_TESTING_AMOUNT[0], 0.2))
        cancelled = sum(o.events[-1] == "CANCEL_ORDER" for o in rows) / len(rows)
        self.assertTrue(abs(cancelled - gen.CARD_TESTING_CANCEL_SHARE) <= 0.15, cancelled)

    def test_repeated_payment_failure_is_human_paced(self) -> None:
        rows = self.by_type["REPEATED_PAYMENT_FAILURE"]
        low, high = gen.REPEATED_FAILURE_FAILURES
        self.assertTrue(all(low <= o.events.count("FAIL_PAYMENT") <= high for o in rows))
        retry = median(_gaps(rows, "FAIL_PAYMENT", "RETRY_PAYMENT"))
        self.assertTrue(_close(retry, gen.NORMAL_GAPS[("FAIL_PAYMENT", "RETRY_PAYMENT")][0], 0.25), retry)
        self.assertTrue(_close(median(o.amount for o in rows), gen.NORMAL_AMOUNT[0], 0.2))

    def test_fast_fulfilment(self) -> None:
        rows = self.by_type["FAST_FULFILMENT"]
        self.assertTrue(all(o.events == HAPPY for o in rows))
        for (a, b), (target, _sigma) in gen.FAST_FULFILMENT_GAPS.items():
            fast = median(_gaps(rows, a, b))
            self.assertTrue(_close(fast, target, 0.2), (a, b, fast))
            self.assertLess(fast, gen.NORMAL_GAPS[(a, b)][0])

    def test_stalled_delivery(self) -> None:
        rows = self.by_type["STALLED_DELIVERY"]
        self.assertTrue(all(o.events == HAPPY for o in rows))
        stall = median(_gaps(rows, "SHIP_ITEMS", "DELIVER_ITEMS"))
        self.assertTrue(_close(stall, gen.STALLED_DELIVERY_GAP[0], 0.15), stall)

    def test_high_value_quick_refund(self) -> None:
        rows = self.by_type["HIGH_VALUE_QUICK_REFUND"]
        self.assertTrue(all(o.events[0] == "CONFIRM_PAYMENT" and o.events[-1] == "REFUND_ORDER" for o in rows))
        self.assertTrue(_close(median(o.amount for o in rows), gen.HIGH_VALUE_REFUND_AMOUNT[0], 0.1))
        refund_gaps = [(o.timestamps[-1] - o.timestamps[-2]).total_seconds() for o in rows]
        self.assertTrue(_close(median(refund_gaps), gen.HIGH_VALUE_REFUND_GAP[0], 0.25))

    def test_every_rule_violation_type_breaks_the_rule_it_names(self) -> None:
        expected_error = {"EVENT_AFTER_END": "TERMINAL_STATE_VIOLATION", "UNKNOWN_EVENT": "UNKNOWN_EVENT"}
        violations = [o for o in self.orders if o.category == "RULE_VIOLATION"]
        self.assertEqual({o.anomaly_type for o in violations}, set(gen.VIOLATION_TYPES))
        for o in violations:
            error = validate_trace(o.events)["error"]
            self.assertEqual(error["error_type"], expected_error.get(o.anomaly_type, "INVALID_TRANSITION"), o.trace_id)

    # --- no giveaway feature -------------------------------------------------

    def test_no_single_measurement_separates_any_anomaly_type(self) -> None:
        report = separability_report(self.orders, SEPARABILITY_FALSE_ALARM_BUDGET)
        self.assertEqual(set(report), set(gen.ANOMALY_TYPES))
        for anomaly_type, best in report.items():
            self.assertLess(best["recall"], MAX_SINGLE_MEASUREMENT_RECALL, (anomaly_type, best))


class TestGenerateErrors(unittest.TestCase):

    def test_negative_split_size_is_rejected(self) -> None:
        sizes = dict(SMALL, train=gen.SplitSize(normal=-1, per_anomaly_type=4, rule_violations=0))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(GenerateError) as ctx:
            gen.generate_dataset(Path(tmp) / "x.csv", sizes=sizes)
        self.assertIn("[GENERATE]", str(ctx.exception))
        self.assertIn("-1", str(ctx.exception))

    def test_unknown_split_name_is_rejected(self) -> None:
        sizes = dict(SMALL, holdout=gen.SplitSize(normal=1, per_anomaly_type=0, rule_violations=0))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(GenerateError) as ctx:
            gen.generate_dataset(Path(tmp) / "x.csv", sizes=sizes)
        self.assertIn("holdout", str(ctx.exception))

    def test_a_label_that_disagrees_with_the_fsm_is_rejected(self) -> None:
        with self.assertRaises(GenerateError) as ctx:
            gen.check_label(HAPPY, "RULE_VIOLATION")
        self.assertIn("RULE_VIOLATION", str(ctx.exception))
        with self.assertRaises(GenerateError):
            gen.check_label(["SHIP_ITEMS"], "NORMAL")
        gen.check_label(HAPPY, "NORMAL")  # agrees: no error


if __name__ == "__main__":
    unittest.main()
