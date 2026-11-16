"""LOAD checkpoint: reading and checking the event log CSV (backend/ml/dataset/loader.py)."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
import tempfile
import unittest

from backend.core.errors import DataLoadError
from backend.ml.dataset import generator as gen
from backend.ml.dataset.loader import (
    DEMO_IN_PROGRESS,
    DEMO_NORMAL,
    DEMO_PER_ANOMALY_TYPE,
    DEMO_RULE_VIOLATIONS,
    demo_sample,
    example_orders,
    load_orders,
    select,
)

SMALL = {
    "train": gen.SplitSize(normal=60, per_anomaly_type=3, rule_violations=0),
    "validation": gen.SplitSize(normal=20, per_anomaly_type=2, rule_violations=0),
    "test": gen.SplitSize(normal=70, per_anomaly_type=4, rule_violations=24),
}


class LoaderTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = gen.generate_dataset(Path(self._tmp.name) / "orders.csv", sizes=SMALL)
        with open(self.path, newline="", encoding="utf-8") as fh:
            self.rows = list(csv.DictReader(fh))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def rewrite(self, rows: list[dict], columns: list[str] | None = None) -> Path:
        path = Path(self._tmp.name) / "edited.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns or gen.CSV_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def assert_load_error(self, path: Path, *fragments: str) -> None:
        with self.assertRaises(DataLoadError) as ctx:
            load_orders(path)
        message = str(ctx.exception)
        self.assertTrue(message.startswith("[LOAD] expected "), message)
        for fragment in fragments:
            self.assertIn(fragment, message)


class TestLoadHappyPath(LoaderTestCase):

    def test_every_row_becomes_a_typed_record(self) -> None:
        orders = load_orders(self.path)
        self.assertEqual(len(orders), len(self.rows))
        first = orders[0]
        self.assertEqual(first.trace_id, "ORD_00001")
        self.assertEqual(len(first.events), len(first.timestamps))
        self.assertTrue(all(t.tzinfo is not None for t in first.timestamps))
        self.assertIsInstance(first.amount, float)
        self.assertEqual({o.split for o in orders}, set(gen.SPLITS))

    def test_select_filters_by_category_and_split(self) -> None:
        orders = load_orders(self.path)
        normal_test = select(orders, "NORMAL", "test")
        self.assertEqual(len(normal_test), SMALL["test"].normal)
        self.assertTrue(all(o.category == "NORMAL" and o.split == "test" for o in normal_test))
        self.assertEqual(len(select(orders, split="validation")),
                         SMALL["validation"].normal + 5 * SMALL["validation"].per_anomaly_type)


class TestLoadErrors(LoaderTestCase):

    def test_missing_file(self) -> None:
        self.assert_load_error(Path(self._tmp.name) / "nope.csv", "an existing CSV file", "nope.csv")

    def test_wrong_columns(self) -> None:
        columns = [c for c in gen.CSV_COLUMNS if c != "event_timestamps"]
        rows = [{c: r[c] for c in columns} for r in self.rows]
        self.assert_load_error(self.rewrite(rows, columns), "columns", "event_timestamps")

    def test_file_with_no_orders(self) -> None:
        self.assert_load_error(self.rewrite([]), "at least one order", "0")

    def test_unknown_split(self) -> None:
        self.rows[3]["split"] = "holdout"
        self.assert_load_error(self.rewrite(self.rows), "holdout", "ORD_00004")

    def test_unknown_category(self) -> None:
        self.rows[0]["anomaly_category"] = "WEIRD"
        self.assert_load_error(self.rewrite(self.rows), "WEIRD", "ORD_00001")

    def test_anomaly_type_that_does_not_fit_the_category(self) -> None:
        self.rows[0]["anomaly_type"] = "CARD_TESTING"  # row 1 is a NORMAL order
        self.assert_load_error(self.rewrite(self.rows), "CARD_TESTING", "ORD_00001")

    def test_event_and_timestamp_counts_differ(self) -> None:
        self.rows[0]["event_timestamps"] = self.rows[0]["event_timestamps"].split(";")[0] + ";2026-02-01T00:00:00Z;2026-02-02T00:00:00Z;2026-02-03T00:00:00Z;2026-02-04T00:00:00Z;2026-02-05T00:00:00Z;2026-02-06T00:00:00Z;2026-02-07T00:00:00Z;2026-02-08T00:00:00Z;2026-02-09T00:00:00Z;2026-02-10T00:00:00Z;2026-02-11T00:00:00Z;2026-02-12T00:00:00Z;2026-02-13T00:00:00Z;2026-02-14T00:00:00Z"
        self.assert_load_error(self.rewrite(self.rows), "timestamps", "ORD_00001")

    def test_unreadable_timestamp(self) -> None:
        stamps = self.rows[0]["event_timestamps"].split(";")
        stamps[0] = "yesterday"
        self.rows[0]["event_timestamps"] = ";".join(stamps)
        self.assert_load_error(self.rewrite(self.rows), "yesterday", "ORD_00001")

    def test_timestamps_that_go_backwards(self) -> None:
        row = next(r for r in self.rows if r["event_sequence"].count(";") >= 2)
        stamps = row["event_timestamps"].split(";")
        stamps[0], stamps[1] = stamps[1], stamps[0]
        row["event_timestamps"] = ";".join(stamps)
        self.assert_load_error(self.rewrite(self.rows), "strictly increasing", row["trace_id"])

    def test_bad_amounts(self) -> None:
        for bad in ["0", "-5", "abc", "nan", "inf", ""]:
            with self.subTest(amount=bad):
                rows = [dict(r) for r in self.rows]
                rows[2]["order_amount"] = bad
                self.assert_load_error(self.rewrite(rows), "positive", "ORD_00003")

    def test_rule_valid_column_that_disagrees_with_the_fsm(self) -> None:
        self.rows[0]["is_rule_valid"] = "False"
        self.assert_load_error(self.rewrite(self.rows), "is_rule_valid", "ORD_00001")

    def test_duplicate_trace_id(self) -> None:
        self.rows[5]["trace_id"] = self.rows[4]["trace_id"]
        self.assert_load_error(self.rewrite(self.rows), "unique", self.rows[4]["trace_id"])


class TestDemoAndExamples(LoaderTestCase):

    def test_demo_sample_has_the_documented_mix_in_a_fixed_order(self) -> None:
        orders = load_orders(self.path)
        sample = demo_sample(orders)
        counts = Counter(o.category for o in sample)
        self.assertEqual(counts["NORMAL"], DEMO_NORMAL)
        self.assertEqual(counts["IN_PROGRESS"], DEMO_IN_PROGRESS)
        self.assertEqual(counts["RULE_VIOLATION"], DEMO_RULE_VIOLATIONS)
        self.assertEqual(counts["STATISTICAL_ANOMALY"], DEMO_PER_ANOMALY_TYPE * len(gen.ANOMALY_TYPES))
        self.assertTrue(all(o.split == "test" for o in sample))
        self.assertEqual([o.trace_id for o in sample], [o.trace_id for o in demo_sample(orders)])
        in_progress = [o for o in sample if o.category == "IN_PROGRESS"]
        self.assertTrue(all(len(o.events) == 2 for o in in_progress))

    def test_demo_sample_fails_loudly_when_the_test_split_is_too_small(self) -> None:
        orders = [o for o in load_orders(self.path) if o.category != "RULE_VIOLATION"]
        with self.assertRaises(DataLoadError) as ctx:
            demo_sample(orders)
        self.assertIn("RULE_VIOLATION", str(ctx.exception))

    def test_example_orders_cover_every_kind_of_order(self) -> None:
        examples = example_orders(load_orders(self.path))
        kinds = {e.anomaly_type if e.category != "NORMAL" else "NORMAL" for e in examples}
        self.assertTrue(set(gen.ANOMALY_TYPES) <= kinds)
        self.assertIn("NORMAL", kinds)
        self.assertIn("IN_PROGRESS", {e.category for e in examples})
        self.assertTrue(any(e.category == "RULE_VIOLATION" for e in examples))
        self.assertTrue(all(e.split == "test" for e in examples))


if __name__ == "__main__":
    unittest.main()
