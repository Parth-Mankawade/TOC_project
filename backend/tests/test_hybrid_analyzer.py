"""INPUT and ANALYZE checkpoints (backend/fsm/parsing.py, backend/pipeline/analyzer.py).

Every analysis uses real models trained on the fixture data (tests/fixtures.py).
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import unittest

from backend.core.errors import InputError
from backend.fsm.parsing import parse_event_sequence, parse_timestamp, parse_timestamps
from backend.ml.dataset.generator import format_timestamp
from backend.pipeline.analyzer import (
    TIMING_SKIPPED_NO_DATA,
    analyze_trace,
    describe_measurement,
    timing_reasons,
)
from backend.tests.fixtures import HAPPY, trained


def stamps_of(order) -> list[str]:
    return [format_timestamp(t) for t in order.timestamps]


class TestParsing(unittest.TestCase):

    def test_event_sequence_from_text_or_list(self) -> None:
        self.assertEqual(parse_event_sequence(" CONFIRM_PAYMENT ; PACK_ITEMS "), ["CONFIRM_PAYMENT", "PACK_ITEMS"])
        self.assertEqual(parse_event_sequence(["CONFIRM_PAYMENT", " PACK_ITEMS"]), ["CONFIRM_PAYMENT", "PACK_ITEMS"])

    def test_bad_event_sequences(self) -> None:
        for bad in ["", "   ", "CONFIRM_PAYMENT;;PACK_ITEMS", "CONFIRM_PAYMENT;", [], ["CONFIRM_PAYMENT", ""], [1]]:
            with self.subTest(bad=bad), self.assertRaises(InputError) as ctx:
                parse_event_sequence(bad)
            self.assertTrue(str(ctx.exception).startswith("[INPUT] expected "))
            self.assertIn("event_sequence", str(ctx.exception))

    def test_timestamps_are_returned_in_utc(self) -> None:
        self.assertEqual(parse_timestamp("2026-05-24T11:00:00+05:30"),
                         datetime(2026, 5, 24, 5, 30, tzinfo=timezone.utc))

    def test_bad_timestamps(self) -> None:
        for bad, fragment in [("yesterday", "yesterday"), ("2026-05-24T05:30:31", "timezone")]:
            with self.subTest(bad=bad), self.assertRaises(InputError) as ctx:
                parse_timestamp(bad)
            self.assertIn(fragment, str(ctx.exception))
        with self.assertRaises(InputError) as ctx:
            parse_timestamps(["2026-05-24T05:30:31Z", "2026-05-24T05:30:31Z"])
        self.assertIn("event_timestamps[1]", str(ctx.exception))


class TestVerdicts(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.t = trained()
        cls.test_orders = [o for o in cls.t.orders if o.split == "test"]

    def analyze(self, events, stamps=None, amount=None):
        return analyze_trace(events, stamps, amount, models=self.t)

    def test_rule_violation_names_the_step_and_the_allowed_events(self) -> None:
        r = self.analyze(["CONFIRM_PAYMENT", "SHIP_ITEMS"])
        self.assertEqual((r["verdict"], r["is_anomaly"], r["anomaly_type"]), ("RULE_VIOLATION", True, "RULE_VIOLATION"))
        error = r["fsm_evaluation"]["error"]
        self.assertEqual((error["error_type"], error["failed_step"], error["failed_event"], error["state_at_failure"]),
                         ("INVALID_TRANSITION", 2, "SHIP_ITEMS", "PAID"))
        self.assertEqual(error["allowed_events"], ["PACK_ITEMS", "REFUND_ORDER"])
        self.assertIn("step 2", r["summary"])
        self.assertIn("PACK_ITEMS, REFUND_ORDER", r["summary"])
        self.assertEqual(r["diagnostics"]["severity"], "CRITICAL")
        self.assertIsNone(r["ml_evaluation"])
        self.assertIsNone(r["timing_evaluation"])
        self.assertIn("RULE_VIOLATION", r["ml_skip_reason"])

    def test_unknown_event_and_event_after_the_end(self) -> None:
        self.assertEqual(self.analyze(["HACK_CHECKOUT"])["fsm_evaluation"]["error"]["error_type"], "UNKNOWN_EVENT")
        after_end = self.analyze(HAPPY + ["REFUND_ORDER"])
        self.assertEqual(after_end["fsm_evaluation"]["error"]["error_type"], "TERMINAL_STATE_VIOLATION")
        self.assertIn("DELIVERED", after_end["summary"])

    def test_unfinished_order_lists_the_endings_still_possible(self) -> None:
        r = self.analyze(["CONFIRM_PAYMENT", "PACK_ITEMS"])
        self.assertEqual((r["verdict"], r["is_anomaly"]), ("VALID_INCOMPLETE", False))
        self.assertEqual(r["diagnostics"]["severity"], "INFO")
        factors = r["diagnostics"]["key_factors"]
        self.assertIn("Possible outcome DELIVERED: next SHIP_ITEMS -> DELIVER_ITEMS", factors)
        self.assertIn("Possible outcome REFUNDED: next REFUND_ORDER", factors)

    def test_normal_order_with_and_without_timing_data(self) -> None:
        delivered = next(o for o in self.test_orders if o.category == "NORMAL" and o.events == HAPPY
                         and self.analyze(o.events, stamps_of(o), o.amount)["verdict"] == "NORMAL")
        r = self.analyze(delivered.events, stamps_of(delivered), delivered.amount)
        self.assertEqual(r["diagnostics"]["severity"], "INFO")
        self.assertEqual(r["timing_evaluation"]["model_name"], self.t.timing.detector_name)
        self.assertIsNone(r["timing_skip_reason"])
        without = self.analyze(delivered.events)
        self.assertEqual(without["verdict"], "NORMAL")
        self.assertIsNone(without["timing_evaluation"])
        self.assertEqual(without["timing_skip_reason"], TIMING_SKIPPED_NO_DATA)
        self.assertIn("not run", " ".join(without["diagnostics"]["key_factors"]))

    def test_many_failed_payments_is_a_statistical_anomaly_with_reasons(self) -> None:
        events = ["FAIL_PAYMENT", "RETRY_PAYMENT"] * 5 + HAPPY
        r = self.analyze(events)
        self.assertEqual((r["verdict"], r["anomaly_type"]), ("STATISTICAL_ANOMALY", "STATISTICAL_ANOMALY"))
        self.assertTrue(r["ml_evaluation"]["is_ml_anomaly"])
        self.assertEqual(r["diagnostics"]["severity"], "WARNING")
        factors = " ".join(r["diagnostics"]["key_factors"])
        self.assertIn("FAIL_PAYMENT", factors)
        self.assertIn("card testing", r["diagnostics"]["recommended_action"])

    def test_verdicts_reproduce_the_reported_hybrid_numbers(self) -> None:
        """The API's verdict logic and the evaluation report must agree on every test order."""
        hybrid = next(m for m in self.t.results["test"]["models"] if m["model"] == "hybrid")
        caught = false_alarms = 0
        for o in self.test_orders:
            if o.category == "RULE_VIOLATION":
                continue
            r = self.analyze(o.events, stamps_of(o), o.amount)
            flagged = r["verdict"] == "STATISTICAL_ANOMALY"
            self.assertEqual(flagged, r["ml_evaluation"]["is_ml_anomaly"] or r["timing_evaluation"]["is_timing_anomaly"])
            caught += flagged and o.is_anomaly
            false_alarms += flagged and not o.is_anomaly
        self.assertEqual((caught, false_alarms), (hybrid["anomalies_caught"], hybrid["false_alarms"]))

    def test_every_generated_rule_violation_is_rejected(self) -> None:
        for o in self.test_orders:
            if o.category == "RULE_VIOLATION":
                self.assertEqual(self.analyze(o.events)["verdict"], "RULE_VIOLATION", o.trace_id)


class TestInputErrors(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.t = trained()
        cls.order = next(o for o in cls.t.orders if o.category == "NORMAL" and o.events == HAPPY)

    def assert_input_error(self, fragment: str, stamps, amount) -> None:
        with self.assertRaises(InputError) as ctx:
            analyze_trace(self.order.events, stamps, amount, models=self.t)
        self.assertTrue(str(ctx.exception).startswith("[INPUT] expected "))
        self.assertIn(fragment, str(ctx.exception))

    def test_timing_inputs_come_together(self) -> None:
        self.assert_input_error("together", stamps_of(self.order), None)
        self.assert_input_error("together", None, 1200.0)

    def test_one_timestamp_per_event(self) -> None:
        self.assert_input_error("4 timestamps", stamps_of(self.order)[:3], 1200.0)

    def test_positive_finite_amount(self) -> None:
        for bad in (0.0, -5.0, math.nan, math.inf):
            with self.subTest(amount=bad):
                self.assert_input_error("positive", stamps_of(self.order), bad)


class TestExplanations(unittest.TestCase):

    def measurement(self, unit: str, value: float, percentile: float, step=None, unusual=False):
        return {"name": "PACK_ITEMS -> SHIP_ITEMS" if unit == "seconds" else "order_amount", "step": step,
                "unit": unit, "value": value, "normal_low": 1.0, "normal_high": 2.0,
                "percentile": percentile, "is_unusual": unusual}

    def test_describe_measurement_says_which_way_and_how_far(self) -> None:
        slow = describe_measurement(self.measurement("seconds", 90_000, 99.9, step=3, unusual=True))
        self.assertIn("step 3", slow)
        self.assertIn("slower than 99.9% of normal orders", slow)
        cheap = describe_measurement(self.measurement("currency", 40.0, 0.5, unusual=True))
        self.assertIn("lower than 99.5% of normal orders", cheap)

    def test_a_combination_is_explained_when_no_single_value_is_unusual(self) -> None:
        timing = {"is_timing_anomaly": True, "measurements": [
            self.measurement("currency", 4_000.0, 96.0),
            self.measurement("seconds", 2_000.0, 4.0, step=2),
            self.measurement("seconds", 9_000.0, 60.0, step=3),
        ]}
        reasons = timing_reasons(timing)
        self.assertIn("combination", reasons[0])
        self.assertIn("higher than 96.0% of normal orders", reasons[1])
        self.assertIn("faster than 96.0% of normal orders", reasons[2])
        self.assertEqual(len(reasons), 3)  # the 60th-percentile step is not a reason


if __name__ == "__main__":
    unittest.main()
