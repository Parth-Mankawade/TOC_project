"""FEATURES checkpoint: turning an order into the timing/amount model's inputs (backend/ml/timing/features.py).

Expected values are worked out by hand from small, exact statistics.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import tempfile
import unittest

from backend.core.errors import FeatureError, TrainError
from backend.fsm.reachability import legal_step_kinds
from backend.ml.dataset import generator as gen
from backend.ml.dataset.loader import load_orders, select
from backend.ml.timing.features import (
    MODEL_INPUTS,
    NORMAL_BAND,
    FeatureStats,
    LogNormalStats,
    fit_feature_stats,
    measurements,
    order_features,
    step_durations,
    step_name,
)

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
HAPPY = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]


def times(*gaps: float) -> list[datetime]:
    out = [T0]
    for g in gaps:
        out.append(out[-1] + timedelta(seconds=g))
    return out


def exact_stats() -> FeatureStats:
    """Amount: ln-mean ln(1000), ln-std 1. Every step: ln-mean ln(3600), ln-std 1. Failures: 0 x98, 1 x1, 2 x1."""
    step = LogNormalStats(mean_ln=math.log(3600), std_ln=1.0, count=50)
    return FeatureStats(
        amount=LogNormalStats(mean_ln=math.log(1000), std_ln=1.0, count=100),
        steps={step_name(a, b): step for a, b in legal_step_kinds()},
        failure_counts={0: 98, 1: 1, 2: 1},
        training_sample_count=100,
    )


class TestLogNormalStats(unittest.TestCase):

    def test_fit_z_and_percentile(self) -> None:
        stats = LogNormalStats.fit([math.e ** 0, math.e ** 2])
        self.assertAlmostEqual(stats.mean_ln, 1.0)
        self.assertAlmostEqual(stats.std_ln, 1.0)
        self.assertAlmostEqual(stats.z(math.e), 0.0)
        self.assertAlmostEqual(stats.z(math.e ** 2), 1.0)
        self.assertAlmostEqual(stats.percentile(math.e), 50.0)
        self.assertAlmostEqual(stats.quantile(50.0), math.e)

    def test_fit_needs_positive_values_with_spread(self) -> None:
        with self.assertRaises(TrainError):
            LogNormalStats.fit([5.0, 5.0, 5.0])
        with self.assertRaises(TrainError):
            LogNormalStats.fit([1.0, 0.0])


class TestOrderFeatures(unittest.TestCase):

    def test_delivered_order(self) -> None:
        f = order_features(exact_stats(), HAPPY, times(3600, 3600 * math.e, 3600 / math.e), 1000 * math.e ** 2)
        self.assertEqual(list(f), MODEL_INPUTS)
        self.assertAlmostEqual(f["amount_z"], 2.0)
        self.assertAlmostEqual(f["slowest_step_z"], 1.0)
        self.assertAlmostEqual(f["fastest_step_z"], -1.0)
        self.assertAlmostEqual(f["mean_step_z"], 0.0)
        self.assertEqual((f["payment_failures"], f["refunded"], f["cancelled"]), (0.0, 0.0, 0.0))

    def test_order_with_no_steps_has_no_timing_deviation(self) -> None:
        f = order_features(exact_stats(), ["CANCEL_ORDER"], times(), 1000.0)
        self.assertEqual((f["slowest_step_z"], f["fastest_step_z"], f["mean_step_z"]), (0.0, 0.0, 0.0))
        self.assertEqual(f["cancelled"], 1.0)

    def test_failures_and_refund_are_counted(self) -> None:
        events = ["FAIL_PAYMENT", "RETRY_PAYMENT", "CONFIRM_PAYMENT", "REFUND_ORDER"]
        f = order_features(exact_stats(), events, times(3600, 3600, 3600), 1000.0)
        self.assertEqual((f["payment_failures"], f["refunded"], f["cancelled"]), (1.0, 1.0, 0.0))

    def test_bad_inputs_raise_feature_errors_that_name_the_problem(self) -> None:
        stats = exact_stats()
        cases = {
            "positive": (HAPPY, times(1, 1, 1), 0.0),
            "4 timestamps": (HAPPY, times(1, 1), 1000.0),
            "strictly increasing": (HAPPY, times(1, 0, 1), 1000.0),
            "FSM-valid, complete": (["CONFIRM_PAYMENT", "SHIP_ITEMS", "PACK_ITEMS"], times(1, 1), 1000.0),
            "PACK_ITEMS": (["CONFIRM_PAYMENT", "PACK_ITEMS"], times(1), 1000.0),  # valid but not finished
        }
        for fragment, (events, stamps, amount) in cases.items():
            with self.subTest(fragment), self.assertRaises(FeatureError) as ctx:
                order_features(stats, events, stamps, amount)
            self.assertTrue(str(ctx.exception).startswith("[FEATURES] expected "))
            self.assertIn(fragment, str(ctx.exception))

    def test_a_step_kind_the_model_never_learned(self) -> None:
        stats = exact_stats()
        del stats.steps["SHIP_ITEMS -> DELIVER_ITEMS"]
        with self.assertRaises(FeatureError) as ctx:
            order_features(stats, HAPPY, times(1, 1, 1), 1000.0)
        self.assertIn("SHIP_ITEMS -> DELIVER_ITEMS", str(ctx.exception))


class TestMeasurements(unittest.TestCase):

    def test_each_value_gets_its_percentile_among_normal_orders(self) -> None:
        events = ["FAIL_PAYMENT", "RETRY_PAYMENT", "CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]
        rows = measurements(exact_stats(), events, times(3600, 3600, 3600, 3600 * math.e ** 3, 3600), 1000.0)
        by_name = {(r["name"], r["step"]): r for r in rows}
        amount = by_name[("order_amount", None)]
        self.assertAlmostEqual(amount["percentile"], 50.0)
        self.assertFalse(amount["is_unusual"])
        slow = by_name[("PACK_ITEMS -> SHIP_ITEMS", 5)]
        self.assertAlmostEqual(slow["percentile"], 99.865, places=2)  # z = 3
        self.assertTrue(slow["is_unusual"])
        low, high = NORMAL_BAND
        self.assertAlmostEqual(slow["normal_low"], exact_stats().steps["PACK_ITEMS -> SHIP_ITEMS"].quantile(low))
        self.assertAlmostEqual(slow["normal_high"], exact_stats().steps["PACK_ITEMS -> SHIP_ITEMS"].quantile(high))
        failures = by_name[("payment_failures", None)]
        self.assertAlmostEqual(failures["percentile"], 98.5)  # 98 below, 1 equal (counted half) out of 100
        self.assertEqual(len([r for r in rows if r["step"] is not None]), len(events) - 1)


class TestFitFeatureStats(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cls.train = select(load_orders(gen.generate_dataset(Path(tmp) / "o.csv")), "NORMAL", "train")

    def test_learns_every_legal_step_kind(self) -> None:
        stats = fit_feature_stats([(o.events, o.timestamps, o.amount) for o in self.train])
        self.assertEqual(set(stats.steps), {step_name(a, b) for a, b in legal_step_kinds()})
        self.assertEqual(stats.training_sample_count, len(self.train))
        self.assertAlmostEqual(math.exp(stats.amount.mean_ln), gen.NORMAL_AMOUNT[0], delta=0.1 * gen.NORMAL_AMOUNT[0])

    def test_too_few_orders(self) -> None:
        with self.assertRaises(TrainError) as ctx:
            fit_feature_stats([(o.events, o.timestamps, o.amount) for o in self.train[:10]])
        self.assertIn("10", str(ctx.exception))

    def test_a_step_kind_missing_from_training(self) -> None:
        happy_only = [(o.events, o.timestamps, o.amount) for o in self.train if o.events == HAPPY]
        with self.assertRaises(TrainError) as ctx:
            fit_feature_stats(happy_only)
        self.assertIn("FAIL_PAYMENT -> ", str(ctx.exception))

    def test_an_invalid_training_order(self) -> None:
        bad = [(o.events, o.timestamps, o.amount) for o in self.train]
        bad[3] = (["SHIP_ITEMS"], bad[3][1][:1], bad[3][2])
        with self.assertRaises(TrainError) as ctx:
            fit_feature_stats(bad)
        self.assertIn("SHIP_ITEMS", str(ctx.exception))


class TestStepDurations(unittest.TestCase):

    def test_durations_between_consecutive_events(self) -> None:
        steps = step_durations(HAPPY, times(10, 20, 30))
        self.assertEqual([(s.step, s.from_event, s.to_event, s.seconds) for s in steps], [
            (2, "CONFIRM_PAYMENT", "PACK_ITEMS", 10.0),
            (3, "PACK_ITEMS", "SHIP_ITEMS", 20.0),
            (4, "SHIP_ITEMS", "DELIVER_ITEMS", 30.0),
        ])


if __name__ == "__main__":
    unittest.main()
