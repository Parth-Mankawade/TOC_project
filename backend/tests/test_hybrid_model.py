"""CALIBRATE checkpoint (backend/ml/hybrid.py): four checks, one threshold, one false-alarm budget.

Uses real models trained on the fixture data (tests/fixtures.py); trained() has
already calibrated them, so the tests refit an uncalibrated pair where they need one.
"""

from __future__ import annotations

import unittest

import numpy as np

from backend.core.errors import FeatureError, TrainError
from backend.ml import hybrid
from backend.ml.evaluation.evaluate import timed
from backend.ml.markov.model import fit_markov_model, path_log10_probability, score_trace
from backend.ml.timing.model import DEFAULT_THRESHOLD, anomaly_scores, check_levels, fit_timing_model
from backend.tests.fixtures import trained


class TestHybrid(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.t = trained()
        orders = cls.t.orders
        cls.train = [o for o in orders if o.split == "train" and o.category == "NORMAL"]
        cls.val_normal = [o for o in orders if o.split == "validation" and o.category == "NORMAL"]
        cls.val_anomalies = [o for o in orders if o.split == "validation" and o.is_anomaly]
        cls.markov = fit_markov_model([o.events for o in cls.train], [o.events for o in cls.val_normal])
        cls.timing = fit_timing_model(timed(cls.train), timed(cls.val_normal), cls.t.timing.detector_name)

    def test_levels_are_the_path_level_and_the_three_timing_levels(self) -> None:
        orders = timed(self.val_anomalies)
        levels = hybrid.levels(self.markov, self.timing, orders)
        self.assertEqual(list(levels), ["path", "cluster", "trend", "deviation"])
        expected_path = [self.markov.level(path_log10_probability(self.markov, e)) for e, _, _ in orders]
        np.testing.assert_allclose(levels["path"], expected_path)
        for check, values in check_levels(self.timing, orders).items():
            np.testing.assert_allclose(levels[check], values)
        np.testing.assert_allclose(hybrid.hybrid_levels(self.markov, self.timing, orders),
                                   np.max(list(levels.values()), axis=0))

    def test_uncalibrated_models_each_flag_their_own_one_percent(self) -> None:
        self.assertEqual(self.timing.threshold, DEFAULT_THRESHOLD)
        self.assertEqual(self.markov.threshold_log10_probability, self.markov.rare_log10_probability)
        flagged = float(np.mean(hybrid.hybrid_scores(self.markov, self.timing, timed(self.val_normal)) > 0))
        self.assertGreater(flagged, hybrid.FALSE_ALARM_BUDGET)      # four checks at 1% each: more than the budget

    def test_calibrate_meets_the_budget_on_validation_normals(self) -> None:
        """At most the budget, plus one order for rounding. Fewer only when ties force it: many
        orders share a path, so many share a path level. If the threshold falls on a group of equal
        levels, the group is all in or all out, and taking it in would exceed the budget."""
        n = len(self.val_normal)
        thresholds = []
        for budget in (hybrid.FALSE_ALARM_BUDGET, 0.03):
            markov, timing = hybrid.calibrate(self.markov, self.timing, timed(self.val_normal), budget)
            levels = np.sort(hybrid.hybrid_levels(markov, timing, timed(self.val_normal)))[::-1]
            flagged = int(np.sum(levels > timing.threshold))
            self.assertLessEqual(flagged, budget * n + 1, budget)
            if flagged < budget * n - 1:
                next_level = levels[flagged]                       # highest level not flagged
                group = int(np.sum(np.isclose(levels, next_level, rtol=0, atol=1e-12)))
                self.assertGreater(flagged + group, budget * n + 1,
                                   f"budget {budget}: {flagged} flagged; the next {group} equal levels would fit")
            thresholds.append(timing.threshold)
        self.assertGreater(thresholds[0], thresholds[1])     # a smaller budget needs a higher threshold

    def test_calibrate_gives_both_models_the_same_threshold_and_changes_nothing_else(self) -> None:
        markov, timing = hybrid.calibrate(self.markov, self.timing, timed(self.val_normal))
        self.assertGreater(timing.threshold, DEFAULT_THRESHOLD)      # stricter than each check alone
        self.assertAlmostEqual(markov.level(markov.threshold_log10_probability), timing.threshold, places=9)
        self.assertEqual(markov.probabilities, self.markov.probabilities)
        self.assertEqual(timing.scales, self.timing.scales)
        self.assertIs(timing.detector, self.timing.detector)
        self.assertEqual((self.markov.threshold_log10_probability, self.timing.threshold),
                         (self.markov.rare_log10_probability, DEFAULT_THRESHOLD))    # the inputs are untouched

    def test_hybrid_flag_is_the_or_of_the_two_models_own_decisions(self) -> None:
        """What the analyzer does (Markov decision OR timing decision) is what hybrid_scores reports."""
        markov, timing = self.t.markov, self.t.timing
        orders = timed(self.val_normal + self.val_anomalies)
        by_models = [score_trace(markov, e)["is_ml_anomaly"] or s > 0
                     for (e, _, _), s in zip(orders, anomaly_scores(timing, orders))]
        self.assertEqual(by_models, list(hybrid.hybrid_scores(markov, timing, orders) > 0))
        self.assertTrue(any(by_models) and not all(by_models))

    def test_the_saved_models_are_calibrated_to_the_budget(self) -> None:
        flagged = float(np.mean(hybrid.hybrid_scores(self.t.markov, self.t.timing, timed(self.val_normal)) > 0))
        self.assertAlmostEqual(flagged, hybrid.FALSE_ALARM_BUDGET, delta=1 / len(self.val_normal))
        self.assertEqual(self.t.results["hybrid"]["threshold_level"], self.t.timing.threshold)

    def test_bad_budgets_and_bad_orders_are_errors(self) -> None:
        for budget in (0.0, 1.0, -0.1):
            with self.subTest(budget=budget), self.assertRaises(TrainError) as ctx:
                hybrid.calibrate(self.markov, self.timing, timed(self.val_normal), budget)
            self.assertIn("budget", str(ctx.exception))
        with self.assertRaises(TrainError):
            hybrid.threshold_for([-1.0, -0.5, -0.2, -0.1], 0.5)      # no positive threshold
        events, stamps, amount = timed(self.val_normal)[0]
        with self.assertRaises(FeatureError):
            hybrid.levels(self.markov, self.timing, [(events[:1], stamps[:1], amount)])


if __name__ == "__main__":
    unittest.main()
