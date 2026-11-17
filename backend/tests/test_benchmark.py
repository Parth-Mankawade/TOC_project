"""The multi-seed benchmark and the alternatives it measures
(backend/ml/evaluation/benchmark.py, backend/ml/evaluation/experiments.py).

The benchmark itself is run on a small dataset and two seeds: these tests check
that it is put together correctly, not how good the numbers are.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from backend.core.errors import EvaluateError, TrainError
from backend.ml import hybrid
from backend.ml.dataset.generator import ANOMALY_TYPES
from backend.ml.evaluation import experiments
from backend.ml.evaluation.benchmark import METRICS, NOT_ADOPTED, orders_for_seed, run_benchmark
from backend.tests.fixtures import FIXTURE_SIZES

HAPPY = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]
ONE_RETRY = ["FAIL_PAYMENT", "RETRY_PAYMENT"] + HAPPY
TWO_RETRIES = ["FAIL_PAYMENT", "RETRY_PAYMENT"] * 2 + HAPPY


class TestBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.seeds = [42, 7]
        cls.report = run_benchmark(cls.seeds, FIXTURE_SIZES)

    def test_every_method_is_reported_on_both_splits(self) -> None:
        expected = (["previous_hybrid", "tuned_range_rule", "zscore_or_markov", "hybrid"]
                    + [f"hybrid_without_{check}" for check in hybrid.CHECKS] + NOT_ADOPTED)
        self.assertEqual(sorted(self.report["methods"]), sorted(expected))
        for name, method in self.report["methods"].items():
            for split in ("validation", "test"):
                summary = method[split]
                self.assertEqual(len(summary["f1_by_seed"]), len(self.seeds), name)
                self.assertEqual(list(summary["caught_by_type"]), ANOMALY_TYPES, name)
                for metric in ("f1", "precision", "recall", "false_alarm_rate"):
                    self.assertTrue(0.0 <= summary[metric]["mean"] <= 1.0, (name, split, metric))
                    self.assertGreaterEqual(summary[metric]["std"], 0.0)

    def test_mean_is_the_mean_of_the_seeds(self) -> None:
        summary = self.report["methods"]["hybrid"]["test"]
        self.assertAlmostEqual(summary["f1"]["mean"], float(np.mean(summary["f1_by_seed"])), places=4)
        self.assertEqual(set(METRICS), {"f1", "precision", "recall", "false_alarm_rate", "roc_auc", "pr_auc"})

    def test_single_score_methods_flag_the_budget_on_validation_normals(self) -> None:
        n = FIXTURE_SIZES["validation"].normal
        for name, method in self.report["methods"].items():
            if name in ("previous_hybrid", "zscore_or_markov", "stacking"):
                continue        # OR rules flag 1% per part; stacking is thresholded on half of the orders
            rate = method["validation"]["false_alarm_rate"]["mean"]
            self.assertAlmostEqual(rate, self.report["false_alarm_budget"], delta=1.5 / n, msg=name)
            self.assertIsNotNone(method["test"]["pr_auc"], name)

    def test_or_rules_have_no_single_score(self) -> None:
        for name in ("previous_hybrid", "zscore_or_markov"):
            self.assertIsNone(self.report["methods"][name]["test"]["roc_auc"])
            self.assertIsNone(self.report["methods"][name]["test"]["pr_auc"])

    def test_leave_one_type_out_is_reported_for_every_type(self) -> None:
        left_out = self.report["stacking_leave_one_type_out"]
        self.assertEqual(list(left_out), ANOMALY_TYPES)
        for anomaly_type, by_split in left_out.items():
            self.assertEqual(by_split["test"]["total"], FIXTURE_SIZES["test"].per_anomaly_type)
            self.assertTrue(0 <= by_split["test"]["mean_caught"] <= by_split["test"]["total"], anomaly_type)

    def test_seed_data_matches_the_requested_sizes_and_never_mixes_splits(self) -> None:
        data = orders_for_seed(42, FIXTURE_SIZES)
        self.assertEqual(len(data.train), FIXTURE_SIZES["train"].normal)
        self.assertEqual(len(data.validation_normal), FIXTURE_SIZES["validation"].normal)
        self.assertEqual(len(data.validation_anomalies), FIXTURE_SIZES["validation"].per_anomaly_type * len(ANOMALY_TYPES))
        self.assertEqual({o.split for o in data.train}, {"train"})
        self.assertEqual({o.split for o in data.validation_normal + data.validation_anomalies}, {"validation"})
        self.assertEqual({o.split for o in data.test}, {"test"})
        self.assertFalse(any(o.is_anomaly for o in data.train + data.validation_normal))

    def test_no_seeds_is_an_error(self) -> None:
        with self.assertRaises(EvaluateError):
            run_benchmark([], FIXTURE_SIZES)


class TestExperiments(unittest.TestCase):

    def test_retry_aware_chain_separates_first_attempts_from_retries(self) -> None:
        """Worked by hand. Training: HAPPY twice, ONE_RETRY, TWO_RETRIES. alpha = 1.
        Each state is counted separately for 0, 1 and 2+ failed payments so far:
          ORDER_CREATED, 0 failures: CONFIRM 2, FAIL 2 (3 events allowed) -> P(CONFIRM) = P(FAIL) = 3/7
          PAYMENT_FAILED, 1 failure: RETRY 2           (3 allowed)        -> P(RETRY) = 3/5
          ORDER_CREATED, 1 failure:  CONFIRM 1, FAIL 1 (3 allowed)        -> P(CONFIRM) = 2/5
          PAID, 0 failures: PACK 2 (2 allowed) -> 3/4      PAID, 1 failure: PACK 1 -> 2/3
          PACKED, 0 failures: SHIP 2 (2 allowed) -> 3/4    PACKED, 1 failure: SHIP 1 -> 2/3
          SHIPPED: DELIVER is the only event allowed -> 1
        The plain chain would give CONFIRM_PAYMENT from ORDER_CREATED one probability, not 3/7 and 2/5."""
        log10_path = experiments.fit_retry_aware_path([HAPPY, HAPPY, ONE_RETRY, TWO_RETRIES])
        self.assertAlmostEqual(log10_path(HAPPY), math.log10((3 / 7) * (3 / 4) * (3 / 4) * 1), places=12)
        self.assertAlmostEqual(log10_path(ONE_RETRY),
                               math.log10((3 / 7) * (3 / 5) * (2 / 5) * (2 / 3) * (2 / 3) * 1), places=12)

    def test_retry_aware_chain_rejects_a_rule_violation(self) -> None:
        with self.assertRaises(TrainError):
            experiments.fit_retry_aware_path([["CONFIRM_PAYMENT", "SHIP_ITEMS"]])

    def test_rank_average_adds_the_tail_shares_of_every_detector(self) -> None:
        normal = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([10.0, 20.0, 30.0])}
        scores = {"a": np.array([0.0, 5.0]), "b": np.array([0.0, 50.0])}
        # Below every normal score: share (3 + 1) / (3 + 1) = 1 -> 0. Above every one: 1 / 4 -> log10(4), twice.
        np.testing.assert_allclose(experiments.rank_average(normal, scores), [0.0, 2 * math.log10(4)])

    def test_stack_needs_both_classes_and_scores_anomalies_higher(self) -> None:
        levels = np.array([[0.0, 0.1], [0.2, 0.0], [0.1, 0.1], [3.0, 2.0], [2.5, 3.0], [3.5, 2.5]])
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = experiments.fit_stack(levels, labels)(levels)
        self.assertGreater(scores[labels == 1].min(), scores[labels == 0].max())
        with self.assertRaises(TrainError):
            experiments.fit_stack(levels[:3], labels[:3])


if __name__ == "__main__":
    unittest.main()
