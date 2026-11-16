"""TRAIN / MODEL checkpoints for the Markov chain path model (backend/ml/markov/model.py).

Probabilities are worked out by hand from 4 training orders (alpha = 1, k = allowed events):
  ORDER_CREATED   visited 5 times: CONFIRM 3, CANCEL 1, FAIL 1 -> (3+1)/(5+3), (1+1)/8, (1+1)/8
  PAYMENT_FAILED  visited once:    RETRY 1                    -> (1+1)/(1+3), 1/4, 1/4
  PAID            visited 3 times: PACK 2, REFUND 1           -> (2+1)/(3+2), (1+1)/5
  PACKED          visited twice:   SHIP 2                     -> (2+1)/(2+2), 1/4
  SHIPPED         visited twice:   DELIVER 2                  -> (2+1)/(2+1)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from backend.core.errors import FeatureError, ModelLoadError, TrainError
from backend.fsm.transitions import TRANSITION_TABLE
from backend.ml.markov.model import (
    THRESHOLD_PERCENTILE,
    describe_probability,
    fit_markov_model,
    load_markov_model,
    model_to_dict,
    save_markov_model,
    score_trace,
)

HAPPY = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]
REFUND_AFTER_RETRY = ["FAIL_PAYMENT", "RETRY_PAYMENT", "CONFIRM_PAYMENT", "REFUND_ORDER"]
TRAINING = [HAPPY, HAPPY, ["CANCEL_ORDER"], REFUND_AFTER_RETRY]
EXPECTED = {
    ("ORDER_CREATED", "CONFIRM_PAYMENT"): 0.5,   # (3 + 1) / (5 + 3): one visit comes back after RETRY
    ("ORDER_CREATED", "CANCEL_ORDER"): 0.25,
    ("ORDER_CREATED", "FAIL_PAYMENT"): 0.25,
    ("PAYMENT_FAILED", "RETRY_PAYMENT"): 0.5,
    ("PAYMENT_FAILED", "CONFIRM_PAYMENT"): 0.25,
    ("PAYMENT_FAILED", "CANCEL_ORDER"): 0.25,
    ("PAID", "PACK_ITEMS"): 0.6,
    ("PAID", "REFUND_ORDER"): 0.4,
    ("PACKED", "SHIP_ITEMS"): 0.75,
    ("PACKED", "REFUND_ORDER"): 0.25,
    ("SHIPPED", "DELIVER_ITEMS"): 1.0,
}
LOG_HAPPY = math.log10(0.5 * 0.6 * 0.75 * 1.0)
LOG_REFUND_AFTER_RETRY = math.log10(0.25 * 0.5 * 0.5 * 0.4)
VALIDATION = [HAPPY] * 99 + [REFUND_AFTER_RETRY]


def expected_threshold() -> float:
    """1st percentile of 100 sorted values [LOG_REFUND_AFTER_RETRY, LOG_HAPPY x 99], linear interpolation:
    position 0.01 * (100 - 1) = 0.99 between the 1st and 2nd value."""
    return LOG_REFUND_AFTER_RETRY + 0.99 * (LOG_HAPPY - LOG_REFUND_AFTER_RETRY)


class TestMarkovTraining(unittest.TestCase):

    def setUp(self) -> None:
        self.model = fit_markov_model(TRAINING, VALIDATION, alpha=1.0)

    def test_probabilities_match_hand_calculation(self) -> None:
        self.assertEqual(set(self.model.probabilities), set(TRANSITION_TABLE))
        for key, expected in EXPECTED.items():
            self.assertAlmostEqual(self.model.probabilities[key], expected, places=12, msg=key)

    def test_probabilities_out_of_each_state_sum_to_one(self) -> None:
        for state in {state for state, _ in TRANSITION_TABLE}:
            total = sum(p for (s, _), p in self.model.probabilities.items() if s == state)
            self.assertAlmostEqual(total, 1.0, places=12, msg=state)

    def test_unseen_allowed_transition_gets_a_small_nonzero_probability(self) -> None:
        self.assertEqual(self.model.counts[("PAYMENT_FAILED", "CONFIRM_PAYMENT")], 0)
        self.assertGreater(self.model.probabilities[("PAYMENT_FAILED", "CONFIRM_PAYMENT")], 0.0)

    def test_threshold_is_the_first_percentile_of_validation_paths(self) -> None:
        self.assertEqual(THRESHOLD_PERCENTILE, 1.0)
        self.assertAlmostEqual(self.model.threshold_log10_probability, expected_threshold(), places=12)
        self.assertEqual((self.model.training_sample_count, self.model.validation_sample_count), (4, 100))


class TestMarkovTrainErrors(unittest.TestCase):

    def assert_train_error(self, fragment: str, train, validation=VALIDATION, alpha: float = 1.0) -> None:
        with self.assertRaises(TrainError) as ctx:
            fit_markov_model(train, validation, alpha=alpha)
        self.assertTrue(str(ctx.exception).startswith("[TRAIN] expected "))
        self.assertIn(fragment, str(ctx.exception))

    def test_rule_violation_in_training_data(self) -> None:
        self.assert_train_error("SHIP_ITEMS", [HAPPY, ["CONFIRM_PAYMENT", "SHIP_ITEMS"]])

    def test_unfinished_order_in_training_data(self) -> None:
        self.assert_train_error("is_complete=False", [HAPPY, ["CONFIRM_PAYMENT", "PACK_ITEMS"]])

    def test_no_training_orders(self) -> None:
        self.assert_train_error("at least one", [])

    def test_alpha_must_be_positive(self) -> None:
        for alpha in (0.0, -1.0):
            self.assert_train_error("alpha", TRAINING, alpha=alpha)

    def test_too_few_validation_orders(self) -> None:
        self.assert_train_error("validation", TRAINING, validation=[HAPPY] * 10)

    def test_invalid_validation_order(self) -> None:
        self.assert_train_error("SHIP_ITEMS", TRAINING, validation=[HAPPY] * 99 + [["SHIP_ITEMS"]])


class TestMarkovScoring(unittest.TestCase):

    def setUp(self) -> None:
        self.model = fit_markov_model(TRAINING, VALIDATION, alpha=1.0)

    def test_path_probability_is_the_product_of_step_probabilities(self) -> None:
        result = score_trace(self.model, HAPPY)
        self.assertAlmostEqual(result["path_log10_probability"], LOG_HAPPY, places=12)
        self.assertEqual(
            [(t["step"], t["from_state"], t["event"], t["to_state"], t["probability"]) for t in result["transitions"]],
            [(1, "ORDER_CREATED", "CONFIRM_PAYMENT", "PAID", 0.5), (2, "PAID", "PACK_ITEMS", "PACKED", 0.6),
             (3, "PACKED", "SHIP_ITEMS", "SHIPPED", 0.75), (4, "SHIPPED", "DELIVER_ITEMS", "DELIVERED", 1.0)],
        )

    def test_common_path_is_normal(self) -> None:
        result = score_trace(self.model, HAPPY)
        self.assertEqual((result["prediction"], result["is_ml_anomaly"]), ("NORMAL", False))
        self.assertAlmostEqual(result["anomaly_score"], expected_threshold() - LOG_HAPPY, places=12)
        self.assertLess(result["anomaly_score"], 0)

    def test_path_rarer_than_the_threshold_is_an_anomaly(self) -> None:
        result = score_trace(self.model, REFUND_AFTER_RETRY)
        self.assertEqual((result["prediction"], result["is_ml_anomaly"]), ("ANOMALY", True))
        self.assertAlmostEqual(result["anomaly_score"], expected_threshold() - LOG_REFUND_AFTER_RETRY, places=12)

    def test_scoring_a_rule_violation_raises_a_feature_error(self) -> None:
        with self.assertRaises(FeatureError) as ctx:
            score_trace(self.model, ["CONFIRM_PAYMENT", "SHIP_ITEMS"])
        self.assertIn("SHIP_ITEMS", str(ctx.exception))

    def test_describe_probability(self) -> None:
        self.assertEqual(describe_probability(math.log10(0.791)), "79.1% of orders")
        self.assertEqual(describe_probability(math.log10(1 / 27300)), "about 1 in 27,300 orders")


class TestMarkovPersistence(unittest.TestCase):

    def setUp(self) -> None:
        self.model = fit_markov_model(TRAINING, VALIDATION, alpha=1.0)
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "markov.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def assert_load_error(self, fragment: str) -> None:
        with self.assertRaises(ModelLoadError) as ctx:
            load_markov_model(self.path)
        self.assertTrue(str(ctx.exception).startswith("[MODEL] expected "))
        self.assertIn(fragment, str(ctx.exception))

    def test_save_and_load_round_trip(self) -> None:
        save_markov_model(self.model, self.path)
        self.assertEqual(load_markov_model(self.path), self.model)

    def test_missing_file(self) -> None:
        self.assert_load_error("markov.json")

    def test_invalid_json(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        self.assert_load_error("JSON")

    def test_missing_field(self) -> None:
        data = model_to_dict(self.model)
        del data["threshold_log10_probability"]
        self._write(data)
        self.assert_load_error("threshold_log10_probability")

    def test_missing_transition(self) -> None:
        data = model_to_dict(self.model)
        data["transitions"] = data["transitions"][1:]
        self._write(data)
        self.assert_load_error("transitions")

    def test_transition_not_in_the_fsm(self) -> None:
        data = model_to_dict(self.model)
        data["transitions"][0]["to_state"] = "DELIVERED"
        self._write(data)
        self.assert_load_error("DELIVERED")

    def test_probabilities_that_do_not_sum_to_one(self) -> None:
        data = model_to_dict(self.model)
        data["transitions"][0]["probability"] = 0.99
        self._write(data)
        self.assert_load_error("sum to 1")


if __name__ == "__main__":
    unittest.main()
