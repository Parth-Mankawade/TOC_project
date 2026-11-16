"""The analysis service (backend/services/anomaly_service.py): loads each model once, only when an
order needs it, and analyzes single orders and batches. Uses the fixture models and real files."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from backend.core.errors import InputError, ModelLoadError
from backend.services.anomaly_service import AnomalyService, OrderInput
from backend.tests.fixtures import HAPPY, request_body, trained


class TestService(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.t = trained()
        cls.service = AnomalyService(cls.t.paths.markov, cls.t.paths.timing)
        cls.order = next(o for o in cls.t.orders if o.split == "test" and o.category == "NORMAL" and o.events == HAPPY)

    def test_result_echoes_the_order_id_and_timing_inputs(self) -> None:
        body = request_body(self.order)
        r = self.service.analyze_order(OrderInput(**body))
        self.assertEqual((r["order_id"], r["event_timestamps"], r["order_amount"]),
                         (body["order_id"], body["event_timestamps"], body["order_amount"]))
        self.assertIn(r["verdict"], ("NORMAL", "STATISTICAL_ANOMALY"))
        self.assertIsNotNone(r["timing_evaluation"])

    def test_each_model_is_loaded_once_and_reused(self) -> None:
        service = AnomalyService(self.t.paths.markov, self.t.paths.timing)
        self.assertIs(service.markov_model(), service.markov_model())
        self.assertIs(service.timing_model(), service.timing_model())

    def test_batch_summary_counts_every_verdict(self) -> None:
        orders = [OrderInput("CONFIRM_PAYMENT;SHIP_ITEMS"), OrderInput(["CONFIRM_PAYMENT", "PACK_ITEMS"]),
                  OrderInput(["FAIL_PAYMENT", "RETRY_PAYMENT"] * 5 + HAPPY), OrderInput(HAPPY)]
        batch = self.service.analyze_batch(orders)
        self.assertEqual([r["verdict"] for r in batch["results"]],
                         ["RULE_VIOLATION", "VALID_INCOMPLETE", "STATISTICAL_ANOMALY", "NORMAL"])
        self.assertEqual(batch["summary"], {
            "total_orders": 4, "normal_count": 1, "rule_violation_count": 1, "statistical_anomaly_count": 1,
            "valid_incomplete_count": 1, "total_anomalies": 2, "anomaly_rate": 0.5,
        })

    def test_an_empty_batch_is_an_input_error(self) -> None:
        with self.assertRaises(InputError) as ctx:
            self.service.analyze_batch([])
        self.assertIn("at least one order", str(ctx.exception))

    def test_readiness_loads_both_models(self) -> None:
        ready = self.service.readiness()
        self.assertEqual(ready["service_status"], "OPERATIONAL")
        self.assertEqual(ready["timing_model"]["model_name"], self.t.timing.detector_name)
        self.assertEqual(ready["markov_model"]["path"], str(self.t.paths.markov))


class TestMissingModels(unittest.TestCase):
    """A service whose model files do not exist."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        missing = Path(self._tmp.name)
        self.service = AnomalyService(missing / "markov.json", missing / "timing.joblib")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_rule_checks_need_no_model(self) -> None:
        self.assertEqual(self.service.analyze_order(OrderInput("CONFIRM_PAYMENT;SHIP_ITEMS"))["verdict"], "RULE_VIOLATION")
        self.assertEqual(self.service.analyze_order(OrderInput(["CONFIRM_PAYMENT"]))["verdict"], "VALID_INCOMPLETE")

    def test_a_complete_order_fails_loudly_naming_the_missing_file(self) -> None:
        with self.assertRaises(ModelLoadError) as ctx:
            self.service.analyze_order(OrderInput(HAPPY))
        self.assertIn("markov.json", str(ctx.exception))

    def test_readiness_fails_loudly(self) -> None:
        with self.assertRaises(ModelLoadError):
            self.service.readiness()


if __name__ == "__main__":
    unittest.main()
