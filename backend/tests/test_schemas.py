"""INPUT checkpoint at the API boundary, and the response schemas every stored result must pass
(backend/schemas/). Response checks use real analyzer output from the fixture models."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import unittest

from pydantic import ValidationError

from backend.ml.dataset.generator import format_timestamp
from backend.pipeline.analyzer import analyze_trace
from backend.schemas.anomaly import AnalysisRecordResponse, TimingMeasurementSchema
from backend.schemas.order import BatchAnalysisRequest, WorkflowAnalysisRequest
from backend.tests.fixtures import HAPPY, trained

STAMPS = ["2026-03-01T09:00:00Z", "2026-03-01T17:00:00Z", "2026-03-01T22:00:00Z", "2026-03-04T22:00:00Z"]


def record(result: dict, **extra) -> dict:
    base = {"id": 1, "order_id": None, "input_format": "EVENT_LIST", "raw_input_payload": HAPPY,
            "created_at": datetime(2026, 3, 1, tzinfo=timezone.utc), "event_timestamps": None, "order_amount": None}
    base.update({k: v for k, v in result.items() if k not in base})
    base.update(extra)
    return base


class TestRequests(unittest.TestCase):

    def assert_422(self, body: dict, fragment: str) -> None:
        with self.assertRaises(ValidationError) as ctx:
            WorkflowAnalysisRequest.model_validate(body)
        self.assertIn(fragment, str(ctx.exception))

    def test_string_or_list_with_optional_timing(self) -> None:
        WorkflowAnalysisRequest.model_validate({"event_sequence": "CONFIRM_PAYMENT;PACK_ITEMS"})
        WorkflowAnalysisRequest.model_validate({"order_id": "A1", "event_sequence": HAPPY,
                                                "event_timestamps": STAMPS, "order_amount": 1200.0})

    def test_timing_inputs_must_come_together(self) -> None:
        self.assert_422({"event_sequence": HAPPY, "event_timestamps": STAMPS}, "together")
        self.assert_422({"event_sequence": HAPPY, "order_amount": 10.0}, "together")

    def test_timestamps_must_match_events_and_increase(self) -> None:
        self.assert_422({"event_sequence": HAPPY, "event_timestamps": STAMPS[:3], "order_amount": 1.0}, "4 timestamps")
        self.assert_422({"event_sequence": HAPPY, "event_timestamps": [STAMPS[0]] * 4, "order_amount": 1.0},
                        "event_timestamps[1]")
        self.assert_422({"event_sequence": HAPPY, "event_timestamps": ["soon"] * 4, "order_amount": 1.0}, "soon")

    def test_amount_must_be_positive_and_types_strict(self) -> None:
        self.assert_422({"event_sequence": HAPPY, "event_timestamps": STAMPS, "order_amount": 0.0}, "order_amount")
        self.assert_422({"order_id": 7, "event_sequence": HAPPY}, "order_id")
        self.assert_422({"event_sequence": "CONFIRM_PAYMENT;;PACK_ITEMS"}, "event_sequence")

    def test_batch_size_limits(self) -> None:
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest.model_validate({"traces": []})
        with self.assertRaises(ValidationError):
            BatchAnalysisRequest.model_validate({"traces": ["CANCEL_ORDER"] * 501})
        BatchAnalysisRequest.model_validate({"traces": ["CANCEL_ORDER", HAPPY, {"event_sequence": HAPPY}]})


class TestResponses(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        t = trained()
        cls.timed = analyze_trace(HAPPY, STAMPS, 1200.0, models=t)
        cls.violation = analyze_trace(["CONFIRM_PAYMENT", "SHIP_ITEMS"], models=t)

    def test_real_results_are_valid_records(self) -> None:
        AnalysisRecordResponse.model_validate(record(self.timed, event_timestamps=STAMPS, order_amount=1200.0))
        AnalysisRecordResponse.model_validate(record(self.violation))

    def assert_rejected(self, edit, fragment: str) -> None:
        bad = copy.deepcopy(record(self.timed, event_timestamps=STAMPS, order_amount=1200.0))
        edit(bad)
        with self.assertRaises(ValidationError) as ctx:
            AnalysisRecordResponse.model_validate(bad)
        self.assertIn(fragment, str(ctx.exception))

    def test_path_check_must_be_consistent(self) -> None:
        self.assert_rejected(lambda r: r["ml_evaluation"].update(prediction="ANOMALY"), "prediction")
        self.assert_rejected(lambda r: r["ml_evaluation"]["transitions"][0].update(step=2), "numbered")
        self.assert_rejected(lambda r: r["ml_evaluation"]["transitions"][0].update(probability=0.123456),
                             "path_log10_probability")

    def test_timing_check_must_be_consistent(self) -> None:
        self.assert_rejected(lambda r: r["timing_evaluation"].update(model_name="svm"), "model_name")
        self.assert_rejected(lambda r: r["timing_evaluation"].update(anomaly_score=5.0), "anomaly_score")
        self.assert_rejected(lambda r: r["timing_evaluation"]["checks"].pop(), "checks")
        self.assert_rejected(lambda r: r["timing_evaluation"]["checks"][0].update(
            is_flagged=not r["timing_evaluation"]["checks"][0]["is_flagged"]), "is_flagged")
        self.assert_rejected(lambda r: r["timing_evaluation"].pop("trend_z"), "trend_z")

    def test_unusual_flag_follows_the_percentile(self) -> None:
        good = {"name": "order_amount", "step": None, "unit": "currency", "value": 9000.0, "normal_low": 300.0,
                "normal_high": 4500.0, "percentile": 99.2, "is_unusual": True}
        TimingMeasurementSchema.model_validate(good)
        with self.assertRaises(ValidationError) as ctx:
            TimingMeasurementSchema.model_validate(dict(good, is_unusual=False))
        self.assertIn("is_unusual", str(ctx.exception))

    def test_created_at_must_be_utc(self) -> None:
        self.assert_rejected(lambda r: r.update(created_at=datetime(2026, 3, 1)), "created_at")

    def test_skipped_checks_must_be_explicit(self) -> None:
        def drop(r):
            del r["timing_skip_reason"]
        self.assert_rejected(drop, "timing_skip_reason")


if __name__ == "__main__":
    unittest.main()
