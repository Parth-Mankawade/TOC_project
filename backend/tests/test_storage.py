"""STORAGE checkpoint: saving and reading analysis records (backend/database/, backend/api/mapping.py).

Records are real analyzer results (fixture models) stored in an in-memory SQLite database.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import copy
import unittest

from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.mapping import record_to_response
from backend.core.errors import StorageError
from backend.database.connection import init_db
from backend.database.models import WorkflowAnalysis
from backend.database.repository import (
    create_analysis_batch,
    create_analysis_record,
    get_analyses_by_order_id,
    get_analysis_by_id,
    get_summary_metrics,
    list_analyses,
)
from backend.services.anomaly_service import AnomalyService, OrderInput
from backend.tests.fixtures import HAPPY, request_body, trained

T0 = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


class StorageTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        t = trained()
        cls.service = AnomalyService(t.paths.markov, t.paths.timing)
        normal = next(o for o in t.orders if o.category == "NORMAL" and o.events == HAPPY
                      and cls.service.analyze_order(OrderInput(**request_body(o)))["verdict"] == "NORMAL")
        body = request_body(normal)
        cls.timed = cls.service.analyze_order(OrderInput(**body))
        cls.timed_raw = body["event_sequence"]
        cls.violation = cls.service.analyze_order(OrderInput("CONFIRM_PAYMENT;SHIP_ITEMS", order_id="ORD-V"))
        cls.incomplete = cls.service.analyze_order(OrderInput(["CONFIRM_PAYMENT", "PACK_ITEMS"], order_id="ORD-I"))
        cls.anomaly = cls.service.analyze_order(OrderInput(["FAIL_PAYMENT", "RETRY_PAYMENT"] * 5 + HAPPY))

    def setUp(self) -> None:
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        init_db(self.engine)
        self.db = sessionmaker(bind=self.engine, expire_on_commit=False)()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()


class TestRoundTrip(StorageTestCase):

    def test_every_kind_of_result_is_stored_and_read_back_unchanged(self) -> None:
        cases = [(self.timed, self.timed_raw), (self.violation, "CONFIRM_PAYMENT;SHIP_ITEMS"),
                 (self.incomplete, ["CONFIRM_PAYMENT", "PACK_ITEMS"]),
                 (self.anomaly, ["FAIL_PAYMENT", "RETRY_PAYMENT"] * 5 + HAPPY)]
        for result, raw in cases:
            with self.subTest(verdict=result["verdict"]):
                response = record_to_response(create_analysis_record(self.db, result, raw)).model_dump()
                for key in ("order_id", "verdict", "is_anomaly", "anomaly_type", "summary", "diagnostics",
                            "ml_skip_reason", "timing_skip_reason", "event_timestamps", "order_amount"):
                    self.assertEqual(response[key], result[key], key)
                self.assertEqual(response["fsm_evaluation"]["final_state"], result["fsm_evaluation"]["final_state"])
                self.assertEqual(response["raw_input_payload"], raw)
                if result["ml_evaluation"] is not None:
                    self.assertEqual(response["ml_evaluation"]["transitions"], result["ml_evaluation"]["transitions"])
                if result["timing_evaluation"] is not None:
                    self.assertEqual(response["timing_evaluation"], result["timing_evaluation"])


class TestStorageErrors(StorageTestCase):

    def assert_storage_error(self, fragment: str, result, raw=HAPPY) -> None:
        with self.assertRaises(StorageError) as ctx:
            create_analysis_record(self.db, result, raw)
        self.assertTrue(str(ctx.exception).startswith("[STORAGE] expected "))
        self.assertIn(fragment, str(ctx.exception))

    def test_inconsistent_result_is_not_stored(self) -> None:
        bad = copy.deepcopy(self.anomaly)
        bad["ml_evaluation"]["prediction"] = "NORMAL"
        self.assert_storage_error("prediction", bad)
        self.assertEqual(list_analyses(self.db)[1], 0)

    def test_timing_result_without_its_inputs_is_not_stored(self) -> None:
        bad = dict(self.timed, event_timestamps=None, order_amount=None)
        self.assert_storage_error("event_timestamps", bad, self.timed_raw)

    def test_raw_input_must_be_text_or_a_list_of_text(self) -> None:
        self.assert_storage_error("raw input", self.violation, ("CONFIRM_PAYMENT",))

    def test_batch_is_all_or_nothing(self) -> None:
        bad = copy.deepcopy(self.anomaly)
        bad["verdict"] = "WEIRD"
        with self.assertRaises(StorageError):
            create_analysis_batch(self.db, [self.violation, bad], ["CONFIRM_PAYMENT;SHIP_ITEMS", HAPPY])
        self.assertEqual(list_analyses(self.db)[1], 0)
        with self.assertRaises(StorageError) as ctx:
            create_analysis_batch(self.db, [self.violation], [])
        self.assertIn("1 raw inputs", str(ctx.exception))

    def test_database_constraints_reject_rows_that_skip_validation(self) -> None:
        stored = create_analysis_record(self.db, self.violation, "CONFIRM_PAYMENT;SHIP_ITEMS")
        with self.assertRaises(IntegrityError):
            self.db.execute(text("UPDATE workflow_analyses SET verdict = 'MAYBE' WHERE id = :id"), {"id": stored.id})
        self.db.rollback()

    def test_a_corrupted_stored_record_fails_loudly_on_read(self) -> None:
        stored = create_analysis_record(self.db, self.anomaly, HAPPY)
        self.db.execute(text("UPDATE workflow_analyses SET path_log10_probability = -0.5 WHERE id = :id"),
                        {"id": stored.id})
        self.db.commit()
        self.db.expire_all()
        with self.assertRaises(StorageError) as ctx:
            record_to_response(get_analysis_by_id(self.db, stored.id))
        self.assertIn(f"record {stored.id}", str(ctx.exception))


class TestQueries(StorageTestCase):

    def setUp(self) -> None:
        super().setUp()
        results = [self.violation, self.incomplete, self.anomaly, self.timed, self.violation]
        raws = ["CONFIRM_PAYMENT;SHIP_ITEMS", ["CONFIRM_PAYMENT", "PACK_ITEMS"],
                ["FAIL_PAYMENT", "RETRY_PAYMENT"] * 5 + HAPPY, self.timed_raw, "CONFIRM_PAYMENT;SHIP_ITEMS"]
        for i, (result, raw) in enumerate(zip(results, raws)):
            create_analysis_record(self.db, result, raw, timestamp=T0 + timedelta(minutes=i))

    def test_list_filters_and_pages_newest_first(self) -> None:
        records, total = list_analyses(self.db, skip=0, limit=2)
        self.assertEqual(total, 5)
        self.assertEqual([r.id for r in records], [5, 4])
        violations, total = list_analyses(self.db, verdict="RULE_VIOLATION")
        self.assertEqual((total, {r.verdict for r in violations}), (2, {"RULE_VIOLATION"}))
        anomalies, total = list_analyses(self.db, is_anomaly=True)
        self.assertEqual(total, 3)

    def test_order_history_is_newest_first(self) -> None:
        history = get_analyses_by_order_id(self.db, "ORD-V")
        self.assertEqual([r.id for r in history], [5, 1])

    def test_summary_metrics(self) -> None:
        m = get_summary_metrics(self.db)
        self.assertEqual((m["total_analyses"], m["rule_violation_count"], m["statistical_anomaly_count"],
                          m["valid_incomplete_count"], m["normal_count"]), (5, 2, 1, 1, 1))
        self.assertEqual((m["total_anomalies"], m["anomaly_rate"]), (3, 0.6))
        self.assertEqual(m["unique_orders_count"], 3)

    def test_summary_of_an_empty_database_has_no_rate(self) -> None:
        self.db.execute(text("DELETE FROM workflow_analyses"))
        self.db.commit()
        m = get_summary_metrics(self.db)
        self.assertEqual((m["total_analyses"], m["anomaly_rate"]), (0, None))

    def test_bad_query_parameters(self) -> None:
        with self.assertRaises(StorageError):
            list_analyses(self.db, verdict="MAYBE")
        with self.assertRaises(StorageError):
            list_analyses(self.db, skip=-1)


class TestInitDb(unittest.TestCase):

    def test_creates_the_table_and_checks_it_afterwards(self) -> None:
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        init_db(engine)
        with engine.connect() as conn:
            self.assertEqual(conn.execute(text("SELECT COUNT(*) FROM workflow_analyses")).scalar(), 0)
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE workflow_analyses DROP COLUMN summary"))
        with self.assertRaises(StorageError) as ctx:
            init_db(engine)
        self.assertIn("summary", str(ctx.exception))
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
