"""API checkpoint (backend/main.py, backend/api/routes.py): every endpoint, against a real app.

Each test starts the app with create_app(Settings(...)) on a temporary SQLite file and the
fixture models (tests/fixtures.py). Nothing is patched or faked.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.ml.dataset.loader import (
    DEMO_IN_PROGRESS,
    DEMO_NORMAL,
    DEMO_PER_ANOMALY_TYPE,
    DEMO_RULE_VIOLATIONS,
)
from backend.ml.dataset.generator import ANOMALY_TYPES, VIOLATION_TYPES
from backend.main import create_app, unexpected_error_response
from backend.tests.fixtures import HAPPY, request_body, trained

DEMO_TOTAL = DEMO_NORMAL + DEMO_IN_PROGRESS + DEMO_RULE_VIOLATIONS + DEMO_PER_ANOMALY_TYPE * len(ANOMALY_TYPES)


class ApiTestCase(unittest.TestCase):

    def settings(self, **overrides) -> Settings:
        t = trained()
        base = Settings(
            database_url=f"sqlite:///{(Path(self._tmp.name) / 'api.db').as_posix()}",
            markov_model_path=t.paths.markov,
            timing_model_path=t.paths.timing,
            dataset_path=t.paths.dataset,
            results_path=t.paths.results,
            frontend_dir=Path(self._tmp.name) / "no-dashboard",
            init_db_on_startup=True,
            cors_origins=[],
            database_echo=False,
        )
        return replace(base, **overrides)

    def start(self, **overrides) -> TestClient:
        client = TestClient(create_app(self.settings(**overrides)))
        client.__enter__()  # runs startup (creates the table) until tearDown
        self._clients.append(client)
        return client

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._clients: list[TestClient] = []
        self.client = self.start()

    def tearDown(self) -> None:
        for client in self._clients:
            client.__exit__(None, None, None)
        self._tmp.cleanup()

    def missing(self, name: str) -> Path:
        return Path(self._tmp.name) / "missing" / name

    def assert_error(self, response, status: int, checkpoint: str, *fragments: str) -> dict:
        self.assertEqual(response.status_code, status, response.text)
        error = response.json()["error"]
        self.assertEqual(error["checkpoint"], checkpoint, error)
        for fragment in fragments:
            self.assertIn(fragment, error["message"])
        return error


class TestHealth(ApiTestCase):

    def test_liveness(self) -> None:
        for path in ("/health", "/api/v1/health/live"):
            self.assertEqual(self.client.get(path).json(), {"status": "alive"})

    def test_readiness_reports_what_was_loaded(self) -> None:
        for path in ("/api/v1/health", "/api/v1/health/ready"):
            body = self.client.get(path).json()
            self.assertEqual((body["service_status"], body["database"]), ("OPERATIONAL", "ok"))
            self.assertEqual(body["timing_model"]["model_name"], trained().timing.detector_name)

    def test_readiness_with_missing_models_is_503_and_names_the_file(self) -> None:
        client = self.start(markov_model_path=self.missing("markov.json"))
        self.assert_error(client.get("/api/v1/health/ready"), 503, "MODEL", "markov.json")

    def test_readiness_without_the_table_is_503(self) -> None:
        other = Path(self._tmp.name) / "empty.db"
        client = self.start(init_db_on_startup=False, database_url=f"sqlite:///{other.as_posix()}")
        self.assert_error(client.get("/api/v1/health/ready"), 503, "STORAGE", "workflow_analyses")


class TestAnalyses(ApiTestCase):

    def post(self, body: dict):
        return self.client.post("/api/v1/analyses", json=body)

    def test_rule_violation_is_stored_and_can_be_fetched(self) -> None:
        created = self.post({"order_id": "A1", "event_sequence": "CONFIRM_PAYMENT;SHIP_ITEMS"})
        self.assertEqual(created.status_code, 201)
        body = created.json()
        self.assertEqual((body["verdict"], body["diagnostics"]["severity"]), ("RULE_VIOLATION", "CRITICAL"))
        fetched = self.client.get(f"/api/v1/analyses/{body['id']}")
        self.assertEqual(fetched.json(), body)

    def test_order_with_timing_data(self) -> None:
        order = next(o for o in trained().orders if o.split == "test" and o.category == "NORMAL")
        body = self.post(request_body(order)).json()
        self.assertEqual(body["timing_evaluation"]["model_name"], trained().timing.detector_name)
        self.assertEqual(body["event_timestamps"], request_body(order)["event_timestamps"])
        self.assertTrue(all("percentile" in m for m in body["timing_evaluation"]["measurements"]))
        self.assertTrue(all("is_unusual" in t for t in body["ml_evaluation"]["transitions"]))

    def test_bad_input_is_a_422_that_names_the_field(self) -> None:
        stamps = ["2026-03-01T09:00:00Z"] * 4
        error = self.assert_error(self.post({"event_sequence": HAPPY, "event_timestamps": stamps}),
                                  422, "INPUT", "together")
        self.assertTrue(error["message"].startswith("[INPUT] expected "))
        self.assert_error(self.post({"event_sequence": HAPPY, "event_timestamps": ["x"] * 4, "order_amount": 5.0}),
                          422, "INPUT", "event_timestamps[0]")
        typed = self.assert_error(self.post({"order_id": 7, "event_sequence": HAPPY}), 422, "INPUT", "order_id")
        self.assertIn("got 7", typed["message"])

    def test_complete_order_with_missing_models_is_503_but_rule_checks_still_work(self) -> None:
        client = self.start(markov_model_path=self.missing("markov.json"))
        self.assertEqual(client.post("/api/v1/analyses", json={"event_sequence": "SHIP_ITEMS"}).status_code, 201)
        self.assert_error(client.post("/api/v1/analyses", json={"event_sequence": HAPPY}), 503, "MODEL", "markov.json")

    def test_batch(self) -> None:
        response = self.client.post("/api/v1/analyses/batch", json={"traces": [
            "CONFIRM_PAYMENT;SHIP_ITEMS", HAPPY, {"order_id": "B1", "event_sequence": ["CONFIRM_PAYMENT", "PACK_ITEMS"]},
        ]})
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual([r["verdict"] for r in body["results"]], ["RULE_VIOLATION", "NORMAL", "VALID_INCOMPLETE"])
        self.assertEqual((body["summary"]["total_orders"], body["summary"]["total_anomalies"]), (3, 1))
        self.assert_error(self.client.post("/api/v1/analyses/batch", json={"traces": []}), 422, "INPUT", "traces")

    def test_list_filter_paginate_history_and_metrics(self) -> None:
        self.assertIsNone(self.client.get("/api/v1/metrics").json()["anomaly_rate"])
        for body in ({"order_id": "A1", "event_sequence": "SHIP_ITEMS"}, {"order_id": "A1", "event_sequence": HAPPY},
                     {"event_sequence": "CANCEL_ORDER;PACK_ITEMS"}):
            self.assertEqual(self.post(body).status_code, 201)
        page = self.client.get("/api/v1/analyses", params={"verdict": "RULE_VIOLATION", "limit": 1}).json()
        self.assertEqual((page["total"], len(page["items"])), (2, 1))
        history = self.client.get("/api/v1/orders/A1/analyses").json()
        self.assertEqual([a["verdict"] for a in history["analyses"]], ["NORMAL", "RULE_VIOLATION"])
        metrics = self.client.get("/api/v1/metrics").json()
        self.assertEqual((metrics["total_analyses"], metrics["rule_violation_count"]), (3, 2))
        self.assertAlmostEqual(metrics["anomaly_rate"], 2 / 3)
        self.assert_error(self.client.get("/api/v1/analyses", params={"verdict": "MAYBE"}), 422, "INPUT", "MAYBE")
        self.assert_error(self.client.get("/api/v1/analyses/999"), 404, "API", "999")


class TestReadOnly(ApiTestCase):

    def test_fsm_definition_and_reachability(self) -> None:
        body = self.client.get("/api/v1/fsm").json()
        self.assertEqual((len(body["states"]), len(body["events"]), len(body["transitions"])), (8, 8, 11))
        self.assertTrue(body["reachability"]["is_deadlock_free"])
        self.assertTrue(body["reachability"]["language_is_infinite"])

    def test_models_report_is_the_saved_files(self) -> None:
        body = self.client.get("/api/v1/models").json()
        t = trained()
        self.assertEqual(body["evaluation"], t.results)
        self.assertEqual(body["timing_model"]["model_name"], t.timing.detector_name)
        self.assertEqual(body["markov_chain"]["training_sample_count"], t.markov.training_sample_count)
        self.assertEqual(body["unusual_step_probability"], 0.1)

    def test_models_report_without_its_file_is_503_naming_it(self) -> None:
        client = self.start(results_path=self.missing("results.json"))
        self.assert_error(client.get("/api/v1/models"), 503, "EVALUATE", "results.json")

    def test_examples_are_real_test_orders_of_every_kind(self) -> None:
        examples = self.client.get("/api/v1/examples").json()["examples"]
        kinds = {e["anomaly_type"] for e in examples}
        self.assertTrue(set(ANOMALY_TYPES) | set(VIOLATION_TYPES) <= kinds)
        for e in examples:
            self.assertEqual(len(e["event_timestamps"]), len(e["event_sequence"]))
            self.assertGreater(e["order_amount"], 0)

    def test_demo_orders_and_loading_them(self) -> None:
        demo = self.client.get("/api/v1/demo").json()
        self.assertEqual(len(demo["orders"]), DEMO_TOTAL)
        self.assertEqual(sum(demo["composition"].values()), DEMO_TOTAL)
        summary = self.client.post("/api/v1/demo").json()
        self.assertEqual(summary["total_orders"], DEMO_TOTAL)
        self.assertEqual(summary["rule_violation_count"], DEMO_RULE_VIOLATIONS)
        self.assertEqual(summary["valid_incomplete_count"], DEMO_IN_PROGRESS)
        self.assertEqual(self.client.get("/api/v1/metrics").json()["total_analyses"], DEMO_TOTAL)

    def test_demo_without_the_dataset_is_503_naming_it(self) -> None:
        client = self.start(dataset_path=self.missing("orders.csv"))
        self.assert_error(client.get("/api/v1/demo"), 503, "LOAD", "orders.csv")


class TestErrorEnvelope(ApiTestCase):

    def test_unknown_path_and_wrong_method(self) -> None:
        self.assertEqual(self.assert_error(self.client.get("/api/v1/nope"), 404, "API")["code"], "NOT_FOUND")
        self.assertEqual(self.assert_error(self.client.delete("/api/v1/metrics"), 405, "API")["code"],
                         "METHOD_NOT_ALLOWED")

    def test_unexpected_errors_keep_their_message(self) -> None:
        response = unexpected_error_response(RuntimeError("disk full"))
        self.assertEqual(response.status_code, 500)
        error = json.loads(response.body)["error"]
        self.assertEqual(error["checkpoint"], "API")
        self.assertIn("RuntimeError: disk full", error["message"])


class TestDashboardMount(ApiTestCase):

    def test_built_dashboard_is_served_at_app(self) -> None:
        dist = Path(self._tmp.name) / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>dashboard</html>", encoding="utf-8")
        client = self.start(frontend_dir=dist)
        root = client.get("/", follow_redirects=False)
        self.assertEqual((root.status_code, root.headers["location"]), (307, "/app/"))
        self.assertIn("dashboard", client.get("/app/").text)

    def test_without_a_build_the_root_is_a_404(self) -> None:
        self.assert_error(self.client.get("/"), 404, "API")


if __name__ == "__main__":
    unittest.main()
