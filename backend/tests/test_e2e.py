"""End to end, from an empty folder: GENERATE -> LOAD -> TRAIN -> EVALUATE -> API, the way the
dashboard uses it. Nothing is patched; no file from the repository is read."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.main import create_app
from backend.ml.markov.model import load_markov_model
from backend.ml.pipeline import PipelinePaths, run_pipeline
from backend.ml.timing.model import load_timing_model
from backend.pipeline.analyzer import analyze_trace
from backend.tests.fixtures import FIXTURE_SIZES


class Loaded:
    def __init__(self, paths: PipelinePaths) -> None:
        self.markov, self.timing = load_markov_model(paths.markov), load_timing_model(paths.timing)

    def markov_model(self):
        return self.markov

    def timing_model(self):
        return self.timing


class TestEndToEnd(unittest.TestCase):

    def test_full_flow_from_an_empty_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = PipelinePaths(root / "orders.csv", root / "markov.json", root / "timing.joblib",
                                  root / "results.json")
            results = run_pipeline(paths, sizes=FIXTURE_SIZES)

            settings = Settings(database_url=f"sqlite:///{(root / 'e2e.db').as_posix()}",
                                markov_model_path=paths.markov, timing_model_path=paths.timing,
                                dataset_path=paths.dataset, results_path=paths.results,
                                frontend_dir=root / "no-dashboard", init_db_on_startup=True,
                                cors_origins=[], database_echo=False)
            with TestClient(create_app(settings)) as client:
                # 1. The backend is ready and reports the model the pipeline chose.
                ready = client.get("/api/v1/health/ready").json()
                self.assertEqual(ready["timing_model"]["model_name"], results["model_selection"]["chosen"])

                # 2. The report the dashboard shows is the pipeline's report, and ML beats both baselines.
                report = client.get("/api/v1/models").json()["evaluation"]
                self.assertEqual(report, results)
                head = report["headline"]
                self.assertGreater(head["hybrid"], head["range_rule"])
                self.assertGreater(head["hybrid"], head["always_normal"])

                # 3. Load the demo orders; each verdict fits how the order was generated.
                demo = client.get("/api/v1/demo").json()["orders"]
                stored = client.post("/api/v1/demo")
                self.assertEqual(stored.status_code, 201)
                page = client.get("/api/v1/analyses", params={"limit": 1000}).json()
                verdict_by_id = {a["order_id"]: a["verdict"] for a in page["items"]}
                expected = {"RULE_VIOLATION": {"RULE_VIOLATION"}, "IN_PROGRESS": {"VALID_INCOMPLETE"},
                            "NORMAL": {"NORMAL", "STATISTICAL_ANOMALY"},
                            "STATISTICAL_ANOMALY": {"NORMAL", "STATISTICAL_ANOMALY"}}
                for order in demo:
                    self.assertIn(verdict_by_id[order["order_id"]], expected[order["category"]], order["order_id"])

                # 4. Analyzing one example through the API gives what the models give directly.
                example = client.get("/api/v1/examples").json()["examples"][0]
                body = {k: example[k] for k in ("order_id", "event_sequence", "event_timestamps", "order_amount")}
                via_api = client.post("/api/v1/analyses", json=body).json()
                direct = analyze_trace(example["event_sequence"], example["event_timestamps"],
                                       example["order_amount"], models=Loaded(paths))
                self.assertEqual(via_api["verdict"], direct["verdict"])
                self.assertEqual(via_api["timing_evaluation"]["anomaly_score"],
                                 direct["timing_evaluation"]["anomaly_score"])


if __name__ == "__main__":
    unittest.main()
