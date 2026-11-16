"""TRAIN / MODEL checkpoints for the timing-and-amount model (backend/ml/timing/model.py).

Candidates (Isolation Forest, k-means clustering, Local Outlier Factor) are fitted
on normal training orders; each threshold flags 1% of normal validation orders.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from backend.core.errors import ModelLoadError, TrainError
from backend.ml.dataset import generator as gen
from backend.ml.dataset.loader import load_orders, select
from backend.ml.timing.features import MODEL_INPUTS
from backend.ml.timing.model import (
    DETECTOR_NAMES,
    NEAR_TIE_F1,
    THRESHOLD_PERCENTILE,
    anomaly_scores,
    choose_detector,
    fit_timing_model,
    load_timing_model,
    save_timing_model,
    score_order,
    select_detector,
)


def timed(orders):
    return [(o.events, o.timestamps, o.amount) for o in orders]


class TimingData(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orders = load_orders(gen.generate_dataset(Path(tmp) / "o.csv"))
        cls.train = timed(select(orders, "NORMAL", "train"))
        cls.val_normal = timed(select(orders, "NORMAL", "validation"))
        cls.val_anomalies = timed(select(orders, "STATISTICAL_ANOMALY", "validation"))
        cls.models = {name: fit_timing_model(cls.train, cls.val_normal, name) for name in DETECTOR_NAMES}


class TestCandidates(TimingData):

    def test_every_candidate_flags_at_most_one_percent_of_validation_normals(self) -> None:
        self.assertEqual(THRESHOLD_PERCENTILE, 99.0)
        for name, model in self.models.items():
            flagged = float(np.mean(anomaly_scores(model, self.val_normal) > 0))
            self.assertTrue(0.005 <= flagged <= 0.01, (name, flagged))

    def test_every_candidate_scores_anomalies_higher_than_normal_orders(self) -> None:
        for name, model in self.models.items():
            normal = anomaly_scores(model, self.val_normal)
            anomalous = anomaly_scores(model, self.val_anomalies)
            self.assertGreater(float(np.median(anomalous)), float(np.median(normal)), name)

    def test_score_order_reports_inputs_measurements_and_a_consistent_verdict(self) -> None:
        model = self.models["isolation_forest"]
        events, stamps, amount = self.val_anomalies[0]
        result = score_order(model, events, stamps, amount)
        self.assertEqual(list(result["inputs"]), MODEL_INPUTS)
        self.assertEqual(result["model_name"], "isolation_forest")
        self.assertEqual(result["is_timing_anomaly"], result["anomaly_score"] > 0)
        self.assertEqual(result["prediction"], "ANOMALY" if result["is_timing_anomaly"] else "NORMAL")
        self.assertEqual(result["measurements"][0]["name"], "order_amount")
        self.assertAlmostEqual(result["anomaly_score"], float(anomaly_scores(model, [(events, stamps, amount)])[0]))


class TestSelection(TimingData):

    def test_select_detector_reports_every_candidate_and_picks_by_the_rule(self) -> None:
        model, report = select_detector(self.train, self.val_normal, self.val_anomalies)
        self.assertEqual(set(report["candidates"]), set(DETECTOR_NAMES))
        f1 = {name: c["f1"] for name, c in report["candidates"].items()}
        self.assertEqual(report["chosen"], choose_detector(f1))
        self.assertEqual(model.detector_name, report["chosen"])
        for c in report["candidates"].values():
            self.assertTrue(0.0 <= c["false_alarm_rate"] <= 0.01)

    def test_choose_detector_prefers_ps9_models_on_a_near_tie(self) -> None:
        self.assertEqual(NEAR_TIE_F1, 0.02)
        self.assertEqual(choose_detector({"isolation_forest": 0.70, "kmeans": 0.71, "local_outlier_factor": 0.715}),
                         "isolation_forest")
        self.assertEqual(choose_detector({"isolation_forest": 0.60, "kmeans": 0.69, "local_outlier_factor": 0.70}),
                         "kmeans")
        self.assertEqual(choose_detector({"isolation_forest": 0.60, "kmeans": 0.65, "local_outlier_factor": 0.70}),
                         "local_outlier_factor")


class TestTrainErrors(TimingData):

    def test_unknown_detector(self) -> None:
        with self.assertRaises(TrainError) as ctx:
            fit_timing_model(self.train, self.val_normal, "svm")
        self.assertIn("svm", str(ctx.exception))

    def test_too_few_validation_orders(self) -> None:
        with self.assertRaises(TrainError) as ctx:
            fit_timing_model(self.train, self.val_normal[:20], "kmeans")
        self.assertIn("20", str(ctx.exception))


class TestPersistence(TimingData):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "timing.joblib"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def assert_load_error(self, fragment: str) -> None:
        with self.assertRaises(ModelLoadError) as ctx:
            load_timing_model(self.path)
        self.assertTrue(str(ctx.exception).startswith("[MODEL] expected "))
        self.assertIn(fragment, str(ctx.exception))

    def test_round_trip_gives_identical_scores_and_a_readable_sidecar(self) -> None:
        for name, model in self.models.items():
            save_timing_model(model, self.path)
            loaded = load_timing_model(self.path)
            np.testing.assert_allclose(anomaly_scores(loaded, self.val_anomalies),
                                       anomaly_scores(model, self.val_anomalies))
            self.assertTrue(self.path.with_suffix(".json").is_file(), name)

    def test_missing_file(self) -> None:
        self.assert_load_error("timing.joblib")

    def test_not_a_saved_model(self) -> None:
        joblib.dump([1, 2, 3], self.path)
        self.assert_load_error("detector")

    def test_saved_with_another_scikit_learn_version(self) -> None:
        save_timing_model(self.models["kmeans"], self.path)
        payload = joblib.load(self.path)
        payload["metadata"]["sklearn_version"] = "0.0.1"
        joblib.dump(payload, self.path)
        self.assert_load_error("0.0.1")

    def test_saved_with_other_inputs(self) -> None:
        save_timing_model(self.models["kmeans"], self.path)
        payload = joblib.load(self.path)
        payload["metadata"]["model_inputs"] = ["amount_z"]
        joblib.dump(payload, self.path)
        self.assert_load_error("model_inputs")


if __name__ == "__main__":
    unittest.main()
