"""TRAIN / MODEL checkpoints for the timing-and-amount model (backend/ml/timing/model.py).

Candidates (Isolation Forest, k-means clustering, Local Outlier Factor) are fitted
on normal training orders. Every check (cluster, trend, deviation) is scaled on the
normal validation orders so that level 1 is the point 1% of them exceed.
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
from backend.ml.timing.features import MODEL_INPUTS, timing_summary
from backend.ml.timing.model import (
    CHECKS,
    DEFAULT_THRESHOLD,
    DETECTOR_NAMES,
    NEAR_TIE_F1,
    SCALE_PERCENTILE,
    Scale,
    anomaly_scores,
    check_levels,
    choose_detector,
    detector_scores,
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

    def test_every_check_puts_one_percent_of_validation_normals_above_level_one(self) -> None:
        self.assertEqual((SCALE_PERCENTILE, DEFAULT_THRESHOLD), (99.0, 1.0))
        for name, model in self.models.items():
            self.assertEqual(model.threshold, DEFAULT_THRESHOLD)
            levels = check_levels(model, self.val_normal)
            self.assertEqual(list(levels), CHECKS)
            for check in CHECKS:
                flagged = float(np.mean(levels[check] > 1))
                self.assertTrue(0.005 <= flagged <= 0.01, (name, check, flagged))
                self.assertAlmostEqual(float(np.median(levels[check])), 0.0, places=9, msg=(name, check))
            np.testing.assert_allclose(detector_scores(model, self.val_normal), levels["cluster"] - 1)

    def test_anomaly_score_is_the_highest_check_level_minus_the_threshold(self) -> None:
        model = self.models["kmeans"]
        levels = check_levels(model, self.val_anomalies)
        expected = np.max([levels[c] for c in CHECKS], axis=0) - model.threshold
        np.testing.assert_allclose(anomaly_scores(model, self.val_anomalies), expected)

    def test_trend_and_deviation_levels_come_from_the_z_scores(self) -> None:
        model = self.models["kmeans"]
        events, stamps, amount = self.val_anomalies[0]
        summary = timing_summary(model.stats, events, stamps, amount)
        levels = check_levels(model, [(events, stamps, amount)])
        for check in ("trend", "deviation"):
            scale = model.scales[check]
            self.assertAlmostEqual(float(levels[check][0]), (summary[check] - scale.median) / (scale.high - scale.median))

    def test_every_candidate_scores_anomalies_higher_than_normal_orders(self) -> None:
        for name, model in self.models.items():
            normal = detector_scores(model, self.val_normal)
            anomalous = detector_scores(model, self.val_anomalies)
            self.assertGreater(float(np.median(anomalous)), float(np.median(normal)), name)

    def test_scale(self) -> None:
        scale = Scale.fit(list(range(101)), "example")        # median 50, 99th percentile 99
        self.assertEqual((scale.median, scale.high), (50.0, 99.0))
        self.assertEqual((scale.level(50.0), scale.level(99.0), scale.level(148.0)), (0.0, 1.0, 2.0))
        with self.assertRaises(TrainError) as ctx:
            Scale.fit([3.0] * 200, "example")
        self.assertIn("example", str(ctx.exception))

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
        self.assertEqual([c["name"] for c in result["checks"]], CHECKS)
        self.assertEqual(result["anomaly_score"], max(c["score"] for c in result["checks"]))
        for c in result["checks"]:
            self.assertEqual(c["is_flagged"], c["score"] > 0)
        self.assertEqual(result["trend_z"], timing_summary(model.stats, events, stamps, amount)["trend_z"])


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

    def test_saved_before_the_checks_existed(self) -> None:
        save_timing_model(self.models["kmeans"], self.path)
        payload = joblib.load(self.path)
        del payload["metadata"]["checks"]
        joblib.dump(payload, self.path)
        self.assert_load_error("checks")


if __name__ == "__main__":
    unittest.main()
