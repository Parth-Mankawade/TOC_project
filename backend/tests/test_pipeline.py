"""EVALUATE checkpoint and the one-command pipeline (backend/ml/pipeline.py, backend/ml/evaluation/evaluate.py).

Runs GENERATE -> LOAD -> TRAIN -> save -> reload -> EVALUATE on a smaller dataset in a
temporary folder, then checks the report. Nothing here depends on files saved earlier.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from backend.core.errors import EvaluateError
from backend.ml.dataset import generator as gen
from backend.ml.evaluation.evaluate import BASELINES, MODEL_ORDER, SEEN_TYPES
from backend.ml.markov.model import load_markov_model
from backend.ml.pipeline import PipelinePaths, format_summary, run_pipeline
from backend.ml.timing.model import DETECTOR_NAMES, load_timing_model

FIXTURE_SIZES = {
    "train": gen.SplitSize(normal=1_500, per_anomaly_type=30, rule_violations=0),
    "validation": gen.SplitSize(normal=600, per_anomaly_type=10, rule_violations=0),
    "test": gen.SplitSize(normal=1_500, per_anomaly_type=20, rule_violations=60),
}


def paths_in(folder: str) -> PipelinePaths:
    root = Path(folder)
    return PipelinePaths(dataset=root / "orders.csv", markov=root / "markov.json",
                         timing=root / "timing.joblib", results=root / "results.json")


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.paths = paths_in(cls._tmp.name)
        cls.results = run_pipeline(cls.paths, sizes=FIXTURE_SIZES)
        cls.models = {m["model"]: m for m in cls.results["test"]["models"]}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_writes_every_artifact_and_the_saved_models_load(self) -> None:
        for path in (self.paths.dataset, self.paths.markov, self.paths.timing, self.paths.results):
            self.assertTrue(path.is_file(), path)
        self.assertEqual(json.loads(self.paths.results.read_text(encoding="utf-8")), self.results)
        load_markov_model(self.paths.markov)
        timing = load_timing_model(self.paths.timing)
        self.assertEqual(timing.detector_name, self.results["model_selection"]["chosen"])

    def test_report_covers_every_model_and_baseline(self) -> None:
        self.assertEqual(list(self.models), MODEL_ORDER)
        self.assertTrue(set(BASELINES) <= set(self.models))
        for name in DETECTOR_NAMES:
            self.assertIn(f"timing_{name}", self.models)

    def test_counts_add_up_to_the_test_split(self) -> None:
        test = FIXTURE_SIZES["test"]
        for name, m in self.models.items():
            self.assertEqual(m["normal_orders"], test.normal, name)
            self.assertEqual(m["anomalous_orders"], test.per_anomaly_type * len(gen.ANOMALY_TYPES), name)
            self.assertEqual(sum(c["caught"] for c in m["caught_by_type"].values()), m["anomalies_caught"], name)
            self.assertEqual({t: c["total"] for t, c in m["caught_by_type"].items()},
                             {t: test.per_anomaly_type for t in gen.ANOMALY_TYPES}, name)

    def test_always_normal_baseline_catches_nothing(self) -> None:
        m = self.models["always_normal"]
        self.assertEqual((m["anomalies_caught"], m["false_alarms"], m["f1"]), (0, 0, 0.0))

    def test_scores_are_reported_with_both_auc_measures(self) -> None:
        for name, m in self.models.items():
            scored = name not in ("always_normal", "range_rule", "zscore_or_markov")
            for key in ("roc_auc", "pr_auc"):
                self.assertEqual(m[key] is not None, scored, (name, key))
                if scored:
                    self.assertTrue(0.0 <= m[key] <= 1.0, (name, key))

    def test_hybrid_section_describes_the_shared_threshold(self) -> None:
        section = self.results["hybrid"]
        self.assertEqual(section["checks"], ["path", "cluster", "trend", "deviation"])
        self.assertEqual(section["validation_normal_orders"], FIXTURE_SIZES["validation"].normal)
        self.assertAlmostEqual(section["validation_flag_rate"], section["false_alarm_budget"], places=3)
        self.assertEqual(section["threshold_level"], load_timing_model(self.paths.timing).threshold)
        # No single check may use more than the whole budget.
        for check, rate in section["validation_flag_rate_by_check"].items():
            self.assertLessEqual(rate, section["validation_flag_rate"], check)

    def test_random_forest_lists_what_it_was_trained_on(self) -> None:
        rf = self.models["random_forest"]
        self.assertEqual(rf["trained_on_types"], SEEN_TYPES)
        self.assertEqual(sorted(rf["never_seen_types"]), sorted(set(gen.ANOMALY_TYPES) - set(SEEN_TYPES)))

    def test_model_selection_used_the_validation_split(self) -> None:
        selection = self.results["model_selection"]
        self.assertEqual(selection["split"], "validation")
        self.assertEqual(set(selection["candidates"]), set(DETECTOR_NAMES))
        self.assertEqual(selection["normal_orders"], FIXTURE_SIZES["validation"].normal)

    def test_rule_violations_are_reported_separately(self) -> None:
        rv = self.results["rule_violations"]
        self.assertEqual(rv["orders"], FIXTURE_SIZES["test"].rule_violations)
        self.assertEqual(rv["rejected_by_fsm"], rv["orders"])
        self.assertEqual(sum(rv["by_type"].values()), rv["orders"])

    def test_dataset_and_separability_sections(self) -> None:
        self.assertEqual(self.results["dataset"]["test_anomaly_share"],
                         round(20 * 5 / (1_500 + 20 * 5), 4))
        self.assertEqual(set(self.results["separability"]["by_type"]), set(gen.ANOMALY_TYPES))

    def test_summary_names_the_headline_numbers(self) -> None:
        summary = format_summary(self.results)
        for name in ("always_normal", "tuned_range_rule", "zscore_or_markov", "hybrid", "F1", "PR-AUC"):
            self.assertIn(name, summary)

    def test_same_seed_gives_the_same_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            again = run_pipeline(paths_in(tmp), sizes=FIXTURE_SIZES)
        self.assertEqual(again, self.results)


class TestEvaluateErrors(unittest.TestCase):

    def test_a_test_split_without_anomalies_cannot_be_evaluated(self) -> None:
        sizes = dict(FIXTURE_SIZES, test=gen.SplitSize(normal=300, per_anomaly_type=0, rule_violations=0))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(EvaluateError) as ctx:
            run_pipeline(paths_in(tmp), sizes=sizes)
        self.assertTrue(str(ctx.exception).startswith("[EVALUATE] expected "))
        self.assertIn("anomal", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
