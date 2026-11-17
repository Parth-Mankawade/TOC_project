"""Tests that fail when the models get worse, or when the test split leaks into fitting.

Everything runs on the fixture data (tests/fixtures.py): real models, a smaller
generated dataset, a fixed seed. The limits are below what the models reach
today, with room for small changes; a model that scored like a random guess
(ROC-AUC 0.5) would fail every test in TestDetectorsBeatRandom.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import tempfile
from typing import List, Optional, Sequence
import unittest
from unittest import mock

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from backend.ml import hybrid
from backend.ml import pipeline
from backend.ml.dataset import generator as gen
from backend.ml.dataset.loader import load_orders
from backend.ml.evaluation.evaluate import timed
from backend.ml.markov.model import path_log10_probability
from backend.ml.timing.model import DETECTOR_NAMES, detector_scores, fit_candidates
from backend.tests.fixtures import FIXTURE_SIZES, trained

MIN_AUC = 0.8
# Markov chain against path frequencies it was not fitted on (paths seen at least MIN_PATH_COUNT times).
MIN_PATH_COUNT = 3
MIN_PATH_RANK_CORRELATION = 0.85    # trained chain: 0.95; chains with random probabilities: 0.09 to 0.66
MAX_PATH_LOG10_ERROR = 0.3          # trained chain: 0.16; chains with random probabilities: 0.48 to 0.97


class TestDetectorsBeatRandom(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.t = trained()
        orders = cls.t.orders
        cls.train = [o for o in orders if o.split == "train" and o.category == "NORMAL"]
        cls.normal = [o for o in orders if o.split == "validation" and o.category == "NORMAL"]
        cls.anomalies = [o for o in orders if o.split == "validation" and o.is_anomaly]
        cls.rows = cls.normal + cls.anomalies
        cls.levels = hybrid.levels(cls.t.markov, cls.t.timing, timed(cls.rows))

    def auc(self, scores: Sequence[float], types: Optional[List[str]] = None) -> float:
        """ROC-AUC on the validation split: normal orders against all anomalies, or only the given types."""
        keep = [not o.is_anomaly or types is None or o.anomaly_type in types for o in self.rows]
        labels = [int(o.is_anomaly) for o, k in zip(self.rows, keep) if k]
        return float(roc_auc_score(labels, [s for s, k in zip(scores, keep) if k]))

    def test_every_cluster_candidate(self) -> None:
        candidates = fit_candidates(timed(self.train), timed(self.normal))
        self.assertEqual(sorted(candidates), sorted(DETECTOR_NAMES))
        for name, model in candidates.items():
            self.assertGreater(self.auc(detector_scores(model, timed(self.rows))), MIN_AUC, name)

    def test_hybrid(self) -> None:
        self.assertGreater(self.auc(np.max(list(self.levels.values()), axis=0)), MIN_AUC)

    def test_path_check_on_the_event_order_anomalies(self) -> None:
        # The Markov chain sees only the order of events, so it is judged on the two types that differ there.
        self.assertGreater(self.auc(self.levels["path"], ["CARD_TESTING", "REPEATED_PAYMENT_FAILURE"]), MIN_AUC)

    def test_path_probabilities_match_how_often_each_path_occurs(self) -> None:
        """The chain predicts how often each order of events happens. Checked on the normal
        validation orders, which the probabilities were not counted from. The AUC test above
        cannot see a chain with wrong probabilities: any chain makes long failure loops rare."""
        counts = Counter(tuple(o.events) for o in self.normal)
        paths = [p for p, n in counts.items() if n >= MIN_PATH_COUNT]
        predicted = np.asarray([path_log10_probability(self.t.markov, p) for p in paths])
        observed = np.asarray([math.log10(counts[p] / len(self.normal)) for p in paths])
        self.assertGreaterEqual(len(paths), 5)
        self.assertGreater(spearmanr(predicted, observed)[0], MIN_PATH_RANK_CORRELATION)
        self.assertLess(float(np.mean(np.abs(predicted - observed))), MAX_PATH_LOG10_ERROR)

    def test_trend_check_on_fast_fulfilment(self) -> None:
        self.assertGreater(self.auc(self.levels["trend"], ["FAST_FULFILMENT"]), MIN_AUC)

    def test_deviation_check_on_stalled_delivery(self) -> None:
        self.assertGreater(self.auc(self.levels["deviation"], ["STALLED_DELIVERY"]), MIN_AUC)

    def test_trend_and_deviation_checks_on_all_anomalies(self) -> None:
        for check in ("trend", "deviation"):
            self.assertGreater(self.auc(self.levels[check]), MIN_AUC, check)


class TestHybridBeatsTheBaselines(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.models = {m["model"]: m for m in trained().results["test"]["models"]}

    def test_f1_above_every_baseline(self) -> None:
        hybrid_f1 = self.models["hybrid"]["f1"]
        for baseline in ("always_normal", "range_rule", "tuned_range_rule", "zscore_or_markov"):
            self.assertGreater(hybrid_f1, self.models[baseline]["f1"], baseline)

    def test_f1_above_each_of_its_checks_alone(self) -> None:
        chosen = trained().results["model_selection"]["chosen"]
        for part in ("markov_chain", f"timing_{chosen}", "trend_check", "deviation_check"):
            self.assertGreater(self.models["hybrid"]["f1"], self.models[part]["f1"], part)

    def test_pr_auc_above_the_tuned_range_rule(self) -> None:
        self.assertGreater(self.models["hybrid"]["pr_auc"], self.models["tuned_range_rule"]["pr_auc"])

    def test_false_alarm_rate_stays_near_the_budget(self) -> None:
        # The threshold is set on validation orders; on unseen test orders the rate may drift, but not double.
        self.assertLess(self.models["hybrid"]["false_alarm_rate"], 2 * hybrid.FALSE_ALARM_BUDGET)

    def test_more_than_half_of_the_anomalies_are_caught(self) -> None:
        self.assertGreater(self.models["hybrid"]["recall"], 0.5)


def _same(a, b) -> bool:
    """Equal, allowing the last digits of a float to differ (k-means sums in a varying order)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _keys(orders) -> set:
    """Identify timed orders by their timestamps (no two generated orders share a start second and steps)."""
    return {tuple(stamps) for _, stamps, _ in orders}


class TestNoTestSplitInFitting(unittest.TestCase):

    def test_fitting_functions_receive_only_train_and_validation_orders(self) -> None:
        seen = {"train": set(), "validation": set()}
        counts = {}
        real = {name: getattr(pipeline, name) for name in ("fit_markov_model", "fit_candidates", "validation_report")}
        real_calibrate = hybrid.calibrate

        def fit_markov_model(traces, validation_traces):
            counts["markov"] = (len(traces), len(validation_traces))
            return real["fit_markov_model"](traces, validation_traces)

        def fit_candidates(train, validation_normal):
            seen["train"] |= _keys(train)
            seen["validation"] |= _keys(validation_normal)
            return real["fit_candidates"](train, validation_normal)

        def validation_report(candidates, validation_normal, validation_anomalies):
            seen["validation"] |= _keys(validation_normal) | _keys(validation_anomalies)
            return real["validation_report"](candidates, validation_normal, validation_anomalies)

        def calibrate(markov, timing, validation_normal):
            seen["validation"] |= _keys(validation_normal)
            return real_calibrate(markov, timing, validation_normal)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(pipeline, "fit_markov_model", fit_markov_model), \
                mock.patch.object(pipeline, "fit_candidates", fit_candidates), \
                mock.patch.object(pipeline, "validation_report", validation_report), \
                mock.patch.object(pipeline.hybrid, "calibrate", calibrate):
            root = Path(tmp)
            paths = pipeline.PipelinePaths(root / "o.csv", root / "m.json", root / "t.joblib", root / "r.json")
            pipeline.run_pipeline(paths, sizes=FIXTURE_SIZES)
            orders = load_orders(paths.dataset)

        by_split = {split: {tuple(o.timestamps) for o in orders if o.split == split}
                    for split in ("train", "validation", "test")}
        self.assertTrue(seen["train"] and seen["validation"])
        self.assertTrue(seen["train"] <= by_split["train"])
        self.assertTrue(seen["validation"] <= by_split["validation"])
        self.assertFalse((seen["train"] | seen["validation"]) & by_split["test"])
        # The Markov chain gets event lists only, so check how many: the normal train and validation orders.
        self.assertEqual(counts["markov"], (FIXTURE_SIZES["train"].normal, FIXTURE_SIZES["validation"].normal))

    def test_saved_models_do_not_change_when_the_test_split_changes(self) -> None:
        """The generator draws train, then validation, then test orders, so a different test
        split leaves the other two as they were. If any test order reached fitting, selection
        or thresholds, the saved models would differ."""
        other_test = dict(FIXTURE_SIZES, test=gen.SplitSize(normal=400, per_anomaly_type=7, rule_violations=10))
        saved = []
        for sizes in (FIXTURE_SIZES, other_test):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = pipeline.PipelinePaths(root / "o.csv", root / "m.json", root / "t.joblib", root / "r.json")
                results = pipeline.run_pipeline(paths, sizes=sizes)
                saved.append({"markov": json.loads(paths.markov.read_text(encoding="utf-8")),
                              "timing": json.loads(paths.timing.with_suffix(".json").read_text(encoding="utf-8")),
                              "selection": results["model_selection"], "hybrid": results["hybrid"]})
        for part in saved[0]:
            self.assertTrue(_same(saved[0][part], saved[1][part]), part)
        # The check itself can tell models apart: a different seed gives different ones.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = pipeline.PipelinePaths(root / "o.csv", root / "m.json", root / "t.joblib", root / "r.json")
            pipeline.run_pipeline(paths, seed=7, sizes=FIXTURE_SIZES)
            self.assertFalse(_same(saved[0]["markov"], json.loads(paths.markov.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    unittest.main()
