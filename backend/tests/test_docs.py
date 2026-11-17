"""The result tables in the documents must match the saved reports
(backend/ml/evaluation/report.py).

If this fails after the pipeline or the benchmark was re-run:
    .venv\\Scripts\\python.exe -m backend.ml.evaluation.report
"""

from __future__ import annotations

import json
import unittest

from backend.core.errors import EvaluateError
from backend.ml.evaluation import report
from backend.ml.evaluation.benchmark import DEFAULT_BENCHMARK_PATH
from backend.ml.evaluation.evaluate import DEFAULT_RESULTS_PATH


class TestDocumentsMatchTheReports(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = json.loads(DEFAULT_RESULTS_PATH.read_text(encoding="utf-8"))
        cls.benchmark = json.loads(DEFAULT_BENCHMARK_PATH.read_text(encoding="utf-8"))
        cls.blocks = report.blocks(cls.results, cls.benchmark)

    def test_every_document_is_current(self) -> None:
        for document in report.DOCUMENTS:
            text = document.read_text(encoding="utf-8")
            self.assertGreaterEqual(len(report.block_names(text)), 3, document.name)
            self.assertEqual(report.apply(text, self.blocks, document.name), text,
                             f"{document.name} is out of date: run python -m backend.ml.evaluation.report")

    def test_readme_shows_the_headline_numbers_of_both_reports(self) -> None:
        readme = report.DOCUMENTS[0].read_text(encoding="utf-8")
        hybrid = next(m for m in self.results["test"]["models"] if m["model"] == "hybrid")
        self.assertIn(f"**{hybrid['f1']:.3f}**", readme)
        seeds = self.benchmark["methods"]["hybrid"]["test"]["f1"]
        self.assertIn(f"**{seeds['mean']:.3f} ± {seeds['std']:.3f}**", readme)

    def test_the_two_reports_describe_the_same_model(self) -> None:
        """Seed 42 is in both: the benchmark's first seed must give the pipeline's hybrid F1."""
        self.assertEqual(self.benchmark["seeds"][0], 42)
        hybrid = next(m for m in self.results["test"]["models"] if m["model"] == "hybrid")
        self.assertEqual(self.benchmark["methods"]["hybrid"]["test"]["f1_by_seed"][0], hybrid["f1"])
        self.assertEqual(self.benchmark["false_alarm_budget"], self.results["hybrid"]["false_alarm_budget"])

    def test_a_changed_number_is_detected(self) -> None:
        changed = json.loads(json.dumps(self.results))
        next(m for m in changed["test"]["models"] if m["model"] == "hybrid")["f1"] += 0.001
        readme = report.DOCUMENTS[0].read_text(encoding="utf-8")
        self.assertNotEqual(report.apply(readme, report.blocks(changed, self.benchmark)), readme)


class TestApply(unittest.TestCase):

    def test_replaces_only_what_is_between_the_markers(self) -> None:
        text = "before\n<!-- generated:a:start -->\nold\n<!-- generated:a:end -->\nafter\n"
        self.assertEqual(report.apply(text, {"a": "new\nrows"}),
                         "before\n<!-- generated:a:start -->\nnew\nrows\n<!-- generated:a:end -->\nafter\n")
        self.assertEqual(report.block_names(text), ["a"])

    def test_unknown_or_unclosed_blocks_are_errors(self) -> None:
        with self.assertRaises(EvaluateError) as ctx:
            report.apply("<!-- generated:nope:start -->\n<!-- generated:nope:end -->", {"a": "x"})
        self.assertIn("nope", str(ctx.exception))
        with self.assertRaises(EvaluateError):
            report.apply("<!-- generated:a:start -->\nno end marker", {"a": "x"})


if __name__ == "__main__":
    unittest.main()
