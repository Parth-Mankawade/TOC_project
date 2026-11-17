"""The result tables in the documents must match the saved reports
(backend/ml/evaluation/report.py).

If this fails after the pipeline or the benchmark was re-run:
    .venv\\Scripts\\python.exe -m backend.ml.evaluation.report
"""

from __future__ import annotations

import json
from typing import Dict, List
import unittest

from backend.core.errors import EvaluateError
from backend.ml.evaluation import report
from backend.ml.evaluation.benchmark import DEFAULT_BENCHMARK_PATH
from backend.ml.evaluation.evaluate import DEFAULT_RESULTS_PATH
from backend.ml.markov.model import DEFAULT_MODEL_PATH as MARKOV_PATH
from backend.ml.markov.model import describe_probability, load_markov_model, score_trace

HAPPY = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]


def _flat(text: str) -> str:
    return " ".join(text.split())


def _minus(value: float) -> str:
    return f"{value:.2f}".replace("-", "−")


def written_facts(results: dict, benchmark: dict) -> Dict[str, List[str]]:
    """Numbers written by hand in the prose of the documents, each recomputed from the saved files.
    Keyed by document file name; every phrase must appear in that document (whitespace ignored)."""
    markov = load_markov_model(MARKOV_PATH)
    p = markov.probabilities
    section = results["hybrid"]
    selection = {n: c["f1"] for n, c in results["model_selection"]["candidates"].items()}
    recalls = [r["recall"] for r in results["separability"]["by_type"].values()]
    before = benchmark["methods"]["previous_hybrid"]
    flag_rates = before["validation"]["false_alarm_rate_by_seed"]
    methods = benchmark["methods"]
    val_f1 = {name: m["validation"]["f1"]["mean"] for name, m in methods.items()}
    gains = [a - b for a, b in zip(methods["hybrid"]["test"]["f1_by_seed"], before["test"]["f1_by_seed"])]
    caught = {name: {t: c["mean"] for t, c in m["test"]["caught_by_type"].items()} for name, m in methods.items()}
    left_out = benchmark["stacking_leave_one_type_out"]
    loop = p[("ORDER_CREATED", "FAIL_PAYMENT")] * p[("PAYMENT_FAILED", "RETRY_PAYMENT")]
    rates = section["validation_flag_rate_by_check"]
    viva = [
        f"level {section['threshold_level']:.3f}",
        f"(currently {section['threshold_level']:.3f})",
        f"paths rarer than {describe_probability(markov.threshold_log10_probability)}",
        f"{markov.validation_flag_rate:.2%} of normal validation orders are",
        f"(a typical path: {describe_probability(markov.median_log10_probability)})",
        f"(the rarest-1% point: {describe_probability(markov.rare_log10_probability)})",
        f"P(CONFIRM_PAYMENT | ORDER_CREATED) = {p[('ORDER_CREATED', 'CONFIRM_PAYMENT')]:.3f}",
        f"P(FAIL_PAYMENT | ORDER_CREATED) = {p[('ORDER_CREATED', 'FAIL_PAYMENT')]:.3f}",
        f"P(DELIVER_ITEMS | SHIPPED) = {p[('SHIPPED', 'DELIVER_ITEMS')]:.3f}",
        f"{p[('ORDER_CREATED', 'FAIL_PAYMENT')]:.3f} × {p[('PAYMENT_FAILED', 'RETRY_PAYMENT')]:.3f} ≈ {loop:.3f}",
        f"flag {rates['path']:.2%} (path), {rates['cluster']:.2%} (cluster), {rates['trend']:.2%} (trend) and "
        f"{rates['deviation']:.2%} (deviation)",
        f"| Isolation Forest | {selection['isolation_forest']:.3f} |",
        f"**{selection['kmeans']:.3f}** (chosen)",
        f"| Local Outlier Factor | {selection['local_outlier_factor']:.3f} |",
        f"({selection['kmeans']:.3f} vs {selection['isolation_forest']:.3f} vs {selection['local_outlier_factor']:.3f})",
        f"catches only {min(recalls):.0%}–{max(recalls):.0%}".replace("%–", "–"),
        f"({min(flag_rates):.2%} to {max(flag_rates):.2%} over five seeds)",
        f"current threshold, level {section['threshold_level']:.3f}",
    ]
    for failures in (2, 3, 4, 5):
        r = score_trace(markov, ["FAIL_PAYMENT", "RETRY_PAYMENT"] * failures + HAPPY)
        viva.append(f"| {failures} | {describe_probability(r['path_log10_probability'])} | {_minus(r['anomaly_score'])} | "
                    f"{'yes' if r['is_ml_anomaly'] else 'no'} |")
    improvements = [
        f"which came to {min(flag_rates):.2%}–{max(flag_rates):.2%} of normal validation orders",
        f"the cluster check adds {val_f1['hybrid'] - val_f1['hybrid_without_cluster']:.3f} F1 and the deviation "
        f"check {val_f1['hybrid'] - val_f1['hybrid_without_deviation']:.3f}",
        f"stalled delivery drops to {left_out['STALLED_DELIVERY']['test']['mean_caught']:.1f} of 50, against "
        f"{caught['hybrid']['STALLED_DELIVERY']:.1f} for the label-free hybrid",
        f"fast fulfilment drops from {caught['stacking']['FAST_FULFILMENT']:.1f} to "
        f"{left_out['FAST_FULFILMENT']['test']['mean_caught']:.1f}",
        f"by {sum(gains) / len(gains):.3f} on average (at least {min(gains):.3f})",
        f"fast fulfilment goes from {caught['previous_hybrid']['FAST_FULFILMENT']:.1f} to "
        f"{caught['hybrid']['FAST_FULFILMENT']:.1f} caught of 50",
        f"stalled delivery from {caught['previous_hybrid']['STALLED_DELIVERY']:.1f} to "
        f"{caught['hybrid']['STALLED_DELIVERY']:.1f}",
    ]
    readme = [f"at a {results['separability']['false_alarm_budget']:.0%} false-alarm rate no single measurement "
              f"catches more than {max(recalls):.0%}"]
    return {"VIVA_GUIDE.md": viva, "IMPROVEMENTS.md": improvements, "README.md": readme}


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

    def test_numbers_written_in_the_prose_match_the_saved_files(self) -> None:
        facts = written_facts(self.results, self.benchmark)
        for document in report.DOCUMENTS:
            text = _flat(document.read_text(encoding="utf-8"))
            for phrase in facts[document.name]:
                self.assertIn(_flat(phrase), text, document.name)

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
