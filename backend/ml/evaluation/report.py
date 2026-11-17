"""Write the result tables in the documents from results.json and benchmark.json.

    .venv\\Scripts\\python.exe -m backend.ml.evaluation.report

A document holds a table between two marker lines:

    <!-- generated:test-results:start -->
    ...
    <!-- generated:test-results:end -->

This command replaces what is between the markers of every block it knows, in
every file in DOCUMENTS. No number in those tables is typed by hand, and
backend/tests/test_docs.py fails when a document no longer matches the two
JSON files (re-run this command after the pipeline or the benchmark).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

from backend.core.errors import EvaluateError
from backend.ml.evaluation.benchmark import DEFAULT_BENCHMARK_PATH
from backend.ml.evaluation.evaluate import DEFAULT_RESULTS_PATH

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTS = [REPO_ROOT / "README.md", REPO_ROOT / "docs" / "VIVA_GUIDE.md", REPO_ROOT / "IMPROVEMENTS.md"]

TYPE_LABELS = {
    "CARD_TESTING": "Card testing",
    "REPEATED_PAYMENT_FAILURE": "Repeated failure",
    "FAST_FULFILMENT": "Fast fulfilment",
    "STALLED_DELIVERY": "Stalled delivery",
    "HIGH_VALUE_QUICK_REFUND": "High-value refund",
}
MODEL_LABELS = {
    "always_normal": 'Always "normal" (trivial baseline)',
    "range_rule": "Range rule, untuned (outside the training min-max)",
    "tuned_range_rule": "Range rule, tuned to the hybrid's false-alarm budget",
    "zscore_or_markov": "z-score sum OR Markov chain (no fitted detector)",
    "markov_chain": "Markov chain alone (path check)",
    "timing_isolation_forest": "Isolation Forest alone",
    "timing_kmeans": "k-means alone",
    "timing_local_outlier_factor": "Local Outlier Factor alone",
    "trend_check": "Trend check alone",
    "deviation_check": "Deviation check alone",
    "hybrid": "**Hybrid: the four checks, one threshold (the app)**",
    "random_forest": "Random Forest (supervised comparison)",
}
METHOD_LABELS = {
    "previous_hybrid": "Before: Markov chain OR k-means",
    "tuned_range_rule": "Baseline: tuned range rule",
    "zscore_or_markov": "Baseline: z-score sum OR Markov chain",
    "hybrid": "**After: hybrid (four checks, one threshold)**",
    "hybrid_without_path": "Hybrid without the path check",
    "hybrid_without_cluster": "Hybrid without the cluster check",
    "hybrid_without_trend": "Hybrid without the trend check",
    "hybrid_without_deviation": "Hybrid without the deviation check",
    "refund_inputs": "Two refund inputs for the cluster check",
    "retry_aware_path": "Retry-aware Markov chain as the path check",
    "detector_ensemble": "Ensemble of the three detectors as the cluster check",
    "stacking": "Stacking (logistic regression; validation = held-out half)",
}
MAIN_METHODS = ["previous_hybrid", "tuned_range_rule", "zscore_or_markov", "hybrid"]
ABLATIONS = ["hybrid", "hybrid_without_path", "hybrid_without_cluster", "hybrid_without_trend",
             "hybrid_without_deviation"]


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    align = [":---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _number(value: Optional[float]) -> str:
    return "–" if value is None else f"{value:.3f}"


def _mean_std(value: Optional[Dict[str, float]], digits: int = 3) -> str:
    return "–" if value is None else f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"


def _bold(text: str, on: bool) -> str:
    return f"**{text}**" if on else text


# ---------------------------------------------------------------------------
# Blocks from results.json (seed 42, test split)
# ---------------------------------------------------------------------------

def test_results(results: Dict[str, Any]) -> str:
    models = results["test"]["models"]
    header = ["Model", "F1", "Precision", "Recall", "PR-AUC", f"Caught / {models[0]['anomalous_orders']:,}",
              f"False alarms / {models[0]['normal_orders']:,}"]
    rows = []
    for m in models:
        strong = m["model"] == "hybrid"
        rows.append([MODEL_LABELS[m["model"]], _bold(_number(m["f1"]), strong), _number(m["precision"]),
                     _number(m["recall"]), _number(m["pr_auc"]), _bold(str(m["anomalies_caught"]), strong),
                     f"{m['false_alarms']} ({m['false_alarm_rate']:.1%})"])
    return _table(header, rows)


def caught_by_type(results: Dict[str, Any]) -> str:
    by_name = {m["model"]: m for m in results["test"]["models"]}
    chosen = f"timing_{results['model_selection']['chosen']}"
    columns = [("tuned_range_rule", "Tuned range rule"), ("zscore_or_markov", "z-score OR Markov"),
               ("markov_chain", "Path"), (chosen, "Cluster"), ("trend_check", "Trend"),
               ("deviation_check", "Deviation"), ("hybrid", "**Hybrid**"), ("random_forest", "Random Forest")]
    total = by_name["hybrid"]["caught_by_type"][results["dataset"]["anomaly_types"][0]]["total"]
    rows = [[TYPE_LABELS[t]] + [_bold(str(by_name[name]["caught_by_type"][t]["caught"]), name == "hybrid")
                                for name, _ in columns]
            for t in results["dataset"]["anomaly_types"]]
    return _table([f"Type (of {total} each)"] + [label for _, label in columns], rows)


# ---------------------------------------------------------------------------
# Blocks from benchmark.json (five seeds)
# ---------------------------------------------------------------------------

def _seed_table(benchmark: Dict[str, Any], names: Sequence[str], split: str) -> str:
    header = ["Method", "F1", "Precision", "Recall", "False-alarm rate", "ROC-AUC", "PR-AUC"]
    rows = []
    for name in names:
        m = benchmark["methods"][name][split]
        rows.append([METHOD_LABELS[name], _bold(_mean_std(m["f1"]), name == "hybrid"), _mean_std(m["precision"]),
                     _mean_std(m["recall"]), _mean_std(m["false_alarm_rate"], 4), _mean_std(m["roc_auc"]),
                     _mean_std(m["pr_auc"])])
    return _table(header, rows)


def _type_table(benchmark: Dict[str, Any], names: Sequence[str], split: str) -> str:
    types = list(TYPE_LABELS)
    total = benchmark["methods"][names[0]][split]["caught_by_type"][types[0]]["total"]
    rows = [[METHOD_LABELS[name]] + [f"{benchmark['methods'][name][split]['caught_by_type'][t]['mean']:.1f}"
                                     for t in types]
            for name in names]
    return _table([f"Method (mean caught of {total})"] + [TYPE_LABELS[t] for t in types], rows)


def _f1_by_seed(benchmark: Dict[str, Any], names: Sequence[str], split: str) -> str:
    rows = [[METHOD_LABELS[name]] + [f"{f1:.3f}" for f1 in benchmark["methods"][name][split]["f1_by_seed"]]
            for name in names]
    return _table(["Method"] + [f"seed {seed}" for seed in benchmark["seeds"]], rows)


def _decision_table(benchmark: Dict[str, Any], names: Sequence[str]) -> str:
    header = ["Variant", "Validation F1", "Validation PR-AUC", "Test F1", "Test PR-AUC"]
    rows = []
    for name in names:
        validation, test = benchmark["methods"][name]["validation"], benchmark["methods"][name]["test"]
        rows.append([METHOD_LABELS[name], _mean_std(validation["f1"]), _mean_std(validation["pr_auc"]),
                     _mean_std(test["f1"]), _mean_std(test["pr_auc"])])
    return _table(header, rows)


def _leave_one_type_out(benchmark: Dict[str, Any]) -> str:
    stacking, hybrid = benchmark["methods"]["stacking"]["test"], benchmark["methods"]["hybrid"]["test"]
    rows = []
    for t, label in TYPE_LABELS.items():
        left_out = benchmark["stacking_leave_one_type_out"][t]["test"]
        rows.append([label, f"{stacking['caught_by_type'][t]['mean']:.1f}", f"{left_out['mean_caught']:.1f}",
                     f"{hybrid['caught_by_type'][t]['mean']:.1f}"])
    total = hybrid["caught_by_type"][next(iter(TYPE_LABELS))]["total"]
    return _table([f"Type (mean caught of {total}, test split)", "Stacking, type seen in fitting",
                   "Stacking, type left out of fitting", "Hybrid (never sees labels)"], rows)


def blocks(results: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, str]:
    """Every generated block: {name: markdown}."""
    not_adopted = benchmark["not_adopted"]
    return {
        "test-results": test_results(results),
        "caught-by-type": caught_by_type(results),
        "seeds-test": _seed_table(benchmark, MAIN_METHODS, "test"),
        "seeds-validation": _seed_table(benchmark, MAIN_METHODS, "validation"),
        "seeds-test-by-type": _type_table(benchmark, MAIN_METHODS, "test"),
        "seeds-test-f1": _f1_by_seed(benchmark, MAIN_METHODS, "test"),
        "ablations": _decision_table(benchmark, ABLATIONS),
        "ablations-by-type": _type_table(benchmark, ABLATIONS, "test"),
        "not-adopted": _decision_table(benchmark, ["hybrid"] + not_adopted),
        "not-adopted-by-type": _type_table(benchmark, ["hybrid"] + not_adopted, "test"),
        "leave-one-type-out": _leave_one_type_out(benchmark),
    }


# ---------------------------------------------------------------------------
# Applying blocks to documents
# ---------------------------------------------------------------------------

def _pattern(name: str) -> "re.Pattern[str]":
    return re.compile(rf"(<!-- generated:{re.escape(name)}:start -->\n)(?:.*?\n)?(<!-- generated:{re.escape(name)}:end -->)",
                      re.DOTALL)


def block_names(text: str) -> List[str]:
    return re.findall(r"<!-- generated:([a-z0-9-]+):start -->", text)


def apply(text: str, generated: Dict[str, str], source: str = "document") -> str:
    """The text with every generated block replaced by its current content.

    Raises:
        EvaluateError: A marker names a block that does not exist, or has no end marker.
    """
    for name in block_names(text):
        if name not in generated:
            raise EvaluateError(f"a generated block in {sorted(generated)}", name, source)
        text, count = _pattern(name).subn(lambda match: match.group(1) + generated[name] + "\n" + match.group(2), text)
        if count != 1:
            raise EvaluateError(f"one start and one end marker for block '{name}'", f"{count} complete pairs", source)
    return text


def load_blocks(results_path: Path = DEFAULT_RESULTS_PATH, benchmark_path: Path = DEFAULT_BENCHMARK_PATH) -> Dict[str, str]:
    """Blocks from the two saved reports.

    Raises:
        EvaluateError: If either file is missing.
    """
    for path, command in ((results_path, "backend.ml.pipeline"), (benchmark_path, "backend.ml.evaluation.benchmark")):
        if not path.is_file():
            raise EvaluateError(f"a saved report (run: python -m {command})", "no file", str(path))
    return blocks(json.loads(results_path.read_text(encoding="utf-8")),
                  json.loads(benchmark_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    generated = load_blocks()
    for document in DOCUMENTS:
        before = document.read_text(encoding="utf-8")
        after = apply(before, generated, str(document))
        if after != before:
            document.write_text(after, encoding="utf-8", newline="\n")
        print(f"{document.relative_to(REPO_ROOT)}: {len(block_names(before))} blocks, "
              f"{'updated' if after != before else 'already current'}")
