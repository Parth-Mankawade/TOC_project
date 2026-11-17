"""One command for the whole ML pipeline: GENERATE -> LOAD -> TRAIN -> CALIBRATE -> EVALUATE.

    .venv\\Scripts\\python.exe -m backend.ml.pipeline

Writes (default paths):
  backend/ml/dataset/data/orders_event_log.csv     the synthetic event log
  backend/ml/models/markov_chain.json              event-order model
  backend/ml/models/timing_model.joblib (+ .json)  timing-and-amount model chosen on the validation split
  backend/ml/evaluation/results.json               the test-split report the dashboard shows

The saved model files are loaded back from disk before evaluation, so the report
describes exactly the files the API will use. Any failure stops the run with a
named error from the checkpoint that failed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from backend.ml import hybrid
from backend.ml.dataset.generator import OUTPUT_CSV, RANDOM_SEED, SPLIT_SIZES, SplitSize, generate_dataset
from backend.ml.dataset.loader import load_orders, select
from backend.ml.evaluation.evaluate import DEFAULT_RESULTS_PATH, evaluate
from backend.ml.markov.model import DEFAULT_MODEL_PATH as DEFAULT_MARKOV_PATH
from backend.ml.markov.model import fit_markov_model, load_markov_model, save_markov_model
from backend.ml.timing.model import DEFAULT_MODEL_PATH as DEFAULT_TIMING_PATH
from backend.ml.timing.model import fit_candidates, load_timing_model, save_timing_model, validation_report


@dataclass(frozen=True)
class PipelinePaths:
    dataset: Path
    markov: Path
    timing: Path
    results: Path


DEFAULT_PATHS = PipelinePaths(OUTPUT_CSV, DEFAULT_MARKOV_PATH, DEFAULT_TIMING_PATH, DEFAULT_RESULTS_PATH)


def run_pipeline(paths: PipelinePaths = DEFAULT_PATHS, seed: int = RANDOM_SEED,
                 sizes: Mapping[str, SplitSize] = SPLIT_SIZES) -> Dict[str, Any]:
    """Generate the data, train and save both models, reload them, evaluate, and write the report."""
    generate_dataset(paths.dataset, seed=seed, sizes=sizes)
    orders = load_orders(paths.dataset)

    train = select(orders, "NORMAL", "train")
    val_normal = select(orders, "NORMAL", "validation")
    val_anomalies = select(orders, "STATISTICAL_ANOMALY", "validation")

    def timed(rows):
        return [(o.events, o.timestamps, o.amount) for o in rows]

    markov = fit_markov_model([o.events for o in train], [o.events for o in val_normal])
    candidates = fit_candidates(timed(train), timed(val_normal))
    selection = validation_report(candidates, timed(val_normal), timed(val_anomalies))
    markov, timing = hybrid.calibrate(markov, candidates[selection["chosen"]], timed(val_normal))
    save_markov_model(markov, paths.markov)
    save_timing_model(timing, paths.timing)

    markov = load_markov_model(paths.markov)
    candidates[selection["chosen"]] = load_timing_model(paths.timing)
    results = evaluate(orders, markov, candidates, selection)

    paths.results.parent.mkdir(parents=True, exist_ok=True)
    paths.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def format_summary(results: Dict[str, Any]) -> str:
    """Plain-text table of the test-split results."""
    lines = [
        f"Test split: {results['test']['models'][0]['normal_orders']} normal orders, "
        f"{results['test']['models'][0]['anomalous_orders']} anomalies "
        f"({results['dataset']['test_anomaly_share']:.1%}).",
        f"Timing model chosen on validation: {results['model_selection']['chosen']}",
        "",
        f"Hybrid threshold level {results['hybrid']['threshold_level']:.3f}: flags "
        f"{results['hybrid']['validation_flag_rate']:.2%} of normal validation orders.",
        "",
        f"{'model':<30}{'F1':>7}{'precision':>11}{'recall':>8}{'PR-AUC':>8}{'false alarms':>15}{'caught':>10}",
    ]
    for m in results["test"]["models"]:
        pr_auc = "      -" if m["pr_auc"] is None else f"{m['pr_auc']:>7.3f}"
        lines.append(f"{m['model']:<30}{m['f1']:>7.3f}{m['precision']:>11.3f}{m['recall']:>8.3f} {pr_auc}"
                     f"{m['false_alarms']:>8} / {m['normal_orders']:<5}{m['anomalies_caught']:>5} / {m['anomalous_orders']}")
    head = results["headline"]
    lines += ["", f"Headline F1: hybrid {head['hybrid']:.3f} vs zscore_or_markov {head['zscore_or_markov']:.3f} "
                  f"vs tuned_range_rule {head['tuned_range_rule']:.3f} vs always_normal {head['always_normal']:.3f}"]
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_summary(run_pipeline()))
