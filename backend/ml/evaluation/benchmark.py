"""Multi-seed benchmark: the hybrid, the model it replaced, fair baselines, ablations
and the alternatives that were not adopted, each on SEEDS freshly generated datasets.

    .venv\\Scripts\\python.exe -m backend.ml.evaluation.benchmark

Writes backend/ml/evaluation/benchmark.json. The committed dataset and the saved
models are not read or written: every seed's orders are generated in memory.

How the comparison is kept fair:
  - Every method is fitted on the train split and thresholded on the normal
    validation orders. A method with one score flags hybrid.FALSE_ALARM_BUDGET of
    them. The two OR rules (previous_hybrid, zscore_or_markov) flag 1% per part,
    as the earlier model did; their actual false-alarm rates are reported.
  - Results are reported on the validation split and on the test split. Design
    decisions (which checks, which inputs, what to drop) were made on the
    validation columns only.
  - Stacking needs labels, so it splits the validation orders in half: one half
    fits the meta-model, the other sets its threshold. Its validation numbers
    are on that second half only. Leave-one-type-out refits it without one
    anomaly type and counts how many orders of that type it still catches.

Methods:
  previous_hybrid          the model this replaced: Markov chain OR the cluster detector
  tuned_range_rule         see evaluate.py
  zscore_or_markov         see evaluate.py
  hybrid                   the four checks under one threshold (what the app uses)
  hybrid_without_<check>   ablation: the hybrid with one check removed
  refund_inputs            not adopted: the cluster check with two more inputs
  retry_aware_path         not adopted: the path check from a retry-aware Markov chain
  detector_ensemble        not adopted: the cluster check from all three detectors
  stacking                 not adopted: logistic regression over the four levels
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from backend.core.errors import EvaluateError
from backend.ml import hybrid
from backend.ml.dataset.generator import ANOMALY_TYPES, SPLIT_SIZES, SplitSize, generate_orders
from backend.ml.dataset.loader import OrderRecord
from backend.ml.evaluation import experiments
from backend.ml.evaluation.evaluate import cutoff, fit_tuned_range_rule, metrics, sum_squared_z, timed
from backend.ml.markov.model import fit_markov_model
from backend.ml.timing.model import (
    DETECTOR_NAMES,
    SCALE_PERCENTILE,
    Scale,
    check_levels,
    choose_detector,
    fit_candidates,
    validation_report,
)

DEFAULT_BENCHMARK_PATH = Path(__file__).resolve().parent / "benchmark.json"
SEEDS = [42, 7, 1, 2, 3]
METRICS = ["f1", "precision", "recall", "false_alarm_rate", "roc_auc", "pr_auc"]
NOT_ADOPTED = ["refund_inputs", "retry_aware_path", "detector_ensemble", "stacking"]
STACK_SPLIT_SEED = 0
DECIMALS = 4


@dataclass(frozen=True)
class SeedData:
    train: List[OrderRecord]                 # normal training orders
    validation_normal: List[OrderRecord]
    validation_anomalies: List[OrderRecord]
    test: List[OrderRecord]                  # normal and anomalous test orders


def orders_for_seed(seed: int, sizes: Mapping[str, SplitSize] = SPLIT_SIZES) -> SeedData:
    """Generate one dataset in memory (amounts rounded to 2 decimals, as the CSV stores them)."""
    rows = [OrderRecord(f"ORD_{i:05d}", split, order.category, order.anomaly_type, order.events, stamps,
                        float(f"{order.amount:.2f}"))
            for i, (split, order, stamps) in enumerate(generate_orders(seed, sizes), start=1)]

    def pick(split: str, category: str) -> List[OrderRecord]:
        return [o for o in rows if o.split == split and o.category == category]

    return SeedData(pick("train", "NORMAL"), pick("validation", "NORMAL"), pick("validation", "STATISTICAL_ANOMALY"),
                    pick("test", "NORMAL") + pick("test", "STATISTICAL_ANOMALY"))


def _both(name: str, data: SeedData, validation_flags: Sequence[bool], test_flags: Sequence[bool],
          validation_scores: Optional[Sequence[float]] = None,
          test_scores: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    validation = data.validation_normal + data.validation_anomalies
    return {"validation": metrics(name, validation, validation_flags, validation_scores),
            "test": metrics(name, data.test, test_flags, test_scores)}


def run_seed(seed: int, sizes: Mapping[str, SplitSize] = SPLIT_SIZES,
             budget: float = hybrid.FALSE_ALARM_BUDGET) -> Dict[str, Any]:
    """Every method on one generated dataset: {method: {"validation": metrics, "test": metrics}}."""
    data = orders_for_seed(seed, sizes)
    n_normal = len(data.validation_normal)
    validation = data.validation_normal + data.validation_anomalies
    t_train, t_normal, t_anomalies = timed(data.train), timed(data.validation_normal), timed(data.validation_anomalies)
    t_validation, t_test = timed(validation), timed(data.test)
    own_share = 1.0 - SCALE_PERCENTILE / 100.0

    markov = fit_markov_model([o.events for o in data.train], [o.events for o in data.validation_normal])
    candidates = fit_candidates(t_train, t_normal)
    chosen = validation_report(candidates, t_normal, t_anomalies)["chosen"]
    timing = candidates[chosen]

    results: Dict[str, Any] = {}
    v_levels, t_levels = hybrid.levels(markov, timing, t_validation), hybrid.levels(markov, timing, t_test)

    def single_score(name: str, validation_scores: np.ndarray, test_scores: np.ndarray) -> None:
        """A method with one score: flag `budget` of the normal validation orders."""
        cut = cutoff(validation_scores[:n_normal], budget)
        results[name] = _both(name, data, validation_scores > cut, test_scores > cut, validation_scores, test_scores)

    def fused(name: str, validation_parts: Sequence[np.ndarray], test_parts: Sequence[np.ndarray]) -> None:
        single_score(name, np.max(validation_parts, axis=0), np.max(test_parts, axis=0))

    # The model this branch replaced: each part flags its own 1%, verdict = OR.
    results["previous_hybrid"] = _both("previous_hybrid", data,
                                       (v_levels["path"] > 1) | (v_levels["cluster"] > 1),
                                       (t_levels["path"] > 1) | (t_levels["cluster"] > 1))

    # Baselines.
    tuned = fit_tuned_range_rule(data.train)
    single_score("tuned_range_rule", np.asarray([tuned(o) for o in validation]),
                 np.asarray([tuned(o) for o in data.test]))
    z_validation, z_test = sum_squared_z(timing.stats, validation), sum_squared_z(timing.stats, data.test)
    z_cut = cutoff(z_validation[:n_normal], own_share)
    results["zscore_or_markov"] = _both("zscore_or_markov", data, (z_validation > z_cut) | (v_levels["path"] > 1),
                                        (z_test > z_cut) | (t_levels["path"] > 1))

    # The hybrid and its ablations.
    fused("hybrid", [v_levels[c] for c in hybrid.CHECKS], [t_levels[c] for c in hybrid.CHECKS])
    for dropped in hybrid.CHECKS:
        kept = [c for c in hybrid.CHECKS if c != dropped]
        fused(f"hybrid_without_{dropped}", [v_levels[c] for c in kept], [t_levels[c] for c in kept])

    # Not adopted: two more inputs for the cluster detector, chosen by the same validation rule.
    scorers = experiments.refund_input_detectors(timing.stats, t_train)
    raw_validation = {name: scorer(t_validation) for name, scorer in scorers.items()}
    f1_by_name = {}
    for name, raw in raw_validation.items():
        flags = raw > cutoff(raw[:n_normal], own_share)
        f1_by_name[name] = metrics(name, validation, flags)["f1"]
    refund_chosen = choose_detector(f1_by_name)
    refund_scale = Scale.fit(raw_validation[refund_chosen][:n_normal], "refund-input detector")
    others = [c for c in hybrid.CHECKS if c != "cluster"]
    fused("refund_inputs", [v_levels[c] for c in others] + [refund_scale.level(raw_validation[refund_chosen])],
          [t_levels[c] for c in others] + [refund_scale.level(scorers[refund_chosen](t_test))])

    # Not adopted: retry-aware Markov chain as the path check.
    log10_path = experiments.fit_retry_aware_path([o.events for o in data.train])
    surprise_validation = np.asarray([-log10_path(o.events) for o in validation])
    surprise_test = np.asarray([-log10_path(o.events) for o in data.test])
    retry_scale = Scale.fit(surprise_validation[:n_normal], "retry-aware path")
    others = [c for c in hybrid.CHECKS if c != "path"]
    fused("retry_aware_path", [v_levels[c] for c in others] + [retry_scale.level(surprise_validation)],
          [t_levels[c] for c in others] + [retry_scale.level(surprise_test)])

    # Not adopted: all three detectors, rank-averaged, as the cluster check.
    v_detectors = {name: check_levels(candidates[name], t_validation)["cluster"] for name in DETECTOR_NAMES}
    t_detectors = {name: check_levels(candidates[name], t_test)["cluster"] for name in DETECTOR_NAMES}
    normal_detectors = {name: values[:n_normal] for name, values in v_detectors.items()}
    ensemble_validation = experiments.rank_average(normal_detectors, v_detectors)
    ensemble_test = experiments.rank_average(normal_detectors, t_detectors)
    ensemble_scale = Scale.fit(ensemble_validation[:n_normal], "detector ensemble")
    others = [c for c in hybrid.CHECKS if c != "cluster"]
    fused("detector_ensemble", [v_levels[c] for c in others] + [ensemble_scale.level(ensemble_validation)],
          [t_levels[c] for c in others] + [ensemble_scale.level(ensemble_test)])

    results["stacking"], results["stacking_leave_one_type_out"], results["hybrid_on_stacking_half"] = _stacking(
        data, v_levels, t_levels, budget)
    results["chosen_detector"] = chosen
    results["refund_inputs_chosen_detector"] = refund_chosen
    return results


def _stacking(data: SeedData, v_levels: Dict[str, np.ndarray], t_levels: Dict[str, np.ndarray], budget: float):
    """Fit on one half of the validation orders, threshold on the other half's normal orders."""
    validation = data.validation_normal + data.validation_anomalies
    X_validation = np.column_stack([v_levels[c] for c in hybrid.CHECKS])
    X_test = np.column_stack([t_levels[c] for c in hybrid.CHECKS])
    labels = np.asarray([int(o.is_anomaly) for o in validation])
    types = np.asarray([o.anomaly_type for o in validation])

    rng = np.random.default_rng(STACK_SPLIT_SEED)
    fit_half = np.zeros(len(validation), dtype=bool)
    for kind in sorted(set(types.tolist())):                       # same share of every kind in each half
        index = np.flatnonzero(types == kind)
        fit_half[rng.permutation(index)[: len(index) // 2]] = True
    held = ~fit_half
    held_rows = [o for o, keep in zip(validation, held) if keep]
    held_normal = held & (labels == 0)

    def fit_and_score(mask: np.ndarray, name: str) -> Dict[str, Any]:
        score = experiments.fit_stack(X_validation[mask], labels[mask])
        cut = cutoff(score(X_validation[held_normal]), budget)
        held_scores, test_scores = score(X_validation[held]), score(X_test)
        return {"validation": metrics(name, held_rows, held_scores > cut, held_scores),
                "test": metrics(name, data.test, test_scores > cut, test_scores)}

    stacked = fit_and_score(fit_half, "stacking")
    left_out = {}
    for anomaly_type in ANOMALY_TYPES:
        result = fit_and_score(fit_half & (types != anomaly_type), f"stacking without {anomaly_type}")
        left_out[anomaly_type] = {split: result[split]["caught_by_type"][anomaly_type] for split in result}

    # The hybrid thresholded and scored on the same held-out half, so the validation numbers can be compared.
    level = np.max(X_validation, axis=1)
    cut = cutoff(level[held_normal], budget)
    hybrid_half = metrics("hybrid_on_stacking_half", held_rows, level[held] > cut, level[held])
    return stacked, left_out, hybrid_half


# ---------------------------------------------------------------------------
# Summary over seeds
# ---------------------------------------------------------------------------

def _mean_std(values: Sequence[Optional[float]]) -> Optional[Dict[str, float]]:
    if any(v is None for v in values):
        return None
    return {"mean": round(float(np.mean(values)), DECIMALS), "std": round(float(np.std(values)), DECIMALS)}


def _summarise(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean and standard deviation over seeds of one method's metrics on one split."""
    summary: Dict[str, Any] = {metric: _mean_std([run[metric] for run in runs]) for metric in METRICS}
    summary["f1_by_seed"] = [run["f1"] for run in runs]
    summary["caught_by_type"] = {
        t: {"mean": round(float(np.mean([run["caught_by_type"][t]["caught"] for run in runs])), 2),
            "total": runs[0]["caught_by_type"][t]["total"]}
        for t in ANOMALY_TYPES
    }
    return summary


def run_benchmark(seeds: Sequence[int] = SEEDS, sizes: Mapping[str, SplitSize] = SPLIT_SIZES,
                  progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Run every method on every seed and summarise.

    Raises:
        EvaluateError: If no seeds are given.
    """
    if not seeds:
        raise EvaluateError("at least one seed", list(seeds), "run_benchmark")
    per_seed = []
    for seed in seeds:
        if progress:
            progress(f"seed {seed}")
        per_seed.append(run_seed(seed, sizes))

    method_names = [name for name, value in per_seed[0].items() if isinstance(value, dict) and "test" in value]
    methods = {name: {split: _summarise([run[name][split] for run in per_seed]) for split in ("validation", "test")}
               for name in method_names}
    leave_one_out = {
        t: {split: {"mean_caught": round(float(np.mean([run["stacking_leave_one_type_out"][t][split]["caught"]
                                                         for run in per_seed])), 2),
                    "total": per_seed[0]["stacking_leave_one_type_out"][t][split]["total"]}
            for split in ("validation", "test")}
        for t in ANOMALY_TYPES
    }
    return {
        "seeds": list(seeds),
        "false_alarm_budget": hybrid.FALSE_ALARM_BUDGET,
        "orders_per_seed": {split: {"normal": sizes[split].normal, "per_anomaly_type": sizes[split].per_anomaly_type}
                            for split in ("train", "validation", "test")},
        "chosen_detector_by_seed": [run["chosen_detector"] for run in per_seed],
        "refund_inputs_chosen_detector_by_seed": [run["refund_inputs_chosen_detector"] for run in per_seed],
        "not_adopted": list(NOT_ADOPTED),
        "methods": methods,
        "hybrid_on_stacking_half": _summarise([run["hybrid_on_stacking_half"] for run in per_seed]),
        "stacking_leave_one_type_out": leave_one_out,
    }


if __name__ == "__main__":
    report = run_benchmark(progress=print)
    DEFAULT_BENCHMARK_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for split in ("validation", "test"):
        print(f"\n{split}: mean over seeds {report['seeds']}")
        print(f"{'method':<28}{'F1':>8}{'precision':>11}{'recall':>8}{'FPR':>8}{'PR-AUC':>8}")
        for name, method in report["methods"].items():
            m = method[split]
            pr_auc = "       -" if m["pr_auc"] is None else f"{m['pr_auc']['mean']:>8.3f}"
            print(f"{name:<28}{m['f1']['mean']:>8.3f}{m['precision']['mean']:>11.3f}{m['recall']['mean']:>8.3f}"
                  f"{m['false_alarm_rate']['mean']:>8.4f}{pr_auc}")
