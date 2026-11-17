"""Timing-and-amount model (TRAIN / MODEL checkpoints): are this legal order's
times, amount and outcome far from those of typical normal orders?

The order of events is the Markov chain's job. This model runs three checks
(CHECKS) on the numbers in backend/ml/timing/features.py:

  cluster    a detector fitted on MODEL_INPUTS of normal training orders (one of
             the candidates below): how far is the order from typical normal orders?
  trend      were the steps consistently faster, or consistently slower, than normal?
  deviation  is any value, or are several, far outside its normal range?

trend and deviation come straight from the z-scores (features.timing_summary).
They were added because the detector alone missed two kinds of order: a distance
over seven inputs is dominated by the inputs that are normal, so an order whose
every step is moderately fast, or whose single delivery step is slow, stayed
below the detector's threshold.

Candidates (PS9 lists Isolation Forest and clustering):
  isolation_forest      IsolationForest, 200 trees: random splits isolate unusual
                        orders in fewer steps than ordinary ones
  kmeans                k-means clustering of normal orders (k picked by silhouette
                        score); an order far from every cluster centre is unusual
  local_outlier_factor  LocalOutlierFactor, 20 neighbours: an order in a sparser
                        region than its nearest normal orders is unusual

Inputs are standardized with the training means and standard deviations, so each
input counts on the same scale. Every check gives a raw score where higher means
more unusual. Raw scores of different checks are not comparable, so each is put
on a common scale using the normal validation orders (Scale):

    level = (raw - median) / (99th percentile - median)

level 0 is a typical normal order; level 1 is the point only 1 in 100 normal
orders exceeds. A check flags an order when its level is above the model's
threshold:

    check score   = level - threshold            (> 0: this check flags the order)
    anomaly_score = highest check score          (> 0 means anomaly)

A freshly fitted model has threshold 1.0 (each check flags 1% of normal
validation orders on its own). backend/ml/hybrid.py then sets one threshold for
these three checks and the Markov chain's path check together, so that all four
checks combined flag a fixed share of normal validation orders.

Choosing the candidate the API uses (select_detector): highest F1 of the cluster
check on the validation split, each candidate flagging its most unusual 1% of
normal validation orders. If a model named in PS9 is within NEAR_TIE_F1 of the
best, it is preferred, in PS9_PREFERENCE order.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from backend.core.errors import FeatureError, ModelLoadError, TrainError
from backend.ml.timing.features import (
    MODEL_INPUTS,
    NORMAL_BAND,
    FeatureStats,
    LogNormalStats,
    TimedOrder,
    fit_feature_stats,
    measurements,
    order_features,
    timing_summary,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "timing_model.joblib"

DETECTOR_NAMES = ["isolation_forest", "kmeans", "local_outlier_factor"]
PS9_PREFERENCE = ["isolation_forest", "kmeans"]  # named in PS9's list of ML techniques
NEAR_TIE_F1 = 0.02
CHECKS = ["cluster", "trend", "deviation"]
SCALE_PERCENTILE = 99.0       # level 1 = the raw score only 1% of normal validation orders exceed
DEFAULT_THRESHOLD = 1.0       # each check flags its most unusual 1% until hybrid.calibrate() sets a shared threshold
MIN_VALIDATION_ORDERS = 100
RANDOM_STATE = 42

IF_TREES = 200
LOF_NEIGHBORS = 20
KMEANS_K_CANDIDATES = range(2, 9)
KMEANS_SILHOUETTE_SAMPLE = 2_000   # silhouette needs all pairwise distances, so use a sample
KMEANS_INIT_RUNS = 10


@dataclass(frozen=True)
class Scale:
    """Median and SCALE_PERCENTILE-th percentile of a raw score over normal validation orders."""

    median: float
    high: float

    @classmethod
    def fit(cls, raw: Sequence[float], what: str) -> "Scale":
        median, high = float(np.median(raw)), float(np.percentile(raw, SCALE_PERCENTILE))
        if not high > median:
            raise TrainError(f"a {SCALE_PERCENTILE:g}th percentile above the median",
                             f"median {median}, percentile {high}", f"{what} scores of normal validation orders")
        return cls(median, high)

    def level(self, raw: Any) -> Any:
        return (raw - self.median) / (self.high - self.median)


@dataclass
class TimingModel:
    detector_name: str
    detector: Any                 # fitted scikit-learn estimator
    scaler: StandardScaler
    stats: FeatureStats
    scales: Dict[str, Scale]      # one per check in CHECKS
    threshold: float              # level above which a check flags an order
    detector_params: Dict[str, Any]
    training_sample_count: int
    validation_sample_count: int
    sklearn_version: str


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _matrix(stats: FeatureStats, orders: Sequence[TimedOrder]) -> np.ndarray:
    rows = [list(order_features(stats, e, t, a).values()) for e, t, a in orders]
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(MODEL_INPUTS))


def _training_matrix(stats: FeatureStats, orders: Sequence[TimedOrder], what: str) -> np.ndarray:
    try:
        return _matrix(stats, orders)
    except FeatureError as exc:
        raise TrainError("valid, complete normal orders", str(exc), what) from exc


def _summary_scores(stats: FeatureStats, orders: Sequence[TimedOrder]) -> Dict[str, np.ndarray]:
    summaries = [timing_summary(stats, e, t, a) for e, t, a in orders]
    return {name: np.asarray([s[name] for s in summaries], dtype=np.float64) for name in ("trend", "deviation")}


def _raw_scores(name: str, detector: Any, X: np.ndarray) -> np.ndarray:
    """Higher = more unusual."""
    if name == "kmeans":
        return detector.transform(X).min(axis=1)       # distance to the nearest cluster centre
    return -detector.score_samples(X)                   # IF and LOF score normal orders higher


def _choose_k(X: np.ndarray) -> Tuple[int, Dict[int, float]]:
    rng = np.random.default_rng(RANDOM_STATE)
    sample = X if len(X) <= KMEANS_SILHOUETTE_SAMPLE else X[rng.choice(len(X), KMEANS_SILHOUETTE_SAMPLE, replace=False)]
    scores = {}
    for k in KMEANS_K_CANDIDATES:
        labels = KMeans(n_clusters=k, n_init=KMEANS_INIT_RUNS, random_state=RANDOM_STATE).fit_predict(sample)
        scores[k] = float(silhouette_score(sample, labels))
    return max(scores, key=lambda k: (scores[k], -k)), scores


def _fit_detector(name: str, X: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
    if name == "isolation_forest":
        return IsolationForest(n_estimators=IF_TREES, random_state=RANDOM_STATE).fit(X), {"n_estimators": IF_TREES}
    if name == "local_outlier_factor":
        return LocalOutlierFactor(n_neighbors=LOF_NEIGHBORS, novelty=True).fit(X), {"n_neighbors": LOF_NEIGHBORS}
    k, silhouettes = _choose_k(X)
    detector = KMeans(n_clusters=k, n_init=KMEANS_INIT_RUNS, random_state=RANDOM_STATE).fit(X)
    return detector, {"n_clusters": k, "silhouette_by_k": silhouettes}


def fit_timing_model(train_orders: Sequence[TimedOrder], validation_normal_orders: Sequence[TimedOrder],
                     detector_name: str, stats: Optional[FeatureStats] = None) -> TimingModel:
    """Fit one candidate on normal training orders; scale every check on normal validation orders.

    The returned model has threshold DEFAULT_THRESHOLD.

    Raises:
        TrainError: Unknown detector, too few validation orders, invalid orders, or a
            check whose scores on normal validation orders cannot be scaled.
    """
    if detector_name not in DETECTOR_NAMES:
        raise TrainError(f"a detector in {DETECTOR_NAMES}", detector_name, "detector_name")
    if len(validation_normal_orders) < MIN_VALIDATION_ORDERS:
        raise TrainError(f"at least {MIN_VALIDATION_ORDERS} normal validation orders",
                         len(validation_normal_orders), "fit_timing_model")
    if stats is None:
        stats = fit_feature_stats(train_orders)
    train_matrix = _training_matrix(stats, train_orders, "training orders")
    scaler = StandardScaler().fit(train_matrix)
    detector, params = _fit_detector(detector_name, scaler.transform(train_matrix))
    validation = scaler.transform(_training_matrix(stats, validation_normal_orders, "validation orders"))
    raw = dict(_summary_scores(stats, validation_normal_orders),
               cluster=_raw_scores(detector_name, detector, validation))
    scales = {check: Scale.fit(raw[check], check) for check in CHECKS}
    return TimingModel(detector_name, detector, scaler, stats, scales, DEFAULT_THRESHOLD, params,
                       len(train_orders), len(validation_normal_orders), sklearn.__version__)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def check_levels(model: TimingModel, orders: Sequence[TimedOrder]) -> Dict[str, np.ndarray]:
    """The level of every check in CHECKS for each order (0 = typical normal order, 1 = the
    point 1% of normal validation orders exceed).

    Raises:
        FeatureError: If any order cannot be turned into model inputs.
    """
    X = model.scaler.transform(_matrix(model.stats, orders))
    raw = dict(_summary_scores(model.stats, orders), cluster=_raw_scores(model.detector_name, model.detector, X))
    return {check: model.scales[check].level(raw[check]) for check in CHECKS}


def cluster_raw_scores(model: TimingModel, orders: Sequence[TimedOrder]) -> np.ndarray:
    """The detector's own score before scaling (higher = more unusual)."""
    return _raw_scores(model.detector_name, model.detector, model.scaler.transform(_matrix(model.stats, orders)))


def detector_scores(model: TimingModel, orders: Sequence[TimedOrder]) -> np.ndarray:
    """The cluster check on its own: level - 1, so > 0 means more unusual than 99% of
    normal validation orders. Used to compare candidates; independent of the threshold."""
    return check_levels(model, orders)["cluster"] - DEFAULT_THRESHOLD


def anomaly_scores(model: TimingModel, orders: Sequence[TimedOrder]) -> np.ndarray:
    """Highest check level - threshold for each order; > 0 means anomaly.

    Raises:
        FeatureError: If any order cannot be turned into model inputs.
    """
    levels = check_levels(model, orders)
    return np.max([levels[check] for check in CHECKS], axis=0) - model.threshold


def score_order(model: TimingModel, events: Sequence[str], timestamps: Sequence[Any], amount: float) -> Dict[str, Any]:
    """Score one FSM-valid, complete order.

    Returns:
        prediction ("NORMAL"/"ANOMALY"), is_timing_anomaly, anomaly_score (highest check
        score), model_name, checks (name, score and is_flagged for each check in CHECKS),
        trend_z (signed: negative = steps faster than normal overall), inputs (the
        detector's inputs) and measurements (amount, failed payments and every step
        duration, each with its normal range and percentile among normal orders).

    Raises:
        FeatureError: Bad amount, timestamps or order.
    """
    levels = check_levels(model, [(events, timestamps, amount)])
    checks = []
    for check in CHECKS:
        check_score = float(levels[check][0]) - model.threshold
        checks.append({"name": check, "score": check_score, "is_flagged": check_score > 0})
    score = max(c["score"] for c in checks)
    return {
        "prediction": "ANOMALY" if score > 0 else "NORMAL",
        "is_timing_anomaly": score > 0,
        "anomaly_score": score,
        "model_name": model.detector_name,
        "checks": checks,
        "trend_z": timing_summary(model.stats, events, timestamps, amount)["trend_z"],
        "inputs": order_features(model.stats, events, timestamps, amount),
        "measurements": measurements(model.stats, events, timestamps, amount),
    }


# ---------------------------------------------------------------------------
# Choosing a candidate on the validation split
# ---------------------------------------------------------------------------

def choose_detector(f1_by_name: Dict[str, float]) -> str:
    """Highest F1; a PS9-named model within NEAR_TIE_F1 of the best is preferred."""
    best = max(f1_by_name.values())
    for name in PS9_PREFERENCE:
        if name in f1_by_name and f1_by_name[name] >= best - NEAR_TIE_F1 - 1e-12:
            return name
    return max(sorted(f1_by_name), key=lambda n: f1_by_name[n])


def fit_candidates(train_orders: Sequence[TimedOrder],
                   validation_normal_orders: Sequence[TimedOrder]) -> Dict[str, TimingModel]:
    """Every candidate in DETECTOR_NAMES, fitted on the same statistics."""
    stats = fit_feature_stats(train_orders)
    return {name: fit_timing_model(train_orders, validation_normal_orders, name, stats) for name in DETECTOR_NAMES}


def validation_report(models: Dict[str, TimingModel], validation_normal_orders: Sequence[TimedOrder],
                      validation_anomalies: Sequence[TimedOrder]) -> Dict[str, Any]:
    """Compare fitted candidates on the validation split and name the chosen one.

    Raises:
        TrainError: No validation anomalies to compare on.
    """
    if not validation_anomalies:
        raise TrainError("validation anomalies to compare candidates on", 0, "validation_report")
    y_true = [0] * len(validation_normal_orders) + [1] * len(validation_anomalies)
    candidates: Dict[str, Dict[str, Any]] = {}
    for name, model in models.items():
        flags_normal = detector_scores(model, validation_normal_orders) > 0
        flags_anomaly = detector_scores(model, validation_anomalies) > 0
        y_pred = [int(f) for f in flags_normal] + [int(f) for f in flags_anomaly]
        candidates[name] = {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "false_alarm_rate": float(np.mean(flags_normal)),
            "params": {k: v for k, v in model.detector_params.items() if k != "silhouette_by_k"},
        }
    return {
        "split": "validation",
        "metric": "f1",
        "rule": f"highest validation F1 of the cluster check; a model named in PS9 "
                f"({', '.join(PS9_PREFERENCE)}) within {NEAR_TIE_F1} of the best is preferred",
        "threshold_percentile": SCALE_PERCENTILE,
        "normal_orders": len(validation_normal_orders),
        "anomalous_orders": len(validation_anomalies),
        "candidates": candidates,
        "chosen": choose_detector({name: c["f1"] for name, c in candidates.items()}),
    }


def select_detector(train_orders: Sequence[TimedOrder], validation_normal_orders: Sequence[TimedOrder],
                    validation_anomalies: Sequence[TimedOrder]) -> Tuple[TimingModel, Dict[str, Any]]:
    """Fit every candidate, compare them on the validation split, return the chosen one and the comparison."""
    models = fit_candidates(train_orders, validation_normal_orders)
    report = validation_report(models, validation_normal_orders, validation_anomalies)
    return models[report["chosen"]], report


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _stats_dict(stats: LogNormalStats) -> Dict[str, float]:
    return {
        "mean_ln": stats.mean_ln,
        "std_ln": stats.std_ln,
        "count": stats.count,
        "median": stats.quantile(50.0),
        "normal_low": stats.quantile(NORMAL_BAND[0]),
        "normal_high": stats.quantile(NORMAL_BAND[1]),
    }


def model_metadata(model: TimingModel) -> Dict[str, Any]:
    """Everything about the model except the fitted estimator, as plain JSON values."""
    return {
        "model_name": model.detector_name,
        "detector_params": model.detector_params,
        "model_inputs": list(MODEL_INPUTS),
        "checks": {check: {"median": model.scales[check].median, "high": model.scales[check].high}
                   for check in CHECKS},
        "scale_percentile": SCALE_PERCENTILE,
        "threshold": model.threshold,
        "normal_band_percentiles": list(NORMAL_BAND),
        "training_sample_count": model.training_sample_count,
        "validation_sample_count": model.validation_sample_count,
        "sklearn_version": model.sklearn_version,
        "amount": _stats_dict(model.stats.amount),
        "steps": {name: _stats_dict(st) for name, st in sorted(model.stats.steps.items())},
        "failure_counts": {str(k): n for k, n in sorted(model.stats.failure_counts.items())},
    }


def save_timing_model(model: TimingModel, path: Optional[Union[str, Path]] = None) -> Path:
    """Save the estimator and scaler with joblib, and the metadata as readable JSON next to it."""
    target = Path(path) if path else DEFAULT_MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = model_metadata(model)
    joblib.dump({"detector": model.detector, "scaler": model.scaler, "metadata": metadata}, target)
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return target


def _stats_from(d: Dict[str, Any]) -> LogNormalStats:
    return LogNormalStats(float(d["mean_ln"]), float(d["std_ln"]), int(d["count"]))


def load_timing_model(path: Optional[Union[str, Path]] = None) -> TimingModel:
    """Load a saved model and check it matches this code and the installed scikit-learn.

    Raises:
        ModelLoadError: Missing or unreadable file, wrong contents, different inputs,
            or a different scikit-learn version.
    """
    target = Path(path) if path else DEFAULT_MODEL_PATH
    if not target.is_file():
        raise ModelLoadError("a saved timing model (run: python -m backend.ml.pipeline)", "no file", str(target))
    try:
        payload = joblib.load(target)
    except Exception as exc:  # any unpickling failure: add the file name and stop
        raise ModelLoadError("a joblib file", f"{type(exc).__name__}: {exc}", str(target)) from exc
    if not isinstance(payload, dict) or set(payload) != {"detector", "scaler", "metadata"}:
        shape = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ModelLoadError("a dict with keys detector, scaler, metadata", shape, str(target))
    meta = payload["metadata"]
    if meta["model_inputs"] != MODEL_INPUTS:
        raise ModelLoadError(f"model_inputs {MODEL_INPUTS}", meta["model_inputs"], str(target))
    if meta["sklearn_version"] != sklearn.__version__:
        raise ModelLoadError(f"scikit-learn {sklearn.__version__} (pinned in requirements.txt)",
                             meta["sklearn_version"], str(target))
    if meta["model_name"] not in DETECTOR_NAMES:
        raise ModelLoadError(f"a detector in {DETECTOR_NAMES}", meta["model_name"], str(target))
    stats = FeatureStats(
        amount=_stats_from(meta["amount"]),
        steps={name: _stats_from(d) for name, d in meta["steps"].items()},
        failure_counts={int(k): int(n) for k, n in meta["failure_counts"].items()},
        training_sample_count=int(meta["training_sample_count"]),
    )
    if sorted(meta.get("checks", {})) != sorted(CHECKS):
        raise ModelLoadError(f"scales for the checks {CHECKS} (an older model file? retrain with "
                             "python -m backend.ml.pipeline)", sorted(meta.get("checks", {})), str(target))
    scales = {check: Scale(float(meta["checks"][check]["median"]), float(meta["checks"][check]["high"]))
              for check in CHECKS}
    return TimingModel(meta["model_name"], payload["detector"], payload["scaler"], stats, scales,
                       float(meta["threshold"]), dict(meta["detector_params"]), int(meta["training_sample_count"]),
                       int(meta["validation_sample_count"]), meta["sklearn_version"])
