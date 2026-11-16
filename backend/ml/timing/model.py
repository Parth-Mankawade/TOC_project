"""Timing-and-amount model (TRAIN / MODEL checkpoints): are this legal order's
times, amount and outcome unusual together?

The order of events is the Markov chain's job. This model reads the numbers in
backend/ml/timing/features.py (MODEL_INPUTS) and learns what normal orders look like.

Candidates (PS9 lists Isolation Forest and clustering):
  isolation_forest      IsolationForest, 200 trees: random splits isolate unusual
                        orders in fewer steps than ordinary ones
  kmeans                k-means clustering of normal orders (k picked by silhouette
                        score); an order far from every cluster centre is unusual
  local_outlier_factor  LocalOutlierFactor, 20 neighbours: an order in a sparser
                        region than its nearest normal orders is unusual

Inputs are standardized with the training means and standard deviations, so each
input counts on the same scale. Every candidate gives a raw score where higher
means more unusual. The threshold is the THRESHOLD_PERCENTILE-th percentile of
the raw scores of normal validation orders, so about 1 in 100 normal orders is
flagged.

    anomaly_score = raw score - threshold     (> 0 means anomaly)

Choosing the candidate the API uses (select_detector): highest F1 on the
validation split (normal and anomalous validation orders). If a model named in
PS9 is within NEAR_TIE_F1 of the best, it is preferred, in PS9_PREFERENCE order.
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
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "timing_model.joblib"

DETECTOR_NAMES = ["isolation_forest", "kmeans", "local_outlier_factor"]
PS9_PREFERENCE = ["isolation_forest", "kmeans"]  # named in PS9's list of ML techniques
NEAR_TIE_F1 = 0.02
THRESHOLD_PERCENTILE = 99.0   # flag the most unusual 1% of normal validation orders
MIN_VALIDATION_ORDERS = 100
RANDOM_STATE = 42

IF_TREES = 200
LOF_NEIGHBORS = 20
KMEANS_K_CANDIDATES = range(2, 9)
KMEANS_SILHOUETTE_SAMPLE = 2_000   # silhouette needs all pairwise distances, so use a sample
KMEANS_INIT_RUNS = 10


@dataclass
class TimingModel:
    detector_name: str
    detector: Any                 # fitted scikit-learn estimator
    scaler: StandardScaler
    stats: FeatureStats
    threshold: float
    detector_params: Dict[str, Any]
    training_sample_count: int
    validation_sample_count: int
    sklearn_version: str


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _matrix(stats: FeatureStats, orders: Sequence[TimedOrder]) -> np.ndarray:
    return np.asarray([list(order_features(stats, e, t, a).values()) for e, t, a in orders], dtype=np.float64)


def _training_matrix(stats: FeatureStats, orders: Sequence[TimedOrder], what: str) -> np.ndarray:
    try:
        return _matrix(stats, orders)
    except FeatureError as exc:
        raise TrainError("valid, complete normal orders", str(exc), what) from exc


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
    """Fit one candidate on normal training orders; set its threshold on normal validation orders.

    Raises:
        TrainError: Unknown detector, too few validation orders, or invalid orders.
    """
    if detector_name not in DETECTOR_NAMES:
        raise TrainError(f"a detector in {DETECTOR_NAMES}", detector_name, "detector_name")
    if len(validation_normal_orders) < MIN_VALIDATION_ORDERS:
        raise TrainError(f"at least {MIN_VALIDATION_ORDERS} normal validation orders",
                         len(validation_normal_orders), "fit_timing_model")
    if stats is None:
        stats = fit_feature_stats(train_orders)
    scaler = StandardScaler().fit(_training_matrix(stats, train_orders, "training orders"))
    detector, params = _fit_detector(detector_name,
                                     scaler.transform(_training_matrix(stats, train_orders, "training orders")))
    validation = scaler.transform(_training_matrix(stats, validation_normal_orders, "validation orders"))
    threshold = float(np.percentile(_raw_scores(detector_name, detector, validation), THRESHOLD_PERCENTILE))
    return TimingModel(detector_name, detector, scaler, stats, threshold, params,
                       len(train_orders), len(validation_normal_orders), sklearn.__version__)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def anomaly_scores(model: TimingModel, orders: Sequence[TimedOrder]) -> np.ndarray:
    """raw score - threshold for each order; > 0 means anomaly.

    Raises:
        FeatureError: If any order cannot be turned into model inputs.
    """
    X = model.scaler.transform(_matrix(model.stats, orders))
    return _raw_scores(model.detector_name, model.detector, X) - model.threshold


def score_order(model: TimingModel, events: Sequence[str], timestamps: Sequence[Any], amount: float) -> Dict[str, Any]:
    """Score one FSM-valid, complete order.

    Returns:
        prediction ("NORMAL"/"ANOMALY"), is_timing_anomaly, anomaly_score, model_name,
        inputs (the model inputs) and measurements (amount, failed payments and every
        step duration, each with its normal range and percentile among normal orders).

    Raises:
        FeatureError: Bad amount, timestamps or order.
    """
    inputs = order_features(model.stats, events, timestamps, amount)
    X = model.scaler.transform(np.asarray([list(inputs.values())], dtype=np.float64))
    score = float(_raw_scores(model.detector_name, model.detector, X)[0]) - model.threshold
    return {
        "prediction": "ANOMALY" if score > 0 else "NORMAL",
        "is_timing_anomaly": score > 0,
        "anomaly_score": score,
        "model_name": model.detector_name,
        "inputs": inputs,
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
        flags_normal = anomaly_scores(model, validation_normal_orders) > 0
        flags_anomaly = anomaly_scores(model, validation_anomalies) > 0
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
        "rule": f"highest validation F1; a model named in PS9 ({', '.join(PS9_PREFERENCE)}) "
                f"within {NEAR_TIE_F1} of the best is preferred",
        "threshold_percentile": THRESHOLD_PERCENTILE,
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
        "threshold": model.threshold,
        "threshold_percentile": THRESHOLD_PERCENTILE,
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
    return TimingModel(meta["model_name"], payload["detector"], payload["scaler"], stats, float(meta["threshold"]),
                       dict(meta["detector_params"]), int(meta["training_sample_count"]),
                       int(meta["validation_sample_count"]), meta["sklearn_version"])
