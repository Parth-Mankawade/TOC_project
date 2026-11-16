"""Real trained models for the analyzer, service, API and end-to-end tests.

trained() runs the whole pipeline (generate -> train -> save -> evaluate) once per
test session on FIXTURE_SIZES, in a temporary folder, and returns the files and
the loaded models. No test depends on files saved in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import tempfile
from typing import Any, Dict, List

from backend.ml.dataset import generator as gen
from backend.ml.dataset.generator import format_timestamp
from backend.ml.dataset.loader import OrderRecord, load_orders
from backend.ml.markov.model import MarkovPathModel, load_markov_model
from backend.ml.pipeline import PipelinePaths, run_pipeline
from backend.ml.timing.model import TimingModel, load_timing_model

FIXTURE_SIZES = {
    "train": gen.SplitSize(normal=1_500, per_anomaly_type=30, rule_violations=0),
    "validation": gen.SplitSize(normal=600, per_anomaly_type=10, rule_violations=0),
    "test": gen.SplitSize(normal=1_500, per_anomaly_type=20, rule_violations=60),
}

HAPPY = ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]


@dataclass(frozen=True)
class Trained:
    paths: PipelinePaths
    orders: List[OrderRecord]
    markov: MarkovPathModel
    timing: TimingModel
    results: Dict[str, Any]

    def markov_model(self) -> MarkovPathModel:   # ModelSource interface used by the analyzer
        return self.markov

    def timing_model(self) -> TimingModel:
        return self.timing


_TMP = tempfile.TemporaryDirectory(prefix="toc-fixture-")


@lru_cache(maxsize=1)
def trained() -> Trained:
    root = Path(_TMP.name)
    paths = PipelinePaths(dataset=root / "orders.csv", markov=root / "markov_chain.json",
                          timing=root / "timing_model.joblib", results=root / "results.json")
    results = run_pipeline(paths, sizes=FIXTURE_SIZES)
    return Trained(paths, load_orders(paths.dataset), load_markov_model(paths.markov),
                   load_timing_model(paths.timing), results)


def request_body(order: OrderRecord, with_timing: bool = True) -> Dict[str, Any]:
    """An API request for a dataset order."""
    body: Dict[str, Any] = {"order_id": order.trace_id, "event_sequence": list(order.events)}
    if with_timing:
        body["event_timestamps"] = [format_timestamp(t) for t in order.timestamps]
        body["order_amount"] = order.amount
    return body
