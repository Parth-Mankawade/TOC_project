"""The hybrid model: the Markov chain's path check and the timing model's three
checks, combined into one score with one threshold (CALIBRATE checkpoint).

Every check puts an order on the same scale, its level (0 = a typical normal
order, 1 = the point only 1% of normal validation orders exceed):

    path       Markov chain: how rare is this order of events?
    cluster    detector on the timing/amount inputs: how far from typical normal orders?
    trend      were the steps consistently faster, or consistently slower, than normal?
    deviation  is any value, or are several, far outside its normal range?

    hybrid level = the highest of the four levels
    hybrid score = hybrid level - threshold          (> 0 means anomaly)

So an order is an anomaly when at least one check puts it above the threshold.
calibrate() picks the threshold so that FALSE_ALARM_BUDGET of the normal
validation orders are above it, and stores it in both models: the Markov model
keeps it as a path probability, the timing model as a level. The analyzer can
then score an order that has no timestamps with the Markov model alone, under
the same threshold.

Before this, the two models each flagged their own most unusual 1% and the
verdict was their OR; that gave no single score and no way to hold the total
false-alarm rate fixed. FALSE_ALARM_BUDGET is the share of normal validation
orders that rule flagged (1.45% to 1.75% over five seeds), so results stay comparable.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Sequence, Tuple

import numpy as np

from backend.core.errors import TrainError
from backend.ml.markov.model import MarkovPathModel, path_log10_probability, with_level_threshold
from backend.ml.timing.features import TimedOrder
from backend.ml.timing.model import CHECKS as TIMING_CHECKS
from backend.ml.timing.model import TimingModel, check_levels

CHECKS = ["path"] + TIMING_CHECKS
FALSE_ALARM_BUDGET = 0.015   # share of normal validation orders the hybrid flags


def levels(markov: MarkovPathModel, timing: TimingModel, orders: Sequence[TimedOrder]) -> Dict[str, np.ndarray]:
    """The level of every check in CHECKS for each valid, complete order.

    Raises:
        FeatureError: If any order is not FSM-valid and complete, or cannot be turned into inputs.
    """
    result = dict(check_levels(timing, orders))
    result["path"] = np.asarray([markov.level(path_log10_probability(markov, events)) for events, _, _ in orders],
                                dtype=np.float64)
    return {check: result[check] for check in CHECKS}


def hybrid_levels(markov: MarkovPathModel, timing: TimingModel, orders: Sequence[TimedOrder],
                  checks: Sequence[str] = CHECKS) -> np.ndarray:
    """The highest level over the given checks, for each order."""
    by_check = levels(markov, timing, orders)
    return np.max([by_check[check] for check in checks], axis=0)


def hybrid_scores(markov: MarkovPathModel, timing: TimingModel, orders: Sequence[TimedOrder]) -> np.ndarray:
    """hybrid level - threshold for each order; > 0 means anomaly."""
    return hybrid_levels(markov, timing, orders) - timing.threshold


def threshold_for(normal_levels: Sequence[float], budget: float) -> float:
    """The level that `budget` of the given normal orders exceed.

    Raises:
        TrainError: If the budget is not between 0 and 1, or the threshold would not be positive.
    """
    if not 0.0 < budget < 1.0:
        raise TrainError("a false-alarm budget between 0 and 1", budget, "threshold_for")
    threshold = float(np.percentile(normal_levels, 100.0 * (1.0 - budget)))
    if not threshold > 0:
        raise TrainError("a positive threshold level", threshold, f"budget {budget}")
    return threshold


def calibrate(markov: MarkovPathModel, timing: TimingModel, validation_normal_orders: Sequence[TimedOrder],
              budget: float = FALSE_ALARM_BUDGET) -> Tuple[MarkovPathModel, TimingModel]:
    """Copies of both models that share one threshold: the hybrid level that `budget` of
    the normal validation orders exceed.

    Raises:
        TrainError: Bad budget, or no positive threshold.
        FeatureError: If a validation order is not valid and complete.
    """
    threshold = threshold_for(hybrid_levels(markov, timing, validation_normal_orders), budget)
    traces = [events for events, _, _ in validation_normal_orders]
    return with_level_threshold(markov, threshold, traces), replace(timing, threshold=threshold)
