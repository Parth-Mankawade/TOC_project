"""ANALYZE checkpoint: one order through the FSM and both ML models.

Order of checks:
1. FSM. An invalid order is a RULE_VIOLATION; a valid but unfinished one is
   VALID_INCOMPLETE. Neither model runs: the rules already decide these orders.
2. Valid, complete orders:
   - Markov chain: how likely is this event order under normal behaviour?
     Always runs.
   - Timing-and-amount model: are the times, the amount and the outcome
     unusual together? Runs when event timestamps and the order amount are given.
   - Flagged by either model -> STATISTICAL_ANOMALY, otherwise NORMAL.

Every result carries a one-line summary, diagnostics (severity, root cause, key
factors, recommended action) and the full output of every check that ran.
Models are requested from a ModelSource only when an order needs them.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

from backend.core.errors import AnalyzeError, InputError
from backend.fsm.parsing import parse_event_sequence, parse_timestamps
from backend.fsm.reachability import possible_outcomes
from backend.fsm.validator import validate_trace
from backend.ml.markov.model import MarkovPathModel, describe_probability, score_trace
from backend.ml.timing.features import format_duration
from backend.ml.timing.model import TimingModel, score_order

PATH_SKIPPED_RULE_VIOLATION = "Path check skipped: the order breaks a workflow rule (RULE_VIOLATION)."
PATH_SKIPPED_INCOMPLETE = "Path check skipped: the order has not finished yet (VALID_INCOMPLETE)."
TIMING_SKIPPED_RULE_VIOLATION = "Timing check skipped: the order breaks a workflow rule (RULE_VIOLATION)."
TIMING_SKIPPED_INCOMPLETE = "Timing check skipped: the order has not finished yet (VALID_INCOMPLETE)."
TIMING_SKIPPED_NO_DATA = "Timing check skipped: event timestamps and order amount were not provided."

# A kind of step is unusual when its probability, multiplied over every time it occurs
# in the order, is below 10%: one rare step (e.g. 0.03), or a common one repeated (0.12 x 5 times).
UNUSUAL_STEP_PROBABILITY = 0.1
CARD_TESTING_HINT_FAILURES = 2        # failed payments from which the card-testing hint is given
COMBINATION_MIN_DISTANCE = 30.0       # percentile points from the median for a value to count as contributing
MAX_COMBINATION_REASONS = 2           # values listed when only the combination is unusual

SEVERITY = {"RULE_VIOLATION": "CRITICAL", "STATISTICAL_ANOMALY": "WARNING",
            "VALID_INCOMPLETE": "INFO", "NORMAL": "INFO"}


class ModelSource(Protocol):
    def markov_model(self) -> MarkovPathModel: ...
    def timing_model(self) -> TimingModel: ...


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

_NAMES = {"order_amount": "Order amount", "payment_failures": "Failed payments"}
_DIRECTION = {"seconds": ("slower", "faster"), "currency": ("higher", "lower"), "count": ("more", "fewer")}


def _format_value(unit: str, value: float) -> str:
    if unit == "seconds":
        return format_duration(value)
    if unit == "currency":
        return f"{value:,.2f}"
    return f"{value:g}"


def describe_measurement(m: Dict[str, Any]) -> str:
    """e.g. 'PACK_ITEMS -> SHIP_ITEMS at step 3: 45 s, faster than 99.9% of normal orders (normal 1.2 h to 20.4 h)'."""
    unit = m["unit"]
    label = f"{m['name']} at step {m['step']}" if unit == "seconds" else _NAMES[m["name"]]
    more, less = _DIRECTION[unit]
    p = m["percentile"]
    side = f"{more} than {p:.1f}% of normal orders" if p >= 50 else f"{less} than {100 - p:.1f}% of normal orders"
    return (f"{label}: {_format_value(unit, m['value'])}, {side} "
            f"(normal {_format_value(unit, m['normal_low'])} to {_format_value(unit, m['normal_high'])})")


def timing_reasons(timing: Dict[str, Any]) -> List[str]:
    """Why the timing model flagged an order: every value outside the middle 95% of normal
    orders, or, when there is none, the values furthest from normal that make up the combination."""
    measurements = timing["measurements"]
    unusual = [m for m in measurements if m["is_unusual"]]
    if unusual:
        return [describe_measurement(m) for m in unusual]
    contributing = sorted((m for m in measurements if abs(m["percentile"] - 50.0) >= COMBINATION_MIN_DISTANCE),
                          key=lambda m: -abs(m["percentile"] - 50.0))[:MAX_COMBINATION_REASONS]
    return (["No single value is outside the middle 95% of normal orders; the combination is what is unusual:"]
            + [describe_measurement(m) for m in contributing])


def mark_unusual_steps(transitions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Copy of the path's transitions, each with is_unusual (see UNUSUAL_STEP_PROBABILITY)."""
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for t in transitions:
        grouped.setdefault((t["from_state"], t["event"]), []).append(t)
    unusual = {key for key, ts in grouped.items() if math.prod(t["probability"] for t in ts) < UNUSUAL_STEP_PROBABILITY}
    return [dict(t, is_unusual=(t["from_state"], t["event"]) in unusual) for t in transitions]


def _unusual_steps(transitions: Sequence[Dict[str, Any]]) -> List[str]:
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for t in transitions:
        if t["is_unusual"]:
            grouped.setdefault((t["from_state"], t["event"]), []).append(t)
    lines = []
    for (state, event), ts in grouped.items():
        p = ts[0]["probability"]
        together = f", {p ** len(ts):.2g} together" if len(ts) > 1 else ""
        lines.append(f"Unusual step {event} from {state}: {len(ts)} time(s) at step(s) "
                     f"{', '.join(str(t['step']) for t in ts)}; probability {p:.3f} each{together}")
    return lines


def _rule_violation_summary(error: Dict[str, Any]) -> str:
    step, event, state, kind = error["failed_step"], error["failed_event"], error["state_at_failure"], error["error_type"]
    if kind == "INVALID_TRANSITION":
        return (f"Rule violation at step {step}: '{event}' is not allowed in state '{state}'. "
                f"Allowed: {', '.join(error['allowed_events'])}.")
    if kind == "UNKNOWN_EVENT":
        return f"Rule violation at step {step}: '{event}' is not an event of this workflow."
    if kind == "TERMINAL_STATE_VIOLATION":
        return f"Rule violation at step {step}: '{event}' happened after the order had ended in '{state}'."
    raise AnalyzeError("an FSM error type INVALID_TRANSITION, UNKNOWN_EVENT or TERMINAL_STATE_VIOLATION",
                       kind, f"step {step}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def _fsm_evaluation(fsm: Dict[str, Any]) -> Dict[str, Any]:
    error = fsm["error"]
    return {
        "is_rule_valid": bool(fsm["is_valid"]),
        "is_complete": bool(fsm["is_complete"]),
        "final_state": str(fsm["final_state"]),
        "total_steps": int(fsm["total_steps"]),
        "successful_transitions": int(fsm["successful_transitions"]),
        "error": None if error is None else {
            "error_type": str(error["error_type"]),
            "failed_step": int(error["failed_step"]),
            "failed_event": str(error["failed_event"]),
            "state_at_failure": str(error["state_at_failure"]),
            "allowed_events": [str(e) for e in error["allowed_events"]],
            "message": str(error["message"]),
        },
    }


def _result(verdict: str, summary: str, fsm: Dict[str, Any], diagnostics: Dict[str, Any],
            ml: Optional[Dict[str, Any]], ml_skip: Optional[str],
            timing: Optional[Dict[str, Any]], timing_skip: Optional[str]) -> Dict[str, Any]:
    anomaly_type = verdict if verdict in ("RULE_VIOLATION", "STATISTICAL_ANOMALY") else None
    return {
        "verdict": verdict,
        "is_anomaly": anomaly_type is not None,
        "anomaly_type": anomaly_type,
        "summary": summary,
        "diagnostics": dict(diagnostics, severity=SEVERITY[verdict]),
        "fsm_evaluation": fsm,
        "ml_evaluation": ml,
        "ml_skip_reason": ml_skip,
        "timing_evaluation": timing,
        "timing_skip_reason": timing_skip,
    }


def check_timing_inputs(events: List[str], event_timestamps: Any, order_amount: Any) -> Optional[List[Any]]:
    """Parsed timestamps, or None when no timing data was sent. Raises InputError on bad timing inputs."""
    if (event_timestamps is None) != (order_amount is None):
        raise InputError("event_timestamps and order_amount together (both or neither)",
                         f"event_timestamps {'given' if event_timestamps is not None else 'missing'}, "
                         f"order_amount {'given' if order_amount is not None else 'missing'}", "request")
    if event_timestamps is None:
        return None
    stamps = parse_timestamps(event_timestamps)
    if len(stamps) != len(events):
        raise InputError(f"{len(events)} timestamps (one per event)", f"{len(stamps)} timestamps", "event_timestamps")
    if (isinstance(order_amount, bool) or not isinstance(order_amount, (int, float))
            or not math.isfinite(order_amount) or not order_amount > 0):
        raise InputError("a positive, finite order_amount", order_amount, "order_amount")
    return stamps


def analyze_trace(event_sequence: Union[str, Sequence[str]], event_timestamps: Optional[Sequence[Any]] = None,
                  order_amount: Optional[float] = None, *, models: ModelSource) -> Dict[str, Any]:
    """Analyze one order. Verdicts: RULE_VIOLATION, VALID_INCOMPLETE, STATISTICAL_ANOMALY, NORMAL.

    Raises:
        InputError: Bad event sequence, timestamps or amount, or only one of the two timing inputs.
        ModelLoadError: A model this order needs cannot be loaded (from the ModelSource).
    """
    events = parse_event_sequence(event_sequence)
    stamps = check_timing_inputs(events, event_timestamps, order_amount)
    fsm = _fsm_evaluation(validate_trace(events))

    if not fsm["is_rule_valid"]:
        error = fsm["error"]
        summary = _rule_violation_summary(error)
        allowed = (", ".join(error["allowed_events"]) if error["allowed_events"]
                   else "none: the order had already ended")
        diagnostics = {
            "root_cause": summary,
            "key_factors": [f"Error type: {error['error_type']}", f"Failed step: {error['failed_step']}",
                            f"Offending event: {error['failed_event']}", f"State at failure: {error['state_at_failure']}",
                            f"Allowed events there: {allowed}"],
            "recommended_action": f"Check why '{error['failed_event']}' was recorded at step {error['failed_step']}; "
                                  "the workflow does not allow it there.",
        }
        return _result("RULE_VIOLATION", summary, fsm, diagnostics, None, PATH_SKIPPED_RULE_VIOLATION,
                       None, TIMING_SKIPPED_RULE_VIOLATION)

    state = fsm["final_state"]
    if not fsm["is_complete"]:
        outcomes = [f"Possible outcome {end}: next {' -> '.join(path)}" for end, path in possible_outcomes(state).items()]
        diagnostics = {
            "root_cause": f"The order is legal so far and is still in progress in state '{state}'.",
            "key_factors": [f"Current state: {state}", f"Steps so far: {fsm['total_steps']}"] + outcomes,
            "recommended_action": f"No action: wait for the next event from state '{state}'.",
        }
        return _result("VALID_INCOMPLETE", f"In progress: legal so far, currently in state '{state}'.", fsm,
                       diagnostics, None, PATH_SKIPPED_INCOMPLETE, None, TIMING_SKIPPED_INCOMPLETE)

    markov = models.markov_model()
    path = score_trace(markov, events)
    ml = {k: path[k] for k in ("prediction", "is_ml_anomaly", "anomaly_score", "path_log10_probability")}
    ml["transitions"] = mark_unusual_steps(path["transitions"])
    timing = None
    timing_skip: Optional[str] = TIMING_SKIPPED_NO_DATA
    if stamps is not None:
        timing = score_order(models.timing_model(), events, stamps, float(order_amount))
        timing_skip = None

    frequency = describe_probability(path["path_log10_probability"])
    threshold = describe_probability(markov.threshold_log10_probability)
    path_factors = [f"Event order frequency under normal behaviour: {frequency}",
                    f"Path check flags orders rarer than {threshold} (the rarest 1% of normal paths)",
                    f"Path anomaly score: {path['anomaly_score']:.2f} (above 0 means flagged; +1 = 10x rarer)"]
    timing_factors = (["Timing/amount check: not run (no event timestamps and order amount provided)"]
                      if timing is None else
                      [f"Timing/amount model ({timing['model_name']}) score: {timing['anomaly_score']:.2f} "
                       "(above 0 means flagged)"])

    path_flagged = path["is_ml_anomaly"]
    timing_flagged = timing is not None and timing["is_timing_anomaly"]
    if not (path_flagged or timing_flagged):
        timing_note = ("its timing and amount are within normal behaviour" if timing is not None
                       else "its timing was not checked")
        summary = (f"Normal: the order finished in '{state}'; this event order occurs in {frequency} under "
                   f"normal behaviour, and {timing_note}.")
        diagnostics = {
            "root_cause": f"Legal, complete order ending in '{state}', not flagged by either model.",
            "key_factors": [f"Final state: {state}"] + path_factors + timing_factors,
            "recommended_action": "No action needed.",
        }
        return _result("NORMAL", summary, fsm, diagnostics, ml, None, timing, timing_skip)

    causes: List[str] = []
    actions: List[str] = []
    factors = list(path_factors)
    if path_flagged:
        causes.append(f"its event order occurs in only {frequency} under normal behaviour (flagged below {threshold})")
        steps = _unusual_steps(ml["transitions"])
        factors += steps
        actions.append("Review why the unusual steps happened." if steps
                       else "No single step is rare, but the whole path is; review the order end to end.")
        if events.count("FAIL_PAYMENT") >= CARD_TESTING_HINT_FAILURES:
            actions.append("Repeated payment failures can mean card testing (someone trying stolen card numbers) "
                           "or a payment gateway problem.")
    factors += timing_factors
    if timing_flagged:
        reasons = timing_reasons(timing)
        causes.append("its timing, amount and outcome are unusual together")
        factors += reasons
        actions.append("Check the values listed as unusual against the order's records.")
    summary = "Statistical anomaly: the order is legal and complete, but " + "; and ".join(causes) + "."
    diagnostics = {
        "root_cause": "The order follows every rule, but " + " and ".join(causes) + ".",
        "key_factors": factors,
        "recommended_action": " ".join(actions),
    }
    return _result("STATISTICAL_ANOMALY", summary, fsm, diagnostics, ml, None, timing, timing_skip)
