"""Pydantic v2 schemas for FSM, ML, and operational diagnostics results.

Defines schemas for deterministic automata validation, Markov chain path
scoring, and combined workflow analysis audit records.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.ml.timing.features import MODEL_INPUTS as TIMING_MODEL_INPUTS
from backend.ml.timing.features import NORMAL_BAND
from backend.ml.timing.model import CHECKS as TIMING_CHECKS
from backend.ml.timing.model import DETECTOR_NAMES

# Supported domain taxonomies
VerdictType = Literal["NORMAL", "RULE_VIOLATION", "STATISTICAL_ANOMALY", "VALID_INCOMPLETE"]
AnomalyCategoryType = Literal["RULE_VIOLATION", "STATISTICAL_ANOMALY"]
DiagnosticSeverityType = Literal["CRITICAL", "WARNING", "INFO"]
InputFormatType = Literal["DELIMITED_STRING", "EVENT_LIST"]


class FsmErrorDetail(BaseModel):
    """Structured details of an FSM transition rule violation."""

    error_type: str = Field(..., description="Category of FSM error, e.g. INVALID_TRANSITION")
    failed_step: int = Field(..., ge=1, description="1-based step index where failure occurred")
    failed_event: str = Field(..., description="Event symbol that violated transition rules")
    state_at_failure: str = Field(..., description="Automaton state when failure occurred")
    allowed_events: List[str] = Field(..., description="Permitted event symbols from the failure state (empty after the end)")
    message: str = Field(..., description="The FSM's own description of the failure")


class FsmEvaluationSchema(BaseModel):
    """Deterministic FSM validation results."""

    is_rule_valid: bool = Field(..., description="True if all transitions conform to FSM rules")
    is_complete: bool = Field(..., description="True if trace terminated in an approved terminal state")
    final_state: str = Field(..., description="Automaton state upon trace termination")
    total_steps: int = Field(..., ge=0, description="Total events evaluated")
    successful_transitions: int = Field(..., ge=0, description="Number of successful state transitions")
    error: Optional[FsmErrorDetail] = Field(..., description="FSM violation details if invalid, else null")


def _strict_finite_float(v: Any, field_name: str) -> float:
    """Reject booleans, strings, NaN and infinity; return a float."""
    if type(v) is bool or not isinstance(v, (int, float)):
        raise ValueError(f"Field '{field_name}' must be numeric (int or float), got {type(v).__name__}")
    float_val = float(v)
    if not math.isfinite(float_val):
        raise ValueError(f"Field '{field_name}' must be finite, got {v!r}")
    return float_val


class TransitionProbabilitySchema(BaseModel):
    """One step of the order path and how likely it is under normal behaviour."""

    step: int = Field(..., ge=1, description="1-based step number")
    from_state: str = Field(..., description="State before the event")
    event: str = Field(..., description="Event taken at this step")
    to_state: str = Field(..., description="State after the event")
    probability: float = Field(..., gt=0.0, le=1.0, description="P(event | from_state) learned from normal orders")
    is_unusual: bool = Field(..., description="True when this kind of step, over all its occurrences in the "
                                              "order, has a combined probability below 10%")

    @field_validator("step", mode="before")
    @classmethod
    def validate_step(cls, v: Any) -> int:
        if type(v) is not int:
            raise ValueError(f"Field 'step' must be an integer, got {type(v).__name__}")
        return v

    @field_validator("probability", mode="before")
    @classmethod
    def validate_probability(cls, v: Any) -> float:
        return _strict_finite_float(v, "probability")


class MlEvaluationSchema(BaseModel):
    """Markov chain path model results: how likely is this order path under normal behaviour."""

    prediction: Literal["NORMAL", "ANOMALY"] = Field(..., description="ML classification verdict")
    is_ml_anomaly: bool = Field(..., description="True if statistical anomaly, False if normal")
    anomaly_score: float = Field(
        ...,
        description="threshold - log10 P(path); positive means anomaly, each +1 is 10x rarer than the threshold. "
                    "The threshold is shared with the timing model's checks (one false-alarm budget for all).",
    )
    path_log10_probability: float = Field(..., le=0.0, description="log10 of the path probability")
    transitions: List[TransitionProbabilitySchema] = Field(
        ..., min_length=1, description="Every step of the path with its probability"
    )

    @field_validator("anomaly_score", "path_log10_probability", mode="before")
    @classmethod
    def validate_finite_score(cls, v: Any, info) -> float:
        """Validate that ML scores are finite numeric values and not booleans or strings."""
        return _strict_finite_float(v, info.field_name)

    @model_validator(mode="after")
    def validate_consistency(self) -> MlEvaluationSchema:
        """Prediction, flag, score and step probabilities must all agree."""
        if self.prediction == "NORMAL" and self.is_ml_anomaly is not False:
            raise ValueError("Inconsistent ML evaluation: prediction 'NORMAL' requires is_ml_anomaly=False.")
        if self.prediction == "ANOMALY" and self.is_ml_anomaly is not True:
            raise ValueError("Inconsistent ML evaluation: prediction 'ANOMALY' requires is_ml_anomaly=True.")
        if self.is_ml_anomaly != (self.anomaly_score > 0.0):
            raise ValueError(
                f"Inconsistent ML evaluation: anomaly_score {self.anomaly_score} must be > 0 "
                f"exactly when the prediction is 'ANOMALY'."
            )

        steps = [t.step for t in self.transitions]
        if steps != list(range(1, len(steps) + 1)):
            raise ValueError(f"Transition steps must be numbered 1..n in order, got {steps}.")

        expected_log10 = sum(math.log10(t.probability) for t in self.transitions)
        if not math.isclose(self.path_log10_probability, expected_log10, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"path_log10_probability {self.path_log10_probability} does not match the step "
                f"probabilities (expected {expected_log10})."
            )
        return self


class TimingMeasurementSchema(BaseModel):
    """One measured value of the order compared with its normal range from training data."""

    name: str = Field(..., description="'order_amount', 'payment_failures', or a step kind like 'PACK_ITEMS -> SHIP_ITEMS'")
    step: Optional[int] = Field(default=None, ge=1, description="Step number for step durations, else null")
    unit: Literal["seconds", "currency", "count"]
    value: float
    normal_low: float = Field(..., description="2.5th percentile among normal training orders")
    normal_high: float = Field(..., description="97.5th percentile among normal training orders")
    percentile: float = Field(..., ge=0.0, le=100.0, description="Share of normal orders below this value, in %")
    is_unusual: bool = Field(..., description="True when the percentile is outside the middle 95% (2.5 to 97.5)")

    @field_validator("value", "normal_low", "normal_high", "percentile", mode="before")
    @classmethod
    def validate_numbers(cls, v: Any, info) -> float:
        return _strict_finite_float(v, info.field_name)

    @model_validator(mode="after")
    def validate_consistency(self) -> TimingMeasurementSchema:
        if (self.unit == "seconds") != (self.step is not None):
            raise ValueError("'step' is required for step durations (unit 'seconds') and only for them.")
        if self.is_unusual != (not NORMAL_BAND[0] <= self.percentile <= NORMAL_BAND[1]):
            raise ValueError(f"'is_unusual' must be true exactly when the percentile is outside {list(NORMAL_BAND)}.")
        return self


class TimingCheckSchema(BaseModel):
    """One of the timing model's checks on an order."""

    name: str = Field(..., description=f"One of {TIMING_CHECKS}")
    score: float = Field(..., description="level - threshold, where level 0 is a typical normal order and level 1 "
                                          "the point 1% of normal validation orders exceed; positive means flagged")
    is_flagged: bool

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, v: Any) -> float:
        return _strict_finite_float(v, "score")

    @model_validator(mode="after")
    def validate_consistency(self) -> TimingCheckSchema:
        if self.is_flagged != (self.score > 0.0):
            raise ValueError("Inconsistent timing check: score must be > 0 exactly when is_flagged is true.")
        return self


class TimingEvaluationSchema(BaseModel):
    """Timing-and-amount model results on the order's times, amount and outcome."""

    prediction: Literal["NORMAL", "ANOMALY"]
    is_timing_anomaly: bool
    anomaly_score: float = Field(..., description="The highest check score; positive means anomaly")
    model_name: str = Field(..., description=f"The detector behind the cluster check: one of {DETECTOR_NAMES}")
    checks: List[TimingCheckSchema] = Field(..., description=f"One entry per check, in the order {TIMING_CHECKS}")
    trend_z: float = Field(..., description="Sum of the step z-scores / sqrt(number of steps); negative means the "
                                            "steps were faster than normal overall")
    inputs: Dict[str, float] = Field(..., description=f"The numbers the model scored: {TIMING_MODEL_INPUTS}")
    measurements: List[TimingMeasurementSchema] = Field(..., min_length=2)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if v not in DETECTOR_NAMES:
            raise ValueError(f"'model_name' must be one of {DETECTOR_NAMES}, got {v!r}.")
        return v

    @field_validator("anomaly_score", "trend_z", mode="before")
    @classmethod
    def validate_score(cls, v: Any, info) -> float:
        return _strict_finite_float(v, info.field_name)

    @field_validator("inputs", mode="before")
    @classmethod
    def validate_inputs(cls, v: Any) -> Dict[str, float]:
        if not isinstance(v, dict) or list(v.keys()) != TIMING_MODEL_INPUTS:
            raise ValueError(f"'inputs' must have exactly the keys {TIMING_MODEL_INPUTS} in that order.")
        return {k: _strict_finite_float(val, k) for k, val in v.items()}

    @model_validator(mode="after")
    def validate_consistency(self) -> TimingEvaluationSchema:
        if (self.prediction == "ANOMALY") != self.is_timing_anomaly:
            raise ValueError("Inconsistent timing evaluation: prediction and is_timing_anomaly disagree.")
        if self.is_timing_anomaly != (self.anomaly_score > 0.0):
            raise ValueError("Inconsistent timing evaluation: anomaly_score must be > 0 exactly for ANOMALY.")
        if [c.name for c in self.checks] != TIMING_CHECKS:
            raise ValueError(f"'checks' must be exactly {TIMING_CHECKS} in that order.")
        if not math.isclose(self.anomaly_score, max(c.score for c in self.checks), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Inconsistent timing evaluation: anomaly_score must equal the highest check score.")
        return self


class DiagnosticsSchema(BaseModel):
    """Operational diagnostics and actionable remediation advice."""

    severity: DiagnosticSeverityType = Field(..., description="Operational severity: CRITICAL, WARNING, or INFO")
    root_cause: str = Field(..., description="Concise explanation of the underlying cause")
    key_factors: List[str] = Field(..., description="Key factors contributing to diagnosis")
    recommended_action: str = Field(..., description="Remediation steps for operational staff")


_SEVERITY = {"RULE_VIOLATION": "CRITICAL", "STATISTICAL_ANOMALY": "WARNING", "VALID_INCOMPLETE": "INFO", "NORMAL": "INFO"}


class AnalysisResultSchema(BaseModel):
    """One complete analysis result, checked as a whole before it is stored or returned."""

    order_id: Optional[str] = Field(..., description="Associated order identifier, or null")
    verdict: VerdictType = Field(..., description="Decision verdict in 4-state taxonomy")
    is_anomaly: bool = Field(..., description="True if trace is an FSM violation or ML anomaly")
    anomaly_type: Optional[AnomalyCategoryType] = Field(..., description="Category of anomaly if flagged, else null")
    summary: str = Field(..., description="High-level human-readable verdict summary")
    fsm_evaluation: FsmEvaluationSchema
    event_timestamps: Optional[List[str]] = Field(..., description="Timestamps as submitted, or null")
    order_amount: Optional[float] = Field(..., description="Order amount as submitted, or null")
    ml_evaluation: Optional[MlEvaluationSchema] = Field(..., description="Path check; null when skipped")
    ml_skip_reason: Optional[str] = Field(..., description="Reason why the path check was skipped, else null")
    timing_evaluation: Optional[TimingEvaluationSchema] = Field(..., description="Timing check; null when skipped")
    timing_skip_reason: Optional[str] = Field(..., description="Reason why the timing check was skipped, else null")
    diagnostics: DiagnosticsSchema

    @model_validator(mode="after")
    def validate_result(self) -> "AnalysisResultSchema":
        fsm, ml, timing = self.fsm_evaluation, self.ml_evaluation, self.timing_evaluation
        flagged = self.verdict in ("RULE_VIOLATION", "STATISTICAL_ANOMALY")
        if self.is_anomaly != flagged or self.anomaly_type != (self.verdict if flagged else None):
            raise ValueError(f"is_anomaly={self.is_anomaly} and anomaly_type={self.anomaly_type!r} "
                             f"do not fit verdict {self.verdict}.")
        if self.diagnostics.severity != _SEVERITY[self.verdict]:
            raise ValueError(f"severity {self.diagnostics.severity} does not fit verdict {self.verdict}.")
        if (ml is None) == (self.ml_skip_reason is None):
            raise ValueError("Exactly one of ml_evaluation and ml_skip_reason must be given.")
        if (timing is None) == (self.timing_skip_reason is None):
            raise ValueError("Exactly one of timing_evaluation and timing_skip_reason must be given.")
        if (self.event_timestamps is None) != (self.order_amount is None):
            raise ValueError("event_timestamps and order_amount must be given together.")
        if timing is not None and self.event_timestamps is None:
            raise ValueError("A timing evaluation needs the event_timestamps and order_amount it was computed from.")
        expected = {
            "RULE_VIOLATION": not fsm.is_rule_valid,
            "VALID_INCOMPLETE": fsm.is_rule_valid and not fsm.is_complete,
            "NORMAL": fsm.is_rule_valid and fsm.is_complete and ml is not None and not ml.is_ml_anomaly
                      and (timing is None or not timing.is_timing_anomaly),
            "STATISTICAL_ANOMALY": fsm.is_rule_valid and fsm.is_complete and ml is not None
                                   and (ml.is_ml_anomaly or (timing is not None and timing.is_timing_anomaly)),
        }
        if not expected[self.verdict]:
            raise ValueError(f"verdict {self.verdict} does not follow from the FSM and model results.")
        return self


class AnalysisRecordResponse(AnalysisResultSchema):
    """A stored analysis: the result plus its record id, the raw input and when it was stored."""

    id: int = Field(..., ge=1, description="Database record primary key")
    input_format: InputFormatType = Field(..., description="Format classification: DELIMITED_STRING or EVENT_LIST")
    raw_input_payload: Any = Field(..., description="Verbatim original input trace")
    created_at: datetime = Field(..., description="Timezone-aware UTC timestamp of analysis record creation")

    @field_validator("created_at")
    @classmethod
    def validate_utc_datetime(cls, v: Any) -> datetime:
        """Validate that created_at is timezone-aware and has zero UTC offset."""
        if not isinstance(v, datetime):
            raise ValueError(f"'created_at' must be a datetime object, got {type(v).__name__}")
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("'created_at' must be a timezone-aware UTC datetime, not naive.")
        if v.utcoffset() != timedelta(0):
            raise ValueError(
                f"'created_at' must be in UTC with zero offset, got offset {v.utcoffset()}."
            )
        return v.astimezone(timezone.utc)
