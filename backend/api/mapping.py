"""Stored record -> API response (STORAGE checkpoint, read side).

The flat database columns are rebuilt into the nested response and validated
again; a stored row that no longer passes the schema raises StorageError
naming the record, instead of being returned half-correct.
"""

from __future__ import annotations

from datetime import timezone

from pydantic import ValidationError

from backend.core.errors import StorageError
from backend.database.models import WorkflowAnalysis
from backend.database.repository import validation_summary
from backend.schemas.anomaly import AnalysisRecordResponse


def record_to_response(record: WorkflowAnalysis) -> AnalysisRecordResponse:
    ml = None
    if record.ml_evaluated:
        ml = {"prediction": record.ml_prediction, "is_ml_anomaly": record.is_ml_anomaly,
              "anomaly_score": record.anomaly_score, "path_log10_probability": record.path_log10_probability,
              "transitions": record.ml_transitions}
    data = {
        "id": record.id,
        "order_id": record.order_id,
        "input_format": record.input_format,
        "raw_input_payload": record.raw_input_payload,
        "verdict": record.verdict,
        "is_anomaly": record.is_anomaly,
        "anomaly_type": record.anomaly_type,
        "summary": record.summary,
        "fsm_evaluation": {
            "is_rule_valid": record.is_rule_valid,
            "is_complete": record.is_complete,
            "final_state": record.final_state,
            "total_steps": record.total_steps,
            "successful_transitions": record.successful_transitions,
            "error": record.fsm_error,
        },
        "event_timestamps": record.event_timestamps,
        "order_amount": record.order_amount,
        "ml_evaluation": ml,
        "ml_skip_reason": record.ml_skip_reason,
        "timing_evaluation": record.timing_evaluation,
        "timing_skip_reason": record.timing_skip_reason,
        "diagnostics": {
            "severity": record.diagnostic_severity,
            "root_cause": record.diagnostic_root_cause,
            "key_factors": record.diagnostic_key_factors,
            "recommended_action": record.diagnostic_recommended_action,
        },
        "created_at": record.created_at.replace(tzinfo=timezone.utc),  # stored as naive UTC
    }
    try:
        return AnalysisRecordResponse.model_validate(data)
    except ValidationError as exc:
        raise StorageError("a stored record that passes AnalysisRecordResponse", validation_summary(exc),
                           f"record {record.id}") from exc
