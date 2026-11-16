"""Saving and reading analysis records (STORAGE checkpoint).

Every result is validated against AnalysisResultSchema before it is written, so
nothing inconsistent is stored; the database's CHECK constraints guard the same
rules again. Any database failure becomes a StorageError and nothing is half-written.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.core.errors import StorageError
from backend.database.models import WorkflowAnalysis
from backend.schemas.anomaly import AnalysisResultSchema

SUPPORTED_VERDICTS: Tuple[str, ...] = ("NORMAL", "RULE_VIOLATION", "STATISTICAL_ANOMALY", "VALID_INCOMPLETE")

T = TypeVar("T")


def validation_summary(exc: ValidationError) -> str:
    """One line per problem: 'field.path: message'."""
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'result'}: {e['msg']}" for e in exc.errors())


def classify_raw_input(raw_input: Any) -> Tuple[str, Any]:
    """('DELIMITED_STRING', text) or ('EVENT_LIST', list), kept exactly as submitted."""
    if isinstance(raw_input, str):
        return "DELIMITED_STRING", raw_input
    if isinstance(raw_input, list) and all(isinstance(item, str) for item in raw_input):
        return "EVENT_LIST", raw_input
    raise StorageError("raw input as text or a list of text", repr(raw_input), "raw_input")


def to_naive_utc(timestamp: Optional[datetime]) -> datetime:
    """UTC without tzinfo, as SQLite stores it. None means now."""
    if timestamp is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    if timestamp.tzinfo is not None:
        return timestamp.astimezone(timezone.utc).replace(tzinfo=None)
    return timestamp


def _build_record(result: Dict[str, Any], raw_input: Union[str, List[str]],
                  timestamp: Optional[datetime]) -> WorkflowAnalysis:
    input_format, payload = classify_raw_input(raw_input)
    try:
        r = AnalysisResultSchema.model_validate(result)
    except ValidationError as exc:
        raise StorageError("an analysis result that passes AnalysisResultSchema", validation_summary(exc),
                           "analysis result") from exc
    fsm, ml, timing, diag = r.fsm_evaluation, r.ml_evaluation, r.timing_evaluation, r.diagnostics
    return WorkflowAnalysis(
        order_id=r.order_id,
        input_format=input_format,
        raw_input_payload=payload,
        verdict=r.verdict,
        is_anomaly=r.is_anomaly,
        anomaly_type=r.anomaly_type,
        summary=r.summary,
        is_rule_valid=fsm.is_rule_valid,
        is_complete=fsm.is_complete,
        final_state=fsm.final_state,
        total_steps=fsm.total_steps,
        successful_transitions=fsm.successful_transitions,
        fsm_error=None if fsm.error is None else fsm.error.model_dump(),
        ml_evaluated=ml is not None,
        ml_prediction=None if ml is None else ml.prediction,
        is_ml_anomaly=None if ml is None else ml.is_ml_anomaly,
        anomaly_score=None if ml is None else ml.anomaly_score,
        path_log10_probability=None if ml is None else ml.path_log10_probability,
        ml_transitions=None if ml is None else [t.model_dump() for t in ml.transitions],
        ml_skip_reason=r.ml_skip_reason,
        event_timestamps=r.event_timestamps,
        order_amount=r.order_amount,
        timing_evaluation=None if timing is None else timing.model_dump(),
        timing_skip_reason=r.timing_skip_reason,
        diagnostic_severity=diag.severity,
        diagnostic_root_cause=diag.root_cause,
        diagnostic_key_factors=diag.key_factors,
        diagnostic_recommended_action=diag.recommended_action,
        created_at=to_naive_utc(timestamp),
    )


def _save(db: Session, records: List[WorkflowAnalysis]) -> List[WorkflowAnalysis]:
    try:
        db.add_all(records)
        db.commit()
        for record in records:
            db.refresh(record)
    except SQLAlchemyError as exc:
        db.rollback()
        raise StorageError(f"{len(records)} record(s) saved", f"{type(exc).__name__}: {exc}", "database write") from exc
    return records


def _read(what: str, query: Callable[[], T]) -> T:
    try:
        return query()
    except SQLAlchemyError as exc:
        raise StorageError(f"a readable '{WorkflowAnalysis.__tablename__}' table", f"{type(exc).__name__}: {exc}",
                           what) from exc


def create_analysis_record(db: Session, analysis_result: Dict[str, Any], raw_input: Union[str, List[str]],
                           timestamp: Optional[datetime] = None) -> WorkflowAnalysis:
    """Validate and save one result. Raises StorageError on an invalid result or a database failure."""
    return _save(db, [_build_record(analysis_result, raw_input, timestamp)])[0]


def create_analysis_batch(db: Session, analysis_results: Sequence[Dict[str, Any]],
                          raw_inputs: Sequence[Union[str, List[str]]],
                          timestamp: Optional[datetime] = None) -> List[WorkflowAnalysis]:
    """Validate every result first, then save all of them in one transaction (all or nothing)."""
    if len(analysis_results) != len(raw_inputs):
        raise StorageError(f"{len(analysis_results)} raw inputs (one per result)", f"{len(raw_inputs)} raw inputs",
                           "batch")
    records = [_build_record(r, raw, timestamp) for r, raw in zip(analysis_results, raw_inputs)]
    return _save(db, records)


def get_analysis_by_id(db: Session, analysis_id: int) -> Optional[WorkflowAnalysis]:
    """The record with this id, or None when there is no such record."""
    return _read(f"analysis {analysis_id}", lambda: db.get(WorkflowAnalysis, analysis_id))


def get_analyses_by_order_id(db: Session, order_id: str, limit: int = 50) -> List[WorkflowAnalysis]:
    """Every analysis of one order, newest first."""
    if limit <= 0:
        raise StorageError("a positive limit", limit, "get_analyses_by_order_id")
    stmt = (select(WorkflowAnalysis).where(WorkflowAnalysis.order_id == order_id)
            .order_by(WorkflowAnalysis.created_at.desc(), WorkflowAnalysis.id.desc()).limit(limit))
    return _read(f"history of order {order_id}", lambda: list(db.scalars(stmt).all()))


def list_analyses(db: Session, skip: int = 0, limit: int = 50, verdict: Optional[str] = None,
                  is_anomaly: Optional[bool] = None) -> Tuple[List[WorkflowAnalysis], int]:
    """One page of records (newest first) and how many records match in total."""
    if skip < 0 or limit <= 0:
        raise StorageError("skip >= 0 and limit > 0", f"skip={skip}, limit={limit}", "list_analyses")
    if verdict is not None and verdict not in SUPPORTED_VERDICTS:
        raise StorageError(f"a verdict in {list(SUPPORTED_VERDICTS)}", verdict, "list_analyses")
    conditions = []
    if verdict is not None:
        conditions.append(WorkflowAnalysis.verdict == verdict)
    if is_anomaly is not None:
        conditions.append(WorkflowAnalysis.is_anomaly == is_anomaly)
    count_stmt = select(func.count(WorkflowAnalysis.id)).where(*conditions)
    page_stmt = (select(WorkflowAnalysis).where(*conditions)
                 .order_by(WorkflowAnalysis.created_at.desc(), WorkflowAnalysis.id.desc()).offset(skip).limit(limit))
    return _read("analysis list", lambda: (list(db.scalars(page_stmt).all()), int(db.scalar(count_stmt))))


def get_summary_metrics(db: Session) -> Dict[str, Any]:
    """Counts per verdict over every stored record; anomaly_rate is None when nothing is stored."""
    def count(*conditions: Any) -> int:
        return int(db.scalar(select(func.count(WorkflowAnalysis.id)).where(*conditions)))

    def query() -> Dict[str, Any]:
        by_verdict = {v: count(WorkflowAnalysis.verdict == v) for v in SUPPORTED_VERDICTS}
        total = sum(by_verdict.values())
        anomalies = by_verdict["RULE_VIOLATION"] + by_verdict["STATISTICAL_ANOMALY"]
        unique_orders = int(db.scalar(select(func.count(func.distinct(WorkflowAnalysis.order_id)))
                                      .where(WorkflowAnalysis.order_id.is_not(None))))
        return {
            "total_analyses": total,
            "normal_count": by_verdict["NORMAL"],
            "rule_violation_count": by_verdict["RULE_VIOLATION"],
            "statistical_anomaly_count": by_verdict["STATISTICAL_ANOMALY"],
            "valid_incomplete_count": by_verdict["VALID_INCOMPLETE"],
            "total_anomalies": anomalies,
            "anomaly_rate": anomalies / total if total else None,
            "unique_orders_count": unique_orders,
        }

    return _read("summary metrics", query)
