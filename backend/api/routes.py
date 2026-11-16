"""REST endpoints (API checkpoint).

Nothing here catches an error: named errors from the checkpoints below travel to
the handlers in backend/main.py, which turn them into JSON with the right status.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, Path as PathParam, Query, Request, status
from sqlalchemy.orm import Session

from backend.api.dependencies import get_db_session, get_service, get_settings
from backend.api.mapping import record_to_response
from backend.core.config import Settings
from backend.core.errors import EvaluateError, InputError, NotFoundError
from backend.database.connection import check_table
from backend.database.repository import (
    SUPPORTED_VERDICTS,
    create_analysis_batch,
    create_analysis_record,
    get_analyses_by_order_id,
    get_analysis_by_id,
    get_summary_metrics,
    list_analyses,
)
from backend.fsm.reachability import analyze_reachability
from backend.fsm.states import ALL_EVENTS, ALL_STATES, INITIAL_STATE, TERMINAL_STATES
from backend.fsm.transitions import TRANSITION_TABLE
from backend.ml.dataset.generator import format_timestamp
from backend.ml.dataset.loader import OrderRecord, demo_sample, example_orders, load_orders
from backend.ml.markov.model import model_to_dict
from backend.ml.timing.model import model_metadata
from backend.pipeline.analyzer import UNUSUAL_STEP_PROBABILITY
from backend.schemas.anomaly import AnalysisRecordResponse
from backend.schemas.common import (
    BatchAnalysisResponse,
    BatchSummaryMetrics,
    DatabaseSummaryMetricsResponse,
    DemoOrdersResponse,
    ExampleOrder,
    ExamplesResponse,
    FsmDefinitionResponse,
    LivenessResponse,
    ModelsInfoResponse,
    OrderHistoryResponse,
    PaginatedAnalysesResponse,
    ReadinessResponse,
)
from backend.schemas.order import BatchAnalysisRequest, TraceItemRequest, WorkflowAnalysisRequest
from backend.services.anomaly_service import AnomalyService, OrderInput

router = APIRouter(prefix="/api/v1", tags=["Workflow analyses"])
root_router = APIRouter(tags=["Liveness"])

_dataset_cache: Dict[Tuple[str, int], List[OrderRecord]] = {}


def _dataset(settings: Settings) -> List[OrderRecord]:
    """The event log, read once per file version. Raises DataLoadError if it is missing or malformed."""
    path = settings.dataset_path
    key = (str(path), path.stat().st_mtime_ns) if path.is_file() else (str(path), -1)
    if key not in _dataset_cache:
        _dataset_cache[key] = load_orders(path)
    return _dataset_cache[key]


def _example(o: OrderRecord) -> ExampleOrder:
    return ExampleOrder(order_id=o.trace_id, category=o.category, anomaly_type=o.anomaly_type,
                        event_sequence=list(o.events), event_timestamps=[format_timestamp(t) for t in o.timestamps],
                        order_amount=o.amount)


def _order_input(item: Union[TraceItemRequest, WorkflowAnalysisRequest]) -> OrderInput:
    return OrderInput(event_sequence=item.event_sequence, order_id=item.order_id,
                      event_timestamps=item.event_timestamps, order_amount=item.order_amount)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@root_router.get("/health", response_model=LivenessResponse, summary="The process is running")
@router.get("/health/live", response_model=LivenessResponse, summary="The process is running")
def get_liveness() -> LivenessResponse:
    return LivenessResponse(status="alive")


@router.get("/health", response_model=ReadinessResponse, summary="Database table and both models are usable")
@router.get("/health/ready", response_model=ReadinessResponse, summary="Database table and both models are usable")
def get_readiness(request: Request, service: AnomalyService = Depends(get_service)) -> ReadinessResponse:
    """200 only when the analysis table exists and both model files load; otherwise 503 saying why."""
    check_table(request.app.state.engine)
    return ReadinessResponse(database="ok", **service.readiness())


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

@router.post("/analyses", response_model=AnalysisRecordResponse, status_code=status.HTTP_201_CREATED,
             summary="Analyze one order and store the result")
def create_analysis(body: WorkflowAnalysisRequest, service: AnomalyService = Depends(get_service),
                    db: Session = Depends(get_db_session)) -> AnalysisRecordResponse:
    result = service.analyze_order(_order_input(body))
    return record_to_response(create_analysis_record(db, result, body.event_sequence))


@router.post("/analyses/batch", response_model=BatchAnalysisResponse, status_code=status.HTTP_201_CREATED,
             summary="Analyze 1-500 orders and store all results in one transaction")
def create_batch_analysis(body: BatchAnalysisRequest, service: AnomalyService = Depends(get_service),
                          db: Session = Depends(get_db_session)) -> BatchAnalysisResponse:
    orders: List[OrderInput] = []
    raw_inputs: List[Union[str, List[str]]] = []
    for item in body.traces:
        if isinstance(item, TraceItemRequest):
            orders.append(_order_input(item))
            raw_inputs.append(item.event_sequence)
        else:
            orders.append(OrderInput(event_sequence=item))
            raw_inputs.append(item)
    batch = service.analyze_batch(orders)
    records = create_analysis_batch(db, batch["results"], raw_inputs)
    return BatchAnalysisResponse(results=[record_to_response(r) for r in records],
                                 summary=BatchSummaryMetrics(**batch["summary"]))


@router.get("/analyses", response_model=PaginatedAnalysesResponse, summary="Stored analyses, newest first")
def get_analyses_page(skip: int = Query(0, ge=0), limit: int = Query(50, gt=0, le=1000),
                      verdict: Optional[str] = Query(None), is_anomaly: Optional[bool] = Query(None),
                      db: Session = Depends(get_db_session)) -> PaginatedAnalysesResponse:
    if verdict is not None and verdict not in SUPPORTED_VERDICTS:
        raise InputError(f"a verdict in {list(SUPPORTED_VERDICTS)}", verdict, "verdict")
    records, total = list_analyses(db, skip=skip, limit=limit, verdict=verdict, is_anomaly=is_anomaly)
    return PaginatedAnalysesResponse(items=[record_to_response(r) for r in records], total=total, skip=skip, limit=limit)


@router.get("/analyses/{analysis_id}", response_model=AnalysisRecordResponse, summary="One stored analysis")
def get_single_analysis(analysis_id: int = PathParam(..., gt=0),
                        db: Session = Depends(get_db_session)) -> AnalysisRecordResponse:
    record = get_analysis_by_id(db, analysis_id)
    if record is None:
        raise NotFoundError("a stored analysis with this id", "no such record", f"analysis_id={analysis_id}")
    return record_to_response(record)


@router.get("/orders/{order_id}/analyses", response_model=OrderHistoryResponse, summary="Every analysis of one order")
def get_order_history(order_id: str = PathParam(..., min_length=1, max_length=64), limit: int = Query(50, gt=0, le=1000),
                      db: Session = Depends(get_db_session)) -> OrderHistoryResponse:
    analyses = [record_to_response(r) for r in get_analyses_by_order_id(db, order_id, limit)]
    return OrderHistoryResponse(order_id=order_id, returned_count=len(analyses), analyses=analyses)


@router.get("/metrics", response_model=DatabaseSummaryMetricsResponse, summary="Counts over every stored analysis")
def get_metrics(db: Session = Depends(get_db_session)) -> DatabaseSummaryMetricsResponse:
    return DatabaseSummaryMetricsResponse(**get_summary_metrics(db))


# ---------------------------------------------------------------------------
# Automaton, models and evaluation (read-only)
# ---------------------------------------------------------------------------

@router.get("/fsm", response_model=FsmDefinitionResponse, summary="The workflow automaton and its reachability analysis")
def get_fsm_definition() -> FsmDefinitionResponse:
    return FsmDefinitionResponse(
        states=sorted(ALL_STATES), events=sorted(ALL_EVENTS), initial_state=INITIAL_STATE.value,
        terminal_states=sorted(TERMINAL_STATES),
        transitions=[{"from_state": s, "event": e, "to_state": t} for (s, e), t in sorted(TRANSITION_TABLE.items())],
        reachability=analyze_reachability(),
    )


def _evaluation_report(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise EvaluateError("the evaluation report (run: python -m backend.ml.pipeline)", "no file", str(path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluateError("valid JSON", str(exc), str(path)) from exc


@router.get("/models", response_model=ModelsInfoResponse,
            summary="What both models learned, and how every model did on the held-out test split")
def get_models_info(service: AnomalyService = Depends(get_service),
                    settings: Settings = Depends(get_settings)) -> ModelsInfoResponse:
    return ModelsInfoResponse(markov_chain=model_to_dict(service.markov_model()),
                              timing_model=model_metadata(service.timing_model()),
                              evaluation=_evaluation_report(settings.results_path),
                              unusual_step_probability=UNUSUAL_STEP_PROBABILITY)


# ---------------------------------------------------------------------------
# Real test-split orders for the dashboard
# ---------------------------------------------------------------------------

@router.get("/examples", response_model=ExamplesResponse, summary="One real test order of every kind")
def get_examples(settings: Settings = Depends(get_settings)) -> ExamplesResponse:
    return ExamplesResponse(examples=[_example(o) for o in example_orders(_dataset(settings))])


@router.get("/demo", response_model=DemoOrdersResponse, summary="The demo stream of test orders (not analyzed)")
def get_demo_orders(settings: Settings = Depends(get_settings)) -> DemoOrdersResponse:
    orders = demo_sample(_dataset(settings))
    return DemoOrdersResponse(orders=[_example(o) for o in orders],
                              composition=dict(sorted(Counter(o.category for o in orders).items())))


@router.post("/demo", response_model=BatchSummaryMetrics, status_code=status.HTTP_201_CREATED,
             summary="Analyze and store the demo stream in one go")
def load_demo_orders(service: AnomalyService = Depends(get_service), db: Session = Depends(get_db_session),
                     settings: Settings = Depends(get_settings)) -> BatchSummaryMetrics:
    orders = demo_sample(_dataset(settings))
    inputs = [OrderInput(event_sequence=list(o.events), order_id=o.trace_id,
                         event_timestamps=[format_timestamp(t) for t in o.timestamps], order_amount=o.amount)
              for o in orders]
    batch = service.analyze_batch(inputs)
    create_analysis_batch(db, batch["results"], [list(o.events) for o in orders])
    return BatchSummaryMetrics(**batch["summary"])
