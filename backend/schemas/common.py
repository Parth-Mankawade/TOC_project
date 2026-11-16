"""Pydantic v2 schemas for health, lists, metrics, errors, the automaton, models and example orders."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from backend.schemas.anomaly import AnalysisRecordResponse


class LivenessResponse(BaseModel):
    status: Literal["alive"]


class LoadedModelInfo(BaseModel):
    path: str
    model_name: str
    training_sample_count: int


class ReadinessResponse(BaseModel):
    """Returned only when the database table exists and both model files load."""

    service_status: Literal["OPERATIONAL"]
    database: Literal["ok"]
    markov_model: LoadedModelInfo
    timing_model: LoadedModelInfo
    supported_verdicts: List[str]


class BatchSummaryMetrics(BaseModel):
    total_orders: int = Field(..., ge=1)
    normal_count: int = Field(..., ge=0)
    rule_violation_count: int = Field(..., ge=0)
    statistical_anomaly_count: int = Field(..., ge=0)
    valid_incomplete_count: int = Field(..., ge=0)
    total_anomalies: int = Field(..., ge=0, description="Rule violations + statistical anomalies")
    anomaly_rate: float = Field(..., ge=0.0, le=1.0, description="total_anomalies / total_orders")


class BatchAnalysisResponse(BaseModel):
    results: List[AnalysisRecordResponse]
    summary: BatchSummaryMetrics


class OrderHistoryResponse(BaseModel):
    order_id: str
    returned_count: int = Field(..., ge=0)
    analyses: List[AnalysisRecordResponse] = Field(..., description="Newest first")


class PaginatedAnalysesResponse(BaseModel):
    items: List[AnalysisRecordResponse] = Field(..., description="Newest first")
    total: int = Field(..., ge=0, description="Matching records in the database")
    skip: int = Field(..., ge=0)
    limit: int = Field(..., gt=0)


class DatabaseSummaryMetricsResponse(BaseModel):
    total_analyses: int = Field(..., ge=0)
    normal_count: int = Field(..., ge=0)
    rule_violation_count: int = Field(..., ge=0)
    statistical_anomaly_count: int = Field(..., ge=0)
    valid_incomplete_count: int = Field(..., ge=0)
    total_anomalies: int = Field(..., ge=0)
    anomaly_rate: Optional[float] = Field(..., ge=0.0, le=1.0, description="null when nothing is stored yet")
    unique_orders_count: int = Field(..., ge=0, description="Distinct non-null order IDs")


class ErrorDetail(BaseModel):
    code: str = Field(..., description="Machine-readable code, e.g. INPUT_ERROR, MODEL_LOAD_ERROR, NOT_FOUND")
    checkpoint: str = Field(..., description="Where it failed: INPUT, MODEL, LOAD, EVALUATE, STORAGE, ANALYZE or API")
    message: str = Field(..., description="[CHECKPOINT] expected ..., got ...; input=...")
    details: Optional[List[str]] = Field(..., description="Every problem found, when there is more than one")


class ErrorResponse(BaseModel):
    error: ErrorDetail


class FsmTransitionSchema(BaseModel):
    from_state: str
    event: str
    to_state: str


class ReachabilitySchema(BaseModel):
    """Graph-search results over the automaton (backend/fsm/reachability.py)."""

    initial_state: str
    reachable_states: List[str]
    unreachable_states: List[str]
    states_that_can_finish: List[str]
    deadlock_states: List[str]
    is_deadlock_free: bool
    shortest_paths: Dict[str, List[str]] = Field(..., description="Shortest event sequence reaching each state")
    cycles: List[List[str]] = Field(..., description="Groups of states that can reach each other (loops)")
    accepted_sequences_by_length: List[int] = Field(
        ..., description="Number of valid, complete event sequences of length 0, 1, 2, ..."
    )
    language_is_infinite: bool


class FsmDefinitionResponse(BaseModel):
    """The workflow automaton M = (Q, Sigma, delta, q0, F) and its reachability analysis."""

    states: List[str]
    events: List[str]
    initial_state: str
    terminal_states: List[str]
    transitions: List[FsmTransitionSchema]
    reachability: ReachabilitySchema


class ModelsInfoResponse(BaseModel):
    """What the two models learned, and the held-out evaluation of every model."""

    markov_chain: Dict[str, Any] = Field(..., description="The saved Markov chain (markov_chain.json)")
    timing_model: Dict[str, Any] = Field(..., description="The saved timing model's metadata (timing_model.json)")
    evaluation: Dict[str, Any] = Field(..., description="The pipeline's test-split report (results.json)")
    unusual_step_probability: float = Field(..., description="Steps below this combined probability are unusual")


class ExampleOrder(BaseModel):
    """A real order from the test split, with the label the generator gave it."""

    order_id: str
    category: Literal["NORMAL", "STATISTICAL_ANOMALY", "RULE_VIOLATION", "IN_PROGRESS"]
    anomaly_type: str
    event_sequence: List[str]
    event_timestamps: List[str]
    order_amount: float


class ExamplesResponse(BaseModel):
    examples: List[ExampleOrder]


class DemoOrdersResponse(BaseModel):
    orders: List[ExampleOrder] = Field(..., description="The demo stream, in its fixed shuffled order")
    composition: Dict[str, int] = Field(..., description="How many orders of each category the stream holds")
