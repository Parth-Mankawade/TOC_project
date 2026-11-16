"""Request schemas (INPUT checkpoint at the API boundary).

The event sequence may be "A;B;C" text or a list of names and is stored exactly as
sent. Event timestamps and the order amount are optional but must come together,
one timestamp per event, strictly increasing. Problems raise InputError, which
FastAPI reports as HTTP 422 with the message "[INPUT] expected ..., got ...; input=...".
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator

from backend.fsm.parsing import parse_event_sequence
from backend.pipeline.analyzer import check_timing_inputs


class _TraceFields(BaseModel):
    model_config = ConfigDict(strict=True)

    order_id: Optional[StrictStr] = Field(default=None, max_length=64, description="Optional order identifier")
    event_sequence: Union[StrictStr, List[StrictStr]] = Field(
        ..., description="Events as 'A;B;C' text or a list of event names")
    event_timestamps: Optional[List[StrictStr]] = Field(
        default=None, description="Optional ISO-8601 timestamps with timezone, one per event, strictly increasing "
                                  "(e.g. '2026-05-24T05:30:31Z'). Turns on the timing/amount check.")
    order_amount: Optional[float] = Field(default=None, gt=0, description="Order value; sent together with event_timestamps")

    @field_validator("event_sequence")
    @classmethod
    def validate_event_sequence(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        parse_event_sequence(v)
        return v

    @model_validator(mode="after")
    def validate_timing_inputs(self) -> "_TraceFields":
        check_timing_inputs(parse_event_sequence(self.event_sequence), self.event_timestamps, self.order_amount)
        return self


class WorkflowAnalysisRequest(_TraceFields):
    """One order to analyze and store."""


class TraceItemRequest(_TraceFields):
    """One order inside a batch."""


class BatchAnalysisRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    traces: List[Union[TraceItemRequest, StrictStr, List[StrictStr]]] = Field(
        ..., min_length=1, max_length=500, description="1 to 500 orders: objects, 'A;B;C' text, or lists of names")
