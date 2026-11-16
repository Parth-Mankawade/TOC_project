"""Named errors, one per pipeline checkpoint.

Every message has the same shape, so a failure says where it happened, what was
expected, what actually arrived, and which input caused it:

    [LOAD] expected column 'event_timestamps', got ['trace_id', 'split']; input=data/orders.csv

Checkpoints, in pipeline order:
    GENERATE  synthetic data generation          GenerateError
    LOAD      reading and checking the CSV       DataLoadError
    FEATURES  turning an order into model inputs FeatureError
    TRAIN     fitting a model                    TrainError
    MODEL     loading a saved model file         ModelLoadError
    EVALUATE  measuring models on held-out data  EvaluateError
    INPUT     an order sent for analysis         InputError (also a ValueError)
    ANALYZE   combining FSM and model results    AnalyzeError
    STORAGE   saving or reading analysis records StorageError
"""

from __future__ import annotations

from typing import Any

_MAX_SHOWN = 200  # characters of a value shown in a message before it is shortened


def _shown(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= _MAX_SHOWN:
        return text
    return f"{text[:_MAX_SHOWN]}... ({len(text)} chars)"


class PipelineError(Exception):
    """Base class: never raised directly."""

    checkpoint = "PIPELINE"

    def __init__(self, expected: str, actual: Any, input: Any) -> None:
        self.expected = expected
        self.actual = _shown(actual)
        self.input = _shown(input)
        super().__init__(f"[{self.checkpoint}] expected {expected}, got {self.actual}; input={self.input}")


class GenerateError(PipelineError):
    checkpoint = "GENERATE"


class DataLoadError(PipelineError):
    checkpoint = "LOAD"


class FeatureError(PipelineError):
    checkpoint = "FEATURES"


class TrainError(PipelineError):
    checkpoint = "TRAIN"


class ModelLoadError(PipelineError):
    checkpoint = "MODEL"


class EvaluateError(PipelineError):
    checkpoint = "EVALUATE"


class InputError(PipelineError, ValueError):
    """Bad input sent for analysis. Also a ValueError, so request validators report it as a 422."""

    checkpoint = "INPUT"


class AnalyzeError(PipelineError):
    checkpoint = "ANALYZE"


class StorageError(PipelineError):
    checkpoint = "STORAGE"


class NotFoundError(PipelineError):
    """A requested record does not exist (HTTP 404)."""

    checkpoint = "API"
