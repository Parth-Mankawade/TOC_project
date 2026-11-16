"""Named pipeline errors (backend/core/errors.py): one type per checkpoint, one message format."""

from __future__ import annotations

import unittest

from backend.core.errors import (
    AnalyzeError,
    DataLoadError,
    EvaluateError,
    FeatureError,
    GenerateError,
    InputError,
    ModelLoadError,
    PipelineError,
    StorageError,
    TrainError,
)

ALL_ERRORS = [GenerateError, DataLoadError, FeatureError, TrainError, ModelLoadError,
              EvaluateError, InputError, AnalyzeError, StorageError]


class TestPipelineErrors(unittest.TestCase):

    def test_message_names_checkpoint_expected_actual_and_input(self) -> None:
        err = DataLoadError(
            expected="column 'event_timestamps'",
            actual=["trace_id", "split"],
            input="data/orders.csv",
        )
        self.assertEqual(
            str(err),
            "[LOAD] expected column 'event_timestamps', got ['trace_id', 'split']; input=data/orders.csv",
        )
        self.assertEqual(err.checkpoint, "LOAD")
        self.assertEqual(err.expected, "column 'event_timestamps'")

    def test_every_checkpoint_has_its_own_name(self) -> None:
        names = [cls.checkpoint for cls in ALL_ERRORS]
        self.assertEqual(len(names), len(set(names)))
        for cls in ALL_ERRORS:
            self.assertTrue(issubclass(cls, PipelineError))
            self.assertTrue(str(cls("x", "y", "z")).startswith(f"[{cls.checkpoint}] "))

    def test_long_values_are_shortened_but_keep_their_length(self) -> None:
        err = FeatureError("a short list", list(range(500)), "order X")
        self.assertIn("chars)", err.actual)
        self.assertLess(len(err.actual), 260)

    def test_input_error_is_a_value_error_so_request_validators_report_it(self) -> None:
        self.assertTrue(issubclass(InputError, ValueError))
        with self.assertRaises(ValueError):
            raise InputError("an ISO-8601 timestamp", "yesterday", "event_timestamps[0]")


if __name__ == "__main__":
    unittest.main()
