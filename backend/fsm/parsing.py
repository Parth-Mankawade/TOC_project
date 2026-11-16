"""Parse an order sent for analysis (INPUT checkpoint).

Nothing is repaired or skipped: an empty event name, an unreadable timestamp,
a timestamp without a timezone, or times that do not increase raise InputError.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Sequence, Union

from backend.core.errors import InputError


def parse_event_sequence(sequence: Union[str, Sequence[str]]) -> List[str]:
    """Event names from "A;B;C" text or a list. Spaces around names are removed.

    "CONFIRM_PAYMENT; PACK_ITEMS" -> ["CONFIRM_PAYMENT", "PACK_ITEMS"]

    Raises:
        InputError: Not text or a list, a name that is not text, an empty name, or no events.
    """
    if isinstance(sequence, str):
        tokens: List[Any] = sequence.split(";")
    elif isinstance(sequence, (list, tuple)):
        tokens = list(sequence)
    else:
        raise InputError("a ';'-separated string or a list of event names", type(sequence).__name__, "event_sequence")
    if not all(isinstance(t, str) for t in tokens):
        raise InputError("event names as text", tokens, "event_sequence")
    events = [t.strip() for t in tokens]
    if not events or any(e == "" for e in events):
        raise InputError("one or more non-empty event names separated by ';'", repr(sequence), "event_sequence")
    return events


def parse_timestamp(value: Union[str, datetime], where: str = "event_timestamps") -> datetime:
    """Parse an ISO-8601 timestamp such as '2026-05-24T05:30:31Z'. Returned in UTC.

    Raises:
        InputError: Not ISO-8601, or no timezone.
    """
    if isinstance(value, datetime):
        ts = value
    elif isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value.strip())
        except ValueError:
            raise InputError("an ISO-8601 timestamp such as 2026-05-24T05:30:31Z", value, where) from None
    else:
        raise InputError("an ISO-8601 timestamp string", type(value).__name__, where)
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise InputError("a timestamp with a timezone, such as 2026-05-24T05:30:31Z", value, where)
    return ts.astimezone(timezone.utc)


def parse_timestamps(values: Sequence[Union[str, datetime]]) -> List[datetime]:
    """Parse every timestamp and require each to be later than the one before.

    Raises:
        InputError: An invalid timestamp, or one not later than the previous one.
    """
    stamps = [parse_timestamp(v, f"event_timestamps[{i}]") for i, v in enumerate(values)]
    for i in range(1, len(stamps)):
        if stamps[i] <= stamps[i - 1]:
            raise InputError(f"a time later than event_timestamps[{i - 1}] ({values[i - 1]})", values[i],
                             f"event_timestamps[{i}]")
    return stamps
