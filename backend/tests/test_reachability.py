"""Tests for reachability analysis of the workflow automaton (backend/fsm/reachability.py),
with no patching: a broken automaton is passed in as a table.

Expected values are worked out by hand from the 11 transitions in fsm/transitions.py.

Run with:
    .venv\\Scripts\\python.exe -m unittest backend.tests.test_reachability -v
"""

from __future__ import annotations

import unittest

from backend.fsm import reachability
from backend.fsm.reachability import (
    analyze_reachability,
    count_accepted_by_length,
    legal_step_kinds,
    possible_outcomes,
    shortest_paths_from,
    states_that_can_finish,
    strongly_connected_components,
)
from backend.fsm.states import ALL_STATES
from backend.fsm.validator import validate_trace


class TestLegalStepKinds(unittest.TestCase):
    """Pairs of events that can follow each other in a valid order (worked out by hand)."""

    def test_the_eleven_pairs(self) -> None:
        self.assertEqual(legal_step_kinds(), sorted([
            ("FAIL_PAYMENT", "RETRY_PAYMENT"), ("FAIL_PAYMENT", "CONFIRM_PAYMENT"), ("FAIL_PAYMENT", "CANCEL_ORDER"),
            ("RETRY_PAYMENT", "CONFIRM_PAYMENT"), ("RETRY_PAYMENT", "FAIL_PAYMENT"), ("RETRY_PAYMENT", "CANCEL_ORDER"),
            ("CONFIRM_PAYMENT", "PACK_ITEMS"), ("CONFIRM_PAYMENT", "REFUND_ORDER"),
            ("PACK_ITEMS", "SHIP_ITEMS"), ("PACK_ITEMS", "REFUND_ORDER"),
            ("SHIP_ITEMS", "DELIVER_ITEMS"),
        ]))

    def test_every_pair_in_a_valid_order_is_listed(self) -> None:
        events = ["FAIL_PAYMENT", "RETRY_PAYMENT", "CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"]
        self.assertTrue(validate_trace(events)["is_valid"])
        self.assertTrue(set(zip(events, events[1:])) <= set(legal_step_kinds()))


class TestReachability(unittest.TestCase):

    def test_every_state_is_reachable_from_order_created(self) -> None:
        self.assertEqual(set(shortest_paths_from("ORDER_CREATED")), set(ALL_STATES))

    def test_shortest_paths_are_minimal_and_valid(self) -> None:
        paths = shortest_paths_from("ORDER_CREATED")
        self.assertEqual(paths["ORDER_CREATED"], [])
        self.assertEqual(paths["CANCELLED"], ["CANCEL_ORDER"])
        self.assertEqual(paths["PAYMENT_FAILED"], ["FAIL_PAYMENT"])
        self.assertEqual(paths["REFUNDED"], ["CONFIRM_PAYMENT", "REFUND_ORDER"])
        self.assertEqual(paths["DELIVERED"], ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"])
        for state, path in paths.items():
            result = validate_trace(path)
            self.assertTrue(result["is_valid"], state)
            self.assertEqual(result["final_state"], state)

    def test_every_state_can_still_finish_so_there_are_no_deadlocks(self) -> None:
        self.assertEqual(states_that_can_finish(), set(ALL_STATES))
        report = analyze_reachability()
        self.assertEqual(report["deadlock_states"], [])
        self.assertTrue(report["is_deadlock_free"])

    def test_the_only_cycle_is_the_payment_retry_loop(self) -> None:
        cycles = [c for c in strongly_connected_components() if len(c) > 1]
        self.assertEqual(cycles, [["ORDER_CREATED", "PAYMENT_FAILED"]])

    def test_accepted_sequences_by_length(self) -> None:
        # length 1: X | 2: C R, F X | 3: C P R, F C R, F Rt X
        # length 4: C P S D, F C P R, F Rt C R, F Rt F X | then 4 for every longer length
        self.assertEqual(count_accepted_by_length(8), [0, 1, 2, 3, 4, 4, 4, 4, 4])
        self.assertTrue(analyze_reachability()["language_is_infinite"])

    def test_possible_outcomes_from_an_unfinished_order(self) -> None:
        self.assertEqual(possible_outcomes("PACKED"), {
            "DELIVERED": ["SHIP_ITEMS", "DELIVER_ITEMS"],
            "REFUNDED": ["REFUND_ORDER"],
        })
        self.assertEqual(possible_outcomes("SHIPPED"), {"DELIVERED": ["DELIVER_ITEMS"]})
        self.assertEqual(possible_outcomes("DELIVERED"), {"DELIVERED": []})

    def test_detects_a_deadlock_when_one_exists(self) -> None:
        # Remove SHIPPED -> DELIVERED: SHIPPED can no longer finish.
        broken = {k: v for k, v in reachability.TRANSITION_TABLE.items() if k != ("SHIPPED", "DELIVER_ITEMS")}
        report = analyze_reachability(table=broken)
        self.assertEqual(report["deadlock_states"], ["SHIPPED"])
        self.assertFalse(report["is_deadlock_free"])
        self.assertIn("DELIVERED", report["unreachable_states"])

    def test_no_cycle_means_finite_language(self) -> None:
        broken = {k: v for k, v in reachability.TRANSITION_TABLE.items() if k != ("PAYMENT_FAILED", "RETRY_PAYMENT")}
        report = analyze_reachability(max_length=10, table=broken)
        self.assertEqual(report["cycles"], [])
        self.assertFalse(report["language_is_infinite"])
        self.assertEqual(report["accepted_sequences_by_length"][-1], 0)


if __name__ == "__main__":
    unittest.main()
