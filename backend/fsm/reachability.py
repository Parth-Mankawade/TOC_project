"""Reachability analysis of the order workflow automaton.

All answers come from graph search over the transition table (the 11 transitions
in transitions.py unless another table is passed, e.g. a deliberately broken one in tests):

  reachable states       which states can be reached from ORDER_CREATED (forward BFS)
  can finish             from which states a terminal state can still be reached (backward search)
  deadlocks              reachable, non-terminal states from which no terminal state is reachable
  shortest witness       the shortest event sequence that reaches each state (BFS keeps the first path found)
  cycles                 strongly connected components with more than one state (the payment retry loop)
  accepted by length     how many valid, complete event sequences of each length exist; a non-zero
                         count at every large length means the language is infinite (because of the cycle)

If every reachable state can finish, the workflow has no deadlocks: whatever legal events
happen, an order can always still be completed, cancelled or refunded.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Mapping, Set, Tuple

from backend.fsm.states import ALL_STATES, INITIAL_STATE, TERMINAL_STATES
from backend.fsm.transitions import TRANSITION_TABLE

Table = Mapping[Tuple[str, str], str]

# Deterministic order everywhere, so results never depend on dict iteration order.
_STATES: List[str] = sorted(ALL_STATES)


def successors(state: str, table: Table = TRANSITION_TABLE) -> List[Tuple[str, str]]:
    """(event, next_state) pairs leaving `state`, sorted by event name."""
    return sorted((event, nxt) for (s, event), nxt in table.items() if s == state)


def legal_step_kinds(table: Table = TRANSITION_TABLE) -> List[Tuple[str, str]]:
    """Every (event, next event) pair that can occur in a valid order, sorted.

    (e1, e2) is legal when some transition q --e1--> q' exists and e2 is allowed in q'.
    These are exactly the kinds of step whose duration the timing model learns.
    """
    return sorted({(e1, e2) for (_, e1), q in table.items() for (s, e2) in table if s == q})


def shortest_paths_from(start: str, table: Table = TRANSITION_TABLE) -> Dict[str, List[str]]:
    """Every state reachable from `start`, with the shortest event sequence that reaches it."""
    paths: Dict[str, List[str]] = {start: []}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for event, nxt in successors(state, table):
            if nxt not in paths:
                paths[nxt] = paths[state] + [event]
                queue.append(nxt)
    return paths


def states_that_can_finish(table: Table = TRANSITION_TABLE) -> Set[str]:
    """States from which some terminal state is reachable (backward search from the terminal states)."""
    can_finish = set(TERMINAL_STATES)
    changed = True
    while changed:
        changed = False
        for (state, _event), nxt in table.items():
            if nxt in can_finish and state not in can_finish:
                can_finish.add(state)
                changed = True
    return can_finish


def strongly_connected_components(table: Table = TRANSITION_TABLE) -> List[List[str]]:
    """Groups of states that can all reach each other; singletons are included."""
    reach = {s: set(shortest_paths_from(s, table)) for s in _STATES}
    components: List[List[str]] = []
    seen: Set[str] = set()
    for s in _STATES:
        if s in seen:
            continue
        component = sorted(t for t in _STATES if t in reach[s] and s in reach[t])
        seen.update(component)
        components.append(component)
    return components


def count_accepted_by_length(max_length: int, table: Table = TRANSITION_TABLE) -> List[int]:
    """counts[n] = number of valid event sequences of length n that end in a terminal state."""
    current = {s: 0 for s in _STATES}
    current[INITIAL_STATE.value] = 1
    counts = [1 if INITIAL_STATE.value in TERMINAL_STATES else 0]
    for _ in range(max_length):
        nxt = {s: 0 for s in _STATES}
        for state, ways in current.items():
            if ways and state not in TERMINAL_STATES:
                for _event, to_state in successors(state, table):
                    nxt[to_state] += ways
        current = nxt
        counts.append(sum(current[s] for s in TERMINAL_STATES))
    return counts


def possible_outcomes(state: str, table: Table = TRANSITION_TABLE) -> Dict[str, List[str]]:
    """Terminal states still reachable from `state`, each with the shortest event sequence to it."""
    paths = shortest_paths_from(state, table)
    return {t: paths[t] for t in sorted(TERMINAL_STATES) if t in paths}


def analyze_reachability(max_length: int = 12, table: Table = TRANSITION_TABLE) -> Dict[str, Any]:
    """Full report used by the API and the documentation."""
    initial = INITIAL_STATE.value
    witnesses = shortest_paths_from(initial, table)
    can_finish = states_that_can_finish(table)
    reachable = [s for s in _STATES if s in witnesses]
    deadlocks = [s for s in reachable if s not in can_finish]
    cycles = [c for c in strongly_connected_components(table) if len(c) > 1]
    return {
        "initial_state": initial,
        "reachable_states": reachable,
        "unreachable_states": [s for s in _STATES if s not in witnesses],
        "states_that_can_finish": sorted(can_finish),
        "deadlock_states": deadlocks,
        "is_deadlock_free": not deadlocks,
        "shortest_paths": {s: witnesses[s] for s in reachable},
        "cycles": cycles,
        "accepted_sequences_by_length": count_accepted_by_length(max_length, table),
        # Infinitely many valid complete sequences exist iff some reachable cycle can still finish:
        # the cycle can be repeated any number of times before finishing.
        "language_is_infinite": any(
            any(s in witnesses for s in c) and any(s in can_finish for s in c) for c in cycles
        ),
    }
