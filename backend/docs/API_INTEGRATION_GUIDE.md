# API Integration Guide

Base URL: `http://127.0.0.1:8000`. All bodies are JSON. Interactive docs: `/docs`.
Every example below is a real response from this backend (long arrays shortened with `...`).

## 1. Endpoints

| Method | Path | What it does |
| :--- | :--- | :--- |
| GET | `/health`, `/api/v1/health/live` | The process is running: `{"status": "alive"}` |
| GET | `/api/v1/health`, `/api/v1/health/ready` | 200 only if the database table exists and both model files load; otherwise 503 saying which |
| POST | `/api/v1/analyses` | Analyze one order and store the result (201) |
| POST | `/api/v1/analyses/batch` | Analyze 1-500 orders, stored all-or-nothing (201) |
| GET | `/api/v1/analyses?skip=&limit=&verdict=&is_anomaly=` | Stored analyses, newest first, with the total count |
| GET | `/api/v1/analyses/{id}` | One stored analysis (404 if none) |
| GET | `/api/v1/orders/{order_id}/analyses` | Every analysis of one order, newest first |
| GET | `/api/v1/metrics` | Counts per verdict over everything stored (`anomaly_rate` is null when nothing is stored) |
| GET | `/api/v1/fsm` | The automaton (states, events, transitions) and its reachability analysis |
| GET | `/api/v1/models` | Both saved models, the evaluation report, and the unusual-step limit |
| GET | `/api/v1/examples` | One real test-split order of every kind, with the generator's label |
| GET | `/api/v1/demo` | The 100-order demo stream (not analyzed) and its composition |
| POST | `/api/v1/demo` | Analyze and store the demo stream at once; returns the batch summary |

## 2. Request

```json
{
  "order_id": "ORD-2",
  "event_sequence": ["CONFIRM_PAYMENT", "PACK_ITEMS", "SHIP_ITEMS", "DELIVER_ITEMS"],
  "event_timestamps": ["2026-03-01T09:00:00Z", "2026-03-01T17:10:00Z", "2026-03-01T22:05:00Z", "2026-03-04T21:30:00Z"],
  "order_amount": 1250.0
}
```

- `event_sequence`: required; `"A;B;C"` text or a list of names. No empty names. Stored exactly as sent.
- `order_id`: optional, up to 64 characters; groups analyses of one order.
- `event_timestamps` and `order_amount`: optional, but **both or neither**. One ISO-8601 timestamp with a
  timezone per event, strictly increasing; amount > 0. With them, the timing-and-amount check runs.
- Batch: `{"traces": [ <request object> | "A;B;C" | ["A", "B"] , ... ]}`.

## 3. Response: one analysis

Rule violation (`"CONFIRM_PAYMENT;SHIP_ITEMS"`):

```json
{
  "id": 1,
  "order_id": "ORD-1",
  "verdict": "RULE_VIOLATION",
  "is_anomaly": true,
  "anomaly_type": "RULE_VIOLATION",
  "summary": "Rule violation at step 2: 'SHIP_ITEMS' is not allowed in state 'PAID'. Allowed: PACK_ITEMS, REFUND_ORDER.",
  "fsm_evaluation": {
    "is_rule_valid": false, "is_complete": false, "final_state": "PAID", "total_steps": 2, "successful_transitions": 1,
    "error": {
      "error_type": "INVALID_TRANSITION", "failed_step": 2, "failed_event": "SHIP_ITEMS", "state_at_failure": "PAID",
      "allowed_events": ["PACK_ITEMS", "REFUND_ORDER"],
      "message": "Illegal transition: Event 'SHIP_ITEMS' is not permitted in state 'PAID'. Permitted events are: PACK_ITEMS, REFUND_ORDER."
    }
  },
  "ml_evaluation": null,
  "ml_skip_reason": "Path check skipped: the order breaks a workflow rule (RULE_VIOLATION).",
  "timing_evaluation": null,
  "timing_skip_reason": "Timing check skipped: the order breaks a workflow rule (RULE_VIOLATION).",
  "diagnostics": {
    "severity": "CRITICAL",
    "root_cause": "Rule violation at step 2: 'SHIP_ITEMS' is not allowed in state 'PAID'. Allowed: PACK_ITEMS, REFUND_ORDER.",
    "key_factors": ["Error type: INVALID_TRANSITION", "Failed step: 2", "..."],
    "recommended_action": "Check why 'SHIP_ITEMS' was recorded at step 2; the workflow does not allow it there."
  },
  "input_format": "DELIMITED_STRING", "raw_input_payload": "CONFIRM_PAYMENT;SHIP_ITEMS",
  "event_timestamps": null, "order_amount": null, "created_at": "2026-11-16T10:00:00Z"
}
```

A legal, complete order with timing data (the request above) also has both model results:

```json
{
  "verdict": "NORMAL",
  "summary": "Normal: the order finished in 'DELIVERED'; this event order occurs in 75.8% of orders under normal behaviour, and its times, amount and outcome are within normal behaviour.",
  "ml_evaluation": {
    "prediction": "NORMAL", "is_ml_anomaly": false, "anomaly_score": -3.701, "path_log10_probability": -0.120,
    "transitions": [
      {"step": 1, "from_state": "ORDER_CREATED", "event": "CONFIRM_PAYMENT", "to_state": "PAID", "probability": 0.826, "is_unusual": false},
      "..."
    ]
  },
  "timing_evaluation": {
    "prediction": "NORMAL", "is_timing_anomaly": false, "anomaly_score": -1.032, "model_name": "kmeans",
    "checks": [
      {"name": "cluster", "score": -1.032, "is_flagged": false},
      {"name": "trend", "score": -1.480, "is_flagged": false},
      {"name": "deviation", "score": -1.349, "is_flagged": false}
    ],
    "trend_z": -0.038,
    "inputs": {"amount_z": 0.055, "slowest_step_z": 0.015, "fastest_step_z": -0.044, "mean_step_z": -0.022,
               "payment_failures": 0.0, "refunded": 0.0, "cancelled": 0.0},
    "measurements": [
      {"name": "order_amount", "step": null, "unit": "currency", "value": 1250.0,
       "normal_low": 250.12, "normal_high": 5721.36, "percentile": 52.19, "is_unusual": false},
      "... one row for failed payments and one per step duration ..."
    ]
  }
}
```

Field rules (checked by the schema before anything is stored or returned):

- `verdict`: `RULE_VIOLATION` (FSM rejected), `VALID_INCOMPLETE` (legal, unfinished), `STATISTICAL_ANOMALY`
  (legal, complete, flagged by either model), `NORMAL`.
- `anomaly_score` > 0 means that check flagged the order, for both models. Both models share one
  threshold, set so that all checks together flag 1.5% of normal validation orders.
- `timing_evaluation.checks`: always `cluster`, `trend`, `deviation`, in that order. `score` is the
  check's level minus the threshold (level 0 = a typical normal order, level 1 = the point 1% of normal
  validation orders exceed); `is_flagged` is `score > 0`. `timing_evaluation.anomaly_score` is the
  highest of the three scores. `trend_z` is negative when the steps were faster than normal overall.
- `ml_evaluation` and `ml_skip_reason`: exactly one is non-null; the same for `timing_*`.
- `is_unusual` on a transition: that kind of step's probability, multiplied over all its occurrences in
  the order, is below 10%. On a measurement: its percentile among normal orders is outside 2.5-97.5.

## 4. Errors

Every error has the same shape, and the message names the step that failed, what was expected,
what arrived and which input caused it:

```json
{
  "error": {
    "code": "INPUT_ERROR",
    "checkpoint": "INPUT",
    "message": "[INPUT] expected event_timestamps and order_amount together (both or neither), got event_timestamps given, order_amount missing; input=request",
    "details": ["[INPUT] expected event_timestamps and order_amount together (both or neither), got event_timestamps given, order_amount missing; input=request"]
  }
}
```

| Status | When | `checkpoint` |
| :--- | :--- | :--- |
| 422 | Bad request: wrong types, empty event name, timing inputs not together, bad timestamps, invalid filter | `INPUT` |
| 404 | No such record, unknown path (405: wrong method) | `API` |
| 503 | A model file, the dataset, the evaluation report or the database table is missing or unreadable | `MODEL`, `LOAD`, `EVALUATE`, `STORAGE` |
| 500 | A bug; the real exception text is kept in the message | `ANALYZE`, `FEATURES`, `API` |

`details` lists every problem when a request has more than one (otherwise it repeats the message or is null).
Rule violations are **not** errors: they are a verdict, returned with 201.
