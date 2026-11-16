# Dataset Specification

`backend/ml/dataset/generator.py` writes `backend/ml/dataset/data/orders_event_log.csv`. Seed 42: the
same seed always gives the identical file. Every rule below is a named constant in the generator with
the assumption behind it; this page summarises them. Labels are set by the same function that builds
each order, and every row is re-checked against the automaton (a mismatch raises `GenerateError`).

## Columns

| Column | Meaning |
| :--- | :--- |
| `trace_id` | `ORD_00001` ... |
| `split` | `train`, `validation` or `test`, assigned before any model sees the data |
| `anomaly_category` | `NORMAL`, `STATISTICAL_ANOMALY` or `RULE_VIOLATION` |
| `anomaly_type` | `NONE`, one of the five anomaly types, or one of the six violation types |
| `event_sequence` | events joined by `;` |
| `event_timestamps` | one UTC timestamp per event, joined by `;` |
| `order_amount` | order value |
| `is_rule_valid`, `is_complete` | the automaton's result (the loader checks they agree) |

## Sizes

| Split | Normal | Each anomaly type (x5) | Rule violations | Used for |
| :--- | ---: | ---: | ---: | :--- |
| train | 4,000 | 100 | 0 | fitting (normal only; anomaly rows only for the supervised Random Forest) |
| validation | 2,000 | 20 | 0 | thresholds and choosing the timing model |
| test | 5,000 | 50 | 500 | the final report, used once |

Assumption: about 5% of completed orders are anomalous (250 of 5,250 test orders), rare like a real
review queue, with 50 test examples per type.

## Normal orders

A random walk over the automaton:

| From | Next event (probability) |
| :--- | :--- |
| first payment | confirm 0.85, fail 0.10, cancel 0.05 |
| after a decline | retry 0.75, confirm (bank re-check) 0.08, cancel 0.17 |
| retry attempt | confirm 0.57, fail again 0.35, cancel 0.08 |
| paid | pack 0.95, refund 0.05 |
| packed | ship 0.97, refund 0.03 |
| shipped | deliver 1.0 |

Waiting times are log-normal (always positive, occasionally long): retry after a decline about 75 s,
packing about 8 h after payment, shipping about 5 h after packing, delivery about 3 days, refund about
3 h after payment. Amount: log-normal, median 1,200 (90% between about 320 and 4,500).
Resulting shares: 90% of orders never fail payment, 7.4% fail once, 1.9% twice; 85% are delivered,
8% cancelled, 7% refunded.

## Anomalies (legal and complete, but suspicious)

Each type overlaps normal orders on every single measurement; only the combination or the sequence is
unusual.

| Type | Rule |
| :--- | :--- |
| `CARD_TESTING` | 2-6 failed payments, retried a bit faster than people (about 35-45 s), small amount (median 250); half give up |
| `REPEATED_PAYMENT_FAILURE` | 2-6 failed payments at normal human pace and a normal amount |
| `FAST_FULFILMENT` | packing (about 2 h), shipping (about 1.2 h) and delivery (about 1.35 days) all about two standard deviations faster than normal |
| `STALLED_DELIVERY` | delivery takes about 9 days instead of about 3 |
| `HIGH_VALUE_QUICK_REFUND` | amount around 4,500 (top 5% of normal) refunded about 30 min after payment |

**No giveaway feature.** `backend/ml/dataset/audit.py` finds, for each type, the single measurement that
catches the most of it while flagging at most 1% of normal orders. A generator test fails if any type
reaches 90%. Current values: card testing 79% (failed payments), repeated failures 81% (failed
payments), fast fulfilment 37%, stalled delivery 74%, high-value refund 57%.

## Rule violations

A fulfilled order broken by one named edit, invalid by construction:
`MISSING_PAYMENT` (no payment), `SKIPPED_STEP` (no packing or no shipping), `WRONG_ORDER` (two adjacent
steps swapped), `DUPLICATE_EVENT` (payment, packing or shipping twice), `EVENT_AFTER_END` (an event after
delivery), `UNKNOWN_EVENT` (a name outside the alphabet).
