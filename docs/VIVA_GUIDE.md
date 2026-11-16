# Viva Guide: how the codebase produces its results

Problem statement 9: *Anomaly Detection in Process Workflows Using Automata and ML.* This guide
follows one path through the code, from the data to the dashboard, and gives the real numbers.
Every number here can be reproduced with `python -m backend.ml.pipeline` (fixed seed 42).

---

## 1. The project in four sentences

1. Online-shop orders are modelled as a **deterministic finite automaton (DFA)**: an order is a string
   of events, and the automaton decides whether the string follows the workflow rules.
2. Orders that break a rule are **rule violations**; legal orders that have not finished are **in progress**.
3. Legal, finished orders are checked by **two machine-learning models** trained on normal orders: a
   **Markov chain** (is this order of events rare?) and a **timing-and-amount model** (are the times,
   amount and outcome unusual together?). If either flags the order, it is a **statistical anomaly**.
4. Every verdict comes with an explanation, and a dashboard shows the order's path on the automaton,
   the unusual transitions and the anomaly scores.

### How the PS9 objectives map to the code

| PS9 objective | Implementation |
| :--- | :--- |
| Workflow states, events, transition rules | `backend/fsm/states.py`, `transitions.py` |
| Generate or collect normal and anomalous logs | `backend/ml/dataset/generator.py` (generated, seeded) |
| Finite-state model to find invalid transitions | `backend/fsm/validator.py` |
| ML for statistical or sequential anomalies | `backend/ml/markov/` (sequential), `backend/ml/timing/` (statistical) |
| Combine rule violations with ML scores | `backend/pipeline/analyzer.py` |
| Evaluate, explain each flag | `backend/ml/evaluation/evaluate.py`, explanations in `analyzer.py` |
| TOC concepts: state-transition systems, DFA, regular languages, reachability | `backend/fsm/` incl. `reachability.py` |
| ML techniques: Isolation Forest, clustering, Random Forest, sequence models | all four are trained and compared |
| Dashboard: paths, unusual transitions, anomaly scores | `frontend/` |

---

## 2. The automaton (TOC part)

**M = (Q, Σ, δ, q₀, F)** in `backend/fsm/`:

- **Q** (8 states): ORDER_CREATED, PAYMENT_FAILED, PAID, PACKED, SHIPPED, DELIVERED, CANCELLED, REFUNDED
- **Σ** (8 events): CONFIRM_PAYMENT, FAIL_PAYMENT, RETRY_PAYMENT, CANCEL_ORDER, PACK_ITEMS, SHIP_ITEMS, DELIVER_ITEMS, REFUND_ORDER
- **q₀** = ORDER_CREATED
- **F** (final/accepting) = DELIVERED, CANCELLED, REFUNDED
- **δ**: 11 allowed (state, event) pairs, in `TRANSITION_TABLE`:

| From | Event | To |
| :--- | :--- | :--- |
| ORDER_CREATED | CONFIRM_PAYMENT / FAIL_PAYMENT / CANCEL_ORDER | PAID / PAYMENT_FAILED / CANCELLED |
| PAYMENT_FAILED | RETRY_PAYMENT / CONFIRM_PAYMENT / CANCEL_ORDER | ORDER_CREATED / PAID / CANCELLED |
| PAID | PACK_ITEMS / REFUND_ORDER | PACKED / REFUNDED |
| PACKED | SHIP_ITEMS / REFUND_ORDER | SHIPPED / REFUNDED |
| SHIPPED | DELIVER_ITEMS | DELIVERED |

δ is a **partial** transition function: any pair not in the table is a rule violation. (Adding a
"dead/trap state" for every missing pair would make it a complete DFA; the behaviour is the same.)

### How an order is checked (`validator.py: validate_trace`)

Start in q₀ and read the events one by one. For each event, in this order of precedence:

1. If the current state is final → **TERMINAL_STATE_VIOLATION** (something happened after the order ended).
2. If the event is not in Σ → **UNKNOWN_EVENT**.
3. If δ(state, event) is undefined → **INVALID_TRANSITION**.
4. Otherwise move to δ(state, event).

The result records whether the order is valid, whether it ended in a final state (complete), the
exact failing step, and the events that *were* allowed there.

- Example: `CONFIRM_PAYMENT, SHIP_ITEMS` fails at step 2: SHIP_ITEMS is not allowed in PAID; allowed: PACK_ITEMS, REFUND_ORDER.
- Example: `CONFIRM_PAYMENT, PACK_ITEMS` is valid but **incomplete** (still in PACKED).

### Reachability analysis (`reachability.py`), computed by graph search

| Question | Method | Result |
| :--- | :--- | :--- |
| Which states can be reached from q₀? | breadth-first search | all 8 |
| From which states can a final state still be reached? | backward search from F | all 8 |
| Deadlocks (reachable, not final, cannot finish)? | reachable minus can-finish | **none**: every order can always still end |
| Cycles? | strongly connected components | one: **ORDER_CREATED ↔ PAYMENT_FAILED** (the payment retry loop) |
| Shortest way to each state | BFS keeps the first path | e.g. DELIVERED: CONFIRM_PAYMENT → PACK_ITEMS → SHIP_ITEMS → DELIVER_ITEMS |
| Valid complete orders of each length 0, 1, 2, ... | dynamic programming over states | 0, 1, 2, 3, 4, 4, 4, 4, ... |
| Is the language finite? | a reachable cycle that can still finish | **infinite** (the retry loop can repeat any number of times), but **regular**, because a finite automaton accepts it |

The dashboard's **Workflow & reachability** page shows all of this, and *Play this path* animates
the shortest route to any state.

---

## 3. The data (`backend/ml/dataset/`)

PS9 says "generate or collect". Real order logs do not come with anomaly labels, so the data is
**generated** by `generator.py` with a fixed seed (42): the same file every time.

### Normal orders

A random walk over the automaton with written-down branch probabilities, for example:

| In state | Next event (probability) |
| :--- | :--- |
| ORDER_CREATED (first payment) | confirm 0.85, fail 0.10, cancel 0.05 |
| PAYMENT_FAILED | retry 0.75, confirm 0.08, cancel 0.17 |
| ORDER_CREATED (after a retry) | confirm 0.57, fail again 0.35, cancel 0.08 |
| PAID | pack 0.95, refund 0.05 |
| PACKED | ship 0.97, refund 0.03 |

Times between events are **log-normal** (always positive, sometimes much longer than usual): packing
≈ 8 h after payment, shipping ≈ 5 h, delivery ≈ 3 days. Amounts are log-normal with median 1,200.

### Five anomaly types (legal and complete, but suspicious)

| Type | What makes it suspicious | Which model should see it |
| :--- | :--- | :--- |
| Card testing | 2–6 failed payments, retried a bit faster than people, small amount | Markov (many failures) + timing |
| Repeated payment failure | 2–6 failed payments at normal pace | Markov (event order) |
| Fast fulfilment | packing, shipping and delivery all faster than normal at once | timing |
| Stalled delivery | delivery ≈ 9 days instead of ≈ 3 | timing |
| High-value quick refund | large amount refunded ≈ 30 min after payment | timing (combination of amount + outcome + time) |

**Key design rule: no giveaway feature.** Each type overlaps normal orders on every single
measurement, so a simple threshold cannot catch it; only combinations or sequences are unusual.
`audit.py` checks this: for each type, the best single measurement (at ≤ 1% false alarms) catches
only 37–81%, and a test fails if any type reaches 90%.

*Why this matters:* an earlier version had anomalies far outside normal ranges (e.g. amounts of
40,000+). A plain "outside the normal range" rule then beat the ML, which made the ML pointless.

### Rule violations

A normal fulfilled order broken by one named edit: missing payment, skipped step, wrong order,
duplicate event, event after the end, unknown event. Each is invalid by construction.

### Splits, made before any training

| Split | Normal | Anomalies | Used for |
| :--- | ---: | ---: | :--- |
| train | 4,000 | 500 | fitting the models (normal orders only) |
| validation | 2,000 | 100 | setting thresholds, choosing the timing model |
| test | 5,000 | 250 (+500 violations) | the final results, used once |

About 5% of test orders are anomalies: rare, like a real review queue.

---

## 4. Model 1: Markov chain, for the order of events (`backend/ml/markov/model.py`)

A **probabilistic finite automaton**: the same states and transitions as the DFA, but each allowed
transition gets a probability learned from normal training orders.

```
P(event | state) = (count(state, event) + α) / (count(state) + α · k)      α = 1, k = events allowed in that state
P(order)         = product of P(event | state) over its steps
```

α = 1 (**Laplace smoothing**) gives allowed-but-never-seen transitions a small non-zero probability.

Learned values, for example: P(CONFIRM_PAYMENT | ORDER_CREATED) = 0.826, P(FAIL_PAYMENT |
ORDER_CREATED) = 0.117, P(DELIVER_ITEMS | SHIPPED) = 1.000.

**Decision:** compute log₁₀ P(path) for the 2,000 normal *validation* orders; the threshold is the
**1st percentile** of those values (≈ "rarer than 1 in 1,990 orders"). An order whose path is rarer
is flagged.

```
path score = threshold − log₁₀ P(path)        > 0 means flagged; +1 means 10× rarer than the threshold
```

Example: 5 failed payments, then a normal delivery. P(FAIL_PAYMENT | ORDER_CREATED) = 0.123 five
times gives 0.123⁵ ≈ 2.9 × 10⁻⁵ for that step alone. The whole path occurs in about 1 in 292,000
normal orders → flagged.

**What it cannot see:** times and amounts. A normal-looking path that happened too fast looks normal to it.

---

## 5. Model 2: timing and amount (`backend/ml/timing/`)

### Features (`features.py`)

From normal training orders the code learns, for the amount and for each of the 11 kinds of step
(e.g. PACK_ITEMS → SHIP_ITEMS), the mean and standard deviation of ln(value). A value's **z-score** is
how many standard deviations its logarithm is from normal. Each order becomes **7 numbers**:

| Input | Meaning |
| :--- | :--- |
| amount_z | how unusual the amount is |
| slowest_step_z | the step that was slowest compared with its usual time |
| fastest_step_z | the step that was fastest compared with its usual time |
| mean_step_z | is the whole order unusually fast or slow? |
| payment_failures | number of failed payments |
| refunded, cancelled | how the order ended (0 or 1) |

The 11 kinds of step are derived from the automaton itself: pairs of events that can follow each
other in a valid order.

### Three candidates (PS9's "Isolation Forest" and "clustering", plus LOF)

All are trained on normal orders only (unsupervised), on standardized inputs:

| Candidate | Idea | Score (higher = more unusual) |
| :--- | :--- | :--- |
| Isolation Forest (200 trees) | random splits isolate unusual points in fewer steps | negative of sklearn's score |
| **k-means clustering** (k chosen by silhouette score → 5 clusters) | normal orders form groups | distance to the nearest cluster centre |
| Local Outlier Factor (20 neighbours) | unusual points sit in sparser regions than their neighbours | negative of sklearn's score |

Each threshold = the **99th percentile** of the scores of normal validation orders, so about 1% of
normal orders are flagged.

### Choosing one, on the validation split only

| Candidate | Validation F1 |
| :--- | ---: |
| Isolation Forest | 0.400 |
| **k-means clustering** | **0.537** (chosen) |
| Local Outlier Factor | 0.389 |

Rule fixed in advance: highest validation F1, preferring a PS9-named model if it is within 0.02 of
the best. The test split was not used for this choice.

---

## 6. Combining everything (`backend/pipeline/analyzer.py`)

```
order ─► FSM ─┬─ invalid ─────────────► RULE_VIOLATION        (severity CRITICAL; no ML: the rules decide)
              ├─ valid, not finished ─► VALID_INCOMPLETE      (INFO; lists the endings still reachable)
              └─ valid, finished ─────► Markov chain  ┐
                                         timing model ┴─ either flags ─► STATISTICAL_ANOMALY (WARNING)
                                                        neither flags ─► NORMAL (INFO)
```

The timing model runs only when timestamps and the amount are supplied.

**Why ML is not used for rule violations:** the automaton already decides them exactly. Measuring
the automaton on violations would be circular, since a violation *is* what the automaton rejects.

### The explanation

Every result has a summary, a root cause, key factors and a recommended action:

- **Unusual step** (amber on the diagram): a kind of step whose probability, multiplied over all its
  occurrences in the order, is below 10%. Either one rare step (e.g. 0.03), or a common step repeated
  (0.12 five times).
- **Unusual value**: its **percentile** among normal orders is outside the middle 95%, e.g.
  "took 1.2 h, faster than 98.1% of normal orders".
- **Combination**: when no single value is unusual but the timing model still flags the order, the
  explanation says so and lists the values furthest from normal.

---

## 7. Results (`backend/ml/evaluation/evaluate.py`, test split, used once)

Metrics, with the positive class = anomaly:

- **Precision**: share of alarms that are real anomalies.
- **Recall**: share of anomalies caught.
- **F1**: 2 · P · R / (P + R), high only when both are high. This is the headline metric.
  (Accuracy is misleading here: always answering "normal" is already 95% accurate.)

| Model | F1 | Precision | Recall | Caught / 250 | False alarms / 5,000 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Always "normal" (trivial baseline) | 0.000 | – | 0.000 | 0 | 0 |
| Range rule, no ML (outside training min–max) | 0.335 | 0.633 | 0.228 | 57 | 33 |
| Markov chain alone | 0.353 | 0.589 | 0.252 | 63 | 44 |
| Isolation Forest alone | 0.271 | 0.517 | 0.184 | 46 | 43 |
| k-means alone | 0.466 | 0.662 | 0.360 | 90 | 46 |
| LOF alone | 0.338 | 0.596 | 0.236 | 59 | 40 |
| **Markov + k-means (the app)** | **0.470** | 0.554 | **0.408** | **102** | 82 |
| Random Forest (supervised comparison) | 0.646 | 0.712 | 0.592 | 148 | 60 |

Caught per type (of 50 each):

| Type | Range rule | Markov | k-means | **App** | Random Forest |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Card testing | 19 | 32 | 28 | **34** | 49 |
| Repeated failure | 10 | 31 | 25 | **31** | 11 |
| Fast fulfilment | 1 | 0 | 12 | **12** | 42 |
| Stalled delivery | 13 | 0 | 2 | **2** | 0 |
| High-value refund | 14 | 0 | 23 | **23** | 46 |

How to read this:

1. **The ML beats the no-ML baseline**: F1 0.470 vs 0.335; 102 anomalies caught vs 57.
2. **The two models complement each other**: the Markov chain catches the sequence anomalies (card
   testing, repeated failures) and nothing else; k-means adds the timing and amount anomalies.
3. **The Random Forest scores higher only because it was given labels** for three types (card testing,
   fast fulfilment, high-value refund). On the two types it never saw, it caught 11 and **0**.
   Real systems rarely have labels for every new kind of anomaly, so the app uses unsupervised models.
4. **Honest weakness:** stalled delivery is caught poorly (2/50); the range rule does better there
   (13/50), because it is a single extreme value and the multi-feature model weighs it less.
5. All 500 rule violations are rejected by the automaton. That is by definition, not a detection
   rate: it checks the generated data.

The dashboard's **Models & results** page reads these numbers from `results.json`; nothing is typed in.

---

## 8. The rest of the system

- **API** (`backend/main.py`, `api/routes.py`, FastAPI): analyze one order or a batch, history,
  metrics, the automaton, the models report, real example orders, the demo stream.
- **Database** (SQLite via SQLAlchemy): every analysis is stored. Each result is validated against a
  schema before saving (e.g. the verdict must follow from the FSM and model results).
- **Dashboard** (React): live demo stream, analysis-flow animation, step-by-step path playback, history.
- **Errors**: every step raises its own named error with one message format, e.g.
  `[MODEL] expected a saved Markov model, got no file; input=...markov_chain.json`. The API returns it
  with the right HTTP status and the dashboard shows it. Nothing fails silently.
- **Tests**: `run_tests.py` runs 193 backend + 24 frontend tests, with no mocks of the code under test.
  They include an end-to-end test from an empty folder (generate → train → evaluate → API) and a test
  that the app's verdicts reproduce the reported numbers exactly.

---

## 9. Live demo plan (about 5 minutes)

1. **Workflow & reachability**: explain M = (Q, Σ, δ, q₀, F); point out no deadlocks, the retry
   cycle, and the infinite but regular language. Click DELIVERED → *Play this path*.
2. **Overview → Start stream**: 100 test orders flow through. Point at the counters (caught, false
   alarms, violations rejected). Click an order to open its explanation.
3. **Analyze**: pick "Skipped a step" → rule violation, red step, allowed events. Then a caught
   anomaly (e.g. "Repeated failures" or "High-value quick refund") → amber steps, scores, *How the
   verdict was reached*.
4. **Models & results**: the headline table, the validation choice, the "no giveaway feature" check.

Some example orders are honest misses (the generator made an anomaly but the system says Normal).
The page says so and quotes the test numbers. Use one to show the system's limits openly.

---

## 10. Likely questions and answers

**Why a finite automaton for the rules?** The workflow is a fixed set of states with allowed moves,
which is exactly a DFA. Checking an order is reading a string: linear time, exact, and it names the
failing step.

**Is the language of valid orders regular? Finite?** Regular, because a DFA accepts it. Infinite,
because the cycle ORDER_CREATED ↔ PAYMENT_FAILED can repeat any number of times and still finish;
there are 4 valid orders of every length ≥ 4.

**What is reachability used for?** To prove no state is unreachable or a deadlock (every order can
always still finish), to show the shortest way to each state, and to list the endings still possible
for an in-progress order.

**Why is a Markov chain "machine learning"?** Its transition probabilities are learned from data
(maximum-likelihood counts with Laplace smoothing), and it predicts how likely a new sequence is. It
is a sequence model, one of PS9's listed techniques.

**Why unsupervised models?** In practice labelled anomalies are rare and new kinds keep appearing.
The Random Forest result shows the cost of relying on labels: 0/50 on a type it never saw.

**How were thresholds chosen?** From normal *validation* orders, never the test set: the Markov chain
flags paths rarer than 99% of normal paths; the timing model flags orders more unusual than 99% of
normal orders. So each flags about 1% of normal orders by design.

**How do you know there is no data leakage?** Splits are assigned in the generator before any
training. Models fit on train only; thresholds and model choice use validation only; the test split
is scored once. Labels are never model inputs.

**Why is F1 only 0.47?** The anomalies are deliberately close to normal behaviour; an earlier version
with obvious anomalies scored about 0.98, but a simple rule did just as well, which proved nothing.
0.47 against 0.335 for the no-ML rule is an honest, meaningful improvement.

**Why synthetic data?** PS9 allows "generate or collect". Real logs have no anomaly labels to evaluate
against, and do not match this workflow's states. The generator's rules are written down, seeded and
tested, so results are reproducible.

**Why k-means and not Isolation Forest?** All three were tried; k-means had the best validation F1
(0.537 vs 0.400 vs 0.389), and that rule was fixed before looking at test results.

**What would you improve?** Real data with injected anomalies; a second-order Markov chain (to tell a
first payment from a retry); a dedicated detector for single extreme values such as stalled delivery.
