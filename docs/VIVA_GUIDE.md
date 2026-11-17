# Viva Guide: how the codebase produces its results

Problem statement 9: *Anomaly Detection in Process Workflows Using Automata and ML.* This guide
follows one path through the code, from the data to the dashboard, and gives the real numbers.
Every number here can be reproduced with `python -m backend.ml.pipeline` (fixed seed 42) and, for the
five-seed tables, `python -m backend.ml.evaluation.benchmark`. The result tables in section 7 are
written into this file from the two JSON reports by `python -m backend.ml.evaluation.report`.

---

## 1. The project in four sentences

1. Online-shop orders are modelled as a **deterministic finite automaton (DFA)**: an order is a string
   of events, and the automaton decides whether the string follows the workflow rules.
2. Orders that break a rule are **rule violations**; legal orders that have not finished are **in progress**.
3. Legal, finished orders are checked by **two machine-learning models** trained on normal orders: a
   **Markov chain** (is this order of events rare?) and a **timing-and-amount model** with three checks
   (are the times, amount and outcome far from typical normal orders? were the steps consistently fast
   or slow? is any value far outside its normal range?). The four checks share **one threshold**. If
   any check is above it, the order is a **statistical anomaly**.
4. Every verdict comes with an explanation, and a dashboard shows the order's path on the automaton,
   the unusual transitions and the anomaly scores.

### How the PS9 objectives map to the code

| PS9 objective | Implementation |
| :--- | :--- |
| Workflow states, events, transition rules | `backend/fsm/states.py`, `transitions.py` |
| Generate or collect normal and anomalous logs | `backend/ml/dataset/generator.py` (generated, seeded) |
| Finite-state model to find invalid transitions | `backend/fsm/validator.py` |
| ML for statistical or sequential anomalies | `backend/ml/markov/` (sequential), `backend/ml/timing/` (statistical), `backend/ml/hybrid.py` (one threshold) |
| Combine rule violations with ML scores | `backend/pipeline/analyzer.py` |
| Evaluate, explain each flag | `backend/ml/evaluation/evaluate.py` and `benchmark.py`, explanations in `analyzer.py` |
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

| Type | What makes it suspicious | Which check should see it |
| :--- | :--- | :--- |
| Card testing | 2–6 failed payments, retried a bit faster than people, small amount | path (many failures), cluster |
| Repeated payment failure | 2–6 failed payments at normal pace | path (event order), cluster |
| Fast fulfilment | packing, shipping and delivery all faster than normal at once | trend |
| Stalled delivery | delivery ≈ 9 days instead of ≈ 3 | deviation |
| High-value quick refund | large amount refunded ≈ 30 min after payment | cluster |

**Key design rule: no giveaway feature.** Each type overlaps normal orders on every single
measurement. `audit.py` checks this: for each type, the best single measurement catches only 37–81%
when it may flag at most 1% of normal orders, and a test fails if any type reaches 90%. A single
measurement catches more if more false alarms are allowed; the small false-alarm budget is what makes
the task hard.

*Why this matters:* an earlier version had anomalies far outside normal ranges (e.g. amounts of
40,000+). A plain "outside the normal range" rule then beat the ML, which made the ML pointless.

### Rule violations

A normal fulfilled order broken by one named edit: missing payment, skipped step, wrong order,
duplicate event, event after the end, unknown event. Each is invalid by construction.

### Splits, made before any training

| Split | Normal | Anomalies | Used for |
| :--- | ---: | ---: | :--- |
| train | 4,000 | 500 | fitting the models (normal orders only) |
| validation | 2,000 | 100 | scaling the checks, setting the threshold, choosing the detector |
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

**Scale:** compute log₁₀ P(path) for the 2,000 normal *validation* orders. Two reference points are
kept: the **median** (a typical path: 75.8% of orders) and the **1st percentile** (the rarest-1% point:
about 1 in 1,990 orders). They give every path a **level**: 0 at the median, 1 at the rarest-1% point.

**Decision:** the path is flagged when its level is above the shared threshold (section 6). With the
current threshold, level 1.164, that means paths rarer than about 1 in 6,623 orders; 0.45% of normal
validation orders are.

```
path score = threshold − log₁₀ P(path)        > 0 means flagged; +1 means 10× rarer than the threshold
```

Example: failed payments, each followed by a retry, then a normal delivery. One fail-and-retry loop
multiplies the path probability by 0.117 × 0.744 ≈ 0.087.

| Failed payments | Path occurs in | Path score | Flagged |
| ---: | :--- | ---: | :--- |
| 2 | about 1 in 174 orders | −1.58 | no |
| 3 | about 1 in 1,990 orders | −0.52 | no |
| 4 | about 1 in 22,813 orders | 0.54 | yes |
| 5 | about 1 in 261,582 orders | 1.60 | yes |

**What it cannot see:** times and amounts. A normal-looking path that happened too fast looks normal to it.

**A limit of this chain:** the next event depends only on the current state, so a first payment
attempt and an attempt after three failures get the same probabilities. A chain whose state also
counts the failed payments was measured and did not improve the results (section 7), so it is not used.

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

For the comparison below, each candidate flags the orders above the **99th percentile** of its
scores on normal validation orders, so about 1% of normal orders.

### Choosing one, on the validation split only

| Candidate | Validation F1 |
| :--- | ---: |
| Isolation Forest | 0.400 |
| **k-means clustering** | **0.537** (chosen) |
| Local Outlier Factor | 0.389 |

Rule fixed in advance: highest validation F1, preferring a PS9-named model if it is within 0.02 of
the best. The test split was not used for this choice. The chosen detector is the **cluster check**.

### Two more checks, straight from the z-scores (`features.py: timing_summary`)

The cluster check measures one distance over all seven inputs. An order whose steps are each a little
fast, or whose single delivery step is slow, has six ordinary inputs and stays under that distance.
Two checks cover those cases:

| Check | Formula | High when |
| :--- | :--- | :--- |
| **trend** | \|sum of the step z-scores\| / √(number of steps) | every step is a little fast, or a little slow |
| **deviation** | −log₁₀ of the chi-square tail probability of (sum of squared step z-scores + squared amount z-score), one degree of freedom per value | one value, or several, are far from normal |

Under normal behaviour each z-score has mean 0 and variance 1, so the trend value is about 0.7 for a
typical order whatever its length, and the chi-square tail probability says how often normal orders
have a sum of squares that large. Neither check has anything to fit beyond the z-score statistics.

Like the path, each check gets a **level** from the normal validation orders: 0 at their median,
1 at their 99th percentile.

---

## 6. Combining everything (`backend/pipeline/analyzer.py`)

```
order ─► FSM ─┬─ invalid ─────────────► RULE_VIOLATION        (severity CRITICAL; no ML: the rules decide)
              ├─ valid, not finished ─► VALID_INCOMPLETE      (INFO; lists the endings still reachable)
              └─ valid, finished ─────► path level      (Markov chain)  ┐
                                         cluster level   (k-means)       │ highest level
                                         trend level                     ├─ above the threshold ─► STATISTICAL_ANOMALY (WARNING)
                                         deviation level                 ┘ otherwise ───────────► NORMAL (INFO)
```

The three timing checks run only when timestamps and the amount are supplied; without them the path
check decides alone, under the same threshold.

### One threshold for the four checks (`backend/ml/hybrid.py`)

The **hybrid level** of an order is the highest of its four levels. `calibrate()` computes it for the
2,000 normal validation orders and sets the threshold at the level that **1.5%** of them exceed
(currently 1.164). Alone, the checks flag 0.45% (path), 0.45% (cluster), 0.45% (trend) and 0.35%
(deviation) of normal validation orders; a few orders are flagged by more than one.

Why not let each check flag its own 1% and take the OR, as the earlier version did with two checks?
With four checks that would flag up to 4% of normal orders. One threshold keeps the total fixed, and
gives the hybrid one score, so ROC-AUC and PR-AUC can be reported for it. 1.5% is what the earlier
two-check OR flagged on validation data (1.45% to 1.75% over five seeds), so the results stay comparable.

**Why ML is not used for rule violations:** the automaton already decides them exactly. Measuring
the automaton on violations would be circular, since a violation *is* what the automaton rejects.

### The explanation

Every result has a summary, a root cause, key factors and a recommended action:

- **Unusual step** (amber on the diagram): a kind of step whose probability, multiplied over all its
  occurrences in the order, is below 10%. Either one rare step (e.g. 0.03), or a common step repeated
  (0.12 five times).
- **Unusual value**: its **percentile** among normal orders is outside the middle 95%, e.g.
  "took 1.2 h, faster than 98.1% of normal orders".
- **Which check flagged it**: the explanation gives one cause per flagged check ("its steps were
  consistently faster than normal", "one or more of its values are far outside their normal range",
  "its times, amount and outcome are far from those of typical normal orders") and lists every check's score.
- When no single value is outside the middle 95% but a check still flags the order, the explanation
  says so and lists the values furthest from typical.

---

## 7. Results (`backend/ml/evaluation/evaluate.py`, test split, used once)

Metrics, with the positive class = anomaly:

- **Precision**: share of alarms that are real anomalies.
- **Recall**: share of anomalies caught.
- **F1**: 2 · P · R / (P + R), high only when both are high. This is the headline metric.
  (Accuracy is misleading here: always answering "normal" is already 95% accurate.)
- **False-alarm rate**: share of normal orders flagged.
- **ROC-AUC**: how well a score ranks anomalies above normal orders (0.5 = random, 1 = perfect).
- **PR-AUC** (average precision): the same idea, but it counts false alarms against the number of
  real anomalies, so it is the more telling one when anomalies are rare. A model that only gives a
  yes/no answer has neither.

The baselines:

- **Tuned range rule** (no ML): every measurement gets a band that leaves the same share of normal
  training values outside on each side, and the band width is set on normal validation orders so that
  the rule flags as many of them as the hybrid does.
- **z-score sum OR Markov chain** (no fitted detector): the sum of squared z-scores flagging its top
  1% of normal validation orders, OR the Markov chain flagging its rarest 1%.
- The **untuned range rule** (outside the min–max seen in training) is kept for reference; it is a
  weak baseline because nothing about it is tuned.

### Seed 42: the dataset in the repository (test split)

<!-- generated:test-results:start -->
| Model | F1 | Precision | Recall | PR-AUC | Caught / 250 | False alarms / 5,000 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always "normal" (trivial baseline) | 0.000 | 0.000 | 0.000 | – | 0 | 0 (0.0%) |
| Range rule, untuned (outside the training min-max) | 0.335 | 0.633 | 0.228 | – | 57 | 33 (0.7%) |
| Range rule, tuned to the hybrid's false-alarm budget | 0.426 | 0.478 | 0.384 | 0.398 | 96 | 105 (2.1%) |
| z-score sum OR Markov chain (no fitted detector) | 0.485 | 0.548 | 0.436 | – | 109 | 90 (1.8%) |
| Markov chain alone (path check) | 0.353 | 0.589 | 0.252 | 0.329 | 63 | 44 (0.9%) |
| Isolation Forest alone | 0.271 | 0.517 | 0.184 | 0.411 | 46 | 43 (0.9%) |
| k-means alone | 0.466 | 0.662 | 0.360 | 0.507 | 90 | 46 (0.9%) |
| Local Outlier Factor alone | 0.338 | 0.596 | 0.236 | 0.402 | 59 | 40 (0.8%) |
| Trend check alone | 0.335 | 0.633 | 0.228 | 0.349 | 57 | 33 (0.7%) |
| Deviation check alone | 0.303 | 0.487 | 0.220 | 0.305 | 55 | 58 (1.2%) |
| **Hybrid: the four checks, one threshold (the app)** | **0.552** | 0.595 | 0.516 | 0.543 | **129** | 88 (1.8%) |
| Random Forest (supervised comparison) | 0.646 | 0.712 | 0.592 | 0.650 | 148 | 60 (1.2%) |
<!-- generated:test-results:end -->

Caught per type:

<!-- generated:caught-by-type:start -->
| Type (of 50 each) | Tuned range rule | z-score OR Markov | Path | Cluster | Trend | Deviation | **Hybrid** | Random Forest |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Card testing | 27 | 34 | 32 | 28 | 2 | 3 | **34** | 49 |
| Repeated failure | 30 | 34 | 31 | 25 | 0 | 2 | **32** | 11 |
| Fast fulfilment | 15 | 21 | 0 | 12 | 36 | 22 | **29** | 42 |
| Stalled delivery | 24 | 12 | 0 | 2 | 13 | 14 | **15** | 0 |
| High-value refund | 0 | 8 | 0 | 23 | 6 | 14 | **19** | 46 |
<!-- generated:caught-by-type:end -->

### Five seeds (42, 7, 1, 2, 3): mean ± standard deviation, test split

One seed can be lucky, so `benchmark.py` generates five datasets and repeats everything on each.
"Before" is the earlier model (Markov chain OR k-means, each flagging its own 1%).

<!-- generated:seeds-test:start -->
| Method | F1 | Precision | Recall | False-alarm rate | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before: Markov chain OR k-means | 0.533 ± 0.039 | 0.597 ± 0.043 | 0.487 ± 0.063 | 0.0167 ± 0.0037 | – | – |
| Baseline: tuned range rule | 0.444 ± 0.038 | 0.528 ± 0.048 | 0.386 ± 0.044 | 0.0175 ± 0.0035 | 0.898 ± 0.012 | 0.421 ± 0.045 |
| Baseline: z-score sum OR Markov chain | 0.526 ± 0.041 | 0.585 ± 0.041 | 0.481 ± 0.059 | 0.0172 ± 0.0033 | – | – |
| **After: hybrid (four checks, one threshold)** | **0.603 ± 0.035** | 0.626 ± 0.027 | 0.586 ± 0.062 | 0.0176 ± 0.0026 | 0.940 ± 0.008 | 0.626 ± 0.052 |
<!-- generated:seeds-test:end -->

F1 on each seed:

<!-- generated:seeds-test-f1:start -->
| Method | seed 42 | seed 7 | seed 1 | seed 2 | seed 3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Before: Markov chain OR k-means | 0.470 | 0.563 | 0.515 | 0.537 | 0.581 |
| Baseline: tuned range rule | 0.426 | 0.454 | 0.383 | 0.462 | 0.498 |
| Baseline: z-score sum OR Markov chain | 0.485 | 0.569 | 0.483 | 0.512 | 0.580 |
| **After: hybrid (four checks, one threshold)** | 0.552 | 0.613 | 0.581 | 0.612 | 0.657 |
<!-- generated:seeds-test-f1:end -->

Mean number caught per type:

<!-- generated:seeds-test-by-type:start -->
| Method (mean caught of 50) | Card testing | Repeated failure | Fast fulfilment | Stalled delivery | High-value refund |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Before: Markov chain OR k-means | 39.4 | 34.2 | 16.8 | 3.2 | 28.2 |
| Baseline: tuned range rule | 31.0 | 28.2 | 12.4 | 24.8 | 0.2 |
| Baseline: z-score sum OR Markov chain | 37.8 | 35.6 | 22.2 | 14.6 | 10.0 |
| **After: hybrid (four checks, one threshold)** | 37.4 | 32.2 | 37.2 | 15.6 | 24.0 |
<!-- generated:seeds-test-by-type:end -->

### What each check contributes (hybrid with one check removed)

<!-- generated:ablations:start -->
| Variant | Validation F1 | Validation PR-AUC | Test F1 | Test PR-AUC |
| :--- | ---: | ---: | ---: | ---: |
| **After: hybrid (four checks, one threshold)** | 0.642 ± 0.043 | 0.665 ± 0.037 | 0.603 ± 0.035 | 0.626 ± 0.052 |
| Hybrid without the path check | 0.623 ± 0.010 | 0.644 ± 0.032 | 0.580 ± 0.047 | 0.609 ± 0.062 |
| Hybrid without the cluster check | 0.623 ± 0.038 | 0.653 ± 0.040 | 0.594 ± 0.041 | 0.625 ± 0.050 |
| Hybrid without the trend check | 0.591 ± 0.053 | 0.637 ± 0.038 | 0.556 ± 0.036 | 0.593 ± 0.050 |
| Hybrid without the deviation check | 0.632 ± 0.039 | 0.650 ± 0.034 | 0.599 ± 0.020 | 0.622 ± 0.043 |
<!-- generated:ablations:end -->

<!-- generated:ablations-by-type:start -->
| Method (mean caught of 50) | Card testing | Repeated failure | Fast fulfilment | Stalled delivery | High-value refund |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **After: hybrid (four checks, one threshold)** | 37.4 | 32.2 | 37.2 | 15.6 | 24.0 |
| Hybrid without the path check | 33.4 | 25.0 | 39.0 | 16.2 | 24.4 |
| Hybrid without the cluster check | 36.2 | 33.2 | 39.6 | 17.0 | 14.4 |
| Hybrid without the trend check | 37.6 | 33.0 | 21.2 | 14.4 | 25.0 |
| Hybrid without the deviation check | 38.4 | 33.2 | 40.4 | 8.2 | 26.2 |
<!-- generated:ablations-by-type:end -->

### Measured and not used

<!-- generated:not-adopted:start -->
| Variant | Validation F1 | Validation PR-AUC | Test F1 | Test PR-AUC |
| :--- | ---: | ---: | ---: | ---: |
| **After: hybrid (four checks, one threshold)** | 0.642 ± 0.043 | 0.665 ± 0.037 | 0.603 ± 0.035 | 0.626 ± 0.052 |
| Two refund inputs for the cluster check | 0.627 ± 0.067 | 0.618 ± 0.068 | 0.599 ± 0.027 | 0.595 ± 0.019 |
| Retry-aware Markov chain as the path check | 0.623 ± 0.037 | 0.651 ± 0.035 | 0.585 ± 0.051 | 0.614 ± 0.057 |
| Ensemble of the three detectors as the cluster check | 0.640 ± 0.039 | 0.682 ± 0.039 | 0.608 ± 0.027 | 0.638 ± 0.046 |
| Stacking (logistic regression; validation = held-out half) | 0.616 ± 0.042 | 0.692 ± 0.043 | 0.609 ± 0.032 | 0.646 ± 0.042 |
<!-- generated:not-adopted:end -->

None of these beat the hybrid on the validation split, which is where such choices are made, so none
is in the app. `IMPROVEMENTS.md` has the detail, including what stacking catches of a type it was
never shown.

How to read this:

1. **The hybrid beats both fair baselines on every one of the five seeds**, at about the same
   false-alarm rate. The margin over the z-score baseline is modest: a small but consistent improvement.
2. **Each check has its own job.** The path check catches the event-order anomalies (card testing,
   repeated failures); the trend check catches fast fulfilment; the deviation check catches stalled
   delivery; the cluster check catches high-value refunds. Removing any one lowers the validation F1.
3. **The Random Forest scores higher only because it was given labels** for three types (card testing,
   fast fulfilment, high-value refund). On the two types it never saw, it caught far fewer, and
   **0** stalled deliveries. Real systems rarely have labels for every new kind of anomaly, so the app
   uses models that learn from normal orders only.
4. **Honest weaknesses:** most stalled deliveries and about half of the high-value refunds are still
   missed. Compared with the earlier model, a few card-testing, repeated-failure and refund catches
   were given up, because four checks now share the false-alarm budget that two used to. The
   contribution of the k-means check on its own is small; most of the gain comes from the trend check.
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
- **Tests**: `run_tests.py` runs the backend and frontend tests, with no mocks of the code under test.
  They include an end-to-end test from an empty folder (generate → train → evaluate → API), a test
  that the app's verdicts reproduce the reported numbers exactly, tests that fail if a model gets
  worse (every detector must rank anomalies with ROC-AUC above 0.8; the hybrid must beat every
  baseline), a test that the saved models do not change when the test split changes, and a test that
  the result tables in the documents match the JSON reports.

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

**How were thresholds chosen?** From normal *validation* orders, never the test set. Each check is
put on one scale (level 0 = typical normal order, level 1 = the point 1% of normal validation orders
exceed). One threshold on the highest level is then set so that 1.5% of normal validation orders are
flagged in total.

**Why one threshold instead of an OR of separate ones?** With four checks, separate 1% thresholds
would add up to as much as 4% false alarms. One threshold keeps the total fixed, and gives the hybrid
a single score, so ROC-AUC and PR-AUC can be measured for it.

**Is your baseline fair?** Two baselines are tuned the same way as the model: a range rule whose band
width is set on validation data to the same false-alarm share, and a z-score sum combined with the
Markov chain. The hybrid beats both on all five seeds, by a modest margin.

**How do you know there is no data leakage?** Splits are assigned in the generator before any
training. Models fit on train only; thresholds and model choice use validation only; the test split
is scored once. Labels are never model inputs.

**Why is F1 only about 0.6?** The anomalies are deliberately close to normal behaviour, and only about
1.5% of normal orders may be flagged. An earlier version with obvious anomalies scored about 0.98,
but a simple rule did just as well, which proved nothing. With more false alarms allowed, more
anomalies would be caught; the small false-alarm budget is what makes it hard.

**Why synthetic data?** PS9 allows "generate or collect". Real logs have no anomaly labels to evaluate
against, and do not match this workflow's states. The generator's rules are written down, seeded and
tested, so results are reproducible.

**Why k-means and not Isolation Forest?** All three were tried; k-means had the best validation F1
(0.537 vs 0.400 vs 0.389), and that rule was fixed before looking at test results.

**Does k-means use combinations of features?** Not in a way that was measurable: an earlier check
trained it on inputs whose columns were shuffled independently and it ranked anomalies about as well.
So the description is "far from typical normal orders", not "unusual together".

**What would you improve?** Real data with injected anomalies; a one-sided check for slow steps
(stalled delivery is still mostly missed); a check that looks at the amount and the refund delay of
refunded orders only. A retry-aware Markov chain, an ensemble of the three detectors and a stacked
model were measured and did not beat the current model on validation data.
