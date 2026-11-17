# Anomaly Detection in Process Workflows Using Automata and ML (PS9)

A workflow-monitoring system for online-shop orders. Every order is a sequence of events
(pay, pack, ship, deliver, ...). The system:

1. **Checks the rules** with a finite automaton (DFA): any step the workflow does not allow is a
   **rule violation**; a legal order that has not finished is **in progress**.
2. **Checks legal, finished orders with machine learning**, using four checks that share one threshold:
   - **path**: a **Markov chain** (a sequence model, i.e. a probabilistic automaton) asks *is this order of events rare?*
   - **cluster**: a detector fitted on normal orders asks *are the times, the amount and the outcome far
     from typical normal orders?* Three candidates from PS9's list are trained (Isolation Forest, k-means
     clustering, Local Outlier Factor); the one with the best F1 on a validation split is used (currently **k-means**).
   - **trend**: *were the steps consistently faster, or consistently slower, than normal?*
   - **deviation**: *is any value far outside its normal range?*
   - The threshold is set on normal validation orders so that the four checks together flag 1.5% of them.
     Flagged by any check: **statistical anomaly**. Otherwise **normal**.
3. **Explains every verdict** and shows it on a dashboard: the order's path drawn on the automaton
   (playable step by step), unusual transitions highlighted, and every anomaly score.

| PS9 asks for | Where |
| :--- | :--- |
| Workflow states, events, rules | `backend/fsm/` (8 states, 8 events, 11 transitions), `backend/docs/WORKFLOW_SPEC.md` |
| Normal and anomalous event logs | `backend/ml/dataset/generator.py` (seeded synthetic data, rules written down) |
| Finite-state model for invalid transitions | `backend/fsm/validator.py` |
| ML for statistical or sequential anomalies | `backend/ml/markov/` (sequence), `backend/ml/timing/` (Isolation Forest, k-means, LOF, trend and deviation checks), `backend/ml/hybrid.py` (one threshold for all), Random Forest as a supervised comparison |
| Combine rule violations with ML scores | `backend/pipeline/analyzer.py` |
| Evaluate, and explain each flag | `backend/ml/evaluation/evaluate.py` (one seed), `benchmark.py` (five seeds); every result has a root cause, key factors and an action |
| TOC: DFA, regular languages, reachability | `backend/fsm/reachability.py` (reachable states, deadlocks, cycles, infinite regular language) |
| Dashboard: paths, unusual transitions, scores | `frontend/` (React) |

## Results (held-out test split, used once)

5,000 normal orders and 250 anomalies (4.8%), 50 of each of five types. Models learn from normal
training orders only; thresholds and the choice of detector use a separate validation split.

Seed 42, the dataset in this repository (`backend/ml/evaluation/results.json`):

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

Caught per anomaly type, same split:

<!-- generated:caught-by-type:start -->
| Type (of 50 each) | Tuned range rule | z-score OR Markov | Path | Cluster | Trend | Deviation | **Hybrid** | Random Forest |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Card testing | 27 | 34 | 32 | 28 | 2 | 3 | **34** | 49 |
| Repeated failure | 30 | 34 | 31 | 25 | 0 | 2 | **32** | 11 |
| Fast fulfilment | 15 | 21 | 0 | 12 | 36 | 22 | **29** | 42 |
| Stalled delivery | 24 | 12 | 0 | 2 | 13 | 14 | **15** | 0 |
| High-value refund | 0 | 8 | 0 | 23 | 6 | 14 | **19** | 46 |
<!-- generated:caught-by-type:end -->

Mean ± standard deviation over five generated datasets, seeds 42, 7, 1, 2, 3, test split
(`backend/ml/evaluation/benchmark.json`). "Before" is the model this one replaced:

<!-- generated:seeds-test:start -->
| Method | F1 | Precision | Recall | False-alarm rate | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before: Markov chain OR k-means | 0.533 ± 0.039 | 0.591 ± 0.040 | 0.492 ± 0.068 | 0.0173 ± 0.0038 | – | – |
| Baseline: tuned range rule | 0.444 ± 0.038 | 0.528 ± 0.048 | 0.386 ± 0.044 | 0.0175 ± 0.0035 | 0.898 ± 0.012 | 0.421 ± 0.045 |
| Baseline: z-score sum OR Markov chain | 0.526 ± 0.041 | 0.585 ± 0.041 | 0.481 ± 0.059 | 0.0172 ± 0.0033 | – | – |
| **After: hybrid (four checks, one threshold)** | **0.603 ± 0.035** | 0.626 ± 0.027 | 0.586 ± 0.062 | 0.0176 ± 0.0026 | 0.940 ± 0.008 | 0.626 ± 0.052 |
<!-- generated:seeds-test:end -->

<!-- generated:seeds-test-by-type:start -->
| Method (mean caught of 50) | Card testing | Repeated failure | Fast fulfilment | Stalled delivery | High-value refund |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Before: Markov chain OR k-means | 40.0 | 34.8 | 16.8 | 3.2 | 28.2 |
| Baseline: tuned range rule | 31.0 | 28.2 | 12.4 | 24.8 | 0.2 |
| Baseline: z-score sum OR Markov chain | 37.8 | 35.6 | 22.2 | 14.6 | 10.0 |
| **After: hybrid (four checks, one threshold)** | 37.4 | 32.2 | 37.2 | 15.6 | 24.0 |
<!-- generated:seeds-test-by-type:end -->

How to read this:

- The hybrid beats both fair baselines on every one of the five seeds, at about the same false-alarm
  rate. The margin over the z-score baseline (which uses no fitted detector) is modest, and the
  standard deviations are about half as large as that margin.
- Most of the gain over the earlier model is in two types: fast fulfilment and stalled delivery. It
  costs a few catches in the other three, because four checks now share the false-alarm budget that
  two used to.
- It still misses most stalled deliveries and about half of the high-value refunds. The anomalies are
  deliberately close to normal orders: at a 1% false-alarm rate no single measurement catches more
  than 81% of any type (checked by `backend/ml/dataset/audit.py` and a test). Single measurements
  catch more if more false alarms are allowed; the small false-alarm budget is what makes it hard.
- The supervised Random Forest scores best because it was shown labelled examples of three types; it
  caught **0 of 50** stalled deliveries, a type it never saw. Real systems rarely have labels for every
  new kind of anomaly, which is why the app uses models that learn from normal orders only.
- All 500 generated rule violations are rejected by the automaton. That is true by definition (it checks
  the generator), not a detection rate.

The tables above are written by `python -m backend.ml.evaluation.report` from the two JSON files, and
a test fails if they drift. The dashboard's *Models & results* page reads `results.json` directly.
[IMPROVEMENTS.md](IMPROVEMENTS.md) lists what was tried, what was kept and what was dropped.
Synthetic data: not a claim about real-world performance.

## Run it (Windows PowerShell, from this folder)

Once:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
cd frontend; npm install; npm run build; cd ..
```

Generate the data, train both models, evaluate (about 15 seconds):

```powershell
.\.venv\Scripts\python.exe -m backend.ml.pipeline
```

Optional: the five-seed benchmark (about 3 minutes), and rewriting the tables in this file from it:

```powershell
.\.venv\Scripts\python.exe -m backend.ml.evaluation.benchmark
.\.venv\Scripts\python.exe -m backend.ml.evaluation.report
```

Start the app and open **http://127.0.0.1:8000/** (it redirects to the dashboard at `/app/`):

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000
```

## For the demo

- **Overview → Live demo stream → Start stream**: 100 real test orders go through the API one at a time;
  the diagram shows each order's path and the counters compare every verdict with how the generator made
  that order (anomalies caught, false alarms, rule violations).
- **Analyze**: pick a real test order of any kind, press Analyze. *How the verdict was reached* lights up
  each check in order; the execution path plays through the automaton step by step (play, pause, step,
  speed). Or build your own order event by event.
- **Workflow & reachability**: the automaton, its formal definition, deadlock/cycle/language facts, and
  *Play this path* for the shortest route to any state.
- **Models & results**: every number in the table above, the validation choice, and the data check.

## Tests: one command

```powershell
.\.venv\Scripts\python.exe run_tests.py
```

Runs the backend tests (pytest), the frontend type check, and the frontend tests (Vitest).
Nothing is mocked except the browser's `fetch` in the frontend API tests; the backend tests train real
models on a smaller generated dataset in a temporary folder, so they need no saved files. The
end-to-end test (`backend/tests/test_e2e.py`) starts from an empty folder: generate → train → evaluate →
API.

`backend/tests/test_model_quality.py` fails if the models get worse: every detector must rank
anomalies well above chance (ROC-AUC above 0.8), the hybrid must beat every baseline, and the saved
models must not change when the test split changes. `backend/tests/test_docs.py` fails if a result
table in the documents no longer matches the JSON files.

## Errors

Nothing fails silently. Every error names the step that failed, what was expected, what arrived and
which input caused it, for example:

```
[MODEL] expected a saved Markov model (run: python -m backend.ml.pipeline), got no file; input=...\markov_chain.json
```

The API returns it as JSON with the right status (422 bad input, 404 missing record, 503 missing
model/data/report/database, 500 bug) and the dashboard shows the message where the data would be.

## Layout

```text
backend/
  fsm/          automaton, validator, input parsing, reachability
  ml/           dataset (generator, loader, audit), markov/, timing/, hybrid.py, evaluation/, pipeline.py
  pipeline/     analyzer: FSM + both models -> verdict and explanation
  services/     analysis service (loads each model once, when first needed)
  database/     SQLite storage of every analysis (history)
  api/, main.py REST API (FastAPI)
  core/         settings and the named errors
  tests/        backend tests
  docs/         WORKFLOW_SPEC.md, DATASET_SPEC.md, API_INTEGRATION_GUIDE.md
frontend/       React dashboard (see frontend/README.md)
docs/           SETUP_GUIDE.md (run it on a new computer), VIVA_GUIDE.md (how the results are produced)
IMPROVEMENTS.md what changed in the detection models, with the measurements behind each decision
run_tests.py    every test, one command
```
