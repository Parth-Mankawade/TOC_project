# Anomaly Detection in Process Workflows Using Automata and ML (PS9)

A workflow-monitoring system for online-shop orders. Every order is a sequence of events
(pay, pack, ship, deliver, ...). The system:

1. **Checks the rules** with a finite automaton (DFA): any step the workflow does not allow is a
   **rule violation**; a legal order that has not finished is **in progress**.
2. **Checks legal, finished orders with machine learning**:
   - a **Markov chain** (a sequence model, i.e. a probabilistic automaton) asks *is this order of events rare?*
   - a **timing-and-amount model** asks *are the times, the amount and the outcome unusual together?*
     Three candidates from PS9's list are trained (Isolation Forest, k-means clustering, Local Outlier
     Factor); the one with the best F1 on a validation split is used (currently **k-means**).
   - Flagged by either model: **statistical anomaly**. Otherwise **normal**.
3. **Explains every verdict** and shows it on a dashboard: the order's path drawn on the automaton
   (playable step by step), unusual transitions highlighted, and every anomaly score.

| PS9 asks for | Where |
| :--- | :--- |
| Workflow states, events, rules | `backend/fsm/` (8 states, 8 events, 11 transitions), `backend/docs/WORKFLOW_SPEC.md` |
| Normal and anomalous event logs | `backend/ml/dataset/generator.py` (seeded synthetic data, rules written down) |
| Finite-state model for invalid transitions | `backend/fsm/validator.py` |
| ML for statistical or sequential anomalies | `backend/ml/markov/` (sequence), `backend/ml/timing/` (Isolation Forest, k-means, LOF), Random Forest as a supervised comparison |
| Combine rule violations with ML scores | `backend/pipeline/analyzer.py` |
| Evaluate, and explain each flag | `backend/ml/evaluation/evaluate.py`; every result has a root cause, key factors and an action |
| TOC: DFA, regular languages, reachability | `backend/fsm/reachability.py` (reachable states, deadlocks, cycles, infinite regular language) |
| Dashboard: paths, unusual transitions, scores | `frontend/` (React) |

## Results (held-out test split, used once)

5,000 normal orders and 250 anomalies (4.8%), 50 of each of five types. Models learn from normal
training orders only; thresholds and the choice of timing model use a separate validation split.

| Model | F1 | Anomalies caught | False alarms |
| :--- | ---: | ---: | ---: |
| Always "normal" (trivial baseline) | 0.000 | 0 / 250 | 0 / 5,000 |
| Range rule, no ML (outside the normal min–max) | 0.335 | 57 | 33 |
| Markov chain alone | 0.353 | 63 | 44 |
| k-means alone | 0.466 | 90 | 46 |
| **Markov chain + k-means (what the app uses)** | **0.470** | **102** | **82** |
| Random Forest (supervised comparison) | 0.646 | 148 | 60 |

- The ML beats the no-ML rule by a clear margin.
- The anomalies are deliberately hard: no single measurement can catch more than 81% of any type at a
  1% false-alarm rate (checked by `backend/ml/dataset/audit.py` and a test), so some are missed.
- The supervised Random Forest scores best because it was shown labelled examples of three types; it
  caught **0 of 50** stalled deliveries, a type it never saw. Real systems rarely have labels for every
  new kind of anomaly, which is why the app uses the unsupervised models.
- All 500 generated rule violations are rejected by the automaton. That is true by definition (it checks
  the generator), not a detection rate.

These numbers come from `backend/ml/evaluation/results.json`, which the pipeline writes and the
dashboard's *Models & results* page reads. Synthetic data: not a claim about real-world performance.

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

Runs the backend tests (193, pytest), the frontend type check, and the frontend tests (24, Vitest).
Nothing is mocked except the browser's `fetch` in the frontend API tests; the backend tests train real
models on a smaller generated dataset in a temporary folder, so they need no saved files. The
end-to-end test (`backend/tests/test_e2e.py`) starts from an empty folder: generate → train → evaluate →
API.

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
  ml/           dataset (generator, loader, audit), markov/, timing/, evaluation/, pipeline.py
  pipeline/     analyzer: FSM + both models -> verdict and explanation
  services/     analysis service (loads each model once, when first needed)
  database/     SQLite storage of every analysis (history)
  api/, main.py REST API (FastAPI)
  core/         settings and the named errors
  tests/        backend tests
  docs/         WORKFLOW_SPEC.md, DATASET_SPEC.md, API_INTEGRATION_GUIDE.md
frontend/       React dashboard (see frontend/README.md)
docs/           SETUP_GUIDE.md (run it on a new computer), VIVA_GUIDE.md (how the results are produced)
run_tests.py    every test, one command
```
