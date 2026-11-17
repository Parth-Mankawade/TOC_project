# Backend

FastAPI service plus the ML pipeline. Setup, results and the demo are in the root `README.md`;
this file describes how the backend is built.

## Pipeline checkpoints

Each checkpoint checks its input and output and raises its own named error
(`backend/core/errors.py`), with the message `[CHECKPOINT] expected ..., got ...; input=...`.

| Checkpoint | Code | Error |
| :--- | :--- | :--- |
| GENERATE: seeded synthetic event log | `ml/dataset/generator.py` | `GenerateError` |
| LOAD: read and check every CSV row | `ml/dataset/loader.py` | `DataLoadError` |
| FEATURES: order -> model inputs | `ml/timing/features.py` | `FeatureError` |
| TRAIN: fit models, scale every check on validation data | `ml/markov/model.py`, `ml/timing/model.py` | `TrainError` |
| CALIBRATE: one threshold for all four checks | `ml/hybrid.py` | `TrainError` |
| MODEL: load a saved model file | same | `ModelLoadError` |
| EVALUATE: test split, baselines, report | `ml/evaluation/evaluate.py` | `EvaluateError` |
| INPUT: an order sent for analysis | `fsm/parsing.py`, `schemas/order.py` | `InputError` (HTTP 422) |
| ANALYZE: FSM + models -> verdict + explanation | `pipeline/analyzer.py` | `AnalyzeError` |
| STORAGE: save and read analyses | `database/` , `api/mapping.py` | `StorageError` |
| API: missing record | `api/routes.py` | `NotFoundError` (HTTP 404) |

`python -m backend.ml.pipeline` runs GENERATE -> LOAD -> TRAIN -> CALIBRATE -> save -> reload -> EVALUATE and writes:

- `ml/dataset/data/orders_event_log.csv`: 12,350 orders (train / validation / test)
- `ml/models/markov_chain.json`: transition probabilities and threshold
- `ml/models/timing_model.joblib` and `.json`: the chosen timing model and its readable metadata
- `ml/evaluation/results.json`: the test-split report

`python -m backend.ml.evaluation.benchmark` repeats the comparison on five freshly generated datasets
(seeds 42, 7, 1, 2, 3; nothing saved in the repository is read or changed) and writes
`ml/evaluation/benchmark.json`. `python -m backend.ml.evaluation.report` rewrites the result tables in
`README.md`, `docs/VIVA_GUIDE.md` and `IMPROVEMENTS.md` from the two JSON files.
`ml/evaluation/experiments.py` holds alternatives that were measured and not adopted; the app does not use it.

## Models

Four checks. Each puts an order on the same scale, its **level**: 0 is a typical normal order and 1 is
the point only 1% of normal validation orders exceed.

- **path**, the Markov chain (`ml/markov/model.py`): P(event | state) from normal training orders with
  Laplace smoothing (alpha = 1). The probability of a path is the product of its steps.
  Score = threshold - log10 P(path).
- **cluster** (`ml/timing/`): each order becomes 7 numbers (amount z-score, slowest, fastest and mean
  step z-score, failed payments, refunded, cancelled), standardized. Candidates: Isolation Forest (200
  trees), k-means (k by silhouette score; score = distance to the nearest centre), Local Outlier Factor
  (20 neighbours). The candidate with the best validation F1 (each flagging its most unusual 1%) is
  used; a PS9-named model within 0.02 F1 of the best wins.
- **trend** (`ml/timing/features.py: timing_summary`): |sum of the step z-scores| / sqrt(number of
  steps). High when every step is a little fast, or a little slow.
- **deviation** (same function): -log10 of the chi-square tail probability of the sum of squared
  z-scores (steps and amount). High when one value, or several, are far from normal.
- **One threshold** (`ml/hybrid.py`): the hybrid level of an order is its highest level over the four
  checks. `calibrate()` sets the threshold so that 1.5% of normal validation orders are above it, and
  stores it in both model files. An order is flagged when any check is above it. Each timing check's
  score in the API is level - threshold.
- **Unusual step** (shown in amber): a kind of step whose probability, multiplied over every time it
  occurs in the order, is below 10%.
- **Random Forest** (`ml/evaluation/evaluate.py`): supervised comparison only, trained on three of the
  five anomaly types.

## Settings

`core/config.py` holds every path. Environment variables: `TOC_DATABASE_URL` (default
`backend/toc_workflow.db`), `TOC_DATABASE_ECHO`, `INIT_DB_ON_STARTUP` (default true), `CORS_ORIGINS`.
`create_app(Settings(...))` builds an app on any files, which is how the tests run a real app in a
temporary folder.

The database table is checked on startup; a database file from an older version (missing columns)
raises a `StorageError` saying so. Delete it and it is recreated.

## API

See `docs/API_INTEGRATION_GUIDE.md`. Interactive docs: http://127.0.0.1:8000/docs.
