# Detection improvements (branch `improve-detection`)

What changed in the anomaly-detection models, what was measured, what was kept and what was dropped.
Every table here is written from `backend/ml/evaluation/benchmark.json` by
`python -m backend.ml.evaluation.report`; a test fails if a table and the JSON file disagree.

Reproduce (about 3 minutes; reads and writes nothing else in the repository):

```powershell
.\.venv\Scripts\python.exe -m backend.ml.evaluation.benchmark
```

## Rules the comparison follows

- `generator.py`, the anomaly definitions, the seed logic and the committed dataset are unchanged.
  Re-running the pipeline regenerates a byte-identical CSV.
- Five datasets are generated in memory, seeds 42, 7, 1, 2, 3, each with 4,000 normal training
  orders, 2,000 normal + 100 anomalous validation orders, and 5,000 normal + 250 anomalous test orders.
- Everything is fitted on the train split. Every threshold, the choice of detector and every decision
  below (which checks to keep, which ideas to drop) uses the validation split only.
- False-alarm budget: every method that has one score flags **1.5%** of normal validation orders. The
  earlier model was an OR of two parts flagging 1% each, which came to 1.45%–1.75% of normal
  validation orders over the five seeds, so 1.5% keeps the numbers comparable. The two OR rules in the
  tables keep their 1% per part; their actual false-alarm rates are in the tables.
- ROC-AUC and PR-AUC need a score. An OR of two yes/no decisions has none, so those cells are "–".

## 1. Before and after, with fair baselines

- **Before**: the model on `main`. Markov chain OR k-means, each flagging its own most unusual 1%.
- **Tuned range rule** (no ML): every measurement gets a band of equal tail share among normal
  training orders; the band width is set on validation data to the 1.5% budget.
- **z-score sum OR Markov chain** (no fitted detector): sum of squared z-scores flagging its top 1%,
  OR the Markov chain flagging its rarest 1%.
- **After**: the hybrid. Four checks (path, cluster, trend, deviation) under one threshold.

Test split, mean ± standard deviation over the five seeds:

<!-- generated:seeds-test:start -->
| Method | F1 | Precision | Recall | False-alarm rate | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before: Markov chain OR k-means | 0.533 ± 0.039 | 0.597 ± 0.043 | 0.487 ± 0.063 | 0.0167 ± 0.0037 | – | – |
| Baseline: tuned range rule | 0.444 ± 0.038 | 0.528 ± 0.048 | 0.386 ± 0.044 | 0.0175 ± 0.0035 | 0.898 ± 0.012 | 0.421 ± 0.045 |
| Baseline: z-score sum OR Markov chain | 0.526 ± 0.041 | 0.585 ± 0.041 | 0.481 ± 0.059 | 0.0172 ± 0.0033 | – | – |
| **After: hybrid (four checks, one threshold)** | **0.603 ± 0.035** | 0.626 ± 0.027 | 0.586 ± 0.062 | 0.0176 ± 0.0026 | 0.940 ± 0.008 | 0.626 ± 0.052 |
<!-- generated:seeds-test:end -->

Validation split (the one decisions were made on):

<!-- generated:seeds-validation:start -->
| Method | F1 | Precision | Recall | False-alarm rate | ROC-AUC | PR-AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before: Markov chain OR k-means | 0.540 ± 0.053 | 0.615 ± 0.030 | 0.486 ± 0.078 | 0.0152 ± 0.0021 | – | – |
| Baseline: tuned range rule | 0.487 ± 0.041 | 0.585 ± 0.024 | 0.418 ± 0.048 | 0.0147 ± 0.0004 | 0.910 ± 0.011 | 0.467 ± 0.048 |
| Baseline: z-score sum OR Markov chain | 0.546 ± 0.046 | 0.598 ± 0.022 | 0.504 ± 0.065 | 0.0168 ± 0.0008 | – | – |
| **After: hybrid (four checks, one threshold)** | **0.642 ± 0.043** | 0.672 ± 0.021 | 0.616 ± 0.063 | 0.0149 ± 0.0002 | 0.946 ± 0.006 | 0.665 ± 0.037 |
<!-- generated:seeds-validation:end -->

Test F1 on each seed:

<!-- generated:seeds-test-f1:start -->
| Method | seed 42 | seed 7 | seed 1 | seed 2 | seed 3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Before: Markov chain OR k-means | 0.470 | 0.563 | 0.515 | 0.537 | 0.581 |
| Baseline: tuned range rule | 0.426 | 0.454 | 0.383 | 0.462 | 0.498 |
| Baseline: z-score sum OR Markov chain | 0.485 | 0.569 | 0.483 | 0.512 | 0.580 |
| **After: hybrid (four checks, one threshold)** | 0.552 | 0.613 | 0.581 | 0.612 | 0.657 |
<!-- generated:seeds-test-f1:end -->

The hybrid is above the earlier model and above both baselines on each of the five seeds.

## 2. Caught per anomaly type, before and after

<!-- generated:seeds-test-by-type:start -->
| Method (mean caught of 50) | Card testing | Repeated failure | Fast fulfilment | Stalled delivery | High-value refund |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Before: Markov chain OR k-means | 39.4 | 34.2 | 16.8 | 3.2 | 28.2 |
| Baseline: tuned range rule | 31.0 | 28.2 | 12.4 | 24.8 | 0.2 |
| Baseline: z-score sum OR Markov chain | 37.8 | 35.6 | 22.2 | 14.6 | 10.0 |
| **After: hybrid (four checks, one threshold)** | 37.4 | 32.2 | 37.2 | 15.6 | 24.0 |
<!-- generated:seeds-test-by-type:end -->

Fast fulfilment and stalled delivery improve a lot. Card testing, repeated failure and high-value
refund each lose a few catches: four checks now share the budget two used to have, so the path and
cluster checks each work with a stricter threshold than before.

## 3. What was tried

### Kept

| Change | What it is | Why it was kept |
| :--- | :--- | :--- |
| Trend check | \|sum of step z-scores\| / √steps, in `timing/features.py` | Largest single gain. Without it the validation F1 falls the most, and fast fulfilment is mostly missed. |
| Deviation check | −log₁₀ of the chi-square tail probability of the squared z-scores | Small F1 gain on validation; it is what catches stalled deliveries. |
| One threshold for all checks (`ml/hybrid.py`) | Each check is put on one scale; the highest level is compared with one threshold set to the budget | Holds the total false-alarm share fixed however many checks there are, and gives the hybrid a score, so it now has a ROC-AUC and a PR-AUC. |
| Path check (Markov chain) and cluster check (k-means) | Unchanged models, now two of the four checks | Removing either lowers the validation F1. |

The hybrid with one check removed (each variant re-thresholded to the same budget):

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

Read with care: the cluster and deviation checks each add about 0.01–0.02 F1, which is smaller than
the spread between seeds. They are kept because the validation mean is higher with them and because
each covers a type the others do not (high-value refunds, stalled deliveries).

### Tried and not adopted

<!-- generated:not-adopted:start -->
| Variant | Validation F1 | Validation PR-AUC | Test F1 | Test PR-AUC |
| :--- | ---: | ---: | ---: | ---: |
| **After: hybrid (four checks, one threshold)** | 0.642 ± 0.043 | 0.665 ± 0.037 | 0.603 ± 0.035 | 0.626 ± 0.052 |
| Two refund inputs for the cluster check | 0.627 ± 0.067 | 0.618 ± 0.068 | 0.599 ± 0.027 | 0.595 ± 0.019 |
| Retry-aware Markov chain as the path check | 0.623 ± 0.037 | 0.651 ± 0.035 | 0.585 ± 0.051 | 0.614 ± 0.057 |
| Ensemble of the three detectors as the cluster check | 0.640 ± 0.039 | 0.682 ± 0.039 | 0.608 ± 0.027 | 0.638 ± 0.046 |
| Stacking (logistic regression; validation = held-out half) | 0.616 ± 0.042 | 0.692 ± 0.043 | 0.609 ± 0.032 | 0.646 ± 0.042 |
<!-- generated:not-adopted:end -->

<!-- generated:not-adopted-by-type:start -->
| Method (mean caught of 50) | Card testing | Repeated failure | Fast fulfilment | Stalled delivery | High-value refund |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **After: hybrid (four checks, one threshold)** | 37.4 | 32.2 | 37.2 | 15.6 | 24.0 |
| Two refund inputs for the cluster check | 34.6 | 30.8 | 37.4 | 15.8 | 24.4 |
| Retry-aware Markov chain as the path check | 34.2 | 29.6 | 37.0 | 15.6 | 23.8 |
| Ensemble of the three detectors as the cluster check | 37.8 | 32.2 | 37.8 | 16.0 | 25.2 |
| Stacking (logistic regression; validation = held-out half) | 41.0 | 32.8 | 31.0 | 14.2 | 26.6 |
<!-- generated:not-adopted-by-type:end -->

| Idea | Result on validation | Decision |
| :--- | :--- | :--- |
| **Retry-aware Markov chain** (state also counts failed payments, so a first attempt and a retry differ) | Lower F1 than the plain chain as the path check. The better model explains repeated failures in normal orders better too, so orders with several failures stand out less. | Not adopted. The plain chain stays. |
| **More timing inputs** (z-score of the wait before a refund; amount z-score of refunded orders) | Lower F1, and the validation rule then picks Local Outlier Factor instead of k-means on four of five seeds. Per-step z-scores as detector inputs were also tried while exploring and made the detector worse at fast fulfilment; the trend and deviation checks use every step's z-score instead. | Reverted. The detector keeps its seven inputs. |
| **Ensemble** (rank average of Isolation Forest, k-means and LOF as the cluster check) | Same F1 as k-means alone (within 0.002), slightly higher PR-AUC. | Not adopted: three detectors to explain and maintain for no F1 gain. |
| **Stacking** (logistic regression over the four check levels) | Lower F1 than the hybrid on the same held-out validation half, and it needs labelled anomalies. See below. | Not adopted. |

The code for these four is in `backend/ml/evaluation/experiments.py`, used only by the benchmark.

On the test split the ensemble and stacking come out slightly above the hybrid. The differences are
far inside the spread between seeds, and choosing by test results is exactly what the test split must
not be used for, so the validation result stands.

## 4. Leave-one-type-out for stacking

Stacking is the only supervised component that was measured. Half of the validation orders fit the
meta-model; the other half's normal orders set its threshold. It was then refitted five times, each
time without one anomaly type, and scored on that type in the test split:

<!-- generated:leave-one-type-out:start -->
| Type (mean caught of 50, test split) | Stacking, type seen in fitting | Stacking, type left out of fitting | Hybrid (never sees labels) |
| :--- | ---: | ---: | ---: |
| Card testing | 41.0 | 39.6 | 37.4 |
| Repeated failure | 32.8 | 28.2 | 32.2 |
| Fast fulfilment | 31.0 | 25.8 | 37.2 |
| Stalled delivery | 14.2 | 4.8 | 15.6 |
| High-value refund | 26.6 | 23.6 | 24.0 |
<!-- generated:leave-one-type-out:end -->

Left out of fitting, stalled delivery drops to about a third of what the label-free hybrid catches,
and fast fulfilment drops clearly too. A stacked model learns which checks mattered for the types it
was shown. That is the reason it is not used, more than its F1.

## 5. Tests that guard quality

- `backend/tests/test_model_quality.py`
  - every cluster candidate, the trend check, the deviation check and the hybrid must reach ROC-AUC
    above 0.8 on the fixture validation set; the path check must on the two event-order types;
  - the hybrid must beat the trivial baseline, both range rules, the z-score baseline and each of its
    own checks alone, and keep its false-alarm rate under twice the budget;
  - the fitting functions must receive only train and validation orders;
  - the saved models must not change when the test split is generated differently.
- These were checked by breaking the models on purpose: with the detector returning random scores,
  2 of these tests fail; with the Markov chain returning random probabilities, 5 fail; with random
  trend and deviation scores, 8 fail. (Before this branch, a random k-means passed every test.)
- `backend/tests/test_hybrid_model.py`: the shared threshold meets the budget and both model files
  hold the same threshold.
- `backend/tests/test_docs.py`: the result tables in `README.md`, `docs/VIVA_GUIDE.md` and this file
  match `results.json` and `benchmark.json`.

## 6. What changed in the app

- `timing_evaluation` in the API has two new fields: `checks` (score and flag for cluster, trend,
  deviation) and `trend_z`. `anomaly_score` is now the highest check score. Both models' thresholds
  come from the shared threshold.
- Explanations name the check that flagged the order ("its steps were consistently faster than
  normal") and list every check's score. The wording "unusual together" is gone from the code, the
  dashboard and the documents: no test showed that the detector uses interactions between inputs.
- The dashboard shows the three check scores on a result, and the tuned baselines and PR-AUC on the
  *Models & results* page.
- The saved model files and `results.json` were regenerated. **A database file from before this
  branch holds analyses in the old format: delete `backend\toc_workflow.db` once** (it is recreated
  empty), otherwise the History page reports a validation error for the old rows.
- `scipy` is now listed in `backend/requirements.txt` (scikit-learn already installed it).

## 7. Things to know

- A first run of the benchmark printed test-split numbers while the two extra detector inputs were
  still in the code. The decision to remove them was taken from the validation numbers in that same
  output, as were the others, but the test numbers of that variant had been seen.
- A comment in `generator.py` still says of the anomalies that "only the combination is unusual".
  That file was not to be modified, so the comment stays; `DATASET_SPEC.md` now words it correctly.
- Results are on synthetic data and say nothing about real order logs.

## Verdict

The hybrid is a real but moderate improvement. On five seeds its test F1 is higher than the earlier
model's on every seed, by roughly twice the spread between seeds, at about the same false-alarm rate,
and it is above both fair baselines on every seed as well. Almost all of the gain comes from two
simple checks computed from z-scores, not from a better learned model: the trend check lifts fast
fulfilment from about a third caught to about three quarters, and the deviation check lifts stalled
delivery from almost none to about a third. The learned parts changed little, and two of the ideas
aimed at them (a retry-aware Markov chain, more detector inputs) made things slightly worse. It still
fails on most stalled deliveries and about half of the high-value refunds, it gave up a few catches
on the three types the earlier model was better at, and the k-means check adds little that the other
three checks do not already cover.
