# DHIS2 Clinical and Stock Candidate Audit

## Purpose and status

This is a read-only, exploratory audit of additional routine-surveillance and
stock variables that might improve district-level malaria forecasting. It did
not alter the application database, modelling dataset, selected-model artifact,
production forecasts, source malaria series, thresholds, or pipeline records.

The DHIS2 login was verified interactively. Credentials were used only by a
temporary in-memory `requests.Session`, were cleared when each client context
closed, and were not written to this report, source code, environment files,
command arguments, audit artifacts, or the database.

## Data inspected

The district-week audit covered 2020W01 through 2025W52: 313 weeks, 146
districts, and 45,698 expected district-week combinations. There were no
duplicate analytics rows and no unknown district organisation-unit identifiers.

| Candidate | Observed rows | Coverage |
| --- | ---: | ---: |
| Reporting completeness | 45,698 | 100.0000% |
| Confirmed malaria | 45,451 | 99.4595% |
| Suspected malaria | 45,445 | 99.4464% |
| RDT tested | 45,435 | 99.4245% |
| RDT positive | 45,429 | 99.4114% |
| Microscopy tested | 44,962 | 98.3894% |
| Microscopy positive | 44,502 | 97.3828% |
| OPD total attendance | 45,451 | 99.4595% |
| AL stock balance | 45,412 | 99.3742% |
| Artesunate stock balance | 45,226 | 98.9671% |
| Malaria RDT stock balance | 45,410 | 99.3698% |

Several treatment fields were too sparse for a primary model: microscopy
negative treated had 27.56% coverage, malaria not tested but treated had 33.65%,
and RDT negative treated had 42.59%.

## Data-quality findings

High apparent coverage does not mean the variables are analysis-ready. The
audit found values that are epidemiologically or arithmetically implausible,
including maxima of 11,010,672 suspected cases, 27,020,497 RDT tests,
142,401,934 microscopy tests, 800,812,990 new OPD attendances, and
1,962,148,839 AL stock units in a district-week.

Internal checks also found:

| Check | Violations / checked | Rate |
| --- | ---: | ---: |
| RDT positive greater than RDT tested | 45 / 45,429 | 0.0991% |
| Microscopy positive greater than microscopy tested | 98 / 44,500 | 0.2202% |
| RDT-positive treated greater than RDT positive | 3,040 / 45,428 | 6.6919% |
| Microscopy-positive treated greater than microscopy positive | 2,477 / 44,443 | 5.5734% |
| New OPD attendance greater than total OPD attendance | 1,766 / 45,451 | 3.8855% |

Confirmed malaria was not required to equal RDT-positive plus
microscopy-positive counts: 93.01% of comparable rows differed. The median
difference is small, but the fields must not be treated as interchangeable
without confirming the Uganda HMIS aggregation definitions.

The experiment therefore used only conservative derived predictors. Positivity
was set missing whenever positives exceeded tests; count volumes were log
transformed; missing values were imputed from the training fold only; and all
candidate predictors were lagged by 1, 2, and 4 weeks. No same-week or future
value was supplied to a forecast.

## Controlled Random Forest ablation

The experiment used the same 12 rolling-origin folds, the same selected Random
Forest count hyperparameters, and the same training-only district seasonal
surge thresholds as the production validation. It was a fixed-parameter
ablation, not a production retraining or a new model-selection run.

The baseline block reproduced every authoritative pooled production metric at
all three horizons exactly. This is the primary validity check for the
comparison.

### Count performance

| Horizon | Baseline WAPE | Testing-quality WAPE | Change (points) | Result |
| ---: | ---: | ---: | ---: | --- |
| 4 weeks | 34.3113% | 34.1802% | -0.1311 | Tiny improvement |
| 8 weeks | 38.5616% | 38.6289% | +0.0673 | Tiny deterioration |
| 12 weeks | 39.6427% | 39.7544% | +0.1116 | Tiny deterioration |

The testing-quality block comprised lagged reporting completeness, valid test
positivity, and the RDT share of all malaria tests. Completeness alone generally
worsened count performance. A fuller clinical block adding suspected cases,
total tests, valid negatives, and OPD attendance worsened WAPE at every horizon
(34.5476%, 39.1592%, and 40.4818%).

### Surge warning performance

The operational surge decision still came from the count forecast versus the
approved district seasonal threshold. No separate surge classifier was silently
introduced.

| Horizon | Baseline sensitivity | Candidate sensitivity | Baseline F1 | Candidate F1 | Candidate precision |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 weeks | 73.28% | 73.84% | 62.01% | 62.28% | 53.86% |
| 8 weeks | 73.08% | 74.13% | 60.36% | 61.14% | 52.02% |
| 12 weeks | 72.81% | 74.42% | 61.11% | 61.58% | 52.53% |

These are modest sensitivity/F1 gains with mixed specificity and precision;
they are not evidence of a material improvement in overall model accuracy.

### District distribution

| Horizon | Districts with lower WAPE | Districts with higher WAPE | Median WAPE-point change | Interquartile change |
| ---: | ---: | ---: | ---: | ---: |
| 4 weeks | 75 | 71 | -0.275 | -1.637 to +1.424 |
| 8 weeks | 74 | 72 | -0.004 | -2.152 to +1.996 |
| 12 weeks | 72 | 74 | +0.085 | -1.670 to +2.238 |

The pooled result is not hiding a broad district-level gain. Improvements and
deteriorations are almost evenly split, especially at 8 and 12 weeks.

## Stock assessment

The weekly district stock balances are not suitable as stockout predictors.
They contain extreme values and aggregate quantities across facilities. A zero
balance, missing report, number of facilities reporting, and number of
facilities stocked out cannot be distinguished from a district sum alone.

A read-only facility-level sample for 2022W30, 2024W20, and 2025W52 found no
zero rows for AL, artesunate, or malaria RDT stock balances: every returned
analytics row was positive. This indicates zero/missing suppression in the
analytics response. Treating absent rows as zero stockout would therefore bias
the feature.

Metadata discovery identified monthly HMIS 105 “days out of stock” elements for
AL, injectable artesunate, and malaria RDTs. However, a valid facility-reporting
denominator could not be derived from the available aggregate reporting-rate
expression or indicator metadata. A stockout proportion is deferred until raw
facility data values and the HMIS 105 reporting denominator can be obtained and
validated. No stock variable was added to the model.

## Decision

Do not replace or retrain the production model from this audit. The additional
clinical variables did not produce consistent or practically substantial count
improvement, and the stock variables do not yet have an honest denominator.

The defensible next study is a prespecified candidate-feature analysis after
district biostatistician adjudication of impossible clinical values. It should
include nested training-only tuning, bootstrap confidence intervals for metric
deltas, district-level stability, calibration and threshold-utility analysis,
and a temporal holdout not used for feature selection. Stockout features should
be tested only after facility reporting and missingness are explicitly resolved.

The title **“District-Level Malaria Surge Prediction and Early Warning in Uganda
Using Machine Learning, Routine Surveillance and Climate Data”** remains
accurate. Confirmed malaria and reporting completeness are already routine
surveillance data. The title does not require every explored HMIS variable to be
retained.

## Reproducibility artifacts

- `data/clinical/dhis2_candidate_district_week_2020_2025.csv`
- `data/quality/dhis2_candidate_data_audit_2020_2025.json`
- `data/quality/dhis2_facility_stock_sample_audit.json`
- `data/experiments/clinical_ablation_20260721/ablation_summary.json`
- `data/experiments/clinical_ablation_20260721/testing_quality_district_horizon_metrics.csv`
- `data/experiments/clinical_ablation_20260721/testing_quality_district_distribution.json`

All listed artifacts are aggregate, non-secret, read-only audit or experimental
outputs. None is a published production forecast artifact.
