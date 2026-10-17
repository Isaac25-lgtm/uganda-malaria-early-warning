<h1 align="center">Uganda Malaria Early Warning System</h1>

<p align="center">
  <strong>Climate-Driven Machine Learning Prediction for District-Level Malaria Surges in Uganda</strong>
</p>

<p align="center">
  <em>An MSc research proof-of-concept that integrates routine HMIS surveillance and satellite-derived<br>
  climate predictors into an auditable district early-warning workflow and research dashboard.</em>
</p>

<p align="center">
  <img alt="Districts" src="https://img.shields.io/badge/districts-146-0e4d4d">
  <img alt="Horizons" src="https://img.shields.io/badge/horizons-4%20%7C%208%20%7C%2012%20weeks-176a63">
  <img alt="Models" src="https://img.shields.io/badge/core%20models-4-0d9488">
  <img alt="Tests" src="https://img.shields.io/badge/tests-automated-16a34a">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-3776ab">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-ea580c">
</p>

---

## Overview

This repository turns weekly DHIS2 malaria surveillance and satellite-derived climate data into
forward-looking district surge warnings for all **146 districts and cities in Uganda**. For each
district it produces a confirmed-case forecast at 4, 8 and 12 weeks ahead, compares that forecast
with a district- and season-specific threshold, and presents the resulting warning, its evidence and
its limitations in a read-only research dashboard.

**Correction notice (October 2026).** After the dissertation was submitted, a methodological review
found and corrected several validation and reporting errors (see [Corrections](#corrections-october-2026)).
The results below are the corrected values. Some differ from the submitted dissertation, and
several earlier headline claims no longer hold. The submitted figures are kept only as provenance.

**What the corrected evidence supports**

- On the same historical folds, Random Forest has lower count error than a same-week-last-year
  seasonal baseline at every horizon. It also has lower error than simple recent-average forecasts
  when every method is given the same case information.
- The surge warning detects about 71–73% of threshold-defined surges, but roughly half of its alerts
  are false, and its F2 score is close to (and on these folds below) that of a policy that alerts
  every district every week. Warning performance must therefore be judged on recall, precision and
  false-alert workload together, not on F2 alone.
- Lagged climate predictors give small, consistently favourable point estimates, but their
  incremental value is not statistically established in the prespecified analysis.
- Direct surge classifiers did not provide a credible improvement over the count-based warning.

All evaluation is retrospective. This is a decision-support research prototype, not an operational
outbreak-detection system.

---

## Table of contents

- [1. Research context](#1-research-context)
- [2. Contribution](#2-contribution)
- [3. Data sources](#3-data-sources)
- [4. Methodology](#4-methodology)
- [5. Results](#5-results)
- [Corrections, October 2026](#corrections-october-2026)
- [6. System architecture](#6-system-architecture)
- [7. Application features](#7-application-features)
- [8. Installation](#8-installation)
- [9. Testing and verification](#9-testing-and-verification)
- [10. Data availability](#10-data-availability)
- [11. Limitations](#11-limitations)
- [12. Ethics and responsible use](#12-ethics-and-responsible-use)
- [13. Repository structure](#13-repository-structure)
- [14. Citation](#14-citation)

---

## 1. Research context

Malaria remains a leading cause of morbidity in Uganda, with spatially heterogeneous and seasonal
transmission. District teams currently compare reported cases with historical endemic channels,
largely by hand, which detects a surge only after it has been reported. This study asks whether
routine DHIS2 surveillance, combined with gridded climate data, can support **district-level,
multi-horizon surge early warning** across all 146 districts and cities. The question is operational
and predictive, not causal.

**Study scope**

| Dimension | Specification |
| --- | --- |
| Spatial unit | 146 districts and cities (canonical `district_id`) |
| Temporal unit | ISO-8601 epidemiological week (Monday start), 2020–2025 |
| Forecast horizons | 4, 8 and 12 weeks ahead |
| Primary objective | District surge warning against a seasonal Q75 threshold |
| Supporting objective | Weekly confirmed malaria case-count forecasting (HMIS 033b) |

---

## 2. Contribution

1. **National district-level, multi-horizon workflow** — weekly forecasts and surge warnings for all
   146 districts at three horizons, on the routine DHIS2 platform.
2. **District- and season-specific thresholds** — training-only Q75 references for the target week
   ±2 weeks, with the 2022 epidemic excluded from the normal reference but retained for training.
3. **Leakage-controlled chronological validation** — rolling-origin evaluation with training-only
   preprocessing, horizon-aware label eligibility and, after correction, an embargo on inner tuning.
4. **Transparent warning evaluation** — recall, precision, false alerts, missed surges and alert rate
   are reported alongside F2, together with an always-alert reference that exposes what F2 alone rewards.
5. **Same-data benchmarking** — adapted published methods re-evaluated on this study's own panel
   under identical folds and warning rule, with adaptation fidelity recorded.
6. **Auditable handling of reporting errors** — six district-confirmed errors are held in a
   hash-verified exclusion register; raw values stay unchanged.
7. **An end-to-end auditable prototype** — data quality, climate integration, model evaluation,
   explainability, governance and reporting in one application.

---

## 3. Data sources

| Source | Variable | Unit / resolution |
| --- | --- | --- |
| **DHIS2 HMIS 033b** | Confirmed malaria cases | District-week counts |
| **DHIS2 HMIS 033b** | Facility reporting completeness | Percentage of expected reports (descriptive only) |
| **CHIRPS** | Rainfall | mm, weekly total of daily district polygon means |
| **ERA5-Land** | Temperature | °C, weekly mean |
| **ERA5-Land** | Relative humidity | %, derived per cell-hour from temperature and dewpoint |
| **UBOS** | District boundaries | Administrative polygons |

Wind speed and population denominators are not model inputs.

---

## 4. Methodology

### 4.1 Surveillance data preparation

- Missing reports remain missing and are never zero-filled; explicit zeros stay zero.
- Six confirmed reporting errors are excluded as outcomes, validation truth and threshold history.
- Outcomes are never imputed. Bounded linear interpolation (gaps of at most two weeks) is used only
  to build predictor histories, and only when its right-hand anchor was observed by the forecast origin.

Coverage: 45,698 expected district-weeks, 45,451 received, 247 absent at source, 6 confirmed errors,
45,445 valid outcomes.

### 4.2 Predictors

28 raw predictors: district identity, two time terms, case lags (1, 2, 4, 8, 12 weeks), rolling
case means (4, 8, 12 weeks), interpolation indicators, and rainfall, temperature and humidity lagged
4, 8 and 12 weeks **before the forecast origin**. District identity is one-hot encoded on training
rows only (unseen categories are ignored).

### 4.3 Validation

```
Chronological rolling-origin validation
├── 12 evenly spaced origins per horizon (one target week per origin)
├── No random splitting at any stage
├── Preprocessing fitted on training rows only
├── Training labels restricted to targets observable by the origin
└── Inner hyperparameter tuning embargoed at the earliest inner validation origin
```

Because each origin scores a single calendar week across all districts, the 12 origins are 12
dependent target weeks, not 1,750 independent observations. Supplementary analyses therefore add a
92-week weekly evaluation (2024–2025) and temporal block-bootstrap uncertainty.

### 4.4 Surge definition

A surge is an observed count above the district's training-only Q75 for the target epidemiological
week ±2 weeks (2022 excluded from the reference). Where fewer than eight seasonal observations exist,
the reference falls back to the district's broader history; this affects 25.0%, 21.1% and 16.7% of
references at 4, 8 and 12 weeks on the original folds. A warning is a forecast above that same threshold.

### 4.5 Models and selection

Four core models: a seasonal historical baseline, a district-specific autoregressive model with
climate regressors (labelled SARIMAX; the fitted specification is non-seasonal, with a
regression-only fallback in about 59% of forecasts), and nationally pooled Random Forest and XGBoost.

Surge warning is the primary objective. Count WAPE is reported as a supporting measure and is not
used on its own to select the warning approach.

---

## 5. Results

All results are retrospective development results on 2020–2025 data, after correction.

### 5.1 Original twelve-origin evaluation (corrected)

| Horizon | Model | WAPE | Recall | Precision | F2 | False alerts | Missed surges |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 weeks | **Random Forest** | 34.31% | 73.2% | 54.3% | 0.685 | 331 | 144 |
| 4 weeks | XGBoost | 36.44% | 69.0% | 50.4% | 0.642 | 365 | 167 |
| 8 weeks | **Random Forest** | 39.74% | 70.9% | 52.1% | 0.661 | 373 | 166 |
| 8 weeks | XGBoost | 40.35% | 68.7% | 50.4% | 0.640 | 386 | 179 |
| 12 weeks | **Random Forest** | 39.64% | 72.8% | 53.1% | 0.677 | 359 | 152 |
| 12 weeks | XGBoost | 39.21% | 70.6% | 53.9% | 0.665 | 337 | 164 |

Each horizon has about 1,750 scored district-weeks. Against the seasonal baseline (WAPE 48.82%,
48.48% and 47.27%), Random Forest reduces count error by 29.7%, 18.0% and 16.1% relative to that
baseline. At 12 weeks XGBoost has the lower point WAPE; the paired difference from Random Forest is
not statistically established. Random Forest has the higher F2 at every horizon.

**Reading F2 correctly.** About 31% of district-weeks are surges under this definition, so a policy
that alerts every district every week already scores F2 ≈ 0.69, 0.71 and 0.70. The Random Forest
warning's F2 is close to that reference while alerting on about 41–44% of district-weeks, with
roughly half its alerts false. F2 should be read together with precision, alert rate and false-alert
workload.

### 5.2 Equal-information weekly evaluation (92 target weeks, 2024–2025)

When every method receives case data only up to one week before the origin:

| Horizon | Random Forest WAPE | Recent 4-week mean WAPE | Persistence WAPE | Seasonal baseline WAPE |
| --- | --- | --- | --- | --- |
| 4 weeks | 25.21% | 28.32% | 28.81% | 42.40% |
| 8 weeks | 29.25% | 32.98% | 34.38% | 42.39% |
| 12 weeks | 31.95% | 35.39% | 37.61% | 42.39% |

Random Forest has lower count error than these simple forecasts, with temporal-block intervals
favouring it. Its warning recall advantage over the simple forecasts is smaller and not established
by the count result.

### 5.3 Climate contribution

Removing all climate predictors from Random Forest (fixed parameters, original folds) raises WAPE by
about 1.0, 1.0 and 1.3 percentage points at 4, 8 and 12 weeks, but every interval includes zero. An
exploratory, post hoc 92-week comparison favours climate by about 1.0 and 0.9 points at 4 and 12 weeks
(intervals exclude zero) and is inconclusive at 8 weeks. Climate lags are measured from the forecast
origin, so at 12 weeks the shortest climate lag is 16 weeks before the target. These are predictive
associations, not causal effects.

### 5.4 Direct surge classifiers

Class-balanced Random Forest and XGBoost surge classifiers were re-evaluated with corrected,
chronologically valid cutoff selection. Their F2 (about 0.70–0.73) exceeds the count-based warning only
by alerting on 82–97% of district-weeks, close to the always-alert reference, with about three times as
many false alerts. They were not adopted; the corrected count-based Random Forest warning is retained
as the research baseline at all three horizons.

### 5.5 Same-data published-method benchmark

Adapted Zinszer-inspired ARIMAX, Teklehaimanot-inspired Poisson distributed-lag and two sensitivity
models were evaluated on identical folds. These are adaptations, not replications. After correction,
the adapted ARIMAX has a slightly lower point WAPE than Random Forest at 8 weeks (38.69% vs 39.74%),
with a paired interval that includes zero; Random Forest retains the higher F2.

---

## Corrections, October 2026

Corrected in `app/` (commit history records each change):

- **Inner tuning leakage:** the hyperparameter tuning split could use training labels dated after the
  inner validation origins; labels are now embargoed at the earliest inner origin. This changed the
  8-week Random Forest parameters (WAPE 38.56% → 39.74%) and the 12-week XGBoost result.
- **Interpolation timing:** an interpolated predictor now requires its right-hand anchor to be
  observed by the forecast origin.
- **SARIMAX calendar:** absent weeks are kept as missing slots instead of compressing the series.
- **Metric consistency:** tie-aware average precision; one ranking score (forecast minus threshold)
  for every model; Brier score only for true probabilities.
- **Reporting:** district surge metrics now follow the operational count-threshold method; the
  explainability API reports F2 as the surge selection metric; the methods report no longer
  describes the rejected classifier as operational.

Earlier statements that Random Forest was selected in every position, that the model "detects 73%
of surge weeks and cuts forecast error by 30%", and that the warning is well balanced are superseded
by the corrected results above.

---

## 6. System architecture

```
run.py                       Application entry point (debug disabled)
app/
├── __init__.py              Application factory, blueprint registration
├── config.py                PostgreSQL-only configuration; fails fast if unset
├── models/                  SQLAlchemy models (research contract + session identity)
├── blueprints/              auth, dashboard, explainability, lifecycle_api, reports, settings, …
├── services/                Evidence, modelling and presentation services
├── templates/               Jinja templates
└── static/                  CSS, JavaScript, vendored Chart.js and Leaflet
tests/                       31 test modules — unit, contract, security, regression
research_comparisons/        Same-data published-method benchmark
research_extensions_2026_10/ Corrected re-evaluation and supervisor-response analyses
research_extensions_2026_10_round2/  Surge-closure analyses, methods and verification records
docs/                        Handover, retraining report, API reference
migrations/                  Alembic migrations
```

Design principles: evidence over assertion, fail closed on missing or malformed evidence, read-only
serving (asserted in tests), and offline operation with vendored front-end libraries.

---

## 7. Application features

| Module | Description |
| --- | --- |
| **Dashboard** | National and district views, target-week forecast against threshold, choropleth risk map, searchable district risk table |
| **Surveillance Data** | Coverage, received versus missing district-weeks, reporting completeness |
| **Data Quality & Preparation** | Lifecycle stages with leakage controls |
| **Climate Signals** | Rainfall, temperature and humidity with 4/8/12-week lags |
| **Model Evaluation** | Count and surge metrics for the four core models, selection rules, validation method, feature importance |
| **Forecasts & Risk** | Issue, horizon and district filters with an interpretation panel |
| **Reports & Exports** | Report products in PDF, Excel, CSV and print HTML |
| **Settings & Governance** | Read-only governance sections including integrity verification |

The dashboard displays a single frozen research issue (2025W52). It does not refresh data, retrain
models or confirm outbreaks, and it has not been evaluated with district health users.

---

## 8. Installation

### Prerequisites

- Python 3.12 (3.11+ supported)
- PostgreSQL 14 or later (PostgreSQL only; no SQLite fallback)
- A DHIS2 account with read access to the relevant data elements

### Setup

```bash
git clone https://github.com/Isaac25-lgtm/uganda-malaria-early-warning.git
cd uganda-malaria-early-warning

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-models.txt      # modelling dependencies
pip install -r requirements-climate.txt     # climate processing dependencies

cp .env.example .env                        # then edit .env with local values
flask db upgrade
python run.py                               # http://127.0.0.1:5000
```

### Configuration

All configuration is supplied through environment variables; see [`.env.example`](.env.example).
The application fails fast if `DATABASE_URL` is missing or is not a PostgreSQL URL, and requires
`SECRET_KEY` outside local development. DHIS2 credentials are entered at sign-in, verified live
against `/api/me`, and never stored. The configured DHIS2 URL must use HTTPS with certificate
verification enabled. Set `SESSION_COOKIE_SECURE=true` when serving over HTTPS.

---

## 9. Testing and verification

The public source-only gate requires no private database or unpublished artifacts:

```bash
python -m compileall app tests
python -m unittest -v tests.test_dhis2_client tests.test_modelling_dataset tests.test_model_performance tests.test_repository_hardening
```

Maintainers with the governed PostgreSQL database and complete local `data/` directory can run the
full suite:

```bash
python -m unittest discover -s tests -v
```

The research analyses carry their own verification: preserved-file hash checks, independent
recomputation of thresholds and metrics from saved predictions, representative refits, and focused
scientific tests (see `research_extensions_2026_10_round2/VERIFICATION_REPORT.md`). These are
internal cross-checks, not independent replication.

> Database- and artifact-backed tests intentionally fail closed when governed evidence is missing.
> Passing the public gate is not evidence that the private results were reproduced.

---

## 10. Data availability

The `data/` directory, row-level predictions, model binaries and research run outputs are **not
distributed**. They derive from Uganda Ministry of Health HMIS 033b surveillance records, which are
governed by the Ministry and are not the author's to redistribute; some diagnostic artifacts also
embed local paths, and the volume exceeds repository limits.

A later-period (2026) evaluation was carried out locally; its results are withheld from this
repository pending research approvals.

Source data may be requested from the custodians: the Uganda Ministry of Health (HMIS/DHIS2), the
[Climate Hazards Center](https://www.chc.ucsb.edu/data/chirps) (CHIRPS) and the
[Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) (ERA5-Land).

---

## 11. Limitations

- **Retrospective and limited in time.** Twelve target weeks per horizon in the main evaluation; all
  2020–2025 data have been inspected during development. No prospective validation has been done.
- **Constructed surge labels.** Surges are threshold exceedances, not adjudicated outbreaks, and the
  threshold relies on a broader-history fallback for many early references.
- **Warning workload.** About half of alerts are false and no acceptable workload has been
  established with intended users.
- **F2 is not sufficient.** At this prevalence an always-alert policy scores F2 ≈ 0.70.
- **Counts, not incidence.** No population-standardised incidence is modelled; comparisons between
  districts of different size describe reported burden.
- **Surveillance dependence.** Forecasts inherit reporting completeness and data-quality limits.
- **Association, not causation.** Climate and feature-importance results describe prediction only.
- **Pooled metrics.** National results hide district variation; district-level metrics rest on few
  target weeks and are unstable.
- **Unvalidated prototype.** The dashboard has not undergone usability or operational evaluation.

---

## 12. Ethics and responsible use

This system processes aggregate district-week counts; no patient-identifiable records are stored in
this repository. Research Ethics Committee approval and written Ministry of Health research
authorisation are **pending** at the time of writing; this repository does not establish an ethics
exemption or publication clearance.

Forecasts are decision-support outputs: an elevated risk category identifies districts for review and
does not confirm an outbreak or trigger automatic action. Ministry of Health identifiers describe the
data custodian and context and do not imply official endorsement.

---

## 13. Repository structure

```
.
├── app/                               Flask application (see §6)
├── docs/
│   ├── HANDOVER.md                    Architecture, code corrections and operations
│   ├── RETRAINING_CLEANING_REPORT.md  Cleaning, retraining and validation report
│   └── explainability_api.md          Explainability API reference
├── research_comparisons/              Same-data published-method benchmark (code and methods)
├── research_extensions_2026_10/       Corrected re-evaluation scripts and methods
├── research_extensions_2026_10_round2/  Surge-closure scripts, methods, verification and revision plan
├── migrations/                        Alembic database migrations
├── tests/                             Test suite (31 modules)
├── .env.example                       Configuration template (no secrets)
├── requirements*.txt                  Core, modelling and climate dependencies
└── run.py                             Entry point
```

---

## 14. Citation

```bibtex
@mastersthesis{omoding_malaria_ews,
  author  = {Omoding, Isaac},
  title   = {Climate-Driven Machine Learning Prediction for District-Level
             Malaria Surges in Uganda},
  school  = {Uganda Christian University},
  year    = {2026},
  type    = {{MSc} Dissertation},
  note    = {Available at:
             https://github.com/Isaac25-lgtm/uganda-malaria-early-warning}
}
```

---

## Acknowledgements

Surveillance data are collected by health facilities across Uganda and curated through the Ministry
of Health HMIS. Rainfall estimates are provided by the Climate Hazards Center (CHIRPS) and reanalysis
variables by the Copernicus Climate Data Store (ERA5-Land). District biostatisticians provided the
confirmations underpinning the reporting-error register.

---

<p align="center">
  <sub>MSc Data Science &amp; Analytics · Uganda Christian University ·
  District malaria surge early warning for 146 districts</sub>
</p>
