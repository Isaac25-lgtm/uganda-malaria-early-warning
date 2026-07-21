<h1 align="center">Uganda Malaria Early Warning System</h1>

<p align="center">
  <strong>District-Level Malaria Surge Prediction and Early Warning Using Climate Data in Uganda</strong>
</p>

<p align="center">
  <em>An MSc research proof-of-concept integrating routine HMIS surveillance, satellite-derived<br>
  climate predictors and validated machine-learning models into an auditable early-warning dashboard.</em>
</p>

<p align="center">
  <img alt="Districts" src="https://img.shields.io/badge/districts-146-0e4d4d">
  <img alt="Horizons" src="https://img.shields.io/badge/horizons-4%20%7C%208%20%7C%2012%20weeks-176a63">
  <img alt="Models" src="https://img.shields.io/badge/approved%20models-4-0d9488">
  <img alt="Tests" src="https://img.shields.io/badge/tests-400%20passing-16a34a">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12-3776ab">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-ea580c">
</p>

---

> [!IMPORTANT]
> **Research proof-of-concept — not an autonomous outbreak detector.**
> The system supports epidemiological review and early action. A High or Very High risk
> classification is an early-warning signal warranting review; it does **not** confirm an
> outbreak. Approximately 27% of labelled surges remain undetected and precision is
> 51–54%, so false alerts are material. These outputs do not replace epidemiological
> judgement, field investigation or official outbreak confirmation.

---

## Table of contents

- [1. Research context](#1-research-context)
- [2. Contribution](#2-contribution)
- [3. Data sources](#3-data-sources)
- [4. Methodology](#4-methodology)
- [5. Results](#5-results)
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

Malaria remains a leading cause of morbidity and mortality in Uganda, where transmission is
spatially heterogeneous and seasonally driven. Routine surveillance through the District
Health Information Software 2 (DHIS2) platform captures weekly confirmed cases, while
satellite-derived climate products characterise the environmental conditions that precede
changes in transmission.

This study investigates whether routinely available surveillance and climate data can
support **district-level, multi-horizon malaria surge early warning** across all 146 Ugandan
districts and cities. The research question is deliberately operational: not whether climate
*causes* malaria, but whether climate-informed models can provide sufficiently timely and
balanced warnings to support district-level review.

**Study scope**

| Dimension | Specification |
| --- | --- |
| Spatial unit | 146 districts and cities (canonical `district_id`) |
| Temporal unit | Epidemiological week (ISO-8601, Monday start), 2020–2025 |
| Forecast horizons | 4, 8 and 12 weeks ahead |
| Primary outcome | Weekly confirmed malaria cases (HMIS 033b) |
| Secondary outcome | District surge classification against a seasonal threshold |

---

## 2. Contribution

This work contributes:

1. **Multi-horizon district forecasting at national coverage** — weekly forecasts for all
   146 districts at three horizons, rather than a single aggregate or sentinel-site model.
2. **Joint count and surge evaluation** — models are assessed simultaneously on continuous
   error (MAE, RMSE, MAPE, WAPE, R²) and on surge detection (sensitivity, specificity,
   precision, F1, F2, PR-AUC, Brier score).
3. **District–season-specific thresholds** — surge reference values are computed per district
   for the target epidemiological week ±2 weeks, rather than a single national cut-off.
4. **Defensible treatment of a true epidemic** — the 2022 national epidemic is retained in
   count-model training but excluded from the normal-reference threshold, so a real epidemic
   does not silently redefine "normal".
5. **Auditable handling of confirmed reporting errors** — six district-biostatistician-confirmed
   data-entry errors are recorded in a hash-verified exclusion registry; raw values remain
   unchanged and fully auditable.
6. **Leakage-resistant temporal validation** — chronological rolling-origin validation with 12
   evenly spaced folds across multiple years, with preprocessing fitted on training rows only.
7. **An empirical operational finding** — dedicated class-balanced classifiers produced
   excessive false alerts on this dataset; a count-versus-seasonal-threshold rule proved more
   operationally balanced. All classifier results are retained rather than discarded.
8. **An end-to-end reproducible system** — surveillance quality, climate integration,
   explainability, governance, reporting and a district dashboard in one auditable application.

---

## 3. Data sources

| Source | Variable | Unit / resolution |
| --- | --- | --- |
| **DHIS2 HMIS 033b** | Confirmed malaria cases (`fUflbWWhouR`, code `033B-CD01a`) | District-week counts |
| **DHIS2 HMIS 033b** | Facility reporting completeness | Percentage of expected reports |
| **CHIRPS** | Rainfall | Millimetres, weekly total, district polygon mean |
| **ERA5-Land** | Temperature | Degrees Celsius, weekly mean |
| **ERA5-Land** | Relative humidity | Percent, derived from temperature and dewpoint |
| **UBOS** | District boundaries and population denominators | Administrative polygons; 2024 denominators |

Climate rasters are aggregated to district polygons and aligned to epidemiological weeks.
Wind speed is **not** used, and no negative binomial specification is included; both
exclusions are verified programmatically against the production manifest at runtime rather
than merely asserted in documentation.

---

## 4. Methodology

### 4.1 Surveillance data preparation

Raw DHIS2 values and database records are **preserved unchanged**. Analytical treatment is
applied downstream and documented:

- **Missing reports remain missing.** Absent district-weeks are never zero-filled; an
  explicitly reported zero remains zero. The two are distinguished throughout.
- **Confirmed reporting errors** (six district-weeks, verified with district biostatisticians)
  are recorded in a SHA-256-verified exclusion registry and treated as missing for outcomes,
  validation truth and threshold history.
- **Outcomes are never imputed.**
- **Bounded linear interpolation** is used *only* to construct predictor histories across one-
  or two-week internal gaps. Leading, trailing and longer gaps remain missing, and
  interpolation indicators are retained as predictors.

Coverage evidence: 45,698 expected district-weeks, 45,451 received, 247 absent at source,
6 confirmed errors, 253 analytical missing outcomes.

### 4.2 Modelling dataset

45,451 district-week rows across 146 districts with 34 predictor columns, comprising malaria
case lags and rolling means, lagged climate variables (4, 8 and 12 weeks), calendar position,
district identity and missingness indicators.

### 4.3 Validation protocol

```
Chronological rolling-origin validation
├── 12 evenly spaced folds across multiple years (not the final 12 origins)
├── No random splitting at any stage
├── Preprocessing fitted on training rows only
├── Targets aligned strictly to future weeks
└── Missing outcomes excluded rather than imputed
```

### 4.4 Surge definition

Surge thresholds are **district-specific, training-only 75th-percentile (Q75)** values for the
target epidemiological week ±2 weeks. Districts with fewer than eight seasonal observations
fall back to an expanding district history. The 2022 national epidemic is excluded from this
normal-reference baseline only — it remains in count-model training.

The **surge score** is `predicted cases − district threshold`, expressed in cases. It is
**not** a probability and must not be interpreted as one.

### 4.5 Model selection

Four approved candidates were evaluated: **Baseline/Naive**, **SARIMAX**, **Random Forest**
and **XGBoost**.

- *Count selection*: lowest WAPE, with RMSE as fallback where WAPE is unavailable.
- *Surge selection*: sensitivity ≥ 0.60 and specificity ≥ 0.40 preferred, then ranked by F2,
  sensitivity, PR-AUC, precision and specificity.
- **Accuracy is never used as a selection criterion**, because surge weeks are rare and a model
  can appear accurate while missing most surges.

Separate class-balanced classifiers were also trained (Random Forest with class weights,
XGBoost with `scale_pos_weight`; SMOTE was not used). These generated excessive false alerts,
so the operational count-versus-seasonal-threshold method was selected. Classifier results are
retained in the repository evidence rather than omitted.

---

## 5. Results

**Random Forest** is selected for both count prediction and operational surge warning at all
three horizons.

| Horizon | MAE | RMSE | WAPE | R² | Sensitivity | Specificity | Precision | F1 | F2 | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **4 weeks** | 393.75 | 627.54 | 34.31% | 0.695 | 73.3% | 71.9% | 53.7% | 62.0% | 68.3% | 0.626 |
| **8 weeks** | 462.76 | 693.42 | 38.56% | 0.656 | 73.1% | 66.5% | 51.4% | 60.4% | 67.4% | 0.607 |
| **12 weeks** | 460.33 | 701.07 | 39.64% | 0.636 | 72.8% | 69.3% | 52.7% | 61.1% | 67.6% | 0.609 |

Under the same validation protocol, Random Forest reduced WAPE relative to the naive baseline
by **29.72%, 20.45% and 16.13%** at 4, 8 and 12 weeks respectively.

> [!WARNING]
> **WAPE reduction is not accuracy.** These figures describe relative error reduction against a
> naive benchmark. They do not indicate that ~30% of outbreaks are detected, nor that
> predictions are ~70% correct.

### District-level performance

Random Forest and XGBoost are national-pooled models that nevertheless produce, and are
evaluated on, predictions for every district. Headline metrics are pooled across all
district-week predictions; district-level distributions are reported alongside them.

| Horizon | Median district WAPE | Interquartile range | Districts > 50% WAPE | Districts with negative R² |
| --- | --- | --- | --- | --- |
| 4 weeks | 35.6% | 30.1–42.6% | 18 | 72 |
| 8 weeks | 40.2% | 32.2–49.3% | 36 | 81 |
| 12 weeks | 40.0% | 33.6–50.8% | 39 | 72 |

Each district contributes only 12 validation predictions per horizon, so district-level R² and
sensitivity are statistically unstable. Pooled metrics are therefore reported as primary,
supplemented by WAPE distributions and confusion-count denominators. A district showing 100%
sensitivity on a single event is **not** treated as strong evidence.

---

## 6. System architecture

```
run.py                       Application entry point (debug disabled)
app/
├── __init__.py              Application factory, blueprint registration
├── config.py                PostgreSQL-only configuration; fails fast if unset
├── models/                  SQLAlchemy models (research contract + session identity)
├── blueprints/
│   ├── auth/                Live DHIS2 credential verification
│   ├── dashboard/           Dashboard page and dashboard APIs
│   ├── explainability/      Explainability and system-evidence APIs
│   ├── lifecycle_api/       Read-only surveillance and data-quality APIs
│   ├── dhis2 · preprocessing · climate · modelling · forecasts · howitworks
│   ├── reports/             Report generation (PDF, Excel, CSV, print HTML)
│   └── settings/            Settings and governance page
├── services/                Evidence, modelling and presentation services
├── templates/               Jinja templates
└── static/                  CSS, JavaScript, vendored Chart.js and Leaflet
tests/                       26 test modules — unit, contract, security, regression
docs/                        Handover, retraining report, API reference
migrations/                  Alembic migrations
```

**Design principles**

- *Evidence over assertion* — statuses, counts and model selections are derived from validated
  artifacts at runtime. Nothing analytical is hardcoded in the interface.
- *Fail closed* — missing, malformed or inconsistent evidence produces a controlled
  unavailable response rather than a partially authoritative page.
- *Read-only serving* — no HTTP request performs a database write; this is asserted by
  capturing SQL statements during tests.
- *Offline capable* — Chart.js and Leaflet are vendored locally; no CDN, external font or
  map-tile request is made at runtime.

---

## 7. Application features

| Module | Description |
| --- | --- |
| **Dashboard** | National and district views, weekly observed series with target-week forecast and threshold markers, choropleth risk map joined on `district_id`, paginated and searchable district risk table |
| **Surveillance Data** | Coverage, received versus missing district-weeks, explicit-zero accounting, reporting completeness |
| **Data Quality & Preparation** | Eight-stage lifecycle with evidence-derived leakage controls |
| **Climate Signals** | Rainfall, temperature and humidity with 4/8/12-week lags, national and district context |
| **Model Evaluation** | Count and surge metrics for all four models, selection rules, validation method, feature importance (native and permutation), threshold interpretation |
| **Forecasts & Risk** | Issue, horizon and district filters with a per-district interpretation panel |
| **How It Works** | Pipeline status derived from published evidence |
| **Reports & Exports** | Seven report products in PDF, Excel, CSV and print-friendly HTML |
| **Settings & Governance** | Eleven read-only governance sections including live integrity verification |

---

## 8. Installation

### Prerequisites

- Python 3.12 (3.11+ supported)
- PostgreSQL 14 or later — the application is PostgreSQL-only; there is no SQLite fallback
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
The application fails fast if `DATABASE_URL` is missing or is not a PostgreSQL URL, and
requires `SECRET_KEY` outside local development.

> [!CAUTION]
> **DHIS2 credentials are never stored.** They are supplied at web sign-in or through hidden
> CLI prompts, verified live against the DHIS2 `/api/me` endpoint, and then discarded. No
> username, password or token is written to `.env`, PostgreSQL, the session cookie, pipeline
> parameters, manifests or logs. Never add DHIS2 credentials to a configuration file.

**Authentication model.** Sign-in verifies live DHIS2 credentials but does **not** pull new
surveillance data; the dashboard displays the published production issue week. The session is
a signed, browser-session-only cookie holding four non-secret identity fields, with
server-side reauthentication after 30 minutes. No default administrative account is created
anywhere in the system.

---

## 9. Testing and verification

```bash
python -m compileall app tests
python -m unittest discover -s tests -v
```

The suite comprises **400 tests across 26 modules**, covering analytical contracts,
authentication and session security, API response contracts, export security (formula
injection, path traversal, temporary-file cleanup), read-only database behaviour,
scientific-honesty assertions and production-integrity fingerprinting.

Production integrity is verified by a deterministic SHA-256 forecast fingerprint recomputed on
demand, together with row, district, null, duplicate and composite-key checks.

> Tests requiring the database and research artifacts will skip or fail without a local
> PostgreSQL instance and the unpublished `data/` directory (see below).

---

## 10. Data availability

The `data/` directory is **not distributed with this repository**. It contains Uganda Ministry
of Health HMIS 033b surveillance records, DHIS2 metadata, CHIRPS and ERA5-Land raster
products, the derived modelling dataset and fitted model binaries — approximately 9 GB in
total.

It is withheld because:

1. Surveillance records are governed by the Uganda Ministry of Health and are not the author's
   to redistribute;
2. DHIS2 metadata describes internal organisational structure;
3. Some diagnostic artifacts embed local filesystem paths;
4. The volume substantially exceeds repository size limits.

Source data may be requested from the respective custodians: the Uganda Ministry of Health
(HMIS/DHIS2), the [Climate Hazards Center](https://www.chc.ucsb.edu/data/chirps) (CHIRPS) and
the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) (ERA5-Land).

---

## 11. Limitations

- **Surveillance dependence.** Forecasts inherit the quality of routine reporting. Weeks with
  low completeness may understate true burden.
- **Completeness is not incidence.** HMIS 033b reporting completeness measures form
  submission, not malaria burden.
- **Association, not causation.** Climate variables are predictive inputs. The pipeline
  computes no climate–malaria correlation and claims no causal effect or best-correlating lag.
- **Fixed population denominators.** UBOS 2024 district denominators are applied across the
  2020–2025 period because year-specific district projections are not part of the dataset.
- **Target-week only.** Models predict the specified target week; intermediate weekly forecasts
  are not generated, and no prediction interval is displayed unless the selected model
  publishes one.
- **Pooled importance.** Feature importance is national-pooled, model-specific and predictive.
  It does not describe an individual district and does not explain any single production
  forecast; per-prediction production explanations are unavailable.
- **District instability.** Twelve validation predictions per district per horizon are
  insufficient for stable district-level R² or sensitivity.
- **Not comparable across protocols.** Earlier and current count metrics are not directly
  comparable, because validation design, data cleaning and surge labelling all changed. The
  stricter multi-year evaluation produced higher WAPE than the earlier final-period
  evaluation; this is reported rather than concealed.
- **No prospective validation.** All evaluation is retrospective. Prospective and external
  validation remain outstanding.

### Recommended further work

Confidence intervals for all reported metrics; deeper district-performance distribution
analysis; predictor ablation studies; threshold sensitivity analysis; and prospective and
external validation in an operational setting.

---

## 12. Ethics and responsible use

This system was developed for academic research. It processes aggregate district-week counts
and contains no patient-identifiable information. Outputs are intended to support
epidemiological review, not to trigger automated action.

Users must not interpret a risk classification as confirmation of an outbreak, must not present
forecasts as a substitute for epidemiological judgement, and must not infer causal
climate–malaria relationships from these models.

---

## 13. Repository structure

```
.
├── app/                          Flask application (see §6)
├── docs/
│   ├── HANDOVER.md               Architecture, routes, data dictionary, operations
│   ├── RETRAINING_CLEANING_REPORT.md   Cleaning, retraining and validation report
│   └── explainability_api.md     Explainability API reference
├── migrations/                   Alembic database migrations
├── tests/                        Test suite (26 modules)
├── .env.example                  Configuration template (no secrets)
├── requirements.txt              Core dependencies
├── requirements-models.txt       Modelling dependencies
├── requirements-climate.txt      Climate processing dependencies
└── run.py                        Entry point
```

---

## 14. Citation

If you reference this work, please cite:

```bibtex
@mastersthesis{omoding_malaria_ews,
  author  = {Omoding, Isaac},
  title   = {District-Level Malaria Surge Prediction and Early Warning
             Using Climate Data in Uganda},
  school  = {Uganda Christian University},
  year    = {2026},
  type    = {{MSc} Dissertation},
  note    = {Available at:
             https://github.com/Isaac25-lgtm/uganda-malaria-early-warning}
}
```

---

## Acknowledgements

Surveillance data are collected by health facilities across Uganda and curated through the
Ministry of Health HMIS. Rainfall estimates are provided by the Climate Hazards Center
(CHIRPS) and reanalysis climate variables by the Copernicus Climate Data Store (ERA5-Land).
District biostatisticians provided the confirmations underpinning the reporting-error registry.

---

<p align="center">
  <sub>Research proof-of-concept · Supports review and early action ·
  Does not replace epidemiological judgement or official outbreak confirmation</sub>
</p>
