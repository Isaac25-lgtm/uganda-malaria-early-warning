# Uganda District-Level Malaria Early Warning Dashboard — Handover

**Status:** research proof-of-concept, feature-complete for demonstration and academic
supervision. Not an operational alert system.

**Production issue week:** 2025W52 · **Districts:** 146 · **Production forecasts:** 1,752
**Deterministic forecast fingerprint:** `2854b51afc0e06e1284232a584b4a8b91e3ea60e6a79e82452b850396cc144f9`

---

## 1. What the system does

It combines weekly confirmed malaria cases (DHIS2 HMIS 033b), HMIS 033b reporting
completeness, and climate predictors (CHIRPS rainfall, ERA5-Land temperature and
derived humidity) to publish **district-level malaria forecasts and a count-threshold
surge assessment** for Uganda at 4, 8 and 12-week horizons.

It **supports review and early action**. It does not replace epidemiological judgement
or official outbreak confirmation.

---

## 2. Architecture

```
run.py                      entry point (debug=False)
app/__init__.py             application factory, blueprint registration, nav groups
app/config.py               PostgreSQL-only config; fails fast without DATABASE_URL
app/models/                 SQLAlchemy models (research + auth)
app/blueprints/
  auth/                     live DHIS2 credential verification, login/logout
  dashboard/                main dashboard page + 6 dashboard APIs
  explainability/           explainability APIs + /api/system/{pipeline-status,governance}
  lifecycle_api/            small read-only surveillance + data-quality APIs
  dhis2, preprocessing, climate, modelling, forecasts, howitworks   lifecycle pages
  reports/                  Reports & Exports page, generation, print views
  settings/                 Settings & Governance page
app/services/               read-only evidence + presentation services (see §4)
app/templates/              Jinja templates (base, dashboard, lifecycle, reports, settings)
app/static/                 CSS, JS, vendored Chart.js + Leaflet, MoH logo
tests/                      unit, contract, security and regression tests
data/                       approved artifacts (quality, models, boundaries) — read-only
```

**No internal HTTP.** Server-side code composes services directly; nothing calls this
Flask app over HTTP from within itself.

---

## 3. Routes

| Route | Auth | Purpose |
|---|---|---|
| `/` | session | Main dashboard |
| `/auth/login`, `/auth/logout` | — / session | DHIS2-verified sign-in, sign-out |
| `/dhis2/` | session | Surveillance Data |
| `/preprocessing/` | session | Data Quality & Preparation |
| `/climate/` | session | Climate Signals |
| `/modelling/` | session | Model Evaluation |
| `/forecasts/` | session | Forecasts & Risk |
| `/how-it-works/` | session | How It Works |
| `/reports/`, `/reports/generate`, `/reports/print` | session | Reports & Exports |
| `/settings/` | session | Settings & Governance |
| `/api/dashboard/*` (6) | JSON 401 | summary, weekly-series, risk-table, map, climate-summary, model-performance |
| `/api/explainability/*` (5) | JSON 401 | overview, model-selection, validation-method, threshold-method, feature-importance |
| `/api/system/pipeline-status`, `/api/system/governance` | JSON 401 | system evidence |
| `/api/lifecycle/surveillance`, `/api/lifecycle/data-quality` | JSON 401 | lifecycle evidence |

Pages redirect anonymous users to login; APIs return **JSON 401** so `fetch` never
receives an HTML login page.

---

## 4. Key services

| Service | Responsibility |
|---|---|
| `dashboard_data.py` | Dashboard sections, parameter validation, selected-model resolution |
| `explainability.py` | Allowlisted artifact loading, importance contracts, pipeline status |
| `data_quality_evidence.py` | **The single** data-quality evidence builder (API + report share it) |
| `governance.py` | Settings & Governance evidence composition (no second implementation) |
| `report_data_service.py` | Report registry, request validation, report contexts |
| `report_export_service.py` | PDF/Excel/CSV rendering, formula-injection guard, temp files |
| `report_renderers.py` | Per-report PDF blocks and Excel sheets |
| `project_state.py` | Project-state audit (read-only calls memoised per request) |
| `risk_classification.py` | Risk classification (single source of truth) |

---

## 5. Analytical contract (do not change without re-validation)

- **Approved models:** Baseline/Naive, SARIMAX, Random Forest, XGBoost. No negative
  binomial model; no wind-speed predictor. Both exclusions are *verified* against the
  production manifest at runtime, not asserted.
- **Selected count models:** Random Forest at h4, h8 and h12
- **Selected surge model:** Random Forest at all three horizons
- **Count selection:** lowest WAPE; RMSE when WAPE unavailable
- **Surge selection:** require sensitivity >=0.60 and specificity >=0.40 where
  possible, then highest F2 with sensitivity, PR-AUC, precision and specificity as
  tie-breakers; **accuracy alone is never used**
- **Validation:** chronological rolling-origin, 12 numbered folds, no random split,
  preprocessing fitted on training rows only
- **Threshold:** district seasonal 75th percentile from valid training-only history in the target epi week +/-2; 2022 is excluded only from the normal-reference baseline
  (training-only). **Surge score = predicted cases - threshold, in cases, not a probability.**
- **Forecast:** target week only; intermediate weekly forecasts are not generated;
  intervals shown only when the selected count model publishes them
- **Missing data:** missing malaria reports stay missing and are never zero-filled;
  explicit zeros stay zero

---

## 6. Data dictionary (core surfaces)

| Field | Meaning |
|---|---|
| `district_id` | Canonical district key (e.g. `UGA_ABIM`) — the join key everywhere, including the map |
| `predicted_cases` | Count-model prediction for the target week |
| `surge_predicted_cases` | Surge-model prediction for the target week |
| `threshold_used` | District seasonal threshold (75th percentile of training-only normal-reference history) |
| `predicted_surge_score` | `surge_predicted_cases − threshold_used`, in **cases** |
| `risk_level` | Very High / High / Moderate / Low / No data, from `risk_classification.py` |
| `risk_basis` | Backend explanation of why that level was assigned |
| `reporting_completeness` | HMIS 033b form submission percentage |
| `issue_label` / `target_label` | Production issue week / forecast target week (e.g. 2025W52 → 2026W04) |
| Coverage | expected 45,698 − received 45,451 = **247 missing district-weeks (absent rows)** |

---

## 7. Operational notes

**Run:** `.venv\Scripts\python.exe run.py` (127.0.0.1:5000, `debug=False`).

**Requirements:** `requirements.txt` (Flask stack, psycopg2, requests) plus pinned
`XlsxWriter>=3.2,<4.0` and `reportlab>=4.0,<5.0` for exports. Chart.js and Leaflet are
**vendored locally** — the app makes no CDN, font or tile request and works offline
except for DHIS2 sign-in.

**Environment:** `DATABASE_URL` (PostgreSQL, mandatory — no SQLite fallback),
`SECRET_KEY`, `DHIS2_BASE_URL` and non-secret DHIS2 settings. **No DHIS2 username or
password is stored anywhere**; credentials are verified live at sign-in and discarded.

**Sign-in vs data:** login verifies live DHIS2 credentials but **does not pull new
malaria data**. The dashboard always shows the published production issue week.

**Sessions:** signed client-side Flask cookie, browser-session only, 30-minute
server-side reauthentication enforced in the user loader.

**Do not run** extraction, climate processing, dataset builds, training, validation,
production fitting, forecast generation or migrations against this dataset — the
published outputs are frozen and fingerprinted.

**Known test hazard:** `tests/test_production_forecasting.py` exercises publication
logic. It is transaction-isolated and rolls back, but treat it as sensitive; verify the
fingerprint after any change to it.

---

## 8. Academic interpretation notes

- A **High or Very High risk level is an early-warning signal for review**, not
  confirmation of an outbreak.
- **Climate variables are associated predictors, not established causes.** The pipeline
  computes no climate–malaria correlation and claims no best-correlating lag.
- **Reporting completeness measures form submission, not malaria burden.** Low completeness weeks may understate cases.
- **Feature importance is predictive, pooled and model-specific.** It is national-pooled
  validation evidence: it does not describe a single district and does not explain any
  individual production forecast. Per-prediction production contributions are **not**
  available and are reported as unavailable.
- **Negative permutation importance** means shuffling that predictor did not worsen
  validation MAE; the sign is preserved and never clamped.
- District aggregation can hide sub-district variation.

---

## 9. Deferred / not implemented

Scheduled report delivery, report subscriptions, public share links, PowerPoint export,
user administration, model administration, production-refresh controls, data-ingestion
controls, retraining, forecast regeneration, and audit-log administration. Settings is
**read-only governance** — it exposes no operational control.

### What can actually write state

To be precise, rather than claiming nothing writes:

- **User-configurable settings** are limited to `default_horizon` and `reduce_motion`.
  These are the *only* dashboard settings a user can change, they are stored
  **browser-local** (`localStorage`, versioned and allowlisted), and they never enter
  the Flask session, PostgreSQL or any analytical output.
- **Authentication writes session state.** Signing in creates a signed client-side
  session cookie and signing out clears it. No DHIS2 credential or token is stored.
- **Explicitly invoked CLI pipeline commands can write data and artifacts.** They are
  never triggered by the web application — only by an operator running them directly.
  `flask verify-project-state` also writes its proof artifact.

The presentation APIs and pages do not expose data-write controls. Governance and
report generation are explicitly guarded by tests that capture SQL statements and
reject inserts, updates, deletes and schema-changing statements.

**Do not run the pipeline commands against this frozen production dataset** (see
Operational notes). The published outputs are fingerprinted and must stay unchanged.

---

## 10. Verification commands

```bash
.venv\Scripts\python.exe -m compileall app tests
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m flask --app run.py verify-project-state   # writes an artifact
```

Integrity to re-check after any change: 1,752 forecast rows · IDs 11–1765 · 146 active
districts · 0 null predictions · 0 duplicate keys · unchanged fingerprint · map response
below 2 MB · `app.json.compact` True · `run.py` `debug=False`.
