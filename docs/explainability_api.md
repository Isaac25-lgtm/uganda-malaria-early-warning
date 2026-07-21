# Explainability & System APIs (Phase 29B)

Read-only, authenticated APIs that expose **already-published** evidence. They
never train, predict, recalculate a winner, write to the database, or load a
production model. Every artifact is read through a symbolic allowlist, validated,
and scrubbed of paths/secrets before it is returned.

**Auth:** existing DHIS2-backed session with 30-minute expiry. Unauthenticated
requests receive JSON `401` (`status: "unauthenticated"`), matching the dashboard
API convention.

**Envelope (all endpoints):**

```json
{"status":"available","reason":null,"data":{},"params":{},
 "empty_states":[],"metadata":{},"warnings":[],"limitations":[]}
```

Invalid parameters → `400` `status:"invalid_request"` (values are **rejected,
never clamped**). Missing/malformed evidence → `503` `status:"unavailable"`.

**Fail-closed & schema validation (29B corrective).** Every JSON artifact is
schema-validated on load (root type, required blocks/fields, value types,
supported horizons); NaN/Infinity (and overflow like `1e999`) are rejected. Any
malformed shape/type/value, or absent required evidence, becomes a controlled
`503` — no `AttributeError`/`TypeError`/`ValueError`/stack trace escapes.
`validation-method` will not claim 12 folds when validation evidence is absent;
`overview` requires selection **and** forecast evidence; `model-selection`
requires a winner and comparison; `pipeline-status` requires forecast evidence
and a database-verified active-district count (never hardcoded).

**Final schema closure (Stage A).** Selection evidence must carry exactly
horizons 4/8/12 in both selected blocks and **exactly four unique approved model
comparisons** per horizon (Baseline/Naive, SARIMAX, Random Forest, XGBoost), with
finite winning metrics (count: WAPE or RMSE; surge: F1 and sensitivity) — winners
are preserved, never recalculated. Forecast evidence must carry integer issue/target
weeks for all three horizons, consistent row counts, and `published=true`/
`status=success`; a null target week never returns 200. The 12-fold claim is made
**only** from fold-bearing tree validation (`random_forest`/`xgboost`) with
`validation_method=rolling_origin`, folds 1–12 per horizon, and the leakage flags
(`no_random_shuffle`, `horizon_aware_gap`, `per_origin_threshold_training_only`,
`trainable_targets_observed_by_origin`) all true. Importance CSV closure also
rejects methods outside the two approved per model, blank fields, and negative
native impurity/gain, and validates every `overall` against the mean of folds 1–12
during structural validation. Every violation → controlled **503**.

## Endpoints

| Endpoint | Parameters | Source evidence |
|---|---|---|
| `GET /api/explainability/overview` | `horizon=4\|8\|12` (default 4) | `forecast_generation_summary.json`, `model_performance_selection_summary.json` |
| `GET /api/explainability/model-selection` | `horizon=4\|8\|12` | `model_performance_selection_summary.json` |
| `GET /api/explainability/validation-method` | — | `random_forest_validation_summary.json` |
| `GET /api/explainability/threshold-method` | `horizon`, optional `district_id`, `issue_year`/`issue_week` | dashboard risk services (published values reused) |
| `GET /api/explainability/feature-importance` | `model=random_forest\|xgboost`, `horizon`, `method=native\|permutation`, `limit=1-50`, `grouped=true\|false` | `*_feature_importance.csv`, `*_model_card.json` |
| `GET /api/system/pipeline-status` | — | `build_project_state_summary(write=False)`, `forecast_generation_summary.json` |

## Feature-importance contract

**CSV structural validation (all violations → 503, never warnings):** `model_name`
must match the artifact (`Random Forest`/`XGBoost`); horizons restricted to 4/8/12;
folds must be exactly 1–12 plus `overall` (fold 0/13 or arbitrary strings rejected);
no duplicate feature/fold/method/horizon rows; every feature must appear in all 12
  numbered folds; exactly **181 native** / **28 permutation** features; each `overall`
row must equal the mean of that feature's folds 1–12. `fold_count_used` is reported
as 12 only after this verification.

- **Native** (`impurity` for Random Forest, `gain` for XGBoost): 181 transformed
  features. Means come from **numbered folds 1–12**; `overall` is validated against
  that mean and never summed in. Values are **normalized across all 181 features
  before `limit`**. Zero-importance features retained; `is_zero_importance` exact.
- **Permutation** (`permutation_validation_mae`): 28 original predictors, including
  eight explicit case-interpolation indicators.
  **Signed** mean MAE change, folds 1–12; negatives preserved (never clamped).
  Optional `positive_share_percentage` uses `max(value,0)` but the signed value is
  always present and labelled.
- **`limit` applies to both modes:** `grouped=false` returns ≤ `limit` ranked
  features plus the full `group_summary`; `grouped=true` returns ≤ `limit` ranked
  **families** (each with `family_rank` and `fold_coverage`), omits individual
  features, and does not duplicate the group summary.
- **Families:** Malaria History, Climate Signals, Time and Season, District
  Identity, Missing-Value Indicators, Other. Native grouped **before**
  normalization; permutation grouped as **signed** raw means.
- **`method_notes`** state that RF impurity can favour highly splittable /
  high-cardinality features, native tree importance is model-specific, and a
  negative permutation value means shuffling did not worsen validation MAE.
- **Readable labels**, e.g. `numeric__cases_rolling_mean_8` → "8-week rolling mean
  of malaria cases"; `numeric__temperature_lag_12` → "Temperature 12 weeks earlier";
  `district__district_id_UGA_YUMBE` → "District identity: Yumbe";
  `numeric__missingindicator_cases_lag_4` → "Missing-value indicator: malaria cases
  4 weeks earlier".

## Other corrected contracts

- **model-selection** exposes explicit `metric_name`/`metric_value` — count uses
  **WAPE** when present else **RMSE**; surge uses **F1** with **sensitivity** named
  as tie-breaker. Metrics are finite JSON numbers. Winners are read from evidence
  (`winner_recalculated: false`).
- **threshold-method** exposes the exact **Low/Moderate/High/Very High** bands
  (ratio = surge-model predicted cases ÷ district threshold), mirroring
  `app.services.risk_classification` (single source; not reimplemented). Per-district
  values, including `risk_basis`, are reused from the published forecast; no
  probability is fabricated and no outbreak is implied.
- **pipeline-status** returns an evidence-derived `overall_state`, a
  **database-verified** `active_districts` count (`active_districts_verified: true`),
  component statuses, model/dashboard status, and evidence timestamps.

## What is deliberately NOT provided

- No per-prediction / production contribution endpoint (validation-row
  contributions must not be presented as explanations of the 2025W52 forecast).
- No district-specific importance (tree-model importance is national-pooled).
- No causal claims; importance is predictive association only.
- Accuracy appears as secondary context because it exists in the artifact, but it
  is explicitly stated to have **not** been used for model selection.
