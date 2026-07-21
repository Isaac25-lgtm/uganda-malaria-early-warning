# Malaria EWS Cleaning, Threshold and Retraining Report

## Completion status

Completed and verified on 2026-07-21. The modelling dataset, all four validation
pipelines, model selection, production models, 1,752 production forecasts,
dashboard evidence, reports, explainability and governance outputs were rebuilt.

## Approved case-error handling

The raw DHIS2/database values were not edited. Six researcher-confirmed reporting
errors, documented as having District Biostatistician consultation, are registered
in `data/reference/malaria_case_exclusions.csv`:

| District-week | Original | Previous | Following |
|---|---:|---:|---:|
| GULU CITY 2021W50 | 5,892 | 768 | 763 |
| IBANDA 2025W25 | 40,963 | 787 | 1,140 |
| JINJA 2021W34 | 7,233 | 556 | 550 |
| KALIRO 2024W31 | 14,413 | 2,835 | 2,433 |
| KISORO 2025W15 | 1,727 | 30 | 34 |
| KISORO 2025W17 | 557 | 34 | 45 |

For analysis, each is missing as an outcome, validation truth and threshold-history
observation. It is never outcome-imputed. A bounded linear interpolation is used
only when constructing predictor lags/rolling histories. The same predictor-only
rule applies to internal source gaps of one or two weeks. Interpolation is not used
for leading, trailing or longer gaps, and every interpolated predictor is flagged.

Evidence totals: 247 absent source district-weeks, 6 confirmed errors, 253 analytical
missing outcomes, 19 bounded source gaps interpolated for predictors, 6 confirmed
errors interpolated for predictors, 25 predictor-only interpolations in total.

## Modelling changes

- Kept 2022 in case-count model training; it was not deleted or cleaned away.
- Removed population and incidence from the modelling CSV, predictor contract,
  preflight gate, dashboard methods, reports and governance narrative. The legacy
  database columns/import utility remain untouched and are not model inputs.
- Rebuilt 45,451 district-week rows across 146 districts with 34 model features.
- Added eight interpolation-status predictors for case lags and rolling windows.
- Retained training-fold-only imputation and one-hot encoding; outcomes are never
  imputed and no random split is used.
- Changed the 12 rolling-origin folds from the last 12 eligible origins to 12
  evenly spaced historical origins, improving multi-year temporal coverage.
- Replaced the non-seasonal surge reference with a district-specific training-only
  Q75 for the target epidemiological week plus/minus two weeks. The confirmed 2022
  epidemic stays in count training but is excluded from the normal-reference
  threshold. Fewer than eight seasonal observations triggers district expanding-
  history fallback.
- Added separate class-balanced Random Forest and XGBoost surge classifiers. Random
  Forest uses class weights; XGBoost uses `scale_pos_weight`; no SMOTE is used.
- Added F2, PR-AUC and Brier score. Candidate probability cutoffs must preferably
  meet sensitivity >= 0.60 and specificity >= 0.40, then are ranked by F2,
  sensitivity, PR-AUC, precision and specificity. Accuracy is not a selector.
- Compared each dedicated classifier with count forecast versus seasonal threshold.
  The dedicated classifiers produced excessive false alerts, so the count-threshold
  method was selected operationally for RF and XGBoost while classifier metrics
  remain in the model cards for transparency.

## Final validation results

Random Forest was selected for both count forecasting and operational surge warning
at 4, 8 and 12 weeks.

| Horizon | MAE | RMSE | WAPE | R-squared | Sensitivity | Specificity | Precision | F1 | F2 | PR-AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 393.75 | 627.54 | 34.31% | 0.695 | 73.3% | 71.9% | 53.7% | 62.0% | 68.3% | 0.626 |
| 8 | 462.76 | 693.42 | 38.56% | 0.656 | 73.1% | 66.5% | 51.4% | 60.4% | 67.4% | 0.607 |
| 12 | 460.33 | 701.07 | 39.64% | 0.636 | 72.8% | 69.3% | 52.7% | 61.1% | 67.6% | 0.609 |

Against the baseline under the same revised validation protocol, RF reduced WAPE by
29.72% (h4), 20.45% (h8) and 16.13% (h12). These are relative error reductions, not
claims of percent accuracy.

XGBoost WAPE was 36.44%, 40.38% and 40.70%; its operational surge sensitivity was
69.0%, 69.9% and 72.8%, with specificity 69.4%, 64.1% and 65.5%.

The old and new validation numbers are not directly comparable because the old run
used the last 12 eligible origins while the revised run spreads 12 origins across
the study period and also changes cleaning and surge labels. Under the more demanding
multi-year test, count WAPE is higher than the old estimate; this is an honest
generalisation estimate, not evidence that cleaning alone worsened the model. Surge
sensitivity and F1 are substantially better, with lower specificity as the expected
trade-off for finding more true surge weeks.

## Production output and transactional event

The first forecast publication attempt failed because the new threshold description
exceeded the existing 120-character database field. The transaction rolled back:
the prior 1,752 rows stayed intact and pipeline run 6906 was marked failed. The label
was shortened without a migration, a regression test was added, and run 6907 then
completed successfully.

Current production integrity: issue 2025W52; targets 2026W04, 2026W08 and 2026W12;
1,752 rows; IDs 11-1765; 146 active districts; 146 rows in every model-horizon slice;
0 null case predictions; 0 duplicate, missing or unexpected composite keys. Canonical
fingerprint: `2854b51afc0e06e1284232a584b4a8b91e3ea60e6a79e82452b850396cc144f9`.

## Verification

- `python -m compileall app tests`: PASS.
- `python -m unittest discover -s tests -v`: 399 tests, 0 failures.
- `node --check`: all 9 JavaScript files pass.
- `flask verify-project-state`: all nine components complete; Prompt 29 gate yes.
- HTTP smoke test: login 200, anonymous dashboard 302 to login, anonymous API 401.
- No database migration or schema change; raw DHIS2 and climate source data unchanged.
- Dashboard available at `http://127.0.0.1:5000`; real DHIS2 credentials are required
  and are not persisted by the application.

## Source files changed

New files: `app/services/case_data_cleaning.py`,
`data/reference/malaria_case_exclusions.csv`, `tests/test_case_data_cleaning.py`, and
this report.

Modified services: `dashboard_data.py`, `data_quality_evidence.py`,
`explainability.py`, `forecast_generation.py`, `governance.py`,
`model_performance.py`, `model_validation.py`, `modelling_dataset.py`,
`production_models.py`, `project_state.py`, `report_data_service.py`,
`report_renderers.py`, and model services `baseline.py`,
`pooled_tree_validation.py`, `random_forest.py`, `sarimax.py`, `xgboost_model.py`.

Modified presentation/documentation: `app/static/js/dashboard.js`,
`app/static/js/settings.js`, `app/templates/dashboard/index.html`, lifecycle
`climate.html`, `data_quality.html`, `surveillance.html`,
`docs/explainability_api.md`, and `docs/HANDOVER.md`.

Modified tests: `test_dashboard_data.py`, `test_explainability.py`,
`test_governance.py`, `test_hardening.py`, `test_lifecycle.py`,
`test_model_validation.py`, `test_pooled_tree_validation.py`,
`test_production_forecasting.py`, `test_reports.py`, and `test_sarimax.py`.

Generated outputs were rebuilt under `data/modelling`, `data/models` and
`data/quality`, including validation predictions and metrics, fit diagnostics,
feature importance, model cards, checkpoints, 447 final production model artifacts,
the production manifest, forecast-generation evidence, model-selection evidence,
cleaning evidence, modelling-dataset evidence and project-state evidence.

## Interpretation limits

The selected model is useful for research early warning but is not a confirmed
outbreak detector. WAPE of 34-40% means case-count error remains material. Sensitivity
near 73% means roughly 27% of labelled surges were missed in validation; precision
near 51-54% means a meaningful share of alerts were false positives. Results require
epidemiological review, reporting-quality review and external/temporal validation
before operational deployment or a claim of clinical/public-health effectiveness.
