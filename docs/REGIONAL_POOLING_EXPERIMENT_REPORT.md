# DHIS2-Subregion Pooling Experiment — Random Forest and XGBoost

**Status:** Experimental. This report documents a controlled experiment that does **not**
replace production models, forecasts, or the model-selection artefact, and promotes no model.
No DHIS2 login, network pull, source refresh, schema change, UI change, or production forecast
was performed. All experimental outputs are written beneath `data/experiments/regional_pooling/`.

---

## 1. Research rationale

The production Random Forest and XGBoost models are **national-pooled**: one model family is
trained on district-week observations from all 146 districts, with `district_id` fitted as a
training-fold-only one-hot feature. Malaria transmission may behave differently across Uganda's
operational **DHIS2 subregions** because climate, altitude, ecology, seasonality, health-service
access and reporting behaviour vary geographically.

This experiment tests whether **pooling districts within their existing DHIS2 subregions**
improves future-week prediction relative to national pooling, using matched, leakage-safe,
forward-in-time validation. It does **not** assume regional models are better; the null position
is that national pooling remains preferable unless matched out-of-time evidence and its
uncertainty support a different conclusion.

Four strategies are compared, differing **only** in training scope:

1. National-pooled Random Forest (the existing national strategy).
2. DHIS2-subregion-pooled Random Forest with explicit national fallback.
3. National-pooled XGBoost.
4. DHIS2-subregion-pooled XGBoost with explicit national fallback.

Everything else is held identical: the same modelling dataset, the same 4/8/12-week targets, the
same approved predictors, the same cleaning and missing-outcome rules, the same district-seasonal
threshold method (2022 excluded from the normal reference only), the same hyperparameter grids and
train-only tuning, the same deterministic seed, the same temporal origins and target weeks, the
same count/surge definitions, and the same selection metrics.

---

## 2. Architecture and files

The experiment reuses the single leakage-safe engine in
`app/services/models/pooled_tree_validation.py`. No preprocessing, threshold, metric, classifier,
fold, or publication logic is duplicated.

**New files**

| File | Purpose |
| --- | --- |
| `app/services/models/regional_mapping.py` | Strict, read-only loader for the reviewed DHIS2 subregion mapping, with fail-closed validation against the confirmed audit. |
| `app/services/models/regional_tree_validation.py` | Orchestration: fold scoping, feasibility audit, national-baseline reuse-or-rebuild, per-subregion validation, regional-with-fallback assembly, atomic publication, matched comparison, and district-cluster bootstrap. |
| `tests/test_regional_tree_validation.py` | 15 tests covering mapping validation, fold scoping, feasibility, fallback assembly, and an end-to-end real-fit run. |
| `docs/REGIONAL_POOLING_EXPERIMENT_REPORT.md` | This report. |

**Extended files (backward-compatible; national behaviour preserved byte-for-byte by default)**

| File | Change |
| --- | --- |
| `app/services/models/pooled_tree_validation.py` | Added optional `rows`, `fixed_folds_by_horizon`, `scope_signature`, `model_scope_label`, and `compute_permutation_importance` arguments to `run_pooled_tree_validation`, all defaulting to the exact prior national behaviour. |
| `app/services/models/random_forest.py`, `xgboost_model.py` | Added thin `run_regional_*_experiment` wrappers. |
| `app/commands/modelling.py` | Added CLI commands `audit-regional-pooling`, `train-regional-random-forest`, `train-regional-xgboost`, `compare-regional-pooling`. |

**Protected and unchanged:** `data/quality/model_performance_selection_summary.json`,
`data/models/production/`, the national RF/XGBoost validation artefacts,
`forecast_generation.py`, `production_models.py`, `dashboard_data.py`, the model-selection rules,
dashboard APIs/templates/JS/CSS, authentication, schema/migrations, and all source data. Their
SHA-256 hashes were recorded before the experiment and re-verified afterward (Section 8).

---

## 3. Mapping audit

Source: `data/reference/dhis2_orgunit_mapping_review.csv` (frozen, reviewed). The loader fails
closed unless the mapping reproduces the confirmed audit exactly. Verified:

- 146 district rows, 146 unique `district_id` values, zero blank `parent_name`, zero duplicate or
  contradictory assignments.
- 15 subregions with the exact expected district counts:

| Subregion | Districts | Role |
| --- | --- | --- |
| Acholi | 9 | regional pooled |
| Ankole | 13 | regional pooled |
| Bugisu | 10 | regional pooled |
| Bukedi | 7 | regional pooled |
| Bunyoro | 9 | regional pooled |
| Busoga | 12 | regional pooled |
| **Kampala** | **1** | **national fallback (single district)** |
| Karamoja | 9 | regional pooled |
| Kigezi | 6 | regional pooled |
| Lango | 10 | regional pooled |
| North Buganda | 12 | regional pooled |
| South Buganda | 14 | regional pooled |
| Teso | 11 | regional pooled |
| Tooro | 10 | regional pooled |
| West Nile | 13 | regional pooled |

**Kampala** is a single district and is **never** trained as a regionally pooled model. It is
served by the matching national candidate, labelled `scope_source=national_fallback_single_district`.
Fourteen subregions are poolable.

The experiment also cross-checks the mapping against the active 146-district frame in the
modelling dataset: every dataset district is mapped, and every mapped district is in the frame.

---

## 4. Validation design (leakage safety)

- The shared national rolling-origin folds (12 evenly spaced folds per horizon; origins begin
  `2020-12-28`, `2021-06-07`, …) are built **once** and **scoped** to each subregion, so national
  and regional candidates use **identical origin and target dates**. Regional dates are never
  independently selected.
- For every subregion, horizon and fold, training rows are filtered to the subregion; the model
  trains only on rows whose target was observable by the fold origin; validation is only on the
  matching future target week.
- Imputation, missing indicators and district one-hot encoding are fitted on that fold's scoped
  training rows only. Outcomes are never imputed.
- District surge thresholds are computed only from that district's approved training-only history
  (target epi week ±2, 2022 excluded from the normal reference). Because a district's full history
  is present within its own subregion, its threshold is **identical** to the national one — verified
  by a unit test. No shared regional threshold is computed.
- Hyperparameters are tuned per subregion on an inner chronological holdout of the earliest scoped
  training fold; outer validation outcomes are never inspected. Regional tuning may select different
  parameters than national.

**Regional-with-fallback rule:** for each matched key, the combined output uses the regional
prediction where a valid regional candidate exists, otherwise the matched national prediction,
recording the reason. National and regional forecasts are **never averaged**.

---

## 5. Feasibility audit (pre-training)

Computed before any fitting, over all 15 subregions × 3 horizons × 12 folds = **540 cells**. For
each cell it reports district count, trainable target rows, validation rows, surge-positive and
surge-negative validation counts, prevalence, and the fallback decision. Highlights:

- **36 cells** require national fallback — exactly the 36 Kampala cells (1 subregion × 3 horizons
  × 12 folds), all `national_fallback_single_district`. Every poolable subregion/horizon/fold has
  non-empty scoped training and validation targets (`count_candidate_feasible = true`), so no
  poolable cell needed the no-regional-target fallback.
- Example cells (Acholi, horizon 4): fold 1 — 441 trainable rows, 9 validation rows, 0 surge-positive
  / 9 surge-negative; fold 2 — 648 trainable rows, 9 validation, 2 surge-positive / 7 surge-negative.

A regional surge classifier fails closed or uses the approved controlled fallback when its training
labels do not contain both classes; every such event is recorded in the run diagnostics. The
operational surge decision uses the count-versus-district-threshold method, consistent with the
national engine.

The full per-cell table is written to `data/experiments/regional_pooling/feasibility_audit.json`.

---

## 6. National baseline (matched comparison)

The existing national validation predictions are accepted for the matched comparison only if the
national summary reports a published success, the modelling-dataset signature matches, fold origins
and target dates match the requested horizons/max-folds, and prediction keys are complete and
unique. Both families passed these checks (`reusable = true`), so the comparison matches each
regional prediction against the **exact production national prediction** on
`(model family, horizon, validation fold, district_id, target week)`. If any check had failed, a
fresh experimental national baseline would have been built under
`data/experiments/regional_pooling/national_baseline/` without overwriting production.

---

## 7. Metrics and interpretation rules

- **Count:** MAE, RMSE, MAPE, WAPE, R² — nationally, by subregion, and by district. Count selection
  reuses the production rule (lowest WAPE where available, RMSE fallback).
- **Surge:** sensitivity, specificity, precision, F1, F2, ROC-AUC, PR-AUC, Brier. Surge selection
  reuses the production rule (highest F2, then sensitivity, PR-AUC, precision, specificity; accuracy
  never selects).
- **Relative WAPE reduction** = (national WAPE − regional WAPE) / national WAPE × 100, labelled
  "relative error reduction," never "percentage accuracy."
- **Uncertainty:** a deterministic **district-cluster bootstrap** (1000 iterations, fixed seed)
  resamples complete district histories and reports a 95% interval for the paired pooled-WAPE
  difference and relative reduction. An interval that includes zero is not evidence of improvement.
  Subregions with too few districts for reliable inference are flagged rather than asserted superior.

---

## 8. Production integrity

Recorded before the experiment and re-verified after (the experiment writes only under
`data/experiments/regional_pooling/` and performs no database writes — `record_pipeline_run` is
`False` throughout):

- `data/quality/model_performance_selection_summary.json` — `17d66dba…` (unchanged)
- `data/quality/random_forest_validation_summary.json` — `88f224bd…` (unchanged)
- `data/quality/xgboost_validation_summary.json` — `c6d23733…` (unchanged)
- `data/modelling/random_forest_validation_predictions.csv` — `14bce6ca…` (unchanged)
- `data/modelling/xgboost_validation_predictions.csv` — `7ced8f98…` (unchanged)
- `app/services/model_performance.py` — `70ffd937…` (unchanged)

Frozen production-forecast record (`data/quality/forecast_generation_diagnostics.json`): issue week
2025W52, 1,752 rows, 146 districts, 0 null predictions, 0 duplicate composite keys, content
fingerprint `5046f7cd…`. The production database was not opened by any experiment code path.

---

## 9. Tests

- `tests/test_regional_tree_validation.py` — 15 tests, all passing: exact 146-district / 15-subregion
  mapping and counts; rejection of missing, duplicate, contradictory and out-of-frame mappings;
  Kampala as explicit national fallback; regional training contains no other subregion's rows;
  national and regional candidates share identical folds; district thresholds identical national vs
  scoped; fallback assembly never averages; WAPE-reduction math is correct and never labelled
  accuracy; a dry-run performs no writes; and an end-to-end real-fit run whose comparison keys match
  the national baseline exactly.
- National regression preserved: `tests/test_pooled_tree_validation.py` (8), `test_model_validation.py`
  (11) and `test_model_performance.py` (6) all pass unchanged, confirming the national-pooled path is
  byte-for-byte unaffected by the additive engine extensions. Combined with the 15 regional tests,
  **40 tests pass**.
- Test modules that construct the Flask app and query PostgreSQL were not exercised in this session
  because the local database server was not accepting connections; the experiment itself performs no
  database access, and the modules above cover the code paths this work touches.

---

## 10. Reproducibility and environment notes

- Deterministic seed `24024`; source SHA-256 hashes, package versions, timestamps and configuration
  are recorded in each run's `fit_diagnostics.json` and the manifest.
- **Full methodological fidelity.** The final, reported run fitted regional candidates at the
  production controlled thread count (`controlled_threads: 2`) and computed permutation importance
  exactly as the national runs do, so the only difference between the compared strategies is training
  scope. This is verifiable in the published artefacts: both `fit_diagnostics.json` files record
  `controlled_threads: 2` and `permutation_importance_computed: true`, and each family's
  `feature_importance.csv` contains both native (`impurity`/`gain`) and `permutation_validation_mae`
  rows.
- An earlier attempt on this machine was throttled by an active power-saver plan, which made
  `n_jobs = 2` worker-process spawning unusable. A temporary single-threaded, no-permutation
  workaround was used during that attempt and is **not** part of the reported results: once the power
  plan was corrected, the workaround was reverted, the throttled partial output was deleted, and both
  families were rerun from scratch under the configuration described above. Thread count changes only
  parallelism, not results (RandomForest and XGBoost are deterministic with respect to `n_jobs` given
  a fixed `random_state`), so this history does not affect the final artefacts.

---

## 11. How to run

The experiment reads the frozen modelling dataset and requires no DHIS2 credentials or network.
Under a normally provisioned environment with the database available:

```bash
flask modelling audit-regional-pooling
flask modelling train-regional-random-forest --horizons 4,8,12 --max-folds 12 --dry-run
flask modelling train-regional-random-forest --horizons 4,8,12 --max-folds 12
flask modelling train-regional-xgboost --horizons 4,8,12 --max-folds 12
flask modelling compare-regional-pooling --models random_forest,xgboost --horizons 4,8,12
```

Runs are resumable: each subregion/horizon/fold writes a checkpoint keyed by dataset signature,
scope, selected parameters, fold and horizon. Re-invoking the same command reuses completed folds
instantly and continues from the first incomplete one. Use `--replace` to force a refit.

---

## 12. Results

Both model families completed the full design: 14 poolable subregions × 3 horizons × 12 folds =
**504 fold-jobs per family**, all successful. Random Forest ran in 3,726 s and XGBoost in 2,810 s.
Both reused the existing production national artefact as the matched baseline
(`national_baseline_source = existing_production_artifact`), and both matched comparisons are
**complete** (`comparison_complete = true`) with no unmatched keys.

Scope accounting is identical for both families and exactly as designed:
**5,215 regional predictions + 36 national-fallback = 5,251**, matching the national prediction-key
count. The 36 fallbacks are precisely Kampala (1 district × 3 horizons × 12 folds), labelled
`national_fallback_single_district`. No poolable subregion/horizon/fold required a
no-regional-target fallback, and no prediction was ever averaged.

### 12.1 Count performance — national vs regional-with-fallback

**Random Forest**

| Horizon | National WAPE | Regional WAPE | Relative WAPE reduction | Districts improved | District-cluster bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- |
| 4 weeks | 34.311% | 34.301% | +0.030% | 79 / 146 | [−2.39, +2.53] — includes 0 |
| 8 weeks | 38.562% | 37.934% | +1.628% | 75 / 146 | [−0.71, +4.11] — includes 0 |
| 12 weeks | 39.643% | 39.853% | −0.531% | 73 / 146 | [−3.17, +2.25] — includes 0 |

**XGBoost**

| Horizon | National WAPE | Regional WAPE | Relative WAPE reduction | Districts improved | District-cluster bootstrap 95% CI |
| --- | --- | --- | --- | --- | --- |
| 4 weeks | 36.440% | 34.948% | +4.093% | 97 / 146 | **[+1.12, +7.13] — excludes 0** |
| 8 weeks | 40.382% | 38.985% | +3.459% | 85 / 146 | **[+0.79, +6.37] — excludes 0** |
| 12 weeks | 40.703% | 39.952% | +1.846% | 86 / 146 | [−1.01, +4.95] — includes 0 |

Positive values favour regional pooling. These are **relative error reductions, not accuracy**.

### 12.2 Cross-family ranking (the decisive comparison)

Pooling scope cannot be judged inside one family alone, because the production count model is
selected across families by lowest WAPE. Ranking all four strategies:

| Horizon | National RF | Regional RF | National XGB | Regional XGB | Lowest WAPE |
| --- | --- | --- | --- | --- | --- |
| 4 weeks | 34.311% | 34.301% | 36.440% | 34.948% | regional RF (ties national RF) |
| 8 weeks | 38.562% | **37.934%** | 40.382% | 38.985% | regional RF |
| 12 weeks | **39.643%** | 39.853% | 40.703% | 39.952% | national RF |

**Random Forest remains the strongest family at every horizon regardless of pooling scope.** The
statistically supported XGBoost gains do not lift regional XGBoost above either Random Forest
variant at any horizon — regional pooling narrows XGBoost's gap to Random Forest rather than
producing a new leader.

### 12.3 Surge detection trade-off

Regional pooling systematically **trades sensitivity for specificity** in both families:

| Family | Horizon | National sens / spec / F2 | Regional sens / spec / F2 |
| --- | --- | --- | --- |
| Random Forest | 4 | 0.733 / 0.719 / 0.683 | 0.701 / 0.757 / 0.668 |
| Random Forest | 8 | 0.731 / 0.665 / 0.674 | 0.682 / 0.719 / 0.648 |
| Random Forest | 12 | 0.728 / 0.693 / 0.676 | 0.687 / 0.718 / 0.650 |
| XGBoost | 4 | 0.690 / 0.694 / 0.642 | 0.673 / 0.765 / 0.648 |
| XGBoost | 8 | 0.699 / 0.641 / 0.643 | 0.692 / 0.712 / 0.655 |
| XGBoost | 12 | 0.728 / 0.655 / 0.666 | 0.676 / 0.726 / 0.643 |

For **Random Forest, regional pooling lowers sensitivity at all three horizons** (0.733→0.701,
0.731→0.682, 0.728→0.687) and lowers F2 at all three. Under the approved surge-selection rule —
highest F2, which deliberately weights sensitivity for early warning — this is the **wrong
direction**: fewer true surges detected in exchange for fewer false alerts. XGBoost gains slightly
on F2 at 4 and 8 weeks but loses at 12.

### 12.4 Conclusion

**Regional pooling produces mixed and inconclusive evidence.**

- For **XGBoost**, DHIS2-subregion pooling gives a genuine, uncertainty-supported reduction in count
  error at 4 and 8 weeks (bootstrap intervals exclude zero, ~2 of 3 districts improved). This is a
  real finding and is retained.
- For **Random Forest** — the model currently selected for production at all three horizons —
  regional pooling gives **no reliable count improvement** (every bootstrap interval includes zero,
  district improvement is near a coin-flip at 73–79 of 146) and **degrades surge F2 and sensitivity
  at all three horizons**.
- Regional XGBoost never overtakes Random Forest, so the XGBoost gain does not change which model
  would be selected.

Accordingly, this experiment provides **no basis for adopting subregion pooling in production**, and
no model is promoted. The words "better" are withheld: the only uncertainty-supported improvements
are confined to one non-selected family at two of three horizons, and they are accompanied by a
sensitivity loss that runs against the early-warning objective. A defensible next step would be a
prespecified study of *why* subregional scope helps the gradient-boosted family but not the forest —
for example whether it reflects regularisation and training-set size rather than genuine
geographic heterogeneity — with nested tuning and a temporal holdout not used for selection.

### 12.5 Compute note

Both families completed at **full methodological fidelity**: regional candidates inherit the
production controlled thread count and permutation importance is computed exactly as in the national
runs, so no configuration differs between the compared strategies other than training scope.

An earlier attempt on this machine was throttled by an active power-saver plan, which pinned the CPU
below its base clock and made worker-process spawning unusable (a single Random Forest fit that takes
0.33 s at the production thread count was observed at 79 s, and `n_jobs=2` did not complete at all).
With the power plan corrected, the same fit runs in 0.33 s and the complete experiment — 1,008
fold-jobs across both families, plus tuning, bootstrap and publication — finished in about 110
minutes total (Random Forest 3,726 s, XGBoost 2,810 s). No results from the throttled attempt were
carried forward; that partial output was deleted and both families were rerun from scratch under the
final configuration.

### 12.6 Verification scope (what is and is not confirmed)

The **40 focused modelling tests** (regional 15, pooled 8, model_validation 11, model_performance 6)
pass, and all protected artefacts are hash-identical to baseline. During the experiment session the
~20 test modules that construct the Flask app and query PostgreSQL could not run because the local
database service was unavailable.

**Full-suite verification has since been completed.** With PostgreSQL restored,
`python -m unittest discover -s tests -v` passes end to end: **430 tests, 0 failures, 0 errors, 0
skipped** (the single `OSError: disk full` line in the log is a deliberately injected mock in
`test_reports` exercising graceful disk-full handling, and is immediately followed by `ok`).
`python -m compileall app tests` is clean. The experiment itself performs no database access, so its
integrity never depended on the database being available.

### 12.7 Artefacts

All published beneath `data/experiments/regional_pooling/`:

- `random_forest/` and `xgboost/` — `validation_summary.json`, `validation_predictions.csv`
  (with `subregion` and `scope_source` columns), `district_horizon_metrics.csv`,
  `subregion_horizon_metrics.csv`, `fit_diagnostics.json`, `model_card.json`,
  `feature_importance.csv`, plus per-subregion artefact sets and resumable checkpoints.
- `comparison_summary_random_forest.json`, `comparison_summary_xgboost.json`,
  `comparison_rows_random_forest.csv`, `comparison_rows_xgboost.csv`.
- `region_mapping.csv`, `feasibility_audit.json`.
