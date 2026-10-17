# Literature-based forecasting benchmark — isolated comparison study

> **Isolated, supervisor-requested comparison.** NOT part of the frozen main
> forecasting pipeline, dashboard, production model-selection workflow, or
> dissertation results unless separately approved.

**Question:** do selected literature malaria-forecasting methodologies
outperform the frozen current Uganda district-week models under the *same*
leakage-controlled 12-fold rolling-origin evaluation? **Not** an attempt to
prove Random Forest is best — the evidence is reported exactly as it is.

## Execution (standalone module — no Flask, no `app/cli.py`)
```
python -m research_comparisons.literature_based_forecasting_benchmark.src.run --run-id 20260828 [ACTION]
```
Actions: `--baseline`, `--validate-gate`, `--dry-run`, `--final-integrity`, or
(no flag) the full run once the model code is implemented.

## Layout
```
literature_based_forecasting_benchmark/
  config/        frozen config, canonical/frozen registries, method classification
  src/           benchmark code (integrity.py, run.py, + model modules in later phases)
  tests/         focused benchmark tests
  literature/    pointer to the per-run literature gate (lives under runs/<id>/literature/)
  runs/<id>/     ALL outputs for a run: baseline/final manifests, literature gate,
                 fold manifest, predictions, logs, metrics, bootstrap, plots, report
  docs/          study notes
```

## Revised method classification (after the literature gate)
- **PRIMARY (winner table):**
  1. `zinszer_arimax` — Zinszer-inspired **adapted** district-level ARIMAX / dynamic regression.
  2. `tekle_poisson_dl` — Teklehaimanot-inspired adapted Poisson autoregressive distributed-lag.
- **NON-PRIMARY (reported separately, kept OUT of the winner table):**
  - `katwesige_nb` — Katwesige-inspired adapted Negative Binomial **sensitivity analysis**
    (authoritative full Makerere thesis not accessible → NB specifics not verifiable → not primary).
  - `davis_exploratory_nb_dl` — Davis-inspired reduced NB distributed-lag **exploratory** analysis.
- **EXCLUDED:** `ippolito_zambia` (seasonal burden/totals target, single low-transmission catchment —
  materially different method); `epidemia_2017` (conceptual integration system, not one specified model).

## Reused canonical logic (read-only)
`app/services/model_validation.py` (folds, `district_threshold`, `evaluate_predictions`,
`classification_metrics`), `app/services/modelling_dataset.py` (`HORIZONS`, targets),
`app/services/model_features.py` (training-only imputer), `app/services/case_data_cleaning.py`.

WAPE is an error measure — never reported as "accuracy" and never `100 − WAPE`.
