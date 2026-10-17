"""Frozen configuration and registries for the literature-based benchmark.

ISOLATED SUPERVISOR-REQUESTED COMPARISON — reads canonical artifacts read-only,
writes only under runs/<RUN_ID>/. Registers nothing in the Flask application.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- run identity + locations ----------------------------------------------
# The corrected, lineage-audited run (20260828r1) is the default; the superseded
# preliminary run (20260828) is preserved read-only. Override via LB_RUN_ID.
RUN_ID = os.environ.get("LB_RUN_ID", "20260828r1")
# config/__init__.py -> config -> literature_based_forecasting_benchmark -> research_comparisons -> REPO_ROOT
REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = REPO_ROOT / "research_comparisons" / "literature_based_forecasting_benchmark"
RUNS_DIR = BENCHMARK_ROOT / "runs"
RUN_DIR = RUNS_DIR / RUN_ID  # ALL outputs for this run live under here

# --- frozen evaluation frame (reuse canonical logic; do not redefine) -------
HORIZONS = (4, 8, 12)
MAX_FOLDS = 12
MIN_TRAIN_WEEKS = 52

# --- uncertainty ------------------------------------------------------------
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260828
BOOTSTRAP_CI = 0.95

# --- bounded parallelism ----------------------------------------------------
ARIMAX_MAX_WORKERS = 2

# --- canonical inputs (READ-ONLY) -------------------------------------------
CANONICAL_DATASET = REPO_ROOT / "data/modelling/modelling_dataset_2020_2025.csv"
CANONICAL_DATASET_SHA256 = (
    "252fc3ccffbf324e85ede290dc2cb9a3a0cbf8dbaf0ee90d768c4e5dd52f0a06"
)
FROZEN_PREDICTION_ARTIFACTS = {
    "random_forest": "data/modelling/random_forest_validation_predictions.csv",
    "xgboost": "data/modelling/xgboost_validation_predictions.csv",
    "sarimax": "data/modelling/sarimax_validation_predictions.csv",
}
FROZEN_METRIC_ARTIFACTS = {
    "random_forest": "data/modelling/random_forest_district_horizon_metrics.csv",
    "xgboost": "data/modelling/xgboost_district_horizon_metrics.csv",
    "sarimax": "data/modelling/sarimax_district_horizon_metrics.csv",
    "baseline_summary": "data/quality/baseline_validation_summary.json",
}
CANONICAL_SUMMARIES = [
    "data/quality/modelling_dataset_summary.json",
    "data/quality/malaria_case_cleaning_summary.json",
]
CANONICAL_MODULES = [
    "app/services/model_validation.py",
    "app/services/modelling_dataset.py",
    "app/services/model_features.py",
    "app/services/case_data_cleaning.py",
    "app/services/models/random_forest.py",
    "app/services/models/xgboost_model.py",
    "app/services/models/sarimax.py",
    "app/services/models/baseline.py",
    "app/services/models/pooled_tree_validation.py",
]
PRODUCTION_FORECAST_DIR = "data/models/production"

# Frozen RF WAPE from the canonical validation. WAPE is an error measure and is
# NEVER reported as "accuracy" and NEVER converted via 100 - WAPE.
FROZEN_RF_WAPE = {
    4: "34.31125070533906567595533234",
    8: "38.56155871652542",
    12: "approximately 39.64",
}

# --- paired-comparison key (matches the frozen prediction CSV schema) -------
PAIRED_KEY_COLUMNS = ("district_id", "horizon", "forecast_origin_week", "target_week")

# --- output isolation guard -------------------------------------------------
# The experiment must write ONLY under RUN_DIR. Any path resolving under one of
# these roots (and not under RUN_DIR) is rejected by leakage/output guards.
FORBIDDEN_OUTPUT_ROOTS = (
    "app",
    "data/models",
    "data/modelling",
    "data/quality",
    "data/reference",
    "data/climate",
    "data/boundaries",
    "data/experiments",  # explicitly forbidden: outputs live under runs/, not here
    "docs",
    "tests",
)

# ===========================================================================
# PREDECLARED literature method classifications (spec Section E, revised).
# DO NOT change after seeing results.
#
# PRIMARY (winner table): only the two with adequate source strength.
# NON-PRIMARY (kept OUT of the winner table): Katwesige (LOW source strength;
#   authoritative full thesis not accessible -> Negative Binomial *sensitivity
#   analysis*, not a primary challenger) and Davis (exploratory).
# ===========================================================================
PRIMARY_METHODS = {
    "zinszer_arimax": {
        "adaptation_class": "Zinszer-inspired adapted district-level ARIMAX (dynamic regression)",
        "source": "Zinszer et al., Uganda (facility-level multivariate ARIMA with environmental + clinical predictors)",
        "scope": "district-specific (one model per district x horizon x fold)",
        "distribution": "adapted ARIMAX / dynamic-regression (Gaussian) on district count series",
        "notes": "OUR implementation is an adapted ARIMAX/dynamic-regression; the paper's exact "
                 "orders/predictor-selection are not established in the accessible source, so no claim "
                 "is made that the paper used exactly this formulation. Vegetation and clinical inputs "
                 "unavailable; district (not facility) unit.",
    },
    "tekle_poisson_dl": {
        "adaptation_class": "Teklehaimanot-inspired adapted Poisson autoregressive distributed-lag",
        "source": "Teklehaimanot et al., Ethiopia, 2004 (Malaria Journal, Paper I & II; Paper II full text verified)",
        "scope": "national pooled with district fixed effects (one model per horizon x fold)",
        "distribution": "Poisson, log link (report overdispersion; do NOT switch to NB)",
        "notes": "Origin-observable direct-horizon adaptation (original target-relative weather lags "
                 "leak at h8/h12). ERA5-Land mean temperature substitutes original min/max. No offset.",
    },
}
# Non-primary: reported separately, EXCLUDED from the primary winner table.
SENSITIVITY_METHODS = {
    "katwesige_nb": {
        "adaptation_class": "Katwesige-inspired adapted Negative Binomial sensitivity analysis",
        "source": "Katwesige, Justine Fay, 2021 — unpublished Makerere University MSc dissertation (Gulu, DHIS2 + climate)",
        "scope": "national pooled (sensitivity only)",
        "distribution": "Negative Binomial, log link",
        "status": "NON-PRIMARY: authoritative full thesis not accessible; exact NB predictors, lag "
                  "structure, model form and validation NOT verifiable -> reported as a sensitivity "
                  "analysis, kept OUT of the primary winner table. MLP excluded; no accuracy claims reused.",
    },
}
EXPLORATORY_METHODS = {
    "davis_exploratory_nb_dl": {
        "adaptation_class": "Davis-inspired reduced distributed-lag NB sensitivity analysis",
        "source": "Davis et al., 2019 (Environmental Modelling & Software; GA driver-selection)",
        "scope": "national pooled (exploratory only)",
        "distribution": "Negative Binomial distributed lag (reduced)",
        "status": "NOT a Davis replication (omits GA feature selection, spatial clustering, remote "
                  "sensing, long lags). Runs after primaries pass; kept OUT of the primary winner table.",
    },
}
NON_PRIMARY_METHODS = {**SENSITIVITY_METHODS, **EXPLORATORY_METHODS}
EXCLUDED_METHODS = {
    "ippolito_zambia": {
        "source": "Ippolito et al., Zambia (Malaria Journal 2025, single Choma sub-district catchment)",
        "reason": "The published prediction target is seasonal malaria burden/totals (an outbreak "
                  "framing in one low-transmission catchment) rather than weekly district-level direct "
                  "4-, 8- and 12-week-ahead counts; adapting it would create a materially different "
                  "method. (Full text was behind a paywall; exclusion rests on the study's stated "
                  "single-catchment, low-transmission, outbreak-target design.)",
    },
    "epidemia_2017": {
        "source": "Merkord et al., 2017 (EPIDEMIA system)",
        "reason": "Conceptual system/integration and decision-support reference, not one reproducibly "
                  "specified forecasting model. Not reproduced as a model; not described as SARIMAX.",
    },
}

BENCHMARK_LABEL = "literature-based benchmark"  # Katwesige unpublished
PRIMARY_METHOD_IDS = tuple(PRIMARY_METHODS)  # ("zinszer_arimax", "tekle_poisson_dl")
