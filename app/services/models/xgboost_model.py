"""National-pooled XGBoost rolling-origin validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.services.modelling_dataset import DATASET_PATH, HORIZONS
from app.services.models.pooled_tree_validation import (
    PooledTreeConfig,
    PooledTreeValidationSummary,
    run_pooled_tree_validation,
)

XGBOOST_ARTIFACT = Path("data/quality/xgboost_validation_summary.json")
XGBOOST_PREDICTIONS = Path("data/modelling/xgboost_validation_predictions.csv")
XGBOOST_DISTRICT_METRICS = Path("data/modelling/xgboost_district_horizon_metrics.csv")
XGBOOST_DIAGNOSTICS = Path("data/quality/xgboost_fit_diagnostics.json")
XGBOOST_MODEL_CARD = Path("data/models/xgboost_model_card.json")
XGBOOST_FEATURE_IMPORTANCE = Path("data/models/xgboost_feature_importance.csv")
XGBOOST_CHECKPOINT_DIR = Path("data/models/xgboost_checkpoints")

XGBOOST_CONFIG = PooledTreeConfig(
    model_key="xgboost",
    model_name="XGBoost",
    dependency="xgboost",
    native_importance_method="gain",
    parameter_candidates=(
        {
            "n_estimators": 96,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 160,
            "max_depth": 4,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 2.0,
        },
        {
            "n_estimators": 220,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 3.0,
            "min_child_weight": 3,
        },
    ),
    classifier_parameter_candidates=(
        {
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 2.0,
        },
        {
            "n_estimators": 180,
            "max_depth": 4,
            "learning_rate": 0.04,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 3.0,
        },
        {
            "n_estimators": 240,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "reg_lambda": 4.0,
            "min_child_weight": 3,
        },
    ),
    controlled_threads=2,
)


def run_xgboost_validation(
    *,
    horizons: tuple[int, ...] = HORIZONS,
    dry_run: bool = False,
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path = XGBOOST_ARTIFACT,
    predictions_path: Path = XGBOOST_PREDICTIONS,
    district_metrics_path: Path = XGBOOST_DISTRICT_METRICS,
    diagnostics_path: Path = XGBOOST_DIAGNOSTICS,
    model_card_path: Path = XGBOOST_MODEL_CARD,
    feature_importance_path: Path = XGBOOST_FEATURE_IMPORTANCE,
    checkpoint_dir: Path = XGBOOST_CHECKPOINT_DIR,
    max_folds: int = 12,
    replace: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    record_pipeline_run: bool = True,
) -> PooledTreeValidationSummary:
    return run_pooled_tree_validation(
        config=XGBOOST_CONFIG,
        horizons=horizons,
        dry_run=dry_run,
        dataset_path=dataset_path,
        artifact_path=artifact_path,
        predictions_path=predictions_path,
        district_metrics_path=district_metrics_path,
        diagnostics_path=diagnostics_path,
        model_card_path=model_card_path,
        feature_importance_path=feature_importance_path,
        checkpoint_dir=checkpoint_dir,
        max_folds=max_folds,
        replace=replace,
        progress_callback=progress_callback,
        record_pipeline_run=record_pipeline_run,
    )


def run_regional_xgboost_experiment(**kwargs: Any):
    """Run the DHIS2-subregion pooling experiment for XGBoost.

    Thin wrapper over the shared regional orchestration engine; experimental
    only and never touches production artefacts.
    """
    from app.services.models.regional_tree_validation import run_regional_pooling_experiment

    return run_regional_pooling_experiment(XGBOOST_CONFIG, **kwargs)
