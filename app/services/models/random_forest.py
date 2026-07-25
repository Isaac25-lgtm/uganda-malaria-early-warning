"""National-pooled Random Forest rolling-origin validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from app.services.modelling_dataset import DATASET_PATH, HORIZONS
from app.services.models.pooled_tree_validation import (
    PooledTreeConfig,
    PooledTreeValidationSummary,
    run_pooled_tree_validation,
)

RANDOM_FOREST_ARTIFACT = Path("data/quality/random_forest_validation_summary.json")
RANDOM_FOREST_PREDICTIONS = Path("data/modelling/random_forest_validation_predictions.csv")
RANDOM_FOREST_DISTRICT_METRICS = Path(
    "data/modelling/random_forest_district_horizon_metrics.csv"
)
RANDOM_FOREST_DIAGNOSTICS = Path("data/quality/random_forest_fit_diagnostics.json")
RANDOM_FOREST_MODEL_CARD = Path("data/models/random_forest_model_card.json")
RANDOM_FOREST_FEATURE_IMPORTANCE = Path("data/models/random_forest_feature_importance.csv")
RANDOM_FOREST_CHECKPOINT_DIR = Path("data/models/random_forest_checkpoints")

RANDOM_FOREST_CONFIG = PooledTreeConfig(
    model_key="random_forest",
    model_name="Random Forest",
    dependency="sklearn",
    native_importance_method="impurity",
    parameter_candidates=(
        {
            "n_estimators": 48,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 64,
            "max_depth": 16,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 96,
            "max_depth": 12,
            "min_samples_leaf": 4,
            "max_features": 0.7,
        },
    ),
    classifier_parameter_candidates=(
        {
            "n_estimators": 96,
            "max_depth": 10,
            "min_samples_leaf": 3,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 128,
            "max_depth": 16,
            "min_samples_leaf": 4,
            "max_features": 0.7,
        },
        {
            "n_estimators": 160,
            "max_depth": None,
            "min_samples_leaf": 6,
            "max_features": "sqrt",
        },
    ),
    controlled_threads=2,
)


def run_random_forest_validation(
    *,
    horizons: tuple[int, ...] = HORIZONS,
    dry_run: bool = False,
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path = RANDOM_FOREST_ARTIFACT,
    predictions_path: Path = RANDOM_FOREST_PREDICTIONS,
    district_metrics_path: Path = RANDOM_FOREST_DISTRICT_METRICS,
    diagnostics_path: Path = RANDOM_FOREST_DIAGNOSTICS,
    model_card_path: Path = RANDOM_FOREST_MODEL_CARD,
    feature_importance_path: Path = RANDOM_FOREST_FEATURE_IMPORTANCE,
    checkpoint_dir: Path = RANDOM_FOREST_CHECKPOINT_DIR,
    max_folds: int = 12,
    replace: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    record_pipeline_run: bool = True,
) -> PooledTreeValidationSummary:
    return run_pooled_tree_validation(
        config=RANDOM_FOREST_CONFIG,
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


def run_regional_random_forest_experiment(**kwargs: Any):
    """Run the DHIS2-subregion pooling experiment for Random Forest.

    Thin wrapper over the shared regional orchestration engine; experimental
    only and never touches production artefacts.
    """
    from app.services.models.regional_tree_validation import run_regional_pooling_experiment

    return run_regional_pooling_experiment(RANDOM_FOREST_CONFIG, **kwargs)
