"""Shared national-pooled rolling-origin validation for tree regressors."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterator

from flask import has_app_context

from app.extensions import db
from app.models import PipelineRun
from app.services.model_features import build_feature_spec
from app.services.model_validation import (
    PredictionRecord,
    ValidationFold,
    ValidationSummary,
    build_rolling_origin_folds,
    classification_metrics,
    district_threshold,
    evaluate_predictions,
)
from app.services.modelling_dataset import DATASET_PATH, HORIZONS, read_modelling_dataset

RANDOM_STATE = 24024
CHECKPOINT_VERSION = 2

POOLED_CATEGORICAL_FEATURES = ("district_id",)
POOLED_NUMERIC_FEATURES = (
    "epi_week",
    "continuous_week_index",
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_4",
    "cases_lag_8",
    "cases_lag_12",
    "cases_rolling_mean_4",
    "cases_rolling_mean_8",
    "cases_rolling_mean_12",
    "cases_lag_1_interpolated",
    "cases_lag_2_interpolated",
    "cases_lag_4_interpolated",
    "cases_lag_8_interpolated",
    "cases_lag_12_interpolated",
    "cases_rolling_interpolated_count_4",
    "cases_rolling_interpolated_count_8",
    "cases_rolling_interpolated_count_12",
    "rainfall_lag_4",
    "rainfall_lag_8",
    "rainfall_lag_12",
    "temperature_lag_4",
    "temperature_lag_8",
    "temperature_lag_12",
    "humidity_lag_4",
    "humidity_lag_8",
    "humidity_lag_12",
)
POOLED_FEATURES = POOLED_CATEGORICAL_FEATURES + POOLED_NUMERIC_FEATURES

PREDICTION_COLUMNS = (
    "model_name",
    "district_id",
    "district_name",
    "horizon",
    "validation_fold",
    "forecast_origin_week",
    "target_week",
    "observed_cases",
    "predicted_cases",
    "district_surge_threshold",
    "observed_surge_status",
    "predicted_surge_score",
    "predicted_surge_classification",
    "surge_probability",
    "surge_probability_cutoff",
    "count_threshold_surge_classification",
    "feature_availability_status",
    "missing_predictor_count",
    "model_status",
    "selected_parameters",
    "forecast_clipped_nonnegative",
    "feature_contributions",
)

DISTRICT_METRIC_COLUMNS = (
    "model_name",
    "district_id",
    "district_name",
    "horizon",
    "status",
    "completed_folds",
    "failed_folds",
    "excluded_missing_target",
    "n",
    "mae",
    "rmse",
    "mape",
    "wape",
    "r_squared",
    "surge_n",
    "tp",
    "tn",
    "fp",
    "fn",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "f2",
    "accuracy",
    "auc",
    "pr_auc",
    "brier_score",
)

FEATURE_IMPORTANCE_COLUMNS = (
    "model_name",
    "horizon",
    "validation_fold",
    "importance_method",
    "feature_name",
    "importance",
)


class PooledTreeValidationError(RuntimeError):
    """Raised when pooled tree validation cannot proceed safely."""


@dataclass(frozen=True)
class PooledTreeConfig:
    model_key: str
    model_name: str
    dependency: str
    native_importance_method: str
    parameter_candidates: tuple[dict[str, Any], ...]
    classifier_parameter_candidates: tuple[dict[str, Any], ...] = ()
    controlled_threads: int = 2


@dataclass
class PooledTreeValidationSummary(ValidationSummary):
    model_key: str = ""
    pipeline_run_id: int | None = None
    job_counts: dict[str, int] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    published: bool = False
    checkpoints_reused: int = 0
    runtime_seconds: float = 0.0
    selected_parameters_by_horizon: dict[int, dict[str, Any]] = field(default_factory=dict)
    selected_classifier_parameters_by_horizon: dict[int, dict[str, Any]] = field(
        default_factory=dict
    )
    selected_probability_cutoff_by_horizon: dict[int, float] = field(default_factory=dict)
    count_threshold_surge_metrics_by_horizon: dict[int, dict[str, Any]] = field(
        default_factory=dict
    )
    classifier_surge_metrics_by_horizon: dict[int, dict[str, Any]] = field(
        default_factory=dict
    )
    operational_surge_method_by_horizon: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "model_key": self.model_key,
                "pipeline_run_id": self.pipeline_run_id,
                "job_counts": self.job_counts,
                "artifact_paths": self.artifact_paths,
                "published": self.published,
                "checkpoints_reused": self.checkpoints_reused,
                "runtime_seconds": round(self.runtime_seconds, 3),
                "selected_parameters_by_horizon": self.selected_parameters_by_horizon,
                "selected_classifier_parameters_by_horizon": (
                    self.selected_classifier_parameters_by_horizon
                ),
                "selected_probability_cutoff_by_horizon": (
                    self.selected_probability_cutoff_by_horizon
                ),
                "count_threshold_surge_metrics_by_horizon": (
                    self.count_threshold_surge_metrics_by_horizon
                ),
                "classifier_surge_metrics_by_horizon": (
                    self.classifier_surge_metrics_by_horizon
                ),
                "operational_surge_method_by_horizon": (
                    self.operational_surge_method_by_horizon
                ),
                "pooled_tree_specification": {
                    "scope": (
                        "one national pooled count regressor and one class-balanced "
                        "surge classifier per horizon and fold"
                    ),
                    "categorical_features": list(POOLED_CATEGORICAL_FEATURES),
                    "numeric_features": list(POOLED_NUMERIC_FEATURES),
                    "district_encoding": (
                        "one-hot encoding fitted on each training fold with unknown "
                        "categories ignored"
                    ),
                    "numeric_missing_values": (
                        "median imputation with missing indicators fitted on training rows only"
                    ),
                    "missing_outcomes": "excluded; never imputed or zero-filled",
                    "hyperparameter_selection": (
                        "small grid selected on an inner chronological holdout from the "
                        "earliest outer training fold for each horizon; outer validation "
                        "outcomes are never used for tuning"
                    ),
                    "surge_classifier": (
                        "probability model fitted on fold-specific seasonal district "
                        "surge labels; cutoff chosen on inner chronological data"
                    ),
                    "count_threshold_benchmark": (
                        "predicted_cases > fold-specific district seasonal threshold"
                    ),
                    "negative_prediction_rule": "clip count predictions below zero to zero",
                    "random_state": RANDOM_STATE,
                    "no_random_shuffle": True,
                },
            }
        )
        return payload


def run_pooled_tree_validation(
    *,
    config: PooledTreeConfig,
    horizons: tuple[int, ...] = HORIZONS,
    dry_run: bool = False,
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path,
    predictions_path: Path,
    district_metrics_path: Path,
    diagnostics_path: Path,
    model_card_path: Path,
    feature_importance_path: Path,
    checkpoint_dir: Path,
    max_folds: int = 12,
    replace: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    record_pipeline_run: bool = True,
) -> PooledTreeValidationSummary:
    """Run resumable national-pooled validation for Random Forest or XGBoost."""
    started = time.perf_counter()
    _validate_request(config, horizons, max_folds)
    rows = read_modelling_dataset(dataset_path)
    if not rows:
        raise PooledTreeValidationError(f"Modelling dataset is empty: {dataset_path}.")
    feature_spec = build_feature_spec(list(rows[0].keys()))
    _validate_feature_contract(rows[0].keys())
    districts = _districts(rows)
    folds_by_horizon = {
        horizon: build_rolling_origin_folds(rows, horizon=horizon, max_folds=max_folds)
        for horizon in horizons
    }
    job_total = sum(len(folds) for folds in folds_by_horizon.values())
    summary = PooledTreeValidationSummary(
        model_name=config.model_name,
        model_key=config.model_key,
        model_scope="national_pooled",
        horizons=horizons,
        dry_run=dry_run,
        status="dry-run" if dry_run else "running",
        feature_columns=list(POOLED_FEATURES),
        forbidden_columns_excluded=feature_spec.forbidden_columns_excluded,
        row_counts={
            "modelling_rows": len(rows),
            "districts": len(districts),
            "jobs": job_total,
        },
        artifact_paths={
            "predictions_csv": str(predictions_path),
            "district_horizon_metrics_csv": str(district_metrics_path),
            "overall_metrics_json": str(artifact_path),
            "fit_diagnostics_json": str(diagnostics_path),
            "model_card_json": str(model_card_path),
            "feature_importance_csv": str(feature_importance_path),
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    for horizon, folds in folds_by_horizon.items():
        summary.validation_cutoffs.extend(_fold_proofs(folds))
        summary.horizon_gap_proof.extend(_gap_proofs(folds))
        if not folds:
            summary.warnings.append(f"No validation folds were available for horizon {horizon}.")

    if importlib.util.find_spec(config.dependency) is None:
        summary.status = "blocked_missing_dependency"
        summary.warnings.append(
            f"{config.dependency} is required for {config.model_name}. Install modelling "
            "dependencies with: pip install -r requirements-models.txt"
        )
        summary.runtime_seconds = time.perf_counter() - started
        return summary
    if dry_run:
        summary.job_counts = {"planned": job_total, "total": job_total}
        summary.runtime_seconds = time.perf_counter() - started
        return summary

    dataset_signature = _dataset_signature(dataset_path)
    pipeline_run = _start_pipeline_run(
        config=config,
        summary=summary,
        dataset_path=dataset_path,
        max_folds=max_folds,
        replace=replace,
        enabled=record_pipeline_run,
    )
    jobs: list[dict[str, Any]] = []
    tuning: dict[int, dict[str, Any]] = {}
    completed = 0

    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for horizon in horizons:
            folds = folds_by_horizon[horizon]
            count_tuning = _select_horizon_parameters(config, folds)
            classifier_tuning = _select_horizon_classifier(config, folds)
            tuning[horizon] = {
                "count_regression": count_tuning,
                "surge_classification": classifier_tuning,
            }
            selected_parameters = count_tuning["selected_parameters"]
            selected_classifier_parameters = classifier_tuning["selected_parameters"]
            probability_cutoff = float(classifier_tuning["probability_cutoff"])
            summary.selected_parameters_by_horizon[horizon] = selected_parameters
            summary.selected_classifier_parameters_by_horizon[horizon] = (
                selected_classifier_parameters
            )
            summary.selected_probability_cutoff_by_horizon[horizon] = probability_cutoff
            for fold_number, fold in enumerate(folds, start=1):
                checkpoint_path = checkpoint_dir / f"h{horizon}_fold{fold_number}.json"
                if replace:
                    checkpoint_path.unlink(missing_ok=True)
                job = _read_checkpoint(
                    checkpoint_path,
                    config=config,
                    dataset_signature=dataset_signature,
                    horizon=horizon,
                    fold_number=fold_number,
                    fold=fold,
                    max_folds=max_folds,
                    selected_parameters=selected_parameters,
                    selected_classifier_parameters=selected_classifier_parameters,
                    probability_cutoff=probability_cutoff,
                )
                reused = job is not None
                if job is None:
                    job = _run_fold_job(
                        config=config,
                        fold=fold,
                        fold_number=fold_number,
                        selected_parameters=selected_parameters,
                        selected_classifier_parameters=selected_classifier_parameters,
                        probability_cutoff=probability_cutoff,
                    )
                    _write_checkpoint(
                        checkpoint_path,
                        job=job,
                        config=config,
                        dataset_signature=dataset_signature,
                        max_folds=max_folds,
                        selected_parameters=selected_parameters,
                        selected_classifier_parameters=selected_classifier_parameters,
                        probability_cutoff=probability_cutoff,
                    )
                else:
                    summary.checkpoints_reused += 1
                jobs.append(job)
                completed += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "model": config.model_name,
                            "horizon": horizon,
                            "validation_fold": fold_number,
                            "completed": completed,
                            "total": job_total,
                            "status": job["status"],
                            "checkpoint_reused": reused,
                        }
                    )

        predictions = [row for job in jobs for row in job.get("predictions", [])]
        importances = [row for job in jobs for row in job.get("feature_importance", [])]
        importances.extend(_aggregate_feature_importance(importances, config.model_name))
        _populate_summary_metrics(summary, jobs, predictions)
        district_metrics = _district_metric_rows(
            config=config,
            rows=rows,
            districts=districts,
            folds_by_horizon=folds_by_horizon,
            jobs=jobs,
            predictions=predictions,
        )
        summary.row_counts["training_missing_target_exclusions"] = sum(
            int(job.get("training_missing_target_exclusions", 0)) for job in jobs
        )
        summary.row_counts["validation_missing_target_exclusions"] = sum(
            int(row["excluded_missing_target"]) for row in district_metrics
        )
        diagnostics = _diagnostics_payload(
            config=config,
            summary=summary,
            jobs=jobs,
            tuning=tuning,
            dataset_signature=dataset_signature,
        )
        model_card = _model_card_payload(config, summary, importances)
        summary.runtime_seconds = time.perf_counter() - started
        summary.published = True

        if summary.status != "success" and _is_valid_success_artifact(artifact_path):
            summary.published = False
            summary.warnings.append(
                f"A complete prior {config.model_name} publication was preserved because "
                "this run was not fully successful. New job details remain in checkpoints."
            )
        else:
            staged = _stage_outputs(
                summary=summary,
                predictions=predictions,
                district_metrics=district_metrics,
                diagnostics=diagnostics,
                model_card=model_card,
                feature_importance=importances,
                targets={
                    "summary": artifact_path,
                    "predictions": predictions_path,
                    "district_metrics": district_metrics_path,
                    "diagnostics": diagnostics_path,
                    "model_card": model_card_path,
                    "feature_importance": feature_importance_path,
                },
            )
            with publish_staged_outputs(staged):
                _finish_pipeline_run(pipeline_run, summary)
            pipeline_run = None
        if pipeline_run is not None:
            _finish_pipeline_run(pipeline_run, summary)
            pipeline_run = None
        return summary
    except BaseException as exc:
        _fail_pipeline_run(pipeline_run, exc, summary)
        raise


def _select_horizon_parameters(
    config: PooledTreeConfig, folds: list[ValidationFold]
) -> dict[str, Any]:
    default = dict(config.parameter_candidates[0])
    if not folds:
        return {
            "status": "default_no_outer_fold",
            "selected_parameters": default,
            "candidate_results": [],
        }
    rows = folds[0].trainable_rows
    target_column = f"target_cases_h{folds[0].horizon}"
    target_date_column = f"target_week_start_date_h{folds[0].horizon}"
    unique_dates = sorted(
        {str(row[target_date_column]) for row in rows if row.get(target_date_column)}
    )
    holdout_weeks = min(8, max(2, len(unique_dates) // 5))
    if len(unique_dates) <= holdout_weeks + 8:
        return {
            "status": "default_insufficient_inner_history",
            "selected_parameters": default,
            "candidate_results": [],
            "available_target_weeks": len(unique_dates),
        }
    holdout_dates = set(unique_dates[-holdout_weeks:])
    inner_train = [row for row in rows if str(row[target_date_column]) not in holdout_dates]
    inner_validation = [row for row in rows if str(row[target_date_column]) in holdout_dates]
    if not inner_train or not inner_validation:
        return {
            "status": "default_empty_inner_split",
            "selected_parameters": default,
            "candidate_results": [],
        }

    train_x = _feature_frame(inner_train)
    train_y = _target_array(inner_train, target_column)
    validation_x = _feature_frame(inner_validation)
    validation_y = _target_array(inner_validation, target_column)
    results: list[dict[str, Any]] = []
    for parameters in config.parameter_candidates:
        try:
            pipeline = _build_pipeline(config, parameters)
            pipeline.fit(train_x, train_y)
            predicted = pipeline.predict(validation_x)
            mae = float(sum(abs(float(a) - float(p)) for a, p in zip(validation_y, predicted))) / len(validation_y)
            results.append(
                {
                    "parameters": dict(parameters),
                    "inner_validation_mae": mae,
                    "status": "success",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "parameters": dict(parameters),
                    "inner_validation_mae": None,
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    successful = [item for item in results if item["status"] == "success"]
    selected = min(successful, key=lambda item: item["inner_validation_mae"]) if successful else None
    return {
        "status": "success" if selected else "default_all_candidates_failed",
        "selected_parameters": dict(selected["parameters"] if selected else default),
        "candidate_results": results,
        "inner_training_rows": len(inner_train),
        "inner_validation_rows": len(inner_validation),
        "inner_validation_target_weeks": sorted(holdout_dates),
        "outer_validation_used_for_tuning": False,
    }


def _select_horizon_classifier(
    config: PooledTreeConfig, folds: list[ValidationFold]
) -> dict[str, Any]:
    candidates = config.classifier_parameter_candidates or config.parameter_candidates
    default = dict(candidates[0])
    fallback = {
        "status": "default_insufficient_inner_history",
        "selected_parameters": default,
        "probability_cutoff": 0.5,
        "candidate_results": [],
            "selection_objective": (
                "Require sensitivity >=0.60 and specificity >=0.40 where possible; "
                "then maximize F2, sensitivity, PR-AUC, precision and specificity"
            ),
        "outer_validation_used_for_tuning": False,
    }
    if not folds:
        return {**fallback, "status": "default_no_outer_fold"}
    fold = folds[0]
    rows = fold.trainable_rows
    target_date_column = f"target_week_start_date_h{fold.horizon}"
    unique_dates = sorted(
        {str(row[target_date_column]) for row in rows if row.get(target_date_column)}
    )
    holdout_weeks = min(8, max(2, len(unique_dates) // 5))
    if len(unique_dates) <= holdout_weeks + 8:
        return {**fallback, "available_target_weeks": len(unique_dates)}
    holdout_dates = set(unique_dates[-holdout_weeks:])
    inner_train = [row for row in rows if str(row[target_date_column]) not in holdout_dates]
    inner_validation = [row for row in rows if str(row[target_date_column]) in holdout_dates]
    if not inner_train or not inner_validation:
        return {**fallback, "status": "default_empty_inner_split"}

    train_x = _feature_frame(inner_train)
    validation_x = _feature_frame(inner_validation)
    train_y = _surge_target_array(inner_train, inner_train, fold.horizon)
    validation_y = _surge_target_array(inner_train, inner_validation, fold.horizon)
    if len(set(bool(value) for value in train_y)) < 2:
        return {
            **fallback,
            "status": "default_single_class_inner_training",
            "inner_training_rows": len(inner_train),
            "inner_validation_rows": len(inner_validation),
        }
    positive_weight = _positive_class_weight(train_y)
    results: list[dict[str, Any]] = []
    for parameters in candidates:
        try:
            pipeline = _build_classifier_pipeline(
                config, parameters, positive_weight=positive_weight
            )
            pipeline.fit(train_x, train_y)
            probabilities = _positive_probabilities(pipeline, validation_x)
            for cutoff in (0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
                metrics = classification_metrics(
                    [bool(value) for value in validation_y],
                    [float(value) >= cutoff for value in probabilities],
                    [Decimal(str(value)) for value in probabilities],
                )
                results.append(
                    {
                        "parameters": dict(parameters),
                        "probability_cutoff": cutoff,
                        "metrics": metrics,
                        "status": "success",
                    }
                )
        except Exception as exc:
            results.append(
                {
                    "parameters": dict(parameters),
                    "probability_cutoff": None,
                    "metrics": {},
                    "status": "failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    successful = [item for item in results if item["status"] == "success"]
    balanced = [
        item
        for item in successful
        if float((item.get("metrics") or {}).get("sensitivity") or 0) >= 0.60
        and float((item.get("metrics") or {}).get("specificity") or 0) >= 0.40
    ]
    candidate_pool = balanced or successful
    selected = max(
        candidate_pool,
        key=_classifier_selection_key if balanced else _fallback_classifier_selection_key,
    ) if candidate_pool else None
    return {
        "status": "success" if selected else "default_all_candidates_failed",
        "selected_parameters": dict(selected["parameters"] if selected else default),
        "probability_cutoff": float(selected["probability_cutoff"] if selected else 0.5),
        "candidate_results": results,
        "inner_training_rows": len(inner_train),
        "inner_validation_rows": len(inner_validation),
        "inner_validation_target_weeks": sorted(holdout_dates),
        "positive_class_weight": positive_weight,
        "selection_objective": (
            "Require sensitivity >=0.60 and specificity >=0.40 where possible; "
            "then maximize F2, sensitivity, PR-AUC, precision and specificity"
        ),
        "outer_validation_used_for_tuning": False,
    }


def _classifier_selection_key(item: dict[str, Any]) -> tuple[float, ...]:
    metrics = item.get("metrics") or {}
    return tuple(
        float(metrics.get(name) or -1)
        for name in ("f2", "sensitivity", "pr_auc", "precision", "specificity")
    )


def _fallback_classifier_selection_key(item: dict[str, Any]) -> tuple[float, ...]:
    metrics = item.get("metrics") or {}
    sensitivity = float(metrics.get("sensitivity") or 0)
    specificity = float(metrics.get("specificity") or 0)
    return (
        min(sensitivity, specificity),
        float(metrics.get("f2") or -1),
        float(metrics.get("pr_auc") or -1),
        float(metrics.get("precision") or -1),
    )


def _run_fold_job(
    *,
    config: PooledTreeConfig,
    fold: ValidationFold,
    fold_number: int,
    selected_parameters: dict[str, Any],
    selected_classifier_parameters: dict[str, Any],
    probability_cutoff: float,
) -> dict[str, Any]:
    target_column = f"target_cases_h{fold.horizon}"
    target_date_column = f"target_week_start_date_h{fold.horizon}"
    horizon_gap_exclusions = sum(
        _is_missing(row.get(target_date_column))
        or date.fromisoformat(str(row[target_date_column])) > fold.origin_date
        for row in fold.train_rows
    )
    missing_target_exclusions = sum(
        not _is_missing(row.get(target_date_column))
        and date.fromisoformat(str(row[target_date_column])) <= fold.origin_date
        and _is_missing(row.get(target_column))
        for row in fold.train_rows
    )
    job: dict[str, Any] = {
        "model_name": config.model_name,
        "horizon": fold.horizon,
        "validation_fold": fold_number,
        "origin_date": fold.origin_date.isoformat(),
        "target_date": fold.target_date.isoformat(),
        "status": "failed",
        "selected_parameters": dict(selected_parameters),
        "selected_classifier_parameters": dict(selected_classifier_parameters),
        "surge_probability_cutoff": probability_cutoff,
        "training_rows": len(fold.trainable_rows),
        "validation_rows": len(fold.validation_rows),
        "horizon_gap_exclusions": horizon_gap_exclusions,
        "training_missing_target_exclusions": missing_target_exclusions,
        "predictions": [],
        "feature_importance": [],
        "warnings": [],
        "failure_reason": None,
        "negative_predictions_clipped": 0,
    }
    if not fold.trainable_rows or not fold.validation_rows:
        job["failure_reason"] = "Training or validation rows are empty."
        return job
    try:
        train_x = _feature_frame(fold.trainable_rows)
        train_y = _target_array(fold.trainable_rows, target_column)
        train_surge_y = _surge_target_array(
            fold.train_rows, fold.trainable_rows, fold.horizon
        )
        validation_x = _feature_frame(fold.validation_rows)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline = _build_pipeline(config, selected_parameters)
            pipeline.fit(train_x, train_y)
            raw_predictions = pipeline.predict(validation_x)
            classifier, constant_probability = _fit_surge_classifier(
                config=config,
                parameters=selected_classifier_parameters,
                train_x=train_x,
                train_y=train_surge_y,
            )
            surge_probabilities = (
                _positive_probabilities(classifier, validation_x)
                if classifier is not None
                else [constant_probability] * len(fold.validation_rows)
            )
        job["warnings"] = sorted({str(item.message) for item in caught})
        transformed_names = [str(name) for name in pipeline.named_steps["preprocess"].get_feature_names_out()]
        job["transformed_feature_names"] = transformed_names
        job["preprocessing_audit"] = {
            "fit_rows": len(fold.trainable_rows),
            "validation_rows_seen_during_fit": 0,
            "numeric_imputation": "training-fold median with missing indicators",
            "district_encoding": "training-fold one-hot; handle_unknown=ignore",
            "outcome_imputation": False,
            "surge_class_weight": _positive_class_weight(train_surge_y),
            "surge_classifier_constant_probability": constant_probability,
        }
        native_rows = _native_importance_rows(
            config=config,
            pipeline=pipeline,
            transformed_names=transformed_names,
            horizon=fold.horizon,
            fold_number=fold_number,
        )
        permutation_rows = _permutation_importance_rows(
            config=config,
            pipeline=pipeline,
            validation_x=validation_x,
            validation_y=_target_array(fold.validation_rows, target_column),
            horizon=fold.horizon,
            fold_number=fold_number,
        )
        job["feature_importance"] = native_rows + permutation_rows
        contributions = _prediction_contributions(config, pipeline, validation_x, transformed_names)
        for index, (row, raw_prediction, surge_probability) in enumerate(
            zip(fold.validation_rows, raw_predictions, surge_probabilities)
        ):
            raw_value = float(raw_prediction)
            clipped = raw_value < 0
            predicted_value = max(0.0, raw_value)
            if clipped:
                job["negative_predictions_clipped"] += 1
            actual = Decimal(str(row[target_column]))
            threshold = district_threshold(
                fold.train_rows,
                str(row["district_id"]),
                target_epi_week=int(row[f"target_epi_week_h{fold.horizon}"]),
            )
            predicted = Decimal(str(predicted_value)).quantize(Decimal("0.000001"))
            surge_score = (predicted - threshold).quantize(Decimal("0.000001"))
            missing_count = sum(_is_missing(row.get(feature)) for feature in POOLED_FEATURES)
            job["predictions"].append(
                {
                    "model_name": config.model_name,
                    "district_id": str(row["district_id"]),
                    "district_name": str(row.get("district_name") or row["district_id"]),
                    "horizon": fold.horizon,
                    "validation_fold": fold_number,
                    "forecast_origin_week": fold.origin_date.isoformat(),
                    "target_week": fold.target_date.isoformat(),
                    "observed_cases": str(actual),
                    "predicted_cases": str(predicted),
                    "district_surge_threshold": str(threshold),
                    "observed_surge_status": actual > threshold,
                    "predicted_surge_score": str(surge_score),
                    "predicted_surge_classification": (
                        float(surge_probability) >= probability_cutoff
                    ),
                    "surge_probability": str(
                        Decimal(str(surge_probability)).quantize(Decimal("0.000001"))
                    ),
                    "surge_probability_cutoff": probability_cutoff,
                    "count_threshold_surge_classification": surge_score > 0,
                    "feature_availability_status": "complete" if missing_count == 0 else "imputed_predictors",
                    "missing_predictor_count": missing_count,
                    "model_status": "success",
                    "selected_parameters": json.dumps(selected_parameters, sort_keys=True),
                    "forecast_clipped_nonnegative": clipped,
                    "feature_contributions": json.dumps(contributions[index], sort_keys=True) if contributions else "",
                }
            )
        job["status"] = "success"
        return job
    except Exception as exc:
        job["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return job


def _build_pipeline(config: PooledTreeConfig, parameters: dict[str, Any]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    district_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(POOLED_NUMERIC_FEATURES)),
            ("district", district_pipeline, list(POOLED_CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
    if config.model_key == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        estimator = RandomForestRegressor(
            **parameters,
            random_state=RANDOM_STATE,
            n_jobs=config.controlled_threads,
        )
    elif config.model_key == "xgboost":
        from xgboost import XGBRegressor

        estimator = XGBRegressor(
            **parameters,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=config.controlled_threads,
            tree_method="hist",
            verbosity=0,
        )
    else:
        raise PooledTreeValidationError(f"Unsupported pooled model {config.model_key!r}.")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _build_classifier_pipeline(
    config: PooledTreeConfig,
    parameters: dict[str, Any],
    *,
    positive_weight: float,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
                ),
                list(POOLED_NUMERIC_FEATURES),
            ),
            (
                "district",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                list(POOLED_CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )
    if config.model_key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(
            **parameters,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=config.controlled_threads,
        )
    elif config.model_key == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            **parameters,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=positive_weight,
            random_state=RANDOM_STATE,
            n_jobs=config.controlled_threads,
            tree_method="hist",
            verbosity=0,
        )
    else:
        raise PooledTreeValidationError(f"Unsupported pooled model {config.model_key!r}.")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _fit_surge_classifier(*, config, parameters, train_x, train_y):
    classes = sorted({bool(value) for value in train_y})
    if len(classes) < 2:
        return None, 1.0 if classes and classes[0] else 0.0
    pipeline = _build_classifier_pipeline(
        config,
        parameters,
        positive_weight=_positive_class_weight(train_y),
    )
    pipeline.fit(train_x, train_y)
    return pipeline, None


def _positive_probabilities(pipeline, frame):
    probabilities = pipeline.predict_proba(frame)
    classes = list(pipeline.named_steps["model"].classes_)
    positive_index = classes.index(1) if 1 in classes else classes.index(True)
    return probabilities[:, positive_index]


def _positive_class_weight(values) -> float:
    positives = sum(bool(value) for value in values)
    negatives = len(values) - positives
    return float(negatives / positives) if positives else 1.0


def _surge_target_array(
    reference_rows: list[dict[str, Any]],
    labelled_rows: list[dict[str, Any]],
    horizon: int,
):
    import numpy as np

    output: list[int] = []
    target_column = f"target_cases_h{horizon}"
    target_week_column = f"target_epi_week_h{horizon}"
    threshold_cache: dict[tuple[str, int], Decimal] = {}
    for row in labelled_rows:
        district_id = str(row["district_id"])
        target_week = int(row[target_week_column])
        key = (district_id, target_week)
        if key not in threshold_cache:
            threshold_cache[key] = district_threshold(
                reference_rows,
                district_id,
                target_epi_week=target_week,
            )
        actual = _decimal(row.get(target_column))
        if actual is None:
            raise PooledTreeValidationError(
                f"Missing outcome reached surge-label fitting for {target_column}."
            )
        output.append(int(actual > threshold_cache[key]))
    return np.asarray(output, dtype=int)


def _feature_frame(rows: list[dict[str, Any]]):
    import pandas as pd

    frame = pd.DataFrame(index=range(len(rows)))
    for column in POOLED_NUMERIC_FEATURES:
        frame[column] = pd.to_numeric([row.get(column) for row in rows], errors="coerce")
    for column in POOLED_CATEGORICAL_FEATURES:
        frame[column] = [None if _is_missing(row.get(column)) else str(row.get(column)) for row in rows]
    return frame[list(POOLED_FEATURES)]


def _target_array(rows: list[dict[str, Any]], target_column: str):
    import numpy as np

    values: list[float] = []
    for row in rows:
        value = _decimal(row.get(target_column))
        if value is None:
            raise PooledTreeValidationError(
                f"Missing outcome reached model fitting for {target_column}; outcomes are not imputed."
            )
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _native_importance_rows(
    *,
    config: PooledTreeConfig,
    pipeline,
    transformed_names: list[str],
    horizon: int,
    fold_number: int,
) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    if config.model_key == "random_forest":
        values = [float(value) for value in model.feature_importances_]
    else:
        scores = model.get_booster().get_score(importance_type="gain")
        values = [float(scores.get(f"f{index}", 0.0)) for index in range(len(transformed_names))]
    return [
        {
            "model_name": config.model_name,
            "horizon": horizon,
            "validation_fold": fold_number,
            "importance_method": config.native_importance_method,
            "feature_name": name,
            "importance": value,
        }
        for name, value in zip(transformed_names, values)
    ]


def _permutation_importance_rows(
    *,
    config: PooledTreeConfig,
    pipeline,
    validation_x,
    validation_y,
    horizon: int,
    fold_number: int,
) -> list[dict[str, Any]]:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipeline,
        validation_x,
        validation_y,
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return [
        {
            "model_name": config.model_name,
            "horizon": horizon,
            "validation_fold": fold_number,
            "importance_method": "permutation_validation_mae",
            "feature_name": str(name),
            "importance": float(value),
        }
        for name, value in zip(validation_x.columns, result.importances_mean)
    ]


def _prediction_contributions(config, pipeline, validation_x, transformed_names):
    if config.model_key != "xgboost":
        return []
    import xgboost as xgb

    transformed = pipeline.named_steps["preprocess"].transform(validation_x)
    values = pipeline.named_steps["model"].get_booster().predict(
        xgb.DMatrix(transformed), pred_contribs=True
    )
    output: list[dict[str, float]] = []
    for row in values:
        pairs = [(name, float(value)) for name, value in zip(transformed_names, row[:-1])]
        top = sorted(pairs, key=lambda item: abs(item[1]), reverse=True)[:10]
        contribution = {name: value for name, value in top}
        contribution["BIAS"] = float(row[-1])
        output.append(contribution)
    return output


def _aggregate_feature_importance(
    rows: list[dict[str, Any]], model_name: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["horizon"]), str(row["importance_method"]), str(row["feature_name"]))].append(float(row["importance"]))
    return [
        {
            "model_name": model_name,
            "horizon": horizon,
            "validation_fold": "overall",
            "importance_method": method,
            "feature_name": feature,
            "importance": sum(values) / len(values),
        }
        for (horizon, method, feature), values in sorted(grouped.items())
    ]


def _populate_summary_metrics(
    summary: PooledTreeValidationSummary,
    jobs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    counts = Counter(str(job.get("status", "failed")) for job in jobs)
    counts["total"] = len(jobs)
    summary.job_counts = dict(counts)
    summary.row_counts["prediction_rows"] = len(predictions)
    summary.row_counts["failed_jobs"] = counts.get("failed", 0)
    for horizon in summary.horizons:
        rows = [row for row in predictions if int(row["horizon"]) == horizon]
        records = [_prediction_record(row) for row in rows]
        count_metrics, classifier_metrics = evaluate_predictions(records)
        count_threshold_metrics = classification_metrics(
            [_bool(row["observed_surge_status"]) for row in rows],
            [_bool(row["count_threshold_surge_classification"]) for row in rows],
            [Decimal(str(row["predicted_surge_score"])) for row in rows],
        )
        operational_method, operational_metrics = _select_operational_surge_method(
            classifier_metrics=classifier_metrics,
            count_threshold_metrics=count_threshold_metrics,
        )
        summary.count_metrics_by_horizon[horizon] = count_metrics
        summary.classifier_surge_metrics_by_horizon[horizon] = classifier_metrics
        summary.count_threshold_surge_metrics_by_horizon[horizon] = count_threshold_metrics
        summary.operational_surge_method_by_horizon[horizon] = operational_method
        summary.surge_metrics_by_horizon[horizon] = operational_metrics
        if records:
            first = records[0]
            summary.threshold_proof.append(
                {
                    "horizon": horizon,
                    "district_id": first.district_id,
                    "origin_date": first.origin_date.isoformat(),
                    "target_date": first.target_date.isoformat(),
                    "threshold_training_only": str(first.threshold),
                    "surge_score": str(first.surge_score),
                }
            )
    all_horizons_have_predictions = all(
        summary.count_metrics_by_horizon[horizon]["n"] > 0 for horizon in summary.horizons
    )
    if jobs and counts.get("failed", 0) == 0 and all_horizons_have_predictions:
        summary.status = "success"
    elif predictions:
        summary.status = "partial_success"
        summary.warnings.append("One or more pooled horizon-fold jobs failed; successful jobs were retained.")
    else:
        summary.status = "failed"
        summary.warnings.append("No prediction rows were produced by pooled validation.")


def _select_operational_surge_method(
    *,
    classifier_metrics: dict[str, Any],
    count_threshold_metrics: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    candidates = (
        ("dedicated_classifier", classifier_metrics),
        ("count_threshold", count_threshold_metrics),
    )
    eligible = [
        item
        for item in candidates
        if _metric_float(item[1], "sensitivity") >= 0.60
        and _metric_float(item[1], "specificity") >= 0.40
    ]
    pool = eligible or list(candidates)
    selected = max(
        pool,
        key=lambda item: (
            _metric_float(item[1], "f2", default=-1),
            _metric_float(item[1], "sensitivity", default=-1),
            _metric_float(item[1], "pr_auc", default=-1),
            _metric_float(item[1], "precision", default=-1),
            _metric_float(item[1], "specificity", default=-1),
        ),
    )
    return selected[0], selected[1]


def _metric_float(
    metrics: dict[str, Any], name: str, *, default: float = 0.0
) -> float:
    value = metrics.get(name)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _prediction_record(row: dict[str, Any]) -> PredictionRecord:
    return PredictionRecord(
        district_id=str(row["district_id"]),
        horizon=int(row["horizon"]),
        origin_date=date.fromisoformat(str(row["forecast_origin_week"])),
        target_date=date.fromisoformat(str(row["target_week"])),
        actual=Decimal(str(row["observed_cases"])),
        predicted=Decimal(str(row["predicted_cases"])),
        threshold=Decimal(str(row["district_surge_threshold"])),
        actual_surge=_bool(row["observed_surge_status"]),
        predicted_surge=_bool(row["predicted_surge_classification"]),
        surge_score=Decimal(str(row["surge_probability"])),
    )


def _district_metric_rows(
    *,
    config: PooledTreeConfig,
    rows: list[dict[str, Any]],
    districts: list[str],
    folds_by_horizon: dict[int, list[ValidationFold]],
    jobs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = {
        str(row["district_id"]): str(row.get("district_name") or row["district_id"])
        for row in rows
    }
    source = {(str(row["district_id"]), str(row["week_start_date"])): row for row in rows}
    output: list[dict[str, Any]] = []
    for horizon in config_horizons_from_folds(folds_by_horizon):
        horizon_jobs = [job for job in jobs if int(job["horizon"]) == horizon]
        failed_folds = sum(job["status"] != "success" for job in horizon_jobs)
        for district_id in districts:
            district_predictions = [
                row
                for row in predictions
                if int(row["horizon"]) == horizon and row["district_id"] == district_id
            ]
            records = [_prediction_record(row) for row in district_predictions]
            count_metrics, surge_metrics = evaluate_predictions(records)
            excluded = 0
            for fold in folds_by_horizon[horizon]:
                row = source.get((district_id, fold.origin_date.isoformat()))
                if row is None or _is_missing(row.get(f"target_cases_h{horizon}")):
                    excluded += 1
            status = (
                "success"
                if records and failed_folds == 0 and excluded == 0
                else "partial_success"
                if records
                else "failed"
            )
            output.append(
                {
                    "model_name": config.model_name,
                    "district_id": district_id,
                    "district_name": names.get(district_id, district_id),
                    "horizon": horizon,
                    "status": status,
                    "completed_folds": len({row["validation_fold"] for row in district_predictions}),
                    "failed_folds": failed_folds,
                    "excluded_missing_target": excluded,
                    **count_metrics,
                    **{f"surge_{key}" if key == "n" else key: value for key, value in surge_metrics.items()},
                }
            )
    return output


def config_horizons_from_folds(folds_by_horizon: dict[int, list[ValidationFold]]) -> list[int]:
    return list(folds_by_horizon)


def _diagnostics_payload(
    *,
    config: PooledTreeConfig,
    summary: PooledTreeValidationSummary,
    jobs: list[dict[str, Any]],
    tuning: dict[int, dict[str, Any]],
    dataset_signature: dict[str, Any],
) -> dict[str, Any]:
    job_details = []
    for job in jobs:
        job_details.append(
            {
                key: value
                for key, value in job.items()
                if key not in {"predictions", "feature_importance"}
            }
        )
    return {
        "model_name": config.model_name,
        "status": summary.status,
        "dataset_signature": dataset_signature,
        "hyperparameter_tuning": tuning,
        "jobs": job_details,
        "job_counts": summary.job_counts,
        "training_only_preprocessing": True,
        "outer_validation_used_for_tuning": False,
        "random_state": RANDOM_STATE,
        "controlled_threads": config.controlled_threads,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _model_card_payload(
    config: PooledTreeConfig,
    summary: PooledTreeValidationSummary,
    importances: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate_native = [
        row
        for row in importances
        if row["validation_fold"] == "overall"
        and row["importance_method"] == config.native_importance_method
    ]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in aggregate_native:
        grouped[str(row["feature_name"])].append(float(row["importance"]))
    top_predictors = [
        {"feature_name": name, "mean_native_importance": sum(values) / len(values)}
        for name, values in sorted(
            grouped.items(), key=lambda item: sum(item[1]) / len(item[1]), reverse=True
        )[:20]
    ]
    explainability = [
        "Permutation importance uses only each outer validation fold and reports change in MAE.",
        (
            "Impurity importance is model-native but can favor high-cardinality or highly "
            "splittable predictors and does not establish causality."
            if config.model_key == "random_forest"
            else "Native XGBoost gain importance measures split gain and does not establish causality."
        ),
    ]
    if config.model_key == "xgboost":
        explainability.append(
            "Per-prediction contributions use XGBoost pred_contribs on the transformed feature matrix."
        )
    return {
        "model_name": config.model_name,
        "model_scope": "national_pooled",
        "primary_task": "future malaria case-count regression and surge classification",
        "horizons_weeks": list(summary.horizons),
        "predictors_used": list(POOLED_FEATURES),
        "transformed_feature_names": (
            "numeric feature and missing-indicator names plus training-fold district one-hot names; "
            "the complete names are retained in fit diagnostics and feature-importance output"
        ),
        "district_encoding": "one-hot fitted on training rows only; unseen districts are ignored safely",
        "validation": (
            "The shared rolling-origin folds use origin-week predictors, targets exactly h weeks "
            "ahead, and labels observed by each training origin only. No random splitting."
        ),
        "preprocessing": (
            "Numeric median imputation and missing indicators are fitted only on training rows. "
            "District one-hot encoding is fitted only on training rows. Outcomes are never imputed."
        ),
        "hyperparameter_selection": (
            "A small grid is selected independently by horizon using an inner chronological "
            "holdout from the earliest outer training fold, then recorded on every outer fold."
        ),
        "selected_parameters_by_horizon": summary.selected_parameters_by_horizon,
        "selected_classifier_parameters_by_horizon": (
            summary.selected_classifier_parameters_by_horizon
        ),
        "selected_probability_cutoff_by_horizon": (
            summary.selected_probability_cutoff_by_horizon
        ),
        "surge_threshold": (
            "For each prediction, the district 75th percentile uses training-only valid "
            "observations in the target epidemiological week +/-2 weeks. Confirmed epidemic "
            "year 2022 remains in count training but is excluded from the normal-reference "
            "threshold. Sparse seasonal history falls back to district expanding history."
        ),
        "surge_classifier": (
            "A separate class-balanced classifier is fitted per horizon. Parameters and the "
            "probability cutoff are selected on an inner chronological holdout. Candidate "
            "operating points must reach sensitivity >=0.60 and specificity >=0.40 where "
            "possible, then are ranked by F2, sensitivity, PR-AUC, precision and specificity. "
            "Temporal SMOTE is not used."
        ),
        "classifier_surge_metrics": summary.classifier_surge_metrics_by_horizon,
        "count_threshold_benchmark_metrics": (
            summary.count_threshold_surge_metrics_by_horizon
        ),
        "operational_surge_method_by_horizon": (
            summary.operational_surge_method_by_horizon
        ),
        "operational_surge_selection": (
            "Compare the dedicated classifier with the count-versus-seasonal-threshold "
            "method. Prefer methods with sensitivity >=0.60 and specificity >=0.40, "
            "then maximize F2 with sensitivity, PR-AUC, precision and specificity as "
            "tie-breakers. Both methods remain reported."
        ),
        "negative_prediction_rule": "Predicted counts below zero are clipped to zero and flagged.",
        "explainability": explainability,
        "top_overall_predictors": top_predictors,
        "limitations": [
            "Associations and feature importance are not causal effects.",
            "Twelve evenly spaced rolling-origin folds improve temporal coverage but do not represent every week.",
            "Missing predictors are imputed, which can reduce local sensitivity.",
            "A national pooled model can underfit district-specific dynamics despite district encoding.",
        ],
        "status": summary.status,
    }


def _stage_outputs(
    *,
    summary: PooledTreeValidationSummary,
    predictions: list[dict[str, Any]],
    district_metrics: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    model_card: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    targets: dict[str, Path],
) -> dict[Path, Path]:
    staged: dict[Path, Path] = {}
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".staged", dir=target.parent, delete=False
        ) as handle:
            staged[Path(handle.name)] = target
    by_target = {target: staged_path for staged_path, target in staged.items()}
    try:
        _write_json(by_target[targets["summary"]], summary.to_dict())
        _write_csv(by_target[targets["predictions"]], predictions, PREDICTION_COLUMNS)
        _write_csv(by_target[targets["district_metrics"]], district_metrics, DISTRICT_METRIC_COLUMNS)
        _write_json(by_target[targets["diagnostics"]], diagnostics)
        _write_json(by_target[targets["model_card"]], model_card)
        _write_csv(
            by_target[targets["feature_importance"]],
            feature_importance,
            FEATURE_IMPORTANCE_COLUMNS,
        )
    except BaseException:
        for staged_path in staged:
            staged_path.unlink(missing_ok=True)
        raise
    return staged


@contextmanager
def publish_staged_outputs(staged: dict[Path, Path]) -> Iterator[None]:
    """Publish a set of staged files as one rollback-capable operation."""
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for staged_path, target in staged.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with tempfile.NamedTemporaryFile(
                    prefix=f".{target.name}.", suffix=".backup", dir=target.parent, delete=False
                ) as handle:
                    backup = Path(handle.name)
                backup.unlink()
                target.replace(backup)
                backups[target] = backup
            staged_path.replace(target)
            published.append(target)
        yield
    except BaseException:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                backup.replace(target)
        raise
    else:
        for backup in backups.values():
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass
    finally:
        for staged_path in staged:
            staged_path.unlink(missing_ok=True)


def _write_checkpoint(
    path: Path,
    *,
    job: dict[str, Any],
    config: PooledTreeConfig,
    dataset_signature: dict[str, Any],
    max_folds: int,
    selected_parameters: dict[str, Any],
    selected_classifier_parameters: dict[str, Any],
    probability_cutoff: float,
) -> None:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_key": config.model_key,
        "dataset_signature": dataset_signature,
        "max_folds": max_folds,
        "features": list(POOLED_FEATURES),
        "selected_parameters": selected_parameters,
        "selected_classifier_parameters": selected_classifier_parameters,
        "probability_cutoff": probability_cutoff,
        "job": job,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _read_checkpoint(
    path: Path,
    *,
    config: PooledTreeConfig,
    dataset_signature: dict[str, Any],
    horizon: int,
    fold_number: int,
    fold: ValidationFold,
    max_folds: int,
    selected_parameters: dict[str, Any],
    selected_classifier_parameters: dict[str, Any],
    probability_cutoff: float,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    job = payload.get("job")
    if (
        payload.get("checkpoint_version") != CHECKPOINT_VERSION
        or payload.get("model_key") != config.model_key
        or payload.get("dataset_signature") != dataset_signature
        or payload.get("max_folds") != max_folds
        or payload.get("features") != list(POOLED_FEATURES)
        or payload.get("selected_parameters") != selected_parameters
        or payload.get("selected_classifier_parameters")
        != selected_classifier_parameters
        or not isinstance(job, dict)
        or job.get("status") != "success"
        or int(job.get("horizon", -1)) != horizon
        or int(job.get("validation_fold", -1)) != fold_number
        or job.get("origin_date") != fold.origin_date.isoformat()
        or job.get("target_date") != fold.target_date.isoformat()
    ):
        return None
    if float(payload.get("probability_cutoff", -1)) != probability_cutoff:
        for prediction in job.get("predictions") or []:
            probability = _decimal(prediction.get("surge_probability"))
            if probability is None:
                return None
            prediction["predicted_surge_classification"] = (
                probability >= Decimal(str(probability_cutoff))
            )
            prediction["surge_probability_cutoff"] = probability_cutoff
        job["surge_probability_cutoff"] = probability_cutoff
    return job


def _start_pipeline_run(
    *,
    config: PooledTreeConfig,
    summary: PooledTreeValidationSummary,
    dataset_path: Path,
    max_folds: int,
    replace: bool,
    enabled: bool,
) -> PipelineRun | None:
    if not enabled or not has_app_context():
        return None
    run = PipelineRun(
        run_type="model_training",
        status="running",
        started_at=datetime.now(timezone.utc),
        source_name=f"{config.model_name} national-pooled validation",
        parameters={
            "model_name": config.model_name,
            "model_scope": "national_pooled",
            "dataset_path": str(dataset_path),
            "horizons": list(summary.horizons),
            "max_folds": max_folds,
            "replace": replace,
            "features": list(POOLED_FEATURES),
            "random_state": RANDOM_STATE,
            "controlled_threads": config.controlled_threads,
        },
    )
    db.session.add(run)
    db.session.commit()
    summary.pipeline_run_id = run.id
    return run


def _finish_pipeline_run(
    run: PipelineRun | None, summary: PooledTreeValidationSummary
) -> None:
    if run is None:
        return
    run.status = "success" if summary.status == "success" else "failed"
    run.completed_at = datetime.now(timezone.utc)
    run.row_count_raw = summary.row_counts.get("modelling_rows", 0)
    run.row_count_processed = summary.row_counts.get("prediction_rows", 0)
    run.warning_count = len(summary.warnings)
    run.error_message = None if summary.status == "success" else f"Validation status: {summary.status}."
    run.manifest = {
        "validation_status": summary.status,
        "published": summary.published,
        "job_counts": summary.job_counts,
        "artifact_paths": summary.artifact_paths,
        "count_metrics_by_horizon": summary.count_metrics_by_horizon,
        "surge_metrics_by_horizon": summary.surge_metrics_by_horizon,
        "classifier_surge_metrics_by_horizon": (
            summary.classifier_surge_metrics_by_horizon
        ),
        "count_threshold_surge_metrics_by_horizon": (
            summary.count_threshold_surge_metrics_by_horizon
        ),
        "operational_surge_method_by_horizon": (
            summary.operational_surge_method_by_horizon
        ),
        "selected_parameters_by_horizon": summary.selected_parameters_by_horizon,
        "selected_classifier_parameters_by_horizon": (
            summary.selected_classifier_parameters_by_horizon
        ),
        "selected_probability_cutoff_by_horizon": (
            summary.selected_probability_cutoff_by_horizon
        ),
        "warnings": summary.warnings,
    }
    db.session.commit()


def _fail_pipeline_run(
    run: PipelineRun | None,
    exc: BaseException,
    summary: PooledTreeValidationSummary,
) -> None:
    if run is None:
        return
    db.session.rollback()
    stored = db.session.get(PipelineRun, run.id)
    if stored is None:
        return
    stored.status = "failed"
    stored.completed_at = datetime.now(timezone.utc)
    stored.row_count_raw = summary.row_counts.get("modelling_rows", 0)
    stored.row_count_processed = summary.row_counts.get("prediction_rows", 0)
    stored.warning_count = len(summary.warnings)
    stored.error_message = f"{type(exc).__name__}: {str(exc)[:900]}"
    stored.manifest = {
        "validation_status": "failed",
        "published": False,
        "job_counts": summary.job_counts,
        "artifact_paths": summary.artifact_paths,
    }
    db.session.commit()


def _validate_request(
    config: PooledTreeConfig, horizons: tuple[int, ...], max_folds: int
) -> None:
    if config.model_key not in {"random_forest", "xgboost"}:
        raise PooledTreeValidationError(f"Unsupported model key {config.model_key!r}.")
    if not horizons:
        raise PooledTreeValidationError("At least one horizon is required.")
    unsupported = sorted(set(horizons) - set(HORIZONS))
    if unsupported:
        raise PooledTreeValidationError(
            "Unsupported horizon(s): " + ", ".join(map(str, unsupported))
        )
    if max_folds <= 0:
        raise PooledTreeValidationError("max_folds must be greater than zero.")
    if not config.parameter_candidates:
        raise PooledTreeValidationError("At least one hyperparameter candidate is required.")
    if not config.classifier_parameter_candidates:
        raise PooledTreeValidationError(
            "At least one surge-classifier hyperparameter candidate is required."
        )


def _validate_feature_contract(columns: Any) -> None:
    available = set(columns)
    missing = [feature for feature in POOLED_FEATURES if feature not in available]
    if missing:
        raise PooledTreeValidationError(
            "Completed modelling dataset is missing required pooled predictors: "
            + ", ".join(missing)
        )
    if any(feature.startswith("target_") for feature in POOLED_FEATURES):
        raise PooledTreeValidationError("Target fields are forbidden in pooled predictors.")
    if any("wind" in feature.lower() for feature in POOLED_FEATURES):
        raise PooledTreeValidationError("Wind fields are forbidden in pooled predictors.")
    climate = [feature for feature in POOLED_FEATURES if feature.startswith(("rainfall_", "temperature_", "humidity_"))]
    if any("_lag_" not in feature for feature in climate):
        raise PooledTreeValidationError("Only lagged climate fields are permitted in pooled predictors.")


def _districts(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["district_id"]) for row in rows})


def _fold_proofs(folds: list[ValidationFold]) -> list[dict[str, Any]]:
    return [
        {
            "horizon": fold.horizon,
            "validation_fold": index,
            "origin_week": fold.origin_date.isoformat(),
            "target_week": fold.target_date.isoformat(),
            "train_rows": len(fold.train_rows),
            "trainable_rows": len(fold.trainable_rows),
            "validation_rows": len(fold.validation_rows),
        }
        for index, fold in enumerate(folds, start=1)
    ]


def _gap_proofs(folds: list[ValidationFold]) -> list[dict[str, Any]]:
    return [
        {
            "horizon": fold.horizon,
            "validation_fold": index,
            "origin": fold.origin_date.isoformat(),
            "target": fold.target_date.isoformat(),
            "gap_days": (fold.target_date - fold.origin_date).days,
            "gap_weeks": (fold.target_date - fold.origin_date).days // 7,
        }
        for index, fold in enumerate(folds, start=1)
    ]


def _dataset_signature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _is_valid_success_artifact(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "success" or not payload.get("published"):
        return False
    count_metrics = payload.get("count_metrics_by_horizon") or {}
    surge_metrics = payload.get("surge_metrics_by_horizon") or {}
    for horizon in payload.get("horizons") or []:
        count = count_metrics.get(str(horizon), count_metrics.get(horizon))
        surge = surge_metrics.get(str(horizon), surge_metrics.get(horizon))
        if not isinstance(count, dict) or int(count.get("n", 0)) <= 0:
            return False
        if not isinstance(surge, dict) or int(surge.get("n", 0)) <= 0:
            return False
    artifacts = payload.get("artifact_paths") or {}
    required = (
        "predictions_csv",
        "district_horizon_metrics_csv",
        "fit_diagnostics_json",
        "model_card_json",
        "feature_importance_csv",
    )
    return all(artifacts.get(name) and Path(artifacts[name]).exists() for name in required)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _decimal(value: Any) -> Decimal | None:
    if _is_missing(value):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}
