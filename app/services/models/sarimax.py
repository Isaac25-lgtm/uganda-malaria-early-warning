"""District-specific SARIMAX rolling-origin validation."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator

from flask import has_app_context

from app.extensions import db
from app.models import PipelineRun
from app.services.model_features import TrainingOnlyImputer, build_feature_spec
from app.services.model_validation import (
    PredictionRecord,
    ValidationFold,
    ValidationSummary,
    build_rolling_origin_folds,
    district_threshold,
    evaluate_predictions,
)
from app.services.modelling_dataset import DATASET_PATH, HORIZONS, read_modelling_dataset

SARIMAX_ARTIFACT = Path("data/quality/sarimax_validation_summary.json")
SARIMAX_PREDICTIONS = Path("data/modelling/sarimax_validation_predictions.csv")
SARIMAX_DISTRICT_METRICS = Path("data/modelling/sarimax_district_horizon_metrics.csv")
SARIMAX_DIAGNOSTICS = Path("data/quality/sarimax_fit_diagnostics.json")
SARIMAX_MODEL_CARD = Path("data/models/sarimax_model_card.json")
SARIMAX_CHECKPOINT_DIR = Path("data/models/sarimax_checkpoints")

SARIMAX_EXOGENOUS_FEATURES = (
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
SARIMAX_ORDER_CANDIDATES = (
    ((1, 0, 0), (1, 0, 0, 52)),
    ((1, 0, 0), (0, 0, 0, 0)),
)
NONSEASONAL_FALLBACK = ((1, 0, 0), (0, 0, 0, 0))
REGRESSION_FALLBACK = ((0, 0, 0), (0, 0, 0, 0))
MIN_OBSERVED_HISTORY = 52
MIN_SEASONAL_HISTORY = 104
CHECKPOINT_VERSION = 1

PREDICTION_COLUMNS = (
    "district_id",
    "district_name",
    "horizon",
    "validation_fold",
    "forecast_origin_week",
    "target_week",
    "observed_cases",
    "predicted_cases",
    "observed_surge_status",
    "predicted_surge_score",
    "predicted_surge_classification",
    "surge_threshold",
    "selected_sarimax_order",
    "selected_seasonal_order",
    "exogenous_features_used",
    "convergence_status",
    "fallback_status",
    "forecast_clipped_nonnegative",
    "warning_or_failure_reason",
)

DISTRICT_METRIC_COLUMNS = (
    "district_id",
    "district_name",
    "horizon",
    "status",
    "completed_folds",
    "failed_folds",
    "excluded_missing_target_or_origin",
    "missing_history_rows",
    "selected_sarimax_order",
    "selected_seasonal_order",
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
    "accuracy",
    "auc",
)


class SARIMAXValidationError(RuntimeError):
    """Raised when real SARIMAX validation cannot proceed safely."""


@dataclass
class SARIMAXValidationSummary(ValidationSummary):
    pipeline_run_id: int | None = None
    job_counts: dict[str, int] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    published: bool = False
    checkpoints_reused: int = 0
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "pipeline_run_id": self.pipeline_run_id,
                "job_counts": self.job_counts,
                "artifact_paths": self.artifact_paths,
                "published": self.published,
                "checkpoints_reused": self.checkpoints_reused,
                "runtime_seconds": round(self.runtime_seconds, 3),
                "sarimax_specification": {
                    "order_candidates": [
                        {"order": list(order), "seasonal_order": list(seasonal)}
                        for order, seasonal in SARIMAX_ORDER_CANDIDATES
                    ],
                    "seasonal_period_weeks": 52,
                    "minimum_observed_history": MIN_OBSERVED_HISTORY,
                    "order_selection": "lowest converged AIC on earliest training fold only",
                    "future_exogenous_policy": (
                        "origin-week lagged climate values held constant through the "
                        "forecast horizon; no future climate rows are read"
                    ),
                },
            }
        )
        return payload


def run_sarimax_validation(
    *,
    horizons: tuple[int, ...] = HORIZONS,
    dry_run: bool = False,
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path = SARIMAX_ARTIFACT,
    predictions_path: Path = SARIMAX_PREDICTIONS,
    district_metrics_path: Path = SARIMAX_DISTRICT_METRICS,
    diagnostics_path: Path = SARIMAX_DIAGNOSTICS,
    model_card_path: Path = SARIMAX_MODEL_CARD,
    checkpoint_dir: Path = SARIMAX_CHECKPOINT_DIR,
    district_ids: tuple[str, ...] | None = None,
    max_folds: int = 12,
    replace: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    record_pipeline_run: bool = True,
) -> SARIMAXValidationSummary:
    """Run resumable district-horizon SARIMAX rolling-origin validation."""
    started = time.perf_counter()
    _validate_horizons(horizons)
    if max_folds <= 0:
        raise SARIMAXValidationError("max_folds must be greater than zero.")
    rows = read_modelling_dataset(dataset_path)
    if not rows:
        raise SARIMAXValidationError(f"Modelling dataset is empty: {dataset_path}.")
    feature_spec = build_feature_spec(list(rows[0].keys()))
    _validate_exogenous_contract(rows[0].keys())
    selected_districts = _selected_districts(rows, district_ids)
    folds_by_horizon = {
        horizon: build_rolling_origin_folds(
            rows, horizon=horizon, max_folds=max_folds
        )
        for horizon in horizons
    }
    summary = SARIMAXValidationSummary(
        model_name="SARIMAX",
        model_scope="district_specific",
        horizons=horizons,
        dry_run=dry_run,
        status="dry-run" if dry_run else "running",
        feature_columns=list(SARIMAX_EXOGENOUS_FEATURES),
        forbidden_columns_excluded=feature_spec.forbidden_columns_excluded,
        row_counts={
            "modelling_rows": len(rows),
            "districts": len(selected_districts),
            "jobs": len(selected_districts) * len(horizons),
        },
        artifact_paths={
            "predictions_csv": str(predictions_path),
            "district_horizon_metrics_csv": str(district_metrics_path),
            "overall_metrics_json": str(artifact_path),
            "fit_diagnostics_json": str(diagnostics_path),
            "model_card_json": str(model_card_path),
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    for horizon, folds in folds_by_horizon.items():
        summary.validation_cutoffs.extend(_fold_proofs(folds))
        summary.horizon_gap_proof.extend(_gap_proofs(folds))
        if not folds:
            summary.warnings.append(f"No validation folds were available for horizon {horizon}.")

    if importlib.util.find_spec("statsmodels") is None:
        summary.status = "blocked_missing_dependency"
        summary.warnings.append(
            "statsmodels is required for SARIMAX. Install modelling dependencies "
            "with: pip install -r requirements-models.txt"
        )
        summary.runtime_seconds = time.perf_counter() - started
        return summary
    if dry_run:
        summary.job_counts = {"planned": summary.row_counts["jobs"]}
        summary.runtime_seconds = time.perf_counter() - started
        return summary

    pipeline_run = _start_pipeline_run(
        summary=summary,
        dataset_path=dataset_path,
        selected_districts=selected_districts,
        max_folds=max_folds,
        replace=replace,
        enabled=record_pipeline_run,
    )
    dataset_signature = _dataset_signature(dataset_path)
    jobs: list[dict[str, Any]] = []
    total_jobs = len(selected_districts) * len(horizons)
    completed_jobs = 0

    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for horizon in horizons:
            folds = folds_by_horizon[horizon]
            for district_id in selected_districts:
                checkpoint_path = checkpoint_dir / f"{district_id}_h{horizon}.json"
                if replace:
                    checkpoint_path.unlink(missing_ok=True)
                job = _read_checkpoint(
                    checkpoint_path,
                    dataset_signature=dataset_signature,
                    horizon=horizon,
                    max_folds=max_folds,
                )
                reused = job is not None
                if job is None:
                    job = _run_district_horizon_job(
                        rows=rows,
                        folds=folds,
                        district_id=district_id,
                        horizon=horizon,
                    )
                    _write_checkpoint(
                        checkpoint_path,
                        job=job,
                        dataset_signature=dataset_signature,
                        max_folds=max_folds,
                    )
                else:
                    summary.checkpoints_reused += 1
                jobs.append(job)
                completed_jobs += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "completed": completed_jobs,
                            "total": total_jobs,
                            "district_id": district_id,
                            "horizon": horizon,
                            "status": job["status"],
                            "checkpoint_reused": reused,
                        }
                    )

        predictions = [prediction for job in jobs for prediction in job["predictions"]]
        _populate_summary_metrics(summary, jobs, predictions)
        summary.runtime_seconds = time.perf_counter() - started
        district_metrics = [_district_metric_row(job) for job in jobs]
        diagnostics = _diagnostics_payload(
            summary=summary,
            jobs=jobs,
            dataset_signature=dataset_signature,
        )
        model_card = _model_card_payload(summary)
        summary.published = True

        if summary.status == "failed" and _is_valid_success_artifact(artifact_path):
            summary.published = False
            summary.warnings.append(
                "A complete prior SARIMAX publication was preserved because this "
                "run was not fully successful. New job details remain in checkpoints."
            )
        else:
            staged = _stage_outputs(
                summary=summary,
                predictions=predictions,
                district_metrics=district_metrics,
                diagnostics=diagnostics,
                model_card=model_card,
                targets={
                    "summary": artifact_path,
                    "predictions": predictions_path,
                    "district_metrics": district_metrics_path,
                    "diagnostics": diagnostics_path,
                    "model_card": model_card_path,
                },
            )
            with _publish_staged_outputs(staged):
                _finish_pipeline_run(pipeline_run, summary)
            pipeline_run = None
        if pipeline_run is not None:
            _finish_pipeline_run(pipeline_run, summary)
            pipeline_run = None
        return summary
    except BaseException as exc:
        _fail_pipeline_run(pipeline_run, exc, summary)
        raise


def _run_district_horizon_job(
    *,
    rows: list[dict[str, Any]],
    folds: list[ValidationFold],
    district_id: str,
    horizon: int,
) -> dict[str, Any]:
    district_rows = sorted(
        (row for row in rows if row["district_id"] == district_id),
        key=lambda row: row["week_start_date"],
    )
    district_name = next(
        (str(row.get("district_name") or district_id) for row in district_rows),
        district_id,
    )
    applicable_folds = [
        (index, fold, matches[0])
        for index, fold in enumerate(folds, start=1)
        if (
            matches := [
                row for row in fold.validation_rows if row["district_id"] == district_id
            ]
        )
    ]
    excluded = len(folds) - len(applicable_folds)
    selection = _select_orders(
        fold=applicable_folds[0][1] if applicable_folds else None,
        district_id=district_id,
    )
    predictions: list[dict[str, Any]] = []
    fold_diagnostics: list[dict[str, Any]] = []
    missing_history_rows = 0

    for fold_index, fold, validation_row in applicable_folds:
        training_rows = sorted(
            (row for row in fold.train_rows if row["district_id"] == district_id),
            key=lambda row: row["week_start_date"],
        )
        missing_history_rows += sum(
            row.get("confirmed_malaria_cases") in {None, ""} for row in training_rows
        )
        fit = _forecast_fold(
            training_rows=training_rows,
            validation_row=validation_row,
            horizon=horizon,
            selected_order=tuple(selection["selected_order"]),
            selected_seasonal_order=tuple(selection["selected_seasonal_order"]),
        )
        diagnostic = {
            "validation_fold": fold_index,
            "origin_week": fold.origin_date.isoformat(),
            "target_week": fold.target_date.isoformat(),
            **fit,
        }
        diagnostic.pop("predicted", None)
        fold_diagnostics.append(diagnostic)
        if fit["status"] == "failed":
            continue
        actual = Decimal(str(validation_row[f"target_cases_h{horizon}"]))
        predicted = Decimal(str(fit["predicted"])).quantize(Decimal("0.0001"))
        threshold = district_threshold(
            fold.train_rows,
            district_id,
            target_epi_week=fold.target_date.isocalendar().week,
        )
        predictions.append(
            {
                "district_id": district_id,
                "district_name": district_name,
                "horizon": horizon,
                "validation_fold": fold_index,
                "forecast_origin_week": fold.origin_date.isoformat(),
                "target_week": fold.target_date.isoformat(),
                "observed_cases": str(actual),
                "predicted_cases": str(predicted),
                "observed_surge_status": actual > threshold,
                "predicted_surge_score": str(predicted),
                "predicted_surge_classification": predicted > threshold,
                "surge_threshold": str(threshold),
                "selected_sarimax_order": json.dumps(fit["order"]),
                "selected_seasonal_order": json.dumps(fit["seasonal_order"]),
                "exogenous_features_used": json.dumps(SARIMAX_EXOGENOUS_FEATURES),
                "convergence_status": fit["converged"],
                "fallback_status": fit["fallback_status"],
                "forecast_clipped_nonnegative": fit["forecast_clipped_nonnegative"],
                "warning_or_failure_reason": fit["warning_or_failure_reason"],
            }
        )

    failed_folds = sum(item["status"] == "failed" for item in fold_diagnostics)
    fallback_folds = sum(
        item.get("fallback_status") not in {None, "none"}
        for item in fold_diagnostics
        if item["status"] != "failed"
    )
    if not predictions:
        status = "failed"
    elif failed_folds:
        status = "partial_success"
    elif fallback_folds:
        status = "fallback"
    else:
        status = "success"
    return {
        "district_id": district_id,
        "district_name": district_name,
        "horizon": horizon,
        "status": status,
        "selected_order": selection["selected_order"],
        "selected_seasonal_order": selection["selected_seasonal_order"],
        "order_selection": selection,
        "predictions": predictions,
        "fold_diagnostics": fold_diagnostics,
        "completed_folds": len(predictions),
        "failed_folds": failed_folds,
        "excluded_missing_target_or_origin": excluded,
        "missing_history_rows": missing_history_rows,
        "fallback_folds": fallback_folds,
    }


def _select_orders(
    *, fold: ValidationFold | None, district_id: str
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if fold is None:
        return {
            "selected_order": list(NONSEASONAL_FALLBACK[0]),
            "selected_seasonal_order": list(NONSEASONAL_FALLBACK[1]),
            "selection_status": "no_evaluation_fold",
            "candidates": candidates,
        }
    training_rows = sorted(
        (row for row in fold.train_rows if row["district_id"] == district_id),
        key=lambda row: row["week_start_date"],
    )
    observed = sum(
        row.get("confirmed_malaria_cases") not in {None, ""} for row in training_rows
    )
    for order, seasonal_order in SARIMAX_ORDER_CANDIDATES:
        if seasonal_order[-1] == 52 and observed < MIN_SEASONAL_HISTORY:
            candidates.append(
                {
                    "order": list(order),
                    "seasonal_order": list(seasonal_order),
                    "status": "skipped_insufficient_seasonal_history",
                    "aic": None,
                }
            )
            continue
        fit = _fit_sarimax(
            training_rows=training_rows,
            order=order,
            seasonal_order=seasonal_order,
            forecast_steps=0,
            origin_row=None,
        )
        candidates.append(
            {
                "order": list(order),
                "seasonal_order": list(seasonal_order),
                "status": fit["status"],
                "converged": fit.get("converged", False),
                "aic": fit.get("aic"),
                "warnings": fit.get("warnings", []),
                "failure_reason": fit.get("failure_reason"),
            }
        )
    usable = [
        item
        for item in candidates
        if item.get("converged") and item.get("aic") is not None
    ]
    if usable:
        selected = min(usable, key=lambda item: float(item["aic"]))
        selection_status = "selected_by_training_aic"
    else:
        selected = {
            "order": list(NONSEASONAL_FALLBACK[0]),
            "seasonal_order": list(NONSEASONAL_FALLBACK[1]),
        }
        selection_status = "controlled_nonseasonal_fallback"
    return {
        "selected_order": selected["order"],
        "selected_seasonal_order": selected["seasonal_order"],
        "selection_status": selection_status,
        "selection_fold_origin": fold.origin_date.isoformat(),
        "validation_outcomes_used_for_selection": False,
        "candidates": candidates,
    }


def _forecast_fold(
    *,
    training_rows: list[dict[str, Any]],
    validation_row: dict[str, Any],
    horizon: int,
    selected_order: tuple[int, ...],
    selected_seasonal_order: tuple[int, ...],
) -> dict[str, Any]:
    attempts = [
        (selected_order, selected_seasonal_order, "none"),
        (*NONSEASONAL_FALLBACK, "nonseasonal"),
        (*REGRESSION_FALLBACK, "regression_only"),
    ]
    unique_attempts: list[tuple[tuple[int, ...], tuple[int, ...], str]] = []
    for attempt in attempts:
        if attempt[:2] not in [(item[0], item[1]) for item in unique_attempts]:
            unique_attempts.append(attempt)
    reasons: list[str] = []
    for order, seasonal_order, fallback_status in unique_attempts:
        fit = _fit_sarimax(
            training_rows=training_rows,
            order=order,
            seasonal_order=seasonal_order,
            forecast_steps=horizon,
            origin_row=validation_row,
        )
        if fit["status"] != "success" or not fit.get("converged"):
            reasons.append(
                f"order={order}, seasonal={seasonal_order}: "
                + str(fit.get("failure_reason") or "did not converge")
            )
            continue
        raw_prediction = float(fit["forecast"])
        clipped = raw_prediction < 0
        predicted = max(0.0, raw_prediction)
        warning_text = "; ".join(fit.get("warnings", [])[:5])
        return {
            "status": "success",
            "predicted": predicted,
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "converged": True,
            "fallback_status": fallback_status,
            "forecast_clipped_nonnegative": clipped,
            "warning_or_failure_reason": warning_text,
            "prediction_interval_lower": fit.get("prediction_interval_lower"),
            "prediction_interval_upper": fit.get("prediction_interval_upper"),
            "fitted_state": fit.get("fitted_state"),
        }
    return {
        "status": "failed",
        "order": list(selected_order),
        "seasonal_order": list(selected_seasonal_order),
        "converged": False,
        "fallback_status": "failed",
        "forecast_clipped_nonnegative": False,
        "warning_or_failure_reason": "; ".join(reasons)[:2000],
    }


def _fit_sarimax(
    *,
    training_rows: list[dict[str, Any]],
    order: tuple[int, ...],
    seasonal_order: tuple[int, ...],
    forecast_steps: int,
    origin_row: dict[str, Any] | None,
) -> dict[str, Any]:
    import numpy as np
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    observed = sum(
        row.get("confirmed_malaria_cases") not in {None, ""} for row in training_rows
    )
    if observed < MIN_OBSERVED_HISTORY:
        return {
            "status": "failed",
            "converged": False,
            "failure_reason": (
                f"Only {observed} observed malaria weeks; "
                f"{MIN_OBSERVED_HISTORY} are required."
            ),
        }
    missing_features = [
        column
        for column in SARIMAX_EXOGENOUS_FEATURES
        if not any(row.get(column) not in {None, ""} for row in training_rows)
    ]
    if missing_features:
        return {
            "status": "failed",
            "converged": False,
            "failure_reason": "Training fold has no values for: " + ", ".join(missing_features),
        }

    imputer = TrainingOnlyImputer(list(SARIMAX_EXOGENOUS_FEATURES)).fit(training_rows)
    transformed_train = imputer.transform(training_rows)
    train_exog = np.asarray(
        [
            [float(row[column]) for column in SARIMAX_EXOGENOUS_FEATURES]
            for row in transformed_train
        ],
        dtype="float64",
    )
    means = train_exog.mean(axis=0)
    scales = train_exog.std(axis=0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    train_exog = (train_exog - means) / scales
    endog = np.asarray(
        [
            float(row["confirmed_malaria_cases"])
            if row.get("confirmed_malaria_cases") not in {None, ""}
            else np.nan
            for row in training_rows
        ],
        dtype="float64",
    )
    future_exog = None
    if forecast_steps:
        if origin_row is None:
            raise SARIMAXValidationError("origin_row is required for forecasting.")
        transformed_origin = imputer.transform([origin_row])[0]
        origin_values = np.asarray(
            [float(transformed_origin[column]) for column in SARIMAX_EXOGENOUS_FEATURES],
            dtype="float64",
        )
        origin_values = (origin_values - means) / scales
        future_exog = np.repeat(origin_values.reshape(1, -1), forecast_steps, axis=0)

    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = SARIMAX(
                endog,
                exog=train_exog,
                order=order,
                seasonal_order=seasonal_order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
                missing="none",
            )
            result = model.fit(disp=False, maxiter=50)
            caught = [str(item.message) for item in captured]
        converged = bool(result.mle_retvals.get("converged", False))
        payload: dict[str, Any] = {
            "status": "success",
            "converged": converged,
            "aic": float(result.aic) if math.isfinite(float(result.aic)) else None,
            "warnings": caught,
            "training_rows": len(training_rows),
            "observed_training_rows": observed,
            "missing_outcome_rows_preserved": len(training_rows) - observed,
            "imputer_fit_rows": imputer.fit_row_count,
            "fitted_state": {
                "parameter_names": [str(name) for name in result.param_names],
                "parameter_values": [float(value) for value in result.params],
                "exogenous_imputer_statistics": {
                    column: str(imputer.statistics_[column])
                    for column in SARIMAX_EXOGENOUS_FEATURES
                },
                "exogenous_means": [float(value) for value in means],
                "exogenous_scales": [float(value) for value in scales],
                "training_start_week": str(training_rows[0]["week_start_date"]),
                "training_end_week": str(training_rows[-1]["week_start_date"]),
            },
        }
        if forecast_steps:
            forecast_result = result.get_forecast(steps=forecast_steps, exog=future_exog)
            forecast = forecast_result.predicted_mean
            value = float(forecast[-1])
            if not math.isfinite(value):
                return {
                    **payload,
                    "status": "failed",
                    "failure_reason": "SARIMAX produced a non-finite forecast.",
                }
            payload["forecast"] = value
            try:
                interval = np.asarray(forecast_result.conf_int(alpha=0.05), dtype="float64")
                lower = float(interval[-1, 0])
                upper = float(interval[-1, 1])
                if math.isfinite(lower) and math.isfinite(upper):
                    payload["prediction_interval_lower"] = max(0.0, lower)
                    payload["prediction_interval_upper"] = max(max(0.0, lower), upper)
            except Exception as exc:
                payload["warnings"].append(
                    f"Prediction interval unavailable: {type(exc).__name__}: {exc}"
                )
        return payload
    except Exception as exc:
        return {
            "status": "failed",
            "converged": False,
            "warnings": caught,
            "failure_reason": f"{type(exc).__name__}: {str(exc)[:1000]}",
        }


def _populate_summary_metrics(
    summary: SARIMAXValidationSummary,
    jobs: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    counts = {
        "success": sum(job["status"] == "success" for job in jobs),
        "fallback": sum(job["status"] == "fallback" for job in jobs),
        "partial_success": sum(job["status"] == "partial_success" for job in jobs),
        "failed": sum(job["status"] == "failed" for job in jobs),
        "total": len(jobs),
    }
    summary.job_counts = counts
    summary.row_counts.update(
        {
            "prediction_rows": len(predictions),
            "failed_folds": sum(job["failed_folds"] for job in jobs),
            "excluded_missing_target_or_origin": sum(
                job["excluded_missing_target_or_origin"] for job in jobs
            ),
        }
    )
    for horizon in summary.horizons:
        records = [
            _prediction_record(item)
            for item in predictions
            if int(item["horizon"]) == horizon
        ]
        count_metrics, surge_metrics = evaluate_predictions(records)
        summary.count_metrics_by_horizon[horizon] = count_metrics
        summary.surge_metrics_by_horizon[horizon] = surge_metrics
    if counts["failed"] or counts["partial_success"]:
        summary.status = "partial_success" if predictions else "failed"
        summary.warnings.append(
            f"{counts['failed']} job(s) failed and {counts['partial_success']} "
            "job(s) completed only partially; see fit diagnostics."
        )
    else:
        summary.status = "success"
    if predictions:
        first = predictions[0]
        summary.threshold_proof.append(
            {
                "horizon": first["horizon"],
                "district_id": first["district_id"],
                "origin_date": first["forecast_origin_week"],
                "target_date": first["target_week"],
                "threshold_training_only": first["surge_threshold"],
                "actual": first["observed_cases"],
                "predicted": first["predicted_cases"],
            }
        )


def _prediction_record(item: dict[str, Any]) -> PredictionRecord:
    return PredictionRecord(
        district_id=str(item["district_id"]),
        horizon=int(item["horizon"]),
        origin_date=date.fromisoformat(str(item["forecast_origin_week"])),
        target_date=date.fromisoformat(str(item["target_week"])),
        actual=Decimal(str(item["observed_cases"])),
        predicted=Decimal(str(item["predicted_cases"])),
        threshold=Decimal(str(item["surge_threshold"])),
        actual_surge=_boolean(item["observed_surge_status"]),
        predicted_surge=_boolean(item["predicted_surge_classification"]),
        surge_score=Decimal(str(item["predicted_surge_score"])),
    )


def _district_metric_row(job: dict[str, Any]) -> dict[str, Any]:
    records = [_prediction_record(item) for item in job["predictions"]]
    count_metrics, surge_metrics = evaluate_predictions(records)
    return {
        "district_id": job["district_id"],
        "district_name": job["district_name"],
        "horizon": job["horizon"],
        "status": job["status"],
        "completed_folds": job["completed_folds"],
        "failed_folds": job["failed_folds"],
        "excluded_missing_target_or_origin": job["excluded_missing_target_or_origin"],
        "missing_history_rows": job["missing_history_rows"],
        "selected_sarimax_order": json.dumps(job["selected_order"]),
        "selected_seasonal_order": json.dumps(job["selected_seasonal_order"]),
        **count_metrics,
        "surge_n": surge_metrics["n"],
        **{key: value for key, value in surge_metrics.items() if key != "n"},
    }


def _diagnostics_payload(
    *,
    summary: SARIMAXValidationSummary,
    jobs: list[dict[str, Any]],
    dataset_signature: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "SARIMAX",
        "status": summary.status,
        "dataset_signature": dataset_signature,
        "job_counts": summary.job_counts,
        "exogenous_features": list(SARIMAX_EXOGENOUS_FEATURES),
        "optimizer_warnings_captured_not_printed": True,
        "jobs": jobs,
    }


def _model_card_payload(summary: SARIMAXValidationSummary) -> dict[str, Any]:
    return {
        "model_name": "SARIMAX",
        "model_scope": "one time-series model per district and forecast horizon",
        "purpose": "Forecast district malaria case counts 4, 8 and 12 weeks ahead.",
        "inputs": {
            "malaria_history": (
                "Observed confirmed_malaria_cases through each forecast origin. "
                "Missing outcomes remain missing in the state-space model and are never zero-filled."
            ),
            "climate_exogenous_variables": list(SARIMAX_EXOGENOUS_FEATURES),
            "future_climate_assumption": (
                "Only lagged climate values available at the forecast origin are used. "
                "They are held constant over the forecast horizon; future climate rows are not read."
            ),
        },
        "validation": (
            "The shared Baseline/Naive rolling-origin folds are reused exactly. "
            "Each training fold ends at the origin, and the target is origin plus horizon weeks."
        ),
        "order_selection": (
            "The lowest AIC among converged approved candidates is selected using "
            "the earliest training fold only. Validation outcomes never select orders."
        ),
        "weekly_seasonality": (
            "A controlled seasonal AR candidate with period 52 is considered when at "
            f"least {MIN_SEASONAL_HISTORY} observed weeks exist."
        ),
        "fallbacks": (
            "An unstable seasonal fit falls back to nonseasonal AR(1), then to a "
            "SARIMAX regression-only specification. No baseline forecast is substituted."
        ),
        "surge_definition": (
            "Observed and predicted counts are compared with the district seasonal "
            "75th-percentile threshold from valid training-only history in the target "
            "epi week +/-2. The confirmed epidemic year 2022 is excluded only from the "
            "normal-reference threshold, not from count-model training."
        ),
        "preprocessing": (
            "Climate imputation and standardization statistics are fitted on training "
            "rows only, then applied to the origin row."
        ),
        "limitations": [
            "Future lagged climate is approximated by holding origin information constant.",
            "SARIMAX may be unstable for districts with short or sparse histories.",
            "Count forecasts are clipped at zero but are not probabilistic count distributions.",
            "Twelve evenly spaced eligible rolling-origin folds cover multiple years but not every week.",
        ],
        "validation_job_counts": summary.job_counts,
        "status": summary.status,
    }


def _stage_outputs(
    *,
    summary: SARIMAXValidationSummary,
    predictions: list[dict[str, Any]],
    district_metrics: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    model_card: dict[str, Any],
    targets: dict[str, Path],
) -> dict[Path, Path]:
    def stage_for(target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.",
            suffix=".staged",
            dir=target.parent,
            delete=False,
        ) as handle:
            return Path(handle.name)

    staged = {
        stage_for(targets["summary"]): targets["summary"],
        stage_for(targets["predictions"]): targets["predictions"],
        stage_for(targets["district_metrics"]): targets["district_metrics"],
        stage_for(targets["diagnostics"]): targets["diagnostics"],
        stage_for(targets["model_card"]): targets["model_card"],
    }
    staged_paths = list(staged)
    try:
        _write_json(staged_paths[0], summary.to_dict())
        _write_csv(
            staged_paths[1],
            predictions,
            PREDICTION_COLUMNS,
        )
        _write_csv(
            staged_paths[2],
            district_metrics,
            DISTRICT_METRIC_COLUMNS,
        )
        _write_json(staged_paths[3], diagnostics)
        _write_json(staged_paths[4], model_card)
    except BaseException:
        for staged_path in staged_paths:
            staged_path.unlink(missing_ok=True)
        raise
    return staged


@contextmanager
def _publish_staged_outputs(staged: dict[Path, Path]) -> Iterator[None]:
    backups: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for staged_path, target in staged.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with tempfile.NamedTemporaryFile(
                    prefix=f".{target.name}.",
                    suffix=".backup",
                    dir=target.parent,
                    delete=False,
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
    dataset_signature: dict[str, Any],
    max_folds: int,
) -> None:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "dataset_signature": dataset_signature,
        "max_folds": max_folds,
        "exogenous_features": list(SARIMAX_EXOGENOUS_FEATURES),
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
    dataset_signature: dict[str, Any],
    horizon: int,
    max_folds: int,
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
        or payload.get("dataset_signature") != dataset_signature
        or payload.get("max_folds") != max_folds
        or payload.get("exogenous_features") != list(SARIMAX_EXOGENOUS_FEATURES)
        or not isinstance(job, dict)
        or int(job.get("horizon", -1)) != horizon
        or job.get("status") not in {"success", "fallback", "partial_success"}
    ):
        return None
    return job


def _start_pipeline_run(
    *,
    summary: SARIMAXValidationSummary,
    dataset_path: Path,
    selected_districts: list[str],
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
        source_name="SARIMAX district-specific validation",
        parameters={
            "model_name": "SARIMAX",
            "dataset_path": str(dataset_path),
            "horizons": list(summary.horizons),
            "district_count": len(selected_districts),
            "max_folds": max_folds,
            "replace": replace,
            "exogenous_features": list(SARIMAX_EXOGENOUS_FEATURES),
        },
    )
    db.session.add(run)
    db.session.commit()
    summary.pipeline_run_id = run.id
    return run


def _finish_pipeline_run(
    run: PipelineRun | None, summary: SARIMAXValidationSummary
) -> None:
    if run is None:
        return
    run.status = "success" if summary.status == "success" else "failed"
    run.completed_at = datetime.now(timezone.utc)
    run.row_count_raw = summary.row_counts.get("modelling_rows", 0)
    run.row_count_processed = summary.row_counts.get("prediction_rows", 0)
    run.warning_count = len(summary.warnings)
    run.error_message = (
        None
        if summary.status == "success"
        else f"SARIMAX validation completed with status {summary.status}."
    )
    run.manifest = {
        "validation_status": summary.status,
        "published": summary.published,
        "job_counts": summary.job_counts,
        "artifact_paths": summary.artifact_paths,
        "count_metrics_by_horizon": summary.count_metrics_by_horizon,
        "surge_metrics_by_horizon": summary.surge_metrics_by_horizon,
        "warnings": summary.warnings,
    }
    db.session.commit()


def _fail_pipeline_run(
    run: PipelineRun | None,
    exc: BaseException,
    summary: SARIMAXValidationSummary,
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


def _validate_horizons(horizons: tuple[int, ...]) -> None:
    if not horizons:
        raise SARIMAXValidationError("At least one horizon is required.")
    unsupported = sorted(set(horizons) - set(HORIZONS))
    if unsupported:
        raise SARIMAXValidationError(
            "Unsupported SARIMAX horizon(s): " + ", ".join(map(str, unsupported))
        )


def _validate_exogenous_contract(columns: Any) -> None:
    available = set(columns)
    missing = sorted(set(SARIMAX_EXOGENOUS_FEATURES) - available)
    if missing:
        raise SARIMAXValidationError(
            "Modelling dataset is missing approved SARIMAX climate fields: "
            + ", ".join(missing)
        )
    if any("current" in column or "target" in column or "wind" in column for column in SARIMAX_EXOGENOUS_FEATURES):
        raise SARIMAXValidationError("SARIMAX exogenous feature contract is not leakage-safe.")


def _selected_districts(
    rows: list[dict[str, Any]], district_ids: tuple[str, ...] | None
) -> list[str]:
    available = sorted({str(row["district_id"]) for row in rows})
    if district_ids is None:
        return available
    requested = list(dict.fromkeys(district_ids))
    missing = sorted(set(requested) - set(available))
    if missing:
        raise SARIMAXValidationError(
            "Requested district(s) are absent from the modelling dataset: "
            + ", ".join(missing)
        )
    return requested


def _dataset_signature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _fold_proofs(folds: list[ValidationFold]) -> list[dict[str, Any]]:
    return [
        {
            "horizon": fold.horizon,
            "origin_week": fold.origin_date.isoformat(),
            "target_week": fold.target_date.isoformat(),
            "train_rows": len(fold.train_rows),
            "trainable_rows": len(fold.trainable_rows),
            "validation_rows": len(fold.validation_rows),
        }
        for fold in folds
    ]


def _gap_proofs(folds: list[ValidationFold]) -> list[dict[str, Any]]:
    return [
        {
            "horizon": fold.horizon,
            "origin": fold.origin_date.isoformat(),
            "target": fold.target_date.isoformat(),
            "gap_days": (fold.target_date - fold.origin_date).days,
            "gap_weeks": (fold.target_date - fold.origin_date).days // 7,
        }
        for fold in folds
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _is_valid_success_artifact(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "success" and bool(
        payload.get("count_metrics_by_horizon")
    )


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}
