"""Leakage-safe final production fitting and transactional forecast publication."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterator

from sqlalchemy import inspect

from app.extensions import db
from app.models import DistrictReference, ForecastOutput, PipelineRun
from app.services.model_validation import district_threshold
from app.services.models.pooled_tree_validation import POOLED_FEATURES
from app.services.models.random_forest import RANDOM_FOREST_CONFIG
from app.services.models.xgboost_model import XGBOOST_CONFIG
from app.services.modelling_dataset import (
    DATASET_PATH,
    HORIZONS,
    modelling_dataset_ready_for_downstream,
    read_modelling_dataset,
)
from app.services.preprocessing import iso_week_dates
from app.services.production_models import (
    ProductionModelError,
    baseline_production_forecast,
    fit_pooled_production_model,
    fit_sarimax_production_model,
    load_validated_parameters,
    load_validated_sarimax_specification,
)
from app.services.risk_classification import classify_forecast_risk

FORECAST_SUMMARY_PATH = Path("data/quality/forecast_generation_summary.json")
FORECAST_DIAGNOSTICS_PATH = Path("data/quality/forecast_generation_diagnostics.json")
PRODUCTION_MODEL_DIR = Path("data/models/production")
PRODUCTION_MANIFEST_PATH = PRODUCTION_MODEL_DIR / "production_model_manifest.json"
SARIMAX_CHECKPOINT_DIR = Path("data/models/sarimax_checkpoints")

VALIDATION_ARTIFACTS = {
    "baseline": Path("data/quality/baseline_validation_summary.json"),
    "sarimax": Path("data/quality/sarimax_validation_summary.json"),
    "random_forest": Path("data/quality/random_forest_validation_summary.json"),
    "xgboost": Path("data/quality/xgboost_validation_summary.json"),
}
MODEL_CARD_PATHS = {
    "random_forest": Path("data/models/random_forest_model_card.json"),
    "xgboost": Path("data/models/xgboost_model_card.json"),
}
MODEL_NAMES = {
    "baseline": "Baseline/Naive",
    "sarimax": "SARIMAX",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
MODEL_SCOPES = {
    "baseline": "district_specific",
    "sarimax": "district_specific",
    "random_forest": "national_pooled",
    "xgboost": "national_pooled",
}
THRESHOLD_METHOD = (
    "district seasonal Q75, target epi week +/-2, training-only valid history; "
    "2022 excluded from reference"
)


class ForecastGenerationError(RuntimeError):
    """Raised when forecasts cannot be generated safely."""


@dataclass
class ForecastGenerationSummary:
    dry_run: bool
    issue_year: int | None
    issue_week: int | None
    models_requested: tuple[str, ...]
    horizons: tuple[int, ...]
    status: str = "planned"
    issue_week_start_date: str | None = None
    target_weeks: dict[int, dict[str, Any]] = field(default_factory=dict)
    production_fitting_ready: bool = False
    expected_model_fits: dict[str, int] = field(default_factory=dict)
    fit_counts: dict[str, int] = field(default_factory=dict)
    models_skipped: dict[str, str] = field(default_factory=dict)
    expected_forecast_rows: int = 0
    generated_forecast_rows: int = 0
    inserted_rows: int = 0
    updated_rows: int = 0
    final_issue_row_count: int = 0
    duplicate_count_before: int = 0
    duplicate_count_after: int = 0
    negative_predictions_clipped: int = 0
    null_prediction_count: int = 0
    surge_output_type_counts: dict[str, int] = field(default_factory=dict)
    composite_index_confirmed: bool = False
    pipeline_run_id: int | None = None
    published: bool = False
    artifact_paths: dict[str, str] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self.__dict__.copy()
        payload["models_requested"] = list(self.models_requested)
        payload["horizons"] = list(self.horizons)
        payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        return payload


def target_week_from_origin(
    issue_year: int, issue_week: int, horizon: int
) -> tuple[int, int, date]:
    if horizon not in HORIZONS:
        raise ForecastGenerationError(f"Unsupported horizon {horizon}.")
    origin_start, _ = iso_week_dates(issue_year, issue_week)
    target_start = origin_start + timedelta(weeks=horizon)
    iso = target_start.isocalendar()
    return iso.year, iso.week, target_start


def clip_non_negative(value: Decimal) -> tuple[Decimal, bool]:
    if value < 0:
        return Decimal("0"), True
    return value, False


def generate_forecasts(
    *,
    latest: bool = False,
    as_of_year: int | None = None,
    as_of_week: int | None = None,
    models: tuple[str, ...] = ("baseline", "sarimax", "random_forest", "xgboost"),
    horizons: tuple[int, ...] = HORIZONS,
    dry_run: bool = False,
    dataset_path: Path = DATASET_PATH,
    summary_path: Path = FORECAST_SUMMARY_PATH,
    diagnostics_path: Path = FORECAST_DIAGNOSTICS_PATH,
    production_dir: Path = PRODUCTION_MODEL_DIR,
    manifest_path: Path | None = None,
    validation_artifacts: dict[str, Path] | None = None,
    model_card_paths: dict[str, Path] | None = None,
    sarimax_checkpoint_dir: Path = SARIMAX_CHECKPOINT_DIR,
    district_ids: tuple[str, ...] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    record_pipeline_run: bool = True,
    publish_database: bool = True,
) -> ForecastGenerationSummary:
    """Fit final models and atomically publish one requested forecast issue."""
    started = time.perf_counter()
    _validate_requested_models_and_horizons(models, horizons)
    issue_year, issue_week = _resolve_issue_week(
        latest=latest, as_of_year=as_of_year, as_of_week=as_of_week
    )
    issue_date, _ = iso_week_dates(issue_year, issue_week)
    validation_paths = validation_artifacts or VALIDATION_ARTIFACTS
    card_paths = model_card_paths or MODEL_CARD_PATHS
    final_manifest_path = manifest_path or production_dir / "production_model_manifest.json"
    selected_districts = _selected_active_districts(district_ids)
    if district_ids is None and len(selected_districts) != 146:
        raise ForecastGenerationError(
            f"district_reference contains {len(selected_districts)} active districts, expected 146."
        )
    summary = ForecastGenerationSummary(
        dry_run=dry_run,
        issue_year=issue_year,
        issue_week=issue_week,
        issue_week_start_date=issue_date.isoformat(),
        models_requested=models,
        horizons=horizons,
        composite_index_confirmed=forecast_composite_index_exists(),
        artifact_paths={
            "production_model_directory": str(production_dir),
            "production_model_manifest": str(final_manifest_path),
            "forecast_generation_summary": str(summary_path),
            "forecast_generation_diagnostics": str(diagnostics_path),
        },
    )
    summary.target_weeks = {
        horizon: _target_dict(issue_year, issue_week, horizon) for horizon in horizons
    }
    summary.expected_forecast_rows = len(selected_districts) * len(models) * len(horizons)
    summary.expected_model_fits = _expected_model_fits(
        models, horizons, len(selected_districts)
    )

    errors = _forecast_precondition_errors(
        models=models,
        horizons=horizons,
        districts=selected_districts,
        issue_date=issue_date,
        dataset_path=dataset_path,
        validation_artifacts=validation_paths,
        model_card_paths=card_paths,
        sarimax_checkpoint_dir=sarimax_checkpoint_dir,
        require_database=publish_database,
    )
    if errors:
        raise ForecastGenerationError("; ".join(errors))
    dataset_rows = read_modelling_dataset(dataset_path)
    if not dataset_rows:
        raise ForecastGenerationError(f"Modelling dataset is empty: {dataset_path}.")
    required_features = set()
    if set(models) & {"random_forest", "xgboost"}:
        required_features.update(POOLED_FEATURES)
    if "sarimax" in models:
        from app.services.models.sarimax import SARIMAX_EXOGENOUS_FEATURES

        required_features.update(SARIMAX_EXOGENOUS_FEATURES)
    missing_features = sorted(required_features - set(dataset_rows[0]))
    if missing_features:
        raise ForecastGenerationError(
            "Modelling dataset is missing required production predictors: "
            + ", ".join(missing_features)
        )
    rows = [
        row
        for row in dataset_rows
        if date.fromisoformat(str(row["week_start_date"])) <= issue_date
        and str(row["district_id"]) in selected_districts
    ]
    origin_by_district = {
        str(row["district_id"]): row
        for row in rows
        if str(row["week_start_date"]) == issue_date.isoformat()
    }
    missing_origins = sorted(set(selected_districts) - set(origin_by_district))
    if missing_origins:
        raise ForecastGenerationError(
            "Issue-week modelling rows are missing for: " + ", ".join(missing_origins)
        )
    summary.production_fitting_ready = True
    if dry_run:
        summary.status = "dry-run"
        summary.runtime_seconds = time.perf_counter() - started
        return summary

    dataset_signature = _dataset_signature(dataset_path)
    pipeline_run = _start_pipeline_run(
        summary=summary,
        dataset_path=dataset_path,
        district_count=len(selected_districts),
        enabled=record_pipeline_run,
    )
    if publish_database and pipeline_run is None:
        raise ForecastGenerationError(
            "Database publication requires a recorded forecast_generation PipelineRun."
        )

    staged: dict[Path, Path] = {}
    forecast_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    fit_diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    progress_total = sum(summary.expected_model_fits.values()) + (
        len(horizons) if "baseline" in models else 0
    )
    progress_completed = 0

    try:
        for model_key in models:
            if model_key == "baseline":
                for horizon in horizons:
                    target = summary.target_weeks[horizon]
                    horizon_forecasts: list[dict[str, Any]] = []
                    for district_id in selected_districts:
                        district_rows = [
                            row for row in rows if str(row["district_id"]) == district_id
                        ]
                        try:
                            forecast = baseline_production_forecast(
                                district_rows=district_rows,
                                issue_date=issue_date,
                                target_epi_week=int(target["target_epi_week"]),
                            )
                            horizon_forecasts.append(
                                _forecast_row(
                                    model_key=model_key,
                                    horizon=horizon,
                                    issue_year=issue_year,
                                    issue_week=issue_week,
                                    issue_date=issue_date,
                                    target=target,
                                    district_id=district_id,
                                    forecast=forecast,
                                    rows=rows,
                                    dataset_signature=dataset_signature,
                                    model_signature={"version": "approved-baseline-rule-v1"},
                                )
                            )
                        except Exception as exc:
                            failures.append(_failure(model_key, horizon, district_id, exc))
                    forecast_rows.extend(horizon_forecasts)
                    entry = _baseline_manifest_entry(
                        horizon=horizon,
                        issue_year=issue_year,
                        issue_week=issue_week,
                        issue_date=issue_date,
                        rows=rows,
                        dataset_signature=dataset_signature,
                        forecast_count=len(horizon_forecasts),
                    )
                    manifest_entries.append(entry)
                    fit_diagnostics.append(dict(entry))
                    progress_completed += 1
                    _report_progress(
                        progress_callback,
                        completed=progress_completed,
                        total=progress_total,
                        model_name=MODEL_NAMES[model_key],
                        horizon=horizon,
                        district_id=None,
                        status="success" if len(horizon_forecasts) == len(selected_districts) else "partial_success",
                    )

            elif model_key == "sarimax":
                for horizon in horizons:
                    target = summary.target_weeks[horizon]
                    for district_id in selected_districts:
                        final_artifact = (
                            production_dir
                            / "sarimax"
                            / f"h{horizon}"
                            / f"{district_id}.json"
                        )
                        staged_artifact = _new_staged_path(final_artifact)
                        staged[staged_artifact] = final_artifact
                        district_rows = [
                            row for row in rows if str(row["district_id"]) == district_id
                        ]
                        try:
                            job = fit_sarimax_production_model(
                                district_id=district_id,
                                district_rows=district_rows,
                                origin_row=origin_by_district[district_id],
                                horizon=horizon,
                                checkpoint_path=sarimax_checkpoint_dir
                                / f"{district_id}_h{horizon}.json",
                                staged_artifact_path=staged_artifact,
                                final_artifact_path=final_artifact,
                                dataset_signature=dataset_signature,
                            )
                        except Exception as exc:
                            job = _failure(model_key, horizon, district_id, exc)
                            job["forecasts"] = []
                            staged.pop(staged_artifact, None)
                            staged_artifact.unlink(missing_ok=True)
                        if job.get("fit_status") == "success":
                            manifest_entries.append(_manifest_entry(job, issue_year, issue_week, issue_date))
                            for forecast in job["forecasts"]:
                                forecast_rows.append(
                                    _forecast_row(
                                        model_key=model_key,
                                        horizon=horizon,
                                        issue_year=issue_year,
                                        issue_week=issue_week,
                                        issue_date=issue_date,
                                        target=target,
                                        district_id=district_id,
                                        forecast=forecast,
                                        rows=rows,
                                        dataset_signature=dataset_signature,
                                        model_signature=job.get("model_signature") or {},
                                    )
                                )
                        else:
                            failures.append(job)
                        fit_diagnostics.append(_compact_job(job))
                        progress_completed += 1
                        _report_progress(
                            progress_callback,
                            completed=progress_completed,
                            total=progress_total,
                            model_name=MODEL_NAMES[model_key],
                            horizon=horizon,
                            district_id=district_id,
                            status=str(job.get("status", "failed")),
                        )

            else:
                config = (
                    RANDOM_FOREST_CONFIG if model_key == "random_forest" else XGBOOST_CONFIG
                )
                for horizon in horizons:
                    final_artifact = production_dir / f"{model_key}_h{horizon}_pipeline.joblib"
                    staged_artifact = _new_staged_path(final_artifact)
                    staged[staged_artifact] = final_artifact
                    try:
                        job = fit_pooled_production_model(
                            config=config,
                            rows=rows,
                            origin_rows=[origin_by_district[item] for item in selected_districts],
                            issue_date=issue_date,
                            horizon=horizon,
                            model_card_path=card_paths[model_key],
                            staged_artifact_path=staged_artifact,
                            final_artifact_path=final_artifact,
                            dataset_signature=dataset_signature,
                        )
                    except Exception as exc:
                        job = _failure(model_key, horizon, None, exc)
                        job["forecasts"] = []
                        failures.append(job)
                        staged.pop(staged_artifact, None)
                        staged_artifact.unlink(missing_ok=True)
                    if job.get("fit_status") == "success":
                        manifest_entries.append(_manifest_entry(job, issue_year, issue_week, issue_date))
                        target = summary.target_weeks[horizon]
                        for forecast in job["forecasts"]:
                            forecast_rows.append(
                                _forecast_row(
                                    model_key=model_key,
                                    horizon=horizon,
                                    issue_year=issue_year,
                                    issue_week=issue_week,
                                    issue_date=issue_date,
                                    target=target,
                                    district_id=str(forecast["district_id"]),
                                    forecast=forecast,
                                    rows=rows,
                                    dataset_signature=dataset_signature,
                                    model_signature=job.get("model_signature") or {},
                                )
                            )
                    fit_diagnostics.append(_compact_job(job))
                    progress_completed += 1
                    _report_progress(
                        progress_callback,
                        completed=progress_completed,
                        total=progress_total,
                        model_name=MODEL_NAMES[model_key],
                        horizon=horizon,
                        district_id=None,
                        status=str(job.get("status", "failed")),
                    )

        summary.generated_forecast_rows = len(forecast_rows)
        summary.negative_predictions_clipped = sum(
            bool(row["forecast_clipped_nonnegative"]) for row in forecast_rows
        )
        summary.null_prediction_count = sum(
            row.get("predicted_cases") is None for row in forecast_rows
        )
        summary.surge_output_type_counts = dict(
            Counter(row["surge_output_type"] for row in forecast_rows)
        )
        summary.fit_counts = dict(
            Counter(str(job.get("status", "failed")) for job in fit_diagnostics)
        )
        validation_errors = _validate_generated_rows(
            forecast_rows=forecast_rows,
            expected_count=summary.expected_forecast_rows,
            models=models,
            horizons=horizons,
            districts=selected_districts,
        )
        if failures or validation_errors:
            summary.status = "partial_success" if forecast_rows else "failed"
            summary.warnings.extend(validation_errors)
            summary.warnings.append(
                f"{len(failures)} production fit or forecast job(s) failed; nothing was published."
            )
            summary.runtime_seconds = time.perf_counter() - started
            _cleanup_staged(staged)
            _fail_pipeline_run(pipeline_run, summary, failures)
            return summary

        if publish_database:
            inserted, updated, duplicates_before = _upsert_forecast_rows(
                forecast_rows, pipeline_run
            )
            summary.inserted_rows = inserted
            summary.updated_rows = updated
            summary.duplicate_count_before = duplicates_before
            summary.final_issue_row_count = _selected_issue_row_count(
                issue_year, issue_week, models, horizons, selected_districts
            )
            if summary.final_issue_row_count != summary.expected_forecast_rows:
                raise ForecastGenerationError(
                    "Staged database row count does not equal the expected forecast count."
                )
        summary.status = "success"
        summary.published = True
        summary.runtime_seconds = time.perf_counter() - started
        manifest = _production_manifest(
            summary=summary,
            dataset_signature=dataset_signature,
            entries=manifest_entries,
        )
        diagnostics = {
            "status": summary.status,
            "issue_week": f"{issue_year}W{issue_week:02d}",
            "dataset_signature": dataset_signature,
            "jobs": fit_diagnostics,
            "forecast_rows": forecast_rows,
            "failures": failures,
            "no_future_climate_rows_read": True,
            "surge_score_rule": "predicted_cases - training-only district threshold",
            "interval_policy": (
                "Only SARIMAX model-based 95% intervals are populated; Baseline, Random Forest, and XGBoost intervals remain null."
            ),
        }
        staged_manifest = _new_staged_path(final_manifest_path)
        staged_summary = _new_staged_path(summary_path)
        staged_diagnostics = _new_staged_path(diagnostics_path)
        _write_json(staged_manifest, manifest)
        _write_json(staged_summary, summary.to_dict())
        _write_json(staged_diagnostics, diagnostics)
        staged[staged_manifest] = final_manifest_path
        staged[staged_summary] = summary_path
        staged[staged_diagnostics] = diagnostics_path

        with publish_staged_outputs(staged):
            if pipeline_run is not None:
                _finish_pipeline_run(pipeline_run, summary)
            elif publish_database:
                raise ForecastGenerationError("Forecast database transaction has no PipelineRun.")
        pipeline_run = None
        return summary
    except BaseException as exc:
        db.session.rollback()
        _cleanup_staged(staged)
        _fail_pipeline_run_exception(pipeline_run, summary, exc)
        raise


def forecast_composite_index_exists() -> bool:
    inspector = inspect(db.engine)
    indexes = inspector.get_indexes(ForecastOutput.__tablename__)
    for index in indexes:
        columns = index.get("column_names") or []
        if columns[:3] == ["district_id", "target_epi_year", "target_epi_week"]:
            return True
    return False


def _resolve_issue_week(
    *, latest: bool, as_of_year: int | None, as_of_week: int | None
) -> tuple[int, int]:
    if latest:
        from app.models import DistrictWeekFeature

        feature = DistrictWeekFeature.query.order_by(
            DistrictWeekFeature.week_start_date.desc()
        ).first()
        if feature is None:
            raise ForecastGenerationError("No feature rows exist for latest forecast issue week.")
        return feature.epi_year, feature.epi_week
    if as_of_year is None or as_of_week is None:
        raise ForecastGenerationError("Use --latest or provide --as-of-year and --as-of-week.")
    try:
        iso_week_dates(as_of_year, as_of_week)
    except ValueError as exc:
        raise ForecastGenerationError(str(exc)) from exc
    return as_of_year, as_of_week


def _forecast_precondition_errors(
    *,
    models: tuple[str, ...],
    horizons: tuple[int, ...],
    districts: list[str],
    issue_date: date,
    dataset_path: Path,
    validation_artifacts: dict[str, Path],
    model_card_paths: dict[str, Path],
    sarimax_checkpoint_dir: Path,
    require_database: bool,
) -> list[str]:
    errors: list[str] = []
    dependencies = {"joblib"}
    if "sarimax" in models:
        dependencies.add("statsmodels")
    if "random_forest" in models:
        dependencies.add("sklearn")
    if "xgboost" in models:
        dependencies.update({"sklearn", "xgboost"})
    missing_dependencies = sorted(
        dependency
        for dependency in dependencies
        if importlib.util.find_spec(dependency) is None
    )
    if missing_dependencies:
        errors.append(
            "Missing production modelling dependencies: "
            + ", ".join(missing_dependencies)
        )
    if len(districts) == 146 and DistrictReference.query.filter_by(active=True).count() != 146:
        errors.append("district_reference must contain 146 active districts.")
    if dataset_path == DATASET_PATH:
        ready, message = modelling_dataset_ready_for_downstream(dataset_path)
        if not ready:
            errors.append(message)
    elif not dataset_path.exists():
        errors.append(f"Modelling dataset not found at {dataset_path}.")
    if require_database and not forecast_composite_index_exists():
        errors.append("Forecast target composite index is missing.")
    for model in models:
        path = validation_artifacts.get(model)
        payload = _read_json(path) if path else None
        if payload is None:
            errors.append(f"Validation artifact missing or invalid for {model}: {path}")
        elif payload.get("status") != "success":
            acceptable_sarimax_partial = (
                model == "sarimax"
                and payload.get("status") == "partial_success"
                and int((payload.get("job_counts") or {}).get("failed", -1)) == 0
                and int((payload.get("row_counts") or {}).get("prediction_rows", 0)) > 0
            )
            if not acceptable_sarimax_partial:
                errors.append(f"Validation artifact for {model} is not successful or an approved zero-failure SARIMAX partial validation.")
        elif payload.get("model_name") != MODEL_NAMES[model]:
            errors.append(f"Validation artifact model name is invalid for {model}.")
    for model in set(models) & {"random_forest", "xgboost"}:
        card = model_card_paths.get(model)
        for horizon in horizons:
            try:
                load_validated_parameters(
                    model_card_path=card,
                    model_name=MODEL_NAMES[model],
                    horizon=horizon,
                )
            except (ProductionModelError, TypeError) as exc:
                errors.append(str(exc))
    if "sarimax" in models:
        for district_id in districts:
            for horizon in horizons:
                checkpoint = sarimax_checkpoint_dir / f"{district_id}_h{horizon}.json"
                try:
                    load_validated_sarimax_specification(
                        checkpoint_path=checkpoint,
                        district_id=district_id,
                        horizon=horizon,
                    )
                except ProductionModelError as exc:
                    errors.append(str(exc))
                    if len(errors) >= 20:
                        errors.append("Additional SARIMAX checkpoint errors were suppressed.")
                        return errors
    return errors


def _forecast_row(
    *,
    model_key: str,
    horizon: int,
    issue_year: int,
    issue_week: int,
    issue_date: date,
    target: dict[str, Any],
    district_id: str,
    forecast: dict[str, Any],
    rows: list[dict[str, Any]],
    dataset_signature: dict[str, Any],
    model_signature: dict[str, Any],
) -> dict[str, Any]:
    predicted = Decimal(str(forecast["predicted_cases"])).quantize(Decimal("0.0001"))
    predicted, clipped_here = clip_non_negative(predicted)
    threshold = district_threshold(
        rows,
        district_id,
        target_epi_week=int(target["target_epi_week"]),
    ).quantize(Decimal("0.0001"))
    surge_score = (predicted - threshold).quantize(Decimal("0.0001"))
    probability_value = forecast.get("surge_probability")
    surge_probability = (
        Decimal(str(probability_value)).quantize(Decimal("0.00001"))
        if probability_value is not None
        else None
    )
    operational_surge_method = forecast.get("operational_surge_method")
    if operational_surge_method == "count_threshold":
        predicted_surge = surge_score > 0
        risk_probability = None
    else:
        flag = forecast.get("predicted_surge_flag")
        predicted_surge = surge_score > 0 if flag is None else bool(flag)
        risk_probability = surge_probability
    risk = classify_forecast_risk(
        predicted_surge_flag=predicted_surge,
        surge_probability=risk_probability,
        predicted_cases=predicted,
        threshold_used=threshold,
    )
    lower = forecast.get("prediction_interval_lower")
    upper = forecast.get("prediction_interval_upper")
    return {
        "district_id": district_id,
        "model_key": model_key,
        "model_name": MODEL_NAMES[model_key],
        "model_scope": MODEL_SCOPES[model_key],
        "forecast_origin_epi_year": issue_year,
        "forecast_origin_epi_week": issue_week,
        "forecast_origin_date": issue_date.isoformat(),
        "horizon_weeks": horizon,
        "target_epi_year": int(target["target_epi_year"]),
        "target_epi_week": int(target["target_epi_week"]),
        "target_week_start_date": str(target["target_week_start_date"]),
        "predicted_cases": str(predicted),
        "prediction_interval_lower": str(max(0.0, float(lower))) if lower is not None else None,
        "prediction_interval_upper": str(max(0.0, float(upper))) if upper is not None else None,
        "surge_probability": str(surge_probability) if surge_probability is not None else None,
        "threshold_used": str(threshold),
        "predicted_surge_score": str(surge_score),
        "predicted_surge_flag": predicted_surge,
        "risk_level": risk.risk_level,
        "risk_basis": risk.risk_basis,
        "surge_output_type": risk.surge_output_type,
        "threshold_method": THRESHOLD_METHOD,
        "model_status": forecast.get("model_status", "success"),
        "fallback_status": forecast.get("fallback_status", "none"),
        "feature_availability_status": forecast.get("feature_availability_status", "complete"),
        "missing_predictor_count": int(forecast.get("missing_predictor_count", 0)),
        "forecast_clipped_nonnegative": bool(
            forecast.get("forecast_clipped_nonnegative") or clipped_here
        ),
        "model_version": model_signature.get("sha256")
        or model_signature.get("version")
        or "unknown",
        "dataset_signature": dataset_signature,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "explanation": forecast.get("explanation"),
    }


def _upsert_forecast_rows(
    forecast_rows: list[dict[str, Any]], pipeline_run: PipelineRun | None
) -> tuple[int, int, int]:
    if pipeline_run is None:
        raise ForecastGenerationError("PipelineRun is required for forecast publication.")
    issue_year = int(forecast_rows[0]["forecast_origin_epi_year"])
    issue_week = int(forecast_rows[0]["forecast_origin_epi_week"])
    model_names = sorted({str(row["model_name"]) for row in forecast_rows})
    horizons = sorted({int(row["horizon_weeks"]) for row in forecast_rows})
    districts = sorted({str(row["district_id"]) for row in forecast_rows})
    existing_rows = ForecastOutput.query.filter(
        ForecastOutput.forecast_origin_epi_year == issue_year,
        ForecastOutput.forecast_origin_epi_week == issue_week,
        ForecastOutput.model_name.in_(model_names),
        ForecastOutput.horizon_weeks.in_(horizons),
        ForecastOutput.district_id.in_(districts),
    ).all()
    grouped: dict[tuple[str, str, int, int, int], list[ForecastOutput]] = defaultdict(list)
    for existing in existing_rows:
        grouped[_database_key(existing)].append(existing)
    duplicates = sum(max(0, len(items) - 1) for items in grouped.values())
    if duplicates:
        raise ForecastGenerationError(
            f"Found {duplicates} duplicate matching forecast rows before publication."
        )
    inserted = 0
    updated = 0
    generated_at = datetime.now(timezone.utc)
    for row in forecast_rows:
        key = _row_key(row)
        existing = grouped.get(key, [])
        if existing:
            output = existing[0]
            updated += 1
        else:
            output = ForecastOutput()
            db.session.add(output)
            inserted += 1
        output.pipeline_run_id = pipeline_run.id
        output.district_id = row["district_id"]
        output.model_name = row["model_name"]
        output.model_scope = row["model_scope"]
        output.forecast_origin_epi_year = row["forecast_origin_epi_year"]
        output.forecast_origin_epi_week = row["forecast_origin_epi_week"]
        output.forecast_origin_date = date.fromisoformat(row["forecast_origin_date"])
        output.horizon_weeks = row["horizon_weeks"]
        output.target_epi_year = row["target_epi_year"]
        output.target_epi_week = row["target_epi_week"]
        output.target_week_start_date = date.fromisoformat(row["target_week_start_date"])
        output.predicted_cases = Decimal(row["predicted_cases"])
        output.prediction_interval_lower = (
            Decimal(row["prediction_interval_lower"])
            if row["prediction_interval_lower"] is not None
            else None
        )
        output.prediction_interval_upper = (
            Decimal(row["prediction_interval_upper"])
            if row["prediction_interval_upper"] is not None
            else None
        )
        output.surge_probability = None
        output.predicted_surge_flag = row["predicted_surge_flag"]
        output.risk_level = row["risk_level"]
        output.threshold_used = Decimal(row["threshold_used"])
        output.threshold_method = row["threshold_method"]
        output.created_at = generated_at
    db.session.flush()
    return inserted, updated, duplicates


def _selected_issue_row_count(
    issue_year: int,
    issue_week: int,
    models: tuple[str, ...],
    horizons: tuple[int, ...],
    districts: list[str],
) -> int:
    return ForecastOutput.query.filter(
        ForecastOutput.forecast_origin_epi_year == issue_year,
        ForecastOutput.forecast_origin_epi_week == issue_week,
        ForecastOutput.model_name.in_([MODEL_NAMES[item] for item in models]),
        ForecastOutput.horizon_weeks.in_(horizons),
        ForecastOutput.district_id.in_(districts),
    ).count()


def _validate_generated_rows(
    *,
    forecast_rows: list[dict[str, Any]],
    expected_count: int,
    models: tuple[str, ...],
    horizons: tuple[int, ...],
    districts: list[str],
) -> list[str]:
    errors: list[str] = []
    keys = [_row_key(row) for row in forecast_rows]
    if len(forecast_rows) != expected_count:
        errors.append(
            f"Generated {len(forecast_rows)} forecast rows, expected {expected_count}."
        )
    if len(set(keys)) != len(keys):
        errors.append("Generated forecast rows contain duplicate issue/model/district/horizon keys.")
    for model in models:
        for horizon in horizons:
            represented = {
                row["district_id"]
                for row in forecast_rows
                if row["model_name"] == MODEL_NAMES[model]
                and int(row["horizon_weeks"]) == horizon
            }
            if represented != set(districts):
                errors.append(
                    f"{MODEL_NAMES[model]} h{horizon} represents {len(represented)} districts, expected {len(districts)}."
                )
    if any(row.get("predicted_cases") is None for row in forecast_rows):
        errors.append("Generated forecast rows contain null predicted cases.")
    return errors


def _production_manifest(
    *,
    summary: ForecastGenerationSummary,
    dataset_signature: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "manifest_version": 1,
        "status": summary.status,
        "issue_epi_year": summary.issue_year,
        "issue_epi_week": summary.issue_week,
        "issue_week_start_date": summary.issue_week_start_date,
        "models": [MODEL_NAMES[item] for item in summary.models_requested],
        "horizons": list(summary.horizons),
        "expected_forecast_rows": summary.expected_forecast_rows,
        "dataset_signature": dataset_signature,
        "approved_pooled_predictors": list(POOLED_FEATURES),
        "entries": entries,
        "leakage_controls": {
            "training_through_issue_week_only": True,
            "target_columns_excluded": True,
            "future_climate_rows_read": False,
            "validation_estimators_reused": False,
            "outcome_imputation": False,
            "missing_malaria_zero_filled": False,
            "random_split": False,
            "surge_threshold_training_only": True,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _manifest_entry(
    job: dict[str, Any], issue_year: int, issue_week: int, issue_date: date
) -> dict[str, Any]:
    keys = (
        "model_name",
        "model_scope",
        "district_id",
        "horizon",
        "training_row_count",
        "latest_training_week",
        "latest_training_feature_week",
        "predictors",
        "selected_parameters",
        "preprocessing",
        "dataset_signature",
        "model_signature",
        "model_artifact_path",
        "fit_status",
        "fallback_information",
        "limitations",
        "missing_outcome_exclusions",
    )
    return {
        "issue_week": f"{issue_year}W{issue_week:02d}",
        "issue_week_start_date": issue_date.isoformat(),
        **{key: job.get(key) for key in keys},
    }


def _baseline_manifest_entry(
    *,
    horizon: int,
    issue_year: int,
    issue_week: int,
    issue_date: date,
    rows: list[dict[str, Any]],
    dataset_signature: dict[str, Any],
    forecast_count: int,
) -> dict[str, Any]:
    return {
        "model_name": "Baseline/Naive",
        "model_scope": "district_specific",
        "issue_week": f"{issue_year}W{issue_week:02d}",
        "issue_week_start_date": issue_date.isoformat(),
        "district_id": None,
        "horizon": horizon,
        "training_row_count": len(rows),
        "latest_training_week": issue_date.isoformat(),
        "latest_training_feature_week": issue_date.isoformat(),
        "predictors": ["historical confirmed_malaria_cases", "target epi_week"],
        "selected_parameters": {
            "rule": "same historical epi week, else trailing four observed counts"
        },
        "preprocessing": "none",
        "dataset_signature": dataset_signature,
        "model_signature": {"version": "approved-baseline-rule-v1"},
        "model_artifact_path": None,
        "fit_status": "success" if forecast_count else "failed",
        "status": "success" if forecast_count else "failed",
        "fallback_information": "deterministic_rule",
        "limitations": [
            "No prediction interval is available for the deterministic baseline rule."
        ],
        "missing_outcome_exclusions": sum(
            row.get("confirmed_malaria_cases") in {None, ""} for row in rows
        ),
    }


def _expected_model_fits(
    models: tuple[str, ...], horizons: tuple[int, ...], district_count: int
) -> dict[str, int]:
    return {
        MODEL_NAMES[model]: (
            district_count * len(horizons)
            if model == "sarimax"
            else len(horizons)
            if model in {"random_forest", "xgboost"}
            else 0
        )
        for model in models
    }


def _validate_requested_models_and_horizons(
    models: tuple[str, ...], horizons: tuple[int, ...]
) -> None:
    if not models:
        raise ForecastGenerationError("At least one model is required.")
    unsupported_models = sorted(set(models) - set(MODEL_NAMES))
    if unsupported_models:
        raise ForecastGenerationError(
            "Unsupported model request(s): " + ", ".join(unsupported_models)
        )
    if len(set(models)) != len(models):
        raise ForecastGenerationError("Model requests must not contain duplicates.")
    if not horizons:
        raise ForecastGenerationError("At least one horizon is required.")
    unsupported_horizons = sorted(set(horizons) - set(HORIZONS))
    if unsupported_horizons:
        raise ForecastGenerationError(
            "Unsupported horizon(s): " + ", ".join(map(str, unsupported_horizons))
        )
    if len(set(horizons)) != len(horizons):
        raise ForecastGenerationError("Horizon requests must not contain duplicates.")


def _selected_active_districts(district_ids: tuple[str, ...] | None) -> list[str]:
    active = {
        row.district_id
        for row in DistrictReference.query.filter_by(active=True)
        .order_by(DistrictReference.district_id)
        .all()
    }
    if district_ids is None:
        return sorted(active)
    requested = set(district_ids)
    missing = sorted(requested - active)
    if missing:
        raise ForecastGenerationError(
            "Requested districts are not active: " + ", ".join(missing)
        )
    return sorted(requested)


def _start_pipeline_run(
    *,
    summary: ForecastGenerationSummary,
    dataset_path: Path,
    district_count: int,
    enabled: bool,
) -> PipelineRun | None:
    if not enabled:
        return None
    run = PipelineRun(
        run_type="forecast_generation",
        status="running",
        started_at=datetime.now(timezone.utc),
        source_name="Final production forecast generation",
        parameters={
            "issue_year": summary.issue_year,
            "issue_week": summary.issue_week,
            "models": list(summary.models_requested),
            "horizons": list(summary.horizons),
            "district_count": district_count,
            "expected_forecast_rows": summary.expected_forecast_rows,
            "dataset_path": str(dataset_path),
        },
    )
    db.session.add(run)
    db.session.commit()
    summary.pipeline_run_id = run.id
    return run


def _finish_pipeline_run(
    run: PipelineRun, summary: ForecastGenerationSummary
) -> None:
    run.status = "success"
    run.completed_at = datetime.now(timezone.utc)
    run.row_count_raw = summary.expected_forecast_rows
    run.row_count_processed = summary.generated_forecast_rows
    run.warning_count = len(summary.warnings)
    run.error_message = None
    run.manifest = {
        "status": summary.status,
        "published": summary.published,
        "issue_year": summary.issue_year,
        "issue_week": summary.issue_week,
        "models": list(summary.models_requested),
        "horizons": list(summary.horizons),
        "expected_forecast_rows": summary.expected_forecast_rows,
        "final_issue_row_count": summary.final_issue_row_count,
        "artifact_paths": summary.artifact_paths,
        "fit_counts": summary.fit_counts,
    }
    db.session.commit()


def _fail_pipeline_run(
    run: PipelineRun | None,
    summary: ForecastGenerationSummary,
    failures: list[dict[str, Any]],
) -> None:
    if run is None:
        return
    db.session.rollback()
    stored = db.session.get(PipelineRun, run.id)
    if stored is None:
        return
    stored.status = "failed"
    stored.completed_at = datetime.now(timezone.utc)
    stored.row_count_raw = summary.expected_forecast_rows
    stored.row_count_processed = summary.generated_forecast_rows
    stored.warning_count = len(summary.warnings)
    stored.error_message = f"Production forecast generation completed with {len(failures)} failure(s)."
    stored.manifest = {
        "status": summary.status,
        "published": False,
        "failure_count": len(failures),
        "failures": failures[:50],
        "artifact_paths": summary.artifact_paths,
    }
    db.session.commit()


def _fail_pipeline_run_exception(
    run: PipelineRun | None,
    summary: ForecastGenerationSummary,
    exc: BaseException,
) -> None:
    if run is None:
        return
    db.session.rollback()
    stored = db.session.get(PipelineRun, run.id)
    if stored is None:
        return
    stored.status = "failed"
    stored.completed_at = datetime.now(timezone.utc)
    stored.row_count_raw = summary.expected_forecast_rows
    stored.row_count_processed = summary.generated_forecast_rows
    stored.warning_count = len(summary.warnings)
    stored.error_message = f"{type(exc).__name__}: {str(exc)[:900]}"
    stored.manifest = {
        "status": "failed",
        "published": False,
        "artifact_paths": summary.artifact_paths,
    }
    db.session.commit()


def _target_dict(issue_year: int, issue_week: int, horizon: int) -> dict[str, Any]:
    year, week, start_date = target_week_from_origin(issue_year, issue_week, horizon)
    return {
        "target_epi_year": year,
        "target_epi_week": week,
        "target_week_start_date": start_date.isoformat(),
    }


def _dataset_signature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _new_staged_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.", suffix=".staged", dir=target.parent, delete=False
    ) as handle:
        return Path(handle.name)


@contextmanager
def publish_staged_outputs(staged: dict[Path, Path]) -> Iterator[None]:
    """Publish all model/proof files with rollback if DB commit or file moves fail."""
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
            _remove_path(target)
        for target, backup in backups.items():
            if backup.exists():
                backup.replace(target)
        raise
    else:
        for backup in backups.values():
            try:
                _remove_path(backup)
            except OSError:
                pass
    finally:
        _cleanup_staged(staged)


def _cleanup_staged(staged: dict[Path, Path]) -> None:
    for staged_path in staged:
        _remove_path(staged_path)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _database_key(row: ForecastOutput) -> tuple[str, str, int, int, int]:
    return (
        row.district_id,
        row.model_name,
        row.forecast_origin_epi_year,
        row.forecast_origin_epi_week,
        row.horizon_weeks,
    )


def _row_key(row: dict[str, Any]) -> tuple[str, str, int, int, int]:
    return (
        str(row["district_id"]),
        str(row["model_name"]),
        int(row["forecast_origin_epi_year"]),
        int(row["forecast_origin_epi_week"]),
        int(row["horizon_weeks"]),
    )


def _compact_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key != "forecasts"}


def _failure(
    model_key: str, horizon: int, district_id: str | None, exc: BaseException
) -> dict[str, Any]:
    return {
        "model_name": MODEL_NAMES[model_key],
        "model_scope": MODEL_SCOPES[model_key],
        "district_id": district_id,
        "horizon": horizon,
        "status": "failed",
        "fit_status": "failed",
        "fallback_information": "failed",
        "failure_reason": f"{type(exc).__name__}: {str(exc)[:1500]}",
    }


def _report_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    completed: int,
    total: int,
    model_name: str,
    horizon: int,
    district_id: str | None,
    status: str,
) -> None:
    if callback is None:
        return
    callback(
        {
            "completed": completed,
            "total": total,
            "model": model_name,
            "horizon": horizon,
            "district_id": district_id,
            "status": status,
        }
    )
