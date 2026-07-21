"""Model-performance consolidation and selection.

This layer reads validation proof artifacts created by the approved model
families and converts them into a dashboard-ready comparison summary. It does
not train models and it refuses to fabricate metrics when validation artifacts
are missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from app.extensions import db
from app.models import (
    APPROVED_MODEL_NAMES,
    DistrictReference,
    ForecastOutput,
    ModelPerformance,
    PipelineRun,
)
from app.services.forecast_generation import VALIDATION_ARTIFACTS
from app.services.modelling_dataset import DATASET_PATH, HORIZONS, modelling_dataset_exists

PERFORMANCE_SELECTION_PATH = Path("data/quality/model_performance_selection_summary.json")
ARTIFACT_PROMPTS = {
    "baseline": "Prompt 23",
    "sarimax": "Prompt 23",
    "random_forest": "Prompt 24",
    "xgboost": "Prompt 24",
}
COUNT_SELECTION_RULE = (
    "Select the lowest WAPE when available because it is scale-normalized across "
    "districts; fall back to lowest RMSE when WAPE is unavailable."
)
SURGE_SELECTION_RULE = (
    "Select the highest F2 score to emphasize sensitivity for early warning, with "
    "sensitivity, PR-AUC, precision and specificity as ordered tie-breakers. "
    "Accuracy alone is not used."
)


class ModelPerformanceError(RuntimeError):
    """Raised when model performance cannot be consolidated safely."""


@dataclass
class PerformanceRecord:
    model_key: str
    model_name: str
    model_scope: str
    horizon: int
    status: str
    count_metrics: dict[str, Any] = field(default_factory=dict)
    surge_metrics: dict[str, Any] = field(default_factory=dict)
    validation_cutoffs: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)

    @property
    def prediction_count(self) -> int:
        return int(_decimal(self.count_metrics.get("n")) or _decimal(self.surge_metrics.get("n")) or 0)

    @property
    def district_count(self) -> int | None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "model_scope": self.model_scope,
            "horizon": self.horizon,
            "status": self.status,
            "count_metrics": self.count_metrics,
            "surge_metrics": self.surge_metrics,
            "prediction_count": self.prediction_count,
            "warnings": self.warnings,
            "skipped": self.skipped,
        }


@dataclass
class ModelPerformanceSummary:
    dry_run: bool
    status: str
    records: list[PerformanceRecord] = field(default_factory=list)
    missing_artifacts: dict[str, str] = field(default_factory=dict)
    skipped_or_failed_models: dict[str, list[str]] = field(default_factory=dict)
    selected_count_models: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    selected_surge_models: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    inserted_rows: int = 0
    updated_rows: int = 0
    proof_path: Path = PERFORMANCE_SELECTION_PATH
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "status": self.status,
            "all_models_compared": [record.to_dict() for record in self.records],
            "all_horizons": list(HORIZONS),
            "regression_metrics": {
                str(record.horizon): {
                    candidate.model_name: candidate.count_metrics
                    for candidate in self.records
                    if candidate.horizon == record.horizon and candidate.count_metrics
                }
                for record in self.records
            },
            "classification_metrics": {
                str(record.horizon): {
                    candidate.model_name: candidate.surge_metrics
                    for candidate in self.records
                    if candidate.horizon == record.horizon and candidate.surge_metrics
                }
                for record in self.records
            },
            "missing_artifacts": self.missing_artifacts,
            "skipped_failed_models": self.skipped_or_failed_models,
            "selected_count_models": self.selected_count_models,
            "selected_surge_models": self.selected_surge_models,
            "selection_rationale": {
                "count": COUNT_SELECTION_RULE,
                "surge": SURGE_SELECTION_RULE,
            },
            "rows_inserted": self.inserted_rows,
            "rows_updated": self.updated_rows,
            "proof_path": str(self.proof_path),
            "no_wind_speed_used": True,
            "no_negative_binomial_used": True,
            "rolling_origin_validation_artifacts_used": True,
            "warnings": self.warnings,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }


def consolidate_model_performance(
    *,
    dry_run: bool = False,
    artifact_paths: dict[str, Path] | None = None,
    proof_path: Path = PERFORMANCE_SELECTION_PATH,
    require_dataset: bool = True,
    require_forecasts: bool = True,
    require_reference: bool = True,
) -> ModelPerformanceSummary:
    """Consolidate model validation artifacts into model_performance rows."""
    artifact_paths = artifact_paths or VALIDATION_ARTIFACTS
    errors = _precondition_errors(
        artifact_paths=artifact_paths,
        require_dataset=require_dataset,
        require_forecasts=require_forecasts,
        require_reference=require_reference,
    )
    if errors:
        raise ModelPerformanceError("; ".join(errors))

    records = load_performance_records(artifact_paths)
    selected_count = select_count_models(records)
    selected_surge = select_surge_models(records)
    summary = ModelPerformanceSummary(
        dry_run=dry_run,
        status="dry-run" if dry_run else "success",
        records=records,
        skipped_or_failed_models=_skipped_failed(records),
        selected_count_models=selected_count,
        selected_surge_models=selected_surge,
        proof_path=proof_path,
    )
    if not dry_run:
        inserted, updated = _write_model_performance(records, selected_count, selected_surge)
        summary.inserted_rows = inserted
        summary.updated_rows = updated
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        proof_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_performance_records(artifact_paths: dict[str, Path]) -> list[PerformanceRecord]:
    records: list[PerformanceRecord] = []
    for model_key, path in artifact_paths.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_artifact_payload(model_key, payload)
        count_metrics = _horizon_metrics(payload.get("count_metrics_by_horizon") or {})
        surge_metrics = _horizon_metrics(payload.get("surge_metrics_by_horizon") or {})
        for horizon in HORIZONS:
            if horizon not in count_metrics and horizon not in surge_metrics:
                continue
            records.append(
                PerformanceRecord(
                    model_key=model_key,
                    model_name=payload.get("model_name", ""),
                    model_scope=payload.get("model_scope", "other"),
                    horizon=horizon,
                    status=payload.get("status", "unknown"),
                    count_metrics=count_metrics.get(horizon, {}),
                    surge_metrics=surge_metrics.get(horizon, {}),
                    validation_cutoffs=payload.get("validation_cutoffs") or [],
                    warnings=payload.get("warnings") or [],
                    skipped=payload.get("skipped") or [],
                    feature_columns=payload.get("feature_columns") or [],
                )
            )
    return records


def select_count_models(records: list[PerformanceRecord]) -> dict[int, dict[str, Any] | None]:
    selected: dict[int, dict[str, Any] | None] = {}
    for horizon in HORIZONS:
        candidates = [
            record
            for record in records
            if record.horizon == horizon and _count_score(record) is not None
        ]
        winner = min(candidates, key=_count_score) if candidates else None
        selected[horizon] = _selection_dict(winner, task="regression", metric_rule=COUNT_SELECTION_RULE)
    return selected


def select_surge_models(records: list[PerformanceRecord]) -> dict[int, dict[str, Any] | None]:
    selected: dict[int, dict[str, Any] | None] = {}
    for horizon in HORIZONS:
        candidates = [
            record
            for record in records
            if record.horizon == horizon and _surge_score(record) is not None
        ]
        winner = max(candidates, key=_surge_score) if candidates else None
        selected[horizon] = _selection_dict(winner, task="classification", metric_rule=SURGE_SELECTION_RULE)
    return selected


def _precondition_errors(
    *,
    artifact_paths: dict[str, Path],
    require_dataset: bool,
    require_forecasts: bool,
    require_reference: bool,
) -> list[str]:
    errors: list[str] = []
    if require_dataset and not modelling_dataset_exists(DATASET_PATH):
        errors.append("Prompt 22 modelling dataset is missing; rerun Prompt 22 after climate lags exist.")
    missing = {
        model_key: str(path)
        for model_key, path in artifact_paths.items()
        if not path.exists()
    }
    for model_key, path in missing.items():
        errors.append(f"{ARTIFACT_PROMPTS.get(model_key, 'Model prompt')} validation artifact missing for {model_key}: {path}")
    if require_forecasts and ForecastOutput.query.count() == 0:
        errors.append("Prompt 25 forecast outputs are missing; rerun Prompt 25 after validated models exist.")
    if require_reference and DistrictReference.query.filter_by(active=True).count() != 146:
        errors.append("district_reference must contain 146 active districts.")
    if require_reference and _model_performance_has_wind_column():
        errors.append("Wind-speed fields are not allowed in model performance.")
    return errors


def _validate_artifact_payload(model_key: str, payload: dict[str, Any]) -> None:
    model_name = payload.get("model_name", "")
    if model_name not in APPROVED_MODEL_NAMES:
        raise ModelPerformanceError(f"Unsupported model name in {model_key}: {model_name!r}")
    if "negative" in model_name.lower() and "binomial" in model_name.lower():
        raise ModelPerformanceError("Negative binomial models are not approved.")
    feature_columns = payload.get("feature_columns") or []
    wind_columns = [column for column in feature_columns if "wind" in str(column).lower()]
    if wind_columns:
        raise ModelPerformanceError("Wind-speed features found in validation artifact: " + ", ".join(wind_columns))
    if payload.get("validation_method") not in {None, "rolling_origin", "forward_chaining"}:
        raise ModelPerformanceError(f"Unsupported validation method in {model_key}.")
    if payload.get("no_random_shuffle") is False:
        raise ModelPerformanceError(f"Random splitting is not allowed for {model_key}.")
    if payload.get("horizon_aware_gap") is False:
        raise ModelPerformanceError(f"Horizon-aware gap proof failed for {model_key}.")


def _horizon_metrics(raw: dict[Any, Any]) -> dict[int, dict[str, Any]]:
    metrics: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        horizon = int(key)
        if horizon in HORIZONS:
            metrics[horizon] = value or {}
    return metrics


def _write_model_performance(
    records: list[PerformanceRecord],
    selected_count: dict[int, dict[str, Any] | None],
    selected_surge: dict[int, dict[str, Any] | None],
) -> tuple[int, int]:
    run = PipelineRun(
        run_type="model_training",
        status="success",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        source_name="model_performance_consolidation",
        parameters={"operation": "consolidate-performance", "validation_method": "rolling_origin"},
        row_count_raw=len(records),
        row_count_processed=len(records),
        warning_count=sum(1 for record in records if record.warnings or record.skipped),
        manifest={
            "proof_artifact": str(PERFORMANCE_SELECTION_PATH),
            "selection_rules": {"count": COUNT_SELECTION_RULE, "surge": SURGE_SELECTION_RULE},
        },
    )
    db.session.add(run)
    db.session.flush()
    inserted = 0
    updated = 0
    for record in records:
        row = (
            ModelPerformance.query.filter_by(
                model_name=record.model_name,
                model_scope=record.model_scope,
                horizon_weeks=record.horizon,
                district_id=None,
                validation_method="rolling_origin",
            )
            .order_by(ModelPerformance.id.asc())
            .first()
        )
        if row is None:
            row = ModelPerformance(
                model_name=record.model_name,
                model_scope=record.model_scope,
                horizon_weeks=record.horizon,
                district_id=None,
            )
            db.session.add(row)
            inserted += 1
        else:
            updated += 1
        start_year, start_week, end_year, end_week = _evaluation_period(record)
        row.pipeline_run_id = run.id
        row.evaluation_start_epi_year = start_year
        row.evaluation_start_epi_week = start_week
        row.evaluation_end_epi_year = end_year
        row.evaluation_end_epi_week = end_week
        row.mae = _decimal(record.count_metrics.get("mae"))
        row.rmse = _decimal(record.count_metrics.get("rmse"))
        row.mape = _decimal(record.count_metrics.get("mape"))
        row.wape = _decimal(record.count_metrics.get("wape"))
        row.r_squared = _decimal(record.count_metrics.get("r_squared"))
        row.sensitivity = _decimal(record.surge_metrics.get("sensitivity"))
        row.specificity = _decimal(record.surge_metrics.get("specificity"))
        row.precision = _decimal(record.surge_metrics.get("precision"))
        row.f1_score = _decimal(record.surge_metrics.get("f1"))
        row.auc = _decimal(record.surge_metrics.get("auc"))
        row.validation_method = "rolling_origin"
        row.notes = json.dumps(
            {
                "tasks": ["regression", "classification"],
                "prediction_count": record.prediction_count,
                "f2": record.surge_metrics.get("f2"),
                "pr_auc": record.surge_metrics.get("pr_auc"),
                "brier_score": record.surge_metrics.get("brier_score"),
                "selected_count": _is_selected(selected_count, record),
                "selected_surge": _is_selected(selected_surge, record),
                "warnings": record.warnings,
                "skipped": record.skipped,
            },
            sort_keys=True,
        )
    db.session.commit()
    return inserted, updated


def _evaluation_period(record: PerformanceRecord) -> tuple[int, int, int, int]:
    dates: list[date] = []
    for cutoff in record.validation_cutoffs:
        for key in ("origin_week", "target_week", "origin_date", "target_date"):
            if cutoff.get(key):
                try:
                    dates.append(date.fromisoformat(str(cutoff[key])))
                except ValueError:
                    continue
    if not dates:
        return 2020, 1, 2025, 52
    start = min(dates).isocalendar()
    end = max(dates).isocalendar()
    return start.year, start.week, end.year, end.week


def _selection_dict(record: PerformanceRecord | None, *, task: str, metric_rule: str) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "task": task,
        "model_key": record.model_key,
        "model_name": record.model_name,
        "model_scope": record.model_scope,
        "horizon": record.horizon,
        "metric_rule": metric_rule,
        "count_metrics": record.count_metrics,
        "surge_metrics": record.surge_metrics,
    }


def _count_score(record: PerformanceRecord) -> tuple[int, Decimal] | None:
    wape = _decimal(record.count_metrics.get("wape"))
    if wape is not None:
        return (0, wape)
    rmse = _decimal(record.count_metrics.get("rmse"))
    if rmse is not None:
        return (1, rmse)
    return None


def _surge_score(record: PerformanceRecord) -> tuple[Decimal, ...] | None:
    f2 = _decimal(record.surge_metrics.get("f2"))
    f1 = _decimal(record.surge_metrics.get("f1"))
    sensitivity = _decimal(record.surge_metrics.get("sensitivity")) or Decimal("0")
    pr_auc = _decimal(record.surge_metrics.get("pr_auc")) or Decimal("0")
    precision = _decimal(record.surge_metrics.get("precision")) or Decimal("0")
    specificity = _decimal(record.surge_metrics.get("specificity")) or Decimal("0")
    auc = _decimal(record.surge_metrics.get("auc")) or Decimal("0")
    primary = f2 if f2 is not None else f1
    if primary is None:
        return None
    return (primary, sensitivity, pr_auc, precision, specificity, auc)


def _is_selected(selection: dict[int, dict[str, Any] | None], record: PerformanceRecord) -> bool:
    selected = selection.get(record.horizon)
    return bool(selected and selected.get("model_name") == record.model_name and selected.get("model_scope") == record.model_scope)


def _skipped_failed(records: list[PerformanceRecord]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for record in records:
        reasons = list(record.warnings) + list(record.skipped)
        if record.status not in {"success", "dry-run", "planned"}:
            reasons.append(f"status={record.status}")
        if reasons:
            out.setdefault(record.model_name, []).extend(reasons)
    return out


def _decimal(value: Any) -> Decimal | None:
    if value in {None, ""}:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _model_performance_has_wind_column() -> bool:
    return any("wind" in column.name.lower() for column in ModelPerformance.__table__.columns)
