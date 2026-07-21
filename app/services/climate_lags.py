"""Climate lag feature creation for district_week_features."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db
from app.models import DistrictWeekClimate, DistrictWeekFeature, PipelineRun

APPROVED_LAGS = (4, 8, 12)
APPROVED_VARIABLES = ("rainfall", "temperature", "humidity")
CURRENT_FIELD_BY_VARIABLE = {
    "rainfall": "rainfall_current",
    "temperature": "temperature_current",
    "humidity": "humidity_current",
}
LAG_FIELD_BY_VARIABLE = {
    variable: {lag: f"{variable}_lag_{lag}" for lag in APPROVED_LAGS}
    for variable in APPROVED_VARIABLES
}


class ClimateLagError(RuntimeError):
    """Raised when climate lag creation cannot safely continue."""


@dataclass
class LagFieldSummary:
    field_name: str
    non_null_count: int = 0
    null_count: int = 0
    planned_updates: int = 0


@dataclass
class LagAlignmentCheck:
    district_id: str
    target_epi_year: int
    target_epi_week: int
    target_week_start_date: str
    lag_weeks: int
    source_epi_year: int | None
    source_epi_week: int | None
    source_week_start_date: str
    source_value: str | None
    lag_value_on_target: str | None
    field_name: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ClimateLagSummary:
    dry_run: bool
    start_year: int
    end_year: int
    variables: tuple[str, ...]
    lags: tuple[int, ...]
    pipeline_run_id: int | None = None
    feature_rows_considered: int = 0
    climate_rows_available: int = 0
    rows_with_updates: int = 0
    field_summaries: dict[str, LagFieldSummary] = field(default_factory=dict)
    alignment_checks: list[LagAlignmentCheck] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def manifest(self) -> dict[str, Any]:
        return {
            "source_table": "district_week_climate",
            "target_table": "district_week_features",
            "variables": list(self.variables),
            "lags": list(self.lags),
            "start_year": self.start_year,
            "end_year": self.end_year,
            "week_convention": "ISO-8601 Monday-start using week_start_date minus N*7 days",
            "feature_rows_considered": self.feature_rows_considered,
            "climate_rows_available": self.climate_rows_available,
            "rows_with_updates": self.rows_with_updates,
            "field_summaries": {
                field: {
                    "non_null_count": item.non_null_count,
                    "null_count": item.null_count,
                    "planned_updates": item.planned_updates,
                }
                for field, item in self.field_summaries.items()
            },
            "alignment_checks": [check.to_dict() for check in self.alignment_checks],
            "warnings": self.warnings,
        }


def create_climate_lags(
    *,
    start_year: int,
    end_year: int,
    variables: tuple[str, ...] = APPROVED_VARIABLES,
    lags: tuple[int, ...] = APPROVED_LAGS,
    dry_run: bool = False,
    replace_existing: bool = False,
) -> ClimateLagSummary:
    """Create lagged climate predictors on existing district_week_features rows."""
    if start_year > end_year:
        raise ClimateLagError("start_year must be <= end_year.")
    variables = _validate_variables(variables)
    lags = _validate_lags(lags)
    fields = [LAG_FIELD_BY_VARIABLE[variable][lag] for variable in variables for lag in lags]
    summary = ClimateLagSummary(
        dry_run=dry_run,
        start_year=start_year,
        end_year=end_year,
        variables=variables,
        lags=lags,
        field_summaries={field: LagFieldSummary(field_name=field) for field in fields},
    )

    features = (
        DistrictWeekFeature.query.filter(
            DistrictWeekFeature.epi_year >= start_year,
            DistrictWeekFeature.epi_year <= end_year,
        )
        .order_by(
            DistrictWeekFeature.district_id,
            DistrictWeekFeature.week_start_date,
            DistrictWeekFeature.epi_year,
            DistrictWeekFeature.epi_week,
        )
        .all()
    )
    summary.feature_rows_considered = len(features)
    if not features:
        raise ClimateLagError("No district_week_features rows found for selected period.")

    min_start = min(feature.week_start_date for feature in features if feature.week_start_date)
    max_end = max(feature.week_end_date for feature in features if feature.week_end_date)
    if min_start is None or max_end is None:
        raise ClimateLagError("Feature rows must have week_start_date/week_end_date.")
    lag_start = min_start - timedelta(weeks=max(lags))

    climate_rows = (
        DistrictWeekClimate.query.filter(
            DistrictWeekClimate.week_start_date >= lag_start,
            DistrictWeekClimate.week_start_date <= max_end,
        )
        .order_by(DistrictWeekClimate.district_id, DistrictWeekClimate.week_start_date)
        .all()
    )
    summary.climate_rows_available = len(climate_rows)
    if not climate_rows:
        raise ClimateLagError(
            "district_week_climate is empty for the required lag window. Run Prompt 20 first."
        )

    climate_by_key = {
        (row.district_id, row.week_start_date): row for row in climate_rows
    }
    run: PipelineRun | None = None
    if not dry_run:
        run = PipelineRun(
            run_type="climate_lag_creation",
            status="running",
            started_at=_utcnow(),
            source_name="district_week_climate",
            parameters={
                "start_year": start_year,
                "end_year": end_year,
                "variables": list(variables),
                "lags": list(lags),
                "replace_existing": replace_existing,
                "approved_fields": fields,
            },
        )
        db.session.add(run)
        db.session.commit()
        summary.pipeline_run_id = run.id

    try:
        for feature in features:
            row_changed = False
            for variable in variables:
                current_field = CURRENT_FIELD_BY_VARIABLE[variable]
                for lag in lags:
                    lag_field = LAG_FIELD_BY_VARIABLE[variable][lag]
                    source_start = feature.week_start_date - timedelta(weeks=lag)
                    source = climate_by_key.get((feature.district_id, source_start))
                    value = getattr(source, current_field) if source is not None else None
                    field_summary = summary.field_summaries[lag_field]
                    if value is None:
                        field_summary.null_count += 1
                        continue
                    field_summary.non_null_count += 1
                    existing_value = getattr(feature, lag_field)
                    if existing_value is not None and not replace_existing:
                        continue
                    field_summary.planned_updates += 1
                    row_changed = True
                    if not dry_run:
                        setattr(feature, lag_field, value)
            if row_changed:
                summary.rows_with_updates += 1

        summary.alignment_checks = _build_alignment_checks(
            features=features,
            climate_by_key=climate_by_key,
            variables=variables,
            lags=lags,
            dry_run=dry_run,
        )
        if not dry_run:
            run.status = "success"
            run.completed_at = _utcnow()
            run.row_count_raw = summary.climate_rows_available
            run.row_count_processed = summary.rows_with_updates
            run.warning_count = len(summary.warnings)
            run.manifest = summary.manifest()
            db.session.commit()
        return summary
    except Exception as exc:
        db.session.rollback()
        if run is not None:
            run = db.session.get(PipelineRun, run.id)
            if run is not None:
                run.status = "failed"
                run.completed_at = _utcnow()
                run.error_message = str(exc)[:1000]
                run.manifest = summary.manifest()
                db.session.commit()
        if isinstance(exc, ClimateLagError):
            raise
        if isinstance(exc, SQLAlchemyError):
            raise ClimateLagError("Database update failed during climate lag creation.") from exc
        raise


def _build_alignment_checks(
    *,
    features: list[DistrictWeekFeature],
    climate_by_key: dict[tuple[str, Any], DistrictWeekClimate],
    variables: tuple[str, ...],
    lags: tuple[int, ...],
    dry_run: bool,
) -> list[LagAlignmentCheck]:
    checks: list[LagAlignmentCheck] = []
    sample_specs = [
        ("ordinary within-year week", lambda row: row.epi_week >= 20, min(lags)),
        (
            "year-boundary proof: 2021W02 lag_4 crosses into 2020",
            lambda row: row.epi_year == 2021 and row.epi_week == 2 and 4 in lags,
            4,
        ),
        (
            "week-53 proof: 2021W12 lag_12 resolves to 2020W53",
            lambda row: row.epi_year == 2021 and row.epi_week == 12 and 12 in lags,
            12,
        ),
    ]
    variable = variables[0]
    for note, predicate, lag in sample_specs:
        if lag not in lags:
            continue
        target = next((row for row in features if predicate(row)), None)
        if target is None:
            continue
        source_start = target.week_start_date - timedelta(weeks=lag)
        source = climate_by_key.get((target.district_id, source_start))
        source_iso = source_start.isocalendar()
        field_name = LAG_FIELD_BY_VARIABLE[variable][lag]
        current_field = CURRENT_FIELD_BY_VARIABLE[variable]
        source_value = getattr(source, current_field) if source is not None else None
        lag_value = source_value if dry_run else getattr(target, field_name)
        checks.append(
            LagAlignmentCheck(
                district_id=target.district_id,
                target_epi_year=target.epi_year,
                target_epi_week=target.epi_week,
                target_week_start_date=target.week_start_date.isoformat(),
                lag_weeks=lag,
                source_epi_year=source_iso.year,
                source_epi_week=source_iso.week,
                source_week_start_date=source_start.isoformat(),
                source_value=_decimal_text(source_value),
                lag_value_on_target=_decimal_text(lag_value),
                field_name=field_name,
                note=note,
            )
        )
    return checks


def _validate_variables(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        cleaned = value.strip().lower()
        if "wind" in cleaned:
            raise ClimateLagError("Wind speed is not an approved climate lag variable.")
        if cleaned not in APPROVED_VARIABLES:
            raise ClimateLagError(
                f"Unsupported climate lag variable {value!r}. Approved variables: "
                + ", ".join(APPROVED_VARIABLES)
            )
        normalized.append(cleaned)
    return tuple(dict.fromkeys(normalized))


def _validate_lags(values: tuple[int, ...]) -> tuple[int, ...]:
    cleaned: list[int] = []
    for value in values:
        if value not in APPROVED_LAGS:
            raise ClimateLagError(
                f"Unsupported climate lag {value}. Approved lags: "
                + ", ".join(str(item) for item in APPROVED_LAGS)
            )
        cleaned.append(value)
    return tuple(dict.fromkeys(cleaned))


def _decimal_text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
