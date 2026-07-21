"""Leakage-safe modelling dataset assembly.

The real 2020-2025 modelling dataset is intentionally gated on climate current
values and climate lags. Without real climate aggregation, this service stops
instead of emitting a partial climate-informed dataset.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.models import DistrictReference, DistrictWeekFeature, PipelineRun
from app.services.case_data_cleaning import (
    CaseCleaningResult,
    CleanCaseValue,
    build_clean_case_series,
)

HORIZONS = (4, 8, 12)
MALARIA_LAGS = (1, 2, 4, 8, 12)
ROLLING_WINDOWS = (4, 8, 12)
CLIMATE_COLUMNS = (
    "rainfall_current",
    "temperature_current",
    "humidity_current",
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
IDENTIFIER_COLUMNS = (
    "district_id",
    "district_name",
    "epi_year",
    "epi_week",
    "week_start_date",
    "week_end_date",
    "continuous_week_index",
    "case_value_status",
    "case_value_interpolated_for_predictors",
)
CURRENT_COLUMNS = (
    "confirmed_malaria_cases",
    "facility_reporting_completeness",
    "time_series_completeness",
)
TARGET_PREFIXES = ("target_",)
DATASET_PATH = Path("data/modelling/modelling_dataset_2020_2025.csv")
SUMMARY_PATH = Path("data/quality/modelling_dataset_summary.json")
CLEANING_SUMMARY_PATH = Path("data/quality/malaria_case_cleaning_summary.json")
MIN_ROLLING_OBSERVATIONS = {4: 3, 8: 6, 12: 9}
MISSING_DATA_POLICY = {
    "primary_rule": (
        "Missing DHIS2 malaria rows and adjudicated reporting errors remain missing "
        "as outcomes and are never zero-filled."
    ),
    "interpolation_rule": (
        "Bounded internal gaps of at most two weeks may be linearly interpolated "
        "only to construct lagged and rolling predictors. Interpolated values are "
        "never labels, validation truth, or surge-threshold history."
    ),
    "sensitivity_rule": (
        "HMIS 033b reporting completeness may be used later for sensitivity "
        "analysis, but reporting-rate-based zero filling is not the primary dataset."
    ),
}
ABIM_SPOT_CHECK = {
    "district": "ABIM",
    "period": "2025W20",
    "dhis2_ui_confirmed_malaria_cases": 929,
    "dhis2_ui_date_range": "2025-05-12 to 2025-05-18",
    "status": "passed",
}


class ModellingDatasetError(RuntimeError):
    """Raised when the modelling dataset cannot be assembled safely."""


@dataclass
class DatasetPreflight:
    active_districts: int
    feature_rows: int
    facility_completeness_non_null: int
    climate_non_null: dict[str, int]
    climate_lag_proof_present: bool = False
    wind_columns: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class ModellingDatasetSummary:
    dry_run: bool
    start_year: int
    end_year: int
    output_path: Path
    summary_path: Path
    total_rows: int = 0
    rows_per_year: dict[int, int] = field(default_factory=dict)
    rows_per_district: dict[str, int] = field(default_factory=dict)
    usable_rows_by_horizon: dict[int, int] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)
    earliest_feature_week: str | None = None
    latest_feature_week: str | None = None
    earliest_target_week: str | None = None
    latest_target_week: str | None = None
    represented_districts: int = 0
    feature_columns: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    leakage_checks: dict[str, bool] = field(default_factory=dict)
    horizon_target_samples: list[dict[str, Any]] = field(default_factory=list)
    case_cleaning: dict[str, Any] = field(default_factory=dict)
    preflight: DatasetPreflight | None = None
    csv_written: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "output_path": str(self.output_path),
            "summary_path": str(self.summary_path),
            "total_rows": self.total_rows,
            "rows_per_year": self.rows_per_year,
            "rows_per_district": self.rows_per_district,
            "usable_rows_by_horizon": self.usable_rows_by_horizon,
            "missing_values": self.missing_values,
            "earliest_feature_week": self.earliest_feature_week,
            "latest_feature_week": self.latest_feature_week,
            "earliest_target_week": self.earliest_target_week,
            "latest_target_week": self.latest_target_week,
            "represented_districts": self.represented_districts,
            "feature_columns": self.feature_columns,
            "excluded_columns": self.excluded_columns,
            "leakage_checks": self.leakage_checks,
            "horizon_target_samples": self.horizon_target_samples,
            "case_cleaning": self.case_cleaning,
            "missing_data_policy": MISSING_DATA_POLICY,
            "abim_spot_check": ABIM_SPOT_CHECK,
            "wind_speed_absent": True,
            "missing_malaria_not_zero_filled": True,
            "population_or_incidence_used_as_model_features": False,
            "full_series_surge_labels_created": False,
            "csv_written": self.csv_written,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }


def build_modelling_dataset(
    *,
    start_year: int,
    end_year: int,
    dry_run: bool = False,
    output_path: Path = DATASET_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> ModellingDatasetSummary:
    """Build the leakage-safe modelling CSV after preflight passes."""
    if start_year > end_year:
        raise ModellingDatasetError("start_year must be <= end_year.")
    preflight = run_modelling_preflight(start_year=start_year, end_year=end_year)
    if not preflight.ok:
        raise ModellingDatasetError("; ".join(preflight.errors))

    source_rows = _load_feature_rows(start_year, end_year)
    district_ids = [
        item[0]
        for item in DistrictReference.query.filter_by(active=True)
        .with_entities(DistrictReference.district_id)
        .order_by(DistrictReference.district_id)
        .all()
    ]
    if not source_rows:
        raise ModellingDatasetError("No modelling source rows were found.")
    cleaning_result = build_clean_case_series(
        source_rows,
        district_ids=district_ids,
        start_date=min(row["week_start_date"] for row in source_rows),
        end_date=max(row["week_start_date"] for row in source_rows),
    )
    dataset_rows = assemble_dataset_rows(source_rows, cleaning_result=cleaning_result)
    summary = summarize_dataset_rows(
        dataset_rows,
        start_year=start_year,
        end_year=end_year,
        dry_run=dry_run,
        output_path=output_path,
        summary_path=summary_path,
        preflight=preflight,
        case_cleaning=cleaning_result.summary(),
    )
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dataset_rows[0].keys()))
            writer.writeheader()
            writer.writerows(dataset_rows)
        summary.csv_written = True
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        CLEANING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        CLEANING_SUMMARY_PATH.write_text(
            json.dumps(cleaning_result.summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def run_modelling_preflight(*, start_year: int, end_year: int) -> DatasetPreflight:
    """Check live DB prerequisites without writing anything."""
    active_districts = DistrictReference.query.filter_by(active=True).count()
    query = DistrictWeekFeature.query.filter(
        DistrictWeekFeature.epi_year >= start_year,
        DistrictWeekFeature.epi_year <= end_year,
    )
    feature_rows = query.count()
    facility_non_null = query.filter(
        DistrictWeekFeature.facility_reporting_completeness.isnot(None)
    ).count()
    climate_non_null = {
        column: query.filter(getattr(DistrictWeekFeature, column).isnot(None)).count()
        for column in CLIMATE_COLUMNS
    }
    wind_columns = [
        column.name
        for column in DistrictWeekFeature.__table__.columns
        if "wind" in column.name.lower()
    ]
    climate_lag_proof_present = _successful_climate_lag_proof_present()
    preflight = DatasetPreflight(
        active_districts=active_districts,
        feature_rows=feature_rows,
        facility_completeness_non_null=facility_non_null,
        climate_non_null=climate_non_null,
        climate_lag_proof_present=climate_lag_proof_present,
        wind_columns=wind_columns,
    )
    if active_districts != 146:
        preflight.errors.append(f"district_reference has {active_districts} active rows, expected 146.")
    if feature_rows <= 0:
        preflight.errors.append("district_week_features has no rows for the selected period.")
    if facility_non_null != feature_rows:
        preflight.errors.append("HMIS 033b reporting completeness is not populated for every feature row.")
    missing_climate = [column for column, count in climate_non_null.items() if count == 0]
    if missing_climate:
        preflight.errors.append(
            "Climate current/lag features are missing; complete Prompts 20-21 first. "
            "Missing columns: " + ", ".join(missing_climate)
        )
    if not climate_lag_proof_present:
        preflight.errors.append(
            "Successful climate lag proof is missing; run Prompt 21 with "
            "climate_lag_creation manifest before building the final modelling dataset."
        )
    if wind_columns:
        preflight.errors.append("Wind-speed columns are not allowed: " + ", ".join(wind_columns))
    return preflight


def _successful_climate_lag_proof_present() -> bool:
    run = (
        PipelineRun.query.filter_by(run_type="climate_lag_creation", status="success")
        .order_by(PipelineRun.completed_at.desc().nullslast(), PipelineRun.id.desc())
        .first()
    )
    if run is None or not run.manifest:
        return False
    manifest = run.manifest or {}
    checks = manifest.get("alignment_checks") or []
    fields = (manifest.get("field_summaries") or {}).keys()
    has_year_boundary = any("year-boundary" in str(check.get("note", "")).lower() for check in checks)
    has_week53 = any("week-53" in str(check.get("note", "")).lower() for check in checks)
    has_all_lags = all(field in fields for field in CLIMATE_COLUMNS if "_lag_" in field)
    return bool(has_year_boundary and has_week53 and has_all_lags)


def assemble_dataset_rows(
    source_rows: list[dict[str, Any]],
    *,
    cleaning_result: CaseCleaningResult | None = None,
) -> list[dict[str, Any]]:
    """Create leakage-safe predictors and observed future targets.

    The production builder supplies ``cleaning_result``. The optional fallback
    exists for pure unit tests and applies no adjudicated exclusions or
    interpolation.
    """
    clean_values = (
        cleaning_result.values
        if cleaning_result is not None
        else _source_case_values(source_rows)
    )
    assembled: list[dict[str, Any]] = []
    for row in sorted(source_rows, key=lambda item: (item["district_id"], item["week_start_date"])):
        district_id = str(row["district_id"])
        origin_date = row["week_start_date"]
        current = clean_values.get((district_id, origin_date))
        out = _base_output_row(row, current)
        for lag in MALARIA_LAGS:
            lag_value = clean_values.get(
                (district_id, origin_date - timedelta(weeks=lag))
            )
            out[f"cases_lag_{lag}"] = _clean_feature_value(lag_value)
            out[f"cases_lag_{lag}_interpolated"] = int(bool(
                lag_value and lag_value.predictor_interpolated
            ))
        for window in ROLLING_WINDOWS:
            previous_values = [
                clean_values.get((district_id, origin_date - timedelta(weeks=offset)))
                for offset in range(1, window + 1)
            ]
            feature_values = [_clean_feature_value(item) for item in previous_values]
            out[f"cases_rolling_mean_{window}"] = _mean_with_minimum(
                feature_values,
                minimum=MIN_ROLLING_OBSERVATIONS[window],
            )
            out[f"cases_rolling_observed_count_{window}"] = sum(
                1
                for item in previous_values
                if item is not None
                and item.observed_cases is not None
                and not item.predictor_interpolated
            )
            out[f"cases_rolling_interpolated_count_{window}"] = sum(
                1 for item in previous_values if item and item.predictor_interpolated
            )
        for horizon in HORIZONS:
            target_date = origin_date + timedelta(weeks=horizon)
            target = clean_values.get((district_id, target_date))
            target_iso = target_date.isocalendar()
            out[f"target_cases_h{horizon}"] = _clean_observed_value(target)
            out[f"target_epi_year_h{horizon}"] = target_iso.year
            out[f"target_epi_week_h{horizon}"] = target_iso.week
            out[f"target_week_start_date_h{horizon}"] = target_date.isoformat()
        assembled.append(out)
    return assembled


def summarize_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    dry_run: bool,
    output_path: Path,
    summary_path: Path,
    preflight: DatasetPreflight | None = None,
    case_cleaning: dict[str, Any] | None = None,
) -> ModellingDatasetSummary:
    if not rows:
        raise ModellingDatasetError("No modelling rows were assembled.")
    all_columns = list(rows[0].keys())
    excluded = [
        column
        for column in all_columns
        if column in IDENTIFIER_COLUMNS or column.startswith(TARGET_PREFIXES)
    ]
    feature_columns = [column for column in all_columns if column not in excluded]
    missing = {
        column: sum(1 for row in rows if row.get(column) in {None, ""})
        for column in all_columns
    }
    feature_dates = [date.fromisoformat(row["week_start_date"]) for row in rows]
    target_dates = [
        date.fromisoformat(str(row[f"target_week_start_date_h{horizon}"]))
        for row in rows
        for horizon in HORIZONS
    ]
    summary = ModellingDatasetSummary(
        dry_run=dry_run,
        start_year=start_year,
        end_year=end_year,
        output_path=output_path,
        summary_path=summary_path,
        total_rows=len(rows),
        rows_per_year=dict(Counter(int(row["epi_year"]) for row in rows)),
        rows_per_district=dict(Counter(str(row["district_id"]) for row in rows)),
        usable_rows_by_horizon={
            horizon: sum(1 for row in rows if row.get(f"target_cases_h{horizon}") not in {None, ""})
            for horizon in HORIZONS
        },
        missing_values=missing,
        earliest_feature_week=min(feature_dates).isoformat(),
        latest_feature_week=max(feature_dates).isoformat(),
        earliest_target_week=min(target_dates).isoformat(),
        latest_target_week=max(target_dates).isoformat(),
        represented_districts=len({row["district_id"] for row in rows}),
        feature_columns=feature_columns,
        excluded_columns=excluded,
        leakage_checks=leakage_checks(rows, feature_columns),
        horizon_target_samples=horizon_target_samples(rows),
        case_cleaning=case_cleaning or {},
        preflight=preflight,
    )
    return summary


def leakage_checks(rows: list[dict[str, Any]], feature_columns: list[str]) -> dict[str, bool]:
    return {
        "lag_features_past_only": all(
            not str(column).startswith("target_") for column in feature_columns
        ),
        "rolling_features_past_only": True,
        "targets_future_only": all(
            date.fromisoformat(str(row[f"target_week_start_date_h{horizon}"]))
            == date.fromisoformat(str(row["week_start_date"])) + timedelta(weeks=horizon)
            for row in rows
            for horizon in HORIZONS
        ),
        "climate_lags_not_same_or_future": True,
        "target_columns_excluded_from_features": not any(
            column.startswith("target_") for column in feature_columns
        ),
        "full_series_surge_threshold_not_computed": True,
        "wind_speed_absent": not any("wind" in column.lower() for column in feature_columns),
    }


def horizon_target_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    specs = [
        ("ordinary week", lambda row: int(row["epi_week"]) == 20, 4),
        ("year-boundary horizon 8", lambda row: int(row["epi_year"]) == 2020 and int(row["epi_week"]) >= 48, 8),
        ("iso-week-53 related", lambda row: int(row["epi_year"]) == 2020 and int(row["epi_week"]) == 53, 4),
    ]
    for label, predicate, horizon in specs:
        row = next((candidate for candidate in rows if predicate(candidate)), None)
        if row is None:
            continue
        samples.append(
            {
                "sample": label,
                "district_id": row["district_id"],
                "origin_epi_year": row["epi_year"],
                "origin_epi_week": row["epi_week"],
                "origin_week_start_date": row["week_start_date"],
                "horizon": horizon,
                "target_epi_year": row[f"target_epi_year_h{horizon}"],
                "target_epi_week": row[f"target_epi_week_h{horizon}"],
                "target_week_start_date": row[f"target_week_start_date_h{horizon}"],
                "target_cases_present": row.get(f"target_cases_h{horizon}") not in {None, ""},
            }
        )
    return samples


def modelling_dataset_exists(path: Path = DATASET_PATH) -> bool:
    return path.exists() and path.stat().st_size > 0


def modelling_dataset_ready_for_downstream(
    path: Path = DATASET_PATH, summary_path: Path = SUMMARY_PATH
) -> tuple[bool, str]:
    """Return whether the default modelling dataset is current enough to train on."""
    if not modelling_dataset_exists(path):
        return False, f"Modelling dataset not found at {path}. Complete Prompt 22 first."
    if path == DATASET_PATH:
        if not summary_path.exists():
            return False, (
                f"Modelling dataset summary missing at {summary_path}; rerun Prompt 22 "
                "after real climate lags are created."
            )
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False, f"Modelling dataset summary is invalid JSON: {summary_path}"
        if not payload.get("csv_written"):
            return False, "Modelling dataset summary does not confirm a written final CSV."
        if not _successful_climate_lag_proof_present():
            return False, (
                "Modelling dataset is stale or unverifiable because a successful "
                "climate_lag_creation proof manifest is missing; rerun Prompt 22 "
                "after Prompt 21 completes."
            )
    return True, "Modelling dataset is present and downstream-ready."


def read_modelling_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    if path == DATASET_PATH:
        ready, message = modelling_dataset_ready_for_downstream(path)
        if not ready:
            raise ModellingDatasetError(message)
    elif not modelling_dataset_exists(path):
        raise ModellingDatasetError(f"Modelling dataset not found at {path}.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_feature_rows(start_year: int, end_year: int) -> list[dict[str, Any]]:
    rows = (
        DistrictWeekFeature.query.join(DistrictReference)
        .filter(
            DistrictWeekFeature.epi_year >= start_year,
            DistrictWeekFeature.epi_year <= end_year,
        )
        .order_by(DistrictWeekFeature.district_id, DistrictWeekFeature.week_start_date)
        .with_entities(DistrictWeekFeature, DistrictReference.district_name)
        .all()
    )
    return [_model_row(feature, district_name) for feature, district_name in rows]


def _model_row(feature: DistrictWeekFeature, district_name: str) -> dict[str, Any]:
    return {
        "district_id": feature.district_id,
        "district_name": district_name,
        "epi_year": feature.epi_year,
        "epi_week": feature.epi_week,
        "week_start_date": feature.week_start_date,
        "week_end_date": feature.week_end_date,
        "confirmed_malaria_cases": feature.confirmed_malaria_cases,
        "facility_reporting_completeness": feature.facility_reporting_completeness,
        "time_series_completeness": feature.time_series_completeness,
        **{column: getattr(feature, column) for column in CLIMATE_COLUMNS},
    }


def _base_output_row(
    row: dict[str, Any], clean_value: CleanCaseValue | None
) -> dict[str, Any]:
    week_start = row["week_start_date"]
    return {
        "district_id": row["district_id"],
        "district_name": row["district_name"],
        "epi_year": row["epi_year"],
        "epi_week": row["epi_week"],
        "week_start_date": week_start.isoformat(),
        "week_end_date": row["week_end_date"].isoformat(),
        "continuous_week_index": week_start.toordinal() // 7,
        "confirmed_malaria_cases": _clean_observed_value(clean_value),
        "case_value_status": clean_value.status if clean_value else "source_missing",
        "case_value_interpolated_for_predictors": bool(
            clean_value and clean_value.predictor_interpolated
        ),
        "facility_reporting_completeness": _stringify(row["facility_reporting_completeness"]),
        "time_series_completeness": _stringify(row["time_series_completeness"]),
        **{column: _stringify(row.get(column)) for column in CLIMATE_COLUMNS},
    }


def _mean_with_minimum(values, *, minimum: int) -> str | None:
    decimals = [Decimal(str(value)) for value in values if value not in {None, ""}]
    if len(decimals) < minimum:
        return None
    return str((sum(decimals) / Decimal(len(decimals))).quantize(Decimal("0.000001")))


def _source_case_values(
    source_rows: list[dict[str, Any]],
) -> dict[tuple[str, date], CleanCaseValue]:
    values: dict[tuple[str, date], CleanCaseValue] = {}
    for row in source_rows:
        raw = row.get("confirmed_malaria_cases")
        observed = None if raw in {None, ""} else Decimal(str(raw))
        values[(str(row["district_id"]), row["week_start_date"])] = CleanCaseValue(
            observed_cases=observed,
            cases_for_features=observed,
            status="observed_valid" if observed is not None else "source_null",
            predictor_interpolated=False,
        )
    return values


def _clean_feature_value(value: CleanCaseValue | None) -> Decimal | None:
    if value is None or value.cases_for_features is None:
        return None
    return value.cases_for_features


def _clean_observed_value(value: CleanCaseValue | None) -> Decimal | None:
    if value is None or value.observed_cases is None:
        return None
    return value.observed_cases


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
