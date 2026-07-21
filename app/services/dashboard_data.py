"""Dashboard-ready backend data services.

The service converts already validated model outputs into stable JSON-shaped
payloads for the dashboard. It does not create forecasts, train models, or fill
missing surveillance values with zeros.

Two selected models are involved at every horizon and they are never merged:

* the selected **count** model supplies predicted cases and the weekly chart;
* the selected **surge** model supplies the risk assessment, the threshold
  comparison and the map colouring.

At h4 and h12 those are different models (XGBoost vs Random Forest), so both
names travel with the data and are surfaced in the interface.

No trained surge probability exists in the current production outputs, so risk
is a transparent count-threshold assessment: predicted cases are compared with
a district threshold derived only from observed history through the issue week.
``predicted_surge_score`` is that difference in cases and is never a probability.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from sqlalchemy import distinct, func

from app.extensions import db
from app.models import DistrictReference, DistrictWeekFeature, ForecastOutput, ModelPerformance
from app.services.model_performance import PERFORMANCE_SELECTION_PATH
from app.services.risk_classification import AT_RISK_LEVELS, NO_DATA_RISK, classify_forecast_risk

# Approved, unmodified boundary input (read-only for this module).
PREPARED_GEOJSON_PATH = Path("data/boundaries/uganda_districts_prepared.geojson")
# Derived, simplified artifact built from the approved boundary for browser use.
WEB_MAP_GEOJSON_PATH = Path("data/boundaries/uganda_districts_webmap.geojson")
WEB_MAP_PROOF_PATH = Path("data/quality/dashboard_web_map_summary.json")

DASHBOARD_PROOF_PATH = Path("data/quality/dashboard_data_summary.json")
DASHBOARD_DIR = Path("data/dashboard")
SUMMARY_OUTPUT_PATH = DASHBOARD_DIR / "dashboard_summary_latest.json"
RISK_TABLE_OUTPUT_PATH = DASHBOARD_DIR / "risk_table_latest.json"
MAP_RISK_OUTPUT_PATH = DASHBOARD_DIR / "map_risk_latest.geojson"

# Simplification chosen so the browser payload stays far below the 2 MB ceiling
# while keeping all 146 features valid. 0.001 degrees is roughly 110 m, which is
# sub-pixel at national zoom.
WEB_MAP_SIMPLIFY_TOLERANCE_DEGREES = 0.001
WEB_MAP_COORDINATE_DECIMALS = 4
WEB_MAP_MAX_BYTES = 2_000_000
EXPECTED_DISTRICTS = 146

HORIZONS = (4, 8, 12)
DEFAULT_HORIZON = 4
ALL_DISTRICTS = "all"

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

CLIMATE_VARIABLES = (
    {"variable": "rainfall", "label": "Rainfall", "source": "CHIRPS", "unit": "mm", "unit_label": "millimetres"},
    {"variable": "temperature", "label": "Temperature", "source": "ERA5-Land", "unit": "°C", "unit_label": "degrees Celsius"},
    {"variable": "humidity", "label": "Humidity", "source": "ERA5-Land (derived from temperature and dewpoint)", "unit": "%", "unit_label": "percent"},
)

RISK_ASSESSMENT_LABEL = "Count-threshold risk assessment"
RISK_ASSESSMENT_NOTE = (
    "Risk compares predicted cases with a district threshold derived only from "
    "observed history up to the issue week. It is not a trained surge probability "
    "and a high risk level does not confirm an outbreak."
)
THRESHOLD_METHOD_NOTE = (
    "District seasonal 75th percentile from valid training-only history in the "
    "target epi week +/-2; 2022 is excluded only from the normal-reference baseline."
)
NATIONAL_AGGREGATION_NOTE = "National view sums district values; it is not a national re-estimate."
NATIONAL_THRESHOLD_NOTE = (
    "In the Uganda view, the forecast and benchmark are sums across districts. "
    "Each district still uses its own seasonal benchmark."
)


class DashboardDataError(RuntimeError):
    """Raised when dashboard data cannot be built honestly."""


class DashboardParameterError(ValueError):
    """Raised when a caller supplies an invalid dashboard query parameter."""


@dataclass(frozen=True)
class DashboardParams:
    """Validated dashboard request parameters."""

    horizon: int = DEFAULT_HORIZON
    district_id: str | None = None
    issue_year: int | None = None
    issue_week: int | None = None

    @property
    def is_national(self) -> bool:
        return self.district_id is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "district_id": self.district_id,
            "scope": "national" if self.is_national else "district",
            "issue_year": self.issue_year,
            "issue_week": self.issue_week,
            "issue_label": issue_label(self.issue_year, self.issue_week),
        }


@dataclass
class DashboardBuildSummary:
    dry_run: bool
    status: str
    horizon: int
    risk_counts: dict[str, int] = field(default_factory=dict)
    no_data_districts: list[str] = field(default_factory=list)
    proof_path: Path = DASHBOARD_PROOF_PATH
    generated_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "status": self.status,
            "horizon": self.horizon,
            "risk_counts": self.risk_counts,
            "no_data_districts": self.no_data_districts,
            "proof_path": str(self.proof_path),
            "generated_paths": [str(path) for path in self.generated_paths],
            "warnings": self.warnings,
            "no_wind_speed_used": True,
            "missing_values_not_zero_filled": True,
            "map_join_key": "district_id",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
def issue_label(year: int | None, week: int | None) -> str | None:
    if year is None or week is None:
        return None
    return f"{year}W{int(week):02d}"


def available_issues() -> list[dict[str, Any]]:
    """Return the production forecast issues that actually exist, newest first."""
    rows = (
        db.session.query(
            ForecastOutput.forecast_origin_epi_year,
            ForecastOutput.forecast_origin_epi_week,
            func.min(ForecastOutput.forecast_origin_date),
        )
        .group_by(ForecastOutput.forecast_origin_epi_year, ForecastOutput.forecast_origin_epi_week)
        .order_by(
            ForecastOutput.forecast_origin_epi_year.desc(),
            ForecastOutput.forecast_origin_epi_week.desc(),
        )
        .all()
    )
    return [
        {
            "issue_year": year,
            "issue_week": week,
            "issue_label": issue_label(year, week),
            "issue_week_start_date": start.isoformat() if start else None,
        }
        for year, week, start in rows
    ]


def latest_issue() -> dict[str, Any] | None:
    issues = available_issues()
    return issues[0] if issues else None


def resolve_params(
    *,
    horizon: int = DEFAULT_HORIZON,
    district_id: str | None = None,
    issue_year: int | None = None,
    issue_week: int | None = None,
) -> DashboardParams:
    """Resolve defaults (latest issue) without any request object."""
    if issue_year is None or issue_week is None:
        issue = latest_issue()
        if issue is not None:
            issue_year = issue["issue_year"]
            issue_week = issue["issue_week"]
    return DashboardParams(
        horizon=horizon,
        district_id=district_id,
        issue_year=issue_year,
        issue_week=issue_week,
    )


def parse_dashboard_params(args: Mapping[str, str]) -> DashboardParams:
    """Validate query parameters, raising DashboardParameterError on bad input."""
    horizon_raw = (args.get("horizon") or "").strip()
    horizon = DEFAULT_HORIZON
    if horizon_raw:
        try:
            horizon = int(horizon_raw)
        except ValueError:
            raise DashboardParameterError(
                f"horizon must be one of {', '.join(str(h) for h in HORIZONS)}."
            ) from None
        if horizon not in HORIZONS:
            raise DashboardParameterError(
                f"horizon must be one of {', '.join(str(h) for h in HORIZONS)}."
            )

    district_raw = (args.get("district_id") or "").strip()
    district_id: str | None = None
    if district_raw and district_raw.lower() != ALL_DISTRICTS:
        exists = DistrictReference.query.filter_by(district_id=district_raw, active=True).count()
        if not exists:
            raise DashboardParameterError(
                f"district_id {district_raw!r} is not an active district. "
                f"Use 'all' or a valid district_id."
            )
        district_id = district_raw

    year_raw = (args.get("issue_year") or "").strip()
    week_raw = (args.get("issue_week") or "").strip()
    if bool(year_raw) != bool(week_raw):
        raise DashboardParameterError("issue_year and issue_week must be supplied together.")

    issue_year: int | None = None
    issue_week: int | None = None
    if year_raw and week_raw:
        try:
            issue_year = int(year_raw)
            issue_week = int(week_raw)
        except ValueError:
            raise DashboardParameterError("issue_year and issue_week must be integers.") from None
        known = {(item["issue_year"], item["issue_week"]) for item in available_issues()}
        if (issue_year, issue_week) not in known:
            available = ", ".join(sorted(issue_label(y, w) or "" for y, w in known)) or "none"
            raise DashboardParameterError(
                f"issue {issue_label(issue_year, issue_week)} has no production forecasts. "
                f"Available issues: {available}."
            )

    return resolve_params(
        horizon=horizon,
        district_id=district_id,
        issue_year=issue_year,
        issue_week=issue_week,
    )


# --------------------------------------------------------------------------
# Model selection (consumed, never re-decided here)
# --------------------------------------------------------------------------
def _load_model_selection() -> dict[str, Any]:
    return json.loads(PERFORMANCE_SELECTION_PATH.read_text(encoding="utf-8"))


def _selection_entry(selection: dict[str, Any], key: str, horizon: int) -> dict[str, Any] | None:
    block = selection.get(key) or {}
    return block.get(str(horizon)) or block.get(horizon)


def selected_models(horizon: int, selection: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the selected count and surge models for a horizon, kept separate."""
    selection = selection if selection is not None else _load_model_selection()
    rationale = selection.get("selection_rationale") or {}
    out: dict[str, Any] = {}
    for task, key in (("count", "selected_count_models"), ("surge", "selected_surge_models")):
        entry = _selection_entry(selection, key, horizon) or {}
        out[task] = {
            "model_name": entry.get("model_name"),
            "model_key": entry.get("model_key"),
            "model_scope": entry.get("model_scope"),
            "metric_rule": entry.get("metric_rule") or rationale.get(task),
            "horizon": horizon,
        }
    out["same_model"] = bool(
        out["count"]["model_name"]
        and out["count"]["model_name"] == out["surge"]["model_name"]
    )
    return out


def selected_model_name(horizon: int, task: str) -> str | None:
    return selected_models(horizon)[task]["model_name"]


# --------------------------------------------------------------------------
# Preconditions and empty payload
# --------------------------------------------------------------------------
def dashboard_precondition_errors(*, require_outputs: bool) -> list[str]:
    errors: list[str] = []
    if require_outputs and ForecastOutput.query.count() == 0:
        errors.append("Prompt 25 forecast outputs are missing; complete Prompt 25 first.")
    if require_outputs and not PERFORMANCE_SELECTION_PATH.exists():
        errors.append("Prompt 26 model selection output is missing; complete Prompt 26 first.")
    if DistrictReference.query.filter_by(active=True).count() != EXPECTED_DISTRICTS:
        errors.append(f"district_reference must contain {EXPECTED_DISTRICTS} active districts.")
    if not PREPARED_GEOJSON_PATH.exists():
        errors.append(f"Prepared GeoJSON is missing: {PREPARED_GEOJSON_PATH}")
    if DistrictWeekFeature.query.count() == 0:
        errors.append("district_week_features has no observed panel rows.")
    missing_climate = [
        column
        for column in CLIMATE_COLUMNS
        if DistrictWeekFeature.query.filter(getattr(DistrictWeekFeature, column).isnot(None)).count() == 0
    ]
    if missing_climate:
        errors.append("Climate features/lags are missing: " + ", ".join(missing_climate))
    wind_columns = [
        column.name
        for table in (DistrictWeekFeature.__table__, ForecastOutput.__table__, ModelPerformance.__table__)
        for column in table.columns
        if "wind" in column.name.lower()
    ]
    if wind_columns:
        errors.append("Wind-speed fields are not allowed: " + ", ".join(wind_columns))
    return errors


def empty_dashboard_payload(reason: str, *, horizon: int = DEFAULT_HORIZON) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "params": DashboardParams(horizon=horizon).to_dict(),
        "summary": {
            "confirmed_malaria_cases": None,
            "previous_week_change_percent": None,
            "reporting_completeness_label": "HMIS 033b form/reporting completeness",
            "reporting_completeness": None,
            "districts_at_surge_risk": None,
            "forecast_horizon_weeks": horizon,
        },
        "weekly_series": {"rows": [], "intervals_available": False},
        "risk_table": {"rows": [], "risk_counts": {}},
        "map": {"type": "FeatureCollection", "features": [], "missing_geometry_districts": []},
        "climate_summary": {"rows": [], "status": "unavailable"},
        "model_performance": {"rows": [], "selected_models": {}, "message": "No model performance metrics available."},
        "filters": {"horizons": list(HORIZONS), "districts": [], "issues": []},
        "empty_states": [
            "No forecasts available.",
            "No selected model available.",
            "No climate data available.",
            "No model performance metrics available.",
        ],
    }


# --------------------------------------------------------------------------
# Shared query helpers
# --------------------------------------------------------------------------
def _issue_week_start(params: DashboardParams):
    row = (
        DistrictWeekFeature.query.with_entities(DistrictWeekFeature.week_start_date)
        .filter_by(epi_year=params.issue_year, epi_week=params.issue_week)
        .first()
    )
    if row and row[0]:
        return row[0]
    forecast = (
        ForecastOutput.query.with_entities(ForecastOutput.forecast_origin_date)
        .filter_by(
            forecast_origin_epi_year=params.issue_year,
            forecast_origin_epi_week=params.issue_week,
        )
        .first()
    )
    return forecast[0] if forecast else None


def _feature_rows_for_week(week_start, district_id: str | None = None) -> dict[str, dict[str, Any]]:
    if week_start is None:
        return {}
    query = DistrictWeekFeature.query.filter_by(week_start_date=week_start)
    if district_id:
        query = query.filter_by(district_id=district_id)
    return {
        row.district_id: {
            "confirmed_malaria_cases": row.confirmed_malaria_cases,
            "facility_reporting_completeness": row.facility_reporting_completeness,
        }
        for row in query.all()
    }


def _forecast_rows(
    *,
    model_name: str | None,
    params: DashboardParams,
    district_id: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return one forecast row per district for the issue/horizon/model."""
    if not model_name:
        return {}
    query = ForecastOutput.query.filter_by(
        model_name=model_name,
        horizon_weeks=params.horizon,
        forecast_origin_epi_year=params.issue_year,
        forecast_origin_epi_week=params.issue_week,
    )
    if district_id:
        query = query.filter_by(district_id=district_id)
    rows: dict[str, dict[str, Any]] = {}
    for forecast in query.all():
        rows[forecast.district_id] = {
            "predicted_cases": forecast.predicted_cases,
            "prediction_interval_lower": forecast.prediction_interval_lower,
            "prediction_interval_upper": forecast.prediction_interval_upper,
            "surge_probability": forecast.surge_probability,
            "predicted_surge_flag": forecast.predicted_surge_flag,
            "threshold_used": forecast.threshold_used,
            "threshold_method": forecast.threshold_method,
            "model_scope": forecast.model_scope,
            "target_epi_year": forecast.target_epi_year,
            "target_epi_week": forecast.target_epi_week,
            "target_week_start_date": forecast.target_week_start_date,
        }
    return rows


def _target_week(params: DashboardParams, model_name: str | None) -> dict[str, Any]:
    if not model_name:
        return {"target_epi_year": None, "target_epi_week": None, "target_week_start_date": None, "target_label": None}
    row = (
        ForecastOutput.query.with_entities(
            ForecastOutput.target_epi_year,
            ForecastOutput.target_epi_week,
            ForecastOutput.target_week_start_date,
        )
        .filter_by(
            model_name=model_name,
            horizon_weeks=params.horizon,
            forecast_origin_epi_year=params.issue_year,
            forecast_origin_epi_week=params.issue_week,
        )
        .first()
    )
    if not row:
        return {"target_epi_year": None, "target_epi_week": None, "target_week_start_date": None, "target_label": None}
    year, week, start = row
    return {
        "target_epi_year": year,
        "target_epi_week": week,
        "target_week_start_date": start.isoformat() if start else None,
        "target_label": issue_label(year, week),
    }


def _district_rows(params: DashboardParams | None = None, *, respect_filter: bool = True) -> list[dict[str, Any]]:
    query = DistrictReference.query.filter_by(active=True)
    if respect_filter and params is not None and params.district_id:
        query = query.filter_by(district_id=params.district_id)
    return [
        {"district_id": row.district_id, "district_name": row.district_name}
        for row in query.order_by(DistrictReference.district_id).all()
    ]


# --------------------------------------------------------------------------
# Section: filters
# --------------------------------------------------------------------------
def build_filters_section(params: DashboardParams | None = None) -> dict[str, Any]:
    """Filter options. Only issues that really exist are offered."""
    return {
        "horizons": [{"value": horizon, "label": f"{horizon} weeks"} for horizon in HORIZONS],
        "default_horizon": DEFAULT_HORIZON,
        "districts": _district_rows(respect_filter=False),
        "issues": available_issues(),
        "district_all_value": ALL_DISTRICTS,
    }


# --------------------------------------------------------------------------
# Section: risk rows (shared by risk-table, summary and map)
# --------------------------------------------------------------------------
def build_risk_rows(params: DashboardParams, *, respect_district_filter: bool = True) -> list[dict[str, Any]]:
    """Build risk rows joining the selected count and surge models per district.

    Count model supplies predicted cases. Surge model supplies the risk
    assessment, threshold comparison and score. Both names stay on every row.
    """
    models = selected_models(params.horizon)
    count_name = models["count"]["model_name"]
    surge_name = models["surge"]["model_name"]
    district_filter = params.district_id if respect_district_filter else None

    districts = _district_rows(params, respect_filter=respect_district_filter)
    count_rows = _forecast_rows(model_name=count_name, params=params, district_id=district_filter)
    surge_rows = _forecast_rows(model_name=surge_name, params=params, district_id=district_filter)

    week_start = _issue_week_start(params)
    previous_start = week_start - timedelta(weeks=1) if week_start else None
    observed = _feature_rows_for_week(week_start, district_filter)
    previous = _feature_rows_for_week(previous_start, district_filter)

    count_target = _target_week(params, count_name)
    surge_target = _target_week(params, surge_name)

    rows: list[dict[str, Any]] = []
    for district in districts:
        district_id = district["district_id"]
        count = count_rows.get(district_id) or {}
        surge = surge_rows.get(district_id) or {}
        observed_row = observed.get(district_id) or {}
        previous_row = previous.get(district_id) or {}

        threshold = surge.get("threshold_used")
        if threshold is None:
            threshold = count.get("threshold_used")

        risk = classify_forecast_risk(
            predicted_surge_flag=surge.get("predicted_surge_flag"),
            surge_probability=surge.get("surge_probability"),
            predicted_cases=surge.get("predicted_cases"),
            threshold_used=surge.get("threshold_used"),
        )

        surge_cases = _decimal(surge.get("predicted_cases"))
        threshold_decimal = _decimal(threshold)
        score = None
        if surge_cases is not None and threshold_decimal is not None:
            score = surge_cases - threshold_decimal

        observed_cases = observed_row.get("confirmed_malaria_cases")
        completeness = observed_row.get("facility_reporting_completeness")

        rows.append(
            {
                "district_id": district_id,
                "district_name": district["district_name"],
                "risk_level": risk.risk_level,
                "risk_basis": risk.risk_basis,
                "surge_output_type": risk.surge_output_type,
                "risk_assessment_label": RISK_ASSESSMENT_LABEL,
                "trend": _trend(observed_cases, previous_row.get("confirmed_malaria_cases")),
                "observed_cases": _json_value(observed_cases),
                # Count model output.
                "forecast_cases": _json_value(count.get("predicted_cases")),
                "predicted_cases": _json_value(count.get("predicted_cases")),
                "count_model_name": count_name,
                "count_model_scope": count.get("model_scope") or models["count"]["model_scope"],
                "prediction_interval_lower": _json_value(count.get("prediction_interval_lower")),
                "prediction_interval_upper": _json_value(count.get("prediction_interval_upper")),
                # Surge model output.
                "surge_predicted_cases": _json_value(surge.get("predicted_cases")),
                "surge_model_name": surge_name,
                "surge_model_scope": surge.get("model_scope") or models["surge"]["model_scope"],
                "threshold_used": _json_value(threshold),
                "threshold_method": surge.get("threshold_method") or count.get("threshold_method"),
                "predicted_surge_score": _json_value(score),
                "predicted_surge_score_unit": "cases",
                "surge_probability": _json_value(surge.get("surge_probability")),
                # Context.
                "reporting_completeness": _json_value(completeness),
                "observed_cases_missing": observed_cases is None,
                "reporting_completeness_missing": completeness is None,
                "forecast_horizon_weeks": params.horizon,
                "issue_year": params.issue_year,
                "issue_week": params.issue_week,
                "issue_label": issue_label(params.issue_year, params.issue_week),
                "target_epi_year": count_target["target_epi_year"] or surge_target["target_epi_year"],
                "target_epi_week": count_target["target_epi_week"] or surge_target["target_epi_week"],
                "target_week_start_date": count_target["target_week_start_date"] or surge_target["target_week_start_date"],
                "target_label": count_target["target_label"] or surge_target["target_label"],
                "no_data": risk.no_data,
            }
        )
    return rows


def build_risk_table_section(params: DashboardParams) -> dict[str, Any]:
    rows = build_risk_rows(params)
    models = selected_models(params.horizon)
    return {
        "rows": rows,
        "risk_counts": dict(Counter(row["risk_level"] for row in rows)),
        "selected_models": models,
        "risk_assessment_label": RISK_ASSESSMENT_LABEL,
        "risk_assessment_note": RISK_ASSESSMENT_NOTE,
        "threshold_method_note": THRESHOLD_METHOD_NOTE,
        "surge_probability_available": any(row["surge_probability"] is not None for row in rows),
        "score_definition": "predicted_surge_score = surge-model predicted cases − district threshold (cases, not a probability).",
        "params": params.to_dict(),
    }


# --------------------------------------------------------------------------
# Section: summary
# --------------------------------------------------------------------------
def build_summary_section(params: DashboardParams) -> dict[str, Any]:
    week_start = _issue_week_start(params)
    previous_start = week_start - timedelta(weeks=1) if week_start else None
    current = _feature_rows_for_week(week_start, params.district_id)
    previous = _feature_rows_for_week(previous_start, params.district_id)

    current_total = _sum_present(row.get("confirmed_malaria_cases") for row in current.values())
    previous_total = _sum_present(row.get("confirmed_malaria_cases") for row in previous.values())

    completeness_values = [
        _decimal(row.get("facility_reporting_completeness"))
        for row in current.values()
        if _decimal(row.get("facility_reporting_completeness")) is not None
    ]
    reporting = None
    if completeness_values:
        reporting = sum(completeness_values, Decimal("0")) / Decimal(len(completeness_values))

    risk_rows = build_risk_rows(params)
    models = selected_models(params.horizon)
    target = _target_week(params, models["count"]["model_name"]) or {}

    district_name = None
    if params.district_id:
        reference = DistrictReference.query.filter_by(district_id=params.district_id).first()
        district_name = reference.district_name if reference else params.district_id

    return {
        "confirmed_malaria_cases": _json_value(current_total),
        "previous_week_change_percent": _json_value(_pct_change(current_total, previous_total)),
        "reporting_completeness_label": "HMIS 033b form/reporting completeness",
        "reporting_completeness": _json_value(reporting),
        "reporting_completeness_note": (
            "Reporting completeness describes how much of the HMIS 033b reporting "
            "was received. It is not a measure of malaria burden."
        ),
        "districts_at_surge_risk": sum(1 for row in risk_rows if row["risk_level"] in AT_RISK_LEVELS),
        "districts_in_scope": len(risk_rows),
        "at_risk_levels": sorted(AT_RISK_LEVELS),
        "forecast_horizon_weeks": params.horizon,
        "issue_year": params.issue_year,
        "issue_week": params.issue_week,
        "issue_label": issue_label(params.issue_year, params.issue_week),
        "issue_week_start_date": week_start.isoformat() if week_start else None,
        "target_epi_year": target.get("target_epi_year"),
        "target_epi_week": target.get("target_epi_week"),
        "target_week_start_date": target.get("target_week_start_date"),
        "target_label": target.get("target_label"),
        "latest_epi_year": params.issue_year,
        "latest_epi_week": params.issue_week,
        "count_model_name": models["count"]["model_name"],
        "surge_model_name": models["surge"]["model_name"],
        "same_model": models["same_model"],
        "scope": "national" if params.is_national else "district",
        "district_id": params.district_id,
        "district_name": district_name,
        "aggregation_note": NATIONAL_AGGREGATION_NOTE if params.is_national else f"Values are for {district_name} only.",
        "missing_current_district_rows": sum(1 for row in current.values() if row.get("confirmed_malaria_cases") is None),
        "risk_assessment_label": RISK_ASSESSMENT_LABEL,
        "risk_assessment_note": RISK_ASSESSMENT_NOTE,
        "params": params.to_dict(),
    }


# --------------------------------------------------------------------------
# Section: weekly series
# --------------------------------------------------------------------------
def build_weekly_series_section(params: DashboardParams) -> dict[str, Any]:
    """Observed weeks up to the issue week, then appended forecast target rows.

    Forecast rows are appended as future weeks rather than joined onto observed
    weeks, because the target weeks lie beyond the observed panel. Prediction
    intervals appear only when the selected count model supplies them; the
    service never substitutes a different model to obtain intervals.
    """
    models = selected_models(params.horizon)
    count_name = models["count"]["model_name"]
    surge_name = models["surge"]["model_name"]
    week_start = _issue_week_start(params)
    target = _target_week(params, count_name)

    district_name = None
    if params.district_id:
        reference = DistrictReference.query.filter_by(district_id=params.district_id).first()
        district_name = reference.district_name if reference else params.district_id

    observed_query = DistrictWeekFeature.query.with_entities(
        DistrictWeekFeature.epi_year,
        DistrictWeekFeature.epi_week,
        DistrictWeekFeature.week_start_date,
        func.sum(DistrictWeekFeature.confirmed_malaria_cases),
        func.count(DistrictWeekFeature.confirmed_malaria_cases),
    )
    if params.district_id:
        observed_query = observed_query.filter(DistrictWeekFeature.district_id == params.district_id)
    if week_start is not None:
        observed_query = observed_query.filter(DistrictWeekFeature.week_start_date <= week_start)
    observed = (
        observed_query.group_by(
            DistrictWeekFeature.epi_year,
            DistrictWeekFeature.epi_week,
            DistrictWeekFeature.week_start_date,
        )
        .order_by(DistrictWeekFeature.week_start_date)
        .all()
    )

    rows: list[dict[str, Any]] = [
        {
            "epi_year": year,
            "epi_week": week,
            "week_label": issue_label(year, week),
            "week_start_date": start.isoformat() if start else None,
            "period": "observed",
            "actual_cases": _json_value(actual),
            "districts_reporting": reporting,
            "predicted_cases": None,
            "prediction_interval_lower": None,
            "prediction_interval_upper": None,
            "surge_threshold": None,
        }
        for year, week, start, actual, reporting in observed
    ]

    forecast_query = ForecastOutput.query.with_entities(
        ForecastOutput.target_epi_year,
        ForecastOutput.target_epi_week,
        ForecastOutput.target_week_start_date,
        func.sum(ForecastOutput.predicted_cases),
        func.sum(ForecastOutput.prediction_interval_lower),
        func.sum(ForecastOutput.prediction_interval_upper),
        func.count(ForecastOutput.prediction_interval_lower),
    ).filter_by(
        model_name=count_name,
        horizon_weeks=params.horizon,
        forecast_origin_epi_year=params.issue_year,
        forecast_origin_epi_week=params.issue_week,
    )
    if params.district_id:
        forecast_query = forecast_query.filter(ForecastOutput.district_id == params.district_id)
    forecasts = (
        forecast_query.group_by(
            ForecastOutput.target_epi_year,
            ForecastOutput.target_epi_week,
            ForecastOutput.target_week_start_date,
        )
        .order_by(ForecastOutput.target_week_start_date)
        .all()
    )

    threshold_query = ForecastOutput.query.with_entities(
        ForecastOutput.target_epi_year,
        ForecastOutput.target_epi_week,
        ForecastOutput.target_week_start_date,
        func.sum(ForecastOutput.threshold_used),
        func.count(ForecastOutput.threshold_used),
    ).filter_by(
        model_name=surge_name,
        horizon_weeks=params.horizon,
        forecast_origin_epi_year=params.issue_year,
        forecast_origin_epi_week=params.issue_week,
    )
    if params.district_id:
        threshold_query = threshold_query.filter(
            ForecastOutput.district_id == params.district_id
        )
    thresholds = {
        (year, week, start): (threshold, threshold_count)
        for year, week, start, threshold, threshold_count in threshold_query.group_by(
            ForecastOutput.target_epi_year,
            ForecastOutput.target_epi_week,
            ForecastOutput.target_week_start_date,
        ).all()
    }

    intervals_available = False
    for year, week, start, predicted, lower, upper, interval_count in forecasts:
        threshold, threshold_count = thresholds.get((year, week, start), (None, 0))
        has_intervals = bool(interval_count) and lower is not None and upper is not None
        intervals_available = intervals_available or has_intervals
        rows.append(
            {
                "epi_year": year,
                "epi_week": week,
                "week_label": issue_label(year, week),
                "week_start_date": start.isoformat() if start else None,
                "period": "forecast",
                "actual_cases": None,
                "districts_reporting": None,
                "predicted_cases": _json_value(predicted),
                "prediction_interval_lower": _json_value(lower) if has_intervals else None,
                "prediction_interval_upper": _json_value(upper) if has_intervals else None,
                "surge_threshold": _json_value(threshold) if threshold_count else None,
            }
        )

    return {
        "rows": rows,
        "observed_rows": sum(1 for row in rows if row["period"] == "observed"),
        "forecast_rows": sum(1 for row in rows if row["period"] == "forecast"),
        "intervals_available": intervals_available,
        "intervals_note": (
            f"This chart shows the selected target week only. {count_name} does not "
            "provide a prediction range for this forecast."
        )
        if not intervals_available
        else f"Prediction intervals are published by {count_name} for this horizon.",
        "count_model_name": count_name,
        "surge_model_name": surge_name,
        "threshold_model_name": surge_name,
        "selected_models": models,
        "forecast_horizon_weeks": params.horizon,
        "district_id": params.district_id,
        "district_name": district_name,
        "target_label": target.get("target_label"),
        "threshold_note": (
            "Green shows confirmed cases already reported. The blue diamond is the "
            "model's expected cases for the target week. The red X is the seasonal "
            "surge benchmark for that same week. A forecast above the benchmark needs "
            "early review; it does not confirm an outbreak."
        ),
        "threshold_method_note": THRESHOLD_METHOD_NOTE,
        "national_threshold_note": NATIONAL_THRESHOLD_NOTE if params.is_national else None,
        "scope": "national" if params.is_national else "district",
        "issue_label": issue_label(params.issue_year, params.issue_week),
        "issue_week_start_date": week_start.isoformat() if week_start else None,
        "params": params.to_dict(),
    }


# --------------------------------------------------------------------------
# Section: climate
# --------------------------------------------------------------------------
def build_climate_section(params: DashboardParams) -> dict[str, Any]:
    """Current and lagged climate values for the issue week.

    No correlation is computed anywhere in the pipeline, so no 'best-correlating
    lag' is claimed. The highlighted lag is simply the lag aligned to the
    selected forecast horizon.
    """
    week_start = _issue_week_start(params)
    total_districts = DistrictReference.query.filter_by(active=True).count()
    scope_districts = 1 if params.district_id else total_districts

    rows: list[dict[str, Any]] = []
    for variable in CLIMATE_VARIABLES:
        measures: dict[str, Any] = {}
        for key, column in (
            ("current", f"{variable['variable']}_current"),
            ("lag_4", f"{variable['variable']}_lag_4"),
            ("lag_8", f"{variable['variable']}_lag_8"),
            ("lag_12", f"{variable['variable']}_lag_12"),
        ):
            attribute = getattr(DistrictWeekFeature, column)
            query = DistrictWeekFeature.query.with_entities(
                func.avg(attribute), func.count(attribute)
            )
            if week_start is not None:
                query = query.filter(DistrictWeekFeature.week_start_date == week_start)
            if params.district_id:
                query = query.filter(DistrictWeekFeature.district_id == params.district_id)
            value, non_null = query.first() or (None, 0)
            measures[key] = {
                "value": _json_value(_round(value, 2)),
                "districts_with_value": int(non_null or 0),
                "column": column,
            }

        highlighted = f"lag_{params.horizon}"
        rows.append(
            {
                "variable": variable["variable"],
                "label": variable["label"],
                "source": variable["source"],
                "unit": variable["unit"],
                "unit_label": variable["unit_label"],
                "measures": measures,
                "highlighted_lag_weeks": params.horizon,
                "highlighted_lag_key": highlighted,
                "highlighted_lag_reason": (
                    f"Shown because the selected forecast horizon is {params.horizon} weeks. "
                    "This is not a best-correlating lag; lag-malaria correlation is not computed."
                ),
                "coverage": {
                    "districts_expected": scope_districts,
                    "districts_with_current_value": measures["current"]["districts_with_value"],
                },
                "status": "available" if measures["current"]["districts_with_value"] else "unavailable",
            }
        )

    return {
        "rows": rows,
        "data_week_label": issue_label(params.issue_year, params.issue_week),
        "data_week_start_date": week_start.isoformat() if week_start else None,
        "scope": "national" if params.is_national else "district",
        "district_id": params.district_id,
        "aggregation": "unweighted_district_mean" if params.is_national else "district_value",
        "aggregation_label": (
            "Unweighted mean across districts; each district contributes equally."
            if params.is_national
            else "Value for the selected district."
        ),
        "correlation_available": False,
        "correlation_note": (
            "Correlation between climate lags and malaria cases is not computed in this "
            "pipeline, so no best-correlating lag is claimed."
        ),
        "causation_note": (
            "Climate variables are associated predictors used by the models. Association "
            "does not establish causation and does not explain malaria transmission on its own."
        ),
        "status": "available" if any(row["status"] == "available" for row in rows) else "unavailable",
        "params": params.to_dict(),
    }


# --------------------------------------------------------------------------
# Section: model performance
# --------------------------------------------------------------------------
def build_model_performance_section(params: DashboardParams) -> dict[str, Any]:
    """National rolling-origin validation metrics. Never district-filtered."""
    selection = _load_model_selection()
    per_horizon = {horizon: selected_models(horizon, selection) for horizon in HORIZONS}

    rows: list[dict[str, Any]] = []
    for perf in ModelPerformance.query.order_by(
        ModelPerformance.horizon_weeks, ModelPerformance.model_name
    ).all():
        horizon_models = per_horizon.get(perf.horizon_weeks, {})
        count_selected = bool(horizon_models.get("count", {}).get("model_name") == perf.model_name)
        surge_selected = bool(horizon_models.get("surge", {}).get("model_name") == perf.model_name)
        rows.append(
            {
                "model_name": perf.model_name,
                "model_scope": perf.model_scope,
                "task": "regression_and_classification",
                "horizon": perf.horizon_weeks,
                "mae": _json_value(perf.mae),
                "rmse": _json_value(perf.rmse),
                "mape": _json_value(perf.mape),
                "wape": _json_value(perf.wape),
                "r_squared": _json_value(perf.r_squared),
                "sensitivity": _json_value(perf.sensitivity),
                "specificity": _json_value(perf.specificity),
                "precision": _json_value(perf.precision),
                "f1": _json_value(perf.f1_score),
                "auc": _json_value(perf.auc),
                "validation_method": perf.validation_method,
                "selected_for_count": count_selected,
                "selected_for_surge": surge_selected,
                "selected_labels": [
                    label
                    for label, flag in (("Count", count_selected), ("Surge", surge_selected))
                    if flag
                ],
            }
        )

    rationale = selection.get("selection_rationale") or {}
    return {
        "rows": rows,
        "selected_models": {str(horizon): per_horizon[horizon] for horizon in HORIZONS},
        "selection_rules": {
            "count": rationale.get("count"),
            "surge": rationale.get("surge"),
        },
        "accuracy_note": (
            f"{rationale.get('surge') or 'Surge model selection evidence is unavailable.'} "
            "Accuracy alone is not used, because surge weeks are rare and a model can score "
            "high accuracy while missing most surges."
        ),
        "scope_note": (
            "These are national rolling-origin validation metrics across all districts. "
            "They are not recomputed for a selected district and do not change with the district filter."
        ),
        "horizon": params.horizon,
        "params": params.to_dict(),
        "message": None if rows else "No model performance metrics available.",
    }


# --------------------------------------------------------------------------
# Section: map
# --------------------------------------------------------------------------
def _load_web_map_geojson() -> dict[str, Any]:
    """Load the derived simplified boundary, cached on modification time."""
    global _WEB_MAP_CACHE
    if not WEB_MAP_GEOJSON_PATH.exists():
        raise DashboardDataError(
            f"Simplified web-map boundary is missing: {WEB_MAP_GEOJSON_PATH}. "
            "Run 'flask dashboard build-web-map' to derive it from the approved boundary."
        )
    mtime = WEB_MAP_GEOJSON_PATH.stat().st_mtime
    if _WEB_MAP_CACHE is not None and _WEB_MAP_CACHE[0] == mtime:
        return _WEB_MAP_CACHE[1]
    data = json.loads(WEB_MAP_GEOJSON_PATH.read_text(encoding="utf-8"))
    _WEB_MAP_CACHE = (mtime, data)
    return data


_WEB_MAP_CACHE: tuple[float, dict[str, Any]] | None = None


def map_risk_layer(geojson: dict[str, Any], risk_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Join risk rows onto boundary features using district_id."""
    by_id = {row["district_id"]: row for row in risk_rows}
    features: list[dict[str, Any]] = []
    matched_ids: set[str] = set()
    for feature in geojson.get("features", []):
        properties = dict(feature.get("properties") or {})
        district_id = properties.get("district_id")
        risk = by_id.get(district_id)
        if risk is None:
            risk = {
                "district_id": district_id,
                "district_name": properties.get("district_name"),
                "risk_level": NO_DATA_RISK,
                "observed_cases": None,
                "forecast_cases": None,
                "surge_probability": None,
                "no_data": True,
            }
        else:
            matched_ids.add(district_id)
        properties = {
            "district_id": district_id,
            "district_name": risk.get("district_name") or properties.get("district_name"),
            "risk_level": risk.get("risk_level", NO_DATA_RISK),
            "no_data": bool(risk.get("no_data")),
            "observed_cases": risk.get("observed_cases"),
            "predicted_cases": risk.get("forecast_cases"),
            "surge_predicted_cases": risk.get("surge_predicted_cases"),
            "threshold_used": risk.get("threshold_used"),
            "predicted_surge_score": risk.get("predicted_surge_score"),
            "surge_probability": risk.get("surge_probability"),
            "count_model_name": risk.get("count_model_name"),
            "surge_model_name": risk.get("surge_model_name"),
            "risk_basis": risk.get("risk_basis"),
            "forecast_horizon_weeks": risk.get("forecast_horizon_weeks"),
            "target_label": risk.get("target_label"),
        }
        features.append({"type": "Feature", "properties": properties, "geometry": feature.get("geometry")})
    expected_ids = {row["district_id"] for row in risk_rows}
    missing_geometry = sorted(expected_ids - matched_ids)
    return {"type": "FeatureCollection", "features": features, "missing_geometry_districts": missing_geometry}


def build_map_section(params: DashboardParams) -> dict[str, Any]:
    """National choropleth. The district filter highlights, it never removes context."""
    risk_rows = build_risk_rows(params, respect_district_filter=False)
    layer = map_risk_layer(_load_web_map_geojson(), risk_rows)
    models = selected_models(params.horizon)
    layer.update(
        {
            "join_key": "district_id",
            "selected_district_id": params.district_id,
            "scope": "national",
            "scope_note": "The map always shows all districts for context; the selected district is highlighted.",
            "risk_counts": dict(Counter(row["risk_level"] for row in risk_rows)),
            "risk_levels": ["Very High", "High", "Moderate", "Low", NO_DATA_RISK],
            "risk_assessment_label": RISK_ASSESSMENT_LABEL,
            "risk_assessment_note": RISK_ASSESSMENT_NOTE,
            "count_model_name": models["count"]["model_name"],
            "surge_model_name": models["surge"]["model_name"],
            "boundary_source": "UBOS district boundaries, simplified for web display from the approved prepared boundary.",
            "params": params.to_dict(),
        }
    )
    return layer


# --------------------------------------------------------------------------
# Derived web-map artifact
# --------------------------------------------------------------------------
def _polygonal_only(geometry):
    """Return polygonal geometry only, or None. Repair can emit stray lines."""
    from shapely.ops import unary_union

    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if geometry.geom_type == "GeometryCollection":
        parts = [part for part in geometry.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]
        if not parts:
            return None
        merged = unary_union(parts)
        return merged if merged.geom_type in {"Polygon", "MultiPolygon"} else None
    return None


def _valid_polygonal(geometry):
    """Return a valid polygonal geometry, repairing if needed, else None."""
    from shapely.validation import make_valid

    candidate = _polygonal_only(geometry)
    if candidate is None:
        return None
    if candidate.is_valid:
        return candidate
    return _polygonal_only(make_valid(candidate))


def _rounded(geometry, decimals: int):
    from shapely.ops import transform

    return transform(lambda *coords: tuple(round(value, decimals) for value in coords), geometry)


def build_web_map_geojson(
    *,
    source: Path = PREPARED_GEOJSON_PATH,
    destination: Path = WEB_MAP_GEOJSON_PATH,
    tolerance: float = WEB_MAP_SIMPLIFY_TOLERANCE_DEGREES,
    decimals: int = WEB_MAP_COORDINATE_DECIMALS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Derive a browser-sized boundary from the approved prepared GeoJSON.

    The approved boundary is never modified. Simplification preserves each
    feature's district_id and geometry validity.

    Coordinate rounding can introduce self-intersections, so every candidate is
    re-validated after rounding and repaired, then progressively backed off
    (finer rounding, unrounded, original) until a valid polygon survives.
    """
    from shapely.geometry import mapping, shape

    if not source.exists():
        raise DashboardDataError(f"Prepared GeoJSON is missing: {source}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    features: list[dict[str, Any]] = []
    repaired = 0
    fallback = 0

    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        original = shape(feature["geometry"])
        simplified = original.simplify(tolerance, preserve_topology=True)

        # Preferred candidate first, then progressively safer fallbacks.
        candidates = (
            _rounded(simplified, decimals),
            _rounded(simplified, decimals + 1),
            simplified,
            original,
        )
        chosen = None
        for index, candidate in enumerate(candidates):
            resolved = _valid_polygonal(candidate)
            if resolved is not None:
                chosen = resolved
                if index > 0:
                    fallback += 1
                if not candidate.is_valid:
                    repaired += 1
                break
        if chosen is None:
            raise DashboardDataError(
                f"Could not derive valid geometry for {properties.get('district_id')}."
            )

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "district_id": properties.get("district_id"),
                    "district_name": properties.get("district_name"),
                },
                "geometry": mapping(chosen),
            }
        )

    collection = {"type": "FeatureCollection", "features": features}
    serialized = json.dumps(collection, separators=(",", ":"))
    size_bytes = len(serialized.encode("utf-8"))

    district_ids = [feature["properties"].get("district_id") for feature in features]
    geometry_types = Counter(feature["geometry"]["type"] for feature in features)
    invalid = [
        feature["properties"].get("district_id")
        for feature in features
        if not shape(feature["geometry"]).is_valid
    ]

    summary = {
        "status": "success",
        "dry_run": dry_run,
        "source_path": str(source),
        "destination_path": str(destination),
        "source_size_bytes": source.stat().st_size,
        "output_size_bytes": size_bytes,
        "output_size_mb": round(size_bytes / 1_000_000, 3),
        "max_allowed_bytes": WEB_MAP_MAX_BYTES,
        "within_size_budget": size_bytes <= WEB_MAP_MAX_BYTES,
        "feature_count": len(features),
        "unique_district_ids": len(set(district_ids)),
        "null_district_ids": sum(1 for value in district_ids if not value),
        "geometry_types": dict(geometry_types),
        "invalid_geometries": invalid,
        "repaired_geometries": repaired,
        "fallback_to_original": fallback,
        "simplify_tolerance_degrees": tolerance,
        "coordinate_decimals": decimals,
        "approved_boundary_modified": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    problems: list[str] = []
    if len(features) != EXPECTED_DISTRICTS:
        problems.append(f"expected {EXPECTED_DISTRICTS} features, produced {len(features)}")
    if len(set(district_ids)) != EXPECTED_DISTRICTS:
        problems.append(f"expected {EXPECTED_DISTRICTS} unique district_id values, produced {len(set(district_ids))}")
    if summary["null_district_ids"]:
        problems.append("some features have no district_id")
    if invalid:
        problems.append(f"invalid geometry for: {', '.join(str(item) for item in invalid[:5])}")
    if set(geometry_types) - {"Polygon", "MultiPolygon"}:
        problems.append(f"unexpected geometry types: {sorted(set(geometry_types))}")
    if size_bytes > WEB_MAP_MAX_BYTES:
        problems.append(f"output {size_bytes} bytes exceeds the {WEB_MAP_MAX_BYTES} byte budget")
    if problems:
        summary["status"] = "failed"
        summary["problems"] = problems
        raise DashboardDataError("Web-map derivation failed: " + "; ".join(problems))

    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
        WEB_MAP_PROOF_PATH.parent.mkdir(parents=True, exist_ok=True)
        WEB_MAP_PROOF_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        global _WEB_MAP_CACHE
        _WEB_MAP_CACHE = None
    return summary


# --------------------------------------------------------------------------
# Whole payload + artifact build
# --------------------------------------------------------------------------
def dashboard_payload(
    *,
    horizon: int = DEFAULT_HORIZON,
    params: DashboardParams | None = None,
    allow_unavailable: bool = True,
) -> dict[str, Any]:
    """Return every dashboard section. Routes build sections independently."""
    errors = dashboard_precondition_errors(require_outputs=True)
    if errors:
        if allow_unavailable:
            return empty_dashboard_payload("; ".join(errors), horizon=horizon)
        raise DashboardDataError("; ".join(errors))

    if params is None:
        params = resolve_params(horizon=horizon)

    return {
        "status": "available",
        "reason": None,
        "params": params.to_dict(),
        "summary": build_summary_section(params),
        "weekly_series": build_weekly_series_section(params),
        "risk_table": build_risk_table_section(params),
        "map": build_map_section(params),
        "climate_summary": build_climate_section(params),
        "model_performance": build_model_performance_section(params),
        "filters": build_filters_section(params),
        "selection": _load_model_selection(),
        "empty_states": [],
    }


def build_dashboard_data(
    *, latest: bool = True, horizon: int = DEFAULT_HORIZON, dry_run: bool = False
) -> DashboardBuildSummary:
    """Build dashboard JSON artifacts when all upstream outputs exist."""
    errors = dashboard_precondition_errors(require_outputs=True)
    if errors:
        raise DashboardDataError("; ".join(errors))
    payload = dashboard_payload(horizon=horizon, allow_unavailable=False)
    risk_rows = payload["risk_table"]["rows"]
    risk_counts = dict(Counter(row["risk_level"] for row in risk_rows))
    no_data = [row["district_id"] for row in risk_rows if row["no_data"]]
    summary = DashboardBuildSummary(
        dry_run=dry_run,
        status="dry-run" if dry_run else "success",
        horizon=horizon,
        risk_counts=risk_counts,
        no_data_districts=no_data,
        generated_paths=[SUMMARY_OUTPUT_PATH, RISK_TABLE_OUTPUT_PATH, MAP_RISK_OUTPUT_PATH],
    )
    if not dry_run:
        DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
        DASHBOARD_PROOF_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_OUTPUT_PATH.write_text(json.dumps(payload["summary"], indent=2, sort_keys=True), encoding="utf-8")
        RISK_TABLE_OUTPUT_PATH.write_text(json.dumps(payload["risk_table"], indent=2, sort_keys=True), encoding="utf-8")
        MAP_RISK_OUTPUT_PATH.write_text(json.dumps(payload["map"], indent=2, sort_keys=True), encoding="utf-8")
        DASHBOARD_PROOF_PATH.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------
# Backwards-compatible helper retained for tests and artifact building
# --------------------------------------------------------------------------
def risk_table_from_rows(
    *,
    district_rows: list[dict[str, Any]],
    forecast_rows: dict[str, dict[str, Any]],
    observed_rows: dict[str, dict[str, Any]],
    previous_rows: dict[str, dict[str, Any]],
    horizon: int,
    count_model_name: str | None = None,
    surge_model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Pure row builder used by unit tests and offline checks."""
    rows: list[dict[str, Any]] = []
    for district in district_rows:
        district_id = district["district_id"]
        forecast = forecast_rows.get(district_id) or {}
        observed = observed_rows.get(district_id) or {}
        previous = previous_rows.get(district_id) or {}
        risk = classify_forecast_risk(
            predicted_surge_flag=forecast.get("predicted_surge_flag"),
            surge_probability=forecast.get("surge_probability"),
            predicted_cases=forecast.get("predicted_cases"),
            threshold_used=forecast.get("threshold_used"),
        )
        cases = _decimal(forecast.get("predicted_cases"))
        threshold = _decimal(forecast.get("threshold_used"))
        score = cases - threshold if cases is not None and threshold is not None else None
        rows.append(
            {
                "district_id": district_id,
                "district_name": district["district_name"],
                "risk_level": risk.risk_level,
                "risk_basis": risk.risk_basis,
                "surge_output_type": risk.surge_output_type,
                "trend": _trend(observed.get("confirmed_malaria_cases"), previous.get("confirmed_malaria_cases")),
                "observed_cases": _json_value(observed.get("confirmed_malaria_cases")),
                "forecast_cases": _json_value(forecast.get("predicted_cases")),
                "threshold_used": _json_value(forecast.get("threshold_used")),
                "predicted_surge_score": _json_value(score),
                "surge_probability": _json_value(forecast.get("surge_probability")),
                "count_model_name": count_model_name,
                "surge_model_name": surge_model_name,
                "forecast_horizon_weeks": horizon,
                "no_data": risk.no_data,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Value helpers
# --------------------------------------------------------------------------
def _trend(current: Any, previous: Any) -> str:
    curr = _decimal(current)
    prev = _decimal(previous)
    if curr is None or prev is None:
        return "unknown"
    if curr > prev:
        return "increasing"
    if curr < prev:
        return "decreasing"
    return "stable"


def _sum_present(values) -> Decimal | None:
    present = [_decimal(value) for value in values if _decimal(value) is not None]
    if not present:
        return None
    return sum(present, Decimal("0"))


def _pct_change(current: Decimal | None, previous: Decimal | None) -> Decimal | None:
    if current is None or previous in {None, Decimal("0")}:
        return None
    return ((current - previous) / previous * Decimal("100")).quantize(Decimal("0.0001"))


def _round(value: Any, places: int) -> Decimal | None:
    decimal = _decimal(value)
    if decimal is None:
        return None
    return decimal.quantize(Decimal(10) ** -places)


def _json_value(value: Any) -> int | float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    decimal = _decimal(value)
    if decimal is None:
        return str(value)
    if decimal == decimal.to_integral_value():
        return int(decimal)
    return float(decimal)


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
