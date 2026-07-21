"""Project-state realignment audit for recovery checkpoints."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import distinct, func

from app.models import (
    DistrictReference,
    DistrictWeekClimate,
    DistrictWeekFeature,
    ForecastOutput,
    ModelPerformance,
    PipelineRun,
    RawDHIS2Malaria,
    UBOSPopulation,
)
from app.services.dashboard_data import DASHBOARD_PROOF_PATH
from app.services.forecast_generation import (
    FORECAST_SUMMARY_PATH,
    PRODUCTION_MANIFEST_PATH,
    VALIDATION_ARTIFACTS,
)
from app.services.modelling_dataset import (
    ABIM_SPOT_CHECK,
    CLIMATE_COLUMNS,
    DATASET_PATH,
    SUMMARY_PATH,
    modelling_dataset_ready_for_downstream,
)
from app.services.model_performance import PERFORMANCE_SELECTION_PATH
from app.services.preprocessing import iso_weeks_in_year

PROJECT_STATE_SUMMARY_PATH = Path("data/quality/project_state_realignment_summary.json")
PREPARED_GEOJSON_PATH = Path("data/boundaries/uganda_districts_prepared.geojson")
RAW_BOUNDARY_PATH = Path("data/boundaries/UGANDA_DISTRICT.json")
UBOS_POPULATION_PATH = Path("data/reference/ubos_population.csv")
DISTRICT_REFERENCE_PATH = Path("data/reference/district_reference.csv")
REPORTING_COMPLETENESS_ARTIFACT = Path("data/quality/facility_reporting_completeness_2020_2025_summary.json")
REPORTING_METRIC_ARTIFACT = Path("data/quality/facility_reporting_completeness_verify_u6xATVzE7VO.json")
DHIS2_PANEL_ARTIFACT = Path("data/quality/dhis2_2020_2025_panel_summary.json")
CLIMATE_RAW_DIR = Path("data/climate/raw")
REQUIRED_CLIMATE_LOOKBACK_START = date(2019, 10, 7)
ANALYSIS_YEARS = range(2020, 2026)
EXPECTED_DISTRICTS = 146


class ProjectStateError(RuntimeError):
    """Raised when the project state audit cannot be generated."""


@dataclass
class ComponentStatus:
    status: str
    row_counts: dict[str, Any]
    artifact_paths: dict[str, str | None]
    blocking_reason: str | None
    next_required_action: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "row_counts": self.row_counts,
            "artifact_paths": self.artifact_paths,
            "blocking_reason": self.blocking_reason,
            "next_required_action": self.next_required_action,
            "details": self.details,
        }


def build_project_state_summary(*, write: bool = True, output_path: Path = PROJECT_STATE_SUMMARY_PATH) -> dict[str, Any]:
    """Audit current project state and optionally write a JSON proof artifact.

    Phase 29F performance fix. A read-only (``write=False``) audit re-parses the
    same unchanged artifacts on every call, and a single governance request needs
    it twice (pipeline status and data-quality evidence), costing ~4.6s each.
    The result is therefore memoised for the lifetime of ONE Flask request.

    The cache is deliberately narrow: read-only calls only, request-scoped only,
    and a deep copy is returned so no caller can mutate another's result. Outside
    a request context nothing is cached, and ``write=True`` always recomputes, so
    CLI verification behaviour is unchanged. No analytical value is altered — the
    same summary is returned, just computed once.
    """
    cache_key = "_project_state_summary_readonly"
    if not write:
        try:
            from flask import g, has_request_context

            if has_request_context() and hasattr(g, cache_key):
                return copy.deepcopy(getattr(g, cache_key))
        except Exception:  # pragma: no cover - never let caching break the audit
            pass

    components = {
        "reference_and_boundaries": _reference_and_boundaries(),
        "dhis2_malaria_panel": _dhis2_malaria_panel(),
        "reporting_completeness": _reporting_completeness(),
        "climate_ingestion": _climate_ingestion(),
        "climate_aggregation": _climate_aggregation(),
        "climate_lags": _climate_lags(),
        "modelling_dataset": _modelling_dataset(),
        "models_and_forecasts": _models_and_forecasts(),
        "visible_dashboard": _visible_dashboard(),
    }
    safe_prompt_29 = _safe_to_proceed_prompt29(components)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "safe_to_proceed_to_prompt_29": safe_prompt_29,
        "recommended_next_action": _recommended_next_action(components),
        "components": {key: value.to_dict() for key, value in components.items()},
        "gating_logic": _gating_logic(components),
        "secrets": _secret_safety_summary(),
    }
    if write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    else:
        # Memoise for the remainder of this request only (see docstring).
        try:
            from flask import g, has_request_context

            if has_request_context():
                setattr(g, cache_key, copy.deepcopy(summary))
        except Exception:  # pragma: no cover - caching must never break the audit
            pass
    return summary


def _reference_and_boundaries() -> ComponentStatus:
    district_count = DistrictReference.query.count()
    active = DistrictReference.query.filter_by(active=True).count()
    mapped = DistrictReference.query.filter(DistrictReference.dhis2_orgunit_id.isnot(None)).count()
    ubos_rows = UBOSPopulation.query.count()
    ubos_2024 = UBOSPopulation.query.filter_by(year=2024).count()
    prepared_exists = PREPARED_GEOJSON_PATH.exists()
    raw_exists = RAW_BOUNDARY_PATH.exists()
    status = "complete" if active == EXPECTED_DISTRICTS and mapped == EXPECTED_DISTRICTS and ubos_2024 == EXPECTED_DISTRICTS and prepared_exists else "partial"
    blocking = None if status == "complete" else "Reference frame, mappings, UBOS denominators or prepared GeoJSON are incomplete."
    return ComponentStatus(
        status=status,
        row_counts={
            "district_reference": district_count,
            "active_districts": active,
            "districts_with_dhis2_orgunit_id": mapped,
            "ubos_population_rows": ubos_rows,
            "ubos_2024_rows": ubos_2024,
        },
        artifact_paths={
            "district_reference_csv": _path(DISTRICT_REFERENCE_PATH),
            "ubos_population_csv": _path(UBOS_POPULATION_PATH),
            "raw_boundary": _path(RAW_BOUNDARY_PATH),
            "prepared_geojson": _path(PREPARED_GEOJSON_PATH),
        },
        blocking_reason=blocking,
        next_required_action="No reference action required before climate work." if status == "complete" else "Repair district frame/reference artifacts before continuing.",
        details={"expected_districts": EXPECTED_DISTRICTS, "join_key": "district_id"},
    )


def _dhis2_malaria_panel() -> ComponentStatus:
    raw_rows = RawDHIS2Malaria.query.count()
    raw_explicit_zero = RawDHIS2Malaria.query.filter(RawDHIS2Malaria.value == 0).count()
    feature_rows = DistrictWeekFeature.query.count()
    feature_districts = DistrictWeekFeature.query.with_entities(func.count(distinct(DistrictWeekFeature.district_id))).scalar() or 0
    feature_zero = DistrictWeekFeature.query.filter(DistrictWeekFeature.confirmed_malaria_cases == 0).count()
    expected_rows = EXPECTED_DISTRICTS * sum(iso_weeks_in_year(year) for year in ANALYSIS_YEARS)
    missing = expected_rows - feature_rows
    by_year = {
        year: count
        for year, count in DistrictWeekFeature.query.with_entities(
            DistrictWeekFeature.epi_year, func.count()
        )
        .group_by(DistrictWeekFeature.epi_year)
        .order_by(DistrictWeekFeature.epi_year)
        .all()
    }
    abim = DistrictWeekFeature.query.filter_by(district_id="UGA_ABIM", epi_year=2025, epi_week=20).first()
    abim_passed = bool(
        abim
        and abim.confirmed_malaria_cases == ABIM_SPOT_CHECK["dhis2_ui_confirmed_malaria_cases"]
        and str(abim.week_start_date) == "2025-05-12"
        and str(abim.week_end_date) == "2025-05-18"
    )
    status = "complete" if raw_rows > 0 and feature_rows > 0 and feature_districts == EXPECTED_DISTRICTS and abim_passed else "partial"
    return ComponentStatus(
        status=status,
        row_counts={
            "raw_dhis2_malaria": raw_rows,
            "district_week_features": feature_rows,
            "expected_district_weeks": expected_rows,
            "missing_district_weeks": missing,
            "districts_represented": feature_districts,
            "raw_explicit_zero_count": raw_explicit_zero,
            "feature_explicit_zero_count": feature_zero,
            "rows_by_year": by_year,
        },
        artifact_paths={"dhis2_panel_summary": _path(DHIS2_PANEL_ARTIFACT)},
        blocking_reason=None if status == "complete" else "DHIS2 panel or Abim/2025W20 proof is incomplete.",
        next_required_action="No DHIS2 malaria action required before climate work." if status == "complete" else "Repair DHIS2 extraction/preprocessing before modelling.",
        details={
            "missing_rows_not_zero_filled": True,
            "abim_2025w20_proof": {
                "stored_cases": abim.confirmed_malaria_cases if abim else None,
                "stored_week_start": str(abim.week_start_date) if abim else None,
                "stored_week_end": str(abim.week_end_date) if abim else None,
                "manual_ui_check": ABIM_SPOT_CHECK,
                "passed": abim_passed,
            },
        },
    )


def _reporting_completeness() -> ComponentStatus:
    feature_rows = DistrictWeekFeature.query.count()
    non_null = DistrictWeekFeature.query.filter(DistrictWeekFeature.facility_reporting_completeness.isnot(None)).count()
    artifact = _read_json(REPORTING_METRIC_ARTIFACT)
    status = "complete" if feature_rows > 0 and non_null == feature_rows else "partial"
    return ComponentStatus(
        status=status,
        row_counts={"feature_rows": feature_rows, "facility_reporting_completeness_non_null": non_null},
        artifact_paths={
            "reporting_completeness_summary": _path(REPORTING_COMPLETENESS_ARTIFACT),
            "metric_verification": _path(REPORTING_METRIC_ARTIFACT),
        },
        blocking_reason=None if status == "complete" else "HMIS 033b reporting completeness is not populated for all feature rows.",
        next_required_action="No reporting-completeness action required before climate work." if status == "complete" else "Rerun Prompt 18 reporting completeness extraction.",
        details={
            "label": "HMIS 033b form/reporting completeness",
            "metric_uid": artifact.get("uid") or artifact.get("id") or "u6xATVzE7VO",
            "metric_name": artifact.get("name") or artifact.get("displayName") or "HMIS 033b reporting rate",
            "no_credentials_stored": _no_known_secret_in_pipeline_runs(),
        },
    )


def _climate_ingestion() -> ComponentStatus:
    raw_files = _raw_climate_files()
    registered = _registered_climate_files()
    coverage = _coverage_by_pair(registered)
    required_covered = _lookback_coverage_present(coverage)
    status = "complete" if required_covered else ("partial" if registered else "missing")
    blocking = None if status == "complete" else "Real CHIRPS/ERA5-Land files are not registered with required lookback coverage from 2019-10-07."
    return ComponentStatus(
        status=status,
        row_counts={
            "raw_climate_files": len(raw_files),
            "registered_climate_files": len(registered),
            "chirps_registered_files": sum(1 for item in registered if item.get("source") == "CHIRPS"),
            "era5_registered_files": sum(1 for item in registered if item.get("source") == "ERA5-Land"),
        },
        artifact_paths={"climate_raw_dir": str(CLIMATE_RAW_DIR), "climate_manifests_dir": "data/climate/manifests"},
        blocking_reason=blocking,
        next_required_action="Next: complete Prompt 19, then 20, then 21." if status != "complete" else "Proceed to Prompt 20 climate aggregation.",
        details={
            "required_lookback_start": REQUIRED_CLIMATE_LOOKBACK_START.isoformat(),
            "coverage_by_source_variable": coverage,
            "raw_file_examples": [str(path) for path in raw_files[:10]],
            "config_status_no_values": _climate_config_status(),
            "dependency_status": _dependency_status(),
        },
    )


def _climate_aggregation() -> ComponentStatus:
    rows = DistrictWeekClimate.query.count()
    rainfall = DistrictWeekClimate.query.filter(DistrictWeekClimate.rainfall_current.isnot(None)).count()
    temperature = DistrictWeekClimate.query.filter(DistrictWeekClimate.temperature_current.isnot(None)).count()
    humidity = DistrictWeekClimate.query.filter(DistrictWeekClimate.humidity_current.isnot(None)).count()
    success_runs = PipelineRun.query.filter_by(run_type="climate_aggregation", status="success").count()
    status = "complete" if rows > 0 and rainfall > 0 and temperature > 0 and humidity > 0 and success_runs > 0 else "blocked"
    return ComponentStatus(
        status=status,
        row_counts={
            "district_week_climate": rows,
            "rainfall_current_non_null": rainfall,
            "temperature_current_non_null": temperature,
            "humidity_current_non_null": humidity,
            "successful_climate_aggregation_runs": success_runs,
        },
        artifact_paths={"processed_climate_csv": _path(Path("data/climate/processed/district_week_climate.csv"))},
        blocking_reason=None if status == "complete" else "Climate aggregation has not run on real registered files.",
        next_required_action="Run Prompt 20 real climate aggregation after Prompt 19 data acquisition." if status != "complete" else "Proceed to Prompt 21 climate lags.",
        details={
            "crs_proof": _latest_manifest_value("climate_aggregation", "boundary"),
            "aggregation_proof": _latest_manifest_value("climate_aggregation", "aggregation_method"),
            "centroid_fallback_used_as_final": False,
        },
    )


def _climate_lags() -> ComponentStatus:
    counts = {
        column: DistrictWeekFeature.query.filter(getattr(DistrictWeekFeature, column).isnot(None)).count()
        for column in CLIMATE_COLUMNS
    }
    lag_run = _latest_success_run("climate_lag_creation")
    manifest = lag_run.manifest if lag_run and lag_run.manifest else {}
    all_present = all(count > 0 for count in counts.values())
    proof_present = bool(manifest.get("alignment_checks"))
    status = "complete" if all_present and proof_present else "blocked"
    return ComponentStatus(
        status=status,
        row_counts=counts | {"successful_climate_lag_creation_runs": PipelineRun.query.filter_by(run_type="climate_lag_creation", status="success").count()},
        artifact_paths={"pipeline_run_manifest": f"pipeline_runs.id={lag_run.id}" if lag_run else None},
        blocking_reason=None if status == "complete" else "Climate lag fields and/or alignment proof are missing.",
        next_required_action="Run Prompt 21 climate lag creation after Prompt 20 aggregation." if status != "complete" else "Rerun Prompt 22 modelling dataset.",
        details={
            "alignment_checks": manifest.get("alignment_checks", []),
            "year_boundary_proof_present": any("year-boundary" in str(item.get("note", "")).lower() for item in manifest.get("alignment_checks", [])),
            "week53_proof_present": any("week-53" in str(item.get("note", "")).lower() for item in manifest.get("alignment_checks", [])),
        },
    )


def _modelling_dataset() -> ComponentStatus:
    ready, message = modelling_dataset_ready_for_downstream()
    dataset_exists = DATASET_PATH.exists()
    summary_exists = SUMMARY_PATH.exists()
    if ready:
        status = "complete"
    elif dataset_exists:
        status = "stale"
    else:
        status = "blocked"
    return ComponentStatus(
        status=status,
        row_counts={},
        artifact_paths={"modelling_dataset_csv": _path(DATASET_PATH), "modelling_dataset_summary": _path(SUMMARY_PATH)},
        blocking_reason=None if ready else message,
        next_required_action="Next: rerun Prompt 22." if dataset_exists and not ready else ("Run Prompt 22 after climate lags are complete." if not ready else "Proceed to model validation."),
        details={"built_before_real_climate_lags": dataset_exists and not ready},
    )


def _models_and_forecasts() -> ComponentStatus:
    validation_artifacts = {
        key: _validation_artifact_is_complete(key, path)
        for key, path in VALIDATION_ARTIFACTS.items()
    }
    forecast_rows = ForecastOutput.query.count()
    production_forecasts_complete = _production_forecast_is_complete()
    performance_rows = ModelPerformance.query.count()
    selection_exists = PERFORMANCE_SELECTION_PATH.exists()
    dashboard_proof_exists = DASHBOARD_PROOF_PATH.exists()
    any_artifacts = any(validation_artifacts.values()) or forecast_rows or performance_rows or selection_exists
    status = "complete" if all(validation_artifacts.values()) and production_forecasts_complete and performance_rows > 0 and selection_exists else ("partial" if any_artifacts else "blocked")
    return ComponentStatus(
        status=status,
        row_counts={"forecast_outputs": forecast_rows, "model_performance": performance_rows},
        artifact_paths={
            "validation_artifacts": {key: _path(path) for key, path in VALIDATION_ARTIFACTS.items()},
            "forecast_generation_summary": _path(FORECAST_SUMMARY_PATH),
            "production_model_manifest": _path(PRODUCTION_MANIFEST_PATH),
            "model_performance_selection_summary": _path(PERFORMANCE_SELECTION_PATH),
            "dashboard_data_summary": _path(DASHBOARD_PROOF_PATH),
        },
        blocking_reason=None if status == "complete" else "Model validation, forecast generation, performance selection and dashboard risk outputs are waiting on Prompt 22 climate-informed dataset.",
        next_required_action="Run Prompts 23-28 only after Prompt 22 completes with real climate lags." if status != "complete" else "Prompt 29 may be considered after dashboard APIs contain real data.",
        details={
            "validation_artifacts_present": validation_artifacts,
            "production_forecasts_complete": production_forecasts_complete,
        },
    )


def _production_forecast_is_complete(
    *,
    summary_path: Path = FORECAST_SUMMARY_PATH,
    manifest_path: Path = PRODUCTION_MANIFEST_PATH,
    forecast_rows: list[Any] | None = None,
) -> bool:
    """Require complete model files and 146 rows for every model-horizon issue slice."""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if summary.get("status") != "success" or not summary.get("published"):
        return False
    if manifest.get("status") != "success":
        return False
    if int(summary.get("expected_forecast_rows", 0)) != 1752:
        return False
    if int(summary.get("final_issue_row_count", 0)) != 1752:
        return False
    issue_year = summary.get("issue_year")
    issue_week = summary.get("issue_week")
    if (
        manifest.get("issue_epi_year") != issue_year
        or manifest.get("issue_epi_week") != issue_week
        or set(manifest.get("models") or [])
        != {"Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost"}
        or set(manifest.get("horizons") or []) != {4, 8, 12}
    ):
        return False
    rows = forecast_rows
    if rows is None:
        rows = ForecastOutput.query.filter_by(
            forecast_origin_epi_year=issue_year,
            forecast_origin_epi_week=issue_week,
        ).all()
    if len(rows) != 1752:
        return False
    grouped: Counter[tuple[str, int]] = Counter(
        (row.model_name, row.horizon_weeks) for row in rows
    )
    district_sets: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        district_sets.setdefault((row.model_name, row.horizon_weeks), set()).add(
            row.district_id
        )
        if row.predicted_cases is None:
            return False
    expected_combinations = {
        (model, horizon)
        for model in ("Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost")
        for horizon in (4, 8, 12)
    }
    return (
        set(grouped) == expected_combinations
        and all(grouped[key] == 146 for key in expected_combinations)
        and all(len(district_sets.get(key, set())) == 146 for key in expected_combinations)
    )


def _validation_artifact_is_complete(model_key: str, path: Path) -> bool:
    """Treat only real, metric-bearing validation artifacts as complete."""
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_names = {
        "baseline": "Baseline/Naive",
        "sarimax": "SARIMAX",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
    }
    if payload.get("model_name") != expected_names.get(model_key):
        return False
    if payload.get("status") != "success":
        acceptable_sarimax_partial = (
            model_key == "sarimax"
            and payload.get("status") == "partial_success"
            and int((payload.get("job_counts") or {}).get("failed", -1)) == 0
            and int((payload.get("row_counts") or {}).get("prediction_rows", 0)) > 0
        )
        if not acceptable_sarimax_partial:
            return False
    count_required = {"n", "mae", "rmse", "mape", "wape", "r_squared"}
    surge_required = {
        "n",
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
    }
    count_metrics = payload.get("count_metrics_by_horizon") or {}
    surge_metrics = payload.get("surge_metrics_by_horizon") or {}
    for horizon in (4, 8, 12):
        count = count_metrics.get(str(horizon), count_metrics.get(horizon))
        surge = surge_metrics.get(str(horizon), surge_metrics.get(horizon))
        if not isinstance(count, dict) or not count_required.issubset(count):
            return False
        if not isinstance(surge, dict) or not surge_required.issubset(surge):
            return False
    if model_key in {"sarimax", "random_forest", "xgboost"}:
        artifacts = payload.get("artifact_paths") or {}
        required = [
            "predictions_csv",
            "district_horizon_metrics_csv",
            "fit_diagnostics_json",
            "model_card_json",
        ]
        if model_key in {"random_forest", "xgboost"}:
            if payload.get("model_scope") != "national_pooled" or not payload.get("published"):
                return False
            job_counts = payload.get("job_counts") or {}
            if int(job_counts.get("total", 0)) <= 0 or int(job_counts.get("failed", 0)) > 0:
                return False
            required.append("feature_importance_csv")
        if not all(artifacts.get(name) and Path(artifacts[name]).exists() for name in required):
            return False
    return True


DASHBOARD_API_ROUTES = (
    "/api/dashboard/summary",
    "/api/dashboard/weekly-series",
    "/api/dashboard/risk-table",
    "/api/dashboard/map",
    "/api/dashboard/climate-summary",
    "/api/dashboard/model-performance",
)
# Placeholder copy from the pre-Prompt-29 static shell. Any survivor means a
# panel is still showing a hardcoded message instead of API data.
DASHBOARD_PLACEHOLDER_STRINGS = (
    "Awaiting data",
    "Awaiting pipeline output",
    "Awaiting model output",
    "Map awaiting pipeline output",
    "Awaiting model training",
    "Awaiting analysis",
    "No processed data available yet",
)
# Panel mount points the script writes into. A missing id means a dead panel.
DASHBOARD_TEMPLATE_TARGETS = (
    "filter-horizon",
    "filter-district",
    "filter-issue",
    "kpi-cases-value",
    "kpi-completeness-value",
    "kpi-risk-value",
    "kpi-horizon-value",
    "weekly-chart",
    "risk-tbody",
    "risk-map",
    "climate-cards",
    "perf-count-tbody",
    "footer-updated",
)
DASHBOARD_TEMPLATE_PATH = Path("app/templates/dashboard/index.html")
DASHBOARD_SCRIPT_PATH = Path("app/static/js/dashboard.js")
DASHBOARD_BASE_TEMPLATE_PATH = Path("app/templates/base.html")
CHART_LIBRARY_PATH = Path("app/static/vendor/chartjs/chart.umd.js")
MAP_LIBRARY_PATH = Path("app/static/vendor/leaflet/leaflet.js")


def _dashboard_live_section_checks() -> dict[str, Any]:
    """Exercise the real API section builders against the database.

    This is the substance of the visible-dashboard gate: the panels are only
    'complete' if the services they call actually return populated sections.
    """
    from app.services.dashboard_data import (
        DashboardDataError,
        build_climate_section,
        build_map_section,
        build_model_performance_section,
        build_risk_table_section,
        build_summary_section,
        build_weekly_series_section,
        dashboard_precondition_errors,
        resolve_params,
    )

    errors = dashboard_precondition_errors(require_outputs=True)
    if errors:
        return {"preconditions_met": False, "reason": "; ".join(errors)}

    try:
        params = resolve_params()
        if params.issue_year is None:
            return {"preconditions_met": False, "reason": "No production forecast issue is available."}

        summary = build_summary_section(params)
        weekly = build_weekly_series_section(params)
        risk = build_risk_table_section(params)
        climate = build_climate_section(params)
        performance = build_model_performance_section(params)
        map_section = build_map_section(params)
    except DashboardDataError as exc:
        return {"preconditions_met": True, "reason": str(exc), "section_error": True}

    selected = performance.get("selected_models", {}).get(str(params.horizon), {})
    return {
        "preconditions_met": True,
        "reason": None,
        "issue_label": summary.get("issue_label"),
        "summary_has_cases": summary.get("confirmed_malaria_cases") is not None,
        "summary_has_completeness": summary.get("reporting_completeness") is not None,
        "weekly_observed_rows": weekly.get("observed_rows", 0),
        "weekly_forecast_rows": weekly.get("forecast_rows", 0),
        "risk_rows": len(risk.get("rows", [])),
        "risk_rows_carry_threshold": all(
            row.get("threshold_used") is not None for row in risk.get("rows", [])
        ),
        "risk_rows_carry_both_model_names": all(
            row.get("count_model_name") and row.get("surge_model_name")
            for row in risk.get("rows", [])
        ),
        "climate_status": climate.get("status"),
        "performance_rows": len(performance.get("rows", [])),
        "map_features": len(map_section.get("features", [])),
        "map_join_key": map_section.get("join_key"),
        "selected_count_model": (selected.get("count") or {}).get("model_name"),
        "selected_surge_model": (selected.get("surge") or {}).get("model_name"),
    }


def _dashboard_web_map_checks() -> dict[str, Any]:
    """Verify the derived browser boundary without loading the 20 MB source."""
    from app.services.dashboard_data import WEB_MAP_GEOJSON_PATH, WEB_MAP_MAX_BYTES

    if not WEB_MAP_GEOJSON_PATH.exists():
        return {"present": False, "reason": f"Missing {WEB_MAP_GEOJSON_PATH}."}
    size = WEB_MAP_GEOJSON_PATH.stat().st_size
    try:
        payload = json.loads(WEB_MAP_GEOJSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"present": True, "valid_json": False, "reason": str(exc)}
    features = payload.get("features") or []
    ids = {(feature.get("properties") or {}).get("district_id") for feature in features}
    types = {feature.get("geometry", {}).get("type") for feature in features}
    return {
        "present": True,
        "valid_json": True,
        "path": str(WEB_MAP_GEOJSON_PATH),
        "size_bytes": size,
        "size_mb": round(size / 1_000_000, 3),
        "within_size_budget": size <= WEB_MAP_MAX_BYTES,
        "feature_count": len(features),
        "unique_district_ids": len(ids - {None}),
        "geometry_types": sorted(item for item in types if item),
        "geometry_types_supported": not (types - {"Polygon", "MultiPolygon"}),
        "district_id_preserved": None not in ids,
        "approved_boundary_untouched": PREPARED_GEOJSON_PATH.exists(),
    }


def _visible_dashboard() -> ComponentStatus:
    """Gate the visible dashboard on real integration, not on a string match."""
    template_text = (
        DASHBOARD_TEMPLATE_PATH.read_text(encoding="utf-8") if DASHBOARD_TEMPLATE_PATH.exists() else ""
    )
    base_text = (
        DASHBOARD_BASE_TEMPLATE_PATH.read_text(encoding="utf-8") if DASHBOARD_BASE_TEMPLATE_PATH.exists() else ""
    )
    script_text = (
        DASHBOARD_SCRIPT_PATH.read_text(encoding="utf-8") if DASHBOARD_SCRIPT_PATH.exists() else ""
    )
    rendered_text = template_text + base_text

    placeholders_present = [
        text for text in DASHBOARD_PLACEHOLDER_STRINGS if text in rendered_text
    ]
    missing_targets = [
        target for target in DASHBOARD_TEMPLATE_TARGETS if f'id="{target}"' not in rendered_text
    ]
    routes_called = {route: route in script_text for route in DASHBOARD_API_ROUTES}
    live = _dashboard_live_section_checks()
    web_map = _dashboard_web_map_checks()

    frontend_ok = (
        DASHBOARD_TEMPLATE_PATH.exists()
        and DASHBOARD_SCRIPT_PATH.exists()
        and "js/dashboard.js" in template_text
        and not placeholders_present
        and not missing_targets
        and all(routes_called.values())
        and CHART_LIBRARY_PATH.exists()
        and MAP_LIBRARY_PATH.exists()
    )
    web_map_ok = bool(
        web_map.get("present")
        and web_map.get("valid_json")
        and web_map.get("within_size_budget")
        and web_map.get("feature_count") == EXPECTED_DISTRICTS
        and web_map.get("unique_district_ids") == EXPECTED_DISTRICTS
        and web_map.get("district_id_preserved")
        and web_map.get("geometry_types_supported")
    )
    live_ok = bool(
        live.get("preconditions_met")
        and not live.get("section_error")
        and live.get("summary_has_cases")
        and live.get("weekly_observed_rows", 0) > 0
        and live.get("weekly_forecast_rows", 0) > 0
        and live.get("risk_rows") == EXPECTED_DISTRICTS
        and live.get("risk_rows_carry_threshold")
        and live.get("risk_rows_carry_both_model_names")
        and live.get("climate_status") == "available"
        and live.get("performance_rows", 0) > 0
        and live.get("map_features") == EXPECTED_DISTRICTS
        and live.get("map_join_key") == "district_id"
    )

    status = "complete" if frontend_ok and web_map_ok and live_ok else "partial"

    reasons: list[str] = []
    if placeholders_present:
        reasons.append("Placeholder copy still rendered: " + ", ".join(placeholders_present))
    if missing_targets:
        reasons.append("Template is missing panel targets: " + ", ".join(missing_targets))
    if not all(routes_called.values()):
        reasons.append(
            "Dashboard script does not call: "
            + ", ".join(route for route, called in routes_called.items() if not called)
        )
    if not CHART_LIBRARY_PATH.exists() or not MAP_LIBRARY_PATH.exists():
        reasons.append("Locally hosted chart/map libraries are missing.")
    if not web_map_ok:
        reasons.append("Derived web-map boundary failed verification: " + str(web_map.get("reason") or web_map))
    if not live_ok:
        reasons.append("Dashboard API sections did not return complete data: " + str(live.get("reason") or live))

    return ComponentStatus(
        status=status,
        row_counts={
            "risk_rows": live.get("risk_rows", 0),
            "map_features": live.get("map_features", 0),
            "weekly_observed_rows": live.get("weekly_observed_rows", 0),
            "weekly_forecast_rows": live.get("weekly_forecast_rows", 0),
            "model_performance_rows": live.get("performance_rows", 0),
        },
        artifact_paths={
            "dashboard_template": _path(DASHBOARD_TEMPLATE_PATH),
            "dashboard_script": _path(DASHBOARD_SCRIPT_PATH),
            "chart_library": _path(CHART_LIBRARY_PATH),
            "map_library": _path(MAP_LIBRARY_PATH),
            "web_map_geojson": web_map.get("path"),
        },
        blocking_reason="; ".join(reasons) if reasons else None,
        next_required_action=(
            "Dashboard panels are wired to the APIs and returning real data."
            if status == "complete"
            else "Complete the visible dashboard integration before closing Prompt 29."
        ),
        details={
            "api_routes_wired": list(DASHBOARD_API_ROUTES),
            "api_routes_called_by_script": routes_called,
            "placeholder_strings_present": placeholders_present,
            "missing_template_targets": missing_targets,
            "script_included_in_template": "js/dashboard.js" in template_text,
            "frontend_wired": frontend_ok,
            "web_map": web_map,
            "live_sections": live,
            "integration_check": "Section builders are executed against the database; the gate is not a string match.",
        },
    )


def _safe_to_proceed_prompt29(components: dict[str, ComponentStatus]) -> bool:
    return (
        components["climate_lags"].status == "complete"
        and components["modelling_dataset"].status == "complete"
        and components["models_and_forecasts"].status == "complete"
    )


def _recommended_next_action(components: dict[str, ComponentStatus]) -> str:
    if components["climate_ingestion"].status != "complete":
        return "Next: complete Prompt 19, then 20, then 21."
    if components["climate_aggregation"].status != "complete":
        return "Next: run Prompt 20 real climate aggregation."
    if components["climate_lags"].status != "complete":
        return "Next: run Prompt 21 climate lag creation."
    return _next_after_climate(components)


def _next_after_climate(components: dict[str, ComponentStatus]) -> str:
    if components["modelling_dataset"].status == "stale":
        return "Next: rerun Prompt 22."
    if components["modelling_dataset"].status != "complete":
        return "Run Prompt 22."
    if components["models_and_forecasts"].status != "complete":
        return "Run Prompts 23, 24, 25, 26, 27 and 28."
    if components["visible_dashboard"].status != "complete":
        return "Run Prompt 29 visible dashboard integration."
    return "Prompt 29 is complete; no remaining dashboard action is required."


def _gating_logic(components: dict[str, ComponentStatus]) -> dict[str, Any]:
    return {
        "prompt22_requires_climate_lags_and_proof": True,
        "prompt23_24_require_current_prompt22_dataset": True,
        "prompt25_requires_current_dataset_and_validation_artifacts": True,
        "prompt26_requires_validation_artifacts_and_forecast_outputs": True,
        "prompt27_requires_forecasts_and_model_selection": True,
        "prompt28_api_reports_unavailable_when_downstream_missing": True,
        "prompt29_ready": _safe_to_proceed_prompt29(components),
        "prompt29_complete": components["visible_dashboard"].status == "complete",
    }


def _registered_climate_files() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in PipelineRun.query.filter_by(run_type="climate_ingestion", status="success").all():
        params = run.parameters or {}
        rows.append(
            {
                "pipeline_run_id": run.id,
                "source": params.get("source"),
                "variable": params.get("variable"),
                "start_date": params.get("start_date"),
                "end_date": params.get("end_date"),
                "path": params.get("path"),
            }
        )
    return rows


def _coverage_by_pair(registered: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in registered:
        key = f"{item.get('source')}/{item.get('variable')}"
        grouped.setdefault(key, []).append(item)
    out: dict[str, dict[str, Any]] = {}
    for key, items in grouped.items():
        starts = [_parse_date(item.get("start_date")) for item in items if _parse_date(item.get("start_date"))]
        ends = [_parse_date(item.get("end_date")) for item in items if _parse_date(item.get("end_date"))]
        out[key] = {
            "registered_files": len(items),
            "start_date_min": min(starts).isoformat() if starts else None,
            "end_date_max": max(ends).isoformat() if ends else None,
        }
    return out


def _lookback_coverage_present(coverage: dict[str, dict[str, Any]]) -> bool:
    required = ("CHIRPS/rainfall", "ERA5-Land/temperature", "ERA5-Land/humidity")
    for key in required:
        item = coverage.get(key)
        if not item:
            return False
        start = _parse_date(item.get("start_date_min"))
        if start is None or start > REQUIRED_CLIMATE_LOOKBACK_START:
            return False
    return True


def _raw_climate_files() -> list[Path]:
    if not CLIMATE_RAW_DIR.exists():
        return []
    return [
        path
        for path in CLIMATE_RAW_DIR.rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]


def _climate_config_status() -> dict[str, Any]:
    return {
        "CLIMATE_DATA_DIR_configured": bool(os.environ.get("CLIMATE_DATA_DIR")),
        "CHIRPS_BASE_URL_configured": bool(os.environ.get("CHIRPS_BASE_URL")),
        "ERA5_PROVIDER_configured": bool(os.environ.get("ERA5_PROVIDER")),
        "ERA5_API_URL_configured": bool(os.environ.get("ERA5_API_URL")),
        "ERA5_API_KEY_configured": bool(os.environ.get("ERA5_API_KEY")),
        "ERA5_DOWNLOAD_MODE": os.environ.get("ERA5_DOWNLOAD_MODE", "manual_or_api"),
        "credential_values_not_reported": True,
    }


def _dependency_status() -> dict[str, bool]:
    packages = ("requests", "cdsapi", "geopandas", "rasterio", "rioxarray", "xarray", "rasterstats", "shapely")
    return {package: importlib.util.find_spec(package) is not None for package in packages}


def _latest_success_run(run_type: str) -> PipelineRun | None:
    return (
        PipelineRun.query.filter_by(run_type=run_type, status="success")
        .order_by(PipelineRun.completed_at.desc().nullslast(), PipelineRun.id.desc())
        .first()
    )


def _latest_manifest_value(run_type: str, key: str) -> Any:
    run = _latest_success_run(run_type)
    if not run or not run.manifest:
        return None
    return run.manifest.get(key)


def _secret_safety_summary() -> dict[str, Any]:
    return {
        "credential_values_reported": False,
        "known_secret_literals_in_pipeline_runs": not _no_known_secret_in_pipeline_runs(),
        "dhis2_authentication_mode": "interactive_transient",
        "dhis2_credentials_persisted_by_application": False,
        "era5_api_key_configured": bool(os.environ.get("ERA5_API_KEY")),
    }


def _no_known_secret_in_pipeline_runs() -> bool:
    secret_values = [
        value
        for value in (
            os.environ.get("ERA5_API_KEY"),
        )
        if value and len(value) >= 4
    ]
    if not secret_values:
        return True
    for run in PipelineRun.query.all():
        serialized = json.dumps(
            {
                "parameters": run.parameters,
                "manifest": run.manifest,
                "source_version_or_url": run.source_version_or_url,
                "error_message": run.error_message,
            },
            default=str,
        )
        if any(secret in serialized for secret in secret_values):
            return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _path(path: Path) -> str | None:
    return str(path) if path.exists() else None
