"""Phase 29D report context builders.

Every report context is assembled by calling the approved server-side services
directly (dashboard_data, explainability, project_state). There are no internal
HTTP calls back into Flask, no new analytical definitions, no recalculated
winners/thresholds/classifications, and no database writes.

A missing or malformed required artifact fails the affected report closed with
``ReportUnavailableError`` so a partially authoritative document is never
produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.services import dashboard_data as dd
from app.services import explainability as ex

# --------------------------------------------------------------------------
# Registry: report types, labels, filters and the ONLY supported formats.
# The frontend renders formats from this registry, so an unsupported button
# can never appear.
# --------------------------------------------------------------------------
FORMAT_PDF = "pdf"
FORMAT_XLSX = "xlsx"
FORMAT_CSV = "csv"
FORMAT_HTML = "html"

RISK_LEVELS = ("Very High", "High", "Moderate", "Low", "No data")

REPORTS: dict[str, dict[str, Any]] = {
    "national_summary": {
        "title": "National Early Warning Summary",
        "purpose": "Concise national picture for the selected issue week and forecast horizon.",
        "filters": ["issue", "horizon"],
        "formats": [FORMAT_PDF, FORMAT_HTML, FORMAT_XLSX],
        "orientation": "portrait",
    },
    "district_forecast": {
        "title": "District Forecast Summary",
        "purpose": "Focused surveillance, forecast and risk summary for one district.",
        "filters": ["issue", "horizon", "district_required"],
        "formats": [FORMAT_PDF, FORMAT_HTML, FORMAT_XLSX],
        "orientation": "portrait",
    },
    "district_risk_list": {
        "title": "District Risk List",
        "purpose": "Structured operational list of district risk classifications.",
        "filters": ["issue", "horizon", "risk_level", "district_optional"],
        "formats": [FORMAT_XLSX, FORMAT_CSV, FORMAT_HTML, FORMAT_PDF],
        "orientation": "landscape",
    },
    "model_performance": {
        "title": "Model Performance Summary",
        "purpose": "How the four approved models performed and how models were selected.",
        "filters": ["issue", "horizon"],
        "formats": [FORMAT_PDF, FORMAT_XLSX, FORMAT_HTML],
        "orientation": "landscape",
    },
    "data_quality": {
        "title": "Data Quality Summary",
        "purpose": "Quality checks applied to surveillance data before modelling.",
        "filters": ["issue"],
        "formats": [FORMAT_PDF, FORMAT_XLSX, FORMAT_HTML],
        "orientation": "portrait",
    },
    "climate_signals": {
        "title": "Climate Signals Summary",
        "purpose": "Climate predictors and lags used by the forecasting models.",
        "filters": ["issue", "horizon", "district_optional"],
        "formats": [FORMAT_PDF, FORMAT_XLSX, FORMAT_HTML],
        "orientation": "portrait",
    },
    "methods": {
        "title": "Methods and Interpretation Summary",
        "purpose": "Research-methods summary suitable for supervision and system review.",
        "filters": ["issue"],
        "formats": [FORMAT_PDF, FORMAT_HTML],
        "orientation": "portrait",
    },
}

RESEARCH_USE_STATEMENT = (
    "Research proof-of-concept. These outputs support review and early action. "
    "They do not replace epidemiological judgement or official outbreak confirmation."
)

CORE_LIMITATIONS = [
    "A High or Very High risk level is an early-warning signal for review; it does not confirm an outbreak.",
    "DHIS2 reporting quality affects forecasts; weeks with low reporting completeness may understate cases.",
    "Missing malaria reports remain missing and are never treated as zero.",
    "HMIS 033b reporting completeness measures form submission, not malaria burden.",
    "Climate variables are predictive inputs associated with malaria patterns; they do not establish causation.",
    "The selected model predicts the target week only; intermediate weekly forecasts are not generated.",
    "Confirmed reporting errors are excluded from outcomes only after documented district consultation; raw values remain auditable.",
    "Model quality is judged on error and surge-detection metrics together; accuracy alone is not used.",
]

DATA_SOURCES = [
    "DHIS2 HMIS 033b confirmed malaria cases (data element fUflbWWhouR, code 033B-CD01a).",
    "HMIS 033b facility reporting completeness.",
    "CHIRPS rainfall (millimetres, weekly total).",
    "ERA5-Land temperature (degrees Celsius, weekly mean).",
    "ERA5-Land humidity (relative humidity percent, derived from temperature and dewpoint).",
    "UBOS district boundaries.",
]


class ReportError(RuntimeError):
    """Base report failure."""


class ReportParameterError(ValueError):
    """Invalid caller parameter (maps to HTTP 400)."""


class ReportUnavailableError(ReportError):
    """Required evidence missing/malformed (maps to HTTP 503)."""


@dataclass
class ReportRequest:
    report_type: str
    fmt: str
    horizon: int = dd.DEFAULT_HORIZON
    district_id: str | None = None
    risk_level: str | None = None
    issue_year: int | None = None
    issue_week: int | None = None

    @property
    def spec(self) -> dict[str, Any]:
        return REPORTS[self.report_type]


def validate_request(args) -> ReportRequest:
    """Strict allowlist validation. Never accepts a path or arbitrary filename."""
    report_type = (args.get("report_type") or "").strip()
    if report_type not in REPORTS:
        raise ReportParameterError("Unknown report type.")
    spec = REPORTS[report_type]

    fmt = (args.get("format") or "").strip().lower()
    if fmt not in spec["formats"]:
        raise ReportParameterError("The requested export format is not supported for this report.")

    horizon_raw = (args.get("horizon") or str(dd.DEFAULT_HORIZON)).strip()
    try:
        horizon = int(horizon_raw)
    except ValueError:
        raise ReportParameterError("horizon must be one of 4, 8, 12.") from None
    if horizon not in dd.HORIZONS:
        raise ReportParameterError("horizon must be one of 4, 8, 12.")

    district_raw = (args.get("district_id") or "").strip()
    district_id: str | None = None
    if district_raw and district_raw.lower() != dd.ALL_DISTRICTS:
        from app.models import DistrictReference

        if not DistrictReference.query.filter_by(district_id=district_raw, active=True).count():
            raise ReportParameterError("Unknown district.")
        district_id = district_raw
    if "district_required" in spec["filters"] and not district_id:
        raise ReportParameterError("This report requires a district selection.")

    risk_level = None
    risk_raw = (args.get("risk_level") or "").strip()
    if risk_raw and risk_raw.lower() != "all":
        if risk_raw not in RISK_LEVELS:
            raise ReportParameterError("Unknown risk level.")
        risk_level = risk_raw

    year_raw = (args.get("issue_year") or "").strip()
    week_raw = (args.get("issue_week") or "").strip()
    issue_year = issue_week = None
    if year_raw or week_raw:
        if not (year_raw and week_raw):
            raise ReportParameterError("issue_year and issue_week must be supplied together.")
        try:
            issue_year, issue_week = int(year_raw), int(week_raw)
        except ValueError:
            raise ReportParameterError("issue_year and issue_week must be integers.") from None
        known = {(i["issue_year"], i["issue_week"]) for i in dd.available_issues()}
        if (issue_year, issue_week) not in known:
            raise ReportParameterError("The requested production issue does not exist.")

    return ReportRequest(
        report_type=report_type, fmt=fmt, horizon=horizon, district_id=district_id,
        risk_level=risk_level, issue_year=issue_year, issue_week=issue_week,
    )


# --------------------------------------------------------------------------
# Shared metadata
# --------------------------------------------------------------------------
def _params(req: ReportRequest) -> dd.DashboardParams:
    params = dd.resolve_params(
        horizon=req.horizon, district_id=req.district_id,
        issue_year=req.issue_year, issue_week=req.issue_week,
    )
    if params.issue_year is None or params.issue_week is None:
        raise ReportUnavailableError("No production forecast issue is available.")
    return params


def _metadata(req: ReportRequest, params: dd.DashboardParams, *, target_label=None,
              count_model=None, surge_model=None, district_name=None) -> dict[str, Any]:
    from flask import current_app

    spec = req.spec
    evidence_generated = None
    dataset_signature = None
    try:
        forecast = ex.load_json("forecast_generation")
        evidence_generated = forecast.get("generated_at_utc")
    except ex.ExplainabilityError:
        evidence_generated = None
    try:
        manifest = ex.load_json("production_manifest")
        sig = (manifest.get("dataset_signature") or {}).get("sha256")
        if isinstance(sig, str):
            dataset_signature = sig[:16]  # short, non-path, non-secret
    except ex.ExplainabilityError:
        dataset_signature = None

    return {
        "report_title": spec["title"],
        "report_type": req.report_type,
        "issue_label": dd.issue_label(params.issue_year, params.issue_week),
        "issue_year": params.issue_year,
        "issue_week": params.issue_week,
        "horizon_weeks": params.horizon if "horizon" in spec["filters"] else None,
        "target_label": target_label,
        "district_id": params.district_id,
        "district_name": district_name,
        "count_model_name": count_model,
        "surge_model_name": surge_model,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "application_version": current_app.config.get("APP_VERSION"),
        "application_name": current_app.config.get("APP_NAME"),
        "report_schema_version": "29D.1",
        "data_sources": list(DATA_SOURCES),
        "research_use": RESEARCH_USE_STATEMENT,
        "evidence_generated_at_utc": evidence_generated,
        "dataset_signature_short": dataset_signature,
    }


def _selected(params: dd.DashboardParams) -> dict[str, Any]:
    try:
        return dd.selected_models(params.horizon)
    except Exception as exc:  # noqa: BLE001 - controlled unavailable
        raise ReportUnavailableError("Model-selection evidence is unavailable.") from exc


# --------------------------------------------------------------------------
# Report contexts
# --------------------------------------------------------------------------
def _guard(fn, message):
    try:
        return fn()
    except (ex.ExplainabilityError, dd.DashboardDataError) as exc:
        raise ReportUnavailableError(message) from exc
    except ReportError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ReportUnavailableError(message) from exc


def build_context(req: ReportRequest) -> dict[str, Any]:
    """Build the full report context for a validated request."""
    errors = dd.dashboard_precondition_errors(require_outputs=True)
    if errors:
        raise ReportUnavailableError("Validated forecast evidence is unavailable.")
    builder = _BUILDERS[req.report_type]
    return builder(req)


def _national_summary(req: ReportRequest) -> dict[str, Any]:
    params = _params(req)
    models = _selected(params)
    summary = _guard(lambda: dd.build_summary_section(params), "National summary evidence is unavailable.")
    risk = _guard(lambda: dd.build_risk_table_section(params), "District risk evidence is unavailable.")
    weekly = _guard(lambda: dd.build_weekly_series_section(params), "Weekly series evidence is unavailable.")

    rows = risk["rows"]
    counts = risk.get("risk_counts", {})
    priority = sorted(
        rows,
        key=lambda r: (
            {"Very High": 4, "High": 3, "Moderate": 2, "Low": 1, "No data": 0}.get(r["risk_level"], 0),
            r["predicted_surge_score"] if r["predicted_surge_score"] is not None else -1e12,
        ),
        reverse=True,
    )[:25]

    meta = _metadata(req, params, target_label=summary.get("target_label"),
                     count_model=models["count"]["model_name"], surge_model=models["surge"]["model_name"])
    executive = [
        f"Confirmed malaria cases in the issue week: {_fmt(summary.get('confirmed_malaria_cases'))}.",
        f"Average HMIS 033b reporting completeness: {_fmt(summary.get('reporting_completeness'), 1)}%.",
        f"Districts in scope: {_fmt(summary.get('districts_in_scope'))}.",
        f"Districts at High risk: {counts.get('High', 0)}.",
        f"Districts at Very High risk: {counts.get('Very High', 0)}.",
        f"Selected count model: {models['count']['model_name']}.",
        f"Selected surge model: {models['surge']['model_name']}.",
        f"Target week: {summary.get('target_label') or 'unavailable'}.",
    ]
    return {
        "metadata": meta, "summary": summary, "risk_counts": counts,
        "priority_districts": priority, "weekly": weekly, "all_rows": rows,
        "executive_summary": executive, "selected_models": models,
        "limitations": list(CORE_LIMITATIONS),
    }


def _district_forecast(req: ReportRequest) -> dict[str, Any]:
    params = _params(req)
    models = _selected(params)
    summary = _guard(lambda: dd.build_summary_section(params), "District summary evidence is unavailable.")
    risk = _guard(lambda: dd.build_risk_table_section(params), "District risk evidence is unavailable.")
    weekly = _guard(lambda: dd.build_weekly_series_section(params), "Weekly series evidence is unavailable.")
    climate = _guard(lambda: dd.build_climate_section(params), "Climate evidence is unavailable.")

    rows = risk["rows"]
    row = next((r for r in rows if r["district_id"] == params.district_id), None)
    if row is None:
        raise ReportUnavailableError("No published risk row for the selected district.")

    # Deterministic interpretation built ONLY from published values + backend risk_basis.
    score = row["predicted_surge_score"]
    interpretation = [
        f"{row['district_name']} is classified as {row['risk_level']} risk for the "
        f"{params.horizon}-week horizon."
    ]
    if score is not None:
        if score > 0:
            interpretation.append(
                f"The surge-model prediction exceeds the district threshold by {_fmt(abs(score))} cases."
            )
        elif score < 0:
            interpretation.append(
                f"The surge-model prediction remains {_fmt(abs(score))} cases below the district threshold."
            )
        else:
            interpretation.append("The surge-model prediction equals the district threshold.")
    if row.get("risk_basis"):
        interpretation.append(row["risk_basis"])

    meta = _metadata(req, params, target_label=row.get("target_label"),
                     count_model=row.get("count_model_name"), surge_model=row.get("surge_model_name"),
                     district_name=row.get("district_name"))
    return {
        "metadata": meta, "summary": summary, "district": row, "weekly": weekly,
        "climate": climate, "interpretation": interpretation, "selected_models": models,
        "limitations": list(CORE_LIMITATIONS),
    }


def _district_risk_list(req: ReportRequest) -> dict[str, Any]:
    params = _params(req)
    models = _selected(params)
    # The list is always national in scope; a district filter narrows rows only.
    national = dd.resolve_params(horizon=params.horizon, issue_year=params.issue_year,
                                 issue_week=params.issue_week)
    risk = _guard(lambda: dd.build_risk_table_section(national), "District risk evidence is unavailable.")
    rows = risk["rows"]
    if req.risk_level:
        rows = [r for r in rows if r["risk_level"] == req.risk_level]
    if req.district_id:
        rows = [r for r in rows if r["district_id"] == req.district_id]

    meta = _metadata(req, params,
                     target_label=rows[0]["target_label"] if rows else None,
                     count_model=models["count"]["model_name"], surge_model=models["surge"]["model_name"])
    meta["risk_level_filter"] = req.risk_level or "All"
    return {
        "metadata": meta, "rows": rows, "risk_counts": risk.get("risk_counts", {}),
        "selected_models": models, "score_definition": risk.get("score_definition"),
        "limitations": list(CORE_LIMITATIONS),
    }


def _model_performance(req: ReportRequest) -> dict[str, Any]:
    params = _params(req)
    perf = _guard(lambda: dd.build_model_performance_section(params), "Model performance evidence is unavailable.")
    selection = _guard(lambda: ex.build_model_selection(params.horizon), "Model-selection evidence is unavailable.")
    validation = _guard(ex.build_validation_method, "Validation evidence is unavailable.")
    importance = None
    try:
        importance = ex.build_feature_importance(
            model="random_forest", horizon=params.horizon, method="native", limit=15, grouped=False
        )
    except ex.ExplainabilityError:
        importance = None  # optional section; omitted rather than faked

    models = _selected(params)
    meta = _metadata(req, params, count_model=models["count"]["model_name"],
                     surge_model=models["surge"]["model_name"])
    return {
        "metadata": meta, "performance": perf, "selection": selection["data"],
        "validation": validation["data"],
        "importance": importance["data"] if importance else None,
        "importance_limitations": importance["limitations"] if importance else [],
        "limitations": list(CORE_LIMITATIONS),
    }


def _data_quality(req: ReportRequest) -> dict[str, Any]:
    """Data Quality report — consumes the SINGLE approved evidence builder.

    The identical validated evidence backs /api/lifecycle/data-quality, so the
    page and the report can never disagree. Broken leakage/model-card or coverage
    evidence raises DataQualityEvidenceError, which fails the report closed (503).
    Leakage controls and the complete-lag count come from validated evidence, not
    static prose.
    """
    params = _params(req)
    from app.services.data_quality_evidence import (
        DataQualityEvidenceError,
        build_data_quality_evidence,
    )

    try:
        evidence = build_data_quality_evidence()
    except DataQualityEvidenceError as exc:
        raise ReportUnavailableError(str(exc)) from exc

    meta = _metadata(req, params)
    return {
        "metadata": meta,
        "coverage": evidence["coverage"],
        "components": evidence["components"],
        "climate_lag_columns": evidence["climate_lag_columns_verified"],
        "rows_with_complete_climate_lags": evidence["rows_with_complete_climate_lags"],
        "leakage_controls": evidence["leakage_controls"],
        "evidence_warnings": evidence["warnings"],
        "definitions": evidence["definitions"],
        "limitations": list(CORE_LIMITATIONS),
    }


def _climate_signals(req: ReportRequest) -> dict[str, Any]:
    params = _params(req)
    climate = _guard(lambda: dd.build_climate_section(params), "Climate evidence is unavailable.")
    district_name = None
    if params.district_id:
        from app.models import DistrictReference

        ref = DistrictReference.query.filter_by(district_id=params.district_id).first()
        district_name = ref.district_name if ref else params.district_id
    meta = _metadata(req, params, district_name=district_name)
    return {
        "metadata": meta, "climate": climate,
        "scope_note": climate.get("aggregation_label"),
        "limitations": list(CORE_LIMITATIONS),
    }


def _methods(req: ReportRequest) -> dict[str, Any]:
    params = _params(req)
    validation = _guard(ex.build_validation_method, "Validation evidence is unavailable.")
    selection_by_horizon = {}
    for horizon in dd.HORIZONS:
        selection_by_horizon[horizon] = _selected(dd.resolve_params(horizon=horizon))
    meta = _metadata(req, params)
    sections = [
        ("Study purpose",
         "District-level malaria surge prediction and early warning for Uganda using weekly confirmed "
         "malaria cases, reporting completeness and climate predictors."),
        ("Data sources", " ".join(DATA_SOURCES)),
        ("District reference and boundaries",
         "146 canonical Uganda districts and cities with UBOS boundaries, joined on district_id."),
        ("Surveillance-data preparation",
         "Weekly confirmed malaria cases (HMIS 033b) for 2020-2025 are validated to ISO epidemiological weeks."),
        ("Missing-data policy",
         "Missing DHIS2 malaria rows remain missing and are never zero-filled. Explicit zeros remain zero."),
        ("Adjudicated reporting errors",
         "Confirmed erroneous district-weeks are retained in the audit registry, treated as missing analytical outcomes and excluded from validation truth."),
        ("Predictor-only interpolation",
         "Only bounded internal gaps of one or two weeks are linearly interpolated for lagged and rolling predictors; outcomes are never imputed."),
        ("Climate preparation",
         "CHIRPS rainfall and ERA5-Land temperature and humidity are aggregated to district weeks by boundary."),
        ("Climate lags",
         "Rainfall, temperature and humidity lags are generated at 4, 8 and 12 weeks, always preceding the target."),
        ("Modelling dataset",
         "A leakage-checked district-week panel with future-aligned targets and training-only preprocessing."),
        ("Approved models", "Baseline/Naive, SARIMAX, Random Forest and XGBoost. No negative binomial model and no wind-speed variable are used."),
        ("Rolling-origin validation",
         "Chronological rolling-origin validation with 12 numbered folds. No random split is used."),
        ("Count-model selection",
         "Lowest WAPE when available; RMSE is used when WAPE is unavailable."),
        ("Surge-model selection",
         "Require sensitivity >=0.60 and specificity >=0.40 where possible; then highest F2, "
         "with sensitivity, PR-AUC, precision and specificity as tie-breakers. Accuracy alone is not used."),
        ("Threshold interpretation",
         "The district threshold is the training-only 75th percentile in the target epidemiological week +/-2 weeks. "
         "The confirmed epidemic year 2022 remains in count-model training but is excluded from the normal-reference threshold."),
        ("Surge classification",
         "Random Forest and XGBoost use a separate class-balanced probability classifier, with its cutoff selected inside chronological training data. The count-threshold rule is retained as a benchmark."),
        ("Forecast publication",
         "Forecasts are published for a single production issue week and predict the target week only."),
        ("Main limitations", " ".join(CORE_LIMITATIONS)),
        ("Responsible-use statement", RESEARCH_USE_STATEMENT),
    ]
    return {
        "metadata": meta, "sections": sections, "validation": validation["data"],
        "selection_by_horizon": selection_by_horizon, "limitations": list(CORE_LIMITATIONS),
    }


_BUILDERS = {
    "national_summary": _national_summary,
    "district_forecast": _district_forecast,
    "district_risk_list": _district_risk_list,
    "model_performance": _model_performance,
    "data_quality": _data_quality,
    "climate_signals": _climate_signals,
    "methods": _methods,
}


def _fmt(value, dp: int = 0) -> str:
    if value is None:
        return "unavailable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:,.{dp}f}"


def report_registry() -> list[dict[str, Any]]:
    """Registry for the Reports page; formats come from here, never the frontend."""
    return [
        {
            "report_type": key,
            "title": spec["title"],
            "purpose": spec["purpose"],
            "filters": spec["filters"],
            "formats": spec["formats"],
        }
        for key, spec in REPORTS.items()
    ]
