"""Read-only Settings & Governance evidence builder (Phase 29E, corrective pass).

Assembles the eleven Settings sections entirely from *existing approved
evidence*: the pipeline-status/explainability loaders, the data-quality evidence
builder, the threshold-method service, the report registry, application config,
and live read-only database checks.

It creates NO second project-state, model-selection, threshold, forecast-
integrity or data-quality implementation — it composes the approved ones.

Nothing on this page is a hardcoded claim:

* excluded-model and excluded-predictor statements are derived from the approved
  model list and the manifest's approved predictor list;
* forecast contracts come from validated forecast-generation evidence, the
  threshold-method service and the database;
* authentication facts come from Flask/Flask-WTF configuration, the request and
  the current user — not from asserted booleans;
* integrity is recomputed live, including the canonical deterministic
  fingerprint and a full composite-key check.

Required evidence that is missing, malformed, contradictory or incomplete either
fails the whole build closed (``GovernanceEvidenceError``) or marks the affected
section explicitly unavailable. The builder performs no write, loads no model
binary, runs no CLI command, makes no live DHIS2 call, and exposes no path,
secret, credential, token, session id, hostname or internal row id.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func

from app.extensions import db
from app.models import DistrictReference, DistrictWeekFeature, ForecastOutput

HORIZONS = (4, 8, 12)
NOT_PUBLISHED = "Not published"

# Methods/variables the approved scope explicitly excludes. Their absence is
# PROVEN below against the approved model list and predictor list; it is never
# asserted.
EXCLUDED_MODEL_TOKENS = ("negative binomial", "poisson", "prophet")
EXCLUDED_PREDICTOR_TOKENS = ("wind", "windspeed", "wind_speed")

RESEARCH_LIMITATIONS = [
    "This is a research proof-of-concept; it supports review and early action, not operational decisions.",
    "A High or Very High risk level is an early-warning signal for review and does not confirm an outbreak.",
    "DHIS2 reporting quality affects forecasts; missing reports are never treated as zero.",
    "HMIS 033b reporting completeness measures form submission, not malaria burden.",
    "Climate variables are predictive inputs associated with malaria patterns; they do not establish causation.",
    "Forecasts predict the target week only; intermediate weekly forecasts are not generated.",
    "Confirmed reporting errors are excluded from analytical outcomes only after documented consultation; raw values remain auditable.",
    "Model quality is judged on error and surge-detection metrics together; accuracy alone is not used.",
]

# Only preferences with genuinely observable behaviour are published. Anything
# that is not actually implemented is absent from the schema, the UI and any claim.
PREFERENCE_SCHEMA = {
    "version": 1,
    "storage_key": "mews.preferences.v1",
    "preferences": {
        "default_horizon": {
            "type": "enum", "values": [4, 8, 12], "default": 4,
            "effect": "Sets the initial forecast-horizon filter on pages that expose one.",
        },
        "reduce_motion": {
            "type": "bool", "default": False,
            "effect": "Disables CSS transitions and animations across the application.",
        },
    },
    "note": (
        "Display preferences are stored only in this browser (localStorage). They never enter the "
        "Flask session, PostgreSQL or any analytical output, and cannot change issue week, model "
        "selection, thresholds, risk levels or forecast values."
    ),
}

# The pipeline stages required by the Phase 29E specification, mapped to the
# approved project-state components that evidence them.
PIPELINE_STAGES = [
    ("District Reference and Boundaries", "reference_and_boundaries"),
    ("DHIS2 Malaria Panel", "dhis2_malaria_panel"),
    ("Reporting Completeness", "reporting_completeness"),
    ("Climate Ingestion", "climate_ingestion"),
    ("Climate Aggregation", "climate_aggregation"),
    ("Climate Lags", "climate_lags"),
    ("Modelling Dataset", "modelling_dataset"),
    ("Model Validation", "models_and_forecasts"),
    ("Production Model Fitting", "models_and_forecasts"),
    ("Forecast Generation", "models_and_forecasts"),
    ("Model Performance Consolidation", "models_and_forecasts"),
    ("Dashboard Publication", "visible_dashboard"),
]
STATUS_LABEL = {"complete": "Complete", "partial": "Partial", "stale": "Warning"}


class GovernanceEvidenceError(RuntimeError):
    """Required governance/integrity evidence is missing, malformed or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GovernanceEvidenceError(message)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _validated_database_targets(rows) -> dict[int, tuple[int, int]]:
    """Require exactly one valid production target week for every horizon."""
    targets: dict[int, tuple[int, int]] = {}
    for raw_horizon, raw_year, raw_week in rows:
        _require(
            isinstance(raw_horizon, int)
            and isinstance(raw_year, int)
            and isinstance(raw_week, int)
            and raw_year > 0
            and 1 <= raw_week <= 53,
            "Database target-week evidence is malformed.",
        )
        _require(raw_horizon in HORIZONS, "Database target-week evidence contains an unexpected horizon.")
        _require(raw_horizon not in targets, "Database contains contradictory target weeks for one horizon.")
        targets[raw_horizon] = (raw_year, raw_week)

    _require(set(targets) == set(HORIZONS), "Database target-week evidence is incomplete.")
    return targets


def _optional_json(key: str):
    from app.services.explainability import ExplainabilityError, load_json

    try:
        return load_json(key)
    except ExplainabilityError:
        return None


def _label(status):
    if not status:
        return "Unavailable"
    return STATUS_LABEL.get(str(status).lower(), str(status).title())


# --------------------------------------------------------------------------
# Live integrity: fingerprint + full composite key verification
# --------------------------------------------------------------------------
def _forecast_integrity(issue_year, issue_week, approved_models) -> dict[str, Any]:
    """Recompute integrity live: counts, fingerprint and composite key coverage."""
    try:
        with db.session.no_autoflush:
            rows = ForecastOutput.query.order_by(ForecastOutput.id).all()
            active = [d.district_id for d in
                      DistrictReference.query.filter_by(active=True).order_by(DistrictReference.district_id).all()]
            null_predictions = ForecastOutput.query.filter(
                ForecastOutput.predicted_cases.is_(None)).count()
            issues = db.session.query(
                ForecastOutput.forecast_origin_epi_year, ForecastOutput.forecast_origin_epi_week
            ).distinct().all()
            duplicate_keys = (
                db.session.query(
                    ForecastOutput.district_id, ForecastOutput.model_name,
                    ForecastOutput.horizon_weeks, ForecastOutput.forecast_origin_epi_year,
                    ForecastOutput.forecast_origin_epi_week)
                .group_by(
                    ForecastOutput.district_id, ForecastOutput.model_name,
                    ForecastOutput.horizon_weeks, ForecastOutput.forecast_origin_epi_year,
                    ForecastOutput.forecast_origin_epi_week)
                .having(func.count() > 1).count()
            )
            target_rows = db.session.query(
                ForecastOutput.horizon_weeks, ForecastOutput.target_epi_year,
                ForecastOutput.target_epi_week).distinct().all()
    except Exception as exc:  # noqa: BLE001
        raise GovernanceEvidenceError("Forecast integrity could not be read from the database.") from exc

    _require(bool(rows) and bool(active), "Forecast integrity evidence is empty.")
    db_targets = _validated_database_targets(target_rows)

    # Canonical deterministic fingerprint — the established serialization,
    # including pipeline_run_id. The digest exposes no internal row id.
    payload = [
        [r.id, r.district_id, r.model_name, r.horizon_weeks, str(r.predicted_cases),
         str(r.threshold_used), r.risk_level, r.pipeline_run_id]
        for r in rows
    ]
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    # Complete composite key set: district x approved model x horizon x issue.
    expected_keys = set()
    if issue_year is not None and issue_week is not None:
        for district in active:
            for model in approved_models:
                for horizon in HORIZONS:
                    expected_keys.add((district, model, horizon, issue_year, issue_week))
    actual_keys = {
        (r.district_id, r.model_name, int(r.horizon_weeks),
         r.forecast_origin_epi_year, r.forecast_origin_epi_week) for r in rows
    }
    missing_keys = expected_keys - actual_keys
    unexpected_keys = actual_keys - expected_keys

    return {
        "production_forecast_rows": len(rows),
        "active_districts": len(active),
        "approved_model_count": len(approved_models),
        "null_predicted_cases": int(null_predictions),
        "duplicate_forecast_keys": int(duplicate_keys),
        "production_issue_count": len(issues),
        "single_production_issue": len(issues) == 1,
        "forecast_fingerprint_sha256": fingerprint,
        "expected_composite_keys": len(expected_keys),
        "actual_composite_keys": len(actual_keys),
        "missing_composite_keys": len(missing_keys),      # counts only, never row ids
        "unexpected_composite_keys": len(unexpected_keys),
        "composite_key_set_complete": not missing_keys and not unexpected_keys,
        "database_target_weeks": {str(h): {"target_epi_year": v[0], "target_epi_week": v[1]}
                                  for h, v in sorted(db_targets.items())},
        "verified_at_utc": _now_iso(),
    }


def _latest_data_weeks() -> dict[str, Any]:
    """Latest observed malaria / completeness / processed-climate weeks."""
    try:
        with db.session.no_autoflush:
            malaria = db.session.query(func.max(DistrictWeekFeature.week_start_date)).filter(
                DistrictWeekFeature.confirmed_malaria_cases.isnot(None)).scalar()
            completeness = db.session.query(func.max(DistrictWeekFeature.week_start_date)).filter(
                DistrictWeekFeature.facility_reporting_completeness.isnot(None)).scalar()
            climate = db.session.query(func.max(DistrictWeekFeature.week_start_date)).filter(
                DistrictWeekFeature.rainfall_current.isnot(None)).scalar()
    except Exception:  # noqa: BLE001
        return {"latest_malaria_week": NOT_PUBLISHED,
                "latest_completeness_week": NOT_PUBLISHED,
                "latest_processed_climate_week": NOT_PUBLISHED}
    return {
        "latest_malaria_week": malaria.isoformat() if malaria else NOT_PUBLISHED,
        "latest_completeness_week": completeness.isoformat() if completeness else NOT_PUBLISHED,
        "latest_processed_climate_week": climate.isoformat() if climate else NOT_PUBLISHED,
    }


# --------------------------------------------------------------------------
# Main builder
# --------------------------------------------------------------------------
def build_governance() -> dict[str, Any]:
    """Build every Settings section from approved evidence, or fail closed."""
    from flask import current_app, request
    from flask_login import current_user

    from app.services.dashboard_data import HORIZONS as DASH_HORIZONS
    from app.services.dashboard_data import resolve_params, selected_models
    from app.services.data_quality_evidence import (
        DataQualityEvidenceError,
        build_data_quality_evidence,
    )
    from app.services.explainability import (
        ExplainabilityError,
        build_overview,
        build_pipeline_status,
        build_threshold_method,
        build_validation_method,
    )
    from app.services.report_data_service import report_registry

    warnings: list[str] = []

    # ---- required evidence ---------------------------------------------
    try:
        pipeline = build_pipeline_status()["data"]
    except ExplainabilityError as exc:
        raise GovernanceEvidenceError("Pipeline status evidence is unavailable.") from exc

    forecast = _optional_json("forecast_generation")
    _require(isinstance(forecast, dict), "Forecast-generation evidence is unavailable.")
    manifest = _optional_json("production_manifest")
    _require(isinstance(manifest, dict), "Production manifest evidence is unavailable.")

    approved_models = manifest.get("models")
    _require(isinstance(approved_models, list) and approved_models,
             "Approved model list is missing from the production manifest.")
    approved_predictors = manifest.get("approved_pooled_predictors")
    _require(isinstance(approved_predictors, list) and approved_predictors,
             "Approved predictor list is missing from the production manifest.")

    issue_year = forecast.get("issue_year")
    issue_week = forecast.get("issue_week")
    _require(isinstance(issue_year, int) and isinstance(issue_week, int),
             "Forecast-generation evidence has no valid production issue.")
    issue = f"{issue_year}W{int(issue_week):02d}"

    target_weeks = forecast.get("target_weeks")
    _require(isinstance(target_weeks, dict), "Forecast target weeks are unavailable.")
    published_targets = {}
    for horizon in HORIZONS:
        entry = target_weeks.get(str(horizon)) or target_weeks.get(horizon)
        _require(isinstance(entry, dict)
                 and isinstance(entry.get("target_epi_year"), int)
                 and isinstance(entry.get("target_epi_week"), int),
                 f"Target week for horizon {horizon} is missing or malformed.")
        published_targets[str(horizon)] = {
            "target_epi_year": entry["target_epi_year"],
            "target_epi_week": entry["target_epi_week"],
            "target_label": f"{entry['target_epi_year']}W{int(entry['target_epi_week']):02d}",
            "target_week_start_date": entry.get("target_week_start_date"),
        }

    integrity = _forecast_integrity(issue_year, issue_week, approved_models)

    # Cross-check the database target weeks against the published evidence.
    target_mismatches = []
    for horizon, published in published_targets.items():
        actual = integrity["database_target_weeks"].get(horizon)
        if actual and (actual["target_epi_year"] != published["target_epi_year"]
                       or actual["target_epi_week"] != published["target_epi_week"]):
            target_mismatches.append(horizon)
    _require(not target_mismatches,
             "Database target weeks contradict the published forecast evidence.")

    # ---- derived exclusions (never asserted) ----------------------------
    model_blob = " ".join(str(m).lower() for m in approved_models)
    predictor_blob = " ".join(str(p).lower() for p in approved_predictors)
    excluded_models = [
        {"method": token, "present_in_approved_models": token in model_blob}
        for token in EXCLUDED_MODEL_TOKENS
    ]
    excluded_predictors = [
        {"variable": token, "present_in_approved_predictors": token in predictor_blob}
        for token in EXCLUDED_PREDICTOR_TOKENS
    ]

    dataset_signature = None
    sig = (manifest.get("dataset_signature") or {}).get("sha256")
    if isinstance(sig, str):
        dataset_signature = sig[:16]  # short, non-path, non-secret

    # ---- downstream service availability (probed, not asserted) --------
    try:
        build_overview(4)
        explainability_status = "Available"
    except Exception:  # noqa: BLE001
        explainability_status = "Unavailable"
        warnings.append("Explainability services are unavailable.")

    try:
        dq_evidence = build_data_quality_evidence()
        lifecycle_status = "Available"
    except DataQualityEvidenceError as exc:
        dq_evidence = None
        lifecycle_status = "Unavailable"
        warnings.append(f"Lifecycle data-quality evidence is unavailable: {exc}")

    registry = report_registry()
    reports_status = "Available" if registry else "Unavailable"

    try:
        threshold = build_threshold_method(resolve_params(horizon=DASH_HORIZONS[0]))["data"]
    except Exception:  # noqa: BLE001
        threshold = None
        warnings.append("Threshold-method evidence is unavailable.")

    # ---- 1. System Overview --------------------------------------------
    system_overview = {
        "system_name": current_app.config.get("APP_NAME"),
        "subtitle": current_app.config.get("APP_SUBTITLE"),
        "application_version": current_app.config.get("APP_VERSION") or NOT_PUBLISHED,
        "active_districts": integrity["active_districts"],
        "production_issue_week": issue,
        "forecast_horizons": list(HORIZONS),
        "approved_model_count": len(approved_models),
        "production_forecast_rows": integrity["production_forecast_rows"],
        "overall_project_state": _label(pipeline.get("overall_state")),
        "dashboard_status": _label(pipeline.get("dashboard_status")),
        "explainability_status": explainability_status,
        "lifecycle_status": lifecycle_status,
        "reports_status": reports_status,
        "generated_at_utc": _now_iso(),
    }

    # ---- 2. Authentication and Session Security -------------------------
    # Derived from Flask/Flask-WTF configuration, the request and current user.
    from app.services.dhis2_auth import DHIS2_IDENTITY_VERSION

    csrf_configured = bool(current_app.config.get("WTF_CSRF_ENABLED", True))
    csrf_extension = "csrf" in current_app.extensions
    reauth_minutes = current_app.config.get("DHIS2_LOGIN_SESSION_MINUTES")
    authenticated = bool(getattr(current_user, "is_authenticated", False))
    verified_at = getattr(current_user, "verified_at", None)
    session_age_seconds = None
    session_expires_in_seconds = None
    if isinstance(verified_at, int) and isinstance(reauth_minutes, int):
        age = int(datetime.now(timezone.utc).timestamp()) - verified_at
        session_age_seconds = max(age, 0)
        session_expires_in_seconds = max(reauth_minutes * 60 - age, 0)

    identity_payload = None
    to_session = getattr(current_user, "to_session", None)
    if authenticated and callable(to_session):
        identity_payload = to_session()
    _require(isinstance(identity_payload, dict), "Authenticated identity evidence is unavailable.")
    identity_fields = list(identity_payload)

    authentication = {
        "method": "Live DHIS2 credential verification at sign-in",
        "session_transport": "Signed client-side Flask session cookie (browser-session only)",
        "session_signed_with_secret_key": bool(current_app.config.get("SECRET_KEY")),
        "session_permanent": bool(current_app.config.get("SESSION_PERMANENT", False)),
        "cookie_httponly": bool(current_app.config.get("SESSION_COOKIE_HTTPONLY", True)),
        "cookie_samesite": current_app.config.get("SESSION_COOKIE_SAMESITE") or NOT_PUBLISHED,
        "cookie_secure": bool(current_app.config.get("SESSION_COOKIE_SECURE", False)),
        "csrf_configured": csrf_configured,
        "csrf_extension_registered": csrf_extension,
        "csrf_scope": (
            "Global CSRFProtect extension (all POST routes)" if csrf_extension
            else "Per-form Flask-WTF tokens on submitted forms (no global CSRFProtect extension)"
        ),
        "reauthentication_minutes": reauth_minutes if reauth_minutes is not None else NOT_PUBLISHED,
        "identity_fields_stored": identity_fields,
        "identity_version": DHIS2_IDENTITY_VERSION,
        "currently_authenticated": authenticated,
        "current_session_age_seconds": session_age_seconds,
        "current_session_expires_in_seconds": session_expires_in_seconds,
        "request_is_secure": bool(getattr(request, "is_secure", False)),
        "roles_supported": [],  # no invented roles
        "login_vs_data_note": (
            "Sign-in verifies live DHIS2 credentials. It does not pull new malaria data; the "
            "dashboard shows the published production issue week only."
        ),
        "credential_storage_note": (
            "The DHIS2 identity payload contains only the four non-secret fields listed above. "
            "The signed Flask session cookie also contains Flask-Login session-management "
            "metadata; no DHIS2 credential or token is stored in the identity payload."
        ),
    }

    # ---- 3. Data Provenance and Freshness -------------------------------
    provenance = {
        "surveillance": {
            "source": "DHIS2 HMIS 033b confirmed malaria cases (fUflbWWhouR / 033B-CD01a)",
            "production_issue_week": issue,
            "freshness_note": "Frozen at the published production issue week; not a live feed.",
        },
        "reporting_completeness": {"source": "HMIS 033b facility reporting completeness",
                                   "note": "Form submission, not malaria burden."},
        "climate": {
            "rainfall": "CHIRPS (mm, weekly total)",
            "temperature": "ERA5-Land (°C, weekly mean)",
            "humidity": "ERA5-Land relative humidity (%, derived from temperature and dewpoint)",
        },
        "latest_weeks": _latest_data_weeks(),
        "forecast_evidence_generated_at_utc": forecast.get("generated_at_utc") or NOT_PUBLISHED,
        "dataset_signature": dataset_signature or NOT_PUBLISHED,
    }

    # ---- 4. Pipeline Status (required stages) ---------------------------
    raw_components = pipeline.get("components", {}) or {}

    def component_status(key):
        value = raw_components.get(key)
        return value.get("status") if isinstance(value, dict) else value

    def component_rows(key):
        value = raw_components.get(key)
        counts = value.get("row_counts") if isinstance(value, dict) else None
        if isinstance(counts, dict) and counts:
            return sum(v for v in counts.values() if isinstance(v, int))
        return None

    stages = []
    for title, key in PIPELINE_STAGES:
        status = component_status(key)
        stages.append({
            "stage": title,
            "status": _label(status),
            "output_count": component_rows(key),
            "evidence_component": key,
            "unavailable_reason": None if status else "No approved evidence for this stage.",
        })
    stages.append({"stage": "Explainability Services", "status": explainability_status,
                   "output_count": None, "evidence_component": "explainability_loaders",
                   "unavailable_reason": None if explainability_status == "Available" else "Loader unavailable."})
    stages.append({"stage": "Lifecycle Pages", "status": lifecycle_status,
                   "output_count": None, "evidence_component": "data_quality_evidence",
                   "unavailable_reason": None if lifecycle_status == "Available" else "Evidence unavailable."})
    stages.append({"stage": "Reports and Exports", "status": reports_status,
                   "output_count": len(registry) or None, "evidence_component": "report_registry",
                   "unavailable_reason": None if registry else "Report registry is empty."})

    pipeline_status = {
        "overall_state": _label(pipeline.get("overall_state")),
        "stages": stages,
        "components": {k: component_status(k) for k in raw_components},
        "issue_label": issue,
        "warnings": pipeline.get("warnings", []),
        "verified_at_utc": _now_iso(),
    }

    # ---- 5. Model Governance --------------------------------------------
    try:
        validation = build_validation_method()["data"]
    except ExplainabilityError as exc:
        raise GovernanceEvidenceError("Validation-method evidence is unavailable.") from exc

    selected_by_horizon = {}
    for horizon in DASH_HORIZONS:
        models = selected_models(horizon)
        selected_by_horizon[str(horizon)] = {
            "count_model": models["count"]["model_name"],
            "surge_model": models["surge"]["model_name"],
            "count_rule": models["count"].get("metric_rule"),
            "surge_rule": models["surge"].get("metric_rule"),
        }

    card_status = {}
    for name, key in (("SARIMAX", "sarimax_card"), ("Random Forest", "random_forest_card"),
                      ("XGBoost", "xgboost_card")):
        card = _optional_json(key)
        card_status[name] = (card.get("status") if isinstance(card, dict) and card.get("status")
                             else ("Available" if card else "Unavailable"))
    validation_status = {}
    for name, key in (("Baseline/Naive", "baseline_validation"), ("SARIMAX", "sarimax_validation"),
                      ("Random Forest", "random_forest_validation"), ("XGBoost", "xgboost_validation")):
        summary = _optional_json(key)
        validation_status[name] = (summary.get("status") if isinstance(summary, dict) and summary.get("status")
                                   else ("Available" if summary else "Unavailable"))

    model_governance = {
        "approved_models": list(approved_models),
        "approved_predictor_count": len(approved_predictors),
        "validation_method": validation.get("validation_method") or NOT_PUBLISHED,
        "numbered_folds": validation.get("numbered_folds"),
        "selected_models_by_horizon": selected_by_horizon,
        "selection_rules": {
            "count": "Lowest WAPE when available; RMSE when WAPE is unavailable.",
            "surge": (
                "Require sensitivity >=0.60 and specificity >=0.40 where possible; "
                "then highest F2 with sensitivity, PR-AUC, precision and specificity "
                "as tie-breakers. Accuracy alone is not used."
            ),
        },
        "model_card_status": card_status,
        "validation_artifact_status": validation_status,
        "production_artifact_status": manifest.get("status") or NOT_PUBLISHED,
        "production_manifest_version": manifest.get("manifest_version") or NOT_PUBLISHED,
        "dataset_signature": dataset_signature or NOT_PUBLISHED,
        "excluded_methods": excluded_models,
        "excluded_predictors": excluded_predictors,
    }

    # ---- 6. Forecast Governance ------------------------------------------
    forecast_governance = {
        "production_issue_week": issue,
        "forecast_horizons": list(HORIZONS),
        "published_target_weeks": published_targets,
        "database_target_weeks": integrity["database_target_weeks"],
        "target_weeks_agree_with_database": not target_mismatches,
        "publication_status": forecast.get("status") or NOT_PUBLISHED,
        "published": forecast.get("published"),
        "generation_timestamp_utc": forecast.get("generated_at_utc") or NOT_PUBLISHED,
        "model_count": len(approved_models),
        "expected_forecast_rows": forecast.get("expected_forecast_rows"),
        "actual_forecast_rows": integrity["production_forecast_rows"],
        "row_counts_agree": forecast.get("expected_forecast_rows") == integrity["production_forecast_rows"],
        "null_predicted_cases": integrity["null_predicted_cases"],
        "duplicate_forecast_keys": integrity["duplicate_forecast_keys"],
        "single_production_issue": integrity["single_production_issue"],
        "schema_version": manifest.get("manifest_version") or NOT_PUBLISHED,
        "threshold_method": (threshold or {}).get("method") or NOT_PUBLISHED,
        "threshold_unit": (threshold or {}).get("unit") or NOT_PUBLISHED,
        "threshold_history_cutoff": (threshold or {}).get("history_cutoff") or NOT_PUBLISHED,
        "surge_score_definition": (threshold or {}).get("surge_score_definition") or NOT_PUBLISHED,
        "risk_interpretation": (threshold or {}).get("risk_interpretation") or NOT_PUBLISHED,
        "threshold_is_probability": (threshold or {}).get("is_probability"),
        "threshold_confirms_outbreak": (threshold or {}).get("confirms_outbreak"),
    }

    # ---- 7. Data Quality Rules -------------------------------------------
    if dq_evidence is not None:
        data_quality = {
            "available": True,
            "coverage": dq_evidence["coverage"],
            "rows_with_complete_climate_lags": dq_evidence["rows_with_complete_climate_lags"],
            "leakage_controls": {k: v["value"] for k, v in dq_evidence["leakage_controls"].items()},
            "leakage_control_sources": {k: v["source"] for k, v in dq_evidence["leakage_controls"].items()},
            "definitions": dq_evidence["definitions"],
            "warnings": dq_evidence["warnings"],
        }
    else:
        data_quality = {"available": False, "reason": "Data-quality evidence is unavailable."}

    # ---- 8. Reports and Export Status -------------------------------------
    reports = {
        "status": reports_status,
        "report_count": len(registry),
        "reports": [
            {
                "report_type": r["report_type"],
                "title": r["title"],
                "formats": r["formats"],
                "url": f"/reports/?report_type={r['report_type']}",
                "print_url": (f"/reports/print?report_type={r['report_type']}"
                              if "html" in r["formats"] and "district_required" not in r["filters"]
                              else None),
            }
            for r in registry
        ],
        "supported_formats": sorted({f for r in registry for f in r["formats"]}),
        "deferred": ["Scheduled email delivery", "Report subscriptions", "Public report links",
                     "PowerPoint export"],
    }

    # ---- 11. Research Status ---------------------------------------------
    research = {
        "status": "Research proof-of-concept",
        "responsible_use": (
            "This system supports review and early action. It does not replace epidemiological "
            "judgement or official outbreak confirmation."
        ),
        "limitations": list(RESEARCH_LIMITATIONS),
    }

    # ---- warnings ---------------------------------------------------------
    warnings.extend(pipeline.get("warnings", []) or [])
    if not forecast_governance["row_counts_agree"]:
        warnings.append("Published forecast row count does not match the database.")
    if integrity["duplicate_forecast_keys"]:
        warnings.append("Duplicate production forecast keys detected.")
    if integrity["null_predicted_cases"]:
        warnings.append("Null predicted cases detected.")
    if not integrity["composite_key_set_complete"]:
        warnings.append(
            f"Composite forecast key set incomplete: {integrity['missing_composite_keys']} missing, "
            f"{integrity['unexpected_composite_keys']} unexpected.")
    for item in excluded_models:
        if item["present_in_approved_models"]:
            warnings.append(f"Excluded method '{item['method']}' appears in the approved model list.")
    for item in excluded_predictors:
        if item["present_in_approved_predictors"]:
            warnings.append(f"Excluded predictor '{item['variable']}' appears in the approved predictors.")

    return {
        "data": {
            "system_overview": system_overview,
            "authentication": authentication,
            "data_provenance": provenance,
            "pipeline": pipeline_status,
            "models": model_governance,
            "forecasts": forecast_governance,
            "data_quality_rules": data_quality,
            "reports": reports,
            "integrity": integrity,
            "display_preferences": PREFERENCE_SCHEMA,
            "research": research,
        },
        "metadata": {
            "response_generated_at_utc": _now_iso(),
            "evidence_generated_at_utc": forecast.get("generated_at_utc") or NOT_PUBLISHED,
            "project_state_write": False,
        },
        "warnings": warnings,
        "limitations": list(RESEARCH_LIMITATIONS),
    }
