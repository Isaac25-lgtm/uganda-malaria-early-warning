"""Small read-only presentation APIs for lifecycle pages (Phase 29C).

These endpoints exist only because the surveillance and data-quality pages need
counts the existing dashboard/explainability APIs do not expose. They are strictly
read-only (no add/flush/commit; queries run under db.session.no_autoflush), reuse
the existing DHIS2 session auth, return the standard envelope, and never expose
paths, credentials, or raw artifacts.

Coverage and leakage claims are taken from *approved evidence* (project-state
row-counts, the modelling-dataset summary and the production manifest) and are
cross-checked against the database. They are never recomputed with a second
definition and never hardcoded. Any inconsistency fails closed with a 503.
"""
from flask import jsonify
from sqlalchemy import func

from app.blueprints.dashboard.routes import api_login_required
from app.blueprints.lifecycle_api import bp
from app.extensions import db
from app.models import DistrictReference, DistrictWeekFeature

# The nine approved climate-lag columns are defined once, in the shared
# data-quality evidence service, and re-exported here for existing importers.
from app.services.data_quality_evidence import CLIMATE_LAG_COLUMNS  # noqa: E402


def _envelope(data, warnings=None, limitations=None, metadata=None):
    return (
        jsonify(
            {
                "status": "available",
                "reason": None,
                "data": data,
                "params": {},
                "empty_states": [],
                "metadata": metadata or {},
                "warnings": warnings or [],
                "limitations": limitations or [],
            }
        ),
        200,
    )


def _unavailable(message):
    return (
        jsonify(
            {
                "status": "unavailable",
                "reason": message,
                "data": None,
                "params": {},
                "empty_states": ["Evidence unavailable."],
                "metadata": {},
                "warnings": [],
                "limitations": [],
            }
        ),
        503,
    )


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@bp.route("/api/lifecycle/surveillance")
@api_login_required
def api_surveillance():
    """Coverage from approved project-state evidence, cross-checked to the DB.

    Missing district-weeks are ABSENT rows, so they cannot be counted as NULL
    rows in DistrictWeekFeature. The authoritative counts come from the
    project-state realignment evidence and are validated for internal
    consistency (expected − received == missing) and agreement with the DB.
    """
    from app.services.project_state import build_project_state_summary

    try:
        summary = build_project_state_summary(write=False)
    except Exception:  # pragma: no cover - controlled unavailable
        return _unavailable("Project-state coverage evidence is unavailable.")
    panel = (summary.get("components") or {}).get("dhis2_malaria_panel") or {}
    counts = panel.get("row_counts") or {}

    expected = _as_int(counts.get("expected_district_weeks"))
    received = _as_int(counts.get("district_week_features"))
    missing = _as_int(counts.get("missing_district_weeks"))
    explicit_zero = _as_int(counts.get("feature_explicit_zero_count"))
    districts_represented = _as_int(counts.get("districts_represented"))

    if None in (expected, received, missing):
        return _unavailable("Surveillance coverage evidence is missing required totals.")
    # Internal consistency of the approved evidence.
    if expected - received != missing:
        return _unavailable("Surveillance coverage evidence is internally inconsistent.")

    try:
        with db.session.no_autoflush:
            db_rows = db.session.query(func.count(DistrictWeekFeature.id)).scalar() or 0
            active_districts = DistrictReference.query.filter_by(active=True).count()
            latest = (
                DistrictWeekFeature.query.order_by(DistrictWeekFeature.week_start_date.desc()).first()
            )
            week_bounds = db.session.query(
                func.min(DistrictWeekFeature.week_start_date),
                func.max(DistrictWeekFeature.week_start_date),
            ).first()
            latest_total = None
            latest_completeness = None
            if latest is not None:
                latest_total = db.session.query(
                    func.sum(DistrictWeekFeature.confirmed_malaria_cases)
                ).filter(DistrictWeekFeature.week_start_date == latest.week_start_date).scalar()
                latest_completeness = db.session.query(
                    func.avg(DistrictWeekFeature.facility_reporting_completeness)
                ).filter(DistrictWeekFeature.week_start_date == latest.week_start_date).scalar()
    except Exception:  # pragma: no cover
        return _unavailable("Surveillance database counts could not be read.")

    # The received count must agree with the actual stored panel rows.
    if received != db_rows:
        return _unavailable("Surveillance evidence disagrees with the stored panel row count.")

    data = {
        "latest_week_label": f"{latest.epi_year}W{int(latest.epi_week):02d}" if latest else None,
        "latest_week_start_date": latest.week_start_date.isoformat() if latest else None,
        "coverage_start": week_bounds[0].isoformat() if week_bounds and week_bounds[0] else None,
        "coverage_end": week_bounds[1].isoformat() if week_bounds and week_bounds[1] else None,
        "expected_district_weeks": expected,
        "received_district_weeks": received,
        "confirmed_records": received,
        "missing_district_weeks": missing,
        "explicit_zero_records": explicit_zero,
        "districts_in_panel": districts_represented if districts_represented is not None else active_districts,
        "active_districts": int(active_districts),
        "latest_week_confirmed_cases": int(latest_total) if latest_total is not None else None,
        "latest_week_reporting_completeness": (
            round(float(latest_completeness), 2) if latest_completeness is not None else None
        ),
        "coverage_verified_against_database": True,
        "missing_is_not_zero": True,
        "missing_are_absent_rows": True,
        "missing_definition": "No malaria value was received for the district-week (the row is absent).",
        "zero_definition": "A value of zero was explicitly reported.",
        "completeness_note": (
            "Reporting completeness reflects HMIS 033b form submission. It does not "
            "measure malaria burden or incidence."
        ),
        "data_element": "fUflbWWhouR (033B-CD01a) confirmed malaria cases",
    }
    return _envelope(
        data,
        limitations=[
            "Missing district-weeks are absent records, not zeros, and are never zero-filled.",
            "Counts describe received surveillance records, not true malaria burden.",
            "Reporting completeness is HMIS 033b submission, not incidence.",
        ],
        metadata={"coverage_source": "project_state realignment evidence", "project_state_write": False},
    )


@bp.route("/api/lifecycle/data-quality")
@api_login_required
def api_data_quality():
    """Preparation-stage statuses + evidence-derived counts and leakage controls.

    The evidence itself is built by the single approved builder in
    ``app.services.data_quality_evidence`` so this endpoint and the Phase 29D
    Data Quality report can never disagree. Any missing, malformed or
    inconsistent required evidence fails closed with a controlled 503.
    """
    from app.services.data_quality_evidence import (
        DataQualityEvidenceError,
        build_data_quality_evidence,
    )

    try:
        evidence = build_data_quality_evidence()
    except DataQualityEvidenceError as exc:
        return _unavailable(str(exc))
    except Exception:  # noqa: BLE001 - never leak internals
        return _unavailable("Data-quality evidence could not be read.")

    data = {
        "stages": evidence["stages"],
        "panel_rows": evidence["panel_rows"],
        "active_districts": evidence["active_districts"],
        "rows_with_complete_climate_lags": evidence["rows_with_complete_climate_lags"],
        "climate_lag_columns_verified": evidence["climate_lag_columns_verified"],
        "leakage_controls": evidence["leakage_controls"],
    }
    return _envelope(
        data,
        warnings=evidence["warnings"],
        limitations=evidence["limitations"],
        metadata={
            "leakage_source": evidence["evidence_sources"],
            "project_state_write": False,
        },
    )
