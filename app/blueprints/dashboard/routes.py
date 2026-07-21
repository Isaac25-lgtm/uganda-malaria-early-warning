"""Main dashboard page and dashboard data API routes.

Every API endpoint builds only its own section, so a card refresh never loads or
serialises the district boundary. Only ``/api/dashboard/map`` touches GeoJSON.

The page requires a live-DHIS2-verified browser session. APIs answer ``fetch``
with JSON 401 instead of an HTML redirect so the browser console stays clean
when that session expires.
"""
from functools import wraps

from flask import jsonify, render_template, request
from flask_login import current_user, login_required

from app.blueprints.dashboard import bp
from app.services.dashboard_data import (
    DEFAULT_HORIZON,
    DashboardDataError,
    DashboardParameterError,
    build_climate_section,
    build_filters_section,
    build_map_section,
    build_model_performance_section,
    build_risk_table_section,
    build_summary_section,
    build_weekly_series_section,
    dashboard_precondition_errors,
    parse_dashboard_params,
)


def api_login_required(view):
    """Require a session, but answer unauthenticated API calls with JSON 401."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            return (
                jsonify(
                    {
                        "status": "unauthenticated",
                        "reason": "Your session has ended. Sign in again to load dashboard data.",
                        "data": None,
                        "empty_states": ["Sign in to load dashboard data."],
                    }
                ),
                401,
            )
        return view(*args, **kwargs)

    return wrapped


@bp.route("/")
@login_required
def index():
    """Render the dashboard shell with real filter options.

    Filter options are rendered server-side so the page never invents a week or
    a district. Panel values are loaded from the APIs by dashboard.js.
    """
    errors = dashboard_precondition_errors(require_outputs=True)
    if errors:
        return render_template(
            "dashboard/index.html",
            dashboard_status="unavailable",
            dashboard_reason="; ".join(errors),
            filters={"horizons": [], "districts": [], "issues": []},
            default_horizon=DEFAULT_HORIZON,
            default_issue=None,
        )

    filters = build_filters_section()
    issues = filters.get("issues") or []
    return render_template(
        "dashboard/index.html",
        dashboard_status="available",
        dashboard_reason=None,
        filters=filters,
        default_horizon=DEFAULT_HORIZON,
        default_issue=issues[0] if issues else None,
    )


@bp.route("/api/dashboard/summary")
@api_login_required
def api_dashboard_summary():
    return _dashboard_section(build_summary_section)


@bp.route("/api/dashboard/weekly-series")
@api_login_required
def api_dashboard_weekly_series():
    return _dashboard_section(build_weekly_series_section)


@bp.route("/api/dashboard/risk-table")
@api_login_required
def api_dashboard_risk_table():
    return _dashboard_section(build_risk_table_section)


@bp.route("/api/dashboard/map")
@api_login_required
def api_dashboard_map():
    return _dashboard_section(build_map_section)


@bp.route("/api/dashboard/climate-summary")
@api_login_required
def api_dashboard_climate_summary():
    return _dashboard_section(build_climate_section)


@bp.route("/api/dashboard/model-performance")
@api_login_required
def api_dashboard_model_performance():
    return _dashboard_section(build_model_performance_section)


def _dashboard_section(builder):
    """Validate parameters, then build one section only."""
    try:
        params = parse_dashboard_params(request.args)
    except DashboardParameterError as exc:
        return (
            jsonify(
                {
                    "status": "invalid_request",
                    "reason": str(exc),
                    "data": None,
                    "params": None,
                    "empty_states": [str(exc)],
                }
            ),
            400,
        )

    errors = dashboard_precondition_errors(require_outputs=True)
    if errors:
        return (
            jsonify(
                {
                    "status": "unavailable",
                    "reason": "; ".join(errors),
                    "data": None,
                    "params": params.to_dict(),
                    "empty_states": ["Forecast output unavailable."],
                }
            ),
            503,
        )

    if params.issue_year is None or params.issue_week is None:
        return (
            jsonify(
                {
                    "status": "unavailable",
                    "reason": "No production forecast issue is available.",
                    "data": None,
                    "params": params.to_dict(),
                    "empty_states": ["Forecast output unavailable."],
                }
            ),
            503,
        )

    try:
        data = builder(params)
    except DashboardDataError as exc:
        return (
            jsonify(
                {
                    "status": "unavailable",
                    "reason": str(exc),
                    "data": None,
                    "params": params.to_dict(),
                    "empty_states": [str(exc)],
                }
            ),
            503,
        )

    return (
        jsonify(
            {
                "status": "available",
                "reason": None,
                "data": data,
                "params": params.to_dict(),
                "empty_states": [],
            }
        ),
        200,
    )
