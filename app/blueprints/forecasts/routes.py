"""Forecasts & Risk lifecycle page (read-only presentation).

Filter options (issue weeks, districts, horizons) are rendered server-side from
the same read-only service the dashboard uses, so the page never invents a week
or district. Panel values load from the existing dashboard APIs via forecasts.js.
"""
from flask import render_template
from flask_login import login_required

from app.blueprints.forecasts import bp
from app.services.dashboard_data import (
    DEFAULT_HORIZON,
    build_filters_section,
    dashboard_precondition_errors,
)


@bp.route("/")
@login_required
def index():
    errors = dashboard_precondition_errors(require_outputs=True)
    if errors:
        return render_template(
            "lifecycle/forecasts_risk.html",
            filters={"horizons": [], "districts": [], "issues": []},
            default_horizon=DEFAULT_HORIZON,
        )
    return render_template(
        "lifecycle/forecasts_risk.html",
        filters=build_filters_section(),
        default_horizon=DEFAULT_HORIZON,
    )
