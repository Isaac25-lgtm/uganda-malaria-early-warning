"""Climate Signals lifecycle page (read-only presentation)."""
from flask import render_template
from flask_login import login_required

from app.blueprints.climate import bp
from app.services.dashboard_data import (
    DEFAULT_HORIZON,
    build_filters_section,
    dashboard_precondition_errors,
)


@bp.route("/")
@login_required
def index():
    if dashboard_precondition_errors(require_outputs=True):
        return render_template(
            "lifecycle/climate.html",
            filters={"horizons": [], "districts": []},
            default_horizon=DEFAULT_HORIZON,
        )
    return render_template(
        "lifecycle/climate.html",
        filters=build_filters_section(),
        default_horizon=DEFAULT_HORIZON,
    )
