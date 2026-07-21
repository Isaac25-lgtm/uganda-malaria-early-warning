"""Settings & Governance page (Phase 29E).

The page is a thin, authenticated shell. All governance evidence comes from the
canonical read-only endpoint ``GET /api/system/governance`` (registered beside
``/api/system/pipeline-status`` in the explainability blueprint), so there is a
single governance implementation rather than two.

Display preferences are handled entirely client-side in localStorage; the server
never stores a preference.
"""
from flask import render_template
from flask_login import login_required

from app.blueprints.settings import bp


@bp.route("/")
@login_required
def index():
    """Settings & Governance page. Values load from /api/system/governance."""
    return render_template("settings/index.html")
