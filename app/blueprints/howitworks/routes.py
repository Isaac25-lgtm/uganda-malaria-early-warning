"""How It Works — a small authenticated, read-only system-guide page.

The page shell is server-rendered; live status comes from the existing
/api/system/pipeline-status API (no CLI, no project-state computation in JS).
"""
from flask import render_template
from flask_login import login_required

from app.blueprints.howitworks import bp


@bp.route("/")
@login_required
def index():
    return render_template("lifecycle/how_it_works.html")
