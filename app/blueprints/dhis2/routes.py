"""Surveillance Data lifecycle page (read-only presentation)."""
from flask import render_template
from flask_login import login_required

from app.blueprints.dhis2 import bp


@bp.route("/")
@login_required
def index():
    return render_template("lifecycle/surveillance.html")
