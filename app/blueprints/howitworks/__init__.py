from flask import Blueprint

bp = Blueprint("howitworks", __name__, url_prefix="/how-it-works")

from app.blueprints.howitworks import routes  # noqa: E402,F401
