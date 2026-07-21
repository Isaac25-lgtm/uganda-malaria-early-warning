from flask import Blueprint

bp = Blueprint("climate", __name__, url_prefix="/climate")

from app.blueprints.climate import routes  # noqa: E402,F401  (register routes)
