from flask import Blueprint

bp = Blueprint("dhis2", __name__, url_prefix="/dhis2")

from app.blueprints.dhis2 import routes  # noqa: E402,F401  (register routes)
