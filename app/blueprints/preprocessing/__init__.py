from flask import Blueprint

bp = Blueprint("preprocessing", __name__, url_prefix="/preprocessing")

from app.blueprints.preprocessing import routes  # noqa: E402,F401  (register routes)
