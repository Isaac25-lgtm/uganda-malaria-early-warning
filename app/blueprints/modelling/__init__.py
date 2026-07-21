from flask import Blueprint

bp = Blueprint("modelling", __name__, url_prefix="/modelling")

from app.blueprints.modelling import routes  # noqa: E402,F401  (register routes)
