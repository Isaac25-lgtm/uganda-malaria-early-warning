from flask import Blueprint

bp = Blueprint("forecasts", __name__, url_prefix="/forecasts")

from app.blueprints.forecasts import routes  # noqa: E402,F401  (register routes)
