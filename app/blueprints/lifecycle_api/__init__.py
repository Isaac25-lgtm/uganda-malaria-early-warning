from flask import Blueprint

bp = Blueprint("lifecycle_api", __name__)

from app.blueprints.lifecycle_api import routes  # noqa: E402,F401
