from flask import Blueprint

bp = Blueprint("explainability", __name__)

from app.blueprints.explainability import routes  # noqa: E402,F401  (register routes)
