"""Application factory for the Malaria Early Warning Dashboard."""
from __future__ import annotations

from flask import Flask, request

from app.config import Config
from app.extensions import db, login_manager, migrate

# Sidebar navigation grouped by lifecycle stage. Each item is
# (endpoint, blueprint key, label, icon). Routes are unchanged; only labels,
# grouping and the new How It Works entry are added (Phase 29C).
NAV_GROUPS = [
    ("OVERVIEW", [
        ("dashboard.index", "dashboard", "Dashboard", "dashboard"),
    ]),
    ("DATA PIPELINE", [
        ("dhis2.index", "dhis2", "Surveillance Data", "dhis2"),
        ("preprocessing.index", "preprocessing", "Data Quality & Preparation", "preprocessing"),
        ("climate.index", "climate", "Climate Signals", "climate"),
    ]),
    ("ANALYTICS", [
        ("modelling.index", "modelling", "Model Evaluation", "modelling"),
        ("forecasts.index", "forecasts", "Forecasts & Risk", "forecasts"),
    ]),
    ("SYSTEM GUIDE", [
        ("howitworks.index", "howitworks", "How It Works", "info"),
    ]),
    ("OUTPUTS", [
        ("reports.index", "reports", "Reports & Exports", "reports"),
    ]),
    ("ADMINISTRATION", [
        ("settings.index", "settings", "Settings & Governance", "settings"),
    ]),
]

# Flat list retained for any code/tests that still expect (endpoint, key, label).
NAV_ITEMS = [(endpoint, key, label) for _group, items in NAV_GROUPS for endpoint, key, label, _icon in items]


def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__)

    # Static, non-sensitive defaults, then secrets + DB URL from the
    # environment (this call fails fast if DATABASE_URL is missing/invalid).
    app.config.from_object(config_class)
    config_class.init_runtime(app)
    # Keep API payloads compact in development and production.
    app.json.compact = True

    _init_extensions(app)
    _register_blueprints(app)
    _register_cli(app)
    _register_context(app)
    _register_security_headers(app)

    return app


def _init_extensions(app: Flask) -> None:
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    # Import models so Alembic autogenerate and the login user_loader see them.
    from app import models  # noqa: F401


def _register_blueprints(app: Flask) -> None:
    from app.blueprints.auth import bp as auth_bp
    from app.blueprints.climate import bp as climate_bp
    from app.blueprints.dashboard import bp as dashboard_bp
    from app.blueprints.dhis2 import bp as dhis2_bp
    from app.blueprints.explainability import bp as explainability_bp
    from app.blueprints.forecasts import bp as forecasts_bp
    from app.blueprints.howitworks import bp as howitworks_bp
    from app.blueprints.lifecycle_api import bp as lifecycle_api_bp
    from app.blueprints.modelling import bp as modelling_bp
    from app.blueprints.preprocessing import bp as preprocessing_bp
    from app.blueprints.reports import bp as reports_bp
    from app.blueprints.settings import bp as settings_bp

    for blueprint in (
        auth_bp,
        dashboard_bp,
        dhis2_bp,
        preprocessing_bp,
        climate_bp,
        modelling_bp,
        forecasts_bp,
        reports_bp,
        settings_bp,
        explainability_bp,
        howitworks_bp,
        lifecycle_api_bp,
    ):
        app.register_blueprint(blueprint)


def _register_cli(app: Flask) -> None:
    from app.cli import register_cli

    register_cli(app)


def _register_context(app: Flask) -> None:
    from app.blueprints.auth.forms import LogoutForm
    from app.models import User

    @app.shell_context_processor
    def _shell_ctx():  # pragma: no cover - convenience only
        return {"db": db, "User": User}

    @app.context_processor
    def _inject_identity():
        return {
            "app_name": app.config["APP_NAME"],
            "app_subtitle": app.config["APP_SUBTITLE"],
            "app_version": app.config["APP_VERSION"],
            "nav_items": NAV_ITEMS,
            "nav_groups": NAV_GROUPS,
            "logout_form": LogoutForm(),
        }


def _register_security_headers(app: Flask) -> None:
    """Apply browser hardening without changing application response bodies."""

    @app.after_request
    def _secure_response(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), geolocation=(), microphone=()"
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "base-uri 'self'; frame-ancestors 'none'; object-src 'none'",
        )
        if request.endpoint != "static":
            response.headers.setdefault("Cache-Control", "no-store")
            response.headers.setdefault("Pragma", "no-cache")
        if request.is_secure:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response
