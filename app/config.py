"""Application configuration.

PostgreSQL is mandatory from the proof-of-concept stage onward. There is
deliberately NO SQLite fallback and no hardcoded database credentials: the
connection string is read from ``DATABASE_URL`` and the application fails fast
with a helpful message if it is absent or invalid.
"""
from __future__ import annotations

import os
import secrets
import warnings
from datetime import timedelta


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def resolve_bool_env(name: str, *, default: bool) -> bool:
    """Read a conventional boolean environment variable or fail clearly."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false.")


def resolve_database_url() -> str:
    """Return a validated PostgreSQL URL from the environment, or raise."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ConfigError(
            "DATABASE_URL is not set.\n"
            "This application is PostgreSQL-only and has no SQLite fallback.\n"
            "Set it before starting the app or running any flask command, e.g.:\n"
            "  set DATABASE_URL="
            "postgresql+psycopg2://postgres:PASSWORD@localhost:5432/malaria_ews\n"
            "See .env.example for all required variables."
        )

    # Some managed providers still emit the legacy 'postgres://' scheme, which
    # SQLAlchemy no longer accepts; normalise it to 'postgresql://'.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]

    if url.startswith("sqlite"):
        raise ConfigError(
            "SQLite is not supported. This application requires PostgreSQL. "
            "Point DATABASE_URL at a PostgreSQL server."
        )
    if not url.startswith("postgresql"):
        raise ConfigError(
            "DATABASE_URL must be a PostgreSQL connection string (it should "
            "start with 'postgresql'). Only PostgreSQL is supported; there is "
            "no SQLite fallback."
        )
    return url


def resolve_secret_key() -> str:
    """Return SECRET_KEY from the environment, or a temporary dev key."""
    key = os.environ.get("SECRET_KEY")
    if key:
        return key

    is_development = os.environ.get("FLASK_ENV") == "development" or os.environ.get(
        "FLASK_DEBUG"
    ) in {"1", "true", "True"}
    if not is_development:
        raise ConfigError(
            "SECRET_KEY is not set. Set SECRET_KEY in the environment before "
            "running outside local development."
        )

    warnings.warn(
        "SECRET_KEY is not set; generating a temporary key for this process. "
        "Sessions will not survive a restart. Set SECRET_KEY for real use.",
        stacklevel=2,
    )
    return secrets.token_urlsafe(48)


class Config:
    """Base configuration applied to every application instance."""

    # Product identity (surfaced in templates). Foundation version.
    APP_NAME = "Malaria Early Warning Dashboard"
    APP_SUBTITLE = "Uganda District-Level Surge Prediction"
    APP_VERSION = "0.1.0"

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Verify connections before use so a restarted Postgres doesn't wedge the app.
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)
    DHIS2_LOGIN_SESSION_MINUTES = 30

    @staticmethod
    def init_runtime(app) -> None:
        """Populate secret + database settings, failing fast if invalid."""
        app.config["SQLALCHEMY_DATABASE_URI"] = resolve_database_url()
        app.config["SECRET_KEY"] = resolve_secret_key()
        app.config["SESSION_COOKIE_SECURE"] = resolve_bool_env(
            "SESSION_COOKIE_SECURE", default=False
        )
        app.config["REMEMBER_COOKIE_SECURE"] = app.config["SESSION_COOKIE_SECURE"]
