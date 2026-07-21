"""Entry point for the Malaria Early Warning Dashboard.

Exposes ``app`` for the Flask CLI (``flask db ...``,
``flask verify-project-state``) and for WSGI servers. The application is built
by the factory in ``app.create_app`` and fails fast if ``DATABASE_URL`` is not
set.

Sign-in verifies live DHIS2 credentials; there is no local account command.
"""
# Load a local .env (if python-dotenv is installed) BEFORE the app is created,
# so configuration is available when the factory reads the environment.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional
    pass

from app import create_app  # noqa: E402  (import after dotenv load)

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
