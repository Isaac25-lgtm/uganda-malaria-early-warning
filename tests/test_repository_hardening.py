from __future__ import annotations

import unittest
from pathlib import Path

from app import create_app


class SourceOnlyConfig:
    TESTING = True
    WTF_CSRF_ENABLED = False
    APP_NAME = "Test Malaria Dashboard"
    APP_SUBTITLE = "Source-only verification"
    APP_VERSION = "test"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    REMEMBER_COOKIE_HTTPONLY = True
    REMEMBER_COOKIE_SAMESITE = "Lax"
    DHIS2_LOGIN_SESSION_MINUTES = 30

    @staticmethod
    def init_runtime(app) -> None:
        app.config["SQLALCHEMY_DATABASE_URI"] = (
            "postgresql+psycopg2://unused:unused@localhost/unused"
        )
        app.config["SECRET_KEY"] = "source-only-test-secret"
        app.config["SESSION_COOKIE_SECURE"] = False
        app.config["REMEMBER_COOKIE_SECURE"] = False


class RepositoryHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app(SourceOnlyConfig)

    def test_security_headers_are_present_without_private_data(self):
        response = self.app.test_client().get("/auth/login")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertEqual(response.headers["Referrer-Policy"], "same-origin")
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        self.assertIn("no-store", response.headers["Cache-Control"])

    def test_hsts_is_only_emitted_for_https_requests(self):
        client = self.app.test_client()
        self.assertNotIn("Strict-Transport-Security", client.get("/auth/login").headers)
        secure = client.get("/auth/login", base_url="https://localhost")
        self.assertIn("max-age=31536000", secure.headers["Strict-Transport-Security"])

    def test_logout_is_not_a_get_endpoint(self):
        response = self.app.test_client().get("/auth/logout")
        self.assertEqual(response.status_code, 405)

    def test_public_configuration_contains_no_live_metadata_uid(self):
        example = Path(".env.example").read_text(encoding="utf-8")
        self.assertIn("DHIS2_BASE_URL=https://", example)
        self.assertIn("SESSION_COOKIE_SECURE=false", example)
        self.assertNotIn("fUflbWWhouR", example)
        self.assertNotIn("u6xATVzE7VO", example)

    def test_readme_does_not_overclaim_ethics_or_public_reproducibility(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        self.assertNotIn("no individual-level ethical approval was required", readme)
        self.assertNotIn("Everything is reproducible and auditable", readme)
        self.assertNotIn("Fixed population denominators", readme)


if __name__ == "__main__":
    unittest.main()
