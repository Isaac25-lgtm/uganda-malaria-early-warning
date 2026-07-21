from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from app import create_app
from app.services.dhis2_auth import (
    DHIS2_IDENTITY_SESSION_KEY,
    DHIS2SessionUser,
    authenticate_dhis2_user,
)
from app.services.dhis2_client import DHIS2AuthError


class DHIS2AuthenticationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL is required for authentication tests.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app.config["WTF_CSRF_ENABLED"] = False

    @staticmethod
    def _identity(*, verified_at: int | None = None) -> DHIS2SessionUser:
        return DHIS2SessionUser(
            id="dhis2-user-uid",
            display_name="Verified Analyst",
            verified_at=verified_at or int(time.time()),
        )

    def test_login_verifies_live_identity_and_session_contains_no_credentials(self):
        username = "submitted-dhis2-username"
        password = "submitted-dhis2-password"
        client = self.app.test_client()
        with patch(
            "app.blueprints.auth.routes.authenticate_dhis2_user",
            return_value=self._identity(),
        ) as authenticate:
            response = client.post(
                "/auth/login",
                data={"username": username, "password": password},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], "/")
        authenticate.assert_called_once_with(username, password)
        with client.session_transaction() as stored:
            serialized = json.dumps(dict(stored), sort_keys=True)
            self.assertIn(DHIS2_IDENTITY_SESSION_KEY, stored)
            self.assertNotIn(username, serialized)
            self.assertNotIn(password, serialized)
            self.assertNotIn("_remember", stored)
            self.assertFalse(stored.permanent)

    def test_rejected_dhis2_credentials_do_not_create_a_session(self):
        client = self.app.test_client()
        with patch(
            "app.blueprints.auth.routes.authenticate_dhis2_user",
            side_effect=DHIS2AuthError("rejected"),
        ):
            response = client.post(
                "/auth/login",
                data={"username": "wrong", "password": "wrong"},
                follow_redirects=False,
            )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login", response.headers["Location"])
        self.assertEqual(client.get("/").status_code, 302)
        with client.session_transaction() as stored:
            self.assertNotIn(DHIS2_IDENTITY_SESSION_KEY, stored)
            self.assertNotIn("_user_id", stored)

    def test_live_api_payload_is_reduced_to_non_secret_identity(self):
        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _traceback):
                return None

            @staticmethod
            def verify_login():
                return {
                    "id": "uid-123",
                    "displayName": "District Analyst",
                    "username": "must-not-be-retained",
                    "authorities": ["ALL"],
                }

        with patch(
            "app.services.dhis2_auth.DHIS2Client.from_credentials",
            return_value=FakeClient(),
        ) as factory:
            identity = authenticate_dhis2_user("operator", "operation-password")
        factory.assert_called_once_with("operator", "operation-password")
        self.assertEqual(identity.id, "uid-123")
        self.assertEqual(identity.display_name, "District Analyst")
        self.assertNotIn("operator", json.dumps(identity.to_session()))
        self.assertNotIn("operation-password", json.dumps(identity.to_session()))
        self.assertNotIn("must-not-be-retained", json.dumps(identity.to_session()))

    def test_expired_dhis2_verification_requires_login_again(self):
        client = self.app.test_client()
        expired = self._identity(verified_at=int(time.time()) - 31 * 60)
        with client.session_transaction() as stored:
            stored["_user_id"] = expired.get_id()
            stored["_fresh"] = True
            stored[DHIS2_IDENTITY_SESSION_KEY] = expired.to_session()
        response = client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login", response.headers["Location"])

    def test_logout_clears_the_complete_session(self):
        client = self.app.test_client()
        with patch(
            "app.blueprints.auth.routes.authenticate_dhis2_user",
            return_value=self._identity(),
        ):
            client.post(
                "/auth/login",
                data={"username": "operator", "password": "operation-password"},
            )
        response = client.post("/auth/logout", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        with client.session_transaction() as stored:
            self.assertNotIn(DHIS2_IDENTITY_SESSION_KEY, stored)
            self.assertNotIn("_user_id", stored)

    def test_logout_rejects_get_requests(self):
        client = self.app.test_client()
        self.assertEqual(client.get("/auth/logout").status_code, 405)

    def test_logout_requires_a_valid_csrf_token(self):
        client = self.app.test_client()
        identity = self._identity()
        with client.session_transaction() as stored:
            stored[DHIS2_IDENTITY_SESSION_KEY] = identity.to_session()
            stored["_user_id"] = identity.get_id()
            stored["_fresh"] = True

        self.app.config["WTF_CSRF_ENABLED"] = True
        try:
            response = client.post("/auth/logout", follow_redirects=False)
        finally:
            self.app.config["WTF_CSRF_ENABLED"] = False
        self.assertEqual(response.status_code, 400)

    def test_cli_prompts_for_operation_scoped_credentials_and_closes_client(self):
        class FakeClient:
            closed = False

            @staticmethod
            def verify_login():
                return {"id": "uid-123", "displayName": "CLI Analyst"}

            def close(self):
                self.closed = True

        fake = FakeClient()
        runner = self.app.test_cli_runner()
        with patch(
            "app.commands.dhis2.DHIS2Client.from_credentials",
            return_value=fake,
        ) as factory:
            result = runner.invoke(
                args=["dhis2", "verify-login"],
                input="cli-operator\noperation-password\n",
            )
        self.assertEqual(result.exit_code, 0, result.output)
        factory.assert_called_once_with("cli-operator", "operation-password")
        self.assertNotIn("operation-password", result.output)
        self.assertTrue(fake.closed)

    def test_sample_environment_has_no_dhis2_credential_fields(self):
        text = Path(".env.example").read_text(encoding="utf-8")
        self.assertNotIn("DHIS2_USERNAME", text)
        self.assertNotIn("DHIS2_PASSWORD", text)
        self.assertNotIn("ADMIN_PASSWORD", text)


if __name__ == "__main__":
    unittest.main()
