from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.services.dhis2_client import (
    DHIS2Client,
    DHIS2Config,
    DHIS2ConfigError,
    DHIS2HTTPError,
    format_week_period,
    load_dhis2_config,
    normalize_base_url,
)


class FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = "<hidden>"

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.auth = None

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DHIS2ClientTests(unittest.TestCase):
    def test_base_url_normalisation(self):
        self.assertEqual(
            normalize_base_url("https://hmis.example.org///"),
            "https://hmis.example.org",
        )

    def test_base_url_rejects_insecure_http(self):
        with self.assertRaisesRegex(DHIS2ConfigError, "must use https"):
            normalize_base_url("http://hmis.example.org")

    def test_base_url_rejects_embedded_credentials(self):
        with self.assertRaisesRegex(DHIS2ConfigError, "embedded credentials"):
            normalize_base_url("https://analyst:secret@hmis.example.org")

    def test_base_url_rejects_query_and_fragment(self):
        for value in (
            "https://hmis.example.org?token=secret",
            "https://hmis.example.org#fragment",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(DHIS2ConfigError, "query string or fragment"):
                    normalize_base_url(value)

    def test_missing_config_reports_required_names(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(DHIS2ConfigError) as ctx:
                load_dhis2_config("", "")
        message = str(ctx.exception)
        self.assertIn("DHIS2_BASE_URL", message)
        self.assertIn("DHIS2 username", message)
        self.assertIn("DHIS2 password", message)

    def test_explicit_credentials_are_cleared_when_client_closes(self):
        with patch.dict(
            os.environ,
            {"DHIS2_BASE_URL": "https://hmis.example.org"},
            clear=True,
        ):
            client = DHIS2Client.from_credentials("analyst", "operation-secret")
        self.assertEqual(client.session.auth, ("analyst", "operation-secret"))
        client.close()
        self.assertIsNone(client.session.auth)
        self.assertEqual(client.config.username, "")
        self.assertEqual(client.config.password, "")

    def test_config_rejects_disabled_tls_verification(self):
        with patch.dict(
            os.environ,
            {
                "DHIS2_BASE_URL": "https://hmis.example.org",
                "DHIS2_VERIFY_SSL": "false",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(DHIS2ConfigError, "must remain true"):
                load_dhis2_config("analyst", "operation-secret")

    def test_password_not_in_http_error(self):
        config = DHIS2Config(
            base_url="https://hmis.example.org",
            username="analyst",
            password="mock-password-value",
        )
        client = DHIS2Client(
            config,
            session=FakeSession([FakeResponse(500, {"message": "server"})]),
            retries=0,
            retry_sleep_seconds=0,
        )
        with self.assertRaises(DHIS2HTTPError) as ctx:
            client.get("/api/me")
        self.assertNotIn("mock-password-value", str(ctx.exception))

    def test_metadata_fetch_falls_back_to_indicator(self):
        config = DHIS2Config(
            base_url="https://hmis.example.org",
            username="analyst",
            password="secret",
        )
        session = FakeSession(
            [
                FakeResponse(404, {"message": "missing"}),
                FakeResponse(200, {"id": "abc", "name": "Malaria indicator"}),
            ]
        )
        client = DHIS2Client(config, session=session, retries=0, retry_sleep_seconds=0)
        resource_type, payload = client.fetch_metadata_object("abc")
        self.assertEqual(resource_type, "indicator")
        self.assertEqual(payload["id"], "abc")

    def test_week_period_formatting(self):
        self.assertEqual(format_week_period(2025, 1), "2025W01")
        self.assertEqual(format_week_period(2025, 20), "2025W20")

    def test_retry_for_transient_get(self):
        config = DHIS2Config(
            base_url="https://hmis.example.org",
            username="analyst",
            password="secret",
        )
        session = FakeSession(
            [FakeResponse(503, {"message": "busy"}), FakeResponse(200, {"ok": True})]
        )
        client = DHIS2Client(config, session=session, retries=1, retry_sleep_seconds=0)
        self.assertEqual(client.get("/api/me"), {"ok": True})
        self.assertEqual(len(session.calls), 2)


if __name__ == "__main__":
    unittest.main()
