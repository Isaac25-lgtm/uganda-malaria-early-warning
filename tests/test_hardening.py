"""Phase 29F hardening regression tests.

Locks in the two defects corrected during the final audit:

1. Project-state evidence was re-parsed twice per governance request (~4.6s each).
   It is now memoised for the lifetime of one request. These tests prove the
   memo is request-scoped, read-only, copy-safe and analytically inert.

2. Two superseded templates still claimed Reports/Settings were "planned for a
   later implementation phase" / "not included". Those claims are gone.

Also carries the system-wide audit assertions (auth coverage, leak-free
responses, honest forecast presentation, config hardening) so a regression in
any of them fails the suite rather than only the audit script.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from app import create_app

# Central authentication inventory. Every authenticated HTML view belongs in
# PAGES (they redirect anonymous users) and every JSON endpoint in APIS (they
# return JSON 401). Requests must be VALID so a 400 can never mask a missing
# auth check. `/auth/login` is deliberately absent — it must stay public — and
# `/auth/logout` is asserted separately because it redirects rather than 401s.
PAGES = ["/", "/dhis2/", "/preprocessing/", "/climate/", "/modelling/", "/forecasts/",
         "/how-it-works/", "/reports/", "/settings/",
         "/reports/print?report_type=methods"]                       # authenticated print view
APIS = ["/api/dashboard/summary", "/api/dashboard/weekly-series", "/api/dashboard/risk-table",
        "/api/dashboard/map", "/api/dashboard/climate-summary", "/api/dashboard/model-performance",
        "/api/explainability/overview", "/api/explainability/model-selection",
        "/api/explainability/validation-method", "/api/explainability/threshold-method",
        "/api/explainability/feature-importance?model=random_forest",
        "/api/system/pipeline-status", "/api/system/governance",
        "/api/lifecycle/surveillance", "/api/lifecycle/data-quality",
        "/reports/generate?report_type=methods&format=pdf"]          # report generation
LEAK_TOKENS = ["postgresql://", "SECRET_KEY", "DHIS2_PASSWORD", "Traceback",
               "artifact_paths", "proof_path", ".joblib", "/home/"]


class SupersededTemplateTests(unittest.TestCase):
    """Defect 2: superseded templates must not contradict the shipped system."""

    def test_superseded_templates_carry_no_false_deferral_claim(self):
        for rel in ("app/templates/lifecycle/reports.html",
                    "app/templates/lifecycle/settings.html"):
            text = Path(rel).read_text(encoding="utf-8")
            self.assertNotIn("planned for a later implementation phase", text, rel)
            self.assertNotIn("not included in the current research implementation", text, rel)
            self.assertIn("SUPERSEDED", text, rel)

    def test_live_pages_are_the_implemented_ones(self):
        reports_routes = Path("app/blueprints/reports/routes.py").read_text(encoding="utf-8")
        settings_routes = Path("app/blueprints/settings/routes.py").read_text(encoding="utf-8")
        self.assertIn("reports/index.html", reports_routes)
        self.assertIn("settings/index.html", settings_routes)
        self.assertNotIn("lifecycle/reports.html", reports_routes)
        self.assertNotIn("lifecycle/settings.html", settings_routes)


class ProjectStateMemoTests(unittest.TestCase):
    """Defect 1: read-only project state is computed once per request."""

    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL required.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.app.config["WTF_CSRF_ENABLED"] = False

    def _client(self):
        from app.services.dhis2_auth import DHIS2_IDENTITY_SESSION_KEY, DHIS2SessionUser
        ident = DHIS2SessionUser(id="hardening", display_name="TESTER",
                                 verified_at=int(datetime.now(timezone.utc).timestamp()))
        client = self.app.test_client()
        with client.session_transaction() as s:
            s[DHIS2_IDENTITY_SESSION_KEY] = ident.to_session()
            s["_user_id"] = ident.get_id()
            s["_fresh"] = True
        return client

    def _count_recomputes(self, url):
        import app.services.project_state as ps
        calls = {"n": 0}
        real = ps._reference_and_boundaries

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        with patch("app.services.project_state._reference_and_boundaries", side_effect=counting):
            self._client().get(url)
        return calls["n"]

    def test_governance_computes_project_state_once_per_request(self):
        # Governance needs it for pipeline status AND data-quality evidence.
        self.assertEqual(self._count_recomputes("/api/system/governance"), 1)

    def test_memo_does_not_leak_between_requests(self):
        # Each new request must recompute; the memo is request-scoped only.
        self.assertEqual(self._count_recomputes("/api/system/pipeline-status"), 1)
        self.assertEqual(self._count_recomputes("/api/system/pipeline-status"), 1)

    def test_no_caching_outside_a_request_context(self):
        import app.services.project_state as ps
        calls = {"n": 0}
        real = ps._reference_and_boundaries

        def counting(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        with self.app.app_context():
            with patch("app.services.project_state._reference_and_boundaries", side_effect=counting):
                first = ps.build_project_state_summary(write=False)
                second = ps.build_project_state_summary(write=False)
        # Outside a request there is no memo: both calls recompute.
        self.assertEqual(calls["n"], 2)
        self.assertIsNot(first, second)
        self.assertEqual(set(first["components"]), set(second["components"]))

    def test_memo_returns_copies_so_callers_cannot_poison_it(self):
        import app.services.project_state as ps
        with self.app.test_request_context("/"):
            first = ps.build_project_state_summary(write=False)
            first["components"]["injected"] = "tamper"
            second = ps.build_project_state_summary(write=False)
        self.assertNotIn("injected", second["components"])

    def test_memoised_summary_is_analytically_identical(self):
        """A cached summary must equal a fresh one in EVERY field but the timestamp.

        `generated_at_utc` is inherently variable (it records when the audit ran),
        so it is the only field excluded. Everything else — component statuses,
        row_counts, artifact_paths, blocking_reason, next_required_action,
        details, gating_logic and secrets — must match exactly, proving the memo
        is analytically inert rather than merely status-equivalent.
        """
        import app.services.project_state as ps

        def without_timestamp(summary):
            stripped = copy.deepcopy(summary)
            stripped.pop("generated_at_utc", None)
            return stripped

        with self.app.test_request_context("/"):
            cached_first = ps.build_project_state_summary(write=False)
            cached_again = ps.build_project_state_summary(write=False)   # served from memo
        with self.app.app_context():
            fresh = ps.build_project_state_summary(write=False)

        # Whole-object equality, not a status-only spot check.
        self.assertEqual(without_timestamp(cached_first), without_timestamp(fresh))
        self.assertEqual(without_timestamp(cached_again), without_timestamp(fresh))

        # Guard the exclusion: the timestamp must be the ONLY differing key, and
        # the deep structures the earlier version never inspected must be present.
        differing = [k for k in fresh
                     if cached_first.get(k) != fresh.get(k)]
        self.assertTrue(set(differing) <= {"generated_at_utc"}, differing)
        for required in ("components", "gating_logic", "secrets",
                         "safe_to_proceed_to_prompt_29", "recommended_next_action"):
            self.assertIn(required, fresh)
        sample = fresh["components"]["dhis2_malaria_panel"]
        for required in ("status", "row_counts", "artifact_paths",
                         "blocking_reason", "next_required_action", "details"):
            self.assertIn(required, sample, required)
            self.assertEqual(cached_first["components"]["dhis2_malaria_panel"][required],
                             sample[required], required)


class SystemAuditTests(unittest.TestCase):
    """System-wide assertions carried over from the Phase 29F audit sweep."""

    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL required.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.app.config["WTF_CSRF_ENABLED"] = False

    def _client(self, expired=False):
        from app.services.dhis2_auth import DHIS2_IDENTITY_SESSION_KEY, DHIS2SessionUser
        verified = int(datetime.now(timezone.utc).timestamp()) - (3600 if expired else 0)
        ident = DHIS2SessionUser(id="audit", display_name="TESTER", verified_at=verified)
        client = self.app.test_client()
        with client.session_transaction() as s:
            s[DHIS2_IDENTITY_SESSION_KEY] = ident.to_session()
            s["_user_id"] = ident.get_id()
            s["_fresh"] = True
        return client

    def test_every_page_requires_authentication(self):
        anon = self.app.test_client()
        self.assertEqual(len(PAGES), 10, "page inventory is incomplete")
        for url in PAGES:
            resp = anon.get(url)
            self.assertEqual(resp.status_code, 302, url)
            self.assertIn("/auth/login", resp.headers["Location"], url)

    def test_logout_requires_authentication_and_login_stays_public(self):
        anon = self.app.test_client()
        # Logout is protected but redirects rather than returning JSON 401.
        logout = anon.get("/auth/logout")
        self.assertEqual(logout.status_code, 302)
        self.assertIn("/auth/login", logout.headers["Location"])
        # The login page itself must remain reachable, or nobody could sign in.
        self.assertEqual(anon.get("/auth/login").status_code, 200)

    def test_every_api_returns_json_401_when_anonymous(self):
        anon = self.app.test_client()
        self.assertEqual(len(APIS), 16, "API inventory is incomplete")
        for url in APIS:
            resp = anon.get(url)
            self.assertEqual(resp.status_code, 401, url)
            self.assertEqual(resp.mimetype, "application/json", url)

    def test_auth_inventory_covers_every_registered_route(self):
        """A new Flask route cannot silently escape the central auth inventory."""
        inventoried = {url.split("?", 1)[0] for url in PAGES + APIS}
        inventoried.update({"/auth/login", "/auth/logout", "/static/<path:filename>"})
        registered = {rule.rule for rule in self.app.url_map.iter_rules()}

        self.assertEqual(len(registered), 29, registered)
        self.assertEqual(inventoried, registered)

    def test_expired_session_rejected_everywhere(self):
        expired = self._client(expired=True)
        for url in APIS:
            self.assertEqual(expired.get(url).status_code, 401, url)

    def test_no_secret_or_path_leakage_in_any_api(self):
        client = self._client()
        for url in APIS:
            body = client.get(url).get_data().decode("utf-8", errors="replace")
            for token in LEAK_TOKENS:
                self.assertNotIn(token, body, f"{url} leaked {token}")
            self.assertIsNone(re.search(r"[A-Za-z]:\\\\", body), url)

    def test_invalid_parameters_return_controlled_errors(self):
        client = self._client()
        for url in ("/api/dashboard/summary?horizon=7",
                    "/api/explainability/feature-importance?model=baseline",
                    "/reports/generate?report_type=nope&format=pdf",
                    "/reports/generate?report_type=methods&format=xlsx"):
            resp = client.get(url)
            self.assertEqual(resp.status_code, 400, url)
            body = resp.get_data(as_text=True)
            self.assertNotIn("Traceback", body, url)
            self.assertNotIn("sqlalchemy", body.lower(), url)

    def test_forecast_presentation_stays_honest(self):
        data = self._client().get("/api/dashboard/weekly-series?horizon=4").get_json()["data"]
        forecast_rows = [r for r in data["rows"] if r["period"] == "forecast"]
        self.assertEqual(len(forecast_rows), 1, "must be a single target-week forecast")
        if not data["intervals_available"]:
            for row in data["rows"]:
                self.assertIsNone(row["prediction_interval_lower"])
                self.assertIsNone(row["prediction_interval_upper"])

    def test_no_placeholder_or_mojibake_in_rendered_pages(self):
        client = self._client()
        banned = ["TODO", "FIXME", "Lorem ipsum", "coming soon", "Awaiting pipeline output",
                  "Awaiting model output", "planned for a later implementation phase",
                  "â€œ", "â€™", "â†’", "RÂ²"]
        for url in PAGES:
            html = client.get(url).get_data(as_text=True)
            for token in banned:
                self.assertNotIn(token, html, f"{url} contains {token!r}")

    def test_config_hardening_holds(self):
        self.assertTrue(self.app.json.compact)
        self.assertNotIn("debug=True", Path("run.py").read_text(encoding="utf-8"))
        map_bytes = len(self._client().get("/api/dashboard/map").data)
        self.assertLessEqual(map_bytes, 2_000_000, f"map response is {map_bytes} bytes")

    def test_no_external_cdn_or_font_requests(self):
        for root in (Path("app/templates"), Path("app/static/js"), Path("app/static/css")):
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in (".html", ".js", ".css"):
                    text = path.read_text(encoding="utf-8", errors="replace")
                    for token in ("https://cdn.", "https://unpkg.com", "https://fonts.googleapis",
                                  "https://fonts.gstatic", "tile.openstreetmap"):
                        self.assertNotIn(token, text, f"{path} references {token}")

    def test_production_fingerprint_unchanged_by_browsing(self):
        from app.models import ForecastOutput

        def fingerprint():
            with self.app.app_context():
                rows = ForecastOutput.query.order_by(ForecastOutput.id).all()
                payload = [[r.id, r.district_id, r.model_name, r.horizon_weeks,
                            str(r.predicted_cases), str(r.threshold_used), r.risk_level,
                            r.pipeline_run_id] for r in rows]
                return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

        before = fingerprint()
        client = self._client()
        for url in PAGES + APIS:
            client.get(url)
        self.assertEqual(before, fingerprint())
        self.assertEqual(
            before, "2854b51afc0e06e1284232a584b4a8b91e3ea60e6a79e82452b850396cc144f9")


if __name__ == "__main__":
    unittest.main()
