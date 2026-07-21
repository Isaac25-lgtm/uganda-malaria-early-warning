"""Phase 29E Settings & Governance tests.

Covers authentication/expiry, the eleven evidence-derived sections, fail-closed
behaviour on malformed required evidence, absence of secrets/paths/roles, the
report-registry-sourced availability, the display-preference allowlist contract,
SQL no-writes and a deterministic forecast snapshot.
"""
from __future__ import annotations

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
from app.extensions import db

GOVERNANCE = "/api/system/governance"   # canonical route
SECTIONS = (
    "system_overview", "authentication", "data_provenance", "pipeline", "models",
    "forecasts", "data_quality_rules", "reports", "integrity", "display_preferences",
    "research",
)


class GovernancePureTests(unittest.TestCase):
    def test_database_targets_require_one_valid_value_per_horizon(self):
        from app.services.governance import GovernanceEvidenceError, _validated_database_targets

        self.assertEqual(
            _validated_database_targets([(4, 2026, 4), (8, 2026, 8), (12, 2026, 12)]),
            {4: (2026, 4), 8: (2026, 8), 12: (2026, 12)},
        )
        broken = (
            [(4, 2026, 4), (4, 2026, 5), (8, 2026, 8), (12, 2026, 12)],
            [(4, 2026, 4), (8, 2026, 8)],
            [(4, 2026, 4), (8, 2026, 8), (12, 2026, 12), (6, 2026, 6)],
            [(4, 2026, 0), (8, 2026, 8), (12, 2026, 12)],
        )
        for rows in broken:
            with self.subTest(rows=rows):
                with self.assertRaises(GovernanceEvidenceError):
                    _validated_database_targets(rows)

    def test_preference_schema_is_versioned_and_allowlisted(self):
        from app.services.governance import PREFERENCE_SCHEMA
        self.assertEqual(PREFERENCE_SCHEMA["version"], 1)
        # Only preferences with genuinely observable behaviour are published.
        self.assertEqual(set(PREFERENCE_SCHEMA["preferences"]),
                         {"default_horizon", "reduce_motion"})
        for spec in PREFERENCE_SCHEMA["preferences"].values():
            self.assertIn("effect", spec)
        # The note must promise the preferences never touch the server or analytics.
        note = PREFERENCE_SCHEMA["note"].lower()
        self.assertIn("localstorage", note)
        self.assertIn("never enter the flask session", note)

    def test_settings_js_declares_matching_allowlist(self):
        js = Path("app/static/js/preferences.js").read_text(encoding="utf-8")
        for key in ("default_horizon", "reduce_motion"):
            self.assertIn(key, js)
        # Unimplemented preferences must be absent everywhere.
        settings_js = Path("app/static/js/settings.js").read_text(encoding="utf-8")
        settings_html = Path("app/templates/settings/index.html").read_text(encoding="utf-8")
        for removed in ("density", "number_format"):
            self.assertNotIn(removed, js)
            self.assertNotIn("pref-" + removed, settings_js)
            self.assertNotIn("pref-" + removed, settings_html)
        # Preferences are localStorage-only and versioned; they are never uploaded.
        self.assertIn("localStorage", js)
        self.assertIn("mews.preferences.v1", js)
        self.assertNotIn('method: "POST"', js)
        self.assertNotIn('method: "PUT"', js)
        # The shared preferences module makes no network call at all.
        self.assertNotIn("fetch(", js)
        # The Settings page's only network call is the read-only governance GET.
        self.assertEqual(settings_js.count("fetch("), 1)
        self.assertIn("/api/system/governance", settings_js)

    def test_reports_page_applies_canonical_governance_link(self):
        js = Path("app/static/js/reports.js").read_text(encoding="utf-8")
        self.assertIn("URLSearchParams", js)
        self.assertIn('.get("report_type")', js)
        self.assertIn("applyRequestedReport();", js)


class GovernanceRouteTests(unittest.TestCase):
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
        ident = DHIS2SessionUser(id="gov-test", display_name="TESTER", verified_at=verified)
        client = self.app.test_client()
        with client.session_transaction() as s:
            s[DHIS2_IDENTITY_SESSION_KEY] = ident.to_session()
            s["_user_id"] = ident.get_id()
            s["_fresh"] = True
        return client

    # ---- auth ----------------------------------------------------------
    def test_settings_page_and_api_require_auth(self):
        anon = self.app.test_client()
        self.assertEqual(anon.get("/settings/").status_code, 302)
        resp = anon.get(GOVERNANCE)
        self.assertEqual(resp.status_code, 401)
        self.assertEqual(resp.mimetype, "application/json")
        self.assertEqual(resp.get_json()["status"], "unauthenticated")

    def test_expired_session_rejected(self):
        self.assertEqual(self._client(expired=True).get(GOVERNANCE).status_code, 401)

    def test_page_renders_under_auth(self):
        self.assertEqual(self._client().get("/settings/").status_code, 200)

    # ---- envelope + sections -------------------------------------------
    def test_governance_returns_all_eleven_sections(self):
        response = self._client().get(GOVERNANCE)
        self.assertEqual(response.headers.get("Cache-Control"), "no-store")
        body = response.get_json()
        self.assertEqual(body["status"], "available")
        for key in ("status", "reason", "data", "params", "empty_states",
                    "metadata", "warnings", "limitations"):
            self.assertIn(key, body)
        for section in SECTIONS:
            self.assertIn(section, body["data"], section)

    # ---- evidence-derived (not hardcoded) ------------------------------
    def test_integrity_is_recomputed_live_with_canonical_fingerprint(self):
        it = self._client().get(GOVERNANCE).get_json()["data"]["integrity"]
        self.assertEqual(it["production_forecast_rows"], 1752)
        self.assertEqual(it["active_districts"], 146)
        self.assertEqual(it["null_predicted_cases"], 0)
        self.assertEqual(it["duplicate_forecast_keys"], 0)
        self.assertTrue(it["single_production_issue"])
        # Canonical deterministic fingerprint, including pipeline_run_id.
        self.assertEqual(it["forecast_fingerprint_sha256"],
                         "2854b51afc0e06e1284232a584b4a8b91e3ea60e6a79e82452b850396cc144f9")

    def test_composite_key_set_is_fully_verified(self):
        it = self._client().get(GOVERNANCE).get_json()["data"]["integrity"]
        # 146 districts x 4 approved models x 3 horizons x 1 issue.
        self.assertEqual(it["expected_composite_keys"], 146 * 4 * 3)
        self.assertEqual(it["actual_composite_keys"], 1752)
        self.assertEqual(it["missing_composite_keys"], 0)
        self.assertEqual(it["unexpected_composite_keys"], 0)
        self.assertTrue(it["composite_key_set_complete"])

    def test_pipeline_components_are_plain_status_strings(self):
        # Regression: the pipeline loader returns {status, row_counts} objects.
        # Governance must flatten them, or the page renders "[object Object]".
        components = self._client().get(GOVERNANCE).get_json()["data"]["pipeline"]["components"]
        self.assertEqual(len(components), 9)
        for name, status in components.items():
            self.assertIsInstance(status, str, f"{name} status must be a string")
            self.assertNotIn("object", status.lower())

    def test_models_match_approved_selection(self):
        m = self._client().get(GOVERNANCE).get_json()["data"]["models"]
        self.assertEqual(set(m["approved_models"]),
                         {"Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost"})
        self.assertEqual(m["selected_models_by_horizon"]["4"]["count_model"], "Random Forest")
        self.assertEqual(m["selected_models_by_horizon"]["8"]["count_model"], "Random Forest")
        self.assertEqual(m["selected_models_by_horizon"]["12"]["count_model"], "Random Forest")
        for h in ("4", "8", "12"):
            self.assertEqual(m["selected_models_by_horizon"][h]["surge_model"], "Random Forest")

    def test_reports_status_comes_from_registry(self):
        from app.services.report_data_service import report_registry
        reports = self._client().get(GOVERNANCE).get_json()["data"]["reports"]
        with self.app.app_context():
            registry = report_registry()
        self.assertEqual(reports["report_count"], len(registry))
        self.assertEqual(set(reports["supported_formats"]),
                         {f for r in registry for f in r["formats"]})

    def test_data_quality_section_uses_shared_evidence(self):
        dq = self._client().get(GOVERNANCE).get_json()["data"]["data_quality_rules"]
        self.assertTrue(dq["available"])
        self.assertEqual(dq["coverage"]["expected_district_weeks"], 45698)
        self.assertEqual(dq["coverage"]["missing_district_weeks"], 247)
        self.assertEqual(dq["rows_with_complete_climate_lags"], 45451)

    def test_authentication_section_states_no_roles_and_no_secrets(self):
        from app.services.dhis2_auth import DHIS2SessionUser

        a = self._client().get(GOVERNANCE).get_json()["data"]["authentication"]
        self.assertEqual(a["reauthentication_minutes"], 30)
        self.assertEqual(a["roles_supported"], [])
        # No unsupported global persistence boolean is claimed. Instead the
        # response states exactly which fields the session cookie carries.
        self.assertNotIn("credentials_persisted", a)
        self.assertNotIn("dhis2_secret_stored", a)
        self.assertIn("credential_storage_note", a)
        identity = DHIS2SessionUser(id="evidence", display_name="Evidence", verified_at=1)
        self.assertEqual(a["identity_fields_stored"], list(identity.to_session()))
        self.assertIn("identity payload", a["credential_storage_note"].lower())
        self.assertIn("flask-login", a["credential_storage_note"].lower())
        self.assertIn("does not pull new malaria data", a["login_vs_data_note"])

    def test_no_unsupported_roles_anywhere(self):
        blob = json.dumps(self._client().get(GOVERNANCE).get_json())
        for role in ("Administrator", "Epidemiologist", "MoH Analyst", "Data Scientist", "System Owner"):
            self.assertNotIn(role, blob)

    # ---- fail closed ---------------------------------------------------
    def test_pipeline_evidence_missing_fails_closed(self):
        from app.services import explainability as ex
        with patch("app.services.explainability.build_pipeline_status",
                   side_effect=ex.ExplainabilityError("gone")):
            resp = self._client().get(GOVERNANCE)
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["status"], "unavailable")

    def test_data_quality_degrades_without_failing_whole_page(self):
        from app.services import data_quality_evidence as dqe
        with patch("app.services.data_quality_evidence.build_data_quality_evidence",
                   side_effect=dqe.DataQualityEvidenceError("dq gone")):
            body = self._client().get(GOVERNANCE).get_json()
        # Optional data-quality degrades to unavailable; the page stays available.
        self.assertEqual(body["status"], "available")
        self.assertFalse(body["data"]["data_quality_rules"]["available"])
        self.assertTrue(any("data-quality evidence is unavailable" in w.lower()
                            for w in body["warnings"]), body["warnings"])

    def test_no_secrets_paths_or_traces_leak(self):
        blob = self._client().get(GOVERNANCE).get_data(as_text=True)
        for token in ("postgresql://", "SECRET_KEY", "Traceback", "artifact_paths",
                      "proof_path", ".joblib", "DHIS2_PASSWORD"):
            self.assertNotIn(token, blob, token)
        self.assertIsNone(re.search(r"[A-Za-z]:\\\\", blob))
        self.assertNotIn("password", blob.lower())

    # ---- read-only -----------------------------------------------------
    def test_governance_performs_no_sql_writes(self):
        from sqlalchemy import event
        with self.app.app_context():
            engine = db.engine
        statements = []

        def before(conn, cursor, statement, params, context, executemany):
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", before)
        try:
            with patch.object(db.session, "commit") as commit, patch.object(db.session, "flush") as flush:
                self._client().get(GOVERNANCE)
                commit.assert_not_called()
                flush.assert_not_called()
        finally:
            event.remove(engine, "before_cursor_execute", before)
        for statement in statements:
            head = statement.strip().split(None, 1)[0].upper() if statement.strip() else ""
            self.assertNotIn(head, {"INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER"})

    def test_forecast_snapshot_unchanged(self):
        from app.models import ForecastOutput

        def fingerprint():
            with self.app.app_context():
                rows = ForecastOutput.query.order_by(ForecastOutput.id).all()
                payload = [[r.id, r.district_id, r.model_name, r.horizon_weeks,
                            str(r.predicted_cases), str(r.threshold_used), r.risk_level] for r in rows]
                return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), len(rows)

        before = fingerprint()
        self._client().get(GOVERNANCE)
        self._client().get("/settings/")
        after = fingerprint()
        self.assertEqual(before, after)
        self.assertEqual(before[1], 1752)

    # ---- regression ----------------------------------------------------
    def test_phase_29abcd_regression_intact(self):
        client = self._client()
        for url in ("/", "/api/dashboard/summary", "/api/explainability/overview",
                    "/api/system/pipeline-status", "/api/lifecycle/data-quality", "/reports/"):
            self.assertEqual(client.get(url).status_code, 200, url)
        self.assertTrue(self.app.json.compact)

    def test_removed_settings_prefixed_governance_alias_is_404(self):
        self.assertEqual(self._client().get("/settings/api/system/governance").status_code, 404)


    # ---- corrective pass: derived claims --------------------------------
    def test_excluded_methods_and_predictors_are_derived_not_asserted(self):
        m = self._client().get(GOVERNANCE).get_json()["data"]["models"]
        methods = {x["method"]: x["present_in_approved_models"] for x in m["excluded_methods"]}
        self.assertIn("negative binomial", methods)
        self.assertFalse(methods["negative binomial"])
        predictors = {x["variable"]: x["present_in_approved_predictors"] for x in m["excluded_predictors"]}
        self.assertIn("wind", predictors)
        self.assertFalse(predictors["wind"])
        # Proven against the manifest's own lists rather than asserted.
        from app.services.explainability import load_json
        with self.app.app_context():
            manifest = load_json("production_manifest")
        self.assertEqual(m["approved_models"], manifest["models"])
        self.assertEqual(m["approved_predictor_count"], len(manifest["approved_pooled_predictors"]))

    def test_authentication_is_derived_from_configuration_and_request(self):
        a = self._client().get(GOVERNANCE).get_json()["data"]["authentication"]
        self.assertIn("client-side", a["session_transport"].lower())
        self.assertEqual(a["identity_fields_stored"], ["version", "id", "display_name", "verified_at"])
        self.assertTrue(a["currently_authenticated"])
        self.assertIsInstance(a["current_session_expires_in_seconds"], int)
        self.assertEqual(a["reauthentication_minutes"], 30)
        self.assertEqual(a["roles_supported"], [])
        # CSRF is read from the real Flask-WTF configuration: this test app
        # disables it, so the response must report that truthfully.
        self.assertFalse(a["csrf_configured"])
        self.assertIn("csrf_scope", a)

    def test_governance_makes_no_live_dhis2_call(self):
        with patch("app.services.dhis2_client.DHIS2Client.get") as get_spy:
            with patch("app.services.dhis2_client.DHIS2Client.verify_login") as login_spy:
                self._client().get(GOVERNANCE)
                get_spy.assert_not_called()
                login_spy.assert_not_called()

    def test_forecast_governance_publishes_all_target_weeks(self):
        f = self._client().get(GOVERNANCE).get_json()["data"]["forecasts"]
        self.assertEqual(set(f["published_target_weeks"]), {"4", "8", "12"})
        self.assertEqual(f["published_target_weeks"]["4"]["target_label"], "2026W04")
        self.assertEqual(f["published_target_weeks"]["8"]["target_label"], "2026W08")
        self.assertEqual(f["published_target_weeks"]["12"]["target_label"], "2026W12")
        self.assertTrue(f["target_weeks_agree_with_database"])
        self.assertTrue(f["published"])
        self.assertEqual(f["publication_status"], "success")
        self.assertEqual(f["model_count"], 4)
        self.assertEqual(f["expected_forecast_rows"], 1752)
        self.assertEqual(f["actual_forecast_rows"], 1752)
        self.assertTrue(f["row_counts_agree"])
        self.assertNotEqual(f["schema_version"], "Not published")

    def test_threshold_prose_comes_from_the_service(self):
        from app.services.dashboard_data import resolve_params
        from app.services.explainability import build_threshold_method
        f = self._client().get(GOVERNANCE).get_json()["data"]["forecasts"]
        with self.app.app_context():
            expected = build_threshold_method(resolve_params(horizon=4))["data"]
        self.assertEqual(f["threshold_method"], expected["method"])
        self.assertEqual(f["surge_score_definition"], expected["surge_score_definition"])
        self.assertFalse(f["threshold_is_probability"])
        self.assertFalse(f["threshold_confirms_outbreak"])

    def test_pipeline_reports_every_required_stage(self):
        pipeline = self._client().get(GOVERNANCE).get_json()["data"]["pipeline"]
        titles = [s["stage"] for s in pipeline["stages"]]
        for required in ("District Reference and Boundaries", "DHIS2 Malaria Panel",
                         "Reporting Completeness", "Climate Ingestion", "Climate Aggregation",
                         "Climate Lags", "Modelling Dataset", "Model Validation",
                         "Production Model Fitting", "Forecast Generation",
                         "Model Performance Consolidation", "Dashboard Publication",
                         "Explainability Services", "Lifecycle Pages", "Reports and Exports"):
            self.assertIn(required, titles, required)
        for stage in pipeline["stages"]:
            self.assertIn(stage["status"],
                          ("Complete", "Partial", "Warning", "Unavailable", "Available"))

    def test_overview_includes_downstream_service_status(self):
        o = self._client().get(GOVERNANCE).get_json()["data"]["system_overview"]
        for key in ("explainability_status", "lifecycle_status", "reports_status"):
            self.assertIn(o[key], ("Available", "Unavailable"), key)

    def test_provenance_includes_latest_processed_weeks(self):
        latest = self._client().get(GOVERNANCE).get_json()["data"]["data_provenance"]["latest_weeks"]
        for key in ("latest_malaria_week", "latest_completeness_week",
                    "latest_processed_climate_week"):
            self.assertNotEqual(latest[key], "Not published", key)

    def test_model_and_validation_artifact_status_reported(self):
        m = self._client().get(GOVERNANCE).get_json()["data"]["models"]
        self.assertEqual(set(m["model_card_status"]), {"SARIMAX", "Random Forest", "XGBoost"})
        self.assertEqual(set(m["validation_artifact_status"]),
                         {"Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost"})
        self.assertNotEqual(m["dataset_signature"], "Not published")
        self.assertNotEqual(m["production_artifact_status"], "Not published")

    def test_reports_section_links_to_real_products(self):
        reports = self._client().get(GOVERNANCE).get_json()["data"]["reports"]
        self.assertEqual(reports["report_count"], 7)
        for item in reports["reports"]:
            self.assertEqual(item["url"], f"/reports/?report_type={item['report_type']}")
        district = next(x for x in reports["reports"] if x["report_type"] == "district_forecast")
        self.assertIsNone(district["print_url"])
        for item in reports["reports"]:
            if item["print_url"]:
                self.assertEqual(self._client().get(item["print_url"]).status_code, 200, item["report_type"])

    def test_no_last_tested_or_health_claim(self):
        blob = self._client().get(GOVERNANCE).get_data(as_text=True).lower()
        self.assertNotIn("last tested", blob)
        self.assertNotIn("healthy", blob)

    # ---- corrective pass: fail closed on broken evidence -----------------
    def _broken_load(self, mutate):
        from app.services import explainability as ex
        real = ex.load_json

        def broken(key):
            payload = real(key)
            if key in ("forecast_generation", "production_manifest"):
                payload = json.loads(json.dumps(payload))
                mutate(key, payload)
            return payload
        return broken

    def test_missing_target_weeks_fails_closed(self):
        def mutate(key, payload):
            if key == "forecast_generation":
                payload.pop("target_weeks", None)
        with patch("app.services.explainability.load_json",
                   side_effect=self._broken_load(mutate)):
            resp = self._client().get(GOVERNANCE)
        self.assertEqual(resp.status_code, 503)

    def test_missing_one_target_horizon_fails_closed(self):
        def mutate(key, payload):
            if key == "forecast_generation":
                payload["target_weeks"].pop("8", None)
        with patch("app.services.explainability.load_json",
                   side_effect=self._broken_load(mutate)):
            resp = self._client().get(GOVERNANCE)
        self.assertEqual(resp.status_code, 503)

    def test_malformed_manifest_fails_closed(self):
        def mutate(key, payload):
            if key == "production_manifest":
                payload.pop("models", None)
        with patch("app.services.explainability.load_json",
                   side_effect=self._broken_load(mutate)):
            resp = self._client().get(GOVERNANCE)
        self.assertEqual(resp.status_code, 503)

    def test_contradictory_target_weeks_fail_closed(self):
        def mutate(key, payload):
            if key == "forecast_generation":
                payload["target_weeks"]["4"]["target_epi_week"] = 41
        with patch("app.services.explainability.load_json",
                   side_effect=self._broken_load(mutate)):
            resp = self._client().get(GOVERNANCE)
        self.assertEqual(resp.status_code, 503)
        self.assertIn("contradict", resp.get_json()["reason"].lower())


if __name__ == "__main__":
    unittest.main()
