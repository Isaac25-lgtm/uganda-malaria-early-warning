"""Phase 29C lifecycle page + presentation-API tests.

Covers authenticated/unauthenticated routes, grouped navigation, honest deferred
Reports/Settings, absence of fake controls, page content contracts (missing-vs-zero,
completeness definition, climate sources/units/non-causal language, model coverage,
target-only forecast, risk warning), the small lifecycle APIs, SQL no-writes and a
deterministic forecast snapshot.
"""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app import create_app
from app.extensions import db

LIFECYCLE_ROUTES = [
    "/dhis2/", "/preprocessing/", "/climate/", "/modelling/",
    "/forecasts/", "/how-it-works/", "/reports/", "/settings/",
]
LIFECYCLE_APIS = ["/api/lifecycle/surveillance", "/api/lifecycle/data-quality"]


class LifecycleTemplateTests(unittest.TestCase):
    """Static template assertions; no database required."""

    def _read(self, rel):
        return Path(rel).read_text(encoding="utf-8")

    def test_settings_is_a_real_governance_page(self):
        # Phase 29E replaced the Settings deferral with a real, read-only
        # governance page driven by /api/system/governance.
        settings = self._read("app/templates/settings/index.html")
        self.assertIn("System Overview", settings)
        self.assertIn("System Integrity", settings)
        self.assertIn("Display Preferences", settings)
        self.assertIn("settings.js", settings)
        # It must not claim operational controls that do not exist.
        self.assertNotIn("System administration and governance controls are not included",
                         settings)

    def test_reports_page_is_no_longer_a_placeholder(self):
        # Phase 29D replaced the deferral with a functional Reports module.
        html = self._read("app/templates/reports/index.html")
        self.assertNotIn("planned for a later implementation phase", html)
        self.assertIn("Generate structured reports", html)

    def test_surveillance_distinguishes_missing_and_zero(self):
        html = self._read("app/templates/lifecycle/surveillance.html")
        self.assertIn("No malaria value was received for the district-week.", html)
        self.assertIn("A value of zero was explicitly reported.", html)
        self.assertIn("Missing reports are not converted to zero.", html)
        self.assertIn("does not measure malaria burden", html)

    def test_climate_states_sources_units_and_non_causal(self):
        html = self._read("app/templates/lifecycle/climate.html")
        self.assertIn("do not prove", html)  # non-causal language
        self.assertIn("unweighted mean across districts", html)
        # The page may only DISCLAIM a best-correlating lag, never assert one.
        self.assertNotIn("best correlating lag is", html.lower())
        self.assertIn("no best-correlating lag is claimed", html.lower())

    def test_forecasts_states_target_only_and_risk_warning(self):
        html = self._read("app/templates/lifecycle/forecasts_risk.html")
        self.assertIn("target week only", html)
        self.assertIn("not", html.lower())
        self.assertIn("not</strong> confirmation of an outbreak", html)

    def test_model_evaluation_marks_accuracy_not_used(self):
        html = self._read("app/templates/lifecycle/model_evaluation.html")
        self.assertIn("not used for model selection", html.lower())
        for tab in ("Count Forecasting", "Surge Detection", "Model Selection", "Validation Method", "Feature Importance"):
            self.assertIn(tab, html)

    def test_model_evaluation_prose_matches_current_selection(self):
        # Regression: after retraining, Random Forest is selected for BOTH count
        # and surge at every horizon. The page must not carry the obsolete claim
        # that XGBoost is the selected count model.
        html = self._read("app/templates/lifecycle/model_evaluation.html")
        self.assertNotIn("selected count model at 4 and 12 weeks", html)
        self.assertNotIn("the selected count model", html)  # XGBoost is not selected
        lowered = html.lower()
        # XGBoost must be described as a candidate, not a selected model.
        self.assertIn("not selected", lowered)
        # Random Forest is described as selected for both tasks at all horizons.
        self.assertIn("selected for both count prediction and operational surge warning", lowered)

        # Anchor the prose to the authoritative selection artifact: Random Forest
        # must genuinely be the selected count model at every horizon, so this
        # test fails if a future retraining changes selection without updating
        # the visible sentence.
        selection = json.loads(
            Path("data/quality/model_performance_selection_summary.json").read_text(encoding="utf-8"))
        for horizon in ("4", "8", "12"):
            count = (selection.get("selected_count_models") or {}).get(horizon, {})
            surge = (selection.get("selected_surge_models") or {}).get(horizon, {})
            self.assertEqual(count.get("model_name"), "Random Forest", f"count h{horizon}")
            self.assertEqual(surge.get("model_name"), "Random Forest", f"surge h{horizon}")

    def test_no_analytics_recalculated_in_js(self):
        # The lifecycle JS must not recompute winners/thresholds/risk.
        for rel in ("app/static/js/model_evaluation.js", "app/static/js/forecasts.js", "app/static/js/lifecycle.js"):
            js = self._read(rel)
            self.assertNotIn("classify", js.lower())
            self.assertNotIn("threshold =", js.lower())

    def test_nav_groups_defined(self):
        init = self._read("app/__init__.py")
        for group in ("OVERVIEW", "DATA PIPELINE", "ANALYTICS", "SYSTEM GUIDE", "OUTPUTS", "ADMINISTRATION"):
            self.assertIn(group, init)

    def test_forecast_summary_error_clears_stale_values(self):
        js = self._read("app/static/js/forecasts.js")
        # The summary failure path must clear values, not be an empty callback.
        self.assertIn("function summaryError", js)
        self.assertIn("summaryError", js)
        self.assertNotIn("}, function () {});", js)  # the old empty callback is gone
        # And a fresh load hides the district panel so stale interpretation cannot persist.
        self.assertIn('"fc-district-panel").hidden = true', js)

    def test_forecast_has_selected_district_panel(self):
        html = self._read("app/templates/lifecycle/forecasts_risk.html")
        self.assertIn('id="fc-district-panel"', html)
        for field in ("fc-di-count", "fc-di-surge", "fc-di-threshold", "fc-di-score",
                      "fc-di-risk", "fc-di-basis", "fc-di-completeness"):
            self.assertIn(field, html)
        self.assertIn("does not confirm an outbreak", html)

    def test_model_eval_has_family_view_and_independent_importance_guard(self):
        js = self._read("app/static/js/model_evaluation.js")
        self.assertIn("grouped=", js)
        self.assertIn('grouped ? "true" : "false"', js)
        self.assertIn("impGen", js)   # independent generation guard
        self.assertIn("impAbort", js)
        html = self._read("app/templates/lifecycle/model_evaluation.html")
        self.assertIn('id="me-view"', html)
        self.assertIn("Thresholds", html)

    def test_climate_has_district_selector_and_guard(self):
        html = self._read("app/templates/lifecycle/climate.html")
        self.assertIn('id="climate-district"', html)
        self.assertIn("AbortController", html)
        self.assertIn("gen !== S.gen", html)  # generation guard


class LifecycleRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL required.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.app.config["WTF_CSRF_ENABLED"] = False

    def _client(self, expired=False):
        from app.services.dhis2_auth import DHIS2SessionUser, DHIS2_IDENTITY_SESSION_KEY
        verified = int(datetime.now(timezone.utc).timestamp()) - (3600 if expired else 0)
        ident = DHIS2SessionUser(id="lifecycle-test", display_name="TESTER", verified_at=verified)
        client = self.app.test_client()
        with client.session_transaction() as s:
            s[DHIS2_IDENTITY_SESSION_KEY] = ident.to_session()
            s["_user_id"] = ident.get_id()
            s["_fresh"] = True
        return client

    def test_all_lifecycle_routes_require_auth(self):
        client = self.app.test_client()
        for route in LIFECYCLE_ROUTES:
            resp = client.get(route)
            self.assertEqual(resp.status_code, 302, route)
            self.assertIn("/auth/login", resp.headers["Location"])

    def test_all_lifecycle_routes_render_under_auth(self):
        client = self._client()
        for route in LIFECYCLE_ROUTES:
            resp = client.get(route)
            self.assertEqual(resp.status_code, 200, route)

    def test_navigation_groups_and_labels_render(self):
        html = self._client().get("/how-it-works/").get_data(as_text=True)
        for group in ("OVERVIEW", "DATA PIPELINE", "ANALYTICS", "SYSTEM GUIDE", "OUTPUTS", "ADMINISTRATION"):
            self.assertIn(group, html)
        for label in ("Surveillance Data", "Data Quality &amp; Preparation", "Climate Signals",
                      "Model Evaluation", "Forecasts &amp; Risk", "How It Works",
                      "Reports &amp; Exports", "Settings &amp; Governance"):
            self.assertIn(label, html)

    def test_active_route_marked(self):
        html = self._client().get("/forecasts/").get_data(as_text=True)
        self.assertIn('aria-current="page"', html)

    def test_lifecycle_apis_require_auth(self):
        client = self.app.test_client()
        for api in LIFECYCLE_APIS:
            resp = client.get(api)
            self.assertEqual(resp.status_code, 401, api)
            self.assertEqual(resp.get_json()["status"], "unauthenticated")

    def test_expired_session_rejected_on_lifecycle_api(self):
        resp = self._client(expired=True).get("/api/lifecycle/surveillance")
        self.assertEqual(resp.status_code, 401)

    def test_surveillance_coverage_from_approved_evidence(self):
        # The corrected coverage must come from approved project-state evidence,
        # NOT from counting NULL rows (missing weeks are absent rows).
        data = self._client().get("/api/lifecycle/surveillance").get_json()["data"]
        self.assertEqual(data["expected_district_weeks"], 45698)
        self.assertEqual(data["received_district_weeks"], 45451)
        self.assertEqual(data["missing_district_weeks"], 247)
        self.assertEqual(data["expected_district_weeks"] - data["received_district_weeks"],
                         data["missing_district_weeks"])
        self.assertTrue(data["missing_are_absent_rows"])
        self.assertTrue(data["missing_is_not_zero"])
        self.assertTrue(data["coverage_verified_against_database"])
        self.assertEqual(data["active_districts"], 146)

    def test_surveillance_fails_closed_on_inconsistent_evidence(self):
        from unittest.mock import patch
        bad = {
            "components": {"dhis2_malaria_panel": {"status": "complete", "row_counts": {
                "expected_district_weeks": 45698, "district_week_features": 45451,
                "missing_district_weeks": 999,  # inconsistent: 45698-45451 != 999
                "feature_explicit_zero_count": 0, "districts_represented": 146}}}}
        with patch("app.services.project_state.build_project_state_summary", return_value=bad):
            resp = self._client().get("/api/lifecycle/surveillance")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["status"], "unavailable")

    def test_surveillance_fails_closed_on_missing_evidence(self):
        from unittest.mock import patch
        with patch("app.services.project_state.build_project_state_summary",
                   return_value={"components": {"dhis2_malaria_panel": {"row_counts": {}}}}):
            resp = self._client().get("/api/lifecycle/surveillance")
        self.assertEqual(resp.status_code, 503)

    def test_data_quality_full_lag_contract_and_evidence_controls(self):
        data = self._client().get("/api/lifecycle/data-quality").get_json()["data"]
        self.assertEqual(len(data["stages"]), 8)
        # Complete-lag count must verify all nine approved lag columns.
        self.assertEqual(len(data["climate_lag_columns_verified"]), 9)
        self.assertIn("rows_with_complete_climate_lags", data)
        # Leakage controls are evidence-derived {value, source}, not literal True.
        for key in (
            "missing_not_zero_filled",
            "future_target_alignment",
            "training_only_preprocessing",
            "adjudicated_errors_excluded_from_outcomes",
            "bounded_predictor_only_interpolation",
        ):
            self.assertIn(key, data["leakage_controls"])
            self.assertIs(data["leakage_controls"][key]["value"], True)
            self.assertIn("source", data["leakage_controls"][key])
            self.assertNotEqual(data["leakage_controls"][key]["source"], "unavailable")
        self.assertEqual(
            data["leakage_controls"]["training_only_preprocessing"]["source"],
            "approved SARIMAX, Random Forest and XGBoost model cards",
        )

    def test_data_quality_fails_closed_when_leakage_evidence_missing(self):
        from unittest.mock import patch
        from app.services import explainability as ex
        with patch("app.services.explainability.load_json", side_effect=ex.ExplainabilityError("gone")):
            resp = self._client().get("/api/lifecycle/data-quality")
        self.assertEqual(resp.status_code, 503)

    def test_data_quality_fails_closed_on_malformed_preprocessing_card(self):
        from unittest.mock import patch
        from app.services import explainability as ex

        real_load = ex.load_json

        def malformed_card(key, *args, **kwargs):
            if key == "sarimax_card":
                return {"model_name": "SARIMAX", "status": "success"}
            return real_load(key, *args, **kwargs)

        with patch("app.services.explainability.load_json", side_effect=malformed_card):
            resp = self._client().get("/api/lifecycle/data-quality")
        self.assertEqual(resp.status_code, 503)

    def test_data_quality_fails_closed_on_malformed_cleaning_count(self):
        from unittest.mock import patch
        from app.services import explainability as ex

        real_load = ex.load_json

        def malformed_cleaning(key, *args, **kwargs):
            payload = real_load(key, *args, **kwargs)
            if key == "modelling_dataset":
                payload["case_cleaning"]["confirmed_erroneous_district_weeks"] = "not-a-count"
            return payload

        with patch("app.services.explainability.load_json", side_effect=malformed_cleaning):
            resp = self._client().get("/api/lifecycle/data-quality")
        self.assertEqual(resp.status_code, 503)

    def test_data_quality_template_binds_each_control_source(self):
        html = Path("app/templates/lifecycle/data_quality.html").read_text(encoding="utf-8")
        for key in (
            "missing_not_zero_filled",
            "future_target_alignment",
            "training_only_preprocessing",
            "adjudicated_errors_excluded_from_outcomes",
            "bounded_predictor_only_interpolation",
        ):
            self.assertIn(f'data-field="leakage_controls.{key}.source"', html)

    def test_lifecycle_apis_emit_no_sql_writes(self):
        from sqlalchemy import event
        client = self._client()
        with self.app.app_context():
            engine = db.engine
        statements = []
        def before(conn, cursor, statement, *a):
            statements.append(statement)
        event.listen(engine, "before_cursor_execute", before)
        try:
            for api in LIFECYCLE_APIS:
                client.get(api)
        finally:
            event.remove(engine, "before_cursor_execute", before)
        writes = [s for s in statements if s.lstrip()[:6].upper() in ("INSERT", "UPDATE", "DELETE")]
        self.assertEqual(writes, [])

    def test_no_lifecycle_response_leaks_paths_or_secrets(self):
        client = self._client()
        for api in LIFECYCLE_APIS:
            body = client.get(api).get_data(as_text=True)
            self.assertNotIn("postgresql://", body)
            self.assertNotIn(".joblib", body)
            self.assertNotIn("password", body.lower())

    def test_climate_api_supports_district_and_horizon(self):
        client = self._client()
        national = client.get("/api/dashboard/climate-summary?district_id=all&horizon=8").get_json()["data"]
        district = client.get("/api/dashboard/climate-summary?district_id=UGA_ABIM&horizon=8").get_json()["data"]
        self.assertEqual(national["aggregation"], "unweighted_district_mean")
        self.assertEqual(district["aggregation"], "district_value")
        for row in national["rows"]:
            self.assertEqual(row["highlighted_lag_weeks"], 8)

    def test_feature_importance_grouped_and_individual(self):
        client = self._client()
        feats = client.get("/api/explainability/feature-importance?model=random_forest&horizon=4&method=native&grouped=false").get_json()["data"]
        fams = client.get("/api/explainability/feature-importance?model=random_forest&horizon=4&method=native&grouped=true").get_json()["data"]
        self.assertIn("features", feats)
        self.assertIn("families", fams)
        self.assertNotIn("features", fams)

    def test_threshold_method_renders_bands(self):
        data = self._client().get("/api/explainability/threshold-method?horizon=4").get_json()["data"]
        levels = [b["risk_level"] for b in data["risk_bands"]]
        self.assertEqual(levels, ["Very High", "High", "Moderate", "Low"])

    def test_forecast_district_interpretation_values_from_backend(self):
        # The panel is populated from the risk-table row for the selected district.
        data = self._client().get("/api/dashboard/risk-table?district_id=UGA_ABIM&horizon=4").get_json()["data"]
        self.assertEqual(len(data["rows"]), 1)
        row = data["rows"][0]
        for key in ("predicted_cases", "surge_predicted_cases", "threshold_used",
                    "predicted_surge_score", "risk_level", "risk_basis",
                    "count_model_name", "surge_model_name", "reporting_completeness",
                    "issue_label", "target_label", "forecast_horizon_weeks"):
            self.assertIn(key, row)

    def test_forecast_snapshot_identical_across_lifecycle_requests(self):
        from app.models import ForecastOutput
        def snap():
            with self.app.app_context():
                rows = ForecastOutput.query.order_by(ForecastOutput.id).all()
                payload = [[r.id, r.district_id, r.model_name, r.horizon_weeks, str(r.predicted_cases),
                            str(r.threshold_used), r.risk_level, r.pipeline_run_id] for r in rows]
                return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), len(rows)
        before = snap()
        client = self._client()
        for route in LIFECYCLE_ROUTES + LIFECYCLE_APIS:
            client.get(route)
        after = snap()
        self.assertEqual(before, after)
        self.assertEqual(before[1], 1752)


if __name__ == "__main__":
    unittest.main()
