from __future__ import annotations

import json
import os
import time
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from app import create_app
from app.services import dashboard_data as dashboard_module
from app.services.dhis2_auth import DHIS2SessionUser
from app.services.dashboard_data import (
    HORIZONS,
    WEB_MAP_GEOJSON_PATH,
    WEB_MAP_MAX_BYTES,
    DashboardParameterError,
    DashboardParams,
    build_climate_section,
    build_model_performance_section,
    build_risk_table_section,
    build_summary_section,
    build_weekly_series_section,
    dashboard_payload,
    empty_dashboard_payload,
    map_risk_layer,
    parse_dashboard_params,
    resolve_params,
    risk_table_from_rows,
    selected_models,
)
from app.services.risk_classification import classify_forecast_risk

TEMPLATE_PATH = Path("app/templates/dashboard/index.html")
BASE_TEMPLATE_PATH = Path("app/templates/base.html")
SCRIPT_PATH = Path("app/static/js/dashboard.js")
CHART_LIB_PATH = Path("app/static/vendor/chartjs/chart.umd.js")
MAP_LIB_PATH = Path("app/static/vendor/leaflet/leaflet.js")
MOH_LOGO_PATH = Path("app/static/img/uganda-moh-logo.jfif")

PLACEHOLDER_STRINGS = (
    "Awaiting data",
    "Awaiting pipeline output",
    "Awaiting model output",
    "Map awaiting pipeline output",
    "Awaiting model training",
    "Awaiting analysis",
    "No processed data available yet",
)

API_ROUTES = (
    "/api/dashboard/summary",
    "/api/dashboard/weekly-series",
    "/api/dashboard/risk-table",
    "/api/dashboard/map",
    "/api/dashboard/climate-summary",
    "/api/dashboard/model-performance",
)


class DashboardDataPureTests(unittest.TestCase):
    """Pure logic; no database required."""

    def test_probability_risk_thresholds(self):
        very_high = classify_forecast_risk(
            predicted_surge_flag=True,
            surge_probability=Decimal("0.80"),
            predicted_cases=100,
            threshold_used=50,
        )
        self.assertEqual(very_high.risk_level, "Very High")
        self.assertEqual(very_high.surge_output_type, "trained_classifier")
        moderate = classify_forecast_risk(
            predicted_surge_flag=False,
            surge_probability=Decimal("0.45"),
            predicted_cases=50,
            threshold_used=100,
        )
        self.assertEqual(moderate.risk_level, "Moderate")

    def test_count_threshold_fallback_when_probability_unavailable(self):
        high = classify_forecast_risk(
            predicted_surge_flag=False,
            surge_probability=None,
            predicted_cases=120,
            threshold_used=100,
        )
        self.assertEqual(high.risk_level, "High")
        self.assertEqual(high.surge_output_type, "derived_count_threshold_benchmark")

    def test_no_data_district_is_retained_and_missing_not_zeroed(self):
        districts = [
            {"district_id": "UGA_A", "district_name": "A"},
            {"district_id": "UGA_B", "district_name": "B"},
        ]
        forecasts = {
            "UGA_A": {
                "predicted_cases": Decimal("90"),
                "threshold_used": Decimal("100"),
                "predicted_surge_flag": False,
                "surge_probability": None,
            }
        }
        observed = {"UGA_A": {"confirmed_malaria_cases": None}}
        rows = risk_table_from_rows(
            district_rows=districts,
            forecast_rows=forecasts,
            observed_rows=observed,
            previous_rows={},
            horizon=4,
        )
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["observed_cases"])
        self.assertTrue(rows[1]["no_data"])
        self.assertEqual(rows[1]["risk_level"], "No data")

    def test_surge_score_is_cases_minus_threshold_not_a_probability(self):
        rows = risk_table_from_rows(
            district_rows=[{"district_id": "UGA_A", "district_name": "A"}],
            forecast_rows={
                "UGA_A": {
                    "predicted_cases": Decimal("120.5"),
                    "threshold_used": Decimal("100"),
                    "predicted_surge_flag": False,
                    "surge_probability": None,
                }
            },
            observed_rows={},
            previous_rows={},
            horizon=4,
        )
        self.assertAlmostEqual(rows[0]["predicted_surge_score"], 20.5, places=4)
        self.assertIsNone(rows[0]["surge_probability"])

    def test_map_data_is_keyed_by_district_id(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"district_id": "UGA_A", "district_name": "A"},
                    "geometry": {"type": "Polygon", "coordinates": []},
                }
            ],
        }
        risk_rows = [
            {
                "district_id": "UGA_A",
                "district_name": "A",
                "risk_level": "High",
                "observed_cases": 10,
                "forecast_cases": 20,
                "surge_probability": 0.7,
                "no_data": False,
            },
            {
                "district_id": "UGA_B",
                "district_name": "B",
                "risk_level": "No data",
                "observed_cases": None,
                "forecast_cases": None,
                "surge_probability": None,
                "no_data": True,
            },
        ]
        layer = map_risk_layer(geojson, risk_rows)
        props = layer["features"][0]["properties"]
        self.assertEqual(props["district_id"], "UGA_A")
        self.assertEqual(props["risk_level"], "High")
        self.assertEqual(layer["missing_geometry_districts"], ["UGA_B"])

    def test_empty_payload_schema_is_stable(self):
        payload = empty_dashboard_payload("missing upstream data", horizon=8)
        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(payload["summary"]["forecast_horizon_weeks"], 8)
        self.assertIn("rows", payload["risk_table"])
        self.assertIn("features", payload["map"])
        self.assertIn("HMIS 033b", payload["summary"]["reporting_completeness_label"])
        self.assertFalse(payload["weekly_series"]["intervals_available"])
        self.assertEqual(payload["map"]["features"], [])


class DashboardFrontendAssetTests(unittest.TestCase):
    def test_official_moh_logo_is_used_for_institutional_identity(self):
        self.assertTrue(MOH_LOGO_PATH.exists())
        self.assertGreater(MOH_LOGO_PATH.stat().st_size, 0)
        self.assertIn("img/uganda-moh-logo.jfif", BASE_TEMPLATE_PATH.read_text(encoding="utf-8"))
        login = Path("app/templates/auth/login.html").read_text(encoding="utf-8")
        self.assertIn("img/uganda-moh-logo.jfif", login)
        self.assertIn("Republic of Uganda Ministry of Health", login)

    """The visible layer must consume the API and host its libraries locally."""

    @classmethod
    def setUpClass(cls):
        cls.template = TEMPLATE_PATH.read_text(encoding="utf-8")
        cls.base = BASE_TEMPLATE_PATH.read_text(encoding="utf-8")
        cls.script = SCRIPT_PATH.read_text(encoding="utf-8")
        cls.css = Path("app/static/css/app.css").read_text(encoding="utf-8")

    def test_placeholder_copy_is_removed(self):
        rendered = self.template + self.base
        for text in PLACEHOLDER_STRINGS:
            self.assertNotIn(text, rendered, f"placeholder still present: {text}")

    def test_script_is_included_and_calls_every_api_route(self):
        self.assertIn("js/dashboard.js", self.template)
        for route in API_ROUTES:
            self.assertIn(route, self.script, f"dashboard.js never calls {route}")

    def test_template_exposes_every_panel_target(self):
        rendered = self.template + self.base
        for target in (
            "filter-horizon",
            "filter-district",
            "filter-district-search",
            "filter-issue",
            "page-title",
            "kpi-cases-value",
            "kpi-completeness-value",
            "kpi-risk-value",
            "kpi-horizon-value",
            "weekly-chart",
            "risk-tbody",
            "risk-map",
            "climate-cards",
            "perf-count-tbody",
            "perf-surge-tbody",
            "perf-selection",
            "footer-updated",
        ):
            self.assertIn(f'id="{target}"', rendered, f"missing DOM target: {target}")

    def test_climate_and_performance_are_each_full_width_rows(self):
        # Phase 29A: each occupies its own full-width row, never side by side.
        self.assertNotIn("grid-secondary-2", self.template)
        climate_index = self.template.index('data-panel="climate"')
        performance_index = self.template.index('data-panel="performance"')
        between = self.template[climate_index:performance_index]
        # A full-width section boundary must separate them.
        self.assertIn("grid-single", between, "climate and performance are not separated into rows")

    def test_model_performance_uses_accessible_tabs(self):
        for needle in (
            'role="tablist"',
            'id="tab-count"',
            'id="tab-surge"',
            'id="tab-selection"',
            'id="tab-validation"',
            'role="tabpanel"',
        ):
            self.assertIn(needle, self.template, f"missing tab markup: {needle}")

    def test_no_accuracy_metric_is_shown(self):
        # The performance API returns no accuracy field; the UI must not invent one.
        self.assertNotIn(">Accuracy<", self.template)
        self.assertNotIn("row.accuracy", self.script)

    def test_overview_uses_prescribed_explanations(self):
        self.assertIn("early review of possible district-level malaria surges", self.script)
        self.assertIn("summarises recent malaria reporting", self.script)

    def test_forecast_zone_not_a_continuous_line(self):
        # Forecast dataset shows a point (showLine false / single target), never a joined line.
        self.assertIn("Intermediate weekly forecasts are not", self.template)

    def test_weekly_chart_uses_figure_specific_interpretation(self):
        self.assertIn("function weeklyInterpretation(data)", self.script)
        self.assertIn("target.predicted_cases", self.script)
        self.assertIn("target.surge_threshold", self.script)
        self.assertIn("cases above the threshold", self.script)
        self.assertIn("cases below the threshold", self.script)
        self.assertIn("data.district_name", self.script)
        self.assertIn("state.range", self.script)
        self.assertIn("point forecast rather than a guaranteed exact count", self.script)
        self.assertNotIn("note.push(data.threshold_note)", self.script)

    # ---- Phase 29A corrective pass ------------------------------------
    def test_latest_request_wins_uses_generation_and_abort(self):
        # Defect 1: monotonic generation + AbortController + aborted handling.
        self.assertIn("state.generation", self.script)
        self.assertIn("AbortController", self.script)
        self.assertIn('kind === "aborted"', self.script)
        # A superseded response must be dropped by comparing generations.
        self.assertIn("generation !== state.generation", self.script)

    def test_aborted_fetch_is_not_treated_as_api_error(self):
        # The guarded fail path returns early on an aborted kind (no error state).
        self.assertRegex(self.script, r'if \(err && err\.kind === "aborted"\) return;')

    def test_risk_table_has_real_pagination_controls(self):
        # Defect 2: Previous/Next buttons + accessible status, 25-row default.
        for needle in ('id="risk-pager"', 'id="risk-prev"', 'id="risk-next"', 'id="risk-page-status"'):
            self.assertIn(needle, self.template, needle)
        self.assertIn("pageSize: 25", self.script)
        # Sort/search happen on the full set before the page slice.
        self.assertIn("rows.slice(start, start + state.pageSize)", self.script)
        # Filter/search/sort reset to page one.
        self.assertGreaterEqual(self.script.count("state.page = 1"), 3)

    def test_risk_table_generates_a_district_specific_interpretation_for_each_visible_row(self):
        for phrase in (
            "Risk-assessment cases",
            "Surge threshold",
            "Surge score",
            "Completeness",
            "how many weeks ahead the forecast refers to",
        ):
            self.assertIn(phrase, self.template, phrase)
        self.assertIn('id="risk-interpretations-title"', self.template)
        self.assertIn("function riskInterpretationHtml(row)", self.script)
        self.assertIn("rows.map(riskInterpretationHtml)", self.script)
        self.assertIn("The dashboard forecasts", self.script)
        self.assertIn("seasonal surge threshold", self.script)
        self.assertIn("Completeness is", self.script)
        self.assertIn("A High or Very High result supports early review", self.script)
        self.assertNotIn("How to read each row:", self.script)
        self.assertNotIn("Agago's usual high-season", self.script)
        self.assertNotIn("No trained surge probability exists", self.script)

    def test_sortable_headers_are_real_buttons_for_keyboard(self):
        # Defect 3: <button> inside each sortable th; aria-sort stays on th.
        self.assertIn('class="th-sort"', self.template)
        self.assertIn('aria-sort="none"', self.template)
        self.assertIn(".th-sort", self.script)
        # Sorting state uses classes (asc/desc), not colour alone.
        self.assertIn("sorted-asc", self.script)
        self.assertIn("sorted-desc", self.script)

    def test_chart_direct_labels_plugin_present(self):
        # Defect 4: custom direct-label plugin, no external datalabels package.
        self.assertIn("directLabels", self.script)
        self.assertIn("Forecast ", self.script)
        self.assertIn("Threshold ", self.script)
        self.assertIn("targetLabel", self.script)
        self.assertNotIn("chartjs-plugin-datalabels", self.script)
        # A label is hidden when its source value is missing (lastPoint returns null).
        self.assertIn("if (!point) return null", self.script)

    def test_map_labels_are_district_view_only(self):
        # Defect 5: permanent labels drawn on selection, cleared in national view.
        self.assertIn("drawMapLabels", self.script)
        self.assertIn("clearMapLabels", self.script)
        self.assertIn("map-label-selected", self.script)
        # Collision-safe placement + national view clears labels.
        self.assertIn("overlaps", self.script)
        # National branch (no selected district) clears labels.
        self.assertRegex(self.script, r"clearMapLabels\(\);\s*\n\s*if \(state\.nationalBounds\)")

    def test_map_labels_styled_and_hover_tooltip_preserved(self):
        self.assertIn(".map-label", self.css)
        self.assertIn("map-tooltip", self.script)  # hover tooltip retained

    def test_chart_and_map_libraries_are_hosted_locally(self):
        self.assertTrue(CHART_LIB_PATH.exists(), "Chart.js is not vendored locally")
        self.assertTrue(MAP_LIB_PATH.exists(), "Leaflet is not vendored locally")
        self.assertIn("vendor/chartjs/chart.umd.js", self.template)
        self.assertIn("vendor/leaflet/leaflet.js", self.template)

    def test_no_external_cdn_or_tile_dependency_at_runtime(self):
        for text in (self.template, self.base, self.script):
            self.assertNotIn("https://cdn.", text)
            self.assertNotIn("https://unpkg.com", text)
            self.assertNotIn("tile.openstreetmap", text)

    def test_research_use_note_is_visible(self):
        for phrase in (
            "does not confirm an outbreak",
            "do not replace epidemiological judgement",
            "Confirmed reporting errors",
            "does not establish causation",
            "Accuracy alone is not used",
        ):
            self.assertIn(phrase, self.template, f"research-use note missing: {phrase}")

    def test_frontend_states_current_surge_selection_rule(self):
        self.assertIn("selected using F2", self.script)
        self.assertIn("sensitivity and specificity", self.script)
        self.assertNotIn("selected using the highest F1", self.script)


class DashboardWebMapArtifactTests(unittest.TestCase):
    """The derived boundary must be browser-sized and joinable."""

    def test_web_map_is_present_valid_and_within_budget(self):
        self.assertTrue(WEB_MAP_GEOJSON_PATH.exists(), "derived web-map GeoJSON is missing")
        size = WEB_MAP_GEOJSON_PATH.stat().st_size
        self.assertLessEqual(size, WEB_MAP_MAX_BYTES, f"web map is {size} bytes")

        payload = json.loads(WEB_MAP_GEOJSON_PATH.read_text(encoding="utf-8"))
        features = payload["features"]
        self.assertEqual(len(features), 146)

        ids = [feature["properties"]["district_id"] for feature in features]
        self.assertEqual(len(set(ids)), 146)
        self.assertNotIn(None, ids)

        types = {feature["geometry"]["type"] for feature in features}
        self.assertTrue(types <= {"Polygon", "MultiPolygon"}, f"unexpected geometry: {types}")

    def test_approved_boundary_is_not_replaced_by_the_derived_artifact(self):
        self.assertTrue(dashboard_module.PREPARED_GEOJSON_PATH.exists())
        self.assertNotEqual(dashboard_module.PREPARED_GEOJSON_PATH, WEB_MAP_GEOJSON_PATH)


class DashboardDatabaseTestCase(unittest.TestCase):
    """Shared app context for tests that read the research database."""

    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL is required for database-backed tests.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.ctx = cls.app.app_context()
        cls.ctx.push()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.pop()


class DashboardParameterTests(DashboardDatabaseTestCase):
    def test_defaults_to_horizon_4_and_latest_issue(self):
        params = parse_dashboard_params({})
        self.assertEqual(params.horizon, 4)
        self.assertIsNone(params.district_id)
        self.assertIsNotNone(params.issue_year)
        self.assertIsNotNone(params.issue_week)

    def test_every_approved_horizon_is_accepted(self):
        for horizon in HORIZONS:
            params = parse_dashboard_params({"horizon": str(horizon)})
            self.assertEqual(params.horizon, horizon)

    def test_invalid_horizon_is_rejected(self):
        for value in ("6", "0", "-4", "four", "4.5"):
            with self.assertRaises(DashboardParameterError):
                parse_dashboard_params({"horizon": value})

    def test_district_all_means_national(self):
        self.assertIsNone(parse_dashboard_params({"district_id": "all"}).district_id)
        self.assertIsNone(parse_dashboard_params({"district_id": ""}).district_id)

    def test_unknown_district_is_rejected(self):
        with self.assertRaises(DashboardParameterError):
            parse_dashboard_params({"district_id": "UGA_NOWHERE"})

    def test_issue_year_and_week_must_be_supplied_together(self):
        with self.assertRaises(DashboardParameterError):
            parse_dashboard_params({"issue_year": "2025"})
        with self.assertRaises(DashboardParameterError):
            parse_dashboard_params({"issue_week": "52"})

    def test_unknown_issue_is_rejected_rather_than_invented(self):
        with self.assertRaises(DashboardParameterError):
            parse_dashboard_params({"issue_year": "2024", "issue_week": "10"})


class DashboardSectionTests(DashboardDatabaseTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.params = resolve_params()
        if cls.params.issue_year is None:
            raise unittest.SkipTest("No production forecast issue available.")

    def test_count_and_surge_models_follow_published_selection(self):
        models = selected_models(4)
        self.assertEqual(models["count"]["model_name"], "Random Forest")
        self.assertEqual(models["surge"]["model_name"], "Random Forest")
        self.assertTrue(models["same_model"])
        self.assertEqual(selected_models(8)["count"]["model_name"], "Random Forest")
        self.assertEqual(selected_models(12)["count"]["model_name"], "Random Forest")
        for horizon in HORIZONS:
            self.assertEqual(selected_models(horizon)["surge"]["model_name"], "Random Forest")

    def test_risk_rows_keep_explicit_count_and_surge_model_roles(self):
        section = build_risk_table_section(resolve_params(horizon=4))
        rows = section["rows"]
        self.assertEqual(len(rows), 146)
        for row in rows:
            self.assertEqual(row["count_model_name"], "Random Forest")
            self.assertEqual(row["surge_model_name"], "Random Forest")
            self.assertEqual(row["predicted_cases"], row["surge_predicted_cases"])

    def test_surge_score_equals_surge_cases_minus_threshold(self):
        rows = build_risk_table_section(self.params)["rows"]
        checked = 0
        for row in rows:
            if row["surge_predicted_cases"] is None or row["threshold_used"] is None:
                continue
            expected = float(row["surge_predicted_cases"]) - float(row["threshold_used"])
            self.assertAlmostEqual(float(row["predicted_surge_score"]), expected, places=3)
            checked += 1
        self.assertGreater(checked, 0)

    def test_risk_rows_never_present_a_probability_that_does_not_exist(self):
        section = build_risk_table_section(self.params)
        if not section["surge_probability_available"]:
            for row in section["rows"]:
                self.assertIsNone(row["surge_probability"])
            self.assertIn("count-threshold", section["risk_assessment_label"].lower())

    def test_weekly_series_appends_future_forecast_rows(self):
        section = build_weekly_series_section(self.params)
        observed = [row for row in section["rows"] if row["period"] == "observed"]
        forecast = [row for row in section["rows"] if row["period"] == "forecast"]
        self.assertGreater(len(observed), 0)
        self.assertGreater(len(forecast), 0, "no future forecast row was appended")
        for row in forecast:
            self.assertIsNotNone(row["predicted_cases"])
            self.assertIsNone(row["actual_cases"])
        # Forecast weeks must be strictly later than every observed week.
        last_observed = max(row["week_start_date"] for row in observed)
        for row in forecast:
            self.assertGreater(row["week_start_date"], last_observed)

    def test_weekly_observed_data_stops_at_the_issue_week(self):
        section = build_weekly_series_section(self.params)
        observed = [row for row in section["rows"] if row["period"] == "observed"]
        issue_start = section["issue_week_start_date"]
        for row in observed:
            self.assertLessEqual(row["week_start_date"], issue_start)

    def test_weekly_series_uses_the_selected_count_model(self):
        section = build_weekly_series_section(resolve_params(horizon=4))
        self.assertEqual(section["count_model_name"], "Random Forest")

    def test_intervals_absent_when_the_count_model_publishes_none(self):
        section = build_weekly_series_section(resolve_params(horizon=4))
        if not section["intervals_available"]:
            for row in section["rows"]:
                self.assertIsNone(row["prediction_interval_lower"])
                self.assertIsNone(row["prediction_interval_upper"])
            self.assertIn("Random Forest", section["intervals_note"])

    def test_threshold_appears_only_at_forecast_targets(self):
        section = build_weekly_series_section(self.params)
        self.assertEqual(section["threshold_model_name"], section["surge_model_name"])
        for row in section["rows"]:
            if row["period"] == "observed":
                self.assertIsNone(row["surge_threshold"])

    def test_weekly_chart_notes_explain_observed_forecast_and_threshold(self):
        national = build_weekly_series_section(resolve_params(horizon=4))
        self.assertIn("Green shows confirmed cases", national["threshold_note"])
        self.assertIn("blue diamond", national["threshold_note"])
        self.assertIn("red X", national["threshold_note"])
        self.assertIn("does not confirm an outbreak", national["threshold_note"])
        self.assertIn("sums across districts", national["national_threshold_note"])
        self.assertIn("does not provide a prediction range", national["intervals_note"])
        self.assertEqual(national["forecast_horizon_weeks"], 4)
        self.assertEqual(national["target_label"], "2026W04")
        self.assertIsNone(national["district_name"])

        district = build_weekly_series_section(resolve_params(horizon=4, district_id="UGA_AGAGO"))
        self.assertEqual(district["district_name"].upper(), "AGAGO")
        self.assertEqual(district["target_label"], "2026W04")

    def test_district_filter_changes_summary_weekly_risk_and_climate(self):
        national = resolve_params(horizon=4)
        district = resolve_params(horizon=4, district_id="UGA_ABIM")

        national_summary = build_summary_section(national)
        district_summary = build_summary_section(district)
        self.assertNotEqual(
            national_summary["confirmed_malaria_cases"],
            district_summary["confirmed_malaria_cases"],
        )
        self.assertEqual(district_summary["scope"], "district")
        self.assertEqual(district_summary["districts_in_scope"], 1)

        self.assertEqual(len(build_risk_table_section(district)["rows"]), 1)

        national_weekly = build_weekly_series_section(national)
        district_weekly = build_weekly_series_section(district)
        self.assertNotEqual(
            national_weekly["rows"][-1]["predicted_cases"],
            district_weekly["rows"][-1]["predicted_cases"],
        )

        national_climate = build_climate_section(national)
        district_climate = build_climate_section(district)
        self.assertEqual(national_climate["aggregation"], "unweighted_district_mean")
        self.assertEqual(district_climate["aggregation"], "district_value")

    def test_map_ignores_the_district_filter_but_marks_the_selection(self):
        section = dashboard_module.build_map_section(
            resolve_params(horizon=4, district_id="UGA_ABIM")
        )
        self.assertEqual(len(section["features"]), 146, "map must keep national context")
        self.assertEqual(section["selected_district_id"], "UGA_ABIM")
        self.assertEqual(section["join_key"], "district_id")

    def test_map_payload_stays_within_the_browser_budget(self):
        section = dashboard_module.build_map_section(self.params)
        size = len(json.dumps(section).encode("utf-8"))
        self.assertLessEqual(size, WEB_MAP_MAX_BYTES, f"map payload is {size} bytes")
        self.assertEqual(section["missing_geometry_districts"], [])

    def test_non_map_sections_never_load_the_geojson(self):
        original = dashboard_module._load_web_map_geojson

        def explode():
            raise AssertionError("non-map section loaded the GeoJSON")

        dashboard_module._load_web_map_geojson = explode
        try:
            build_summary_section(self.params)
            build_weekly_series_section(self.params)
            build_risk_table_section(self.params)
            build_climate_section(self.params)
            build_model_performance_section(self.params)
        finally:
            dashboard_module._load_web_map_geojson = original

    def test_climate_reports_units_and_never_claims_a_best_correlating_lag(self):
        section = build_climate_section(self.params)
        units = {row["variable"]: row["unit"] for row in section["rows"]}
        self.assertEqual(units["rainfall"], "mm")
        self.assertEqual(units["temperature"], "°C")
        self.assertEqual(units["humidity"], "%")
        self.assertFalse(section["correlation_available"])
        for row in section["rows"]:
            self.assertIn("current", row["measures"])
            for horizon in HORIZONS:
                self.assertIn(f"lag_{horizon}", row["measures"])
        self.assertNotIn("best", section["correlation_note"].split("no best-correlating")[0].lower())

    def test_climate_highlighted_lag_follows_the_selected_horizon(self):
        for horizon in HORIZONS:
            section = build_climate_section(resolve_params(horizon=horizon))
            for row in section["rows"]:
                self.assertEqual(row["highlighted_lag_weeks"], horizon)
                self.assertEqual(row["highlighted_lag_key"], f"lag_{horizon}")

    def test_model_performance_covers_all_four_models_and_marks_selection(self):
        section = build_model_performance_section(resolve_params(horizon=4))
        h4 = [row for row in section["rows"] if row["horizon"] == 4]
        self.assertEqual(
            {row["model_name"] for row in h4},
            {"Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost"},
        )
        for row in h4:
            for metric in ("mae", "rmse", "mape", "wape", "r_squared", "sensitivity", "specificity", "precision", "f1", "auc"):
                self.assertIn(metric, row)
        selected_count = [row["model_name"] for row in h4 if row["selected_for_count"]]
        selected_surge = [row["model_name"] for row in h4 if row["selected_for_surge"]]
        self.assertEqual(selected_count, ["Random Forest"])
        self.assertEqual(selected_surge, ["Random Forest"])
        self.assertIn("F2", section["accuracy_note"])

    def test_model_performance_is_national_and_ignores_the_district_filter(self):
        national = build_model_performance_section(resolve_params(horizon=4))
        district = build_model_performance_section(
            resolve_params(horizon=4, district_id="UGA_ABIM")
        )
        self.assertEqual(national["rows"], district["rows"])
        self.assertIn("national", district["scope_note"].lower())

    def test_payload_never_advertises_illustrative_figures(self):
        payload = dashboard_payload(allow_unavailable=True)
        self.assertNotIn("illustrative", json.dumps(payload).lower())


class DashboardRouteTests(unittest.TestCase):
    """Route tests deliberately hold no app context.

    Flask reuses an already-pushed app context for test-client requests, so a
    long-lived one lets flask_login's ``g._login_user`` leak from a logged-in
    test into a later request and make an anonymous client look authenticated.
    Each request here pushes its own context, so anonymous really is anonymous.
    """

    PASSWORD = "dashboard-test-password"

    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL is required for route tests.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.username = "__dashboard_api_test__"

    def _logged_in_client(self):
        client = self.app.test_client()
        identity = DHIS2SessionUser(
            id="dhis2-test-user-id",
            display_name="Dashboard Test",
            verified_at=int(time.time()),
        )
        with patch(
            "app.blueprints.auth.routes.authenticate_dhis2_user",
            return_value=identity,
        ):
            response = client.post(
                "/auth/login",
                data={"username": self.username, "password": self.PASSWORD},
                follow_redirects=False,
            )
        self.assertIn(response.status_code, {302, 303})
        return client

    def test_dashboard_api_requires_auth_and_answers_fetch_with_json(self):
        client = self.app.test_client()
        for route in API_ROUTES:
            response = client.get(route)
            self.assertEqual(response.status_code, 401, route)
            self.assertEqual(response.mimetype, "application/json", route)
            self.assertEqual(response.get_json()["status"], "unauthenticated")

    def test_dashboard_page_still_redirects_unauthenticated_users(self):
        response = self.app.test_client().get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login", response.headers["Location"])

    def test_every_api_returns_available_under_auth(self):
        client = self._logged_in_client()
        for route in API_ROUTES:
            response = client.get(route)
            self.assertEqual(response.status_code, 200, f"{route} -> {response.status_code}")
            payload = response.get_json()
            self.assertEqual(payload["status"], "available", route)
            self.assertIsNotNone(payload["data"], route)
            self.assertEqual(payload["empty_states"], [], route)

    def test_api_accepts_documented_parameters(self):
        client = self._logged_in_client()
        response = client.get(
            "/api/dashboard/summary?horizon=12&district_id=UGA_ABIM&issue_year=2025&issue_week=52"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["params"]["horizon"], 12)
        self.assertEqual(payload["params"]["district_id"], "UGA_ABIM")
        self.assertEqual(payload["params"]["issue_label"], "2025W52")
        self.assertEqual(payload["data"]["forecast_horizon_weeks"], 12)

    def test_api_rejects_invalid_parameters_with_json_400(self):
        client = self._logged_in_client()
        for query in (
            "horizon=6",
            "horizon=abc",
            "district_id=UGA_NOWHERE",
            "issue_year=2024&issue_week=10",
            "issue_year=2025",
        ):
            response = client.get(f"/api/dashboard/summary?{query}")
            self.assertEqual(response.status_code, 400, query)
            self.assertEqual(response.mimetype, "application/json", query)
            payload = response.get_json()
            self.assertEqual(payload["status"], "invalid_request", query)
            self.assertTrue(payload["reason"], query)

    def test_dashboard_route_loads_under_auth_without_placeholder_copy(self):
        client = self._logged_in_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        for text in PLACEHOLDER_STRINGS:
            self.assertNotIn(text, html, f"rendered page still shows: {text}")

    def test_rendered_page_offers_only_real_filter_options(self):
        client = self._logged_in_client()
        html = client.get("/").get_data(as_text=True)
        self.assertIn("2025W52", html)
        self.assertIn('id="filter-horizon"', html)
        self.assertIn('id="filter-district"', html)
        self.assertIn('id="filter-district-search"', html)
        self.assertIn('role="combobox"', html)
        self.assertIn("All Districts (Uganda)", html)
        self.assertEqual(html.count('class="district-option"'), 147)
        # Exactly one production issue exists; none may be invented.
        self.assertEqual(html.count('value="2025-52"'), 1)
        self.assertNotIn("2025-51", html)
        self.assertIn('id="filter-issue" disabled', html)

    def test_api_returns_stable_contract_keys(self):
        client = self._logged_in_client()
        payload = client.get("/api/dashboard/summary").get_json()
        for key in ("status", "reason", "data", "params", "empty_states"):
            self.assertIn(key, payload)
        self.assertIn("reporting_completeness_label", payload["data"])
        self.assertIn("HMIS 033b", payload["data"]["reporting_completeness_label"])
        self.assertNotIn("illustrative", str(payload).lower())

    def test_numeric_values_are_json_numbers_not_strings(self):
        client = self._logged_in_client()
        data = client.get("/api/dashboard/risk-table").get_json()["data"]
        row = data["rows"][0]
        for key in ("predicted_cases", "threshold_used", "predicted_surge_score"):
            if row[key] is not None:
                self.assertIsInstance(row[key], (int, float), f"{key} is {type(row[key])}")

    def test_debug_map_http_response_stays_within_browser_budget(self):
        original_debug = self.app.debug
        self.app.debug = True
        try:
            response = self._logged_in_client().get("/api/dashboard/map")
        finally:
            self.app.debug = original_debug
        self.assertEqual(response.status_code, 200)
        self.assertLessEqual(
            len(response.data),
            WEB_MAP_MAX_BYTES,
            f"debug HTTP map response is {len(response.data)} bytes",
        )


if __name__ == "__main__":
    unittest.main()
