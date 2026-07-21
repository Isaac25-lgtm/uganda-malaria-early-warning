"""Phase 29D Reports & Exports tests.

Covers authentication, report/format allowlists, filter validation, per-report
content contracts, file security (safe filenames, traversal, temp cleanup, no
secrets), PDF/Excel/CSV integrity, formula-injection protection, read-only
database behaviour and Phase 29A-29C regression.
"""
from __future__ import annotations

import csv as csvmod
import hashlib
import io
import json
import os
import re
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.services import report_data_service as rds
from app.services.report_export_service import safe_filename, sanitize_cell

ALL_REPORTS = tuple(rds.REPORTS.keys())


class ReportPureTests(unittest.TestCase):
    """No app context required."""

    def test_formula_injection_guarded_for_strings_only(self):
        for probe in ("=cmd|' /C calc'!A0", "+1+1", "-2+3", "@SUM(A1)"):
            guarded = sanitize_cell(probe)
            self.assertTrue(guarded.startswith("'"), probe)
        # Genuine numerics/bools/None must NOT become text.
        self.assertEqual(sanitize_cell(1234), 1234)
        self.assertEqual(sanitize_cell(12.5), 12.5)
        self.assertIs(sanitize_cell(True), True)
        self.assertIsNone(sanitize_cell(None))
        self.assertEqual(sanitize_cell("normal"), "normal")

    def test_safe_filename_rejects_separators_and_traversal(self):
        name = safe_filename("../../etc/passwd", "a/b", "c\\d", extension="pdf")
        self.assertNotIn("/", name)
        self.assertNotIn("\\", name)
        self.assertNotIn("..", name)
        self.assertTrue(name.endswith(".pdf"))

    def test_registry_exposes_only_implemented_formats(self):
        for item in rds.report_registry():
            self.assertTrue(item["formats"])
            for fmt in item["formats"]:
                self.assertIn(fmt, {"pdf", "xlsx", "csv", "html"})
        # PowerPoint must NOT be offered.
        blob = json.dumps(rds.report_registry()).lower()
        self.assertNotIn("pptx", blob)
        self.assertNotIn("powerpoint", blob)


class ReportRouteTests(unittest.TestCase):
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
        ident = DHIS2SessionUser(id="report-test", display_name="TESTER", verified_at=verified)
        client = self.app.test_client()
        with client.session_transaction() as s:
            s[DHIS2_IDENTITY_SESSION_KEY] = ident.to_session()
            s["_user_id"] = ident.get_id()
            s["_fresh"] = True
        return client

    def _gen(self, client, report_type, fmt, extra=""):
        return client.get(f"/reports/generate?report_type={report_type}&format={fmt}&horizon=4{extra}")

    # ---- authentication ------------------------------------------------
    def test_reports_page_requires_auth(self):
        resp = self.app.test_client().get("/reports/")
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/auth/login", resp.headers["Location"])

    def test_export_endpoints_require_auth(self):
        client = self.app.test_client()
        self.assertEqual(client.get("/reports/generate?report_type=methods&format=pdf").status_code, 401)
        self.assertEqual(client.get("/reports/print?report_type=methods").status_code, 302)

    def test_expired_session_rejected(self):
        resp = self._gen(self._client(expired=True), "methods", "pdf")
        self.assertEqual(resp.status_code, 401)

    def test_no_live_dhis2_call_during_generation(self):
        with patch("app.services.dhis2_client.DHIS2Client.get") as spy:
            self._gen(self._client(), "national_summary", "pdf")
            spy.assert_not_called()

    # ---- selection + format allowlist ----------------------------------
    def test_every_report_and_declared_format_generates(self):
        client = self._client()
        for report_type, spec in rds.REPORTS.items():
            extra = "&district_id=UGA_ABIM" if "district_required" in spec["filters"] else ""
            for fmt in spec["formats"]:
                if fmt == "html":
                    resp = client.get(f"/reports/print?report_type={report_type}&horizon=4{extra}")
                else:
                    resp = self._gen(client, report_type, fmt, extra)
                self.assertEqual(resp.status_code, 200, f"{report_type}/{fmt}")
                self.assertGreater(len(resp.data), 500, f"{report_type}/{fmt} produced an empty file")

    def test_unsupported_format_rejected(self):
        client = self._client()
        # methods has no Excel/CSV; risk list has no... etc.
        self.assertEqual(self._gen(client, "methods", "xlsx").status_code, 400)
        self.assertEqual(self._gen(client, "methods", "csv").status_code, 400)
        self.assertEqual(self._gen(client, "national_summary", "csv").status_code, 400)
        self.assertEqual(self._gen(client, "climate_signals", "pptx").status_code, 400)

    def test_invalid_report_type_rejected(self):
        self.assertEqual(self._gen(self._client(), "not_a_report", "pdf").status_code, 400)

    # ---- filters -------------------------------------------------------
    def test_all_horizons_accepted_and_invalid_rejected(self):
        client = self._client()
        for horizon in (4, 8, 12):
            resp = client.get(f"/reports/generate?report_type=national_summary&format=pdf&horizon={horizon}")
            self.assertEqual(resp.status_code, 200, horizon)
        for bad in (5, 0, -4, 99):
            resp = client.get(f"/reports/generate?report_type=national_summary&format=pdf&horizon={bad}")
            self.assertEqual(resp.status_code, 400, bad)

    def test_district_validation(self):
        client = self._client()
        self.assertEqual(self._gen(client, "district_forecast", "pdf", "&district_id=UGA_ABIM").status_code, 200)
        self.assertEqual(self._gen(client, "district_forecast", "pdf", "&district_id=UGA_NOWHERE").status_code, 400)
        # District is REQUIRED for this report.
        self.assertEqual(self._gen(client, "district_forecast", "pdf").status_code, 400)

    def test_fake_historical_issue_rejected(self):
        resp = self._client().get(
            "/reports/generate?report_type=national_summary&format=pdf&issue_year=2024&issue_week=10")
        self.assertEqual(resp.status_code, 400)

    def test_invalid_risk_level_rejected(self):
        resp = self._gen(self._client(), "district_risk_list", "csv", "&risk_level=Catastrophic")
        self.assertEqual(resp.status_code, 400)

    # ---- file security -------------------------------------------------
    def test_headers_and_safe_filenames(self):
        client = self._client()
        resp = self._gen(client, "district_risk_list", "csv")
        disposition = resp.headers["Content-Disposition"]
        self.assertIn("attachment;", disposition)
        self.assertNotIn("..", disposition)
        self.assertNotIn("/", disposition.split("filename=")[1])
        self.assertEqual(resp.headers.get("Cache-Control"), "no-store")
        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")

    def test_caller_cannot_supply_filename_or_path(self):
        client = self._client()
        resp = client.get("/reports/generate?report_type=methods&format=pdf"
                          "&filename=../../evil.pdf&path=/etc/passwd&output=/tmp/x")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("malaria_methods", resp.headers["Content-Disposition"])
        self.assertNotIn("evil", resp.headers["Content-Disposition"])

    def test_no_report_written_into_static_directory(self):
        before = set(Path("app/static").rglob("*"))
        self._gen(self._client(), "national_summary", "pdf")
        self._gen(self._client(), "district_risk_list", "xlsx")
        self.assertEqual(before, set(Path("app/static").rglob("*")))

    def test_temporary_files_are_cleaned_up(self):
        import tempfile
        tmp = Path(tempfile.gettempdir())
        before = set(tmp.glob("mews_report_*"))
        client = self._client()
        for _ in range(3):
            resp = self._gen(client, "national_summary", "pdf")
            resp.close()  # triggers call_on_close cleanup
        after = set(tmp.glob("mews_report_*"))
        self.assertEqual(after - before, set(), "temporary report files were left behind")

    def test_generated_files_contain_no_secrets_or_paths(self):
        client = self._client()
        for report_type, spec in rds.REPORTS.items():
            extra = "&district_id=UGA_ABIM" if "district_required" in spec["filters"] else ""
            for fmt in spec["formats"]:
                if fmt == "html":
                    continue
                blob = self._gen(client, report_type, fmt, extra).data
                for token in (b"postgresql://", b"SECRET_KEY", b"DHIS2_PASSWORD", b"Traceback"):
                    self.assertNotIn(token, blob, f"{report_type}/{fmt} leaked {token!r}")

    # ---- PDF -----------------------------------------------------------
    def test_pdf_signature_and_content(self):
        from pypdf import PdfReader
        client = self._client()
        for report_type, spec in rds.REPORTS.items():
            if "pdf" not in spec["formats"]:
                continue
            extra = "&district_id=UGA_ABIM" if "district_required" in spec["filters"] else ""
            data = self._gen(client, report_type, "pdf", extra).data
            self.assertTrue(data.startswith(b"%PDF"), report_type)
            reader = PdfReader(io.BytesIO(data))
            self.assertGreaterEqual(len(reader.pages), 1, report_type)
            text = " ".join((p.extract_text() or "") for p in reader.pages)
            self.assertIn("Ministry of Health", text, report_type)
            self.assertIn("2025W52", text, report_type)
            self.assertIn("Page 1", text, report_type)

    # ---- Excel ---------------------------------------------------------
    def test_excel_workbook_structure_and_safety(self):
        client = self._client()
        expected = {
            "national_summary": {"Metadata", "KPI Summary", "District Risk", "Model Selection", "Limitations"},
            "district_risk_list": {"Metadata", "District Risk", "Limitations"},
        }
        for report_type, sheets in expected.items():
            extra = "&district_id=UGA_ABIM" if "district_required" in rds.REPORTS[report_type]["filters"] else ""
            data = self._gen(client, report_type, "xlsx", extra).data
            self.assertTrue(data.startswith(b"PK"), report_type)
            zf = zipfile.ZipFile(io.BytesIO(data))
            names = zf.namelist()
            self.assertFalse(any("vbaProject" in n for n in names), "workbook contains macros")
            self.assertFalse(any("externalLink" in n for n in names), "workbook has external links")
            workbook = zf.read("xl/workbook.xml").decode("utf-8")
            found = set(re.findall(r'<sheet name="([^"]+)"', workbook))
            self.assertTrue(sheets.issubset(found), f"{report_type}: {found}")

    def test_excel_risk_sheet_has_frozen_header_filter_and_numbers(self):
        data = self._gen(self._client(), "district_risk_list", "xlsx").data
        zf = zipfile.ZipFile(io.BytesIO(data))
        sheet = zf.read("xl/worksheets/sheet2.xml").decode("utf-8")
        self.assertIn("<pane ", sheet, "header row is not frozen")
        self.assertIn("<autoFilter", sheet, "autofilter is not enabled")
        self.assertGreater(len(re.findall(r"<v>", sheet)), 100, "numeric cells are missing")

    # ---- CSV -----------------------------------------------------------
    def test_csv_contract(self):
        resp = self._gen(self._client(), "district_risk_list", "csv")
        self.assertTrue(resp.data.startswith(b"\xef\xbb\xbf"), "CSV is not UTF-8 BOM encoded")
        self.assertEqual(resp.headers["Content-Type"].count("charset"), 1)
        rows = list(csvmod.reader(io.StringIO(resp.data.decode("utf-8-sig"))))
        self.assertEqual(len(rows) - 1, 146, "expected one row per district")
        self.assertEqual(rows[0][0], "District ID")
        self.assertNotIn("Trend", rows[0], "unsupported Trend column present")
        self.assertFalse(any("<" in cell for row in rows for cell in row), "HTML leaked into CSV")

    def test_csv_risk_filter_narrows_rows(self):
        client = self._client()
        everything = self._gen(client, "district_risk_list", "csv")
        high = self._gen(client, "district_risk_list", "csv", "&risk_level=High")
        all_rows = list(csvmod.reader(io.StringIO(everything.data.decode("utf-8-sig"))))
        high_rows = list(csvmod.reader(io.StringIO(high.data.decode("utf-8-sig"))))
        self.assertLess(len(high_rows), len(all_rows))
        for row in high_rows[1:]:
            self.assertEqual(row[2], "High")

    # ---- content contracts ---------------------------------------------
    def test_national_report_matches_service_values(self):
        """Report values must equal the source service values, not another page."""
        from app.services import dashboard_data as dd
        with self.app.app_context():
            params = dd.resolve_params(horizon=4)
            summary = dd.build_summary_section(params)
            models = dd.selected_models(4)
            ctx = rds.build_context(rds.ReportRequest(report_type="national_summary", fmt="pdf", horizon=4))
        self.assertEqual(ctx["summary"]["confirmed_malaria_cases"], summary["confirmed_malaria_cases"])
        self.assertEqual(ctx["metadata"]["count_model_name"], models["count"]["model_name"])
        self.assertEqual(ctx["metadata"]["surge_model_name"], models["surge"]["model_name"])
        self.assertEqual(ctx["metadata"]["issue_label"], "2025W52")

    def test_district_report_reuses_backend_risk_basis(self):
        from app.services import dashboard_data as dd
        with self.app.app_context():
            params = dd.resolve_params(horizon=4, district_id="UGA_ABIM")
            row = next(r for r in dd.build_risk_table_section(params)["rows"]
                       if r["district_id"] == "UGA_ABIM")
            ctx = rds.build_context(rds.ReportRequest(
                report_type="district_forecast", fmt="pdf", horizon=4, district_id="UGA_ABIM"))
        self.assertEqual(ctx["district"]["predicted_cases"], row["predicted_cases"])
        self.assertEqual(ctx["district"]["threshold_used"], row["threshold_used"])
        self.assertEqual(ctx["district"]["predicted_surge_score"], row["predicted_surge_score"])
        # Interpretation must include the backend risk_basis verbatim, not a re-derivation.
        self.assertIn(row["risk_basis"], ctx["interpretation"])

    def test_data_quality_report_uses_corrected_coverage_contract(self):
        with self.app.app_context():
            ctx = rds.build_context(rds.ReportRequest(report_type="data_quality", fmt="pdf"))
        cov = ctx["coverage"]
        self.assertEqual(cov["expected_district_weeks"], 45698)
        self.assertEqual(cov["received_district_weeks"], 45451)
        self.assertEqual(cov["missing_district_weeks"], 247)
        self.assertEqual(len(ctx["climate_lag_columns"]), 9)

    # ---- Phase 29D corrective pass: shared evidence + fail-closed --------
    def test_data_quality_report_and_lifecycle_api_share_one_builder(self):
        """The report and the lifecycle API must return identical validated evidence."""
        client = self._client()
        api = client.get("/api/lifecycle/data-quality").get_json()["data"]
        with self.app.app_context():
            ctx = rds.build_context(rds.ReportRequest(report_type="data_quality", fmt="pdf"))
        self.assertEqual(api["rows_with_complete_climate_lags"],
                         ctx["rows_with_complete_climate_lags"])
        self.assertEqual(api["rows_with_complete_climate_lags"], 45451)
        self.assertEqual({k: v["value"] for k, v in api["leakage_controls"].items()},
                         {k: v["value"] for k, v in ctx["leakage_controls"].items()})

    def test_data_quality_report_publishes_evidence_derived_controls(self):
        with self.app.app_context():
            ctx = rds.build_context(rds.ReportRequest(report_type="data_quality", fmt="pdf"))
        controls = ctx["leakage_controls"]
        for key in (
            "missing_not_zero_filled",
            "future_target_alignment",
            "training_only_preprocessing",
            "adjudicated_errors_excluded_from_outcomes",
            "bounded_predictor_only_interpolation",
        ):
            self.assertIn(key, controls)
            self.assertIsInstance(controls[key]["value"], bool)
            self.assertTrue(controls[key]["source"])
            self.assertNotEqual(controls[key]["source"], "unavailable")

    def test_data_quality_fails_closed_on_broken_model_card_evidence(self):
        """Broken preprocessing evidence must 503 BOTH the report and the API."""
        from app.services import explainability as ex
        original = ex.load_json

        def broken(key):
            payload = original(key)
            if key == "random_forest_card":
                payload = dict(payload)
                payload["preprocessing"] = "does not mention training rows"
            return payload

        client = self._client()
        with patch("app.services.explainability.load_json", side_effect=broken):
            report = client.get("/reports/generate?report_type=data_quality&format=pdf")
            api = client.get("/api/lifecycle/data-quality")
        self.assertEqual(report.status_code, 503)
        self.assertEqual(report.get_json()["status"], "unavailable")
        self.assertEqual(api.status_code, 503)
        self.assertNotIn("Traceback", report.get_data(as_text=True))

    def test_data_quality_fails_closed_when_critical_control_is_false(self):
        """Well-formed evidence must not publish a report when a control is disproved."""
        from app.services import explainability as ex
        original = ex.load_json

        def disproved(key):
            payload = original(key)
            if key == "production_manifest":
                payload = json.loads(json.dumps(payload))
                payload["leakage_controls"]["missing_malaria_zero_filled"] = True
            return payload

        client = self._client()
        with patch("app.services.explainability.load_json", side_effect=disproved):
            report = client.get("/reports/generate?report_type=data_quality&format=pdf")
            api = client.get("/api/lifecycle/data-quality")
        self.assertEqual(report.status_code, 503)
        self.assertEqual(report.get_json()["status"], "unavailable")
        self.assertEqual(api.status_code, 503)
        self.assertNotIn("Traceback", report.get_data(as_text=True))

    def test_data_quality_fails_closed_on_inconsistent_coverage(self):
        from app.services import data_quality_evidence as dqe
        bad = {"components": {"dhis2_malaria_panel": {"status": "complete", "row_counts": {
            "expected_district_weeks": 45698, "district_week_features": 45451,
            "missing_district_weeks": 999, "feature_explicit_zero_count": 0,
            "districts_represented": 146}}}}
        client = self._client()
        with patch("app.services.project_state.build_project_state_summary", return_value=bad):
            resp = client.get("/reports/generate?report_type=data_quality&format=xlsx")
        self.assertEqual(resp.status_code, 503)

    def test_temp_file_creation_failure_returns_controlled_json(self):
        client = self._client()
        with patch("app.blueprints.reports.routes.temp_path", side_effect=OSError("disk full")):
            resp = client.get("/reports/generate?report_type=methods&format=pdf")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.mimetype, "application/json")
        body = resp.get_data(as_text=True)
        self.assertNotIn("Traceback", body)
        self.assertNotIn("disk full", body)
        self.assertNotIn(":\\", body)  # no absolute path leak

    def test_oversized_output_returns_controlled_response(self):
        from app.blueprints.reports import routes
        client = self._client()
        with patch.object(routes, "MAX_OUTPUT_BYTES", 10):  # force any report over the ceiling
            resp = client.get("/reports/generate?report_type=methods&format=pdf")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["status"], "too_large")

    def test_concurrent_generation_limit_returns_busy(self):
        from app.blueprints.reports import routes
        client = self._client()
        # Exhaust every generation slot, then a further request must be refused.
        acquired = [routes._generation_slots.acquire(blocking=False)
                    for _ in range(routes.MAX_CONCURRENT_GENERATIONS)]
        try:
            self.assertTrue(all(acquired))
            resp = client.get("/reports/generate?report_type=methods&format=pdf")
            self.assertEqual(resp.status_code, 503)
            self.assertEqual(resp.get_json()["status"], "busy")
        finally:
            for _ in acquired:
                routes._generation_slots.release()

    def test_generation_limit_precedes_context_build_and_covers_print(self):
        from app.blueprints.reports import routes
        client = self._client()
        acquired = [routes._generation_slots.acquire(blocking=False)
                    for _ in range(routes.MAX_CONCURRENT_GENERATIONS)]
        try:
            self.assertTrue(all(acquired))
            with patch.object(routes, "build_context") as build:
                download = client.get("/reports/generate?report_type=methods&format=pdf")
                printable = client.get("/reports/print?report_type=methods")
            self.assertEqual(download.status_code, 503)
            self.assertEqual(download.get_json()["status"], "busy")
            self.assertEqual(printable.status_code, 503)
            self.assertIn("report service is busy", printable.get_data(as_text=True).lower())
            build.assert_not_called()
        finally:
            for _ in acquired:
                routes._generation_slots.release()

    def test_reports_page_uses_searchable_district_combobox(self):
        html = self._client().get("/reports/").get_data(as_text=True)
        self.assertIn('id="rp-district-search"', html)
        self.assertIn('role="combobox"', html)
        self.assertIn('id="rp-district-options"', html)
        self.assertIn('role="listbox"', html)
        # The hidden select remains the source of truth with canonical ids.
        self.assertIn('<select id="rp-district" hidden', html)
        self.assertIn("combobox.js", html)

    def test_model_report_has_four_models_and_selection(self):
        with self.app.app_context():
            ctx = rds.build_context(rds.ReportRequest(report_type="model_performance", fmt="pdf", horizon=4))
        rows = [r for r in ctx["performance"]["rows"] if r["horizon"] == 4]
        self.assertEqual({r["model_name"] for r in rows},
                         {"Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost"})
        self.assertEqual(ctx["selection"]["selected_count_model"], "Random Forest")
        self.assertEqual(ctx["selection"]["selected_surge_model"], "Random Forest")
        self.assertFalse(ctx["selection"]["winner_recalculated"])

    def test_methods_report_states_approved_methodology_only(self):
        with self.app.app_context():
            ctx = rds.build_context(rds.ReportRequest(report_type="methods", fmt="pdf"))
        blob = json.dumps(ctx["sections"]).lower()
        self.assertIn("rolling-origin", blob)
        self.assertIn("no random split", blob)
        self.assertIn("adjudicated reporting errors", blob)
        self.assertIn("predictor-only interpolation", blob)
        self.assertNotIn("population denominator", blob)
        self.assertNotIn("incidence target", blob)
        self.assertIn("never zero-filled", blob)
        # The excluded methods may only be DISCLAIMED, never claimed as used.
        self.assertIn("no negative binomial model", blob)
        self.assertIn("no wind-speed variable", blob)
        self.assertNotRegex(blob, r"(?<!no )negative binomial (model )?is used")
        approved = [s for h, s in ctx["sections"] if h == "Approved models"][0]
        self.assertIn("Baseline/Naive, SARIMAX, Random Forest and XGBoost", approved)

    def test_climate_report_is_non_causal(self):
        with self.app.app_context():
            ctx = rds.build_context(rds.ReportRequest(report_type="climate_signals", fmt="pdf", horizon=4))
        note = (ctx["climate"].get("causation_note") or "").lower()
        self.assertIn("does not establish causation", note)
        self.assertFalse(ctx["climate"].get("correlation_available"))

    def test_reports_carry_limitations_and_research_use(self):
        with self.app.app_context():
            for report_type in ALL_REPORTS:
                extra = {"district_id": "UGA_ABIM"} if "district_required" in rds.REPORTS[report_type]["filters"] else {}
                ctx = rds.build_context(rds.ReportRequest(report_type=report_type, fmt="pdf", horizon=4, **extra))
                self.assertTrue(ctx["limitations"], report_type)
                self.assertIn("does not confirm an outbreak", " ".join(ctx["limitations"]))
                self.assertIn("research", ctx["metadata"]["research_use"].lower())

    # ---- fail closed ----------------------------------------------------
    def test_missing_evidence_fails_report_closed(self):
        client = self._client()
        with patch("app.services.report_data_service.dd.build_summary_section",
                   side_effect=rds.dd.DashboardDataError("gone")):
            resp = self._gen(client, "national_summary", "pdf")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["status"], "unavailable")
        self.assertNotIn("Traceback", resp.get_data(as_text=True))

    # ---- read-only ------------------------------------------------------
    def test_no_database_writes_during_generation(self):
        client = self._client()
        statements = []
        from sqlalchemy import event

        with self.app.app_context():
            engine = db.engine

        def before_cursor(conn, cursor, statement, params, context, executemany):
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", before_cursor)
        try:
            with patch.object(db.session, "commit") as commit, patch.object(db.session, "flush") as flush:
                for report_type, spec in rds.REPORTS.items():
                    extra = "&district_id=UGA_ABIM" if "district_required" in spec["filters"] else ""
                    self._gen(client, report_type, spec["formats"][0], extra)
                commit.assert_not_called()
                flush.assert_not_called()
        finally:
            event.remove(engine, "before_cursor_execute", before_cursor)
        for statement in statements:
            head = statement.strip().split(None, 1)[0].upper() if statement.strip() else ""
            self.assertNotIn(head, {"INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER"},
                             f"write statement issued: {statement[:120]}")

    def test_forecast_fingerprint_unchanged_by_report_generation(self):
        from app.models import ForecastOutput

        def fingerprint():
            with self.app.app_context():
                rows = ForecastOutput.query.order_by(ForecastOutput.id).all()
                payload = [[r.id, r.district_id, r.model_name, r.horizon_weeks,
                            str(r.predicted_cases), str(r.threshold_used), r.risk_level]
                           for r in rows]
                return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), len(rows)

        before, count_before = fingerprint()
        client = self._client()
        for report_type, spec in rds.REPORTS.items():
            extra = "&district_id=UGA_ABIM" if "district_required" in spec["filters"] else ""
            self._gen(client, report_type, spec["formats"][0], extra)
        after, count_after = fingerprint()
        self.assertEqual(before, after)
        self.assertEqual(count_before, 1752)
        self.assertEqual(count_after, 1752)

    # ---- page + regression ----------------------------------------------
    def test_reports_page_renders_registry_only(self):
        html = self._client().get("/reports/").get_data(as_text=True)
        for spec in rds.REPORTS.values():
            self.assertIn(spec["title"], html)
        self.assertNotIn("PowerPoint", html)
        self.assertNotIn("pptx", html.lower())
        self.assertIn("do not trigger new data", html)

    def test_placeholder_text_removed(self):
        html = self._client().get("/reports/").get_data(as_text=True)
        self.assertNotIn("planned for a later implementation phase", html)

    def test_phase_29abc_regression_intact(self):
        client = self._client()
        for url in ("/", "/api/dashboard/summary", "/api/explainability/overview",
                    "/api/system/pipeline-status", "/dhis2/", "/modelling/", "/forecasts/"):
            self.assertEqual(client.get(url).status_code, 200, url)
        self.assertTrue(self.app.json.compact)
        map_bytes = len(client.get("/api/dashboard/map").data)
        self.assertLessEqual(map_bytes, 2_000_000)


if __name__ == "__main__":
    unittest.main()
