from __future__ import annotations

import json
import os
import unittest
from decimal import Decimal
from unittest.mock import patch

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.models import DistrictReference, DistrictWeekFeature, PipelineRun
from app.services.dhis2_client import DHIS2Config, DHIS2HTTPError
from app.services.dhis2_extraction import PeriodChunk
from app.services.reporting_completeness import (
    ReportingCompletenessError,
    extract_reporting_completeness,
    normalize_completeness_value,
    search_reporting_metrics,
    verify_reporting_metric,
)


TEST_URL_PREFIX = "https://hmis.example.org/prompt18"


class FakeClient:
    def __init__(self, *, payloads=None):
        self.config = DHIS2Config(
            base_url=TEST_URL_PREFIX,
            username="analyst",
            password="never-store-this-password",
        )
        self.payloads = list(payloads or [])
        self.calls = []

    def verify_login(self):
        return {"id": "me"}

    def get(self, path, *, params=None):
        self.calls.append((path, params))
        if path == "/api/indicators.json":
            return {
                "indicators": [
                    {
                        "id": "DX_REPORT",
                        "code": "RR",
                        "name": "Weekly facility reporting rate",
                        "displayName": "Weekly facility reporting rate",
                        "numerator": "#{actual_reports}",
                        "denominator": "#{expected_reports}",
                    }
                ]
            }
        if path in {
            "/api/dataElements.json",
            "/api/programIndicators.json",
            "/api/dataSets.json",
        }:
            key = {
                "/api/dataElements.json": "dataElements",
                "/api/programIndicators.json": "programIndicators",
                "/api/dataSets.json": "dataSets",
            }[path]
            return {key: []}
        if path == "/api/indicators/DX_REPORT.json":
            return {
                "id": "DX_REPORT",
                "code": "RR",
                "name": "Weekly facility reporting rate",
                "displayName": "Weekly facility reporting rate",
                "numerator": "#{actual_reports}",
                "denominator": "#{expected_reports}",
            }
        if path.endswith("/MISSING.json"):
            raise DHIS2HTTPError("missing", status_code=404)
        raise AssertionError(f"Unexpected fake DHIS2 path: {path}")

    def analytics_query(self, *, dx, periods, level):
        self.calls.append(("analytics", dx, list(periods), level))
        if self.payloads:
            return self.payloads.pop(0)
        return analytics_payload([])


def analytics_payload(rows):
    return {
        "headers": [
            {"name": "dx"},
            {"name": "pe"},
            {"name": "ou"},
            {"name": "value"},
        ],
        "rows": rows,
        "metaData": {
            "items": {
                "DX_REPORT": {"name": "Weekly facility reporting rate"},
                "TEST_REPORT_OU": {"name": "Mapped District"},
            }
        },
    }


class ReportingCompletenessPureTests(unittest.TestCase):
    def test_normalize_completeness_values_to_percent(self):
        self.assertEqual(
            normalize_completeness_value(Decimal("0.875")),
            (Decimal("87.50"), "proportion_0_1_to_percent"),
        )
        self.assertEqual(
            normalize_completeness_value(Decimal("87.5")),
            (Decimal("87.50"), "percent_0_100"),
        )
        with self.assertRaises(ReportingCompletenessError):
            normalize_completeness_value(Decimal("145"))

    def test_search_and_verify_reporting_metric_candidate(self):
        client = FakeClient()
        search = search_reporting_metrics(client, "reporting rate")
        self.assertEqual(len(search.candidates), 1)
        self.assertEqual(search.candidates[0].uid, "DX_REPORT")
        self.assertEqual(
            search.candidates[0].suitability,
            "plausible_reporting_completeness_metric",
        )

        verification = verify_reporting_metric(client, "DX_REPORT")
        self.assertTrue(verification.usable_for_dry_run)
        self.assertTrue(verification.metadata_plausible)
        self.assertTrue(verification.extractable_dx)


class ReportingCompletenessDBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL is required for DB integration tests.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        self._cleanup()

    def tearDown(self):
        self._cleanup()
        self.ctx.pop()

    def _cleanup(self):
        DistrictWeekFeature.query.filter(
            DistrictWeekFeature.district_id.like("UGA_TEST_REPORT%")
        ).delete(synchronize_session=False)
        DistrictReference.query.filter(
            DistrictReference.district_id.like("UGA_TEST_REPORT%")
        ).delete(synchronize_session=False)
        PipelineRun.query.filter(
            PipelineRun.source_version_or_url.like(f"{TEST_URL_PREFIX}%")
        ).delete(synchronize_session=False)
        db.session.commit()

    def _env(self):
        return {
            "DHIS2_REPORTING_COMPLETENESS_DX": "DX_REPORT",
            "DHIS2_DISTRICT_LEVEL": "3",
        }

    def _district_and_feature(self):
        db.session.add(
            DistrictReference(
                district_id="UGA_TEST_REPORT_A",
                district_name="Mapped District",
                dhis2_orgunit_id="TEST_REPORT_OU",
                active=True,
            )
        )
        db.session.add(
            DistrictWeekFeature(
                district_id="UGA_TEST_REPORT_A",
                epi_year=2025,
                epi_week=1,
                confirmed_malaria_cases=10,
            )
        )
        db.session.commit()

    def test_dry_run_does_not_write_completeness(self):
        self._district_and_feature()
        client = FakeClient(
            payloads=[analytics_payload([["DX_REPORT", "2025W01", "TEST_REPORT_OU", "0.9"]])]
        )

        with patch.dict(os.environ, self._env(), clear=False):
            summary = extract_reporting_completeness(
                client,
                [PeriodChunk(label="2025", periods=["2025W01"])],
                dry_run=True,
            )

        self.assertEqual(summary.rows_normalized, 1)
        self.assertEqual(summary.rows_updated, 0)
        row = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_REPORT_A", epi_year=2025, epi_week=1
        ).one()
        self.assertIsNone(row.facility_reporting_completeness)

    def test_real_run_updates_existing_feature_without_creating_missing_week(self):
        self._district_and_feature()
        client = FakeClient(
            payloads=[
                analytics_payload(
                    [
                        ["DX_REPORT", "2025W01", "TEST_REPORT_OU", "87.5"],
                        ["DX_REPORT", "2025W02", "TEST_REPORT_OU", "91.0"],
                    ]
                )
            ]
        )

        with patch.dict(os.environ, self._env(), clear=False):
            summary = extract_reporting_completeness(
                client,
                [PeriodChunk(label="2025", periods=["2025W01", "2025W02"])],
            )

        self.assertEqual(summary.rows_normalized, 2)
        self.assertEqual(summary.rows_updated, 1)
        self.assertEqual(summary.rows_without_feature, 1)
        row = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_REPORT_A", epi_year=2025, epi_week=1
        ).one()
        self.assertEqual(row.facility_reporting_completeness, Decimal("87.50"))
        self.assertEqual(
            DistrictWeekFeature.query.filter_by(
                district_id="UGA_TEST_REPORT_A", epi_year=2025, epi_week=2
            ).count(),
            0,
        )

        run = PipelineRun.query.filter_by(
            run_type="dhis2_reporting_completeness"
        ).order_by(PipelineRun.id.desc()).first()
        self.assertIsNotNone(run)
        serialized = json.dumps({"parameters": run.parameters, "manifest": run.manifest})
        self.assertNotIn("never-store-this-password", serialized)
        self.assertIn(
            "district_week_features.facility_reporting_completeness",
            serialized,
        )


if __name__ == "__main__":
    unittest.main()
