from __future__ import annotations

import os
import unittest
from decimal import Decimal
from unittest.mock import patch

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.models import DistrictReference, DistrictWeekFeature, PipelineRun, RawDHIS2Malaria
from app.services.dhis2_client import DHIS2Config, DHIS2HTTPError, DHIS2TimeoutError
from app.services.dhis2_extraction import (
    DHIS2ExtractionError,
    PeriodChunk,
    build_extraction_plan,
    extract_malaria_raw,
    generate_week_periods,
    parse_analytics_rows,
    validate_extractable_metadata,
)


TEST_URL_PREFIX = "https://hmis.example.org/prompt7"


class FakeClient:
    def __init__(self, *, payloads=None, failures=None, base_url=TEST_URL_PREFIX):
        self.config = DHIS2Config(
            base_url=base_url,
            username="analyst",
            password="never-store-this-password",
        )
        self.payloads = list(payloads or [])
        self.failures = list(failures or [])
        self.calls: list[list[str]] = []

    def verify_login(self):
        return {"id": "me"}

    def fetch_metadata_object(self, uid):
        return (
            "dataElement",
            {
                "id": uid,
                "name": "Malaria (Confirmed) - Cases",
                "displayName": "Malaria (Confirmed) - Cases",
                "aggregationType": "SUM",
                "categoryCombo": {"name": "default"},
                "categoryOptionCombos": [{"id": "default", "name": "default"}],
            },
        )

    def fetch_org_units_at_level(self, level):
        return [{"id": "TEST_OU_MAPPED", "name": "Mapped"}]

    def analytics_query(self, *, dx, periods, level):
        self.calls.append(list(periods))
        if self.failures:
            failure = self.failures.pop(0)
            if failure is not None:
                raise failure
        if self.payloads:
            return self.payloads.pop(0)
        return {
            "headers": [
                {"name": "dx"},
                {"name": "pe"},
                {"name": "ou"},
                {"name": "value"},
            ],
            "rows": [],
            "metaData": {"items": {}},
        }


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
                "DX_TEST": {"name": "Malaria (Confirmed) - Cases"},
                "TEST_OU_MAPPED": {"name": "Mapped District"},
                "TEST_OU_UNMAPPED": {"name": "Unmapped District"},
            }
        },
    }


class DHIS2ExtractionPureTests(unittest.TestCase):
    def test_period_generation_zero_pads_and_validates(self):
        self.assertEqual(generate_week_periods(2025, week=1), ["2025W01"])
        self.assertEqual(generate_week_periods(2025, week=20), ["2025W20"])
        with self.assertRaises(DHIS2ExtractionError):
            generate_week_periods(2025, start_week=20, end_week=1)

    def test_build_plan_defaults_to_52_weeks(self):
        chunks = build_extraction_plan(year=2025)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].periods[0], "2025W01")
        self.assertEqual(chunks[0].periods[-1], "2025W52")

    def test_build_plan_uses_iso_week_53_for_2020(self):
        chunks = build_extraction_plan(year=2020)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0].periods), 53)
        self.assertEqual(chunks[0].periods[-1], "2020W53")

    def test_invalid_week_53_for_52_week_year_is_rejected(self):
        with self.assertRaises(DHIS2ExtractionError):
            build_extraction_plan(year=2025, week=53)

    def test_parse_analytics_rows_by_headers(self):
        payload = {
            "headers": [
                {"name": "ou"},
                {"name": "value"},
                {"name": "pe"},
                {"name": "dx"},
            ],
            "rows": [["TEST_OU_MAPPED", "0", "2025W01", "DX_TEST"]],
            "metaData": {"items": {"TEST_OU_MAPPED": {"displayName": "Mapped"}}},
        }
        rows = parse_analytics_rows(payload, fallback_dx="DX_TEST", fallback_dx_name=None)
        self.assertEqual(rows[0].value, Decimal("0"))
        self.assertEqual(rows[0].epi_week, 1)
        self.assertEqual(rows[0].period, "2025W01")
        self.assertEqual(rows[0].dhis2_orgunit_name, "Mapped")

    def test_parse_analytics_rows_normalizes_unpadded_week_labels(self):
        rows = parse_analytics_rows(
            analytics_payload([["DX_TEST", "2025W1", "TEST_OU_MAPPED", "7"]]),
            fallback_dx="DX_TEST",
            fallback_dx_name=None,
        )
        self.assertEqual(rows[0].period, "2025W01")
        self.assertEqual(rows[0].epi_week, 1)

    def test_disaggregation_guard_blocks_non_default_combo(self):
        with self.assertRaises(DHIS2ExtractionError):
            validate_extractable_metadata(
                "dataElement",
                {
                    "categoryCombo": {"name": "Age/Sex"},
                    "categoryOptionCombos": [{"id": "a"}, {"id": "b"}],
                },
            )


class DHIS2ExtractionDBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
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
            DistrictWeekFeature.district_id.like("UGA_TEST_EXTRACT%")
        ).delete(synchronize_session=False)
        RawDHIS2Malaria.query.filter(
            RawDHIS2Malaria.dhis2_orgunit_id.like("TEST_OU_%")
        ).delete(synchronize_session=False)
        DistrictReference.query.filter(
            DistrictReference.district_id.like("UGA_TEST_EXTRACT%")
        ).delete(synchronize_session=False)
        PipelineRun.query.filter(
            PipelineRun.source_version_or_url.like(f"{TEST_URL_PREFIX}%")
        ).delete(synchronize_session=False)
        db.session.commit()

    def _env(self):
        return {
            "DHIS2_CONFIRMED_MALARIA_DX": "DX_TEST",
            "DHIS2_DISTRICT_LEVEL": "3",
        }

    def test_explicit_zero_stored_and_unmapped_kept_null_without_feature_rows(self):
        db.session.add(
            DistrictReference(
                district_id="UGA_TEST_EXTRACT_MAPPED",
                district_name="Mapped",
                dhis2_orgunit_id="TEST_OU_MAPPED",
            )
        )
        db.session.commit()
        client = FakeClient(
            payloads=[
                analytics_payload(
                    [
                        ["DX_TEST", "2025W01", "TEST_OU_MAPPED", "0"],
                        ["DX_TEST", "2025W01", "TEST_OU_UNMAPPED", "7"],
                    ]
                )
            ],
            base_url=f"{TEST_URL_PREFIX}/zero",
        )
        with patch.dict(os.environ, self._env(), clear=False):
            summary = extract_malaria_raw(
                client, [PeriodChunk(label="2025-test", periods=["2025W01"])]
            )

        self.assertEqual(summary.rows_inserted, 2)
        self.assertEqual(summary.explicit_zero_values, 1)
        mapped = RawDHIS2Malaria.query.filter_by(dhis2_orgunit_id="TEST_OU_MAPPED").one()
        unmapped = RawDHIS2Malaria.query.filter_by(
            dhis2_orgunit_id="TEST_OU_UNMAPPED"
        ).one()
        self.assertEqual(mapped.value, Decimal("0.0000"))
        self.assertEqual(mapped.district_id, "UGA_TEST_EXTRACT_MAPPED")
        self.assertIsNone(unmapped.district_id)
        run = db.session.get(PipelineRun, summary.pipeline_run_id)
        serialized = str(run.parameters) + str(run.manifest)
        self.assertNotIn("never-store-this-password", serialized)
        self.assertEqual(
            DistrictWeekFeature.query.filter(
                DistrictWeekFeature.district_id.like("UGA_TEST_EXTRACT%")
            ).count(),
            0,
        )

    def test_dry_run_does_not_insert_pipeline_or_raw_rows(self):
        client = FakeClient(base_url=f"{TEST_URL_PREFIX}/dry-run")
        with patch.dict(os.environ, self._env(), clear=False):
            summary = extract_malaria_raw(
                client,
                [PeriodChunk(label="dry", periods=["2025W01"])],
                dry_run=True,
            )
        self.assertTrue(summary.dry_run)
        self.assertIsNone(summary.pipeline_run_id)
        self.assertEqual(client.calls, [["2025W01"]])
        self.assertEqual(
            RawDHIS2Malaria.query.filter(
                RawDHIS2Malaria.dhis2_orgunit_id.like("TEST_OU_%")
            ).count(),
            0,
        )
        self.assertEqual(
            PipelineRun.query.filter(
                PipelineRun.source_version_or_url.like(f"{TEST_URL_PREFIX}/dry-run%")
            ).count(),
            0,
        )

    def test_failed_extraction_marks_pipeline_run_failed(self):
        client = FakeClient(
            failures=[DHIS2HTTPError("server failed", status_code=500)],
            base_url=f"{TEST_URL_PREFIX}/failed",
        )
        with patch.dict(os.environ, self._env(), clear=False):
            with self.assertRaises(DHIS2ExtractionError):
                extract_malaria_raw(
                    client, [PeriodChunk(label="fail", periods=["2025W01"])]
                )
        run = PipelineRun.query.filter_by(source_version_or_url=f"{TEST_URL_PREFIX}/failed").one()
        self.assertEqual(run.status, "failed")
        self.assertNotIn("never-store-this-password", run.error_message or "")

    def test_yearly_timeout_falls_back_to_smaller_chunks(self):
        client = FakeClient(
            failures=[DHIS2TimeoutError("too slow"), DHIS2TimeoutError("too slow")],
            base_url=f"{TEST_URL_PREFIX}/fallback",
        )
        with patch.dict(os.environ, self._env(), clear=False):
            summary = extract_malaria_raw(
                client,
                [PeriodChunk(label="2025", periods=["2025W01", "2025W02"])],
            )
        self.assertTrue(summary.fallback_used)
        self.assertEqual(client.calls[-2:], [["2025W01"], ["2025W02"]])
        self.assertEqual(summary.empty_periods, {"2025W01", "2025W02"})


if __name__ == "__main__":
    unittest.main()
