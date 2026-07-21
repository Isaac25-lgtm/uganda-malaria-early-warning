from __future__ import annotations

import os
import unittest
from decimal import Decimal

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.models import DistrictReference, DistrictWeekFeature, PipelineRun, RawDHIS2Malaria
from app.services.preprocessing import (
    PreprocessingError,
    PreprocessingSelection,
    build_district_week_features,
    build_selection,
    expected_periods_for_years,
    iso_weeks_in_year,
)


TEST_URL_PREFIX = "https://hmis.example.org/prompt8"


class PreprocessingCalendarTests(unittest.TestCase):
    def test_iso_week_denominator_for_study_years(self):
        self.assertEqual(iso_weeks_in_year(2020), 53)
        for year in range(2021, 2026):
            self.assertEqual(iso_weeks_in_year(year), 52)
        periods = expected_periods_for_years(2020, 2020)
        self.assertEqual(periods[0], "2020W01")
        self.assertEqual(periods[-1], "2020W53")


class PreprocessingDBTests(unittest.TestCase):
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
            DistrictWeekFeature.district_id.like("UGA_TEST_PREP%")
        ).delete(synchronize_session=False)
        RawDHIS2Malaria.query.filter(
            RawDHIS2Malaria.dhis2_orgunit_id.like("TEST_PREP_%")
        ).delete(synchronize_session=False)
        DistrictReference.query.filter(
            DistrictReference.district_id.like("UGA_TEST_PREP%")
        ).delete(synchronize_session=False)

        test_source_ids = [
            row.id
            for row in PipelineRun.query.filter(
                PipelineRun.source_version_or_url.like(f"{TEST_URL_PREFIX}%")
            ).all()
        ]
        if test_source_ids:
            for run in PipelineRun.query.filter_by(
                run_type="preprocessing", source_name="raw_dhis2_malaria"
            ).all():
                source_ids = (run.parameters or {}).get("source_run_ids") or []
                if set(source_ids) & set(test_source_ids):
                    db.session.delete(run)
            PipelineRun.query.filter(PipelineRun.id.in_(test_source_ids)).delete(
                synchronize_session=False
            )
        db.session.commit()

    def _district(self, district_id="UGA_TEST_PREP_A", orgunit_id="TEST_PREP_OU_A"):
        district = DistrictReference(
            district_id=district_id,
            district_name=district_id.replace("_", " "),
            dhis2_orgunit_id=orgunit_id,
        )
        db.session.add(district)
        db.session.commit()
        return district

    def _source_run(self, periods):
        run = PipelineRun(
            run_type="dhis2_extraction",
            status="success",
            source_name="DHIS2",
            source_version_or_url=f"{TEST_URL_PREFIX}/{len(periods)}",
            parameters={"periods": periods},
        )
        db.session.add(run)
        db.session.commit()
        return run

    def _raw(
        self,
        run,
        *,
        orgunit_id="TEST_PREP_OU_A",
        district_id=None,
        period="2025W01",
        value=Decimal("1"),
        dx="DX_TEST",
        completeness=None,
    ):
        year, week = period.split("W", 1)
        row = RawDHIS2Malaria(
            pipeline_run_id=run.id,
            dhis2_orgunit_id=orgunit_id,
            dhis2_orgunit_name=orgunit_id,
            district_id=district_id,
            data_element_id=dx,
            data_element_name="Malaria (Confirmed) - Cases",
            period=period,
            epi_year=int(year),
            epi_week=int(week),
            value=value,
            reporting_completeness=completeness,
            raw_payload={"test": True},
        )
        db.session.add(row)
        db.session.commit()
        return row

    def test_explicit_zero_counts_observed_and_missing_week_is_not_manufactured(self):
        self._district()
        run = self._source_run(["2025W01", "2025W02", "2025W03"])
        self._raw(run, period="2025W01", value=Decimal("0"), completeness=Decimal("88.5"))
        self._raw(run, period="2025W02", value=Decimal("12"), completeness=Decimal("91.0"))

        selection = PreprocessingSelection(
            source_run_ids=[run.id],
            expected_periods=["2025W01", "2025W02", "2025W03"],
        )
        summary = build_district_week_features(selection)

        self.assertEqual(summary.inserted, 2)
        zero_row = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_PREP_A", epi_year=2025, epi_week=1
        ).one()
        self.assertEqual(zero_row.confirmed_malaria_cases, 0)
        self.assertEqual(zero_row.facility_reporting_completeness, Decimal("88.50"))
        self.assertEqual(zero_row.time_series_completeness, Decimal("66.67"))
        self.assertIsNone(zero_row.population)
        self.assertIsNone(zero_row.incidence_per_1000)
        self.assertIsNone(zero_row.rainfall_current)
        self.assertEqual(
            DistrictWeekFeature.query.filter_by(
                district_id="UGA_TEST_PREP_A", epi_year=2025, epi_week=3
            ).count(),
            0,
        )

    def test_unmapped_org_units_are_excluded(self):
        run = self._source_run(["2025W01"])
        self._raw(run, orgunit_id="TEST_PREP_UNMAPPED", period="2025W01", value=Decimal("5"))
        summary = build_district_week_features(
            PreprocessingSelection(source_run_ids=[run.id], expected_periods=["2025W01"]),
            dry_run=True,
        )
        self.assertEqual(summary.raw_rows_unmapped, 1)
        self.assertIn("TEST_PREP_UNMAPPED", summary.unmapped_org_units)
        self.assertEqual(
            DistrictWeekFeature.query.filter(
                DistrictWeekFeature.district_id.like("UGA_TEST_PREP%")
            ).count(),
            0,
        )

    def test_duplicate_same_district_week_is_rejected(self):
        self._district()
        run = self._source_run(["2025W01"])
        self._raw(run, period="2025W01", value=Decimal("3"), dx="DX_A")
        self._raw(run, period="2025W01", value=Decimal("4"), dx="DX_B")
        with self.assertRaises(PreprocessingError):
            build_district_week_features(
                PreprocessingSelection(source_run_ids=[run.id], expected_periods=["2025W01"])
            )

    def test_many_to_one_requires_explicit_flag(self):
        self._district()
        run = self._source_run(["2025W01"])
        self._raw(
            run,
            orgunit_id="TEST_PREP_SPLIT_A",
            district_id="UGA_TEST_PREP_A",
            period="2025W01",
            value=Decimal("3"),
        )
        self._raw(
            run,
            orgunit_id="TEST_PREP_SPLIT_B",
            district_id="UGA_TEST_PREP_A",
            period="2025W01",
            value=Decimal("4"),
        )
        selection = PreprocessingSelection(source_run_ids=[run.id], expected_periods=["2025W01"])
        with self.assertRaises(PreprocessingError):
            build_district_week_features(selection)
        summary = build_district_week_features(selection, allow_many_to_one=True)
        row = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_PREP_A", epi_year=2025, epi_week=1
        ).one()
        self.assertEqual(row.confirmed_malaria_cases, 7)
        self.assertIn("UGA_TEST_PREP_A", summary.districts_processed)

    def test_dry_run_does_not_write_features_or_preprocessing_run(self):
        self._district()
        run = self._source_run(["2025W01"])
        self._raw(run, period="2025W01", value=Decimal("9"))
        selection = PreprocessingSelection(source_run_ids=[run.id], expected_periods=["2025W01"])
        summary = build_district_week_features(selection, dry_run=True)
        self.assertEqual(summary.inserted, 1)
        self.assertEqual(
            DistrictWeekFeature.query.filter(
                DistrictWeekFeature.district_id.like("UGA_TEST_PREP%")
            ).count(),
            0,
        )
        created_for_source = [
            pipeline_run
            for pipeline_run in PipelineRun.query.filter_by(
                run_type="preprocessing", source_name="raw_dhis2_malaria"
            ).all()
            if run.id in ((pipeline_run.parameters or {}).get("source_run_ids") or [])
        ]
        self.assertEqual(created_for_source, [])

    def test_52_week_source_for_2020_reports_iso_week_53_mismatch(self):
        self._district()
        periods = [f"2020W{week:02d}" for week in range(1, 53)]
        run = self._source_run(periods)
        self._raw(run, period="2020W01", value=Decimal("1"))
        selection = build_selection(pipeline_run_id=run.id)
        summary = build_district_week_features(selection, dry_run=True)
        self.assertTrue(any("omit 2020W53" in warning for warning in summary.warnings))


if __name__ == "__main__":
    unittest.main()
