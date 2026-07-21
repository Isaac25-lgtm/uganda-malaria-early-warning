from __future__ import annotations

import os
import unittest
from decimal import Decimal

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.models import DistrictReference, DistrictWeekClimate, DistrictWeekFeature, PipelineRun
from app.services.climate_lags import ClimateLagError, create_climate_lags
from app.services.preprocessing import iso_week_dates


class ClimateLagDBTests(unittest.TestCase):
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
        DistrictWeekClimate.query.filter(
            DistrictWeekClimate.district_id.like("UGA_TEST_LAG%")
        ).delete(synchronize_session=False)
        DistrictWeekFeature.query.filter(
            DistrictWeekFeature.district_id.like("UGA_TEST_LAG%")
        ).delete(synchronize_session=False)
        DistrictReference.query.filter(
            DistrictReference.district_id.like("UGA_TEST_LAG%")
        ).delete(synchronize_session=False)
        PipelineRun.query.filter(
            PipelineRun.source_name == "district_week_climate",
            PipelineRun.run_type == "climate_lag_creation",
            PipelineRun.row_count_raw <= 3,
        ).delete(
            synchronize_session=False
        )
        PipelineRun.query.filter(
            PipelineRun.source_name == "test climate lag source"
        ).delete(synchronize_session=False)
        db.session.commit()

    def _district(self):
        db.session.add(
            DistrictReference(
                district_id="UGA_TEST_LAG_A",
                district_name="TEST LAG A",
                active=True,
            )
        )
        db.session.commit()

    def _feature(self, year: int, week: int):
        start, end = iso_week_dates(year, week)
        db.session.add(
            DistrictWeekFeature(
                district_id="UGA_TEST_LAG_A",
                epi_year=year,
                epi_week=week,
                week_start_date=start,
                week_end_date=end,
                confirmed_malaria_cases=1,
            )
        )

    def _climate(self, year: int, week: int, rainfall: str):
        start, end = iso_week_dates(year, week)
        db.session.add(
            DistrictWeekClimate(
                district_id="UGA_TEST_LAG_A",
                epi_year=year,
                epi_week=week,
                week_start_date=start,
                week_end_date=end,
                rainfall_current=Decimal(rainfall),
                temperature_current=Decimal("22.5000"),
                humidity_current=Decimal("75.0000"),
                source_rainfall="test",
                source_temperature="test",
                source_humidity="test",
                aggregation_method="test_zonal",
                pipeline_run_id=self.run.id,
            )
        )

    def _seed(self):
        self._district()
        self.run = PipelineRun(
            run_type="climate_aggregation",
            status="success",
            source_name="test climate lag source",
        )
        db.session.add(self.run)
        db.session.commit()
        self._feature(2100, 2)
        self._feature(2100, 12)
        self._feature(2104, 20)
        self._climate(2099, 51, "51.000000")
        self._climate(2099, 53, "53.000000")
        self._climate(2104, 16, "516.000000")
        db.session.commit()

    def test_lags_use_week_start_dates_across_year_and_week53(self):
        self._seed()
        summary = create_climate_lags(
            start_year=2100,
            end_year=2104,
            variables=("rainfall",),
            lags=(4, 12),
        )
        self.assertEqual(summary.rows_with_updates, 3)

        row_2100w02 = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_LAG_A", epi_year=2100, epi_week=2
        ).one()
        self.assertEqual(row_2100w02.rainfall_lag_4, Decimal("51.000000"))

        row_2100w12 = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_LAG_A", epi_year=2100, epi_week=12
        ).one()
        self.assertEqual(row_2100w12.rainfall_lag_12, Decimal("53.000000"))

        row_2104w20 = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_LAG_A", epi_year=2104, epi_week=20
        ).one()
        self.assertEqual(row_2104w20.rainfall_lag_4, Decimal("516.000000"))
        self.assertTrue(summary.alignment_checks)

    def test_dry_run_writes_nothing(self):
        self._seed()
        summary = create_climate_lags(
            start_year=2100,
            end_year=2104,
            variables=("rainfall",),
            lags=(4,),
            dry_run=True,
        )
        self.assertGreater(summary.rows_with_updates, 0)
        row = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_LAG_A", epi_year=2100, epi_week=2
        ).one()
        self.assertIsNone(row.rainfall_lag_4)

    def test_missing_climate_rows_stop_lag_creation(self):
        self._district()
        self._feature(2104, 20)
        db.session.commit()
        with self.assertRaises(ClimateLagError):
            create_climate_lags(start_year=2104, end_year=2104)


if __name__ == "__main__":
    unittest.main()
