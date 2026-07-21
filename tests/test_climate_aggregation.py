from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.models import (
    DistrictReference,
    DistrictWeekClimate,
    DistrictWeekFeature,
    PipelineRun,
)
from app.services.climate_aggregation import (
    ClimateAggregationError,
    ClimateWeekRow,
    PreparedBoundaryProof,
    RegisteredClimateFile,
    _load_boundary_geodataframe,
    aggregate_weekly_climate,
    apply_processed_climate_to_features,
    build_real_climate_week_rows,
    kelvin_to_celsius,
    relative_humidity_percent,
    validate_prepared_boundary_for_climate,
    write_processed_climate_rows,
)
from app.services.preprocessing import iso_week_dates


class ClimateAggregationPureTests(unittest.TestCase):
    def test_temperature_conversion_and_humidity_derivation(self):
        self.assertEqual(kelvin_to_celsius(300), Decimal("26.8500"))
        rh = relative_humidity_percent(temperature_celsius=26.85, dewpoint_celsius=20.0)
        self.assertGreater(rh, Decimal("60"))
        self.assertLess(rh, Decimal("100"))

    def test_missing_prepared_boundary_reports_clear_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = validate_prepared_boundary_for_climate(Path(tmpdir) / "missing.geojson")
        self.assertFalse(summary.ok)
        self.assertIn("Prepared boundary GeoJSON not found", summary.errors[0])

    def test_duplicate_district_week_rows_are_rejected_before_write(self):
        week_start, week_end = iso_week_dates(2025, 1)
        row = ClimateWeekRow(
            district_id="UGA_TEST_CLIMATE_DUP",
            district_name="TEST CLIMATE DUP",
            epi_year=2025,
            epi_week=1,
            week_start_date=week_start,
            week_end_date=week_end,
            rainfall_current=Decimal("10.0"),
        )
        with self.assertRaises(ClimateAggregationError):
            write_processed_climate_rows([row, row], pipeline_run_id=None, dry_run=True)


class ClimateAggregationDBTests(unittest.TestCase):
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
            DistrictWeekClimate.district_id.like("UGA_TEST_CLIMATE%")
        ).delete(synchronize_session=False)
        DistrictWeekFeature.query.filter(
            DistrictWeekFeature.district_id.like("UGA_TEST_CLIMATE%")
        ).delete(synchronize_session=False)
        DistrictReference.query.filter(
            DistrictReference.district_id.like("UGA_TEST_CLIMATE%")
        ).delete(synchronize_session=False)
        PipelineRun.query.filter(
            PipelineRun.source_name.like("test climate aggregation%")
        ).delete(synchronize_session=False)
        for run in PipelineRun.query.filter_by(run_type="climate_aggregation").all():
            parameters = run.parameters or {}
            if str(parameters.get("end_date", "")).startswith("2098-"):
                db.session.delete(run)
        db.session.commit()

    def test_prepared_boundary_proof_uses_loaded_reference_frame(self):
        if DistrictReference.query.filter_by(active=True).count() != 146:
            self.skipTest("Full 146-district reference frame is not loaded.")
        summary = validate_prepared_boundary_for_climate()
        self.assertTrue(summary.ok)
        self.assertEqual(summary.feature_count, 146)
        self.assertEqual(summary.unique_district_id_count, 146)
        self.assertFalse(summary.extra_boundary_ids)

    def test_prepared_boundary_helper_is_defined_and_loads_146_features(self):
        if DistrictReference.query.filter_by(active=True).count() != 146:
            self.skipTest("Full 146-district reference frame is not loaded.")
        self.assertTrue(callable(_load_boundary_geodataframe))
        gdf = _load_boundary_geodataframe()
        self.assertEqual(type(gdf).__name__, "GeoDataFrame")
        self.assertEqual(len(gdf), 146)
        self.assertEqual(gdf["district_id"].nunique(), 146)
        self.assertEqual(gdf.crs.to_epsg(), 4326)
        self.assertEqual(set(gdf.geometry.geom_type), {"Polygon", "MultiPolygon"})

    def test_real_row_builder_exercises_prepared_boundary_loader(self):
        districts = DistrictReference.query.filter_by(active=True).all()
        if len(districts) != 146:
            self.skipTest("Full 146-district reference frame is not loaded.")
        week_start, week_end = iso_week_dates(2025, 1)
        rainfall = {
            (row.district_id, 2025, 1): Decimal("10.000000") for row in districts
        }
        temperature = {
            (row.district_id, 2025, 1): Decimal("24.0000") for row in districts
        }
        humidity = {
            (row.district_id, 2025, 1): Decimal("70.0000") for row in districts
        }

        def era5_values(**kwargs):
            values = temperature if kwargs["variable"] == "temperature" else humidity
            return values, {"raster_crs": "EPSG:4326"}, []

        with (
            patch(
                "app.services.climate_aggregation._aggregate_chirps_rainfall",
                return_value=(rainfall, {"raster_crs": ["EPSG:4326"]}, []),
            ),
            patch(
                "app.services.climate_aggregation._aggregate_era5_weekly",
                side_effect=era5_values,
            ),
        ):
            rows, proof, warnings = build_real_climate_week_rows(
                files=[],
                start_date=week_start,
                end_date=week_end,
                expected_weeks=[(2025, 1)],
            )

        self.assertEqual(len(rows), 146)
        self.assertEqual(proof["districts"], 146)
        self.assertTrue(proof["crs_verified"])
        self.assertEqual(warnings, [])

    def test_aggregate_dry_run_reports_complete_real_climate_files(self):
        if DistrictReference.query.filter_by(active=True).count() != 146:
            self.skipTest("Full 146-district reference frame is not loaded.")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "district_week_climate.csv"
            summary = aggregate_weekly_climate(
                start_date=date(2020, 1, 1),
                end_date=date(2020, 1, 31),
                dry_run=True,
                output_path=output_path,
            )
        self.assertEqual(summary.status, "dry-run")
        self.assertFalse(output_path.exists())
        self.assertEqual(summary.missing_pairs, [])
        self.assertGreaterEqual(len(summary.files_found), 3)

    def test_write_and_apply_updates_only_current_climate_fields(self):
        district_id = "UGA_TEST_CLIMATE_APPLY"
        db.session.add(
            DistrictReference(
                district_id=district_id,
                district_name="TEST CLIMATE APPLY",
                active=True,
            )
        )
        run = PipelineRun(
            run_type="climate_aggregation",
            status="success",
            source_name="test climate aggregation write",
        )
        db.session.add(run)
        week_start, week_end = iso_week_dates(2025, 1)
        db.session.add(
            DistrictWeekFeature(
                district_id=district_id,
                epi_year=2025,
                epi_week=1,
                week_start_date=week_start,
                week_end_date=week_end,
                rainfall_lag_4=None,
                temperature_lag_4=None,
                humidity_lag_4=None,
            )
        )
        db.session.commit()

        climate_row = ClimateWeekRow(
            district_id=district_id,
            district_name="TEST CLIMATE APPLY",
            epi_year=2025,
            epi_week=1,
            week_start_date=week_start,
            week_end_date=week_end,
            rainfall_current=Decimal("12.500000"),
            temperature_current=Decimal("24.1250"),
            humidity_current=Decimal("71.2500"),
            source_rainfall="CHIRPS",
            source_temperature="ERA5-Land",
            source_humidity="ERA5-Land",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            write_summary = write_processed_climate_rows(
                [climate_row],
                output_path=Path(tmpdir) / "district_week_climate.csv",
                pipeline_run_id=run.id,
            )
        self.assertEqual(write_summary.rows_inserted, 1)

        apply_summary = apply_processed_climate_to_features(
            start_year=2025,
            end_year=2025,
            pipeline_run_id=run.id,
        )
        self.assertEqual(apply_summary.rows_updated, 1)
        feature = DistrictWeekFeature.query.filter_by(district_id=district_id).one()
        self.assertEqual(feature.rainfall_current, Decimal("12.500000"))
        self.assertEqual(feature.temperature_current, Decimal("24.1250"))
        self.assertEqual(feature.humidity_current, Decimal("71.2500"))
        self.assertIsNone(feature.rainfall_lag_4)
        self.assertIsNone(feature.temperature_lag_4)
        self.assertIsNone(feature.humidity_lag_4)

    def test_real_pipeline_rolls_back_climate_and_feature_writes_on_failure(self):
        district_id = "UGA_TEST_CLIMATE_TRANSACTION"
        week_start, week_end = iso_week_dates(2098, 1)
        db.session.add(
            DistrictReference(
                district_id=district_id,
                district_name="TEST CLIMATE TRANSACTION",
                active=True,
            )
        )
        db.session.add(
            DistrictWeekFeature(
                district_id=district_id,
                epi_year=2098,
                epi_week=1,
                week_start_date=week_start,
                week_end_date=week_end,
                rainfall_current=Decimal("1.000000"),
                temperature_current=Decimal("2.0000"),
                humidity_current=Decimal("3.0000"),
            )
        )
        db.session.commit()
        climate_row = ClimateWeekRow(
            district_id=district_id,
            district_name="TEST CLIMATE TRANSACTION",
            epi_year=2098,
            epi_week=1,
            week_start_date=week_start,
            week_end_date=week_end,
            rainfall_current=Decimal("11.000000"),
            temperature_current=Decimal("22.0000"),
            humidity_current=Decimal("73.0000"),
            source_rainfall="CHIRPS",
            source_temperature="ERA5-Land",
            source_humidity="ERA5-Land",
        )
        registered_files = [
            RegisteredClimateFile(
                pipeline_run_id=index,
                source=source,
                variable=variable,
                path=Path(__file__),
                start_date=week_start,
                end_date=week_end,
                file_exists=True,
            )
            for index, (source, variable) in enumerate(
                (
                    ("CHIRPS", "rainfall"),
                    ("ERA5-Land", "temperature"),
                    ("ERA5-Land", "humidity"),
                ),
                start=1,
            )
        ]
        boundary = PreparedBoundaryProof(
            path=Path("data/boundaries/uganda_districts_prepared.geojson"),
            exists=True,
            feature_count=146,
            unique_district_id_count=146,
            reference_count=146,
            crs="EPSG:4326",
        )

        def fail_after_feature_mutation(**_kwargs):
            feature = DistrictWeekFeature.query.filter_by(
                district_id=district_id, epi_year=2098, epi_week=1
            ).one()
            feature.rainfall_current = Decimal("99.000000")
            raise ClimateAggregationError("forced apply failure")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "district_week_climate.csv"
            output_path.write_text("original\n", encoding="utf-8")
            with (
                patch(
                    "app.services.climate_aggregation.validate_prepared_boundary_for_climate",
                    return_value=boundary,
                ),
                patch(
                    "app.services.climate_aggregation.missing_optional_geospatial_dependencies",
                    return_value=[],
                ),
                patch(
                    "app.services.climate_aggregation.discover_registered_climate_files",
                    return_value=registered_files,
                ),
                patch(
                    "app.services.climate_aggregation.build_real_climate_week_rows",
                    return_value=([climate_row], {"crs_verified": True}, []),
                ),
                patch(
                    "app.services.climate_aggregation.apply_processed_climate_to_features",
                    side_effect=fail_after_feature_mutation,
                ),
            ):
                with self.assertRaisesRegex(ClimateAggregationError, "forced apply failure"):
                    aggregate_weekly_climate(
                        start_date=week_start,
                        end_date=week_end,
                        apply_to_features=True,
                        output_path=output_path,
                    )
            self.assertEqual(output_path.read_text(encoding="utf-8"), "original\n")

        self.assertIsNone(
            DistrictWeekClimate.query.filter_by(
                district_id=district_id, epi_year=2098, epi_week=1
            ).first()
        )
        feature = DistrictWeekFeature.query.filter_by(
            district_id=district_id, epi_year=2098, epi_week=1
        ).one()
        self.assertEqual(feature.rainfall_current, Decimal("1.000000"))
        failed_runs = [
            run
            for run in PipelineRun.query.filter_by(run_type="climate_aggregation").all()
            if (run.parameters or {}).get("start_date") == week_start.isoformat()
        ]
        self.assertEqual(len(failed_runs), 1)
        self.assertEqual(failed_runs[0].status, "failed")
        self.assertIn("forced apply failure", failed_runs[0].error_message)


if __name__ == "__main__":
    unittest.main()
