from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.models import PipelineRun
from app.services.climate_data import (
    ClimateDataError,
    boundary_dependency_status,
    download_era5,
    plan_chirps_download,
    plan_era5_download,
    register_climate_file,
    required_climate_date_range,
    validate_climate_file,
    validate_source_variable,
)


class ClimateDataPureTests(unittest.TestCase):
    def test_approved_source_variable_pairs(self):
        self.assertEqual(validate_source_variable("CHIRPS", "rainfall"), ("CHIRPS", "rainfall"))
        self.assertEqual(
            validate_source_variable("ERA5-Land", "temperature"),
            ("ERA5-Land", "temperature"),
        )
        self.assertEqual(
            validate_source_variable("ERA5", "humidity"),
            ("ERA5-Land", "humidity"),
        )

    def test_wind_speed_is_rejected(self):
        with self.assertRaises(ClimateDataError):
            validate_source_variable("ERA5-Land", "wind_speed")

    def test_missing_file_validation_fails(self):
        validation = validate_climate_file(
            source="CHIRPS",
            variable="rainfall",
            path=Path("data/climate/raw/chirps/missing.tif"),
            start_date="2025-01-01",
            end_date="2025-01-07",
        )
        self.assertFalse(validation.ok)
        self.assertIn("not found", validation.errors[0])

    def test_download_dry_run_does_not_require_api(self):
        with patch.dict(os.environ, {"CHIRPS_BASE_URL": ""}, clear=False):
            plan = plan_chirps_download(
                start_date="2025-01-01",
                end_date="2025-01-31",
                dry_run=True,
            )
        self.assertTrue(plan.dry_run)
        self.assertTrue(plan.configured)
        self.assertEqual(plan.chunks_planned, 31)
        self.assertIn("chirps-v2.0.2025.01.01.tif.gz", str(plan.target_path))

    def test_required_climate_range_includes_12_week_lookback(self):
        coverage = required_climate_date_range(
            analysis_start_year=2020,
            analysis_end_year=2025,
            lookback_weeks=12,
        )
        self.assertEqual(coverage.analysis_start_date.isoformat(), "2019-12-30")
        self.assertEqual(coverage.download_start_date.isoformat(), "2019-10-07")
        self.assertEqual(coverage.download_end_date.isoformat(), "2025-12-28")

    def test_era5_humidity_plans_temperature_and_dewpoint(self):
        with patch.dict(os.environ, {"ERA5_API_KEY": ""}, clear=False):
            plan = plan_era5_download(
                variable="humidity",
                start_date="2025-01-01",
                end_date="2025-02-28",
                dry_run=True,
            )
        self.assertEqual(plan.variable, "humidity")
        self.assertEqual(plan.chunks_planned, 2)
        self.assertEqual(
            plan.requested_variables,
            ["2m_temperature", "2m_dewpoint_temperature"],
        )
        self.assertIn("derived", plan.humidity_approach)

    def test_era5_real_download_requires_token(self):
        with patch.dict(os.environ, {"ERA5_API_KEY": ""}, clear=False):
            with self.assertRaises(ClimateDataError):
                download_era5(
                    variable="temperature",
                    start_date="2025-01-01",
                    end_date="2025-01-31",
                    dry_run=False,
                )

    def test_boundary_dependency_status_reports_paths(self):
        status = boundary_dependency_status()
        self.assertIsInstance(status.raw_boundary_exists, bool)
        self.assertIsInstance(status.prepared_boundary_exists, bool)
        self.assertTrue(status.message)


class ClimateDataDBTests(unittest.TestCase):
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
        self.created_pipeline_ids: list[int] = []

    def tearDown(self):
        if self.created_pipeline_ids:
            PipelineRun.query.filter(PipelineRun.id.in_(self.created_pipeline_ids)).delete(
                synchronize_session=False
            )
            db.session.commit()
        self.ctx.pop()

    def test_register_file_creates_safe_pipeline_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "climate"
            source_file = data_dir / "raw" / "chirps" / "test.tif"
            source_file.parent.mkdir(parents=True, exist_ok=True)
            source_file.write_bytes(b"tiny test raster placeholder")

            with patch.dict(os.environ, {"CLIMATE_DATA_DIR": str(data_dir)}, clear=False):
                summary = register_climate_file(
                    source="CHIRPS",
                    variable="rainfall",
                    path=source_file,
                    start_date="2025-01-01",
                    end_date="2025-01-07",
                )
            self.created_pipeline_ids.append(summary.pipeline_run_id)

            run = db.session.get(PipelineRun, summary.pipeline_run_id)
            self.assertEqual(run.run_type, "climate_ingestion")
            self.assertEqual(run.status, "success")
            self.assertEqual(run.parameters["operation"], "file_register")
            self.assertEqual(run.parameters["variable"], "rainfall")
            serialized = str(run.parameters) + str(run.manifest)
            self.assertNotIn("ERA5_API_KEY", serialized)
            self.assertTrue(summary.manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
