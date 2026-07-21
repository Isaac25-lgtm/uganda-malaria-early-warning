from __future__ import annotations

import csv
import copy
import json
import math
import os
import tempfile
import unittest
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.models import DistrictReference, ForecastOutput, PipelineRun
from app.services.forecast_generation import (
    THRESHOLD_METHOD,
    MODEL_NAMES,
    generate_forecasts,
)
from app.services.models.pooled_tree_validation import (
    POOLED_FEATURES,
    POOLED_NUMERIC_FEATURES,
    _feature_frame,
)
from app.services.models.random_forest import RANDOM_FOREST_CONFIG
from app.services.models.sarimax import SARIMAX_EXOGENOUS_FEATURES
from app.services.models.xgboost_model import XGBOOST_CONFIG
from app.services.production_models import (
    baseline_production_forecast,
    fit_pooled_production_model,
    fit_sarimax_production_model,
)
from app.services.project_state import _production_forecast_is_complete


class ProductionModelPureTests(unittest.TestCase):
    def test_threshold_method_label_fits_existing_database_contract(self):
        self.assertLessEqual(len(THRESHOLD_METHOD), 120)

    def test_baseline_forecast_is_deterministic(self):
        rows = [
            {"week_start_date": "2024-01-22", "epi_week": "4", "confirmed_malaria_cases": "90"},
            {"week_start_date": "2025-12-15", "epi_week": "51", "confirmed_malaria_cases": "20"},
            {"week_start_date": "2025-12-22", "epi_week": "52", "confirmed_malaria_cases": "22"},
        ]
        first = baseline_production_forecast(
            district_rows=rows, issue_date=date(2025, 12, 22), target_epi_week=4
        )
        second = baseline_production_forecast(
            district_rows=rows, issue_date=date(2025, 12, 22), target_epi_week=4
        )
        self.assertEqual(first, second)
        self.assertEqual(first["predicted_cases"], 90.0)
        self.assertIsNone(first["prediction_interval_lower"])

    def test_final_pooled_models_reuse_validated_parameters_and_serialize_pipeline(self):
        rows = _fixture_rows()
        issue_date = date.fromisoformat(rows[-1]["week_start_date"])
        origin_rows = [
            row for row in rows if row["week_start_date"] == issue_date.isoformat()
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for config in (RANDOM_FOREST_CONFIG, XGBOOST_CONFIG):
                with self.subTest(model=config.model_name):
                    card = root / f"{config.model_key}_card.json"
                    parameters = dict(config.parameter_candidates[0])
                    classifier_parameters = dict(
                        config.classifier_parameter_candidates[0]
                    )
                    card.write_text(
                        json.dumps(
                            {
                                "status": "success",
                                "model_name": config.model_name,
                                "model_scope": "national_pooled",
                                "predictors_used": list(POOLED_FEATURES),
                                "selected_parameters_by_horizon": {"4": parameters},
                                "selected_classifier_parameters_by_horizon": {
                                    "4": classifier_parameters
                                },
                                "selected_probability_cutoff_by_horizon": {"4": 0.5},
                                "operational_surge_method_by_horizon": {
                                    "4": "dedicated_classifier"
                                },
                            }
                        ),
                        encoding="utf-8",
                    )
                    artifact = root / f"{config.model_key}_h4.joblib"
                    job = fit_pooled_production_model(
                        config=config,
                        rows=rows,
                        origin_rows=origin_rows,
                        issue_date=issue_date,
                        horizon=4,
                        model_card_path=card,
                        staged_artifact_path=artifact,
                        final_artifact_path=artifact,
                        dataset_signature={"sha256": "fixture"},
                    )
                    self.assertEqual(job["status"], "success")
                    self.assertEqual(job["selected_parameters"], parameters)
                    self.assertEqual(
                        job["selected_classifier_parameters"], classifier_parameters
                    )
                    self.assertEqual(job["missing_outcome_exclusions"], 1)
                    self.assertEqual(len(job["forecasts"]), 2)
                    self.assertTrue(all(item["prediction_interval_lower"] is None for item in job["forecasts"]))
                    self.assertTrue(
                        all(item["surge_probability"] is not None for item in job["forecasts"])
                    )
                    self.assertLessEqual(job["latest_training_week"], issue_date.isoformat())
                    import joblib

                    loaded = joblib.load(artifact)
                    predictions = loaded.predict(_feature_frame(origin_rows))
                    self.assertEqual(len(predictions), 2)
                    self.assertTrue(loaded.production_metadata_["training_only_preprocessing"])
                    self.assertFalse(loaded.production_metadata_["validation_estimators_reused"])
                    self.assertEqual(
                        set(
                            loaded.named_steps["preprocess"]
                            .named_transformers_["district"]
                            .named_steps["onehot"]
                            .categories_[0]
                        ),
                        {"UGA_A", "UGA_B"},
                    )

    def test_sarimax_final_fit_preserves_fallback_chain_and_clipping(self):
        rows = _fixture_rows(districts=(("UGA_A", "A", 0),))
        origin = rows[-1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "checkpoint.json"
            checkpoint.write_text(
                json.dumps(
                    {
                        "job": {
                            "district_id": "UGA_A",
                            "horizon": 4,
                            "status": "success",
                            "selected_order": [1, 0, 0],
                            "selected_seasonal_order": [1, 0, 0, 52],
                        }
                    }
                ),
                encoding="utf-8",
            )
            failed = {"status": "failed", "converged": False, "failure_reason": "fit failed"}
            successful = {
                "status": "success",
                "converged": True,
                "forecast": -5.0,
                "warnings": [],
                "prediction_interval_lower": 0.0,
                "prediction_interval_upper": 4.0,
                "fitted_state": {"parameter_values": [1.0]},
            }
            artifact = root / "sarimax.json"
            with patch(
                "app.services.models.sarimax._fit_sarimax",
                side_effect=[failed, failed, successful],
            ):
                job = fit_sarimax_production_model(
                    district_id="UGA_A",
                    district_rows=rows,
                    origin_row=origin,
                    horizon=4,
                    checkpoint_path=checkpoint,
                    staged_artifact_path=artifact,
                    final_artifact_path=artifact,
                    dataset_signature={"sha256": "fixture"},
                )
            self.assertEqual(job["status"], "fallback")
            self.assertEqual(job["fallback_information"], "regression_only")
            self.assertEqual(job["forecasts"][0]["predicted_cases"], 0.0)
            self.assertTrue(job["forecasts"][0]["forecast_clipped_nonnegative"])
            self.assertEqual(job["forecasts"][0]["prediction_interval_upper"], 4.0)
            state = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertTrue(state["origin_exogenous_policy"].startswith("Issue-week"))

    def test_project_state_requires_all_1752_issue_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            manifest = root / "manifest.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "published": True,
                        "expected_forecast_rows": 1752,
                        "final_issue_row_count": 1752,
                        "issue_year": 2025,
                        "issue_week": 52,
                    }
                ),
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "issue_epi_year": 2025,
                        "issue_epi_week": 52,
                        "models": list(MODEL_NAMES.values()),
                        "horizons": [4, 8, 12],
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                SimpleNamespace(
                    model_name=model,
                    horizon_weeks=horizon,
                    district_id=f"UGA_{district:03d}",
                    predicted_cases=1,
                )
                for model in MODEL_NAMES.values()
                for horizon in (4, 8, 12)
                for district in range(146)
            ]
            self.assertTrue(
                _production_forecast_is_complete(
                    summary_path=summary,
                    manifest_path=manifest,
                    forecast_rows=rows,
                )
            )
            self.assertFalse(
                _production_forecast_is_complete(
                    summary_path=summary,
                    manifest_path=manifest,
                    forecast_rows=rows[:-1],
                )
            )


class ProductionForecastDBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL is required for production forecast tests.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()

    def test_full_dry_run_is_ready_and_writes_nothing(self):
        with self.app.app_context(), tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = generate_forecasts(
                latest=True,
                dry_run=True,
                summary_path=root / "summary.json",
                diagnostics_path=root / "diagnostics.json",
                production_dir=root / "production",
            )
            self.assertEqual(summary.status, "dry-run")
            self.assertTrue(summary.production_fitting_ready)
            self.assertEqual(summary.expected_forecast_rows, 1752)
            self.assertEqual(summary.expected_model_fits["SARIMAX"], 438)
            self.assertEqual(summary.target_weeks[4]["target_week_start_date"], "2026-01-19")
            self.assertFalse((root / "summary.json").exists())
            self.assertFalse((root / "production").exists())

    def test_transactional_idempotent_baseline_publication_and_schema_compatibility(self):
        with self.app.app_context(), tempfile.TemporaryDirectory() as tmpdir:
            district_id = DistrictReference.query.filter_by(active=True).order_by(DistrictReference.district_id).first().district_id
            root = Path(tmpdir)
            production_query = ForecastOutput.query.filter_by(
                district_id=district_id,
                model_name="Baseline/Naive",
                forecast_origin_epi_year=2025,
                forecast_origin_epi_week=52,
                horizon_weeks=4,
            )
            original = _database_row_snapshot(production_query.one_or_none())
            db.session.rollback()
            try:
                with patch.object(db.session, "commit", side_effect=db.session.flush):
                    existing = production_query.one_or_none()
                    if existing is not None:
                        db.session.delete(existing)
                        db.session.flush()
                    kwargs = {
                        "latest": True,
                        "models": ("baseline",),
                        "horizons": (4,),
                        "district_ids": (district_id,),
                        "summary_path": root / "summary.json",
                        "diagnostics_path": root / "diagnostics.json",
                        "production_dir": root / "production",
                    }
                    first = generate_forecasts(**kwargs)
                    self.assertEqual(first.status, "success")
                    self.assertEqual((first.inserted_rows, first.updated_rows), (1, 0))
                    output = production_query.one()
                    self.assertIsNone(output.prediction_interval_lower)
                    self.assertIsNone(output.prediction_interval_upper)
                    self.assertIsNotNone(output.threshold_used)
                    self.assertIn(output.risk_level, {"Low", "Moderate", "High", "Very High"})

                    second = generate_forecasts(**kwargs)
                    self.assertEqual((second.inserted_rows, second.updated_rows), (0, 1))
                    self.assertEqual(second.final_issue_row_count, 1)
                    self.assertEqual(production_query.count(), 1)
                    manifest = json.loads(
                        (root / "production" / "production_model_manifest.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(manifest["status"], "success")
                    self.assertTrue(manifest["leakage_controls"]["training_through_issue_week_only"])
                    self.assertFalse(manifest["leakage_controls"]["future_climate_rows_read"])
                    entry = manifest["entries"][0]
                    self.assertEqual(entry["model_name"], "Baseline/Naive")
                    self.assertIn("dataset_signature", entry)
                    diagnostics = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
                    self.assertEqual(diagnostics["forecast_rows"][0]["model_status"], "success")
                    self.assertIn("predicted_surge_score", diagnostics["forecast_rows"][0])
            finally:
                db.session.rollback()
            self.assertEqual(_database_row_snapshot(production_query.one_or_none()), original)

    def test_publish_failure_rolls_back_forecast_row_and_marks_run_failed(self):
        with self.app.app_context(), tempfile.TemporaryDirectory() as tmpdir:
            district_id = DistrictReference.query.filter_by(active=True).order_by(DistrictReference.district_id).first().district_id
            root = Path(tmpdir)
            before_ids = {
                row.id
                for row in PipelineRun.query.filter_by(run_type="forecast_generation").all()
            }
            production_query = ForecastOutput.query.filter_by(
                district_id=district_id,
                model_name="Baseline/Naive",
                forecast_origin_epi_year=2025,
                forecast_origin_epi_week=52,
                horizon_weeks=8,
            )
            original = _database_row_snapshot(production_query.one_or_none())

            @contextmanager
            def interrupted(_staged):
                raise RuntimeError("interrupted publication")
                yield

            try:
                with patch(
                    "app.services.forecast_generation.publish_staged_outputs",
                    interrupted,
                ), self.assertRaisesRegex(RuntimeError, "interrupted publication"):
                    generate_forecasts(
                        latest=True,
                        models=("baseline",),
                        horizons=(8,),
                        district_ids=(district_id,),
                        summary_path=root / "summary.json",
                        diagnostics_path=root / "diagnostics.json",
                        production_dir=root / "production",
                    )
                self.assertEqual(_database_row_snapshot(production_query.one_or_none()), original)
                failed_runs = PipelineRun.query.filter(
                    PipelineRun.run_type == "forecast_generation",
                    PipelineRun.id.notin_(before_ids),
                ).all()
                self.assertEqual(len(failed_runs), 1)
                self.assertEqual(failed_runs[0].status, "failed")
            finally:
                PipelineRun.query.filter(
                    PipelineRun.run_type == "forecast_generation",
                    PipelineRun.id.notin_(before_ids),
                ).delete(synchronize_session=False)
                db.session.commit()

    def test_partial_model_failure_does_not_publish(self):
        with self.app.app_context(), tempfile.TemporaryDirectory() as tmpdir:
            district_ids = tuple(
                row.district_id
                for row in DistrictReference.query.filter_by(active=True)
                .order_by(DistrictReference.district_id)
                .limit(2)
                .all()
            )
            root = Path(tmpdir)

            def fail_second(**kwargs):
                if kwargs["district_rows"][0]["district_id"] == district_ids[1]:
                    raise RuntimeError("forced forecast failure")
                return baseline_production_forecast(**kwargs)

            with patch(
                "app.services.forecast_generation.baseline_production_forecast",
                side_effect=fail_second,
            ):
                summary = generate_forecasts(
                    latest=True,
                    models=("baseline",),
                    horizons=(4,),
                    district_ids=district_ids,
                    summary_path=root / "summary.json",
                    diagnostics_path=root / "diagnostics.json",
                    production_dir=root / "production",
                    record_pipeline_run=False,
                    publish_database=False,
                )
            self.assertEqual(summary.status, "partial_success")
            self.assertEqual(summary.generated_forecast_rows, 1)
            self.assertFalse(summary.published)
            self.assertFalse((root / "summary.json").exists())


def _database_row_snapshot(row: ForecastOutput | None) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        column.name: copy.deepcopy(getattr(row, column.name))
        for column in ForecastOutput.__table__.columns
    }


def _fixture_rows(
    *, districts=(("UGA_A", "A", 0), ("UGA_B", "B", 7)), weeks: int = 64
) -> list[dict[str, str]]:
    start = date(2024, 1, 1)
    series = {
        district_id: [30 + offset + index // 8 + index % 5 for index in range(weeks)]
        for district_id, _, offset in districts
    }
    output: list[dict[str, str]] = []
    for district_id, district_name, offset in districts:
        values = series[district_id]
        for index, cases in enumerate(values):
            week = start + timedelta(weeks=index)
            row = {
                "district_id": district_id,
                "district_name": district_name,
                "epi_year": str(week.isocalendar().year),
                "epi_week": str(week.isocalendar().week),
                "week_start_date": week.isoformat(),
                "week_end_date": (week + timedelta(days=6)).isoformat(),
                "continuous_week_index": str(week.toordinal() // 7),
                "confirmed_malaria_cases": str(cases),
            }
            for lag in (1, 2, 4, 8, 12):
                row[f"cases_lag_{lag}"] = str(values[index - lag]) if index >= lag else ""
                row[f"cases_lag_{lag}_interpolated"] = "0"
            for window in (4, 8, 12):
                history = values[max(0, index - window):index]
                row[f"cases_rolling_mean_{window}"] = str(sum(history) / len(history)) if history else ""
                row[f"cases_rolling_interpolated_count_{window}"] = "0"
            climate_columns = [
                name
                for name in POOLED_NUMERIC_FEATURES
                if name.startswith(("rainfall_", "temperature_", "humidity_"))
            ]
            for climate_index, column in enumerate(climate_columns, start=1):
                row[column] = str(20 + offset + climate_index + math.sin(index / 5))
            for column in SARIMAX_EXOGENOUS_FEATURES:
                row.setdefault(column, "20")
            target_index = index + 4
            row["target_week_start_date_h4"] = (week + timedelta(weeks=4)).isoformat()
            row["target_epi_week_h4"] = str(
                (week + timedelta(weeks=4)).isocalendar().week
            )
            row["target_cases_h4"] = str(values[target_index]) if target_index < weeks else ""
            if district_id == "UGA_A" and index == 20:
                row["target_cases_h4"] = ""
            output.append(row)
    output.sort(key=lambda row: (row["week_start_date"], row["district_id"]))
    return output


if __name__ == "__main__":
    unittest.main()
