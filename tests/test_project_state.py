from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

from dotenv import load_dotenv

from app import create_app
from app.services.project_state import (
    _validation_artifact_is_complete,
    build_project_state_summary,
)


class ProjectStateRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL is required for project state tests.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.ctx = cls.app.app_context()
        cls.ctx.push()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.pop()

    def test_recovery_summary_reports_completed_dashboard(self):
        summary = build_project_state_summary(write=False)
        self.assertTrue(summary["safe_to_proceed_to_prompt_29"])
        self.assertEqual(
            summary["recommended_next_action"],
            "Prompt 29 is complete; no remaining dashboard action is required.",
        )
        self.assertEqual(summary["components"]["reference_and_boundaries"]["status"], "complete")
        self.assertEqual(summary["components"]["dhis2_malaria_panel"]["status"], "complete")
        self.assertEqual(summary["components"]["reporting_completeness"]["status"], "complete")
        self.assertEqual(summary["components"]["climate_ingestion"]["status"], "complete")
        self.assertEqual(summary["components"]["climate_aggregation"]["status"], "complete")
        self.assertEqual(summary["components"]["climate_lags"]["status"], "complete")
        self.assertEqual(summary["components"]["modelling_dataset"]["status"], "complete")
        self.assertEqual(summary["components"]["models_and_forecasts"]["status"], "complete")
        self.assertEqual(summary["components"]["visible_dashboard"]["status"], "complete")

    def test_prompt29_gate_requires_real_downstream_outputs(self):
        summary = build_project_state_summary(write=False)
        self.assertTrue(summary["gating_logic"]["prompt29_ready"])
        self.assertTrue(summary["gating_logic"]["prompt29_complete"])
        self.assertTrue(summary["gating_logic"]["prompt22_requires_climate_lags_and_proof"])
        self.assertTrue(summary["gating_logic"]["prompt28_api_reports_unavailable_when_downstream_missing"])
        self.assertFalse(summary["secrets"]["credential_values_reported"])

    def test_planned_sarimax_artifact_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sarimax.json"
            path.write_text(
                json.dumps(
                    {
                        "model_name": "SARIMAX",
                        "status": "planned",
                        "count_metrics_by_horizon": {},
                        "surge_metrics_by_horizon": {},
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(_validation_artifact_is_complete("sarimax", path))

    def test_sarimax_completion_requires_metrics_and_companion_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = {
                "predictions_csv": root / "predictions.csv",
                "district_horizon_metrics_csv": root / "district_metrics.csv",
                "fit_diagnostics_json": root / "diagnostics.json",
                "model_card_json": root / "model_card.json",
            }
            for output in paths.values():
                output.write_text("proof", encoding="utf-8")
            count = {
                "n": 10,
                "mae": "1",
                "rmse": "2",
                "mape": "3",
                "wape": "4",
                "r_squared": "0.5",
            }
            surge = {
                "n": 10,
                "tp": 2,
                "tn": 6,
                "fp": 1,
                "fn": 1,
                "sensitivity": "0.66",
                "specificity": "0.85",
                "precision": "0.66",
                "f1": "0.66",
                "accuracy": "0.80",
                "auc": "0.75",
            }
            artifact = root / "sarimax.json"
            artifact.write_text(
                json.dumps(
                    {
                        "model_name": "SARIMAX",
                        "status": "success",
                        "count_metrics_by_horizon": {
                            str(horizon): count for horizon in (4, 8, 12)
                        },
                        "surge_metrics_by_horizon": {
                            str(horizon): surge for horizon in (4, 8, 12)
                        },
                        "artifact_paths": {
                            name: str(path) for name, path in paths.items()
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_validation_artifact_is_complete("sarimax", artifact))
            paths["fit_diagnostics_json"].unlink()
            self.assertFalse(_validation_artifact_is_complete("sarimax", artifact))

    def test_pooled_models_require_real_published_outputs_and_feature_importance(self):
        count = {
            "n": 10,
            "mae": "1",
            "rmse": "2",
            "mape": "3",
            "wape": "4",
            "r_squared": "0.5",
        }
        surge = {
            "n": 10,
            "tp": 2,
            "tn": 6,
            "fp": 1,
            "fn": 1,
            "sensitivity": "0.66",
            "specificity": "0.85",
            "precision": "0.66",
            "f1": "0.66",
            "accuracy": "0.80",
            "auc": "0.75",
        }
        for model_key, model_name in (("random_forest", "Random Forest"), ("xgboost", "XGBoost")):
            with self.subTest(model=model_name), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                paths = {
                    "predictions_csv": root / "predictions.csv",
                    "district_horizon_metrics_csv": root / "district_metrics.csv",
                    "fit_diagnostics_json": root / "diagnostics.json",
                    "model_card_json": root / "model_card.json",
                    "feature_importance_csv": root / "importance.csv",
                }
                for output in paths.values():
                    output.write_text("proof", encoding="utf-8")
                artifact = root / "summary.json"
                payload = {
                    "model_name": model_name,
                    "model_scope": "national_pooled",
                    "status": "success",
                    "published": True,
                    "job_counts": {"success": 36, "total": 36},
                    "count_metrics_by_horizon": {
                        str(horizon): count for horizon in (4, 8, 12)
                    },
                    "surge_metrics_by_horizon": {
                        str(horizon): surge for horizon in (4, 8, 12)
                    },
                    "artifact_paths": {name: str(path) for name, path in paths.items()},
                }
                artifact.write_text(json.dumps(payload), encoding="utf-8")
                self.assertTrue(_validation_artifact_is_complete(model_key, artifact))
                payload["status"] = "planned"
                artifact.write_text(json.dumps(payload), encoding="utf-8")
                self.assertFalse(_validation_artifact_is_complete(model_key, artifact))
                payload["status"] = "success"
                artifact.write_text(json.dumps(payload), encoding="utf-8")
                paths["feature_importance_csv"].unlink()
                self.assertFalse(_validation_artifact_is_complete(model_key, artifact))


if __name__ == "__main__":
    unittest.main()
