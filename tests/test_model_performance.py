from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.services.model_performance import (
    ModelPerformanceError,
    consolidate_model_performance,
    load_performance_records,
    select_count_models,
    select_surge_models,
)


class ModelPerformancePureTests(unittest.TestCase):
    def test_consolidation_and_selection_from_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _artifact_set(Path(tmpdir))
            summary = consolidate_model_performance(
                dry_run=True,
                artifact_paths=paths,
                require_dataset=False,
                require_forecasts=False,
                require_reference=False,
            )
        self.assertEqual(len(summary.records), 12)
        self.assertEqual(summary.selected_count_models[4]["model_name"], "Random Forest")
        self.assertEqual(summary.selected_surge_models[4]["model_name"], "Random Forest")
        self.assertEqual(summary.inserted_rows, 0)

    def test_regression_and_classification_ranking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = load_performance_records(_artifact_set(Path(tmpdir)))
        self.assertEqual(select_count_models(records)[8]["model_name"], "Random Forest")
        self.assertEqual(select_surge_models(records)[12]["model_name"], "Random Forest")

    def test_missing_artifact_reports_prompt_to_rerun(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _artifact_set(Path(tmpdir))
            paths["xgboost"] = Path(tmpdir) / "missing.json"
            with self.assertRaises(ModelPerformanceError) as ctx:
                consolidate_model_performance(
                    dry_run=True,
                    artifact_paths=paths,
                    require_dataset=False,
                    require_forecasts=False,
                    require_reference=False,
                )
        self.assertIn("Prompt 24", str(ctx.exception))

    def test_wind_speed_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _artifact_set(Path(tmpdir), baseline_features=["rainfall_lag_4", "wind_speed"])
            with self.assertRaises(ModelPerformanceError):
                consolidate_model_performance(
                    dry_run=True,
                    artifact_paths=paths,
                    require_dataset=False,
                    require_forecasts=False,
                    require_reference=False,
                )

    def test_negative_binomial_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _artifact_set(Path(tmpdir), baseline_name="Negative Binomial")
            with self.assertRaises(ModelPerformanceError):
                consolidate_model_performance(
                    dry_run=True,
                    artifact_paths=paths,
                    require_dataset=False,
                    require_forecasts=False,
                    require_reference=False,
                )

    def test_dry_run_is_idempotent_for_same_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = _artifact_set(Path(tmpdir))
            first = consolidate_model_performance(
                dry_run=True,
                artifact_paths=paths,
                require_dataset=False,
                require_forecasts=False,
                require_reference=False,
            ).to_dict()
            second = consolidate_model_performance(
                dry_run=True,
                artifact_paths=paths,
                require_dataset=False,
                require_forecasts=False,
                require_reference=False,
            ).to_dict()
        first.pop("generated_at_utc", None)
        second.pop("generated_at_utc", None)
        self.assertEqual(first, second)


def _artifact_set(
    folder: Path,
    *,
    baseline_features: list[str] | None = None,
    baseline_name: str = "Baseline/Naive",
) -> dict[str, Path]:
    specs = {
        "baseline": (baseline_name, "district_specific", "30", "0.20", baseline_features or ["rainfall_lag_4"]),
        "sarimax": ("SARIMAX", "district_specific", "20", "0.30", ["rainfall_lag_4"]),
        "random_forest": ("Random Forest", "national_pooled", "10", "0.80", ["rainfall_lag_4"]),
        "xgboost": ("XGBoost", "national_pooled", "15", "0.70", ["rainfall_lag_4"]),
    }
    paths: dict[str, Path] = {}
    for key, (name, scope, wape, f1, features) in specs.items():
        path = folder / f"{key}.json"
        payload = {
            "model_name": name,
            "model_scope": scope,
            "horizons": [4, 8, 12],
            "status": "success",
            "validation_method": "rolling_origin",
            "no_random_shuffle": True,
            "horizon_aware_gap": True,
            "feature_columns": features,
            "validation_cutoffs": [
                {"origin_week": "2025-01-06", "target_week": "2025-02-03"}
            ],
            "count_metrics_by_horizon": {
                str(horizon): {"n": "10", "mae": "1", "rmse": "2", "wape": wape}
                for horizon in (4, 8, 12)
            },
            "surge_metrics_by_horizon": {
                str(horizon): {"n": "10", "f1": f1, "sensitivity": f1, "specificity": "0.90", "auc": f1}
                for horizon in (4, 8, 12)
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[key] = path
    return paths


if __name__ == "__main__":
    unittest.main()
