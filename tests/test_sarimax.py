from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from app.services.model_validation import build_rolling_origin_folds
from app.services.models.sarimax import (
    DISTRICT_METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    SARIMAX_EXOGENOUS_FEATURES,
    _forecast_fold,
    _publish_staged_outputs,
    run_sarimax_validation,
)
from app.services.modelling_dataset import read_modelling_dataset


class SARIMAXValidationTests(unittest.TestCase):
    def test_exogenous_contract_contains_only_approved_lagged_climate(self):
        self.assertEqual(len(SARIMAX_EXOGENOUS_FEATURES), 9)
        self.assertTrue(all("_lag_" in column for column in SARIMAX_EXOGENOUS_FEATURES))
        self.assertFalse(any("current" in column for column in SARIMAX_EXOGENOUS_FEATURES))
        self.assertFalse(any("target" in column for column in SARIMAX_EXOGENOUS_FEATURES))
        self.assertFalse(any("wind" in column for column in SARIMAX_EXOGENOUS_FEATURES))

    def test_completed_dataset_dry_run_is_not_stale_planned_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "fixture.csv"
            _write_fixture(dataset)
            summary = run_sarimax_validation(
                dataset_path=dataset,
                horizons=(4,),
                dry_run=True,
                max_folds=1,
            )
        self.assertEqual(summary.status, "dry-run")
        self.assertEqual(summary.job_counts, {"planned": 1})
        self.assertFalse(any("gated" in warning.lower() for warning in summary.warnings))

    def test_real_sarimax_fit_uses_baseline_folds_and_writes_required_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "fixture.csv"
            _write_fixture(dataset)
            predictions = root / "predictions.csv"
            metrics = root / "metrics.csv"
            summary_path = root / "summary.json"
            diagnostics = root / "diagnostics.json"
            model_card = root / "model_card.json"
            summary = run_sarimax_validation(
                dataset_path=dataset,
                horizons=(4,),
                max_folds=1,
                artifact_path=summary_path,
                predictions_path=predictions,
                district_metrics_path=metrics,
                diagnostics_path=diagnostics,
                model_card_path=model_card,
                checkpoint_dir=root / "checkpoints",
                record_pipeline_run=False,
            )

            rows = read_modelling_dataset(dataset)
            baseline_folds = build_rolling_origin_folds(
                rows, horizon=4, max_folds=1
            )
            self.assertEqual(summary.status, "success")
            self.assertTrue(summary.published)
            self.assertEqual(
                summary.validation_cutoffs[0]["origin_week"],
                baseline_folds[0].origin_date.isoformat(),
            )
            self.assertEqual(
                summary.validation_cutoffs[0]["target_week"],
                baseline_folds[0].target_date.isoformat(),
            )
            self.assertEqual(summary.count_metrics_by_horizon[4]["n"], 1)
            self.assertEqual(summary.surge_metrics_by_horizon[4]["n"], 1)
            self.assertEqual(
                set(summary.count_metrics_by_horizon[4]),
                {"n", "mae", "rmse", "mape", "wape", "r_squared"},
            )
            self.assertEqual(
                set(summary.surge_metrics_by_horizon[4]),
                {
                    "n",
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                    "sensitivity",
                    "specificity",
                    "precision",
                    "f1",
                    "f2",
                    "accuracy",
                    "auc",
                    "pr_auc",
                    "brier_score",
                },
            )
            with predictions.open(newline="", encoding="utf-8") as handle:
                prediction_reader = csv.DictReader(handle)
                prediction_rows = list(prediction_reader)
                self.assertEqual(tuple(prediction_reader.fieldnames), PREDICTION_COLUMNS)
            self.assertEqual(len(prediction_rows), 1)
            self.assertEqual(
                json.loads(prediction_rows[0]["exogenous_features_used"]),
                list(SARIMAX_EXOGENOUS_FEATURES),
            )
            with metrics.open(newline="", encoding="utf-8") as handle:
                metric_reader = csv.DictReader(handle)
                metric_rows = list(metric_reader)
                self.assertEqual(tuple(metric_reader.fieldnames), DISTRICT_METRIC_COLUMNS)
            self.assertEqual(len(metric_rows), 1)
            self.assertIn("converged", diagnostics.read_text(encoding="utf-8"))
            card = json.loads(model_card.read_text(encoding="utf-8"))
            self.assertIn("weekly_seasonality", card)
            self.assertIn("fallbacks", card)

            resumed = run_sarimax_validation(
                dataset_path=dataset,
                horizons=(4,),
                max_folds=1,
                artifact_path=summary_path,
                predictions_path=predictions,
                district_metrics_path=metrics,
                diagnostics_path=diagnostics,
                model_card_path=model_card,
                checkpoint_dir=root / "checkpoints",
                record_pipeline_run=False,
            )
            self.assertEqual(resumed.checkpoints_reused, 1)
            self.assertEqual(resumed.status, "success")

    def test_nonseasonal_fallback_and_total_failure_are_recorded(self):
        training = _fixture_rows()[:60]
        validation = _fixture_rows()[59]
        failed = {
            "status": "failed",
            "converged": False,
            "failure_reason": "seasonal instability",
        }
        successful = {
            "status": "success",
            "converged": True,
            "forecast": 25.0,
            "warnings": ["controlled warning"],
        }
        with patch(
            "app.services.models.sarimax._fit_sarimax",
            side_effect=[failed, successful],
        ):
            result = _forecast_fold(
                training_rows=training,
                validation_row=validation,
                horizon=4,
                selected_order=(1, 0, 0),
                selected_seasonal_order=(1, 0, 0, 52),
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["fallback_status"], "nonseasonal")
        self.assertIn("controlled warning", result["warning_or_failure_reason"])

        with patch(
            "app.services.models.sarimax._fit_sarimax",
            return_value=failed,
        ):
            result = _forecast_fold(
                training_rows=training,
                validation_row=validation,
                horizon=4,
                selected_order=(1, 0, 0),
                selected_seasonal_order=(1, 0, 0, 52),
            )
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["converged"])
        self.assertIn("seasonal instability", result["warning_or_failure_reason"])

    def test_publication_failure_restores_all_previous_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.json"
            second = root / "second.csv"
            first.write_text("old first", encoding="utf-8")
            second.write_text("old second", encoding="utf-8")
            staged_first = root / "new-first.tmp"
            staged_second = root / "new-second.tmp"
            staged_first.write_text("new first", encoding="utf-8")
            staged_second.write_text("new second", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with _publish_staged_outputs(
                    {staged_first: first, staged_second: second}
                ):
                    raise RuntimeError("interrupted publication")

            self.assertEqual(first.read_text(encoding="utf-8"), "old first")
            self.assertEqual(second.read_text(encoding="utf-8"), "old second")


def _write_fixture(path: Path) -> None:
    rows = _fixture_rows()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture_rows() -> list[dict[str, str]]:
    start = date(2023, 1, 2)
    cases = [20 + index // 8 + (index % 13) for index in range(72)]
    rows: list[dict[str, str]] = []
    for index, value in enumerate(cases):
        week = start + timedelta(weeks=index)
        target_index = index + 4
        row = {
            "district_id": "UGA_TEST_SARIMAX",
            "district_name": "TEST SARIMAX",
            "epi_year": str(week.isocalendar().year),
            "epi_week": str(week.isocalendar().week),
            "week_start_date": week.isoformat(),
            "week_end_date": (week + timedelta(days=6)).isoformat(),
            "continuous_week_index": str(week.toordinal() // 7),
            "confirmed_malaria_cases": str(value),
            "target_cases_h4": (
                str(cases[target_index]) if target_index < len(cases) else ""
            ),
            "target_epi_week_h4": (
                str((week + timedelta(weeks=4)).isocalendar().week)
                if target_index < len(cases)
                else ""
            ),
            "target_week_start_date_h4": (
                (week + timedelta(weeks=4)).isoformat()
                if target_index < len(cases)
                else ""
            ),
        }
        for feature_index, column in enumerate(SARIMAX_EXOGENOUS_FEATURES, start=1):
            row[column] = str(
                20
                + feature_index
                + math.sin(index / (feature_index + 1)) * (feature_index + 0.5)
                + ((index * feature_index) % (7 + feature_index)) / 10
            )
        rows.append(row)
    return rows


if __name__ == "__main__":
    unittest.main()
