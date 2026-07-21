from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from app.services.model_features import ModelFeatureError, TrainingOnlyImputer, build_feature_spec
from app.services.model_validation import (
    PredictionRecord,
    build_rolling_origin_folds,
    classification_metrics,
    district_threshold,
    evaluate_predictions,
)
from app.services.models.baseline import _baseline_prediction
from app.services.models.random_forest import run_random_forest_validation
from app.services.models.pooled_tree_validation import POOLED_FEATURES
from app.services.models.sarimax import SARIMAX_EXOGENOUS_FEATURES, run_sarimax_validation
from app.services.models.xgboost_model import run_xgboost_validation


def mrow(district: str, start: date, cases: int):
    return {
        "district_id": district,
        "week_start_date": start.isoformat(),
        "confirmed_malaria_cases": str(cases),
        "target_week_start_date_h4": (start + timedelta(weeks=4)).isoformat(),
        "target_cases_h4": str(cases + 4),
    }


class ModelValidationPureTests(unittest.TestCase):
    def test_feature_builder_excludes_targets_and_rejects_wind(self):
        spec = build_feature_spec(["district_id", "epi_week", "cases_lag_1", "target_cases_h4"])
        self.assertIn("cases_lag_1", spec.feature_columns)
        self.assertNotIn("target_cases_h4", spec.feature_columns)
        with self.assertRaises(ModelFeatureError):
            build_feature_spec(["wind_speed"])

    def test_rolling_origin_folds_are_ordered_and_gap_aware(self):
        rows = [mrow("A", date(2025, 1, 6) + timedelta(weeks=index), index) for index in range(70)]
        folds = build_rolling_origin_folds(rows, horizon=4, min_train_weeks=52, max_folds=3)
        self.assertEqual(len(folds), 3)
        for fold in folds:
            self.assertEqual((fold.target_date - fold.origin_date).days, 28)
            self.assertTrue(all(date.fromisoformat(row["week_start_date"]) <= fold.origin_date for row in fold.train_rows))
            self.assertTrue(
                all(
                    date.fromisoformat(row["target_week_start_date_h4"]) <= fold.origin_date
                    for row in fold.trainable_rows
                )
            )
            self.assertLess(len(fold.trainable_rows), len(fold.train_rows))

    def test_trainable_rows_exclude_recent_rows_with_unobserved_targets(self):
        rows = [mrow("A", date(2025, 1, 6) + timedelta(weeks=index), index) for index in range(60)]
        fold = build_rolling_origin_folds(rows, horizon=4, min_train_weeks=52, max_folds=1)[0]
        excluded_recent = [
            row
            for row in fold.train_rows
            if date.fromisoformat(row["target_week_start_date_h4"]) > fold.origin_date
        ]
        self.assertEqual(len(excluded_recent), 4)
        self.assertFalse(any(row in fold.trainable_rows for row in excluded_recent))

    def test_threshold_uses_training_rows_only(self):
        train = [
            {"district_id": "A", "confirmed_malaria_cases": value}
            for value in [1, 2, 3, 100]
        ]
        self.assertEqual(district_threshold(train, "A"), 3)

    def test_seasonal_threshold_excludes_2022_from_reference_only(self):
        train = []
        for year in (2020, 2021):
            for week in (8, 9, 10, 11, 12):
                train.append(
                    {
                        "district_id": "A",
                        "epi_year": year,
                        "epi_week": week,
                        "confirmed_malaria_cases": 100 + week,
                    }
                )
        for week in (8, 9, 10, 11, 12):
            train.append(
                {
                    "district_id": "A",
                    "epi_year": 2022,
                    "epi_week": week,
                    "confirmed_malaria_cases": 10000,
                }
            )
        threshold = district_threshold(train, "A", target_epi_week=10)
        self.assertLess(threshold, Decimal("1000"))
        self.assertIn(2022, {row["epi_year"] for row in train})

    def test_evenly_spaced_folds_cover_multiple_years(self):
        rows = [
            mrow("A", date(2020, 1, 6) + timedelta(weeks=index), index)
            for index in range(300)
        ]
        folds = build_rolling_origin_folds(
            rows, horizon=4, min_train_weeks=52, max_folds=12
        )
        self.assertEqual(len(folds), 12)
        self.assertGreaterEqual(len({fold.origin_date.year for fold in folds}), 4)
        self.assertEqual(folds, sorted(folds, key=lambda item: item.origin_date))

    def test_imputer_fits_training_rows_only(self):
        imputer = TrainingOnlyImputer(["x"]).fit([{"x": 1}, {"x": 3}])
        transformed = imputer.transform([{"x": ""}])
        self.assertEqual(imputer.fit_row_count, 2)
        self.assertEqual(transformed[0]["x"], "2")
        self.assertEqual(transformed[0]["x_was_missing"], "1")

    def test_metrics_handle_zero_actuals(self):
        metrics = classification_metrics([True, False], [True, True], [None, None])
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)

    def test_count_metrics_handle_all_zero_actuals(self):
        record = PredictionRecord(
            district_id="A",
            horizon=4,
            origin_date=date(2025, 1, 6),
            target_date=date(2025, 2, 3),
            actual=Decimal("0"),
            predicted=Decimal("2"),
            threshold=Decimal("1"),
            actual_surge=False,
            predicted_surge=True,
            surge_score=Decimal("2"),
        )
        count_metrics, _ = evaluate_predictions([record])
        self.assertIsNone(count_metrics["mape"])
        self.assertIsNone(count_metrics["wape"])

    def test_baseline_prediction_prefers_historical_same_epi_week(self):
        train_rows = [
            {"week_start_date": "2024-05-13", "epi_week": "20", "confirmed_malaria_cases": "90"},
            {"week_start_date": "2025-04-28", "epi_week": "18", "confirmed_malaria_cases": "10"},
            {"week_start_date": "2025-05-05", "epi_week": "19", "confirmed_malaria_cases": "11"},
        ]
        predicted = _baseline_prediction(
            train_rows,
            origin_date=date(2025, 5, 5),
            target_epi_week=20,
        )
        self.assertEqual(predicted, Decimal("90"))

    def test_model_runner_dependency_gates_do_not_create_fake_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "model.csv"
            _write_minimal_dataset(dataset_path)
            sarimax = run_sarimax_validation(dataset_path=dataset_path, dry_run=True)
            rf = run_random_forest_validation(dataset_path=dataset_path, dry_run=True)
            xgb = run_xgboost_validation(dataset_path=dataset_path, dry_run=True)
        self.assertEqual(sarimax.model_scope, "district_specific")
        if importlib.util.find_spec("statsmodels") is None:
            self.assertEqual(sarimax.status, "blocked_missing_dependency")
        if importlib.util.find_spec("sklearn") is None:
            self.assertEqual(rf.status, "blocked_missing_dependency")
        if importlib.util.find_spec("xgboost") is None:
            self.assertEqual(xgb.status, "blocked_missing_dependency")
        if importlib.util.find_spec("sklearn") is not None:
            self.assertEqual(rf.status, "dry-run")
        if importlib.util.find_spec("xgboost") is not None:
            self.assertEqual(xgb.status, "dry-run")
        self.assertEqual(rf.feature_columns, xgb.feature_columns)
        self.assertEqual(rf.count_metrics_by_horizon, {})
        self.assertEqual(xgb.surge_metrics_by_horizon, {})


def _write_minimal_dataset(path: Path) -> None:
    rows = [
        {
            "district_id": "UGA_TEST",
            "district_name": "TEST",
            "epi_year": "2025",
            "epi_week": "1",
            "week_start_date": "2025-01-06",
            "confirmed_malaria_cases": "10",
            "rainfall_lag_4": "12",
            "target_cases_h4": "14",
            "target_week_start_date_h4": "2025-02-03",
            **{column: "1" for column in SARIMAX_EXOGENOUS_FEATURES},
            **{column: "1" for column in POOLED_FEATURES if column != "district_id"},
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
