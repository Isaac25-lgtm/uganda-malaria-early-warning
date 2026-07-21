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
from app.services.models.pooled_tree_validation import (
    CHECKPOINT_VERSION,
    DISTRICT_METRIC_COLUMNS,
    FEATURE_IMPORTANCE_COLUMNS,
    POOLED_FEATURES,
    POOLED_NUMERIC_FEATURES,
    PREDICTION_COLUMNS,
    RANDOM_STATE,
    _build_pipeline,
    _classifier_selection_key,
    _fallback_classifier_selection_key,
    _feature_frame,
    _read_checkpoint,
    _select_operational_surge_method,
    publish_staged_outputs,
)
from app.services.models.random_forest import (
    RANDOM_FOREST_CONFIG,
    run_random_forest_validation,
)
from app.services.models.xgboost_model import XGBOOST_CONFIG, run_xgboost_validation
from app.services.modelling_dataset import read_modelling_dataset


class PooledTreeValidationTests(unittest.TestCase):
    def test_operational_surge_method_rejects_false_alert_heavy_classifier(self):
        classifier = {
            "f2": "0.72",
            "sensitivity": "0.98",
            "specificity": "0.15",
            "precision": "0.35",
            "pr_auc": "0.55",
        }
        count_threshold = {
            "f2": "0.68",
            "sensitivity": "0.73",
            "specificity": "0.69",
            "precision": "0.53",
            "pr_auc": "0.61",
        }

        method, metrics = _select_operational_surge_method(
            classifier_metrics=classifier,
            count_threshold_metrics=count_threshold,
        )

        self.assertEqual(method, "count_threshold")
        self.assertIs(metrics, count_threshold)

    def test_classifier_operating_point_ranking_enforces_usable_balance(self):
        near_all_positive = {
            "metrics": {
                "f2": 0.71,
                "sensitivity": 0.99,
                "specificity": 0.04,
                "pr_auc": 0.55,
                "precision": 0.33,
            }
        }
        balanced = {
            "metrics": {
                "f2": 0.62,
                "sensitivity": 0.73,
                "specificity": 0.48,
                "pr_auc": 0.53,
                "precision": 0.38,
            }
        }

        self.assertGreater(
            _classifier_selection_key(near_all_positive),
            _classifier_selection_key(balanced),
        )
        self.assertGreater(
            _fallback_classifier_selection_key(balanced),
            _fallback_classifier_selection_key(near_all_positive),
        )
        self.assertGreaterEqual(balanced["metrics"]["sensitivity"], 0.60)
        self.assertGreaterEqual(balanced["metrics"]["specificity"], 0.40)

    def test_checkpoint_reuse_reclassifies_probability_only_when_parameters_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.json"
            rows = _fixture_rows()
            fold = build_rolling_origin_folds(rows, horizon=4, max_folds=1)[0]
            parameters = dict(RANDOM_FOREST_CONFIG.parameter_candidates[0])
            classifier_parameters = dict(
                (RANDOM_FOREST_CONFIG.classifier_parameter_candidates or
                 RANDOM_FOREST_CONFIG.parameter_candidates)[0]
            )
            signature = {"sha256": "fixture", "size_bytes": 1}
            job = {
                "status": "success",
                "horizon": 4,
                "validation_fold": 1,
                "origin_date": fold.origin_date.isoformat(),
                "target_date": fold.target_date.isoformat(),
                "surge_probability_cutoff": 0.30,
                "predictions": [{
                    "surge_probability": "0.420000",
                    "predicted_surge_classification": True,
                    "surge_probability_cutoff": 0.30,
                }],
            }
            path.write_text(
                json.dumps({
                    "checkpoint_version": CHECKPOINT_VERSION,
                    "model_key": RANDOM_FOREST_CONFIG.model_key,
                    "dataset_signature": signature,
                    "max_folds": 1,
                    "features": list(POOLED_FEATURES),
                    "selected_parameters": parameters,
                    "selected_classifier_parameters": classifier_parameters,
                    "probability_cutoff": 0.30,
                    "job": job,
                }),
                encoding="utf-8",
            )

            reused = _read_checkpoint(
                path,
                config=RANDOM_FOREST_CONFIG,
                dataset_signature=signature,
                horizon=4,
                fold_number=1,
                fold=fold,
                max_folds=1,
                selected_parameters=parameters,
                selected_classifier_parameters=classifier_parameters,
                probability_cutoff=0.45,
            )
            rejected = _read_checkpoint(
                path,
                config=RANDOM_FOREST_CONFIG,
                dataset_signature=signature,
                horizon=4,
                fold_number=1,
                fold=fold,
                max_folds=1,
                selected_parameters=parameters,
                selected_classifier_parameters={**classifier_parameters, "max_depth": 99},
                probability_cutoff=0.45,
            )

        self.assertIsNotNone(reused)
        self.assertEqual(reused["predictions"][0]["surge_probability"], "0.420000")
        self.assertFalse(reused["predictions"][0]["predicted_surge_classification"])
        self.assertEqual(reused["predictions"][0]["surge_probability_cutoff"], 0.45)
        self.assertIsNone(rejected)

    def test_preprocessing_is_training_only_handles_unseen_district_and_is_deterministic(self):
        rows = _fixture_rows()
        training_rows = [row for row in rows if row["district_id"] in {"UGA_A", "UGA_B"}][:80]
        validation_row = next(row for row in rows if row["district_id"] == "UGA_C")
        for index, row in enumerate(training_rows):
            row["rainfall_lag_4"] = "1" if index % 2 == 0 else "3"
        validation_row["rainfall_lag_4"] = ""
        train_x = _feature_frame(training_rows)
        validation_x = _feature_frame([validation_row])
        train_y = [float(row["confirmed_malaria_cases"]) for row in training_rows]
        parameters = RANDOM_FOREST_CONFIG.parameter_candidates[0]

        first = _build_pipeline(RANDOM_FOREST_CONFIG, parameters)
        second = _build_pipeline(RANDOM_FOREST_CONFIG, parameters)
        first.fit(train_x, train_y)
        second.fit(train_x, train_y)
        first_prediction = first.predict(validation_x)[0]
        second_prediction = second.predict(validation_x)[0]

        numeric_imputer = first.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
        rainfall_index = list(POOLED_NUMERIC_FEATURES).index("rainfall_lag_4")
        self.assertEqual(numeric_imputer.statistics_[rainfall_index], 2.0)
        onehot = first.named_steps["preprocess"].named_transformers_["district"].named_steps["onehot"]
        self.assertEqual(set(onehot.categories_[0]), {"UGA_A", "UGA_B"})
        self.assertNotIn("UGA_C", onehot.categories_[0])
        self.assertAlmostEqual(first_prediction, second_prediction, places=12)

    def test_completed_dataset_removes_gate_and_uses_shared_folds_and_approved_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "fixture.csv"
            _write_fixture(dataset)
            summary = run_random_forest_validation(
                dataset_path=dataset,
                horizons=(4,),
                dry_run=True,
                max_folds=1,
            )
            shared_fold = build_rolling_origin_folds(
                read_modelling_dataset(dataset), horizon=4, max_folds=1
            )[0]

        self.assertEqual(summary.status, "dry-run")
        self.assertEqual(summary.model_scope, "national_pooled")
        self.assertEqual(summary.job_counts, {"planned": 1, "total": 1})
        self.assertEqual(summary.validation_cutoffs[0]["origin_week"], shared_fold.origin_date.isoformat())
        self.assertEqual(summary.validation_cutoffs[0]["target_week"], shared_fold.target_date.isoformat())
        self.assertEqual(tuple(summary.feature_columns), POOLED_FEATURES)
        self.assertTrue({"cases_lag_1", "cases_lag_12", "rainfall_lag_4", "humidity_lag_12"}.issubset(summary.feature_columns))
        self.assertFalse(any("current" in column for column in summary.feature_columns))
        self.assertFalse(any("target" in column for column in summary.feature_columns))
        self.assertFalse(any("wind" in column for column in summary.feature_columns))
        self.assertFalse(any("gated" in warning.lower() for warning in summary.warnings))

    def test_real_random_forest_fit_outputs_metrics_resume_and_preserves_prior_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "fixture.csv"
            _write_fixture(dataset)
            paths = _output_paths(root, "random_forest")
            progress: list[dict] = []
            summary = run_random_forest_validation(
                dataset_path=dataset,
                horizons=(4,),
                max_folds=1,
                progress_callback=progress.append,
                record_pipeline_run=False,
                **paths,
            )

            self.assertEqual(summary.status, "success")
            self.assertTrue(summary.published)
            self.assertEqual(summary.job_counts, {"success": 1, "total": 1})
            self.assertEqual(summary.count_metrics_by_horizon[4]["n"], 2)
            self.assertEqual(summary.surge_metrics_by_horizon[4]["n"], 2)
            self.assertEqual(
                set(summary.count_metrics_by_horizon[4]),
                {"n", "mae", "rmse", "mape", "wape", "r_squared"},
            )
            self.assertEqual(
                set(summary.surge_metrics_by_horizon[4]),
                {
                    "n", "tp", "tn", "fp", "fn", "sensitivity", "specificity",
                    "precision", "f1", "f2", "accuracy", "auc", "pr_auc",
                    "brier_score",
                },
            )
            self.assertEqual(progress[0]["model"], "Random Forest")
            self.assertEqual(progress[0]["validation_fold"], 1)
            self.assertEqual(progress[0]["status"], "success")

            prediction_rows, prediction_fields = _read_csv(paths["predictions_path"])
            metric_rows, metric_fields = _read_csv(paths["district_metrics_path"])
            importance_rows, importance_fields = _read_csv(paths["feature_importance_path"])
            self.assertEqual(tuple(prediction_fields), PREDICTION_COLUMNS)
            self.assertEqual(tuple(metric_fields), DISTRICT_METRIC_COLUMNS)
            self.assertEqual(tuple(importance_fields), FEATURE_IMPORTANCE_COLUMNS)
            self.assertEqual(len(prediction_rows), 2)
            self.assertEqual(len(metric_rows), 3)
            self.assertNotIn("UGA_C", {row["district_id"] for row in prediction_rows})
            missing_target_metric = next(row for row in metric_rows if row["district_id"] == "UGA_C")
            self.assertEqual(missing_target_metric["excluded_missing_target"], "1")
            self.assertEqual(missing_target_metric["n"], "0")
            self.assertEqual({row["importance_method"] for row in importance_rows}, {"impurity", "permutation_validation_mae"})
            self.assertTrue(any(row["validation_fold"] == "overall" for row in importance_rows))

            diagnostics = json.loads(paths["diagnostics_path"].read_text(encoding="utf-8"))
            audit = diagnostics["jobs"][0]["preprocessing_audit"]
            self.assertGreater(audit["fit_rows"], 0)
            self.assertEqual(audit["validation_rows_seen_during_fit"], 0)
            self.assertFalse(audit["outcome_imputation"])
            self.assertEqual(diagnostics["jobs"][0]["training_missing_target_exclusions"], 1)
            self.assertEqual(summary.row_counts["validation_missing_target_exclusions"], 1)
            self.assertIn("numeric__", diagnostics["jobs"][0]["transformed_feature_names"][0])
            card = json.loads(paths["model_card_path"].read_text(encoding="utf-8"))
            self.assertEqual(card["model_scope"], "national_pooled")
            self.assertIn("Impurity importance", " ".join(card["explainability"]))
            self.assertEqual(RANDOM_FOREST_CONFIG.controlled_threads, 2)
            self.assertEqual(RANDOM_STATE, 24024)

            first_predictions = paths["predictions_path"].read_text(encoding="utf-8")
            resumed = run_random_forest_validation(
                dataset_path=dataset,
                horizons=(4,),
                max_folds=1,
                record_pipeline_run=False,
                **paths,
            )
            self.assertEqual(resumed.checkpoints_reused, 1)
            self.assertEqual(first_predictions, paths["predictions_path"].read_text(encoding="utf-8"))

            with patch(
                "app.services.models.pooled_tree_validation._build_pipeline",
                side_effect=RuntimeError("forced fold failure"),
            ):
                failed = run_random_forest_validation(
                    dataset_path=dataset,
                    horizons=(4,),
                    max_folds=1,
                    replace=True,
                    record_pipeline_run=False,
                    **paths,
                )
            self.assertEqual(failed.status, "failed")
            self.assertFalse(failed.published)
            preserved = json.loads(paths["artifact_path"].read_text(encoding="utf-8"))
            self.assertEqual(preserved["status"], "success")
            checkpoint = json.loads((paths["checkpoint_dir"] / "h4_fold1.json").read_text(encoding="utf-8"))
            self.assertIn("forced fold failure", checkpoint["job"]["failure_reason"])

    def test_real_xgboost_fit_produces_gain_permutation_and_contributions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "fixture.csv"
            _write_fixture(dataset)
            paths = _output_paths(root, "xgboost")
            summary = run_xgboost_validation(
                dataset_path=dataset,
                horizons=(4,),
                max_folds=1,
                record_pipeline_run=False,
                **paths,
            )
            self.assertEqual(summary.status, "success")
            self.assertEqual(summary.model_scope, "national_pooled")
            predictions, _ = _read_csv(paths["predictions_path"])
            importance, _ = _read_csv(paths["feature_importance_path"])
            self.assertTrue(all(json.loads(row["feature_contributions"]) for row in predictions))
            self.assertEqual({row["importance_method"] for row in importance}, {"gain", "permutation_validation_mae"})
            diagnostics = json.loads(paths["diagnostics_path"].read_text(encoding="utf-8"))
            self.assertFalse(diagnostics["outer_validation_used_for_tuning"])
            self.assertEqual(diagnostics["controlled_threads"], 2)
            self.assertEqual(XGBOOST_CONFIG.controlled_threads, 2)

    def test_publication_failure_restores_every_previous_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.csv"
            second = root / "second.json"
            staged_first = root / "new-first.tmp"
            staged_second = root / "new-second.tmp"
            first.write_text("old first", encoding="utf-8")
            second.write_text("old second", encoding="utf-8")
            staged_first.write_text("new first", encoding="utf-8")
            staged_second.write_text("new second", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with publish_staged_outputs({staged_first: first, staged_second: second}):
                    raise RuntimeError("interrupted")

            self.assertEqual(first.read_text(encoding="utf-8"), "old first")
            self.assertEqual(second.read_text(encoding="utf-8"), "old second")


def _output_paths(root: Path, model_key: str) -> dict:
    return {
        "artifact_path": root / f"{model_key}_summary.json",
        "predictions_path": root / f"{model_key}_predictions.csv",
        "district_metrics_path": root / f"{model_key}_district_metrics.csv",
        "diagnostics_path": root / f"{model_key}_diagnostics.json",
        "model_card_path": root / f"{model_key}_model_card.json",
        "feature_importance_path": root / f"{model_key}_importance.csv",
        "checkpoint_dir": root / f"{model_key}_checkpoints",
    }


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_fixture(path: Path) -> None:
    rows = _fixture_rows()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture_rows() -> list[dict[str, str]]:
    start = date(2023, 1, 2)
    districts = (("UGA_A", "DISTRICT A", 0), ("UGA_B", "DISTRICT B", 7), ("UGA_C", "DISTRICT C", 14))
    weeks = 72
    series = {
        district_id: [
            30 + offset + index // 9 + (index % 7) + int(3 * math.sin(index / 5))
            for index in range(weeks)
        ]
        for district_id, _, offset in districts
    }
    output: list[dict[str, str]] = []
    for district_id, district_name, offset in districts:
        values = series[district_id]
        for index, cases in enumerate(values):
            week = start + timedelta(weeks=index)
            row: dict[str, str] = {
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
            for window in (4, 8, 12):
                history = values[max(0, index - window):index]
                row[f"cases_rolling_mean_{window}"] = str(sum(history) / len(history)) if history else ""
                row[f"cases_rolling_interpolated_count_{window}"] = "0"
            for lag in (1, 2, 4, 8, 12):
                row[f"cases_lag_{lag}_interpolated"] = "0"
            for climate_index, column in enumerate(
                [name for name in POOLED_NUMERIC_FEATURES if name.startswith(("rainfall_", "temperature_", "humidity_"))],
                start=1,
            ):
                row[column] = str(20 + offset / 4 + climate_index + math.sin(index / (climate_index + 2)))
            for horizon in (4, 8, 12):
                target_index = index + horizon
                row[f"target_week_start_date_h{horizon}"] = (week + timedelta(weeks=horizon)).isoformat()
                row[f"target_epi_week_h{horizon}"] = str((week + timedelta(weeks=horizon)).isocalendar().week)
                row[f"target_cases_h{horizon}"] = str(values[target_index]) if target_index < weeks else ""
            if district_id == "UGA_B" and index == 67:
                row["rainfall_lag_4"] = ""
            if district_id == "UGA_C" and index in {30, 67}:
                row["target_cases_h4"] = ""
            output.append(row)
    output.sort(key=lambda row: (row["district_id"], row["week_start_date"]))
    return output


if __name__ == "__main__":
    unittest.main()
