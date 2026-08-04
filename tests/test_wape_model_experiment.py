"""Focused safeguards for the standalone exploratory WAPE runner."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from datetime import date, timedelta


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_wape_model_experiment.py"
SPEC = importlib.util.spec_from_file_location("wape_model_experiment", MODULE_PATH)
assert SPEC and SPEC.loader
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)


class WapeExperimentContractTests(unittest.TestCase):
    def test_feature_variants_exclude_targets_population_and_reporting_completeness(self):
        EXPERIMENT.validate_experiment_contract()
        for variant in EXPERIMENT.FEATURE_VARIANTS:
            features = set(variant.numeric_features) | set(EXPERIMENT.POOLED_CATEGORICAL_FEATURES)
            self.assertFalse(features & EXPERIMENT.FORBIDDEN_COLUMNS)
            self.assertFalse(any(feature.startswith("target_") for feature in features))

    def test_fourier_variant_uses_only_origin_week_information(self):
        variant = next(item for item in EXPERIMENT.FEATURE_VARIANTS if item.add_fourier_seasonality)
        row = {feature: "1" for feature in EXPERIMENT.POOLED_NUMERIC_FEATURES}
        row.update(
            {
                "district_id": "UGA_TEST",
                "epi_week": "13",
                "rainfall_current": "10",
                "temperature_current": "25",
                "humidity_current": "60",
                "target_cases_h4": "999999",
            }
        )
        frame = EXPERIMENT.build_feature_frame([row], variant)
        self.assertNotIn("target_cases_h4", frame.columns)
        self.assertAlmostEqual(frame.loc[0, "season_sin_1"], math.sin(2 * math.pi * 13 / 53.0))

    def test_negative_binomial_dispersion_is_nonnegative_and_training_only(self):
        alpha = EXPERIMENT.estimate_negative_binomial_alpha(EXPERIMENT.np.asarray([1.0, 2.0, 20.0]))
        self.assertGreater(alpha, 0.0)
        self.assertEqual(EXPERIMENT.estimate_negative_binomial_alpha(EXPERIMENT.np.asarray([5.0])), 1e-6)

    def test_chunk_selectors_reject_unknown_models_and_keep_requested_horizon(self):
        selected = EXPERIMENT._select_horizons("4")
        self.assertEqual(selected, (4,))
        with self.assertRaises(ValueError):
            EXPERIMENT._select_items("not_a_model", EXPERIMENT.MODEL_SPECS, label="model key")

    def test_xgboost_control_is_pinned_to_the_saved_production_h4_configuration(self):
        summary_path = Path("data/quality/xgboost_validation_summary.json")
        published = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = published["selected_parameters_by_horizon"]["4"]
        control = next(spec for spec in EXPERIMENT.MODEL_SPECS if spec.key == "xgboost_control")
        self.assertEqual(control.parameters, expected)

    def test_long_history_uses_only_matching_prior_calendar_weeks(self):
        origin = date(2025, 1, 6)
        rows = []
        for weeks, cases in ((0, 500), (1, 100), (26, 2600), (52, 5200)):
            rows.append(
                {
                    "district_id": "UGA_TEST",
                    "week_start_date": (origin - timedelta(weeks=weeks)).isoformat(),
                    "confirmed_malaria_cases": str(cases),
                }
            )
        augmented = EXPERIMENT.add_long_history_features(rows)
        current = next(row for row in augmented if row["week_start_date"] == origin.isoformat())
        self.assertEqual(current["cases_lag_26"], "2600.0")
        self.assertEqual(current["cases_lag_52"], "5200.0")
        self.assertEqual(current["cases_rolling_observed_count_26"], "2")
        self.assertEqual(current["cases_rolling_mean_26"], "1350.0")

    def test_long_history_keeps_blank_source_outcomes_missing(self):
        origin = date(2025, 1, 6)
        rows = [
            {
                "district_id": "UGA_TEST",
                "week_start_date": origin.isoformat(),
                "confirmed_malaria_cases": "500",
            },
            {
                "district_id": "UGA_TEST",
                "week_start_date": (origin - timedelta(weeks=26)).isoformat(),
                "confirmed_malaria_cases": "",
            },
        ]
        augmented = EXPERIMENT.add_long_history_features(rows)
        current = next(row for row in augmented if row["week_start_date"] == origin.isoformat())
        self.assertEqual(current["cases_lag_26"], "")
        self.assertEqual(current["cases_rolling_observed_count_26"], "0")


if __name__ == "__main__":
    unittest.main()
