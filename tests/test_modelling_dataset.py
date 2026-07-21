from __future__ import annotations

import unittest
from datetime import date, timedelta

from app.services.modelling_dataset import (
    assemble_dataset_rows,
    horizon_target_samples,
    leakage_checks,
    summarize_dataset_rows,
)
from app.services.preprocessing import iso_week_dates


def row(district_id: str, year: int, week: int, cases: int | None):
    start, end = iso_week_dates(year, week)
    return {
        "district_id": district_id,
        "district_name": district_id,
        "epi_year": year,
        "epi_week": week,
        "week_start_date": start,
        "week_end_date": end,
        "confirmed_malaria_cases": cases,
        "population": 1000,
        "incidence_per_1000": None if cases is None else cases,
        "facility_reporting_completeness": 90,
        "time_series_completeness": 100,
        "rainfall_current": 1,
        "temperature_current": 20,
        "humidity_current": 70,
        "rainfall_lag_4": 1,
        "rainfall_lag_8": 1,
        "rainfall_lag_12": 1,
        "temperature_lag_4": 20,
        "temperature_lag_8": 20,
        "temperature_lag_12": 20,
        "humidity_lag_4": 70,
        "humidity_lag_8": 70,
        "humidity_lag_12": 70,
    }


class ModellingDatasetPureTests(unittest.TestCase):
    def test_horizon_alignment_crosses_year_and_week53(self):
        rows = [
            row("UGA_TEST", 2020, 50, 10),
            row("UGA_TEST", 2020, 51, 11),
            row("UGA_TEST", 2020, 52, 12),
            row("UGA_TEST", 2020, 53, 13),
            row("UGA_TEST", 2021, 1, 14),
            row("UGA_TEST", 2021, 2, 15),
            row("UGA_TEST", 2021, 3, 16),
            row("UGA_TEST", 2021, 4, 17),
            row("UGA_TEST", 2021, 5, 18),
            row("UGA_TEST", 2021, 6, 19),
        ]
        dataset = assemble_dataset_rows(rows)
        origin = next(item for item in dataset if item["epi_year"] == 2020 and item["epi_week"] == 53)
        self.assertEqual(origin["target_epi_year_h4"], 2021)
        self.assertEqual(origin["target_epi_week_h4"], 4)
        self.assertEqual(origin["target_cases_h4"], 17)

    def test_rolling_features_use_past_only_and_missing_stays_missing(self):
        rows = [
            row("UGA_TEST", 2025, 1, 1),
            row("UGA_TEST", 2025, 2, None),
            row("UGA_TEST", 2025, 3, 3),
            row("UGA_TEST", 2025, 4, 100),
            row("UGA_TEST", 2025, 5, 5),
        ]
        dataset = assemble_dataset_rows(rows)
        target = next(item for item in dataset if item["epi_week"] == 5)
        self.assertEqual(target["cases_lag_1"], 100)
        self.assertEqual(target["cases_lag_2"], 3)
        self.assertEqual(target["cases_rolling_mean_4"], "34.666667")
        missing_origin = next(item for item in dataset if item["epi_week"] == 2)
        self.assertIsNone(missing_origin["confirmed_malaria_cases"])

    def test_leakage_checks_exclude_targets(self):
        dataset = assemble_dataset_rows([row("UGA_TEST", 2025, week, week) for week in range(1, 8)])
        feature_columns = [column for column in dataset[0] if not column.startswith("target_")]
        checks = leakage_checks(dataset, feature_columns)
        self.assertTrue(checks["target_columns_excluded_from_features"])
        self.assertTrue(checks["targets_future_only"])


if __name__ == "__main__":
    unittest.main()
