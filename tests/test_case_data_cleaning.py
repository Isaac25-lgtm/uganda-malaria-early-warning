from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from app.services.case_data_cleaning import (
    CaseDataCleaningError,
    build_clean_case_series,
    load_case_exclusions,
)


class CaseDataCleaningTests(unittest.TestCase):
    def test_approved_registry_contains_exactly_six_confirmed_records(self):
        records = load_case_exclusions()
        self.assertEqual(len(records), 6)
        self.assertEqual(
            {record.key for record in records},
            {
                ("UGA_GULU_CITY", 2021, 50),
                ("UGA_JINJA", 2021, 34),
                ("UGA_KALIRO", 2024, 31),
                ("UGA_IBANDA", 2025, 25),
                ("UGA_KISORO", 2025, 15),
                ("UGA_KISORO", 2025, 17),
            },
        )
        self.assertTrue(all(item.adjudication_status == "confirmed" for item in records))

    def test_confirmed_error_is_missing_outcome_but_interpolated_for_predictors(self):
        start = date.fromisocalendar(2025, 1, 1)
        rows = [_row(start + timedelta(weeks=index), value) for index, value in enumerate((10, 20, 999, 40, 50))]
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.csv"
            _write_registry(registry, original=999, previous=20, following=40)
            result = build_clean_case_series(
                rows,
                district_ids=["UGA_TEST"],
                start_date=start,
                end_date=start + timedelta(weeks=4),
                exclusion_path=registry,
            )
        cleaned = result.values[("UGA_TEST", start + timedelta(weeks=2))]
        self.assertIsNone(cleaned.observed_cases)
        self.assertEqual(str(cleaned.cases_for_features), "30.000000")
        self.assertTrue(cleaned.predictor_interpolated)
        self.assertEqual(result.confirmed_error_count, 1)
        self.assertEqual(result.predictor_interpolated_confirmed_error_count, 1)
        self.assertFalse(result.summary()["outcome_imputation"])

    def test_unbounded_and_long_gaps_are_not_interpolated(self):
        start = date.fromisocalendar(2025, 1, 1)
        rows = [
            _row(start, 10),
            _row(start + timedelta(weeks=1), 20),
            _row(start + timedelta(weeks=2), 30),
            _row(start + timedelta(weeks=6), 70),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.csv"
            _write_registry(
                registry,
                year=2025,
                week=2,
                original=20,
                previous=10,
                following=30,
            )
            result = build_clean_case_series(
                rows,
                district_ids=["UGA_TEST"],
                start_date=start,
                end_date=start + timedelta(weeks=6),
                exclusion_path=registry,
            )
        for index in (3, 4, 5):
            value = result.values[("UGA_TEST", start + timedelta(weeks=index))]
            self.assertIsNone(value.cases_for_features)
            self.assertFalse(value.predictor_interpolated)

    def test_registry_value_mismatch_fails_closed(self):
        start = date.fromisocalendar(2025, 1, 1)
        rows = [_row(start + timedelta(weeks=index), value) for index, value in enumerate((10, 20, 30))]
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.csv"
            _write_registry(registry, year=2025, week=2, original=999, previous=10, following=30)
            with self.assertRaisesRegex(CaseDataCleaningError, "expected 999"):
                build_clean_case_series(
                    rows,
                    district_ids=["UGA_TEST"],
                    start_date=start,
                    end_date=start + timedelta(weeks=2),
                    exclusion_path=registry,
                )


def _row(week_start: date, cases: int) -> dict:
    iso = week_start.isocalendar()
    return {
        "district_id": "UGA_TEST",
        "district_name": "TEST",
        "epi_year": iso.year,
        "epi_week": iso.week,
        "week_start_date": week_start,
        "confirmed_malaria_cases": cases,
    }


def _write_registry(
    path: Path,
    *,
    year: int = 2025,
    week: int = 3,
    original: int,
    previous: int,
    following: int,
) -> None:
    fields = (
        "district_id", "district_name", "epi_year", "epi_week",
        "original_cases", "previous_cases", "following_cases", "reason_code",
        "adjudication_status", "consultation_role", "consultation_reference",
        "analytical_action",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "district_id": "UGA_TEST",
                "district_name": "TEST",
                "epi_year": year,
                "epi_week": week,
                "original_cases": original,
                "previous_cases": previous,
                "following_cases": following,
                "reason_code": "confirmed_reporting_error",
                "adjudication_status": "confirmed",
                "consultation_role": "District Biostatistician",
                "consultation_reference": "Test evidence",
                "analytical_action": "exclude_outcome_interpolate_predictors",
            }
        )


if __name__ == "__main__":
    unittest.main()
