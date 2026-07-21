from __future__ import annotations

import os
import unittest
from datetime import date
from decimal import Decimal

from dotenv import load_dotenv

from app import create_app
from app.services.forecast_generation import (
    _resolve_issue_week,
    clip_non_negative,
    forecast_composite_index_exists,
    target_week_from_origin,
)


class ForecastGenerationPureTests(unittest.TestCase):
    def test_target_week_crosses_year_boundary_and_week53(self):
        year, week, start = target_week_from_origin(2020, 53, 4)
        self.assertEqual((year, week), (2021, 4))
        self.assertEqual(start.isoformat(), "2021-01-25")

    def test_approved_2025w52_target_weeks(self):
        self.assertEqual(target_week_from_origin(2025, 52, 4), (2026, 4, date(2026, 1, 19)))
        self.assertEqual(target_week_from_origin(2025, 52, 8), (2026, 8, date(2026, 2, 16)))
        self.assertEqual(target_week_from_origin(2025, 52, 12), (2026, 12, date(2026, 3, 16)))

    def test_clip_non_negative_reports_clipping(self):
        value, clipped = clip_non_negative(Decimal("-2.5"))
        self.assertEqual(value, Decimal("0"))
        self.assertTrue(clipped)
        value, clipped = clip_non_negative(Decimal("2.5"))
        self.assertEqual(value, Decimal("2.5"))
        self.assertFalse(clipped)


class ForecastGenerationDBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL is required for DB integration tests.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()

    def test_forecast_composite_index_exists(self):
        with self.app.app_context():
            self.assertTrue(forecast_composite_index_exists())

    def test_latest_issue_week_resolution(self):
        with self.app.app_context():
            self.assertEqual(_resolve_issue_week(latest=True, as_of_year=None, as_of_week=None), (2025, 52))


if __name__ == "__main__":
    unittest.main()
