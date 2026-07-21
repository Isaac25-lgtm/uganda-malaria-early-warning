from __future__ import annotations

import os
import unittest
from decimal import Decimal

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.models import DistrictReference, DistrictWeekFeature, UBOSPopulation
from app.services.population import apply_population_to_features


class PopulationApplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL is required for DB integration tests.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()
        self._cleanup()

    def tearDown(self):
        self._cleanup()
        self.ctx.pop()

    def _cleanup(self):
        DistrictWeekFeature.query.filter(
            DistrictWeekFeature.district_id.like("UGA_TEST_POP%")
        ).delete(synchronize_session=False)
        UBOSPopulation.query.filter(
            UBOSPopulation.district_id.like("UGA_TEST_POP%")
        ).delete(synchronize_session=False)
        DistrictReference.query.filter(
            DistrictReference.district_id.like("UGA_TEST_POP%")
        ).delete(synchronize_session=False)
        db.session.commit()

    def _district(self, district_id: str) -> None:
        db.session.add(
            DistrictReference(
                district_id=district_id,
                district_name=district_id.replace("_", " "),
                active=True,
            )
        )
        db.session.commit()

    def test_apply_population_computes_weekly_incidence(self):
        self._district("UGA_TEST_POP_A")
        db.session.add(
            UBOSPopulation(
                district_id="UGA_TEST_POP_A",
                district_name="UGA TEST POP A",
                year=2024,
                population=200000,
                source="UBOS",
            )
        )
        db.session.add(
            DistrictWeekFeature(
                district_id="UGA_TEST_POP_A",
                epi_year=2024,
                epi_week=1,
                confirmed_malaria_cases=100,
            )
        )
        db.session.commit()

        summary = apply_population_to_features(2024)
        self.assertEqual(summary.rows_updated, 1)
        row = DistrictWeekFeature.query.filter_by(district_id="UGA_TEST_POP_A").one()
        self.assertEqual(row.population, 200000)
        self.assertEqual(row.incidence_per_1000, Decimal("0.500000"))

    def test_missing_denominator_leaves_population_and_incidence_null(self):
        self._district("UGA_TEST_POP_MISSING")
        db.session.add(
            DistrictWeekFeature(
                district_id="UGA_TEST_POP_MISSING",
                epi_year=2024,
                epi_week=1,
                confirmed_malaria_cases=100,
            )
        )
        db.session.commit()

        summary = apply_population_to_features(2024)
        self.assertEqual(summary.rows_updated, 0)
        self.assertEqual(summary.missing_denominator_rows, 1)
        row = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_POP_MISSING"
        ).one()
        self.assertIsNone(row.population)
        self.assertIsNone(row.incidence_per_1000)

    def test_dry_run_does_not_write(self):
        self._district("UGA_TEST_POP_DRY")
        db.session.add(
            UBOSPopulation(
                district_id="UGA_TEST_POP_DRY",
                district_name="UGA TEST POP DRY",
                year=2024,
                population=100000,
                source="UBOS",
            )
        )
        db.session.add(
            DistrictWeekFeature(
                district_id="UGA_TEST_POP_DRY",
                epi_year=2024,
                epi_week=1,
                confirmed_malaria_cases=10,
            )
        )
        db.session.commit()

        summary = apply_population_to_features(2024, dry_run=True)
        self.assertEqual(summary.rows_to_update, 1)
        row = DistrictWeekFeature.query.filter_by(district_id="UGA_TEST_POP_DRY").one()
        self.assertIsNone(row.population)
        self.assertIsNone(row.incidence_per_1000)

    def test_fixed_2024_denominator_requires_explicit_use_year_and_updates_range(self):
        self._district("UGA_TEST_POP_FIXED")
        db.session.add(
            UBOSPopulation(
                district_id="UGA_TEST_POP_FIXED",
                district_name="UGA TEST POP FIXED",
                year=2024,
                population=50000,
                source="UBOS",
            )
        )
        db.session.add(
            DistrictWeekFeature(
                district_id="UGA_TEST_POP_FIXED",
                epi_year=2020,
                epi_week=1,
                confirmed_malaria_cases=25,
            )
        )
        db.session.commit()

        summary = apply_population_to_features(
            use_year=2024,
            apply_to_years=range(2020, 2026),
        )
        self.assertEqual(summary.rows_updated, 1)
        self.assertIsNotNone(summary.fixed_denominator_assumption)
        row = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_POP_FIXED", epi_year=2020
        ).one()
        self.assertEqual(row.population, 50000)
        self.assertEqual(row.incidence_per_1000, Decimal("0.500000"))

    def test_fixed_2024_denominator_can_apply_to_approved_period_subset(self):
        self._district("UGA_TEST_POP_SUBSET")
        db.session.add(
            UBOSPopulation(
                district_id="UGA_TEST_POP_SUBSET",
                district_name="UGA TEST POP SUBSET",
                year=2024,
                population=1000,
                source="UBOS",
            )
        )
        db.session.add(
            DistrictWeekFeature(
                district_id="UGA_TEST_POP_SUBSET",
                epi_year=2025,
                epi_week=1,
                confirmed_malaria_cases=5,
            )
        )
        db.session.commit()

        summary = apply_population_to_features(
            use_year=2024,
            apply_to_years=[2025],
        )
        self.assertEqual(summary.rows_updated, 1)
        self.assertIsNotNone(summary.fixed_denominator_assumption)
        row = DistrictWeekFeature.query.filter_by(
            district_id="UGA_TEST_POP_SUBSET", epi_year=2025
        ).one()
        self.assertEqual(row.population, 1000)
        self.assertEqual(row.incidence_per_1000, Decimal("5.000000"))


if __name__ == "__main__":
    unittest.main()
