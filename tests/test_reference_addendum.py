from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.models import DistrictReference, UBOSPopulation
from app.services.reference_data import (
    EXPECTED_ACTIVE_DISTRICTS,
    district_id_from_name,
    generate_district_reference_csv_from_boundary,
    import_ubos_population,
    prepare_raw_boundaries,
)


class ReferenceAddendumTests(unittest.TestCase):
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
        UBOSPopulation.query.filter(
            UBOSPopulation.district_id.like("UGA_TEST_ADDENDUM%")
        ).delete(synchronize_session=False)
        DistrictReference.query.filter(
            DistrictReference.district_id.like("UGA_TEST_ADDENDUM%")
        ).delete(synchronize_session=False)
        db.session.commit()

    def test_prepare_raw_boundaries_writes_district_id_geojson(self):
        if DistrictReference.query.filter_by(active=True).count() != EXPECTED_ACTIVE_DISTRICTS:
            self.skipTest("Full 146-district reference frame is not loaded.")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "uganda_districts_prepared.geojson"
            summary = prepare_raw_boundaries(output_path=output_path)

            self.assertTrue(summary.prepared_written)
            self.assertEqual(summary.feature_count, EXPECTED_ACTIVE_DISTRICTS)
            self.assertEqual(summary.district_name_field, "District")
            self.assertEqual(summary.valid_geometry_count, EXPECTED_ACTIVE_DISTRICTS)
            prepared = json.loads(output_path.read_text(encoding="utf-8"))
            first_props = prepared["features"][0]["properties"]
            self.assertIn("district_id", first_props)
            self.assertIn("district_name", first_props)
            self.assertIn("original_District", first_props)

    def test_district_id_generation_matches_prompt_11_rule(self):
        self.assertEqual(district_id_from_name("MADI-OKOLLO"), "UGA_MADI_OKOLLO")
        self.assertEqual(district_id_from_name("ARUA CITY"), "UGA_ARUA_CITY")
        self.assertEqual(district_id_from_name(" St. Mary's "), "UGA_ST_MARYS")

    def test_generate_district_reference_csv_from_boundary(self):
        features = []
        names = ["MADI-OKOLLO"] + [f"TEST GENERATION {index:03d}" for index in range(1, 146)]
        for index, district_name in enumerate(names, start=1):
            features.append(
                {
                    "type": "Feature",
                    "properties": {"District": district_name},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [index, 0],
                                [index + 0.1, 0],
                                [index + 0.1, 0.1],
                                [index, 0.1],
                                [index, 0],
                            ]
                        ],
                    },
                }
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "UGANDA_DISTRICT.json"
            output_path = Path(tmpdir) / "district_reference.csv"
            raw_path.write_text(
                json.dumps({"type": "FeatureCollection", "features": features}),
                encoding="utf-8",
            )
            summary = generate_district_reference_csv_from_boundary(
                raw_path=raw_path,
                output_path=output_path,
            )

            self.assertTrue(summary.written)
            with output_path.open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), EXPECTED_ACTIVE_DISTRICTS)
            madi = next(row for row in rows if row["district_name"] == "MADI-OKOLLO")
            self.assertEqual(madi["district_id"], "UGA_MADI_OKOLLO")
            self.assertEqual(madi["active"], "true")
            self.assertEqual(madi["dhis2_orgunit_id"], "")

    def test_ubos_population_import_maps_by_name_and_stores_one_year(self):
        db.session.add(
            DistrictReference(
                district_id="UGA_TEST_ADDENDUM_POP",
                district_name="TEST ADDENDUM POP",
                active=True,
            )
        )
        db.session.commit()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ubos_population.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "district_id",
                        "district_name",
                        "year",
                        "population",
                        "source",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "district_id": "",
                        "district_name": "TEST ADDENDUM POP",
                        "year": "2024",
                        "population": "123,456",
                        "source": "UBOS",
                        "notes": "test denominator",
                    }
                )

            summary = import_ubos_population(csv_path=csv_path)
            self.assertEqual(summary.inserted, 1)
            row = UBOSPopulation.query.filter_by(
                district_id="UGA_TEST_ADDENDUM_POP", year=2024
            ).one()
            self.assertEqual(row.population, 123456)
            self.assertEqual(row.source, "UBOS")

    def test_ubos_population_requires_year(self):
        db.session.add(
            DistrictReference(
                district_id="UGA_TEST_ADDENDUM_NOYEAR",
                district_name="TEST ADDENDUM NOYEAR",
                active=True,
            )
        )
        db.session.commit()

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ubos_population.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "district_id",
                        "district_name",
                        "year",
                        "population",
                        "source",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "district_id": "UGA_TEST_ADDENDUM_NOYEAR",
                        "district_name": "TEST ADDENDUM NOYEAR",
                        "year": "",
                        "population": "100",
                        "source": "UBOS",
                        "notes": "",
                    }
                )

            summary = import_ubos_population(csv_path=csv_path, dry_run=True)
            self.assertTrue(summary.errors)
            self.assertIn("year is required", summary.errors[0])
            self.assertEqual(
                UBOSPopulation.query.filter_by(
                    district_id="UGA_TEST_ADDENDUM_NOYEAR"
                ).count(),
                0,
            )


if __name__ == "__main__":
    unittest.main()
