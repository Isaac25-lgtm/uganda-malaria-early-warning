from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.models import DistrictReference
from app.services.dhis2_mapping import (
    build_mapping_summary,
    check_week_53_schema,
)


def _reference(district_id: str, district_name: str, existing_uid: str | None = None):
    return DistrictReference(
        district_id=district_id,
        district_name=district_name,
        dhis2_orgunit_id=existing_uid,
        active=True,
    )


def _orgunit(uid: str, name: str):
    return {
        "id": uid,
        "name": name,
        "displayName": name,
        "level": 3,
        "parent": {"id": "PARENT", "name": "Region"},
        "children": [],
    }


class DHIS2MappingPureTests(unittest.TestCase):
    def _alias_path(self) -> Path:
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, newline="")
        handle.write("dhis2_name,district_name,reason\n")
        handle.write("LUWERO,LUWEERO,reviewed spelling variant\n")
        handle.write("SEMBABULE,SSEMBABULE,reviewed spelling variant\n")
        handle.close()
        return Path(handle.name)

    def test_exact_normalized_and_reviewed_alias_matches(self):
        alias_path = self._alias_path()
        try:
            refs = [
                _reference("UGA_ABIM", "ABIM"),
                _reference("UGA_MADI_OKOLLO", "MADI-OKOLLO"),
                _reference("UGA_LUWEERO", "LUWEERO"),
            ]
            orgs = [
                _orgunit("OU_ABIM", "ABIM"),
                _orgunit("OU_MADI", "Madi-Okollo District"),
                _orgunit("OU_LUWERO", "Luwero District"),
            ]
            summary = build_mapping_summary(orgs, refs, level=3, alias_path=alias_path)
        finally:
            alias_path.unlink(missing_ok=True)

        self.assertEqual(summary.exact_matches, 1)
        self.assertEqual(summary.normalized_matches, 1)
        self.assertEqual(summary.alias_matches, 1)
        self.assertEqual(summary.unmatched_dhis2_org_units, [])
        self.assertEqual(summary.unmatched_districts, [])

    def test_one_to_one_bijection_passes_for_complete_146_frame(self):
        refs = [
            _reference(f"UGA_TEST_{index:03d}", f"TEST {index:03d}")
            for index in range(1, 147)
        ]
        orgs = [
            _orgunit(f"OU_{index:03d}", f"Test {index:03d} District")
            for index in range(1, 147)
        ]
        summary = build_mapping_summary(orgs, refs, level=3, alias_path=Path("missing.csv"))
        self.assertTrue(summary.one_to_one_passed)
        self.assertEqual(summary.matched_count, 146)

    def test_duplicate_uid_breaks_bijection(self):
        refs = [
            _reference(f"UGA_TEST_{index:03d}", f"TEST {index:03d}")
            for index in range(1, 147)
        ]
        orgs = [
            _orgunit("OU_DUP" if index in {1, 2} else f"OU_{index:03d}", f"Test {index:03d} District")
            for index in range(1, 147)
        ]
        summary = build_mapping_summary(orgs, refs, level=3, alias_path=Path("missing.csv"))
        self.assertFalse(summary.one_to_one_passed)
        self.assertEqual(summary.duplicate_dhis2_uids, ["OU_DUP"])

    def test_existing_mapping_change_is_reported(self):
        refs = [_reference("UGA_ABIM", "ABIM", existing_uid="OLD_UID")]
        orgs = [_orgunit("NEW_UID", "Abim District")]
        summary = build_mapping_summary(orgs, refs, level=3, alias_path=Path("missing.csv"))
        self.assertEqual(summary.mapping_changes, ["UGA_ABIM: dhis2_orgunit_id 'OLD_UID' -> 'NEW_UID'"])


class DHIS2MappingDBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL is required for DB integration tests.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()

    def setUp(self):
        self.ctx = self.app.app_context()
        self.ctx.push()

    def tearDown(self):
        db.session.rollback()
        self.ctx.pop()

    def test_week_53_constraints_are_consistent(self):
        summary = check_week_53_schema()
        self.assertTrue(summary.passed)
        self.assertTrue(summary.district_week_features_allows_53)
        self.assertTrue(summary.district_week_climate_allows_53)
        self.assertTrue(summary.raw_dhis2_malaria_allows_53)


if __name__ == "__main__":
    unittest.main()
