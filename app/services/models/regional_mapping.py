"""Strict DHIS2 subregion mapping loader for the regional-pooling experiment.

This module is read-only. It reads the frozen, reviewed district-to-DHIS2
mapping (``data/reference/dhis2_orgunit_mapping_review.csv``) and validates it
against the confirmed audit before any regional model is fitted. It never
contacts DHIS2, never requires credentials, and never mutates a source file.

The canonical join key is ``district_id`` and the frozen DHIS2 subregion
assignment is the mapping's ``parent_name`` column. Values are preserved
verbatim; districts are never silently normalized or reassigned.
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REGIONAL_MAPPING_PATH = Path("data/reference/dhis2_orgunit_mapping_review.csv")

# Confirmed audit of the reviewed mapping. The loader fails closed unless the
# mapping reproduces these subregion names and district counts exactly.
EXPECTED_SUBREGION_DISTRICT_COUNTS: dict[str, int] = {
    "Acholi": 9,
    "Ankole": 13,
    "Bugisu": 10,
    "Bukedi": 7,
    "Bunyoro": 9,
    "Busoga": 12,
    "Kampala": 1,
    "Karamoja": 9,
    "Kigezi": 6,
    "Lango": 10,
    "North Buganda": 12,
    "South Buganda": 14,
    "Teso": 11,
    "Tooro": 10,
    "West Nile": 13,
}
EXPECTED_DISTRICT_TOTAL = 146
EXPECTED_SUBREGION_TOTAL = 15

# Kampala is a single district and must never be trained as a regionally pooled
# model. It is served by the national candidate as an explicit fallback.
SINGLE_DISTRICT_SUBREGIONS = frozenset(
    name for name, count in EXPECTED_SUBREGION_DISTRICT_COUNTS.items() if count < 2
)

REGION_MAPPING_EXPORT_COLUMNS = (
    "district_id",
    "district_name",
    "dhis2_orgunit_id",
    "dhis2_orgunit_name",
    "subregion",
)


class RegionalMappingError(RuntimeError):
    """Raised when the reviewed DHIS2 subregion mapping cannot be trusted."""


@dataclass(frozen=True)
class DistrictMapping:
    district_id: str
    district_name: str
    dhis2_orgunit_id: str
    dhis2_orgunit_name: str
    subregion: str


@dataclass
class RegionalMapping:
    """A validated district-to-DHIS2-subregion mapping."""

    districts: tuple[DistrictMapping, ...]
    source_path: Path
    source_sha256: str
    subregion_counts: dict[str, int] = field(default_factory=dict)

    @property
    def district_ids(self) -> frozenset[str]:
        return frozenset(item.district_id for item in self.districts)

    @property
    def subregions(self) -> tuple[str, ...]:
        return tuple(sorted(self.subregion_counts))

    def subregion_of(self, district_id: str) -> str:
        for item in self.districts:
            if item.district_id == district_id:
                return item.subregion
        raise RegionalMappingError(f"District {district_id!r} is not in the mapping.")

    def district_ids_in(self, subregion: str) -> tuple[str, ...]:
        return tuple(
            sorted(item.district_id for item in self.districts if item.subregion == subregion)
        )

    def is_single_district_subregion(self, subregion: str) -> bool:
        return subregion in SINGLE_DISTRICT_SUBREGIONS or self.subregion_counts.get(subregion, 0) < 2

    def poolable_subregions(self) -> tuple[str, ...]:
        """Subregions with two or more districts (excludes Kampala)."""
        return tuple(name for name in self.subregions if not self.is_single_district_subregion(name))

    def fallback_subregions(self) -> tuple[str, ...]:
        """Single-district subregions served by the national fallback."""
        return tuple(name for name in self.subregions if self.is_single_district_subregion(name))

    def audit(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "district_total": len(self.districts),
            "subregion_total": len(self.subregion_counts),
            "subregion_district_counts": dict(sorted(self.subregion_counts.items())),
            "poolable_subregions": list(self.poolable_subregions()),
            "national_fallback_subregions": list(self.fallback_subregions()),
            "expected_matches_reviewed_audit": (
                self.subregion_counts == EXPECTED_SUBREGION_DISTRICT_COUNTS
            ),
        }


def load_regional_mapping(
    path: Path = REGIONAL_MAPPING_PATH,
    *,
    approved_district_ids: set[str] | frozenset[str] | None = None,
    expected_subregion_counts: dict[str, int] | None = None,
) -> RegionalMapping:
    """Load and strictly validate the reviewed DHIS2 subregion mapping.

    ``approved_district_ids`` is the canonical active 146-district frame. When
    provided, the loader fails if a mapping district is absent from the frame or
    if a frame district is unmapped.

    ``expected_subregion_counts`` defaults to the confirmed reviewed audit
    (:data:`EXPECTED_SUBREGION_DISTRICT_COUNTS`) and should be left at its default
    in production. It is overridable only so that fast unit tests can exercise the
    engine on a small synthetic mapping; the production CLI never overrides it.
    """
    audit_counts = expected_subregion_counts or EXPECTED_SUBREGION_DISTRICT_COUNTS
    expected_district_total = sum(audit_counts.values())
    expected_subregion_total = len(audit_counts)
    if not path.exists():
        raise RegionalMappingError(f"DHIS2 subregion mapping file is missing: {path}.")
    source_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if "district_id" not in header:
            raise RegionalMappingError("Mapping is missing the required 'district_id' column.")
        if "parent_name" not in header:
            raise RegionalMappingError("Mapping is missing the required 'parent_name' column.")
        raw_rows = list(reader)

    districts: list[DistrictMapping] = []
    seen_assignments: dict[str, set[str]] = defaultdict(set)
    for line_number, row in enumerate(raw_rows, start=2):
        district_id = (row.get("district_id") or "").strip()
        subregion = (row.get("parent_name") or "").strip()
        if not district_id:
            raise RegionalMappingError(f"Blank district_id at mapping line {line_number}.")
        if not subregion:
            raise RegionalMappingError(
                f"Blank parent_name (subregion) for {district_id} at mapping line {line_number}."
            )
        districts.append(
            DistrictMapping(
                district_id=district_id,
                district_name=(row.get("district_name") or "").strip(),
                dhis2_orgunit_id=(row.get("dhis2_orgunit_id") or "").strip(),
                dhis2_orgunit_name=(row.get("dhis2_orgunit_name") or "").strip(),
                subregion=subregion,
            )
        )
        seen_assignments[district_id].add(subregion)

    # One district must never appear under more than one subregion.
    contradictory = {
        district_id: sorted(subs)
        for district_id, subs in seen_assignments.items()
        if len(subs) > 1
    }
    if contradictory:
        raise RegionalMappingError(
            "Districts assigned to multiple subregions: "
            + "; ".join(f"{k} -> {v}" for k, v in sorted(contradictory.items()))
        )

    district_id_counts = Counter(item.district_id for item in districts)
    duplicates = sorted(k for k, v in district_id_counts.items() if v > 1)
    if duplicates:
        raise RegionalMappingError("Duplicate district_id rows in mapping: " + ", ".join(duplicates))

    if len(districts) != expected_district_total:
        raise RegionalMappingError(
            f"Mapping has {len(districts)} districts, expected {expected_district_total}."
        )

    subregion_counts = dict(Counter(item.subregion for item in districts))
    if len(subregion_counts) != expected_subregion_total:
        raise RegionalMappingError(
            f"Mapping has {len(subregion_counts)} subregions, expected {expected_subregion_total}: "
            + ", ".join(sorted(subregion_counts))
        )

    unexpected = sorted(set(subregion_counts) - set(audit_counts))
    missing = sorted(set(audit_counts) - set(subregion_counts))
    if unexpected or missing:
        raise RegionalMappingError(
            "Subregion names differ from the reviewed audit. "
            + (f"Unexpected: {unexpected}. " if unexpected else "")
            + (f"Missing: {missing}." if missing else "")
        )
    count_mismatches = {
        name: (subregion_counts[name], audit_counts[name])
        for name in audit_counts
        if subregion_counts[name] != audit_counts[name]
    }
    if count_mismatches:
        raise RegionalMappingError(
            "Subregion district counts differ from the reviewed audit (actual, expected): "
            + "; ".join(f"{k}={v[0]} (expected {v[1]})" for k, v in sorted(count_mismatches.items()))
        )

    if approved_district_ids is not None:
        approved = set(approved_district_ids)
        mapped = set(item.district_id for item in districts)
        unmapped = sorted(approved - mapped)
        if unmapped:
            raise RegionalMappingError(
                "Districts in the approved 146-district frame are unmapped: " + ", ".join(unmapped)
            )
        absent = sorted(mapped - approved)
        if absent:
            raise RegionalMappingError(
                "Mapping districts absent from the approved 146-district frame: " + ", ".join(absent)
            )

    return RegionalMapping(
        districts=tuple(districts),
        source_path=path,
        source_sha256=source_sha256,
        subregion_counts=subregion_counts,
    )


def write_region_mapping_csv(mapping: RegionalMapping, path: Path) -> None:
    """Write the experimental region_mapping.csv, preserving source values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REGION_MAPPING_EXPORT_COLUMNS))
        writer.writeheader()
        for item in sorted(mapping.districts, key=lambda d: (d.subregion, d.district_id)):
            writer.writerow(
                {
                    "district_id": item.district_id,
                    "district_name": item.district_name,
                    "dhis2_orgunit_id": item.dhis2_orgunit_id,
                    "dhis2_orgunit_name": item.dhis2_orgunit_name,
                    "subregion": item.subregion,
                }
            )
