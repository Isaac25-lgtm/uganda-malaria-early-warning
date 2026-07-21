"""Reference-data import and validation services.

The district reference table is the authority for all later analytical joins.
This module deliberately does not fabricate districts, DHIS2 IDs, UBOS codes or
geometry. It only validates curated files and safely upserts curated rows.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import func

from app.extensions import db
from app.models import DistrictReference, UBOSPopulation

DISTRICT_REFERENCE_COLUMNS = (
    "district_id",
    "district_name",
    "dhis2_orgunit_id",
    "dhis2_orgunit_name",
    "ubos_code",
    "region",
    "subregion",
    "active",
)
DISTRICT_ID_PATTERN = re.compile(r"^UGA_[A-Z0-9]+(?:_[A-Z0-9]+)*$")
DEFAULT_METADATA_PATH = Path("data/reference/district_frame_metadata.yml")
DEFAULT_RAW_BOUNDARY_PATH = Path("data/boundaries/UGANDA_DISTRICT.json")
DEFAULT_PREPARED_BOUNDARY_PATH = Path("data/boundaries/uganda_districts_prepared.geojson")
DEFAULT_UBOS_POPULATION_PATH = Path("data/reference/ubos_population.csv")
EXPECTED_ACTIVE_DISTRICTS = 146
UBOS_POPULATION_COLUMNS = (
    "district_id",
    "district_name",
    "year",
    "population",
    "source",
    "notes",
)


@dataclass
class ImportSummary:
    dry_run: bool = False
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    deactivated: int = 0
    warnings: list[str] = field(default_factory=list)
    mapping_changes: list[str] = field(default_factory=list)


@dataclass
class ValidationSummary:
    active_count: int
    expected_active_count: int | None
    metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class BoundaryValidationSummary:
    feature_count: int
    reference_count: int
    expected_feature_count: int | None
    missing_district_ids: list[str] = field(default_factory=list)
    extra_boundary_ids: list[str] = field(default_factory=list)
    duplicate_boundary_ids: list[str] = field(default_factory=list)
    missing_property_count: int = 0
    possible_name_matches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class RawBoundaryPreparationSummary:
    raw_path: Path
    output_path: Path
    feature_count: int = 0
    expected_feature_count: int = EXPECTED_ACTIVE_DISTRICTS
    district_name_field: str | None = None
    valid_geometry_count: int = 0
    invalid_geometry_count: int = 0
    matched_feature_count: int = 0
    unmatched_boundary_names: list[str] = field(default_factory=list)
    reference_names_without_boundary: list[str] = field(default_factory=list)
    duplicate_boundary_names: list[str] = field(default_factory=list)
    duplicate_reference_names: list[str] = field(default_factory=list)
    prepared_written: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class DistrictReferenceCsvGenerationSummary:
    raw_path: Path
    output_path: Path
    feature_count: int = 0
    expected_feature_count: int = EXPECTED_ACTIVE_DISTRICTS
    district_name_field: str | None = None
    valid_geometry_count: int = 0
    invalid_geometry_count: int = 0
    generated_row_count: int = 0
    duplicate_district_names: list[str] = field(default_factory=list)
    duplicate_district_ids: list[str] = field(default_factory=list)
    written: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class UBOSPopulationImportSummary:
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    row_count: int = 0
    years: list[int] = field(default_factory=list)
    unmatched_names: list[str] = field(default_factory=list)
    districts_without_population: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ReferenceDataError(ValueError):
    """Raised when a reference-data file is invalid."""


def blank_to_none(value: Any) -> str | None:
    """Normalize blank CSV values to None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_bool(value: Any, *, default: bool = True) -> bool:
    """Parse common CSV boolean values."""
    text = "" if value is None else str(value).strip().lower()
    if not text:
        return default
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    raise ReferenceDataError(f"Invalid active value {value!r}; use true/false.")


def validate_district_id(value: str) -> None:
    """Validate curated app-internal district_id format."""
    if not DISTRICT_ID_PATTERN.match(value):
        raise ReferenceDataError(
            f"Invalid district_id {value!r}. Use curated app IDs like "
            "UGA_PADER, UGA_GULU_CITY or UGA_KAMPALA. Do not derive IDs "
            "casually from district names."
        )


def district_id_from_name(value: str) -> str:
    """Create the deterministic Prompt-11 district_id from a boundary name."""
    name = blank_to_none(value)
    if not name:
        raise ReferenceDataError("District name is required to generate district_id.")
    cleaned = name.upper().strip()
    cleaned = re.sub(r"['’]", "", cleaned)
    cleaned = re.sub(r"[^A-Z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ReferenceDataError(f"District name {value!r} produced an empty ID.")
    district_id = f"UGA_{cleaned}"
    validate_district_id(district_id)
    return district_id


def generate_district_reference_csv_from_boundary(
    raw_path: Path = DEFAULT_RAW_BOUNDARY_PATH,
    output_path: Path = Path("data/reference/district_reference.csv"),
    *,
    overwrite: bool = False,
) -> DistrictReferenceCsvGenerationSummary:
    """Generate the canonical Prompt-11 district_reference CSV from boundaries."""
    summary = DistrictReferenceCsvGenerationSummary(
        raw_path=raw_path, output_path=output_path
    )
    if not raw_path.exists():
        summary.errors.append(f"Raw boundary file not found: {raw_path}")
        return summary
    if output_path.exists() and not overwrite:
        summary.errors.append(
            f"Output CSV already exists: {output_path}. Use --overwrite after review."
        )
        return summary

    payload = _read_geojson(raw_path)
    features = payload.get("features")
    if not isinstance(features, list):
        raise ReferenceDataError("Raw boundary FeatureCollection must contain features.")

    summary.feature_count = len(features)
    if summary.feature_count != EXPECTED_ACTIVE_DISTRICTS:
        summary.warnings.append(
            f"Raw boundary feature count is {summary.feature_count}, expected "
            f"{EXPECTED_ACTIVE_DISTRICTS}."
        )

    rows: list[dict[str, Any]] = []
    boundary_names: list[str] = []
    district_ids: list[str] = []
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            summary.errors.append(f"Feature {index} is not a GeoJSON feature object.")
            continue
        props = feature.get("properties")
        if not isinstance(props, dict):
            summary.errors.append(f"Feature {index} has missing properties.")
            continue
        if "District" not in props:
            summary.errors.append("Raw boundary file is missing the 'District' field.")
            continue
        summary.district_name_field = "District"
        district_name = blank_to_none(props.get("District"))
        if not district_name:
            summary.errors.append(f"Feature {index} has blank District name.")
            continue
        boundary_names.append(district_name)

        if _valid_polygon_geometry(feature.get("geometry")):
            summary.valid_geometry_count += 1
        else:
            summary.invalid_geometry_count += 1
            summary.errors.append(
                f"Feature {index} ({district_name}) has missing or non-Polygon geometry."
            )
            continue

        district_id = district_id_from_name(district_name)
        district_ids.append(district_id)
        rows.append(
            {
                "district_id": district_id,
                "district_name": district_name,
                "dhis2_orgunit_id": "",
                "dhis2_orgunit_name": "",
                "ubos_code": "",
                "region": "",
                "subregion": "",
                "active": "true",
            }
        )

    summary.duplicate_district_names = sorted(
        name
        for name, count in Counter(_normalize_name(name) for name in boundary_names).items()
        if count > 1
    )
    summary.duplicate_district_ids = sorted(
        district_id for district_id, count in Counter(district_ids).items() if count > 1
    )
    if summary.duplicate_district_names:
        summary.errors.append("Duplicate normalized District names exist in raw boundaries.")
    if summary.duplicate_district_ids:
        summary.errors.append(
            "Duplicate generated district_id values: "
            + ", ".join(summary.duplicate_district_ids)
        )

    summary.generated_row_count = len(rows)
    if not summary.errors and len(rows) == EXPECTED_ACTIVE_DISTRICTS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows.sort(key=lambda row: row["district_name"])
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DISTRICT_REFERENCE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        summary.written = True
    elif not summary.errors:
        summary.warnings.append(
            "District reference CSV was not written because the generated row count "
            "does not match the 146-district analysis frame."
        )
    return summary


def read_frame_metadata(path: Path = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    """Read a simple key/value YAML metadata file without adding a dependency."""
    if not path.exists():
        return {}

    metadata: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        cleaned = value.strip().strip('"').strip("'")
        if cleaned.isdigit():
            metadata[key.strip()] = int(cleaned)
        else:
            metadata[key.strip()] = cleaned
    return metadata


def _read_district_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ReferenceDataError(f"District reference CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ReferenceDataError("District reference CSV is empty.")

        missing_columns = [
            column for column in DISTRICT_REFERENCE_COLUMNS if column not in reader.fieldnames
        ]
        if missing_columns:
            raise ReferenceDataError(
                "District reference CSV is missing required columns: "
                + ", ".join(missing_columns)
            )

        rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(reader, start=2):
            district_id = blank_to_none(row.get("district_id"))
            district_name = blank_to_none(row.get("district_name"))
            if not district_id:
                raise ReferenceDataError(f"Line {line_number}: district_id is required.")
            if not district_name:
                raise ReferenceDataError(
                    f"Line {line_number}: district_name is required."
                )
            validate_district_id(district_id)

            rows.append(
                {
                    "district_id": district_id,
                    "district_name": district_name,
                    "dhis2_orgunit_id": blank_to_none(row.get("dhis2_orgunit_id")),
                    "dhis2_orgunit_name": blank_to_none(row.get("dhis2_orgunit_name")),
                    "ubos_code": blank_to_none(row.get("ubos_code")),
                    "region": blank_to_none(row.get("region")),
                    "subregion": blank_to_none(row.get("subregion")),
                    "active": parse_bool(row.get("active"), default=True),
                }
            )

    _validate_csv_uniqueness(rows)
    return rows


def _validate_csv_uniqueness(rows: list[dict[str, Any]]) -> None:
    for field_name in ("district_id", "district_name", "dhis2_orgunit_id", "ubos_code"):
        values = [row[field_name] for row in rows if row.get(field_name)]
        if field_name == "district_name":
            values = [_normalize_name(value) for value in values]
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            raise ReferenceDataError(
                f"Duplicate {field_name} values in CSV: " + ", ".join(duplicates)
            )


def import_districts(
    csv_path: Path,
    *,
    allow_mapping_changes: bool = False,
    deactivate_missing: bool = False,
    dry_run: bool = False,
) -> ImportSummary:
    """Safely upsert curated district reference rows by district_id."""
    rows = _read_district_csv(csv_path)
    summary = ImportSummary(dry_run=dry_run)

    if not rows:
        summary.warnings.append("CSV contained no district rows; nothing was imported.")
        return summary

    imported_ids = {row["district_id"] for row in rows}
    existing_by_id = {
        district.district_id: district
        for district in DistrictReference.query.filter(
            DistrictReference.district_id.in_(imported_ids)
        ).all()
    }

    for row in rows:
        district = existing_by_id.get(row["district_id"])
        if district is None:
            summary.inserted += 1
            if not dry_run:
                db.session.add(DistrictReference(**row))
            if not row.get("dhis2_orgunit_id"):
                summary.warnings.append(
                    f"{row['district_id']}: missing optional dhis2_orgunit_id"
                )
            if not row.get("ubos_code"):
                summary.warnings.append(f"{row['district_id']}: missing optional ubos_code")
            continue

        changes: dict[str, Any] = {}
        blocked_mapping_change = False
        blocked_mapping_fields: set[str] = set()
        for field_name, new_value in row.items():
            old_value = getattr(district, field_name)
            if field_name in {"dhis2_orgunit_id", "ubos_code"}:
                if old_value and old_value != new_value:
                    message = (
                        f"{row['district_id']}: {field_name} mapping change "
                        f"{old_value!r} -> {new_value!r}"
                    )
                    summary.mapping_changes.append(message)
                    if not allow_mapping_changes:
                        blocked_mapping_change = True
                        blocked_mapping_fields.add(field_name)
                        continue
            if (
                field_name == "dhis2_orgunit_name"
                and "dhis2_orgunit_id" in blocked_mapping_fields
            ):
                continue
            if old_value != new_value:
                changes[field_name] = new_value

        if blocked_mapping_change:
            summary.warnings.append(
                f"{row['district_id']}: mapping change blocked; rerun with "
                "--allow-mapping-changes to apply it."
            )

        if changes:
            summary.updated += 1
            if not dry_run:
                for field_name, new_value in changes.items():
                    setattr(district, field_name, new_value)
        else:
            summary.skipped += 1

    if deactivate_missing:
        missing_query = DistrictReference.query.filter(
            DistrictReference.active.is_(True),
            ~DistrictReference.district_id.in_(imported_ids),
        )
        for district in missing_query.all():
            summary.deactivated += 1
            if not dry_run:
                district.active = False
            summary.warnings.append(
                f"{district.district_id}: deactivated because it is absent from CSV"
            )
    else:
        missing_active = DistrictReference.query.filter(
            DistrictReference.active.is_(True),
            ~DistrictReference.district_id.in_(imported_ids),
        ).count()
        if missing_active:
            summary.warnings.append(
                f"{missing_active} active district(s) are absent from the CSV. "
                "They were not deleted or deactivated."
            )

    if not dry_run:
        db.session.commit()
    return summary


def validate_districts(
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> ValidationSummary:
    """Validate the current district_reference table and frame metadata."""
    metadata = read_frame_metadata(metadata_path)
    expected = metadata.get("expected_active_districts")
    if expected is None:
        expected = EXPECTED_ACTIVE_DISTRICTS

    active_count = DistrictReference.query.filter_by(active=True).count()
    summary = ValidationSummary(
        active_count=active_count,
        expected_active_count=expected,
        metadata=metadata,
    )

    if active_count != expected:
        summary.warnings.append(
            f"Active district count is {active_count}, expected {expected}. "
            "This is allowed during early development but final modelling should "
            "use the expected analysis frame unless the research design changes."
        )

    if not metadata_path.exists():
        summary.warnings.append(f"Frame metadata file not found: {metadata_path}")
    if not metadata.get("reference_date"):
        summary.warnings.append("District-frame reference_date is missing.")

    _append_duplicate_query_warnings(
        summary.warnings,
        "dhis2_orgunit_id",
        DistrictReference.dhis2_orgunit_id,
    )
    _append_duplicate_query_warnings(summary.warnings, "ubos_code", DistrictReference.ubos_code)

    missing_names = DistrictReference.query.filter(
        func.coalesce(DistrictReference.district_name, "") == ""
    ).count()
    if missing_names:
        summary.errors.append(f"{missing_names} district(s) have missing district_name.")

    duplicate_ids = (
        db.session.query(DistrictReference.district_id)
        .group_by(DistrictReference.district_id)
        .having(func.count(DistrictReference.id) > 1)
        .all()
    )
    if duplicate_ids:
        summary.errors.append(
            "Duplicate district_id values found: "
            + ", ".join(row[0] for row in duplicate_ids)
        )

    return summary


def prepare_raw_boundaries(
    raw_path: Path = DEFAULT_RAW_BOUNDARY_PATH,
    output_path: Path = DEFAULT_PREPARED_BOUNDARY_PATH,
) -> RawBoundaryPreparationSummary:
    """Validate raw Uganda district boundaries and prepare district_id GeoJSON."""
    summary = RawBoundaryPreparationSummary(raw_path=raw_path, output_path=output_path)
    if not raw_path.exists():
        summary.errors.append(f"Raw boundary file not found: {raw_path}")
        return summary

    payload = _read_geojson(raw_path)
    features = payload.get("features")
    if not isinstance(features, list):
        raise ReferenceDataError("Raw boundary FeatureCollection must contain features.")

    summary.feature_count = len(features)
    if summary.feature_count != EXPECTED_ACTIVE_DISTRICTS:
        summary.warnings.append(
            f"Raw boundary feature count is {summary.feature_count}, expected "
            f"{EXPECTED_ACTIVE_DISTRICTS}."
        )

    reference_rows = DistrictReference.query.filter_by(active=True).all()
    if not reference_rows:
        summary.warnings.append(
            "district_reference is empty; names can be validated but prepared "
            "district_id GeoJSON cannot be written yet."
        )

    reference_by_name: dict[str, DistrictReference] = {}
    reference_name_counts = Counter(_normalize_name(row.district_name) for row in reference_rows)
    summary.duplicate_reference_names = sorted(
        row.district_name
        for row in reference_rows
        if reference_name_counts[_normalize_name(row.district_name)] > 1
    )
    if summary.duplicate_reference_names:
        summary.errors.append(
            "Duplicate normalized district names exist in district_reference."
        )
    else:
        reference_by_name = {
            _normalize_name(row.district_name): row for row in reference_rows
        }

    boundary_names: list[str] = []
    prepared_features: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            summary.errors.append(f"Feature {index} is not a GeoJSON feature object.")
            continue
        props = feature.get("properties")
        if not isinstance(props, dict):
            summary.errors.append(f"Feature {index} has missing properties.")
            continue

        if "District" not in props:
            summary.errors.append("Raw boundary file is missing the 'District' field.")
            continue
        summary.district_name_field = "District"
        raw_name = blank_to_none(props.get("District"))
        if raw_name is None:
            summary.errors.append(f"Feature {index} has blank District name.")
            continue
        boundary_names.append(raw_name)

        if _valid_polygon_geometry(feature.get("geometry")):
            summary.valid_geometry_count += 1
        else:
            summary.invalid_geometry_count += 1
            summary.errors.append(
                f"Feature {index} ({raw_name}) has missing or non-Polygon geometry."
            )
            continue

        reference = reference_by_name.get(_normalize_name(raw_name))
        if reference is None:
            summary.unmatched_boundary_names.append(raw_name)
            continue

        summary.matched_feature_count += 1
        prepared_features.append(
            {
                "type": "Feature",
                "properties": {
                    "district_id": reference.district_id,
                    "district_name": reference.district_name,
                    "original_District": raw_name,
                },
                "geometry": feature["geometry"],
            }
        )

    summary.duplicate_boundary_names = sorted(
        name for name, count in Counter(_normalize_name(name) for name in boundary_names).items()
        if count > 1
    )
    if summary.duplicate_boundary_names:
        summary.errors.append("Duplicate normalized District names exist in raw boundaries.")

    boundary_name_set = {_normalize_name(name) for name in boundary_names}
    summary.reference_names_without_boundary = sorted(
        row.district_name
        for row in reference_rows
        if _normalize_name(row.district_name) not in boundary_name_set
    )
    if summary.unmatched_boundary_names:
        summary.warnings.append(
            f"{len(summary.unmatched_boundary_names)} boundary District name(s) "
            "do not match active district_reference names."
        )
    if summary.reference_names_without_boundary:
        summary.warnings.append(
            f"{len(summary.reference_names_without_boundary)} active district_reference "
            "name(s) were not found in raw boundaries."
        )

    can_write = (
        not summary.errors
        and bool(reference_rows)
        and not summary.unmatched_boundary_names
        and not summary.reference_names_without_boundary
        and len(prepared_features) == EXPECTED_ACTIVE_DISTRICTS
    )
    if can_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prepared = {
            "type": "FeatureCollection",
            "name": "uganda_districts_prepared",
            "features": prepared_features,
        }
        output_path.write_text(json.dumps(prepared, ensure_ascii=True), encoding="utf-8")
        summary.prepared_written = True
    elif not reference_rows:
        summary.warnings.append(
            "Prepared GeoJSON was not written because district_reference must be "
            "loaded before boundary names can be mapped to canonical district_id."
        )
    elif not summary.errors:
        summary.warnings.append(
            "Prepared GeoJSON was not written because boundary/reference name sets "
            "are not a complete 146-feature match."
        )

    return summary


def import_ubos_population(
    csv_path: Path = DEFAULT_UBOS_POPULATION_PATH,
    *,
    dry_run: bool = False,
) -> UBOSPopulationImportSummary:
    """Validate and optionally import approved UBOS district-year populations."""
    rows, summary = _read_ubos_population_csv(csv_path)
    if summary.errors:
        return summary

    if dry_run:
        summary.inserted = len(rows)
        return summary

    for row in rows:
        existing = UBOSPopulation.query.filter_by(
            district_id=row["district_id"], year=row["year"]
        ).first()
        if existing is None:
            db.session.add(UBOSPopulation(**row))
            summary.inserted += 1
            continue

        changes = {
            field_name: value
            for field_name, value in row.items()
            if getattr(existing, field_name) != value
        }
        if changes:
            for field_name, value in changes.items():
                setattr(existing, field_name, value)
            summary.updated += 1
        else:
            summary.skipped += 1

    db.session.commit()
    return summary


def _append_duplicate_query_warnings(
    warnings: list[str], label: str, column: Any
) -> None:
    rows = (
        db.session.query(column, func.count(DistrictReference.id))
        .filter(column.isnot(None), column != "")
        .group_by(column)
        .having(func.count(DistrictReference.id) > 1)
        .all()
    )
    for value, count in rows:
        warnings.append(f"Duplicate non-empty {label} {value!r} appears {count} times.")


def validate_boundaries(
    geojson_path: Path,
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> BoundaryValidationSummary:
    """Validate prepared district GeoJSON against active district_reference IDs."""
    if not geojson_path.exists():
        raise ReferenceDataError(f"Boundary GeoJSON not found: {geojson_path}")

    try:
        payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReferenceDataError(f"Invalid GeoJSON JSON: {exc}") from exc

    if payload.get("type") != "FeatureCollection":
        raise ReferenceDataError("Boundary file must be a GeoJSON FeatureCollection.")

    features = payload.get("features")
    if not isinstance(features, list):
        raise ReferenceDataError("GeoJSON FeatureCollection must contain a features list.")

    metadata = read_frame_metadata(metadata_path)
    expected = metadata.get("expected_active_districts", EXPECTED_ACTIVE_DISTRICTS)
    feature_ids: list[str] = []
    missing_property_count = 0
    feature_names_by_id: dict[str, str] = {}
    for index, feature in enumerate(features, start=1):
        props = feature.get("properties") if isinstance(feature, dict) else None
        if not isinstance(props, dict):
            missing_property_count += 1
            continue
        district_id = blank_to_none(props.get("district_id"))
        if not district_id:
            missing_property_count += 1
            continue
        feature_ids.append(district_id)
        feature_name = _extract_feature_name(props)
        if feature_name:
            feature_names_by_id[district_id] = feature_name

    duplicate_ids = sorted(
        district_id for district_id, count in Counter(feature_ids).items() if count > 1
    )
    boundary_id_set = set(feature_ids)
    reference_rows = DistrictReference.query.filter_by(active=True).all()
    reference_id_set = {row.district_id for row in reference_rows}
    reference_names = {
        _normalize_name(row.district_name): row.district_id for row in reference_rows
    }

    missing_ids = sorted(reference_id_set - boundary_id_set)
    extra_ids = sorted(boundary_id_set - reference_id_set)
    possible_name_matches = []
    for boundary_id in extra_ids:
        feature_name = feature_names_by_id.get(boundary_id)
        normalized = _normalize_name(feature_name)
        if normalized and normalized in reference_names:
            possible_name_matches.append(
                f"Boundary {boundary_id} name {feature_name!r} matches reference "
                f"{reference_names[normalized]} but district_id differs."
            )

    summary = BoundaryValidationSummary(
        feature_count=len(features),
        reference_count=len(reference_id_set),
        expected_feature_count=expected,
        missing_district_ids=missing_ids,
        extra_boundary_ids=extra_ids,
        duplicate_boundary_ids=duplicate_ids,
        missing_property_count=missing_property_count,
        possible_name_matches=possible_name_matches,
    )

    if len(features) != expected:
        summary.warnings.append(
            f"GeoJSON feature count is {len(features)}, expected {expected}."
        )
    if missing_property_count:
        summary.errors.append(
            f"{missing_property_count} feature(s) are missing properties.district_id."
        )
    if duplicate_ids:
        summary.errors.append(
            "Duplicate district_id values in GeoJSON: " + ", ".join(duplicate_ids)
        )
    if len(features) == expected and (missing_ids or extra_ids):
        summary.warnings.append(
            "Feature count matches expected frame, but district_id sets differ. "
            "This suggests a possible boundary vintage mismatch."
        )
    if possible_name_matches:
        summary.warnings.append(
            "Some boundary names suggest renamed/split/merged districts or mismatched IDs."
        )
    if missing_ids:
        summary.warnings.append(
            f"{len(missing_ids)} active district_reference ID(s) lack boundaries."
        )
    if extra_ids:
        summary.warnings.append(
            f"{len(extra_ids)} GeoJSON district_id value(s) lack active district_reference matches."
        )

    return summary


def _read_geojson(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReferenceDataError(f"Invalid GeoJSON JSON: {exc}") from exc

    if payload.get("type") != "FeatureCollection":
        raise ReferenceDataError("Boundary file must be a GeoJSON FeatureCollection.")
    return payload


def _valid_polygon_geometry(geometry: Any) -> bool:
    if not isinstance(geometry, dict):
        return False
    geometry_type = geometry.get("type")
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        return False
    coordinates = geometry.get("coordinates")
    return isinstance(coordinates, list) and bool(coordinates)


def _read_ubos_population_csv(
    path: Path,
) -> tuple[list[dict[str, Any]], UBOSPopulationImportSummary]:
    summary = UBOSPopulationImportSummary()
    if not path.exists():
        summary.errors.append(f"UBOS population CSV not found: {path}")
        return [], summary

    reference_rows = DistrictReference.query.filter_by(active=True).all()
    reference_by_id = {row.district_id: row for row in reference_rows}
    reference_by_name: dict[str, DistrictReference] = {}
    duplicate_reference_names = {
        name
        for name, count in Counter(
            _normalize_name(row.district_name) for row in reference_rows
        ).items()
        if count > 1
    }
    if duplicate_reference_names:
        summary.errors.append(
            "Cannot safely match UBOS district names because district_reference "
            "contains duplicate normalized district names."
        )
    else:
        reference_by_name = {
            _normalize_name(row.district_name): row for row in reference_rows
        }

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            summary.errors.append("UBOS population CSV is empty.")
            return [], summary
        missing_columns = [
            column for column in UBOS_POPULATION_COLUMNS if column not in reader.fieldnames
        ]
        if missing_columns:
            summary.errors.append(
                "UBOS population CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
            return [], summary

        rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int]] = set()
        source_years: set[int] = set()
        populated_district_ids: set[str] = set()

        for line_number, csv_row in enumerate(reader, start=2):
            district_id = blank_to_none(csv_row.get("district_id"))
            district_name = blank_to_none(csv_row.get("district_name"))
            year_raw = blank_to_none(csv_row.get("year"))
            population_raw = blank_to_none(csv_row.get("population"))
            source = blank_to_none(csv_row.get("source"))
            notes = blank_to_none(csv_row.get("notes"))

            if not district_name:
                summary.errors.append(f"Line {line_number}: district_name is required.")
                continue
            if not year_raw:
                summary.errors.append(f"Line {line_number}: year is required.")
                continue
            if not population_raw:
                summary.errors.append(f"Line {line_number}: population is required.")
                continue
            if not source:
                summary.errors.append(f"Line {line_number}: source is required.")
                continue
            if "ubos" not in source.lower():
                summary.warnings.append(
                    f"Line {line_number}: source {source!r} does not contain 'UBOS'."
                )

            try:
                year = int(year_raw)
            except ValueError:
                summary.errors.append(f"Line {line_number}: year must be an integer.")
                continue
            try:
                population = int(str(population_raw).replace(",", ""))
            except ValueError:
                summary.errors.append(
                    f"Line {line_number}: population must be a positive integer."
                )
                continue
            if population <= 0:
                summary.errors.append(
                    f"Line {line_number}: population must be greater than zero."
                )
                continue

            reference = None
            if district_id:
                validate_district_id(district_id)
                reference = reference_by_id.get(district_id)
                if reference is None:
                    summary.errors.append(
                        f"Line {line_number}: district_id {district_id!r} is not in "
                        "active district_reference."
                    )
                    continue
                if _normalize_name(reference.district_name) != _normalize_name(district_name):
                    summary.warnings.append(
                        f"Line {line_number}: district_name {district_name!r} differs "
                        f"from reference name {reference.district_name!r} for {district_id}."
                    )
            else:
                reference = reference_by_name.get(_normalize_name(district_name))
                if reference is None:
                    summary.unmatched_names.append(district_name)
                    continue

            key = (reference.district_id, year)
            if key in seen_keys:
                summary.errors.append(
                    f"Line {line_number}: duplicate population row for "
                    f"{reference.district_id} in {year}."
                )
                continue
            seen_keys.add(key)
            source_years.add(year)
            populated_district_ids.add(reference.district_id)
            rows.append(
                {
                    "district_id": reference.district_id,
                    "district_name": reference.district_name,
                    "year": year,
                    "population": population,
                    "source": source,
                    "notes": notes,
                }
            )

    summary.row_count = len(rows)
    summary.years = sorted(source_years)
    if summary.unmatched_names:
        summary.errors.append(
            "Unmatched UBOS district names: " + ", ".join(sorted(summary.unmatched_names))
        )
    if reference_rows and source_years:
        summary.districts_without_population = sorted(
            row.district_name
            for row in reference_rows
            if row.district_id not in populated_district_ids
        )
        if summary.districts_without_population:
            summary.warnings.append(
                f"{len(summary.districts_without_population)} active district(s) "
                "do not have population rows in this CSV."
            )
    if not reference_rows:
        summary.warnings.append(
            "district_reference is empty; UBOS population names cannot be mapped "
            "to canonical district_id yet."
        )
    return rows, summary


def _extract_feature_name(props: dict[str, Any]) -> str | None:
    for key in ("district_name", "name", "NAME", "DNAME", "DISTRICT", "district"):
        value = blank_to_none(props.get(key))
        if value:
            return value
    return None


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())
