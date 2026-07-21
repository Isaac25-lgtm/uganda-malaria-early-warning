"""DHIS2 district org-unit mapping and week-convention verification.

This module updates only DHIS2 mapping fields on ``district_reference``. It
does not extract malaria case data, preprocess raw rows, or create analytical
features.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.extensions import db
from app.models import DistrictReference
from app.services.dhis2_client import (
    DHIS2Client,
    DHIS2ClientError,
    DHIS2HTTPError,
    read_district_level_from_env,
    read_required_env,
)
from app.services.preprocessing import iso_week_dates
from app.services.reference_data import (
    DEFAULT_METADATA_PATH,
    EXPECTED_ACTIVE_DISTRICTS,
    blank_to_none,
    read_frame_metadata,
)

DEFAULT_ALIAS_PATH = Path("data/reference/dhis2_orgunit_name_aliases.csv")
DEFAULT_MAPPING_REVIEW_PATH = Path("data/reference/dhis2_orgunit_mapping_review.csv")


class DHIS2MappingError(RuntimeError):
    """Raised when DHIS2 district mapping cannot safely continue."""


@dataclass(frozen=True)
class OrgUnitMatch:
    district_id: str
    district_name: str
    dhis2_orgunit_id: str
    dhis2_orgunit_name: str
    match_type: str
    parent_name: str | None = None
    child_sample: str | None = None
    alias_reason: str | None = None
    existing_dhis2_orgunit_id: str | None = None
    existing_dhis2_orgunit_name: str | None = None


@dataclass
class OrgUnitMappingSummary:
    dry_run: bool
    level: int
    org_units_returned: int
    active_reference_count: int
    proposed_updates: list[OrgUnitMatch] = field(default_factory=list)
    exact_matches: int = 0
    normalized_matches: int = 0
    alias_matches: int = 0
    rows_updated: int = 0
    unmatched_dhis2_org_units: list[str] = field(default_factory=list)
    unmatched_districts: list[str] = field(default_factory=list)
    duplicate_dhis2_uids: list[str] = field(default_factory=list)
    duplicate_dhis2_names: list[str] = field(default_factory=list)
    duplicate_matched_district_ids: list[str] = field(default_factory=list)
    duplicate_matched_orgunit_ids: list[str] = field(default_factory=list)
    mapping_changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    review_csv_path: Path | None = None

    @property
    def matched_count(self) -> int:
        return len(self.proposed_updates)

    @property
    def one_to_one_passed(self) -> bool:
        return (
            self.active_reference_count == EXPECTED_ACTIVE_DISTRICTS
            and self.org_units_returned == EXPECTED_ACTIVE_DISTRICTS
            and self.matched_count == EXPECTED_ACTIVE_DISTRICTS
            and not self.unmatched_dhis2_org_units
            and not self.unmatched_districts
            and not self.duplicate_dhis2_uids
            and not self.duplicate_dhis2_names
            and not self.duplicate_matched_district_ids
            and not self.duplicate_matched_orgunit_ids
        )


@dataclass
class OrgUnitLevelSummary:
    levels: list[dict[str, Any]]
    counts_by_level: dict[int, int]
    requested_level: int | None = None


@dataclass
class WeekPeriodVerificationSummary:
    period: str
    app_start_date: date
    app_end_date: date
    weekly_iso_format: str | None = None
    weekly_iso_duration: str | None = None
    dhis2_start_date: str | None = None
    dhis2_end_date: str | None = None
    analytics_row_count: int | None = None
    status: str = "unverified"
    warnings: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.status.startswith("dhis2_weekly_iso_confirmed")


@dataclass
class WeekSchemaSummary:
    district_week_features_allows_53: bool
    district_week_climate_allows_53: bool
    raw_dhis2_malaria_allows_53: bool
    details: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.district_week_features_allows_53
            and self.district_week_climate_allows_53
            and self.raw_dhis2_malaria_allows_53
        )


def load_aliases(path: Path = DEFAULT_ALIAS_PATH) -> dict[str, tuple[str, str]]:
    """Load reviewed DHIS2-name aliases keyed by normalized DHIS2 base name."""
    aliases: dict[str, tuple[str, str]] = {}
    if not path.exists():
        return aliases

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dhis2_name", "district_name", "reason"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise DHIS2MappingError(
                f"Alias CSV {path} must contain columns: dhis2_name, district_name, reason."
            )
        for line_number, row in enumerate(reader, start=2):
            dhis2_name = blank_to_none(row.get("dhis2_name"))
            district_name = blank_to_none(row.get("district_name"))
            reason = blank_to_none(row.get("reason")) or "reviewed alias"
            if not dhis2_name or not district_name:
                raise DHIS2MappingError(
                    f"Alias CSV {path} line {line_number}: dhis2_name and "
                    "district_name are required."
                )
            aliases[_match_key(_strip_dhis2_district_suffix(dhis2_name))] = (
                _exact_key(district_name),
                reason,
            )
    return aliases


def build_mapping_summary(
    org_units: list[dict[str, Any]],
    reference_rows: list[DistrictReference],
    *,
    level: int,
    alias_path: Path = DEFAULT_ALIAS_PATH,
    dry_run: bool = True,
) -> OrgUnitMappingSummary:
    aliases = load_aliases(alias_path)
    active_reference_rows = [row for row in reference_rows if row.active]
    summary = OrgUnitMappingSummary(
        dry_run=dry_run,
        level=level,
        org_units_returned=len(org_units),
        active_reference_count=len(active_reference_rows),
    )

    org_ids = [str(org_unit.get("id") or "") for org_unit in org_units]
    summary.duplicate_dhis2_uids = sorted(
        value for value, count in Counter(org_ids).items() if value and count > 1
    )
    org_name_keys = [
        _match_key(_org_unit_name(org_unit)) for org_unit in org_units if _org_unit_name(org_unit)
    ]
    summary.duplicate_dhis2_names = sorted(
        value for value, count in Counter(org_name_keys).items() if value and count > 1
    )

    reference_by_exact = {_exact_key(row.district_name): row for row in active_reference_rows}
    reference_by_match_key = _unique_reference_key_map(active_reference_rows)
    matched_district_ids: list[str] = []
    matched_orgunit_ids: list[str] = []
    matched_orgunit_source: dict[str, str] = {}

    for org_unit in org_units:
        orgunit_id = blank_to_none(org_unit.get("id"))
        orgunit_name = _org_unit_name(org_unit)
        if not orgunit_id or not orgunit_name:
            summary.unmatched_dhis2_org_units.append(
                f"{orgunit_id or 'missing-id'} | {orgunit_name or 'missing-name'}"
            )
            continue

        reference: DistrictReference | None = None
        match_type = ""
        alias_reason = None
        exact_key = _exact_key(orgunit_name)
        if exact_key in reference_by_exact:
            reference = reference_by_exact[exact_key]
            match_type = "exact"
        else:
            stripped_key = _match_key(_strip_dhis2_district_suffix(orgunit_name))
            reference = reference_by_match_key.get(stripped_key)
            if reference is not None:
                match_type = "normalized"
            elif stripped_key in aliases:
                reference_name_key, alias_reason = aliases[stripped_key]
                reference = reference_by_exact.get(reference_name_key)
                if reference is None:
                    reference = reference_by_match_key.get(_match_key(reference_name_key))
                if reference is not None:
                    match_type = "alias"

        if reference is None:
            summary.unmatched_dhis2_org_units.append(f"{orgunit_id} | {orgunit_name}")
            continue

        matched_district_ids.append(reference.district_id)
        matched_orgunit_ids.append(orgunit_id)
        matched_orgunit_source[reference.district_id] = orgunit_id
        if match_type == "exact":
            summary.exact_matches += 1
        elif match_type == "normalized":
            summary.normalized_matches += 1
        elif match_type == "alias":
            summary.alias_matches += 1

        summary.proposed_updates.append(
            OrgUnitMatch(
                district_id=reference.district_id,
                district_name=reference.district_name,
                dhis2_orgunit_id=orgunit_id,
                dhis2_orgunit_name=orgunit_name,
                match_type=match_type,
                parent_name=_parent_name(org_unit),
                child_sample=_child_sample(org_unit),
                alias_reason=alias_reason,
                existing_dhis2_orgunit_id=reference.dhis2_orgunit_id,
                existing_dhis2_orgunit_name=reference.dhis2_orgunit_name,
            )
        )

    summary.duplicate_matched_district_ids = sorted(
        value for value, count in Counter(matched_district_ids).items() if count > 1
    )
    summary.duplicate_matched_orgunit_ids = sorted(
        value for value, count in Counter(matched_orgunit_ids).items() if count > 1
    )
    summary.unmatched_districts = sorted(
        f"{row.district_id} | {row.district_name}"
        for row in active_reference_rows
        if row.district_id not in matched_orgunit_source
    )
    for match in summary.proposed_updates:
        if match.existing_dhis2_orgunit_id and (
            match.existing_dhis2_orgunit_id != match.dhis2_orgunit_id
        ):
            summary.mapping_changes.append(
                f"{match.district_id}: dhis2_orgunit_id "
                f"{match.existing_dhis2_orgunit_id!r} -> {match.dhis2_orgunit_id!r}"
            )

    if summary.active_reference_count != EXPECTED_ACTIVE_DISTRICTS:
        summary.warnings.append(
            f"Active district_reference count is {summary.active_reference_count}, "
            f"expected {EXPECTED_ACTIVE_DISTRICTS}."
        )
    if summary.org_units_returned != EXPECTED_ACTIVE_DISTRICTS:
        summary.warnings.append(
            f"DHIS2 LEVEL-{level} returned {summary.org_units_returned} org units, "
            f"expected {EXPECTED_ACTIVE_DISTRICTS}; this may not be the district level."
        )
    return summary


def map_district_orgunits(
    client: DHIS2Client,
    *,
    level: int | None = None,
    dry_run: bool = True,
    allow_mapping_changes: bool = False,
    alias_path: Path = DEFAULT_ALIAS_PATH,
    review_csv_path: Path = DEFAULT_MAPPING_REVIEW_PATH,
) -> OrgUnitMappingSummary:
    selected_level = level or read_district_level_from_env()
    org_units = client.fetch_org_units_at_level(selected_level)
    reference_rows = DistrictReference.query.order_by(DistrictReference.district_name).all()
    summary = build_mapping_summary(
        org_units,
        reference_rows,
        level=selected_level,
        alias_path=alias_path,
        dry_run=dry_run,
    )
    _write_mapping_review_csv(summary, review_csv_path)
    summary.review_csv_path = review_csv_path

    if not summary.one_to_one_passed:
        raise DHIS2MappingError(
            "One-to-one DHIS2 district mapping was not proven. Review "
            f"{review_csv_path} and resolve unmatched/duplicate mappings before writing."
        )
    if summary.mapping_changes and not allow_mapping_changes:
        raise DHIS2MappingError(
            "Existing DHIS2 org-unit mapping changes were detected. Review the "
            "mapping changes and rerun with --allow-mapping-changes only after approval."
        )
    if dry_run:
        return summary

    rows_by_id = {
        row.district_id: row
        for row in DistrictReference.query.filter(
            DistrictReference.district_id.in_(
                [match.district_id for match in summary.proposed_updates]
            )
        ).all()
    }
    for match in summary.proposed_updates:
        district = rows_by_id[match.district_id]
        if (
            district.dhis2_orgunit_id != match.dhis2_orgunit_id
            or district.dhis2_orgunit_name != match.dhis2_orgunit_name
        ):
            district.dhis2_orgunit_id = match.dhis2_orgunit_id
            district.dhis2_orgunit_name = match.dhis2_orgunit_name
            summary.rows_updated += 1

    db.session.commit()
    update_frame_metadata(
        {
            "district_frame_source": "data/boundaries/UGANDA_DISTRICT.json",
            "active_districts_cities": EXPECTED_ACTIVE_DISTRICTS,
            "district_id_format": "UGA_<NORMALISED_NAME>",
            "population_denominator_assumption": (
                "UBOS 2024 fixed denominator across 2020-2025 for proof-of-concept "
                "incidence standardisation"
            ),
            "dhis2_orgunit_level": selected_level,
            "dhis2_mapping_status": f"mapped_{summary.matched_count}_of_{EXPECTED_ACTIVE_DISTRICTS}",
            "dhis2_mapping_date": date.today().isoformat(),
        }
    )
    return summary


def inspect_orgunit_levels(
    client: DHIS2Client,
    *,
    requested_level: int | None = None,
) -> OrgUnitLevelSummary:
    levels = client.fetch_org_unit_levels()
    counts: dict[int, int] = {}
    for level_info in levels:
        raw_level = level_info.get("level")
        if raw_level is None:
            continue
        try:
            level = int(raw_level)
        except (TypeError, ValueError):
            continue
        try:
            counts[level] = len(client.fetch_org_units_at_level(level))
        except DHIS2ClientError:
            counts[level] = -1
    return OrgUnitLevelSummary(
        levels=sorted(levels, key=lambda item: int(item.get("level") or 0)),
        counts_by_level=counts,
        requested_level=requested_level,
    )


def verify_week_period(
    client: DHIS2Client,
    *,
    period: str,
    update_metadata: bool = True,
) -> WeekPeriodVerificationSummary:
    year, week = _parse_week_period(period)
    app_start, app_end = iso_week_dates(year, week)
    summary = WeekPeriodVerificationSummary(
        period=f"{year}W{week:02d}",
        app_start_date=app_start,
        app_end_date=app_end,
    )

    weekly_type = _weekly_period_type(client.fetch_period_types())
    if weekly_type:
        summary.weekly_iso_format = weekly_type.get("isoFormat")
        summary.weekly_iso_duration = weekly_type.get("isoDuration")

    start_date, end_date, date_warning = _try_fetch_period_dates(client, summary.period)
    summary.dhis2_start_date = start_date
    summary.dhis2_end_date = end_date
    if date_warning:
        summary.warnings.append(date_warning)

    try:
        dx = read_required_env("DHIS2_CONFIRMED_MALARIA_DX")
        level = read_district_level_from_env()
        payload = client.analytics_query(dx=dx, periods=[summary.period], level=level)
        rows = payload.get("rows") or []
        summary.analytics_row_count = len(rows) if isinstance(rows, list) else None
    except DHIS2ClientError as exc:
        summary.warnings.append(
            f"Analytics smoke test for {summary.period} failed: {exc}"
        )

    if start_date and end_date:
        if start_date == app_start.isoformat() and end_date == app_end.isoformat():
            summary.status = "dhis2_weekly_iso_confirmed_by_explicit_dates"
        else:
            summary.status = "mismatch"
            summary.warnings.append(
                "DHIS2 period dates do not match the app ISO Monday-Sunday dates."
            )
    elif (
        summary.weekly_iso_format == "yyyyWn"
        and summary.weekly_iso_duration == "P7D"
        and summary.analytics_row_count is not None
    ):
        summary.status = "dhis2_weekly_iso_confirmed_by_live_period_type_metadata"
        summary.warnings.append(
            "This DHIS2 instance did not expose explicit weekly start/end dates; "
            "confirmation is based on live Weekly period type metadata "
            "(isoFormat yyyyWn, isoDuration P7D) and analytics accepting the period."
        )
    else:
        summary.status = "unverified"
        summary.warnings.append(
            "DHIS2 weekly start/end dates were not verified. Do not preprocess "
            "district-week features until the week convention is resolved."
        )

    if update_metadata:
        update_frame_metadata(
            {
                "week_convention_status": summary.status,
                "week_convention_last_checked_period": summary.period,
                "week_convention_last_checked_date": date.today().isoformat(),
                "week_convention_notes": (
                    "App uses ISO-8601 Monday-start weeks; live DHIS2 Weekly "
                    f"period metadata reports isoFormat={summary.weekly_iso_format or 'unknown'} "
                    f"and isoDuration={summary.weekly_iso_duration or 'unknown'}."
                ),
            }
        )
    return summary


def check_week_53_schema() -> WeekSchemaSummary:
    """Check DB check constraints that must allow epi_week 1-53."""
    constraint_rows = db.session.execute(
        text(
            """
            SELECT conname, pg_get_constraintdef(oid) AS definition
            FROM pg_constraint
            WHERE conname IN (
                'ck_dwf_epi_week',
                'ck_dwc_epi_week',
                'ck_raw_dhis2_epi_week'
            )
            ORDER BY conname
            """
        )
    ).mappings()
    constraints = {row["conname"]: row["definition"] for row in constraint_rows}

    def allows_53(name: str) -> bool:
        definition = constraints.get(name, "")
        compact = re.sub(r"\s+", " ", definition)
        return "53" in compact and "epi_week" in compact

    details = [f"{name}: {definition}" for name, definition in constraints.items()]
    return WeekSchemaSummary(
        district_week_features_allows_53=allows_53("ck_dwf_epi_week"),
        district_week_climate_allows_53=allows_53("ck_dwc_epi_week"),
        raw_dhis2_malaria_allows_53=allows_53("ck_raw_dhis2_epi_week"),
        details=details,
    )


def week_alignment_is_verified(
    metadata_path: Path = DEFAULT_METADATA_PATH,
) -> tuple[bool, str]:
    metadata = read_frame_metadata(metadata_path)
    status = str(metadata.get("week_convention_status") or "").strip()
    verified = status.startswith("dhis2_weekly_iso_confirmed")
    return verified, status or "missing"


def update_frame_metadata(
    updates: dict[str, Any],
    *,
    path: Path = DEFAULT_METADATA_PATH,
) -> None:
    """Update the project's flat key:value metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    rendered_updates = {key: _metadata_value(value) for key, value in updates.items()}
    seen: set[str] = set()
    output_lines: list[str] = []

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in line:
            output_lines.append(line)
            continue
        key = line.split(":", 1)[0].strip()
        if key in rendered_updates:
            output_lines.append(f"{key}: {rendered_updates[key]}")
            seen.add(key)
        else:
            output_lines.append(line)

    for key, value in rendered_updates.items():
        if key not in seen:
            output_lines.append(f"{key}: {value}")

    path.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")


def _write_mapping_review_csv(
    summary: OrgUnitMappingSummary,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "district_id",
        "district_name",
        "dhis2_orgunit_id",
        "dhis2_orgunit_name",
        "match_type",
        "parent_name",
        "child_sample",
        "existing_dhis2_orgunit_id",
        "existing_dhis2_orgunit_name",
        "mapping_change",
        "alias_reason",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for match in sorted(summary.proposed_updates, key=lambda item: item.district_name):
            mapping_change = (
                bool(match.existing_dhis2_orgunit_id)
                and match.existing_dhis2_orgunit_id != match.dhis2_orgunit_id
            )
            writer.writerow(
                {
                    "district_id": match.district_id,
                    "district_name": match.district_name,
                    "dhis2_orgunit_id": match.dhis2_orgunit_id,
                    "dhis2_orgunit_name": match.dhis2_orgunit_name,
                    "match_type": match.match_type,
                    "parent_name": match.parent_name or "",
                    "child_sample": match.child_sample or "",
                    "existing_dhis2_orgunit_id": match.existing_dhis2_orgunit_id or "",
                    "existing_dhis2_orgunit_name": match.existing_dhis2_orgunit_name or "",
                    "mapping_change": "yes" if mapping_change else "no",
                    "alias_reason": match.alias_reason or "",
                }
            )


def _org_unit_name(org_unit: dict[str, Any]) -> str:
    return str(org_unit.get("displayName") or org_unit.get("name") or "").strip()


def _parent_name(org_unit: dict[str, Any]) -> str | None:
    parent = org_unit.get("parent")
    if not isinstance(parent, dict):
        return None
    return blank_to_none(parent.get("displayName") or parent.get("name"))


def _child_sample(org_unit: dict[str, Any]) -> str | None:
    children = org_unit.get("children")
    if not isinstance(children, list) or not children:
        return None
    names = [
        str(child.get("displayName") or child.get("name") or "").strip()
        for child in children[:3]
        if isinstance(child, dict)
    ]
    return "; ".join(name for name in names if name) or None


def _unique_reference_key_map(
    reference_rows: list[DistrictReference],
) -> dict[str, DistrictReference]:
    rows_by_key: dict[str, list[DistrictReference]] = {}
    for row in reference_rows:
        rows_by_key.setdefault(_match_key(row.district_name), []).append(row)
    return {
        key: rows[0]
        for key, rows in rows_by_key.items()
        if key and len(rows) == 1
    }


def _strip_dhis2_district_suffix(value: str) -> str:
    cleaned = _exact_key(value)
    return re.sub(r"\s+DISTRICT$", "", cleaned).strip()


def _exact_key(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().upper())


def _match_key(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]+", "", _exact_key(value))


def _weekly_period_type(period_types: list[dict[str, Any]]) -> dict[str, Any] | None:
    for period_type in period_types:
        name = str(period_type.get("name") or period_type.get("id") or "")
        if name.lower() == "weekly":
            return period_type
    return None


def _try_fetch_period_dates(
    client: DHIS2Client,
    period: str,
) -> tuple[str | None, str | None, str | None]:
    candidate_paths = (
        f"/api/periods/{period}.json",
        f"/api/periods/{period}",
    )
    for path in candidate_paths:
        try:
            payload = client.get(path, params={"fields": "id,name,startDate,endDate"})
        except DHIS2HTTPError as exc:
            if exc.status_code == 404:
                continue
            return None, None, f"Could not fetch DHIS2 period dates from {path}: {exc}"
        except DHIS2ClientError as exc:
            return None, None, f"Could not fetch DHIS2 period dates from {path}: {exc}"
        start = blank_to_none(payload.get("startDate"))
        end = blank_to_none(payload.get("endDate"))
        if start and end:
            return start, end, None
    return None, None, (
        "This DHIS2 instance did not return explicit startDate/endDate for "
        f"{period} via the tested period endpoints."
    )


def _parse_week_period(period: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{4})W(\d{1,2})", period.strip())
    if not match:
        raise DHIS2MappingError("Period must use DHIS2 weekly format like 2025W20.")
    year = int(match.group(1))
    week = int(match.group(2))
    if week < 1 or week > 53:
        raise DHIS2MappingError("Weekly period week number must be between 1 and 53.")
    return year, week


def _metadata_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return ""
    text_value = str(value)
    if any(char in text_value for char in [":", "#"]):
        escaped = text_value.replace('"', "'")
        return f'"{escaped}"'
    return text_value
