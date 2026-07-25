"""DHIS2 raw confirmed malaria extraction service.

This layer stores raw DHIS2 analytics rows only. It does not preprocess,
impute, aggregate missing rows, create districts or write district-week
features.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db
from app.models import DistrictReference, PipelineRun, RawDHIS2Malaria
from app.services.dhis2_client import (
    DHIS2Client,
    DHIS2ClientError,
    DHIS2HTTPError,
    DHIS2ResponseFormatError,
    DHIS2TimeoutError,
    TRANSIENT_STATUS_CODES,
    format_week_period,
    read_district_level_from_env,
    read_required_env,
)


class DHIS2ExtractionError(RuntimeError):
    """Raised when raw DHIS2 extraction cannot safely continue."""


@dataclass(frozen=True)
class PeriodChunk:
    label: str
    periods: list[str]


@dataclass(frozen=True)
class ParsedAnalyticsRow:
    dhis2_orgunit_id: str
    dhis2_orgunit_name: str | None
    data_element_id: str
    data_element_name: str | None
    period: str
    epi_year: int
    epi_week: int
    value: Decimal
    reporting_completeness: Decimal | None
    raw_payload: dict[str, Any]


@dataclass
class ExtractionSummary:
    dry_run: bool
    pipeline_run_id: int | None = None
    dx_uid: str | None = None
    dx_object_type: str | None = None
    dx_name: str | None = None
    aggregation_type: str | None = None
    district_level: int | None = None
    requested_periods: list[str] = field(default_factory=list)
    row_count_returned: int = 0
    rows_inserted: int = 0
    duplicate_rows_skipped: int = 0
    explicit_zero_values: int = 0
    mapped_org_units: set[str] = field(default_factory=set)
    unmapped_org_units: set[str] = field(default_factory=set)
    empty_periods: set[str] = field(default_factory=set)
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)


def generate_week_periods(
    year: int,
    *,
    start_week: int | None = None,
    end_week: int | None = None,
    week: int | None = None,
    include_week_53: bool = False,
) -> list[str]:
    """Generate valid DHIS2 weekly periods using the shared ISO week convention."""
    weeks_in_year = _iso_weeks_in_year(year)
    if week is not None:
        _validate_week(week, max_week=weeks_in_year)
        return [format_week_period(year, week)]

    start = start_week or 1
    end = end_week or weeks_in_year
    _validate_week(start, max_week=weeks_in_year)
    _validate_week(end, max_week=weeks_in_year)
    if start > end:
        raise DHIS2ExtractionError("start_week must be less than or equal to end_week.")
    if include_week_53 and weeks_in_year == 53 and end < 53 and end_week is None:
        end = 53
    return [format_week_period(year, week_number) for week_number in range(start, end + 1)]


def build_extraction_plan(
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    year: int | None = None,
    start_week: int | None = None,
    end_week: int | None = None,
    week: int | None = None,
    include_week_53: bool = False,
) -> list[PeriodChunk]:
    """Build year-level extraction chunks."""
    if year is not None and (start_year is not None or end_year is not None):
        raise DHIS2ExtractionError("Use either --year or --start-year/--end-year, not both.")

    if year is not None:
        years = [year]
    else:
        start = start_year or 2020
        end = end_year or start
        if start > end:
            raise DHIS2ExtractionError("start_year must be less than or equal to end_year.")
        years = list(range(start, end + 1))

    chunks: list[PeriodChunk] = []
    for selected_year in years:
        periods = generate_week_periods(
            selected_year,
            start_week=start_week,
            end_week=end_week,
            week=week,
            include_week_53=include_week_53,
        )
        chunks.append(PeriodChunk(label=str(selected_year), periods=periods))
    return chunks


def validate_extractable_metadata(resource_type: str, metadata: dict[str, Any]) -> list[str]:
    """Validate metadata before extraction and return non-fatal warnings."""
    if resource_type not in {"dataElement", "indicator", "programIndicator"}:
        raise DHIS2ExtractionError(
            f"Unsupported DHIS2 dx object type {resource_type!r}; extraction stopped."
        )

    warnings: list[str] = []
    category_combo = metadata.get("categoryCombo") or {}
    combo_name = (
        str(category_combo.get("displayName") or category_combo.get("name") or "")
        .strip()
        .lower()
    )
    category_option_combos = metadata.get("categoryOptionCombos") or []
    if resource_type == "dataElement" and combo_name and combo_name != "default":
        raise DHIS2ExtractionError(
            "Configured DHIS2 data element appears to use a non-default "
            f"category combo ({combo_name!r}). Extraction stopped because a "
            "category option combo/disaggregation decision is required."
        )
    if (
        resource_type == "dataElement"
        and isinstance(category_option_combos, list)
        and len(category_option_combos) > 1
        and combo_name != "default"
    ):
        raise DHIS2ExtractionError(
            "Configured DHIS2 data element has multiple category option combos. "
            "Extraction stopped until disaggregation is verified."
        )

    if resource_type == "programIndicator":
        warnings.append(
            "Configured dx is a programIndicator. Confirm it is valid for the "
            "aggregate confirmed malaria case definition before extraction."
        )
    if not metadata.get("aggregationType"):
        warnings.append("DHIS2 metadata did not report aggregationType.")
    warnings.append(
        "Facility reporting completeness was not extracted from the case analytics "
        "response; reporting_completeness will remain null unless DHIS2 returns a "
        "dedicated completeness column."
    )
    return warnings


def parse_analytics_rows(
    payload: dict[str, Any],
    *,
    fallback_dx: str,
    fallback_dx_name: str | None,
) -> list[ParsedAnalyticsRow]:
    """Parse standard DHIS2 analytics response rows by headers when present."""
    rows = payload.get("rows")
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise DHIS2ResponseFormatError("Analytics response 'rows' must be a list.")

    headers = payload.get("headers") or []
    indexes = _analytics_header_indexes(headers)
    metadata_items = _metadata_items(payload)
    parsed: list[ParsedAnalyticsRow] = []

    for raw_row in rows:
        if not isinstance(raw_row, list):
            raise DHIS2ResponseFormatError("Each analytics row must be a list.")
        dx_value = _row_value(raw_row, indexes, "dx") or fallback_dx
        period = _row_value(raw_row, indexes, "pe")
        orgunit_id = _row_value(raw_row, indexes, "ou")
        value_raw = _row_value(raw_row, indexes, "value")
        if not period or not orgunit_id or value_raw is None:
            raise DHIS2ResponseFormatError(
                "Analytics row missing one of required dx/pe/ou/value fields."
            )
        epi_year, epi_week = parse_period(period)
        normalized_period = format_week_period(epi_year, epi_week)
        value = _decimal_value(value_raw, field_name="value")
        reporting_completeness = None
        if "reporting_completeness" in indexes:
            completeness_raw = _row_value(raw_row, indexes, "reporting_completeness")
            reporting_completeness = (
                _decimal_value(completeness_raw, field_name="reporting_completeness")
                if completeness_raw not in {None, ""}
                else None
            )

        parsed.append(
            ParsedAnalyticsRow(
                dhis2_orgunit_id=orgunit_id,
                dhis2_orgunit_name=_metadata_name(metadata_items, orgunit_id),
                data_element_id=dx_value,
                data_element_name=_metadata_name(metadata_items, dx_value) or fallback_dx_name,
                period=normalized_period,
                epi_year=epi_year,
                epi_week=epi_week,
                value=value,
                reporting_completeness=reporting_completeness,
                raw_payload={"row": raw_row, "headers": headers},
            )
        )

    return parsed


def extract_malaria_raw(
    client: DHIS2Client,
    chunks: list[PeriodChunk],
    *,
    dry_run: bool = False,
) -> ExtractionSummary:
    """Run a raw DHIS2 malaria extraction."""
    dx = read_required_env("DHIS2_CONFIRMED_MALARIA_DX")
    level = read_district_level_from_env()
    summary = ExtractionSummary(
        dry_run=dry_run,
        requested_periods=[period for chunk in chunks for period in chunk.periods],
    )

    # Preconditions: credential, metadata and org-unit checks before extraction.
    client.verify_login()
    resource_type, metadata = client.fetch_metadata_object(dx)
    metadata_warnings = validate_extractable_metadata(resource_type, metadata)
    summary.warnings.extend(metadata_warnings)
    summary.dx_uid = dx
    summary.dx_object_type = resource_type
    summary.dx_name = metadata.get("displayName") or metadata.get("name")
    summary.aggregation_type = metadata.get("aggregationType")
    summary.district_level = level
    org_units = client.fetch_org_units_at_level(level)
    if not org_units:
        raise DHIS2ExtractionError(
            f"DHIS2 returned no organisation units at LEVEL-{level}; extraction stopped."
        )

    dx_name = metadata.get("displayName") or metadata.get("name")
    district_by_orgunit = {
        district.dhis2_orgunit_id: district.district_id
        for district in DistrictReference.query.filter(
            DistrictReference.dhis2_orgunit_id.isnot(None),
            DistrictReference.dhis2_orgunit_id != "",
        ).all()
    }
    if not district_by_orgunit:
        summary.warnings.append(
            "district_reference has no DHIS2 org-unit mappings; district_id will be null."
        )

    if dry_run:
        return _dry_run_chunks(
            client=client,
            dx=dx,
            dx_name=dx_name,
            level=level,
            chunks=chunks,
            district_by_orgunit=district_by_orgunit,
            summary=summary,
        )

    run = _create_pipeline_run(
        client=client,
        dx=dx,
        dx_object_type=resource_type,
        metadata=metadata,
        level=level,
        chunks=chunks,
        warnings=summary.warnings,
    )
    summary.pipeline_run_id = run.id

    try:
        seen_keys: set[tuple[int, str, str, str]] = set()
        for chunk in chunks:
            summary = _extract_chunk_with_fallback(
                client=client,
                dx=dx,
                dx_name=dx_name,
                level=level,
                chunk=chunk,
                run=run,
                district_by_orgunit=district_by_orgunit,
                seen_keys=seen_keys,
                summary=summary,
            )

        requested_set = set(summary.requested_periods)
        returned_set = {
            period
            for (pipeline_run_id, _orgunit_id, _dx, period) in seen_keys
            if pipeline_run_id == run.id
        }
        summary.empty_periods = requested_set - returned_set
        run.status = "success"
        run.completed_at = _utcnow()
        run.row_count_raw = summary.row_count_returned
        run.row_count_processed = summary.rows_inserted
        run.warning_count = len(summary.warnings)
        run.manifest = _summary_manifest(summary)
        db.session.commit()
        return summary
    except Exception as exc:
        db.session.rollback()
        run = db.session.get(PipelineRun, run.id)
        if run is not None:
            run.status = "failed"
            run.completed_at = _utcnow()
            run.row_count_raw = summary.row_count_returned
            run.row_count_processed = summary.rows_inserted
            run.warning_count = len(summary.warnings)
            run.error_message = _safe_error_message(exc)
            run.manifest = _summary_manifest(summary)
            db.session.commit()
        if isinstance(exc, DHIS2ExtractionError):
            raise
        if isinstance(exc, DHIS2ClientError):
            raise DHIS2ExtractionError(str(exc)) from exc
        if isinstance(exc, SQLAlchemyError):
            raise DHIS2ExtractionError("Database insert failed during extraction.") from exc
        raise


def _dry_run_chunks(
    *,
    client: DHIS2Client,
    dx: str,
    dx_name: str | None,
    level: int,
    chunks: list[PeriodChunk],
    district_by_orgunit: dict[str, str],
    summary: ExtractionSummary,
) -> ExtractionSummary:
    seen_keys: set[tuple[str, str, str]] = set()
    returned_periods: set[str] = set()
    for chunk in chunks:
        summary = _dry_run_chunk_with_fallback(
            client=client,
            dx=dx,
            dx_name=dx_name,
            level=level,
            chunk=chunk,
            district_by_orgunit=district_by_orgunit,
            seen_keys=seen_keys,
            returned_periods=returned_periods,
            summary=summary,
        )
    summary.empty_periods = set(summary.requested_periods) - returned_periods
    return summary


def _dry_run_chunk_with_fallback(
    *,
    client: DHIS2Client,
    dx: str,
    dx_name: str | None,
    level: int,
    chunk: PeriodChunk,
    district_by_orgunit: dict[str, str],
    seen_keys: set[tuple[str, str, str]],
    returned_periods: set[str],
    summary: ExtractionSummary,
) -> ExtractionSummary:
    try:
        payload = client.analytics_query(dx=dx, periods=chunk.periods, level=level)
        return _summarize_dry_run_payload(
            payload=payload,
            dx=dx,
            dx_name=dx_name,
            district_by_orgunit=district_by_orgunit,
            seen_keys=seen_keys,
            returned_periods=returned_periods,
            summary=summary,
        )
    except (DHIS2TimeoutError, DHIS2HTTPError) as exc:
        if isinstance(exc, DHIS2HTTPError) and exc.status_code not in TRANSIENT_STATUS_CODES:
            raise
        if len(chunk.periods) == 1:
            raise
        summary.fallback_used = True
        summary.warnings.append(
            f"Dry-run chunk {chunk.label} failed with {type(exc).__name__}; "
            "falling back to smaller chunks."
        )

    for fallback_chunk in _quarter_chunks(chunk):
        try:
            payload = client.analytics_query(dx=dx, periods=fallback_chunk.periods, level=level)
            summary = _summarize_dry_run_payload(
                payload=payload,
                dx=dx,
                dx_name=dx_name,
                district_by_orgunit=district_by_orgunit,
                seen_keys=seen_keys,
                returned_periods=returned_periods,
                summary=summary,
            )
        except (DHIS2TimeoutError, DHIS2HTTPError) as exc:
            if isinstance(exc, DHIS2HTTPError) and exc.status_code not in TRANSIENT_STATUS_CODES:
                raise
            if len(fallback_chunk.periods) == 1:
                raise
            summary.warnings.append(
                f"Dry-run fallback chunk {fallback_chunk.label} failed; "
                "falling back to individual weeks."
            )
            for period in fallback_chunk.periods:
                payload = client.analytics_query(dx=dx, periods=[period], level=level)
                summary = _summarize_dry_run_payload(
                    payload=payload,
                    dx=dx,
                    dx_name=dx_name,
                    district_by_orgunit=district_by_orgunit,
                    seen_keys=seen_keys,
                    returned_periods=returned_periods,
                    summary=summary,
                )
    return summary


def _summarize_dry_run_payload(
    *,
    payload: dict[str, Any],
    dx: str,
    dx_name: str | None,
    district_by_orgunit: dict[str, str],
    seen_keys: set[tuple[str, str, str]],
    returned_periods: set[str],
    summary: ExtractionSummary,
) -> ExtractionSummary:
    parsed_rows = parse_analytics_rows(
        payload, fallback_dx=dx, fallback_dx_name=dx_name
    )
    summary.row_count_returned += len(parsed_rows)
    for row in parsed_rows:
        key = (row.dhis2_orgunit_id, row.data_element_id, row.period)
        if key in seen_keys:
            summary.duplicate_rows_skipped += 1
            continue
        seen_keys.add(key)
        returned_periods.add(row.period)
        if district_by_orgunit.get(row.dhis2_orgunit_id):
            summary.mapped_org_units.add(row.dhis2_orgunit_id)
        else:
            summary.unmapped_org_units.add(row.dhis2_orgunit_id)
        if row.value == 0:
            summary.explicit_zero_values += 1
    return summary


def _extract_chunk_with_fallback(
    *,
    client: DHIS2Client,
    dx: str,
    dx_name: str | None,
    level: int,
    chunk: PeriodChunk,
    run: PipelineRun,
    district_by_orgunit: dict[str, str],
    seen_keys: set[tuple[int, str, str, str]],
    summary: ExtractionSummary,
) -> ExtractionSummary:
    try:
        payload = client.analytics_query(dx=dx, periods=chunk.periods, level=level)
        return _store_payload_rows(
            payload=payload,
            fallback_dx=dx,
            fallback_dx_name=dx_name,
            run=run,
            district_by_orgunit=district_by_orgunit,
            seen_keys=seen_keys,
            summary=summary,
        )
    except (DHIS2TimeoutError, DHIS2HTTPError) as exc:
        if isinstance(exc, DHIS2HTTPError) and exc.status_code not in TRANSIENT_STATUS_CODES:
            raise
        if len(chunk.periods) == 1:
            raise
        summary.fallback_used = True
        summary.warnings.append(
            f"Chunk {chunk.label} failed with {type(exc).__name__}; falling back to smaller chunks."
        )

    for fallback_chunk in _quarter_chunks(chunk):
        try:
            payload = client.analytics_query(dx=dx, periods=fallback_chunk.periods, level=level)
            summary = _store_payload_rows(
                payload=payload,
                fallback_dx=dx,
                fallback_dx_name=dx_name,
                run=run,
                district_by_orgunit=district_by_orgunit,
                seen_keys=seen_keys,
                summary=summary,
            )
        except (DHIS2TimeoutError, DHIS2HTTPError) as exc:
            if isinstance(exc, DHIS2HTTPError) and exc.status_code not in TRANSIENT_STATUS_CODES:
                raise
            if len(fallback_chunk.periods) == 1:
                raise
            summary.warnings.append(
                f"Fallback chunk {fallback_chunk.label} failed; falling back to individual weeks."
            )
            for period in fallback_chunk.periods:
                payload = client.analytics_query(dx=dx, periods=[period], level=level)
                summary = _store_payload_rows(
                    payload=payload,
                    fallback_dx=dx,
                    fallback_dx_name=dx_name,
                    run=run,
                    district_by_orgunit=district_by_orgunit,
                    seen_keys=seen_keys,
                    summary=summary,
                )
    return summary


def _store_payload_rows(
    *,
    payload: dict[str, Any],
    fallback_dx: str,
    fallback_dx_name: str | None,
    run: PipelineRun,
    district_by_orgunit: dict[str, str],
    seen_keys: set[tuple[int, str, str, str]],
    summary: ExtractionSummary,
) -> ExtractionSummary:
    parsed_rows = parse_analytics_rows(
        payload, fallback_dx=fallback_dx, fallback_dx_name=fallback_dx_name
    )
    summary.row_count_returned += len(parsed_rows)

    for row in parsed_rows:
        key = (run.id, row.dhis2_orgunit_id, row.data_element_id, row.period)
        if key in seen_keys:
            summary.duplicate_rows_skipped += 1
            summary.warnings.append(
                f"Duplicate analytics row skipped for orgUnit={row.dhis2_orgunit_id}, "
                f"dx={row.data_element_id}, period={row.period}."
            )
            continue
        seen_keys.add(key)

        # Presentation evidence: replace the source-specific DHIS2 UID with the approved common district key.
        district_id = district_by_orgunit.get(row.dhis2_orgunit_id)
        if district_id:
            summary.mapped_org_units.add(row.dhis2_orgunit_id)
        else:
            summary.unmapped_org_units.add(row.dhis2_orgunit_id)

        db.session.add(
            RawDHIS2Malaria(
                pipeline_run_id=run.id,
                dhis2_orgunit_id=row.dhis2_orgunit_id,
                dhis2_orgunit_name=row.dhis2_orgunit_name,
                district_id=district_id,
                data_element_id=row.data_element_id,
                data_element_name=row.data_element_name,
                period=row.period,
                epi_year=row.epi_year,
                epi_week=row.epi_week,
                value=row.value,
                reporting_completeness=row.reporting_completeness,
                raw_payload=row.raw_payload,
                extracted_at=_utcnow(),
            )
        )
        summary.rows_inserted += 1
        if row.value == 0:
            summary.explicit_zero_values += 1

    db.session.commit()
    return summary


def _create_pipeline_run(
    *,
    client: DHIS2Client,
    dx: str,
    dx_object_type: str,
    metadata: dict[str, Any],
    level: int,
    chunks: list[PeriodChunk],
    warnings: list[str],
) -> PipelineRun:
    safe_base_url = client.config.base_url
    run = PipelineRun(
        run_type="dhis2_extraction",
        status="running",
        started_at=_utcnow(),
        source_name="DHIS2",
        source_version_or_url=safe_base_url,
        parameters={
            "dx": dx,
            "dx_object_type": dx_object_type,
            "dx_name": metadata.get("displayName") or metadata.get("name"),
            "aggregation_type": metadata.get("aggregationType"),
            "district_level": level,
            "periods": [period for chunk in chunks for period in chunk.periods],
            "chunks": [{"label": chunk.label, "periods": chunk.periods} for chunk in chunks],
        },
        manifest={"warnings": warnings},
    )
    db.session.add(run)
    db.session.commit()
    return run


def _summary_manifest(summary: ExtractionSummary) -> dict[str, Any]:
    return {
        "dx_uid": summary.dx_uid,
        "dx_object_type": summary.dx_object_type,
        "dx_name": summary.dx_name,
        "aggregation_type": summary.aggregation_type,
        "district_level": summary.district_level,
        "requested_period_count": len(summary.requested_periods),
        "empty_periods": sorted(summary.empty_periods),
        "mapped_org_unit_count": len(summary.mapped_org_units),
        "unmapped_org_unit_count": len(summary.unmapped_org_units),
        "unmapped_org_units": sorted(summary.unmapped_org_units),
        "explicit_zero_values": summary.explicit_zero_values,
        "duplicate_rows_skipped": summary.duplicate_rows_skipped,
        "fallback_used": summary.fallback_used,
        "warnings": summary.warnings,
    }


def _analytics_header_indexes(headers: Any) -> dict[str, int]:
    if not isinstance(headers, list) or not headers:
        return {"dx": 0, "pe": 1, "ou": 2, "value": 3}

    indexes: dict[str, int] = {}
    for index, header in enumerate(headers):
        if not isinstance(header, dict):
            continue
        raw_name = str(
            header.get("name")
            or header.get("column")
            or header.get("dimension")
            or header.get("valueType")
            or ""
        ).strip()
        normalized = raw_name.lower()
        if normalized in {"dx", "data", "dataelement", "data_element"}:
            indexes["dx"] = index
        elif normalized in {"pe", "period"}:
            indexes["pe"] = index
        elif normalized in {"ou", "organisationunit", "orgunit", "org_unit"}:
            indexes["ou"] = index
        elif normalized in {"value", "val"}:
            indexes["value"] = index
        elif normalized in {"reporting_completeness", "reporting completeness"}:
            indexes["reporting_completeness"] = index

    missing = {"dx", "pe", "ou", "value"} - set(indexes)
    if missing:
        raise DHIS2ResponseFormatError(
            "Analytics response headers missing required columns: "
            + ", ".join(sorted(missing))
        )
    return indexes


def _row_value(row: list[Any], indexes: dict[str, int], key: str) -> Any:
    index = indexes.get(key)
    if index is None or index >= len(row):
        return None
    return row[index]


def _metadata_items(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metaData") or payload.get("metadata") or {}
    items = metadata.get("items") if isinstance(metadata, dict) else {}
    return items if isinstance(items, dict) else {}


def _metadata_name(items: dict[str, Any], uid: str) -> str | None:
    item = items.get(uid)
    if isinstance(item, dict):
        return item.get("displayName") or item.get("name")
    return None


def parse_period(period: str) -> tuple[int, int]:
    if len(period) < 6 or "W" not in period:
        raise DHIS2ResponseFormatError(f"Invalid DHIS2 weekly period {period!r}.")
    year_text, week_text = period.split("W", 1)
    try:
        year = int(year_text)
        week = int(week_text)
    except ValueError as exc:
        raise DHIS2ResponseFormatError(f"Invalid DHIS2 weekly period {period!r}.") from exc
    _validate_week(week, max_week=53)
    return year, week


def _decimal_value(value: Any, *, field_name: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise DHIS2ResponseFormatError(f"Invalid numeric {field_name}: {value!r}") from exc


def _validate_week(week: int, *, max_week: int = 53) -> None:
    if week < 1 or week > max_week:
        raise DHIS2ExtractionError(f"Week must be between 1 and {max_week}.")


def _iso_weeks_in_year(year: int) -> int:
    return date(year, 12, 28).isocalendar().week


def _quarter_chunks(chunk: PeriodChunk) -> list[PeriodChunk]:
    return [
        PeriodChunk(label=f"{chunk.label}-chunk-{index // 13 + 1}", periods=chunk.periods[index : index + 13])
        for index in range(0, len(chunk.periods), 13)
    ]


def _safe_error_message(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:1000]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
