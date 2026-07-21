"""DHIS2 facility reporting completeness discovery and extraction.

This layer is intentionally conservative. It can search and verify DHIS2
metadata that may represent facility reporting completeness, then update the
existing district-week analytical panel only after a specific metric UID is
configured. It never fabricates completeness values and never creates
district-week feature rows.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from statistics import median
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db
from app.models import DistrictReference, DistrictWeekFeature, PipelineRun
from app.services.dhis2_client import (
    DHIS2Client,
    DHIS2ClientError,
    DHIS2HTTPError,
    DHIS2ResponseFormatError,
    DHIS2TimeoutError,
    TRANSIENT_STATUS_CODES,
    read_district_level_from_env,
    read_required_env,
)
from app.services.dhis2_extraction import (
    PeriodChunk,
    parse_analytics_rows,
)


REPORTING_COMPLETENESS_ENV = "DHIS2_REPORTING_COMPLETENESS_DX"
FALLBACK_STATEMENT = (
    "Facility reporting completeness could not be extracted at district-week "
    "grain from available DHIS2 metadata. Until a suitable reporting-rate "
    "metric is confirmed, the dashboard should either leave the Reporting "
    "Completeness KPI unavailable or explicitly use time_series_completeness "
    "as an analytical data-availability proxy, clearly labelled as such."
)


class ReportingCompletenessError(RuntimeError):
    """Raised when reporting completeness cannot safely be extracted."""


@dataclass(frozen=True)
class ReportingMetricCandidate:
    uid: str
    resource_type: str
    name: str | None = None
    display_name: str | None = None
    code: str | None = None
    period_type: str | None = None
    value_type: str | None = None
    aggregation_type: str | None = None
    formula: str | None = None
    suitability: str = "requires_human_review"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "resource_type": self.resource_type,
            "name": self.name,
            "display_name": self.display_name,
            "code": self.code,
            "period_type": self.period_type,
            "value_type": self.value_type,
            "aggregation_type": self.aggregation_type,
            "formula": self.formula,
            "suitability": self.suitability,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ReportingMetricVerification:
    candidate: ReportingMetricCandidate
    extractable_dx: bool
    metadata_plausible: bool
    warnings: tuple[str, ...] = ()

    @property
    def usable_for_dry_run(self) -> bool:
        return self.extractable_dx and self.metadata_plausible

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "extractable_dx": self.extractable_dx,
            "metadata_plausible": self.metadata_plausible,
            "usable_for_dry_run": self.usable_for_dry_run,
            "warnings": list(self.warnings),
            "fallback_statement": FALLBACK_STATEMENT
            if not self.usable_for_dry_run
            else None,
        }


@dataclass
class ReportingSearchSummary:
    query: str
    candidates: list[ReportingMetricCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warnings": self.warnings,
            "fallback_statement": FALLBACK_STATEMENT,
        }


@dataclass
class ReportingCompletenessSummary:
    dry_run: bool
    pipeline_run_id: int | None = None
    dx_uid: str | None = None
    dx_object_type: str | None = None
    dx_name: str | None = None
    district_level: int | None = None
    requested_periods: list[str] = field(default_factory=list)
    row_count_returned: int = 0
    rows_normalized: int = 0
    rows_updated: int = 0
    rows_without_feature: int = 0
    duplicate_rows_skipped: int = 0
    mapped_org_units: set[str] = field(default_factory=set)
    unmapped_org_units: set[str] = field(default_factory=set)
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)
    normalization_rules: Counter[str] = field(default_factory=Counter)
    normalized_values: list[Decimal] = field(default_factory=list)

    @property
    def min_value(self) -> Decimal | None:
        return min(self.normalized_values) if self.normalized_values else None

    @property
    def median_value(self) -> Decimal | None:
        if not self.normalized_values:
            return None
        return Decimal(str(median(self.normalized_values))).quantize(Decimal("0.01"))

    @property
    def max_value(self) -> Decimal | None:
        return max(self.normalized_values) if self.normalized_values else None


def search_reporting_metrics(
    client: DHIS2Client,
    query: str,
    *,
    max_results: int = 25,
) -> ReportingSearchSummary:
    """Search DHIS2 metadata for possible reporting-completeness metrics."""
    query = query.strip()
    if not query:
        raise ReportingCompletenessError("Search query must not be empty.")

    summary = ReportingSearchSummary(query=query)
    seen: set[tuple[str, str]] = set()
    search_specs = (
        (
            "indicator",
            "/api/indicators.json",
            "indicators",
            "id,code,name,displayName,numerator,denominator,"
            "indicatorType[id,name,displayName,factor,number],annualized",
        ),
        (
            "dataElement",
            "/api/dataElements.json",
            "dataElements",
            "id,code,name,displayName,valueType,aggregationType,domainType,periodType",
        ),
        (
            "programIndicator",
            "/api/programIndicators.json",
            "programIndicators",
            "id,code,name,displayName,expression,filter,analyticsType,aggregationType",
        ),
        (
            "dataSet",
            "/api/dataSets.json",
            "dataSets",
            "id,code,name,displayName,periodType,"
            "indicators[id,name,displayName],"
            "dataSetElements[dataElement[id,name,displayName]]",
        ),
    )

    for resource_type, path, payload_key, fields in search_specs:
        for filter_field in ("name", "displayName", "code"):
            try:
                payload = client.get(
                    path,
                    params={
                        "filter": f"{filter_field}:ilike:{query}",
                        "fields": fields,
                        "pageSize": max_results,
                    },
                )
            except DHIS2HTTPError as exc:
                if exc.status_code == 404:
                    continue
                summary.warnings.append(
                    f"{resource_type} search by {filter_field} failed with HTTP {exc.status_code}."
                )
                continue

            items = payload.get(payload_key)
            if not isinstance(items, list):
                summary.warnings.append(
                    f"{resource_type} search response did not contain {payload_key} list."
                )
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                uid = str(item.get("id") or "").strip()
                if not uid:
                    continue
                key = (resource_type, uid)
                if key in seen:
                    continue
                seen.add(key)
                summary.candidates.append(_candidate_from_metadata(resource_type, item))

    summary.candidates.sort(key=lambda candidate: _candidate_sort_key(candidate))
    return summary


def verify_reporting_metric(
    client: DHIS2Client, uid: str
) -> ReportingMetricVerification:
    """Fetch and conservatively verify one reporting-completeness metric UID."""
    uid = uid.strip()
    if not uid:
        raise ReportingCompletenessError("Reporting completeness UID must not be empty.")

    resources = (
        (
            "indicator",
            f"/api/indicators/{uid}.json",
            "id,code,name,displayName,numerator,denominator,"
            "indicatorType[id,name,displayName,factor,number],annualized",
        ),
        (
            "dataElement",
            f"/api/dataElements/{uid}.json",
            "id,code,name,displayName,valueType,aggregationType,domainType,periodType,"
            "dataSetElements[dataSet[id,name,displayName,periodType]]",
        ),
        (
            "programIndicator",
            f"/api/programIndicators/{uid}.json",
            "id,code,name,displayName,expression,filter,analyticsType,aggregationType",
        ),
        (
            "dataSet",
            f"/api/dataSets/{uid}.json",
            "id,code,name,displayName,periodType,"
            "indicators[id,name,displayName],"
            "dataSetElements[dataElement[id,name,displayName]]",
        ),
    )

    last_not_found: DHIS2HTTPError | None = None
    for resource_type, path, fields in resources:
        try:
            metadata = client.get(path, params={"fields": fields})
        except DHIS2HTTPError as exc:
            if exc.status_code == 404:
                last_not_found = exc
                continue
            raise
        candidate = _candidate_from_metadata(resource_type, metadata)
        extractable_dx = resource_type in {"indicator", "dataElement", "programIndicator"}
        metadata_plausible = _appears_reporting_completeness(candidate)
        warnings = list(candidate.warnings)
        if not extractable_dx:
            warnings.append(
                "Data set metadata is useful for review, but this command only extracts "
                "analytics dx values from indicators, data elements or program indicators."
            )
        if not metadata_plausible:
            warnings.append(
                "Name, code or formula does not clearly describe facility reporting "
                "rate/completeness. Do not use this UID without human metadata confirmation."
            )
        warnings.append(
            "Before any real extraction, run a dry-run analytics query and confirm it "
            "returns mapped district rows for weekly periods."
        )
        return ReportingMetricVerification(
            candidate=candidate,
            extractable_dx=extractable_dx,
            metadata_plausible=metadata_plausible,
            warnings=tuple(warnings),
        )

    raise ReportingCompletenessError(
        f"DHIS2 UID {uid!r} was not found as an indicator, data element, "
        "program indicator or data set."
    ) from last_not_found


def normalize_completeness_value(value: Any) -> tuple[Decimal, str]:
    """Normalize DHIS2 reporting completeness values to a 0-100 percent scale."""
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ReportingCompletenessError(
            f"Invalid reporting completeness numeric value: {value!r}."
        ) from exc

    if Decimal("0") <= decimal_value <= Decimal("1"):
        return (decimal_value * Decimal("100")).quantize(Decimal("0.01")), "proportion_0_1_to_percent"
    if Decimal("0") <= decimal_value <= Decimal("100"):
        return decimal_value.quantize(Decimal("0.01")), "percent_0_100"
    raise ReportingCompletenessError(
        "Reporting completeness values must be between 0 and 1 or between 0 and 100."
    )


def extract_reporting_completeness(
    client: DHIS2Client,
    chunks: list[PeriodChunk],
    *,
    dry_run: bool = False,
) -> ReportingCompletenessSummary:
    """Extract a verified DHIS2 metric into district_week_features."""
    dx = read_required_env(REPORTING_COMPLETENESS_ENV)
    level = read_district_level_from_env()
    summary = ReportingCompletenessSummary(
        dry_run=dry_run,
        requested_periods=[period for chunk in chunks for period in chunk.periods],
    )

    client.verify_login()
    verification = verify_reporting_metric(client, dx)
    candidate = verification.candidate
    summary.dx_uid = dx
    summary.dx_object_type = candidate.resource_type
    summary.dx_name = candidate.display_name or candidate.name
    summary.district_level = level
    summary.warnings.extend(verification.warnings)

    if not verification.usable_for_dry_run:
        summary.warnings.append(FALLBACK_STATEMENT)
        raise ReportingCompletenessError(FALLBACK_STATEMENT)

    district_by_orgunit = _district_by_orgunit()
    if not district_by_orgunit:
        raise ReportingCompletenessError(
            "district_reference has no DHIS2 org-unit mappings; reporting completeness "
            "cannot be keyed by canonical district_id."
        )

    if dry_run:
        return _run_chunks(
            client=client,
            dx=dx,
            dx_name=summary.dx_name,
            level=level,
            chunks=chunks,
            district_by_orgunit=district_by_orgunit,
            run=None,
            summary=summary,
        )

    run = _create_pipeline_run(
        client=client,
        dx=dx,
        candidate=candidate,
        level=level,
        chunks=chunks,
        warnings=summary.warnings,
    )
    summary.pipeline_run_id = run.id

    try:
        summary = _run_chunks(
            client=client,
            dx=dx,
            dx_name=summary.dx_name,
            level=level,
            chunks=chunks,
            district_by_orgunit=district_by_orgunit,
            run=run,
            summary=summary,
        )
        run.status = "success"
        run.completed_at = _utcnow()
        run.row_count_raw = summary.row_count_returned
        run.row_count_processed = summary.rows_updated
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
            run.row_count_processed = summary.rows_updated
            run.warning_count = len(summary.warnings)
            run.error_message = _safe_error_message(exc)
            run.manifest = _summary_manifest(summary)
            db.session.commit()
        if isinstance(exc, ReportingCompletenessError):
            raise
        if isinstance(exc, DHIS2ClientError):
            raise ReportingCompletenessError(str(exc)) from exc
        if isinstance(exc, SQLAlchemyError):
            raise ReportingCompletenessError(
                "Database update failed during reporting completeness extraction."
            ) from exc
        raise


def _run_chunks(
    *,
    client: DHIS2Client,
    dx: str,
    dx_name: str | None,
    level: int,
    chunks: list[PeriodChunk],
    district_by_orgunit: dict[str, str],
    run: PipelineRun | None,
    summary: ReportingCompletenessSummary,
) -> ReportingCompletenessSummary:
    seen_keys: set[tuple[str, str, str]] = set()
    for chunk in chunks:
        summary = _run_chunk_with_fallback(
            client=client,
            dx=dx,
            dx_name=dx_name,
            level=level,
            chunk=chunk,
            district_by_orgunit=district_by_orgunit,
            run=run,
            seen_keys=seen_keys,
            summary=summary,
        )
    return summary


def _run_chunk_with_fallback(
    *,
    client: DHIS2Client,
    dx: str,
    dx_name: str | None,
    level: int,
    chunk: PeriodChunk,
    district_by_orgunit: dict[str, str],
    run: PipelineRun | None,
    seen_keys: set[tuple[str, str, str]],
    summary: ReportingCompletenessSummary,
) -> ReportingCompletenessSummary:
    try:
        payload = client.analytics_query(dx=dx, periods=chunk.periods, level=level)
        return _process_payload(
            payload=payload,
            dx=dx,
            dx_name=dx_name,
            district_by_orgunit=district_by_orgunit,
            seen_keys=seen_keys,
            run=run,
            summary=summary,
        )
    except (DHIS2TimeoutError, DHIS2HTTPError) as exc:
        if isinstance(exc, DHIS2HTTPError) and exc.status_code not in TRANSIENT_STATUS_CODES:
            raise
        if len(chunk.periods) == 1:
            raise
        summary.fallback_used = True
        summary.warnings.append(
            f"Reporting completeness chunk {chunk.label} failed with {type(exc).__name__}; "
            "falling back to smaller chunks."
        )

    for fallback_chunk in _quarter_chunks(chunk):
        try:
            payload = client.analytics_query(dx=dx, periods=fallback_chunk.periods, level=level)
            summary = _process_payload(
                payload=payload,
                dx=dx,
                dx_name=dx_name,
                district_by_orgunit=district_by_orgunit,
                seen_keys=seen_keys,
                run=run,
                summary=summary,
            )
        except (DHIS2TimeoutError, DHIS2HTTPError) as exc:
            if isinstance(exc, DHIS2HTTPError) and exc.status_code not in TRANSIENT_STATUS_CODES:
                raise
            if len(fallback_chunk.periods) == 1:
                raise
            summary.warnings.append(
                f"Reporting completeness fallback chunk {fallback_chunk.label} failed; "
                "falling back to individual weeks."
            )
            for period in fallback_chunk.periods:
                payload = client.analytics_query(dx=dx, periods=[period], level=level)
                summary = _process_payload(
                    payload=payload,
                    dx=dx,
                    dx_name=dx_name,
                    district_by_orgunit=district_by_orgunit,
                    seen_keys=seen_keys,
                    run=run,
                    summary=summary,
                )
    return summary


def _process_payload(
    *,
    payload: dict[str, Any],
    dx: str,
    dx_name: str | None,
    district_by_orgunit: dict[str, str],
    seen_keys: set[tuple[str, str, str]],
    run: PipelineRun | None,
    summary: ReportingCompletenessSummary,
) -> ReportingCompletenessSummary:
    parsed_rows = parse_analytics_rows(payload, fallback_dx=dx, fallback_dx_name=dx_name)
    summary.row_count_returned += len(parsed_rows)

    for row in parsed_rows:
        key = (row.dhis2_orgunit_id, row.data_element_id, row.period)
        if key in seen_keys:
            summary.duplicate_rows_skipped += 1
            summary.warnings.append(
                f"Duplicate reporting completeness row skipped for orgUnit="
                f"{row.dhis2_orgunit_id}, dx={row.data_element_id}, period={row.period}."
            )
            continue
        seen_keys.add(key)

        district_id = district_by_orgunit.get(row.dhis2_orgunit_id)
        if not district_id:
            summary.unmapped_org_units.add(row.dhis2_orgunit_id)
            continue
        summary.mapped_org_units.add(row.dhis2_orgunit_id)

        normalized, rule = normalize_completeness_value(row.value)
        summary.rows_normalized += 1
        summary.normalization_rules[rule] += 1
        summary.normalized_values.append(normalized)

        feature = DistrictWeekFeature.query.filter_by(
            district_id=district_id,
            epi_year=row.epi_year,
            epi_week=row.epi_week,
        ).first()
        if feature is None:
            summary.rows_without_feature += 1
            continue
        if run is not None:
            feature.facility_reporting_completeness = normalized
            summary.rows_updated += 1

    if run is not None:
        db.session.commit()
    return summary


def _candidate_from_metadata(
    resource_type: str, metadata: dict[str, Any]
) -> ReportingMetricCandidate:
    formula_parts: list[str] = []
    for key in ("numerator", "denominator", "expression", "filter"):
        value = metadata.get(key)
        if value:
            formula_parts.append(f"{key}: {value}")
    formula = " | ".join(formula_parts) or None
    period_type = _period_type_name(metadata.get("periodType"))
    warnings: list[str] = []
    if resource_type == "dataSet":
        warnings.append(
            "Data set search result is metadata only; a DHIS2 analytics reporting-rate "
            "UID must still be confirmed before extraction."
        )
    if not period_type and resource_type in {"dataElement", "dataSet"}:
        warnings.append("Metadata did not expose a period type.")

    candidate = ReportingMetricCandidate(
        uid=str(metadata.get("id") or ""),
        resource_type=resource_type,
        name=metadata.get("name"),
        display_name=metadata.get("displayName"),
        code=metadata.get("code"),
        period_type=period_type,
        value_type=metadata.get("valueType"),
        aggregation_type=metadata.get("aggregationType"),
        formula=formula,
        warnings=tuple(warnings),
    )
    suitability = (
        "plausible_reporting_completeness_metric"
        if _appears_reporting_completeness(candidate)
        else "requires_human_review"
    )
    return ReportingMetricCandidate(
        uid=candidate.uid,
        resource_type=candidate.resource_type,
        name=candidate.name,
        display_name=candidate.display_name,
        code=candidate.code,
        period_type=candidate.period_type,
        value_type=candidate.value_type,
        aggregation_type=candidate.aggregation_type,
        formula=candidate.formula,
        suitability=suitability,
        warnings=candidate.warnings,
    )


def _appears_reporting_completeness(candidate: ReportingMetricCandidate) -> bool:
    text = " ".join(
        str(part or "")
        for part in (
            candidate.name,
            candidate.display_name,
            candidate.code,
            candidate.formula,
        )
    ).lower()
    if "reporting rate" in text or "reporting completeness" in text:
        return True
    if "completeness" in text and "report" in text:
        return True
    if "actual reports" in text and "expected reports" in text:
        return True
    if "actual" in text and "expected" in text and "report" in text:
        return True
    return False


def _candidate_sort_key(candidate: ReportingMetricCandidate) -> tuple[int, str, str]:
    plausible_rank = 0 if candidate.suitability.startswith("plausible") else 1
    return (
        plausible_rank,
        candidate.resource_type,
        (candidate.display_name or candidate.name or candidate.uid).lower(),
    )


def _period_type_name(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("displayName") or value.get("name") or value.get("id")
    if value:
        return str(value)
    return None


def _district_by_orgunit() -> dict[str, str]:
    rows = DistrictReference.query.filter(
        DistrictReference.dhis2_orgunit_id.isnot(None),
        DistrictReference.dhis2_orgunit_id != "",
        DistrictReference.active.is_(True),
    ).all()
    return {row.dhis2_orgunit_id: row.district_id for row in rows}


def _create_pipeline_run(
    *,
    client: DHIS2Client,
    dx: str,
    candidate: ReportingMetricCandidate,
    level: int,
    chunks: list[PeriodChunk],
    warnings: list[str],
) -> PipelineRun:
    run = PipelineRun(
        run_type="dhis2_reporting_completeness",
        status="running",
        started_at=_utcnow(),
        source_name="DHIS2 reporting completeness",
        source_version_or_url=client.config.base_url,
        parameters={
            "dx": dx,
            "dx_object_type": candidate.resource_type,
            "dx_name": candidate.display_name or candidate.name,
            "district_level": level,
            "periods": [period for chunk in chunks for period in chunk.periods],
            "chunks": [{"label": chunk.label, "periods": chunk.periods} for chunk in chunks],
            "storage": "district_week_features.facility_reporting_completeness",
        },
        manifest={"warnings": warnings},
    )
    db.session.add(run)
    db.session.commit()
    return run


def _summary_manifest(summary: ReportingCompletenessSummary) -> dict[str, Any]:
    return {
        "dx_uid": summary.dx_uid,
        "dx_object_type": summary.dx_object_type,
        "dx_name": summary.dx_name,
        "district_level": summary.district_level,
        "requested_period_count": len(summary.requested_periods),
        "mapped_org_unit_count": len(summary.mapped_org_units),
        "unmapped_org_unit_count": len(summary.unmapped_org_units),
        "unmapped_org_units": sorted(summary.unmapped_org_units),
        "rows_normalized": summary.rows_normalized,
        "rows_without_feature": summary.rows_without_feature,
        "duplicate_rows_skipped": summary.duplicate_rows_skipped,
        "normalization_rules": dict(summary.normalization_rules),
        "min_value": _decimal_text(summary.min_value),
        "median_value": _decimal_text(summary.median_value),
        "max_value": _decimal_text(summary.max_value),
        "fallback_used": summary.fallback_used,
        "warnings": summary.warnings,
    }


def _quarter_chunks(chunk: PeriodChunk) -> list[PeriodChunk]:
    return [
        PeriodChunk(label=f"{chunk.label}-chunk-{index // 13 + 1}", periods=chunk.periods[index : index + 13])
        for index in range(0, len(chunk.periods), 13)
    ]


def _decimal_text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None


def _safe_error_message(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:1000]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
