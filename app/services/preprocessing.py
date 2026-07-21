"""DHIS2 raw-to-district-week preprocessing service.

This service builds the DHIS2-derived portion of district_week_features. It
does not merge climate data, compute climate lags, train models or forecast.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db
from app.models import (
    DistrictReference,
    DistrictWeekFeature,
    PipelineRun,
    RawDHIS2Malaria,
)
from app.services.dhis2_extraction import parse_period
from app.services.reference_data import EXPECTED_ACTIVE_DISTRICTS


class PreprocessingError(RuntimeError):
    """Raised when preprocessing cannot safely continue."""


@dataclass(frozen=True)
class PreprocessingSelection:
    source_run_ids: list[int]
    expected_periods: list[str]
    filter_start_year: int | None = None
    filter_end_year: int | None = None


@dataclass
class PreprocessingSummary:
    dry_run: bool
    preprocessing_run_id: int | None = None
    source_run_ids: list[int] = field(default_factory=list)
    expected_period_count: int = 0
    raw_rows_considered: int = 0
    raw_rows_mapped: int = 0
    raw_rows_unmapped: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    districts_processed: set[str] = field(default_factory=set)
    unmapped_org_units: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PreparedFeature:
    district_id: str
    epi_year: int
    epi_week: int
    confirmed_malaria_cases: int
    facility_reporting_completeness: Decimal | None


def iso_weeks_in_year(year: int) -> int:
    """Return the number of ISO weeks in a year."""
    return date(year, 12, 28).isocalendar().week


def iso_week_dates(year: int, week: int) -> tuple[date, date]:
    """Return Monday-Sunday dates for an ISO week."""
    try:
        start = date.fromisocalendar(year, week, 1)
    except ValueError as exc:
        raise PreprocessingError(f"Invalid ISO epi week {year}W{week:02d}.") from exc
    return start, date.fromisocalendar(year, week, 7)


def iso_periods_for_year(year: int) -> list[str]:
    return [f"{year}W{week:02d}" for week in range(1, iso_weeks_in_year(year) + 1)]


def expected_periods_for_years(start_year: int, end_year: int) -> list[str]:
    if start_year > end_year:
        raise PreprocessingError("start_year must be less than or equal to end_year.")
    periods: list[str] = []
    for year in range(start_year, end_year + 1):
        periods.extend(iso_periods_for_year(year))
    return periods


def season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "DJF"
    if month in {3, 4, 5}:
        return "MAM"
    if month in {6, 7, 8}:
        return "JJA"
    return "SON"


def build_selection(
    *,
    pipeline_run_id: int | None = None,
    latest_successful_dhis2_run: bool = False,
    start_year: int | None = None,
    end_year: int | None = None,
    year: int | None = None,
) -> PreprocessingSelection:
    """Resolve source DHIS2 runs and expected ISO weekly periods."""
    if pipeline_run_id and latest_successful_dhis2_run:
        raise PreprocessingError("Use either --pipeline-run-id or --latest-successful-dhis2-run.")
    if year is not None and (start_year is not None or end_year is not None):
        raise PreprocessingError("Use either --year or --start-year/--end-year, not both.")

    if pipeline_run_id:
        source_run = _get_successful_dhis2_run(pipeline_run_id)
        return PreprocessingSelection(
            source_run_ids=[source_run.id],
            expected_periods=_periods_from_run_or_rows(source_run),
        )

    if latest_successful_dhis2_run:
        source_run = (
            PipelineRun.query.filter_by(run_type="dhis2_extraction", status="success")
            .order_by(PipelineRun.completed_at.desc().nullslast(), PipelineRun.id.desc())
            .first()
        )
        if source_run is None:
            raise PreprocessingError("No successful DHIS2 extraction pipeline run found.")
        return PreprocessingSelection(
            source_run_ids=[source_run.id],
            expected_periods=_periods_from_run_or_rows(source_run),
        )

    if year is not None:
        selected_start = selected_end = year
    else:
        selected_start = start_year or 2020
        selected_end = end_year or selected_start

    source_runs = (
        PipelineRun.query.filter_by(run_type="dhis2_extraction", status="success")
        .join(RawDHIS2Malaria, RawDHIS2Malaria.pipeline_run_id == PipelineRun.id)
        .filter(
            RawDHIS2Malaria.epi_year >= selected_start,
            RawDHIS2Malaria.epi_year <= selected_end,
        )
        .distinct()
        .all()
    )
    if not source_runs:
        raise PreprocessingError(
            "No successful DHIS2 extraction raw rows found for the requested period."
        )
    return PreprocessingSelection(
        source_run_ids=[run.id for run in source_runs],
        expected_periods=expected_periods_for_years(selected_start, selected_end),
        filter_start_year=selected_start,
        filter_end_year=selected_end,
    )


def build_district_week_features(
    selection: PreprocessingSelection,
    *,
    dry_run: bool = False,
    allow_many_to_one: bool = False,
    replace_existing: bool = False,
    require_full_frame: bool = False,
) -> PreprocessingSummary:
    """Build district_week_features from successful raw DHIS2 rows."""
    summary = PreprocessingSummary(
        dry_run=dry_run,
        source_run_ids=selection.source_run_ids,
        expected_period_count=len(selection.expected_periods),
    )
    expected_period_set = set(selection.expected_periods)
    if not expected_period_set:
        raise PreprocessingError("No expected periods resolved for preprocessing.")

    active_district_count = DistrictReference.query.filter_by(active=True).count()
    if active_district_count != EXPECTED_ACTIVE_DISTRICTS:
        message = (
            f"Active district_reference count is {active_district_count}, expected "
            f"{EXPECTED_ACTIVE_DISTRICTS}. Partial-development mode is allowed, "
            "but final modelling should use the full approved frame."
        )
        if require_full_frame:
            raise PreprocessingError(message)
        summary.warnings.append(message)

    reference_rows = DistrictReference.query.filter(
        DistrictReference.dhis2_orgunit_id.isnot(None),
        DistrictReference.dhis2_orgunit_id != "",
    ).all()
    districts_by_orgunit: dict[str, set[str]] = defaultdict(set)
    for district in reference_rows:
        districts_by_orgunit[district.dhis2_orgunit_id].add(district.district_id)
    duplicated_orgunits = {
        orgunit_id: sorted(district_ids)
        for orgunit_id, district_ids in districts_by_orgunit.items()
        if len(district_ids) > 1
    }
    if duplicated_orgunits:
        details = "; ".join(
            f"{orgunit_id}: {', '.join(district_ids)}"
            for orgunit_id, district_ids in sorted(duplicated_orgunits.items())
        )
        raise PreprocessingError(
            "One DHIS2 org unit maps to multiple canonical district_id values in "
            f"district_reference. Fix the curated mapping before preprocessing. Details: {details}"
        )
    district_by_orgunit = {
        orgunit_id: next(iter(district_ids))
        for orgunit_id, district_ids in districts_by_orgunit.items()
    }

    raw_query = RawDHIS2Malaria.query.filter(
        RawDHIS2Malaria.pipeline_run_id.in_(selection.source_run_ids)
    )
    if selection.filter_start_year is not None:
        raw_query = raw_query.filter(
            RawDHIS2Malaria.epi_year >= selection.filter_start_year,
            RawDHIS2Malaria.epi_year <= selection.filter_end_year,
        )
    raw_rows = raw_query.order_by(
        RawDHIS2Malaria.district_id,
        RawDHIS2Malaria.dhis2_orgunit_id,
        RawDHIS2Malaria.epi_year,
        RawDHIS2Malaria.epi_week,
    ).all()
    summary.raw_rows_considered = len(raw_rows)

    grouped: dict[tuple[str, int, int], list[RawDHIS2Malaria]] = defaultdict(list)
    orgunits_by_district: dict[str, set[str]] = defaultdict(set)
    observed_periods_by_district: dict[str, set[str]] = defaultdict(set)

    for raw in raw_rows:
        period = f"{raw.epi_year}W{raw.epi_week:02d}"
        if period not in expected_period_set:
            summary.warnings.append(
                f"Raw row period {period} is outside the expected preprocessing period."
            )
            continue
        district_id = raw.district_id or district_by_orgunit.get(raw.dhis2_orgunit_id)
        if not district_id:
            summary.raw_rows_unmapped += 1
            summary.unmapped_org_units.add(raw.dhis2_orgunit_id)
            continue
        if raw.value is None:
            summary.warnings.append(
                f"Raw row for {district_id} {period} has null value and was skipped; "
                "missing DHIS2 values are not converted to zero."
            )
            continue
        summary.raw_rows_mapped += 1
        grouped[(district_id, raw.epi_year, raw.epi_week)].append(raw)
        orgunits_by_district[district_id].add(raw.dhis2_orgunit_id)
        observed_periods_by_district[district_id].add(period)

    many_to_one = {
        district_id: sorted(orgunits)
        for district_id, orgunits in orgunits_by_district.items()
        if len(orgunits) > 1
    }
    if many_to_one and not allow_many_to_one:
        details = "; ".join(
            f"{district_id}: {', '.join(orgunits)}"
            for district_id, orgunits in sorted(many_to_one.items())
        )
        raise PreprocessingError(
            "Multiple DHIS2 org units map to one canonical district_id. "
            "Review and rerun with --allow-many-to-one only if this is an "
            f"approved historical mapping. Details: {details}"
        )
    if many_to_one:
        summary.warnings.append(
            "Approved many-to-one DHIS2 org-unit mapping used for: "
            + "; ".join(sorted(many_to_one))
        )

    prepared: list[PreparedFeature] = []
    for key, rows in grouped.items():
        district_id, epi_year, epi_week = key
        orgunits = {row.dhis2_orgunit_id for row in rows}
        if len(rows) > 1 and len(orgunits) == 1:
            raise PreprocessingError(
                f"Duplicate raw rows for {district_id} {epi_year}W{epi_week:02d} "
                f"from org unit {next(iter(orgunits))}; refusing to collapse duplicates."
            )
        if len(rows) > 1 and not allow_many_to_one:
            raise PreprocessingError(
                f"Multiple raw rows for {district_id} {epi_year}W{epi_week:02d}; "
                "rerun with --allow-many-to-one only if this mapping is approved."
            )
        confirmed_cases = sum(_integer_cases(row.value) for row in rows)
        completeness_values = {
            row.reporting_completeness for row in rows if row.reporting_completeness is not None
        }
        facility_completeness = (
            next(iter(completeness_values)) if len(completeness_values) == 1 else None
        )
        if len(rows) > 1 and len(completeness_values) > 1:
            summary.warnings.append(
                f"Facility reporting completeness set null for {district_id} "
                f"{epi_year}W{epi_week:02d} because multiple source values exist."
            )
        prepared.append(
            PreparedFeature(
                district_id=district_id,
                epi_year=epi_year,
                epi_week=epi_week,
                confirmed_malaria_cases=confirmed_cases,
                facility_reporting_completeness=facility_completeness,
            )
        )

    if dry_run:
        summary.inserted = len(prepared)
        summary.districts_processed = {feature.district_id for feature in prepared}
        _append_week_mismatch_warnings(summary, observed_periods_by_district, expected_period_set)
        return summary

    run = PipelineRun(
        run_type="preprocessing",
        status="running",
        started_at=_utcnow(),
        source_name="raw_dhis2_malaria",
        parameters={
            "source_run_ids": selection.source_run_ids,
            "expected_periods": selection.expected_periods,
            "calendar": "ISO-8601 Monday-start, provisional pending final DHIS2 calendar confirmation",
            "allow_many_to_one": allow_many_to_one,
            "replace_existing": replace_existing,
        },
    )
    db.session.add(run)
    db.session.commit()
    summary.preprocessing_run_id = run.id

    try:
        completeness_by_district = {
            district_id: _percentage(len(periods), len(expected_period_set))
            for district_id, periods in observed_periods_by_district.items()
        }
        for feature in prepared:
            week_start, week_end = iso_week_dates(feature.epi_year, feature.epi_week)
            existing = DistrictWeekFeature.query.filter_by(
                district_id=feature.district_id,
                epi_year=feature.epi_year,
                epi_week=feature.epi_week,
            ).first()

            values = {
                "confirmed_malaria_cases": feature.confirmed_malaria_cases,
                "facility_reporting_completeness": feature.facility_reporting_completeness,
                "time_series_completeness": completeness_by_district.get(feature.district_id),
                "week_start_date": week_start,
                "week_end_date": week_end,
                "month": week_start.month,
                "quarter": ((week_start.month - 1) // 3) + 1,
                "season": season_for_month(week_start.month),
                "population": None,
                "incidence_per_1000": None,
                "source_pipeline_run_id": run.id,
            }

            if existing is None:
                db.session.add(
                    DistrictWeekFeature(
                        district_id=feature.district_id,
                        epi_year=feature.epi_year,
                        epi_week=feature.epi_week,
                        **values,
                    )
                )
                summary.inserted += 1
            elif (
                existing.source_pipeline_run_id == run.id
                or _existing_is_same_source(existing, selection)
                or replace_existing
            ):
                for field_name, value in values.items():
                    setattr(existing, field_name, value)
                summary.updated += 1
            else:
                summary.skipped += 1
                summary.warnings.append(
                    f"Skipped existing district_week_features row for "
                    f"{feature.district_id} {feature.epi_year}W{feature.epi_week:02d}; "
                    "use --replace-existing to overwrite rows from another source."
                )
            summary.districts_processed.add(feature.district_id)

        _append_week_mismatch_warnings(summary, observed_periods_by_district, expected_period_set)
        run.status = "success"
        run.completed_at = _utcnow()
        run.row_count_raw = summary.raw_rows_considered
        run.row_count_processed = summary.inserted + summary.updated
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
            run.row_count_raw = summary.raw_rows_considered
            run.row_count_processed = summary.inserted + summary.updated
            run.warning_count = len(summary.warnings)
            run.error_message = str(exc)[:1000]
            run.manifest = _summary_manifest(summary)
            db.session.commit()
        if isinstance(exc, PreprocessingError):
            raise
        if isinstance(exc, SQLAlchemyError):
            raise PreprocessingError("Database write failed during preprocessing.") from exc
        raise


def _get_successful_dhis2_run(pipeline_run_id: int) -> PipelineRun:
    run = db.session.get(PipelineRun, pipeline_run_id)
    if run is None:
        raise PreprocessingError(f"Pipeline run {pipeline_run_id} was not found.")
    if run.run_type != "dhis2_extraction" or run.status != "success":
        raise PreprocessingError(
            f"Pipeline run {pipeline_run_id} is not a successful DHIS2 extraction run."
        )
    return run


def _periods_from_run_or_rows(run: PipelineRun) -> list[str]:
    parameters = run.parameters or {}
    periods = parameters.get("periods")
    if isinstance(periods, list) and periods:
        return [str(period) for period in periods]
    raw_periods = (
        db.session.query(RawDHIS2Malaria.period)
        .filter_by(pipeline_run_id=run.id)
        .distinct()
        .order_by(RawDHIS2Malaria.period)
        .all()
    )
    if not raw_periods:
        raise PreprocessingError(f"Pipeline run {run.id} has no raw DHIS2 rows.")
    return [row[0] for row in raw_periods]


def _append_week_mismatch_warnings(
    summary: PreprocessingSummary,
    observed_periods_by_district: dict[str, set[str]],
    expected_period_set: set[str],
) -> None:
    if not observed_periods_by_district:
        summary.warnings.append("No mapped raw rows were available for preprocessing.")
        return
    for year in sorted({parse_period(period)[0] for period in expected_period_set}):
        if iso_weeks_in_year(year) == 53:
            week_53 = f"{year}W53"
            if week_53 not in expected_period_set:
                summary.warnings.append(
                    f"ISO year {year} has 53 weeks, but the selected source periods "
                    f"omit {week_53}. This usually means extraction used a 52-week "
                    "request; review the denominator before final analysis."
                )
                continue
            if week_53 in expected_period_set and not any(
                week_53 in periods for periods in observed_periods_by_district.values()
            ):
                summary.warnings.append(
                    f"ISO year {year} has 53 weeks, but {week_53} has no observed "
                    "raw rows. Check whether extraction omitted week 53."
                )


def _summary_manifest(summary: PreprocessingSummary) -> dict[str, Any]:
    return {
        "source_run_ids": summary.source_run_ids,
        "expected_period_count": summary.expected_period_count,
        "raw_rows_considered": summary.raw_rows_considered,
        "raw_rows_mapped": summary.raw_rows_mapped,
        "raw_rows_unmapped": summary.raw_rows_unmapped,
        "unmapped_org_units": sorted(summary.unmapped_org_units),
        "districts_processed": sorted(summary.districts_processed),
        "inserted": summary.inserted,
        "updated": summary.updated,
        "skipped": summary.skipped,
        "warnings": summary.warnings,
    }


def _existing_is_same_source(
    existing: DistrictWeekFeature, selection: PreprocessingSelection
) -> bool:
    if existing.source_pipeline_run_id is None:
        return False
    source_run = db.session.get(PipelineRun, existing.source_pipeline_run_id)
    if source_run is None or source_run.run_type != "preprocessing":
        return False
    parameters = source_run.parameters or {}
    previous_source_ids = parameters.get("source_run_ids")
    return sorted(previous_source_ids or []) == sorted(selection.source_run_ids)


def _integer_cases(value: Decimal) -> int:
    if value < 0:
        raise PreprocessingError("Confirmed malaria cases cannot be negative.")
    if value != value.to_integral_value():
        raise PreprocessingError(
            f"Confirmed malaria case value {value} is not an integer count."
        )
    return int(value)


def _percentage(observed: int, expected: int) -> Decimal:
    if expected <= 0:
        raise PreprocessingError("Expected week denominator must be greater than zero.")
    return (Decimal(observed) / Decimal(expected) * Decimal(100)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
