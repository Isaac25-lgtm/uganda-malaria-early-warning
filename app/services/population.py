"""UBOS population denominator application service."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

from app.extensions import db
from app.models import DistrictWeekFeature, UBOSPopulation


class PopulationError(RuntimeError):
    """Raised when population denominators cannot be applied safely."""


@dataclass
class PopulationApplySummary:
    year: int
    dry_run: bool
    denominator_year: int | None = None
    target_years: list[int] = field(default_factory=list)
    fixed_denominator_assumption: str | None = None
    feature_rows_considered: int = 0
    denominator_count: int = 0
    rows_to_update: int = 0
    rows_updated: int = 0
    rows_unchanged: int = 0
    missing_denominator_rows: int = 0
    districts_missing_denominator: set[str] = field(default_factory=set)
    incidence_computed: int = 0
    incidence_left_null: int = 0
    warnings: list[str] = field(default_factory=list)


FIXED_2024_DENOMINATOR_ASSUMPTION = (
    "UBOS 2024 district/city population denominators are used as fixed "
    "denominators across the 2020-2025 analysis period for incidence "
    "standardisation because year-specific district population projections are "
    "not currently included in the project dataset."
)


def apply_population_to_features(
    year: int | None = None,
    *,
    use_year: int | None = None,
    apply_to_years: Iterable[int] | None = None,
    dry_run: bool = False,
) -> PopulationApplySummary:
    """Apply approved UBOS denominators to district_week_features.

    ``year`` preserves the same-year workflow. Cross-year use requires callers
    to pass ``use_year`` and an explicit ``apply_to_years`` iterable.
    """
    if year is not None and use_year is not None:
        raise PopulationError("Use either year or use_year, not both.")
    denominator_year = use_year if use_year is not None else year
    if denominator_year is None:
        raise PopulationError("A denominator year is required.")
    _validate_year(denominator_year, label="denominator year")

    target_years = sorted(set(apply_to_years or [denominator_year]))
    if not target_years:
        raise PopulationError("At least one target epi year is required.")
    for target_year in target_years:
        _validate_year(target_year, label="target epi year")

    cross_year = any(target_year != denominator_year for target_year in target_years)
    if cross_year and use_year is None:
        raise PopulationError(
            "Cross-year denominator application requires explicit use_year."
        )
    fixed_assumption = None
    if cross_year:
        approved_years = set(range(2020, 2026))
        if denominator_year != 2024 or not set(target_years).issubset(approved_years):
            raise PopulationError(
                "Cross-year denominator application is currently approved only for "
                "UBOS 2024 applied explicitly within the 2020-2025 analysis period."
            )
        fixed_assumption = FIXED_2024_DENOMINATOR_ASSUMPTION

    denominators = {
        row.district_id: row
        for row in UBOSPopulation.query.filter_by(year=denominator_year).all()
    }
    summary = PopulationApplySummary(
        year=denominator_year,
        dry_run=dry_run,
        denominator_year=denominator_year,
        target_years=target_years,
        fixed_denominator_assumption=fixed_assumption,
        denominator_count=len(denominators),
    )
    if not denominators:
        summary.warnings.append(
            f"No UBOS population denominators found for {denominator_year}; "
            "no feature rows updated."
        )

    features = (
        DistrictWeekFeature.query.filter(DistrictWeekFeature.epi_year.in_(target_years))
        .order_by(DistrictWeekFeature.district_id, DistrictWeekFeature.epi_week)
        .all()
    )
    summary.feature_rows_considered = len(features)
    if not features:
        summary.warnings.append(
            "No district_week_features rows found for the requested target year(s); "
            "population/incidence application remains pending preprocessing data."
        )

    for feature in features:
        denominator = denominators.get(feature.district_id)
        if denominator is None:
            summary.missing_denominator_rows += 1
            summary.districts_missing_denominator.add(feature.district_id)
            continue

        incidence = None
        if feature.confirmed_malaria_cases is not None:
            incidence = _incidence_per_1000(
                feature.confirmed_malaria_cases, denominator.population
            )
            summary.incidence_computed += 1
        else:
            summary.incidence_left_null += 1

        needs_update = (
            feature.population != denominator.population
            or feature.incidence_per_1000 != incidence
        )
        if needs_update:
            summary.rows_to_update += 1
            if not dry_run:
                feature.population = denominator.population
                feature.incidence_per_1000 = incidence
                summary.rows_updated += 1
        else:
            summary.rows_unchanged += 1

    if summary.districts_missing_denominator:
        summary.warnings.append(
            f"{len(summary.districts_missing_denominator)} district(s) with feature "
            f"rows lack UBOS denominators for {denominator_year}."
        )

    if not dry_run and summary.rows_to_update:
        db.session.commit()

    return summary


def _validate_year(year: int, *, label: str) -> None:
    if year < 1900 or year > 2100:
        raise PopulationError(f"{label} must be between 1900 and 2100.")


def _incidence_per_1000(cases: int, population: int) -> Decimal:
    if population <= 0:
        raise PopulationError("population must be positive.")
    if cases < 0:
        raise PopulationError("confirmed_malaria_cases cannot be negative.")
    return (Decimal(cases) / Decimal(population) * Decimal(1000)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )
