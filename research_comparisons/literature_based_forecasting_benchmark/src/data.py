"""Experiment-local, lineage-audited, as-of-origin data access.

Reads the canonical modelling CSV READ-ONLY. Never alters it.

Lineage-safety (corrected in run 20260828r1):
- Case predictors use ONLY OBSERVED confirmed cases (never the canonical
  two-sided interpolated ``cases_lag_*`` columns, which can depend on an
  observation after the week they describe). Gaps and the six confirmed errors
  are left missing and imputed by TRAINING-fold preprocessing.
- Every lookup returns the exact source week it consumed, so provenance lineage
  is COMPUTED, not declared.
- Origin-availability convention: origin-week (O) data are NOT used as
  predictors. Callers use case/climate lags >= 1 only; a guard fails before
  prediction if any consumed source week is after the origin.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from typing import Any

from research_comparisons.literature_based_forecasting_benchmark import config as C

ClimateSeries = dict[tuple[str, str], dict[str, float | None]]
ObservedCases = dict[tuple[str, str], float]


def load_canonical_rows() -> list[dict[str, Any]]:
    path = C.CANONICAL_DATASET
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Canonical modelling dataset unavailable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Canonical modelling dataset has no rows: {path}")
    return rows


def load_error_keys() -> set[tuple[str, str, str]]:
    """The six confirmed reporting errors -> (district_id, epi_year, epi_week)."""
    path = C.REPO_ROOT / "data/reference/malaria_case_exclusions.csv"
    keys: set[tuple[str, str, str]] = set()
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                keys.add((str(row["district_id"]), str(row["epi_year"]), str(row["epi_week"])))
    return keys


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _iso(day: date) -> str:
    return day.isoformat()


def week_minus(origin_iso: str, weeks: int) -> str:
    return _iso(date.fromisoformat(origin_iso) - timedelta(weeks=weeks))


def build_observed_cases(rows: list[dict[str, Any]], error_keys: set[tuple[str, str, str]]) -> ObservedCases:
    """(district_id, week_start_date) -> OBSERVED confirmed cases only.

    Excludes the six confirmed errors and any blank/absent value. Each value is
    an observation at its own week, so it can never depend on a later week.
    """
    observed: ObservedCases = {}
    for row in rows:
        district = str(row["district_id"])
        key = (district, str(row["epi_year"]), str(row["epi_week"]))
        if key in error_keys:
            continue
        value = _to_float(row.get("confirmed_malaria_cases"))
        if value is None:
            continue
        observed[(district, str(row["week_start_date"]))] = value
    return observed


def build_climate_series(rows: list[dict[str, Any]]) -> ClimateSeries:
    """(district_id, week_start_date) -> current-week climate (observed at that week)."""
    series: ClimateSeries = {}
    for row in rows:
        key = (str(row["district_id"]), str(row["week_start_date"]))
        series[key] = {
            "rainfall": _to_float(row.get("rainfall_current")),
            "temperature": _to_float(row.get("temperature_current")),
            "humidity": _to_float(row.get("humidity_current")),
        }
    return series


def observed_case_at(observed: ObservedCases, district: str, origin_iso: str, lag: int) -> tuple[float | None, str]:
    """Observed cases at (origin - lag). Returns (value_or_None, source_week)."""
    source_week = week_minus(origin_iso, lag)
    return observed.get((district, source_week)), source_week


def climate_at(series: ClimateSeries, district: str, origin_iso: str, lag: int, var: str) -> tuple[float | None, str]:
    """Climate at (origin - lag). Returns (value_or_None, source_week)."""
    source_week = week_minus(origin_iso, lag)
    entry = series.get((district, source_week))
    return (None if entry is None else entry.get(var)), source_week


def almon_terms(lag_values: list[float | None], degree: int) -> list[float] | None:
    """degree-th Almon polynomial-distributed-lag transform, or None if any lag missing."""
    if any(value is None for value in lag_values):
        return None
    return [float(sum((k ** j) * value for k, value in enumerate(lag_values))) for j in range(degree + 1)]


def district_rows_upto(train_rows: list[dict[str, Any]], district_id: str, origin_iso: str) -> list[dict[str, Any]]:
    """District rows dated on/before the origin (for district-level ARIMAX)."""
    return sorted(
        (row for row in train_rows
         if str(row["district_id"]) == district_id and str(row["week_start_date"]) <= origin_iso),
        key=lambda row: str(row["week_start_date"]),
    )


class Lineage:
    """Accumulates the ACTUAL source weeks consumed by a prediction's features."""

    def __init__(self, origin_iso: str) -> None:
        self.origin = origin_iso
        self.max_case_source: str | None = None
        self.max_climate_source: str | None = None
        self.records: list[dict[str, Any]] = []

    def add(self, family: str, kind: str, source_week: str, observed: bool) -> None:
        status = "observed" if observed else "training_imputed_or_unavailable"
        self.records.append({"family": family, "source_week": source_week, "status": status})
        if observed:
            if kind == "case":
                self.max_case_source = source_week if self.max_case_source is None else max(self.max_case_source, source_week)
            elif kind == "climate":
                self.max_climate_source = source_week if self.max_climate_source is None else max(self.max_climate_source, source_week)

    def max_feature_source(self) -> str | None:
        candidates = [d for d in (self.max_case_source, self.max_climate_source) if d is not None]
        return max(candidates) if candidates else None

    def violates_origin(self) -> bool:
        # ISO date strings compare chronologically.
        for source in (self.max_case_source, self.max_climate_source):
            if source is not None and source > self.origin:
                return True
        return False


def row_lineage(
    district: str,
    origin_iso: str,
    *,
    case_lags: tuple[int, ...],
    climate_lags: dict[str, tuple[int, ...]],
    observed: ObservedCases,
    climate_series: ClimateSeries,
) -> Lineage:
    """Compute the real source-date lineage for one prediction row's features."""
    lin = Lineage(origin_iso)
    for lag in case_lags:
        value, source = observed_case_at(observed, district, origin_iso, lag)
        lin.add(f"case_lag_{lag}", "case", source, value is not None)
    for var, lags in climate_lags.items():
        for lag in lags:
            value, source = climate_at(climate_series, district, origin_iso, lag, var)
            lin.add(f"{var}_lag_{lag}", "climate", source, value is not None)
    return lin


def lineage_summary(lin: Lineage) -> dict[str, Any]:
    return {
        "max_case_source_date": lin.max_case_source,
        "max_climate_source_date": lin.max_climate_source,
        "max_feature_source_date": lin.max_feature_source(),
        "violates_origin": lin.violates_origin(),
    }
