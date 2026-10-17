"""Adjudicated malaria-case cleaning for analytical preprocessing.

Raw DHIS2 rows remain immutable. Confirmed reporting errors are masked only in
the analytical series, and short bounded gaps may be interpolated only for
lagged/rolling predictor construction. Interpolated values are never outcomes,
validation truth, or threshold inputs.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

EXCLUSION_REGISTRY_PATH = Path("data/reference/malaria_case_exclusions.csv")
MAX_INTERPOLATION_GAP_WEEKS = 2
REQUIRED_COLUMNS = (
    "district_id",
    "district_name",
    "epi_year",
    "epi_week",
    "original_cases",
    "previous_cases",
    "following_cases",
    "reason_code",
    "adjudication_status",
    "consultation_role",
    "consultation_reference",
    "analytical_action",
)


class CaseDataCleaningError(RuntimeError):
    """Raised when adjudicated cleaning cannot be applied reproducibly."""


@dataclass(frozen=True)
class CaseExclusion:
    district_id: str
    district_name: str
    epi_year: int
    epi_week: int
    original_cases: Decimal
    previous_cases: Decimal
    following_cases: Decimal
    reason_code: str
    adjudication_status: str
    consultation_role: str
    consultation_reference: str
    analytical_action: str

    @property
    def key(self) -> tuple[str, int, int]:
        return self.district_id, self.epi_year, self.epi_week


@dataclass(frozen=True)
class CleanCaseValue:
    observed_cases: Decimal | None
    cases_for_features: Decimal | None
    status: str
    predictor_interpolated: bool
    interpolation_gap_weeks: int | None = None
    feature_available_from: date | None = None


@dataclass(frozen=True)
class CaseCleaningResult:
    values: dict[tuple[str, date], CleanCaseValue]
    exclusions: tuple[CaseExclusion, ...]
    source_missing_count: int
    confirmed_error_count: int
    predictor_interpolated_source_missing_count: int
    predictor_interpolated_confirmed_error_count: int
    registry_sha256: str

    def summary(self) -> dict[str, Any]:
        return {
            "source_missing_district_weeks": self.source_missing_count,
            "confirmed_erroneous_district_weeks": self.confirmed_error_count,
            "analytical_missing_outcomes": self.source_missing_count
            + self.confirmed_error_count,
            "predictor_only_interpolated_source_missing": (
                self.predictor_interpolated_source_missing_count
            ),
            "predictor_only_interpolated_confirmed_errors": (
                self.predictor_interpolated_confirmed_error_count
            ),
            "predictor_only_interpolated_total": (
                self.predictor_interpolated_source_missing_count
                + self.predictor_interpolated_confirmed_error_count
            ),
            "outcome_imputation": False,
            "maximum_interpolated_gap_weeks": MAX_INTERPOLATION_GAP_WEEKS,
            "registry_path": str(EXCLUSION_REGISTRY_PATH),
            "registry_sha256": self.registry_sha256,
            "excluded_records": [
                {
                    "district_id": item.district_id,
                    "district_name": item.district_name,
                    "epi_year": item.epi_year,
                    "epi_week": item.epi_week,
                    "original_cases": str(item.original_cases),
                    "previous_cases": str(item.previous_cases),
                    "following_cases": str(item.following_cases),
                    "reason_code": item.reason_code,
                    "adjudication_status": item.adjudication_status,
                    "consultation_role": item.consultation_role,
                    "consultation_reference": item.consultation_reference,
                    "analytical_action": item.analytical_action,
                }
                for item in self.exclusions
            ],
        }


def load_case_exclusions(
    path: Path = EXCLUSION_REGISTRY_PATH,
) -> tuple[CaseExclusion, ...]:
    if not path.exists():
        raise CaseDataCleaningError(f"Case-exclusion registry is missing: {path}.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CaseDataCleaningError("Case-exclusion registry has no header.")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise CaseDataCleaningError(
                "Case-exclusion registry is missing columns: " + ", ".join(missing)
            )
        rows = list(reader)
    if not rows:
        raise CaseDataCleaningError("Case-exclusion registry is empty.")

    exclusions: list[CaseExclusion] = []
    seen: set[tuple[str, int, int]] = set()
    for line_number, row in enumerate(rows, start=2):
        try:
            exclusion = CaseExclusion(
                district_id=_required(row, "district_id", line_number),
                district_name=_required(row, "district_name", line_number),
                epi_year=int(_required(row, "epi_year", line_number)),
                epi_week=int(_required(row, "epi_week", line_number)),
                original_cases=_decimal(row.get("original_cases"), "original_cases", line_number),
                previous_cases=_decimal(row.get("previous_cases"), "previous_cases", line_number),
                following_cases=_decimal(row.get("following_cases"), "following_cases", line_number),
                reason_code=_required(row, "reason_code", line_number),
                adjudication_status=_required(row, "adjudication_status", line_number),
                consultation_role=_required(row, "consultation_role", line_number),
                consultation_reference=_required(
                    row, "consultation_reference", line_number
                ),
                analytical_action=_required(row, "analytical_action", line_number),
            )
        except ValueError as exc:
            raise CaseDataCleaningError(
                f"Invalid numeric value on exclusion-registry line {line_number}."
            ) from exc
        if exclusion.epi_week < 1 or exclusion.epi_week > 53:
            raise CaseDataCleaningError(
                f"Invalid epidemiological week on exclusion-registry line {line_number}."
            )
        if exclusion.adjudication_status != "confirmed":
            raise CaseDataCleaningError(
                f"Exclusion {exclusion.key} is not confirmed and cannot be applied."
            )
        if exclusion.reason_code != "confirmed_reporting_error":
            raise CaseDataCleaningError(
                f"Exclusion {exclusion.key} has an unsupported reason code."
            )
        if exclusion.analytical_action != "exclude_outcome_interpolate_predictors":
            raise CaseDataCleaningError(
                f"Exclusion {exclusion.key} has an unsupported analytical action."
            )
        if exclusion.key in seen:
            raise CaseDataCleaningError(
                f"Duplicate exclusion-registry key: {exclusion.key}."
            )
        seen.add(exclusion.key)
        exclusions.append(exclusion)
    return tuple(sorted(exclusions, key=lambda item: item.key))


def build_clean_case_series(
    source_rows: list[dict[str, Any]],
    *,
    district_ids: list[str],
    start_date: date,
    end_date: date,
    exclusion_path: Path = EXCLUSION_REGISTRY_PATH,
) -> CaseCleaningResult:
    """Build observed and predictor-only case series on a complete weekly frame."""
    if start_date > end_date:
        raise CaseDataCleaningError("Cleaning start date must not exceed end date.")
    exclusions = load_case_exclusions(exclusion_path)
    exclusion_by_key = {item.key: item for item in exclusions}
    observed: dict[tuple[str, date], Decimal | None] = {}
    source_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in source_rows:
        district_id = str(row["district_id"])
        row_date = _date(row.get("week_start_date"))
        key = (district_id, int(row["epi_year"]), int(row["epi_week"]))
        if key in source_by_key:
            raise CaseDataCleaningError(f"Duplicate analytical source row: {key}.")
        source_by_key[key] = row
        observed[(district_id, row_date)] = _optional_decimal(
            row.get("confirmed_malaria_cases")
        )

    for exclusion in exclusions:
        row = source_by_key.get(exclusion.key)
        if row is None:
            raise CaseDataCleaningError(
                f"Confirmed exclusion is absent from the analytical source: {exclusion.key}."
            )
        actual = _optional_decimal(row.get("confirmed_malaria_cases"))
        if actual != exclusion.original_cases:
            raise CaseDataCleaningError(
                f"Confirmed exclusion {exclusion.key} expected {exclusion.original_cases} "
                f"cases but source contains {actual}."
            )
        row_date = _date(row.get("week_start_date"))
        previous = observed.get((exclusion.district_id, row_date - timedelta(weeks=1)))
        following = observed.get((exclusion.district_id, row_date + timedelta(weeks=1)))
        if previous != exclusion.previous_cases or following != exclusion.following_cases:
            raise CaseDataCleaningError(
                f"Adjacent-value evidence changed for confirmed exclusion {exclusion.key}."
            )
        observed[(exclusion.district_id, row_date)] = None

    weekly_dates: list[date] = []
    cursor = start_date
    while cursor <= end_date:
        weekly_dates.append(cursor)
        cursor += timedelta(weeks=1)

    values: dict[tuple[str, date], CleanCaseValue] = {}
    source_missing = 0
    interpolated_source = 0
    interpolated_errors = 0
    excluded_date_keys = {
        (
            item.district_id,
            date.fromisocalendar(item.epi_year, item.epi_week, 1),
        )
        for item in exclusions
    }
    for district_id in district_ids:
        series = [observed.get((district_id, item)) for item in weekly_dates]
        present_dates = {
            _date(row["week_start_date"])
            for key, row in source_by_key.items()
            if key[0] == district_id
        }
        for index, row_date in enumerate(weekly_dates):
            is_excluded = (district_id, row_date) in excluded_date_keys
            raw_present = row_date in present_dates
            observed_value = series[index]
            if not raw_present:
                source_missing += 1
            interpolated, gap_size = _bounded_linear_interpolation(series, index)
            use_interpolation = observed_value is None and interpolated is not None
            feature_available_from = None
            if use_interpolation:
                right = index + 1
                while series[right] is None:
                    right += 1
                feature_available_from = weekly_dates[right]
            if use_interpolation:
                if is_excluded:
                    interpolated_errors += 1
                elif not raw_present:
                    interpolated_source += 1
            status = (
                "confirmed_error_excluded"
                if is_excluded
                else "source_missing"
                if not raw_present
                else "observed_valid"
                if observed_value is not None
                else "source_null"
            )
            values[(district_id, row_date)] = CleanCaseValue(
                observed_cases=observed_value,
                cases_for_features=(
                    interpolated if use_interpolation else observed_value
                ),
                status=status,
                predictor_interpolated=use_interpolation,
                interpolation_gap_weeks=gap_size if use_interpolation else None,
                feature_available_from=feature_available_from,
            )

    registry_hash = hashlib.sha256(exclusion_path.read_bytes()).hexdigest()
    return CaseCleaningResult(
        values=values,
        exclusions=exclusions,
        source_missing_count=source_missing,
        confirmed_error_count=len(exclusions),
        predictor_interpolated_source_missing_count=interpolated_source,
        predictor_interpolated_confirmed_error_count=interpolated_errors,
        registry_sha256=registry_hash,
    )


def _bounded_linear_interpolation(
    series: list[Decimal | None], index: int
) -> tuple[Decimal | None, int | None]:
    if series[index] is not None:
        return None, None
    left = index - 1
    while left >= 0 and series[left] is None:
        left -= 1
    right = index + 1
    while right < len(series) and series[right] is None:
        right += 1
    if left < 0 or right >= len(series):
        return None, None
    gap_size = right - left - 1
    if gap_size < 1 or gap_size > MAX_INTERPOLATION_GAP_WEEKS:
        return None, None
    fraction = Decimal(index - left) / Decimal(right - left)
    value = series[left] + (series[right] - series[left]) * fraction
    return value.quantize(Decimal("0.000001")), gap_size


def _required(row: dict[str, Any], column: str, line_number: int) -> str:
    value = str(row.get(column) or "").strip()
    if not value:
        raise CaseDataCleaningError(
            f"Case-exclusion registry line {line_number} has blank {column}."
        )
    return value


def _decimal(value: Any, column: str, line_number: int) -> Decimal:
    parsed = _optional_decimal(value)
    if parsed is None or parsed < 0:
        raise CaseDataCleaningError(
            f"Case-exclusion registry line {line_number} has invalid {column}."
        )
    return parsed


def _optional_decimal(value: Any) -> Decimal | None:
    if value in {None, ""}:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CaseDataCleaningError(f"Invalid malaria case value: {value!r}.") from exc


def _date(value: Any) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise CaseDataCleaningError(f"Invalid week start date: {value!r}.") from exc
