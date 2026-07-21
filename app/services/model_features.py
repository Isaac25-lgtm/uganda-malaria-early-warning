"""Leakage-safe model feature selection and in-fold preprocessing helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

IDENTIFIER_COLUMNS = {
    "district_id",
    "district_name",
    "week_start_date",
    "week_end_date",
}
FORBIDDEN_PREFIXES = ("target_",)
FORBIDDEN_EXACT = {
    "surge_observed",
    "risk_level",
    "predicted_cases",
    "surge_probability",
    "predicted_surge_flag",
}
ALLOWED_CATEGORICAL = {"district_id"}


class ModelFeatureError(RuntimeError):
    """Raised when unsafe model features are requested."""


@dataclass
class FeatureSpec:
    feature_columns: list[str]
    forbidden_columns_excluded: list[str]
    categorical_columns: list[str] = field(default_factory=list)


class TrainingOnlyImputer:
    """Simple median imputer fit only on training rows."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.statistics_: dict[str, Decimal] = {}
        self.fit_row_count: int = 0

    def fit(self, rows: list[dict[str, Any]]) -> "TrainingOnlyImputer":
        self.fit_row_count = len(rows)
        for column in self.columns:
            values = sorted(
                _decimal(row.get(column)) for row in rows if _decimal(row.get(column)) is not None
            )
            if not values:
                self.statistics_[column] = Decimal("0")
                continue
            midpoint = len(values) // 2
            if len(values) % 2:
                self.statistics_[column] = values[midpoint]
            else:
                self.statistics_[column] = (values[midpoint - 1] + values[midpoint]) / Decimal("2")
        return self

    def transform(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        transformed: list[dict[str, Any]] = []
        for row in rows:
            copy = dict(row)
            for column in self.columns:
                if _decimal(copy.get(column)) is None:
                    copy[column] = str(self.statistics_.get(column, Decimal("0")))
                    copy[f"{column}_was_missing"] = "1"
                else:
                    copy[f"{column}_was_missing"] = "0"
            transformed.append(copy)
        return transformed


def build_feature_spec(columns: list[str]) -> FeatureSpec:
    """Return leakage-safe feature columns from a modelling dataset header."""
    forbidden: list[str] = []
    feature_columns: list[str] = []
    for column in columns:
        lowered = column.lower()
        if "wind" in lowered:
            raise ModelFeatureError("Wind-speed features are not allowed in modelling.")
        if (
            column in IDENTIFIER_COLUMNS
            or column in FORBIDDEN_EXACT
            or column.startswith(FORBIDDEN_PREFIXES)
        ):
            forbidden.append(column)
            continue
        feature_columns.append(column)
    if any(column.startswith("target_") for column in feature_columns):
        raise ModelFeatureError("Target columns leaked into model features.")
    return FeatureSpec(
        feature_columns=feature_columns,
        forbidden_columns_excluded=forbidden,
        categorical_columns=[column for column in feature_columns if column in ALLOWED_CATEGORICAL],
    )


def numeric_feature_columns(rows: list[dict[str, Any]], feature_columns: list[str]) -> list[str]:
    numeric: list[str] = []
    for column in feature_columns:
        if column in ALLOWED_CATEGORICAL:
            continue
        if any(_decimal(row.get(column)) is not None for row in rows):
            numeric.append(column)
    return numeric


def _decimal(value: Any) -> Decimal | None:
    if value in {None, ""}:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
