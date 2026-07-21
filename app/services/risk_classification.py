"""Transparent surge-risk classification for dashboard data."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

NO_DATA_RISK = "No data"
LOW = "Low"
MODERATE = "Moderate"
HIGH = "High"
VERY_HIGH = "Very High"
AT_RISK_LEVELS = {HIGH, VERY_HIGH}


@dataclass(frozen=True)
class RiskClassification:
    risk_level: str
    risk_basis: str
    surge_output_type: str
    no_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "risk_basis": self.risk_basis,
            "surge_output_type": self.surge_output_type,
            "no_data": self.no_data,
        }


def classify_forecast_risk(
    *,
    predicted_surge_flag: bool | None,
    surge_probability: Any,
    predicted_cases: Any,
    threshold_used: Any,
) -> RiskClassification:
    """Classify risk using probability first, with count-threshold fallback."""
    probability = _decimal(surge_probability)
    cases = _decimal(predicted_cases)
    threshold = _decimal(threshold_used)
    if predicted_surge_flag is None and probability is None and cases is None:
        return RiskClassification(
            risk_level=NO_DATA_RISK,
            risk_basis="No forecast output is available for this district/horizon.",
            surge_output_type="unavailable",
            no_data=True,
        )

    if probability is not None:
        if predicted_surge_flag is True and probability >= Decimal("0.75"):
            return RiskClassification(
                VERY_HIGH,
                "Predicted surge flag is true and surge probability is at least 0.75.",
                "trained_classifier",
            )
        if predicted_surge_flag is True or probability >= Decimal("0.60"):
            return RiskClassification(
                HIGH,
                "Predicted surge flag is true or surge probability is at least 0.60.",
                "trained_classifier",
            )
        if probability >= Decimal("0.40"):
            return RiskClassification(
                MODERATE,
                "Surge probability is at least 0.40 but below high-risk criteria.",
                "trained_classifier",
            )
        return RiskClassification(
            LOW,
            "Surge probability is below 0.40 and no high-risk flag is present.",
            "trained_classifier",
        )

    if cases is not None and threshold is not None and threshold > 0:
        ratio = cases / threshold
        if predicted_surge_flag is True and ratio >= Decimal("1.50"):
            return RiskClassification(
                VERY_HIGH,
                "Count-threshold benchmark: predicted cases are at least 1.5x the district threshold and surge flag is true.",
                "derived_count_threshold_benchmark",
            )
        if predicted_surge_flag is True or ratio >= Decimal("1.00"):
            return RiskClassification(
                HIGH,
                "Count-threshold benchmark: predicted cases meet/exceed the district threshold or surge flag is true.",
                "derived_count_threshold_benchmark",
            )
        if ratio >= Decimal("0.80"):
            return RiskClassification(
                MODERATE,
                "Count-threshold benchmark: predicted cases are elevated but below the surge threshold.",
                "derived_count_threshold_benchmark",
            )
        return RiskClassification(
            LOW,
            "Count-threshold benchmark: predicted cases are below elevated-risk criteria.",
            "derived_count_threshold_benchmark",
        )

    if predicted_surge_flag is True:
        return RiskClassification(
            HIGH,
            "Predicted surge flag is true, but probability and threshold details are unavailable.",
            "derived_flag_only",
        )

    return RiskClassification(
        LOW,
        "Forecast exists, but no surge probability or usable threshold is available and no surge flag is present.",
        "limited_forecast_metadata",
    )


def _decimal(value: Any) -> Decimal | None:
    if value in {None, ""}:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
