"""Metric + prediction-record helpers — all reuse canonical logic.

Count metrics (WAPE primary, MAE, RMSE, R2, MAPE zero-safe) and the Q75 warning
workflow come straight from app.services.model_validation. WAPE is an error
measure; never reported as accuracy; never 100 - WAPE.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from app.services.model_validation import (  # canonical, read-only
    PredictionRecord,
    classification_metrics,
    district_threshold,
    evaluate_predictions,
)


def build_record(
    row: dict[str, Any],
    horizon: int,
    predicted_value: float,
    train_rows: list[dict[str, Any]],
) -> PredictionRecord:
    """One prediction -> canonical PredictionRecord using the Q75 threshold.

    Warning workflow: predicted count -> existing fold-specific district Q75
    threshold -> warning label (predicted_surge = predicted > threshold).
    """
    target_column = f"target_cases_h{horizon}"
    actual = Decimal(str(row[target_column]))
    predicted = Decimal(str(max(0.0, float(predicted_value)))).quantize(Decimal("0.000001"))
    threshold = district_threshold(
        train_rows,
        str(row["district_id"]),
        target_epi_week=int(row[f"target_epi_week_h{horizon}"]),
    )
    score = (predicted - threshold).quantize(Decimal("0.000001"))
    return PredictionRecord(
        district_id=str(row["district_id"]),
        horizon=horizon,
        origin_date=date.fromisoformat(str(row["week_start_date"])),
        target_date=date.fromisoformat(str(row[f"target_week_start_date_h{horizon}"])),
        actual=actual,
        predicted=predicted,
        threshold=threshold,
        actual_surge=actual > threshold,
        predicted_surge=score > 0,
        surge_score=score,
    )


def evaluate(records: list[PredictionRecord]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Canonical count + warning metrics for a set of records."""
    return evaluate_predictions(records)


def warning_metrics(records: list[PredictionRecord]) -> dict[str, Any]:
    return classification_metrics(
        [r.actual_surge for r in records],
        [r.predicted_surge for r in records],
        [r.surge_score for r in records],
    )


def paired_wape(records: list[PredictionRecord]) -> float | None:
    total_actual = sum((r.actual for r in records), Decimal("0"))
    if total_actual == 0:
        return None
    total_error = sum((abs(r.predicted - r.actual) for r in records), Decimal("0"))
    return float(total_error / total_actual * Decimal("100"))


def paired_mae(records: list[PredictionRecord]) -> float | None:
    if not records:
        return None
    total_error = sum((abs(r.predicted - r.actual) for r in records), Decimal("0"))
    return float(total_error / Decimal(len(records)))
