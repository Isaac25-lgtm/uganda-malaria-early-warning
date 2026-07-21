"""Shared leakage-safe rolling-origin validation utilities."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from app.services.modelling_dataset import HORIZONS

SURGE_QUANTILE = Decimal("0.75")
SURGE_SEASONAL_WINDOW_WEEKS = 2
SURGE_MINIMUM_SEASONAL_OBSERVATIONS = 8
SURGE_REFERENCE_EXCLUDED_YEARS = frozenset({2022})


class ModelValidationError(RuntimeError):
    """Raised when validation would leak future information."""


@dataclass(frozen=True)
class ValidationFold:
    horizon: int
    origin_date: date
    target_date: date
    train_rows: list[dict[str, Any]]
    trainable_rows: list[dict[str, Any]]
    validation_rows: list[dict[str, Any]]


@dataclass
class PredictionRecord:
    district_id: str
    horizon: int
    origin_date: date
    target_date: date
    actual: Decimal
    predicted: Decimal
    threshold: Decimal
    actual_surge: bool
    predicted_surge: bool
    surge_score: Decimal | None = None


@dataclass
class ValidationSummary:
    model_name: str
    model_scope: str
    horizons: tuple[int, ...]
    dry_run: bool
    status: str = "planned"
    row_counts: dict[str, int] = field(default_factory=dict)
    validation_cutoffs: list[dict[str, Any]] = field(default_factory=list)
    horizon_gap_proof: list[dict[str, Any]] = field(default_factory=list)
    threshold_proof: list[dict[str, Any]] = field(default_factory=list)
    count_metrics_by_horizon: dict[int, dict[str, Any]] = field(default_factory=dict)
    surge_metrics_by_horizon: dict[int, dict[str, Any]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    forbidden_columns_excluded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_scope": self.model_scope,
            "horizons": list(self.horizons),
            "dry_run": self.dry_run,
            "status": self.status,
            "row_counts": self.row_counts,
            "validation_cutoffs": self.validation_cutoffs,
            "horizon_gap_proof": self.horizon_gap_proof,
            "threshold_proof": self.threshold_proof,
            "count_metrics_by_horizon": self.count_metrics_by_horizon,
            "surge_metrics_by_horizon": self.surge_metrics_by_horizon,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "feature_columns": self.feature_columns,
            "forbidden_columns_excluded": self.forbidden_columns_excluded,
            "validation_method": "rolling_origin",
            "no_random_shuffle": True,
            "per_origin_threshold_training_only": True,
            "horizon_aware_gap": True,
            "trainable_targets_observed_by_origin": True,
            "no_wind": not any("wind" in column.lower() for column in self.feature_columns),
        }


def build_rolling_origin_folds(
    rows: list[dict[str, Any]],
    *,
    horizon: int,
    min_train_weeks: int = 52,
    max_folds: int = 12,
) -> list[ValidationFold]:
    if horizon not in HORIZONS:
        raise ModelValidationError(f"Unsupported horizon {horizon}.")
    unique_dates = sorted({date.fromisoformat(str(row["week_start_date"])) for row in rows})
    target_column = f"target_week_start_date_h{horizon}"
    cases_column = f"target_cases_h{horizon}"
    validation_by_target: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        target = row.get(target_column)
        if target in {None, ""} or row.get(cases_column) in {None, ""}:
            continue
        validation_by_target.setdefault(str(target), []).append(row)

    eligible_origins = [
        origin_date
        for origin_date in unique_dates[min_train_weeks:]
        if (origin_date + timedelta(weeks=horizon)).isoformat() in validation_by_target
    ]
    if max_folds and len(eligible_origins) > max_folds:
        eligible_origins = _evenly_spaced_origins(eligible_origins, max_folds)

    folds: list[ValidationFold] = []
    for origin_date in eligible_origins:
        target_date = origin_date + timedelta(weeks=horizon)
        train_rows = [
            row
            for row in rows
            if date.fromisoformat(str(row["week_start_date"])) <= origin_date
        ]
        trainable_rows = [
            row
            for row in train_rows
            if row.get(target_column) not in {None, ""}
            and date.fromisoformat(str(row[target_column])) <= origin_date
            and row.get(cases_column) not in {None, ""}
        ]
        validation_rows = validation_by_target[target_date.isoformat()]
        folds.append(
            ValidationFold(
                horizon=horizon,
                origin_date=origin_date,
                target_date=target_date,
                train_rows=train_rows,
                trainable_rows=trainable_rows,
                validation_rows=validation_rows,
            )
        )
    _assert_folds_are_leakage_safe(folds)
    return folds


def district_threshold(
    train_rows: list[dict[str, Any]],
    district_id: str,
    *,
    target_epi_week: int | None = None,
    quantile: Decimal = SURGE_QUANTILE,
) -> Decimal:
    """Return a training-only district threshold with seasonal context.

    Confirmed epidemic-year observations remain available to count models but
    are excluded from the normal-reference threshold. When fewer than eight
    observations exist in the target-week +/-2 window, the function falls back
    to the district's eligible expanding history and never borrows other
    districts or future rows.
    """
    eligible_rows = [
        row
        for row in train_rows
        if str(row.get("district_id")) == district_id
        and _decimal(row.get("confirmed_malaria_cases")) is not None
        and _row_epi_year(row) not in SURGE_REFERENCE_EXCLUDED_YEARS
    ]
    selected_rows = eligible_rows
    if target_epi_week is not None:
        if target_epi_week < 1 or target_epi_week > 53:
            raise ModelValidationError(f"Invalid target epidemiological week {target_epi_week}.")
        seasonal_rows = [
            row
            for row in eligible_rows
            if _circular_week_distance(_row_epi_week(row), target_epi_week)
            <= SURGE_SEASONAL_WINDOW_WEEKS
        ]
        if len(seasonal_rows) >= SURGE_MINIMUM_SEASONAL_OBSERVATIONS:
            selected_rows = seasonal_rows
    values = sorted(
        value
        for row in selected_rows
        if (value := _decimal(row.get("confirmed_malaria_cases"))) is not None
    )
    if not values:
        return Decimal("0")
    index = int((Decimal(len(values) - 1) * quantile).to_integral_value(rounding="ROUND_FLOOR"))
    return values[index]


def district_threshold_proof(
    train_rows: list[dict[str, Any]], district_id: str, *, target_epi_week: int
) -> dict[str, Any]:
    eligible = [
        row
        for row in train_rows
        if str(row.get("district_id")) == district_id
        and _decimal(row.get("confirmed_malaria_cases")) is not None
        and _row_epi_year(row) not in SURGE_REFERENCE_EXCLUDED_YEARS
    ]
    seasonal = [
        row
        for row in eligible
        if _circular_week_distance(_row_epi_week(row), target_epi_week)
        <= SURGE_SEASONAL_WINDOW_WEEKS
    ]
    seasonal_used = len(seasonal) >= SURGE_MINIMUM_SEASONAL_OBSERVATIONS
    return {
        "district_id": district_id,
        "target_epi_week": target_epi_week,
        "quantile": str(SURGE_QUANTILE),
        "seasonal_window_weeks_each_side": SURGE_SEASONAL_WINDOW_WEEKS,
        "minimum_seasonal_observations": SURGE_MINIMUM_SEASONAL_OBSERVATIONS,
        "eligible_history_count": len(eligible),
        "seasonal_history_count": len(seasonal),
        "seasonal_window_used": seasonal_used,
        "fallback": None if seasonal_used else "district_expanding_history",
        "excluded_reference_years": sorted(SURGE_REFERENCE_EXCLUDED_YEARS),
        "threshold": str(
            district_threshold(
                train_rows, district_id, target_epi_week=target_epi_week
            )
        ),
    }


def evaluate_predictions(records: list[PredictionRecord]) -> tuple[dict[str, Any], dict[str, Any]]:
    actuals = [record.actual for record in records]
    preds = [record.predicted for record in records]
    errors = [abs(pred - actual) for pred, actual in zip(preds, actuals)]
    mae = _mean(errors)
    rmse = _sqrt(_mean([(pred - actual) ** 2 for pred, actual in zip(preds, actuals)]))
    nonzero_pairs = [(pred, actual) for pred, actual in zip(preds, actuals) if actual != 0]
    mape = (
        _mean([abs((pred - actual) / actual) * Decimal("100") for pred, actual in nonzero_pairs])
        if nonzero_pairs
        else None
    )
    total_actual = sum(actuals, Decimal("0"))
    wape = (sum(errors, Decimal("0")) / total_actual * Decimal("100")) if total_actual else None
    r2 = _r_squared(actuals, preds)
    count_metrics = {
        "n": len(records),
        "mae": _text(mae),
        "rmse": _text(rmse),
        "mape": _text(mape),
        "wape": _text(wape),
        "r_squared": _text(r2),
    }
    surge_metrics = classification_metrics(
        [record.actual_surge for record in records],
        [record.predicted_surge for record in records],
        [record.surge_score for record in records],
    )
    return count_metrics, surge_metrics


def classification_metrics(
    actual: list[bool],
    predicted: list[bool],
    scores: list[Decimal | None] | None = None,
) -> dict[str, Any]:
    counts = Counter(zip(actual, predicted))
    tp = counts[(True, True)]
    tn = counts[(False, False)]
    fp = counts[(False, True)]
    fn = counts[(True, False)]
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    accuracy = _ratio(tp + tn, tp + tn + fp + fn)
    auc = _auc(actual, scores or [])
    f2 = _ratio(5 * tp, 5 * tp + 4 * fn + fp)
    pr_auc = _pr_auc(actual, scores or [])
    brier_score = _brier_score(actual, scores or [])
    return {
        "n": len(actual),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": _text(sensitivity),
        "specificity": _text(specificity),
        "precision": _text(precision),
        "f1": _text(f1),
        "f2": _text(f2),
        "accuracy": _text(accuracy),
        "auc": _text(auc),
        "pr_auc": _text(pr_auc),
        "brier_score": _text(brier_score),
    }


def write_validation_artifact(path: Path, summary: ValidationSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _evenly_spaced_origins(origins: list[date], count: int) -> list[date]:
    if count <= 0 or len(origins) <= count:
        return origins
    if count == 1:
        return [origins[-1]]
    indices = [round(index * (len(origins) - 1) / (count - 1)) for index in range(count)]
    return [origins[index] for index in indices]


def _row_epi_year(row: dict[str, Any]) -> int:
    value = row.get("epi_year")
    if value not in {None, ""}:
        return int(value)
    if row.get("week_start_date") not in {None, ""}:
        return date.fromisoformat(str(row["week_start_date"])).isocalendar().year
    return 0


def _row_epi_week(row: dict[str, Any]) -> int:
    value = row.get("epi_week")
    if value not in {None, ""}:
        return int(value)
    if row.get("week_start_date") not in {None, ""}:
        return date.fromisoformat(str(row["week_start_date"])).isocalendar().week
    raise ModelValidationError("Threshold history row has no epidemiological week.")


def _circular_week_distance(first: int, second: int) -> int:
    direct = abs(first - second)
    return min(direct, 53 - direct)


def _assert_folds_are_leakage_safe(folds: list[ValidationFold]) -> None:
    for fold in folds:
        if fold.target_date != fold.origin_date + timedelta(weeks=fold.horizon):
            raise ModelValidationError("Target date does not match origin + horizon.")
        if any(date.fromisoformat(str(row["week_start_date"])) > fold.origin_date for row in fold.train_rows):
            raise ModelValidationError("Training fold includes rows after the origin date.")
        if any(
            date.fromisoformat(str(row[f"target_week_start_date_h{fold.horizon}"])) > fold.origin_date
            for row in fold.trainable_rows
        ):
            raise ModelValidationError("Trainable fold includes target labels after the origin date.")
        if any(
            date.fromisoformat(str(row["week_start_date"])) > fold.origin_date
            for row in fold.validation_rows
        ):
            # Validation rows are feature rows at the origin, not target-period rows.
            raise ModelValidationError("Validation feature row occurs after the origin date.")


def _decimal(value: Any) -> Decimal | None:
    if value in {None, ""}:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _mean(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    return (sum(values, Decimal("0")) / Decimal(len(values))).quantize(Decimal("0.000001"))


def _sqrt(value: Decimal | None) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(math.sqrt(float(value)))).quantize(Decimal("0.000001"))


def _r_squared(actuals: list[Decimal], preds: list[Decimal]) -> Decimal | None:
    if len(actuals) < 2:
        return None
    mean_actual = sum(actuals, Decimal("0")) / Decimal(len(actuals))
    ss_tot = sum((actual - mean_actual) ** 2 for actual in actuals)
    if ss_tot == 0:
        return None
    ss_res = sum((actual - pred) ** 2 for actual, pred in zip(actuals, preds))
    return (Decimal("1") - ss_res / ss_tot).quantize(Decimal("0.000001"))


def _ratio(numerator: int, denominator: int) -> Decimal | None:
    if denominator == 0:
        return None
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.000001"))


def _auc(actual: list[bool], scores: list[Decimal | None]) -> Decimal | None:
    usable = [(a, s) for a, s in zip(actual, scores) if s is not None]
    positives = [item for item in usable if item[0]]
    negatives = [item for item in usable if not item[0]]
    if not positives or not negatives:
        return None
    wins = Decimal("0")
    total = Decimal(len(positives) * len(negatives))
    for _, pos_score in positives:
        for _, neg_score in negatives:
            if pos_score > neg_score:
                wins += 1
            elif pos_score == neg_score:
                wins += Decimal("0.5")
    return (wins / total).quantize(Decimal("0.000001"))


def _pr_auc(actual: list[bool], scores: list[Decimal | None]) -> Decimal | None:
    usable = [(label, score) for label, score in zip(actual, scores) if score is not None]
    if not usable or not any(label for label, _ in usable):
        return None
    ordered = sorted(usable, key=lambda item: item[1], reverse=True)
    positives = sum(label for label, _ in ordered)
    true_positives = 0
    false_positives = 0
    previous_recall = Decimal("0")
    area = Decimal("0")
    for label, _ in ordered:
        if label:
            true_positives += 1
        else:
            false_positives += 1
        recall = Decimal(true_positives) / Decimal(positives)
        precision = Decimal(true_positives) / Decimal(true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area.quantize(Decimal("0.000001"))


def _brier_score(actual: list[bool], scores: list[Decimal | None]) -> Decimal | None:
    usable = [
        (Decimal("1") if label else Decimal("0"), score)
        for label, score in zip(actual, scores)
        if score is not None
    ]
    if not usable:
        return None
    if any(score < 0 or score > 1 for _, score in usable):
        return None
    return (
        sum((score - label) ** 2 for label, score in usable) / Decimal(len(usable))
    ).quantize(Decimal("0.000001"))


def _text(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None
