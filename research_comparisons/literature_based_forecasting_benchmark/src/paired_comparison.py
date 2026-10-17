"""Paired RF-versus-challenger comparison on exact common rows.

Loads the FROZEN Random Forest validation predictions read-only. The paired
comparison keeps only rows where RF and the challenger both produced a valid
prediction AND district_id, horizon, origin week, target week, and observed
value all match exactly. Maximum-coverage metrics (each method on all its valid
rows) are reported separately and are never called a fair head-to-head.
"""
from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal
from typing import Any

from app.services.model_validation import PredictionRecord, evaluate_predictions

from research_comparisons.literature_based_forecasting_benchmark import config as C

Key = tuple[str, int, str, str]  # (district_id, horizon, origin_week, target_week)


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def key_of(record: PredictionRecord) -> Key:
    return (record.district_id, record.horizon, record.origin_date.isoformat(), record.target_date.isoformat())


def load_frozen_records(model: str = "random_forest") -> dict[Key, PredictionRecord]:
    """Frozen validation predictions -> {key: PredictionRecord} (read-only)."""
    path = C.REPO_ROOT / C.FROZEN_PREDICTION_ARTIFACTS[model]
    records: dict[Key, PredictionRecord] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("model_status", "success")) != "success":
                continue
            if row.get("predicted_cases") in (None, ""):
                continue
            actual = Decimal(str(row["observed_cases"]))
            predicted = Decimal(str(row["predicted_cases"]))
            threshold = Decimal(str(row["district_surge_threshold"]))
            score = predicted - threshold
            record = PredictionRecord(
                district_id=str(row["district_id"]),
                horizon=int(row["horizon"]),
                origin_date=date.fromisoformat(str(row["forecast_origin_week"])),
                target_date=date.fromisoformat(str(row["target_week"])),
                actual=actual,
                predicted=predicted,
                threshold=threshold,
                actual_surge=_bool(row["observed_surge_status"]),
                predicted_surge=_bool(row["count_threshold_surge_classification"]),
                surge_score=score,
            )
            records[key_of(record)] = record
    return records


def _metrics(records: list[PredictionRecord]) -> dict[str, Any]:
    count, surge = evaluate_predictions(records)
    return {"count": count, "warning": surge}


def compare(
    rf_records: dict[Key, PredictionRecord],
    challenger_records: dict[Key, PredictionRecord],
    horizon: int,
) -> dict[str, Any]:
    """Max-coverage + exact paired RF-vs-challenger metrics for one horizon."""
    rf_h = {k: v for k, v in rf_records.items() if k[1] == horizon}
    ch_h = {k: v for k, v in challenger_records.items() if k[1] == horizon}

    common: list[Key] = []
    mismatched_observed = 0
    for key in set(rf_h) & set(ch_h):
        if rf_h[key].actual == ch_h[key].actual:
            common.append(key)
        else:
            mismatched_observed += 1

    rf_common = [rf_h[k] for k in common]
    ch_common = [ch_h[k] for k in common]
    rf_paired = _metrics(rf_common) if rf_common else None
    ch_paired = _metrics(ch_common) if ch_common else None

    def wape(block):
        return None if block is None else block["count"]["wape"]

    def mae(block):
        return None if block is None else block["count"]["mae"]

    wape_diff = rel_wape_diff = mae_diff = None
    if rf_paired and ch_paired and wape(rf_paired) is not None and wape(ch_paired) is not None:
        rf_w = float(wape(rf_paired))
        ch_w = float(wape(ch_paired))
        wape_diff = ch_w - rf_w  # challenger - RF (positive => challenger worse)
        rel_wape_diff = (ch_w - rf_w) / rf_w if rf_w else None
        mae_diff = float(mae(ch_paired)) - float(mae(rf_paired))

    return {
        "horizon": horizon,
        "rf_coverage": len(rf_h),
        "challenger_coverage": len(ch_h),
        "common_row_count": len(common),
        "common_districts": len({k[0] for k in common}),
        "common_origins": len({k[2] for k in common}),
        "mismatched_observed_dropped": mismatched_observed,
        "max_coverage_challenger": _metrics(list(ch_h.values())) if ch_h else None,
        "paired_rf": rf_paired,
        "paired_challenger": ch_paired,
        "paired_wape_diff_challenger_minus_rf": wape_diff,
        "paired_relative_wape_diff": rel_wape_diff,
        "paired_mae_diff_challenger_minus_rf": mae_diff,
        "common_keys": common,
    }
