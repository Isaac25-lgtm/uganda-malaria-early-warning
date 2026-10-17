"""Paired district-cluster bootstrap for challenger-vs-RF differences.

Resamples DISTRICTS with replacement (preserving within-district dependence),
keeping all paired folds/rows for each sampled district, and forms percentile
confidence intervals for the pooled paired WAPE and MAE differences
(challenger - RF). Deterministic (fixed seed). Not a naive independent-obs test;
conclusions are not centred on p-values, and an interval excluding zero is not
treated as proof of universal superiority.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

import numpy as np

from app.services.model_validation import PredictionRecord

from research_comparisons.literature_based_forecasting_benchmark import config as C

Key = tuple[str, int, str, str]


def _wape(records: list[PredictionRecord]) -> float | None:
    total = sum((r.actual for r in records), Decimal("0"))
    if total == 0:
        return None
    error = sum((abs(r.predicted - r.actual) for r in records), Decimal("0"))
    return float(error / total * Decimal("100"))


def _mae(records: list[PredictionRecord]) -> float | None:
    if not records:
        return None
    return float(sum((abs(r.predicted - r.actual) for r in records), Decimal("0")) / Decimal(len(records)))


def paired_cluster_bootstrap(
    rf_by_key: dict[Key, PredictionRecord],
    challenger_by_key: dict[Key, PredictionRecord],
    common_keys: list[Key],
    *,
    seed: int = C.BOOTSTRAP_SEED,
    replicates: int = C.BOOTSTRAP_REPLICATES,
    ci: float = C.BOOTSTRAP_CI,
) -> dict[str, Any] | None:
    if not common_keys:
        return None
    by_district: dict[str, list[Key]] = defaultdict(list)
    for key in common_keys:
        by_district[key[0]].append(key)
    districts = sorted(by_district)
    n = len(districts)

    point_wape = None
    point_mae = None
    rf_all = [rf_by_key[k] for k in common_keys]
    ch_all = [challenger_by_key[k] for k in common_keys]
    if _wape(rf_all) is not None and _wape(ch_all) is not None:
        point_wape = _wape(ch_all) - _wape(rf_all)
    if _mae(rf_all) is not None and _mae(ch_all) is not None:
        point_mae = _mae(ch_all) - _mae(rf_all)

    rng = np.random.default_rng(seed)
    wape_diffs: list[float] = []
    mae_diffs: list[float] = []
    for _ in range(replicates):
        sampled = rng.integers(0, n, size=n)
        keys: list[Key] = []
        for idx in sampled:
            keys.extend(by_district[districts[idx]])
        rf = [rf_by_key[k] for k in keys]
        ch = [challenger_by_key[k] for k in keys]
        rf_w, ch_w = _wape(rf), _wape(ch)
        rf_m, ch_m = _mae(rf), _mae(ch)
        if rf_w is not None and ch_w is not None:
            wape_diffs.append(ch_w - rf_w)
        if rf_m is not None and ch_m is not None:
            mae_diffs.append(ch_m - rf_m)

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        return [float(np.percentile(values, lo_pct)), float(np.percentile(values, hi_pct))]

    return {
        "method": "paired district-cluster percentile bootstrap",
        "replicates": replicates,
        "seed": seed,
        "ci": ci,
        "n_districts": n,
        "point_wape_diff_challenger_minus_rf": point_wape,
        "point_mae_diff_challenger_minus_rf": point_mae,
        "wape_diff_ci": interval(wape_diffs),
        "mae_diff_ci": interval(mae_diffs),
        "wape_replicates_used": len(wape_diffs),
        "interpretation_note": "An interval excluding zero indicates a consistent paired "
        "difference under this benchmark; it does NOT prove universal superiority.",
    }


_WARN_METRICS = ("f2", "sensitivity", "precision", "f1", "specificity")


def _confusion(records: list[PredictionRecord]):
    tp = fp = fn = tn = 0
    for r in records:
        if r.actual_surge and r.predicted_surge:
            tp += 1
        elif r.actual_surge and not r.predicted_surge:
            fn += 1
        elif (not r.actual_surge) and r.predicted_surge:
            fp += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _warning_metrics(records: list[PredictionRecord]) -> dict[str, float | None]:
    tp, fp, fn, tn = _confusion(records)
    f2_d = 5 * tp + 4 * fn + fp
    f1_d = 2 * tp + fp + fn
    return {
        "f2": (5 * tp / f2_d) if f2_d else None,
        "f1": (2 * tp / f1_d) if f1_d else None,
        "sensitivity": (tp / (tp + fn)) if (tp + fn) else None,
        "precision": (tp / (tp + fp)) if (tp + fp) else None,
        "specificity": (tn / (tn + fp)) if (tn + fp) else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def paired_warning_bootstrap(
    rf_by_key: dict[Key, PredictionRecord],
    challenger_by_key: dict[Key, PredictionRecord],
    common_keys: list[Key],
    *,
    seed: int = C.BOOTSTRAP_SEED,
    replicates: int = C.BOOTSTRAP_REPLICATES,
    ci: float = C.BOOTSTRAP_CI,
) -> dict[str, Any] | None:
    """Paired district-cluster bootstrap of warning-metric differences (challenger - RF).

    F2 is the central operational metric; reported with sensitivity, precision,
    F1 and specificity. Same exact paired district/fold rows in every replicate.
    """
    if not common_keys:
        return None
    by_district: dict[str, list[Key]] = defaultdict(list)
    for key in common_keys:
        by_district[key[0]].append(key)
    districts = sorted(by_district)
    n = len(districts)

    def diff_for(keys: list[Key]) -> dict[str, float | None]:
        rf = _warning_metrics([rf_by_key[k] for k in keys])
        ch = _warning_metrics([challenger_by_key[k] for k in keys])
        return {m: (ch[m] - rf[m]) if (ch[m] is not None and rf[m] is not None) else None for m in _WARN_METRICS}

    rf_point = _warning_metrics([rf_by_key[k] for k in common_keys])
    ch_point = _warning_metrics([challenger_by_key[k] for k in common_keys])
    point_diff = diff_for(common_keys)

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {m: [] for m in _WARN_METRICS}
    for _ in range(replicates):
        sampled = rng.integers(0, n, size=n)
        keys: list[Key] = []
        for idx in sampled:
            keys.extend(by_district[districts[idx]])
        d = diff_for(keys)
        for m in _WARN_METRICS:
            if d[m] is not None:
                samples[m].append(d[m])

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        return [float(np.percentile(values, lo_pct)), float(np.percentile(values, hi_pct))]

    return {
        "method": "paired district-cluster percentile bootstrap (warning metrics)",
        "replicates": replicates, "seed": seed, "ci": ci, "n_districts": n,
        "rf_point": rf_point, "challenger_point": ch_point,
        "point_diff_challenger_minus_rf": point_diff,
        "ci_diff_challenger_minus_rf": {m: interval(samples[m]) for m in _WARN_METRICS},
        "interpretation_note": "F2 is the central operational metric (missing true surges is "
        "costlier than some false alerts). An F2 interval including zero is INCONCLUSIVE at that "
        "horizon; do not call it a statistical tie or a win. Interpret with sensitivity/precision/"
        "specificity together. Do not average F2 across horizons as a selection rule.",
    }
