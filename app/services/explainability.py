"""Read-only explainability evidence services.

Phase 29B. These services expose *already published* validation, selection,
threshold and feature-importance evidence. They never train, predict, write,
recalculate a winner, or load a production model. Every artifact is read through
a symbolic allowlist, validated, scrubbed of paths/secrets, and cached with an
mtime+size key. Callers receive copied, immutable-by-convention values.

Model-selection, thresholds and risk classification are NOT reimplemented here;
the dashboard services remain the single source for those. This module only
curates and explains.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------
# Symbolic allowlist. Callers pass keys, never paths — path traversal is
# structurally impossible because arbitrary paths cannot be requested.
# --------------------------------------------------------------------------
ALLOWLIST: dict[str, Path] = {
    "project_state": Path("data/quality/project_state_realignment_summary.json"),
    "modelling_dataset": Path("data/quality/modelling_dataset_summary.json"),
    "dashboard_data": Path("data/quality/dashboard_data_summary.json"),
    "forecast_generation": Path("data/quality/forecast_generation_summary.json"),
    "production_manifest": Path("data/models/production/production_model_manifest.json"),
    "selection": Path("data/quality/model_performance_selection_summary.json"),
    "baseline_validation": Path("data/quality/baseline_validation_summary.json"),
    "sarimax_validation": Path("data/quality/sarimax_validation_summary.json"),
    "sarimax_card": Path("data/models/sarimax_model_card.json"),
    "random_forest_validation": Path("data/quality/random_forest_validation_summary.json"),
    "random_forest_card": Path("data/models/random_forest_model_card.json"),
    "random_forest_importance": Path("data/models/random_forest_feature_importance.csv"),
    "xgboost_validation": Path("data/quality/xgboost_validation_summary.json"),
    "xgboost_card": Path("data/models/xgboost_model_card.json"),
    "xgboost_importance": Path("data/models/xgboost_feature_importance.csv"),
}

# Defensive ceiling so a swapped-in huge file cannot exhaust memory.
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024

# Native importance method per model, and the public contract counts.
NATIVE_METHOD = {"random_forest": "impurity", "xgboost": "gain"}
NATIVE_LABEL = {"random_forest": "mean impurity importance", "xgboost": "mean gain"}
PERMUTATION_METHOD = "permutation_validation_mae"
EXPECTED_NATIVE_FEATURES = 181
EXPECTED_PERMUTATION_FEATURES = 28
EXPECTED_ACTIVE_DISTRICTS = 146
FOLD_COUNT = 12
TREE_MODELS = ("random_forest", "xgboost")
HORIZONS = (4, 8, 12)

IMPORTANCE_CSV_COLUMNS = {
    "model_name",
    "horizon",
    "validation_fold",
    "importance_method",
    "feature_name",
    "importance",
}

EXPLAINABILITY_LIMITS = [
    "Feature importance is measured under national_pooled validation, not per district.",
    "district_specific=false: these values do not describe any single district's model.",
    "production_prediction_explanation=false: this does not explain the specific 2025W52 forecast.",
    "Importance is predictive association, not causation.",
    "Correlated features can redistribute importance between one another.",
    "Pooled importance may not describe every district equally.",
    "Permutation importance depends on the validation samples used.",
]


class ExplainabilityError(RuntimeError):
    """Raised when required evidence is missing or malformed (maps to 503)."""


class ExplainabilityParameterError(ValueError):
    """Raised on an invalid caller parameter (maps to 400)."""


# --------------------------------------------------------------------------
# Path / secret scrubbing
# --------------------------------------------------------------------------
_SECRET_KEY_RE = re.compile(
    r"(password|passwd|secret|token|credential|api[_-]?key|authorization|database_url|dsn|connection[_-]?string)",
    re.IGNORECASE,
)
_PATH_VALUE_RE = re.compile(r"([a-zA-Z]:\\|\\\\|/(?:home|users|mnt|var|etc|root|opt)/)", re.IGNORECASE)
_URL_RE = re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s\"']+", re.IGNORECASE)


def _looks_like_path(value: str) -> bool:
    if _PATH_VALUE_RE.search(value):
        return True
    # Windows drive path or a value ending in a known artifact suffix.
    return bool(re.search(r"\.(json|csv|joblib|pkl|log)\b", value, re.IGNORECASE)) and (
        "\\" in value or "/" in value
    )


def scrub(value: Any) -> Any:
    """Recursively drop path/secret/URL leakage from a value to be returned."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _SECRET_KEY_RE.search(key):
                continue
            # Keys that are known to carry filesystem paths are dropped wholesale.
            if isinstance(key, str) and key.lower() in {
                "artifact_paths",
                "proof_path",
                "path",
                "paths",
                "generated_paths",
                "output_path",
                "source_path",
                "destination_path",
                "dataset_signature",
                "checkpoints",
                "secrets",
            }:
                continue
            cleaned[key] = scrub(item)
        return cleaned
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, tuple):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        if _URL_RE.search(value) or _looks_like_path(value):
            return "[redacted]"
        return value
    return value


# --------------------------------------------------------------------------
# Bounded, mtime+size-invalidated cache of validated artifacts
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _CacheEntry:
    signature: tuple[float, int]
    value: Any


_CACHE: dict[str, _CacheEntry] = {}
_CACHE_LIMIT = 32


def _artifact_signature(path: Path) -> tuple[float, int]:
    stat = path.stat()
    return (stat.st_mtime, stat.st_size)


def _read_text(path: Path) -> str:
    size = path.stat().st_size
    if size > MAX_ARTIFACT_BYTES:
        raise ExplainabilityError(f"Evidence artifact is unexpectedly large ({size} bytes).")
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExplainabilityError("Evidence artifact is not valid UTF-8.") from exc


def _load(key: str, parser: Callable[[str], Any]) -> Any:
    if key not in ALLOWLIST:
        raise ExplainabilityError(f"Unknown evidence key: {key!r}.")
    path = ALLOWLIST[key]
    if not path.exists():
        raise ExplainabilityError(f"Required evidence is unavailable: {key}.")
    signature = _artifact_signature(path)
    cached = _CACHE.get(key)
    if cached is not None and cached.signature == signature:
        return copy.deepcopy(cached.value)  # copied so callers cannot mutate cache
    text = _read_text(path)
    value = parser(text)
    if len(_CACHE) >= _CACHE_LIMIT and key not in _CACHE:
        # Simple bound: drop an arbitrary existing entry.
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = _CacheEntry(signature=signature, value=value)
    return copy.deepcopy(value)


def _reject_constant(token: str) -> float:
    # json.loads calls this for the literal NaN / Infinity / -Infinity tokens.
    raise ExplainabilityError(f"Evidence artifact contains a non-finite JSON value ({token}).")


def _assert_finite(value: Any) -> None:
    """Recursively reject non-finite numbers, including overflow like 1e999.

    ``parse_constant`` only catches the literal NaN/Infinity tokens; a numeric
    literal such as 1e999 parses to float('inf') and must be caught here.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ExplainabilityError("Evidence artifact contains a non-finite number.")
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)


def _parse_json(text: str) -> Any:
    try:
        parsed = json.loads(text, parse_constant=_reject_constant)
    except ExplainabilityError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise ExplainabilityError("Evidence artifact is not valid JSON.") from exc
    _assert_finite(parsed)
    return parsed


# --------------------------------------------------------------------------
# JSON schema validation. Each validator raises ExplainabilityError (-> 503)
# on a wrong root type, a missing required block/field, or a wrong value type.
# --------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExplainabilityError(message)


def _finite_number(value: Any) -> float | int | None:
    """Convert a string/number metric to a finite JSON number, or None."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number == int(number):
        return int(number)
    return round(number, 6)


def _finite_metrics(metrics: Any) -> dict[str, Any]:
    """Convert every value in a metrics block to a finite JSON number."""
    if not isinstance(metrics, dict):
        return {}
    return {key: _finite_number(value) for key, value in metrics.items()}


def _require_dict(value: Any, name: str) -> dict:
    _require(isinstance(value, dict), f"{name} must be a JSON object.")
    return value


def _require_keys(payload: dict, keys: tuple[str, ...], name: str) -> None:
    missing = [k for k in keys if k not in payload]
    _require(not missing, f"{name} is missing required field(s): {', '.join(missing)}.")


APPROVED_MODEL_NAMES = ("Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost")


def _finite_present(metrics: Any, field: str) -> bool:
    return isinstance(metrics, dict) and _finite_number(metrics.get(field)) is not None


def _validate_selection(payload: Any) -> Any:
    root = _require_dict(payload, "Selection evidence")
    _require_keys(
        root,
        ("selected_count_models", "selected_surge_models", "all_models_compared", "selection_rationale"),
        "Selection evidence",
    )
    count_block = _require_dict(root["selected_count_models"], "selected_count_models")
    surge_block = _require_dict(root["selected_surge_models"], "selected_surge_models")
    _require(isinstance(root["all_models_compared"], list), "all_models_compared must be a list.")

    # Exactly horizons 4, 8, 12 in both selected-model blocks (string keys).
    for name, block in (("selected_count_models", count_block), ("selected_surge_models", surge_block)):
        keys = {int(k) for k in block if str(k).isdigit()}
        _require(keys == set(HORIZONS), f"{name} must cover exactly horizons 4, 8, 12; found {sorted(keys)}.")

    for horizon in HORIZONS:
        count = count_block.get(str(horizon)) or count_block.get(horizon)
        surge = surge_block.get(str(horizon)) or surge_block.get(horizon)
        count = _require_dict(count, f"selected_count_models[{horizon}]")
        surge = _require_dict(surge, f"selected_surge_models[{horizon}]")
        _require(count.get("model_name") in APPROVED_MODEL_NAMES,
                 f"count winner for horizon {horizon} is not an approved model.")
        _require(surge.get("model_name") in APPROVED_MODEL_NAMES,
                 f"surge winner for horizon {horizon} is not an approved model.")
        # Winning metrics must be finite: count needs WAPE or RMSE; surge needs F2 and sensitivity.
        cm = count.get("count_metrics")
        _require(_finite_present(cm, "wape") or _finite_present(cm, "rmse"),
                 f"count winner for horizon {horizon} has no finite WAPE or RMSE.")
        sm = surge.get("surge_metrics")
        _require(_finite_present(sm, "f2") and _finite_present(sm, "sensitivity"),
                 f"surge winner for horizon {horizon} has no finite F2 and sensitivity.")

        # Exactly four unique approved model comparisons for this horizon.
        compared = [m for m in root["all_models_compared"]
                    if isinstance(m, dict) and m.get("horizon") == horizon]
        names = [m.get("model_name") for m in compared]
        _require(len(compared) == 4, f"horizon {horizon} must have exactly 4 model comparisons; found {len(compared)}.")
        _require(set(names) == set(APPROVED_MODEL_NAMES),
                 f"horizon {horizon} comparisons must be the four approved models; found {sorted(str(n) for n in names)}.")
        _require(len(set(names)) == 4, f"horizon {horizon} has duplicate model comparisons.")
        for m in compared:
            _require(isinstance(m.get("model_scope"), str) and m["model_scope"],
                     f"comparison {m.get('model_name')} h{horizon} has no model_scope.")
            _require(isinstance(m.get("status"), str), f"comparison {m.get('model_name')} h{horizon} has no status.")
            _require(m.get("count_metrics") is None or isinstance(m["count_metrics"], dict),
                     f"comparison {m.get('model_name')} h{horizon} count_metrics must be an object.")
            _require(m.get("surge_metrics") is None or isinstance(m["surge_metrics"], dict),
                     f"comparison {m.get('model_name')} h{horizon} surge_metrics must be an object.")
    return root


def _validate_forecast_generation(payload: Any) -> Any:
    root = _require_dict(payload, "Forecast-generation evidence")
    _require_keys(
        root,
        ("issue_week", "issue_year", "horizons", "target_weeks",
         "expected_forecast_rows", "final_issue_row_count", "published", "status"),
        "Forecast-generation evidence",
    )
    _require(isinstance(root["issue_week"], int) and isinstance(root["issue_year"], int),
             "Forecast-generation issue_week/issue_year must be integers.")
    _require(isinstance(root["horizons"], list) and {int(h) for h in root["horizons"]} == set(HORIZONS),
             "Forecast-generation horizons must be exactly 4, 8, 12.")
    _require(isinstance(root["expected_forecast_rows"], int) and root["expected_forecast_rows"] > 0,
             "expected_forecast_rows must be a positive integer.")
    _require(isinstance(root["final_issue_row_count"], int) and root["final_issue_row_count"] > 0,
             "final_issue_row_count must be a positive integer.")
    _require(root["final_issue_row_count"] == root["expected_forecast_rows"],
             "final_issue_row_count is inconsistent with expected_forecast_rows.")
    _require(root.get("published") is True and root.get("status") == "success",
             "Forecast-generation evidence does not prove successful publication.")

    targets = _require_dict(root["target_weeks"], "target_weeks")
    for horizon in HORIZONS:
        entry = targets.get(str(horizon)) or targets.get(horizon)
        entry = _require_dict(entry, f"target_weeks[{horizon}]")
        _require(isinstance(entry.get("target_epi_year"), int) and isinstance(entry.get("target_epi_week"), int),
                 f"target_weeks[{horizon}] must have integer target_epi_year and target_epi_week.")
    return root


def _validate_basic_validation_summary(payload: Any) -> Any:
    """Lighter validator for validation summaries whose cutoffs are not numbered
    folds (Baseline is horizon-only; SARIMAX is district-specific)."""
    root = _require_dict(payload, "Validation-method evidence")
    _require_keys(root, ("validation_method", "horizons"), "Validation-method evidence")
    _require(root["validation_method"] == "rolling_origin",
             "validation_method must be 'rolling_origin'.")
    _require(isinstance(root["horizons"], list) and set(int(h) for h in root["horizons"]) <= set(HORIZONS),
             "Validation horizons must be a subset of 4, 8, 12.")
    return root


def _validate_tree_validation_summary(payload: Any) -> Any:
    """Strict validator for fold-bearing tree-model validation evidence.

    The 12-fold claim is only made from evidence that actually carries numbered
    folds 1-12 for every horizon plus the four leakage-safety flags.
    """
    root = _validate_basic_validation_summary(payload)
    _require_keys(root, ("validation_cutoffs",), "Validation-method evidence")
    _require({int(h) for h in root["horizons"]} == set(HORIZONS),
             "Tree validation horizons must be exactly 4, 8, 12.")
    for flag in ("no_random_shuffle", "horizon_aware_gap",
                 "per_origin_threshold_training_only", "trainable_targets_observed_by_origin"):
        _require(root.get(flag) is True, f"Validation evidence requires {flag} to be true.")

    cutoffs = root["validation_cutoffs"]
    _require(isinstance(cutoffs, list) and cutoffs, "validation_cutoffs must be a non-empty list.")
    # Exactly folds 1..12 per horizon; no 0/13, no duplicates, none missing.
    per_horizon: dict[int, list[int]] = {}
    for item in cutoffs:
        _require(isinstance(item, dict), "validation_cutoffs entries must be objects.")
        horizon = item.get("horizon")
        fold = item.get("validation_fold")
        _require(isinstance(horizon, int) and horizon in HORIZONS,
                 f"validation cutoff has an unsupported horizon {horizon!r}.")
        _require(isinstance(fold, int) and 1 <= fold <= FOLD_COUNT,
                 f"validation cutoff fold {fold!r} is outside 1-12.")
        per_horizon.setdefault(horizon, []).append(fold)
    for horizon in HORIZONS:
        folds = sorted(per_horizon.get(horizon, []))
        _require(folds == list(range(1, FOLD_COUNT + 1)),
                 f"horizon {horizon} must have exactly folds 1-12; found {folds}.")
    return root


def _validate_model_card(payload: Any) -> Any:
    return _require_dict(payload, "Model card")


_JSON_SCHEMAS: dict[str, Callable[[Any], Any]] = {
    "selection": _validate_selection,
    "forecast_generation": _validate_forecast_generation,
    # Fold-bearing tree validation gets the strict 12-fold schema.
    "random_forest_validation": _validate_tree_validation_summary,
    "xgboost_validation": _validate_tree_validation_summary,
    # Baseline (horizon-only) and SARIMAX (district-specific) use the basic schema.
    "baseline_validation": _validate_basic_validation_summary,
    "sarimax_validation": _validate_basic_validation_summary,
    "random_forest_card": _validate_model_card,
    "xgboost_card": _validate_model_card,
    "sarimax_card": _validate_model_card,
}


def load_json(key: str, *, validate: bool = True) -> Any:
    payload = _load(key, _parse_json)
    if validate and key in _JSON_SCHEMAS:
        _JSON_SCHEMAS[key](payload)
    return payload


def _parse_importance_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or set(reader.fieldnames) != IMPORTANCE_CSV_COLUMNS:
        raise ExplainabilityError("Feature-importance CSV has an unexpected column set.")
    rows: list[dict[str, Any]] = []
    for raw in reader:
        try:
            importance = float(raw["importance"])
        except (TypeError, ValueError) as exc:
            raise ExplainabilityError("Feature-importance CSV has a non-numeric value.") from exc
        if not math.isfinite(importance):
            raise ExplainabilityError("Feature-importance CSV contains a non-finite number.")
        rows.append(
            {
                "model_name": raw["model_name"],
                "horizon": raw["horizon"],
                "validation_fold": raw["validation_fold"],
                "importance_method": raw["importance_method"],
                "feature_name": raw["feature_name"],
                "importance": importance,
            }
        )
    return rows


MODEL_DISPLAY_NAME = {"random_forest": "Random Forest", "xgboost": "XGBoost"}
_NUMBERED_FOLDS = tuple(str(i) for i in range(1, FOLD_COUNT + 1))
_VALID_FOLDS = set(_NUMBERED_FOLDS) | {"overall"}
_EXPECTED_METHODS = {
    "random_forest": ("impurity", PERMUTATION_METHOD),
    "xgboost": ("gain", PERMUTATION_METHOD),
}
_EXPECTED_FEATURE_COUNT = {
    "impurity": EXPECTED_NATIVE_FEATURES,
    "gain": EXPECTED_NATIVE_FEATURES,
    PERMUTATION_METHOD: EXPECTED_PERMUTATION_FEATURES,
}


_NATIVE_METHODS = {"impurity", "gain"}


def _validate_importance_structure(rows: list[dict[str, Any]], model: str) -> None:
    """Enforce the full importance contract. Any violation is a 503, not a warning."""
    expected_name = MODEL_DISPLAY_NAME[model]
    approved_methods = set(_EXPECTED_METHODS[model])
    for r in rows:
        # No blank required fields.
        for field in ("model_name", "feature_name", "horizon", "validation_fold", "importance_method"):
            _require(str(r.get(field, "")).strip() != "", f"Importance CSV has a blank {field}.")
        _require(r["model_name"] == expected_name,
                 f"Importance CSV model_name {r['model_name']!r} does not match {expected_name!r}.")
        _require(r["horizon"] in ("4", "8", "12"),
                 f"Importance CSV has an unsupported horizon {r['horizon']!r}.")
        _require(r["validation_fold"] in _VALID_FOLDS,
                 f"Importance CSV has an invalid fold {r['validation_fold']!r} (only 1-12 and 'overall' allowed).")
        # Only the two methods approved for this model.
        _require(r["importance_method"] in approved_methods,
                 f"Importance CSV has an unsupported method {r['importance_method']!r} for {model}.")

    for method in _EXPECTED_METHODS[model]:
        expected_features = _EXPECTED_FEATURE_COUNT[method]
        is_native = method in _NATIVE_METHODS
        for horizon in ("4", "8", "12"):
            scoped = [r for r in rows if r["importance_method"] == method and r["horizon"] == horizon]
            _require(scoped, f"Importance CSV is missing {method} rows for horizon {horizon}.")

            # No duplicate (feature, fold) pairs; collect per-fold values and overall.
            seen: set[tuple[str, str]] = set()
            per_fold: dict[str, set[str]] = {}
            fold_values: dict[str, dict[str, float]] = {}
            overall_values: dict[str, float] = {}
            for r in scoped:
                key = (r["feature_name"], r["validation_fold"])
                _require(key not in seen,
                         f"Duplicate importance row for {r['feature_name']} fold {r['validation_fold']} "
                         f"({method} h{horizon}).")
                seen.add(key)
                # Native impurity/gain must be non-negative; permutation may be signed.
                if is_native:
                    _require(r["importance"] >= 0,
                             f"Native {method} importance for {r['feature_name']} is negative.")
                if r["validation_fold"] == "overall":
                    overall_values[r["feature_name"]] = r["importance"]
                else:
                    per_fold.setdefault(r["validation_fold"], set()).add(r["feature_name"])
                    fold_values.setdefault(r["feature_name"], {})[r["validation_fold"]] = r["importance"]

            # Exactly folds 1..12 present.
            _require(set(per_fold) == set(_NUMBERED_FOLDS),
                     f"{method} h{horizon} must contain exactly folds 1-12; found {sorted(per_fold)}.")

            # Every fold has the same, expected feature set.
            feature_universe = set().union(*per_fold.values()) if per_fold else set()
            _require(len(feature_universe) == expected_features,
                     f"{method} h{horizon} must have exactly {expected_features} features; "
                     f"found {len(feature_universe)}.")
            for fold, feats in per_fold.items():
                _require(feats == feature_universe,
                         f"{method} h{horizon} fold {fold} does not cover all features.")

            # Overall rows must exist for every feature AND equal the mean of folds 1-12.
            _require(set(overall_values) == feature_universe,
                     f"{method} h{horizon} 'overall' rows must cover every feature.")
            for feature in feature_universe:
                folds = fold_values[feature]
                mean = sum(folds.values()) / len(folds)
                _require(abs(mean - overall_values[feature]) <= 1e-6,
                         f"{method} h{horizon} 'overall' for {feature!r} does not equal the mean of folds 1-12.")


def load_importance_rows(model: str) -> list[dict[str, Any]]:
    if model not in MODEL_DISPLAY_NAME:
        raise ExplainabilityError(f"Unknown importance model: {model!r}.")
    rows = _load(f"{model}_importance", _parse_importance_csv)
    _validate_importance_structure(rows, model)
    return rows


def file_modified_at_utc(key: str) -> str | None:
    """Filesystem mtime, explicitly labelled as file time (never a data time)."""
    path = ALLOWLIST.get(key)
    if not path or not path.exists():
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _embedded_generated_at(key: str) -> str | None:
    try:
        payload = load_json(key)
    except ExplainabilityError:
        return None
    if isinstance(payload, dict):
        for field in ("generated_at_utc", "generated_at", "created_at_utc"):
            if payload.get(field):
                return payload[field]
    return None


# --------------------------------------------------------------------------
# Feature families
# --------------------------------------------------------------------------
FAMILY_MALARIA = "Malaria History"
FAMILY_CLIMATE = "Climate Signals"
FAMILY_TIME = "Time and Season"
FAMILY_DISTRICT = "District Identity"
FAMILY_MISSING = "Missing-Value Indicators"
FAMILY_OTHER = "Other"

FAMILY_ORDER = [
    FAMILY_MALARIA,
    FAMILY_CLIMATE,
    FAMILY_TIME,
    FAMILY_DISTRICT,
    FAMILY_MISSING,
    FAMILY_OTHER,
]

FAMILY_DESCRIPTION = {
    FAMILY_MALARIA: "Recent confirmed-case history: lags and rolling means of past malaria counts.",
    FAMILY_CLIMATE: "Lagged rainfall, temperature and humidity — associated predictors, not causes.",
    FAMILY_TIME: "Calendar position such as epidemiological week and a continuous week index.",
    FAMILY_DISTRICT: "Persistent local differences captured by district identity; not causal.",
    FAMILY_MISSING: "Indicators marking where a malaria value was missing (missingness is informative).",
    FAMILY_OTHER: "Valid features that did not match a known family.",
}


def feature_family(name: str) -> str:
    """Map a transformed or raw feature name to a readable family."""
    lower = name.lower()
    # Missingness must be checked before 'cases' because the indicator names
    # contain the case column they flag.
    if "missingindicator" in lower or "missing_" in lower or lower.endswith("_missing"):
        return FAMILY_MISSING
    if lower.startswith("district__") or lower == "district_id" or "district_id" in lower:
        return FAMILY_DISTRICT
    if "cases_" in lower or lower.endswith("_cases") or "case_lag" in lower:
        return FAMILY_MALARIA
    if any(token in lower for token in ("rainfall", "temperature", "humidity", "dewpoint")):
        return FAMILY_CLIMATE
    if any(token in lower for token in ("epi_week", "week_index", "continuous_week", "season", "month", "_year")):
        return FAMILY_TIME
    return FAMILY_OTHER


def _district_title(code: str) -> str:
    """UGA_YUMBE -> Yumbe; UGA_FORT_PORTAL_CITY -> Fort Portal City."""
    trimmed = code[4:] if code.upper().startswith("UGA_") else code
    return trimmed.replace("_", " ").title()


def _describe_measure(base: str) -> str:
    """Turn a raw measure token like 'cases_lag_4' into readable English."""
    lag = re.search(r"lag_(\d+)", base)
    roll = re.search(r"rolling_mean_(\d+)", base)
    if base.startswith("cases_rolling_mean") and roll:
        return f"{roll.group(1)}-week rolling mean of malaria cases"
    if base.startswith("cases_lag") and lag:
        return f"malaria cases {lag.group(1)} weeks earlier"
    for var, label in (("temperature", "Temperature"), ("rainfall", "Rainfall"), ("humidity", "Humidity")):
        if base.startswith(var) and lag:
            return f"{label} {lag.group(1)} weeks earlier"
    if base == "epi_week":
        return "Epidemiological week of year"
    if base in ("continuous_week_index", "week_index"):
        return "Continuous week index"
    if base == "district_id":
        return "District identity"
    return base.replace("_", " ")


def readable_feature_label(name: str) -> str:
    """Human-readable label for a transformed or raw feature name."""
    base = name.split("__", 1)[1] if "__" in name else name
    if base.startswith("district_id_"):
        return "District identity: " + _district_title(base[len("district_id_"):])
    if base == "district_id":
        return "District identity"
    if base.startswith("missingindicator_"):
        inner = base[len("missingindicator_"):]
        return "Missing-value indicator: " + _describe_measure(inner)
    return _describe_measure(base)


# --------------------------------------------------------------------------
# Importance aggregation
# --------------------------------------------------------------------------
def _numbered_folds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for numbered folds 1..12 only — 'overall' is never a 13th fold."""
    return [r for r in rows if r["validation_fold"].isdigit()]


def _overall_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["validation_fold"] == "overall"]


def _aggregate_importance(
    rows: list[dict[str, Any]], model: str, horizon: int, method: str
) -> tuple[dict[str, float], dict[str, int], bool]:
    """Return {feature: mean importance}, {feature: fold coverage}, overall_validated.

    Means come from numbered folds 1..12. The precomputed 'overall' rows are
    validated against that mean; a mismatch or incomplete fold coverage is a hard
    contract violation (ExplainabilityError -> 503), never a soft warning.
    """
    scoped = [
        r
        for r in rows
        if int(r["horizon"]) == horizon and r["importance_method"] == method
    ]
    _require(scoped, f"No {method} importance evidence for horizon {horizon}.")
    numbered = _numbered_folds(scoped)
    means: dict[str, float] = {}
    coverage: dict[str, int] = {}
    grouped: dict[str, list[float]] = {}
    for r in numbered:
        grouped.setdefault(r["feature_name"], []).append(r["importance"])
    for feature, values in grouped.items():
        # Every feature must have all 12 numbered folds; anything else is invalid.
        _require(len(values) == FOLD_COUNT,
                 f"Feature {feature!r} has {len(values)} folds, expected {FOLD_COUNT}.")
        means[feature] = sum(values) / len(values)
        coverage[feature] = len(values)

    # Cross-check EVERY feature's 'overall' row against the recomputed mean.
    overall = {r["feature_name"]: r["importance"] for r in _overall_rows(scoped)}
    _require(set(overall) == set(means),
             f"'overall' rows for {method} h{horizon} do not cover the same features as folds 1-12.")
    for feature, published in overall.items():
        recomputed = means[feature]
        _require(abs(recomputed - published) <= 1e-6,
                 f"'overall' importance for {feature!r} ({method} h{horizon}) does not match the mean of folds 1-12.")
    return means, coverage, True


NATIVE_METHOD_NOTES = {
    "random_forest": [
        "Random Forest impurity importance can favour features that are highly "
        "splittable or high-cardinality (for example one-hot district identity), "
        "so magnitudes should be read with care.",
        "Native tree importance is model-specific: it describes this model's splits, "
        "not a universal predictor ranking.",
        "Importance is predictive association, not causation.",
    ],
    "xgboost": [
        "XGBoost gain importance sums the loss improvement a feature provided at its "
        "splits; it can favour features used in many boosting rounds.",
        "Native tree importance is model-specific: it describes this model's splits, "
        "not a universal predictor ranking.",
        "Importance is predictive association, not causation.",
    ],
}
PERMUTATION_METHOD_NOTES = [
    "Permutation importance is the mean increase in validation MAE when a predictor "
    "is shuffled; larger positive values mean the model relied on that predictor.",
    "A negative permutation value means shuffling the predictor did NOT worsen "
    "validation MAE (the model performed at least as well without it here).",
    "Values depend on the validation samples and are predictive, not causal.",
]


def _family_coverage(means: dict[str, float], coverage: dict[str, int], family: str) -> int:
    """Minimum fold coverage across a family's features (verified to be 12)."""
    members = [coverage[f] for f in means if feature_family(f) == family]
    return min(members) if members else 0


def native_importance(model: str, horizon: int, limit: int, grouped: bool) -> dict[str, Any]:
    """Model-native importance (RF impurity / XGBoost gain), pooled, non-causal."""
    method = NATIVE_METHOD[model]
    rows = load_importance_rows(model)
    means, coverage, _ = _aggregate_importance(rows, model, horizon, method)

    feature_count = len(means)
    total = sum(means.values())
    # Normalize across ALL features before the limit is applied.
    normalized = {f: (v / total * 100.0 if total > 0 else 0.0) for f, v in means.items()}
    rounded_normalized = _rounded_percentages(normalized)

    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    features = [
        {
            "feature_name": name,
            "label": readable_feature_label(name),
            "family": feature_family(name),
            "raw_mean_importance": raw,
            "normalized_percentage": rounded_normalized[name],
            "rank": index + 1,
            "fold_coverage": coverage.get(name, 0),
            "is_zero_importance": raw == 0.0,
        }
        for index, (name, raw) in enumerate(ranked)
    ]

    group_summary = _group_native(means, normalized, coverage)

    warnings: list[str] = []
    other_features = [f for f in features if f["family"] == FAMILY_OTHER]
    if other_features:
        warnings.append(f"{len(other_features)} feature(s) did not match a known family and are grouped under Other.")

    data: dict[str, Any] = {
        "model": model,
        "model_name": MODEL_DISPLAY_NAME[model],
        "horizon": horizon,
        "method": "native",
        "native_method": method,
        "importance_definition": NATIVE_LABEL[model],
        "total_features": feature_count,
        "fold_count_used": FOLD_COUNT,  # verified: every feature had all 12 folds
        "overall_rows_validated": True,  # a mismatch would have raised
        "normalized_over_all_features_before_limit": True,
        "method_notes": NATIVE_METHOD_NOTES[model],
    }
    _attach_importance(data, features, group_summary, limit, grouped)
    return {"data": data, "warnings": warnings}


def _rounded_percentages(values: dict[str, float]) -> dict[str, float]:
    rounded = {name: round(value, 6) for name, value in values.items()}
    if rounded and sum(values.values()) > 0:
        anchor = max(values, key=values.get)
        rounded[anchor] = round(rounded[anchor] + (100.0 - sum(rounded.values())), 6)
    return rounded


def _group_native(
    means: dict[str, float], normalized: dict[str, float], coverage: dict[str, int]
) -> list[dict[str, Any]]:
    """Group native values BEFORE normalization, then express as a share."""
    total = sum(means.values())
    families: dict[str, dict[str, Any]] = {}
    for feature, raw in means.items():
        fam = feature_family(feature)
        entry = families.setdefault(
            fam, {"family": fam, "raw_mean_importance": 0.0, "feature_count": 0}
        )
        entry["raw_mean_importance"] += raw
        entry["feature_count"] += 1
    ordered = [families[f] for f in FAMILY_ORDER if f in families]
    ordered.sort(key=lambda e: e["raw_mean_importance"], reverse=True)
    for index, entry in enumerate(ordered):
        entry["normalized_percentage"] = round(
            entry["raw_mean_importance"] / total * 100.0 if total > 0 else 0.0, 6
        )
        entry["family_rank"] = index + 1
        entry["fold_coverage"] = _family_coverage(means, coverage, entry["family"])
        entry["description"] = FAMILY_DESCRIPTION[entry["family"]]
        entry["is_causal"] = False
    return ordered


def permutation_importance(model: str, horizon: int, limit: int, grouped: bool) -> dict[str, Any]:
    """Signed permutation-MAE importance across folds 1-12 for 28 predictors."""
    rows = load_importance_rows(model)
    means, coverage, _ = _aggregate_importance(rows, model, horizon, PERMUTATION_METHOD)

    feature_count = len(means)
    # Rank by SIGNED mean descending. Negatives are preserved, never clamped.
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    positive_total = sum(max(v, 0.0) for v in means.values())

    features = [
        {
            "feature_name": name,
            "label": readable_feature_label(name),
            "family": feature_family(name),
            "signed_mean_mae_change": value,
            "positive_share_percentage": round(
                max(value, 0.0) / positive_total * 100.0 if positive_total > 0 else 0.0, 6
            ),
            "rank": index + 1,
            "fold_coverage": coverage.get(name, 0),
            "sign": "positive" if value > 0 else ("negative" if value < 0 else "zero"),
        }
        for index, (name, value) in enumerate(ranked)
    ]

    group_summary = _group_permutation(means, coverage)

    data: dict[str, Any] = {
        "model": model,
        "model_name": MODEL_DISPLAY_NAME[model],
        "horizon": horizon,
        "method": "permutation",
        "native_method": PERMUTATION_METHOD,
        "importance_definition": "mean increase in validation MAE when the predictor is permuted (signed)",
        "value_field": "signed_mean_mae_change",
        "signs_preserved": True,
        "negatives_clamped": False,
        "positive_share_is_labelled": True,
        "total_features": feature_count,
        "fold_count_used": FOLD_COUNT,  # verified
        "overall_rows_validated": True,
        "method_notes": PERMUTATION_METHOD_NOTES,
    }
    _attach_importance(data, features, group_summary, limit, grouped)
    return {"data": data, "warnings": []}


def _group_permutation(means: dict[str, float], coverage: dict[str, int]) -> list[dict[str, Any]]:
    """Group permutation values as SIGNED raw means (no clamping)."""
    families: dict[str, dict[str, Any]] = {}
    for feature, value in means.items():
        fam = feature_family(feature)
        entry = families.setdefault(
            fam, {"family": fam, "signed_mean_mae_change": 0.0, "feature_count": 0}
        )
        entry["signed_mean_mae_change"] += value
        entry["feature_count"] += 1
    ordered = [families[f] for f in FAMILY_ORDER if f in families]
    ordered.sort(key=lambda e: e["signed_mean_mae_change"], reverse=True)
    for index, entry in enumerate(ordered):
        entry["family_rank"] = index + 1
        entry["fold_coverage"] = _family_coverage(means, coverage, entry["family"])
        entry["description"] = FAMILY_DESCRIPTION[entry["family"]]
        entry["is_causal"] = False
    return ordered


def _attach_importance(data, features, group_summary, limit, grouped):
    """Apply `limit` consistently: top-N families (grouped) or top-N features."""
    if grouped:
        # At most `limit` ranked families; individual features are omitted, and the
        # full group_summary is not duplicated.
        data["families"] = group_summary[: min(limit, len(group_summary))]
        data["returned_family_count"] = min(limit, len(group_summary))
    else:
        data["features"] = features[: min(limit, len(features))]
        data["group_summary"] = group_summary
        data["returned_feature_count"] = min(limit, len(features))


def strongest_family_narrative(group_summary: list[dict[str, Any]], value_key: str) -> str | None:
    """Narrative built ONLY from the returned evidence, never invented."""
    positives = [g for g in group_summary if g.get(value_key, 0) > 0]
    if not positives:
        return None
    top = max(positives, key=lambda g: g[value_key])
    return (
        f"In this pooled validation evidence, the strongest family is "
        f"'{top['family']}'. This is a predictive association under national pooled "
        f"validation, not a causal claim or a per-district explanation."
    )


# --------------------------------------------------------------------------
# Feature-importance endpoint assembly
# --------------------------------------------------------------------------
def build_feature_importance(
    *, model: str, horizon: int, method: str, limit: int, grouped: bool
) -> dict[str, Any]:
    if model not in TREE_MODELS:
        raise ExplainabilityParameterError(
            "model must be one of random_forest, xgboost. Baseline and SARIMAX are not tree models."
        )
    if horizon not in HORIZONS:
        raise ExplainabilityParameterError("horizon must be one of 4, 8, 12.")
    if method not in ("native", "permutation"):
        raise ExplainabilityParameterError("method must be one of native, permutation.")
    if not isinstance(limit, int) or limit < 1 or limit > 50:
        raise ExplainabilityParameterError("limit must be an integer between 1 and 50.")

    if method == "native":
        built = native_importance(model, horizon, limit, grouped)
        value_key = "raw_mean_importance"
    else:
        built = permutation_importance(model, horizon, limit, grouped)
        value_key = "signed_mean_mae_change"

    data = built["data"]
    group_summary = data.get("families") if grouped else data.get("group_summary")
    narrative = strongest_family_narrative(group_summary or [], value_key)

    return {
        "data": scrub(data),
        "metadata": {
            "source": f"{model}_feature_importance.csv",
            "evidence_generated_at": _embedded_generated_at(f"{model}_card"),
            "file_modified_at_utc": file_modified_at_utc(f"{model}_importance"),
            "strongest_family_narrative": narrative,
        },
        "warnings": built["warnings"],
        "limitations": list(EXPLAINABILITY_LIMITS),
    }


# --------------------------------------------------------------------------
# Selection, validation, overview, threshold, pipeline evidence
# --------------------------------------------------------------------------
def _model_comparison(selection: dict[str, Any], horizon: int) -> list[dict[str, Any]]:
    """All four model comparisons for a horizon from published evidence only.

    ``all_models_compared`` is a flat list of {model, horizon, metrics}; filter
    to the requested horizon and expose only the published metric blocks.
    """
    out: list[dict[str, Any]] = []
    all_models = selection.get("all_models_compared") or []
    if not isinstance(all_models, list):
        return out
    for item in all_models:
        if not isinstance(item, dict) or item.get("horizon") != horizon:
            continue
        out.append(
            {
                "model_name": item.get("model_name"),
                "model_scope": item.get("model_scope"),
                "status": item.get("status"),
                "count_metrics": _finite_metrics(item.get("count_metrics")),
                "surge_metrics": _finite_metrics(item.get("surge_metrics")),
            }
        )
    return out


def build_model_selection(horizon: int) -> dict[str, Any]:
    if horizon not in HORIZONS:
        raise ExplainabilityParameterError("horizon must be one of 4, 8, 12.")
    try:
        selection = load_json("selection")
    except ExplainabilityError:
        raise
    rationale = selection.get("selection_rationale") or {}

    def entry(kind: str) -> dict[str, Any]:
        block = selection.get(kind) or {}
        item = block.get(str(horizon)) or block.get(horizon) or {}
        return scrub(item)

    count = entry("selected_count_models")
    surge = entry("selected_surge_models")

    comparison = _model_comparison(selection, horizon)
    # Fail closed: winners and the four-model comparison must be present.
    _require(count.get("model_name"), f"Selection evidence has no count winner for horizon {horizon}.")
    _require(surge.get("model_name"), f"Selection evidence has no surge winner for horizon {horizon}.")
    _require(len(comparison) >= 1, f"Selection evidence has no model comparison for horizon {horizon}.")

    # Explicit primary metric: count uses WAPE when present, else RMSE.
    count_metrics = _finite_metrics(count.get("count_metrics"))
    if count_metrics.get("wape") is not None:
        count_metric_name, count_metric_value = "WAPE", count_metrics["wape"]
    else:
        count_metric_name, count_metric_value = "RMSE", count_metrics.get("rmse")
    surge_metrics = _finite_metrics(surge.get("surge_metrics"))

    data = {
        "horizon": horizon,
        "selected_count_model": count.get("model_name"),
        "selected_surge_model": surge.get("model_name"),
        "count_selection": {
            "model_name": count.get("model_name"),
            "metric_name": count_metric_name,
            "metric_value": count_metric_value,
            "metric_rule": count.get("metric_rule") or rationale.get("count"),
            "count_metrics": count_metrics,
            "model_scope": count.get("model_scope"),
        },
        "surge_selection": {
            "model_name": surge.get("model_name"),
            "metric_name": "F2",
            "metric_value": surge_metrics.get("f2"),
            "tie_breaker_name": "sensitivity",
            "tie_breaker_value": surge_metrics.get("sensitivity"),
            "metric_rule": surge.get("metric_rule") or rationale.get("surge"),
            "surge_metrics": surge_metrics,
            "model_scope": surge.get("model_scope"),
        },
        "all_models_compared": comparison,
        "selection_rules": {"count": rationale.get("count"), "surge": rationale.get("surge")},
        "validation_method": "chronological rolling-origin",
        "accuracy_policy": (
            "Accuracy appears in the artifact as secondary context only. It was NOT used to "
            "select any model, because surge weeks are rare and accuracy can be high while "
            "most surges are missed."
        ),
        "winner_recalculated": False,
    }
    return {
        "data": scrub(data),
        "metadata": {
            "source": "model_performance_selection_summary.json",
            "evidence_generated_at": _embedded_generated_at("selection"),
            "file_modified_at_utc": file_modified_at_utc("selection"),
        },
        "warnings": [],
        "limitations": [
            "Selection uses published rolling-origin validation evidence; the winner is not recalculated here.",
        ],
    }


def build_validation_method() -> dict[str, Any]:
    # Fail closed: the fold count and method are only claimed when a real
    # validation summary is present and schema-valid. No default fabrication.
    # Only fold-bearing tree validation summaries can substantiate the 12-fold
    # claim; the strict schema verifies folds 1-12 per horizon and the flags.
    summary = None
    source_key = None
    for key in ("random_forest_validation", "xgboost_validation"):
        try:
            summary = load_json(key)  # strict schema: folds 1-12 verified here
            source_key = key
            break
        except ExplainabilityError:
            continue
    if summary is None:
        raise ExplainabilityError(
            "Validation-method evidence is unavailable; the 12-fold claim cannot be verified."
        )
    method = summary.get("validation_method")
    horizons = summary.get("horizons", list(HORIZONS))
    data = {
        "validation_method": "chronological rolling-origin",
        "raw_method_key": method,
        "numbered_folds": FOLD_COUNT,
        "approved_horizons": horizons,
        "training_before_validation": True,
        "random_split": False,
        "preprocessing_fit_on_training_only": True,
        "future_target_alignment": True,
        "missing_outcomes_excluded": True,
        "thresholds_origin_safe": True,
        "sarimax_scope": "district_specific",
        "tree_model_scope": "national_pooled",
        "statements": [
            "Models were evaluated with chronological rolling-origin validation.",
            "Each origin used only information available before that origin.",
            "There is no random shuffling or random train/test split.",
            "Preprocessing was fit on training data only.",
            "Targets are aligned to the future forecast week.",
            "Weeks with missing outcomes are excluded, never zero-filled.",
            "Surge thresholds are computed origin-safe from training history only.",
            "SARIMAX is district-specific; Random Forest and XGBoost are national-pooled.",
        ],
    }
    return {
        "data": scrub(data),
        "metadata": {
            "source": f"{source_key}_summary.json",
            "file_modified_at_utc": file_modified_at_utc(source_key),
        },
        "warnings": [],
        "limitations": [
            "Rolling-origin validation estimates out-of-sample error; it is not a guarantee of future accuracy.",
        ],
    }


AVAILABLE_METHODS = [
    {"method": "native", "models": ["random_forest", "xgboost"], "available": True,
     "description": "Model-native global importance (impurity / gain)."},
    {"method": "permutation", "models": ["random_forest", "xgboost"], "available": True,
     "description": "Permutation validation-MAE importance for original predictors."},
]
UNAVAILABLE_METHODS = [
    {"method": "production_per_prediction_contributions", "available": False,
     "reason": "No stable production-aligned per-prediction contribution artifact is confirmed. "
               "Validation-row contributions exist but must not be presented as explanations of the "
               "2025W52 production forecast."},
    {"method": "district_specific_importance", "available": False,
     "reason": "Tree-model importance is national-pooled; per-district importance is not published."},
    {"method": "causal_effect", "available": False,
     "reason": "Only predictive association is available; no causal identification was performed."},
]


def build_overview(horizon: int) -> dict[str, Any]:
    if horizon not in HORIZONS:
        raise ExplainabilityParameterError("horizon must be one of 4, 8, 12.")
    from app.services.dashboard_data import selected_models

    # Fail closed: overview requires both selection and forecast evidence.
    selection = load_json("selection")
    forecast = load_json("forecast_generation")
    rationale = selection.get("selection_rationale") or {}
    models = selected_models(horizon, selection)
    operational_method = _operational_surge_method(
        models["surge"]["model_name"], horizon
    )
    target = (forecast.get("target_weeks") or {}).get(str(horizon)) or {}

    data = {
        "issue_week": forecast.get("issue_week"),
        "issue_year": forecast.get("issue_year"),
        "issue_label": (f"{forecast.get('issue_year')}W{int(forecast.get('issue_week')):02d}"
                        if forecast.get("issue_year") and forecast.get("issue_week") else None),
        "horizon": horizon,
        "target_epi_week": target.get("target_epi_week"),
        "target_epi_year": target.get("target_epi_year"),
        "approved_models": ["Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost"],
        "selected_count_model": models["count"]["model_name"],
        "selected_surge_model": models["surge"]["model_name"],
        "selection_rules": {"count": rationale.get("count"), "surge": rationale.get("surge")},
        "validation_method": "chronological rolling-origin",
        "validation_scope": {"sarimax": "district_specific", "tree_models": "national_pooled"},
        "threshold_summary": {
            "method": "district seasonal 75th percentile from training-only normal-reference history",
            "unit": "cases",
            "origin_safe": True,
            "probability": operational_method == "dedicated_classifier",
            "operational_method": operational_method,
        },
        "explainability_methods_available": AVAILABLE_METHODS,
        "explainability_methods_unavailable": UNAVAILABLE_METHODS,
    }
    return {
        "data": scrub(data),
        "metadata": {
            "evidence_generated_at": forecast.get("generated_at_utc"),
            "selection_generated_at": _embedded_generated_at("selection"),
            "file_modified_at_utc": file_modified_at_utc("forecast_generation"),
        },
        "warnings": [],
        "limitations": list(EXPLAINABILITY_LIMITS),
    }


def build_threshold_method(params) -> dict[str, Any]:
    """Threshold method + optional per-district reuse of published values."""
    from app.services.dashboard_data import build_risk_rows, selected_models

    selection = load_json("selection")
    surge_model = selected_models(params.horizon, selection)["surge"]["model_name"]
    operational_method = _operational_surge_method(surge_model, params.horizon)
    uses_probability = operational_method == "dedicated_classifier"

    # The bands below mirror app.services.risk_classification exactly. They are
    # described here, not reimplemented: the per-district row still comes
    # from classify_forecast_risk via build_risk_rows.
    data: dict[str, Any] = {
        "method": "district seasonal 75th percentile from valid training-only history",
        "unit": "cases",
        "history_cutoff": "target epi week +/-2 from history available at the origin; 2022 excluded only from the normal-reference threshold",
        "surge_score_definition": "predicted_surge_score = surge-model predicted cases - district threshold",
        "surge_score_unit": "cases",
        "ratio_definition": "ratio = surge-model predicted cases / district threshold",
        "risk_bands": _risk_band_descriptions(uses_probability),
        "count_threshold_benchmark": (
            "Predicted cases minus the seasonal district threshold is the operational "
            "surge method when it is better balanced than the classifier."
        ),
        "positive_interpretation": "A positive score means predicted cases exceed the district threshold.",
        "negative_interpretation": "A negative score means predicted cases are below the district threshold.",
        "risk_interpretation": (
            "Risk uses the validated operational method selected for this model and horizon. "
            "High/Very High supports review and does not confirm an outbreak."
        ),
        "bands_source": "app.services.risk_classification (single source; not reimplemented)",
        "operational_method": operational_method,
        "is_probability": uses_probability,
        "confirms_outbreak": False,
        "recalculated": False,
    }

    warnings: list[str] = []
    if params.district_id:
        rows = build_risk_rows(params)  # reuses published values; nothing recalculated
        row = next((r for r in rows if r["district_id"] == params.district_id), None)
        if row is None:
            warnings.append("No published risk row for the selected district at this issue/horizon.")
        else:
            data["district"] = {
                "district_id": row["district_id"],
                "district_name": row["district_name"],
                "issue_label": row["issue_label"],
                "target_label": row["target_label"],
                "count_model_name": row["count_model_name"],
                "surge_model_name": row["surge_model_name"],
                "predicted_cases": row["predicted_cases"],
                "surge_model_predicted_cases": row["surge_predicted_cases"],
                "threshold_used": row["threshold_used"],
                "predicted_surge_score": row["predicted_surge_score"],
                "risk_level": row["risk_level"],
                "risk_basis": row["risk_basis"],  # reused from classify_forecast_risk
                "surge_probability": row["surge_probability"],
                "values_reused_from_published_forecast": True,
            }
    return {
        "data": scrub(data),
        "metadata": {"file_modified_at_utc": file_modified_at_utc("selection")},
        "warnings": warnings,
        "limitations": [
            "The threshold is a district-specific review benchmark; classifier probability is reported separately.",
            "A high risk level does not confirm an outbreak; it warrants epidemiological review.",
        ],
    }


def _operational_surge_method(model_name: str, horizon: int) -> str:
    if model_name in {"Baseline/Naive", "SARIMAX"}:
        return "count_threshold"
    card_key = {
        "Random Forest": "random_forest_card",
        "XGBoost": "xgboost_card",
    }.get(model_name)
    if card_key is None:
        raise ExplainabilityError(f"Unsupported selected surge model: {model_name}.")
    card = load_json(card_key)
    methods = card.get("operational_surge_method_by_horizon") or {}
    method = methods.get(str(horizon), methods.get(horizon))
    if method not in {"dedicated_classifier", "count_threshold"}:
        raise ExplainabilityError(
            f"Operational surge-method evidence is unavailable for {model_name} h{horizon}."
        )
    return str(method)


def _risk_band_descriptions(uses_probability: bool) -> list[dict[str, str]]:
    if uses_probability:
        return [
            {"risk_level": "Very High", "rule": "probability >= 0.75 and surge flag is true", "meaning": "Strongest classifier review signal."},
            {"risk_level": "High", "rule": "probability >= 0.60 or surge flag is true", "meaning": "High-priority classifier review signal."},
            {"risk_level": "Moderate", "rule": "0.40 <= probability below high-risk criteria", "meaning": "Intermediate classifier review signal."},
            {"risk_level": "Low", "rule": "probability < 0.40 and no high-risk flag", "meaning": "Lower classifier review priority."},
        ]
    return [
        {"risk_level": "Very High", "rule": "predicted cases >= 1.5 x threshold and surge flag is true", "meaning": "Strongest count-threshold review signal."},
        {"risk_level": "High", "rule": "predicted cases >= threshold or surge flag is true", "meaning": "High-priority count-threshold review signal."},
        {"risk_level": "Moderate", "rule": "0.8 x threshold <= predicted cases < threshold", "meaning": "Elevated but below-threshold review signal."},
        {"risk_level": "Low", "rule": "predicted cases < 0.8 x threshold", "meaning": "Lower count-threshold review priority."},
    ]


def build_pipeline_status() -> dict[str, Any]:
    """Curated project-state view. Never returns the raw project-state object.

    Fails closed: required production facts (forecast evidence, verifiable active
    district count) must be present. The district count is verified against the
    database, never hardcoded when its evidence is absent.
    """
    from app.services.project_state import build_project_state_summary

    summary = build_project_state_summary(write=False)
    _require(isinstance(summary, dict) and summary.get("components"),
             "Project-state evidence is unavailable.")
    components = {}
    for name, comp in (summary.get("components") or {}).items():
        # Only curated, non-path fields — never artifact_paths/commands/next actions with paths.
        components[name] = {
            "status": comp.get("status"),
            "row_counts": scrub(comp.get("row_counts") or {}),
        }

    # Forecast evidence is a required production fact.
    forecast = load_json("forecast_generation")

    # Verify the active district count against the database instead of hardcoding.
    active_districts = None
    try:
        from app.models import DistrictReference

        active_districts = DistrictReference.query.filter_by(active=True).count()
    except Exception:  # pragma: no cover - DB access issues surface as unavailable
        active_districts = None
    _require(active_districts is not None,
             "Active district count could not be verified from the database.")

    dashboard_status = (summary.get("components") or {}).get("visible_dashboard", {}).get("status")
    models_status = (summary.get("components") or {}).get("models_and_forecasts", {}).get("status")

    # Evidence-derived overall state.
    statuses = [c.get("status") for c in components.values()]
    if all(s == "complete" for s in statuses):
        overall_state = "complete"
    elif any(s in (None, "missing", "failed") for s in statuses):
        overall_state = "incomplete"
    else:
        overall_state = "partial"

    data = {
        "overall_state": overall_state,
        "components": components,
        "issue_week": forecast.get("issue_week"),
        "issue_year": forecast.get("issue_year"),
        "issue_label": (f"{forecast.get('issue_year')}W{int(forecast.get('issue_week')):02d}"
                        if forecast.get("issue_year") and forecast.get("issue_week") else None),
        "production_forecast_rows": forecast.get("final_issue_row_count"),
        "expected_forecast_rows": forecast.get("expected_forecast_rows"),
        "active_districts": active_districts,
        "active_districts_verified": True,
        "models": ["Baseline/Naive", "SARIMAX", "Random Forest", "XGBoost"],
        "horizons": list(HORIZONS),
        "dashboard_status": dashboard_status,
        "models_status": models_status,
        "warnings": scrub(summary.get("warnings") or []),
    }
    return {
        "data": scrub(data),
        "metadata": {
            "project_state_write": False,
            "evidence_generated_at": forecast.get("generated_at_utc"),
            "file_modified_at_utc": file_modified_at_utc("forecast_generation"),
        },
        "warnings": [],
        "limitations": [
            "Pipeline status is a curated summary of published evidence, not the raw project-state object.",
        ],
    }
