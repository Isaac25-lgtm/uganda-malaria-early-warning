"""Standalone entrypoint for the isolated literature-based forecasting benchmark.

Run as a module (NOT a Flask command; registers nothing in the app):

    python -m research_comparisons.literature_based_forecasting_benchmark.src.run --run-id 20260828 [ACTION]

Actions:
    --baseline          write runs/<id>/baseline_integrity_manifest.json (read-only over canonical)
    --validate-gate     check the literature gate + revised method classification
    --dry-run           pre-flight guards (hashes/schemas/folds/dates/isolation/gate/jobs/pkgs/disk) — no fitting
    --final-integrity   re-verify protected artifacts vs the baseline (fail on drift)
    --rebuild-report    regenerate report-only outputs from an existing run; no fitting
    (full run)          fit challengers + paired comparison + bootstrap + report (checkpointed, resumable)
    --resume            with the full run: skip fold cells already checkpointed
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import types
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from research_comparisons.literature_based_forecasting_benchmark import config as C
from research_comparisons.literature_based_forecasting_benchmark.src import (
    bootstrap,
    data,
    davis_exploratory_nb_dl,
    folds,
    integrity,
    katwesige_nb,
    leakage,
    metrics,
    paired_comparison,
    reporting,
    tekle_poisson_dl,
    zinszer_arimax,
)

METHODS: dict[str, Any] = {
    "tekle_poisson_dl": tekle_poisson_dl,
    "zinszer_arimax": zinszer_arimax,
    "katwesige_nb": katwesige_nb,
    "davis_exploratory_nb_dl": davis_exploratory_nb_dl,
}
PRIMARY = ("tekle_poisson_dl", "zinszer_arimax")
NON_PRIMARY = ("katwesige_nb", "davis_exploratory_nb_dl")

EXPECTED_PRIMARY = {"zinszer_arimax", "tekle_poisson_dl"}
EXPECTED_NON_PRIMARY = {"katwesige_nb", "davis_exploratory_nb_dl"}
EXPECTED_EXCLUDED = {"ippolito_zambia", "epidemia_2017"}
LITERATURE_FILES = (
    "paper_verification.md",
    "paper_to_code_specification.json",
    "adaptation_register.csv",
    "excluded_methods.md",
)


# --------------------------------------------------------------------------- gate
def validate_gate() -> list[str]:
    problems: list[str] = []
    lit = C.RUN_DIR / "literature"
    for name in LITERATURE_FILES:
        if not (lit / name).exists():
            problems.append(f"missing literature artifact: {name}")
    spec_path = lit / "paper_to_code_specification.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        primary = set(spec.get("primary_methods", {}))
        non_primary = set(spec.get("non_primary_methods", {}))
        excluded = set(spec.get("excluded_methods", {}))
        if primary != EXPECTED_PRIMARY:
            problems.append(f"primary methods {sorted(primary)} != {sorted(EXPECTED_PRIMARY)}")
        if "katwesige_nb" in primary:
            problems.append("Katwesige must NOT be a primary method")
        if not EXPECTED_NON_PRIMARY.issubset(non_primary):
            problems.append(f"non-primary must include {sorted(EXPECTED_NON_PRIMARY)}")
        if not EXPECTED_EXCLUDED.issubset(excluded):
            problems.append(f"excluded must include {sorted(EXPECTED_EXCLUDED)}")
    if set(C.PRIMARY_METHODS) != EXPECTED_PRIMARY:
        problems.append(f"config PRIMARY_METHODS {sorted(C.PRIMARY_METHODS)} != {sorted(EXPECTED_PRIMARY)}")
    return problems


# --------------------------------------------------------------------------- helpers
def build_context(rows: list[dict[str, Any]]) -> types.SimpleNamespace:
    error_keys = data.load_error_keys()
    return types.SimpleNamespace(
        observed=data.build_observed_cases(rows, error_keys),
        climate_series=data.build_climate_series(rows),
        error_keys=error_keys,
    )


def _checkpoint_path(method_id: str, horizon: int, fold_number: int) -> Path:
    return C.RUN_DIR / "checkpoints" / method_id / f"h{horizon}_fold{fold_number}.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    leakage.assert_output_isolated(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def run_fold_cell(method_id: str, horizon: int, fold, fold_number: int, ctx) -> dict[str, Any]:
    """Fit one (method, horizon, fold); return a checkpoint-serialisable payload."""
    module = METHODS[method_id]
    started = time.perf_counter()
    out = module.predict_fold(fold, horizon, ctx)
    elapsed = time.perf_counter() - started
    training_max = leakage.training_max_target_date(fold.trainable_rows, horizon)
    threshold_max = leakage.threshold_history_max_date(fold.train_rows)
    predictions = []
    for item in out["predictions"]:
        row = item["row"]
        lin = item.get("lineage") or {}
        record = {
            "district_id": str(row["district_id"]),
            "district_name": row.get("district_name", ""),
            "origin_date": str(row["week_start_date"]),
            "target_date": str(row[f"target_week_start_date_h{horizon}"]),
            "epi_year": row.get("epi_year", ""),
            "epi_week": row.get("epi_week", ""),
            "observed_cases": row.get(f"target_cases_h{horizon}", ""),
            "predicted_cases": item.get("predicted"),
            "status": item["status"],
            "failure_reason": item.get("failure_reason"),
            "selected_order": item.get("selected_order"),
            "max_case_source_date": lin.get("max_case_source_date"),
            "max_climate_source_date": lin.get("max_climate_source_date"),
            "max_feature_source_date": lin.get("max_feature_source_date"),
            "lineage_violates_origin": lin.get("violates_origin"),
        }
        predictions.append(record)
    return {
        "method_id": method_id,
        "adaptation_class": getattr(module, "ADAPTATION_CLASS", method_id),
        "is_primary": getattr(module, "IS_PRIMARY", False),
        "horizon": horizon,
        "fold_number": fold_number,
        "origin_date": fold.origin_date.isoformat(),
        "target_date": fold.target_date.isoformat(),
        "status": out["status"],
        "failure_reason": out.get("failure_reason"),
        "diagnostics": out["diagnostics"],
        "elapsed_seconds": round(elapsed, 3),
        "training_max_target_date": training_max.isoformat() if training_max else None,
        "threshold_history_max_date": threshold_max.isoformat() if threshold_max else None,
        "predictions": predictions,
    }


# --------------------------------------------------------------------------- dry run
def dry_run() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    dataset_ok = integrity._sha256_file(C.CANONICAL_DATASET) == C.CANONICAL_DATASET_SHA256
    add("canonical_dataset_hash", dataset_ok, C.CANONICAL_DATASET_SHA256)

    rf_path = C.REPO_ROOT / C.FROZEN_PREDICTION_ARTIFACTS["random_forest"]
    with rf_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    required = {"district_id", "horizon", "forecast_origin_week", "target_week",
                "observed_cases", "predicted_cases", "district_surge_threshold",
                "observed_surge_status", "count_threshold_surge_classification"}
    add("frozen_rf_schema", required.issubset(set(header)), sorted(required - set(header)))

    rows = data.load_canonical_rows()
    fbh = folds.folds_by_horizon(rows)
    fold_ok = all(len(fbh[h]) == C.MAX_FOLDS for h in C.HORIZONS)
    add("folds_12_per_horizon", fold_ok, {h: len(fbh[h]) for h in C.HORIZONS})
    try:
        folds.assert_target_after_origin(rows)
        add("target_after_origin", True)
    except Exception as exc:  # noqa: BLE001
        add("target_after_origin", False, str(exc))

    # date lineage sample
    lineage_ok = True
    for h in C.HORIZONS:
        for fold in fbh[h]:
            if (fold.target_date - fold.origin_date).days != h * 7:
                lineage_ok = False
    add("date_lineage_gap_equals_horizon", lineage_ok)

    # output isolation
    try:
        leakage.assert_output_isolated(C.RUN_DIR / "checkpoints" / "probe.json")
        iso_ok = True
        iso_detail = ""
    except Exception as exc:  # noqa: BLE001
        iso_ok, iso_detail = False, str(exc)
    add("output_isolation_under_run_dir", iso_ok, iso_detail)
    try:
        leakage.assert_output_isolated(C.REPO_ROOT / "data" / "modelling" / "x.csv")
        add("output_isolation_rejects_canonical", False, "did not reject canonical path")
    except leakage.OutputIsolationError:
        add("output_isolation_rejects_canonical", True)

    add("literature_gate", not validate_gate(), validate_gate())

    n_cells = len(METHODS) * len(C.HORIZONS) * C.MAX_FOLDS
    arimax_fits = len(fbh[4][0].validation_rows) * len(C.HORIZONS) * C.MAX_FOLDS  # ~ districts * h * folds
    add("expected_job_counts", True, {"fold_cells": n_cells, "approx_arimax_district_fits": arimax_fits})

    pkgs = {}
    for mod in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib"):
        try:
            __import__(mod)
            pkgs[mod] = "ok"
        except Exception as exc:  # noqa: BLE001
            pkgs[mod] = f"MISSING: {exc}"
    add("packages_available", all(v == "ok" for v in pkgs.values()), pkgs)

    free_gb = shutil.disk_usage(C.RUN_DIR if C.RUN_DIR.exists() else C.REPO_ROOT).free / (1024 ** 3)
    add("disk_capacity_gb", free_gb > 1.0, round(free_gb, 1))

    # database-write prohibition: no active Flask app context => no DB session/writes.
    # (The SQLAlchemy db object may be imported transitively via canonical utilities,
    # but it is unusable without an app context, which this benchmark never opens.)
    try:
        from flask import has_app_context
        active_ctx = bool(has_app_context())
    except Exception:  # noqa: BLE001
        active_ctx = False
    add("no_active_app_context_no_db_writes", not active_ctx,
        "db unusable without an app context; benchmark never opens one or writes rows")

    payload = {
        "kind": "dry_run",
        "run_id": C.RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "all_ok": all(check["ok"] for check in checks),
        "checks": checks,
    }
    _atomic_write_json(C.RUN_DIR / "dry_run_report.json", payload)
    return payload


# --------------------------------------------------------------------------- full run
def full_run(resume: bool) -> dict[str, Any]:
    gate = validate_gate()
    if gate:
        raise RuntimeError("literature gate failed: " + "; ".join(gate))
    if not (C.RUN_DIR / "baseline_integrity_manifest.json").exists():
        integrity.write_baseline()

    started = time.perf_counter()
    rows = data.load_canonical_rows()
    ctx = build_context(rows)
    folds.write_fold_manifest(rows, C.RUN_DIR / "fold_manifest.csv")
    folds_by_h = folds.folds_by_horizon(rows)

    run_manifest = {
        "run_id": C.RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "methods": {mid: getattr(m, "ADAPTATION_CLASS", mid) for mid, m in METHODS.items()},
        "primary_methods": list(PRIMARY),
        "non_primary_methods": list(NON_PRIMARY),
        "horizons": list(C.HORIZONS),
        "max_folds": C.MAX_FOLDS,
        "bootstrap": {"replicates": C.BOOTSTRAP_REPLICATES, "seed": C.BOOTSTRAP_SEED, "ci": C.BOOTSTRAP_CI},
        "canonical_dataset_sha256": C.CANONICAL_DATASET_SHA256,
        "warning_workflow": "predicted count -> district Q75 threshold -> warning label",
        "notes": "WAPE is an error measure; never accuracy; never 100-WAPE.",
    }
    _atomic_write_json(C.RUN_DIR / "run_manifest.json", run_manifest)

    logs: list[dict[str, Any]] = []
    per_method_records: dict[str, dict[paired_comparison.Key, Any]] = {mid: {} for mid in METHODS}
    provenance_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []

    for method_id, module in METHODS.items():
        for horizon in C.HORIZONS:
            for fold_number, fold in enumerate(folds_by_h[horizon], start=1):
                ckpt = _checkpoint_path(method_id, horizon, fold_number)
                if resume and ckpt.exists():
                    payload = json.loads(ckpt.read_text(encoding="utf-8"))
                else:
                    payload = run_fold_cell(method_id, horizon, fold, fold_number, ctx)
                    _atomic_write_json(ckpt, payload)
                logs.append({k: payload[k] for k in
                             ("method_id", "horizon", "fold_number", "status", "failure_reason",
                              "elapsed_seconds", "diagnostics")})
                diagnostics_rows.append({"method_id": method_id, "horizon": horizon,
                                         "fold_number": fold_number, **_flat_diag(payload)})
                # rebuild records + provenance (threshold uses the fold's real train_rows)
                for pred in payload["predictions"]:
                    threshold_str = None
                    if pred["status"] == "success" and pred["predicted_cases"] is not None:
                        rec = metrics.build_record(
                            _row_stub(pred, horizon), horizon, float(pred["predicted_cases"]), fold.train_rows
                        )
                        per_method_records[method_id][paired_comparison.key_of(rec)] = rec
                        threshold_str = str(rec.threshold)
                    provenance_rows.append(_provenance(payload, pred, method_id, module, threshold_str))
                print(f"  {method_id} h{horizon} fold {fold_number}: {payload['status']} "
                      f"({payload['elapsed_seconds']}s)  [{time.perf_counter()-started:.0f}s]", flush=True)

    # leakage audit
    leak_fail = [p for p in provenance_rows if p["leakage_check_passed"] is False]
    _write_csv(C.RUN_DIR / "predictions_provenance.csv", provenance_rows, leakage.PROVENANCE_COLUMNS)
    _atomic_write_json(C.RUN_DIR / "job_convergence_log.json", logs)
    _write_csv(C.RUN_DIR / "fit_diagnostics.csv", diagnostics_rows,
               tuple(diagnostics_rows[0].keys()) if diagnostics_rows else ("method_id",))

    # metrics: max-coverage + paired vs RF + bootstrap
    rf_records = paired_comparison.load_frozen_records("random_forest")
    coverage_metrics: dict[str, Any] = {}
    paired_metrics: dict[str, Any] = {}
    bootstrap_metrics: dict[str, Any] = {}
    warning_bootstrap_metrics: dict[str, Any] = {}
    for method_id in METHODS:
        coverage_metrics[method_id] = {}
        paired_metrics[method_id] = {}
        bootstrap_metrics[method_id] = {}
        warning_bootstrap_metrics[method_id] = {}
        for horizon in C.HORIZONS:
            cmp = paired_comparison.compare(rf_records, per_method_records[method_id], horizon)
            coverage_metrics[method_id][str(horizon)] = cmp["max_coverage_challenger"]
            paired = {k: v for k, v in cmp.items() if k != "common_keys"}
            paired_metrics[method_id][str(horizon)] = paired
            bootstrap_metrics[method_id][str(horizon)] = bootstrap.paired_cluster_bootstrap(
                rf_records, per_method_records[method_id], cmp["common_keys"]
            )
            warning_bootstrap_metrics[method_id][str(horizon)] = bootstrap.paired_warning_bootstrap(
                rf_records, per_method_records[method_id], cmp["common_keys"]
            )

    _atomic_write_json(C.RUN_DIR / "coverage_metrics.json", coverage_metrics)
    _atomic_write_json(C.RUN_DIR / "paired_rf_comparison.json", paired_metrics)
    _atomic_write_json(C.RUN_DIR / "bootstrap_results.json", bootstrap_metrics)
    _atomic_write_json(C.RUN_DIR / "paired_warning_bootstrap_results.json", warning_bootstrap_metrics)
    _atomic_write_json(C.RUN_DIR / "leakage_audit.json", {
        "provenance_rows": len(provenance_rows),
        "leakage_failures": len(leak_fail),
        "leakage_failure_examples": leak_fail[:20],
        "all_passed": not leak_fail,
    })

    runtime = round(time.perf_counter() - started, 1)
    _atomic_write_json(C.RUN_DIR / "runtime_report.json", {
        "runtime_seconds": runtime,
        "fold_cells": sum(1 for _ in logs),
        "leakage_failures": len(leak_fail),
    })

    if leak_fail:
        raise RuntimeError(f"LEAKAGE AUDIT FAILED: {len(leak_fail)} predictions failed leakage checks — stopping.")

    reporting.build_report(
        run_dir=C.RUN_DIR, primary=PRIMARY, non_primary=NON_PRIMARY,
        coverage=coverage_metrics, paired=paired_metrics, bootstrap=bootstrap_metrics,
        warning_bootstrap=warning_bootstrap_metrics, methods=METHODS, runtime_seconds=runtime,
    )

    final_path, drift = integrity.verify_final()
    return {"runtime_seconds": runtime, "leakage_failures": len(leak_fail),
            "integrity_ok": not drift, "integrity_drift": drift, "final_manifest": str(final_path)}


def rebuild_report() -> Path:
    """Regenerate report-only outputs for an existing run without model fitting."""
    required = {
        "coverage": "coverage_metrics.json",
        "paired": "paired_rf_comparison.json",
        "bootstrap": "bootstrap_results.json",
        "warning_bootstrap": "paired_warning_bootstrap_results.json",
        "runtime": "runtime_report.json",
    }
    payloads: dict[str, Any] = {}
    for key, filename in required.items():
        path = C.RUN_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Cannot rebuild report; missing existing run artifact: {path}")
        payloads[key] = json.loads(path.read_text(encoding="utf-8"))
    provenance_path = C.RUN_DIR / "predictions_provenance.csv"
    if not provenance_path.exists():
        raise FileNotFoundError(f"Cannot rebuild report; missing existing run artifact: {provenance_path}")
    with provenance_path.open("r", encoding="utf-8", newline="") as handle:
        provenance_audit = audit_existing_provenance(list(csv.DictReader(handle)))
    _atomic_write_json(C.RUN_DIR / "report_rebuild_audit.json", provenance_audit)
    if not provenance_audit["all_passed"]:
        raise RuntimeError(
            "Cannot rebuild report: stored provenance violates the strict source-date rule; "
            f"{provenance_audit['violation_count']} violation(s) found."
        )
    reporting.build_report(
        run_dir=C.RUN_DIR,
        primary=PRIMARY,
        non_primary=NON_PRIMARY,
        coverage=payloads["coverage"],
        paired=payloads["paired"],
        bootstrap=payloads["bootstrap"],
        warning_bootstrap=payloads["warning_bootstrap"],
        methods=METHODS,
        runtime_seconds=payloads["runtime"].get("runtime_seconds"),
    )
    return C.RUN_DIR / "BENCHMARK_REPORT.md"


def audit_existing_provenance(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Validate stored raw-source dates for a report-only rebuild.

    Training targets and threshold-history dates are intentionally excluded: they
    are not prediction features and may legitimately equal the forecast origin.
    """
    source_columns = (
        "max_case_source_date",
        "max_climate_source_date",
        "max_feature_source_date",
    )
    violations: list[dict[str, str]] = []
    for row in rows:
        origin = row.get("origin_date")
        if not origin:
            violations.append({"reason": "missing_origin_date", "method_id": row.get("method_id", "")})
            continue
        for column in source_columns:
            source = row.get(column)
            if source and source >= origin:
                violations.append({
                    "reason": "source_not_strictly_before_origin",
                    "column": column,
                    "source_date": source,
                    "origin_date": origin,
                    "method_id": row.get("method_id", ""),
                    "district_id": row.get("district_id", ""),
                })
        if row.get("leakage_check_passed", "").strip().lower() != "true":
            violations.append({
                "reason": "stored_leakage_check_not_passed",
                "method_id": row.get("method_id", ""),
                "district_id": row.get("district_id", ""),
            })
    return {
        "run_id": C.RUN_ID,
        "provenance_rows": len(rows),
        "strict_source_columns": list(source_columns),
        "violation_count": len(violations),
        "violation_examples": violations[:20],
        "all_passed": not violations,
    }


def _flat_diag(payload: dict[str, Any]) -> dict[str, Any]:
    diag = payload.get("diagnostics") or {}
    flat = {"status": payload["status"], "elapsed_seconds": payload["elapsed_seconds"]}
    for key in ("converged", "pearson_dispersion", "deviance_dispersion", "nb_alpha", "aic",
                "n_train", "n_features", "successful_fits", "failed_fits", "districts"):
        if key in diag:
            flat[key] = diag[key]
    return flat


def _row_stub(pred: dict[str, Any], horizon: int) -> dict[str, Any]:
    return {
        "district_id": pred["district_id"],
        "week_start_date": pred["origin_date"],
        f"target_week_start_date_h{horizon}": pred["target_date"],
        f"target_cases_h{horizon}": pred["observed_cases"],
        f"target_epi_week_h{horizon}": _target_epi_week(pred),
    }


def _target_epi_week(pred: dict[str, Any]) -> int:
    return date.fromisoformat(pred["target_date"]).isocalendar().week


def _provenance(payload, pred, method_id, module, threshold) -> dict[str, Any]:
    origin = payload["origin_date"]
    target = payload["target_date"]
    feature_cutoff = pred.get("max_feature_source_date")  # COMPUTED from consumed sources
    passed, _reason = leakage.check_prediction_leakage(
        origin_date=origin, target_date=target, feature_cutoff_date=feature_cutoff,
        training_max_target_date=payload["training_max_target_date"],
        threshold_history_max_date=payload["threshold_history_max_date"],
    )
    if pred.get("lineage_violates_origin"):
        passed = False
    return {
        "run_id": C.RUN_ID,
        "method_id": method_id,
        "adaptation_class": getattr(module, "ADAPTATION_CLASS", method_id),
        "district_id": pred["district_id"],
        "horizon": payload["horizon"],
        "fold_number": payload["fold_number"],
        "origin_date": origin,
        "target_date": target,
        "observed_cases": pred["observed_cases"],
        "predicted_cases": pred["predicted_cases"],
        "max_case_source_date": pred.get("max_case_source_date"),
        "max_climate_source_date": pred.get("max_climate_source_date"),
        "max_feature_source_date": feature_cutoff,
        "feature_cutoff_date": feature_cutoff,
        "training_max_target_date": payload["training_max_target_date"],
        "threshold": threshold,
        "threshold_history_max_date": payload["threshold_history_max_date"],
        "leakage_check_passed": passed,
        "prediction_status": pred["status"],
        "failure_reason": pred.get("failure_reason"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns) -> None:
    leakage.assert_output_isolated(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--validate-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--final-integrity", action="store_true")
    parser.add_argument("--rebuild-report", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.run_id != C.RUN_ID:
        parser.error(f"--run-id {args.run_id!r} != frozen config RUN_ID {C.RUN_ID!r}")

    if args.baseline:
        print("baseline written:", integrity.write_baseline())
        return 0
    if args.validate_gate:
        problems = validate_gate()
        print("LITERATURE GATE OK" if not problems else "LITERATURE GATE FAILED:\n  " + "\n  ".join(problems))
        return 0 if not problems else 1
    if args.final_integrity:
        path, drift = integrity.verify_final()
        print("final integrity:", path, "OK" if not drift else f"DRIFT {drift}")
        return 0 if not drift else 1
    if args.rebuild_report:
        print("report rebuilt:", rebuild_report())
        return 0
    if args.dry_run:
        payload = dry_run()
        for check in payload["checks"]:
            print(f"  [{'OK' if check['ok'] else 'FAIL'}] {check['check']}: {check['detail']}")
        print("DRY RUN:", "PASS" if payload["all_ok"] else "FAIL")
        return 0 if payload["all_ok"] else 1

    summary = full_run(resume=args.resume)
    print("RUN COMPLETE:", json.dumps(summary, default=str))
    return 0 if summary["integrity_ok"] and not summary["leakage_failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
