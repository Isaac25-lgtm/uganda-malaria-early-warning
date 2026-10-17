"""Evidence tables, plots, and BENCHMARK_REPORT.md.

Only Zinszer and Teklehaimanot may appear in the primary winner table. Katwesige
(NB sensitivity) and Davis (exploratory) are labelled non-primary in every table,
plot, and conclusion. WAPE is an error measure; never accuracy; never 100-WAPE.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from research_comparisons.literature_based_forecasting_benchmark import config as C
from research_comparisons.literature_based_forecasting_benchmark.src import leakage

WINNER_COLUMNS = (
    "method_id", "classification", "horizon", "rf_coverage", "challenger_coverage",
    "common_rows", "common_districts", "rf_paired_wape", "challenger_paired_wape",
    "wape_diff_challenger_minus_rf", "relative_wape_diff", "mae_diff_challenger_minus_rf",
    "challenger_paired_f2", "challenger_paired_sensitivity", "challenger_paired_specificity",
    "rf_paired_f2", "bootstrap_point_wape_diff", "bootstrap_wape_ci_low", "bootstrap_wape_ci_high",
    "challenger_fit_failures",
)


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt3(value, sign: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if sign else f"{value:.3f}"


def _fmt_ci(ci_low, ci_high) -> str:
    return f"[{ci_low:+.3f}, {ci_high:+.3f}]" if ci_low is not None else "n/a"


def _rows(primary, non_primary, paired, bootstrap, methods) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    order = list(primary) + list(non_primary)
    for method_id in order:
        classification = "primary" if method_id in primary else (
            "non_primary_sensitivity" if method_id == "katwesige_nb" else "non_primary_exploratory")
        for horizon in C.HORIZONS:
            block = paired[method_id][str(horizon)]
            boot = bootstrap[method_id][str(horizon)] or {}
            pr = block.get("paired_rf") or {}
            pc = block.get("paired_challenger") or {}
            rf_count = (pr.get("count") or {})
            ch_count = (pc.get("count") or {})
            ch_warn = (pc.get("warning") or {})
            rf_warn = (pr.get("warning") or {})
            wape_ci = boot.get("wape_diff_ci") or [None, None]
            out.append({
                "method_id": method_id,
                "classification": classification,
                "horizon": horizon,
                "rf_coverage": block.get("rf_coverage"),
                "challenger_coverage": block.get("challenger_coverage"),
                "common_rows": block.get("common_row_count"),
                "common_districts": block.get("common_districts"),
                "rf_paired_wape": _num(rf_count.get("wape")),
                "challenger_paired_wape": _num(ch_count.get("wape")),
                "wape_diff_challenger_minus_rf": block.get("paired_wape_diff_challenger_minus_rf"),
                "relative_wape_diff": block.get("paired_relative_wape_diff"),
                "mae_diff_challenger_minus_rf": block.get("paired_mae_diff_challenger_minus_rf"),
                "challenger_paired_f2": _num(ch_warn.get("f2")),
                "challenger_paired_sensitivity": _num(ch_warn.get("sensitivity")),
                "challenger_paired_specificity": _num(ch_warn.get("specificity")),
                "rf_paired_f2": _num(rf_warn.get("f2")),
                "bootstrap_point_wape_diff": boot.get("point_wape_diff_challenger_minus_rf"),
                "bootstrap_wape_ci_low": wape_ci[0],
                "bootstrap_wape_ci_high": wape_ci[1],
                "challenger_fit_failures": block.get("challenger_coverage") is not None
                and (block.get("rf_coverage") or 0) - (block.get("challenger_coverage") or 0),
            })
    return out


def _write_csv(path: Path, rows, columns) -> None:
    leakage.assert_output_isolated(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _report_title(run_id: str) -> str:
    return f"# Literature-based forecasting benchmark - BENCHMARK REPORT (run {run_id})"


def _method_lineage_schema(method_id: str, module: Any) -> dict[str, Any]:
    """Describe one method's source-date contract without raw data values."""
    if method_id == "zinszer_arimax":
        return {
            "method_id": method_id,
            "classification": "primary",
            "adaptation_class": getattr(module, "ADAPTATION_CLASS", method_id),
            "case_feature_family": {
                "name": "observed_case_history",
                "allowed_source_weeks": "all available observed weeks through origin minus 1",
            },
            "climate_feature_families": {
                variable: {
                    "allowed_source_weeks": "for an h-week forecast, target week minus (h + 1); "
                    "across forecast steps this ranges from origin minus (h + 1) through origin minus 1",
                }
                for variable in getattr(module, "EXOG_VARS", ())
            },
            "feature_construction": "District-specific ARIMAX/dynamic regression; case endog ends at origin minus 1.",
        }

    climate_lags = {
        variable: list(lags)
        for variable, lags in getattr(module, "CLIMATE_LAGS", {}).items()
    }
    case_lags = list(getattr(module, "CASE_LAGS", ()))
    return {
        "method_id": method_id,
        "classification": getattr(
            module,
            "CLASSIFICATION",
            "primary" if getattr(module, "IS_PRIMARY", False) else "non_primary",
        ),
        "adaptation_class": getattr(module, "ADAPTATION_CLASS", method_id),
        "case_feature_family": {
            "name": "case_ma_123" if method_id == "tekle_poisson_dl" else "observed_case_lags",
            "allowed_lags_weeks": case_lags,
        },
        "climate_feature_families": {
            variable: {"allowed_lags_weeks": lags}
            for variable, lags in climate_lags.items()
        },
        "feature_construction": (
            "Fourth-degree Almon distributed-lag terms for rainfall and temperature; "
            "case_ma_123 is the mean of observed case lags 1, 2 and 3."
            if method_id == "tekle_poisson_dl"
            else "Direct lag features plus pre-origin trend and seasonal terms."
        ),
    }


def _feature_lineage_schema(methods: dict[str, Any]) -> dict[str, Any]:
    """Return the compact contract that explains provenance date columns."""
    return {
        "run_id": C.RUN_ID,
        "artifact_purpose": (
            "Defines the calculation-level source-date lineage for predictions in "
            "predictions_provenance.csv. It contains no patient or prediction values."
        ),
        "origin_availability_convention": {
            "origin_week_inputs_used_as_predictors": False,
            "case_inputs": (
                "Observed confirmed cases only. Absent values and the six confirmed reporting errors "
                "remain unavailable to feature construction; no two-sided interpolation is used."
            ),
            "climate_inputs": "Only the listed prior-lag district climate values are eligible.",
            "strict_source_rule": "Every raw case or climate source date consumed by a predictor must be before the origin date.",
            "missing_value_handling": "Any required unavailable predictor is handled by training-fold preprocessing, never by a following observation.",
        },
        "provenance_columns": {
            "max_case_source_date": "Latest observed case week actually consumed by the prediction; null when no observed case input was consumed.",
            "max_climate_source_date": "Latest district climate week actually consumed by the prediction; null when no climate input was consumed.",
            "max_feature_source_date": "Maximum of the consumed case and climate source dates; the value audited against the strict source rule.",
            "feature_cutoff_date": "Alias of max_feature_source_date retained for the generic leakage guard.",
            "training_max_target_date": "Latest training outcome date; it may equal the origin because it is not a prediction feature.",
            "threshold_history_max_date": "Latest threshold-reference history date; it may equal the origin because it is not a prediction feature.",
        },
        "methods": [_method_lineage_schema(method_id, module) for method_id, module in methods.items()],
    }


def write_feature_lineage_schema(run_dir: Path, methods: dict[str, Any]) -> Path:
    path = run_dir / "feature_lineage_schema.json"
    leakage.assert_output_isolated(path)
    path.write_text(json.dumps(_feature_lineage_schema(methods), indent=2) + "\n", encoding="utf-8")
    return path


def _plots(rows, primary, plots_dir: Path) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return f"plots skipped (matplotlib unavailable: {exc})"
    leakage.assert_output_isolated(plots_dir / "probe.png")
    plots_dir.mkdir(parents=True, exist_ok=True)
    horizons = list(C.HORIZONS)

    # Paired WAPE: RF vs each PRIMARY challenger
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    xs = range(len(horizons))
    rf_vals = []
    for h in horizons:
        row = next(r for r in rows if r["method_id"] == primary[0] and r["horizon"] == h)
        rf_vals.append(row["rf_paired_wape"])
    ax.bar([x - width for x in xs], rf_vals, width, label="Random Forest (frozen)")
    for i, method_id in enumerate(primary):
        vals = [next(r for r in rows if r["method_id"] == method_id and r["horizon"] == h)["challenger_paired_wape"] for h in horizons]
        ax.bar([x + (i * width) for x in xs], vals, width, label=method_id)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"h{h}" for h in horizons])
    ax.set_ylabel("Paired WAPE (error; lower is better)")
    ax.set_title("Primary paired WAPE: RF vs literature challengers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "primary_paired_wape.png", dpi=130)
    plt.close(fig)
    return f"plots written to {plots_dir}"


WARNING_COLUMNS = (
    "method_id", "classification", "horizon", "common_rows",
    "rf_f2", "challenger_f2", "f2_diff_point", "f2_diff_ci_low", "f2_diff_ci_high", "f2_verdict",
    "sensitivity_diff_point", "sensitivity_diff_ci_low", "sensitivity_diff_ci_high",
    "precision_diff_point", "precision_diff_ci_low", "precision_diff_ci_high",
)


def _f2_verdict(ci) -> str:
    """ci = 95% interval of (challenger - RF) F2 difference."""
    if not ci or ci[0] is None:
        return "n/a"
    lo, hi = ci
    if lo > 0 and hi > 0:
        return "challenger higher F2"
    if lo < 0 and hi < 0:
        return "RF higher F2"
    return "inconclusive"


def _warning_rows(primary, non_primary, warning_bootstrap) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for method_id in list(primary) + list(non_primary):
        classification = "primary" if method_id in primary else (
            "non_primary_sensitivity" if method_id == "katwesige_nb" else "non_primary_exploratory")
        for horizon in C.HORIZONS:
            wb = warning_bootstrap.get(method_id, {}).get(str(horizon)) or {}
            point = wb.get("point_diff_challenger_minus_rf") or {}
            ci = wb.get("ci_diff_challenger_minus_rf") or {}
            f2_ci = ci.get("f2") or [None, None]
            sens_ci = ci.get("sensitivity") or [None, None]
            prec_ci = ci.get("precision") or [None, None]
            out.append({
                "method_id": method_id, "classification": classification, "horizon": horizon,
                "common_rows": (wb.get("rf_point") or {}).get("tp", 0) + (wb.get("rf_point") or {}).get("fn", 0)
                + (wb.get("rf_point") or {}).get("fp", 0) + (wb.get("rf_point") or {}).get("tn", 0),
                "rf_f2": (wb.get("rf_point") or {}).get("f2"),
                "challenger_f2": (wb.get("challenger_point") or {}).get("f2"),
                "f2_diff_point": point.get("f2"),
                "f2_diff_ci_low": f2_ci[0], "f2_diff_ci_high": f2_ci[1], "f2_verdict": _f2_verdict(f2_ci),
                "sensitivity_diff_point": point.get("sensitivity"),
                "sensitivity_diff_ci_low": sens_ci[0], "sensitivity_diff_ci_high": sens_ci[1],
                "precision_diff_point": point.get("precision"),
                "precision_diff_ci_low": prec_ci[0], "precision_diff_ci_high": prec_ci[1],
            })
    return out


def build_report(*, run_dir: Path, primary, non_primary, coverage, paired, bootstrap, warning_bootstrap, methods, runtime_seconds) -> None:
    write_feature_lineage_schema(run_dir, methods)
    rows = _rows(primary, non_primary, paired, bootstrap, methods)
    _write_csv(run_dir / "primary_winner_table.csv", [r for r in rows if r["classification"] == "primary"], WINNER_COLUMNS)
    _write_csv(run_dir / "non_primary_table.csv", [r for r in rows if r["classification"] != "primary"], WINNER_COLUMNS)
    _write_csv(run_dir / "all_methods_table.csv", rows, WINNER_COLUMNS)
    warning_rows = _warning_rows(primary, non_primary, warning_bootstrap)
    _write_csv(run_dir / "paired_warning_bootstrap_table.csv", warning_rows, WARNING_COLUMNS)
    plots_note = _plots(rows, list(primary), run_dir / "plots")

    def line(row):
        rw = row["rf_paired_wape"]; cw = row["challenger_paired_wape"]
        ci_lo, ci_hi = row["bootstrap_wape_ci_low"], row["bootstrap_wape_ci_high"]
        ci = f"[{ci_lo:.2f}, {ci_hi:.2f}]" if ci_lo is not None else "n/a"
        return (f"| {row['method_id']} | h{row['horizon']} | {row['common_rows']} | "
                f"{rw:.2f} | {cw:.2f} | {row['wape_diff_challenger_minus_rf']:+.2f} | {ci} | "
                f"{row['challenger_paired_f2']} / {row['rf_paired_f2']} |")

    primary_rows = [r for r in rows if r["classification"] == "primary"]
    nonprimary_rows = [r for r in rows if r["classification"] != "primary"]

    report = []
    report.append(_report_title(C.RUN_ID) + "\n")
    report.append("> Isolated, supervisor-requested comparison. NOT part of the frozen main pipeline, "
                  "dashboard, production model selection, or dissertation results unless separately approved. "
                  "**WAPE is an error measure — never accuracy, never 100−WAPE.**\n")
    report.append("## Included / excluded methods\n"
                  "- **PRIMARY (winner table):** Teklehaimanot-inspired Poisson distributed-lag; "
                  "Zinszer-inspired adapted district-level ARIMAX (dynamic regression).\n"
                  "- **NON-PRIMARY (out of winner table):** Katwesige-inspired NB **sensitivity analysis** "
                  "(authoritative full thesis inaccessible); Davis-inspired reduced NB distributed-lag **exploratory**.\n"
                  "- **EXCLUDED:** Ippolito (seasonal burden/totals, single low-transmission catchment — materially "
                  "different method); **EPIDEMIA 2017 was NOT reproduced as one model** (conceptual integration system).\n")
    report.append("## Primary paired RF-vs-challenger comparison\n")
    report.append("Only exact common rows (RF and challenger both valid; district, horizon, origin, "
                  "target, observed all match). Max-coverage metrics are in `coverage_metrics.json` "
                  "and are not a fair head-to-head.\n")
    report.append("| method | h | common rows | RF paired WAPE | challenger paired WAPE | ΔWAPE (ch−RF) | bootstrap 95% CI ΔWAPE | F2 ch/RF |")
    report.append("|---|---|---|---|---|---|---|---|")
    for row in primary_rows:
        report.append(line(row))
    report.append("\n## Non-primary sensitivity / exploratory (NOT in the winner table)\n")
    report.append("| method | h | common rows | RF paired WAPE | challenger paired WAPE | ΔWAPE (ch−RF) | bootstrap 95% CI ΔWAPE | F2 ch/RF |")
    report.append("|---|---|---|---|---|---|---|---|")
    for row in nonprimary_rows:
        report.append(line(row))

    report.append("\n## Surge-detection (F2) — paired district-cluster bootstrap (2000 reps)\n")
    report.append("F2 is the central operational warning metric (missing true surges is costlier than "
                  "some false alerts). Difference = challenger − RF; CI = 95% percentile on the same paired "
                  "district/fold rows. **An F2 interval including zero is INCONCLUSIVE at that horizon — not "
                  "a statistical tie, not a win.** WAPE remains the count-error metric; neither is 'accuracy'.\n")
    report.append("| method | h | RF F2 | ch F2 | ΔF2 (ch−RF) | 95% CI ΔF2 | verdict | Δsens 95% CI | Δprec 95% CI |")
    report.append("|---|---|---|---|---|---|---|---|---|")
    for wr in warning_rows:
        report.append(
            f"| {wr['method_id']} | h{wr['horizon']} | {_fmt3(wr['rf_f2'])} | {_fmt3(wr['challenger_f2'])} | "
            f"{_fmt3(wr['f2_diff_point'], sign=True)} | {_fmt_ci(wr['f2_diff_ci_low'], wr['f2_diff_ci_high'])} | "
            f"{wr['f2_verdict']} | {_fmt_ci(wr['sensitivity_diff_ci_low'], wr['sensitivity_diff_ci_high'])} | "
            f"{_fmt_ci(wr['precision_diff_ci_low'], wr['precision_diff_ci_high'])} |")
    report.append("\n**F2 conclusions (primary challengers, per horizon):**")
    for wr in warning_rows:
        if wr["classification"] != "primary":
            continue
        verdict = wr["f2_verdict"]
        if verdict == "inconclusive":
            report.append(f"- {wr['method_id']} vs RF at h{wr['horizon']}: **inconclusive on F2** (95% CI includes zero) — not a tie, not one beating the other.")
        elif verdict == "RF higher F2":
            report.append(f"- RF showed **higher paired surge-detection F2** than {wr['method_id']} at h{wr['horizon']} under the fixed Q75 rule.")
        elif verdict == "challenger higher F2":
            report.append(f"- {wr['method_id']} showed **higher paired F2** than RF at h{wr['horizon']}.")
    report.append("(F2 read with sensitivity/precision/specificity/TP/FP/TN/FN in `paired_warning_bootstrap_results.json`; "
                  "no unweighted mean-F2 selection across horizons.)\n")

    report.append("\n## Predeclared interpretation\n"
                  "- Primary ranking is lowest paired common-row WAPE, read alongside coverage, the leakage "
                  "audit, convergence/failure stability, MAE/RMSE, warning trade-offs (F2/F1/sensitivity/"
                  "precision/specificity together), and bootstrap uncertainty.\n"
                  "- A bootstrap interval excluding zero indicates a consistent paired difference under this "
                  "benchmark; it does **not** prove universal superiority.\n"
                  "- The Davis adaptation cannot establish the performance of the full original Davis method.\n")
    report.append("## Adaptation limitations\n"
                  "- Every challenger is an origin-observable, direct-horizon adaptation on the shared 12 folds "
                  "(see `literature/adaptation_register.csv`). Teklehaimanot's target-relative weather lags would "
                  "leak future weather at h8/h12 and were made origin-observable; ERA5-Land mean temperature "
                  "substitutes the original min/max. Zinszer is an adapted ARIMAX/dynamic-regression, not the "
                  "paper's exact formulation; vegetation and clinical inputs were unavailable.\n")
    report.append(f"## Coverage / failures\nSee `paired_rf_comparison.json`, `fit_diagnostics.csv`, and "
                  f"`job_convergence_log.json`. Failed fits remain explicit failed records (no borrowing).\n")
    report.append(f"## Runtime\n{runtime_seconds} s (see `runtime_report.json`).\n")
    report.append(f"## Protected-artifact integrity\nSee `final_integrity_manifest.json`. This benchmark did "
                  f"**not** change any thesis, dashboard, production forecast, model-selection artifact, source "
                  f"data, schema, or canonical frozen result. Plots: {plots_note}.\n")

    path = run_dir / "BENCHMARK_REPORT.md"
    leakage.assert_output_isolated(path)
    path.write_text("\n".join(report), encoding="utf-8")
