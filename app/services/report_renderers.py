"""Phase 29D per-report renderers (PDF blocks, Excel sheets, CSV rows).

Renderers only arrange values that the report context already contains. They
perform no analytical calculation, invent no value, and omit any section whose
evidence is unavailable rather than substituting a placeholder number.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, Spacer

from app.services.report_export_service import (
    _data_table,
    _kv_table,
    _styles,
    render_csv,
    render_pdf,
    render_xlsx,
)

RISK_ORDER = {"Very High": 4, "High": 3, "Moderate": 2, "Low": 1, "No data": 0}


def _n(value, dp=0):
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{dp}f}"
    except (TypeError, ValueError):
        return str(value)


def _signed(value, dp=0):
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{'+' if number > 0 else ''}{number:,.{dp}f}"


def _meta_pairs(meta: dict[str, Any]) -> list[tuple[str, Any]]:
    pairs = [
        ("Report type", meta.get("report_title")),
        ("Production issue week", meta.get("issue_label")),
        ("Forecast horizon", f"{meta['horizon_weeks']} weeks" if meta.get("horizon_weeks") else None),
        ("Target week", meta.get("target_label")),
        ("District", meta.get("district_name")),
        ("Selected count model", meta.get("count_model_name")),
        ("Selected surge model", meta.get("surge_model_name")),
        ("Generated (server)", meta.get("generated_at_utc")),
        ("Evidence generated", meta.get("evidence_generated_at_utc")),
        ("Dataset signature", meta.get("dataset_signature_short")),
        ("Application version", meta.get("application_version")),
        ("Report schema", meta.get("report_schema_version")),
    ]
    return [(k, v) for k, v in pairs if v not in (None, "")]


def _header_blocks(ctx, st, subtitle=None):
    meta = ctx["metadata"]
    blocks = [
        Paragraph(meta["report_title"], st["title"]),
        Paragraph(subtitle or "Uganda District-Level Malaria Early Warning — research output", st["sub"]),
        _kv_table(_meta_pairs(meta), st),
        Spacer(1, 6),
    ]
    return blocks


def _limitations_blocks(ctx, st, heading="Interpretation and limitations"):
    blocks = [Paragraph(heading, st["h2"])]
    for item in ctx.get("limitations", []):
        blocks.append(Paragraph(f"• {item}", st["small"]))
    blocks.append(Spacer(1, 4))
    blocks.append(Paragraph(ctx["metadata"].get("research_use", ""), st["small"]))
    return blocks


def _sources_blocks(ctx, st):
    blocks = [Paragraph("Data sources", st["h2"])]
    for item in ctx["metadata"].get("data_sources", []):
        blocks.append(Paragraph(f"• {item}", st["small"]))
    return blocks


# --------------------------------------------------------------------------
# PDF renderers
# --------------------------------------------------------------------------
def _pdf_national(ctx, st):
    b = _header_blocks(ctx, st, "National early warning summary for the published production issue.")
    b.append(Paragraph("Executive summary", st["h2"]))
    for line in ctx["executive_summary"]:
        b.append(Paragraph(f"• {line}", st["body"]))
    b.append(Spacer(1, 6))

    counts = ctx["risk_counts"]
    b.append(Paragraph("Risk distribution", st["h2"]))
    b.append(_data_table(
        ["Risk level", "Districts"],
        [[level, counts.get(level, 0)] for level in ("Very High", "High", "Moderate", "Low", "No data")
         if counts.get(level)],
        st, col_widths=[60 * mm, 30 * mm], risk_col=0))
    b.append(Spacer(1, 8))

    weekly = ctx.get("weekly") or {}
    forecast_rows = [r for r in weekly.get("rows", []) if r.get("period") == "forecast"]
    observed = [r for r in weekly.get("rows", []) if r.get("period") == "observed"]
    b.append(Paragraph("National trend and target-week forecast", st["h2"]))
    trend_rows = []
    for row in observed[-8:]:
        trend_rows.append([row["week_label"], "Observed", _n(row["actual_cases"]), "—"])
    for row in forecast_rows:
        trend_rows.append([row["week_label"], "Forecast target", _n(row["predicted_cases"]),
                           _n(row["surge_threshold"])])
    b.append(_data_table(["Week", "Period", "Cases", "Threshold"], trend_rows, st,
                         col_widths=[28 * mm, 34 * mm, 32 * mm, 32 * mm]))
    b.append(Paragraph(weekly.get("threshold_note", ""), st["small"]))
    b.append(Paragraph(weekly.get("intervals_note", ""), st["small"]))
    b.append(PageBreak())

    b.append(Paragraph("Priority districts", st["h2"]))
    rows = [[r["district_name"], r["risk_level"], _n(r["predicted_cases"]),
             _n(r["surge_predicted_cases"]), _n(r["threshold_used"]),
             _signed(r["predicted_surge_score"]),
             _n(r["reporting_completeness"], 1) if r["reporting_completeness"] is not None else "—",
             f"{r['forecast_horizon_weeks']}w"]
            for r in ctx["priority_districts"]]
    b.append(_data_table(
        ["District", "Risk", "Predicted", "Surge model", "Threshold", "Score", "Complete %", "Horizon"],
        rows, st, col_widths=[34 * mm, 21 * mm, 23 * mm, 24 * mm, 23 * mm, 20 * mm, 22 * mm, 16 * mm],
        risk_col=1))
    b.append(Spacer(1, 8))
    b += _limitations_blocks(ctx, st)
    b += _sources_blocks(ctx, st)
    return b


def _pdf_district(ctx, st):
    d = ctx["district"]
    b = _header_blocks(ctx, st, f"District forecast summary for {d['district_name']}.")
    b.append(Paragraph("Forecast summary", st["h2"]))
    b.append(_kv_table([
        ("Risk level", d["risk_level"]),
        (f"Count-model predicted cases ({d.get('count_model_name')})", _n(d["predicted_cases"])),
        (f"Surge-model predicted cases ({d.get('surge_model_name')})", _n(d["surge_predicted_cases"])),
        ("District threshold", _n(d["threshold_used"])),
        ("Surge score (cases)", _signed(d["predicted_surge_score"])),
        ("Reporting completeness", _n(d["reporting_completeness"], 1) + "%" if d["reporting_completeness"] is not None else "Not reported"),
        ("Observed cases (issue week)", _n(d["observed_cases"])),
        ("Issue → target week", f"{d.get('issue_label')} → {d.get('target_label')}"),
    ], st))
    b.append(Spacer(1, 6))

    b.append(Paragraph("Interpretation", st["h2"]))
    for line in ctx["interpretation"]:
        b.append(Paragraph(line, st["body"]))
    b.append(Spacer(1, 6))

    weekly = ctx.get("weekly") or {}
    observed = [r for r in weekly.get("rows", []) if r.get("period") == "observed"][-10:]
    forecast = [r for r in weekly.get("rows", []) if r.get("period") == "forecast"]
    b.append(Paragraph("Recent observed cases and target-week forecast", st["h2"]))
    rows = [[r["week_label"], "Observed", _n(r["actual_cases"]), "—"] for r in observed]
    rows += [[r["week_label"], "Forecast target", _n(r["predicted_cases"]), _n(r["surge_threshold"])]
             for r in forecast]
    b.append(_data_table(["Week", "Period", "Cases", "Threshold"], rows, st,
                         col_widths=[30 * mm, 36 * mm, 32 * mm, 32 * mm]))
    b.append(Paragraph(weekly.get("threshold_note", ""), st["small"]))

    climate = ctx.get("climate") or {}
    if climate.get("rows"):
        b.append(Paragraph("Climate context", st["h2"]))
        crows = []
        for row in climate["rows"]:
            m = row.get("measures", {})
            crows.append([
                row["label"], row["source"], row["unit"],
                _n((m.get("current") or {}).get("value"), 1),
                _n((m.get("lag_4") or {}).get("value"), 1),
                _n((m.get("lag_8") or {}).get("value"), 1),
                _n((m.get("lag_12") or {}).get("value"), 1),
            ])
        b.append(_data_table(["Variable", "Source", "Unit", "Current", "4w lag", "8w lag", "12w lag"],
                             crows, st, col_widths=[26 * mm, 44 * mm, 16 * mm, 22 * mm, 20 * mm, 20 * mm, 20 * mm]))
        b.append(Paragraph(climate.get("causation_note", ""), st["small"]))
    b.append(Spacer(1, 8))
    b += _limitations_blocks(ctx, st)
    return b


def _pdf_risk_list(ctx, st):
    b = _header_blocks(ctx, st, "Operational district risk list for the selected issue and horizon.")
    rows = [[r["district_id"], r["district_name"], r["risk_level"], _n(r["predicted_cases"]),
             _n(r["surge_predicted_cases"]), _n(r["threshold_used"]), _signed(r["predicted_surge_score"]),
             _n(r["reporting_completeness"], 1) if r["reporting_completeness"] is not None else "—",
             r.get("target_label") or "—"]
            for r in ctx["rows"]]
    b.append(Paragraph(f"Districts listed: {len(rows)}", st["body"]))
    b.append(Spacer(1, 4))
    b.append(_data_table(
        ["District ID", "District", "Risk", "Predicted", "Surge model", "Threshold", "Score", "Complete %", "Target"],
        rows, st,
        col_widths=[30 * mm, 34 * mm, 20 * mm, 24 * mm, 25 * mm, 24 * mm, 21 * mm, 22 * mm, 23 * mm],
        risk_col=2))
    b.append(Spacer(1, 6))
    b.append(Paragraph(ctx.get("score_definition") or "", st["small"]))
    b += _limitations_blocks(ctx, st)
    return b


def _pdf_model(ctx, st):
    b = _header_blocks(ctx, st, "Validation performance and model selection for the approved models.")
    horizon = ctx["metadata"].get("horizon_weeks")
    rows = [r for r in ctx["performance"].get("rows", []) if str(r.get("horizon")) == str(horizon)]

    b.append(Paragraph(f"Count forecasting metrics (horizon {horizon} weeks)", st["h2"]))
    b.append(_data_table(
        ["Model", "Scope", "MAE", "RMSE", "MAPE", "WAPE", "R²", "Selected"],
        [[r["model_name"], r.get("model_scope"), _n(r["mae"], 1), _n(r["rmse"], 1), _n(r["mape"], 1),
          _n(r["wape"], 1), _n(r["r_squared"], 3), "Count" if r.get("selected_for_count") else "—"]
         for r in rows], st,
        col_widths=[34 * mm, 34 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 20 * mm, 22 * mm]))
    b.append(Spacer(1, 6))

    b.append(Paragraph(f"Surge detection metrics (horizon {horizon} weeks)", st["h2"]))
    b.append(_data_table(
        ["Model", "Sensitivity", "Specificity", "Precision", "F1", "AUC", "Selected"],
        [[r["model_name"], _n(r["sensitivity"], 3), _n(r["specificity"], 3), _n(r["precision"], 3),
          _n(r["f1"], 3), _n(r["auc"], 3), "Surge" if r.get("selected_for_surge") else "—"]
         for r in rows], st,
        col_widths=[38 * mm, 26 * mm, 26 * mm, 26 * mm, 22 * mm, 22 * mm, 24 * mm]))
    b.append(Paragraph("Accuracy is available in the underlying artifact but is NOT used for model "
                       "selection, because surge weeks are rare and accuracy can be high while most "
                       "surges are missed.", st["small"]))
    b.append(Spacer(1, 6))

    sel = ctx["selection"]
    b.append(Paragraph("Model selection", st["h2"]))
    b.append(_kv_table([
        ("Selected count model", sel.get("selected_count_model")),
        ("Count selection rule", (sel.get("selection_rules") or {}).get("count")),
        ("Selected surge model", sel.get("selected_surge_model")),
        ("Surge selection rule", (sel.get("selection_rules") or {}).get("surge")),
        ("Winner recalculated", "No — published evidence is used"),
    ], st))
    b.append(Spacer(1, 6))

    val = ctx["validation"]
    b.append(Paragraph("Validation method", st["h2"]))
    for line in val.get("statements", []):
        b.append(Paragraph(f"• {line}", st["small"]))

    imp = ctx.get("importance")
    if imp and imp.get("features"):
        b.append(PageBreak())
        b.append(Paragraph(f"Feature importance — {imp.get('model_name')} ({imp.get('importance_definition')})", st["h2"]))
        b.append(_data_table(
            ["Rank", "Feature", "Family", "Share %"],
            [[f["rank"], f["label"], f["family"], _n(f["normalized_percentage"], 2)]
             for f in imp["features"]], st,
            col_widths=[16 * mm, 92 * mm, 46 * mm, 24 * mm]))
        for line in ctx.get("importance_limitations", []):
            b.append(Paragraph(f"• {line}", st["small"]))
    b.append(Spacer(1, 6))
    b += _limitations_blocks(ctx, st)
    return b


def _pdf_quality(ctx, st):
    b = _header_blocks(ctx, st, "Data quality checks applied before modelling.")
    cov = ctx["coverage"]
    b.append(Paragraph("Surveillance coverage", st["h2"]))
    b.append(_kv_table([
        ("Expected district-weeks", _n(cov["expected_district_weeks"])),
        ("Received district-weeks", _n(cov["received_district_weeks"])),
        ("Missing district-weeks (absent rows)", _n(cov["missing_district_weeks"])),
        ("Explicit-zero records", _n(cov["explicit_zero_records"])),
        ("Raw DHIS2 rows", _n(cov["raw_dhis2_rows"])),
        ("Districts represented", _n(cov["districts_represented"])),
    ], st))
    b.append(Spacer(1, 6))

    b.append(Paragraph("Pipeline component status", st["h2"]))
    b.append(_data_table(["Component", "Status"],
                         [[k.replace("_", " ").title(), v] for k, v in ctx["components"].items()],
                         st, col_widths=[80 * mm, 40 * mm]))
    b.append(Spacer(1, 6))

    # Leakage controls: evidence-derived value + source (never static prose).
    b.append(Paragraph("Leakage controls (evidence-derived)", st["h2"]))
    lc = ctx.get("leakage_controls") or {}
    label = {
        "missing_not_zero_filled": "Missing not zero-filled",
        "future_target_alignment": "Future-target alignment",
        "training_only_preprocessing": "Training-only preprocessing",
        "adjudicated_errors_excluded_from_outcomes": "Adjudicated errors excluded from outcomes",
        "bounded_predictor_only_interpolation": "Bounded predictor-only interpolation",
    }
    b.append(_data_table(
        ["Control", "Verified", "Source"],
        [[label.get(k, k), "Yes" if v.get("value") is True else "No", v.get("source")]
         for k, v in lc.items()],
        st, col_widths=[52 * mm, 22 * mm, 96 * mm]))
    b.append(Paragraph(
        f"Rows with all nine climate lags present: {_n(ctx.get('rows_with_complete_climate_lags'))} "
        f"of {_n(ctx['coverage'].get('received_district_weeks'))}.", st["small"]))
    b.append(Spacer(1, 6))

    b.append(Paragraph("Quality rules and definitions", st["h2"]))
    for term, meaning in ctx["definitions"]:
        b.append(Paragraph(f"<b>{term}.</b> {meaning}", st["small"]))
    b.append(Spacer(1, 4))
    b.append(Paragraph("Climate lag columns verified: " + ", ".join(ctx["climate_lag_columns"]), st["small"]))
    if ctx.get("evidence_warnings"):
        b.append(Spacer(1, 4))
        b.append(Paragraph("Evidence warnings", st["h2"]))
        for warning in ctx["evidence_warnings"]:
            b.append(Paragraph(f"• {warning}", st["small"]))
    b.append(Spacer(1, 6))
    b += _limitations_blocks(ctx, st)
    return b


def _pdf_climate(ctx, st):
    climate = ctx["climate"]
    b = _header_blocks(ctx, st, "Climate predictors used as supporting inputs to the forecasting models.")
    b.append(Paragraph("Climate values", st["h2"]))
    b.append(Paragraph(f"Data week {climate.get('data_week_label') or '—'} · {ctx.get('scope_note') or ''}", st["body"]))
    rows = []
    for row in climate.get("rows", []):
        m = row.get("measures", {})
        rows.append([row["label"], row["source"], row["unit"],
                     _n((m.get("current") or {}).get("value"), 1),
                     _n((m.get("lag_4") or {}).get("value"), 1),
                     _n((m.get("lag_8") or {}).get("value"), 1),
                     _n((m.get("lag_12") or {}).get("value"), 1),
                     f"{(m.get('current') or {}).get('districts_with_value', 0)}"])
    b.append(_data_table(["Variable", "Source", "Unit", "Current", "4w lag", "8w lag", "12w lag", "Districts"],
                         rows, st,
                         col_widths=[24 * mm, 42 * mm, 15 * mm, 20 * mm, 18 * mm, 18 * mm, 18 * mm, 18 * mm]))
    b.append(Spacer(1, 6))
    b.append(Paragraph("Interpretation", st["h2"]))
    b.append(Paragraph(climate.get("correlation_note", ""), st["small"]))
    b.append(Paragraph(climate.get("causation_note", ""), st["small"]))
    b.append(Spacer(1, 6))
    b += _limitations_blocks(ctx, st)
    return b


def _pdf_methods(ctx, st):
    b = _header_blocks(ctx, st, "Research methods and interpretation summary.")
    for index, (heading, text) in enumerate(ctx["sections"], start=1):
        b.append(Paragraph(f"{index}. {heading}", st["h2"]))
        b.append(Paragraph(text, st["body"]))
    b.append(Spacer(1, 6))
    b.append(Paragraph("Selected models by horizon", st["h2"]))
    b.append(_data_table(
        ["Horizon", "Count model", "Surge model"],
        [[f"{h} weeks", s["count"]["model_name"], s["surge"]["model_name"]]
         for h, s in ctx["selection_by_horizon"].items()], st,
        col_widths=[32 * mm, 55 * mm, 55 * mm]))
    return b


_PDF = {
    "national_summary": _pdf_national,
    "district_forecast": _pdf_district,
    "district_risk_list": _pdf_risk_list,
    "model_performance": _pdf_model,
    "data_quality": _pdf_quality,
    "climate_signals": _pdf_climate,
    "methods": _pdf_methods,
}


# --------------------------------------------------------------------------
# Excel sheet builders
# --------------------------------------------------------------------------
def _meta_sheet(ctx):
    meta = ctx["metadata"]
    rows = [[k, v] for k, v in _meta_pairs(meta)]
    rows.append(["Research use", meta.get("research_use")])
    for index, source in enumerate(meta.get("data_sources", []), start=1):
        rows.append([f"Data source {index}", source])
    return {"name": "Metadata", "headers": [], "rows": rows}


def _limits_sheet(ctx):
    return {"name": "Limitations", "headers": ["Limitation"],
            "rows": [[item] for item in ctx.get("limitations", [])]}


RISK_HEADERS = ["District ID", "District", "Risk level", "Predicted cases", "Surge-model cases",
                "Threshold", "Surge score", "Reporting completeness", "Target week",
                "Count model", "Surge model", "Horizon weeks"]


def _risk_rows(rows):
    return [[r["district_id"], r["district_name"], r["risk_level"], r["predicted_cases"],
             r["surge_predicted_cases"], r["threshold_used"], r["predicted_surge_score"],
             r["reporting_completeness"], r.get("target_label"), r.get("count_model_name"),
             r.get("surge_model_name"), r.get("forecast_horizon_weeks")] for r in rows]


def _xlsx_sheets(report_type, ctx):
    if report_type == "national_summary":
        s = ctx["summary"]
        return [
            _meta_sheet(ctx),
            {"name": "KPI Summary", "headers": ["Indicator", "Value"], "rows": [
                ["Confirmed malaria cases", s.get("confirmed_malaria_cases")],
                ["Reporting completeness (%)", s.get("reporting_completeness")],
                ["Districts in scope", s.get("districts_in_scope")],
                ["Districts at surge risk", s.get("districts_at_surge_risk")],
                ["Selected count model", ctx["selected_models"]["count"]["model_name"]],
                ["Selected surge model", ctx["selected_models"]["surge"]["model_name"]],
                ["Target week", s.get("target_label")],
            ]},
            {"name": "District Risk", "headers": RISK_HEADERS, "rows": _risk_rows(ctx["all_rows"]),
             "percent_cols": {7}},
            {"name": "Model Selection", "headers": ["Task", "Model", "Rule"], "rows": [
                ["Count", ctx["selected_models"]["count"]["model_name"], ctx["selected_models"]["count"].get("metric_rule")],
                ["Surge", ctx["selected_models"]["surge"]["model_name"], ctx["selected_models"]["surge"].get("metric_rule")],
            ]},
            _limits_sheet(ctx),
        ]
    if report_type == "district_forecast":
        d = ctx["district"]
        weekly = [[r["week_label"], r["period"], r["actual_cases"], r["predicted_cases"],
                   r["surge_threshold"]] for r in (ctx.get("weekly") or {}).get("rows", [])]
        climate_rows = []
        for row in (ctx.get("climate") or {}).get("rows", []):
            m = row.get("measures", {})
            climate_rows.append([row["label"], row["source"], row["unit"],
                                 (m.get("current") or {}).get("value"),
                                 (m.get("lag_4") or {}).get("value"),
                                 (m.get("lag_8") or {}).get("value"),
                                 (m.get("lag_12") or {}).get("value")])
        return [
            _meta_sheet(ctx),
            {"name": "District Summary", "headers": ["Field", "Value"], "rows": [
                ["District", d["district_name"]], ["District ID", d["district_id"]],
                ["Risk level", d["risk_level"]],
                ["Count-model predicted cases", d["predicted_cases"]],
                ["Surge-model predicted cases", d["surge_predicted_cases"]],
                ["Threshold", d["threshold_used"]],
                ["Surge score (cases)", d["predicted_surge_score"]],
                ["Reporting completeness (%)", d["reporting_completeness"]],
                ["Risk basis", d.get("risk_basis")],
                ["Issue week", d.get("issue_label")], ["Target week", d.get("target_label")],
            ]},
            {"name": "Weekly Series", "headers": ["Week", "Period", "Observed cases",
                                                  "Predicted cases", "Threshold"], "rows": weekly},
            {"name": "Climate Signals", "headers": ["Variable", "Source", "Unit", "Current",
                                                    "4w lag", "8w lag", "12w lag"], "rows": climate_rows},
            _limits_sheet(ctx),
        ]
    if report_type == "district_risk_list":
        return [
            _meta_sheet(ctx),
            {"name": "District Risk", "headers": RISK_HEADERS, "rows": _risk_rows(ctx["rows"]),
             "percent_cols": {7}},
            _limits_sheet(ctx),
        ]
    if report_type == "model_performance":
        horizon = ctx["metadata"].get("horizon_weeks")
        rows = [r for r in ctx["performance"].get("rows", []) if str(r.get("horizon")) == str(horizon)]
        sheets = [
            _meta_sheet(ctx),
            {"name": "Count Metrics", "headers": ["Model", "Scope", "MAE", "RMSE", "MAPE", "WAPE",
                                                  "R squared", "Selected for count"],
             "rows": [[r["model_name"], r.get("model_scope"), r["mae"], r["rmse"], r["mape"],
                       r["wape"], r["r_squared"], bool(r.get("selected_for_count"))] for r in rows]},
            {"name": "Surge Metrics", "headers": ["Model", "Sensitivity", "Specificity", "Precision",
                                                  "F1", "AUC", "Selected for surge"],
             "rows": [[r["model_name"], r["sensitivity"], r["specificity"], r["precision"],
                       r["f1"], r["auc"], bool(r.get("selected_for_surge"))] for r in rows],
             "note": "Accuracy is not used for model selection."},
            {"name": "Model Selection", "headers": ["Task", "Model", "Rule"], "rows": [
                ["Count", ctx["selection"].get("selected_count_model"),
                 (ctx["selection"].get("selection_rules") or {}).get("count")],
                ["Surge", ctx["selection"].get("selected_surge_model"),
                 (ctx["selection"].get("selection_rules") or {}).get("surge")],
            ]},
            {"name": "Validation Method", "headers": ["Statement"],
             "rows": [[s] for s in ctx["validation"].get("statements", [])]},
        ]
        imp = ctx.get("importance")
        if imp and imp.get("features"):
            sheets.append({"name": "Feature Importance",
                           "headers": ["Rank", "Feature", "Family", "Raw mean", "Share %"],
                           "rows": [[f["rank"], f["label"], f["family"], f["raw_mean_importance"],
                                     f["normalized_percentage"]] for f in imp["features"]]})
        sheets.append(_limits_sheet(ctx))
        return sheets
    if report_type == "data_quality":
        cov = ctx["coverage"]
        return [
            _meta_sheet(ctx),
            {"name": "Data Counts", "headers": ["Measure", "Value"], "rows": [
                ["Expected district-weeks", cov["expected_district_weeks"]],
                ["Received district-weeks", cov["received_district_weeks"]],
                ["Missing district-weeks (absent rows)", cov["missing_district_weeks"]],
                ["Explicit-zero records", cov["explicit_zero_records"]],
                ["Raw DHIS2 rows", cov["raw_dhis2_rows"]],
                ["Districts represented", cov["districts_represented"]],
            ]},
            {"name": "Pipeline Status", "headers": ["Component", "Status"],
             "rows": [[k.replace("_", " ").title(), v] for k, v in ctx["components"].items()]},
            {"name": "Leakage Controls", "headers": ["Control", "Verified", "Source"],
             "rows": [[k.replace("_", " ").title(), bool(v.get("value")), v.get("source")]
                      for k, v in (ctx.get("leakage_controls") or {}).items()]},
            {"name": "Quality Checks", "headers": ["Rule", "Meaning"],
             "rows": [[t, m] for t, m in ctx["definitions"]]},
            _limits_sheet(ctx),
        ]
    if report_type == "climate_signals":
        rows = []
        for row in ctx["climate"].get("rows", []):
            m = row.get("measures", {})
            rows.append([row["label"], row["source"], row["unit"],
                         (m.get("current") or {}).get("value"),
                         (m.get("lag_4") or {}).get("value"),
                         (m.get("lag_8") or {}).get("value"),
                         (m.get("lag_12") or {}).get("value"),
                         (m.get("current") or {}).get("districts_with_value")])
        return [
            _meta_sheet(ctx),
            {"name": "Climate Signals",
             "headers": ["Variable", "Source", "Unit", "Current", "4w lag", "8w lag", "12w lag",
                         "Districts with value"],
             "rows": rows, "note": ctx.get("scope_note")},
            _limits_sheet(ctx),
        ]
    raise KeyError(report_type)


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------
def render_report_pdf(report_type: str, ctx: dict[str, Any], path: Path) -> None:
    st = _styles()
    blocks = _PDF[report_type](ctx, st)
    render_pdf(blocks, ctx["metadata"], path, REPORT_ORIENTATION.get(report_type, "portrait"))


def render_report_xlsx(report_type: str, ctx: dict[str, Any], path: Path) -> None:
    render_xlsx(_xlsx_sheets(report_type, ctx), path)


def render_report_csv(report_type: str, ctx: dict[str, Any]) -> bytes:
    if report_type == "district_risk_list":
        return render_csv(RISK_HEADERS, _risk_rows(ctx["rows"]))
    raise KeyError(report_type)


from app.services.report_data_service import REPORTS as _R  # noqa: E402

REPORT_ORIENTATION = {key: spec.get("orientation", "portrait") for key, spec in _R.items()}
