"""Reports & Exports (Phase 29D).

Authenticated, read-only report generation. Files are rendered into a private
temporary directory and streamed back with an attachment disposition, then
deleted. No caller-supplied path or filename is accepted, no database write
occurs, and no pipeline command is ever invoked.
"""
from __future__ import annotations

import logging
import os
import threading

from flask import Response, render_template, request
from flask_login import login_required

from app.blueprints.dashboard.routes import api_login_required
from app.blueprints.reports import bp
from app.services import dashboard_data as dd
from app.services.report_data_service import (
    FORMAT_CSV,
    FORMAT_HTML,
    FORMAT_PDF,
    RISK_LEVELS,
    ReportParameterError,
    ReportUnavailableError,
    build_context,
    report_registry,
    validate_request,
)
from app.services.report_export_service import content_type, safe_filename, temp_path
from app.services.report_renderers import (
    render_report_csv,
    render_report_pdf,
    render_report_xlsx,
)

log = logging.getLogger(__name__)

# Resource controls (no job queue — a lightweight in-process guard, per the prompt).
MAX_OUTPUT_BYTES = 25 * 1024 * 1024          # controlled response if a report exceeds this
MAX_CONCURRENT_GENERATIONS = 4               # bound simultaneous heavy generations
_generation_slots = threading.BoundedSemaphore(MAX_CONCURRENT_GENERATIONS)

PRINT_TEMPLATES = {
    "national_summary": "reports/print/national_summary.html",
    "district_forecast": "reports/print/district_forecast.html",
    "district_risk_list": "reports/print/district_risk_list.html",
    "model_performance": "reports/print/model_performance.html",
    "data_quality": "reports/print/data_quality.html",
    "climate_signals": "reports/print/climate_signals.html",
    "methods": "reports/print/methods.html",
}


@bp.route("/")
@login_required
def index():
    """Reports page. Filter options and formats come from the server registry."""
    filters = {"horizons": [], "districts": [], "issues": []}
    if not dd.dashboard_precondition_errors(require_outputs=True):
        filters = dd.build_filters_section()
    return render_template(
        "reports/index.html",
        registry=report_registry(),
        filters=filters,
        risk_levels=RISK_LEVELS,
        default_horizon=dd.DEFAULT_HORIZON,
    )


def _download_name(req, extension):
    parts = ["malaria", req.report_type]
    if req.district_id:
        parts.append(req.district_id)
    issue_label = None
    if req.issue_year and req.issue_week:
        issue_label = dd.issue_label(req.issue_year, req.issue_week)
    else:
        issues = dd.available_issues()
        if issues:
            issue_label = issues[0]["issue_label"]
    if issue_label:
        parts.append(issue_label)
    if "horizon" in req.spec["filters"]:
        parts.append(f"h{req.horizon}")
    return safe_filename(*parts, extension=extension)


@bp.route("/generate")
@api_login_required
def generate():
    """Generate and stream one report. Read-only, parameter-bounded, side-effect free."""
    try:
        req = validate_request(request.args)
    except ReportParameterError as exc:
        return _json_error(str(exc), 400, "invalid_request")

    if req.fmt == FORMAT_HTML:
        # Print-friendly HTML is served by the dedicated print route.
        return _json_error("Use the print view for HTML output.", 400, "invalid_request")

    # Acquire before building the context: evidence queries are part of report
    # generation and must not bypass the resource bound under concurrent load.
    if not _generation_slots.acquire(blocking=False):
        return _json_error("The report service is busy. Please try again in a moment.", 503, "busy")
    try:
        try:
            ctx = build_context(req)
        except ReportParameterError as exc:
            return _json_error(str(exc), 400, "invalid_request")
        except ReportUnavailableError as exc:
            return _json_error(str(exc), 503, "unavailable")
        except Exception:  # noqa: BLE001 - never leak internals
            log.exception("Report context build failed for %s", req.report_type)
            return _json_error(
                "The selected report could not be generated because validated evidence is unavailable.",
                503, "unavailable")

        if req.fmt == FORMAT_CSV:
            try:
                payload = render_report_csv(req.report_type, ctx)
            except Exception:  # noqa: BLE001
                log.exception("CSV render failed for %s", req.report_type)
                return _json_error("The report could not be generated.", 503, "unavailable")
            return _bounded_response(payload, content_type(FORMAT_CSV), _download_name(req, "csv"))

        extension = "pdf" if req.fmt == FORMAT_PDF else "xlsx"
        # Temp-file creation is guarded together with rendering and reading, so a
        # temporary-storage failure returns controlled JSON (never an HTML 500),
        # and cleanup runs on every path including that failure.
        path = None
        try:
            path = temp_path(extension)
            if req.fmt == FORMAT_PDF:
                render_report_pdf(req.report_type, ctx, path)
            else:
                render_report_xlsx(req.report_type, ctx, path)
            payload = path.read_bytes()
        except Exception:  # noqa: BLE001
            log.exception("%s generation failed for %s", extension, req.report_type)
            return _json_error("The report could not be generated.", 503, "unavailable")
        finally:
            if path is not None:
                _remove(path)

        if not payload:
            return _json_error("The report could not be generated.", 503, "unavailable")
        return _bounded_response(payload, content_type(req.fmt), _download_name(req, extension))
    finally:
        _generation_slots.release()


def _bounded_response(payload, mimetype, filename):
    """Return the file, or a controlled error if it exceeds the size ceiling."""
    if len(payload) > MAX_OUTPUT_BYTES:
        log.warning("Report exceeded the output size ceiling (%d bytes)", len(payload))
        return _json_error("The requested report is too large to generate.", 503, "too_large")
    return Response(
        payload,
        mimetype=mimetype,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(payload)),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@bp.route("/print")
@login_required
def print_view():
    """Authenticated print-friendly HTML report."""
    args = request.args.to_dict()
    args["format"] = FORMAT_HTML
    try:
        req = validate_request(args)
    except ReportParameterError as exc:
        return render_template("reports/print/unavailable.html", reason=str(exc)), 400

    if not _generation_slots.acquire(blocking=False):
        return render_template(
            "reports/print/unavailable.html",
            reason="The report service is busy. Please try again in a moment.",
        ), 503
    try:
        try:
            ctx = build_context(req)
        except ReportUnavailableError as exc:
            return render_template("reports/print/unavailable.html", reason=str(exc)), 503
        except Exception:  # noqa: BLE001
            log.exception("Print context build failed for %s", req.report_type)
            return render_template(
                "reports/print/unavailable.html",
                reason="The selected report could not be generated because validated evidence is unavailable.",
            ), 503
        return render_template(PRINT_TEMPLATES[req.report_type], ctx=ctx, meta=ctx["metadata"])
    finally:
        _generation_slots.release()


def _remove(path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _json_error(message, status, kind):
    from flask import jsonify

    return (
        jsonify({
            "status": kind, "reason": message, "data": None, "params": None,
            "empty_states": [message], "metadata": {}, "warnings": [], "limitations": [],
        }),
        status,
    )
