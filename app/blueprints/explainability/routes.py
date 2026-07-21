"""Read-only explainability and system-status API routes (Phase 29B).

Every endpoint is authenticated with the existing DHIS2-backed session, builds
its response from published evidence only, and never writes to the database or
filesystem. The dashboard response envelope is preserved and extended with
``metadata``, ``warnings`` and ``limitations``.
"""
from flask import jsonify, request

# Reuse the dashboard's JSON-401 auth decorator so session/expiry behaviour and
# the fetch-friendly contract are identical across the app (no new auth logic).
from app.blueprints.dashboard.routes import api_login_required
from app.blueprints.explainability import bp
from app.services.dashboard_data import (
    DashboardParameterError,
    parse_dashboard_params,
)
from app.services.explainability import (
    ExplainabilityError,
    ExplainabilityParameterError,
    build_feature_importance,
    build_model_selection,
    build_overview,
    build_pipeline_status,
    build_threshold_method,
    build_validation_method,
)

HORIZONS = (4, 8, 12)


def _envelope(built, params=None):
    """Wrap a service result in the standard envelope with HTTP 200."""
    return (
        jsonify(
            {
                "status": "available",
                "reason": None,
                "data": built.get("data", {}),
                "params": params or {},
                "empty_states": [],
                "metadata": built.get("metadata", {}),
                "warnings": built.get("warnings", []),
                "limitations": built.get("limitations", []),
            }
        ),
        200,
    )


def _invalid(message):
    return (
        jsonify(
            {
                "status": "invalid_request",
                "reason": message,
                "data": None,
                "params": None,
                "empty_states": [message],
                "metadata": {},
                "warnings": [],
                "limitations": [],
            }
        ),
        400,
    )


def _unavailable(message):
    return (
        jsonify(
            {
                "status": "unavailable",
                "reason": message,
                "data": None,
                "params": {},
                "empty_states": ["Explainability evidence unavailable."],
                "metadata": {},
                "warnings": [],
                "limitations": [],
            }
        ),
        503,
    )


def _horizon_arg(default=4):
    raw = (request.args.get("horizon") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ExplainabilityParameterError("horizon must be one of 4, 8, 12.")
    if value not in HORIZONS:
        raise ExplainabilityParameterError("horizon must be one of 4, 8, 12.")
    return value


@bp.route("/api/explainability/overview")
@api_login_required
def api_overview():
    try:
        horizon = _horizon_arg()
        built = build_overview(horizon)
    except ExplainabilityParameterError as exc:
        return _invalid(str(exc))
    except ExplainabilityError as exc:
        return _unavailable(str(exc))
    except Exception:  # defensive: no AttributeError/TypeError/ValueError/trace escapes
        return _unavailable("Explainability evidence could not be processed.")
    return _envelope(built, {"horizon": horizon})


@bp.route("/api/explainability/model-selection")
@api_login_required
def api_model_selection():
    try:
        horizon = _horizon_arg()
        built = build_model_selection(horizon)
    except ExplainabilityParameterError as exc:
        return _invalid(str(exc))
    except ExplainabilityError as exc:
        return _unavailable(str(exc))
    except Exception:  # defensive: no AttributeError/TypeError/ValueError/trace escapes
        return _unavailable("Explainability evidence could not be processed.")
    return _envelope(built, {"horizon": horizon})


@bp.route("/api/explainability/validation-method")
@api_login_required
def api_validation_method():
    try:
        built = build_validation_method()
    except ExplainabilityError as exc:
        return _unavailable(str(exc))
    except Exception:  # defensive: no AttributeError/TypeError/ValueError/trace escapes
        return _unavailable("Explainability evidence could not be processed.")
    return _envelope(built, {})


@bp.route("/api/explainability/threshold-method")
@api_login_required
def api_threshold_method():
    # District is optional; horizon/district/issue validation is reused from the
    # dashboard so a bad district_id returns the same 400 contract.
    try:
        params = parse_dashboard_params(request.args)
    except DashboardParameterError as exc:
        return _invalid(str(exc))
    try:
        built = build_threshold_method(params)
    except ExplainabilityError as exc:
        return _unavailable(str(exc))
    except Exception:  # defensive: no AttributeError/TypeError/ValueError/trace escapes
        return _unavailable("Explainability evidence could not be processed.")
    return _envelope(built, params.to_dict())


@bp.route("/api/explainability/feature-importance")
@api_login_required
def api_feature_importance():
    model = (request.args.get("model") or "").strip().lower()
    method = (request.args.get("method") or "native").strip().lower()
    grouped_raw = (request.args.get("grouped") or "false").strip().lower()
    horizon_raw = (request.args.get("horizon") or "4").strip()
    limit_raw = (request.args.get("limit") or "20").strip()

    if grouped_raw not in ("true", "false"):
        return _invalid("grouped must be true or false.")
    grouped = grouped_raw == "true"
    try:
        horizon = int(horizon_raw)
    except ValueError:
        return _invalid("horizon must be one of 4, 8, 12.")
    try:
        limit = int(limit_raw)
    except ValueError:
        return _invalid("limit must be an integer between 1 and 50.")

    try:
        built = build_feature_importance(
            model=model, horizon=horizon, method=method, limit=limit, grouped=grouped
        )
    except ExplainabilityParameterError as exc:
        return _invalid(str(exc))
    except ExplainabilityError as exc:
        return _unavailable(str(exc))
    except Exception:  # defensive: no AttributeError/TypeError/ValueError/trace escapes
        return _unavailable("Explainability evidence could not be processed.")
    return _envelope(
        built,
        {"model": model, "horizon": horizon, "method": method, "limit": limit, "grouped": grouped},
    )


@bp.route("/api/system/pipeline-status")
@api_login_required
def api_pipeline_status():
    try:
        built = build_pipeline_status()
    except ExplainabilityError as exc:
        return _unavailable(str(exc))
    except Exception:  # defensive: no AttributeError/TypeError/ValueError/trace escapes
        return _unavailable("Explainability evidence could not be processed.")
    return _envelope(built, {})


@bp.route("/api/system/governance")
@api_login_required
def api_governance():
    """Canonical read-only governance evidence (Phase 29E).

    Lives beside /api/system/pipeline-status so both system endpoints share one
    prefix and one auth/error convention. There is a single implementation; the
    Settings page consumes this route.
    """
    from app.services.governance import GovernanceEvidenceError, build_governance

    try:
        built = build_governance()
    except GovernanceEvidenceError as exc:
        return _unavailable(str(exc))
    except Exception:  # defensive: never leak internals
        return _unavailable("Governance evidence could not be processed.")
    response, status = _envelope(built, {})
    response.headers["Cache-Control"] = "no-store"
    return response, status
