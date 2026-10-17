"""Read-only protected-artifact integrity manifest.

Records a baseline fingerprint of every canonical / frozen / protected artifact
BEFORE the benchmark runs, and re-verifies it AFTER, failing if anything
protected changed. Writes ONLY under runs/<RUN_ID>/.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from research_comparisons.literature_based_forecasting_benchmark import config as C


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_map(relpaths) -> dict:
    out: dict[str, dict] = {}
    for rel in relpaths:
        rel = str(rel)
        path = C.REPO_ROOT / rel
        is_file = path.exists() and path.is_file()
        out[rel] = {
            "exists": path.exists(),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size if is_file else None,
        }
    return out


def _dir_fingerprint(rel: str) -> dict:
    base = C.REPO_ROOT / rel
    if not base.exists():
        return {"exists": False, "file_count": 0, "aggregate_sha256": None}
    entries = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            entries.append(
                (path.relative_to(base).as_posix(), _sha256_file(path), path.stat().st_size)
            )
    aggregate = hashlib.sha256()
    top_level_counts: dict[str, int] = {}
    for relp, sha, size in entries:
        aggregate.update(f"{relp}\0{sha}\0{size}\n".encode("utf-8"))
        top = relp.split("/", 1)[0]
        top_level_counts[top] = top_level_counts.get(top, 0) + 1
    return {
        "exists": True,
        "file_count": len(entries),
        "aggregate_sha256": aggregate.hexdigest(),
        "top_level_counts": top_level_counts,
    }


def _git_status() -> dict:
    def run(args):
        try:
            proc = subprocess.run(
                ["git", *args], cwd=C.REPO_ROOT, capture_output=True, text=True, timeout=60
            )
            return proc.stdout.strip()
        except Exception as exc:  # pragma: no cover
            return f"UNAVAILABLE: {exc}"

    return {
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "head": run(["rev-parse", "HEAD"]),
        "porcelain": run(["status", "--porcelain"]).splitlines(),
    }


def _python_env() -> dict:
    versions: dict[str, str] = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "xgboost"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception as exc:  # pragma: no cover
            versions[mod] = f"UNAVAILABLE: {exc}"
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def build_manifest(kind: str) -> dict:
    dataset_rel = C.CANONICAL_DATASET.relative_to(C.REPO_ROOT).as_posix()
    return {
        "kind": kind,
        "run_id": C.RUN_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Isolated literature-based benchmark; read-only over canonical artifacts.",
        "git": _git_status(),
        "python_env": _python_env(),
        "canonical_dataset": {
            **_hash_map([dataset_rel]),
            "expected_sha256": C.CANONICAL_DATASET_SHA256,
            "matches_expected": _sha256_file(C.CANONICAL_DATASET) == C.CANONICAL_DATASET_SHA256,
        },
        "frozen_prediction_artifacts": _hash_map(C.FROZEN_PREDICTION_ARTIFACTS.values()),
        "frozen_metric_artifacts": _hash_map(C.FROZEN_METRIC_ARTIFACTS.values()),
        "canonical_summaries": _hash_map(C.CANONICAL_SUMMARIES),
        "canonical_modules": _hash_map(C.CANONICAL_MODULES),
        "production_forecast_fingerprint": _dir_fingerprint(C.PRODUCTION_FORECAST_DIR),
    }


_PROTECTED_SECTIONS = (
    "canonical_dataset",
    "frozen_prediction_artifacts",
    "frozen_metric_artifacts",
    "canonical_summaries",
    "canonical_modules",
)


def write_baseline() -> Path:
    C.RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest("baseline")
    path = C.RUN_DIR / "baseline_integrity_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def verify_final() -> tuple[Path, list[str]]:
    baseline_path = C.RUN_DIR / "baseline_integrity_manifest.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline manifest missing: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = build_manifest("final")
    drift: list[str] = []
    for section in _PROTECTED_SECTIONS:
        before, after = baseline.get(section, {}), current.get(section, {})
        for key in sorted(set(before) | set(after)):
            b, a = before.get(key), after.get(key)
            if isinstance(b, dict) and isinstance(a, dict) and b.get("sha256") != a.get("sha256"):
                drift.append(f"{section}:{key}")
    if baseline.get("production_forecast_fingerprint", {}).get("aggregate_sha256") != current[
        "production_forecast_fingerprint"
    ].get("aggregate_sha256"):
        drift.append("production_forecast_fingerprint:aggregate_sha256")
    current["protected_integrity_ok"] = not drift
    current["protected_drift"] = drift
    path = C.RUN_DIR / "final_integrity_manifest.json"
    path.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    return path, drift
