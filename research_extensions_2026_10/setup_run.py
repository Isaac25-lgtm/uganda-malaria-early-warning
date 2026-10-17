"""Freeze pre-correction source and local evidence without reading secrets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_extensions_2026_10"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = OUT / "baseline" / "manifest.json"
    if manifest_path.exists():
        print("Baseline already frozen; refusing to replace it.")
        return
    paths = []
    for folder in ("app", "tests", "migrations", "docs"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".md", ".html", ".css", ".js"})
    paths.extend(p for p in (ROOT / "research_comparisons").rglob("*.py"))
    paths.extend(ROOT / p for p in ("README.md", ".gitignore", "requirements.txt", "requirements-models.txt", "requirements-climate.txt", "run.py"))
    paths.extend((ROOT / "data/quality").glob("*.json"))
    paths.extend((ROOT / "data/modelling").glob("*.csv"))
    paths.extend((ROOT / "data/models").glob("*.json"))
    paths.extend((ROOT / "data/models").glob("*importance.csv"))
    paths.extend((ROOT / "data/reference").glob("*.csv"))
    paths.extend((ROOT / "data/reference").glob("*.yml"))
    paths.append(ROOT / "data/models/production/production_model_manifest.json")
    manifest = []
    for p in sorted(set(paths)):
        if not p.exists() or "__pycache__" in p.parts:
            continue
        relative = p.relative_to(ROOT)
        dest = OUT / "baseline" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        manifest.append({"path": relative.as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)})
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    payload = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True), "python": platform.python_version(), "packages": packages, "files": manifest, "surveillance_end_date_cap": "2026-06-30", "new_dhis2_extraction": "not_started_pending_documented_research_authorization"}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "environment-lock.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(packages.items(), key=lambda kv: kv[0].lower())) + "\n", encoding="utf-8")
    print(json.dumps({"frozen_files": len(manifest), "total_bytes": sum(x["bytes"] for x in manifest)}))


if __name__ == "__main__":
    main()
