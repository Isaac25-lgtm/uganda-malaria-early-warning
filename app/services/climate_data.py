"""Climate data ingestion foundation.

This layer tracks raw CHIRPS and ERA5-Land files only. It does not perform
zonal aggregation, weekly aggregation, lag creation or district_week_features
updates.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from app.extensions import db
from app.models import PipelineRun
from app.services.preprocessing import iso_week_dates

APPROVED_SOURCES = ("CHIRPS", "ERA5-Land")
APPROVED_VARIABLES = ("rainfall", "temperature", "humidity")
SOURCE_VARIABLES = {
    "CHIRPS": ("rainfall",),
    "ERA5-Land": ("temperature", "humidity"),
}
EXPECTED_EXTENSIONS = {
    "CHIRPS": (".tif", ".tiff", ".tif.gz", ".tiff.gz"),
    "ERA5-Land": (".nc", ".grib", ".grb"),
}
DEFAULT_CHIRPS_BASE_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05"
)
CHIRPS_PRODUCT_VERSION = "CHIRPS v2.0 global daily p05"
ERA5_LAND_DATASET = "reanalysis-era5-land"
ERA5_PRODUCT_VERSION = "ERA5-Land hourly data"
ERA5_UGANDA_AREA = [4.3, 29.5, -1.6, 35.2]  # north, west, south, east
ERA5_HOURLY_TIMES = [f"{hour:02d}:00" for hour in range(24)]
ERA5_VARIABLES = {
    "temperature": ("2m_temperature",),
    "humidity": ("2m_temperature", "2m_dewpoint_temperature"),
}
RAW_BOUNDARY_PATH = Path("data/boundaries/UGANDA_DISTRICT.json")
PREPARED_BOUNDARY_PATH = Path("data/boundaries/uganda_districts_prepared.geojson")


class ClimateDataError(RuntimeError):
    """Raised when climate ingestion input is invalid."""


@dataclass(frozen=True)
class BoundaryDependencyStatus:
    raw_boundary_exists: bool
    prepared_boundary_exists: bool
    message: str


@dataclass
class ClimateSourceStatus:
    climate_data_dir: Path
    chirps_base_url_configured: bool
    era5_provider: str | None
    era5_api_url_configured: bool
    era5_api_key_configured: bool
    era5_download_mode: str
    boundary_status: BoundaryDependencyStatus


@dataclass
class ClimateFileValidation:
    source: str
    variable: str
    path: Path
    start_date: date
    end_date: date
    exists: bool
    non_empty: bool
    extension_ok: bool
    size_bytes: int = 0
    checksum_sha256: str | None = None
    product_version: str | None = None
    units: str | None = None
    crs: str | None = None
    detected_variables: list[str] = field(default_factory=list)
    detected_time_range: tuple[str | None, str | None] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class ClimateRegisterSummary:
    pipeline_run_id: int
    manifest_path: Path
    validation: ClimateFileValidation


@dataclass(frozen=True)
class ClimateDateRange:
    analysis_start_week: str
    analysis_end_week: str
    analysis_start_date: date
    analysis_end_date: date
    lag_lookback_start_date: date
    download_start_date: date
    download_end_date: date
    lookback_weeks: int


@dataclass
class ClimateDownloadChunk:
    start_date: date
    end_date: date
    target_path: Path
    planned_url: str | None = None
    cds_request: dict[str, Any] | None = None
    exists: bool = False
    downloaded: bool = False
    skipped: bool = False
    registered_pipeline_run_id: int | None = None
    error: str | None = None


@dataclass
class ClimateDownloadPlan:
    source: str
    variable: str
    start_date: date
    end_date: date
    dry_run: bool
    configured: bool
    target_path: Path
    planned_url: str | None = None
    product_version: str | None = None
    units: str | None = None
    humidity_approach: str | None = None
    requested_variables: list[str] = field(default_factory=list)
    chunks: list[ClimateDownloadChunk] = field(default_factory=list)
    chunks_planned: int = 0
    chunks_downloaded: int = 0
    chunks_skipped_existing: int = 0
    chunks_failed: int = 0
    manifest_paths: list[Path] = field(default_factory=list)
    message: str = ""


def climate_data_dir() -> Path:
    return Path(os.environ.get("CLIMATE_DATA_DIR", "data/climate"))


def ensure_climate_directories(base_dir: Path | None = None) -> None:
    base = base_dir or climate_data_dir()
    for path in (
        base / "raw" / "chirps",
        base / "raw" / "era5_land",
        base / "processed",
        base / "manifests",
    ):
        path.mkdir(parents=True, exist_ok=True)


def climate_source_status() -> ClimateSourceStatus:
    return ClimateSourceStatus(
        climate_data_dir=climate_data_dir(),
        chirps_base_url_configured=bool(os.environ.get("CHIRPS_BASE_URL", "").strip()),
        era5_provider=_blank_to_none(os.environ.get("ERA5_PROVIDER")),
        era5_api_url_configured=bool(os.environ.get("ERA5_API_URL", "").strip()),
        era5_api_key_configured=bool(os.environ.get("ERA5_API_KEY", "").strip()),
        era5_download_mode=os.environ.get("ERA5_DOWNLOAD_MODE", "manual_or_api"),
        boundary_status=boundary_dependency_status(),
    )


def boundary_dependency_status() -> BoundaryDependencyStatus:
    raw_exists = RAW_BOUNDARY_PATH.exists()
    prepared_exists = PREPARED_BOUNDARY_PATH.exists()
    if prepared_exists:
        message = "Prepared boundary GeoJSON is available for later zonal aggregation."
    elif raw_exists:
        message = (
            "Raw boundary exists but prepared boundary is missing. Run "
            "`flask reference prepare-boundaries` after district_reference is loaded."
        )
    else:
        message = (
            "Raw and prepared boundary files are missing. Place "
            "data/boundaries/UGANDA_DISTRICT.json before climate aggregation work."
        )
    return BoundaryDependencyStatus(raw_exists, prepared_exists, message)


def validate_source_variable(source: str, variable: str) -> tuple[str, str]:
    normalized_source = _normalize_source(source)
    normalized_variable = _normalize_variable(variable)
    if normalized_variable not in SOURCE_VARIABLES[normalized_source]:
        raise ClimateDataError(
            f"Variable {variable!r} is not approved for {normalized_source}. "
            "Approved pairs are CHIRPS/rainfall and ERA5-Land/temperature|humidity."
        )
    return normalized_source, normalized_variable


def validate_climate_file(
    *,
    source: str,
    variable: str,
    path: Path,
    start_date: str | date,
    end_date: str | date,
) -> ClimateFileValidation:
    normalized_source, normalized_variable = validate_source_variable(source, variable)
    start = _parse_date(start_date, label="start_date")
    end = _parse_date(end_date, label="end_date")
    if start > end:
        raise ClimateDataError("start_date must be less than or equal to end_date.")

    candidate = Path(path)
    exists = candidate.exists()
    size = candidate.stat().st_size if exists else 0
    expected_extensions = EXPECTED_EXTENSIONS[normalized_source]
    validation = ClimateFileValidation(
        source=normalized_source,
        variable=normalized_variable,
        path=candidate,
        start_date=start,
        end_date=end,
        exists=exists,
        non_empty=size > 0,
        extension_ok=_extension(candidate) in expected_extensions,
        size_bytes=size,
        product_version=_default_product_version(normalized_source),
        units=_default_units(normalized_source, normalized_variable),
    )
    if not exists:
        validation.errors.append(f"Climate file not found: {candidate}")
    if exists and size <= 0:
        validation.errors.append(f"Climate file is empty: {candidate}")
    if not validation.extension_ok:
        validation.errors.append(
            f"{normalized_source} files must use one of: "
            + ", ".join(expected_extensions)
        )
    if exists and size > 0:
        validation.checksum_sha256 = _sha256(candidate)
        _inspect_optional_metadata(validation)
    return validation


def register_climate_file(
    *,
    source: str,
    variable: str,
    path: Path,
    start_date: str | date,
    end_date: str | date,
    mode: str = "manual",
    product_version: str | None = None,
    units: str | None = None,
    source_url: str | None = None,
    humidity_approach: str | None = None,
    requested_variables: list[str] | None = None,
) -> ClimateRegisterSummary:
    """Register a manually placed climate file in pipeline_runs and manifest JSON."""
    ensure_climate_directories()
    validation = validate_climate_file(
        source=source,
        variable=variable,
        path=path,
        start_date=start_date,
        end_date=end_date,
    )
    if not validation.ok:
        raise ClimateDataError("; ".join(validation.errors))
    if product_version:
        validation.product_version = product_version
    if units:
        validation.units = units

    manifest = _manifest_from_validation(
        validation,
        mode=mode,
        source_url=source_url,
        humidity_approach=humidity_approach,
        requested_variables=requested_variables,
    )
    manifest_path = _write_manifest(manifest)
    run = PipelineRun(
        run_type="climate_ingestion",
        status="success",
        started_at=_utcnow(),
        completed_at=_utcnow(),
        source_name=validation.source,
        source_version_or_url=None,
        parameters={
            "operation": "file_register" if mode == "manual" else "api_download_register",
            "mode": mode,
            "source": validation.source,
            "variable": validation.variable,
            "product_version": validation.product_version,
            "units": validation.units,
            "start_date": validation.start_date.isoformat(),
            "end_date": validation.end_date.isoformat(),
            "file_path": str(validation.path),
            "file_size_bytes": validation.size_bytes,
            "checksum_sha256": validation.checksum_sha256,
            "source_url": source_url,
            "humidity_approach": humidity_approach,
            "requested_variables": requested_variables or [],
        },
        row_count_raw=1,
        row_count_processed=0,
        warning_count=len(validation.warnings),
        manifest=manifest,
    )
    db.session.add(run)
    db.session.commit()
    return ClimateRegisterSummary(
        pipeline_run_id=run.id,
        manifest_path=manifest_path,
        validation=validation,
    )


def validate_raw_climate_files(base_dir: Path | None = None) -> list[Path]:
    """Return raw climate files currently present under data/climate/raw."""
    base = base_dir or climate_data_dir()
    raw_dir = base / "raw"
    if not raw_dir.exists():
        return []
    return sorted(
        path for path in raw_dir.rglob("*") if path.is_file() and path.name != ".gitkeep"
    )


def plan_chirps_download(
    *,
    start_date: str | date,
    end_date: str | date,
    dry_run: bool = True,
) -> ClimateDownloadPlan:
    start = _parse_date(start_date, label="start_date")
    end = _parse_date(end_date, label="end_date")
    if start > end:
        raise ClimateDataError("start_date must be less than or equal to end_date.")
    chunks = _chirps_chunks(start, end)
    target = chunks[0].target_path if chunks else climate_data_dir() / "raw" / "chirps"
    planned_url = chunks[0].planned_url if chunks else None
    configured = True
    message = (
        "Dry-run only; no files downloaded."
        if dry_run
        else "Ready to download CHIRPS daily rainfall files."
    )
    return ClimateDownloadPlan(
        source="CHIRPS",
        variable="rainfall",
        start_date=start,
        end_date=end,
        dry_run=dry_run,
        configured=configured,
        target_path=target,
        planned_url=planned_url,
        product_version=CHIRPS_PRODUCT_VERSION,
        units="mm/day raw; weekly aggregation sums to mm/week",
        requested_variables=["precipitation"],
        chunks=chunks,
        chunks_planned=len(chunks),
        message=message,
    )


def download_chirps(
    *,
    start_date: str | date,
    end_date: str | date,
    dry_run: bool = True,
    force: bool = False,
) -> ClimateDownloadPlan:
    """Download CHIRPS daily rainfall files and register each successful chunk."""
    ensure_climate_directories()
    plan = plan_chirps_download(start_date=start_date, end_date=end_date, dry_run=dry_run)
    if dry_run:
        return plan

    for chunk in plan.chunks:
        chunk.target_path.parent.mkdir(parents=True, exist_ok=True)
        chunk.exists = chunk.target_path.exists() and chunk.target_path.stat().st_size > 0
        if chunk.exists and not force:
            chunk.skipped = True
            plan.chunks_skipped_existing += 1
        else:
            try:
                _download_url(chunk.planned_url, chunk.target_path)
                chunk.downloaded = True
                plan.chunks_downloaded += 1
            except ClimateDataError as exc:
                chunk.error = str(exc)
                plan.chunks_failed += 1
                continue

        try:
            summary = register_climate_file(
                source="CHIRPS",
                variable="rainfall",
                path=chunk.target_path,
                start_date=chunk.start_date,
                end_date=chunk.end_date,
                mode="api",
                product_version=CHIRPS_PRODUCT_VERSION,
                units="mm/day raw; weekly aggregation sums to mm/week",
                source_url=chunk.planned_url,
                requested_variables=["precipitation"],
            )
            chunk.registered_pipeline_run_id = summary.pipeline_run_id
            plan.manifest_paths.append(summary.manifest_path)
        except ClimateDataError as exc:
            chunk.error = str(exc)
            plan.chunks_failed += 1

    if plan.chunks_failed:
        plan.message = (
            f"CHIRPS download completed with {plan.chunks_failed} failed chunk(s). "
            "Failed chunks were not registered."
        )
    else:
        plan.message = "CHIRPS download/register completed."
    return plan


def plan_era5_download(
    *,
    variable: str,
    start_date: str | date,
    end_date: str | date,
    dry_run: bool = True,
) -> ClimateDownloadPlan:
    source, normalized_variable = validate_source_variable("ERA5-Land", variable)
    start = _parse_date(start_date, label="start_date")
    end = _parse_date(end_date, label="end_date")
    if start > end:
        raise ClimateDataError("start_date must be less than or equal to end_date.")
    chunks = _era5_chunks(normalized_variable, start, end)
    target = chunks[0].target_path if chunks else climate_data_dir() / "raw" / "era5_land"
    api_url = _blank_to_none(os.environ.get("ERA5_API_URL"))
    api_key_configured = bool(os.environ.get("ERA5_API_KEY", "").strip())
    provider = _blank_to_none(os.environ.get("ERA5_PROVIDER")) or "CDS"
    configured = provider.upper() == "CDS" and api_key_configured
    requested_variables = list(ERA5_VARIABLES[normalized_variable])
    message = (
        "Dry-run only; no files downloaded."
        if dry_run
        else "ERA5 API key must be configured before real download."
    )
    return ClimateDownloadPlan(
        source=source,
        variable=normalized_variable,
        start_date=start,
        end_date=end,
        dry_run=dry_run,
        configured=configured,
        target_path=target,
        planned_url=api_url or "CDS API default endpoint",
        product_version=ERA5_PRODUCT_VERSION,
        units=_default_units(source, normalized_variable),
        humidity_approach=(
            "derived_from_2m_temperature_and_2m_dewpoint_temperature"
            if normalized_variable == "humidity"
            else None
        ),
        requested_variables=requested_variables,
        chunks=chunks,
        chunks_planned=len(chunks),
        message=message,
    )


def download_era5(
    *,
    variable: str,
    start_date: str | date,
    end_date: str | date,
    dry_run: bool = True,
    force: bool = False,
) -> ClimateDownloadPlan:
    """Download ERA5-Land monthly chunks via CDS API and register each chunk."""
    ensure_climate_directories()
    plan = plan_era5_download(
        variable=variable,
        start_date=start_date,
        end_date=end_date,
        dry_run=dry_run,
    )
    if dry_run:
        return plan
    if not plan.configured:
        raise ClimateDataError(
            "ERA5_API_KEY is required for ERA5-Land API download. If the CDS "
            "licence/terms have not been accepted, accept them in the CDS account "
            "browser page, then rerun the command. No token was printed or stored."
        )

    try:
        import cdsapi  # type: ignore
    except ImportError as exc:
        raise ClimateDataError(
            "The optional cdsapi package is required for ERA5-Land downloads. "
            "Install climate dependencies before real API download."
        ) from exc

    api_url = _blank_to_none(os.environ.get("ERA5_API_URL"))
    api_key = os.environ.get("ERA5_API_KEY", "").strip()
    client_kwargs: dict[str, str] = {"key": api_key}
    if api_url:
        client_kwargs["url"] = api_url
    client = cdsapi.Client(**client_kwargs)

    for chunk in plan.chunks:
        chunk.target_path.parent.mkdir(parents=True, exist_ok=True)
        chunk.exists = chunk.target_path.exists() and chunk.target_path.stat().st_size > 0
        if chunk.exists and not force:
            chunk.skipped = True
            plan.chunks_skipped_existing += 1
        else:
            try:
                client.retrieve(ERA5_LAND_DATASET, chunk.cds_request, str(chunk.target_path))
                if not chunk.target_path.exists() or chunk.target_path.stat().st_size <= 0:
                    raise ClimateDataError(f"ERA5 download produced an empty file: {chunk.target_path}")
                chunk.downloaded = True
                plan.chunks_downloaded += 1
            except Exception as exc:  # CDS raises mixed exception types.
                chunk.error = (
                    "ERA5-Land CDS request failed. Confirm API credentials and accept "
                    "the ERA5-Land licence/terms in the CDS account. "
                    + str(exc)[:500]
                )
                plan.chunks_failed += 1
                continue

        try:
            summary = register_climate_file(
                source="ERA5-Land",
                variable=plan.variable,
                path=chunk.target_path,
                start_date=chunk.start_date,
                end_date=chunk.end_date,
                mode="api",
                product_version=ERA5_PRODUCT_VERSION,
                units=plan.units,
                humidity_approach=plan.humidity_approach,
                requested_variables=plan.requested_variables,
            )
            chunk.registered_pipeline_run_id = summary.pipeline_run_id
            plan.manifest_paths.append(summary.manifest_path)
        except ClimateDataError as exc:
            chunk.error = str(exc)
            plan.chunks_failed += 1

    if plan.chunks_failed:
        plan.message = (
            f"ERA5-Land download completed with {plan.chunks_failed} failed chunk(s). "
            "Failed chunks were not registered."
        )
    else:
        plan.message = "ERA5-Land download/register completed."
    return plan


def required_climate_date_range(
    *,
    analysis_start_year: int = 2020,
    analysis_end_year: int = 2025,
    lookback_weeks: int = 12,
) -> ClimateDateRange:
    """Return the climate coverage needed for current values plus lag lookback."""
    analysis_start_date, _ = iso_week_dates(analysis_start_year, 1)
    final_week = date(analysis_end_year, 12, 28).isocalendar().week
    _, analysis_end_date = iso_week_dates(analysis_end_year, final_week)
    lag_start = analysis_start_date - timedelta(weeks=lookback_weeks)
    return ClimateDateRange(
        analysis_start_week=f"{analysis_start_year}W01",
        analysis_end_week=f"{analysis_end_year}W{final_week:02d}",
        analysis_start_date=analysis_start_date,
        analysis_end_date=analysis_end_date,
        lag_lookback_start_date=lag_start,
        download_start_date=lag_start,
        download_end_date=analysis_end_date,
        lookback_weeks=lookback_weeks,
    )


def _normalize_source(source: str) -> str:
    cleaned = source.strip().lower().replace("_", "-")
    if cleaned == "chirps":
        return "CHIRPS"
    if cleaned in {"era5-land", "era5 land", "era5"}:
        return "ERA5-Land"
    raise ClimateDataError(
        f"Unsupported climate source {source!r}. Approved sources: "
        + ", ".join(APPROVED_SOURCES)
    )


def _normalize_variable(variable: str) -> str:
    cleaned = variable.strip().lower().replace("_", "-")
    if "wind" in cleaned:
        raise ClimateDataError("Wind speed is not an approved climate variable.")
    if cleaned in APPROVED_VARIABLES:
        return cleaned
    raise ClimateDataError(
        f"Unsupported climate variable {variable!r}. Approved variables: "
        + ", ".join(APPROVED_VARIABLES)
    )


def _parse_date(value: str | date, *, label: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ClimateDataError(f"{label} must be an ISO date, e.g. 2025-01-01.") from exc


def _manifest_from_validation(
    validation: ClimateFileValidation,
    *,
    mode: str,
    source_url: str | None = None,
    humidity_approach: str | None = None,
    requested_variables: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "source": validation.source,
        "variable": validation.variable,
        "mode": mode,
        "product_version": validation.product_version,
        "units": validation.units,
        "source_url": source_url,
        "humidity_approach": humidity_approach,
        "requested_variables": requested_variables or [],
        "start_date": validation.start_date.isoformat(),
        "end_date": validation.end_date.isoformat(),
        "file_path": str(validation.path),
        "file_size_bytes": validation.size_bytes,
        "checksum_sha256": validation.checksum_sha256,
        "crs": validation.crs,
        "detected_variables": validation.detected_variables,
        "detected_time_range": validation.detected_time_range,
        "warnings": validation.warnings,
        "registered_at": _utcnow().isoformat(),
        "boundary_dependency": {
            "raw_boundary_exists": boundary_dependency_status().raw_boundary_exists,
            "prepared_boundary_exists": boundary_dependency_status().prepared_boundary_exists,
        },
    }


def _write_manifest(manifest: dict[str, Any]) -> Path:
    manifests_dir = climate_data_dir() / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = (
        f"{timestamp}_{manifest['source'].lower().replace('-', '_')}_"
        f"{manifest['variable']}_{manifest['start_date']}_{manifest['end_date']}.json"
    )
    path = manifests_dir / filename
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chirps_chunks(start: date, end: date) -> list[ClimateDownloadChunk]:
    base_url = (_blank_to_none(os.environ.get("CHIRPS_BASE_URL")) or DEFAULT_CHIRPS_BASE_URL).rstrip("/")
    chunks: list[ClimateDownloadChunk] = []
    current = start
    while current <= end:
        filename = f"chirps-v2.0.{current:%Y.%m.%d}.tif.gz"
        target = climate_data_dir() / "raw" / "chirps" / f"{current:%Y}" / filename
        chunks.append(
            ClimateDownloadChunk(
                start_date=current,
                end_date=current,
                target_path=target,
                planned_url=f"{base_url}/{current:%Y}/{filename}",
                exists=target.exists() and target.stat().st_size > 0 if target.exists() else False,
            )
        )
        current += timedelta(days=1)
    return chunks


def _era5_chunks(variable: str, start: date, end: date) -> list[ClimateDownloadChunk]:
    chunks: list[ClimateDownloadChunk] = []
    current = date(start.year, start.month, 1)
    while current <= end:
        month_start = max(start, current)
        next_month = _next_month(current)
        month_end = min(end, next_month - timedelta(days=1))
        if month_start <= month_end:
            target = (
                climate_data_dir()
                / "raw"
                / "era5_land"
                / variable
                / f"era5_land_{variable}_{month_start:%Y_%m}.nc"
            )
            days = [
                f"{day:02d}"
                for day in range(month_start.day, month_end.day + 1)
            ]
            request = {
                "product_type": "reanalysis",
                "variable": list(ERA5_VARIABLES[variable]),
                "year": f"{month_start.year}",
                "month": f"{month_start.month:02d}",
                "day": days,
                "time": ERA5_HOURLY_TIMES,
                "area": ERA5_UGANDA_AREA,
                "format": "netcdf",
            }
            chunks.append(
                ClimateDownloadChunk(
                    start_date=month_start,
                    end_date=month_end,
                    target_path=target,
                    planned_url=ERA5_LAND_DATASET,
                    cds_request=request,
                    exists=target.exists() and target.stat().st_size > 0 if target.exists() else False,
                )
            )
        current = next_month
    return chunks


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _download_url(url: str | None, target_path: Path) -> None:
    if not url:
        raise ClimateDataError("Download URL was not planned.")
    try:
        response = requests.get(url, stream=True, timeout=120)
    except requests.RequestException as exc:
        raise ClimateDataError(f"Could not download {url}: {exc}") from exc
    if response.status_code != 200:
        raise ClimateDataError(f"Download failed for {url} with HTTP {response.status_code}.")
    with target_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    if target_path.stat().st_size <= 0:
        raise ClimateDataError(f"Downloaded file is empty: {target_path}")


def _extension(path: Path) -> str:
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if len(suffixes) >= 2 and suffixes[-2:] in ([".tif", ".gz"], [".tiff", ".gz"]):
        return "".join(suffixes[-2:])
    return path.suffix.lower()


def _default_product_version(source: str) -> str:
    return CHIRPS_PRODUCT_VERSION if source == "CHIRPS" else ERA5_PRODUCT_VERSION


def _default_units(source: str, variable: str) -> str:
    if source == "CHIRPS":
        return "mm/day raw; weekly aggregation sums to mm/week"
    if variable == "temperature":
        return "Kelvin raw; aggregation converts to Celsius"
    if variable == "humidity":
        return (
            "Kelvin temperature and dewpoint raw; aggregation derives relative "
            "humidity percent"
        )
    return "unknown"


def _inspect_optional_metadata(validation: ClimateFileValidation) -> None:
    """Best-effort metadata inspection when optional climate libraries exist."""
    if validation.source == "CHIRPS":
        try:
            import rasterio  # type: ignore
        except ImportError:
            validation.warnings.append(
                "rasterio is not installed; CRS and raster metadata were not inspected."
            )
            return
        try:
            with rasterio.open(validation.path) as dataset:
                validation.crs = str(dataset.crs) if dataset.crs else None
                validation.detected_variables = [validation.variable]
        except Exception as exc:
            validation.warnings.append(f"rasterio could not inspect file metadata: {exc}")
        return

    try:
        import xarray as xr  # type: ignore
    except ImportError:
        validation.warnings.append(
            "xarray is not installed; NetCDF/GRIB variables, units and time metadata were not inspected."
        )
        return
    try:
        dataset = xr.open_dataset(validation.path)
        try:
            validation.detected_variables = list(dataset.data_vars)
            units = [
                str(dataset[var].attrs.get("units"))
                for var in dataset.data_vars
                if dataset[var].attrs.get("units")
            ]
            if units:
                validation.units = "; ".join(sorted(set(units)))
            if "time" in dataset.coords and dataset.time.size:
                values = dataset.time.values
                validation.detected_time_range = (str(values[0]), str(values[-1]))
            crs = dataset.attrs.get("crs") or dataset.attrs.get("spatial_ref")
            validation.crs = str(crs) if crs else None
        finally:
            dataset.close()
    except Exception as exc:
        validation.warnings.append(f"xarray could not inspect climate file metadata: {exc}")


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
