"""District-week climate aggregation foundation.

This module validates the prepared district boundaries, plans aggregation from
registered CHIRPS/ERA5-Land files, writes processed district-week climate rows,
and can apply current-week climate values to existing district_week_features.
It deliberately does not create climate lags, train models or forecast.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import tempfile
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator

from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db
from app.models import DistrictReference, DistrictWeekClimate, DistrictWeekFeature, PipelineRun
from app.services.climate_data import (
    SOURCE_VARIABLES,
    PREPARED_BOUNDARY_PATH,
    climate_data_dir,
    validate_source_variable,
)
from app.services.preprocessing import expected_periods_for_years, iso_week_dates
from app.services.reference_data import EXPECTED_ACTIVE_DISTRICTS, blank_to_none

PROCESSED_CLIMATE_COLUMNS = (
    "district_id",
    "district_name",
    "epi_year",
    "epi_week",
    "week_start_date",
    "week_end_date",
    "rainfall_current",
    "temperature_current",
    "humidity_current",
    "source_rainfall",
    "source_temperature",
    "source_humidity",
    "aggregation_method",
    "created_at",
    "pipeline_run_id",
)
WEEK_CONVENTION = (
    "ISO-8601 Monday-start, shared with DHIS2 preprocessing; provisional pending "
    "final DHIS2 calendar confirmation"
)
DEFAULT_AGGREGATION_METHOD = "zonal_statistics_pending_real_geospatial_inputs"
FINAL_AGGREGATION_METHOD = (
    "area_aware_zonal_statistics_polygon_mean_then_weekly_temporal_aggregation"
)
REQUIRED_GEOSPATIAL_DEPENDENCIES = (
    "geopandas",
    "rasterio",
    "rioxarray",
    "xarray",
    "rasterstats",
    "numpy",
    "pandas",
    "shapely",
)


class ClimateAggregationError(RuntimeError):
    """Raised when climate aggregation cannot continue safely."""


@dataclass
class PreparedBoundaryProof:
    path: Path
    exists: bool
    feature_count: int = 0
    expected_feature_count: int = EXPECTED_ACTIVE_DISTRICTS
    unique_district_id_count: int = 0
    reference_count: int = 0
    geometry_type_counts: dict[str, int] = field(default_factory=dict)
    duplicate_district_ids: list[str] = field(default_factory=list)
    missing_district_ids: list[str] = field(default_factory=list)
    extra_boundary_ids: list[str] = field(default_factory=list)
    crs_note: str = ""
    crs: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exists and not self.errors


@dataclass(frozen=True)
class RegisteredClimateFile:
    pipeline_run_id: int
    source: str
    variable: str
    path: Path
    start_date: date
    end_date: date
    file_exists: bool


@dataclass(frozen=True)
class ZoneGrid:
    district_ids: list[str]
    district_names: dict[str, str]
    zones: Any
    crs: str
    transform: Any
    shape: tuple[int, int]
    window: Any | None = None
    all_touched: bool = False
    missing_with_center_pixels: list[str] = field(default_factory=list)
    fallback_masks: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClimateWeekRow:
    district_id: str
    district_name: str
    epi_year: int
    epi_week: int
    week_start_date: date
    week_end_date: date
    rainfall_current: Decimal | None = None
    temperature_current: Decimal | None = None
    humidity_current: Decimal | None = None
    source_rainfall: str | None = None
    source_temperature: str | None = None
    source_humidity: str | None = None
    aggregation_method: str = DEFAULT_AGGREGATION_METHOD


@dataclass
class ClimateWriteSummary:
    dry_run: bool
    output_path: Path
    pipeline_run_id: int | None
    row_count: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    csv_written: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ClimateApplySummary:
    dry_run: bool
    climate_rows_considered: int = 0
    matched_feature_rows: int = 0
    rows_to_update: int = 0
    rows_updated: int = 0
    unmatched_climate_rows: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ClimateAggregationSummary:
    dry_run: bool
    start_date: date
    end_date: date
    requested_pairs: list[tuple[str, str]]
    output_path: Path
    boundary: PreparedBoundaryProof
    expected_week_count: int
    files_found: list[RegisteredClimateFile] = field(default_factory=list)
    missing_pairs: list[str] = field(default_factory=list)
    missing_dependencies: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pipeline_run_id: int | None = None
    row_count: int = 0
    write_summary: ClimateWriteSummary | None = None
    apply_summary: ClimateApplySummary | None = None
    status: str = "planned"


def kelvin_to_celsius(value: float | Decimal) -> Decimal:
    """Convert Kelvin to Celsius for ERA5-Land temperature outputs."""
    return (Decimal(str(value)) - Decimal("273.15")).quantize(Decimal("0.0001"))


def relative_humidity_percent(
    temperature_celsius: float | Decimal,
    dewpoint_celsius: float | Decimal,
) -> Decimal:
    """Derive RH percent using the August-Roche-Magnus approximation."""
    temperature = float(temperature_celsius)
    dewpoint = float(dewpoint_celsius)
    exponent = (17.625 * dewpoint / (243.04 + dewpoint)) - (
        17.625 * temperature / (243.04 + temperature)
    )
    rh = 100.0 * math.exp(exponent)
    bounded = max(0.0, min(100.0, rh))
    return Decimal(str(bounded)).quantize(Decimal("0.0001"))


def missing_optional_geospatial_dependencies() -> list[str]:
    return [
        dependency
        for dependency in REQUIRED_GEOSPATIAL_DEPENDENCIES
        if importlib.util.find_spec(dependency) is None
    ]


def processed_output_path() -> Path:
    return climate_data_dir() / "processed" / "district_week_climate.csv"


def validate_prepared_boundary_for_climate(
    path: Path = PREPARED_BOUNDARY_PATH,
) -> PreparedBoundaryProof:
    """Validate the prepared district_id-keyed GeoJSON before climate work."""
    summary = PreparedBoundaryProof(path=path, exists=path.exists())
    if not path.exists():
        summary.errors.append(
            f"Prepared boundary GeoJSON not found: {path}. Run "
            "`flask reference prepare-boundaries` after loading district_reference."
        )
        return summary

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        summary.errors.append(f"Prepared boundary GeoJSON is invalid JSON: {exc}")
        return summary
    if payload.get("type") != "FeatureCollection":
        summary.errors.append("Prepared boundary file must be a GeoJSON FeatureCollection.")
        return summary

    features = payload.get("features")
    if not isinstance(features, list):
        summary.errors.append("Prepared boundary FeatureCollection must contain features.")
        return summary
    summary.feature_count = len(features)
    if summary.feature_count != EXPECTED_ACTIVE_DISTRICTS:
        summary.errors.append(
            f"Prepared boundary feature count is {summary.feature_count}, expected "
            f"{EXPECTED_ACTIVE_DISTRICTS}."
        )

    district_ids: list[str] = []
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            summary.errors.append(f"Feature {index} is not a GeoJSON feature object.")
            continue
        props = feature.get("properties")
        if not isinstance(props, dict):
            summary.errors.append(f"Feature {index} is missing properties.")
            continue
        district_id = blank_to_none(props.get("district_id"))
        district_name = blank_to_none(props.get("district_name"))
        if not district_id:
            summary.errors.append(f"Feature {index} is missing properties.district_id.")
        else:
            district_ids.append(district_id)
        if not district_name:
            summary.errors.append(f"Feature {index} is missing properties.district_name.")

        geometry = feature.get("geometry")
        geometry_type = geometry.get("type") if isinstance(geometry, dict) else None
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            summary.errors.append(
                f"Feature {index} ({district_id or 'unknown'}) has non-Polygon geometry."
            )
        else:
            summary.geometry_type_counts[geometry_type] = (
                summary.geometry_type_counts.get(geometry_type, 0) + 1
            )

    counts: dict[str, int] = {}
    for district_id in district_ids:
        counts[district_id] = counts.get(district_id, 0) + 1
    summary.duplicate_district_ids = sorted(
        district_id for district_id, count in counts.items() if count > 1
    )
    summary.unique_district_id_count = len(counts)
    if summary.duplicate_district_ids:
        summary.errors.append(
            "Duplicate district_id values in prepared boundary: "
            + ", ".join(summary.duplicate_district_ids)
        )

    reference_ids = {
        row.district_id for row in DistrictReference.query.filter_by(active=True).all()
    }
    summary.reference_count = len(reference_ids)
    boundary_ids = set(district_ids)
    summary.missing_district_ids = sorted(reference_ids - boundary_ids)
    summary.extra_boundary_ids = sorted(boundary_ids - reference_ids)
    if summary.missing_district_ids:
        summary.errors.append(
            f"{len(summary.missing_district_ids)} active district_reference ID(s) "
            "are missing from prepared boundaries."
        )
    if summary.extra_boundary_ids:
        summary.errors.append(
            f"{len(summary.extra_boundary_ids)} prepared boundary district_id value(s) "
            "are not active in district_reference."
        )

    if payload.get("crs"):
        summary.crs = str(payload.get("crs"))
        summary.crs_note = "CRS metadata is present in the GeoJSON and must be reviewed before final zonal statistics."
    else:
        summary.crs = "EPSG:4326"
        summary.crs_note = (
            "No GeoJSON CRS member found; GeoJSON coordinates are handled as "
            "WGS84/EPSG:4326 under the GeoJSON convention and must still be "
            "checked against raster CRS before final zonal aggregation."
        )
    return summary


def _load_boundary_geodataframe(
    path: Path = PREPARED_BOUNDARY_PATH,
) -> Any:
    """Load the one approved prepared boundary after its canonical validation."""
    proof = validate_prepared_boundary_for_climate(path)
    if not proof.ok:
        raise ClimateAggregationError(
            "Prepared boundary GeoJSON is missing or invalid: "
            + "; ".join(proof.errors)
        )

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ClimateAggregationError(
            "geopandas is required to load prepared district boundaries."
        ) from exc

    try:
        gdf = gpd.read_file(path)
    except Exception as exc:
        raise ClimateAggregationError(
            f"Prepared boundary GeoJSON could not be loaded as geospatial data: {path}."
        ) from exc

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ClimateAggregationError(
            f"Prepared boundary loader did not return a GeoDataFrame: {path}."
        )
    if "district_id" not in gdf.columns:
        raise ClimateAggregationError(
            "Prepared boundary GeoDataFrame is missing the district_id column."
        )
    district_ids = [blank_to_none(value) for value in gdf["district_id"].tolist()]
    if len(gdf) != EXPECTED_ACTIVE_DISTRICTS:
        raise ClimateAggregationError(
            f"Prepared boundary GeoDataFrame has {len(gdf)} features; expected "
            f"{EXPECTED_ACTIVE_DISTRICTS}."
        )
    if any(district_id is None for district_id in district_ids):
        raise ClimateAggregationError(
            "Prepared boundary GeoDataFrame contains blank district_id values."
        )
    unique_district_ids = {district_id for district_id in district_ids if district_id}
    if len(unique_district_ids) != EXPECTED_ACTIVE_DISTRICTS:
        raise ClimateAggregationError(
            "Prepared boundary GeoDataFrame must contain exactly "
            f"{EXPECTED_ACTIVE_DISTRICTS} unique district_id values; found "
            f"{len(unique_district_ids)}."
        )
    invalid_geometry_types = sorted(
        set(gdf.geometry.geom_type.dropna()) - {"Polygon", "MultiPolygon"}
    )
    if gdf.geometry.isna().any() or gdf.geometry.is_empty.any() or invalid_geometry_types:
        detail = (
            ", ".join(invalid_geometry_types)
            if invalid_geometry_types
            else "missing or empty geometry"
        )
        raise ClimateAggregationError(
            "Prepared boundary GeoDataFrame contains unsupported geometry: " + detail
        )

    # RFC 7946 GeoJSON coordinates are WGS84 when no legacy CRS member is present.
    if proof.crs == "EPSG:4326":
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    elif gdf.crs is None:
        raise ClimateAggregationError(
            "Prepared boundary GeoJSON declares CRS metadata that geopandas could not read."
        )
    gdf["district_id"] = district_ids
    return gdf


def aggregate_weekly_climate(
    *,
    start_date: date,
    end_date: date,
    source: str | None = None,
    variable: str | None = None,
    dry_run: bool = False,
    apply_to_features: bool = False,
    output_path: Path | None = None,
) -> ClimateAggregationSummary:
    """Plan or run climate aggregation from registered raw climate files."""
    if start_date > end_date:
        raise ClimateAggregationError("start_date must be <= end_date.")
    pairs = _requested_pairs(source=source, variable=variable)
    boundary = validate_prepared_boundary_for_climate()
    expected_weeks = _weeks_for_date_range(start_date, end_date)
    summary = ClimateAggregationSummary(
        dry_run=dry_run,
        start_date=start_date,
        end_date=end_date,
        requested_pairs=pairs,
        output_path=output_path or processed_output_path(),
        boundary=boundary,
        expected_week_count=len(expected_weeks),
    )

    if not boundary.ok:
        summary.warnings.extend(boundary.errors)
        if dry_run:
            return summary
        raise ClimateAggregationError("; ".join(boundary.errors))

    summary.missing_dependencies = missing_optional_geospatial_dependencies()
    if summary.missing_dependencies:
        summary.warnings.append(
            "Missing required geospatial dependencies for final zonal aggregation: "
            + ", ".join(summary.missing_dependencies)
            + ". Install climate dependencies before writing retained climate values."
        )

    summary.files_found = discover_registered_climate_files(
        pairs=pairs, start_date=start_date, end_date=end_date
    )
    summary.missing_pairs = _missing_coverage_pairs(
        pairs=pairs,
        files=summary.files_found,
        start_date=start_date,
        end_date=end_date,
    )
    if summary.missing_pairs:
        summary.warnings.append(
            "Missing registered climate file coverage for: "
            + ", ".join(summary.missing_pairs)
        )

    if dry_run:
        summary.status = "dry-run"
        return summary

    if summary.missing_dependencies:
        raise ClimateAggregationError(summary.warnings[-1])

    run = PipelineRun(
        run_type="climate_aggregation",
        status="running",
        started_at=_utcnow(),
        source_name="CHIRPS/ERA5-Land registered climate files",
        parameters={
            "sources": sorted({source for source, _ in pairs}),
            "variables": sorted({variable for _, variable in pairs}),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "week_convention": WEEK_CONVENTION,
            "boundary_file": str(PREPARED_BOUNDARY_PATH),
            "boundary_feature_count": boundary.feature_count,
            "output_path": str(summary.output_path),
            "aggregation_method": FINAL_AGGREGATION_METHOD,
            "apply_to_features": apply_to_features,
        },
    )
    db.session.add(run)
    db.session.commit()
    summary.pipeline_run_id = run.id
    staged_output: Path | None = None

    try:
        staged_output = _temporary_output_path(summary.output_path)
        if summary.missing_pairs:
            raise ClimateAggregationError(summary.warnings[-1])
        rows, proof, aggregation_warnings = build_real_climate_week_rows(
            files=summary.files_found,
            start_date=start_date,
            end_date=end_date,
            expected_weeks=expected_weeks,
        )
        summary.warnings.extend(aggregation_warnings)
        write_summary = write_processed_climate_rows(
            rows,
            output_path=staged_output,
            pipeline_run_id=run.id,
            commit=False,
        )
        write_summary.output_path = summary.output_path
        summary.write_summary = write_summary
        summary.row_count = write_summary.row_count
        if apply_to_features:
            apply_start_year, apply_end_year = _feature_year_bounds_for_range(
                start_date=start_date,
                end_date=end_date,
            )
            if apply_start_year is not None and apply_end_year is not None:
                summary.apply_summary = apply_processed_climate_to_features(
                    start_year=apply_start_year,
                    end_year=apply_end_year,
                    pipeline_run_id=run.id,
                    commit=False,
                )
                summary.warnings.extend(summary.apply_summary.warnings)
            else:
                summary.warnings.append(
                    "No district_week_features rows matched the climate date range; "
                    "current climate values were not applied to features."
                )
        run = db.session.get(PipelineRun, run.id)
        if run is None:
            raise ClimateAggregationError(
                "Climate aggregation pipeline run disappeared before completion."
            )
        run.status = "success"
        run.completed_at = _utcnow()
        run.row_count_raw = len(summary.files_found)
        run.row_count_processed = summary.row_count
        run.warning_count = len(summary.warnings)
        run.manifest = {
            "requested_pairs": [f"{src}/{var}" for src, var in pairs],
            "files_found": [_file_manifest(file) for file in summary.files_found],
            "missing_pairs": summary.missing_pairs,
            "missing_dependencies": summary.missing_dependencies,
            "week_convention": WEEK_CONVENTION,
            "aggregation_method": FINAL_AGGREGATION_METHOD,
            "boundary": _boundary_manifest(boundary),
            "crs_and_grid_proof": proof,
            "write_summary": {
                "row_count": write_summary.row_count,
                "rows_inserted": write_summary.rows_inserted,
                "rows_updated": write_summary.rows_updated,
                "csv_written": write_summary.csv_written,
                "output_path": str(write_summary.output_path),
            },
            "apply_summary": (
                {
                    "climate_rows_considered": summary.apply_summary.climate_rows_considered,
                    "matched_feature_rows": summary.apply_summary.matched_feature_rows,
                    "rows_updated": summary.apply_summary.rows_updated,
                    "unmatched_climate_rows": summary.apply_summary.unmatched_climate_rows,
                }
                if summary.apply_summary is not None
                else None
            ),
            "warnings": summary.warnings,
            "no_centroid_fallback": True,
            "no_wind_speed": True,
            "humidity_approach": (
                "relative_humidity_derived_from_2m_temperature_and_"
                "2m_dewpoint_temperature"
            ),
        }
        db.session.flush()
        with _promote_staged_output(staged_output, summary.output_path):
            db.session.commit()
        summary.status = "success"
        return summary
    except Exception as exc:
        db.session.rollback()
        run = db.session.get(PipelineRun, run.id)
        if run is not None:
            run.status = "failed"
            run.completed_at = _utcnow()
            run.row_count_raw = len(summary.files_found)
            run.row_count_processed = summary.row_count
            run.warning_count = len(summary.warnings)
            run.error_message = str(exc)[:1000]
            run.manifest = {
                "requested_pairs": [f"{src}/{var}" for src, var in pairs],
                "files_found": [_file_manifest(file) for file in summary.files_found],
                "missing_pairs": summary.missing_pairs,
                "missing_dependencies": summary.missing_dependencies,
                "week_convention": WEEK_CONVENTION,
                "warnings": summary.warnings,
            }
            db.session.commit()
        summary.status = "failed"
        if isinstance(exc, ClimateAggregationError):
            raise
        if isinstance(exc, SQLAlchemyError):
            raise ClimateAggregationError("Database write failed during climate aggregation.") from exc
        raise
    finally:
        if staged_output is not None:
            staged_output.unlink(missing_ok=True)


def build_real_climate_week_rows(
    *,
    files: list[RegisteredClimateFile],
    start_date: date,
    end_date: date,
    expected_weeks: list[tuple[int, int]],
) -> tuple[list[ClimateWeekRow], dict[str, Any], list[str]]:
    """Build complete district-week climate rows from real CHIRPS and ERA5 files."""
    districts = DistrictReference.query.filter_by(active=True).order_by(
        DistrictReference.district_id
    ).all()
    if len(districts) != EXPECTED_ACTIVE_DISTRICTS:
        raise ClimateAggregationError(
            f"Expected {EXPECTED_ACTIVE_DISTRICTS} active districts before climate "
            f"aggregation, found {len(districts)}."
        )
    district_names = {row.district_id: row.district_name for row in districts}
    gdf = _load_boundary_geodataframe()

    rainfall_values, rainfall_proof, rainfall_warnings = _aggregate_chirps_rainfall(
        files=_latest_registered_files(files, "CHIRPS", "rainfall"),
        gdf=gdf,
        start_date=start_date,
        end_date=end_date,
        expected_weeks=expected_weeks,
    )
    temperature_values, temperature_proof, temperature_warnings = _aggregate_era5_weekly(
        files=_latest_registered_files(files, "ERA5-Land", "temperature"),
        gdf=gdf,
        variable="temperature",
        start_date=start_date,
        end_date=end_date,
        expected_weeks=expected_weeks,
    )
    humidity_values, humidity_proof, humidity_warnings = _aggregate_era5_weekly(
        files=_latest_registered_files(files, "ERA5-Land", "humidity"),
        gdf=gdf,
        variable="humidity",
        start_date=start_date,
        end_date=end_date,
        expected_weeks=expected_weeks,
    )

    rows: list[ClimateWeekRow] = []
    missing_values: list[str] = []
    district_ids = sorted(district_names)
    for epi_year, epi_week in expected_weeks:
        week_start, week_end = iso_week_dates(epi_year, epi_week)
        if week_start < start_date or week_end > end_date:
            continue
        for district_id in district_ids:
            key = (district_id, epi_year, epi_week)
            rainfall = rainfall_values.get(key)
            temperature = temperature_values.get(key)
            humidity = humidity_values.get(key)
            if rainfall is None:
                missing_values.append(f"{district_id} {epi_year}W{epi_week:02d} rainfall")
            if temperature is None:
                missing_values.append(f"{district_id} {epi_year}W{epi_week:02d} temperature")
            if humidity is None:
                missing_values.append(f"{district_id} {epi_year}W{epi_week:02d} humidity")
            rows.append(
                ClimateWeekRow(
                    district_id=district_id,
                    district_name=district_names[district_id],
                    epi_year=epi_year,
                    epi_week=epi_week,
                    week_start_date=week_start,
                    week_end_date=week_end,
                    rainfall_current=rainfall,
                    temperature_current=temperature,
                    humidity_current=humidity,
                    source_rainfall="CHIRPS",
                    source_temperature="ERA5-Land",
                    source_humidity="ERA5-Land",
                    aggregation_method=FINAL_AGGREGATION_METHOD,
                )
            )
    if missing_values:
        preview = ", ".join(missing_values[:20])
        suffix = f"; ... {len(missing_values) - 20} more" if len(missing_values) > 20 else ""
        raise ClimateAggregationError(
            "Climate aggregation produced missing values for complete source coverage: "
            + preview
            + suffix
        )

    proof = {
        "rainfall": rainfall_proof,
        "temperature": temperature_proof,
        "humidity": humidity_proof,
        "expected_weeks": len(expected_weeks),
        "districts": len(district_ids),
        "rows_built": len(rows),
        "crs_verified": True,
    }
    warnings = rainfall_warnings + temperature_warnings + humidity_warnings
    return rows, proof, warnings


def _latest_registered_files(
    files: Iterable[RegisteredClimateFile], source: str, variable: str
) -> list[RegisteredClimateFile]:
    """Select the newest registration for each source date span."""
    latest: dict[tuple[date, date], RegisteredClimateFile] = {}
    for file in files:
        if file.source != source or file.variable != variable or not file.file_exists:
            continue
        key = (file.start_date, file.end_date)
        current = latest.get(key)
        if current is None or file.pipeline_run_id > current.pipeline_run_id:
            latest[key] = file
    return sorted(
        latest.values(),
        key=lambda file: (file.start_date, file.end_date, file.pipeline_run_id),
    )


def _aggregate_chirps_rainfall(
    *,
    files: list[RegisteredClimateFile],
    gdf: Any,
    start_date: date,
    end_date: date,
    expected_weeks: list[tuple[int, int]],
) -> tuple[dict[tuple[str, int, int], Decimal], dict[str, Any], list[str]]:
    """Aggregate daily CHIRPS polygon means into district ISO-week totals."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window, from_bounds

    expected_week_set = set(expected_weeks)
    weekly_totals: dict[tuple[str, int, int], Decimal] = defaultdict(
        lambda: Decimal("0")
    )
    observed_dates: set[date] = set()
    grid_cache: dict[tuple[Any, ...], ZoneGrid] = {}
    raster_crs_values: set[str] = set()
    fallback_districts: set[str] = set()

    for file in files:
        observation_date = file.start_date
        if observation_date < start_date or observation_date > end_date:
            continue
        if file.start_date != file.end_date:
            raise ClimateAggregationError(
                f"CHIRPS registration must represent one daily raster: {file.path}."
            )
        try:
            with rasterio.open(_rasterio_path(file.path)) as src:
                if src.crs is None:
                    raise ClimateAggregationError(
                        f"CHIRPS raster has no CRS metadata: {file.path}."
                    )
                raster_crs = src.crs.to_string()
                raster_crs_values.add(raster_crs)
                aligned = _align_boundaries_to_crs(gdf, src.crs, file.path)
                minx, miny, maxx, maxy = aligned.total_bounds
                window = from_bounds(minx, miny, maxx, maxy, src.transform)
                window = window.round_offsets().round_lengths().intersection(
                    Window(0, 0, src.width, src.height)
                )
                if window.width <= 0 or window.height <= 0:
                    raise ClimateAggregationError(
                        f"Prepared boundaries do not overlap CHIRPS raster: {file.path}."
                    )
                transform = src.window_transform(window)
                shape = (int(window.height), int(window.width))
                signature = (
                    raster_crs,
                    tuple(transform),
                    shape,
                    int(window.col_off),
                    int(window.row_off),
                )
                grid = grid_cache.get(signature)
                if grid is None:
                    grid = _build_zone_grid(
                        gdf=aligned,
                        raster_crs=src.crs,
                        transform=transform,
                        shape=shape,
                        window=window,
                    )
                    grid_cache[signature] = grid
                values = src.read(1, window=window, masked=True)
                array = np.asarray(values.filled(np.nan), dtype="float64")
                array[~np.isfinite(array) | (array < 0)] = np.nan
                district_means = _zonal_means(array, grid)
        except ClimateAggregationError:
            raise
        except Exception as exc:
            raise ClimateAggregationError(
                f"Failed to aggregate CHIRPS raster {file.path}: {exc}"
            ) from exc

        iso = observation_date.isocalendar()
        week = (iso.year, iso.week)
        if week not in expected_week_set:
            continue
        observed_dates.add(observation_date)
        fallback_districts.update(grid.missing_with_center_pixels)
        for district_id, value in district_means.items():
            weekly_totals[(district_id, iso.year, iso.week)] += Decimal(str(value))

    _require_daily_coverage(
        source_label="CHIRPS/rainfall",
        observed_dates=observed_dates,
        start_date=start_date,
        end_date=end_date,
    )
    values = {
        key: value.quantize(Decimal("0.000001"))
        for key, value in weekly_totals.items()
    }
    warnings: list[str] = []
    if fallback_districts:
        warnings.append(
            f"CHIRPS used polygon all-touched masks for {len(fallback_districts)} "
            "district(s) without a raster-cell center; no centroids were used."
        )
    proof = {
        "source_files": len(files),
        "days_aggregated": len(observed_dates),
        "raster_crs": sorted(raster_crs_values),
        "boundary_crs": str(gdf.crs),
        "grid_count": len(grid_cache),
        "polygon_method": "rasterized polygon zones with all-touched polygon fallback",
        "fallback_district_count": len(fallback_districts),
    }
    return values, proof, warnings


def _aggregate_era5_weekly(
    *,
    files: list[RegisteredClimateFile],
    gdf: Any,
    variable: str,
    start_date: date,
    end_date: date,
    expected_weeks: list[tuple[int, int]],
) -> tuple[dict[tuple[str, int, int], Decimal], dict[str, Any], list[str]]:
    """Aggregate hourly ERA5-Land polygon means into district ISO-week means."""
    import numpy as np
    import pandas as pd
    import xarray as xr
    from rasterio.transform import from_origin

    if variable not in {"temperature", "humidity"}:
        raise ClimateAggregationError(f"Unsupported ERA5 aggregation variable: {variable}.")

    expected_week_set = set(expected_weeks)
    weekly_sums: dict[tuple[str, int, int], float] = defaultdict(float)
    weekly_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    observed_dates: set[date] = set()
    grid_cache: dict[tuple[Any, ...], ZoneGrid] = {}
    fallback_districts: set[str] = set()
    observations = 0

    for file in files:
        try:
            with _era5_netcdf_path(file.path) as netcdf_path:
                with xr.open_dataset(netcdf_path, engine="netcdf4") as dataset:
                    time_name = _coordinate_name(dataset, ("valid_time", "time"))
                    latitude_name = _coordinate_name(dataset, ("latitude", "lat"))
                    longitude_name = _coordinate_name(dataset, ("longitude", "lon"))
                    times = pd.to_datetime(dataset[time_name].values)
                    latitudes = np.asarray(dataset[latitude_name].values, dtype="float64")
                    longitudes = np.asarray(dataset[longitude_name].values, dtype="float64")
                    if latitudes.ndim != 1 or longitudes.ndim != 1:
                        raise ClimateAggregationError(
                            f"ERA5 coordinates must be one-dimensional: {file.path}."
                        )
                    if len(latitudes) < 2 or len(longitudes) < 2:
                        raise ClimateAggregationError(
                            f"ERA5 coordinate grid is too small: {file.path}."
                        )
                    x_resolution = abs(float(np.median(np.diff(longitudes))))
                    y_resolution = abs(float(np.median(np.diff(latitudes))))
                    transform = from_origin(
                        float(longitudes.min()) - x_resolution / 2,
                        float(latitudes.max()) + y_resolution / 2,
                        x_resolution,
                        y_resolution,
                    )
                    shape = (len(latitudes), len(longitudes))
                    signature = ("EPSG:4326", tuple(transform), shape)
                    grid = grid_cache.get(signature)
                    if grid is None:
                        grid = _build_zone_grid(
                            gdf=gdf,
                            raster_crs="EPSG:4326",
                            transform=transform,
                            shape=shape,
                        )
                        grid_cache[signature] = grid

                    t2m_name = _data_variable_name(
                        dataset, ("t2m", "2m_temperature", "temperature_2m")
                    )
                    temperature = np.asarray(
                        dataset[t2m_name]
                        .transpose(time_name, latitude_name, longitude_name)
                        .values,
                        dtype="float64",
                    )
                    if latitudes[0] < latitudes[-1]:
                        temperature = temperature[:, ::-1, :]
                    if longitudes[0] > longitudes[-1]:
                        temperature = temperature[:, :, ::-1]

                    if variable == "temperature":
                        stack = temperature - 273.15
                    else:
                        d2m_name = _data_variable_name(
                            dataset,
                            ("d2m", "2m_dewpoint_temperature", "dewpoint_temperature_2m"),
                        )
                        dewpoint = np.asarray(
                            dataset[d2m_name]
                            .transpose(time_name, latitude_name, longitude_name)
                            .values,
                            dtype="float64",
                        )
                        if latitudes[0] < latitudes[-1]:
                            dewpoint = dewpoint[:, ::-1, :]
                        if longitudes[0] > longitudes[-1]:
                            dewpoint = dewpoint[:, :, ::-1]
                        temperature_c = temperature - 273.15
                        dewpoint_c = dewpoint - 273.15
                        stack = 100.0 * np.exp(
                            (17.625 * dewpoint_c / (243.04 + dewpoint_c))
                            - (17.625 * temperature_c / (243.04 + temperature_c))
                        )
                        stack = np.clip(stack, 0.0, 100.0)

                    district_values = _zonal_means_stack(stack, grid)
        except ClimateAggregationError:
            raise
        except Exception as exc:
            raise ClimateAggregationError(
                f"Failed to aggregate ERA5-Land {variable} file {file.path}: {exc}"
            ) from exc

        fallback_districts.update(grid.missing_with_center_pixels)
        for time_index, timestamp in enumerate(times):
            observation_date = timestamp.date()
            if observation_date < start_date or observation_date > end_date:
                continue
            iso = observation_date.isocalendar()
            if (iso.year, iso.week) not in expected_week_set:
                continue
            observed_dates.add(observation_date)
            observations += 1
            for district_index, district_id in enumerate(grid.district_ids):
                value = district_values[time_index, district_index]
                if not np.isfinite(value):
                    continue
                key = (district_id, iso.year, iso.week)
                weekly_sums[key] += float(value)
                weekly_counts[key] += 1

    _require_daily_coverage(
        source_label=f"ERA5-Land/{variable}",
        observed_dates=observed_dates,
        start_date=start_date,
        end_date=end_date,
    )
    values = {
        key: Decimal(str(total / weekly_counts[key])).quantize(Decimal("0.0001"))
        for key, total in weekly_sums.items()
        if weekly_counts[key]
    }
    warnings: list[str] = []
    if fallback_districts:
        warnings.append(
            f"ERA5-Land {variable} used polygon all-touched masks for "
            f"{len(fallback_districts)} district(s) without a raster-cell center; "
            "no centroids were used."
        )
    proof = {
        "source_files": len(files),
        "days_aggregated": len(observed_dates),
        "hourly_observations": observations,
        "raster_crs": "EPSG:4326",
        "boundary_crs": str(gdf.crs),
        "grid_count": len(grid_cache),
        "polygon_method": "rasterized polygon zones with all-touched polygon fallback",
        "fallback_district_count": len(fallback_districts),
        "temporal_method": "mean of hourly district polygon means by ISO week",
    }
    return values, proof, warnings


def _align_boundaries_to_crs(gdf: Any, raster_crs: Any, source_path: Path) -> Any:
    """Return boundaries explicitly aligned to a raster CRS."""
    from pyproj import CRS

    if raster_crs is None:
        raise ClimateAggregationError(f"Raster CRS is missing: {source_path}.")
    if gdf.crs is None:
        raise ClimateAggregationError(
            f"Prepared boundary CRS is missing before reading raster {source_path}."
        )
    try:
        boundary_crs = CRS.from_user_input(gdf.crs)
        target_crs = CRS.from_user_input(raster_crs)
    except Exception as exc:
        raise ClimateAggregationError(
            f"Could not compare boundary and raster CRS for {source_path}."
        ) from exc
    return gdf if boundary_crs == target_crs else gdf.to_crs(target_crs)


def _build_zone_grid(
    *,
    gdf: Any,
    raster_crs: Any,
    transform: Any,
    shape: tuple[int, int],
    window: Any | None = None,
) -> ZoneGrid:
    """Rasterize polygons on an aligned grid and retain polygon-only fallbacks."""
    import numpy as np
    from rasterio.features import geometry_mask, rasterize

    aligned = _align_boundaries_to_crs(gdf, raster_crs, PREPARED_BOUNDARY_PATH)
    aligned = aligned.sort_values("district_id").reset_index(drop=True)
    district_ids = aligned["district_id"].tolist()
    district_names = {
        row.district_id: str(getattr(row, "district_name", row.district_id))
        for row in aligned.itertuples()
    }
    shapes = [
        (geometry, index)
        for index, geometry in enumerate(aligned.geometry, start=1)
    ]
    zones = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    present = set(int(value) for value in np.unique(zones) if value > 0)
    missing = [
        district_id
        for index, district_id in enumerate(district_ids, start=1)
        if index not in present
    ]
    fallback_masks: dict[str, Any] = {}
    for district_id in missing:
        geometry = aligned.loc[aligned["district_id"] == district_id, "geometry"].iloc[0]
        mask = geometry_mask(
            [geometry],
            out_shape=shape,
            transform=transform,
            invert=True,
            all_touched=True,
        )
        if not mask.any():
            raise ClimateAggregationError(
                f"District {district_id} does not overlap the raster grid as a polygon."
            )
        fallback_masks[district_id] = mask
    return ZoneGrid(
        district_ids=district_ids,
        district_names=district_names,
        zones=zones,
        crs=str(aligned.crs),
        transform=transform,
        shape=shape,
        window=window,
        all_touched=bool(missing),
        missing_with_center_pixels=missing,
        fallback_masks=fallback_masks,
    )


def _zonal_means(array: Any, grid: ZoneGrid) -> dict[str, float]:
    import numpy as np

    values: dict[str, float] = {}
    for index, district_id in enumerate(grid.district_ids, start=1):
        mask = grid.fallback_masks.get(district_id)
        if mask is None:
            mask = grid.zones == index
        district_values = array[mask]
        valid = district_values[np.isfinite(district_values)]
        if valid.size:
            values[district_id] = float(valid.mean())
    return values


def _zonal_means_stack(stack: Any, grid: ZoneGrid) -> Any:
    import numpy as np

    output = np.full((stack.shape[0], len(grid.district_ids)), np.nan, dtype="float64")
    for district_index, district_id in enumerate(grid.district_ids):
        mask = grid.fallback_masks.get(district_id)
        if mask is None:
            mask = grid.zones == district_index + 1
        district_values = stack[:, mask]
        finite = np.isfinite(district_values)
        counts = finite.sum(axis=1)
        sums = np.where(finite, district_values, 0.0).sum(axis=1)
        valid_rows = counts > 0
        output[valid_rows, district_index] = sums[valid_rows] / counts[valid_rows]
    return output


def _rasterio_path(path: Path) -> str:
    if path.name.lower().endswith((".tif.gz", ".tiff.gz")):
        return "/vsigzip/" + path.resolve().as_posix()
    return str(path)


@contextmanager
def _era5_netcdf_path(path: Path) -> Iterator[Path]:
    """Yield a NetCDF path, extracting CDS ZIP responses with .nc names."""
    if not zipfile.is_zipfile(path):
        yield path
        return
    with zipfile.ZipFile(path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".nc")
        ]
        if len(members) != 1:
            raise ClimateAggregationError(
                f"ERA5 archive must contain exactly one NetCDF file: {path}."
            )
        member = members[0]
        member_path = Path(member.filename)
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ClimateAggregationError(
                f"ERA5 archive contains an unsafe member path: {path}."
            )
        with tempfile.TemporaryDirectory(prefix="era5_netcdf_") as temp_dir:
            archive.extract(member, temp_dir)
            yield Path(temp_dir) / member_path


def _coordinate_name(dataset: Any, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in dataset.coords:
            return candidate
    raise ClimateAggregationError(
        "Climate dataset is missing a required coordinate; tried "
        + ", ".join(candidates)
        + "."
    )


def _data_variable_name(dataset: Any, candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in dataset.data_vars:
            return candidate
    raise ClimateAggregationError(
        "Climate dataset is missing a required data variable; tried "
        + ", ".join(candidates)
        + "."
    )


def _require_daily_coverage(
    *,
    source_label: str,
    observed_dates: set[date],
    start_date: date,
    end_date: date,
) -> None:
    expected_dates: set[date] = set()
    current = start_date
    while current <= end_date:
        expected_dates.add(current)
        current += timedelta(days=1)
    missing = sorted(expected_dates - observed_dates)
    if missing:
        preview = ", ".join(value.isoformat() for value in missing[:10])
        suffix = f"; ... {len(missing) - 10} more" if len(missing) > 10 else ""
        raise ClimateAggregationError(
            f"{source_label} has no usable raster observations for {len(missing)} "
            f"date(s): {preview}{suffix}."
        )


def _feature_year_bounds_for_range(
    *, start_date: date, end_date: date
) -> tuple[int | None, int | None]:
    years = [
        epi_year
        for epi_year, epi_week in _weeks_for_date_range(start_date, end_date)
        if iso_week_dates(epi_year, epi_week)[0] >= start_date
        and iso_week_dates(epi_year, epi_week)[1] <= end_date
    ]
    return (min(years), max(years)) if years else (None, None)


def _boundary_manifest(boundary: PreparedBoundaryProof) -> dict[str, Any]:
    return {
        "path": str(boundary.path),
        "feature_count": boundary.feature_count,
        "unique_district_id_count": boundary.unique_district_id_count,
        "reference_count": boundary.reference_count,
        "geometry_type_counts": boundary.geometry_type_counts,
        "crs": boundary.crs,
        "crs_note": boundary.crs_note,
    }


def discover_registered_climate_files(
    *,
    pairs: list[tuple[str, str]],
    start_date: date,
    end_date: date,
) -> list[RegisteredClimateFile]:
    """Find successful climate_ingestion registrations overlapping the request."""
    rows = (
        PipelineRun.query.filter_by(run_type="climate_ingestion", status="success")
        .order_by(PipelineRun.id)
        .all()
    )
    pair_set = set(pairs)
    files: list[RegisteredClimateFile] = []
    for run in rows:
        params = run.parameters or {}
        source = params.get("source")
        variable = params.get("variable")
        if not isinstance(source, str) or not isinstance(variable, str):
            continue
        try:
            normalized_pair = validate_source_variable(source, variable)
        except Exception:
            continue
        if normalized_pair not in pair_set:
            continue
        try:
            file_start = date.fromisoformat(str(params.get("start_date")))
            file_end = date.fromisoformat(str(params.get("end_date")))
        except ValueError:
            continue
        if file_end < start_date or file_start > end_date:
            continue
        file_path = Path(str(params.get("file_path", "")))
        files.append(
            RegisteredClimateFile(
                pipeline_run_id=run.id,
                source=normalized_pair[0],
                variable=normalized_pair[1],
                path=file_path,
                start_date=file_start,
                end_date=file_end,
                file_exists=file_path.exists(),
            )
        )
    return files


def write_processed_climate_rows(
    rows: Iterable[ClimateWeekRow],
    *,
    output_path: Path | None = None,
    pipeline_run_id: int | None,
    dry_run: bool = False,
    commit: bool = True,
) -> ClimateWriteSummary:
    """Validate and write processed climate rows to CSV and PostgreSQL."""
    row_list = list(rows)
    target = output_path or processed_output_path()
    summary = ClimateWriteSummary(
        dry_run=dry_run,
        output_path=target,
        pipeline_run_id=pipeline_run_id,
        row_count=len(row_list),
    )
    duplicate_keys = _duplicate_climate_keys(row_list)
    if duplicate_keys:
        raise ClimateAggregationError(
            "Duplicate district-week climate rows: " + ", ".join(duplicate_keys)
        )
    for row in row_list:
        expected_start, expected_end = iso_week_dates(row.epi_year, row.epi_week)
        if row.week_start_date != expected_start or row.week_end_date != expected_end:
            raise ClimateAggregationError(
                f"{row.district_id} {row.epi_year}W{row.epi_week:02d} dates do not "
                "match the shared ISO week helper."
            )
    if dry_run:
        return summary
    if pipeline_run_id is None:
        raise ClimateAggregationError("pipeline_run_id is required for a real write.")

    target.parent.mkdir(parents=True, exist_ok=True)
    now = _utcnow().isoformat()
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROCESSED_CLIMATE_COLUMNS)
        writer.writeheader()
        for row in row_list:
            writer.writerow(_row_to_csv(row, pipeline_run_id=pipeline_run_id, created_at=now))
    summary.csv_written = True

    for row in row_list:
        existing = DistrictWeekClimate.query.filter_by(
            district_id=row.district_id,
            epi_year=row.epi_year,
            epi_week=row.epi_week,
        ).first()
        values = _row_to_model_values(row, pipeline_run_id=pipeline_run_id)
        if existing is None:
            db.session.add(
                DistrictWeekClimate(
                    district_id=row.district_id,
                    epi_year=row.epi_year,
                    epi_week=row.epi_week,
                    **values,
                )
            )
            summary.rows_inserted += 1
        else:
            for field_name, value in values.items():
                setattr(existing, field_name, value)
            summary.rows_updated += 1
    if commit:
        db.session.commit()
    return summary


def apply_processed_climate_to_features(
    *,
    start_year: int,
    end_year: int,
    pipeline_run_id: int | None = None,
    dry_run: bool = False,
    commit: bool = True,
) -> ClimateApplySummary:
    """Apply current climate values to existing district_week_features rows only."""
    if start_year > end_year:
        raise ClimateAggregationError("start_year must be <= end_year.")
    query = DistrictWeekClimate.query.filter(
        DistrictWeekClimate.epi_year >= start_year,
        DistrictWeekClimate.epi_year <= end_year,
    )
    if pipeline_run_id is not None:
        query = query.filter_by(pipeline_run_id=pipeline_run_id)
    climate_rows = query.order_by(
        DistrictWeekClimate.district_id,
        DistrictWeekClimate.epi_year,
        DistrictWeekClimate.epi_week,
    ).all()
    summary = ClimateApplySummary(
        dry_run=dry_run,
        climate_rows_considered=len(climate_rows),
    )
    if not climate_rows:
        summary.warnings.append("No processed district_week_climate rows found to apply.")
        return summary

    for climate_row in climate_rows:
        feature = DistrictWeekFeature.query.filter_by(
            district_id=climate_row.district_id,
            epi_year=climate_row.epi_year,
            epi_week=climate_row.epi_week,
        ).first()
        if feature is None:
            summary.unmatched_climate_rows += 1
            continue
        summary.matched_feature_rows += 1
        needs_update = (
            feature.rainfall_current != climate_row.rainfall_current
            or feature.temperature_current != climate_row.temperature_current
            or feature.humidity_current != climate_row.humidity_current
        )
        if not needs_update:
            continue
        summary.rows_to_update += 1
        if not dry_run:
            feature.rainfall_current = climate_row.rainfall_current
            feature.temperature_current = climate_row.temperature_current
            feature.humidity_current = climate_row.humidity_current
            summary.rows_updated += 1
    if summary.unmatched_climate_rows:
        summary.warnings.append(
            f"{summary.unmatched_climate_rows} climate row(s) had no existing "
            "district_week_features match. No feature rows were created."
        )
    if not dry_run and summary.rows_updated and commit:
        db.session.commit()
    return summary


def _temporary_output_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}.",
        suffix=".staged.csv",
        dir=target.parent,
        delete=False,
    ) as handle:
        return Path(handle.name)


@contextmanager
def _promote_staged_output(staged: Path, target: Path) -> Iterator[None]:
    """Swap in a staged CSV and restore the old file if the DB commit fails."""
    backup: Path | None = None
    if target.exists():
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.stem}.",
            suffix=".backup.csv",
            dir=target.parent,
            delete=False,
        ) as handle:
            backup = Path(handle.name)
        backup.unlink()
        target.replace(backup)
    try:
        staged.replace(target)
        yield
    except Exception:
        target.unlink(missing_ok=True)
        if backup is not None and backup.exists():
            backup.replace(target)
        raise
    else:
        if backup is not None:
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass


def _requested_pairs(
    *, source: str | None, variable: str | None
) -> list[tuple[str, str]]:
    if source and variable:
        return [validate_source_variable(source, variable)]
    if source:
        normalized_source, _ = validate_source_variable(source, SOURCE_VARIABLES[_source_key(source)][0])
        return [(normalized_source, variable) for variable in SOURCE_VARIABLES[normalized_source]]
    if variable:
        variable_text = variable.strip().lower().replace("_", "-")
        if variable_text == "rainfall":
            return [validate_source_variable("CHIRPS", variable)]
        return [validate_source_variable("ERA5-Land", variable)]
    return [
        ("CHIRPS", "rainfall"),
        ("ERA5-Land", "temperature"),
        ("ERA5-Land", "humidity"),
    ]


def _source_key(source: str) -> str:
    normalized, _ = validate_source_variable(source, "rainfall" if source.lower().startswith("chirps") else "temperature")
    return normalized


def _weeks_for_date_range(start_date: date, end_date: date) -> list[tuple[int, int]]:
    start_iso = start_date.isocalendar()
    current_year = start_iso.year
    current_week = start_iso.week
    weeks: list[tuple[int, int]] = []
    while True:
        week_start, week_end = iso_week_dates(current_year, current_week)
        if week_start > end_date:
            break
        weeks.append((current_year, current_week))
        next_day = week_end.fromordinal(week_end.toordinal() + 1)
        current_year = next_day.isocalendar().year
        current_week = next_day.isocalendar().week
    return weeks


def start_end_from_years(
    start_year: int, end_year: int, *, lookback_weeks: int = 0
) -> tuple[date, date]:
    periods = expected_periods_for_years(start_year, end_year)
    first_year = int(periods[0][:4])
    first_week = int(periods[0][5:])
    last_year = int(periods[-1][:4])
    last_week = int(periods[-1][5:])
    first_start, _ = iso_week_dates(first_year, first_week)
    _, last_end = iso_week_dates(last_year, last_week)
    if lookback_weeks:
        first_start = first_start - timedelta(weeks=lookback_weeks)
    return first_start, last_end


def _missing_coverage_pairs(
    *,
    pairs: list[tuple[str, str]],
    files: list[RegisteredClimateFile],
    start_date: date,
    end_date: date,
) -> list[str]:
    missing: list[str] = []
    for source, variable in pairs:
        candidates = [
            file
            for file in files
            if file.source == source and file.variable == variable and file.file_exists
        ]
        if not candidates:
            missing.append(f"{source}/{variable}")
            continue
        if min(file.start_date for file in candidates) > start_date or max(
            file.end_date for file in candidates
        ) < end_date:
            missing.append(f"{source}/{variable} incomplete {start_date} to {end_date}")
    return missing


def _duplicate_climate_keys(rows: list[ClimateWeekRow]) -> list[str]:
    counts: dict[tuple[str, int, int], int] = {}
    for row in rows:
        key = (row.district_id, row.epi_year, row.epi_week)
        counts[key] = counts.get(key, 0) + 1
    return [
        f"{district_id} {epi_year}W{epi_week:02d}"
        for (district_id, epi_year, epi_week), count in counts.items()
        if count > 1
    ]


def _row_to_csv(
    row: ClimateWeekRow, *, pipeline_run_id: int, created_at: str
) -> dict[str, Any]:
    values = {
        "district_id": row.district_id,
        "district_name": row.district_name,
        "epi_year": row.epi_year,
        "epi_week": row.epi_week,
        "week_start_date": row.week_start_date.isoformat(),
        "week_end_date": row.week_end_date.isoformat(),
        "rainfall_current": _decimal_or_blank(row.rainfall_current),
        "temperature_current": _decimal_or_blank(row.temperature_current),
        "humidity_current": _decimal_or_blank(row.humidity_current),
        "source_rainfall": row.source_rainfall or "",
        "source_temperature": row.source_temperature or "",
        "source_humidity": row.source_humidity or "",
        "aggregation_method": row.aggregation_method,
        "created_at": created_at,
        "pipeline_run_id": pipeline_run_id,
    }
    return values


def _row_to_model_values(row: ClimateWeekRow, *, pipeline_run_id: int) -> dict[str, Any]:
    return {
        "week_start_date": row.week_start_date,
        "week_end_date": row.week_end_date,
        "rainfall_current": row.rainfall_current,
        "temperature_current": row.temperature_current,
        "humidity_current": row.humidity_current,
        "source_rainfall": row.source_rainfall,
        "source_temperature": row.source_temperature,
        "source_humidity": row.source_humidity,
        "aggregation_method": row.aggregation_method,
        "pipeline_run_id": pipeline_run_id,
    }


def _decimal_or_blank(value: Decimal | None) -> str:
    return "" if value is None else str(value)


def _file_manifest(file: RegisteredClimateFile) -> dict[str, Any]:
    return {
        "pipeline_run_id": file.pipeline_run_id,
        "source": file.source,
        "variable": file.variable,
        "path": str(file.path),
        "start_date": file.start_date.isoformat(),
        "end_date": file.end_date.isoformat(),
        "file_exists": file.file_exists,
    }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
