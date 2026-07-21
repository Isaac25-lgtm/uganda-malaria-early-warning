"""Climate data ingestion CLI commands."""
from __future__ import annotations

from pathlib import Path

import click
from flask.cli import with_appcontext

from app.services.climate_aggregation import (
    ClimateAggregationError,
    aggregate_weekly_climate,
    start_end_from_years,
)
from app.services.climate_data import (
    ClimateDataError,
    climate_source_status,
    download_chirps,
    download_era5,
    ensure_climate_directories,
    plan_chirps_download,
    plan_era5_download,
    register_climate_file,
    required_climate_date_range,
    validate_climate_file,
    validate_raw_climate_files,
)
from app.services.climate_lags import ClimateLagError, create_climate_lags


@click.group("climate")
def climate_cli() -> None:
    """Climate source setup, file registration and download planning."""


@climate_cli.command("sources")
@with_appcontext
def sources_command() -> None:
    """Print approved climate sources and local configuration status."""
    ensure_climate_directories()
    status = climate_source_status()
    click.echo("Climate data sources")
    click.echo("  Approved sources:")
    click.echo("    - CHIRPS: rainfall")
    click.echo("    - ERA5-Land: temperature, humidity")
    click.echo(f"  Climate data dir: {status.climate_data_dir}")
    click.echo(f"  CHIRPS_BASE_URL configured: {'yes' if status.chirps_base_url_configured else 'no'}")
    click.echo(f"  ERA5_PROVIDER: {status.era5_provider or 'not configured'}")
    click.echo(f"  ERA5_API_URL configured: {'yes' if status.era5_api_url_configured else 'no'}")
    click.echo(f"  ERA5_API_KEY configured: {'yes' if status.era5_api_key_configured else 'no'}")
    click.echo(f"  ERA5_DOWNLOAD_MODE: {status.era5_download_mode}")
    click.echo(f"  Raw boundary exists: {'yes' if status.boundary_status.raw_boundary_exists else 'no'}")
    click.echo(
        f"  Prepared boundary exists: "
        f"{'yes' if status.boundary_status.prepared_boundary_exists else 'no'}"
    )
    click.echo(f"  Boundary note: {status.boundary_status.message}")
    click.echo("  No wind-speed source or variable is approved.")


@climate_cli.command("required-range")
@click.option("--start-year", type=int, default=2020, show_default=True)
@click.option("--end-year", type=int, default=2025, show_default=True)
@click.option("--lookback-weeks", type=int, default=12, show_default=True)
@with_appcontext
def required_range_command(start_year: int, end_year: int, lookback_weeks: int) -> None:
    """Print required climate coverage including lag lookback."""
    coverage = required_climate_date_range(
        analysis_start_year=start_year,
        analysis_end_year=end_year,
        lookback_weeks=lookback_weeks,
    )
    click.echo("Required climate date coverage")
    click.echo(f"  Analysis start week: {coverage.analysis_start_week}")
    click.echo(f"  Analysis end week: {coverage.analysis_end_week}")
    click.echo(f"  Analysis start date: {coverage.analysis_start_date}")
    click.echo(f"  Analysis end date: {coverage.analysis_end_date}")
    click.echo(f"  Lag lookback weeks: {coverage.lookback_weeks}")
    click.echo(f"  Lag lookback start date: {coverage.lag_lookback_start_date}")
    click.echo(f"  Download start date: {coverage.download_start_date}")
    click.echo(f"  Download end date: {coverage.download_end_date}")


@climate_cli.command("validate-files")
@click.option(
    "--path",
    "path",
    type=click.Path(exists=False, dir_okay=True, path_type=Path),
    default=None,
    help="Optional file or directory to inspect; defaults to data/climate/raw.",
)
@with_appcontext
def validate_files_command(path: Path | None) -> None:
    """List raw climate files currently present for manual registration."""
    ensure_climate_directories()
    target = path
    if target is None:
        files = validate_raw_climate_files()
        click.echo("Raw climate file scan")
        click.echo(f"  Files found: {len(files)}")
        for file_path in files[:50]:
            click.echo(f"    - {file_path}")
        if len(files) > 50:
            click.echo(f"    ... {len(files) - 50} more")
        return

    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.is_file())
        click.echo(f"Directory scan: {target}")
        click.echo(f"  Files found: {len(files)}")
        for file_path in files[:50]:
            click.echo(f"    - {file_path}")
        return

    click.echo(f"File: {target}")
    click.echo(f"  Exists: {'yes' if target.exists() else 'no'}")
    click.echo(f"  Non-empty: {'yes' if target.exists() and target.stat().st_size > 0 else 'no'}")
    click.echo(f"  Extension: {target.suffix or 'none'}")


@climate_cli.command("register-file")
@click.option("--source", required=True, help="CHIRPS or ERA5-Land.")
@click.option("--variable", required=True, help="rainfall, temperature or humidity.")
@click.option(
    "--path",
    "file_path",
    required=True,
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    help="Local raw climate file path.",
)
@click.option("--start-date", required=True, help="Inclusive source date, YYYY-MM-DD.")
@click.option("--end-date", required=True, help="Inclusive source date, YYYY-MM-DD.")
@click.option("--dry-run", is_flag=True, help="Validate only; do not write manifest or pipeline run.")
@with_appcontext
def register_file_command(
    source: str,
    variable: str,
    file_path: Path,
    start_date: str,
    end_date: str,
    dry_run: bool,
) -> None:
    """Register a manually placed raw climate file."""
    try:
        if dry_run:
            validation = validate_climate_file(
                source=source,
                variable=variable,
                path=file_path,
                start_date=start_date,
                end_date=end_date,
            )
            if not validation.ok:
                raise ClimateDataError("; ".join(validation.errors))
            summary = None
        else:
            summary = register_climate_file(
                source=source,
                variable=variable,
                path=file_path,
                start_date=start_date,
                end_date=end_date,
            )
            validation = summary.validation
    except ClimateDataError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("Climate file registration")
    click.echo(f"  Mode: {'dry-run' if dry_run else 'write'}")
    click.echo(f"  Source: {validation.source}")
    click.echo(f"  Variable: {validation.variable}")
    click.echo(f"  File: {validation.path}")
    click.echo(f"  Date range: {validation.start_date} to {validation.end_date}")
    click.echo(f"  Size bytes: {validation.size_bytes}")
    click.echo(f"  SHA256: {validation.checksum_sha256 or 'not computed'}")
    if summary is not None:
        click.echo(f"  pipeline_run_id: {summary.pipeline_run_id}")
        click.echo(f"  Manifest path: {summary.manifest_path}")


@climate_cli.command("download-chirps")
@click.option("--start-date", required=True, help="Inclusive source date, YYYY-MM-DD.")
@click.option("--end-date", required=True, help="Inclusive source date, YYYY-MM-DD.")
@click.option("--dry-run", is_flag=True, help="Plan only; do not download.")
@click.option("--force", is_flag=True, help="Redownload chunks that already exist.")
@with_appcontext
def download_chirps_command(
    start_date: str, end_date: str, dry_run: bool, force: bool
) -> None:
    """Download/register CHIRPS rainfall files or plan the request."""
    try:
        if dry_run:
            plan = plan_chirps_download(
                start_date=start_date, end_date=end_date, dry_run=True
            )
        else:
            plan = download_chirps(
                start_date=start_date,
                end_date=end_date,
                dry_run=False,
                force=force,
            )
    except ClimateDataError as exc:
        raise click.ClickException(str(exc)) from exc
    _print_download_plan(plan)


@climate_cli.command("download-era5")
@click.option("--variable", required=True, help="temperature or humidity.")
@click.option("--start-date", required=True, help="Inclusive source date, YYYY-MM-DD.")
@click.option("--end-date", required=True, help="Inclusive source date, YYYY-MM-DD.")
@click.option("--dry-run", is_flag=True, help="Plan only; do not download.")
@click.option("--force", is_flag=True, help="Redownload chunks that already exist.")
@with_appcontext
def download_era5_command(
    variable: str, start_date: str, end_date: str, dry_run: bool, force: bool
) -> None:
    """Download/register ERA5-Land files or plan the request."""
    try:
        if dry_run:
            plan = plan_era5_download(
                variable=variable,
                start_date=start_date,
                end_date=end_date,
                dry_run=True,
            )
        else:
            plan = download_era5(
                variable=variable,
                start_date=start_date,
                end_date=end_date,
                dry_run=False,
                force=force,
            )
    except ClimateDataError as exc:
        raise click.ClickException(str(exc)) from exc
    _print_download_plan(plan)


def _print_download_plan(plan) -> None:
    click.echo("Climate download plan")
    click.echo(f"  Source: {plan.source}")
    click.echo(f"  Variable: {plan.variable}")
    click.echo(f"  Mode: {'dry-run' if plan.dry_run else 'download'}")
    click.echo(f"  Date range: {plan.start_date} to {plan.end_date}")
    click.echo(f"  Product/version: {plan.product_version or 'not provided'}")
    click.echo(f"  Units: {plan.units or 'not provided'}")
    if plan.humidity_approach:
        click.echo(f"  Humidity approach: {plan.humidity_approach}")
    if plan.requested_variables:
        click.echo(f"  Requested source variables: {', '.join(plan.requested_variables)}")
    click.echo(f"  Configured for API/download: {'yes' if plan.configured else 'no'}")
    click.echo(f"  Planned URL/base endpoint: {plan.planned_url or 'not configured'}")
    click.echo(f"  Target path: {plan.target_path}")
    click.echo(f"  Chunks planned: {plan.chunks_planned}")
    click.echo(f"  Chunks downloaded: {plan.chunks_downloaded}")
    click.echo(f"  Chunks skipped because already present: {plan.chunks_skipped_existing}")
    click.echo(f"  Chunks failed: {plan.chunks_failed}")
    if plan.chunks:
        click.echo("  First chunks:")
        for chunk in plan.chunks[:5]:
            click.echo(
                f"    - {chunk.start_date} to {chunk.end_date}: {chunk.target_path}"
            )
        if len(plan.chunks) > 5:
            click.echo(f"    ... {len(plan.chunks) - 5} more")
    if plan.manifest_paths:
        click.echo("  Manifests:")
        for manifest_path in plan.manifest_paths[:10]:
            click.echo(f"    - {manifest_path}")
        if len(plan.manifest_paths) > 10:
            click.echo(f"    ... {len(plan.manifest_paths) - 10} more")
    click.echo(f"  Message: {plan.message}")
    if plan.dry_run:
        click.echo("  No files were downloaded.")


@climate_cli.command("aggregate-weekly")
@click.option("--start-date", default=None, help="Inclusive ISO date, YYYY-MM-DD.")
@click.option("--end-date", default=None, help="Inclusive ISO date, YYYY-MM-DD.")
@click.option("--start-year", type=int, default=None, help="Start ISO epi year.")
@click.option("--end-year", type=int, default=None, help="End ISO epi year.")
@click.option("--source", default=None, help="Optional source: CHIRPS or ERA5-Land.")
@click.option("--variable", default=None, help="Optional variable: rainfall, temperature or humidity.")
@click.option("--dry-run", is_flag=True, help="Validate and plan without writing output.")
@click.option(
    "--apply-to-features",
    is_flag=True,
    help="After a real aggregation, update only current climate fields on existing district_week_features rows.",
)
@with_appcontext
def aggregate_weekly_command(
    start_date: str | None,
    end_date: str | None,
    start_year: int | None,
    end_year: int | None,
    source: str | None,
    variable: str | None,
    dry_run: bool,
    apply_to_features: bool,
) -> None:
    """Plan or run district-week climate aggregation."""
    try:
        selected_start, selected_end = _resolve_aggregation_dates(
            start_date=start_date,
            end_date=end_date,
            start_year=start_year,
            end_year=end_year,
        )
        summary = aggregate_weekly_climate(
            start_date=selected_start,
            end_date=selected_end,
            source=source,
            variable=variable,
            dry_run=dry_run,
            apply_to_features=apply_to_features,
        )
    except (ClimateDataError, ClimateAggregationError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    _print_aggregation_summary(summary)


@climate_cli.command("create-lags")
@click.option("--start-year", type=int, required=True, help="Start ISO epi year.")
@click.option("--end-year", type=int, required=True, help="End ISO epi year.")
@click.option(
    "--lags",
    default="4,8,12",
    show_default=True,
    help="Comma-separated approved lags.",
)
@click.option(
    "--variables",
    default="rainfall,temperature,humidity",
    show_default=True,
    help="Comma-separated approved variables.",
)
@click.option("--replace-existing", is_flag=True, help="Replace existing non-null lag values.")
@click.option("--dry-run", is_flag=True, help="Plan lag updates without writing.")
@with_appcontext
def create_lags_command(
    start_year: int,
    end_year: int,
    lags: str,
    variables: str,
    replace_existing: bool,
    dry_run: bool,
) -> None:
    """Create 4/8/12 week climate lag fields on district_week_features."""
    try:
        selected_lags = tuple(int(item.strip()) for item in lags.split(",") if item.strip())
        selected_variables = tuple(item.strip() for item in variables.split(",") if item.strip())
        summary = create_climate_lags(
            start_year=start_year,
            end_year=end_year,
            variables=selected_variables,
            lags=selected_lags,
            replace_existing=replace_existing,
            dry_run=dry_run,
        )
    except (ClimateLagError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("Climate lag creation")
    click.echo(f"  Mode: {'dry-run' if summary.dry_run else 'write'}")
    if summary.pipeline_run_id is not None:
        click.echo(f"  pipeline_run_id: {summary.pipeline_run_id}")
    click.echo(f"  Date years: {summary.start_year} to {summary.end_year}")
    click.echo(f"  Variables: {', '.join(summary.variables)}")
    click.echo(f"  Lags: {', '.join(str(lag) for lag in summary.lags)}")
    click.echo(f"  district_week_features rows considered: {summary.feature_rows_considered}")
    click.echo(f"  district_week_climate rows available: {summary.climate_rows_available}")
    click.echo(f"  Rows with planned/actual updates: {summary.rows_with_updates}")
    click.echo("  Lag field coverage:")
    for field_name, field_summary in sorted(summary.field_summaries.items()):
        click.echo(
            f"    - {field_name}: non-null source={field_summary.non_null_count}, "
            f"null source={field_summary.null_count}, updates={field_summary.planned_updates}"
        )
    if summary.alignment_checks:
        click.echo("  Alignment samples:")
        for check in summary.alignment_checks:
            click.echo(
                f"    - {check.note}: {check.district_id} "
                f"{check.target_epi_year}W{check.target_epi_week:02d} "
                f"({check.target_week_start_date}) lag {check.lag_weeks} -> "
                f"{check.source_epi_year}W{check.source_epi_week:02d} "
                f"({check.source_week_start_date}); value {check.source_value}"
            )
    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")


def _resolve_aggregation_dates(
    *,
    start_date: str | None,
    end_date: str | None,
    start_year: int | None,
    end_year: int | None,
):
    using_dates = start_date is not None or end_date is not None
    using_years = start_year is not None or end_year is not None
    if using_dates and using_years:
        raise ClimateAggregationError("Use either date range or year range, not both.")
    if using_dates:
        if not start_date or not end_date:
            raise ClimateAggregationError("--start-date and --end-date are required together.")
        from datetime import date

        return date.fromisoformat(start_date), date.fromisoformat(end_date)
    if using_years:
        if start_year is None:
            raise ClimateAggregationError("--start-year is required with --end-year.")
        selected_end_year = end_year or start_year
        return start_end_from_years(start_year, selected_end_year, lookback_weeks=12)
    raise ClimateAggregationError("Use --start-date/--end-date or --start-year/--end-year.")


def _print_aggregation_summary(summary) -> None:
    click.echo("Climate weekly aggregation")
    click.echo(f"  Mode: {'dry-run' if summary.dry_run else 'write'}")
    click.echo(f"  Status: {summary.status}")
    click.echo(f"  Date range: {summary.start_date} to {summary.end_date}")
    click.echo(
        "  Requested source/variables: "
        + ", ".join(f"{source}/{variable}" for source, variable in summary.requested_pairs)
    )
    click.echo(f"  Week convention: ISO-8601 Monday-start shared helper")
    click.echo(f"  Expected weeks: {summary.expected_week_count}")
    click.echo(f"  Output path: {summary.output_path}")
    if summary.pipeline_run_id is not None:
        click.echo(f"  pipeline_run_id: {summary.pipeline_run_id}")
    click.echo("  Boundary proof:")
    click.echo(f"    Prepared boundary exists: {'yes' if summary.boundary.exists else 'no'}")
    click.echo(f"    Feature count: {summary.boundary.feature_count}")
    click.echo(f"    Unique district_id values: {summary.boundary.unique_district_id_count}")
    click.echo(f"    Active reference districts: {summary.boundary.reference_count}")
    click.echo(
        "    Geometry types: "
        + (", ".join(f"{k}={v}" for k, v in summary.boundary.geometry_type_counts.items()) or "none")
    )
    click.echo(f"    CRS note: {summary.boundary.crs_note or 'not available'}")
    click.echo("  Climate coverage:")
    click.echo(f"    Registered files found: {len(summary.files_found)}")
    for file in summary.files_found[:20]:
        click.echo(
            f"      - run {file.pipeline_run_id}: {file.source}/{file.variable} "
            f"{file.start_date} to {file.end_date}, exists={'yes' if file.file_exists else 'no'}"
        )
    if len(summary.files_found) > 20:
        click.echo(f"      ... {len(summary.files_found) - 20} more")
    click.echo(
        "    Missing coverage: "
        + (", ".join(summary.missing_pairs) if summary.missing_pairs else "none")
    )
    click.echo(
        "    Missing geospatial dependencies: "
        + (", ".join(summary.missing_dependencies) if summary.missing_dependencies else "none")
    )
    click.echo(f"  District-week rows written: {summary.row_count}")
    if summary.apply_summary is not None:
        click.echo(f"  Applied to district_week_features: {summary.apply_summary.rows_updated}")
    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")
