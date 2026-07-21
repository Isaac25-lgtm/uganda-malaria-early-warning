"""Dashboard data build commands."""
from __future__ import annotations

import click
from flask.cli import with_appcontext

from app.services.dashboard_data import (
    DashboardDataError,
    build_dashboard_data,
    build_web_map_geojson,
)


@click.group("dashboard")
def dashboard_cli() -> None:
    """Dashboard-ready backend data commands."""


@dashboard_cli.command("build-data")
@click.option("--latest", is_flag=True, help="Use latest available forecast issue week.")
@click.option("--horizon", type=click.Choice(["4", "8", "12"]), default="4", show_default=True)
@click.option("--dry-run", is_flag=True)
@with_appcontext
def build_data_command(latest: bool, horizon: str, dry_run: bool) -> None:
    """Build dashboard JSON artifacts from validated forecast outputs."""
    try:
        summary = build_dashboard_data(latest=latest, horizon=int(horizon), dry_run=dry_run)
    except DashboardDataError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Dashboard data build")
    click.echo(f"  Mode: {'dry-run' if dry_run else 'write'}")
    click.echo(f"  Status: {summary.status}")
    click.echo(f"  Horizon: {summary.horizon} weeks")
    click.echo(f"  Risk counts: {summary.risk_counts}")
    click.echo(f"  No-data districts: {len(summary.no_data_districts)}")
    click.echo(f"  Proof artifact: {summary.proof_path}")
    if summary.generated_paths:
        click.echo("  Generated outputs:")
        for path in summary.generated_paths:
            click.echo(f"    - {path}")


@dashboard_cli.command("build-web-map")
@click.option("--dry-run", is_flag=True, help="Verify only; do not write the artifact.")
@click.option(
    "--tolerance",
    type=float,
    default=None,
    help="Simplification tolerance in degrees (default: service setting).",
)
def build_web_map_command(dry_run: bool, tolerance: float | None) -> None:
    """Derive a browser-sized district boundary from the approved prepared GeoJSON.

    The approved boundary is read only and never modified. The derived artifact
    keeps district_id so the map still joins on the canonical key.
    """
    kwargs = {"dry_run": dry_run}
    if tolerance is not None:
        kwargs["tolerance"] = tolerance
    try:
        summary = build_web_map_geojson(**kwargs)
    except DashboardDataError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("Dashboard web-map derivation")
    click.echo(f"  Mode: {'dry-run' if dry_run else 'write'}")
    click.echo(f"  Status: {summary['status']}")
    click.echo(f"  Source: {summary['source_path']} ({summary['source_size_bytes'] / 1_000_000:.1f} MB)")
    click.echo(f"  Output: {summary['destination_path']} ({summary['output_size_mb']} MB)")
    click.echo(f"  Within {summary['max_allowed_bytes'] / 1_000_000:.0f} MB budget: {summary['within_size_budget']}")
    click.echo(f"  Features: {summary['feature_count']}")
    click.echo(f"  Unique district_id values: {summary['unique_district_ids']}")
    click.echo(f"  Geometry types: {summary['geometry_types']}")
    click.echo(f"  Invalid geometries: {len(summary['invalid_geometries'])}")
    click.echo(f"  Approved boundary modified: {summary['approved_boundary_modified']}")
