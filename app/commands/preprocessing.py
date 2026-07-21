"""DHIS2 raw-to-district-week preprocessing CLI commands."""
from __future__ import annotations

import click
from flask.cli import with_appcontext

from app.services.preprocessing import (
    PreprocessingError,
    build_district_week_features,
    build_selection,
)
from app.services.dhis2_mapping import week_alignment_is_verified


@click.group("preprocessing")
def preprocessing_cli() -> None:
    """Build cleaned district-week features from raw DHIS2 records."""


@preprocessing_cli.command("build-district-week")
@click.option(
    "--pipeline-run-id",
    type=int,
    default=None,
    help="Use one successful DHIS2 extraction pipeline run.",
)
@click.option(
    "--latest-successful-dhis2-run",
    is_flag=True,
    help="Use the latest successful DHIS2 extraction run.",
)
@click.option("--start-year", type=int, default=None, help="Start epi year.")
@click.option("--end-year", type=int, default=None, help="End epi year.")
@click.option("--year", type=int, default=None, help="Build one epi year only.")
@click.option("--dry-run", is_flag=True, help="Validate and summarize without writing.")
@click.option(
    "--allow-many-to-one",
    is_flag=True,
    help="Allow approved multiple DHIS2 org units to aggregate to one district_id.",
)
@click.option(
    "--replace-existing",
    is_flag=True,
    help="Overwrite district-week rows from another source pipeline run.",
)
@click.option(
    "--require-full-frame",
    is_flag=True,
    help="Fail unless district_reference has the full expected active district frame.",
)
@with_appcontext
def build_district_week_command(
    pipeline_run_id: int | None,
    latest_successful_dhis2_run: bool,
    start_year: int | None,
    end_year: int | None,
    year: int | None,
    dry_run: bool,
    allow_many_to_one: bool,
    replace_existing: bool,
    require_full_frame: bool,
) -> None:
    """Convert raw_dhis2_malaria rows into district_week_features."""
    try:
        verified, status = week_alignment_is_verified()
        if not verified:
            raise PreprocessingError(
                "Week convention is not verified in district_frame_metadata.yml "
                f"(status={status!r}). Run `flask dhis2 verify-week-period` before "
                "building district_week_features."
            )
        selection = build_selection(
            pipeline_run_id=pipeline_run_id,
            latest_successful_dhis2_run=latest_successful_dhis2_run,
            start_year=start_year,
            end_year=end_year,
            year=year,
        )
        summary = build_district_week_features(
            selection,
            dry_run=dry_run,
            allow_many_to_one=allow_many_to_one,
            replace_existing=replace_existing,
            require_full_frame=require_full_frame,
        )
    except PreprocessingError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("DHIS2 preprocessing to district-week panel")
    if dry_run:
        click.echo("  Mode: dry-run (no preprocessing pipeline_run and no features written)")
        click.echo(f"  Rows that would be inserted/updated: {summary.inserted}")
    else:
        click.echo(f"  preprocessing_pipeline_run_id: {summary.preprocessing_run_id}")
        click.echo(f"  Inserted: {summary.inserted}")
        click.echo(f"  Updated: {summary.updated}")
        click.echo(f"  Skipped existing rows: {summary.skipped}")
    click.echo(f"  Source DHIS2 run IDs: {', '.join(map(str, summary.source_run_ids))}")
    click.echo(f"  Expected weeks in denominator: {summary.expected_period_count}")
    click.echo(f"  Raw rows considered: {summary.raw_rows_considered}")
    click.echo(f"  Raw rows mapped: {summary.raw_rows_mapped}")
    click.echo(f"  Raw rows unmapped/excluded: {summary.raw_rows_unmapped}")
    click.echo(f"  Districts processed: {len(summary.districts_processed)}")

    if summary.unmapped_org_units:
        click.echo("  Unmapped org units excluded:")
        for orgunit_id in sorted(summary.unmapped_org_units)[:20]:
            click.echo(f"    - {orgunit_id}")
        if len(summary.unmapped_org_units) > 20:
            click.echo(f"    ... {len(summary.unmapped_org_units) - 20} more")

    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")
