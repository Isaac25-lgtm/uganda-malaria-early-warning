"""Reference-data CLI commands."""
from __future__ import annotations

from pathlib import Path

import click
from flask.cli import with_appcontext

from app.services.reference_data import (
    DEFAULT_METADATA_PATH,
    DEFAULT_PREPARED_BOUNDARY_PATH,
    DEFAULT_RAW_BOUNDARY_PATH,
    DEFAULT_UBOS_POPULATION_PATH,
    ReferenceDataError,
    generate_district_reference_csv_from_boundary,
    import_districts,
    import_ubos_population,
    prepare_raw_boundaries,
    validate_boundaries,
    validate_districts,
)


@click.group("reference")
def reference_cli() -> None:
    """District reference and boundary validation commands."""


@reference_cli.command("generate-district-reference-csv")
@click.option(
    "--raw-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_RAW_BOUNDARY_PATH,
    show_default=True,
    help="Raw Uganda district boundary GeoJSON with a District property.",
)
@click.option(
    "--output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/reference/district_reference.csv"),
    show_default=True,
    help="Generated district_reference CSV output path.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite an existing generated CSV after review.",
)
@with_appcontext
def generate_district_reference_csv_command(
    raw_path: Path, output_path: Path, overwrite: bool
) -> None:
    """Generate district_reference.csv from the raw boundary District names."""
    try:
        summary = generate_district_reference_csv_from_boundary(
            raw_path=raw_path,
            output_path=output_path,
            overwrite=overwrite,
        )
    except ReferenceDataError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("District reference CSV generation")
    click.echo(f"  Raw path: {summary.raw_path}")
    click.echo(f"  Output path: {summary.output_path}")
    click.echo(f"  Feature count: {summary.feature_count}")
    click.echo(f"  Expected feature count: {summary.expected_feature_count}")
    click.echo(f"  District name field detected: {summary.district_name_field or 'missing'}")
    click.echo(f"  Valid Polygon/MultiPolygon geometries: {summary.valid_geometry_count}")
    click.echo(f"  Invalid geometries: {summary.invalid_geometry_count}")
    click.echo(f"  Generated rows: {summary.generated_row_count}")
    click.echo(f"  Written: {'yes' if summary.written else 'no'}")

    if summary.duplicate_district_names:
        click.echo("  Duplicate District names:")
        for name in summary.duplicate_district_names:
            click.echo(f"    - {name}")
    if summary.duplicate_district_ids:
        click.echo("  Duplicate generated district_id values:")
        for district_id in summary.duplicate_district_ids:
            click.echo(f"    - {district_id}")
    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")
    if summary.errors:
        click.echo("  Errors:")
        for error in summary.errors:
            click.echo(f"    - {error}")
        raise click.ClickException("District reference CSV generation failed.")


@reference_cli.command("import-districts")
@click.argument("csv_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--allow-mapping-changes",
    is_flag=True,
    help="Allow explicit changes to existing DHIS2 org-unit or UBOS mappings.",
)
@click.option(
    "--deactivate-missing",
    is_flag=True,
    help="Mark active districts absent from this CSV as inactive. Never deletes rows.",
)
@click.option("--dry-run", is_flag=True, help="Validate and summarize without writing.")
@with_appcontext
def import_districts_command(
    csv_path: Path,
    allow_mapping_changes: bool,
    deactivate_missing: bool,
    dry_run: bool,
) -> None:
    """Import curated districts into district_reference."""
    try:
        summary = import_districts(
            csv_path,
            allow_mapping_changes=allow_mapping_changes,
            deactivate_missing=deactivate_missing,
            dry_run=dry_run,
        )
    except ReferenceDataError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("District import summary")
    click.echo(f"  Mode: {'dry-run' if summary.dry_run else 'write'}")
    click.echo(f"  Inserted: {summary.inserted}")
    click.echo(f"  Updated: {summary.updated}")
    click.echo(f"  Skipped: {summary.skipped}")
    click.echo(f"  Deactivated: {summary.deactivated}")

    if summary.mapping_changes:
        click.echo("  Mapping changes:")
        for message in summary.mapping_changes:
            click.echo(f"    - {message}")
        if not allow_mapping_changes:
            click.echo(
                "  Mapping changes were reported but not applied. "
                "Use --allow-mapping-changes only after review."
            )

    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")


@reference_cli.command("validate-districts")
@click.option(
    "--metadata",
    "metadata_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_METADATA_PATH,
    show_default=True,
    help="District frame metadata file.",
)
@with_appcontext
def validate_districts_command(metadata_path: Path) -> None:
    """Validate district_reference and district frame metadata."""
    summary = validate_districts(metadata_path)

    click.echo("District reference validation")
    click.echo(f"  Active districts: {summary.active_count}")
    click.echo(f"  Expected active districts: {summary.expected_active_count}")
    reference_date = summary.metadata.get("reference_date") or "MISSING"
    click.echo(f"  Frame reference date: {reference_date}")

    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")

    if summary.errors:
        click.echo("  Errors:")
        for error in summary.errors:
            click.echo(f"    - {error}")
        raise click.ClickException("District validation failed.")

    click.echo("District validation completed.")


@reference_cli.command("prepare-boundaries")
@click.option(
    "--raw-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_RAW_BOUNDARY_PATH,
    show_default=True,
    help="Raw Uganda district boundary GeoJSON with a District property.",
)
@click.option(
    "--output-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_PREPARED_BOUNDARY_PATH,
    show_default=True,
    help="Prepared GeoJSON output keyed by canonical district_id.",
)
@with_appcontext
def prepare_boundaries_command(raw_path: Path, output_path: Path) -> None:
    """Validate raw boundaries and prepare district_id-keyed GeoJSON."""
    try:
        summary = prepare_raw_boundaries(raw_path=raw_path, output_path=output_path)
    except ReferenceDataError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("Raw boundary preparation")
    click.echo(f"  Raw path: {summary.raw_path}")
    click.echo(f"  Output path: {summary.output_path}")
    click.echo(f"  Feature count: {summary.feature_count}")
    click.echo(f"  Expected feature count: {summary.expected_feature_count}")
    click.echo(f"  District name field detected: {summary.district_name_field or 'missing'}")
    click.echo(f"  Valid Polygon/MultiPolygon geometries: {summary.valid_geometry_count}")
    click.echo(f"  Invalid geometries: {summary.invalid_geometry_count}")
    click.echo(f"  Matched features: {summary.matched_feature_count}")
    click.echo(f"  Prepared GeoJSON written: {'yes' if summary.prepared_written else 'no'}")

    if summary.unmatched_boundary_names:
        click.echo("  Unmatched boundary District names:")
        for name in summary.unmatched_boundary_names[:20]:
            click.echo(f"    - {name}")
        if len(summary.unmatched_boundary_names) > 20:
            click.echo(f"    ... {len(summary.unmatched_boundary_names) - 20} more")
    if summary.reference_names_without_boundary:
        click.echo("  Reference names without raw boundary:")
        for name in summary.reference_names_without_boundary[:20]:
            click.echo(f"    - {name}")
        if len(summary.reference_names_without_boundary) > 20:
            click.echo(f"    ... {len(summary.reference_names_without_boundary) - 20} more")
    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")
    if summary.errors:
        click.echo("  Errors:")
        for error in summary.errors:
            click.echo(f"    - {error}")
        raise click.ClickException("Raw boundary preparation failed.")


@reference_cli.command("import-ubos-population")
@click.argument(
    "csv_path",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_UBOS_POPULATION_PATH,
)
@click.option("--dry-run", is_flag=True, help="Validate without writing rows.")
@with_appcontext
def import_ubos_population_command(csv_path: Path, dry_run: bool) -> None:
    """Import approved UBOS district-year population denominators."""
    try:
        summary = import_ubos_population(csv_path=csv_path, dry_run=dry_run)
    except ReferenceDataError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("UBOS population import")
    click.echo(f"  CSV path: {csv_path}")
    click.echo(f"  Mode: {'dry-run' if dry_run else 'write'}")
    click.echo(f"  Parsed rows: {summary.row_count}")
    click.echo(f"  Years: {', '.join(map(str, summary.years)) or 'none'}")
    click.echo(f"  Inserted: {summary.inserted}")
    click.echo(f"  Updated: {summary.updated}")
    click.echo(f"  Skipped: {summary.skipped}")
    click.echo(f"  Unmatched UBOS names: {len(summary.unmatched_names)}")
    click.echo(
        f"  Active reference districts without population rows: "
        f"{len(summary.districts_without_population)}"
    )

    if summary.unmatched_names:
        click.echo("  Unmatched names:")
        for name in summary.unmatched_names[:20]:
            click.echo(f"    - {name}")
        if len(summary.unmatched_names) > 20:
            click.echo(f"    ... {len(summary.unmatched_names) - 20} more")
    if summary.districts_without_population:
        click.echo("  Districts without population rows:")
        for name in summary.districts_without_population[:20]:
            click.echo(f"    - {name}")
        if len(summary.districts_without_population) > 20:
            click.echo(f"    ... {len(summary.districts_without_population) - 20} more")
    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")
    if summary.errors:
        click.echo("  Errors:")
        for error in summary.errors:
            click.echo(f"    - {error}")
        raise click.ClickException("UBOS population import failed.")


@reference_cli.command("validate-boundaries")
@click.argument("geojson_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--metadata",
    "metadata_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_METADATA_PATH,
    show_default=True,
    help="District frame metadata file.",
)
@with_appcontext
def validate_boundaries_command(geojson_path: Path, metadata_path: Path) -> None:
    """Validate prepared GeoJSON boundaries by district_id."""
    try:
        summary = validate_boundaries(geojson_path, metadata_path)
    except ReferenceDataError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("Boundary validation")
    click.echo(f"  Feature count: {summary.feature_count}")
    click.echo(f"  Active reference districts: {summary.reference_count}")
    click.echo(f"  Expected feature count: {summary.expected_feature_count}")
    click.echo(f"  Missing district_id properties: {summary.missing_property_count}")
    click.echo(f"  Duplicate GeoJSON district_id values: {len(summary.duplicate_boundary_ids)}")
    click.echo(f"  Boundaries without reference match: {len(summary.extra_boundary_ids)}")
    click.echo(f"  Reference districts without boundary: {len(summary.missing_district_ids)}")

    if summary.duplicate_boundary_ids:
        click.echo("  Duplicate IDs:")
        for district_id in summary.duplicate_boundary_ids:
            click.echo(f"    - {district_id}")
    if summary.extra_boundary_ids:
        click.echo("  Boundary IDs not in active district_reference:")
        for district_id in summary.extra_boundary_ids[:20]:
            click.echo(f"    - {district_id}")
        if len(summary.extra_boundary_ids) > 20:
            click.echo(f"    ... {len(summary.extra_boundary_ids) - 20} more")
    if summary.missing_district_ids:
        click.echo("  Active district_reference IDs missing from GeoJSON:")
        for district_id in summary.missing_district_ids[:20]:
            click.echo(f"    - {district_id}")
        if len(summary.missing_district_ids) > 20:
            click.echo(f"    ... {len(summary.missing_district_ids) - 20} more")
    if summary.possible_name_matches:
        click.echo("  Possible name-based mismatch hints:")
        for message in summary.possible_name_matches:
            click.echo(f"    - {message}")

    if summary.warnings:
        click.echo("  Warnings:")
        for warning in summary.warnings:
            click.echo(f"    - {warning}")

    if summary.errors:
        click.echo("  Errors:")
        for error in summary.errors:
            click.echo(f"    - {error}")
        raise click.ClickException("Boundary validation failed.")

    click.echo("Boundary validation completed.")
