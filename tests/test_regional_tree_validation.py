from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from app.services.model_validation import build_rolling_origin_folds, district_threshold
from app.services.models.pooled_tree_validation import POOLED_NUMERIC_FEATURES, PooledTreeConfig
from app.services.models.random_forest import RANDOM_FOREST_CONFIG


def _fast_rf_config() -> PooledTreeConfig:
    """A minimal Random Forest config for pipeline-correctness tests.

    Single candidate, few trees and single-threaded fitting keep the end-to-end
    test fast; it exercises the full regional pipeline, not model quality.
    """
    params = ({"n_estimators": 8, "max_depth": 6, "min_samples_leaf": 2, "max_features": "sqrt"},)
    return PooledTreeConfig(
        model_key="random_forest",
        model_name="Random Forest",
        dependency="sklearn",
        native_importance_method="impurity",
        parameter_candidates=params,
        classifier_parameter_candidates=params,
        controlled_threads=1,
    )
from app.services.models.regional_mapping import (
    EXPECTED_SUBREGION_DISTRICT_COUNTS,
    RegionalMappingError,
    load_regional_mapping,
)
from app.services.models.regional_tree_validation import (
    RegionalTreeValidationError,
    _assemble_regional_with_fallback,
    _read_predictions,
    _relative_reduction,
    _scope_fold,
    build_feasibility_audit,
    experiment_output_paths,
    run_regional_comparison,
    run_regional_pooling_experiment,
)

from decimal import Decimal


# --------------------------------------------------------------------------- #
# Mapping validation tests
# --------------------------------------------------------------------------- #
class RegionalMappingTests(unittest.TestCase):
    def test_reviewed_mapping_matches_exact_audit(self):
        mapping = load_regional_mapping()
        self.assertEqual(len(mapping.districts), 146)
        self.assertEqual(len(mapping.district_ids), 146)
        self.assertEqual(len(mapping.subregion_counts), 15)
        self.assertEqual(mapping.subregion_counts, EXPECTED_SUBREGION_DISTRICT_COUNTS)
        self.assertEqual(mapping.subregion_counts["South Buganda"], 14)
        self.assertEqual(mapping.subregion_counts["Kampala"], 1)
        self.assertTrue(mapping.audit()["expected_matches_reviewed_audit"])

    def test_kampala_is_explicit_national_fallback(self):
        mapping = load_regional_mapping()
        self.assertTrue(mapping.is_single_district_subregion("Kampala"))
        self.assertIn("Kampala", mapping.fallback_subregions())
        self.assertNotIn("Kampala", mapping.poolable_subregions())
        self.assertEqual(len(mapping.poolable_subregions()), 14)

    def test_missing_column_fails(self):
        mapping = self._write(["district_id"], [["UGA_A"]])
        with self.assertRaises(RegionalMappingError):
            load_regional_mapping(mapping)

    def test_duplicate_district_fails(self):
        rows = [["UGA_A", "Acholi"], ["UGA_A", "Acholi"]]
        with self.assertRaises(RegionalMappingError):
            load_regional_mapping(self._write(["district_id", "parent_name"], rows))

    def test_contradictory_subregion_fails(self):
        rows = [["UGA_A", "Acholi"], ["UGA_A", "Lango"]]
        with self.assertRaises(RegionalMappingError):
            load_regional_mapping(self._write(["district_id", "parent_name"], rows))

    def test_wrong_counts_fail(self):
        # Two districts, one subregion -> not the reviewed audit.
        rows = [["UGA_A", "Acholi"], ["UGA_B", "Acholi"]]
        with self.assertRaises(RegionalMappingError):
            load_regional_mapping(self._write(["district_id", "parent_name"], rows))

    def test_unknown_district_vs_frame_fails(self):
        with self.assertRaises(RegionalMappingError):
            load_regional_mapping(approved_district_ids={"UGA_ONLY_ONE"})

    def _write(self, header, rows) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "map.csv"
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return tmp


# --------------------------------------------------------------------------- #
# Fold scoping + feasibility + assembly (fast, no model fitting)
# --------------------------------------------------------------------------- #
class RegionalScopingTests(unittest.TestCase):
    def test_scoped_fold_contains_only_subregion_rows_and_same_dates(self):
        rows = _fixture_rows()
        national = build_rolling_origin_folds(rows, horizon=4, max_folds=1)[0]
        ids = frozenset({"UGA_A1", "UGA_A2"})
        scoped = _scope_fold(national, ids)
        self.assertEqual(scoped.origin_date, national.origin_date)
        self.assertEqual(scoped.target_date, national.target_date)
        self.assertTrue(scoped.train_rows)
        self.assertTrue(all(str(r["district_id"]) in ids for r in scoped.train_rows))
        self.assertTrue(all(str(r["district_id"]) in ids for r in scoped.trainable_rows))
        self.assertTrue(all(str(r["district_id"]) in ids for r in scoped.validation_rows))
        # No other subregion's district leaked in.
        self.assertNotIn("UGA_B1", {str(r["district_id"]) for r in scoped.train_rows})

    def test_district_threshold_is_identical_national_and_scoped(self):
        rows = _fixture_rows()
        national = build_rolling_origin_folds(rows, horizon=4, max_folds=1)[0]
        ids = frozenset({"UGA_A1", "UGA_A2"})
        scoped = _scope_fold(national, ids)
        val = next(r for r in scoped.validation_rows if str(r["district_id"]) == "UGA_A1")
        week = int(val["target_epi_week_h4"])
        national_threshold = district_threshold(national.train_rows, "UGA_A1", target_epi_week=week)
        scoped_threshold = district_threshold(scoped.train_rows, "UGA_A1", target_epi_week=week)
        self.assertEqual(national_threshold, scoped_threshold)

    def test_feasibility_audit_flags_single_district_fallback(self):
        mapping = _fixture_mapping()
        rows = _fixture_rows()
        national_folds = {4: build_rolling_origin_folds(rows, horizon=4, max_folds=2)}
        audit = build_feasibility_audit(mapping, national_folds, (4,))
        kampala_cells = [
            row for row in audit["per_subregion_horizon_fold"] if row["subregion"] == "Kampala"
        ]
        self.assertTrue(kampala_cells)
        self.assertTrue(all(c["decision"] == "national_fallback_single_district" for c in kampala_cells))
        self.assertIn("Kampala", audit["national_fallback_subregions"])

    def test_assembly_uses_national_fallback_and_never_averages(self):
        mapping = _fixture_mapping()
        horizons = (4,)
        national = {
            (4, "1", "UGA_A1", "2024-01-01"): _pred("UGA_A1", "100", scope=None),
            (4, "1", "UGA_K1", "2024-01-01"): _pred("UGA_K1", "200", scope=None),
            (4, "1", "UGA_B1", "2024-01-01"): _pred("UGA_B1", "300", scope=None),
        }
        from app.services.models.regional_tree_validation import SubregionRun

        runs = {
            "Ankole": SubregionRun(
                subregion="Ankole",
                district_ids=("UGA_A1", "UGA_A2"),
                status="published",
                predictions={(4, "1", "UGA_A1", "2024-01-01"): _pred("UGA_A1", "111", scope=None)},
            ),
            # Bugisu produced no predictions -> fallback for its districts.
            "Bugisu": SubregionRun(
                subregion="Bugisu", district_ids=("UGA_B1",), status="failed", predictions={}
            ),
        }
        combined, counts, fallback_log = _assemble_regional_with_fallback(
            mapping=mapping,
            national_predictions=national,
            subregion_runs=runs,
            horizons=horizons,
        )
        by_district = {row["district_id"]: row for row in combined}
        # Ankole district uses the regional prediction (111), not an average.
        self.assertEqual(by_district["UGA_A1"]["predicted_cases"], "111")
        self.assertEqual(by_district["UGA_A1"]["scope_source"], "dhis2_subregion_pooled")
        # Kampala uses national fallback (single district).
        self.assertEqual(by_district["UGA_K1"]["predicted_cases"], "200")
        self.assertEqual(by_district["UGA_K1"]["scope_source"], "national_fallback_single_district")
        # Bugisu district falls back to national (300) because no regional candidate.
        self.assertEqual(by_district["UGA_B1"]["predicted_cases"], "300")
        self.assertEqual(by_district["UGA_B1"]["scope_source"], "national_fallback_no_regional_prediction")
        self.assertEqual(counts.get("dhis2_subregion_pooled"), 1)

    def test_relative_reduction_math_and_label(self):
        self.assertAlmostEqual(_relative_reduction(Decimal("40"), Decimal("30")), 25.0)
        self.assertIsNone(_relative_reduction(Decimal("0"), Decimal("30")))
        self.assertIsNone(_relative_reduction(None, Decimal("30")))
        # Regional worse than national -> negative reduction (never relabeled as accuracy).
        self.assertLess(_relative_reduction(Decimal("30"), Decimal("40")), 0)


# --------------------------------------------------------------------------- #
# End-to-end real fit on a small multi-subregion fixture
# --------------------------------------------------------------------------- #
class RegionalExperimentEndToEndTests(unittest.TestCase):
    def test_experiment_runs_scopes_training_and_matches_comparison_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset.csv"
            mapping_path = root / "mapping.csv"
            _write_fixture(dataset)
            _write_fixture_mapping(mapping_path)
            experiment_root = root / "experiments"

            config = _fast_rf_config()
            summary = run_regional_pooling_experiment(
                config,
                horizons=(4,),
                max_folds=1,
                dataset_path=dataset,
                mapping_path=mapping_path,
                root=experiment_root,
                expected_subregion_counts=FIXTURE_SUBREGION_COUNTS,
            )
            self.assertEqual(summary.status, "success")
            self.assertTrue(summary.published)
            self.assertEqual(sorted(summary.poolable_subregions), ["Ankole", "Bugisu"])
            self.assertEqual(summary.fallback_subregions, ["Kampala"])
            self.assertGreater(summary.scope_source_counts.get("dhis2_subregion_pooled", 0), 0)
            self.assertGreater(
                summary.scope_source_counts.get("national_fallback_single_district", 0), 0
            )

            paths = experiment_output_paths("random_forest", root=experiment_root)
            self.assertTrue(paths["validation_predictions"].exists())
            self.assertTrue(paths["subregion_horizon_metrics"].exists())
            self.assertTrue((experiment_root / "region_mapping.csv").exists())
            self.assertTrue((experiment_root / "feasibility_audit.json").exists())

            # No production artefact was written under data/.
            regional_rows, _ = _read_csv(paths["validation_predictions"])
            self.assertTrue(any(r["scope_source"] == "dhis2_subregion_pooled" for r in regional_rows))
            kampala_rows = [r for r in regional_rows if r["district_id"] == "UGA_K1"]
            self.assertTrue(kampala_rows)
            self.assertTrue(
                all(r["scope_source"] == "national_fallback_single_district" for r in kampala_rows)
            )

            # Matched comparison keys line up exactly with the national baseline.
            comparison = run_regional_comparison(
                config,
                horizons=(4,),
                dataset_path=dataset,
                mapping_path=mapping_path,
                root=experiment_root,
                bootstrap_iterations=50,
                expected_subregion_counts=FIXTURE_SUBREGION_COUNTS,
            )
            self.assertTrue(comparison["comparison_complete"])
            self.assertEqual(comparison["unmatched_national_keys"], 0)
            self.assertEqual(comparison["unmatched_regional_keys"], 0)
            block = comparison["per_horizon"][4]
            self.assertEqual(
                block["districts_total"],
                block["districts_improved"] + block["districts_unchanged"] + block["districts_worsened"],
            )
            self.assertIn("district_cluster_bootstrap", block)

    def test_dry_run_writes_nothing_and_validates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset.csv"
            mapping_path = root / "mapping.csv"
            _write_fixture(dataset)
            _write_fixture_mapping(mapping_path)
            experiment_root = root / "experiments"

            summary = run_regional_pooling_experiment(
                _fast_rf_config(),
                horizons=(4,),
                max_folds=1,
                dry_run=True,
                dataset_path=dataset,
                mapping_path=mapping_path,
                root=experiment_root,
                expected_subregion_counts=FIXTURE_SUBREGION_COUNTS,
            )
            self.assertEqual(summary.status, "dry-run")
            self.assertFalse(summary.published)
            # Dry-run performs no writes.
            self.assertFalse(experiment_output_paths("random_forest", root=experiment_root)["validation_predictions"].exists())
            self.assertFalse((experiment_root / "region_mapping.csv").exists())

    def test_unsupported_horizon_rejected(self):
        with self.assertRaises(RegionalTreeValidationError):
            run_regional_pooling_experiment(RANDOM_FOREST_CONFIG, horizons=(5,), dry_run=True)


# --------------------------------------------------------------------------- #
# Fixtures: two poolable subregions (Ankole: A1,A2; Bugisu: B1,B2) + Kampala (K1)
# --------------------------------------------------------------------------- #
FIXTURE_SUBREGION_COUNTS = {"Ankole": 2, "Bugisu": 2, "Kampala": 1}


def _fixture_mapping():
    tmp = Path(tempfile.mkdtemp()) / "map.csv"
    _write_fixture_mapping(tmp)
    return load_regional_mapping(
        tmp,
        approved_district_ids=_fixture_district_ids(),
        expected_subregion_counts=FIXTURE_SUBREGION_COUNTS,
    )


def _fixture_district_ids() -> set[str]:
    return {"UGA_A1", "UGA_A2", "UGA_B1", "UGA_B2", "UGA_K1"}


def _write_fixture_mapping(path: Path) -> None:
    rows = [
        ("UGA_A1", "Ankole"),
        ("UGA_A2", "Ankole"),
        ("UGA_B1", "Bugisu"),
        ("UGA_B2", "Bugisu"),
        ("UGA_K1", "Kampala"),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["district_id", "district_name", "dhis2_orgunit_id", "dhis2_orgunit_name", "parent_name"])
        for district_id, subregion in rows:
            writer.writerow([district_id, district_id.replace("UGA_", ""), f"OU_{district_id}", district_id, subregion])


# The fixture mapping intentionally differs from the reviewed 146-district audit,
# so load it with a matching approved frame and bypass the national-count audit by
# monkeypatching expectations only inside the loader call path used by the engine.
# Instead of patching, the engine loads mapping with approved_district_ids equal to
# the fixture districts; the reviewed-audit counts still apply, so we relax them for
# tests by pointing the loader at a fixture-sized expectation via a subclassed call.
def _pred(district_id: str, cases: str, *, scope) -> dict[str, str]:
    return {
        "model_name": "Random Forest",
        "district_id": district_id,
        "district_name": district_id,
        "horizon": "4",
        "validation_fold": "1",
        "forecast_origin_week": "2023-12-04",
        "target_week": "2024-01-01",
        "observed_cases": "120",
        "predicted_cases": cases,
        "district_surge_threshold": "100",
        "observed_surge_status": "True",
        "predicted_surge_score": "10",
        "predicted_surge_classification": "True",
        "surge_probability": "0.5",
        "surge_probability_cutoff": "0.5",
        "count_threshold_surge_classification": "True",
        "feature_availability_status": "complete",
        "missing_predictor_count": "0",
        "model_status": "success",
        "selected_parameters": "{}",
        "forecast_clipped_nonnegative": "False",
        "feature_contributions": "",
    }


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def _write_fixture(path: Path) -> None:
    rows = _fixture_rows()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture_rows() -> list[dict[str, str]]:
    start = date(2023, 1, 2)
    districts = (
        ("UGA_A1", "ANKOLE ONE", 0),
        ("UGA_A2", "ANKOLE TWO", 5),
        ("UGA_B1", "BUGISU ONE", 9),
        ("UGA_B2", "BUGISU TWO", 13),
        ("UGA_K1", "KAMPALA ONE", 17),
    )
    weeks = 72
    series = {
        district_id: [
            30 + offset + index // 9 + (index % 7) + int(3 * math.sin(index / 5))
            for index in range(weeks)
        ]
        for district_id, _, offset in districts
    }
    output: list[dict[str, str]] = []
    for district_id, district_name, offset in districts:
        values = series[district_id]
        for index, cases in enumerate(values):
            week = start + timedelta(weeks=index)
            row: dict[str, str] = {
                "district_id": district_id,
                "district_name": district_name,
                "epi_year": str(week.isocalendar().year),
                "epi_week": str(week.isocalendar().week),
                "week_start_date": week.isoformat(),
                "week_end_date": (week + timedelta(days=6)).isoformat(),
                "continuous_week_index": str(week.toordinal() // 7),
                "confirmed_malaria_cases": str(cases),
            }
            for lag in (1, 2, 4, 8, 12):
                row[f"cases_lag_{lag}"] = str(values[index - lag]) if index >= lag else ""
            for window in (4, 8, 12):
                history = values[max(0, index - window):index]
                row[f"cases_rolling_mean_{window}"] = str(sum(history) / len(history)) if history else ""
                row[f"cases_rolling_interpolated_count_{window}"] = "0"
            for lag in (1, 2, 4, 8, 12):
                row[f"cases_lag_{lag}_interpolated"] = "0"
            for climate_index, column in enumerate(
                [name for name in POOLED_NUMERIC_FEATURES if name.startswith(("rainfall_", "temperature_", "humidity_"))],
                start=1,
            ):
                row[column] = str(20 + offset / 4 + climate_index + math.sin(index / (climate_index + 2)))
            for horizon in (4, 8, 12):
                target_index = index + horizon
                row[f"target_week_start_date_h{horizon}"] = (week + timedelta(weeks=horizon)).isoformat()
                row[f"target_epi_week_h{horizon}"] = str((week + timedelta(weeks=horizon)).isocalendar().week)
                row[f"target_cases_h{horizon}"] = str(values[target_index]) if target_index < weeks else ""
            output.append(row)
    output.sort(key=lambda row: (row["district_id"], row["week_start_date"]))
    return output


if __name__ == "__main__":
    unittest.main()
