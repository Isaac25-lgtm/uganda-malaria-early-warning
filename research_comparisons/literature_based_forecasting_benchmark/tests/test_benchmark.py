"""Focused + adversarial lineage guards for the isolated benchmark (Sections L & D)."""
from __future__ import annotations

import types
import unittest
from datetime import date, timedelta
from decimal import Decimal

from app.services.model_validation import PredictionRecord, build_rolling_origin_folds, district_threshold

from research_comparisons.literature_based_forecasting_benchmark import config as C
from research_comparisons.literature_based_forecasting_benchmark.src import (
    bootstrap,
    data,
    davis_exploratory_nb_dl,
    folds,
    katwesige_nb,
    leakage,
    metrics,
    paired_comparison,
    reporting,
    run as run_module,
    tekle_poisson_dl,
    zinszer_arimax,
)

_ROWS = data.load_canonical_rows()
_ERR = data.load_error_keys()
_OBS = data.build_observed_cases(_ROWS, _ERR)
_CLIM = data.build_climate_series(_ROWS)
POOLED = (tekle_poisson_dl, katwesige_nb, davis_exploratory_nb_dl)


def _ctx(obs=None, clim=None):
    return types.SimpleNamespace(observed=obs if obs is not None else _OBS,
                                 climate_series=clim if clim is not None else _CLIM, error_keys=_ERR)


class OutputIsolationTests(unittest.TestCase):
    def test_rejects_canonical_paths(self):
        for rel in ("data/modelling/x.csv", "data/models/y.json", "app/z.py", "tests/t.py"):
            with self.assertRaises(leakage.OutputIsolationError):
                leakage.assert_output_isolated(C.REPO_ROOT / rel)

    def test_accepts_run_dir(self):
        self.assertTrue(leakage.assert_output_isolated(C.RUN_DIR / "sub" / "f.json"))


class FoldTests(unittest.TestCase):
    def test_folds_match_canonical_and_counts(self):
        for horizon, expected in ((4, 1750), (8, 1751), (12, 1750)):
            mine = folds.build_folds(_ROWS, horizon)
            canonical = build_rolling_origin_folds(_ROWS, horizon=horizon, max_folds=C.MAX_FOLDS)
            self.assertEqual([f.origin_date for f in mine], [f.origin_date for f in canonical])
            self.assertEqual(sum(len(f.validation_rows) for f in mine), expected)

    def test_target_after_origin_and_gap(self):
        folds.assert_target_after_origin(_ROWS)


class ThresholdAndWarningTests(unittest.TestCase):
    def test_threshold_uses_canonical_q75(self):
        fold = folds.build_folds(_ROWS, 4)[-1]
        row = fold.validation_rows[0]
        rec = metrics.build_record(row, 4, 123.0, fold.train_rows)
        expected = district_threshold(fold.train_rows, str(row["district_id"]),
                                      target_epi_week=int(row["target_epi_week_h4"]))
        self.assertEqual(rec.threshold, expected)

    def test_warning_is_count_over_threshold(self):
        fold = folds.build_folds(_ROWS, 4)[-1]
        row = fold.validation_rows[0]
        thr = district_threshold(fold.train_rows, str(row["district_id"]),
                                 target_epi_week=int(row["target_epi_week_h4"]))
        self.assertTrue(metrics.build_record(row, 4, float(thr) + 1000.0, fold.train_rows).predicted_surge)
        self.assertFalse(metrics.build_record(row, 4, 0.0, fold.train_rows).predicted_surge)


class LineageInvariantTests(unittest.TestCase):
    """Section D: computed lineage never after origin, all models, h4/h8/h12."""

    def test_pooled_lineage_never_after_origin(self):
        for horizon in C.HORIZONS:
            fold = folds.build_folds(_ROWS, horizon)[-1]
            origin = fold.origin_date.isoformat()
            for module in POOLED:
                for row in fold.validation_rows[:40]:
                    lin = data.row_lineage(str(row["district_id"]), str(row["week_start_date"]),
                                           case_lags=module.CASE_LAGS, climate_lags=module.CLIMATE_LAGS,
                                           observed=_OBS, climate_series=_CLIM)
                    self.assertFalse(lin.violates_origin())
                    mfs = lin.max_feature_source()
                    if mfs is not None:
                        self.assertLess(mfs, origin)  # strictly before origin (origin-week not used)

    def test_stored_lineage_respects_lags(self):
        fold = folds.build_folds(_ROWS, 4)[-1]
        for row in fold.validation_rows[:40]:
            origin = str(row["week_start_date"])
            lin = data.row_lineage(str(row["district_id"]), origin,
                                   case_lags=tekle_poisson_dl.CASE_LAGS, climate_lags=tekle_poisson_dl.CLIMATE_LAGS,
                                   observed=_OBS, climate_series=_CLIM)
            if lin.max_case_source is not None:
                self.assertLessEqual(lin.max_case_source, data.week_minus(origin, 1))  # <= O-1
            if lin.max_climate_source is not None:
                self.assertLessEqual(lin.max_climate_source, data.week_minus(origin, 4))  # <= O-4

    def test_zinszer_lineage_never_after_origin(self):
        fold = folds.build_folds(_ROWS, 4)[-1]
        sub = types.SimpleNamespace(origin_date=fold.origin_date, target_date=fold.target_date,
                                    train_rows=fold.train_rows, trainable_rows=fold.trainable_rows,
                                    validation_rows=fold.validation_rows[:3])
        out = zinszer_arimax.predict_fold(sub, 4, _ctx())
        origin = fold.origin_date.isoformat()
        for pred in out["predictions"]:
            if pred["status"] == "success":
                self.assertFalse(pred["lineage"]["violates_origin"])
                self.assertLess(pred["lineage"]["max_feature_source_date"], origin)


class AdversarialInjectionTests(unittest.TestCase):
    """Section D: a distinctive post-origin value cannot affect features."""

    def _row(self):
        fold = folds.build_folds(_ROWS, 4)[-1]
        return fold.validation_rows[0], fold.origin_date.isoformat()

    def test_post_origin_case_injection_no_effect(self):
        row, origin = self._row()
        district = str(row["district_id"])
        for module in POOLED:
            base = module.features(row, _ctx())
            poisoned = dict(_OBS)
            for ahead in (0, 1, 2, 4):  # origin week and after
                poisoned[(district, data.week_minus(origin, -ahead))] = 10 ** 9
            self.assertEqual(base, module.features(row, _ctx(obs=poisoned)),
                             f"{module.METHOD_ID} consumed an origin/post-origin case value")

    def test_post_origin_climate_injection_no_effect(self):
        row, origin = self._row()
        district = str(row["district_id"])
        for module in POOLED:
            base = module.features(row, _ctx())
            poisoned = {k: dict(v) for k, v in _CLIM.items()}
            for ahead in (0, 1, 2, 3):
                key = (district, data.week_minus(origin, -ahead))
                poisoned[key] = {"rainfall": 10 ** 9, "temperature": 10 ** 9, "humidity": 10 ** 9}
            self.assertEqual(base, module.features(row, _ctx(clim=poisoned)),
                             f"{module.METHOD_ID} consumed an origin/post-origin climate value")

    def test_following_observation_not_used_for_gap(self):
        # observed-only access never interpolates from a following (post-origin) week.
        row, origin = self._row()
        district = str(row["district_id"])
        obs = dict(_OBS)
        obs.pop((district, data.week_minus(origin, 1)), None)  # make O-1 a gap
        obs[(district, data.week_minus(origin, -1))] = 10 ** 9  # observe O+1 distinctively
        value, source = data.observed_case_at(obs, district, origin, 1)
        self.assertIsNone(value)  # gap stays missing; O+1 is never borrowed
        self.assertLess(source, origin)


class ProvenanceTests(unittest.TestCase):
    """Section D: provenance is computed, not hardcoded to origin; violations fail."""

    _payload = {"origin_date": "2024-01-01", "target_date": "2024-01-29", "horizon": 4, "fold_number": 1,
                "training_max_target_date": "2024-01-01", "threshold_history_max_date": "2024-01-01"}

    def test_not_hardcoded_to_origin(self):
        pred = {"district_id": "D", "observed_cases": "5", "predicted_cases": "4", "status": "success",
                "max_feature_source_date": "2023-12-25", "max_case_source_date": "2023-12-25",
                "max_climate_source_date": "2023-12-04", "lineage_violates_origin": False}
        prov = run_module._provenance(self._payload, pred, "tekle_poisson_dl", tekle_poisson_dl, "10")
        self.assertEqual(prov["feature_cutoff_date"], "2023-12-25")
        self.assertNotEqual(prov["feature_cutoff_date"], self._payload["origin_date"])
        self.assertTrue(prov["leakage_check_passed"])

    def test_violation_fails(self):
        pred = {"district_id": "D", "observed_cases": "5", "predicted_cases": "4", "status": "success",
                "max_feature_source_date": "2024-01-08", "max_case_source_date": "2024-01-08",
                "max_climate_source_date": "2023-12-04", "lineage_violates_origin": True}
        prov = run_module._provenance(self._payload, pred, "tekle_poisson_dl", tekle_poisson_dl, "10")
        self.assertFalse(prov["leakage_check_passed"])

    def test_origin_week_feature_source_fails(self):
        pred = {"district_id": "D", "observed_cases": "5", "predicted_cases": "4", "status": "success",
                "max_feature_source_date": "2024-01-01", "max_case_source_date": "2024-01-01",
                "max_climate_source_date": None, "lineage_violates_origin": False}
        prov = run_module._provenance(self._payload, pred, "tekle_poisson_dl", tekle_poisson_dl, "10")
        self.assertFalse(prov["leakage_check_passed"])


class ReportingMetadataTests(unittest.TestCase):
    def test_report_title_uses_active_run_id(self):
        self.assertEqual(
            reporting._report_title(C.RUN_ID),
            f"# Literature-based forecasting benchmark - BENCHMARK REPORT (run {C.RUN_ID})",
        )

    def test_feature_lineage_schema_describes_all_methods(self):
        schema = reporting._feature_lineage_schema(run_module.METHODS)
        self.assertEqual(schema["run_id"], C.RUN_ID)
        self.assertFalse(schema["origin_availability_convention"]["origin_week_inputs_used_as_predictors"])
        methods = {item["method_id"]: item for item in schema["methods"]}
        self.assertEqual(set(methods), set(run_module.METHODS))
        self.assertEqual(methods["tekle_poisson_dl"]["case_feature_family"]["allowed_lags_weeks"], [1, 2, 3])
        self.assertEqual(
            methods["zinszer_arimax"]["climate_feature_families"]["rainfall"]["allowed_source_weeks"],
            "for an h-week forecast, target week minus (h + 1); across forecast steps this ranges from origin minus (h + 1) through origin minus 1",
        )

    def test_existing_provenance_audit_requires_strictly_prior_sources(self):
        valid = [{
            "origin_date": "2024-01-01", "max_case_source_date": "2023-12-25",
            "max_climate_source_date": "2023-12-18", "max_feature_source_date": "2023-12-25",
            "leakage_check_passed": "True", "method_id": "tekle_poisson_dl", "district_id": "D",
        }]
        self.assertTrue(run_module.audit_existing_provenance(valid)["all_passed"])
        invalid = [dict(valid[0], max_feature_source_date="2024-01-01")]
        audit = run_module.audit_existing_provenance(invalid)
        self.assertFalse(audit["all_passed"])
        self.assertEqual(audit["violation_count"], 1)


class PairedBootstrapTests(unittest.TestCase):
    def _rec(self, district, actual, predicted, a_surge, p_surge):
        return PredictionRecord(district_id=district, horizon=4,
                                origin_date=date(2024, 1, 1), target_date=date(2024, 1, 29),
                                actual=Decimal(actual), predicted=Decimal(predicted), threshold=Decimal("50"),
                                actual_surge=a_surge, predicted_surge=p_surge, surge_score=Decimal("0"))

    def test_paired_common_exact(self):
        rf = {("D1", 4, "2024-01-01", "2024-01-29"): self._rec("D1", "100", "90", True, True)}
        ch = {("D1", 4, "2024-01-01", "2024-01-29"): self._rec("D1", "100", "80", True, True),
              ("D2", 4, "2024-01-01", "2024-01-29"): self._rec("D2", "50", "40", False, False)}
        self.assertEqual(paired_comparison.compare(rf, ch, 4)["common_row_count"], 1)

    def test_warning_bootstrap_deterministic(self):
        keys = [("D%d" % i, 4, "2024-01-01", "2024-01-29") for i in range(8)]
        rf = {k: self._rec(k[0], "100", "90", i % 2 == 0, i % 2 == 0) for i, k in enumerate(keys)}
        ch = {k: self._rec(k[0], "100", "80", i % 2 == 0, i % 3 == 0) for i, k in enumerate(keys)}
        a = bootstrap.paired_warning_bootstrap(rf, ch, keys, replicates=200)
        b = bootstrap.paired_warning_bootstrap(rf, ch, keys, replicates=200)
        self.assertEqual(a["ci_diff_challenger_minus_rf"]["f2"], b["ci_diff_challenger_minus_rf"]["f2"])
        self.assertIn("sensitivity", a["ci_diff_challenger_minus_rf"])


class ClassificationTests(unittest.TestCase):
    def test_failed_fit_stays_failed(self):
        fold = folds.build_folds(_ROWS, 4)[-1]
        stub = types.SimpleNamespace(origin_date=fold.origin_date, target_date=fold.target_date,
                                     train_rows=[], trainable_rows=[], validation_rows=fold.validation_rows[:1])
        out = tekle_poisson_dl.predict_fold(stub, 4, _ctx(obs={}, clim={}))
        self.assertEqual(out["status"], "failed")
        self.assertTrue(all(p["status"] == "failed" and p["predicted"] is None for p in out["predictions"]))

    def test_ippolito_excluded_and_davis_not_primary(self):
        self.assertIn("ippolito_zambia", C.EXCLUDED_METHODS)
        self.assertNotIn("davis_exploratory_nb_dl", C.PRIMARY_METHODS)
        self.assertEqual(set(C.PRIMARY_METHODS), {"zinszer_arimax", "tekle_poisson_dl"})

    def test_gate_passes(self):
        self.assertEqual(run_module.validate_gate(), [])


if __name__ == "__main__":
    unittest.main()
