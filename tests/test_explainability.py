"""Phase 29B explainability + system-status API tests.

Covers: allowlist/traversal safety, malformed/oversized/NaN artifacts, cache
hit/invalidation/mutation isolation, path/secret scrubbing, authentication and
expiry, the importance contracts (181 native / 28 permutation, fold handling,
overall exclusion, normalization-before-limit, signed permutation), family
mapping, grouped semantics, invalid-parameter 400s (never clamped), threshold
reuse, accuracy policy, pipeline write=False, and absence of DB writes.
"""
from __future__ import annotations

import io
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from app import create_app
from app.extensions import db
from app.services import explainability as ex

RF_IMPORTANCE = Path("data/models/random_forest_feature_importance.csv")


class ScrubAndFamilyTests(unittest.TestCase):
    """Pure logic; no app context or database needed."""

    def test_scrub_removes_paths_secrets_urls_and_known_path_keys(self):
        payload = {
            "artifact_paths": {"a": "C:/secret/file.json"},
            "proof_path": "data/x.json",
            "password": "hunter2",
            "database_url": "postgresql://u:p@host/db",
            "nested": {"token": "abc", "keep": "value", "url": "https://x.y/z"},
            "winpath": "C:\\\\Users\\\\me\\\\model.joblib",
            "safe_number": 42,
        }
        cleaned = ex.scrub(payload)
        self.assertNotIn("artifact_paths", cleaned)
        self.assertNotIn("proof_path", cleaned)
        self.assertNotIn("password", cleaned)
        self.assertNotIn("database_url", cleaned)
        self.assertNotIn("token", cleaned["nested"])
        self.assertEqual(cleaned["nested"]["keep"], "value")
        self.assertEqual(cleaned["nested"]["url"], "[redacted]")
        self.assertEqual(cleaned["winpath"], "[redacted]")
        self.assertEqual(cleaned["safe_number"], 42)

    def test_family_mapping_covers_all_known_shapes(self):
        cases = {
            "numeric__cases_rolling_mean_8": ex.FAMILY_MALARIA,
            "numeric__cases_lag_4": ex.FAMILY_MALARIA,
            "numeric__missingindicator_cases_lag_1": ex.FAMILY_MISSING,
            "numeric__temperature_lag_12": ex.FAMILY_CLIMATE,
            "numeric__rainfall_lag_4": ex.FAMILY_CLIMATE,
            "numeric__epi_week": ex.FAMILY_TIME,
            "numeric__continuous_week_index": ex.FAMILY_TIME,
            "district__district_id_UGA_YUMBE": ex.FAMILY_DISTRICT,
            "district_id": ex.FAMILY_DISTRICT,
        }
        for name, family in cases.items():
            self.assertEqual(ex.feature_family(name), family, name)

    def test_missing_indicator_beats_cases_family(self):
        # The indicator name contains 'cases' but must map to Missing-Value.
        self.assertEqual(
            ex.feature_family("numeric__missingindicator_cases_rolling_mean_8"),
            ex.FAMILY_MISSING,
        )

    def test_unknown_feature_falls_to_other(self):
        self.assertEqual(ex.feature_family("numeric__mystery_signal"), ex.FAMILY_OTHER)


class ArtifactSafetyTests(unittest.TestCase):
    def setUp(self):
        ex._CACHE.clear()

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ex.ExplainabilityError):
            ex.load_json("../../etc/passwd")
        with self.assertRaises(ex.ExplainabilityError):
            ex.load_json("not_in_allowlist")

    def test_allowlist_has_no_traversal_and_only_data_paths(self):
        for key, path in ex.ALLOWLIST.items():
            self.assertNotIn("..", str(path), key)
            self.assertTrue(str(path).replace("\\", "/").startswith("data/"), key)

    def test_malformed_json_raises_controlled_error(self):
        with patch.object(ex, "_read_text", return_value="{not json"):
            with self.assertRaises(ex.ExplainabilityError):
                ex._load("selection", ex._parse_json)

    def test_non_utf8_raises_controlled_error(self):
        with patch("pathlib.Path.read_bytes", return_value=b"\xff\xfe\x00bad"):
            with patch("pathlib.Path.stat") as st:
                st.return_value.st_size = 4
                with self.assertRaises(ex.ExplainabilityError):
                    ex._read_text(RF_IMPORTANCE)

    def test_oversized_artifact_is_refused(self):
        with patch("pathlib.Path.stat") as st:
            st.return_value.st_size = ex.MAX_ARTIFACT_BYTES + 1
            with self.assertRaises(ex.ExplainabilityError):
                ex._read_text(RF_IMPORTANCE)

    def test_csv_with_bad_columns_is_rejected(self):
        with self.assertRaises(ex.ExplainabilityError):
            ex._parse_importance_csv("a,b,c\n1,2,3\n")

    def test_csv_with_non_finite_number_is_rejected(self):
        bad = "model_name,horizon,validation_fold,importance_method,feature_name,importance\n" \
              "random_forest,4,1,impurity,f,inf\n"
        with self.assertRaises(ex.ExplainabilityError):
            ex._parse_importance_csv(bad)

    def test_cache_hit_and_mtime_size_invalidation(self):
        ex._CACHE.clear()
        first = ex.load_json("selection")
        self.assertIn("selection", ex._CACHE)
        # A hit returns an equal but distinct object (deep-copied).
        second = ex.load_json("selection")
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        # Changing the signature forces a reload.
        old = ex._CACHE["selection"]
        ex._CACHE["selection"] = ex._CacheEntry(signature=(0.0, 1), value={"stale": True})
        reloaded = ex.load_json("selection")
        self.assertNotEqual(reloaded, {"stale": True})

    def test_cached_value_mutation_does_not_poison_cache(self):
        ex._CACHE.clear()
        first = ex.load_json("selection")
        first["injected"] = "tamper"
        second = ex.load_json("selection")
        self.assertNotIn("injected", second)


class ImportanceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.ctx = cls.app.app_context()
        cls.ctx.push()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.pop()

    def test_native_contract_181_features_and_overall_excluded(self):
        for model in ("random_forest", "xgboost"):
            built = ex.build_feature_importance(
                model=model, horizon=4, method="native", limit=20, grouped=False
            )["data"]
            self.assertEqual(built["total_features"], 181)
            self.assertEqual(built["fold_count_used"], 12)
            self.assertTrue(built["overall_rows_validated"])
            self.assertEqual(built["native_method"], "impurity" if model == "random_forest" else "gain")

    def test_permutation_contract_28_predictors_preserves_signed_values(self):
        built = ex.build_feature_importance(
            model="random_forest", horizon=4, method="permutation", limit=20, grouped=False
        )["data"]
        self.assertEqual(built["total_features"], 28)
        self.assertFalse(built["negatives_clamped"])
        self.assertTrue(built["features"])
        for feature in built["features"]:
            self.assertIn("signed_mean_mae_change", feature)
            self.assertIsInstance(feature["signed_mean_mae_change"], (int, float))

    def test_normalization_is_over_all_features_before_limit(self):
        full = ex.native_importance("random_forest", 4, 181, False)["data"]["features"]
        self.assertAlmostEqual(sum(f["normalized_percentage"] for f in full), 100.0, places=3)
        limited = ex.build_feature_importance(
            model="random_forest", horizon=4, method="native", limit=5, grouped=False
        )["data"]
        # Top-5 percentages match the all-feature normalization (not renormalized to 5).
        self.assertLess(sum(f["normalized_percentage"] for f in limited["features"]), 100.0)
        self.assertEqual(limited["features"][0]["normalized_percentage"], full[0]["normalized_percentage"])

    def test_overall_equals_mean_of_numbered_folds(self):
        rows = ex.load_importance_rows("random_forest")
        means, coverage, validated = ex._aggregate_importance(rows, "random_forest", 4, "impurity")
        self.assertTrue(validated)
        self.assertTrue(all(c == 12 for c in coverage.values()))

    def test_zero_importance_flag_is_exact_and_all_features_retained(self):
        # Exact (non-tautological): all 181 features are present, and the
        # is_zero_importance flag equals raw==0 for every one of them.
        built = ex.native_importance("random_forest", 4, 181, False)["data"]
        self.assertEqual(len(built["features"]), 181)
        for f in built["features"]:
            self.assertEqual(f["is_zero_importance"], f["raw_mean_importance"] == 0.0, f["feature_name"])

    def test_grouped_limit_one_returns_single_family(self):
        built = ex.build_feature_importance(
            model="random_forest", horizon=4, method="native", limit=1, grouped=True
        )["data"]
        self.assertEqual(len(built["families"]), 1)
        self.assertEqual(built["families"][0]["family_rank"], 1)
        self.assertNotIn("features", built)

    def test_grouped_families_have_rank_and_fold_coverage(self):
        built = ex.build_feature_importance(
            model="xgboost", horizon=8, method="native", limit=6, grouped=True
        )["data"]
        for index, fam in enumerate(built["families"]):
            self.assertEqual(fam["family_rank"], index + 1)
            self.assertEqual(fam["fold_coverage"], 12)

    def test_required_readable_labels(self):
        cases = {
            "numeric__cases_rolling_mean_8": "8-week rolling mean of malaria cases",
            "numeric__temperature_lag_12": "Temperature 12 weeks earlier",
            "district__district_id_UGA_YUMBE": "District identity: Yumbe",
            "numeric__missingindicator_cases_lag_4": "Missing-value indicator: malaria cases 4 weeks earlier",
        }
        for name, label in cases.items():
            self.assertEqual(ex.readable_feature_label(name), label, name)

    def test_native_notes_mention_splittable_and_model_specific(self):
        notes = " ".join(ex.native_importance("random_forest", 4, 20, False)["data"]["method_notes"]).lower()
        self.assertIn("splittable", notes)
        self.assertIn("model-specific", notes)

    def test_permutation_notes_explain_negative(self):
        notes = " ".join(
            ex.permutation_importance("random_forest", 4, 20, False)["data"]["method_notes"]
        ).lower()
        self.assertIn("did not worsen", notes)

    def test_grouped_true_omits_individual_features(self):
        built = ex.build_feature_importance(
            model="xgboost", horizon=8, method="native", limit=20, grouped=True
        )["data"]
        self.assertIn("families", built)
        self.assertNotIn("features", built)

    def test_grouped_false_returns_features_and_group_summary(self):
        built = ex.build_feature_importance(
            model="xgboost", horizon=8, method="native", limit=20, grouped=False
        )["data"]
        self.assertIn("features", built)
        self.assertIn("group_summary", built)

    def test_invalid_parameters_raise_not_clamp(self):
        with self.assertRaises(ex.ExplainabilityParameterError):
            ex.build_feature_importance(model="baseline", horizon=4, method="native", limit=20, grouped=False)
        with self.assertRaises(ex.ExplainabilityParameterError):
            ex.build_feature_importance(model="sarimax", horizon=4, method="native", limit=20, grouped=False)
        with self.assertRaises(ex.ExplainabilityParameterError):
            ex.build_feature_importance(model="random_forest", horizon=5, method="native", limit=20, grouped=False)
        with self.assertRaises(ex.ExplainabilityParameterError):
            ex.build_feature_importance(model="random_forest", horizon=4, method="native", limit=99, grouped=False)
        with self.assertRaises(ex.ExplainabilityParameterError):
            ex.build_feature_importance(model="random_forest", horizon=4, method="bogus", limit=20, grouped=False)

    def test_limitations_state_scope_and_non_causal(self):
        built = ex.build_feature_importance(
            model="random_forest", horizon=4, method="native", limit=20, grouped=False
        )
        text = " ".join(built["limitations"]).lower()
        self.assertIn("national_pooled", text)
        self.assertIn("district_specific=false", text)
        self.assertIn("production_prediction_explanation=false", text)
        self.assertIn("not causation", text)


class ExplainabilityEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.ctx = cls.app.app_context()
        cls.ctx.push()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.pop()

    def test_model_selection_all_horizons_exact_models_no_recalc(self):
        expected_count = {4: "Random Forest", 8: "Random Forest", 12: "Random Forest"}
        for horizon in (4, 8, 12):
            built = ex.build_model_selection(horizon)["data"]
            self.assertEqual(built["selected_count_model"], expected_count[horizon])
            self.assertEqual(built["selected_surge_model"], "Random Forest")
            self.assertFalse(built["winner_recalculated"])
            self.assertEqual(len(built["all_models_compared"]), 4)

    def test_accuracy_is_secondary_and_not_used_for_selection(self):
        built = ex.build_model_selection(4)["data"]
        self.assertIn("not used", built["accuracy_policy"].lower())

    def test_validation_method_states_twelve_folds_and_rules(self):
        built = ex.build_validation_method()["data"]
        self.assertEqual(built["numbered_folds"], 12)
        self.assertFalse(built["random_split"])
        self.assertTrue(built["training_before_validation"])
        self.assertEqual(built["sarimax_scope"], "district_specific")
        self.assertEqual(built["tree_model_scope"], "national_pooled")

    def test_threshold_reuses_published_values_no_probability(self):
        from app.services.dashboard_data import resolve_params
        params = resolve_params(horizon=4, district_id="UGA_PADER")
        built = ex.build_threshold_method(params)["data"]
        self.assertFalse(built["is_probability"])
        self.assertFalse(built["confirms_outbreak"])
        self.assertIn("district", built)
        self.assertTrue(built["district"]["values_reused_from_published_forecast"])
        self.assertIsNone(built["district"]["surge_probability"])

    def test_threshold_without_district_has_no_district_block(self):
        from app.services.dashboard_data import resolve_params
        params = resolve_params(horizon=4)
        built = ex.build_threshold_method(params)["data"]
        self.assertNotIn("district", built)

    def test_overview_lists_available_and_unavailable_methods(self):
        built = ex.build_overview(4)["data"]
        self.assertEqual(built["selected_count_model"], "Random Forest")
        self.assertEqual(built["selected_surge_model"], "Random Forest")
        methods = {m["method"] for m in built["explainability_methods_available"]}
        self.assertEqual(methods, {"native", "permutation"})
        unavailable = {m["method"] for m in built["explainability_methods_unavailable"]}
        self.assertIn("production_per_prediction_contributions", unavailable)

    def test_pipeline_status_uses_write_false_and_hides_raw_state(self):
        with patch("app.services.project_state.build_project_state_summary") as mock:
            mock.return_value = {
                "components": {"visible_dashboard": {"status": "complete", "row_counts": {}},
                               "models_and_forecasts": {"status": "complete", "row_counts": {}}},
                "warnings": [],
                "recommended_next_action": "C:/secret/path should not leak",
            }
            built = ex.build_pipeline_status()
            mock.assert_called_once_with(write=False)
        data = built["data"]
        self.assertEqual(data["active_districts"], 146)
        self.assertNotIn("recommended_next_action", json.dumps(data))
        self.assertNotIn("C:/secret", json.dumps(built))

    def test_no_production_contribution_endpoint_is_exposed(self):
        rules = {r.rule for r in self.app.url_map.iter_rules()}
        for rule in rules:
            self.assertNotIn("contribution", rule.lower())
            self.assertNotIn("shap", rule.lower())


class SchemaAndContractTests(unittest.TestCase):
    """Schema validation, NaN/Infinity, and CSV contract violations (all 503)."""

    def setUp(self):
        ex._CACHE.clear()

    def test_every_allowlisted_json_artifact_loads_and_validates(self):
        json_keys = [k for k in ex.ALLOWLIST if str(ex.ALLOWLIST[k]).endswith(".json")]
        for key in json_keys:
            payload = ex.load_json(key)  # runs schema validation where one exists
            self.assertIsInstance(payload, (dict, list), key)

    def test_every_importance_csv_passes_structural_validation(self):
        for model in ("random_forest", "xgboost"):
            rows = ex.load_importance_rows(model)  # raises on any contract violation
            self.assertTrue(rows)

    def test_json_nan_and_infinity_are_rejected(self):
        for bad in ('{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}', '{"x": 1e999}'):
            with self.assertRaises(ex.ExplainabilityError, msg=bad):
                ex._parse_json(bad)

    def test_malformed_json_root_type_is_rejected(self):
        with patch.object(ex, "_read_text", return_value="[]"):
            ex._CACHE.clear()
            with self.assertRaises(ex.ExplainabilityError):
                ex.load_json("selection")

    def test_missing_required_json_fields_are_rejected(self):
        with patch.object(ex, "_read_text", return_value='{"selected_count_models": {}}'):
            ex._CACHE.clear()
            with self.assertRaises(ex.ExplainabilityError):
                ex.load_json("selection")

    def _csv(self, rows):
        header = "model_name,horizon,validation_fold,importance_method,feature_name,importance"
        return header + "\n" + "\n".join(rows)

    def test_csv_fold_13_is_rejected(self):
        with patch.object(ex, "_read_text", return_value=self._csv(["Random Forest,4,13,impurity,f,0.5"])):
            ex._CACHE.clear()
            with self.assertRaises(ex.ExplainabilityError):
                ex.load_importance_rows("random_forest")

    def test_csv_fold_0_is_rejected(self):
        with patch.object(ex, "_read_text", return_value=self._csv(["Random Forest,4,0,impurity,f,0.5"])):
            ex._CACHE.clear()
            with self.assertRaises(ex.ExplainabilityError):
                ex.load_importance_rows("random_forest")

    def test_csv_wrong_model_name_is_rejected(self):
        with patch.object(ex, "_read_text", return_value=self._csv(["XGBoost,4,1,impurity,f,0.5"])):
            ex._CACHE.clear()
            with self.assertRaises(ex.ExplainabilityError):
                ex.load_importance_rows("random_forest")

    def test_csv_duplicate_rows_are_rejected(self):
        dup = self._csv(["Random Forest,4,1,impurity,f,0.5", "Random Forest,4,1,impurity,f,0.6"])
        with patch.object(ex, "_read_text", return_value=dup):
            ex._CACHE.clear()
            with self.assertRaises(ex.ExplainabilityError):
                ex.load_importance_rows("random_forest")

    def test_csv_missing_folds_are_rejected(self):
        # Only folds 1-11 for one feature (missing fold 12) -> contract violation.
        rows = [f"Random Forest,4,{i},impurity,onlyfeat,0.5" for i in range(1, 12)]
        with patch.object(ex, "_read_text", return_value=self._csv(rows)):
            ex._CACHE.clear()
            with self.assertRaises(ex.ExplainabilityError):
                ex.load_importance_rows("random_forest")

    def test_overall_mismatch_fails_because_of_the_mismatch(self):
        # Full-contract fixture: the real valid CSV with EXACTLY ONE overall value
        # changed. The failure must be attributed to the overall/mean mismatch, not
        # to an earlier feature-count or fold failure.
        real = ex.ALLOWLIST["random_forest_importance"].read_text(encoding="utf-8")
        lines = real.splitlines()
        # First control: the untouched real text must pass structural validation.
        ex._CACHE.clear()
        with patch.object(ex, "_read_text", return_value=real):
            ex.load_importance_rows("random_forest")  # no raise

        # Mutate exactly one 'overall' impurity row's value.
        mutated = None
        out = []
        for line in lines:
            parts = line.split(",")
            if mutated is None and len(parts) == 6 and parts[2] == "overall" and parts[3] == "impurity":
                parts[5] = str(float(parts[5]) + 5.0)  # break the mean equality
                mutated = parts[4]
            out.append(",".join(parts))
        self.assertIsNotNone(mutated, "no overall impurity row found to mutate")

        ex._CACHE.clear()
        with patch.object(ex, "_read_text", return_value="\n".join(out)):
            with self.assertRaises(ex.ExplainabilityError) as ctx:
                ex.load_importance_rows("random_forest")
        message = str(ctx.exception).lower()
        self.assertIn("overall", message)
        self.assertIn("mean of folds", message)
        self.assertNotIn("must have exactly", message)  # not a feature-count failure
        ex._CACHE.clear()

    def test_native_negative_importance_is_rejected(self):
        real = ex.ALLOWLIST["random_forest_importance"].read_text(encoding="utf-8")
        lines = real.splitlines()
        out = []
        flipped = False
        for line in lines:
            parts = line.split(",")
            if not flipped and len(parts) == 6 and parts[3] == "impurity" and parts[2] == "1":
                parts[5] = "-0.5"
                flipped = True
            out.append(",".join(parts))
        ex._CACHE.clear()
        with patch.object(ex, "_read_text", return_value="\n".join(out)):
            with self.assertRaises(ex.ExplainabilityError):
                ex.load_importance_rows("random_forest")
        ex._CACHE.clear()

    def test_unsupported_importance_method_is_rejected(self):
        real = ex.ALLOWLIST["xgboost_importance"].read_text(encoding="utf-8")
        # 'impurity' is not an approved method for XGBoost (only gain + permutation).
        tampered = real.replace(",gain,", ",impurity,", 1)
        ex._CACHE.clear()
        with patch.object(ex, "_read_text", return_value=tampered):
            with self.assertRaises(ex.ExplainabilityError):
                ex.load_importance_rows("xgboost")
        ex._CACHE.clear()

    def test_blank_required_csv_field_is_rejected(self):
        rows = self._csv(["Random Forest,4,1,impurity,,0.5"])  # blank feature_name
        ex._CACHE.clear()
        with patch.object(ex, "_read_text", return_value=rows):
            with self.assertRaises(ex.ExplainabilityError):
                ex.load_importance_rows("random_forest")
        ex._CACHE.clear()


class ExplainabilityRouteTests(unittest.TestCase):
    """HTTP layer: auth, envelope, 400/503, no DB writes."""

    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL required.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.app.config["WTF_CSRF_ENABLED"] = False

    def _client(self, expired=False):
        from app.services.dhis2_auth import DHIS2SessionUser, DHIS2_IDENTITY_SESSION_KEY
        verified = int(datetime.now(timezone.utc).timestamp())
        if expired:
            verified -= 60 * 60  # an hour ago -> past the 30-minute window
        ident = DHIS2SessionUser(id="explo-test", display_name="TESTER", verified_at=verified)
        client = self.app.test_client()
        with client.session_transaction() as s:
            s[DHIS2_IDENTITY_SESSION_KEY] = ident.to_session()
            s["_user_id"] = ident.get_id()
            s["_fresh"] = True
        return client

    ENDPOINTS = (
        "/api/explainability/overview?horizon=4",
        "/api/explainability/model-selection?horizon=8",
        "/api/explainability/validation-method",
        "/api/explainability/threshold-method?district_id=UGA_PADER&horizon=4",
        "/api/explainability/feature-importance?model=random_forest&horizon=4&method=native&limit=20&grouped=false",
        "/api/system/pipeline-status",
    )

    def test_all_endpoints_require_authentication_json_401(self):
        client = self.app.test_client()
        for url in self.ENDPOINTS:
            resp = client.get(url)
            self.assertEqual(resp.status_code, 401, url)
            self.assertEqual(resp.mimetype, "application/json", url)
            self.assertEqual(resp.get_json()["status"], "unauthenticated", url)

    def test_expired_session_is_rejected(self):
        client = self._client(expired=True)
        resp = client.get("/api/explainability/overview?horizon=4")
        self.assertEqual(resp.status_code, 401)

    def test_all_endpoints_available_with_full_envelope(self):
        client = self._client()
        for url in self.ENDPOINTS:
            resp = client.get(url)
            self.assertEqual(resp.status_code, 200, url)
            body = resp.get_json()
            for key in ("status", "reason", "data", "params", "empty_states", "metadata", "warnings", "limitations"):
                self.assertIn(key, body, f"{url} missing {key}")
            self.assertEqual(body["status"], "available", url)

    def test_invalid_parameters_return_json_400(self):
        client = self._client()
        for url in (
            "/api/explainability/overview?horizon=7",
            "/api/explainability/model-selection?horizon=3",
            "/api/explainability/feature-importance?model=baseline&horizon=4",
            "/api/explainability/feature-importance?model=random_forest&horizon=4&limit=0",
            "/api/explainability/feature-importance?model=random_forest&horizon=4&limit=51",
            "/api/explainability/feature-importance?model=random_forest&method=bogus",
            "/api/explainability/feature-importance?model=random_forest&grouped=maybe",
            "/api/explainability/threshold-method?district_id=UGA_NOWHERE",
        ):
            resp = client.get(url)
            self.assertEqual(resp.status_code, 400, url)
            self.assertEqual(resp.get_json()["status"], "invalid_request", url)

    def test_endpoints_do_not_write_to_database(self):
        client = self._client()
        with patch.object(db.session, "commit") as commit, patch.object(db.session, "flush") as flush:
            for url in self.ENDPOINTS:
                client.get(url)
            commit.assert_not_called()
            flush.assert_not_called()

    def test_responses_contain_no_paths_or_secrets(self):
        client = self._client()
        for url in self.ENDPOINTS:
            body = client.get(url).get_data(as_text=True)
            self.assertNotIn("artifact_paths", body, url)
            self.assertNotIn("proof_path", body, url)
            self.assertNotIn("postgresql://", body, url)
            self.assertNotIn(".joblib", body, url)

    def test_missing_evidence_returns_controlled_503_envelope(self):
        client = self._client()
        # Force every evidence read to fail; endpoints must return a clean 503
        # envelope, never a 500 or a stack trace.
        with patch("app.services.explainability._read_text",
                   side_effect=ex.ExplainabilityError("missing evidence")):
            ex._CACHE.clear()
            for url in (
                "/api/explainability/overview?horizon=4",
                "/api/explainability/model-selection?horizon=4",
                "/api/explainability/validation-method",
                "/api/explainability/feature-importance?model=random_forest&horizon=4",
                "/api/system/pipeline-status",
            ):
                resp = client.get(url)
                self.assertEqual(resp.status_code, 503, url)
                body = resp.get_json()
                self.assertEqual(body["status"], "unavailable", url)
                self.assertIsNone(body["data"], url)
                for key in ("status", "reason", "data", "empty_states", "metadata", "warnings", "limitations"):
                    self.assertIn(key, body, url)
        ex._CACHE.clear()

    def test_unexpected_error_is_contained_as_503_no_trace(self):
        client = self._client()
        # The route imported build_overview by value, so patch it on the route module.
        with patch("app.blueprints.explainability.routes.build_overview", side_effect=TypeError("boom")):
            resp = client.get("/api/explainability/overview?horizon=4")
        self.assertEqual(resp.status_code, 503)
        body = resp.get_json()
        self.assertEqual(body["status"], "unavailable")
        self.assertNotIn("Traceback", resp.get_data(as_text=True))
        self.assertNotIn("boom", resp.get_data(as_text=True))

    def test_no_sql_writes_are_emitted(self):
        # Detect INSERT/UPDATE/DELETE at the engine level, not just commit/flush.
        client = self._client()
        statements = []

        from sqlalchemy import event

        with self.app.app_context():
            engine = db.engine

        def before_cursor(conn, cursor, statement, params, context, executemany):
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", before_cursor)
        try:
            for url in self.ENDPOINTS:
                client.get(url)
        finally:
            event.remove(engine, "before_cursor_execute", before_cursor)

        writes = [s for s in statements if s.lstrip()[:6].upper() in ("INSERT", "UPDATE", "DELETE")]
        self.assertEqual(writes, [], f"unexpected write statements: {writes}")

    def test_forecast_snapshot_is_identical_before_and_after(self):
        # Deterministic DB snapshot around all endpoint requests.
        from app.models import ForecastOutput
        import hashlib

        def snapshot():
            with self.app.app_context():
                rows = ForecastOutput.query.order_by(ForecastOutput.id).all()
                payload = [
                    [r.id, r.district_id, r.model_name, r.horizon_weeks, str(r.predicted_cases),
                     str(r.threshold_used), r.risk_level, r.pipeline_run_id]
                    for r in rows
                ]
                return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), len(rows)

        before = snapshot()
        client = self._client()
        for url in self.ENDPOINTS:
            client.get(url)
        after = snapshot()
        self.assertEqual(before, after)
        self.assertEqual(before[1], 1752)


class StageASchemaClosureHTTPTests(unittest.TestCase):
    """Authenticated route tests proving HTTP 503 for incomplete/invalid evidence."""

    @classmethod
    def setUpClass(cls):
        load_dotenv(".env")
        if not os.environ.get("DATABASE_URL"):
            raise unittest.SkipTest("DATABASE_URL required.")
        os.environ.setdefault("SECRET_KEY", "test-secret")
        cls.app = create_app()
        cls.app.config["WTF_CSRF_ENABLED"] = False
        cls.selection = json.loads(Path("data/quality/model_performance_selection_summary.json").read_text(encoding="utf-8"))
        cls.forecast = json.loads(Path("data/quality/forecast_generation_summary.json").read_text(encoding="utf-8"))
        cls.validation = json.loads(Path("data/quality/random_forest_validation_summary.json").read_text(encoding="utf-8"))

    def setUp(self):
        ex._CACHE.clear()

    def tearDown(self):
        ex._CACHE.clear()

    def _client(self):
        from app.services.dhis2_auth import DHIS2SessionUser, DHIS2_IDENTITY_SESSION_KEY
        ident = DHIS2SessionUser(id="stagea", display_name="TESTER",
                                 verified_at=int(datetime.now(timezone.utc).timestamp()))
        client = self.app.test_client()
        with client.session_transaction() as s:
            s[DHIS2_IDENTITY_SESSION_KEY] = ident.to_session()
            s["_user_id"] = ident.get_id()
            s["_fresh"] = True
        return client

    def _assert_503(self, url, read_text_value):
        client = self._client()
        with patch("app.services.explainability._read_text", return_value=read_text_value):
            ex._CACHE.clear()
            resp = client.get(url)
        self.assertEqual(resp.status_code, 503, f"{url}\n{resp.get_data(as_text=True)[:200]}")
        body = resp.get_json()
        self.assertEqual(body["status"], "unavailable")
        self.assertIsNone(body["data"])

    def _mutate(self, base, fn):
        clone = json.loads(json.dumps(base))
        fn(clone)
        return json.dumps(clone)

    def test_503_when_one_comparison_instead_of_four(self):
        def drop(c):
            c["all_models_compared"] = [m for m in c["all_models_compared"]
                                        if not (m["horizon"] == 4 and m["model_name"] in ("SARIMAX", "Baseline/Naive", "Random Forest"))]
        self._assert_503("/api/explainability/model-selection?horizon=4", self._mutate(self.selection, drop))

    def test_503_on_duplicate_model_comparison(self):
        def dup(c):
            first = next(m for m in c["all_models_compared"] if m["horizon"] == 4)
            extra = json.loads(json.dumps(first))
            # remove a different model, add a duplicate of `first` -> 4 rows but a dup
            c["all_models_compared"] = [m for m in c["all_models_compared"]
                                        if not (m["horizon"] == 4 and m["model_name"] == "SARIMAX")]
            c["all_models_compared"].append(extra)
        self._assert_503("/api/explainability/model-selection?horizon=4", self._mutate(self.selection, dup))

    def test_503_on_unsupported_model_comparison(self):
        def rename(c):
            for m in c["all_models_compared"]:
                if m["horizon"] == 4 and m["model_name"] == "SARIMAX":
                    m["model_name"] = "MysteryModel"
        self._assert_503("/api/explainability/model-selection?horizon=4", self._mutate(self.selection, rename))

    def test_503_on_null_winner_metric(self):
        def nullify(c):
            entry = c["selected_count_models"]["4"]
            entry["count_metrics"]["wape"] = None
            entry["count_metrics"]["rmse"] = None
        self._assert_503("/api/explainability/model-selection?horizon=4", self._mutate(self.selection, nullify))

    def test_503_on_missing_target_weeks(self):
        def drop(c):
            del c["target_weeks"]
        self._assert_503("/api/explainability/overview?horizon=4", self._mutate(self.forecast, drop))

    def test_503_on_missing_one_target_horizon(self):
        # pipeline-status reads forecast_generation via _read_text; a forecast that
        # is missing one target horizon must make it unavailable.
        def drop(c):
            del c["target_weeks"]["8"]
        self._assert_503("/api/system/pipeline-status", self._mutate(self.forecast, drop))

    def test_503_on_wrong_validation_method(self):
        def wrong(c):
            c["validation_method"] = "kfold"
        self._assert_503("/api/explainability/validation-method", self._mutate(self.validation, wrong))

    def test_503_on_empty_validation_horizons(self):
        def empty(c):
            c["horizons"] = []
        self._assert_503("/api/explainability/validation-method", self._mutate(self.validation, empty))

    def test_503_on_missing_validation_fold(self):
        def drop(c):
            c["validation_cutoffs"] = [x for x in c["validation_cutoffs"]
                                       if not (x["horizon"] == 4 and x["validation_fold"] == 12)]
        self._assert_503("/api/explainability/validation-method", self._mutate(self.validation, drop))

    def test_503_on_duplicate_validation_fold(self):
        def dup(c):
            first = next(x for x in c["validation_cutoffs"] if x["horizon"] == 4 and x["validation_fold"] == 1)
            c["validation_cutoffs"].append(json.loads(json.dumps(first)))
            c["validation_cutoffs"] = [x for x in c["validation_cutoffs"]
                                       if not (x["horizon"] == 4 and x["validation_fold"] == 2)]
        self._assert_503("/api/explainability/validation-method", self._mutate(self.validation, dup))

    def test_503_on_unsupported_importance_method_via_http(self):
        real = ex.ALLOWLIST["xgboost_importance"].read_text(encoding="utf-8").replace(",gain,", ",impurity,", 1)
        client = self._client()
        with patch("app.services.explainability._read_text", return_value=real):
            ex._CACHE.clear()
            resp = client.get("/api/explainability/feature-importance?model=xgboost&horizon=4&method=native")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.get_json()["status"], "unavailable")

    def test_503_on_incorrect_overall_mean_via_http(self):
        real = ex.ALLOWLIST["random_forest_importance"].read_text(encoding="utf-8")
        out = []
        mutated = False
        for line in real.splitlines():
            parts = line.split(",")
            if not mutated and len(parts) == 6 and parts[2] == "overall" and parts[3] == "impurity":
                parts[5] = str(float(parts[5]) + 5.0)
                mutated = True
            out.append(",".join(parts))
        client = self._client()
        with patch("app.services.explainability._read_text", return_value="\n".join(out)):
            ex._CACHE.clear()
            resp = client.get("/api/explainability/feature-importance?model=random_forest&horizon=4&method=native")
        self.assertEqual(resp.status_code, 503)


if __name__ == "__main__":
    unittest.main()
