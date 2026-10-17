"""Regression tests for the October 2026 scientific correctness repairs."""
from datetime import date, timedelta
from decimal import Decimal
import unittest

from app.services.model_validation import (
    _pr_auc, _seasonal_week_distance, ModelValidationError, district_threshold,
)
from app.services.models.pooled_tree_validation import (
    _chronological_inner_split, _prediction_record,
)
from app.services.models.sarimax import _complete_weekly_history
from app.services.risk_classification import classify_forecast_risk


class ResearchCorrectionTests(unittest.TestCase):
    def test_case_margin_is_never_reported_as_probability_brier_score(self):
        from app.services.model_validation import classification_metrics
        margins=[Decimal('.2'),Decimal('.8')]
        self.assertIsNone(classification_metrics([False,True],[False,True],margins)['brier_score'])
        self.assertEqual(classification_metrics([False,True],[False,True],margins,score_is_probability=True)['brier_score'], '0.040000')
    def test_interpolation_cannot_use_a_future_right_anchor(self):
        from app.services.case_data_cleaning import CleanCaseValue
        from app.services.modelling_dataset import _clean_feature_value
        value = CleanCaseValue(None, Decimal('20'), 'source_null', True, 2, date(2025,1,27))
        self.assertIsNone(_clean_feature_value(value, as_of=date(2025,1,20)))
        self.assertEqual(_clean_feature_value(value, as_of=date(2025,1,27)), Decimal('20'))

    def test_inner_fit_never_sees_labels_after_first_heldout_origin(self):
        for h in (4, 8, 12):
            start = date(2020, 1, 6)
            rows = [{"week_start_date": (start + timedelta(weeks=i)).isoformat(),
                     f"target_week_start_date_h{h}": (start + timedelta(weeks=i+h)).isoformat()}
                    for i in range(50)]
            holdout = {r[f"target_week_start_date_h{h}"] for r in rows[-8:]}
            training, validation = _chronological_inner_split(rows, horizon=h, holdout_dates=holdout)
            cutoff = min(r["week_start_date"] for r in validation)
            self.assertTrue(training)
            self.assertTrue(all(r[f"target_week_start_date_h{h}"] <= cutoff for r in training))
            self.assertFalse(set(map(id, training)) & set(map(id, validation)))

    def test_average_precision_ties_are_order_invariant(self):
        scores = [Decimal(".5"), Decimal(".5")]
        self.assertEqual(_pr_auc([True, False], scores), Decimal(".500000"))
        self.assertEqual(_pr_auc([False, True], scores), Decimal(".500000"))

    def test_average_precision_agrees_with_standard_implementation(self):
        from sklearn.metrics import average_precision_score
        labels = [True, False, True, False, False, True]
        scores = [Decimal(x) for x in (".8", ".8", ".8", ".2", ".2", ".1")]
        self.assertAlmostEqual(float(_pr_auc(labels, scores)), average_precision_score(labels, [float(x) for x in scores]), places=6)

    def test_missing_week_keeps_calendar_slot_and_missing_outcome(self):
        rows = [{"week_start_date": "2025-01-06", "confirmed_malaria_cases": "10"},
                {"week_start_date": "2025-01-20", "confirmed_malaria_cases": "30"}]
        weekly = _complete_weekly_history(rows)
        self.assertEqual(len(weekly), 3)
        self.assertEqual(weekly[1]["week_start_date"], "2025-01-13")
        self.assertIsNone(weekly[1]["confirmed_malaria_cases"])
        self.assertEqual(weekly[2]["confirmed_malaria_cases"], "30")

    def test_empty_threshold_history_fails_instead_of_inventing_zero(self):
        with self.assertRaises(ModelValidationError):
            district_threshold([], "A", target_epi_week=10)

    def test_seasonal_distance_respects_reference_year_length(self):
        self.assertEqual(_seasonal_week_distance({"epi_year": 2021, "epi_week": 52}, 1), 1)
        self.assertEqual(_seasonal_week_distance({"epi_year": 2020, "epi_week": 53}, 1), 1)
        self.assertEqual(_seasonal_week_distance({"epi_year": 2020, "epi_week": 52}, 1), 2)

    def test_count_threshold_record_uses_operational_flag_and_margin(self):
        row = {"district_id": "A", "horizon": 4, "forecast_origin_week": "2025-01-06",
               "target_week": "2025-02-03", "observed_cases": "20", "predicted_cases": "5",
               "district_surge_threshold": "10", "observed_surge_status": True,
               "predicted_surge_classification": True, "surge_probability": ".9",
               "count_threshold_surge_classification": False, "predicted_surge_score": "-5"}
        operational = _prediction_record(row, surge_method="count_threshold")
        self.assertFalse(operational.predicted_surge)
        self.assertEqual(operational.surge_score, Decimal("-5"))
        self.assertTrue(_prediction_record(row).predicted_surge)

    def test_zero_threshold_has_no_usable_ratio_risk(self):
        risk = classify_forecast_risk(predicted_surge_flag=False, surge_probability=None,
                                     predicted_cases=0, threshold_used=0)
        self.assertTrue(risk.no_data)

    def test_selection_explanation_names_actual_f2_rule(self):
        from app.services.explainability import build_model_selection
        selected = build_model_selection(horizon=4)["data"]["surge_selection"]
        self.assertEqual(selected["metric_name"], "F2")
        self.assertEqual(selected["metric_value"], selected["surge_metrics"]["f2"])


if __name__ == "__main__":
    unittest.main()
