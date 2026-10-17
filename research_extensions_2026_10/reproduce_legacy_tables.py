"""Replay frozen metric definitions; do not refit or overwrite submitted results."""
import csv
import importlib.util
import json
import sys
from datetime import date
from decimal import Decimal
from research_extensions_2026_10.run_study import OUT,RESULTS,DATA,write_json


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    validation=load('_frozen_validation_v1',OUT/'baseline/app/services/model_validation.py')
    baseline=load('_frozen_baseline_v1',OUT/'baseline/app/services/models/baseline.py')
    for name in ['PredictionRecord','ValidationSummary','build_rolling_origin_folds','district_threshold','evaluate_predictions','write_validation_artifact']:
        setattr(baseline,name,getattr(validation,name))
    baseline_result=baseline.run_baseline_validation(dataset_path=DATA,artifact_path=RESULTS/'legacy_baseline_reproduction_summary.json')
    checks=[]
    for model in ['baseline','random_forest','xgboost','sarimax']:
        expected=json.loads((OUT/f'baseline/data/quality/{model}_validation_summary.json').read_text())
        if model=='baseline':count=baseline_result.count_metrics_by_horizon;surge=baseline_result.surge_metrics_by_horizon
        else:
            rows=list(csv.DictReader((OUT/f'baseline/data/modelling/{model}_validation_predictions.csv').open(encoding='utf-8')))
            count={};surge={}
            for h in [4,8,12]:
                records=[]
                for row in rows:
                    if int(row['horizon'])!=h:continue
                    threshold=row['surge_threshold'] if model=='sarimax' else row['district_surge_threshold']
                    records.append(validation.PredictionRecord(district_id=row['district_id'],horizon=h,
                        origin_date=date.fromisoformat(row['forecast_origin_week']),target_date=date.fromisoformat(row['target_week']),
                        actual=Decimal(row['observed_cases']),predicted=Decimal(row['predicted_cases']),threshold=Decimal(threshold),
                        actual_surge=row['observed_surge_status'].lower()=='true',
                        predicted_surge=row['predicted_surge_classification' if model=='sarimax' else 'count_threshold_surge_classification'].lower()=='true',
                        surge_score=Decimal(row['predicted_surge_score'])))
                count[h],surge[h]=validation.evaluate_predictions(records)
        for h in [4,8,12]:
            for family,computed,published in [('count',count[h],expected['count_metrics_by_horizon'][str(h)]),
                ('surge',surge[h],expected['surge_metrics_by_horizon'][str(h)])]:
                for key,value in published.items():
                    measured=computed[key]
                    # CSV predictions are rounded to 4 decimals. The legacy in-memory
                    # pipeline can retain more precision than the export for ties.
                    difference=None if value is None or measured is None else str(Decimal(str(measured))-Decimal(str(value)))
                    exact=measured==value
                    at_report_precision=(measured is None and value is None) or (difference is not None and abs(Decimal(difference))<=Decimal('0.000001'))
                    checks.append({'model':model,'horizon':h,'family':family,'metric':key,'frozen_summary':value,
                        'recomputed':measured,'difference':difference,'exact_export_replay':exact,'within_1e_minus6':at_report_precision})
    report={'definition':'Frozen pre-correction functions; original scores/thresholds/operational flags. Saved tree/SAR CSV precision retained. Baseline rule recomputed from frozen dataset.',
        'checks':checks,'all_count_and_confusion_metrics_match_reporting_precision':all(c['within_1e_minus6'] for c in checks if c['family']=='count' or c['metric'] in ['tp','fp','fn','tn','n']),
        'nonexact_reporting_items':[c for c in checks if not c['within_1e_minus6']],
        'note':'This is exported-number reproduction, not a fresh refit of every legacy tree/SAR estimator. Tie-order and rounded-score discrepancies are disclosed, not silently adjusted.'}
    write_json(RESULTS/'legacy_table_reproduction.json',report)
    assert report['all_count_and_confusion_metrics_match_reporting_precision']
    print(json.dumps({k:report[k] for k in ['all_count_and_confusion_metrics_match_reporting_precision','nonexact_reporting_items']}),flush=True)


if __name__=='__main__':main()
