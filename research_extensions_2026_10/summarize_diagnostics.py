"""Coverage reasons and compact district distributions from saved evidence."""
import json
from collections import Counter
import pandas as pd
from research_extensions_2026_10.run_study import RESULTS,EXPERIMENTS,write_json


def main():
    d=json.loads((RESULTS/'sarimax/diagnostics.json').read_text())
    # Diagnostics structure is explicit; never infer coverage from job status alone.
    districts=d['district_horizon_diagnostics'] if 'district_horizon_diagnostics' in d else d['jobs']
    all_folds=[f for job in districts for f in job['fold_diagnostics']]
    failed=[f for f in all_folds if f['status']=='failed']
    report={'job_counts':d['job_counts'],'fold_status_counts':dict(Counter(f['status'] for f in all_folds)),
        'failed_fold_count':len(failed),'failed_fold_reasons':dict(Counter('insufficient_observed_history' if 'observed malaria weeks' in f['warning_or_failure_reason'] else 'other' for f in failed)),
        'failed_fold_numbers':dict(Counter(str(f['validation_fold']) for f in failed)),
        'failed_examples':[{'origin':f['origin_week'],'reason':f['warning_or_failure_reason']} for f in failed[:3]],
        'successful_prediction_rows':sum(len(j['predictions']) for j in districts),
        'fallback_prediction_rows':sum(j['fallback_folds'] for j in districts),
        'seasonal_order_counts':dict(Counter(str(j['selected_seasonal_order']) for j in districts))}
    write_json(RESULTS/'sarimax_coverage_reasons.json',report)
    metrics=pd.read_csv(RESULTS/'district_metrics.csv')
    rows=[]
    for experiment,f in metrics.groupby('experiment'):
        for key in ['wape','sensitivity','precision','f2']:
            valid=f[key].dropna()
            rows.append({'experiment':experiment,'metric':key,'defined_districts':len(valid),
                'q10':valid.quantile(.1),'q25':valid.quantile(.25),'median':valid.median(),'q75':valid.quantile(.75),'q90':valid.quantile(.9)})
    pd.DataFrame(rows).to_csv(RESULTS/'district_metric_distribution.csv',index=False)
    permutation=[]
    for path in sorted((EXPERIMENTS/'importance').glob('legacy12_rf_h*_start2020_all_*.json')):
        horizon=int(path.stem.split('_h')[1].split('_')[0]);origin=path.stem.rsplit('_',1)[1]
        for entry in json.loads(path.read_text()):permutation.append({'horizon':horizon,'origin':origin,**entry})
    p=pd.DataFrame(permutation)
    assert len(p.origin.unique())>=12 and len(p)==36*7
    p.to_csv(RESULTS/'grouped_permutation_per_origin.csv',index=False)
    summary=p.groupby(['horizon','group']).mean_mae_increase.agg(['count','mean','median','min','max']).reset_index()
    summary['interpretation']='Three within-origin permutation repeats; descriptive origin distribution, not causal or a sampling confidence interval. Time is constant within origin.'
    summary.to_csv(RESULTS/'grouped_permutation_summary.csv',index=False)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
