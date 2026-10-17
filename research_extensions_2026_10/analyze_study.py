"""Auditable follow-up tables and paired uncertainty, using completed predictions."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.model_validation import build_rolling_origin_folds, _seasonal_week_distance
from app.services.models.baseline import _baseline_prediction
from research_extensions_2026_10.run_study import DATA, OUT, RESULTS, EXPERIMENTS, HORIZONS, SEED, Study, metrics, write_json, summarize

KEYS=['district_id','horizon','origin','target']


def load_experiment(name):
    files=sorted((EXPERIMENTS/'predictions').glob(name+'_*.csv'))
    if not files:return None
    frame=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    if frame.duplicated(KEYS).any():raise ValueError('Duplicate forecast keys '+name)
    return frame


def legacy_predictions():
    """Saved count outputs with their original labels; no refitting claimed."""
    outputs=[]
    for model,filename in [('rf','random_forest'),('xgb','xgboost'),('sarimax','sarimax')]:
        f=pd.read_csv(OUT/'baseline/data/modelling'/f'{filename}_validation_predictions.csv')
        threshold='district_surge_threshold' if model in ('rf','xgb') else 'surge_threshold'
        f=f.rename(columns={'forecast_origin_week':'origin','target_week':'target','observed_cases':'actual',
                            'predicted_cases':'predicted',threshold:'threshold'})
        f=f[KEYS+['actual','predicted','threshold']]
        f.to_csv(RESULTS/f'legacy_{model}_predictions.csv',index=False)
        for h,g in f.groupby('horizon'):outputs.append({'model':model,'horizon':h,**metrics(g)})
    pd.DataFrame(outputs).to_csv(RESULTS/'legacy_recomputed_metrics.csv',index=False)


def simple_baselines(study, expanded=False):
    """Independent simple forecasts on identical eligible feature/target rows."""
    outputs={name:[] for name in ['seasonal_naive','persistence','recent_mean4','seasonal_mean']}
    for h in HORIZONS:
        if expanded:
            targets=pd.date_range('2024-03-25','2025-12-22',freq='W-MON')
            pairs=[((t-timedelta(weeks=h)).date(),t.date()) for t in targets]
        else:
            pairs=[(f.origin_date,f.target_date) for f in build_rolling_origin_folds(study.rows,horizon=h)]
        for origin,target in pairs:
            o,t=origin.isoformat(),target.isoformat()
            val=np.flatnonzero((study.dates==o)&(study.target_dates[h]==t)&np.isfinite(study.y[h]))
            for i in val:
                d=str(study.ids[i]);history=[r for r in study.history[d] if r['week_start_date']<=o]
                observed=[r for r in history if r['confirmed_malaria_cases']!='']
                if not observed:continue
                targetweek=target.isocalendar().week
                seasonal=[float(r['confirmed_malaria_cases']) for r in observed if int(r['epi_week'])==targetweek]
                predictions={'seasonal_naive':float(_baseline_prediction(history,origin_date=origin,target_epi_week=targetweek)),
                             'persistence':float(observed[-1]['confirmed_malaria_cases']),
                             'recent_mean4':float(np.mean([float(r['confirmed_malaria_cases']) for r in observed[-4:]])),
                             'seasonal_mean':float(np.mean(seasonal)) if seasonal else float(np.mean([float(r['confirmed_malaria_cases']) for r in observed[-4:]]))}
                proof=study.threshold(d,o,t)
                for name,pred in predictions.items():
                    outputs[name].append({'district_id':d,'horizon':h,'origin':o,'target':t,'actual':study.y[h][i],
                                          'predicted':pred,'threshold':float(proof['threshold']),
                                          'threshold_fallback':not proof['seasonal_window_used']})
    protocol='weekly_common' if expanded else 'legacy12'
    for name,rows in outputs.items():
        f=pd.DataFrame(rows)
        for h,g in f.groupby('horizon'):
            g.to_csv(RESULTS/f'{protocol}_{name}_h{h}_predictions.csv',index=False)
            # One file per scenario is intentional: these models do not require checkpointed fitting.
    write_json(RESULTS/f'{protocol}_simple_baseline_specification.json',{
        'training_history':'all observed past outcomes through origin; not imputed; current vintage',
        'seasonal_naive':'latest observed matching ISO week; fallback most recent four observed outcomes',
        'persistence':'most recent observed outcome at or before origin',
        'recent_mean4':'mean of most recent four observed outcomes (may span a calendar gap)',
        'seasonal_mean':'mean of all observed matching ISO weeks; recent-four fallback',
        'score':'predicted minus full-past corrected district Q75 threshold',
        'evaluation':'identical valid origin/target rows to tree comparisons'})


def contributions(frame):
    y,p,t=[frame[c].to_numpy(float) for c in ['actual','predicted','threshold']]
    a,b=y>t,p>t
    return np.column_stack([y,np.abs(p-y),np.square(p-y),a&b,~a&b,a&~b,~a&~b,np.ones(len(y))])


def from_sums(s):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.column_stack([100*s[:,1]/s[:,0],s[:,1]/s[:,7],
                                5*s[:,3]/(5*s[:,3]+4*s[:,5]+s[:,4]),
                                s[:,3]/(s[:,3]+s[:,5]),s[:,3]/(s[:,3]+s[:,4])])


METRIC_NAMES=['wape','mae','f2','sensitivity','precision']


def paired_bootstrap(first,second,*,cluster,draws=2000,block=12):
    merged=first.merge(second,on=KEYS,suffixes=('_first','_second'),validate='one_to_one')
    if len(merged)==0:raise ValueError('No paired rows')
    if not np.array_equal(merged.actual_first,merged.actual_second):raise ValueError('Paired outcomes differ')
    if not np.array_equal(merged.threshold_first,merged.threshold_second):raise ValueError('Paired thresholds differ')
    a=merged[KEYS].assign(actual=merged.actual_first,predicted=merged.predicted_first,threshold=merged.threshold_first)
    b=merged[KEYS].assign(actual=merged.actual_second,predicted=merged.predicted_second,threshold=merged.threshold_second)
    c= 'district_id' if cluster=='district' else 'target'
    labels=sorted(merged[c].unique()); indexes={label:i for i,label in enumerate(labels)}
    bins=np.array([indexes[x] for x in merged[c]])
    aggregates=[]
    for f in [a,b]:
        sums=np.zeros((len(labels),8));np.add.at(sums,bins,contributions(f));aggregates.append(sums)
    rng=np.random.default_rng(SEED)
    if cluster=='moving_block':
        n=len(labels);weights=np.zeros((draws,n))
        for j in range(draws):
            # Circular, fixed-length temporal blocks; preserves paired district rows.
            starts=rng.integers(0,n,size=int(np.ceil(n/block)))
            indices=np.concatenate([(start+np.arange(block))%n for start in starts])[:n]
            weights[j]=np.bincount(indices,minlength=n)
    else:weights=rng.multinomial(len(labels),np.full(len(labels),1/len(labels)),size=draws)
    distributions=[from_sums(weights@sums) for sums in aggregates]
    point=[from_sums(sums.sum(axis=0,keepdims=True))[0] for sums in aggregates]
    results=[]
    for i,name in enumerate(METRIC_NAMES):
        valid=np.isfinite(distributions[0][:,i]) & np.isfinite(distributions[1][:,i])
        delta=distributions[0][valid,i]-distributions[1][valid,i]
        lo,hi=np.quantile(delta,[.025,.975]) if len(delta) else [np.nan,np.nan]
        results.append({'metric':name,'first':point[0][i],'second':point[1][i],
                        'difference_first_minus_second':point[0][i]-point[1][i],
                        'ci_lower':lo,'ci_upper':hi,'valid_draws':int(valid.sum()),
                        'zero_included':bool(lo<=0<=hi),'cluster':cluster,'clusters':len(labels),
                        'block_length':block if cluster=='moving_block' else None,
                        'paired_rows':len(merged),'first_total_rows':len(first),'second_total_rows':len(second)})
    return results


def threshold_grid(study,frame):
    """Fixed predictions, alternative labels; never reselect models on this grid."""
    rows=[];details=[];base=frame.copy()
    for include2022 in [False,True]:
        windows=[1,2,3] if not include2022 else [2]
        quantiles=[.70,.75,.80] if not include2022 else [.75]
        for window in windows:
            for quantile in quantiles:
                thresholds=[];fallbacks=[]
                for r in base.itertuples():
                    history=[x for x in study.history[r.district_id] if x['week_start_date']<=r.origin and x['confirmed_malaria_cases']!='' and (include2022 or int(x['epi_year'])!=2022)]
                    targetweek=date.fromisoformat(r.target).isocalendar().week
                    seasonal=[x for x in history if _seasonal_week_distance(x,targetweek)<=window]
                    fallback=len(seasonal)<8;ref=history if fallback else seasonal
                    values=sorted(float(x['confirmed_malaria_cases']) for x in ref)
                    if not values:raise ValueError('Empty threshold reference')
                    thresholds.append(values[int(Decimal(len(values)-1)*Decimal(str(quantile)))]);fallbacks.append(fallback)
                altered=base.assign(threshold=thresholds,threshold_fallback=fallbacks)
                for h,g in altered.groupby('horizon'):
                    rows.append({'horizon':h,'quantile':quantile,'window':window,'include2022':include2022,
                                 'fallback_count':int(g.threshold_fallback.sum()),'fallback_share':float(g.threshold_fallback.mean()),**metrics(g)})
                if window==2 and quantile==.75 and not include2022:
                    if not np.array_equal(altered.threshold.to_numpy(),base.threshold.to_numpy()):raise ValueError('Primary grid and fitting thresholds differ')
    pd.DataFrame(rows).to_csv(RESULTS/'threshold_sensitivity.csv',index=False)


def all_tables():
    summarize();legacy_predictions()
    summary=pd.read_csv(RESULTS/'master_metrics.csv')
    comparisons=[];coverage=[];workload=[];uncertainty=[]
    for h in HORIZONS:
        for model in ['rf','xgb']:
            full=load_experiment(f'legacy12_{model}_h{h}_start2020_all')
            if full is None or full.origin.nunique()!=12:continue
            old=pd.read_csv(RESULTS/f'legacy_{model}_predictions.csv');old=old[old.horizon.eq(h)]
            joined=full.merge(old[KEYS+['threshold']],on=KEYS,suffixes=('','_original'),validate='one_to_one')
            corrected_old_labels=joined.assign(threshold=joined.threshold_original)
            old_new_labels=old.drop(columns='threshold').merge(full[KEYS+['threshold']],on=KEYS,validate='one_to_one')
            for tag,g in [('submitted_labels',old),('corrected_predictions_original_threshold',corrected_old_labels),
                          ('original_predictions_corrected_threshold',old_new_labels),('corrected',full),
                          ('corrected_excluding_2025_12_22',full[full.target.ne('2025-12-22')])]:
                coverage.append({'model':model,'horizon':h,'version':tag,**metrics(g)})
            for ratio in [1,2,4,8]:
                m=metrics(full);workload.append({'model':model,'horizon':h,'cost_false_negative_to_false_positive':ratio,
                    'weighted_errors':ratio*m['fn']+m['fp'],'weighted_errors_per_100_predictions':100*(ratio*m['fn']+m['fp'])/m['n'],**m})
            # Metric uncertainty is distinct from uncertainty of a district forecast.
            for cluster in ['district','origin']:
                # Paired first=self returns zero differences; recover absolute CIs below.
                labels=full.district_id if cluster=='district' else full.target
                uniques=sorted(labels.unique());bins=pd.Categorical(labels,categories=uniques).codes
                sums=np.zeros((len(uniques),8));np.add.at(sums,bins,contributions(full))
                rng=np.random.default_rng(SEED);weights=rng.multinomial(len(uniques),np.full(len(uniques),1/len(uniques)),size=2000)
                dist=from_sums(weights@sums);point=from_sums(sums.sum(axis=0,keepdims=True))[0]
                for i,name in enumerate(METRIC_NAMES):
                    values=dist[:,i];values=values[np.isfinite(values)];lo,hi=np.quantile(values,[.025,.975])
                    uncertainty.append({'model':model,'horizon':h,'cluster':cluster,'clusters':len(uniques),
                                        'metric':name,'value':point[i],'ci_lower':lo,'ci_upper':hi,'valid_draws':len(values)})
            for group in ['none','rainfall','temperature','humidity'] if model=='rf' else ['none']:
                reduced=load_experiment(f'legacy12_{model}_h{h}_start2020_{group}')
                if reduced is None or reduced.origin.nunique()!=12:continue
                for cluster in ['district','origin']:
                    for r in paired_bootstrap(full,reduced,cluster=cluster):comparisons.append({'comparison':f'{model}_all_minus_{group}','horizon':h,**r})
        # Separate count and warning point rankings, with paired uncertainty.
        rf=load_experiment(f'legacy12_rf_h{h}_start2020_all')
        xgb=load_experiment(f'legacy12_xgb_h{h}_start2020_all')
        if rf is not None and xgb is not None and rf.origin.nunique()==12 and xgb.origin.nunique()==12:
            for cluster in ['district','origin']:
                for r in paired_bootstrap(rf,xgb,cluster=cluster):comparisons.append({'comparison':'rf_minus_xgb','horizon':h,**r})
        # SARIMAX strict coverage comparison, only once corrected run exists.
        corrected=RESULTS/'sarimax/predictions.csv'
        if corrected.exists():
            sar=pd.read_csv(corrected).rename(columns={'forecast_origin_week':'origin','target_week':'target','observed_cases':'actual','predicted_cases':'predicted','surge_threshold':'threshold'})
            sar=sar[sar.horizon.eq(h)]
            rf=load_experiment(f'legacy12_rf_h{h}_start2020_all')
            if rf is not None and rf.origin.nunique()==12:
                for cluster in ['district','origin']:
                    for r in paired_bootstrap(rf,sar,cluster=cluster):comparisons.append({'comparison':'rf_minus_corrected_sarimax','horizon':h,**r})
        # Window comparisons require all 92 target weeks for every window.
        windows=[load_experiment(f'weekly_common_rf_h{h}_start{s}_all') for s in [2020,2021,2022,2023]]
        if all(f is not None and f.target.nunique()==92 for f in windows):
            keysets=[set(map(tuple,f[KEYS].to_numpy())) for f in windows]
            if not all(keys==keysets[0] for keys in keysets):raise ValueError('Window samples differ')
            for start,f in zip([2021,2022,2023],windows[1:]):
                for cluster in ['district','moving_block']:
                    for r in paired_bootstrap(windows[0],f,cluster=cluster):comparisons.append({'comparison':f'2020_start_minus_{start}_start','horizon':h,**r})
        reduced=load_experiment(f'weekly_climate_rf_h{h}_start2020_none')
        if windows[0] is not None and windows[0].target.nunique()==92 and reduced is not None and reduced.target.nunique()==92:
            for cluster in ['district','moving_block']:
                for r in paired_bootstrap(windows[0],reduced,cluster=cluster):comparisons.append({'comparison':'weekly_rf_all_minus_none','horizon':h,**r})
    pd.DataFrame(comparisons).to_csv(RESULTS/'paired_comparisons.csv',index=False)
    pd.DataFrame(coverage).to_csv(RESULTS/'legacy_corrected_comparison.csv',index=False)
    pd.DataFrame(workload).to_csv(RESULTS/'warning_workload_cost_sensitivity.csv',index=False)
    pd.DataFrame(uncertainty).to_csv(RESULTS/'metric_uncertainty.csv',index=False)
    complete=[load_experiment(f'legacy12_rf_h{h}_start2020_all') for h in HORIZONS]
    if all(f is not None and f.origin.nunique()==12 for f in complete):
        study=Study();threshold_grid(study,pd.concat(complete,ignore_index=True));study.quality()
    meta={'bootstrap_draws':2000,'seed':SEED,'district_interval':'conditional on evaluated origins; does not model dependence between neighbouring districts',
          'origin_interval':'resamples entire national target-week clusters; only twelve sparse clusters in original evaluation',
          'weekly_interval':'circular moving block bootstrap, block length twelve target weeks; exploratory, few effective blocks',
          'sign':'first minus second; negative count-error difference favours first, positive warning-metric difference favours first',
          'interpretation':'zero-including intervals are inconclusive, not proof of equivalence; no multiplicity-adjusted confirmatory claims',
          'label':'retrospective development follow-up; no prospective issuance or clinical utility established'}
    write_json(RESULTS/'analysis_specification.json',meta)
    additional_comparisons()
    data_quality_detail()
    print('Analysis tables refreshed',flush=True)


def additional_comparisons():
    """Fair four-model workload and rescoring of frozen adapted benchmarks."""
    common_summary=[];costs=[];literature=[];literature_metrics=[]
    benchmark=Path('research_comparisons/literature_based_forecasting_benchmark/runs/20260828r1/predictions_provenance.csv')
    bench=pd.read_csv(benchmark).rename(columns={'origin_date':'origin','target_date':'target','observed_cases':'actual','predicted_cases':'predicted'}) if benchmark.exists() else None
    for h in HORIZONS:
        rf=load_experiment(f'legacy12_rf_h{h}_start2020_all')
        xgb=load_experiment(f'legacy12_xgb_h{h}_start2020_all')
        if rf is None or rf.origin.nunique()!=12:continue
        if bench is not None:
            for method in ['zinszer_arimax','tekle_poisson_dl']:
                b=bench[bench.horizon.eq(h)&bench.method_id.eq(method)&bench.prediction_status.eq('success')].copy()
                # Predictions reused, not refitted. Warning labels/score updated consistently.
                b=b.drop(columns=['threshold']).merge(rf[KEYS+['threshold']],on=KEYS,validate='one_to_one')
                for cluster in ['district','origin']:
                    for r in paired_bootstrap(rf,b,cluster=cluster):literature.append({'comparison':f'corrected_rf_minus_frozen_{method}','horizon':h,**r})
                literature_metrics.append({'model':method,'horizon':h,'fits':'frozen adapted benchmark, not rerun',**metrics(b)})
        sarpath=RESULTS/'sarimax/predictions.csv'
        naivepath=RESULTS/f'legacy12_seasonal_naive_h{h}_predictions.csv'
        if not sarpath.exists() or not naivepath.exists() or xgb is None or xgb.origin.nunique()!=12:continue
        sar=pd.read_csv(sarpath).rename(columns={'forecast_origin_week':'origin','target_week':'target','observed_cases':'actual','predicted_cases':'predicted','surge_threshold':'threshold'})
        sar=sar[sar.horizon.eq(h)]
        models={'rf':rf,'xgb':xgb,'sarimax':sar,'seasonal_naive':pd.read_csv(naivepath)}
        keys=set(map(tuple,rf[KEYS].to_numpy()))
        for f in models.values():keys &= set(map(tuple,f[KEYS].to_numpy()))
        keep=pd.DataFrame(sorted(keys),columns=KEYS)
        for model,f in models.items():
            paired=keep.merge(f,on=KEYS,validate='one_to_one');m=metrics(paired)
            common_summary.append({'model':model,'horizon':h,'maximum_coverage_rows':len(f),'operational_method':'count_threshold',**m})
            for ratio in [1,2,4,8]:costs.append({'model':model,'horizon':h,'cost_fn_to_fp':ratio,
                'weighted_errors':ratio*m['fn']+m['fp'],'weighted_errors_per_100':100*(ratio*m['fn']+m['fp'])/m['n'],
                'f1':m['f1'],'f2':m['f2'],'sensitivity':m['sensitivity'],'specificity':m['specificity'],
                'precision':m['precision'],'tp':m['tp'],'fp':m['fp'],'fn':m['fn'],'tn':m['tn'],'n':m['n']})
    pd.DataFrame(common_summary).to_csv(RESULTS/'four_model_common_sample.csv',index=False)
    pd.DataFrame(costs).to_csv(RESULTS/'four_model_cost_sensitivity.csv',index=False)
    pd.DataFrame(literature).to_csv(RESULTS/'corrected_rf_adapted_literature_comparisons.csv',index=False)
    pd.DataFrame(literature_metrics).to_csv(RESULTS/'frozen_adapted_methods_rescored.csv',index=False)


def data_quality_detail():
    source=pd.read_csv(DATA);source['date']=pd.to_datetime(source.week_start_date)
    dates=pd.date_range('2019-12-30','2025-12-22',freq='W-MON');records=[]
    for d,g in source.groupby('district_id'):
        missing=set(dates)-set(g.date);first,last=g.date.min(),g.date.max()
        records.append({'district_id':d,'first_assembled_week':str(first.date()),'last_assembled_week':str(last.date()),
            'source_absent':len(missing),'leading_absences':sum(t<first for t in missing),
            'internal_absences':sum(first<t<last for t in missing),'trailing_absences':sum(t>last for t in missing),
            'adjudicated_errors':int(g.confirmed_malaria_cases.isna().sum())})
    frame=pd.DataFrame(records);frame.to_csv(RESULTS/'missingness_position_audit.csv',index=False)
    summaries=[]
    for h in HORIZONS:
        path=RESULTS/f'legacy12_rf_h{h}_start2020_all_predictions.csv'
        if path.exists():
            pred=pd.read_csv(path);quality=source[['district_id','week_start_date','facility_reporting_completeness']].rename(columns={'week_start_date':'target','facility_reporting_completeness':'target_completeness'})
            pred=pred.merge(quality,on=['district_id','target'],validate='many_to_one')
            for label,mask in [('under80',pred.target_completeness.lt(80)),('at_least80',pred.target_completeness.ge(80)),('unavailable',pred.target_completeness.isna())]:
                if mask.any():summaries.append({'horizon':h,'target_reporting_completeness':label,**metrics(pred[mask])})
    pd.DataFrame(summaries).to_csv(RESULTS/'reporting_completeness_performance_sensitivity.csv',index=False)
    write_json(RESULTS/'data_quality_interpretation.json',{
        'leading_absences':int(frame.leading_absences.sum()),'internal_absences':int(frame.internal_absences.sum()),
        'trailing_absences':int(frame.trailing_absences.sum()),'source_absent':int(frame.source_absent.sum()),
        '2021_source_absent':0,'2021_adjudicated_errors':2,
        'first_assembled_week_distribution':frame.first_assembled_week.value_counts().sort_index().to_dict(),
        'interpretation':'Leading gaps are predominantly the first three weeks in 72 districts; their cause is not established. Do not attribute them to COVID or district formation without custodian evidence. Completeness strata are descriptive, not a causal or adjusted comparison.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['baselines','tables']);p.add_argument('--expanded',action='store_true');args=p.parse_args()
    RESULTS.mkdir(parents=True,exist_ok=True)
    if args.phase=='baselines':simple_baselines(Study(),expanded=args.expanded)
    else:all_tables()
