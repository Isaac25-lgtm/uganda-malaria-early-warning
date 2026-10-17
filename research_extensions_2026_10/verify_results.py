"""Independent arithmetic, key, artifact and temporal-lineage checks."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import importlib.metadata
import subprocess
from bisect import bisect_right
from collections import Counter
from datetime import date,datetime,timedelta,timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from sqlalchemy import text
from app import create_app
from app.extensions import db
from app.models import ForecastOutput
from app.services.governance import _forecast_integrity
from app.services.models.pooled_tree_validation import _feature_frame,_chronological_inner_split,POOLED_NUMERIC_FEATURES
from app.services.model_validation import build_rolling_origin_folds,district_threshold_proof
from research_extensions_2026_10.evaluate_2026 import read_frozen_checkpoint,CUTOFF
from research_extensions_2026_10.run_study import OUT,RESULTS,EXPERIMENTS,DATA,HORIZONS,write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(require_complete=False):
    manifest=json.loads((OUT/'baseline/manifest.json').read_text());copies=[];protected=[]
    for entry in manifest['files']:
        relative=entry['path'];frozen=OUT/'baseline'/relative
        assert sha(frozen)==entry['sha256'],f'Frozen copy changed: {relative}'
        copies.append(relative)
        if relative.startswith('data/'):
            assert sha(Path(relative))==entry['sha256'],f'Protected artifact changed: {relative}'
            protected.append(relative)
    external=json.loads((OUT/'baseline/supporting_inputs/manifest.json').read_text())
    for entry in external['files']:
        assert sha(Path(entry['copy']))==entry['sha256']
        assert sha(Path(entry['source']))==entry['sha256'],'Original supplied document changed.'
    source=list(csv.DictReader(DATA.open(encoding='utf-8')))
    truth={(r['district_id'],r['week_start_date']):r['confirmed_malaria_cases'] for r in source}
    extended=list(csv.DictReader((OUT/'private/dhis2_2026/extended_dataset.csv').open(encoding='utf-8')))
    truth_extended={(r['district_id'],r['week_start_date']):r['confirmed_malaria_cases'] for r in extended}
    source_history={}
    for row in source:source_history.setdefault(row['district_id'],[]).append(row)
    historical_thresholds={}
    def threshold_at(district,origin,target):
        key=district,origin,target
        if key not in historical_thresholds:
            history=[r for r in source_history[district] if r['week_start_date']<=origin]
            historical_thresholds[key]=Decimal(district_threshold_proof(history,district,target_epi_week=date.fromisoformat(target).isocalendar().week)['threshold'])
        return historical_thresholds[key]
    extraction=json.loads((OUT/'private/dhis2_2026/manifest.json').read_text())
    assert sha(OUT/'protocol_2026_locked.md')==extraction['protocol_sha256']
    assert datetime.fromisoformat(extraction['protocol_verified_before_extraction_at_utc'])<datetime.fromisoformat(extraction['extracted_at_utc'])
    assert extraction['periods']==[f'2026W{w:02d}' for w in range(1,27)]
    assert extraction['last_requested_week_end']<='2026-06-30'
    for entry in extraction['files']:
        assert sha(OUT/'private/dhis2_2026'/entry['file'])==entry['sha256']
    lineage=json.loads((RESULTS/'dataset_lineage_audit.json').read_text())
    assert sha(OUT/'private/dhis2_2026/extended_dataset.csv')==lineage['extended_sha256']
    assert lineage['historical_predictor_or_observed_outcome_changes']==0
    summaries=pd.read_csv(RESULTS/'master_metrics.csv')
    district=pd.read_csv(RESULTS/'district_metrics.csv')
    checked=[]
    for experiment in summaries[summaries.complete].experiment:
        rows=list(csv.DictReader((RESULTS/f'{experiment}_predictions.csv').open(encoding='utf-8')))
        seen=set();errors=Decimal(0);total=Decimal(0);counts=Counter()
        for r in rows:
            key=(r['district_id'],int(r['horizon']),r['origin'],r['target']);assert key not in seen;seen.add(key)
            origin,target=date.fromisoformat(r['origin']),date.fromisoformat(r['target'])
            assert target-origin==timedelta(weeks=int(r['horizon']))
            observed=truth_extended if target.isocalendar().year>=2026 else truth
            assert Decimal(r['actual'])==Decimal(observed[r['district_id'],r['target']])
            y,p,t=[Decimal(r[c]) for c in ['actual','predicted','threshold']]
            assert p.is_finite() and p>=0 and t.is_finite() and t>=0
            assert t==threshold_at(r['district_id'],r['origin'],r['target']),'Historical threshold used a different reference.'
            errors+=abs(p-y);total+=y;counts[y>t,p>t]+=1
        published=summaries[summaries.experiment.eq(experiment)].iloc[0]
        assert abs(float(errors/total*100)-published.wape)<1e-8
        pairs={'tp':(True,True),'fp':(False,True),'fn':(True,False),'tn':(False,False)}
        for metric,pair in pairs.items():
            assert counts[pair]==int(published[metric])
            assert int(district[district.experiment.eq(experiment)][metric].sum())==counts[pair]
        checked.append({'experiment':experiment,'rows':len(rows),'arithmetic_and_district_sums':'passed'})
    print('Historical experiment arithmetic and past-only thresholds checked:',len(checked),flush=True)
    checkpoint_count=0
    baseline_data_sha=sha(DATA)
    eligible_targets={}
    for horizon in HORIZONS:
        for start in [2020,2021,2022,2023]:
            first=date.fromisocalendar(start,1,1).isoformat()
            eligible_targets[horizon,start]=sorted(r[f'target_week_start_date_h{horizon}'] for r in source
                if r['week_start_date']>=first and r[f'target_cases_h{horizon}']!='')
    for path in (EXPERIMENTS/'predictions').glob('*.json'):
        m=json.loads(path.read_text());assert m['max_training_target']<=m['origin']
        tag=m['experiment'];assert tag==path.stem
        h=int(tag.split('_h')[1].split('_')[0]);start=int(tag.split('_start')[1].split('_')[0])
        model=m['model'];group=tag.rsplit('_',2)[1]
        expected_params=json.loads((RESULTS/f'corrected_tuning_{model}_h{h}.json').read_text())['selected_parameters']
        assert m['parameters']==expected_params
        targets=eligible_targets[h,start];count=bisect_right(targets,m['origin'])
        assert count>0 and count==m['train_rows'] and targets[count-1]==m['max_training_target'],'Training start or cutoff changed.'
        numeric=list(POOLED_NUMERIC_FEATURES)
        if group!='all':numeric=[c for c in numeric if not c.startswith(('rainfall_','temperature_','humidity_')) or c.startswith(group+'_')]
        assert m['features']==numeric+['district_id']
        expected_data=lineage['extended_sha256'] if tag.startswith('holdout_fixed_origin_') else baseline_data_sha
        assert m['data_sha256']==expected_data
        csv_path=path.with_suffix('.csv');f=pd.read_csv(csv_path);assert len(f)==m['validation_rows']
        if m.get('predictions_sha256'):assert sha(csv_path)==m['predictions_sha256']
        checkpoint_count+=1
    holdout_checked=[];threshold_reference={};history={}
    for row in source:
        if row['week_start_date']<=CUTOFF:history.setdefault(row['district_id'],[]).append(row)
    for published in csv.DictReader((RESULTS/'2026_locked_metrics.csv').open(encoding='utf-8')):
        if published['evaluation']!='frozen_history':continue
        h,start=int(published['horizon']),int(published['start'])
        path=RESULTS/f'2026_frozen_history_rf_h{h}_start{start}_predictions.csv'
        params=json.loads((RESULTS/f'corrected_tuning_rf_h{h}.json').read_text())['selected_parameters']
        read_frozen_checkpoint(path,start=start,horizon=h,parameters=params,input_sha=lineage['extended_sha256'],protocol_sha=extraction['protocol_sha256'])
        rows=list(csv.DictReader(path.open(encoding='utf-8')));errors=Decimal(0);total=Decimal(0);counts=Counter()
        for row in rows:
            district,target=row['district_id'],row['target'];y,p,t=[Decimal(row[c]) for c in ['actual','predicted','threshold']]
            assert y==Decimal(truth_extended[district,target])
            key=district,target
            if key not in threshold_reference:
                threshold_reference[key]=Decimal(district_threshold_proof(history[district],district,target_epi_week=date.fromisoformat(target).isocalendar().week)['threshold'])
            assert t==threshold_reference[key],'Frozen pre-2026 threshold reference changed.'
            errors+=abs(p-y);total+=y;counts[y>t,p>t]+=1
        assert abs(float(errors/total*100)-float(published['wape']))<1e-8
        for metric,pair in {'tp':(True,True),'fp':(False,True),'fn':(True,False),'tn':(False,False)}.items():assert counts[pair]==int(published[metric])
        holdout_checked.append({'horizon':h,'start':start,'rows':len(rows),'fitting_threshold_and_arithmetic_checks':'passed'})
    assert len(holdout_checked)==12
    print('Frozen 2026 fitting/threshold/metric checks passed for all 12 scenarios.',flush=True)
    simple_checked=[]
    for published in csv.DictReader((RESULTS/'weekly_simple_baseline_metrics.csv').open(encoding='utf-8')):
        model,h=published['model'],int(published['horizon'])
        path=RESULTS/f'weekly_common_{model}_h{h}_predictions.csv'
        rows=list(csv.DictReader(path.open(encoding='utf-8')));seen=set();errors=Decimal(0);total=Decimal(0);counts=Counter()
        for row in rows:
            district,origin,target=row['district_id'],row['origin'],row['target'];key=district,origin,target
            assert key not in seen;seen.add(key)
            assert date.fromisoformat(target)-date.fromisoformat(origin)==timedelta(weeks=h)
            y,p,t=[Decimal(row[c]) for c in ['actual','predicted','threshold']]
            assert y==Decimal(truth[district,target]) and p.is_finite() and p>=0 and t==threshold_at(district,origin,target)
            observed=[r for r in source_history[district] if r['week_start_date']<=origin and r['confirmed_malaria_cases']!='']
            recent=[Decimal(r['confirmed_malaria_cases']) for r in observed[-4:]]
            seasonal=[Decimal(r['confirmed_malaria_cases']) for r in observed if int(r['epi_week'])==date.fromisoformat(target).isocalendar().week]
            expected=(Decimal(observed[-1]['confirmed_malaria_cases']) if model=='persistence' else
                sum(recent)/len(recent) if model=='recent_mean4' else
                (seasonal[-1] if seasonal else sum(recent)/len(recent)) if model=='seasonal_naive' else
                (sum(seasonal)/len(seasonal) if seasonal else sum(recent)/len(recent)))
            assert abs(p-expected)<Decimal('0.00000001'),'Simple forecast rule did not reproduce.'
            errors+=abs(p-y);total+=y;counts[y>t,p>t]+=1
        assert len(rows)==int(published['n']) and abs(float(errors/total*100)-float(published['wape']))<1e-8
        for metric,pair in {'tp':(True,True),'fp':(False,True),'fn':(True,False),'tn':(False,False)}.items():assert counts[pair]==int(published[metric])
        simple_checked.append({'model':model,'horizon':h,'rows':len(rows),'formula_and_arithmetic_checks':'passed'})
    print('Simple forecast formulas independently reproduced:',len(simple_checked),flush=True)
    tuning_proofs=[]
    for h in HORIZONS:
        fold=build_rolling_origin_folds(source,horizon=h)[0]
        dates=sorted({r[f'target_week_start_date_h{h}'] for r in fold.trainable_rows});n=min(8,max(2,len(dates)//5))
        train,val=_chronological_inner_split(fold.trainable_rows,horizon=h,holdout_dates=set(dates[-n:]))
        cutoff=min(r['week_start_date'] for r in val);latest=max(r[f'target_week_start_date_h{h}'] for r in train)
        assert latest<=cutoff
        tuning_proofs.append({'horizon':h,'earliest_inner_validation_origin':cutoff,'latest_inner_training_target':latest,'gap_violation':False})
    production=json.loads(Path('data/models/production/production_model_manifest.json').read_text());model_files=0;replayed=[]
    load_dotenv();app=create_app()
    with app.app_context():
        db.session.execute(text('SET TRANSACTION READ ONLY'))
        integrity=_forecast_integrity(2025,52,['Baseline/Naive','SARIMAX','Random Forest','XGBoost'])
        assert integrity['forecast_fingerprint_sha256']=='2854b51afc0e06e1284232a584b4a8b91e3ea60e6a79e82452b850396cc144f9'
        stored={(r.model_name,r.horizon_weeks,r.district_id):r.predicted_cases for r in ForecastOutput.query.all()}
        for entry in production['entries']:
            if not entry.get('model_artifact_path'):continue
            path=Path(entry['model_artifact_path']);assert sha(path)==entry['model_signature']['sha256'];model_files+=1
            if entry['model_name'] in ['Random Forest','XGBoost']:
                pipeline=joblib.load(path);rows=[r for r in source if r['week_start_date']==entry['issue_week_start_date']]
                p=np.maximum(0,pipeline.predict(_feature_frame(rows)))
                # Production converts NumPy predictions to Python float before storage.
                # str(np.float32) otherwise shortens precision before Decimal rounding.
                diffs=[abs(Decimal(str(float(value))).quantize(Decimal('.0001'))-stored[entry['model_name'],entry['horizon'],r['district_id']]) for r,value in zip(rows,p)]
                assert max(diffs)==0, (entry['model_name'],entry['horizon'],str(max(diffs)),[(r['district_id'],str(v),str(stored[entry['model_name'],entry['horizon'],r['district_id']])) for r,v,d in zip(rows,p,diffs) if d][:5])
                replayed.append({'model':entry['model_name'],'horizon':entry['horizon'],'forecasts_checked':len(rows),'maximum_difference':str(max(diffs))})
        db.session.rollback()
    counts={prefix:len(list((EXPERIMENTS/'predictions').glob(prefix+'*.csv'))) for prefix in ['legacy12_','weekly_common_','weekly_climate_','holdout_fixed_origin_']}
    if require_complete:assert counts=={'legacy12_':252,'weekly_common_':1104,'weekly_climate_':276,'holdout_fixed_origin_':6},counts
    if require_complete:
        for state_file in ['window_worker_state.json','climate_worker_state.json']:
            state=json.loads((RESULTS/state_file).read_text())
            assert len(state['workers'])==3 and all(w['exit_code']==0 for w in state['workers'].values()),state
        reconstruction=json.loads((RESULTS/'2026_model_reconstruction_audit.json').read_text())
        assert reconstruction['complete'] and reconstruction['passed'] and len(reconstruction['checks'])==12
        for check in reconstruction['checks']:
            assert check['maximum_absolute_difference']<1e-8
            assert sha(Path(check['model_archive']))==check['model_sha256']
            prediction_file=RESULTS/f"2026_frozen_history_rf_h{check['horizon']}_start{check['start']}_predictions.csv"
            assert sha(prediction_file)==check['saved_prediction_sha256']
        overlap_checks=[]
        keys=['district_id','horizon','origin','target']
        for h in HORIZONS:
            for group,prefix in [('all','weekly_common'),('none','weekly_climate')]:
                old=pd.read_csv(RESULTS/f'legacy12_rf_h{h}_start2020_{group}_predictions.csv')
                weekly=pd.read_csv(RESULTS/f'{prefix}_rf_h{h}_start2020_{group}_predictions.csv')
                paired=old.merge(weekly,on=keys,suffixes=('_sparse','_weekly'),validate='one_to_one')
                assert len(paired)>0
                assert np.array_equal(paired.actual_sparse,paired.actual_weekly) and np.array_equal(paired.threshold_sparse,paired.threshold_weekly)
                difference=float(abs(paired.predicted_sparse-paired.predicted_weekly).max())
                assert difference<1e-8,(h,group,difference)
                overlap_checks.append({'horizon':h,'group':group,'shared_rows':len(paired),'maximum_prediction_difference':difference})
    result={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_copies_checked':len(copies),'original_data_artifacts_unchanged':len(protected),
            'supplied_original_documents_unchanged':len(external['files']),
            'production_model_files_hashed':model_files,'loaded_tree_pipeline_replay':replayed,'production_integrity':integrity,
            'experiments_checked':checked,'checkpoint_temporal_checks':checkpoint_count,'inner_tuning_proofs':tuning_proofs,
            'scenario_file_counts':counts,'full_scenario_completion_required':require_complete,'passed':True}
    result['frozen_2026_history_checks']=holdout_checked
    result['2026_protocol_and_extraction_hashes_verified']=True
    result['simple_baseline_checks']=simple_checked
    result['unique_historical_threshold_references_checked']=len(historical_thresholds)
    if require_complete:result['sparse_weekly_overlap_checks']=overlap_checks
    write_json(RESULTS/'result_verification.json',result)
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    (OUT/'environment-final-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items(),key=lambda kv:kv[0].lower()))+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['frozen_copies_checked','original_data_artifacts_unchanged','production_model_files_hashed','checkpoint_temporal_checks','passed']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');args=p.parse_args();main(args.require_complete)
