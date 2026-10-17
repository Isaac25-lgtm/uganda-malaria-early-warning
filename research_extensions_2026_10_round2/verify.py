"""Independent date/reference/Decimal arithmetic and preservation verification."""
import ast,csv,json,math,subprocess,sys,time,io
from decimal import Decimal
from datetime import date,timedelta
import numpy as np
import pandas as pd
from research_extensions_2026_10_round2.common import *

def decimal_metrics(frame):
    records=frame.to_dict('records');actual=[Decimal(str(r['actual'])) for r in records];pred=[Decimal(str(r['predicted'])) for r in records];threshold=[Decimal(str(r['threshold'])) for r in records]
    tp=fp=fn=tn=0
    for a,p,t in zip(actual,pred,threshold):
        if a>t and p>t:tp+=1
        elif a<=t and p>t:fp+=1
        elif a>t and p<=t:fn+=1
        else:tn+=1
    total=sum(actual)
    return {'n':len(records),'wape':float(100*sum(abs(a-p) for a,p in zip(actual,pred))/total) if total else None,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f2':5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else None}

def main():
    start=time.monotonic();s=Study();baseline=json.loads((OUT/'baseline_manifest.json').read_text());checked=0
    support=json.loads((OUT/'support_status.json').read_text())
    assert support['complete'] and all(v==0 for v in {r['step']:r['exit_code'] for r in support['steps']}.values())
    assert json.loads((OUT/'support_recovery_completion.json').read_text())['passed']
    for r in baseline['files']:
        assert sha(ROOT/r['path'])==r['sha256'],'Preserved input/result changed: '+r['path'];checked+=1
    external=json.loads((ROOT/'research_extensions_2026_10/baseline/supporting_inputs/manifest.json').read_text())
    for r in external['files']:
        assert sha(r['source'])==r['sha256'] and sha(r['copy'])==r['sha256']
    model_manifest=json.loads((ROOT/'data/models/production/production_model_manifest.json').read_text());model_files=0
    for r in model_manifest['entries']:
        artifact=r.get('model_artifact_path');signature=r.get('model_signature') or {}
        if artifact and signature.get('sha256'):
            assert sha(ROOT/artifact)==signature['sha256'];model_files+=1
    tuning=[]
    for p in sorted(RESULTS.glob('tuning_*.json')):
        r=json.loads(p.read_text());assert r['data_sha256']==s.signature
        assert r['max_inner_training_target']<=r['inner_label_cutoff']==cutoff(r['earliest_inner_origin'],r['delay_weeks'])
        assert r['selected_parameters'] in [a['parameters'] for a in r['candidates'] if a['status']=='complete']
        tuning.append(p.name)
    history={}
    for d,rows in s.history.items():
        obs=[r for r in rows if r['confirmed_malaria_cases']!='']
        history[d]={'dates':np.array([r['week_start_date'] for r in obs]),'values':np.array([float(r['confirmed_malaria_cases']) for r in obs]),'years':np.array([int(r['epi_year']) for r in obs]),'weeks':np.array([int(r['epi_week']) for r in obs])}
    threshold_cache={}
    def independent_threshold(d,limit,target):
        key=d,limit,date.fromisoformat(target).isocalendar().week
        if key not in threshold_cache:
            hist=history[d];mask=(hist['dates']<=limit)&(hist['years']!=2022)
            values=hist['values'][mask];years=hist['years'][mask];weeks=hist['weeks'][mask]
            periods=np.array([date(int(y),12,28).isocalendar().week for y in years]);gap=np.abs(np.minimum(weeks,periods)-np.minimum(key[2],periods));seasonal=values[np.minimum(gap,periods-gap)<=2]
            ref=np.sort(seasonal if len(seasonal)>=8 else values);threshold_cache[key]=ref[3*(len(ref)-1)//4]
        return threshold_cache[key]
    truth={(r['district_id'],r['week_start_date']):float(r['confirmed_malaria_cases']) for r in s.rows if r['confirmed_malaria_cases']!=''}
    checkpoints=[]
    contract=json.loads((OUT/'execution_contract.json').read_text())
    for path in sorted(CHECKPOINTS.glob('*.csv')):
        r=json.loads(path.with_suffix('.json').read_text());f=pd.read_csv(path);h=r['horizon'];delay=r['delay_weeks'];origin=r['origin'];limit=cutoff(origin,delay)
        assert r['data_sha256']==s.signature and r['predictions_sha256']==sha(path)
        assert r['code_sha256']==contract['tree_code_sha256']
        tuning_record=json.loads((RESULTS/f'tuning_{r["model"]}_h{h}_delay{delay}.json').read_text())
        assert r['parameters']==tuning_record['selected_parameters']
        expected=np.flatnonzero((s.dates<=origin)&(s.target_dates[h]<=limit)&np.isfinite(s.y[h]))
        val=np.flatnonzero((s.dates==origin)&(s.target_dates[h]==r['target'])&np.isfinite(s.y[h]))
        assert len(expected)==r['train_rows'] and max(s.target_dates[h][expected])==r['max_training_target']<=limit
        assert len(f)==len(val)==r['validation_rows'] and set(f.district_id)==set(s.ids[val]) and not f.duplicated(KEYS).any()
        assert np.isfinite(f.predicted).all() and (f.predicted>=0).all()
        for row in f.itertuples():
            assert date.fromisoformat(row.target)-date.fromisoformat(row.origin)==timedelta(weeks=h)
            assert row.actual==truth[row.district_id,row.target]
            assert row.threshold==independent_threshold(row.district_id,limit,row.target)
        checkpoints.append(path.name)
    # Recompute simple forecast formulas without calling the experiment builder.
    baseline_rows=0
    for path in sorted(RESULTS.glob('baselines_*_predictions.csv')):
        delay=int(path.stem.split('_delay')[1].split('_')[0]);f=pd.read_csv(path)
        assert not f.duplicated(['method']+KEYS).any()
        for row in f.itertuples():
            hist=history[row.district_id];idx=np.flatnonzero(hist['dates']<=row.case_cutoff);values=hist['values'][idx];weeks=hist['weeks'][idx];recent=values[-4:];season=values[weeks==date.fromisoformat(row.target).isocalendar().week]
            if row.method=='persistence':expected=values[-1]
            elif row.method=='recent_mean4':expected=sum(Decimal(str(v)) for v in recent)/len(recent)
            elif row.method=='seasonal_naive':expected=season[-1] if len(season) else sum(recent)/len(recent)
            else:expected=sum(season)/len(season) if len(season) else sum(recent)/len(recent)
            assert row.case_cutoff==cutoff(row.origin,delay)
            assert abs(float(expected)-row.predicted)<1e-8
            assert row.actual==truth[row.district_id,row.target]
            assert row.threshold==independent_threshold(row.district_id,row.case_cutoff,row.target)
            baseline_rows+=1
    table=pd.read_csv(RESULTS/'EQUAL_INFORMATION_BENCHMARKS.csv');arithmetic=[]
    for row in table.itertuples():
        if row.method in ['rf','xgb']:f=pd.read_csv(RESULTS/f'{row.evaluation}_{row.method}_h{row.horizon}_delay{row.delay}_predictions.csv')
        elif row.method.startswith('direct_'):
            f=pd.read_csv(RESULTS/f'direct_ridge_h{row.horizon}_predictions.csv');f=f[f.method.eq(row.method)]
            assert (f.max_training_label<=f.case_cutoff).all()
            for item in f.itertuples():assert item.threshold==independent_threshold(item.district_id,item.case_cutoff,item.target)
        else:
            f=pd.read_csv(RESULTS/f'baselines_{row.evaluation}_h{row.horizon}_delay{row.delay}_predictions.csv');f=f[f.method.eq(row.method)]
        independent=decimal_metrics(f)
        for k,v in independent.items():
            recorded=getattr(row,k)
            if v is None:assert pd.isna(recorded)
            else:assert abs(float(recorded)-v)<1e-8,(row.method,row.horizon,k,recorded,v)
        assert independent['tp']+independent['fp']+independent['fn']+independent['tn']==len(f)
        arithmetic.append({'evaluation':row.evaluation,'method':row.method,'horizon':row.horizon,'delay':row.delay,'passed':True})
    # Rebuild representative ridge fits from source rows with augmented least
    # squares, separately from the runner's normal-equation implementation.
    ridge_checks=[]
    climate=[f'{name}_lag_{k}' for name in ['rainfall','temperature','humidity'] for k in [4,8,12]]
    source_index={(r['district_id'],r['week_start_date']):r for r in s.rows}
    def independent_ridge_row(d,origin,target,seasonal):
        o=date.fromisoformat(origin);t=date.fromisoformat(target);values=[]
        for k in [1,2,4,8,12]:values.append(truth.get((d,(o-timedelta(weeks=k)).isoformat()),np.nan))
        for k,minimum in [(4,3),(8,6),(12,9)]:
            past=[truth[(d,(o-timedelta(weeks=j)).isoformat())] for j in range(1,k+1) if (d,(o-timedelta(weeks=j)).isoformat()) in truth]
            values.append(float(np.mean(past)) if len(past)>=minimum else np.nan)
        record=source_index[d,origin]
        values.extend(float(record[c]) if record[c]!='' else np.nan for c in climate)
        if seasonal:
            values.append(truth.get((d,(t-timedelta(weeks=52)).isoformat()),np.nan))
            phase=(t-date(2000,1,3)).days/7/52.1775
            values.extend(f(2*np.pi*k*phase) for k in [1,2] for f in [np.sin,np.cos])
        return values
    ridge_spec=json.loads((RESULTS/'ridge_specification.json').read_text())
    assert ridge_spec['data_sha256']==s.signature and ridge_spec['code_sha256']==sha(OUT/'classical.py')
    for h in HORIZONS:
        frame=pd.read_csv(RESULTS/f'direct_ridge_h{h}_predictions.csv')
        assert not frame.duplicated(['method']+KEYS).any()
        eligible=pd.read_csv(RESULTS/f'baselines_weekly92_h{h}_delay1_predictions.csv')
        eligible_keys=set(map(tuple,eligible[eligible.method.eq('persistence')][KEYS].to_numpy()))
        for method,g in frame.groupby('method'):
            assert set(map(tuple,g[KEYS].to_numpy()))==eligible_keys
        for d,g in frame.groupby('district_id'):
            dates=np.sort(s.target_dates[h][(s.ids==d)&np.isfinite(s.y[h])])
            counts=np.searchsorted(dates,g.case_cutoff.to_numpy(),side='right')
            assert (counts==g.train_rows.to_numpy()).all()
            assert (dates[counts-1]==g.max_training_label.to_numpy()).all()
        for row in frame.itertuples():
            assert row.actual==truth[row.district_id,row.target]
            assert row.case_cutoff==cutoff(row.origin,1)
            assert date.fromisoformat(row.target)-date.fromisoformat(row.origin)==timedelta(weeks=h)
            assert row.predicted>=0 and np.isfinite(row.predicted) and row.status=='complete'
        for method,g in frame.groupby('method'):
            row=g.sort_values(['target','district_id']).iloc[0];d=row.district_id
            ids=np.flatnonzero((s.ids==d)&(s.target_dates[h]<=row.case_cutoff)&np.isfinite(s.y[h]))
            assert len(ids)==row.train_rows and max(s.target_dates[h][ids])==row.max_training_label
            seasonal=method=='direct_seasonal_arx_ridge'
            train=np.array([independent_ridge_row(d,s.dates[i],s.target_dates[h][i],seasonal) for i in ids])
            val=np.array(independent_ridge_row(d,row.origin,row.target,seasonal))
            med=pd.DataFrame(train).median().fillna(0).to_numpy()
            train=np.where(np.isfinite(train),train,med);val=np.where(np.isfinite(val),val,med)
            center=train.mean(axis=0);scale=train.std(axis=0);scale[scale==0]=1
            train=(train-center)/scale;val=(val-center)/scale;y=s.y[h][ids];ym=y.mean()
            coef=np.linalg.lstsq(np.vstack([train,np.eye(train.shape[1])]),np.r_[y-ym,np.zeros(train.shape[1])],rcond=None)[0]
            prediction=max(0,float(val@coef+ym));difference=abs(prediction-row.predicted)
            assert difference<1e-7,(method,h,difference)
            ridge_checks.append({'method':method,'horizon':h,'district_id':d,'origin':row.origin,'train_rows':len(ids),'absolute_prediction_difference':difference,'passed':True})
    write_json(RESULTS/'independent_ridge_replays.json',ridge_checks)
    surge_source=table.copy()
    extra=pd.read_csv(RESULTS/'SURGE_CANDIDATE_METRICS.csv')
    extra=extra[extra.policy_type.eq('direct_classifier')|extra.evaluation.eq('later2026')].copy()
    extra['evaluation']=extra.evaluation.replace({'original12':'legacy12'})
    surge_source=pd.concat([surge_source,extra],ignore_index=True)
    surge_checks=0
    surge=pd.read_csv(RESULTS/'SURGE_FIRST_COMPARISON.csv')
    for row in surge.itertuples():
        source=surge_source[(surge_source.evaluation==row.evaluation)&(surge_source.horizon==row.horizon)&(surge_source.delay==row.delay)]
        if row.policy_type=='non_predictive_reference':
            ref=source[source.complete].iloc[0];positive=int(ref.tp+ref.fn);negative=int(ref.fp+ref.tn)
            expected=(positive,negative,0,0) if row.method=='always_alert_reference' else (0,0,positive,negative)
            assert (row.tp,row.fp,row.fn,row.tn)==expected and row.n==positive+negative
        else:
            ref=source[source.method.eq(row.method)].iloc[0]
            for key in ['n','tp','fp','fn','tn']:assert getattr(row,key)==ref[key]
        denominator=5*row.tp+4*row.fn+row.fp
        if denominator:assert abs(row.f2-5*row.tp/denominator)<1e-12
        assert abs(row.alerts_per_target_week-(row.tp+row.fp)/row.target_weeks)<1e-10
        assert abs(row.no_skill_average_precision-(row.tp+row.fn)/row.n)<1e-12
        surge_checks+=1
    for row in pd.read_csv(RESULTS/'SEPARATE_WARNING_AND_COUNT_CHOICES.csv').itertuples():
        candidates=surge_source[(surge_source.evaluation==row.evaluation)&(surge_source.horizon==row.horizon)&(surge_source.delay==row.delay)&surge_source.complete].to_dict('records')
        warning=min(candidates,key=lambda r:(-r['f2'],-r['sensitivity'],-r['precision'],r['method']))
        counts=[r for r in candidates if pd.notna(r.get('wape'))];count=min(counts,key=lambda r:(r['wape'],r['method'])) if counts else None
        assert row.warning_point_choice_among_tested_policies==warning['method']
        assert row.count_point_choice==count['method'] if count is not None else pd.isna(row.count_point_choice)
    seasonal_checks=[]
    seasonal_registry=json.loads((RESULTS/'seasonal_registry.json').read_text());assert len(seasonal_registry)==144
    for record in seasonal_registry:
        assert record['data_sha256']==s.signature and record['code_sha256']==sha(OUT/'classical.py')
        if record['status']=='complete' and record['method']!='adapted_arimax':assert record['converged'] is True
    seasonal=pd.read_csv(RESULTS/'seasonal_subset_predictions.csv')
    assert not seasonal.duplicated(['method']+KEYS).any()
    for row in seasonal.itertuples():
        assert row.actual==truth[row.district_id,row.target]
        assert row.threshold==independent_threshold(row.district_id,cutoff(row.origin,1),row.target)
        assert row.predicted>=0 and np.isfinite(row.predicted)
        assert date.fromisoformat(row.target)-date.fromisoformat(row.origin)==timedelta(weeks=row.horizon)
    for row in pd.read_csv(RESULTS/'seasonal_subset_metrics.csv').itertuples():
        allh=seasonal[seasonal.horizon.eq(row.horizon)]
        frame=allh[allh.method.eq(row.method)]
        if row.coverage_type=='plain_annual_common_rows':
            common=allh[allh.method.eq('plain_ar1')][KEYS].merge(allh[allh.method.eq('annual_sarima')][KEYS],on=KEYS,validate='one_to_one')
            frame=frame.merge(common,on=KEYS,validate='one_to_one')
        for k,value in decimal_metrics(frame).items():
            recorded=getattr(row,k)
            assert pd.isna(recorded) if value is None else abs(float(recorded)-value)<1e-8
        seasonal_checks.append({'horizon':row.horizon,'method':row.method,'coverage':row.coverage_type,'rows':len(frame),'passed':True})
    audit=json.loads((RESULTS/'fresh_h8_reproduction.json').read_text());assert audit['passed']
    assert json.loads((OUT/'audit_final_execution.json').read_text())['exit_code']==0
    from research_extensions_2026_10_round2.run_trees import pipeline
    ctx=Context();new_replays=[]
    for model in ['rf','xgb']:
        candidates=sorted(CHECKPOINTS.glob(f'{model}_h8_delay1_*.csv'))
        if not candidates:continue
        path=candidates[0];r=json.loads(path.with_suffix('.json').read_text())
        train=ctx.training(8,r['origin'],1);val=ctx.validation(8,r['origin'],r['target']);x=ctx.features(1)
        estimator=pipeline(model,r['parameters'],1,threads=contract['new_tree_fit_threads'])
        estimator.fit(x.iloc[train],ctx.study.y[8][train]);pred=np.maximum(0,estimator.predict(x.iloc[val]))
        serial=pd.read_csv(io.StringIO(pd.DataFrame({'predicted':pred}).to_csv(index=False))).predicted.to_numpy()
        stored=pd.read_csv(path).set_index('district_id');expected=np.array([stored.loc[d,'predicted'] for d in ctx.study.ids[val]])
        delta=float(np.abs(serial-expected).max());assert delta<1e-8,(model,delta)
        new_replays.append({'model':model,'horizon':8,'delay':1,'origin':r['origin'],'rows':len(val),'threads':contract['new_tree_fit_threads'],'maximum_csv_difference':delta,'passed':True})
    write_json(RESULTS/'new_model_replays.json',new_replays)
    from research_extensions_2026_10_round2.verify_surge import main as verify_surge
    surge_verification=verify_surge(s,truth,independent_threshold,ctx)
    # Tests execute in a real child process; shell stderr handling cannot hide failures.
    with (OUT/'tests_final.stdout.log').open('w',encoding='utf-8') as stdout,(OUT/'tests_final.stderr.log').open('w',encoding='utf-8') as stderr:
        p=subprocess.run([sys.executable,'-m','unittest','research_extensions_2026_10_round2.test_round2','tests.test_research_corrections','tests.test_research_statistics','tests.test_holdout_checkpoints','-v'],stdout=stdout,stderr=stderr)
    assert p.returncode==0,'Scientific test run failed; inspect tests_final.stderr.log'
    parsed=[]
    for path in OUT.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'));parsed.append(path.name)
    completion=json.loads((RESULTS/'analysis_completion.json').read_text())
    write_json(RESULTS/'verification.json',{'at_utc':now(),'passed':True,'preserved_files_checked':checked,'production_model_files_checked':model_files,'original_external_documents_checked':len(external['files']),'tree_checkpoints_checked':len(checkpoints),'independent_reference_keys_checked':len(threshold_cache),'simple_forecast_rows_formula_checked':baseline_rows,'scenario_arithmetic_checks':arithmetic,'seasonal_subset_arithmetic_checks':seasonal_checks,'independent_ridge_replays':ridge_checks,'surge_analysis_rows_checked':surge_checks,'surge_classifier_verification':surge_verification,'tuning_contracts_checked':tuning,'fresh_new_model_replays':new_replays,'scientific_tests':{'new_tests':11,'repeated_existing_tests':18,'total':29,'actual_exit_code':p.returncode},'python_files_parsed':parsed,'full_experiment_completion':completion['status']=='complete' and surge_verification['full_defined_completion'],'source_snapshot_sha256':s.signature,'seconds':time.monotonic()-start,'engineering_test_status':'Production code unchanged, verified by hashes. Previous 471 distinct tests and 74 live HTTP checks retained as prior evidence, not represented as freshly rerun.'})
    print(json.dumps({'passed':True,'preserved_files':checked,'checkpoints':len(checkpoints),'simple_formula_rows':baseline_rows,'new_tests':11,'existing_tests_repeated':18,'complete':completion['status']=='complete'}),flush=True)

if __name__=='__main__':main()
