"""Bounded corrected RF/XGB surge comparison; no new model families."""
from __future__ import annotations
import json,time,warnings
from dataclasses import replace
from datetime import date,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from app.services.model_validation import build_rolling_origin_folds
from research_extensions_2026_10_round2.common import *
from research_extensions_2026_10_round2.safe_io import write_json

FOLDER=OUT/'surge_checkpoints';FOLDER.mkdir(exist_ok=True)
EXTENDED=ROOT/'research_extensions_2026_10/private/dhis2_2026/extended_dataset.csv'
FREEZE='2025-12-22';CUTS=(.30,.40,.45,.50,.55,.60,.70)

def code_signature():return sha(Path(__file__))

def warning_metrics(f):
    a=f.actual.to_numpy()>f.threshold.to_numpy();b=f.predicted_surge.to_numpy(bool)
    tp,fp,fn,tn=[int(v.sum()) for v in [a&b,~a&b,a&~b,~a&~b]]
    ratio=lambda n,d:n/d if d else np.nan
    scores=f.probability.to_numpy() if 'probability' in f else (f.predicted-f.threshold).to_numpy()
    return {'n':len(f),'districts':f.district_id.nunique(),'target_weeks':f.target.nunique(),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'sensitivity':ratio(tp,tp+fn),'specificity':ratio(tn,tn+fp),'precision':ratio(tp,tp+fp),'f2':ratio(5*tp,5*tp+4*fn+fp),'f1':ratio(2*tp,2*tp+fp+fn),'prevalence':float(a.mean()),'alert_rate':float(b.mean()),'average_precision':float(average_precision_score(a,scores)) if a.any() else np.nan,'roc_auc':float(roc_auc_score(a,scores)) if len(np.unique(a))==2 else np.nan}

def labels(ctx,h):
    """Fast historical-origin references, independently checked later."""
    s=ctx.study;values=np.full(len(s.rows),np.nan);proofs=[]
    for d,rows in s.history.items():
        observed=[r for r in rows if r['confirmed_malaria_cases']!='' and int(r['epi_year'])!=2022]
        ds=np.array([r['week_start_date'] for r in observed]);ys=np.array([float(r['confirmed_malaria_cases']) for r in observed])
        weeks=np.array([int(r['epi_week']) for r in observed]);periods=np.array([date(int(r['epi_year']),12,28).isocalendar().week for r in observed])
        seasons={}
        for w in range(1,54):
            delta=np.abs(np.minimum(weeks,periods)-np.minimum(w,periods));mask=np.minimum(delta,periods-delta)<=2
            seasons[w]=(ds[mask],ys[mask])
        for i in np.flatnonzero((s.ids==d)&np.isfinite(s.y[h])&(s.target_dates[h]<=FREEZE)):
            origin=s.dates[i];limit=cutoff(origin,1);target=s.target_dates[h][i];w=date.fromisoformat(target).isocalendar().week
            sd,sy=seasons[w];n=np.searchsorted(ds,limit,side='right');ns=np.searchsorted(sd,limit,side='right')
            reference=sy[:ns] if ns>=8 else ys[:n]
            if not len(reference):continue
            k=3*(len(reference)-1)//4;threshold=float(np.partition(reference,k)[k]);values[i]=float(s.y[h][i]>threshold)
            proofs.append({'district_id':d,'horizon':h,'origin':origin,'target':target,'reference_cutoff':limit,'threshold':threshold,'seasonal_n':int(ns),'eligible_n':int(n),'label':int(values[i])})
    atomic_csv(pd.DataFrame(proofs),RESULTS/f'classifier_training_label_proofs_h{h}.csv')
    return values

def estimator(model,parameters,group,y):
    config=replace(CONFIGS[model],controlled_threads=3)
    p=tree._build_classifier_pipeline(config,parameters,positive_weight=tree._positive_class_weight(y))
    if group=='none':
        name,transformer,columns=p.named_steps['preprocess'].transformers[0]
        p.named_steps['preprocess'].transformers[0]=(name,transformer,[c for c in columns if not c.startswith(('rainfall_','temperature_','humidity_'))])
    return p

def tune(ctx,h,model,group,y):
    path=RESULTS/f'classifier_tuning_{model}_h{h}_{group}.json'
    contract={'model':model,'horizon':h,'group':group,'data_sha256':ctx.study.signature,'code_sha256':code_signature(),'threads':3}
    if path.exists():
        r=json.loads(path.read_text());assert all(r[k]==v for k,v in contract.items());return r
    fold=build_rolling_origin_folds(ctx.rows,horizon=h)[0];ids=ctx.training(h,fold.origin_date.isoformat(),1);ids=ids[np.isfinite(y[ids])]
    dates=sorted(set(ctx.study.target_dates[h][ids]));hold=set(dates[-min(8,max(2,len(dates)//5)):]);val=ids[np.isin(ctx.study.target_dates[h][ids],list(hold))]
    earliest=min(ctx.study.dates[val]);limit=cutoff(earliest,1)
    train=ids[(~np.isin(ctx.study.target_dates[h][ids],list(hold)))&(ctx.study.target_dates[h][ids]<=limit)]
    assert len(train) and len(val) and max(ctx.study.target_dates[h][train])<=limit
    x=ctx.features(1);candidates=[]
    for index,parameters in enumerate(CONFIGS[model].classifier_parameter_candidates):
        started=time.monotonic()
        try:
            p=estimator(model,parameters,group,y[train]);p.fit(x.iloc[train],y[train].astype(int));prob=tree._positive_probabilities(p,x.iloc[val])
            for cutoff_value in CUTS:
                a=y[val].astype(bool);b=prob>=cutoff_value;tp=int((a&b).sum());fp=int((~a&b).sum());fn=int((a&~b).sum());tn=int((~a&~b).sum())
                candidates.append({'candidate':index,'parameters':parameters,'probability_cutoff':cutoff_value,'f2':5*tp/(5*tp+4*fn+fp) if 5*tp+4*fn+fp else 0.,'sensitivity':tp/(tp+fn) if tp+fn else 0.,'precision':tp/(tp+fp) if tp+fp else 0.,'specificity':tn/(tn+fp) if tn+fp else 0.,'average_precision':float(average_precision_score(a,prob)),'status':'complete','candidate_fit_seconds':time.monotonic()-started})
        except Exception as e:candidates.append({'candidate':index,'parameters':parameters,'status':'failed','error':repr(e)})
    successful=[r for r in candidates if r['status']=='complete'];assert successful,'All classifier candidates failed'
    selected=max(successful,key=lambda r:tuple(r[k] for k in ['f2','sensitivity','average_precision','precision','specificity']))
    result={**contract,'selected_parameters':selected['parameters'],'probability_cutoff':selected['probability_cutoff'],'candidates':candidates,'earliest_inner_origin':earliest,'inner_label_cutoff':limit,'max_inner_training_target':max(ctx.study.target_dates[h][train]),'inner_training_rows':len(train),'inner_validation_rows':len(val),'outer_origin':fold.origin_date.isoformat(),'objective':'F2, recall, AP, precision, specificity; stable existing candidate/cutoff order for exact ties','outer_or_2026_outcomes_used_for_selection':False}
    write_json(path,result);return result

def extended_context():
    ctx=Context.__new__(Context);ctx.study=Study(EXTENDED);ctx.rows=ctx.study.rows
    ctx.index={(r['district_id'],r['week_start_date']):i for i,r in enumerate(ctx.rows)};source=[dict(r) for r in ctx.rows]
    for e in load_case_exclusions():source[ctx.index[e.district_id,date.fromisocalendar(e.epi_year,e.epi_week,1).isoformat()]]['confirmed_malaria_cases']=str(e.original_cases)
    clean=build_clean_case_series(source,district_ids=sorted(ctx.study.history),start_date=date.fromisoformat(min(ctx.study.dates)),end_date=date.fromisoformat(max(ctx.study.dates)))
    ctx.clean=clean.values;ctx.frames={};ctx.proofs={};ctx.code_hash=code_signature()
    assert max(ctx.study.dates)<='2026-06-30'
    return ctx

def run_job(ctx,h,model,group,y,selection,*,frozen=False):
    s=ctx.study;frame_name='later2026' if frozen else 'original12';tag=f'{frame_name}_{model}_h{h}_{group}'
    jobs=[(FREEZE,None)] if frozen else [(f.origin_date.isoformat(),f.target_date.isoformat()) for f in build_rolling_origin_folds(ctx.rows,horizon=h)]
    outputs=[]
    for origin,target in jobs:
        name=tag+'_'+origin;path=FOLDER/(name+'.csv');metadata=path.with_suffix('.json')
        contract={'data_sha256':s.signature,'code_sha256':code_signature(),'parameters':selection['selected_parameters'],'probability_cutoff':selection['probability_cutoff'],'model':model,'group':group,'horizon':h,'evaluation':frame_name,'fit_origin':origin,'threads':3}
        if path.exists() and metadata.exists():
            old=json.loads(metadata.read_text());assert all(old[k]==v for k,v in contract.items()) and old['predictions_sha256']==sha(path);outputs.append(pd.read_csv(path));continue
        limit=FREEZE if frozen else cutoff(origin,1)
        train=np.flatnonzero((s.target_dates[h]<=limit)&np.isfinite(s.y[h])&np.isfinite(y))
        if frozen:
            targets=set(pd.date_range('2026-03-23','2026-06-22',freq='W-MON').strftime('%Y-%m-%d'))
            val=np.flatnonzero(np.isin(s.target_dates[h],list(targets))&np.isfinite(s.y[h]))
        else:val=ctx.validation(h,origin,target)
        assert len(train) and len(val) and max(s.target_dates[h][train])<=limit and max(s.target_dates[h][train])<=FREEZE
        x=ctx.features(1);started=time.monotonic();p=estimator(model,selection['selected_parameters'],group,y[train]);p.fit(x.iloc[train],y[train].astype(int));prob=tree._positive_probabilities(p,x.iloc[val])
        assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
        origins=s.dates[val];targets=s.target_dates[h][val]
        thresholds=[float(s.threshold(str(s.ids[i]),FREEZE if frozen else cutoff(s.dates[i],1),s.target_dates[h][i])['threshold']) for i in val]
        frame=pd.DataFrame({'district_id':s.ids[val],'horizon':h,'origin':origins,'target':targets,'actual':s.y[h][val],'threshold':thresholds,'probability':prob,'probability_cutoff':selection['probability_cutoff'],'predicted_surge':prob>=selection['probability_cutoff']})
        atomic_csv(frame,path);write_json(metadata,{**contract,'status':'complete','train_rows':len(train),'max_training_target':max(s.target_dates[h][train]),'label_cutoff':limit,'validation_rows':len(val),'predictions_sha256':sha(path),'seconds':time.monotonic()-started,'features':'strict O-1; '+group,'training_labels':'historical-origin O-1 Q75','recorded_at_utc':now()});outputs.append(frame)
        print(json.dumps({'completed':name,'seconds':time.monotonic()-started}),flush=True)
        if time.monotonic()>DEADLINE:raise TimeoutError('Surge comparison 90-minute total budget reached')
    return pd.concat(outputs,ignore_index=True)

def frozen_count(ctx,h):
    from research_extensions_2026_10_round2.run_trees import pipeline
    s=ctx.study;path=FOLDER/f'later2026_rf_count_h{h}.csv';meta=path.with_suffix('.json');parameters=json.loads((RESULTS/f'tuning_rf_h{h}_delay1.json').read_text())['selected_parameters']
    contract={'data_sha256':s.signature,'code_sha256':code_signature(),'parameters':parameters,'threads':3,'fit_cutoff':FREEZE}
    if path.exists() and meta.exists():
        r=json.loads(meta.read_text());assert all(r[k]==v for k,v in contract.items()) and sha(path)==r['predictions_sha256'];return pd.read_csv(path)
    ids=np.flatnonzero((s.target_dates[h]<=FREEZE)&np.isfinite(s.y[h]));targets=set(pd.date_range('2026-03-23','2026-06-22',freq='W-MON').strftime('%Y-%m-%d'));val=np.flatnonzero(np.isin(s.target_dates[h],list(targets))&np.isfinite(s.y[h]))
    p=pipeline('rf',parameters,1,threads=3);name,transformer,cols=p.named_steps['preprocess'].transformers[0];p.named_steps['preprocess'].transformers[0]=(name,transformer,[c for c in cols if not c.startswith(('rainfall_','temperature_','humidity_'))]);x=ctx.features(1);started=time.monotonic();p.fit(x.iloc[ids],s.y[h][ids]);prediction=np.maximum(0,p.predict(x.iloc[val]))
    threshold=[float(s.threshold(str(s.ids[i]),FREEZE,s.target_dates[h][i])['threshold']) for i in val]
    frame=pd.DataFrame({'district_id':s.ids[val],'horizon':h,'origin':s.dates[val],'target':s.target_dates[h][val],'actual':s.y[h][val],'predicted':prediction,'threshold':threshold});frame['predicted_surge']=frame.predicted>frame.threshold
    atomic_csv(frame,path);write_json(meta,{**contract,'evaluation':'later2026','model':'rf_count_strict','horizon':h,'group':'none','status':'complete','train_rows':len(ids),'max_training_target':max(s.target_dates[h][ids]),'label_cutoff':FREEZE,'validation_rows':len(val),'predictions_sha256':sha(path),'seconds':time.monotonic()-started});return frame

DEADLINE=float('inf')
def main():
    global DEADLINE
    started=time.monotonic();DEADLINE=started+5400;ctx=Context();extended=None;failed=[]
    write_json(RESULTS/'surge_execution_contract.json',{'recorded_at_utc':now(),'code_sha256':code_signature(),'data_sha256':ctx.study.signature,'extended_sha256':sha(EXTENDED),'fit_cutoff_2026':FREEZE,'threads':3,'maximum_seconds':5400,'expected_outer_checkpoints':153,'candidates_per_selection':3,'cutoffs':CUTS,'source_plan':'SURGE_CLOSURE_PLAN.md','plan_sha256':sha(OUT/'SURGE_CLOSURE_PLAN.md'),'scope_addendum_sha256':sha(OUT/'review_inputs/IDE_SCOPE_AND_CLOSURE_ADDENDUM_2026-10-15.md'),'2026_status':'Previously inspected exploratory later-period evaluation; no model/threshold fitted on 2026 labels'})
    for h in HORIZONS:
        if time.monotonic()>DEADLINE:break
        y=labels(ctx,h)
        for group in ['none','all']:
            for model in ['rf','xgb']:
                if time.monotonic()>DEADLINE:break
                try:
                    selection=tune(ctx,h,model,group,y)
                    run_job(ctx,h,model,group,y,selection)
                    if group=='none':
                        if extended is None:extended=extended_context()
                        ey=np.full(len(extended.rows),np.nan)
                        for i,r in enumerate(ctx.rows):ey[extended.index[r['district_id'],r['week_start_date']]]=y[i]
                        run_job(extended,h,model,group,ey,selection,frozen=True)
                except Exception as e:failed.append({'horizon':h,'model':model,'group':group,'error':repr(e)});print(json.dumps({'failure':failed[-1]}),flush=True)
        if extended is not None and time.monotonic()<=DEADLINE:
            try:frozen_count(extended,h)
            except Exception as e:failed.append({'horizon':h,'model':'rf_count_strict','group':'none','error':repr(e)})
    completed=len(list(FOLDER.glob('*.json')))
    write_json(RESULTS/'surge_execution.json',{'completed_at_utc':now(),'completed_checkpoints':completed,'expected_checkpoints':153,'failed':failed,'unrun_or_failed':153-completed,'complete':completed==153 and not failed,'seconds':time.monotonic()-started,'code_sha256':code_signature(),'actual_exit_code':0})
    print(json.dumps({'surge_checkpoints':completed,'expected':153,'failed_groups':len(failed)}),flush=True)

if __name__=='__main__':main()
