"""Equal-case-information experiments, frozen earliest-fold tuning and resumable fits."""
from __future__ import annotations
import argparse,json,time,warnings
from dataclasses import replace
from datetime import timedelta
import numpy as np
import pandas as pd
from app.services.model_validation import build_rolling_origin_folds
from research_extensions_2026_10_round2.common import *

def pipeline(model,parameters,delay,threads=3):
    p=tree._build_pipeline(replace(CONFIGS[model],controlled_threads=threads),parameters)
    if delay==0:
        name,transformer,cols=p.named_steps['preprocess'].transformers[0]
        p.named_steps['preprocess'].transformers[0]=(name,transformer,list(cols)+['cases_current'])
    return p

def tune(ctx,model,h,delay):
    path=RESULTS/f'tuning_{model}_h{h}_delay{delay}.json'
    signature={'data_sha256':ctx.study.signature,'code_sha256':ctx.code_hash,'model':model,'horizon':h,'delay_weeks':delay}
    if path.exists():
        old=json.loads(path.read_text())
        assert all(old[k]==v for k,v in signature.items())
        return old['selected_parameters']
    fold=build_rolling_origin_folds(ctx.rows,horizon=h)[0]
    ids=ctx.training(h,fold.origin_date.isoformat(),delay)
    dates=sorted(set(ctx.study.target_dates[h][ids]));hold=set(dates[-min(8,max(2,len(dates)//5)):])
    val=ids[np.isin(ctx.study.target_dates[h][ids],list(hold))]
    earliest=min(ctx.study.dates[val]);limit=cutoff(earliest,delay)
    train=ids[(~np.isin(ctx.study.target_dates[h][ids],list(hold)))&(ctx.study.target_dates[h][ids]<=limit)]
    assert len(train) and len(val) and max(ctx.study.target_dates[h][train])<=limit
    x=ctx.features(delay);scores=[]
    for parameters in CONFIGS[model].parameter_candidates:
        started=time.monotonic()
        try:
            p=pipeline(model,parameters,delay)
            p.fit(x.iloc[train],ctx.study.y[h][train]);prediction=p.predict(x.iloc[val])
            score=float(np.abs(prediction-ctx.study.y[h][val]).mean())
            scores.append({'parameters':parameters,'inner_validation_mae':score,'status':'complete','seconds':time.monotonic()-started})
        except Exception as e:scores.append({'parameters':parameters,'status':'failed','error':str(e),'seconds':time.monotonic()-started})
    successful=[s for s in scores if s['status']=='complete']
    if not successful:raise ValueError('All tuning candidates failed')
    selected=min(successful,key=lambda s:s['inner_validation_mae'])['parameters']
    write_json(path,{**signature,'selected_parameters':selected,'candidates':scores,'earliest_inner_origin':earliest,'inner_label_cutoff':limit,'max_inner_training_target':max(ctx.study.target_dates[h][train]),'inner_training_rows':len(train),'inner_validation_rows':len(val),'outer_origin':fold.origin_date.isoformat(),'selection':'minimum MAE; stable first-candidate tie','recorded_at_utc':now()})
    return selected

def fit(ctx,model,h,delay,origin,target,parameters):
    tag=f'{model}_h{h}_delay{delay}_{origin}';path=CHECKPOINTS/f'{tag}.csv'
    expected={'data_sha256':ctx.study.signature,'code_sha256':ctx.code_hash,'parameters':parameters,'model':model,'horizon':h,'delay_weeks':delay,'origin':origin,'target':target}
    old=valid_checkpoint(path,expected)
    if old is not None:return
    train=ctx.training(h,origin,delay);val=ctx.validation(h,origin,target);x=ctx.features(delay)
    assert len(train) and len(val)
    started=time.monotonic();p=pipeline(model,parameters,delay)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');p.fit(x.iloc[train],ctx.study.y[h][train]);pred=np.maximum(0,p.predict(x.iloc[val]))
    proofs=[ctx.threshold(str(ctx.study.ids[i]),origin,target,delay) for i in val]
    frame=pd.DataFrame({'district_id':ctx.study.ids[val],'horizon':h,'origin':origin,'target':target,'actual':ctx.study.y[h][val],'predicted':pred,'threshold':[float(a['threshold']) for a in proofs],'threshold_fallback':[not a['seasonal_window_used'] for a in proofs]})
    assert np.isfinite(pred).all()
    atomic_csv(frame,path)
    write_json(path.with_suffix('.json'),{**expected,'validation_rows':len(frame),'train_rows':len(train),'max_training_target':max(ctx.study.target_dates[h][train]),'label_cutoff':cutoff(origin,delay),'predictor_case_cutoff':cutoff(origin,max(1,delay)) if delay else origin,'runtime_seconds':time.monotonic()-started,'predictions_sha256':sha(path),'status':'complete','recorded_at_utc':now()})
    print(json.dumps({'complete':tag,'seconds':round(time.monotonic()-started,2)}),flush=True)

def jobs(ctx,h):
    legacy=[(f.origin_date.isoformat(),f.target_date.isoformat()) for f in build_rolling_origin_folds(ctx.rows,horizon=h)]
    weekly=[((t-timedelta(weeks=h)).date().isoformat(),t.date().isoformat()) for t in pd.date_range('2024-03-25','2025-12-22',freq='W-MON')]
    # Complete the highest-priority weekly strict RF before secondary sensitivities.
    result=[('rf',1,o,t) for o,t in sorted(set(weekly+legacy))]
    result += [(m,d,o,t) for d in [0,2,1] for m in ['rf','xgb'] if not (d==1 and m=='rf') for o,t in legacy]
    return result

def main(h):
    ctx=Context();planned=jobs(ctx,h);started=time.monotonic();failed=[];completed=0
    for model,delay,origin,target in planned:
        if time.monotonic()-started>3*3600:break
        try:fit(ctx,model,h,delay,origin,target,tune(ctx,model,h,delay));completed+=1
        except Exception as e:
            failed.append({'model':model,'delay':delay,'origin':origin,'error':f'{type(e).__name__}: {e}'})
            print(json.dumps({'failed':failed[-1]}),flush=True)
        write_json(RESULTS/f'tree_status_h{h}.json',{'planned':len(planned),'completed':completed,'failed':failed,'unrun':len(planned)-completed-len(failed),'seconds':time.monotonic()-started,'at_utc':now()})
    write_json(RESULTS/f'tree_status_h{h}.json',{'planned':len(planned),'completed':completed,'failed':failed,'unrun':len(planned)-completed-len(failed),'seconds':time.monotonic()-started,'at_utc':now(),'worker_finished':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--horizon',type=int,choices=HORIZONS,required=True);main(p.parse_args().horizon)
