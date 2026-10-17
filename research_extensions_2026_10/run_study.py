"""Checkpointed local follow-up, never writes production or contacts DHIS2.

Run phases separately: python -m research_extensions_2026_10.run_study core|expanded|summarize
All outcome eligibility is horizon aware. New results remain development evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import warnings
import os
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from app.services.models import pooled_tree_validation as tree
from app.services.models.random_forest import RANDOM_FOREST_CONFIG
from app.services.models.xgboost_model import XGBOOST_CONFIG
from app.services.model_validation import build_rolling_origin_folds, district_threshold_proof

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_extensions_2026_10"
RESULTS = OUT / "results"
EXPERIMENTS = OUT / "experiments"
DATA = ROOT / "data/modelling/modelling_dataset_2020_2025.csv"
SEED = 20261015
HORIZONS = (4, 8, 12)
CONFIGS = {"rf": RANDOM_FOREST_CONFIG, "xgb": XGBOOST_CONFIG}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_name(path.name+'.next')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    os.replace(temporary,path)


def metrics(frame):
    y, p, threshold = [frame[c].to_numpy(float) for c in ("actual", "predicted", "threshold")]
    a, b = y > threshold, p > threshold
    tp, fp, fn, tn = [int(x.sum()) for x in (a & b, ~a & b, a & ~b, ~a & ~b)]
    divide = lambda n, d: n / d if d else None
    error = p-y
    r2 = 1 - np.square(error).sum()/np.square(y-y.mean()).sum() if len(y)>1 and np.square(y-y.mean()).sum() else None
    nonzero = y != 0
    return {"n": len(y), "districts": frame.district_id.nunique(), "target_weeks": frame.target.nunique(),
            "wape": divide(100*np.abs(error).sum(), y.sum()), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())), "r2": r2,
            "mape": float(np.mean(np.abs(error[nonzero]/y[nonzero]))*100) if nonzero.any() else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "prevalence": float(a.mean()),
            "sensitivity": divide(tp,tp+fn), "specificity": divide(tn,tn+fp),
            "precision": divide(tp,tp+fp), "f1": divide(2*tp,2*tp+fp+fn),
            "f2": divide(5*tp,5*tp+4*fn+fp), "alert_rate": float(b.mean()),
            "average_precision": float(average_precision_score(a,p-threshold)) if a.any() else None,
            "roc_auc": float(roc_auc_score(a,p-threshold)) if len(np.unique(a))==2 else None}


class Study:
    def __init__(self, dataset_path=DATA):
        self.rows = list(csv.DictReader(dataset_path.open(encoding="utf-8", newline="")))
        self.X = tree._feature_frame(self.rows)
        self.dates = np.array([r["week_start_date"] for r in self.rows])
        self.ids = np.array([r["district_id"] for r in self.rows])
        self.y = {h: pd.to_numeric([r[f"target_cases_h{h}"] for r in self.rows], errors="coerce").astype(float) for h in HORIZONS}
        self.target_dates = {h: np.array([r[f"target_week_start_date_h{h}"] for r in self.rows]) for h in HORIZONS}
        self.history = {d: [r for r in self.rows if r['district_id']==d] for d in sorted(set(self.ids))}
        self.signature = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        self.threshold_cache = {}
        self.parameters = {}
        RESULTS.mkdir(parents=True, exist_ok=True)
        EXPERIMENTS.mkdir(parents=True, exist_ok=True)

    def tune(self, model, h):
        path = RESULTS / f"corrected_tuning_{model}_h{h}.json"
        if path.exists():
            result=json.loads(path.read_text())
        else:
            folds=build_rolling_origin_folds(self.rows, horizon=h)
            result=tree._select_horizon_parameters(CONFIGS[model],folds)
            result['protocol']='earliest outer fold, embargoed at earliest inner validation origin; fixed thereafter'
            result['data_sha256']=self.signature
            write_json(path,result)
        self.parameters[model,h]=result['selected_parameters']
        return result['selected_parameters']

    def threshold(self, district, origin, target):
        key=district,origin,target
        if key not in self.threshold_cache:
            history=[r for r in self.history[district] if r['week_start_date']<=origin]
            proof=district_threshold_proof(history,district,target_epi_week=date.fromisoformat(target).isocalendar().week)
            self.threshold_cache[key]=proof
        return self.threshold_cache[key]

    def fit(self, model, h, origin, target, *, start=2020, group='all', protocol='legacy12'):
        tag=f'{protocol}_{model}_h{h}_start{start}_{group}_{origin}'
        path=EXPERIMENTS/'predictions'/f'{tag}.csv'
        metadata_path=path.with_suffix('.json')
        importance_path=EXPERIMENTS/'importance'/f'{tag}.json'
        importance_required=protocol=='legacy12' and group=='all' and model=='rf'
        if path.exists() and metadata_path.exists() and (not importance_required or importance_path.exists()):
            metadata=json.loads(metadata_path.read_text())
            if metadata['data_sha256']!=self.signature or metadata['parameters']!=self.parameters[model,h]:
                raise ValueError('Checkpoint signature/parameters changed: '+tag)
            cached=pd.read_csv(path)
            if len(cached)!=metadata['validation_rows'] or cached.duplicated(['district_id','horizon','origin','target']).any():
                raise ValueError('Incomplete/duplicate checkpoint: '+tag)
            if metadata.get('predictions_sha256') and metadata['predictions_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError('Checkpoint bytes changed: '+tag)
            return cached
        train=np.flatnonzero((self.dates>=date.fromisocalendar(start,1,1).isoformat()) & (self.dates<=origin) & (self.target_dates[h]<=origin) & np.isfinite(self.y[h]))
        val=np.flatnonzero((self.dates==origin) & (self.target_dates[h]==target) & np.isfinite(self.y[h]))
        if not len(train) or not len(val):raise ValueError(f'Empty train/validation {tag}')
        numeric=list(tree.POOLED_NUMERIC_FEATURES)
        if group != 'all':numeric=[c for c in numeric if not c.startswith(('rainfall_','temperature_','humidity_')) or c.startswith(group+'_')]
        limits_path=OUT/'compute_limits.json'
        if protocol=='weekly_climate':limits_path=OUT/'climate_compute_limits.json'
        limits=json.loads(limits_path.read_text()) if limits_path.exists() else {}
        threads=int(limits.get('expanded_threads' if protocol=='weekly_common' else 'core_threads',os.environ.get('RESEARCH_THREADS','6')))
        if protocol=='weekly_common':threads=int(limits.get('expanded_threads_by_horizon',{}).get(str(h),threads))
        if protocol=='weekly_climate':threads=int(limits.get('threads_by_horizon',{}).get(str(h),1))
        config=replace(CONFIGS[model],controlled_threads=threads)
        pipeline=tree._build_pipeline(config,self.parameters[model,h])
        pipeline.named_steps['preprocess'].transformers[0]=('numeric',pipeline.named_steps['preprocess'].transformers[0][1],numeric)
        started=time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pipeline.fit(self.X.iloc[train],self.y[h][train])
            pred=np.maximum(0,pipeline.predict(self.X.iloc[val]))
        proofs=[self.threshold(str(self.ids[i]),origin,target) for i in val]
        result=pd.DataFrame({'district_id':self.ids[val],'horizon':h,'origin':origin,'target':target,
                             'actual':self.y[h][val],'predicted':pred,
                             'threshold':[float(p['threshold']) for p in proofs],
                             'threshold_fallback':[not p['seasonal_window_used'] for p in proofs],
                             'seasonal_history_count':[p['seasonal_history_count'] for p in proofs]})
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.next');result.to_csv(temporary,index=False);os.replace(temporary,path)
        write_json(path.with_suffix('.json'),{'experiment':tag,'model':model,'features':numeric+['district_id'],'parameters':self.parameters[model,h],
                   'train_rows':len(train),'validation_rows':len(val),'max_training_target':max(self.target_dates[h][train]),
                   'origin':origin,'data_sha256':self.signature,'runtime_seconds':time.monotonic()-started,
                   'controlled_threads':threads,'predictions_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'complete'})
        if protocol=='legacy12' and group=='all' and model=='rf':
            self.importance(pipeline,val,result,path.stem)
        print(json.dumps({'complete':tag,'seconds':round(time.monotonic()-started,2),'n':len(val)}),flush=True)
        return result

    def importance(self,pipeline,val,result,tag):
        rng=np.random.default_rng(SEED); x=self.X.iloc[val].copy(); baseline=np.abs(result.predicted.to_numpy()-result.actual.to_numpy()).mean();out=[]
        for group in ['cases','rainfall','temperature','humidity','climate','time','district']:
            cols=[c for c in x if (c.startswith(group+'_') or (group=='climate' and c.startswith(('rainfall_','temperature_','humidity_'))) or (group=='time' and c in ['epi_week','continuous_week_index']) or (group=='district' and c=='district_id'))]
            values=[]
            for repeat in range(3):
                altered=x.copy();order=rng.permutation(len(x));altered.loc[:,cols]=x.iloc[order][cols].to_numpy()
                values.append(float(np.abs(pipeline.predict(altered)-result.actual.to_numpy()).mean()-baseline))
            out.append({'group':group,'mean_mae_increase':float(np.mean(values)),'repeats':3,'permutation_scope':'district rows within one forecast origin; diagnostic, not causal'})
        write_json(EXPERIMENTS/'importance'/f'{tag}.json',out)

    def core(self):
        for h in HORIZONS:
            folds=build_rolling_origin_folds(self.rows,horizon=h)
            for model in ['rf','xgb']:
                self.tune(model,h)
                groups=['all','none','rainfall','temperature','humidity'] if model=='rf' else ['all','none']
                for group in groups:
                    for f in folds:self.fit(model,h,f.origin_date.isoformat(),f.target_date.isoformat(),group=group)
        self.quality()

    def expanded(self, horizons=HORIZONS):
        targets=[];t=date(2024,3,25)
        while t<=date(2025,12,22):targets.append(t);t+=timedelta(weeks=1)
        for h in horizons:
            self.tune('rf',h)
            for start in [2020,2021,2022,2023]:
                for t in targets:
                    o=t-timedelta(weeks=h)
                    self.fit('rf',h,o.isoformat(),t.isoformat(),start=start,protocol='weekly_common')
        self.quality()

    def quality(self):
        r=pd.DataFrame(self.rows);r['epi_year']=r.epi_year.astype(int)
        observed=r[['district_id','epi_year','epi_week']].copy();observed['assembled']=1
        frame=[]
        for d in sorted(set(self.ids)):
            for y in range(2020,2026):
                for w in range(1,date(y,12,28).isocalendar().week+1):frame.append((d,y,str(w)))
        q=pd.DataFrame(frame,columns=['district_id','epi_year','epi_week']);q['epi_week']=q.epi_week.astype(int);observed['epi_week']=observed.epi_week.astype(int)
        q=q.merge(observed,how='left');q['source_absent']=q.assembled.isna().astype(int)
        errors=r[r.confirmed_malaria_cases.eq('')][['district_id','epi_year','epi_week']].copy();errors['epi_week']=errors.epi_week.astype(int);errors['confirmed_error']=1
        q=q.merge(errors,how='left');q['confirmed_error']=q.confirmed_error.fillna(0).astype(int)
        q.to_csv(RESULTS/'missingness_district_week.csv',index=False)
        q.groupby(['district_id','epi_year'])[['source_absent','confirmed_error']].sum().to_csv(RESULTS/'missingness_district_year.csv')
        q.groupby('epi_year')[['source_absent','confirmed_error']].sum().to_csv(RESULTS/'missingness_year.csv')

    def weekly_ablation(self,horizons=HORIZONS):
        targets=[];t=date(2024,3,25)
        while t<=date(2025,12,22):targets.append(t);t+=timedelta(weeks=1)
        for h in horizons:
            self.tune('rf',h)
            for t in targets:self.fit('rf',h,(t-timedelta(weeks=h)).isoformat(),t.isoformat(),group='none',protocol='weekly_climate')


def summarize():
    summary=[];folds=[];districts=[]
    paths=list((EXPERIMENTS/'predictions').glob('*.csv'))
    grouped={}
    for p in paths:
        name=p.stem.rsplit('_',1)[0];grouped.setdefault(name,[]).append(p)
    for name,files in sorted(grouped.items()):
        f=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
        expected=92 if name.startswith(('weekly_common','weekly_climate')) else 12 if name.startswith('legacy12') else 1
        summary.append({'experiment':name,'completed_origins':f.origin.nunique(),'expected_origins':expected,
                        'complete':f.origin.nunique()==expected,**metrics(f)})
        for origin,g in f.groupby('origin'):folds.append({'experiment':name,'origin':origin,'target':g.target.iloc[0],**metrics(g)})
        for district,g in f.groupby('district_id'):districts.append({'experiment':name,'district_id':district,**metrics(g)})
        f.to_csv(RESULTS/f'{name}_predictions.csv',index=False)
    pd.DataFrame(summary).to_csv(RESULTS/'master_metrics.csv',index=False)
    pd.DataFrame(folds).to_csv(RESULTS/'per_origin_metrics.csv',index=False)
    pd.DataFrame(districts).to_csv(RESULTS/'district_metrics.csv',index=False)
    print('Summarized',len(paths),'completed fold experiments',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['core','expanded','weekly_ablation','summarize']);parser.add_argument('--horizon',type=int,choices=HORIZONS);args=parser.parse_args()
    if args.phase=='summarize':summarize()
    elif args.phase in ['expanded','weekly_ablation']:getattr(Study(),args.phase)(horizons=(args.horizon,) if args.horizon else HORIZONS)
    else:getattr(Study(),args.phase)()
