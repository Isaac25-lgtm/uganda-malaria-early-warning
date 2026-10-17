from __future__ import annotations
import hashlib,json,os
from datetime import date,timedelta,datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from app.services.models import pooled_tree_validation as tree
from app.services.case_data_cleaning import build_clean_case_series,load_case_exclusions
from app.services.modelling_dataset import _clean_feature_value,MIN_ROLLING_OBSERVATIONS
from research_extensions_2026_10.run_study import Study,DATA,HORIZONS,CONFIGS,metrics,write_json
from research_extensions_2026_10.analyze_study import KEYS,paired_bootstrap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_extensions_2026_10_round2'
RESULTS=OUT/'results';CHECKPOINTS=OUT/'checkpoints'
for p in [RESULTS,CHECKPOINTS]:p.mkdir(parents=True,exist_ok=True)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def cutoff(origin,delay):return (date.fromisoformat(str(origin))-timedelta(weeks=delay)).isoformat()

def atomic_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.next');frame.to_csv(temp,index=False);os.replace(temp,path)

class Context:
    def __init__(self):
        self.study=Study();self.rows=self.study.rows
        self.index={(r['district_id'],r['week_start_date']):i for i,r in enumerate(self.rows)}
        source=[dict(r) for r in self.rows]
        for e in load_case_exclusions():
            day=date.fromisocalendar(e.epi_year,e.epi_week,1).isoformat()
            source[self.index[e.district_id,day]]['confirmed_malaria_cases']=str(e.original_cases)
        clean=build_clean_case_series(source,district_ids=sorted(self.study.history),start_date=date.fromisoformat(min(self.study.dates)),end_date=date.fromisoformat(max(self.study.dates)))
        self.clean=clean.values;self.frames={};self.proofs={}
        self.code_hash=hashlib.sha256((OUT/'common.py').read_bytes()+(OUT/'run_trees.py').read_bytes()).hexdigest()

    def features(self,delay):
        if delay in self.frames:return self.frames[delay]
        x=self.study.X.copy();shift=max(0,delay-1)
        # Recompute anchors as of the assumed available case week for EACH row.
        for i,r in enumerate(self.rows):
            d=r['district_id'];o=date.fromisoformat(r['week_start_date']);available=o-timedelta(weeks=delay)
            for lag in [1,2,4,8,12]:
                item=self.clean.get((d,o-timedelta(weeks=lag+shift)))
                value=_clean_feature_value(item,as_of=available)
                x.at[i,f'cases_lag_{lag}']=float(value) if value is not None else np.nan
                x.at[i,f'cases_lag_{lag}_interpolated']=int(bool(item and item.predictor_interpolated and value is not None))
            for w in [4,8,12]:
                items=[self.clean.get((d,o-timedelta(weeks=k+shift))) for k in range(1,w+1)]
                values=[_clean_feature_value(item,as_of=available) for item in items]
                valid=[float(v) for v in values if v is not None]
                x.at[i,f'cases_rolling_mean_{w}']=np.mean(valid) if len(valid)>=MIN_ROLLING_OBSERVATIONS[w] else np.nan
                x.at[i,f'cases_rolling_interpolated_count_{w}']=sum(bool(item and item.predictor_interpolated and v is not None) for item,v in zip(items,values))
        if delay==0:
            x['cases_current']=pd.to_numeric([r['confirmed_malaria_cases'] for r in self.rows],errors='coerce')
        self.frames[delay]=x
        return x

    def threshold(self,d,origin,target,delay):
        key=d,origin,target,delay
        if key not in self.proofs:self.proofs[key]=self.study.threshold(d,cutoff(origin,delay),target)
        return self.proofs[key]

    def validation(self,h,origin,target):
        s=self.study
        return np.flatnonzero((s.dates==origin)&(s.target_dates[h]==target)&np.isfinite(s.y[h]))

    def training(self,h,origin,delay):
        s=self.study
        return np.flatnonzero((s.dates<=origin)&(s.target_dates[h]<=cutoff(origin,delay))&np.isfinite(s.y[h]))

def valid_checkpoint(path,expected):
    meta=path.with_suffix('.json')
    if not path.exists() or not meta.exists():return None
    item=json.loads(meta.read_text())
    for k,v in expected.items():
        if item.get(k)!=v:raise ValueError(f'Checkpoint contract mismatch {path.name}: {k}')
    if sha(path)!=item['predictions_sha256']:raise ValueError('Checkpoint bytes changed: '+path.name)
    frame=pd.read_csv(path)
    if len(frame)!=item['validation_rows'] or frame.duplicated(KEYS).any():raise ValueError('Checkpoint missing/duplicate rows')
    return frame
