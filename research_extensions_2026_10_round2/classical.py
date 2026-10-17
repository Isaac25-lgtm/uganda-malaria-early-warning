"""Bounded seasonal feasibility and matched direct district ARX ridge."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,warnings
from datetime import date,timedelta
import numpy as np
import pandas as pd
from research_extensions_2026_10_round2.common import *

ORIGINS=['2024-03-18','2024-09-16','2025-03-17','2025-09-15']

def selected_districts(s):
    frame=pd.DataFrame(s.rows);frame['value']=pd.to_numeric(frame.confirmed_malaria_cases,errors='coerce')
    totals=frame[frame.epi_year.astype(int)<=2023].groupby('district_id').value.sum().sort_values(kind='stable')
    return totals.iloc[np.linspace(0,len(totals)-1,12).round().astype(int)].index.tolist()

def seasonal_job(district,origin,method,destination):
    import statsmodels.api as sm
    from research_comparisons.literature_based_forecasting_benchmark.src import data,zinszer_arimax
    s=Study();last=cutoff(origin,1);history=[r for r in s.history[district] if r['week_start_date']<=last]
    started=time.monotonic();result={'district_id':district,'origin':origin,'method':method,'case_cutoff':last,'data_sha256':s.signature,'code_sha256':sha(Path(__file__)),'status':'failed'}
    try:
        if method=='adapted_arimax':
            observed=data.build_observed_cases(s.rows,data.load_error_keys());climate=data.build_climate_series(s.rows);preds={};orders={}
            # The source adaptation has horizon-specific (h+1)-lag climate designs.
            for h in HORIZONS:
                endog,exog,future,max_case,max_climate=zinszer_arimax._weekly_frame(district,origin,h,observed,climate)
                predicted,order,aic=zinszer_arimax._fit_forecast(endog,exog,future,h+1)
                assert max_case<=last and max_climate<=last
                preds[h]=max(0,float(predicted));orders[h]={'order':order,'aic':aic,'max_case':max_case,'max_climate':max_climate}
            result.update(status='complete',predictions=preds,orders=orders,convergence_policy='Original adapted helper accepts finite-AIC fits without testing optimizer convergence; diagnostic reproduction only, not a repaired national benchmark.')
        else:
            grid=pd.date_range(min(r['week_start_date'] for r in history),last,freq='W-MON');lookup={r['week_start_date']:float(r['confirmed_malaria_cases']) for r in history if r['confirmed_malaria_cases']!=''}
            y=np.array([lookup.get(t.date().isoformat(),np.nan) for t in grid]);scale=float(np.nanmean(y)) or 1.
            if np.isfinite(y).sum()<104:raise ValueError('Fewer than 104 observed weeks')
            seasonal=(1,0,0,52) if method=='annual_sarima' else (0,0,0,0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fit=sm.tsa.SARIMAX(y/scale,order=(1,0,0),seasonal_order=seasonal,trend='c',enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=50,method='lbfgs',low_memory=True)
            converged=bool(fit.mle_retvals.get('converged',False));p=np.asarray(fit.forecast(13))*scale
            result.update(converged=converged,aic=float(fit.aic),order=[1,0,0],seasonal_order=list(seasonal),observed_weeks=int(np.isfinite(y).sum()),calendar_weeks=len(y),warnings=list(dict.fromkeys(str(w.message) for w in caught)))
            if not converged:raise ValueError('Optimizer did not converge within 50 iterations')
            if not np.isfinite(p).all():raise ValueError('Nonfinite forecast')
            result.update(status='complete',predictions={h:max(0,float(p[h])) for h in HORIZONS})
    except Exception as e:result['error']=f'{type(e).__name__}: {e}'
    result['seconds']=time.monotonic()-started;write_json(destination,result)

def seasonal():
    s=Study();districts=selected_districts(s);folder=OUT/'classical_jobs';folder.mkdir(exist_ok=True);registry=[]
    write_json(RESULTS/'seasonal_subset_design.json',{'districts':districts,'origins':ORIGINS,'selection':'12 evenly spaced 2020–2023 observed burden ranks; no 2024–2025 scores used','scope':'Feasibility subset only; not representative national validation','timeout_seconds':60,'maxiter':50,'data_sha256':s.signature})
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for d in districts:
        for o in ORIGINS:
            for method in ['plain_ar1','annual_sarima','adapted_arimax']:
                path=folder/f'{method}_{d}_{o}.json'
                if path.exists():
                    old=json.loads(path.read_text());assert old['data_sha256']==s.signature and old['code_sha256']==sha(Path(__file__))
                    registry.append(old);continue
                started=time.monotonic()
                try:
                    p=subprocess.run([sys.executable,'-m','research_extensions_2026_10_round2.classical','job','--district',d,'--origin',o,'--method',method,'--destination',str(path)],env=env,capture_output=True,text=True,timeout=60)
                    if p.returncode:raise RuntimeError(p.stderr[-1000:])
                    result=json.loads(path.read_text())
                except Exception as e:
                    result={'district_id':d,'origin':o,'method':method,'status':'timeout' if isinstance(e,subprocess.TimeoutExpired) else 'failed','data_sha256':s.signature,'code_sha256':sha(Path(__file__)),'seconds':time.monotonic()-started,'error':f'{type(e).__name__}: {e}'};write_json(path,result)
                registry.append(result);print(json.dumps({k:result[k] for k in ['district_id','origin','method','status','seconds']}),flush=True)
                write_json(RESULTS/'seasonal_progress.json',{'planned':144,'attempted':len(registry),'complete':False,'at_utc':now()})
    predictions=[]
    for r in registry:
        if r['status']!='complete':continue
        for h in HORIZONS:
            t=(date.fromisoformat(r['origin'])+timedelta(weeks=h)).isoformat();actual=next((x['confirmed_malaria_cases'] for x in s.history[r['district_id']] if x['week_start_date']==t),'')
            if actual=='':continue
            proof=s.threshold(r['district_id'],cutoff(r['origin'],1),t)
            predictions.append({'method':r['method'],'district_id':r['district_id'],'origin':r['origin'],'target':t,'horizon':h,'actual':float(actual),'predicted':r['predictions'][str(h)],'threshold':float(proof['threshold'])})
    atomic_csv(pd.DataFrame(predictions),RESULTS/'seasonal_subset_predictions.csv')
    write_json(RESULTS/'seasonal_registry.json',registry)
    write_json(RESULTS/'seasonal_progress.json',{'planned':144,'attempted':len(registry),'status_counts':pd.Series([r['status'] for r in registry]).value_counts().to_dict(),'complete':True,'at_utc':now()})

def ridge_prediction(train,val,y):
    med=np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0 for col in train.T])
    train=np.where(np.isfinite(train),train,med);val=np.where(np.isfinite(val),val,med)
    mean=train.mean(axis=0);sd=train.std(axis=0);sd[sd==0]=1;train=(train-mean)/sd;val=(val-mean)/sd
    ym=y.mean();coef=np.linalg.solve(train.T@train+np.eye(train.shape[1]),train.T@(y-ym))
    return max(0,float((val@coef+ym)[0]))

def ridge():
    s=Study();x0=s.X.copy();numeric=[c for c in x0 if c.startswith(('cases_lag_','cases_rolling_mean_','rainfall_','temperature_','humidity_')) and 'interpolated' not in c]
    # Original predictor interpolations can require the origin week. For strict
    # delay-1 ridge use observed calendar case inputs, no interpolation.
    lookup={(r['district_id'],r['week_start_date']):float(r['confirmed_malaria_cases']) for r in s.rows if r['confirmed_malaria_cases']!=''}
    for i,r in enumerate(s.rows):
        d=r['district_id'];o=date.fromisoformat(r['week_start_date'])
        for lag in [1,2,4,8,12]:x0.at[i,f'cases_lag_{lag}']=lookup.get((d,(o-timedelta(weeks=lag)).isoformat()),np.nan)
        for w in [4,8,12]:
            values=[lookup[(d,(o-timedelta(weeks=k)).isoformat())] for k in range(1,w+1) if (d,(o-timedelta(weeks=k)).isoformat()) in lookup]
            x0.at[i,f'cases_rolling_mean_{w}']=np.mean(values) if len(values)>={4:3,8:6,12:9}[w] else np.nan
    base=x0[numeric].to_numpy(float);outputs=[];started=time.monotonic()
    for h in HORIZONS:
        targetdates=[date.fromisoformat(t) for t in s.target_dates[h]]
        phase=np.array([(t-date(2000,1,3)).days/7/52.1775 for t in targetdates])
        annual=np.array([lookup.get((r['district_id'],(t-timedelta(weeks=52)).isoformat()),np.nan) for r,t in zip(s.rows,targetdates)])
        xseason=np.column_stack([base,annual,*[f(2*np.pi*k*phase) for k in [1,2] for f in [np.sin,np.cos]]])
        for target in pd.date_range('2024-03-25','2025-12-22',freq='W-MON'):
            t=target.date().isoformat();o=(target-timedelta(weeks=h)).date().isoformat();available=cutoff(o,1)
            for d in sorted(s.history):
                ids=np.flatnonzero((s.ids==d)&(s.target_dates[h]<=available)&np.isfinite(s.y[h]));vi=np.flatnonzero((s.ids==d)&(s.dates==o)&(s.target_dates[h]==t)&np.isfinite(s.y[h]))
                if not len(vi) or len(ids)<52:continue
                proof=s.threshold(d,available,t)
                for method,x in [('direct_arx_ridge',base),('direct_seasonal_arx_ridge',xseason)]:
                    prediction=ridge_prediction(x[ids],x[vi],s.y[h][ids])
                    outputs.append({'method':method,'district_id':d,'origin':o,'target':t,'horizon':h,'actual':s.y[h][vi[0]],'predicted':prediction,'threshold':float(proof['threshold']),'train_rows':len(ids),'max_training_label':max(s.target_dates[h][ids]),'case_cutoff':available,'status':'complete'})
            if time.monotonic()-started>1800:raise TimeoutError('District ridge 30-minute budget reached')
        atomic_csv(pd.DataFrame([r for r in outputs if r['horizon']==h]),RESULTS/f'direct_ridge_h{h}_predictions.csv')
        print(json.dumps({'ridge_horizon_complete':h,'seconds':time.monotonic()-started}),flush=True)
    write_json(RESULTS/'ridge_specification.json',{'seconds':time.monotonic()-started,'alpha':1,'data_sha256':s.signature,'code_sha256':sha(Path(__file__)),'preprocessing':'Per-district/fold training medians and standardization; observed case inputs without interpolation; raw-count ridge with intercept and nonnegative clipping','seasonal_additions':'Target-minus-52-week observed count; two target-calendar Fourier harmonics with 52.1775-week period. All target-calendar terms deterministic; annual case precedes origin for h<=12.','scope':'Explicit seasonal ARX, not SARIMA or exact published-method replication','fit_records':len(outputs),'status':'complete'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['seasonal','ridge','job']);p.add_argument('--district');p.add_argument('--origin');p.add_argument('--method');p.add_argument('--destination',type=Path);a=p.parse_args()
    if a.phase=='job':seasonal_job(a.district,a.origin,a.method,a.destination)
    else:globals()[a.phase]()
