"""Summarize only explicit matched scenarios; mark partial work visibly."""
import json,subprocess,sys,time
import numpy as np
import pandas as pd
from app.services.model_validation import build_rolling_origin_folds
from research_extensions_2026_10_round2.common import *

def load_tree(model,h,delay):
    files=sorted(CHECKPOINTS.glob(f'{model}_h{h}_delay{delay}_*.csv'))
    return pd.concat([pd.read_csv(p) for p in files],ignore_index=True) if files else None

def main():
    # A separate, explicitly logged bounded fit stage, after all tree workers
    # finish. A completed bounded pass is never automatically searched again.
    from research_extensions_2026_10_round2.surge_classifiers import code_signature
    status=RESULTS/'surge_execution.json'
    if status.exists():
        assert json.loads(status.read_text())['code_sha256']==code_signature()
    else:
        started=time.monotonic()
        with (OUT/'classifiers.stdout.log').open('w',encoding='utf-8') as stdout,(OUT/'classifiers.stderr.log').open('w',encoding='utf-8') as stderr:
            child=subprocess.run([sys.executable,'-u','-m','research_extensions_2026_10_round2.surge_classifiers'],stdout=stdout,stderr=stderr)
        write_json(OUT/'classifier_stage_execution.json',{'actual_exit_code':child.returncode,'seconds':time.monotonic()-started})
        assert child.returncode==0,'Bounded classifier stage failed; inspect classifiers.stderr.log'
    s=Study();summary=pd.read_csv(RESULTS/'simple_equal_information_metrics.csv').to_dict('records');pairs=[];registry=[];districts=[];district_pairs=[]
    for p in sorted(CHECKPOINTS.glob('*.json')):
        r=json.loads(p.read_text());registry.append({'experiment':p.stem,'status':r['status'],'model':r['model'],'horizon':r['horizon'],'delay':r['delay_weeks'],'origin':r['origin'],'target':r['target'],'train_rows':r['train_rows'],'validation_rows':r['validation_rows'],'max_training_label':r['max_training_target'],'label_cutoff':r['label_cutoff'],'data_sha256':r['data_sha256'],'predictions_sha256':r['predictions_sha256'],'parameters':json.dumps(r['parameters'],sort_keys=True),'seconds':r['runtime_seconds']})
    expected_checkpoints=0
    for h in HORIZONS:
        original=set(f.target_date.isoformat() for f in build_rolling_origin_folds(s.rows,horizon=h));weekly=set(t.date().isoformat() for t in pd.date_range('2024-03-25','2025-12-22',freq='W-MON'))
        expected_checkpoints+=len(original|weekly)+60
        for family,targets in [('legacy12',original),('weekly92',weekly)]:
            for delay in [0,1,2]:
                trees={}
                for model in ['rf','xgb']:
                    if family=='weekly92' and (model!='rf' or delay!=1):continue
                    f=load_tree(model,h,delay)
                    if f is None:continue
                    f=f[f.target.isin(targets)].copy();complete=set(f.target)==targets
                    if f.empty:continue
                    assert not f.duplicated(KEYS).any();trees[model]=f
                    summary.append({'evaluation':family,'horizon':h,'delay':delay,'method':model,'completed_origins':f.target.nunique(),'expected_origins':len(targets),'complete':complete,**metrics(f)})
                    atomic_csv(f,RESULTS/f'{family}_{model}_h{h}_delay{delay}_predictions.csv')
                    for d,g in f.groupby('district_id'):districts.append({'evaluation':family,'horizon':h,'delay':delay,'method':model,'district_id':d,**metrics(g)})
                    if not complete:continue
                    base=pd.read_csv(RESULTS/f'baselines_{family}_h{h}_delay{delay}_predictions.csv')
                    for method,b in base.groupby('method'):
                        assert set(map(tuple,b[KEYS].to_numpy()))==set(map(tuple,f[KEYS].to_numpy()))
                        if family=='weekly92' and model=='rf':
                            matched=f.merge(b,on=KEYS,suffixes=('_rf','_simple'),validate='one_to_one')
                            for district,g in matched.groupby('district_id'):
                                total=g.actual_rf.sum();rf_wape=100*np.abs(g.predicted_rf-g.actual_rf).sum()/total if total else np.nan;simple_wape=100*np.abs(g.predicted_simple-g.actual_simple).sum()/total if total else np.nan
                                district_pairs.append({'horizon':h,'delay':delay,'comparison':'rf_minus_'+method,'district_id':district,'paired_rows':len(g),'rf_wape':rf_wape,'simple_wape':simple_wape,'difference_points':rf_wape-simple_wape})
                        for cluster in ['district','moving_block' if family=='weekly92' else 'origin']:
                            for record in paired_bootstrap(f,b,cluster=cluster):pairs.append({'evaluation':family,'comparison':f'{model}_minus_{method}','horizon':h,'delay':delay,**record})
                if len(trees)==2 and all(set(f.target)==targets for f in trees.values()):
                    for cluster in ['district','origin']:
                        for record in paired_bootstrap(trees['rf'],trees['xgb'],cluster=cluster):pairs.append({'evaluation':family,'comparison':'rf_minus_xgb','horizon':h,'delay':delay,**record})
        ridge_path=RESULTS/f'direct_ridge_h{h}_predictions.csv'
        if ridge_path.exists():
            ridge=pd.read_csv(ridge_path);rf=load_tree('rf',h,1)
            if rf is not None:rf=rf[rf.target.isin(weekly)]
            for method,f in ridge.groupby('method'):
                summary.append({'evaluation':'weekly92','horizon':h,'delay':1,'method':method,'complete':set(f.target)==weekly,**metrics(f)})
                if rf is not None and set(rf.target)==weekly:
                    for cluster in ['district','moving_block']:
                        for record in paired_bootstrap(rf,f,cluster=cluster):pairs.append({'evaluation':'weekly92','comparison':'rf_minus_'+method,'horizon':h,'delay':1,**record})
            first=ridge[ridge.method.eq('direct_seasonal_arx_ridge')];second=ridge[ridge.method.eq('direct_arx_ridge')]
            for cluster in ['district','moving_block']:
                for record in paired_bootstrap(first,second,cluster=cluster):pairs.append({'evaluation':'weekly92','comparison':'seasonal_ridge_minus_plain_ridge','horizon':h,'delay':1,**record})
    seasonal_path=RESULTS/'seasonal_subset_predictions.csv'
    if seasonal_path.exists():
        seasonal=pd.read_csv(seasonal_path);rows=[];matched=[]
        for (h,method),g in seasonal.groupby(['horizon','method']):rows.append({'horizon':h,'method':method,'coverage_type':'own_valid_rows','possible_rows':48,**metrics(g)})
        for h,g in seasonal.groupby('horizon'):
            methods={method:f for method,f in g.groupby('method')}
            if 'plain_ar1' in methods and 'annual_sarima' in methods:
                keys=methods['plain_ar1'][KEYS].merge(methods['annual_sarima'][KEYS],on=KEYS,validate='one_to_one')
                if keys.empty:continue
                for method in ['plain_ar1','annual_sarima']:
                    f=methods[method].merge(keys,on=KEYS,validate='one_to_one');rows.append({'horizon':h,'method':method,'coverage_type':'plain_annual_common_rows','possible_rows':48,**metrics(f)})
                for cluster in ['district','origin']:
                    for record in paired_bootstrap(methods['annual_sarima'],methods['plain_ar1'],cluster=cluster):matched.append({'horizon':h,'comparison':'annual_sarima_minus_plain_ar1','scope':'12 predetermined districts, 4 origins; few temporal clusters','delay':1,**record})
        atomic_csv(pd.DataFrame(rows),RESULTS/'seasonal_subset_metrics.csv');atomic_csv(pd.DataFrame(matched),RESULTS/'seasonal_subset_paired.csv')
    summary=pd.DataFrame(summary)
    # Simple-baseline rows carry no flag; an object column of Python bools makes
    # `~` a bitwise negation (~True == -2), so coerce to a real boolean dtype.
    summary['complete']=summary['complete'].fillna(True).astype(bool)
    atomic_csv(summary,RESULTS/'EQUAL_INFORMATION_BENCHMARKS.csv')
    atomic_csv(pd.DataFrame(pairs),RESULTS/'MASTER_PAIRED_RESULTS.csv')
    tree_checkpoint_count=len(registry)
    for r in registry:r['family']='tree_outer_fit'
    completed_tags={r['experiment'] for r in registry}
    for h in HORIZONS:
        original=[(f.origin_date.isoformat(),f.target_date.isoformat()) for f in build_rolling_origin_folds(s.rows,horizon=h)]
        weekly=[((t-pd.Timedelta(weeks=h)).date().isoformat(),t.date().isoformat()) for t in pd.date_range('2024-03-25','2025-12-22',freq='W-MON')]
        planned=[('rf',1,o,t) for o,t in sorted(set(original+weekly))]+[(m,d,o,t) for d in [0,2,1] for m in ['rf','xgb'] if not (d==1 and m=='rf') for o,t in original]
        state=json.loads((RESULTS/f'tree_status_h{h}.json').read_text());failed={(a['model'],a['delay'],a['origin']):a['error'] for a in state['failed']}
        for m,d,o,t in planned:
            tag=f'{m}_h{h}_delay{d}_{o}'
            if tag in completed_tags:continue
            error=failed.get((m,d,o))
            registry.append({'experiment':tag,'family':'tree_outer_fit','model':m,'horizon':h,'delay':d,'origin':o,'target':t,'status':'failed' if error else 'unrun','reason':error or 'Bounded worker runtime limit; no result claimed'})
    for p in RESULTS.glob('tuning_*.json'):
        r=json.loads(p.read_text())
        for i,a in enumerate(r['candidates']):registry.append({'experiment':p.stem+f'_candidate{i+1}','family':'inner_tuning_candidate','model':r['model'],'horizon':r['horizon'],'delay':r['delay_weeks'],'status':a['status'],'parameters':json.dumps(a['parameters'],sort_keys=True),'seconds':a['seconds'],'reason':a.get('error'),'data_sha256':r['data_sha256']})
    if (RESULTS/'seasonal_registry.json').exists():
        seasonal_records=json.loads((RESULTS/'seasonal_registry.json').read_text())
        write_json(RESULTS/'seasonal_fit_audit.json',[{k:v for k,v in r.items() if k!='predictions'} for r in seasonal_records])
        for r in seasonal_records:registry.append({'experiment':r['method']+'_'+r['district_id']+'_'+r['origin'],'family':'classical_district_origin_job','model':r['method'],'district_id':r['district_id'],'origin':r['origin'],'horizon':'4,8,12','status':r['status'],'seconds':r['seconds'],'reason':r.get('error'),'data_sha256':r['data_sha256'],'code_sha256':r['code_sha256']})
    for h in HORIZONS:
        path=RESULTS/f'direct_ridge_h{h}_predictions.csv'
        if not path.exists():continue
        for r in pd.read_csv(path).itertuples():registry.append({'experiment':f'{r.method}_{r.district_id}_h{h}_{r.origin}','family':'district_direct_ridge_fit','model':r.method,'district_id':r.district_id,'horizon':h,'delay':1,'origin':r.origin,'target':r.target,'status':r.status,'train_rows':r.train_rows,'max_training_label':r.max_training_label,'label_cutoff':r.case_cutoff,'parameters':'alpha=1; see ridge_specification.json'})
    atomic_csv(pd.DataFrame(registry),RESULTS/'EXPERIMENT_REGISTRY.csv')
    atomic_csv(pd.DataFrame(districts),RESULTS/'equal_information_district_metrics.csv')
    if district_pairs:
        detail=pd.DataFrame(district_pairs);atomic_csv(detail,RESULTS/'strict_weekly_district_comparisons.csv')
        district_summary=[]
        for (h,comparison),g in detail.groupby(['horizon','comparison']):
            district_summary.append({'horizon':h,'comparison':comparison,'districts':len(g),'share_lower_RF_WAPE':float((g.difference_points<-1e-8).mean()),'share_equal_within_1e_minus8':float((g.difference_points.abs()<=1e-8).mean()),'median_RF_minus_simple_points':float(g.difference_points.median()),'equal_district_mean_difference_points':float(g.difference_points.mean())})
        atomic_csv(pd.DataFrame(district_summary),RESULTS/'strict_weekly_district_summary.csv')
    # Objective-specific point choices are descriptive, never a new holdout-selected winner.
    choices=[]
    for keys,g in summary[summary.complete].groupby(['evaluation','horizon','delay']):
        for objective,ascending in [('wape',True),('f1',False),('f2',False)]:
            row=g.sort_values([objective,'method'],ascending=[ascending,True]).iloc[0]
            choices.append({'evaluation':keys[0],'horizon':keys[1],'delay':keys[2],'objective':objective,'point_choice':row.method,'value':row[objective]})
        for ratio in [1,2,4,8]:
            f=g.assign(cost=ratio*g.fn+g.fp);row=f.sort_values(['cost','method']).iloc[0]
            choices.append({'evaluation':keys[0],'horizon':keys[1],'delay':keys[2],'objective':f'{ratio}FN_plus_FP','point_choice':row.method,'value':row.cost})
    atomic_csv(pd.DataFrame(choices),RESULTS/'objective_selection_ledger.csv')
    write_json(RESULTS/'analysis_completion.json',{'at_utc':now(),'checkpoint_count':tree_checkpoint_count,'expected_checkpoints':expected_checkpoints,'registry_records':len(registry),'tree_scenarios_complete':int(summary[summary.method.isin(['rf','xgb'])].complete.sum()),'tree_scenarios_expected':21,'incomplete_scenarios':summary[~summary.complete][['evaluation','horizon','delay','method']].to_dict('records'),'status':'complete' if tree_checkpoint_count==expected_checkpoints else 'partial','original_and_2026_outputs_reused_read_only':True})
    from research_extensions_2026_10_round2.surge_analyze import main as compare_classifiers
    compare_classifiers()
    from research_extensions_2026_10_round2.surge_review import main as review_surges
    review_surges()
    print('Summarized',tree_checkpoint_count,'tree checkpoints,',len(registry),'registry records and',len(summary),'scenarios.',flush=True)

if __name__=='__main__':main()
