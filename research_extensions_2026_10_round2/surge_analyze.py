"""Matched direct-classifier comparisons and a conservative closure verdict."""
import json
import numpy as np
import pandas as pd
from app.services.model_validation import build_rolling_origin_folds
from research_extensions_2026_10_round2.common import *
from research_extensions_2026_10_round2.surge_classifiers import FOLDER,FREEZE,warning_metrics

def binary_frame(f):
    b=f.copy();b['actual']=(f.actual>f.threshold).astype(float)
    b['predicted']=f.predicted_surge.astype(float) if 'predicted_surge' in f else (f.predicted>f.threshold).astype(float)
    b['threshold']=.5
    return b[KEYS+['actual','predicted','threshold']]

def pairs(first,second,evaluation,h,comparison):
    result=[];a=binary_frame(first);b=binary_frame(second)
    assert set(map(tuple,a[KEYS].to_numpy()))==set(map(tuple,b[KEYS].to_numpy()))
    designs=[('district',12),('origin',12)] if evaluation=='original12' else [('district',12),('moving_block',4),('moving_block',8)]
    for cluster,block in designs:
        for r in paired_bootstrap(a,b,cluster=cluster,block=block):
            if r['metric'] in ['f2','sensitivity','precision']:result.append({'evaluation':evaluation,'horizon':h,'comparison':comparison,**r})
    return result

def load(model,h,group,evaluation):
    files=sorted(FOLDER.glob(f'{evaluation}_{model}_h{h}_{group}_*.csv'))
    return pd.concat([pd.read_csv(p) for p in files],ignore_index=True) if files else None

def main():
    rows=[];paired=[];registry=[];frames={};verdicts=[]
    for p in sorted(FOLDER.glob('*.json')):
        r=json.loads(p.read_text());registry.append({'experiment':p.stem,'family':'surge_classifier_or_frozen_count','model':r['model'],'horizon':r['horizon'],'group':r['group'],'evaluation':r['evaluation'],'status':r['status'],'train_rows':r['train_rows'],'max_training_label':r['max_training_target'],'label_cutoff':r['label_cutoff'],'validation_rows':r['validation_rows'],'parameters':json.dumps(r['parameters'],sort_keys=True),'probability_cutoff':r.get('probability_cutoff'),'data_sha256':r['data_sha256'],'code_sha256':r['code_sha256'],'predictions_sha256':r['predictions_sha256'],'seconds':r['seconds']})
    for h in HORIZONS:
        baseline=pd.read_csv(RESULTS/f'legacy12_rf_h{h}_delay1_predictions.csv');baseline['predicted_surge']=baseline.predicted>baseline.threshold
        frames['original12',h,'rf_count_strict_all']=baseline
        rows.append({'evaluation':'original12','horizon':h,'delay':1,'method':'rf_count_strict_all','group':'all','policy_type':'forecast_above_fixed_Q75','complete':baseline.target.nunique()==12,**metrics(baseline)})
        xgb_path=RESULTS/f'legacy12_xgb_h{h}_delay1_predictions.csv'
        if xgb_path.exists():
            xgb=pd.read_csv(xgb_path);xgb['predicted_surge']=xgb.predicted>xgb.threshold
            frames['original12',h,'xgb_count_strict_all']=xgb
            rows.append({'evaluation':'original12','horizon':h,'delay':1,'method':'xgb_count_strict_all','group':'all','policy_type':'forecast_above_fixed_Q75','complete':xgb.target.nunique()==12,**metrics(xgb)})
            # Reuse the already computed matched RF/XGB intervals, reversing
            # signs/limits rather than running the same bootstrap again.
            prior=pd.read_csv(RESULTS/'MASTER_PAIRED_RESULTS.csv')
            prior=prior[(prior.evaluation=='legacy12')&(prior.horizon==h)&(prior.delay==1)&prior.comparison.eq('rf_minus_xgb')&prior.metric.isin(['f2','sensitivity','precision'])]
            for r in prior.to_dict('records'):
                r['first'],r['second']=r['second'],r['first'];r['difference_first_minus_second']=-r['difference_first_minus_second'];r['ci_lower'],r['ci_upper']=-r['ci_upper'],-r['ci_lower'];r['evaluation']='original12';r['comparison']='xgb_count_strict_all_minus_rf_count_strict_all';paired.append(r)
        p=FOLDER/f'later2026_rf_count_h{h}.csv'
        if p.exists():
            b=pd.read_csv(p);frames['later2026',h,'rf_count_strict_none']=b
            rows.append({'evaluation':'later2026','horizon':h,'delay':1,'method':'rf_count_strict_none','group':'none','policy_type':'forecast_above_fixed_Q75','complete':b.target.nunique()==14,**metrics(b)})
        for evaluation in ['original12','later2026']:
            base_method='rf_count_strict_all' if evaluation=='original12' else 'rf_count_strict_none';base=frames.get((evaluation,h,base_method))
            for group in ['all','none'] if evaluation=='original12' else ['none']:
                for model in ['rf','xgb']:
                    f=load(model,h,group,evaluation)
                    if f is None:continue
                    name=f'{model}_classifier_{group}';frames[evaluation,h,name]=f;complete=f.target.nunique()==(12 if evaluation=='original12' else 14)
                    rows.append({'evaluation':evaluation,'horizon':h,'delay':1,'method':name,'group':group,'policy_type':'direct_classifier','complete':complete,**warning_metrics(f)})
                    if complete and base is not None and base.target.nunique()==(12 if evaluation=='original12' else 14):paired.extend(pairs(f,base,evaluation,h,name+'_minus_'+base_method))
            if evaluation=='original12':
                for model in ['rf','xgb']:
                    a=frames.get((evaluation,h,f'{model}_classifier_all'));b=frames.get((evaluation,h,f'{model}_classifier_none'))
                    if a is not None and b is not None and a.target.nunique()==b.target.nunique()==12:paired.extend(pairs(a,b,evaluation,h,model+'_classifier_all_minus_none'))
    summary=pd.DataFrame(rows);uncertainty=pd.DataFrame(paired)
    atomic_csv(summary,RESULTS/'SURGE_CANDIDATE_METRICS.csv');atomic_csv(uncertainty,RESULTS/'SURGE_PAIRED_COMPARISONS.csv')
    completed_names={r['experiment'] for r in registry};s=Study()
    for h in HORIZONS:
        planned=[(f'original12_{m}_h{h}_{g}_{f.origin_date.isoformat()}',m,g) for m in ['rf','xgb'] for g in ['all','none'] for f in build_rolling_origin_folds(s.rows,horizon=h)]
        planned += [(f'later2026_{m}_h{h}_none_{FREEZE}',m,'none') for m in ['rf','xgb']]+[(f'later2026_rf_count_h{h}','rf_count_strict','none')]
        for name,model,group in planned:
            if name not in completed_names:registry.append({'experiment':name,'family':'surge_classifier_or_frozen_count','model':model,'horizon':h,'group':group,'status':'not_completed','reason':'Bounded pass: see surge_execution.json failed groups/runtime; no result claimed'})
    for p in RESULTS.glob('classifier_tuning_*.json'):
        t=json.loads(p.read_text());seen=set()
        for r in t['candidates']:
            if r['candidate'] in seen:continue
            seen.add(r['candidate']);registry.append({'experiment':p.stem+f'_candidate{r["candidate"]}','family':'surge_inner_candidate_fit','model':t['model'],'horizon':t['horizon'],'group':t['group'],'status':r['status'],'parameters':json.dumps(r['parameters'],sort_keys=True),'seconds':r.get('candidate_fit_seconds'),'reason':r.get('error')})
    atomic_csv(pd.DataFrame(registry),RESULTS/'SURGE_EXPERIMENT_REGISTER.csv')
    for h in HORIZONS:
        dev=summary[(summary.evaluation=='original12')&(summary.horizon==h)&summary.complete]
        candidates=dev[~dev.method.eq('rf_count_strict_all')].sort_values(['f2','sensitivity','precision','method'],ascending=[False,False,False,True])
        if candidates.empty or dev[dev.method.eq('rf_count_strict_all')].empty:
            verdicts.append({'horizon':h,'verdict':'inconclusive','reason':'Direct comparison incomplete','recommendation':'Retain corrected RF count-warning research baseline; no promotion'});continue
        best=candidates.iloc[0];base=dev[dev.method.eq('rf_count_strict_all')].iloc[0]
        p=uncertainty[(uncertainty.evaluation=='original12')&(uncertainty.horizon==h)&uncertainty.comparison.eq(best.method+'_minus_rf_count_strict_all')&uncertainty.metric.eq('f2')]
        history=candidates[candidates.group.eq('none')].sort_values(['f2','sensitivity','precision','method'],ascending=[False,False,False,True])
        later=summary[(summary.evaluation=='later2026')&(summary.horizon==h)&summary.complete]
        chosen=history.iloc[0].method if not history.empty else None
        lp=uncertainty[(uncertainty.evaluation=='later2026')&(uncertainty.horizon==h)&uncertainty.comparison.eq(str(chosen)+'_minus_rf_count_strict_none')&uncertainty.metric.eq('f2')]
        later_candidate=later[later.method.eq(chosen)];later_base=later[later.method.eq('rf_count_strict_none')]
        nonincreased_fp=best.fp<=base.fp and not later_candidate.empty and not later_base.empty and later_candidate.iloc[0].fp<=later_base.iloc[0].fp
        statistically_supported=len(p)==2 and (p.ci_lower>0).all() and len(lp)==3 and (lp.ci_lower>0).all()
        absent=len(p)==2 and (p.ci_upper<=0).all()
        development_eligible=[];supported=[]
        for candidate in history.itertuples():
            dp=uncertainty[(uncertainty.evaluation=='original12')&(uncertainty.horizon==h)&uncertainty.comparison.eq(candidate.method+'_minus_rf_count_strict_all')&uncertainty.metric.eq('f2')]
            if len(dp)==2 and (dp.ci_lower>0).all() and candidate.fp<=base.fp:development_eligible.append(candidate.method)
        # Choose on development only, then confirm/reject that one choice later.
        # Never switch to another candidate because its 2026 result looks better.
        confirmation=development_eligible[0] if development_eligible else None
        if confirmation:
            later_p=uncertainty[(uncertainty.evaluation=='later2026')&(uncertainty.horizon==h)&uncertainty.comparison.eq(confirmation+'_minus_rf_count_strict_none')&uncertainty.metric.eq('f2')]
            later_c=later[later.method.eq(confirmation)]
            if len(later_p)==3 and (later_p.ci_lower>0).all() and not later_c.empty and not later_base.empty and later_c.iloc[0].fp<=later_base.iloc[0].fp:supported.append(confirmation)
        # No-skill reference: alerting every district-week. F2 at ~31% prevalence
        # rewards near-universal alerting, so a candidate's F2 gain is read against it.
        def always_f2(row):
            positive=float(row.tp+row.fn);negative=float(row.fp+row.tn);return 5*positive/(5*positive+negative) if positive else np.nan
        always_dev=always_f2(base);always_later=always_f2(later_base.iloc[0]) if not later_base.empty else np.nan
        higher=candidates[candidates.f2>base.f2]
        workload_failure=not higher.empty and (higher.fp>base.fp).all() and not development_eligible
        if supported:
            verdict='statistical improvement supported; operational acceptability unestablished';recommendation=f'Prefer {supported[0]} as research candidate for supervisor review; no deployment promotion'
        elif workload_failure:
            verdict='no credible improvement: every higher-F2 candidate increased false alerts (near always-alert behaviour)';recommendation='Retain corrected RF count-warning research baseline; stop searching'
        elif absent:
            verdict='absent for the strongest development point candidate';recommendation='Retain corrected RF count-warning research baseline; stop searching'
        else:
            verdict='inconclusive';recommendation='Retain corrected RF count-warning research baseline; stop searching'
        verdicts.append({'horizon':h,'strongest_development_point_candidate':best.method,'candidate_f2':best.f2,'baseline_f2':base.f2,'f2_difference':best.f2-base.f2,'candidate_fp':int(best.fp),'baseline_fp':int(base.fp),'history_candidate_selected_on_development':chosen,'later_history_f2':float(later_candidate.iloc[0].f2) if not later_candidate.empty else None,'later_baseline_f2':float(later_base.iloc[0].f2) if not later_base.empty else None,'paired_development_support':bool(len(p)==2 and (p.ci_lower>0).all()),'paired_later_support':bool(len(lp)==3 and (lp.ci_lower>0).all()),'false_alerts_not_increased':bool(nonincreased_fp),'verdict':verdict,'recommendation':recommendation,'limitations':'Candidate picked on inspected development data; 2026 already inspected; full-climate specification lacks 2026 validation; workload ceiling and clinical utility unestablished. Statistical intervals do not adjust for candidate selection/multiplicity.'})
        verdicts[-1]['confirmation_candidate_selected_on_development']=confirmation
        verdicts[-1].update({'candidate_alert_rate':float(best.alert_rate),'baseline_alert_rate':float(base.alert_rate),'always_alert_f2_development':always_dev,'always_alert_f2_later':always_later,'baseline_f2_below_always_alert_development':bool(base.f2<always_dev),'candidate_f2_minus_always_alert_development':float(best.f2-always_dev),'higher_f2_candidates_all_increased_false_alerts':bool(workload_failure)})
    write_json(RESULTS/'SURGE_IMPROVEMENT_VERDICT.json',{'generated_at_utc':now(),'horizons':verdicts,'no_further_automatic_search':True,'primary_objective':'surge warning','training_cutoff':FREEZE,'new_2026_training':False,'operational_workload_accepted':False})
    # Registry includes model fits; probabilities and raw surveillance stay local.
    p=RESULTS/'EXPERIMENT_REGISTRY.csv';existing=pd.read_csv(p);combined=pd.concat([existing,pd.DataFrame(registry)],ignore_index=True);atomic_csv(combined,p)
    completion=json.loads((RESULTS/'analysis_completion.json').read_text());completion['registry_records']=len(combined);completion['surge_complete']=json.loads((RESULTS/'surge_execution.json').read_text())['complete'];write_json(RESULTS/'analysis_completion.json',completion)
    print('Direct surge comparisons and closure verdict recorded.',flush=True)

if __name__=='__main__':main()
