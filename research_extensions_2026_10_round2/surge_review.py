"""Surge-first interpretation; separate warning and count decisions."""
import numpy as np
import pandas as pd
from research_extensions_2026_10_round2.common import OUT,RESULTS,atomic_csv,write_json,now

def main():
    table=pd.read_csv(RESULTS/'EQUAL_INFORMATION_BENCHMARKS.csv')
    table['policy_type']='forecast_above_fixed_Q75'
    extra=pd.read_csv(RESULTS/'SURGE_CANDIDATE_METRICS.csv')
    extra=extra[extra.policy_type.eq('direct_classifier')|extra.evaluation.eq('later2026')].copy()
    extra['evaluation']=extra.evaluation.replace({'original12':'legacy12'})
    table=pd.concat([table,extra],ignore_index=True)
    rows=[];decisions=[]
    for keys,group in table.groupby(['evaluation','horizon','delay']):
        evaluation,h,delay=keys;complete=group[group.complete].copy()
        for r in group.to_dict('records'):
            rows.append(r)
        if complete.empty:continue
        reference=complete.iloc[0];positive=int(reference.tp+reference.fn);negative=int(reference.fp+reference.tn)
        for method in ['always_alert_reference','never_alert_reference']:
            always=method.startswith('always');tp=positive if always else 0;fp=negative if always else 0;fn=0 if always else positive;tn=0 if always else negative
            rows.append({'evaluation':evaluation,'horizon':h,'delay':delay,'method':method,'policy_type':'non_predictive_reference','complete':True,'n':positive+negative,'target_weeks':reference.target_weeks,'districts':reference.districts,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'prevalence':positive/(positive+negative),'sensitivity':1. if always else 0.,'precision':positive/(positive+negative) if always else np.nan,'f2':5*tp/(5*tp+4*fn+fp) if positive else np.nan,'alert_rate':1. if always else 0.})
        warning=complete.sort_values(['f2','sensitivity','precision','method'],ascending=[False,False,False,True]).iloc[0]
        counts=complete[complete.wape.notna()].sort_values(['wape','method']);count=counts.iloc[0] if len(counts) else None
        decisions.append({'evaluation':evaluation,'horizon':h,'delay':delay,'warning_point_choice_among_tested_policies':warning.method,'warning_f2':warning.f2,'warning_recall':warning.sensitivity,'warning_precision':warning.precision,'warning_missed_surges':int(warning.fn),'warning_false_alerts':int(warning.fp),'count_point_choice':count.method if count is not None else None,'count_wape':count.wape if count is not None else np.nan,'same_choice':warning.method==count.method if count is not None else False,'status':'Inspected-data point comparisons, not a deployment selection; later2026 point choice is descriptive only'})
    result=pd.DataFrame(rows)
    result['missed_fraction']=result.fn/(result.tp+result.fn).replace(0,np.nan)
    result['false_alert_fraction']=result.fp/(result.tp+result.fp).replace(0,np.nan)
    result['alerts_per_target_week']=(result.tp+result.fp)/result.target_weeks
    result['no_skill_average_precision']=result.prevalence
    atomic_csv(result,RESULTS/'SURGE_FIRST_COMPARISON.csv')
    atomic_csv(pd.DataFrame(decisions),RESULTS/'SEPARATE_WARNING_AND_COUNT_CHOICES.csv')
    write_json(RESULTS/'surge_priority_amendment.json',{'recorded_at_utc':now(),'user_priority':'Surge first; count WAPE separate; scientific closure with no automatic third round.','timing':'Priority clarified after h8 count inspection; subsequent user scope addendum explicitly requested corrected direct RF/XGB surge comparisons. SURGE_CLOSURE_PLAN.md was recorded before those fits. All evidence remains exploratory.','new_fits':'See surge_execution.json; one bounded existing-family comparison, no further search.','event_definition':'Q75 unchanged; probability cutoff is a separate decision rule.','working_summary':'F2, recall, precision, missed surges, false alerts, workload and matched uncertainty.','limitation':'No outer/2026 cutoff optimization, untouched holdout, accepted clinical workload or deployment promotion.'})

if __name__=='__main__':main()
