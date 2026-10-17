"""Score every prespecified simple weekly comparator; retain unfavorable results."""
import pandas as pd
from research_extensions_2026_10.run_study import RESULTS,HORIZONS,metrics,write_json
from research_extensions_2026_10.analyze_study import load_experiment,paired_bootstrap,KEYS


def main():
    summaries=[];comparisons=[]
    for h in HORIZONS:
        rf=load_experiment(f'weekly_common_rf_h{h}_start2020_all')
        for model in ['seasonal_naive','persistence','recent_mean4','seasonal_mean']:
            frame=pd.read_csv(RESULTS/f'weekly_common_{model}_h{h}_predictions.csv')
            assert frame.target.nunique()==92 and not frame.duplicated(KEYS).any()
            summaries.append({'model':model,'horizon':h,**metrics(frame)})
            if rf is not None and rf.target.nunique()==92:
                assert set(map(tuple,frame[KEYS].to_numpy()))==set(map(tuple,rf[KEYS].to_numpy()))
                for cluster in ['district','moving_block']:
                    for comparison in paired_bootstrap(rf,frame,cluster=cluster):
                        comparisons.append({'comparison':f'rf_start2020_minus_{model}','horizon':h,**comparison})
    pd.DataFrame(summaries).to_csv(RESULTS/'weekly_simple_baseline_metrics.csv',index=False)
    pd.DataFrame(comparisons).to_csv(RESULTS/'weekly_rf_simple_baseline_paired.csv',index=False)
    write_json(RESULTS/'weekly_simple_baseline_interpretation.json',{
        'scope':'All four prespecified simple models on the same weekly targets; no tuned winner or 2026 selection.',
        'information_boundary':'Simple models can use the observed origin-week count. RF raw cases and rolling features end at origin minus one week. All assume origin-week information is available by actual issuance, whose historical timing is unknown.',
        'count_fitting_2022':'Retained for all simple count forecasts; excluded only from the shared warning reference.',
        'paired_comparisons_available_horizons':sorted(set(r['horizon'] for r in comparisons)),
        'caveat':'The RF-versus-simple difference combines algorithm and chosen input design; do not claim a pure architecture effect.'})
    print(pd.DataFrame(summaries)[['model','horizon','wape','precision','sensitivity','f2']].to_string(index=False),flush=True)


if __name__=='__main__':main()
