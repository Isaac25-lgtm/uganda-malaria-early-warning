"""Reference sufficiency, heterogeneity and dependence diagnostics on frozen predictions."""
import json
from datetime import date
import numpy as np
import pandas as pd
from app.services.model_validation import _seasonal_week_distance
from research_extensions_2026_10_round2.common import *

PRIOR=ROOT/'research_extensions_2026_10/results'

def main():
    s=Study();district_rows=[];support=[];strata=[];sufficient=[];blocks=[];fallback=[]
    burden=pd.DataFrame(s.rows);burden['cases']=pd.to_numeric(burden.confirmed_malaria_cases,errors='coerce')
    past=burden[burden.epi_year.astype(int)<=2023].groupby('district_id').cases.sum()
    groups=pd.qcut(past.rank(method='first'),3,labels=['low','middle','high']).astype(str).to_dict()
    for family in ['legacy12','weekly_common']:
        for h in HORIZONS:
            full=pd.read_csv(PRIOR/f'{family}_rf_h{h}_start2020_all_predictions.csv')
            for row in full.itertuples():
                history=[r for r in s.history[row.district_id] if r['week_start_date']<=row.origin and r['confirmed_malaria_cases']!='' and int(r['epi_year'])!=2022]
                targetweek=date.fromisoformat(row.target).isocalendar().week
                seasonal=[r for r in history if _seasonal_week_distance(r,targetweek)<=2]
                count=len(seasonal);years=len(set(r['epi_year'] for r in seasonal));used=count>=8
                ref=seasonal if used else history
                threshold=sorted(float(r['confirmed_malaria_cases']) for r in ref)[int((len(ref)-1)*.75)]
                assert threshold==row.threshold
                support.append({'evaluation':family,'district_id':row.district_id,'horizon':h,'origin':row.origin,'target':row.target,'seasonal_observations':count,'seasonal_years':years,'actual_reference_observations':len(ref),'actual_reference_years':len(set(r['epi_year'] for r in ref)),'fallback':not used,'sufficient_3_year_seasonal':count>=8 and years>=3})
            info=pd.DataFrame([r for r in support if r['evaluation']==family and r['horizon']==h])
            f=full.merge(info,on=KEYS,validate='one_to_one').assign(burden_group=lambda x:x.district_id.map(groups))
            for district,g in f.groupby('district_id'):
                district_rows.append({'evaluation':family,'horizon':h,'district_id':district,'burden_group':groups[district],'fallback_share':float(g.fallback.mean()),'reference_median':float(g.actual_reference_observations.median()),'seasonal_years_median':float(g.seasonal_years.median()),**metrics(g)})
            for group,g in f.groupby('burden_group'):strata.append({'evaluation':family,'horizon':h,'burden_group':group,**metrics(g)})
            for status,g in f.groupby('fallback'):fallback.append({'evaluation':family,'horizon':h,'fallback':bool(status),**metrics(g)})
            keep=f[f.sufficient_3_year_seasonal]
            sufficient.append({'evaluation':family,'horizon':h,'policy':'abstain_if_seasonal_n_lt8_or_years_lt3','total_rows':len(f),'eligible_rows':len(keep),'abstained_rows':len(f)-len(keep),'coverage':len(keep)/len(f),**metrics(keep)})
            if family=='weekly_common':
                none=pd.read_csv(PRIOR/f'weekly_climate_rf_h{h}_start2020_none_predictions.csv')
                for block in [4,8,12,16]:
                    for record in paired_bootstrap(full,none,cluster='moving_block',block=block):blocks.append({'comparison':'weekly_full_minus_none','horizon':h,**record})
    atomic_csv(pd.DataFrame(support),RESULTS/'threshold_reference_detail.csv')
    atomic_csv(pd.DataFrame(district_rows),RESULTS/'DISTRICT_AND_SEASON_DIAGNOSTICS.csv')
    atomic_csv(pd.DataFrame(strata),RESULTS/'burden_strata.csv')
    atomic_csv(pd.DataFrame(fallback),RESULTS/'fallback_strata.csv')
    atomic_csv(pd.DataFrame(sufficient),RESULTS/'reference_sufficiency_abstention.csv')
    atomic_csv(pd.DataFrame(blocks),RESULTS/'block_length_sensitivity.csv')
    f=pd.DataFrame(support)
    summary=f.groupby(['evaluation','horizon']).agg(rows=('fallback','size'),fallback_share=('fallback','mean'),seasonal_n_min=('seasonal_observations','min'),seasonal_n_median=('seasonal_observations','median'),seasonal_n_max=('seasonal_observations','max'),seasonal_years_median=('seasonal_years','median'),sufficient_share=('sufficient_3_year_seasonal','mean')).reset_index()
    atomic_csv(summary,RESULTS/'threshold_reference_summary.csv')
    dist=pd.DataFrame(district_rows);agg=[]
    for (family,h),g in dist.groupby(['evaluation','horizon']):
        agg.append({'evaluation':family,'horizon':h,'equal_district_mean_wape':g.wape.mean(),'district_median_wape':g.wape.median(),'district_p90_wape':g.wape.quantile(.9),'district_max_wape':g.wape.max(),'median_f2':g.f2.median(),'undefined_wape':int(g.wape.isna().sum())})
    atomic_csv(pd.DataFrame(agg),RESULTS/'district_error_distribution.csv')
    worst=dist.sort_values('wape',ascending=False).groupby(['evaluation','horizon']).head(15)
    atomic_csv(worst,RESULTS/'worst_decile_districts.csv')
    missing=pd.read_csv(PRIOR/'missingness_district_year.csv').groupby('district_id')[['source_absent','confirmed_error']].sum().reset_index()
    atomic_csv(dist.merge(missing,on='district_id'),RESULTS/'district_missingness_association.csv')
    write_json(RESULTS/'diagnostic_limits.json',{'burden_strata':'Terciles of 2020–2023 summed observed caseload; not incidence/population. Associations descriptive, no causal/fairness certification.','geography':'Reference region/subregion columns are blank; no region labels fabricated. District distributions reported.','abstention':'Reference sufficiency selected before these new diagnostics. Abstentions are not true negatives; coverage reported. Conditional performance cannot prove a policy improvement.','threshold_uncertainty':'Reference counts/years/fallback and abstention audited. A calibrated probabilistic threshold or validated outbreak definition is not established.','spatial_dependence':'District clusters do not resolve spatial correlation. Temporal block sensitivity is not a hierarchical spatial bootstrap.','source_revision':'No new surveillance accessed. Existing downloaded vintage and prior limited source audits retained.'})
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
