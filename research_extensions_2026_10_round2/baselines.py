"""Four untuned forecasts under the same assumed available case cutoff."""
from datetime import date
import numpy as np
import pandas as pd
from app.services.model_validation import build_rolling_origin_folds
from research_extensions_2026_10_round2.common import *

def main():
    s=Study();summary=[]
    for family in ['legacy12','weekly92']:
        for h in HORIZONS:
            if family=='legacy12':pairs=[(f.origin_date.isoformat(),f.target_date.isoformat()) for f in build_rolling_origin_folds(s.rows,horizon=h)]
            else:pairs=[((t-pd.Timedelta(weeks=h)).date().isoformat(),t.date().isoformat()) for t in pd.date_range('2024-03-25','2025-12-22',freq='W-MON')]
            for delay in [0,1,2]:
                out=[]
                for o,t in pairs:
                    limit=cutoff(o,delay);week=date.fromisoformat(t).isocalendar().week
                    ids=np.flatnonzero((s.dates==o)&(s.target_dates[h]==t)&np.isfinite(s.y[h]))
                    for i in ids:
                        d=str(s.ids[i]);history=[r for r in s.history[d] if r['week_start_date']<=limit and r['confirmed_malaria_cases']!='']
                        recent=[float(r['confirmed_malaria_cases']) for r in history[-4:]]
                        seasonal=[float(r['confirmed_malaria_cases']) for r in history if int(r['epi_week'])==week]
                        predictions={'persistence':recent[-1],'recent_mean4':float(np.mean(recent)),'seasonal_naive':seasonal[-1] if seasonal else float(np.mean(recent)),'seasonal_mean':float(np.mean(seasonal)) if seasonal else float(np.mean(recent))}
                        proof=s.threshold(d,limit,t)
                        for model,p in predictions.items():out.append({'method':model,'district_id':d,'horizon':h,'origin':o,'target':t,'case_cutoff':limit,'actual':s.y[h][i],'predicted':p,'threshold':float(proof['threshold'])})
                frame=pd.DataFrame(out);atomic_csv(frame,RESULTS/f'baselines_{family}_h{h}_delay{delay}_predictions.csv')
                for method,g in frame.groupby('method'):summary.append({'evaluation':family,'horizon':h,'delay':delay,'method':method,**metrics(g)})
    atomic_csv(pd.DataFrame(summary),RESULTS/'simple_equal_information_metrics.csv')
    write_json(RESULTS/'simple_baseline_specification.json',{'methods':['persistence','recent_mean4','seasonal_naive','seasonal_mean'],'recent_mean4':'Last four observed reports, possibly spanning missing calendar weeks. No imputed outcomes.','seasonal_naive':'Latest observed matching ISO week, otherwise recent mean4. Not a fixed 52-calendar-week shift.','delays':[0,1,2],'availability':'Case cutoff affects predictors and shared thresholds. Actual arrival timestamps unknown.','all_targets':'Original valid rows; no target selection based on performance.'})
    print('Completed 72 baseline method/horizon/delay/evaluation scenarios.',flush=True)

if __name__=='__main__':main()
