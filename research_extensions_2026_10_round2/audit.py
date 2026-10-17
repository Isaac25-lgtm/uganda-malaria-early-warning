"""Fresh-process tuning/prediction reproduction and precise source contracts."""
import ast,json,time,io
from dataclasses import replace
from datetime import date,timedelta
import numpy as np
import pandas as pd
from app.services.model_validation import build_rolling_origin_folds
from research_extensions_2026_10_round2.common import *

def main():
    s=Study();source=ROOT/'research_extensions_2026_10/baseline/app/services/models/pooled_tree_validation.py'
    function=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='_select_horizon_parameters')
    namespace=dict(vars(tree));exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    old_select=namespace['_select_horizon_parameters'];boundaries=[];selection=[];replays=[]
    for h in HORIZONS:
        folds=build_rolling_origin_folds(s.rows,horizon=h);rows=folds[0].trainable_rows
        target=f'target_week_start_date_h{h}';dates=sorted(set(r[target] for r in rows));hold=set(dates[-min(8,max(2,len(dates)//5)):])
        old=[r for r in rows if r[target] not in hold];new,val=tree._chronological_inner_split(rows,horizon=h,holdout_dates=hold)
        origin=min(r['week_start_date'] for r in val)
        boundaries.append({'horizon':h,'earliest_inner_origin':origin,'legacy_max_training_label':max(r[target] for r in old),'corrected_max_training_label':max(r[target] for r in new),'legacy_future_label_rows':sum(r[target]>origin for r in old),'legacy_train_rows':len(old),'corrected_train_rows':len(new),'validation_rows':len(val)})
        # First-round RF choices for all horizons already have frozen tuning records.
        for model in ['rf','xgb']:
            if h==8:
                config=replace(CONFIGS[model],controlled_threads=2)
                for name,fn in [('legacy',old_select),('corrected',tree._select_horizon_parameters)]:
                    result=fn(config,folds);write_json(RESULTS/f'audit_tuning_{model}_h8_{name}.json',result)
                    selection.append({'horizon':h,'model':model,'version':name,'selected_parameters':json.dumps(result['selected_parameters'],sort_keys=True),'candidate_count':len(result['candidate_results'])})
                params=json.loads((RESULTS/f'audit_tuning_{model}_h8_corrected.json').read_text())['selected_parameters']
                recorded=json.loads((ROOT/f'research_extensions_2026_10/results/corrected_tuning_{model}_h8.json').read_text())
                assert params==recorded['selected_parameters']
                o=folds[0].origin_date.isoformat();t=folds[0].target_date.isoformat();train=np.flatnonzero((s.target_dates[h]<=o)&np.isfinite(s.y[h]));val_ids=np.flatnonzero((s.dates==o)&(s.target_dates[h]==t)&np.isfinite(s.y[h]))
                saved_path=ROOT/f'research_extensions_2026_10/experiments/predictions/legacy12_{model}_h8_start2020_all_{o}.csv'
                saved_metadata=json.loads(saved_path.with_suffix('.json').read_text())
                # The initial RF checkpoints predate the added metadata field;
                # preserved run_study_initial_threads2.py fixes those fits at 2.
                fit_config=replace(config,controlled_threads=saved_metadata.get('controlled_threads',2))
                p=tree._build_pipeline(fit_config,params);p.fit(s.X.iloc[train],s.y[h][train]);prediction=np.maximum(0,p.predict(s.X.iloc[val_ids]))
                saved=pd.read_csv(saved_path).set_index('district_id')
                expected=np.array([saved.loc[d,'predicted'] for d in s.ids[val_ids]])
                raw_difference=float(np.abs(expected-prediction).max())
                # XGBoost emits float32. Pandas preserves that dtype when the
                # original checkpoint is serialized, producing compact decimals.
                # Compare the same CSV representation, rather than float64
                # parsing against unrounded binary float32 values.
                serialized=pd.read_csv(io.StringIO(pd.DataFrame({'predicted':prediction}).to_csv(index=False))).predicted.to_numpy()
                maximum=float(np.abs(expected-serialized).max());assert maximum<1e-8,{'model':model,'raw_difference':raw_difference,'serialized_difference':maximum}
                replays.append({'model':model,'horizon':h,'origin':o,'n':len(prediction),'fit_threads':fit_config.controlled_threads,'tuning_threads':config.controlled_threads,'maximum_difference_after_same_csv_serialization':maximum,'raw_binary_vs_csv_difference':raw_difference,'passed':True})
    atomic_csv(pd.DataFrame(boundaries),RESULTS/'inner_boundary_audit.csv');atomic_csv(pd.DataFrame(selection),RESULTS/'independent_h8_selection.csv')
    write_json(RESULTS/'fresh_h8_reproduction.json',{'passed':True,'checks':replays,'method':'Fresh process reran original and corrected inner candidate grids at h8, then refit each corrected first outer fold; no round-one outputs overwritten.','legacy_selector_sha256':sha(source)})
    # Exact source functions show the adaptation is already calendar complete and
    # consumes O-1 cases/climate, while convergence is not gated.
    ar=ROOT/'research_comparisons/literature_based_forecasting_benchmark/src/zinszer_arimax.py'
    write_json(RESULTS/'adapted_arimax_source_audit.json',{'source':str(ar.relative_to(ROOT)),'sha256':sha(ar),'case_cutoff':'O-1','calendar':'Explicit weekly grid retains absent endog slots','climate':'Two variables; (h+1)-lag climate; future exog source <= O-1','strategy':'Recursive h+1 steps; direct RF differs','order_selection':'Five AIC candidates within each district/horizon/fold; RF/XGB three MAE candidates only at earliest fold','convergence_gate':'Source does not require optimizer convergence before accepting finite AIC','tuning_embargo':'No pooled-tree inner validation selector; the exact corrected inner-split bug does not apply','comparability':'Different predictors, fitting strategy, search effort and case cutoff. Original predictions frozen; subset rerun is diagnostic, not a nationally standardized new benchmark.'})
    timeline=[]
    for h in HORIZONS:
        o=date(2025,12,22);t=o+timedelta(weeks=h)
        for delay in [0,1,2]:
            case=o-timedelta(weeks=delay);feature=o-timedelta(weeks=h+delay)
            timeline.append({'horizon':h,'delay_assumption':delay,'forecast_origin_week_start':o.isoformat(),'latest_training_feature_week':feature.isoformat(),'latest_training_label_week':case.isoformat(),'case_cutoff_week_start':case.isoformat(),'target_week_start':t.isoformat(),'latest_climate_observation_week_start':(o-timedelta(weeks=4)).isoformat(),'climate_release_timestamp':'not archived; later reanalysis vintage','actual_issue_timestamp':'not established','week_note':'An origin-week complete count cannot be available at its Monday start; this is a retrospective indexing convention.'})
    atomic_csv(pd.DataFrame(timeline),RESULTS/'worked_timelines.csv')
    print('Fresh tuning and prediction audit passed.',flush=True)

if __name__=='__main__':main()
