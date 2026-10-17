"""Independent surge reference/arithmetic checks and representative refits."""
import csv,io,json
from decimal import Decimal
import numpy as np
import pandas as pd
from app.services.model_validation import classification_metrics
from research_extensions_2026_10_round2.common import *
from research_extensions_2026_10_round2 import surge_classifiers as run

def main(study,truth,threshold,ctx):
    execution=json.loads((RESULTS/'surge_execution.json').read_text());contract=json.loads((RESULTS/'surge_execution_contract.json').read_text())
    assert execution['code_sha256']==contract['code_sha256']==run.code_signature()
    assert contract['data_sha256']==study.signature and contract['extended_sha256']==sha(run.EXTENDED)
    assert contract['plan_sha256']==sha(OUT/'SURGE_CLOSURE_PLAN.md')
    assert contract['scope_addendum_sha256']==sha(OUT/'review_inputs/IDE_SCOPE_AND_CLOSURE_ADDENDUM_2026-10-15.md')
    assert json.loads((OUT/'classifier_stage_execution.json').read_text())['actual_exit_code']==0
    extended=list(csv.DictReader(run.EXTENDED.open(encoding='utf-8')))
    assert max(r['week_start_date'] for r in extended)<='2026-06-30'
    extended_truth={(r['district_id'],r['week_start_date']):float(r['confirmed_malaria_cases']) for r in extended if r['confirmed_malaria_cases']!=''}
    proofs={};label_rows=0;tunings=0;frames={}
    for h in HORIZONS:
        p=RESULTS/f'classifier_training_label_proofs_h{h}.csv'
        if not p.exists():continue
        f=pd.read_csv(p);proofs[h]=f
        assert not f.duplicated(['district_id','origin']).any()
        for r in f.itertuples():
            assert r.target<=run.FREEZE and r.reference_cutoff==cutoff(r.origin,1)
            assert r.threshold==threshold(r.district_id,r.reference_cutoff,r.target)
            assert r.label==int(truth[r.district_id,r.target]>r.threshold)
            label_rows+=1
    for path in RESULTS.glob('classifier_tuning_*.json'):
        r=json.loads(path.read_text());assert r['data_sha256']==study.signature and r['code_sha256']==run.code_signature()
        assert r['max_inner_training_target']<=r['inner_label_cutoff']==cutoff(r['earliest_inner_origin'],1)
        good=[a for a in r['candidates'] if a['status']=='complete']
        selected=max(good,key=lambda a:tuple(a[k] for k in ['f2','sensitivity','average_precision','precision','specificity']))
        assert selected['parameters']==r['selected_parameters'] and selected['probability_cutoff']==r['probability_cutoff']
        assert not r['outer_or_2026_outcomes_used_for_selection'];tunings+=1
    checks=[]
    for path in sorted(run.FOLDER.glob('*.csv')):
        r=json.loads(path.with_suffix('.json').read_text());f=pd.read_csv(path);h=r['horizon']
        assert r['code_sha256']==run.code_signature() and r['predictions_sha256']==sha(path)
        assert not f.duplicated(KEYS).any() and len(f)==r['validation_rows']
        later=r['evaluation']=='later2026';assert r['data_sha256']==(contract['extended_sha256'] if later else study.signature)
        limit=r['label_cutoff'];assert r['max_training_target']<=limit<=run.FREEZE
        assert limit==(run.FREEZE if later else cutoff(r['fit_origin'],1))
        if r['model']=='rf_count_strict':
            assert r['parameters']==json.loads((RESULTS/f'tuning_rf_h{h}_delay1.json').read_text())['selected_parameters']
            train_dates=study.target_dates[h][np.isfinite(study.y[h])&(study.target_dates[h]<=run.FREEZE)]
        else:
            train_dates=proofs[h].target[proofs[h].target<=limit].to_numpy()
            selection=json.loads((RESULTS/f'classifier_tuning_{r["model"]}_h{h}_{r["group"]}.json').read_text())
            assert r['parameters']==selection['selected_parameters'] and r['probability_cutoff']==selection['probability_cutoff']
        assert len(train_dates)==r['train_rows'] and max(train_dates)==r['max_training_target']
        if later:assert f.target.nunique()==14 and len(f)==2044 and (f.origin>run.FREEZE).all()
        for row in f.itertuples():
            assert (date.fromisoformat(row.target)-date.fromisoformat(row.origin)).days==7*h
            assert row.actual==(extended_truth if later else truth)[row.district_id,row.target]
            assert row.threshold==threshold(row.district_id,run.FREEZE if later else cutoff(row.origin,1),row.target)
            if r['model']!='rf_count_strict':
                assert np.isfinite(row.probability) and 0<=row.probability<=1
                assert row.predicted_surge==(row.probability>=row.probability_cutoff)
            else:assert row.predicted_surge==(row.predicted>row.threshold)
        name=f'{r["model"]}_classifier_{r["group"]}' if r['model']!='rf_count_strict' else 'rf_count_strict_none'
        frames.setdefault((r['evaluation'],h,name),[]).append(f)
        checks.append(path.name)
    table=pd.read_csv(RESULTS/'SURGE_CANDIDATE_METRICS.csv');arithmetic=[]
    for r in table.itertuples():
        if r.method in ['rf_count_strict_all','xgb_count_strict_all']:
            family=r.method.split('_')[0];f=pd.read_csv(RESULTS/f'legacy12_{family}_h{r.horizon}_delay1_predictions.csv');pred=f.predicted>f.threshold;scores=f.predicted-f.threshold
        else:
            f=pd.concat(frames[r.evaluation,r.horizon,r.method],ignore_index=True);pred=f.predicted_surge;scores=f.probability if 'probability' in f else f.predicted-f.threshold
        result=classification_metrics((f.actual>f.threshold).tolist(),pred.tolist(),[Decimal(str(v)) for v in scores],score_is_probability=r.policy_type=='direct_classifier')
        for key in ['n','tp','fp','fn','tn','f2','sensitivity','precision']:
            expected=result[key];recorded=getattr(r,key)
            if expected is None:assert pd.isna(recorded)
            else:assert abs(float(expected)-float(recorded))<(1e-8 if key in ['n','tp','fp','fn','tn'] else 5.1e-7),(r.method,key,expected,recorded)
        tp,fp,fn=[Decimal(result[k]) for k in ['tp','fp','fn']]
        if 5*tp+4*fn+fp:assert abs(float(5*tp/(5*tp+4*fn+fp))-r.f2)<1e-8
        if result['pr_auc'] is not None:assert abs(float(result['pr_auc'])-r.average_precision)<5.1e-7
        arithmetic.append({'evaluation':r.evaluation,'horizon':r.horizon,'method':r.method,'passed':True})
    replays=[]
    for model in ['rf','xgb']:
        paths=sorted(run.FOLDER.glob(f'original12_{model}_h8_all_*.csv'))
        if not paths:continue
        path=paths[0];r=json.loads(path.with_suffix('.json').read_text());proof=proofs[8];y=np.full(len(study.rows),np.nan)
        for row in proof.itertuples():y[ctx.index[row.district_id,row.origin]]=row.label
        ids=np.flatnonzero((study.target_dates[8]<=r['label_cutoff'])&np.isfinite(y)&np.isfinite(study.y[8]));stored=pd.read_csv(path);target=stored.target.iloc[0];val=ctx.validation(8,r['fit_origin'],target)
        p=run.estimator(model,r['parameters'],'all',y[ids]);x=ctx.features(1);p.fit(x.iloc[ids],y[ids].astype(int));prob=tree._positive_probabilities(p,x.iloc[val])
        serial=pd.read_csv(io.StringIO(pd.DataFrame({'p':prob}).to_csv(index=False))).p.to_numpy();indexed=stored.set_index('district_id');expected=np.array([indexed.loc[d,'probability'] for d in study.ids[val]])
        difference=float(np.abs(serial-expected).max());assert difference<1e-8
        replays.append({'model':model,'horizon':8,'group':'all','origin':r['fit_origin'],'maximum_csv_difference':difference,'passed':True})
    output={'passed':True,'completed_checkpoints_checked':len(checks),'expected_checkpoints':153,'full_defined_completion':execution['complete'],'training_label_references_checked':label_rows,'tuning_contracts_checked':tunings,'aggregate_arithmetic':arithmetic,'fresh_classifier_replays':replays,'2026_training_labels':0,'source_snapshot_unchanged':True}
    write_json(RESULTS/'surge_verification.json',output);return output
