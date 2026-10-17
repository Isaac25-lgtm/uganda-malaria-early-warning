import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from research_extensions_2026_10_round2.common import Context,cutoff,valid_checkpoint,atomic_csv,sha

class AvailabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.ctx=Context()

    def test_training_labels_stop_at_assumed_available_report(self):
        for h in [4,8,12]:
            for delay in [0,1,2]:
                ids=self.ctx.training(h,'2024-01-01',delay)
                self.assertLessEqual(max(self.ctx.study.target_dates[h][ids]),cutoff('2024-01-01',delay))
                self.assertNotIn('2024-01-01',self.ctx.study.target_dates[h][ids] if delay else [])

    def test_case_lags_follow_calendar_delay(self):
        for delay in [0,1,2]:
            x=self.ctx.features(delay)
            checked=0
            for i in range(100,len(self.ctx.rows),379):
                r=self.ctx.rows[i];day=cutoff(r['week_start_date'],max(1,delay));j=self.ctx.index.get((r['district_id'],day))
                if j is not None and self.ctx.rows[j]['confirmed_malaria_cases']!='':
                    self.assertEqual(float(self.ctx.rows[j]['confirmed_malaria_cases']),x.loc[i,'cases_lag_1']);checked+=1
            self.assertGreater(checked,50)

    def test_current_count_is_observed_not_target(self):
        x=self.ctx.features(0)
        for i in range(0,len(x),997):
            text=self.ctx.rows[i]['confirmed_malaria_cases']
            self.assertTrue(np.isnan(x.loc[i,'cases_current']) if text=='' else x.loc[i,'cases_current']==float(text))
        self.assertNotIn('cases_current',self.ctx.features(1))

    def test_reference_changes_only_with_case_cutoff(self):
        d=self.ctx.rows[-1]['district_id']
        a=self.ctx.threshold(d,'2024-04-01','2024-06-24',2)
        b=self.ctx.study.threshold(d,'2024-03-18','2024-06-24')
        self.assertEqual(a,b)

class CheckpointTests(unittest.TestCase):
    def test_progress_writer_retries_until_windows_reader_releases(self):
        import os,time
        from concurrent.futures import ThreadPoolExecutor
        from research_extensions_2026_10_round2.safe_io import read_json,write_json
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'progress.json';p.write_text('{"value":0}')
            with ThreadPoolExecutor(max_workers=1) as pool:
                with p.open('r') as reader:
                    future=pool.submit(write_json,p,{'value':1});time.sleep(.15)
                    if os.name=='nt':self.assertFalse(future.done())
                    self.assertEqual(reader.read(),'{"value":0}')
                future.result(timeout=4)
            self.assertEqual(read_json(p),{'value':1})

    def test_contract_and_byte_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a.csv';f=pd.DataFrame([{'district_id':'D','horizon':4,'origin':'2024-01-01','target':'2024-01-29','actual':1,'predicted':1,'threshold':1}]);atomic_csv(f,p)
            p.with_suffix('.json').write_text(json.dumps({'delay':1,'validation_rows':1,'predictions_sha256':sha(p)}))
            self.assertEqual(len(valid_checkpoint(p,{'delay':1})),1)
            with self.assertRaises(ValueError):valid_checkpoint(p,{'delay':0})
            p.write_text(p.read_text()+'\n')
            with self.assertRaises(ValueError):valid_checkpoint(p,{'delay':1})

class SurgeClosureTests(unittest.TestCase):
    def test_existing_classifier_families_fit_both_feature_contracts(self):
        from research_extensions_2026_10_round2 import surge_classifiers as module
        rng=np.random.default_rng(15)
        x=pd.DataFrame({c:rng.normal(size=64) for c in module.tree.POOLED_NUMERIC_FEATURES})
        x['district_id']=['A','B']*32;y=np.array([0,1]*32)
        for model in ['rf','xgb']:
            for group in ['all','none']:
                p=module.estimator(model,module.CONFIGS[model].classifier_parameter_candidates[0],group,y)
                p.fit(x,y);prob=module.tree._positive_probabilities(p,x)
                self.assertTrue(np.isfinite(prob).all());self.assertTrue(((prob>=0)&(prob<=1)).all())
                columns=p.named_steps['preprocess'].transformers[0][2]
                self.assertEqual(any(c.startswith('rainfall_') for c in columns),group=='all')

    @staticmethod
    def context(start,counts):
        from datetime import date,timedelta
        from types import SimpleNamespace
        rows=[]
        for i,count in enumerate(counts):
            d=date.fromisoformat(start)+timedelta(weeks=i);y,w,_=d.isocalendar()
            rows.append({'district_id':'D','week_start_date':d.isoformat(),'epi_year':str(y),'epi_week':str(w),'confirmed_malaria_cases':str(count)})
        dates=np.array([r['week_start_date'] for r in rows])
        targets=np.array([(date.fromisoformat(d)+timedelta(weeks=4)).isoformat() for d in dates])
        return SimpleNamespace(study=SimpleNamespace(rows=rows,history={'D':rows},ids=np.array(['D']*len(rows)),dates=dates,target_dates={4:targets},y={4:np.full(len(rows),50.)}))

    def test_training_event_reference_cannot_see_origin_or_future_case(self):
        from unittest.mock import patch
        from research_extensions_2026_10_round2 import surge_classifiers as module
        with tempfile.TemporaryDirectory() as td,patch.object(module,'RESULTS',Path(td)):
            first=module.labels(self.context('2019-12-30',[10,20,1000]),4)
            second=module.labels(self.context('2019-12-30',[10,20,1000000000]),4)
            np.testing.assert_array_equal(first,second)
            proof=pd.read_csv(Path(td)/'classifier_training_label_proofs_h4.csv')
            self.assertTrue((proof.threshold==10).all())
            self.assertTrue((proof.reference_cutoff<proof.origin).all())

    def test_frozen_classifier_training_labels_exclude_2026_targets(self):
        from unittest.mock import patch
        from research_extensions_2026_10_round2 import surge_classifiers as module
        with tempfile.TemporaryDirectory() as td,patch.object(module,'RESULTS',Path(td)):
            ctx=self.context('2025-11-03',[10]*9);labels=module.labels(ctx,4)
            self.assertTrue(np.isfinite(labels).any())
            self.assertTrue(np.isnan(labels[ctx.study.target_dates[4]>'2025-12-22']).all())

class ClassicalTests(unittest.TestCase):
    def test_ridge_preprocessing_is_training_only_and_unit_invariant(self):
        from research_extensions_2026_10_round2.classical import ridge_prediction
        train=np.array([[1.,np.nan],[2.,4.],[3.,5.],[4.,6.]])
        y=np.array([10.,20.,30.,40.]);val=np.array([[2.5,np.nan]])
        prediction=ridge_prediction(train,val,y)
        scaled=ridge_prediction(train*100,val*100,y)
        self.assertAlmostEqual(prediction,scaled,places=10)
        self.assertAlmostEqual(prediction,25.,places=10)

    def test_annual_target_case_precedes_issue_for_every_horizon(self):
        from datetime import date,timedelta
        for h in [4,8,12]:
            origin=date(2025,1,6);target=origin+timedelta(weeks=h)
            self.assertLessEqual(target-timedelta(weeks=52),origin-timedelta(weeks=1))

if __name__=='__main__':unittest.main()
