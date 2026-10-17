"""Prevent changed data/specification or partial results entering a locked evaluation."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from research_extensions_2026_10.evaluate_2026 import CUTOFF,read_frozen_checkpoint,tree


class HoldoutCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'predictions.csv';self.parameters={'n_estimators':64}
        rows=[]
        for t in pd.date_range('2026-03-23','2026-06-22',freq='W-MON'):
            for d in range(146):rows.append({'district_id':f'D{d:03d}','horizon':4,
                'origin':(t-pd.Timedelta(weeks=4)).strftime('%Y-%m-%d'),'target':t.strftime('%Y-%m-%d'),
                'actual':10.,'predicted':11.,'threshold':9.})
        pd.DataFrame(rows).to_csv(self.path,index=False)
        self.metadata={'train_start':2020,'fit_cutoff':CUTOFF,'max_training_target':CUTOFF,
            'parameters':self.parameters,'input_sha256':'data','protocol_sha256':'protocol',
            'threshold_reference_cutoff':CUTOFF,'target_weeks':14,'refit_on_2026':False,
            'features':[c for c in tree.POOLED_NUMERIC_FEATURES if not c.startswith(('rainfall_','temperature_','humidity_'))]+['district_id'],
            'predictions_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest()}
        self.save_metadata()

    def save_metadata(self):self.path.with_suffix('.json').write_text(json.dumps(self.metadata))

    def read(self,**updates):
        arguments={'start':2020,'horizon':4,'parameters':self.parameters,'input_sha':'data','protocol_sha':'protocol'}
        return read_frozen_checkpoint(self.path,**{**arguments,**updates})

    def test_compatible_checkpoint_and_specification_drift(self):
        self.assertEqual(len(self.read()),2044)
        for change in [{'input_sha':'different'},{'protocol_sha':'different'},{'parameters':{'n_estimators':96}}]:
            with self.subTest(change=change),self.assertRaises(ValueError):self.read(**change)

    def test_partial_duplicate_and_corrupt_checkpoints(self):
        original=self.path.read_bytes();self.path.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.read()
        self.path.write_bytes(original);self.metadata.pop('predictions_sha256');self.save_metadata()
        frame=pd.read_csv(self.path);frame.iloc[:-1].to_csv(self.path,index=False)
        with self.assertRaisesRegex(ValueError,'incomplete'):self.read()
        frame.iloc[-1]=frame.iloc[0];frame.to_csv(self.path,index=False)
        with self.assertRaisesRegex(ValueError,'duplicate'):self.read()

    def test_future_fitting_boundary_and_changed_features(self):
        self.metadata['max_training_target']='2026-01-05';self.save_metadata()
        with self.assertRaisesRegex(ValueError,'boundary'):self.read()
        self.metadata['max_training_target']=CUTOFF;self.metadata['features'].append('rainfall_lag_4');self.save_metadata()
        with self.assertRaisesRegex(ValueError,'features|feature'):self.read()


if __name__=='__main__':unittest.main()
