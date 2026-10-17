"""Checks for paired research inference, rather than only model execution."""
import unittest
import pandas as pd
from research_extensions_2026_10.analyze_study import paired_bootstrap


class PairedResearchStatisticsTests(unittest.TestCase):
    @staticmethod
    def frames():
        rows=[{'district_id':d,'horizon':4,'origin':o,'target':t,'actual':10.,'threshold':8.,'predicted':10.}
              for d in ['A','B'] for o,t in [('2025-01-06','2025-02-03'),('2025-02-03','2025-03-03')]]
        a=pd.DataFrame(rows);return a,a.assign(predicted=20.)

    def test_known_constant_effect_has_exact_cluster_interval(self):
        a,b=self.frames()
        for cluster in ['district','origin','moving_block']:
            result=paired_bootstrap(a,b,cluster=cluster,draws=100)
            wape=next(r for r in result if r['metric']=='wape')
            self.assertEqual(wape['difference_first_minus_second'],-100.)
            self.assertEqual((wape['ci_lower'],wape['ci_upper']),(-100.,-100.))
            self.assertEqual(wape['paired_rows'],4)

    def test_pairing_is_invariant_to_row_order(self):
        a,b=self.frames()
        first=paired_bootstrap(a,b,cluster='district',draws=100)
        second=paired_bootstrap(a.iloc[::-1],b.sample(frac=1,random_state=1),cluster='district',draws=100)
        self.assertEqual(first,second)

    def test_mismatched_outcomes_or_thresholds_fail(self):
        a,b=self.frames()
        for column in ['actual','threshold']:
            for difference in [1,0.000001]:
                changed=b.copy();changed.loc[0,column]+=difference
                with self.assertRaises(ValueError):paired_bootstrap(a,changed,cluster='district',draws=10)

    def test_partial_coverage_is_reported_explicitly(self):
        a,b=self.frames()
        result=paired_bootstrap(a.iloc[:2],b,cluster='district',draws=10)
        self.assertEqual(result[0]['paired_rows'],2)
        self.assertEqual(result[0]['first_total_rows'],2)
        self.assertEqual(result[0]['second_total_rows'],4)


if __name__=='__main__':unittest.main()
