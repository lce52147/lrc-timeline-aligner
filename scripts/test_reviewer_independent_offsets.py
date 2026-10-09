"""S2 reviewer clock calibration needs neither baseline nor BK."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reviewer_independent_offsets import independent_pairwise_offsets


class PairwiseOffsetsTests(unittest.TestCase):
    def rows(self, n=50, outliers=0):
        raw=[]
        for j in range(n):
            base=j*2+5.
            times={'HUBP':base+.04,'WX':base-.02,'XLSR':base+.01}
            if j<outliers:times['WX']=base+4.2
            raw.append({'reviewer_times':times,'expected_current_seconds':base+1000})
        return raw
    def test_reference_clocks_not_needed(self):
        data=self.rows()
        result=independent_pairwise_offsets(data)
        self.assertEqual(result.status,'RELATIVE_CLOCK_ONLY_UNVERIFIED_ABSOLUTE')
        self.assertAlmostEqual(result.offsets_s['HUBP'],.03,places=4)
        self.assertAlmostEqual(result.offsets_s['WX'],-.03,places=4)
        self.assertAlmostEqual(result.offsets_s['XLSR'],0,places=4)
        self.assertFalse(result.absolute_offset_identifiable)
    def test_outlier_reviewed_rows_robust(self):
        res=independent_pairwise_offsets(self.rows(outliers=12))
        self.assertEqual(res.status,'RELATIVE_CLOCK_ONLY_UNVERIFIED_ABSOLUTE')
        self.assertAlmostEqual(res.offsets_s['WX'],-.03,places=4)
    def test_common_absolute_shift_indistinguishable(self):
        original=self.rows()
        changed=[]
        for row in original:
            changed.append({'reviewer_times':{k:v+12.5 for k,v in row['reviewer_times'].items()},
                            'expected_current_seconds':-1000})
        self.assertEqual(independent_pairwise_offsets(original).offsets_s,
                         independent_pairwise_offsets(changed).offsets_s)
    def test_incomplete_graph_fails_closed(self):
        values=[{'reviewer_times':{'HUBP':i*2+.03,'WX':i*2+.02}} for i in range(20)]
        z=independent_pairwise_offsets(values)
        self.assertEqual(z.status,'BLOCKED_DISCONNECTED_REVIEWERS')
        self.assertEqual(z.offsets_s,{})
    def test_completely_unmatched_blocks(self):
        values=[{'reviewer_times':{'HUBP':1.,'WX':7.,'XLSR':13.}}]*20
        z=independent_pairwise_offsets(values)
        self.assertEqual(z.status,'BLOCKED_NO_PAIRWISE_EVIDENCE')
    def test_reviewer_duplicate_identity(self):
        z=independent_pairwise_offsets(self.rows(),reviewers=['WX','WX'])
        self.assertEqual(z.status,'BLOCKED_REVIEWER_IDENTITIES')


if __name__=='__main__':
    unittest.main()
