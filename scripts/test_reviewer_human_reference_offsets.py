"""No heldout BK leakage into human-assisted Reviewer calibration."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reviewer_human_reference_offsets import calibrate_reviewers_against_human_gold


class HumanReferenceOffsetsTests(unittest.TestCase):
    def setUp(self):
        self.gold=[float(i*3) for i in range(24)]
        self.rows=[{'reviewer_times':{
          'HUBP':t+.025,'WX':t-.030,'XLSR':t+.010
        }} for t in self.gold]
        self.train=[i for i in range(24) if i%4!=0]
    def test_true_absolute_offset_with_training_only(self):
        x=calibrate_reviewers_against_human_gold(
            self.rows,self.gold,calibration_row_indexes=self.train)
        self.assertEqual(x.status,'HUMAN_BK_ASSISTED_CALIBRATED_NOT_INDEPENDENT_AUTO')
        self.assertEqual(x.offsets_s,{'HUBP':.025,'WX':-.03,'XLSR':.01})
        self.assertFalse(x.independent_auto_eligible)
    def test_heldout_gold_time_is_not_read(self):
        base=calibrate_reviewers_against_human_gold(
            self.rows,self.gold,calibration_row_indexes=self.train)
        perturbed=list(self.gold)
        for i in range(0,len(self.gold),4):perturbed[i]+=140.
        new=calibrate_reviewers_against_human_gold(
            self.rows,perturbed,calibration_row_indexes=self.train)
        self.assertEqual(base,new)
    def test_bad_reviewer_drift_blocks_it_without_blocking_others(self):
        for i,row in enumerate(self.rows):
            row['reviewer_times']['WX']+=i*.025
        x=calibrate_reviewers_against_human_gold(
            self.rows,self.gold,calibration_row_indexes=self.train)
        self.assertNotIn('WX',x.offsets_s)
        self.assertIn('HUBP',x.offsets_s)
    def test_outliers_far_from_human_reference_rejected(self):
        for index in self.train[:5]:
            self.rows[index]['reviewer_times']['HUBP']+=9.
        x=calibrate_reviewers_against_human_gold(
            self.rows,self.gold,calibration_row_indexes=self.train)
        self.assertAlmostEqual(x.offsets_s['HUBP'],.025)
    def test_no_train_row_fails(self):
        x=calibrate_reviewers_against_human_gold(
            self.rows,self.gold,calibration_row_indexes=[])
        self.assertEqual(x.status,'BLOCKED_CALIBRATION_SPLIT')
    def test_bad_shape_fails(self):
        x=calibrate_reviewers_against_human_gold(
            self.rows,self.gold[:4],calibration_row_indexes=self.train)
        self.assertEqual(x.status,'BLOCKED_WRONG_REFERENCE_SHAPE')


if __name__=='__main__':
    unittest.main()
