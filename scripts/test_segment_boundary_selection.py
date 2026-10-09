"""Fail-closed local CTC boundary competition tests."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from segment_boundary_selection import CtcBoundaryRow,select_ctc_boundary_variant

class BoundarySelectionTests(unittest.TestCase):
    def test_good_shift_needs_whole_pair_gain(self):
        old=[CtcBoundaryRow(7,36.8,.5),CtcBoundaryRow(8,39.5,.001),CtcBoundaryRow(9,41.62,.58)]
        new=[CtcBoundaryRow(7,36.8,.5),CtcBoundaryRow(8,40.64,.54),CtcBoundaryRow(9,44.4,.56)]
        result=select_ctc_boundary_variant(old,new,moved_row=8)
        self.assertEqual(result.status,'RESEARCH_ACCEPTED_NOT_CENTRAL')
        self.assertGreater(result.average_gain,.04)
    def test_low_moved_score_rejected(self):
        old=[CtcBoundaryRow(37,133.,.49),CtcBoundaryRow(38,135.48,.005)]
        new=[CtcBoundaryRow(37,133.,.32),CtcBoundaryRow(38,133.8,.041)]
        self.assertEqual(select_ctc_boundary_variant(old,new,moved_row=38).status,'REJECTED_BOUNDARY')
    def test_unmoved_row_confidence_regression_rejected(self):
        old=[CtcBoundaryRow(1,1.,.60),CtcBoundaryRow(2,2.,.01)]
        new=[CtcBoundaryRow(1,1.2,.05),CtcBoundaryRow(2,3.2,.80)]
        self.assertEqual(select_ctc_boundary_variant(old,new,moved_row=2).reason,'other-row-score-regression')
    def test_nonmonotonic_rejected(self):
        old=[CtcBoundaryRow(1,1.,.1),CtcBoundaryRow(2,2.,.1)]
        new=[CtcBoundaryRow(1,2.,.5),CtcBoundaryRow(2,1.,.5)]
        self.assertEqual(select_ctc_boundary_variant(old,new,moved_row=1).reason,'nonmonotonic-timestamps')
    def test_missing_or_reordered_rows_rejected(self):
        old=[CtcBoundaryRow(1,1.,.1),CtcBoundaryRow(2,2.,.1)]
        new=[CtcBoundaryRow(2,1.,.5),CtcBoundaryRow(1,2.,.5)]
        self.assertEqual(select_ctc_boundary_variant(old,new,moved_row=1).reason,'row-identity-or-order-mismatch')
    def test_positive_moved_gain_without_net_gain_is_rejected(self):
        old=[CtcBoundaryRow(1,1.,.7),CtcBoundaryRow(2,2.,.2)]
        new=[CtcBoundaryRow(1,1.,.6),CtcBoundaryRow(2,2.,.31)]
        self.assertEqual(select_ctc_boundary_variant(old,new,moved_row=2).reason,'insufficient-net-ctc-evidence')

if __name__=='__main__':
    unittest.main()
