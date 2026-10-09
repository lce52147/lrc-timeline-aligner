"""S1 duplicate single-line recurrence: no authority and fail-closed cases."""
from pathlib import Path
import sys, unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from segment_duplicate_line import infer_duplicate_line_acoustic_hits
from segment_recurrence import AcousticRepeatMatch,AcousticRepeatScan
from segment_structure import VocalPassage


class DuplicateLineAcousticsTests(unittest.TestCase):
    def setUp(self):
        self.lyrics=['intro']+['I am dreaming']*8+['last']
        # Fixed external vocal-passages are untrusted audio/VAD propositions.
        self.passages=tuple(VocalPassage(i*10.,i*10.+7.5) for i in range(10))
        self.priors=[float(i)*10 for i in range(10)]
        self.proposed=tuple(float(i)*10 for i in range(10))
        self.proposed=list(self.proposed)
        self.proposed[2]=20.2
        self.proposed[7]=70.1
        self.priors[2]=7.
        self.priors[7]=57.
        self.dp=SimpleNamespace(
          entry_times=tuple(self.proposed),entry_passages=tuple(range(10)),
          passages=self.passages)
        self.features=np.ones((600,15),dtype=np.float32)
        self.db=np.full(600,-17.,dtype=np.float32)
    def infer(self,**kwargs):
        return infer_duplicate_line_acoustic_hits(
            self.lyrics,self.dp,self.priors,self.features,self.db,
            hop_seconds=.02,**kwargs)
    def test_produces_only_unverified_ordered_acoustic_hypotheses(self):
        scan=AcousticRepeatScan(
            'RESEARCH_UNVERIFIED_OCCURRENCE',7.5,
            (AcousticRepeatMatch(20.0,1.),AcousticRepeatMatch(70.0,.92),
             AcousticRepeatMatch(80.1,.73)))
        with patch('segment_duplicate_line.scan_vocal_recurrences',return_value=scan):
            res=self.infer()
        self.assertEqual(res.status,'RESEARCH_UNVERIFIED_OCCURRENCE')
        self.assertEqual(res.reference_row,3)
        self.assertEqual([(h.row,h.time) for h in res.hits],[(3,20.),(8,70.),(9,80.1)])
        self.assertIn('do-not-prove',res.reason)
    def test_second_far_duplicate_required(self):
        scan=AcousticRepeatScan('RESEARCH_UNVERIFIED_OCCURRENCE',7.5,
             (AcousticRepeatMatch(20.0,1.),AcousticRepeatMatch(80.1,.8)))
        with patch('segment_duplicate_line.scan_vocal_recurrences',return_value=scan):
            res=self.infer()
        self.assertEqual(res.status,'BLOCKED_DUPLICATE_LINE')
        self.assertEqual(res.reason,'no-second-severe-occurrence')
    def test_distinct_large_duplicate_lexical_groups_rejected(self):
        lyrics=['one']*5+['two']*5
        with patch('segment_duplicate_line.scan_vocal_recurrences',side_effect=AssertionError('No scan')):
            got=infer_duplicate_line_acoustic_hits(lyrics,self.dp,self.priors,
                      self.features,self.db,hop_seconds=.02)
        self.assertEqual(got.reason,'multiple-distinct-duplicate-phrases')
    def test_no_reference_when_global_ctc_stable(self):
        self.priors=list(self.proposed)
        with patch('segment_duplicate_line.scan_vocal_recurrences',side_effect=AssertionError('No scan')):
            got=self.infer()
        self.assertEqual(got.reason,'no-large-displacement-acoustic-reference')
    def test_acoustic_no_activity_fails(self):
        with patch('segment_duplicate_line.scan_vocal_recurrences',return_value=
           AcousticRepeatScan('BLOCKED_NO_VOCAL_ACTIVITY',7.5,())):
            got=self.infer()
        self.assertEqual(got.reason,'unusable-acoustic-recurrence')
    def test_competing_near_identical_hit_distances_are_ambiguous(self):
        scan=AcousticRepeatScan('RESEARCH_UNVERIFIED_OCCURRENCE',7.5,
            (AcousticRepeatMatch(20.12,1.),AcousticRepeatMatch(20.3,.9),
             AcousticRepeatMatch(70.03,.85)))
        with patch('segment_duplicate_line.scan_vocal_recurrences',return_value=scan):
            got=self.infer()
        self.assertEqual(got.reason,'insufficient-monotone-matched-occurrences')


if __name__=='__main__': unittest.main()
