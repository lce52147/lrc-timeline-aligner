"""S1 Central bridge: atomic input gate and no forged independence."""
import copy
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from segment_central_bridge import stage_segment_constrained_ctc


def records():
    return [
        {'row':1,'time':2.0,'ctc_score':.37,'passage':0,
         'token_spans':[{'char':'a','start':2.0,'end':2.02,'score':.12}]},
        {'row':2,'time':8.0,'ctc_score':.02,'passage':1,
         'token_spans':[{'char':'b','start':8.0,'end':8.02,'score':.01}]},
    ]


class CentralStagingTests(unittest.TestCase):
    def setUp(self):
        self.original={'backend':'ctc','assignments':[
            {'alignment_hypotheses':[{'source':'existing-producer','raw_candidate_time':1.2}]},
            {'alignment_hypotheses':[]},
        ]}
    def stage(self,values=None):
        return stage_segment_constrained_ctc(
            self.original, records() if values is None else values,
            entry_count=2,source_revision='real-bounded-ctc-v1'
        )
    def test_normal_stage_preserves_baseline_and_other_producers(self):
        saved=copy.deepcopy(self.original)
        output,status=self.stage()
        self.assertEqual(status,'STAGED_FOR_CENTRAL_NO_AUTHORITY')
        self.assertEqual(self.original,saved)
        self.assertEqual(output['assignments'][0]['alignment_hypotheses'][0]['source'],'existing-producer')
        row=output['assignments'][0]['alignment_hypotheses'][-1]
        self.assertEqual(row['source'],'segment-constrained-ctc')
        self.assertEqual(row['spans'][0]['start'],2.0)
        self.assertEqual(row['identity_support'],'supported')
        for name in ('acoustic_support','direct_onset_support'):
            self.assertEqual(row[name],'unavailable')
        self.assertFalse(row['independent_content_identity'])
        self.assertFalse(row['authoritative'])
    def test_weak_row_cannot_claim_content_identity(self):
        output,status=self.stage()
        self.assertEqual(output['assignments'][1]['alignment_hypotheses'][-1]['identity_support'],'unavailable')
    def test_missing_one_row_rejects_entire_song(self):
        result,status=self.stage(records()[:1])
        self.assertEqual(status,'BLOCKED_PARTIAL_SONG')
        self.assertIs(result,self.original)
    def test_duplicate_row_rejected(self):
        value=records();value[1]['row']=1
        self.assertEqual(self.stage(value)[1],'BLOCKED_ROW_IDENTITY')
    def test_time_reverse_rejected(self):
        value=records();value[1]['time']=1.2;value[1]['token_spans'][0]['start']=1.2
        self.assertEqual(self.stage(value)[1],'BLOCKED_NONMONOTONIC_SONG')
    def test_passage_reverse_rejected(self):
        value=records();value[0]['passage']=4
        self.assertEqual(self.stage(value)[1],'BLOCKED_NONMONOTONIC_PASSAGE')
    def test_first_token_not_at_opening_rejected(self):
        value=records();value[1]['token_spans'][0]['start']=8.3;value[1]['token_spans'][0]['end']=8.32
        self.assertEqual(self.stage(value)[1],'BLOCKED_OPENING_SPAN_MISMATCH')
    def test_synthetic_token_zero_coverage_rejected(self):
        value=records();value[1]['token_spans']=[]
        self.assertEqual(self.stage(value)[1],'BLOCKED_INCOMPLETE_CTC_EVIDENCE')
    def test_non_numeric_span_is_rejected(self):
        value=records();value[0]['token_spans'][0]['score']='high'
        self.assertEqual(self.stage(value)[1],'BLOCKED_INVALID_TOKEN_SPANS')
    def test_bad_hypothesis_shape_is_atomic(self):
        self.original['assignments'][1]['alignment_hypotheses']=4
        original=copy.deepcopy(self.original)
        got,status=self.stage()
        self.assertEqual(status,'BLOCKED_HYPOTHESIS_SHAPE')
        self.assertIs(got,self.original)
        self.assertEqual(original,self.original)
    def test_idempotent_same_source_without_duplicate_hypotheses(self):
        first,status=self.stage()
        again,status2=stage_segment_constrained_ctc(
            first,records(),entry_count=2,source_revision='real-bounded-ctc-v1')
        self.assertEqual(status2,'STAGED_FOR_CENTRAL_NO_AUTHORITY')
        self.assertEqual(len(again['assignments'][0]['alignment_hypotheses']),2)
        self.assertEqual(first,again)


if __name__=='__main__':
    unittest.main()