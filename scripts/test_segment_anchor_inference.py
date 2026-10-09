"""S1 automatically inferred acoustic passage anchors: fail-closed invariants."""
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from segment_anchor_inference import (
    RepeatedSection, find_unambiguous_repeated_section,
    infer_repeated_passage_anchors, qualify_structural_repair,
)
from segment_structure import VocalPassage


class AnchorInferenceTests(unittest.TestCase):
    def test_kage_lexical_repeats_found_without_timestamps(self):
        chorus=['忘れて','シルエットダンス','知らない','法則','ただ舞う','今日を称えて']
        lines=['intro']*11+chorus+['verse']*25+chorus+['outro']*4
        pair,status=find_unambiguous_repeated_section(lines)
        self.assertEqual(pair,RepeatedSection(12,43,6))
        self.assertEqual(status,'LEXICAL_REPEAT_UNVERIFIED')

    def test_repeated_identical_single_line_is_ambiguous(self):
        lines=['verse','verse2']+['I am dreaming']*3+['other']*4+['I am dreaming']*4
        block,status=find_unambiguous_repeated_section(lines)
        self.assertIsNone(block)
        self.assertEqual(status,'BLOCKED_NO_UNAMBIGUOUS_REPEATED_LYRICS')

    def test_three_occurrences_fail_closed(self):
        lines=['x','y','z','one','a','b','one','a','b','one','a','b']
        block,status=find_unambiguous_repeated_section(lines)
        self.assertIsNone(block)
        self.assertTrue(status.startswith('BLOCKED_'))

    def test_ambiguous_audio_rejected(self):
        lyrics=['a','b','c','d','e','f','x','y','z','a','b','c','d','e','f']
        # All identical timbre -> every passage has a perfect repeated match.
        # Should never claim a unique ownership pair.
        features=np.full((1000,15),1/np.sqrt(15),dtype=np.float32)
        db=np.full(1000,-20,dtype=np.float32)
        passages=[VocalPassage(2+i*4,5+i*4) for i in range(4)]
        result=infer_repeated_passage_anchors(lyrics,passages,features,db,hop_seconds=.02)
        self.assertEqual(result.anchors,())

    def test_no_audio_never_makes_anchor(self):
        lines=['x','y','z','one','a','b','other','one','a','b']
        passages=[VocalPassage(0,3),VocalPassage(5,8)]
        result=infer_repeated_passage_anchors(lines,passages,
            np.ones((600,15),dtype=np.float32),
            np.full(600,-100,dtype=np.float32),hop_seconds=.02)
        self.assertEqual(result.anchors,())

    def test_automatic_activation_requires_two_broken_repeats(self):
        anchors=((12,17,3),(43,48,10))
        proposals=[float(i) for i in range(52)]
        ctcs=list(proposals)
        for a,b,_ in anchors:
            for row in range(a,b+1):ctcs[row-1]-=12.
        good=qualify_structural_repair(anchors,proposals,ctcs)
        self.assertEqual(good.status,'RESEARCH_TRIGGER_UNVERIFIED_OCCURRENCE')
        for row in range(43,49):ctcs[row-1]=proposals[row-1]-.5
        blocked=qualify_structural_repair(anchors,proposals,ctcs)
        self.assertEqual(blocked.status,'BLOCKED_NO_BILATERAL_STRUCTURAL_COLLAPSE')

    def test_clean_control_does_not_trigger_repair(self):
        proposals=[float(i*3) for i in range(40)]
        global_ctc=list(proposals)
        for row in (14,15,38,39):global_ctc[row-1]+=.8
        result=qualify_structural_repair(((13,16,1),(37,40,3)),proposals,global_ctc)
        self.assertEqual(result.status,'BLOCKED_NO_BILATERAL_STRUCTURAL_COLLAPSE')

    def test_nonfinite_inputs_blocked(self):
        chorus=['a','b','c']
        text=chorus+['x','y']+chorus
        frames=np.ones((1200,15),dtype=np.float32);frames[4]=np.nan
        res=infer_repeated_passage_anchors(text,
            [VocalPassage(1,6),VocalPassage(10,15)],
            frames,np.full(1200,-20,dtype=np.float32),hop_seconds=.02)
        self.assertEqual(res.status,'BLOCKED_INVALID_ACOUSTICS')


if __name__=='__main__':
    unittest.main()