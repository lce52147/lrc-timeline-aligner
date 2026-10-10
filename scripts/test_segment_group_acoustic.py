import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from segment_structure import VocalPassage
from segment_group_acoustic import (
    placement_decision,contiguous_passage_groups,check_group_placement)

class GroupAcousticTests(unittest.TestCase):
    def test_margin_competes_against_neighbor(self):
        self.assertTrue(placement_decision(.40,[.31,.35]))
        self.assertFalse(placement_decision(.40,[.39,.12]))
        self.assertFalse(placement_decision(.40,[float('nan')]))
    def test_monotonic_group_builder(self):
        self.assertEqual(contiguous_passage_groups([0,0,2,2,3]),
                         [(0,2,0),(2,4,2),(4,5,3)])
    def _runner(self,score):
        def run(cmd,*,timeout):
            out=Path(cmd[cmd.index('--output')+1])
            transcript=json.loads(Path(cmd[cmd.index('--transcript')+1]).read_text(encoding='utf8'))
            out.write_text(json.dumps({'entries':[{'ctc_score':score} for _ in transcript]}),encoding='utf8')
            return SimpleNamespace(returncode=0,stderr='')
        return run
    def _invoke(self,alt):
        return check_group_placement(
            rows=[{'ctc_score':.40}]*3,
            passage_ids=[0,1,2],passages=[VocalPassage(0,1),VocalPassage(2,3),VocalPassage(4,5)],
            lyric_texts=['aa','bb','cc'],stem=Path('unused-stem'),
            duration=6.,python=Path('unused-python'),helper=Path('unused-helper'),
            device='auto',runner=self._runner(alt))
    def test_wrong_passage_group_gets_rejected(self):
        d=self._invoke(.41)
        self.assertEqual(d['status'],'OK')
        self.assertTrue(d['fallback'])
        self.assertEqual(d['unsupported_rows'],[1,2,3])
        self.assertEqual(len(d['groups']),3)
    def test_strong_assigned_acoustics_supported(self):
        d=self._invoke(.10)
        self.assertEqual(d['status'],'OK')
        self.assertFalse(d['fallback'])
        self.assertEqual(d['unsupported_rows'],[])
    def test_unproven_group_never_timing_trusted(self):
        from auto_lrc import assignment_timing_is_trusted
        import auto_lrc
        entry={'entry':1,'timestamp':12.3,'score':.9,
               'group_placement_unsupported':True}
        self.assertFalse(assignment_timing_is_trusted(entry))
        self.assertTrue(auto_lrc._suspicious_row_requires_review({
            'flags':['group_placement_unsupported'],'severity':'high',
            'review_required':True}))

    def test_p6_weak_ctc_untrusted_and_actionable(self):
        import auto_lrc
        item={'entry':1,'timestamp':10.0,'ctc_score':.004,
              'segment_weak_ctc_untrusted':True}
        self.assertFalse(auto_lrc.assignment_timing_is_trusted(item))
        row={'entry':1,'severity':'high',
             'flags':['segment_weak_ctc_support'],'review_required':True}
        self.assertTrue(auto_lrc._suspicious_row_requires_review(row))

    def test_invalid_row_map_fails_closed(self):
        x=check_group_placement(
            rows=[{'ctc_score':.3}]*2,passage_ids=[1,0],
            passages=[VocalPassage(0,1),VocalPassage(2,3)],
            lyric_texts=['a','b'],stem=Path('unused-stem'),duration=4.,
            python=Path('unused-python'),helper=Path('unused-helper'),
            device='auto',runner=self._runner(.1))
        self.assertEqual(x['status'],'ERROR')
