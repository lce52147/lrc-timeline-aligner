"""Integration: production post-score vocal-only 3+ duplicate audit.

No pre-Central trust/clock promotion. Known stable multi-lyric repeat needs
actual onset observations from each real isolated-vocal segment.
"""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import auto_lrc as a


class ProductionPostscoreThreeRepeatsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stem=Path(self.tmp.name)/'vocal.wav'
        self.stem.write_bytes(b'audio-source-stub')
        self.entries=[a.LyricEntry([t]) for t in ('alpha','beta')*3]
        self.times=[2.,3.,12.,13.,22.,23.]
        self.report={
          'ctc_audio_path':str(self.stem),'ctc_audio_source':'vocal-stem',
          'backend':'ctc','suspicious_alignments':[],
          'assignments':[{'timestamp':t,'review_required':True,
                          'ctc_token_spans':[{'char':'a','start':t,'end':t+.03,'score':.75}]}
                         for t in self.times],
        }
    def run_audit(self,fail_index=None):
        saved=list(self.times)
        fake=SimpleNamespace(segments=[(1,25)])
        def fake_peak(features,t):
            if fail_index is not None and self.times[fail_index]==t:
                return {'time':t+1.,'delta_seconds':1.,'strength':.8}
            return {'time':t,'delta_seconds':0.,'strength':.8}
        def capture(assignment,**kwargs):
            assignment.setdefault('alignment_hypotheses',[]).append(kwargs)
        with patch.object(a,'analyze_audio',return_value=fake),\
             patch.object(a,'nearest_local_peak',side_effect=fake_peak),\
             patch.object(a,'record_alignment_hypothesis',side_effect=capture):
            changes=a.apply_post_score_audio_validation(
                 self.stem,self.entries,self.times,self.report,30.)
        self.assertEqual(self.times,saved)
        self.assertTrue(all(row['review_required'] for row in self.report['assignments']))
        return changes
    def test_all_three_occurrences_create_advisory_candidate_only(self):
        result=self.run_audit()
        self.assertEqual(len(result),6)
        self.assertEqual(self.report['duplicate_lyric_offset_consensus']['stable_cluster_count'],1)
        middle=self.report['assignments'][2]
        self.assertEqual(middle['audio_validation_resolution'],'duplicate-lyric-audio-onset-consensus')
        self.assertTrue(middle['audio_validation_duplicate_consensus']['consecutive'])
        self.assertTrue(middle['audio_validation_candidate_only'])
        self.assertEqual(len(middle['alignment_hypotheses']),1)
        self.assertEqual(middle['alignment_hypotheses'][0]['source'],'audio-validated-current')
        self.assertEqual(middle['alignment_hypotheses'][0]['direct_onset_support'],'unavailable')
    def test_false_vocal_peak_cannot_prove_one_repeat(self):
        result=self.run_audit(fail_index=4)
        self.assertTrue(all(row['entry']!=5 for row in result))
        self.assertNotIn('audio_validation_duplicate_consensus',self.report['assignments'][4])
        self.assertFalse(self.report['assignments'][4].get('audio_validation_candidate_only',False))
    def test_no_stem_means_no_positive_duplicate_audio_voting(self):
        self.report['ctc_audio_source']='full-mix'
        result=a.apply_post_score_audio_validation(
             self.stem,self.entries,self.times,self.report,30.)
        self.assertEqual(result,[])
        self.assertEqual(self.report['post_score_audio_validation_status'],
                          'skipped-no-isolated-vocal-stem')


if __name__=='__main__': unittest.main()
