import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import auto_lrc as a

class IsolatedPostscoreTests(unittest.TestCase):
 def test_no_isolated_stem_skips_without_fullmix(self):
  rows=[a.LyricEntry(['one']),a.LyricEntry(['two'])]
  report={'assignments':[{'review_required':True},{'review_required':False}],
          'suspicious_alignments':[],'ctc_audio_source':'mix'}
  with patch.object(a,'analyze_audio') as analyze:
   result=a.apply_post_score_audio_validation(Path('fullmix.flac'),rows,[1.,2.],report,10.)
  self.assertEqual(result,[])
  analyze.assert_not_called()
  self.assertEqual(report['post_score_audio_validation_status'],'skipped-no-isolated-vocal-stem')
 def test_isolated_stem_checks_all_rows(self):
  rows=[a.LyricEntry(['one']),a.LyricEntry(['two']),a.LyricEntry(['three'])]
  with tempfile.TemporaryDirectory() as tmp:
   vocal=Path(tmp)/'vocals.mp3';vocal.write_bytes(b'fake')
   report={'assignments':[{'review_required':False} for _ in rows],
           'suspicious_alignments':[],'ctc_audio_source':'vocal-stem','ctc_audio_path':str(vocal)}
   fake=object()
   with patch.object(a,'analyze_audio',return_value=fake) as analyze, \
        patch.object(a,'nearest_local_peak',return_value={'time':1.,'delta_seconds':0.,'strength':.8}):
    a.apply_post_score_audio_validation(Path('fullmix.flac'),rows,[1.,2.,3.],report,10.)
   self.assertEqual(analyze.call_args.args[0],vocal)
   self.assertEqual(len(report['post_score_all_line_vocal_audit']),3)
   self.assertEqual(report['post_score_audio_validation_status'],'isolated-vocal-all-lines-checked')
   self.assertTrue(all(x['acoustic_onset_nearby'] for x in report['post_score_all_line_vocal_audit']))
if __name__=='__main__':unittest.main()
