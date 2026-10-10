import unittest
from segment_asr_corroboration import check_asr_corroboration

class ASRCorroborationTests(unittest.TestCase):
    def test_refuse_when_paragraphs_match_old_clock_better(self):
        r=check_asr_corroboration(
            lyric_texts=['alpha','beta','gamma'],
            global_times=[1.,3.,5.],skeleton_times=[11.,13.,15.],
            raw_asr_events=[(0.,7.,'alpha beta gamma'),
                            (10.,16.,'random unrelated')])
        self.assertTrue(r['fallback'])
        self.assertEqual(r['reason'],'INSUFFICIENT_ASR_CORROBORATION')
    def test_accept_strongly_improved_raw_text(self):
        r=check_asr_corroboration(
            lyric_texts=['alpha','beta','gamma'],
            global_times=[1.,3.,5.],skeleton_times=[11.,13.,15.],
            raw_asr_events=[(0.,7.,'noise'),(10.,12.,'alpha'),
                            (12.,14.,'beta'),(14.,16.,'gamma')])
        self.assertFalse(r['fallback'])
        self.assertEqual(r['reason'],'CORROBORATED')
    def test_missing_raw_keeps_old_no_asr_path(self):
        r=check_asr_corroboration(
            lyric_texts=['alpha'],global_times=[1.],skeleton_times=[10.],
            raw_asr_events=())
        self.assertEqual(r['status'],'UNAVAILABLE')
        self.assertFalse(r['fallback'])
    def test_row_count_fails_closed(self):
        r=check_asr_corroboration(lyric_texts=['a'],global_times=[],
                                 skeleton_times=[1.],raw_asr_events=[(0,2,'a')])
        self.assertTrue(r['fallback'])
