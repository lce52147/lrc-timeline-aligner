"""Safe duplicate >2 vocal consensus, no added trust or clock updates."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from duplicate_audio_consensus import consecutive_duplicate_vocal_support


def peaks(*indexes):
    return {i:{"delta_seconds":0.03,"time":float(i),"strength":.8} for i in indexes}


class AudioDuplicateConsensusTests(unittest.TestCase):
    def test_pair_matches_existing_two_repeat_geometry(self):
        positive,detail=consecutive_duplicate_vocal_support(
         lyric_indexes=[2,8],current_index=8,timestamps=[float(i*2) for i in range(10)],
         peaks=peaks(2,8),stable_cluster_medians={6:12.})
        self.assertTrue(positive)
        self.assertEqual(detail["entries"],[3,9])
    def test_three_occurrences_first_middle_last_can_have_support(self):
        times=[float(i)*2 for i in range(16)]
        for n in (3,8,13):
            positive,detail=consecutive_duplicate_vocal_support(
             lyric_indexes=[3,8,13],current_index=n,timestamps=times,
             peaks=peaks(3,8,13),stable_cluster_medians={5:10.})
            self.assertTrue(positive)
            if n==8:self.assertTrue(detail["consecutive"])
    def test_repeated_single_line_has_no_independent_cluster(self):
        positive,detail=consecutive_duplicate_vocal_support(
         lyric_indexes=[3,8,13],current_index=8,timestamps=[float(i)*2 for i in range(16)],
         peaks=peaks(3,8,13),stable_cluster_medians={})
        self.assertFalse(positive)
    def test_early_false_peak_does_not_certify_duplicate(self):
        p=peaks(3,8,13);p[13]["delta_seconds"]=2.0
        positive,detail=consecutive_duplicate_vocal_support(
         lyric_indexes=[3,8,13],current_index=13,timestamps=[float(i)*2 for i in range(16)],
         peaks=p,stable_cluster_medians={5:10.})
        self.assertFalse(positive)
    def test_nonconsecutive_pair_does_not_gain_support(self):
        positive,_=consecutive_duplicate_vocal_support(
         lyric_indexes=[3,8,13],current_index=13,timestamps=[float(i)*2 for i in range(16)],
         peaks=peaks(3,13),stable_cluster_medians={10:20.})
        self.assertFalse(positive)
    def test_invalid_row_order_rejected(self):
        positive,detail=consecutive_duplicate_vocal_support(
         lyric_indexes=[8,3,13],current_index=8,timestamps=[float(i)*2 for i in range(16)],
         peaks=peaks(3,8,13),stable_cluster_medians={5:10.})
        self.assertFalse(positive)


if __name__=='__main__': unittest.main()
