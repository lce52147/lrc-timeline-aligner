import unittest
from types import SimpleNamespace
from segment_peer_outlier import exclude_uncorroborated_whisperx_fusion as choose

def candidate(source,time,with_evidence=True):
    return SimpleNamespace(
        source=source,written_time=SimpleNamespace(seconds=time),
        acoustic_evidence=object() if with_evidence else None,
        direct_onset_evidence=object() if with_evidence else None)

class SegmentPeerOutlierTests(unittest.TestCase):
    def test_distant_whisperx_conflicts_with_raw_ctc_consensus(self):
        c=candidate("ctc-vocal-onset-confirmed",46.41)
        r=candidate("raw-vocal-independent-fusion",46.1)
        w=candidate("whisperx-vocal-independent-fusion",45.32)
        kept,discarded=choose([c,r,w])
        self.assertEqual(kept,(c,r))
        self.assertEqual(discarded,(w,))
    def test_no_raw_evidence_means_no_exclusion(self):
        c=candidate("ctc-vocal-onset-confirmed",46.41)
        w=candidate("whisperx-vocal-independent-fusion",45.32)
        self.assertEqual(choose([c,w]),((c,w),()))
    def test_nearby_whisperx_is_preserved(self):
        c=candidate("ctc-vocal-onset-confirmed",46.41)
        r=candidate("raw-vocal-independent-fusion",46.1)
        w=candidate("whisperx-vocal-independent-fusion",45.91)
        self.assertEqual(choose([c,r,w]),((c,r,w),()))
    def test_unsupported_raw_candidate_cannot_exclude(self):
        c=candidate("ctc-vocal-onset-confirmed",46.41)
        r=candidate("raw-vocal-independent-fusion",46.1,False)
        w=candidate("whisperx-vocal-independent-fusion",45.32)
        self.assertEqual(choose([c,r,w]),((c,r,w),()))
    def test_distant_ctc_raw_pair_cannot_exclude(self):
        c=candidate("ctc-vocal-onset-confirmed",46.41)
        r=candidate("raw-vocal-independent-fusion",44.1)
        w=candidate("whisperx-vocal-independent-fusion",45.32)
        self.assertEqual(choose([c,r,w]),((c,r,w),()))
