"""Writer-level atomic Option A rollback is strict and without Central edits."""
import unittest
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import auto_lrc as a


class OptionARollbackTests(unittest.TestCase):
    def times(self,*s):
        return tuple(a.canonicalize_lrc_time(t) for t in s)
    def test_segment_backtrack_detects_row_and_rejects_write(self):
        got=a.segment_postcentral_nonmonotonic(
          {"segment_skeleton_status":{"status":"SEGMENT_CURRENT"}},
          self.times(20.1,21.2,19.7,22.))
        self.assertEqual(got,(3,))
    def test_ctc_fallback_never_restarts(self):
        got=a.segment_postcentral_nonmonotonic(
          {"segment_skeleton_status":{"status":"FALLBACK_CTC"}},
          self.times(20.1,21.2,19.7,22.))
        self.assertEqual(got,())
    def test_equal_times_dont_trigger_backward_false_positive(self):
        got=a.segment_postcentral_nonmonotonic(
          {"segment_skeleton_status":{"status":"SEGMENT_CURRENT"}},
          self.times(20.1,20.1,20.12))
        self.assertEqual(got,())
    def test_only_full_original_mode_is_retried(self):
        args=a.build_parser().parse_args(["X:/fake.flac","--skeleton","segment"])
        called=[]
        def fake_process(_p,argument):
            called.append((argument.skeleton,getattr(argument,"option_a_rollback_reason",None)))
            if len(called)==1:
                raise a.SegmentSkeletonRollback("segment Central backtracks #12")
        with patch.object(a,"process_audio",side_effect=fake_process):
            exitcode=a.main(["X:/fake.flac","--skeleton","segment"])
        self.assertEqual(exitcode,0)
        self.assertEqual([x[0] for x in called],["segment","ctc"])
        self.assertEqual(called[1][1],"segment Central backtracks #12")


if __name__=="__main__":unittest.main()
