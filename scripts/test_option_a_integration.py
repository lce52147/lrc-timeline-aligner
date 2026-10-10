"""Option A CLI/backend handshake and all-song fallback invariants."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import auto_lrc as a
from segment_skeleton_consensus import TestedRow,choose_acoustic_joint_passages


class OptionAIntegrationTests(unittest.TestCase):
    def _case(self,skeleton):
        with TemporaryDirectory() as temp:
            vocal=Path(temp)/"vocals.mp3"
            vocal.write_bytes(b"fake-stem")
            original=[1.,4.,7.,10.]
            report={"backend":"ctc","ctc_audio_source":"vocal-stem",
                "ctc_audio_path":str(vocal),
                "assignments":[{"entry":i+1,"timestamp":time}
                               for i,time in enumerate(original)]}
            args=SimpleNamespace(skeleton=skeleton,timing_source="auto",
                                 whisperx_device="cuda")
            with patch.object(a,"run_ctc_alignment",return_value=(original,report)),\
                 patch.object(a,"entry_sung_text",side_effect=lambda x:x),\
                 patch("segment_skeleton_production.run_segment_skeleton",
                   return_value=([2.,5.,8.,11.],{**report,"segment_skeleton_status":{
                     "status":"SEGMENT_CURRENT","whole_song":True}},
                   {"status":"SEGMENT_CURRENT"})) as segment:
                times,result,error=a.try_ctc_candidate(
                    Path(temp)/"song.flac",["one","two","three","four"],22.,args)
                return times,result,error,segment.call_count
    def test_segment_default_is_skeleton_current_no_bk(self):
        times,report,error,calls=self._case("segment")
        self.assertEqual(times,[2.,5.,8.,11.])
        self.assertEqual(report["segment_skeleton_status"]["status"],"SEGMENT_CURRENT")
        self.assertEqual(calls,1)
        self.assertIsNone(error)
    def test_explicit_ctc_bypasses_segment_producer(self):
        times,report,error,calls=self._case("ctc")
        self.assertEqual(times,[1.,4.,7.,10.])
        self.assertEqual(report["segment_skeleton_status"]["status"],"CTC_SELECTED")
        self.assertEqual(calls,0)
    def test_joint_dp_rejects_nonmonotonic_occurrence(self):
        rows=[
          (TestedRow(0,3.,.8,"A"),TestedRow(1,1.,.8,"A")),
          (TestedRow(1,7.,.8,"B"),TestedRow(2,9.,.8,"B"))]
        result=choose_acoustic_joint_passages(rows)
        self.assertEqual(result.status,"RESEARCH_JOINT_PATH_NO_AUTHORITY")
        self.assertTrue(all(b.time>a.time for a,b in zip(result.rows,result.rows[1:])))
    def test_joint_dp_rejects_conflicting_family(self):
        path=[TestedRow(0,5.,.5,"A"),TestedRow(0,5.,.6,"A")]
        result=choose_acoustic_joint_passages([path])
        self.assertEqual(result.status,"BLOCKED_NO_MONOTONIC_JOINT_PATH")


if __name__=="__main__":
    unittest.main()
