"""Untrusted S1 CTC can schedule independent Whisper at missed passages."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import auto_lrc as a


class S1WhisperAcquisitionTests(unittest.TestCase):
    def capability(self):
        return a.LocalWhisperCapability(
            kind="whispercpp-whisperx-local",backend_lineage="whisper-family",
            audio_path="vocal.wav",audio_sha256="stem-sha",
            whisper_cli_path="cli",whisper_cli_sha256="cli-sha",
            whisper_model_path="model",whisper_model_sha256="model-sha",
            whisperx_python_path="python",whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha",language="ja",device="cuda",
            protocol_revision=a._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="independent-asr-capability",
        )
    def cand(self, row, time, source, *, current=False):
        spans=tuple(
            a.TimingTokenSpan(time+0.12*i,time+0.12*i+0.06,.6,c)
            for i,c in enumerate("abc"))
        return a.make_timing_candidate(
            entry_index=row,entry_text=f"unique line {row}",
            source=source,raw_time=time,spans=spans,confidence=.6,
            identity_support="supported",sequence_support="supported",
            current=current,source_artifact={"local":source,"row":row},
        )
    def setup_case(self, displaced=True):
        words=[a.LyricEntry([f"unique line {i}"]) for i in range(3)]
        base=(10.,12.,14.)
        targets=(50.,53.,56.) if displaced else (10.03,12.05,14.04)
        sets=tuple((
            self.cand(i,base[i],"ctc-current",current=True),
            self.cand(i,targets[i],"segment-constrained-ctc"),
        ) for i in range(3))
        report={"backend":"ctc","assignments":[
            {"entry":i+1,"timestamp":base[i],"score":.5,"segment":i}
            for i in range(3)]}
        return words,sets,report
    def test_probe_listens_to_far_s1_audio_not_collapsed_baseline(self):
        words,sets,report=self.setup_case()
        req=a.plan_local_whisper_requests(report,self.capability(),words,sets,100.)
        new=[x for x in req if x.window_start>45.]
        self.assertGreaterEqual(len(new),1)
        self.assertTrue(all(x.window_end<=61. for x in new))
        self.assertTrue(all(x.window_end-x.window_start<=a.LOCAL_WHISPER_MAX_WINDOW_SECONDS
                            for x in new))
        self.assertTrue(all(all(
            token in {c.candidate_id for c in sets[idx]
                      if c.source=="segment-constrained-ctc"}
            for idx,token in zip(x.context_entry_indexes,x.context_candidate_ids))
            for x in new))
        self.assertTrue(all(x.request_revision for x in new))
    def test_close_s1_does_not_spawn_second_whisper_probe(self):
        words,sets,report=self.setup_case(displaced=False)
        req=a.plan_local_whisper_requests(report,self.capability(),words,sets,100.)
        self.assertFalse(any(x.window_start>40. for x in req))
    def test_missing_s1_neighbor_prevents_partial_probe(self):
        words,sets,report=self.setup_case()
        sets=sets[:1]+((sets[1][0],),)+sets[2:]
        req=a.plan_local_whisper_requests(report,self.capability(),words,sets,100.)
        self.assertFalse(any(x.window_start>40. for x in req))
    def test_planner_creates_no_identity_claim_in_report(self):
        words,sets,report=self.setup_case()
        original=dict(report)
        a.plan_local_whisper_requests(report,self.capability(),words,sets,100.)
        self.assertEqual(report,original)
        self.assertFalse(any("occurrence_binding" in x for x in report['assignments']))


if __name__=="__main__":unittest.main()
