"""S1 production runner: all-or-nothing Central staging and cheap gates."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from segment_production import try_stage_structural_ctc
from segment_anchor_inference import (
    InferredAnchors, RepeatedSection, PassagePairEvidence, StructuralRepairActivation,
)
from segment_structure import VocalPassage, PassageBundle, PassageAlignment


class ProductionS1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.stem = self.folder / "stem.wav"
        self.helper = self.folder / "ctc_align.py"
        self.stem.write_bytes(b"stub")
        self.helper.write_text("stub", encoding="utf-8")
        self.text = ["intro"] + ["alpha", "beta", "gamma"] + ["verse"] * 7 + ["alpha", "beta", "gamma"] + ["outro"] * 4
        self.assertEqual(len(self.text), 18)
        self.report = {
            "backend": "ctc", "ctc_audio_source": "vocal-stem",
            "assignments": [{"entry": i+1, "timestamp": float(2 + i)}
                            for i in range(18)],
        }
        self.api = SimpleNamespace(
            decode_audio=lambda _: [0.1],
            frame_audio=lambda _: [0.1],
            analyze_audio=lambda *_: SimpleNamespace(segments=[(0, 12), (14, 29)]),
            SAMPLE_RATE=16000, HOP_SECONDS=.02,
        )
        self.anchor = InferredAnchors(
            "RESEARCH_UNVERIFIED_OCCURRENCE", ((2, 4, 0), (12, 14, 1)),
            RepeatedSection(2, 12, 3),
            (PassagePairEvidence(0, 1, .9, 14.0),), (),
        )
        first = PassageBundle(1, 9, 0, tuple(range(1, 10)), tuple(range(2, 5)))
        second = PassageBundle(10, 18, 1, tuple(range(14, 23)), tuple(range(12, 15)))
        self.dp = PassageAlignment(
            "UNVERIFIED_OCCURRENCE", (VocalPassage(0, 12), VocalPassage(14, 29)),
            (first, second), (0,) * 9 + (1,) * 9,
            tuple(range(1, 10)) + tuple(range(14, 23)), 0.1, (),
        )
        self.patches = [
            patch("segment_production.vocal_cepstral_features", return_value=([1], [1])),
            patch("segment_production.infer_repeated_passage_anchors", return_value=self.anchor),
            patch("segment_production.assign_lyrics_to_vocal_passages", return_value=self.dp),
            patch("segment_production.qualify_structural_repair", return_value=StructuralRepairActivation(
                "RESEARCH_TRIGGER_UNVERIFIED_OCCURRENCE", (7.0, 8.0), tuple(range(2, 5)))),
        ]
    def run_producer(self, **kw):
        params=dict(report=self.report,lyric_texts=self.text,vocal_stem=self.stem,
                    duration=60., audio_api=self.api, python_exe=Path(sys.executable),
                    ctc_helper=self.helper, device="cpu", helper_runner=self.fake_runner)
        params.update(kw)
        return try_stage_structural_ctc(**params)
    def fake_runner(self, cmd, **kw):
        lines=json.loads(Path(cmd[cmd.index("--transcript")+1]).read_text(encoding="utf-8"))
        lo=float(cmd[cmd.index("--start")+1])
        payload={"entries":[]}
        for i,t in enumerate(lines):
            stamp=round(lo+0.8+i*.75,3)
            payload["entries"].append({
                "start":stamp,"ctc_score":.32,"token_spans":[
                    {"char":"x","start":stamp,"end":stamp+.02,"score":.38}],
            })
        Path(cmd[cmd.index("--output")+1]).write_text(json.dumps(payload),encoding="utf-8")
        return SimpleNamespace(returncode=0,stderr="")
    def test_chorus_success_only_stages_hypotheses(self):
        with self.patches[0],self.patches[1],self.patches[2],self.patches[3]:
            after,diag=self.run_producer()
        self.assertEqual(diag["status"],"STAGED_FOR_CENTRAL_NO_AUTHORITY")
        self.assertEqual(diag["rows"],18)
        self.assertEqual(diag["bundles"],2)
        self.assertNotIn("alignment_hypotheses",self.report["assignments"][0])
        for row in after["assignments"]:
            item=row["alignment_hypotheses"][0]
            self.assertEqual(item["source"],"segment-constrained-ctc")
            self.assertEqual(item["acoustic_support"],"unavailable")
            self.assertEqual(item["direct_onset_support"],"unavailable")
            self.assertFalse(item["authoritative"])
    def test_ambiguous_repeated_lyric_is_cheap_no_gpu(self):
        clean_report=dict(self.report)
        clean_report["assignments"]=list(self.report["assignments"])
        with patch.object(self.api, "decode_audio", side_effect=AssertionError("should not decode")):
            out,diag=self.run_producer(lyric_texts=["not unique "+str(i) for i in range(18)])
        self.assertIs(out,self.report)
        self.assertTrue(diag["status"].startswith("BLOCKED_"))
    def test_non_ctc_is_unchanged(self):
        self.report["backend"]="whisperx"
        with patch.object(self.api,"decode_audio",side_effect=AssertionError("should not decode")):
            out,diag=self.run_producer()
        self.assertIs(out,self.report)
        self.assertEqual(diag["status"],"BLOCKED_NOT_ISOLATED_VOCAL_CTC")
    def test_missing_helper_is_unchanged(self):
        self.helper.unlink()
        out,diag=self.run_producer()
        self.assertIs(out,self.report)
        self.assertEqual(diag["status"],"BLOCKED_CTC_HELPER_MISSING")
    def test_bundle_budget_blocks_without_GPU(self):
        with self.patches[0],self.patches[1],self.patches[2],self.patches[3]:
            def forbid(*_args,**_kw):
                raise AssertionError("no GPU budget")
            out,diag=self.run_producer(max_bundles=1,helper_runner=forbid)
        self.assertIs(out,self.report)
        self.assertEqual(diag["status"],"BLOCKED_GPU_BUNDLE_BUDGET")
    def test_in_process_heavy_ctc_reuse_rebinds_without_rerunning(self):
        runner=self.fake_runner
        calls=[]
        def counted(cmd,**kwargs):
            calls.append(cmd)
            return runner(cmd,**kwargs)
        with self.patches[0],self.patches[1],self.patches[2],self.patches[3]:
            first,initial=self.run_producer(helper_runner=counted)
            self.assertEqual(initial["status"],"STAGED_FOR_CENTRAL_NO_AUTHORITY")
            self.assertFalse(initial["runtime_cache_hit"])
            count_before=len(calls)
            with patch.object(self.api,"decode_audio",side_effect=AssertionError("no repeat decode")):
                second,reused=self.run_producer(helper_runner=counted)
        self.assertEqual(len(calls),count_before)
        self.assertTrue(reused["runtime_cache_hit"])
        self.assertEqual(first,second)
        self.assertEqual(initial["source_revision"],reused["source_revision"])

    def test_cache_invalidated_by_changed_global_prior(self):
        calls=[]
        def counted(cmd,**kwargs):
            calls.append(cmd)
            return self.fake_runner(cmd,**kwargs)
        with self.patches[0],self.patches[1],self.patches[2],self.patches[3]:
            first,initial=self.run_producer(helper_runner=counted)
            count_before=len(calls)
            self.report["assignments"][0]["timestamp"]+=.7
            after,changed=self.run_producer(helper_runner=counted)
        self.assertGreater(len(calls),count_before)
        self.assertFalse(changed["runtime_cache_hit"])
        self.assertEqual(initial["status"],changed["status"])

    def test_cache_invalidated_by_source_file_change(self):
        calls=[]
        def counted(cmd,**kwargs):
            calls.append(cmd)
            return self.fake_runner(cmd,**kwargs)
        with self.patches[0],self.patches[1],self.patches[2],self.patches[3]:
            first,initial=self.run_producer(helper_runner=counted)
            count_before=len(calls)
            self.stem.write_bytes(b"modified-stem-bytes")
            after,changed=self.run_producer(helper_runner=counted)
        self.assertGreater(len(calls),count_before)
        self.assertFalse(changed["runtime_cache_hit"])

    def test_one_gpu_failure_rejects_full_song(self):
        def runner(cmd,**kw):
            if "passage_01" in cmd[cmd.index("--output")+1]:
                return SimpleNamespace(returncode=1,stderr="CUDA blocked")
            return self.fake_runner(cmd,**kw)
        with self.patches[0],self.patches[1],self.patches[2],self.patches[3]:
            out,diag=self.run_producer(helper_runner=runner)
        self.assertIs(out,self.report)
        self.assertEqual(diag["status"],"BLOCKED_LOCAL_CTC_FAILED")
    def test_corrupt_gpu_spans_rejects_full_song(self):
        def runner(cmd,**kw):
            done=self.fake_runner(cmd,**kw)
            file=Path(cmd[cmd.index("--output")+1])
            result=json.loads(file.read_text(encoding="utf-8"))
            if "passage_01" in str(file):
                result["entries"][0]["token_spans"]=[]
                file.write_text(json.dumps(result),encoding="utf-8")
            return done
        with self.patches[0],self.patches[1],self.patches[2],self.patches[3]:
            out,diag=self.run_producer(helper_runner=runner)
        self.assertIs(out,self.report)
        self.assertEqual(diag["status"],"BLOCKED_INCOMPLETE_CTC_EVIDENCE")


if __name__ == "__main__":
    unittest.main()