"""GPU weak-passage-edge search: acoustic score and atomicity invariants."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent))
from segment_boundary_search import compete_ctc_adjacent_boundaries
from segment_structure import VocalPassage,PassageBundle


class BoundarySearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder=Path(self.tmp.name)
        self.bundles=[
            PassageBundle(1,2,0,(.8,1.7),()),
            PassageBundle(3,4,1,(3.6,4.5),()),
        ]
        self.passages=[VocalPassage(0,2.8),VocalPassage(3,6)]
        self.words=["alpha","beta","gamma","delta"]
        self.rows=[
            {"row":1,"time":.8,"ctc_score":.5,"passage":0,"token_spans":[{"start":.8,"end":.82,"score":.5}]},
            {"row":2,"time":1.7,"ctc_score":.005,"passage":0,"token_spans":[{"start":1.7,"end":1.72,"score":.005}]},
            {"row":3,"time":3.6,"ctc_score":.55,"passage":1,"token_spans":[{"start":3.6,"end":3.62,"score":.55}]},
            {"row":4,"time":4.5,"ctc_score":.53,"passage":1,"token_spans":[{"start":4.5,"end":4.52,"score":.53}]},
        ]
    def fake_runner(self,cmd,**kw):
        lines=json.loads(Path(cmd[cmd.index("--transcript")+1]).read_text(encoding="utf-8"))
        path=Path(cmd[cmd.index("--output")+1])
        scores=(.51,) if len(lines)==1 else (.55,.57,.56)
        starts=(.8,) if len(lines)==1 else (3.12,3.72,4.32)
        data={"entries":[{"start":start,"ctc_score":score,
                          "token_spans":[{"char":"a","start":start,"end":start+.02,"score":score}]}
                          for start,score in zip(starts,scores)]}
        path.write_text(json.dumps(data),encoding="utf-8")
        return SimpleNamespace(returncode=0,stderr="")
    def run_search(self,**opts):
        kw=dict(rows=self.rows,bundles=self.bundles,passages=self.passages,
                lyric_texts=self.words,anchored=(),stem=self.folder/"stem.wav",
                duration=6.0,python_exe=self.folder/"python",
                helper=self.folder/"ctc.py",device="cuda",temp_dir=self.folder,
                runner=self.fake_runner)
        kw.update(opts)
        return compete_ctc_adjacent_boundaries(**kw)
    def test_repair_weak_edge_without_reference_clock(self):
        changed,diag=self.run_search()
        self.assertEqual(diag["status"],"BOUNDARY_RESEARCH_COMPLETE")
        self.assertEqual(diag["proposals"],1)
        self.assertEqual(diag["decisions"][0]["status"],"RESEARCH_ACCEPTED_NOT_CENTRAL")
        self.assertEqual(changed[1]["passage"],1)
        self.assertAlmostEqual(changed[1]["time"],3.12)
        self.assertEqual(self.rows[1]["passage"],0)
    def test_persistently_weak_alt_is_rejected(self):
        def weak(cmd,**kwargs):
            result=self.fake_runner(cmd,**kwargs)
            dest=Path(cmd[cmd.index("--output")+1])
            body=json.loads(dest.read_text(encoding="utf-8"))
            for x in body["entries"]:x["ctc_score"]=.01
            dest.write_text(json.dumps(body),encoding="utf-8")
            return result
        got,diag=self.run_search(runner=weak)
        self.assertEqual(diag["decisions"][0]["status"],"REJECTED_BOUNDARY")
        self.assertEqual(got,self.rows)
    def test_fail_cuda_is_fail_closed(self):
        got,diag=self.run_search(runner=lambda *a,**k:SimpleNamespace(returncode=1,stderr="CUDA error"))
        self.assertIsNone(got)
        self.assertEqual(diag["status"],"BLOCKED_LOCAL_BOUNDARY_CTC_FAILED")
    def test_anchor_prevents_move(self):
        # The weak left-tail is part of an independently unresolved repeat,
        # so even a high-score alternative cannot silently remap it.
        got,diag=self.run_search(anchored=((1,2,0),))
        self.assertEqual(got,self.rows)
        self.assertEqual(diag["proposals"],0)
    def test_proposal_budget_fail_closed(self):
        got,diag=self.run_search(max_proposals=0)
        self.assertIsNone(got)
        self.assertEqual(diag["status"],"BLOCKED_TOO_MANY_WEAK_BOUNDARIES")


if __name__=="__main__":
    unittest.main()
