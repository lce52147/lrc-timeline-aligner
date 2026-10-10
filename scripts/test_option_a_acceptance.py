import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from option_a_acceptance import Score,score_song,gate_a6,FIXED_SONGS


class A6GateTests(unittest.TestCase):
    def entry(self,errors,*,song="04 影色舞",mode="segment"):
        b=[("line "+str(i),i*400) for i in range(len(errors))]
        n=[(text,time+err) for (text,time),err in zip(b,errors)]
        return score_song(song=song,mode=mode,labels=b,produced=n,baseline=b,
                          segment_status="SEGMENT_CURRENT")
    def test_strict_identity_is_not_silent(self):
        with self.assertRaisesRegex(ValueError,"STRICT_LYRIC_IDENTITY"):
            score_song(song="x",mode="segment",labels=[("a",100)],
                       produced=[("b",100)])
    def test_nonnumeric_clock_rejected(self):
        with self.assertRaisesRegex(ValueError,"INVALID_TIMESTAMPS"):
            score_song(song="x",mode="segment",labels=[("a",100)],
                       produced=[("a",1.5)])
    def test_new_wrong_gt1_is_strict(self):
        a=self.entry([0,0,101,102])
        self.assertEqual(a.new_big_errors,(3,4))
    def test_missing_mode_refuses_pass(self):
        result=gate_a6([self.entry([0,0])])
        self.assertEqual(result["status"],"INCOMPLETE_A6_PAIRING")
        self.assertEqual(len(result["missing"]),13)
    def test_fixed_seven_perfect_pass(self):
        outcomes=[]
        for song in FIXED_SONGS:
            n=52 if song=="04 影色舞" else 30
            for mode in ("segment","ctc"):
                outcomes.append(self.entry([0]*n,song=song,mode=mode))
        result=gate_a6(outcomes)
        self.assertEqual(result["status"],"PASS_A6_REPRESENTATIVE")
    def test_normal_control_new_gt1_blocks(self):
        outcomes=[]
        for song in FIXED_SONGS:
            n=52 if song=="04 影色舞" else 30
            for mode in ("segment","ctc"):
                errors=[0]*n
                if mode=="segment" and song=="10.方舟":errors[3]=102
                outcomes.append(self.entry(errors,song=song,mode=mode))
        result=gate_a6(outcomes)
        self.assertEqual(result["status"],"BLOCKED_A6")
        self.assertTrue(any(x["reason"]=="CONTROL_NEW_GT1" for x in result["blockers"]))


if __name__=="__main__":unittest.main()
