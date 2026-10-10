"""S5 release gate cannot be faked with partial rows or simple PCO gains."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from s5_regression_gate import evaluate_s5_rows,evaluate_s5_corpus


class S5GateTests(unittest.TestCase):
    def rows(self):
        return [
          {"text":"intro","bk_s":10.,"main_s":10.,"candidate_s":10.,"main_trusted":True,"candidate_trusted":True},
          {"text":"refrain","bk_s":20.,"main_s":18.,"candidate_s":20.,"main_trusted":True,"candidate_trusted":True},
          {"text":"bridge","bk_s":30.,"main_s":30.,"candidate_s":30.,"main_trusted":True,"candidate_trusted":True},
          {"text":"refrain","bk_s":40.,"main_s":40.,"candidate_s":40.,"main_trusted":True,"candidate_trusted":True}
        ]
    def test_valid_fixes(self):
        result=evaluate_s5_rows(self.rows(),song_id="demo",expected_rows=4)
        self.assertTrue(result["eligible_for_S5"])
        self.assertEqual(result["counts"]["main_wrong_gt1s"],1)
        self.assertEqual(result["counts"]["candidate_wrong_gt1s"],0)
        self.assertEqual(result["counts"]["repeated_rows"],2)
        self.assertEqual(result["trusted"]["main_wrong_trusted_count"],1)
    def test_malformed_missing_fails_closed(self):
        self.assertEqual(evaluate_s5_rows(self.rows()[:2],song_id="demo",expected_rows=4)["status"],
                         "BLOCKED_MISSING_OR_PARTIAL_ROWS")
    def test_new_nonmonotonic_candidate_blocks_release(self):
        rows=self.rows();rows[2]["candidate_s"]=5.
        song=evaluate_s5_rows(rows,song_id="demo")
        self.assertEqual(song["new_backtrack_pairs"],[2])
        self.assertFalse(evaluate_s5_corpus([song],expected_songs=1)["checks"]["no_new_backtracks"])
    def test_inherited_main_backtrack_is_a_warning(self):
        rows=self.rows()
        rows[2]["main_s"]=5.
        rows[2]["candidate_s"]=5.
        song=evaluate_s5_rows(rows,song_id="inherited")
        self.assertEqual(song["inherited_backtrack_pairs"],[2])
        self.assertEqual(song["new_backtrack_pairs"],[])
        gate=evaluate_s5_corpus([song],expected_songs=1)
        self.assertTrue(gate["checks"]["no_new_backtracks"])
        self.assertEqual(gate["inherited_backtracks"][0]["reason"],"INHERITED_BACKTRACK")
    def test_missing_reference_clock_blocks(self):
        rows=self.rows();del rows[2]["bk_s"]
        self.assertEqual(evaluate_s5_rows(rows,song_id="demo")["status"],"BLOCKED_INVALID_CLOCK")
    def test_broken_original_main_does_not_block_scoring(self):
        rows=self.rows();rows[1]['main_s']=8.
        result=evaluate_s5_rows(rows,song_id='broken-main')
        self.assertEqual(result['status'],'POSTHOC_GATE_COMPUTED')
        self.assertEqual(result['counts']['main_wrong_gt1s'],1)

    def test_new_over_second_regression_fails(self):
        rows=self.rows();rows[2]["candidate_s"]=27.
        a=evaluate_s5_rows(rows,song_id="a")
        b=evaluate_s5_rows(self.rows(),song_id="b")
        final=evaluate_s5_corpus([a,b],expected_songs=2)
        self.assertFalse(final["release_pass"])
        self.assertFalse(final["checks"]["no_new_gt1s"])
        self.assertEqual(final["new_wrong_song_ids"],["a"])
    def test_full_corpus_gate_with_healthy_control(self):
        rows=self.rows()
        good=evaluate_s5_rows(rows,song_id="good")
        control=[
          {"text":"stable","bk_s":11.,"main_s":11.,"candidate_s":11.,
           "main_trusted":True,"candidate_trusted":True}
        ]
        ctl=evaluate_s5_rows(control,song_id="control")
        global_gate=evaluate_s5_corpus([good,ctl],expected_songs=2)
        self.assertTrue(global_gate["release_pass"])
        self.assertEqual(global_gate["counts"]["rows"],5)
    def test_duplicates_or_partial_songs_block(self):
        one=evaluate_s5_rows(self.rows(),song_id="same")
        self.assertEqual(evaluate_s5_corpus([one,one],expected_songs=2)["status"],
                         "BLOCKED_INCOMPLETE_S5_CORPUS")
        self.assertFalse(evaluate_s5_corpus([one],expected_songs=2)["release_pass"])
    def test_missing_reviewer_trust_prevents_release(self):
        rows=self.rows()
        del rows[1]["candidate_trusted"]
        result=evaluate_s5_rows(rows,song_id="a")
        self.assertFalse(evaluate_s5_corpus([result],expected_songs=1)["release_pass"])
    def test_tiny_PCO_gain_is_not_sufficient(self):
        rows=self.rows();rows[1]["candidate_s"]=20.;rows[2]["candidate_s"]=31.2
        outcome=evaluate_s5_rows(rows,song_id="shifted")
        checks=evaluate_s5_corpus([outcome],expected_songs=1)["checks"]
        self.assertTrue(checks["PCO_non_decreasing"])
        self.assertFalse(checks["no_new_gt1s"])


if __name__=="__main__":unittest.main()