"""S1 DP invariant tests: no BK anchors or acoustic ownership fabrication."""
import unittest

from segment_structure import (
    VocalPassage, PassageBundle, merge_vocal_passages,
    assign_lyrics_to_vocal_passages,validate_passage_bundles,choose_atomic_central_passage,
)

class VocalPassageTests(unittest.TestCase):
    def test_gap_merging_keeps_song_silence(self):
        p=merge_vocal_passages([(2,4),(4.3,5.7),(12,14),(15,15.2)],duration=20)
        self.assertEqual([(x.start,x.end) for x in p],[(2.,5.7),(12.,14.)])
    def test_many_lines_per_passage(self):
        d=assign_lyrics_to_vocal_passages([(1,8),(20,28)],duration=30,
            lyric_lengths=[10,9,8,7],ctc_priors=[2,3,21,22])
        self.assertEqual(d.status,"UNVERIFIED_OCCURRENCE")
        self.assertEqual(len(d.entry_passages),4)
        self.assertTrue(all(d.entry_passages[i]<=d.entry_passages[i+1] for i in range(3)))
    def test_ctc_is_weak_prior_not_hard_limit(self):
        p=[(1.,5.),(15.,20.),(31.,38.)]
        a=assign_lyrics_to_vocal_passages(p,duration=42,lyric_lengths=[22,22,22],
            ctc_priors=[1.2,1.3,1.4])
        self.assertEqual(a.status,"UNVERIFIED_OCCURRENCE")
        self.assertEqual(len(a.entry_times),3)
        self.assertTrue(all(any(left<=t<=right for left,right in p) for t in a.entry_times))
    def test_no_silent_candidate(self):
        d=assign_lyrics_to_vocal_passages([],duration=35,lyric_lengths=[1])
        self.assertEqual(d.status,"BLOCKED_NO_SINGING_OR_LYRICS")
    def test_atomic_postcentral_refuses_partial_chorus(self):
        kw=dict(passage=VocalPassage(60,76),baseline_times=[90.,91.,92.],
            local_ctc_times=[61.0,62.5,66.3],
            local_ctc_candidate_ids=["a","b","c"],
            independent_occurrence_witnesses=[True,True,True],
            previous_bound=58.,next_bound=88.)
        selected,status=choose_atomic_central_passage(**kw,
            central_selected_ids=["a","old","c"])
        self.assertEqual(status,"REJECTED_PARTIAL_CENTRAL_ACCEPTANCE")
        self.assertEqual(selected,(90.,91.,92.))
        complete,status=choose_atomic_central_passage(**kw,
            central_selected_ids=["a","b","c"])
        self.assertEqual(status,"ACCEPTED_ATOMIC_BY_CENTRAL")
        self.assertEqual(complete,(61.,62.5,66.3))

    def test_at_least_one_unproved_row_blocks_whole_section(self):
        result,status=choose_atomic_central_passage(
            passage=VocalPassage(60,76),baseline_times=[90.,91.],
            local_ctc_times=[61.,62.],
            local_ctc_candidate_ids=["a","b"],central_selected_ids=["a","b"],
            independent_occurrence_witnesses=[True,False])
        self.assertEqual(status,"REJECTED_UNPROVED_OCCURRENCE")
        self.assertEqual(result,(90.,91.))
    def test_no_unowned_commit(self):
        b=(PassageBundle(1,2,0,(1.2,2.2),()),)
        status,errors=validate_passage_bundles(
            (VocalPassage(1.,4.),),b,2,required_independent_ownership=True)
        self.assertEqual(status,"REJECTED_ATOMIC")
        self.assertIn("independent-occurrence-proof-missing",errors)
    def test_atomic_rejects_one_wrong_row(self):
        b=(PassageBundle(1,2,0,(1.2,15.2),()),)
        status,errors=validate_passage_bundles((VocalPassage(1,4),),b,2,
            required_independent_ownership=True,proof_rows=frozenset([1,2]))
        self.assertEqual(status,"REJECTED_ATOMIC")
        self.assertIn("outside-vocal-passage",errors)
    def test_atomic_rejects_cross_passage_time_reverse(self):
        b=(PassageBundle(1,1,1,(20,),()),PassageBundle(2,2,0,(2,),()))
        status,errors=validate_passage_bundles(
            (VocalPassage(1,5),VocalPassage(18,25)),b,2,
            required_independent_ownership=False)
        self.assertEqual(status,"REJECTED_ATOMIC")
        self.assertIn("passage-order-not-strict",errors)
    def test_gap_over_segment_capacity_is_blocked(self):
        d=assign_lyrics_to_vocal_passages([(0,2)],duration=4,lyric_lengths=[1]*13,
            max_lines_per_passage=12)
        self.assertEqual(d.status,"BLOCKED_INSUFFICIENT_PASSAGE_CAPACITY")
    def test_ctc_wrong_length_is_rejected(self):
        with self.assertRaises(ValueError):
            assign_lyrics_to_vocal_passages([(0,5)],duration=8,
                lyric_lengths=[5,5],ctc_priors=[1])
if __name__=="__main__":
    unittest.main()
