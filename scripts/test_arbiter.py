import unittest

import arbiter


class GrossRescueArbiterTests(unittest.TestCase):
    def decide(
        self,
        *,
        current_valid: bool = True,
        current_cs: int = 1000,
        selected_cs: int = 1060,
        selected_source: str = "ctc-local",
    ) -> arbiter.GrossRescueDecision:
        return arbiter.decide_gross_rescue(
            current_valid=current_valid,
            current_centiseconds=current_cs,
            selected_centiseconds=selected_cs,
            selected_source=selected_source,
        )

    def test_invalid_current_never_overrides_central_selection(self) -> None:
        decision = self.decide(current_valid=False, selected_cs=2000)
        self.assertFalse(decision.use_current)
        self.assertEqual(decision.reason, "current-invalid")

    def test_valid_current_is_preserved_at_or_below_500ms(self) -> None:
        decision = self.decide(selected_cs=1050)
        self.assertTrue(decision.use_current)
        self.assertEqual(decision.shift_ms, 500)
        self.assertEqual(decision.reason, "within-gross-rescue-threshold")

    def test_valid_current_is_preserved_for_registered_asr_fusion(self) -> None:
        for source in sorted(arbiter.ASR_FUSION_SOURCES):
            with self.subTest(source=source):
                decision = self.decide(selected_cs=1060, selected_source=source)
                self.assertTrue(decision.use_current)
                self.assertEqual(decision.reason, "registered-asr-fusion")

    def test_large_non_asr_shift_keeps_central_selection(self) -> None:
        decision = self.decide(selected_cs=1060, selected_source="ctc-local")
        self.assertFalse(decision.use_current)
        self.assertEqual(decision.shift_ms, 600)
        self.assertEqual(decision.reason, "gross-rescue-eligible")


class ReviewerValidityArbiterTests(unittest.TestCase):
    def test_reviewer_agreements_can_score_non_current_candidate(self) -> None:
        agreeing, present = arbiter.reviewer_agreements_at(
            candidate_seconds=9.50,
            reviewer_times={"HUBP": 9.52, "WX": 9.49, "XLSR": 10.01},
            offsets={"HUBP": 0.02, "WX": 0.0, "XLSR": 0.01},
            reviewers=("HUBP", "WX", "XLSR"),
        )
        self.assertEqual(agreeing, ("HUBP", "WX"))
        self.assertEqual(present, ("HUBP", "WX", "XLSR"))

    def test_one_corrected_reviewer_can_endorse_r2b(self) -> None:
        decision = arbiter.decide_reviewer_validity(
            current_seconds=10.0,
            expected_current_seconds=10.0,
            reviewer_times={"HUBP": 10.08, "WX": None, "XLSR": 11.0},
            offsets={"HUBP": 0.03, "XLSR": 0.0},
            reviewers=("HUBP", "WX", "XLSR"),
            required_agreements=1,
        )
        self.assertTrue(decision.endorsed)
        self.assertEqual(decision.agreeing_reviewers, ("HUBP",))

    def test_missing_or_unqualified_reviewer_is_not_a_vote(self) -> None:
        decision = arbiter.decide_reviewer_validity(
            current_seconds=10.0,
            expected_current_seconds=10.0,
            reviewer_times={"HUBP": 10.01},
            offsets={},
            reviewers=("HUBP",),
            required_agreements=1,
        )
        self.assertFalse(decision.endorsed)

    def test_stale_current_binding_is_not_endorsed(self) -> None:
        decision = arbiter.decide_reviewer_validity(
            current_seconds=10.1,
            expected_current_seconds=10.0,
            reviewer_times={"HUBP": 10.01},
            offsets={"HUBP": 0.01},
            reviewers=("HUBP",),
            required_agreements=1,
        )
        self.assertFalse(decision.endorsed)
        self.assertEqual(decision.reason, "current-binding-mismatch")

    def test_only_registered_soft_failures_can_be_relaxed(self) -> None:
        soft = ("previous-tail-ownership-unknown", "insufficient-minimum-proof")
        self.assertTrue(arbiter.reviewer_validity_can_relax(soft, endorsed=True))
        self.assertFalse(
            arbiter.reviewer_validity_can_relax(
                (*soft, "temporal-incoherence"), endorsed=True
            )
        )
        self.assertFalse(arbiter.reviewer_validity_can_relax(soft, endorsed=False))


if __name__ == "__main__":
    unittest.main()
