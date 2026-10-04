from __future__ import annotations

import unittest

import reviewer_layer


class ReviewerTrustProfileTests(unittest.TestCase):
    def _decision(self, times: dict[str, float | None], offsets: dict[str, float] | None = None) -> reviewer_layer.ReviewerRowDecision:
        reviewers = ("HUBP", "WX", "XLSR", "HUB")
        return reviewer_layer.classify_row(
            final_time=10.0,
            reviewer_times=times,
            offsets=offsets if offsets is not None else {name: 0.0 for name in reviewers},
            reviewers=reviewers,
        )

    def test_profiles_map_to_fixed_shipped_rules(self) -> None:
        strict = reviewer_layer.resolve_trust_profile("strict")
        balanced = reviewer_layer.resolve_trust_profile("balanced")
        loose = reviewer_layer.resolve_trust_profile("loose")
        self.assertEqual((strict.tau_percent, strict.selected_rule_id, strict.enabled), (2.0, "ALL_REVIEW", False))
        self.assertEqual((balanced.tau_percent, balanced.selected_rule_id, balanced.enabled), (5.0, "HUBP+WX+XLSR+HUB/>=3", True))
        self.assertEqual((loose.tau_percent, loose.selected_rule_id, loose.enabled), (10.0, "HUBP+WX+XLSR+HUB/>=2", True))

    def test_balanced_accepts_exactly_three_of_four_and_rejects_two(self) -> None:
        three = self._decision({"HUBP": 10.00, "WX": 10.01, "XLSR": 10.05, "HUB": 10.20})
        two = self._decision({"HUBP": 10.00, "WX": 10.05, "XLSR": 10.06, "HUB": 10.20})
        self.assertTrue(reviewer_layer.trusted_by_profile(three, profile="balanced"))
        self.assertFalse(reviewer_layer.trusted_by_profile(two, profile="balanced"))

    def test_loose_accepts_exactly_two_of_four_and_rejects_one(self) -> None:
        two = self._decision({"HUBP": 10.00, "WX": 10.05, "XLSR": 10.06, "HUB": 10.20})
        one = self._decision({"HUBP": 10.00, "WX": 10.051, "XLSR": 10.06, "HUB": 10.20})
        self.assertTrue(reviewer_layer.trusted_by_profile(two, profile="loose"))
        self.assertFalse(reviewer_layer.trusted_by_profile(one, profile="loose"))

    def test_missing_required_reviewer_fails_closed_even_with_three_agreements(self) -> None:
        missing_hub = self._decision({"HUBP": 10.00, "WX": 10.01, "XLSR": 10.02, "HUB": None})
        self.assertFalse(reviewer_layer.trusted_by_profile(missing_hub, profile="balanced"))
        self.assertFalse(reviewer_layer.trusted_by_profile(missing_hub, profile="loose"))

    def test_missing_offset_is_missing_reviewer_not_zero_offset(self) -> None:
        decision = self._decision(
            {"HUBP": 10.00, "WX": 10.01, "XLSR": 10.02, "HUB": 10.03},
            offsets={"HUBP": 0.0, "WX": 0.0, "XLSR": 0.0},
        )
        self.assertIn("HUB", decision.missing_reviewers)
        self.assertEqual(decision.present_count, 3)
        self.assertFalse(reviewer_layer.trusted_by_profile(decision, profile="balanced"))

    def test_default_profile_marks_no_reviewer_trust(self) -> None:
        self.assertEqual(reviewer_layer.DEFAULT_TRUST_PROFILE, "none")
        unanimous = self._decision({"HUBP": 10.0, "WX": 10.0, "XLSR": 10.0, "HUB": 10.0})
        self.assertFalse(reviewer_layer.trusted_by_profile(unanimous))
        self.assertFalse(reviewer_layer.trusted_by_profile(unanimous, profile="strict"))

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reviewer_layer.resolve_trust_profile("aggressive")


if __name__ == "__main__":
    unittest.main()
