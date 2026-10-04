from __future__ import annotations

import unittest

import reviewer_layer
import run_reviewer_analysis


class ReviewerLayerTests(unittest.TestCase):
    def test_estimate_offset_uses_only_sub_500ms_rows(self) -> None:
        offset = reviewer_layer.estimate_offset_seconds(
            [10.0, 20.0, 30.0, 40.0],
            [10.02, 20.04, 31.20, None],
        )
        self.assertAlmostEqual(offset, 0.03, places=6)

    def test_all_available_agree_is_l3_with_missing_reported(self) -> None:
        decision = reviewer_layer.classify_row(
            final_time=10.0,
            reviewer_times={"HUBP": 10.03, "HUB": None, "WX": 9.98},
            offsets={"HUBP": 0.0, "HUB": 0.0, "WX": 0.0},
            reviewers=("HUBP", "HUB", "WX"),
        )
        self.assertEqual(decision.label, "L3")
        self.assertEqual(decision.present_count, 2)
        self.assertEqual(decision.agree_count, 2)
        self.assertEqual(decision.missing_reviewers, ("HUB",))

    def test_mixed_votes_map_to_l2_l1_l0(self) -> None:
        reviewers = ("HUBP", "HUB", "WX")
        offsets = {name: 0.0 for name in reviewers}
        l2 = reviewer_layer.classify_row(
            final_time=10.0,
            reviewer_times={"HUBP": 10.01, "HUB": 10.04, "WX": 10.20},
            offsets=offsets,
            reviewers=reviewers,
        )
        l1 = reviewer_layer.classify_row(
            final_time=10.0,
            reviewer_times={"HUBP": 10.01, "HUB": 10.10, "WX": 10.20},
            offsets=offsets,
            reviewers=reviewers,
        )
        l0 = reviewer_layer.classify_row(
            final_time=10.0,
            reviewer_times={"HUBP": 10.10, "HUB": 10.15, "WX": 10.20},
            offsets=offsets,
            reviewers=reviewers,
        )
        self.assertEqual((l2.label, l1.label, l0.label), ("L2", "L1", "L0"))

    def test_offset_is_subtracted_before_agreement(self) -> None:
        decision = reviewer_layer.classify_row(
            final_time=20.0,
            reviewer_times={"HUBP": 20.08},
            offsets={"HUBP": 0.04},
            reviewers=("HUBP",),
        )
        self.assertEqual(decision.label, "L3")
        self.assertAlmostEqual(decision.corrected_times["HUBP"], 20.04, places=6)

    def test_full_view_treats_missing_as_disagreement_for_at_least_two(self) -> None:
        decision = reviewer_layer.classify_row(
            final_time=10.0,
            reviewer_times={"HUBP": 10.01, "HUB": None, "WX": 10.02},
            offsets={"HUBP": 0.0, "HUB": 0.0, "WX": 0.0},
            reviewers=("HUBP", "HUB", "WX"),
        )
        self.assertFalse(reviewer_layer.trusted_by_rule(decision, rule="all", view="common"))
        self.assertFalse(reviewer_layer.trusted_by_rule(decision, rule="all", view="full"))
        self.assertFalse(reviewer_layer.trusted_by_rule(decision, rule="at-least-2", view="common"))
        self.assertFalse(reviewer_layer.trusted_by_rule(decision, rule="at-least-2", view="full"))

    def test_full_label_threshold_counts_missing_as_disagreement_vote(self) -> None:
        decision = reviewer_layer.classify_row(
            final_time=10.0,
            reviewer_times={"HUBP": 10.01, "HUB": None, "WX": 10.02},
            offsets={"HUBP": 0.0, "HUB": 0.0, "WX": 0.0},
            reviewers=("HUBP", "HUB", "WX"),
        )
        row = {
            "song_id": "synthetic",
            "entry": 1,
            "final_time": 10.0,
            "reference_time": 10.0,
        }
        metrics = run_reviewer_analysis.aggregate_label_thresholds(
            [(row, decision)], view="full"
        )
        self.assertEqual(metrics[">=L3"]["trusted_count"], 0)
        self.assertEqual(metrics[">=L2"]["trusted_count"], 0)
        self.assertEqual(metrics[">=L1"]["trusted_count"], 0)
        self.assertEqual(metrics["L0"]["count"], 1)

    def test_split_members_can_select_acceptance_partition(self) -> None:
        split = {
            "dev": [{"id": "dev-song"}],
            "acceptance": [{"id": "accept-song"}],
        }
        selected = run_reviewer_analysis.split_members_for_partition(split, "acceptance")
        self.assertEqual(selected, {"accept-song": {"id": "accept-song"}})

    def test_acceptance_output_is_aggregate_only(self) -> None:
        payload = {
            "metadata": {"partition": "acceptance"},
            "offsets": {"global": {"HUBP": 0.01}, "per_song": {"song-a": {"HUBP": 0.02}}},
            "analysis": {
                "per_song": {
                    "HUBP": {
                        "rules": {
                            "all": {
                                "full": {"coverage_percent": 50.0, "per_song": {"song-a": {"trusted_count": 1}}},
                                "common": {"coverage_percent": 60.0, "per_song": {"song-a": {"trusted_count": 1}}},
                            }
                        }
                    }
                }
            },
            "sidecar_rows": {"song-a::1": {"final_time": 10.0}},
        }

        sanitized = run_reviewer_analysis.output_payload_for_partition(payload, "acceptance")

        self.assertNotIn("sidecar_rows", sanitized)
        self.assertNotIn("per_song", sanitized["offsets"])
        self.assertNotIn("per_song", sanitized["analysis"]["per_song"]["HUBP"]["rules"]["all"]["full"])
        self.assertNotIn("per_song", sanitized["analysis"]["per_song"]["HUBP"]["rules"]["all"]["common"])
        self.assertEqual(sanitized["analysis"]["per_song"]["HUBP"]["rules"]["all"]["full"]["coverage_percent"], 50.0)
        self.assertEqual(sanitized["acceptance_privacy"], "aggregate_only")


if __name__ == "__main__":
    unittest.main()
