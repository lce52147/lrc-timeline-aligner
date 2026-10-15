from __future__ import annotations

import unittest
from types import SimpleNamespace

import auto_lrc
from multimodel_consensus import (
    build_windows,
    choose_consensus_overrides,
    model_family,
    model_identity,
)


class MultiModelConsensusTests(unittest.TestCase):
    def test_family_deduplication_and_tau_boundary(self) -> None:
        self.assertEqual(model_family("mms1b:eng:roman"), "mms1b")
        result = choose_consensus_overrides(
            [1.0, 8.0, 20.0],
            ["selected_valid", "provisional_unresolved", "selected_valid"],
            {1: {
                "mms": 4.00,
                "mms1b:eng:roman": 4.05,
                "mms1b:jpn:native": 4.06,
                "en-ctc:large": 4.10,
                "ja-ctc:large": 4.25,
            }},
        )
        self.assertEqual(result[1].support_families, ("en-ctc", "mms", "mms1b"))
        self.assertAlmostEqual(result[1].seconds, 4.055, places=6)

    def test_selected_valid_is_never_rewritten(self) -> None:
        result = choose_consensus_overrides(
            [1.0, 8.0],
            ["selected_valid", "selected_valid"],
            {0: {"mms": 4.0, "en-ctc": 4.05, "ja-ctc": 4.08}},
        )
        self.assertEqual(result, {})

    def test_g8_requires_mms_support(self) -> None:
        result = choose_consensus_overrides(
            [1.0, 8.0],
            ["selected_valid", "provisional_unresolved"],
            {1: {"en-ctc": 4.0, "ja-ctc": 4.05, "mms1b": 4.08}},
        )
        self.assertEqual(result, {})

    def test_g2_blocks_raw_order_violation(self) -> None:
        result = choose_consensus_overrides(
            [1.0, 8.0, 20.0],
            ["selected_valid", "provisional_unresolved", "selected_valid"],
            {1: {"mms": 21.0, "en-ctc": 21.05, "ja-ctc": 21.08}},
            raw_anchors=((2, 20.0),),
        )
        self.assertEqual(result, {})

    def test_g5_blocks_collision_with_unchanged_row(self) -> None:
        result = choose_consensus_overrides(
            [1.0, 8.0, 4.1],
            ["selected_valid", "provisional_unresolved", "selected_valid"],
            {1: {"mms": 4.0, "en-ctc": 4.05, "ja-ctc": 4.08}},
        )
        self.assertEqual(result, {})

    def test_g5_revokes_both_colliding_rewrites(self) -> None:
        result = choose_consensus_overrides(
            [1.0, 8.0, 20.0],
            ["selected_valid", "provisional_unresolved", "provisional_unresolved"],
            {
                1: {"mms": 15.0, "en-ctc": 15.05, "ja-ctc": 15.08},
                2: {"mms": 15.1, "en-ctc": 15.14, "ja-ctc": 15.18},
            },
        )
        self.assertEqual(result, {})

    def test_g9_revokes_both_rewrites_when_order_is_reversed(self) -> None:
        result = choose_consensus_overrides(
            [1.0, 8.0, 20.0],
            ["selected_valid", "provisional_unresolved", "provisional_unresolved"],
            {
                1: {"mms": 20.0, "en-ctc": 20.05, "ja-ctc": 20.08},
                2: {"mms": 15.0, "en-ctc": 15.05, "ja-ctc": 15.08},
            },
        )
        self.assertEqual(result, {})

    def test_windows_follow_selected_valid_bounds(self) -> None:
        windows = build_windows(
            [1.0, 8.0, 20.0],
            ["selected_valid", "provisional_unresolved", "selected_valid"],
            ["a", "b", "c"],
        )
        self.assertEqual(windows[0]["entry_index"], 1)
        self.assertEqual(windows[0]["text_start_index"], 0)
        self.assertEqual(windows[0]["target_offset"], 1)
        self.assertEqual(windows[0]["start"], 0.7)
        self.assertEqual(windows[0]["end"], 19.95)

    def test_off_returns_the_existing_state_without_model_work(self) -> None:
        state = object()
        report: dict[str, object] = {}
        result = auto_lrc._apply_multimodel_consensus_runtime(
            state, (), report, [], SimpleNamespace(multimodel_consensus="off")
        )
        self.assertIs(result, state)
        self.assertEqual(report["multimodel_consensus"]["reason"], "disabled")

    def test_unknown_model_identity_is_explicitly_unverified(self) -> None:
        identity = model_identity("unsupported-model")
        self.assertEqual(identity, {
            "model_revision": "UNVERIFIED",
            "model_blob_sha256": "UNVERIFIED",
        })


if __name__ == "__main__":
    unittest.main()
