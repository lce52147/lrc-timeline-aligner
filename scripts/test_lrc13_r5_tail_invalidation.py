from __future__ import annotations

import inspect
import unittest

import auto_lrc


class RecoveryTailInvalidationTests(unittest.TestCase):
    def test_recovery_changed_entries_are_collected_from_both_recovery_paths(self) -> None:
        seed = {
            auto_lrc._HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY: {
                "applied_trigger_entries": [1, 3],
            },
            auto_lrc._PEER_EXISTING_CANDIDATE_RECOVERY_KEY: {
                "applied_trigger_entries": [2],
            },
        }
        self.assertEqual(
            auto_lrc._postcentral_recovery_changed_entries(seed),
            frozenset({1, 2, 3}),
        )
        malformed = {
            auto_lrc._HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY: {
                "applied_trigger_entries": ["1"],
            }
        }
        self.assertIsNone(
            auto_lrc._postcentral_recovery_changed_entries(malformed)
        )

    def test_postcentral_rebind_drops_pre_recovery_external_tail_for_changed_predecessor(self) -> None:
        source = inspect.getsource(
            auto_lrc._postcentral_rebind_existing_peer_conflicts
        )
        self.assertIn("recovery_changed_entries", source)
        self.assertIn("tail_rows.get(index - 1)", source)
        self.assertIn("index not in recovery_changed_entries", source)
        self.assertIn("else None", source)


if __name__ == "__main__":
    unittest.main()