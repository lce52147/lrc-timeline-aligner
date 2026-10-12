from __future__ import annotations

import unittest

import auto_lrc


class SparseBracketTests(unittest.TestCase):
    @staticmethod
    def row(timestamp: float, segment: int, score: float = 0.9) -> dict[str, object]:
        return {
            "timestamp": timestamp,
            "segment": segment,
            "score": score,
        }

    def test_one_sided_sparse_context_is_not_a_bracket(self) -> None:
        entries = [
            auto_lrc.LyricEntry(["A"]),
            auto_lrc.LyricEntry(["B"]),
            auto_lrc.LyricEntry(["X"]),
            auto_lrc.LyricEntry(["C"]),
            auto_lrc.LyricEntry(["D"]),
            auto_lrc.LyricEntry(["X"]),
            auto_lrc.LyricEntry(["E"]),
        ]
        assignments: list[object] = [
            self.row(1.0, 1),
            self.row(2.0, 2),
            self.row(3.0, 3),
            self.row(4.0, 4, 0.1),
            self.row(5.0, 5, 0.1),
            self.row(6.0, 6, 0.1),
            self.row(7.0, 7, 0.1),
        ]
        self.assertIsNone(
            auto_lrc._sparse_unique_sequence_anchor_set(
                entries, assignments, 2, max_items=9
            )
        )
        self.assertIsNone(
            auto_lrc._entry_occurrence_binding(
                entries, assignments, 2, require_context=True
            )
        )

    def test_two_sided_sparse_context_remains_available(self) -> None:
        entries = [
            auto_lrc.LyricEntry(["A"]),
            auto_lrc.LyricEntry(["B"]),
            auto_lrc.LyricEntry(["C"]),
            auto_lrc.LyricEntry(["X"]),
            auto_lrc.LyricEntry(["D"]),
            auto_lrc.LyricEntry(["E"]),
            auto_lrc.LyricEntry(["F"]),
            auto_lrc.LyricEntry(["X"]),
            auto_lrc.LyricEntry(["G"]),
        ]
        assignments: list[object] = [
            self.row(1.0, 1, 0.1),
            self.row(2.0, 2),
            self.row(3.0, 3, 0.1),
            self.row(4.0, 4),
            self.row(5.0, 5, 0.1),
            self.row(6.0, 6),
            self.row(7.0, 7, 0.1),
            self.row(8.0, 8, 0.1),
            self.row(9.0, 9, 0.1),
        ]
        sparse = auto_lrc._sparse_unique_sequence_anchor_set(
            entries, assignments, 3, max_items=9
        )
        self.assertIsNotNone(sparse)
        assert sparse is not None
        _, _, indexes = sparse
        self.assertTrue(any(item < 3 for item in indexes))
        self.assertTrue(any(item > 3 for item in indexes))

    def test_strong_sparse_context_requires_both_sides(self) -> None:
        self.assertFalse(
            auto_lrc._occurrence_binding_has_strong_context(
                {
                    "mode": "sparse-sequence-bracket",
                    "entry_index": 2,
                    "anchor_entry_indexes": [0, 1, 2],
                }
            )
        )
        self.assertTrue(
            auto_lrc._occurrence_binding_has_strong_context(
                {
                    "mode": "sparse-sequence-bracket",
                    "entry_index": 2,
                    "anchor_entry_indexes": [1, 2, 3],
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
