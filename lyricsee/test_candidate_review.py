"""Tests for candidate review functionality."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from candidate_review import (
    save_candidates,
    load_candidates_metadata,
    record_review_decision,
    load_review_decisions,
)
from lyricsee_core import LyricCandidate, LyricPair


class CandidateReviewTests(unittest.TestCase):
    """Test candidate saving and review."""

    def test_save_candidates_creates_directory_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            candidate = LyricCandidate(
                source="test",
                url="http://example.com/lyrics",
                title="Test Song",
                artist="Test Artist",
                lines=[LyricPair("Line 1"), LyricPair("Line 2")],
            )

            save_candidates(root, "test_stem", [candidate])

            candidates_dir = root / "candidates" / "test_stem"
            self.assertTrue(candidates_dir.exists())
            self.assertTrue((candidates_dir / "candidate_0.txt").exists())
            self.assertTrue((candidates_dir / "candidates.json").exists())

    def test_save_candidates_with_multiple(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            candidates = [
                LyricCandidate(
                    source=f"source_{i}",
                    url=f"http://example.com/lyric{i}",
                    title="Song",
                    artist="Artist",
                    lines=[LyricPair(f"Line {i}")],
                )
                for i in range(3)
            ]

            save_candidates(root, "multi_test", candidates)

            metadata = load_candidates_metadata(root, "multi_test")
            self.assertEqual(len(metadata), 3)
            self.assertEqual(metadata[0]["source"], "source_0")
            self.assertEqual(metadata[1]["source"], "source_1")
            self.assertEqual(metadata[2]["source"], "source_2")

    def test_save_candidates_with_rejection_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            candidate = LyricCandidate(
                source="test",
                url="http://example.com/lyrics",
                title="Song",
                artist="Artist",
                lines=[LyricPair("Line 1")],
            )

            save_candidates(root, "test", [candidate], "source_unavailable")

            metadata = load_candidates_metadata(root, "test")
            self.assertEqual(metadata[0]["reason"], "source_unavailable")

    def test_record_and_load_review_decision_accept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            record_review_decision(root, "song1", "accept", 0)

            decisions = load_review_decisions(root)
            self.assertIn("song1", decisions)
            self.assertEqual(decisions["song1"]["decision"], "accept")
            self.assertEqual(decisions["song1"]["candidate_idx"], 0)

    def test_record_and_load_review_decision_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            record_review_decision(root, "song2", "reject", None)

            decisions = load_review_decisions(root)
            self.assertIn("song2", decisions)
            self.assertEqual(decisions["song2"]["decision"], "reject")
            self.assertIsNone(decisions["song2"]["candidate_idx"])

    def test_review_decisions_multiple_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            record_review_decision(root, "song1", "accept", 1)
            record_review_decision(root, "song2", "reject", None)
            record_review_decision(root, "song3", "accept", 0)

            decisions = load_review_decisions(root)
            self.assertEqual(len(decisions), 3)
            self.assertEqual(decisions["song1"]["candidate_idx"], 1)
            self.assertEqual(decisions["song2"]["decision"], "reject")
            self.assertEqual(decisions["song3"]["candidate_idx"], 0)

    def test_load_candidates_metadata_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            metadata = load_candidates_metadata(root, "nonexistent")
            self.assertIsNone(metadata)

    def test_load_review_decisions_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            decisions = load_review_decisions(root)
            self.assertEqual(decisions, {})

    def test_candidate_similarity_calculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Create two similar candidates
            candidates = [
                LyricCandidate(
                    source="source1",
                    url="http://example1.com",
                    title="Song",
                    artist="Artist",
                    lines=[LyricPair("Same line"), LyricPair("Different 1")],
                ),
                LyricCandidate(
                    source="source2",
                    url="http://example2.com",
                    title="Song",
                    artist="Artist",
                    lines=[LyricPair("Same line"), LyricPair("Different 2")],
                ),
            ]

            save_candidates(root, "similarity_test", candidates)

            metadata = load_candidates_metadata(root, "similarity_test")
            # Should have similarity calculated
            similarities = metadata[0].get("similarity_to_others", {})
            # JSON keys are strings when serialized/deserialized
            self.assertIn("1", similarities)
            self.assertGreater(similarities["1"], 0)  # Some similarity due to "Same line"

    def test_save_candidates_preserves_text(self):
        """Verify that original lyrics are saved as text files."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            candidate = LyricCandidate(
                source="test_source",
                url="http://example.com",
                title="Song",
                artist="Artist",
                lines=[LyricPair("First line"), LyricPair("Second line"), LyricPair("Third line")],
            )

            save_candidates(root, "text_test", [candidate])

            lyrics_file = root / "candidates" / "text_test" / "candidate_0.txt"
            self.assertTrue(lyrics_file.exists())

            content = lyrics_file.read_text(encoding="utf-8")
            self.assertEqual(content, "First line\nSecond line\nThird line")


if __name__ == "__main__":
    unittest.main()
