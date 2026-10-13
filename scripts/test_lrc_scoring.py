"""Focused scorer regression gates (no GPU, no BK modification)."""
from __future__ import annotations
import json
import unittest
from lrc_scoring import ScoringError, align_gold, count_metrics, compare_phases, load_manifest, reorder_assignments_by_timestamp, parsed_rows


def row(t, bk, status, **kw):
    return {"time": t, "bk": bk, "status": status, "trusted": status == "trusted",
            "source": "ctc-current", "reasons": [],
            "song": "unit", "case": 1, "line": kw.get("line", 1),
            "text": kw.get("text", "lyric")}


class LrcScoringTests(unittest.TestCase):
    def test_manifest_can_pin_uncommitted_candidate_source_hash(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as folder:
            root = Path(folder)
            cases = [
                {"index": index, "status": "PRODUCED", "exit": 0}
                for index in range(1, 40)
            ]
            (root / "MANIFEST.json").write_text(json.dumps({
                "phase": "CURRENT39_39_PRODUCED",
                "count": 39,
                "total": 39,
                "validation_of_commit": "base-head",
                "auto_lrc_sha256": "working-source",
                "cases": cases,
            }), encoding="utf-8")
            loaded = load_manifest(
                root,
                "CURRENT39_39_PRODUCED",
                "base-head",
                expected_source_sha256="working-source",
            )
            self.assertEqual(len(loaded), 39)
            with self.assertRaisesRegex(ScoringError, "MANIFEST_WRONG_SOURCE"):
                load_manifest(
                    root,
                    "CURRENT39_39_PRODUCED",
                    "base-head",
                    expected_source_sha256="different-source",
                )
            with self.assertRaisesRegex(ScoringError, "MANIFEST_SOURCE_PIN_REQUIRED"):
                load_manifest(
                    root,
                    "CURRENT39_39_PRODUCED",
                    "base-head",
                )

    def test_review_not_conflated_with_unverified(self):
        rows = [row(1, 1, "trusted"), row(2, 3.2, "review-required"),
                row(3, 4.2, "unverified")]
        summary = count_metrics(rows)
        self.assertEqual(summary["all"]["gt1"], 2)
        self.assertEqual(summary["REVIEW"]["rows"], 1)
        self.assertEqual(summary["UNVERIFIED"]["rows"], 1)
        self.assertEqual(summary["TRUSTED"]["rows"], 1)
        self.assertEqual(summary["NO_REVIEW"]["gt1"], 1)
        self.assertEqual(summary["TRUSTED"]["gt1"], 0)
        self.assertAlmostEqual(summary["review_request_ratio"], 1 / 3)

    def test_threshold_boundaries(self):
        rows = [row(1.0, 1.2, "trusted"), row(1.0, 1.5, "review-required"),
                row(1.0, 2.0, "unverified")]
        a = count_metrics(rows)["all"]
        self.assertEqual(a["PCO200"], 1)
        self.assertEqual(a["gt05"], 1)
        self.assertEqual(a["gt1"], 0)

    def test_parser_order_remap_duplicate_text_occurrence(self):
        source = [{"text": x} for x in ("a", "b", "a")]
        gold = [("a", 2), ("a", 4), ("b", 3)]
        aligned, deviated = align_gold(source, gold, "sample")
        self.assertTrue(deviated)
        self.assertEqual(aligned, [("a", 2), ("b", 3), ("a", 4)])

    def test_mismatched_text_rejected(self):
        with self.assertRaisesRegex(ScoringError, "BK_TEXT_MISMATCH"):
            align_gold([{"text": "line"}], [("other", 1)], "sample")

    def test_count_mismatch_rejected(self):
        with self.assertRaisesRegex(ScoringError, "BK_COUNT_MISMATCH"):
            align_gold([{"text": "line"}], [], "sample")

    def test_sorted_lrc_preserves_original_assignment_trust(self):
        from types import SimpleNamespace
        lyrics = [SimpleNamespace(source_time_cs=x) for x in (100, 200, 300)]
        assignments = [{"timestamp": 3.0, "verification_status": "review-required"},
                       {"timestamp": 1.0, "verification_status": "trusted"},
                       {"timestamp": 2.0, "verification_status": "unverified"}]
        sorted_rows = reorder_assignments_by_timestamp(lyrics, assignments, "fixture")
        self.assertEqual([x["verification_status"] for x in sorted_rows],
                         ["trusted", "unverified", "review-required"])

    def test_inconsistent_timestamp_multiset_fails_closed(self):
        from types import SimpleNamespace
        lyrics = [SimpleNamespace(source_time_cs=x) for x in (100, 200)]
        with self.assertRaisesRegex(ScoringError, "REPORT_TIME_MULTISET_MISMATCH"):
            reorder_assignments_by_timestamp(lyrics, [{"timestamp": 1}, {"timestamp": 3}], "fixture")

    def test_title_only_and_legacy_provenance_are_sung_rows(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as folder:
            lyric = Path(folder) / "Song.lrc"
            lyric.write_text("[re:LRC tools]\\n[00:00.00]Song\\n[00:01.00]sing\\n".replace("\\n", "\n"), encoding="utf-8")
            rows = parsed_rows(lyric, Path(folder) / "Song.flac")
            self.assertEqual(len(rows), 2)
            self.assertEqual([int(x[1]*100) for x in rows], [0, 100])

    def test_unicode_instrumental_marker_not_scored(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as folder:
            lyric = Path(folder) / "Song.lrc"
            lyric.write_text("[00:01.00]（間奏）\\n[00:02.00]歌詞\\n".replace("\\n", "\n"), encoding="utf-8")
            rows = parsed_rows(lyric, Path(folder) / "Song.flac")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], 2.0)

    def test_empty_corpus_fails_closed(self):
        with self.assertRaisesRegex(ScoringError, "EMPTY_CORPUS"):
            count_metrics([])

    def test_exact_diff_and_status_independent(self):
        a = [row(40.62, 40.60, "unverified", line=8)]
        b = [row(38.46, 40.60, "review-required", line=8)]
        res = compare_phases({"rows": a}, {"rows": b})
        self.assertEqual(len(res["timestamp_changes"]), 1)
        self.assertEqual(len(res["trust_status_changes"]), 1)
        self.assertEqual(len(res["new_gt1"]), 1)
        self.assertEqual(res["timestamp_changes"][0]["delta_s"], -2.16)

    def test_lexical_identity_required(self):
        a = [row(1, 1, "trusted", text="a")]
        b = [row(1, 1, "trusted", text="b")]
        with self.assertRaisesRegex(ScoringError, "PHASE_LEXICAL_BK_DIFFERENCE"):
            compare_phases({"rows": a}, {"rows": b})


if __name__ == "__main__":
    unittest.main()
