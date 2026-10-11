from pathlib import Path
import tempfile
import unittest
import auto_lrc

class ReviewSidecarTests(unittest.TestCase):
    def test_sidecar_metadata_and_review_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "song.review.tsv"
            report = {"timing_entries": 2, "review_required_count": 1,
                      "lyric_preflight": {"status": "INPUT_SUSPECT"},
                      "assignments": [
                          {"entry": 1, "timestamp": 2.0, "romaji": "sayonara",
                           "ctc_first_token_start": 2.0, "verification_status": "trusted"},
                          {"entry": 2, "timestamp": 3.0, "romaji": "arigatou",
                           "ctc_first_token_start": 3.0, "review_required": True,
                           "verification_status": "review-required",
                           "final_trust_reason_codes": ["previous-tail-overlap"]}]}
            entries = [auto_lrc.LyricEntry(["さよなら"]), auto_lrc.LyricEntry(["ありがとう"])]
            auto_lrc.write_review_tsv(path, report, entries)
            data = path.read_text(encoding="utf-8-sig")
            self.assertIn("fast_singing=TRUE", data.splitlines()[0])
            self.assertIn("review_percent=50.0", data.splitlines()[0])
            self.assertIn("INPUT_SUSPECT", data.splitlines()[0])
            self.assertIn("ありがとう\tREVIEW\tprevious-tail-overlap", data)
            self.assertNotIn("さよなら\tTRUSTED", data)


    def test_untrusted_rows_and_unknown_speed_are_not_silently_omitted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "song.review.tsv"
            report = {"timing_entries": 2, "review_required_count": 0,
                      "assignments": [
                          {"entry": 1, "timestamp": 2.0, "verification_status": "trusted"},
                          {"entry": 2, "timestamp": 4.0, "verification_status": "unverified",
                           "final_trust_reason_codes": ["insufficient-minimum-proof"]}]}
            entries = [auto_lrc.LyricEntry(["One"]), auto_lrc.LyricEntry(["Two"])]
            auto_lrc.write_review_tsv(path, report, entries)
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            self.assertIn("fast_singing=UNKNOWN", lines[0])
            self.assertIn("preflight=NOT_RUN", lines[0])
            self.assertIn("Two\tUNTRUSTED\tinsufficient-minimum-proof", lines[-1])
            self.assertEqual(len(lines), 3)



    def test_heuristic_report_without_assignments_still_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "song.review.tsv"
            auto_lrc.write_review_tsv(
                path, {"timing_entries": 1}, [auto_lrc.LyricEntry(["hello"])]
            )
            self.assertIn("hello\tUNTRUSTED", path.read_text(encoding="utf-8-sig"))

class LyricParserRegressionTests(unittest.TestCase):
    def test_wrapped_instrumental_markers_are_removed_without_losing_lyrics(self):
        markers = ["（間奏）", "【間奏】", "［間奏］", "[Instrumental]", "(intro)", "{outro}"]
        entries = [auto_lrc.LyricEntry([text]) for text in markers]
        sung = auto_lrc.LyricEntry(["這段歌詞提到間奏，但不是標記"])
        entries.append(sung)
        kept, removed = auto_lrc.remove_instrumental_markers(entries)
        self.assertEqual(kept, [sung])
        self.assertEqual([row["text"] for row in removed], markers)


if __name__ == "__main__":
    unittest.main()