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

    def test_millisecond_timestamps_round_half_up_and_carry(self):
        cases = {
            "[00:01.124]": 112,
            "[00:01.125]": 113,
            "[00:01.005]": 101,
            "[00:59.995]": 6000,
            "[01:00.999]": 6100,
        }
        for tag, expected in cases.items():
            with self.subTest(tag=tag):
                self.assertEqual(auto_lrc.lrc_time_key(tag), expected)

    def test_multi_timestamp_lines_expand_and_sort_without_merging(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "repeat.lrc"
            path.write_text(
                "[00:10.00][00:30.00]Chorus\n"
                "[00:20.00]Verse\n"
                "[00:00.00]Intro\n"
                "[00:20.00]Second at same time\n",
                encoding="utf-8",
            )
            document = auto_lrc.load_lyrics(path)
        self.assertTrue(document.saw_timestamps)
        self.assertEqual(
            [entry.source_time_cs for entry in document.entries],
            [0, 1000, 2000, 2000, 3000],
        )
        self.assertEqual(
            [entry.lines for entry in document.entries],
            [["Intro"], ["Chorus"], ["Verse"], ["Second at same time"], ["Chorus"]],
        )

    def test_plain_title_is_kept_and_only_matching_generated_card_removed(self):
        from unittest import mock
        entries = [
            auto_lrc.LyricEntry(["Title"]),
            auto_lrc.LyricEntry(["Artist - Title"], 0),
            auto_lrc.LyricEntry(["Artist - Title"], 100),
            auto_lrc.LyricEntry(["Another - Title"], 0),
            auto_lrc.LyricEntry(["Artist - Other"], 0),
        ]
        with mock.patch.object(auto_lrc, "audio_track_labels", return_value=("Artist", "Title")):
            kept = auto_lrc.remove_generated_title_cards(entries, Path("song.flac"))
        self.assertEqual(kept, [entries[0], entries[2], entries[3], entries[4]])

    def test_translation_pairing_uses_whole_file_ratio_and_direction(self):
        sparse = [
            auto_lrc.LyricEntry([line]) for line in
            ["さよなら", "再見", "Hello world", "This is another", "Last line"]
        ]
        self.assertEqual(auto_lrc.merge_adjacent_translation_lines(sparse), sparse)
        dense = [
            auto_lrc.LyricEntry([line]) for line in
            ["さよなら", "再見", "ありがとう", "謝謝", "Ending"]
        ]
        self.assertEqual(
            [entry.lines for entry in auto_lrc.merge_adjacent_translation_lines(dense)],
            [["さよなら", "再見"], ["ありがとう", "謝謝"], ["Ending"]],
        )
        reversed_order = [
            auto_lrc.LyricEntry(["再見"]), auto_lrc.LyricEntry(["さよなら"])
        ]
        self.assertEqual(
            auto_lrc.merge_adjacent_translation_lines(reversed_order),
            reversed_order,
        )

    def test_unsupported_letter_scripts_are_not_treated_as_instrumental(self):
        korean = auto_lrc.LyricEntry(["사랑해"])
        cyrillic = auto_lrc.LyricEntry(["Привет"])
        music_note = auto_lrc.LyricEntry(["♪"])
        kept, removed = auto_lrc.remove_instrumental_markers(
            [korean, cyrillic, music_note]
        )
        self.assertEqual(kept, [korean, cyrillic])
        self.assertEqual([row["text"] for row in removed], ["♪"])


if __name__ == "__main__":
    unittest.main()