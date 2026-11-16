"""The title card the generator writes for a song without an ARTIST tag is not a lyric when that LRC is used as lyric input."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc
import lrc_scoring
import lyrics_io
from lyrics_io import LyricEntry, remove_generated_title_cards

AUDIO = Path("Song File.flac")


def labels(artist, title):
    return mock.patch.object(lyrics_io, "audio_track_labels", return_value=(artist, title))


class TitleOnlyCardTests(unittest.TestCase):
    def texts(self, entries, **kwargs):
        return [entry.lines[0] for entry in remove_generated_title_cards(entries, AUDIO, **kwargs)]

    def entries(self, first="Tagged Title", at=0):
        return [LyricEntry([first], at), LyricEntry(["synthetic lyric line"], 1890)]

    def test_the_generated_card_is_removed_from_lyric_input_when_the_title_tag_is_not_the_file_name(self):
        with labels("", "Tagged Title"):
            self.assertEqual(self.texts(self.entries(), title_only=True), ["synthetic lyric line"])

    def test_the_card_is_matched_like_the_artist_title_card_ignoring_case_and_spacing(self):
        with labels("", "Tagged  Title"):
            self.assertEqual(self.texts(self.entries("tagged title"), title_only=True), ["synthetic lyric line"])

    def test_scoring_and_every_other_caller_keep_the_old_result(self):
        with labels("", "Tagged Title"):
            self.assertEqual(self.texts(self.entries()), ["Tagged Title", "synthetic lyric line"])

    def test_a_zero_time_line_with_other_text_is_a_lyric(self):
        with labels("", "Tagged Title"):
            self.assertEqual(self.texts(self.entries("a real first line"), title_only=True), ["a real first line", "synthetic lyric line"])

    def test_the_title_sung_later_is_a_lyric(self):
        with labels("", "Tagged Title"):
            self.assertEqual(self.texts(self.entries(at=500), title_only=True), ["Tagged Title", "synthetic lyric line"])

    def test_a_two_line_entry_is_a_lyric(self):
        entries = [LyricEntry(["Tagged Title", "translation"], 0), LyricEntry(["synthetic lyric line"], 1890)]
        with labels("", "Tagged Title"):
            self.assertEqual(len(remove_generated_title_cards(entries, AUDIO, title_only=True)), 2)

    def test_with_an_artist_the_title_alone_is_a_lyric_and_artist_title_is_still_removed(self):
        with labels("Some Artist", "Tagged Title"):
            self.assertEqual(self.texts(self.entries(), title_only=True), ["Tagged Title", "synthetic lyric line"])
            self.assertEqual(self.texts(self.entries("Some Artist - Tagged Title"), title_only=True), ["synthetic lyric line"])
            self.assertEqual(self.texts(self.entries("Some Artist - Tagged Title")), ["synthetic lyric line"])

    def test_a_card_read_from_a_real_lrc_file_is_dropped_by_the_lyric_input_path_only(self):
        with tempfile.TemporaryDirectory() as tmp, labels("", "Tagged Title"):
            lrc = Path(tmp) / "Song File.lrc"
            lrc.write_text("[ti:Tagged Title]\n[re:x]\n[length:03:00.00]\n[by:LRC tools]\n[00:00.00]Tagged Title\n[00:05.00]first\n[00:10.00]second\n",
                           encoding="utf-8-sig")
            document = lyrics_io.load_lyrics(lrc)
            self.assertEqual(len(document.entries), 3)
            kept = remove_generated_title_cards(document.entries, AUDIO, title_only=True)
            self.assertEqual([entry.lines[0] for entry in kept], ["first", "second"])
            # scoring reads the same file through the default call: unchanged
            self.assertEqual([text for text, _time in lrc_scoring.parsed_rows(lrc, AUDIO)][0], auto_lrc.normalize_match_text("Tagged Title"))

    def test_the_lyric_input_callers_ask_for_it_and_scoring_does_not(self):
        import inspect
        import r2_pipeline
        self.assertIn("remove_generated_title_cards(lyrics.entries, audio_path, title_only=True)", inspect.getsource(auto_lrc))
        self.assertIn("remove_generated_title_cards(document.entries, audio_path, title_only=True)", inspect.getsource(r2_pipeline))
        self.assertIn("remove_generated_title_cards(checked_entries, audio_path, title_only=True)", inspect.getsource(lyrics_io))
        self.assertNotIn("title_only", inspect.getsource(lrc_scoring))


if __name__ == "__main__":
    unittest.main()
