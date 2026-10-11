"""Transformation audit must reflect real parsed input changes."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import auto_lrc


class PreflightTransformations(unittest.TestCase):
    def test_timed_translation_merges_and_empty_normalization_are_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "paired.lrc"
            path.write_text(
                "[00:01.00]Hello world\n[00:01.00]你好世界\n"
                "[00:04.00]Goodbye moon\n[00:04.00]再見月亮\n"
                "[00:07.00]Open the door\n[00:07.00]打開大門\n"
                "[00:10.00]Another line\n[00:10.00]另一句歌詞\n"
                "[00:13.00]♪\n",
                encoding="utf-8",
            )
            document = auto_lrc.load_lyrics(path)
            with mock.patch.object(auto_lrc, "audio_track_labels", return_value=("Artist", "Song")):
                result = auto_lrc.build_lyric_preflight(
                    path, document, document.entries, Path("song.flac")
                )
        kinds = [item["kind"] for item in result["transformations"]]
        self.assertEqual(kinds.count("translation-pair-merged"), 4)
        self.assertIn("normalized-empty-removed", kinds)


if __name__ == "__main__":
    unittest.main()
