"""Historical paired-timestamp lyric compatibility regression."""
import tempfile
import unittest
from pathlib import Path
import auto_lrc


class TimestampedTranslationPairs(unittest.TestCase):
    def test_repeated_paired_source_and_translation_rows_stay_one_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "paired.lrc"
            path.write_text(
                "[00:01.00]Hello world\n[00:01.00]你好世界\n"
                "[00:04.00]Goodbye moon\n[00:04.00]再見月亮\n"
                "[00:07.00]Open the door\n[00:07.00]打開大門\n"
                "[00:10.00]Another line\n[00:10.00]另一句歌詞\n",
                encoding="utf-8",
            )
            entries = auto_lrc.load_lyrics(path).entries
        self.assertEqual(len(entries), 4)
        self.assertEqual(entries[0].lines, ["Hello world", "你好世界"])

    def test_one_same_timestamp_pair_is_not_automatically_translation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "same_time.lrc"
            path.write_text(
                "[00:01.00]Hello world\n[00:01.00]Goodbye moon\n"
                "[00:04.00]Open the door\n",
                encoding="utf-8",
            )
            entries = auto_lrc.load_lyrics(path).entries
        self.assertEqual(len(entries), 3)


if __name__ == "__main__":
    unittest.main()
