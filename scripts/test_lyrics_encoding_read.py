"""Alignment reads user-provided lyric files through the shared encoding detector (text_encoding)."""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import lyrics_io
import text_encoding
from runtime_base import LrcError

SIMPLIFIED_LINES = ["这是第一句歌词", "爱你一万年", "我们都是好孩子"]
# the advice the review page gives for an LRC of uncertain encoding (index.html ENCODING_UNCERTAIN_TEXT)
REFUSAL_ADVICE = "請用文字編輯器以正確的編碼開啟（看得到正確歌詞）後另存成 UTF-8"
TRADITIONAL_LINES = ["這是第一句歌詞，請你相信我", "愛你一萬年，我的朋友", "我們都是好孩子，一起唱歌"]
JAPANESE_LINES = ["これはテストです", "ひらがなとカタカナ", "夜空の星を見上げて"]


def _texts(document):
    return [line for entry in document.entries for line in entry.lines]


class ReadUserLyricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name, raw):
        path = Path(self.tmp.name) / name
        path.write_bytes(raw)
        return path

    def _load(self, name, raw):
        path = self._write(name, raw)
        err = io.StringIO()
        with redirect_stderr(err):
            document = lyrics_io.load_lyrics(path)
        return document, err.getvalue()

    def test_gbk_is_read_and_logged(self):
        document, log = self._load("gbk.txt", "\n".join(SIMPLIFIED_LINES).encode("gb18030"))
        self.assertEqual(_texts(document), SIMPLIFIED_LINES)
        self.assertIn("read as gb18030", log)

    def test_big5_is_read_and_logged(self):
        document, log = self._load("big5.txt", "\n".join(TRADITIONAL_LINES).encode("cp950"))
        self.assertEqual(_texts(document), TRADITIONAL_LINES)
        self.assertIn("read as cp950", log)

    def test_shift_jis_is_read_and_logged(self):
        document, log = self._load("sjis.txt", "\n".join(JAPANESE_LINES).encode("cp932"))
        self.assertEqual(_texts(document), JAPANESE_LINES)
        self.assertIn("read as cp932", log)

    def test_utf16_with_bom_both_byte_orders(self):
        for name, raw in (("le.txt", b"\xff\xfe" + "\n".join(TRADITIONAL_LINES).encode("utf-16-le")),
                          ("be.txt", b"\xfe\xff" + "\n".join(TRADITIONAL_LINES).encode("utf-16-be"))):
            document, log = self._load(name, raw)
            self.assertEqual(_texts(document), TRADITIONAL_LINES, name)
            self.assertIn("read as utf-16", log)

    def test_utf8_bom_and_plain_utf8_are_silent(self):
        for name, raw in (("bom.txt", b"\xef\xbb\xbf" + "\n".join(TRADITIONAL_LINES).encode("utf-8")),
                          ("plain.txt", "\n".join(TRADITIONAL_LINES).encode("utf-8"))):
            document, log = self._load(name, raw)
            self.assertEqual(_texts(document), TRADITIONAL_LINES, name)
            self.assertNotIn("read as", log)

    def test_lrc_with_a_gbk_body_keeps_its_times(self):
        raw = "[00:01.00]这是第一句歌词\n[00:03.50]爱你一万年\n[00:06.00]我们都是好孩子\n".encode("gb18030")
        document, _log = self._load("gbk.lrc", raw)
        self.assertTrue(document.saw_timestamps)
        self.assertEqual([entry.source_time_cs for entry in document.entries], [100, 350, 600])

    def test_uncertain_file_is_refused_in_chinese(self):
        path = self._write("uncertain.txt", "爱你\n".encode("gb18030"))  # two characters: Big5 reads the same bytes
        with self.assertRaises(LrcError) as caught:
            lyrics_io.load_lyrics(path)
        message = str(caught.exception)
        self.assertIn("無法確定", message)
        self.assertIn(REFUSAL_ADVICE, message)
        for word in ("記事本", "Big5", "GB18030", "Shift-JIS"):
            self.assertNotIn(word, message)

    def test_kanji_only_shift_jis_is_refused_as_uncertain(self):
        with self.assertRaises(LrcError):
            lyrics_io.load_lyrics(self._write("kanji.txt", "東京の夜\n".encode("cp932")))

    def test_undecodable_file_is_refused(self):
        with self.assertRaises(LrcError) as caught:
            lyrics_io.load_lyrics(self._write("junk.txt", b"abc\x81\x20\xff\xff\xfe\n"))
        self.assertIn("無法判斷", str(caught.exception))
        self.assertIn(REFUSAL_ADVICE, str(caught.exception))
        self.assertNotIn("記事本", str(caught.exception))

    def test_preflight_and_anchor_hints_use_the_same_reader(self):
        gbk = self._write("gbk.txt", "\n".join(SIMPLIFIED_LINES).encode("gb18030"))
        with redirect_stderr(io.StringIO()):
            document = lyrics_io.load_lyrics(gbk)
        preflight = lyrics_io.build_lyric_preflight(gbk, document, document.entries, Path(self.tmp.name) / "a.flac")
        self.assertIsInstance(preflight, dict)
        bad = self._write("hints.txt", b"abc\x81\x20\xff\xff\xfe\n")
        with self.assertRaises(LrcError):
            lyrics_io.load_anchor_hints(bad)

    def test_shared_reader_returns_text_and_codec(self):
        path = self._write("gbk.txt", "\n".join(SIMPLIFIED_LINES).encode("gb18030"))
        self.assertEqual(text_encoding.read_lyric_text(path), ("\n".join(SIMPLIFIED_LINES), "gb18030"))
        with self.assertRaises(text_encoding.LyricEncodingError):
            text_encoding.read_lyric_text(self._write("uncertain.txt", "爱你\n".encode("gb18030")))


if __name__ == "__main__":
    unittest.main()
