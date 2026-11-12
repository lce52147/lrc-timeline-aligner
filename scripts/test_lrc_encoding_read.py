"""Review page reads LRC files in legacy encodings and millisecond tags; undecodable files say so instead of showing no rows."""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import review_server
import test_lrc14_review_server as _fixture_module

GBK_TEXT = "[ti:歌词 (《穿越边缘》)]\r\n[00:01.543]这是第一句\r\n[00:03.210]爱你一万年\r\n[00:05.000]third\r\n"


class DecodeTests(unittest.TestCase):
    def test_utf8_and_bom_pass_through(self) -> None:
        raw = "[00:01.00]你好\n".encode("utf-8")
        self.assertEqual(review_server._canonical_lrc(raw), raw)
        bommed = b"\xef\xbb\xbf" + raw
        self.assertEqual(review_server._canonical_lrc(bommed), bommed)

    def test_utf16_with_bom(self) -> None:
        raw = "[00:01.00]你好\r\n".encode("utf-16")
        self.assertEqual(review_server._canonical_lrc(raw), "[00:01.00]你好\r\n".encode("utf-8"))

    def test_gb18030(self) -> None:
        text, encoding = review_server._decode_lrc(GBK_TEXT.encode("gb18030"))
        self.assertEqual((text, encoding), (GBK_TEXT, "gb18030"))

    def test_big5(self) -> None:
        text = "[00:01.00]我愛你，這是歌詞\n"
        self.assertEqual(review_server._decode_lrc(text.encode("cp950")), (text, "cp950"))

    def test_undecodable_raises(self) -> None:
        with self.assertRaises(review_server.LrcDecodeError):
            review_server._canonical_lrc(b"[00:01.00]\x81\x20\xff\xff\xfe\n")
        with self.assertRaises(review_server.LrcDecodeError):
            review_server._canonical_lrc(b"\xff\xfe\x00")  # UTF-16 BOM, odd byte count

    def test_millisecond_tags_become_centiseconds(self) -> None:
        raw = b"[00:01.543]a\n*[10:00.004]b*\n[00:59.996]c\n[00:01.50]kept\n"
        self.assertEqual(review_server._canonical_lrc(raw), b"[00:01.54]a\n*[10:00.00]b*\n[01:00.00]c\n[00:01.50]kept\n")


class ReviewReadsLegacyLrcTests(unittest.TestCase):
    _start = _fixture_module.ReviewServerTests._start
    _request = _fixture_module.ReviewServerTests._request
    _json_request = _fixture_module.ReviewServerTests._json_request
    _detail = _fixture_module.ReviewServerTests._detail

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"LRC_EXPORT_ARCHIVE_DIR": "off"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._start(False)
        for sidecar in self.root.glob("song.auto.lrc"):
            sidecar.unlink()

    def tearDown(self) -> None:
        _fixture_module.ReviewServerTests.tearDown(self)

    def test_gbk_millisecond_lrc_shows_rows_and_saves_utf8_with_backup(self) -> None:
        original = GBK_TEXT.encode("gb18030")
        self.output.write_bytes(original)
        self.store.refresh_if_changed()
        detail = self._detail()
        self.assertFalse(detail["read_only"], detail["read_only_reason"])
        self.assertEqual([row["lyric"] for row in detail["rows"]], ["这是第一句", "爱你一万年", "third"])
        body = {"base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 160, "source": "ear", "nudge_cs": 6}]}
        status, _headers, data = self._json_request("POST", "/api/song/0/save", body)
        self.assertEqual(status, 200, data)
        saved = self.output.read_bytes()
        text = saved.decode("utf-8")  # strict: the file on disk is UTF-8 now
        self.assertIn("[00:01.60]这是第一句", text)
        self.assertIn("[00:03.21]爱你一万年", text)
        backups = list(self.root.glob("song.lrc.orig-*.bak"))
        self.assertEqual([path.read_bytes() for path in backups], [original])
        again = self._detail()
        self.assertEqual([row["lyric"] for row in again["rows"]], ["这是第一句", "爱你一万年", "third"])
        status, _headers, _data = self._json_request(
            "POST", "/api/song/0/save",
            {"base_sha256": again["base_sha256"], "edits": [{"entry": 3, "cs": 700, "source": "ear", "nudge_cs": 0}]})
        self.assertEqual(status, 200)
        self.assertEqual(len(list(self.root.glob("song.lrc.orig-*.bak"))), 1)  # a UTF-8 file is not backed up again

    def test_undecodable_lrc_is_read_only_with_a_reason(self) -> None:
        self.output.write_bytes(b"[00:01.00]\x81\x20\xff\xff\xfe\n[00:02.00]x\n")
        self.store.refresh_if_changed()
        detail = self._detail()
        self.assertTrue(detail["read_only"])
        self.assertEqual(detail["read_only_reason"], "lrc-undecodable")
        self.assertEqual(detail["rows"], [])
        self.assertEqual(self._json_request("POST", "/api/song/0/save", {"base_sha256": detail["base_sha256"], "edits": []})[0], 403)


if __name__ == "__main__":
    unittest.main()
