"""Review page: simplified-Chinese detection and the one-click s2tw conversion (POST /api/song/N/convert)."""
from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import review_server
import test_lrc14_review_server as _fixture_module
import zh_traditional

CHINESE = "[00:01.54]这是第一句\r\n[00:03.21]爱你一万年\r\n[00:05.00]third\r\n"
CHINESE_MS = "[00:01.543]这是第一句\r\n[00:03.210]爱你一万年\r\n[00:05.000]third\r\n"
JAPANESE = "[00:01.00]君の声が聞こえる\r\n[00:03.00]爱\r\n[00:05.00]爱のうた\r\n"
TRADITIONAL = "[00:01.00]這是第一句\r\n[00:03.00]愛你一萬年\r\n[00:05.00]third\r\n"
TAG = re.compile(rb"^\[\d\d:\d\d\.\d\d\]")


class ConvertTests(unittest.TestCase):
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

    def _load(self, raw: bytes):
        self.output.write_bytes(raw)
        self.store.refresh_if_changed()
        return self._detail()

    def _convert(self, detail, entries, sha=None):
        return self._json_request("POST", "/api/song/0/convert", {"base_sha256": sha or detail["base_sha256"], "entries": entries})

    def test_chinese_song_candidates_start_checked(self) -> None:
        detail = self._load(CHINESE.encode("gb18030"))
        info = detail["simplified"]
        self.assertEqual((info["count"], info["japanese"]), (2, False))
        self.assertEqual([(line["entry"], line["before"], line["after"], line["default_checked"]) for line in info["lines"]],
                         [(1, "这是第一句", "這是第一句", True), (2, "爱你一万年", "愛你一萬年", True)])

    def test_s2tw_not_s2twp(self) -> None:
        self.assertEqual(zh_traditional.convert_line("奔驰 循环"), "奔馳 循環")

    def test_japanese_song_candidates_start_unchecked_and_kana_lines_are_skipped(self) -> None:
        detail = self._load(JAPANESE.encode("utf-8"))
        info = detail["simplified"]
        self.assertEqual((info["count"], info["japanese"]), (1, True))
        self.assertEqual([(line["entry"], line["before"], line["default_checked"]) for line in info["lines"]], [(2, "爱", False)])

    def test_traditional_song_has_no_candidates(self) -> None:
        info = self._load(TRADITIONAL.encode("utf-8"))["simplified"]
        self.assertEqual((info["count"], info["lines"]), (0, []))

    def test_opencc_unavailable_hides_the_field_and_the_server_keeps_working(self) -> None:
        with mock.patch.object(zh_traditional, "_converter", None), mock.patch.object(zh_traditional, "_loaded", True):
            detail = self._load(CHINESE.encode("utf-8"))
            self.assertIsNone(detail["simplified"])
            status, _headers, data = self._convert(detail, [1])
            self.assertEqual(status, 503, data)
            self.assertEqual(self.output.read_bytes(), CHINESE.encode("utf-8"))

    def test_stale_base_sha256_is_refused_and_nothing_changes(self) -> None:
        detail = self._load(CHINESE.encode("utf-8"))
        status, _headers, data = self._convert(detail, [1, 2], sha="0" * 64)
        self.assertEqual(status, 409, data)
        self.assertEqual(self.output.read_bytes(), CHINESE.encode("utf-8"))
        self.assertEqual(list(self.root.glob("*.bak")), [])

    def test_bad_entries_are_rejected(self) -> None:
        detail = self._load(CHINESE.encode("utf-8"))
        for entries in ([0], [99], [1, 1], ["1"], [True]):
            self.assertEqual(self._convert(detail, entries)[0], 400, entries)

    def test_only_listed_entries_change_and_tags_and_other_lines_keep_their_bytes(self) -> None:
        original = CHINESE.encode("utf-8")
        detail = self._load(original)
        status, _headers, data = self._convert(detail, [1])
        self.assertEqual(status, 200, data)
        before, after = original.split(b"\n"), self.output.read_bytes().split(b"\n")
        self.assertEqual(len(before), len(after))
        for old, new in zip(before[:-1], after[:-1]):
            self.assertEqual(TAG.match(old).group(0), TAG.match(new).group(0))
        self.assertEqual(after[0], "[00:01.54]這是第一句\r".encode("utf-8"))
        self.assertEqual(after[1:], before[1:])  # entry 2 was not listed, the last line is not Chinese-simplified
        self.assertEqual(self._detail()["simplified"]["count"], 1)

    def test_gbk_file_is_converted_to_utf8_with_both_backups(self) -> None:
        original = CHINESE_MS.encode("gb18030")
        detail = self._load(original)
        status, _headers, data = self._convert(detail, [1, 2])
        self.assertEqual(status, 200, data)
        text = self.output.read_bytes().decode("utf-8")  # strict
        self.assertIn("[00:01.54]這是第一句", text)  # the legacy millisecond tag is canonicalised as a save does
        self.assertIn("愛你一萬年", text)
        self.assertEqual([p.read_bytes() for p in self.root.glob("song.lrc.orig-*.bak")], [original])
        pre = list(self.root.glob("song.lrc.pre-s2tw-*.bak"))
        self.assertEqual([p.read_bytes() for p in pre], [review_server._canonical_lrc(original)])
        self.assertEqual(self._detail()["simplified"]["count"], 0)

    def test_utf8_file_gets_a_pre_conversion_backup(self) -> None:
        original = CHINESE.encode("utf-8")
        detail = self._load(original)
        self.assertEqual(self._convert(detail, [1, 2])[0], 200)
        self.assertEqual([p.read_bytes() for p in self.root.glob("*.bak")], [original])

    def test_conversion_keeps_a_reviewed_song_editable(self) -> None:
        detail = self._load(CHINESE.encode("utf-8"))
        status, _headers, data = self._json_request(
            "POST", "/api/song/0/save",
            {"base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 160, "source": "ear", "nudge_cs": 6}]})
        self.assertEqual(status, 200, data)
        self.assertTrue(self.root.joinpath("song.auto.lrc").is_file())
        detail = self._detail()
        self.assertEqual(self._convert(detail, [1, 2])[0], 200)
        after = self._detail()
        self.assertFalse(after["read_only"], after["read_only_reason"])
        self.assertFalse(after["manual_edited"], after["manual_edit_reason"])
        self.assertIn("[00:01.60]這是第一句", self.output.read_text(encoding="utf-8"))
        status, _headers, data = self._json_request(
            "POST", "/api/song/0/save",
            {"base_sha256": after["base_sha256"], "edits": [{"entry": 3, "cs": 700, "source": "ear", "nudge_cs": 0}]})
        self.assertEqual(status, 200, data)


if __name__ == "__main__":
    unittest.main()
