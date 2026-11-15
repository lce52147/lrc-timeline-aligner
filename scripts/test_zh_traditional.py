"""Review page: simplified-Chinese detection and the one-click s2tw conversion (POST /api/song/N/convert)."""
from __future__ import annotations

import json
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


    def test_convert_response_maps_old_line_keys_to_new_ones(self) -> None:
        detail = self._load(CHINESE.encode("utf-8"))
        old = {row["entry"]: row["line_key"] for row in detail["rows"]}
        status, _headers, data = self._convert(detail, [1])
        self.assertEqual(status, 200, data)
        new = {row["entry"]: row["line_key"] for row in self._detail()["rows"]}
        self.assertEqual(json.loads(data)["line_key_map"], {old[1]: new[1]})   # entry 2 was not converted and keeps its key
        self.assertNotEqual(old[1], new[1])
        self.assertEqual(old[2], new[2])

    def test_nothing_converted_reports_zero_and_writes_nothing(self) -> None:
        detail = self._load(JAPANESE.encode("utf-8"))
        generation = self.store.generation
        status, _headers, data = self._convert(detail, [1, 3])   # both kana lines
        self.assertEqual(status, 200, data)
        self.assertEqual(json.loads(data)["converted"], 0)
        self.assertEqual(json.loads(data)["generation"], generation)
        self.assertEqual(self.output.read_bytes(), JAPANESE.encode("utf-8"))
        self.assertEqual(list(self.root.glob("*.bak")), [])
        status, _headers, data = self._convert(self._detail(), [2])
        self.assertEqual((status, json.loads(data)["converted"]), (200, 1))

    @staticmethod
    def _songs(lines, uncertain=False):
        info = zh_traditional.song_candidates([{"entry": i + 1, "lyrics": [text]} for i, text in enumerate(lines)], (), uncertain)
        return info["count"], [(item["before"], item["default_checked"]) for item in info["lines"]]

    def test_traditional_song_lists_nothing_even_when_s2tw_would_rewrite_a_variant(self) -> None:
        names = ["范瑋琪", "于文文", "后弘", "游鴻明", "郁可唯", "台北", "裏面", "了解", "祢是神", "這是第一句"]
        for text in names:
            self.assertEqual(self._songs([text, "這是第一句"]), (0, []), text)
        info = self._load(chr(10).join("[00:%02d.00]%s" % (i + 1, text) for i, text in enumerate(names)).encode("utf-8"))["simplified"]
        self.assertEqual((info["count"], info["lines"], info["headers"]), (0, [], []))

    def test_real_simplified_lines_are_listed_and_checked(self) -> None:
        texts = ["后来的我们", "我爱你", "这是我们的歌", "说好的幸福呢", "爱情转移", "着急"]
        count, lines = self._songs(texts)
        self.assertEqual((count, lines), (6, [(text, True) for text in texts]))
        self.assertEqual(self._songs(texts, True), (6, [(text, False) for text in texts]))

    def test_variant_line_in_a_simplified_song_is_listed_but_not_checked(self) -> None:
        self.assertEqual(self._songs(["后面", "我们"]), (1, [("后面", False), ("我们", True)]))

    def test_japanese_kanji_are_not_simplified_but_simplified_hanzi_in_a_japanese_song_is_unchecked(self) -> None:
        self.assertEqual(self._songs(["国境", "独奏", "きみ"]), (0, []))
        self.assertEqual(self._songs(["国境", "独奏", "きみ", "我们一起走"]), (1, [("国境", False), ("独奏", False), ("我们一起走", False)]))

    def test_header_only_simplified_is_a_candidate_and_count_counts_simplified_items(self) -> None:
        info = zh_traditional.song_candidates([{"entry": 1, "lyrics": ["后面"]}], [(0, "[ti:爱你]")])
        self.assertEqual((info["count"], [i["default_checked"] for i in info["lines"]], [i["default_checked"] for i in info["headers"]]), (1, [False], [True]))
        self.assertEqual(zh_traditional.song_candidates([{"entry": 1, "lyrics": ["后面"]}], [(0, "[ti:范瑋琪]")])["lines"], [])


    def test_halfwidth_katakana_and_katakana_extensions_are_kana(self) -> None:
        for text in ("ｱｲ爱你", "ｶﾞ爱", "ㇰ爱", "ｦ爱"):
            self.assertTrue(zh_traditional.has_kana(text), text)
            self.assertIsNone(zh_traditional.convert_line(text), text)
        info = zh_traditional.song_candidates([{"entry": 1, "lyrics": ["ｱｲ爱你"]}, {"entry": 2, "lyrics": ["爱你"]}])
        self.assertEqual(([line["entry"] for line in info["lines"]], info["japanese"]), ([2], True))
        self.assertEqual([line["default_checked"] for line in info["lines"]], [False])

    def test_middle_dot_and_long_vowel_mark_are_not_kana(self) -> None:
        for text in ("哈利・波特", "爱ー你", "・", "ー"):
            self.assertFalse(zh_traditional.has_kana(text), text)
        self.assertEqual(zh_traditional.convert_line("哈利・波特 爱你一万年"), "哈利・波特 愛你一萬年")
        info = zh_traditional.song_candidates([{"entry": 1, "lyrics": ["哈利・波特"]}, {"entry": 2, "lyrics": ["爱你一万年"]}])
        self.assertEqual((info["japanese"], [line["default_checked"] for line in info["lines"]]), (False, [True]))
        for kana in ("か", "カ", "ヽ"):
            self.assertTrue(zh_traditional.has_kana(kana), kana)

    def test_uncertain_encoding_leaves_every_candidate_unchecked(self) -> None:
        rows = [{"entry": 1, "lyrics": ["爱你"]}]
        headers = [(0, "[ti:爱你]")]
        sure = zh_traditional.song_candidates(rows, headers)
        self.assertEqual((sure["uncertain"], sure["lines"][0]["default_checked"], sure["headers"][0]["default_checked"]), (False, True, True))
        unsure = zh_traditional.song_candidates(rows, headers, True)
        self.assertEqual((unsure["uncertain"], unsure["lines"][0]["default_checked"], unsure["headers"][0]["default_checked"]), (True, False, False))
        self.assertIs(self._load(CHINESE.encode("utf-8"))["simplified"]["uncertain"], False)

    def test_header_tags_are_candidates_and_only_their_text_changes(self) -> None:
        original = ("[ti:爱你一万年]\r\n[ar:歌手]\r\n[al:Album]\r\n[by:tool]\r\n[offset:0]\r\n" + CHINESE).encode("utf-8")
        detail = self._load(original)
        info = detail["simplified"]
        self.assertEqual(info["count"], 3)   # 2 lyric lines + [ti:]; [ar:] is already Traditional
        self.assertEqual([(item["line"], item["tag"], item["before"], item["after"], item["default_checked"]) for item in info["headers"]],
                         [(0, "ti", "爱你一万年", "愛你一萬年", True)])
        status, _headers, data = self._json_request(
            "POST", "/api/song/0/convert", {"base_sha256": detail["base_sha256"], "entries": [], "headers": [0]})
        self.assertEqual(status, 200, data)
        self.assertEqual(json.loads(data)["converted"], 1)
        before, after = original.split(b"\n"), self.output.read_bytes().split(b"\n")
        self.assertEqual(after[0], "[ti:愛你一萬年]\r".encode("utf-8"))
        self.assertEqual(after[1:], before[1:])   # lyric lines were not listed
        self.assertEqual([p.read_bytes() for p in self.root.glob("*.bak")], [original])
        self.assertEqual(self._detail()["simplified"]["headers"], [])

    def test_header_rules_match_lyric_lines(self) -> None:
        self.assertIsNone(zh_traditional.convert_header("[ti:君の声]"))   # kana: skipped like a lyric line
        self.assertIsNone(zh_traditional.convert_header("[ar:歌手]"))   # nothing to change
        self.assertIsNone(zh_traditional.convert_header("[00:01.00]爱你"))   # not an ID tag
        self.assertEqual(zh_traditional.convert_header("[al:爱你]  "), ("爱你", "愛你"))
        detail = self._load(("[ti:爱你]\r\n" + JAPANESE).encode("utf-8"))
        info = detail["simplified"]
        self.assertEqual((info["japanese"], info["headers"][0]["default_checked"], info["lines"][0]["default_checked"]), (True, False, False))

    def test_invalid_header_indexes_are_rejected(self) -> None:
        detail = self._load(("[ti:爱你]\r\n" + CHINESE).encode("utf-8"))
        for headers in ([5], [-1], [0, 0], [1], ["0"], [True], "0"):   # line 1 is a lyric line, not an ID tag
            status, _headers, data = self._json_request(
                "POST", "/api/song/0/convert", {"base_sha256": detail["base_sha256"], "entries": [], "headers": headers})
            self.assertEqual(status, 400, (headers, data))


if __name__ == "__main__":
    unittest.main()
