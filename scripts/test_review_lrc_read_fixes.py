"""Review page reads: several times on one line, a trailing * that is lyric text, [offset:]."""
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


class _PageFixture(unittest.TestCase):
    _start = _fixture_module.ReviewServerTests._start
    _request = _fixture_module.ReviewServerTests._request
    _json_request = _fixture_module.ReviewServerTests._json_request
    _detail = _fixture_module.ReviewServerTests._detail
    _real_flac = _fixture_module.ReviewServerTests._real_flac

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"LRC_EXPORT_ARCHIVE_DIR": "off"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._start(False)
        for sidecar in self.root.glob("song.auto.lrc"):
            sidecar.unlink()

    def tearDown(self) -> None:
        _fixture_module.ReviewServerTests.tearDown(self)

    def _load(self, data: bytes):
        self.output.write_bytes(data)
        self.store.refresh_if_changed()
        return self._detail()

    def _save(self, detail, edits=(), deletes=()):
        body = {"base_sha256": detail["base_sha256"],
                "edits": [{"entry": e, "cs": cs, "source": "ear", "nudge_cs": 0} for e, cs in edits]}
        if deletes:
            body["deletes"] = [{"entry": e, "line_key": detail["rows"][e - 1]["line_key"]} for e in deletes]
        return self._json_request("POST", "/api/song/0/save", body)

    def _backups(self):
        return [path.read_bytes() for path in self.root.glob("song.lrc.orig-*.bak")]


class MultiTagTests(_PageFixture):
    def test_canonical_expands_sorts_stably_and_keeps_headers_on_top(self) -> None:
        raw = (b"[ti:t]\n[00:10.00][00:40.00]chorus\n[00:10.00][00:40.00]chorus tr\n[00:20.00]verse\n"
               b"[00:30.50][01:00.000]x\n\n")
        want = (b"[ti:t]\n[00:10.00]chorus\n[00:10.00]chorus tr\n[00:20.00]verse\n[00:30.50]x\n"
                b"[00:40.00]chorus\n[00:40.00]chorus tr\n[01:00.00]x\n\n")
        self.assertEqual(review_server._canonical_lrc(raw), want)
        self.assertEqual(review_server._canonical_lrc(want), want)

    def test_single_tag_files_are_not_reordered(self) -> None:
        raw = b"[00:20.00]b\n[00:10.00]a\n"
        self.assertEqual(review_server._canonical_lrc(raw), raw)

    def test_millisecond_and_odd_tags_in_a_chain_are_all_converted(self) -> None:
        self.assertEqual(review_server._canonical_lrc(b"[0:01.543][00:03][00:05.5]a\n"),
                         b"[00:01.54]a\n[00:03.00]a\n[00:05.50]a\n")
        self.assertEqual(review_server._canonical_lrc(b"*[00:01.00][00:02.00]a*\r\n"), b"*[00:01.00]a*\r\n*[00:02.00]a*\r\n")

    def test_same_time_twice_on_a_line_is_one_line(self) -> None:
        self.assertEqual(review_server._canonical_lrc(b"[00:01.00][00:01.00]a\n"), b"[00:01.00]a\n")

    def test_page_shows_every_occurrence_with_its_time_and_a_notice(self) -> None:
        original = b"[00:01.00][00:50.00]hook\n[00:02.00]verse\n[00:30.00]coda\n"
        detail = self._load(original)
        self.assertFalse(detail["read_only"], detail["read_only_reason"])
        self.assertEqual([(row["current_cs"], row["lyric"]) for row in detail["rows"]],
                         [(100, "hook"), (200, "verse"), (3000, "coda"), (5000, "hook")])
        self.assertEqual(detail["multi_tag_lines"], 1)

    def test_translation_pair_with_multi_tags_keeps_original_then_translation(self) -> None:
        detail = self._load(b"[00:01.00][00:20.00]orig\n[00:01.00][00:20.00]trans\n[00:10.00]mid\n")
        self.assertEqual([(row["current_cs"], row["lyrics"]) for row in detail["rows"]],
                         [(100, ["orig", "trans"]), (1000, ["mid"]), (2000, ["orig", "trans"])])

    def test_editing_one_time_touches_only_that_line_and_leaves_a_backup(self) -> None:
        original = b"[00:01.00][00:30.00]hook\n[00:02.00]verse\n[00:05.00]end\n"
        detail = self._load(original)
        status, _headers, data = self._save(detail, edits=[(4, 3100)])
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(), b"[00:01.00]hook\n[00:02.00]verse\n[00:05.00]end\n[00:31.00]hook\n")
        self.assertEqual(self._backups(), [original])
        again = json.loads(data)
        self.assertEqual(again["multi_tag_lines"], 0)

    def test_deleting_the_second_occurrence_keeps_the_first(self) -> None:
        detail = self._load(b"[00:01.00][00:30.00]hook\n[00:02.00]verse\n[00:05.00]end\n")
        status, _headers, data = self._save(detail, deletes=[4])
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(), b"[00:01.00]hook\n[00:02.00]verse\n[00:05.00]end\n")

    def test_convert_to_traditional_writes_the_expanded_form(self) -> None:
        detail = self._load("[00:01.00][00:30.00]爱你\n[00:02.00]我们\n".encode("utf-8"))
        self.assertEqual(len(detail["rows"]), 3)
        body = {"base_sha256": detail["base_sha256"], "entries": [1, 2, 3]}
        status, _headers, data = self._json_request("POST", "/api/song/0/convert", body)
        if status == 503:
            self.skipTest("OpenCC unavailable")
        self.assertEqual(status, 200, data)
        text = self.output.read_bytes().decode("utf-8")
        self.assertEqual(text, "[00:01.00]愛你\n[00:02.00]我們\n[00:30.00]愛你\n")

    def test_export_check_counts_the_expanded_lines_and_the_tag_text_is_expanded(self) -> None:
        self._real_flac()
        self._load(b"[00:01.00][00:30.00]hook\n[00:02.00]verse\n")
        status, _headers, data = self._json_request("POST", "/api/song/0/export", {"audio": "", "check": True})
        self.assertEqual((status, json.loads(data)["lrc_lines"]), (200, 3))
        text = review_server.lyrics_tag_export.lrc_text(review_server._read_lrc(self.output))
        self.assertEqual(text.split("\r\n"), ["[00:01.00]hook", "[00:02.00]verse", "[00:30.00]hook"])

    def test_export_writes_the_expanded_text_into_the_flac(self) -> None:
        self._real_flac()
        self._load(b"[00:01.00][00:30.00]hook\n[00:02.00]verse\n")
        status, _headers, data = self._json_request("POST", "/api/song/0/export", {"audio": "song.flac"})
        self.assertEqual(status, 200, data)
        import mutagen.flac
        text = mutagen.flac.FLAC(str(self.audio))["LYRICS"][0]
        self.assertEqual(text.split("\r\n"), ["[00:01.00]hook", "[00:02.00]verse", "[00:30.00]hook"])


class TrailingStarTests(_PageFixture):
    SONG = b"[00:01.00]one\n*[00:02.00]f***\n*[00:03.00]Song Title*\n[00:04.00]end*\n[00:05.00]tail\n"

    def test_strip_rules(self) -> None:
        strip = review_server._strip_line_marks
        self.assertEqual(strip(b"*[00:10.00]f***"), b"[00:10.00]f***")
        self.assertEqual(strip(b"*[00:10.00]f**"), b"[00:10.00]f**")
        self.assertEqual(strip(b"*[00:10.00]f*"), b"[00:10.00]f")
        self.assertEqual(strip(b"*[00:10.00]f *\r"), b"[00:10.00]f\r")
        self.assertEqual(strip(b"**[00:10.00]f*"), b"[00:10.00]f")
        self.assertEqual(strip(b"[00:10.00]f*"), b"[00:10.00]f*")
        self.assertEqual(strip(b"[00:10.00]f***"), b"[00:10.00]f***")
        self.assertEqual(strip(b"*[00:01.00]Song Title*"), b"[00:01.00]Song Title")

    def test_page_shows_the_star_run_as_lyric_text(self) -> None:
        detail = self._load(self.SONG)
        self.assertFalse(detail["read_only"], detail["read_only_reason"])
        self.assertEqual([row["lyric"] for row in detail["rows"]], ["one", "f***", "Song Title", "end*", "tail"])
        self.assertEqual([row["user_marked"] for row in detail["rows"]], [False, True, True, False, False])

    def test_saving_a_time_changes_only_the_tag(self) -> None:
        detail = self._load(self.SONG)
        status, _headers, data = self._save(detail, edits=[(2, 250), (3, 350), (4, 450)])
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(),
                         b"[00:01.00]one\n*[00:02.50]f***\n*[00:03.50]Song Title*\n[00:04.50]end*\n[00:05.00]tail\n")

    def test_saving_a_time_keeps_crlf_and_double_marks(self) -> None:
        detail = self._load(b"[00:01.00]one\r\n** [00:02.00]f**\r\n[00:05.00]tail\r\n")
        status, _headers, data = self._save(detail, edits=[(2, 250)])
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(), b"[00:01.00]one\r\n** [00:02.50]f**\r\n[00:05.00]tail\r\n")

    def test_deleting_a_line_leaves_the_others_untouched(self) -> None:
        detail = self._load(self.SONG)
        status, _headers, data = self._save(detail, deletes=[1])
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(),
                         b"*[00:02.00]f***\n*[00:03.00]Song Title*\n[00:04.00]end*\n[00:05.00]tail\n")

    def test_export_text_drops_marks_but_keeps_a_star_run(self) -> None:
        self._real_flac()
        self._load(self.SONG)
        status, _headers, data = self._json_request("POST", "/api/song/0/export", {"audio": "song.flac"})
        self.assertEqual(status, 200, data)
        import mutagen.flac
        text = mutagen.flac.FLAC(str(self.audio))["LYRICS"][0]
        self.assertEqual(text.split("\r\n"), ["[00:01.00]one", "[00:02.00]f***", "[00:03.00]Song Title",
                                               "[00:04.00]end*", "[00:05.00]tail"])

    def test_convert_to_traditional_keeps_marks_and_stars(self) -> None:
        detail = self._load("[00:01.00]one\n*[00:02.00]爱你***\n*[00:03.00]我们*\n[00:05.00]tail\n".encode("utf-8"))
        status, _headers, data = self._json_request(
            "POST", "/api/song/0/convert", {"base_sha256": detail["base_sha256"], "entries": [2, 3]})
        if status == 503:
            self.skipTest("OpenCC unavailable")
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes().decode("utf-8"),
                         "[00:01.00]one\n*[00:02.00]愛你***\n*[00:03.00]我們*\n[00:05.00]tail\n")


if __name__ == "__main__":
    unittest.main()
