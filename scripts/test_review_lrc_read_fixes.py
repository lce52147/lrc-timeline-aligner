"""Review page reads: several times on one line, a trailing * that is lyric text, [offset:]."""
from __future__ import annotations

import json
import os
import subprocess
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

    def test_saving_a_time_clears_the_marks_and_keeps_every_other_byte(self) -> None:
        detail = self._load(self.SONG)
        status, _headers, data = self._save(detail, edits=[(2, 250), (3, 350), (4, 450)])
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(),
                         b"[00:01.00]one\n[00:02.50]f***\n[00:03.50]Song Title\n[00:04.50]end*\n[00:05.00]tail\n")

    def test_saving_a_time_keeps_crlf_and_the_star_run(self) -> None:
        detail = self._load(b"[00:01.00]one\r\n** [00:02.00]f**\r\n[00:05.00]tail\r\n")
        status, _headers, data = self._save(detail, edits=[(2, 250)])
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(), b"[00:01.00]one\r\n[00:02.50]f**\r\n[00:05.00]tail\r\n")

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


class OffsetTests(_PageFixture):
    def test_positive_offset_shows_the_lyric_earlier(self) -> None:
        raw = b"[ti:t]\n[offset:+500]\n[00:10.00]a\n[00:20.25]b\n"
        self.assertEqual(review_server._canonical_lrc(raw), b"[ti:t]\n[00:09.50]a\n[00:19.75]b\n")

    def test_negative_offset_shows_the_lyric_later(self) -> None:
        self.assertEqual(review_server._canonical_lrc(b"[offset:-1500]\n[00:10.00]a\n"), b"[00:11.50]a\n")
        self.assertEqual(review_server._canonical_lrc(b"[OFFSET: -250 ]\r\n[00:10.00]a\r\n"), b"[00:10.25]a\r\n")

    def test_result_is_rounded_to_a_hundredth_and_clamped_at_zero(self) -> None:
        self.assertEqual(review_server._canonical_lrc(b"[offset:+503]\n[00:10.00]a\n"), b"[00:09.50]a\n")
        self.assertEqual(review_server._canonical_lrc(b"[offset:+507]\n[00:10.00]a\n"), b"[00:09.49]a\n")
        self.assertEqual(review_server._canonical_lrc(b"[offset:+5000]\n[00:01.00]a\n[00:09.00]b\n"),
                         b"[00:00.00]a\n[00:04.00]b\n")
        self.assertEqual(review_server._canonical_lrc(b"[offset:+500]\n[00:10.123]a\n"), b"[00:09.62]a\n")

    def test_offset_applies_to_every_tag_of_a_multi_tag_line(self) -> None:
        self.assertEqual(review_server._canonical_lrc(b"[offset:+1000]\n[00:10.00][00:40.00]a\n"),
                         b"[00:09.00]a\n[00:39.00]a\n")

    def test_zero_offset_and_odd_values_are_left_alone(self) -> None:
        for raw in (b"[offset:0]\n[00:10.00]a\n", b"[offset:+0]\n[00:10.00]a\n", b"[offset:abc]\n[00:10.00]a\n"):
            self.assertEqual(review_server._canonical_lrc(raw), raw)

    def test_page_shows_and_plays_the_converted_time_with_a_notice(self) -> None:
        detail = self._load(b"[offset:+500]\n[00:01.00]one\n[00:02.00]two\n[00:05.00]end\n")
        self.assertFalse(detail["read_only"], detail["read_only_reason"])
        self.assertEqual([row["current_cs"] for row in detail["rows"]], [50, 150, 450])
        self.assertEqual(detail["offset_ms"], 500)

    def test_save_writes_the_converted_form_and_keeps_the_original(self) -> None:
        original = b"[offset:+500]\n[00:01.00]one\n[00:02.00]two\n[00:05.00]end\n"
        detail = self._load(original)
        status, _headers, data = self._save(detail, edits=[(2, 160)])
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(), b"[00:00.50]one\n[00:01.60]two\n[00:04.50]end\n")
        self.assertEqual(self._backups(), [original])
        self.assertEqual(json.loads(data)["offset_ms"], 0)

    def test_export_text_has_the_converted_times(self) -> None:
        self._real_flac()
        self._load(b"[offset:-1000]\n[00:01.00]one\n[00:02.00]two\n")
        status, _headers, data = self._json_request("POST", "/api/song/0/export", {"audio": "song.flac"})
        self.assertEqual(status, 200, data)
        import mutagen.flac
        text = mutagen.flac.FLAC(str(self.audio))["LYRICS"][0]
        self.assertEqual(text.split("\r\n"), ["[00:02.00]one", "[00:03.00]two"])


class TitleCardOffsetTests(_PageFixture):
    """The tool's title card is 0 s in the file as written: an [offset:] never moves it (a real lyric at 0 s moves like any other)."""
    HEADER = b"[ti:Song]\n[ar:Artist]\n"

    def test_negative_offset_keeps_the_title_card_at_zero_and_moves_the_rest(self) -> None:
        raw = self.HEADER + b"[offset:-300]\n[00:00.00]Artist - Song\n[00:10.00]a\n[00:20.00]b\n"
        self.assertEqual(review_server._canonical_lrc(raw), self.HEADER + b"[00:00.00]Artist - Song\n[00:10.30]a\n[00:20.30]b\n")

    def test_positive_offset_keeps_the_title_card_at_zero_and_moves_the_rest(self) -> None:
        raw = self.HEADER + b"[offset:+300]\n[00:00.00]Artist - Song\n[00:10.00]a\n"
        self.assertEqual(review_server._canonical_lrc(raw), self.HEADER + b"[00:00.00]Artist - Song\n[00:09.70]a\n")

    def test_a_card_without_an_artist_is_recognized_by_the_header(self) -> None:
        raw = b"[ti:Song]\n[offset:-300]\n[00:00.00]Song\n[00:10.00]a\n"
        self.assertEqual(review_server._canonical_lrc(raw), b"[ti:Song]\n[00:00.00]Song\n[00:10.30]a\n")

    def test_a_card_is_recognized_when_the_offset_header_comes_after_the_card_header_or_before_it(self) -> None:
        for headers in (b"[offset:-300]\n[ti:Song]\n", b"[ti:Song]\n[offset:-300]\n"):
            self.assertEqual(review_server._canonical_lrc(headers + b"[00:00.00]Song\n[00:10.00]a\n"), b"[ti:Song]\n[00:00.00]Song\n[00:10.30]a\n")

    def test_a_real_lyric_at_zero_seconds_moves_like_any_other(self) -> None:
        for raw in (b"[offset:-300]\n[00:00.00]a plain lyric\n[00:10.00]b\n", b"[ti:Song]\n[offset:-300]\n[00:00.00]one\n[00:00.00]two\n[00:10.00]b\n"):
            out = review_server._canonical_lrc(raw)
            self.assertIn(b"[00:00.30]", out, raw)
            self.assertNotIn(b"[00:00.00]", out, raw)

    def test_only_the_first_timed_line_can_be_the_card(self) -> None:
        raw = b"[ti:Song]\n[offset:-300]\n[00:02.00]first\n[00:00.00]Artist - Song\n"
        self.assertEqual(review_server._canonical_lrc(raw), b"[ti:Song]\n[00:02.30]first\n[00:00.30]Artist - Song\n")

    def test_the_card_stays_when_the_text_is_not_a_card_only_at_zero_without_an_offset(self) -> None:
        raw = b"[ti:Song]\n[00:00.00]Song\n[00:10.00]a\n"
        self.assertEqual(review_server._canonical_lrc(raw), raw)

    def test_page_shows_no_extra_row_for_the_card_and_save_writes_it_back_at_zero(self) -> None:
        for offset, shown in ((b"-300", [1030, 2030]), (b"+300", [970, 1970])):
            original = self.HEADER + b"[offset:" + offset + b"]\n[00:00.00]Artist - Song\n[00:10.00]one\n[00:20.00]two\n"
            detail = self._load(original)
            self.assertFalse(detail["read_only"], detail["read_only_reason"])
            self.assertEqual([row["current_cs"] for row in detail["rows"]], shown, offset)
            status, _headers, data = self._save(detail, edits=[(2, shown[1] + 10)])
            self.assertEqual(status, 200, data)
            self.assertEqual(self.output.read_bytes(), self.HEADER + b"[00:00.00]Artist - Song\n" + ("[00:%05.2f]one\n[00:%05.2f]two\n" % (shown[0] / 100, (shown[1] + 10) / 100)).encode())
            self.assertEqual(self._backups()[-1], original)


class OffsetHeaderAndRepeatedTagTests(_PageFixture):
    def test_the_same_offset_twice_is_applied_once_and_both_headers_go(self) -> None:
        raw = b"[offset:+500]\n[offset:+500]\n[00:10.00]a\n"
        self.assertEqual(review_server._canonical_lrc(raw), b"[00:09.50]a\n")
        info = review_server._canonical_lrc_info(raw)[1]
        self.assertEqual((info["offset_ms"], info["offset_values"]), (500, [500]))

    def test_different_offsets_use_the_first_report_every_value_and_drop_every_header(self) -> None:
        raw = b"[ti:t]\n[offset:+500]\n[00:10.00]a\n[offset:-200]\n[00:20.00]b\n[OFFSET: +500 ]\n"
        data, info = review_server._canonical_lrc_info(raw)
        self.assertEqual(data, b"[ti:t]\n[00:09.50]a\n[00:19.50]b\n")
        self.assertEqual((info["offset_ms"], info["offset_values"]), (500, [500, -200]))

    def test_page_detail_and_save_for_different_offsets(self) -> None:
        original = b"[offset:+500]\n[offset:-200]\n[00:01.00]one\n[00:02.00]two\n[00:05.00]end\n"
        detail = self._load(original)
        self.assertEqual((detail["offset_ms"], detail["offset_values"]), (500, [500, -200]))
        self.assertEqual([row["current_cs"] for row in detail["rows"]], [50, 150, 450])
        status, _headers, data = self._save(detail, edits=[(2, 160)])
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(), b"[00:00.50]one\n[00:01.60]two\n[00:04.50]end\n")
        self.assertEqual(self._backups(), [original])
        self.assertEqual(json.loads(data)["offset_values"], [])

    def test_an_offset_of_zero_first_is_reported_and_kept_as_the_one_used(self) -> None:
        raw = b"[offset:0]\n[offset:300]\n[00:10.00]a\n"
        data, info = review_server._canonical_lrc_info(raw)
        self.assertEqual(data, b"[00:10.00]a\n")
        self.assertEqual((info["offset_ms"], info["offset_values"]), (0, [0, 300]))
        detail = self._load(raw)
        self.assertEqual((detail["offset_ms"], detail["offset_values"]), (0, [0, 300]))
        self.assertEqual([row["current_cs"] for row in detail["rows"]], [1000])

    def test_a_lone_zero_offset_or_two_zeros_say_nothing_and_leave_the_file(self) -> None:
        for raw in (b"[offset:0]\n[00:10.00]a\n", b"[offset:0]\n[offset: -0 ]\n[00:10.00]a\n"):
            data, info = review_server._canonical_lrc_info(raw)
            self.assertEqual((data, info["offset_ms"], info["offset_values"]), (raw, 0, []), raw)

    def test_the_same_time_twice_on_a_line_counts_as_a_multi_tag_line(self) -> None:
        info = review_server._canonical_lrc_info(b"[00:01.00][00:01.00]a\n[00:02.00]b\n")
        self.assertEqual(info[0], b"[00:01.00]a\n[00:02.00]b\n")
        self.assertEqual(info[1]["multi_tag_lines"], 1)
        self.assertEqual(review_server._canonical_lrc_info(b"[00:01.00][0:01]a\n")[1]["multi_tag_lines"], 1)
        self.assertEqual(review_server._canonical_lrc_info(b"[00:01.00]a\n[00:02.00]b\n")[1]["multi_tag_lines"], 0)

    def test_page_detail_and_save_for_a_repeated_tag(self) -> None:
        original = b"[00:01.00][00:01.00]hook\n[00:02.00]verse\n"
        detail = self._load(original)
        self.assertEqual(detail["multi_tag_lines"], 1)
        self.assertEqual([(row["current_cs"], row["lyric"]) for row in detail["rows"]], [(100, "hook"), (200, "verse")])
        status, _headers, data = self._save(detail, edits=[(2, 210)])
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(), b"[00:01.00]hook\n[00:02.10]verse\n")
        self.assertEqual(self._backups(), [original])

    def test_the_page_notes_say_it(self) -> None:
        """index.html's fileShapeNotes, run on its own: the texts for the repeated tag, the same offset, and different offsets."""
        html = (SCRIPTS / "review_page" / "index.html").read_text(encoding="utf-8")
        start = html.index("function fileShapeNotes(detail)")
        end = html.index("\n  }\n", start) + 4
        harness = html[start:end] + """
  const out = {
    none: fileShapeNotes({}),
    repeated: fileShapeNotes({multi_tag_lines: 1}),
    same: fileShapeNotes({offset_ms: 500, offset_values: [500]}),
    different: fileShapeNotes({offset_ms: -200, offset_values: [-200, 500, 0]}),
    old_server: fileShapeNotes({offset_ms: 500}),
    zero_first: fileShapeNotes({offset_ms: 0, offset_values: [0, 300]}),
    zero_alone: fileShapeNotes({offset_ms: 0, offset_values: []}),
  };
  console.log(JSON.stringify(out));
"""
        result = subprocess.run(["node", "-"], input=harness, capture_output=True, text=True, encoding="utf-8", creationflags=0x08000000 if os.name == "nt" else 0)
        self.assertEqual(result.returncode, 0, result.stderr)
        notes = json.loads(result.stdout)
        self.assertEqual(notes["none"], [])
        self.assertEqual(len(notes["repeated"]), 1)
        self.assertIn("同一個時間重複", notes["repeated"][0])
        self.assertEqual(notes["same"], ["此檔有 offset +500 ms，已換算進時間；存檔會照換算後的格式寫回（原檔有備份）"])
        self.assertEqual(notes["old_server"], notes["same"])
        self.assertEqual(len(notes["different"]), 1)
        for fragment in ("3 個不同的 offset", "-200 ms、+500 ms、0 ms", "只採用第一個（-200 ms）", "全部 offset 標頭都會換算掉", "原檔有備份"):
            self.assertIn(fragment, notes["different"][0])
        self.assertEqual(notes["zero_alone"], [])
        self.assertEqual(len(notes["zero_first"]), 1)
        for fragment in ("2 個不同的 offset", "0 ms、+300 ms", "只採用第一個（0 ms）", "原檔有備份"):
            self.assertIn(fragment, notes["zero_first"][0])

class SongIdentityTests(_PageFixture):
    """A write that names another song than the one at that position is refused with 409 and writes nothing."""
    OTHER = "D:\elsewhere\other.align-report.json"

    def _post(self, endpoint, payload):
        return self._request("POST", "/api/song/0/" + endpoint, json.dumps(payload).encode(), **{
            "X-Review-Token": self.server.token, "Origin": f"http://{self.base}", "Content-Type": "application/json"})

    def _tree(self):
        return {path.name: path.read_bytes() for path in self.root.iterdir() if path.is_file()}

    def test_decision_with_another_song_id_is_refused_and_without_one_is_recorded(self) -> None:
        before = self._tree()
        for extra in ({"song_id": self.OTHER}, {"song_id": ""}, {"song_id": 3}, {"song_id": None}):
            status, _h, data = self._post("decision", {"entry": 2, "decision": "keep", "cs": None, "source": "keep", **extra})
            self.assertEqual(status, 409, data)
            self.assertTrue(json.loads(data)["song_changed"])
        self.assertEqual(self._tree(), before)
        self.assertFalse(list(self.root.glob("*.review-decisions.jsonl")))
        # A page loaded before a hot update sends no song_id and cannot show an error: its decision is still logged.
        status, _h, data = self._post("decision", {"entry": 2, "decision": "keep", "cs": None, "source": "keep"})
        self.assertEqual(status, 200, data)
        logs = list(self.root.glob("*.review-decisions.jsonl"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(len(logs[0].read_text(encoding="utf-8").splitlines()), 1)
        status, _h, data = self._post("decision", {"song_id": self._detail()["report"], "entry": 2, "decision": "keep", "cs": None, "source": "keep"})
        self.assertEqual(status, 200, data)
        self.assertEqual(len(logs[0].read_text(encoding="utf-8").splitlines()), 2)

    def test_romaji_and_export_without_a_song_id_say_the_page_is_outdated(self) -> None:
        before = self._tree()
        for endpoint, payload in (("romaji", {"entry": 1, "romaji": "ichi"}), ("export", {"audio": "", "check": True}),
                                  ("export", {"audio": "song.flac"})):
            status, _h, data = self._post(endpoint, payload)
            self.assertEqual(status, 409, data)
            body = json.loads(data)
            self.assertIs(body["page_outdated"], True)
            self.assertNotIn("song_changed", body)
            self.assertEqual(body["error"], review_server.PAGE_OUTDATED_MESSAGE)
            self.assertIn("F5", body["error"])
        self.assertEqual(self._tree(), before)
        self.assertFalse(list(self.root.glob("*.romaji-overrides.json")))

    def test_the_page_from_before_the_update_shows_that_message(self) -> None:
        """The api() and the two error displays of the page at 956b3514 (it sends no song_id): the text reaches the user."""
        old_api = r"""
  const token = 't';
  const api = async (path, options={}) => {
    const headers = Object.assign({'X-Review-Token':token}, options.body ? {'Content-Type':'application/json'} : {}, options.headers || {});
    try {
      const response = await fetch(path, Object.assign({}, options, {headers}));
      const data = await response.json().catch(() => ({error:response.statusText}));
      if (!response.ok) throw Error(response.status === 403 ? 'forbidden' : (data.error || response.statusText));
      return data;
    } catch (error) { throw error; }
  };
  const shown = {};
  const edit = {};
  const romajiPost = async body => {
    try { return await api('/api/song/0/romaji', {method:'POST', body:JSON.stringify(Object.assign({entry:1}, body))}); }
    catch (error) { edit.message = error.message; return null; }
  };
  const doExport = async () => {
    try { await api('/api/song/0/export', {method:'POST', body:JSON.stringify({audio:'song.flac'})}); }
    catch (error) { shown.export = error && error.message ? error.message : String(error); }
  };
"""
        harness = old_api + """
  globalThis.fetch = async () => ({ok:false, status:409, statusText:'Conflict', json: async () => ({error:MESSAGE, page_outdated:true})});
  (async () => { await romajiPost({romaji:'ichi'}); await doExport(); console.log(JSON.stringify({romaji:edit.message, export:shown.export})); })();
""".replace("MESSAGE", json.dumps(review_server.PAGE_OUTDATED_MESSAGE))
        result = subprocess.run(["node", "-"], input=harness, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"romaji": review_server.PAGE_OUTDATED_MESSAGE, "export": review_server.PAGE_OUTDATED_MESSAGE})

    def test_the_same_report_written_with_other_slashes_is_accepted(self) -> None:
        status, _h, data = self._post("decision", {"song_id": self._detail()["report"].replace("\\", "/"), "entry": 2, "decision": "keep", "cs": None, "source": "keep"})
        self.assertEqual(status, 200, data)

    def test_romaji_edit_with_another_song_id_is_refused(self) -> None:
        before = self._tree()
        status, _h, data = self._post("romaji", {"song_id": self.OTHER, "entry": 1, "romaji": "ichi"})
        self.assertEqual(status, 409, data)
        self.assertEqual(self._tree(), before)
        self.assertFalse(list(self.root.glob("*.romaji-overrides.json")))

    def test_export_check_and_export_with_another_song_id_are_refused(self) -> None:
        before = self._tree()
        for payload in ({"audio": "", "check": True}, {"audio": "song.flac"}):
            status, _h, data = self._post("export", {"song_id": self.OTHER, **payload})
            self.assertEqual(status, 409, data)
        self.assertEqual(self._tree(), before)
        self.assertFalse((self.root / "export.log").exists())

    def test_save_and_convert_with_another_song_id_are_refused_and_without_one_still_work(self) -> None:
        detail = self._detail()
        before = self._tree()
        edit = {"entry": 1, "cs": 150, "source": "ear", "nudge_cs": 0}
        status, _h, data = self._post("save", {"song_id": self.OTHER, "base_sha256": detail["base_sha256"], "edits": [edit]})
        self.assertEqual(status, 409, data)
        status, _h, data = self._post("convert", {"song_id": self.OTHER, "base_sha256": detail["base_sha256"], "entries": [1]})
        self.assertEqual(status, 409, data)
        self.assertEqual(self._tree(), before)
        status, _h, data = self._post("save", {"base_sha256": detail["base_sha256"], "edits": [edit]})
        self.assertEqual(status, 200, data)
        detail = self._detail()
        status, _h, data = self._post("save", {"song_id": self._detail()["report"], "base_sha256": detail["base_sha256"],
                                               "edits": [{**edit, "cs": 160}]})
        self.assertEqual(status, 200, data)
        self.assertIn(b"[00:01.60]one", self.output.read_bytes())


    def test_the_page_names_the_song_in_every_write_and_reloads_on_409(self) -> None:
        html = (SCRIPTS / "review_page" / "index.html").read_text(encoding="utf-8")
        for endpoint in ("decision", "romaji", "export", "convert", "save"):
            self.assertIn("/" + endpoint, html)
        self.assertEqual(html.count("song_id:draftId("), 6, "decision, save, export check, romaji, convert, export")
        self.assertIn("response.status === 409 && data.song_changed", html)
        self.assertIn("load().catch", html)


if __name__ == "__main__":
    unittest.main()
