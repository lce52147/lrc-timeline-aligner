"""Review page: delete a whole line (save body `deletes`). Uses the review-server fixture song (entries one / two a+two b / three / four / five)."""
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



class ReviewDeleteTests(unittest.TestCase):
    _start = _fixture_module.ReviewServerTests._start
    _request = _fixture_module.ReviewServerTests._request
    _json_request = _fixture_module.ReviewServerTests._json_request
    _detail = _fixture_module.ReviewServerTests._detail

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"LRC_EXPORT_ARCHIVE_DIR": "off"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._start(False)

    def tearDown(self) -> None:
        _fixture_module.ReviewServerTests.tearDown(self)

    def _save(self, edits=(), deletes=None, base=None, keys=None, detail=None):
        detail = detail or self._detail()
        body = {"base_sha256": base or detail["base_sha256"],
                "edits": [{"entry": e, "cs": c, "source": "ear", "nudge_cs": 0} for e, c in edits]}
        if deletes is not None:
            rows = {row["entry"]: row for row in detail["rows"]}
            body["deletes"] = [{"entry": e, "line_key": (keys or {}).get(e, rows[e]["line_key"])} for e in deletes]
        status, _headers, data = self._json_request("POST", "/api/song/0/save", body)
        return status, json.loads(data)

    def _log(self):
        return json.loads(review_server.review_artifact_paths(self.report)[1].read_text(encoding="utf-8"))

    def test_single_delete(self) -> None:
        status, result = self._save(deletes=[1])
        self.assertEqual(status, 200)
        self.assertEqual(self.output.read_bytes(), self.original.replace(b"[00:01.00]one\n", b""))
        self.assertEqual([row["lyric"] for row in result["rows"]], ["two a\ntwo b", "three\nfour", "five"])

    def test_multi_line_entry_delete_removes_every_physical_line(self) -> None:
        status, _result = self._save(deletes=[2])
        self.assertEqual(status, 200)
        data = self.output.read_bytes()
        self.assertNotIn(b"two a", data)
        self.assertNotIn(b"two b", data)
        self.assertEqual(data, self.original.replace(b"[00:02.00]two a\n[00:02.00]two b\n", b""))
        entry = self._log()["deleted_lines"][0]
        self.assertEqual((entry["entry"], entry["cs"], entry["lines"]), (2, 200, ["[00:02.00]two a", "[00:02.00]two b"]))

    def test_delete_and_time_edit_in_the_same_save(self) -> None:
        status, result = self._save(edits=[(3, 330)], deletes=[1, 5])
        self.assertEqual(status, 200)
        self.assertEqual(self.output.read_bytes(), (
            b"\xef\xbb\xbf[ar:test]\n[al:album]\n[00:00.00]title\n"
            b"[00:02.00]two a\n[00:02.00]two b\n[00:03.30]three\n[00:03.00]four\n"))
        self.assertEqual([row["current_cs"] for row in result["rows"]], [200, 330, 300])
        log = self._log()
        self.assertEqual([item["entry"] for item in log["edits"]], [3])
        self.assertEqual([item["entry"] for item in log["deleted_lines"]], [1, 5])

    def test_deleted_entry_is_not_a_neighbour_for_collision_handling(self) -> None:
        # entry 3 asks for 03.00, the time of entry 4; with 4 deleted in the same save nothing is shifted.
        status, result = self._save(edits=[(3, 300)], deletes=[4])
        self.assertEqual(status, 200)
        self.assertNotIn("adjusted_cs", result)

    def test_edit_and_delete_of_the_same_entry_is_400(self) -> None:
        before = self.output.read_bytes()
        status, result = self._save(edits=[(2, 250)], deletes=[2])
        self.assertEqual(status, 400, result)
        self.assertEqual(self.output.read_bytes(), before)

    def test_stale_line_key_is_422(self) -> None:
        before = self.output.read_bytes()
        status, result = self._save(deletes=[1], keys={1: "0:" + "0" * 40})
        self.assertEqual(status, 422, result)
        self.assertIn("第 1 句", result["error"])
        self.assertEqual(self.output.read_bytes(), before)

    def test_stale_sha_is_409(self) -> None:
        before = self.output.read_bytes()
        status, result = self._save(deletes=[1], base="0" * 64)
        self.assertEqual((status, result["error"]), (409, "stale base_sha256"))
        self.assertEqual(self.output.read_bytes(), before)

    def test_delete_everything_is_400(self) -> None:
        before = self.output.read_bytes()
        status, result = self._save(deletes=[1, 2, 3, 4, 5])
        self.assertEqual((status, result["error"]), (400, "不能刪光所有句子"))
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(self._save(deletes=[1, 2, 3, 4])[0], 200)   # one timed line left: allowed

    def test_unknown_or_duplicate_delete_is_400(self) -> None:
        detail = self._detail()
        key = detail["rows"][0]["line_key"]
        for deletes in ([{"entry": 9, "line_key": key}], [{"entry": 1, "line_key": key}, {"entry": 1, "line_key": key}], [{"entry": 1}]):
            body = {"base_sha256": detail["base_sha256"], "edits": [], "deletes": deletes}
            self.assertEqual(self._json_request("POST", "/api/song/0/save", body)[0], 400)

    def test_crlf_and_bom_are_preserved(self) -> None:
        self._start(True)
        status, _result = self._save(deletes=[2])
        self.assertEqual(status, 200)
        self.assertEqual(self.output.read_bytes(), (
            b"\xef\xbb\xbf[ar:test]\r\n[al:album]\r\n[00:00.00]title\r\n"
            b"[00:01.00]one\r\n[00:03.00]three\r\n[00:03.00]four\r\n[01:02.00]five\r\n"))
        logged = self._log()["deleted_lines"][0]["lines"]
        self.assertEqual([line.rstrip(chr(13)) for line in logged], ["[00:02.00]two a", "[00:02.00]two b"])
        self.assertTrue(all(line.endswith(chr(13)) for line in logged), "logged text is the exact original line")

    def test_deleted_lines_are_logged_beside_edits(self) -> None:
        self._save(edits=[(1, 150)], deletes=[4])
        log = self._log()
        self.assertEqual(len(log["edits"]), 1)
        deleted = log["deleted_lines"]
        self.assertEqual(len(deleted), 1)
        self.assertEqual((deleted[0]["entry"], deleted[0]["cs"], deleted[0]["lines"], deleted[0]["source"]),
                         (4, 300, ["[00:03.00]four"], "review-page-delete"))
        self.assertTrue(deleted[0]["line_key"] and deleted[0]["utc"])
        self.assertEqual(log["last_lrc_sha256"], deleted[0]["lrc_sha256_after"])

    def test_song_reopens_editable_and_a_later_edit_saves(self) -> None:
        status, result = self._save(deletes=[1, 2])
        self.assertEqual(status, 200)
        self.assertFalse(result["read_only"], result["read_only_reason"])
        self.assertEqual(result["entry_model"], "lrc")
        detail = self._detail()
        self.assertFalse(detail["read_only"], detail["read_only_reason"])
        self.assertEqual([row["lyric"] for row in detail["rows"]], ["three\nfour", "five"])
        status, result = self._save(edits=[(2, 6300)], detail=detail)
        self.assertEqual(status, 200, result)
        self.assertIn(b"[01:03.00]five", self.output.read_bytes())

    def test_old_body_without_deletes_is_unchanged(self) -> None:
        detail = self._detail()
        body = {"base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 125, "source": "ear", "nudge_cs": 25}]}
        self.assertEqual(self._json_request("POST", "/api/song/0/save", body)[0], 200)
        self.assertIn(b"[00:01.25]one", self.output.read_bytes())
        self.assertNotIn("deleted_lines", self._log())
        body = {"base_sha256": self._detail()["base_sha256"], "edits": [], "extra": 1}
        self.assertEqual(self._json_request("POST", "/api/song/0/save", body)[0], 400)

    def test_empty_deletes_list_changes_nothing(self) -> None:
        before = self.output.read_bytes()
        status, _result = self._save(deletes=[])
        self.assertEqual(status, 200)
        self.assertEqual(self.output.read_bytes(), before)

    def _direct_delete_save(self, fail_calls):
        auto, edits_path = review_server.review_artifact_paths(self.report)
        auto.write_bytes(self.output.read_bytes())
        edits_path.write_text(json.dumps({"schema": 1, "auto_lrc_sha256": review_server._sha256_file(auto), "edits": []}), encoding="utf-8")
        detail = self._detail()
        row = next(item for item in detail["rows"] if item["entry"] == 1)
        body = {"base_sha256": detail["base_sha256"], "edits": [], "deletes": [{"entry": 1, "line_key": row["line_key"]}]}
        real_replace = review_server.os.replace
        calls = 0

        def replace(source, destination):
            nonlocal calls
            calls += 1
            if calls in fail_calls:
                raise OSError("injected")
            return real_replace(source, destination)

        with mock.patch.object(review_server.os, "replace", side_effect=replace):
            return review_server._save_song(self.server.store.song(0), body), edits_path

    def test_delete_log_failure_restores_the_lrc(self) -> None:
        before_output, = (self.output.read_bytes(),)
        (status, result), edits_path = self._direct_delete_save({2})
        before_log = edits_path.read_bytes()
        self.assertEqual(status, 500)
        self.assertIs(result["lrc_saved"], False)
        self.assertEqual(self.output.read_bytes(), before_output)
        self.assertEqual(json.loads(before_log).get("deleted_lines"), None)
        self.assertFalse(list(self.root.glob("*.stage")))

    def test_delete_log_and_rollback_failure_returns_text_and_writes_fallback(self) -> None:
        before_output = self.output.read_bytes()
        (status, result), edits_path = self._direct_delete_save({2, 3, 4, 5})
        self.assertEqual(status, 500)
        self.assertIs(result["lrc_saved"], True)
        self.assertNotEqual(self.output.read_bytes(), before_output)
        self.assertEqual(result["deleted_lines"][0]["lines"], ["[00:01.00]one"])
        fallback = edits_path.with_name(edits_path.name[:-len(".review-edits.json")] + ".review-deleted-fallback.jsonl")
        self.assertTrue(fallback.is_file())
        self.assertIn("one", json.loads(fallback.read_text(encoding="utf-8").splitlines()[0])["lines"][0])
        self.assertIn(fallback.name, result["error"])
        self.assertIn("deleted_lines", result)
        self.assertNotIn("deleted_lines", json.loads(edits_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
