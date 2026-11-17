"""An LRC of uncertain encoding (a very short GBK file that also reads as Big5) is never rewritten, replaced or exported by the review server."""
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

UNCERTAIN = "[00:01.00]爱你\n[00:02.00]x\n".encode("gb18030")
CERTAIN = "[00:01.00]愛你\n[00:02.00]x\n".encode("utf-8")


class EncodingRefusalTests(unittest.TestCase):
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

    def _files(self):
        return sorted((path.name, path.read_bytes()) for path in self.root.rglob("*") if path.is_file() and path.suffix != ".json")

    def _post(self, route: str, body: dict):
        status, headers, data = self._json_request("POST", "/api/song/0/" + route, body)
        try:
            data = json.loads(data)
        except ValueError:
            pass
        return status, headers, data

    def _writes(self, detail: dict):
        sha = detail["base_sha256"]
        return {
            "save-time": ("save", {"base_sha256": sha, "edits": [{"entry": 1, "cs": 160, "source": "ear", "nudge_cs": 0}]}),
            "save-delete": ("save", {"base_sha256": sha, "edits": [], "deletes": [{"entry": 2, "line_key": "x"}]}),
            "convert": ("convert", {"base_sha256": sha, "entries": [1]}),
            "export": ("export", {"audio": "song.flac"}),
            "export-check": ("export", {"audio": "", "check": True}),
        }

    def test_every_write_is_refused_and_leaves_nothing_behind(self) -> None:
        detail = self._load(UNCERTAIN)
        self.assertIs(detail["encoding_uncertain"], True)
        before = self._files()
        for name, (route, body) in self._writes(detail).items():
            with self.subTest(name):
                status, _headers, data = self._post(route, body)
                self.assertEqual(status, 409, data)
                self.assertEqual(data["code"], "encoding_uncertain")
                self.assertEqual(data["error"], review_server.ENCODING_UNCERTAIN_MESSAGE)   # a page from before the check shows ``error`` as it is
                self.assertEqual(data["message"], review_server.ENCODING_UNCERTAIN_MESSAGE)
                self.assertEqual(self.output.read_bytes(), UNCERTAIN)
                self.assertEqual(self._files(), before)
        self.assertEqual(list(self.root.glob("*.orig-*")), [])

    def test_the_same_writes_pass_the_encoding_check_for_a_certain_file(self) -> None:
        detail = self._load(CERTAIN)
        self.assertIs(detail["encoding_uncertain"], False)
        for name, (route, body) in self._writes(detail).items():
            with self.subTest(name):
                status, _headers, data = self._post(route, body)
                self.assertNotEqual((status, data.get("code") if isinstance(data, dict) else None), (409, "encoding_uncertain"), data)
        status, _headers, data = self._post("save", {"base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 160, "source": "ear", "nudge_cs": 0}]})
        self.assertIn(status, (200, 403, 409, 422), data)

    def test_utf8_save_still_writes(self) -> None:
        raw = b"[00:01.50]first\n[00:03.00]second\n[00:05.50]third\n[00:07.00]fourth\n"
        detail = self._load(raw)
        status, _headers, data = self._post("save", {"base_sha256": detail["base_sha256"], "edits": [{"entry": 2, "cs": 320, "source": "ear", "nudge_cs": 0}]})
        self.assertEqual(status, 200, data)
        self.assertIn(b"[00:03.20]second", self.output.read_bytes())

    def _romaji_state(self):
        return sorted((str(path.relative_to(self.root)), path.read_bytes()) for path in self.root.rglob("*") if path.is_file() and path.name.endswith(".romaji-overrides.json")),             (self.shared.read_bytes() if self.shared.exists() else None)

    def test_decision_still_works_and_leaves_the_lrc_alone(self) -> None:
        self._load(UNCERTAIN)
        status, _headers, data = self._post("decision", {"entry": 1, "decision": "keep"})
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(), UNCERTAIN)

    def test_romaji_is_refused_and_writes_neither_the_song_file_nor_the_shared_dictionary(self) -> None:
        self.shared = self.root / "shared-reading-overrides.txt"
        self.shared.write_text("# kept\n", encoding="utf-8")
        patcher = mock.patch.dict(os.environ, {"LRC_READING_OVERRIDES": str(self.shared)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._load(UNCERTAIN)
        before, files = self._romaji_state(), self._files()
        bodies = {"line": {"entry": 1, "romaji": "ai"}, "clear": {"entry": 1, "clear": True}, "dictionary": {"entry": 1, "surface": "爱你", "reading": "あいにきて"}}
        for name, body in bodies.items():
            with self.subTest(name):
                status, _headers, data = self._post("romaji", body)
                self.assertEqual((status, data["code"], data["error"]), (409, "encoding_uncertain", review_server.ENCODING_UNCERTAIN_MESSAGE), data)
                self.assertEqual(self._romaji_state(), before)
                self.assertEqual(self._files(), files)
                self.assertEqual(self.output.read_bytes(), UNCERTAIN)
        self.assertEqual(before[0], [])

    def test_romaji_is_written_for_a_certain_file(self) -> None:
        self.shared = self.root / "shared-reading-overrides.txt"
        patcher = mock.patch.dict(os.environ, {"LRC_READING_OVERRIDES": str(self.shared)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._load(CERTAIN)
        status, _headers, data = self._post("romaji", {"entry": 1, "romaji": "ai"})
        self.assertEqual(status, 200, data)
        self.assertEqual(len(self._romaji_state()[0]), 1)

    def test_a_page_from_before_the_check_shows_the_whole_sentence(self) -> None:
        """api() of the page at 956b3514 (line 232) shows ``data.error``; fed this server's real 409 body it shows the sentence."""
        detail = self._load(UNCERTAIN)
        status, _headers, body = self._post("convert", {"base_sha256": detail["base_sha256"], "entries": [1]})
        self.assertEqual(status, 409)
        harness = """
  const api = async () => {
    const response = {ok:false, status:409, statusText:'Conflict', json: async () => (BODY)};
    const data = await response.json().catch(() => ({error:response.statusText}));
    if (!response.ok) throw Error(response.status === 403 ? 'forbidden' : (data.error || response.statusText));
  };
  api().catch(error => console.log(JSON.stringify({convert: '轉換失敗：' + error.message, plain: error.message})));
""".replace("BODY", json.dumps(body))
        result = subprocess.run(["node", "-"], input=harness, capture_output=True, text=True, encoding="utf-8", creationflags=0x08000000 if os.name == "nt" else 0)
        self.assertEqual(result.returncode, 0, result.stderr)
        shown = json.loads(result.stdout)
        self.assertEqual(shown["plain"], review_server.ENCODING_UNCERTAIN_MESSAGE)
        self.assertNotIn("encoding_uncertain", shown["convert"])

    def test_a_file_replaced_after_listing_is_judged_as_it_is_now(self) -> None:
        detail = self._load(CERTAIN)
        self.output.write_bytes(UNCERTAIN)   # replaced behind the store's back: its cached flag still says certain
        status, _headers, data = self._post("save", {"base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 160, "source": "ear", "nudge_cs": 0}]})
        self.assertEqual((status, data["code"]), (409, "encoding_uncertain"))
        self.assertEqual(self.output.read_bytes(), UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
