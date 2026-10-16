from __future__ import annotations

import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import review_server


class ReviewServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._start(False)

    def tearDown(self) -> None:
        if getattr(self, "server", None):
            self.server.shutdown()
            self.thread.join(timeout=3)
            self.server.server_close()
        if getattr(self, "temp", None):
            self.temp.cleanup()

    def _start(self, crlf: bool) -> None:
        if getattr(self, "server", None):
            self.server.shutdown()
            self.thread.join(timeout=3)
            self.server.server_close()
        if getattr(self, "temp", None):
            self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.output = root / "song.lrc"
        self.audio = root / "song.flac"
        self.stem = root / "song-stem.wav"
        newline = b"\r\n" if crlf else b"\n"
        self.original = b"\xef\xbb\xbf" + newline.join([
            b"[ar:test]", b"[00:01.00]one", b"[00:02.00]two", b"[00:03.00]three", b""
        ])
        self.output.write_bytes(self.original)
        self.audio.write_bytes(b"0123456789")
        self.stem.write_bytes(b"stem-audio")
        report = {
            "audio_path": str(self.audio), "ctc_audio_path": str(self.stem),
            "output_path": str(self.output), "display_lines": 3,
            "duration_seconds": 5.0,
            "published_lrc_sha256": hashlib.sha256(self.original).hexdigest(),
            "assignments": [
                {"entry": 1, "lrc_tag": "[00:01.00]", "review_required": True,
                 "verification_status": "review", "candidates": [{"time": 1.5, "source": "ctc", "score": .9}],
                 "ctc_first_token_start": 1.4},
                {"entry": 2, "lrc_tag": "[00:02.00]", "review_required": False,
                 "verification_status": "trusted", "candidates": [{"time": 2.4, "source": "whisper", "score": .8}]},
                {"entry": 3, "lrc_tag": "[00:03.00]", "review_required": False,
                 "verification_status": "trusted", "candidates": []},
            ],
        }
        self.report = root / "song.align-report.json"
        self.report.write_text(json.dumps(report), encoding="utf-8")
        self.store = review_server.ReviewStore(root, [self.report])
        self.server = review_server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), review_server.ReviewHandler)
        self.server.store = self.store
        self.server.token = "test-token"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"127.0.0.1:{self.server.server_port}"

    def _request(self, method: str, path: str, body: bytes | None = None, **headers):
        connection = http.client.HTTPConnection(self.server.server_address[0], self.server.server_port, timeout=5)
        connection.request(method, path, body=body, headers={"Host": self.base, **headers})
        response = connection.getresponse()
        data = response.read()
        result = response.status, dict(response.getheaders()), data
        connection.close()
        return result

    def _json_request(self, method: str, path: str, payload: object, **headers):
        body = json.dumps(payload).encode()
        return self._request(method, path, body, **{"X-Review-Token": self.server.token,
            "Origin": f"http://{self.base}", "Content-Type": "application/json", **headers})

    def test_api_auth_and_headers(self) -> None:
        status, headers, _ = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(self._request("GET", "/api/songs")[0], 403)
        status, headers, data = self._request("GET", "/api/songs", **{"X-Review-Token": self.server.token})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(json.loads(data)[0]["rows"], 3)
        self.assertEqual(self._request("GET", "/api/songs", **{"Host": "localhost"})[0], 403)

    def test_audio_ranges_and_head(self) -> None:
        status, headers, data = self._request("GET", f"/audio/0/flac?t={self.server.token}", Range="bytes=1-3")
        self.assertEqual((status, data), (206, b"123"))
        self.assertEqual(headers["Content-Range"], "bytes 1-3/10")
        self.assertEqual(headers["Content-Length"], "3")
        for value in ("bytes=1-2,4-5", "bytes=4-2"):
            self.assertEqual(self._request("GET", f"/audio/0/flac?t={self.server.token}", Range=value)[0], 200)
        self.assertEqual(self._request("GET", f"/audio/0/flac?t={self.server.token}", Range="bytes=20-")[0], 416)
        self.assertEqual(self._request("GET", f"/audio/0/flac?t={self.server.token}", Range="bytes=-0")[0], 416)
        status, headers, data = self._request("HEAD", f"/audio/0/flac?t={self.server.token}", Range="bytes=1-3")
        self.assertEqual((status, data, headers["Content-Length"]), (206, b"", "3"))

    def test_save_preserves_bom_and_lf(self) -> None:
        detail = json.loads(self._request("GET", f"/api/song/0", **{"X-Review-Token": self.server.token})[2])
        payload = {"base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 150, "source": "ctc", "nudge_cs": 50}]}
        self.assertEqual(self._json_request("POST", "/api/song/0/save", payload)[0], 200)
        self.assertEqual(self.output.read_bytes(), b"\xef\xbb\xbf[ar:test]\n[00:01.50]one\n[00:02.00]two\n[00:03.00]three\n")
        self.assertTrue(self.report.with_name("song.auto.lrc").is_file())
        self.assertEqual(self._json_request("POST", "/api/song/0/save", payload)[0], 409)

    def test_save_preserves_bom_and_crlf(self) -> None:
        self._start(True)
        detail = json.loads(self._request("GET", "/api/song/0", **{"X-Review-Token": self.server.token})[2])
        payload = {"base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 150, "source": "ctc", "nudge_cs": 50}]}
        self.assertEqual(self._json_request("POST", "/api/song/0/save", payload)[0], 200)
        self.assertEqual(self.output.read_bytes(), b"\xef\xbb\xbf[ar:test]\r\n[00:01.50]one\r\n[00:02.00]two\r\n[00:03.00]three\r\n")

    def test_lock_returns_423_without_replacing_output(self) -> None:
        report_path = self.report
        auto, edits = review_server.review_artifact_paths(report_path)
        auto.write_bytes(self.original)
        edits.write_text(json.dumps({"schema": 1, "auto_lrc_sha256": review_server._sha256_file(auto), "edits": []}), encoding="utf-8")
        song = self.store.song(0)
        body = {"base_sha256": hashlib.sha256(self.original).hexdigest(), "edits": [{"entry": 1, "cs": 150, "source": "ctc", "nudge_cs": 50}]}
        with mock.patch.object(review_server.os, "replace", side_effect=PermissionError):
            status, result = review_server._save_song(song, body)
        self.assertEqual(status, 423)
        self.assertEqual(self.output.read_bytes(), self.original)
        self.assertIn("locked (close the player)", result["error"])

    def test_summary_is_ascii_safe_and_import_isolated(self) -> None:
        code = "import sys;sys.path.insert(0,%r);import review_server;print('numpy' in sys.modules,'auto_lrc' in sys.modules)" % str(SCRIPTS)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False False")
        summary = subprocess.run([sys.executable, str(SCRIPTS / "review_server.py"), "--dump-summary", str(self.report)], capture_output=True, text=True)
        self.assertEqual(summary.returncode, 0, summary.stderr)
        self.assertEqual(len(summary.stdout.splitlines()), 1)
        self.assertEqual(json.loads(summary.stdout)["rows"], 3)


if __name__ == "__main__":
    unittest.main()
