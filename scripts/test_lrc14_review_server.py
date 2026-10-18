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

    def _start(self, crlf: bool, ambiguous: bool = False) -> None:
        if getattr(self, "server", None):
            self.server.shutdown()
            self.thread.join(timeout=3)
            self.server.server_close()
        if getattr(self, "temp", None):
            self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "song.lrc"
        self.audio = self.root / "song.flac"
        self.stem = self.root / "song-stem.wav"
        newline = b"\r\n" if crlf else b"\n"
        lines = [
            b"[ar:test]", b"[al:album]", b"[00:00.00]title",
            b"[00:01.00]one", b"[00:02.00]two a", b"[00:02.00]two b",
            b"[00:03.00]three",
        ]
        if ambiguous:
            lines.extend([b"[00:03.00]three extra"])
        lines.extend([b"[00:03.00]four", b"[01:02.00]five", b""])
        self.original = b"\xef\xbb\xbf" + newline.join(lines)
        self.output.write_bytes(self.original)
        self.audio.write_bytes(b"0123456789")
        self.stem.write_bytes(b"stem-audio")
        assignments = [
            {"entry": 1, "lrc_tag": "[00:01.00]", "review_required": True,
             "verification_status": "review-required", "candidates": [{"time": 1.5, "source": "ctc", "score": .9}]},
            {"entry": 2, "lrc_tag": "[00:02.00]", "review_required": True,
             "verification_status": "review-required", "candidates": [{"time": 2.4, "source": "whisper", "score": .8}]},
            {"entry": 3, "lrc_tag": "[00:03.00]", "review_required": True,
             "verification_status": "review-required", "candidates": [{"time": 3.2, "source": "ctc", "score": .7}]},
            {"entry": 4, "lrc_tag": "[00:03.00]", "review_required": True,
             "verification_status": "review-required", "candidates": [{"time": 3.5, "source": "whisper", "score": .6}]},
            {"entry": 5, "lrc_tag": "[01:02.00]", "review_required": True,
             "verification_status": "review-required", "candidates": [{"time": 62.2, "source": "ctc", "score": .5}]},
        ]
        self.report = self.root / "song.align-report.json"
        report = {
            "audio_path": str(self.audio), "ctc_audio_path": str(self.stem),
            "output_path": str(self.output), "display_lines": 7 if ambiguous else 6,
            "timing_entries": 5, "duration_seconds": 100.0,
            "published_lrc_sha256": hashlib.sha256(self.original).hexdigest(),
            "assignments": assignments,
        }
        self.report.write_text(json.dumps(report), encoding="utf-8")
        self.store = review_server.ReviewStore(self.root, [self.report])
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
        return self._request(method, path, body, **{
            "X-Review-Token": self.server.token,
            "Origin": f"http://{self.base}",
            "Content-Type": "application/json",
            **headers,
        })

    def _detail(self):
        return json.loads(self._request(
            "GET", "/api/song/0", **{"X-Review-Token": self.server.token}
        )[2])

    def test_selector_exposes_precentral_recommendation(self) -> None:
        candidates = review_server._candidate_rows(
            {
                "candidates": [
                    {"time": 36.42, "source": "ctc-current", "score": .86},
                    {"time": 37.01, "source": "ctc-nearby", "score": .90},
                ],
                "precentral_candidate_recommendation": {
                    "time": 37.0, "source": "raw_asr", "score": .98,
                },
            },
            3642,
            3642,
        )
        self.assertIn(3700, [item["cs"] for item in candidates])
        diagnostic = next(item for item in candidates if item["cs"] == 3700)
        self.assertTrue(diagnostic["diagnostic_only"])
        self.assertIn("raw_asr", diagnostic["sources"])
        self.assertIn("ctc-nearby", diagnostic["sources"])

    def test_mapping_payload_and_adjacent_lines(self) -> None:
        detail = self._detail()
        self.assertFalse(detail["read_only"])
        self.assertEqual(len(detail["rows"]), 5)
        self.assertEqual(detail["rows"][1]["lyrics"], ["two a", "two b"])
        self.assertEqual(detail["rows"][2]["previous"]["cs"], 200)
        self.assertEqual(detail["rows"][2]["next"]["lyrics"], ["four"])
        self.assertEqual(detail["rows"][4]["current_cs"], 6200)
        self.assertIn("base_sha256", detail)

    def test_entry_key_survives_non_monotonic_timestamps_and_tsv_order(self) -> None:
        output = self.root / "nonmonotonic.lrc"
        output.write_bytes(b"[00:20.00]one\n[00:10.00]two\n[00:30.00]three\n")
        report = self.root / "nonmonotonic.align-report.json"
        report.write_text(json.dumps({
            "audio_path": str(self.audio),
            "ctc_audio_path": str(self.stem),
            "output_path": str(output),
            "display_lines": 3,
            "timing_entries": 3,
            "assignments": [
                {"entry": 1, "lrc_tag": "[00:20.00]", "review_required": True, "verification_status": "review-required"},
                {"entry": 2, "lrc_tag": "[00:10.00]", "review_required": True, "verification_status": "review-required"},
                {"entry": 3, "lrc_tag": "[00:30.00]", "review_required": False, "verification_status": "unverified"},
            ],
        }), encoding="utf-8")
        tsv = report.with_name(report.name.replace(".align-report.json", ".review.tsv"))
        tsv.write_text(
            "\ufeff# review_rows=2\n"
            "entry\ttime\tlyric\tstatus\treason_codes\tseverity_rank\tseverity_score\tseverity_reasons\ttimestamp_status\tlrc_time\tprev_time\tnext_time\tcandidate_times\tcandidate_spread_s\n"
            "2\t00:10.00\ttwo\tREVIEW\tflag\t1\t9\treview\t\t00:10.00\t\t\t\t\n"
            "1\t00:20.00\tone\tREVIEW\tflag\t2\t3\treview\t\t00:20.00\t\t\t\t\n"
            "3\t00:30.00\tthree\tUNTRUSTED\tunverified\t3\t0\tuntrusted\t\t00:30.00\t\t\t\t\n",
            encoding="utf-8",
        )
        store = review_server.ReviewStore(self.root, [report])
        rows = store.song(0)["rows"]
        self.assertEqual(
            [(row["entry"], row["lyric"], row["review"]) for row in rows],
            [(1, "one", "REVIEW"), (2, "two", "REVIEW"), (3, "three", "UNTRUSTED")],
        )

    def test_page_keeps_lyric_order_when_severity_differs(self) -> None:
        page = (SCRIPTS / "review_page" / "index.html").read_text(encoding="utf-8")
        self.assertIn("severity-badge", page)
        self.assertIn("a.song.index - b.song.index || a.row.entry - b.row.entry", page)
        self.assertIn("candidate-play", page)
        self.assertIn("candidate-apply", page)
        self.assertIn("stage(item, candidate.cs", page)
        self.assertIn("CTC first-sound candidate", page)
        self.assertIn("ASR diagnostic hint", page)
        self.assertIn("diagnostic only", page)
        self.assertIn("color-scheme:dark", page)
        source = page.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        harness = r"""
const vm = require('vm');
const source = SOURCE;
const elements = new Map();
function element(id='') {
  return {id, children:[], dataset:{}, className:'', textContent:'', disabled:false,
    addEventListener(type, fn) { this['on' + type] = fn; },
    appendChild(child) { this.children.push(child); },
    querySelectorAll() { return []; }, blur() {},
    set innerHTML(value) { this._html = value; this.children = []; },
    get innerHTML() { return this._html || ''; }}; }
global.document = {
  getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
  createElement() { return element(); },
  addEventListener() {},
};
global.location = {search:'?t=test', hash:'', origin:'http://test'};
global.addEventListener = () => {};
const response = (data) => ({ok:true, status:200, statusText:'ok', json:async () => data});
const rows = [
  {entry:1, review:'REVIEW', current_cs:2000, auto_cs:2000, lyrics:['one'], candidates:[], severity_score:1, severity_rank:2, candidate_spread:null},
  {entry:2, review:'REVIEW', current_cs:1000, auto_cs:1000, lyrics:['two'], candidates:[], severity_score:9, severity_rank:1, candidate_spread:null},
];
global.fetch = async (path) => {
  if (path === '/api/songs') return response([{index:0, song:'song', read_only:false, has_flac:false, has_stem:false}]);
  if (path === '/api/song/0') return response({base_sha256:'x', server_read_only:false, has_flac:true, has_stem:true, rows});
  throw Error('unexpected fetch ' + path);
};
const settle = async () => { for (let i=0; i<12; i++) await new Promise(resolve => setImmediate(resolve)); };
(async () => {
  vm.runInThisContext(source);
  await settle();
  const queue = elements.get('queue');
  console.log(JSON.stringify({
    tabs: queue.children[0].children.map(child => child.textContent),
    keys: queue.children[1].children.map(child => child.dataset.key),
    audio: queue.children[1].children[1]._html.includes('data-kind="flac" >play FLAC'),
    nudge: queue.children[1].children[0]._html.includes('data-delta="-10"')
  }));
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
""".replace("SOURCE", json.dumps(source))
        result = subprocess.run(["node", "-"], input=harness, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"tabs": ["song (2)"], "keys": ["0:1", "0:2"], "audio": True, "nudge": True})

    def test_acceptance_save_three_entries_and_second_save(self) -> None:
        detail = self._detail()
        first = {
            "base_sha256": detail["base_sha256"],
            "edits": [
                {"entry": 2, "cs": 250, "source": "ctc", "nudge_cs": 50},
                {"entry": 4, "cs": 350, "source": "whisper", "nudge_cs": 50},
                {"entry": 5, "cs": 6205, "source": "manual", "nudge_cs": 5},
            ],
        }
        self.assertEqual(self._json_request("POST", "/api/song/0/save", first)[0], 200)
        expected = (
            b"\xef\xbb\xbf[ar:test]\n[al:album]\n[00:00.00]title\n"
            b"[00:01.00]one\n[00:02.50]two a\n[00:02.50]two b\n"
            b"[00:03.00]three\n[00:03.50]four\n[01:02.05]five\n"
        )
        self.assertEqual(self.output.read_bytes(), expected)
        auto, edits = review_server.review_artifact_paths(self.report)
        self.assertEqual(auto.read_bytes(), self.original)
        self.assertEqual(len(json.loads(edits.read_text())["edits"]), 3)
        second_detail = self._detail()
        second = {
            "base_sha256": second_detail["base_sha256"],
            "edits": [{"entry": 1, "cs": 110, "source": "manual", "nudge_cs": 10}],
        }
        self.assertEqual(self._json_request("POST", "/api/song/0/save", second)[0], 200)
        self.assertEqual(auto.read_bytes(), self.original)
        self.assertEqual(len(json.loads(edits.read_text())["edits"]), 4)
        self.assertEqual(self.output.read_bytes().splitlines()[3], b"[00:01.10]one")

    def test_api_auth_security_and_read_only(self) -> None:
        self.assertEqual(self._request("GET", "/api/songs")[0], 403)
        self.assertEqual(self._request("GET", "/api/songs", **{"X-Review-Token": "wrong"})[0], 403)
        self.assertEqual(self._request("GET", "/api/songs", **{"Host": "localhost"})[0], 403)
        self.assertEqual(self._request("GET", "/api/song/99", **{"X-Review-Token": self.server.token})[0], 404)
        self.assertEqual(self._request("GET", "/audio/0/../x?t=test-token")[0], 404)
        detail = self._detail()
        body = {"base_sha256": detail["base_sha256"], "edits": [], "path": "x"}
        self.assertEqual(self._json_request("POST", "/api/song/0/save", body)[0], 400)
        self.assertEqual(self._json_request("POST", "/api/song/0/save", {"base_sha256": detail["base_sha256"], "edits": []}, Origin="http://evil")[0], 403)
        self.server.store.read_only = True
        self.assertTrue(self._detail()["server_read_only"])
        self.assertEqual(self._json_request("POST", "/api/song/0/save", {"base_sha256": detail["base_sha256"], "edits": []})[0], 403)

    def test_audio_ranges_missing_stem_and_head(self) -> None:
        status, headers, data = self._request("GET", f"/audio/0/flac?t={self.server.token}", Range="bytes=2-5")
        self.assertEqual((status, data, headers["Content-Range"]), (206, b"2345", "bytes 2-5/10"))
        status, headers, data = self._request("GET", f"/audio/0/flac?t={self.server.token}", Range="bytes=2-99")
        self.assertEqual((status, data, headers["Content-Range"]), (206, b"23456789", "bytes 2-9/10"))
        status, headers, data = self._request("GET", f"/audio/0/flac?t={self.server.token}", Range="bytes=-3")
        self.assertEqual((status, data, headers["Content-Range"]), (206, b"789", "bytes 7-9/10"))
        self.assertEqual(self._request("GET", f"/audio/0/flac?t={self.server.token}", Range="bytes=99-")[0], 416)
        status, headers, data = self._request("HEAD", f"/audio/0/flac?t={self.server.token}", Range="bytes=1-3")
        self.assertEqual((status, data, headers["Content-Length"]), (206, b"", "3"))
        self.stem.unlink()
        self.assertEqual(self._request("GET", f"/audio/0/stem?t={self.server.token}")[0], 404)
        self.assertEqual(self._request("GET", f"/audio/0/flac?t={self.server.token}")[0], 200)

    def test_ambiguous_dp_is_read_only(self) -> None:
        self._start(False, ambiguous=True)
        detail = self._detail()
        self.assertTrue(detail["read_only"])
        self.assertEqual(detail["read_only_reason"], "ambiguous-timing-region")

    def test_outside_edit_after_baseline_is_read_only(self) -> None:
        detail = self._detail()
        payload = {"base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 110, "source": "x", "nudge_cs": 10}]}
        self.assertEqual(self._json_request("POST", "/api/song/0/save", payload)[0], 200)
        self.output.write_bytes(self.output.read_bytes().replace(b"two a", b"two X"))
        current = self.server.store.song(0)
        self.assertTrue(current["read_only"])
        self.assertEqual(current["read_only_reason"], "lrc-lyric-content-changed")

    def test_lock_leaves_bytes_log_and_stage_unchanged(self) -> None:
        auto, edits = review_server.review_artifact_paths(self.report)
        auto.write_bytes(self.original)
        edits.write_text(json.dumps({"schema": 1, "auto_lrc_sha256": review_server._sha256_file(auto), "edits": []}), encoding="utf-8")
        before_log = edits.read_bytes()
        song = self.server.store.song(0)
        body = {"base_sha256": hashlib.sha256(self.original).hexdigest(), "edits": [{"entry": 1, "cs": 150, "source": "ctc", "nudge_cs": 50}]}
        with mock.patch.object(review_server.os, "replace", side_effect=PermissionError):
            status, result = review_server._save_song(song, body)
        self.assertEqual(status, 423)
        self.assertEqual(self.output.read_bytes(), self.original)
        self.assertEqual(edits.read_bytes(), before_log)
        self.assertFalse(list(self.root.glob("*.stage")))

    def test_dump_summary_is_non_destructive(self) -> None:
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.root.rglob("*") if path.is_file()}
        result = subprocess.run([sys.executable, str(SCRIPTS / "review_server.py"), "--dump-summary", str(self.report)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 1)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["rows"], 5)
        after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_bk_exclusion_and_public_api_paths(self) -> None:
        bk = self.root / "lrc bk"
        bk.mkdir()
        report = bk / "bad.align-report.json"
        data = json.loads(self.report.read_text())
        data["output_path"] = str(bk / "candidate.lrc")
        data["audio_path"] = str(bk / "bad.flac")
        data["ctc_audio_path"] = str(bk / "bad.wav")
        (bk / "candidate.lrc").write_bytes(self.original)
        (bk / "bad.flac").write_bytes(b"x")
        (bk / "bad.wav").write_bytes(b"x")
        report.write_text(json.dumps(data), encoding="utf-8")
        store = review_server.ReviewStore(self.root, [self.root])
        self.assertEqual(len(store.songs), 1)
        payload = json.loads(self._request("GET", "/api/songs", **{"X-Review-Token": self.server.token})[2])
        self.assertNotIn(str(self.root), json.dumps(payload))

    def test_isolation_and_page_contract(self) -> None:
        code = "import sys;sys.path.insert(0,%r);import review_server;print('numpy' in sys.modules,'auto_lrc' in sys.modules)" % str(SCRIPTS)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False False")
        source = (SCRIPTS / "review_server.py").read_text(encoding="utf-8").splitlines()
        self.assertFalse(any("checked_lrc_candidates" in line or "checked-lrc" in line for line in source))
        hits = [line for line in source if "checked references" in line.casefold() or "lrc bk" in line.casefold()]
        self.assertEqual(len(hits), 1)
        page = (SCRIPTS / "review_page" / "index.html").read_text(encoding="utf-8")
        for marker in ("currentTime = start", "loadedmetadata", "BracketLeft", "base_sha256:song.detail.base_sha256", "beforeunload", "server_read_only", "response.status === 409", "reloaded and kept staged edits",
                       "base_cs:item.row.current_cs", "candidate.cs, sourceText(candidate), 0", "actualDelta = cs - before",
                       "dropped staged edit", "edit.base_cs !== item.row.current_cs", "button.blur()",
                       "event.target?.type !== 'checkbox'", "const conflict = conflicted(item)",
                       "const loadMessage = await load()", "alert('This timestamp changed after staging"):
            self.assertIn(marker, page)
        self.assertIn("const start = targetCs / 100;", page)
        self.assertIn("const stop = start + 2;", page)
        self.assertNotIn("targetCs / 100 - 2", page)
        self.assertIn("className = 'song-tabs'", page)
        self.assertIn("state.activeSong", page)
        self.assertIn("song.song + ' (' + pending + ')'", page)
        self.assertIn('class="nudge-button"', page)
        self.assertIn('data-delta="-10"', page)
        self.assertIn('data-delta="10"', page)
        self.assertIn("nudge(item, Number(button.dataset.delta))", page)

    def test_bom_crlf_save_bytes_and_stale_save(self) -> None:
        self._start(True)
        detail = self._detail()
        payload = {"base_sha256": detail["base_sha256"], "edits": [{"entry": 2, "cs": 250, "source": "ctc", "nudge_cs": 50}]}
        self.assertEqual(self._json_request("POST", "/api/song/0/save", payload)[0], 200)
        expected = (
            b"\xef\xbb\xbf[ar:test]\r\n[al:album]\r\n[00:00.00]title\r\n"
            b"[00:01.00]one\r\n[00:02.50]two a\r\n[00:02.50]two b\r\n"
            b"[00:03.00]three\r\n[00:03.00]four\r\n[01:02.00]five\r\n"
        )
        self.assertEqual(self.output.read_bytes(), expected)
        _auto, edits = review_server.review_artifact_paths(self.report)
        before_log = edits.read_bytes()
        self.assertEqual(self._json_request("POST", "/api/song/0/save", payload)[0], 409)
        self.assertEqual(self.output.read_bytes(), expected)
        self.assertEqual(edits.read_bytes(), before_log)

    def test_stale_api_save_returns_409_with_changed_base_timestamp(self) -> None:
        stale = self._detail()
        fresh = {
            "base_sha256": stale["base_sha256"],
            "edits": [{"entry": 1, "cs": 125, "source": "other-reviewer", "nudge_cs": 25}],
        }
        self.assertEqual(self._json_request("POST", "/api/song/0/save", fresh)[0], 200)
        stale_edit = {
            "base_sha256": stale["base_sha256"],
            "edits": [{"entry": 1, "cs": 110, "source": "manual", "nudge_cs": 10}],
        }
        status, _headers, body = self._json_request("POST", "/api/song/0/save", stale_edit)
        self.assertEqual((status, json.loads(body)["error"]), (409, "stale base_sha256"))

    def test_client_409_reload_marks_conflict_and_blocks_resubmit(self) -> None:
        page = (SCRIPTS / "review_page" / "index.html").read_text(encoding="utf-8")
        source = page.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        harness = r"""
const vm = require('vm');
const source = SOURCE;
const elements = new Map(), documentListeners = {};
let lastRow = null, serverCs = 100, postCount = 0;
function element(id='') {
  return {id, children:[], dataset:{}, className:'', textContent:'', disabled:false,
    addEventListener(type, fn) { this['on' + type] = fn; },
    appendChild(child) { this.children.push(child); if (String(child.className).includes('row ')) lastRow = child; },
    querySelectorAll() { return []; }, blur() {},
    set innerHTML(value) { this._html = value; this.children = []; }, get innerHTML() { return this._html || ''; }}; }
global.document = {
  getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
  createElement() { return element(); },
  addEventListener(type, fn) { documentListeners[type] = fn; },
};
global.location = {search:'?t=test', hash:'', origin:'http://test'};
global.addEventListener = () => {};
const response = (ok, status, data) => ({ok, status, statusText:'error', json:async () => data});
function detail() { return {base_sha256:serverCs === 100 ? 'old' : 'new', server_read_only:false, rows:[{
  entry:1, review:'REVIEW', current_cs:serverCs, candidates:[], lyrics:['one'], severity_score:1
}]}; }
global.fetch = async (path, options={}) => {
  if (path === '/api/songs') return response(true, 200, [{index:0, song:'song', read_only:false, has_flac:false, has_stem:false}]);
  if (path === '/api/song/0' && !options.method) return response(true, 200, detail());
  if (path === '/api/song/0/save') { postCount++; serverCs = 125; return response(false, 409, {error:'stale base_sha256'}); }
  throw Error('unexpected fetch ' + path);
};
const settle = async () => { for (let i=0; i<12; i++) await new Promise(resolve => setImmediate(resolve)); };
(async () => {
  vm.runInThisContext(source);
  await settle();
  const event = extra => Object.assign({ctrlKey:false, altKey:false, metaKey:false, shiftKey:false,
    key:'', code:'', target:null, preventDefault(){}}, extra);
  documentListeners.keydown(event({key:']', code:'BracketRight'}));
  documentListeners.keydown(event({ctrlKey:true, key:'s'}));
  await settle();
  const marked = !!lastRow && lastRow.className.includes('conflict');
  documentListeners.keydown(event({ctrlKey:true, key:'s'}));
  await settle();
  console.log(JSON.stringify({marked, postCount, serverCs}));
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
""".replace("SOURCE", json.dumps(source))
        result = subprocess.run(["node", "-"], input=harness, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"marked": True, "postCount": 1, "serverCs": 125})

    def test_missing_lrc_rounding_and_non_ascii_token(self) -> None:
        self.assertEqual(review_server._tag_cs(1.005), 101)
        self.assertEqual(review_server._seconds_cs(1.005), 101)
        self.assertFalse(review_server._safe_compare("é", self.server.token))
        data = json.loads(self.report.read_text())
        data.pop("output_path")
        missing = self.root / "missing.align-report.json"
        missing.write_text(json.dumps(data), encoding="utf-8")
        store = review_server.ReviewStore(self.root, [missing])
        song = store.song(0)
        self.assertEqual(song["read_only_reason"], "lrc-missing")
        status, _result = review_server._save_song(song, {"base_sha256": "0" * 64, "edits": []})
        self.assertEqual(status, 403)
        self.assertFalse((self.root / "candidate.lrc").exists())

    def test_malformed_audio_report_is_excluded(self) -> None:
        data = json.loads(self.report.read_text())
        data["audio_path"] = None
        malformed = self.root / "malformed.align-report.json"
        malformed.write_text(json.dumps(data), encoding="utf-8")
        store = review_server.ReviewStore(self.root, [malformed])
        self.assertEqual(store.songs, [])
        self.assertIn(malformed.resolve(), store.excluded)

    def test_non_object_reports_are_excluded_at_every_discovery_site(self) -> None:
        bad = self.root / "bad.align-report.json"
        bad.write_text("[]", encoding="utf-8")
        for target in (bad, self.audio, self.root / "empty"):
            if target.name == "empty":
                target.mkdir()
            reports, excluded = review_server._report_files([target], self.root)
            self.assertNotIn(bad.name, {path.name for path in reports})
            self.assertIn(bad.name, {path.name for path in excluded})

    def test_startup_lists_excluded_report_names(self) -> None:
        bad = self.root / "bad.align-report.json"
        bad.write_text("null", encoding="utf-8")
        fake_server = mock.Mock()
        fake_server.server_port = 1234
        with mock.patch.object(review_server.http.server, "ThreadingHTTPServer", return_value=fake_server), \
             mock.patch("builtins.print") as output:
            review_server.run_server([bad], self.root, no_browser=True)
        startup = output.call_args_list[-1].args[0]
        self.assertEqual(json.loads(startup)["excluded_files"], [bad.name])


if __name__ == "__main__":
    unittest.main()
