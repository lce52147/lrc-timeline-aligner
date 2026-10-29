from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import threading
import time
import unittest
import wave
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import review_server
import review_presentation
import lrc_review_tsv


class ReviewServerTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"LRC_EXPORT_ARCHIVE_DIR": "off"})  # never move LRCs into the repo's checked/
        patcher.start()
        self.addCleanup(patcher.stop)
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

    def test_reason_codes_normalize_dicts_scalars_and_fallbacks(self) -> None:
        self.assertEqual(
            lrc_review_tsv.reason_codes_for(
                {"final_trust_reason_codes": [{"code": "unknown-code"}, {"reason": "plain"}, 7]},
                "REVIEW",
            ),
            ["unknown-code", "plain", "7"],
        )
        self.assertEqual(
            lrc_review_tsv.reason_codes_for(
                {"final_trust_reason_codes": [], "flags": ["candidate_disagreement"]},
                "REVIEW",
            ),
            ["review-required"],
        )
        self.assertEqual(lrc_review_tsv.reason_codes_for({"flags": []}, "REVIEW"), ["review-required"])
        self.assertEqual(lrc_review_tsv.reason_codes_for(None, "UNTRUSTED"), ["unverified"])
        self.assertEqual(lrc_review_tsv.reason_codes_for({"reason_codes": "legacy-a, legacy-b"}, "REVIEW"), ["legacy-a", "legacy-b"])

    def test_review_tsv_empty_final_codes_use_status_fallback(self) -> None:
        path = self.root / "empty-final.review.tsv"
        report = {
            "assignments": [{
                "entry": 1,
                "timestamp": 1.0,
                "review_required": True,
                "verification_status": "review-required",
                "final_trust_reason_codes": [],
                "flags": ["candidate_disagreement"],
            }],
            "lyric_preflight": {"status": "OK"},
        }
        lrc_review_tsv.write_review_tsv(path, report, [SimpleNamespace(lines=["line"])])
        self.assertEqual(path.read_text(encoding="utf-8").splitlines()[-1].split("\t")[4], "review-required")

    def test_reason_code_union_is_mapped_and_unknowns_are_safe(self) -> None:
        fixture = json.loads((SCRIPTS / "test_fixtures" / "reason_codes_b1_39.json").read_text(encoding="utf-8"))
        page = (SCRIPTS / "review_page" / "index.html").read_text(encoding="utf-8")
        for code in fixture["codes"]:
            self.assertIn("'" + code + "'", page)
        self.assertIn("需要人工確認（", page)
        self.assertIn("JSON.stringify(value)", page)
        self.assertIn("未標示原因", page)

    def test_raw_assignment_reasons_survive_without_review_tsv(self) -> None:
        data = json.loads(self.report.read_text(encoding="utf-8"))
        data["assignments"][0].pop("final_trust_reason_codes", None)
        data["assignments"][0]["flags"] = [{"code": "candidate_disagreement"}]
        self.assertFalse(self.report.with_name("song.review.tsv").exists())
        detail = review_server._song_payload(self.report, data)
        row = detail["rows"][0]
        self.assertEqual(row["reason_codes"], ["candidate_disagreement"])
        self.assertEqual(row["reason"], "candidate_disagreement")

    def test_tsv_rows_use_assignment_reasons_without_stringifying_multiple_codes(self) -> None:
        data = json.loads(self.report.read_text(encoding="utf-8"))
        data["assignments"][0]["final_trust_reason_codes"] = ["candidate_disagreement", "low_confidence"]
        tsv = self.report.with_name("song.review.tsv")
        tsv.write_text(
            "# review_rows=1\n"
            "entry\ttime\tlyric\tstatus\treason_codes\tseverity_rank\tseverity_score\tseverity_reasons\ttimestamp_status\tlrc_time\tprev_time\tnext_time\tcandidate_times\tcandidate_spread_s\n"
            "1\t00:01.00\tone\tREVIEW\tcandidate_disagreement,low_confidence\t1\t9\treview\t\t00:01.00\t\t\t\t\n",
            encoding="utf-8",
        )
        row = review_server._song_payload(self.report, data)["rows"][0]
        self.assertEqual(row["reason_codes"], ["candidate_disagreement", "low_confidence"])
        self.assertEqual(row["reason"], "candidate_disagreement,low_confidence")

    def test_selector_exposes_precentral_recommendation(self) -> None:
        candidates, extra_candidates = review_server._candidate_rows(
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
        self.assertEqual(extra_candidates, [])
        self.assertIn(3701, [item["cs"] for item in candidates])
        diagnostic = next(item for item in candidates if item["cs"] == 3701)
        self.assertFalse(diagnostic["diagnostic_only"])
        self.assertIn("raw_asr", diagnostic["sources"])
        self.assertIn("ctc-nearby", diagnostic["sources"])

    def test_selector_marks_raw_only_recommendation_diagnostic(self) -> None:
        candidates, extra_candidates = review_server._candidate_rows(
            {"precentral_candidate_recommendation": {"time": 37.0, "source": "raw_asr", "score": .98}},
            3642,
            3642,
        )
        self.assertEqual(extra_candidates, [])
        diagnostic = next(item for item in candidates if item["cs"] == 3700)
        self.assertTrue(diagnostic["diagnostic_only"])

    def test_selector_skips_missing_timestamps_without_crashing(self) -> None:
        candidates, extra_candidates = review_server._candidate_rows(
            {"candidates": [{"source": "bad", "time": None}, "malformed"], "ctc_first_token_start": None},
            None,
            100,
        )
        self.assertEqual([item["cs"] for item in candidates], [100])
        self.assertEqual(extra_candidates, [])

    def test_selector_keeps_accepted_candidates_before_rejected_candidates(self) -> None:
        accepted = [
            {"time": 2.0 + index, "source": f"accepted-{index}", "score": 0.1 + index / 100}
            for index in range(5)
        ]
        rejected = [
            {"time": 20.0 + index, "source": f"rejected-{index}", "score": 0.99 - index / 100}
            for index in range(5)
        ]
        candidates, extra_candidates = review_server._candidate_rows(
            {"candidates": accepted, "rejected_candidates": rejected},
            100,
            110,
        )
        self.assertEqual(len(extra_candidates), 3)
        nonrequired = [item for item in candidates if item["cs"] not in {100, 110}]
        self.assertEqual(
            {item["sources"][0] for item in nonrequired[:5]},
            {f"accepted-{index}" for index in range(5)},
        )
        self.assertEqual([item["cs"] for item in candidates], sorted(item["cs"] for item in candidates))
        self.assertEqual(
            {source for item in candidates for source in item["sources"] if source.startswith("accepted-")},
            {f"accepted-{index}" for index in range(5)},
        )

    def test_candidate_agreement_annotations_and_extras(self) -> None:
        def candidates(items):
            return review_server._candidate_rows({"candidates": items}, None, None)

        merged, extras = candidates([
            {"time": 1.00, "source": "a", "score": .9},
            {"time": 1.04, "source": "b", "score": .8},
        ])
        self.assertEqual(extras, [])
        self.assertEqual(merged[0]["cs"], 100)
        self.assertEqual([member["cs"] for member in merged[0]["members"]], [100, 104])
        self.assertEqual(merged[0]["agree_count"], 1)
        self.assertEqual(merged[0]["agree_span_cs"], 4)

        split, _ = candidates([
            {"time": 1.00, "source": "a", "score": .9},
            {"time": 1.06, "source": "b", "score": .8},
        ])
        self.assertEqual([item["cs"] for item in split], [100, 106])
        self.assertEqual([item["agree_count"] for item in split], [1, 1])

        annotated, _ = candidates([
            {"time": 1.00, "source": "a", "score": .9},
            {"time": 1.14, "source": "b", "score": .8},
        ])
        self.assertEqual([item["agree_count"] for item in annotated], [1, 1])
        self.assertEqual([item["agree_span_cs"] for item in annotated], [14, 14])
        no_annotation, _ = candidates([
            {"time": 1.00, "source": "a", "score": .9},
            {"time": 1.16, "source": "b", "score": .8},
        ])
        self.assertEqual([item["agree_count"] for item in no_annotation], [0, 0])

        required, _ = review_server._candidate_rows(
            {"candidates": [{"time": 1.04, "source": "raw_asr", "score": .98}]},
            100,
            100,
        )
        required_chip = next(item for item in required if item["cs"] == 100)
        self.assertTrue(required_chip["required"])
        self.assertTrue(next(member for member in required_chip["members"] if member["source"] == "raw_asr")["diagnostic"])
        self.assertEqual(required_chip["agree_count"], 0)
        self.assertTrue(required_chip["diagnostic_only"] is False)

        diagnostic, _ = candidates([
            {"time": 37.00, "source": "raw_asr", "score": .98},
            {"time": 37.01, "source": "ctc", "score": .90},
        ])
        self.assertEqual(diagnostic[0]["cs"], 3701)
        self.assertFalse(diagnostic[0]["diagnostic_only"])

        mixed, _ = candidates([
            {"time": 1.00, "source": "accepted", "score": .9},
            {"time": 1.04, "source": "rejected", "score": .8},
        ])
        self.assertFalse(mixed[0]["rejected"])
        shuffled, _ = candidates([
            {"time": 1.04, "source": "b", "score": .8},
            {"time": 1.00, "source": "a", "score": .9},
        ])
        ordered, _ = candidates([
            {"time": 1.00, "source": "a", "score": .9},
            {"time": 1.04, "source": "b", "score": .8},
        ])
        self.assertEqual(shuffled, ordered)

        chain, _ = candidates([
            {"time": 0.00, "source": "a", "score": .9},
            {"time": 0.04, "source": "b", "score": .8},
            {"time": 0.08, "source": "c", "score": .7},
        ])
        self.assertEqual([item["cs"] for item in chain], [0, 8])
        self.assertEqual([len(item["members"]) for item in chain], [2, 1])

        many, extra_count = candidates([
            {"time": index, "source": f"candidate-{index}", "score": 1.0 - index / 100}
            for index in range(1, 13)
        ])
        self.assertEqual(len(many), 9)
        self.assertEqual(len(extra_count), 3)
        self.assertEqual([item["cs"] for item in many + extra_count], list(range(100, 1300, 100)))

    def test_stem_peaks_endpoint_cache_and_validation(self) -> None:
        samples = []
        for index in range(40000):
            seconds = index / 16000
            samples.append(int(0.8 * 32767 * (1 if 1.0 <= seconds < 1.2 else 0)))
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))
            self.stem.write_bytes(buffer.getvalue())
        before_lrc = self.output.read_bytes()
        with review_server._PEAK_CACHE_LOCK:
            review_server._PEAK_CACHE.clear()

        status, _, body = self._request(
            "GET", "/api/song/0/peaks?from=0&to=200&bins=20&t=test-token"
        )
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["sample_rate"], 16000)
        with mock.patch.object(review_server, "_song_payload", wraps=review_server._song_payload) as payload_builder:
            self.assertEqual(
                self._request(
                    "GET", "/api/song/0/peaks?from=0&to=200&bins=20&t=test-token"
                )[0],
                200,
            )
            self.assertEqual(payload_builder.call_count, 0)
        self.assertGreater(payload["peaks"][10][1], .5)
        self.assertGreater(payload["peaks"][11][1], .5)
        self.assertEqual(payload["peaks"][12], [0.0, 0.0])

        status, _, body = self._request(
            "GET", "/api/song/0/peaks?from=50&to=250&bins=20&t=test-token"
        )
        self.assertEqual(status, 200)
        shifted = json.loads(body)["peaks"]
        self.assertGreater(shifted[5][1], .5)
        self.assertGreater(shifted[6][1], .5)

        with mock.patch.object(review_server.wave, "open", wraps=wave.open) as opened:
            with review_server._PEAK_CACHE_LOCK:
                review_server._PEAK_CACHE.clear()
            review_server._read_stem_peaks(self.stem, 0, 200, 20)
            review_server._read_stem_peaks(self.stem, 0, 200, 20)
            self.assertEqual(opened.call_count, 1)
            stat = self.stem.stat()
            self.stem.touch()
            os.utime(self.stem, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            review_server._read_stem_peaks(self.stem, 0, 200, 20)
            self.assertEqual(opened.call_count, 2)
            with review_server._PEAK_CACHE_LOCK:
                review_server._PEAK_CACHE.clear()
            for index in range(257):
                review_server._read_stem_peaks(self.stem, index, index + 200, 20)
            with review_server._PEAK_CACHE_LOCK:
                self.assertEqual(len(review_server._PEAK_CACHE), 256)
            calls = opened.call_count
            review_server._read_stem_peaks(self.stem, 0, 200, 20)
            self.assertEqual(opened.call_count, calls + 1)

        self.assertEqual(
            self._request("GET", "/api/song/0/peaks?from=0&to=200&bins=20")[0],
            403,
        )
        self.assertEqual(
            self._request("GET", "/api/song/0/peaks?from=0&to=801&bins=20&t=test-token")[0],
            400,
        )
        self.assertEqual(
            self._request("GET", "/api/song/0/peaks?from=0&to=200&bins=601&t=test-token")[0],
            400,
        )
        self.server.store.songs[0]["ctc_audio_path"] = str(self.root / "missing.wav")
        self.assertEqual(
            self._request("GET", "/api/song/0/peaks?from=0&to=200&bins=20&t=test-token")[0],
            404,
        )
        invalid = self.root / "invalid.wav"
        with wave.open(str(invalid), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(b"\0\0" * 16)
        self.server.store.songs[0]["ctc_audio_path"] = str(invalid)
        self.assertEqual(
            self._request("GET", "/api/song/0/peaks?from=0&to=200&bins=20&t=test-token")[0],
            415,
        )
        self.assertEqual(self.output.read_bytes(), before_lrc)

    def test_one_line_report_with_missing_assignment_timestamp_stays_operable(self) -> None:
        output = self.root / "one-line.lrc"
        output.write_bytes(b"[00:01.00]one\n")
        report = self.root / "one-line.align-report.json"
        report.write_text(json.dumps({
            "audio_path": str(self.audio), "ctc_audio_path": str(self.stem),
            "output_path": str(output), "display_lines": 1, "timing_entries": 1,
            "assignments": [{"entry": 1, "lrc_tag": "[00:01.00]", "candidates": [{"time": None}]}],
        }), encoding="utf-8")
        detail = review_server.ReviewStore(self.root, [report]).song(0)
        self.assertEqual(len(detail["rows"]), 1)
        self.assertEqual(detail["rows"][0]["candidates"][0]["cs"], 100)

    def test_mapping_payload_and_adjacent_lines(self) -> None:
        detail = self._detail()
        self.assertFalse(detail["read_only"])
        self.assertEqual(len(detail["rows"]), 5)
        self.assertEqual(detail["rows"][1]["lyrics"], ["two a", "two b"])
        self.assertEqual(detail["rows"][2]["previous"]["cs"], 200)
        self.assertEqual(detail["rows"][2]["next"]["lyrics"], ["four"])
        self.assertEqual(detail["rows"][4]["current_cs"], 6200)
        self.assertIn("base_sha256", detail)

    def _move_song(self, folder_name: str) -> Path:
        folder = self.root / folder_name
        folder.mkdir()
        reports = self.root / "reports"
        reports.mkdir()
        self.report = self.report.rename(reports / self.report.name)
        self.output = self.output.rename(folder / self.output.name)
        self.audio = self.audio.rename(folder / self.audio.name)
        return reports

    def test_moved_song_is_reviewed_and_saved_where_it_now_is(self) -> None:
        reports = self._move_song("checked")
        for target in (reports.parent / "checked", self.output, self.audio):
            store = review_server.ReviewStore(reports, [target], scan_report_dir=False)
            self.assertEqual(len(store.songs), 1, target)
            song = store.songs[0]
            self.assertFalse(song["read_only"], song["read_only_reason"])
            self.assertEqual(Path(song["output_path"]).resolve(), self.output.resolve())
            self.assertEqual(Path(song["audio_path"]).resolve(), self.audio.resolve())
            self.assertTrue(song["has_flac"])
        self.server.store = store
        detail = self._detail()
        body = {"base_sha256": detail["base_sha256"], "edits": [{"entry": 2, "cs": 250, "source": "ctc", "nudge_cs": 50}]}
        self.assertEqual(self._json_request("POST", "/api/song/0/save", body)[0], 200)
        self.assertIn(b"[00:02.50]two a", self.output.read_bytes())

    def test_missing_flac_is_found_beside_its_lrc(self) -> None:
        data = json.loads(self.report.read_text(encoding="utf-8"))
        data["audio_path"] = str(self.root / "old music folder" / self.audio.name)
        self.report.write_text(json.dumps(data), encoding="utf-8")
        store = review_server.ReviewStore(self.root, [self.report])
        song = store.songs[0]
        self.assertTrue(song["has_flac"])
        self.assertEqual(Path(song["audio_path"]).resolve(), self.audio.resolve())
        self.assertEqual(Path(song["output_path"]).resolve(), self.output.resolve())

    def test_dropping_a_flac_in_refreshes_without_restart(self) -> None:
        self.audio.unlink()
        store = review_server.ReviewStore(self.root, [self.root])
        self.server.store = store
        token = {"X-Review-Token": self.server.token}
        self.assertFalse(store.songs[0]["has_flac"])
        first = json.loads(self._request("GET", "/api/version", **token)[2])["generation"]
        self.assertEqual(json.loads(self._request("GET", "/api/version", **token)[2])["generation"], first)
        self.audio.write_bytes(b"0123456789")
        second = json.loads(self._request("GET", "/api/version", **token)[2])["generation"]
        self.assertEqual(second, first + 1)
        self.assertTrue(store.songs[0]["has_flac"])
        self.assertEqual(self._request("GET", "/api/version")[0], 403)

    # ---- import panel ----
    def _token(self):
        return {"X-Review-Token": self.server.token}

    def _fake_wrapper(self) -> Path:
        script = self.root / "fake_align.py"
        script.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "audio, report_dir = Path(sys.argv[1]), Path(sys.argv[2])\n"
            "print('LRC tools: generating same-folder .lrc', flush=True)\n"
            "if audio.stem.startswith('bad'):\n"
            "    print('ERROR: boom'); sys.exit(1)\n"
            "lrc = audio.with_suffix('.lrc')\n"
            "lrc.write_bytes(b'[00:01.00]one\\n[00:02.00]two\\n')\n"
            "report = {'audio_path': str(audio), 'output_path': str(lrc), 'timing_entries': 2, 'duration_seconds': 10.0,\n"
            "          'assignments': [{'entry': 1, 'lrc_tag': '[00:01.00]', 'review_required': True, 'verification_status': 'review-required', 'candidates': []},\n"
            "                          {'entry': 2, 'lrc_tag': '[00:02.00]', 'review_required': False, 'verification_status': 'trusted', 'candidates': []}]}\n"
            "(report_dir / (audio.stem + '--abcdef1234.align-report.json')).write_text(json.dumps(report), encoding='utf-8')\n",
            encoding="utf-8",
        )
        return script

    def _wait_job(self, job_id: str, timeout: float = 15.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            payload = json.loads(self._request("GET", "/api/jobs", **self._token())[2])
            job = next((j for j in payload["jobs"] if j["id"] == job_id), None)
            if job and all(item["status"] not in ("queued", "running") for item in job["items"]):
                return job
            time.sleep(0.1)
        self.fail(f"job {job_id} did not finish")

    def test_import_inspect_follows_lyric_discovery_order(self) -> None:
        folder = self.root / "album"
        folder.mkdir()
        for name in ("a.flac", "a.lyrics.lrc", "a.txt", "b.flac", "b.txt", "c.flac", "notes.md"):
            (folder / name).write_bytes(b"x")
        self.server.jobs = review_server.JobRunner(self.store, self.root)
        status, _headers, data = self._json_request("POST", "/api/import/inspect", {"paths": [str(folder), str(self.audio), str(folder / "notes.md")]})
        self.assertEqual(status, 200)
        payload = json.loads(data)
        by_name = {item["name"]: item for item in payload["items"]}
        self.assertEqual([item["name"] for item in payload["items"]], ["a", "b", "c", "song"])
        self.assertTrue(by_name["a"]["lyrics"].endswith("a.lyrics.lrc"))
        self.assertTrue(by_name["b"]["lyrics"].endswith("b.txt"))
        self.assertIsNone(by_name["c"]["lyrics"])
        self.assertFalse(by_name["a"]["existing_lrc"])
        self.assertTrue(by_name["song"]["existing_lrc"])
        self.assertEqual([item["reason"] for item in payload["skipped"]], ["not-a-flac"])

    def test_import_lyrics_are_written_once_beside_the_flac(self) -> None:
        audio = self.root / "fresh.flac"
        audio.write_bytes(b"x")
        self.server.jobs = review_server.JobRunner(self.store, self.root)
        status, _h, data = self._json_request("POST", "/api/import/lyrics", {"audio": str(audio), "text": "line one\r\nline two\r\n"})
        self.assertEqual(status, 200, data)
        target = self.root / "fresh.lyrics.txt"
        self.assertEqual(target.read_bytes(), b"line one\nline two\n")
        self.assertEqual(self._json_request("POST", "/api/import/lyrics", {"audio": str(audio), "text": "again"})[0], 409)
        other = self.root / "other.flac"
        other.write_bytes(b"x")
        self.assertEqual(self._json_request("POST", "/api/import/lyrics", {"audio": str(other), "text": "  "})[0], 400)
        self.assertFalse((self.root / "other.lyrics.txt").exists())

    def test_jobs_run_the_wrapper_and_land_in_the_song_list(self) -> None:
        fake = self._fake_wrapper()
        good, bad, bare = (self.root / "good.flac", self.root / "bad.flac", self.root / "bare.flac")
        for audio in (good, bad, bare):
            audio.write_bytes(b"x")
        (self.root / "good.txt").write_text("one\ntwo\n", encoding="utf-8")
        (self.root / "bad.txt").write_text("one\ntwo\n", encoding="utf-8")
        store = review_server.ReviewStore(self.root, [self.report], scan_report_dir=False)
        self.server.store = store
        runner = review_server.JobRunner(store, self.root)
        self.server.jobs = runner
        with mock.patch.object(review_server, "_align_command",
                               side_effect=lambda audio, overwrite, report_dir: [sys.executable, str(fake), str(audio), str(report_dir)]):
            status, _h, data = self._json_request("POST", "/api/jobs", {"items": [{"audio": str(good)}, {"audio": str(bad)}, {"audio": str(bare)}]})
            self.assertEqual(status, 200, data)
            created = json.loads(data)
            self.assertEqual([r["reason"] for r in created["rejected"]], ["缺歌詞"])
            job = self._wait_job(created["job"])
        by_name = {item["name"]: item for item in job["items"]}
        self.assertEqual(by_name["good"]["status"], "done")
        self.assertTrue(by_name["good"]["report"].endswith("good--abcdef1234.align-report.json"))
        self.assertEqual(by_name["bad"]["status"], "failed")
        self.assertEqual(by_name["bad"]["error"], "ERROR: boom")
        self.assertTrue(Path(by_name["bad"]["log"]).is_file())
        songs = json.loads(self._request("GET", "/api/songs", **self._token())[2])
        self.assertEqual(sorted(song["song"] for song in songs), ["good--abcdef1234", "song"])
        self.assertTrue((runner.jobs_dir / (job["id"] + ".json")).is_file())
        self.assertEqual(self._json_request("POST", "/api/jobs", {"items": []})[0], 400)

    def test_decisions_are_logged_beside_the_report(self) -> None:
        status, _h, data = self._json_request("POST", "/api/song/0/decision", {"entry": 2, "decision": "keep", "cs": None, "source": "keep"})
        self.assertEqual(status, 200, data)
        self.assertEqual(self._json_request("POST", "/api/song/0/decision", {"entry": 2, "decision": "change", "cs": 250, "source": "drag"})[0], 200)
        log = self.report.with_name(self.report.name.replace(".align-report.json", ".review-decisions.jsonl"))
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([(r["entry"], r["decision"], r["chosen_cs"]) for r in records], [(2, "keep", records[0]["current_cs"]), (2, "change", 250)])
        self.assertIsInstance(records[0]["reason_codes"], list)
        self.assertIn("review", records[0])
        self.assertEqual(self._json_request("POST", "/api/song/0/decision", {"entry": 2, "decision": "maybe"})[0], 400)
        self.assertEqual(self._json_request("POST", "/api/song/0/decision", {"entry": 99, "decision": "keep"})[0], 400)
        self.assertEqual(self._json_request("POST", "/api/song/0/decision", {"entry": 2, "decision": "keep", "extra": 1})[0], 400)
        self.server.store = review_server.ReviewStore(self.root, [self.report], read_only=True)
        self.assertEqual(self._json_request("POST", "/api/song/0/decision", {"entry": 2, "decision": "keep"})[0], 403)
        self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 2)
        import review_decisions_summary
        summary = review_decisions_summary.summarize(review_decisions_summary.load_decisions(log.parent))
        self.assertEqual(summary["decisions"], 1, "the later change overrides the earlier keep")
        self.assertTrue(all(counts["change"] == 1 for counts in summary["by_code"].values()))

    def test_import_endpoints_are_disabled_when_read_only(self) -> None:
        store = review_server.ReviewStore(self.root, [self.report], read_only=True)
        self.server.store = store
        self.server.jobs = review_server.JobRunner(store, self.root)
        payload = json.loads(self._request("GET", "/api/jobs", **self._token())[2])
        self.assertFalse(payload["import_enabled"])
        self.assertEqual(self._json_request("POST", "/api/import/inspect", {"paths": [str(self.audio)]})[0], 403)
        self.assertEqual(self._json_request("POST", "/api/jobs", {"items": [{"audio": str(self.audio)}]})[0], 403)
        self.assertEqual(self._request("POST", "/api/import/inspect", b"{}", **{"Content-Type": "application/json"})[0], 403)

    def test_song_load_failure_keeps_stub_in_api_song_list(self) -> None:
        other = self.root / "z-song.align-report.json"
        other.write_text(self.report.read_text(encoding="utf-8"), encoding="utf-8")
        real_payload = review_server._song_payload

        def payload_or_fail(path, report, anchor=None):
            if path.name == self.report.name:
                raise RuntimeError("payload exploded")
            return real_payload(path, report, anchor)

        with mock.patch.object(review_server, "_song_payload", side_effect=payload_or_fail):
            store = review_server.ReviewStore(self.root, [self.report, other])
            self.server.store = store
            songs = json.loads(self._request("GET", "/api/songs", **{"X-Review-Token": self.server.token})[2])
            failed = next(song for song in songs if song["song"] == "song")
            healthy = next(song for song in songs if song["song"] == "z-song")
            self.assertIn("load_error", failed)
            self.assertTrue(failed["read_only"])
            self.assertGreater(healthy["rows"], 0)
            detail = json.loads(self._request(
                "GET", f"/api/song/{failed['index']}", **{"X-Review-Token": self.server.token}
            )[2])
            self.assertEqual(detail["rows"], [])
            self.assertTrue(detail["read_only"])
            self.assertIn("payload exploded", detail["load_error"])

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
        self.assertIn("a.song.index - b.song.index || a.row.entry - b.row.entry", page)
        # 1.6: the main card shows at most two alternatives plus keep; raw candidates live in the details fold.
        self.assertIn("optionsOf(row)", page)
        self.assertIn("row.options", page)
        self.assertIn("row.reason_text", page)
        self.assertIn('class="option ', page)
        self.assertIn("candidate-play", page)
        self.assertIn("candidate-apply", page)
        self.assertIn("stage(item, candidate.cs", page)
        self.assertIn("CTC first-sound candidate", page)
        self.assertIn("CTC first-sound (posterior)", page)
        self.assertIn("ASR diagnostic hint", page)
        self.assertIn("diagnostic only", page)
        self.assertIn("row.extra_candidates || []", page)
        self.assertIn('<details class="more"', page)
        self.assertIn("data-candidate-list", page)
        self.assertIn("agree_count", page)
        self.assertIn("className = 'peaks'", page)
        self.assertIn("/peaks?from=", page)
        self.assertIn("flacPeaks(item, fromCs, toCs, bins)", page)
        self.assertIn("無法顯示波形：", page)
        self.assertIn("Math.round((event.clientX - drag.startX) * 600 / stripPx())", page)
        self.assertIn("stage(item, previewCs, 'drag', 0)", page)
        self.assertIn("playAround(item, previewCs)", page)
        self.assertIn("R2 skipped:", page)
        self.assertIn("load error:", page)
        self.assertIn("這首沒有需要審的句子。", page)
        self.assertIn("找不到這首的 LRC 檔，無法審核", page)
        self.assertIn("首無法審核（LRC 已搬走或是舊版報告）", page)
        self.assertIn("const unreachable = song => !!song.read_only && !(song.detail?.rows || []).length;", page)
        self.assertIn("state.songs.find(song => !unreachable(song))", page)
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
  {entry:1, review:'REVIEW', current_cs:2000, auto_cs:2000, lyrics:['one'], candidates:Array.from({length:9}, (_,i) => ({cs:3000+i*100, sources:['ctc'], diagnostic_only:false})), extra_candidates:Array.from({length:3}, (_,i) => ({cs:4000+i*100, sources:['whisper'], diagnostic_only:false})), severity_score:1, severity_rank:2, candidate_spread:null},
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
  const queue = elements.get('queue'), rail = elements.get('rail');
  const card = queue.children.find(child => String(child.className).startsWith('row '));
  const list = queue.children.find(child => String(child.className).startsWith('queue-list'));
  const html = card._html;
  console.log(JSON.stringify({
    tabs: rail.children.filter(child => String(child.className).startsWith('song-tab')).map(child => child._html.includes('song') && child._html.includes('剩 2')),
    sections: rail.children.filter(child => child.className === 'rail-sec').map(child => child.textContent),
    key: card.dataset.key,
    queue: (list.children.find(child => child.className === 'list') || list).children.map(child => child.className),
    audio: html.includes('class="play" data-kind="flac"'),
    nudge: html.includes('data-delta="-10"'),
    options: (html.match(/aria-label="option /g) || []).length,
    extras: html.includes('<details class="more"') && html.includes('data-candidate-list="extra"')
  }));
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
""".replace("SOURCE", json.dumps(source))
        result = subprocess.run(["node", "-"], input=harness, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"tabs": [True], "sections": ["待確認 1"], "key": "0:1", "queue": ["q cur", "q"], "audio": True, "nudge": True, "options": 3, "extras": True})

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

    def _hand_edit(self, replacements: dict[bytes, bytes]) -> None:
        data = self.output.read_bytes()
        for old, new in replacements.items():
            self.assertIn(old, data)
            data = data.replace(old, new)
        self.output.write_bytes(data)
        self.store.refresh_if_changed()

    def test_star_marked_lrc_is_shown_flagged_not_hidden(self) -> None:
        self._hand_edit({b"[00:02.00]two a": b"*[00:02.00]two a", b"[00:02.00]two b": b"*[00:02.00]two b",
                         b"[00:03.00]four": b"*[00:03.00]four*"})
        detail = self._detail()
        self.assertFalse(detail["read_only"])
        self.assertTrue(detail["manual_edited"])
        self.assertEqual(detail["manual_edit_reason"], "lrc-differs-from-published")
        self.assertEqual(detail["marked_rows"], 2)
        self.assertEqual(len(detail["rows"]), 5)
        rows = {row["entry"]: row for row in detail["rows"]}
        self.assertEqual([entry for entry, row in rows.items() if row["user_marked"]], [2, 4])
        self.assertEqual(rows[2]["lyric"], "two a\ntwo b")  # `*` stripped in the view only
        self.assertEqual(rows[4]["lyric"], "four")
        self.assertEqual((rows[2]["review"], rows[2]["risk"], rows[2]["risk_basis"]), ("REVIEW", "high", "user-mark"))
        self.assertIn("user-mark", rows[2]["reason_codes"])
        self.assertTrue(any(option["kind"] != "keep" for option in rows[2]["options"]))  # published candidates kept
        self.assertFalse(rows[1]["user_marked"])
        listing = json.loads(self._request("GET", "/api/songs", **{"X-Review-Token": self.server.token})[2])
        self.assertTrue(listing[0]["manual_edited"])
        # Re-alignment must still refuse to overwrite the hand edit.
        import lrc_manual_edits
        self.assertEqual(lrc_manual_edits.manual_edit_reason(self.output, self.report), "lrc-differs-from-published")

    def test_hand_edited_timestamp_is_read_only_with_reason(self) -> None:
        self._hand_edit({b"[00:03.00]three": b"[00:03.20]three"})
        detail = self._detail()
        self.assertTrue(detail["read_only"])
        self.assertIsNotNone(detail["read_only_reason"])
        response = self._json_request("POST", "/api/song/0/save", {"base_sha256": detail["base_sha256"], "edits": []})
        status, body = response[0], json.loads(response[2])
        self.assertEqual(status, 403)
        self.assertEqual(body["error"], detail["read_only_reason"])

    def test_converted_song_same_tags_new_text_multiline_saves_only_its_group(self) -> None:
        self._hand_edit({b"two a": b"zwei a", b"two b": b"zwei b", b"[00:01.00]one": b"[00:01.00]eins"})
        detail = self._detail()
        self.assertFalse(detail["read_only"])
        before = self.output.read_bytes()
        rows = {row["entry"]: row for row in detail["rows"]}
        self.assertEqual(rows[2]["lyric"], "zwei a\nzwei b")
        self.assertTrue(rows[2]["candidates"] or rows[2]["options"])
        body = {"base_sha256": detail["base_sha256"],
                "edits": [{"entry": 2, "cs": 215, "source": "whisper", "nudge_cs": 15}]}
        status = self._json_request("POST", "/api/song/0/save", body)[0]
        self.assertEqual(status, 200)
        after = self.output.read_bytes()
        changed = [(x, y) for x, y in zip(before.split(b"\n"), after.split(b"\n")) if x != y]
        self.assertEqual(len(changed), 2)
        self.assertTrue(all(y.startswith(b"[00:02.15]zwei") for _x, y in changed))

    def test_save_keeps_marks_on_lines_that_were_not_edited(self) -> None:
        self._hand_edit({b"[00:02.00]two a": b"*[00:02.00]two a", b"[00:02.00]two b": b"*[00:02.00]two b",
                         b"[00:03.00]four": b"*[00:03.00]four*"})
        detail = self._detail()
        body = {"base_sha256": detail["base_sha256"],
                "edits": [{"entry": 4, "cs": 350, "source": "whisper", "nudge_cs": 50}]}
        self.assertEqual(self._json_request("POST", "/api/song/0/save", body)[0], 200)
        saved = self.output.read_bytes()
        self.assertIn(b"*[00:02.00]two a\n*[00:02.00]two b\n", saved)  # untouched marks survive byte for byte
        self.assertIn(b"[00:03.50]four\n", saved)  # the edited line is resolved, its mark goes
        self.assertNotIn(b"four*", saved)
        auto, edits = review_server.review_artifact_paths(self.report)
        self.assertTrue(auto.is_file() and edits.is_file())
        again = self._detail()
        self.assertFalse(again["read_only"])
        self.assertEqual(again["marked_rows"], 1)
        self.assertEqual(again["rows"][3]["current_cs"], 350)

    def _real_flac(self) -> None:
        import shutil
        try:
            import mutagen.flac  # noqa: F401
        except ImportError:
            self.skipTest("needs mutagen")
        if shutil.which(review_server.runtime_paths.ffmpeg()) is None:
            self.skipTest("needs ffmpeg")
        subprocess.run([review_server.runtime_paths.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                        "-c:a", "flac", str(self.audio)], check=True, creationflags=review_server.lyrics_tag_export.PROC_FLAGS)
        self.store.refresh_if_changed()

    def test_export_writes_the_saved_lrc_into_the_audio_tag(self) -> None:
        self._real_flac()
        status, _headers, data = self._json_request("POST", "/api/song/0/export", {"audio": "", "check": True})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data), {"targets": [{"name": "song.flac", "tag": "LYRICS", "has_lyrics": False, "lines": 0}], "lrc_lines": 9})
        lrc_before = self.output.read_bytes()
        status, _headers, data = self._json_request("POST", "/api/song/0/export", {"audio": "song.flac"})
        result = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual({key: result[key] for key in ("name", "tag", "lines", "archive")},
                         {"name": "song.flac", "tag": "LYRICS", "lines": 9, "archive": {"mode": "off"}})
        self.assertIsInstance(result["seconds"], float)
        import mutagen.flac
        text = mutagen.flac.FLAC(str(self.audio))["LYRICS"][0]
        self.assertEqual(text, __import__("lyrics_tag_export").lrc_text(lrc_before))
        self.assertEqual(self.output.read_bytes(), lrc_before, "export leaves the .lrc as saved")
        self.assertTrue((self.root / "export-log.jsonl").is_file())

    def test_export_archives_the_lrc_and_the_song_then_shows_as_exported(self) -> None:
        self._real_flac()
        archive = self.root / "bk"
        with mock.patch.dict(os.environ, {"LRC_EXPORT_ARCHIVE_DIR": str(archive)}):
            status, _headers, data = self._json_request("POST", "/api/song/0/export", {"audio": "song.flac"})
        self.assertEqual(status, 200)
        moved = Path(json.loads(data)["archive"]["to"])
        self.assertEqual(moved, archive / self.output.name)
        self.assertTrue(moved.is_file())
        self.assertFalse(self.output.exists(), "the .lrc left its folder")
        self.assertTrue(self.audio.is_file())
        report_path = next(Path(self.store.report_dir).glob("*.align-report.json"))
        payload = review_server._song_payload(report_path, json.loads(report_path.read_text(encoding="utf-8")))
        self.assertTrue(payload["read_only"])
        self.assertEqual(payload["read_only_reason"], "lrc-exported")

    def test_m4a_exported_song_with_another_audio_path_shows_as_exported(self) -> None:
        self._real_flac()
        m4a = self.audio.with_suffix(".m4a")
        subprocess.run([review_server.runtime_paths.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                        "-c:a", "aac", str(m4a)], check=True, creationflags=review_server.lyrics_tag_export.PROC_FLAGS)
        with mock.patch.dict(os.environ, {"LRC_EXPORT_ARCHIVE_DIR": str(self.root / "bk")}):
            status, _headers, _data = self._json_request("POST", "/api/song/0/export", {"audio": "song.m4a"})
        self.assertEqual(status, 200)
        self.assertFalse(self.output.exists())
        report_path = next(Path(self.store.report_dir).glob("*.align-report.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertIsNone(review_server.lyrics_tag_export.read_lyrics(self.audio))  # flac carries nothing
        payload = review_server._song_payload(report_path, report)
        self.assertEqual(payload["read_only_reason"], "lrc-exported")
        wav = self.audio.with_suffix(".wav")
        wav.write_bytes(b"RIFF")
        self.assertTrue(review_server._lyrics_embedded(str(wav)))  # unsupported format is skipped, siblings are read
        self.assertFalse(review_server._lyrics_embedded(str(self.root / "nothing.wav")))

    def test_failed_export_does_not_archive(self) -> None:
        archive = self.root / "bk"
        with mock.patch.dict(os.environ, {"LRC_EXPORT_ARCHIVE_DIR": str(archive)}):
            status, _headers, _data = self._json_request("POST", "/api/song/0/export", {"audio": "song.flac"})
        self.assertEqual(status, 422)
        self.assertTrue(self.output.is_file())
        self.assertFalse(archive.exists())

    def test_export_rejects_foreign_audio_unknown_fields_and_bad_origin(self) -> None:
        self._real_flac()
        other = self.root / "other.flac"
        other.write_bytes(b"x")
        for body in ({"audio": "other.flac"}, {"audio": str(other)}, {"audio": "song.flac", "path": "x"}):
            self.assertEqual(self._json_request("POST", "/api/song/0/export", body)[0], 400)
        self.assertEqual(self._json_request("POST", "/api/song/0/export", {"audio": "song.flac"}, Origin="http://evil")[0], 403)
        self.assertEqual(other.read_bytes(), b"x")

    def test_export_refuses_a_fake_audio_file_and_leaves_it_intact(self) -> None:
        try:
            import mutagen.flac  # noqa: F401
        except ImportError:
            self.skipTest("needs mutagen")
        status, _headers, data = self._json_request("POST", "/api/song/0/export", {"audio": "song.flac"})
        self.assertEqual(status, 422)
        self.assertEqual(self.audio.read_bytes(), b"0123456789")

    def test_successful_save_changes_only_review_output_whitelist(self) -> None:
        before = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.root.rglob("*") if path.is_file()
        }
        detail = self._detail()
        body = {
            "base_sha256": detail["base_sha256"],
            "edits": [{"entry": 1, "cs": 150, "source": "ctc", "nudge_cs": 50}],
        }
        self.assertEqual(self._json_request("POST", "/api/song/0/save", body)[0], 200)
        after = {
            path: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.root.rglob("*") if path.is_file()
        }
        changed = {
            path.name for path in set(before) | set(after)
            if before.get(path) != after.get(path)
        }
        auto, edits = review_server.review_artifact_paths(self.report)
        self.assertEqual(changed, {self.output.name, auto.name, edits.name})
        self.assertFalse(list(self.root.glob("*.stage")))

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
        with mock.patch.object(review_server.os, "replace", side_effect=[PermissionError()] * 3) as replace:
            status, result = review_server._save_song(song, body)
        self.assertEqual(status, 423)
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(self.output.read_bytes(), self.original)
        self.assertEqual(edits.read_bytes(), before_log)
        self.assertFalse(list(self.root.glob("*.stage")))

    def test_fsync_failure_leaves_first_save_completely_unchanged(self) -> None:
        song = self.server.store.song(0)
        body = {"base_sha256": hashlib.sha256(self.original).hexdigest(), "edits": [{"entry": 1, "cs": 150, "source": "ctc", "nudge_cs": 50}]}
        with mock.patch.object(review_server.os, "fsync", side_effect=OSError("disk full")):
            status, result = review_server._save_song(song, body)
        auto, edits = review_server.review_artifact_paths(self.report)
        self.assertEqual(status, 500)
        self.assertEqual(self.output.read_bytes(), self.original)
        self.assertFalse(auto.exists())
        self.assertFalse(edits.exists())
        self.assertFalse(list(self.root.glob("*.stage")))

    def test_first_save_auto_publish_failure_leaves_no_artifacts(self) -> None:
        song = self.server.store.song(0)
        body = {"base_sha256": hashlib.sha256(self.original).hexdigest(), "edits": [{"entry": 1, "cs": 150, "source": "ctc", "nudge_cs": 50}]}
        with mock.patch.object(review_server.os, "replace", side_effect=PermissionError("locked")):
            status, result = review_server._save_song(song, body)
        auto, edits = review_server.review_artifact_paths(self.report)
        self.assertEqual(status, 500)
        self.assertEqual(self.output.read_bytes(), self.original)
        self.assertFalse(auto.exists())
        self.assertFalse(edits.exists())
        self.assertFalse(list(self.root.glob("*.stage")))

    def test_first_save_log_stage_failure_leaves_no_artifacts(self) -> None:
        auto, edits_path = review_server.review_artifact_paths(self.report)
        song = self.server.store.song(0)
        body = {"base_sha256": hashlib.sha256(self.original).hexdigest(), "edits": [{"entry": 1, "cs": 150, "source": "ctc", "nudge_cs": 50}]}
        real_stage = review_server._stage_bytes

        def stage_or_fail(path, data):
            if path.resolve() == edits_path.resolve():
                raise OSError("log unavailable")
            return real_stage(path, data)

        with mock.patch.object(review_server, "_stage_bytes", side_effect=stage_or_fail):
            status, result = review_server._save_song(song, body)
        self.assertEqual(status, 500)
        self.assertEqual(self.output.read_bytes(), self.original)
        self.assertFalse(auto.exists())
        self.assertFalse(edits_path.exists())
        self.assertFalse(list(self.root.glob("*.stage")))

    def test_postcommit_log_failure_reports_saved_lrc_and_preserves_old_log(self) -> None:
        auto, edits_path = review_server.review_artifact_paths(self.report)
        auto.write_bytes(self.original)
        edits_path.write_text(json.dumps({"schema": 1, "auto_lrc_sha256": review_server._sha256_file(auto), "edits": []}), encoding="utf-8")
        before_log = edits_path.read_bytes()
        before_output = self.output.read_bytes()
        song = self.server.store.song(0)
        body = {"base_sha256": hashlib.sha256(before_output).hexdigest(), "edits": [{"entry": 1, "cs": 150, "source": "ctc", "nudge_cs": 50}]}
        real_replace = review_server.os.replace
        calls = 0

        def replace_output_only(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("log unavailable")
            return real_replace(source, destination)

        with mock.patch.object(review_server.os, "replace", side_effect=replace_output_only):
            status, result = review_server._save_song(song, body)
        self.assertEqual(status, 500)
        self.assertTrue(result["lrc_saved"])
        self.assertNotEqual(self.output.read_bytes(), before_output)
        self.assertEqual(edits_path.read_bytes(), before_log)
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

    def test_checked_references_root_is_reviewable(self) -> None:
        self.assertFalse(review_server._path_is_excluded(self.root / "LRC tools checked references" / "song.lrc"))
        self.assertTrue(review_server._path_is_excluded(self.root / "LRC tools checked references" / "LRC bk" / "song.lrc"))

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
        excluded = json.loads(self._request("GET", "/api/excluded", **{"X-Review-Token": self.server.token})[2])
        self.assertEqual(set(excluded), {"excluded_files"})
        self.assertNotIn(str(self.root), json.dumps(excluded))
        self.assertEqual(self._request("GET", "/api/excluded")[0], 403)

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
        for marker in ("currentTime = start", "loadedmetadata", "event.key === '['", "event.key === '{'", "base_sha256:song.detail.base_sha256", "beforeunload", "server_read_only", "response.status === 409", "reloaded and kept staged edits",
                       "base_cs:item.row.current_cs", "candidate.cs, sourceText(candidate), 0", "actualDelta = cs - before",
                       "dropped staged edit", "edit.base_cs !== item.row.current_cs", "button.blur()",
                       "event.target?.type !== 'checkbox'", "const conflict = conflicted(item)",
                       "const loadMessage = await load()", "這筆的時間在暫存後被改過", "/api/excluded", "emptyText()", "/api/version", "refreshJobs(); watchFiles();", "新增歌曲", "/api/import/inspect", "import-start"):
            self.assertIn(marker, page)
        self.assertNotIn("alert(", page)
        self.assertIn("/^[1-3]$/.test(event.key)", page)
        self.assertIn("const start = targetCs / 100;", page)
        self.assertIn("const stop = start + 2;", page)
        self.assertNotIn("targetCs / 100 - 2", page)
        self.assertIn("className = 'song-tab'", page)
        self.assertIn("state.activeSong", page)
        self.assertIn("leftOf(song) > 0", page)
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

    def test_unsaved_edits_draft_in_local_storage(self) -> None:
        page = (SCRIPTS / "review_page" / "index.html").read_text(encoding="utf-8")
        source = page.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        hook = "  load().then(() => { refreshJobs(); watchFiles(); });\n})();"
        self.assertIn(hook, source.replace("\r\n", "\n"))
        source = source.replace("\r\n", "\n").replace(hook, "  globalThis.__t = {state, stage};\n  load().then(() => { refreshJobs(); watchFiles(); });\n})();")
        harness = r"""
const vm = require('vm');
const source = SOURCE;
const storage = new Map();
const timers = new Map();
let timerId = 0;
const localStorage = {getItem:k => storage.has(k) ? storage.get(k) : null, setItem:(k, v) => storage.set(k, String(v)), removeItem:k => storage.delete(k)};
// getElementById-only DOM shim: no querySelector / setAttribute
function element(id='') {
  return {id, children:[], dataset:{}, className:'', textContent:'', disabled:false, hidden:id === 'draft-restored' || id === 'draft-stale', listeners:{},
    addEventListener(type, fn) { this.listeners[type] = () => fn({type, currentTarget:this, target:this, preventDefault(){}, stopPropagation(){}}); }, appendChild(child) { this.children.push(child); },
    querySelectorAll() { return []; }, blur() {},
    set innerHTML(value) { this._html = value; this.children = []; }, get innerHTML() { return this._html || ''; }}; }
const settle = async () => { for (let i=0; i<12; i++) await new Promise(resolve => setImmediate(resolve)); };
async function boot(sha, opts={}) {
  const elements = new Map(), docListeners = {}, winListeners = {};
  const doc = {getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); }, createElement() { return element(); },
    addEventListener(type, fn) { docListeners[type] = fn; }, visibilityState:'visible'};
  const row = {entry:1, review:'REVIEW', current_cs:100, auto_cs:100, lyrics:['one'], candidates:[{cs:200, sources:['ctc'], diagnostic_only:false}], severity_score:1};
  const saves = [];
  const response = (ok, status, data) => ({ok, status, statusText:'x', json:async () => data});
  const context = {console, document:doc, localStorage, location:{search:'?t=test', hash:'', origin:'http://test'}, addEventListener(type, fn) { winListeners[type] = fn; },
    URLSearchParams, encodeURIComponent, Map, Set, Number, String, Math, JSON, Object, Error, RegExp, Date, Promise, Array,
    setTimeout(fn, ms) { const id = ++timerId; timers.set(id, {fn, ms}); return id; }, clearTimeout(id) { timers.delete(id); }, setInterval() { return 0; }, clearInterval() {},
    fetch: async (path, options={}) => {
      if (path === '/api/songs') return response(true, 200, [{index:0, song:'demo', read_only:false, has_flac:false, has_stem:false}]);
      if (path === '/api/song/0' && !options.method) return response(true, 200, {base_sha256:sha, server_read_only:false, has_flac:true, has_stem:true, rows:[row]});
      if (path === '/api/song/0/save') { saves.push(JSON.parse(options.body)); return opts.saveFails ? response(false, 500, {error:'boom'}) : response(true, 200, {base_sha256:'after', server_read_only:false, has_flac:true, has_stem:true, rows:[Object.assign({}, row, {current_cs:150})]}); }
      if (path === '/api/jobs') return response(true, 200, {import_enabled:false, jobs:[]});
      if (path === '/api/version') return response(true, 200, {generation:0});
      throw Error('unexpected fetch ' + path);
    }};
  context.globalThis = context;
  vm.runInNewContext(SOURCE_VAR, context);
  await settle();
  return {t:context.__t, el:id => doc.getElementById(id), docListeners, winListeners, saves, context};
}
const drafts = () => JSON.parse(storage.get('lrc-review-drafts') || '{}');
const flushTimers = () => { for (const [id, t] of [...timers]) { if (t.ms === 750) { timers.delete(id); t.fn(); } } };
(async () => {
  const out = {};
  // 1. an edit is written to the draft, debounced
  let p = await boot('x');
  p.t.stage(p.t.state.rows[0], 150, 'manual', 50);
  out.beforeDebounce = storage.has('lrc-review-drafts');
  flushTimers();
  const d = drafts().demo;
  out.written = {sha:d.base_sha256, n:d.edits.length, entry:d.edits[0].entry, cs:d.edits[0].cs, src:d.edits[0].source, at:typeof d.edits[0].edited_at};
  // 2. reload with the same sha restores it
  p = await boot('x');
  out.restored = {edits:p.t.state.edits.size, dirty:p.t.state.dirty, cs:p.t.state.edits.get('0:1').cs, hidden:p.el('draft-restored').hidden, text:p.el('draft-restored-text').textContent, staleHidden:p.el('draft-stale').hidden};
  // 3. different sha: not applied, draft kept, banner offers discard / apply
  p = await boot('y');
  out.stale = {edits:p.t.state.edits.size, kept:!!drafts().demo, hidden:p.el('draft-stale').hidden, text:p.el('draft-stale-text').textContent, restoredHidden:p.el('draft-restored').hidden};
  p.t.stage(p.t.state.rows[0], 175, 'manual', 0); flushTimers();
  out.staleUntouched = drafts().demo.edits[0].cs;
  p.el('draft-stale-apply').listeners.click();
  out.applied = {edits:p.t.state.edits.size, staleHidden:p.el('draft-stale').hidden};
  storage.clear(); p = await boot('x'); p.t.stage(p.t.state.rows[0], 150, 'manual', 50); flushTimers();
  p = await boot('y');
  p.el('draft-stale-discard').listeners.click();
  out.discarded = {kept:storage.has('lrc-review-drafts'), hidden:p.el('draft-stale').hidden};
  // 4. a failed save keeps the draft, a good save clears it
  p = await boot('x');
  p.t.stage(p.t.state.rows[0], 150, 'manual', 50); flushTimers();
  p = await boot('x', {saveFails:true});
  p.el('save').listeners.click(); await settle();
  out.failedSave = {kept:!!drafts().demo};
  p = await boot('x');
  p.el('save').listeners.click(); await settle();
  out.goodSave = {posted:p.saves.length, cleared:!storage.has('lrc-review-drafts')};
  // 5. beforeunload / visibilitychange write without waiting for the debounce
  p = await boot('x');
  p.t.stage(p.t.state.rows[0], 150, 'manual', 50);
  p.winListeners.beforeunload({preventDefault(){}});
  out.unload = !!drafts().demo;
  storage.clear();
  p = await boot('x');
  p.t.stage(p.t.state.rows[0], 150, 'manual', 50);
  p.docListeners.visibilitychange();
  out.hiddenNoop = storage.has('lrc-review-drafts');
  p.context.document.visibilityState = 'hidden';
  p.docListeners.visibilitychange();
  out.hidden = !!drafts().demo;
  console.log(JSON.stringify(out));
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
""".replace("SOURCE_VAR", "source").replace("SOURCE", json.dumps(source), 1)
        flags = 0x08000000 | 0x00004000 if os.name == "nt" else 0
        result = subprocess.run(["node", "-"], input=harness, capture_output=True, text=True, encoding="utf-8", creationflags=flags)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = json.loads(result.stdout)
        self.assertFalse(out["beforeDebounce"], "the write is debounced")
        self.assertEqual(out["written"], {"sha": "x", "n": 1, "entry": 1, "cs": 150, "src": "manual", "at": "number"})
        self.assertEqual(out["restored"]["edits"], 1)
        self.assertTrue(out["restored"]["dirty"])
        self.assertEqual(out["restored"]["cs"], 150)
        self.assertFalse(out["restored"]["hidden"])
        self.assertRegex(out["restored"]["text"], r"^已還原 1 筆未儲存的修改（來自 \d\d:\d\d）$")
        self.assertTrue(out["restored"]["staleHidden"])
        self.assertEqual(out["stale"]["edits"], 0, "a draft for a changed file is not applied")
        self.assertTrue(out["stale"]["kept"])
        self.assertFalse(out["stale"]["hidden"])
        self.assertEqual(out["stale"]["text"], "有 1 筆舊的未儲存修改，但檔案已變動")
        self.assertTrue(out["stale"]["restoredHidden"])
        self.assertEqual(out["staleUntouched"], 150, "new edits do not overwrite a pending stale draft")
        self.assertEqual(out["applied"], {"edits": 1, "staleHidden": True})
        self.assertEqual(out["discarded"], {"kept": False, "hidden": True})
        self.assertTrue(out["failedSave"]["kept"], "a failed save keeps the draft")
        self.assertEqual(out["goodSave"], {"posted": 1, "cleared": True})
        self.assertTrue(out["unload"])
        self.assertFalse(out["hiddenNoop"])
        self.assertTrue(out["hidden"])

    def test_draft_autosave_scenarios(self) -> None:
        flags = 0x08000000 | 0x00004000 if os.name == "nt" else 0
        result = subprocess.run(["node", str(SCRIPTS / "test_review_drafts.js")], capture_output=True, text=True, encoding="utf-8", creationflags=flags)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("13/13 passed", result.stdout)

    def test_review_port_default_is_fixed_and_falls_back_when_busy(self) -> None:
        import socket
        parser_default = review_server.default_port()
        self.assertEqual(parser_default, 58400)
        with mock.patch.dict(os.environ, {"LRC_REVIEW_PORT": "58411"}):
            self.assertEqual(review_server.default_port(), 58411)
        with mock.patch.dict(os.environ, {"LRC_REVIEW_PORT": "junk"}):
            self.assertEqual(review_server.default_port(), 58400)
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        busy = blocker.getsockname()[1]
        server = None
        try:
            server = review_server.bind_server(busy)
            self.assertGreater(server.server_port, busy)
            self.assertLessEqual(server.server_port, busy + review_server.PORT_FALLBACK_SPAN)
        finally:
            if server is not None:
                server.server_close()
            blocker.close()
        self.assertNotIn("--port", (SCRIPTS.parent / "Review LRC.bat").read_text(encoding="utf-8", errors="replace"))

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
        result = subprocess.run(["node", "-"], input=harness, capture_output=True, text=True, encoding="utf-8")
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


    def test_targets_file_deduplicates_and_warns_once_for_missing_paths(self) -> None:
        missing = self.root / "missing.align-report.json"
        targets = self.root / "batch-targets.txt"
        targets.write_text(
            f"{self.report}\n\n{missing}\n{missing}\n",
            encoding="utf-8",
        )
        loaded = review_server._read_targets_file(targets)
        self.assertEqual(loaded, [str(self.report), str(missing)])
        errors = io.StringIO()
        with redirect_stderr(errors):
            review_server._warn_missing_targets(loaded + [str(missing)])
        self.assertEqual(errors.getvalue().count("review target not found"), 1)
        store = review_server.ReviewStore(
            self.root, loaded, scan_report_dir=False
        )
        self.assertEqual([song["report"] for song in store.songs], [str(self.report.resolve())])

    def test_candidate_rows_include_alignment_hypotheses(self) -> None:
        rows, extra = review_server._candidate_rows({
            "candidates": [{"time": 1.00, "source": "ctc-current", "score": .9}],
            "alignment_hypotheses": [
                {"source": "ctc-local-window-wide", "raw_candidate_time": 2.5, "confidence": 0.3, "mode": "region-radius-3"},
                {"source": "broken", "raw_candidate_time": None},
            ],
        }, 100, 100)
        self.assertEqual(extra, [])
        wide = next(item for item in rows if item["cs"] == 250)
        self.assertEqual((wide["label"], wide["score"], wide["rejected"], wide["diagnostic_only"]), ("ctc-local-window-wide", 0.3, False, False))
        options = review_presentation.present_options(rows, extra, 100)
        self.assertIn(250, [option["cs"] for option in options])

    def test_rows_carry_risk_basis_and_fit_and_decisions_log_them(self) -> None:
        data = json.loads(self.report.read_text(encoding="utf-8"))
        # Give every line a romaji and a plausible CTC span so line 3 fits between its neighbours.
        for item in data["assignments"]:
            item["romaji"] = "abcdefghij"
            item["ctc_token_spans"] = [{"start": 0.0, "end": 0.8, "score": 0.5}]
        data["assignments"][2]["verification_status"] = "review-required"
        self.report.write_text(json.dumps(data), encoding="utf-8")
        self.store.reload()
        rows = {row["entry"]: row for row in self.store.cached_song(0)["rows"]}
        self.assertEqual(set(rows), {1, 2, 3, 4, 5})
        for row in rows.values():
            self.assertIn("risk_basis", row)
            self.assertIn("fit", row)
        self.assertIsNone(rows[1]["fit"], "the first line has no previous line to measure against")
        for entry in (2, 3, 4, 5):
            self.assertEqual(set(rows[entry]["fit"]), {"slack_prev_s", "slack_next_s", "span_ratio", "fits"})
            self.assertIn(rows[entry]["risk_basis"], ("fit", "default", "agreement", "model-disagreement", "many-codes"))
            if rows[entry]["risk_basis"] == "fit":
                self.assertEqual(rows[entry]["risk"], "low")
                self.assertTrue(rows[entry]["fit"]["fits"])
        self.assertIsNone(rows[5]["fit"]["slack_next_s"], "the last line has no next line")
        self.assertEqual(rows[3]["fit"]["span_ratio"], 0.8, "neighbouring lines 1 s apart over 10 chars give 0.1 s/char; a 0.8 s span is 0.8 of the expected 1 s")
        status, result = review_server._record_decision(self.store.cached_song(0), {"entry": 3, "decision": "keep"})
        self.assertEqual(status, 200)
        logged = json.loads((self.root / "song.review-decisions.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(logged["risk_basis"], rows[3]["risk_basis"])
        self.assertEqual(logged["risk"], rows[3]["risk"])

    def test_many_flac_targets_parse_each_report_once(self) -> None:
        # Loading a 180-song review used to parse every report once per FLAC target (O(n^2)).
        songs = []
        for index in range(12):
            audio = self.root / f"many-{index:02d}.flac"
            output = self.root / f"many-{index:02d}.lrc"
            audio.write_bytes(b"flac")
            output.write_bytes(self.original)
            data = json.loads(self.report.read_text(encoding="utf-8"))
            data.update({"audio_path": str(audio), "output_path": str(output), "ctc_audio_path": ""})
            (self.root / f"many-{index:02d}.align-report.json").write_text(json.dumps(data), encoding="utf-8")
            songs.append(audio)
        expected = review_server.ReviewStore(self.root, songs, scan_report_dir=False).songs
        with mock.patch.object(review_server, "_load_report", wraps=review_server._load_report) as parses, \
             mock.patch.object(review_server.json, "loads", wraps=review_server.json.loads) as decodes:
            store = review_server.ReviewStore(self.root, songs, scan_report_dir=False)
        reports_in_dir = len(list(self.root.glob("*.align-report.json")))
        self.assertEqual(parses.call_count, reports_in_dir)
        self.assertEqual(decodes.call_count, reports_in_dir)
        self.assertEqual([song["song"] for song in store.songs], [f"many-{index:02d}" for index in range(12)])
        self.assertEqual(json.dumps(store.songs, default=str, sort_keys=True), json.dumps(expected, default=str, sort_keys=True))

if __name__ == "__main__":
    unittest.main()


class ArchivedReportsStayHiddenTest(unittest.TestCase):
    def test_directory_target_skips_archive_subfolder(self) -> None:
        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "_archive" / "stale_2026-10-21").mkdir(parents=True)
            live = root / "live--1.align-report.json"
            live.write_text(_json.dumps({"assignments": [], "audio_path": str(root / "a.flac")}), encoding="utf-8")
            (root / "_archive" / "stale_2026-10-21" / "old--2.align-report.json").write_text(
                _json.dumps({"assignments": [], "audio_path": str(root / "b.flac")}), encoding="utf-8")
            found, _excluded = review_server._report_files([root], root)
            self.assertEqual([path.name for path in found], ["live--1.align-report.json"])


class ReviewMappingSafetyTests(unittest.TestCase):
    """AP4: a unique tag-structure match is trusted; anything else is read-only with its specific reason."""

    @staticmethod
    def _report(display_lines, *tags):
        return {"published_lrc_sha256": "dummy", "display_lines": display_lines,
                "assignments": [{"entry": i + 1, "lrc_tag": tag} for i, tag in enumerate(tags)]}

    def _mapping(self, report, data):
        return review_server._song_mapping(report, report["assignments"], data)

    def test_f1_inserted_timed_line_is_read_only(self) -> None:
        report = self._report(3, "[00:01.00]", "[00:02.00]", "[00:03.00]")
        mapping, reason = self._mapping(report, b"[00:01.00]a\n[00:01.50]b\n[00:02.00]c\n[00:03.00]d\n")
        self.assertIsNone(mapping)
        self.assertEqual(reason, "timing-entry-count-mismatch")

    def test_f1_untimed_and_early_header_lines_are_skipped_by_region(self) -> None:
        report = self._report(3, "[00:01.00]", "[00:02.00]", "[00:03.00]")
        mapping, reason = self._mapping(
            report, b"[ar:x]\n\n[00:00.00]title\n[00:01.00]a\n[00:02.00]b\n[00:03.00]c\n")
        self.assertIsNone(reason)
        self.assertEqual(mapping, [(3,), (4,), (5,)])

    def test_f1_hand_edited_timestamp_is_read_only(self) -> None:
        report = self._report(3, "[00:01.00]", "[00:02.00]", "[00:03.00]")
        mapping, reason = self._mapping(report, b"[00:01.00]a\n[00:02.00]b\n[00:03.20]c\n")
        self.assertIsNone(mapping)
        self.assertIsNotNone(reason)

    def test_f2_absorbed_neighbour_line_is_read_only(self) -> None:
        report = self._report(3, "[00:01.00]", "[00:02.00]", "[00:03.00]")
        mapping, reason = self._mapping(report, b"[00:01.00]a\n[00:02.00]b\n[00:03.00]c\n[00:03.00]d\n")
        self.assertIsNone(mapping)
        self.assertIsNotNone(reason)

    def test_converted_song_multiline_entries_map_by_tag_structure(self) -> None:
        report = self._report(4, "[00:01.00]", "[00:02.00]", "[00:03.00]")
        mapping, reason = self._mapping(report, b"[00:01.00]x\n[00:02.00]y\n[00:02.00]z\n[00:03.00]w\n")
        self.assertIsNone(reason)
        self.assertEqual(mapping, [(0,), (1, 2), (3,)])

    def test_f3_title_card_at_zero_with_first_entry_at_zero(self) -> None:
        report = self._report(2, "[00:00.00]", "[00:02.00]")
        mapping, reason = self._mapping(report, b"[ti:x]\n[00:00.00]a\n[00:02.00]b\n")
        self.assertIsNone(reason)
        self.assertEqual(mapping, [(1,), (2,)])

    def test_f4_specific_reason_is_not_masked(self) -> None:
        report = self._report(3, "[00:01.00]", "[00:01.00]", "[00:02.00]")
        report["timing_entries"] = 99
        mapping, reason = self._mapping(report, b"[00:01.00]a\n[00:01.00]b\n[00:01.00]c\n[00:02.10]d\n")
        self.assertIsNone(mapping)
        self.assertIsNotNone(reason)
        report = self._report(4, "[00:01.00]", "[00:01.00]", "[00:01.00]")
        mapping, reason = self._mapping(report, b"[00:01.00]a\n[00:01.00]b\n[00:01.00]c\n[00:01.00]d\n")
        self.assertEqual((mapping, reason), (None, "ambiguous-timing-region"))

    def test_duet_equal_tags_unchanged_structure_never_guesses(self) -> None:
        report = self._report(3, "[00:01.00]", "[00:01.00]", "[00:02.00]")
        mapping, reason = self._mapping(report, b"[00:01.00]a\n[00:01.00]b\n[00:02.00]c\n")
        self.assertEqual((mapping, reason), ([(0,), (1,), (2,)], None))

    def test_f5_trailing_blank_lines_are_ignored_and_preserved(self) -> None:
        report = self._report(2, "[00:01.00]", "[00:02.00]")
        data = b"[00:01.00]a\n[00:02.00]b\n\n  \n"
        report["published_lrc_sha256"] = hashlib.sha256(data).hexdigest()
        mapping, reason = self._mapping(report, data)
        self.assertEqual((mapping, reason), ([(0,), (1,)], None))
        bom, trailing, lines = review_server._split_lrc(data)
        self.assertEqual(review_server._join_lrc(bom, trailing, lines), data)

    def test_f6_trailing_star_without_leading_mark_is_lyric_text(self) -> None:
        self.assertEqual(review_server._strip_line_marks(b"[00:01.00]Song Title*"), b"[00:01.00]Song Title*")
        self.assertEqual(review_server._line_marks(b"[00:01.00]Song Title*"), (False, False))
        self.assertEqual(review_server._strip_line_marks(b"*[00:01.00]Song Title*"), b"[00:01.00]Song Title")
        self.assertEqual(review_server._strip_line_marks(b"*[00:01.00]Song Title"), b"[00:01.00]Song Title")
