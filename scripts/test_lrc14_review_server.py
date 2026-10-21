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
import unittest
import wave
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import review_server
import lrc_review_tsv


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

    def test_song_load_failure_keeps_stub_in_api_song_list(self) -> None:
        other = self.root / "z-song.align-report.json"
        other.write_text(self.report.read_text(encoding="utf-8"), encoding="utf-8")
        real_payload = review_server._song_payload

        def payload_or_fail(path, report):
            if path.name == self.report.name:
                raise RuntimeError("payload exploded")
            return real_payload(path, report)

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
        self.assertIn("首找不到 LRC，無法審核", page)
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
    tabs: rail.children.map(child => child._html.includes('song') && child._html.includes('2/2')),
    key: card.dataset.key,
    queue: (list.children.find(child => child.className === 'list') || list).children.map(child => child.className),
    audio: html.includes('class="play" data-kind="flac"'),
    nudge: html.includes('data-delta="-10"'),
    options: (html.match(/aria-label="option /g) || []).length,
    extras: html.includes('<details class="more"') && html.includes('data-candidate-list="extra"')
  }));
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
""".replace("SOURCE", json.dumps(source))
        result = subprocess.run(["node", "-"], input=harness, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"tabs": [True], "key": "0:1", "queue": ["q cur", "q"], "audio": True, "nudge": True, "options": 3, "extras": True})

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
        for marker in ("currentTime = start", "loadedmetadata", "event.key === '['", "event.key === '{'", "base_sha256:song.detail.base_sha256", "beforeunload", "server_read_only", "response.status === 409", "reloaded and kept staged edits",
                       "base_cs:item.row.current_cs", "candidate.cs, sourceText(candidate), 0", "actualDelta = cs - before",
                       "dropped staged edit", "edit.base_cs !== item.row.current_cs", "button.blur()",
                       "event.target?.type !== 'checkbox'", "const conflict = conflicted(item)",
                       "const loadMessage = await load()", "這筆的時間在暫存後被改過"):
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

if __name__ == "__main__":
    unittest.main()
