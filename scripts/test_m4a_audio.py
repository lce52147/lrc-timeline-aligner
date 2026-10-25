from __future__ import annotations

import hashlib
import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc
import lrc_batch_summary
import r2_pipeline
import review_server
import runtime_paths

REPO_ROOT = SCRIPTS.parent
STEM = "08 R.I.P."  # the stem itself ends with a dot


class AudioExtensionSetTests(unittest.TestCase):
    def test_flac_and_m4a_are_accepted_case_insensitively(self) -> None:
        for name in ("a.flac", "a.FLAC", "a.m4a", "A.M4A", f"{STEM}.m4a"):
            self.assertTrue(runtime_paths.is_audio_file(name), name)
        for name in ("a.txt", "a.lrc", "a.m4a.txt", "a"):
            self.assertFalse(runtime_paths.is_audio_file(name), name)

    def test_wrapper_lists_the_same_extensions(self) -> None:
        text = (REPO_ROOT / "align-lrc.ps1").read_text(encoding="utf-8-sig")
        expected = ", ".join(f'"{ext}"' for ext in runtime_paths.AUDIO_EXTENSIONS)
        self.assertIn(f"$audioExtensions = @({expected})", text)


class AutoLrcEntryTests(unittest.TestCase):
    def _process(self, name: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / name
            audio.write_bytes(b"not real audio")
            args = SimpleNamespace(lyrics=None, timing_source="lyrics")
            with self.assertRaises(auto_lrc.LrcError) as caught:
                auto_lrc.process_audio(audio, args)
        return str(caught.exception)

    def test_m4a_passes_the_extension_check(self) -> None:
        # With no lyric file beside it the run stops at lyric discovery, i.e. past the audio check.
        self.assertIn("No lyric text found", self._process(f"{STEM}.m4a"))

    def test_unsupported_extension_is_still_refused(self) -> None:
        self.assertIn("audio is supported", self._process("song.ogg"))

    def test_lyrics_are_found_beside_a_stem_with_trailing_dot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / f"{STEM}.m4a"
            lyrics = Path(tmp) / f"{STEM}.txt"
            audio.write_bytes(b"x")
            lyrics.write_text("line\n", encoding="utf-8")
            self.assertEqual(auto_lrc.find_lyrics(audio), lyrics)
            self.assertEqual(audio.with_suffix(".lrc").name, f"{STEM}.lrc")


class R2PipelineTests(unittest.TestCase):
    def test_sibling_audio_prefers_existing_m4a_and_defaults_to_flac(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lyrics = root / f"{STEM}.txt"
            self.assertEqual(r2_pipeline._sibling_audio(lyrics), root / f"{STEM}.flac")
            (root / f"{STEM}.m4a").write_bytes(b"x")
            self.assertEqual(r2_pipeline._sibling_audio(lyrics), root / f"{STEM}.m4a")
            (root / f"{STEM}.flac").write_bytes(b"x")
            self.assertEqual(r2_pipeline._sibling_audio(lyrics), root / f"{STEM}.flac")


class BatchSummaryTests(unittest.TestCase):
    def test_expand_inputs_takes_m4a_files_and_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = root / f"{STEM}.m4a"
            song.write_bytes(b"x")
            (root / "other.flac").write_bytes(b"x")
            (root / "notes.txt").write_bytes(b"x")
            self.assertEqual([p.name for p in lrc_batch_summary._expand_inputs([str(song)])], [song.name])
            self.assertEqual([p.name for p in lrc_batch_summary._expand_inputs([str(root)])], [song.name, "other.flac"])


class ReviewServerM4aTests(unittest.TestCase):
    def test_expand_audio_lists_m4a_from_folder_file_and_lyric_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = root / f"{STEM}.m4a"
            for name in (song.name, f"{STEM}.txt", f"{STEM}.lrc", "plain.flac", "readme.md"):
                (root / name).write_bytes(b"x")
            audios, _skipped = review_server._expand_audio([str(root)])
            self.assertEqual([a.name for a in audios], [song.name, "plain.flac"])
            self.assertEqual(review_server._expand_audio([str(song)])[0], [song])
            for lyric in (f"{STEM}.txt", f"{STEM}.lrc"):
                self.assertEqual(review_server._expand_audio([str(root / lyric)])[0], [song], lyric)
            self.assertEqual(review_server._expand_audio([str(root / "readme.md")])[1][0]["reason"], "not-a-flac")

    def test_inspect_finds_lyrics_for_m4a_with_trailing_dot_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / f"{STEM}.m4a").write_bytes(b"x")
            (root / f"{STEM}.txt").write_text("line\n", encoding="utf-8")
            item = review_server._inspect_imports([str(root)])["items"][0]
            self.assertEqual(item["name"], STEM)
            self.assertTrue(item["lyrics"].endswith(f"{STEM}.txt"))
            self.assertFalse(item["existing_lrc"])

    def test_relocated_report_finds_the_m4a_beside_the_lrc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / f"{STEM}.m4a").write_bytes(b"x")
            (root / f"{STEM}.lrc").write_bytes(b"x")
            report = {"audio_path": str(root / "gone" / f"{STEM}.flac"), "output_path": str(root / "gone" / f"{STEM}.lrc")}
            moved = review_server._relocated(report, root)
            self.assertEqual(Path(moved["audio_path"]), root / f"{STEM}.m4a")
            self.assertEqual(Path(moved["output_path"]), root / f"{STEM}.lrc")

    def test_review_page_lists_m4a_song_and_serves_audio_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / f"{STEM}.m4a"
            output = root / f"{STEM}.lrc"
            audio.write_bytes(b"0123456789")
            data = b"[00:01.00]one\n"
            output.write_bytes(data)
            report = root / "song.align-report.json"
            report.write_text(json.dumps({
                "audio_path": str(audio), "ctc_audio_path": str(audio), "output_path": str(output),
                "display_lines": 1, "timing_entries": 1, "duration_seconds": 10.0,
                "published_lrc_sha256": hashlib.sha256(data).hexdigest(),
                "assignments": [{"entry": 1, "lrc_tag": "[00:01.00]", "review_required": True,
                                 "verification_status": "review-required",
                                 "candidates": [{"time": 1.2, "source": "ctc", "score": .9}]}],
            }), encoding="utf-8")
            server = review_server.http.server.ThreadingHTTPServer(("127.0.0.1", 0), review_server.ReviewHandler)
            server.store = review_server.ReviewStore(root, [report])
            server.token = "tok"
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host = f"127.0.0.1:{server.server_port}"

                def get(path: str, **headers):
                    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                    conn.request("GET", path, headers={"Host": host, **headers})
                    resp = conn.getresponse()
                    body = resp.read()
                    conn.close()
                    return resp.status, dict(resp.getheaders()), body

                songs = json.loads(get("/api/songs", **{"X-Review-Token": "tok"})[2])
                listed = songs["songs"] if isinstance(songs, dict) else songs
                self.assertEqual(len(listed), 1)
                detail = json.loads(get("/api/song/0", **{"X-Review-Token": "tok"})[2])
                self.assertTrue(detail["has_flac"])
                status, headers, body = get("/audio/0/flac?t=tok")
                self.assertEqual((status, body), (200, b"0123456789"))
                self.assertEqual(headers["Content-Type"], "audio/mp4")
            finally:
                server.shutdown()
                thread.join(timeout=3)
                server.server_close()


if __name__ == "__main__":
    unittest.main()
