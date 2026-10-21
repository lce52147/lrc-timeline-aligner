from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
BATCH_PATH = HERE / "batch_playlist.py"


def load_batch():
    if not BATCH_PATH.exists():
        raise AssertionError("batch_playlist.py has not been implemented")
    spec = importlib.util.spec_from_file_location("batch_playlist", BATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise AssertionError("could not load batch_playlist.py")
    spec.loader.exec_module(module)
    return module


class BatchPlaylistTests(unittest.TestCase):
    def test_playlist_paths_resolve_relative_entries_and_preserve_order(self):
        batch = load_batch()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            playlist = root / "list.m3u8"
            playlist.write_text(
                "#EXTM3U\nMusic\\A.flac\n#EXTINF:1,B\nMusic\\B.flac\n",
                encoding="utf-8",
            )
            self.assertEqual(
                batch.playlist_paths(playlist),
                [root / "Music" / "A.flac", root / "Music" / "B.flac"],
            )

    def test_probe_embedded_lyrics_finds_case_insensitive_lyrics_tag(self):
        batch = load_batch()

        def runner(args, **kwargs):
            payload = {
                "format": {
                    "tags": {
                        "ARTIST": "Artist",
                        "TITLE": "Song",
                        "LYRICS": "[00:01.00]line",
                    }
                }
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

        info = batch.probe_embedded_lyrics(Path("song.flac"), runner=runner)
        self.assertTrue(info.has_lyrics)
        self.assertEqual(info.artist, "Artist")
        self.assertEqual(info.title, "Song")
        self.assertEqual(info.timed_lines, 1)

    def test_render_markdown_compares_reference_and_generated_state_without_lyrics(self):
        batch = load_batch()
        rows = [
            {
                "index": 1,
                "artist": "A",
                "title": "Embedded",
                "status": "embedded_lyrics",
                "reference_lines": 42,
                "source": "",
                "line_count": 0,
                "coverage": None,
                "output": "",
                "error": "",
            },
            {
                "index": 2,
                "artist": "B",
                "title": "Generated",
                "status": "generated",
                "reference_lines": 0,
                "source": "netease",
                "line_count": 40,
                "coverage": 1.0,
                "output": "txt/Generated.lyrics.txt",
                "error": "",
            },
        ]
        markdown = batch.render_markdown(rows, playlist=Path("DF.m3u8"))
        self.assertIn("embedded_lyrics", markdown)
        self.assertIn("generated", markdown)
        self.assertIn("100.0%", markdown)
        self.assertNotIn("[00:01.00]", markdown)


if __name__ == "__main__":
    unittest.main()
