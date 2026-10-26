from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import lyrics_tag_export
import runtime_paths

try:
    import mutagen.flac
    import mutagen.mp4
except ImportError:  # pragma: no cover - the venv installs mutagen from requirements-asr.txt
    mutagen = None

HAVE_FFMPEG = shutil.which(runtime_paths.ffmpeg()) is not None

LRC = "﻿[ti:Song]\r\n[re:LRC tools ctc-forced-align]\n[00:01.00]一行\n[00:02.50]二行\n".encode("utf-8")
EXPECTED = "[ti:Song]\r\n[re:LRC tools ctc-forced-align]\r\n[00:01.00]一行\r\n[00:02.50]二行"


def _make(path: Path, *args: str) -> None:
    subprocess.run([runtime_paths.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                    *args, str(path)], check=True)


@unittest.skipIf(mutagen is None or not HAVE_FFMPEG, "needs mutagen and ffmpeg")
class LyricsTagExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "outputs"

    def _flac(self, with_lyrics: bool = True) -> Path:
        path = self.root / "song.flac"
        _make(path, "-c:a", "flac")
        handle = mutagen.flac.FLAC(str(path))
        handle["TITLE"] = ["Song"]
        handle["ARTIST"] = ["Artist", "Second"]
        if with_lyrics:
            handle["lyrics"] = ["old line one\r\nold line two"]
        picture = mutagen.flac.Picture()
        picture.type, picture.mime, picture.data = 3, "image/png", b"\x89PNG-not-really" * 20
        handle.add_picture(picture)
        handle.save()
        return path

    def _m4a(self) -> Path:
        path = self.root / "song.m4a"
        _make(path, "-c:a", "aac")
        handle = mutagen.mp4.MP4(str(path))
        handle["\xa9nam"] = ["Song"]
        handle["\xa9lyr"] = ["old lyrics"]
        handle["covr"] = [mutagen.mp4.MP4Cover(b"\x89PNG-not-really" * 20, mutagen.mp4.MP4Cover.FORMAT_PNG)]
        handle.save()
        return path

    def test_text_is_crlf_without_bom_or_trailing_break(self) -> None:
        self.assertEqual(lyrics_tag_export.lrc_text(LRC), EXPECTED)

    def test_flac_round_trip_preserves_everything_else(self) -> None:
        path = self._flac()
        before = lyrics_tag_export._snapshot(path, "LYRICS")
        result = lyrics_tag_export.export(path, LRC, self.work)
        self.assertEqual(result, {"name": "song.flac", "tag": "LYRICS", "lines": 4})
        handle = mutagen.flac.FLAC(str(path))
        self.assertEqual(handle["LYRICS"], [EXPECTED])
        self.assertEqual(len(handle["lyrics"]), 1, "the old LYRICS is replaced, not duplicated")
        self.assertEqual(lyrics_tag_export._snapshot(path, "LYRICS"), before)
        self.assertEqual(handle["ARTIST"], ["Artist", "Second"])
        self.assertEqual(len(handle.pictures), 1)
        self.assertEqual(list((self.work / lyrics_tag_export.CACHE_NAME).iterdir()), [], "temp copy removed on success")

    def test_flac_without_existing_lyrics(self) -> None:
        path = self._flac(with_lyrics=False)
        self.assertEqual(lyrics_tag_export.info(path), {"name": "song.flac", "tag": "LYRICS", "has_lyrics": False, "lines": 0})
        lyrics_tag_export.export(path, LRC, self.work)
        self.assertEqual(lyrics_tag_export.info(path), {"name": "song.flac", "tag": "LYRICS", "has_lyrics": True, "lines": 4})
        record = json.loads((self.work / lyrics_tag_export.LOG_NAME).read_text(encoding="utf-8"))
        self.assertEqual(record["old"], "none")

    def test_m4a_round_trip(self) -> None:
        path = self._m4a()
        before = lyrics_tag_export._snapshot(path, "©lyr")
        result = lyrics_tag_export.export(path, LRC, self.work)
        self.assertEqual(result, {"name": "song.m4a", "tag": "©lyr", "lines": 4})
        handle = mutagen.mp4.MP4(str(path))
        self.assertEqual(handle["\xa9lyr"], [EXPECTED])
        self.assertEqual(lyrics_tag_export._snapshot(path, "©lyr"), before)
        self.assertEqual(len(handle["covr"]), 1)

    def test_log_records_the_previous_value(self) -> None:
        path = self._flac()
        lyrics_tag_export.export(path, LRC, self.work)
        record = json.loads((self.work / lyrics_tag_export.LOG_NAME).read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(record["old"], "old line one\r\nold line two")
        self.assertEqual(record["file"], str(path))
        self.assertEqual(record["tag"], "LYRICS")
        self.assertIn("utc", record)

    def test_unsupported_type_is_refused_and_untouched(self) -> None:
        path = self.root / "song.mp3"
        path.write_bytes(b"not touched")
        with self.assertRaisesRegex(lyrics_tag_export.ExportError, "不支援"):
            lyrics_tag_export.export(path, LRC, self.work)
        self.assertEqual(path.read_bytes(), b"not touched")
        self.assertFalse((self.work / lyrics_tag_export.LOG_NAME).exists())

    def test_read_only_file_is_refused_before_anything_is_written(self) -> None:
        import os
        import stat
        path = self._flac()
        original = path.read_bytes()
        os.chmod(path, stat.S_IREAD)
        self.addCleanup(os.chmod, path, stat.S_IREAD | stat.S_IWRITE)
        with self.assertRaisesRegex(lyrics_tag_export.ExportError, "唯讀"):
            lyrics_tag_export.export(path, LRC, self.work)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.work / lyrics_tag_export.LOG_NAME).exists(), "nothing logged for a refused export")

    def test_line_count_accepts_any_line_ending(self) -> None:
        self.assertEqual(lyrics_tag_export._line_count("a\nb\r\nc\rd"), 4)
        self.assertEqual(lyrics_tag_export._line_count(""), 0)

    def test_failed_readback_restores_the_original_bytes(self) -> None:
        path = self._flac()
        original = path.read_bytes()
        with mock.patch.object(lyrics_tag_export, "read_lyrics", side_effect=[ "old", "wrong text" ]):
            with self.assertRaisesRegex(lyrics_tag_export.ExportError, "已還原"):
                lyrics_tag_export.export(path, LRC, self.work)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list((self.work / lyrics_tag_export.CACHE_NAME).iterdir()), [])

    def test_changed_audio_restores_the_original_bytes(self) -> None:
        path = self._flac()
        original = path.read_bytes()
        real = lyrics_tag_export._decoded_md5
        calls = []

        def fake(target):
            calls.append(target)
            return real(target) if len(calls) == 1 else "0" * 32

        with mock.patch.object(lyrics_tag_export, "_decoded_md5", side_effect=fake):
            with self.assertRaisesRegex(lyrics_tag_export.ExportError, "音訊解碼內容改變.*已還原"):
                lyrics_tag_export.export(path, LRC, self.work)
        self.assertEqual(path.read_bytes(), original)

    def test_write_exception_restores_the_original_bytes(self) -> None:
        path = self._flac()
        original = path.read_bytes()

        def corrupt(target, name, text):
            Path(target).write_bytes(b"broken")
            raise OSError("disk full")

        with mock.patch.object(lyrics_tag_export, "_write_tag", side_effect=corrupt):
            with self.assertRaisesRegex(lyrics_tag_export.ExportError, "寫入標籤失敗.*已還原"):
                lyrics_tag_export.export(path, LRC, self.work)
        self.assertEqual(path.read_bytes(), original)

    def test_candidates_lists_the_songs_own_audio_files(self) -> None:
        flac = self._flac()
        m4a = self._m4a()
        (self.root / "other.flac").write_bytes(b"x")
        self.assertEqual([item.name for item in lyrics_tag_export.candidates(m4a)], ["song.m4a", "song.flac"])
        self.assertEqual([item.name for item in lyrics_tag_export.candidates(flac)], ["song.flac", "song.m4a"])


if __name__ == "__main__":
    unittest.main()
