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
                    *args, str(path)], check=True, creationflags=0x08004000)


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

    def test_changed_audio_payload_restores_the_original_bytes(self) -> None:
        for make, name in ((self._flac, "LYRICS"), (self._m4a, "\xa9lyr")):
            with self.subTest(name=name):
                path = make()
                original = path.read_bytes()
                real = lyrics_tag_export._payload_digest
                calls = []

                def fake(target, tag):
                    calls.append(target)
                    return real(target, tag) if len(calls) == 1 else "0" * 64

                with mock.patch.object(lyrics_tag_export, "_payload_digest", side_effect=fake), \
                        mock.patch.object(lyrics_tag_export, "_decoded_md5") as decoded:
                    with self.assertRaisesRegex(lyrics_tag_export.ExportError, "音訊內容改變.*已還原"):
                        lyrics_tag_export.export(path, LRC, self.work)
                decoded.assert_not_called()
                self.assertEqual(path.read_bytes(), original)

    def test_real_payload_flip_is_caught_byte_for_byte(self) -> None:
        path = self._flac()
        original = path.read_bytes()
        real_write = lyrics_tag_export._write_tag

        def write_and_flip(target, name, text):
            real_write(target, name, text)
            data = bytearray(Path(target).read_bytes())
            data[-5] ^= 0xFF
            Path(target).write_bytes(bytes(data))

        with mock.patch.object(lyrics_tag_export, "_write_tag", side_effect=write_and_flip):
            with self.assertRaisesRegex(lyrics_tag_export.ExportError, "已還原"):
                lyrics_tag_export.export(path, LRC, self.work)
        self.assertEqual(path.read_bytes(), original)

    def test_payload_digest_ignores_tags_but_not_audio(self) -> None:
        flac, m4a = self._flac(), self._m4a()
        before = (lyrics_tag_export._payload_digest(flac, "LYRICS"), lyrics_tag_export._payload_digest(m4a, "\xa9lyr"))
        self.assertTrue(all(len(item) == 64 for item in before))
        lyrics_tag_export._write_tag(flac, "LYRICS", "x" * 5000)
        lyrics_tag_export._write_tag(m4a, "\xa9lyr", "x" * 5000)
        self.assertEqual((lyrics_tag_export._payload_digest(flac, "LYRICS"), lyrics_tag_export._payload_digest(m4a, "\xa9lyr")), before)

    def test_unwalkable_container_falls_back_to_the_ffmpeg_decode(self) -> None:
        path = self._flac()
        with mock.patch.object(lyrics_tag_export, "_payload_digest", return_value=None), \
                mock.patch.object(lyrics_tag_export, "_decoded_md5", return_value="a" * 32) as decoded:
            snap = lyrics_tag_export._snapshot(path, "LYRICS")
            self.assertEqual(snap["audio_md5"], "a" * 32)
            self.assertNotIn("audio_payload", snap)
            lyrics_tag_export.export(path, LRC, self.work)
        self.assertGreaterEqual(decoded.call_count, 3)
        self.assertEqual(mutagen.flac.FLAC(str(path))["LYRICS"], [EXPECTED])

    def test_ffmpeg_decode_never_pops_a_window(self) -> None:
        path = self._flac()
        with mock.patch.object(lyrics_tag_export.subprocess, "run", wraps=subprocess.run) as run:
            lyrics_tag_export._decoded_md5(path)
        self.assertEqual(run.call_args.kwargs["creationflags"], lyrics_tag_export.NO_WINDOW)
        if sys.platform == "win32":
            self.assertEqual(lyrics_tag_export.NO_WINDOW, 0x08000000)

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


class ArchiveLrcTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lrc = self.root / "song.lrc"
        self.lrc.write_text("[00:01.00]x", encoding="utf-8")
        self.work = self.root / "outputs"
        self.work.mkdir()
        self.dest = self.root / "bk"

    def _log(self):
        return [json.loads(line) for line in (self.work / lyrics_tag_export.LOG_NAME).read_text(encoding="utf-8").splitlines()]

    def test_move_to_folder_and_log(self) -> None:
        result = lyrics_tag_export.archive_lrc(self.lrc, self.work, "song.flac", ("move", self.dest))
        self.assertEqual(result, {"mode": "move", "to": str(self.dest / "song.lrc")})
        self.assertFalse(self.lrc.exists())
        self.assertEqual((self.dest / "song.lrc").read_text(encoding="utf-8"), "[00:01.00]x")
        self.assertEqual(self._log()[-1]["archive"], "move")

    def test_collision_gets_a_timestamp_suffix_and_never_overwrites(self) -> None:
        self.dest.mkdir()
        (self.dest / "song.lrc").write_text("older", encoding="utf-8")
        result = lyrics_tag_export.archive_lrc(self.lrc, self.work, "song.flac", ("move", self.dest))
        self.assertRegex(Path(result["to"]).name, r"^song_\d{8}-\d{6}\.lrc$")
        self.assertEqual((self.dest / "song.lrc").read_text(encoding="utf-8"), "older")
        self.assertEqual(Path(result["to"]).read_text(encoding="utf-8"), "[00:01.00]x")

    def test_off_leaves_the_file_and_logs_nothing(self) -> None:
        result = lyrics_tag_export.archive_lrc(self.lrc, self.work, "song.flac", ("off", None))
        self.assertEqual(result, {"mode": "off"})
        self.assertTrue(self.lrc.is_file())
        self.assertFalse((self.work / lyrics_tag_export.LOG_NAME).exists())

    def test_only_lrc_files_are_ever_moved(self) -> None:
        text = self.root / "song.lyrics.txt"
        text.write_text("words", encoding="utf-8")
        result = lyrics_tag_export.archive_lrc(text, self.work, "song.flac", ("move", self.dest))
        self.assertEqual(result["mode"], "failed")
        self.assertTrue(text.is_file())

    def test_move_failure_keeps_the_lrc_and_reports_it(self) -> None:
        with mock.patch.object(lyrics_tag_export.shutil, "move", side_effect=OSError("denied")):
            result = lyrics_tag_export.archive_lrc(self.lrc, self.work, "song.flac", ("move", self.dest))
        self.assertEqual(result["mode"], "failed")
        self.assertTrue(self.lrc.is_file())

    def test_delete_uses_the_recycle_bin(self) -> None:
        with mock.patch.object(lyrics_tag_export, "_recycle", side_effect=lambda path: Path(path).unlink() or True) as recycle:
            result = lyrics_tag_export.archive_lrc(self.lrc, self.work, "song.flac", ("delete", None))
        recycle.assert_called_once()
        self.assertEqual(result, {"mode": "delete"})
        self.assertFalse(self.lrc.exists())

    def test_setting_parsing(self) -> None:
        setting = runtime_paths.export_archive_setting
        self.assertEqual(setting({}), ("move", runtime_paths.project_root() / "checked"))
        self.assertEqual(setting({"LRC_EXPORT_ARCHIVE_DIR": ""}), ("off", None))
        self.assertEqual(setting({"LRC_EXPORT_ARCHIVE_DIR": " OFF "}), ("off", None))
        self.assertEqual(setting({"LRC_EXPORT_ARCHIVE_DIR": "delete"}), ("delete", None))
        self.assertEqual(setting({"LRC_EXPORT_ARCHIVE_DIR": str(self.dest)}), ("move", self.dest))
        self.assertEqual(setting({"LRC_EXPORT_ARCHIVE_DIR": "bk"}), ("move", runtime_paths.project_root() / "bk"))

    @unittest.skipIf(sys.platform != "win32", "Recycle Bin is Windows-only")
    def test_real_recycle_bin_round_trip(self) -> None:
        result = lyrics_tag_export.archive_lrc(self.lrc, self.work, "song.flac", ("delete", None))
        self.assertEqual(result, {"mode": "delete"})
        self.assertFalse(self.lrc.exists())


if __name__ == "__main__":
    unittest.main()
