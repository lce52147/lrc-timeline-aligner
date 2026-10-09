"""Regression for Demucs Windows-unsafe filenames; no GPU/network use."""
import re
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import auto_lrc


class DemucsSafeFilenameTests(unittest.TestCase):
    def test_vocal_ctc_uses_safe_stem_and_preserves_cache_identity(self):
        for filename in ["03.ぼうやの夢よ .flac", "title..flac", "ＡＢＣ：Ｑ ［！］.flac"]:
            with self.subTest(filename=filename), TemporaryDirectory() as tmp:
                root = Path(tmp)
                audio = root / filename
                audio.write_bytes(b"original audio bytes")
                py = root / "python.exe"
                py.touch()
                cache_root = root / "cache"
                called = []

                def fake_run(argv, **kw):
                    staged = Path(argv[-1])
                    self.assertNotEqual(staged, audio)
                    self.assertTrue(staged.exists())
                    self.assertEqual(staged.read_bytes(), audio.read_bytes())
                    self.assertRegex(staged.stem, r"^audio_[A-Za-z0-9_-]+$")
                    self.assertNotIn(" ", staged.stem)
                    self.assertEqual(argv[argv.index("--shifts") + 1], "0")
                    called.append(staged)
                    output_dir = Path(argv[argv.index("-o") + 1])
                    produced = output_dir / "htdemucs" / staged.stem / "vocals.mp3"
                    produced.parent.mkdir(parents=True)
                    produced.write_bytes(b"vocal bytes")
                    return subprocess.CompletedProcess(argv, 0, "", "")

                with (
                    mock.patch.object(auto_lrc, "default_ctc_python", return_value=py),
                    mock.patch.object(auto_lrc, "vocal_cache_identity", return_value={"vocal_cache_key":"same-key"}) as identity,
                    mock.patch.object(auto_lrc, "DEFAULT_VOCAL_CACHE_DIR", cache_root),
                    mock.patch.object(auto_lrc, "_demucs_vocal_stem_valid", return_value=True),
                    mock.patch.object(auto_lrc.subprocess, "run", side_effect=fake_run),
                ):
                    actual, source, note = auto_lrc.prepare_vocal_ctc_audio(audio, type("Args", (), {"vocal_ctc": True})())
                    self.assertEqual((source, note), ("vocal-stem", "generated"))
                    self.assertEqual(actual, cache_root / "same-key" / "vocals.mp3")
                    self.assertEqual(actual.read_bytes(), b"vocal bytes")
                    self.assertEqual(audio.read_bytes(), b"original audio bytes")
                    identity.assert_called_once_with(audio, py)
                    again = auto_lrc.prepare_vocal_ctc_audio(audio, type("Args", (), {"vocal_ctc": True})())
                    self.assertEqual(again[2], "cache-hit")
                self.assertEqual(len(called), 1)
                self.assertFalse(called[0].exists(), "temporary alias should be cleaned up")

    def test_vocal_onset_uses_safe_staged_input(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "trailing .flac"
            audio.write_bytes(b"original")
            py = root / "python.exe"
            py.touch()

            def fake_run(argv, **kw):
                stage = Path(argv[-1])
                self.assertTrue(stage.exists())
                self.assertTrue(stage.stem.isascii())
                self.assertTrue(re.fullmatch(r"audio_[A-Za-z0-9_-]+", stage.stem))
                out = Path(argv[argv.index("-o") + 1]) / "htdemucs" / stage.stem / "vocals.mp3"
                out.parent.mkdir(parents=True)
                out.write_bytes(b"vocal")
                return subprocess.CompletedProcess(argv, 0, "", "")

            with (
                mock.patch.object(auto_lrc, "default_ctc_python", return_value=py),
                mock.patch.object(auto_lrc.subprocess, "run", side_effect=fake_run),
                mock.patch.object(auto_lrc, "analyze_vocal_onsets", return_value="features"),
            ):
                result = auto_lrc.vocal_onset_features(audio, 90.0, type("Args", (), {"vocal_onset_refine": True})())
            self.assertEqual(result, ("features", None))
            self.assertEqual(audio.read_bytes(), b"original")

    def test_windows_reserved_tag_is_never_used_as_a_folder_stem(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "normal.flac"
            src.write_bytes(b"source")
            staged = auto_lrc._stage_demucs_input(src, root, "CON")
            self.assertEqual(staged.name, "audio_CON.flac")
            self.assertEqual(staged.read_bytes(), src.read_bytes())

    def test_unsupported_extension_fails_explicitly(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "file.歌"