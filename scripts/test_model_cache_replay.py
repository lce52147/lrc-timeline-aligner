#!/usr/bin/env python3
"""Model-output cache: batch whisper.cpp reuse and cache-only launch refusal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

# Unit tests were written against uncached helpers and mock them per test; a
# persisted entry from an earlier run must never answer for a mocked helper.
# unittest discovery imports every module before running any test, so this
# applies to the whole suite.  Tests below opt in with a temporary cache dir.
os.environ.setdefault("LRC_MODEL_CACHE", "0")

import auto_lrc
from auto_lrc import AsrSegment, LyricEntry, ModelCacheMiss


class ModelCacheReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.td = TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        root = Path(self.td.name)
        self.cli = root / "whisper-cli.exe"
        self.cli.write_bytes(b"cli")
        self.model = root / "ggml.bin"
        self.model.write_bytes(b"model")
        self.wavs = []
        for name in ("a", "b"):
            wav = root / name / "input.wav"
            wav.parent.mkdir()
            wav.write_bytes(f"pcm-{name}".encode())
            self.wavs.append((name, wav))
        self.args = argparse.Namespace(
            whisper_cli=str(self.cli), whisper_model=str(self.model),
            whisper_language="ja", model_cache=True, model_cache_only=False,
        )
        for patch in (
            mock.patch.object(auto_lrc, "DEFAULT_MODEL_CACHE_DIR", root / "model-outputs"),
            mock.patch.object(auto_lrc, "_whispercpp_multifile_supported", return_value=True),
            mock.patch.object(auto_lrc, "_gpu_coordination_enabled", return_value=False),
            mock.patch.dict(os.environ, {"LRC_MODEL_CACHE": "1", "LRC_MODEL_CACHE_ONLY": "0"}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def fake_batch_run(self, command, **_kwargs):
        for index, token in enumerate(command):
            if token == "-of":
                base = Path(command[index + 1])
                base.with_suffix(".json").write_text(json.dumps({"transcription": [
                    {"offsets": {"from": 1000, "to": 2500}, "text": f"text-{base.parent.name}"},
                ]}), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    def test_batch_fills_cache_then_cache_only_replays_without_launch(self) -> None:
        with mock.patch.object(auto_lrc, "run_command", side_effect=self.fake_batch_run) as run:
            first, errors = auto_lrc._run_whispercpp_wav_batch(self.wavs, self.args, suppress_nst=False)
        self.assertEqual(errors, {})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(first["a"], [AsrSegment(1.0, 2.5, "text-a")])
        cache_root = Path(auto_lrc.DEFAULT_MODEL_CACHE_DIR)
        before = {
            str(path.relative_to(cache_root)): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in cache_root.rglob("*")
            if path.is_file()
        }

        with mock.patch.dict(os.environ, {"LRC_MODEL_CACHE_ONLY": "1"}), \
                mock.patch.object(auto_lrc, "run_command", side_effect=AssertionError("launched")):
            replay, errors = auto_lrc._run_whispercpp_wav_batch(self.wavs, self.args, suppress_nst=False)
        self.assertEqual(errors, {})
        self.assertEqual(replay, first)
        after = {
            str(path.relative_to(cache_root)): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in cache_root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_cache_only_miss_is_not_an_ordinary_failure(self) -> None:
        self.assertFalse(issubclass(ModelCacheMiss, Exception))
        with mock.patch.dict(os.environ, {"LRC_MODEL_CACHE_ONLY": "1"}), \
                mock.patch.object(auto_lrc, "run_command", side_effect=AssertionError("launched")):
            with self.assertRaises(ModelCacheMiss):
                auto_lrc._run_whispercpp_wav_batch(self.wavs, self.args, suppress_nst=False)

    def test_cache_only_refuses_every_gpu_entry_point(self) -> None:
        with mock.patch.dict(os.environ, {"LRC_MODEL_CACHE_ONLY": "1"}):
            for phase in (auto_lrc.gpu_compute_phase, auto_lrc.whispercpp_gpu_lock):
                with self.assertRaises(ModelCacheMiss):
                    with phase():
                        self.fail("model launch was allowed")

    def test_cli_flags_do_not_clear_inherited_cache_only(self) -> None:
        args = auto_lrc.build_parser().parse_args(["song.flac"])
        with mock.patch.dict(os.environ, {"LRC_MODEL_CACHE_ONLY": "1"}):
            self.assertEqual(auto_lrc._apply_model_cache_flags(args), "cache-only")
            self.assertEqual(os.environ["LRC_MODEL_CACHE_ONLY"], "1")
        args = auto_lrc.build_parser().parse_args(["song.flac", "--model-cache-only", "--no-model-cache"])
        with self.assertRaises(auto_lrc.LrcError):
            auto_lrc._apply_model_cache_flags(args)


    def test_cache_only_missing_raw_backend_fails_before_ctc(self) -> None:
        for missing_name in ("whisper_cli", "whisper_model"):
            with self.subTest(missing_name=missing_name):
                args = argparse.Namespace(**vars(self.args))
                args.model_cache_only = True
                setattr(args, missing_name, str(Path(self.td.name) / f"missing-{missing_name}"))
                with mock.patch.object(auto_lrc, "try_ctc_candidate", side_effect=AssertionError("CTC reached")):
                    with self.assertRaisesRegex(ModelCacheMiss, "raw-ASR-backend-unavailable"):
                        auto_lrc.run_auto_backend_competition(
                            self.wavs[0][1], [], 1.0, args, None
                        )

    def test_cache_only_missing_whisperx_python_fails_before_whisperx(self) -> None:
        args = argparse.Namespace(**vars(self.args))
        args.model_cache_only = True
        args.whisperx_python = str(Path(self.td.name) / "missing-whisperx-python")
        with mock.patch.object(
            auto_lrc, "try_whispercpp_diagnostic_candidate",
            return_value=([], {"assignments": []}, None, []),
        ), mock.patch.object(auto_lrc, "default_ctc_ready", return_value=False), mock.patch.object(
            auto_lrc, "decode_temp_wav", return_value=None
        ):
            with self.assertRaisesRegex(ModelCacheMiss, "whisperx-python-unavailable"):
                auto_lrc.run_auto_backend_competition(
                    self.wavs[0][1], [LyricEntry(["text"])], 1.0, args, None
                )

    def test_cache_only_raw_execution_failure_is_model_cache_miss(self) -> None:
        args = argparse.Namespace(**vars(self.args))
        args.model_cache_only = True
        with mock.patch.object(auto_lrc, "run_whispercpp", side_effect=auto_lrc.LrcError("CLI failed")):
            with self.assertRaisesRegex(ModelCacheMiss, "model-cache-miss:raw-ASR:CLI failed"):
                auto_lrc.try_whispercpp_diagnostic_candidate(
                    self.wavs[0][1], [LyricEntry(["text"])], 1.0, args
                )

    def test_ordinary_raw_backend_status_is_machine_readable_warning(self) -> None:
        args = argparse.Namespace(**vars(self.args))
        args.whisper_cli = str(Path(self.td.name) / "missing-cli")
        status = auto_lrc._raw_asr_backend_status(args)
        report: dict[str, object] = {}
        auto_lrc._attach_raw_asr_status(report, status)
        self.assertFalse(status["available"])
        self.assertEqual(status["failure_kind"], "backend-unavailable")
        self.assertEqual(status["fallback"], "empty-raw-asr-events")
        self.assertFalse(report["raw_asr_backend"]["available"])
        self.assertIn("raw ASR unavailable", report["raw_asr_warning"])


if __name__ == "__main__":
    unittest.main()
