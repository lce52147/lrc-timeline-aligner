from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

import auto_lrc


def lock_is_held(path: Path) -> bool:
    """True when another handle already holds the byte-range lock."""
    with path.open("a+b", buffering=0) as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            return True
    return False


class Failed:
    returncode = 1
    stdout = ""
    stderr = "fake whisper.cpp failure"


class WhisperCppGpuLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.lock_path = root / "gpu.lock"
        patcher = mock.patch.object(auto_lrc, "WHISPERCPP_GPU_LOCK_PATH", self.lock_path)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"LRC_WHISPERCPP_GPU_LOCK": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.cli = root / "whisper-cli.exe"
        self.model = root / "model.bin"
        self.cli.write_bytes(b"x")
        self.model.write_bytes(b"x")
        self.args = type(
            "Args", (), {"whisper_cli": str(self.cli), "whisper_model": str(self.model), "whisper_language": "ja"}
        )()

    def test_a_second_process_waits_for_the_holder(self) -> None:
        holder = textwrap.dedent(
            f"""
            import os, sys, time
            path = {str(self.lock_path)!r}
            handle = open(path, "a+b", buffering=0)
            if handle.tell() == 0:
                handle.write(b"0")
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            print("held", flush=True)
            time.sleep(1.5)
            """
        )
        child = subprocess.Popen([sys.executable, "-c", holder], stdout=subprocess.PIPE, text=True)
        self.assertEqual(child.stdout.readline().strip(), "held")
        started = time.monotonic()
        with auto_lrc.whispercpp_gpu_lock():
            waited = time.monotonic() - started
        child.wait(timeout=10)
        child.stdout.close()
        self.assertGreater(waited, 0.8)

    def test_single_file_run_holds_the_lock_during_whisper_cpp(self) -> None:
        seen: list[bool] = []

        def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
            seen.append(lock_is_held(self.lock_path))
            return Failed()

        with mock.patch.object(auto_lrc, "run_command", fake_run):
            with self.assertRaises(auto_lrc.LrcError):
                auto_lrc._run_whispercpp_wav(Path(self.temp.name) / "a.wav", self.args, suppress_nst=False)
        self.assertEqual(seen, [True])
        self.assertFalse(lock_is_held(self.lock_path))

    def test_batch_run_holds_the_lock_for_every_whisper_cpp_call(self) -> None:
        seen: list[bool] = []

        def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
            seen.append(lock_is_held(self.lock_path))
            return Failed()

        items = [("a", Path(self.temp.name) / "a.wav"), ("b", Path(self.temp.name) / "b.wav")]
        with mock.patch.object(auto_lrc, "run_command", fake_run), \
                mock.patch.object(auto_lrc, "_whispercpp_multifile_supported", return_value=True):
            results, errors = auto_lrc._run_whispercpp_wav_batch(items, self.args, suppress_nst=False)
        # One batch call, then the one-shot fallback for each file.
        self.assertEqual(seen, [True, True, True])
        self.assertEqual(results, {})
        self.assertEqual(sorted(errors), ["a", "b"])
        self.assertFalse(lock_is_held(self.lock_path))

    def test_resident_helpers_are_parked_before_whisper_cpp_starts(self) -> None:
        events: list[str] = []

        def fake_park(mode, *, reason):  # type: ignore[no-untyped-def]
            events.append(f"park:{mode}:{reason}")
            return 1

        def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
            events.append("whisper.cpp")
            return Failed()

        with mock.patch.object(auto_lrc, "_park_persistent_helper_mode", fake_park), \
                mock.patch.object(auto_lrc, "run_command", fake_run):
            with self.assertRaises(auto_lrc.LrcError):
                auto_lrc._run_whispercpp_wav(Path(self.temp.name) / "a.wav", self.args, suppress_nst=False)
        self.assertEqual(
            events,
            ["park:whisperx:before-whispercpp", "park:ctc:before-whispercpp", "whisper.cpp"],
        )

    def test_environment_can_disable_the_lock(self) -> None:
        seen: list[bool] = []

        def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
            seen.append(lock_is_held(self.lock_path))
            return Failed()

        with mock.patch.dict(os.environ, {"LRC_WHISPERCPP_GPU_LOCK": "0"}), \
                mock.patch.object(auto_lrc, "run_command", fake_run):
            with self.assertRaises(auto_lrc.LrcError):
                auto_lrc._run_whispercpp_wav(Path(self.temp.name) / "a.wav", self.args, suppress_nst=False)
        self.assertEqual(seen, [False])


if __name__ == "__main__":
    unittest.main()
