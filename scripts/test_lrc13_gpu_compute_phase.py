"""Two songs share the GPU: one song's whisper.cpp run and the other song's
CTC/WhisperX/HuBERT/Demucs work never overlap."""
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
from test_lrc13_whispercpp_gpu_lock import lock_is_held


def hold_lock(path: Path, seconds: float) -> subprocess.Popen:
    """Another process (the other song) holding *path* for *seconds*."""
    script = textwrap.dedent(
        f"""
        import os, time
        path = {str(path)!r}
        os.makedirs(os.path.dirname(path), exist_ok=True)
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
        time.sleep({seconds})
        """
    )
    child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    assert child.stdout is not None and child.stdout.readline().strip() == "held"
    return child


def finish(child: subprocess.Popen) -> None:
    child.wait(timeout=10)
    if child.stdout is not None:
        child.stdout.close()


class GpuComputePhaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.whisper_lock = root / "gpu.lock"
        self.compute_dir = root / "compute"
        for name, value in (("WHISPERCPP_GPU_LOCK_PATH", self.whisper_lock), ("GPU_COMPUTE_LOCK_DIR", self.compute_dir)):
            patcher = mock.patch.object(auto_lrc, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"LRC_WHISPERCPP_GPU_LOCK": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.parks: list[str] = []
        park = mock.patch.object(
            auto_lrc, "_park_persistent_helper_mode",
            lambda mode, *, reason: self.parks.append(f"{mode}:{reason}") or 1,
        )
        park.start()
        self.addCleanup(park.stop)

    def test_gpu_work_parks_and_waits_while_another_song_runs_whispercpp(self) -> None:
        child = hold_lock(self.whisper_lock, 1.5)
        started = time.monotonic()
        with auto_lrc.gpu_compute_phase():
            waited = time.monotonic() - started
        finish(child)
        self.assertGreater(waited, 0.8)
        self.assertEqual(self.parks, ["whisperx:other-song-whispercpp", "ctc:other-song-whispercpp"])

    def test_whispercpp_waits_for_another_songs_gpu_work(self) -> None:
        child = hold_lock(self.compute_dir / "999999.lock", 1.5)
        started = time.monotonic()
        with auto_lrc.whispercpp_gpu_lock():
            waited = time.monotonic() - started
        finish(child)
        self.assertGreater(waited, 0.8)

    def test_idle_songs_do_not_delay_each_other(self) -> None:
        started = time.monotonic()
        with auto_lrc.gpu_compute_phase():
            pass
        with auto_lrc.whispercpp_gpu_lock():
            pass
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(self.parks, ["whisperx:before-whispercpp", "ctc:before-whispercpp"])

    def test_own_gpu_work_inside_own_whispercpp_does_not_deadlock(self) -> None:
        started = time.monotonic()
        with auto_lrc.whispercpp_gpu_lock():
            with auto_lrc.gpu_compute_phase():
                pass
        self.assertLess(time.monotonic() - started, 0.5)

    def test_helper_requests_run_inside_the_phase(self) -> None:
        marker = self.compute_dir / f"{os.getpid()}.lock"
        seen: list[bool] = []

        def fake_now(command, *, mode, timeout_seconds=None):  # type: ignore[no-untyped-def]
            seen.append(lock_is_held(marker))
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(auto_lrc, "_run_model_helper_command_now", fake_now):
            auto_lrc._run_model_helper_command(["python", "helper.py"], mode="ctc")
        self.assertEqual(seen, [True])
        self.assertFalse(lock_is_held(marker))

    def test_environment_switch_disables_coordination(self) -> None:
        child = hold_lock(self.whisper_lock, 1.5)
        try:
            with mock.patch.dict(os.environ, {"LRC_WHISPERCPP_GPU_LOCK": "0"}):
                started = time.monotonic()
                with auto_lrc.gpu_compute_phase():
                    waited = time.monotonic() - started
        finally:
            finish(child)
        self.assertLess(waited, 0.5)
        self.assertEqual(self.parks, [])


if __name__ == "__main__":
    unittest.main()
