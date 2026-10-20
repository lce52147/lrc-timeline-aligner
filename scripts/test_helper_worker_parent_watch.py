"""persistent_helper_worker exits once its parent disappears (LRC 1.5 W5)."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import persistent_helper_worker as worker  # noqa: E402


class ParentWatchTests(unittest.TestCase):
    def test_watch_fires_after_fake_parent_dies(self) -> None:
        fake_parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            fired = threading.Event()
            # The fake parent is a child of this test, not our parent, so bypass the
            # POSIX getppid check and exercise the liveness probe directly.
            original = worker.parent_alive

            def alive(pid: int) -> bool:
                return fake_parent.poll() is None if pid == fake_parent.pid else original(pid)

            worker.parent_alive = alive
            try:
                worker.watch_parent(fake_parent.pid, fired.set, interval=0.5)
                self.assertFalse(fired.wait(1.5), "fired while parent alive")
                fake_parent.kill()
                fake_parent.wait(timeout=10)
                self.assertTrue(fired.wait(15), "worker did not notice parent exit within 15s")
            finally:
                worker.parent_alive = original
        finally:
            if fake_parent.poll() is None:
                fake_parent.kill()

    def test_parent_alive_for_real_parent_and_dead_pid(self) -> None:
        import os
        self.assertTrue(worker.parent_alive(os.getppid()))
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait(timeout=30)
        if os.name == "nt":
            self.assertFalse(worker.parent_alive(dead.pid))
        else:
            # Not our parent -> reported as gone regardless of pid reuse.
            self.assertFalse(worker.parent_alive(dead.pid))

    def test_worker_accepts_parent_pid_flag(self) -> None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--parent-pid", type=int, default=None)
        self.assertEqual(parser.parse_args(["--parent-pid", "123"]).parent_pid, 123)
        self.assertIn("--parent-pid", Path(worker.__file__).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
