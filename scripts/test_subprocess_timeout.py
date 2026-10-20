"""_run_logged kills a hung reviewer and reports TIMEOUT (LRC 1.5 W5)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import r2_pipeline  # noqa: E402


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); check its state.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
        return state != "Z"
    except OSError:
        return True


class SubprocessTimeoutTests(unittest.TestCase):
    def test_hung_reviewer_is_killed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "reviewers.log"
            started = time.monotonic()
            code = r2_pipeline._run_logged(
                [sys.executable, "-c", "import time; time.sleep(60)"], log, timeout=2,
            )
            self.assertEqual(code, r2_pipeline.EXIT_TIMEOUT)
            self.assertLess(time.monotonic() - started, 40)
            text = log.read_text(encoding="utf-8")
            match = re.search(r"timeout: killed pid (\d+)", text)
            self.assertIsNotNone(match, text)
            pid = int(match.group(1))
            deadline = time.monotonic() + 10
            while _pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.2)
            self.assertFalse(_pid_alive(pid), "timed-out child still running")
            self.assertEqual(r2_pipeline.reviewer_step_status("WX", code), "TIMEOUT")

    def test_launch_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "reviewers.log"
            code = r2_pipeline._run_logged([str(Path(tmp) / "no-such-binary")], log, timeout=5)
            self.assertEqual(code, r2_pipeline.EXIT_LAUNCH_ERROR)
            self.assertEqual(r2_pipeline.reviewer_step_status("WX", code), "LAUNCH_ERROR")

    def test_normal_exit_code_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "reviewers.log"
            code = r2_pipeline._run_logged([sys.executable, "-c", "import sys; print('hi'); sys.exit(3)"], log, timeout=30)
            self.assertEqual(code, 3)
            self.assertIn("hi", log.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
