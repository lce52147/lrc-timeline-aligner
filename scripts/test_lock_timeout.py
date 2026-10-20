"""File-lock waits give up after LRC_LOCK_TIMEOUT_SECONDS with LrcError (LRC 1.5 W5)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc  # noqa: E402

HOLDER = """
import sys, time
sys.path.insert(0, {scripts!r})
import auto_lrc
from pathlib import Path
with auto_lrc._exclusive_file_lock(Path({path!r}), timeout_seconds=5, busy_message="holder"):
    print("held", flush=True)
    time.sleep(30)
"""


class LockTimeoutTests(unittest.TestCase):
    def test_env_default_and_override(self) -> None:
        self.assertEqual(auto_lrc._lock_timeout_seconds({}), 1800.0)
        self.assertEqual(auto_lrc._lock_timeout_seconds({"LRC_LOCK_TIMEOUT_SECONDS": "7"}), 7.0)
        self.assertIsNone(auto_lrc._lock_timeout_seconds({"LRC_LOCK_TIMEOUT_SECONDS": "0"}))

    def test_held_lock_times_out_with_lrc_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / "gpu.lock"
            holder = subprocess.Popen(
                [sys.executable, "-c", HOLDER.format(scripts=str(SCRIPTS), path=str(lock_path))],
                stdout=subprocess.PIPE, text=True,
            )
            try:
                self.assertEqual((holder.stdout.readline() or "").strip(), "held")
                started = time.monotonic()
                with self.assertRaises(auto_lrc.LrcError) as ctx:
                    with auto_lrc._exclusive_file_lock(lock_path, timeout_seconds=1, busy_message="test"):
                        pass
                self.assertLess(time.monotonic() - started, 10)
                self.assertIn(f"lock-timeout:{lock_path}", str(ctx.exception))
            finally:
                holder.kill()
                holder.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
