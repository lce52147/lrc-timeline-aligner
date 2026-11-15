from __future__ import annotations

import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

import review_server


def _proc(alive=True):
    return mock.Mock(pid=4321, poll=mock.Mock(return_value=None if alive else 1))


class KillProcessTreeTests(unittest.TestCase):
    def test_windows_runs_taskkill_tree_without_window(self) -> None:
        proc = _proc()
        with mock.patch.object(review_server.os, "name", "nt"), \
                mock.patch.object(review_server.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            review_server._kill_process_tree(proc)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["taskkill", "/PID", "4321", "/T", "/F"])
        self.assertEqual(kwargs["creationflags"], 0x08000000)
        self.assertTrue(kwargs["capture_output"])
        self.assertEqual(kwargs["timeout"], 15)
        proc.terminate.assert_not_called()

    def test_windows_taskkill_failure_falls_back_to_terminate(self) -> None:
        for outcome in (SimpleNamespace(returncode=128), OSError("x"), subprocess.TimeoutExpired("taskkill", 15)):
            proc = _proc()
            kwargs = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
            with mock.patch.object(review_server.os, "name", "nt"), \
                    mock.patch.object(review_server.subprocess, "run", **kwargs):
                review_server._kill_process_tree(proc)
            proc.terminate.assert_called_once_with()

    def test_non_windows_terminates(self) -> None:
        proc = _proc()
        with mock.patch.object(review_server.os, "name", "posix"), \
                mock.patch.object(review_server.subprocess, "run") as run:
            review_server._kill_process_tree(proc)
        run.assert_not_called()
        proc.terminate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
