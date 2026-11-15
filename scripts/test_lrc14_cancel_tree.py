from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from pathlib import Path
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
        with mock.patch.dict(review_server.os.environ, {"SystemRoot": r"C:\Win"}), \
                mock.patch.object(review_server.os, "name", "nt"), \
                mock.patch.object(review_server.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run2:
            review_server._kill_process_tree(_proc())
        self.assertEqual(run2.call_args[0][0][0], review_server.os.path.join(r"C:\Win", "System32", "taskkill.exe"))
        self.assertEqual(args[0][1:], ["/PID", "4321", "/T", "/F"])
        self.assertTrue(args[0][0].lower().endswith("taskkill.exe"))
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

    def test_an_already_finished_process_is_not_killed(self) -> None:
        proc = _proc(alive=False)
        with mock.patch.object(review_server.os, "name", "nt"), mock.patch.object(review_server.subprocess, "run") as run:
            review_server._kill_process_tree(proc)
        run.assert_not_called()

    def test_non_windows_terminates(self) -> None:
        proc = _proc()
        with mock.patch.object(review_server.os, "name", "posix"), \
                mock.patch.object(review_server.subprocess, "run") as run:
            review_server._kill_process_tree(proc)
        run.assert_not_called()
        proc.terminate.assert_called_once_with()


class _Store:
    def add_target(self, audio):
        pass


class CancelStartRaceTests(unittest.TestCase):
    def _runner(self, folder):
        runner = review_server.JobRunner(_Store(), folder)
        job = {"id": "job1", "items": [{"audio": str(Path(folder) / "a.flac"), "name": "a", "status": "queued", "overwrite": False}]}
        runner.jobs.append(job)
        return runner, job

    def test_cancel_between_next_item_and_running_never_starts_the_process(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            runner, job = self._runner(folder)
            picked, index = runner._next_item()
            self.assertEqual(runner.cancel("job1")[0], 200)
            self.assertEqual(job["items"][0]["status"], "cancelled")
            with mock.patch.object(review_server.subprocess, "Popen") as popen:
                runner._run_item(picked, index)
            popen.assert_not_called()
            self.assertEqual(job["items"][0]["status"], "cancelled")

    def test_cancelled_job_id_alone_also_blocks_the_start(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            runner, job = self._runner(folder)
            runner.cancelled.add("job1")
            with mock.patch.object(review_server.subprocess, "Popen") as popen:
                runner._run_item(job, 0)
            popen.assert_not_called()
            self.assertEqual(job["items"][0]["status"], "queued")

    def test_cancel_after_popen_before_registration_kills_the_process(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            runner, job = self._runner(folder)
            proc = mock.Mock(pid=77, stdout=io.StringIO(""))
            proc.wait.return_value = 1

            def popen(*_args, **_kwargs):
                runner.cancel("job1")   # runs while current is still None: nothing to kill yet
                return proc
            with mock.patch.object(review_server.subprocess, "Popen", side_effect=popen), \
                mock.patch.object(review_server, "_align_command", return_value=["x"]), \
                mock.patch.object(review_server, "_kill_process_tree") as kill:
                runner._run_item(job, 0)
            kill.assert_called_once_with(proc)
            self.assertEqual(job["items"][0]["status"], "cancelled")

    def test_stdout_is_closed_after_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            runner, job = self._runner(folder)
            proc = mock.Mock(pid=77, stdout=io.StringIO("line\n"))
            proc.wait.return_value = 1
            with mock.patch.object(review_server.subprocess, "Popen", return_value=proc), \
                mock.patch.object(review_server, "_align_command", return_value=["x"]):
                runner._run_item(job, 0)
            self.assertTrue(proc.stdout.closed)


if __name__ == "__main__":
    unittest.main()
