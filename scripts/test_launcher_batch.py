from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CREATE_NO_WINDOW = 0x08000000
BELOW_NORMAL = 0x00004000
FLAGS = CREATE_NO_WINDOW | BELOW_NORMAL if os.name == "nt" else 0

# Loaded through PYTHONPATH: records the command line of every python start and
# exits before the real script runs, so no launcher test starts a real aligner or
# review server. STUB_FAIL_MATCH makes the calls whose arguments contain it exit 1.
STUB_SITECUSTOMIZE = '''\
import os, sys
_out = os.environ.get("STUB_OUT")
if _out:
    _code = 0
    _match = os.environ.get("STUB_FAIL_MATCH")
    if _match and any(_match in a for a in sys.argv[1:]):
        _code = 1
    with open(_out, "a", encoding="utf-8", newline="\\n") as _f:
        _f.write("\\x1f".join(sys.argv) + "\\x1e" + str(_code) + "\\n")
    sys.stdout.flush()
    os._exit(_code)
'''


def read_stub_calls(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    calls = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line:
            calls.append(line.split("\x1e")[0].split("\x1f"))
    return calls


@unittest.skipUnless(os.name == "nt", "the launchers are Windows batch files")
class LauncherStubCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        stub_dir = self.tmp / "stub"
        stub_dir.mkdir()
        (stub_dir / "sitecustomize.py").write_text(STUB_SITECUSTOMIZE, encoding="utf-8")
        self.calls_file = self.tmp / "calls.txt"
        self.env = dict(os.environ)
        for name in ("LRC_VENV_PYTHON", "STUB_FAIL_MATCH", "LRC_REPORT_DIR"):
            self.env.pop(name, None)
        self.env.update(
            STUB_OUT=str(self.calls_file),
            PYTHONPATH=str(stub_dir),
            LRC_NO_PAUSE="1",
            LRC_NO_REVIEW="1",
            LRC_VENV_PYTHON=sys.executable,
            LRC_REPORT_DIR=str(self.tmp / "reports"),
        )

    def run_cmdline(self, command_line: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            command_line,
            env=self.env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=FLAGS,
            stdin=subprocess.DEVNULL,
            timeout=180,
        )

    def run_bat(self, name: str, *args: str) -> subprocess.CompletedProcess:
        # Every argument quoted, like an Explorer drop; cmd /c expands %NAME% here.
        quoted = " ".join(f'"{a}"' for a in args)
        return self.run_cmdline(f'cmd /d /c ""{REPO_ROOT / name}" {quoted}"')


class ReviewBatchPythonChoiceTests(LauncherStubCase):
    def test_review_bat_uses_lrc_venv_python_when_set(self) -> None:
        result = self.run_bat("Review LRC.bat", "--report-dir", str(self.tmp / "reports"))
        calls = read_stub_calls(self.calls_file)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 1, calls)
        self.assertTrue(calls[0][0].endswith("review_server.py"), calls)
        self.assertEqual(calls[0][1:], ["--report-dir", str(self.tmp / "reports")])

    def test_review_bat_ignores_a_missing_or_directory_override(self) -> None:
        for bad in (str(self.tmp / "nope" / "python.exe"), str(self.tmp)):
            with self.subTest(override=bad):
                self.env["LRC_VENV_PYTHON"] = bad
                result = self.run_bat("Review LRC.bat", "--report-dir", "x")
                # Falls back to the default .venv-asr: it runs there (the stub is
                # loaded through PYTHONPATH) or reports it missing.
                if (REPO_ROOT / ".venv-asr" / "Scripts" / "python.exe").exists():
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                else:
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("Python runtime not found", result.stdout)


if __name__ == "__main__":
    unittest.main()
