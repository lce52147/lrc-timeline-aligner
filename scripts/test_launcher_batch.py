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
        self.tmp = Path(self._tmp.name).resolve()  # long form: PowerShell expands 8.3 names
        stub_dir = self.tmp / "stub"
        stub_dir.mkdir()
        (stub_dir / "sitecustomize.py").write_text(STUB_SITECUSTOMIZE, encoding="utf-8")
        (self.tmp / "reports").mkdir()
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

class AlignBatchDroppedNameTests(LauncherStubCase):
    NAMES = [
        "pct%TEMP%x.flac",
        "dbl%%pct.flac",
        "%PATH%.flac",
        "bang!name.flac",
        "amp&name.flac",
        "caret^name.flac",
        "sp ace name.flac",
        "\u4e2d\u6587\u6b4c.flac",
        "\u65e5\u672c\u8a9e\u306e\u6b4c.flac",
    ]

    def make_audio(self, name: str) -> str:
        path = self.tmp / "songs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"")
        return str(path)

    def aligned_paths(self) -> list[str]:
        # The aligner calls end with the song path; the batch summary call does not.
        return [c[-1] for c in read_stub_calls(self.calls_file) if c[0].endswith("r2_pipeline.py")]

    def test_dropped_names_reach_the_aligner_verbatim(self) -> None:
        for name in self.NAMES:
            with self.subTest(name=name):
                self.calls_file.unlink(missing_ok=True)
                path = self.make_audio(name)
                result = self.run_bat("Align LRC.bat", path)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.aligned_paths(), [path])

    def test_several_dropped_files_keep_their_own_names(self) -> None:
        paths = [self.make_audio(n) for n in ("a%TEMP%b.flac", "plain.flac", "c%%d!&^.flac")]
        result = self.run_bat("Align LRC.bat", *paths)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.aligned_paths(), paths)

class ReviewBatchDroppedNameTests(LauncherStubCase):
    """Review LRC.bat: a dropped name that holds a percent-enclosed word reaches review_server.py as it was dropped."""
    NAMES = AlignBatchDroppedNameTests.NAMES

    def make_audio(self, name: str) -> str:
        path = self.tmp / "songs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"")
        return str(path)

    def review_calls(self) -> list[list[str]]:
        return [call for call in read_stub_calls(self.calls_file) if call[0].endswith("review_server.py")]

    def test_dropped_names_reach_the_review_server_verbatim(self) -> None:
        for name in self.NAMES:
            with self.subTest(name=name):
                self.calls_file.unlink(missing_ok=True)
                path = self.make_audio(name)
                result = self.run_bat("Review LRC.bat", path)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = self.review_calls()
                self.assertEqual(len(calls), 1, calls)
                self.assertEqual(calls[0][1:], [path])
                self.assertNotIn("Could not recover", result.stdout + result.stderr)

    def test_several_dropped_files_and_options_keep_their_own_names(self) -> None:
        paths = [self.make_audio(n) for n in ("a%TEMP%b.flac", "plain.flac", "c%%d!&^.flac")]
        result = self.run_bat("Review LRC.bat", "--read-only", *paths)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.review_calls()[0][1:], ["--read-only", *paths])

    def test_a_dropped_folder_with_a_percent_word_keeps_its_name(self) -> None:
        folder = self.tmp / "dir%TEMP%x"
        folder.mkdir()
        result = self.run_bat("Review LRC.bat", str(folder))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.review_calls()[0][1:], [str(folder)])

    def test_without_arguments_the_default_report_folder_is_used_and_nothing_is_recovered(self) -> None:
        result = self.run_bat("Review LRC.bat")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.review_calls()
        self.assertEqual(calls[0][1:], ["--report-dir", str(REPO_ROOT / "outputs" / "reports")])

    def test_the_exit_code_of_the_server_is_the_exit_code_of_the_launcher(self) -> None:
        self.env["STUB_FAIL_MATCH"] = "failing.flac"
        result = self.run_bat("Review LRC.bat", self.make_audio("failing.flac"))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_the_launcher_run_through_its_8_3_name_still_recovers_the_names(self) -> None:
        import ctypes
        import shutil
        copy = self.tmp / "long launcher folder"
        copy.mkdir()
        for name in ("Review LRC.bat", "review-lrc.ps1", "launcher-names.ps1"):
            shutil.copy2(REPO_ROOT / name, copy / name)
        buffer = ctypes.create_unicode_buffer(520)
        ctypes.windll.kernel32.GetShortPathNameW(str(copy / "Review LRC.bat"), buffer, 520)
        short = buffer.value
        if not short or short.lower().endswith("review lrc.bat"):
            self.skipTest("8.3 names are not enabled on this volume")
        path = self.make_audio("pct%TEMP%x.flac")
        result = self.run_cmdline(f'cmd /d /c ""{short}" "{path}""')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.review_calls()[0][1:], [path])

    def test_names_are_left_alone_and_nothing_fails_when_the_parent_is_not_the_launcher(self) -> None:
        script = (f'. "{REPO_ROOT / "launcher-names.ps1"}"; '
                  "Restore-DroppedNames -Received @('a%TEMP%b','c') -LauncherPath 'C:/nope/Review LRC.bat'")
        result = self.run_cmdline(f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{script}"')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.split(), ["a%TEMP%b", "c"])
        self.assertEqual(result.stderr.strip(), "")


class ReviewBatchSingleDashArgumentTests(LauncherStubCase):
    """Review LRC.bat: whatever is typed after the file name reaches review_server.py as written, whether or not PowerShell has a parameter of that name."""
    make_audio = ReviewBatchDroppedNameTests.make_audio
    review_calls = ReviewBatchDroppedNameTests.review_calls

    def typed(self, tail: str) -> list[str]:
        self.calls_file.unlink(missing_ok=True)
        result = self.run_cmdline(f'cmd /d /c ""{REPO_ROOT / "Review LRC.bat"}" {tail}"')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.review_calls()
        self.assertEqual(len(calls), 1, (tail, calls))
        return calls[0][1:]

    def test_single_dash_flags_are_not_taken_by_powershell(self) -> None:
        audio = self.make_audio("a b.flac")
        for tail, expected in (
            ("-d", ["-d"]),
            ("-p 1", ["-p", "1"]),
            (f'-Verbose "{audio}"', ["-Verbose", audio]),
            (f'-OutVariable x "{audio}"', ["-OutVariable", "x", audio]),
            ("-FromLauncher", ["-FromLauncher"]),
            ("-x 5", ["-x", "5"]),
            (f'-FromLauncher --read-only "{audio}"', ["-FromLauncher", "--read-only", audio]),
        ):
            with self.subTest(tail=tail):
                self.assertEqual(self.typed(tail), expected)

    def test_special_characters_still_reach_the_server_as_written(self) -> None:
        names = ("it's (live) [2024].flac", "a&b c.flac", "a;b,c=d.flac", "a!b^c.flac", "100% love.flac", "$env_USERNAME x.flac",
                 "a`b x.flac", "歌 曲君の名は.flac")
        for name in names:
            with self.subTest(name=name):
                path = self.make_audio(name)
                self.assertEqual(self.typed(f'"{path}"'), [path])

    def test_the_launcher_marker_is_not_an_argument(self) -> None:
        self.assertEqual(self.typed("--read-only"), ["--read-only"])


class AlignBatchSummaryTests(LauncherStubCase):
    def drop(self, *names: str) -> list[str]:
        songs = self.tmp / "songs"
        songs.mkdir(exist_ok=True)
        paths = []
        for name in names:
            (songs / name).write_bytes(b"")
            paths.append(str(songs / name))
        return paths

    def test_failed_middle_song_still_prints_every_result(self) -> None:
        self.env["STUB_FAIL_MATCH"] = "second.flac"
        paths = self.drop("first.flac", "second.flac", "third.flac")
        result = self.run_bat("Align LRC.bat", *paths)
        out = result.stdout
        self.assertNotEqual(result.returncode, 0, out + result.stderr)
        self.assertIn("Batch result: 2 of 3 succeeded, 1 failed", out)
        self.assertIn("OK     [1/3] first.flac (exit 0)", out)
        self.assertIn("FAILED [2/3] second.flac (exit 1)", out)
        self.assertIn("OK     [3/3] third.flac (exit 0)", out)
        aligned = [c[-1] for c in read_stub_calls(self.calls_file) if c[0].endswith("r2_pipeline.py")]
        self.assertEqual(aligned, paths)

    def test_all_songs_succeeding_exits_zero(self) -> None:
        paths = self.drop("first.flac", "second.flac")
        result = self.run_bat("Align LRC.bat", *paths)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Batch result: 2 of 2 succeeded, 0 failed", result.stdout)

if __name__ == "__main__":
    unittest.main()
