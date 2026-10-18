import contextlib
import ast
import io
import inspect
import json
import os
import subprocess
import sys
import tempfile
import threading
from contextlib import redirect_stderr, redirect_stdout
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

import lrc_stage_timings as st
import r2_pipeline
import auto_lrc


class _Clock:
    def __init__(self):
        self.value = 0.0
    def __call__(self):
        self.value += 0.01
        return self.value


class TimingModuleTests(unittest.TestCase):
    def setUp(self):
        st.set_clock(time_fn := _Clock())
        self.clock = time_fn
        os.environ.pop("LRC_STAGE_TIMINGS", None)

    def tearDown(self):
        st.set_clock(None)
        os.environ.pop("LRC_STAGE_TIMINGS", None)

    def test_nested_clock_records_nonnegative_and_parent_sum(self):
        def child(value):
            return value + 1
        def parent(value):
            return st.timed("child", child)(value)
        with st.session("standalone", "audio") as data:
            self.assertEqual(parent(1), 2)
        self.assertLessEqual(sum(x["seconds"] for x in data["stages"] if x["parent"] is None), data["total_seconds"] + 1e-6)
        starts = [x["start_s"] for x in data["stages"]]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(x["seconds"] >= 0 for x in data["stages"]))

    def test_rows_aggregate_cap_and_startup_are_reported(self):
        namespace = {"PROGRESS_START": 1.0}
        with mock.patch.object(st.time, "monotonic", return_value=3.5):
            st.install(namespace, ())
            with st.session("standalone", "audio") as data:
                repeated = st.timed("same", lambda: None)
                repeated(); repeated()
                for index in range(205):
                    st.timed(f"row-{index}", lambda: None)()
        same = next(row for row in data["stages"] if row["name"] == "same")
        self.assertEqual((same["calls"], same["errors"]), (2, 0))
        self.assertEqual(len(data["stages"]), 200)
        self.assertEqual(data["stages_truncated"], 6)
        self.assertEqual(data["startup_seconds"], 2.5)

    def test_exception_identity_and_base_exception_are_recorded(self):
        error = ValueError("bad")
        def fail():
            raise error
        with st.session("standalone", "audio") as data:
            with self.assertRaises(ValueError) as caught:
                st.timed("fail", fail)()
        self.assertIs(caught.exception, error)
        self.assertEqual(data["stages"][-1]["status"], "error:ValueError")

        class Special(BaseException):
            pass
        special = Special()
        def fail_base():
            raise special
        with st.session("standalone", "audio") as data:
            with self.assertRaises(Special) as caught:
                st.timed("base", fail_base)()
        self.assertIs(caught.exception, special)
        self.assertEqual(data["stages"][-1]["status"], "error:Special")

    def test_transparency_and_disabled_mode(self):
        marker = object()
        def original(a, *, b):
            return marker, a, b
        wrapped = st.timed("original", original)
        self.assertIs(wrapped(1, b=2)[0], marker)
        self.assertIs(wrapped.__wrapped__, original)
        self.assertEqual(wrapped.__name__, "original")
        self.assertTrue(inspect.getsource(wrapped).lstrip().startswith("def original"))
        with st.session("standalone", "audio") as data:
            os.environ["LRC_STAGE_TIMINGS"] = "0"
            wrapped(1, b=2)
        self.assertEqual(data["stages"], [])
        with st.session("standalone", "audio") as data:
            os.environ.pop("LRC_STAGE_TIMINGS")
            wrapped(1, b=2)
        self.assertEqual(len(data["stages"]), 1)

    def test_bookkeeping_failure_is_transparent(self):
        def original():
            return 7
        wrapped = st.timed("original", original)
        with st.session("standalone", "audio") as data:
            class BrokenList(list):
                def append(self, value):
                    raise RuntimeError("bookkeeping")
            data["stages"] = BrokenList()
            self.assertEqual(wrapped(), 7)

    def test_clock_failure_calls_original_once_untimed(self):
        calls = []
        wrapped = st.timed("original", lambda: calls.append(1) or 7)
        with st.session("standalone", "audio") as data:
            with mock.patch.object(st, "_CLOCK", side_effect=RuntimeError("clock")):
                self.assertEqual(wrapped(), 7)
        self.assertEqual(calls, [1])
        self.assertEqual(data["stages"], [])

    def test_exit_clock_failure_clears_stack_and_preserves_call_semantics(self):
        error = RuntimeError("original")
        with st.session("standalone", "audio") as data:
            with mock.patch.object(st, "_CLOCK", side_effect=[1.0, 2.0, RuntimeError("clock")]):
                self.assertEqual(st.timed("ok", lambda: 7)(), 7)
            self.assertEqual(getattr(st._TLS, "stack", []), [])
        self.assertEqual(data["stages"], [])

        def fail():
            raise error

        with st.session("standalone", "audio") as data:
            with mock.patch.object(st, "_CLOCK", side_effect=[1.0, 2.0, RuntimeError("clock")]):
                with self.assertRaises(RuntimeError) as caught:
                    st.timed("fail", fail)()
            self.assertIs(caught.exception, error)
            self.assertEqual(getattr(st._TLS, "stack", []), [])
        self.assertEqual(data["stages"], [])

    def test_worker_stage_is_separate(self):
        def original():
            return 3
        wrapped = st.timed("worker", original)
        with st.session("standalone", "audio") as data:
            thread = threading.Thread(target=wrapped)
            thread.start()
            thread.join()
        self.assertEqual(data["stages"], [])
        self.assertEqual(data["worker_stages"]["worker"]["calls"], 1)

    def test_session_owner_is_the_thread_that_opens_it(self):
        result = {}
        def run():
            with st.session("standalone", "audio") as data:
                st.timed("owner", lambda: None)()
                result.update(data)
        thread = threading.Thread(target=run)
        thread.start(); thread.join()
        self.assertEqual(result["stages"][0]["name"], "owner")
        self.assertEqual(result["worker_stages"], {})

    def test_cache_counters_and_vocal_status(self):
        class ModelCacheMiss(BaseException):
            pass
        values = iter([object(), None, ModelCacheMiss()])
        def read(kind, *args, **kwargs):
            value = next(values)
            if isinstance(value, BaseException):
                raise value
            return value
        def write(kind, *args, **kwargs):
            return None
        def stem(*args, **kwargs):
            return "x", "vocal-stem", "cache-hit"
        with st.session("standalone", "audio") as data:
            r = st.timed("cache-read", read, cache_kind="read")
            self.assertIsNotNone(r("raw-asr"))
            self.assertIsNone(r("raw-asr"))
            with self.assertRaises(ModelCacheMiss):
                r("raw-asr")
            st.timed("cache-write", write, cache_kind="write")("raw-asr")
            st.timed("separation", stem, function_name="prepare_vocal_ctc_audio")()
        self.assertEqual(data["model_cache"]["raw-asr"], {"hit": 1, "miss": 2, "fill": 1})
        self.assertEqual(data["vocal_stem"], ["cache-hit"])

    def test_install_skips_missing_and_is_idempotent(self):
        namespace = {"exists": lambda: 1}
        spec = (("exists", "exists", "top"), ("missing", "missing", "top"))
        st.install(namespace, spec)
        first = namespace["exists"]
        st.install(namespace, spec)
        self.assertIs(namespace["exists"], first)
        self.assertIn("missing", st.install_report()["missing"])

    def test_install_uses_function_name_for_vocal_status(self):
        namespace = {"prepare_vocal_ctc_audio": lambda: ("x", "vocals", "cache-hit")}
        st.install(namespace, (("prepare_vocal_ctc_audio", "separation", "nested"),))
        with st.session("standalone", "audio") as data:
            namespace["prepare_vocal_ctc_audio"]()
        self.assertEqual(data["vocal_stem"], ["cache-hit"])

    def test_no_bk_imports_and_no_numpy(self):
        source = (Path(__file__).with_name("lrc_stage_timings.py")).read_text(encoding="utf-8")
        for forbidden in ("checked_lrc_candidates", "checked-lrc", "LRC bk", "checked references"):
            self.assertNotIn(forbidden, source)
        code = "import sys; import lrc_stage_timings; assert 'numpy' not in sys.modules; assert 'auto_lrc' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class ProductionIntegrationTests(unittest.TestCase):
    def _run(self, root, enabled=True, no_summary=False):
        audio = root / "song.flac"
        lyrics = root / "song.txt"
        args_list = [str(audio), "--lyrics", str(lyrics), "--output", str(root / "song.lrc"),
                     "--timing-source", "lyrics", "--report-dir", str(root / "reports"), "--overwrite"]
        if no_summary:
            args_list.append("--no-review-summary")
        args = auto_lrc.build_parser().parse_args(args_list)
        os.environ["LRC_STAGE_TIMINGS"] = "1" if enabled else "0"
        output = io.StringIO()
        with mock.patch.object(auto_lrc, "run_command", return_value=SimpleNamespace(returncode=0, stdout='{"format":{"duration":"2.0"}}', stderr="")), mock.patch.object(auto_lrc, "audio_track_labels", return_value=("", "Song")), redirect_stdout(output):
            auto_lrc.process_audio(audio, args)
        report_path, _, _ = auto_lrc.report_artifact_paths(audio, root / "song.lrc", root / "reports")
        return (root / "song.lrc").read_bytes(), json.loads(report_path.read_text(encoding="utf-8")), output.getvalue()

    def test_process_audio_timing_is_additive_and_key_ordered(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "song.flac"
            lyrics = root / "song.txt"
            audio.write_bytes(b"audio")
            lyrics.write_text("[00:01.00]line\n", encoding="utf-8")
            old_bytes, old_report, old_out = self._run(root, enabled=False)
            new_bytes, new_report, new_out = self._run(root, enabled=True)
            self.assertEqual(old_bytes, new_bytes)
            old_no_timing = dict(old_report)
            old_no_timing.pop("stage_timings", None)
            new_no_timing = dict(new_report)
            new_no_timing.pop("stage_timings", None)
            self.assertEqual(old_no_timing, new_no_timing)
            self.assertEqual(list(old_no_timing), list(new_no_timing))
            keys = list(new_report)
            self.assertEqual(keys[keys.index("stage_timings") + 1], "published_lrc_sha256")
            self.assertEqual(new_report["stage_timings"]["pass"], "standalone")
            names = {row["name"] for row in new_report["stage_timings"]["stages"]}
            self.assertTrue({"load-lyrics", "probe-duration", "write-review-tsv", "write-lrc"} <= names, repr(names))
            self.assertNotIn("song", names)
            self.assertGreaterEqual(new_report["stage_timings"]["unaccounted_seconds"], 0.0)
            self.assertEqual(new_out.count("TIMINGS pass=standalone status=ok"), 1)
            _, _, suppressed = self._run(root, enabled=True, no_summary=True)
            self.assertNotIn("TIMINGS pass=standalone", suppressed)

    def test_song_wrapper_carries_real_main_rollback_and_attempt_two_is_terminal(self):
        calls = []
        attempts = []
        def fake(audio, args):
            calls.append(getattr(args, "option_a_rollback_reason", None))
            attempts.append(st._ACTIVE["attempts"])
            if len(calls) == 1:
                raise auto_lrc.SegmentSkeletonRollback("retry")
            return Path("out.lrc")
        namespace = {"process_audio": fake}
        st.install(namespace, (("process_audio", "song", "song"),))
        output = io.StringIO()
        with mock.patch.object(auto_lrc, "process_audio", namespace["process_audio"]), \
             mock.patch.object(auto_lrc, "_apply_model_cache_flags", return_value="off"), \
             mock.patch.object(auto_lrc, "_managed_temp_snapshot", return_value=set()), \
             mock.patch.object(auto_lrc, "_cleanup_managed_temp_dirs_since"), redirect_stdout(output):
            self.assertEqual(auto_lrc.main(["song.flac"]), 0)
        self.assertEqual(calls, [None, "retry"])
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(output.getvalue().count("TIMINGS pass=standalone"), 1)

        calls.clear(); attempts.clear(); output = io.StringIO()
        def always_fail(audio, args):
            attempts.append(st._ACTIVE["attempts"])
            raise auto_lrc.SegmentSkeletonRollback("retry")
        namespace = {"process_audio": always_fail}
        st.install(namespace, (("process_audio", "song", "song"),))
        with mock.patch.object(auto_lrc, "process_audio", namespace["process_audio"]), \
             mock.patch.object(auto_lrc, "_apply_model_cache_flags", return_value="off"), \
             mock.patch.object(auto_lrc, "_managed_temp_snapshot", return_value=set()), \
             mock.patch.object(auto_lrc, "_cleanup_managed_temp_dirs_since"), redirect_stdout(output):
            self.assertEqual(auto_lrc.main(["song.flac"]), 1)
        self.assertEqual(attempts, [1, 2])
        self.assertEqual(output.getvalue().count("TIMINGS pass=standalone"), 1)
        self.assertIn("status=error:SegmentSkeletonRollback", output.getvalue())

    def test_stale_pending_session_is_dropped_before_next_song(self):
        args = SimpleNamespace(
            stage_timings_pass="standalone",
            no_review_summary=True,
            option_a_rollback_reason=None,
        )
        def rollback(audio, parsed):
            raise auto_lrc.SegmentSkeletonRollback("retry")
        with self.assertRaises(auto_lrc.SegmentSkeletonRollback):
            st._song_wrapper(rollback, (Path("old.flac"), args), {}, "song")

        error_output = io.StringIO()
        with redirect_stderr(error_output):
            result = st._song_wrapper(lambda audio, parsed: Path("new.lrc"), (Path("new.flac"), args), {}, "song")
        self.assertEqual(result, Path("new.lrc"))
        self.assertEqual(error_output.getvalue(), "WARN: stage timings: dropped pending session\n")
        self.assertEqual(st._PENDING, {})

    def test_song_session_setup_failure_calls_original_once_untimed(self):
        args = SimpleNamespace(
            stage_timings_pass="standalone",
            no_review_summary=False,
            option_a_rollback_reason=None,
        )
        calls = []
        with mock.patch.object(st, "_CLOCK", side_effect=RuntimeError("clock")):
            result = st._song_wrapper(lambda audio, parsed: calls.append(audio) or Path("out.lrc"), (Path("song.flac"), args), {}, "song")
        self.assertEqual(result, Path("out.lrc"))
        self.assertEqual(calls, [Path("song.flac")])

    def test_song_wrapper_records_stage_failure_and_same_exception(self):
        class LrcError(Exception):
            pass
        error = LrcError("bad")
        class Args:
            stage_timings_pass = "standalone"
            no_review_summary = False
            option_a_rollback_reason = None
        def fake(audio, args):
            raise error
        output = io.StringIO()
        with redirect_stdout(output):
            with self.assertRaises(LrcError) as caught:
                st._song_wrapper(fake, (Path("song.flac"), Args()), {}, "song")
        self.assertIs(caught.exception, error)
        self.assertIn("TIMINGS pass=standalone status=error:LrcError", output.getvalue())


class StageArgsTests(unittest.TestCase):
    def test_stage_timings_args_preserve_pinned_tails(self):
        raw = ["--timing-source", "auto", "--stage-timings-pass", "x", "song.flac"]
        baseline = r2_pipeline.prepare_baseline_args(raw, output=Path("b.lrc"), report_dir=Path("br"))
        self.assertEqual(baseline[-7:], ["--arbiter", "off", "--output", "b.lrc", "--report-dir", "br", "--overwrite"])
        self.assertEqual(baseline.count("--stage-timings-pass"), 1)
        self.assertEqual(baseline[baseline.index("--stage-timings-pass") + 1], "baseline")
        final = r2_pipeline.prepare_final_args(raw, Path("evidence.json"))
        self.assertEqual(final[-4:], ["--arbiter", "reviewer-validity", "--reviewer-evidence", "evidence.json"])
        self.assertEqual(final.count("--stage-timings-pass"), 1)
        self.assertEqual(final[final.index("--stage-timings-pass") + 1], "final")
        self.assertEqual(r2_pipeline.auto_lrc.build_parser().parse_args(["song.flac", "--stage-timings-pass", "final"]).stage_timings_pass, "final")

    def test_stage_spec_rebinding_contract(self):
        tree = ast.parse(Path(auto_lrc.__file__).read_text(encoding="utf-8"))
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        defs = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        imports = {alias.asname or alias.name for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        for name, _label, _role in auto_lrc._STAGE_TIMING_SPEC:
            self.assertTrue(name in defs or name == "write_review_tsv" and name in imports, name)
            if name in defs:
                self.assertEqual(defs[name].decorator_list, [], name)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name:
                    self.assertIsInstance(getattr(node, "parent", None), ast.Call, name)
        self.assertEqual(st.install_report()["missing"], [])
        for name, _label, _role in auto_lrc._STAGE_TIMING_SPEC:
            self.assertTrue(getattr(vars(auto_lrc)[name], "__stage_timing_wrapper__", False), name)


class R2TimingTests(unittest.TestCase):
    def _backend_report(self, audio, output, report_dir):
        report_path, _, _ = auto_lrc.report_artifact_paths(audio, output, report_dir)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"assignments": [{"timestamp": 1.0}], "stage_timings": {"total_seconds": 1.25}, "audio_path": str(audio)}), encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")

    def test_reviewer_statuses_include_seconds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lyrics = root / "song.txt"
            vocal = root / "vocals.wav"
            report_path = root / "baseline.json"
            output_path = root / "evidence.json"
            work = root / "reviewers"
            lyrics.write_text("\n".join(f"かな{i}" for i in range(1, 9)) + "\n", encoding="utf-8")
            vocal.write_bytes(b"fake")
            assignments = [{"entry": i, "timestamp": float(i), "ctc_first_token_start": float(i), "candidates": [{"source": "ctc-current", "time": float(i)}]} for i in range(1, 9)]
            report_path.write_text(json.dumps({"lyrics_path": str(lyrics), "audio_path": str(root / "song.flac"), "ctc_audio_path": str(vocal), "assignments": assignments}), encoding="utf-8")
            with mock.patch.object(r2_pipeline, "build_reviewer_commands", return_value=[("WX", ["wx"]), ("XLSR", ["xlsr"])]), mock.patch.object(r2_pipeline, "_run_logged", return_value=0):
                result = r2_pipeline.acquire_reviewer_evidence(report_path, output_path, work)
        self.assertIn("seconds", result["statuses"]["WX"])
        self.assertIn("seconds", result["statuses"]["XLSR"])

    def test_r2_main_writes_timings_when_baseline_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "song.flac"
            audio.write_bytes(b"audio")
            run_dir = r2_pipeline.reviewer_work_directory(root / "work", audio)
            stale_status = run_dir / "reviewers" / "reviewer-status.json"
            stale_status.parent.mkdir(parents=True)
            stale_status.write_text(json.dumps({"schema": 1, "steps": {"STALE": {"seconds": 9.0}}}), encoding="utf-8")
            with mock.patch.object(r2_pipeline, "_run_backend", return_value=1):
                result = r2_pipeline.main(["--work-dir", str(root / "work"), "--", str(audio), "--timing-source", "lyrics", "--output", str(root / "final.lrc"), "--report-dir", str(root / "reports")])
            timings = json.loads((run_dir / "r2-timings.json").read_text(encoding="utf-8"))
            self.assertFalse(stale_status.exists())
        self.assertEqual(result, 1)
        self.assertEqual([row["status"] for row in timings["stages"]], ["failed", "not-run", "not-run", "not-run"])
        self.assertIsNone(timings["reviewers"])
        self.assertIsNotNone(timings["total_seconds"])

    def test_r2_main_full_mocked_run_writes_overhead_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "song.flac"
            audio.write_bytes(b"audio")
            def backend(args):
                args = list(args)
                output = Path(args[args.index("--output") + 1]) if "--output" in args else audio.with_suffix(".lrc")
                report_dir = Path(args[args.index("--report-dir") + 1]) if "--report-dir" in args else root / "reports"
                self._backend_report(audio, output, report_dir)
                return 0
            def acquire(report_path, output_path, work_dir, **kwargs):
                work_dir.mkdir(parents=True, exist_ok=True)
                (work_dir / "reviewer-data.json").write_text(json.dumps({"songs": []}), encoding="utf-8")
                return {"statuses": {"WX": {"status": "OK", "seconds": 0.1}}, "evidence": str(output_path)}
            output = io.StringIO()
            with mock.patch.object(r2_pipeline, "_run_backend", side_effect=backend), mock.patch.object(r2_pipeline, "acquire_reviewer_evidence", side_effect=acquire), mock.patch.object(r2_pipeline, "build_reviewer_trust_report", return_value={"profile": "none", "trusted_count": 0, "review_count": 0}), mock.patch.object(r2_pipeline, "build_lyric_review_messages", return_value=[]), mock.patch.object(r2_pipeline, "build_sparse_reviewer_notice", return_value=None), redirect_stdout(output):
                result = r2_pipeline.main(["--work-dir", str(root / "work"), "--", str(audio), "--timing-source", "lyrics", "--output", str(root / "final.lrc"), "--report-dir", str(root / "reports")])
            run_dir = r2_pipeline.reviewer_work_directory(root / "work", audio)
            timings = json.loads((run_dir / "r2-timings.json").read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(len(timings["stages"]), 4)
        self.assertIn("process_overhead_seconds", timings)
        self.assertIn("R2 TIMINGS", output.getvalue())
        for stage in range(1, 5):
            self.assertIn(f"[{stage}/4] 完成，用時", output.getvalue())

    def test_r2_manual_edit_refusal_writes_no_timing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "song.flac"
            output = root / "song.lrc"
            audio.write_bytes(b"audio"); output.write_text("edited", encoding="utf-8")
            with mock.patch.object(r2_pipeline, "manual_edit_reason", return_value="edited"), \
                 mock.patch.object(r2_pipeline, "journal_path", return_value=root / "missing-journal"):
                result = r2_pipeline.main(["--work-dir", str(root / "work"), "--", str(audio), "--output", str(output), "--overwrite"])
            run_dir = r2_pipeline.reviewer_work_directory(root / "work", audio)
            self.assertEqual(result, 2)
            self.assertFalse((run_dir / "r2-timings.json").exists())

    def test_r2_parser_error_writes_no_timing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            work_dir = Path(folder) / "work"
            with self.assertRaises(SystemExit) as caught:
                r2_pipeline.main(["--work-dir", str(work_dir)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(list(work_dir.rglob("r2-timings.json")))

    def test_r2_partial_reviewer_failure_reads_status_file_and_reraises_same_exception(self):
        error = RuntimeError("reviewer failed")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "song.flac"; audio.write_bytes(b"audio")
            lyrics = root / "song.txt"; lyrics.write_text("\n".join(f"かな{i}" for i in range(1, 9)), encoding="utf-8")
            vocal = root / "vocals.wav"; vocal.write_bytes(b"vocal")
            def backend(args):
                args = list(args)
                output = Path(args[args.index("--output") + 1])
                report_dir = Path(args[args.index("--report-dir") + 1])
                report_path, _, _ = auto_lrc.report_artifact_paths(audio, output, report_dir)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                assignments = [{"entry": i, "timestamp": float(i), "ctc_first_token_start": float(i), "candidates": [{"source": "ctc-current", "time": float(i)}]} for i in range(1, 9)]
                report_path.write_text(json.dumps({"lyrics_path": str(lyrics), "audio_path": str(audio), "ctc_audio_path": str(vocal), "assignments": assignments}), encoding="utf-8")
                output.parent.mkdir(parents=True, exist_ok=True); output.write_text("", encoding="utf-8")
                return 0
            with mock.patch.object(r2_pipeline, "_run_backend", side_effect=backend), \
                 mock.patch.object(r2_pipeline, "build_reviewer_commands", return_value=[("HUBP_PLAN", ["plan"]), ("WX", ["wx"])]), \
                 mock.patch.object(r2_pipeline, "_run_logged", side_effect=[0, error]):
                with self.assertRaises(RuntimeError) as caught:
                    r2_pipeline.main(["--work-dir", str(root / "work"), "--", str(audio), "--output", str(root / "final.lrc")])
            run_dir = r2_pipeline.reviewer_work_directory(root / "work", audio)
            timings = json.loads((run_dir / "r2-timings.json").read_text(encoding="utf-8"))
        self.assertIs(caught.exception, error)
        self.assertIn("HUBP_PLAN", timings["reviewers"])
        self.assertIsNotNone(timings["reviewers"]["HUBP_PLAN"]["seconds"])
        self.assertEqual(timings["stages"][1]["status"], "error:RuntimeError")


if __name__ == "__main__":
    unittest.main()
