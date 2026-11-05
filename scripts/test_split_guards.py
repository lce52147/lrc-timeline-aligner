"""Guards added with the auto_lrc split: multi-file source hash and run_hub opening-evidence call."""
from __future__ import annotations

import ast
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import auto_lrc
import runtime_base

SCRIPTS = Path(__file__).resolve().parent


EXPECTED_SOURCE_FILES = (
    "acoustic_hypotheses.py",
    "asr_backends.py",
    "audio_features.py",
    "auto_lrc.py",
    "backend_bundle.py",
    "central_commit.py",
    "central_commit_tail.py",
    "central_peer_recovery.py",
    "ctc_refinement.py",
    "local_whisper_retry.py",
    "lyrics_io.py",
    "neutral_arbitration.py",
    "report_trust.py",
    "runtime_base.py",
    "timing_core.py",
)


class SourceHashCoversSubmodulesTests(unittest.TestCase):
    def _copy_sources(self, folder):
        copy = Path(folder)
        for path in runtime_base._algorithm_source_files():
            shutil.copy2(path, copy / path.name)
        return copy

    def test_hash_file_set_is_pinned(self):
        names = [p.name for p in runtime_base._algorithm_source_files()]
        self.assertEqual(names, sorted(EXPECTED_SOURCE_FILES))
        for name in names:
            self.assertTrue((SCRIPTS / name).is_file(), name)

    def test_missing_listed_module_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = self._copy_sources(folder)
            (copy / "timing_core.py").unlink()
            with mock.patch.object(runtime_base, "__file__", str(copy / "runtime_base.py")):
                with self.assertRaisesRegex(RuntimeError, "timing_core.py"):
                    runtime_base._algorithm_source_files()
                with self.assertRaises(runtime_base.LrcError):
                    runtime_base.validated_loaded_source_sha256()

    def test_editing_a_submodule_changes_the_hash(self):
        expected = [p.name for p in runtime_base._algorithm_source_files()]
        with tempfile.TemporaryDirectory() as folder:
            copy = self._copy_sources(folder)
            with mock.patch.object(runtime_base, "__file__", str(copy / "runtime_base.py")):
                files = runtime_base._algorithm_source_files()
                self.assertEqual({p.parent for p in files}, {copy.resolve()})
                self.assertEqual([p.name for p in files], expected)
                baseline = runtime_base._algorithm_source_sha256()
                self.assertEqual(runtime_base.validated_loaded_source_sha256(), baseline)
                for path in files:
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n# edit\n")
                    try:
                        self.assertNotEqual(runtime_base._algorithm_source_sha256(), baseline, path.name)
                        with self.assertRaises(runtime_base.LrcError, msg=path.name):
                            runtime_base.validated_loaded_source_sha256()
                    finally:
                        path.write_bytes(original)
                self.assertEqual(runtime_base._algorithm_source_sha256(), baseline)


class RunHubOpeningEvidenceTests(unittest.TestCase):
    def test_call_shape_matches_the_removed_wrapper(self):
        tree = ast.parse((SCRIPTS / "reviewers" / "run_hub.py").read_text(encoding="utf-8"))
        compares = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Attribute)
            and node.left.func.attr == "_known_lyric_hubert_opening_evidence"
        ]
        self.assertEqual(len(compares), 1)
        node = compares[0]
        self.assertIsInstance(node.ops[0], ast.IsNot)
        self.assertIsNone(node.comparators[0].value)
        self.assertEqual(len(node.left.args), 3)

    def test_helper_fails_closed_on_stub_inputs(self):
        for payload in ([], None, [{"char": "a"}]):
            self.assertIsNone(auto_lrc._known_lyric_hubert_opening_evidence(0, payload, "rev"), payload)


class LateBoundCallbackTests(unittest.TestCase):
    def test_report_trust_callbacks_are_bound_to_the_real_functions(self):
        import report_trust
        for name in report_trust._LATE_BOUND_CALLBACKS:
            self.assertIs(getattr(report_trust, name), getattr(auto_lrc, name), name)

    def test_audio_api_is_bound_in_every_module_that_late_binds_it(self):
        import asr_backends
        import central_commit
        for module in (asr_backends, central_commit):
            self.assertEqual(module._LATE_BOUND_CALLBACKS, ("_AUDIO_API",))
            self.assertIs(module._AUDIO_API, auto_lrc._AUDIO_API, module.__name__)

    def test_bind_rejects_unknown_names(self):
        import asr_backends
        import central_commit
        import report_trust
        for module in (report_trust, asr_backends, central_commit):
            with self.assertRaises(AttributeError, msg=module.__name__):
                module.bind_late_callbacks(not_a_callback=lambda: None)

    def test_unbound_placeholders_raise_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "_x is not bound yet"):
            runtime_base.unbound_callback("_x")(1, key=2)
        with self.assertRaisesRegex(RuntimeError, "_NS is not bound yet"):
            runtime_base.UnboundNamespace("_NS").anything
        namespace = {}
        runtime_base.bind_late_callbacks(namespace, ("_x",), {"_x": 5})
        self.assertEqual(namespace, {"_x": 5})
        with self.assertRaises(AttributeError):
            runtime_base.bind_late_callbacks(namespace, ("_x",), {"_y": 5})


if __name__ == "__main__":
    unittest.main()
