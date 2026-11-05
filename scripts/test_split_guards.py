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


class SourceHashCoversSubmodulesTests(unittest.TestCase):
    def test_editing_a_submodule_changes_the_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder)
            for path in runtime_base._algorithm_source_files():
                shutil.copy2(path, copy / path.name)
            fake_file = str(copy / "runtime_base.py")
            with mock.patch.object(runtime_base, "__file__", fake_file):
                files = runtime_base._algorithm_source_files()
                self.assertEqual([p.name for p in files], sorted(p.name for p in runtime_base._algorithm_source_files()))
                baseline = runtime_base._algorithm_source_sha256()
                for path in files:
                    original = path.read_bytes()
                    path.write_bytes(original + b"\n# edit\n")
                    try:
                        self.assertNotEqual(runtime_base._algorithm_source_sha256(), baseline, path.name)
                    finally:
                        path.write_bytes(original)
                self.assertEqual(runtime_base._algorithm_source_sha256(), baseline)

    def test_hash_files_include_every_existing_split_module(self):
        names = {p.name for p in runtime_base._algorithm_source_files()}
        self.assertIn("auto_lrc.py", names)
        for module in runtime_base._ALGORITHM_SOURCE_MODULES:
            if (SCRIPTS / f"{module}.py").is_file():
                self.assertIn(f"{module}.py", names)


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

    def test_bind_rejects_unknown_names(self):
        import report_trust
        with self.assertRaises(AttributeError):
            report_trust.bind_late_callbacks(not_a_callback=lambda: None)


if __name__ == "__main__":
    unittest.main()
