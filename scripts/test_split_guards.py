"""Guards added with the auto_lrc split: multi-file source hash and run_hub opening-evidence call."""
from __future__ import annotations

import ast
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import auto_lrc
import runtime_base
import split_patch

SCRIPTS = Path(__file__).resolve().parent


EXPECTED_SOURCE_FILES = (
    "acoustic_hypotheses.py",
    "asr_backends.py",
    "audio_features.py",
    "auto_lrc.py",
    "backend_bundle.py",
    "candidate_regions.py",
    "central_commit.py",
    "central_commit_tail.py",
    "crossline_retry_execution.py",
    "crossline_retry_planning.py",
    "ctc_boundary_evidence.py",
    "ctc_prefix_recovery.py",
    "ctc_refinement.py",
    "ctc_whisperx_refinement.py",
    "ctc_window_realign.py",
    "hubert_existing_candidate_recovery.py",
    "hypothesis_provenance.py",
    "known_lyric_hubert_planning.py",
    "local_whisper_requests.py",
    "local_whisper_retry.py",
    "lyrics_io.py",
    "neutral_arbitration.py",
    "postcentral_recovery.py",
    "postcentral_tail_acquisition.py",
    "recurrence_hypotheses.py",
    "report_trust.py",
    "runtime_base.py",
    "runtime_evidence_binding.py",
    "tail_occurrence_evidence.py",
    "timing_core.py",
    "timing_evidence_types.py",
    "vocal_onset_hypotheses.py",
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

    def test_report_trust_callbacks_are_the_central_commit_owner_objects(self):
        import central_commit
        import report_trust
        self.assertEqual(
            sorted(report_trust._LATE_BOUND_CALLBACKS),
            ["_final_post_central_material_conflict_entries", "_suspicious_entry_requires_review_for_decision"],
        )
        for name in report_trust._LATE_BOUND_CALLBACKS:
            self.assertIs(getattr(report_trust, name), getattr(central_commit, name), name)

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


class SplitPatchAssignRestoreTests(unittest.TestCase):
    NAME = "analyze_audio"

    def _bindings(self):
        return {m: id(getattr(__import__(m), self.NAME)) for m in split_patch.SPLIT_MODULES if hasattr(__import__(m), self.NAME)}

    def tearDown(self):
        split_patch._ACTIVE.clear()

    def test_restore_callable_puts_back_every_module_binding(self):
        before = self._bindings()
        self.assertGreater(len(before), 1)
        restore = split_patch.assign(self.NAME, lambda *a, **k: None)
        self.assertNotEqual(self._bindings(), before)
        with self.assertRaises(AssertionError):
            split_patch.assert_clean()
        restore()
        self.assertEqual(self._bindings(), before)
        split_patch.assert_clean()

    def test_facade_only_restore_is_reported(self):
        original = auto_lrc.analyze_audio
        before = self._bindings()
        restore = split_patch.assign(self.NAME, lambda *a, **k: None)
        try:
            setattr(auto_lrc, self.NAME, original)  # the facade-only restore this guard exists for
            with self.assertRaisesRegex(AssertionError, self.NAME):
                split_patch.assert_clean()
        finally:
            restore()
        self.assertEqual(self._bindings(), before)

    def test_assigning_the_originals_back_counts_as_restoring(self):
        original = auto_lrc.analyze_audio
        before = self._bindings()
        split_patch.assign(self.NAME, lambda *a, **k: None)
        split_patch.assign(self.NAME, original)
        self.assertEqual(self._bindings(), before)
        split_patch.assert_clean()

    def test_modules_that_assign_clean_up_in_tear_down_module(self):
        for path in sorted(SCRIPTS.glob("test_*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            if not any(
                isinstance(n, ast.Call) and ast.unparse(n.func) == "split_patch.assign" for n in ast.walk(tree)
            ):
                continue
            hooks = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tearDownModule"]
            self.assertEqual(len(hooks), 1, f"{path.name} needs tearDownModule calling split_patch.assert_clean()")
            self.assertIn("split_patch.assert_clean", ast.unparse(hooks[0]), path.name)


class KnownLyricHubertHelperSourceTests(unittest.TestCase):
    """The helper source tokens are substituted in the owner module, not the facade."""

    DEV_6D153C7C_REVISION = "d8b64f3df558618f1bc68c7c7940c1e018ed76c5bd80f84e1ee4adf2f6bd12a3"

    def test_owner_source_has_no_placeholder_tokens(self) -> None:
        import postcentral_recovery

        self.assertNotIn("__LRC_KNOWN_LYRIC_HUBERT_", postcentral_recovery._KNOWN_LYRIC_HUBERT_HELPER_SOURCE)

    def test_facade_reexports_the_owner_object(self) -> None:
        import postcentral_recovery

        self.assertIs(
            auto_lrc._KNOWN_LYRIC_HUBERT_HELPER_SOURCE,
            postcentral_recovery._KNOWN_LYRIC_HUBERT_HELPER_SOURCE,
        )

    def test_helper_revision_matches_dev_export(self) -> None:
        import postcentral_recovery

        self.assertEqual(
            postcentral_recovery._known_lyric_hubert_helper_revision(),
            self.DEV_6D153C7C_REVISION,
        )


class Split2ModuleGuards(unittest.TestCase):
    """Guards for the split2 submodules (ctc_refinement, timing_core, postcentral_recovery, local_whisper_retry, acoustic_hypotheses)."""

    SURFACES = {
        "ctc_refinement": ("ctc_boundary_evidence", "ctc_window_realign", "ctc_prefix_recovery", "ctc_whisperx_refinement"),
        "timing_core": ("timing_evidence_types", "tail_occurrence_evidence", "candidate_regions"),
        "postcentral_recovery": ("known_lyric_hubert_planning", "postcentral_tail_acquisition", "hubert_existing_candidate_recovery"),
        "local_whisper_retry": ("crossline_retry_planning", "crossline_retry_execution", "local_whisper_requests"),
        "acoustic_hypotheses": ("vocal_onset_hypotheses", "recurrence_hypotheses", "hypothesis_provenance"),
    }
    # (name count, sha256 of the sorted names) of each surface at dev 6a7b31e2, before the split.
    DEV_6A7B31E2_NAMES = {
        "ctc_refinement": (58, "4f92d10b5bad58281d062c3286e1b0fea5fdbc6f4f98d89bf873ef2a5dad5bdd"),
        "timing_core": (171, "2b5de4681a3ed89d9d9c24d8deabb34c0d16686959d6a230667ab56583eb90d8"),
        "postcentral_recovery": (53, "020fbfbb45bc33fc4e90efabd1d2008264156ccaaea477f83d60f85526d0e17e"),
        "local_whisper_retry": (36, "c4ad02e34abdd23dc9f8da12da91c590fa414161df884c0525cdcf32057a41c3"),
        "acoustic_hypotheses": (73, "c19ba3747627fb24fa9ad64431ca34d21e3560ef349770b03c8a27ae77d236ed"),
    }

    @staticmethod
    def _tree(module_name):
        return ast.parse((SCRIPTS / f"{module_name}.py").read_text(encoding="utf-8"))

    @classmethod
    def _defined_names(cls, module_name):
        names = []
        for node in cls._tree(module_name).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Assign):
                names.extend(n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.append(node.target.id)
        return names

    def test_every_new_module_is_in_the_hash_list(self):
        listed = set(runtime_base._ALGORITHM_SOURCE_MODULES)
        for surface, subs in self.SURFACES.items():
            self.assertIn(surface, listed)
            for sub in subs:
                self.assertIn(sub, listed, sub)

    def test_surface_name_set_matches_dev(self):
        import hashlib
        for surface, subs in self.SURFACES.items():
            names = set(self._defined_names(surface))
            for sub in subs:
                self.assertFalse(names & set(self._defined_names(sub)), f"{sub} redefines a name of {surface}")
                names |= set(self._defined_names(sub))
            count, digest = self.DEV_6A7B31E2_NAMES[surface]
            self.assertEqual(len(names), count, surface)
            self.assertEqual(hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest(), digest, surface)

    def test_surface_reexports_every_submodule_name_by_identity(self):
        import importlib
        for surface, subs in self.SURFACES.items():
            surface_module = importlib.import_module(surface)
            for sub in subs:
                owner = importlib.import_module(sub)
                for name in self._defined_names(sub):
                    self.assertIs(getattr(surface_module, name), getattr(owner, name), f"{surface}.{name}")
                    self.assertIs(getattr(auto_lrc, name, getattr(owner, name)), getattr(owner, name), f"auto_lrc.{name}")

    def test_no_module_binds_a_different_object_under_an_owned_name(self):
        import importlib
        for surface, subs in self.SURFACES.items():
            for module_name in (surface,) + subs:
                owner = importlib.import_module(module_name)
                for name in self._defined_names(module_name):
                    for other in split_patch.SPLIT_MODULES:
                        other_module = sys.modules.get(other)
                        if other_module is not None and hasattr(other_module, name):
                            self.assertIs(getattr(other_module, name), getattr(owner, name), f"{other}.{name} vs {module_name}")

    def test_submodules_never_import_their_surface_or_auto_lrc(self):
        forbidden_for = {"auto_lrc"}
        for surface, subs in self.SURFACES.items():
            for sub in subs:
                imported = set()
                for node in ast.walk(self._tree(sub)):
                    if isinstance(node, ast.ImportFrom):
                        imported.add((node.module or "").split(".")[0])
                    elif isinstance(node, ast.Import):
                        imported.update(a.name.split(".")[0] for a in node.names)
                self.assertFalse(imported & (forbidden_for | {surface}), f"{sub} imports {imported & (forbidden_for | {surface})}")

    def test_import_graph_of_split_modules_is_acyclic(self):
        modules = set(runtime_base._ALGORITHM_SOURCE_MODULES) | {"auto_lrc"}
        graph = {}
        for module_name in modules:
            edges = set()
            for node in self._tree(module_name).body:
                if isinstance(node, ast.ImportFrom) and node.module in modules:
                    edges.add(node.module)
                elif isinstance(node, ast.Import):
                    edges.update(a.name for a in node.names if a.name in modules)
            graph[module_name] = edges
        state = {}

        def visit(name, path):
            self.assertNotEqual(state.get(name), 1, f"import cycle: {' -> '.join(path + [name])}")
            if state.get(name) == 2:
                return
            state[name] = 1
            for dep in sorted(graph[name]):
                visit(dep, path + [name])
            state[name] = 2

        for name in sorted(graph):
            visit(name, [])

    def test_every_new_file_is_at_most_2500_lines(self):
        for surface, subs in self.SURFACES.items():
            for module_name in (surface,) + subs:
                lines = (SCRIPTS / f"{module_name}.py").read_text(encoding="utf-8").count("\n") + 1
                self.assertLessEqual(lines, 2500, module_name)

    def test_stage_timing_wrappers_live_in_the_owner_of_the_function(self):
        import importlib
        import lrc_stage_timings
        owners = {}
        for surface, subs in self.SURFACES.items():
            for module_name in (surface,) + subs:
                for name in self._defined_names(module_name):
                    owners[name] = module_name
        seen = 0
        for name, _label, _role in lrc_stage_timings.STAGE_TIMING_SPEC:
            if name not in owners:
                continue
            seen += 1
            owner = importlib.import_module(owners[name])
            function = getattr(owner, name)
            self.assertTrue(getattr(function, "__stage_timing_wrapper__", False), f"{owners[name]}.{name} not wrapped in its owner")
            for surface in self.SURFACES:
                self.assertIs(getattr(importlib.import_module(surface), name, function), function, f"{surface}.{name}")
        self.assertEqual(seen, 7)

    def test_known_lyric_hubert_helper_source_is_substituted_in_its_owner(self):
        import known_lyric_hubert_planning
        import postcentral_recovery
        self.assertNotIn("__LRC_KNOWN_LYRIC_HUBERT_", known_lyric_hubert_planning._KNOWN_LYRIC_HUBERT_HELPER_SOURCE)
        self.assertIs(
            postcentral_recovery._KNOWN_LYRIC_HUBERT_HELPER_SOURCE,
            known_lyric_hubert_planning._KNOWN_LYRIC_HUBERT_HELPER_SOURCE,
        )
        self.assertFalse(hasattr(known_lyric_hubert_planning, "_known_lyric_hubert_contract_token"))

    def test_patched_name_readers_stay_in_the_surface_module(self):
        # tests assign/patch these names on the surface module; their readers must read the surface binding.
        expected = {
            "ctc_refinement": ("apply_vocal_onset_tiebreak", "run_ctc_prefix_probe", "apply_final_ctc_timing_guard"),
            "local_whisper_retry": (
                "execute_local_whisper_request",
                "_prepare_local_whisper_batch",
                "_local_whisper_vocal_fusion_candidate",
                "collect_central_timing_candidates",
            ),
        }
        for surface, names in expected.items():
            defined = set(self._defined_names(surface))
            for name in names:
                self.assertIn(name, defined, f"{name} must be defined in {surface}")
            self.assertIs(getattr(sys.modules[surface], names[0]).__module__, surface)


def tearDownModule() -> None:
    split_patch.assert_clean()


if __name__ == "__main__":
    unittest.main()
