"""The final pass remembers slow lookups instead of redoing them per song.

Covers the memos added for the final-pass speed-up: tail-evidence context
validity is remembered per frozen object, and the WhisperX alignment-model
lookup (a fresh interpreter importing whisperx) is remembered on disk while the
interpreter, helper, language and hub cache stay unchanged.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc  # noqa: E402
import file_digest  # noqa: E402
import runtime_paths  # noqa: E402


def _scope(entry_index: int = 0) -> auto_lrc.TailEvidenceScope:
    return auto_lrc.TailEvidenceScope(
        entry_index=entry_index,
        producer="whisperx",
        backend_lineage="whisperx",
        model_identity="model",
        helper_revision="helper",
        capability_revision="capability",
        audio_revision="audio",
        transcript_revisions=("t1",),
        row_revisions=("r1", "r2"),
    )


def _context(**overrides) -> auto_lrc.TailEvidenceContext:
    values = dict(
        backend_lineage="whisperx",
        model_identity="model",
        helper_revision="helper",
        capability_revision="capability",
        audio_revision="audio",
        transcript_revisions=("t1",),
        row_revisions=("r1", "r2"),
        scopes=(_scope(),),
    )
    values.update(overrides)
    return auto_lrc.TailEvidenceContext(**values)


class TailContextValidityMemoTests(unittest.TestCase):
    def setUp(self) -> None:
        auto_lrc._TAIL_CONTEXT_VALIDITY.clear()

    def tearDown(self) -> None:
        auto_lrc._TAIL_CONTEXT_VALIDITY.clear()

    def test_verdict_is_computed_once_per_context_object(self) -> None:
        context = _context()
        with mock.patch.object(
            auto_lrc, "_tail_context_is_valid_uncached", wraps=auto_lrc._tail_context_is_valid_uncached
        ) as uncached:
            self.assertTrue(auto_lrc._tail_context_is_valid(context))
            self.assertTrue(auto_lrc._tail_context_is_valid(context))
            self.assertEqual(uncached.call_count, 1)
            # An equal but distinct frozen object is validated on its own.
            self.assertTrue(auto_lrc._tail_context_is_valid(_context()))
            self.assertEqual(uncached.call_count, 2)

    def test_invalid_contexts_are_remembered_as_invalid(self) -> None:
        unsorted = _context(row_revisions=("r2", "r1"))
        with mock.patch.object(
            auto_lrc, "_tail_context_is_valid_uncached", wraps=auto_lrc._tail_context_is_valid_uncached
        ) as uncached:
            self.assertFalse(auto_lrc._tail_context_is_valid(unsorted))
            self.assertFalse(auto_lrc._tail_context_is_valid(unsorted))
            self.assertEqual(uncached.call_count, 1)
        self.assertFalse(auto_lrc._tail_context_is_valid(None))
        self.assertFalse(auto_lrc._tail_context_is_valid(_context(backend_lineage="unknown-backend")))
        self.assertFalse(auto_lrc._tail_context_is_valid(_context(transcript_revisions=())))
        self.assertFalse(auto_lrc._tail_context_is_valid(_context(scopes=())))

    def test_memo_keeps_the_context_alive_and_is_bounded(self) -> None:
        context = _context()
        self.assertTrue(auto_lrc._tail_context_is_valid(context))
        self.assertIs(auto_lrc._TAIL_CONTEXT_VALIDITY[id(context)][0], context)
        for index in range(4096):
            auto_lrc._TAIL_CONTEXT_VALIDITY[-index - 1] = (object(), True)
        self.assertTrue(auto_lrc._tail_context_is_valid(_context(audio_revision="other")))
        self.assertEqual(len(auto_lrc._TAIL_CONTEXT_VALIDITY), 1)


class WhisperXModelIdentityMemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.hub = self.root / "hub"
        self.hub.mkdir()
        self.helper = self.root / "whisperx_refine.py"
        self.helper.write_text("# helper\n", encoding="utf-8")
        self.env = mock.patch.dict(os.environ, {"LRC_MODEL_CACHE_DIR": str(self.root / "cache"), "HF_HUB_CACHE": str(self.hub)})
        self.env.start()
        self.hub_patch = mock.patch.object(runtime_paths, "hf_hub_cache_candidates", return_value=[self.hub])
        self.hub_patch.start()
        file_digest._MEMO.clear()
        auto_lrc._whisperx_model_identity.cache_clear()

    def tearDown(self) -> None:
        auto_lrc._whisperx_model_identity.cache_clear()
        self.hub_patch.stop()
        self.env.stop()
        self.temp.cleanup()

    def _lookup(self, language: str = "ja") -> tuple[str, str]:
        return auto_lrc._whisperx_model_identity(Path(sys.executable), self.helper, language)

    def test_resolved_identity_survives_a_new_process(self) -> None:
        with mock.patch.object(auto_lrc, "_probe_whisperx_model_identity", return_value=("align-model", "abc123")) as probe:
            self.assertEqual(self._lookup(), ("align-model", "abc123"))
            self.assertEqual(self._lookup(), ("align-model", "abc123"))
            self.assertEqual(probe.call_count, 1)
            auto_lrc._whisperx_model_identity.cache_clear()  # a new interpreter has no in-process memo
            self.assertEqual(self._lookup(), ("align-model", "abc123"))
            self.assertEqual(probe.call_count, 1, "the on-disk memo answers without probing")
        self.assertTrue(file_digest.memo_path("whisperx-align-model.json").is_file())

    def test_hub_change_helper_change_or_language_retires_the_memo(self) -> None:
        with mock.patch.object(auto_lrc, "_probe_whisperx_model_identity", return_value=("align-model", "abc123")) as probe:
            self._lookup()
            auto_lrc._whisperx_model_identity.cache_clear()
            self._lookup("en")
            self.assertEqual(probe.call_count, 2, "another language is another lookup")
            auto_lrc._whisperx_model_identity.cache_clear()
            self.helper.write_text("# helper changed\n", encoding="utf-8")
            file_digest._MEMO.clear()
            self._lookup()
            self.assertEqual(probe.call_count, 3, "a changed helper probes again")
            auto_lrc._whisperx_model_identity.cache_clear()
            (self.hub / "models--new" / "snapshots").mkdir(parents=True)
            self._lookup()
            self.assertEqual(probe.call_count, 4, "a new hub snapshot probes again")
            auto_lrc._whisperx_model_identity.cache_clear()
            self._lookup()
            self.assertEqual(probe.call_count, 4, "unchanged inputs answer from disk")

    def test_unresolved_probe_is_not_remembered(self) -> None:
        with mock.patch.object(auto_lrc, "_probe_whisperx_model_identity", return_value=("unresolved", "unresolved")) as probe:
            self.assertEqual(self._lookup(), ("unresolved", "unresolved"))
            auto_lrc._whisperx_model_identity.cache_clear()
            self.assertEqual(self._lookup(), ("unresolved", "unresolved"))
            self.assertEqual(probe.call_count, 2)
        self.assertEqual(file_digest.memo_load("whisperx-align-model.json"), {})

    def test_hub_fingerprint_tracks_snapshots_and_refs(self) -> None:
        self.assertEqual(auto_lrc._hf_hub_fingerprint(), [])
        repo = self.hub / "models--org--model"
        (repo / "snapshots").mkdir(parents=True)
        (self.hub / "not-a-model").mkdir()
        first = auto_lrc._hf_hub_fingerprint()
        self.assertEqual([row[0] for row in first], [str(repo)])
        self.assertEqual(len(first[0]), 5)
        self.assertEqual(first[0][3:], [None, None], "a repo without refs/main is still listed")
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("deadbeef", encoding="utf-8")
        second = auto_lrc._hf_hub_fingerprint()
        self.assertNotEqual(first, second)
        self.assertEqual(auto_lrc._hf_hub_fingerprint(), second)


if __name__ == "__main__":
    unittest.main()
