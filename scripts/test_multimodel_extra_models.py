"""Opt-in extra consensus models (LRC_MULTIMODEL_EXTRA_MODELS) must be pinned and explicit."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import multimodel_consensus as mm


class ExtraModelSpecsTest(unittest.TestCase):
    def test_default_is_unchanged_without_env(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(mm.EXTRA_MODELS_ENV, None)
            self.assertEqual(mm.extra_model_specs(), ())
            self.assertEqual(mm.active_model_specs(), tuple(mm.MODEL_SPECS))

    def test_karaoke_spec_is_accepted_and_appended(self) -> None:
        specs = mm.active_model_specs(mm.KARAOKE_JA_SPEC)
        self.assertEqual(specs[:-1], tuple(mm.MODEL_SPECS))
        self.assertEqual(specs[-1], "en-ctc:NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn:roman")
        self.assertIn(mm.KARAOKE_JA_REPO, mm.HF_MODEL_REVISIONS)
        self.assertNotIn(mm.KARAOKE_JA_SPEC, mm.MODEL_SPECS)

    def test_env_is_read_split_and_deduplicated(self) -> None:
        value = f" {mm.KARAOKE_JA_SPEC} ; {mm.KARAOKE_JA_SPEC}, mms,"
        with mock.patch.dict(os.environ, {mm.EXTRA_MODELS_ENV: value}, clear=False):
            self.assertEqual(mm.extra_model_specs(), (mm.KARAOKE_JA_SPEC,))

    def test_unpinned_or_malformed_specs_fail_loudly(self) -> None:
        for bad in (
            "en-ctc:someone/unpinned-model:roman",
            "ja-ctc:torchaudio:native",
            f"en-ctc:{mm.KARAOKE_JA_REPO}:kana",
            f"mms1b:{mm.KARAOKE_JA_REPO}:roman",
            "karaoke",
        ):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                mm.extra_model_specs(bad)


if __name__ == "__main__":
    unittest.main()
