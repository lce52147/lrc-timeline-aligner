"""Tests for ``scripts/runtime_paths.py`` (LRC 1.5 W2)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import runtime_paths  # noqa: E402

PROJECT_ROOT = SCRIPTS.parent
_ENV_NAMES = (
    "LRC_VENV_PYTHON", "LRC_WHISPER_CLI", "LRC_WHISPER_MODEL", "LRC_FFMPEG",
    "HF_HUB_CACHE", "HF_HOME", "LRC_MODEL_CACHE_DIR", "LRC_REPORT_DIR",
)


def _clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _ENV_NAMES}
    env.update(extra)
    return env


class DefaultPathTests(unittest.TestCase):
    def test_defaults_match_v14_layout(self) -> None:
        with mock.patch.dict(os.environ, _clean_env(), clear=True):
            self.assertEqual(runtime_paths.project_root(), PROJECT_ROOT)
            self.assertEqual(
                runtime_paths.whisper_cli(),
                PROJECT_ROOT / "tools" / "whisper.cpp" / "Release" / "whisper-cli.exe",
            )
            self.assertEqual(
                runtime_paths.whisper_model(),
                PROJECT_ROOT / "models" / "whisper.cpp" / "ggml-large-v3.bin",
            )
            self.assertEqual(
                runtime_paths.venv_python(),
                PROJECT_ROOT / ".venv-asr" / "Scripts" / "python.exe",
            )
            self.assertEqual(runtime_paths.ffmpeg(), "ffmpeg")
            self.assertEqual(runtime_paths.cache_root(), PROJECT_ROOT / "outputs" / "cache")
            self.assertEqual(
                runtime_paths.model_cache_dir(), PROJECT_ROOT / "outputs" / "cache" / "model-outputs"
            )
            self.assertEqual(
                runtime_paths.vocal_cache_dir(), PROJECT_ROOT / "outputs" / "cache" / "vocals"
            )
            self.assertEqual(runtime_paths.gpu_lock_dir(), PROJECT_ROOT / "outputs" / "cache")
            self.assertEqual(runtime_paths.reports_dir(), PROJECT_ROOT / "outputs" / "reports")

    def test_default_identity_strings_unchanged(self) -> None:
        # Cache identities embed str(path); the default must equal the 1.4 expression.
        with mock.patch.dict(os.environ, _clean_env(), clear=True):
            legacy = Path(runtime_paths.__file__).resolve().parents[1] / "tools" / "whisper.cpp" / "Release" / "whisper-cli.exe"
            self.assertEqual(str(runtime_paths.whisper_cli()), str(legacy))


class OverrideTests(unittest.TestCase):
    def test_env_overrides_are_expanded_and_resolved(self) -> None:
        base = Path(os.getcwd()).resolve()
        env = _clean_env(
            LRC_BASE=str(base),
            LRC_VENV_PYTHON=os.path.join("${LRC_BASE}", "venv", "python"),
            LRC_WHISPER_CLI=os.path.join("${LRC_BASE}", "cli", "whisper-cli"),
            LRC_WHISPER_MODEL=os.path.join("${LRC_BASE}", "model.bin"),
            LRC_FFMPEG=os.path.join("${LRC_BASE}", "bin", "ffmpeg"),
            LRC_MODEL_CACHE_DIR=os.path.join("${LRC_BASE}", "cache_x"),
            LRC_REPORT_DIR=os.path.join("${LRC_BASE}", "reports_x"),
        )
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(runtime_paths.venv_python(), base / "venv" / "python")
            self.assertEqual(runtime_paths.whisper_cli(), base / "cli" / "whisper-cli")
            self.assertEqual(runtime_paths.whisper_model(), base / "model.bin")
            self.assertEqual(runtime_paths.ffmpeg(), str(base / "bin" / "ffmpeg"))
            self.assertEqual(runtime_paths.cache_root(), base / "cache_x")
            self.assertEqual(runtime_paths.model_cache_dir(), base / "cache_x" / "model-outputs")
            self.assertEqual(runtime_paths.vocal_cache_dir(), base / "cache_x" / "vocals")
            self.assertEqual(runtime_paths.reports_dir(), base / "reports_x")

    def test_gpu_lock_dir_follows_model_cache_dir(self) -> None:
        base = Path(os.getcwd()).resolve()
        with mock.patch.dict(os.environ, _clean_env(LRC_MODEL_CACHE_DIR=str(base / "cache_y")), clear=True):
            self.assertEqual(runtime_paths.gpu_lock_dir(), base / "cache_y")
        with mock.patch.dict(os.environ, _clean_env(), clear=True):
            self.assertEqual(runtime_paths.gpu_lock_dir(), PROJECT_ROOT / "outputs" / "cache")

    def test_blank_override_falls_back_to_default(self) -> None:
        with mock.patch.dict(os.environ, _clean_env(LRC_REPORT_DIR="   "), clear=True):
            self.assertEqual(runtime_paths.reports_dir(), PROJECT_ROOT / "outputs" / "reports")


class HfHubCacheTests(unittest.TestCase):
    def test_hf_home_with_and_without_hub_suffix_agree(self) -> None:
        base = Path(os.getcwd()).resolve() / "huggingface"
        with_hub = runtime_paths.hf_hub_cache({"HF_HOME": str(base / "hub")})
        without_hub = runtime_paths.hf_hub_cache({"HF_HOME": str(base)})
        self.assertEqual(with_hub, without_hub)
        self.assertEqual(with_hub.name, "hub")

    def test_hf_hub_cache_wins_over_hf_home(self) -> None:
        base = Path(os.getcwd()).resolve()
        got = runtime_paths.hf_hub_cache({"HF_HUB_CACHE": str(base / "direct"), "HF_HOME": str(base / "home")})
        self.assertEqual(got, base / "direct")

    def test_doctor_and_multimodel_use_same_rule(self) -> None:
        import doctor
        import multimodel_consensus

        base = Path(os.getcwd()).resolve() / "hf"
        env = _clean_env(HF_HOME=str(base))
        with mock.patch.dict(os.environ, env, clear=True):
            from_doctor = doctor._hf_hub_path({"HF_HOME": str(base), "HF_HUB_CACHE": None})
            candidates = runtime_paths.hf_hub_cache_candidates()
            self.assertEqual(from_doctor, base / "hub")
            self.assertEqual(candidates[0], base / "hub")
            # multimodel_consensus looks up snapshots under the same roots.
            with mock.patch.object(multimodel_consensus.runtime_paths, "hf_hub_cache_candidates", return_value=[base / "hub"]) as patched:
                self.assertIsNone(multimodel_consensus._hf_snapshot("org/repo", "rev"))
                patched.assert_called_once()


class DescribeTests(unittest.TestCase):
    def test_describe_reports_sources(self) -> None:
        base = Path(os.getcwd()).resolve()
        with mock.patch.dict(os.environ, _clean_env(LRC_REPORT_DIR=str(base / "r")), clear=True):
            rows = {row["name"]: row for row in runtime_paths.describe()}
        self.assertEqual(rows["reports_dir"]["source"], "env:LRC_REPORT_DIR")
        self.assertEqual(rows["whisper_cli"]["source"], "default")
        self.assertIn("exists", rows["project_root"])
        self.assertTrue(rows["project_root"]["exists"])


if __name__ == "__main__":
    unittest.main()
