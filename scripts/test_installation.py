from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import multimodel_consensus


ROOT = Path(__file__).resolve().parents[1]


class InstallationContractTests(unittest.TestCase):
    def test_asr_lock_keeps_validated_versions_without_torch_runtime_builds(self) -> None:
        lines = (ROOT / "requirements-asr.lock.txt").read_text(encoding="utf-8-sig").splitlines()
        requirements = [line for line in lines if line and not line.startswith("#")]
        self.assertIn("whisperx==3.8.6", requirements)
        self.assertIn("transformers==4.57.6", requirements)
        self.assertIn("demucs==4.0.1", requirements)
        blocked = ("torch==", "torchaudio==", "torchvision==", "triton==", "nvidia-")
        self.assertFalse(any(line.lower().startswith(blocked) for line in requirements))
        self.assertFalse(any("@" in line or line.startswith("-e ") for line in requirements))

    def test_wrapper_prefers_repo_venv_and_repo_ffmpeg(self) -> None:
        source = (ROOT / "align-lrc.ps1").read_text(encoding="utf-8-sig")
        local_python = 'Join-Path $scriptDir ".venv-asr\\Scripts\\python.exe"'
        path_python = "Get-Command python -ErrorAction SilentlyContinue"
        self.assertIn(local_python, source)
        self.assertIn(path_python, source)
        self.assertLess(source.index(local_python), source.index(path_python))
        self.assertIn('Join-Path $scriptDir "tools\\ffmpeg\\bin"', source)
        self.assertIn('& $pythonExe @pipelineArgs', source)
        self.assertIn('& $pythonExe @argsList', source)

    def test_torch_identity_uses_real_torch_home_hub_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "hub" / "checkpoints" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"isolated-checkpoint")
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with mock.patch.dict(os.environ, {"TORCH_HOME": str(root)}, clear=False):
                actual = multimodel_consensus._torch_blob("model.pt")
        self.assertEqual(actual, expected)

    def test_hf_identity_prefers_pinned_snapshot_when_cache_has_extra_snapshots(self) -> None:
        repo = "facebook/wav2vec2-large-960h-lv60-self"
        revision = multimodel_consensus.HF_MODEL_REVISIONS[repo]
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "hub"
            snapshots = hub / "models--facebook--wav2vec2-large-960h-lv60-self" / "snapshots"
            pinned = snapshots / revision
            extra = snapshots / "bdb144224e9bb4a242a52f223186f120f6589a6b"
            pinned.mkdir(parents=True)
            extra.mkdir(parents=True)
            (pinned / "pytorch_model.bin").write_bytes(b"pinned")
            (extra / "model.safetensors").write_bytes(b"converted")
            with mock.patch.dict(os.environ, {"HUGGINGFACE_HUB_CACHE": str(hub)}, clear=False):
                self.assertEqual(multimodel_consensus._hf_snapshot(repo, revision), pinned)

    def test_hf_identity_prefers_safetensors_when_both_weight_formats_are_cached(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp)
            safe = snapshot / "model.safetensors"
            safe.write_bytes(b"safe")
            (snapshot / "pytorch_model.bin").write_bytes(b"legacy")
            blob = hashlib.sha256(safe.read_bytes()).hexdigest()
            expected = hashlib.sha256(f"model.safetensors:{blob}".encode("utf-8")).hexdigest()
            self.assertEqual(multimodel_consensus._snapshot_blob(snapshot), expected)

    def test_setup_pins_external_binary_archives(self) -> None:
        source = (ROOT / "setup.ps1").read_text(encoding="utf-8-sig")
        self.assertIn('ffmpegVersion = "8.1.2"', source)
        self.assertIn("db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec", source)
        self.assertIn('whisperTag = "b5454"', source)
        self.assertIn("afef0b881c500958921c3f5523b50e59ee2ec9b6f5cbd25b324c51ed308a957a", source)
        self.assertIn("6ba69e3482d7826214f90a6a9c84ca07782aec1e1d0c6a7c30c994fd5d816ccb", source)
        self.assertIn("64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2", source)

    def test_prefetch_limits_mms_to_production_files(self) -> None:
        source = (ROOT / "scripts" / "prefetch_models.py").read_text(encoding="utf-8-sig")
        self.assertIn('HF_HUB_DISABLE_XET", "1"', source)
        self.assertIn('"adapter.eng.safetensors"', source)
        self.assertIn('"adapter.jpn.safetensors"', source)
        self.assertIn('ENGLISH_XLSR_REPO = "jonatasgrosman/wav2vec2-large-xlsr-53-english"', source)
        self.assertIn('"model.safetensors"', source)
        self.assertIn('else ENGLISH_XLSR_FILES if repo == ENGLISH_XLSR_REPO', source)


if __name__ == "__main__":
    unittest.main()
