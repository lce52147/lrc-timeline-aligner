from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from multimodel_consensus import MODEL_SPECS  # noqa: E402


HF_REVISIONS = {
    "facebook/mms-1b-all": "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd",
    "facebook/wav2vec2-large-960h-lv60-self": "54074b1c16f4de6a5ad59affb4caa8f2ea03a119",
    "jonatasgrosman/wav2vec2-large-xlsr-53-english": "569a6236e92bd5f7652a0420bfe9bb94c5664080",
}
JAPANESE_XLSR_REPO = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
JAPANESE_XLSR_EXPECTED = "cf031e020336460d15a417eba710bbc5bb43be9a"
HUBP_REPO = "prj-beatrice/japanese-hubert-base-phoneme-ctc"
HUBP_REVISION = "1ec4eb3c45b2a1cafb7c477d447df34ca03070f2"
KNOWN_HUBERT_REPO = "TKU410410103/hubert-base-japanese-asr"
KNOWN_HUBERT_REVISION = "9206637a0f07d78aa9101ff9439a9d7fe1a43778"
MMS1B_FILES = (
    "adapter.eng.safetensors",
    "adapter.jpn.safetensors",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _repo_from_spec(spec: str) -> str | None:
    if spec.startswith("mms1b:"):
        return "facebook/mms-1b-all"
    if spec.startswith(("en-ctc:", "ja-ctc:")):
        repo = spec.split(":", 2)[1]
        if repo != "torchaudio":
            return repo
    return None


def _download_snapshot(
    repo: str,
    revision: str,
    *,
    local_dir: Path | None = None,
    allow_patterns: tuple[str, ...] | None = None,
) -> Path:
    print(f"PREFETCH hf {repo}@{revision}", flush=True)
    path = snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(local_dir) if local_dir is not None else None,
        allow_patterns=allow_patterns,
        max_workers=2,
    )
    return Path(path)


def _prefetch_hf() -> None:
    repos = {repo for spec in MODEL_SPECS if (repo := _repo_from_spec(spec))}
    for repo in sorted(repos):
        if repo == JAPANESE_XLSR_REPO:
            continue
        revision = HF_REVISIONS.get(repo)
        if revision is None:
            raise RuntimeError(f"un-pinned MODEL_SPECS repo: {repo}")
        patterns = MMS1B_FILES if repo == "facebook/mms-1b-all" else None
        path = _download_snapshot(repo, revision, allow_patterns=patterns)
        if path.name != revision:
            raise RuntimeError(f"snapshot revision mismatch for {repo}: {path.name} != {revision}")

    japanese = _download_snapshot(JAPANESE_XLSR_REPO, "main")
    if japanese.name != JAPANESE_XLSR_EXPECTED:
        print(
            f"WARN japanese XLSR main moved: {japanese.name} != {JAPANESE_XLSR_EXPECTED}",
            file=sys.stderr,
            flush=True,
        )

    _download_snapshot(
        HUBP_REPO,
        HUBP_REVISION,
        local_dir=ROOT / "models" / "hubert-phoneme-ctc",
    )
    _download_snapshot(KNOWN_HUBERT_REPO, KNOWN_HUBERT_REVISION)


def _prefetch_torchaudio() -> None:
    bundles = (
        ("MMS_FA", torchaudio.pipelines.MMS_FA),
        ("WAV2VEC2_ASR_LARGE_960H", torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H),
        ("WAV2VEC2_ASR_BASE_960H", torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H),
    )
    for name, bundle in bundles:
        print(f"PREFETCH torchaudio {name}", flush=True)
        model = bundle.get_model()
        del model
        gc.collect()


def main() -> int:
    _prefetch_hf()
    _prefetch_torchaudio()
    print("PREFETCH complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
