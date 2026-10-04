#!/usr/bin/env python3
"""Run WhisperX forced alignment for LRC tools.

This helper is invoked from auto_lrc.py through the local .venv-asr Python
environment so the drag/drop entry point can keep using the normal Python on
PATH.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import warnings
from pathlib import Path

import whisperx


WHISPERX_HELPER_PROTOCOL_REVISION = "whisperx-refine-runtime-provenance-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_provenance(language_code: str) -> dict[str, str] | None:
    """Resolve the exact loaded alignment model identity from the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from whisperx import alignment
    except Exception:
        return None
    model_id = alignment.DEFAULT_ALIGN_MODELS_HF.get(language_code)
    if not isinstance(model_id, str) or not model_id.strip():
        model_id = alignment.DEFAULT_ALIGN_MODELS_TORCH.get(language_code)
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    if model_id in getattr(alignment.torchaudio.pipelines, "__all__", ()):
        # The current Japanese path is HuggingFace; do not invent a cache
        # revision for a torchaudio model whose resolved bundle is unavailable.
        return None
    config_path = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(config_path, str) or not config_path:
        return None
    # Keep the snapshot directory before resolving the symlink; resolving the
    # config itself would incorrectly make ``blobs/`` look like the revision.
    snapshot_dir = Path(config_path).parent
    revision = snapshot_dir.name
    model_files = sorted(
        item for item in snapshot_dir.iterdir()
        if item.is_file()
        and (
            item.name == "pytorch_model.bin"
            or item.name == "model.safetensors"
            or re.match(r"^(pytorch_model|model)-\d+-of-\d+\.(bin|safetensors)$", item.name)
        )
    )
    if not model_files:
        return None
    blob_revisions: list[str] = []
    for model_file in model_files:
        resolved = model_file.resolve()
        if not resolved.is_file():
            return None
        blob_name = resolved.name
        if not re.fullmatch(r"[0-9a-f]{64}", blob_name):
            blob_name = sha256_file(resolved)
        blob_revisions.append(blob_name)
    return {
        "align_model_id": model_id,
        "align_model_revision": revision,
        "align_model_blob_sha256": hashlib.sha256(
            json.dumps(blob_revisions, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if len(blob_revisions) > 1 else blob_revisions[0],
    }


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align transcript segments with WhisperX.")
    parser.add_argument("--audio", required=True, help="16 kHz mono WAV path.")
    parser.add_argument("--transcript", required=True, help="JSON list of transcript segments.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--language", default="ja", help="Alignment language code, default: ja.")
    parser.add_argument("--device", default="auto", help="WhisperX device, default: auto.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    transcript_path = Path(args.transcript)
    output_path = Path(args.output)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(transcript, list):
        raise SystemExit("transcript JSON must be a list")
    device = resolve_device(args.device)

    input_transcript_revision = hashlib.sha256(
        json.dumps(transcript, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, metadata = whisperx.load_align_model(language_code=args.language, device=device)
        result = whisperx.align(
            transcript,
            model,
            metadata,
            args.audio,
            device,
            return_char_alignments=True,
            print_progress=False,
        )
    result["device"] = device
    model = model_provenance(args.language)
    if model is not None:
        result["_runtime_whisperx_alignment_provenance"] = {
            **model,
            "helper_sha256": sha256_file(Path(__file__).resolve()),
            "helper_protocol_revision": WHISPERX_HELPER_PROTOCOL_REVISION,
            "python_sha256": sha256_file(Path(sys.executable).resolve()),
            "whisperx_package_version": importlib.metadata.version("whisperx"),
            "alignment_wav_sha256": sha256_file(Path(args.audio).resolve()),
            "device": device,
            "language": args.language,
            "raw_transcript_revision": input_transcript_revision,
        }

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
