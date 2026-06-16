#!/usr/bin/env python3
"""Run WhisperX forced alignment for LRC tools.

This helper is invoked from auto_lrc.py through the local .venv-asr Python
environment so the drag/drop entry point can keep using the normal Python on
PATH.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import whisperx


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

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
