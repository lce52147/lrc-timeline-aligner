#!/usr/bin/env python3
"""Run MMS/CTC forced alignment for LRC tools.

This helper is intentionally isolated from auto_lrc.py so the main drag/drop
script can stay lightweight while the ASR venv owns torch/torchaudio imports.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pykakasi
import torch
import torchaudio


SAMPLE_RATE = 16_000


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_stdio()


def decode_audio(audio_path: Path) -> torch.Tensor:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "-",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed: {stderr}")
    if not proc.stdout:
        raise RuntimeError("ffmpeg decoded no audio")
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio).unsqueeze(0)


def compact_latin(text: str) -> str:
    return re.sub(r"[^a-z']+", "", unicodedata.normalize("NFKC", text).lower())


def build_romanizer():
    return pykakasi.kakasi()


def romanize(converter: object, text: str) -> str:
    converted = converter.convert(text)  # type: ignore[attr-defined]
    return compact_latin("".join(str(item.get("hepburn", "")) for item in converted if isinstance(item, dict)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MMS/CTC forced alignment helper for LRC tools.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--transcript", required=True, help="JSON list of lyric line strings.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audio_path = Path(args.audio)
    transcript_path = Path(args.transcript)
    output_path = Path(args.output)

    lines = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(lines, list) or not all(isinstance(item, str) for item in lines):
        raise RuntimeError("--transcript must be a JSON list of strings")

    converter = build_romanizer()
    romanized = [romanize(converter, line) for line in lines]
    if any(not item for item in romanized):
        empty = [index + 1 for index, item in enumerate(romanized) if not item]
        raise RuntimeError(f"Could not romanize transcript entries: {empty}")

    bundle = torchaudio.pipelines.MMS_FA
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    waveform = decode_audio(audio_path)
    duration = waveform.shape[1] / SAMPLE_RATE
    model = bundle.get_model(with_star=False).to(device).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    with torch.inference_mode():
        emissions, _ = model(waveform.to(device))
    emission = emissions[0].cpu()
    tokens = tokenizer(romanized)
    spans = aligner(emission, tokens)
    seconds_per_frame = duration / emission.shape[0]

    rows: list[dict[str, object]] = []
    for index, (line, romaji, item_spans) in enumerate(zip(lines, romanized, spans), start=1):
        if not item_spans:
            rows.append(
                {
                    "entry": index,
                    "text": line,
                    "romaji": romaji,
                    "start": None,
                    "end": None,
                    "ctc_score": None,
                }
            )
            continue
        start = item_spans[0].start * seconds_per_frame
        end = item_spans[-1].end * seconds_per_frame
        score = sum(float(span.score) for span in item_spans) / max(1, len(item_spans))
        rows.append(
            {
                "entry": index,
                "text": line,
                "romaji": romaji,
                "start": round(start, 3),
                "end": round(end, 3),
                "ctc_score": round(score, 6),
                "tokens": len(item_spans),
            }
        )

    payload = {
        "backend": "mms_ctc",
        "device": device,
        "sample_rate": SAMPLE_RATE,
        "duration": round(duration, 3),
        "emission_frames": int(emission.shape[0]),
        "entries": rows,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
