#!/usr/bin/env python3
"""Forced-align known Japanese lyric lines with a Japanese Wav2Vec2 CTC model."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModelForCTC, AutoProcessor


MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
SAMPLE_RATE = 16_000


def decode_audio(audio_path: Path, start: float | None = None, end: float | None = None) -> torch.Tensor:
    command = ["ffmpeg", "-v", "error"]
    if start is not None:
        command.extend(["-ss", f"{start:.3f}"])
    command.extend(["-i", str(audio_path)])
    if end is not None:
        command.extend(["-t", f"{max(0.01, end - start):.3f}"] if start is not None else ["-to", f"{end:.3f}"])
    command.extend(["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"])
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode or not proc.stdout:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace") or "ffmpeg decode failed")
    return torch.frombuffer(bytearray(proc.stdout), dtype=torch.int16).float().div(32768.0).unsqueeze(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    if args.start is not None and args.start < 0:
        raise RuntimeError("--start must be non-negative")
    if args.end is not None and args.end <= 0:
        raise RuntimeError("--end must be positive")
    if args.start is not None and args.end is not None and args.end <= args.start:
        raise RuntimeError("--end must be greater than --start")
    time_offset = float(args.start or 0.0)

    lines = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
    if not isinstance(lines, list) or not all(isinstance(line, str) and line for line in lines):
        raise RuntimeError("--transcript must be a non-empty JSON string list")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCTC.from_pretrained(MODEL_NAME).to(device).eval()
    waveform = decode_audio(Path(args.audio), args.start, args.end)
    with torch.inference_mode():
        logits = model(waveform.to(device)).logits
    log_probs = torch.log_softmax(logits, dim=-1).cpu()

    separator = processor.tokenizer("|").input_ids
    line_tokens = [processor.tokenizer(line).input_ids for line in lines]
    flattened: list[int] = []
    for index, tokens in enumerate(line_tokens):
        flattened.extend(tokens)
        if index + 1 < len(line_tokens):
            flattened.extend(separator)
    targets = torch.tensor(flattened, dtype=torch.int32).unsqueeze(0)
    aligned, scores = torchaudio.functional.forced_align(log_probs, targets, blank=processor.tokenizer.pad_token_id)
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0], blank=processor.tokenizer.pad_token_id)
    if len(spans) != len(flattened):
        raise RuntimeError(f"token span mismatch: {len(spans)} != {len(flattened)}")
    duration = waveform.shape[1] / SAMPLE_RATE
    seconds_per_frame = duration / log_probs.shape[1]
    rows: list[dict[str, object]] = []
    offset = 0
    low_run = 0
    collapse_entries: list[int] = []
    for index, tokens in enumerate(line_tokens, start=1):
        item_spans = spans[offset : offset + len(tokens)]
        offset += len(tokens)
        if index < len(line_tokens):
            offset += len(separator)
        mean_log_score = sum(float(span.score) for span in item_spans) / max(1, len(item_spans))
        if mean_log_score <= -8.0:
            low_run += 1
        else:
            low_run = 0
        if low_run >= 3:
            collapse_entries.append(index)
        rows.append(
            {
                "entry": index,
                "text": lines[index - 1],
                "start": round(time_offset + item_spans[0].start * seconds_per_frame, 3),
                "end": round(time_offset + item_spans[-1].end * seconds_per_frame, 3),
                "ja_ctc_log_score": round(mean_log_score, 6),
                "ctc_score": round(math.exp(max(-20.0, mean_log_score)), 8),
                "tokens": len(item_spans),
                "token_spans": [
                    {
                        "token": processor.tokenizer.convert_ids_to_tokens(int(token)),
                        "start": round(time_offset + span.start * seconds_per_frame, 3),
                        "end": round(time_offset + span.end * seconds_per_frame, 3),
                        "log_score": round(float(span.score), 6),
                    }
                    for token, span in zip(tokens, item_spans)
                ],
            }
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "backend": "japanese_wav2vec2_ctc",
                "model": MODEL_NAME,
                "device": device,
                "sample_rate": SAMPLE_RATE,
                "duration": round(duration, 3),
                "window_start": time_offset,
                "window_end": round(time_offset + duration, 3),
                "emission_frames": int(log_probs.shape[1]),
                "collapse_entries": collapse_entries,
                "collapse_detected": bool(collapse_entries),
                "entries": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
