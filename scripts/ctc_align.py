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


def decode_audio(audio_path: Path, start: float | None = None, end: float | None = None) -> torch.Tensor:
    command = ["ffmpeg", "-v", "error"]
    if start is not None:
        command.extend(["-ss", f"{start:.3f}"])
    command.extend(["-i", str(audio_path)])
    if end is not None:
        if start is not None:
            command.extend(["-t", f"{max(0.01, end - start):.3f}"])
        else:
            command.extend(["-to", f"{end:.3f}"])
    command.extend(
        [
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "-",
        ]
    )
    proc = subprocess.run(
        command,
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
    # Apostrophes in English contractions are orthographic only.  Keeping one
    # in a forced-alignment target can split ``I'll`` into a fake silent token
    # and let the initial I attach to the preceding lyric tail.
    return re.sub(r"[^a-z]+", "", unicodedata.normalize("NFKC", text).lower())


def build_romanizer():
    return pykakasi.kakasi()


def romanize(converter: object, text: str) -> str:
    converted = converter.convert(text)  # type: ignore[attr-defined]
    return compact_latin("".join(str(item.get("hepburn", "")) for item in converted if isinstance(item, dict)))


def build_unidic_tagger() -> object | None:
    try:
        from fugashi import Tagger

        return Tagger()
    except (ImportError, RuntimeError):
        return None


def transcript_romanize(converter: object, tagger: object | None, text: str) -> tuple[str, str]:
    if tagger is not None:
        try:
            reading = "".join(
                str(getattr(token.feature, "pron", "") or token.surface) for token in tagger(text)  # type: ignore[operator]
            )
            romanized = romanize(converter, reading)
            if romanized:
                return romanized, "unidic-reading"
        except (RuntimeError, ValueError):
            pass
    return romanize(converter, text), "pykakasi-orthography"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MMS/CTC forced alignment helper for LRC tools.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--transcript", required=True, help="JSON list of lyric line strings.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--romanization", choices=("auto", "pykakasi", "unidic"), default="auto")
    parser.add_argument("--start", type=float, help="Optional source-audio window start in seconds.")
    parser.add_argument("--end", type=float, help="Optional source-audio window end in seconds.")
    return parser


def token_peak_candidates(
    emission: torch.Tensor,
    token_id: int,
    start_frame: int,
    end_frame: int,
    seconds_per_frame: float,
    time_offset: float = 0.0,
    limit: int = 8,
) -> list[dict[str, float]]:
    if end_frame <= start_frame:
        return []
    values = emission[start_frame:end_frame, token_id].exp().numpy()
    if values.size == 0:
        return []
    peaks: list[tuple[float, int]] = []
    for offset, value in enumerate(values):
        left = values[offset - 1] if offset > 0 else -1.0
        right = values[offset + 1] if offset + 1 < values.size else -1.0
        if value >= left and value >= right:
            peaks.append((float(value), start_frame + offset))
    peaks.sort(reverse=True)
    return [
        {
            "time": round(time_offset + frame * seconds_per_frame, 3),
            "score": round(score, 6),
        }
        for score, frame in peaks[:limit]
    ]


def configure_cuda_cpu_feature_extractor(model: torch.nn.Module) -> None:
    """Keep the full-song convolution frontend on CPU and transformer on CUDA.

    Wav2Vec2's early convolutions create very large intermediate tensors for
    multi-minute waveforms. The encoder still processes all frames on CUDA,
    without acoustic windowing or changed alignment/token semantics.
    """
    from types import MethodType

    extractor = model.model.feature_extractor.cpu()
    target_device = next(model.model.encoder.parameters()).device
    # The persistent CTC worker reuses the same model across requests.
    # Always wrap the original forward, never the wrapper from an earlier run.
    if not hasattr(extractor, "_lrc_ctc_original_forward"):
        extractor._lrc_ctc_original_forward = extractor.forward
    original_forward = extractor._lrc_ctc_original_forward

    def cpu_feature_forward(self, waveform, length):
        # Preserve full-length group normalization and convolution context.
        features, lengths = original_forward(
            waveform.to("cpu"),
            length.to("cpu") if length is not None else None,
        )
        return (
            features.to(target_device),
            lengths.to(target_device) if lengths is not None else None,
        )

    extractor.forward = MethodType(cpu_feature_forward, extractor)
    print("CTC CUDA hybrid: full-song feature extractor=cpu, encoder=cuda",
          file=sys.stderr, flush=True)


def main() -> int:
    args = build_parser().parse_args()
    audio_path = Path(args.audio)
    transcript_path = Path(args.transcript)
    output_path = Path(args.output)
    if args.start is not None and args.start < 0:
        raise RuntimeError("--start must be non-negative")
    if args.end is not None and args.end <= 0:
        raise RuntimeError("--end must be positive")
    if args.start is not None and args.end is not None and args.end <= args.start:
        raise RuntimeError("--end must be greater than --start")
    time_offset = float(args.start or 0.0)

    lines = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(lines, list) or not all(isinstance(item, str) for item in lines):
        raise RuntimeError("--transcript must be a JSON list of strings")

    converter = build_romanizer()
    tagger = build_unidic_tagger() if args.romanization != "pykakasi" else None
    if args.romanization == "unidic" and tagger is None:
        raise RuntimeError("--romanization unidic requires fugashi with a UniDic dictionary")
    pairs = [transcript_romanize(converter, tagger, line) for line in lines]
    romanized = [pair[0] for pair in pairs]
    romanization_sources = [pair[1] for pair in pairs]
    if any(not item for item in romanized):
        empty = [index + 1 for index, item in enumerate(romanized) if not item]
        raise RuntimeError(f"Could not romanize transcript entries: {empty}")

    bundle = torchaudio.pipelines.MMS_FA
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    waveform = decode_audio(audio_path, args.start, args.end)
    duration = waveform.shape[1] / SAMPLE_RATE
    model = bundle.get_model(with_star=False).to(device).eval()
    if str(device).startswith("cuda"):
        configure_cuda_cpu_feature_extractor(model)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    # FP32 memory-efficient SDPA keeps full-song context while avoiding the
    # quadratic attention matrix allocation on CUDA. Never silently fall back
    # to the high-memory math backend if this kernel is unavailable.
    if str(device).startswith("cuda"):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attention_context = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        from contextlib import nullcontext
        attention_context = nullcontext()
    with torch.inference_mode(), attention_context:
        emissions, _ = model(waveform if str(device).startswith("cuda") else waveform.to(device))
    emission = emissions[0].cpu()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
        print(
            "CTC CUDA peak allocated="
            f"{torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB "
            "reserved="
            f"{torch.cuda.max_memory_reserved(device) / 2**20:.1f} MiB",
            file=sys.stderr, flush=True,
        )
    tokens = tokenizer(romanized)
    spans = aligner(emission, tokens)
    seconds_per_frame = duration / emission.shape[0]

    rows: list[dict[str, object]] = []
    previous_end_frame = 0
    for index, (line, romaji, item_spans, romanization_source) in enumerate(
        zip(lines, romanized, spans, romanization_sources), start=1
    ):
        if not item_spans:
            rows.append(
                {
                    "entry": index,
                    "text": line,
                    "romaji": romaji,
                    "romanization_source": romanization_source,
                    "start": None,
                    "end": None,
                    "ctc_score": None,
                }
            )
            continue
        start = time_offset + item_spans[0].start * seconds_per_frame
        end = time_offset + item_spans[-1].end * seconds_per_frame
        score = sum(float(span.score) for span in item_spans) / max(1, len(item_spans))
        first_token_id = tokenizer.dictionary.get(romaji[0])
        first_token_candidates: list[dict[str, float]] = []
        if first_token_id is not None:
            first_token_candidates = token_peak_candidates(
                emission,
                int(first_token_id),
                max(0, item_spans[0].start - int(round(2.5 / seconds_per_frame))),
                max(previous_end_frame + 1, item_spans[0].start),
                seconds_per_frame,
                time_offset,
            )
        token_spans = [
            {
                "char": char,
                "start": round(time_offset + span.start * seconds_per_frame, 3),
                "end": round(time_offset + span.end * seconds_per_frame, 3),
                "score": round(float(span.score), 6),
            }
            for char, span in zip(romaji, item_spans)
        ]
        rows.append(
            {
                "entry": index,
                "text": line,
                "romaji": romaji,
                "romanization_source": romanization_source,
                "start": round(start, 3),
                "end": round(end, 3),
                "ctc_score": round(score, 6),
                "tokens": len(item_spans),
                "token_spans": token_spans,
                "first_token_candidates": first_token_candidates,
            }
        )
        previous_end_frame = item_spans[-1].end

    payload = {
        "backend": "mms_ctc",
        "romanization": "unidic-reading" if tagger is not None else "pykakasi-orthography",
        "device": device,
        "sample_rate": SAMPLE_RATE,
        "duration": round(duration, 3),
        "window_start": time_offset,
        "window_end": round(time_offset + duration, 3),
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
