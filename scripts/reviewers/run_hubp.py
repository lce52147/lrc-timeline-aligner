from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoModelForCTC, AutoProcessor


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "hubert-phoneme-ctc"
SAMPLE_RATE = 16_000
CHUNK_SECONDS = 30.0
OVERLAP_SECONDS = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_audio(path: Path) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[-2000:])
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--max-songs", type=int, default=0)
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return result


def save(output: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def infer_emissions(processor, model, audio: np.ndarray, device: str) -> tuple[torch.Tensor, list[float], list[dict[str, object]]]:
    chunk_samples = int(round(CHUNK_SECONDS * SAMPLE_RATE))
    overlap_samples = int(round(OVERLAP_SECONDS * SAMPLE_RATE))
    step = chunk_samples - overlap_samples
    all_logits: list[torch.Tensor] = []
    all_starts: list[float] = []
    chunks: list[dict[str, object]] = []
    start_sample = 0
    chunk_index = 0
    while start_sample < len(audio):
        end_sample = min(len(audio), start_sample + chunk_samples)
        chunk = audio[start_sample:end_sample]
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.inference_mode():
            logits = model(input_values).logits[0].float().cpu()
        frame_count = int(logits.shape[0])
        if frame_count <= 0:
            raise RuntimeError("HUBP empty emission")
        start_sec = start_sample / SAMPLE_RATE
        duration = len(chunk) / SAMPLE_RATE
        frame_seconds = duration / frame_count
        frame_starts = [start_sec + i * frame_seconds for i in range(frame_count)]
        keep_left = start_sec + (OVERLAP_SECONDS / 2.0 if start_sample > 0 else 0.0)
        keep_right = start_sec + duration - (OVERLAP_SECONDS / 2.0 if end_sample < len(audio) else 0.0)
        keep = [i for i, value in enumerate(frame_starts) if value >= keep_left and value < keep_right]
        if not keep:
            raise RuntimeError(f"HUBP empty retained chunk {chunk_index}")
        left = keep[0]
        right = keep[-1] + 1
        all_logits.append(logits[left:right])
        all_starts.extend(frame_starts[left:right])
        chunks.append({
            "index": chunk_index, "start": round(start_sec, 6),
            "end": round(start_sec + duration, 6), "raw_frames": frame_count,
            "kept_frames": right - left, "keep_left": round(keep_left, 6), "keep_right": round(keep_right, 6),
        })
        if end_sample >= len(audio):
            break
        start_sample += step
        chunk_index += 1
    return torch.cat(all_logits, dim=0).log_softmax(dim=-1), all_starts, chunks


def main() -> int:
    args = parser().parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    model_file = MODEL / "model.safetensors"
    if sha256_file(model_file) != str(plan["model_sha256"]):
        raise RuntimeError("HUBP model hash mismatch")
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(MODEL, local_files_only=True, use_safetensors=True).to(device).eval()
    if args.output.exists():
        output = json.loads(args.output.read_text(encoding="utf-8-sig"))
        if output.get("model_sha256") != plan.get("model_sha256"):
            raise RuntimeError("existing HUBP output model identity mismatch")
    else:
        output = {
            "schema": 1, "source": "HUBP", "status": "RUNNING", "device": device,
            "model_revision": plan["model_revision"], "model_sha256": plan["model_sha256"],
            "sample_rate": SAMPLE_RATE, "chunk_seconds": CHUNK_SECONDS,
            "overlap_seconds": OVERLAP_SECONDS, "rows": {}, "songs": {}, "failures": [],
        }
    rows_out: dict[str, object] = output["rows"]  # type: ignore[assignment]
    songs_out: dict[str, object] = output["songs"]  # type: ignore[assignment]
    processed_songs = 0
    total = len(plan["songs"])

    for song in plan["songs"]:
        song_id = str(song["id"])
        if song_id in songs_out and songs_out[song_id].get("status") in {"OK", "UNSUPPORTED"}:
            continue
        ready_rows = [row for row in song["rows"] if row["status"] == "READY"]
        if not ready_rows:
            for row in song["rows"]:
                key = f"{song_id}::{row['entry']}"
                rows_out[key] = {
                    "song_id": song_id, "entry": row["entry"], "time": None,
                    "status": "MISSING", "reason": row.get("reason", "unsupported"),
                }
            songs_out[song_id] = {"status": "UNSUPPORTED", "reason": "no-supported-rows"}
            save(output, args.output)
            continue
        if args.max_songs and processed_songs >= args.max_songs:
            output["status"] = "PARTIAL"
            save(output, args.output)
            print(json.dumps({"status": "PARTIAL", "processed_songs": processed_songs}))
            return 0
        try:
            audio_path = Path(str(song["vocal_audio_path"]))
            if sha256_file(audio_path) != str(song["vocal_audio_sha256"]):
                raise RuntimeError("vocal hash mismatch")
            audio = decode_audio(audio_path)
            log_probs, frame_starts, chunks = infer_emissions(processor, model, audio, device)
            target_ids: list[int] = []
            row_ranges: list[tuple[dict[str, object], int, int]] = []
            for row in ready_rows:
                left = len(target_ids)
                target_ids.extend(int(value) for value in row["token_ids"])
                row_ranges.append((row, left, len(target_ids)))
            targets = torch.tensor(target_ids, dtype=torch.int32).unsqueeze(0)
            path, scores = torchaudio.functional.forced_align(log_probs.unsqueeze(0), targets, blank=0)
            spans = torchaudio.functional.merge_tokens(path[0], scores[0], blank=0)
            if len(spans) != len(target_ids):
                raise RuntimeError(f"forced-align span cardinality {len(spans)} != {len(target_ids)}")
            observed_ids = [int(span.token) for span in spans]
            if observed_ids != target_ids:
                raise RuntimeError("forced-align token sequence mismatch")
            for row, left, right in row_ranges:
                first = spans[left]
                frame_index = int(first.start)
                if not 0 <= frame_index < len(frame_starts):
                    raise RuntimeError("forced-align frame index out of range")
                time_value = float(frame_starts[frame_index])
                key = f"{song_id}::{row['entry']}"
                rows_out[key] = {
                    "song_id": song_id, "entry": row["entry"], "time": round(time_value, 6),
                    "status": "OK", "first_phoneme": row["phonemes"][0],
                    "first_score": round(float(first.score), 8), "phoneme_count": right - left,
                }
            for row in song["rows"]:
                if row["status"] == "READY":
                    continue
                key = f"{song_id}::{row['entry']}"
                rows_out[key] = {
                    "song_id": song_id, "entry": row["entry"], "time": None,
                    "status": "MISSING", "reason": row.get("reason", "unsupported"),
                }
            songs_out[song_id] = {
                "status": "OK", "duration": round(len(audio) / SAMPLE_RATE, 6),
                "emission_frames": len(frame_starts), "target_phonemes": len(target_ids), "chunks": chunks,
            }
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"[:2000]
            songs_out[song_id] = {"status": "FAILED", "reason": reason}
            output["failures"].append({"song_id": song_id, "error": reason})
            for row in song["rows"]:
                key = f"{song_id}::{row['entry']}"
                rows_out[key] = {
                    "song_id": song_id, "entry": row["entry"], "time": None,
                    "status": "FAILED", "reason": reason,
                }
        processed_songs += 1
        save(output, args.output)
        print(f"HUBP {song['song_index']}/{total} {songs_out[song_id]['status']}", flush=True)

    ok = sum(1 for item in rows_out.values() if item.get("status") == "OK")
    missing = sum(1 for item in rows_out.values() if item.get("status") == "MISSING")
    failed = sum(1 for item in rows_out.values() if item.get("status") == "FAILED")
    output["status"] = "COMPLETE"
    output["summary"] = {"ok": ok, "missing": missing, "failed": failed, "total": len(rows_out)}
    save(output, args.output)
    print(json.dumps(output["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
