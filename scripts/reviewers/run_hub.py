from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import unicodedata
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REVIEW = PROJECT / "_review"
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import auto_lrc  # noqa: E402
from common import select_songs  # noqa: E402

import torch  # noqa: E402
import torchaudio  # noqa: E402
from huggingface_hub import try_to_load_from_cache  # noqa: E402
from transformers import AutoModelForCTC, AutoProcessor  # noqa: E402

MODEL_ID = auto_lrc._KNOWN_LYRIC_HUBERT_MODEL_ID
MODEL_REVISION = auto_lrc._KNOWN_LYRIC_HUBERT_MODEL_REVISION
MODEL_BLOB_SHA256 = auto_lrc._KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256
SAMPLE_RATE = 16_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_reading(text: str, vocab: dict[str, int]) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    clean: list[str] = []
    for char in normalized:
        if char.isspace():
            if "|" in vocab and (not clean or clean[-1] != "|"):
                clean.append("|")
            continue
        if char in vocab:
            clean.append(char)
            continue
        if unicodedata.category(char).startswith("P"):
            continue
        raise ValueError(f"unsupported-reading-character:{ord(char):04x}")
    while clean and clean[-1] == "|":
        clean.pop()
    if not clean:
        raise ValueError("empty-clean-reading")
    return "".join(clean)


def decode_audio(path: Path) -> torch.Tensor:
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "s16le", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode or not proc.stdout:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace") or "ffmpeg decode failed")
    return torch.frombuffer(bytearray(proc.stdout), dtype=torch.int16).float().div(32768.0).unsqueeze(0)


def local_model_identity() -> tuple[str, str]:
    weight = try_to_load_from_cache(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    if not isinstance(weight, str):
        raise RuntimeError("HuBERT model.safetensors is not cached at pinned revision")
    actual = sha256_file(Path(weight))
    if actual.lower() != MODEL_BLOB_SHA256.lower():
        raise RuntimeError(f"HuBERT model SHA mismatch: {actual}")
    return actual, auto_lrc._known_lyric_hubert_helper_revision()


def song_has_complete_hubert(song: dict[str, object]) -> bool:
    state = song.get("HUB")
    rows = song.get("rows")
    if not isinstance(state, dict) or state.get("status") not in {"OK", "UNSUPPORTED_TEXT"}:
        return False
    if not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        sources = row.get("sources")
        measured = sources.get("HUB") if isinstance(sources, dict) else None
        if not isinstance(measured, dict) or "time" not in measured:
            return False
    return True


def align_song(
    processor,
    model,
    device: str,
    audio_path: Path,
    readings: list[str],
    entry_indexes: list[int],
    audio_revision: str,
) -> tuple[list[dict[str, object]], int]:
    vocab = {str(k): int(v) for k, v in processor.tokenizer.get_vocab().items()}
    blank = processor.tokenizer.pad_token_id
    if not isinstance(blank, int) or blank < 0:
        raise RuntimeError("HuBERT blank token unavailable")
    separator_id = vocab.get("|")
    if separator_id is None:
        raise RuntimeError("HuBERT separator token unavailable")
    cleaned = [clean_reading(reading, vocab) for reading in readings]
    line_tokens = [[vocab[char] for char in line] for line in cleaned]
    flattened: list[int] = []
    for index, tokens in enumerate(line_tokens):
        flattened.extend(tokens)
        if index + 1 < len(line_tokens):
            flattened.append(separator_id)
    waveform = decode_audio(audio_path)
    with torch.inference_mode():
        inputs = processor(
            waveform.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs.input_values.to(device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        logits = model(input_values, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits, dim=-1).cpu()
    targets = torch.tensor(flattened, dtype=torch.int32).unsqueeze(0)
    aligned, scores = torchaudio.functional.forced_align(log_probs, targets, blank=blank)
    spans = torchaudio.functional.merge_tokens(aligned[0], scores[0], blank=blank)
    if len(spans) != len(flattened):
        raise RuntimeError(f"HuBERT token span mismatch: {len(spans)} != {len(flattened)}")
    duration = waveform.shape[1] / SAMPLE_RATE
    seconds_per_frame = duration / log_probs.shape[1]
    result: list[dict[str, object]] = []
    offset = 0
    for row_index, (entry_index, text, tokens) in enumerate(zip(entry_indexes, cleaned, line_tokens)):
        item_spans = spans[offset : offset + len(tokens)]
        offset += len(tokens)
        if row_index + 1 < len(line_tokens):
            offset += 1
        if len(item_spans) != len(text) or not item_spans:
            raise RuntimeError(f"HuBERT row span mismatch at entry {entry_index}")
        span_payload: list[dict[str, object]] = []
        for char, span in zip(text, item_spans):
            start = float(span.start) * seconds_per_frame
            end = float(span.end) * seconds_per_frame
            log_score = float(span.score)
            span_payload.append({
                "char": char,
                "start": round(start, 6),
                "end": round(end, 6),
                "score": round(math.exp(max(-30.0, min(0.0, log_score))), 8),
            })
        request_revision = auto_lrc._timing_content_digest({
            "stage": "arbiter-stage1-whole-song-hubert",
            "entry": entry_index,
            "reading": text,
            "audio_revision": audio_revision,
            "model_revision": MODEL_REVISION,
        })
        supported = auto_lrc._known_lyric_hubert_opening_is_supported(
            entry_index - 1, span_payload, request_revision
        )
        result.append({
            "entry": entry_index,
            "time": round(float(span_payload[0]["start"]), 6),
            "end": round(float(span_payload[-1]["end"]), 6),
            "trusted": bool(supported),
            "score": round(sum(float(item["score"]) for item in span_payload) / len(span_payload), 8),
            "reading": text,
            "opening_supported": bool(supported),
        })
    return result, int(log_probs.shape[1])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--song-id", action="append", default=[])
    result.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    return result


def save(payload: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parser().parse_args()
    payload = json.loads(args.data.read_text(encoding="utf-8-sig"))
    songs = select_songs(payload, args.song_id)
    if not isinstance(payload.get("sources"), dict):
        payload["sources"] = {}
    model_sha, helper_revision = local_model_identity()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    existing_ok = sum(
        isinstance(song.get("HUB"), dict) and song["HUB"].get("status") == "OK"
        for song in songs
    )
    existing_unsupported = sum(
        isinstance(song.get("HUB"), dict) and song["HUB"].get("status") == "UNSUPPORTED_TEXT"
        for song in songs
    )
    existing_failed = sum(
        isinstance(song.get("HUB"), dict) and song["HUB"].get("status") == "FAILED"
        for song in songs
    )
    payload["sources"]["HUB"] = {
        "status": "RUNNING",
        "mode": "whole-song shared S03 vocal stem; pykakasi hiragana; pinned HuBERT CTC",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_blob_sha256": model_sha,
        "helper_protocol_revision": auto_lrc._KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION,
        "helper_revision": helper_revision,
        "device": device,
        "sample_rate": SAMPLE_RATE,
        "selected_song_ids": [str(song["id"]) for song in songs],
        "songs_ok": existing_ok,
        "songs_failed": existing_failed,
        "songs_unsupported_text": existing_unsupported,
    }
    save(payload, args.data)

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    ).to(device).eval()

    songs_ok = existing_ok
    songs_failed = existing_failed
    songs_unsupported = existing_unsupported
    failures: list[dict[str, object]] = []
    total = len(songs)
    for song_index, song in enumerate(songs, start=1):
        if song_has_complete_hubert(song):
            print(f"[{song_index}/{total}] HUB SKIP", flush=True)
            continue
        rows = song["rows"]
        readings: list[str] = []
        unsupported_reason: str | None = None
        for row in rows:
            reading = auto_lrc._known_lyric_hiragana_reading(str(row["text"]))
            if not reading:
                unsupported_reason = "reading-unavailable"
                break
            readings.append(reading)
        if unsupported_reason is None:
            vocab = {str(k): int(v) for k, v in processor.tokenizer.get_vocab().items()}
            try:
                for reading in readings:
                    clean_reading(reading, vocab)
            except ValueError as exc:
                unsupported_reason = str(exc)
        if unsupported_reason is not None:
            songs_unsupported += 1
            for row in rows:
                row["sources"]["HUB"] = {
                    "time": None,
                    "trusted": False,
                    "score": None,
                    "reason": unsupported_reason,
                }
            song["HUB"] = {"status": "UNSUPPORTED_TEXT", "reason": unsupported_reason}
            print(f"[{song_index}/{total}] HUB UNSUPPORTED", flush=True)
            payload["sources"]["HUB"]["songs_unsupported_text"] = songs_unsupported
            save(payload, args.data)
            continue
        try:
            measured_rows, emission_frames = align_song(
                processor,
                model,
                device,
                Path(song["vocal_audio_path"]),
                readings,
                [int(row["entry"]) for row in rows],
                str(song["vocal_audio_sha256"]),
            )
            if len(measured_rows) != len(rows):
                raise RuntimeError("HuBERT row-count mismatch")
            for target, measured in zip(rows, measured_rows):
                target["sources"]["HUB"] = measured
            song["HUB"] = {"status": "OK", "emission_frames": emission_frames}
            songs_ok += 1
            print(f"[{song_index}/{total}] HUB OK rows={len(rows)}", flush=True)
        except Exception as exc:
            songs_failed += 1
            failures.append({
                "id": song["id"],
                "title": song["title"],
                "error_class": type(exc).__name__,
                "error": str(exc)[:2000],
            })
            for row in rows:
                row["sources"]["HUB"] = {
                    "time": None,
                    "trusted": False,
                    "score": None,
                    "error": f"{type(exc).__name__}:{exc}"[:1000],
                }
            song["HUB"] = {"status": "FAILED", "error": f"{type(exc).__name__}:{exc}"[:2000]}
            print(f"[{song_index}/{total}] HUB FAILED class={type(exc).__name__}", flush=True)
        payload["sources"]["HUB"]["songs_ok"] = songs_ok
        payload["sources"]["HUB"]["songs_failed"] = songs_failed
        payload["sources"]["HUB"]["songs_unsupported_text"] = songs_unsupported
        save(payload, args.data)

    payload["sources"]["HUB"]["status"] = "READY" if songs_ok else "UNAVAILABLE"
    payload["sources"]["HUB"]["failures"] = failures
    save(payload, args.data)
    print(json.dumps({"HUB_ok": songs_ok, "HUB_failed": songs_failed, "HUB_unsupported": songs_unsupported}))
    return 0 if songs_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
