from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REVIEW = PROJECT / "_review"
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import japanese_ctc_align as jactc  # noqa: E402
from common import select_songs  # noqa: E402

import torch  # noqa: E402
import torchaudio  # noqa: E402
from huggingface_hub import try_to_load_from_cache  # noqa: E402
from transformers import AutoModelForCTC, AutoProcessor  # noqa: E402

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_revision_from_cache_path(path: Path) -> str:
    parts = path.parts
    try:
        index = parts.index("snapshots")
    except ValueError as exc:
        raise RuntimeError(f"not a Hugging Face snapshot path: {path}") from exc
    if index + 1 >= len(parts):
        raise RuntimeError(f"snapshot revision missing from path: {path}")
    return parts[index + 1]


def local_model_identity() -> tuple[str, str, str | None]:
    config = try_to_load_from_cache(jactc.MODEL_NAME, "config.json", revision="main")
    if not isinstance(config, str):
        raise RuntimeError("XLSR config is not cached")
    revision = snapshot_revision_from_cache_path(Path(config))
    weight_path: Path | None = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        value = try_to_load_from_cache(jactc.MODEL_NAME, name, revision=revision)
        if isinstance(value, str):
            weight_path = Path(value)
            break
    weight_sha = sha256_file(weight_path) if weight_path is not None else None
    return revision, sha256_file(Path(config)), weight_sha


def align_song(processor, model, device: str, audio: Path, lines: list[str]) -> tuple[list[dict[str, object]], list[int], int]:
    waveform = jactc.decode_audio(audio)
    with torch.inference_mode():
        logits = model(waveform.to(device)).logits
    log_probs = torch.log_softmax(logits, dim=-1).cpu()
    separator = processor.tokenizer("|").input_ids
    line_tokens = [processor.tokenizer(line).input_ids for line in lines]
    if any(not tokens for tokens in line_tokens):
        raise RuntimeError("empty XLSR tokenization")
    flattened: list[int] = []
    for index, tokens in enumerate(line_tokens):
        flattened.extend(tokens)
        if index + 1 < len(line_tokens):
            flattened.extend(separator)
    targets = torch.tensor(flattened, dtype=torch.int32).unsqueeze(0)
    aligned, scores = torchaudio.functional.forced_align(
        log_probs, targets, blank=processor.tokenizer.pad_token_id
    )
    spans = torchaudio.functional.merge_tokens(
        aligned[0], scores[0], blank=processor.tokenizer.pad_token_id
    )
    if len(spans) != len(flattened):
        raise RuntimeError(f"token span mismatch: {len(spans)} != {len(flattened)}")
    duration = waveform.shape[1] / jactc.SAMPLE_RATE
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
        rows.append({
            "entry": index,
            "time": round(item_spans[0].start * seconds_per_frame, 6),
            "end": round(item_spans[-1].end * seconds_per_frame, 6),
            "log_score": round(mean_log_score, 8),
            "score": round(math.exp(max(-20.0, mean_log_score)), 10),
            "tokens": len(tokens),
        })
    return rows, collapse_entries, int(log_probs.shape[1])


def song_has_complete_xlsr(song: dict[str, object]) -> bool:
    state = song.get("XLSR")
    rows = song.get("rows")
    if not isinstance(state, dict) or state.get("status") != "OK" or not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        sources = row.get("sources")
        measured = sources.get("XLSR") if isinstance(sources, dict) else None
        if not isinstance(measured, dict) or not isinstance(measured.get("time"), (int, float)):
            return False
    return True


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
    revision, config_sha, weight_sha = local_model_identity()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    existing_ok = sum(song_has_complete_xlsr(song) for song in songs)
    payload["sources"]["XLSR"] = {
        "status": "RUNNING",
        "mode": "whole-song shared S03 vocal stem",
        "model_id": jactc.MODEL_NAME,
        "model_revision": revision,
        "config_sha256": config_sha,
        "model_blob_sha256": weight_sha,
        "helper_sha256": sha256_file(SCRIPTS / "japanese_ctc_align.py"),
        "device": device,
        "sample_rate": jactc.SAMPLE_RATE,
        "selected_song_ids": [str(song["id"]) for song in songs],
        "songs_ok": existing_ok,
        "songs_failed": 0,
    }
    save(payload, args.data)

    processor = AutoProcessor.from_pretrained(jactc.MODEL_NAME, revision=revision, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(
        jactc.MODEL_NAME, revision=revision, local_files_only=True
    ).to(device).eval()

    songs_ok = existing_ok
    songs_failed = 0
    failures: list[dict[str, object]] = []
    total = len(songs)
    for song_index, song in enumerate(songs, start=1):
        if song_has_complete_xlsr(song):
            print(f"[{song_index}/{total}] XLSR SKIP", flush=True)
            continue
        rows = song["rows"]
        lines = [str(row["text"]) for row in rows]
        try:
            aligned_rows, collapse_entries, emission_frames = align_song(
                processor, model, device, Path(song["vocal_audio_path"]), lines
            )
            if len(aligned_rows) != len(rows):
                raise RuntimeError("XLSR row-count mismatch")
            collapse_set = set(collapse_entries)
            for target, measured in zip(rows, aligned_rows):
                target["sources"]["XLSR"] = {
                    "time": measured["time"],
                    "trusted": int(measured["entry"]) not in collapse_set,
                    "score": measured["score"],
                    "log_score": measured["log_score"],
                    "collapse": int(measured["entry"]) in collapse_set,
                }
            song["XLSR"] = {
                "status": "OK",
                "collapse_entries": collapse_entries,
                "emission_frames": emission_frames,
            }
            songs_ok += 1
            print(f"[{song_index}/{total}] XLSR OK rows={len(rows)} collapse={len(collapse_entries)}", flush=True)
        except Exception as exc:
            songs_failed += 1
            failures.append({
                "id": song["id"], "title": song["title"],
                "error_class": type(exc).__name__, "error": str(exc)[:2000],
            })
            for target in rows:
                target["sources"]["XLSR"] = {
                    "time": None, "trusted": False, "score": None,
                    "error": f"{type(exc).__name__}:{exc}"[:1000],
                }
            song["XLSR"] = {"status": "FAILED", "error": f"{type(exc).__name__}:{exc}"[:2000]}
            print(f"[{song_index}/{total}] XLSR FAILED class={type(exc).__name__}", flush=True)
        payload["sources"]["XLSR"]["songs_ok"] = songs_ok
        payload["sources"]["XLSR"]["songs_failed"] = songs_failed
        save(payload, args.data)

    payload["sources"]["XLSR"]["status"] = "READY" if songs_ok else "UNAVAILABLE"
    payload["sources"]["XLSR"]["failures"] = failures
    save(payload, args.data)
    print(json.dumps({"XLSR_ok": songs_ok, "XLSR_failed": songs_failed}, ensure_ascii=False))
    return 0 if songs_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
