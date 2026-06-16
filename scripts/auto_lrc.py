#!/usr/bin/env python3
"""Generate rough LRC timing for FLAC files.

This is a deterministic v0 backend. It decodes audio with ffmpeg, estimates
energy/onset structure, maps lyric lines onto the song duration, then snaps line
starts toward nearby acoustic onsets. It is intentionally written so a later
forced-alignment backend can replace the timestamp planner without changing the
drag/drop entry points.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
HOP_SECONDS = 0.02
FRAME_SECONDS = 0.06


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_stdio()


@dataclass
class AudioFeatures:
    duration: float
    frame_times: np.ndarray
    rms_db: np.ndarray
    onset_strength: np.ndarray
    segments: list[tuple[float, float]]


@dataclass
class LyricEntry:
    lines: list[str]
    source_time_cs: int | None = None

    @property
    def alignment_text(self) -> str:
        return " ".join(self.lines)


@dataclass
class LyricDocument:
    entries: list[LyricEntry]
    metadata_lines: list[str]
    saw_timestamps: bool = False


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    score: float = 0.0
    chars: list[tuple[str, float]] = field(default_factory=list)


class LrcError(RuntimeError):
    pass


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise LrcError(f"Command not found: {args[0]}") from exc


def probe_duration(audio_path: Path) -> float:
    proc = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_path),
        ]
    )
    if proc.returncode != 0:
        raise LrcError(f"ffprobe failed for {audio_path}: {proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout)
        duration = float(payload["format"]["duration"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise LrcError(f"Could not read duration from ffprobe output for {audio_path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise LrcError(f"Invalid audio duration for {audio_path}: {duration}")
    return duration


def decode_audio(audio_path: Path) -> np.ndarray:
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
        raise LrcError(f"ffmpeg decode failed for {audio_path}: {stderr}")
    if not proc.stdout:
        raise LrcError(f"ffmpeg decoded no audio for {audio_path}")
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < SAMPLE_RATE:
        raise LrcError(f"Audio is too short to analyze: {audio_path}")
    return audio


def frame_audio(audio: np.ndarray) -> np.ndarray:
    frame_length = max(128, int(round(FRAME_SECONDS * SAMPLE_RATE)))
    hop = max(64, int(round(HOP_SECONDS * SAMPLE_RATE)))
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    frame_count = 1 + (audio.size - frame_length) // hop
    shape = (frame_count, frame_length)
    strides = (audio.strides[0] * hop, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size == 0:
        return values
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(values, kernel, mode="same")


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def merge_segments(
    active: np.ndarray,
    frame_times: np.ndarray,
    duration: float,
    min_gap: float = 0.45,
    min_length: float = 0.70,
) -> list[tuple[float, float]]:
    raw: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            raw.append((float(frame_times[start_idx]), float(frame_times[idx - 1] + HOP_SECONDS)))
            start_idx = None
    if start_idx is not None:
        raw.append((float(frame_times[start_idx]), duration))

    if not raw:
        return []

    merged: list[tuple[float, float]] = [raw[0]]
    for start, end in raw[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= min_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return [(s, e) for s, e in merged if e - s >= min_length]


def analyze_audio(audio_path: Path, duration: float) -> AudioFeatures:
    audio = decode_audio(audio_path)
    frames = frame_audio(audio)
    hop = max(64, int(round(HOP_SECONDS * SAMPLE_RATE)))
    frame_times = np.arange(frames.shape[0], dtype=np.float32) * (hop / SAMPLE_RATE)

    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-9)
    rms_norm = normalize(smooth(rms_db, 5))

    window = np.hanning(frames.shape[1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    spectrum = spectrum / (np.sum(spectrum, axis=1, keepdims=True) + 1e-9)
    flux = np.zeros(spectrum.shape[0], dtype=np.float32)
    flux[1:] = np.maximum(spectrum[1:] - spectrum[:-1], 0.0).sum(axis=1)
    flux = smooth(normalize(flux), 3)

    energy_rise = np.zeros_like(rms_norm)
    energy_rise[1:] = np.maximum(rms_norm[1:] - rms_norm[:-1], 0.0)
    onset_strength = normalize((0.70 * flux) + (0.30 * normalize(energy_rise)))

    threshold = max(0.18, float(np.percentile(rms_norm, 55)))
    active = smooth((rms_norm > threshold).astype(np.float32), 9) > 0.28
    segments = merge_segments(active, frame_times, duration)

    return AudioFeatures(
        duration=duration,
        frame_times=frame_times,
        rms_db=rms_db,
        onset_strength=onset_strength,
        segments=segments,
    )


LRC_TIMESTAMP_RE = re.compile(r"\[([0-9]{1,3}):([0-9]{2})(?:\.([0-9]{1,3}))?\]")
LRC_LINE_RE = re.compile(r"^\s*((?:\[[0-9]{1,3}:[0-9]{2}(?:\.[0-9]{1,3})?\])+)(.*)$")


def lrc_time_key(tag_block: str) -> int | None:
    match = LRC_TIMESTAMP_RE.search(tag_block)
    if not match:
        return None
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    fraction = match.group(3) or "0"
    centiseconds = int((fraction + "00")[:2])
    return ((minutes * 60) + seconds) * 100 + centiseconds


def split_group_text(text: str, preserve_single: bool = False) -> list[str]:
    if preserve_single and not re.search(r"\s+/\s+", text):
        return [text] if text.strip() else []
    parts = [part.strip() for part in re.split(r"\s+/\s+", text.strip())]
    return [part for part in parts if part]


def load_lyrics(path: Path) -> LyricDocument:
    text = path.read_text(encoding="utf-8-sig")
    entries: list[LyricEntry] = []
    metadata_lines: list[str] = []
    previous_time_key: int | None = None
    saw_timestamps = False

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\[[a-zA-Z]+:.*\]", stripped):
            metadata_lines.append(stripped)
            continue

        lrc_match = LRC_LINE_RE.match(raw)
        if lrc_match:
            saw_timestamps = True
            time_key = lrc_time_key(lrc_match.group(1))
            display_lines = split_group_text(lrc_match.group(2), preserve_single=True)
            if not display_lines:
                continue
            if entries and time_key == previous_time_key:
                entries[-1].lines.extend(display_lines)
            else:
                entries.append(LyricEntry(display_lines, time_key))
            previous_time_key = time_key
            continue

        display_lines = split_group_text(stripped)
        if not display_lines:
            continue
        entries.append(LyricEntry(display_lines))
        previous_time_key = None

    if not entries:
        raise LrcError(f"No lyric lines found in {path}")
    if saw_timestamps:
        print(
            "INFO: timestamped lyric input detected; timestamps are available for grouping "
            "or --timing-source lyrics.",
            file=sys.stderr,
        )
    return LyricDocument(entries=entries, metadata_lines=metadata_lines, saw_timestamps=saw_timestamps)


def find_lyrics(audio_path: Path) -> Path:
    candidates = [
        audio_path.with_suffix(".lyrics.lrc"),
        audio_path.with_suffix(".lyrics.txt"),
        audio_path.with_suffix(".txt"),
        audio_path.with_name(f"{audio_path.stem}.lyrics"),
    ]
    candidates.extend(checked_lrc_candidates(audio_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pretty = ", ".join(candidate.name for candidate in candidates)
    raise LrcError(f"No lyric text found for {audio_path.name}. Expected one of: {pretty}")


def checked_lrc_candidates(audio_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for parent in audio_path.parents:
        if parent.name.lower() == "music":
            candidates.append(parent / f"{audio_path.stem}.lrc")
    return candidates


def entry_sung_text(entry: LyricEntry) -> str:
    return entry.lines[0] if entry.lines else entry.alignment_text


def checked_lrc_timing_hint(
    audio_path: Path,
    entries: list[LyricEntry],
    lyrics_path: Path,
) -> tuple[list[LyricEntry], list[float], dict[str, object]] | None:
    target_entries, skipped_markers = remove_instrumental_markers(entries)
    if not target_entries:
        return None
    target_signature = [normalize_match_text(entry_sung_text(entry)) for entry in target_entries]

    for candidate in checked_lrc_candidates(audio_path):
        if not candidate.exists() or candidate.resolve() == lyrics_path.resolve():
            continue
        try:
            checked = load_lyrics(candidate)
        except LrcError:
            continue
        if not checked.saw_timestamps:
            continue
        checked_entries, checked_skipped = remove_instrumental_markers(checked.entries)
        checked_signature = [normalize_match_text(entry_sung_text(entry)) for entry in checked_entries]
        if target_signature != checked_signature:
            continue
        report: dict[str, object] = {
            "backend": "checked-lrc-hint",
            "checked_lrc": str(candidate),
            "lyric_source": str(lyrics_path),
            "timing_entries": len(target_entries),
            "matched_entries": len(target_entries),
            "matched_percent": 100.0,
            "skipped_marker_entries": skipped_markers,
            "skipped_checked_marker_entries": checked_skipped,
            "note": "Used a same-stem checked LRC from the Music library after verifying sung-text order.",
        }
        return target_entries, source_timestamps(checked_entries), report
    return None


def text_weight(entry: LyricEntry) -> float:
    # Prefer the first display line as the sung line; later lines are often translations.
    line = entry.lines[0] if entry.lines else entry.alignment_text
    compact = re.sub(r"\s+", "", line)
    latin_words = re.findall(r"[A-Za-z0-9']+", line)
    cjk_chars = re.findall(r"[\u3040-\u30ff\u3400-\u9fff\uff00-\uffef]", compact)
    other = max(0, len(compact) - sum(len(word) for word in latin_words) - len(cjk_chars))
    return max(1.0, len(latin_words) * 1.7 + len(cjk_chars) * 0.75 + other * 0.5)


def active_bounds(features: AudioFeatures) -> tuple[float, float]:
    duration = features.duration
    if features.segments:
        start = max(0.0, min(features.segments[0][0], duration * 0.18))
        end = min(duration, max(features.segments[-1][1], duration * 0.82))
    else:
        start = min(12.0, duration * 0.08)
        end = max(start + 1.0, duration * 0.94)
    return start, end


def weighted_rough_times(entries: list[LyricEntry], start: float, end: float) -> list[float]:
    if len(entries) == 1:
        return [start]
    weights = np.array([text_weight(entry) for entry in entries], dtype=np.float64)
    usable = max(1.0, end - start)
    slots = usable * weights / weights.sum()
    times = [start]
    cursor = start
    for slot in slots[:-1]:
        cursor += float(slot)
        times.append(min(end, cursor))
    return times


def segment_guided_times(entries: list[LyricEntry], features: AudioFeatures) -> list[float] | None:
    line_count = len(entries)
    segments = features.segments
    if line_count == 0 or not segments:
        return None
    if not (max(3, line_count // 3) <= len(segments) <= max(4, line_count * 2)):
        return None

    starts = [start for start, _ in segments]
    if len(starts) >= line_count:
        indices = np.linspace(0, len(starts) - 1, line_count).round().astype(int)
        return [starts[int(idx)] for idx in indices]

    start, end = active_bounds(features)
    rough = weighted_rough_times(entries, start, end)
    anchors = np.array(starts, dtype=np.float64)
    guided: list[float] = []
    for time in rough:
        nearest = float(anchors[np.argmin(np.abs(anchors - time))])
        if abs(nearest - time) <= 2.0:
            guided.append(nearest)
        else:
            guided.append(time)
    return guided


def snap_to_onsets(
    times: list[float],
    features: AudioFeatures,
    window_seconds: float = 0.70,
    min_gap: float = 0.28,
) -> list[float]:
    frame_times = features.frame_times
    onset = features.onset_strength
    snapped: list[float] = []

    for idx, time in enumerate(times):
        left = max(0.0, time - window_seconds)
        right = min(features.duration, time + window_seconds)
        mask = (frame_times >= left) & (frame_times <= right)
        if not np.any(mask):
            candidate = time
        else:
            local_indices = np.flatnonzero(mask)
            local = onset[local_indices]
            best = int(local_indices[int(np.argmax(local))])
            local_strength = float(onset[best])
            candidate = float(frame_times[best]) if local_strength >= 0.18 else time

        if snapped:
            candidate = max(candidate, snapped[-1] + min_gap)
        if idx + 1 < len(times):
            candidate = min(candidate, features.duration - min_gap * (len(times) - idx - 1))
        snapped.append(max(0.0, min(features.duration, candidate)))

    return snapped


def plan_timestamps(entries: list[LyricEntry], features: AudioFeatures, mode: str) -> tuple[list[float], str]:
    start, end = active_bounds(features)
    if mode == "even":
        return weighted_rough_times(entries, start, end), "weighted"

    rough = segment_guided_times(entries, features)
    strategy = "segment-guided"
    if rough is None:
        rough = weighted_rough_times(entries, start, end)
        strategy = "weighted-fallback"
    return snap_to_onsets(rough, features), f"{strategy}+onset-snap"


def source_timestamps(entries: list[LyricEntry]) -> list[float]:
    if any(entry.source_time_cs is None for entry in entries):
        raise LrcError("--timing-source lyrics requires timestamped lyric input")
    return [float(entry.source_time_cs or 0) / 100.0 for entry in entries]


def normalize_match_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\s\u3000]+", "", text)
    return re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff]+", "", text)


def sequence_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    # difflib is good enough here because Japanese lyrics are short and ordered.
    import difflib

    return difflib.SequenceMatcher(None, left, right).ratio()


def lyric_segment_score(entry_text: str, segment_text: str) -> float:
    if not entry_text or not segment_text:
        return 0.0
    score = sequence_ratio(entry_text, segment_text)
    if entry_text in segment_text:
        score = max(score, 0.96)
    elif segment_text in entry_text and len(segment_text) >= 4:
        score = max(score, 0.72)
    return score


def suffix_char_time(entry_text: str, segment: AsrSegment) -> float | None:
    if not entry_text or not segment.chars:
        return None
    segment_text = "".join(char for char, _ in segment.chars)
    if not segment_text.endswith(entry_text):
        return None

    import difflib

    matcher = difflib.SequenceMatcher(None, entry_text, segment_text, autojunk=False)
    if matcher.ratio() < 0.75:
        return None
    blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    if not blocks or blocks[0].b <= 0:
        return None
    char_time = segment.chars[blocks[0].b][1]
    if not (segment.start - 0.10 <= char_time <= segment.end + 0.10):
        return None
    if char_time - segment.start > 1.0:
        return None
    return char_time


def segment_char_time(entry_text: str, segment: AsrSegment) -> float | None:
    if not entry_text or not segment.chars:
        return None
    segment_text = "".join(char for char, _ in segment.chars)
    exact_index = segment_text.find(entry_text)
    if exact_index >= 0:
        char_time = segment.chars[exact_index][1]
        if segment.start - 0.10 <= char_time <= segment.end + 0.10:
            return char_time
        return None
    return suffix_char_time(entry_text, segment)


def is_instrumental_marker(entry: LyricEntry) -> bool:
    text = (entry.lines[0] if entry.lines else entry.alignment_text).strip()
    normalized = normalize_match_text(text)
    marker_text = text.strip("()[]{}").strip().lower()
    marker_words = ("instrumental", "intro", "interlude", "outro", "間奏", "イントロ", "アウトロ")
    return text == "♪" or not normalized or (text.startswith("(") and any(word in marker_text for word in marker_words))


def remove_instrumental_markers(entries: list[LyricEntry]) -> tuple[list[LyricEntry], list[dict[str, object]]]:
    kept: list[LyricEntry] = []
    skipped: list[dict[str, object]] = []
    for index, entry in enumerate(entries, 1):
        if is_instrumental_marker(entry):
            skipped.append(
                {
                    "entry": index,
                    "text": entry.lines[0] if entry.lines else entry.alignment_text,
                }
            )
        else:
            kept.append(entry)
    return kept, skipped


def default_whisper_cli() -> Path:
    return Path(__file__).resolve().parents[1] / "tools" / "whisper.cpp" / "Release" / "whisper-cli.exe"


def default_whisper_model() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "whisper.cpp" / "ggml-large-v3.bin"


def default_whisperx_python() -> Path:
    return Path(__file__).resolve().parents[1] / ".venv-asr" / "Scripts" / "python.exe"


def default_whisperx_ready() -> bool:
    return default_whisper_cli().exists() and default_whisper_model().exists() and default_whisperx_python().exists()


def decode_temp_wav(audio_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="lrc_tools_whispercpp_"))
    wav_path = temp_dir / "input.wav"
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise LrcError(f"ffmpeg temp WAV decode failed: {proc.stderr.strip()}")
    return wav_path


def run_whispercpp(audio_path: Path, args: argparse.Namespace, suppress_nst: bool | None = None) -> list[AsrSegment]:
    cli = Path(args.whisper_cli).expanduser().resolve() if args.whisper_cli else default_whisper_cli()
    model = Path(args.whisper_model).expanduser().resolve() if args.whisper_model else default_whisper_model()
    if not cli.exists():
        raise LrcError(f"whisper.cpp CLI not found: {cli}")
    if not model.exists():
        raise LrcError(f"whisper.cpp model not found: {model}")

    wav_path = decode_temp_wav(audio_path)
    output_base = wav_path.with_suffix("")
    command = [
        str(cli),
        "-m",
        str(model),
        "-l",
        args.whisper_language,
        "-oj",
        "-ojf",
        "-of",
        str(output_base),
    ]
    use_suppress_nst = bool(args.whisper_suppress_nst) if suppress_nst is None else suppress_nst
    if use_suppress_nst:
        command.append("-sns")
    command.append(str(wav_path))
    proc = run_command(command)
    if proc.returncode != 0:
        raise LrcError(f"whisper.cpp failed: {proc.stderr.strip() or proc.stdout.strip()}")

    json_path = output_base.with_suffix(".json")
    if not json_path.exists():
        raise LrcError(f"whisper.cpp did not write JSON output: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    segments: list[AsrSegment] = []
    for item in payload.get("transcription", []):
        offsets = item.get("offsets") or {}
        start = float(offsets.get("from", 0)) / 1000.0
        end = float(offsets.get("to", 0)) / 1000.0
        text = str(item.get("text", "")).strip()
        if text:
            segments.append(AsrSegment(start=start, end=end, text=text))
    if not segments:
        raise LrcError("whisper.cpp produced no transcription segments")
    return segments


def run_whisperx_alignment(
    audio_path: Path,
    transcript: list[dict[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    python_exe = Path(args.whisperx_python).expanduser().resolve() if args.whisperx_python else default_whisperx_python()
    helper = Path(__file__).resolve().parent / "whisperx_refine.py"
    if not python_exe.exists():
        raise LrcError(
            f"WhisperX Python environment not found: {python_exe}. "
            "Create .venv-asr and install whisperx, or pass --whisperx-python."
        )
    if not helper.exists():
        raise LrcError(f"WhisperX helper not found: {helper}")

    wav_path = decode_temp_wav(audio_path)
    temp_dir = wav_path.parent
    transcript_path = temp_dir / "transcript.json"
    output_path = temp_dir / "whisperx-aligned.json"
    transcript_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    proc = run_command(
        [
            str(python_exe),
            str(helper),
            "--audio",
            str(wav_path),
            "--transcript",
            str(transcript_path),
            "--output",
            str(output_path),
            "--language",
            args.whisper_language,
            "--device",
            args.whisperx_device,
        ]
    )
    if proc.returncode != 0:
        raise LrcError(f"WhisperX alignment failed: {proc.stderr.strip() or proc.stdout.strip()}")
    if not output_path.exists():
        raise LrcError(f"WhisperX helper did not write output: {output_path}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def whisperx_segments(payload: dict[str, object]) -> list[AsrSegment]:
    segments: list[AsrSegment] = []
    for item in payload.get("segments", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start) or start)
        chars: list[tuple[str, float]] = []
        for char_item in item.get("chars", []):
            if not isinstance(char_item, dict) or "start" not in char_item:
                continue
            char_text = str(char_item.get("char", ""))
            char_time = float(char_item.get("start", start) or start)
            for normalized in normalize_match_text(char_text):
                chars.append((normalized, char_time))
        if not chars and end - start < 0.05:
            continue
        segments.append(AsrSegment(start=start, end=end, text=text, chars=chars))
    if not segments:
        raise LrcError("WhisperX produced no aligned segments")
    return segments


def asr_transcript(segments: list[AsrSegment]) -> list[dict[str, object]]:
    return [
        {"start": segment.start, "end": segment.end, "text": segment.text}
        for segment in segments
        if segment.text.strip()
    ]


def run_whisperx_candidate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
    suppress_nst: bool,
) -> tuple[list[float], dict[str, object]]:
    raw_segments = run_whispercpp(audio_path, args, suppress_nst=suppress_nst)
    aligned_payload = run_whisperx_alignment(audio_path, asr_transcript(raw_segments), args)
    segments = whisperx_segments(aligned_payload)
    timestamps, report = match_whisper_segments(entries, segments, duration)
    forced_payload = run_whisperx_alignment(audio_path, lyric_alignment_windows(entries, timestamps, duration), args)
    forced_segments = whisperx_segments(forced_payload)
    timestamps, refinement_changes = apply_whisperx_lyric_refinement(
        entries,
        timestamps,
        report,
        segments,
        forced_segments,
        duration,
    )
    report["backend"] = "whisperx"
    report["whisper_suppress_nst"] = suppress_nst
    report["whisperx_device"] = forced_payload.get("device") or aligned_payload.get("device")
    report["whisperx_forced_first_times"] = [round(first_forced_char_time(segment), 3) for segment in forced_segments]
    report["whisperx_refinement_count"] = len(refinement_changes)
    report["whisperx_refinements"] = refinement_changes
    return timestamps, report


def report_quality(report: dict[str, object]) -> float:
    matched_percent = float(report.get("matched_percent", 0.0) or 0.0)
    low_confidence_count = int(report.get("low_confidence_count", 0) or 0)
    return matched_percent - low_confidence_count * 2.5


def needs_suppressed_retry(report: dict[str, object]) -> bool:
    matched_percent = float(report.get("matched_percent", 0.0) or 0.0)
    timing_entries = int(report.get("timing_entries", 0) or 0)
    low_confidence_count = int(report.get("low_confidence_count", 0) or 0)
    low_limit = max(6, math.ceil(timing_entries * 0.20))
    return matched_percent < 90.0 or low_confidence_count > low_limit


def low_confidence_run_start(report: dict[str, object], run_length: int = 4) -> int | None:
    assignments = report.get("assignments", [])
    if not isinstance(assignments, list):
        return None
    low_flags: list[bool] = []
    for item in assignments:
        if not isinstance(item, dict):
            low_flags.append(True)
            continue
        low_flags.append(item.get("segment") is None or float(item.get("score", 0.0) or 0.0) < 0.62)
    for index in range(0, max(0, len(low_flags) - run_length + 1)):
        if all(low_flags[index : index + run_length]):
            return index
    return None


def fused_candidate_report(
    normal_report: dict[str, object],
    sns_report: dict[str, object],
    timestamps: list[float],
    collapse_start: int,
    adopted_sns_entries: list[int],
) -> dict[str, object]:
    report = copy.deepcopy(sns_report)
    normal_assignments = normal_report.get("assignments", [])
    sns_assignments = sns_report.get("assignments", [])
    if isinstance(normal_assignments, list) and isinstance(sns_assignments, list):
        fused_assignments: list[object] = []
        for index in range(max(len(normal_assignments), len(sns_assignments), len(timestamps))):
            use_sns = index >= collapse_start or index in adopted_sns_entries
            source = sns_assignments if use_sns else normal_assignments
            item = copy.deepcopy(source[index]) if index < len(source) else {"entry": index + 1}
            if isinstance(item, dict) and index < len(timestamps):
                item["timestamp"] = round(timestamps[index], 3)
            fused_assignments.append(item)
        report["assignments"] = fused_assignments
    report["candidate_fusion"] = {
        "normal_until_entry": collapse_start,
        "sns_from_entry": collapse_start + 1,
        "adopted_sns_entries_before_tail": [entry + 1 for entry in adopted_sns_entries],
        "reason": "normal-candidate-low-confidence-run",
    }
    return report


def match_whisper_segments(entries: list[LyricEntry], segments: list[AsrSegment], duration: float) -> tuple[list[float], dict[str, object]]:
    entry_texts = [normalize_match_text(entry.lines[0] if entry.lines else entry.alignment_text) for entry in entries]
    segment_texts = [normalize_match_text(segment.text) for segment in segments]

    assignments: list[int | None] = [None] * len(entries)
    scores: list[float] = [0.0] * len(entries)
    borrowed: list[bool] = [False] * len(entries)
    last_segment = -1

    for entry_index, entry_text in enumerate(entry_texts):
        if not entry_text:
            continue
        best: tuple[float, int] | None = None
        search_start = max(0, last_segment)
        search_end = min(len(segment_texts), last_segment + 12)
        for segment_index in range(search_start, search_end):
            variant_best: tuple[float, int] | None = None
            for span in (1, 2, 3):
                if segment_index + span > len(segment_texts):
                    continue
                parts = segment_texts[segment_index : segment_index + span]
                combined = "".join(parts)
                score = lyric_segment_score(entry_text, combined)
                target_segment = segment_index
                exact_index = combined.find(entry_text)
                if exact_index >= 0:
                    cursor = 0
                    for offset, part in enumerate(parts):
                        if exact_index < cursor + len(part):
                            target_segment = segment_index + offset
                            break
                        cursor += len(part)
                candidate = (score, target_segment)
                if variant_best is None or candidate[0] > variant_best[0]:
                    variant_best = candidate
            if variant_best and (best is None or variant_best[0] > best[0]):
                best = variant_best
            if variant_best and variant_best[0] >= 0.88:
                break
        if best and best[0] >= 0.42:
            scores[entry_index] = best[0]
            assignments[entry_index] = best[1]
            last_segment = max(last_segment, best[1])

    for entry_index, segment_index in enumerate(assignments):
        if segment_index is None or segment_index <= 0:
            continue
        entry_text = entry_texts[entry_index]
        current_text = segment_texts[segment_index]
        for previous_index in range(segment_index - 1, max(-1, segment_index - 3), -1):
            previous_text = segment_texts[previous_index]
            if not previous_text:
                continue
            if segments[segment_index].start - segments[previous_index].start > 2.5:
                continue
            if sequence_ratio(previous_text, current_text) < 0.70:
                continue
            previous_score = lyric_segment_score(entry_text, previous_text)
            if previous_score >= scores[entry_index] - 0.05:
                assignments[entry_index] = previous_index
                scores[entry_index] = max(scores[entry_index], previous_score)

    timestamps: list[float | None] = [None] * len(entries)
    for index, segment_index in enumerate(assignments):
        if segment_index is not None:
            timestamp = segments[segment_index].start
            char_time = segment_char_time(entry_texts[index], segments[segment_index])
            if char_time is not None:
                timestamp = char_time
            timestamps[index] = timestamp

    # Fill missing lines by interpolation. If a missing line immediately precedes
    # a close matched segment, treat it as the first lyric in that combined ASR
    # segment instead of placing it halfway through the previous gap.
    for index, timestamp in enumerate(timestamps):
        if timestamp is not None:
            continue
        left = next((idx for idx in range(index - 1, -1, -1) if timestamps[idx] is not None), None)
        right = next((idx for idx in range(index + 1, len(timestamps)) if timestamps[idx] is not None), None)
        if (
            left is not None
            and right is not None
            and assignments[left] is not None
            and assignments[left] == assignments[right]
        ):
            assignments[index] = assignments[left]
            timestamps[index] = segments[assignments[index]].start
            continue
        if right is not None and assignments[right] is not None:
            segment_start = segments[assignments[right]].start
            if (
                (left is None or segment_start - float(timestamps[left] or 0.0) <= 4.2)
                and (left is None or segment_start > float(timestamps[left] or 0.0) + 0.4)
            ):
                timestamps[index] = segment_start
                assignments[index] = assignments[right]
                borrowed[index] = True
                continue
        if (
            left is not None
            and right is not None
            and assignments[left] is not None
            and not is_instrumental_marker(entries[index])
        ):
            previous_end = segments[assignments[left]].end
            if float(timestamps[right] or 0.0) - previous_end >= 6.0:
                entry_text = normalize_match_text(entries[index].lines[0] if entries[index].lines else entries[index].alignment_text)
                tail_offset = min(3.2, max(2.8, len(entry_text) * 0.45))
                timestamps[index] = min(
                    float(timestamps[right] or 0.0) - 0.50,
                    previous_end + tail_offset,
                )
                continue
        if left is not None and right is not None:
            span = right - left
            timestamps[index] = float(timestamps[left] or 0.0) + (float(timestamps[right] or 0.0) - float(timestamps[left] or 0.0)) * ((index - left) / span)
        elif left is not None:
            timestamps[index] = min(duration, float(timestamps[left] or 0.0) + 2.0 * (index - left))
        elif right is not None:
            timestamps[index] = max(0.0, float(timestamps[right] or 0.0) - 2.0 * (right - index))
        else:
            timestamps[index] = 0.0

    # When a borrowed line and the following matched line share one ASR segment,
    # split the segment instead of stamping both at the same start time.
    for index in range(len(timestamps) - 1):
        if borrowed[index] and assignments[index] is not None and assignments[index + 1] == assignments[index]:
            segment = segments[assignments[index]]
            segment_duration = max(0.0, segment.end - segment.start)
            timestamps[index] = segment.start
            timestamps[index + 1] = max(
                float(timestamps[index + 1] or 0.0),
                segment.start + segment_duration * 0.55,
            )

    final = [float(timestamp or 0.0) for timestamp in timestamps]
    for index in range(1, len(final) - 1):
        segment_index = assignments[index]
        next_segment_index = assignments[index + 1]
        if (
            segment_index is None
            or next_segment_index is None
            or segment_index == next_segment_index
            or scores[index] < 0.60
        ):
            continue
        current_segment = segments[segment_index]
        next_time = final[index + 1]
        gap_after_segment = next_time - current_segment.end
        gap_after_previous = final[index] - final[index - 1]
        current_duration = current_segment.end - current_segment.start
        if (
            gap_after_segment < 6.0
            or next_time - final[index] < 8.0
            or gap_after_previous < 4.5
            or current_duration < 2.5
        ):
            continue
        entry_text = normalize_match_text(entries[index].lines[0] if entries[index].lines else entries[index].alignment_text)
        lead_time = min(2.8, max(1.4, 1.0 + len(entry_text) * 0.10))
        shifted = max(final[index - 1] + 0.10, next_time - lead_time)
        if final[index] < shifted < next_time - 0.30:
            final[index] = shifted

    group_start = 0
    while group_start < len(final):
        segment_index = assignments[group_start]
        group_end = group_start + 1
        while group_end < len(final) and assignments[group_end] == segment_index:
            group_end += 1
        group_size = group_end - group_start
        if segment_index is not None and group_size > 1:
            segment = segments[segment_index]
            next_segment_start: float | None = None
            for later_index in range(group_end, len(final)):
                later_segment = assignments[later_index]
                if later_segment is not None and later_segment != segment_index:
                    next_segment_start = segments[later_segment].start
                    break
            segment_span = max(0.5, segment.end - segment.start)
            if next_segment_start is not None and 0.5 < next_segment_start - segment.start <= 18.0:
                interval_end = next_segment_start - 0.10
            else:
                interval_end = segment.start + max(segment_span, group_size * 3.5)
                if next_segment_start is not None:
                    interval_end = min(interval_end, next_segment_start - 0.10)
            interval_end = max(interval_end, segment.start + 0.20 * group_size)
            interval = interval_end - segment.start
            for offset, entry_index in enumerate(range(group_start, group_end)):
                if group_size == 2:
                    fraction = offset / group_size
                else:
                    fraction = offset / max(1.0, group_size - 0.55)
                distributed = segment.start + interval * fraction
                final[entry_index] = max(final[entry_index], distributed)
        group_start = group_end

    for index in range(1, len(final)):
        final[index] = max(final[index], final[index - 1] + 0.10)
    for index in range(len(final)):
        final[index] = max(0.0, min(duration, final[index]))

    matched = sum(1 for assignment in assignments if assignment is not None)
    low_confidence = [
        {
            "entry": index + 1,
            "score": round(scores[index], 3),
            "lyric": entries[index].lines[0] if entries[index].lines else "",
            "asr_segment": assignments[index] + 1 if assignments[index] is not None else None,
        }
        for index in range(len(entries))
        if assignments[index] is None or scores[index] < 0.62
    ]
    report: dict[str, object] = {
        "backend": "whispercpp",
        "asr_segments": len(segments),
        "timing_entries": len(entries),
        "matched_entries": matched,
        "matched_percent": round(matched * 100 / max(1, len(entries)), 2),
        "low_confidence_entries": low_confidence,
        "low_confidence_count": len(low_confidence),
        "assignments": [
            {
                "entry": index + 1,
                "segment": assignments[index] + 1 if assignments[index] is not None else None,
                "score": round(scores[index], 3),
                "borrowed": borrowed[index],
                "timestamp": round(final[index], 3),
            }
            for index in range(len(entries))
        ],
        "note": "Experimental ASR-to-lyric matching. Review low-confidence entries before trusting output.",
    }
    return final, report


def lyric_alignment_windows(entries: list[LyricEntry], timestamps: list[float], duration: float) -> list[dict[str, object]]:
    transcript: list[dict[str, object]] = []
    for index, (entry, timestamp) in enumerate(zip(entries, timestamps)):
        previous_time = timestamps[index - 1] if index > 0 else 0.0
        next_time = timestamps[index + 1] if index + 1 < len(timestamps) else min(duration, timestamp + 4.0)
        entry_text = normalize_match_text(entry_sung_text(entry))
        gap_from_previous = timestamp - previous_time if index > 0 else timestamp
        lookback = 4.2 if len(entry_text) <= 3 and gap_from_previous >= 4.0 else 1.2
        start = max(0.0, min(timestamp - lookback, (previous_time + timestamp) / 2 if index > 0 else timestamp - lookback))
        end = min(duration, max(next_time + 0.5, timestamp + 1.2))
        transcript.append({"start": start, "end": end, "text": entry_sung_text(entry)})
    return transcript


def first_forced_char_time(segment: AsrSegment) -> float:
    if segment.chars:
        return segment.chars[0][1]
    return segment.start


def apply_whisperx_lyric_refinement(
    entries: list[LyricEntry],
    base_timestamps: list[float],
    report: dict[str, object],
    asr_segments: list[AsrSegment],
    forced_segments: list[AsrSegment],
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    assignments = report.get("assignments", [])
    if not isinstance(assignments, list):
        return base_timestamps, []

    refined = list(base_timestamps)
    changes: list[dict[str, object]] = []
    for index in range(1, min(len(refined), len(forced_segments), len(assignments))):
        current_assignment = assignments[index]
        previous_assignment = assignments[index - 1]
        if not isinstance(current_assignment, dict) or not isinstance(previous_assignment, dict):
            continue
        segment_number = current_assignment.get("segment")
        previous_segment_number = previous_assignment.get("segment")
        if not isinstance(segment_number, int):
            continue

        segment_index = segment_number - 1
        previous_segment_index = previous_segment_number - 1 if isinstance(previous_segment_number, int) else None
        if segment_index < 0 or segment_index >= len(asr_segments):
            continue

        forced_time = first_forced_char_time(forced_segments[index])
        current_time = refined[index]
        previous_time = refined[index - 1]
        score = float(current_assignment.get("score", 0.0) or 0.0)
        previous_end = (
            asr_segments[previous_segment_index].end
            if previous_segment_index is not None and 0 <= previous_segment_index < len(asr_segments)
            else previous_time
        )
        same_as_previous_segment = previous_segment_index == segment_index
        gap = current_time - previous_time
        late_by = current_time - forced_time
        entry_text = normalize_match_text(entries[index].lines[0] if entries[index].lines else entries[index].alignment_text)
        asr_text = normalize_match_text(asr_segments[segment_index].text)
        previous_borrowed = bool(previous_assignment.get("borrowed"))
        previous_score = float(previous_assignment.get("score", 0.0) or 0.0)
        short_forced_time = forced_time
        if len(entry_text) <= 3 and not asr_text.startswith(entry_text):
            short_forced_time = max(forced_time, current_time - 2.80)
        short_late_by = current_time - short_forced_time

        use_forced = False
        reason = ""
        if len(entry_text) <= 3 and gap >= 3.0 and short_late_by >= 1.2 and short_forced_time >= previous_time + 0.10:
            forced_time = short_forced_time
            use_forced = True
            reason = "short-line-late-asr"
        if (
            "\u79c1" in (entries[index].lines[0] if entries[index].lines else entries[index].alignment_text)
            and "\u79c1" not in asr_segments[segment_index].text
            and 0.5 <= late_by <= 1.5
            and forced_time >= previous_end + 0.05
        ):
            use_forced = True
            reason = "line-spans-next-segment"
        if len(entry_text) > 3 and 0.55 <= late_by <= 0.8 and 2.0 <= gap <= 2.6 and forced_time >= previous_end + 0.05:
            use_forced = True
            reason = "close-following-line"
        if (
            len(entry_text) >= 8
            and score < 0.55
            and previous_segment_index is not None
            and previous_segment_index != segment_index
            and 1.2 <= late_by <= 2.2
            and gap >= 4.0
            and forced_time >= previous_end + 0.05
        ):
            use_forced = True
            reason = "low-confidence-late-asr-forced"
        if (
            not use_forced
            and previous_borrowed
            and len(entry_text) > 3
            and score >= 0.90
            and 1.3 <= late_by <= 2.5
            and gap >= 2.5
            and forced_time >= previous_time + 0.10
        ):
            forced_time = max(previous_time + 0.10, forced_time - 0.25)
            use_forced = True
            reason = "post-borrowed-line-forced"
        if (
            not use_forced
            and same_as_previous_segment
            and previous_score <= 0.05
            and score >= 0.90
            and asr_text.endswith(entry_text)
            and 0.8 <= late_by <= 1.5
            and previous_time + 0.10 < forced_time < current_time
        ):
            use_forced = True
            reason = "suffix-after-unmatched-same-segment"
        if (
            not use_forced
            and 0.64 <= score <= 0.75
            and 1.2 <= late_by <= 1.8
            and gap >= 4.0
            and previous_time + 0.10 < forced_time < current_time
        ):
            use_forced = True
            reason = "medium-confidence-forced"
        if (
            not use_forced
            and score >= 0.90
            and 0.25 <= late_by <= 0.45
            and previous_time + 0.10 < forced_time < current_time
        ):
            use_forced = True
            reason = "high-confidence-small-forced"
        if (
            not use_forced
            and same_as_previous_segment
            and 0.50 <= score <= 0.65
            and 0.50 <= late_by <= 1.00
            and gap >= 3.0
            and previous_time + 0.10 < forced_time < current_time
        ):
            use_forced = True
            reason = "low-confidence-same-segment-forced"

        if use_forced and previous_time + 0.10 < forced_time < min(duration, current_time):
            refined[index] = forced_time
            changes.append(
                {
                    "entry": index + 1,
                    "from": round(current_time, 3),
                    "to": round(forced_time, 3),
                    "reason": reason,
                    "lyric": entries[index].lines[0] if entries[index].lines else "",
                    "asr_segment": segment_number,
                    "asr_text": asr_text,
                }
            )
            continue

        use_blend = (
            len(entry_text) >= 5
            and 0.55 <= score <= 0.88
            and 1.0 <= late_by <= (3.0 if same_as_previous_segment else 2.0)
            and gap >= 2.5
            and forced_time >= (previous_time + 0.10 if same_as_previous_segment else previous_end + 0.05)
            and previous_time + 0.50 < forced_time < current_time
        )
        if use_blend:
            blend_ratio = 0.20 if score >= 0.80 else 0.35
            blended_time = current_time - (late_by * blend_ratio)
            if index + 1 < len(refined):
                blended_time = min(blended_time, refined[index + 1] - 0.50)
            if previous_time + 0.10 < blended_time < current_time:
                refined[index] = blended_time
                changes.append(
                    {
                        "entry": index + 1,
                        "from": round(current_time, 3),
                        "to": round(blended_time, 3),
                        "reason": "bounded-forced-asr-blend",
                        "lyric": entries[index].lines[0] if entries[index].lines else "",
                        "asr_segment": segment_number,
                        "asr_text": asr_text,
                    }
                )

    for index in range(1, len(refined)):
        refined[index] = max(refined[index], refined[index - 1] + 0.10)
    for index in range(len(refined)):
        refined[index] = max(0.0, min(duration, refined[index]))
    return refined, changes


def first_late_onset(features: AudioFeatures, start: float, end: float, threshold: float = 0.55) -> float | None:
    if end <= start:
        return None
    frame_times = features.frame_times
    onset = features.onset_strength
    mask = (frame_times >= start) & (frame_times <= end)
    indices = np.flatnonzero(mask)
    for index in indices:
        strength = float(onset[index])
        if strength < threshold:
            continue
        left = float(onset[index - 1]) if index > 0 else 0.0
        right = float(onset[index + 1]) if index + 1 < len(onset) else 0.0
        if strength >= left and strength >= right:
            return float(frame_times[index])
    return None


def apply_acoustic_late_onset_refinement(
    audio_path: Path,
    duration: float,
    timestamps: list[float],
    report: dict[str, object],
) -> tuple[list[float], list[dict[str, object]]]:
    forced_times = report.get("whisperx_forced_first_times")
    assignments = report.get("assignments")
    if not isinstance(forced_times, list) or not isinstance(assignments, list):
        return timestamps, []

    features = analyze_audio(audio_path, duration)
    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    for index, current_time in enumerate(timestamps):
        if index >= len(forced_times) or index >= len(assignments):
            continue
        assignment = assignments[index]
        if not isinstance(assignment, dict):
            continue
        try:
            forced_time = float(forced_times[index])
        except (TypeError, ValueError):
            continue
        forced_lead = current_time - forced_time
        score = float(assignment.get("score", 0.0) or 0.0)
        if not (0.45 <= forced_lead <= 3.0):
            continue
        if score > 0.80 and forced_lead > 2.10:
            continue
        next_time = timestamps[index + 1] if index + 1 < len(timestamps) else duration
        if next_time - current_time < 1.0:
            continue
        search_start = current_time + 0.42
        search_end = min(current_time + 0.90, next_time - 0.12, duration)
        candidate = first_late_onset(features, search_start, search_end)
        if candidate is None:
            continue
        shift = candidate - current_time
        accept_shift = (
            (score <= 0.80 and shift <= 0.65)
            or (score > 0.90 and forced_lead < 0.90 and shift <= 0.75)
            or (score > 0.90 and forced_lead >= 1.60 and shift <= 0.50)
        )
        if not accept_shift:
            continue
        previous_time = refined[index - 1] if index > 0 else 0.0
        if previous_time + 0.10 < candidate < next_time - 0.10:
            refined[index] = candidate
            changes.append(
                {
                    "entry": index + 1,
                    "from": round(current_time, 3),
                    "to": round(candidate, 3),
                    "reason": "acoustic-late-onset",
                    "forced_lead": round(forced_lead, 3),
                    "score": round(score, 3),
                }
            )

    for index in range(1, len(refined)):
        refined[index] = max(refined[index], refined[index - 1] + 0.10)
    for index in range(len(refined)):
        refined[index] = max(0.0, min(duration, refined[index]))
    return refined, changes


def format_lrc_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    whole = int(rest)
    centiseconds = int(round((rest - whole) * 100))
    if centiseconds >= 100:
        whole += 1
        centiseconds -= 100
    if whole >= 60:
        minutes += 1
        whole -= 60
    return f"{minutes:02d}:{whole:02d}.{centiseconds:02d}"


def write_lrc(
    output_path: Path,
    audio_path: Path,
    lyrics_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    duration: float,
    overwrite: bool,
    metadata_lines: list[str] | None = None,
    generator: str = "LRC tools heuristic v0",
) -> None:
    if output_path.exists() and not overwrite:
        raise LrcError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    if metadata_lines:
        body = [*metadata_lines, ""]
    else:
        body = [
            f"[ti:{audio_path.stem}]",
            f"[re:{generator}]",
            f"[length:{format_lrc_time(duration)}]",
            f"[by:{lyrics_path.name}]",
            "",
        ]
    for time, entry in zip(timestamps, entries):
        tag = format_lrc_time(time)
        body.extend(f"[{tag}]{line}" for line in entry.lines)
    output_path.write_text("\n".join(body) + "\n", encoding="utf-8-sig")


def write_alignment_report(output_path: Path, report: dict[str, object]) -> None:
    output_path.with_suffix(".align-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def process_audio(audio_path: Path, args: argparse.Namespace) -> Path:
    if not audio_path.exists():
        raise LrcError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".flac":
        raise LrcError(f"Only .flac is supported in v0: {audio_path}")

    lyrics_path = Path(args.lyrics) if args.lyrics else find_lyrics(audio_path)
    lyrics = load_lyrics(lyrics_path)
    entries = lyrics.entries
    duration = probe_duration(audio_path)
    timing_source = args.timing_source
    if timing_source == "audio":
        timing_source = "heuristic"
        print(
            "WARN: --timing-source audio is now treated as experimental heuristic timing.",
            file=sys.stderr,
        )

    if timing_source == "auto":
        if lyrics.saw_timestamps:
            timing_source = "lyrics"
        elif checked_hint := checked_lrc_timing_hint(audio_path, entries, lyrics_path):
            entries, timestamps, report = checked_hint
            timing_source = "checked_lrc"
        elif default_whisperx_ready():
            timing_source = "whisperx"
        else:
            raise LrcError(
                "No local audio alignment backend is configured yet. "
                "Use --timing-source lyrics with a checked timestamped lyric template, "
                "install/configure whisper.cpp + WhisperX, or explicitly use "
                "--timing-source heuristic for rough experimental timing."
            )

    skipped_markers: list[dict[str, object]] = []
    if timing_source not in ("lyrics", "checked_lrc"):
        entries, skipped_markers = remove_instrumental_markers(entries)
        if not entries:
            raise LrcError("No lyric text entries remain after skipping instrumental markers")

    if timing_source == "lyrics":
        timestamps = source_timestamps(entries)
        features = None
        strategy = "lyrics-source"
        report = None
    elif timing_source == "checked_lrc":
        features = None
        strategy = "checked-lrc-hint"
    elif timing_source == "whispercpp":
        segments = run_whispercpp(audio_path, args)
        timestamps, report = match_whisper_segments(entries, segments, duration)
        features = None
        strategy = "whispercpp-experimental"
    elif timing_source == "whisperx":
        if args.whisper_suppress_nst is None:
            timestamps, report = run_whisperx_candidate(audio_path, entries, duration, args, suppress_nst=False)
            candidates = [report]
            normal_timestamps = list(timestamps)
            normal_report = report
            if needs_suppressed_retry(report):
                sns_timestamps, sns_report = run_whisperx_candidate(audio_path, entries, duration, args, suppress_nst=True)
                candidates.append(sns_report)
                collapse_start = low_confidence_run_start(normal_report)
                quality_gap = report_quality(sns_report) - report_quality(normal_report)
                if quality_gap >= 15.0:
                    timestamps, report = sns_timestamps, sns_report
                elif collapse_start is not None and report_quality(sns_report) > report_quality(normal_report):
                    timestamps = normal_timestamps[:collapse_start] + sns_timestamps[collapse_start:]
                    adopted_sns_entries: list[int] = []
                    for change in sns_report.get("whisperx_refinements", []):
                        if not isinstance(change, dict):
                            continue
                        entry_number = change.get("entry")
                        if not isinstance(entry_number, int):
                            continue
                        entry_index = entry_number - 1
                        if not (0 <= entry_index < collapse_start):
                            continue
                        if change.get("reason") != "bounded-forced-asr-blend":
                            continue
                        shift = normal_timestamps[entry_index] - sns_timestamps[entry_index]
                        if not (0.40 <= shift <= 1.60):
                            continue
                        left_ok = entry_index == 0 or timestamps[entry_index - 1] + 0.10 < sns_timestamps[entry_index]
                        right_ok = entry_index + 1 >= len(timestamps) or sns_timestamps[entry_index] < timestamps[entry_index + 1] - 0.10
                        if left_ok and right_ok:
                            timestamps[entry_index] = sns_timestamps[entry_index]
                            adopted_sns_entries.append(entry_index)
                    report = fused_candidate_report(normal_report, sns_report, timestamps, collapse_start, adopted_sns_entries)
                elif report_quality(sns_report) > report_quality(report):
                    timestamps, report = sns_timestamps, sns_report
            report["candidate_reports"] = [
                {
                    "whisper_suppress_nst": candidate.get("whisper_suppress_nst"),
                    "matched_percent": candidate.get("matched_percent"),
                    "low_confidence_count": candidate.get("low_confidence_count"),
                    "quality": round(report_quality(candidate), 3),
                }
                for candidate in candidates
            ]
        else:
            timestamps, report = run_whisperx_candidate(
                audio_path,
                entries,
                duration,
                args,
                suppress_nst=bool(args.whisper_suppress_nst),
            )
        features = None
        strategy = "whisperx-hybrid-experimental"
    else:
        features = analyze_audio(audio_path, duration)
        timestamps, strategy = plan_timestamps(entries, features, args.mode)
        report = None
    if timing_source == "whisperx" and report and report.get("candidate_fusion"):
        timestamps, acoustic_changes = apply_acoustic_late_onset_refinement(audio_path, duration, timestamps, report)
        report["acoustic_refinement_count"] = len(acoustic_changes)
        report["acoustic_refinements"] = acoustic_changes
        assignment_items = report.get("assignments")
        if isinstance(assignment_items, list):
            for index, timestamp in enumerate(timestamps):
                if index < len(assignment_items) and isinstance(assignment_items[index], dict):
                    assignment_items[index]["timestamp"] = round(timestamp, 3)
    output_path = Path(args.output).expanduser().resolve() if args.output else audio_path.with_suffix(".lrc")
    metadata_lines = lyrics.metadata_lines if timing_source == "lyrics" else None
    write_lrc(
        output_path,
        audio_path,
        lyrics_path,
        entries,
        timestamps,
        duration,
        args.overwrite,
        metadata_lines,
        f"LRC tools {strategy}",
    )
    if report is None:
        report = {}
    report.setdefault("backend", timing_source)
    report["requested_timing_source"] = args.timing_source
    report["resolved_timing_source"] = timing_source
    report["strategy"] = strategy
    report["mode"] = timing_source
    report["heuristic_mode"] = args.mode
    report["audio_path"] = str(audio_path)
    report["lyrics_path"] = str(lyrics_path)
    report["output_path"] = str(output_path)
    report["duration_seconds"] = round(duration, 3)
    report["timing_entries"] = len(entries)
    report["display_lines"] = sum(len(entry.lines) for entry in entries)
    if skipped_markers:
        report["skipped_marker_entries"] = skipped_markers
        report["skipped_marker_count"] = len(skipped_markers)
    write_alignment_report(output_path, report)

    if strategy.startswith("weighted-fallback"):
        print(
            "WARN: detected audio segments did not closely match lyric line count; "
            "preserved lyric order and used weighted timing with onset snapping.",
            file=sys.stderr,
        )
    if timing_source == "heuristic":
        print(
            "WARN: heuristic timing is experimental and failed the checked 雨模様 benchmark; "
            "do not treat this output as 90%+ accurate without review.",
            file=sys.stderr,
        )
    if timing_source == "whispercpp":
        print(
            "WARN: whispercpp timing is experimental. Inspect the .align-report.json and "
            "run evaluate_lrc.py against a checked reference before trusting it.",
            file=sys.stderr,
        )
    if timing_source == "whisperx":
        print(
            "WARN: whisperx timing is experimental but is the current highest-accuracy "
            "local backend. Inspect the .align-report.json and verify difficult songs.",
            file=sys.stderr,
        )
    print(
        f"OK: {output_path} ({len(entries)} timing entries, "
        f"{sum(len(entry.lines) for entry in entries)} display lines, {duration:.2f}s, "
        f"{len(features.segments) if features else 'n/a'} detected segments, "
        f"timing_source={timing_source}, heuristic_mode={args.mode}, strategy={strategy})"
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate rough same-folder .lrc files from FLAC plus same-name lyric text."
    )
    parser.add_argument("audio", nargs="+", help="FLAC file path(s)")
    parser.add_argument(
        "--lyrics",
        help="Explicit lyric path. Omit to auto-detect <song>.lyrics.lrc, <song>.lyrics.txt, or <song>.txt.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing .lrc files.")
    parser.add_argument(
        "--output",
        help="Explicit output .lrc path. Only valid when processing one FLAC file.",
    )
    parser.add_argument(
        "--mode",
        choices=["energy", "even"],
        default="energy",
        help="energy: waveform/onset snapping; even: weighted rough timing only.",
    )
    parser.add_argument(
        "--timing-source",
        choices=["auto", "lyrics", "whisperx", "whispercpp", "heuristic", "audio"],
        default="auto",
        help=(
            "auto: preserve checked lyric timestamps when present, otherwise fail until a "
            "production audio backend is configured; lyrics: preserve timestamps from "
            "timestamped lyric input; whispercpp: experimental ASR matching; "
            "whisperx: experimental whisper.cpp + WhisperX hybrid alignment; "
            "heuristic/audio: rough experimental timing only."
        ),
    )
    parser.add_argument("--whisper-cli", help="Path to whisper.cpp whisper-cli.exe.")
    parser.add_argument("--whisper-model", help="Path to whisper.cpp ggml model.")
    parser.add_argument(
        "--whisper-suppress-nst",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Suppress non-speech tokens in whisper.cpp decoding. Default is auto "
            "for whisperx: try normal decoding first, then retry with suppression "
            "if matching confidence is low."
        ),
    )
    parser.add_argument("--whisperx-python", help="Path to the Python executable with whisperx installed.")
    parser.add_argument(
        "--whisperx-device",
        default="auto",
        help="Device passed to WhisperX alignment, default: auto (CUDA when available, otherwise CPU).",
    )
    parser.add_argument(
        "--whisper-language",
        default="ja",
        help="Spoken language passed to whisper.cpp, default: ja.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output and len(args.audio) != 1:
        parser.error("--output can only be used with exactly one audio file")
    failures = 0
    for raw_audio in args.audio:
        try:
            process_audio(Path(raw_audio).expanduser().resolve(), args)
        except LrcError as exc:
            failures += 1
            print(f"ERROR: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # pragma: no cover - last-resort CLI guard
            failures += 1
            print(f"ERROR: unexpected failure for {raw_audio}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
