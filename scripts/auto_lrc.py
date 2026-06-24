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
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from export_alignment_audit import build_rows as build_audit_rows
from export_alignment_audit import write_anchor_template
from export_alignment_audit import write_markdown as write_audit_markdown


SAMPLE_RATE = 16_000
HOP_SECONDS = 0.02
FRAME_SECONDS = 0.06
TRUSTED_ALIGNMENT_SCORE = 0.62
COLLAPSE_SEGMENT_SCORE = 0.30
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
DEFAULT_PROBE_DIR = PROJECT_ROOT / "outputs" / "probes"
DEFAULT_VOCAL_CACHE_DIR = PROJECT_ROOT / "outputs" / "cache" / "vocals"


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
class AnchorHint:
    entry_number: int | None
    entry: LyricEntry


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    score: float = 0.0
    chars: list[tuple[str, float]] = field(default_factory=list)


class LrcError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def optional_kakasi_converter() -> object | None:
    try:
        import pykakasi  # type: ignore[import-not-found]
    except Exception:
        return None
    return pykakasi.kakasi()


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


def audio_track_labels(audio_path: Path) -> tuple[str, str]:
    """Read display-safe artist/title labels from FLAC metadata when present."""
    proc = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=TITLE,ARTIST",
            "-of",
            "json",
            str(audio_path),
        ]
    )
    if proc.returncode != 0:
        return "", audio_path.stem
    try:
        tags = json.loads(proc.stdout).get("format", {}).get("tags", {})
    except (AttributeError, json.JSONDecodeError):
        tags = {}
    if not isinstance(tags, dict):
        tags = {}
    artist = str(tags.get("ARTIST") or tags.get("artist") or "").strip()
    title = str(tags.get("TITLE") or tags.get("title") or audio_path.stem).strip()
    if artist and artist.isascii():
        artist = artist.title()
    return artist, title or audio_path.stem


def remove_generated_title_cards(entries: list[LyricEntry], audio_path: Path) -> list[LyricEntry]:
    """Do not feed an LRC tools 00:00 title card back into forced alignment."""
    artist, title = audio_track_labels(audio_path)
    normalized_title = normalize_match_text(title)
    kept: list[LyricEntry] = []
    for entry in entries:
        text = entry_sung_text(entry).strip()
        is_title_card = (
            entry.source_time_cs == 0
            and len(entry.lines) == 1
            and "-" in text
            and normalized_title
            and normalized_title in normalize_match_text(text)
        )
        if not is_title_card:
            kept.append(entry)
    return kept


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


def analyze_vocal_onsets(audio_path: Path, duration: float) -> AudioFeatures:
    """Measure consonant-sensitive onsets from an isolated vocal stem."""
    audio = decode_audio(audio_path)
    frames = frame_audio(audio)
    hop = max(64, int(round(HOP_SECONDS * SAMPLE_RATE)))
    frame_times = np.arange(frames.shape[0], dtype=np.float32) * (hop / SAMPLE_RATE)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-9)

    window = np.hanning(frames.shape[1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32) + 1e-9
    frequencies = np.fft.rfftfreq(frames.shape[1], 1.0 / SAMPLE_RATE)
    consonant_band = (frequencies >= 1200.0) & (frequencies <= 6500.0)
    log_band = np.log(spectrum[:, consonant_band])
    flux = np.zeros(frames.shape[0], dtype=np.float32)
    flux[1:] = np.maximum(log_band[1:] - log_band[:-1], 0.0).mean(axis=1)
    energy_rise = np.zeros_like(rms)
    energy_rise[1:] = np.maximum(rms[1:] - rms[:-1], 0.0)
    onset_strength = normalize((0.75 * normalize(flux)) + (0.25 * normalize(energy_rise)))
    return AudioFeatures(duration, frame_times, rms_db, onset_strength, [])


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
    # Source lyrics commonly place a translation after two spaces.  A single
    # space remains part of the lyric, so Japanese and English lines stay intact.
    separator = r"\s+/\s+|(?:[ \u3000]{2,}|\t+)"
    if preserve_single and not re.search(separator, text):
        return [text] if text.strip() else []
    parts = [part.strip() for part in re.split(separator, text.strip())]
    return [part for part in parts if part]


def lyric_line_script(text: str) -> str:
    if re.search(r"[\u3040-\u30ff]", text):
        return "japanese"
    if len(re.findall(r"[A-Za-z]", text)) >= 2:
        return "latin"
    if re.search(r"[\u3400-\u9fff]", text):
        return "han"
    return "other"


def merge_adjacent_translation_lines(entries: list[LyricEntry]) -> list[LyricEntry]:
    """Treat adjacent source/translation lines as one sung lyric entry."""
    merged: list[LyricEntry] = []
    index = 0
    while index < len(entries):
        current = entries[index]
        following = entries[index + 1] if index + 1 < len(entries) else None
        if (
            following is not None
            and len(current.lines) == 1
            and len(following.lines) == 1
            and current.source_time_cs is None
            and following.source_time_cs is None
        ):
            pair = {lyric_line_script(current.lines[0]), lyric_line_script(following.lines[0])}
            if pair in ({"japanese", "han"}, {"latin", "han"}):
                merged.append(LyricEntry([current.lines[0], following.lines[0]]))
                index += 2
                continue
        merged.append(current)
        index += 1
    return merged


def load_lyrics(path: Path, skip_comments: bool = False) -> LyricDocument:
    text = path.read_text(encoding="utf-8-sig")
    entries: list[LyricEntry] = []
    metadata_lines: list[str] = []
    previous_time_key: int | None = None
    saw_timestamps = False

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if skip_comments and stripped.startswith("#"):
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

    if not saw_timestamps:
        entries = merge_adjacent_translation_lines(entries)
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


def anchor_hint_candidates(audio_path: Path) -> list[Path]:
    return [
        audio_path.with_suffix(".anchors.lrc"),
        audio_path.with_suffix(".anchors.txt"),
        audio_path.with_suffix(".anchor.lrc"),
        audio_path.with_suffix(".anchor.txt"),
    ]


def find_anchor_hints(audio_path: Path, args: argparse.Namespace) -> Path | None:
    explicit = getattr(args, "anchor_hints", None)
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise LrcError(f"Anchor hints not found: {path}")
        if ".anchor-template." in path.name.lower():
            raise LrcError(
                f"Refusing to use unreviewed anchor template as confirmed anchor hints: {path}. "
                "Review it manually, then copy checked rows to a .anchors.lrc file."
            )
        return path
    if getattr(args, "no_anchor_hints", False):
        return None
    for candidate in anchor_hint_candidates(audio_path):
        if candidate.exists():
            return candidate
    return None


ANCHOR_ENTRY_COMMENT_RE = re.compile(r"^#\s*entry\s*=\s*(\d+)\s*$", re.IGNORECASE)


def load_anchor_hints(path: Path) -> tuple[list[AnchorHint], list[dict[str, object]]]:
    text = path.read_text(encoding="utf-8-sig")
    hints: list[AnchorHint] = []
    skipped_markers: list[dict[str, object]] = []
    pending_entry_number: int | None = None
    previous_time_key: int | None = None

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        entry_comment = ANCHOR_ENTRY_COMMENT_RE.match(stripped)
        if entry_comment:
            pending_entry_number = int(entry_comment.group(1))
            continue
        if stripped.startswith("#"):
            continue
        if re.fullmatch(r"\[[a-zA-Z]+:.*\]", stripped):
            continue
        lrc_match = LRC_LINE_RE.match(raw)
        if not lrc_match:
            raise LrcError(f"Anchor hint line must be timestamped LRC syntax: {raw!r}")
        time_key = lrc_time_key(lrc_match.group(1))
        display_lines = split_group_text(lrc_match.group(2), preserve_single=True)
        if not display_lines:
            continue
        entry = LyricEntry(display_lines, time_key)
        if is_instrumental_marker(entry):
            skipped_markers.append({"entry": len(hints) + 1, "text": entry_sung_text(entry)})
            pending_entry_number = None
            previous_time_key = time_key
            continue
        if hints and time_key == previous_time_key:
            hints[-1].entry.lines.extend(display_lines)
        else:
            hints.append(AnchorHint(pending_entry_number, entry))
        pending_entry_number = None
        previous_time_key = time_key

    if not hints:
        raise LrcError(f"No timestamped anchor hints found in {path}")
    return hints, skipped_markers


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
            "assigned_entries": len(target_entries),
            "assigned_percent": 100.0,
            "trusted_entries": len(target_entries),
            "trusted_percent": 100.0,
            "matched_entries": len(target_entries),
            "matched_percent": 100.0,
            "low_confidence_count": 0,
            "low_confidence_percent": 0.0,
            "review_required_count": 0,
            "review_required_percent": 0.0,
            "review_required": False,
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


def ratio_percent(part: int, total: int) -> float:
    return round(part * 100 / max(1, total), 2)


def compact_latin(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", text).lower())


def japanese_romaji(text: str) -> str | None:
    converter = optional_kakasi_converter()
    if converter is None:
        return None
    try:
        converted = converter.convert(text)  # type: ignore[attr-defined]
    except Exception:
        return None
    romaji = "".join(str(item.get("hepburn", "")) for item in converted if isinstance(item, dict))
    romaji = compact_latin(romaji)
    return romaji or None


def romaji_morae(romaji: str) -> list[str]:
    morae: list[str] = []
    cursor = 0
    while cursor < len(romaji):
        if romaji[cursor] == "n" and (cursor + 1 == len(romaji) or romaji[cursor + 1] not in "aeiouy"):
            morae.append("n")
            cursor += 1
            continue
        match = re.match(r"(?:[bcdfghjklmnpqrstvwxyz]*)(?:a|i|u|e|o)", romaji[cursor:])
        if match:
            morae.append(match.group(0))
            cursor += len(match.group(0))
        else:
            morae.append(romaji[cursor])
            cursor += 1
    return morae


def lyric_anchor_profile(entry: LyricEntry) -> dict[str, object]:
    text = entry_sung_text(entry)
    normalized = normalize_match_text(text)
    romaji = japanese_romaji(text)
    profile: dict[str, object] = {
        "entry_text": text,
        "normalized_prefix": normalized[:8],
        "normalized_suffix": normalized[-8:] if normalized else "",
    }
    if romaji:
        morae = romaji_morae(romaji)
        profile.update(
            {
                "phonetic_system": "ja-hepburn-pykakasi",
                "romaji": romaji,
                "romaji_prefix": romaji[:12],
                "romaji_suffix": romaji[-12:],
                "first_mora": morae[0] if morae else "",
                "last_mora": morae[-1] if morae else "",
            }
        )
    else:
        latin = compact_latin(text)
        if latin:
            profile.update(
                {
                    "phonetic_system": "latin-fallback",
                    "romaji": latin,
                    "romaji_prefix": latin[:12],
                    "romaji_suffix": latin[-12:],
                }
            )
        else:
            profile["phonetic_system"] = "unavailable"
    return profile


def repeated_leading_term(text: str) -> tuple[str, int]:
    """Return a visibly repeated opening term, without guessing word breaks."""
    terms = [
        normalize_match_text(term)
        for term in re.split(r"[\s\u3000,、，。！？!?・]+", unicodedata.normalize("NFKC", text).strip())
    ]
    terms = [term for term in terms if term]
    if len(terms) < 2 or len(terms[0]) < 2:
        return "", 0
    first = terms[0]
    repeated = 1
    for term in terms[1:]:
        if term.startswith(first):
            repeated += 1
        else:
            break
    return (first, repeated) if repeated >= 2 else ("", 0)


def infer_spoken_language(entries: list[LyricEntry]) -> str:
    sample = "".join(entry_sung_text(entry) for entry in entries[:12])
    if not sample:
        return "ja"
    latin_count = sum(1 for ch in sample if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    kana_count = sum(1 for ch in sample if "\u3040" <= ch <= "\u30ff")
    cjk_count = sum(1 for ch in sample if "\u3400" <= ch <= "\u9fff")
    total_text = max(1, latin_count + kana_count + cjk_count)
    if latin_count / total_text >= 0.55:
        return "en"
    if kana_count > 0:
        return "ja"
    if cjk_count > 0:
        return "zh"
    return "ja"


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


def estimated_segment_text_time(entry_text: str, raw_entry_text: str, segment: AsrSegment) -> float | None:
    if not entry_text:
        return None
    segment_text = normalize_match_text(segment.text)
    if not segment_text:
        return None

    match_index = segment_text.find(entry_text)
    match_basis_length = len(segment_text)
    if match_index < 0:
        raw_entry_romaji = japanese_romaji(raw_entry_text) or compact_latin(raw_entry_text)
        segment_romaji = japanese_romaji(segment.text) or compact_latin(segment.text)
        if not raw_entry_romaji or not segment_romaji:
            return None
        for prefix_length in range(min(len(raw_entry_romaji), 18), 4, -1):
            prefix = raw_entry_romaji[:prefix_length]
            match_index = segment_romaji.find(prefix)
            if match_index >= 0:
                match_basis_length = len(segment_romaji)
                break
        else:
            return None

    if match_index <= 0:
        return segment.start
    segment_duration = max(0.0, segment.end - segment.start)
    if segment_duration <= 0.05 or match_basis_length <= 0:
        return None
    fraction = min(0.95, max(0.0, match_index / match_basis_length))
    estimated = segment.start + segment_duration * fraction
    if segment.start - 0.10 <= estimated <= segment.end + 0.10:
        return estimated
    return None


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


def default_ctc_python() -> Path:
    return default_whisperx_python()


def default_whisperx_ready() -> bool:
    return default_whisper_cli().exists() and default_whisper_model().exists() and default_whisperx_python().exists()


def default_ctc_ready() -> bool:
    return default_ctc_python().exists() and (Path(__file__).resolve().parent / "ctc_align.py").exists()


def default_japanese_ctc_ready() -> bool:
    return default_ctc_python().exists() and (Path(__file__).resolve().parent / "japanese_ctc_align.py").exists()


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


def run_ctc_alignment(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object]]:
    alignment_audio, audio_source, audio_note = prepare_vocal_ctc_audio(audio_path, args)
    python_exe = default_ctc_python()
    helper = Path(__file__).resolve().parent / "ctc_align.py"
    if not python_exe.exists():
        raise LrcError(f"CTC Python environment not found: {python_exe}")
    if not helper.exists():
        raise LrcError(f"CTC helper not found: {helper}")

    with tempfile.TemporaryDirectory(prefix="lrc-ctc-") as temp_name:
        temp_dir = Path(temp_name)
        transcript_path = temp_dir / "transcript.json"
        output_path = temp_dir / "ctc-aligned.json"
        transcript_path.write_text(
            json.dumps([entry_sung_text(entry) for entry in entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        proc = run_command(
            [
                str(python_exe),
                str(helper),
                "--audio",
                str(alignment_audio),
                "--transcript",
                str(transcript_path),
                "--output",
                str(output_path),
                "--device",
                args.whisperx_device,
            ]
        )
        if proc.returncode != 0:
            raise LrcError(f"CTC forced alignment failed: {proc.stderr.strip() or proc.stdout.strip()}")
        if not output_path.exists():
            raise LrcError(f"CTC helper did not write output: {output_path}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))

    rows = payload.get("entries")
    if not isinstance(rows, list) or len(rows) != len(entries):
        raise LrcError("CTC helper returned an invalid entry list")
    timestamps: list[float] = []
    assignments: list[dict[str, object]] = []
    missing: list[int] = []
    ctc_scores: list[float] = []
    low_score_entries: list[dict[str, object]] = []
    very_low_score_entries: list[dict[str, object]] = []
    low_score_entry_numbers: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            missing.append(index + 1)
            timestamps.append(timestamps[-1] + 0.10 if timestamps else 0.0)
            continue
        start = row.get("start")
        if start is None:
            missing.append(index + 1)
            timestamp = timestamps[-1] + 0.10 if timestamps else 0.0
        else:
            timestamp = float(start)
        timestamps.append(max(0.0, min(duration, timestamp)))
        raw_ctc_score = row.get("ctc_score")
        try:
            ctc_score = float(raw_ctc_score)
        except (TypeError, ValueError):
            ctc_score = 0.0
        if start is not None:
            ctc_scores.append(ctc_score)
            if ctc_score < 0.03:
                low_score_entry_numbers.add(index + 1)
                low_score_entries.append(
                    {
                        "entry": index + 1,
                        "text": entry_sung_text(entries[index]),
                        "ctc_score": round(ctc_score, 6),
                    }
                )
            if ctc_score < 0.01:
                very_low_score_entries.append(
                    {
                        "entry": index + 1,
                        "text": entry_sung_text(entries[index]),
                        "ctc_score": round(ctc_score, 6),
                    }
                )
        # A forced path always has a position, but a weak path is only a draft
        # candidate. Do not let its existence become a fake "trusted" result.
        confidence = 0.90 if ctc_score >= 0.08 else 0.70 if ctc_score >= 0.03 else 0.35
        assignments.append(
            {
                "entry": index + 1,
                "segment": None,
                "score": confidence if start is not None else 0.0,
                "ctc_score": round(ctc_score, 6),
                "romaji": row.get("romaji"),
                "ctc_token_spans": row.get("token_spans"),
                "ctc_first_token_candidates": row.get("first_token_candidates"),
                "timestamp": round(timestamps[-1], 3),
                "timing_repair": "ctc-forced-align",
                "timing_repair_source": "torchaudio-mms-fa",
            }
        )

    for index in range(1, len(timestamps)):
        timestamps[index] = max(timestamps[index], timestamps[index - 1] + 0.10)
        assignments[index]["timestamp"] = round(timestamps[index], 3)
    annotate_ctc_boundary_evidence(assignments)

    low_score_runs: list[list[int]] = []
    run: list[int] = []
    for entry_number in range(1, len(entries) + 1):
        if entry_number in low_score_entry_numbers:
            run.append(entry_number)
        else:
            if len(run) >= 4:
                low_score_runs.append(run)
            run = []
    if len(run) >= 4:
        low_score_runs.append(run)

    collapsed_entries = {entry_number for run in low_score_runs for entry_number in run}
    # A weak individual CTC path is still recorded as low confidence, but it is
    # not evidence that its timestamp is wrong by itself. Escalate only missing
    # alignments or a consecutive low-score collapse.
    ctc_review_entries = set(missing) | collapsed_entries
    suspicious_alignments: list[dict[str, object]] = []
    for entry_number in sorted(ctc_review_entries):
        entry_index = entry_number - 1
        is_missing = entry_number in missing
        is_collapsed = entry_number in collapsed_entries
        flags: list[str] = []
        if is_missing:
            flags.append("ctc_missing_alignment")
        if entry_number in low_score_entry_numbers:
            flags.append("ctc_low_path_confidence")
        if is_collapsed:
            flags.append("ctc_confidence_collapse")
        suspicious_alignments.append(
            {
                "entry": entry_number,
                "text": entry_sung_text(entries[entry_index]),
                "flags": flags,
                "severity": "high" if is_missing or is_collapsed else "medium",
                "review_required": True,
            }
        )

    report: dict[str, object] = {
        "backend": "ctc",
        "strategy": "ctc-forced-align",
        "ctc_backend": payload.get("backend", "mms_ctc"),
        "ctc_device": payload.get("device"),
        "ctc_sample_rate": payload.get("sample_rate"),
        "ctc_emission_frames": payload.get("emission_frames"),
        "ctc_audio_source": audio_source,
        "ctc_audio_path": str(alignment_audio),
        "ctc_audio_note": audio_note,
        "timing_entries": len(entries),
        "assignments": assignments,
        "ctc_missing_entries": missing,
        "ctc_missing_count": len(missing),
        "ctc_score_min": round(min(ctc_scores), 6) if ctc_scores else None,
        "ctc_score_mean": round(sum(ctc_scores) / len(ctc_scores), 6) if ctc_scores else None,
        "ctc_low_score_threshold": 0.03,
        "ctc_very_low_score_threshold": 0.01,
        "ctc_low_score_count": len(low_score_entries),
        "ctc_very_low_score_count": len(very_low_score_entries),
        "ctc_low_score_entries": low_score_entries,
        "ctc_very_low_score_entries": very_low_score_entries,
        "ctc_low_score_runs": low_score_runs,
        "ctc_confidence_collapse": bool(low_score_runs),
        "suspicious_alignment_count": len(suspicious_alignments),
        "review_required_count": len(ctc_review_entries),
        "suspicious_alignment_severity_counts": {
            "high": sum(1 for item in suspicious_alignments if item["severity"] == "high"),
            "medium": sum(1 for item in suspicious_alignments if item["severity"] == "medium"),
            "low": 0,
        },
        "suspicious_alignments": suspicious_alignments,
        "review_required": bool(ctc_review_entries),
        "note": "MMS/CTC forced alignment over the known lyric order.",
    }
    apply_ctc_acoustic_backtrack(alignment_audio, entries, timestamps, report, duration)
    apply_ctc_zero_gap_boundary_realign(alignment_audio, entries, timestamps, report, duration, args)
    update_report_confidence_metrics(report)
    return timestamps, report


def run_japanese_ctc_alignment(
    audio_path: Path, entries: list[LyricEntry], duration: float, args: argparse.Namespace
) -> tuple[list[float], dict[str, object]]:
    python_exe = default_ctc_python()
    helper = Path(__file__).resolve().parent / "japanese_ctc_align.py"
    with tempfile.TemporaryDirectory(prefix="lrc-ja-ctc-") as temp_name:
        temp_dir = Path(temp_name)
        transcript_path = temp_dir / "transcript.json"
        output_path = temp_dir / "japanese-ctc.json"
        transcript_path.write_text(
            json.dumps([entry_sung_text(entry) for entry in entries], ensure_ascii=False), encoding="utf-8"
        )
        proc = run_command(
            [str(python_exe), str(helper), "--audio", str(audio_path), "--transcript", str(transcript_path), "--output", str(output_path), "--device", args.whisperx_device]
        )
        if proc.returncode != 0 or not output_path.exists():
            raise LrcError(f"Japanese CTC alignment failed: {proc.stderr.strip() or proc.stdout.strip()}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = payload.get("entries")
    if not isinstance(rows, list) or len(rows) != len(entries):
        raise LrcError("Japanese CTC helper returned an invalid entry list")
    timestamps = [max(0.0, min(duration, float(row.get("start", 0.0)))) for row in rows if isinstance(row, dict)]
    if len(timestamps) != len(entries):
        raise LrcError("Japanese CTC helper returned malformed timestamps")
    for index in range(1, len(timestamps)):
        timestamps[index] = max(timestamps[index], timestamps[index - 1] + 0.10)
    assignments = []
    for index, row in enumerate(rows):
        assert isinstance(row, dict)
        assignments.append(
            {
                "entry": index + 1,
                "segment": None,
                "score": 0.0,
                "ja_ctc_log_score": row.get("ja_ctc_log_score"),
                "ja_ctc_token_spans": row.get("token_spans"),
                "timestamp": round(timestamps[index], 3),
                "timing_repair": "japanese-ctc-forced-align",
                "timing_repair_source": "japanese-wav2vec2-ctc",
            }
        )
    collapse_entries = payload.get("collapse_entries") if isinstance(payload.get("collapse_entries"), list) else []
    report = {
        "backend": "jactc",
        "strategy": "japanese-ctc-forced-align-experimental",
        "ja_ctc_backend": payload.get("backend"),
        "ja_ctc_model": payload.get("model"),
        "ja_ctc_device": payload.get("device"),
        "ja_ctc_collapse_detected": bool(payload.get("collapse_detected")),
        "ja_ctc_collapse_entries": collapse_entries,
        "timing_entries": len(entries),
        "assignments": assignments,
        "review_required": bool(payload.get("collapse_detected")),
        "review_required_count": len(collapse_entries),
        "note": "Experimental Japanese Wav2Vec2 CTC candidate. Never selected by auto mode.",
    }
    update_report_confidence_metrics(report)
    return timestamps, report


def annotate_ctc_boundary_evidence(assignments: list[object]) -> None:
    """Record sequence-level CTC boundaries so later candidates cannot steal a prior tail."""
    previous_token_end: float | None = None
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            previous_token_end = None
            continue
        spans = assignment.get("ctc_token_spans")
        if not isinstance(spans, list) or not spans:
            assignment.pop("ctc_boundary_gap_seconds", None)
            assignment.pop("ctc_clear_boundary", None)
            previous_token_end = None
            continue
        try:
            first_start = float(spans[0].get("start"))
            first_score = float(spans[0].get("score", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            previous_token_end = None
            continue
        token_ends: list[float] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            try:
                token_ends.append(float(span.get("end")))
            except (TypeError, ValueError):
                continue
        if not token_ends:
            previous_token_end = None
            continue
        assignment["ctc_first_token_start"] = round(first_start, 3)
        assignment["ctc_first_token_score"] = round(first_score, 6)
        if index and previous_token_end is not None:
            gap = first_start - previous_token_end
            assignment["ctc_previous_token_end"] = round(previous_token_end, 3)
            assignment["ctc_boundary_gap_seconds"] = round(gap, 3)
            assignment["ctc_clear_boundary"] = gap >= 0.35 and first_score >= 0.25
        else:
            assignment["ctc_clear_boundary"] = False
        previous_token_end = max(token_ends)


def refresh_ctc_confidence_diagnostics(entries: list[LyricEntry], report: dict[str, object]) -> None:
    """Recompute CTC trust after replacing one or more bounded alignment windows."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return
    missing = {int(value) for value in report.get("ctc_missing_entries", []) if isinstance(value, int)}
    scores: list[float] = []
    low_entries: list[dict[str, object]] = []
    very_low_entries: list[dict[str, object]] = []
    low_numbers: set[int] = set()
    for index, assignment in enumerate(assignments, start=1):
        if not isinstance(assignment, dict):
            missing.add(index)
            continue
        try:
            score = float(assignment.get("ctc_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        scores.append(score)
        assignment["score"] = 0.90 if score >= 0.08 else 0.70 if score >= 0.03 else 0.35
        if score < 0.03:
            low_numbers.add(index)
            item = {"entry": index, "text": entry_sung_text(entries[index - 1]), "ctc_score": round(score, 6)}
            low_entries.append(item)
            if score < 0.01:
                very_low_entries.append(item)

    runs: list[list[int]] = []
    run: list[int] = []
    for entry_number in range(1, len(entries) + 1):
        if entry_number in low_numbers:
            run.append(entry_number)
        else:
            if len(run) >= 4:
                runs.append(run)
            run = []
    if len(run) >= 4:
        runs.append(run)
    annotate_ctc_boundary_evidence(assignments)
    collapsed = {entry for run in runs for entry in run}
    # Keep isolated low-score paths in the low-confidence metrics. They become
    # review-required only when the sequence collapses or an entry is missing.
    review = collapsed | missing
    suspicious = []
    for entry_number in sorted(review):
        flags = []
        if entry_number in missing:
            flags.append("ctc_missing_alignment")
        if entry_number in low_numbers:
            flags.append("ctc_low_path_confidence")
        if entry_number in collapsed:
            flags.append("ctc_confidence_collapse")
        suspicious.append(
            {
                "entry": entry_number,
                "text": entry_sung_text(entries[entry_number - 1]),
                "flags": flags,
                "severity": "high" if entry_number in missing or entry_number in collapsed else "medium",
                "review_required": True,
            }
        )
    report.update(
        {
            "ctc_missing_entries": sorted(missing),
            "ctc_missing_count": len(missing),
            "ctc_score_min": round(min(scores), 6) if scores else None,
            "ctc_score_mean": round(sum(scores) / len(scores), 6) if scores else None,
            "ctc_low_score_count": len(low_entries),
            "ctc_very_low_score_count": len(very_low_entries),
            "ctc_low_score_entries": low_entries,
            "ctc_very_low_score_entries": very_low_entries,
            "ctc_low_score_runs": runs,
            "ctc_confidence_collapse": bool(runs),
            "collapse_detected": bool(runs),
            "suspicious_alignments": suspicious,
            "suspicious_alignment_count": len(suspicious),
            "review_required_count": len(review),
            "review_required": bool(review),
            "suspicious_alignment_severity_counts": {
                "high": sum(1 for item in suspicious if item["severity"] == "high"),
                "medium": sum(1 for item in suspicious if item["severity"] == "medium"),
                "low": 0,
            },
        }
    )
    update_report_confidence_metrics(report)


def apply_ctc_local_window_realign(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    ctc_report: dict[str, object],
    raw_report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    """Re-run collapsed CTC ranges inside raw-ASR-bounded audio windows."""
    raw_assignments = raw_report.get("assignments")
    runs = ctc_report.get("ctc_low_score_runs")
    low_score_entries = ctc_report.get("ctc_low_score_entries")
    assignments = ctc_report.get("assignments")
    if not isinstance(raw_assignments, list) or not isinstance(runs, list) or not isinstance(assignments, list):
        return timestamps, ctc_report, []

    def raw_anchor(index: int) -> float | None:
        if not 0 <= index < len(raw_assignments):
            return None
        item = raw_assignments[index]
        if not isinstance(item, dict) or item.get("borrowed"):
            return None
        try:
            score = float(item.get("score", 0.0) or 0.0)
            timestamp = float(item.get("timestamp"))
        except (TypeError, ValueError):
            return None
        return timestamp if score >= TRUSTED_ALIGNMENT_SCORE else None

    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    errors: list[str] = []
    window_runs = [run for run in runs if isinstance(run, list) and run]
    covered_entries = {int(entry) for run in window_runs for entry in run if isinstance(entry, int)}
    if isinstance(low_score_entries, list):
        for item in low_score_entries:
            if not isinstance(item, dict) or not isinstance(item.get("entry"), int):
                continue
            entry_number = int(item["entry"])
            if entry_number not in covered_entries:
                window_runs.append([entry_number])
                covered_entries.add(entry_number)

    # A high CTC path score does not guarantee that the path starts at the
    # intended lyric onset. Use raw ASR only to identify a bounded re-alignment
    # target; the replacement still comes from known-lyric CTC in that window.
    for index, current_time in enumerate(timestamps):
        if index >= len(raw_assignments) or not isinstance(raw_assignments[index], dict):
            continue
        raw_item = raw_assignments[index]
        try:
            raw_time = float(raw_item.get("timestamp"))
            raw_score = float(raw_item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        borrowed = bool(raw_item.get("borrowed"))
        threshold = 0.45 if raw_score >= TRUSTED_ALIGNMENT_SCORE and not borrowed else 0.75
        if abs(raw_time - current_time) >= threshold:
            if isinstance(assignments[index], dict) and assignments[index].get("ctc_clear_boundary"):
                continue
            entry_number = index + 1
            if entry_number not in covered_entries:
                window_runs.append([entry_number])
                covered_entries.add(entry_number)

    merged_runs: list[list[int]] = []
    for raw_run in sorted(window_runs, key=lambda item: int(item[0])):
        start = int(raw_run[0])
        end = int(raw_run[-1])
        if merged_runs and start <= merged_runs[-1][-1] + 2:
            merged_runs[-1] = list(range(merged_runs[-1][0], max(merged_runs[-1][-1], end) + 1))
        else:
            merged_runs.append(list(range(start, end + 1)))
    window_runs = merged_runs
    python_exe = default_ctc_python()
    helper = Path(__file__).resolve().parent / "ctc_align.py"
    with tempfile.TemporaryDirectory(prefix="lrc-ctc-window-") as temp_name:
        temp_dir = Path(temp_name)
        for window_number, raw_run in enumerate(window_runs, start=1):
            if not isinstance(raw_run, list) or not raw_run:
                continue
            try:
                low_start = int(raw_run[0]) - 1
                low_end = int(raw_run[-1]) - 1
            except (TypeError, ValueError):
                continue
            left_index = next((idx for idx in range(low_start - 1, max(-1, low_start - 4), -1) if raw_anchor(idx) is not None), None)
            right_index = next((idx for idx in range(low_end + 1, min(len(entries), low_end + 4)) if raw_anchor(idx) is not None), None)
            start_index = 0 if left_index is None else left_index + 1
            end_index = len(entries) - 1 if right_index is None else right_index - 1
            # Include the preceding trusted line as CTC context. Starting the
            # audio just after its onset but omitting its text lets the target
            # line steal the preceding line's sustained vowel.
            alignment_start_index = 0 if left_index is None else left_index
            start_time = 0.0 if left_index is None else max(0.0, float(raw_anchor(left_index) or 0.0) - 0.05)
            if right_index is None:
                tail_anchor = raw_anchor(len(entries) - 1)
                end_time = min(duration, (tail_anchor + 3.0) if tail_anchor is not None else duration)
            else:
                end_time = float(raw_anchor(right_index) or duration) - 0.04
            window_duration = end_time - start_time
            if end_index < alignment_start_index or window_duration < 1.0:
                continue
            if window_duration > 45.0:
                errors.append(
                    f"window-{window_number}: skipped over-wide local CTC window ({window_duration:.1f}s)"
                )
                continue
            transcript_path = temp_dir / f"window-{window_number}.json"
            output_path = temp_dir / f"window-{window_number}.result.json"
            transcript_path.write_text(
                json.dumps([entry_sung_text(entry) for entry in entries[alignment_start_index : end_index + 1]], ensure_ascii=False),
                encoding="utf-8",
            )
            command = [
                str(python_exe), str(helper), "--audio", str(audio_path), "--transcript", str(transcript_path),
                "--output", str(output_path), "--device", args.whisperx_device,
                "--start", f"{start_time:.3f}", "--end", f"{end_time:.3f}",
            ]
            proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0 or not output_path.exists():
                errors.append((proc.stderr or proc.stdout).strip()[:240])
                continue
            rows = json.loads(output_path.read_text(encoding="utf-8")).get("entries")
            if not isinstance(rows, list) or len(rows) != end_index - alignment_start_index + 1:
                errors.append(f"window-{window_number}: invalid helper output")
                continue
            window_changes = 0
            for offset, row in enumerate(rows):
                if not isinstance(row, dict) or not isinstance(row.get("start"), (int, float)):
                    continue
                index = alignment_start_index + offset
                if index < start_index:
                    continue
                timestamp = float(row["start"])
                if not start_time <= timestamp <= end_time:
                    continue
                refined[index] = timestamp
                assignment = assignments[index]
                if isinstance(assignment, dict):
                    assignment["timestamp"] = round(timestamp, 3)
                    assignment["ctc_score"] = row.get("ctc_score")
                    assignment["ctc_token_spans"] = row.get("token_spans")
                    assignment["ctc_first_token_candidates"] = row.get("first_token_candidates")
                    assignment["ctc_local_window_realign"] = True
                    assignment["ctc_local_window"] = {"start": round(start_time, 3), "end": round(end_time, 3)}
                window_changes += 1
            if window_changes:
                changes.append(
                    {
                        "entries": list(range(start_index + 1, end_index + 2)),
                        "context_entry": left_index + 1 if left_index is not None else None,
                        "start": round(start_time, 3),
                        "end": round(end_time, 3),
                    }
                )
    for index in range(1, len(refined)):
        refined[index] = max(refined[index], refined[index - 1] + 0.10)
    ctc_report["ctc_local_window_realign"] = {"status": "applied" if changes else "skipped", "windows": changes, "errors": errors}
    if changes:
        refresh_ctc_confidence_diagnostics(entries, ctc_report)
    return refined, ctc_report, changes


def apply_ctc_hybrid_window_realign(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    ctc_report: dict[str, object],
    whisperx_report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    """Re-align only CTC/WhisperX disagreement windows before local fusion."""
    ctc_assignments = ctc_report.get("assignments")
    whisperx_assignments = whisperx_report.get("assignments")
    if not isinstance(ctc_assignments, list) or not isinstance(whisperx_assignments, list):
        return timestamps, ctc_report, []

    target_indices: list[int] = []
    for index, (ctc_item, whisperx_item) in enumerate(zip(ctc_assignments, whisperx_assignments)):
        if not isinstance(ctc_item, dict) or not isinstance(whisperx_item, dict):
            continue
        try:
            ctc_time = float(ctc_item.get("timestamp", timestamps[index]) or timestamps[index])
            whisperx_time = float(whisperx_item.get("timestamp"))
        except (TypeError, ValueError):
            continue
        first_score = 1.0
        spans = ctc_item.get("ctc_token_spans")
        if isinstance(spans, list) and spans:
            try:
                first_score = float(spans[0].get("score", 1.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                pass
        disagreement = abs(ctc_time - whisperx_time)
        if disagreement >= 1.0 or (disagreement >= 0.45 and first_score <= 0.05):
            target_indices.append(index)
    if not target_indices:
        return timestamps, ctc_report, []

    runs: list[list[int]] = []
    for index in target_indices:
        if runs and index <= runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])

    helper = Path(__file__).resolve().parent / "ctc_align.py"
    python_exe = default_ctc_python()
    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="lrc-ctc-hybrid-window-") as temp_name:
        temp_dir = Path(temp_name)
        for number, run in enumerate(runs, start=1):
            run_start, run_end = run[0], run[-1]
            context_start = max(0, run_start - 1)
            context_end = min(len(entries) - 1, run_end + 1)
            right_boundary = min(len(entries) - 1, context_end + 1)
            start_time = max(0.0, refined[context_start] - 0.08)
            end_time = duration if right_boundary == context_end else max(start_time + 1.0, refined[right_boundary] - 0.05)
            window_duration = end_time - start_time
            if window_duration < 1.0 or window_duration > 45.0:
                continue
            transcript_path = temp_dir / f"window-{number}.json"
            output_path = temp_dir / f"window-{number}.result.json"
            transcript_path.write_text(
                json.dumps([entry_sung_text(entry) for entry in entries[context_start : context_end + 1]], ensure_ascii=False),
                encoding="utf-8",
            )
            command = [
                str(python_exe), str(helper), "--audio", str(audio_path), "--transcript", str(transcript_path),
                "--output", str(output_path), "--device", args.whisperx_device,
                "--start", f"{start_time:.3f}", "--end", f"{end_time:.3f}",
            ]
            proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0 or not output_path.exists():
                continue
            rows = json.loads(output_path.read_text(encoding="utf-8")).get("entries")
            if not isinstance(rows, list) or len(rows) != context_end - context_start + 1:
                continue
            indexes_to_update = list(run)
            # A disputed line can make the immediately following line lose a
            # soft initial consonant. Keep that context row only when its own
            # aligned first token is weak; otherwise leave an unflagged line
            # untouched.
            if context_end not in indexes_to_update:
                context_row = rows[context_end - context_start]
                context_spans = context_row.get("token_spans") if isinstance(context_row, dict) else None
                try:
                    context_first_score = float(context_spans[0].get("score", 1.0) or 0.0)
                except (AttributeError, TypeError, ValueError, IndexError):
                    context_first_score = 1.0
                context_start_time = float(context_row.get("start", 0.0) or 0.0) if isinstance(context_row, dict) else 0.0
                context_peaks = context_row.get("first_token_candidates") if isinstance(context_row, dict) else None
                has_strong_early_peak = any(
                    isinstance(peak, dict)
                    and isinstance(peak.get("time"), (int, float))
                    and isinstance(peak.get("score"), (int, float))
                    and float(peak["time"]) < context_start_time - 0.45
                    and float(peak["score"]) >= 0.05
                    for peak in context_peaks
                ) if isinstance(context_peaks, list) else False
                if context_first_score <= 0.05 or has_strong_early_peak:
                    indexes_to_update.append(context_end)
            updated_entries: list[int] = []
            for index in indexes_to_update:
                row = rows[index - context_start]
                if not isinstance(row, dict) or not isinstance(row.get("start"), (int, float)):
                    continue
                timestamp = float(row["start"])
                if not start_time <= timestamp <= end_time:
                    continue
                refined[index] = timestamp
                assignment = ctc_assignments[index]
                if isinstance(assignment, dict):
                    assignment["timestamp"] = round(timestamp, 3)
                    assignment["ctc_score"] = row.get("ctc_score")
                    assignment["ctc_token_spans"] = row.get("token_spans")
                    assignment["ctc_first_token_candidates"] = row.get("first_token_candidates")
                    assignment["ctc_hybrid_window_realign"] = True
                updated_entries.append(index + 1)
            if updated_entries:
                changes.append({
                    "entries": updated_entries,
                    "context_entries": [context_start + 1, context_end + 1],
                    "start": round(start_time, 3),
                    "end": round(end_time, 3),
                })
    if changes:
        ctc_report["ctc_hybrid_window_realign"] = {"status": "applied", "windows": changes}
        refresh_ctc_confidence_diagnostics(entries, ctc_report)
    return refined, ctc_report, changes


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
    aligned_segments = whisperx_segments(aligned_payload)
    timestamps, report = match_whisper_segments(entries, aligned_segments, duration)
    report["backend"] = "whisperx"
    report["asr_segment_timing_source"] = "whisperx_aligned"
    report["whisperx_aligned_asr_segments"] = len(aligned_segments)
    report["whisper_suppress_nst"] = suppress_nst
    report["whisperx_device"] = aligned_payload.get("device")
    report["vocal_regions"] = []
    report["vocal_regions_available"] = False
    report["no_vocal_constraint"] = {
        "status": "placeholder",
        "enforced": False,
        "reason": "No reliable vocal-region detector is wired yet.",
    }
    report["local_window_realign"] = {
        "status": "placeholder",
        "enabled": False,
        "reason": "Disabled until a reliable vocal-region/onset constraint is available.",
    }
    if is_alignment_unusable(report):
        raise LrcError(
            "WhisperX ASR-to-lyric matching is unusable; refusing to write a fake LRC. "
            f"{unusable_alignment_reason(report)}"
        )
    timestamps, intro_changes = repair_intro_hallucination(audio_path, entries, timestamps, report, duration)
    if intro_changes:
        sync_report_assignment_timestamps(report, timestamps)
    timestamps, long_segment_changes = repair_long_segment_hallucinations(
        audio_path,
        entries,
        timestamps,
        report,
        aligned_segments,
        duration,
    )
    if long_segment_changes:
        sync_report_assignment_timestamps(report, timestamps)
    timestamps, tail_changes = repair_tail_hallucination(
        audio_path,
        entries,
        timestamps,
        report,
        aligned_segments,
        duration,
    )
    if tail_changes:
        sync_report_assignment_timestamps(report, timestamps)
    forced_payload = run_whisperx_alignment(audio_path, lyric_alignment_windows(entries, timestamps, duration), args)
    forced_segments = whisperx_segments(forced_payload)
    timestamps, refinement_changes = apply_whisperx_lyric_refinement(
        entries,
        timestamps,
        report,
        aligned_segments,
        forced_segments,
        duration,
    )
    sync_report_assignment_timestamps(report, timestamps)
    report["backend"] = "whisperx"
    report["strategy"] = "whisperx-hybrid-experimental"
    report["whisper_suppress_nst"] = suppress_nst
    report["whisperx_device"] = forced_payload.get("device") or aligned_payload.get("device")
    report["whisper_language"] = args.whisper_language
    report["whisperx_forced_first_times"] = [round(first_forced_char_time(segment), 3) for segment in forced_segments]
    report["whisperx_refinement_count"] = len(refinement_changes)
    report["whisperx_refinements"] = refinement_changes
    timestamps, tail_blend_changes = blend_tail_repair_with_forced(
        entries,
        timestamps,
        report,
        forced_segments,
        duration,
    )
    if tail_blend_changes:
        sync_report_assignment_timestamps(report, timestamps)
    add_alignment_diagnostics(entries, timestamps, report, forced_segments, duration)
    if is_alignment_untrusted(report):
        raise LrcError(
            "WhisperX alignment is untrusted after diagnostics; refusing to write a fake LRC. "
            f"{untrusted_alignment_reason(report)}"
        )
    return timestamps, report


def report_quality(report: dict[str, object]) -> float:
    matched_percent = float(report.get("trusted_percent", report.get("matched_percent", 0.0)) or 0.0)
    low_confidence_count = int(report.get("low_confidence_count", 0) or 0)
    review_required_count = int(report.get("review_required_count", 0) or 0)
    collapse_penalty = 80.0 if report.get("collapse_detected") else 0.0
    return matched_percent - low_confidence_count * 2.5 - review_required_count * 3.0 - collapse_penalty


def candidate_selection_quality(report: dict[str, object]) -> float:
    if report.get("error"):
        return -100.0
    backend = str(report.get("backend", "") or "")
    if backend == "ctc":
        timing_entries = int(report.get("timing_entries", 0) or 0)
        missing_count = int(report.get("ctc_missing_count", 0) or 0)
        low_score_count = int(report.get("ctc_low_score_count", 0) or 0)
        very_low_score_count = int(report.get("ctc_very_low_score_count", 0) or 0)
        review_required_count = int(report.get("review_required_count", 0) or 0)
        quality = 100.0
        quality -= missing_count * 30.0
        quality -= low_score_count * 6.0
        quality -= very_low_score_count * 8.0
        quality -= review_required_count * 2.0
        if timing_entries and missing_count >= math.ceil(timing_entries * 0.20):
            quality -= 50.0
        if report.get("collapse_detected"):
            quality -= 100.0
        return max(-100.0, min(100.0, quality))
    if backend in {"whisperx", "hybrid"}:
        timing_entries = int(report.get("timing_entries", 0) or 0)
        trusted_percent = float(report.get("trusted_percent", report.get("matched_percent", 0.0)) or 0.0)
        low_confidence_percent = float(report.get("low_confidence_percent", 0.0) or 0.0)
        review_required_percent = float(report.get("review_required_percent", 0.0) or 0.0)
        severity_counts = report.get("suspicious_alignment_severity_counts")
        high_risk_count = 0
        if isinstance(severity_counts, dict):
            try:
                high_risk_count = int(severity_counts.get("high", 0) or 0)
            except (TypeError, ValueError):
                high_risk_count = 0
        quality = trusted_percent
        quality -= low_confidence_percent * 0.40
        quality -= review_required_percent * 0.15
        quality -= high_risk_count * 1.0
        if backend == "hybrid":
            quality -= high_risk_count * 1.5
        if timing_entries >= 8 and trusted_percent < 85.0:
            quality -= 10.0
        if report.get("collapse_detected"):
            quality -= 100.0
        return max(-100.0, min(100.0, quality))
    return report_quality(report)


def should_prefer_ctc_over_review_exploded_hybrid(
    hybrid_report: dict[str, object], ctc_report: dict[str, object]
) -> bool:
    """Reject a locally patched hybrid when it becomes much less auditable than CTC."""
    if ctc_report.get("collapse_detected"):
        return False
    try:
        ctc_missing = int(ctc_report.get("ctc_missing_count", 0) or 0)
        ctc_reviews = int(ctc_report.get("review_required_count", 0) or 0)
        hybrid_reviews = int(hybrid_report.get("review_required_count", 0) or 0)
    except (TypeError, ValueError):
        return False
    if ctc_missing:
        return False
    return hybrid_reviews >= max(8, ctc_reviews * 3 + 5)


def choose_onset_consensus_time(
    ctc_time: float,
    ctc_first_score: float,
    whisper_time: float,
    whisper_score: float,
    japanese_ctc_time: float,
) -> tuple[float | None, str | None]:
    """Choose a narrow onset rescue when CTC's first phoneme is unreliable."""
    ctc_whisper_gap = abs(ctc_time - whisper_time)
    whisper_japanese_gap = abs(whisper_time - japanese_ctc_time)
    if (
        ctc_first_score <= 0.01
        and whisper_score >= 0.80
        and 0.35 <= ctc_whisper_gap <= 0.80
        and 0.60 <= whisper_japanese_gap <= 1.00
    ):
        return whisper_time, "high-confidence-whisperx-over-weak-ctc-initial"
    if (
        ctc_first_score <= 0.01
        and whisper_score >= 0.50
        and ctc_whisper_gap >= 0.20
        and whisper_japanese_gap <= 0.15
    ):
        return (whisper_time + japanese_ctc_time) / 2.0, "whisperx-japanese-ctc-onset-consensus"
    if (
        0.05 <= ctc_first_score <= 0.10
        and whisper_score >= 0.80
        and whisper_japanese_gap <= 0.15
        and ctc_whisper_gap >= 0.20
    ):
        return (whisper_time + japanese_ctc_time) / 2.0, "whisperx-japanese-ctc-over-ctc-onset-consensus"
    return None, None


def apply_japanese_onset_consensus_to_ctc(
    timestamps: list[float],
    ctc_report: dict[str, object],
    whisper_timestamps: list[float],
    whisper_report: dict[str, object],
    japanese_timestamps: list[float],
    duration: float,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    assignments = ctc_report.get("assignments")
    whisper_assignments = whisper_report.get("assignments")
    if not isinstance(assignments, list) or not isinstance(whisper_assignments, list):
        return timestamps, ctc_report, []
    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if index >= len(whisper_assignments) or index >= len(whisper_timestamps) or index >= len(japanese_timestamps):
            continue
        whisper_assignment = whisper_assignments[index]
        if not isinstance(assignment, dict) or not isinstance(whisper_assignment, dict):
            continue
        spans = assignment.get("ctc_token_spans")
        if not isinstance(spans, list) or not spans:
            continue
        try:
            ctc_first_score = float(spans[0].get("score", 0.0) or 0.0)
            whisper_score = float(whisper_assignment.get("score", 0.0) or 0.0)
            ctc_time = float(refined[index])
            whisper_time = float(whisper_timestamps[index])
            japanese_time = float(japanese_timestamps[index])
        except (AttributeError, TypeError, ValueError):
            continue
        candidate, reason = choose_onset_consensus_time(
            ctc_time, ctc_first_score, whisper_time, whisper_score, japanese_time
        )
        if candidate is None or reason is None:
            continue
        previous_time = refined[index - 1] if index else 0.0
        next_time = refined[index + 1] if index + 1 < len(refined) else duration
        if not previous_time + 0.05 < candidate < next_time - 0.05:
            continue
        refined[index] = candidate
        assignment["timestamp"] = round(candidate, 3)
        assignment["ctc_onset_consensus"] = True
        assignment["ctc_onset_consensus_reason"] = reason
        assignment["ctc_onset_consensus_candidates"] = {
            "ctc": round(ctc_time, 3), "whisperx": round(whisper_time, 3), "japanese_ctc": round(japanese_time, 3),
        }
        changes.append({"entry": index + 1, "from": round(ctc_time, 3), "to": round(candidate, 3), "reason": reason})
    if changes:
        ctc_report["ctc_onset_consensus"] = changes
    return refined, ctc_report, changes


def candidate_summary(report: dict[str, object], error: str | None = None) -> dict[str, object]:
    quality = candidate_selection_quality(report)
    summary: dict[str, object] = {
        "backend": report.get("backend"),
        "strategy": report.get("strategy"),
        "quality": round(quality, 3),
        "assigned_percent": report.get("assigned_percent"),
        "trusted_percent": report.get("trusted_percent", report.get("matched_percent")),
        "low_confidence_percent": report.get("low_confidence_percent"),
        "review_required_percent": report.get("review_required_percent"),
        "collapse_detected": bool(report.get("collapse_detected")),
        "error": error or report.get("error"),
    }
    if report.get("backend") == "ctc":
        summary.update(
            {
                "ctc_device": report.get("ctc_device"),
                "ctc_score_min": report.get("ctc_score_min"),
                "ctc_score_mean": report.get("ctc_score_mean"),
                "ctc_low_score_count": report.get("ctc_low_score_count"),
                "ctc_very_low_score_count": report.get("ctc_very_low_score_count"),
                "ctc_missing_count": report.get("ctc_missing_count"),
            }
        )
    if report.get("backend") == "whisperx":
        summary.update(
            {
                "whisper_suppress_nst": report.get("whisper_suppress_nst"),
                "whisperx_device": report.get("whisperx_device"),
                "legacy_quality": round(report_quality(report), 3),
            }
        )
    return summary


def needs_suppressed_retry(report: dict[str, object]) -> bool:
    matched_percent = float(report.get("trusted_percent", report.get("matched_percent", 0.0)) or 0.0)
    timing_entries = int(report.get("timing_entries", 0) or 0)
    low_confidence_count = int(report.get("low_confidence_count", 0) or 0)
    low_limit = max(6, math.ceil(timing_entries * 0.20))
    return matched_percent < 90.0 or low_confidence_count > low_limit


def is_alignment_unusable(report: dict[str, object]) -> bool:
    timing_entries = int(report.get("timing_entries", 0) or 0)
    matched_entries = int(report.get("trusted_entries", report.get("matched_entries", 0)) or 0)
    matched_percent = float(report.get("trusted_percent", report.get("matched_percent", 0.0)) or 0.0)
    low_confidence_count = int(report.get("low_confidence_count", 0) or 0)
    if timing_entries <= 0:
        return True
    if matched_entries == 0:
        return True
    if matched_percent < 35.0:
        return True
    if timing_entries >= 6 and low_confidence_count >= math.ceil(timing_entries * 0.80):
        return True
    return False


def is_alignment_untrusted(report: dict[str, object]) -> bool:
    if is_alignment_unusable(report):
        return True
    if report.get("collapse_detected"):
        return True
    timing_entries = int(report.get("timing_entries", 0) or 0)
    trusted_percent = float(report.get("trusted_percent", report.get("matched_percent", 0.0)) or 0.0)
    low_confidence_percent = float(report.get("low_confidence_percent", 0.0) or 0.0)
    high_risk_count = 0
    max_post_long_uncertain_run = 0
    suspicious = report.get("suspicious_alignments")
    if isinstance(suspicious, list):
        high_risk_count = sum(
            1 for item in suspicious if isinstance(item, dict) and item.get("severity") == "high"
        )
        flagged_entries: set[int] = set()
        for item in suspicious:
            if not isinstance(item, dict):
                continue
            flags = item.get("flags")
            if not isinstance(flags, list) or "post_long_segment_region_uncertain" not in flags:
                continue
            try:
                flagged_entries.add(int(item.get("entry", 0)))
            except (TypeError, ValueError):
                continue
        current_run = 0
        for entry_index in range(1, timing_entries + 1):
            if entry_index in flagged_entries:
                current_run += 1
                max_post_long_uncertain_run = max(max_post_long_uncertain_run, current_run)
            else:
                current_run = 0
        report["max_post_long_uncertain_run"] = max_post_long_uncertain_run
    if timing_entries >= 8 and trusted_percent < 85.0:
        return True
    if timing_entries >= 8 and low_confidence_percent >= 50.0:
        return True
    if timing_entries >= 8 and max_post_long_uncertain_run >= 4:
        return True
    if timing_entries >= 8 and high_risk_count >= math.ceil(timing_entries * 0.25):
        return True
    return False


def unusable_alignment_reason(report: dict[str, object]) -> str:
    return (
        f"trusted_entries={report.get('trusted_entries', report.get('matched_entries'))}, "
        f"timing_entries={report.get('timing_entries')}, "
        f"trusted_percent={report.get('trusted_percent', report.get('matched_percent'))}, "
        f"assigned_percent={report.get('assigned_percent')}, "
        f"low_confidence_count={report.get('low_confidence_count')}"
    )


def untrusted_alignment_reason(report: dict[str, object]) -> str:
    segment_runs = report.get("segment_collapse_runs")
    zero_runs = report.get("alignment_collapse_runs")
    return (
        f"trusted_entries={report.get('trusted_entries', report.get('matched_entries'))}, "
        f"timing_entries={report.get('timing_entries')}, "
        f"trusted_percent={report.get('trusted_percent', report.get('matched_percent'))}, "
        f"assigned_percent={report.get('assigned_percent')}, "
        f"low_confidence_percent={report.get('low_confidence_percent')}, "
        f"review_required_percent={report.get('review_required_percent')}, "
        f"collapse_detected={report.get('collapse_detected')}, "
        f"max_post_long_uncertain_run={report.get('max_post_long_uncertain_run')}, "
        f"segment_collapse_runs={segment_runs}, "
        f"alignment_collapse_runs={zero_runs}"
    )


def try_whisperx_candidate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
    suppress_nst: bool,
) -> tuple[list[float], dict[str, object], str | None]:
    try:
        timestamps, report = run_whisperx_candidate(audio_path, entries, duration, args, suppress_nst)
        return timestamps, report, None
    except LrcError as exc:
        return [], {"whisper_suppress_nst": suppress_nst, "error": str(exc)}, str(exc)


def try_ctc_candidate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object], str | None]:
    try:
        timestamps, report = run_ctc_alignment(audio_path, entries, duration, args)
        return timestamps, report, None
    except LrcError as exc:
        return [], {"backend": "ctc", "error": str(exc)}, str(exc)


def ctc_alignment_audio_path(report: dict[str, object], fallback: Path) -> Path:
    candidate = report.get("ctc_audio_path")
    if isinstance(candidate, str):
        path = Path(candidate)
        if path.exists():
            return path
    return fallback


def try_whispercpp_diagnostic_candidate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object], str | None]:
    try:
        raw_segments = run_whispercpp(audio_path, args, suppress_nst=False)
        timestamps, report = match_whisper_segments(entries, raw_segments, duration)
        report["backend"] = "whispercpp_raw"
        report["strategy"] = "whispercpp-raw-diagnostic"
        report["asr_repeated_tail_after_0s"] = has_repeated_asr_tail(raw_segments, 0.0)
        report["raw_asr_segments"] = [
            {"start": round(segment.start, 3), "end": round(segment.end, 3), "text": segment.text}
            for segment in raw_segments
        ]
        report["raw_asr_max_segment_seconds"] = round(
            max((segment.end - segment.start for segment in raw_segments), default=0.0), 3
        )
        report["diagnostic_only"] = True
        return timestamps, report, None
    except LrcError as exc:
        return [], {"backend": "whispercpp_raw", "diagnostic_only": True, "error": str(exc)}, str(exc)


def run_whisperx_best_candidate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object]]:
    if args.whisper_suppress_nst is not None:
        return run_whisperx_candidate(
            audio_path,
            entries,
            duration,
            args,
            suppress_nst=bool(args.whisper_suppress_nst),
        )

    timestamps, report, normal_error = try_whisperx_candidate(
        audio_path, entries, duration, args, suppress_nst=False
    )
    candidates = [report]
    candidate_errors = [normal_error] if normal_error else []
    if normal_error:
        sns_timestamps, sns_report, sns_error = try_whisperx_candidate(
            audio_path, entries, duration, args, suppress_nst=True
        )
        candidates.append(sns_report)
        if sns_error:
            candidate_errors.append(sns_error)
            raise LrcError(
                "No usable WhisperX candidate; refusing to write a fake LRC. "
                + " | ".join(candidate_errors)
            )
        timestamps, report = sns_timestamps, sns_report

    normal_timestamps = list(timestamps)
    normal_report = report
    if needs_suppressed_retry(report):
        sns_timestamps, sns_report, sns_error = try_whisperx_candidate(
            audio_path, entries, duration, args, suppress_nst=True
        )
        candidates.append(sns_report)
        if not sns_error:
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
                    right_ok = (
                        entry_index + 1 >= len(timestamps)
                        or sns_timestamps[entry_index] < timestamps[entry_index + 1] - 0.10
                    )
                    if left_ok and right_ok:
                        timestamps[entry_index] = sns_timestamps[entry_index]
                        adopted_sns_entries.append(entry_index)
                report = fused_candidate_report(
                    normal_report, sns_report, timestamps, collapse_start, adopted_sns_entries
                )
            elif report_quality(sns_report) > report_quality(report):
                timestamps, report = sns_timestamps, sns_report

    report["candidate_reports"] = [
        candidate_summary(candidate, str(candidate.get("error")) if candidate.get("error") else None)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    return timestamps, report


def choose_alignment_candidate(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    usable = [
        candidate
        for candidate in candidates
        if not candidate.get("error")
        and isinstance(candidate.get("timestamps"), list)
        and isinstance(candidate.get("report"), dict)
    ]
    if not usable:
        errors = [
            str(candidate.get("error"))
            for candidate in candidates
            if candidate.get("error")
        ]
        raise LrcError("No usable alignment candidate. " + " | ".join(errors))

    for candidate in usable:
        report = candidate["report"]
        assert isinstance(report, dict)
        candidate["quality"] = candidate_selection_quality(report)

    best = max(usable, key=lambda item: float(item.get("quality", -100.0) or -100.0))
    ctc = next((item for item in usable if item.get("backend") == "ctc"), None)
    if ctc is not None:
        ctc_quality = float(ctc.get("quality", -100.0) or -100.0)
        best_quality = float(best.get("quality", -100.0) or -100.0)
        if ctc_quality >= 80.0 and ctc_quality >= best_quality - 5.0:
            return ctc
    return best


def candidate_selection_reason(selected: dict[str, object], candidates: list[dict[str, object]]) -> str:
    selected_backend = str(selected.get("backend", "unknown"))
    selected_quality = float(selected.get("quality", -100.0) or -100.0)
    parts = [f"selected {selected_backend} quality={selected_quality:.1f}"]
    for candidate in candidates:
        backend = candidate.get("backend", "unknown")
        if candidate.get("error"):
            parts.append(f"{backend} error={candidate.get('error')}")
            continue
        report = candidate.get("report")
        if isinstance(report, dict):
            quality = candidate_selection_quality(report)
            if backend == "ctc":
                parts.append(
                    "ctc "
                    f"quality={quality:.1f}, "
                    f"low={report.get('ctc_low_score_count')}, "
                    f"very_low={report.get('ctc_very_low_score_count')}, "
                    f"missing={report.get('ctc_missing_count')}"
                )
            elif backend == "whisperx":
                parts.append(
                    "whisperx "
                    f"quality={quality:.1f}, "
                    f"trusted={report.get('trusted_percent', report.get('matched_percent'))}, "
                    f"review={report.get('review_required_percent')}, "
                    f"collapse={report.get('collapse_detected')}"
                )
            elif backend == "whispercpp":
                parts.append(
                    "whispercpp "
                    f"quality={quality:.1f}, "
                    f"trusted={report.get('trusted_percent', report.get('matched_percent'))}, "
                    f"review={report.get('review_required_percent')}"
                )
    if selected_backend == "ctc":
        parts.append("ctc is preferred on near-ties because it is forced to the known lyric order")
    return "; ".join(parts)


def raw_asr_is_fallback_eligible(report: dict[str, object]) -> bool:
    """Reject raw fallback when Whisper merged a large section into one segment."""
    try:
        max_segment = float(report.get("raw_asr_max_segment_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return 0.0 < max_segment <= 12.0


def apply_ctc_weak_prefix_recovery(
    timestamps: list[float], report: dict[str, object], duration: float
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    """Recover a sung prefix that CTC skipped before locking onto the line."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps):
        return timestamps, report, []

    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict) or index == 0:
            continue
        try:
            current_time = float(assignment.get("timestamp", refined[index]) or refined[index])
            ctc_score = float(assignment.get("ctc_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        # Whole-line confidence can be high even when only the first phoneme
        # was dropped, so gate on the weak first token and a strong earlier peak.
        if ctc_score > 0.60:
            continue
        spans = assignment.get("ctc_token_spans")
        peaks = assignment.get("ctc_first_token_candidates")
        if not isinstance(spans, list) or not spans or not isinstance(peaks, list):
            continue
        try:
            first_token_score = float(spans[0].get("score", 1.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            continue
        # A bounded hybrid window can reveal a stronger earlier onset even when
        # the eventual CTC token itself is not strictly low confidence.
        if first_token_score > 0.05 and not assignment.get("ctc_hybrid_window_realign"):
            continue

        previous_time = refined[index - 1]
        previous_token_end = previous_time
        previous_assignment = assignments[index - 1]
        if isinstance(previous_assignment, dict):
            previous_spans = previous_assignment.get("ctc_token_spans")
            if isinstance(previous_spans, list):
                for span in previous_spans:
                    if not isinstance(span, dict):
                        continue
                    try:
                        previous_token_end = max(previous_token_end, float(span.get("end")))
                    except (TypeError, ValueError):
                        continue
        next_time = refined[index + 1] if index + 1 < len(refined) else duration
        eligible: list[tuple[float, float]] = []
        for peak in peaks:
            if not isinstance(peak, dict):
                continue
            try:
                peak_time = float(peak.get("time"))
                peak_score = float(peak.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if (
                previous_token_end + 0.03 < peak_time < current_time - 0.45
                and peak_time < next_time - 0.10
                and peak_score >= 0.05
                and current_time - peak_time <= 2.25
            ):
                eligible.append((peak_time, peak_score))
        if not eligible:
            continue
        peak_time, peak_score = max(eligible, key=lambda item: item[1])
        refined[index] = peak_time
        assignment["timestamp"] = round(peak_time, 3)
        assignment["ctc_weak_prefix_recovery"] = True
        assignment["ctc_weak_prefix_recovery_from"] = round(current_time, 3)
        assignment["ctc_weak_prefix_peak_score"] = round(peak_score, 6)
        changes.append(
            {
                "entry": index + 1,
                "from": round(current_time, 3),
                "to": round(peak_time, 3),
                "shift": round(current_time - peak_time, 3),
                "ctc_score": round(ctc_score, 6),
                "first_token_score": round(first_token_score, 6),
                "prefix_peak_score": round(peak_score, 6),
                "reason": "strong-first-token-posterior-before-weak-ctc-prefix",
            }
        )
    if changes:
        report["ctc_weak_prefix_recovery_count"] = len(changes)
        report["ctc_weak_prefix_recoveries"] = changes
    return refined, report, changes


def apply_ctc_crossline_initial_recovery(
    timestamps: list[float], report: dict[str, object], duration: float
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    """Reject a current-line initial token that is actually the prior line's tail."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps):
        return timestamps, report, []

    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    for index in range(1, len(assignments)):
        previous = assignments[index - 1]
        current = assignments[index]
        if not isinstance(previous, dict) or not isinstance(current, dict):
            continue
        previous_spans = previous.get("ctc_token_spans")
        current_spans = current.get("ctc_token_spans")
        if not isinstance(previous_spans, list) or not isinstance(current_spans, list) or len(current_spans) < 2:
            continue
        try:
            previous_end = max(float(span.get("end")) for span in previous_spans if isinstance(span, dict))
            current_start = float(current_spans[0].get("start"))
            next_token_start = float(current_spans[1].get("start"))
        except (AttributeError, TypeError, ValueError):
            continue
        if current_start > previous_end + 0.06 or next_token_start - current_start < 0.70:
            continue
        next_time = refined[index + 1] if index + 1 < len(refined) else duration
        if not previous_end + 0.05 < next_token_start < next_time - 0.10:
            continue
        old_time = refined[index]
        refined[index] = next_token_start
        current["timestamp"] = round(next_token_start, 3)
        current["ctc_crossline_initial_recovery"] = True
        current["ctc_crossline_initial_recovery_from"] = round(old_time, 3)
        changes.append(
            {
                "entry": index + 1,
                "from": round(old_time, 3),
                "to": round(next_token_start, 3),
                "previous_token_end": round(previous_end, 3),
                "detached_initial_gap": round(next_token_start - current_start, 3),
                "reason": "current-initial-token-overlaps-previous-line-tail",
            }
        )
    if changes:
        report["ctc_crossline_initial_recovery_count"] = len(changes)
        report["ctc_crossline_initial_recoveries"] = changes
    return refined, report, changes


def apply_ctc_local_fusion_to_whisperx(
    whisperx_timestamps: list[float],
    whisperx_report: dict[str, object],
    ctc_report: dict[str, object],
    duration: float,
    raw_report: dict[str, object] | None = None,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    whisperx_assignments = whisperx_report.get("assignments")
    ctc_assignments = ctc_report.get("assignments")
    if not isinstance(whisperx_assignments, list) or not isinstance(ctc_assignments, list):
        return whisperx_timestamps, whisperx_report, []
    raw_assignments = raw_report.get("assignments") if isinstance(raw_report, dict) else None

    fused = list(whisperx_timestamps)
    changes: list[dict[str, object]] = []
    max_items = min(len(fused), len(whisperx_assignments), len(ctc_assignments))
    for index in range(max_items):
        whisperx_item = whisperx_assignments[index]
        ctc_item = ctc_assignments[index]
        if not isinstance(whisperx_item, dict) or not isinstance(ctc_item, dict):
            continue
        try:
            whisperx_time = float(whisperx_item.get("timestamp", fused[index]) or fused[index])
            ctc_time = float(ctc_item.get("timestamp"))
            ctc_score = float(ctc_item.get("ctc_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        disagreement = ctc_time - whisperx_time
        prefix_recovered = bool(ctc_item.get("ctc_weak_prefix_recovery"))
        crossline_recovered = bool(ctc_item.get("ctc_crossline_initial_recovery"))
        clear_boundary = bool(ctc_item.get("ctc_clear_boundary"))
        if abs(disagreement) < 1.10 and not ((prefix_recovered or crossline_recovered) and abs(disagreement) >= 0.35):
            continue
        previous_time = fused[index - 1] if index > 0 else 0.0
        next_time = fused[index + 1] if index + 1 < len(fused) else duration

        try:
            whisperx_score = float(whisperx_item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            whisperx_score = 0.0
        previous_same_segment = False
        raw_time: float | None = None
        raw_score: float | None = None
        raw_ctc_agree = False
        if isinstance(raw_assignments, list) and index < len(raw_assignments):
            raw_item = raw_assignments[index]
            if isinstance(raw_item, dict):
                try:
                    raw_time = float(raw_item.get("timestamp"))
                    raw_score = float(raw_item.get("score", 0.0) or 0.0)
                    raw_ctc_agree = raw_score >= 0.72 and abs(raw_time - ctc_time) <= 0.75
                except (TypeError, ValueError):
                    raw_time = None
                    raw_score = None
                    raw_ctc_agree = False
        raw_between_candidates = (
            raw_time is not None
            and raw_score is not None
            and ctc_score >= 0.08
            and raw_score >= 0.82
            and abs(raw_time - whisperx_time) >= 0.25
            and min(ctc_time, whisperx_time) < raw_time < max(ctc_time, whisperx_time)
        )
        raw_high_conflict_anchor = (
            raw_time is not None
            and raw_score is not None
            and raw_score >= 0.95
            and 1.50 <= abs(raw_time - whisperx_time) <= 3.50
            and min(ctc_time, whisperx_time) < raw_time < max(ctc_time, whisperx_time)
            and previous_time + 3.00 < raw_time < next_time - 1.00
        )
        target_time = ctc_time
        target_source = "ctc"
        if raw_high_conflict_anchor:
            target_time = raw_time
            target_source = "raw_asr_conflict_anchor"
        elif raw_between_candidates and not raw_ctc_agree:
            target_time = raw_time
            target_source = "raw_asr_segment_internal"
        elif ctc_score < 0.08:
            continue
        if not (previous_time + 0.10 < target_time < next_time - 0.10):
            continue
        if index > 0 and isinstance(whisperx_assignments[index - 1], dict):
            previous_item = whisperx_assignments[index - 1]
            previous_same_segment = (
                previous_item.get("segment") is not None
                and previous_item.get("segment") == whisperx_item.get("segment")
                and float(previous_item.get("score", 0.0) or 0.0) < TRUSTED_ALIGNMENT_SCORE
            )
        reason = ""
        if target_source == "raw_asr_conflict_anchor":
            reason = "raw-high-confidence-anchor-over-whisperx"
        elif target_source == "raw_asr_segment_internal":
            reason = "raw-internal-anchor-between-ctc-and-whisperx"
        elif prefix_recovered:
            reason = "ctc-weak-prefix-recovery-over-whisperx"
        elif crossline_recovered:
            reason = "ctc-crossline-initial-recovery-over-whisperx"
        elif clear_boundary and ctc_score >= 0.08 and abs(disagreement) >= 0.45:
            reason = "ctc-clear-boundary-over-whisperx"
        elif raw_ctc_agree and abs(disagreement) >= 1.0:
            reason = "ctc-raw-consensus-over-whisperx"
        elif ctc_score >= 0.25 and abs(disagreement) >= 1.50:
            reason = "high-evidence-ctc-over-whisperx"
        elif ctc_score >= 0.18 and abs(disagreement) >= 1.10:
            reason = "moderate-evidence-ctc-over-whisperx"
        elif abs(disagreement) >= 2.0:
            reason = "large-whisperx-ctc-disagreement"
        elif previous_same_segment:
            reason = "previous-low-confidence-same-asr-segment"
        elif whisperx_score < 0.85:
            reason = "low-confidence-whisperx-with-ctc-anchor"
        if not reason:
            continue

        fused[index] = target_time
        changes.append(
            {
                "entry": index + 1,
                "from": round(whisperx_time, 3),
                "to": round(target_time, 3),
                "delta": round(disagreement, 3),
                "reason": reason,
                "target_source": target_source,
                "whisperx_score": round(whisperx_score, 3),
                "ctc_score": round(ctc_score, 6),
                "ctc_timestamp": round(ctc_time, 3),
                "ctc_token_spans": ctc_item.get("ctc_token_spans"),
                "ctc_weak_prefix_recovery": prefix_recovered,
                "ctc_crossline_initial_recovery": crossline_recovered,
                "ctc_clear_boundary": clear_boundary,
                "raw_timestamp": round(raw_time, 3) if raw_time is not None else None,
                "raw_score": round(raw_score, 3) if raw_score is not None else None,
                "raw_ctc_agree": raw_ctc_agree,
            }
        )

    if not changes:
        return whisperx_timestamps, whisperx_report, []

    for index in range(1, len(fused)):
        fused[index] = max(fused[index], fused[index - 1] + 0.10)
    for index in range(len(fused)):
        fused[index] = max(0.0, min(duration, fused[index]))

    report = copy.deepcopy(whisperx_report)
    assignments = report.get("assignments")
    if isinstance(assignments, list):
        for index, timestamp in enumerate(fused):
            if index < len(assignments) and isinstance(assignments[index], dict):
                assignments[index]["timestamp"] = round(timestamp, 3)
                for change in changes:
                    if change["entry"] == index + 1:
                        assignments[index]["ctc_local_fusion"] = True
                        assignments[index]["ctc_local_fusion_reason"] = change["reason"]
                        assignments[index]["ctc_score"] = change["ctc_score"]
                        assignments[index]["ctc_token_spans"] = change.get("ctc_token_spans")
                        assignments[index]["ctc_weak_prefix_recovery"] = bool(
                            change.get("ctc_weak_prefix_recovery")
                        )
                        assignments[index]["ctc_crossline_initial_recovery"] = bool(
                            change.get("ctc_crossline_initial_recovery")
                        )
                        assignments[index]["ctc_clear_boundary"] = bool(change.get("ctc_clear_boundary"))
                        break
    report["backend"] = "hybrid"
    report["strategy"] = "auto-candidate-selection-ctc-local-fusion"
    report["ctc_local_fusion_count"] = len(changes)
    report["ctc_local_fusions"] = changes
    report["ctc_local_fusion_source_backend"] = "whisperx+ctc"
    if isinstance(raw_report, dict):
        report["raw_asr_diagnostic_backend"] = raw_report.get("backend")
        report["raw_asr_repeated_tail_after_0s"] = raw_report.get("asr_repeated_tail_after_0s")
    changes_by_entry = {int(change["entry"]): change for change in changes if isinstance(change.get("entry"), int)}
    suspicious = report.get("suspicious_alignments")
    suspicious_items = list(suspicious) if isinstance(suspicious, list) else []
    fused_entries = {int(change["entry"]) for change in changes if isinstance(change.get("entry"), int)}
    timing_consensus: list[dict[str, object]] = []
    if isinstance(assignments, list):
        for index, assignment in enumerate(assignments):
            if not isinstance(assignment, dict) or index >= max_items:
                continue
            whisperx_item = whisperx_assignments[index]
            ctc_item = ctc_assignments[index]
            raw_item = raw_assignments[index] if isinstance(raw_assignments, list) and index < len(raw_assignments) else None
            if not isinstance(whisperx_item, dict) or not isinstance(ctc_item, dict):
                continue
            try:
                selected_time = float(assignment.get("timestamp", fused[index]) or fused[index])
                whisperx_time = float(whisperx_item.get("timestamp", selected_time) or selected_time)
                whisperx_score = float(whisperx_item.get("score", 0.0) or 0.0)
                ctc_time = float(ctc_item.get("timestamp"))
                ctc_score = float(ctc_item.get("ctc_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            raw_time = None
            raw_score = None
            if isinstance(raw_item, dict):
                try:
                    raw_time = float(raw_item.get("timestamp"))
                    raw_score = float(raw_item.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    raw_time = None
                    raw_score = None
            votes: list[tuple[str, float]] = []
            if whisperx_score >= 0.45 and abs(whisperx_time - selected_time) <= 0.75:
                votes.append(("whisperx", whisperx_time))
            if ctc_score >= 0.02 and abs(ctc_time - selected_time) <= 0.75:
                votes.append(("ctc", ctc_time))
            if raw_time is not None and raw_score is not None and raw_score >= 0.45 and abs(raw_time - selected_time) <= 0.75:
                votes.append(("raw", raw_time))
            vote_times = [time for _, time in votes]
            consensus = len(votes) >= 2 and (max(vote_times) - min(vote_times) <= 0.75 if vote_times else False)
            change = changes_by_entry.get(index + 1)
            raw_high_confidence_anchor = (
                change is not None
                and change.get("reason") == "raw-high-confidence-anchor-over-whisperx"
                and raw_score is not None
                and raw_score >= 0.95
                and raw_time is not None
                and abs(raw_time - selected_time) <= 0.05
            )
            if not consensus and not raw_high_confidence_anchor:
                continue
            reason = "multi-backend-time-consensus" if consensus else "high-confidence-raw-internal-anchor"
            assignment["timing_trusted"] = True
            assignment["timing_trusted_reason"] = reason
            assignment["timing_trusted_sources"] = [source for source, _ in votes] if consensus else ["raw"]
            assignment["timing_trusted_candidate_times"] = {
                "selected": round(selected_time, 3),
                "whisperx": round(whisperx_time, 3),
                "ctc": round(ctc_time, 3),
                "raw": round(raw_time, 3) if raw_time is not None else None,
            }
            timing_consensus.append(
                {
                    "entry": index + 1,
                    "reason": reason,
                    "sources": assignment["timing_trusted_sources"],
                    "candidate_times": assignment["timing_trusted_candidate_times"],
                }
            )
            if change is not None:
                change["fusion_trusted"] = True
                change["fusion_trust_reason"] = reason
    if timing_consensus:
        report["timing_consensus_count"] = len(timing_consensus)
        report["timing_consensus"] = timing_consensus
    for change in changes:
        delta = abs(float(change.get("delta", 0.0) or 0.0))
        fusion_trusted = bool(change.get("fusion_trusted"))
        severity = "resolved" if fusion_trusted else ("high" if delta >= 2.0 else "medium")
        flags = ["ctc_local_fusion", str(change["reason"])]
        if fusion_trusted:
            flags.append("fusion_trusted")
        suspicious_items.append(
            {
                "entry": change["entry"],
                "flags": flags,
                "severity": severity,
                "review_required": not fusion_trusted,
                "candidate_timestamps": {
                    "output": change["to"],
                    "whisperx": change["from"],
                    "ctc": change.get("ctc_timestamp"),
                    "raw": change.get("raw_timestamp"),
                },
                "candidate_scores": {
                    "whisperx": change.get("whisperx_score"),
                    "ctc": change.get("ctc_score"),
                    "raw": change.get("raw_score"),
                },
                "ctc_token_spans": change.get("ctc_token_spans"),
                "delta": change["delta"],
                "fusion_trusted": fusion_trusted,
                "fusion_trust_reason": change.get("fusion_trust_reason"),
            }
        )

    if isinstance(raw_assignments, list):
        for index in range(min(len(whisperx_assignments), len(ctc_assignments), len(raw_assignments))):
            entry_number = index + 1
            if entry_number in fused_entries:
                continue
            whisperx_item = whisperx_assignments[index]
            ctc_item = ctc_assignments[index]
            raw_item = raw_assignments[index]
            if not isinstance(whisperx_item, dict) or not isinstance(ctc_item, dict) or not isinstance(raw_item, dict):
                continue
            try:
                whisperx_time = float(whisperx_item.get("timestamp"))
                whisperx_score = float(whisperx_item.get("score", 0.0) or 0.0)
                raw_time = float(raw_item.get("timestamp"))
                raw_score = float(raw_item.get("score", 0.0) or 0.0)
                ctc_time = float(ctc_item.get("timestamp"))
                ctc_score = float(ctc_item.get("ctc_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            raw_whisperx_delta = raw_time - whisperx_time
            if abs(raw_whisperx_delta) < 1.50:
                continue
            if raw_score < 0.82 or whisperx_score < 0.82:
                continue
            if abs(ctc_time - whisperx_time) <= 0.75:
                continue
            ctc_unusable = ctc_score < 0.08 or abs(ctc_time - raw_time) > 1.25
            if not ctc_unusable:
                continue
            suspicious_items.append(
                {
                    "entry": entry_number,
                    "flags": ["unresolved_backend_disagreement", "raw_whisperx_disagreement"],
                    "severity": "high",
                    "review_required": True,
                    "candidate_timestamps": {
                        "whisperx": round(whisperx_time, 3),
                        "raw": round(raw_time, 3),
                        "ctc": round(ctc_time, 3),
                    },
                    "candidate_scores": {
                        "whisperx": round(whisperx_score, 3),
                        "raw": round(raw_score, 3),
                        "ctc": round(ctc_score, 6),
                    },
                    "delta": round(raw_whisperx_delta, 3),
                }
            )
    suspicious_items = dedupe_suspicious_alignments(suspicious_items)
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    review_required_count = 0
    for item in suspicious_items:
        severity = str(item.get("severity", "low"))
        severity_counts[severity if severity in severity_counts else "low"] += 1
        if item.get("review_required"):
            review_required_count += 1
    report["suspicious_alignment_count"] = len(suspicious_items)
    report["suspicious_alignment_severity_counts"] = severity_counts
    report["suspicious_alignments"] = suspicious_items
    report["review_required_count"] = review_required_count
    report["review_required"] = review_required_count > 0
    update_report_confidence_metrics(report)
    return fused, report, changes


def apply_ctc_micro_refinement_to_whisperx(
    whisperx_timestamps: list[float],
    whisperx_report: dict[str, object],
    ctc_report: dict[str, object],
    duration: float,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    whisperx_assignments = whisperx_report.get("assignments")
    ctc_assignments = ctc_report.get("assignments")
    if not isinstance(whisperx_assignments, list) or not isinstance(ctc_assignments, list):
        return whisperx_timestamps, whisperx_report, []

    refined = list(whisperx_timestamps)
    changes: list[dict[str, object]] = []
    max_items = min(len(refined), len(whisperx_assignments), len(ctc_assignments))
    suspicious_entries = {
        int(item.get("entry"))
        for item in whisperx_report.get("suspicious_alignments", [])
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    for index in range(max_items):
        entry_number = index + 1
        if entry_number not in suspicious_entries:
            continue
        whisperx_item = whisperx_assignments[index]
        ctc_item = ctc_assignments[index]
        if not isinstance(whisperx_item, dict) or not isinstance(ctc_item, dict):
            continue
        if whisperx_item.get("ctc_local_fusion"):
            continue
        try:
            whisperx_time = float(whisperx_item.get("timestamp", refined[index]) or refined[index])
            ctc_time = float(ctc_item.get("timestamp"))
            ctc_score = float(ctc_item.get("ctc_score", 0.0) or 0.0)
            whisperx_score = float(whisperx_item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        delta = ctc_time - whisperx_time
        if abs(delta) < 0.05 or abs(delta) > 0.65:
            continue
        if ctc_score < 0.02:
            continue
        if whisperx_score < 0.55:
            continue
        if delta > 0.30 and whisperx_score < 0.85 and ctc_score >= 0.22:
            continue

        previous_time = refined[index - 1] if index > 0 else 0.0
        next_time = refined[index + 1] if index + 1 < len(refined) else duration
        if not (previous_time + 0.10 < ctc_time < next_time - 0.10):
            continue

        refined[index] = ctc_time
        changes.append(
            {
                "entry": entry_number,
                "from": round(whisperx_time, 3),
                "to": round(ctc_time, 3),
                "delta": round(delta, 3),
                "reason": "small-ctc-whisperx-refinement",
                "whisperx_score": round(whisperx_score, 3),
                "ctc_score": round(ctc_score, 6),
                "ctc_token_spans": ctc_item.get("ctc_token_spans"),
            }
        )

    if not changes:
        return whisperx_timestamps, whisperx_report, []

    for index in range(1, len(refined)):
        refined[index] = max(refined[index], refined[index - 1] + 0.10)
    for index in range(len(refined)):
        refined[index] = max(0.0, min(duration, refined[index]))

    report = copy.deepcopy(whisperx_report)
    assignments = report.get("assignments")
    changes_by_entry = {int(change["entry"]): change for change in changes if isinstance(change.get("entry"), int)}
    if isinstance(assignments, list):
        for index, timestamp in enumerate(refined):
            if index >= len(assignments) or not isinstance(assignments[index], dict):
                continue
            assignments[index]["timestamp"] = round(timestamp, 3)
            change = changes_by_entry.get(index + 1)
            if change is None:
                continue
            assignments[index]["ctc_micro_refinement"] = True
            assignments[index]["ctc_micro_refinement_reason"] = change["reason"]
            assignments[index]["ctc_score"] = change["ctc_score"]
            assignments[index]["ctc_token_spans"] = change.get("ctc_token_spans")
            assignments[index]["timing_trusted"] = True
            assignments[index]["timing_trusted_reason"] = "small-ctc-whisperx-consensus"
            assignments[index]["timing_trusted_sources"] = ["whisperx", "ctc"]
            assignments[index]["timing_trusted_candidate_times"] = {
                "selected": round(timestamp, 3),
                "whisperx": change["from"],
                "ctc": change["to"],
                "raw": None,
            }

    suspicious = report.get("suspicious_alignments")
    suspicious_items = list(suspicious) if isinstance(suspicious, list) else []
    for item in suspicious_items:
        if not isinstance(item, dict):
            continue
        entry = item.get("entry")
        if not isinstance(entry, int) or entry not in changes_by_entry:
            continue
        change = changes_by_entry[entry]
        flags = item.get("flags")
        if not isinstance(flags, list):
            flags = []
        item["flags"] = list(dict.fromkeys([*flags, "ctc_micro_refinement", "timing_trusted"]))
        item["severity"] = "resolved"
        item["review_required"] = False
        candidates = item.get("candidate_timestamps")
        if not isinstance(candidates, dict):
            candidates = {}
        candidates["output"] = change["to"]
        candidates["whisperx"] = change["from"]
        candidates["ctc"] = change["to"]
        item["candidate_timestamps"] = candidates
        item["ctc_token_spans"] = change.get("ctc_token_spans")
    report["suspicious_alignments"] = suspicious_items
    report["ctc_micro_refinement_count"] = len(changes)
    report["ctc_micro_refinements"] = changes
    report["ctc_micro_refinement_source_backend"] = "whisperx+ctc"
    severity_counts = {"high": 0, "medium": 0, "low": 0, "resolved": 0}
    review_required_count = 0
    for item in suspicious_items:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "low"))
        severity_counts[severity if severity in severity_counts else "low"] += 1
        if item.get("review_required"):
            review_required_count += 1
    report["suspicious_alignment_severity_counts"] = severity_counts
    report["review_required_count"] = review_required_count
    report["review_required"] = review_required_count > 0
    update_report_confidence_metrics(report)
    return refined, report, changes


def apply_whisperx_acoustic_boundary_refinement(
    audio_path: Path,
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps):
        return timestamps, report, []
    suspicious_items = [
        item for item in report.get("suspicious_alignments", []) if isinstance(item, dict)
    ]
    suspicious_by_entry = {
        int(item.get("entry")): item
        for item in suspicious_items
        if isinstance(item.get("entry"), int)
    }
    try:
        features = analyze_audio(audio_path, duration)
    except LrcError:
        return timestamps, report, []

    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            continue
        if assignment.get("ctc_micro_refinement") or assignment.get("ctc_local_fusion"):
            continue
        try:
            score = float(assignment.get("score", 0.0) or 0.0)
            current_time = float(assignment.get("timestamp", refined[index]) or refined[index])
        except (TypeError, ValueError):
            continue
        entry_number = index + 1
        risk = suspicious_by_entry.get(entry_number)
        flags = risk.get("flags") if isinstance(risk, dict) else []
        flags = flags if isinstance(flags, list) else []
        previous_time = refined[index - 1] if index > 0 else 0.0
        next_time = refined[index + 1] if index + 1 < len(refined) else duration

        candidate: float | None = None
        reason = ""
        if index == 0 and current_time <= 1.0:
            candidate = first_local_onset(features, 0.20, max(0.21, current_time - 0.10), threshold=0.65)
            if candidate is not None and not (0.12 <= current_time - candidate <= 0.35):
                candidate = None
            reason = "first-line-acoustic-onset"
        elif score >= 0.95 and flags == ["candidate_disagreement"]:
            candidate = first_local_onset(
                features,
                max(previous_time + 0.10, current_time - 0.85),
                current_time - 0.12,
                threshold=0.65,
            )
            if candidate is not None and not (0.45 <= current_time - candidate <= 0.65):
                candidate = None
            reason = "candidate-disagreement-acoustic-backtrack"
        elif (
            score >= 0.95
            and "close_neighbor_onset_uncertain" in flags
            and len(normalize_match_text(str(risk.get("text", "")) if isinstance(risk, dict) else "")) <= 4
        ):
            candidate = first_local_onset(
                features,
                max(previous_time + 0.10, current_time - 0.85),
                current_time - 0.12,
                threshold=0.65,
            )
            if candidate is not None and not (0.45 <= current_time - candidate <= 0.65):
                candidate = None
            reason = "short-line-acoustic-backtrack"
        elif (
            score >= 0.95
            and "close_neighbor_onset_uncertain" in flags
            and "candidate_disagreement" not in flags
            and current_time - previous_time >= 2.50
            and next_time - current_time >= 2.50
        ):
            candidate = first_local_onset(
                features,
                current_time + 0.20,
                min(next_time - 0.10, current_time + 0.70),
                threshold=0.65,
            )
            if candidate is not None and not (0.35 <= candidate - current_time <= 0.60):
                candidate = None
            reason = "lead-in-acoustic-forward-onset"
        elif (
            score >= 0.95
            and "follows_suspicious_long_line" in flags
            and "possible_bad_split_boundary" in flags
            and next_time - current_time <= 2.20
        ):
            candidate = first_local_onset(
                features,
                current_time + 0.10,
                min(next_time - 0.10, current_time + 0.85),
                threshold=0.65,
            )
            if candidate is not None and not (0.25 <= candidate - current_time <= 0.45):
                candidate = None
            reason = "bad-split-acoustic-forward-onset"
        if candidate is None:
            continue
        if not (previous_time + 0.05 < candidate < next_time - 0.05):
            continue

        refined[index] = candidate
        assignment["timestamp"] = round(candidate, 3)
        assignment["acoustic_boundary_refinement"] = True
        assignment["acoustic_boundary_refinement_reason"] = reason
        assignment["timing_trusted"] = True
        assignment["timing_trusted_reason"] = reason
        assignment["timing_trusted_sources"] = ["acoustic"]
        assignment["timing_trusted_candidate_times"] = {
            "selected": round(candidate, 3),
            "whisperx": round(current_time, 3),
            "ctc": assignment.get("ctc_timestamp"),
            "raw": None,
        }
        changes.append(
            {
                "entry": entry_number,
                "from": round(current_time, 3),
                "to": round(candidate, 3),
                "delta": round(candidate - current_time, 3),
                "reason": reason,
                "score": round(score, 3),
            }
        )
        if risk is not None:
            risk_flags = risk.get("flags")
            if not isinstance(risk_flags, list):
                risk_flags = []
            risk["flags"] = list(dict.fromkeys([*risk_flags, "acoustic_boundary_refinement", "timing_trusted"]))
            risk["severity"] = "resolved"
            risk["review_required"] = False
            candidates = risk.get("candidate_timestamps")
            if not isinstance(candidates, dict):
                candidates = {}
            candidates["output"] = round(candidate, 3)
            candidates["whisperx"] = round(current_time, 3)
            risk["candidate_timestamps"] = candidates

    if not changes:
        return timestamps, report, []
    report["suspicious_alignments"] = suspicious_items
    report["acoustic_boundary_refinement_count"] = len(changes)
    report["acoustic_boundary_refinements"] = changes
    severity_counts = {"high": 0, "medium": 0, "low": 0, "resolved": 0}
    review_required_count = 0
    for item in suspicious_items:
        severity = str(item.get("severity", "low"))
        severity_counts[severity if severity in severity_counts else "low"] += 1
        if item.get("review_required"):
            review_required_count += 1
    report["suspicious_alignment_severity_counts"] = severity_counts
    report["review_required_count"] = review_required_count
    report["review_required"] = review_required_count > 0
    update_report_confidence_metrics(report)
    return refined, report, changes


def vocal_onset_features(
    audio_path: Path, duration: float, args: argparse.Namespace
) -> tuple[AudioFeatures | None, str | None]:
    if not args.vocal_onset_refine:
        return None, "disabled"
    python_exe = default_ctc_python()
    if not python_exe.exists():
        return None, "missing-asr-python"
    with tempfile.TemporaryDirectory(prefix="lrc-vocals-") as temp_name:
        temp_dir = Path(temp_name)
        command = [
            str(python_exe),
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "--mp3",
            "-d",
            "cuda",
            "-o",
            str(temp_dir),
            str(audio_path),
        ]
        proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().replace("\n", " ")
            return None, f"demucs-failed: {detail[:240]}"
        stems = list(temp_dir.rglob("vocals.mp3"))
        if not stems:
            return None, "demucs-did-not-write-vocals"
        return analyze_vocal_onsets(stems[0], duration), None


def prepare_vocal_ctc_audio(audio_path: Path, args: argparse.Namespace) -> tuple[Path, str, str | None]:
    """Build or reuse the isolated vocal track used by primary CTC alignment."""
    if not getattr(args, "vocal_ctc", True):
        return audio_path, "mix", "disabled"
    python_exe = default_ctc_python()
    if not python_exe.exists():
        return audio_path, "mix", "missing-asr-python"
    try:
        stat = audio_path.stat()
    except OSError:
        return audio_path, "mix", "audio-stat-failed"
    key = hashlib.sha1(f"{audio_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")).hexdigest()[:16]
    cache_dir = DEFAULT_VOCAL_CACHE_DIR / key
    cached = cache_dir / "vocals.mp3"
    if cached.exists() and cached.stat().st_size > 0:
        return cached, "vocal-stem", "cache-hit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            str(python_exe), "-m", "demucs", "--two-stems", "vocals", "--mp3", "-d", "cuda",
            "-o", str(cache_dir), str(audio_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip().replace("\n", " ")
        return audio_path, "mix", f"demucs-failed: {detail[:180]}"
    stems = list(cache_dir.rglob("vocals.mp3"))
    if not stems:
        return audio_path, "mix", "demucs-did-not-write-vocals"
    stem = stems[0]
    if stem != cached:
        try:
            stem.replace(cached)
        except OSError:
            cached = stem
    return cached, "vocal-stem", "generated"


def nearest_onset_evidence(features: AudioFeatures, timestamp: float) -> tuple[float, float] | None:
    mask = (features.frame_times >= timestamp - 0.35) & (features.frame_times <= timestamp + 0.35)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    peak = max(indices, key=lambda index: float(features.onset_strength[index]))
    return float(features.frame_times[peak]), float(features.onset_strength[peak])


def apply_vocal_onset_tiebreak(
    timestamps: list[float],
    report: dict[str, object],
    ctc_report: dict[str, object],
    audio_path: Path,
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    features, unavailable = vocal_onset_features(audio_path, duration, args)
    if features is None:
        report["vocal_onset_refinement"] = {"status": "skipped", "reason": unavailable}
        return timestamps, report, []
    assignments = report.get("assignments")
    ctc_assignments = ctc_report.get("assignments")
    if not isinstance(assignments, list) or not isinstance(ctc_assignments, list):
        return timestamps, report, []
    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    for index, current_time in enumerate(timestamps):
        if index >= len(assignments) or index >= len(ctc_assignments):
            continue
        assignment = assignments[index]
        ctc_assignment = ctc_assignments[index]
        if not isinstance(assignment, dict) or not isinstance(ctc_assignment, dict):
            continue
        ctc_time = ctc_assignment.get("timestamp")
        if not isinstance(ctc_time, (int, float)):
            continue
        disagreement = abs(float(ctc_time) - current_time)
        if not 0.40 <= disagreement <= 2.0:
            continue
        current_evidence = nearest_onset_evidence(features, current_time)
        ctc_evidence = nearest_onset_evidence(features, float(ctc_time))
        if current_evidence is None or ctc_evidence is None:
            continue
        current_peak, current_strength = current_evidence
        ctc_peak, ctc_strength = ctc_evidence
        if abs(ctc_peak - float(ctc_time)) > 0.18 or abs(current_peak - current_time) < 0.16:
            continue
        if ctc_strength < 0.50 or ctc_strength < current_strength + 0.15:
            continue
        previous_time = refined[index - 1] if index else 0.0
        next_time = timestamps[index + 1] if index + 1 < len(timestamps) else duration
        if not previous_time + 0.10 < ctc_peak < next_time - 0.10:
            continue
        refined[index] = ctc_peak
        assignment["timestamp"] = round(ctc_peak, 3)
        assignment["vocal_onset_tiebreak"] = True
        assignment["timing_repair_source"] = "ctc+demucs-vocal-onset"
        changes.append(
            {
                "entry": index + 1,
                "from": round(current_time, 3),
                "to": round(ctc_peak, 3),
                "ctc_candidate": round(float(ctc_time), 3),
                "vocal_onset_strength": round(ctc_strength, 3),
                "competing_onset_strength": round(current_strength, 3),
                "reason": "demucs-vocal-onset-prefers-ctc",
            }
        )
    report["vocal_onset_refinement"] = {
        "status": "applied",
        "backend": "demucs-htdemucs",
        "change_count": len(changes),
        "changes": changes,
    }
    return refined, report, changes


def flag_unresolved_ctc_disagreements(
    timestamps: list[float], report: dict[str, object], ctc_report: dict[str, object]
) -> None:
    """Do not call a fused timestamp trusted while a CTC alternative remains far away."""
    assignments = report.get("assignments")
    ctc_assignments = ctc_report.get("assignments")
    if not isinstance(assignments, list) or not isinstance(ctc_assignments, list):
        return
    suspicious = report.get("suspicious_alignments")
    if not isinstance(suspicious, list):
        suspicious = []
        report["suspicious_alignments"] = suspicious
    by_entry = {
        item.get("entry"): item
        for item in suspicious
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    for index, current_time in enumerate(timestamps):
        if index >= len(assignments) or index >= len(ctc_assignments):
            continue
        assignment = assignments[index]
        ctc_assignment = ctc_assignments[index]
        if not isinstance(assignment, dict) or not isinstance(ctc_assignment, dict):
            continue
        ctc_time = ctc_assignment.get("timestamp")
        if not isinstance(ctc_time, (int, float)):
            continue
        disagreement = abs(float(ctc_time) - current_time)
        if disagreement < 0.35 or assignment.get("vocal_onset_tiebreak"):
            continue
        entry_number = index + 1
        assignment["unresolved_ctc_disagreement"] = round(disagreement, 3)
        existing = by_entry.get(entry_number)
        if existing is not None:
            existing["review_required"] = True
            existing["severity"] = "high" if disagreement >= 1.0 else "medium"
            flags = existing.get("flags")
            if not isinstance(flags, list):
                flags = []
                existing["flags"] = flags
            if "unresolved_ctc_disagreement" not in flags:
                flags.append("unresolved_ctc_disagreement")
            candidates = existing.get("candidate_timestamps")
            if not isinstance(candidates, dict):
                candidates = {}
                existing["candidate_timestamps"] = candidates
            candidates["output"] = round(current_time, 3)
            candidates["ctc"] = round(float(ctc_time), 3)
            existing["candidate_disagreement_seconds"] = round(disagreement, 3)
            continue
        suspicious.append(
            {
                "entry": entry_number,
                "lyric": assignment.get("lyric", ""),
                "review_required": True,
                "severity": "high" if disagreement >= 1.0 else "medium",
                "flags": ["unresolved_ctc_disagreement"],
                "candidate_timestamps": {
                    "output": round(current_time, 3),
                    "ctc": round(float(ctc_time), 3),
                },
                "candidate_disagreement_seconds": round(disagreement, 3),
            }
        )
    review_count = sum(
        1 for item in suspicious if isinstance(item, dict) and bool(item.get("review_required"))
    )
    report["review_required_count"] = review_count
    report["review_required"] = review_count > 0
    update_report_confidence_metrics(report)


def flag_unresolved_raw_ctc_disagreements(
    timestamps: list[float], report: dict[str, object], raw_report: dict[str, object]
) -> None:
    """Require review when known-lyric CTC and raw ASR disagree on a line onset."""
    assignments = report.get("assignments")
    raw_assignments = raw_report.get("assignments")
    if not isinstance(assignments, list) or not isinstance(raw_assignments, list):
        return
    suspicious = report.get("suspicious_alignments")
    if not isinstance(suspicious, list):
        suspicious = []
        report["suspicious_alignments"] = suspicious
    by_entry = {
        item.get("entry"): item
        for item in suspicious
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    for index, current_time in enumerate(timestamps):
        if index >= len(assignments) or index >= len(raw_assignments):
            continue
        assignment = assignments[index]
        raw_assignment = raw_assignments[index]
        if not isinstance(assignment, dict) or not isinstance(raw_assignment, dict):
            continue
        try:
            raw_time = float(raw_assignment.get("timestamp"))
            raw_score = float(raw_assignment.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        borrowed = bool(raw_assignment.get("borrowed"))
        disagreement = abs(raw_time - current_time)
        threshold = 0.45 if raw_score >= 0.90 and not borrowed else 0.75
        if disagreement < threshold:
            continue
        entry_number = index + 1
        assignment["unresolved_raw_ctc_disagreement"] = round(disagreement, 3)
        severity = "high" if raw_score >= 0.90 or disagreement >= 1.50 else "medium"
        existing = by_entry.get(entry_number)
        if existing is None:
            existing = {
                "entry": entry_number,
                "review_required": True,
                "severity": severity,
                "flags": [],
                "candidate_timestamps": {},
            }
            suspicious.append(existing)
            by_entry[entry_number] = existing
        existing["review_required"] = True
        existing["severity"] = severity
        flags = existing.get("flags")
        if not isinstance(flags, list):
            flags = []
            existing["flags"] = flags
        if "unresolved_raw_ctc_disagreement" not in flags:
            flags.append("unresolved_raw_ctc_disagreement")
        candidates = existing.get("candidate_timestamps")
        if not isinstance(candidates, dict):
            candidates = {}
            existing["candidate_timestamps"] = candidates
        candidates["output"] = round(current_time, 3)
        candidates["raw_asr"] = round(raw_time, 3)
        existing["raw_asr_score"] = round(raw_score, 3)
        existing["raw_asr_borrowed"] = borrowed
        existing["candidate_disagreement_seconds"] = round(disagreement, 3)
    review_count = sum(
        1 for item in suspicious if isinstance(item, dict) and bool(item.get("review_required"))
    )
    report["review_required_count"] = review_count
    report["review_required"] = review_count > 0
    update_report_confidence_metrics(report)


def run_auto_backend_competition(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object], str]:
    candidates: list[dict[str, object]] = []
    diagnostic_candidates: list[dict[str, object]] = []
    raw_diagnostic_report: dict[str, object] | None = None
    raw_diagnostic_timestamps: list[float] | None = None

    if default_ctc_ready():
        ctc_timestamps, ctc_report, ctc_error = try_ctc_candidate(audio_path, entries, duration, args)
        if not ctc_error and isinstance(ctc_report, dict):
            ctc_timestamps, ctc_report, _ = apply_ctc_crossline_initial_recovery(
                ctc_timestamps, ctc_report, duration
            )
            ctc_timestamps, ctc_report, _ = apply_ctc_weak_prefix_recovery(
                ctc_timestamps, ctc_report, duration
            )
        candidates.append(
            {
                "backend": "ctc",
                "timestamps": ctc_timestamps,
                "report": ctc_report,
                "error": ctc_error,
            }
        )

    if default_whisperx_ready():
        raw_timestamps, raw_report, raw_error = try_whispercpp_diagnostic_candidate(audio_path, entries, duration, args)
        if not raw_error and isinstance(raw_report, dict):
            raw_diagnostic_report = raw_report
            raw_diagnostic_timestamps = raw_timestamps
        diagnostic_candidates.append(
            {
                "backend": "whispercpp_raw",
                "timestamps": raw_timestamps,
                "report": raw_report,
                "error": raw_error,
            }
        )
        ctc_candidate = next(
            (candidate for candidate in candidates if candidate.get("backend") == "ctc" and not candidate.get("error")),
            None,
        )
        if (
            ctc_candidate is not None
            and not raw_error
            and isinstance(ctc_candidate.get("report"), dict)
            and isinstance(raw_report, dict)
            and isinstance(ctc_candidate.get("timestamps"), list)
        ):
            window_timestamps, window_report, _ = apply_ctc_local_window_realign(
                ctc_alignment_audio_path(ctc_candidate["report"], audio_path),
                entries,
                list(ctc_candidate["timestamps"]),
                ctc_candidate["report"],
                raw_report,
                duration,
                args,
            )
            window_timestamps, window_report, _ = apply_ctc_crossline_initial_recovery(
                window_timestamps, window_report, duration
            )
            window_timestamps, window_report, _ = apply_ctc_weak_prefix_recovery(
                window_timestamps, window_report, duration
            )
            ctc_candidate["timestamps"] = window_timestamps
            ctc_candidate["report"] = window_report
            # Score CTC only after raw/CTC disagreement diagnostics. Otherwise
            # a clean-looking forced path can beat stronger raw ASR evidence.
            flag_unresolved_raw_ctc_disagreements(window_timestamps, window_report, raw_report)
            raw_quality = candidate_selection_quality(raw_report)
            ctc_quality = candidate_selection_quality(window_report)
            if (
                raw_asr_is_fallback_eligible(raw_report)
                and raw_quality >= 65.0
                and raw_quality > ctc_quality
                and raw_diagnostic_timestamps is not None
            ):
                candidates.append(
                    {
                        "backend": "whispercpp",
                        "timestamps": raw_diagnostic_timestamps,
                        "report": raw_report,
                        "error": None,
                        "fallback_reason": "raw-asr-beats-conflicted-ctc",
                    }
                )

    if default_whisperx_ready():
        try:
            whisperx_timestamps, whisperx_report = run_whisperx_best_candidate(audio_path, entries, duration, args)
            whisperx_error = None
        except LrcError as exc:
            whisperx_timestamps = []
            whisperx_report = {"backend": "whisperx", "error": str(exc)}
            whisperx_error = str(exc)
        candidates.append(
            {
                "backend": "whisperx",
                "timestamps": whisperx_timestamps,
                "report": whisperx_report,
                "error": whisperx_error,
            }
        )

    selected = choose_alignment_candidate(candidates)
    selected_report = copy.deepcopy(selected["report"])
    assert isinstance(selected_report, dict)
    selected_timestamps = list(selected["timestamps"])
    selected_backend = str(selected.get("backend", selected_report.get("backend", "auto")))
    selected_quality = float(selected.get("quality", candidate_selection_quality(selected_report)) or -100.0)
    base_selected_report = copy.deepcopy(selected_report)
    base_selected_timestamps = list(selected_timestamps)
    base_selected_backend = selected_backend
    base_selected_quality = selected_quality
    ctc_candidate: dict[str, object] | None = None

    if selected_backend == "whisperx":
        ctc_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("backend") == "ctc"
                and not candidate.get("error")
                and isinstance(candidate.get("report"), dict)
            ),
            None,
        )
        if ctc_candidate is not None:
            raw_candidate = next(
                (
                    candidate
                    for candidate in diagnostic_candidates
                    if candidate.get("backend") == "whispercpp_raw"
                    and not candidate.get("error")
                    and isinstance(candidate.get("report"), dict)
                ),
                None,
            )
            ctc_window_timestamps, ctc_window_report, _ = apply_ctc_hybrid_window_realign(
                ctc_alignment_audio_path(ctc_candidate["report"], audio_path),
                entries,
                list(ctc_candidate["timestamps"]),
                ctc_candidate["report"],
                selected_report,
                duration,
                args,
            )
            ctc_window_timestamps, ctc_window_report, _ = apply_ctc_crossline_initial_recovery(
                ctc_window_timestamps, ctc_window_report, duration
            )
            ctc_window_timestamps, ctc_window_report, _ = apply_ctc_weak_prefix_recovery(
                ctc_window_timestamps, ctc_window_report, duration
            )
            ctc_candidate["timestamps"] = ctc_window_timestamps
            ctc_candidate["report"] = ctc_window_report
            fused_timestamps, fused_report, fusion_changes = apply_ctc_local_fusion_to_whisperx(
                selected_timestamps,
                selected_report,
                ctc_candidate["report"],
                duration,
                raw_candidate["report"] if raw_candidate is not None else None,
            )
            if fusion_changes:
                selected_timestamps = fused_timestamps
                selected_report = fused_report
                selected_backend = "hybrid"
                selected_quality = candidate_selection_quality(selected_report)
            already_clean = (
                float(selected_report.get("trusted_percent", selected_report.get("matched_percent", 0.0)) or 0.0)
                >= 100.0
                and int(selected_report.get("review_required_count", 0) or 0) == 0
            )
            if not already_clean:
                micro_timestamps, micro_report, micro_changes = apply_ctc_micro_refinement_to_whisperx(
                    selected_timestamps,
                    selected_report,
                    ctc_candidate["report"],
                    duration,
                )
                if micro_changes:
                    selected_timestamps = micro_timestamps
                    selected_report = micro_report
                    selected_backend = "hybrid"
                    selected_quality = candidate_selection_quality(selected_report)
                acoustic_timestamps, acoustic_report, acoustic_changes = apply_whisperx_acoustic_boundary_refinement(
                    audio_path,
                    selected_timestamps,
                    selected_report,
                    duration,
                )
                if acoustic_changes:
                    selected_timestamps = acoustic_timestamps
                    selected_report = acoustic_report
                    selected_backend = "hybrid"
                    selected_quality = candidate_selection_quality(selected_report)
    if ctc_candidate is not None and selected_backend in {"whisperx", "hybrid"}:
        ctc_report = ctc_candidate.get("report")
        if isinstance(ctc_report, dict):
            vocal_timestamps, vocal_report, vocal_changes = apply_vocal_onset_tiebreak(
                selected_timestamps,
                selected_report,
                ctc_report,
                audio_path,
                duration,
                args,
            )
            if vocal_changes:
                selected_timestamps = vocal_timestamps
                selected_report = vocal_report
                selected_backend = "hybrid"
                selected_quality = candidate_selection_quality(selected_report)
            flag_unresolved_ctc_disagreements(selected_timestamps, selected_report, ctc_report)
    if selected_backend == "ctc" and raw_diagnostic_report is not None:
        flag_unresolved_raw_ctc_disagreements(selected_timestamps, selected_report, raw_diagnostic_report)
    if ctc_candidate is not None and selected_backend == "hybrid":
        ctc_report = ctc_candidate.get("report")
        if isinstance(ctc_report, dict) and should_prefer_ctc_over_review_exploded_hybrid(selected_report, ctc_report):
            hybrid_reviews = int(selected_report.get("review_required_count", 0) or 0)
            ctc_reviews = int(ctc_report.get("review_required_count", 0) or 0)
            selected_report = copy.deepcopy(ctc_report)
            selected_timestamps = list(ctc_candidate["timestamps"])
            selected_backend = "ctc"
            selected_quality = candidate_selection_quality(selected_report)
            selected_report["rejected_hybrid_refinement"] = {
                "reason": "hybrid-review-explosion-prefers-ctc",
                "hybrid_review_required_count": hybrid_reviews,
                "ctc_review_required_count": ctc_reviews,
            }
    strong_local_correction = bool(
        selected_report.get("ctc_local_fusions")
        and any(
            isinstance(change, dict)
            and change.get("reason")
            in {
                "high-evidence-ctc-over-whisperx",
                "moderate-evidence-ctc-over-whisperx",
                "ctc-clear-boundary-over-whisperx",
                "ctc-crossline-initial-recovery-over-whisperx",
                "large-whisperx-ctc-disagreement",
                "ctc-weak-prefix-recovery-over-whisperx",
            }
            for change in selected_report.get("ctc_local_fusions", [])
        )
    )
    if selected_backend == "hybrid" and selected_quality + 0.001 < base_selected_quality and not strong_local_correction:
        selected_report = base_selected_report
        selected_timestamps = base_selected_timestamps
        selected_backend = base_selected_backend
        selected_report["rejected_hybrid_refinement"] = {
            "reason": "refinement-quality-regression",
            "baseline_backend": base_selected_backend,
            "baseline_quality": round(base_selected_quality, 3),
            "refined_quality": round(selected_quality, 3),
        }
        selected_quality = base_selected_quality
    elif selected_backend == "hybrid" and selected_quality + 0.001 < base_selected_quality:
        selected_report["retained_hybrid_refinement"] = {
            "reason": "strong-local-ctc-evidence-overrides-global-quality-regression",
            "baseline_quality": round(base_selected_quality, 3),
            "refined_quality": round(selected_quality, 3),
        }
    selected_report["backend"] = selected_backend
    selected_reason = candidate_selection_reason(selected, candidates)
    if selected_backend == "hybrid":
        refinement_count = int(selected_report.get("ctc_micro_refinement_count", 0) or 0)
        acoustic_count = int(selected_report.get("acoustic_boundary_refinement_count", 0) or 0)
        refinement_note = (
            f" and {refinement_count} small CTC refinement(s)"
            if refinement_count
            else ""
        )
        acoustic_note = (
            f" and {acoustic_count} acoustic boundary refinement(s)"
            if acoustic_count
            else ""
        )
        selected_reason = (
            f"selected hybrid quality={selected_quality:.1f} after local CTC/raw fusion"
            f"{refinement_note}{acoustic_note}; "
            f"base selection: {selected_reason}"
        )
    selected_report["candidate_selection"] = {
        "status": "selected",
        "selected_backend": selected_backend,
        "selected_quality": round(selected_quality, 3),
        "selected_reason": selected_reason,
        "tie_policy": "prefer_ctc_when_quality_is_within_5_points_and_ctc_quality_is_at_least_80",
        "post_fusion_selected_summary": candidate_summary(selected_report)
        if selected_backend == "hybrid"
        else None,
        "candidates": [
            candidate_summary(candidate["report"], str(candidate.get("error")) if candidate.get("error") else None)
            for candidate in candidates
            if isinstance(candidate.get("report"), dict)
        ],
        "diagnostic_candidates": [
            candidate_summary(candidate["report"], str(candidate.get("error")) if candidate.get("error") else None)
            for candidate in diagnostic_candidates
            if isinstance(candidate.get("report"), dict)
        ],
    }
    return selected_timestamps, selected_report, selected_backend


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
                segment_index = previous_index
                current_text = previous_text
                break
        if segment_index is None or segment_index <= 0:
            continue
        for previous_index in range(segment_index - 1, max(-1, segment_index - 4), -1):
            previous_text = segment_texts[previous_index]
            if not previous_text:
                continue
            if segments[segment_index].start - segments[previous_index].start > 5.0:
                continue
            if sequence_ratio(previous_text, current_text) < 0.86:
                continue
            if not entry_text.startswith(previous_text):
                continue
            assignments[entry_index] = previous_index
            scores[entry_index] = max(scores[entry_index], 0.80)
            break

    timestamps: list[float | None] = [None] * len(entries)
    for index, segment_index in enumerate(assignments):
        if segment_index is not None:
            timestamp = segments[segment_index].start
            char_time = segment_char_time(entry_texts[index], segments[segment_index])
            if char_time is not None:
                timestamp = char_time
            elif not segments[segment_index].chars:
                raw_entry_text = entries[index].lines[0] if entries[index].lines else entries[index].alignment_text
                estimated_time = estimated_segment_text_time(entry_texts[index], raw_entry_text, segments[segment_index])
                if estimated_time is not None:
                    timestamp = estimated_time
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
        current_text = segment_texts[segment_index]
        entry_text = normalize_match_text(entries[index].lines[0] if entries[index].lines else entries[index].alignment_text)
        direct_segment_match = (
            bool(entry_text)
            and (
                entry_text in current_text
                or current_text in entry_text
                or lyric_segment_score(entry_text, current_text) >= 0.82
            )
        )
        if (
            gap_after_segment < 6.0
            or next_time - final[index] < 8.0
            or gap_after_previous < 4.5
            or current_duration < 2.5
            or direct_segment_match
        ):
            continue
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

    assigned = sum(1 for assignment in assignments if assignment is not None)
    trusted = sum(
        1
        for assignment, score in zip(assignments, scores)
        if assignment is not None and score >= TRUSTED_ALIGNMENT_SCORE
    )
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
    raw_review_entries = [
        {
            "entry": index + 1,
            "text": entry_sung_text(entries[index]),
            "flags": [
                "raw_asr_missing_alignment" if assignments[index] is None else "raw_asr_low_match_score",
                *( ["raw_asr_borrowed_segment"] if borrowed[index] else [] ),
            ],
            "severity": "high" if assignments[index] is None or scores[index] <= 0.0 else "medium",
            "review_required": True,
        }
        for index in range(len(entries))
        if assignments[index] is None or scores[index] < TRUSTED_ALIGNMENT_SCORE or borrowed[index]
    ]
    report: dict[str, object] = {
        "backend": "whispercpp",
        "asr_segments": len(segments),
        "timing_entries": len(entries),
        "assigned_entries": assigned,
        "assigned_percent": ratio_percent(assigned, len(entries)),
        "trusted_entries": trusted,
        "trusted_percent": ratio_percent(trusted, len(entries)),
        "matched_entries": trusted,
        "matched_percent": ratio_percent(trusted, len(entries)),
        "low_confidence_entries": low_confidence,
        "low_confidence_count": len(low_confidence),
        "low_confidence_percent": ratio_percent(len(low_confidence), len(entries)),
        "review_required_count": len(raw_review_entries),
        "review_required_percent": ratio_percent(len(raw_review_entries), len(entries)),
        "review_required": bool(raw_review_entries),
        "suspicious_alignment_count": len(raw_review_entries),
        "suspicious_alignment_severity_counts": {
            "high": sum(1 for item in raw_review_entries if item["severity"] == "high"),
            "medium": sum(1 for item in raw_review_entries if item["severity"] == "medium"),
            "low": 0,
        },
        "suspicious_alignments": raw_review_entries,
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


def first_local_onset(features: AudioFeatures, start: float, end: float, threshold: float = 0.55) -> float | None:
    if end <= start:
        return None
    effective_start = start + 0.20
    indices = np.flatnonzero((features.frame_times >= effective_start) & (features.frame_times <= end))
    for index in indices:
        strength = float(features.onset_strength[index])
        if strength < threshold:
            continue
        left = float(features.onset_strength[index - 1]) if index > 0 else 0.0
        right = float(features.onset_strength[index + 1]) if index + 1 < len(features.onset_strength) else 0.0
        if strength >= left and strength >= right:
            plateau_end = index
            while (
                plateau_end + 1 < len(features.onset_strength)
                and float(features.frame_times[plateau_end + 1]) <= end
                and abs(float(features.onset_strength[plateau_end + 1]) - strength) <= 1e-6
            ):
                plateau_end += 1
            if plateau_end > index:
                return float((features.frame_times[index] + features.frame_times[plateau_end]) / 2.0)
            return float(features.frame_times[index])
    return None


def should_accept_zero_gap_boundary_realign(
    boundary_gap: float,
    current_score: float,
    current_time: float,
    local_time: float,
    local_score: float,
) -> bool:
    """Accept a bounded re-align only when it repairs a collapsed line boundary."""
    return (
        boundary_gap <= 0.04
        and local_time - current_time >= 0.30
        and local_score >= max(0.05, current_score * 1.05)
    )


def apply_ctc_zero_gap_boundary_realign(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Re-align a collapsed CTC line boundary inside a compact lyric window."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return []
    python_exe = default_ctc_python()
    helper = Path(__file__).resolve().parent / "ctc_align.py"
    changes: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="lrc-ctc-boundary-") as temp_name:
        temp_dir = Path(temp_name)
        for index in range(1, len(entries)):
            previous = assignments[index - 1]
            current = assignments[index]
            if not isinstance(previous, dict) or not isinstance(current, dict):
                continue
            previous_spans = previous.get("ctc_token_spans")
            current_spans = current.get("ctc_token_spans")
            if not isinstance(previous_spans, list) or not previous_spans or not isinstance(current_spans, list) or not current_spans:
                continue
            try:
                previous_end = max(float(span.get("end")) for span in previous_spans if isinstance(span, dict))
                current_time = float(timestamps[index])
                current_score = float(current.get("ctc_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            boundary_gap = current_time - previous_end
            if boundary_gap > 0.04:
                continue
            start_index = index - 1
            end_index = min(len(entries), index + 2)
            start_time = max(0.0, timestamps[start_index] - 0.50)
            end_time = min(duration, (timestamps[index + 1] + 0.75) if index + 1 < len(timestamps) else current_time + 5.0)
            if not 1.0 <= end_time - start_time <= 16.0:
                continue
            transcript_path = temp_dir / f"boundary-{index + 1}.json"
            output_path = temp_dir / f"boundary-{index + 1}.result.json"
            transcript_path.write_text(
                json.dumps([entry_sung_text(entry) for entry in entries[start_index:end_index]], ensure_ascii=False),
                encoding="utf-8",
            )
            command = [
                str(python_exe), str(helper), "--audio", str(audio_path), "--transcript", str(transcript_path),
                "--output", str(output_path), "--device", args.whisperx_device,
                "--start", f"{start_time:.3f}", "--end", f"{end_time:.3f}",
            ]
            proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0 or not output_path.exists():
                continue
            rows = json.loads(output_path.read_text(encoding="utf-8")).get("entries")
            if not isinstance(rows, list) or len(rows) < 2 or not isinstance(rows[1], dict):
                continue
            row = rows[1]
            try:
                local_time = float(row.get("start"))
                local_score = float(row.get("ctc_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if not should_accept_zero_gap_boundary_realign(
                boundary_gap, current_score, current_time, local_time, local_score
            ):
                continue
            timestamps[index] = local_time
            current["timestamp"] = round(local_time, 3)
            current["ctc_score"] = row.get("ctc_score")
            current["ctc_token_spans"] = row.get("token_spans")
            current["ctc_first_token_candidates"] = row.get("first_token_candidates")
            current["ctc_zero_gap_boundary_realign"] = True
            current["timing_repair_source"] = "torchaudio-mms-fa+zero-gap-boundary-realign"
            changes.append({"entry": index + 1, "from": round(current_time, 3), "to": round(local_time, 3)})
    if changes:
        annotate_ctc_boundary_evidence(assignments)
        report["ctc_zero_gap_boundary_realign"] = changes
    return changes


def apply_ctc_acoustic_backtrack(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> list[dict[str, object]]:
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps):
        return []
    if len(timestamps) < 3:
        return []

    try:
        features = analyze_audio(audio_path, duration)
    except LrcError:
        return []

    changes: list[dict[str, object]] = []
    for index in range(1, len(timestamps)):
        assignment = assignments[index] if isinstance(assignments[index], dict) else {}
        if assignment.get("timing_repair") != "ctc-forced-align":
            continue
        try:
            ctc_score = float(assignment.get("ctc_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            ctc_score = 0.0
        romaji = str(assignment.get("romaji") or "").lower()

        current_time = timestamps[index]
        previous_time = timestamps[index - 1]
        next_time = timestamps[index + 1] if index + 1 < len(timestamps) else duration
        if current_time - previous_time < 2.4:
            continue
        if next_time - current_time < 1.2:
            continue

        search_start = 0.0
        search_end = current_time
        mode = ""
        candidate: float | None = None
        if ctc_score <= 0.22 and romaji.startswith("r"):
            mode = "r-initial-acoustic"
            search_start = max(previous_time + 0.80, current_time - 0.90)
            search_end = current_time - 0.10
            if search_end - search_start < 0.25:
                continue
            candidate = first_local_onset(features, search_start, search_end, threshold=0.55)
            if candidate is None:
                candidate = first_local_onset(features, search_start, search_end, threshold=0.45)
            min_shift = 0.45
            max_shift = 0.75
        elif (
            ctc_score <= 0.28
            and romaji.startswith(("m", "f", "t"))
            and current_time - previous_time >= 4.0
            and next_time - current_time >= 7.0
        ):
            first_candidates = assignment.get("ctc_first_token_candidates")
            if not isinstance(first_candidates, list) or not first_candidates:
                continue
            top_score = 0.0
            for item in first_candidates:
                if not isinstance(item, dict):
                    continue
                try:
                    top_score = max(top_score, float(item.get("score", 0.0) or 0.0))
                except (TypeError, ValueError):
                    continue
            eligible: list[tuple[float, float, float]] = []
            for item in first_candidates:
                if not isinstance(item, dict):
                    continue
                try:
                    item_time = float(item.get("time", 0.0) or 0.0)
                    item_score = float(item.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                shift = current_time - item_time
                ratio = item_score / top_score if top_score > 0 else 0.0
                if 1.20 <= shift <= 1.90 and item_score >= 0.0005 and ratio >= 0.06:
                    eligible.append((item_time, item_score, ratio))
            if not eligible:
                continue
            eligible.sort(key=lambda value: value[0])
            candidate, candidate_score, candidate_ratio = eligible[0]
            if romaji.startswith("t") and candidate_score < 0.02:
                continue
            mode = "ctc-first-token-posterior"
            search_start = max(previous_time + 0.20, current_time - 2.50)
            search_end = current_time - 0.05
            min_shift = 1.20
            max_shift = 1.90
        elif ctc_score <= 0.12 and romaji.startswith("m") and current_time - previous_time >= 2.5:
            first_candidates = assignment.get("ctc_first_token_candidates")
            if not isinstance(first_candidates, list) or not first_candidates:
                continue
            top_score = 0.0
            for item in first_candidates:
                if not isinstance(item, dict):
                    continue
                try:
                    top_score = max(top_score, float(item.get("score", 0.0) or 0.0))
                except (TypeError, ValueError):
                    continue
            eligible = []
            for item in first_candidates:
                if not isinstance(item, dict):
                    continue
                try:
                    item_time = float(item.get("time", 0.0) or 0.0)
                    item_score = float(item.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                shift = current_time - item_time
                ratio = item_score / top_score if top_score > 0 else 0.0
                if 0.55 <= shift <= 0.85 and item_score >= 0.01 and ratio >= 0.55:
                    eligible.append((item_time, item_score, ratio))
            if not eligible:
                continue
            eligible.sort(key=lambda value: value[0])
            candidate = eligible[0][0]
            mode = "ctc-near-first-token-posterior"
            search_start = max(previous_time + 0.20, current_time - 0.90)
            search_end = current_time - 0.05
            min_shift = 0.55
            max_shift = 0.85
        elif (
            ctc_score <= 0.05
            and romaji.startswith(("s", "t", "o"))
            and current_time - previous_time >= 2.5
            and next_time - current_time >= 2.0
        ):
            mode = "ctc-short-acoustic-onset"
            search_start = max(previous_time + 0.35, current_time - 0.72)
            search_end = current_time - 0.10
            if search_end - search_start < 0.18:
                continue
            candidate = first_local_onset(features, search_start, search_end, threshold=0.50)
            if candidate is None:
                candidate = first_local_onset(features, search_start, search_end, threshold=0.45)
            min_shift = 0.20
            max_shift = 0.55
        else:
            continue
        if candidate is None:
            continue

        shift = current_time - candidate
        if shift < min_shift or shift > max_shift:
            continue
        if candidate <= previous_time + 0.20:
            continue
        if candidate >= next_time - 0.20:
            continue
        previous_units = max(1, len(normalize_match_text(entry_sung_text(entries[index - 1]))))
        current_units = max(1, len(normalize_match_text(entry_sung_text(entries[index]))))
        before_tempo_error = abs(
            ((current_time - previous_time) / previous_units)
            - ((next_time - current_time) / current_units)
        )
        after_tempo_error = abs(
            ((candidate - previous_time) / previous_units)
            - ((next_time - candidate) / current_units)
        )
        # The mixed-track onset can fire on a preceding sustain or instrument.
        # Do not backtrack a short CTC line when its initial consonant is already
        # credible and the proposed move makes adjacent lyric pacing less
        # coherent.  Local CTC can still revise genuinely weak starts later.
        first_token_score = 0.0
        token_spans = assignment.get("ctc_token_spans")
        if isinstance(token_spans, list) and token_spans:
            try:
                first_token_score = float(token_spans[0].get("score", 0.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                first_token_score = 0.0
        harms_pacing = (
            before_tempo_error <= 0.025
            or after_tempo_error > max(0.01, before_tempo_error * 1.30)
        )
        if mode == "ctc-short-acoustic-onset" and first_token_score >= 0.04 and harms_pacing:
            continue
        if mode == "ctc-short-acoustic-onset" and first_token_score >= 0.05:
            continue
        timestamps[index] = candidate
        assignment["timestamp"] = round(candidate, 3)
        assignment["timing_repair_source"] = "torchaudio-mms-fa+acoustic-backtrack"
        assignment["ctc_acoustic_backtrack"] = True
        assignment["ctc_acoustic_backtrack_from"] = round(current_time, 3)
        assignment["ctc_acoustic_backtrack_mode"] = mode
        changes.append(
            {
                "entry": index + 1,
                "from": round(current_time, 3),
                "to": round(candidate, 3),
                "shift": round(shift, 3),
                "ctc_score": round(ctc_score, 6),
                "search_start": round(search_start, 3),
                "search_end": round(search_end, 3),
                "romaji_prefix": romaji[:8],
                "mode": mode,
                "before_tempo_error": round(before_tempo_error, 6),
                "after_tempo_error": round(after_tempo_error, 6),
                "lyric": entries[index].lines[0] if entries[index].lines else "",
            }
        )

    if changes:
        report["ctc_acoustic_backtrack_count"] = len(changes)
        report["ctc_acoustic_backtracks"] = changes
    else:
        report["ctc_acoustic_backtrack_count"] = 0
        report["ctc_acoustic_backtracks"] = []
    return changes


def repair_intro_hallucination(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(timestamps) < 3:
        return timestamps, []
    first_assignment = assignments[0] if isinstance(assignments[0], dict) else {}
    if assignment_score(first_assignment) >= COLLAPSE_SEGMENT_SCORE:
        return timestamps, []
    if not first_assignment.get("borrowed") and first_assignment.get("segment") is None:
        return timestamps, []
    if timestamps[0] >= 10.0:
        return timestamps, []

    anchor_index: int | None = None
    for index in range(1, min(len(timestamps), len(assignments))):
        assignment = assignments[index] if isinstance(assignments[index], dict) else {}
        if assignment_score(assignment) >= TRUSTED_ALIGNMENT_SCORE and timestamps[index] >= 20.0:
            anchor_index = index
            break
    if anchor_index is None or anchor_index < 2:
        return timestamps, []

    anchor_time = timestamps[anchor_index]
    prefix_count = anchor_index
    search_start = max(0.0, anchor_time - min(12.0, 4.0 + prefix_count * 4.0))
    search_end = max(search_start + 0.2, anchor_time - 6.0)
    features = analyze_audio(audio_path, duration)
    repaired_start = first_local_onset(features, search_start, search_end)
    if repaired_start is None:
        return timestamps, []
    if repaired_start <= timestamps[0] + 1.0:
        return timestamps, []

    repaired = list(timestamps)
    changes: list[dict[str, object]] = []
    interval = max(0.2, anchor_time - repaired_start)
    for index in range(prefix_count):
        fraction = index / max(1, prefix_count)
        new_time = repaired_start + interval * fraction
        if index > 0:
            new_time = max(new_time, repaired[index - 1] + 0.10)
        new_time = min(new_time, anchor_time - 0.10 * (prefix_count - index))
        changes.append(
            {
                "entry": index + 1,
                "from": round(repaired[index], 3),
                "to": round(new_time, 3),
                "reason": "intro-hallucination-prefix-redistribution",
                "anchor_entry": anchor_index + 1,
                "anchor_time": round(anchor_time, 3),
                "search_window": [round(search_start, 3), round(search_end, 3)],
                "lyric": entries[index].lines[0] if index < len(entries) and entries[index].lines else "",
            }
        )
        repaired[index] = new_time

    report["intro_hallucination_repair_count"] = len(changes)
    report["intro_hallucination_repairs"] = changes
    return repaired, changes


def repair_long_segment_hallucinations(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    raw_segments: list[AsrSegment],
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(timestamps) < 3:
        return timestamps, []
    features: AudioFeatures | None = None
    repaired = list(timestamps)
    changes: list[dict[str, object]] = []

    for index in range(1, min(len(repaired) - 1, len(assignments))):
        assignment = assignments[index] if isinstance(assignments[index], dict) else {}
        segment_number = assignment.get("segment")
        if not isinstance(segment_number, int):
            continue
        segment_index = segment_number - 1
        if not (0 <= segment_index < len(raw_segments)):
            continue
        segment = raw_segments[segment_index]
        segment_duration = segment.end - segment.start
        next_time = repaired[index + 1]
        score = assignment_score(assignment)
        if not (
            segment_duration >= 18.0
            and score >= 0.80
            and next_time - repaired[index] >= 12.0
            and abs(next_time - segment.end) <= 0.30
        ):
            continue
        text_length = len(normalize_match_text(entry_sung_text(entries[index])))
        lead = min(2.2, max(1.4, text_length * 0.055))
        new_time = max(repaired[index - 1] + 0.10, next_time - lead)
        if not (repaired[index] + 5.0 < new_time < next_time - 0.20):
            continue
        changes.append(
            {
                "entry": index + 1,
                "from": round(repaired[index], 3),
                "to": round(new_time, 3),
                "reason": "long-segment-pre-vocal-hallucination",
                "segment": segment_number,
                "segment_duration": round(segment_duration, 3),
                "next_anchor_time": round(next_time, 3),
                "lyric": entries[index].lines[0] if entries[index].lines else "",
            }
        )
        repaired[index] = new_time

        next_index = index + 1
        if next_index + 1 >= len(repaired):
            continue
        next_assignment = assignments[next_index] if isinstance(assignments[next_index], dict) else {}
        if assignment_score(next_assignment) < TRUSTED_ALIGNMENT_SCORE:
            continue
        if features is None:
            features = analyze_audio(audio_path, duration)
        search_start = repaired[next_index] + 1.20
        search_end = min(repaired[next_index] + 2.60, repaired[next_index + 1] - 0.10)
        onset = first_local_onset(features, search_start, search_end, threshold=0.55)
        if onset is None or not (repaired[next_index] + 0.40 < onset < repaired[next_index + 1] - 0.10):
            continue
        changes.append(
            {
                "entry": next_index + 1,
                "from": round(repaired[next_index], 3),
                "to": round(onset, 3),
                "reason": "post-long-segment-local-onset",
                "previous_repaired_entry": index + 1,
                "search_window": [round(search_start, 3), round(search_end, 3)],
                "lyric": entries[next_index].lines[0] if entries[next_index].lines else "",
            }
        )
        repaired[next_index] = onset

    if changes:
        report["long_segment_repair_count"] = len(changes)
        report["long_segment_repairs"] = changes
    return repaired, changes


def sync_report_assignment_timestamps(report: dict[str, object], timestamps: list[float]) -> None:
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return
    for index, timestamp in enumerate(timestamps):
        if index < len(assignments) and isinstance(assignments[index], dict):
            assignments[index]["timestamp"] = round(timestamp, 3)


def low_confidence_tail_start(report: dict[str, object], min_remaining: int = 4) -> int | None:
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return None
    for index in range(max(1, len(assignments) - 12), len(assignments) - min_remaining + 1):
        remaining = assignments[index:]
        low_count = sum(1 for item in remaining if assignment_score(item) < TRUSTED_ALIGNMENT_SCORE)
        if low_count < max(3, math.ceil(len(remaining) * 0.55)):
            continue
        consecutive_low = 0
        for item in remaining:
            if assignment_score(item) < TRUSTED_ALIGNMENT_SCORE:
                consecutive_low += 1
            else:
                break
        if consecutive_low >= 2:
            return index
    return None


def first_sustained_onset(
    features: AudioFeatures,
    start: float,
    end: float,
    onset_threshold: float = 0.70,
    median_rms_floor: float = -18.0,
    sustain_seconds: float = 3.0,
) -> float | None:
    if end <= start:
        return None
    frame_times = features.frame_times
    mask = (frame_times >= start) & (frame_times <= end)
    for index in np.flatnonzero(mask):
        strength = float(features.onset_strength[index])
        if strength < onset_threshold:
            continue
        time = float(frame_times[index])
        sustain_mask = (frame_times >= time) & (frame_times <= min(time + sustain_seconds, features.duration))
        if not np.any(sustain_mask):
            continue
        if float(np.median(features.rms_db[sustain_mask])) >= median_rms_floor:
            return time
    return None


def has_repeated_asr_tail(segments: list[AsrSegment], after_time: float, min_run: int = 6) -> bool:
    run_text = ""
    run_count = 0
    for segment in segments:
        if segment.start < after_time:
            continue
        normalized = normalize_match_text(segment.text)
        if len(normalized) < 8:
            run_text = ""
            run_count = 0
            continue
        if run_text and sequence_ratio(run_text, normalized) >= 0.86:
            run_count += 1
        else:
            run_text = normalized
            run_count = 1
        if run_count >= min_run:
            return True
    return False


def repair_tail_hallucination(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    segments: list[AsrSegment],
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    tail_start = low_confidence_tail_start(report)
    if tail_start is None or tail_start <= 0:
        return timestamps, []

    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return timestamps, []
    anchor_time = timestamps[tail_start - 1]
    if anchor_time <= 0.0:
        return timestamps, []
    if not has_repeated_asr_tail(segments, anchor_time + 4.0):
        report["tail_hallucination_repair"] = {
            "status": "skipped",
            "reason": "no-repeated-asr-tail",
            "candidate_start_entry": tail_start + 1,
            "anchor_entry": tail_start,
            "anchor_time": round(anchor_time, 3),
        }
        return timestamps, []

    features = analyze_audio(audio_path, duration)
    search_start = min(duration, anchor_time + 8.0)
    search_end = min(duration, anchor_time + 58.0)
    tail_time = first_sustained_onset(features, search_start, search_end)
    if tail_time is None:
        tail_time = min(duration, anchor_time + 12.0)
    if tail_time - anchor_time < 12.0:
        report["tail_hallucination_repair"] = {
            "status": "skipped",
            "reason": "tail-onset-too-close-to-anchor",
            "candidate_start_entry": tail_start + 1,
            "anchor_entry": tail_start,
            "anchor_time": round(anchor_time, 3),
            "tail_start_time": round(tail_time, 3),
        }
        return timestamps, []

    tail_end = duration
    if features.segments:
        tail_end = max(tail_time + 1.0, min(duration, features.segments[-1][1]))
    if tail_end - tail_time < max(8.0, (len(entries) - tail_start) * 1.4):
        return timestamps, []

    tail_times = weighted_rough_times(entries[tail_start:], tail_time, tail_end)
    tail_times = snap_to_onsets(tail_times, features, window_seconds=0.90, min_gap=0.35)
    repaired = list(timestamps)
    changes: list[dict[str, object]] = []
    for offset, entry_index in enumerate(range(tail_start, len(entries))):
        new_time = max(repaired[entry_index - 1] + 0.35 if entry_index > 0 else 0.0, tail_times[offset])
        old_time = repaired[entry_index]
        repaired[entry_index] = min(duration, new_time)
        if entry_index < len(assignments) and isinstance(assignments[entry_index], dict):
            assignment = assignments[entry_index]
            assignment["timestamp"] = round(repaired[entry_index], 3)
            assignment["timing_repair"] = "tail-hallucination-acoustic"
            assignment["timing_repair_source"] = "sustained-onset-weighted-tail"
            assignment["score"] = max(assignment_score(assignment), 0.70)
        changes.append(
            {
                "entry": entry_index + 1,
                "from": round(old_time, 3),
                "to": round(repaired[entry_index], 3),
                "reason": "tail-hallucination-acoustic",
                "lyric": entries[entry_index].lines[0] if entries[entry_index].lines else "",
            }
        )

    report["tail_hallucination_repair"] = {
        "status": "applied",
        "start_entry": tail_start + 1,
        "anchor_entry": tail_start,
        "anchor_time": round(anchor_time, 3),
        "tail_start_time": round(tail_time, 3),
        "tail_end_time": round(tail_end, 3),
        "entries": len(changes),
        "method": "sustained-onset-weighted-tail",
    }
    report["tail_hallucination_repairs"] = changes
    update_report_confidence_metrics(report)
    return repaired, changes


def blend_tail_repair_with_forced(
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    forced_segments: list[AsrSegment],
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return timestamps, []
    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            continue
        if assignment.get("timing_repair") != "tail-hallucination-acoustic":
            continue
        if index >= len(forced_segments):
            continue
        forced_time = first_forced_char_time(forced_segments[index])
        current_time = refined[index]
        previous_time = refined[index - 1] if index > 0 else 0.0
        next_time = refined[index + 1] if index + 1 < len(refined) else duration
        if not (previous_time + 0.35 < forced_time < next_time - 0.35):
            continue
        lead = current_time - forced_time
        if abs(lead) <= 0.35:
            candidate = forced_time
        elif 0.35 < lead <= 6.0:
            candidate = current_time - min(1.60, lead * 0.34)
        else:
            continue
        if not (previous_time + 0.35 < candidate < next_time - 0.35):
            continue
        refined[index] = candidate
        assignment["timestamp"] = round(candidate, 3)
        assignment["timing_repair_source"] = "sustained-onset-weighted-tail+forced-blend"
        changes.append(
            {
                "entry": index + 1,
                "from": round(current_time, 3),
                "to": round(candidate, 3),
                "forced_first": round(forced_time, 3),
                "reason": "tail-forced-blend",
                "lyric": entries[index].lines[0] if entries[index].lines else "",
            }
        )

    if changes:
        existing = report.get("tail_hallucination_repair")
        if isinstance(existing, dict):
            existing["method"] = "sustained-onset-weighted-tail+forced-blend"
            existing["forced_blend_count"] = len(changes)
        report["tail_forced_blend_count"] = len(changes)
        report["tail_forced_blends"] = changes
        update_report_confidence_metrics(report)
    return refined, changes


def assignment_has_timing(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    return item.get("segment") is not None or item.get("timing_repair") is not None


def assignment_score(item: object) -> float:
    if not isinstance(item, dict):
        return 0.0
    try:
        return float(item.get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def assignment_is_trusted(item: object) -> bool:
    if not assignment_has_timing(item):
        return False
    if not isinstance(item, dict):
        return False
    return assignment_score(item) >= TRUSTED_ALIGNMENT_SCORE or bool(item.get("timing_trusted"))


def update_report_confidence_metrics(report: dict[str, object]) -> None:
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return
    timing_entries = int(report.get("timing_entries", len(assignments)) or len(assignments))
    assigned_entries = sum(
        1 for item in assignments if assignment_has_timing(item)
    )
    trusted_entries = sum(
        1
        for item in assignments
        if assignment_is_trusted(item)
    )
    low_confidence_count = sum(
        1
        for item in assignments
        if not assignment_is_trusted(item)
    )
    low_confidence_entries = report.get("low_confidence_entries")
    if isinstance(low_confidence_entries, list):
        filtered_low_confidence: list[object] = []
        for item in low_confidence_entries:
            if not isinstance(item, dict):
                continue
            entry_number = item.get("entry")
            if not isinstance(entry_number, int):
                filtered_low_confidence.append(item)
                continue
            entry_index = entry_number - 1
            if not (0 <= entry_index < len(assignments)):
                filtered_low_confidence.append(item)
                continue
            assignment = assignments[entry_index]
            if not assignment_is_trusted(assignment):
                filtered_low_confidence.append(item)
        report["low_confidence_entries"] = filtered_low_confidence
        low_confidence_count = len(filtered_low_confidence)
    review_required_count = int(report.get("review_required_count", 0) or 0)
    report["assigned_entries"] = assigned_entries
    report["assigned_percent"] = ratio_percent(assigned_entries, timing_entries)
    report["trusted_entries"] = trusted_entries
    report["trusted_percent"] = ratio_percent(trusted_entries, timing_entries)
    report["low_confidence_count"] = low_confidence_count
    report["low_confidence_percent"] = ratio_percent(low_confidence_count, timing_entries)
    report["timing_trusted_entries"] = sum(
        1 for item in assignments if isinstance(item, dict) and item.get("timing_trusted")
    )
    report["review_required_percent"] = ratio_percent(review_required_count, timing_entries)
    report["matched_entries"] = trusted_entries
    report["matched_percent"] = report["trusted_percent"]


def match_anchor_entries(entries: list[LyricEntry], anchors: list[AnchorHint]) -> list[tuple[int, LyricEntry]]:
    normalized_entries = [normalize_match_text(entry_sung_text(entry)) for entry in entries]
    used: set[int] = set()
    previous_index = -1
    matches: list[tuple[int, LyricEntry]] = []
    for hint in anchors:
        anchor = hint.entry
        normalized_anchor = normalize_match_text(entry_sung_text(anchor))
        if not normalized_anchor:
            continue
        if hint.entry_number is not None:
            chosen = hint.entry_number - 1
            if not (0 <= chosen < len(entries)):
                raise LrcError(
                    f"Anchor hint entry={hint.entry_number} is outside lyric range 1..{len(entries)}"
                )
            if chosen in used:
                raise LrcError(f"Duplicate anchor hint for entry={hint.entry_number}")
            if normalized_entries[chosen] != normalized_anchor:
                raise LrcError(
                    "Anchor hint entry="
                    f"{hint.entry_number} text mismatch: expected {entry_sung_text(entries[chosen])!r}, "
                    f"got {entry_sung_text(anchor)!r}"
                )
            used.add(chosen)
            previous_index = chosen
            matches.append((chosen, anchor))
            continue
        candidate_indexes = [
            index
            for index, normalized_entry in enumerate(normalized_entries)
            if index not in used and normalized_entry == normalized_anchor
        ]
        if not candidate_indexes:
            raise LrcError(f"Anchor hint line does not match any lyric entry: {entry_sung_text(anchor)!r}")
        forward_indexes = [index for index in candidate_indexes if index > previous_index]
        chosen = forward_indexes[0] if forward_indexes else candidate_indexes[0]
        used.add(chosen)
        previous_index = chosen
        matches.append((chosen, anchor))
    return matches


def apply_anchor_hints(
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    anchor_path: Path | None,
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    if anchor_path is None:
        report["anchor_hints"] = {"enabled": False}
        return timestamps, []
    anchor_entries, skipped_markers = load_anchor_hints(anchor_path)
    matches = match_anchor_entries(entries, anchor_entries)
    refined = list(timestamps)
    assignments = report.get("assignments")
    suspicious_items = report.get("suspicious_alignments")
    low_confidence_entries = report.get("low_confidence_entries")
    changes: list[dict[str, object]] = []
    anchored_entries: set[int] = set()

    for index, anchor in matches:
        if anchor.source_time_cs is None:
            continue
        anchor_time = max(0.0, min(duration, float(anchor.source_time_cs) / 100.0))
        previous_time = refined[index - 1] if index > 0 else 0.0
        next_time = refined[index + 1] if index + 1 < len(refined) else duration
        if index > 0 and anchor_time <= previous_time + 0.05:
            raise LrcError(
                f"Anchor hint for entry {index + 1} is not after previous timestamp: {format_lrc_time(anchor_time)}"
            )
        if index + 1 < len(refined) and anchor_time >= next_time - 0.05:
            raise LrcError(
                f"Anchor hint for entry {index + 1} is not before next timestamp: {format_lrc_time(anchor_time)}"
            )
        original_time = refined[index]
        refined[index] = anchor_time
        anchored_entries.add(index + 1)
        changes.append(
            {
                "entry": index + 1,
                "text": entry_sung_text(entries[index]),
                "from": round(original_time, 3),
                "to": round(anchor_time, 3),
                "delta": round(anchor_time - original_time, 3),
                "anchor_path": str(anchor_path),
            }
        )
        if isinstance(assignments, list) and index < len(assignments) and isinstance(assignments[index], dict):
            assignment = assignments[index]
            assignment["timestamp"] = round(anchor_time, 3)
            assignment["manual_anchor_hint"] = True
            assignment["manual_anchor_path"] = str(anchor_path)
            assignment["manual_anchor_original_timestamp"] = round(original_time, 3)
            assignment["score"] = max(assignment_score(assignment), 1.0)
            assignment["timing_repair"] = "manual-anchor-hint"
            assignment["timing_repair_source"] = str(anchor_path)

    if isinstance(suspicious_items, list):
        for item in suspicious_items:
            if not isinstance(item, dict) or item.get("entry") not in anchored_entries:
                continue
            item["manual_anchor_resolved"] = True
            item["review_required"] = False
            item["original_severity"] = item.get("severity")
            item["severity"] = "resolved"
            flags = item.get("flags")
            if isinstance(flags, list):
                item["flags"] = list(dict.fromkeys([*flags, "manual_anchor_resolved"]))
            candidates = item.get("candidate_timestamps")
            if isinstance(candidates, dict):
                entry_number = int(item["entry"])
                candidates["manual_anchor"] = round(refined[entry_number - 1], 3)
        report["suspicious_alignments"] = suspicious_items

    if isinstance(low_confidence_entries, list):
        report["low_confidence_entries"] = [
            item
            for item in low_confidence_entries
            if not (isinstance(item, dict) and item.get("entry") in anchored_entries)
        ]

    report["anchor_hints"] = {
        "enabled": True,
        "path": str(anchor_path),
        "applied_count": len(changes),
        "skipped_marker_entries": skipped_markers,
    }
    report["anchor_hint_count"] = len(changes)
    report["anchor_hint_changes"] = changes
    if isinstance(suspicious_items, list):
        severity_counts = {
            "high": sum(1 for item in suspicious_items if isinstance(item, dict) and item.get("severity") == "high"),
            "medium": sum(1 for item in suspicious_items if isinstance(item, dict) and item.get("severity") == "medium"),
            "low": sum(1 for item in suspicious_items if isinstance(item, dict) and item.get("severity") == "low"),
        }
        report["suspicious_alignment_severity_counts"] = severity_counts
        report["review_required_count"] = sum(
            1 for item in suspicious_items if isinstance(item, dict) and item.get("review_required")
        )
        report["review_required"] = int(report["review_required_count"]) > 0
    update_report_confidence_metrics(report)
    selection = report.get("candidate_selection")
    if isinstance(selection, dict):
        post_anchor_summary = candidate_summary(report)
        report["post_anchor_selected_summary"] = post_anchor_summary
        selection["post_anchor_selected_summary"] = post_anchor_summary
        selection["post_anchor_note"] = "Manual anchor hints were applied after backend selection."
    return refined, changes


def severity_rank(severity: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(severity, 0)


def dedupe_suspicious_alignments(items: list[dict[str, object]]) -> list[dict[str, object]]:
    by_entry: dict[int, dict[str, object]] = {}
    for item in items:
        entry = item.get("entry")
        if not isinstance(entry, int):
            continue
        existing = by_entry.get(entry)
        if existing is None:
            merged = copy.deepcopy(item)
            merged["flags"] = list(dict.fromkeys(merged.get("flags", [])))
            by_entry[entry] = merged
            continue
        existing_flags = list(existing.get("flags", []))
        incoming_flags = list(item.get("flags", []))
        existing["flags"] = list(dict.fromkeys(existing_flags + incoming_flags))
        existing_severity = str(existing.get("severity", "low"))
        incoming_severity = str(item.get("severity", "low"))
        if severity_rank(incoming_severity) > severity_rank(existing_severity):
            existing["severity"] = incoming_severity
        existing["review_required"] = bool(existing.get("review_required")) or bool(item.get("review_required"))
        for key in ("candidate_timestamps", "candidate_scores"):
            incoming_value = item.get(key)
            if isinstance(incoming_value, dict):
                existing_value = existing.get(key)
                merged_value = dict(existing_value) if isinstance(existing_value, dict) else {}
                merged_value.update(incoming_value)
                existing[key] = merged_value
        for key in ("delta", "ctc_score", "raw_score", "whisperx_score"):
            if key in item:
                existing[key] = item[key]
    return [by_entry[entry] for entry in sorted(by_entry)]


def find_alignment_collapse_runs(assignments: list[object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    segment_runs: list[dict[str, object]] = []
    zero_runs: list[dict[str, object]] = []

    index = 0
    while index < len(assignments):
        item = assignments[index]
        segment = item.get("segment") if isinstance(item, dict) else None
        score = assignment_score(item)
        if segment is None or score >= COLLAPSE_SEGMENT_SCORE:
            index += 1
            continue
        start = index
        while index < len(assignments):
            current = assignments[index]
            current_segment = current.get("segment") if isinstance(current, dict) else None
            if current_segment != segment or assignment_score(current) >= COLLAPSE_SEGMENT_SCORE:
                break
            index += 1
        if index - start >= 4:
            segment_runs.append(
                {
                    "start_entry": start + 1,
                    "end_entry": index,
                    "segment": segment,
                    "length": index - start,
                    "flag": "segment_collapse",
                }
            )

    index = 0
    while index < len(assignments):
        if assignment_score(assignments[index]) != 0.0:
            index += 1
            continue
        start = index
        while index < len(assignments) and assignment_score(assignments[index]) == 0.0:
            index += 1
        if index - start >= 6:
            zero_runs.append(
                {
                    "start_entry": start + 1,
                    "end_entry": index,
                    "length": index - start,
                    "flag": "alignment_collapse",
                }
            )
    return segment_runs, zero_runs


def add_alignment_diagnostics(
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    forced_segments: list[AsrSegment],
    duration: float,
) -> None:
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return

    forced_times = [first_forced_char_time(segment) for segment in forced_segments]
    anchor_profiles = [lyric_anchor_profile(entry) for entry in entries]
    suspicious: list[dict[str, object]] = []
    suspicious_indexes: set[int] = set()
    high_risk_indexes: set[int] = set()

    for index, entry in enumerate(entries):
        raw_text = entry_sung_text(entry)
        entry_text = normalize_match_text(raw_text)
        if not entry_text:
            continue
        timestamp = timestamps[index] if index < len(timestamps) else 0.0
        previous_time = timestamps[index - 1] if index > 0 else 0.0
        next_time = timestamps[index + 1] if index + 1 < len(timestamps) else duration
        forced_time = forced_times[index] if index < len(forced_times) else None
        assignment = assignments[index] if index < len(assignments) and isinstance(assignments[index], dict) else {}
        score = float(assignment.get("score", 0.0) or 0.0) if isinstance(assignment, dict) else 0.0
        gap_before = timestamp - previous_time if index > 0 else timestamp
        gap_after = next_time - timestamp if index + 1 < len(timestamps) else duration - timestamp
        is_long = len(entry_text) >= 18 or len(raw_text) >= 24
        flags: list[str] = []
        candidate_timestamps: dict[str, float] = {
            "output": round(timestamp, 3),
        }

        if forced_time is not None:
            candidate_timestamps["whisperx_forced_first"] = round(forced_time, 3)
            disagreement = abs(timestamp - forced_time)
            if disagreement >= (2.0 if is_long else 1.25):
                flags.append("candidate_disagreement")
            if is_long and timestamp - forced_time >= 3.0:
                flags.append("possible_midline_anchor")
            if not is_long and gap_before <= 3.5 and disagreement >= 0.70:
                flags.append("close_neighbor_onset_uncertain")
            if forced_time <= previous_time + 0.20:
                flags.append("forced_time_overlaps_previous_line")
        borrowed_assignment = bool(assignment.get("borrowed")) if isinstance(assignment, dict) else False
        if index == 0 and borrowed_assignment:
            next_gap = timestamps[1] - timestamp if len(timestamps) > 1 else 0.0
            if timestamp < 10.0 and next_gap >= 10.0:
                flags.append("possible_instrumental_intro_hallucination")

        if is_long and gap_after <= 3.5:
            flags.append("long_line_close_to_next_entry")
        if is_long and gap_before >= 8.0 and gap_after >= 6.0:
            flags.append("long_line_wide_window")
        if is_long and score < 0.88:
            flags.append("long_line_not_high_confidence")

        if flags:
            severity = "low"
            if "possible_midline_anchor" in flags or "forced_time_overlaps_previous_line" in flags:
                severity = "high"
            if "possible_instrumental_intro_hallucination" in flags:
                severity = "high"
            elif "long_line_close_to_next_entry" in flags or "close_neighbor_onset_uncertain" in flags:
                severity = "medium"
            review_required = severity in ("high", "medium")
            suspicious_indexes.add(index)
            if severity == "high":
                high_risk_indexes.add(index)
            suspicious.append(
                {
                    "entry": index + 1,
                    "text": raw_text,
                    "flags": flags,
                    "severity": severity,
                    "score": round(score, 3),
                    "anchor_profile": anchor_profiles[index],
                    "candidate_timestamps": candidate_timestamps,
                    "gap_before": round(gap_before, 3),
                    "gap_after": round(gap_after, 3),
                    "review_required": review_required,
                }
            )

    for index, entry in enumerate(entries):
        if index == 0 or index in suspicious_indexes or index - 1 not in suspicious_indexes:
            continue
        previous_time = timestamps[index - 1] if index > 0 else 0.0
        timestamp = timestamps[index] if index < len(timestamps) else previous_time
        if timestamp - previous_time > 3.5:
            continue
        raw_text = entry_sung_text(entry)
        assignment = assignments[index] if index < len(assignments) and isinstance(assignments[index], dict) else {}
        score = float(assignment.get("score", 0.0) or 0.0) if isinstance(assignment, dict) else 0.0
        severity = "high" if index - 1 in high_risk_indexes else "medium"
        suspicious_indexes.add(index)
        suspicious.append(
            {
                "entry": index + 1,
                "text": raw_text,
                "flags": ["follows_suspicious_long_line", "possible_bad_split_boundary"],
                "severity": severity,
                "score": round(score, 3),
                "anchor_profile": anchor_profiles[index],
                "candidate_timestamps": {
                    "output": round(timestamp, 3),
                },
                "gap_before": round(timestamp - previous_time, 3),
                "gap_after": round((timestamps[index + 1] if index + 1 < len(timestamps) else duration) - timestamp, 3),
                "review_required": True,
            }
        )

    segment_collapse_runs, alignment_collapse_runs = find_alignment_collapse_runs(assignments)
    collapse_indexes: set[int] = set()
    for run in segment_collapse_runs:
        for entry_number in range(int(run["start_entry"]), int(run["end_entry"]) + 1):
            collapse_indexes.add(entry_number - 1)
    for run in alignment_collapse_runs:
        for entry_number in range(int(run["start_entry"]), int(run["end_entry"]) + 1):
            collapse_indexes.add(entry_number - 1)
    for index in sorted(collapse_indexes):
        raw_text = entry_sung_text(entries[index]) if index < len(entries) else ""
        assignment = assignments[index] if index < len(assignments) and isinstance(assignments[index], dict) else {}
        score = assignment_score(assignment)
        flags = []
        if any(int(run["start_entry"]) <= index + 1 <= int(run["end_entry"]) for run in segment_collapse_runs):
            flags.append("segment_collapse")
        if any(int(run["start_entry"]) <= index + 1 <= int(run["end_entry"]) for run in alignment_collapse_runs):
            flags.append("alignment_collapse")
        suspicious.append(
            {
                "entry": index + 1,
                "text": raw_text,
                "flags": flags,
                "severity": "high",
                "score": round(score, 3),
                "anchor_profile": anchor_profiles[index] if index < len(anchor_profiles) else {},
                "candidate_timestamps": {
                    "output": round(timestamps[index], 3) if index < len(timestamps) else 0.0,
                },
                "gap_before": round((timestamps[index] - timestamps[index - 1]) if 0 < index < len(timestamps) else (timestamps[index] if index < len(timestamps) else 0.0), 3),
                "gap_after": round((timestamps[index + 1] if index + 1 < len(timestamps) else duration) - (timestamps[index] if index < len(timestamps) else 0.0), 3),
                "review_required": True,
            }
        )

    long_repairs = report.get("long_segment_repairs")
    if isinstance(long_repairs, list) and long_repairs:
        repaired_entries = [
            int(item.get("entry"))
            for item in long_repairs
            if isinstance(item, dict) and isinstance(item.get("entry"), int)
        ]
        if repaired_entries:
            start_entry = max(repaired_entries) + 1
            end_entry = min(len(entries), start_entry + 8)
            for entry_number in range(start_entry, end_entry + 1):
                index = entry_number - 1
                if not (0 <= index < len(entries)):
                    continue
                assignment = assignments[index] if index < len(assignments) and isinstance(assignments[index], dict) else {}
                suspicious.append(
                    {
                        "entry": entry_number,
                        "text": entry_sung_text(entries[index]),
                        "flags": ["post_long_segment_region_uncertain"],
                        "severity": "medium",
                        "score": round(assignment_score(assignment), 3),
                        "anchor_profile": anchor_profiles[index] if index < len(anchor_profiles) else {},
                        "candidate_timestamps": {
                            "output": round(timestamps[index], 3) if index < len(timestamps) else 0.0,
                        },
                        "gap_before": round((timestamps[index] - timestamps[index - 1]) if index > 0 else timestamps[index], 3),
                        "gap_after": round((timestamps[index + 1] if index + 1 < len(timestamps) else duration) - timestamps[index], 3),
                        "review_required": True,
                    }
                )

    vocal_regions = report.get("vocal_regions")
    vocal_regions_available = isinstance(vocal_regions, list) and len(vocal_regions) > 0
    if vocal_regions_available:
        def inside_vocal_region(timestamp: float) -> bool:
            for region in vocal_regions:
                if not isinstance(region, dict):
                    continue
                try:
                    start = float(region.get("start", 0.0) or 0.0)
                    end = float(region.get("end", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if start - 0.05 <= timestamp <= end + 0.05:
                    return True
            return False

        for index, timestamp in enumerate(timestamps):
            if inside_vocal_region(timestamp):
                continue
            suspicious.append(
                {
                    "entry": index + 1,
                    "text": entry_sung_text(entries[index]) if index < len(entries) else "",
                    "flags": ["no_vocal_assignment"],
                    "severity": "high",
                    "score": round(assignment_score(assignments[index] if index < len(assignments) else {}), 3),
                    "anchor_profile": anchor_profiles[index] if index < len(anchor_profiles) else {},
                    "candidate_timestamps": {"output": round(timestamp, 3)},
                    "gap_before": round(timestamp - (timestamps[index - 1] if index > 0 else 0.0), 3),
                    "gap_after": round((timestamps[index + 1] if index + 1 < len(timestamps) else duration) - timestamp, 3),
                    "review_required": True,
                }
            )

    suspicious = dedupe_suspicious_alignments(suspicious)
    severity_counts = {
        "high": sum(1 for item in suspicious if item.get("severity") == "high"),
        "medium": sum(1 for item in suspicious if item.get("severity") == "medium"),
        "low": sum(1 for item in suspicious if item.get("severity") == "low"),
    }
    review_required_count = sum(1 for item in suspicious if item.get("review_required"))
    report["suspicious_alignment_count"] = len(suspicious)
    report["review_required_count"] = review_required_count
    report["suspicious_alignment_severity_counts"] = severity_counts
    report["suspicious_alignments"] = suspicious
    report["lyric_anchor_profiles"] = anchor_profiles
    report["segment_collapse_runs"] = segment_collapse_runs
    report["alignment_collapse_runs"] = alignment_collapse_runs
    report["collapse_detected"] = bool(segment_collapse_runs or alignment_collapse_runs)
    report.setdefault("vocal_regions", [])
    report.setdefault("vocal_regions_available", vocal_regions_available)
    report.setdefault(
        "no_vocal_constraint",
        {
            "status": "placeholder",
            "enforced": False,
            "reason": "No reliable vocal-region detector is wired yet.",
        },
    )
    report["review_required"] = review_required_count > 0
    update_report_confidence_metrics(report)


def score_line_timing_candidates(
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> list[float]:
    """Turn backend timestamps into an explainable per-line timing decision."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps):
        return timestamps
    risks = report.get("suspicious_alignments")
    risk_by_entry = {
        int(item.get("entry")): item
        for item in risks
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    } if isinstance(risks, list) else {}
    revised = list(timestamps)

    for index, entry in enumerate(entries):
        assignment = assignments[index]
        if not isinstance(assignment, dict):
            continue
        # Human-reviewed anchors are ground truth for this song. Candidate
        # scoring may describe other evidence, but must never overwrite them.
        if assignment.get("manual_anchor_hint"):
            anchor_time = float(assignment.get("timestamp", revised[index]) or revised[index])
            revised[index] = anchor_time
            assignment["chosen_time"] = round(anchor_time, 3)
            assignment["confidence"] = 1.0
            assignment["candidates"] = [
                {
                    "source": "manual_anchor",
                    "time": round(anchor_time, 3),
                    "score": 1.0,
                    "reasons": ["human-reviewed-anchor"],
                }
            ]
            assignment["rejected_candidates"] = []
            assignment["reasons"] = ["human-reviewed-anchor"]
            assignment["penalties"] = []
            assignment["flags"] = ["manual_anchor_resolved"]
            assignment["review_required"] = False
            continue
        entry_number = index + 1
        current_time = float(assignment.get("timestamp", revised[index]) or revised[index])
        previous_time = revised[index - 1] if index else 0.0
        next_time = revised[index + 1] if index + 1 < len(revised) else duration
        profile = lyric_anchor_profile(entry)
        normalized = normalize_match_text(entry_sung_text(entry))
        is_long = len(normalized) >= 18
        is_short = len(normalized) <= 4
        risk = risk_by_entry.get(entry_number, {})
        candidate_times = risk.get("candidate_timestamps") if isinstance(risk, dict) else {}
        if not isinstance(candidate_times, dict):
            candidate_times = {}

        try:
            selected_score = float(assignment.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            selected_score = 0.0
        candidates: list[dict[str, object]] = [
            {
                "source": "selected_path",
                "time": round(current_time, 3),
                "score": round(0.55 + min(0.35, max(0.0, selected_score) * 0.35), 3),
                "reasons": ["backend-selected-timestamp"],
            }
        ]
        raw_score = risk.get("raw_asr_score") if isinstance(risk, dict) else None
        raw_time = candidate_times.get("raw_asr", candidate_times.get("raw"))
        previous_ctc_token_end: float | None = None
        if index and isinstance(assignments[index - 1], dict):
            previous_spans = assignments[index - 1].get("ctc_token_spans")
            if isinstance(previous_spans, list):
                for span in previous_spans:
                    if not isinstance(span, dict):
                        continue
                    try:
                        token_end = float(span.get("end"))
                    except (TypeError, ValueError):
                        continue
                    previous_ctc_token_end = token_end if previous_ctc_token_end is None else max(previous_ctc_token_end, token_end)
        if isinstance(raw_time, (int, float)):
            try:
                raw_quality = float(raw_score) if raw_score is not None else 0.50
            except (TypeError, ValueError):
                raw_quality = 0.50
            candidates.append(
                {
                    "source": "raw_asr",
                    "time": round(float(raw_time), 3),
                    "score": round(0.35 + min(0.45, max(0.0, raw_quality) * 0.45), 3),
                    "reasons": ["raw-asr-lyric-match"],
                }
            )
        detached_initial_ctc_token = False
        detached_initial_ctc_token_score = 1.0
        token_spans = assignment.get("ctc_token_spans")
        if isinstance(token_spans, list) and len(token_spans) >= 2:
            try:
                first_start = float(token_spans[0].get("start"))
                second_start = float(token_spans[1].get("start"))
                detached_initial_ctc_token = second_start - first_start >= 0.45
                detached_initial_ctc_token_score = float(token_spans[0].get("score", 1.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                detached_initial_ctc_token = False
        if previous_ctc_token_end is not None and isinstance(raw_time, (int, float)):
            for candidate in candidates:
                if candidate.get("source") != "raw_asr":
                    continue
                if float(raw_time) <= previous_ctc_token_end + 0.03:
                    candidate["score"] = min(float(candidate["score"]), 0.10)
                    candidate.setdefault("reasons", []).append("raw-inside-previous-ctc-token-tail")
                break
        try:
            parsed_raw_score = float(raw_score) if raw_score is not None else 0.0
        except (TypeError, ValueError):
            parsed_raw_score = 0.0
        if detached_initial_ctc_token and parsed_raw_score >= 0.90:
            for candidate in candidates:
                if candidate.get("source") != "raw_asr":
                    continue
                candidate["score"] = 0.98
                candidate.setdefault("reasons", []).append("raw-resolves-detached-ctc-initial-token")
                break
        if detached_initial_ctc_token and detached_initial_ctc_token_score < 0.02 and parsed_raw_score < 0.90:
            try:
                second_token_start = float(token_spans[1].get("start"))  # type: ignore[index]
            except (AttributeError, TypeError, ValueError, IndexError):
                second_token_start = current_time
            if previous_time + 0.05 < second_token_start < next_time - 0.05:
                candidates.append(
                    {
                        "source": "ctc_detached_initial_token_recovery",
                        "time": round(second_token_start, 3),
                        "score": 0.98,
                        "reasons": ["second-ctc-token-start-after-detached-initial-token"],
                    }
                )
        forced_time = candidate_times.get("whisperx_forced_first")
        if isinstance(forced_time, (int, float)):
            candidates.append(
                {
                    "source": "whisperx_forced_first",
                    "time": round(float(forced_time), 3),
                    "score": 0.58,
                    "reasons": ["forced-first-character"],
                }
            )
        peaks = assignment.get("ctc_first_token_candidates")
        nearby_peaks: list[dict[str, object]] = []
        if isinstance(peaks, list):
            # A posterior peak inside the preceding lyric's final CTC token is
            # a tail echo, not evidence for this line's opening consonant.
            # Keeping it here falsely turned an already-rejected candidate
            # into a phonetic-anchor review flag.
            peak_floor = (previous_ctc_token_end + 0.03) if previous_ctc_token_end is not None else None
            nearby_peaks = [
                peak
                for peak in peaks
                if isinstance(peak, dict)
                and isinstance(peak.get("time"), (int, float))
                and (peak_floor is None or float(peak["time"]) > peak_floor)
                and abs(float(peak["time"]) - current_time) <= 0.45
            ]
            if nearby_peaks:
                peak = max(nearby_peaks, key=lambda item: float(item.get("score", 0.0) or 0.0))
                peak_score = float(peak.get("score", 0.0) or 0.0)
                candidates.append(
                    {
                        "source": "ctc_nearby_phonetic_peak",
                        "time": round(float(peak["time"]), 3),
                        "score": round(0.48 + min(0.30, peak_score * 0.30), 3),
                        "reasons": ["phonetic-first-token-peak-near-selected"],
                    }
                )

        # CTC can skip to a later occurrence when a line begins with the same
        # visible word two or more times. A strong earlier first-token posterior
        # is then more specific evidence than the collapsed whole-line path.
        leading_term, leading_term_count = repeated_leading_term(entry_sung_text(entry))
        try:
            ctc_score = float(assignment.get("ctc_score", 1.0) or 1.0)
        except (TypeError, ValueError):
            ctc_score = 1.0
        repeated_onset_candidate: dict[str, object] | None = None
        if leading_term_count >= 2 and ctc_score <= 0.18 and isinstance(peaks, list):
            earlier_peaks: list[dict[str, float]] = []
            repeated_peak_floor = previous_time + 0.20
            if previous_ctc_token_end is not None:
                # A repeated-word posterior that still falls inside the prior
                # line's forced token tail is not the first occurrence of this
                # line.  This is the same cross-line leakage seen with raw ASR
                # and nearby phonetic peaks.
                repeated_peak_floor = max(repeated_peak_floor, previous_ctc_token_end + 0.03)
            for peak in peaks:
                if not isinstance(peak, dict):
                    continue
                try:
                    peak_time = float(peak.get("time"))
                    peak_score = float(peak.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if repeated_peak_floor < peak_time <= current_time - 0.45 and peak_score >= 0.005:
                    earlier_peaks.append({"time": peak_time, "score": peak_score})
            if earlier_peaks:
                strongest_score = max(peak["score"] for peak in earlier_peaks)
                credible_peaks = [
                    peak for peak in earlier_peaks
                    if peak["score"] >= max(0.005, strongest_score * 0.65)
                ]
                peak = min(credible_peaks, key=lambda item: item["time"])
                repeated_onset_candidate = {
                    "source": "ctc_repeated_leading_term_onset",
                    "time": round(peak["time"], 3),
                    # This is a deliberately narrow override: repeated visible
                    # text plus a strong earlier same-token posterior.
                    "score": 0.97,
                    "reasons": [
                        "repeated-leading-term-earlier-ctc-onset",
                        f"term={leading_term}",
                        f"repetitions={leading_term_count}",
                    ],
                }
                candidates.append(repeated_onset_candidate)

        penalties: list[dict[str, object]] = []
        flags: list[str] = []
        numeric_times = [float(candidate["time"]) for candidate in candidates]
        spread = max(numeric_times) - min(numeric_times) if len(numeric_times) > 1 else 0.0
        if spread >= 0.75:
            penalties.append({"kind": "candidate_disagreement", "value": 0.25, "seconds": round(spread, 3)})
            flags.append("candidate_disagreement")
        if current_time <= previous_time + 0.12 and index:
            penalties.append({"kind": "previous_line_tail_attachment", "value": 0.20})
            flags.append("previous_line_tail_attachment")
        if is_long and spread >= 1.25:
            penalties.append({"kind": "long_line_disagreement", "value": 0.18})
            flags.append("long_line_disagreement")
        if is_short and spread >= 0.45:
            penalties.append({"kind": "short_line_onset_uncertain", "value": 0.16})
            flags.append("short_line_onset_uncertain")
        first_mora = str(profile.get("first_mora", ""))
        if first_mora and isinstance(peaks, list) and peaks:
            peak_times = [float(item["time"]) for item in nearby_peaks if isinstance(item.get("time"), (int, float))]
            if peak_times and min(abs(current_time - time) for time in peak_times) > 0.35:
                penalties.append({"kind": "phonetic_anchor_disagreement", "value": 0.12, "first_mora": first_mora})
                flags.append("phonetic_anchor_disagreement")

        for candidate in candidates:
            candidate_time = float(candidate["time"])
            if not (previous_time + 0.05 < candidate_time < next_time - 0.05):
                candidate["score"] = round(max(0.0, float(candidate["score"]) - 0.35), 3)
                candidate.setdefault("reasons", []).append("outside-neighbor-bounds")
        ranked = sorted(candidates, key=lambda candidate: float(candidate["score"]), reverse=True)
        chosen = ranked[0]
        chosen_time = float(chosen["time"])
        total_penalty = sum(float(item["value"]) for item in penalties)
        confidence = max(0.0, min(1.0, float(chosen["score"]) - total_penalty))
        safe_to_replace = (
            chosen.get("source") != "selected_path"
            and float(chosen["score"]) >= 0.74
            and confidence >= 0.70
            and previous_time + 0.05 < chosen_time < next_time - 0.05
        )
        if safe_to_replace:
            revised[index] = chosen_time
            assignment["timestamp"] = round(chosen_time, 3)
        else:
            chosen = next(candidate for candidate in candidates if candidate.get("source") == "selected_path")
            chosen_time = current_time
        isolated_low_ctc_path = (
            ctc_score < 0.03
            and not flags
            and spread < 0.75
            and not (isinstance(risk, dict) and risk.get("review_required"))
        )
        raw_resolves_detached_ctc = (
            safe_to_replace
            and chosen.get("source") == "raw_asr"
            and detached_initial_ctc_token
            and parsed_raw_score >= 0.90
        )
        review_required = bool(risk.get("review_required")) if isinstance(risk, dict) else False
        # A lone weak CTC path remains untrusted in the report, but score alone
        # does not establish a timing defect. Require a concrete conflict before
        # escalating it to manual review.
        review_required = review_required or (
            confidence < 0.68 and not isolated_low_ctc_path
        ) or spread >= 1.25
        if raw_resolves_detached_ctc:
            review_required = False
        assignment["chosen_time"] = round(chosen_time, 3)
        assignment["confidence"] = round(confidence, 3)
        assignment["candidates"] = candidates
        assignment["rejected_candidates"] = [candidate for candidate in candidates if candidate is not chosen]
        assignment["reasons"] = list(chosen.get("reasons", []))
        assignment["penalties"] = penalties
        assignment["flags"] = flags
        assignment["review_required"] = review_required
        assignment["phonetic_anchor"] = {
            "system": profile.get("phonetic_system"),
            "first_mora": first_mora,
            "last_mora": profile.get("last_mora"),
        }
        if is_long and (spread >= 1.25 or "long_line_disagreement" in flags):
            midpoint = max(1, len(entry_sung_text(entry)) // 2)
            assignment["split_suggestion"] = {
                "reason": "long-line-multiple-phrase-candidates",
                "suggested_after_text": entry_sung_text(entry)[:midpoint],
            }
        if review_required and isinstance(risk, dict):
            risk["chosen_time"] = round(chosen_time, 3)
            risk["confidence"] = round(confidence, 3)
            risk["candidates"] = candidates
            risk["rejected_candidates"] = assignment["rejected_candidates"]
            risk["reasons"] = assignment["reasons"]
            risk["penalties"] = penalties
            risk["flags"] = list(dict.fromkeys([*(risk.get("flags") if isinstance(risk.get("flags"), list) else []), *flags]))
            risk["review_required"] = True
        elif raw_resolves_detached_ctc and isinstance(risk, dict):
            risk["review_required"] = False
            risk["severity"] = "resolved"
            risk["resolution"] = "high-confidence-raw-resolves-detached-ctc-initial-token"
            risk_flags = risk.get("flags")
            if isinstance(risk_flags, list):
                risk["flags"] = list(dict.fromkeys([*risk_flags, "detached_ctc_initial_token_resolved"]))
    for index in range(1, len(revised)):
        revised[index] = max(revised[index], revised[index - 1] + 0.05)
    if not isinstance(risks, list):
        risks = []
        report["suspicious_alignments"] = risks
    existing_entries = {
        int(item.get("entry"))
        for item in risks
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict) or not assignment.get("review_required"):
            continue
        entry_number = index + 1
        if entry_number in existing_entries:
            continue
        risks.append(
            {
                "entry": entry_number,
                "text": entry_sung_text(entries[index]),
                "flags": assignment.get("flags", []),
                "severity": "medium" if float(assignment.get("confidence", 0.0) or 0.0) >= 0.45 else "high",
                "review_required": True,
                "chosen_time": assignment.get("chosen_time"),
                "confidence": assignment.get("confidence"),
                "candidates": assignment.get("candidates", []),
                "rejected_candidates": assignment.get("rejected_candidates", []),
                "reasons": assignment.get("reasons", []),
                "penalties": assignment.get("penalties", []),
            }
        )
    report["suspicious_alignment_count"] = len(risks)
    report["review_required_count"] = sum(
        1 for item in risks if isinstance(item, dict) and item.get("review_required")
    )
    report["review_required"] = bool(report["review_required_count"])
    report["suspicious_alignment_severity_counts"] = {
        "high": sum(1 for item in risks if isinstance(item, dict) and item.get("severity") == "high"),
        "medium": sum(1 for item in risks if isinstance(item, dict) and item.get("severity") == "medium"),
        "low": sum(1 for item in risks if isinstance(item, dict) and item.get("severity") == "low"),
    }
    update_report_confidence_metrics(report)
    return revised


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
    protected_entries = {
        int(item.get("entry"))
        for item in report.get("long_segment_repairs", [])
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    for index in range(1, min(len(refined), len(forced_segments), len(assignments))):
        if index + 1 in protected_entries:
            continue
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

    # First-pass bounded local re-alignment: when a candidate disagrees with the
    # forced first character by more than 1.5s, only trust the forced time if the
    # neighboring lyric anchors are already trusted and bound the search window.
    local_window_changes: list[dict[str, object]] = []
    local_window_enabled = bool(report.get("enable_local_window_realign", False))
    for index in range(1, min(len(refined) - 1, len(forced_segments), len(assignments) - 1)):
        if not local_window_enabled:
            break
        if index + 1 in protected_entries:
            continue
        current_assignment = assignments[index]
        previous_assignment = assignments[index - 1]
        next_assignment = assignments[index + 1]
        if (
            not isinstance(current_assignment, dict)
            or not isinstance(previous_assignment, dict)
            or not isinstance(next_assignment, dict)
        ):
            continue
        if assignment_score(previous_assignment) < TRUSTED_ALIGNMENT_SCORE:
            continue
        if assignment_score(next_assignment) < TRUSTED_ALIGNMENT_SCORE:
            continue
        forced_time = first_forced_char_time(forced_segments[index])
        current_time = refined[index]
        current_score = assignment_score(current_assignment)
        if current_score >= TRUSTED_ALIGNMENT_SCORE:
            continue
        if (
            report.get("asr_segment_timing_source") == "whispercpp_raw"
            and current_score >= 0.80
            and forced_time < current_time
            and not (current_time - forced_time >= 5.0 and forced_time - refined[index - 1] >= 4.0)
        ):
            continue
        if abs(current_time - forced_time) <= 1.5:
            continue
        if current_score >= 0.90 and abs(current_time - forced_time) > 3.0:
            continue
        left_bound = refined[index - 1] + 0.10
        right_bound = refined[index + 1] - 0.10
        if not (left_bound < forced_time < right_bound):
            continue
        refined[index] = forced_time
        local_window_changes.append(
            {
                "entry": index + 1,
                "from": round(current_time, 3),
                "to": round(forced_time, 3),
                "reason": "bounded-local-window-forced",
                "left_anchor_entry": index,
                "right_anchor_entry": index + 2,
                "lyric": entries[index].lines[0] if entries[index].lines else "",
            }
        )
    changes.extend(local_window_changes)

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

    artist, title = audio_track_labels(audio_path)
    title_label = " ".join(value for value in (artist, title) if value).strip()
    title_card = f"{artist} - {title}" if artist else title
    body = [
        f"[ti:{title_label or audio_path.stem}]",
        f"[re:{generator}]",
        f"[length:{format_lrc_time(duration)}]",
        "[by:LRC tools]",
        f"[00:00.00]{title_card}",
    ]
    for time, entry in zip(timestamps, entries):
        tag = format_lrc_time(time)
        body.extend(f"[{tag}]{line}" for line in entry.lines)
    output_path.write_text("\n".join(body) + "\n", encoding="utf-8-sig")


def report_artifact_paths(audio_path: Path, output_path: Path, report_dir: Path) -> tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    identity = f"{audio_path.resolve()}\0{output_path.resolve()}".casefold().encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:10]
    stem = f"{audio_path.stem}--{digest}"
    return (
        report_dir / f"{stem}.align-report.json",
        report_dir / f"{stem}.review-audit.md",
        report_dir / f"{stem}.anchor-template.lrc",
    )


def write_alignment_report(report_path: Path, report: dict[str, object]) -> None:
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_review_audit(
    output_path: Path, report: dict[str, object], audit_path: Path, review_only: bool = True
) -> Path:
    rows = build_audit_rows(output_path, report, review_only=review_only)
    write_audit_markdown(rows, report, audit_path)
    return audit_path


def write_review_anchor_template(
    output_path: Path, report: dict[str, object], template_path: Path, review_only: bool = True
) -> Path:
    rows = build_audit_rows(output_path, report, review_only=review_only)
    write_anchor_template(rows, template_path)
    return template_path


def strict_review_failure_reason(report: dict[str, object], args: argparse.Namespace) -> str | None:
    review_count = int(report.get("review_required_count", 0) or 0)
    trusted_percent = float(report.get("trusted_percent", report.get("matched_percent", 0.0)) or 0.0)
    min_trusted = args.min_trusted_percent
    if args.strict_review and min_trusted is None:
        min_trusted = 100.0
    if (args.strict_review or args.fail_on_review_required) and review_count > 0:
        return f"review_required_count={review_count}"
    if min_trusted is not None and trusted_percent < float(min_trusted):
        return f"trusted_percent={trusted_percent} < required {float(min_trusted)}"
    if args.strict_review and report.get("collapse_detected"):
        return "collapse_detected=true"
    return None


def process_audio(audio_path: Path, args: argparse.Namespace) -> Path:
    if not audio_path.exists():
        raise LrcError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".flac":
        raise LrcError(f"Only .flac is supported in v0: {audio_path}")

    lyrics_path = Path(args.lyrics) if args.lyrics else find_lyrics(audio_path)
    lyrics = load_lyrics(lyrics_path)
    entries = remove_generated_title_cards(lyrics.entries, audio_path)
    requested_whisper_language = args.whisper_language
    if args.whisper_language == "auto":
        args = copy.copy(args)
        args.whisper_language = infer_spoken_language(entries)
    anchor_path = find_anchor_hints(audio_path, args)
    duration = probe_duration(audio_path)
    timing_source = args.timing_source
    if timing_source == "audio":
        timing_source = "heuristic"
        print(
            "WARN: --timing-source audio is now treated as experimental heuristic timing.",
            file=sys.stderr,
        )

    if timing_source == "auto":
        if lyrics.saw_timestamps and not args.ignore_lyric_timestamps:
            timing_source = "lyrics"
        elif not args.no_checked_lrc_hint and (
            checked_hint := checked_lrc_timing_hint(audio_path, entries, lyrics_path)
        ):
            entries, timestamps, report = checked_hint
            timing_source = "checked_lrc"
        elif default_ctc_ready() and default_whisperx_ready():
            timing_source = "backend_competition"
        elif default_ctc_ready() or default_whisperx_ready():
            timing_source = "backend_competition"
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
        report = {
            "backend": "lyrics",
            "timing_entries": len(entries),
            "assigned_entries": len(entries),
            "assigned_percent": 100.0,
            "trusted_entries": len(entries),
            "trusted_percent": 100.0,
            "matched_entries": len(entries),
            "matched_percent": 100.0,
            "low_confidence_count": 0,
            "low_confidence_percent": 0.0,
            "review_required_count": 0,
            "review_required_percent": 0.0,
            "review_required": False,
            "note": "Used timestamped lyric input as the timing source.",
        }
    elif timing_source == "checked_lrc":
        features = None
        strategy = "checked-lrc-hint"
    elif timing_source == "whispercpp":
        segments = run_whispercpp(audio_path, args)
        timestamps, report = match_whisper_segments(entries, segments, duration)
        features = None
        strategy = "whispercpp-experimental"
    elif timing_source == "ctc":
        timestamps, report = run_ctc_alignment(audio_path, entries, duration, args)
        timestamps, report, _ = apply_ctc_crossline_initial_recovery(timestamps, report, duration)
        timestamps, report, _ = apply_ctc_weak_prefix_recovery(timestamps, report, duration)
        features = None
        strategy = "ctc-forced-align"
    elif timing_source == "jactc":
        timestamps, report = run_japanese_ctc_alignment(audio_path, entries, duration, args)
        features = None
        strategy = "japanese-ctc-forced-align-experimental"
    elif timing_source == "whisperx":
        timestamps, report = run_whisperx_best_candidate(audio_path, entries, duration, args)
        features = None
        strategy = "whisperx-hybrid-experimental"
    elif timing_source == "backend_competition":
        timestamps, report, selected_backend = run_auto_backend_competition(audio_path, entries, duration, args)
        features = None
        timing_source = selected_backend
        strategy = str(report.get("strategy") or "auto-candidate-selection")
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
    if report is None:
        report = {}
    timestamps, anchor_changes = apply_anchor_hints(entries, timestamps, report, anchor_path, duration)
    if anchor_changes:
        print(
            f"INFO: applied {len(anchor_changes)} anchor hint(s) from {anchor_path}.",
            file=sys.stderr,
        )
    if timing_source not in {"lyrics", "checked_lrc", "heuristic"}:
        timestamps = score_line_timing_candidates(entries, timestamps, report, duration)
        sync_report_assignment_timestamps(report, timestamps)
    if args.probe:
        identity = str(audio_path.resolve()).casefold().encode("utf-8")
        suffix = hashlib.sha256(identity).hexdigest()[:10]
        DEFAULT_PROBE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_PROBE_DIR / f"{audio_path.stem}--{suffix}.probe.lrc"
    else:
        output_path = Path(args.output).expanduser().resolve() if args.output else audio_path.with_suffix(".lrc")
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else DEFAULT_REPORT_DIR
    report_path, audit_path, template_path = report_artifact_paths(audio_path, output_path, report_dir)
    metadata_lines = None
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
    report.setdefault("backend", timing_source)
    report["requested_timing_source"] = args.timing_source
    report["resolved_timing_source"] = timing_source
    report["strategy"] = strategy
    report["mode"] = timing_source
    report["heuristic_mode"] = args.mode
    report["audio_path"] = str(audio_path)
    report["lyrics_path"] = str(lyrics_path)
    report["output_path"] = str(output_path)
    report["report_path"] = str(report_path)
    report["duration_seconds"] = round(duration, 3)
    report["timing_entries"] = len(entries)
    report["display_lines"] = sum(len(entry.lines) for entry in entries)
    report["requested_whisper_language"] = requested_whisper_language
    report["resolved_whisper_language"] = args.whisper_language
    if skipped_markers:
        report["skipped_marker_entries"] = skipped_markers
        report["skipped_marker_count"] = len(skipped_markers)
    write_alignment_report(report_path, report)

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
    if report.get("review_required"):
        print(
            f"WARN: alignment report has {report.get('suspicious_alignment_count', 0)} suspicious line(s), "
            f"{report.get('review_required_count', 0)} review-required. "
            "Inspect suspicious_alignments before trusting output.",
            file=sys.stderr,
        )
    if strict_reason := strict_review_failure_reason(report, args):
        review_only = not strict_reason.startswith("trusted_percent=")
        audit_path = write_review_audit(output_path, report, audit_path, review_only=review_only)
        template_path = write_review_anchor_template(output_path, report, template_path, review_only=review_only)
        report["strict_failure_reason"] = strict_reason
        report["review_audit_path"] = str(audit_path)
        report["anchor_template_path"] = str(template_path)
        write_alignment_report(report_path, report)
        raise LrcError(
            f"strict review failed for {output_path}: {strict_reason}. "
            f"Draft LRC, report, review audit, and anchor template were written; inspect {audit_path}, "
            f"{template_path}, or {report_path}."
        )
    print(
        f"OK: {output_path} ({len(entries)} timing entries, "
        f"{sum(len(entry.lines) for entry in entries)} display lines, {duration:.2f}s, "
        f"{len(features.segments) if features else 'n/a'} detected segments, "
        f"timing_source={timing_source}, heuristic_mode={args.mode}, strategy={strategy})"
    )
    print(f"REPORT: {report_path}")
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
        "--probe",
        action="store_true",
        help="Write a disposable candidate to project outputs/probes instead of touching the same-name LRC.",
    )
    parser.add_argument(
        "--report-dir",
        help="Directory for JSON reports and strict-review audit artifacts. Default: project outputs/reports.",
    )
    parser.add_argument(
        "--mode",
        choices=["energy", "even"],
        default="energy",
        help="energy: waveform/onset snapping; even: weighted rough timing only.",
    )
    parser.add_argument(
        "--timing-source",
        choices=["auto", "lyrics", "ctc", "jactc", "whisperx", "whispercpp", "heuristic", "audio"],
        default="auto",
        help=(
            "auto: preserve checked lyric timestamps when present, otherwise run CTC and "
            "WhisperX candidates and select by report quality; lyrics: preserve timestamps "
            "from timestamped lyric input; ctc: MMS/CTC forced alignment over known lyrics; "
            "jactc: experimental Japanese Wav2Vec2 CTC candidate; "
            "whispercpp: experimental ASR matching; "
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
        default="auto",
        help="Spoken language passed to whisper.cpp, default: auto from lyric text.",
    )
    parser.add_argument(
        "--vocal-onset-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use GPU Demucs vocal-stem onset evidence to resolve large CTC/WhisperX disagreements.",
    )
    parser.add_argument(
        "--vocal-ctc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run primary CTC forced alignment on a cached Demucs vocal stem; use --no-vocal-ctc for full-mix fallback.",
    )
    parser.add_argument(
        "--no-checked-lrc-hint",
        action="store_true",
        help="Testing/benchmark option: in auto mode, skip checked same-stem LRC hints and force backend selection.",
    )
    parser.add_argument(
        "--ignore-lyric-timestamps",
        action="store_true",
        help="Testing/recheck option: treat timestamped lyric input as untimed text in auto mode.",
    )
    parser.add_argument(
        "--anchor-hints",
        help=(
            "Optional timestamped LRC subset with manually checked line anchors. "
            "Omit to auto-detect <song>.anchors.lrc/.txt."
        ),
    )
    parser.add_argument(
        "--no-anchor-hints",
        action="store_true",
        help="Do not auto-detect same-folder manual anchor hint files.",
    )
    parser.add_argument(
        "--fail-on-review-required",
        action="store_true",
        help="Exit non-zero after writing outputs if the report contains any review-required line.",
    )
    parser.add_argument(
        "--min-trusted-percent",
        type=float,
        default=None,
        help="Exit non-zero after writing outputs unless trusted_percent is at least this value.",
    )
    parser.add_argument(
        "--strict-review",
        action="store_true",
        help=(
            "Production gate: fail on review-required lines, collapse, or trusted_percent below 100. "
            "Draft outputs are still written for inspection."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output and len(args.audio) != 1:
        parser.error("--output can only be used with exactly one audio file")
    if args.probe and args.output:
        parser.error("--probe cannot be combined with --output")
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
