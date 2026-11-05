"""Extracted from auto_lrc.py (split step 3): lyric loading/preflight, text normalisation, language inference, LRC output and review audit."""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence
import numpy as np
import lrc_timestamp_pairs
from export_alignment_audit import build_rows as build_audit_rows
from export_alignment_audit import write_anchor_template
from export_alignment_audit import write_markdown as write_audit_markdown
from audio_features import AudioFeatures
from runtime_base import CanonicalWrittenTime, LrcError, _timing_content_digest, canonicalize_lrc_time, run_command
from runtime_base import _timing_spans_from_value
import lrc_stage_timings


LRC_ALGORITHM_REVISION = "2026-10-25-lrc-v2.0-segment-realign"


_BACKEND_PROVENANCE_PROJECTION_KEY = "backend_provenance_projection"


_RUNTIME_LYRIC_ENTRIES_KEY = "_runtime_lyric_entries"


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


@lru_cache(maxsize=1)
def optional_kakasi_converter() -> object | None:
    try:
        import pykakasi  # type: ignore[import-not-found]
    except Exception:
        return None
    return pykakasi.kakasi()


def audio_track_labels(audio_path: Path) -> tuple[str, str]:
    """Read display-safe artist/title labels from FLAC metadata when present.

    The labels only drive title-card removal, so a missing ffprobe degrades to
    the file stem exactly like a failed probe instead of aborting the run.
    """
    try:
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
    except LrcError:
        return "", audio_path.stem
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
    return artist, title or audio_path.stem


def remove_generated_title_cards(entries: list[LyricEntry], audio_path: Path) -> list[LyricEntry]:
    """Remove only a matching zero-time Artist - Title card.

    Hand-made LRCs also write the card as Title - Artist (c07, c10 of the
    corpus); both orders must match the file's own tags exactly, so a sung
    line that merely contains " - " is never removed.
    """
    artist, title = audio_track_labels(audio_path)
    normalized_title = normalize_match_text(title).casefold()
    normalized_artist = normalize_match_text(artist).casefold()
    kept: list[LyricEntry] = []
    for entry in entries:
        text = entry_sung_text(entry).strip()
        card = re.fullmatch(r"(.+?)\s+-\s+(.+)", text)
        left = normalize_match_text(card.group(1)).casefold() if card else ""
        right = normalize_match_text(card.group(2)).casefold() if card else ""
        artist_title = right == normalized_title and (
            not normalized_artist or left == normalized_artist
        )
        title_artist = bool(normalized_artist) and (
            left == normalized_title and right == normalized_artist
        )
        is_title_card = (
            entry.source_time_cs == 0
            and len(entry.lines) == 1
            and card is not None
            and bool(normalized_title)
            and (artist_title or title_artist)
        )
        if not is_title_card:
            kept.append(entry)
    return kept


LRC_TIMESTAMP_RE = re.compile(r"\[([0-9]{1,3}):([0-9]{2})(?:\.([0-9]{1,3}))?\]")


LRC_LINE_RE = re.compile(r"^\s*((?:\[[0-9]{1,3}:[0-9]{2}(?:\.[0-9]{1,3})?\])+)(.*)$")


def lrc_time_key(tag_block: str) -> int | None:
    match = LRC_TIMESTAMP_RE.search(tag_block)
    if not match:
        return None
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    fraction = match.group(3) or "0"
    milliseconds = int(fraction.ljust(3, "0"))
    centiseconds = (milliseconds + 5) // 10
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
    """Merge source/translation pairs only when they dominate the whole file."""
    pair_starts: list[int] = []
    index = 0
    while index + 1 < len(entries):
        current, following = entries[index], entries[index + 1]
        is_pair = (
            len(current.lines) == len(following.lines) == 1
            and current.source_time_cs is None
            and following.source_time_cs is None
            and lyric_line_script(current.lines[0]) in {"japanese", "latin"}
            and lyric_line_script(following.lines[0]) == "han"
        )
        if is_pair:
            pair_starts.append(index)
            index += 2
        else:
            index += 1
    if len(pair_starts) * 20 <= len(entries) * 6:
        return entries
    starts = set(pair_starts)
    merged: list[LyricEntry] = []
    index = 0
    while index < len(entries):
        if index in starts:
            merged.append(LyricEntry([
                entries[index].lines[0], entries[index + 1].lines[0]
            ]))
            index += 2
        else:
            merged.append(entries[index])
            index += 1
    return merged


def load_lyrics(path: Path, skip_comments: bool = False) -> LyricDocument:
    text = path.read_text(encoding="utf-8-sig")
    entries: list[LyricEntry] = []
    metadata_lines: list[str] = []
    saw_timestamps = False

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if skip_comments and stripped.startswith("#"):
            continue
        # User-curated BK files can carry this bare provenance marker.
        # Never send it as an untimed sung line to forced alignment.
        if stripped.casefold() == "[oogg_rana]" or re.fullmatch(r"\[[a-zA-Z]+:.*\]", stripped):
            metadata_lines.append(stripped)
            continue

        lrc_match = LRC_LINE_RE.match(raw)
        if lrc_match:
            saw_timestamps = True
            display_lines = split_group_text(lrc_match.group(2), preserve_single=True)
            if not display_lines:
                continue
            for tag in LRC_TIMESTAMP_RE.finditer(lrc_match.group(1)):
                time_key = lrc_time_key(tag.group(0))
                if time_key is not None:
                    entries.append(LyricEntry(list(display_lines), time_key))
            continue

        display_lines = split_group_text(stripped)
        if not display_lines:
            continue
        entries.append(LyricEntry(display_lines))

    if saw_timestamps:
        # Sort only timed entries, keeping untimed rows in their original slots.
        timed = iter(sorted(
            (entry for entry in entries if entry.source_time_cs is not None),
            key=lambda entry: entry.source_time_cs,
        ))
        entries = [
            next(timed) if entry.source_time_cs is not None else entry
            for entry in entries
        ]
    else:
        entries = merge_adjacent_translation_lines(entries)
    if not entries:
        raise LrcError(f"No lyric lines found in {path}")
    if saw_timestamps:
        print(
            "INFO: timestamped lyric input detected; timestamps are available for grouping "
            "or --timing-source lyrics.",
            file=sys.stderr,
        )
    if saw_timestamps:
        entries = lrc_timestamp_pairs.merge_timestamped_pairs(entries)
    return LyricDocument(entries=entries, metadata_lines=metadata_lines, saw_timestamps=saw_timestamps)


def build_lyric_preflight(
    lyrics_path: Path,
    document: LyricDocument,
    kept_entries: list[LyricEntry],
    audio_path: Path,
) -> dict[str, object]:
    """Audit lyric input transformations without modifying the parsed entries."""
    raw_lines = lyrics_path.read_text(encoding="utf-8-sig").splitlines()
    transformations: list[dict[str, object]] = []
    unsupported: list[dict[str, object]] = []
    garbage: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(raw_lines, 1):
        stripped = raw_line.strip()
        if not stripped:
            transformations.append({"kind": "empty-line-ignored", "line": line_number})
            continue
        match = LRC_LINE_RE.match(stripped)
        content = match.group(2) if match else stripped
        if match:
            count = len(list(LRC_TIMESTAMP_RE.finditer(match.group(1))))
            if count > 1:
                transformations.append({
                    "kind": "multi-timestamp-expanded", "line": line_number,
                    "copies": count,
                })
        if len(split_group_text(content, preserve_single=True)) > 1:
            transformations.append({"kind": "display-lines-split", "line": line_number})
        if is_instrumental_marker(LyricEntry([content])):
            transformations.append({"kind": "instrumental-marker-removed", "line": line_number})
        if not normalize_match_text(content) and not any(ch.isalpha() for ch in content):
            transformations.append({"kind": "normalized-empty-removed", "line": line_number})
        if "�" in content or re.search(r"\?{3,}", content) or re.search(
            r"心得|推薦|樓主|作詞|作曲|編曲|翻譯|http|bilibili|BV[0-9A-Za-z]{8,}",
            content, flags=re.IGNORECASE,
        ):
            garbage.append({"line": line_number, "text": content[:160]})
    raw_times = [lrc_time_key(tag.group(0)) for line in raw_lines for tag in LRC_TIMESTAMP_RE.finditer(line)]
    duplicate_times = {time for time, count in collections.Counter(raw_times).items() if time is not None and count > 1}
    kept_ids = {id(entry) for entry in kept_entries}
    for index, entry in enumerate(document.entries, 1):
        if entry.source_time_cs in duplicate_times and len(entry.lines) > 1:
            transformations.append({"kind": "translation-pair-merged", "entry": index})
        if id(entry) not in kept_ids:
            transformations.append({
                "kind": "generated-title-removed", "entry": index,
                "text": entry_sung_text(entry)[:160],
            })
    _, title = audio_track_labels(audio_path)
    if (
        document.entries
        and document.entries[0].source_time_cs is None
        and id(document.entries[0]) in kept_ids
        and normalize_match_text(entry_sung_text(document.entries[0]))
        == normalize_match_text(title)
    ):
        transformations.append({"kind": "plain-title-preserved", "entry": 1})
    if not document.saw_timestamps:
        plain = [line.strip() for line in raw_lines if line.strip()]
        neighbors = set(zip(plain, plain[1:]))
        for index, entry in enumerate(document.entries, 1):
            if len(entry.lines) == 2 and tuple(entry.lines) in neighbors:
                transformations.append({
                    "kind": "translation-pair-merged", "entry": index,
                    "text": list(entry.lines),
                })
    for index, entry in enumerate(kept_entries, 1):
        for line in entry.lines:
            if not normalize_match_text(line) and any(ch.isalpha() for ch in line):
                unsupported.append({
                    "entry": index, "text": line[:160], "code": "UNSUPPORTED_SCRIPT",
                })
    reason_codes = []
    if unsupported:
        reason_codes.append("UNSUPPORTED_SCRIPT")
    if garbage:
        reason_codes.append("SUSPECT_GARBAGE")
    return {
        "status": "INPUT_SUSPECT" if reason_codes else "OK",
        "reason_codes": reason_codes,
        "raw_line_count": len(raw_lines),
        "parsed_entries": len(document.entries),
        "retained_entries": len(kept_entries),
        "transformations": transformations,
        "unsupported_script_entries": unsupported,
        "garbage_entries": garbage,
        "raw_asr_coverage": "NOT_AVAILABLE",
    }


def raw_asr_lyric_coverage(
    entries: list[LyricEntry],
    raw_segments: list[AsrSegment] | None,
) -> dict[str, object]:
    """Measure independent raw-ASR lyric text coverage without rerunning ASR."""
    if not raw_segments:
        return {"status": "NOT_AVAILABLE"}
    transcript = normalize_match_text(" ".join(segment.text for segment in raw_segments))
    lines = [
        normalize_match_text(entry_sung_text(entry))
        for entry in entries
    ]
    lines = [line for line in lines if line]
    if not transcript or not lines:
        return {"status": "NOT_AVAILABLE"}
    positions = [transcript.find(line) for line in lines]
    matched = sum(pos >= 0 for pos in positions)
    fraction = matched / len(lines)
    observed = [pos for pos in positions if pos >= 0]
    order_reversals = sum(b < a for a, b in zip(observed, observed[1:]))
    return {
        "status": (
            "LOW_COVERAGE" if fraction < 0.50 else
            "ORDER_INCONSISTENT" if order_reversals > 3 else "OK"
        ),
        "matched_lines": matched,
        "order_reversals": order_reversals,
        "total_lines": len(lines),
        "coverage_percent": round(fraction * 100.0, 2),
        "source": "existing-raw-whispercpp",
        "diagnostic_only": True,
    }


def find_lyrics(audio_path: Path, *, include_checked: bool = False) -> Path:
    candidates = [
        audio_path.with_suffix(".lyrics.lrc"),
        audio_path.with_suffix(".lyrics.txt"),
        audio_path.with_suffix(".txt"),
        audio_path.with_name(f"{audio_path.stem}.lyrics"),
    ]
    if include_checked:
        candidates.extend(checked_lrc_candidates(audio_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pretty = ", ".join(candidate.name for candidate in candidates)
    raise LrcError(f"No lyric text found for {audio_path.name}. Expected one of: {pretty}")


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
    # Anchor files are assisted evidence. Their presence beside an input must
    # never alter independent auto inference; only an explicit positive CLI
    # path can opt into them.
    return None


def strip_lyric_timestamps(entries: list[LyricEntry]) -> list[LyricEntry]:
    """Retain ordered lyric text while removing all source timing authority."""
    return [LyricEntry(list(entry.lines)) for entry in entries]


def inference_mode_metadata(
    requested_timing_source: str,
    resolved_timing_source: str,
    anchor_path: Path | None,
) -> dict[str, object]:
    assisted_reference = requested_timing_source in {"lyrics", "checked"}
    assisted_anchor = anchor_path is not None
    independent_auto = requested_timing_source == "auto" and not assisted_anchor
    if independent_auto:
        mode = "independent-auto"
    elif requested_timing_source == "auto":
        mode = "assisted-auto-anchor"
    elif requested_timing_source == "checked":
        mode = "assisted-checked"
    elif requested_timing_source == "lyrics":
        mode = "assisted-lyrics"
    else:
        mode = f"explicit-{requested_timing_source}"
    payload = {
        "schema_version": 1,
        "requested_timing_source": requested_timing_source,
        "resolved_timing_source": resolved_timing_source,
        "inference_mode": mode,
        "independent_auto": independent_auto,
        "reference_used_before_inference": assisted_reference,
        "experience_artifact_used": bool(assisted_reference or assisted_anchor),
    }
    return {
        **payload,
        "inference_mode_revision": _timing_content_digest({
            "algorithm_revision": LRC_ALGORITHM_REVISION,
            **payload,
        }),
    }


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
            # The user-maintained BK directory contains already manually
            # corrected references. Only the explicitly selected checked
            # timing path calls this candidate discovery; auto never does.
            candidates.append(parent / "LRC tools checked references" / "LRC bk" / f"{audio_path.stem}.lrc")
            candidates.append(parent / f"{audio_path.stem}.lrc")
            # Human-reviewed references live outside album folders so normal
            # drag-and-drop runs do not accumulate generated LRC files beside
            # the music library. Treat this as a first-class lyric source.
            candidates.append(parent / "LRC tools checked references" / f"{audio_path.stem}.lrc")
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
        checked_entries = remove_generated_title_cards(checked_entries, audio_path)
        checked_signature = [normalize_match_text(entry_sung_text(entry)) for entry in checked_entries]
        if target_signature != checked_signature:
            continue
        report: dict[str, object] = {
            "backend": "checked-lrc-hint",
            "checked_lrc": str(candidate),
            "lyric_source": str(lyrics_path),
            "timing_provenance": (
                "human-reviewed-bk" if candidate.parent.name.casefold() == "lrc bk"
                else "checked-lrc"
            ),
            "uses_reference_timestamps": True,
            "independent_auto_benchmark_eligible": False,
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


def lyric_identity_revision(entries: list[LyricEntry]) -> str:
    """Digest the exact immutable sung-text sequence used by all timing evidence."""
    return _timing_content_digest(
        [unicodedata.normalize("NFKC", entry_sung_text(entry)) for entry in entries]
    )


def assert_lyric_identity_contract(
    entries: list[LyricEntry],
    report: dict[str, object],
    *,
    initialize: bool = False,
) -> str:
    """Fail closed if any report-side phonetic anchor no longer matches source lyrics."""
    revision = lyric_identity_revision(entries)
    existing = report.get("lyric_entry_revision")
    if existing is None:
        if initialize:
            report["lyric_entry_revision"] = revision
        else:
            raise LrcError("Missing immutable lyric-entry revision")
    elif existing != revision:
        raise LrcError("Lyric content changed after timing evidence was generated")

    suspicious = report.get("suspicious_alignments")
    if isinstance(suspicious, list):
        for row in suspicious:
            if not isinstance(row, dict):
                continue
            entry_number = row.get("entry")
            profile = row.get("anchor_profile")
            if not isinstance(entry_number, int) or not isinstance(profile, dict):
                continue
            if not 1 <= entry_number <= len(entries):
                raise LrcError(f"Anchor profile entry {entry_number} is outside lyric range")
            profile_text = profile.get("entry_text")
            if not isinstance(profile_text, str):
                continue
            expected = unicodedata.normalize("NFKC", entry_sung_text(entries[entry_number - 1]))
            observed = unicodedata.normalize("NFKC", profile_text)
            if observed != expected:
                raise LrcError(
                    f"Phonetic anchor text diverged at entry {entry_number}: "
                    f"expected {expected!r}, got {observed!r}"
                )
    return revision


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


def _is_single_cjk_or_kana_term(term: str) -> bool:
    """Whether a one-codepoint visible term can be a Japanese/CJK word."""
    if len(term) != 1:
        return False
    codepoint = ord(term)
    return bool(
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x3040 <= codepoint <= 0x309F
        or 0x30A0 <= codepoint <= 0x30FF
    )


def repeated_leading_term(text: str) -> tuple[str, int]:
    """Return a visibly repeated opening term, without guessing word breaks."""
    terms = [
        normalize_match_text(term)
        for term in re.split(r"[\s\u3000,、，。！？!?・]+", unicodedata.normalize("NFKC", text).strip())
    ]
    terms = [term for term in terms if term]
    if len(terms) < 2:
        return "", 0
    first = terms[0]
    if len(first) < 2 and not _is_single_cjk_or_kana_term(first):
        return "", 0
    repeated = 1
    for term in terms[1:]:
        if term.startswith(first):
            repeated += 1
        else:
            break
    return (first, repeated) if repeated >= 2 else ("", 0)


def has_ambiguous_lyric_prefix(entries: list[LyricEntry], index: int) -> bool:
    """Detect long shared prefixes or suffixes that make raw-ASR line matching ambiguous.

    A verse variant that changes only the opening words ("I'm living without
    you" / "I'm dreaming without you") is as easy to match to the wrong line as
    one that changes only the ending.
    """
    if not (0 <= index < len(entries)):
        return False
    target = normalize_match_text(entry_sung_text(entries[index]))
    if len(target) < 12:
        return False

    def shared_run(left_text: str, right_text: str) -> int:
        shared = 0
        for left, right in zip(left_text, right_text):
            if left != right:
                break
            shared += 1
        return shared

    for other_index, other_entry in enumerate(entries):
        if other_index == index:
            continue
        other = normalize_match_text(entry_sung_text(other_entry))
        if len(other) < 12:
            continue
        # Twelve shared characters (or most of a shorter long phrase) makes a
        # raw segment match non-unique. It may identify the right *phrase*
        # while still selecting the wrong occurrence in the song.
        required = min(12, max(8, min(len(target), len(other)) // 2))
        if (
            shared_run(target, other) >= required
            or shared_run(target[::-1], other[::-1]) >= required
        ):
            return True
    return False


def has_short_duplicate_context(entries: list[LyricEntry], index: int, window: int = 8) -> bool:
    """Return true when a short line can be confused with a nearby longer line."""
    if not (0 <= index < len(entries)):
        return False
    target = normalize_match_text(entry_sung_text(entries[index]))
    if len(target) < 4 or len(target) > 12:
        return False
    start = max(0, index - window)
    end = min(len(entries), index + window + 1)
    for other_index in range(start, end):
        if other_index == index:
            continue
        other = normalize_match_text(entry_sung_text(entries[other_index]))
        if len(other) <= len(target):
            continue
        if target in other:
            return True
    return False


def infer_spoken_language(entries: list[LyricEntry]) -> str:
    # A bilingual opening or an instrumental title card must not determine
    # Whisper language for the rest of the song. Use the full sung transcript.
    sample = "".join(entry_sung_text(entry) for entry in entries)
    if not sample:
        return "ja"
    latin_count = sum(1 for ch in sample if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    kana_count = sum(1 for ch in sample if "\u3040" <= ch <= "\u30ff")
    cjk_count = sum(1 for ch in sample if "\u3400" <= ch <= "\u9fff")
    total_text = max(1, latin_count + kana_count + cjk_count)
    # Romanized English refrains can dominate character count despite the
    # song being predominantly Japanese by lyric line. Use the entire line
    # distribution to avoid switching such mixed-language songs to English.
    sung_lines = [entry_sung_text(e) for e in entries if entry_sung_text(e).strip()]
    english_lines = sum(
        sum("a" <= c.lower() <= "z" for c in line)
        > 1.3 * sum("\u3040" <= c <= "\u30ff" or "\u3400" <= c <= "\u9fff" for c in line)
        for line in sung_lines
    )
    if latin_count / total_text >= 0.55 and (
        kana_count == 0 and cjk_count == 0
        or english_lines >= max(1, math.ceil(.65 * len(sung_lines)))
    ):
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


def is_instrumental_marker(entry: LyricEntry) -> bool:
    text = (entry.lines[0] if entry.lines else entry.alignment_text).strip()
    normalized = normalize_match_text(text)
    marker_text = text.strip("()[]{}（）【】［］").strip().casefold()
    marker_words = ("instrumental", "intro", "interlude", "outro", "間奏", "イントロ", "アウトロ")
    bracket_pairs = (("(", ")"), ("[", "]"), ("{", "}"), ("（", "）"), ("【", "】"), ("［", "］"))
    wrapped = any(text.startswith(a) and text.endswith(b) for a, b in bracket_pairs)
    return (
        text.casefold() == "[oogg_rana]"
        or text == "♪"
        or (not normalized and not any(char.isalpha() for char in text))
        or (wrapped and any(word in marker_text for word in marker_words))
    )


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


def first_forced_char_time(segment: AsrSegment) -> float:
    if segment.chars:
        return segment.chars[0][1]
    return segment.start


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


def format_lrc_time(seconds: float) -> str:
    return canonicalize_lrc_time(seconds).lrc_tag


SOURCE_CLOCK_METADATA_KEYS = frozenset({"offset", "length"})


SOURCE_GENERATOR_METADATA_KEYS = frozenset({"re", "by", "ve", "tool"})


SOURCE_TIME_PRESERVING_TIMING_SOURCES = frozenset({"lyrics", "checked"})


def output_metadata_lines(metadata_lines: Sequence[str] | None, timing_source: str) -> list[str]:
    """Return the source LRC tags that are still true for the written file.

    When the written times are the lyric file's own times (``lyrics`` /
    ``checked``), every source tag round-trips, including ``[offset:]``.
    Re-timed output must not inherit the source clock (``[offset:]``,
    ``[length:]``) or the source generator tags; a player would otherwise
    shift freshly aligned lines by the old file's offset.
    """
    lines = [str(line) for line in (metadata_lines or [])]
    if str(timing_source) in SOURCE_TIME_PRESERVING_TIMING_SOURCES:
        return lines
    kept: list[str] = []
    for line in lines:
        match = re.match(r"^\s*\[([A-Za-z][A-Za-z0-9_-]*):", line)
        key = match.group(1).casefold() if match is not None else None
        if key in SOURCE_CLOCK_METADATA_KEYS or key in SOURCE_GENERATOR_METADATA_KEYS:
            continue
        kept.append(line)
    return kept


LINE_START_LEAD_MS = 0


def resolve_start_lead_ms(cli_value: object = None) -> int:
    """Lead (ms) every line start is written earlier; CLI beats env beats the default."""
    for raw in (cli_value, os.environ.get("LRC_START_LEAD_MS")):
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return max(0, int(float(str(raw).strip())))
        except ValueError:
            continue
    return LINE_START_LEAD_MS


def _previous_line_end_cs(
    previous_assignment: dict | None, previous_cs: int | None
) -> int:
    if previous_cs is None:
        return 0
    spans = _timing_spans_from_value(
        previous_assignment.get("ctc_token_spans") if isinstance(previous_assignment, dict) else None
    )
    end_cs = None
    if spans:
        end = max(float(span.end) for span in spans)
        if end > 0:
            end_cs = int(round(end * 100))
    if end_cs is None or end_cs < previous_cs:
        end_cs = previous_cs + 1
    return end_cs


def apply_line_start_lead(
    report: dict[str, object],
    written_times: tuple[CanonicalWrittenTime, ...],
    lead_ms: int,
) -> tuple[CanonicalWrittenTime, ...]:
    """Write every line start ``lead_ms`` earlier, never before the previous line's end or 0.

    The written LRC tag and the report's written time (assignments, Central rows and
    suspicious rows) move together; per-line ``start_lead_ms`` records what was applied.
    """
    report["start_lead_ms_requested"] = int(lead_ms)
    assignments = report.get("assignments")
    if lead_ms <= 0 or not isinstance(assignments, list) or len(assignments) != len(written_times):
        report["start_lead_applied"] = False
        return written_times
    lead_cs = int(round(lead_ms / 10.0))
    shifted: list[CanonicalWrittenTime] = []
    applied: list[int] = []
    previous_new: int | None = None
    for index, written in enumerate(written_times):
        floor_cs = 0
        if index > 0:
            floor_cs = _previous_line_end_cs(
                assignments[index - 1] if isinstance(assignments[index - 1], dict) else None,
                previous_new,
            )
        new_cs = max(written.centiseconds - lead_cs, min(floor_cs, written.centiseconds), 0)
        moved = written.centiseconds - new_cs
        shifted.append(canonicalize_lrc_time(new_cs / 100.0))
        applied.append(moved * 10)
        previous_new = new_cs
    by_entry = {}
    for index, (old, new, moved_ms) in enumerate(zip(written_times, shifted, applied), 1):
        by_entry[index] = (old, new, moved_ms)
        row = assignments[index - 1]
        if not isinstance(row, dict):
            continue
        row["start_lead_ms"] = moved_ms
        if moved_ms == 0:
            continue
        row["chosen_time"] = new.seconds
        row["canonical_written_centiseconds"] = new.centiseconds
        row["lrc_tag"] = new.lrc_tag
        stamps = row.get("candidate_timestamps")
        if isinstance(stamps, dict) and "output" in stamps:
            stamps["output"] = new.seconds
        if "timestamp" in row:
            row["timestamp"] = new.seconds
        provenance = row.get("candidate_provenance")
        if isinstance(provenance, dict) and "canonical_written_centiseconds" in provenance:
            provenance["canonical_written_centiseconds"] = new.centiseconds
    for key in ("central_timing_state", "suspicious_alignments"):
        rows = report.get(key)
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            entry = row.get("entry")
            if not isinstance(entry, int) or entry not in by_entry:
                continue
            _old, new, moved_ms = by_entry[entry]
            if moved_ms == 0 or "canonical_written_centiseconds" not in row:
                continue
            row["canonical_written_centiseconds"] = new.centiseconds
            if "lrc_tag" in row:
                row["lrc_tag"] = new.lrc_tag
            if "chosen_time" in row:
                row["chosen_time"] = new.seconds
            if "timestamp" in row:
                row["timestamp"] = new.seconds
    projection = report.get(_BACKEND_PROVENANCE_PROJECTION_KEY)
    if isinstance(projection, dict) and "central_state_binding" in projection:
        # The projection binds the Central rows by digest; they just moved by the lead.
        projection["central_state_binding"] = _timing_content_digest(
            report.get("central_timing_state", [])
        )
        projection.pop("projection_revision", None)
        projection["projection_revision"] = _timing_content_digest(projection)
    report["start_lead_applied"] = True
    report["start_lead_lines_moved"] = sum(1 for value in applied if value)
    report["start_lead_lines_clamped"] = sum(1 for value in applied if 0 < value < lead_ms)
    return tuple(shifted)


def write_lrc(
    output_path: Path,
    audio_path: Path,
    lyrics_path: Path,
    entries: list[LyricEntry],
    written_times: tuple[CanonicalWrittenTime, ...],
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
    source_metadata = list(metadata_lines or [])
    source_metadata_keys: set[str] = set()
    for line in source_metadata:
        match = re.match(r"^\s*\[([A-Za-z][A-Za-z0-9_-]*):", line)
        if match is not None:
            source_metadata_keys.add(match.group(1).casefold())
    generated_metadata = []
    if "ti" not in source_metadata_keys:
        generated_metadata.append(f"[ti:{title_label or audio_path.stem}]")
    if "re" not in source_metadata_keys:
        generated_metadata.append(f"[re:{generator}]")
    if "length" not in source_metadata_keys:
        generated_metadata.append(f"[length:{format_lrc_time(duration)}]")
    if "by" not in source_metadata_keys:
        generated_metadata.append("[by:LRC tools]")
    body = [*source_metadata, *generated_metadata, f"[00:00.00]{title_card}"]
    if len(written_times) != len(entries):
        raise LrcError("Immutable written-time count does not match lyric entries")
    for written_time, entry in zip(written_times, entries):
        body.extend(f"[{written_time.lrc_tag}]{line}" for line in entry.lines)
    output_path.write_text("\n".join(body) + "\n", encoding="utf-8-sig")


def annotate_display_romaji(report: dict[str, object], report_path: Path | None = None) -> None:
    """Additive per-entry spaced Hepburn romaji (`romaji_display`) from the shared reading module.

    The existing compact `romaji` field is the aligner's target and is left untouched.  A line the lyric source gave
    furigana for is read from it (what is sung), otherwise through the shared dictionary and the analyzer.
    """
    entries = report.get(_RUNTIME_LYRIC_ENTRIES_KEY)
    assignments = report.get("assignments")
    if not isinstance(entries, (list, tuple)) or not isinstance(assignments, list):
        return
    try:
        import lrc_reading

        system = lrc_reading.reading_system()
        done = False
        furigana = {}
        if report_path is not None:
            try:
                furigana = lrc_reading.load_furigana(Path(report_path).name.replace(".align-report.json", ""))
            except Exception:
                furigana = {}
        for row in assignments:
            number = row.get("entry") if isinstance(row, dict) else None
            if not isinstance(number, int) or not 1 <= number <= len(entries) or "romaji_display" in row:
                continue
            line = entry_sung_text(entries[number - 1])
            if lrc_reading.has_japanese(line):
                value = lrc_reading.romanize_line(line, furigana=furigana.get(lrc_reading.line_key(line)))
                if value:
                    row["romaji_display"] = value
                    row["romaji_display_text"] = line
                    row["romaji_display_source"] = lrc_reading.reading_source(line, furigana)
                    done = True
        if done:
            report["romaji_display_system"] = system
    except Exception:
        return


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


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
