"""Extracted from ctc_refinement.py (split2): CTC local-window and hybrid-window realign, segment realign stage."""

from __future__ import annotations

import lrc_stage_timings
import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
import model_cache
import segment_realign
import segment_skeleton
from timing_core import CTC_WIDE_REGION_FALLBACK_SOURCE, TRUSTED_ALIGNMENT_SCORE, assignment_ctc_path_collapsed, ctc_collapsed_region_flags, record_alignment_hypothesis
from lyrics_io import LyricEntry, entry_sung_text, has_ambiguous_lyric_prefix, japanese_romaji
from audio_features import VocalStemLevels, analyze_audio, vocal_stem_levels
from runtime_base import LrcError, _bounded_retry_protocol_revision, _ctc_model_blob_sha256, _model_cache_read, _model_cache_write, _run_model_helper_command, _stable_file_sha256, default_ctc_python, progress_change_log, progress_log
from ctc_boundary_evidence import (
    ctc_right_edge_pileup_evidence,
    refresh_ctc_confidence_diagnostics,
)


_BOUNDED_RETRY_MODEL_IDENTITY = "torchaudio.pipelines.MMS_FA:known-lyric"


# A collapsed region can only be bounded by sane rows outside it, so its local
# window may exceed the ordinary 45 s split limit.
COLLAPSED_REGION_WINDOW_MAX_SECONDS = 90.0


CTC_WIDE_REGION_RADIUS = 3


CTC_WIDE_REGION_PEER_RADIUS = 2


def apply_ctc_local_window_realign(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    ctc_report: dict[str, object],
    raw_report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
    *,
    region_radius: int = 1,
    shadow: bool = False,
    window_cache: dict[tuple[int, int, str, str], list[object]] | None = None,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    """Re-run collapsed CTC ranges inside raw-ASR-bounded audio windows.

    ``shadow`` replays the same planner with a wider collapsed region
    (``region_radius``) for posthoc comparison only: its rows are recorded under
    ``ctc_local_window_shadow`` and never become Central candidates.
    ``window_cache`` lets a shadow pass reuse helper output for windows the
    primary pass already aligned.
    """
    raw_assignments = raw_report.get("assignments")
    runs = ctc_report.get("ctc_low_score_runs")
    low_score_entries = ctc_report.get("ctc_low_score_entries")
    assignments = ctc_report.get("assignments")
    if not isinstance(raw_assignments, list) or not isinstance(runs, list) or not isinstance(assignments, list):
        return timestamps, ctc_report, []

    def raw_anchor(index: int) -> float | None:
        if not 0 <= index < len(raw_assignments):
            return None
        if has_ambiguous_lyric_prefix(entries, index):
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

    # Same rows Central treats as a collapsed whole-song path.  Their CTC
    # timestamps are the broken part, so they are re-aligned even when raw ASR
    # and the path score raise no alarm, and never serve as window bounds.
    collapsed_region = ctc_collapsed_region_flags(
        [assignment_ctc_path_collapsed(item) for item in assignments],
        radius=region_radius,
    )
    shadow_rows: dict[str, dict[str, object]] = {}

    def in_collapsed_region(index: int) -> bool:
        return 0 <= index < len(collapsed_region) and collapsed_region[index]

    def local_anchor(index: int) -> float | None:
        """Prefer unique raw anchors, then fall back to neighboring CTC bounds."""
        raw_time = raw_anchor(index)
        if raw_time is not None:
            return raw_time
        if 0 <= index < len(timestamps) and not in_collapsed_region(index):
            timestamp = float(timestamps[index])
            if math.isfinite(timestamp):
                return timestamp
        return None

    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    errors: list[str] = []
    rejected_rows: list[dict[str, object]] = []
    vocal_levels: VocalStemLevels | None = None
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
        if has_ambiguous_lyric_prefix(entries, index):
            continue
        threshold = 0.45 if raw_score >= TRUSTED_ALIGNMENT_SCORE and not borrowed else 0.75
        if abs(raw_time - current_time) >= threshold:
            if isinstance(assignments[index], dict) and assignments[index].get("ctc_clear_boundary"):
                continue
            entry_number = index + 1
            if entry_number not in covered_entries:
                window_runs.append([entry_number])
                covered_entries.add(entry_number)

    for index in range(len(entries)):
        entry_number = index + 1
        if in_collapsed_region(index) and entry_number not in covered_entries:
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
    split_windows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="lrc-ctc-window-") as temp_name:
        temp_dir = Path(temp_name)
        # A disagreement can span a full verse.  Re-aligning that whole verse
        # in one 45+ second pass is both costly and less constrained; split it
        # into adjacent windows while retaining a preceding lyric as context.
        pending_runs = [list(run) for run in window_runs]
        window_number = 0
        while pending_runs:
            raw_run = pending_runs.pop(0)
            window_number += 1
            if not isinstance(raw_run, list) or not raw_run:
                continue
            try:
                low_start = int(raw_run[0]) - 1
                low_end = int(raw_run[-1]) - 1
            except (TypeError, ValueError):
                continue
            left_index = next((idx for idx in range(low_start - 1, max(-1, low_start - 4), -1) if local_anchor(idx) is not None), None)
            right_index = next((idx for idx in range(low_end + 1, min(len(entries), low_end + 4)) if local_anchor(idx) is not None), None)
            start_index = 0 if left_index is None else left_index + 1
            end_index = len(entries) - 1 if right_index is None else right_index - 1
            # Include the preceding trusted line as CTC context. Starting the
            # audio just after its onset but omitting its text lets the target
            # line steal the preceding line's sustained vowel.
            alignment_start_index = 0 if left_index is None else left_index
            own_anchor = local_anchor(low_start)
            left_anchor = local_anchor(left_index) if left_index is not None else None
            # A large raw-ASR gap marks a section or instrumental break.  The
            # preceding lyric is not useful CTC context there; keeping it
            # makes repeated lines compete across the break.
            detached_section_start = (
                own_anchor is not None
                and left_anchor is not None
                and own_anchor - left_anchor >= 10.0
            )
            if detached_section_start:
                alignment_start_index = low_start
                start_index = low_start
                start_time = max(0.0, own_anchor - 1.00)
            elif left_index is None:
                # The first disputed lyric has no preceding ASR anchor.  Do
                # not give its local CTC pass the whole instrumental intro:
                # a reliable raw onset is a window boundary, not replacement
                # timing evidence, and prevents fragmented token paths from
                # borrowing noise before the vocal starts.
                start_time = max(0.0, own_anchor - 1.00) if own_anchor is not None else 0.0
            else:
                start_time = max(0.0, float(left_anchor or 0.0) - 0.05)
            if right_index is None:
                # A suffix window has no later lyric anchor.  The final raw/CTC
                # timestamp is only a candidate onset, not a safe audio boundary.
                # Capping the pass at ``tail_anchor + 3`` can exclude the real
                # final-line onset and force the remaining transcript into the
                # last few frames of the window.
                end_time = duration
            else:
                end_time = float(local_anchor(right_index) or duration) - 0.04
            window_duration = end_time - start_time
            if end_index < alignment_start_index or window_duration < 1.0:
                continue
            if window_duration > 45.0:
                # Split only between two rows that can bound a window: the
                # halves are bounded by each other's rows, and a collapsed
                # row's own CTC time is no bound.
                split_points = [
                    offset
                    for offset in range(1, len(raw_run))
                    if local_anchor(int(raw_run[offset - 1]) - 1) is not None
                    and local_anchor(int(raw_run[offset]) - 1) is not None
                ]
                run_collapsed = any(in_collapsed_region(int(entry) - 1) for entry in raw_run)
                if split_points:
                    midpoint = min(
                        split_points,
                        key=lambda offset: (abs(offset - len(raw_run) / 2), offset),
                    )
                    left_run = raw_run[:midpoint]
                    right_run = raw_run[midpoint:]
                    pending_runs.insert(0, right_run)
                    pending_runs.insert(0, left_run)
                    split_windows.append(
                        {
                            "entries": list(raw_run),
                            "window_seconds": round(window_duration, 3),
                            "split_into": [list(left_run), list(right_run)],
                        }
                    )
                    continue
                # A collapsed region with no internal bound is aligned whole:
                # its sane outer neighbours are the only trustworthy limits.
                if not (run_collapsed and window_duration <= COLLAPSED_REGION_WINDOW_MAX_SECONDS):
                    errors.append(
                        f"window-{window_number}: skipped over-wide local CTC window ({window_duration:.1f}s)"
                    )
                    continue
            cache_key = (alignment_start_index, end_index, f"{start_time:.3f}", f"{end_time:.3f}")
            rows = window_cache.get(cache_key) if window_cache is not None else None
            window_request = {
                "alignment_start_index": alignment_start_index,
                "end_index": end_index,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "transcript": [
                    entry_sung_text(entry)
                    for entry in entries[alignment_start_index : end_index + 1]
                ],
            }
            try:
                window_stem_sha = _stable_file_sha256(audio_path)
                window_helper_sha = _stable_file_sha256(helper)
            except OSError:
                window_stem_sha = "unresolved"
                window_helper_sha = "unresolved"
            window_identity = model_cache.identity_local_window_ctc(
                stem_sha256=window_stem_sha,
                model_revision=_BOUNDED_RETRY_MODEL_IDENTITY,
                model_blob_sha256=_ctc_model_blob_sha256(),
                helper_protocol_revision=_bounded_retry_protocol_revision(window_helper_sha),
                helper_sha256=window_helper_sha,
                request=window_request,
            )
            window_digest = model_cache.key_from_identity("local-window-ctc", window_identity)
            if rows is None:
                persisted_rows = _model_cache_read("local-window-ctc", window_digest, window_identity, args)
                if isinstance(persisted_rows, list):
                    rows = persisted_rows
            if rows is None:
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
                proc = _run_model_helper_command(command, mode="ctc")
                if proc.returncode != 0 or not output_path.exists():
                    errors.append((proc.stderr or proc.stdout).strip()[:240])
                    continue
                rows = json.loads(output_path.read_text(encoding="utf-8")).get("entries")
                if window_cache is not None and isinstance(rows, list):
                    window_cache[cache_key] = rows
                if isinstance(rows, list):
                    _model_cache_write("local-window-ctc", window_digest, window_identity, rows, args)
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
                candidate_edge_evidence = ctc_right_edge_pileup_evidence(
                    {
                        "ctc_local_window": {
                            "start": round(start_time, 3),
                            "end": round(end_time, 3),
                        },
                        "ctc_token_spans": row.get("token_spans"),
                    }
                )
                if candidate_edge_evidence.get("right_edge_pileup"):
                    rejected_rows.append(
                        {
                            "entry": index + 1,
                            "candidate": round(timestamp, 3),
                            "reason": "local-window-right-edge-pileup",
                            "window": {
                                "start": round(start_time, 3),
                                "end": round(end_time, 3),
                            },
                            "ctc_score": row.get("ctc_score"),
                            "ctc_right_edge_evidence": candidate_edge_evidence,
                        }
                    )
                    # Do not commit a forced path that visibly ran out of
                    # audio.  Keeping the pre-window timestamp is safer and
                    # lets the later boundary stage evaluate that candidate
                    # instead of propagating a truncated path into the next
                    # lyric boundary.
                    continue
                # A forced path across an instrumental break can start a line
                # in vocal silence; that is no onset (the final self-check's
                # F1 judgment, applied before the row becomes a candidate).
                if vocal_levels is None:
                    vocal_levels = vocal_stem_levels(ctc_report)
                if vocal_levels.absent(round(timestamp, 2)):
                    rejected_rows.append(
                        {
                            "entry": index + 1,
                            "candidate": round(timestamp, 3),
                            "reason": "local-window-vocal-absent",
                            "window": {
                                "start": round(start_time, 3),
                                "end": round(end_time, 3),
                            },
                            "ctc_score": row.get("ctc_score"),
                            "vocal_threshold_db": round(float(vocal_levels.threshold_db or 0.0), 3),
                        }
                    )
                    continue
                assignment = assignments[index]
                if shadow:
                    shadow_rows[str(index + 1)] = {
                        "time": round(timestamp, 3),
                        "ctc_score": row.get("ctc_score"),
                        "window": {"start": round(start_time, 3), "end": round(end_time, 3)},
                    }
                    if isinstance(assignment, dict) and region_radius in (
                        CTC_WIDE_REGION_RADIUS, CTC_WIDE_REGION_PEER_RADIUS
                    ):
                        # Spans let a promoted row bound its successor's tail;
                        # identity stays unavailable so the row is never a
                        # valid clock or a material alternative of its own.
                        try:
                            wide_score = float(row.get("ctc_score", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            wide_score = 0.0
                        record_alignment_hypothesis(
                            assignment,
                            source=CTC_WIDE_REGION_FALLBACK_SOURCE,
                            stage="ctc-local-window-shadow",
                            raw_time=timestamp,
                            confidence=wide_score,
                            spans=row.get("token_spans"),
                            sequence_support="supported",
                            parent_source="ctc-global-mms",
                            parent_time=float(timestamps[index]),
                            window={"start": round(start_time, 3), "end": round(end_time, 3)},
                            mode=f"region-radius-{region_radius}",
                            right_edge_pileup=bool(candidate_edge_evidence.get("right_edge_pileup")),
                            window_truncated=bool(candidate_edge_evidence.get("right_edge_pileup")),
                        )
                elif isinstance(assignment, dict):
                    try:
                        local_score = float(row.get("ctc_score", 0.0) or 0.0)
                    except (TypeError, ValueError):
                        local_score = 0.0
                    record_alignment_hypothesis(
                        assignment,
                        source="ctc-local-window",
                        stage="ctc-local-window-realign",
                        raw_time=timestamp,
                        confidence=local_score,
                        spans=row.get("token_spans"),
                        identity_support="supported" if isinstance(row.get("token_spans"), list) and bool(row.get("token_spans")) else "unavailable",
                        sequence_support="supported",
                        parent_source="ctc-global-mms",
                        parent_time=float(timestamps[index]),
                        window={"start": round(start_time, 3), "end": round(end_time, 3)},
                        right_edge_pileup=bool(candidate_edge_evidence.get("right_edge_pileup")),
                        window_truncated=bool(candidate_edge_evidence.get("right_edge_pileup")),
                    )
                    assignment["ctc_local_window_candidate_generated"] = True
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
    if shadow:
        shadow_report = ctc_report.get("ctc_local_window_shadow")
        if not isinstance(shadow_report, dict):
            shadow_report = {}
            ctc_report["ctc_local_window_shadow"] = shadow_report
        shadow_report[f"region-radius-{region_radius}"] = {
            "authoritative": False,
            "central_candidate": False,
            "windows": changes,
            "split_windows": split_windows,
            "errors": errors,
            "rejected_rows": rejected_rows,
            "rows": shadow_rows,
        }
        return list(timestamps), ctc_report, changes
    ctc_report["ctc_local_window_realign"] = {
        "status": "candidate-generated" if changes else "skipped",
        "authoritative": False,
        "mutates_baseline": False,
        "windows": changes,
        "split_windows": split_windows,
        "errors": errors,
        "rejected_rows": rejected_rows,
    }
    # The global MMS timestamp/spans remain the backend baseline. Local-window
    # rows are immutable alternatives consumed only by the central evaluator.
    return list(timestamps), ctc_report, changes


CTC_LOCAL_WINDOW_SHADOW_RADII = (2, 3)


# ---------------------------------------------------------------------------
# V2 segment-level realignment (LRC 2.0).
#
# ``segment_realign.structural_row_map`` reads the whole-song CTC onsets against
# the lyric structure (repeated-chorus gap templates, a voiced-time syllable
# rate, monotonic order) and flags the short runs of rows the structure cannot
# accept: a chorus line sung in the next occurrence, a block placed in an
# instrumental break, rows that run backwards.  Each run is re-aligned with the
# bounded CTC helper inside the time window of its unflagged neighbours and
# published as ``segment-realign-ctc`` hypotheses.  Central decides; nothing
# here writes a timestamp, and a song with no flagged run makes no helper call.
SEGMENT_REALIGN_REVISION = "2026-10-27-v2.0-row-structure-3"


SEGMENT_REALIGN_SOURCE = "segment-realign-ctc"


SEGMENT_REALIGN_STAGE = "segment-realign"


SEGMENT_REALIGN_MAX_RUNS = 6


SEGMENT_REALIGN_MIN_WINDOW_SECONDS = 1.0


SEGMENT_REALIGN_MAX_WINDOW_SECONDS = 60.0


SEGMENT_REALIGN_MIN_ANCHOR_SHARE = 0.35


SEGMENT_REALIGN_RAW_AGREE_SECONDS = 0.45


# A helper row whose first token sits alone this far before the dense run of tokens
# grabbed a breath or noise; the onset is the first dense token (09.夢遊 e14 on the
# 2.0 replay: stray token 3.7 s early, dense run 0.25 s from the checked onset).
SEGMENT_REALIGN_STRAY_TOKEN_GAP_SECONDS = 1.5


SEGMENT_REALIGN_DENSE_TOKEN_GAP_SECONDS = 1.0


# A bounded pass whose onset lands within this of the CTC timestamp the stage was given
# confirms that row (the structural flag was a false alarm) and writes no hypothesis; on
# the fourth replay 14 of the 20 round-3 hypotheses were such confirmations.
SEGMENT_REALIGN_MIN_SHIFT_SECONDS = 0.5


# CTC score at or above which an uncontradicted row anchors the structural map.  The
# 0.10-0.25 band is 1.9 % wrong (> 0.5 s) on the evaluation set, 1.4 % above 0.30; at 0.25
# songs with a quiet CTC path keep too few anchors (2 of 35 rows on the first replay song).
SEGMENT_REALIGN_ANCHOR_CTC_SCORE = 0.12


def segment_realign_mode(args: argparse.Namespace) -> str:
    """``off`` (default until the 2.0 replay is accepted) or ``auto``; ``LRC_SEGMENT_REALIGN`` overrides the CLI flag."""
    env = os.environ.get("LRC_SEGMENT_REALIGN", "").strip().lower()
    if env in {"off", "0", "false", "no"}:
        return "off"
    if env in {"auto", "on", "1", "true", "yes"}:
        return "auto"
    return "auto" if str(getattr(args, "segment_realign", "off") or "off").lower() == "auto" else "off"


def segment_realign_row_status(
    entries: list[LyricEntry],
    index: int,
    ctc_time: float,
    ctc_assignment: object,
    raw_assignment: object,
) -> str:
    """Pre-Central stand-in for the output trust tier.

    TRUSTED: an independent raw-ASR onset (trusted score, not borrowed,
    unambiguous lyric prefix) agrees with the CTC onset.  UNTRUSTED: a confident,
    uncollapsed CTC row.  Everything else is REVIEW and may be moved by the
    structural map.  A raw onset that contradicts a confident CTC row does not
    demote it: on the evaluation set such rows are right 267 times out of 273
    (raw ASR is the unreliable party there), and demoting them left 01 My
    Dearest with 13 of 43 anchors so the stage skipped the very song it was
    built for.
    """
    if not isinstance(ctc_assignment, dict) or assignment_ctc_path_collapsed(ctc_assignment):
        return "REVIEW"
    try:
        ctc_score = float(ctc_assignment.get("ctc_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        ctc_score = 0.0
    raw_time: float | None = None
    raw_trusted = False
    if isinstance(raw_assignment, dict) and not raw_assignment.get("borrowed"):
        try:
            raw_time = float(raw_assignment.get("timestamp"))
            raw_score = float(raw_assignment.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            raw_time = None
            raw_score = 0.0
        raw_trusted = (
            raw_time is not None
            and math.isfinite(raw_time)
            and raw_score >= TRUSTED_ALIGNMENT_SCORE
            and not has_ambiguous_lyric_prefix(entries, index)
        )
    if raw_trusted and raw_time is not None and abs(raw_time - ctc_time) <= SEGMENT_REALIGN_RAW_AGREE_SECONDS:
        return "TRUSTED"
    return "UNTRUSTED" if ctc_score >= SEGMENT_REALIGN_ANCHOR_CTC_SCORE else "REVIEW"


def segment_realign_onset(row: dict[str, object]) -> tuple[float | None, list[object] | None, bool]:
    """Onset of a bounded-CTC helper row as (seconds, spans, shifted).

    The first token is the onset unless it sits alone more than
    ``SEGMENT_REALIGN_STRAY_TOKEN_GAP_SECONDS`` before the first dense pair of
    tokens; then the onset is that dense token and the stray leading spans are
    dropped, so Central does not read a seconds-long gap inside the opening.
    """
    start = row.get("start")
    if not isinstance(start, (int, float)) or not math.isfinite(float(start)):
        return None, None, False
    spans = row.get("token_spans")
    span_list = [span for span in spans if isinstance(span, dict)] if isinstance(spans, list) else []
    starts: list[float] = []
    for span in span_list:
        value = span.get("start")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return float(start), spans if isinstance(spans, list) else None, False
        starts.append(float(value))
    if len(starts) < 3:
        return float(start), spans if isinstance(spans, list) else None, False
    for index in range(len(starts) - 1):
        if starts[index + 1] - starts[index] <= SEGMENT_REALIGN_DENSE_TOKEN_GAP_SECONDS:
            if index > 0 and starts[index] - starts[0] > SEGMENT_REALIGN_STRAY_TOKEN_GAP_SECONDS:
                return starts[index], span_list[index:], True
            break
    return float(start), spans if isinstance(spans, list) else None, False


def _segment_realign_observations(
    entries: list[LyricEntry],
    timestamps: list[float],
    assignments: list[object],
    raw_assignments: list[object],
) -> tuple[list[segment_realign.RowObservation], list[str], list[str]]:
    texts = [segment_skeleton._text_for_match(entry_sung_text(entry), japanese_romaji) for entry in entries]
    observations: list[segment_realign.RowObservation] = []
    statuses: list[str] = []
    for index in range(len(entries)):
        try:
            current: float | None = float(timestamps[index])
        except (TypeError, ValueError, IndexError):
            current = None
        if current is not None and not math.isfinite(current):
            current = None
        assignment = assignments[index] if index < len(assignments) else None
        raw_assignment = raw_assignments[index] if index < len(raw_assignments) else None
        status = (
            "REVIEW" if current is None
            else segment_realign_row_status(entries, index, current, assignment, raw_assignment)
        )
        hints: list[float] = []
        if isinstance(assignment, dict):
            for hypothesis in assignment.get("alignment_hypotheses") or []:
                if isinstance(hypothesis, dict):
                    try:
                        hints.append(float(hypothesis.get("raw_candidate_time")))
                    except (TypeError, ValueError):
                        continue
        if isinstance(raw_assignment, dict):
            try:
                hints.append(float(raw_assignment.get("timestamp")))
            except (TypeError, ValueError):
                pass
        try:
            confidence = float(assignment.get("ctc_score", 0.0) or 0.0) if isinstance(assignment, dict) else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        observations.append(segment_realign.RowObservation(
            current,
            max(1, len(texts[index])),
            status,
            tuple(sorted({round(hint, 3) for hint in hints if math.isfinite(hint)})),
            confidence,
        ))
        statuses.append(status)
    return observations, statuses, texts


def apply_segment_realign(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    ctc_report: dict[str, object],
    raw_report: dict[str, object] | None,
    duration: float,
    args: argparse.Namespace,
    *,
    window_cache: dict[tuple[int, int, str, str], list[object]] | None = None,
) -> dict[str, object]:
    """Bounded CTC passes over the runs the structural map flags; hypotheses only.

    Diagnostics go to ``ctc_report["segment_realign"]``.  Rows keep their
    timestamps; every accepted helper onset becomes a ``segment-realign-ctc``
    hypothesis in the row's ledger for Central to weigh.
    """
    diagnostics: dict[str, object] = {
        "revision": SEGMENT_REALIGN_REVISION,
        "status": "skipped",
        "reason": None,
        "runs": [],
        "errors": [],
    }
    ctc_report["segment_realign"] = diagnostics
    assignments = ctc_report.get("assignments")
    if (
        not entries
        or not isinstance(assignments, list)
        or len(assignments) != len(entries)
        or len(timestamps) != len(entries)
    ):
        diagnostics["reason"] = "no-assignments"
        return diagnostics
    raw_assignments = raw_report.get("assignments") if isinstance(raw_report, dict) else None
    if not isinstance(raw_assignments, list):
        raw_assignments = []
    observations, statuses, texts = _segment_realign_observations(entries, timestamps, assignments, raw_assignments)
    diagnostics["row_status"] = {key: statuses.count(key) for key in ("TRUSTED", "UNTRUSTED", "REVIEW")}
    try:
        segments = [(float(start), float(end)) for start, end in analyze_audio(audio_path, duration).segments]
    except (LrcError, OSError, ValueError, subprocess.SubprocessError) as exc:
        diagnostics["reason"] = f"audio-analysis-failed: {exc}"[:200]
        return diagnostics
    structure = segment_realign.structural_row_map(
        observations, segments, duration=duration, sim=segment_realign.line_similarity(texts),
    )
    diagnostics["structure"] = {"status": structure.status, "reason": structure.reason, **structure.diagnostics}
    if structure.status != "READY":
        diagnostics["reason"] = structure.reason
        return diagnostics
    anchors = int(structure.diagnostics.get("anchors", 0))
    if anchors < SEGMENT_REALIGN_MIN_ANCHOR_SHARE * len(entries):
        # Too little the structure can hold on to (the Choir class): leave the
        # song to the output-stage whole-song risk marking.
        diagnostics["status"] = "weak-anchors"
        diagnostics["reason"] = f"{anchors}/{len(entries)} anchored rows"
        return diagnostics
    diagnostics["flagged_entries"] = [row.row + 1 for row in structure.rows if row.flagged]
    if not structure.runs:
        diagnostics["status"] = "no-runs"
        return diagnostics

    def displacement(run: segment_realign.RealignRun) -> float:
        moves = [
            abs(proposed - observations[row].time)
            for row, proposed in zip(range(run.first_row, run.last_row + 1), run.proposed)
            if observations[row].time is not None
        ]
        return max(moves) if moves else 0.0

    runs = sorted(structure.runs, key=lambda run: (-displacement(run), run.first_row))[:SEGMENT_REALIGN_MAX_RUNS]
    runs.sort(key=lambda run: run.first_row)
    python_exe = default_ctc_python()
    helper = Path(__file__).resolve().parent / "ctc_align.py"
    try:
        stem_sha = _stable_file_sha256(audio_path)
        helper_sha = _stable_file_sha256(helper)
    except OSError:
        stem_sha = helper_sha = "unresolved"
    vocal_levels: VocalStemLevels | None = None
    generated = 0
    with tempfile.TemporaryDirectory(prefix="lrc-segment-realign-") as temp_name:
        temp_dir = Path(temp_name)
        for number, run in enumerate(runs, start=1):
            first, last = run.first_row, run.last_row
            run_report: dict[str, object] = {
                "entries": list(range(first + 1, last + 2)),
                "proposed": [round(value, 3) for value in run.proposed],
                "start": round(run.start, 3),
                "end": round(run.end, 3),
                "candidates": 0,
                "rejected": [],
            }
            diagnostics["runs"].append(run_report)
            # The preceding unflagged row is CTC context when the window starts
            # at its onset (as the local window does); a narrowed window has none.
            previous = structure.rows[first - 1] if first > 0 else None
            context_index = first - 1 if previous is not None and abs(previous.proposed - run.start) < 0.011 else None
            alignment_start_index = first if context_index is None else context_index
            start_time = max(0.0, run.start - 0.05) if context_index is not None else max(0.0, run.start)
            end_time = duration if last + 1 >= len(entries) else max(start_time, run.end - 0.04)
            window_duration = end_time - start_time
            if not SEGMENT_REALIGN_MIN_WINDOW_SECONDS <= window_duration <= SEGMENT_REALIGN_MAX_WINDOW_SECONDS:
                run_report["skipped"] = f"window {window_duration:.1f}s outside the allowed range"
                continue
            transcript = [entry_sung_text(entry) for entry in entries[alignment_start_index:last + 1]]
            cache_key = (alignment_start_index, last, f"{start_time:.3f}", f"{end_time:.3f}")
            rows = window_cache.get(cache_key) if window_cache is not None else None
            window_request = {
                "alignment_start_index": alignment_start_index,
                "end_index": last,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "transcript": transcript,
            }
            window_identity = model_cache.identity_local_window_ctc(
                stem_sha256=stem_sha,
                model_revision=_BOUNDED_RETRY_MODEL_IDENTITY,
                model_blob_sha256=_ctc_model_blob_sha256(),
                helper_protocol_revision=_bounded_retry_protocol_revision(helper_sha),
                helper_sha256=helper_sha,
                request=window_request,
            )
            window_digest = model_cache.key_from_identity("local-window-ctc", window_identity)
            helper_state = "in-process-cache"
            if rows is None:
                persisted = _model_cache_read("local-window-ctc", window_digest, window_identity, args)
                if isinstance(persisted, list):
                    rows = persisted
                    helper_state = "model-cache"
            if rows is None:
                transcript_path = temp_dir / f"run-{number}.json"
                output_path = temp_dir / f"run-{number}.result.json"
                transcript_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
                command = [
                    str(python_exe), str(helper), "--audio", str(audio_path), "--transcript", str(transcript_path),
                    "--output", str(output_path), "--device", args.whisperx_device,
                    "--start", f"{start_time:.3f}", "--end", f"{end_time:.3f}",
                ]
                proc = _run_model_helper_command(command, mode="ctc")
                if proc.returncode != 0 or not output_path.exists():
                    message = (proc.stderr or proc.stdout).strip()[:240]
                    run_report["error"] = message
                    diagnostics["errors"].append(f"run-{number}: {message}")
                    continue
                rows = json.loads(output_path.read_text(encoding="utf-8")).get("entries")
                helper_state = "helper"
                if isinstance(rows, list):
                    if window_cache is not None:
                        window_cache[cache_key] = rows
                    _model_cache_write("local-window-ctc", window_digest, window_identity, rows, args)
            run_report["helper"] = helper_state
            if not isinstance(rows, list) or len(rows) != last - alignment_start_index + 1:
                run_report["error"] = "invalid helper output"
                diagnostics["errors"].append(f"run-{number}: invalid helper output")
                continue
            window = {"start": round(start_time, 3), "end": round(end_time, 3)}
            for offset, row in enumerate(rows):
                index = alignment_start_index + offset
                if index < first or not isinstance(row, dict):
                    continue
                timestamp, spans, onset_shifted = segment_realign_onset(row)
                if timestamp is None or not start_time <= timestamp <= end_time:
                    continue
                if onset_shifted:
                    run_report["onset_shifted"] = int(run_report.get("onset_shifted", 0)) + 1
                current = float(timestamps[index])
                if abs(timestamp - current) < SEGMENT_REALIGN_MIN_SHIFT_SECONDS:
                    # The bounded pass agrees with the CTC onset the stage was given: the
                    # structural flag was a false alarm and a confirmation carries no
                    # correction, so none is written.
                    run_report["rejected"].append({
                        "entry": index + 1, "candidate": round(timestamp, 3),
                        "current": round(current, 3), "reason": "agrees-with-current",
                    })
                    continue
                edge = ctc_right_edge_pileup_evidence(
                    {"ctc_local_window": dict(window), "ctc_token_spans": spans}
                )
                if edge.get("right_edge_pileup"):
                    run_report["rejected"].append(
                        {"entry": index + 1, "candidate": round(timestamp, 3), "reason": "right-edge-pileup"}
                    )
                    continue
                if vocal_levels is None:
                    vocal_levels = vocal_stem_levels(ctc_report)
                if vocal_levels.absent(round(timestamp, 2)):
                    run_report["rejected"].append(
                        {"entry": index + 1, "candidate": round(timestamp, 3), "reason": "vocal-absent"}
                    )
                    continue
                assignment = assignments[index]
                if not isinstance(assignment, dict):
                    continue
                try:
                    score = float(row.get("ctc_score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                # Identity stays unavailable, as for the local-window wide fallback: a
                # bounded CTC path is not identity evidence.  With identity claimed, the
                # third and fourth replays saw the 終点 e21 hypothesis (79.616, 3.0 s from
                # the global CTC onset, equal to Central's own choice) certified by the
                # joint previous+current boundary observation into a selected
                # ctc-joint-boundary-consensus row; that shrank the multi-model consensus
                # window of e20 from 32 s to 10 s and moved e20 1.03 s off its reference.
                # Raw/WhisperX corroboration and the displaced-row Whisper probe can still
                # lift the candidate; it never satisfies Central's minimum proof alone.
                record_alignment_hypothesis(
                    assignment,
                    source=SEGMENT_REALIGN_SOURCE,
                    stage=SEGMENT_REALIGN_STAGE,
                    raw_time=timestamp,
                    confidence=score,
                    spans=spans,
                    identity_support="unavailable",
                    sequence_support="supported",
                    parent_source="ctc-global-mms",
                    parent_time=float(timestamps[index]),
                    window=dict(window),
                    mode=f"run-{first + 1}-{last + 1}" + ("-onset-shifted" if onset_shifted else ""),
                )
                assignment["segment_realign_candidate_generated"] = True
                run_report["candidates"] = int(run_report["candidates"]) + 1
                generated += 1
    diagnostics["status"] = "candidates" if generated else "no-candidates"
    diagnostics["generated"] = generated
    return diagnostics


def run_segment_realign_stage(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    ctc_report: dict[str, object],
    raw_report: dict[str, object] | None,
    duration: float,
    args: argparse.Namespace,
    window_cache: dict[tuple[int, int, str, str], list[object]] | None,
) -> None:
    """Production entry point: honours ``--segment-realign`` and never raises."""
    if segment_realign_mode(args) == "off":
        ctc_report["segment_realign"] = {"revision": SEGMENT_REALIGN_REVISION, "status": "off"}
        return
    try:
        result = apply_segment_realign(
            audio_path, entries, timestamps, ctc_report, raw_report, duration, args, window_cache=window_cache,
        )
    except Exception as exc:  # a V2 candidate stage must never break production
        ctc_report["segment_realign"] = {
            "revision": SEGMENT_REALIGN_REVISION,
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}"[:240],
        }
        progress_log(f"segment realign failed: {type(exc).__name__}: {exc}")
        return
    progress_change_log(
        "segment realign candidate generated",
        [run for run in result.get("runs", []) if isinstance(run, dict) and run.get("candidates")],
    )


def run_ctc_local_window_shadows(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    ctc_report: dict[str, object],
    raw_report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
    window_cache: dict[tuple[int, int, str, str], list[object]],
) -> None:
    """Posthoc-only replays of the local-window planner with wider collapsed
    regions, so one validation run can compare region sizes.  Never feeds
    Central; a failure is recorded, not raised."""
    assignments = ctc_report.get("assignments")
    if not isinstance(assignments, list) or not any(
        assignment_ctc_path_collapsed(item) for item in assignments
    ):
        return
    for radius in CTC_LOCAL_WINDOW_SHADOW_RADII:
        try:
            apply_ctc_local_window_realign(
                audio_path,
                entries,
                list(timestamps),
                ctc_report,
                raw_report,
                duration,
                args,
                region_radius=radius,
                shadow=True,
                window_cache=window_cache,
            )
        except Exception as exc:  # diagnostics must never break production
            shadow_report = ctc_report.get("ctc_local_window_shadow")
            if not isinstance(shadow_report, dict):
                shadow_report = {}
                ctc_report["ctc_local_window_shadow"] = shadow_report
            shadow_report[f"region-radius-{radius}"] = {"error": repr(exc)[:240]}


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
            end_time = duration if right_boundary == context_end else refined[right_boundary] - 0.05
            window_duration = end_time - start_time
            # R2-6: never widen the window across the next untouched line.
            # A disagreement run squeezed into less than 1 s is skipped instead.
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
            proc = _run_model_helper_command(command, mode="ctc")
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


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
