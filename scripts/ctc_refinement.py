"""Extracted from auto_lrc.py (split step 6): CTC boundary evidence, local window and hybrid realign, prefix recovery, final CTC guard."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
import model_cache
import segment_realign
import segment_skeleton
from report_trust import ACTIONABLE_SUSPICIOUS_FLAGS, ACTIONABLE_TIMING_FAILURES, CTC_BOUNDARY_REVIEW_FLAGS, _RUNTIME_RAW_CTC_CONSENSUS_KEY, _backend_participation, _ctc_disagreement_severity, _merge_suspicious_severity, _raw_disagreement_severity, _record_backend_participation_failure, dedupe_suspicious_alignments, update_report_confidence_metrics
from timing_core import CTC_WIDE_REGION_FALLBACK_SOURCE, EvidenceState, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, TRUSTED_ALIGNMENT_SCORE, _entry_occurrence_binding, assignment_ctc_path_collapsed, ctc_collapsed_region_flags, record_alignment_hypothesis
from lyrics_io import LyricEntry, entry_sung_text, has_ambiguous_lyric_prefix, japanese_romaji, normalize_match_text
from audio_features import AudioFeatures, HOP_SECONDS, VocalStemLevels, analyze_audio, analyze_vocal_onsets, nearest_local_peak, nearest_onset_evidence, vocal_onset_features, vocal_stem_levels
from runtime_base import LrcError, _bounded_retry_protocol_revision, _ctc_model_blob_sha256, _model_cache_read, _model_cache_write, _run_model_helper_command, _stable_file_sha256, _timing_content_digest, default_ctc_python, progress_change_log, progress_log
import lrc_stage_timings


RAW_CTC_CONSENSUS_MIN_SCORE = 0.85


_BOUNDED_RETRY_MODEL_IDENTITY = "torchaudio.pipelines.MMS_FA:known-lyric"


# A collapsed region can only be bounded by sane rows outside it, so its local
# window may exceed the ordinary 45 s split limit.
COLLAPSED_REGION_WINDOW_MAX_SECONDS = 90.0


CTC_WIDE_REGION_RADIUS = 3


CTC_WIDE_REGION_PEER_RADIUS = 2


def ctc_prefix_boundary_evidence(
    assignment: object, boundary_gap: float | None = None
) -> dict[str, object]:
    """Summarize multi-token evidence for a CTC line boundary."""
    empty: dict[str, object] = {
        "available": False,
        "boundary_gap_seconds": boundary_gap,
        "previous_token_score": None,
        "zero_boundary_gap": False,
        "unclear_boundary": True,
        "low_confidence_prefix": False,
        "abnormal_prefix_compression": False,
        "ghost_weak_prefix": False,
        "borrowed_tail_prefix": False,
        "near_gap_weak_prefix": False,
        "detached_weak_prefix": False,
        "unsupported_weak_prefix": False,
        "timestamp_precedes_ctc_support": False,
        "realign_required": False,
    }
    if not isinstance(assignment, dict):
        return empty
    if boundary_gap is None:
        try:
            boundary_gap = float(assignment.get("ctc_boundary_gap_seconds"))
        except (TypeError, ValueError):
            return empty
    spans = assignment.get("ctc_token_spans")
    if not isinstance(spans, list) or not spans:
        return empty
    try:
        effective_index = int(assignment.get("ctc_effective_token_start_index", 0) or 0)
    except (TypeError, ValueError):
        effective_index = 0
    effective_spans = [span for span in spans[effective_index:] if isinstance(span, dict)]
    if not effective_spans:
        return empty
    prefix_count = min(8, len(effective_spans))
    prefix = effective_spans[:prefix_count]
    try:
        starts = [float(span.get("start")) for span in prefix]
        ends = [float(span.get("end")) for span in prefix]
        scores = [float(span.get("score", 0.0) or 0.0) for span in prefix]
        ctc_score = float(assignment.get("ctc_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return empty
    if not starts or not ends or not scores:
        return empty
    prefix_span = max(ends) - min(starts)
    weak_count = sum(1 for score in scores if score < 0.05)
    prefix_mean = sum(scores) / len(scores)
    weak_ratio = weak_count / len(scores)
    reliable_index: int | None = None
    reliable_time: float | None = None
    reliable_score: float | None = None
    first_start = starts[0]
    try:
        boundary_start = float(assignment.get("timestamp", first_start))
    except (TypeError, ValueError):
        boundary_start = first_start
    for index, span in enumerate(effective_spans[:24]):
        try:
            score = float(span.get("score", 0.0) or 0.0)
            start = float(span.get("start"))
        except (TypeError, ValueError):
            continue
        if score >= 0.08:
            reliable_index = index
            reliable_time = start
            reliable_score = score
            break
    reliable_delay = reliable_time - boundary_start if reliable_time is not None else None
    timestamp_token_lead = first_start - boundary_start
    zero_gap = boundary_gap <= 0.04
    unclear = boundary_gap < 0.35 or scores[0] < 0.25
    enough_prefix = prefix_count >= 6
    low_prefix = (
        enough_prefix
        and scores[0] < 0.03
        and prefix_mean < 0.04
        and weak_ratio >= 0.625
    )
    compressed = (
        prefix_count >= 8
        and ctc_score <= 0.15
        and prefix_mean < 0.025
        and weak_ratio >= 0.875
        and prefix_span <= 0.42
    )
    delayed_reliable = reliable_delay is None or reliable_delay >= 0.70
    leading_count = min(3, len(scores))
    leading_weak_ratio = (
        sum(1 for score in scores[:leading_count] if score < 0.05) / leading_count
        if leading_count
        else 0.0
    )
    # A non-zero but very short inter-line gap can still contain one or more
    # ghost CTC characters borrowed from the previous lyric tail.
    ghost_weak_prefix = (
        boundary_gap <= 0.20
        and leading_count >= 3
        and scores[0] < 0.03
        and leading_weak_ratio >= (2.0 / 3.0)
        and (reliable_delay is None or reliable_delay >= 1.0)
    )
    # A 20-40 ms boundary with weak tail/head tokens is not an independent
    # lyric onset.  It is usually the forced path reusing the previous line's
    # release/reverb as the next line's first character.  Keep this separate
    # from ghost_weak_prefix because a genuinely supported token may appear
    # shortly afterwards (e.g. 100-200 ms) and the old delayed-reliable test
    # would otherwise incorrectly clear the boundary.
    try:
        previous_token_score = float(assignment.get("ctc_previous_token_score"))
    except (TypeError, ValueError):
        previous_token_score = 1.0
    borrowed_tail_prefix = (
        boundary_gap <= 0.06
        and scores[0] < 0.03
        and previous_token_score < 0.05
        and leading_count >= 3
        and leading_weak_ratio >= (2.0 / 3.0)
    )
    # ``zero_boundary_gap`` is intentionally very strict (<= 40 ms), but a
    # line can still steal the prior lyric tail with a nominal 60-150 ms gap.
    # Treat that boundary as suspect when almost the entire opening CTC prefix
    # is weak.  This catches the common "...again" -> next-line leak without
    # penalising a genuinely strong onset such as the recovered 02:47 line.
    near_gap_weak_prefix = (
        boundary_gap <= 0.15
        and unclear
        and enough_prefix
        and scores[0] < 0.25
        and prefix_mean < 0.08
        and weak_ratio >= 0.75
        and leading_weak_ratio >= (2.0 / 3.0)
    )
    # If a repair/backtrack moves the selected timestamp materially before
    # the first CTC token, the token score cannot be used as evidence for the
    # repaired timestamp.  Flag the mismatch so downstream validation does
    # not accidentally inherit stale first-token confidence.
    timestamp_precedes_ctc_support = timestamp_token_lead >= 0.35
    # A large inter-line gap does not make a weak opening trustworthy. Breath,
    # reverb and instrumental transients can absorb several CTC characters
    # before the first supported lyric token. Keep this separate from the
    # short-gap cross-line ghost case because the recovery must move forward,
    # not back toward the previous lyric tail.
    detached_weak_prefix = (
        boundary_gap >= 0.35
        and enough_prefix
        and scores[0] < 0.03
        and prefix_mean < 0.05
        and weak_ratio >= 0.625
        and leading_weak_ratio >= (2.0 / 3.0)
        and (reliable_delay is None or reliable_delay >= 0.70)
    )
    unsupported_weak_prefix = (
        enough_prefix
        and scores[0] < 0.03
        and prefix_mean < 0.04
        and weak_ratio >= 0.75
        and (reliable_delay is None or reliable_delay >= 0.70)
    )
    realign_required = (
        zero_gap
        and unclear
        and (compressed or (low_prefix and (ctc_score < 0.03 or delayed_reliable)))
    ) or ghost_weak_prefix or borrowed_tail_prefix or near_gap_weak_prefix or detached_weak_prefix or timestamp_precedes_ctc_support
    return {
        "available": True,
        "boundary_gap_seconds": round(boundary_gap, 3),
        "previous_token_score": assignment.get("ctc_previous_token_score"),
        "zero_boundary_gap": zero_gap,
        "unclear_boundary": unclear,
        "first_token_score": round(scores[0], 6),
        "prefix_token_count": prefix_count,
        "prefix_mean_score": round(prefix_mean, 6),
        "prefix_weak_ratio": round(weak_ratio, 3),
        "leading_three_weak_ratio": round(leading_weak_ratio, 3),
        "prefix_span_seconds": round(prefix_span, 3),
        "prefix_tokens_per_second": round(prefix_count / max(prefix_span, 0.02), 3),
        "low_confidence_prefix": low_prefix,
        "abnormal_prefix_compression": compressed,
        "ghost_weak_prefix": ghost_weak_prefix,
        "borrowed_tail_prefix": borrowed_tail_prefix,
        "near_gap_weak_prefix": near_gap_weak_prefix,
        "detached_weak_prefix": detached_weak_prefix,
        "unsupported_weak_prefix": unsupported_weak_prefix,
        "ctc_token_lead_seconds": round(timestamp_token_lead, 3),
        "timestamp_precedes_ctc_support": timestamp_precedes_ctc_support,
        "next_reliable_token_index": reliable_index,
        "next_reliable_token_time": round(reliable_time, 3) if reliable_time is not None else None,
        "next_reliable_token_score": round(reliable_score, 6) if reliable_score is not None else None,
        "reliable_token_delay_seconds": round(reliable_delay, 3) if reliable_delay is not None else None,
        "realign_required": realign_required,
    }


def has_ctc_compressed_prefix(assignment: object, boundary_gap: float | None = None) -> bool:
    evidence = ctc_prefix_boundary_evidence(assignment, boundary_gap)
    return bool(
        evidence.get("zero_boundary_gap")
        and evidence.get("abnormal_prefix_compression")
    )


def ctc_right_edge_pileup_evidence(assignment: object) -> dict[str, object]:
    """Detect a forced path that ran out of audio and packed its tail at a local window edge."""
    empty: dict[str, object] = {
        "available": False,
        "touches_window_end": False,
        "right_edge_pileup": False,
    }
    if not isinstance(assignment, dict):
        return empty
    local_window = assignment.get("ctc_local_window")
    spans = assignment.get("ctc_token_spans")
    if not isinstance(local_window, dict) or not isinstance(spans, list) or len(spans) < 8:
        return empty
    try:
        window_end = float(local_window.get("end"))
    except (TypeError, ValueError):
        return empty

    parsed_spans: list[tuple[float, float, float]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        try:
            parsed_spans.append(
                (
                    float(span.get("start")),
                    float(span.get("end")),
                    float(span.get("score", 0.0) or 0.0),
                )
            )
        except (TypeError, ValueError):
            continue
    if len(parsed_spans) < 8:
        return empty

    tail_count = min(20, len(parsed_spans))
    tail = parsed_spans[-tail_count:]
    tail_start = min(item[0] for item in tail)
    tail_end = max(item[1] for item in tail)
    tail_span = max(0.02, tail_end - tail_start)
    scores = [item[2] for item in tail]
    tail_mean = sum(scores) / len(scores)
    weak_ratio = sum(1 for score in scores if score < 0.05) / len(scores)
    very_weak_ratio = sum(1 for score in scores if score < 0.02) / len(scores)
    tokens_per_second = tail_count / tail_span
    edge_margin = window_end - tail_end
    touches_window_end = -0.03 <= edge_margin <= 0.10
    right_edge_pileup = bool(
        touches_window_end
        and tokens_per_second >= 20.0
        and weak_ratio >= 0.80
        and very_weak_ratio >= 0.65
        and tail_mean < 0.04
    )
    return {
        "available": True,
        "window_end": round(window_end, 3),
        "tail_end": round(tail_end, 3),
        "edge_margin_seconds": round(edge_margin, 3),
        "tail_token_count": tail_count,
        "tail_span_seconds": round(tail_span, 3),
        "tail_tokens_per_second": round(tokens_per_second, 3),
        "tail_mean_score": round(tail_mean, 6),
        "tail_weak_ratio": round(weak_ratio, 3),
        "tail_very_weak_ratio": round(very_weak_ratio, 3),
        "touches_window_end": touches_window_end,
        "right_edge_pileup": right_edge_pileup,
    }


def annotate_ctc_boundary_evidence(assignments: list[object]) -> None:
    """Record sequence-level CTC boundaries so later candidates cannot steal a prior tail."""
    previous_token_end: float | None = None
    previous_token_score: float | None = None
    previous_right_edge_pileup = False
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            previous_token_end = None
            previous_token_score = None
            previous_right_edge_pileup = False
            continue
        spans = assignment.get("ctc_token_spans")
        if not isinstance(spans, list) or not spans:
            assignment.pop("ctc_boundary_gap_seconds", None)
            assignment.pop("ctc_clear_boundary", None)
            previous_token_end = None
            previous_token_score = None
            previous_right_edge_pileup = False
            continue
        try:
            effective_index = int(assignment.get("ctc_effective_token_start_index", 0) or 0)
            effective_index = max(0, min(effective_index, len(spans) - 1))
            first_span = spans[effective_index]
            first_start = float(first_span.get("start"))
            boundary_start = float(assignment.get("timestamp", first_start))
            first_score = float(first_span.get("score", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            previous_token_end = None
            previous_token_score = None
            previous_right_edge_pileup = False
            continue
        token_ends: list[tuple[float, float]] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            try:
                token_ends.append(
                    (
                        float(span.get("end")),
                        float(span.get("score", 0.0) or 0.0),
                    )
                )
            except (TypeError, ValueError):
                continue
        if not token_ends:
            previous_token_end = None
            previous_token_score = None
            previous_right_edge_pileup = False
            continue
        assignment["ctc_first_token_start"] = round(first_start, 3)
        assignment["ctc_first_token_score"] = round(first_score, 6)
        assignment["ctc_boundary_start"] = round(boundary_start, 3)
        if index and previous_token_end is not None:
            gap = boundary_start - previous_token_end
            assignment["ctc_previous_token_end"] = round(previous_token_end, 3)
            assignment["ctc_previous_token_score"] = (
                round(previous_token_score, 6) if previous_token_score is not None else None
            )
            assignment["ctc_boundary_gap_seconds"] = round(gap, 3)
            evidence = ctc_prefix_boundary_evidence(assignment, gap)
            prior_edge_contamination = bool(previous_right_edge_pileup and gap <= 0.20)
            if prior_edge_contamination:
                evidence["previous_right_edge_pileup"] = True
                evidence["prior_line_truncation_contamination"] = True
                evidence["realign_required"] = True
            else:
                evidence["previous_right_edge_pileup"] = bool(previous_right_edge_pileup)
                evidence["prior_line_truncation_contamination"] = False
            assignment["ctc_clear_boundary"] = (
                gap >= 0.35
                and first_score >= 0.25
                and not evidence.get("timestamp_precedes_ctc_support")
            )
            assignment["ctc_boundary_evidence"] = evidence
            assignment["ctc_zero_boundary_gap"] = bool(evidence.get("zero_boundary_gap"))
            assignment["ctc_unclear_boundary"] = bool(evidence.get("unclear_boundary"))
            assignment["ctc_low_confidence_prefix"] = bool(evidence.get("low_confidence_prefix"))
            assignment["ctc_prefix_compressed"] = bool(evidence.get("abnormal_prefix_compression"))
            assignment["ctc_ghost_weak_prefix"] = bool(evidence.get("ghost_weak_prefix"))
            assignment["ctc_borrowed_tail_prefix"] = bool(evidence.get("borrowed_tail_prefix"))
            assignment["ctc_near_gap_weak_prefix"] = bool(evidence.get("near_gap_weak_prefix"))
            assignment["ctc_prior_line_truncation_contamination"] = bool(
                evidence.get("prior_line_truncation_contamination")
            )
            assignment["ctc_boundary_realign_required"] = bool(evidence.get("realign_required"))
        else:
            assignment["ctc_clear_boundary"] = False
            assignment["ctc_boundary_evidence"] = {"available": False, "reason": "first-entry-or-no-previous-token"}
            assignment["ctc_zero_boundary_gap"] = False
            assignment["ctc_unclear_boundary"] = True
            assignment["ctc_low_confidence_prefix"] = False
            assignment["ctc_prefix_compressed"] = False
            assignment["ctc_ghost_weak_prefix"] = False
            assignment["ctc_borrowed_tail_prefix"] = False
            assignment["ctc_near_gap_weak_prefix"] = False
            assignment["ctc_prior_line_truncation_contamination"] = False
            assignment["ctc_boundary_realign_required"] = False
        edge_evidence = ctc_right_edge_pileup_evidence(assignment)
        assignment["ctc_right_edge_evidence"] = edge_evidence
        assignment["ctc_right_edge_pileup"] = bool(edge_evidence.get("right_edge_pileup"))
        previous_token_end, previous_token_score = max(token_ends, key=lambda item: item[0])
        previous_right_edge_pileup = bool(assignment["ctc_right_edge_pileup"])


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
    right_edge_pileups = {
        index
        for index, assignment in enumerate(assignments, start=1)
        if isinstance(assignment, dict) and assignment.get("ctc_right_edge_pileup")
    }
    # Keep isolated low-score paths in the low-confidence metrics. They become
    # review-required only when the sequence collapses, an entry is missing, or
    # a local forced path demonstrably ran into its right audio boundary.
    review = collapsed | missing | right_edge_pileups
    suspicious = []
    for entry_number in sorted(review):
        flags = []
        if entry_number in missing:
            flags.append("ctc_missing_alignment")
        if entry_number in low_numbers:
            flags.append("ctc_low_path_confidence")
        if entry_number in collapsed:
            flags.append("ctc_confidence_collapse")
        if entry_number in right_edge_pileups:
            flags.extend(["ctc_right_edge_pileup", "ctc_local_window_truncated"])
        item: dict[str, object] = {
            "entry": entry_number,
            "text": entry_sung_text(entries[entry_number - 1]),
            "flags": flags,
            "severity": "high" if entry_number in missing or entry_number in collapsed or entry_number in right_edge_pileups else "medium",
            "review_required": True,
        }
        assignment = assignments[entry_number - 1]
        if isinstance(assignment, dict) and entry_number in right_edge_pileups:
            item["ctc_right_edge_evidence"] = assignment.get("ctc_right_edge_evidence")
        suspicious.append(item)
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
    """Reject initial CTC tokens that were borrowed from the prior lyric tail."""
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
        if current.get("ctc_crossline_initial_recovery"):
            continue
        previous_spans = previous.get("ctc_token_spans")
        current_spans = current.get("ctc_token_spans")
        if not isinstance(previous_spans, list) or not isinstance(current_spans, list) or len(current_spans) < 2:
            continue
        try:
            previous_end = max(float(span.get("end")) for span in previous_spans if isinstance(span, dict))
            parsed_spans = [
                (
                    float(span.get("start")),
                    float(span.get("end")),
                    float(span.get("score", 0.0) or 0.0),
                )
                for span in current_spans
                if isinstance(span, dict)
            ]
        except (AttributeError, TypeError, ValueError):
            continue
        if len(parsed_spans) < 2:
            continue

        current_start = parsed_spans[0][0]
        next_time = refined[index + 1] if index + 1 < len(refined) else duration
        candidate_index: int | None = None
        recovery_mode = ""

        if current_start <= previous_end + 0.06 and parsed_spans[1][0] - current_start >= 0.70:
            candidate_index = 1
            recovery_mode = "single-overlapping-tail-token"
        else:
            evidence = ctc_prefix_boundary_evidence(current, current_start - previous_end)
            if evidence.get("borrowed_tail_prefix"):
                # Short cross-line leakage is different from a detached ghost
                # prefix: the real next-line token can arrive only 80-400 ms
                # later.  Skip only a compact run of very weak tokens and
                # require a reliable token plus downstream support, otherwise
                # leave the timestamp for review rather than guessing.
                search_limit = min(5, len(parsed_spans))
                for probe_index in range(1, search_limit):
                    candidate_start = parsed_spans[probe_index][0]
                    candidate_score = parsed_spans[probe_index][2]
                    preceding_scores = [item[2] for item in parsed_spans[:probe_index]]
                    support_scores = [item[2] for item in parsed_spans[probe_index : probe_index + 5]]
                    support_count = sum(score >= 0.08 for score in support_scores)
                    if (
                        0.08 <= candidate_start - previous_end <= 0.50
                        and 0.06 <= candidate_start - current_start <= 0.45
                        and preceding_scores
                        and all(score < 0.03 for score in preceding_scores)
                        and candidate_score >= 0.08
                        and support_count >= 2
                    ):
                        candidate_index = probe_index
                        recovery_mode = "short-borrowed-tail-prefix"
                        break
            if candidate_index is None and evidence.get("ghost_weak_prefix"):
                search_limit = min(6, len(parsed_spans))
                for probe_index in range(1, search_limit):
                    candidate_start = parsed_spans[probe_index][0]
                    preceding_scores = [item[2] for item in parsed_spans[:probe_index]]
                    inter_token_gap = candidate_start - parsed_spans[probe_index - 1][0]
                    candidate_score = parsed_spans[probe_index][2]
                    reliable_support = any(
                        later_score >= 0.08
                        and 0.0 <= later_start - candidate_start <= 1.35
                        for later_start, _later_end, later_score in parsed_spans[probe_index : probe_index + 5]
                    )
                    relative_score_floor = max(
                        0.03,
                        max(preceding_scores, default=0.0) * 1.80,
                    )
                    if (
                        inter_token_gap >= 0.55
                        and candidate_start - previous_end >= 0.50
                        and candidate_start - current_start >= 0.80
                        and preceding_scores
                        and all(score < 0.05 for score in preceding_scores)
                        and candidate_score >= relative_score_floor
                        and reliable_support
                    ):
                        candidate_index = probe_index
                        recovery_mode = "multi-token-ghost-prefix"
                        break

        if candidate_index is None:
            continue
        recovered_start = parsed_spans[candidate_index][0]
        if not previous_end + 0.05 < recovered_start < next_time - 0.10:
            continue

        old_time = refined[index]
        refined[index] = recovered_start
        current["timestamp"] = round(recovered_start, 3)
        current["ctc_crossline_initial_recovery"] = True
        current["ctc_crossline_initial_recovery_mode"] = recovery_mode
        current["ctc_crossline_initial_recovery_from"] = round(old_time, 3)
        current["ctc_crossline_initial_recovery_skipped_tokens"] = candidate_index
        ghost_span = recovered_start - current_start
        support_scores = [item[2] for item in parsed_spans[candidate_index : candidate_index + 5]]
        support_count = sum(1 for score in support_scores if score >= 0.08)
        candidate_score = parsed_spans[candidate_index][2]
        recovery_trusted = (
            recovery_mode in {"multi-token-ghost-prefix", "short-borrowed-tail-prefix"}
            or (
                recovery_mode == "single-overlapping-tail-token"
                and ghost_span >= 0.80
                and candidate_score >= 0.25
                and support_count >= 2
            )
        )
        current["ctc_crossline_initial_recovery_trusted"] = recovery_trusted
        if recovery_trusted:
            current["ctc_effective_token_start_index"] = candidate_index
        else:
            # A provisional move may be a better timestamp than the token that
            # literally overlaps the previous line, but it is not allowed to
            # rewrite which CTC token provides evidence for the line.  Keeping
            # a stale effective index here made later validators ignore a
            # usable first token and manufacture a weak-prefix failure.
            current.pop("ctc_effective_token_start_index", None)
            current["ctc_crossline_initial_recovery_provisional"] = True
        current["ctc_crossline_initial_recovery_evidence"] = {
            "previous_token_end": round(previous_end, 3),
            "candidate_score": round(candidate_score, 6),
            "candidate_support_count": support_count,
            "candidate_support_scores": [round(score, 6) for score in support_scores],
            "preceding_scores": [
                round(item[2], 6) for item in parsed_spans[:candidate_index]
            ],
            "previous_gap_seconds": round(recovered_start - previous_end, 3),
            "ghost_span_seconds": round(ghost_span, 3),
        }
        changes.append(
            {
                "entry": index + 1,
                "from": round(old_time, 3),
                "to": round(recovered_start, 3),
                "previous_token_end": round(previous_end, 3),
                "detached_initial_gap": round(recovered_start - current_start, 3),
                "skipped_tokens": candidate_index,
                "mode": recovery_mode,
                "trusted": recovery_trusted,
                "reason": (
                    "weak-ghost-prefix-before-supported-lyric-onset"
                    if recovery_mode == "multi-token-ghost-prefix"
                    else (
                        "short-weak-prefix-borrowed-from-previous-line-tail"
                        if recovery_mode == "short-borrowed-tail-prefix"
                        else "current-initial-token-overlaps-previous-line-tail"
                    )
                ),
            }
        )
    if changes:
        report["ctc_crossline_initial_recovery_count"] = len(changes)
        report["ctc_crossline_initial_recoveries"] = changes
    return refined, report, changes


def ctc_forward_supported_prefix_candidate(assignment: object) -> dict[str, object] | None:
    """Find a safely detached initial token after a breath/noise false onset.

    A run of low CTC scores is not sufficient evidence to move a timestamp
    forward: quiet consonants and vowels can be real lyric onsets.  Automatic
    correction is therefore limited to a narrow topology: one or two weak
    initial tokens, followed by a large temporal gap and a strongly supported
    token cluster.  Continuous weak prefixes remain at the original CTC time
    and are available to the audit layer for review instead of being shifted.
    """
    if not isinstance(assignment, dict):
        return None
    spans = assignment.get("ctc_token_spans")
    if not isinstance(spans, list) or len(spans) < 4:
        return None
    parsed: list[tuple[int, float, float, float]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            continue
        try:
            parsed.append(
                (
                    index,
                    float(span.get("start")),
                    float(span.get("end")),
                    float(span.get("score", 0.0) or 0.0),
                )
            )
        except (TypeError, ValueError):
            continue
    if len(parsed) < 4 or parsed[0][3] >= 0.03:
        return None

    first_start = parsed[0][1]
    search_limit = min(len(parsed), 3)
    for position in range(1, search_limit):
        original_index, candidate_start, _candidate_end, candidate_score = parsed[position]
        delay = candidate_start - first_start
        previous_end = parsed[position - 1][2]
        detached_gap = candidate_start - previous_end
        if not 0.80 <= delay <= 4.50:
            continue
        preceding_scores = [item[3] for item in parsed[:position]]
        if not preceding_scores:
            continue
        preceding_weak_ratio = sum(score < 0.05 for score in preceding_scores) / len(preceding_scores)
        preceding_mean = sum(preceding_scores) / len(preceding_scores)
        broad_support_scores = [item[3] for item in parsed[position : position + 12]]
        broad_support_count = sum(score >= 0.08 for score in broad_support_scores)

        strong_detached_initial = (
            position <= 2
            and detached_gap >= 0.55
            and preceding_weak_ratio >= 0.75
            and preceding_mean < 0.05
            and candidate_score >= 0.45
            and broad_support_count >= 4
        )
        if not strong_detached_initial:
            continue
        return {
            "token_index": original_index,
            "candidate_time": round(candidate_start, 3),
            "candidate_score": round(candidate_score, 6),
            "delay_seconds": round(delay, 3),
            "detached_gap_seconds": round(detached_gap, 3),
            "preceding_token_count": position,
            "preceding_mean_score": round(preceding_mean, 6),
            "preceding_weak_ratio": round(preceding_weak_ratio, 3),
            "support_mode": "strong-detached-initial-token",
            "support_count": broad_support_count,
            "support_scores": [round(score, 6) for score in broad_support_scores],
        }
    return None


def apply_ctc_forward_supported_prefix_recovery(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> list[dict[str, object]]:
    """Move forward from breath/noise only when later CTC tokens and acoustics agree."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps):
        return []
    try:
        features = analyze_audio(audio_path, duration)
    except LrcError as exc:
        _record_backend_participation_failure(
            report, "apply_ctc_forward_supported_prefix_recovery", exc
        )
        return []

    changes: list[dict[str, object]] = []
    probes: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict) or index == 0:
            continue
        if assignment.get("ctc_boundary_realign_trusted") or assignment.get(
            "ctc_forward_supported_prefix_recovery_trusted"
        ):
            continue
        candidate = ctc_forward_supported_prefix_candidate(assignment)
        if not isinstance(candidate, dict):
            assignment.pop("ctc_forward_prefix_probe_unresolved", None)
            continue
        try:
            current_time = float(assignment.get("timestamp", timestamps[index]) or timestamps[index])
            candidate_time = float(candidate["candidate_time"])
            candidate_score = float(candidate["candidate_score"])
            candidate_index = int(candidate["token_index"])
        except (TypeError, ValueError, KeyError):
            continue
        support_mode = str(candidate.get("support_mode", "local-supported-token-cluster"))
        strong_detached_initial = support_mode == "strong-detached-initial-token"
        previous_time = timestamps[index - 1]
        next_time = timestamps[index + 1] if index + 1 < len(timestamps) else duration
        previous_end = previous_time
        previous_assignment = assignments[index - 1]
        if isinstance(previous_assignment, dict):
            previous_spans = previous_assignment.get("ctc_token_spans")
            if isinstance(previous_spans, list):
                for span in previous_spans:
                    if not isinstance(span, dict):
                        continue
                    try:
                        previous_end = max(previous_end, float(span.get("end")))
                    except (TypeError, ValueError):
                        continue
        lookback = 0.25 if strong_detached_initial else 0.90
        lookahead = 0.12 if strong_detached_initial else 0.08
        search_start = max(
            previous_end + 0.05,
            current_time + 0.15,
            candidate_time - lookback,
        )
        search_end = min(next_time - 0.10, candidate_time + lookahead)
        acoustic_onset: float | None = None
        acoustic_threshold: float | None = None
        if search_end - search_start >= 0.20:
            for threshold in (0.65, 0.55, 0.45):
                acoustic_onset = first_local_onset(
                    features,
                    search_start,
                    search_end,
                    threshold=threshold,
                    # This window is already bounded by a reliable previous
                    # tail and the independently supported CTC token.  Keeping
                    # the generic detector's anti-edge guard here would erase
                    # a legitimate onset at the exact start of that window.
                    edge_guard_seconds=0.0,
                )
                if acoustic_onset is not None:
                    acoustic_threshold = threshold
                    break

        recovery_time: float | None = None
        recovery_mode = ""
        maximum_onset_lead = 0.08 if strong_detached_initial else 0.85
        if (
            acoustic_onset is not None
            and acoustic_onset >= current_time + 0.25
            and -0.20 <= candidate_time - acoustic_onset
            and candidate_time - acoustic_onset <= maximum_onset_lead + 1e-9
        ):
            recovery_time = acoustic_onset
            recovery_mode = (
                "forward-acoustic-onset-before-strong-detached-token"
                if strong_detached_initial
                else "forward-acoustic-onset-with-ctc-support"
            )
        elif strong_detached_initial and candidate_score >= 0.45:
            # A very strong second/third token plus broad downstream support is
            # sufficient to reject a detached breath token when the onset
            # detector is conservative.
            recovery_time = candidate_time
            recovery_mode = "strong-detached-initial-ctc-token"

        probe = {
            "entry": index + 1,
            "lyric": entry_sung_text(entries[index]),
            "from": round(current_time, 3),
            "supported_token": candidate,
            "search_start": round(search_start, 3),
            "search_end": round(search_end, 3),
            "acoustic_onset": round(acoustic_onset, 3) if acoustic_onset is not None else None,
            "acoustic_threshold": acoustic_threshold,
            "accepted": recovery_time is not None,
            "mode": recovery_mode or "unresolved-weak-prefix",
        }
        probes.append(probe)
        assignment["ctc_forward_supported_prefix_probe"] = probe
        if recovery_time is None:
            assignment["ctc_forward_prefix_probe_unresolved"] = True
            continue
        if not previous_end + 0.05 < recovery_time < next_time - 0.10:
            assignment["ctc_forward_prefix_probe_unresolved"] = True
            probe["accepted"] = False
            probe["mode"] = "candidate-outside-neighbor-bounds"
            continue
        timestamps[index] = recovery_time
        assignment["timestamp"] = round(recovery_time, 3)
        assignment["ctc_forward_supported_prefix_recovery"] = True
        assignment["ctc_forward_supported_prefix_recovery_trusted"] = True
        assignment["ctc_forward_supported_prefix_recovery_mode"] = recovery_mode
        assignment["ctc_forward_supported_prefix_recovery_from"] = round(current_time, 3)
        assignment["ctc_forward_supported_prefix_recovery_token_index"] = candidate_index
        if strong_detached_initial:
            assignment["ctc_effective_token_start_index"] = candidate_index
        assignment.pop("ctc_forward_prefix_probe_unresolved", None)
        assignment["timing_repair_source"] = (
            "torchaudio-mms-fa+forward-supported-prefix-recovery"
        )
        change = {
            **probe,
            "to": round(recovery_time, 3),
            "shift": round(recovery_time - current_time, 3),
        }
        changes.append(change)

    report["ctc_forward_supported_prefix_probes"] = probes
    report["ctc_forward_supported_prefix_probe_count"] = len(probes)
    report["ctc_forward_supported_prefix_recoveries"] = changes
    report["ctc_forward_supported_prefix_recovery_count"] = len(changes)
    if changes:
        annotate_ctc_boundary_evidence(assignments)
    return changes


def clear_stale_ctc_boundary_review_state(report: dict[str, object]) -> None:
    """Remove boundary reviews from an earlier validation stage before recomputing them."""
    assignments = report.get("assignments")
    if isinstance(assignments, list):
        for assignment in assignments:
            if not isinstance(assignment, dict):
                continue
            flags = assignment.get("flags")
            flag_list = list(flags) if isinstance(flags, list) else []
            removed = [flag for flag in flag_list if flag in CTC_BOUNDARY_REVIEW_FLAGS]
            had_boundary_state = bool(
                removed
                or assignment.get("ctc_unresolved_boundary")
                or assignment.get("ctc_right_edge_truncation")
            )
            if not had_boundary_state:
                continue
            remaining = [flag for flag in flag_list if flag not in CTC_BOUNDARY_REVIEW_FLAGS]
            if remaining:
                assignment["flags"] = remaining
            else:
                assignment.pop("flags", None)
                assignment["review_required"] = False
            assignment.pop("ctc_unresolved_boundary", None)
            assignment.pop("ctc_right_edge_truncation", None)
            assignment.pop("ctc_forward_prefix_probe_unresolved", None)

    suspicious = report.get("suspicious_alignments")
    if not isinstance(suspicious, list):
        return
    cleaned: list[dict[str, object]] = []
    for item in suspicious:
        if not isinstance(item, dict):
            continue
        flags = item.get("flags")
        flag_list = list(flags) if isinstance(flags, list) else []
        boundary_flags = [flag for flag in flag_list if flag in CTC_BOUNDARY_REVIEW_FLAGS]
        if not boundary_flags:
            cleaned.append(item)
            continue
        remaining = [flag for flag in flag_list if flag not in CTC_BOUNDARY_REVIEW_FLAGS]
        if not remaining:
            # Boundary-only findings are provisional and must disappear once the
            # current token spans no longer reproduce the issue.
            continue
        revised = dict(item)
        revised["flags"] = remaining
        revised.pop("ctc_boundary_evidence", None)
        revised.pop("ctc_right_edge_evidence", None)
        revised.pop("ctc_boundary_realign_attempt", None)
        cleaned.append(revised)
    report["suspicious_alignments"] = cleaned


def flag_unresolved_ctc_boundary_issues(
    entries: list[LyricEntry],
    report: dict[str, object],
    persist: bool = True,
) -> list[dict[str, object]]:
    """Fail closed only for boundary evidence that survives the final validation stage."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return []
    if persist:
        clear_stale_ctc_boundary_review_state(report)
    suspicious = report.get("suspicious_alignments")
    suspicious_items = list(suspicious) if isinstance(suspicious, list) else []
    by_entry = {
        int(item["entry"]): item
        for item in suspicious_items
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    unresolved: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            continue
        evidence = assignment.get("ctc_boundary_evidence")
        if not isinstance(evidence, dict):
            evidence = {}
        edge_evidence = assignment.get("ctc_right_edge_evidence")
        if not isinstance(edge_evidence, dict):
            edge_evidence = {}
        boundary_recovery_trusted = bool(
            assignment.get("ctc_boundary_realign_trusted")
            or assignment.get("ctc_crossline_initial_recovery_trusted")
            or assignment.get("ctc_forward_supported_prefix_recovery_trusted")
            or assignment.get("ctc_final_timing_recovery_trusted")
        )
        unresolved_boundary = (
            bool(evidence.get("realign_required"))
            or bool(assignment.get("ctc_final_timing_guard_unresolved"))
        ) and not boundary_recovery_trusted
        right_edge_pileup = bool(edge_evidence.get("right_edge_pileup"))
        if not unresolved_boundary and not right_edge_pileup:
            continue
        entry_number = index + 1
        flags = ["ctc_unresolved_boundary"] if unresolved_boundary else []
        if right_edge_pileup:
            flags.extend(["ctc_right_edge_pileup", "ctc_local_window_truncated"])
        if evidence.get("zero_boundary_gap"):
            flags.append("ctc_zero_boundary_gap")
        if evidence.get("low_confidence_prefix"):
            flags.append("ctc_low_confidence_prefix")
        if evidence.get("abnormal_prefix_compression"):
            flags.append("ctc_abnormal_prefix_compression")
        if evidence.get("ghost_weak_prefix"):
            flags.append("ctc_ghost_weak_prefix")
        if evidence.get("borrowed_tail_prefix"):
            flags.append("ctc_borrowed_tail_prefix")
        if evidence.get("near_gap_weak_prefix"):
            flags.append("ctc_near_gap_weak_prefix")
        if evidence.get("prior_line_truncation_contamination"):
            flags.append("ctc_prior_line_truncation_contamination")
        if evidence.get("timestamp_precedes_ctc_support"):
            flags.append("ctc_timestamp_precedes_ctc_support")
        if evidence.get("detached_weak_prefix"):
            flags.append("ctc_detached_weak_prefix")
        if evidence.get("unsupported_weak_prefix"):
            flags.append("ctc_unsupported_weak_prefix")
        if assignment.get("ctc_forward_prefix_probe_unresolved"):
            flags.append("ctc_forward_prefix_probe_unresolved")
        fracture_evidence = assignment.get("ctc_path_fracture_evidence")
        if isinstance(fracture_evidence, dict) and fracture_evidence.get("fractured"):
            flags.append("ctc_path_fracture")
        prefix_probe_evidence = assignment.get("ctc_prefix_probe_evidence")
        if isinstance(prefix_probe_evidence, dict) and not prefix_probe_evidence.get("accepted"):
            flags.append("ctc_prefix_probe_failed")
        if assignment.get("ctc_final_timing_guard_unresolved"):
            flags.append("ctc_final_timing_guard_unresolved")
        diagnostic = {
            "entry": entry_number,
            "flags": flags,
            "evidence": evidence,
            "right_edge_evidence": edge_evidence,
        }
        unresolved.append(diagnostic)
        if not persist:
            continue

        assignment_flags = assignment.get("flags")
        assignment["flags"] = list(dict.fromkeys([
            *(assignment_flags if isinstance(assignment_flags, list) else []),
            *flags,
        ]))
        assignment["ctc_unresolved_boundary"] = unresolved_boundary
        assignment["ctc_right_edge_truncation"] = right_edge_pileup
        assignment["review_required"] = True
        # Boundary/timestamp review is a timing failure, not a lyric-content
        # failure.  Do not mutate the whole-line content score here; the audit
        # reports content_trusted and timing_trusted separately.
        if isinstance(assignment.get("confidence"), (int, float)):
            assignment["confidence"] = min(float(assignment["confidence"]), 0.45)
        for key in (
            "timing_trusted",
            "timing_trusted_reason",
            "timing_trusted_sources",
            "timing_trusted_candidate_times",
        ):
            assignment.pop(key, None)
        item = by_entry.get(entry_number)
        if item is None:
            item = {
                "entry": entry_number,
                "text": entry_sung_text(entries[index]),
            }
            suspicious_items.append(item)
            by_entry[entry_number] = item
        existing_flags = item.get("flags")
        item["flags"] = list(dict.fromkeys([
            *(existing_flags if isinstance(existing_flags, list) else []),
            *flags,
        ]))
        item["severity"] = (
            "high"
            if right_edge_pileup
            or evidence.get("abnormal_prefix_compression")
            or evidence.get("ghost_weak_prefix")
            or evidence.get("detached_weak_prefix")
            else "medium"
        )
        item["review_required"] = True
        item["ctc_boundary_evidence"] = evidence
        item["ctc_right_edge_evidence"] = edge_evidence
        item["ctc_boundary_realign_attempt"] = assignment.get("ctc_boundary_realign_attempt")

    if not persist:
        return unresolved

    report["suspicious_alignments"] = suspicious_items
    report["ctc_unresolved_boundary_count"] = sum(
        1 for item in unresolved if "ctc_unresolved_boundary" in item.get("flags", [])
    )
    report["ctc_right_edge_pileup_count"] = sum(
        1 for item in unresolved if "ctc_right_edge_pileup" in item.get("flags", [])
    )
    report["ctc_unresolved_boundaries"] = unresolved
    report["suspicious_alignment_count"] = len(suspicious_items)
    report["review_required_count"] = sum(
        1 for item in suspicious_items if isinstance(item, dict) and item.get("review_required")
    )
    report["review_required"] = bool(report["review_required_count"])
    report["suspicious_alignment_severity_counts"] = {
        severity: sum(
            1 for item in suspicious_items
            if isinstance(item, dict) and item.get("severity") == severity
        )
        for severity in ("high", "medium", "low", "resolved")
    }
    update_report_confidence_metrics(report)
    return unresolved


def apply_ctc_boundary_validation_stage(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
    stage: str,
) -> tuple[list[float], dict[str, object], dict[str, object]]:
    """Run all boundary-changing CTC helpers in one stable order."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return timestamps, report, {}
    annotate_ctc_boundary_evidence(assignments)
    zero_gap_changes = apply_ctc_zero_gap_boundary_realign(
        audio_path, entries, timestamps, report, duration, args
    )
    if zero_gap_changes:
        refresh_ctc_confidence_diagnostics(entries, report)
    else:
        annotate_ctc_boundary_evidence(assignments)
    timestamps, report, crossline_changes = apply_ctc_crossline_initial_recovery(
        timestamps, report, duration
    )
    annotate_ctc_boundary_evidence(assignments)
    forward_prefix_changes = apply_ctc_forward_supported_prefix_recovery(
        audio_path, entries, timestamps, report, duration
    )
    annotate_ctc_boundary_evidence(assignments)
    timestamps, report, weak_prefix_changes = apply_ctc_weak_prefix_recovery(
        timestamps, report, duration
    )
    annotate_ctc_boundary_evidence(assignments)
    unresolved = flag_unresolved_ctc_boundary_issues(
        entries, report, persist=(stage != "initial-ctc")
    )
    result: dict[str, object] = {
        "stage": stage,
        "zero_gap_realignments": zero_gap_changes,
        "crossline_recoveries": crossline_changes,
        "forward_supported_prefix_recoveries": forward_prefix_changes,
        "weak_prefix_recoveries": weak_prefix_changes,
        "unresolved_boundaries": unresolved,
    }
    history = report.get("ctc_boundary_validation_stages")
    report["ctc_boundary_validation_stages"] = [
        *(history if isinstance(history, list) else []),
        result,
    ]
    return timestamps, report, result


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
        boundary_realign_trusted = bool(ctc_item.get("ctc_boundary_realign_trusted"))
        clear_boundary = bool(ctc_item.get("ctc_clear_boundary"))
        if abs(disagreement) < 1.10 and not (
            (prefix_recovered or crossline_recovered or boundary_realign_trusted)
            and abs(disagreement) >= 0.30
        ):
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
        if boundary_realign_trusted:
            target_time = ctc_time
            target_source = "ctc_boundary_realign"
        elif raw_high_conflict_anchor:
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
        if target_source == "ctc_boundary_realign":
            reason = "ctc-boundary-realign-over-whisperx"
        elif target_source == "raw_asr_conflict_anchor":
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
                "ctc_boundary_realign_trusted": boundary_realign_trusted,
                "ctc_boundary_realign_evidence": ctc_item.get("ctc_boundary_realign_evidence"),
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
                        assignments[index]["ctc_boundary_realign_trusted"] = bool(
                            change.get("ctc_boundary_realign_trusted")
                        )
                        assignments[index]["ctc_zero_gap_boundary_realign"] = bool(
                            change.get("ctc_boundary_realign_trusted")
                        )
                        assignments[index]["ctc_boundary_realign_evidence"] = change.get(
                            "ctc_boundary_realign_evidence"
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
            family_votes: dict[str, tuple[str, float, float]] = {}
            if (
                whisperx_score >= TRUSTED_ALIGNMENT_SCORE
                and abs(whisperx_time - selected_time) <= 0.75
            ):
                family_votes["whisper-family"] = (
                    "whisperx", whisperx_time, whisperx_score
                )
            if ctc_score >= 0.02 and abs(ctc_time - selected_time) <= 0.75:
                family_votes["ctc"] = ("ctc", ctc_time, ctc_score)
            if (
                raw_time is not None
                and raw_score is not None
                and raw_score >= TRUSTED_ALIGNMENT_SCORE
                and abs(raw_time - selected_time) <= 0.75
            ):
                existing_whisper = family_votes.get("whisper-family")
                if existing_whisper is None or raw_score > existing_whisper[2]:
                    family_votes["whisper-family"] = ("raw", raw_time, raw_score)
            votes = [
                (source, time)
                for source, time, _score in family_votes.values()
            ]
            vote_times = [time for _, time in votes]
            consensus = len(family_votes) >= 2 and (
                max(vote_times) - min(vote_times) <= 0.75
                if vote_times else False
            )
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


def _legacy_refinement_row_has_actionable_flags(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    flags = row.get("flags")
    if not isinstance(flags, list):
        return False
    return any(
        str(flag) in ACTIONABLE_SUSPICIOUS_FLAGS
        or str(flag) in ACTIONABLE_TIMING_FAILURES
        for flag in flags
    )


def apply_ctc_micro_refinement_to_whisperx(
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

    refined = list(whisperx_timestamps)
    changes: list[dict[str, object]] = []
    raw_assignments = raw_report.get("assignments") if isinstance(raw_report, dict) else None
    max_items = min(len(refined), len(whisperx_assignments), len(ctc_assignments))
    suspicious_rows = [
        item
        for item in whisperx_report.get("suspicious_alignments", [])
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    ]
    suspicious_entries = {int(item["entry"]) for item in suspicious_rows}
    actionable_suspicious_entries = {
        int(item["entry"])
        for item in suspicious_rows
        if _legacy_refinement_row_has_actionable_flags(item)
    }
    for index in range(max_items):
        entry_number = index + 1
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
        raw_time: float | None = None
        raw_score: float | None = None
        if isinstance(raw_assignments, list) and index < len(raw_assignments):
            raw_item = raw_assignments[index]
            if isinstance(raw_item, dict):
                try:
                    raw_time = float(raw_item.get("timestamp"))
                    raw_score = float(raw_item.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    raw_time = None
                    raw_score = None

        suspicious_entry = entry_number in suspicious_entries
        # WhisperX and raw ASR often share the same decoded segment.  When both
        # land slightly late at the same timestamp but CTC has a strong earlier
        # first-token path, treat that as an onset correction rather than a
        # global backend preference change.
        asr_same_late_segment = (
            not suspicious_entry
            and raw_time is not None
            and raw_score is not None
            and raw_score >= 0.65
            and abs(raw_time - whisperx_time) <= 0.08
            and -0.55 <= delta <= -0.18
            and ctc_score >= 0.18
            and whisperx_score >= 0.65
        )
        if not suspicious_entry and not asr_same_late_segment:
            continue
        if asr_same_late_segment:
            spans = ctc_item.get("ctc_token_spans")
            first_token_score = 0.0
            if isinstance(spans, list) and spans:
                try:
                    first_token_score = float(spans[0].get("score", 0.0) or 0.0)
                except (AttributeError, TypeError, ValueError):
                    first_token_score = 0.0
            if first_token_score < 0.18:
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
                "reason": "asr-late-ctc-onset-refinement" if asr_same_late_segment else "small-ctc-whisperx-refinement",
                "whisperx_score": round(whisperx_score, 3),
                "ctc_score": round(ctc_score, 6),
                "ctc_token_spans": ctc_item.get("ctc_token_spans"),
                "ctc_boundary_realign_trusted": bool(ctc_item.get("ctc_boundary_realign_trusted")),
                "ctc_boundary_realign_evidence": ctc_item.get("ctc_boundary_realign_evidence"),
                "raw_timestamp": round(raw_time, 3) if raw_time is not None else None,
                "raw_score": round(raw_score, 3) if raw_score is not None else None,
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
            assignments[index]["ctc_boundary_realign_trusted"] = bool(
                change.get("ctc_boundary_realign_trusted")
            )
            assignments[index]["ctc_zero_gap_boundary_realign"] = bool(
                change.get("ctc_boundary_realign_trusted")
            )
            assignments[index]["ctc_boundary_realign_evidence"] = change.get(
                "ctc_boundary_realign_evidence"
            )
            refinement_actionable = (index + 1) in actionable_suspicious_entries
            assignments[index]["timing_trusted"] = not refinement_actionable
            assignments[index]["timing_trusted_reason"] = (
                "actionable-suspicious-review-preserved"
                if refinement_actionable
                else "small-ctc-whisperx-consensus"
            )
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
        refinement_actionable = entry in actionable_suspicious_entries
        item["flags"] = list(dict.fromkeys([
            *flags,
            "ctc_micro_refinement",
            *(([]) if refinement_actionable else ["timing_trusted"]),
        ]))
        if refinement_actionable:
            item["review_required"] = True
        else:
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
        if (
            assignment.get("ctc_micro_refinement")
            or assignment.get("ctc_local_fusion")
            or assignment.get("ctc_boundary_realign_trusted")
        ):
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
            and len(normalize_match_text(str(risk.get("text", "")) if isinstance(risk, dict) else "")) <= 12
            and current_time - previous_time >= 3.50
            and next_time - current_time <= 2.20
        ):
            candidate = first_local_onset(
                features,
                max(previous_time + 0.40, current_time - 6.80),
                current_time - 0.45,
                threshold=0.65,
            )
            if candidate is None:
                candidate = first_local_onset(
                    features,
                    max(previous_time + 0.40, current_time - 6.80),
                    current_time - 0.45,
                    threshold=0.55,
                )
            if candidate is not None and not (1.20 <= current_time - candidate <= 6.50):
                candidate = None
            reason = "short-duplicate-bad-split-acoustic-backtrack"
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
        assignment["timing_trusted"] = False
        assignment["timing_trusted_reason"] = (
            "acoustic-boundary-refinement-diagnostic-only"
        )
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
            original_actionable = _legacy_refinement_row_has_actionable_flags(risk)
            risk["flags"] = list(dict.fromkeys([
                *risk_flags,
                "acoustic_boundary_refinement",
            ]))
            risk["review_required"] = bool(risk.get("review_required")) or original_actionable
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


def _invalidate_timing_trust_after_timestamp_mutation(
    assignment: dict[str, object],
    reason: str,
) -> None:
    """Discard timing authority computed for a superseded clock."""
    assignment["timing_trusted"] = False
    assignment["timing_audit_trusted"] = False
    assignment["overall_trusted"] = False
    assignment["verification_status"] = "unverified"
    assignment["timing_trust_invalidated_reason"] = reason
    for key in ("timing_trusted_reason", "timing_trusted_sources", "timing_trusted_candidate_times"):
        assignment.pop(key, None)


def apply_hybrid_ctc_opening_onset_backtrack(
    audio_path: Path,
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    """Backtrack a CTC-fused line when its opening tokens are visibly late.

    This is intentionally narrower than a general acoustic snap: it only runs
    after CTC has already beaten WhisperX by a large margin, and only when the
    first CTC tokens have weak posterior support while nearby later tokens are
    strong.  That pattern means CTC found the right lyric region but latched
    onto the first clear vowel/consonant cluster instead of the display onset.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps):
        return timestamps, report, []
    try:
        features = analyze_audio(audio_path, duration)
    except LrcError:
        return timestamps, report, []

    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict) or not assignment.get("ctc_local_fusion"):
            continue
        if assignment.get("ctc_boundary_realign_trusted"):
            continue
        if assignment.get("ctc_clear_boundary"):
            continue
        try:
            current_time = float(assignment.get("timestamp", refined[index]) or refined[index])
            ctc_score = float(assignment.get("ctc_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if ctc_score < 0.12:
            continue
        spans = assignment.get("ctc_token_spans")
        if not isinstance(spans, list) or len(spans) < 5:
            continue
        first_scores: list[float] = []
        later_scores: list[float] = []
        for span in spans[:4]:
            if not isinstance(span, dict):
                continue
            try:
                first_scores.append(float(span.get("score", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
        for span in spans[4:10]:
            if not isinstance(span, dict):
                continue
            try:
                later_scores.append(float(span.get("score", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
        if not first_scores or max(first_scores) > 0.08:
            continue
        if not later_scores or max(later_scores) < 0.30:
            continue
        previous_time = refined[index - 1] if index else 0.0
        next_time = refined[index + 1] if index + 1 < len(refined) else duration
        search_start = max(previous_time + 0.20, current_time - 0.90)
        search_end = current_time - 0.08
        if search_end - search_start < 0.20:
            continue
        candidate = first_local_onset(features, search_start, search_end, threshold=0.65)
        if candidate is None:
            candidate = first_local_onset(features, search_start, search_end, threshold=0.55)
        if candidate is None:
            continue
        shift = current_time - candidate
        if not (0.25 <= shift <= 0.80):
            continue
        if not (previous_time + 0.05 < candidate < next_time - 0.05):
            continue

        refined[index] = candidate
        assignment["timestamp"] = round(candidate, 3)
        _invalidate_timing_trust_after_timestamp_mutation(assignment, "ctc-opening-onset-backtrack")
        assignment["ctc_opening_onset_backtrack"] = True
        assignment["ctc_opening_onset_backtrack_from"] = round(current_time, 3)
        assignment["timing_repair_source"] = "ctc-local-fusion+acoustic-opening-onset"
        changes.append(
            {
                "entry": index + 1,
                "from": round(current_time, 3),
                "to": round(candidate, 3),
                "shift": round(shift, 3),
                "ctc_score": round(ctc_score, 6),
                "search_start": round(search_start, 3),
                "search_end": round(search_end, 3),
                "reason": "weak-opening-ctc-local-fusion-acoustic-onset",
            }
        )
    if not changes:
        return timestamps, report, []

    report["ctc_opening_onset_backtrack_count"] = len(changes)
    report["ctc_opening_onset_backtracks"] = changes
    changes_by_entry = {int(change["entry"]): change for change in changes if isinstance(change.get("entry"), int)}
    suspicious = report.get("suspicious_alignments")
    suspicious_items = list(suspicious) if isinstance(suspicious, list) else []
    for item in suspicious_items:
        if not isinstance(item, dict):
            continue
        entry = item.get("entry")
        if not isinstance(entry, int) or entry not in changes_by_entry:
            continue
        change = changes_by_entry[entry]
        candidates = item.get("candidate_timestamps")
        if not isinstance(candidates, dict):
            candidates = {}
        candidates["output"] = change["to"]
        candidates["acoustic_onset"] = change["to"]
        item["candidate_timestamps"] = candidates
        flags = item.get("flags")
        if not isinstance(flags, list):
            flags = []
        item["flags"] = list(dict.fromkeys([*flags, "ctc_opening_onset_backtrack"]))
        item["chosen_time"] = change["to"]
    update_report_confidence_metrics(report)
    return refined, report, changes


def apply_vocal_onset_tiebreak(
    timestamps: list[float],
    report: dict[str, object],
    ctc_report: dict[str, object],
    audio_path: Path,
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    features, unavailable = vocal_onset_features(audio_path, duration, args)
    if features is not None:
        participation = _backend_participation(report)
        participation["demucs_device"] = "cuda"
        participation["onset_stem_used"] = True
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
        if assignment.get("ctc_boundary_realign_trusted"):
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
        _invalidate_timing_trust_after_timestamp_mutation(assignment, "demucs-vocal-onset-tiebreak")
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
    if changes:
        update_report_confidence_metrics(report)
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
            existing["severity"] = _merge_suspicious_severity(
                existing.get("severity"),
                _ctc_disagreement_severity(disagreement),
            )
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
                "severity": _ctc_disagreement_severity(disagreement),
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
    timestamps: list[float],
    report: dict[str, object],
    raw_report: dict[str, object],
    entries: list[LyricEntry] | None = None,
) -> None:
    """Use raw ASR as a local diagnostic without letting a bad run poison CTC.

    Raw ASR has no guarantee that its matched segments retain a one-to-one
    relationship with the supplied lyric list.  In particular, it can drift by
    almost a constant amount for an entire verse, or continue emitting
    timestamps after matching has collapsed.  Those are useful *section*
    diagnostics, but are not independent per-line evidence against a
    known-lyric CTC path.
    """
    assignments = report.get("assignments")
    raw_assignments = raw_report.get("assignments")
    if not isinstance(assignments, list) or not isinstance(raw_assignments, list):
        return
    suspicious = report.get("suspicious_alignments")
    if not isinstance(suspicious, list):
        suspicious = []
        report["suspicious_alignments"] = suspicious

    observations: list[dict[str, object]] = []
    for index, current_time in enumerate(timestamps):
        if index >= len(assignments) or index >= len(raw_assignments):
            continue
        raw_assignment = raw_assignments[index]
        if not isinstance(raw_assignment, dict):
            continue
        try:
            raw_time = float(raw_assignment.get("timestamp"))
            raw_score = float(raw_assignment.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        observations.append(
            {
                "index": index,
                "raw_time": raw_time,
                "raw_score": raw_score,
                "borrowed": bool(raw_assignment.get("borrowed")),
                # Consensus evidence must come from a real raw-ASR segment, not
                # an interpolated/borrowed placeholder row.
                "segment_bound": raw_assignment.get("segment") is not None,
                "delta": raw_time - current_time,
            }
        )

    # A same-direction, near-constant offset across three or more neighboring
    # entries is raw-ASR clock/segment drift.  Preserve it as one diagnostic
    # rather than turning every lyric in the verse into a separate failure.
    drift_runs: list[dict[str, object]] = []
    current_run: list[dict[str, object]] = []
    for observation in observations:
        raw_score = float(observation["raw_score"])
        delta = float(observation["delta"])
        eligible = (
            not bool(observation["borrowed"])
            and raw_score >= 0.50
            and abs(delta) >= 0.75
        )
        same_direction = (
            not current_run
            or (float(current_run[-1]["delta"]) * delta > 0)
        )
        close_offset = (
            not current_run
            or abs(float(current_run[-1]["delta"]) - delta) <= 0.65
        )
        run_deltas = [float(item["delta"]) for item in current_run]
        bounded_offset_range = (
            not current_run
            or max([*run_deltas, delta]) - min([*run_deltas, delta]) <= 0.65
        )
        consecutive = not current_run or int(observation["index"]) == int(current_run[-1]["index"]) + 1
        if eligible and same_direction and close_offset and bounded_offset_range and consecutive:
            current_run.append(observation)
            continue
        if len(current_run) >= 3:
            drift_runs.append(
                {
                    "entries": [int(item["index"]) + 1 for item in current_run],
                    "direction": "raw_earlier" if float(current_run[0]["delta"]) < 0 else "raw_later",
                    "median_delta_seconds": round(float(statistics.median(float(item["delta"]) for item in current_run)), 3),
                    "delta_range_seconds": round(
                        max(float(item["delta"]) for item in current_run)
                        - min(float(item["delta"]) for item in current_run),
                        3,
                    ),
                    "mean_raw_score": round(
                        sum(float(item["raw_score"]) for item in current_run) / len(current_run), 3
                    ),
                }
            )
        current_run = [observation] if eligible else []
    if len(current_run) >= 3:
        drift_runs.append(
            {
                "entries": [int(item["index"]) + 1 for item in current_run],
                "direction": "raw_earlier" if float(current_run[0]["delta"]) < 0 else "raw_later",
                "median_delta_seconds": round(float(statistics.median(float(item["delta"]) for item in current_run)), 3),
                "delta_range_seconds": round(
                    max(float(item["delta"]) for item in current_run)
                    - min(float(item["delta"]) for item in current_run),
                    3,
                ),
                "mean_raw_score": round(
                    sum(float(item["raw_score"]) for item in current_run) / len(current_run), 3
                ),
            }
        )
    systematic_drift_entries = {
        entry for drift_run in drift_runs for entry in drift_run["entries"]
    }

    # Raw ASR is already executed as an independent diagnostic backend.  When a
    # real, high-confidence, non-borrowed raw segment independently agrees with
    # the CTC line onset inside a tight tolerance, preserve that agreement as an
    # in-process authority.  It is intentionally runtime-only: serialized report
    # metadata cannot recreate independent evidence on a later run.
    raw_ctc_consensus: dict[int, dict[str, object]] = {}
    for observation in observations:
        index = int(observation["index"])
        entry_number = index + 1
        raw_score = float(observation["raw_score"])
        delta = float(observation["delta"])
        occurrence_binding = (
            _entry_occurrence_binding(entries, raw_assignments, index)
            if entries is not None and len(entries) == len(raw_assignments)
            else None
        )
        if (
            raw_score < RAW_CTC_CONSENSUS_MIN_SCORE
            or bool(observation["borrowed"])
            or not bool(observation["segment_bound"])
            or entry_number in systematic_drift_entries
            or abs(delta) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            or occurrence_binding is None
        ):
            continue
        raw_time = float(observation["raw_time"])
        ctc_time = float(timestamps[index])
        evidence_payload = {
            "entry_index": index,
            "raw_time": round(raw_time, 6),
            "ctc_time": round(ctc_time, 6),
            "raw_score": round(raw_score, 6),
            "max_delta_seconds": RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
            "occurrence_binding": occurrence_binding,
            "producer": "raw-asr-consensus",
        }
        raw_ctc_consensus[index] = {
            **evidence_payload,
            "evidence_revision": _timing_content_digest(evidence_payload),
        }
    report[_RUNTIME_RAW_CTC_CONSENSUS_KEY] = raw_ctc_consensus
    report["raw_ctc_onset_consensus"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "min_raw_score": RAW_CTC_CONSENSUS_MIN_SCORE,
        "max_delta_seconds": RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
        "candidate_count": len(raw_ctc_consensus),
        "entries": [index + 1 for index in sorted(raw_ctc_consensus)],
    }
    report["raw_asr_systematic_drift_runs"] = drift_runs
    report["raw_asr_zero_score_rejections"] = sum(
        1 for item in observations if float(item["raw_score"]) <= 0.0
    )
    by_entry = {
        item.get("entry"): item
        for item in suspicious
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    for observation in observations:
        index = int(observation["index"])
        if index >= len(assignments):
            continue
        assignment = assignments[index]
        if not isinstance(assignment, dict):
            continue
        current_time = timestamps[index]
        raw_time = float(observation["raw_time"])
        raw_score = float(observation["raw_score"])
        borrowed = bool(observation["borrowed"])
        # score=0 was assigned only as a draft placeholder.  It is explicitly
        # not a competing alignment candidate and must not trigger review.
        if raw_score < 0.50:
            continue
        if index + 1 in systematic_drift_entries:
            continue
        disagreement = abs(raw_time - current_time)
        threshold = 0.45 if raw_score >= 0.90 and not borrowed else 0.75
        if disagreement < threshold:
            continue
        entry_number = index + 1
        assignment["unresolved_raw_ctc_disagreement"] = round(disagreement, 3)
        severity = _raw_disagreement_severity(raw_score, disagreement)
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
        existing["severity"] = _merge_suspicious_severity(
            existing.get("severity"), severity
        )
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

    for drift_run in drift_runs:
        entries_in_run = drift_run["entries"]
        if not isinstance(entries_in_run, list) or not entries_in_run:
            continue
        entry_number = int(entries_in_run[0])
        existing = by_entry.get(entry_number)
        if existing is None:
            existing = {
                "entry": entry_number,
                "review_required": False,
                "severity": "medium",
                "flags": [],
                "candidate_timestamps": {},
            }
            suspicious.append(existing)
            by_entry[entry_number] = existing
        flags = existing.get("flags")
        if not isinstance(flags, list):
            flags = []
            existing["flags"] = flags
        if "raw_asr_systematic_drift" not in flags:
            flags.append("raw_asr_systematic_drift")
        existing["raw_asr_systematic_drift"] = drift_run
    review_count = sum(
        1 for item in suspicious if isinstance(item, dict) and bool(item.get("review_required"))
    )
    report["review_required_count"] = review_count
    report["review_required"] = review_count > 0
    update_report_confidence_metrics(report)


def first_local_onset(
    features: AudioFeatures,
    start: float,
    end: float,
    threshold: float = 0.55,
    *,
    edge_guard_seconds: float = 0.20,
) -> float | None:
    if end <= start:
        return None
    effective_start = start + max(0.0, edge_guard_seconds)
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


def should_accept_ctc_boundary_realign(
    current_evidence: dict[str, object],
    local_evidence: dict[str, object],
    current_score: float,
    local_score: float,
    current_time: float,
    local_time: float,
    acoustic_onset_time: float | None,
) -> tuple[bool, list[str]]:
    """Require independent improvements before replacing a collapsed boundary."""
    reasons: list[str] = []
    if not current_evidence.get("realign_required"):
        return False, ["current-boundary-not-pathological"]
    if local_time - current_time < 0.30:
        return False, ["local-shift-too-small"]
    if local_score < max(0.05, current_score * 1.05):
        return False, ["local-ctc-score-not-improved"]

    current_gap = float(current_evidence.get("boundary_gap_seconds", 0.0) or 0.0)
    local_gap = float(local_evidence.get("boundary_gap_seconds", 0.0) or 0.0)
    if local_gap < 0.08 or local_gap - current_gap < 0.08:
        return False, ["local-boundary-gap-not-improved"]
    reasons.append("boundary-gap-improved")
    current_previous_score = float(current_evidence.get("previous_token_score", 0.0) or 0.0)
    local_previous_score = float(local_evidence.get("previous_token_score", 0.0) or 0.0)
    if local_previous_score >= max(0.08, current_previous_score + 0.08):
        reasons.append("previous-line-tail-evidence-improved")

    current_mean = float(current_evidence.get("prefix_mean_score", 0.0) or 0.0)
    local_mean = float(local_evidence.get("prefix_mean_score", 0.0) or 0.0)
    current_weak = float(current_evidence.get("prefix_weak_ratio", 1.0) or 1.0)
    local_weak = float(local_evidence.get("prefix_weak_ratio", 1.0) or 1.0)
    prefix_improved = (
        local_mean >= current_mean + 0.02
        or local_weak <= current_weak - 0.25
        or (
            bool(current_evidence.get("abnormal_prefix_compression"))
            and not bool(local_evidence.get("abnormal_prefix_compression"))
        )
    )
    if not prefix_improved:
        return False, [*reasons, "local-prefix-not-improved"]
    reasons.append("prefix-evidence-improved")

    acoustic_supported = acoustic_onset_time is not None and abs(acoustic_onset_time - local_time) <= 0.30
    local_first = float(local_evidence.get("first_token_score", 0.0) or 0.0)
    reliable_index = local_evidence.get("next_reliable_token_index")
    reliable_score = float(local_evidence.get("next_reliable_token_score", 0.0) or 0.0)
    reliable_delay = local_evidence.get("reliable_token_delay_seconds")
    next_reliable_supported = (
        isinstance(reliable_index, int)
        and isinstance(reliable_delay, (int, float))
        and float(reliable_delay) <= 0.75
        and reliable_score >= 0.08
    )
    phonetic_supported = next_reliable_supported and (local_first >= 0.03 or local_mean >= 0.04)
    if not next_reliable_supported:
        return False, [*reasons, "no-nearby-reliable-token-evidence"]
    reasons.append("next-reliable-token-support")
    if not acoustic_supported and not phonetic_supported:
        return False, [*reasons, "no-acoustic-or-prefix-phonetic-onset-support"]
    reasons.append("acoustic-onset-support" if acoustic_supported else "multi-token-phonetic-support")
    return True, reasons


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
    attempts: list[dict[str, object]] = []
    acoustic_features: AudioFeatures | None = None
    acoustic_checked = False
    annotate_ctc_boundary_evidence(assignments)
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
            current_evidence = ctc_prefix_boundary_evidence(current, boundary_gap)
            if not current_evidence.get("realign_required"):
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
            proc = _run_model_helper_command(command, mode="ctc")
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
            previous_row = rows[0] if isinstance(rows[0], dict) else {}
            previous_local_spans = previous_row.get("token_spans")
            try:
                previous_local_end, previous_local_score = max(
                    (
                        (
                            float(span.get("end")),
                            float(span.get("score", 0.0) or 0.0),
                        )
                        for span in previous_local_spans
                        if isinstance(span, dict)
                    ),
                    key=lambda item: item[0],
                )
            except (TypeError, ValueError):
                previous_local_end = previous_end
                previous_local_score = None
            local_evidence = ctc_prefix_boundary_evidence(
                {
                    "ctc_score": local_score,
                    "ctc_token_spans": row.get("token_spans"),
                    "ctc_previous_token_score": previous_local_score,
                },
                local_time - previous_local_end,
            )
            if not acoustic_checked:
                acoustic_checked = True
                try:
                    acoustic_features = analyze_vocal_onsets(audio_path, duration)
                except LrcError:
                    acoustic_features = None
            acoustic_onset: float | None = None
            acoustic_strength: float | None = None
            if acoustic_features is not None:
                search_start = max(previous_local_end + 0.02, local_time - 0.45)
                search_end = min(end_time - 0.05, local_time + 0.35)
                acoustic_onset = first_local_onset(acoustic_features, search_start, search_end, threshold=0.45)
                if acoustic_onset is None:
                    acoustic_onset = first_local_onset(acoustic_features, search_start, search_end, threshold=0.35)
                if acoustic_onset is not None and len(acoustic_features.frame_times):
                    nearest = int(np.argmin(np.abs(acoustic_features.frame_times - acoustic_onset)))
                    acoustic_strength = float(acoustic_features.onset_strength[nearest])
            accepted, acceptance_reasons = should_accept_ctc_boundary_realign(
                current_evidence,
                local_evidence,
                current_score,
                local_score,
                current_time,
                local_time,
                acoustic_onset,
            )
            previous_bound = timestamps[index - 1] + 0.05
            next_bound = timestamps[index + 1] - 0.05 if index + 1 < len(timestamps) else duration - 0.05
            if accepted and not (previous_bound < local_time < next_bound):
                accepted = False
                acceptance_reasons = [*acceptance_reasons, "candidate-outside-neighbor-bounds"]
            repaired_time = local_time
            if (
                accepted
                and acoustic_onset is not None
                and previous_local_end + 0.05 < acoustic_onset <= local_time + 0.08
                and 0.0 <= local_time - acoustic_onset <= 0.25
                and acoustic_onset - current_time >= 0.30
            ):
                repaired_time = acoustic_onset
                acceptance_reasons = [*acceptance_reasons, "bounded-acoustic-consonant-onset"]
            attempt = {
                "entry": index + 1,
                "from": round(current_time, 3),
                "candidate": round(repaired_time, 3),
                "local_ctc_candidate": round(local_time, 3),
                "accepted": accepted,
                "current_ctc_score": round(current_score, 6),
                "local_ctc_score": round(local_score, 6),
                "current_evidence": current_evidence,
                "local_evidence": local_evidence,
                "acoustic_onset": round(acoustic_onset, 3) if acoustic_onset is not None else None,
                "acoustic_onset_strength": round(acoustic_strength, 3) if acoustic_strength is not None else None,
                "decision_reasons": acceptance_reasons,
            }
            attempts.append(attempt)
            current["ctc_boundary_realign_attempt"] = attempt
            if not accepted:
                continue
            timestamps[index] = repaired_time
            current["timestamp"] = round(repaired_time, 3)
            current["ctc_score"] = row.get("ctc_score")
            current["ctc_token_spans"] = row.get("token_spans")
            current["ctc_first_token_candidates"] = row.get("first_token_candidates")
            current["ctc_zero_gap_boundary_realign"] = True
            current["ctc_boundary_realign_trusted"] = True
            current["ctc_boundary_realign_evidence"] = attempt
            current.pop("ctc_unresolved_boundary", None)
            current["timing_repair_source"] = "torchaudio-mms-fa+zero-gap-boundary-realign"
            changes.append({**attempt, "to": round(repaired_time, 3)})
    if changes:
        annotate_ctc_boundary_evidence(assignments)
        existing_changes = report.get("ctc_zero_gap_boundary_realign")
        report["ctc_zero_gap_boundary_realign"] = [
            *(existing_changes if isinstance(existing_changes, list) else []),
            *changes,
        ]
    if attempts:
        report["ctc_boundary_realign_attempts"] = attempts
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
    except LrcError as exc:
        _record_backend_participation_failure(
            report, "apply_ctc_acoustic_backtrack", exc
        )
        return []

    changes: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    for index in range(1, len(timestamps)):
        assignment = assignments[index] if isinstance(assignments[index], dict) else {}
        if assignment.get("timing_repair") != "ctc-forced-align":
            continue
        try:
            ctc_score = float(assignment.get("ctc_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            ctc_score = 0.0
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
        candidate_score: float | None = None
        candidate_ratio: float | None = None
        first_candidates = assignment.get("ctc_first_token_candidates")
        parsed_candidates: list[tuple[float, float]] = []
        if isinstance(first_candidates, list):
            for item in first_candidates:
                if not isinstance(item, dict):
                    continue
                try:
                    parsed_candidates.append(
                        (float(item.get("time")), float(item.get("score", 0.0) or 0.0))
                    )
                except (TypeError, ValueError):
                    continue
        try:
            previous_token_end = float(assignment.get("ctc_previous_token_end"))
        except (TypeError, ValueError):
            previous_token_end = None
        tail_conflicts = [
            (time_value, score)
            for time_value, score in parsed_candidates
            if previous_token_end is not None
            and time_value > previous_time + 0.20
            and time_value <= previous_token_end + 0.20
        ]
        for time_value, score in tail_conflicts:
            rejection = {
                "entry": index + 1,
                "from": round(current_time, 3),
                "candidate": round(time_value, 3),
                "reason": "candidate-borrows-previous-line-tail",
                "previous_token_end": round(previous_token_end, 3),
                "candidate_gap_seconds": round(time_value - previous_token_end, 3),
                "candidate_score": round(score, 6),
                "lyric": entries[index].lines[0] if entries[index].lines else "",
            }
            rejections.append(rejection)
            assignment["ctc_acoustic_backtrack_rejected"] = True
            assignment["ctc_acoustic_backtrack_rejection_reason"] = rejection["reason"]
        top_score = max((score for _time, score in parsed_candidates), default=0.0)
        eligible = [
            (time_value, score, score / top_score if top_score > 0.0 else 0.0)
            for time_value, score in parsed_candidates
            if previous_time + 0.20 < time_value < current_time - 0.20
            and (previous_token_end is None or time_value - previous_token_end > 0.20)
            and current_time - time_value <= 2.50
            and score >= 0.01
            and (top_score <= 0.0 or score >= top_score * 0.20)
        ]
        if ctc_score <= 0.28 and eligible:
            candidate, candidate_score, candidate_ratio = max(
                eligible, key=lambda value: (value[1], -value[0])
            )
            mode = "ctc-first-token-posterior"
            search_start = max(previous_time + 0.20, current_time - 2.50)
            search_end = current_time - 0.05
            min_shift = 0.20
            max_shift = 2.50
        elif ctc_score <= 0.05:
            prefix_evidence = ctc_prefix_boundary_evidence(
                assignment, assignment.get("ctc_boundary_gap_seconds")
            )
            if prefix_evidence.get("unsupported_weak_prefix") or prefix_evidence.get(
                "detached_weak_prefix"
            ):
                rejections.append(
                    {
                        "entry": index + 1,
                        "from": round(current_time, 3),
                        "mode": "ctc-short-acoustic-onset",
                        "reason": "weak-prefix-has-no-phonetic-support-for-backtrack",
                        "ctc_boundary_evidence": prefix_evidence,
                        "lyric": entries[index].lines[0] if entries[index].lines else "",
                    }
                )
                assignment["ctc_acoustic_backtrack_rejected"] = True
                assignment["ctc_acoustic_backtrack_rejection_reason"] = (
                    "weak-prefix-has-no-phonetic-support-for-backtrack"
                )
                continue
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
        hypothesis_spans: list[dict[str, object]] = []
        acoustic_strength: float | None = None
        acoustic_onset_time: float | None = None
        direct_onset_time: float | None = None
        identity_support: EvidenceState = "unavailable"
        acoustic_support: EvidenceState = "unavailable"
        direct_onset_support: EvidenceState = "unavailable"
        if mode == "ctc-first-token-posterior":
            token = ""
            romaji = assignment.get("romaji")
            if isinstance(romaji, str) and romaji:
                token = romaji[0]
            hypothesis_spans = [{
                "char": token,
                "start": round(float(candidate), 6),
                "end": round(float(candidate) + HOP_SECONDS, 6),
                "score": round(float(candidate_score), 8),
            }]
            identity_support = "supported"
        else:
            if len(features.frame_times):
                nearest_index = int(np.argmin(np.abs(features.frame_times - float(candidate))))
                acoustic_strength = float(features.onset_strength[nearest_index])
            acoustic_onset_time = float(candidate)
            direct_onset_time = float(candidate)
            acoustic_support = "supported"
            direct_onset_support = "supported"
        hypothesis = record_alignment_hypothesis(
            assignment,
            source="ctc-acoustic-backtrack",
            stage="ctc-acoustic-backtrack",
            raw_time=float(candidate),
            confidence=(float(candidate_score) if mode == "ctc-first-token-posterior" else float(acoustic_strength or 0.0)),
            spans=hypothesis_spans,
            identity_support=identity_support,
            sequence_support="supported",
            acoustic_support=acoustic_support,
            direct_onset_support=direct_onset_support,
            acoustic_strength=acoustic_strength,
            acoustic_onset_time=acoustic_onset_time,
            direct_onset_time=direct_onset_time,
            parent_source="ctc-global-mms",
            parent_time=current_time,
            window={"start": round(search_start, 3), "end": round(search_end, 3)},
            mode=mode,
        )
        assignment["ctc_acoustic_backtrack_candidate_generated"] = True
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
    report["ctc_acoustic_backtrack_authoritative"] = False
    report["ctc_acoustic_backtrack_mutates_baseline"] = False
    report["ctc_acoustic_backtrack_rejections"] = rejections
    report["ctc_acoustic_backtrack_rejection_count"] = len(rejections)
    return changes


def lyric_prefix_probe_text(entry: LyricEntry, minimum_chars: int = 8) -> str:
    """Return a compact leading lyric phrase suitable for an independent CTC probe.

    The probe deliberately uses only the opening phrase, so a fractured whole-line
    path cannot drag the candidate to a later repeated word or the right edge.
    """
    text = entry_sung_text(entry).strip()
    if not text:
        return ""
    # Preserve natural phrase boundaries; Japanese full-width spaces are common.
    parts = [
        part.strip(" ,，、。.!！?？…:：;；()（）[]【】{}「」『』\"")
        for part in re.split(r"[\s\u3000]+", text)
        if part.strip()
    ]
    if not parts:
        return text
    selected: list[str] = []
    compact_len = 0
    for part in parts:
        if not part:
            continue
        selected.append(part)
        compact_len += len(normalize_match_text(part))
        if compact_len >= minimum_chars:
            break
        # Two lexical chunks are enough for an onset probe even when the first
        # chunk is very short.
        if len(selected) >= 2:
            break
    return " ".join(selected).strip()


def ctc_path_fracture_evidence(assignment: dict[str, object]) -> dict[str, object]:
    """Detect a forced path that splits one lyric into disconnected time islands."""
    spans = assignment.get("ctc_token_spans")
    if not isinstance(spans, list) or len(spans) < 4:
        return {"fractured": False, "max_gap_seconds": 0.0, "gap_after_token_index": None}
    parsed: list[tuple[float, float, float]] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        try:
            parsed.append((
                float(span.get("start")),
                float(span.get("end")),
                float(span.get("score", 0.0) or 0.0),
            ))
        except (TypeError, ValueError):
            continue
    if len(parsed) < 4:
        return {"fractured": False, "max_gap_seconds": 0.0, "gap_after_token_index": None}
    gaps = [max(0.0, parsed[i + 1][0] - parsed[i][1]) for i in range(len(parsed) - 1)]
    if not gaps:
        return {"fractured": False, "max_gap_seconds": 0.0, "gap_after_token_index": None}
    max_gap = max(gaps)
    gap_index = gaps.index(max_gap)
    positive = sorted(g for g in gaps if g > 0.0)
    median_gap = float(statistics.median(positive)) if positive else 0.0
    # A multi-second hole inside one lyric is categorically different from
    # ordinary mora spacing.  Scale against the line's normal token gap too,
    # so slower songs are not penalised.
    fractured = bool(
        max_gap >= 3.0
        and (
            median_gap <= 0.0
            or max_gap >= max(3.0, median_gap * 8.0)
        )
    )
    return {
        "fractured": fractured,
        "max_gap_seconds": round(max_gap, 3),
        "gap_after_token_index": gap_index if fractured else None,
        "median_positive_gap_seconds": round(median_gap, 3),
    }


def run_ctc_prefix_probe(
    audio_path: Path,
    entry: LyricEntry,
    start_time: float,
    end_time: float,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    """Run an independent MMS/CTC alignment of only the lyric opening phrase."""
    if end_time - start_time < 0.55:
        return None
    prefix_text = lyric_prefix_probe_text(entry)
    if not prefix_text:
        return None
    python_exe = default_ctc_python()
    helper = Path(__file__).resolve().parent / "ctc_align.py"
    if not python_exe.exists() or not helper.exists():
        return None
    with tempfile.TemporaryDirectory(prefix="lrc-ctc-prefix-") as temp_name:
        temp_dir = Path(temp_name)
        transcript_path = temp_dir / "prefix.json"
        output_path = temp_dir / "prefix-result.json"
        transcript_path.write_text(
            json.dumps([prefix_text], ensure_ascii=False),
            encoding="utf-8",
        )
        command = [
            str(python_exe),
            str(helper),
            "--audio",
            str(audio_path),
            "--transcript",
            str(transcript_path),
            "--output",
            str(output_path),
            "--device",
            str(getattr(args, "whisperx_device", "auto") or "auto"),
            "--start",
            f"{start_time:.3f}",
            "--end",
            f"{end_time:.3f}",
        ]
        proc = _run_model_helper_command(command, mode="ctc")
        if proc.returncode != 0 or not output_path.exists():
            return {
                "accepted": False,
                "prefix_text": prefix_text,
                "window": [round(start_time, 3), round(end_time, 3)],
                "error": (proc.stderr or proc.stdout).strip()[:240],
            }
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        rows = payload.get("entries")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            return None
        row = rows[0]
        try:
            candidate = float(row.get("start"))
            score = float(row.get("ctc_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        token_spans = row.get("token_spans")
        token_scores: list[float] = []
        token_times: list[float] = []
        if isinstance(token_spans, list):
            for span in token_spans[:10]:
                if not isinstance(span, dict):
                    continue
                try:
                    token_scores.append(float(span.get("score", 0.0) or 0.0))
                    token_times.append(float(span.get("start")))
                except (TypeError, ValueError):
                    continue
        prefix_mean = float(statistics.mean(token_scores)) if token_scores else 0.0
        weak_ratio = (
            sum(value < 0.05 for value in token_scores) / len(token_scores)
            if token_scores else 1.0
        )
        span_seconds = (
            token_times[-1] - token_times[0]
            if len(token_times) >= 2
            else 0.0
        )
        accepted = bool(
            start_time <= candidate <= end_time
            and score >= 0.045
            and len(token_scores) >= 4
            and prefix_mean >= 0.045
            and weak_ratio <= 0.70
            and span_seconds >= 0.18
        )
        return {
            "accepted": accepted,
            "prefix_text": prefix_text,
            "window": [round(start_time, 3), round(end_time, 3)],
            "candidate": round(candidate, 3),
            "ctc_score": round(score, 6),
            "prefix_mean_score": round(prefix_mean, 6),
            "prefix_weak_ratio": round(weak_ratio, 3),
            "prefix_span_seconds": round(span_seconds, 3),
            "token_spans": token_spans,
            "first_token_candidates": row.get("first_token_candidates"),
        }


def duplicate_block_offset_evidence(
    entries: list[LyricEntry],
    assignments: list[dict[str, object] | object],
    target_index: int,
) -> dict[str, object] | None:
    """Estimate a repeated-block offset from an adjacent duplicate lyric pair.

    This does not choose a timestamp.  It only provides a projection window for
    a later independent vocal/CTC onset probe.
    """
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        normalized = normalize_match_text(entry_sung_text(entry))
        if normalized:
            groups.setdefault(normalized, []).append(index)
    normalized = normalize_match_text(entry_sung_text(entries[target_index]))
    pair = groups.get(normalized, [])
    if len(pair) != 2 or target_index not in pair:
        return None
    first_target, second_target = pair
    # Find an adjacent duplicated lyric whose two occurrences are shifted by
    # the same section offset.  Prefer the immediately preceding line.
    neighbors = [target_index - 1, target_index + 1]
    for neighbor in neighbors:
        if not 0 <= neighbor < len(entries):
            continue
        neighbor_norm = normalize_match_text(entry_sung_text(entries[neighbor]))
        neighbor_pair = groups.get(neighbor_norm, [])
        if len(neighbor_pair) != 2:
            continue
        a, b = neighbor_pair
        if (neighbor == first_target - 1 and a == first_target - 1 and b == second_target - 1) or (
            neighbor == second_target - 1 and a == first_target - 1 and b == second_target - 1
        ) or (
            neighbor == first_target + 1 and a == first_target + 1 and b == second_target + 1
        ) or (
            neighbor == second_target + 1 and a == first_target + 1 and b == second_target + 1
        ):
            if not (0 <= a < len(assignments) and 0 <= b < len(assignments)):
                continue
            left = assignments[a]
            right = assignments[b]
            if not isinstance(left, dict) or not isinstance(right, dict):
                continue
            left_spans = left.get("ctc_token_spans")
            right_spans = right.get("ctc_token_spans")
            if not isinstance(left_spans, list) or not isinstance(right_spans, list):
                continue
            deltas: list[float] = []
            for ls, rs in zip(left_spans, right_spans):
                if not isinstance(ls, dict) or not isinstance(rs, dict):
                    continue
                try:
                    if str(ls.get("char")) != str(rs.get("char")):
                        continue
                    lscore = float(ls.get("score", 0.0) or 0.0)
                    rscore = float(rs.get("score", 0.0) or 0.0)
                    if lscore < 0.05 or rscore < 0.05:
                        continue
                    deltas.append(float(rs.get("start")) - float(ls.get("start")))
                except (TypeError, ValueError):
                    continue
            if len(deltas) < 3:
                continue
            offset = float(statistics.median(deltas))
            spread = float(statistics.median(abs(value - offset) for value in deltas))
            if spread > 0.10:
                continue
            projected = None
            if target_index == first_target:
                other = assignments[second_target]
                if isinstance(other, dict):
                    try:
                        projected = float(other.get("timestamp")) - offset
                    except (TypeError, ValueError):
                        projected = None
            else:
                other = assignments[first_target]
                if isinstance(other, dict):
                    try:
                        projected = float(other.get("timestamp")) + offset
                    except (TypeError, ValueError):
                        projected = None
            return {
                "offset_seconds": round(offset, 3),
                "mad_seconds": round(spread, 3),
                "supporting_entries": [a + 1, b + 1],
                "projected_time": round(projected, 3) if projected is not None else None,
            }
    return None


def apply_final_ctc_timing_guard(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
    args: argparse.Namespace | None = None,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    """Final fail-closed audit/repair for CTC onset boundaries.

    Earlier stages are allowed to alter token spans and candidate timestamps.
    This final pass therefore re-evaluates the *actual timestamp that will be
    written to LRC*.  When a zero-gap/weak-prefix boundary has a later reliable
    CTC token, an isolated-vocal acoustic onset may move the timestamp forward.
    If no such evidence exists, the line remains review-required instead of
    inheriting a high whole-line confidence.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps):
        return timestamps, report, []
    if not any(
        isinstance(item, dict) and isinstance(item.get("ctc_token_spans"), list)
        for item in assignments
    ):
        return timestamps, report, []

    refined = list(timestamps)
    for item in assignments:
        if isinstance(item, dict):
            item.pop("ctc_final_timing_guard_unresolved", None)
            if (
                item.get("ctc_crossline_initial_recovery")
                and not item.get("ctc_crossline_initial_recovery_trusted")
            ):
                # Do not let an earlier provisional cross-line move keep a
                # stale effective-token index.  The final audit must evaluate
                # the token spans that actually begin at the selected time.
                item.pop("ctc_effective_token_start_index", None)
    annotate_ctc_boundary_evidence(assignments)
    try:
        features = analyze_audio(audio_path, duration)
    except LrcError:
        features = None
    try:
        vocal_onset_features = analyze_vocal_onsets(audio_path, duration)
    except LrcError:
        vocal_onset_features = None

    prefix_probe_records: list[dict[str, object]] = []
    rejected_onset_candidates: dict[int, dict[str, object]] = {}
    local_window_report = report.get("ctc_local_window_realign")
    if isinstance(local_window_report, dict):
        rejected_rows = local_window_report.get("rejected_rows")
        if isinstance(rejected_rows, list):
            for row in rejected_rows:
                if not isinstance(row, dict) or row.get("reason") != "local-window-right-edge-pileup":
                    continue
                try:
                    entry_number = int(row.get("entry"))
                    candidate_time = float(row.get("candidate"))
                except (TypeError, ValueError):
                    continue
                rejected_onset_candidates[entry_number] = {
                    **row,
                    "candidate": candidate_time,
                }

    def later_prefix_support(
        assignment: dict[str, object], current_time: float
    ) -> tuple[float, float, int] | None:
        """Find a later token that survives an otherwise weak opening prefix.

        An isolated first-token posterior is not enough when 75% of the first
        eight tokens are weak.  Scan forward for a materially later supported
        token after the weak cluster; the acoustic onset is then searched just
        before that token instead of treating the isolated first character as
        proof of the LRC boundary.
        """
        spans = assignment.get("ctc_token_spans")
        if not isinstance(spans, list):
            return None
        parsed: list[tuple[float, float]] = []
        for span in spans[:16]:
            if not isinstance(span, dict):
                continue
            try:
                parsed.append((float(span.get("start")), float(span.get("score", 0.0) or 0.0)))
            except (TypeError, ValueError):
                continue
        if len(parsed) < 4:
            return None
        for index in range(2, len(parsed)):
            start, score = parsed[index]
            delay = start - current_time
            if not 0.35 <= delay <= 3.00 or score < 0.12:
                continue
            preceding = [value for _time, value in parsed[1:index]]
            if preceding:
                weak_ratio = sum(value < 0.05 for value in preceding) / len(preceding)
                if weak_ratio < 0.60:
                    continue
            return start, score, index
        return None

    changes: list[dict[str, object]] = []
    for index in range(1, len(assignments)):
        previous = assignments[index - 1]
        current = assignments[index]
        if not isinstance(previous, dict) or not isinstance(current, dict):
            continue
        evidence = current.get("ctc_boundary_evidence")
        if not isinstance(evidence, dict) or not evidence.get("available"):
            continue

        try:
            current_time = float(current.get("timestamp", refined[index]) or refined[index])
        except (TypeError, ValueError):
            continue
        previous_end = refined[index - 1]
        previous_spans = previous.get("ctc_token_spans")
        if isinstance(previous_spans, list):
            for span in previous_spans:
                if not isinstance(span, dict):
                    continue
                try:
                    previous_end = max(previous_end, float(span.get("end")))
                except (TypeError, ValueError):
                    continue
        next_time = refined[index + 1] if index + 1 < len(refined) else duration

        try:
            first_score = float(evidence.get("first_token_score", 0.0) or 0.0)
            weak_ratio = float(evidence.get("prefix_weak_ratio", 0.0) or 0.0)
            reliable_delay_raw = evidence.get("reliable_token_delay_seconds")
            reliable_delay = (
                float(reliable_delay_raw)
                if isinstance(reliable_delay_raw, (int, float))
                else None
            )
            reliable_time_raw = evidence.get("next_reliable_token_time")
            reliable_time = (
                float(reliable_time_raw)
                if isinstance(reliable_time_raw, (int, float))
                else None
            )
            reliable_score_raw = evidence.get("next_reliable_token_score")
            reliable_score = (
                float(reliable_score_raw)
                if isinstance(reliable_score_raw, (int, float))
                else 0.0
            )
        except (TypeError, ValueError):
            continue

        zero_gap_weak = bool(
            evidence.get("zero_boundary_gap")
            and evidence.get("unclear_boundary")
            and (
                first_score < 0.08
                or weak_ratio >= 0.50
                or (reliable_delay is not None and reliable_delay >= 0.45)
            )
        )
        near_gap_weak = bool(evidence.get("near_gap_weak_prefix"))
        fracture = ctc_path_fracture_evidence(current)
        current["ctc_path_fracture_evidence"] = fracture
        hard_boundary = bool(
            zero_gap_weak
            or near_gap_weak
            or fracture.get("fractured")
            or evidence.get("borrowed_tail_prefix")
            or evidence.get("ghost_weak_prefix")
            or evidence.get("prior_line_truncation_contamination")
            or evidence.get("timestamp_precedes_ctc_support")
            or evidence.get("realign_required")
        )
        if not hard_boundary:
            continue

        recovery_time: float | None = None
        acoustic_threshold: float | None = None
        recovery_reason = ""

        # v4: never salvage a timestamp from a full-line row whose tail already
        # proved that the path is broken.  Instead, independently re-run CTC on
        # only the lyric opening phrase.  This gives the onset its own semantic
        # evidence and prevents a later "again"/repeated word or a right-edge
        # pileup from donating a timestamp to token 0.
        duplicate_projection = duplicate_block_offset_evidence(
            entries, assignments, index
        )
        prefix_probe: dict[str, object] | None = None
        if args is not None:
            probe_floor = max(previous_end + 0.10, current_time + 0.05)
            probe_ceiling = next_time - 0.12
            projected_time = None
            if isinstance(duplicate_projection, dict):
                value = duplicate_projection.get("projected_time")
                if isinstance(value, (int, float)):
                    projected_time = float(value)
            if projected_time is not None:
                probe_start = max(probe_floor, projected_time - 0.85)
                probe_end = min(probe_ceiling, projected_time + 1.25)
            elif fracture.get("fractured"):
                # A fractured line needs a broad semantic search.  The prefix
                # probe, not the old full-line path, decides where token 0 is.
                probe_start = probe_floor
                probe_end = probe_ceiling
            else:
                probe_start = probe_floor
                probe_end = min(probe_ceiling, current_time + 4.0)

            if probe_end - probe_start >= 0.55:
                prefix_probe = run_ctc_prefix_probe(
                    audio_path,
                    entries[index],
                    probe_start,
                    probe_end,
                    args,
                )
                if isinstance(prefix_probe, dict):
                    prefix_probe["entry"] = index + 1
                    prefix_probe["duplicate_projection"] = duplicate_projection
                    prefix_probe_records.append(prefix_probe)
                    current["ctc_prefix_probe_evidence"] = prefix_probe

        if isinstance(prefix_probe, dict) and prefix_probe.get("accepted"):
            try:
                probe_candidate = float(prefix_probe.get("candidate"))
                probe_score = float(prefix_probe.get("ctc_score", 0.0) or 0.0)
                probe_mean = float(prefix_probe.get("prefix_mean_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                probe_candidate = -1.0
                probe_score = 0.0
                probe_mean = 0.0

            acoustic_peak: dict[str, float] | None = None
            for onset_features, minimum in (
                (vocal_onset_features, 0.28),
                (features, 0.38),
            ):
                if onset_features is None:
                    continue
                candidate_peak = nearest_local_peak(
                    onset_features,
                    probe_candidate,
                    radius=0.24,
                    minimum_strength=minimum,
                )
                if isinstance(candidate_peak, dict):
                    try:
                        if (
                            abs(float(candidate_peak.get("delta_seconds", 9.0)))
                            <= PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
                        ):
                            acoustic_peak = candidate_peak
                            break
                    except (TypeError, ValueError):
                        pass

            # Strong prefix CTC may stand on its own; otherwise require the
            # isolated-vocal onset to agree.  The timestamp remains the CTC
            # prefix start (semantic evidence), not whichever spectral peak is
            # strongest.
            semantic_acoustic_support = bool(
                acoustic_peak is not None
                or (probe_score >= 0.12 and probe_mean >= 0.09)
            )
            if (
                semantic_acoustic_support
                and probe_candidate >= previous_end + 0.10
                and probe_candidate < next_time - 0.10
                and probe_candidate >= current_time + 0.10
            ):
                recovery_time = probe_candidate
                reliable_time = probe_candidate
                reliable_score = probe_score
                reliable_delay = probe_candidate - current_time
                recovery_reason = "independent-lyric-prefix-ctc-probe"
                current["ctc_prefix_probe_recovery"] = True
                current["ctc_prefix_probe_evidence"] = prefix_probe
                current["ctc_prefix_probe_acoustic_peak"] = acoustic_peak

        # For a 40-150 ms weak boundary, the first CTC character can be an
        # isolated accidental hit.  Use a later supported prefix token as a
        # *forward bound*, then search the vocal track for the new line onset
        # shortly before that token.  This is the 03:45 ``again`` failure that
        # v2 missed because token 0 happened to score 0.20.
        forward_support = later_prefix_support(current, current_time) if near_gap_weak or zero_gap_weak else None
        if recovery_time is None and prefix_probe is None and features is not None and forward_support is not None:
            support_time, support_score, support_index = forward_support
            search_start = max(
                previous_end + 0.10,
                current_time + 0.10,
                support_time - 1.10,
            )
            search_end = min(next_time - 0.10, support_time + 0.12)
            if search_end - search_start >= 0.20:
                for threshold in (0.70, 0.60, 0.50, 0.42):
                    onset = first_local_onset(features, search_start, search_end, threshold=threshold)
                    if onset is None:
                        continue
                    if (
                        onset >= previous_end + 0.10
                        and onset >= current_time + 0.10
                        and -0.12 <= support_time - onset <= 1.10
                    ):
                        recovery_time = onset
                        acoustic_threshold = threshold
                        reliable_time = support_time
                        reliable_score = support_score
                        reliable_delay = support_time - current_time
                        current["ctc_final_timing_recovery_support_token_index"] = support_index
                        recovery_reason = "near-gap-weak-prefix-recovered-by-acoustic-onset-and-forward-ctc"
                        break

        # Do not "repair" a compact 40 ms leak by simply picking the next CTC
        # character.  Require a materially later supported token and an acoustic
        # onset after ownership of the previous line has ended.
        if recovery_time is None and prefix_probe is None and (
            features is not None
            and reliable_time is not None
            and reliable_delay is not None
            and reliable_delay >= 0.55
            and reliable_score >= 0.08
        ):
            search_start = max(previous_end + 0.05, current_time + 0.05)
            search_end = min(next_time - 0.10, reliable_time + 0.15)
            if search_end - search_start >= 0.25:
                for threshold in (0.70, 0.60, 0.50, 0.42):
                    onset = first_local_onset(features, search_start, search_end, threshold=threshold)
                    if onset is None:
                        continue
                    # A new lyric needs its own acoustic event.  Keep at least
                    # 120 ms clear of the prior token tail and require it to be
                    # reasonably close to the later CTC support.
                    if (
                        onset >= previous_end + 0.12
                        and onset >= current_time + 0.12
                        and onset <= reliable_time + 0.15
                        and reliable_time - onset <= 1.10
                    ):
                        recovery_time = onset
                        acoustic_threshold = threshold
                        recovery_reason = "zero-gap-weak-prefix-recovered-by-acoustic-onset-and-forward-ctc"
                        break

        if recovery_time is not None:
            old_time = current_time
            refined[index] = recovery_time
            current["timestamp"] = round(recovery_time, 3)
            current["chosen_time"] = round(recovery_time, 3)
            current["ctc_final_timing_recovery"] = True
            current["ctc_final_timing_recovery_trusted"] = True
            current.pop("ctc_final_timing_guard_unresolved", None)
            current["ctc_final_timing_recovery_from"] = round(old_time, 3)
            current["ctc_final_timing_recovery_reliable_token_time"] = round(reliable_time, 3)
            current["ctc_final_timing_recovery_reliable_token_score"] = round(reliable_score, 6)
            current["ctc_final_timing_recovery_acoustic_threshold"] = acoustic_threshold
            current["timing_trusted"] = True
            if recovery_reason == "independent-lyric-prefix-ctc-probe":
                current["timing_trusted_reason"] = "independent-lyric-prefix-ctc-probe"
                current["timing_trusted_sources"] = ["ctc-prefix-probe", "vocal-onset"]
            else:
                current["timing_trusted_reason"] = "final-acoustic-onset-with-forward-ctc-support"
                current["timing_trusted_sources"] = ["acoustic", "ctc-forward-support"]
            current["timing_trusted_candidate_times"] = {
                "selected": round(recovery_time, 3),
                "previous_selected": round(old_time, 3),
                "next_reliable_ctc": round(reliable_time, 3),
            }
            current["timing_repair_source"] = (
                str(current.get("timing_repair_source") or "torchaudio-mms-fa")
                + "+final-timing-guard"
            )
            changes.append(
                {
                    "entry": index + 1,
                    "from": round(old_time, 3),
                    "to": round(recovery_time, 3),
                    "previous_token_end": round(previous_end, 3),
                    "next_reliable_ctc": round(reliable_time, 3),
                    "next_reliable_ctc_score": round(reliable_score, 6),
                    "acoustic_threshold": acoustic_threshold,
                    "reason": recovery_reason or "final-timing-guard-multi-evidence-recovery",
                }
            )
        else:
            current["ctc_final_timing_guard_unresolved"] = True
            current.pop("timing_trusted", None)
            current.pop("timing_trusted_reason", None)
            current.pop("timing_trusted_sources", None)
            current.pop("timing_trusted_candidate_times", None)

    annotate_ctc_boundary_evidence(assignments)
    # Final validation is authoritative: scoring is no longer allowed to clear
    # a boundary defect merely because the whole lyric path had a high score.
    unresolved = flag_unresolved_ctc_boundary_issues(entries, report, persist=True)
    report["ctc_final_timing_guard_count"] = len(changes)
    report["ctc_final_timing_guard_recoveries"] = changes
    report["ctc_final_timing_guard_unresolved"] = unresolved
    report["ctc_prefix_probe_count"] = len(prefix_probe_records)
    report["ctc_prefix_probes"] = prefix_probe_records
    update_report_confidence_metrics(report)
    return refined, report, changes


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
