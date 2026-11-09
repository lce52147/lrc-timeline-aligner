"""Extracted from ctc_refinement.py (split2): weak/crossline/forward prefix recovery, zero-gap realign, acoustic backtrack and boundary validation stage."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from report_trust import _record_backend_participation_failure
from timing_core import EvidenceState, record_alignment_hypothesis
from lyrics_io import LyricEntry, entry_sung_text, normalize_match_text
from audio_features import AudioFeatures, HOP_SECONDS, analyze_audio, analyze_vocal_onsets
from runtime_base import LrcError, _run_model_helper_command, default_ctc_python
from ctc_boundary_evidence import (
    ctc_prefix_boundary_evidence,
    annotate_ctc_boundary_evidence,
    refresh_ctc_confidence_diagnostics,
    flag_unresolved_ctc_boundary_issues,
    first_local_onset,
)


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
