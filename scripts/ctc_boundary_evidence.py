"""Extracted from ctc_refinement.py (split2): CTC prefix/right-edge boundary evidence, confidence refresh, boundary review flags and CTC disagreement flags."""

from __future__ import annotations

import re
import statistics
import numpy as np
from report_trust import CTC_BOUNDARY_REVIEW_FLAGS, _RUNTIME_RAW_CTC_CONSENSUS_KEY, _ctc_disagreement_severity, _merge_suspicious_severity, _raw_disagreement_severity, update_report_confidence_metrics
from timing_core import RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, _entry_occurrence_binding
from lyrics_io import LyricEntry, entry_sung_text, normalize_match_text
from audio_features import AudioFeatures
from runtime_base import _timing_content_digest


RAW_CTC_CONSENSUS_MIN_SCORE = 0.85


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
