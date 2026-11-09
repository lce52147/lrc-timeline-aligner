"""Extracted from ctc_refinement.py (split2): CTC fusion and micro refinement, acoustic boundary refinement and opening-onset backtrack for whisperx rows."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from report_trust import ACTIONABLE_SUSPICIOUS_FLAGS, ACTIONABLE_TIMING_FAILURES, dedupe_suspicious_alignments, update_report_confidence_metrics
from timing_core import TRUSTED_ALIGNMENT_SCORE
from lyrics_io import normalize_match_text
from audio_features import analyze_audio
from runtime_base import LrcError
from ctc_boundary_evidence import (
    first_local_onset,
)


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
