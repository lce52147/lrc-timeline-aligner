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
from ctc_boundary_evidence import (
    RAW_CTC_CONSENSUS_MIN_SCORE,
    ctc_prefix_boundary_evidence,
    has_ctc_compressed_prefix,
    ctc_right_edge_pileup_evidence,
    annotate_ctc_boundary_evidence,
    refresh_ctc_confidence_diagnostics,
    clear_stale_ctc_boundary_review_state,
    flag_unresolved_ctc_boundary_issues,
    flag_unresolved_ctc_disagreements,
    flag_unresolved_raw_ctc_disagreements,
    first_local_onset,
    lyric_prefix_probe_text,
    ctc_path_fracture_evidence,
    duplicate_block_offset_evidence,
)
from ctc_window_realign import (
    _BOUNDED_RETRY_MODEL_IDENTITY,
    COLLAPSED_REGION_WINDOW_MAX_SECONDS,
    CTC_WIDE_REGION_RADIUS,
    CTC_WIDE_REGION_PEER_RADIUS,
    apply_ctc_local_window_realign,
    CTC_LOCAL_WINDOW_SHADOW_RADII,
    SEGMENT_REALIGN_REVISION,
    SEGMENT_REALIGN_SOURCE,
    SEGMENT_REALIGN_STAGE,
    SEGMENT_REALIGN_MAX_RUNS,
    SEGMENT_REALIGN_MIN_WINDOW_SECONDS,
    SEGMENT_REALIGN_MAX_WINDOW_SECONDS,
    SEGMENT_REALIGN_MIN_ANCHOR_SHARE,
    SEGMENT_REALIGN_RAW_AGREE_SECONDS,
    SEGMENT_REALIGN_STRAY_TOKEN_GAP_SECONDS,
    SEGMENT_REALIGN_DENSE_TOKEN_GAP_SECONDS,
    SEGMENT_REALIGN_MIN_SHIFT_SECONDS,
    SEGMENT_REALIGN_ANCHOR_CTC_SCORE,
    segment_realign_mode,
    segment_realign_row_status,
    segment_realign_onset,
    _segment_realign_observations,
    apply_segment_realign,
    run_segment_realign_stage,
    run_ctc_local_window_shadows,
    apply_ctc_hybrid_window_realign,
)
from ctc_prefix_recovery import (
    apply_ctc_weak_prefix_recovery,
    apply_ctc_crossline_initial_recovery,
    ctc_forward_supported_prefix_candidate,
    apply_ctc_forward_supported_prefix_recovery,
    apply_ctc_boundary_validation_stage,
    should_accept_zero_gap_boundary_realign,
    should_accept_ctc_boundary_realign,
    apply_ctc_zero_gap_boundary_realign,
    apply_ctc_acoustic_backtrack,
)
from ctc_whisperx_refinement import (
    apply_ctc_local_fusion_to_whisperx,
    _legacy_refinement_row_has_actionable_flags,
    apply_ctc_micro_refinement_to_whisperx,
    apply_whisperx_acoustic_boundary_refinement,
    _invalidate_timing_trust_after_timestamp_mutation,
    apply_hybrid_ctc_opening_onset_backtrack,
)


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
