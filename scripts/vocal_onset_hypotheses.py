"""Extracted from acoustic_hypotheses.py (split2): vocal onset confirmation and vocal activity re-entry hypotheses with their span, peak and raw-order helpers."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import numpy as np
from report_trust import _record_backend_participation_failure
from timing_core import PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, _candidate_confidence, _finite_unit_confidence, analyze_candidate_regions, assignment_ctc_path_collapsed, ctc_collapsed_region_flags, make_timing_candidate, record_alignment_hypothesis
from lyrics_io import LyricEntry, entry_sung_text, has_ambiguous_lyric_prefix
from audio_features import AudioFeatures, HOP_SECONDS, analyze_audio, analyze_vocal_onsets
from runtime_base import LrcError, _path_sha256
from runtime_base import TimingTokenSpan


VOCAL_ONSET_CONFIRM_RADIUS_SECONDS = 0.18


VOCAL_ONSET_CONFIRM_MIN_STRENGTH = 0.82


VOCAL_ONSET_CONFIRM_MIN_PROMINENCE = 0.35


VOCAL_ACTIVITY_REENTRY_FEATURE_REVISION = "isolated-vocal-activity-reentry-v1"


# Re-entry evidence deliberately reuses analyze_audio()/merge_segments()'s
# conservative 450 ms split rule. A shorter dip is not enough to defeat
# UNKNOWN/VIOLATION ownership; continuous singing therefore remains unverified.
VOCAL_ACTIVITY_REENTRY_MIN_GAP_SECONDS = 0.45


def _candidate_spans_from_assignment(assignment: dict[str, object]) -> tuple[TimingTokenSpan, ...]:
    raw_spans = assignment.get("ctc_token_spans")
    if not isinstance(raw_spans, list):
        return ()
    spans: list[TimingTokenSpan] = []
    for raw in raw_spans:
        if not isinstance(raw, dict):
            continue
        try:
            start = float(raw.get("start"))
            end = float(raw.get("end"))
            score = _finite_unit_confidence(raw.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end) and end >= start):
            continue
        spans.append(TimingTokenSpan(start, end, score, str(raw.get("char") or raw.get("token") or "")))
    return tuple(spans)


def _nearest_local_maximum(
    features: AudioFeatures,
    center: float,
    radius: float,
) -> tuple[float, float, float] | None:
    if features.onset_strength.size == 0 or features.frame_times.size == 0:
        return None
    mask = (features.frame_times >= center - radius) & (features.frame_times <= center + radius)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    best_index = int(indices[np.argmax(features.onset_strength[indices])])
    strength = float(features.onset_strength[best_index])
    context = np.flatnonzero(
        (features.frame_times >= center - 0.80) & (features.frame_times <= center + 0.80)
    )
    baseline = float(np.median(features.onset_strength[context])) if context.size else 0.0
    return float(features.frame_times[best_index]), strength, strength - baseline


def _strong_vocal_onset_near(
    features: AudioFeatures | None,
    center: float,
    *,
    require_absolute_vocal: bool = False,
) -> tuple[float, float, float] | None:
    """Return the existing production-grade vocal onset observation near a time."""
    if features is None or not math.isfinite(center):
        return None
    peak = _nearest_local_maximum(features, center, VOCAL_ONSET_CONFIRM_RADIUS_SECONDS)
    if peak is None:
        return None
    peak_time, strength, prominence = peak
    if require_absolute_vocal:
        # Vocal-only data: 0.4s power window at the detected onset, relative
        # to the high-energy active-vocal distribution (not full-mix music).
        # RMS values are absolute dBFS, unlike normalized onset strength.
        raw_levels = np.asarray(features.rms_db, dtype=np.float64)
        frame_times = np.asarray(features.frame_times, dtype=np.float64)
        usable = raw_levels[np.isfinite(raw_levels)]
        if usable.size < 8:
            return None
        active_floor = float(np.percentile(usable, 55))
        vocal_active = usable[usable >= active_floor]
        if vocal_active.size < 8:
            return None
        active_median = float(np.median(vocal_active))
        mask = (frame_times >= peak_time) & (frame_times < peak_time + 0.4)
        if not np.any(mask):
            return None
        local = np.asarray(raw_levels[mask], dtype=np.float64)
        local = local[np.isfinite(local)]
        if local.size < 3:
            return None
        local_power_db = 10.0 * math.log10(
            max(float(np.mean(np.power(10.0, local / 10.0))), 1e-20)
        )
        if local_power_db < active_median - 20.0:
            return None
    if (
        strength < VOCAL_ONSET_CONFIRM_MIN_STRENGTH
        or prominence < VOCAL_ONSET_CONFIRM_MIN_PROMINENCE
    ):
        return None
    return peak_time, strength, prominence


def _vocal_activity_reentry_for_time(
    features: AudioFeatures,
    current_time: float,
) -> dict[str, float | int] | None:
    """Return a conservative isolated-vocal activity re-entry near a boundary.

    ``analyze_audio`` merges activity separated by <=450 ms, so a surviving
    segment boundary is already evidence of a substantial non-vocal interval.
    The detector never moves a timestamp: it only certifies the existing CTC
    boundary when the isolated vocal stem independently re-enters there.
    """
    if not features.segments:
        return None
    matches: list[dict[str, float | int]] = []
    for segment_index, (segment_start, segment_end) in enumerate(features.segments):
        if abs(float(segment_start) - float(current_time)) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS:
            continue
        previous_end = (
            float(features.segments[segment_index - 1][1])
            if segment_index > 0
            else 0.0
        )
        gap = float(segment_start) - previous_end
        if gap + HOP_SECONDS * 0.51 < VOCAL_ACTIVITY_REENTRY_MIN_GAP_SECONDS:
            continue
        matches.append({
            "segment_index": segment_index,
            "segment_start": float(segment_start),
            "segment_end": float(segment_end),
            "previous_segment_end": previous_end,
            "gap_seconds": gap,
            "distance_seconds": abs(float(segment_start) - float(current_time)),
        })
    if not matches:
        return None
    return min(
        matches,
        key=lambda row: (
            float(row["distance_seconds"]),
            -float(row["gap_seconds"]),
            int(row["segment_index"]),
        ),
    )


VOCAL_ONSET_RAW_ORDER_TOLERANCE_SECONDS = 1.0


VOCAL_ONSET_RAW_ORDER_MIN_CONTRADICTIONS = 2


VOCAL_ONSET_RAW_ORDER_MIN_SCORE = 0.50


def _raw_asr_order_anchors(
    report: dict[str, object],
    entries: list[LyricEntry],
) -> dict[int, float]:
    """Rows whose recorded raw-ASR line onset can witness lyric ORDER.

    Only real (non-borrowed) raw segments recorded by
    ``flag_unresolved_raw_ctc_disagreements``; drift runs are never recorded
    there, and lines with an ambiguous lyric prefix may match a sibling line.
    """
    anchors: dict[int, float] = {}
    rows = report.get("suspicious_alignments")
    if not isinstance(rows, list):
        return anchors
    for row in rows:
        if not isinstance(row, dict) or row.get("raw_asr_borrowed") is not False:
            continue
        entry = row.get("entry")
        stamps = row.get("candidate_timestamps")
        if isinstance(entry, bool) or not isinstance(entry, int) or not isinstance(stamps, dict):
            continue
        try:
            raw_time = float(stamps.get("raw_asr"))
            raw_score = float(row.get("raw_asr_score"))
        except (TypeError, ValueError):
            continue
        index = entry - 1
        if (
            not math.isfinite(raw_time)
            or raw_score < VOCAL_ONSET_RAW_ORDER_MIN_SCORE
            or not 0 <= index < len(entries)
            or has_ambiguous_lyric_prefix(entries, index)
        ):
            continue
        anchors[index] = raw_time
    return anchors


def _raw_asr_order_contradictions(
    anchors: dict[int, float],
    index: int,
    time_seconds: float,
) -> list[int]:
    """Entries whose raw onset says ``index`` cannot start at ``time_seconds``."""
    tolerance = VOCAL_ONSET_RAW_ORDER_TOLERANCE_SECONDS
    return sorted(
        other + 1
        for other, raw_time in anchors.items()
        if (other > index and raw_time < time_seconds - tolerance)
        or (other < index and raw_time > time_seconds + tolerance)
    )


def apply_vocal_onset_confirmation_hypotheses(
    alignment_audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
    *,
    features: AudioFeatures | None = None,
) -> list[dict[str, object]]:
    """Confirm an existing coherent CTC onset with independent vocal acoustics.

    This producer never moves a timestamp.  It only publishes a challenger for
    the exact current timestamp when the current full-line CTC path is already
    internally coherent and a strong, locally-prominent vocal onset is bound to
    the written boundary.  It therefore resolves ownership uncertainty without
    converting a weak or fractured CTC path into trusted timing.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return []
    if features is None:
        try:
            features = analyze_vocal_onsets(alignment_audio_path, duration)
        except LrcError as exc:
            _record_backend_participation_failure(
                report, "apply_vocal_onset_confirmation_hypotheses", exc
            )
            return []
    audio_revision = hashlib.sha256(Path(alignment_audio_path).read_bytes()).hexdigest()
    # A coherent line next to a collapsed path, or a clock that sits after
    # (before) other rows' raw-ASR line onsets, is a path that has lost its
    # place; a vocal onset there confirms *a* line start, not this line's.
    collapsed_region = ctc_collapsed_region_flags(
        [assignment_ctc_path_collapsed(item) for item in assignments]
    )
    raw_order_anchors = _raw_asr_order_anchors(report, entries)
    generated: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            continue
        spans = _candidate_spans_from_assignment(assignment)
        if len(spans) < 4:
            continue
        try:
            current_time = float(assignment.get("timestamp", timestamps[index]) or timestamps[index])
        except (TypeError, ValueError):
            continue
        if collapsed_region[index]:
            skipped.append({"entry": index + 1, "reason": "collapsed-ctc-region"})
            continue
        contradicting = _raw_asr_order_contradictions(raw_order_anchors, index, current_time)
        if len(contradicting) >= VOCAL_ONSET_RAW_ORDER_MIN_CONTRADICTIONS:
            skipped.append({
                "entry": index + 1,
                "reason": "raw-asr-order-contradiction",
                "contradicting_entries": contradicting,
            })
            continue
        current_candidate = make_timing_candidate(
            entry_index=index,
            entry_text=entry_sung_text(entries[index]),
            source="ctc-current-geometry-probe",
            raw_time=current_time,
            spans=spans,
            confidence=_candidate_confidence(assignment),
            identity_support="supported",
            sequence_support="supported",
            current=False,
            right_edge_pileup=bool(assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(assignment.get("ctc_right_edge_truncation")),
            source_artifact={"geometry_probe": True},
        )
        regions = analyze_candidate_regions(current_candidate)
        coherent_opening = bool(
            regions.temporal_coherence == "coherent"
            and regions.onset_proof_end is not None
        )
        direct_onset_resolvable_opening = bool(
            regions.temporal_coherence == "ambiguous"
            and not regions.severe_gap_indexes
            and spans[0].score >= 0.08
            and regions.reliable_support_count >= 8
            and regions.prefix_duration is not None
            and regions.prefix_duration >= 3.0
        )
        if not (coherent_opening or direct_onset_resolvable_opening):
            continue
        peak = _nearest_local_maximum(features, current_time, VOCAL_ONSET_CONFIRM_RADIUS_SECONDS)
        if peak is None:
            continue
        peak_time, strength, prominence = peak
        if strength < VOCAL_ONSET_CONFIRM_MIN_STRENGTH or prominence < VOCAL_ONSET_CONFIRM_MIN_PROMINENCE:
            continue
        evidence_window = {
            "start": round(current_time - VOCAL_ONSET_CONFIRM_RADIUS_SECONDS, 3),
            "end": round(current_time + VOCAL_ONSET_CONFIRM_RADIUS_SECONDS, 3),
            "audio_sha256": audio_revision,
            "audio_duration_seconds": round(float(duration), 6),
            "feature_revision": "vocal-consonant-onset-v1",
            "peak_strength": round(strength, 8),
            "peak_prominence": round(prominence, 8),
        }
        hypothesis = record_alignment_hypothesis(
            assignment,
            source="ctc-vocal-onset-confirmed",
            stage="independent-vocal-onset-confirmation",
            raw_time=current_time,
            confidence=max(_candidate_confidence(assignment), strength),
            spans=assignment.get("ctc_token_spans"),
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=strength,
            acoustic_onset_time=peak_time,
            acoustic_evidence_producer="consonant-onset-detector",
            acoustic_evidence_kind="vocal-onset",
            direct_onset_time=peak_time,
            direct_onset_evidence_producer="consonant-onset-detector",
            direct_onset_evidence_kind="vocal-onset",
            parent_source=str(report.get("backend") or "unknown") + "-current",
            parent_time=current_time,
            window=evidence_window,
            mode="coherent-ctc-plus-bound-vocal-onset",
            right_edge_pileup=bool(assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(assignment.get("ctc_right_edge_truncation")),
        )
        generated.append({
            "entry": index + 1,
            "timestamp": round(current_time, 3),
            "peak_time": round(peak_time, 3),
            "strength": round(strength, 6),
            "prominence": round(prominence, 6),
            "hypothesis_id": hypothesis.get("hypothesis_id"),
        })
    report["vocal_onset_confirmation_hypotheses"] = generated
    report["vocal_onset_confirmation_count"] = len(generated)
    report["vocal_onset_confirmation_skipped"] = skipped
    report["vocal_onset_confirmation_authoritative"] = False
    report["vocal_onset_confirmation_mutates_baseline"] = False
    return generated


def apply_vocal_activity_reentry_hypotheses(
    alignment_audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> list[dict[str, object]]:
    """Publish boundary-grade evidence only for a real vocal re-entry.

    This is intentionally stricter than ``ctc-vocal-onset-confirmed``. A
    consonant/energy peak can occur inside the predecessor's held syllable, so
    generic vocal onset remains non-boundary-grade. Here the isolated vocal
    activity itself must have stopped long enough to form a separate activity
    segment and then restart at the current CTC boundary. That negative
    evidence about the predecessor tail is independent of the current CTC
    candidate and may therefore resolve UNKNOWN ownership without weakening
    the central invariant.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return []
    try:
        activity = analyze_audio(alignment_audio_path, duration)
        audio_revision = _path_sha256(alignment_audio_path)
    except (LrcError, OSError) as exc:
        _record_backend_participation_failure(
            report, "apply_vocal_activity_reentry_hypotheses", exc
        )
        return []
    generated: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            continue
        spans = _candidate_spans_from_assignment(assignment)
        if len(spans) < 4:
            continue
        try:
            current_time = float(assignment.get("timestamp", timestamps[index]) or timestamps[index])
        except (TypeError, ValueError):
            continue
        probe = make_timing_candidate(
            entry_index=index,
            entry_text=entry_sung_text(entries[index]),
            source="ctc-current-reentry-geometry-probe",
            raw_time=current_time,
            spans=spans,
            confidence=_candidate_confidence(assignment),
            identity_support="supported",
            sequence_support="supported",
            current=False,
            right_edge_pileup=bool(assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(assignment.get("ctc_right_edge_truncation")),
            source_artifact={"reentry_geometry_probe": True},
        )
        regions = analyze_candidate_regions(probe)
        geometry_ok = bool(
            regions.temporal_coherence == "coherent"
            and not regions.severe_gap_indexes
            and regions.reliable_support_count >= 3
        )
        if not geometry_ok:
            continue
        reentry = _vocal_activity_reentry_for_time(activity, current_time)
        if reentry is None:
            continue
        segment_start = float(reentry["segment_start"])
        evidence_window = {
            "audio_sha256": audio_revision,
            "audio_duration_seconds": round(float(duration), 6),
            "feature_revision": VOCAL_ACTIVITY_REENTRY_FEATURE_REVISION,
            "segment_index": int(reentry["segment_index"]),
            "segment_start": round(segment_start, 6),
            "segment_end": round(float(reentry["segment_end"]), 6),
            "previous_segment_end": round(float(reentry["previous_segment_end"]), 6),
            "gap_seconds": round(float(reentry["gap_seconds"]), 6),
            "current_time": round(current_time, 6),
        }
        hypothesis = record_alignment_hypothesis(
            assignment,
            source="ctc-vocal-activity-reentry",
            stage="independent-vocal-activity-reentry",
            raw_time=current_time,
            confidence=_candidate_confidence(assignment),
            spans=assignment.get("ctc_token_spans"),
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="unavailable",
            direct_onset_support="supported",
            direct_onset_time=segment_start,
            direct_onset_evidence_producer="vocal-activity-detector",
            direct_onset_evidence_kind="vocal-activity-reentry",
            parent_source=str(report.get("backend") or "unknown") + "-current",
            parent_time=current_time,
            window=evidence_window,
            mode="coherent-ctc-plus-isolated-vocal-reentry",
            right_edge_pileup=bool(assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(assignment.get("ctc_right_edge_truncation")),
        )
        generated.append({
            "entry": index + 1,
            "timestamp": round(current_time, 3),
            "segment_start": round(segment_start, 3),
            "gap_seconds": round(float(reentry["gap_seconds"]), 3),
            "hypothesis_id": hypothesis.get("hypothesis_id"),
        })
    report["vocal_activity_reentry_hypotheses"] = generated
    report["vocal_activity_reentry_count"] = len(generated)
    report["vocal_activity_reentry_authoritative"] = False
    report["vocal_activity_reentry_mutates_baseline"] = False
    return generated
