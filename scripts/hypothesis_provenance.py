"""Extracted from acoustic_hypotheses.py (split2): provenance validators for acoustic hypotheses and global acoustic recurrence hypotheses."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from ctc_refinement import ctc_path_fracture_evidence
from report_trust import _record_backend_participation_failure, assignment_content_is_trusted
from timing_core import EvaluatedBackendTiming, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, TimingCandidate, _candidate_confidence, _finite_unit_confidence, _serialized_delta_matches, _serialized_seconds_equal, analyze_candidate_regions, make_timing_candidate, record_alignment_hypothesis
from lyrics_io import LyricEntry, entry_sung_text, normalize_match_text
from audio_features import AudioFeatures, HOP_SECONDS, _search_recurrence_start, acoustic_recurrence_features, analyze_audio, analyze_vocal_onsets
from runtime_base import LrcError, _path_sha256, _timing_content_digest, canonicalize_lrc_time
from runtime_base import TimingTokenSpan, _timing_spans_from_value
from recurrence_hypotheses import (
    MUTUAL_RECURRENCE_FEATURE_REVISION,
    MUTUAL_RECURRENCE_MIN_SIMILARITY,
    MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN,
    PHONETIC_ONSET_FEATURE_REVISION,
    PHONETIC_ONSET_MIN_POSTERIOR,
    PHONETIC_ONSET_RADIUS_SECONDS,
    PHONETIC_ONSET_MIN_STRENGTH,
    PHONETIC_ONSET_MIN_PROMINENCE,
    TAIL_RELEASE_FEATURE_REVISION,
    TAIL_RELEASE_ONSET_MIN_STRENGTH,
    TAIL_RELEASE_ONSET_MIN_PROMINENCE,
    TAIL_RELEASE_CLUSTER_BREAK_SECONDS,
    TAIL_RELEASE_CONTEXT_SECONDS,
    _project_recurrence_spans_from_geometry,
    _self_bound_hypothesis_record,
    _bound_local_occurrence_identity,
)
from vocal_onset_hypotheses import (
    VOCAL_ONSET_CONFIRM_RADIUS_SECONDS,
    VOCAL_ONSET_CONFIRM_MIN_STRENGTH,
    VOCAL_ONSET_CONFIRM_MIN_PROMINENCE,
    VOCAL_ACTIVITY_REENTRY_FEATURE_REVISION,
    VOCAL_ACTIVITY_REENTRY_MIN_GAP_SECONDS,
    _candidate_spans_from_assignment,
    _nearest_local_maximum,
    _vocal_activity_reentry_for_time,
)


RECURRENCE_FEATURE_REVISION = "logmel-dct-delta-v1"


RECURRENCE_MIN_SIMILARITY = 0.16


RECURRENCE_MIN_DISTINCT_MARGIN = 0.08


def vocal_onset_confirmation_provenance_is_valid(
    hypothesis: dict[str, object],
    assignment: dict[str, object],
    evidence_audio_path: Path,
    *,
    recomputed_features: AudioFeatures | None = None,
    evidence_audio_sha256: str | None = None,
) -> bool:
    """Recompute whether a CTC + consonant-onset hypothesis is independently bound.

    The serialized source label is never sufficient.  The row must bind to the
    actual assignment CTC spans and to the audio artifact used by the onset
    producer.  This prevents a candidate timestamp from manufacturing its own
    direct-onset certificate.
    """
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if (
        hypothesis.get("source") != "ctc-vocal-onset-confirmed"
        or hypothesis.get("stage") != "independent-vocal-onset-confirmation"
        or hypothesis.get("mode") != "coherent-ctc-plus-bound-vocal-onset"
        or hypothesis.get("authoritative") is not False
        or hypothesis.get("identity_support") != "supported"
        or hypothesis.get("sequence_support") != "supported"
        or hypothesis.get("acoustic_support") != "supported"
        or hypothesis.get("direct_onset_support") != "supported"
    ):
        return False
    window = hypothesis.get("window")
    parent = hypothesis.get("parent")
    if not isinstance(window, dict) or not isinstance(parent, dict):
        return False
    if window.get("feature_revision") != "vocal-consonant-onset-v1":
        return False
    spans = hypothesis.get("spans")
    assignment_spans = assignment.get("ctc_token_spans")
    if not isinstance(spans, list) or not spans or spans != assignment_spans:
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        parent_time = float(parent.get("raw_candidate_time"))
        onset_time = float(hypothesis.get("direct_onset_time"))
        acoustic_time = float(hypothesis.get("acoustic_onset_time"))
        strength = float(hypothesis.get("acoustic_strength"))
        window_strength = float(window.get("peak_strength"))
        prominence = float(window.get("peak_prominence"))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (raw_time, parent_time, onset_time, acoustic_time, strength, window_strength, prominence)):
        return False
    if (
        not _serialized_seconds_equal(raw_time, parent_time)
        or not _serialized_seconds_equal(onset_time, acoustic_time)
        or abs(strength - window_strength) > 1e-6
        or abs(onset_time - canonicalize_lrc_time(raw_time).seconds) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
        or strength < VOCAL_ONSET_CONFIRM_MIN_STRENGTH
        or prominence < VOCAL_ONSET_CONFIRM_MIN_PROMINENCE
    ):
        return False
    try:
        audio_sha = evidence_audio_sha256 or _path_sha256(evidence_audio_path)
        evidence_duration = float(window.get("audio_duration_seconds"))
    except (OSError, LrcError, TypeError, ValueError):
        return False
    if not math.isfinite(evidence_duration) or evidence_duration <= 0.0:
        return False
    if window.get("audio_sha256") != audio_sha:
        return False
    try:
        recomputed = recomputed_features or analyze_vocal_onsets(
            evidence_audio_path, evidence_duration
        )
        peak = _nearest_local_maximum(recomputed, raw_time, VOCAL_ONSET_CONFIRM_RADIUS_SECONDS)
    except (LrcError, OSError):
        return False
    if peak is None:
        return False
    recomputed_time, recomputed_strength, recomputed_prominence = peak
    return bool(
        abs(recomputed_time - onset_time) <= HOP_SECONDS * 0.51
        and abs(recomputed_strength - strength) <= 1e-5
        and abs(recomputed_prominence - prominence) <= 1e-5
        and recomputed_strength >= VOCAL_ONSET_CONFIRM_MIN_STRENGTH
        and recomputed_prominence >= VOCAL_ONSET_CONFIRM_MIN_PROMINENCE
    )


def ctc_opening_vocal_fusion_is_valid(
    hypothesis: dict[str, object],
    assignment: dict[str, object],
    evidence_audio_path: Path,
    *,
    vocal_provenance_validated: bool = False,
    recomputed_features: AudioFeatures | None = None,
    evidence_audio_sha256: str | None = None,
) -> bool:
    """Bind lyric-specific CTC opening geometry to independent vocal acoustics.

    A generic vocal peak is not boundary-grade by itself.  It becomes a
    phonetic boundary only when the immutable baseline CTC row independently
    supplies a coherent first-mora geometry at the same canonical start.  The
    fusion never accepts local-retry geometry, fractured openings, truncation,
    or right-edge pileup.
    """
    if not vocal_provenance_validated and not vocal_onset_confirmation_provenance_is_valid(
        hypothesis, assignment, evidence_audio_path,
        recomputed_features=recomputed_features,
        evidence_audio_sha256=evidence_audio_sha256,
    ):
        return False
    try:
        assignment_time = float(assignment.get("timestamp"))
        hypothesis_time = float(hypothesis.get("raw_candidate_time"))
    except (TypeError, ValueError):
        return False
    if (
        not math.isfinite(assignment_time)
        or not math.isfinite(hypothesis_time)
        or canonicalize_lrc_time(assignment_time).centiseconds
        != canonicalize_lrc_time(hypothesis_time).centiseconds
        or bool(assignment.get("ctc_right_edge_pileup"))
        or bool(assignment.get("ctc_right_edge_truncation"))
    ):
        return False
    spans = _candidate_spans_from_assignment(assignment)
    if not spans:
        return False
    probe = make_timing_candidate(
        entry_index=0,
        entry_text="ctc-opening-geometry",
        source="ctc-current-geometry-probe",
        raw_time=assignment_time,
        spans=spans,
        confidence=_candidate_confidence(assignment),
        identity_support="supported",
        sequence_support="supported",
        current=False,
        right_edge_pileup=False,
        window_truncated=False,
        source_artifact={"geometry_only": True},
    )
    regions = analyze_candidate_regions(probe)
    return bool(
        regions.temporal_coherence == "coherent"
        and regions.onset_proof_end is not None
        and regions.fracture_region == "none"
        and not regions.severe_gap_indexes
        and abs(spans[0].start - canonicalize_lrc_time(assignment_time).seconds)
        <= HOP_SECONDS
    )


def vocal_activity_reentry_provenance_is_valid(
    hypothesis: dict[str, object],
    assignment: dict[str, object],
    evidence_audio_path: Path,
) -> bool:
    """Recompute isolated-vocal re-entry evidence from the bound audio.

    Serialized segment metadata is descriptive only. The validator must obtain
    the same activity segmentation from the actual vocal artifact before the
    direct-onset claim can become boundary-grade.
    """
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if (
        hypothesis.get("source") != "ctc-vocal-activity-reentry"
        or hypothesis.get("stage") != "independent-vocal-activity-reentry"
        or hypothesis.get("mode") != "coherent-ctc-plus-isolated-vocal-reentry"
        or hypothesis.get("authoritative") is not False
        or hypothesis.get("identity_support") != "supported"
        or hypothesis.get("sequence_support") != "supported"
        or hypothesis.get("direct_onset_support") != "supported"
    ):
        return False
    window = hypothesis.get("window")
    parent = hypothesis.get("parent")
    if not isinstance(window, dict) or not isinstance(parent, dict):
        return False
    if window.get("feature_revision") != VOCAL_ACTIVITY_REENTRY_FEATURE_REVISION:
        return False
    spans = hypothesis.get("spans")
    assignment_spans = assignment.get("ctc_token_spans")
    if not isinstance(spans, list) or not spans or spans != assignment_spans:
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        parent_time = float(parent.get("raw_candidate_time"))
        onset_time = float(hypothesis.get("direct_onset_time"))
        current_time = float(window.get("current_time"))
        segment_index = int(window.get("segment_index"))
        segment_start = float(window.get("segment_start"))
        segment_end = float(window.get("segment_end"))
        previous_segment_end = float(window.get("previous_segment_end"))
        gap_seconds = float(window.get("gap_seconds"))
        evidence_duration = float(window.get("audio_duration_seconds"))
    except (TypeError, ValueError):
        return False
    numeric = (
        raw_time, parent_time, onset_time, current_time, segment_start,
        segment_end, previous_segment_end, gap_seconds, evidence_duration,
    )
    if not all(math.isfinite(value) for value in numeric):
        return False
    if (
        not _serialized_seconds_equal(raw_time, parent_time)
        or not _serialized_seconds_equal(raw_time, current_time)
        or not _serialized_seconds_equal(onset_time, segment_start)
        or abs(segment_start - current_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
        or segment_end <= segment_start
        or gap_seconds + HOP_SECONDS * 0.51 < VOCAL_ACTIVITY_REENTRY_MIN_GAP_SECONDS
        or not _serialized_delta_matches(
            segment_start, previous_segment_end, gap_seconds, tolerance_microseconds=1
        )
        or evidence_duration <= 0.0
    ):
        return False
    try:
        if window.get("audio_sha256") != _path_sha256(evidence_audio_path):
            return False
        recomputed = analyze_audio(evidence_audio_path, evidence_duration)
        reentry = _vocal_activity_reentry_for_time(recomputed, raw_time)
    except (LrcError, OSError):
        return False
    if reentry is None:
        return False
    return bool(
        int(reentry["segment_index"]) == segment_index
        and abs(float(reentry["segment_start"]) - segment_start) <= HOP_SECONDS * 0.51
        and abs(float(reentry["segment_end"]) - segment_end) <= HOP_SECONDS * 0.51
        and abs(float(reentry["previous_segment_end"]) - previous_segment_end) <= HOP_SECONDS * 0.51
        and abs(float(reentry["gap_seconds"]) - gap_seconds) <= HOP_SECONDS * 0.51
    )


def phonetic_onset_fusion_provenance_is_valid(
    hypothesis: dict[str, object],
    assignment: dict[str, object],
    alignment_audio_path: Path,
) -> bool:
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if (
        hypothesis.get("source") != "ctc-phonetic-onset-fusion"
        or hypothesis.get("stage") != "independent-phonetic-onset-fusion"
        or hypothesis.get("mode") != "coherent-ctc-first-token-plus-independent-onset"
        or hypothesis.get("authoritative") is not False
    ):
        return False
    if (
        not assignment_content_is_trusted(assignment)
        or bool(assignment.get("ctc_right_edge_pileup"))
        or bool(assignment.get("ctc_right_edge_truncation"))
        or bool(ctc_path_fracture_evidence(assignment).get("fractured"))
    ):
        return False
    parent = hypothesis.get("parent")
    if (
        not isinstance(parent, dict)
        or parent.get("source") not in {"ctc-current", "ctc-vocal-onset-confirmed"}
    ):
        return False
    window = hypothesis.get("window")
    if not isinstance(window, dict) or window.get("feature_revision") != PHONETIC_ONSET_FEATURE_REVISION:
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        peak_time = float(window.get("ctc_peak_time"))
        peak_score = float(window.get("ctc_peak_score"))
        onset_time = float(window.get("onset_time"))
        onset_strength = float(window.get("onset_strength"))
        onset_prominence = float(window.get("onset_prominence"))
    except (TypeError, ValueError):
        return False
    try:
        current_assignment_raw = float(assignment.get("timestamp"))
        current_raw_time = float(window.get("current_raw_time"))
        current_written_centiseconds = int(window.get("current_written_centiseconds"))
        parent_time = float(parent.get("raw_candidate_time"))
        window_current_time = float(window.get("current_time"))
        top_acoustic_strength = float(hypothesis.get("acoustic_strength"))
        top_acoustic_time = float(hypothesis.get("acoustic_onset_time"))
        top_direct_time = float(hypothesis.get("direct_onset_time"))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (
        raw_time, peak_time, peak_score, onset_time, onset_strength, onset_prominence,
        current_assignment_raw, current_raw_time, parent_time, window_current_time,
        top_acoustic_strength, top_acoustic_time, top_direct_time,
    )):
        return False
    current_written = canonicalize_lrc_time(current_assignment_raw)
    if (
        not _serialized_seconds_equal(raw_time, peak_time)
        or peak_score < PHONETIC_ONSET_MIN_POSTERIOR
        or abs(peak_time - current_written.seconds) > 0.50
        or not _serialized_seconds_equal(current_raw_time, current_assignment_raw)
        or current_written_centiseconds != current_written.centiseconds
        or canonicalize_lrc_time(current_raw_time).centiseconds != current_written_centiseconds
        or not _serialized_seconds_equal(parent_time, current_raw_time)
        or not _serialized_seconds_equal(window_current_time, current_written.seconds)
        or not isinstance(window.get("current_candidate_id"), str)
        or not window.get("current_candidate_id")
    ):
        return False
    if (
        hypothesis.get("identity_support") != "supported"
        or hypothesis.get("sequence_support") != "supported"
        or hypothesis.get("acoustic_support") != "supported"
        or hypothesis.get("direct_onset_support") != "supported"
        or abs(top_acoustic_strength - onset_strength) > 1e-6
        or not _serialized_seconds_equal(top_acoustic_time, onset_time)
        or not _serialized_seconds_equal(top_direct_time, onset_time)
    ):
        return False
    if abs(onset_time - raw_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS:
        return False
    if onset_strength < PHONETIC_ONSET_MIN_STRENGTH or onset_prominence < PHONETIC_ONSET_MIN_PROMINENCE:
        return False
    peaks = assignment.get("ctc_first_token_candidates")
    if not isinstance(peaks, list) or not any(
        isinstance(peak, dict)
        and _timing_content_digest(peak) == window.get("ctc_peak_digest")
        and _serialized_seconds_equal(peak.get("time", -9999.0), peak_time)
        for peak in peaks
    ):
        return False
    spans = _candidate_spans_from_assignment(assignment)
    if not spans:
        return False
    current = make_timing_candidate(
        entry_index=0,
        entry_text="provenance-check",
        source="ctc-current",
        raw_time=current_assignment_raw,
        spans=spans,
        confidence=_candidate_confidence(assignment),
        identity_support="supported",
        sequence_support="supported",
        current=True,
    )
    if analyze_candidate_regions(current).temporal_coherence != "coherent":
        return False
    try:
        evidence_duration = float(window.get("audio_duration_seconds"))
        if not math.isfinite(evidence_duration) or evidence_duration <= 0.0:
            return False
        if window.get("audio_sha256") != _path_sha256(alignment_audio_path):
            return False
        recomputed = analyze_vocal_onsets(alignment_audio_path, evidence_duration)
        recomputed_peak = _nearest_local_maximum(
            recomputed, peak_time, PHONETIC_ONSET_RADIUS_SECONDS
        )
    except (OSError, LrcError, TypeError, ValueError):
        return False
    if recomputed_peak is None:
        return False
    recomputed_time, recomputed_strength, recomputed_prominence = recomputed_peak
    return bool(
        abs(recomputed_time - onset_time) <= HOP_SECONDS * 0.51
        and abs(recomputed_strength - onset_strength) <= 1e-5
        and abs(recomputed_prominence - onset_prominence) <= 1e-5
        and recomputed_strength >= PHONETIC_ONSET_MIN_STRENGTH
        and recomputed_prominence >= PHONETIC_ONSET_MIN_PROMINENCE
    )


def mutual_recurrence_hypothesis_provenance_is_valid(
    hypothesis: dict[str, object],
    entry: LyricEntry,
    alignment_audio_path: Path,
    assignments: list[object],
    entries: list[LyricEntry],
    entry_index: int,
) -> bool:
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if (
        hypothesis.get("source") != "mutual-acoustic-recurrence"
        or hypothesis.get("stage") != "occurrence-level-mutual-recurrence-search"
        or hypothesis.get("mode") != "reciprocal-exact-duplicate-pitchclass-recurrence"
        or hypothesis.get("authoritative") is not False
    ):
        return False
    window = hypothesis.get("window")
    parent = hypothesis.get("parent")
    if not isinstance(window, dict) or not isinstance(parent, dict):
        return False
    if parent.get("source") != "reciprocal-untrusted-duplicate-pair":
        return False
    if window.get("feature_revision") != MUTUAL_RECURRENCE_FEATURE_REVISION:
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        target_entry = int(window.get("target_entry"))
        peer_entry = int(window.get("peer_entry"))
        peer_candidate = float(window.get("peer_candidate"))
        score = float(window.get("similarity"))
        margin = float(window.get("distinct_margin"))
        onset_strength = float(window.get("onset_strength"))
        onset_prominence = float(window.get("onset_prominence"))
    except (TypeError, ValueError):
        return False
    if target_entry != entry_index + 1 or not (1 <= peer_entry <= len(assignments)) or peer_entry == target_entry:
        return False
    if score < MUTUAL_RECURRENCE_MIN_SIMILARITY or margin < MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN:
        return False
    if onset_strength < 0.45 or onset_prominence < 0.10:
        return False
    expected_lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entry)).encode("utf-8")
    ).hexdigest()
    if window.get("normalized_lyric_sha256") != expected_lyric_sha:
        return False
    peer_index = peer_entry - 1
    if len(entries) != len(assignments):
        return False
    peer_normalized = normalize_match_text(entry_sung_text(entries[peer_index]))
    if not peer_normalized or peer_normalized != normalize_match_text(entry_sung_text(entry)):
        return False
    if hashlib.sha256(peer_normalized.encode("utf-8")).hexdigest() != expected_lyric_sha:
        return False
    current_assignment = assignments[entry_index]
    peer_assignment = assignments[peer_index]
    if not isinstance(current_assignment, dict) or not isinstance(peer_assignment, dict):
        return False
    current_identity = _bound_local_occurrence_identity(current_assignment)
    peer_identity = _bound_local_occurrence_identity(peer_assignment)
    if current_identity is None or peer_identity is None:
        return False
    left_entry = int(window.get("left_entry", -1))
    right_entry = int(window.get("right_entry", -1))
    if {target_entry, peer_entry} != {left_entry, right_entry} or left_entry >= right_entry:
        return False
    expected_current_identity = (
        window.get("left_identity_evidence") if target_entry == left_entry
        else window.get("right_identity_evidence")
    )
    expected_peer_identity = (
        window.get("right_identity_evidence") if target_entry == left_entry
        else window.get("left_identity_evidence")
    )
    if current_identity != expected_current_identity or peer_identity != expected_peer_identity:
        return False
    try:
        expected_raw = float(window.get("left_candidate") if target_entry == left_entry else window.get("right_candidate"))
        expected_peer_candidate = float(window.get("right_candidate") if target_entry == left_entry else window.get("left_candidate"))
    except (TypeError, ValueError):
        return False
    if (
        not _serialized_seconds_equal(raw_time, expected_raw)
        or not _serialized_seconds_equal(peer_candidate, expected_peer_candidate)
    ):
        return False
    peer_hypotheses = peer_assignment.get("alignment_hypotheses")
    pair_id = window.get("pair_id")
    reciprocal = next(
        (
            row for row in peer_hypotheses
            if isinstance(row, dict)
            and row.get("source") == "mutual-acoustic-recurrence"
            and isinstance(row.get("window"), dict)
            and row["window"].get("pair_id") == pair_id
            and int(row["window"].get("peer_entry", -1)) == target_entry
        ),
        None,
    ) if isinstance(peer_hypotheses, list) else None
    if not isinstance(reciprocal, dict):
        return False
    reciprocal_window = reciprocal.get("window")
    reciprocal_parent = reciprocal.get("parent")
    if not isinstance(reciprocal_window, dict) or not isinstance(reciprocal_parent, dict):
        return False
    if (
        reciprocal.get("stage") != "occurrence-level-mutual-recurrence-search"
        or reciprocal.get("mode") != "reciprocal-exact-duplicate-pitchclass-recurrence"
        or reciprocal.get("authoritative") is not False
        or reciprocal_parent.get("source") != "reciprocal-untrusted-duplicate-pair"
        or int(reciprocal_window.get("target_entry", -1)) != peer_entry
        or int(reciprocal_window.get("peer_entry", -1)) != target_entry
    ):
        return False
    reciprocal_projection = {
        key: reciprocal_window.get(key)
        for key in (
            "audio_sha256", "feature_revision", "normalized_lyric_sha256",
            "left_entry", "right_entry", "left_seed", "right_seed",
            "left_identity_evidence", "right_identity_evidence",
            "left_bounds", "right_bounds", "left_candidate", "right_candidate",
            "template_duration", "similarity", "runner_similarity", "distinct_margin",
        )
    }
    if _timing_content_digest(reciprocal_projection) != pair_id:
        return False
    try:
        reciprocal_time = float(reciprocal.get("raw_candidate_time"))
        reciprocal_peer = float(reciprocal_window.get("peer_candidate"))
    except (TypeError, ValueError):
        return False
    if (
        not _serialized_seconds_equal(reciprocal_time, peer_candidate)
        or not _serialized_seconds_equal(reciprocal_peer, raw_time)
    ):
        return False
    # The reciprocal row must bind to an actual exact duplicate LyricEntry,
    # not merely repeat the same hash inside mutable report provenance.
    if reciprocal_window.get("normalized_lyric_sha256") != expected_lyric_sha:
        return False
    try:
        if window.get("audio_sha256") != _path_sha256(alignment_audio_path):
            return False
    except (OSError, LrcError):
        return False
    # Recompute the pair-id projection so arbitrary field edits fail closed.
    pair_projection = {
        key: window.get(key)
        for key in (
            "audio_sha256", "feature_revision", "normalized_lyric_sha256",
            "left_entry", "right_entry", "left_seed", "right_seed",
            "left_identity_evidence", "right_identity_evidence",
            "left_bounds", "right_bounds", "left_candidate", "right_candidate",
            "template_duration", "similarity", "runner_similarity", "distinct_margin",
        )
    }
    if _timing_content_digest(pair_projection) != pair_id:
        return False
    return True


def trusted_tail_release_provenance_is_valid(
    hypothesis: dict[str, object],
    alignment_audio_path: Path,
    assignments: list[object],
    entry_index: int,
) -> bool:
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if (
        hypothesis.get("source") != "trusted-tail-release-onset"
        or hypothesis.get("stage") != "trusted-tail-release-onset-search"
        or hypothesis.get("mode") != "strong-onset-inside-centrally-trusted-tail-release"
        or hypothesis.get("authoritative") is not False
    ):
        return False
    window = hypothesis.get("window")
    parent = hypothesis.get("parent")
    if not isinstance(window, dict) or not isinstance(parent, dict):
        return False
    if parent.get("source") != "central-trusted-predecessor-tail":
        return False
    if window.get("feature_revision") != TAIL_RELEASE_FEATURE_REVISION:
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        start = float(window.get("search_start"))
        end = float(window.get("search_end"))
        strength = float(window.get("onset_strength"))
        prominence = float(window.get("onset_prominence"))
        tail_end = float(window.get("previous_tail_end"))
        old_raw_time = float(window.get("old_candidate_raw_time"))
        old_centiseconds = int(window.get("old_candidate_centiseconds"))
        old_time = float(window.get("old_candidate_time"))
        previous_onset_time = float(window.get("previous_onset_time"))
        cluster_gap = float(window.get("cluster_gap"))
        context_start = float(window.get("context_start"))
    except (TypeError, ValueError):
        return False
    if not (start <= raw_time <= end and _serialized_seconds_equal(start, tail_end)):
        return False
    if old_time > tail_end:
        return False
    if previous_onset_time >= raw_time:
        return False
    if not _serialized_delta_matches(
        raw_time, previous_onset_time, cluster_gap, tolerance_microseconds=1
    ):
        return False
    if cluster_gap < TAIL_RELEASE_CLUSTER_BREAK_SECONDS:
        return False
    if context_start > tail_end or tail_end - context_start > TAIL_RELEASE_CONTEXT_SECONDS + 1e-6:
        return False
    if strength < TAIL_RELEASE_ONSET_MIN_STRENGTH or prominence < TAIL_RELEASE_ONSET_MIN_PROMINENCE:
        return False
    if not isinstance(window.get("previous_candidate_id"), str) or not isinstance(window.get("previous_tail_revision"), str):
        return False
    if entry_index <= 0 or entry_index >= len(assignments):
        return False
    current_assignment = assignments[entry_index]
    previous_assignment = assignments[entry_index - 1]
    if not isinstance(current_assignment, dict) or not isinstance(previous_assignment, dict):
        return False
    try:
        current_assignment_raw = float(current_assignment.get("timestamp"))
    except (TypeError, ValueError):
        return False
    current_written = canonicalize_lrc_time(current_assignment_raw)
    if (
        not _serialized_seconds_equal(current_assignment_raw, old_raw_time)
        or old_centiseconds != current_written.centiseconds
        or canonicalize_lrc_time(old_raw_time).centiseconds != old_centiseconds
        or not _serialized_seconds_equal(current_written.seconds, old_time)
    ):
        return False
    try:
        return window.get("audio_sha256") == _path_sha256(alignment_audio_path)
    except (OSError, LrcError):
        return False


def _trusted_recurrence_reference_geometry(
    assignment: dict[str, object],
    reference: TimingCandidate,
) -> tuple[float, tuple[TimingTokenSpan, ...], dict[str, object]] | None:
    """Return evidence-first full-line geometry for a trusted repeat reference.

    The preferred geometry is the selected central candidate itself.  A selected
    Raw/WhisperX fusion candidate may legitimately carry no token spans, however.
    In that case a self-bound local CTC identity row may provide the *template
    geometry* only when its independently observed start already agrees with the
    trusted selected boundary within the existing cross-backend consensus
    tolerance.  The CTC row is never shifted to the final timestamp, so this does
    not back-fill evidence from the answer.
    """
    if reference.spans:
        return (
            reference.written_time.seconds,
            reference.spans,
            {
                "mode": "selected-candidate-spans",
                "reference_candidate_id": reference.candidate_id,
                "reference_generation_revision": reference.generation_revision,
                "geometry_revision": _timing_content_digest([
                    {
                        "token": span.token,
                        "start": round(span.start, 6),
                        "end": round(span.end, 6),
                        "score": round(span.score, 8),
                    }
                    for span in reference.spans
                ]),
            },
        )
    hypotheses = assignment.get("alignment_hypotheses")
    if not isinstance(hypotheses, list):
        return None
    eligible: list[tuple[float, float, tuple[TimingTokenSpan, ...], dict[str, object]]] = []
    for hypothesis in hypotheses:
        if (
            not isinstance(hypothesis, dict)
            or hypothesis.get("source") not in {
                "ctc-local-window",
                "ctc-opening-local-retry-identity",
            }
            or hypothesis.get("authoritative") is not False
            or hypothesis.get("identity_support") != "supported"
            or hypothesis.get("sequence_support") != "supported"
            or bool(hypothesis.get("window_truncated"))
            or bool(hypothesis.get("right_edge_pileup"))
            or not _self_bound_hypothesis_record(hypothesis)
        ):
            continue
        spans = _timing_spans_from_value(hypothesis.get("spans"))
        if len(spans) < 4:
            continue
        try:
            raw_time = float(hypothesis.get("raw_candidate_time"))
            confidence = float(hypothesis.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(raw_time)
            or abs(raw_time - reference.written_time.seconds)
            > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
        ):
            continue
        provenance = {
            "mode": "self-bound-local-ctc-geometry",
            "reference_candidate_id": reference.candidate_id,
            "reference_generation_revision": reference.generation_revision,
            "geometry_hypothesis_id": hypothesis.get("hypothesis_id"),
            "geometry_source": hypothesis.get("source"),
            "geometry_start": round(raw_time, 6),
            "selected_start": round(reference.written_time.seconds, 6),
            "agreement_seconds": round(abs(raw_time - reference.written_time.seconds), 6),
            "max_agreement_seconds": RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
            "geometry_revision": _timing_content_digest(hypothesis.get("spans")),
        }
        eligible.append((
            _finite_unit_confidence(confidence),
            -abs(raw_time - reference.written_time.seconds),
            spans,
            provenance,
        ))
    if not eligible:
        return None
    confidence, _neg_delta, spans, provenance = max(
        eligible, key=lambda item: (item[0], item[1], str(item[3].get("geometry_hypothesis_id")))
    )
    return float(provenance["geometry_start"]), spans, provenance


def apply_global_acoustic_recurrence_hypotheses(
    alignment_audio_path: Path,
    entries: list[LyricEntry],
    report: dict[str, object],
    evaluated: EvaluatedBackendTiming,
    duration: float,
) -> list[dict[str, object]]:
    """Search the whole song for exact repeated-lyric occurrences.

    A reference occurrence is eligible only when the central evaluator has
    already trusted its timing.  The search is independent of the target MMS
    timestamp and uses the nearest centrally-trusted lyric-order anchors only
    as broad bounds.  No match is emitted unless the full-line acoustic match
    is both strong and separated from the best distinct alternative.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return []
    try:
        features = acoustic_recurrence_features(alignment_audio_path)
        onset_features = analyze_vocal_onsets(alignment_audio_path, duration)
    except LrcError as exc:
        _record_backend_participation_failure(
            report, "apply_global_acoustic_recurrence_hypotheses", exc
        )
        return []
    audio_revision = hashlib.sha256(Path(alignment_audio_path).read_bytes()).hexdigest()
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        normalized = normalize_match_text(entry_sung_text(entry))
        if normalized:
            groups.setdefault(normalized, []).append(index)
    trusted = [bool(audit.timing_trusted) for audit in evaluated.state.audits]
    generated: list[dict[str, object]] = []
    for normalized, indexes in groups.items():
        if len(indexes) < 2:
            continue
        for target_index in indexes:
            if trusted[target_index]:
                continue
            reference_geometry: dict[int, tuple[float, tuple[TimingTokenSpan, ...], dict[str, object]]] = {}
            for reference_index in indexes:
                if not trusted[reference_index] or not assignment_content_is_trusted(assignments[reference_index]):
                    continue
                reference_assignment = assignments[reference_index]
                if not isinstance(reference_assignment, dict):
                    continue
                geometry = _trusted_recurrence_reference_geometry(
                    reference_assignment, evaluated.state.decisions[reference_index].candidate
                )
                if geometry is not None:
                    reference_geometry[reference_index] = geometry
            reference_indexes = sorted(reference_geometry)
            if not reference_indexes:
                continue
            reference_index = min(reference_indexes, key=lambda index: abs(index - target_index))
            reference = evaluated.state.decisions[reference_index].candidate
            template_start, template_spans, geometry_provenance = reference_geometry[reference_index]
            template_end = template_spans[-1].end
            if target_index > reference_index:
                lower = next(
                    (evaluated.state.decisions[index].written_time.seconds + 0.20
                     for index in range(target_index - 1, -1, -1) if trusted[index]),
                    template_end + 0.50,
                )
                upper = next(
                    (evaluated.state.decisions[index].written_time.seconds - 0.20
                     for index in range(target_index + 1, len(entries)) if trusted[index]),
                    duration,
                )
            else:
                lower = next(
                    (evaluated.state.decisions[index].written_time.seconds + 0.20
                     for index in range(target_index - 1, -1, -1) if trusted[index]),
                    0.0,
                )
                upper = next(
                    (evaluated.state.decisions[index].written_time.seconds - 0.20
                     for index in range(target_index + 1, len(entries)) if trusted[index]),
                    template_start - 0.50,
                )
            result = _search_recurrence_start(features, template_start, template_end, lower, upper)
            if result is None:
                continue
            score = float(result["score"])
            margin = float(result["margin"])
            candidate_start = float(result["start"])
            if score < RECURRENCE_MIN_SIMILARITY or margin < RECURRENCE_MIN_DISTINCT_MARGIN:
                continue
            peak = _nearest_local_maximum(onset_features, candidate_start, 0.22)
            if peak is None:
                continue
            peak_time, peak_strength, peak_prominence = peak
            if peak_strength < 0.55 or peak_prominence < 0.18:
                continue
            projected_spans = _project_recurrence_spans_from_geometry(
                template_spans, template_start, candidate_start, score
            )
            if len(projected_spans) < 4:
                continue
            evidence_window = {
                "search_start": round(lower, 3),
                "search_end": round(upper, 3),
                "audio_sha256": audio_revision,
                "feature_revision": RECURRENCE_FEATURE_REVISION,
                "reference_entry": reference_index + 1,
                "reference_candidate_id": reference.candidate_id,
                "reference_selected_start": round(reference.written_time.seconds, 6),
                "reference_start": round(template_start, 3),
                "reference_end": round(template_end, 3),
                "reference_geometry": geometry_provenance,
                "similarity": round(score, 8),
                "runner_similarity": result["runner_score"],
                "distinct_margin": round(margin, 8),
                "normalized_lyric_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            }
            hypothesis = record_alignment_hypothesis(
                assignments[target_index],
                source="global-acoustic-recurrence",
                stage="occurrence-level-global-recurrence-search",
                raw_time=candidate_start,
                confidence=max(0.0, min(1.0, score)),
                spans=projected_spans,
                identity_support="supported",
                sequence_support="supported",
                acoustic_support="supported",
                direct_onset_support="supported",
                acoustic_strength=peak_strength,
                acoustic_onset_time=peak_time,
                acoustic_evidence_producer="global-recurrence-search",
                acoustic_evidence_kind="recurrence-onset",
                direct_onset_time=peak_time,
                direct_onset_evidence_producer="global-recurrence-search",
                direct_onset_evidence_kind="recurrence-onset",
                parent_source="central-trusted-repeat-reference",
                parent_time=template_start,
                window=evidence_window,
                mode="full-line-acoustic-recurrence-with-trusted-reference",
            )
            generated.append({
                "entry": target_index + 1,
                "reference_entry": reference_index + 1,
                "candidate": round(candidate_start, 3),
                "similarity": round(score, 6),
                "distinct_margin": round(margin, 6),
                "peak_time": round(peak_time, 3),
                "peak_strength": round(peak_strength, 6),
                "hypothesis_id": hypothesis.get("hypothesis_id"),
            })
    report["global_acoustic_recurrence_hypotheses"] = generated
    report["global_acoustic_recurrence_count"] = len(generated)
    report["global_acoustic_recurrence_authoritative"] = False
    report["global_acoustic_recurrence_mutates_baseline"] = False
    return generated


def recurrence_hypothesis_provenance_is_valid(
    hypothesis: dict[str, object],
    entry: LyricEntry,
    alignment_audio_path: Path,
) -> bool:
    """Re-bind recurrence authority to its immutable evidence, not its source label."""
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if hypothesis.get("source") != "global-acoustic-recurrence":
        return False
    if hypothesis.get("stage") != "occurrence-level-global-recurrence-search":
        return False
    if hypothesis.get("mode") != "full-line-acoustic-recurrence-with-trusted-reference":
        return False
    if hypothesis.get("authoritative") is not False:
        return False
    parent = hypothesis.get("parent")
    window = hypothesis.get("window")
    if not isinstance(parent, dict) or parent.get("source") != "central-trusted-repeat-reference":
        return False
    if not isinstance(window, dict):
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        search_start = float(window.get("search_start"))
        search_end = float(window.get("search_end"))
        similarity = float(window.get("similarity"))
        margin = float(window.get("distinct_margin"))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (raw_time, search_start, search_end, similarity, margin)):
        return False
    if not (search_start <= raw_time < search_end):
        return False
    if similarity < RECURRENCE_MIN_SIMILARITY or margin < RECURRENCE_MIN_DISTINCT_MARGIN:
        return False
    if window.get("feature_revision") != RECURRENCE_FEATURE_REVISION:
        return False
    reference_candidate_id = window.get("reference_candidate_id")
    if not isinstance(reference_candidate_id, str) or not reference_candidate_id:
        return False
    expected_lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entry)).encode("utf-8")
    ).hexdigest()
    if window.get("normalized_lyric_sha256") != expected_lyric_sha:
        return False
    try:
        expected_audio_sha = _path_sha256(alignment_audio_path)
    except (OSError, LrcError):
        return False
    return bool(window.get("audio_sha256") == expected_audio_sha)
