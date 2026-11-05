"""Extracted from auto_lrc.py (split step 5): report trust: assignment helpers, final line trust, confidence metrics, alignment report."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from timing_core import BackendContentIdentityObservation, CentralTrustSummary, EvaluatedBackendTiming, FinalTimingState, TRUSTED_ALIGNMENT_SCORE, TimingCandidate, _RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY, _RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY, _RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY, _RUNTIME_WHISPERX_TAIL_SCOPES_KEY, _runtime_occurrence_binding_is_valid, candidate_has_bound_acoustic_evidence, candidate_has_bound_direct_onset_evidence, candidate_has_independent_backend_identity
from lyrics_io import AsrSegment, LRC_ALGORITHM_REVISION, LyricEntry, _BACKEND_PROVENANCE_PROJECTION_KEY, _RUNTIME_LYRIC_ENTRIES_KEY, annotate_display_romaji, entry_sung_text, first_forced_char_time, format_lrc_time, load_anchor_hints, lyric_anchor_profile, match_anchor_entries, normalize_match_text, ratio_percent
from audio_features import HOP_SECONDS
from runtime_base import LrcError, _timing_content_digest, canonicalize_lrc_time, validated_loaded_source_sha256
from runtime_base import LrcError as _CallbackLrcError
import lrc_stage_timings


_LATE_BOUND_CALLBACKS = (
    "_final_post_central_material_conflict_entries",
    "_suspicious_entry_requires_review_for_decision",
)


def _unbound_callback(name):
    def _raise(*args, **kwargs):
        raise _CallbackLrcError(f"{name} is not bound yet (bind_late_callbacks has not run)")

    return _raise


# Back-edges into later modules: auto_lrc binds the real functions once they exist.
_final_post_central_material_conflict_entries = _unbound_callback("_final_post_central_material_conflict_entries")
_suspicious_entry_requires_review_for_decision = _unbound_callback("_suspicious_entry_requires_review_for_decision")


def bind_late_callbacks(**callbacks):
    for name, function in callbacks.items():
        if name not in _LATE_BOUND_CALLBACKS:
            raise AttributeError(name)
        globals()[name] = function


COLLAPSE_SEGMENT_SCORE = 0.30


_UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY = "unresolved_evidence_requirements"


_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY = (
    "_runtime_backend_provenance_projection_seed"
)


_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY = (
    "_runtime_postcentral_existing_evidence_bundles"
)


_RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY = (
    "_runtime_postcentral_hubert_current_onset_evidence"
)


_RUNTIME_POSTCENTRAL_HUBERT_BOUNDARY_OBSERVATION_KEY = (
    "_runtime_postcentral_hubert_boundary_observation_evidence"
)


_RUNTIME_POSTCENTRAL_HUBERT_DUPLICATE_RESOLUTION_KEY = (
    "_runtime_postcentral_hubert_duplicate_occurrence_resolution"
)


_RUNTIME_POSTCENTRAL_CTC_TAIL_PRODUCER_CONTEXT_KEY = (
    "_runtime_postcentral_ctc_tail_producer_context"
)


_RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY = (
    "_runtime_cross_backend_material_conflict_entries"
)


_RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICT_DETAILS_KEY = (
    "_runtime_cross_backend_material_conflict_details"
)


_RUNTIME_RAW_CTC_CONSENSUS_KEY = "_runtime_raw_ctc_onset_consensus"


_RUNTIME_RAW_CONTENT_IDENTITY_KEY = "_runtime_raw_content_identity"


_RUNTIME_WHISPERX_CTC_CONSENSUS_KEY = "_runtime_whisperx_ctc_onset_consensus"


_RUNTIME_WHISPERX_ROW_EVIDENCE_KEY = "_runtime_whisperx_row_evidence"


_RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY = "_runtime_whisperx_execution_provenance"


_RUNTIME_WHISPERX_TEMP_PATHS_KEY = "_runtime_whisperx_temp_paths"


_RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY = "_runtime_local_whisper_evidence"


_RUNTIME_RAW_TIMING_DRIFT_FALSIFICATIONS_KEY = (
    "_runtime_raw_timing_drift_falsifications"
)


_RUNTIME_BACKEND_PARTICIPATION_KEY = "_runtime_backend_participation"


def _new_backend_participation() -> dict[str, object]:
    return {
        "attempted": [],
        "available": [],
        "unavailable": [],
        "evidence_producers_failed": [],
        "demucs_device": None,
        "vocal_source": None,
        "onset_stem_used": False,
    }


def _backend_participation(report: dict[str, object]) -> dict[str, object]:
    state = report.get(_RUNTIME_BACKEND_PARTICIPATION_KEY)
    if not isinstance(state, dict):
        state = _new_backend_participation()
        report[_RUNTIME_BACKEND_PARTICIPATION_KEY] = state
    return state


def _record_backend_participation_failure(
    report: dict[str, object], producer: str, exc: BaseException
) -> None:
    state = _backend_participation(report)
    failures = state.setdefault("evidence_producers_failed", [])
    if not isinstance(failures, list):
        failures = []
        state["evidence_producers_failed"] = failures
    failures.append({"producer": producer, "error": f"{type(exc).__name__}: {exc}"})


def _record_backend_unavailable(
    state: dict[str, object], name: str, reason: str
) -> None:
    attempted = state.setdefault("attempted", [])
    if not isinstance(attempted, list):
        attempted = []
        state["attempted"] = attempted
    if name not in attempted:
        attempted.append(name)
    unavailable = state.setdefault("unavailable", [])
    if not isinstance(unavailable, list):
        unavailable = []
        state["unavailable"] = unavailable
    if not any(isinstance(item, dict) and item.get("name") == name for item in unavailable):
        unavailable.append({"name": name, "reason": reason})


def _record_backend_available(state: dict[str, object], name: str) -> None:
    for key in ("attempted", "available"):
        values = state.setdefault(key, [])
        if not isinstance(values, list):
            values = []
            state[key] = values
        if name not in values:
            values.append(name)


def _merge_backend_participation(
    target_report: dict[str, object], source_report: dict[str, object]
) -> None:
    source = source_report.get(_RUNTIME_BACKEND_PARTICIPATION_KEY)
    if not isinstance(source, dict):
        return
    target = _backend_participation(target_report)
    for key in ("attempted", "available"):
        values = target.setdefault(key, [])
        if not isinstance(values, list):
            values = []
            target[key] = values
        for value in source.get(key, []):
            if value not in values:
                values.append(value)
    unavailable = target.setdefault("unavailable", [])
    if not isinstance(unavailable, list):
        unavailable = []
        target["unavailable"] = unavailable
    for item in source.get("unavailable", []):
        if isinstance(item, dict) and not any(
            isinstance(existing, dict) and existing.get("name") == item.get("name")
            and existing.get("reason") == item.get("reason")
            for existing in unavailable
        ):
            unavailable.append(copy.deepcopy(item))
    failures = target.setdefault("evidence_producers_failed", [])
    if not isinstance(failures, list):
        failures = []
        target["evidence_producers_failed"] = failures
    for item in source.get("evidence_producers_failed", []):
        if isinstance(item, dict) and item not in failures:
            failures.append(copy.deepcopy(item))
    for key in ("demucs_device", "vocal_source"):
        value = source.get(key)
        if value is not None:
            target[key] = value
    if source.get("onset_stem_used"):
        target["onset_stem_used"] = True


def _backend_participation_projection(report: dict[str, object]) -> None:
    state = report.get(_RUNTIME_BACKEND_PARTICIPATION_KEY)
    if isinstance(state, dict):
        report["backend_participation"] = copy.deepcopy(state)


def report_quality(report: dict[str, object]) -> float:
    """Single-source ranking metric: end-to-end trust, plus global collapse.

    Low-score, unverified, and review counts are projections of the same row
    state and therefore are not charged again.  This keeps the metric useful for
    ranking instead of turning 60% trust into an arbitrary negative number.
    """
    if report.get("error"):
        return -100.0
    quality = float(
        report.get(
            "overall_trusted_percent",
            report.get("trusted_percent", report.get("matched_percent", 0.0)),
        )
        or 0.0
    )
    if report.get("collapse_detected"):
        quality -= 100.0
    return max(-100.0, min(100.0, quality))


def candidate_selection_quality(report: dict[str, object]) -> float:
    return report_quality(report)


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


CTC_BOUNDARY_REVIEW_FLAGS = {
    "ctc_unresolved_boundary",
    "ctc_zero_boundary_gap",
    "ctc_low_confidence_prefix",
    "ctc_abnormal_prefix_compression",
    "ctc_ghost_weak_prefix",
    "ctc_borrowed_tail_prefix",
    "ctc_near_gap_weak_prefix",
    "ctc_prior_line_truncation_contamination",
    "ctc_timestamp_precedes_ctc_support",
    "ctc_detached_weak_prefix",
    "ctc_unsupported_weak_prefix",
    "ctc_forward_prefix_probe_unresolved",
    "ctc_final_timing_guard_unresolved",
    "ctc_right_edge_pileup",
    "ctc_local_window_truncated",
    "ctc_zero_gap_compressed_prefix",
    "ctc_path_fracture",
    "ctc_prefix_probe_failed",
}


def runtime_local_whisper_evidence_for_entry(
    report: dict[str, object],
    entries: list[LyricEntry],
    entry_index: int,
) -> dict[str, object] | None:
    runtime = report.get(_RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY)
    if not isinstance(runtime, dict) or not 0 <= entry_index < len(entries):
        return None
    evidence = runtime.get(entry_index)
    if not isinstance(evidence, dict):
        return None
    try:
        raw_onset_time = evidence.get("onset_time")
        onset_time = float(raw_onset_time) if raw_onset_time is not None else None
        score = float(evidence.get("score"))
        window_start = float(evidence.get("window_start"))
        window_end = float(evidence.get("window_end"))
    except (TypeError, ValueError):
        return None
    lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entries[entry_index])).encode("utf-8")
    ).hexdigest()
    payload = {
        "entry_index": entry_index,
        "lyric_sha256": lyric_sha,
        "onset_time": (round(onset_time, 6) if onset_time is not None else None),
        "score": round(score, 6),
        "occurrence_binding": evidence.get("occurrence_binding"),
        "request_revision": evidence.get("request_revision"),
        "capability_revision": evidence.get("capability_revision"),
        "audio_revision": evidence.get("audio_revision"),
        "window_start": round(window_start, 6),
        "window_end": round(window_end, 6),
        "raw_transcript_revision": evidence.get("raw_transcript_revision"),
        "aligned_transcript_revision": evidence.get("aligned_transcript_revision"),
        "producer": "local-whisper-aligner",
    }
    if (
        evidence.get("composite_onset_evidence_revision") is not None
        or evidence.get("composite_context_evidence_revision") is not None
    ):
        payload["composite_onset_evidence_revision"] = evidence.get(
            "composite_onset_evidence_revision"
        )
        payload["composite_context_evidence_revision"] = evidence.get(
            "composite_context_evidence_revision"
        )
    if (
        evidence.get("producer") != "local-whisper-aligner"
        or evidence.get("lyric_sha256") != lyric_sha
        or not isinstance(evidence.get("raw_transcript_revision"), str)
        or not evidence.get("raw_transcript_revision")
        or not isinstance(evidence.get("aligned_transcript_revision"), str)
        or not evidence.get("aligned_transcript_revision")
        or score < TRUSTED_ALIGNMENT_SCORE
        or (
            onset_time is not None
            and (
                not math.isfinite(onset_time)
                or not window_start - HOP_SECONDS <= onset_time <= window_end + HOP_SECONDS
            )
        )
        or not _runtime_occurrence_binding_is_valid(
            entries, entry_index, evidence.get("occurrence_binding")
        )
        or evidence.get("evidence_revision") != _timing_content_digest(payload)
    ):
        return None
    return dict(evidence)


def runtime_raw_content_identity_for_entry(
    report: dict[str, object],
    entries: list[LyricEntry],
    entry_index: int,
) -> dict[str, object] | None:
    runtime = report.get(_RUNTIME_RAW_CONTENT_IDENTITY_KEY)
    if not isinstance(runtime, dict):
        return None
    evidence = runtime.get(entry_index)
    if not isinstance(evidence, dict):
        return None
    lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entries[entry_index])).encode("utf-8")
    ).hexdigest()
    try:
        raw_time = float(evidence.get("raw_time"))
        raw_score = float(evidence.get("raw_score"))
    except (TypeError, ValueError):
        return None
    payload = {
        "entry_index": entry_index,
        "lyric_sha256": lyric_sha,
        "raw_time": round(raw_time, 6),
        "raw_score": round(raw_score, 6),
        "segment": evidence.get("segment"),
        "occurrence_binding": evidence.get("occurrence_binding"),
        "producer": "raw-asr-content-identity",
    }
    if (
        evidence.get("producer") != "raw-asr-content-identity"
        or raw_score < TRUSTED_ALIGNMENT_SCORE
        or evidence.get("lyric_sha256") != lyric_sha
        or not _runtime_occurrence_binding_is_valid(
            entries, entry_index, evidence.get("occurrence_binding")
        )
        or evidence.get("evidence_revision") != _timing_content_digest(payload)
    ):
        return None
    return dict(evidence)


def runtime_whisperx_row_evidence_for_entry(
    report: dict[str, object],
    entries: list[LyricEntry],
    entry_index: int,
) -> dict[str, object] | None:
    runtime = report.get(_RUNTIME_WHISPERX_ROW_EVIDENCE_KEY)
    if not isinstance(runtime, dict):
        return None
    evidence = runtime.get(entry_index)
    if not isinstance(evidence, dict):
        return None
    lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entries[entry_index])).encode("utf-8")
    ).hexdigest()
    try:
        score = float(evidence.get("score"))
        segment_start = float(evidence.get("segment_start"))
        segment_end = float(evidence.get("segment_end"))
        first_time = float(evidence.get("first_char_time"))
        last_time = float(evidence.get("last_char_time"))
        char_count = int(evidence.get("char_count", 0) or 0)
    except (TypeError, ValueError):
        return None
    payload = {
        "entry_index": entry_index,
        "lyric_sha256": lyric_sha,
        "score": round(score, 6),
        "segment": evidence.get("segment"),
        "segment_start": round(segment_start, 6),
        "segment_end": round(segment_end, 6),
        "first_char_time": round(first_time, 6),
        "last_char_time": round(last_time, 6),
        "char_count": char_count,
        "suppress_nst": bool(evidence.get("suppress_nst")),
        "occurrence_binding": evidence.get("occurrence_binding"),
        "producer": "whisperx-row-evidence",
    }
    if (
        evidence.get("producer") != "whisperx-row-evidence"
        or evidence.get("lyric_sha256") != lyric_sha
        or score < TRUSTED_ALIGNMENT_SCORE
        or not _runtime_occurrence_binding_is_valid(
            entries, entry_index, evidence.get("occurrence_binding")
        )
        or char_count <= 0
        or segment_end < segment_start
        or not (segment_start - HOP_SECONDS <= first_time <= segment_end + HOP_SECONDS)
        or not (first_time - HOP_SECONDS <= last_time <= segment_end + HOP_SECONDS)
        or evidence.get("evidence_revision") != _timing_content_digest(payload)
    ):
        return None
    return dict(evidence)


def runtime_independent_content_identity_sources_for_entry(
    report: dict[str, object],
    entry_index: int,
) -> tuple[str, ...]:
    """Return validated runtime-only content identity sources for one lyric row.

    Content identity is intentionally orthogonal to timestamp evidence.  Raw ASR
    or WhisperX may prove *which lyric occurrence* this row is even when their
    onset estimate is too far from the selected timing candidate to participate
    in timing consensus.  Existing occurrence binding, lyric digest, score and
    evidence-revision validation remains mandatory via the canonical accessors.
    """
    entries = report.get(_RUNTIME_LYRIC_ENTRIES_KEY)
    if not isinstance(entries, tuple) or not entries:
        return ()
    if not 0 <= entry_index < len(entries) or not all(isinstance(item, LyricEntry) for item in entries):
        return ()
    lyric_entries = list(entries)
    sources: list[str] = []
    if runtime_raw_content_identity_for_entry(report, lyric_entries, entry_index) is not None:
        sources.append("raw-asr-content-identity")
    if runtime_whisperx_row_evidence_for_entry(report, lyric_entries, entry_index) is not None:
        sources.append("whisperx-row-evidence")
    if runtime_local_whisper_evidence_for_entry(report, lyric_entries, entry_index) is not None:
        sources.append("local-whisper-aligner")
    return tuple(sorted(sources))


def runtime_content_identity_projection(report: dict[str, object]) -> list[dict[str, object]]:
    """Canonical digestable projection of runtime content evidence used by central trust."""
    entries = report.get(_RUNTIME_LYRIC_ENTRIES_KEY)
    if not isinstance(entries, tuple):
        return []
    projected: list[dict[str, object]] = []
    raw_runtime = report.get(_RUNTIME_RAW_CONTENT_IDENTITY_KEY)
    whisper_runtime = report.get(_RUNTIME_WHISPERX_ROW_EVIDENCE_KEY)
    local_whisper_runtime = report.get(_RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY)
    for index in range(len(entries)):
        sources = runtime_independent_content_identity_sources_for_entry(report, index)
        if not sources:
            continue
        revisions: dict[str, str] = {}
        if "raw-asr-content-identity" in sources and isinstance(raw_runtime, dict):
            item = raw_runtime.get(index)
            if isinstance(item, dict):
                revisions["raw-asr-content-identity"] = str(item.get("evidence_revision") or "")
        if "whisperx-row-evidence" in sources and isinstance(whisper_runtime, dict):
            item = whisper_runtime.get(index)
            if isinstance(item, dict):
                revisions["whisperx-row-evidence"] = str(item.get("evidence_revision") or "")
        if "local-whisper-aligner" in sources and isinstance(local_whisper_runtime, dict):
            item = local_whisper_runtime.get(index)
            if isinstance(item, dict):
                revisions["local-whisper-aligner"] = str(item.get("evidence_revision") or "")
        projected.append({
            "entry": index + 1,
            "sources": list(sources),
            "evidence_revisions": dict(sorted(revisions.items())),
        })
    return projected


ACTIONABLE_TIMING_FAILURES = frozenset({
    "recurrence-uncorroborated",
    "right-edge-pileup",
    "window-truncated",
    "nonmonotonic-line-start",
    "previous-tail-overlap",
    "temporal-incoherence",
    "revision-mismatch",
    "canonical-time-evidence-mismatch",
    "candidate-span-revision-mismatch",
    "ownership-revision-mismatch",
    "tail-dependency-revision-mismatch",
    "occurrence-evidence-invalid",
    "unbound-or-nonindependent-direct-onset-claim",
} | CTC_BOUNDARY_REVIEW_FLAGS)


# Suspicious diagnostics are intentionally explicit.  Unknown/new diagnostic
# labels do not silently become a user review task; they remain report evidence
# until an owner classifies them as an actionable output defect.
ACTIONABLE_SUSPICIOUS_FLAGS = frozenset({
    "unresolved_raw_ctc_disagreement",
    "unresolved_ctc_disagreement",
    "candidate_disagreement",
    "unresolved-precentral-disagreement",
    "phonetic_anchor_disagreement",
    "duplicate_occurrence_offset_mismatch",
    "ctc_confidence_collapse",
    "group_placement_unsupported",
    "segment_structural_ambiguity",
    "segment_weak_ctc_support",
} | CTC_BOUNDARY_REVIEW_FLAGS)


NON_ACTIONABLE_VERIFICATION_FAILURES = frozenset({
    "no-strict-opening-proof",
    "previous-tail-ownership-unknown",
    "tail-conflict-without-independent-current-opening",
    "insufficient-minimum-proof",
})


def _timing_failure_requires_review(code: str) -> bool:
    """Review only positive, owner-classified defect evidence."""
    return code in ACTIONABLE_TIMING_FAILURES


def _suspicious_row_requires_review(row: object) -> bool:
    if not isinstance(row, dict):
        return False
    if str(row.get("severity") or "").lower() == "resolved" or row.get("resolution"):
        return False
    flags = row.get("flags")
    if not isinstance(flags, list):
        return False
    return any(str(flag) in ACTIONABLE_SUSPICIOUS_FLAGS for flag in flags)


def _suspicious_entry_requires_review(report: dict[str, object], entry_number: int) -> bool:
    suspicious = report.get("suspicious_alignments")
    if not isinstance(suspicious, list):
        return False
    return any(
        isinstance(row, dict)
        and row.get("entry") == entry_number
        and _suspicious_row_requires_review(row)
        for row in suspicious
    )


def _actionable_suspicious_severity_counts(report: dict[str, object]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    suspicious = report.get("suspicious_alignments")
    if not isinstance(suspicious, list):
        return counts
    for row in suspicious:
        if not _suspicious_row_requires_review(row):
            continue
        assert isinstance(row, dict)
        severity = str(row.get("severity") or "medium")
        if severity not in counts:
            severity = "medium"
        counts[severity] += 1
    return counts


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


def assignment_content_is_trusted(item: object) -> bool:
    """Whether the lyric/path match itself is trustworthy, independent of onset."""
    return assignment_has_timing(item) and assignment_score(item) >= TRUSTED_ALIGNMENT_SCORE


def assignment_timing_is_trusted(item: object) -> bool:
    """Whether the selected LRC timestamp has sufficient boundary evidence.

    This is intentionally independent from whole-line/path confidence.  A CTC
    path can match the lyric order well while its first token is still borrowed
    from the previous line, packed against a local-window edge, or unsupported
    for a second after the selected timestamp.
    """
    if not assignment_has_timing(item) or not isinstance(item, dict):
        return False
    if item.get("group_placement_unsupported") or item.get("segment_weak_ctc_untrusted"):
        return False
    if "timestamp_status" in item or "normalized_timing_evidence" in item:
        if item.get("timestamp_status") != "selected_valid":
            return False
        if item.get("timestamp_revision") != item.get("timing_evidence_revision"):
            return False
        evidence = item.get("normalized_timing_evidence")
        if not isinstance(evidence, dict):
            return False
        if evidence.get("computed_for_centiseconds") != item.get(
            "canonical_written_centiseconds"
        ):
            return False
        if evidence.get("structural_failures"):
            return False
        if not evidence.get("minimum_proof_satisfied"):
            return False
        return bool(item.get("timing_trusted"))
    if item.get("manual_anchor_hint"):
        return True

    flags = item.get("flags")
    flag_set = set(flags) if isinstance(flags, list) else set()
    hard_timing_flags = {
        "ctc_unresolved_boundary",
        "ctc_right_edge_pileup",
        "ctc_local_window_truncated",
        "ctc_zero_gap_compressed_prefix",
        "ctc_prior_line_truncation_contamination",
        "ctc_timestamp_precedes_ctc_support",
        "ctc_forward_prefix_probe_unresolved",
        "ctc_near_gap_weak_prefix",
    }
    if flag_set & hard_timing_flags:
        return False
    if item.get("ctc_right_edge_truncation") or item.get("ctc_unresolved_boundary"):
        return False

    edge = item.get("ctc_right_edge_evidence")
    if isinstance(edge, dict) and edge.get("right_edge_pileup"):
        return False

    evidence = item.get("ctc_boundary_evidence")
    recovery_trusted = bool(
        item.get("ctc_boundary_realign_trusted")
        or item.get("ctc_crossline_initial_recovery_trusted")
        or item.get("ctc_forward_supported_prefix_recovery_trusted")
        or item.get("ctc_final_timing_recovery_trusted")
    )
    if isinstance(evidence, dict) and evidence.get("available"):
        if evidence.get("prior_line_truncation_contamination") and not recovery_trusted:
            return False
        if evidence.get("timestamp_precedes_ctc_support") and not recovery_trusted:
            return False
        if evidence.get("realign_required") and not recovery_trusted:
            return False
        try:
            first_score = float(evidence.get("first_token_score", 0.0) or 0.0)
            weak_ratio = float(evidence.get("prefix_weak_ratio", 0.0) or 0.0)
            reliable_delay_raw = evidence.get("reliable_token_delay_seconds")
            reliable_delay = (
                float(reliable_delay_raw)
                if isinstance(reliable_delay_raw, (int, float))
                else 0.0
            )
        except (TypeError, ValueError):
            first_score = 0.0
            weak_ratio = 1.0
            reliable_delay = 999.0
        suspicious_zero_gap = bool(
            evidence.get("zero_boundary_gap")
            and evidence.get("unclear_boundary")
            and (
                first_score < 0.08
                or weak_ratio >= 0.50
                or reliable_delay >= 0.45
            )
        )
        if suspicious_zero_gap and not recovery_trusted:
            return False
        if evidence.get("near_gap_weak_prefix") and not recovery_trusted:
            return False
        if (
            evidence.get("borrowed_tail_prefix")
            or evidence.get("ghost_weak_prefix")
            or evidence.get("detached_weak_prefix")
            or evidence.get("unsupported_weak_prefix")
        ) and not recovery_trusted:
            return False
        return True

    # Explicit multi-source/acoustic timing trust remains valid for non-CTC
    # backends.  Otherwise a non-reviewed timed assignment is accepted here;
    # content confidence is checked separately by assignment_is_trusted().
    if item.get("timing_trusted"):
        return True
    return not bool(item.get("review_required"))


def report_has_verified_central_commit(report: dict[str, object]) -> bool:
    """Return true only for a source-bound, internally consistent central commit.

    Merely carrying a ``central_timing_state`` field is not authority.  Metrics
    may preserve central content identity only when the serialized commit is
    cryptographically bound to this source and each row is rebound to the
    assignment candidate/provenance that central evaluation actually wrote.
    """
    assignments = report.get("assignments")
    state = report.get("central_timing_state")
    selection = report.get("candidate_selection")
    if (
        report.get("timing_state_schema_version") != 3
        or not isinstance(assignments, list)
        or not isinstance(state, list)
        or len(state) != len(assignments)
        or not isinstance(selection, dict)
        or selection.get("status") != "selected"
        or selection.get("metric_stage") != "central-evaluated"
        or selection.get("selected_backend") != report.get("backend")
    ):
        return False
    try:
        source_sha = validated_loaded_source_sha256()
    except (OSError, LrcError):
        return False
    if report.get("algorithm_source_sha256") != source_sha:
        return False

    for index, (raw_assignment, raw_row) in enumerate(zip(assignments, state), start=1):
        if not isinstance(raw_assignment, dict) or not isinstance(raw_row, dict):
            return False
        if raw_row.get("entry") != index or raw_assignment.get("entry") != index:
            return False
        status = raw_assignment.get("timestamp_status")
        if status not in {"selected_valid", "provisional_unresolved"}:
            return False
        if raw_row.get("timestamp_status") != status:
            return False
        centiseconds = raw_assignment.get("canonical_written_centiseconds")
        if not isinstance(centiseconds, int) or raw_row.get("canonical_written_centiseconds") != centiseconds:
            return False
        try:
            timestamp = float(raw_assignment.get("timestamp"))
            row_timestamp = float(raw_row.get("timestamp"))
        except (TypeError, ValueError):
            return False
        if (
            canonicalize_lrc_time(timestamp).centiseconds != centiseconds
            or canonicalize_lrc_time(row_timestamp).centiseconds != centiseconds
        ):
            return False
        provenance = raw_assignment.get("candidate_provenance")
        if not isinstance(provenance, dict):
            return False
        row_candidate_id = raw_row.get("candidate_id")
        row_source = raw_row.get("candidate_source")
        if (
            not isinstance(row_candidate_id, str)
            or not row_candidate_id
            or provenance.get("candidate_id") != row_candidate_id
            or provenance.get("source") != row_source
            or raw_assignment.get("candidate_source") != row_source
            or provenance.get("canonical_written_centiseconds") != centiseconds
        ):
            return False
        timing_trusted = bool(raw_row.get("timing_trusted"))
        if bool(raw_assignment.get("timing_trusted")) != timing_trusted:
            return False
        bound_candidate_id = (
            raw_assignment.get("selected_candidate_id")
            if timing_trusted
            else raw_assignment.get("provisional_candidate_id")
        )
        if bound_candidate_id != row_candidate_id:
            return False
        if timing_trusted != (status == "selected_valid"):
            return False
        if raw_assignment.get("timestamp_revision") != raw_assignment.get("timing_evidence_revision"):
            return False
    return True


# A single, declarative authority for final-line defect classification.
# Unknown diagnostics are INFO until an owner explicitly classifies them.
SILENT_RISK_REVIEW_FLAG = "silent-risk:selected-confidence-le-0.8-and-next-gap-ge-19"


FINAL_LINE_TRUST_CLASSIFICATION: dict[str, str] = {
    **{code: "INFO" for code in {
        "manual_anchor_resolved", "central-selected", "central-provisional",
        "ctc_final_guard_recovered", "ctc_low_path_confidence",
        "raw_candidate_rejected_by_sequence", "raw_asr_systematic_drift",
        "long_line_disagreement", "skeleton_global_ctc_disagreement",
        "short_line_onset_uncertain",
    }},
    **{code: "UNTRUSTED" for code in NON_ACTIONABLE_VERIFICATION_FAILURES | {
        "timing-evidence-unverified", "insufficient-minimum-proof",
        # LRC 1.5 W6 (R1 audit): a CTC/final disagreement alone no longer
        # forces human review; the row's status follows its timing evidence.
        "F4-ctc-final-disagreement",
    }},
    **{code: "REVIEW" for code in ACTIONABLE_TIMING_FAILURES
       | ACTIONABLE_SUSPICIOUS_FLAGS
       | {"recurrence-uncorroborated", "recovery-moved",
          "unresolved-material-alternative", "F1-vocal-absent",
          "F1-vocal-check-unavailable",
          "F3-compressed-or-nonmonotonic", "F3-nonmonotonic",
          "model-disagreement",
          SILENT_RISK_REVIEW_FLAG}},
}


def self_check_flags_require_review(flags: object) -> bool:
    """True when any final self-check flag is classified REVIEW (W6)."""
    if not isinstance(flags, list):
        return False
    return any(FINAL_LINE_TRUST_CLASSIFICATION.get(str(flag)) == "REVIEW" for flag in flags)


@dataclass(frozen=True)
class FinalLineTrustDecision:
    timing_trusted: bool
    review_required: bool
    reason_codes: tuple[str, ...]


def final_line_trust(
    assignment: dict[str, object],
    suspicious_rows: list[dict[str, object]],
    *,
    central_committed: bool,
) -> FinalLineTrustDecision:
    """Compute timing trust and review once from final evidence and classified codes."""
    codes: list[str] = []
    flags = assignment.get("flags")
    if isinstance(flags, list):
        codes.extend(str(code) for code in flags if str(code) not in ACTIONABLE_SUSPICIOUS_FLAGS or str(code) in ACTIONABLE_TIMING_FAILURES)
    self_check = assignment.get("self_check_flags")
    if isinstance(self_check, list):
        codes.extend(str(code) for code in self_check)
    evidence = assignment.get("normalized_timing_evidence")
    if isinstance(evidence, dict):
        failures = evidence.get("structural_failures")
        if isinstance(failures, (list, tuple)):
            codes.extend(str(code) for code in failures)
        if not evidence.get("minimum_proof_satisfied"):
            codes.append("insufficient-minimum-proof")
    for row in suspicious_rows:
        if _suspicious_row_requires_review(row):
            row_flags = row.get("flags")
            if isinstance(row_flags, list):
                codes.extend(str(code) for code in row_flags)
    reasons = list(dict.fromkeys(
        code for code in codes
        if FINAL_LINE_TRUST_CLASSIFICATION.get(code, "INFO") != "INFO"
    ))
    review = bool(
        (central_committed and assignment.get("review_required"))
        or any(FINAL_LINE_TRUST_CLASSIFICATION.get(code) == "REVIEW" for code in reasons)
    )
    trusted = bool(assignment_timing_is_trusted(assignment) and not review)
    if not trusted and not reasons:
        reasons.append("timing-evidence-unverified")
    return FinalLineTrustDecision(trusted, review, tuple(reasons))


def _silent_risk_rule_applies(assignments: list[object], index: int) -> bool:
    if index + 1 >= len(assignments):
        return False
    assignment = assignments[index]
    following = assignments[index + 1]
    if not isinstance(assignment, dict) or not isinstance(following, dict):
        return False
    confidence = assignment.get("confidence")
    current_time = assignment.get("timestamp")
    next_time = following.get("timestamp")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (confidence, current_time, next_time)):
        return False
    if not all(math.isfinite(float(value)) for value in (confidence, current_time, next_time)):
        return False
    return 0.0 <= float(confidence) <= 0.8 and float(next_time) - float(current_time) >= 19.0


def _review_reason_classes_from_trust(
    decision: FinalLineTrustDecision,
    assignment: dict[str, object],
    suspicious_rows: list[dict[str, object]],
) -> list[str]:
    """Keep the review taxonomy derived from the final trust decision."""
    reasons = set(decision.reason_codes)
    classes: list[str] = []
    if any(_timing_failure_requires_review(code) for code in reasons):
        classes.append("actionable-timing-failure")
    if any(
        code in ACTIONABLE_SUSPICIOUS_FLAGS or code == "model-disagreement"
        for code in reasons
    ) or any(_suspicious_row_requires_review(row) for row in suspicious_rows):
        classes.append("actionable-suspicious-conflict")
    if "unresolved-material-alternative" in reasons:
        classes.append("unresolved-material-alternative")
    if self_check_flags_require_review(assignment.get("self_check_flags")):
        classes.append("final-self-check")
    return classes


def finalize_review_summary(report: dict[str, object]) -> None:
    """Single writer of the report-level review summary (LRC 1.5 W4).

    ``review_required``, ``review_required_count``, ``review_required_percent``
    and ``verification_counts`` are derived here, and only here, from the
    per-row ``review_required`` flags.  Row flags themselves are owned by
    ``update_report_confidence_metrics`` (``final_line_trust`` +
    ``_review_reason_classes_from_trust``); any row whose reason classes
    disagree with its flag is a contract violation.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        raise LrcError("Review summary requires report assignments")
    timing_entries = int(report.get("timing_entries", len(assignments)) or len(assignments))
    review_required_count = 0
    trusted_count = 0
    unverified_count = 0
    for item in assignments:
        if not isinstance(item, dict):
            continue
        required = bool(item.get("review_required"))
        if "review_reason_classes" in item:
            classes = item.get("review_reason_classes")
            if not isinstance(classes, list) or required != bool(classes):
                raise LrcError("Review reason classes diverged from review_required")
        review_required_count += required
        status = item.get("verification_status")
        trusted_count += status == "trusted"
        unverified_count += status == "unverified"
    report["review_required_count"] = review_required_count
    report["review_required_percent"] = ratio_percent(review_required_count, timing_entries)
    report["review_required"] = review_required_count > 0
    report["verification_counts"] = {
        "trusted": trusted_count,
        "unverified": unverified_count,
        "review_required": review_required_count,
    }


# LRC 1.7 (B2 ladder over 39 songs vs BK; EvanOG chose the middle cut): a
# REVIEW row whose REVIEW-class reason codes are all in this set was already
# right 92-100% of the time, so it is no longer a human review task.  The
# policy is applied only when the report is written (write_alignment_report and
# just before the review TSV), never inside the pipeline: every intermediate
# trust decision, Central/consensus input and cache key stays byte-identical.
# Demoted rows become UNTRUSTED ("unverified"), never TRUSTED.
WEAK_REVIEW_CODES = frozenset({
    "phonetic_anchor_disagreement",
    "nonmonotonic-line-start",
    "candidate_disagreement",
    "unresolved_raw_ctc_disagreement",
    "unresolved-precentral-disagreement",
    "unresolved-material-alternative",
    "recurrence-uncorroborated",
    "previous-tail-overlap",
})


# Non-review codes that sat on the ladder's chosen side of the cut.  The ladder
# demotes a row only when EVERY reason code it carries is on that side, so a
# row with a weak REVIEW code plus e.g. no-strict-opening-proof stays in REVIEW.
WEAK_COMPANION_CODES = frozenset({
    "F4-ctc-final-disagreement",
    "insufficient-minimum-proof",
    "previous-tail-ownership-unknown",
})


WEAK_REVIEW_POLICY_KEY = "weak_review_policy"


def _weak_review_demotes(item: dict[str, object]) -> bool:
    if item.get("verification_status") != "review-required" and not item.get("review_required"):
        return False
    codes = item.get("final_trust_reason_codes")
    if not isinstance(codes, list) or not codes:
        return False
    # Same rule as the ladder: every code on the row is inside the chosen cut.
    # (Rows held in REVIEW by Central with only F4 also fall inside it.)
    return {str(code) for code in codes} <= WEAK_REVIEW_CODES | WEAK_COMPANION_CODES


def apply_weak_review_policy(report: dict[str, object]) -> list[int]:
    """Demote REVIEW rows that carry only WEAK reason codes to UNTRUSTED (output time only).

    Idempotent; returns the 1-based entries demoted by this call.  Rows demoted
    earlier are recorded under ``report[WEAK_REVIEW_POLICY_KEY]``.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return []
    demoted: list[int] = []
    for index, item in enumerate(assignments):
        if not isinstance(item, dict) or not _weak_review_demotes(item):
            continue
        item["review_required"] = False
        if "review_reason_classes" in item:
            item["review_reason_classes"] = []
        item["overall_trusted"] = False
        item["verification_status"] = "unverified"
        item["weak_review_demoted"] = True
        demoted.append(index + 1)
    if not demoted:
        return []
    entries = set(demoted)
    suspicious = report.get("suspicious_alignments")
    if isinstance(suspicious, list):
        for row in suspicious:
            if isinstance(row, dict) and row.get("entry") in entries and row.get("review_required"):
                row["review_required"] = False
                row["weak_review_demoted"] = True
    for key in ("low_confidence_entries", "unverified_entries"):
        rows = report.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("entry") in entries:
                    row["verification_status"] = "unverified"
                    failures = row.get("trust_failures")
                    if isinstance(failures, list):
                        row["trust_failures"] = [f for f in failures if f != "actionable-review"]
    timing_review = report.get("timing_review_entries")
    if isinstance(timing_review, list):
        report["timing_review_entries"] = [
            row for row in timing_review
            if not (isinstance(row, dict) and row.get("entry") in entries)
        ]
    unverified_rows = report.get("unverified_entries")
    if isinstance(unverified_rows, list):
        known = {row.get("entry") for row in unverified_rows if isinstance(row, dict)}
        for entry in demoted:
            if entry not in known:
                unverified_rows.append({"entry": entry, "verification_status": "unverified",
                                        "trust_failures": ["timing-untrusted"]})
        unverified_rows.sort(key=lambda row: row.get("entry", 0) if isinstance(row, dict) else 0)
    policy = report.get(WEAK_REVIEW_POLICY_KEY)
    previous = policy.get("demoted_entries") if isinstance(policy, dict) else []
    report[WEAK_REVIEW_POLICY_KEY] = {
        "schema": 1,
        "weak_codes": sorted(WEAK_REVIEW_CODES | WEAK_COMPANION_CODES),
        "demoted_entries": sorted(set(previous if isinstance(previous, list) else []) | entries),
    }
    finalize_review_summary(report)
    if "unverified_count" in report:
        timing_entries = int(report.get("timing_entries", len(assignments)) or len(assignments))
        unverified = report["verification_counts"]["unverified"]
        report["unverified_count"] = unverified
        report["unverified_percent"] = ratio_percent(unverified, timing_entries)
    return demoted


def update_report_confidence_metrics(
    report: dict[str, object], *, apply_silent_risk: bool = False
) -> None:
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return
    timing_entries = int(report.get("timing_entries", len(assignments)) or len(assignments))
    assigned_entries = sum(1 for item in assignments if assignment_has_timing(item))
    central_committed = report_has_verified_central_commit(report)

    suspicious = report.get("suspicious_alignments")
    suspicious_rows = [row for row in suspicious if isinstance(row, dict)] if isinstance(suspicious, list) else []
    for row in suspicious_rows:
        # A later final-line downgrade invalidates an older Central resolution.
        entry = row.get("entry")
        if central_committed and isinstance(entry, int) and 1 <= entry <= len(assignments):
            assignment = assignments[entry - 1]
            if isinstance(assignment, dict) and row.get("resolution"):
                flags = assignment.get("flags")
                active_failure = isinstance(flags, list) and any(
                    _timing_failure_requires_review(str(flag)) for flag in flags
                )
                if (
                    active_failure
                    or bool(assignment.get("review_required"))
                    or not assignment_timing_is_trusted(assignment)
                ):
                    row.pop("resolution", None)
                    if row.get("severity") == "resolved":
                        prior = row.get("original_severity")
                        row["severity"] = prior if prior in {"high", "medium", "low"} else "medium"
        row["review_required"] = _suspicious_row_requires_review(row)
    report["suspicious_alignment_count"] = len(suspicious_rows)
    severity_counts = {"high": 0, "medium": 0, "low": 0}
    for row in suspicious_rows:
        severity = str(row.get("severity") or "medium")
        if severity not in severity_counts:
            severity = "medium"
        severity_counts[severity] += 1
    report["suspicious_alignment_severity_counts"] = severity_counts
    report["actionable_suspicious_alignment_severity_counts"] = _actionable_suspicious_severity_counts(report)

    def content_flag(item: object) -> bool:
        if not isinstance(item, dict):
            return False
        if central_committed:
            return bool(item.get("content_trusted"))
        return assignment_content_is_trusted(item)

    def timing_flag(item: object) -> bool:
        if not isinstance(item, dict):
            return False
        return bool(item.get("timing_audit_trusted")) if central_committed else assignment_timing_is_trusted(item)

    central_rows = report.get("central_timing_state")
    for index, item in enumerate(assignments):
        if not isinstance(item, dict):
            continue
        related_suspicious = [
            row for row in suspicious_rows if row.get("entry") == index + 1
        ]
        if (
            apply_silent_risk
            and not bool(item.get("review_required"))
            and _silent_risk_rule_applies(assignments, index)
        ):
            flags = item.get("flags")
            item["flags"] = list(dict.fromkeys([
                *(flags if isinstance(flags, list) else []),
                SILENT_RISK_REVIEW_FLAG,
            ]))
        decision = final_line_trust(
            item, related_suspicious, central_committed=central_committed
        )
        content_trusted = content_flag(item)
        timing_trusted = decision.timing_trusted
        review_required = decision.review_required
        if central_committed or "review_reason_classes" in item:
            item["review_reason_classes"] = _review_reason_classes_from_trust(
                decision, item, related_suspicious
            )
            review_required = bool(item["review_reason_classes"])
        item["content_trusted"] = content_trusted
        item["timing_audit_trusted"] = timing_trusted
        item["final_trust_reason_codes"] = list(decision.reason_codes)
        if central_committed:
            if not isinstance(central_rows, list) or not isinstance(central_rows[index], dict):
                raise LrcError("Verified Central report lost its timing row")
            central_row = central_rows[index]
            central_row["timing_trusted"] = timing_trusted
            item["timing_trusted"] = timing_trusted
            status = "selected_valid" if timing_trusted else "provisional_unresolved"
            item["timestamp_status"] = status
            central_row["timestamp_status"] = status
            candidate_id = central_row["candidate_id"]
            item["selected_candidate_id"] = candidate_id if timing_trusted else None
            item["provisional_candidate_id"] = None if timing_trusted else candidate_id
        item["review_required"] = review_required
        item["overall_trusted"] = bool(content_trusted and timing_trusted and not review_required)
        item["verification_status"] = (
            "review-required"
            if review_required
            else "trusted"
            if item["overall_trusted"]
            else "unverified"
        )
    if central_committed and not report_has_verified_central_commit(report):
        raise LrcError("Unified final trust broke verified Central commit")

    content_trusted_entries = sum(1 for item in assignments if content_flag(item))
    timing_trusted_entries = sum(1 for item in assignments if timing_flag(item))
    trusted_indices = [
        index for index, item in enumerate(assignments)
        if isinstance(item, dict) and bool(item.get("overall_trusted"))
    ]
    review_indices = [
        index for index, item in enumerate(assignments)
        if isinstance(item, dict) and bool(item.get("review_required"))
    ]
    untrusted_indices = [
        index for index in range(len(assignments)) if index not in set(trusted_indices)
    ]
    review_set = set(review_indices)
    unverified_indices = [index for index in untrusted_indices if index not in review_set]

    existing_low = report.get("low_confidence_entries")
    existing_by_entry: dict[int, dict[str, object]] = {}
    if isinstance(existing_low, list):
        for row in existing_low:
            if isinstance(row, dict) and isinstance(row.get("entry"), int):
                existing_by_entry[int(row["entry"])] = row

    def verification_row(index: int) -> dict[str, object]:
        entry_number = index + 1
        item = assignments[index]
        row = dict(existing_by_entry.get(entry_number, {}))
        row.setdefault("entry", entry_number)
        if isinstance(item, dict):
            row.setdefault("score", round(assignment_score(item), 6))
            reasons: list[str] = []
            if not content_flag(item):
                reasons.append("content-untrusted")
            if not timing_flag(item):
                reasons.append("timing-untrusted")
            if item.get("review_required"):
                reasons.append("actionable-review")
            row["trust_failures"] = reasons
            row["verification_status"] = item.get("verification_status")
        return row

    rebuilt_low = [verification_row(index) for index in untrusted_indices]
    unverified_rows = [verification_row(index) for index in unverified_indices]
    report["low_confidence_entries"] = rebuilt_low
    report["unverified_entries"] = unverified_rows

    report["timing_review_entries"] = [
        {
            "entry": index + 1,
            "flags": list(assignments[index].get("flags", []))
            if isinstance(assignments[index], dict) and isinstance(assignments[index].get("flags"), list)
            else [],
        }
        for index in review_indices
    ]

    trusted_entries = len(trusted_indices)
    low_confidence_count = len(untrusted_indices)
    unverified_count = len(unverified_indices)
    report["assigned_entries"] = assigned_entries
    report["assigned_percent"] = ratio_percent(assigned_entries, timing_entries)
    report["content_trusted_entries"] = content_trusted_entries
    report["content_trusted_percent"] = ratio_percent(content_trusted_entries, timing_entries)
    report["timing_trusted_entries"] = timing_trusted_entries
    report["timing_trusted_percent"] = ratio_percent(timing_trusted_entries, timing_entries)
    report["overall_trusted_entries"] = trusted_entries
    report["overall_trusted_percent"] = ratio_percent(trusted_entries, timing_entries)
    report["authority_trusted_entries"] = trusted_entries
    report["authority_trusted_percent"] = report["overall_trusted_percent"]
    report["trusted_entries"] = trusted_entries
    report["trusted_percent"] = report["overall_trusted_percent"]
    report["low_confidence_count"] = low_confidence_count
    report["low_confidence_percent"] = ratio_percent(low_confidence_count, timing_entries)
    report["unverified_count"] = unverified_count
    report["unverified_percent"] = ratio_percent(unverified_count, timing_entries)
    finalize_review_summary(report)
    report["final_line_trust_classification"] = dict(sorted(FINAL_LINE_TRUST_CLASSIFICATION.items()))
    report["trust_metric_semantics"] = {
        "schema_version": 2,
        "timing_trusted_percent": "final line evidence and review classification",
        "authority_trusted_percent": (
            "content and timing are trusted and no actionable review remains"
        ),
        "trusted_percent_legacy_alias": "authority_trusted_percent",
    }
    report["matched_entries"] = trusted_entries
    report["matched_percent"] = report["trusted_percent"]


def assert_report_review_contract(report: dict[str, object]) -> None:
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        raise LrcError("Review contract requires report assignments")
    review_rows = [
        item for item in assignments
        if isinstance(item, dict) and bool(item.get("review_required"))
    ]
    unverified_rows = [
        item for item in assignments
        if isinstance(item, dict) and item.get("verification_status") == "unverified"
    ]
    trusted_rows = [
        item for item in assignments
        if isinstance(item, dict) and item.get("verification_status") == "trusted"
    ]
    review_status_rows = [
        item for item in assignments
        if isinstance(item, dict) and item.get("verification_status") == "review-required"
    ]
    for item in assignments:
        if not isinstance(item, dict) or "review_reason_classes" not in item:
            continue
        classes = item.get("review_reason_classes")
        if not isinstance(classes, list) or bool(item.get("review_required")) != bool(classes):
            raise LrcError("Review reason classes diverged from review_required")
    row_count = len(review_rows)
    count = int(report.get("review_required_count", -1) or 0)
    percent = float(report.get("review_required_percent", -1.0) or 0.0)
    expected_percent = ratio_percent(row_count, len(assignments))
    unverified_count = int(report.get("unverified_count", len(unverified_rows)) or 0)
    statuses_are_partition = (
        len(trusted_rows) + len(unverified_rows) + len(review_status_rows) == len(assignments)
        and len(review_status_rows) == row_count
        and all(item.get("verification_status") == "review-required" for item in review_rows)
    )
    if not (
        report.get("review_required") is (row_count > 0)
        and count == row_count
        and percent == expected_percent
        and unverified_count == len(unverified_rows)
        and statuses_are_partition
    ):
        raise LrcError("Report trusted/unverified/review contract diverged")


def candidate_supplies_independent_content_identity(
    candidate: TimingCandidate,
    entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Whether the selected candidate independently proves lyric identity.

    Identity may come from a provenance-bound recurrence search or from a
    runtime-only independent ASR/alignment backend.  CTC token spans by
    themselves never become independent content authority.
    """
    if candidate_has_independent_backend_identity(candidate, entries):
        return True
    return bool(
        candidate.source in {"global-acoustic-recurrence", "mutual-acoustic-recurrence"}
        and candidate.independent_content_identity
        and candidate.identity_support == "supported"
        and candidate.sequence_support == "supported"
        and candidate_has_bound_acoustic_evidence(candidate)
        and candidate_has_bound_direct_onset_evidence(candidate, entries)
    )


def _central_summary_row_flags(
    report: dict[str, object],
    state: FinalTimingState,
    content_identity_observations: tuple[BackendContentIdentityObservation, ...] = (),
) -> tuple[
    tuple[bool, ...],
    tuple[bool, ...],
    tuple[bool, ...],
    tuple[bool, ...],
    tuple[bool, ...],
    frozenset[int],
]:
    """Return Central content/timing/review flags without changing state."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(state.audits):
        raise LrcError("Central timing row flags do not match report assignments")
    if len(state.decisions) != len(state.audits) or len(state.written_times) != len(state.audits):
        raise LrcError("Central timing decision/written-time/audit length mismatch")
    material_conflict_entries = (
        _final_post_central_material_conflict_entries(state, report)
        | _runtime_peer_review_only_conflicts(report)
    )
    runtime_entries = report.get(_RUNTIME_LYRIC_ENTRIES_KEY)
    occurrence_entries = (
        tuple(runtime_entries)
        if isinstance(runtime_entries, (list, tuple))
        and len(runtime_entries) == len(state.decisions)
        and all(isinstance(item, LyricEntry) for item in runtime_entries)
        else None
    )
    content_observation_entries = {
        item.entry_index
        for item in content_identity_observations
        if (
            item.backend_lineage
            and item.lyric_revision
            and item.occurrence_revision
            and item.evidence_revision
        )
    }
    content_flags = tuple(
        assignment_content_is_trusted(item)
        or candidate_supplies_independent_content_identity(
            decision.candidate, occurrence_entries
        )
        or bool(runtime_independent_content_identity_sources_for_entry(report, index))
        or index in content_observation_entries
        for index, (item, decision) in enumerate(zip(assignments, state.decisions, strict=True))
    )
    timing_flags = tuple(audit.timing_trusted for audit in state.audits)
    review_flags = tuple(
        any(_timing_failure_requires_review(code) for code in audit.failures)
        or _suspicious_entry_requires_review_for_decision(
            report, index + 1, decision, audit
        )
        or index + 1 in material_conflict_entries
        for index, (decision, audit) in enumerate(zip(state.decisions, state.audits))
    )
    overall_flags = tuple(
        content and timing and not review
        for content, timing, review in zip(content_flags, timing_flags, review_flags)
    )
    unverified_flags = tuple(
        (not overall) and (not review)
        for overall, review in zip(overall_flags, review_flags)
    )
    return (
        content_flags,
        timing_flags,
        review_flags,
        overall_flags,
        unverified_flags,
        material_conflict_entries,
    )


def summarize_central_timing_state(
    report: dict[str, object],
    state: FinalTimingState,
    *,
    content_identity_observations: tuple[BackendContentIdentityObservation, ...] = (),
) -> CentralTrustSummary:
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(state.audits):
        raise LrcError("Central timing summary does not match report assignments")
    timing_entries = len(assignments)
    (
        content_flags,
        timing_flags,
        review_flags,
        overall_flags,
        unverified_flags,
        _material_conflict_entries,
    ) = _central_summary_row_flags(
        report,
        state,
        content_identity_observations,
    )
    assigned_entries = sum(assignment_has_timing(item) for item in assignments)
    content_count = sum(content_flags)
    timing_count = sum(timing_flags)
    overall_count = sum(overall_flags)
    unverified_count = sum(unverified_flags)
    review_count = sum(review_flags)
    return CentralTrustSummary(
        timing_entries=timing_entries,
        assigned_entries=assigned_entries,
        content_trusted_entries=content_count,
        timing_trusted_entries=timing_count,
        overall_trusted_entries=overall_count,
        unverified_count=unverified_count,
        review_required_count=review_count,
        assigned_percent=ratio_percent(assigned_entries, timing_entries),
        content_trusted_percent=ratio_percent(content_count, timing_entries),
        timing_trusted_percent=ratio_percent(timing_count, timing_entries),
        overall_trusted_percent=ratio_percent(overall_count, timing_entries),
        unverified_percent=ratio_percent(unverified_count, timing_entries),
        review_required_percent=ratio_percent(review_count, timing_entries),
        review_required=review_count > 0,
    )


def central_evaluated_quality(
    report: dict[str, object],
    summary: CentralTrustSummary,
) -> float:
    """Rank available timing coverage, retaining global error/collapse penalties.

    Review flags can increase when a backend exposes competing evidence.
    Counting them as missing timing proof can favor a sparse backend that has
    no alternatives to review. Final output trust remains audited separately.
    """
    if report.get("error"):
        return -100.0
    quality = summary.timing_trusted_percent
    if report.get("collapse_detected"):
        quality -= 100.0
    return max(-100.0, min(100.0, quality))


def timing_report_evaluation_projection(
    report: dict[str, object],
) -> dict[str, object]:
    """Return exactly the report view consumed by central evaluation/quality."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        raise LrcError("Central timing evaluation requires report assignments")
    projected_assignments: list[dict[str, object]] = []
    for index, item in enumerate(assignments):
        if not isinstance(item, dict):
            raise LrcError(f"Central timing evaluation received invalid assignment {index + 1}")
        projected_assignments.append({
            "position": index,
            "entry": item.get("entry"),
            "score": item.get("score"),
            "segment": item.get("segment"),
            "timing_repair": item.get("timing_repair"),
            "flags": item.get("flags"),
            "selected_candidate_id": item.get("selected_candidate_id"),
            "provisional_candidate_id": item.get("provisional_candidate_id"),
            "selection_revision": item.get("selection_revision"),
            "timestamp_revision": item.get("timestamp_revision"),
            "timing_evidence_revision": item.get("timing_evidence_revision"),
            "canonical_written_centiseconds": item.get("canonical_written_centiseconds"),
        })
    return {
        "backend": report.get("backend"),
        "error": report.get("error"),
        "collapse_detected": report.get("collapse_detected"),
        "ctc_missing_count": report.get("ctc_missing_count"),
        "ctc_low_score_count": report.get("ctc_low_score_count"),
        "ctc_very_low_score_count": report.get("ctc_very_low_score_count"),
        "suspicious_alignment_severity_counts": report.get(
            "suspicious_alignment_severity_counts"
        ),
        "runtime_content_identity": runtime_content_identity_projection(report),
        "assignment_count": len(projected_assignments),
        "assignments": projected_assignments,
    }


def timing_report_evaluation_revision(report: dict[str, object]) -> str:
    try:
        encoded = json.dumps(
            timing_report_evaluation_projection(report),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LrcError(f"Report evaluation projection is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def freeze_generation_diagnostics(diagnostics: dict[str, object]) -> tuple[str, str]:
    try:
        encoded = json.dumps(
            diagnostics,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LrcError(f"Generation diagnostics are not serializable: {exc}") from exc
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def thaw_generation_diagnostics(evaluated: EvaluatedBackendTiming) -> dict[str, object]:
    if (
        hashlib.sha256(evaluated.generation_diagnostics_json.encode("utf-8")).hexdigest()
        != evaluated.generation_diagnostics_revision
    ):
        raise LrcError("Generation diagnostics changed after central preview")
    value = json.loads(evaluated.generation_diagnostics_json)
    if not isinstance(value, dict):
        raise LrcError("Frozen generation diagnostics are not an object")
    return value


_UNRESOLVED_EVIDENCE_CATEGORY_VALUES = frozenset({
    "CONTENT_IDENTITY",
    "OCCURRENCE",
    "CURRENT_OPENING",
    "PREDECESSOR_TAIL",
    "OWNERSHIP_INDEPENDENCE",
    "ALTERNATIVE_FALSIFICATION",
    "CROSS_BACKEND_ARBITRATION",
    "PROVENANCE",
    "SUSPICIOUS_CONFLICT",
})


def _validate_unresolved_evidence_requirements_projection(
    report: dict[str, object],
) -> None:
    projection = report.get(_UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY)
    if projection is None:
        return
    if not isinstance(projection, dict):
        raise LrcError("Unresolved evidence requirements projection is not an object")
    if projection.get("authority") != "diagnostic-only":
        raise LrcError("Unresolved evidence requirements projection claimed authority")
    revision = projection.get("projection_revision")
    if not isinstance(revision, str):
        raise LrcError("Unresolved evidence requirements revision is missing")
    payload = dict(projection)
    payload.pop("projection_revision", None)
    if _timing_content_digest(payload) != revision:
        raise LrcError("Unresolved evidence requirements digest changed")
    binding = projection.get("revision_binding")
    if not isinstance(binding, dict):
        raise LrcError("Unresolved evidence requirements revision binding is missing")
    current_source = validated_loaded_source_sha256()
    if binding.get("algorithm_source_sha256") != current_source:
        raise LrcError("Unresolved evidence requirements source binding changed")
    if binding.get("algorithm_revision") != report.get("algorithm_revision", LRC_ALGORITHM_REVISION):
        raise LrcError("Unresolved evidence requirements algorithm binding changed")
    if binding.get("lyric_entry_revision") != report.get("lyric_entry_revision"):
        raise LrcError("Unresolved evidence requirements lyric binding changed")
    if binding.get("source_audio_sha256") != report.get("source_audio_sha256"):
        raise LrcError("Unresolved evidence requirements source-audio binding changed")
    if binding.get("vocal_stem_sha256") != report.get("vocal_stem_sha256"):
        raise LrcError("Unresolved evidence requirements vocal-stem binding changed")
    expected_selection = _timing_content_digest(report.get("candidate_selection", {}))
    expected_state = _timing_content_digest(report.get("central_timing_state", []))
    if binding.get("selection_revision") != expected_selection:
        raise LrcError("Unresolved evidence requirements selection binding changed")
    if binding.get("central_state_revision") != expected_state:
        raise LrcError("Unresolved evidence requirements Central-state binding changed")
    assignments = report.get("assignments")
    rows = projection.get("rows")
    if not isinstance(assignments, list) or not isinstance(rows, list):
        raise LrcError("Unresolved evidence requirements rows are malformed")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("entry"), int):
            raise LrcError("Unresolved evidence requirement row is malformed")
        entry = int(row["entry"])
        if not 1 <= entry <= len(assignments):
            raise LrcError("Unresolved evidence requirement row is out of range")
        assignment = assignments[entry - 1]
        if not isinstance(assignment, dict) or not (
            assignment.get("review_required")
            or assignment.get("verification_status") == "unverified"
        ):
            raise LrcError("Unresolved evidence requirement row is not unresolved")
        categories = row.get("missing_proof_categories")
        if not isinstance(categories, list) or any(
            category not in _UNRESOLVED_EVIDENCE_CATEGORY_VALUES
            for category in categories
        ):
            raise LrcError("Unresolved evidence requirement category is invalid")
    forbidden_keys = {
        "audio_path", "lyrics_path", "output_path", "report_path",
        "whisper_cli_path", "whisper_model_path", "whisperx_python_path",
        "whisperx_helper_path", "anchor_path", "reference_path",
        "raw_transcript", "raw_transcript_path",
    }
    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                if key_text.startswith("_runtime_") or key_text in forbidden_keys:
                    raise LrcError("Private/path field leaked into unresolved evidence projection")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(projection)


def _runtime_peer_review_only_conflicts(
    report: dict[str, object],
) -> frozenset[int]:
    value = report.get(_RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY)
    if isinstance(value, frozenset) and all(isinstance(item, int) for item in value):
        return value
    if isinstance(value, (tuple, list, set)) and all(
        isinstance(item, int) for item in value
    ):
        return frozenset(value)
    return frozenset()


def _validate_backend_provenance_projection(
    report: dict[str, object],
) -> None:
    projection = report.get(_BACKEND_PROVENANCE_PROJECTION_KEY)
    if projection is None:
        return
    if not isinstance(projection, dict):
        raise LrcError("Backend provenance projection is not an object")
    revision = projection.get("projection_revision")
    if not isinstance(revision, str):
        raise LrcError("Backend provenance projection revision is missing")
    payload = dict(projection)
    payload.pop("projection_revision", None)
    if _timing_content_digest(payload) != revision:
        raise LrcError("Backend provenance projection digest changed")
    if projection.get("central_state_binding") != _timing_content_digest(
        report.get("central_timing_state", [])
    ):
        raise LrcError("Backend provenance projection central-state binding changed")
    if projection.get("selection_binding") != _timing_content_digest(
        report.get("candidate_selection", {})
    ):
        raise LrcError("Backend provenance projection selection binding changed")
    current_source_sha = validated_loaded_source_sha256()
    if projection.get("algorithm_source_sha256") != current_source_sha:
        raise LrcError("Backend provenance projection source binding changed")
    peer_simulation = projection.get("peer_simulation")
    if peer_simulation is not None:
        if not isinstance(peer_simulation, dict):
            raise LrcError("Backend peer simulation projection is not an object")
        if peer_simulation.get("authority") is not False:
            raise LrcError("Backend peer simulation cannot claim authority")
        peer_revision = peer_simulation.get("simulation_revision")
        if not isinstance(peer_revision, str):
            raise LrcError("Backend peer simulation revision is missing")
        peer_payload = dict(peer_simulation)
        peer_payload.pop("simulation_revision", None)
        if _timing_content_digest(peer_payload) != peer_revision:
            raise LrcError("Backend peer simulation digest changed")
        if peer_simulation.get("status") not in {"SUCCESS", "BLOCKED"}:
            raise LrcError("Backend peer simulation status is invalid")
    review_projection = projection.get("cross_backend_material_conflicts")
    if review_projection is not None:
        if not isinstance(review_projection, dict):
            raise LrcError("Cross-backend material review projection is not an object")
        if review_projection.get("authority") != "review-only":
            raise LrcError("Cross-backend material projection claimed non-review authority")
        review_revision = review_projection.get("revision")
        if not isinstance(review_revision, str):
            raise LrcError("Cross-backend material review revision is missing")
        review_payload = dict(review_projection)
        review_payload.pop("revision", None)
        if _timing_content_digest(review_payload) != review_revision:
            raise LrcError("Cross-backend material review projection digest changed")
        if review_projection.get("status") not in {"RETRIEVED", "ABSENT"}:
            raise LrcError("Cross-backend material review projection status is invalid")
        entries = review_projection.get("entries")
        if not isinstance(entries, list):
            raise LrcError("Cross-backend material review entries are invalid")
        assignments = report.get("assignments")
        if isinstance(assignments, list):
            for item in entries:
                if not isinstance(item, dict) or not isinstance(item.get("entry"), int):
                    raise LrcError("Cross-backend material review entry is malformed")
                entry_index = item["entry"] - 1
                if not 0 <= entry_index < len(assignments):
                    raise LrcError("Cross-backend material review entry is out of range")
                assignment = assignments[entry_index]
                if not isinstance(assignment, dict) or not assignment.get("review_required"):
                    raise LrcError("Cross-backend material review was not propagated")


def apply_central_summary_to_report(
    report: dict[str, object],
    summary: CentralTrustSummary,
) -> None:
    report["timing_entries"] = summary.timing_entries
    report["assigned_entries"] = summary.assigned_entries
    report["assigned_percent"] = summary.assigned_percent
    report["content_trusted_entries"] = summary.content_trusted_entries
    report["content_trusted_percent"] = summary.content_trusted_percent
    report["timing_trusted_entries"] = summary.timing_trusted_entries
    report["timing_trusted_percent"] = summary.timing_trusted_percent
    report["overall_trusted_entries"] = summary.overall_trusted_entries
    report["overall_trusted_percent"] = summary.overall_trusted_percent
    report["authority_trusted_entries"] = summary.overall_trusted_entries
    report["authority_trusted_percent"] = summary.overall_trusted_percent
    report["trusted_entries"] = summary.overall_trusted_entries
    report["trusted_percent"] = summary.overall_trusted_percent
    report["matched_entries"] = summary.overall_trusted_entries
    report["matched_percent"] = summary.overall_trusted_percent
    report["unverified_count"] = summary.unverified_count
    report["unverified_percent"] = summary.unverified_percent
    finalize_review_summary(report)
    report["final_line_trust_classification"] = dict(sorted(FINAL_LINE_TRUST_CLASSIFICATION.items()))
    report["trust_metric_semantics"] = {
        "schema_version": 2,
        "timing_trusted_percent": "final line evidence and review classification",
        "authority_trusted_percent": (
            "content and timing are trusted and no actionable review remains"
        ),
        "trusted_percent_legacy_alias": "authority_trusted_percent",
    }


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
            elif "possible_instrumental_intro_hallucination" in flags:
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

    propagation_seed_indexes = set(suspicious_indexes)
    for index, entry in enumerate(entries):
        if (
            index == 0
            or index in suspicious_indexes
            or index - 1 not in propagation_seed_indexes
        ):
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
    report["suspicious_alignment_count"] = len(suspicious)
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
    update_report_confidence_metrics(report)


def _raw_disagreement_severity(raw_score: float, disagreement: float) -> str:
    """Apply the existing raw/CTC diagnostic severity rule."""
    return "high" if raw_score >= 0.90 or disagreement >= 1.50 else "medium"


def _merge_suspicious_severity(existing: object, incoming: str) -> str:
    """Never let a later diagnostic downgrade an existing defect severity."""
    rank = {"resolved": -1, "low": 0, "medium": 1, "high": 2}
    current = str(existing or "low").lower()
    incoming_text = str(incoming or "low").lower()
    if current not in rank:
        current = "low"
    if incoming_text not in rank:
        incoming_text = "low"
    return current if rank[current] >= rank[incoming_text] else incoming_text


def _ctc_disagreement_severity(disagreement: float) -> str:
    """Apply the existing output-vs-CTC diagnostic severity rule."""
    return "high" if disagreement >= 1.0 else "medium"


def _suspicious_row_has_independent_high_severity_owner(row: object) -> bool:
    """Keep an existing high only when another actionable flag owns it."""
    if not isinstance(row, dict) or str(row.get("severity") or "").lower() != "high":
        return False
    flags = row.get("flags")
    if not isinstance(flags, list):
        return False
    return any(
        str(flag) in ACTIONABLE_SUSPICIOUS_FLAGS
        and str(flag)
        not in {"unresolved_raw_ctc_disagreement", "unresolved_ctc_disagreement"}
        for flag in flags
    )


def write_alignment_report(report_path: Path, report: dict[str, object]) -> None:
    annotate_display_romaji(report, report_path)
    apply_weak_review_policy(report)
    report["algorithm_revision"] = LRC_ALGORITHM_REVISION
    _validate_backend_provenance_projection(report)
    _validate_unresolved_evidence_requirements_projection(report)
    _backend_participation_projection(report)
    serializable_report = {
        key: value
        for key, value in report.items()
        if key not in {
            _RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY,
            _RUNTIME_RAW_CTC_CONSENSUS_KEY,
            _RUNTIME_RAW_CONTENT_IDENTITY_KEY,
            _RUNTIME_WHISPERX_CTC_CONSENSUS_KEY,
            _RUNTIME_WHISPERX_ROW_EVIDENCE_KEY,
            _RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY,
            _RUNTIME_LYRIC_ENTRIES_KEY,
            _RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY,
            _RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY,
            _RUNTIME_WHISPERX_TAIL_SCOPES_KEY,
            _RUNTIME_WHISPERX_TEMP_PATHS_KEY,
            _RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY,
            _RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY,
            _RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY,
            _RUNTIME_POSTCENTRAL_CTC_TAIL_PRODUCER_CONTEXT_KEY,
            _RUNTIME_POSTCENTRAL_HUBERT_BOUNDARY_OBSERVATION_KEY,
            _RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY,
            _RUNTIME_POSTCENTRAL_HUBERT_DUPLICATE_RESOLUTION_KEY,
            _RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY,
            _RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICT_DETAILS_KEY,
            _RUNTIME_RAW_TIMING_DRIFT_FALSIFICATIONS_KEY,
            _RUNTIME_BACKEND_PARTICIPATION_KEY,
        }
    }
    try:
        encoded = json.dumps(
            serializable_report, ensure_ascii=False, indent=2, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise LrcError(f"Alignment report is not serializable canonical JSON: {exc}") from exc
    report_path.write_text(encoded + "\n", encoding="utf-8")


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
