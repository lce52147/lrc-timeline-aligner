"""Extracted from auto_lrc.py (split step 9): local whisper and crossline retry planning and execution, central timing candidate collection."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Literal
import model_cache
from asr_backends import _aligned_exact_phonetic_target_matches, _aligned_exact_target_matches, _aligned_target_prefix_matches, _occurrence_binding_time_bounds, _run_whispercpp_wav, _run_whispercpp_wav_batch, _select_unique_aligned_target_match, _select_unique_aligned_target_prefix_match, asr_transcript, build_bounded_retry_capability, build_local_whisper_capability, decode_temp_wav_window, inactive_legacy_timing_diagnostics, lyric_segment_score, match_whisper_segments, run_whisperx_alignment, segment_char_time, whisperx_segments
from acoustic_hypotheses import _strong_vocal_onset_near, ctc_opening_vocal_fusion_is_valid, mutual_recurrence_hypothesis_provenance_is_valid, phonetic_onset_fusion_provenance_is_valid, recurrence_hypothesis_provenance_is_valid, trusted_tail_release_provenance_is_valid, vocal_activity_reentry_provenance_is_valid, vocal_onset_confirmation_provenance_is_valid
from ctc_refinement import CTC_WIDE_REGION_RADIUS, RAW_CTC_CONSENSUS_MIN_SCORE, SEGMENT_REALIGN_SOURCE, ctc_right_edge_pileup_evidence
from report_trust import _RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY, _suspicious_entry_requires_review, assignment_content_is_trusted, freeze_generation_diagnostics, runtime_independent_content_identity_sources_for_entry, runtime_local_whisper_evidence_for_entry, runtime_raw_content_identity_for_entry, runtime_whisperx_row_evidence_for_entry, timing_report_evaluation_revision
from timing_core import BaseRetryEpoch, BoundedRetryCapability, CTC_WIDE_REGION_FALLBACK_SOURCE, CandidateOccurrenceEvidence, CandidateRegionEvidence, CrossLineRetryOutcome, CrossLineRetryRequest, IndependentTailEvidence, LocalWhisperCapability, LocalWhisperOutcome, LocalWhisperRequest, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, RetryCandidateReference, RuntimeTailEvidence, TRUSTED_ALIGNMENT_SCORE, TailEvidenceContext, TailRegionEvidence, TimingCandidate, _RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY, _active_legacy_candidate, _candidate_confidence, _candidate_support_families, _entry_identity_is_unique, _entry_occurrence_binding, _finite_unit_confidence, _make_tail_provenance, _minimal_unique_sequence_context, _occurrence_binding_has_strong_context, _tail_evidence_context_from_report, _tail_helper_revision, analyze_candidate_regions, build_final_timing_state, candidate_has_bound_direct_onset_evidence, candidate_has_boundary_grade_direct_onset_evidence, candidate_occurrence_evidence_from_binding, derive_reliable_tail, generation_diagnostics_semantic_revision, make_timing_candidate, record_alignment_hypothesis, resolve_reliable_tail, runtime_independent_tail_evidence
from lyrics_io import AsrSegment, LyricEntry, _RUNTIME_LYRIC_ENTRIES_KEY, assert_lyric_identity_contract, entry_sung_text, first_forced_char_time, normalize_match_text
from audio_features import AudioFeatures, HOP_SECONDS
from runtime_base import DEFAULT_EVIDENCE_WORKERS, LrcError, _canonical_json_digest, _ctc_model_blob_sha256, _model_cache_read, _model_cache_write, _park_persistent_helper_mode, _path_sha256, _run_model_helper_command, _stable_file_sha256, _timing_content_digest, canonicalize_lrc_time, default_ctc_python, progress_log
from runtime_base import TimingTokenSpan, _timing_spans_from_value
from runtime_evidence_binding import (
    _attach_ctc_wide_region_candidates,
    _augment_candidates_with_joint_boundary_consensus,
    _augment_local_ctc_candidates_with_runtime_backend_consensus,
    _candidate_set_semantic_revision,
    _independent_tail_evidence_payload,
    _retry_semantic_state_revision,
    _tail_authority_diagnostics,
    bind_runtime_ctc_assignment_tail_fallback,
    bind_runtime_whisperx_ctc_consensus,
    bind_runtime_whisperx_row_evidence,
    independent_tail_evidence_payload,
    merge_runtime_independent_tail_evidence,
    runtime_raw_ctc_consensus_for_entry,
    runtime_whisperx_ctc_consensus_for_entry,
)
from crossline_retry_planning import (
    CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
    CROSSLINE_RETRY_MAX_WINDOW_EXPANSIONS,
    _central_preview_generation_candidates,
    bind_ctc_generation_candidate_evidence,
    freeze_base_retry_epoch,
    _retry_candidate_reference,
    plan_crossline_retry_requests,
    plan_joint_boundary_retry_requests,
    crossline_retry_request_is_current,
)
from crossline_retry_execution import (
    _joint_previous_tail_evidence,
    parse_crossline_retry_output,
    _crossline_retry_helper_cache_key,
    _crossline_retry_semantic_key,
    _novel_crossline_retry_requests,
    invoke_crossline_retry,
    record_crossline_retry_identity_hypothesis,
    execute_crossline_retry_epoch,
    _retry_diagnostic_payload,
    _retry_capability_diagnostic_payload,
)
from local_whisper_requests import (
    LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
    LOCAL_WHISPER_MAX_WINDOW_SECONDS,
    plan_local_whisper_requests,
    _local_whisper_execution_request_revision,
    _local_whisper_request_execution_revision,
    _local_whisper_cache_key,
    _local_whisper_outcome_for_request,
    _shift_local_assignment_to_global,
    bind_runtime_local_whisper_evidence,
    _augment_candidates_with_local_whisper_consensus,
    _apply_context_conditioned_whisper_hypothesis_arbitration,
)


def execute_local_whisper_request(
    request: LocalWhisperRequest,
    capability: LocalWhisperCapability,
    entries: list[LyricEntry],
    args: argparse.Namespace,
    *,
    cache: dict[str, tuple[str, object]] | None = None,
    runtime_stats: dict[str, int] | None = None,
    prepared: tuple[list[AsrSegment], dict[str, object]] | None = None,
    preparation_error: str | None = None,
) -> LocalWhisperOutcome:
    """Run one bounded Whisper.cpp -> WhisperX observation; never select a timestamp directly."""
    persistent_cache_key: str | None = None
    persistent_cache_identity: dict[str, object] | None = None
    def finish(
        status: Literal["evidence", "invalid", "error"],
        reason: str,
        *,
        onset_time: float | None = None,
        score: float | None = None,
        evidence_revision: str | None = None,
        occurrence_binding: dict[str, object] | None = None,
        raw_transcript_revision: str | None = None,
        aligned_transcript_revision: str | None = None,
    ) -> LocalWhisperOutcome:
        base = {
            "entry_index": request.entry_index,
            "request_revision": request.request_revision,
            "status": status,
            "reason": reason,
            "onset_time": onset_time,
            "score": score,
            "evidence_revision": evidence_revision,
            "occurrence_binding": occurrence_binding,
            "raw_transcript_revision": raw_transcript_revision,
            "aligned_transcript_revision": aligned_transcript_revision,
        }
        outcome = LocalWhisperOutcome(
            **base,
            outcome_revision=_timing_content_digest(base),
        )
        if (status != "error" or prepared is not None) and persistent_cache_key is not None and persistent_cache_identity is not None:
            _model_cache_write("local-whisper", persistent_cache_key, persistent_cache_identity, {
                "entry_index": outcome.entry_index,
                "request_revision": outcome.request_revision,
                "status": outcome.status,
                "reason": outcome.reason,
                "onset_time": outcome.onset_time,
                "score": outcome.score,
                "evidence_revision": outcome.evidence_revision,
                "occurrence_binding": outcome.occurrence_binding,
                "raw_transcript_revision": outcome.raw_transcript_revision,
                "aligned_transcript_revision": outcome.aligned_transcript_revision,
                "outcome_revision": outcome.outcome_revision,
            }, args)
        return outcome

    if capability.capability_revision != request.capability_revision:
        return finish("invalid", "capability-revision-mismatch")
    if request.audio_revision != capability.audio_sha256:
        return finish("invalid", "audio-revision-mismatch")
    if not 0 <= request.entry_index < len(entries):
        return finish("invalid", "entry-out-of-range")
    source_audio = Path(capability.audio_path)
    try:
        if _stable_file_sha256(source_audio) != capability.audio_sha256:
            return finish("invalid", "source-audio-changed")
    except (OSError, LrcError):
        return finish("invalid", "source-audio-unavailable")

    if cache is None:
        cache = {}
    if runtime_stats is None:
        runtime_stats = {}
    cache_key = _local_whisper_cache_key(request, capability)
    local_identity = model_cache.identity_local_whisper(
        audio_sha256=capability.audio_sha256,
        whisper_model_sha256=capability.whisper_model_sha256,
        protocol_revision=capability.protocol_revision,
        request_revision=_local_whisper_request_execution_revision(request),
    )
    persistent_cache_key = cache_key
    persistent_cache_identity = local_identity
    cached = cache.get(cache_key)
    if isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[1], LocalWhisperOutcome):
        runtime_stats["cache_hits"] = runtime_stats.get("cache_hits", 0) + 1
        return _local_whisper_outcome_for_request(cached[1], request)
    persisted = _model_cache_read("local-whisper", cache_key, local_identity, args)
    if isinstance(persisted, dict) and persisted.get("status") == "error":
        persisted = None
    if isinstance(persisted, dict):
        try:
            outcome = LocalWhisperOutcome(**persisted)
        except (TypeError, ValueError):
            outcome = None
        if outcome is not None:
            outcome = _local_whisper_outcome_for_request(outcome, request)
            cache[cache_key] = (outcome.outcome_revision, outcome)
            runtime_stats["cache_hits"] = runtime_stats.get("cache_hits", 0) + 1
            return outcome
    runtime_stats["cache_misses"] = runtime_stats.get("cache_misses", 0) + 1

    try:
        if preparation_error is not None:
            raise LrcError(preparation_error)
        if prepared is None:
            wav_path = decode_temp_wav_window(
                source_audio, request.window_start, request.window_end
            )
            raw_segments = _run_whispercpp_wav(wav_path, args, suppress_nst=False)
            aligned_payload = run_whisperx_alignment(
                wav_path, asr_transcript(raw_segments), args, predecoded_wav=True
            )
        else:
            raw_segments, aligned_payload = prepared
        aligned_segments = whisperx_segments(aligned_payload)
        runtime_stats["helper_invocations"] = runtime_stats.get("helper_invocations", 0) + 1
    except (LrcError, OSError, ValueError, json.JSONDecodeError) as exc:
        outcome = finish("error", f"local-whisper-failed:{type(exc).__name__}")
        return outcome

    context_entries = [entries[index] for index in request.context_entry_indexes]
    local_duration = request.window_end - request.window_start
    try:
        local_times, local_report = match_whisper_segments(
            context_entries, aligned_segments, local_duration
        )
    except (LrcError, ValueError, IndexError) as exc:
        outcome = finish("error", f"local-match-failed:{type(exc).__name__}")
        return outcome
    assignments = local_report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(context_entries):
        outcome = finish("invalid", "local-assignment-shape-mismatch")
        cache[cache_key] = (outcome.outcome_revision, outcome)
        return outcome

    full_assignments: list[object] = [None] * len(entries)
    target_local_index: int | None = None
    for local_index, global_index in enumerate(request.context_entry_indexes):
        shifted = _shift_local_assignment_to_global(assignments[local_index], request.window_start)
        if shifted is not None:
            full_assignments[global_index] = shifted
        if global_index == request.entry_index:
            target_local_index = local_index
    if target_local_index is None:
        outcome = finish("invalid", "target-context-mapping-missing")
        cache[cache_key] = (outcome.outcome_revision, outcome)
        return outcome
    target_text = normalize_match_text(entry_sung_text(entries[request.entry_index]))
    target_assignment = full_assignments[request.entry_index]
    score = 0.0
    segment_number: int | None = None
    target_assignment_trusted = False
    if isinstance(target_assignment, dict):
        try:
            score = float(target_assignment.get("score", 0.0) or 0.0)
            segment_number = int(target_assignment.get("segment"))
        except (TypeError, ValueError):
            score = 0.0
            segment_number = None
        target_assignment_trusted = bool(
            score >= TRUSTED_ALIGNMENT_SCORE
            and not bool(target_assignment.get("borrowed"))
            and segment_number is not None
            and 1 <= segment_number <= len(aligned_segments)
        )

    occurrence_binding: dict[str, object] | None = None
    recovered_exact_match: dict[str, object] | None = None
    recovered_prefix_match: dict[str, object] | None = None
    recovered_phonetic_match: dict[str, object] | None = None
    if not target_assignment_trusted:
        # Do not promote a weak matcher score.  Recover only from real forced
        # WhisperX characters.  A full exact target is strongest; when the full
        # line is absent from WhisperX segmentation, a forced opening prefix may
        # recover the target only if that synthetic onset makes a full/sparse
        # immutable lyric context uniquely bind the occurrence.  No nearest-time
        # choice is allowed, and partial-prefix recovery never falls back to a
        # global-unique-only identity claim.
        direct_text_score = lyric_segment_score(target_text, target_text)

        def recovered_trials(
            matches: tuple[dict[str, object], ...],
            *,
            prefix_only: bool,
        ) -> list[tuple[dict[str, object], dict[str, object], list[object]]]:
            recovered: list[tuple[dict[str, object], dict[str, object], list[object]]] = []
            for match in matches:
                global_onset = float(match["onset_time"]) + request.window_start
                synthetic = {
                    "timestamp": round(global_onset, 6),
                    "score": direct_text_score,
                    "segment": int(match["start_segment_index"]) + 1,
                    "borrowed": False,
                    "direct_exact_char_stream": not prefix_only,
                    "direct_prefix_char_stream": prefix_only,
                    "char_match_revision": match["match_revision"],
                }
                trial_assignments = list(full_assignments)
                trial_assignments[request.entry_index] = synthetic
                binding = _entry_occurrence_binding(
                    entries, trial_assignments, request.entry_index, require_context=True
                )
                if binding is None and not prefix_only:
                    binding = _entry_occurrence_binding(
                        entries, trial_assignments, request.entry_index
                    )
                if binding is None:
                    continue
                if prefix_only and not _occurrence_binding_has_strong_context(binding):
                    continue
                recovered.append((match, binding, trial_assignments))
            return recovered

        exact_matches = _aligned_exact_target_matches(target_text, aligned_segments)
        recovered = recovered_trials(exact_matches, prefix_only=False)
        unique_recovered = {
            (str(item[0].get("match_revision")), str(item[1].get("binding_revision"))): item
            for item in recovered
        }
        if len(unique_recovered) == 1:
            recovered_exact_match, occurrence_binding, full_assignments = next(iter(unique_recovered.values()))
        elif len(unique_recovered) > 1:
            outcome = finish("invalid", "target-exact-char-occurrence-ambiguous")
            cache[cache_key] = (outcome.outcome_revision, outcome)
            return outcome
        else:
            prefix_matches = _aligned_target_prefix_matches(target_text, aligned_segments)
            recovered_prefix = recovered_trials(prefix_matches, prefix_only=True)
            unique_prefix = {
                (str(item[0].get("match_revision")), str(item[1].get("binding_revision"))): item
                for item in recovered_prefix
            }
            if len(unique_prefix) == 1:
                recovered_prefix_match, occurrence_binding, full_assignments = next(iter(unique_prefix.values()))
            elif len(unique_prefix) > 1:
                outcome = finish("invalid", "target-prefix-char-occurrence-ambiguous")
                cache[cache_key] = (outcome.outcome_revision, outcome)
                return outcome
            else:
                # Exact phonetic identity is a separate content proof.  It can
                # recover a weak text assignment caused by kana/kanji spelling
                # variation, but it does not itself claim a line onset.
                phonetic_matches = _aligned_exact_phonetic_target_matches(
                    entry_sung_text(entries[request.entry_index]), aligned_segments
                )
                phonetic_recovered: list[tuple[dict[str, object], dict[str, object], list[object]]] = []
                for match in phonetic_matches:
                    try:
                        start_segment_index = int(match["start_segment_index"])
                        local_reference_time = float(aligned_segments[start_segment_index].start)
                        phonetic_score = float(match["phonetic_score"])
                    except (KeyError, TypeError, ValueError, IndexError):
                        continue
                    if not math.isfinite(local_reference_time) or phonetic_score < TRUSTED_ALIGNMENT_SCORE:
                        continue
                    synthetic = {
                        "timestamp": round(local_reference_time + request.window_start, 6),
                        "score": round(phonetic_score, 6),
                        "segment": start_segment_index + 1,
                        "borrowed": False,
                        "direct_exact_phonetic_identity": True,
                        "phonetic_match_revision": match["match_revision"],
                    }
                    trial_assignments = list(full_assignments)
                    trial_assignments[request.entry_index] = synthetic
                    binding = _entry_occurrence_binding(
                        entries, trial_assignments, request.entry_index, require_context=True
                    )
                    if binding is None:
                        binding = _entry_occurrence_binding(
                            entries, trial_assignments, request.entry_index
                        )
                    if binding is None:
                        continue
                    phonetic_recovered.append((match, binding, trial_assignments))
                unique_phonetic = {
                    (str(item[0].get("match_revision")), str(item[1].get("binding_revision"))): item
                    for item in phonetic_recovered
                }
                if len(unique_phonetic) == 1:
                    recovered_phonetic_match, occurrence_binding, full_assignments = next(
                        iter(unique_phonetic.values())
                    )
                elif len(unique_phonetic) > 1:
                    outcome = finish("invalid", "target-phonetic-occurrence-ambiguous")
                    cache[cache_key] = (outcome.outcome_revision, outcome)
                    return outcome
                else:
                    outcome = finish(
                        "invalid",
                        "target-local-assignment-untrusted"
                        if not exact_matches and not prefix_matches and not phonetic_matches
                        else "target-independent-identity-occurrence-unproven",
                    )
                    cache[cache_key] = (outcome.outcome_revision, outcome)
                    return outcome

        target_assignment = full_assignments[request.entry_index]
        assert isinstance(target_assignment, dict)
        score = float(target_assignment["score"])
        segment_number = int(target_assignment["segment"])
        target_assignment_trusted = True

    if not target_assignment_trusted or segment_number is None:
        outcome = finish("invalid", "target-local-assignment-untrusted")
        cache[cache_key] = (outcome.outcome_revision, outcome)
        return outcome

    # Prefer a full/sparse trusted sequence bracket.  Unique lyrics retain the
    # weaker global-unique fallback for content identity, but only context-bound
    # observations may participate in hypothesis falsification.
    if occurrence_binding is None:
        occurrence_binding = _entry_occurrence_binding(
            entries, full_assignments, request.entry_index, require_context=True
        )
        if occurrence_binding is None:
            occurrence_binding = _entry_occurrence_binding(
                entries, full_assignments, request.entry_index
            )
    if occurrence_binding is None:
        outcome = finish("invalid", "occurrence-binding-unproven")
        cache[cache_key] = (outcome.outcome_revision, outcome)
        return outcome

    # Timing evidence must be an actual forced-character onset.  Search the full
    # aligned char stream first so a lyric split across WhisperX segments remains
    # directly observable.  No interpolation or nearest-candidate choice is used.
    direct_match = recovered_exact_match or recovered_prefix_match or _select_unique_aligned_target_match(
        target_text, aligned_segments,
        occurrence_binding=occurrence_binding,
        entry_index=request.entry_index,
        preferred_segment_number=segment_number,
    )
    used_prefix_match = recovered_prefix_match is not None
    if direct_match is None:
        direct_match = _select_unique_aligned_target_prefix_match(
            target_text,
            aligned_segments,
            occurrence_binding=occurrence_binding,
            entry_index=request.entry_index,
            preferred_segment_number=segment_number,
        )
        used_prefix_match = direct_match is not None
    direct_local_onset = (
        float(direct_match["onset_time"]) if isinstance(direct_match, dict) else None
    )
    if direct_local_onset is None:
        aligned_segment = aligned_segments[segment_number - 1]
        direct_local_onset = segment_char_time(target_text, aligned_segment)
        if direct_local_onset is None:
            aligned_text = normalize_match_text(aligned_segment.text)
            if aligned_text.startswith(target_text) and target_text:
                direct_local_onset = aligned_segment.start
    onset_time = (
        float(direct_local_onset) + request.window_start
        if direct_local_onset is not None else None
    )
    if onset_time is not None and (
        not math.isfinite(onset_time)
        or onset_time < request.window_start - HOP_SECONDS
        or onset_time > request.window_end + HOP_SECONDS
    ):
        onset_time = None

    lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entries[request.entry_index])).encode("utf-8")
    ).hexdigest()
    raw_projection = [
        {
            "start": round(segment.start, 6),
            "end": round(segment.end, 6),
            "text_digest": _timing_content_digest(normalize_match_text(segment.text)),
        }
        for segment in raw_segments
    ]
    aligned_projection = [
        {
            "start": round(segment.start, 6),
            "end": round(segment.end, 6),
            "text_digest": _timing_content_digest(normalize_match_text(segment.text)),
            "first_char": round(first_forced_char_time(segment), 6),
        }
        for segment in aligned_segments
    ]
    raw_transcript_revision = _timing_content_digest(raw_projection)
    aligned_transcript_revision = _timing_content_digest(aligned_projection)
    evidence_payload = {
        "entry_index": request.entry_index,
        "lyric_sha256": lyric_sha,
        "onset_time": (round(onset_time, 6) if onset_time is not None else None),
        "score": round(score, 6),
        "occurrence_binding": occurrence_binding,
        "request_revision": request.request_revision,
        "capability_revision": capability.capability_revision,
        "audio_revision": capability.audio_sha256,
        "window_start": round(request.window_start, 6),
        "window_end": round(request.window_end, 6),
        "raw_transcript_revision": raw_transcript_revision,
        "aligned_transcript_revision": aligned_transcript_revision,
        "producer": "local-whisper-aligner",
    }
    evidence_revision = _timing_content_digest(evidence_payload)
    if recovered_exact_match is not None:
        outcome_reason = (
            "trusted-local-whisper-exact-char-recovery-and-onset"
            if onset_time is not None
            else "trusted-local-whisper-exact-char-recovery-occurrence-only"
        )
    elif recovered_phonetic_match is not None:
        outcome_reason = (
            "trusted-local-whisper-exact-phonetic-and-onset"
            if onset_time is not None
            else "trusted-local-whisper-exact-phonetic-occurrence-only"
        )
    elif recovered_prefix_match is not None and onset_time is not None:
        outcome_reason = "trusted-local-whisper-context-prefix-recovery-and-onset"
    elif used_prefix_match and onset_time is not None:
        outcome_reason = "trusted-local-whisper-context-prefix-onset"
    elif isinstance(occurrence_binding, dict) and occurrence_binding.get("mode") == "sparse-sequence-bracket":
        outcome_reason = (
            "trusted-local-whisper-sparse-occurrence-and-onset"
            if onset_time is not None
            else "trusted-local-whisper-sparse-occurrence-only"
        )
    else:
        outcome_reason = (
            "trusted-local-whisper-occurrence-and-onset"
            if onset_time is not None
            else "trusted-local-whisper-occurrence-only"
        )
    outcome = finish(
        "evidence",
        outcome_reason,
        onset_time=onset_time,
        score=score,
        evidence_revision=evidence_revision,
        occurrence_binding=occurrence_binding,
        raw_transcript_revision=raw_transcript_revision,
        aligned_transcript_revision=aligned_transcript_revision,
    )
    cache[cache_key] = (outcome.outcome_revision, outcome)
    _model_cache_write("local-whisper", cache_key, local_identity, {
        "entry_index": outcome.entry_index,
        "request_revision": outcome.request_revision,
        "status": outcome.status,
        "reason": outcome.reason,
        "onset_time": outcome.onset_time,
        "score": outcome.score,
        "evidence_revision": outcome.evidence_revision,
        "occurrence_binding": outcome.occurrence_binding,
        "raw_transcript_revision": outcome.raw_transcript_revision,
        "aligned_transcript_revision": outcome.aligned_transcript_revision,
        "outcome_revision": outcome.outcome_revision,
    }, args)
    return outcome


def _prepare_local_whisper_batch(
    work: list[tuple[str, int, LocalWhisperRequest]],
    capability: LocalWhisperCapability,
    args: argparse.Namespace,
) -> tuple[
    dict[str, tuple[list[AsrSegment], dict[str, object]]],
    dict[str, str],
]:
    """Prepare a frozen local-Whisper epoch with batched raw ASR and resident WhisperX.

    This function performs no Central mutation.  It only materializes the same
    per-request raw/aligned observations that execute_local_whisper_request would
    otherwise acquire one process at a time.
    """
    prepared: dict[str, tuple[list[AsrSegment], dict[str, object]]] = {}
    errors: dict[str, str] = {}
    if not work:
        return prepared, errors
    source_audio = Path(capability.audio_path)
    decoded: dict[str, Path] = {}

    def decode_one(item: tuple[str, int, LocalWhisperRequest]) -> tuple[str, Path | None, str | None]:
        key, _ordinal, request = item
        try:
            wav_path = decode_temp_wav_window(source_audio, request.window_start, request.window_end)
            return key, wav_path, None
        except (LrcError, OSError, ValueError) as exc:
            return key, None, f"decode:{type(exc).__name__}:{exc}"

    progress_log(f"local Whisper prepare: decode_windows={len(work)} workers={DEFAULT_EVIDENCE_WORKERS}")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(DEFAULT_EVIDENCE_WORKERS, len(work) or 1),
        thread_name_prefix="lrc-local-decode",
    ) as pool:
        for key, wav_path, error in pool.map(decode_one, work):
            if error is not None or wav_path is None:
                errors[key] = error or "decode:unknown"
            else:
                decoded[key] = wav_path

    raw_inputs = [(key, decoded[key]) for key, _ordinal, _request in work if key in decoded]
    # Raw whisper.cpp gets exclusive ownership of the heavy GPU phase; park the
    # resident WhisperX align model on CPU before entering that phase.
    _park_persistent_helper_mode("whisperx", reason="before-local-whispercpp-batch")
    raw_results, raw_errors = _run_whispercpp_wav_batch(raw_inputs, args, suppress_nst=False)
    errors.update(raw_errors)

    align_items = [
        (key, raw_results[key], decoded[key])
        for key, _ordinal, _request in work
        if key in raw_results and key in decoded and key not in errors
    ]
    progress_log(
        f"local Whisper align batch: requests={len(align_items)} "
        f"resident_workers={DEFAULT_EVIDENCE_WORKERS}"
    )

    def align_one(item: tuple[str, list[AsrSegment], Path]) -> tuple[str, dict[str, object] | None, str | None]:
        key, raw_segments, wav_path = item
        try:
            payload = run_whisperx_alignment(
                wav_path,
                asr_transcript(raw_segments),
                args,
                predecoded_wav=True,
            )
            return key, payload, None
        except (LrcError, OSError, ValueError, json.JSONDecodeError) as exc:
            return key, None, f"align:{type(exc).__name__}:{exc}"

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(DEFAULT_EVIDENCE_WORKERS, len(align_items) or 1),
        thread_name_prefix="lrc-local-whisperx",
    ) as pool:
        for key, payload, error in pool.map(align_one, align_items):
            if error is not None or payload is None:
                errors[key] = error or "align:unknown"
            else:
                prepared[key] = (raw_results[key], payload)
    _park_persistent_helper_mode("whisperx", reason="local-whisperx-align-complete")
    progress_log(
        f"local Whisper prepare complete: prepared={len(prepared)} errors={len(errors)}"
    )
    return prepared, errors


def run_local_whisper_requests(
    requests: tuple[LocalWhisperRequest, ...],
    capability: LocalWhisperCapability,
    entries: list[LyricEntry],
    args: argparse.Namespace,
    *,
    cache: dict[str, tuple[str, object]],
    stats: dict[str, int],
) -> tuple[LocalWhisperOutcome, ...]:
    """Execute one frozen local-Whisper epoch and return outcomes in request order.

    Multi-request epochs always go through _prepare_local_whisper_batch (one
    whisper.cpp batch, one WhisperX alignment batch), whatever the worker
    count.  With one worker this path used to be skipped, so every request
    ran its own whisper.cpp + WhisperX (70 calls for one song), each taking
    the cross-song GPU lock.
    """
    workers = DEFAULT_EVIDENCE_WORKERS
    # A single request has nothing to batch.  Every larger epoch prepares
    # raw ASR and WhisperX alignment in one batch first; the worker count
    # only sets how many evaluations run at once.
    if len(requests) <= 1:
        outcomes: list[LocalWhisperOutcome] = []
        for ordinal, request in enumerate(requests, start=1):
            progress_log(
                f"local Whisper evidence: {ordinal}/{len(requests)} "
                f"entry={request.entry_index + 1} "
                f"window={request.window_start:.2f}-{request.window_end:.2f}s"
            )
            outcome = execute_local_whisper_request(
                request, capability, entries, args,
                cache=cache, runtime_stats=stats,
            )
            outcomes.append(outcome)
            progress_log(
                f"local Whisper evidence: entry={request.entry_index + 1} "
                f"status={outcome.status} reason={outcome.reason}"
            )
        return tuple(outcomes)
    else:
        # Requests are planned from one frozen pre-local-whisper graph and are
        # bound only after every outcome is collected.  They can therefore be
        # executed concurrently without changing Central selection semantics.
        # Preserve deterministic result ordering and legacy cache statistics.
        grouped: dict[str, list[tuple[int, LocalWhisperRequest]]] = {}
        for ordinal, request in enumerate(requests, start=1):
            key = _local_whisper_cache_key(request, capability)
            grouped.setdefault(key, []).append((ordinal, request))

        outcomes_by_ordinal: dict[int, LocalWhisperOutcome] = {}
        work: list[tuple[str, int, LocalWhisperRequest]] = []
        for key, group in grouped.items():
            cached = cache.get(key)
            if (
                isinstance(cached, tuple)
                and len(cached) == 2
                and isinstance(cached[1], LocalWhisperOutcome)
            ):
                for ordinal, grouped_request in group:
                    outcomes_by_ordinal[ordinal] = _local_whisper_outcome_for_request(
                        cached[1], grouped_request
                    )
                stats["cache_hits"] += len(group)
                continue
            first_ordinal, first_request = group[0]
            work.append((key, first_ordinal, first_request))
            stats["cache_misses"] += 1
            if len(group) > 1:
                stats["cache_hits"] += len(group) - 1

        progress_log(
            f"local Whisper evidence: internal_workers={workers} "
            f"unique_exec={len(work)}"
        )
        prepared_local, preparation_errors = _prepare_local_whisper_batch(
            work, capability, args
        )

        def _run_local_parallel(
            item: tuple[str, int, LocalWhisperRequest],
        ) -> tuple[str, int, LocalWhisperOutcome, dict[str, int], dict[str, tuple[str, object]]]:
            key, ordinal, request = item
            private_cache: dict[str, tuple[str, object]] = {}
            private_stats: dict[str, int] = {}
            progress_log(
                f"local Whisper evidence: start {ordinal}/{len(requests)} "
                f"entry={request.entry_index + 1} "
                f"window={request.window_start:.2f}-{request.window_end:.2f}s"
            )
            outcome = execute_local_whisper_request(
                request, capability, entries, args,
                cache=private_cache, runtime_stats=private_stats,
                prepared=prepared_local.get(key),
                preparation_error=preparation_errors.get(key),
            )
            return key, ordinal, outcome, private_stats, private_cache

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(workers, len(work) or 1),
            thread_name_prefix="lrc-local-whisper",
        ) as pool:
            future_map = {pool.submit(_run_local_parallel, item): item for item in work}
            for future in concurrent.futures.as_completed(future_map):
                key, ordinal, outcome, private_stats, private_cache = future.result()
                cache.update(private_cache)
                stats["helper_invocations"] += private_stats.get(
                    "helper_invocations", 0
                )
                for grouped_ordinal, grouped_request in grouped[key]:
                    outcomes_by_ordinal[grouped_ordinal] = _local_whisper_outcome_for_request(
                        outcome, grouped_request
                    )
                progress_log(
                    f"local Whisper evidence: done {ordinal}/{len(requests)} "
                    f"entry={outcome.entry_index + 1} "
                    f"status={outcome.status} reason={outcome.reason}"
                )

        return tuple(
            outcomes_by_ordinal[ordinal]
            for ordinal in range(1, len(requests) + 1)
        )


def _local_whisper_vocal_fusion_candidate(
    *,
    entry_index: int,
    entry: LyricEntry,
    local_evidence: dict[str, object] | None,
    vocal_onset_features: AudioFeatures | None,
    audio_revision: str | None,
    occurrence_evidence: CandidateOccurrenceEvidence | None = None,
) -> TimingCandidate | None:
    """Fuse context-bound Local Whisper identity with an independent vocal onset.

    Local Whisper occurrence-only evidence already proves *which lyric row* is
    present, but intentionally does not manufacture an onset when exact forced
    characters are absent.  The existing isolated-vocal detector is a distinct
    acoustic producer.  When a production-grade vocal onset lies near the
    context-bound Local Whisper semantic time, their fusion may create a real
    boundary-grade candidate.  Global-unique-only identity is excluded here so a
    repeated lyric cannot acquire timing authority without contextual binding.
    """
    if (
        not isinstance(local_evidence, dict)
        or local_evidence.get("onset_time") is not None
        or vocal_onset_features is None
        or not audio_revision
    ):
        return None
    binding = local_evidence.get("occurrence_binding")
    if not _occurrence_binding_has_strong_context(binding):
        return None
    try:
        identity_time = float(binding.get("timestamp"))  # type: ignore[union-attr]
        identity_score = float(local_evidence.get("score"))
    except (TypeError, ValueError, AttributeError):
        return None
    if not math.isfinite(identity_time) or identity_score < TRUSTED_ALIGNMENT_SCORE:
        return None
    peak = _strong_vocal_onset_near(vocal_onset_features, identity_time)
    if peak is None:
        return None
    peak_time, peak_strength, peak_prominence = peak
    artifact = {
        "producer": "local-whisper-vocal-fusion",
        "local_whisper_evidence_revision": local_evidence.get("evidence_revision"),
        "occurrence_binding_revision": (
            binding.get("binding_revision") if isinstance(binding, dict) else None
        ),
        "local_identity_time": round(identity_time, 6),
        "local_identity_score": round(identity_score, 6),
        "vocal_onset_time": round(peak_time, 6),
        "vocal_onset_strength": round(peak_strength, 8),
        "vocal_onset_prominence": round(peak_prominence, 8),
        "audio_sha256": audio_revision,
        "feature_revision": "vocal-consonant-onset-v1",
        "runtime_only": True,
    }
    return make_timing_candidate(
        entry_index=entry_index,
        entry_text=entry_sung_text(entry),
        source="local-whisper-vocal-independent-fusion",
        raw_time=peak_time,
        spans=(),
        confidence=min(identity_score, peak_strength),
        identity_support="supported",
        sequence_support="supported",
        acoustic_support="supported",
        direct_onset_support="supported",
        acoustic_strength=peak_strength,
        acoustic_onset_time=peak_time,
        acoustic_source_artifact=artifact,
        acoustic_evidence_producer="local-whisper-vocal-fusion",
        acoustic_evidence_kind="vocal-onset",
        acoustic_evidence_independent=True,
        direct_onset_time=peak_time,
        direct_onset_source_artifact=artifact,
        direct_onset_evidence_producer="local-whisper-vocal-fusion",
        direct_onset_evidence_kind="vocal-onset",
        direct_onset_evidence_independent=False,
        occurrence_evidence=occurrence_evidence,
        independent_content_identity=True,
        current=False,
        source_artifact=artifact,
    )


def _augment_candidates_with_local_whisper_vocal_consensus(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
    vocal_onset_features: AudioFeatures | None,
    audio_revision: str | None,
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Add context-bound Local Whisper/vocal observations and CTC composites.

    This is an evidence-additive producer.  Local Whisper proves occurrence
    identity; the isolated-vocal detector proves a real acoustic onset; a CTC
    parent may contribute token geometry.  No timestamp is averaged: the
    standalone candidate uses the observed vocal onset, while the composite
    keeps the CTC parent's raw time and binds the same observed onset as direct
    onset evidence.  Whisper-derived parents are excluded so one model lineage
    can never count twice.
    """
    if vocal_onset_features is None or not audio_revision:
        return candidate_sets
    augmented = [list(items) for items in candidate_sets]
    for index, parents in enumerate(candidate_sets):
        evidence = runtime_local_whisper_evidence_for_entry(report, entries, index)
        local_vocal = _local_whisper_vocal_fusion_candidate(
            entry_index=index,
            entry=entries[index],
            local_evidence=evidence,
            vocal_onset_features=vocal_onset_features,
            audio_revision=audio_revision,
            occurrence_evidence=(
                candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="local-whisper-vocal-fusion",
                    occurrence_binding=(
                        evidence.get("occurrence_binding")
                        if isinstance(evidence, dict) else None
                    ),
                    upstream_evidence_revision=(
                        evidence.get("evidence_revision")
                        if isinstance(evidence, dict) else None
                    ),
                )
                if isinstance(evidence, dict) else None
            ),
        )
        if local_vocal is None:
            continue
        augmented[index].append(local_vocal)
        onset_evidence = local_vocal.direct_onset_evidence
        if onset_evidence is None:
            continue
        for parent in parents:
            families = _candidate_support_families(parent)
            if "ctc" not in families or "whisper-family" in families:
                continue
            if (
                parent.identity_support != "supported"
                or parent.sequence_support != "supported"
                or parent.right_edge_pileup
                or parent.window_truncated
                or abs(parent.written_time.seconds - local_vocal.written_time.seconds)
                > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            ):
                continue
            artifact = {
                "producer": "local-whisper-vocal-fusion",
                "local_vocal_candidate_id": local_vocal.candidate_id,
                "local_vocal_generation_revision": local_vocal.generation_revision,
                "local_whisper_evidence_revision": (
                    evidence.get("evidence_revision") if isinstance(evidence, dict) else None
                ),
                "parent_candidate_id": parent.candidate_id,
                "parent_generation_revision": parent.generation_revision,
                "parent_source": parent.source,
                "parent_time": round(parent.raw_time, 6),
                "vocal_onset_time": round(onset_evidence.onset_time, 6),
                "audio_sha256": audio_revision,
                "runtime_only": True,
            }
            augmented[index].append(make_timing_candidate(
                entry_index=index,
                entry_text=entry_sung_text(entries[index]),
                source="ctc-local-whisper-vocal-independent-consensus",
                raw_time=parent.raw_time,
                spans=parent.spans,
                confidence=min(parent.confidence, local_vocal.confidence),
                identity_support="supported",
                sequence_support="supported",
                acoustic_support="supported",
                direct_onset_support="supported",
                acoustic_strength=(
                    local_vocal.acoustic_evidence.strength
                    if local_vocal.acoustic_evidence is not None else None
                ),
                acoustic_onset_time=onset_evidence.onset_time,
                acoustic_source_artifact=artifact,
                acoustic_evidence_producer="local-whisper-vocal-fusion",
                acoustic_evidence_kind="vocal-onset",
                acoustic_evidence_independent=True,
                direct_onset_time=onset_evidence.onset_time,
                direct_onset_source_artifact=artifact,
                direct_onset_evidence_producer="local-whisper-vocal-fusion",
                direct_onset_evidence_kind="cross-backend-onset",
                direct_onset_evidence_independent=True,
                occurrence_evidence=candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="local-whisper-vocal-fusion",
                    occurrence_binding=evidence.get("occurrence_binding"),
                    upstream_evidence_revision=evidence.get("evidence_revision"),
                ),
                independent_content_identity=True,
                current=False,
                prefix_scope=parent.prefix_scope,
                right_edge_pileup=False,
                window_truncated=False,
                source_artifact=artifact,
            ))
    return tuple(
        tuple({item.candidate_id: item for item in items}[key] for key in sorted(
            {item.candidate_id: item for item in items}
        ))
        for items in augmented
    )


def collect_central_timing_candidates(
    audio_path: Path,
    entries: list[LyricEntry],
    scratch_timestamps: list[float],
    report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
    independent_evidence_audio_path: Path | None = None,
    retry_helper_cache: dict[str, tuple[str, object]] | None = None,
    vocal_onset_features: AudioFeatures | None = None,
) -> tuple[tuple[tuple[TimingCandidate, ...], ...], dict[str, object]]:
    """Collect immutable candidates without mutating final timing state.

    Inactive legacy boundary helpers are not called or imported here.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        raise LrcError("Central timing finalizer requires one assignment per lyric entry")
    if len(scratch_timestamps) != len(entries):
        raise LrcError("Central timing scratch clock length does not match lyric entries")
    # Runtime-only immutable lyric objects are retained so content identity can
    # be validated independently from timing provenance during central summary
    # and commit.  This key is explicitly excluded from serialized reports.
    report[_RUNTIME_LYRIC_ENTRIES_KEY] = tuple(entries)
    assert_lyric_identity_contract(entries, report, initialize=True)

    # The legacy prefix guard remains available for historical diagnostics, but
    # it is not executed by production candidate generation and contributes no
    # candidate, trusted bit, recovery, or flag clearing.
    evidence_audio_path = independent_evidence_audio_path or audio_path
    fusion_audio_revision: str | None = None
    if vocal_onset_features is not None:
        try:
            fusion_audio_revision = _path_sha256(evidence_audio_path)
        except (OSError, LrcError):
            fusion_audio_revision = None
    candidate_sets: list[tuple[TimingCandidate, ...]] = []
    for index, (entry, raw_assignment) in enumerate(zip(entries, assignments, strict=True)):
        if not isinstance(raw_assignment, dict):
            raise LrcError(f"Central timing finalizer received invalid assignment {index + 1}")
        # Re-bind the immutable backend baseline instead of trusting a scratch
        # timestamp that a pre-central heuristic may have shifted.  Legacy
        # scoring preserves the backend-selected path as a candidate row.
        baseline_time: float | None = None
        legacy_rows = raw_assignment.get("candidates")
        if isinstance(legacy_rows, list):
            for row in legacy_rows:
                if not isinstance(row, dict) or row.get("source") != "selected_path":
                    continue
                value = row.get("time")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    baseline_time = float(value)
                    break
        if baseline_time is None:
            raw_value = raw_assignment.get("timestamp", scratch_timestamps[index])
            baseline_time = float(raw_value if isinstance(raw_value, (int, float)) else scratch_timestamps[index])
        current_time = baseline_time
        spans = _timing_spans_from_value(raw_assignment.get("ctc_token_spans"))
        backend = str(report.get("backend") or "unknown")
        manual = bool(raw_assignment.get("manual_anchor_hint"))
        if manual:
            manual_value = raw_assignment.get("timestamp", scratch_timestamps[index])
            if isinstance(manual_value, (int, float)) and math.isfinite(float(manual_value)):
                current_time = float(manual_value)
            current_source = "manual-anchor"
        elif backend == "ctc":
            current_source = "ctc-current"
        elif backend == "whisperx":
            current_source = "whisperx-word"
        elif backend in {"lyrics", "checked_lrc"}:
            current_source = "checked-lrc" if backend == "checked_lrc" else "lyrics-source"
        else:
            current_source = f"{backend}-current"
        score = _candidate_confidence(raw_assignment)
        explicit_source = current_source in {"manual-anchor", "checked-lrc", "lyrics-source"}
        ctc_identity = bool(spans)
        aligned_non_ctc = backend == "whisperx" and score >= TRUSTED_ALIGNMENT_SCORE
        current_source_artifact = {
            "backend": backend,
            "timing_repair_source": raw_assignment.get("timing_repair_source"),
            "ctc_local_window": raw_assignment.get("ctc_local_window"),
            "spans": raw_assignment.get("ctc_token_spans"),
            "timestamp": current_time,
        }
        current_candidate = make_timing_candidate(
            entry_index=index,
            entry_text=entry_sung_text(entry),
            source=current_source,
            raw_time=current_time,
            spans=spans,
            confidence=score,
            identity_support="supported" if ctc_identity or aligned_non_ctc or explicit_source else "unavailable",
            sequence_support="supported",
            acoustic_support="unavailable",
            direct_onset_support=(
                "supported" if explicit_source else "unavailable"
            ),
            direct_onset_time=current_time if explicit_source else None,
            direct_onset_source_artifact=(
                current_source_artifact if explicit_source else None
            ),
            direct_onset_evidence_producer=(
                "external-reference" if explicit_source else None
            ),
            direct_onset_evidence_kind=(
                "external-reference" if explicit_source else "generic-onset"
            ),
            direct_onset_evidence_independent=explicit_source,
            current=True,
            right_edge_pileup=bool(raw_assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(raw_assignment.get("ctc_right_edge_truncation")),
            source_artifact=current_source_artifact,
        )
        candidates: list[TimingCandidate] = [current_candidate]

        raw_identity = runtime_raw_content_identity_for_entry(report, entries, index)
        consensus = runtime_raw_ctc_consensus_for_entry(
            report, entries, index, current_time
        )
        if consensus is not None and spans:
            raw_occurrence_evidence = candidate_occurrence_evidence_from_binding(
                entries,
                index,
                producer="raw-asr-consensus",
                occurrence_binding=consensus.get("occurrence_binding"),
                upstream_evidence_revision=consensus.get("evidence_revision"),
            )
            raw_consensus_time = float(consensus["raw_time"])
            raw_consensus_score = float(consensus["raw_score"])
            consensus_artifact = {
                "producer": "raw-asr-consensus",
                "evidence_revision": consensus["evidence_revision"],
                "raw_time": raw_consensus_time,
                "ctc_time": current_time,
                "raw_score": raw_consensus_score,
                "raw_content_identity_revision": (
                    raw_identity.get("evidence_revision") if isinstance(raw_identity, dict) else None
                ),
                "runtime_only": True,
            }
            candidates.append(
                make_timing_candidate(
                    entry_index=index,
                    entry_text=entry_sung_text(entry),
                    source="ctc-raw-independent-consensus",
                    raw_time=current_time,
                    spans=spans,
                    confidence=min(score, raw_consensus_score),
                    identity_support="supported",
                    sequence_support="supported",
                    acoustic_support="unavailable",
                    direct_onset_support="supported",
                    direct_onset_time=raw_consensus_time,
                    direct_onset_source_artifact=consensus_artifact,
                    direct_onset_evidence_producer="raw-asr-consensus",
                    direct_onset_evidence_kind="cross-backend-onset",
                    direct_onset_evidence_independent=True,
                    occurrence_evidence=raw_occurrence_evidence,
                    independent_content_identity=raw_occurrence_evidence is not None,
                    current=False,
                    right_edge_pileup=bool(raw_assignment.get("ctc_right_edge_pileup")),
                    window_truncated=bool(raw_assignment.get("ctc_right_edge_truncation")),
                    source_artifact=consensus_artifact,
                )
            )

        whisperx_consensus = runtime_whisperx_ctc_consensus_for_entry(
            report, entries, index, current_time
        )
        if whisperx_consensus is not None and spans:
            whisperx_occurrence_evidence = candidate_occurrence_evidence_from_binding(
                entries,
                index,
                producer="whisperx-consensus",
                occurrence_binding=whisperx_consensus.get("occurrence_binding"),
                upstream_evidence_revision=whisperx_consensus.get("evidence_revision"),
            )
            whisperx_time = float(whisperx_consensus["whisperx_forced_time"])
            whisperx_score = float(whisperx_consensus["score"])
            whisperx_artifact = {
                "producer": "whisperx-consensus",
                "evidence_revision": whisperx_consensus["evidence_revision"],
                "whisperx_forced_time": whisperx_time,
                "ctc_time": current_time,
                "score": whisperx_score,
                "runtime_only": True,
            }
            candidates.append(
                make_timing_candidate(
                    entry_index=index,
                    entry_text=entry_sung_text(entry),
                    source="ctc-whisperx-independent-consensus",
                    raw_time=current_time,
                    spans=spans,
                    confidence=min(score, whisperx_score),
                    identity_support="supported",
                    sequence_support="supported",
                    acoustic_support="unavailable",
                    direct_onset_support="supported",
                    direct_onset_time=whisperx_time,
                    direct_onset_source_artifact=whisperx_artifact,
                    direct_onset_evidence_producer="whisperx-consensus",
                    direct_onset_evidence_kind="cross-backend-onset",
                    direct_onset_evidence_independent=True,
                    occurrence_evidence=whisperx_occurrence_evidence,
                    independent_content_identity=whisperx_occurrence_evidence is not None,
                    current=False,
                    right_edge_pileup=bool(raw_assignment.get("ctc_right_edge_pileup")),
                    window_truncated=bool(raw_assignment.get("ctc_right_edge_truncation")),
                    source_artifact=whisperx_artifact,
                )
            )


        # Independent row-level backend identity can be fused with the isolated
        # vocal detector without requiring CTC to agree first.  This recovers
        # evidence that was previously thrown away when a backend failed only
        # at the whole-song aggregate level.  The runtime identity timestamp
        # must itself be close to the independently observed vocal onset.
        if isinstance(raw_identity, dict) and vocal_onset_features is not None and fusion_audio_revision:
            try:
                raw_identity_time = float(raw_identity.get("raw_time"))
                raw_identity_score = float(raw_identity.get("raw_score"))
            except (TypeError, ValueError):
                raw_identity_time = math.nan
                raw_identity_score = 0.0
            raw_peak = _strong_vocal_onset_near(
                vocal_onset_features, raw_identity_time, require_absolute_vocal=True
            )
            if raw_peak is not None and raw_identity_score >= TRUSTED_ALIGNMENT_SCORE:
                raw_occurrence_evidence = candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="raw-asr-vocal-fusion",
                    occurrence_binding=raw_identity.get("occurrence_binding"),
                    upstream_evidence_revision=raw_identity.get("evidence_revision"),
                )
                peak_time, peak_strength, peak_prominence = raw_peak
                artifact = {
                    "producer": "raw-asr-vocal-fusion",
                    "raw_content_identity_revision": raw_identity.get("evidence_revision"),
                    "raw_time": round(raw_identity_time, 6),
                    "raw_score": round(raw_identity_score, 6),
                    "vocal_onset_time": round(peak_time, 6),
                    "vocal_onset_strength": round(peak_strength, 8),
                    "vocal_onset_prominence": round(peak_prominence, 8),
                    "audio_sha256": fusion_audio_revision,
                    "feature_revision": "vocal-consonant-onset-v1",
                    "runtime_only": True,
                }
                candidates.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entry),
                        source="raw-vocal-independent-fusion",
                        raw_time=peak_time,
                        spans=(),
                        confidence=min(raw_identity_score, peak_strength),
                        identity_support=(
                            "supported" if raw_occurrence_evidence is not None else "unavailable"
                        ),
                        sequence_support=(
                            "supported" if raw_occurrence_evidence is not None else "unavailable"
                        ),
                        acoustic_support="supported",
                        direct_onset_support="supported",
                        acoustic_strength=peak_strength,
                        acoustic_onset_time=peak_time,
                        acoustic_source_artifact=artifact,
                        acoustic_evidence_producer="raw-asr-vocal-fusion",
                        acoustic_evidence_kind="vocal-onset",
                        acoustic_evidence_independent=True,
                        direct_onset_time=peak_time,
                        direct_onset_source_artifact=artifact,
                        direct_onset_evidence_producer="raw-asr-vocal-fusion",
                        direct_onset_evidence_kind="vocal-onset",
                        direct_onset_evidence_independent=False,
                        occurrence_evidence=raw_occurrence_evidence,
                        independent_content_identity=raw_occurrence_evidence is not None,
                        current=False,
                        source_artifact=artifact,
                    )
                )

        whisperx_row = runtime_whisperx_row_evidence_for_entry(report, entries, index)
        if isinstance(whisperx_row, dict) and vocal_onset_features is not None and fusion_audio_revision:
            try:
                whisperx_forced_time = float(whisperx_row.get("first_char_time"))
                whisperx_row_score = float(whisperx_row.get("score"))
            except (TypeError, ValueError):
                whisperx_forced_time = math.nan
                whisperx_row_score = 0.0
            whisperx_peak = _strong_vocal_onset_near(vocal_onset_features, whisperx_forced_time)
            if whisperx_peak is not None and whisperx_row_score >= TRUSTED_ALIGNMENT_SCORE:
                whisperx_occurrence_evidence = candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="whisperx-vocal-fusion",
                    occurrence_binding=whisperx_row.get("occurrence_binding"),
                    upstream_evidence_revision=whisperx_row.get("evidence_revision"),
                )
                peak_time, peak_strength, peak_prominence = whisperx_peak
                artifact = {
                    "producer": "whisperx-vocal-fusion",
                    "whisperx_row_revision": whisperx_row.get("evidence_revision"),
                    "whisperx_forced_time": round(whisperx_forced_time, 6),
                    "score": round(whisperx_row_score, 6),
                    "vocal_onset_time": round(peak_time, 6),
                    "vocal_onset_strength": round(peak_strength, 8),
                    "vocal_onset_prominence": round(peak_prominence, 8),
                    "audio_sha256": fusion_audio_revision,
                    "feature_revision": "vocal-consonant-onset-v1",
                    "runtime_only": True,
                }
                candidates.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entry),
                        source="whisperx-vocal-independent-fusion",
                        raw_time=whisperx_forced_time,
                        spans=(),
                        confidence=min(whisperx_row_score, peak_strength),
                        identity_support="supported",
                        sequence_support="supported",
                        acoustic_support="supported",
                        direct_onset_support="supported",
                        acoustic_strength=peak_strength,
                        acoustic_onset_time=peak_time,
                        acoustic_source_artifact=artifact,
                        acoustic_evidence_producer="whisperx-vocal-fusion",
                        acoustic_evidence_kind="vocal-onset",
                        acoustic_evidence_independent=True,
                        direct_onset_time=peak_time,
                        direct_onset_source_artifact=artifact,
                        direct_onset_evidence_producer="whisperx-vocal-fusion",
                        direct_onset_evidence_kind="cross-backend-onset",
                        direct_onset_evidence_independent=True,
                        occurrence_evidence=whisperx_occurrence_evidence,
                        independent_content_identity=whisperx_occurrence_evidence is not None,
                        current=False,
                        source_artifact=artifact,
                    )
                )

        local_whisper_identity = runtime_local_whisper_evidence_for_entry(
            report, entries, index
        )
        local_whisper_vocal = _local_whisper_vocal_fusion_candidate(
            entry_index=index,
            entry=entry,
            local_evidence=local_whisper_identity,
            vocal_onset_features=vocal_onset_features,
            audio_revision=fusion_audio_revision,
            occurrence_evidence=(
                candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="local-whisper-vocal-fusion",
                    occurrence_binding=local_whisper_identity.get("occurrence_binding"),
                    upstream_evidence_revision=local_whisper_identity.get("evidence_revision"),
                )
                if isinstance(local_whisper_identity, dict) else None
            ),
        )
        if local_whisper_vocal is not None:
            candidates.append(local_whisper_vocal)

        hypothesis_rows = raw_assignment.get("alignment_hypotheses")
        if isinstance(hypothesis_rows, list):
            for hypothesis in hypothesis_rows:
                if not isinstance(hypothesis, dict):
                    continue
                raw_time = hypothesis.get("raw_candidate_time")
                if not isinstance(raw_time, (int, float)) or not math.isfinite(float(raw_time)):
                    continue
                source = str(hypothesis.get("source") or "precentral-hypothesis")
                if source == "ctc-opening-local-retry-identity":
                    # Identity-only evidence is consumed by occurrence-level
                    # recurrence producers; it must never become a timing
                    # candidate merely because it contains helper spans.
                    continue
                if source == CTC_WIDE_REGION_FALLBACK_SOURCE:
                    # Gated against the finished candidate set below.
                    continue
                independent_identity_bound = False
                independent_acoustic_bound = False
                independent_direct_bound = False
                acoustic_producer: str | None = None
                direct_producer: str | None = None
                composite_identity_revision: str | None = None
                opening_vocal_fusion_bound = False
                raw_vocal_occurrence_bound = False
                acoustic_kind = "generic-acoustic"
                direct_kind = "generic-onset"
                if source == "global-acoustic-recurrence":
                    independent_identity_bound = recurrence_hypothesis_provenance_is_valid(
                        hypothesis, entry, evidence_audio_path
                    )
                    if not independent_identity_bound:
                        continue
                    independent_acoustic_bound = True
                    independent_direct_bound = True
                    acoustic_producer = direct_producer = "global-recurrence-search"
                    acoustic_kind = direct_kind = "recurrence-onset"
                elif source == "mutual-acoustic-recurrence":
                    independent_identity_bound = mutual_recurrence_hypothesis_provenance_is_valid(
                        hypothesis,
                        entry,
                        evidence_audio_path,
                        assignments,
                        entries,
                        index,
                    )
                    if not independent_identity_bound:
                        continue
                    independent_acoustic_bound = True
                    independent_direct_bound = True
                    acoustic_producer = direct_producer = "mutual-recurrence-search"
                    acoustic_kind = direct_kind = "recurrence-onset"
                elif source == "ctc-phonetic-onset-fusion":
                    if not phonetic_onset_fusion_provenance_is_valid(
                        hypothesis, raw_assignment, evidence_audio_path
                    ):
                        continue
                    # Stage-4 R1-2 resolution: keep the validated hypothesis as
                    # diagnostics, but remove this formerly-dead producer from
                    # Central timing authority so cleanup cannot move timestamps.
                    continue
                elif source == "ctc-vocal-onset-confirmed":
                    if not vocal_onset_confirmation_provenance_is_valid(
                        hypothesis, raw_assignment, evidence_audio_path,
                        recomputed_features=vocal_onset_features,
                        evidence_audio_sha256=fusion_audio_revision,
                    ):
                        continue
                    independent_acoustic_bound = True
                    independent_direct_bound = True
                    acoustic_producer = "consonant-onset-detector"
                    acoustic_kind = "vocal-onset"
                    direct_producer = "consonant-onset-detector"
                    direct_kind = "vocal-onset"
                    opening_vocal_fusion_bound = ctc_opening_vocal_fusion_is_valid(
                        hypothesis, raw_assignment, evidence_audio_path,
                        vocal_provenance_validated=True,
                        recomputed_features=vocal_onset_features,
                        evidence_audio_sha256=fusion_audio_revision,
                    )
                    if isinstance(raw_identity, dict):
                        try:
                            raw_identity_time = float(raw_identity.get("raw_time"))
                            raw_identity_score = float(raw_identity.get("raw_score"))
                            hypothesis_onset = float(hypothesis.get("direct_onset_time"))
                        except (TypeError, ValueError):
                            raw_identity_time = math.nan
                            raw_identity_score = 0.0
                            hypothesis_onset = math.nan
                        raw_vocal_occurrence_bound = bool(
                            math.isfinite(raw_identity_time)
                            and math.isfinite(hypothesis_onset)
                            and raw_identity_score >= RAW_CTC_CONSENSUS_MIN_SCORE
                            and abs(raw_identity_time - hypothesis_onset)
                            <= RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
                        )
                        if raw_vocal_occurrence_bound:
                            composite_identity_revision = str(
                                raw_identity.get("evidence_revision") or ""
                            )
                elif source == "ctc-vocal-activity-reentry":
                    if not vocal_activity_reentry_provenance_is_valid(
                        hypothesis, raw_assignment, evidence_audio_path
                    ):
                        continue
                    independent_direct_bound = True
                    direct_producer = "vocal-activity-detector"
                    direct_kind = "vocal-activity-reentry"
                elif source == "trusted-tail-release-onset":
                    if not trusted_tail_release_provenance_is_valid(
                        hypothesis,
                        evidence_audio_path,
                        assignments,
                        index,
                    ):
                        continue
                    # trusted-tail-release-diagnostic-only: a post-tail energy
                    # cluster is not phonetic proof of the next lyric's first
                    # mora. Preserve the serialized hypothesis for diagnostics,
                    # but do not manufacture an always-invalid Central candidate.
                    continue
                spans_value = _timing_spans_from_value(hypothesis.get("spans"))
                acoustic_strength = hypothesis.get("acoustic_strength")
                acoustic_onset_time = hypothesis.get("acoustic_onset_time")
                direct_onset_time = hypothesis.get("direct_onset_time")
                source_artifact = {
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                    "stage": hypothesis.get("stage"),
                    "parent": hypothesis.get("parent"),
                    "window": hypothesis.get("window"),
                    "mode": hypothesis.get("mode"),
                    "raw_content_identity_revision": composite_identity_revision,
                    "authoritative": False,
                }
                candidates.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entry),
                        source=source,
                        raw_time=float(raw_time),
                        spans=spans_value,
                        confidence=float(hypothesis.get("confidence", 0.0) or 0.0),
                        identity_support=str(hypothesis.get("identity_support") or "unavailable"),  # type: ignore[arg-type]
                        sequence_support=str(hypothesis.get("sequence_support") or "unavailable"),  # type: ignore[arg-type]
                        acoustic_support=str(hypothesis.get("acoustic_support") or "unavailable"),  # type: ignore[arg-type]
                        direct_onset_support=(
                            "supported"
                            if independent_direct_bound and isinstance(direct_onset_time, (int, float))
                            else "unavailable"
                        ),
                        acoustic_strength=(float(acoustic_strength) if isinstance(acoustic_strength, (int, float)) else None),
                        acoustic_onset_time=(float(acoustic_onset_time) if isinstance(acoustic_onset_time, (int, float)) else None),
                        acoustic_source_artifact=source_artifact,
                        acoustic_evidence_producer=acoustic_producer,
                        acoustic_evidence_kind=acoustic_kind,
                        acoustic_evidence_independent=independent_acoustic_bound,
                        direct_onset_time=(
                            float(direct_onset_time)
                            if independent_direct_bound and isinstance(direct_onset_time, (int, float))
                            else None
                        ),
                        direct_onset_source_artifact=(
                            source_artifact
                            if independent_direct_bound and isinstance(direct_onset_time, (int, float))
                            else None
                        ),
                        direct_onset_evidence_producer=direct_producer,
                        direct_onset_evidence_kind=direct_kind,
                        direct_onset_evidence_independent=independent_direct_bound,
                        independent_content_identity=independent_identity_bound,
                        current=False,
                        right_edge_pileup=bool(hypothesis.get("right_edge_pileup")),
                        window_truncated=bool(hypothesis.get("window_truncated")),
                        source_artifact=source_artifact,
                    )
                )

                if source == "ctc-vocal-onset-confirmed" and opening_vocal_fusion_bound:
                    fusion_artifact = {
                        **source_artifact,
                        "producer": "ctc-opening-vocal-fusion",
                        "ctc_geometry_revision": _timing_content_digest({
                            "spans": hypothesis.get("spans"),
                            "assignment_time": raw_assignment.get("timestamp"),
                        }),
                    }
                    candidates.append(
                        make_timing_candidate(
                            entry_index=index,
                            entry_text=entry_sung_text(entry),
                            source="ctc-opening-vocal-fusion",
                            raw_time=float(raw_time),
                            spans=spans_value,
                            confidence=float(hypothesis.get("confidence", 0.0) or 0.0),
                            identity_support="supported",
                            sequence_support="supported",
                            acoustic_support="supported",
                            direct_onset_support="supported",
                            acoustic_strength=(float(acoustic_strength) if isinstance(acoustic_strength, (int, float)) else None),
                            acoustic_onset_time=(float(acoustic_onset_time) if isinstance(acoustic_onset_time, (int, float)) else None),
                            acoustic_source_artifact=fusion_artifact,
                            acoustic_evidence_producer="consonant-onset-detector",
                            acoustic_evidence_kind="vocal-onset",
                            acoustic_evidence_independent=True,
                            direct_onset_time=(float(direct_onset_time) if isinstance(direct_onset_time, (int, float)) else None),
                            direct_onset_source_artifact=fusion_artifact,
                            # CTC opening geometry belongs to the candidate itself.
                            # Combining it with an independent vocal peak improves
                            # current-line opening support, but does not make that
                            # peak independently lyric-specific.  Keep this generic
                            # so it can never cross UNKNOWN/VIOLATION ownership.
                            direct_onset_evidence_producer="consonant-onset-detector",
                            direct_onset_evidence_kind="vocal-onset",
                            direct_onset_evidence_independent=True,
                            independent_content_identity=False,
                            current=False,
                            right_edge_pileup=bool(hypothesis.get("right_edge_pileup")),
                            window_truncated=bool(hypothesis.get("window_truncated")),
                            source_artifact=fusion_artifact,
                        )
                    )
                if source == "ctc-vocal-onset-confirmed" and raw_vocal_occurrence_bound:
                    raw_occurrence_evidence = candidate_occurrence_evidence_from_binding(
                        entries,
                        index,
                        producer="raw-asr-vocal-fusion",
                        occurrence_binding=raw_identity.get("occurrence_binding"),
                        upstream_evidence_revision=raw_identity.get("evidence_revision"),
                    )
                    raw_fusion_artifact = {
                        **source_artifact,
                        "producer": "raw-asr-vocal-fusion",
                        "raw_content_identity_revision": composite_identity_revision,
                        "raw_time": round(float(raw_identity.get("raw_time")), 6),
                        "raw_score": round(float(raw_identity.get("raw_score")), 6),
                    }
                    candidates.append(
                        make_timing_candidate(
                            entry_index=index,
                            entry_text=entry_sung_text(entry),
                            source="ctc-raw-vocal-independent-fusion",
                            raw_time=float(raw_time),
                            spans=spans_value,
                            confidence=min(
                                float(hypothesis.get("confidence", 0.0) or 0.0),
                                float(raw_identity.get("raw_score", 0.0) or 0.0),
                            ),
                            identity_support="supported",
                            sequence_support="supported",
                            acoustic_support="supported",
                            direct_onset_support="supported",
                            acoustic_strength=(float(acoustic_strength) if isinstance(acoustic_strength, (int, float)) else None),
                            acoustic_onset_time=(float(acoustic_onset_time) if isinstance(acoustic_onset_time, (int, float)) else None),
                            acoustic_source_artifact=raw_fusion_artifact,
                            acoustic_evidence_producer="consonant-onset-detector",
                            acoustic_evidence_kind="vocal-onset",
                            acoustic_evidence_independent=True,
                            direct_onset_time=(float(direct_onset_time) if isinstance(direct_onset_time, (int, float)) else None),
                            direct_onset_source_artifact=raw_fusion_artifact,
                            direct_onset_evidence_producer="raw-asr-vocal-fusion",
                            direct_onset_evidence_kind="vocal-onset",
                            direct_onset_evidence_independent=False,
                            occurrence_evidence=raw_occurrence_evidence,
                            independent_content_identity=raw_occurrence_evidence is not None,
                            current=False,
                            right_edge_pileup=bool(hypothesis.get("right_edge_pileup")),
                            window_truncated=bool(hypothesis.get("window_truncated")),
                            source_artifact=raw_fusion_artifact,
                        )
                    )

        first_token_candidates = raw_assignment.get("ctc_first_token_candidates")
        if isinstance(first_token_candidates, list):
            for raw_candidate in first_token_candidates:
                if not isinstance(raw_candidate, dict):
                    continue
                try:
                    raw_time = float(raw_candidate.get("time"))
                    raw_score = float(raw_candidate.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                # A first-token posterior is identity evidence, not by itself
                # a complete onset proof.  Its one-token span therefore enters
                # normalization and ownership gating but cannot become valid
                # until another producer contributes coherent prefix/acoustic
                # support for the same written time.
                posterior_span = TimingTokenSpan(
                    canonicalize_lrc_time(raw_time).seconds,
                    canonicalize_lrc_time(raw_time).seconds + HOP_SECONDS,
                    raw_score,
                    "",
                )
                candidates.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entry),
                        source="ctc-first-token-posterior",
                        raw_time=raw_time,
                        spans=(posterior_span,),
                        confidence=raw_score,
                        identity_support="supported",
                        sequence_support="supported",
                        source_artifact=raw_candidate,
                    )
                )

        legacy_candidates = raw_assignment.get("candidates")
        if isinstance(legacy_candidates, list):
            for legacy in legacy_candidates:
                if not _active_legacy_candidate(legacy):
                    continue
                raw_time = legacy.get("time")
                if not isinstance(raw_time, (int, float)):
                    continue
                source = str(legacy.get("source") or "legacy-candidate")
                direct_onset = False
                identity = source in {"raw_asr", "whisperx_forced_first", "whisperx_word"}
                candidates.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entry),
                        source=source.replace("_", "-"),
                        raw_time=float(raw_time),
                        confidence=float(legacy.get("score", 0.0) or 0.0),
                        identity_support="supported" if identity else "unavailable",
                        sequence_support="supported",
                        acoustic_support="unavailable",
                        direct_onset_support="supported" if direct_onset else "unavailable",
                        direct_onset_time=float(raw_time) if direct_onset else None,
                        direct_onset_source_artifact=legacy if direct_onset else None,
                        source_artifact=legacy,
                    )
                )

        if (report.get("segment_skeleton_status") or {}).get("status") == "SEGMENT_CURRENT":
            from segment_peer_outlier import exclude_uncorroborated_whisperx_fusion
            candidates, excluded = exclude_uncorroborated_whisperx_fusion(candidates)
            if excluded:
                raw_assignment["segment_peer_fusion_exclusions"] = [
                    {"source":x.source, "time":x.written_time.seconds,
                     "reason":"uncorroborated-distant-fused-onset"}
                    for x in excluded
                ]
        unique = {candidate.candidate_id: candidate for candidate in candidates}
        candidate_sets.append(tuple(unique[key] for key in sorted(unique)))

    ordinary_candidate_sets = _augment_local_ctc_candidates_with_runtime_backend_consensus(
        tuple(candidate_sets), report, entries
    )
    independent_tail_evidence = runtime_independent_tail_evidence(report, len(entries))
    independent_tail_payload = independent_tail_evidence_payload(independent_tail_evidence)
    fusion_source_counts = collections.Counter(
        candidate.source
        for candidates in ordinary_candidate_sets
        for candidate in candidates
        if candidate.source in {
            "raw-vocal-independent-fusion",
            "whisperx-vocal-independent-fusion",
            "ctc-opening-vocal-fusion",
            "ctc-raw-vocal-independent-fusion",
            "ctc-local-raw-independent-consensus",
            "ctc-local-whisperx-independent-consensus",
        }
    )
    hypothesis_count = sum(
        len(item.get("alignment_hypotheses", []))
        for item in assignments
        if isinstance(item, dict) and isinstance(item.get("alignment_hypotheses"), list)
    )
    diagnostics = {
        "ctc_prefix_probes": [],
        "ctc_prefix_probe_count": 0,
        "legacy_final_guard_changes_discarded": [],
        "legacy_final_guard_change_count": 0,
        "independent_tail_evidence": independent_tail_payload,
        "runtime_evidence_fusion_candidate_counts": dict(sorted(fusion_source_counts.items())),
        "alignment_hypothesis_layer": {
            "status": "active",
            "authoritative_precentral_mutation": False,
            "hypothesis_count": hypothesis_count,
            "central_commit_only": True,
        },
        **inactive_legacy_timing_diagnostics(),
    }
    capability = build_bounded_retry_capability(report, audio_path, args)
    tail_evidence_context = _tail_evidence_context_from_report(
        report, entries=entries, capability=capability
    )
    if tail_evidence_context is not None:
        report[_RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY] = tail_evidence_context
    independent_tail_evidence = runtime_independent_tail_evidence(
        report, len(entries), tail_evidence_context
    )
    independent_tail_payload = independent_tail_evidence_payload(independent_tail_evidence)
    if retry_helper_cache is None:
        retry_helper_cache = {}
    try:
        retry_timeout_seconds = float(getattr(args, "crossline_retry_timeout", 180.0))
    except (TypeError, ValueError):
        retry_timeout_seconds = 180.0
    if not math.isfinite(retry_timeout_seconds) or retry_timeout_seconds <= 0.0:
        retry_timeout_seconds = 180.0
    retry_timing_source_policy = str(
        getattr(args, "timing_source_policy", "default") or "default"
    )
    retry_runtime_stats: dict[str, int] = {
        "cache_hits": 0,
        "cache_misses": 0,
        "helper_invocations": 0,
        "timeouts": 0,
    }
    # Bounded fixed-point retry: resolving one boundary can make the next line's
    # previous-tail ownership provable. Re-plan from the newly evaluated state,
    # but cap passes so recovery can never become an unbounded heuristic loop.
    candidate_sets = ordinary_candidate_sets
    retry_passes: list[dict[str, object]] = []
    all_requests: list[CrossLineRetryRequest] = []
    all_outcomes: list[CrossLineRetryOutcome] = []
    processed_retry_semantics: set[str] = set()
    max_retry_passes = 4
    last_epoch_revision: str | None = None
    tail_evidence_changes = 0
    for pass_index in range(1, max_retry_passes + 1):
        before_candidate_revision, before_tail_revision = _retry_semantic_state_revision(
            candidate_sets,
            independent_tail_evidence,
            tail_evidence_context,
            retry_timing_source_policy,
        )
        epoch = freeze_base_retry_epoch(
            str(report.get("backend") or "unknown"),
            report,
            candidate_sets,
            diagnostics,
            capability,
        )
        last_epoch_revision = epoch.retry_epoch_revision
        requests = plan_crossline_retry_requests(
            epoch,
            capability,
            entries,
            candidate_sets,
            duration,
            independent_tail_evidence,
            tail_evidence_context,
            retry_timing_source_policy,
        )
        if capability is None:
            novel_requests: tuple[CrossLineRetryRequest, ...] = ()
            novel_semantics: tuple[str, ...] = ()
        else:
            novel_requests, novel_semantics = _novel_crossline_retry_requests(
                requests, capability, processed_retry_semantics
            )
        progress_log(
            f"crossline retry pass {pass_index}: requests={len(requests)} "
            f"new_semantic={len(novel_requests)} cache_entries={len(retry_helper_cache)}"
        )
        if not requests:
            retry_passes.append({
                "pass": pass_index,
                "retry_epoch_revision": epoch.retry_epoch_revision,
                "request_count": 0,
                "executed_request_count": 0,
                "new_semantic_request_count": 0,
                "outcome_count": 0,
                "new_candidate_count": 0,
                "semantic_fixed_point": True,
            })
            break
        if not novel_requests:
            progress_log(
                f"crossline retry pass {pass_index}: semantic fixed point; "
                "all planned helper inputs were already evaluated"
            )
            retry_passes.append({
                "pass": pass_index,
                "retry_epoch_revision": epoch.retry_epoch_revision,
                "request_count": len(requests),
                "executed_request_count": 0,
                "new_semantic_request_count": 0,
                "outcome_count": 0,
                "new_candidate_count": 0,
                "semantic_fixed_point": True,
            })
            break
        outcomes = execute_crossline_retry_epoch(
            epoch, capability, entries, candidate_sets, duration, novel_requests,
            helper_cache=retry_helper_cache,
            timeout_seconds=retry_timeout_seconds,
            pass_index=pass_index,
            runtime_stats=retry_runtime_stats,
            independent_tail_evidence=independent_tail_evidence,
            tail_evidence_context=tail_evidence_context,
            timing_source_policy=retry_timing_source_policy,
            worker_count=DEFAULT_EVIDENCE_WORKERS,
        )
        processed_retry_semantics.update(novel_semantics)
        tail_evidence_changes = sum(
            int(merge_runtime_independent_tail_evidence(
                report, outcome.previous_tail_evidence
            ))
            for outcome in outcomes
            if outcome.previous_tail_evidence is not None
        )
        if tail_evidence_changes:
            independent_tail_evidence = runtime_independent_tail_evidence(
                report, len(entries), tail_evidence_context
            )
        augmented = [list(items) for items in candidate_sets]
        existing_ids = {item.candidate_id for items in candidate_sets for item in items}
        new_candidate_count = 0
        for outcome in outcomes:
            record_crossline_retry_identity_hypothesis(report, outcome)
            if (
                outcome.status == "candidate"
                and outcome.candidate is not None
                and outcome.candidate.candidate_id not in existing_ids
            ):
                augmented[outcome.entry_index].append(outcome.candidate)
                existing_ids.add(outcome.candidate.candidate_id)
                new_candidate_count += 1
        candidate_sets = tuple(
            tuple({item.candidate_id: item for item in items}[key] for key in sorted(
                {item.candidate_id: item for item in items}
            ))
            for items in augmented
        )
        candidate_sets = _augment_local_ctc_candidates_with_runtime_backend_consensus(
            candidate_sets, report, entries
        )
        after_candidate_revision, after_tail_revision = _retry_semantic_state_revision(
            candidate_sets,
            independent_tail_evidence,
            tail_evidence_context,
            retry_timing_source_policy,
        )
        semantic_state_changed = bool(
            before_candidate_revision != after_candidate_revision
            or before_tail_revision != after_tail_revision
        )
        all_requests.extend(novel_requests)
        all_outcomes.extend(outcomes)
        retry_passes.append({
            "pass": pass_index,
            "retry_epoch_revision": epoch.retry_epoch_revision,
            "request_count": len(requests),
            "executed_request_count": len(novel_requests),
            "new_semantic_request_count": len(novel_requests),
            "outcome_count": len(outcomes),
            "new_candidate_count": new_candidate_count,
            "previous_tail_evidence_change_count": tail_evidence_changes,
            "candidate_semantic_revision": after_candidate_revision,
            "authoritative_tail_semantic_revision": after_tail_revision,
            "semantic_fixed_point": not semantic_state_changed,
        })
        if new_candidate_count == 0 and not semantic_state_changed:
            break

    # Architectural recovery: after current-line retry reaches its bounded
    # fixed point, obtain a separate joint previous+current boundary observation
    # for every still-unresolved boundary.  The observation cannot be selected
    # directly; it must converge with an already-existing current candidate.
    joint_boundary_requests: tuple[CrossLineRetryRequest, ...] = ()
    joint_boundary_outcomes: tuple[CrossLineRetryOutcome, ...] = ()
    joint_boundary_added = 0
    if capability is not None:
        joint_epoch = freeze_base_retry_epoch(
            str(report.get("backend") or "unknown"),
            report,
            candidate_sets,
            diagnostics,
            capability,
        )
        joint_boundary_requests = plan_joint_boundary_retry_requests(
            joint_epoch, capability, entries, candidate_sets, duration,
            independent_tail_evidence,
            tail_evidence_context,
            retry_timing_source_policy,
        )
        progress_log(
            f"joint boundary retry: requests={len(joint_boundary_requests)} "
            f"cache_entries={len(retry_helper_cache)}"
        )
        if joint_boundary_requests:
            joint_boundary_outcomes = execute_crossline_retry_epoch(
                joint_epoch, capability, entries, candidate_sets, duration,
                joint_boundary_requests,
                helper_cache=retry_helper_cache,
                timeout_seconds=retry_timeout_seconds,
                pass_index="joint",
                runtime_stats=retry_runtime_stats,
                independent_tail_evidence=independent_tail_evidence,
                tail_evidence_context=tail_evidence_context,
                timing_source_policy=retry_timing_source_policy,
                worker_count=DEFAULT_EVIDENCE_WORKERS,
            )
            joint_tail_evidence_changes = sum(
                int(merge_runtime_independent_tail_evidence(
                    report, outcome.previous_tail_evidence
                ))
                for outcome in joint_boundary_outcomes
                if outcome.previous_tail_evidence is not None
            )
            if joint_tail_evidence_changes:
                independent_tail_evidence = runtime_independent_tail_evidence(
                    report, len(entries), tail_evidence_context
                )
            before_ids = {
                item.candidate_id for items in candidate_sets for item in items
            }
            candidate_sets = _augment_candidates_with_joint_boundary_consensus(
                candidate_sets, joint_boundary_outcomes, entries
            )
            after_ids = {
                item.candidate_id for items in candidate_sets for item in items
            }
            joint_boundary_added = len(after_ids - before_ids)
            if joint_boundary_added:
                candidate_sets = _augment_local_ctc_candidates_with_runtime_backend_consensus(
                    candidate_sets, report, entries
                )
            progress_log(
                f"joint boundary retry: observations={len(joint_boundary_outcomes)} "
                f"new_consensus_candidates={joint_boundary_added}"
            )

    # The bounded CTC retry/joint phase is complete for this collection epoch.
    # Park resident MMS_FA workers on CPU before local whisper.cpp/WhisperX work.
    _park_persistent_helper_mode("ctc", reason="bounded-ctc-evidence-complete")

    # v14.6: the CTC graph is now rich enough that the remaining ceiling is
    # usually lack of an *independent* lyric-specific opening/occurrence witness.
    # Run whisper.cpp + WhisperX only for those unresolved/content-untrusted/review
    # rows.  The local observation is never selectable by itself; central may use
    # it only by fusing its onset/identity into an already-existing non-Whisper
    # candidate within the unchanged 0.18 s cross-backend tolerance.
    local_whisper_capability = build_local_whisper_capability(evidence_audio_path, args)
    local_whisper_requests: tuple[LocalWhisperRequest, ...] = ()
    local_whisper_outcomes: tuple[LocalWhisperOutcome, ...] = ()
    local_whisper_added = 0
    local_whisper_stats: dict[str, int] = {
        "cache_hits": 0, "cache_misses": 0, "helper_invocations": 0,
    }
    if local_whisper_capability is not None:
        local_whisper_requests = plan_local_whisper_requests(
            report, local_whisper_capability, entries, candidate_sets, duration,
            independent_tail_evidence,
            tail_evidence_context,
            retry_timing_source_policy,
        )
        # Fixed-point recollection may change upstream candidate IDs and
        # generate hundreds of distinct optional Whisper context windows.
        # Preserve all real cache hits and cover the song with at most 32 new
        # incremental probes per recollection. The first collection remains
        # fully faithful. Deferred probes confer NO timing or identity proof.
        from local_whisper_incremental_budget import budget_incremental_whisper_requests
        local_whisper_requests, deferred_new_probes = budget_incremental_whisper_requests(
            local_whisper_requests,
            retry_helper_cache,
            lambda request: _local_whisper_cache_key(request, local_whisper_capability),
            maximum_new=32,
        )
        local_whisper_stats["budget_deferred"] = deferred_new_probes
        progress_log(
            f"local Whisper evidence: requests={len(local_whisper_requests)} "
            f"cache_entries={len(retry_helper_cache)} "
            f"deferred_untrusted_new={deferred_new_probes}"
        )
        local_whisper_outcomes = run_local_whisper_requests(
            local_whisper_requests, local_whisper_capability, entries, args,
            cache=retry_helper_cache, stats=local_whisper_stats,
        )
        bind_runtime_local_whisper_evidence(
            report, entries, local_whisper_capability,
            local_whisper_requests, local_whisper_outcomes,
        )
        before_ids = {
            item.candidate_id for items in candidate_sets for item in items
        }
        candidate_sets = _augment_candidates_with_local_whisper_vocal_consensus(
            candidate_sets, report, entries, vocal_onset_features, fusion_audio_revision
        )
        candidate_sets = _augment_candidates_with_local_whisper_consensus(
            candidate_sets, report, entries
        )
        after_ids = {
            item.candidate_id for items in candidate_sets for item in items
        }
        local_whisper_added = len(after_ids - before_ids)
        progress_log(
            f"local Whisper evidence: trusted_observations="
            f"{sum(item.status == 'evidence' for item in local_whisper_outcomes)} "
            f"new_consensus_candidates={local_whisper_added}"
        )

    # v14.7: same-model global/local Whisper contradictions are not independent
    # votes.  A weaker whole-song occurrence may be excluded only when the
    # bounded observation is sequence-bracketed and independently corroborated
    # by CTC at the competing onset.
    candidate_sets, hypothesis_falsifications = (
        _apply_context_conditioned_whisper_hypothesis_arbitration(
            candidate_sets, report, entries
        )
    )
    if hypothesis_falsifications:
        progress_log(
            "hypothesis arbitration: falsified_global_whisper_candidates="
            f"{sum(len(item.get('falsified_candidate_ids', [])) for item in hypothesis_falsifications)} "
            f"entries={','.join(str(item.get('entry')) for item in hypothesis_falsifications)}"
        )

    candidate_sets, wide_region_rows = _attach_ctc_wide_region_candidates(
        candidate_sets, report, entries, hypothesis_falsifications
    )

    diagnostics = {
        **diagnostics,
        "ctc_wide_region_fallback": {
            "source": CTC_WIDE_REGION_FALLBACK_SOURCE,
            "region_radius": CTC_WIDE_REGION_RADIUS,
            "rows": wide_region_rows,
            "central_commit_only": True,
        },
        "independent_tail_evidence": independent_tail_evidence_payload(
            runtime_independent_tail_evidence(report, len(entries))
        ),
        "independent_tail_authority": _tail_authority_diagnostics(
            candidate_sets,
            runtime_independent_tail_evidence(report, len(entries)),
            tail_evidence_context,
            retry_timing_source_policy,
        ),
        "crossline_retry": {
            "retry_epoch_revision": last_epoch_revision,
            "capability_revision": capability.capability_revision if capability else None,
            "capability": _retry_capability_diagnostic_payload(capability),
            "requests": [_retry_diagnostic_payload(item) for item in all_requests],
            "outcomes": [_retry_diagnostic_payload(item) for item in all_outcomes],
            "invocation_count": len(all_outcomes),
            "previous_tail_evidence_count": sum(
                item.previous_tail_evidence is not None for item in all_outcomes
            ),
            "helper_invocation_count": retry_runtime_stats["helper_invocations"],
            "helper_cache_hit_count": retry_runtime_stats["cache_hits"],
            "helper_cache_miss_count": retry_runtime_stats["cache_misses"],
            "helper_cache_entry_count": len(retry_helper_cache),
            "helper_timeout_count": retry_runtime_stats["timeouts"],
            "helper_timeout_seconds": retry_timeout_seconds,
            "processed_semantic_request_count": len(processed_retry_semantics),
            "semantic_fixed_point": bool(retry_passes and retry_passes[-1].get("semantic_fixed_point")),
            "recursive_planning": True,
            "max_passes": max_retry_passes,
            "passes": retry_passes,
        },
        "joint_boundary_retry": {
            "capability_revision": capability.capability_revision if capability else None,
            "requests": [_retry_diagnostic_payload(item) for item in joint_boundary_requests],
            "outcomes": [_retry_diagnostic_payload(item) for item in joint_boundary_outcomes],
            "request_count": len(joint_boundary_requests),
            "outcome_count": len(joint_boundary_outcomes),
            "previous_tail_evidence_count": sum(
                item.previous_tail_evidence is not None
                for item in joint_boundary_outcomes
            ),
            "strong_observation_count": sum(
                1
                for item in joint_boundary_outcomes
                if (
                    item.status == "candidate"
                    and item.candidate is not None
                    and item.candidate.source == "ctc-joint-boundary-observation"
                )
            ),
            "new_consensus_candidate_count": joint_boundary_added,
            "previous_tail_evidence_change_count": (
                joint_tail_evidence_changes
                if joint_boundary_requests
                else 0
            ),
            "central_commit_only": True,
            "observation_selectable_directly": False,
        },
        "local_whisper_evidence": {
            "available": local_whisper_capability is not None,
            "capability_revision": (
                local_whisper_capability.capability_revision
                if local_whisper_capability is not None else None
            ),
            "protocol_revision": (
                local_whisper_capability.protocol_revision
                if local_whisper_capability is not None else None
            ),
            "request_count": len(local_whisper_requests),
            "outcome_count": len(local_whisper_outcomes),
            "trusted_observation_count": sum(
                item.status == "evidence" for item in local_whisper_outcomes
            ),
            "invalid_count": sum(
                item.status == "invalid" for item in local_whisper_outcomes
            ),
            "error_count": sum(
                item.status == "error" for item in local_whisper_outcomes
            ),
            "new_consensus_candidate_count": local_whisper_added,
            "cache_hit_count": local_whisper_stats["cache_hits"],
            "cache_miss_count": local_whisper_stats["cache_misses"],
            "helper_invocation_count": local_whisper_stats["helper_invocations"],
            "entries": [
                {
                    "entry": item.entry_index + 1,
                    "status": item.status,
                    "reason": item.reason,
                    "onset_time": (
                        round(item.onset_time, 6) if item.onset_time is not None else None
                    ),
                    "score": (round(item.score, 6) if item.score is not None else None),
                    "evidence_revision": item.evidence_revision,
                    "occurrence_binding_mode": (
                        item.occurrence_binding.get("mode")
                        if isinstance(item.occurrence_binding, dict) else None
                    ),
                }
                for item in local_whisper_outcomes
            ],
            "central_commit_only": True,
            "observation_selectable_directly": False,
            "whispercpp_and_whisperx_count_as_one_lineage": True,
            "global_raw_and_local_whisper_count_as_one_family": True,
        },
        "hypothesis_arbitration": {
            "mode": "context-conditioned-falsification",
            "falsification_count": sum(
                len(item.get("falsified_candidate_ids", []))
                for item in hypothesis_falsifications
            ),
            "entry_count": len(hypothesis_falsifications),
            "entries": list(hypothesis_falsifications),
            "majority_vote_allowed": False,
            "central_commit_only": True,
        },
    }
    return candidate_sets, diagnostics
