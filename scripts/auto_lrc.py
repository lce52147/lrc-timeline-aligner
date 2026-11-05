#!/usr/bin/env python3
"""Generate rough LRC timing for FLAC files.

This is a deterministic v0 backend. It decodes audio with ffmpeg, estimates
energy/onset structure, maps lyric lines onto the song duration, then snaps line
starts toward nearby acoustic onsets. It is intentionally written so a later
forced-alignment backend can replace the timestamp planner without changing the
drag/drop entry points.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import contextlib
import concurrent.futures
import copy
import hashlib
import itertools
import json
import math
import os
import queue
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from ctc_confidence import ctc_alignment_confidence
import arbiter as timing_arbiter
import lrc_timestamp_pairs
import runtime_paths
import file_digest
import model_cache
from model_cache import ModelCacheMiss
from numeral_reading import numerals_for_alignment
import segment_realign
import segment_skeleton
from multimodel_consensus import (
    PROTOCOL_REVISION as MULTIMODEL_PROTOCOL_REVISION,
    active_model_specs,
    build_windows,
    choose_consensus_overrides,
    consensus_report,
    model_disagreement_review,
    model_identity,
)
from export_alignment_audit import build_rows as build_audit_rows
from export_alignment_audit import write_anchor_template
from export_alignment_audit import write_markdown as write_audit_markdown
import asr_backends
from asr_backends import (
    SEGMENT_GATE_OVERRIDE_ATTR,
    SEGMENT_GATE_STATUS_ATTR,
    WHISPERCPP_LOCAL_BATCH_SIZE,
    _LOCAL_WHISPER_PROTOCOL_REVISION,
    _RUNTIME_WHISPERX_ALIGNMENT_METADATA_KEY,
    _aligned_exact_phonetic_target_matches,
    _aligned_exact_target_matches,
    _aligned_target_prefix_matches,
    _apply_multimodel_consensus_runtime,
    _attach_bounded_retry_runtime_capability,
    _bounded_retry_capability_semantic_payload,
    _build_whisperx_runtime_provenance,
    _hf_hub_fingerprint,
    _local_whisper_capability_semantic_payload,
    _occurrence_binding_time_bounds,
    _parse_whispercpp_json,
    _probe_whisperx_model_identity,
    _run_whispercpp_wav,
    _run_whispercpp_wav_batch,
    _select_unique_aligned_target_match,
    _select_unique_aligned_target_prefix_match,
    _set_runtime_arg,
    _whispercpp_cache_identity,
    _whispercpp_cache_lookup,
    _whispercpp_cache_store,
    _whispercpp_multifile_supported,
    _whispercpp_runtime_artifact_hashes,
    _whisperx_alignment_metadata,
    _whisperx_model_identity,
    asr_transcript,
    build_bounded_retry_capability,
    build_local_whisper_capability,
    carry_bounded_retry_capability_seed,
    choose_alignment_candidate,
    choose_onset_consensus_time,
    create_bounded_retry_capability_seed,
    ctc_alignment_audio_path,
    decode_temp_wav,
    decode_temp_wav_window,
    estimated_segment_text_time,
    fused_candidate_report,
    inactive_legacy_timing_diagnostics,
    is_alignment_untrusted,
    is_alignment_unusable,
    low_confidence_run_start,
    lyric_alignment_windows,
    lyric_segment_score,
    match_whisper_segments,
    needs_suppressed_retry,
    raw_asr_is_fallback_eligible,
    run_ctc_alignment,
    run_japanese_ctc_alignment,
    run_whispercpp,
    run_whisperx_alignment,
    run_whisperx_best_candidate,
    run_whisperx_candidate,
    segment_char_time,
    should_prefer_ctc_over_review_exploded_hybrid,
    suffix_char_time,
    try_ctc_candidate,
    try_whispercpp_diagnostic_candidate,
    try_whisperx_candidate,
    untrusted_alignment_reason,
    unusable_alignment_reason,
    whisperx_segments,
)
from acoustic_hypotheses import (
    AUDIO_ADVISORY_REVIEW_FLAGS,
    MUTUAL_RECURRENCE_ANCHOR_RADIUS_SECONDS,
    MUTUAL_RECURRENCE_DISTINCT_SECONDS,
    MUTUAL_RECURRENCE_FEATURE_REVISION,
    MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN,
    MUTUAL_RECURRENCE_MIN_SIMILARITY,
    MUTUAL_RECURRENCE_PEER_RADIUS_SECONDS,
    PHONETIC_ONSET_FEATURE_REVISION,
    PHONETIC_ONSET_MIN_POSTERIOR,
    PHONETIC_ONSET_MIN_PROMINENCE,
    PHONETIC_ONSET_MIN_STRENGTH,
    PHONETIC_ONSET_RADIUS_SECONDS,
    RECURRENCE_FEATURE_REVISION,
    RECURRENCE_MIN_DISTINCT_MARGIN,
    RECURRENCE_MIN_SIMILARITY,
    TAIL_RELEASE_CLUSTER_BREAK_SECONDS,
    TAIL_RELEASE_CONTEXT_SECONDS,
    TAIL_RELEASE_FEATURE_REVISION,
    TAIL_RELEASE_ONSET_MIN_PROMINENCE,
    TAIL_RELEASE_ONSET_MIN_STRENGTH,
    VOCAL_ACTIVITY_REENTRY_FEATURE_REVISION,
    VOCAL_ACTIVITY_REENTRY_MIN_GAP_SECONDS,
    VOCAL_ONSET_CONFIRM_MIN_PROMINENCE,
    VOCAL_ONSET_CONFIRM_MIN_STRENGTH,
    VOCAL_ONSET_CONFIRM_RADIUS_SECONDS,
    VOCAL_ONSET_RAW_ORDER_MIN_CONTRADICTIONS,
    VOCAL_ONSET_RAW_ORDER_MIN_SCORE,
    VOCAL_ONSET_RAW_ORDER_TOLERANCE_SECONDS,
    _best_mutual_recurrence_pair,
    _bound_local_occurrence_identity,
    _candidate_seed_rows,
    _candidate_spans_from_assignment,
    _first_post_tail_cluster_onset,
    _local_onset_candidates,
    _nearest_local_maximum,
    _project_recurrence_spans_from_geometry,
    _raw_asr_order_anchors,
    _raw_asr_order_contradictions,
    _self_bound_hypothesis_record,
    _strong_vocal_onset_near,
    _trusted_bounds_for_entry,
    _trusted_recurrence_reference_geometry,
    _vocal_activity_reentry_for_time,
    apply_acoustic_late_onset_refinement,
    apply_global_acoustic_recurrence_hypotheses,
    apply_mutual_acoustic_recurrence_hypotheses,
    apply_phonetic_onset_fusion_hypotheses,
    apply_post_score_audio_validation,
    apply_short_duplicate_acoustic_recovery,
    apply_trusted_tail_release_onset_hypotheses,
    apply_vocal_activity_reentry_hypotheses,
    apply_vocal_onset_confirmation_hypotheses,
    apply_whisperx_lyric_refinement,
    assert_report_timestamp_equality,
    blend_tail_repair_with_forced,
    ctc_opening_vocal_fusion_is_valid,
    duplicate_lyric_offset_consensus,
    first_late_onset,
    first_sustained_onset,
    has_repeated_asr_tail,
    low_confidence_tail_start,
    mutual_recurrence_hypothesis_provenance_is_valid,
    phonetic_onset_fusion_provenance_is_valid,
    project_fun_asr_exact_positive,
    recurrence_hypothesis_provenance_is_valid,
    repair_intro_hallucination,
    repair_long_segment_hallucinations,
    repair_tail_hallucination,
    score_line_timing_candidates,
    sync_scratch_assignment_timestamps,
    trusted_tail_release_provenance_is_valid,
    vocal_activity_reentry_provenance_is_valid,
    vocal_onset_confirmation_provenance_is_valid,
)
from ctc_refinement import (
    COLLAPSED_REGION_WINDOW_MAX_SECONDS,
    CTC_LOCAL_WINDOW_SHADOW_RADII,
    CTC_WIDE_REGION_PEER_RADIUS,
    CTC_WIDE_REGION_RADIUS,
    RAW_CTC_CONSENSUS_MIN_SCORE,
    SEGMENT_REALIGN_ANCHOR_CTC_SCORE,
    SEGMENT_REALIGN_DENSE_TOKEN_GAP_SECONDS,
    SEGMENT_REALIGN_MAX_RUNS,
    SEGMENT_REALIGN_MAX_WINDOW_SECONDS,
    SEGMENT_REALIGN_MIN_ANCHOR_SHARE,
    SEGMENT_REALIGN_MIN_SHIFT_SECONDS,
    SEGMENT_REALIGN_MIN_WINDOW_SECONDS,
    SEGMENT_REALIGN_RAW_AGREE_SECONDS,
    SEGMENT_REALIGN_REVISION,
    SEGMENT_REALIGN_SOURCE,
    SEGMENT_REALIGN_STAGE,
    SEGMENT_REALIGN_STRAY_TOKEN_GAP_SECONDS,
    _BOUNDED_RETRY_MODEL_IDENTITY,
    _invalidate_timing_trust_after_timestamp_mutation,
    _legacy_refinement_row_has_actionable_flags,
    _segment_realign_observations,
    annotate_ctc_boundary_evidence,
    apply_ctc_acoustic_backtrack,
    apply_ctc_boundary_validation_stage,
    apply_ctc_crossline_initial_recovery,
    apply_ctc_forward_supported_prefix_recovery,
    apply_ctc_hybrid_window_realign,
    apply_ctc_local_fusion_to_whisperx,
    apply_ctc_local_window_realign,
    apply_ctc_micro_refinement_to_whisperx,
    apply_ctc_weak_prefix_recovery,
    apply_ctc_zero_gap_boundary_realign,
    apply_final_ctc_timing_guard,
    apply_hybrid_ctc_opening_onset_backtrack,
    apply_segment_realign,
    apply_vocal_onset_tiebreak,
    apply_whisperx_acoustic_boundary_refinement,
    clear_stale_ctc_boundary_review_state,
    ctc_forward_supported_prefix_candidate,
    ctc_path_fracture_evidence,
    ctc_prefix_boundary_evidence,
    ctc_right_edge_pileup_evidence,
    duplicate_block_offset_evidence,
    first_local_onset,
    flag_unresolved_ctc_boundary_issues,
    flag_unresolved_ctc_disagreements,
    flag_unresolved_raw_ctc_disagreements,
    has_ctc_compressed_prefix,
    lyric_prefix_probe_text,
    refresh_ctc_confidence_diagnostics,
    run_ctc_local_window_shadows,
    run_ctc_prefix_probe,
    run_segment_realign_stage,
    segment_realign_mode,
    segment_realign_onset,
    segment_realign_row_status,
    should_accept_ctc_boundary_realign,
    should_accept_zero_gap_boundary_realign,
)
import report_trust
from report_trust import (
    ACTIONABLE_SUSPICIOUS_FLAGS,
    ACTIONABLE_TIMING_FAILURES,
    COLLAPSE_SEGMENT_SCORE,
    CTC_BOUNDARY_REVIEW_FLAGS,
    FINAL_LINE_TRUST_CLASSIFICATION,
    FinalLineTrustDecision,
    NON_ACTIONABLE_VERIFICATION_FAILURES,
    SILENT_RISK_REVIEW_FLAG,
    WEAK_COMPANION_CODES,
    WEAK_REVIEW_CODES,
    WEAK_REVIEW_POLICY_KEY,
    _RUNTIME_BACKEND_PARTICIPATION_KEY,
    _RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY,
    _RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY,
    _RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICT_DETAILS_KEY,
    _RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY,
    _RUNTIME_POSTCENTRAL_CTC_TAIL_PRODUCER_CONTEXT_KEY,
    _RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY,
    _RUNTIME_POSTCENTRAL_HUBERT_BOUNDARY_OBSERVATION_KEY,
    _RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY,
    _RUNTIME_POSTCENTRAL_HUBERT_DUPLICATE_RESOLUTION_KEY,
    _RUNTIME_RAW_CONTENT_IDENTITY_KEY,
    _RUNTIME_RAW_CTC_CONSENSUS_KEY,
    _RUNTIME_RAW_TIMING_DRIFT_FALSIFICATIONS_KEY,
    _RUNTIME_WHISPERX_CTC_CONSENSUS_KEY,
    _RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY,
    _RUNTIME_WHISPERX_ROW_EVIDENCE_KEY,
    _RUNTIME_WHISPERX_TEMP_PATHS_KEY,
    _UNRESOLVED_EVIDENCE_CATEGORY_VALUES,
    _UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY,
    _actionable_suspicious_severity_counts,
    _backend_participation,
    _backend_participation_projection,
    _central_summary_row_flags,
    _ctc_disagreement_severity,
    _merge_backend_participation,
    _merge_suspicious_severity,
    _new_backend_participation,
    _raw_disagreement_severity,
    _record_backend_available,
    _record_backend_participation_failure,
    _record_backend_unavailable,
    _review_reason_classes_from_trust,
    _runtime_peer_review_only_conflicts,
    _silent_risk_rule_applies,
    _suspicious_entry_requires_review,
    _suspicious_row_has_independent_high_severity_owner,
    _suspicious_row_requires_review,
    _timing_failure_requires_review,
    _validate_backend_provenance_projection,
    _validate_unresolved_evidence_requirements_projection,
    _weak_review_demotes,
    add_alignment_diagnostics,
    apply_anchor_hints,
    apply_central_summary_to_report,
    apply_weak_review_policy,
    assert_report_review_contract,
    assignment_content_is_trusted,
    assignment_has_timing,
    assignment_score,
    assignment_timing_is_trusted,
    candidate_selection_quality,
    candidate_summary,
    candidate_supplies_independent_content_identity,
    central_evaluated_quality,
    dedupe_suspicious_alignments,
    final_line_trust,
    finalize_review_summary,
    find_alignment_collapse_runs,
    freeze_generation_diagnostics,
    report_has_verified_central_commit,
    report_quality,
    runtime_content_identity_projection,
    runtime_independent_content_identity_sources_for_entry,
    runtime_local_whisper_evidence_for_entry,
    runtime_raw_content_identity_for_entry,
    runtime_whisperx_row_evidence_for_entry,
    self_check_flags_require_review,
    severity_rank,
    summarize_central_timing_state,
    thaw_generation_diagnostics,
    timing_report_evaluation_projection,
    timing_report_evaluation_revision,
    update_report_confidence_metrics,
    write_alignment_report,
)
from timing_core import (
    BOUNDARY_GRADE_DIRECT_ONSET_KINDS,
    BackendContentIdentityObservation,
    BackendSelection,
    BaseRetryEpoch,
    BoundedRetryCapability,
    COLLAPSED_PATH_FALLBACK_MIN_SHIFT_SECONDS,
    COLLAPSED_PATH_FALLBACK_SOURCE,
    CORROBORATED_REALIGNMENT_AGREEMENT_SECONDS,
    CORROBORATED_REALIGNMENT_RAW_ONSET_SECONDS,
    CORROBORATED_REALIGNMENT_SOURCES,
    CTC_COLLAPSED_LINE_MAX_SECONDS,
    CTC_COLLAPSED_LINE_MIN_SECONDS,
    CTC_SEVERE_INTERNAL_GAP_SECONDS,
    CTC_WIDE_REGION_FALLBACK_SOURCE,
    CandidateAcousticEvidence,
    CandidateEvaluation,
    CandidateOccurrenceEvidence,
    CandidateOnsetEvidence,
    CandidateRegionEvidence,
    CentralTrustSummary,
    CrossLineRetryOutcome,
    CrossLineRetryRequest,
    EntryTimingFindings,
    EvaluatedBackendBundle,
    EvaluatedBackendTiming,
    EvidenceState,
    FinalTimingState,
    IndependentCurrentOnsetEvidence,
    IndependentTailEvidence,
    LocalWhisperCapability,
    LocalWhisperOutcome,
    LocalWhisperRequest,
    NormalizedTimingEvidence,
    ObservationClusterSummary,
    OwnershipRelationEvidence,
    OwnershipState,
    PREFIX_ACOUSTIC_ASSOCIATION_SECONDS,
    PeerGraphPolicy,
    PostcentralCtcTailProducerContext,
    RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
    RAW_ORDER_ANCHOR_SOURCE,
    RecoveryProvenance,
    RetryCandidateReference,
    RuntimeTailEvidence,
    STRICT_DIRECT_ONSET_KINDS,
    STRICT_DIRECT_ONSET_PRODUCERS,
    TRUSTED_ALIGNMENT_SCORE,
    TailEvidenceContext,
    TailEvidenceProvenance,
    TailEvidenceScope,
    TailObservationBinding,
    TailObservationSet,
    TailRegionEvidence,
    TailValidity,
    TimingCandidate,
    TimingDecision,
    TimingFinding,
    TimingInvariantAudit,
    WhisperXRuntimeProvenance,
    _CANDIDATE_OCCURRENCE_BINDING_KEYS,
    _CANDIDATE_OCCURRENCE_DIRECT_PRODUCERS,
    _CANDIDATE_OCCURRENCE_MODES,
    _CANDIDATE_REGIONS,
    _GENERATION_DIAGNOSTIC_LOCATOR_KEYS,
    _REVIEWER_VALIDITY_DIAGNOSTICS_KEY,
    _ROMAJI_VOWELS,
    _RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY,
    _RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY,
    _RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY,
    _RUNTIME_WHISPERX_TAIL_SCOPES_KEY,
    _SERIALIZED_SECOND_SCALE,
    _TAIL_BACKEND_LINEAGE_FAMILIES,
    _TAIL_CONTEXT_VALIDITY,
    _WHISPERX_HELPER_PROTOCOL_REVISION,
    _active_legacy_candidate,
    _analyze_candidate_regions_uncached,
    _authoritative_tail_observations,
    _bind_independent_tail_to_candidate,
    _boundary_grade_opening_can_cross_tail,
    _bounded_independent_consensus_winner,
    _candidate_confidence,
    _candidate_occurrence_binding_keys_are_canonical,
    _candidate_occurrence_context_indexes,
    _candidate_occurrence_evidence_payload,
    _candidate_occurrence_evidence_projection,
    _candidate_requires_occurrence_evidence,
    _candidate_support_families,
    _candidate_vector,
    _canonical_occurrence_binding_json,
    _collapsed_path_fallback,
    _contradicts_raw_order,
    _corroborated_realignment_fallback,
    _ctc_wide_region_promotions,
    _current_tail_lineage_relation,
    _current_timing_candidate,
    _entry_identity_is_unique,
    _entry_occurrence_binding,
    _external_tail_current_lineage_relation,
    _finite_unit_confidence,
    _generation_diagnostics_semantic_projection,
    _identity_assignment_observation,
    _lineage_families_from_labels,
    _make_tail_provenance,
    _minimal_unique_sequence_context,
    _normalize_tail_observations,
    _occurrence_binding_has_strong_context,
    _opening_mora_proof_end,
    _percentile,
    _raw_order_anchors,
    _record_current_replacement,
    _reviewer_agreement_count,
    _reviewer_validity_endorsement,
    _runtime_occurrence_binding_is_valid,
    _runtime_tail_observation_is_current,
    _runtime_tail_observation_map,
    _semantically_equivalent_valid_winner,
    _serialized_delta_matches,
    _serialized_microseconds,
    _serialized_seconds_equal,
    _single_direct_onset_family,
    _sparse_sequence_fingerprint_is_unique,
    _sparse_unique_sequence_anchor_set,
    _supported_run_metrics,
    _tail_backend_family,
    _tail_context_is_valid,
    _tail_context_is_valid_uncached,
    _tail_context_revision_values,
    _tail_context_scopes_from_report,
    _tail_contexts_are_valid,
    _tail_evidence_context_from_report,
    _tail_helper_revision,
    _tail_provenance_is_valid,
    _tail_provenance_matches_context,
    _tail_provenance_payload,
    _tail_region_result,
    _whisperx_runtime_provenance_is_valid,
    _whisperx_runtime_provenance_payload,
    analyze_candidate_regions,
    apply_central_timing_source_policy,
    assignment_ctc_path_collapsed,
    audit_final_timing_decisions,
    bind_reviewer_validity_diagnostics,
    build_final_timing_state,
    candidate_dominates,
    candidate_has_bound_acoustic_evidence,
    candidate_has_bound_direct_onset_evidence,
    candidate_has_boundary_grade_direct_onset_evidence,
    candidate_has_independent_backend_identity,
    candidate_occurrence_evidence_from_binding,
    candidate_occurrence_evidence_is_valid,
    classify_candidate_ownership,
    ctc_collapsed_region_flags,
    ctc_current_out_of_raw_order,
    ctc_line_extent_collapsed,
    ctc_spans_collapsed,
    current_ctc_path_collapsed,
    current_ctc_path_incoherent,
    derive_reliable_tail,
    evaluate_timing_candidate,
    generation_diagnostics_semantic_revision,
    global_recurrence_has_independent_corroboration,
    load_reviewer_validity_evidence,
    make_timing_candidate,
    normalize_timing_candidate,
    raw_onset_seconds_by_row,
    raw_order_contradicting_candidate_ids,
    record_alignment_hypothesis,
    resolve_reliable_tail,
    reviewer_validity_evidence_from_diagnostics,
    runtime_independent_tail_evidence,
    select_timing_decision,
)
from lyrics_io import (
    ANCHOR_ENTRY_COMMENT_RE,
    AnchorHint,
    AsrSegment,
    LINE_START_LEAD_MS,
    LRC_ALGORITHM_REVISION,
    LRC_LINE_RE,
    LRC_TIMESTAMP_RE,
    LyricDocument,
    LyricEntry,
    SOURCE_CLOCK_METADATA_KEYS,
    SOURCE_GENERATOR_METADATA_KEYS,
    SOURCE_TIME_PRESERVING_TIMING_SOURCES,
    _BACKEND_PROVENANCE_PROJECTION_KEY,
    _RUNTIME_LYRIC_ENTRIES_KEY,
    _is_single_cjk_or_kana_term,
    _previous_line_end_cs,
    active_bounds,
    annotate_display_romaji,
    apply_line_start_lead,
    assert_lyric_identity_contract,
    audio_track_labels,
    build_lyric_preflight,
    checked_lrc_candidates,
    checked_lrc_timing_hint,
    compact_latin,
    entry_sung_text,
    find_anchor_hints,
    find_lyrics,
    first_forced_char_time,
    format_lrc_time,
    has_ambiguous_lyric_prefix,
    has_short_duplicate_context,
    infer_spoken_language,
    inference_mode_metadata,
    is_instrumental_marker,
    japanese_romaji,
    load_anchor_hints,
    load_lyrics,
    lrc_time_key,
    lyric_anchor_profile,
    lyric_identity_revision,
    lyric_line_script,
    match_anchor_entries,
    merge_adjacent_translation_lines,
    normalize_match_text,
    optional_kakasi_converter,
    output_metadata_lines,
    plan_timestamps,
    ratio_percent,
    raw_asr_lyric_coverage,
    remove_generated_title_cards,
    remove_instrumental_markers,
    repeated_leading_term,
    resolve_start_lead_ms,
    romaji_morae,
    segment_guided_times,
    sequence_ratio,
    snap_to_onsets,
    source_timestamps,
    split_group_text,
    strict_review_failure_reason,
    strip_lyric_timestamps,
    text_weight,
    weighted_rough_times,
    write_lrc,
    write_review_anchor_template,
    write_review_audit,
)
from audio_features import (
    AudioFeatures,
    DEFAULT_VOCAL_CACHE_DIR,
    FINAL_SELF_CHECK_BELOW_MEDIAN_DB,
    FINAL_SELF_CHECK_WINDOW_SECONDS,
    FRAME_SECONDS,
    HOP_SECONDS,
    RECURRENCE_MAX_TEMPLATE_SECONDS,
    RECURRENCE_MIN_TEMPLATE_SECONDS,
    SAMPLE_RATE,
    VOCAL_SEPARATOR_PROTOCOL_REVISION,
    VocalStemLevels,
    _AUDIO_FEATURE_CACHE_MAX_ENTRIES,
    _audio_feature_cache,
    _audio_feature_cache_key,
    _audio_feature_cache_store,
    _clone_audio_features,
    _demucs_separator_identity,
    _demucs_vocal_stem_valid,
    _demucs_write_error,
    _mel_filterbank,
    _recurrence_segment,
    _search_recurrence_start,
    _stage_demucs_input,
    acoustic_pitchclass_recurrence_features,
    acoustic_recurrence_features,
    acoustic_recurrence_similarity,
    analyze_audio,
    analyze_vocal_onsets,
    decode_audio,
    frame_audio,
    merge_segments,
    nearest_local_peak,
    nearest_onset_evidence,
    normalize,
    prepare_vocal_ctc_audio,
    probe_duration,
    smooth,
    vocal_cache_identity,
    vocal_onset_features,
    vocal_stem_levels,
)
from runtime_base import (
    AlignmentEvidenceRejected,
    CanonicalWrittenTime,
    DEFAULT_EVIDENCE_WORKERS,
    DEFAULT_LOCK_TIMEOUT_SECONDS,
    DEFAULT_MODEL_CACHE_DIR,
    GPU_COMPUTE_LOCK_DIR,
    LrcError,
    PROGRESS_START,
    SegmentSkeletonRollback,
    WHISPERCPP_GPU_LOCK_PATH,
    WHISPERCPP_PARK_WATCH_SECONDS,
    WhisperXEvidenceBundleError,
    _ALGORITHM_SOURCE_MODULES,
    _BOUNDED_RETRY_HELPER_PROTOCOL,
    _GPU_COMPUTE_STATE,
    _GPU_COMPUTE_STATE_LOCK,
    _LOADED_SOURCE_SHA256,
    _MANAGED_TEMP_DIRS,
    _MANAGED_TEMP_DIRS_LOCK,
    _PARK_WATCHER,
    _PARK_WATCHER_LOCK,
    _PARK_WATCHER_STOP,
    _PERSISTENT_HELPER_DISABLED,
    _PERSISTENT_HELPER_LOGGED,
    _PERSISTENT_HELPER_POOLS,
    _PERSISTENT_HELPER_POOLS_LOCK,
    _PersistentHelperClient,
    _PersistentHelperPool,
    _WHISPERCPP_HELD_HERE,
    _algorithm_source_files,
    _algorithm_source_sha256,
    _apply_model_cache_flags,
    _attach_raw_asr_status,
    _bounded_retry_protocol_revision,
    _canonical_json_digest,
    _cleanup_managed_temp_dirs_since,
    _close_persistent_helper_mode,
    _close_persistent_helper_pools,
    _ctc_model_blob_sha256,
    _effective_raw_asr_paths,
    _effective_whisperx_python,
    _ensure_whispercpp_park_watcher,
    _enter_gpu_compute,
    _evidence_worker_count,
    _exclusive_file_lock,
    _gpu_coordination_enabled,
    _leave_gpu_compute,
    _lock_timeout_seconds,
    _managed_temp_dir,
    _managed_temp_snapshot,
    _model_cache_enabled,
    _model_cache_key,
    _model_cache_only,
    _model_cache_only_active,
    _model_cache_path,
    _model_cache_policy,
    _model_cache_read,
    _model_cache_write,
    _other_process_runs_whispercpp,
    _other_process_runs_whispercpp_confirmed,
    _park_before_whispercpp_handoff,
    _park_persistent_helper_mode,
    _path_sha256,
    _persistent_helper_worker_path,
    _raw_asr_backend_status,
    _raw_asr_warning,
    _refuse_model_launch_in_cache_only,
    _remove_stale_output,
    _run_model_helper_command,
    _run_model_helper_command_now,
    _stable_file_sha256,
    _timing_content_digest,
    _try_file_lock,
    _wait_for_other_gpu_compute,
    _whispercpp_park_watch,
    _whisperx_python_available,
    algorithm_source_text,
    canonicalize_lrc_time,
    configure_stdio,
    default_ctc_python,
    default_ctc_ready,
    default_whisper_cli,
    default_whisper_model,
    default_whisperx_python,
    default_whisperx_ready,
    demucs_cache_key_lock,
    gpu_compute_phase,
    progress_change_log,
    progress_change_summary,
    progress_log,
    progress_report_changes,
    run_command,
    segment_postcentral_nonmonotonic,
    validated_loaded_source_sha256,
    whispercpp_gpu_lock,
)
from runtime_base import TimingTokenSpan, _timing_spans_from_value


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_REPORT_DIR = runtime_paths.reports_dir()
DEFAULT_PROBE_DIR = runtime_paths.probes_dir()
MODEL_CACHE_SCHEMA = model_cache.MODEL_CACHE_SCHEMA


_UNRESOLVED_EVIDENCE_REQUIREMENTS_SCHEMA_VERSION = 1
_SAME_OCCURRENCE_BOUNDARY_EQUIVALENCE_KEY = "same_occurrence_boundary_equivalence"
_BACKEND_PROVENANCE_PROJECTION_SCHEMA_VERSION = 1
_BACKEND_PEER_SIMULATION_SCHEMA_VERSION = 2


RAW_TIMING_DRIFT_MAX_MEDIAN_DELTA_SECONDS = 2.0
CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS = 0.75
CROSSLINE_RETRY_MAX_WINDOW_EXPANSIONS = 3
LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS = 1.50
LOCAL_WHISPER_MAX_WINDOW_SECONDS = 30.0


def progress_quality_summary(label: str, report: dict[str, object]) -> None:
    authority_trusted = report.get(
        "overall_trusted_percent",
        report.get("trusted_percent", report.get("matched_percent")),
    )
    timing_trusted = report.get("timing_trusted_percent", authority_trusted)
    progress_log(
        f"{label}: "
        f"timing_trusted={timing_trusted}, "
        f"authority_trusted={authority_trusted}, "
        f"review={report.get('review_required_count')}, "
        f"unverified={report.get('unverified_count')}, "
        f"low={report.get('low_confidence_count')}, "
        f"quality={candidate_selection_quality(report):.1f}"
    )


CTC_WIDE_REGION_PEER_AGREEMENT_SECONDS = 0.3
CTC_WIDE_REGION_SUPPORT_SECONDS = 0.1
CTC_WIDE_REGION_VETO_SECONDS = 0.5


class _AudioApi:
    """Singleton namespace handed to segment producers as ``audio_api``.

    Attribute reads resolve against this module's globals at call time so that
    test patches and stage-timing wrappers stay visible; unknown names raise
    AttributeError so ``hasattr`` probes keep their meaning.
    """

    def __getattr__(self, name):
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(name) from None


_AUDIO_API = _AudioApi()


def _central_preview_generation_candidates(
    candidates: list[dict[str, object]],
    selected_backend: str,
    selected_timestamps: list[float],
    selected_report: dict[str, object],
) -> list[dict[str, object]]:
    """Replace same-backend/same-timestamp diagnostic clones with the refined seed."""
    selected_payload = {
        "backend": selected_backend,
        "timestamps": list(selected_timestamps),
        "report": selected_report,
        "error": None,
    }
    result: list[dict[str, object]] = []
    inserted = False
    for candidate in candidates:
        if (
            not candidate.get("error")
            and candidate.get("backend") == selected_backend
            and candidate.get("timestamps") == selected_timestamps
        ):
            if not inserted:
                result.append(selected_payload)
                inserted = True
            continue
        result.append(candidate)
    if not inserted:
        result.append(selected_payload)
    return result

def bind_ctc_generation_candidate_evidence(
    generation_candidates: list[dict[str, object]],
    entries: list[LyricEntry],
    whisperx_evidence_candidates: list[tuple[list[float], dict[str, object], str]],
) -> None:
    """Attach existing runtime evidence to each valid CTC generation candidate."""
    for candidate in generation_candidates:
        report = candidate.get("report")
        timestamps = candidate.get("timestamps")
        if (
            candidate.get("backend") != "ctc"
            or candidate.get("error")
            or not isinstance(report, dict)
            or not isinstance(timestamps, list)
            or len(timestamps) != len(entries)
        ):
            continue
        if whisperx_evidence_candidates:
            bind_runtime_whisperx_row_evidence(
                report, entries, whisperx_evidence_candidates
            )
            bind_runtime_whisperx_ctc_consensus(
                report, entries, timestamps, whisperx_evidence_candidates
            )
        bind_runtime_ctc_assignment_tail_fallback(report, entries)


def run_auto_backend_competition(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
    anchor_path: Path | None,
) -> tuple[list[float], dict[str, object], str, BackendSelection]:
    candidates: list[dict[str, object]] = []
    diagnostic_candidates: list[dict[str, object]] = []
    raw_diagnostic_report: dict[str, object] | None = None
    raw_diagnostic_timestamps: list[float] | None = None
    raw_diagnostic_segments: list[AsrSegment] | None = None
    shared_whisper_wav: Path | None = None
    whisperx_evidence_candidates: list[tuple[list[float], dict[str, object], str]] = []
    participation = _new_backend_participation()
    raw_asr_status = _raw_asr_backend_status(args)
    if bool(raw_asr_status["available"]):
        _record_backend_available(participation, "whispercpp_raw")
    else:
        _record_backend_unavailable(
            participation, "whispercpp_raw", _raw_asr_warning(raw_asr_status)
        )
    if not bool(raw_asr_status["available"]):
        if _model_cache_only(args):
            raise ModelCacheMiss(
                "model-cache-miss:raw-ASR-backend-unavailable:"
                + ",".join(str(item) for item in raw_asr_status["missing_components"])
            )
        progress_log("WARNING: " + _raw_asr_warning(raw_asr_status))

    # R2L performance-only scheduling: Raw ASR does not consume CTC output,
    # while CTC local-window realignment *does* consume the Raw diagnostic.
    # Run Raw first so the base CTC pass and its local-window follow-up can share
    # one resident CTC helper lifecycle. Candidate/evidence logic is unchanged.
    if bool(raw_asr_status["available"]):
        raw_asr_status["attempted"] = True
        _close_persistent_helper_mode("ctc", reason="before-global-raw-asr")
        progress_log("running raw ASR diagnostic (pre-CTC schedule)")
        try:
            shared_whisper_wav = decode_temp_wav(audio_path)
        except (LrcError, OSError):
            shared_whisper_wav = None
        raw_timestamps, raw_report, raw_error, raw_diagnostic_segments = try_whispercpp_diagnostic_candidate(
            audio_path,
            entries,
            duration,
            args,
            predecoded_wav=shared_whisper_wav,
        )
        if not raw_error and isinstance(raw_report, dict):
            raw_diagnostic_report = raw_report
            raw_diagnostic_timestamps = raw_timestamps
            progress_log(
                "raw ASR diagnostic done: "
                f"trusted={raw_report.get('trusted_percent', raw_report.get('matched_percent'))}, "
                f"review={raw_report.get('review_required_count')}, "
                f"quality={candidate_selection_quality(raw_report):.1f}"
            )
        else:
            raw_asr_status["failure_kind"] = "execution-failed"
            raw_asr_status["error"] = str(raw_error or "unknown error")
            raw_asr_status["fallback"] = "empty-raw-asr-events"
            progress_log("WARNING: " + _raw_asr_warning(raw_asr_status))
        diagnostic_candidates.append(
            {
                "backend": "whispercpp_raw",
                "timestamps": raw_timestamps,
                "report": raw_report,
                "error": raw_error,
            }
        )

    raw_coverage = raw_asr_lyric_coverage(
        entries, raw_diagnostic_segments if raw_diagnostic_report is not None else None
    )
    ctc_ready = default_ctc_ready()
    ctc_cache_miss_reason: str | None = None
    if not ctc_ready:
        reason = "ctc-python-or-helper-unavailable"
        _record_backend_unavailable(participation, "ctc", reason)
        ctc_cache_miss_reason = reason if _model_cache_only(args) else None
        progress_log("WARNING: CTC backend unavailable; fallback=skip-ctc-candidate")
    else:
        _record_backend_available(participation, "ctc")
        progress_log("running CTC candidate")
        # Synthetic caller/test arguments may be immutable objects; only
        # argparse-style mutable namespaces carry optional raw ASR context.
        raw_skeleton_events = tuple(
            (float(s.start),float(s.end),str(s.text))
            for s in (raw_diagnostic_segments or ())
        )
        try:
            setattr(args, '_skeleton_raw_asr_events', raw_skeleton_events)
        except (AttributeError, TypeError):
            pass
        ctc_timestamps, ctc_report, ctc_error = try_ctc_candidate(audio_path, entries, duration, args)
        if not ctc_error and isinstance(ctc_report, dict):
            progress_log(
                "CTC done: "
                f"device={ctc_report.get('ctc_device')}, "
                f"audio={ctc_report.get('ctc_audio_source')}, "
                f"low={ctc_report.get('ctc_low_score_count')}, "
                f"review={ctc_report.get('review_required_count')}, "
                f"quality={candidate_selection_quality(ctc_report):.1f}"
            )
            progress_report_changes("CTC acoustic backtrack candidate generated", ctc_report, "ctc_acoustic_backtracks")
            initial_boundary_stages = ctc_report.get("ctc_boundary_validation_stages")
            if isinstance(initial_boundary_stages, list) and initial_boundary_stages:
                initial_boundary = initial_boundary_stages[-1]
                if isinstance(initial_boundary, dict):
                    progress_change_log(
                        "CTC legacy forward weak-prefix evidence (inactive)",
                        initial_boundary.get("forward_supported_prefix_recoveries"),
                    )
                    progress_change_log(
                        "CTC legacy unresolved-boundary evidence (inactive)",
                        initial_boundary.get("unresolved_boundaries"),
                    )
        else:
            progress_log(f"CTC failed: {ctc_error}")
        candidates.append(
            {
                "backend": "ctc",
                "timestamps": ctc_timestamps,
                "report": ctc_report,
                "error": ctc_error,
            }
        )

    if bool(raw_asr_status["available"]):
        ctc_candidate = next(
            (candidate for candidate in candidates if candidate.get("backend") == "ctc" and not candidate.get("error")),
            None,
        )
        if (
            ctc_candidate is not None
            and raw_diagnostic_report is not None
            and isinstance(ctc_candidate.get("report"), dict)
            and isinstance(ctc_candidate.get("timestamps"), list)
        ):
            if ctc_candidate["report"].get("segment_skeleton_status", {}).get("status") == "SEGMENT_CURRENT":
                window_timestamps, window_report, local_window_changes = (
                    list(ctc_candidate["timestamps"]), ctc_candidate["report"], []
                )
            else:
                local_window_cache: dict[tuple[int, int, str, str], list[object]] = {}
                window_timestamps, window_report, local_window_changes = apply_ctc_local_window_realign(
                ctc_alignment_audio_path(ctc_candidate["report"], audio_path),
                entries,
                list(ctc_candidate["timestamps"]),
                ctc_candidate["report"],
                raw_diagnostic_report,
                duration,
                args,
                window_cache=local_window_cache,
            )
                run_ctc_local_window_shadows(
                    ctc_alignment_audio_path(ctc_candidate["report"], audio_path),
                    entries,
                    list(ctc_candidate["timestamps"]),
                    window_report,
                    raw_diagnostic_report,
                    duration,
                    args,
                    local_window_cache,
                )
                run_segment_realign_stage(
                    ctc_alignment_audio_path(ctc_candidate["report"], audio_path),
                    entries,
                    list(ctc_candidate["timestamps"]),
                    window_report,
                    raw_diagnostic_report,
                    duration,
                    args,
                    local_window_cache,
                )
            progress_change_log("CTC local window candidate generated", local_window_changes)
            window_report.update(inactive_legacy_timing_diagnostics())
            ctc_candidate["timestamps"] = window_timestamps
            ctc_candidate["report"] = window_report
            progress_log("checking raw/CTC disagreements")
            flag_unresolved_raw_ctc_disagreements(
                window_timestamps, window_report, raw_diagnostic_report, entries
            )
            bind_runtime_raw_content_identity(window_report, entries, raw_diagnostic_report)
            if (
                raw_asr_is_fallback_eligible(raw_diagnostic_report)
                and raw_diagnostic_timestamps is not None
            ):
                candidates.append(
                    {
                        "backend": "whispercpp",
                        "timestamps": raw_diagnostic_timestamps,
                        "report": raw_diagnostic_report,
                        "error": None,
                        "fallback_reason": "raw-asr-beats-conflicted-ctc",
                    }
                )

    whisperx_ready = False
    if bool(raw_asr_status["available"]):
        whisperx_ready = _whisperx_python_available(args)
        if whisperx_ready:
            _record_backend_available(participation, "whisperx")
        else:
            _record_backend_unavailable(
                participation, "whisperx", "whisperx-python-unavailable"
            )
    if ctc_cache_miss_reason is not None:
        raise ModelCacheMiss("model-cache-miss:" + ctc_cache_miss_reason)
    if whisperx_ready:
        _park_persistent_helper_mode("ctc", reason="before-global-whisperx")
        try:
            progress_log(
                "running WhisperX candidate "
                f"(reuse_wav={'yes' if shared_whisper_wav is not None else 'no'}, "
                f"reuse_raw={'yes' if raw_diagnostic_segments is not None else 'no'})"
            )
            whisperx_timestamps, whisperx_report = run_whisperx_best_candidate(
                audio_path,
                entries,
                duration,
                args,
                predecoded_wav=shared_whisper_wav,
                normal_raw_segments=raw_diagnostic_segments,
            )
            whisperx_error = None
            progress_log(
                "WhisperX done: "
                f"trusted={whisperx_report.get('trusted_percent', whisperx_report.get('matched_percent'))}, "
                f"review={whisperx_report.get('review_required_count')}, "
                f"quality={candidate_selection_quality(whisperx_report):.1f}"
            )
        except WhisperXEvidenceBundleError as exc:
            whisperx_evidence_candidates = list(exc.evidence_candidates)
            whisperx_timestamps = []
            whisperx_report = {
                "backend": "whisperx",
                "error": str(exc),
                "evidence_candidate_count": len(whisperx_evidence_candidates),
            }
            whisperx_error = str(exc)
            progress_log(
                f"WhisperX failed globally; preserving line evidence from "
                f"{len(whisperx_evidence_candidates)} candidate(s): {whisperx_error}"
            )
        except LrcError as exc:
            whisperx_timestamps = []
            whisperx_report = {"backend": "whisperx", "error": str(exc)}
            whisperx_error = str(exc)
            progress_log(f"WhisperX failed: {whisperx_error}")
        candidates.append(
            {
                "backend": "whisperx",
                "timestamps": whisperx_timestamps,
                "report": whisperx_report,
                "error": whisperx_error,
            }
        )
        _park_persistent_helper_mode("whisperx", reason="global-whisperx-complete")

    # This is only a generation seed for optional hybrid producers.  The final
    # backend is selected later from immutable central evaluations.
    selected = next(
        (
            candidate for candidate in candidates
            if candidate.get("backend") == "whisperx" and not candidate.get("error")
        ),
        None,
    ) or choose_alignment_candidate(candidates)
    segment_ctc_seed = next(
        (candidate for candidate in candidates
         if candidate.get("backend") == "ctc"
         and not candidate.get("error")
         and isinstance(candidate.get("report"), dict)
         and candidate["report"].get("segment_skeleton_status", {}).get("status")
           == "SEGMENT_CURRENT"),
        None
    )
    if segment_ctc_seed is not None:
        selected = segment_ctc_seed
    candidate_labels = [
        f"{candidate.get('backend')}:{float(candidate.get('quality', candidate_selection_quality(candidate.get('report', {}))) or -100.0):.1f}"
        for candidate in candidates
    ]
    progress_log(
        "hybrid generation seed: "
        f"{selected.get('backend', 'unknown')} "
        f"quality={float(selected.get('quality', candidate_selection_quality(selected.get('report', {}))) or -100.0):.1f}; "
        f"candidates={','.join(candidate_labels)}"
    )
    selected_report = copy.deepcopy(selected["report"])
    assert isinstance(selected_report, dict)
    _attach_raw_asr_status(selected_report, raw_asr_status)
    selected_timestamps = list(selected["timestamps"])
    selected_backend = str(selected.get("backend", selected_report.get("backend", "auto")))
    if selected_backend == "ctc":
        if segment_ctc_seed is not None:
            # Keep REAL independent WhisperX identity/onset evidence without
            # letting the global WhisperX clock compete as another skeleton.
            # The existing runtime binder verifies each non-borrowed row and
            # its occurrence binding; no synthetic witnesses are introduced.
            for whisper_candidate in candidates:
                if (whisper_candidate.get("backend") == "whisperx"
                        and not whisper_candidate.get("error")
                        and isinstance(whisper_candidate.get("report"), dict)
                        and isinstance(whisper_candidate.get("timestamps"), list)
                        and len(whisper_candidate["timestamps"]) == len(entries)):
                    whisperx_evidence_candidates.append((
                        list(whisper_candidate["timestamps"]),
                        whisper_candidate["report"],
                        "complete-independent-whisperx",
                    ))
                    break
        if whisperx_evidence_candidates:
            bind_runtime_whisperx_row_evidence(
                selected_report, entries, whisperx_evidence_candidates
            )
            bind_runtime_whisperx_ctc_consensus(
                selected_report,
                entries,
                selected_timestamps,
                whisperx_evidence_candidates,
            )
            row_summary = selected_report.get("whisperx_row_evidence")
            consensus_summary = selected_report.get("whisperx_ctc_onset_consensus")
            progress_log(
                "WhisperX line evidence preserved: "
                f"rows={row_summary.get('candidate_count', 0) if isinstance(row_summary, dict) else 0}, "
                f"ctc_consensus={consensus_summary.get('candidate_count', 0) if isinstance(consensus_summary, dict) else 0}"
            )
        bind_runtime_ctc_assignment_tail_fallback(selected_report, entries)
    selected_quality = float(selected.get("quality", candidate_selection_quality(selected_report)) or -100.0)
    base_selected_report = copy.deepcopy(selected_report)
    base_selected_timestamps = list(selected_timestamps)
    base_selected_backend = selected_backend
    base_selected_quality = selected_quality
    ctc_candidate: dict[str, object] | None = None

    if selected_backend == "whisperx":
        base_ctc_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate.get("backend") == "ctc"
                and not candidate.get("error")
                and isinstance(candidate.get("report"), dict)
            ),
            None,
        )
        # Hybrid construction is allowed to mutate its own scratch candidate,
        # but never the preserved global CTC backend hypothesis in `candidates`.
        ctc_candidate = copy.deepcopy(base_ctc_candidate) if base_ctc_candidate is not None else None
        if ctc_candidate is not None:
            raw_candidate = next(
                (
                    candidate
                    for candidate in diagnostic_candidates
                    if candidate.get("backend") == "whispercpp_raw"
                    and not candidate.get("error")
                    and isinstance(candidate.get("report"), dict)
                ),
                None,
            )
            ctc_window_timestamps, ctc_window_report, hybrid_window_changes = apply_ctc_hybrid_window_realign(
                ctc_alignment_audio_path(ctc_candidate["report"], audio_path),
                entries,
                list(ctc_candidate["timestamps"]),
                ctc_candidate["report"],
                selected_report,
                duration,
                args,
            )
            progress_change_log("CTC hybrid window realign applied", hybrid_window_changes)
            ctc_window_report.update(inactive_legacy_timing_diagnostics())
            ctc_candidate["timestamps"] = ctc_window_timestamps
            ctc_candidate["report"] = ctc_window_report
            fused_timestamps, fused_report, fusion_changes = apply_ctc_local_fusion_to_whisperx(
                selected_timestamps,
                selected_report,
                ctc_candidate["report"],
                duration,
                raw_candidate["report"] if raw_candidate is not None else None,
            )
            if fusion_changes:
                selected_timestamps = fused_timestamps
                selected_report = fused_report
                selected_backend = "hybrid"
                selected_quality = candidate_selection_quality(selected_report)
            progress_change_log("local CTC/raw fusion applied", fusion_changes)
            micro_timestamps, micro_report, micro_changes = apply_ctc_micro_refinement_to_whisperx(
                selected_timestamps,
                selected_report,
                ctc_candidate["report"],
                duration,
                raw_candidate["report"] if raw_candidate is not None else None,
            )
            if micro_changes:
                selected_timestamps = micro_timestamps
                selected_report = micro_report
                selected_backend = "hybrid"
                selected_quality = candidate_selection_quality(selected_report)
            progress_change_log("micro CTC refinement applied", micro_changes)
            opening_timestamps, opening_report, opening_changes = apply_hybrid_ctc_opening_onset_backtrack(
                audio_path,
                selected_timestamps,
                selected_report,
                duration,
            )
            if opening_changes:
                selected_timestamps = opening_timestamps
                selected_report = opening_report
                selected_backend = "hybrid"
                selected_quality = candidate_selection_quality(selected_report)
            progress_change_log("opening onset backtrack applied", opening_changes)
            acoustic_timestamps, acoustic_report, acoustic_changes = apply_whisperx_acoustic_boundary_refinement(
                audio_path,
                selected_timestamps,
                selected_report,
                duration,
            )
            if acoustic_changes:
                selected_timestamps = acoustic_timestamps
                selected_report = acoustic_report
                selected_backend = "hybrid"
                selected_quality = candidate_selection_quality(selected_report)
            progress_change_log("acoustic boundary refinement applied", acoustic_changes)
    if ctc_candidate is not None and selected_backend in {"whisperx", "hybrid"}:
        ctc_report = ctc_candidate.get("report")
        if isinstance(ctc_report, dict):
            vocal_timestamps, vocal_report, vocal_changes = apply_vocal_onset_tiebreak(
                selected_timestamps,
                selected_report,
                ctc_report,
                audio_path,
                duration,
                args,
            )
            if vocal_changes:
                selected_timestamps = vocal_timestamps
                selected_report = vocal_report
                selected_backend = "hybrid"
                selected_quality = candidate_selection_quality(selected_report)
            progress_change_log("vocal onset tiebreak applied", vocal_changes)
            flag_unresolved_ctc_disagreements(selected_timestamps, selected_report, ctc_report)
    if selected_backend == "ctc" and raw_diagnostic_report is not None:
        flag_unresolved_raw_ctc_disagreements(selected_timestamps, selected_report, raw_diagnostic_report, entries)
    if selected_backend == "hybrid" and ctc_candidate is not None:
        carried_ctc_report = ctc_candidate.get("report")
        if isinstance(carried_ctc_report, dict):
            carry_bounded_retry_capability_seed(selected_report, carried_ctc_report)
    selected_report["generation_diagnostics"] = {
        "metric_stage": "generation-diagnostic",
        "seed_backend": base_selected_backend,
        "seed_legacy_quality": round(base_selected_quality, 3),
        "refined_legacy_quality": round(selected_quality, 3),
    }
    selected_report["backend"] = selected_backend
    if segment_ctc_seed is None and str(getattr(args,"skeleton","ctc"))=="segment":
        # A failed segment pass must retain its whole-song fallback reason
        # even when the unmodified 1.2 competition selects a hybrid/WX
        # backend afterwards. This is diagnostic only.
        for fallback_candidate in candidates:
            failure_status=(
                fallback_candidate.get("report",{}).get("segment_skeleton_status")
                if isinstance(fallback_candidate.get("report"),dict) else None)
            if (fallback_candidate.get("backend")=="ctc"
                    and isinstance(failure_status,dict)
                    and failure_status.get("status")=="FALLBACK_CTC"):
                selected_report["segment_skeleton_status"]=copy.deepcopy(failure_status)
                break
    if segment_ctc_seed is not None:
        # A successful segment skeleton is the sole positional current. The
        # failure path retains the entire original 1.2 backend competition.
        generation_candidates = [{
            "backend": "ctc",
            "timestamps": list(selected_timestamps),
            "report": selected_report,
            "error": None,
        }]
    else:
        generation_candidates = _central_preview_generation_candidates(
            candidates, selected_backend, selected_timestamps, selected_report
        )

    # Preserve independent WhisperX evidence even when it supplied the seed.
    for whisper_candidate in candidates:
        if (
            whisper_candidate.get("backend") == "whisperx"
            and not whisper_candidate.get("error")
            and isinstance(whisper_candidate.get("report"), dict)
            and isinstance(whisper_candidate.get("timestamps"), list)
            and len(whisper_candidate["timestamps"]) == len(entries)
        ):
            if not any(
                evidence_report is whisper_candidate["report"]
                for _, evidence_report, _ in whisperx_evidence_candidates
            ):
                whisperx_evidence_candidates.append((
                    list(whisper_candidate["timestamps"]),
                    whisper_candidate["report"],
                    "complete-independent-whisperx",
                ))
            break
    bind_ctc_generation_candidate_evidence(
        generation_candidates, entries, whisperx_evidence_candidates
    )

    if str(getattr(args, "timing_source_policy", "default")) == "ctc-first":
        # CTC is the single permitted final backend. All earlier ASR/WX
        # acquisition and CTC diagnostic flags stay intact; the opt-in policy
        # only prevents selecting a non-CTC backend's timestamp authority.
        generation_candidates = [
            item for item in generation_candidates
            if item.get("backend") == "ctc"
        ]
        if not generation_candidates:
            raise LrcError("ctc-first requires a CTC final backend")
    preview_records: list[dict[str, object]] = []
    shared_retry_helper_cache: dict[str, tuple[str, object]] = {}
    for candidate in generation_candidates:
        candidate_report = candidate.get("report")
        candidate_timestamps = candidate.get("timestamps")
        if (
            candidate.get("error")
            or not isinstance(candidate_report, dict)
            or not isinstance(candidate_timestamps, list)
            or len(candidate_timestamps) != len(entries)
        ):
            continue
        backend = str(candidate.get("backend") or candidate_report.get("backend") or "unknown")
        report_copy = copy.deepcopy(candidate_report)
        report_copy["backend"] = backend
        report_copy["lyric_raw_asr_coverage"] = raw_coverage
        progress_log(f"central preview backend={backend}")
        preview_timestamps, preview_report, evaluated, _anchor_changes = prepare_backend_timing_preview(
            audio_path,
            entries,
            list(candidate_timestamps),
            report_copy,
            duration,
            args,
            anchor_path,
            retry_helper_cache=shared_retry_helper_cache,
        )
        bundle = _capture_evaluated_backend_bundle(
            evaluated,
            preview_report,
            audio_path=audio_path,
            args=args,
        )
        preview_records.append({
            "timestamps": preview_timestamps,
            "report": preview_report,
            "evaluated": evaluated,
            "bundle": bundle,
        })
    selection = select_evaluated_backend(tuple(
        record["evaluated"]
        for record in preview_records
        if isinstance(record.get("evaluated"), EvaluatedBackendTiming)
    ))
    selected_record = next(
        record for record in preview_records
        if record.get("evaluated") is selection.selected
    )
    selected_timestamps = list(selected_record["timestamps"])
    selected_report = selected_record["report"]
    assert isinstance(selected_report, dict)
    selected_backend = selection.selected.backend
    selected_report[_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = (
        _build_backend_provenance_projection_seed(
            audio_path,
            args,
            generation_candidates,
            preview_records,
            selection,
            whisperx_evidence_candidates,
        )
    )
    # Capture extra tail evidence only after all Central previews have already
    # been evaluated.  This runtime-only bundle set is deliberately invisible
    # to peer simulation, backend selection and neutral arbitration.
    selected_report[_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY] = tuple(
        _postcentral_tail_enriched_bundle(
            record["bundle"],
            record["report"],
            entries,
            whisperx_evidence_candidates,
        )
        for record in preview_records
        if isinstance(record.get("bundle"), EvaluatedBackendBundle)
        and isinstance(record.get("report"), dict)
    )
    postcentral_bundles = selected_report[_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY]
    assert isinstance(postcentral_bundles, tuple)
    # Freeze the real in-process CTC execution authority for one later,
    # post-neutral tail-only producer.  No candidate set or Central input is
    # retained as mutable authority here; the bundle revision and frozen retry
    # epoch bind the helper execution to the already-evaluated CTC preview.
    for record in preview_records:
        evaluated = record.get("evaluated")
        bundle = record.get("bundle")
        ctc_report = record.get("report")
        if (
            not isinstance(evaluated, EvaluatedBackendTiming)
            or evaluated.backend != "ctc"
            or not isinstance(bundle, EvaluatedBackendBundle)
            or not isinstance(ctc_report, dict)
        ):
            continue
        capability = ctc_report.get(_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY)
        if not isinstance(capability, BoundedRetryCapability):
            continue
        tail_context = _tail_evidence_context_from_report(
            ctc_report, entries=entries, capability=capability
        )
        if tail_context is None or not _tail_context_is_valid(tail_context):
            continue
        enriched_bundle = next(
            (
                item for item in postcentral_bundles
                if isinstance(item, EvaluatedBackendBundle)
                and item.evaluated is evaluated
                and item.evaluated.backend == "ctc"
            ),
            None,
        )
        if enriched_bundle is None:
            continue
        retry_epoch = freeze_base_retry_epoch(
            "ctc",
            ctc_report,
            bundle.candidate_sets,
            thaw_generation_diagnostics(evaluated),
            capability,
        )
        selected_report[_RUNTIME_POSTCENTRAL_CTC_TAIL_PRODUCER_CONTEXT_KEY] = (
            PostcentralCtcTailProducerContext(
                capability=capability,
                epoch=retry_epoch,
                tail_context=tail_context,
                bundle_revision=enriched_bundle.bundle_revision,
                candidate_sets_revision=enriched_bundle.candidate_sets_revision,
                helper_cache=shared_retry_helper_cache,
            )
        )
        break

    selected_participation = copy.deepcopy(
        _backend_participation(selected_report)
    )
    selected_report[_RUNTIME_BACKEND_PARTICIPATION_KEY] = participation
    _merge_backend_participation(
        selected_report,
        {_RUNTIME_BACKEND_PARTICIPATION_KEY: selected_participation},
    )
    for candidate in candidates:
        candidate_report = candidate.get("report")
        if isinstance(candidate_report, dict):
            _merge_backend_participation(selected_report, candidate_report)
    for record in preview_records:
        preview_report = record.get("report")
        if isinstance(preview_report, dict):
            _merge_backend_participation(selected_report, preview_report)
    selected_report["generation_diagnostics"] = {
        **(
            selected_report.get("generation_diagnostics")
            if isinstance(selected_report.get("generation_diagnostics"), dict)
            else {}
        ),
        "legacy_confidence_metrics": [
            candidate_summary(candidate["report"], str(candidate.get("error")) if candidate.get("error") else None)
            for candidate in generation_candidates
            if isinstance(candidate.get("report"), dict)
        ],
        "diagnostic_candidates": [
            candidate_summary(candidate["report"], str(candidate.get("error")) if candidate.get("error") else None)
            for candidate in diagnostic_candidates
            if isinstance(candidate.get("report"), dict)
        ],
    }
    progress_log(
        f"final central backend={selected_backend}: "
        f"timing_trusted={selection.selected.summary.timing_trusted_percent}, "
        f"authority_trusted={selection.selected.summary.overall_trusted_percent}, "
        f"review={selection.selected.summary.review_required_count}, "
        f"unverified={selection.selected.summary.unverified_count}, "
        f"quality={selection.selected.quality:.1f}"
    )
    return selected_timestamps, selected_report, selected_backend, selection


def run_backend_competition_over_skeleton_gate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
    anchor_path: Path | None,
) -> tuple[list[float], dict[str, object], str, BackendSelection]:
    """Run the competition; when a segment-skeleton gate verdict was within
    GPU noise of its cutoff, also run the other path and let Central's own
    evaluation choose.  A near-edge gate never decides the whole song alone.
    """
    _set_runtime_arg(args, SEGMENT_GATE_OVERRIDE_ATTR, None)
    _set_runtime_arg(args, SEGMENT_GATE_STATUS_ATTR, None)
    first = run_auto_backend_competition(audio_path, entries, duration, args, anchor_path)
    first_status = getattr(args, SEGMENT_GATE_STATUS_ATTR, None)
    if not (isinstance(first_status, dict) and first_status.get("near_edge_gates")):
        return first

    def branch(status: object) -> str:
        return (
            "segment"
            if isinstance(status, dict) and status.get("status") == "SEGMENT_CURRENT"
            else "fallback"
        )

    first_branch = branch(first_status)
    other_branch = "fallback" if first_branch == "segment" else "segment"
    progress_log(
        f"segment gate near-edge ({','.join(first_status['near_edge_gates'])}); "
        f"also evaluating the {other_branch} path in Central"
    )
    _set_runtime_arg(args, SEGMENT_GATE_OVERRIDE_ATTR, other_branch)
    try:
        second = run_auto_backend_competition(audio_path, entries, duration, args, anchor_path)
        second_status = getattr(args, SEGMENT_GATE_STATUS_ATTR, None)
    finally:
        _set_runtime_arg(args, SEGMENT_GATE_OVERRIDE_ATTR, None)
    options = [(first_branch, first), (branch(second_status), second)]
    # Path label, not run order, breaks an exact tie, so both machines agree.
    chosen_branch, chosen = min(
        options,
        key=lambda item: (
            -round(float(item[1][3].selected.quality), 3),
            0 if item[0] == "segment" else 1,
        ),
    )
    chosen[1]["segment_gate_dual_evaluation"] = {
        "near_edge_gates": list(first_status["near_edge_gates"]),
        "chosen_path": chosen_branch,
        "paths": [
            {
                "path": label,
                "backend": result[2],
                "central_quality": round(float(result[3].selected.quality), 3),
                "skeleton_reason": (
                    result[1].get("segment_skeleton_status", {}).get("reason")
                    if isinstance(result[1].get("segment_skeleton_status"), dict)
                    else None
                ),
            }
            for label, result in options
        ],
        "BK_used": False,
    }
    progress_log(f"segment gate near-edge: Central chose the {chosen_branch} path")
    return chosen


def freeze_base_retry_epoch(
    backend: str,
    report: dict[str, object],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    diagnostics: dict[str, object],
    capability: BoundedRetryCapability | None,
) -> BaseRetryEpoch:
    candidate_revision = _candidate_set_semantic_revision(candidate_sets)
    _diagnostics_json, diagnostics_integrity_revision = freeze_generation_diagnostics(
        diagnostics
    )
    diagnostics_revision = generation_diagnostics_semantic_revision(diagnostics)
    epoch_payload = {
        "backend": backend,
        "report_projection_revision": timing_report_evaluation_revision(report),
        "ordinary_candidate_sets_revision": candidate_revision,
        "ordinary_diagnostics_revision": diagnostics_revision,
        "capability_revision": capability.capability_revision if capability else None,
        "schema_version": 1,
    }
    return BaseRetryEpoch(
        **epoch_payload,
        ordinary_diagnostics_integrity_revision=diagnostics_integrity_revision,
        retry_epoch_revision=_timing_content_digest(epoch_payload),
    )


def _retry_candidate_reference(candidate: TimingCandidate) -> RetryCandidateReference:
    regions = analyze_candidate_regions(candidate)
    return RetryCandidateReference(
        candidate.candidate_id,
        candidate.generation_revision,
        regions.region_revision,
        candidate.written_time.centiseconds,
    )


def plan_crossline_retry_requests(
    epoch: BaseRetryEpoch,
    capability: BoundedRetryCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> tuple[CrossLineRetryRequest, ...]:
    """Plan bounded *current-line opening* recovery.

    The retry target is the current lyric alone.  Previous/next line starts
    bound the search window, while previous-line tail evidence is diagnostic
    only.  This prevents a long backing-vocal/ad-lib tail from forcing the
    current lyric to start after the previous transcript has been serially
    consumed.
    """
    if capability is None or capability.capability_revision != epoch.capability_revision:
        return ()
    if len(entries) != len(candidate_sets) or not math.isfinite(duration) or duration <= 0.0:
        return ()
    state = build_final_timing_state(
        candidate_sets,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        timing_source_policy=timing_source_policy,
    )
    requests: list[CrossLineRetryRequest] = []
    duration_revision = _timing_content_digest({"duration": round(duration, 6)})

    for index in range(1, len(entries)):
        decision = state.decisions[index]
        if decision.status != "provisional_unresolved":
            continue

        previous_decision = state.decisions[index - 1]
        previous_tail = resolve_reliable_tail(
            previous_decision.candidate,
            independent_tail_evidence[index - 1] if independent_tail_evidence else None,
            tail_evidence_context=tail_evidence_context,
        )
        relation = decision.evidence.ownership
        if (
            relation.previous_tail_evidence_revision is not None
            and relation.previous_tail_evidence_revision != previous_tail.tail_evidence_revision
        ):
            continue

        # A current-line-only retry cannot manufacture missing predecessor-tail
        # ownership evidence.  If the current lyric already has coherent intrinsic
        # opening/path proof and the *only* structural blocker is unavailable
        # previous-tail ownership, re-aligning the same current lyric is impossible
        # to change the authority graph.  Preserve retry for ambiguous/incoherent
        # current paths and for valid-tail uncertainty/violation where a changed
        # current boundary can legitimately change the relation.
        structural_failures = set(decision.evidence.structural_failures)
        current_tail = derive_reliable_tail(decision.candidate)
        if (
            relation.status == "unknown"
            and previous_tail.validity != "valid"
            and current_tail.validity == "valid"
            and decision.evidence.temporal_coherence == "coherent"
            and decision.evidence.onset_proof_end is not None
            and decision.evidence.reliable_support_count >= 3
            and structural_failures
            and structural_failures <= {"previous-tail-ownership-unknown"}
        ):
            continue

        current_ref = _retry_candidate_reference(decision.candidate)
        previous_ref = _retry_candidate_reference(previous_decision.candidate)
        next_ref = (
            _retry_candidate_reference(state.decisions[index + 1].candidate)
            if index + 1 < len(entries)
            else None
        )

        # Opening-first recovery aligns only the current lyric.  Do not serially
        # prepend the previous lyric, because that makes overlap impossible.
        context_indexes = (index,)
        context_candidates = (decision.candidate,)
        transcripts = (entry_sung_text(entries[index]),)
        transcript_digests = (_timing_content_digest(transcripts[0]),)
        row_mapping = ((index, 0),)

        current_start = decision.written_time.seconds
        previous_start = previous_decision.written_time.seconds
        previous_spacing = max(HOP_SECONDS, current_start - previous_start)

        # Search several seconds before the provisional start, but never before
        # the previous lyric's own start.  The previous *tail* is not a lower
        # bound: overlapping vocals are legal.
        backtrack = min(4.0, max(1.25, previous_spacing * 0.60))
        window_start = max(
            0.0,
            previous_start + HOP_SECONDS,
            current_start - backtrack,
        )

        if index + 1 < len(entries):
            next_start = state.decisions[index + 1].written_time.seconds
            # A helper row that reaches the right edge is incomplete evidence,
            # not a reason to accept the truncated path.  On the next bounded
            # fixed-point pass, widen by one *existing* 0.75 s overlap unit per
            # observed truncated retry, with a hard finite expansion count.
            truncated_retry_count = sum(
                1
                for candidate in candidate_sets[index]
                if candidate.source == "ctc-opening-local-retry"
                and candidate.window_truncated
            )
            expansion_count = min(
                CROSSLINE_RETRY_MAX_WINDOW_EXPANSIONS, truncated_retry_count
            )
            forward_overlap = CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS * (1 + expansion_count)
            # Give the current line a bounded overlap allowance beyond the next
            # provisional start so ad-libs/backing vocals do not compress its
            # terminal tokens.  Cap the forward horizon because opening
            # recovery must remain local even when the next lyric is far away.
            window_end = min(duration, next_start + forward_overlap, current_start + 14.0)
        else:
            # Terminal line: infer a bounded phrase horizon from the previous
            # inter-line spacing instead of aligning against the entire song.
            forward = min(14.0, max(7.0, previous_spacing * 1.60))
            window_end = min(duration, current_start + forward)

        if not 1.0 <= window_end - window_start <= 20.0:
            continue

        mapping_revision = _timing_content_digest(row_mapping)
        window_revision = _timing_content_digest({
            "mode": "opening-first-single-line",
            "start": round(window_start, 6),
            "end": round(window_end, 6),
            "previous_start": previous_ref.canonical_centiseconds,
            "current": current_ref.candidate_id,
            "next": next_ref.candidate_id if next_ref else None,
            "retry_window_truncation_count": (
                min(CROSSLINE_RETRY_MAX_WINDOW_EXPANSIONS, truncated_retry_count)
                if index + 1 < len(entries) else 0
            ),
            "duration_revision": duration_revision,
        })
        payload = {
            "retry_epoch_revision": epoch.retry_epoch_revision,
            "entry_index": index,
            "ordinal": 1,
            "capability_revision": capability.capability_revision,
            "alignment_audio_revision": capability.alignment_audio_sha256,
            "current": current_ref,
            "previous": previous_ref,
            "next_bound": next_ref,
            "audio_end_seconds": round(duration, 6),
            "audio_duration_revision": duration_revision,
            "previous_tail_evidence_revision": previous_tail.tail_evidence_revision,
            "previous_tail_start": previous_tail.reliable_tail_start,
            "previous_tail_end": previous_tail.reliable_tail_end,
            "previous_tail_uncertainty": previous_tail.uncertainty_seconds,
            "previous_tail_validity": previous_tail.validity,
            "ownership_relation_revision": relation.relation_revision,
            "context_entry_indexes": context_indexes,
            "context_candidate_ids": tuple(item.candidate_id for item in context_candidates),
            "transcript_digests": transcript_digests,
            "transcript_digest": _timing_content_digest(transcripts),
            "row_mapping": row_mapping,
            "row_mapping_revision": mapping_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "window_definition_revision": window_revision,
        }
        revision_payload = {
            **payload,
            "current": current_ref.__dict__,
            "previous": previous_ref.__dict__,
            "next_bound": next_ref.__dict__ if next_ref else None,
        }
        requests.append(CrossLineRetryRequest(
            **payload,
            request_revision=_timing_content_digest(revision_payload),
        ))
    return tuple(sorted(requests, key=lambda item: (item.entry_index, item.request_revision)))


def plan_joint_boundary_retry_requests(
    epoch: BaseRetryEpoch,
    capability: BoundedRetryCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> tuple[CrossLineRetryRequest, ...]:
    """Plan bounded *previous+current lyric* joint boundary observations.

    Current-line-only retry can recover opening geometry, but it cannot directly
    answer which side of an ambiguous acoustic boundary belongs to the previous
    lyric.  This producer therefore aligns the immutable previous and current
    lyric rows *together* inside one bounded window.

    The helper result is observation-only.  It is never eligible for direct
    central selection; a later consensus step must bind it to a separately
    generated current-line candidate within the existing 0.18 s consensus
    tolerance before it may become boundary-grade evidence.
    """
    if capability is None or capability.capability_revision != epoch.capability_revision:
        return ()
    if len(entries) != len(candidate_sets) or not math.isfinite(duration) or duration <= 0.0:
        return ()

    state = build_final_timing_state(
        candidate_sets,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        timing_source_policy=timing_source_policy,
    )
    requests: list[CrossLineRetryRequest] = []
    duration_revision = _timing_content_digest({"duration": round(duration, 6)})

    for index in range(1, len(entries)):
        decision = state.decisions[index]
        if decision.status != "provisional_unresolved":
            continue

        audit_flags = set(decision.evidence.structural_failures)
        lacks_opening = not (
            decision.evidence.onset_proof_end is not None
            or candidate_has_bound_direct_onset_evidence(decision.candidate)
        )
        needs_boundary = bool(
            decision.evidence.ownership.status in {"unknown", "violation"}
            or lacks_opening
            or {
                "previous-tail-ownership-unknown",
                "previous-tail-overlap",
            }.intersection(audit_flags)
        )
        if not needs_boundary:
            continue

        previous_decision = state.decisions[index - 1]
        previous_tail = resolve_reliable_tail(
            previous_decision.candidate,
            independent_tail_evidence[index - 1] if independent_tail_evidence else None,
            tail_evidence_context=tail_evidence_context,
        )
        relation = decision.evidence.ownership
        if (
            relation.previous_tail_evidence_revision is not None
            and relation.previous_tail_evidence_revision != previous_tail.tail_evidence_revision
        ):
            continue

        current_ref = _retry_candidate_reference(decision.candidate)
        previous_ref = _retry_candidate_reference(previous_decision.candidate)
        next_ref = (
            _retry_candidate_reference(state.decisions[index + 1].candidate)
            if index + 1 < len(entries)
            else None
        )

        context_indexes = (index - 1, index)
        context_candidates = (previous_decision.candidate, decision.candidate)
        transcripts = tuple(entry_sung_text(entries[item]) for item in context_indexes)
        transcript_digests = tuple(_timing_content_digest(text) for text in transcripts)
        row_mapping = ((index - 1, 0), (index, 1))

        previous_start = previous_decision.written_time.seconds
        current_start = decision.written_time.seconds
        previous_previous_start = (
            state.decisions[index - 2].written_time.seconds if index >= 2 else 0.0
        )

        # Include enough pre-roll to let the helper observe the previous row's
        # real opening without swallowing the row before it.
        spacing = max(HOP_SECONDS, current_start - previous_start)
        pre_roll = min(3.0, max(1.0, spacing * 0.45))
        lower_neighbor = previous_previous_start + HOP_SECONDS if index >= 2 else 0.0
        window_start = max(lower_neighbor, previous_start - pre_roll)

        if index + 1 < len(entries):
            next_start = state.decisions[index + 1].written_time.seconds
            window_end = min(
                duration,
                next_start + 2.0 * CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
                current_start + 14.0,
            )
        else:
            window_end = min(duration, current_start + min(14.0, max(7.0, spacing * 1.8)))

        if not 2.0 <= window_end - window_start <= 24.0:
            continue

        mapping_revision = _timing_content_digest(row_mapping)
        window_revision = _timing_content_digest({
            "mode": "joint-previous-current-boundary",
            "start": round(window_start, 6),
            "end": round(window_end, 6),
            "previous_previous_start": round(previous_previous_start, 6),
            "previous": previous_ref.candidate_id,
            "current": current_ref.candidate_id,
            "next": next_ref.candidate_id if next_ref else None,
            "duration_revision": duration_revision,
        })
        payload = {
            "retry_epoch_revision": epoch.retry_epoch_revision,
            "entry_index": index,
            "ordinal": 2,
            "capability_revision": capability.capability_revision,
            "alignment_audio_revision": capability.alignment_audio_sha256,
            "current": current_ref,
            "previous": previous_ref,
            "next_bound": next_ref,
            "audio_end_seconds": round(duration, 6),
            "audio_duration_revision": duration_revision,
            "previous_tail_evidence_revision": previous_tail.tail_evidence_revision,
            "previous_tail_start": previous_tail.reliable_tail_start,
            "previous_tail_end": previous_tail.reliable_tail_end,
            "previous_tail_uncertainty": previous_tail.uncertainty_seconds,
            "previous_tail_validity": previous_tail.validity,
            "ownership_relation_revision": relation.relation_revision,
            "context_entry_indexes": context_indexes,
            "context_candidate_ids": tuple(item.candidate_id for item in context_candidates),
            "transcript_digests": transcript_digests,
            "transcript_digest": _timing_content_digest(transcripts),
            "row_mapping": row_mapping,
            "row_mapping_revision": mapping_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "window_definition_revision": window_revision,
        }
        revision_payload = {
            **payload,
            "current": current_ref.__dict__,
            "previous": previous_ref.__dict__,
            "next_bound": next_ref.__dict__ if next_ref else None,
        }
        requests.append(CrossLineRetryRequest(
            **payload,
            request_revision=_timing_content_digest(revision_payload),
        ))
    return tuple(sorted(requests, key=lambda item: (item.entry_index, item.request_revision)))


def crossline_retry_request_is_current(
    request: CrossLineRetryRequest,
    epoch: BaseRetryEpoch,
    capability: BoundedRetryCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> bool:
    planner = (
        plan_joint_boundary_retry_requests
        if len(request.context_entry_indexes) == 2
        else plan_crossline_retry_requests
    )
    return request in planner(
        epoch,
        capability,
        entries,
        candidate_sets,
        duration,
        independent_tail_evidence,
        tail_evidence_context,
        timing_source_policy,
    )


def plan_local_whisper_requests(
    report: dict[str, object],
    capability: LocalWhisperCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> tuple[LocalWhisperRequest, ...]:
    """Plan independent Whisper-family observations only where central evidence is incomplete."""
    if capability is None or len(entries) != len(candidate_sets):
        return ()
    if not math.isfinite(duration) or duration <= 0.0:
        return ()
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return ()
    state = build_final_timing_state(
        candidate_sets,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        timing_source_policy=timing_source_policy,
    )
    requests: list[LocalWhisperRequest] = []
    audio_revision = capability.audio_sha256

    for index, (decision, assignment) in enumerate(zip(state.decisions, assignments)):
        if not isinstance(assignment, dict):
            continue
        timing_need = decision.status == "provisional_unresolved"
        content_need = bool(
            not assignment_content_is_trusted(assignment)
            and not runtime_independent_content_identity_sources_for_entry(report, index)
        )
        review_need = _suspicious_entry_requires_review(report, index + 1)
        if not (timing_need or content_need or review_need):
            continue

        # Build a bounded identity context with one row of redundancy on each
        # available side.  The extra rows are not presumed trusted; they merely
        # give sparse occurrence binding a second anchor when an immediate
        # neighbor is weak.  If the expanded window would exceed the existing
        # 30 s local cap, fall back to the original minimal context.
        if _entry_identity_is_unique(entries, index):
            base_left = max(0, index - 1)
            base_right = min(len(entries) - 1, index + 1)
        else:
            fingerprint = _minimal_unique_sequence_context(entries, index, max_items=9)
            if fingerprint is None:
                continue
            base_left, base_right = fingerprint

        # Preserve the original minimal probe *and* optionally add one broader
        # sparse-context probe.  Broader context is evidence-additive only: it
        # must never replace a narrower request that may expose a cleaner direct
        # onset.  Runtime merging already ranks direct-onset evidence above
        # occurrence-only evidence for the same entry.
        context_options: list[tuple[int, int]] = [(base_left, base_right)]
        # Evaluate a small monotonic context ladder rather than betting the whole
        # observation on one segmentation-sensitive window.  Each broader probe
        # is optional evidence only; the minimal probe is always preserved.
        # Bound the ladder to at most three contexts and nine lyric rows so the
        # helper cost stays finite even on long songs.
        for expansion in (1, 2):
            expanded_left = max(0, base_left - expansion)
            expanded_right = min(len(entries) - 1, base_right + expansion)
            option = (expanded_left, expanded_right)
            if (
                expanded_left < index < expanded_right
                and expanded_right - expanded_left + 1 <= 9
                and option not in context_options
            ):
                context_options.append(option)

        seen_contexts: set[tuple[int, int, int, int]] = set()
        for option_left, option_right in context_options:
            option_left_start = state.decisions[option_left].written_time.seconds
            if option_left > 0:
                option_predecessor = state.decisions[option_left - 1].written_time.seconds
                option_window_start = max(
                    0.0,
                    option_predecessor + HOP_SECONDS,
                    option_left_start - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
                )
            else:
                option_window_start = 0.0

            option_right_start = state.decisions[option_right].written_time.seconds
            if option_right + 1 < len(entries):
                option_successor = state.decisions[option_right + 1].written_time.seconds
                option_window_end = min(
                    duration,
                    option_successor + LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
                )
            else:
                option_window_end = min(duration, option_right_start + 8.0)
            if option_window_end <= option_right_start + 0.5:
                option_window_end = min(duration, option_right_start + 4.0)
            option_length = option_window_end - option_window_start
            if (
                option_left == 0
                and option_length > LOCAL_WHISPER_MAX_WINDOW_SECONDS
            ):
                option_window_start = max(
                    0.0,
                    option_left_start - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
                )
                option_length = option_window_end - option_window_start
            if not 2.0 <= option_length <= LOCAL_WHISPER_MAX_WINDOW_SECONDS:
                continue

            context_indexes = tuple(range(option_left, option_right + 1))
            if not context_indexes or index not in context_indexes or len(context_indexes) > 9:
                continue
            semantic_key = (
                option_left,
                option_right,
                int(round(option_window_start * 1000)),
                int(round(option_window_end * 1000)),
            )
            if semantic_key in seen_contexts:
                continue
            seen_contexts.add(semantic_key)

            context_candidates = tuple(state.decisions[item].candidate for item in context_indexes)
            transcripts = tuple(entry_sung_text(entries[item]) for item in context_indexes)
            transcript_digests = tuple(_timing_content_digest(text) for text in transcripts)

            window_definition = {
                "mode": "unresolved-only-local-whisper",
                "start": round(option_window_start, 6),
                "end": round(option_window_end, 6),
                "context_indexes": list(context_indexes),
                "context_candidate_ids": [item.candidate_id for item in context_candidates],
                "timing_need": timing_need,
                "content_need": content_need,
                "review_need": review_need,
                "capability_revision": capability.capability_revision,
            }
            payload = {
                "entry_index": index,
                "capability_revision": capability.capability_revision,
                "audio_revision": audio_revision,
                "context_entry_indexes": context_indexes,
                "context_candidate_ids": tuple(item.candidate_id for item in context_candidates),
                "transcript_digests": transcript_digests,
                "transcript_digest": _timing_content_digest(transcripts),
                "window_start": round(option_window_start, 6),
                "window_end": round(option_window_end, 6),
                "window_definition_revision": _timing_content_digest(window_definition),
            }
            requests.append(LocalWhisperRequest(
                **payload,
                request_revision=_timing_content_digest({
                    **payload,
                    "context_entry_indexes": list(context_indexes),
                    "context_candidate_ids": list(payload["context_candidate_ids"]),
                    "transcript_digests": list(transcript_digests),
                }),
                execution_revision=_local_whisper_execution_request_revision(payload),
            ))
    # A broken global CTC can collapse an entire verse 10-20 seconds earlier.
    # Local Whisper requests centered on Central's provisional baseline will
    # then listen to the wrong audio, even when S1 has a real bounded CTC path
    # at the correct vocal segment. Add only a capped independent ASR probe
    # around the *untrusted* S1 path. This is evidence acquisition, not timing
    # selection, and does not make S1 direct-onset/identity evidence.
    displaced_segment_rows: list[tuple[float, int, tuple[TimingCandidate, ...]]] = []
    segment_by_row: dict[int, TimingCandidate] = {}
    for index, items in enumerate(candidate_sets):
        proposals = [item for item in items
                     if item.source == "segment-constrained-ctc"
                     and item.spans and not item.window_truncated
                     and not item.right_edge_pileup]
        if not proposals:
            # V2 segment-realign proposals get the same capped independent probe.
            proposals = [item for item in items
                         if item.source == SEGMENT_REALIGN_SOURCE
                         and item.spans and not item.window_truncated
                         and not item.right_edge_pileup]
        if len(proposals) == 1:
            segment_by_row[index] = proposals[0]
            gap = abs(proposals[0].written_time.seconds - state.decisions[index].written_time.seconds)
            if gap >= 3.0 and (
                state.decisions[index].status == "provisional_unresolved"
                or not assignment_content_is_trusted(assignments[index])
                or _suspicious_entry_requires_review(report, index + 1)
            ):
                displaced_segment_rows.append((-gap, index, items))
    for _gap, index, _items in sorted(displaced_segment_rows)[:12]:
        if _entry_identity_is_unique(entries, index):
            first, last = max(0, index - 1), min(len(entries) - 1, index + 1)
        else:
            context = _minimal_unique_sequence_context(entries, index, max_items=9)
            if context is None:
                continue
            first, last = context
        context_indexes = tuple(range(first, last + 1))
        if (
            len(context_indexes) > 9
            or any(j not in segment_by_row for j in context_indexes)
        ):
            continue
        context_candidates = tuple(segment_by_row[j] for j in context_indexes)
        s1_times = tuple(item.written_time.seconds for item in context_candidates)
        if any(a >= b for a, b in zip(s1_times, s1_times[1:])):
            continue
        window_start = max(0.0, s1_times[0] - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS)
        window_end = min(duration, s1_times[-1] + max(4.0, LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS))
        if not 2.0 <= window_end - window_start <= LOCAL_WHISPER_MAX_WINDOW_SECONDS:
            continue
        transcript = tuple(entry_sung_text(entries[j]) for j in context_indexes)
        transcripts_digest = tuple(_timing_content_digest(text) for text in transcript)
        window_definition = {
            "mode": "s1-untrusted-segment-search-not-proof",
            "start": round(window_start, 6),
            "end": round(window_end, 6),
            "context_indexes": list(context_indexes),
            "context_candidate_ids": [x.candidate_id for x in context_candidates],
            "capability_revision": capability.capability_revision,
        }
        payload = {
            "entry_index": index,
            "capability_revision": capability.capability_revision,
            "audio_revision": audio_revision,
            "context_entry_indexes": context_indexes,
            "context_candidate_ids": tuple(x.candidate_id for x in context_candidates),
            "transcript_digests": transcripts_digest,
            "transcript_digest": _timing_content_digest(transcript),
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "window_definition_revision": _timing_content_digest(window_definition),
        }
        requests.append(LocalWhisperRequest(
            **payload,
            request_revision=_timing_content_digest({
                **payload,
                "context_entry_indexes": list(context_indexes),
                "context_candidate_ids": list(payload["context_candidate_ids"]),
                "transcript_digests": list(transcripts_digest),
            }),
            execution_revision=_local_whisper_execution_request_revision(payload),
        ))
    unique = {item.request_revision: item for item in requests}
    return tuple(sorted(
        unique.values(),
        key=lambda item: (item.entry_index, len(item.context_entry_indexes), item.request_revision),
    ))


_R2T_NEUTRAL_CONFLICT_PROBE_KEY = "r2t_unified_neutral_conflict_probe"
_R2T_NEUTRAL_CONFLICT_PROBE_MODE = "r2t-final-material-conflict-source-audio-neutral-local-whisper-v1"


def _r2t_validated_peer_conflict_rows(
    report: dict[str, object],
) -> tuple[dict[str, object] | None, tuple[dict[str, object], ...]]:
    projection = report.get(_BACKEND_PROVENANCE_PROJECTION_KEY)
    if not isinstance(projection, dict):
        return None, ()
    conflicts = projection.get("cross_backend_material_conflicts")
    if not isinstance(conflicts, dict):
        return None, ()
    if conflicts.get("authority") != "review-only" or conflicts.get("status") != "RETRIEVED":
        return conflicts, ()
    revision = conflicts.get("revision")
    if not isinstance(revision, str) or not revision:
        return conflicts, ()
    canonical = dict(conflicts)
    canonical.pop("revision", None)
    if _timing_content_digest(canonical) != revision:
        return conflicts, ()
    rows = conflicts.get("entries")
    if not isinstance(rows, list):
        return conflicts, ()
    return conflicts, tuple(
        item for item in rows
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    )


def _r2t_unresolved_projection_rows(
    report: dict[str, object],
) -> tuple[dict[str, object] | None, tuple[dict[str, object], ...]]:
    projection = report.get(_UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY)
    if not isinstance(projection, dict) or projection.get("status") != "RETRIEVED":
        return projection if isinstance(projection, dict) else None, ()
    revision = projection.get("projection_revision")
    if not isinstance(revision, str) or not revision:
        return projection, ()
    canonical = dict(projection)
    canonical.pop("projection_revision", None)
    if _timing_content_digest(canonical) != revision:
        return projection, ()
    rows = projection.get("rows")
    if not isinstance(rows, list):
        return projection, ()
    return projection, tuple(
        item for item in rows
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    )


def _r2t_material_cluster_rows(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    rows: list[dict[str, object]] = []
    for item in value:
        if (
            isinstance(item, dict)
            and item.get("relation") == "distant"
            and item.get("review_required") is True
            and item.get("explicit_falsification") is not True
        ):
            try:
                anchor = float(item.get("anchor_time"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(anchor):
                continue
            rows.append(item)
    return tuple(rows)


def _r2t_collect_final_material_targets(
    report: dict[str, object],
    final_state: FinalTimingState,
) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
    """Collect final unresolved material clusters from both Central and peer consumers.

    This is reporting/probe planning only.  It consumes already-finalized state,
    never reconstructs candidate semantics, and never changes review or trust.
    """
    targets: dict[int, dict[str, object]] = {}
    unresolved_projection, unresolved_rows = _r2t_unresolved_projection_rows(report)
    peer_projection, peer_rows = _r2t_validated_peer_conflict_rows(report)

    def ensure_target(entry_number: int) -> dict[str, object] | None:
        index = entry_number - 1
        if not 0 <= index < len(final_state.decisions):
            return None
        selected = final_state.decisions[index]
        target = targets.setdefault(index, {
            "entry": entry_number,
            "selected_time": round(selected.written_time.seconds, 6),
            "selected_centiseconds": selected.written_time.centiseconds,
            "selected_candidate_id": selected.candidate.candidate_id,
            "selected_source": selected.candidate.source,
            "owners": [],
            "clusters": [],
        })
        return target

    for row in unresolved_rows:
        if row.get("review_required") is not True:
            continue
        clusters = _r2t_material_cluster_rows(row.get("material_clusters"))
        if not clusters:
            continue
        target = ensure_target(int(row["entry"]))
        if target is None:
            continue
        if "central" not in target["owners"]:
            target["owners"].append("central")
        for cluster in clusters:
            target["clusters"].append({
                "owner": "central",
                "anchor_time": round(float(cluster["anchor_time"]), 6),
                "anchor_centiseconds": canonicalize_lrc_time(float(cluster["anchor_time"])).centiseconds,
                "cluster_revision": cluster.get("cluster_revision"),
                "support_families": copy.deepcopy(cluster.get("support_families", [])),
                "member_candidate_ids": copy.deepcopy(cluster.get("member_candidate_ids", [])),
                "representative_candidate_id": cluster.get("representative_candidate_id"),
                "projection_revision": unresolved_projection.get("projection_revision")
                if isinstance(unresolved_projection, dict) else None,
            })

    for row in peer_rows:
        clusters = _r2t_material_cluster_rows(row.get("clusters"))
        if not clusters:
            continue
        target = ensure_target(int(row["entry"]))
        if target is None:
            continue
        if "peer" not in target["owners"]:
            target["owners"].append("peer")
        for cluster in clusters:
            target["clusters"].append({
                "owner": "peer",
                "anchor_time": round(float(cluster["anchor_time"]), 6),
                "anchor_centiseconds": canonicalize_lrc_time(float(cluster["anchor_time"])).centiseconds,
                "cluster_revision": cluster.get("cluster_revision"),
                "support_families": copy.deepcopy(cluster.get("support_families", [])),
                "member_candidate_ids": copy.deepcopy(cluster.get("member_candidate_ids", [])),
                "representative_candidate_id": cluster.get("representative_candidate_id"),
                "projection_revision": peer_projection.get("revision")
                if isinstance(peer_projection, dict) else None,
            })

    tolerance_centiseconds = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    for index in list(targets):
        target = targets[index]
        selected_centiseconds = int(target["selected_centiseconds"])
        dedup: dict[int, dict[str, object]] = {}
        for cluster in target["clusters"]:
            anchor_centiseconds = int(cluster["anchor_centiseconds"])
            if abs(anchor_centiseconds - selected_centiseconds) <= tolerance_centiseconds:
                continue
            existing = dedup.get(anchor_centiseconds)
            if existing is None:
                existing = {
                    "anchor_time": round(anchor_centiseconds / 100.0, 6),
                    "anchor_centiseconds": anchor_centiseconds,
                    "owners": [],
                    "cluster_revisions": [],
                    "projection_revisions": [],
                    "support_families": [],
                    "member_candidate_ids": [],
                    "representative_candidate_ids": [],
                }
                dedup[anchor_centiseconds] = existing
            owner = cluster.get("owner")
            if isinstance(owner, str) and owner:
                existing["owners"].append(owner)
            revision = cluster.get("cluster_revision")
            if isinstance(revision, str) and revision:
                existing["cluster_revisions"].append(revision)
            projection_revision = cluster.get("projection_revision")
            if isinstance(projection_revision, str) and projection_revision:
                existing["projection_revisions"].append(projection_revision)
            families = cluster.get("support_families")
            if isinstance(families, list):
                existing["support_families"].extend(
                    str(item) for item in families if isinstance(item, str) and item
                )
            member_ids = cluster.get("member_candidate_ids")
            if isinstance(member_ids, list):
                existing["member_candidate_ids"].extend(
                    str(item) for item in member_ids if isinstance(item, str) and item
                )
            representative = cluster.get("representative_candidate_id")
            if isinstance(representative, str) and representative:
                existing["representative_candidate_ids"].append(representative)
        for cluster in dedup.values():
            for key in (
                "owners",
                "cluster_revisions",
                "projection_revisions",
                "support_families",
                "member_candidate_ids",
                "representative_candidate_ids",
            ):
                cluster[key] = sorted(set(cluster[key]))
        target["clusters"] = sorted(
            dedup.values(),
            key=lambda item: int(item["anchor_centiseconds"]),
        )
        target["owners"] = sorted(set(str(item) for item in target["owners"]))
        if not target["clusters"]:
            del targets[index]

    metadata = {
        "central_projection_revision": (
            unresolved_projection.get("projection_revision")
            if isinstance(unresolved_projection, dict) else None
        ),
        "peer_projection_revision": (
            peer_projection.get("revision")
            if isinstance(peer_projection, dict) else None
        ),
        "central_target_count": sum("central" in item["owners"] for item in targets.values()),
        "peer_target_count": sum("peer" in item["owners"] for item in targets.values()),
        "union_target_count": len(targets),
        "materiality_tolerance_centiseconds": tolerance_centiseconds,
    }
    return targets, metadata


def _local_whisper_execution_request_revision(
    payload: dict[str, object],
) -> str:
    """Hash only the bounded helper input; planner provenance stays diagnostic."""
    return _timing_content_digest({
        "schema": "local-whisper-execution-request-v2",
        "entry_index": payload["entry_index"],
        "capability_revision": payload["capability_revision"],
        "audio_revision": payload["audio_revision"],
        "context_entry_indexes": list(payload["context_entry_indexes"]),
        "transcript_digests": list(payload["transcript_digests"]),
        "transcript_digest": payload["transcript_digest"],
        "window_start": payload["window_start"],
        "window_end": payload["window_end"],
    })


def _r2t_plan_neutral_conflict_probe_requests(
    report: dict[str, object],
    capability: LocalWhisperCapability | None,
    entries: list[LyricEntry],
    final_state: FinalTimingState,
    duration: float,
) -> tuple[tuple[LocalWhisperRequest, ...], dict[int, dict[str, object]], dict[str, object]]:
    targets, source_metadata = _r2t_collect_final_material_targets(report, final_state)
    if len(final_state.decisions) != len(entries):
        return (), {}, source_metadata
    if not math.isfinite(duration) or duration <= 0.0:
        return (), {}, source_metadata

    metadata: dict[int, dict[str, object]] = {}
    requests: list[LocalWhisperRequest] = []
    for index, target in sorted(targets.items()):
        entry_number = index + 1
        selected_time = final_state.decisions[index].written_time.seconds
        anchor_times = [float(item["anchor_time"]) for item in target["clusters"]]
        if not anchor_times:
            continue

        if _entry_identity_is_unique(entries, index):
            # Global uniqueness is already sufficient occurrence context.  Keep
            # the neutral window centered on the disputed row so long inter-line
            # gaps do not inflate or suppress the probe.
            left = index
            right = index
        else:
            context = _minimal_unique_sequence_context(entries, index, max_items=9)
            if context is None:
                metadata[index] = {
                    **copy.deepcopy(target),
                    "status": "skipped",
                    "skip_reason": "unique-lyric-context-unavailable",
                }
                continue
            left, right = context

        context_times = [
            final_state.decisions[item].written_time.seconds
            for item in range(left, right + 1)
        ]
        low = min([selected_time, *anchor_times, *context_times])
        high = max([selected_time, *anchor_times, *context_times])
        window_start = max(0.0, low - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS)
        if right + 1 < len(entries):
            successor = final_state.decisions[right + 1].written_time.seconds
            high = max(high, min(successor, high + LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS))
        window_end = min(duration, high + LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS)
        window_length = window_end - window_start
        if not 2.0 <= window_length <= LOCAL_WHISPER_MAX_WINDOW_SECONDS:
            metadata[index] = {
                **copy.deepcopy(target),
                "status": "skipped",
                "skip_reason": "neutral-window-outside-bounds",
                "window_start": round(window_start, 6),
                "window_end": round(window_end, 6),
                "window_length": round(window_length, 6),
            }
            continue

        context_indexes = tuple(range(left, right + 1))
        context_candidates = tuple(final_state.decisions[item].candidate for item in context_indexes)
        transcripts = tuple(entry_sung_text(entries[item]) for item in context_indexes)
        transcript_digests = tuple(_timing_content_digest(item) for item in transcripts)
        window_definition = {
            "mode": _R2T_NEUTRAL_CONFLICT_PROBE_MODE,
            "entry": entry_number,
            "start": round(window_start, 6),
            "end": round(window_end, 6),
            "selected_time": round(selected_time, 6),
            "clusters": copy.deepcopy(target["clusters"]),
            "owners": list(target["owners"]),
            "context_indexes": list(context_indexes),
            "context_candidate_ids": [item.candidate_id for item in context_candidates],
            "source_metadata": source_metadata,
            "capability_revision": capability.capability_revision if capability is not None else None,
        }
        metadata[index] = {
            **copy.deepcopy(target),
            "status": "planned" if capability is not None else "capability-unavailable",
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "context_entries": [item + 1 for item in context_indexes],
            "window_definition_revision": _timing_content_digest(window_definition),
        }
        if capability is None:
            continue

        payload = {
            "entry_index": index,
            "capability_revision": capability.capability_revision,
            "audio_revision": capability.audio_sha256,
            "context_entry_indexes": context_indexes,
            "context_candidate_ids": tuple(item.candidate_id for item in context_candidates),
            "transcript_digests": transcript_digests,
            "transcript_digest": _timing_content_digest(transcripts),
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "window_definition_revision": _timing_content_digest(window_definition),
        }
        requests.append(LocalWhisperRequest(
            **payload,
            request_revision=_timing_content_digest({
                **payload,
                "context_entry_indexes": list(context_indexes),
                "context_candidate_ids": list(payload["context_candidate_ids"]),
                "transcript_digests": list(transcript_digests),
            }),
            execution_revision=_local_whisper_execution_request_revision(payload),
        ))
    return (
        tuple(sorted(requests, key=lambda item: (item.entry_index, item.request_revision))),
        metadata,
        source_metadata,
    )


def _r2t_authority_snapshot(report: dict[str, object]) -> str:
    assignments = report.get("assignments")
    assignment_projection: list[object] = []
    if isinstance(assignments, list):
        for item in assignments:
            if not isinstance(item, dict):
                assignment_projection.append(None)
                continue
            assignment_projection.append({
                "timestamp": item.get("timestamp"),
                "candidate_source": item.get("candidate_source"),
                "candidate_id": item.get("selected_candidate_id") or item.get("provisional_candidate_id"),
                "timing_trusted": item.get("timing_trusted"),
                "review_required": item.get("review_required"),
                "overall_trusted": item.get("overall_trusted"),
                "verification_status": item.get("verification_status"),
            })
    return _timing_content_digest({
        "assignments": assignment_projection,
        "central_timing_state": report.get("central_timing_state"),
        "candidate_selection": report.get("candidate_selection"),
        "review_reason_breakdown": report.get("review_reason_breakdown"),
        "authority_trusted_percent": report.get("authority_trusted_percent"),
        "review_required_percent": report.get("review_required_percent"),
    })


def _r2t_probe_relation(
    observation_time: float | None,
    selected_centiseconds: int,
    clusters: list[dict[str, object]],
) -> dict[str, object]:
    tolerance_centiseconds = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    if observation_time is None or not math.isfinite(observation_time):
        return {
            "status": "no-boundary-observation",
            "selected_distance_centiseconds": None,
            "matched_clusters": [],
        }
    observed_centiseconds = canonicalize_lrc_time(observation_time).centiseconds
    selected_distance = abs(observed_centiseconds - selected_centiseconds)
    matched_clusters: list[dict[str, object]] = []
    for cluster in clusters:
        try:
            anchor_centiseconds = int(cluster.get("anchor_centiseconds"))
        except (TypeError, ValueError):
            continue
        distance = abs(observed_centiseconds - anchor_centiseconds)
        if distance <= tolerance_centiseconds:
            matched_clusters.append({
                "owners": copy.deepcopy(cluster.get("owners", [])),
                "anchor_time": cluster.get("anchor_time"),
                "anchor_centiseconds": anchor_centiseconds,
                "distance_centiseconds": distance,
                "cluster_revisions": copy.deepcopy(cluster.get("cluster_revisions", [])),
            })
    selected_match = selected_distance <= tolerance_centiseconds
    if selected_match and not matched_clusters:
        status = "selected-only"
    elif not selected_match and len(matched_clusters) == 1:
        status = "one-material-cluster-only"
    elif selected_match or matched_clusters:
        status = "ambiguous-multi-cluster"
    else:
        status = "no-cluster-match"
    return {
        "status": status,
        "observation_centiseconds": observed_centiseconds,
        "selected_distance_centiseconds": selected_distance,
        "matched_clusters": matched_clusters,
    }


def run_r2t_unified_neutral_conflict_probe(
    audio_path: Path,
    entries: list[LyricEntry],
    report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
    final_state: FinalTimingState,
) -> tuple[dict[str, object], dict[int, LocalWhisperOutcome]]:
    """Probe every final material conflict from neutral source audio.

    The probe itself remains non-authoritative.  The returned runtime outcomes may
    be consumed only by the bounded Central arbitration stage below.
    """
    _validate_unresolved_evidence_requirements_projection(report)
    _validate_backend_provenance_projection(report)
    authority_before = _r2t_authority_snapshot(report)
    capability = build_local_whisper_capability(audio_path, args)
    requests, metadata, source_metadata = _r2t_plan_neutral_conflict_probe_requests(
        report, capability, entries, final_state, duration
    )
    outcomes: tuple[LocalWhisperOutcome, ...] = ()
    preparation_errors: dict[str, str] = {}
    if capability is not None and requests:
        progress_log(
            f"R2T neutral material probe: targets={len(requests)} workers={DEFAULT_EVIDENCE_WORKERS}"
        )
        work = [
            (_local_whisper_cache_key(request, capability), ordinal, request)
            for ordinal, request in enumerate(requests, start=1)
        ]
        prepared, preparation_errors = _prepare_local_whisper_batch(work, capability, args)

        def execute_one(
            item: tuple[str, int, LocalWhisperRequest],
        ) -> tuple[int, LocalWhisperOutcome]:
            key, ordinal, request = item
            outcome = execute_local_whisper_request(
                request,
                capability,
                entries,
                args,
                cache={},
                runtime_stats={},
                prepared=prepared.get(key),
                preparation_error=preparation_errors.get(key),
            )
            return ordinal, outcome

        by_ordinal: dict[int, LocalWhisperOutcome] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(DEFAULT_EVIDENCE_WORKERS, len(work) or 1),
            thread_name_prefix="lrc-r2t-neutral",
        ) as pool:
            for ordinal, outcome in pool.map(execute_one, work):
                by_ordinal[ordinal] = outcome
        outcomes = tuple(by_ordinal[index] for index in range(1, len(work) + 1))

    result_rows: list[dict[str, object]] = []
    request_by_revision = {item.request_revision: item for item in requests}
    for outcome in outcomes:
        request = request_by_revision.get(outcome.request_revision)
        detail = metadata.get(outcome.entry_index)
        if request is None or not isinstance(detail, dict):
            continue
        occurrence_time = None
        if isinstance(outcome.occurrence_binding, dict):
            try:
                value = float(outcome.occurrence_binding.get("timestamp"))
            except (TypeError, ValueError):
                value = math.nan
            if math.isfinite(value):
                occurrence_time = value
        clusters = detail.get("clusters")
        cluster_rows = clusters if isinstance(clusters, list) else []
        selected_centiseconds = int(detail["selected_centiseconds"])
        onset_relation = _r2t_probe_relation(
            outcome.onset_time,
            selected_centiseconds,
            cluster_rows,
        )
        occurrence_relation = _r2t_probe_relation(
            occurrence_time,
            selected_centiseconds,
            cluster_rows,
        )
        result_rows.append({
            **copy.deepcopy(detail),
            "status": outcome.status,
            "reason": outcome.reason,
            "request_revision": outcome.request_revision,
            "onset_time": round(outcome.onset_time, 6) if outcome.onset_time is not None else None,
            "onset_relation": onset_relation,
            "score": round(outcome.score, 6) if outcome.score is not None else None,
            "occurrence_binding_mode": (
                outcome.occurrence_binding.get("mode")
                if isinstance(outcome.occurrence_binding, dict) else None
            ),
            "occurrence_binding_revision": (
                outcome.occurrence_binding.get("binding_revision")
                if isinstance(outcome.occurrence_binding, dict) else None
            ),
            "occurrence_time": round(occurrence_time, 6) if occurrence_time is not None else None,
            "occurrence_relation": occurrence_relation,
            "evidence_revision": outcome.evidence_revision,
            "outcome_revision": outcome.outcome_revision,
        })

    authority_after = _r2t_authority_snapshot(report)
    if authority_after != authority_before:
        raise LrcError("R2T neutral material probe mutated Central authority state")
    onset_relation_counts = dict(sorted(collections.Counter(
        str(item.get("onset_relation", {}).get("status"))
        for item in result_rows
        if isinstance(item.get("onset_relation"), dict)
    ).items()))
    occurrence_relation_counts = dict(sorted(collections.Counter(
        str(item.get("occurrence_relation", {}).get("status"))
        for item in result_rows
        if isinstance(item.get("occurrence_relation"), dict)
    ).items()))
    skipped = [
        copy.deepcopy(item)
        for item in metadata.values()
        if isinstance(item, dict) and item.get("status") == "skipped"
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "authority": "diagnostic-only",
        "mode": _R2T_NEUTRAL_CONFLICT_PROBE_MODE,
        "status": "complete" if capability is not None else "capability-unavailable",
        **source_metadata,
        "planned_entries": [item.entry_index + 1 for item in requests],
        "target_count": len(requests),
        "evidence_count": sum(item.status == "evidence" for item in outcomes),
        "invalid_count": sum(item.status == "invalid" for item in outcomes),
        "error_count": sum(item.status == "error" for item in outcomes),
        "skipped_count": len(skipped),
        "preparation_error_count": len(preparation_errors),
        "onset_relation_counts": onset_relation_counts,
        "occurrence_relation_counts": occurrence_relation_counts,
        "source_audio_neutral": True,
        "occurrence_time_semantics": "identity-binding-only-not-boundary-authority",
        "onset_relation_semantics": "diagnostic-only-existing-18cs-materiality-contract",
        "candidate_graph_mutated": False,
        "timestamp_mutated": False,
        "trust_mutated": False,
        "central_re_evaluated": False,
        "authority_snapshot_unchanged": authority_after == authority_before,
        "entries": result_rows,
        "skipped_entries": sorted(skipped, key=lambda item: int(item.get("entry", 0))),
    }
    payload["probe_revision"] = _timing_content_digest(payload)
    report[_R2T_NEUTRAL_CONFLICT_PROBE_KEY] = payload
    return payload, {item.entry_index: item for item in outcomes}


_NEUTRAL_BOUNDARY_ARBITRATION_KEY = "neutral_boundary_cross_backend_arbitration"
_NEUTRAL_BOUNDARY_ARBITRATION_MODE = "unique-neutral-onset-material-cluster-central-replay-v1"


def _neutral_arbitration_support_families(candidate: TimingCandidate) -> frozenset[str]:
    """Conservative lineage view used only by neutral-conflict arbitration.

    Keep the production mapper unchanged here, but close its known diagnostic gap
    for ``whispercpp-current`` so this arbitration can never manufacture a false
    independent vote from the same Whisper family.
    """
    families = set(_candidate_support_families(candidate))
    labels = " ".join(filter(None, [
        candidate.source,
        candidate.direct_onset_evidence.producer if candidate.direct_onset_evidence else "",
        candidate.acoustic_evidence.producer if candidate.acoustic_evidence else "",
    ])).lower()
    if "whispercpp" in labels:
        families.add("whisper-family")
    return frozenset(families)


def _neutral_probe_trigger_rows(payload: dict[str, object]) -> tuple[dict[str, object], ...]:
    """Freeze the blind discriminator before any reference evaluation."""
    if payload.get("source_audio_neutral") is not True:
        return ()
    tolerance_cs = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    rows = payload.get("entries")
    if not isinstance(rows, list):
        return ()
    triggered: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "evidence":
            continue
        relation = row.get("onset_relation")
        if not isinstance(relation, dict) or relation.get("status") != "one-material-cluster-only":
            continue
        matched = relation.get("matched_clusters")
        if not isinstance(matched, list) or len(matched) != 1 or not isinstance(matched[0], dict):
            continue
        try:
            selected_distance = int(relation.get("selected_distance_centiseconds"))
            onset_cs = int(relation.get("observation_centiseconds"))
            anchor_cs = int(matched[0].get("anchor_centiseconds"))
            score = float(row.get("score"))
        except (TypeError, ValueError):
            continue
        if (
            selected_distance <= tolerance_cs
            or abs(onset_cs - anchor_cs) > tolerance_cs
            or not math.isfinite(score)
        ):
            continue
        mode = row.get("occurrence_binding_mode")
        if mode not in _CANDIDATE_OCCURRENCE_MODES:
            continue
        triggered.append(row)
    return tuple(triggered)


def _neutral_arbitration_candidate_map(
    selection: BackendSelection,
) -> dict[str, TimingCandidate]:
    found: dict[str, TimingCandidate] = {}
    for evaluated in selection.candidates:
        for candidates in evaluated.candidate_sets:
            for candidate in candidates:
                found.setdefault(candidate.candidate_id, candidate)
    return found


def _neutral_arbitration_valid_candidate_ids(
    selection: BackendSelection,
    entry_index: int,
) -> frozenset[str]:
    """Return only candidates Central already considered structurally valid.

    A raw candidate being present in a backend bundle is not evidence that it
    supports a boundary.  This deliberately excludes candidates already
    rejected for ownership/minimum-proof/structural failures, preventing the
    neutral arbitration from resurrecting them as support or vetoes.
    """
    valid: set[str] = set()
    for evaluated in selection.candidates:
        if not 0 <= entry_index < len(evaluated.state.decisions):
            continue
        decision = evaluated.state.decisions[entry_index]
        if decision.status == "selected_valid" and decision.selected_candidate_id:
            valid.add(decision.selected_candidate_id)
        for rejected in decision.rejected_candidates:
            if rejected.valid:
                valid.add(rejected.candidate.candidate_id)
    return frozenset(valid)


def _neutral_arbitration_matching_cluster(
    row: dict[str, object],
) -> dict[str, object] | None:
    relation = row.get("onset_relation")
    clusters = row.get("clusters")
    if not isinstance(relation, dict) or not isinstance(clusters, list):
        return None
    matched = relation.get("matched_clusters")
    if not isinstance(matched, list) or len(matched) != 1 or not isinstance(matched[0], dict):
        return None
    try:
        anchor_cs = int(matched[0].get("anchor_centiseconds"))
    except (TypeError, ValueError):
        return None
    exact: list[dict[str, object]] = []
    for item in clusters:
        if not isinstance(item, dict):
            continue
        try:
            item_anchor_cs = int(item.get("anchor_centiseconds"))
        except (TypeError, ValueError):
            continue
        if item_anchor_cs == anchor_cs:
            exact.append(item)
    return exact[0] if len(exact) == 1 else None


def _neutral_arbitration_consensus_candidate(
    *,
    entries: list[LyricEntry],
    entry_index: int,
    parent: TimingCandidate,
    outcome: LocalWhisperOutcome,
    cluster: dict[str, object],
    probe_revision: str,
) -> TimingCandidate | None:
    if outcome.status != "evidence" or outcome.onset_time is None or outcome.score is None:
        return None
    occurrence = candidate_occurrence_evidence_from_binding(
        entries,
        entry_index,
        producer="local-whisper-aligner",
        occurrence_binding=outcome.occurrence_binding,
        upstream_evidence_revision=outcome.evidence_revision,
    )
    if occurrence is None:
        return None
    families = _neutral_arbitration_support_families(parent)
    if "whisper-family" in families:
        return None
    if "ctc" in families:
        source = "ctc-local-whisper-neutral-conflict-consensus"
    elif "recurrence" in families:
        source = "recurrence-local-whisper-neutral-conflict-consensus"
    else:
        return None
    artifact = {
        "mode": _NEUTRAL_BOUNDARY_ARBITRATION_MODE,
        "probe_revision": probe_revision,
        "neutral_outcome_revision": outcome.outcome_revision,
        "neutral_evidence_revision": outcome.evidence_revision,
        "parent_candidate_id": parent.candidate_id,
        "parent_generation_revision": parent.generation_revision,
        "cluster_revisions": copy.deepcopy(cluster.get("cluster_revisions", [])),
        "cluster_anchor_centiseconds": cluster.get("anchor_centiseconds"),
        "onset_time": round(float(outcome.onset_time), 6),
    }
    return make_timing_candidate(
        entry_index=entry_index,
        entry_text=entry_sung_text(entries[entry_index]),
        source=source,
        raw_time=parent.raw_time,
        spans=parent.spans,
        confidence=min(parent.confidence, float(outcome.score)),
        identity_support="supported",
        sequence_support="supported",
        acoustic_support="unavailable",
        direct_onset_support="supported",
        direct_onset_time=float(outcome.onset_time),
        direct_onset_source_artifact=artifact,
        direct_onset_evidence_producer="local-whisper-aligner",
        direct_onset_evidence_kind="cross-backend-onset",
        direct_onset_evidence_independent=True,
        occurrence_evidence=occurrence,
        independent_content_identity=True,
        current=False,
        prefix_scope=parent.prefix_scope,
        right_edge_pileup=False,
        window_truncated=False,
        source_artifact=artifact,
    )


def _update_projection_seed_for_arbitrated_selection(
    seed: dict[str, object],
    selection: BackendSelection,
    arbitration_payload: dict[str, object],
) -> dict[str, object]:
    rebound = copy.deepcopy(seed)
    rebound["selection"] = central_candidate_selection_payload(selection)
    rows = rebound.get("candidates")
    by_backend = {item.backend: item for item in selection.candidates}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            backend = str(row.get("backend") or "")
            evaluated = by_backend.get(backend)
            if evaluated is None:
                continue
            row["selection_status"] = (
                "selected" if evaluated is selection.selected else "nonselected"
            )
            row["evaluation"] = central_backend_summary_payload(evaluated)
            row["selection_reason"] = central_selection_reason(selection)
    rebound["neutral_boundary_arbitration"] = copy.deepcopy(arbitration_payload)
    return rebound


_NEUTRAL_ARBITRATION_FINAL_SEMANTIC_FIELDS = (
    "timestamp",
    "canonical_written_centiseconds",
    "selected_candidate_id",
    "provisional_candidate_id",
    "candidate_source",
    "timestamp_status",
    "timing_trusted",
    "content_trusted",
    "content_trust_source",
    "review_required",
    "review_reason_classes",
    "overall_trusted",
    "verification_status",
    "confidence",
)

# ``review_reason_classes`` explains *why* an already-review-required row is
# reviewable.  It is not itself an authority verdict.  A predecessor correction
# may legitimately rebind that taxonomy on a successor while leaving timestamp,
# candidate, trust and verification verdict unchanged (an accepted predecessor-correction
# behavior already demonstrates this dependency class).  Keep the field in diagnostics, but do not let a
# reason-class-only dependent rebind veto an otherwise safe target.
_NEUTRAL_ARBITRATION_HARD_COLLATERAL_FIELDS = frozenset(
    field
    for field in _NEUTRAL_ARBITRATION_FINAL_SEMANTIC_FIELDS
    if field != "review_reason_classes"
)


def _neutral_arbitration_final_semantic_rows(
    report: dict[str, object],
) -> tuple[dict[str, object], ...]:
    """Return the post-consumer semantic surface used by the replay safety gate.

    R2R FINAL established that producer/Central correctness is insufficient when
    downstream suspicious/material-review consumers can resurrect a conflict.
    Neutral arbitration therefore compares the fully projected assignment
    semantics, not only the selected candidate/time/timing-audit tuple.
    Evidence/provenance metadata may legitimately rebind on successor rows and
    is intentionally excluded unless it changes a final semantic field below.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        raise LrcError("Neutral arbitration projection has no assignments")
    rows: list[dict[str, object]] = []
    for index, item in enumerate(assignments):
        if not isinstance(item, dict) or item.get("entry") != index + 1:
            raise LrcError("Neutral arbitration projection assignment is malformed")
        rows.append({
            "entry": index + 1,
            **{key: copy.deepcopy(item.get(key)) for key in _NEUTRAL_ARBITRATION_FINAL_SEMANTIC_FIELDS},
        })
    return tuple(rows)


def _neutral_arbitration_project_selection(
    report: dict[str, object],
    selection: BackendSelection,
) -> tuple[dict[str, object], FinalTimingState]:
    """Privately run the ordinary final projection for one candidate selection.

    This uses the same Central commit and downstream review consumers as
    production, but only on a deep copy.  It cannot write production state.
    """
    projected = copy.deepcopy(report)
    state = _apply_evaluated_backend_projection(selection, projected)
    assert_report_timestamp_equality(projected, state)
    update_report_confidence_metrics(projected)
    assert_report_review_contract(projected)
    return projected, state


def _neutral_arbitration_selection_with_revised_eval(
    selection: BackendSelection,
    original_selected: EvaluatedBackendTiming,
    revised_eval: EvaluatedBackendTiming,
) -> BackendSelection:
    replaced = tuple(
        revised_eval if item is original_selected else item
        for item in selection.candidates
    )
    return select_evaluated_backend(replaced)


def _neutral_arbitration_replay_gate(
    before_report: dict[str, object],
    after_report: dict[str, object],
    target_indexes: frozenset[int] | set[int],
    outcome_map: Mapping[int, LocalWhisperOutcome],
    tolerance_centiseconds: int,
) -> tuple[list[int], list[int], dict[int, tuple[str, ...]]]:
    """Validate one private replay through the complete final consumer path.

    ``target_indexes`` is the exact set of candidate graphs intentionally
    changed.  Any non-target mutation of the hard final verdict surface is
    collateral.  Review/overall/verification state remains blocking so the
    replay cannot repeat the R2R producer-success/consumer-mismatch failure
    mode; a review-reason taxonomy-only rebind is diagnostic and non-blocking.
    """
    targets = set(target_indexes)
    before_rows = _neutral_arbitration_final_semantic_rows(before_report)
    after_rows = _neutral_arbitration_final_semantic_rows(after_report)
    if len(before_rows) != len(after_rows):
        raise LrcError("Neutral arbitration projection changed assignment count")
    collateral: list[int] = []
    failed_targets: list[int] = []
    changed_fields: dict[int, tuple[str, ...]] = {}
    for index, (before, after) in enumerate(zip(before_rows, after_rows)):
        fields = tuple(
            key
            for key in _NEUTRAL_ARBITRATION_FINAL_SEMANTIC_FIELDS
            if before.get(key) != after.get(key)
        )
        if fields:
            changed_fields[index + 1] = fields
        blocking_fields = tuple(
            key for key in fields
            if key in _NEUTRAL_ARBITRATION_HARD_COLLATERAL_FIELDS
        )
        if blocking_fields and index not in targets:
            collateral.append(index + 1)
        if index not in targets:
            continue
        outcome = outcome_map.get(index)
        after_cs = after.get("canonical_written_centiseconds")
        if (
            outcome is None
            or outcome.onset_time is None
            or after.get("timestamp_status") != "selected_valid"
            or after.get("timing_trusted") is not True
            or not isinstance(after_cs, int)
            or abs(
                after_cs
                - canonicalize_lrc_time(float(outcome.onset_time)).centiseconds
            ) > tolerance_centiseconds
        ):
            failed_targets.append(index + 1)
    return collateral, failed_targets, changed_fields


def _precommit_neutral_boundary_arbitration(
    audio_path: Path,
    entries: list[LyricEntry],
    report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
    selection: BackendSelection,
) -> tuple[BackendSelection, dict[str, object], dict[str, object]]:
    """Acquire neutral boundary evidence once, then replay only the selected Central graph.

    The discovery commit occurs on a private report copy.  Production state is
    unchanged until a revised ``EvaluatedBackendTiming`` passes all fail-closed
    invariants and is committed through the ordinary Central owner.
    """
    shadow = copy.deepcopy(report)
    shadow_state = _apply_evaluated_backend_projection(selection, shadow)
    assert_report_timestamp_equality(shadow, shadow_state)
    update_report_confidence_metrics(shadow)
    seed = shadow.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    if isinstance(seed, dict):
        shadow[_BACKEND_PROVENANCE_PROJECTION_KEY] = _finalize_backend_provenance_projection(shadow, seed, selection)
        _validate_backend_provenance_projection(shadow)
    shadow[_UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY] = (
        _build_unresolved_evidence_requirements_projection(shadow, shadow_state)
    )
    probe_payload, outcome_map = run_r2t_unified_neutral_conflict_probe(
        audio_path, entries, shadow, duration, args, shadow_state
    )
    triggers = _neutral_probe_trigger_rows(probe_payload)
    base_payload: dict[str, object] = {
        "schema_version": 1,
        "mode": _NEUTRAL_BOUNDARY_ARBITRATION_MODE,
        "probe_revision": probe_payload.get("probe_revision"),
        "trigger_count": len(triggers),
        "trigger_entries": sorted(
            int(item.get("entry")) for item in triggers if isinstance(item.get("entry"), int)
        ),
        "status": "NO_TRIGGER" if not triggers else "EVALUATING",
        "central_replayed": False,
        "candidate_admission_count": 0,
        "falsified_candidate_count": 0,
        "replay_safety_scope": "full-final-assignment-semantics-v1",
        "consumer_parity_guard": True,
        "entries": [],
    }
    if not triggers:
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, probe_payload, base_payload

    candidate_map = _neutral_arbitration_candidate_map(selection)
    selected_eval = selection.selected
    revised_sets = [list(items) for items in selected_eval.candidate_sets]
    tolerance_cs = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    diagnostics: list[dict[str, object]] = []
    targeted_indexes: set[int] = set()

    for row in triggers:
        entry_number = row.get("entry")
        if not isinstance(entry_number, int):
            continue
        index = entry_number - 1
        if not 0 <= index < len(revised_sets):
            continue
        outcome = outcome_map.get(index)
        cluster = _neutral_arbitration_matching_cluster(row)
        if outcome is None or cluster is None or outcome.onset_time is None or outcome.score is None:
            continue
        original_decision = selected_eval.state.decisions[index]
        selected_candidate = original_decision.candidate
        selected_families = _neutral_arbitration_support_families(selected_candidate)
        if (
            "whisper-family" not in selected_families
            or "ctc" in selected_families
            or "recurrence" in selected_families
            or float(outcome.score) < selected_candidate.confidence
        ):
            continue
        onset_cs = canonicalize_lrc_time(float(outcome.onset_time)).centiseconds
        selected_cs = original_decision.written_time.centiseconds
        if abs(onset_cs - selected_cs) <= tolerance_cs:
            continue

        member_ids = cluster.get("member_candidate_ids")
        if not isinstance(member_ids, list):
            continue
        valid_candidate_ids = _neutral_arbitration_valid_candidate_ids(selection, index)
        parents: list[TimingCandidate] = []
        for candidate_id in member_ids:
            if not isinstance(candidate_id, str):
                continue
            parent = candidate_map.get(candidate_id)
            if parent is None or parent.entry_index != index:
                continue
            families = _neutral_arbitration_support_families(parent)
            if "whisper-family" in families or not ({"ctc", "recurrence"} & set(families)):
                continue
            if abs(parent.written_time.centiseconds - onset_cs) > tolerance_cs:
                continue
            if parent.identity_support != "supported" or parent.sequence_support != "supported":
                continue
            parents.append(parent)
        if not parents:
            continue

        # Fail closed if the selected boundary already has an independent
        # non-Whisper witness anywhere in the frozen backend corpus.
        independent_selected_support = [
            candidate for candidate in candidate_map.values()
            if candidate.entry_index == index
            and candidate.candidate_id in valid_candidate_ids
            and abs(candidate.written_time.centiseconds - selected_cs) <= tolerance_cs
            and "whisper-family" not in _neutral_arbitration_support_families(candidate)
            and bool({"ctc", "recurrence"} & set(_neutral_arbitration_support_families(candidate)))
            and candidate.identity_support == "supported"
            and candidate.sequence_support == "supported"
        ]
        if independent_selected_support:
            continue

        consensus: list[TimingCandidate] = []
        for parent in sorted(parents, key=lambda item: item.candidate_id):
            candidate = _neutral_arbitration_consensus_candidate(
                entries=entries,
                entry_index=index,
                parent=parent,
                outcome=outcome,
                cluster=cluster,
                probe_revision=str(probe_payload.get("probe_revision") or ""),
            )
            if candidate is not None:
                consensus.append(candidate)
        if not consensus:
            continue

        falsified = [
            candidate for candidate in revised_sets[index]
            if abs(candidate.written_time.centiseconds - selected_cs) <= tolerance_cs
            and "whisper-family" in _neutral_arbitration_support_families(candidate)
            and "ctc" not in _neutral_arbitration_support_families(candidate)
            and "recurrence" not in _neutral_arbitration_support_families(candidate)
        ]
        falsified_ids = {item.candidate_id for item in falsified}
        if selected_candidate.candidate_id not in falsified_ids:
            continue
        survivors = [
            item for item in revised_sets[index] if item.candidate_id not in falsified_ids
        ]
        merged = {item.candidate_id: item for item in [*survivors, *consensus]}
        revised_sets[index] = [merged[key] for key in sorted(merged)]
        targeted_indexes.add(index)
        diagnostics.append({
            "entry": entry_number,
            "old_selected_candidate_id": selected_candidate.candidate_id,
            "old_selected_centiseconds": selected_cs,
            "neutral_onset_centiseconds": onset_cs,
            "cluster_anchor_centiseconds": cluster.get("anchor_centiseconds"),
            "parent_candidate_ids": sorted(item.candidate_id for item in parents),
            "admitted_candidate_ids": sorted(item.candidate_id for item in consensus),
            "falsified_candidate_ids": sorted(falsified_ids),
            "neutral_evidence_revision": outcome.evidence_revision,
            "neutral_outcome_revision": outcome.outcome_revision,
        })

    if not targeted_indexes:
        base_payload["status"] = "NO_SAFE_ARBITRATION"
        base_payload["entries"] = diagnostics
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, probe_payload, base_payload

    generation_diagnostics = thaw_generation_diagnostics(selected_eval)
    generation_diagnostics[_NEUTRAL_BOUNDARY_ARBITRATION_KEY] = {
        "mode": _NEUTRAL_BOUNDARY_ARBITRATION_MODE,
        "probe_revision": probe_payload.get("probe_revision"),
        "entries": copy.deepcopy(diagnostics),
    }
    revised_eval = evaluate_backend_timing(
        selected_eval.backend,
        tuple(tuple(items) for items in revised_sets),
        report,
        generation_diagnostics,
        arbiter_mode=selected_eval.arbiter_mode,
        timing_source_policy=selected_eval.timing_source_policy,
    )

    # First try the full blind trigger set exactly as before.  When several
    # independent neutral positives exist in one song, one unsafe trigger must
    # not mask another demonstrably safe trigger.  If the combined private
    # replay fails, isolate every trigger against the same immutable pre-replay
    # Central state, retain only individually safe triggers, then perform one
    # final consolidated Central replay over that safe subset.  Production is
    # still committed only once by the ordinary owner.
    revised_selection = _neutral_arbitration_selection_with_revised_eval(
        selection, selected_eval, revised_eval
    )
    initial_backend_flip = revised_selection.selected.backend != selected_eval.backend
    if initial_backend_flip:
        collateral = []
        failed_targets = sorted(index + 1 for index in targeted_indexes)
        initial_changed_fields: dict[int, tuple[str, ...]] = {}
    else:
        revised_shadow, _revised_shadow_state = _neutral_arbitration_project_selection(
            report, revised_selection
        )
        collateral, failed_targets, initial_changed_fields = (
            _neutral_arbitration_replay_gate(
                shadow,
                revised_shadow,
                targeted_indexes,
                outcome_map,
                tolerance_cs,
            )
        )
    initial_combined_failure = {
        "collateral_entries": list(collateral),
        "failed_target_entries": list(failed_targets),
        "backend_flip": initial_backend_flip,
        "candidate_backend": revised_selection.selected.backend,
        "changed_final_semantic_fields": {
            str(entry): list(fields)
            for entry, fields in sorted(initial_changed_fields.items())
        },
    }
    isolation_rows: list[dict[str, object]] = []
    rejected_target_indexes: set[int] = set()

    if collateral or failed_targets:
        independently_safe: set[int] = set()
        diagnostics_by_index = {
            int(item["entry"]) - 1: item
            for item in diagnostics
            if isinstance(item.get("entry"), int)
        }
        for index in sorted(targeted_indexes):
            isolated_sets = [list(items) for items in selected_eval.candidate_sets]
            isolated_sets[index] = list(revised_sets[index])
            isolated_generation_diagnostics = thaw_generation_diagnostics(selected_eval)
            isolated_generation_diagnostics[_NEUTRAL_BOUNDARY_ARBITRATION_KEY] = {
                "mode": _NEUTRAL_BOUNDARY_ARBITRATION_MODE,
                "probe_revision": probe_payload.get("probe_revision"),
                "entries": copy.deepcopy(
                    [diagnostics_by_index[index]] if index in diagnostics_by_index else []
                ),
                "private_replay_scope": "single-trigger-isolation",
            }
            isolated_eval = evaluate_backend_timing(
                selected_eval.backend,
                tuple(tuple(items) for items in isolated_sets),
                report,
                isolated_generation_diagnostics,
                arbiter_mode=selected_eval.arbiter_mode,
                timing_source_policy=selected_eval.timing_source_policy,
            )
            isolated_selection = _neutral_arbitration_selection_with_revised_eval(
                selection, selected_eval, isolated_eval
            )
            isolated_backend_flip = (
                isolated_selection.selected.backend != selected_eval.backend
            )
            if isolated_backend_flip:
                isolated_collateral = []
                isolated_failed = [index + 1]
                isolated_changed_fields: dict[int, tuple[str, ...]] = {}
            else:
                isolated_shadow, _isolated_shadow_state = (
                    _neutral_arbitration_project_selection(report, isolated_selection)
                )
                isolated_collateral, isolated_failed, isolated_changed_fields = (
                    _neutral_arbitration_replay_gate(
                        shadow,
                        isolated_shadow,
                        {index},
                        outcome_map,
                        tolerance_cs,
                    )
                )
            safe = not isolated_collateral and not isolated_failed
            isolation_rows.append({
                "entry": index + 1,
                "status": "SAFE" if safe else "REJECTED",
                "collateral_entries": isolated_collateral,
                "failed_target_entries": isolated_failed,
                "backend_flip": isolated_backend_flip,
                "candidate_backend": isolated_selection.selected.backend,
                "changed_final_semantic_fields": {
                    str(entry): list(fields)
                    for entry, fields in sorted(isolated_changed_fields.items())
                },
                "evaluation_revision": isolated_eval.evaluation_revision,
                "selection_revision": isolated_selection.selection_revision,
            })
            if safe:
                independently_safe.add(index)
            else:
                rejected_target_indexes.add(index)

        if not independently_safe:
            for item in diagnostics:
                if isinstance(item.get("entry"), int):
                    item["arbitration_disposition"] = "rejected-private-replay"
            base_payload.update({
                "status": "FAIL_CLOSED",
                "collateral_entries": collateral,
                "failed_target_entries": failed_targets,
                "entries": diagnostics,
                "initial_combined_replay": initial_combined_failure,
                "trigger_isolation": isolation_rows,
                "applied_trigger_entries": [],
                "rejected_trigger_entries": sorted(index + 1 for index in targeted_indexes),
            })
            base_payload["revision"] = _timing_content_digest(base_payload)
            return selection, probe_payload, base_payload

        consolidated_sets = [list(items) for items in selected_eval.candidate_sets]
        for index in independently_safe:
            consolidated_sets[index] = list(revised_sets[index])
        consolidated_generation_diagnostics = thaw_generation_diagnostics(selected_eval)
        consolidated_generation_diagnostics[_NEUTRAL_BOUNDARY_ARBITRATION_KEY] = {
            "mode": _NEUTRAL_BOUNDARY_ARBITRATION_MODE,
            "probe_revision": probe_payload.get("probe_revision"),
            "entries": copy.deepcopy([
                item for item in diagnostics
                if isinstance(item.get("entry"), int)
                and int(item["entry"]) - 1 in independently_safe
            ]),
            "private_replay_scope": "safe-subset-consolidation",
        }
        consolidated_eval = evaluate_backend_timing(
            selected_eval.backend,
            tuple(tuple(items) for items in consolidated_sets),
            report,
            consolidated_generation_diagnostics,
            arbiter_mode=selected_eval.arbiter_mode,
            timing_source_policy=selected_eval.timing_source_policy,
        )
        consolidated_selection = _neutral_arbitration_selection_with_revised_eval(
            selection, selected_eval, consolidated_eval
        )
        consolidated_backend_flip = (
            consolidated_selection.selected.backend != selected_eval.backend
        )
        if consolidated_backend_flip:
            consolidated_collateral = []
            consolidated_failed = sorted(index + 1 for index in independently_safe)
            consolidated_changed_fields: dict[int, tuple[str, ...]] = {}
        else:
            consolidated_shadow, _consolidated_shadow_state = (
                _neutral_arbitration_project_selection(report, consolidated_selection)
            )
            (
                consolidated_collateral,
                consolidated_failed,
                consolidated_changed_fields,
            ) = _neutral_arbitration_replay_gate(
                shadow,
                consolidated_shadow,
                independently_safe,
                outcome_map,
                tolerance_cs,
            )
        if consolidated_collateral or consolidated_failed:
            for item in diagnostics:
                if isinstance(item.get("entry"), int):
                    item["arbitration_disposition"] = "rejected-consolidated-replay"
            base_payload.update({
                "status": "FAIL_CLOSED_SAFE_SUBSET",
                "collateral_entries": consolidated_collateral,
                "failed_target_entries": consolidated_failed,
                "backend_flip": consolidated_backend_flip,
                "candidate_backend": consolidated_selection.selected.backend,
                "changed_final_semantic_fields": {
                    str(entry): list(fields)
                    for entry, fields in sorted(consolidated_changed_fields.items())
                },
                "entries": diagnostics,
                "initial_combined_replay": initial_combined_failure,
                "trigger_isolation": isolation_rows,
                "applied_trigger_entries": [],
                "rejected_trigger_entries": sorted(index + 1 for index in targeted_indexes),
            })
            base_payload["revision"] = _timing_content_digest(base_payload)
            return selection, probe_payload, base_payload

        targeted_indexes = independently_safe
        revised_eval = consolidated_eval
        revised_selection = consolidated_selection
        collateral = []
        failed_targets = []
    else:
        rejected_target_indexes = set()

    if revised_selection.selected.backend != selected_eval.backend:
        base_payload.update({
            "status": "FAIL_CLOSED_BACKEND_FLIP",
            "entries": diagnostics,
            "candidate_backend": revised_selection.selected.backend,
        })
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, probe_payload, base_payload

    for item in diagnostics:
        index = int(item["entry"]) - 1
        if index not in targeted_indexes:
            item["arbitration_disposition"] = "rejected-private-replay"
            continue
        decision = revised_eval.state.decisions[index]
        audit = revised_eval.state.audits[index]
        item["arbitration_disposition"] = "applied"
        item["new_selected_candidate_id"] = decision.candidate.candidate_id
        item["new_selected_source"] = decision.candidate.source
        item["new_selected_centiseconds"] = decision.written_time.centiseconds
        item["new_timing_trusted"] = audit.timing_trusted

    applied_diagnostics = [
        item for item in diagnostics
        if isinstance(item.get("entry"), int)
        and int(item["entry"]) - 1 in targeted_indexes
    ]
    payload = {
        **base_payload,
        "status": "APPLIED",
        "central_replayed": True,
        "candidate_admission_count": sum(
            len(item["admitted_candidate_ids"]) for item in applied_diagnostics
        ),
        "falsified_candidate_count": sum(
            len(item["falsified_candidate_ids"]) for item in applied_diagnostics
        ),
        "entries": diagnostics,
        "old_evaluation_revision": selected_eval.evaluation_revision,
        "new_evaluation_revision": revised_eval.evaluation_revision,
        "old_selection_revision": selection.selection_revision,
        "new_selection_revision": revised_selection.selection_revision,
        "collateral_entries": [],
        "applied_trigger_entries": sorted(index + 1 for index in targeted_indexes),
        "rejected_trigger_entries": sorted(
            int(item["entry"])
            for item in diagnostics
            if isinstance(item.get("entry"), int)
            and int(item["entry"]) - 1 not in targeted_indexes
        ),
        "initial_combined_replay": initial_combined_failure,
        "trigger_isolation": isolation_rows,
    }
    payload["revision"] = _timing_content_digest(payload)
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    if isinstance(seed, dict):
        report[_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = (
            _update_projection_seed_for_arbitrated_selection(seed, revised_selection, payload)
        )
    return revised_selection, probe_payload, payload


def _local_whisper_request_execution_revision(request: LocalWhisperRequest) -> str:
    return request.execution_revision or _local_whisper_execution_request_revision(
        request.__dict__
    )


def _local_whisper_cache_key(
    request: LocalWhisperRequest,
    capability: LocalWhisperCapability,
) -> str:
    # Keep the serialized key shape stable so K2 cache entries remain reusable.
    return "local-whisper:" + _timing_content_digest({
        "request_revision": _local_whisper_request_execution_revision(request),
        "capability_revision": capability.capability_revision,
        "protocol_revision": capability.protocol_revision,
    })


def _local_whisper_outcome_for_request(
    outcome: LocalWhisperOutcome,
    request: LocalWhisperRequest,
) -> LocalWhisperOutcome:
    if outcome.request_revision == request.request_revision:
        return outcome
    base = {
        "entry_index": outcome.entry_index,
        "request_revision": request.request_revision,
        "status": outcome.status,
        "reason": outcome.reason,
        "onset_time": outcome.onset_time,
        "score": outcome.score,
        "evidence_revision": outcome.evidence_revision,
        "occurrence_binding": outcome.occurrence_binding,
        "raw_transcript_revision": outcome.raw_transcript_revision,
        "aligned_transcript_revision": outcome.aligned_transcript_revision,
    }
    return LocalWhisperOutcome(
        **base,
        outcome_revision=_timing_content_digest(base),
    )


def _shift_local_assignment_to_global(
    assignment: object,
    offset: float,
) -> dict[str, object] | None:
    if not isinstance(assignment, dict):
        return None
    shifted = copy.deepcopy(assignment)
    try:
        shifted["timestamp"] = round(float(assignment.get("timestamp")) + offset, 6)
    except (TypeError, ValueError):
        return None
    return shifted


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


def bind_runtime_local_whisper_evidence(
    report: dict[str, object],
    entries: list[LyricEntry],
    capability: LocalWhisperCapability | None,
    requests: tuple[LocalWhisperRequest, ...],
    outcomes: tuple[LocalWhisperOutcome, ...],
) -> None:
    if capability is None:
        return
    request_by_revision = {item.request_revision: item for item in requests}

    # Fixed-point recollection may legitimately stop requesting an entry after a
    # first-round local observation changes the candidate graph.  The observation
    # itself is immutable audio/lyric evidence and must not disappear merely
    # because the next planner round no longer needs to execute that request.
    # Seed from canonical-valid prior runtime evidence, then monotonically merge
    # any new observations.  Tampered/stale evidence fails the canonical accessor
    # and is therefore not retained.
    prior_runtime: dict[int, dict[str, object]] = {}
    previous_runtime = report.get(_RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY)
    if isinstance(previous_runtime, dict):
        for entry_index in sorted(previous_runtime):
            if not isinstance(entry_index, int) or not 0 <= entry_index < len(entries):
                continue
            retained = runtime_local_whisper_evidence_for_entry(report, entries, entry_index)
            if retained is not None:
                prior_runtime[entry_index] = retained
    retained_before_merge = len(prior_runtime)

    observations: dict[int, list[dict[str, object]]] = {
        index: [dict(value)] for index, value in prior_runtime.items()
    }
    for outcome in outcomes:
        if outcome.status != "evidence" or outcome.evidence_revision is None:
            continue
        request = request_by_revision.get(outcome.request_revision)
        if request is None or outcome.score is None:
            continue
        lyric_sha = hashlib.sha256(
            normalize_match_text(entry_sung_text(entries[outcome.entry_index])).encode("utf-8")
        ).hexdigest()
        payload = {
            "entry_index": outcome.entry_index,
            "lyric_sha256": lyric_sha,
            "onset_time": (
                round(float(outcome.onset_time), 6)
                if outcome.onset_time is not None else None
            ),
            "score": round(float(outcome.score), 6),
            "occurrence_binding": outcome.occurrence_binding,
            "request_revision": request.request_revision,
            "capability_revision": capability.capability_revision,
            "audio_revision": capability.audio_sha256,
            "window_start": round(request.window_start, 6),
            "window_end": round(request.window_end, 6),
            "raw_transcript_revision": outcome.raw_transcript_revision,
            "aligned_transcript_revision": outcome.aligned_transcript_revision,
            "producer": "local-whisper-aligner",
        }
        evidence = {
            **payload,
            "evidence_revision": _timing_content_digest(payload),
        }
        observations.setdefault(outcome.entry_index, []).append(evidence)

    composite_count = 0
    # A narrow probe may contain the clean forced-character onset while a wider
    # probe contains the sequence context needed to bind a repeated lyric.  They
    # are complementary observations from the same immutable audio/lyric source,
    # not independent votes.  Merge only when both observations are revision-
    # bound, their target times agree inside the existing association tolerance,
    # and the onset lies inside the context bracket.  The timestamp is never
    # averaged or moved: it remains the original direct onset observation.
    for entry_index, items in list(observations.items()):
        onset_items = [item for item in items if item.get("onset_time") is not None]
        context_items = [
            item for item in items
            if _occurrence_binding_has_strong_context(item.get("occurrence_binding"))
        ]
        composites: list[dict[str, object]] = []
        for onset_item in onset_items:
            try:
                onset_time = float(onset_item.get("onset_time"))
                onset_score = float(onset_item.get("score"))
            except (TypeError, ValueError):
                continue
            for context_item in context_items:
                context_binding = context_item.get("occurrence_binding")
                if context_binding == onset_item.get("occurrence_binding"):
                    continue
                if (
                    onset_item.get("lyric_sha256") != context_item.get("lyric_sha256")
                    or onset_item.get("capability_revision") != context_item.get("capability_revision")
                    or onset_item.get("audio_revision") != context_item.get("audio_revision")
                ):
                    continue
                try:
                    context_time = float(context_binding.get("timestamp"))  # type: ignore[union-attr]
                    context_score = float(context_item.get("score"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if (
                    not math.isfinite(onset_time)
                    or not math.isfinite(context_time)
                    or abs(onset_time - context_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
                ):
                    continue
                left_bound, right_bound = _occurrence_binding_time_bounds(
                    context_binding, entry_index
                )
                if left_bound is not None and onset_time <= left_bound - HOP_SECONDS:
                    continue
                if right_bound is not None and onset_time >= right_bound + HOP_SECONDS:
                    continue
                composite_payload = {
                    "entry_index": entry_index,
                    "lyric_sha256": onset_item.get("lyric_sha256"),
                    "onset_time": round(onset_time, 6),
                    "score": round(min(onset_score, context_score), 6),
                    "occurrence_binding": context_binding,
                    "request_revision": onset_item.get("request_revision"),
                    "capability_revision": onset_item.get("capability_revision"),
                    "audio_revision": onset_item.get("audio_revision"),
                    "window_start": round(float(onset_item.get("window_start")), 6),
                    "window_end": round(float(onset_item.get("window_end")), 6),
                    "raw_transcript_revision": onset_item.get("raw_transcript_revision"),
                    "aligned_transcript_revision": onset_item.get("aligned_transcript_revision"),
                    "producer": "local-whisper-aligner",
                    "composite_onset_evidence_revision": onset_item.get("evidence_revision"),
                    "composite_context_evidence_revision": context_item.get("evidence_revision"),
                }
                composite = {
                    **composite_payload,
                    "evidence_revision": _timing_content_digest(composite_payload),
                }
                composites.append(composite)
        if composites:
            unique = {item["evidence_revision"]: item for item in composites}
            observations[entry_index].extend(unique.values())
            composite_count += len(unique)

    def evidence_rank(item: dict[str, object]) -> tuple[int, int, float, str]:
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return (
            int(item.get("onset_time") is not None),
            int(_occurrence_binding_has_strong_context(item.get("occurrence_binding"))),
            score,
            str(item.get("evidence_revision") or ""),
        )

    runtime: dict[int, dict[str, object]] = {}
    for entry_index, items in observations.items():
        if not items:
            continue
        runtime[entry_index] = max(items, key=evidence_rank)

    report[_RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY] = runtime
    report["local_whisper_evidence"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "protocol_revision": capability.protocol_revision,
        "candidate_count": len(runtime),
        "retained_from_prior_collect_count": retained_before_merge,
        "composite_evidence_count": composite_count,
        "entries": [index + 1 for index in sorted(runtime)],
    }


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


def _augment_candidates_with_local_whisper_consensus(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Fuse local Whisper onset/identity into existing candidates without moving time."""
    augmented = [list(items) for items in candidate_sets]
    for index, parents in enumerate(candidate_sets):
        evidence = runtime_local_whisper_evidence_for_entry(report, entries, index)
        if evidence is None:
            continue
        raw_onset_time = evidence.get("onset_time")
        if raw_onset_time is None:
            continue
        onset_time = float(raw_onset_time)
        evidence_score = float(evidence["score"])
        for parent in parents:
            families = _candidate_support_families(parent)
            # Local Whisper is one Whisper-family observation. Never use a
            # Whisper-derived parent to manufacture a second independent vote.
            if "whisper-family" in families:
                continue
            if (
                parent.identity_support != "supported"
                or parent.sequence_support != "supported"
                or parent.right_edge_pileup
                or parent.window_truncated
                or abs(parent.written_time.seconds - onset_time) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            ):
                continue
            if "ctc" in families:
                source = "ctc-local-whisper-independent-consensus"
            elif "recurrence" in families:
                source = "recurrence-local-whisper-independent-consensus"
            else:
                source = "local-whisper-independent-consensus"
            artifact = {
                "producer": "local-whisper-aligner",
                "local_whisper_evidence_revision": evidence["evidence_revision"],
                "parent_candidate_id": parent.candidate_id,
                "parent_generation_revision": parent.generation_revision,
                "parent_source": parent.source,
                "parent_time": round(parent.raw_time, 6),
                "onset_time": round(onset_time, 6),
                "runtime_only": True,
            }
            augmented[index].append(make_timing_candidate(
                entry_index=index,
                entry_text=entry_sung_text(entries[index]),
                source=source,
                raw_time=parent.raw_time,
                spans=parent.spans,
                confidence=min(parent.confidence, evidence_score),
                identity_support="supported",
                sequence_support="supported",
                acoustic_support="unavailable",
                direct_onset_support="supported",
                direct_onset_time=onset_time,
                direct_onset_source_artifact=artifact,
                direct_onset_evidence_producer="local-whisper-aligner",
                direct_onset_evidence_kind="cross-backend-onset",
                direct_onset_evidence_independent=True,
                occurrence_evidence=candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="local-whisper-aligner",
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


def _apply_context_conditioned_whisper_hypothesis_arbitration(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
) -> tuple[tuple[tuple[TimingCandidate, ...], ...], tuple[dict[str, object], ...]]:
    """Invalidate weaker whole-song Whisper hypotheses with stronger bound context.

    This is *not* majority voting.  A global raw-ASR occurrence may be removed only
    when all of the following are provenance-bound facts:

    1. bounded local Whisper re-identifies the lyric inside a trusted full or
       sparse left/target/right sequence bracket and exposes either a direct aligned
       onset or an exact revision-bound independent vocal onset;
    2. an independent CTC+local-Whisper composite converges at that onset inside the
       existing 0.18 s tolerance;
    3. the global raw occurrence is only ``global-unique`` (no sequence bracket), is
       distant from the bounded onset, and is not independently corroborated by CTC
       or recurrence near its own time;
    4. the bounded observation is at least as confident as the raw identity row.

    The losing timestamp is not overwritten.  Its candidate is excluded from the
    central valid graph with an explicit diagnostic that binds both evidence
    revisions.  If any premise is absent, nothing is invalidated and fail-closed
    review behavior is preserved.
    """
    revised: list[tuple[TimingCandidate, ...]] = []
    diagnostics: list[dict[str, object]] = []
    tolerance = RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS

    for index, candidates in enumerate(candidate_sets):
        local = runtime_local_whisper_evidence_for_entry(report, entries, index)
        raw_identity = runtime_raw_content_identity_for_entry(report, entries, index)
        if not isinstance(local, dict) or not isinstance(raw_identity, dict):
            revised.append(candidates)
            continue
        local_binding = local.get("occurrence_binding")
        raw_binding = raw_identity.get("occurrence_binding")
        if not (
            _occurrence_binding_has_strong_context(local_binding)
            and isinstance(raw_binding, dict)
            and raw_binding.get("mode") == "global-unique"
        ):
            revised.append(candidates)
            continue
        try:
            local_score = float(local.get("score"))
            raw_time = float(raw_identity.get("raw_time"))
            raw_score = float(raw_identity.get("raw_score"))
        except (TypeError, ValueError):
            revised.append(candidates)
            continue

        # Independent corroboration must already exist as an actual composite;
        # raw CTC proximity or generic acoustic activity is not inferred here.
        raw_local_onset = local.get("onset_time")
        if raw_local_onset is not None:
            try:
                local_onset = float(raw_local_onset)
            except (TypeError, ValueError):
                revised.append(candidates)
                continue
            local_cluster = [
                candidate for candidate in candidates
                if candidate.source == "ctc-local-whisper-independent-consensus"
                and abs(candidate.written_time.seconds - local_onset) <= tolerance
                and candidate.identity_support == "supported"
                and candidate.sequence_support == "supported"
                and candidate_has_bound_direct_onset_evidence(candidate)
            ]
        else:
            # Occurrence-only Local Whisper can already have a boundary-grade
            # onset in the candidate graph via the independent vocal-fusion
            # producer.  Consume only an exact revision/binding match; never
            # synthesize an onset from the occurrence timestamp itself.
            local_revision = local.get("evidence_revision")
            binding_revision = (
                local_binding.get("binding_revision")
                if isinstance(local_binding, dict) else None
            )
            bound_vocal: list[tuple[TimingCandidate, float]] = []
            for candidate in candidates:
                if (
                    candidate.source != "ctc-local-whisper-vocal-independent-consensus"
                    or candidate.identity_support != "supported"
                    or candidate.sequence_support != "supported"
                    or not candidate_has_boundary_grade_direct_onset_evidence(candidate, entries)
                ):
                    continue
                occurrence = candidate.occurrence_evidence
                onset_evidence = candidate.direct_onset_evidence
                if (
                    occurrence is None
                    or onset_evidence is None
                    or occurrence.producer != "local-whisper-vocal-fusion"
                    or occurrence.upstream_evidence_revision != local_revision
                    or occurrence.binding_revision != binding_revision
                    or onset_evidence.producer != "local-whisper-vocal-fusion"
                ):
                    continue
                onset = float(onset_evidence.onset_time)
                if (
                    math.isfinite(onset)
                    and abs(candidate.written_time.seconds - onset) <= tolerance
                ):
                    bound_vocal.append((candidate, onset))
            canonical_onsets = {round(onset, 6) for _, onset in bound_vocal}
            if len(canonical_onsets) != 1:
                revised.append(candidates)
                continue
            local_onset = next(iter(canonical_onsets))
            local_cluster = [candidate for candidate, _ in bound_vocal]

        if (
            not all(math.isfinite(value) for value in (local_onset, local_score, raw_time, raw_score))
            or abs(local_onset - raw_time) <= tolerance
            or local_score < raw_score
            or not local_cluster
        ):
            revised.append(candidates)
            continue

        # If CTC or recurrence independently supports the raw cluster, it is a
        # genuine competing hypothesis and must remain review-required.
        raw_independent_support = any(
            abs(candidate.written_time.seconds - raw_time) <= tolerance
            and (
                "ctc" in _candidate_support_families(candidate)
                or "recurrence" in _candidate_support_families(candidate)
            )
            and "whisper-family" not in _candidate_support_families(candidate)
            for candidate in candidates
        )
        if raw_independent_support:
            revised.append(candidates)
            continue

        falsified = [
            candidate for candidate in candidates
            if "whisper-family" in _candidate_support_families(candidate)
            and "ctc" not in _candidate_support_families(candidate)
            and "recurrence" not in _candidate_support_families(candidate)
            and abs(candidate.written_time.seconds - raw_time) <= tolerance
            and abs(candidate.written_time.seconds - local_onset) > tolerance
        ]
        if not falsified:
            revised.append(candidates)
            continue
        falsified_ids = {candidate.candidate_id for candidate in falsified}
        survivors = tuple(candidate for candidate in candidates if candidate.candidate_id not in falsified_ids)
        if not survivors:
            revised.append(candidates)
            continue
        revised.append(survivors)
        diagnostics.append({
            "entry": index + 1,
            "reason": "bounded-sequence-context-falsifies-weaker-global-whisper-occurrence",
            "local_whisper_evidence_revision": local.get("evidence_revision"),
            "raw_content_identity_revision": raw_identity.get("evidence_revision"),
            "local_onset": round(local_onset, 6),
            "raw_identity_time": round(raw_time, 6),
            "local_score": round(local_score, 6),
            "raw_score": round(raw_score, 6),
            "context_binding_revision": local_binding.get("binding_revision"),
            "raw_binding_revision": raw_binding.get("binding_revision"),
            "corroborating_candidate_ids": sorted(candidate.candidate_id for candidate in local_cluster),
            "falsified_candidate_ids": sorted(falsified_ids),
            "max_delta_seconds": tolerance,
            "authority": "candidate-exclusion-only-central-still-commits",
        })

    return tuple(revised), tuple(diagnostics)


def _joint_previous_tail_evidence(
    request: CrossLineRetryRequest,
    capability: BoundedRetryCapability,
    rows: list[object],
    parsed_rows: list[tuple[TimingTokenSpan, ...]],
    previous_probe: TimingCandidate | None,
    previous_pair_tail: TailRegionEvidence | None,
    target_probe: TimingCandidate,
    target_regions: CandidateRegionEvidence,
    output_revision: str,
    pair_gap_seconds: float | None,
    edge: dict[str, object],
    touches_edge: bool,
    touches_left_edge: bool,
    strong_joint_boundary: bool,
) -> IndependentTailEvidence | None:
    """Retain strong joint previous-row tail evidence in the existing sidecar.

    The helper's previous probe is an immutable known-lyric observation.  It is
    independent of the selected predecessor timestamp; request planning may use
    selected timestamps to choose a bounded window, but this evidence revision is
    built only from helper output, immutable lyric/transcript binding, audio and
    the probe's terminal geometry.  Weak/edge/truncated pairs fail closed.
    """
    if (
        not strong_joint_boundary
        or previous_probe is None
        or previous_pair_tail is None
        or previous_pair_tail.validity != "valid"
        or previous_pair_tail.reliable_tail_start is None
        or previous_pair_tail.reliable_tail_end is None
        or not math.isfinite(float(previous_pair_tail.reliable_tail_start))
        or not math.isfinite(float(previous_pair_tail.reliable_tail_end))
        or not math.isfinite(float(pair_gap_seconds or 0.0))
        or touches_edge
        or touches_left_edge
        or bool(edge.get("right_edge_pileup"))
        or previous_probe.right_edge_pileup
        or previous_probe.window_truncated
    ):
        return None
    previous_local_index = dict(request.row_mapping).get(request.entry_index - 1)
    if (
        previous_local_index is None
        or not 0 <= previous_local_index < len(parsed_rows)
        or previous_local_index >= len(rows)
        or not isinstance(rows[previous_local_index], dict)
    ):
        return None
    previous_spans = parsed_rows[previous_local_index]
    previous_span_payload = [
        [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
        for span in previous_spans
    ]
    target_span_payload = [
        [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
        for span in target_probe.spans
    ]
    evidence_payload = {
        "schema": "joint-previous-tail-v1",
        "producer": "joint-ctc-boundary-previous-tail",
        "entry_index": request.entry_index - 1,
        "context_entry_indexes": list(request.context_entry_indexes),
        "row_mapping": [list(pair) for pair in request.row_mapping],
        "row_mapping_revision": request.row_mapping_revision,
        "previous_lyric_digest": (
            request.transcript_digests[previous_local_index]
            if previous_local_index < len(request.transcript_digests)
            else None
        ),
        "transcript_digests": list(request.transcript_digests),
        "transcript_digest": request.transcript_digest,
        "capability_revision": capability.capability_revision,
        "helper_sha256": capability.helper_sha256,
        "helper_protocol_revision": capability.helper_protocol_revision,
        "alignment_audio_revision": request.alignment_audio_revision,
        "retry_epoch_revision": request.retry_epoch_revision,
        "request_revision": request.request_revision,
        "output_revision": output_revision,
        "previous_probe_candidate_id": previous_probe.candidate_id,
        "previous_probe_generation_revision": previous_probe.generation_revision,
        "previous_probe_region_revision": analyze_candidate_regions(previous_probe).region_revision,
        "previous_spans_revision": _timing_content_digest(previous_span_payload),
        "previous_spans": previous_span_payload,
        "target_spans_revision": _timing_content_digest(target_span_payload),
        "target_region_revision": target_regions.region_revision,
        "target_onset_proof_end": target_regions.onset_proof_end,
        "target_temporal_coherence": target_regions.temporal_coherence,
        "pair_gap_seconds": round(float(pair_gap_seconds), 6),
        "previous_pair_tail_revision": previous_pair_tail.tail_evidence_revision,
        "tail": {
            "validity": previous_pair_tail.validity,
            "start": round(float(previous_pair_tail.reliable_tail_start), 6),
            "end": round(float(previous_pair_tail.reliable_tail_end), 6),
            "uncertainty": round(float(previous_pair_tail.uncertainty_seconds or 0.0), 6),
            "confidence": round(float(previous_pair_tail.confidence or 0.0), 6),
            "terminal_span_indexes": list(previous_pair_tail.terminal_span_indexes),
            "local_gap_references": [
                round(float(value), 6)
                for value in previous_pair_tail.local_gap_references
            ],
        },
        "previous_touches_left_edge": bool(touches_left_edge),
        "previous_right_edge_pileup": bool(previous_probe.right_edge_pileup),
        "previous_window_truncated": bool(previous_probe.window_truncated),
        "target_right_edge_pileup": bool(edge.get("right_edge_pileup")),
        "target_touches_edge": bool(touches_edge),
    }
    entry_index = request.entry_index - 1
    tail_start = float(previous_pair_tail.reliable_tail_start)
    tail_end = float(previous_pair_tail.reliable_tail_end)
    uncertainty_seconds = max(0.0, float(previous_pair_tail.uncertainty_seconds or 0.0))
    confidence = _finite_unit_confidence(previous_pair_tail.confidence or 0.0)
    producer = "joint-ctc-boundary-previous-tail"
    provenance = _make_tail_provenance(
        entry_index=entry_index,
        tail_start=tail_start,
        tail_end=tail_end,
        uncertainty_seconds=uncertainty_seconds,
        confidence=confidence,
        producer=producer,
        backend_lineage=capability.backend_lineage,
        model_identity=capability.model_identity,
        helper_revision=_tail_helper_revision(
            capability.helper_sha256,
            capability.helper_protocol_revision,
        ),
        capability_revision=capability.capability_revision,
        audio_revision=request.alignment_audio_revision,
        transcript_revision=request.transcript_digest,
        row_revision=request.row_mapping_revision,
    )
    source_revision = provenance.observation_revision if provenance is not None else _timing_content_digest(evidence_payload)
    return IndependentTailEvidence(
        entry_index=entry_index,
        tail_start=tail_start,
        tail_end=tail_end,
        uncertainty_seconds=uncertainty_seconds,
        confidence=confidence,
        producer=producer,
        source_revision=source_revision,
        provenance=provenance,
    )


def parse_crossline_retry_output(
    request: CrossLineRetryRequest,
    capability: BoundedRetryCapability,
    entries: list[LyricEntry],
    payload: object,
) -> CrossLineRetryOutcome:
    def invalid(reason: str) -> CrossLineRetryOutcome:
        base = {
            "retry_epoch_revision": request.retry_epoch_revision,
            "request_revision": request.request_revision,
            "entry_index": request.entry_index,
            "status": "invalid",
            "reason": reason,
            "candidate": None,
            "output_revision": None,
        }
        return CrossLineRetryOutcome(**base, outcome_revision=_timing_content_digest(base))

    if not isinstance(payload, dict):
        return invalid("helper-output-not-object")
    try:
        output_revision = _canonical_json_digest(payload)
    except LrcError:
        return invalid("helper-output-nonfinite-or-noncanonical")
    rows = payload.get("entries")
    if not isinstance(rows, list) or len(rows) != len(request.context_entry_indexes):
        return invalid("helper-row-count-mismatch")
    try:
        payload_window_start = float(payload.get("window_start", -1.0))
        payload_window_end = float(payload.get("window_end", -1.0))
    except (TypeError, ValueError):
        return invalid("helper-window-malformed")
    if round(payload_window_start, 3) != round(request.window_start, 3):
        return invalid("helper-window-start-mismatch")
    if round(payload_window_end, 3) != round(request.window_end, 3):
        return invalid("helper-window-end-mismatch")
    parsed_rows: list[tuple[TimingTokenSpan, ...]] = []
    previous_end = request.window_start
    for local_index, (global_index, row) in enumerate(zip(request.context_entry_indexes, rows)):
        if not isinstance(row, dict) or row.get("entry") != local_index + 1:
            return invalid("helper-row-mapping-mismatch")
        if row.get("text") != entry_sung_text(entries[global_index]):
            return invalid("helper-transcript-mismatch")
        raw_spans = row.get("token_spans")
        romaji = row.get("romaji")
        tokens = row.get("tokens")
        if (
            not isinstance(raw_spans, list)
            or not raw_spans
            or not isinstance(romaji, str)
            or not isinstance(tokens, int)
            or tokens != len(raw_spans)
            or len(romaji) != len(raw_spans)
        ):
            return invalid("helper-row-incomplete")
        spans: list[TimingTokenSpan] = []
        for token_index, raw_span in enumerate(raw_spans):
            if not isinstance(raw_span, dict):
                return invalid("helper-span-malformed")
            try:
                start = float(raw_span.get("start"))
                end = float(raw_span.get("end"))
                score = float(raw_span.get("score"))
            except (TypeError, ValueError):
                return invalid("helper-span-malformed")
            if (
                not all(math.isfinite(value) for value in (start, end, score))
                or start > end
                or start < request.window_start - HOP_SECONDS
                or end > request.window_end + HOP_SECONDS
                or start < previous_end
            ):
                return invalid("helper-span-nonmonotonic-or-outside-window")
            token = str(raw_span.get("char") or "")
            if token != romaji[token_index]:
                return invalid("helper-token-sequence-mismatch")
            spans.append(TimingTokenSpan(start, end, score, token))
            previous_end = end
        try:
            row_start = float(row.get("start"))
            row_end = float(row.get("end"))
        except (TypeError, ValueError):
            return invalid("helper-row-boundary-missing")
        if (
            not math.isfinite(row_start)
            or not math.isfinite(row_end)
            or row_start != spans[0].start
            or row_end != spans[-1].end
        ):
            return invalid("helper-row-boundary-mismatch")
        parsed_rows.append(tuple(spans))
    mapping = dict(request.row_mapping)
    target_local_index = mapping.get(request.entry_index)
    if target_local_index is None or not 0 <= target_local_index < len(parsed_rows):
        return invalid("target-row-mapping-missing")
    target_row = rows[target_local_index]
    assert isinstance(target_row, dict)
    target_spans = parsed_rows[target_local_index]
    joint_pair = bool(
        len(request.context_entry_indexes) == 2
        and request.context_entry_indexes == (request.entry_index - 1, request.entry_index)
        and target_local_index == 1
    )
    local_window = {"end": request.window_end}
    edge = ctc_right_edge_pileup_evidence({
        "ctc_local_window": local_window,
        "ctc_token_spans": [
            {"start": span.start, "end": span.end, "score": span.score}
            for span in target_spans
        ],
    })
    edge_margin = request.window_end - target_spans[-1].end
    touches_edge = -0.03 <= edge_margin <= 0.10
    touches_left_edge = False
    previous_probe: TimingCandidate | None = None
    previous_pair_tail: TailRegionEvidence | None = None
    pair_gap_seconds: float | None = None
    if joint_pair:
        previous_local_index = dict(request.row_mapping).get(request.entry_index - 1)
        if previous_local_index is None or not 0 <= previous_local_index < len(parsed_rows):
            return invalid("joint-previous-row-mapping-missing")
        previous_row = rows[previous_local_index]
        assert isinstance(previous_row, dict)
        previous_spans = parsed_rows[previous_local_index]
        touches_left_edge = (
            -0.03 <= previous_spans[0].start - request.window_start <= 0.10
        )
        try:
            previous_confidence = float(previous_row.get("ctc_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return invalid("helper-previous-score-malformed")
        if not math.isfinite(previous_confidence):
            return invalid("helper-previous-score-nonfinite")
        previous_probe = make_timing_candidate(
            entry_index=request.entry_index - 1,
            entry_text=entry_sung_text(entries[request.entry_index - 1]),
            source="ctc-joint-boundary-previous-probe",
            raw_time=previous_spans[0].start,
            spans=previous_spans,
            confidence=previous_confidence,
            identity_support="supported",
            sequence_support="supported",
            current=False,
            source_artifact={
                "output_revision": output_revision,
                "row_mapping_revision": request.row_mapping_revision,
                "joint_boundary_probe": True,
            },
        )
        previous_pair_tail = derive_reliable_tail(previous_probe)
        pair_gap_seconds = target_spans[0].start - previous_spans[-1].end
    try:
        confidence = float(target_row.get("ctc_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return invalid("helper-target-score-malformed")
    if not math.isfinite(confidence):
        return invalid("helper-target-score-nonfinite")

    probe = make_timing_candidate(
        entry_index=request.entry_index,
        entry_text=entry_sung_text(entries[request.entry_index]),
        source="ctc-joint-boundary-current-probe" if joint_pair else "ctc-opening-local-retry-probe",
        raw_time=target_spans[0].start,
        spans=target_spans,
        confidence=confidence,
        identity_support="supported",
        sequence_support="supported",
        current=False,
        prefix_scope=False,
        right_edge_pileup=bool(edge.get("right_edge_pileup")),
        window_truncated=bool(touches_edge or touches_left_edge),
        source_artifact={"probe": True, "output_revision": output_revision},
    )
    target_regions = analyze_candidate_regions(probe)
    strong_joint_boundary = bool(
        joint_pair
        and previous_pair_tail is not None
        and previous_pair_tail.validity == "valid"
        and previous_pair_tail.reliable_tail_end is not None
        and previous_pair_tail.reliable_tail_end <= target_spans[0].start
        and pair_gap_seconds is not None
        and pair_gap_seconds >= -HOP_SECONDS * 0.51
        and target_regions.temporal_coherence == "coherent"
        and target_regions.onset_proof_end is not None
        and not bool(edge.get("right_edge_pileup"))
        and not touches_edge
        and not touches_left_edge
    )
    source = (
        "ctc-joint-boundary-observation"
        if strong_joint_boundary
        else "ctc-joint-boundary-observation-weak"
        if joint_pair
        else "ctc-opening-local-retry"
    )
    candidate = make_timing_candidate(
        entry_index=request.entry_index,
        entry_text=entry_sung_text(entries[request.entry_index]),
        source=source,
        raw_time=target_spans[0].start,
        spans=target_spans,
        confidence=confidence,
        identity_support="supported",
        sequence_support="supported",
        current=False,
        prefix_scope=False,
        right_edge_pileup=bool(edge.get("right_edge_pileup")),
        window_truncated=bool(touches_edge or touches_left_edge),
        source_artifact={
            "capability_revision": capability.capability_revision,
            "retry_semantic_revision": _crossline_retry_semantic_key(request, capability),
            "helper_input_revision": _crossline_retry_helper_cache_key(request, capability),
            "output_revision": output_revision,
            "transcript_digest": request.transcript_digest,
            "row_mapping_revision": request.row_mapping_revision,
            "joint_boundary": joint_pair,
            "strong_joint_boundary": strong_joint_boundary,
            "pair_gap_seconds": round(pair_gap_seconds, 6) if pair_gap_seconds is not None else None,
            "pair_previous_tail_revision": (
                previous_pair_tail.tail_evidence_revision
                if previous_pair_tail is not None else None
            ),
        },
    )
    previous_tail_evidence = _joint_previous_tail_evidence(
        request,
        capability,
        rows,
        parsed_rows,
        previous_probe,
        previous_pair_tail,
        probe,
        target_regions,
        output_revision,
        pair_gap_seconds,
        edge,
        touches_edge,
        touches_left_edge,
        strong_joint_boundary,
    )
    base = {
        "retry_epoch_revision": request.retry_epoch_revision,
        "request_revision": request.request_revision,
        "entry_index": request.entry_index,
        "status": "candidate",
        "reason": ("complete-joint-boundary-rows" if joint_pair else "complete-known-lyric-target-row"),
        "candidate": candidate,
        "output_revision": output_revision,
        "previous_tail_evidence": previous_tail_evidence,
    }
    return CrossLineRetryOutcome(**base, outcome_revision=_timing_content_digest({
        **base, "candidate": candidate.candidate_id,
        "previous_tail_evidence": (
            {
                "entry_index": previous_tail_evidence.entry_index,
                "tail_start": previous_tail_evidence.tail_start,
                "tail_end": previous_tail_evidence.tail_end,
                "uncertainty_seconds": previous_tail_evidence.uncertainty_seconds,
                "confidence": previous_tail_evidence.confidence,
                "producer": previous_tail_evidence.producer,
                "source_revision": previous_tail_evidence.source_revision,
            }
            if previous_tail_evidence is not None
            else None
        ),
    }))


def _crossline_retry_helper_cache_key(
    request: CrossLineRetryRequest,
    capability: BoundedRetryCapability,
) -> str:
    """Digest only the actual helper inputs, excluding retry-epoch bookkeeping."""
    return _timing_content_digest({
        "schema": 1,
        "helper_sha256": capability.helper_sha256,
        "helper_protocol_revision": capability.helper_protocol_revision,
        "alignment_audio_sha256": request.alignment_audio_revision,
        "model_identity": capability.model_identity,
        "model_blob_sha256": _ctc_model_blob_sha256(),
        "device": capability.device,
        "transcript_digest": request.transcript_digest,
        "window_start": request.window_start,
        "window_end": request.window_end,
    })


def _crossline_retry_semantic_key(
    request: CrossLineRetryRequest,
    capability: BoundedRetryCapability,
) -> str:
    """Stable retry meaning, excluding epoch/request bookkeeping.

    The helper inference key captures the actual model inputs.  Entry identity and
    row mapping are added here because the same acoustic/helper payload can be
    interpreted for different immutable lyric entries, which must remain distinct
    central candidates.
    """
    return _timing_content_digest({
        "schema": 1,
        "entry_index": request.entry_index,
        "row_mapping_revision": request.row_mapping_revision,
        "helper_input_revision": _crossline_retry_helper_cache_key(request, capability),
    })


def _novel_crossline_retry_requests(
    requests: tuple[CrossLineRetryRequest, ...],
    capability: BoundedRetryCapability,
    processed_semantics: set[str],
) -> tuple[tuple[CrossLineRetryRequest, ...], tuple[str, ...]]:
    """Return only requests that can produce genuinely new retry evidence."""
    novel: list[CrossLineRetryRequest] = []
    semantic_keys: list[str] = []
    for request in sorted(requests, key=lambda item: (item.entry_index, item.request_revision)):
        semantic_key = _crossline_retry_semantic_key(request, capability)
        if semantic_key in processed_semantics:
            continue
        novel.append(request)
        semantic_keys.append(semantic_key)
    return tuple(novel), tuple(semantic_keys)


def invoke_crossline_retry(
    request: CrossLineRetryRequest,
    capability: BoundedRetryCapability,
    entries: list[LyricEntry],

    *,
    helper_cache: dict[str, tuple[str, object]] | None = None,
    timeout_seconds: float = 180.0,
    cache_policy: "model_cache.ModelCachePolicy | None" = None,
) -> CrossLineRetryOutcome:
    def error_outcome(reason: str) -> CrossLineRetryOutcome:
        base = {
            "retry_epoch_revision": request.retry_epoch_revision,
            "request_revision": request.request_revision,
            "entry_index": request.entry_index,
            "status": "error",
            "reason": reason,
            "candidate": None,
            "output_revision": None,
        }
        return CrossLineRetryOutcome(**base, outcome_revision=_timing_content_digest(base))

    # R2L performance-only cache fast path: helper payload cache hits are
    # immutable model outputs.  Do not create a temp directory/transcript/command
    # merely to discover that the helper input was already evaluated.  Parsing
    # still runs against the *current* immutable request, preserving all epoch,
    # ownership, and candidate semantics.
    cache_key = _crossline_retry_helper_cache_key(request, capability)
    cached = helper_cache.get(cache_key) if helper_cache is not None else None
    cache_identity = model_cache.identity_crossline_joint_ctc(
        alignment_audio_sha256=request.alignment_audio_revision,
        model_identity=capability.model_identity,
        model_blob_sha256=_ctc_model_blob_sha256(),
        helper_protocol_revision=capability.helper_protocol_revision,
        transcript_digest=request.transcript_digest,
        window_start=request.window_start,
        window_end=request.window_end,
        context_entry_indexes=list(request.context_entry_indexes),
    )
    if cache_policy is None:
        cache_policy = model_cache.policy_from_env_and_args(None)
    if cached is None:
        persisted = _model_cache_read("crossline-joint-ctc", cache_key, cache_identity, cache_policy)
        if persisted is not None and helper_cache is not None:
            if isinstance(persisted, dict) and persisted.get("kind") in {"payload", "error"}:
                helper_cache[cache_key] = (str(persisted["kind"]), persisted.get("value"))
                cached = helper_cache[cache_key]
    if cached is not None:
        result_kind, result_value = cached
        if result_kind == "error":
            return error_outcome(str(result_value))
        return parse_crossline_retry_output(request, capability, entries, result_value)

    with tempfile.TemporaryDirectory(prefix="lrc-crossline-retry-") as temp_name:
        temp_dir = Path(temp_name)
        transcript_path = temp_dir / "transcript.json"
        output_path = temp_dir / "result.json"
        transcript_path.write_text(
            json.dumps(
                [entry_sung_text(entries[index]) for index in request.context_entry_indexes],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        command = [
            str(default_ctc_python()), capability.helper_path,
            "--audio", capability.alignment_audio_path,
            "--transcript", str(transcript_path),
            "--output", str(output_path),
            "--device", capability.device,
            "--start", f"{request.window_start:.3f}",
            "--end", f"{request.window_end:.3f}",
        ]
        try:
            proc = _run_model_helper_command(command, mode="ctc", timeout_seconds=timeout_seconds)
        except subprocess.TimeoutExpired:
            result: tuple[str, object] = ("error", "helper-timeout")
        except (LrcError, OSError, subprocess.SubprocessError):
            result = ("error", "helper-invocation-error")
        else:
            if proc.returncode != 0 or not output_path.is_file():
                result = ("error", "helper-invocation-failed")
            else:
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = None
                result = ("payload", payload)
        # Compare kind/value separately: a successful payload is a dict and
        # therefore unhashable, so tuple-in-set membership would raise.
        transient = (
            result[0] == "error"
            and result[1] in {"helper-timeout", "helper-invocation-error"}
        ) or (result[0] == "payload" and result[1] is None)
        if helper_cache is not None and not transient:
            # R3-6: transient helper failures must not suppress this evidence
            # for every later recollection in the same run.
            helper_cache[cache_key] = result
            _model_cache_write(
                "crossline-joint-ctc", cache_key, cache_identity,
                {"kind": result[0], "value": result[1]},
                cache_policy,
            )
        result_kind, result_value = result
        if result_kind == "error":
            return error_outcome(str(result_value))
        payload = result_value
    return parse_crossline_retry_output(request, capability, entries, payload)


def record_crossline_retry_identity_hypothesis(
    report: dict[str, object],
    outcome: CrossLineRetryOutcome,
) -> dict[str, object] | None:
    """Publish a completed bounded retry as identity-only evidence.

    The retry helper aligned the current immutable lyric row inside a bounded
    audio window.  A complete, non-truncated row can therefore establish local
    occurrence identity for later reciprocal-recurrence reasoning.  It does
    *not* establish an onset by itself and is deliberately excluded from the
    ordinary central timing-candidate collector.
    """
    candidate = outcome.candidate
    if (
        outcome.status != "candidate"
        or candidate is None
        or candidate.source != "ctc-opening-local-retry"
        or candidate.identity_support != "supported"
        or candidate.sequence_support != "supported"
        or len(candidate.spans) < 4
        or candidate.window_truncated
        or candidate.right_edge_pileup
    ):
        return None
    assignments = report.get("assignments")
    if (
        not isinstance(assignments, list)
        or not (0 <= outcome.entry_index < len(assignments))
        or not isinstance(assignments[outcome.entry_index], dict)
    ):
        return None
    assignment = assignments[outcome.entry_index]
    assert isinstance(assignment, dict)
    spans = [
        {
            "char": span.token,
            "start": round(float(span.start), 6),
            "end": round(float(span.end), 6),
            "score": round(float(span.score), 8),
        }
        for span in candidate.spans
    ]
    window = {
        "candidate_id": candidate.candidate_id,
        "generation_revision": candidate.generation_revision,
        "request_revision": outcome.request_revision,
        "retry_epoch_revision": outcome.retry_epoch_revision,
        "output_revision": outcome.output_revision,
        "outcome_revision": outcome.outcome_revision,
        "status": outcome.status,
        "reason": outcome.reason,
        "identity_only": True,
    }
    before = len(assignment.get("alignment_hypotheses", [])) if isinstance(assignment.get("alignment_hypotheses"), list) else 0
    hypothesis = record_alignment_hypothesis(
        assignment,
        source="ctc-opening-local-retry-identity",
        stage="bounded-crossline-retry-identity",
        raw_time=candidate.raw_time,
        confidence=candidate.confidence,
        spans=spans,
        identity_support="supported",
        sequence_support="supported",
        acoustic_support="unavailable",
        direct_onset_support="unavailable",
        parent_source="ctc-opening-local-retry",
        parent_time=candidate.raw_time,
        window=window,
        mode="identity-only-complete-known-lyric-row",
        right_edge_pileup=False,
        window_truncated=False,
    )
    after = len(assignment.get("alignment_hypotheses", [])) if isinstance(assignment.get("alignment_hypotheses"), list) else before
    return hypothesis if after > before else None


def execute_crossline_retry_epoch(
    epoch: BaseRetryEpoch,
    capability: BoundedRetryCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    requests: tuple[CrossLineRetryRequest, ...],
    *,
    helper_cache: dict[str, tuple[str, object]] | None = None,
    timeout_seconds: float = 180.0,
    pass_index: int | str | None = None,
    runtime_stats: dict[str, int] | None = None,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
    worker_count: int = 1,
) -> tuple[CrossLineRetryOutcome, ...]:
    """Execute one frozen retry epoch.

    Performance mode may run independent helper inputs concurrently.  Planning,
    request-currentness checks, cache semantics, result ordering, and all Central
    mutations remain outside the concurrent section and therefore deterministic.
    """
    if capability is None:
        return ()

    seen: set[tuple[str, int]] = set()
    ordered_requests = sorted(requests, key=lambda item: (item.entry_index, item.request_revision))
    pass_label = str(pass_index) if pass_index is not None else "?"
    total = len(ordered_requests)

    # R2L performance-only planning dedup: request currentness is evaluated
    # against a frozen epoch/candidate/tail state.  The legacy per-request
    # helper recomputed the complete planner output for every row even though
    # those inputs cannot change inside this epoch.  Compute each applicable
    # planner exactly once, then preserve the same membership test.
    needs_joint = any(len(item.context_entry_indexes) == 2 for item in ordered_requests)
    needs_regular = any(len(item.context_entry_indexes) != 2 for item in ordered_requests)
    current_joint = set(
        plan_joint_boundary_retry_requests(
            epoch, capability, entries, candidate_sets, duration,
            independent_tail_evidence, tail_evidence_context,
            timing_source_policy,
        )
    ) if needs_joint else set()
    current_regular = set(
        plan_crossline_retry_requests(
            epoch, capability, entries, candidate_sets, duration,
            independent_tail_evidence, tail_evidence_context,
            timing_source_policy,
        )
    ) if needs_regular else set()

    prepared: list[tuple[int, CrossLineRetryRequest, str]] = []
    for ordinal, request in enumerate(ordered_requests, start=1):
        ledger_key = (request.retry_epoch_revision, request.entry_index)
        if ledger_key in seen:
            continue
        seen.add(ledger_key)
        current_requests = current_joint if len(request.context_entry_indexes) == 2 else current_regular
        if request not in current_requests:
            continue
        prepared.append((ordinal, request, _crossline_retry_helper_cache_key(request, capability)))

    if not prepared:
        return ()

    worker_count = max(1, min(int(worker_count or 1), 4))
    if worker_count == 1 or len(prepared) <= 1:
        outcomes: list[CrossLineRetryOutcome] = []
        for ordinal, request, cache_key in prepared:
            cache_hit = helper_cache is not None and cache_key in helper_cache
            if runtime_stats is not None:
                runtime_stats["cache_hits" if cache_hit else "cache_misses"] = (
                    runtime_stats.get("cache_hits" if cache_hit else "cache_misses", 0) + 1
                )
                if not cache_hit:
                    runtime_stats["helper_invocations"] = runtime_stats.get("helper_invocations", 0) + 1
            progress_log(
                f"crossline retry pass {pass_label}: {ordinal}/{total} "
                f"entry={request.entry_index + 1} "
                f"window={request.window_start:.2f}-{request.window_end:.2f}s "
                f"cache={'hit' if cache_hit else 'miss'}"
            )
            if helper_cache is None and timeout_seconds == 180.0:
                outcome = invoke_crossline_retry(request, capability, entries)
            else:
                outcome = invoke_crossline_retry(
                    request, capability, entries,
                    helper_cache=helper_cache, timeout_seconds=timeout_seconds,
                )
            if runtime_stats is not None and outcome.reason == "helper-timeout":
                runtime_stats["timeouts"] = runtime_stats.get("timeouts", 0) + 1
            progress_log(
                f"crossline retry pass {pass_label}: entry={request.entry_index + 1} "
                f"status={outcome.status} reason={outcome.reason}"
            )
            outcomes.append(outcome)
        return tuple(outcomes)

    # Recreate the exact serial cache semantics before dispatch: for every helper
    # input already cached, all requests are hits; otherwise the first request is
    # the one real invocation and later requests sharing that helper input are hits.
    grouped: dict[str, list[tuple[int, CrossLineRetryRequest]]] = {}
    for ordinal, request, cache_key in prepared:
        grouped.setdefault(cache_key, []).append((ordinal, request))

    outcomes_by_ordinal: dict[int, CrossLineRetryOutcome] = {}
    work: list[tuple[str, int, CrossLineRetryRequest]] = []
    for cache_key, group in grouped.items():
        pre_cached = helper_cache is not None and cache_key in helper_cache
        if runtime_stats is not None:
            if pre_cached:
                runtime_stats["cache_hits"] = runtime_stats.get("cache_hits", 0) + len(group)
            else:
                runtime_stats["cache_misses"] = runtime_stats.get("cache_misses", 0) + 1
                runtime_stats["cache_hits"] = runtime_stats.get("cache_hits", 0) + max(0, len(group) - 1)
                runtime_stats["helper_invocations"] = runtime_stats.get("helper_invocations", 0) + 1
        if pre_cached:
            for ordinal, request in group:
                progress_log(
                    f"crossline retry pass {pass_label}: cached {ordinal}/{total} "
                    f"entry={request.entry_index + 1}"
                )
                outcomes_by_ordinal[ordinal] = invoke_crossline_retry(
                    request, capability, entries,
                    helper_cache=helper_cache, timeout_seconds=timeout_seconds,
                )
        else:
            first_ordinal, first_request = group[0]
            work.append((cache_key, first_ordinal, first_request))

    progress_log(
        f"crossline retry pass {pass_label}: parallel_workers={worker_count} "
        f"unique_exec={len(work)}"
    )

    def _run_retry_parallel(
        item: tuple[str, int, CrossLineRetryRequest],
    ) -> tuple[str, int, CrossLineRetryOutcome, dict[str, tuple[str, object]]]:
        cache_key, ordinal, request = item
        private_cache: dict[str, tuple[str, object]] = {}
        progress_log(
            f"crossline retry pass {pass_label}: start {ordinal}/{total} "
            f"entry={request.entry_index + 1} "
            f"window={request.window_start:.2f}-{request.window_end:.2f}s"
        )
        outcome = invoke_crossline_retry(
            request, capability, entries,
            helper_cache=private_cache, timeout_seconds=timeout_seconds,
        )
        return cache_key, ordinal, outcome, private_cache

    if work:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(worker_count, len(work)),
            thread_name_prefix="lrc-ctc-retry",
        ) as pool:
            future_map = {pool.submit(_run_retry_parallel, item): item for item in work}
            for future in concurrent.futures.as_completed(future_map):
                cache_key, ordinal, outcome, private_cache = future.result()
                if helper_cache is not None:
                    helper_cache.update(private_cache)
                outcomes_by_ordinal[ordinal] = outcome
                progress_log(
                    f"crossline retry pass {pass_label}: done {ordinal}/{total} "
                    f"entry={outcome.entry_index + 1} "
                    f"status={outcome.status} reason={outcome.reason}"
                )
                # Same helper payload may map to more than one immutable request.
                # Parse those additional requests only after the representative
                # payload is in the shared cache; no second model invocation occurs.
                for grouped_ordinal, grouped_request in grouped[cache_key][1:]:
                    outcomes_by_ordinal[grouped_ordinal] = invoke_crossline_retry(
                        grouped_request, capability, entries,
                        helper_cache=helper_cache, timeout_seconds=timeout_seconds,
                    )

    ordered_outcomes = tuple(
        outcomes_by_ordinal[ordinal]
        for ordinal, _request, _cache_key in prepared
        if ordinal in outcomes_by_ordinal
    )
    if runtime_stats is not None:
        runtime_stats["timeouts"] = runtime_stats.get("timeouts", 0) + sum(
            item.reason == "helper-timeout" for item in ordered_outcomes
        )
    return ordered_outcomes

def _retry_diagnostic_payload(item: object) -> dict[str, object]:
    if isinstance(item, CrossLineRetryRequest):
        return {
            "entry": item.entry_index + 1,
            "ordinal": item.ordinal,
            "retry_epoch_revision": item.retry_epoch_revision,
            "request_revision": item.request_revision,
            "capability_revision": item.capability_revision,
            "alignment_audio_revision": item.alignment_audio_revision,
            "current": item.current.__dict__,
            "previous": item.previous.__dict__,
            "next_bound": item.next_bound.__dict__ if item.next_bound else None,
            "ownership_relation_revision": item.ownership_relation_revision,
            "previous_tail_evidence_revision": item.previous_tail_evidence_revision,
            "previous_tail": {
                "start": item.previous_tail_start,
                "end": item.previous_tail_end,
                "uncertainty": item.previous_tail_uncertainty,
                "validity": item.previous_tail_validity,
            },
            "audio_end_seconds": item.audio_end_seconds,
            "audio_duration_revision": item.audio_duration_revision,
            "context_entry_indexes": list(item.context_entry_indexes),
            "context_candidate_ids": list(item.context_candidate_ids),
            "transcript_digests": list(item.transcript_digests),
            "window": [item.window_start, item.window_end],
            "window_definition_revision": item.window_definition_revision,
            "transcript_digest": item.transcript_digest,
            "row_mapping": [list(pair) for pair in item.row_mapping],
            "row_mapping_revision": item.row_mapping_revision,
        }
    if isinstance(item, CrossLineRetryOutcome):
        return {
            "entry": item.entry_index + 1,
            "retry_epoch_revision": item.retry_epoch_revision,
            "request_revision": item.request_revision,
            "status": item.status,
            "reason": item.reason,
            "candidate_id": item.candidate.candidate_id if item.candidate else None,
            "output_revision": item.output_revision,
            "outcome_revision": item.outcome_revision,
            "previous_tail_evidence": _independent_tail_evidence_payload(
                item.previous_tail_evidence
            ),
        }
    raise TypeError("Unsupported retry diagnostic item")


def _retry_capability_diagnostic_payload(
    capability: BoundedRetryCapability | None,
) -> dict[str, object] | None:
    if capability is None:
        return None
    return {
        "kind": capability.kind,
        "backend_lineage": capability.backend_lineage,
        "helper_path": capability.helper_path,
        "helper_sha256": capability.helper_sha256,
        "helper_protocol_revision": capability.helper_protocol_revision,
        "alignment_audio_path": capability.alignment_audio_path,
        "alignment_audio_sha256": capability.alignment_audio_sha256,
        "alignment_audio_source": capability.alignment_audio_source,
        "sample_rate": capability.sample_rate,
        "model_identity": capability.model_identity,
        "device": capability.device,
        "capability_revision": capability.capability_revision,
    }


def bind_runtime_raw_content_identity(
    report: dict[str, object],
    entries: list[LyricEntry],
    raw_report: dict[str, object],
) -> None:
    """Bind independent raw-ASR lyric identity without granting timing authority."""
    raw_assignments = raw_report.get("assignments")
    if not isinstance(raw_assignments, list) or len(raw_assignments) != len(entries):
        return
    drift_entries = {
        int(entry)
        for run in report.get("raw_asr_systematic_drift_runs", [])
        if isinstance(run, dict)
        for entry in run.get("entries", [])
        if isinstance(entry, int)
    }
    segment_owners = collections.Counter(
        row.get("segment") for row in raw_assignments
        if isinstance(row, dict) and isinstance(row.get("segment"), int)
    )
    skipped_shared_segment: list[int] = []
    runtime: dict[int, dict[str, object]] = {}
    for index, raw_assignment in enumerate(raw_assignments):
        if (
            isinstance(raw_assignment, dict)
            and isinstance(raw_assignment.get("segment"), int)
            and segment_owners[raw_assignment["segment"]] > 1
        ):
            skipped_shared_segment.append(index + 1)
            continue
        if not isinstance(raw_assignment, dict):
            continue
        occurrence_binding = _entry_occurrence_binding(entries, raw_assignments, index)
        if occurrence_binding is None:
            continue
        try:
            raw_time = float(raw_assignment.get("timestamp"))
            raw_score = float(raw_assignment.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        segment = raw_assignment.get("segment")
        if (
            raw_score < TRUSTED_ALIGNMENT_SCORE
            or bool(raw_assignment.get("borrowed"))
            or segment is None
            or index + 1 in drift_entries
        ):
            continue
        lyric_sha = hashlib.sha256(
            normalize_match_text(entry_sung_text(entries[index])).encode("utf-8")
        ).hexdigest()
        payload = {
            "entry_index": index,
            "lyric_sha256": lyric_sha,
            "raw_time": round(raw_time, 6),
            "raw_score": round(raw_score, 6),
            "segment": int(segment) if isinstance(segment, int) else str(segment),
            "occurrence_binding": occurrence_binding,
            "producer": "raw-asr-content-identity",
        }
        runtime[index] = {
            **payload,
            "evidence_revision": _timing_content_digest(payload),
        }
    report[_RUNTIME_RAW_CONTENT_IDENTITY_KEY] = runtime
    report["raw_asr_content_identity"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "minimum_score": TRUSTED_ALIGNMENT_SCORE,
        "candidate_count": len(runtime),
        "entries": [index + 1 for index in sorted(runtime)],
        "skipped_shared_asr_segment_entries": skipped_shared_segment,
    }


def bind_runtime_whisperx_row_evidence(
    report: dict[str, object],
    entries: list[LyricEntry],
    evidence_candidates: list[tuple[list[float], dict[str, object], str]],
) -> None:
    """Preserve trusted per-line WhisperX identity/boundary geometry at runtime.

    Whole-backend rejection is an aggregate decision.  A row with a real,
    non-borrowed ASR match can still contribute independent identity and forced
    alignment geometry.  Authority remains runtime-only and row review defects
    are excluded before binding.
    """
    best: dict[int, dict[str, object]] = {}
    tail_candidates: dict[int, list[IndependentTailEvidence]] = {}
    scope_records: dict[tuple[int, str], list[TailEvidenceScope]] = {}
    for _candidate_times, candidate_report, _error in evidence_candidates:
        assignments = candidate_report.get("assignments")
        forced_rows = candidate_report.get("whisperx_forced_rows")
        runtime_provenance = candidate_report.get(
            _RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY
        )
        if not isinstance(assignments, list) or not isinstance(forced_rows, list):
            continue
        if len(assignments) != len(entries) or len(forced_rows) != len(entries):
            continue
        actionable_entries = {
            int(row.get("entry"))
            for row in candidate_report.get("suspicious_alignments", [])
            if isinstance(row, dict)
            and isinstance(row.get("entry"), int)
            and _suspicious_row_requires_review(row)
        }
        suppress_nst = bool(candidate_report.get("whisper_suppress_nst"))
        for index, (assignment, forced) in enumerate(zip(assignments, forced_rows)):
            if (
                not isinstance(assignment, dict)
                or not isinstance(forced, dict)
                or index + 1 in actionable_entries
            ):
                continue
            occurrence_binding = _entry_occurrence_binding(entries, assignments, index)
            if occurrence_binding is None:
                continue
            try:
                score = float(assignment.get("score", 0.0) or 0.0)
                segment_start = float(forced.get("segment_start"))
                segment_end = float(forced.get("segment_end"))
                first_time = float(forced.get("first_char_time"))
                last_time = float(forced.get("last_char_time"))
                char_count = int(forced.get("char_count", 0) or 0)
            except (TypeError, ValueError):
                continue
            numeric = (score, segment_start, segment_end, first_time, last_time)
            if (
                not all(math.isfinite(value) for value in numeric)
                or score < TRUSTED_ALIGNMENT_SCORE
                or bool(assignment.get("borrowed"))
                or assignment.get("segment") is None
                or char_count <= 0
                or segment_end < segment_start
                or not (segment_start - HOP_SECONDS <= first_time <= segment_end + HOP_SECONDS)
                or not (first_time - HOP_SECONDS <= last_time <= segment_end + HOP_SECONDS)
            ):
                continue
            lyric_sha = hashlib.sha256(
                normalize_match_text(entry_sung_text(entries[index])).encode("utf-8")
            ).hexdigest()
            payload = {
                "entry_index": index,
                "lyric_sha256": lyric_sha,
                "score": round(score, 6),
                "segment": assignment.get("segment"),
                "segment_start": round(segment_start, 6),
                "segment_end": round(segment_end, 6),
                "first_char_time": round(first_time, 6),
                "last_char_time": round(last_time, 6),
                "char_count": char_count,
                "suppress_nst": suppress_nst,
                "occurrence_binding": occurrence_binding,
                "producer": "whisperx-row-evidence",
            }
            revision = _timing_content_digest(payload)
            row = {**payload, "evidence_revision": revision}
            current = best.get(index)
            if current is None or (score, char_count, -segment_end) > (
                float(current.get("score", 0.0) or 0.0),
                int(current.get("char_count", 0) or 0),
                -float(current.get("segment_end", 0.0) or 0.0),
            ):
                best[index] = row
            uncertainty_seconds = max(0.0, segment_end - last_time)
            provenance: TailEvidenceProvenance | None = None
            scope: TailEvidenceScope | None = None
            if _whisperx_runtime_provenance_is_valid(runtime_provenance):
                assert isinstance(runtime_provenance, WhisperXRuntimeProvenance)
                model_identity = (
                    f"{runtime_provenance.align_model_id}@"
                    f"{runtime_provenance.align_model_revision}#"
                    f"{runtime_provenance.align_model_blob_sha256}"
                )
                helper_revision = _timing_content_digest({
                    "helper_sha256": runtime_provenance.helper_sha256,
                    "helper_protocol_revision": runtime_provenance.helper_protocol_revision,
                })
                transcript_revision = _timing_content_digest({
                    "raw_transcript_revision": runtime_provenance.raw_transcript_revision,
                    "entry_index": index,
                    "lyric_sha256": lyric_sha,
                    "occurrence_binding": occurrence_binding,
                })
                provenance = _make_tail_provenance(
                    entry_index=index,
                    tail_start=first_time,
                    tail_end=last_time,
                    uncertainty_seconds=uncertainty_seconds,
                    confidence=score,
                    producer="whisperx-forced-row",
                    backend_lineage=runtime_provenance.backend_lineage,
                    model_identity=model_identity,
                    helper_revision=helper_revision,
                    capability_revision=runtime_provenance.capability_revision,
                    audio_revision=runtime_provenance.alignment_wav_sha256,
                    transcript_revision=transcript_revision,
                    row_revision=revision,
                )
                if provenance is not None:
                    scope = TailEvidenceScope(
                        entry_index=index,
                        producer="whisperx-forced-row",
                        backend_lineage=runtime_provenance.backend_lineage,
                        model_identity=model_identity,
                        helper_revision=helper_revision,
                        capability_revision=runtime_provenance.capability_revision,
                        audio_revision=runtime_provenance.alignment_wav_sha256,
                        transcript_revisions=(transcript_revision,),
                        row_revisions=(revision,),
                    )
            source_revision = provenance.observation_revision if provenance is not None else revision
            if scope is not None:
                scope_records.setdefault((index, scope.producer), []).append(scope)
            tail_candidates.setdefault(index, []).append(
                IndependentTailEvidence(
                    entry_index=index,
                    tail_start=first_time,
                    tail_end=last_time,
                    uncertainty_seconds=uncertainty_seconds,
                    confidence=score,
                    producer="whisperx-forced-row",
                    source_revision=source_revision,
                    provenance=provenance,
                )
            )

    tails = _runtime_tail_observation_map(report)
    for index, candidates in tail_candidates.items():
        tails.setdefault(index, []).extend(candidates)

    report[_RUNTIME_WHISPERX_ROW_EVIDENCE_KEY] = best
    report[_RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = _freeze_runtime_tail_observation_map(tails)
    scopes: dict[tuple[int, str], TailEvidenceScope] = {}
    for key, values in scope_records.items():
        first = values[0]
        if any(
            (
                item.backend_lineage != first.backend_lineage
                or item.model_identity != first.model_identity
                or item.helper_revision != first.helper_revision
                or item.audio_revision != first.audio_revision
            )
            for item in values[1:]
        ):
            continue
        scopes[key] = replace(first, **{
            "transcript_revisions": tuple(sorted({
                revision
                for item in values
                for revision in item.transcript_revisions
            })),
            "row_revisions": tuple(sorted({
                revision
                for item in values
                for revision in item.row_revisions
            })),
            "capability_revisions": tuple(sorted({
                item.capability_revision
                for item in values[1:]
            })),
        })
    report[_RUNTIME_WHISPERX_TAIL_SCOPES_KEY] = tuple(
        scopes[key]
        for key in sorted(scopes)
    )
    report["whisperx_row_evidence"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "minimum_score": TRUSTED_ALIGNMENT_SCORE,
        "candidate_count": len(best),
        "entries": [index + 1 for index in sorted(best)],
        "tail_evidence_count": len(tails),
    }


def bind_runtime_ctc_assignment_tail_fallback(
    report: dict[str, object],
    entries: list[LyricEntry],
) -> None:
    """Restore predecessor-tail geometry that would otherwise be lost after fusion.

    A selected Raw/WhisperX/onset-fusion candidate can legitimately carry no CTC
    spans even though the authoritative CTC assignment for the same immutable lyric
    row still contains usable terminal geometry.  Losing that geometry turns the
    next boundary into UNKNOWN for an implementation reason rather than an evidence
    reason.

    This is intentionally a *fallback*, not another independent backend vote:
    existing WhisperX row-tail evidence always wins.  The CTC row is admitted only
    when no distinct-backend runtime tail exists for that predecessor, and only when
    the normal ``derive_reliable_tail`` rules already classify its terminal run as
    valid.  No timestamp or tail limit is fabricated.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return
    assert_lyric_identity_contract(entries, report, initialize=True)
    tails = _runtime_tail_observation_map(report)
    generated: list[int] = []
    for index, (entry, assignment) in enumerate(zip(entries, assignments)):
        if index in tails or not isinstance(assignment, dict):
            continue
        spans = _timing_spans_from_value(assignment.get("ctc_token_spans"))
        if len(spans) < 2:
            continue
        raw_time = assignment.get("timestamp")
        if not isinstance(raw_time, (int, float)) or not math.isfinite(float(raw_time)):
            continue
        lyric_sha = hashlib.sha256(
            normalize_match_text(entry_sung_text(entry)).encode("utf-8")
        ).hexdigest()
        span_payload = [
            [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
            for span in spans
        ]
        artifact = {
            "producer": "ctc-assignment-terminal-row",
            "entry_index": index,
            "lyric_sha256": lyric_sha,
            "timing_repair_source": assignment.get("timing_repair_source"),
            "ctc_audio_source": report.get("ctc_audio_source"),
            "ctc_audio_sha256": report.get("ctc_audio_sha256") or report.get("vocal_stem_sha256"),
            "spans_revision": _timing_content_digest(span_payload),
        }
        probe = make_timing_candidate(
            entry_index=index,
            entry_text=entry_sung_text(entry),
            source="ctc-assignment-tail-fallback",
            raw_time=float(raw_time),
            spans=spans,
            confidence=_candidate_confidence(assignment),
            identity_support="supported",
            sequence_support="supported",
            current=False,
            right_edge_pileup=bool(assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(
                assignment.get("ctc_right_edge_truncation")
                or assignment.get("ctc_boundary_realign_required")
            ),
            source_artifact=artifact,
        )
        tail = derive_reliable_tail(probe)
        if (
            tail.validity != "valid"
            or tail.reliable_tail_start is None
            or tail.reliable_tail_end is None
        ):
            continue
        payload = {
            **artifact,
            "tail_start": round(tail.reliable_tail_start, 6),
            "tail_end": round(tail.reliable_tail_end, 6),
            "uncertainty_seconds": round(float(tail.uncertainty_seconds or 0.0), 6),
            "confidence": round(float(tail.confidence or 0.0), 6),
            "tail_evidence_revision": tail.tail_evidence_revision,
        }
        entry_index = index
        tail_start = float(tail.reliable_tail_start)
        tail_end = float(tail.reliable_tail_end)
        uncertainty_seconds = max(0.0, float(tail.uncertainty_seconds or 0.0))
        confidence = _finite_unit_confidence(tail.confidence or 0.0)
        capability = report.get(_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY)
        if isinstance(capability, BoundedRetryCapability):
            backend_lineage = capability.backend_lineage
            model_identity = capability.model_identity
            helper_revision = _tail_helper_revision(
                capability.helper_sha256,
                capability.helper_protocol_revision,
            )
            capability_revision = capability.capability_revision
            audio_revision = capability.alignment_audio_sha256
        else:
            backend_lineage = str(report.get("backend_lineage") or "")
            model_identity = str(report.get("model_identity") or "")
            helper_revision = str(report.get("helper_revision") or "")
            capability_revision = str(report.get("capability_revision") or "")
            audio_revision = str(
                report.get("ctc_audio_sha256") or report.get("vocal_stem_sha256") or ""
            )
        provenance = _make_tail_provenance(
            entry_index=entry_index,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty_seconds,
            confidence=confidence,
            producer="ctc-assignment-terminal-row",
            backend_lineage=backend_lineage,
            model_identity=model_identity,
            helper_revision=helper_revision,
            capability_revision=capability_revision,
            audio_revision=audio_revision,
            transcript_revision=lyric_sha,
            row_revision=_timing_content_digest(span_payload),
        )
        source_revision = (
            provenance.observation_revision
            if provenance is not None else _timing_content_digest(payload)
        )
        observation = IndependentTailEvidence(
            entry_index=entry_index,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty_seconds,
            confidence=confidence,
            producer="ctc-assignment-terminal-row",
            source_revision=source_revision,
            provenance=provenance,
        )
        tails.setdefault(index, []).append(observation)
        generated.append(index + 1)
    report[_RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = _freeze_runtime_tail_observation_map(tails)
    report["ctc_assignment_tail_fallback"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "role": "fallback-only-when-distinct-backend-tail-missing",
        "candidate_count": len(generated),
        "entries": generated,
    }


def _augment_local_ctc_candidates_with_runtime_backend_consensus(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Extend existing cross-backend onset proof to local/retry CTC candidates.

    v14.2 could bind Raw/WhisperX identity only to the baseline ``ctc-current``
    time.  A bounded local CTC retry may produce a better lyric-specific opening,
    yet remain unable to use the same independent backend observation.  This
    helper closes only that provenance gap.  It reuses the existing 0.18 s
    consensus tolerance and keeps the CTC candidate's existing timestamp; no
    averaging or synthetic boundary is introduced.
    """
    # Structural S1 is also local CTC. Independent Raw/WhisperX identities
    # may corroborate the SAME bounded candidate only if they already
    # satisfy the normal 0.18 s consensus and revision-bound occurrence.
    # Bare S1 CTC / vocal presence is never upgraded on its own.
    local_sources = {"ctc-local-window", "ctc-opening-local-retry", "segment-constrained-ctc", SEGMENT_REALIGN_SOURCE}
    augmented: list[tuple[TimingCandidate, ...]] = []
    for index, items in enumerate(candidate_sets):
        rows = list(items)
        raw_identity = runtime_raw_content_identity_for_entry(report, entries, index)
        whisperx_row = runtime_whisperx_row_evidence_for_entry(report, entries, index)
        for parent in items:
            if (
                parent.source not in local_sources
                or not parent.spans
                or parent.right_edge_pileup
                or parent.window_truncated
            ):
                continue
            backend_rows: list[tuple[str, float, float, str, str]] = []
            if isinstance(raw_identity, dict):
                try:
                    raw_time = float(raw_identity.get("raw_time"))
                    raw_score = float(raw_identity.get("raw_score"))
                except (TypeError, ValueError):
                    raw_time = math.nan
                    raw_score = 0.0
                if (
                    math.isfinite(raw_time)
                    and raw_score >= RAW_CTC_CONSENSUS_MIN_SCORE
                    and abs(raw_time - parent.written_time.seconds)
                    <= RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
                ):
                    backend_rows.append((
                        "raw", raw_time, raw_score,
                        str(raw_identity.get("evidence_revision") or ""),
                        "raw-asr-consensus",
                    ))
            if isinstance(whisperx_row, dict):
                try:
                    forced_time = float(whisperx_row.get("first_char_time"))
                    whisperx_score = float(whisperx_row.get("score"))
                except (TypeError, ValueError):
                    forced_time = math.nan
                    whisperx_score = 0.0
                if (
                    math.isfinite(forced_time)
                    and whisperx_score >= TRUSTED_ALIGNMENT_SCORE
                    and abs(forced_time - parent.written_time.seconds)
                    <= RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
                ):
                    backend_rows.append((
                        "whisperx", forced_time, whisperx_score,
                        str(whisperx_row.get("evidence_revision") or ""),
                        "whisperx-consensus",
                    ))
            for label, independent_time, independent_score, evidence_revision, producer in backend_rows:
                identity_evidence = (
                    raw_identity if label == "raw" else whisperx_row
                )
                occurrence_evidence = candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer=producer,
                    occurrence_binding=(
                        identity_evidence.get("occurrence_binding")
                        if isinstance(identity_evidence, dict) else None
                    ),
                    upstream_evidence_revision=(
                        identity_evidence.get("evidence_revision")
                        if isinstance(identity_evidence, dict) else None
                    ),
                )
                artifact = {
                    "producer": f"ctc-local-{label}-consensus",
                    "parent_candidate_id": parent.candidate_id,
                    "parent_generation_revision": parent.generation_revision,
                    "parent_time": round(parent.written_time.seconds, 6),
                    "independent_time": round(independent_time, 6),
                    "independent_score": round(independent_score, 6),
                    "independent_evidence_revision": evidence_revision,
                    "consensus_tolerance_seconds": RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
                    "runtime_only": True,
                }
                rows.append(
                    make_timing_candidate(
                        entry_index=index,
                        entry_text=entry_sung_text(entries[index]),
                        source=f"ctc-local-{label}-independent-consensus",
                        raw_time=parent.written_time.seconds,
                        spans=parent.spans,
                        confidence=min(parent.confidence, independent_score),
                        identity_support="supported",
                        sequence_support="supported",
                        acoustic_support="unavailable",
                        direct_onset_support="supported",
                        direct_onset_time=independent_time,
                        direct_onset_source_artifact=artifact,
                        direct_onset_evidence_producer=producer,
                        direct_onset_evidence_kind="cross-backend-onset",
                        direct_onset_evidence_independent=True,
                        occurrence_evidence=occurrence_evidence,
                        independent_content_identity=occurrence_evidence is not None,
                        current=False,
                        source_artifact=artifact,
                    )
                )
        unique = {candidate.candidate_id: candidate for candidate in rows}
        augmented.append(tuple(unique[key] for key in sorted(unique)))
    return tuple(augmented)


def _joint_consensus_spans(
    parent: TimingCandidate,
    observation: TimingCandidate,
) -> tuple[TimingTokenSpan, ...]:
    """Token spans for a joint-consensus candidate written at the parent time.

    A span-less parent may borrow the joint observation's spans (this gives the
    row a reliable tail for its successor, case34), but only when those spans
    describe the same written time: the observation's first token must start
    within one hop of the parent time, exactly the tolerance that
    ``normalize_timing_evidence`` enforces ("candidate-span-revision-mismatch").
    Borrowing spans that start elsewhere made every such joint candidate
    invalid (43 joint candidates across the 39-song corpus, e.g. ふり #55).
    """
    if parent.spans:
        return parent.spans
    borrowed = observation.spans
    if borrowed and abs(borrowed[0].start - parent.written_time.seconds) <= HOP_SECONDS:
        return borrowed
    return parent.spans


def _augment_candidates_with_joint_boundary_consensus(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    pair_outcomes: tuple[CrossLineRetryOutcome, ...],
    entries: list[LyricEntry],
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Bind joint previous+current CTC boundary observations to separate candidates.

    The pair observation is deliberately not selectable by itself.  It may only
    certify a separately generated current-line candidate when both boundaries
    converge within the already-established 0.18 s consensus tolerance.

    This is stronger than generic acoustic activity: the observation explicitly
    aligns the immutable previous lyric and current lyric in one CTC sequence,
    and the selected timestamp still comes from an independent candidate
    generation path rather than from the pair observation.
    """
    by_entry: dict[int, list[TimingCandidate]] = collections.defaultdict(list)
    for outcome in pair_outcomes:
        candidate = outcome.candidate
        if (
            outcome.status != "candidate"
            or candidate is None
            or candidate.source != "ctc-joint-boundary-observation"
            or candidate.window_truncated
            or candidate.right_edge_pileup
            or not candidate.spans
        ):
            continue
        by_entry[outcome.entry_index].append(candidate)

    augmented: list[tuple[TimingCandidate, ...]] = []
    for index, items in enumerate(candidate_sets):
        rows = list(items)
        observations = by_entry.get(index, [])
        if observations:
            for parent in items:
                if (
                    parent.source.startswith("ctc-joint-boundary")
                    or parent.window_truncated
                    or parent.right_edge_pileup
                    or parent.identity_support != "supported"
                    or parent.sequence_support != "supported"
                ):
                    continue
                for observation in observations:
                    if (
                        abs(
                            observation.written_time.seconds
                            - parent.written_time.seconds
                        )
                        > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
                    ):
                        continue
                    artifact = {
                        "producer": "joint-ctc-boundary-consensus",
                        "parent_candidate_id": parent.candidate_id,
                        "parent_generation_revision": parent.generation_revision,
                        "parent_time": round(parent.written_time.seconds, 6),
                        "pair_candidate_id": observation.candidate_id,
                        "pair_generation_revision": observation.generation_revision,
                        "pair_time": round(observation.written_time.seconds, 6),
                        "pair_span_digest": _timing_content_digest([
                            [
                                round(span.start, 6),
                                round(span.end, 6),
                                round(span.score, 8),
                                span.token,
                            ]
                            for span in observation.spans
                        ]),
                        "consensus_tolerance_seconds": RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
                        "runtime_only": True,
                    }
                    rows.append(
                        make_timing_candidate(
                            entry_index=index,
                            entry_text=entry_sung_text(entries[index]),
                            source="ctc-joint-boundary-consensus",
                            raw_time=parent.written_time.seconds,
                            spans=_joint_consensus_spans(parent, observation),
                            confidence=min(parent.confidence, observation.confidence),
                            identity_support="supported",
                            sequence_support="supported",
                            acoustic_support=parent.acoustic_support,
                            direct_onset_support="supported",
                            acoustic_strength=parent.acoustic_strength,
                            acoustic_onset_time=(
                                parent.acoustic_evidence.onset_time
                                if parent.acoustic_evidence is not None else None
                            ),
                            acoustic_source_artifact=(
                                {
                                    "parent_acoustic_revision":
                                        parent.acoustic_evidence.source_revision
                                }
                                if parent.acoustic_evidence is not None else None
                            ),
                            acoustic_evidence_producer=(
                                parent.acoustic_evidence.producer
                                if parent.acoustic_evidence is not None else None
                            ),
                            acoustic_evidence_kind=(
                                parent.acoustic_evidence.kind
                                if parent.acoustic_evidence is not None
                                else "generic-acoustic"
                            ),
                            acoustic_evidence_independent=(
                                parent.acoustic_evidence.independent
                                if parent.acoustic_evidence is not None else False
                            ),
                            direct_onset_time=observation.written_time.seconds,
                            direct_onset_source_artifact=artifact,
                            direct_onset_evidence_producer="joint-ctc-boundary-consensus",
                            direct_onset_evidence_kind="joint-lyric-boundary",
                            direct_onset_evidence_independent=True,
                            occurrence_evidence=parent.occurrence_evidence,
                            independent_content_identity=(
                                parent.independent_content_identity
                                and parent.occurrence_evidence is not None
                            ),
                            current=False,
                            prefix_scope=parent.prefix_scope,
                            source_artifact=artifact,
                        )
                    )
        unique = {candidate.candidate_id: candidate for candidate in rows}
        augmented.append(tuple(unique[key] for key in sorted(unique)))
    return tuple(augmented)


def _freeze_runtime_tail_observation_map(
    observations: dict[int, list[IndependentTailEvidence]],
) -> dict[int, TailObservationSet]:
    frozen: dict[int, TailObservationSet] = {}
    for index, values in observations.items():
        unique: dict[str, IndependentTailEvidence] = {}
        for item in values:
            revision = (
                item.provenance.observation_revision
                if item.provenance is not None
                else item.source_revision
            )
            unique.setdefault(revision, item)
        if unique:
            frozen[index] = tuple(
                unique[key]
                for key in sorted(unique)
            )
    return frozen


def merge_runtime_independent_tail_evidence(
    report: dict[str, object],
    evidence: IndependentTailEvidence | None,
) -> bool:
    """Append one immutable observation without cross-lineage replacement."""
    if evidence is None or evidence.entry_index < 0:
        return False
    merged = _runtime_tail_observation_map(report)
    before = _freeze_runtime_tail_observation_map(merged)
    merged.setdefault(evidence.entry_index, []).append(evidence)
    after = _freeze_runtime_tail_observation_map(merged)
    report[_RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = after
    return before != after


def _independent_tail_evidence_payload(
    evidence: IndependentTailEvidence | None,
) -> dict[str, object] | None:
    if evidence is None:
        return None
    payload: dict[str, object] = {
        "entry_index": evidence.entry_index,
        "tail_start": round(evidence.tail_start, 6),
        "tail_end": round(evidence.tail_end, 6),
        "uncertainty_seconds": round(evidence.uncertainty_seconds, 6),
        "confidence": round(evidence.confidence, 6),
        "producer": evidence.producer,
        "source_revision": evidence.source_revision,
    }
    if evidence.provenance is None:
        payload["provenance_status"] = "missing"
        payload["authority_level"] = "DIAGNOSTIC_ONLY"
        payload["rejection_reason"] = "missing-tail-provenance"
    elif _tail_provenance_is_valid(evidence):
        payload["provenance_status"] = "valid"
        payload["authority_level"] = "PENDING_PREDECESSOR_LINEAGE_CHECK"
        payload["provenance"] = {
            **_tail_provenance_payload(evidence.provenance),
            "observation_revision": evidence.provenance.observation_revision,
        }
    else:
        payload["provenance_status"] = "invalid"
        payload["authority_level"] = "DIAGNOSTIC_ONLY"
        payload["rejection_reason"] = "invalid-tail-provenance"
    return payload


def independent_tail_evidence_payload(
    evidence: RuntimeTailEvidence | tuple[IndependentTailEvidence | None, ...],
) -> list[object]:
    payload: list[object] = []
    for item in evidence:
        observations = _normalize_tail_observations(item)  # type: ignore[arg-type]
        if not observations:
            payload.append(None)
        elif len(observations) == 1:
            payload.append(_independent_tail_evidence_payload(observations[0]))
        else:
            payload.append([
                _independent_tail_evidence_payload(observation)
                for observation in observations
            ])
    return payload


def _candidate_set_semantic_revision(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> str:
    return _timing_content_digest([
        sorted(candidate.candidate_id for candidate in candidates)
        for candidates in candidate_sets
    ])


def _authoritative_tail_semantic_revision(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    independent_tail_evidence: RuntimeTailEvidence,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> str:
    if len(candidate_sets) != len(independent_tail_evidence):
        return _timing_content_digest({"invalid_length": True})
    try:
        state = build_final_timing_state(
            candidate_sets,
            independent_tail_evidence=independent_tail_evidence,
            tail_evidence_context=tail_evidence_context,
            timing_source_policy=timing_source_policy,
        )
    except (TypeError, ValueError):
        return _timing_content_digest({"invalid_state": True})
    return _timing_content_digest([
        {
            "entry_index": decision.entry_index,
            "ownership_status": decision.evidence.ownership.status,
            "tail_revision": decision.evidence.referenced_tail_evidence_revision,
            "tail_validity": decision.evidence.ownership.previous_tail_validity,
            "tail_end": decision.evidence.ownership.reliable_tail_end,
            "tail_uncertainty": decision.evidence.ownership.uncertainty_seconds,
        }
        for decision in state.decisions
        if decision.entry_index > 0
    ])


def _retry_semantic_state_revision(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    independent_tail_evidence: RuntimeTailEvidence,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> tuple[str, str]:
    return (
        _candidate_set_semantic_revision(candidate_sets),
        _authoritative_tail_semantic_revision(
            candidate_sets,
            independent_tail_evidence,
            tail_evidence_context,
            timing_source_policy,
        ),
    )


def _tail_authority_diagnostics(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    independent_tail_evidence: RuntimeTailEvidence,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> list[dict[str, object]]:
    """Explain Contract-A acceptance/rejection without adding trust logic."""
    if len(candidate_sets) != len(independent_tail_evidence):
        return []
    try:
        state = build_final_timing_state(
            candidate_sets,
            independent_tail_evidence=independent_tail_evidence,
            tail_evidence_context=tail_evidence_context,
            timing_source_policy=timing_source_policy,
        )
    except (TypeError, ValueError):
        return []
    rows: list[dict[str, object]] = []
    for index, observations in enumerate(independent_tail_evidence):
        if not observations or index >= len(state.decisions):
            continue
        predecessor = state.decisions[index].candidate
        families = sorted(_candidate_support_families(predecessor))
        authoritative = _authoritative_tail_observations(
            predecessor, observations, tail_evidence_context
        )
        selected = authoritative[0] if authoritative else None
        intrinsic = derive_reliable_tail(predecessor)
        intrinsic_upper = (
            intrinsic.reliable_tail_end + (intrinsic.uncertainty_seconds or 0.0)
            if intrinsic.validity == "valid" and intrinsic.reliable_tail_end is not None
            else None
        )
        for observation in observations:
            provenance = observation.provenance
            family = _tail_backend_family(provenance.backend_lineage if provenance else None)
            if not _tail_provenance_is_valid(observation):
                authority, reason = "DIAGNOSTIC_ONLY", "invalid-or-missing-provenance"
            elif not _tail_contexts_are_valid(tail_evidence_context):
                authority, reason = "DIAGNOSTIC_ONLY", "missing-tail-evidence-context"
            elif not _tail_provenance_matches_context(observation, tail_evidence_context):
                authority, reason = "DIAGNOSTIC_ONLY", "stale-tail-evidence-context"
            elif not families:
                authority, reason = "REJECTED", "unknown-predecessor-support-family"
            elif family in families:
                authority, reason = "REJECTED", "same-lineage-overlap"
            elif selected is not observation:
                authority, reason = "VALID_NONSELECTED", "another-disjoint-bound-dominates"
            elif intrinsic_upper is not None and intrinsic_upper >= (
                observation.tail_end + observation.uncertainty_seconds
            ):
                authority, reason = "VALID_NONSELECTED", "intrinsic-bound-dominates"
            else:
                authority, reason = "ACCEPTED", "disjoint-lineage-authoritative-bound"
            rows.append({
                "entry": index + 1,
                "producer": observation.producer,
                "backend_lineage": provenance.backend_lineage if provenance else None,
                "candidate_source": predecessor.source,
                "candidate_support_families": families,
                "authority": authority,
                "reason": reason,
                "observation_revision": observation.source_revision,
                "tail_start": round(observation.tail_start, 6),
                "tail_end": round(observation.tail_end, 6),
                "uncertainty_seconds": round(observation.uncertainty_seconds, 6),
            })
    return rows


def bind_runtime_whisperx_ctc_consensus(
    report: dict[str, object],
    entries: list[LyricEntry],
    ctc_timestamps: list[float],
    evidence_candidates: list[tuple[list[float], dict[str, object], str]],
) -> None:
    """Preserve line-valid WhisperX evidence even when the backend fails globally."""
    best: dict[int, dict[str, object]] = {}
    for _candidate_times, candidate_report, _error in evidence_candidates:
        assignments = candidate_report.get("assignments")
        forced_times = candidate_report.get("whisperx_forced_first_times")
        if not isinstance(assignments, list) or not isinstance(forced_times, list):
            continue
        if len(assignments) != len(entries) or len(forced_times) != len(entries):
            continue
        suppress_nst = bool(candidate_report.get("whisper_suppress_nst"))
        for index, (assignment, forced_value) in enumerate(zip(assignments, forced_times)):
            if not isinstance(assignment, dict):
                continue
            occurrence_binding = _entry_occurrence_binding(entries, assignments, index)
            if occurrence_binding is None:
                continue
            try:
                score = float(assignment.get("score", 0.0) or 0.0)
                forced_time = float(forced_value)
                ctc_time = float(ctc_timestamps[index])
            except (TypeError, ValueError, IndexError):
                continue
            if (
                score < TRUSTED_ALIGNMENT_SCORE
                or bool(assignment.get("borrowed"))
                or assignment.get("segment") is None
                or not math.isfinite(forced_time)
                or abs(forced_time - ctc_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
            ):
                continue
            lyric_sha = hashlib.sha256(
                normalize_match_text(entry_sung_text(entries[index])).encode("utf-8")
            ).hexdigest()
            payload = {
                "entry_index": index,
                "lyric_sha256": lyric_sha,
                "whisperx_forced_time": round(forced_time, 6),
                "ctc_time": round(ctc_time, 6),
                "score": round(score, 6),
                "segment": assignment.get("segment"),
                "suppress_nst": suppress_nst,
                "occurrence_binding": occurrence_binding,
                "producer": "whisperx-consensus",
            }
            row = {**payload, "evidence_revision": _timing_content_digest(payload)}
            current = best.get(index)
            if current is None or (score, -abs(forced_time - ctc_time)) > (
                float(current.get("score", 0.0) or 0.0),
                -abs(float(current.get("whisperx_forced_time", 0.0)) - ctc_time),
            ):
                best[index] = row
    report[_RUNTIME_WHISPERX_CTC_CONSENSUS_KEY] = best
    report["whisperx_ctc_onset_consensus"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "minimum_score": TRUSTED_ALIGNMENT_SCORE,
        "max_delta_seconds": PREFIX_ACOUSTIC_ASSOCIATION_SECONDS,
        "candidate_count": len(best),
        "entries": [index + 1 for index in sorted(best)],
    }


def runtime_whisperx_ctc_consensus_for_entry(
    report: dict[str, object],
    entries: list[LyricEntry],
    entry_index: int,
    current_time: float,
) -> dict[str, object] | None:
    runtime = report.get(_RUNTIME_WHISPERX_CTC_CONSENSUS_KEY)
    if not isinstance(runtime, dict):
        return None
    evidence = runtime.get(entry_index)
    if not isinstance(evidence, dict):
        return None
    try:
        forced_time = float(evidence.get("whisperx_forced_time"))
        ctc_time = float(evidence.get("ctc_time"))
        score = float(evidence.get("score"))
    except (TypeError, ValueError):
        return None
    lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entries[entry_index])).encode("utf-8")
    ).hexdigest()
    payload = {
        "entry_index": entry_index,
        "lyric_sha256": lyric_sha,
        "whisperx_forced_time": round(forced_time, 6),
        "ctc_time": round(ctc_time, 6),
        "score": round(score, 6),
        "segment": evidence.get("segment"),
        "suppress_nst": bool(evidence.get("suppress_nst")),
        "occurrence_binding": evidence.get("occurrence_binding"),
        "producer": "whisperx-consensus",
    }
    if (
        evidence.get("producer") != "whisperx-consensus"
        or score < TRUSTED_ALIGNMENT_SCORE
        or evidence.get("lyric_sha256") != lyric_sha
        or not _runtime_occurrence_binding_is_valid(
            entries, entry_index, evidence.get("occurrence_binding")
        )
        or abs(forced_time - ctc_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
        or canonicalize_lrc_time(ctc_time).centiseconds
        != canonicalize_lrc_time(current_time).centiseconds
        or evidence.get("evidence_revision") != _timing_content_digest(payload)
    ):
        return None
    return dict(evidence)


def runtime_raw_ctc_consensus_for_entry(
    report: dict[str, object],
    entries: list[LyricEntry],
    entry_index: int,
    current_time: float,
) -> dict[str, object] | None:
    """Return fresh in-process raw/CTC onset agreement for one entry.

    The authority is deliberately non-serializable and is re-bound to the
    current CTC timestamp.  This prevents stale report metadata from being used
    as independent evidence on another run or after the baseline moved.
    """
    if str(report.get("backend") or "") != "ctc":
        return None
    runtime = report.get(_RUNTIME_RAW_CTC_CONSENSUS_KEY)
    if not isinstance(runtime, dict):
        return None
    evidence = runtime.get(entry_index)
    if not isinstance(evidence, dict):
        return None
    try:
        raw_time = float(evidence.get("raw_time"))
        ctc_time = float(evidence.get("ctc_time"))
        raw_score = float(evidence.get("raw_score"))
        max_delta = float(evidence.get("max_delta_seconds"))
    except (TypeError, ValueError):
        return None
    if (
        evidence.get("producer") != "raw-asr-consensus"
        or raw_score < RAW_CTC_CONSENSUS_MIN_SCORE
        or abs(max_delta - RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS) > 1e-9
        or abs(raw_time - ctc_time) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
        or canonicalize_lrc_time(ctc_time).centiseconds
        != canonicalize_lrc_time(current_time).centiseconds
    ):
        return None
    payload = {
        "entry_index": entry_index,
        "raw_time": round(raw_time, 6),
        "ctc_time": round(ctc_time, 6),
        "raw_score": round(raw_score, 6),
        "max_delta_seconds": RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
        "occurrence_binding": evidence.get("occurrence_binding"),
        "producer": "raw-asr-consensus",
    }
    if (
        not _runtime_occurrence_binding_is_valid(
            entries, entry_index, evidence.get("occurrence_binding")
        )
        or evidence.get("evidence_revision") != _timing_content_digest(payload)
    ):
        return None
    return dict(evidence)


def _attach_ctc_wide_region_candidates(
    candidate_sets: Sequence[tuple[TimingCandidate, ...]],
    report: dict[str, object],
    entries: list[LyricEntry],
    hypothesis_falsifications: Sequence[dict[str, object]],
) -> tuple[tuple[tuple[TimingCandidate, ...], ...], list[dict[str, object]]]:
    """Admit a row's radius-3 replay as its wide-region fallback candidate.

    Agreement: the radius-2 replay lands within 0.3 s, or another candidate
    that is not a first-token posterior lands within 0.1 s.  Veto: any
    independent onset for the row (acoustic recurrence, a trusted local
    Whisper onset or its arbitration) more than 0.5 s away.  The vocal-fusion
    onsets riding on candidates are not vetoes: on c17 #20 one sat 1.3 s off
    BK and blocked a correct wide time.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(candidate_sets):
        return tuple(candidate_sets), []
    independent_onsets: dict[int, list[float]] = collections.defaultdict(list)
    for key in ("mutual_acoustic_recurrence_hypotheses", "global_acoustic_recurrence_hypotheses"):
        rows = report.get(key)
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and isinstance(row.get("candidate"), (int, float)):
                independent_onsets[int(row.get("entry", 0)) - 1].append(float(row["candidate"]))
    for row in hypothesis_falsifications:
        if isinstance(row.get("local_onset"), (int, float)):
            independent_onsets[int(row.get("entry", 0)) - 1].append(float(row["local_onset"]))
    revised: list[tuple[TimingCandidate, ...]] = []
    diagnostics: list[dict[str, object]] = []
    for index, (candidates, assignment) in enumerate(zip(candidate_sets, assignments)):
        hypotheses = assignment.get("alignment_hypotheses") if isinstance(assignment, dict) else None
        wide: dict[int, dict[str, object]] = {}
        for hypothesis in hypotheses if isinstance(hypotheses, list) else []:
            if (
                isinstance(hypothesis, dict)
                and hypothesis.get("source") == CTC_WIDE_REGION_FALLBACK_SOURCE
                and isinstance(hypothesis.get("raw_candidate_time"), (int, float))
            ):
                for radius in (CTC_WIDE_REGION_RADIUS, CTC_WIDE_REGION_PEER_RADIUS):
                    if hypothesis.get("mode") == f"region-radius-{radius}":
                        wide[radius] = hypothesis
        primary = wide.get(CTC_WIDE_REGION_RADIUS)
        if primary is None:
            revised.append(tuple(candidates))
            continue
        seconds = canonicalize_lrc_time(float(primary["raw_candidate_time"])).seconds
        peer = wide.get(CTC_WIDE_REGION_PEER_RADIUS)
        peer_agrees = bool(
            peer is not None
            and abs(float(peer["raw_candidate_time"]) - seconds)
            <= CTC_WIDE_REGION_PEER_AGREEMENT_SECONDS
        )
        supported = any(
            candidate.source != "ctc-first-token-posterior"
            and abs(candidate.written_time.seconds - seconds) <= CTC_WIDE_REGION_SUPPORT_SECONDS
            for candidate in candidates
        )
        onsets = list(independent_onsets.get(index, []))
        local = runtime_local_whisper_evidence_for_entry(report, entries, index)
        if isinstance(local, dict) and isinstance(local.get("onset_time"), (int, float)):
            onsets.append(float(local["onset_time"]))
        vetoed = any(abs(onset - seconds) > CTC_WIDE_REGION_VETO_SECONDS for onset in onsets)
        admitted = (peer_agrees or supported) and not vetoed
        diagnostics.append({
            "entry": index + 1,
            "time": seconds,
            "peer_time": (
                round(float(peer["raw_candidate_time"]), 3) if peer is not None else None
            ),
            "peer_agrees": peer_agrees,
            "candidate_support": supported,
            "independent_onsets": sorted(round(onset, 3) for onset in onsets),
            "vetoed": vetoed,
            "admitted": admitted,
        })
        if not admitted:
            revised.append(tuple(candidates))
            continue
        spans = _timing_spans_from_value(primary.get("spans"))
        wide_candidate = make_timing_candidate(
            entry_index=index,
            entry_text=entry_sung_text(entries[index]),
            source=CTC_WIDE_REGION_FALLBACK_SOURCE,
            raw_time=float(primary["raw_candidate_time"]),
            spans=spans,
            confidence=float(primary.get("confidence", 0.0) or 0.0),
            sequence_support="supported",
            source_artifact=primary,
        )
        unique = {candidate.candidate_id: candidate for candidate in (*candidates, wide_candidate)}
        revised.append(tuple(unique[key] for key in sorted(unique)))
    return tuple(revised), diagnostics


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


def _evidence_report_payload(evidence: NormalizedTimingEvidence) -> dict[str, object]:
    return {
        "candidate_id": evidence.candidate_id,
        "candidate_revision": evidence.candidate_revision,
        "computed_for_centiseconds": evidence.computed_for_centiseconds,
        "region_revision": evidence.region_revision,
        "ownership_relation_revision": evidence.ownership_relation_revision,
        "referenced_tail_evidence_revision": evidence.referenced_tail_evidence_revision,
        "prefix_duration": evidence.prefix_duration,
        "max_internal_gap": evidence.max_internal_gap,
        "median_internal_gap": evidence.median_internal_gap,
        "p90_internal_gap": evidence.p90_internal_gap,
        "p95_internal_gap": evidence.p95_internal_gap,
        "token_density": evidence.token_density,
        "longest_supported_run": evidence.longest_supported_run,
        "opening_supported_run": evidence.opening_supported_run,
        "later_supported_run": evidence.later_supported_run,
        "reliable_support_count": evidence.reliable_support_count,
        "onset_proof_end": evidence.onset_proof_end,
        "island_count": evidence.island_count,
        "severe_gap_count": evidence.severe_gap_count,
        "first_severe_gap_index": evidence.first_severe_gap_index,
        "temporal_coherence": evidence.temporal_coherence,
        "fracture_region": evidence.fracture_region,
        "ownership": {
            "status": evidence.ownership.status,
            "relation_revision": evidence.ownership.relation_revision,
            "current_candidate_id": evidence.ownership.current_candidate_id,
            "current_generation_revision": evidence.ownership.current_generation_revision,
            "current_region_revision": evidence.ownership.current_region_revision,
            "current_centiseconds": evidence.ownership.current_centiseconds,
            "previous_candidate_id": evidence.ownership.previous_candidate_id,
            "previous_generation_revision": evidence.ownership.previous_generation_revision,
            "previous_region_revision": evidence.ownership.previous_region_revision,
            "previous_tail_evidence_revision": evidence.ownership.previous_tail_evidence_revision,
            "previous_tail_validity": evidence.ownership.previous_tail_validity,
            "previous_tail_authority_backend_lineage": (
                evidence.ownership.previous_tail_authority_backend_lineage
            ),
            "previous_tail_authority_producer": (
                evidence.ownership.previous_tail_authority_producer
            ),
            "previous_tail_authority_observation_revision": (
                evidence.ownership.previous_tail_authority_observation_revision
            ),
            "reliable_tail_start": evidence.ownership.reliable_tail_start,
            "reliable_tail_end": evidence.ownership.reliable_tail_end,
            "uncertainty_seconds": evidence.ownership.uncertainty_seconds,
            "confidence": evidence.ownership.confidence,
            "distance_seconds": evidence.ownership.distance_seconds,
            "reason": evidence.ownership.reason,
        },
        "tail_distance_seconds": evidence.tail_distance_seconds,
        "ownership_independent": evidence.ownership_independent,
        "structural_failures": list(evidence.structural_failures),
        "minimum_proof_satisfied": evidence.minimum_proof_satisfied,
    }


def _historical_timing_findings(
    decision: TimingDecision,
    flags: list[str],
) -> tuple[TimingFinding, ...]:
    return tuple(
        TimingFinding(
            finding_id=_timing_content_digest({
                "entry": decision.entry_index,
                "flag": flag,
                "candidate_revision": decision.evidence.candidate_revision,
            }),
            code=flag,
            raised_revision=max(0, decision.selection_revision - 1),
            evidence_scope="historical-legacy-boundary",
        )
        for flag in dict.fromkeys(flags)
    )


def _recovery_has_revision_bound_resolution(
    decision: TimingDecision,
    audit: TimingInvariantAudit,
) -> bool:
    recovery = decision.recovery
    required_invariants = {
        "revision-consistency",
        "canonical-time-evidence-match",
        "structural-validity",
        "minimum-proof",
        "candidate-dominates-current",
    }
    return bool(
        audit.timing_trusted
        and recovery is not None
        and recovery.resolution_mode == "challenger-recovery"
        and recovery.candidate_id == decision.candidate.candidate_id
        and recovery.candidate_id == decision.selected_candidate_id
        and recovery.candidate_revision == decision.evidence.candidate_revision
        and recovery.evidence_revision == decision.timing_evidence_revision
        and recovery.canonical_written_centiseconds == decision.written_time.centiseconds
        and recovery.new_selection_revision == decision.selection_revision
        and len(recovery.resolved_finding_ids) == len(set(recovery.resolved_finding_ids))
        and required_invariants.issubset(recovery.passed_invariants)
    )


def evaluate_final_timing_state_for_report(
    state: FinalTimingState,
    report: dict[str, object],
    *,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    peer_graph: PeerGraphPolicy | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
    reviewer_evidence: tuple[dict[str, object] | None, ...] | None = None,
) -> tuple[FinalTimingState, tuple[EntryTimingFindings, ...]]:
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(state.decisions):
        raise LrcError("Final timing state does not match report assignments")
    base_audits = audit_final_timing_decisions(
        state.decisions,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        occurrence_entries=occurrence_entries,
        reviewer_evidence=reviewer_evidence,
    )
    finding_states: list[EntryTimingFindings] = []
    for decision, audit, raw_assignment in zip(state.decisions, base_audits, assignments):
        if not isinstance(raw_assignment, dict):
            raise LrcError("Final timing evaluation encountered an invalid assignment")
        legacy_flags = raw_assignment.get("flags")
        flag_list = list(legacy_flags) if isinstance(legacy_flags, list) else []
        historical = _historical_timing_findings(
            decision,
            [flag for flag in flag_list if flag in CTC_BOUNDARY_REVIEW_FLAGS],
        )
        explicitly_resolved_ids = (
            set(decision.recovery.resolved_finding_ids)
            if _recovery_has_revision_bound_resolution(decision, audit)
            and decision.recovery is not None
            else set()
        )
        resolved = tuple(
            finding for finding in historical
            if finding.finding_id in explicitly_resolved_ids
        )
        unresolved = tuple(
            finding for finding in historical
            if finding.finding_id not in explicitly_resolved_ids
        )
        finding_states.append(EntryTimingFindings(historical, resolved, unresolved))
    unresolved_codes = tuple(
        tuple(finding.code for finding in findings.unresolved)
        for findings in finding_states
    )
    effective_audits = audit_final_timing_decisions(
        state.decisions,
        unresolved_historical_codes=unresolved_codes,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        occurrence_entries=occurrence_entries,
        reviewer_evidence=reviewer_evidence,
    )
    return (
        FinalTimingState(state.decisions, state.written_times, effective_audits),
        tuple(finding_states),
    )


CENTRAL_RESOLVABLE_SUSPICIOUS_FLAGS = frozenset({
    "unresolved_raw_ctc_disagreement",
    "unresolved_ctc_disagreement",
    "candidate_disagreement",
    "phonetic_anchor_disagreement",
})


def _explicit_whisper_arbitration_resolves_raw_disagreement(
    report: object,
    row: dict[str, object],
    selected_time: float,
) -> bool:
    """Require provenance-bound falsification before clearing a distant raw/CTC conflict."""
    if not isinstance(report, dict):
        return False
    entry = row.get("entry")
    candidate_timestamps = row.get("candidate_timestamps")
    if not isinstance(entry, int) or not isinstance(candidate_timestamps, dict):
        return False
    try:
        raw_time = float(candidate_timestamps.get("raw_asr"))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(raw_time):
        return False
    arbitration = report.get("hypothesis_arbitration")
    if not isinstance(arbitration, dict):
        return False
    rows = arbitration.get("entries")
    if not isinstance(rows, list):
        return False
    tolerance = RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
    for item in rows:
        if not isinstance(item, dict) or item.get("entry") != entry:
            continue
        falsified_ids = item.get("falsified_candidate_ids")
        if not isinstance(falsified_ids, list) or not falsified_ids:
            continue
        try:
            falsified_raw_time = float(item.get("raw_identity_time"))
            resolved_local_onset = float(item.get("local_onset"))
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(falsified_raw_time)
            and math.isfinite(resolved_local_onset)
            and abs(falsified_raw_time - raw_time) <= tolerance
            and abs(resolved_local_onset - selected_time) <= tolerance
        ):
            return True
    return False


def _candidate_evidence_projection(
    evaluation: CandidateEvaluation,
) -> dict[str, object]:
    """Path-free projection of one existing Central candidate evaluation."""

    candidate = evaluation.candidate
    evidence = evaluation.evidence
    ownership = evidence.ownership
    authority_metadata = (
        ownership.previous_tail_authority_backend_lineage,
        ownership.previous_tail_authority_producer,
        ownership.previous_tail_authority_observation_revision,
    )
    lineage_relation = (
        "intrinsic"
        if all(value is None for value in authority_metadata)
        else _current_tail_lineage_relation(candidate, *authority_metadata)
    )

    def onset_projection(value: object) -> dict[str, object] | None:
        if value is None:
            return None
        return {
            "producer": str(getattr(value, "producer", "")),
            "kind": str(getattr(value, "kind", "")),
            "source_revision": str(getattr(value, "source_revision", "")),
            "independent": bool(getattr(value, "independent", False)),
        }

    return {
        "candidate_id": candidate.candidate_id,
        "source": candidate.source,
        "raw_time": round(candidate.raw_time, 6),
        "canonical_written_centiseconds": candidate.written_time.centiseconds,
        "generation_revision": candidate.generation_revision,
        "support_families": sorted(_candidate_support_families(candidate)),
        "identity_support": candidate.identity_support,
        "sequence_support": candidate.sequence_support,
        "direct_onset_support": candidate.direct_onset_support,
        "acoustic_support": candidate.acoustic_support,
        "occurrence_evidence": _candidate_occurrence_evidence_projection(
            candidate.occurrence_evidence
        ),
        "direct_onset_evidence": onset_projection(candidate.direct_onset_evidence),
        "acoustic_evidence": onset_projection(candidate.acoustic_evidence),
        "candidate_evidence_revision": evidence.candidate_revision,
        "region_revision": evidence.region_revision,
        "ownership_relation_revision": evidence.ownership_relation_revision,
        "minimum_proof_satisfied": bool(evidence.minimum_proof_satisfied),
        "temporal_coherence": evidence.temporal_coherence,
        "onset_proof_end": evidence.onset_proof_end,
        "ownership_status": ownership.status,
        "ownership_reason": ownership.reason,
        "ownership_lineage_relation": lineage_relation,
        "previous_tail_validity": ownership.previous_tail_validity,
        "previous_tail_evidence_revision": ownership.previous_tail_evidence_revision,
        "previous_tail_authority_backend_lineage": ownership.previous_tail_authority_backend_lineage,
        "previous_tail_authority_producer": ownership.previous_tail_authority_producer,
        "previous_tail_authority_observation_revision": ownership.previous_tail_authority_observation_revision,
        "structural_failures": list(evidence.structural_failures),
        "rejection_reasons": list(evaluation.rejection_reasons),
    }


def _decision_candidate_evaluations(
    decision: TimingDecision,
) -> tuple[CandidateEvaluation, ...]:
    """Return the selected evaluation plus the immutable rejected graph."""
    selected = CandidateEvaluation(
        candidate=decision.candidate,
        evidence=decision.evidence,
        valid=decision.status == "selected_valid",
        rejection_reasons=decision.evidence.structural_failures,
    )
    return (selected, *decision.rejected_candidates)


def _raw_vocal_occurrence_observation(
    decision: TimingDecision,
    entries: tuple[LyricEntry, ...],
) -> dict[str, object] | None:
    """Return one exact runtime raw-vocal occurrence observation for a row.

    This is timing-drift evidence only.  It preserves the independent raw lyric
    identity contract and never mutates candidate selection or content trust.
    """
    observations: list[dict[str, object]] = []
    for evaluation in _decision_candidate_evaluations(decision):
        candidate = evaluation.candidate
        if (
            candidate.source != "raw-vocal-independent-fusion"
            or not candidate_occurrence_evidence_is_valid(candidate, entries)
        ):
            continue
        occurrence = candidate.occurrence_evidence
        assert occurrence is not None
        try:
            binding = json.loads(occurrence.binding_payload_json)
            raw_time = float(binding.get("timestamp"))
            raw_score = float(binding.get("score"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not (
            isinstance(binding, dict)
            and math.isfinite(raw_time)
            and math.isfinite(raw_score)
        ):
            continue
        observations.append({
            "candidate_id": candidate.candidate_id,
            "occurrence_evidence_revision": occurrence.evidence_revision,
            "binding_revision": occurrence.binding_revision,
            "binding_mode": occurrence.occurrence_mode,
            "raw_time": raw_time,
            "raw_score": raw_score,
        })
    if not observations:
        return None
    return min(
        observations,
        key=lambda item: (-float(item["raw_score"]), str(item["candidate_id"])),
    )


def _ctc_current_clock_time(decision: TimingDecision) -> float | None:
    """Return the exact Central ctc-current candidate clock for drift analysis."""
    values = [
        evaluation.candidate.raw_time
        for evaluation in _decision_candidate_evaluations(decision)
        if evaluation.candidate.source == "ctc-current"
        and math.isfinite(evaluation.candidate.raw_time)
    ]
    if not values:
        return None
    return float(values[0])


def _raw_timing_drift_falsification_details(
    state: FinalTimingState,
    entries: tuple[LyricEntry, ...],
) -> tuple[dict[str, object], ...]:
    """Prove raw *timing* drift without invalidating raw lyric identity.

    Two already-existing evidence shapes are accepted:

    1. At least three consecutive raw-vocal occurrence observations drift from
       the same ctc-current clock in one direction, with every canonical offset
       outside the existing 18-centisecond material boundary and adjacent
       offsets changing by no more than the legacy 0.65-second section-drift
       bound.
    2. A revision-bound sequence/sparse-sequence occurrence binding carries at
       least three trusted anchors whose raw timestamps all drift in one
       direction from timing-trusted final Central decisions by more than the
       same material boundary.

    The target selected decision must itself be timing-trusted, minimum-proof
    satisfied, ownership-clear, and structurally clean.  A raw material
    candidate is falsified only when its exact occurrence proof participates in
    the corresponding drift proof.  No timestamp, selector, confidence, or
    content-identity state is changed here.
    """
    if (
        len(entries) != len(state.decisions)
        or len(state.decisions) != len(state.audits)
    ):
        return ()

    tolerance_centiseconds = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    raw_observations: list[dict[str, object] | None] = []
    ctc_clocks: list[float | None] = []
    for decision in state.decisions:
        raw_observations.append(_raw_vocal_occurrence_observation(decision, entries))
        ctc_clocks.append(_ctc_current_clock_time(decision))

    contiguous_by_index: dict[int, list[dict[str, object]]] = collections.defaultdict(list)
    index = 0
    while index < len(state.decisions):
        observation = raw_observations[index]
        ctc_time = ctc_clocks[index]
        eligible = False
        if observation is not None and ctc_time is not None:
            raw_time = float(observation["raw_time"])
            raw_score = float(observation["raw_score"])
            eligible = bool(
                raw_score >= TRUSTED_ALIGNMENT_SCORE
                and abs(
                    canonicalize_lrc_time(raw_time).centiseconds
                    - canonicalize_lrc_time(ctc_time).centiseconds
                ) > tolerance_centiseconds
            )
        if not eligible:
            index += 1
            continue

        run = [index]
        cursor = index + 1
        while cursor < len(state.decisions):
            current = raw_observations[cursor]
            current_ctc = ctc_clocks[cursor]
            previous = raw_observations[run[-1]]
            previous_ctc = ctc_clocks[run[-1]]
            if (
                current is None
                or current_ctc is None
                or previous is None
                or previous_ctc is None
            ):
                break
            current_raw = float(current["raw_time"])
            current_score = float(current["raw_score"])
            current_delta = current_raw - current_ctc
            previous_delta = float(previous["raw_time"]) - previous_ctc
            current_eligible = bool(
                current_score >= TRUSTED_ALIGNMENT_SCORE
                and abs(
                    canonicalize_lrc_time(current_raw).centiseconds
                    - canonicalize_lrc_time(current_ctc).centiseconds
                ) > tolerance_centiseconds
            )
            if (
                not current_eligible
                or current_delta * previous_delta <= 0.0
                or abs(current_delta - previous_delta) > 0.65
            ):
                break
            run.append(cursor)
            cursor += 1

        if len(run) >= 3:
            run_deltas = [
                float(raw_observations[item]["raw_time"]) - float(ctc_clocks[item])  # type: ignore[index]
                for item in run
            ]
            if (
                abs(float(statistics.median(run_deltas)))
                > RAW_TIMING_DRIFT_MAX_MEDIAN_DELTA_SECONDS
            ):
                index = max(index + 1, cursor)
                continue
            anchors = [
                {
                    "entry": item + 1,
                    "raw_centiseconds": canonicalize_lrc_time(
                        float(raw_observations[item]["raw_time"])  # type: ignore[index]
                    ).centiseconds,
                    "ctc_centiseconds": canonicalize_lrc_time(
                        float(ctc_clocks[item])
                    ).centiseconds,
                    "raw_score": round(
                        float(raw_observations[item]["raw_score"]), 6  # type: ignore[index]
                    ),
                }
                for item in run
            ]
            proof = {
                "proof": "contiguous-drift",
                "entries": [item + 1 for item in run],
                "anchors": anchors,
            }
            for item in run:
                contiguous_by_index[item].append(proof)
        index = max(index + 1, cursor)

    context_certificates: list[dict[str, object]] = []
    seen_context: set[tuple[object, ...]] = set()
    for owner_index, decision in enumerate(state.decisions):
        for evaluation in _decision_candidate_evaluations(decision):
            candidate = evaluation.candidate
            if (
                candidate.source != "raw-vocal-independent-fusion"
                or not candidate_occurrence_evidence_is_valid(candidate, entries)
            ):
                continue
            occurrence = candidate.occurrence_evidence
            assert occurrence is not None
            if occurrence.occurrence_mode not in {
                "sequence-bracket", "sparse-sequence-bracket"
            }:
                continue
            try:
                binding = json.loads(occurrence.binding_payload_json)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(binding, dict):
                continue
            if occurrence.occurrence_mode == "sequence-bracket":
                anchor_indexes = binding.get("context_entry_indexes")
                anchor_timestamps = binding.get("context_timestamps")
                anchor_scores = binding.get("context_scores")
                context_revision = binding.get("context_revision")
            else:
                anchor_indexes = binding.get("anchor_entry_indexes")
                anchor_timestamps = binding.get("anchor_timestamps")
                anchor_scores = binding.get("anchor_scores")
                context_revision = binding.get("fingerprint_revision")
            if not (
                isinstance(anchor_indexes, list)
                and isinstance(anchor_timestamps, list)
                and isinstance(anchor_scores, list)
                and len(anchor_indexes) == len(anchor_timestamps) == len(anchor_scores)
                and len(anchor_indexes) >= 3
            ):
                continue
            anchors: list[dict[str, object]] = []
            valid = True
            for anchor_index, raw_time, raw_score in zip(
                anchor_indexes, anchor_timestamps, anchor_scores
            ):
                if (
                    not isinstance(anchor_index, int)
                    or not 0 <= anchor_index < len(state.decisions)
                    or not isinstance(raw_time, (int, float))
                    or isinstance(raw_time, bool)
                    or not isinstance(raw_score, (int, float))
                    or isinstance(raw_score, bool)
                    or not math.isfinite(float(raw_time))
                    or not math.isfinite(float(raw_score))
                    or float(raw_score) < TRUSTED_ALIGNMENT_SCORE
                    or not state.audits[anchor_index].timing_trusted
                ):
                    valid = False
                    break
                central_time = state.decisions[anchor_index].written_time.seconds
                anchors.append({
                    "entry": anchor_index + 1,
                    "raw_centiseconds": canonicalize_lrc_time(
                        float(raw_time)
                    ).centiseconds,
                    "central_centiseconds": state.decisions[
                        anchor_index
                    ].written_time.centiseconds,
                    "raw_score": round(float(raw_score), 6),
                    "delta_seconds": round(float(raw_time) - central_time, 6),
                })
            if not valid:
                continue
            deltas = [float(item["delta_seconds"]) for item in anchors]
            if not (
                abs(float(statistics.median(deltas)))
                <= RAW_TIMING_DRIFT_MAX_MEDIAN_DELTA_SECONDS
                and (all(value > 0.0 for value in deltas) or all(value < 0.0 for value in deltas))
                and all(
                    abs(
                        int(item["raw_centiseconds"])
                        - int(item["central_centiseconds"])
                    ) > tolerance_centiseconds
                    for item in anchors
                )
            ):
                continue
            signature = (
                occurrence.occurrence_mode,
                tuple(
                    (
                        int(item["entry"]),
                        int(item["raw_centiseconds"]),
                        int(item["central_centiseconds"]),
                        float(item["raw_score"]),
                    )
                    for item in anchors
                ),
            )
            if signature in seen_context:
                continue
            seen_context.add(signature)
            context_certificates.append({
                "proof": "context-bound-drift",
                "owner_entry": owner_index + 1,
                "owner_candidate_id": candidate.candidate_id,
                "occurrence_mode": occurrence.occurrence_mode,
                "binding_revision": occurrence.binding_revision,
                "context_revision": context_revision,
                "anchors": anchors,
            })

    details: list[dict[str, object]] = []
    for target_index, (decision, audit) in enumerate(
        zip(state.decisions, state.audits)
    ):
        if not (
            audit.timing_trusted
            and decision.evidence.minimum_proof_satisfied
            and decision.evidence.ownership.status == "clear"
            and not decision.evidence.structural_failures
        ):
            continue
        primary_observation = raw_observations[target_index]
        for evaluation in decision.rejected_candidates:
            candidate = evaluation.candidate
            if (
                candidate.source != "raw-vocal-independent-fusion"
                or not candidate_occurrence_evidence_is_valid(candidate, entries)
            ):
                continue
            occurrence = candidate.occurrence_evidence
            assert occurrence is not None
            try:
                binding = json.loads(occurrence.binding_payload_json)
                target_raw_time = float(binding.get("timestamp"))
                target_raw_score = float(binding.get("score"))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not (
                isinstance(binding, dict)
                and math.isfinite(target_raw_time)
                and math.isfinite(target_raw_score)
                and target_raw_score >= TRUSTED_ALIGNMENT_SCORE
            ):
                continue

            proofs: list[dict[str, object]] = []
            if (
                primary_observation is not None
                and occurrence.evidence_revision
                == primary_observation.get("occurrence_evidence_revision")
            ):
                proofs.extend(contiguous_by_index.get(target_index, ()))
            target_raw_centiseconds = canonicalize_lrc_time(
                target_raw_time
            ).centiseconds
            for certificate in context_certificates:
                if any(
                    int(anchor.get("entry", -1)) == target_index + 1
                    and int(anchor.get("raw_centiseconds", -1))
                    == target_raw_centiseconds
                    and abs(float(anchor.get("raw_score", -1.0)) - target_raw_score)
                    <= 1e-6
                    for anchor in certificate["anchors"]  # type: ignore[index]
                ):
                    proofs.append(certificate)
            if not proofs:
                continue
            proof_modes = sorted({str(item["proof"]) for item in proofs})
            details.append({
                "entry": target_index + 1,
                "candidate_id": candidate.candidate_id,
                "occurrence_evidence_revision": occurrence.evidence_revision,
                "binding_revision": occurrence.binding_revision,
                "binding_mode": occurrence.occurrence_mode,
                "raw_binding_centiseconds": target_raw_centiseconds,
                "selected_centiseconds": decision.written_time.centiseconds,
                "proof_modes": proof_modes,
                "proofs": copy.deepcopy(proofs),
                "authority": "timing-alternative-falsification-only",
            })
    return tuple(
        sorted(details, key=lambda item: (int(item["entry"]), str(item["candidate_id"])))
    )


def _install_runtime_raw_timing_drift_falsifications(
    report: dict[str, object],
    state: FinalTimingState,
) -> dict[int, frozenset[str]]:
    runtime_entries = report.get(_RUNTIME_LYRIC_ENTRIES_KEY)
    entries = (
        tuple(runtime_entries)
        if isinstance(runtime_entries, (list, tuple))
        and len(runtime_entries) == len(state.decisions)
        and all(isinstance(item, LyricEntry) for item in runtime_entries)
        else ()
    )
    details = (
        _raw_timing_drift_falsification_details(state, entries)
        if entries
        else ()
    )
    by_entry: dict[int, set[str]] = collections.defaultdict(set)
    for item in details:
        entry = item.get("entry")
        candidate_id = item.get("candidate_id")
        if isinstance(entry, int) and isinstance(candidate_id, str):
            by_entry[entry].add(candidate_id)
    frozen = {
        entry: frozenset(sorted(candidate_ids))
        for entry, candidate_ids in by_entry.items()
    }
    report[_RUNTIME_RAW_TIMING_DRIFT_FALSIFICATIONS_KEY] = frozen
    report["raw_timing_drift_falsifications"] = {
        "schema_version": 1,
        "status": "runtime-evidence-generated" if entries else "BLOCKED",
        "authority": "timing-alternative-falsification-only",
        "content_identity_invalidated": False,
        "selector_mutated": False,
        "timestamp_mutated": False,
        "material_boundary_centiseconds": int(
            round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0)
        ),
        "minimum_raw_alignment_score": TRUSTED_ALIGNMENT_SCORE,
        "contiguous_min_entries": 3,
        "contiguous_max_adjacent_offset_change_seconds": 0.65,
        "max_abs_median_delta_seconds": RAW_TIMING_DRIFT_MAX_MEDIAN_DELTA_SECONDS,
        "entry_count": len(frozen),
        "falsification_count": len(details),
        "entries": list(details),
    }
    return frozen


def _runtime_raw_timing_drift_falsified_candidate_ids(
    report: dict[str, object] | None,
    entry_number: int,
) -> frozenset[str]:
    if not isinstance(report, dict):
        return frozenset()
    value = report.get(_RUNTIME_RAW_TIMING_DRIFT_FALSIFICATIONS_KEY)
    if not isinstance(value, dict):
        return frozenset()
    candidate_ids = value.get(entry_number)
    if isinstance(candidate_ids, frozenset) and all(
        isinstance(item, str) for item in candidate_ids
    ):
        return candidate_ids
    if isinstance(candidate_ids, (list, tuple, set)) and all(
        isinstance(item, str) for item in candidate_ids
    ):
        return frozenset(candidate_ids)
    return frozenset()


def _candidate_is_same_occurrence_boundary_equivalent(
    selected: TimingCandidate,
    alternative: TimingCandidate,
) -> bool:
    """Whether two distant written candidates prove one immutable boundary.

    Materiality is defined on boundary hypotheses, not on producer-specific
    timestamp representations.  Two candidates are therefore non-materially
    different when both independently bind the exact same lyric occurrence and
    their revision-bound cross-backend onset observations fall inside the
    existing canonical 18cs consensus tolerance.

    This predicate is symmetric and deliberately does not choose a timestamp,
    transfer trust, or merge candidate identities.  It only prevents the same
    already-proven occurrence/onset boundary from being counted as a distant
    material alternative because two producers encoded the candidate time at
    different points around that onset.
    """
    selected_occurrence = selected.occurrence_evidence
    alternative_occurrence = alternative.occurrence_evidence
    if (
        selected_occurrence is None
        or alternative_occurrence is None
        or not candidate_occurrence_evidence_is_valid(selected)
        or not candidate_occurrence_evidence_is_valid(alternative)
        or selected_occurrence.entry_index != alternative_occurrence.entry_index
        or selected_occurrence.lyric_identity_revision
        != alternative_occurrence.lyric_identity_revision
        or selected_occurrence.lyric_sha256 != alternative_occurrence.lyric_sha256
    ):
        return False

    if (
        selected.identity_support != "supported"
        or selected.sequence_support != "supported"
        or alternative.identity_support != "supported"
        or alternative.sequence_support != "supported"
        or not candidate_has_boundary_grade_direct_onset_evidence(selected)
        or not candidate_has_boundary_grade_direct_onset_evidence(alternative)
    ):
        return False

    selected_onset = selected.direct_onset_evidence
    alternative_onset = alternative.direct_onset_evidence
    if (
        selected_onset is None
        or alternative_onset is None
        or selected_onset.kind != "cross-backend-onset"
        or alternative_onset.kind != "cross-backend-onset"
    ):
        return False

    tolerance_centiseconds = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    selected_onset_centiseconds = canonicalize_lrc_time(
        selected_onset.onset_time
    ).centiseconds
    alternative_onset_centiseconds = canonicalize_lrc_time(
        alternative_onset.onset_time
    ).centiseconds
    # Material equivalence is stricter than the generic 22cs acoustic binding
    # allowance: each written representation must itself remain inside the
    # existing 18cs materiality contract around its bound onset.
    if (
        abs(selected.written_time.centiseconds - selected_onset_centiseconds)
        > tolerance_centiseconds
        or abs(
            alternative.written_time.centiseconds - alternative_onset_centiseconds
        ) > tolerance_centiseconds
    ):
        return False
    return bool(
        abs(selected_onset_centiseconds - alternative_onset_centiseconds)
        <= tolerance_centiseconds
    )


def _candidate_is_exact_evidence_subsumed_by_selected(
    selected: TimingCandidate,
    alternative: TimingCandidate,
) -> bool:
    """Whether ``alternative`` is only a weaker representation of ``selected``.

    This is intentionally narrower than same-lineage or wrapper deduplication.
    The two candidates must carry the exact same immutable occurrence proof,
    independently observed direct onsets from the same producer/kind inside the
    existing consensus tolerance, and the selected candidate must add exactly
    one genuinely independent CTC lineage backed by real CTC token geometry.

    The predicate never chooses a timestamp and never turns a different
    occurrence or a separated onset into the same hypothesis.  It only prevents
    one already-bound occurrence/onset hypothesis from being counted twice when
    the selected representation has strictly stronger independent evidence.
    """
    selected_occurrence = selected.occurrence_evidence
    alternative_occurrence = alternative.occurrence_evidence
    if (
        selected_occurrence is None
        or alternative_occurrence is None
        or not candidate_occurrence_evidence_is_valid(selected)
        or not candidate_occurrence_evidence_is_valid(alternative)
        or selected_occurrence.evidence_revision != alternative_occurrence.evidence_revision
    ):
        return False

    if (
        selected.identity_support != "supported"
        or selected.sequence_support != "supported"
        or alternative.identity_support != "supported"
        or alternative.sequence_support != "supported"
        or not candidate_has_bound_direct_onset_evidence(selected)
        or not candidate_has_bound_direct_onset_evidence(alternative)
    ):
        return False

    selected_onset = selected.direct_onset_evidence
    alternative_onset = alternative.direct_onset_evidence
    if selected_onset is None or alternative_onset is None:
        return False
    if (
        selected_onset.producer != alternative_onset.producer
        or selected_onset.kind != alternative_onset.kind
        or abs(selected_onset.onset_time - alternative_onset.onset_time)
        > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
    ):
        return False

    selected_families = _candidate_support_families(selected)
    alternative_families = _candidate_support_families(alternative)
    if selected_families != alternative_families | {"ctc"}:
        return False
    if "ctc" in alternative_families or not selected.spans:
        return False
    return True


VALID_ALTERNATIVE_MATERIAL_SECONDS = 1.0


def _material_distant_timing_alternatives(
    decision: TimingDecision,
    selected_time: float,
    *,
    falsified_candidate_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, object], ...]:
    """Return the one canonical set of qualifying material alternatives.

    This is the sole materiality owner.  The boolean predicate below delegates
    to this function, while the peer report consumes the same returned details.
    """

    tolerance = RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
    tolerance_centiseconds = int(round(tolerance * 100.0))
    selected_centiseconds = canonicalize_lrc_time(selected_time).centiseconds
    ownership_only_reasons = frozenset({
        "previous-tail-ownership-unknown",
        "external-tail-current-lineage-overlap",
        "external-tail-current-lineage-unresolved",
        "tail-conflict-without-independent-current-opening",
        "insufficient-minimum-proof",
    })
    details: list[dict[str, object]] = []
    for rejected in decision.rejected_candidates:
        candidate = rejected.candidate
        if (
            abs(candidate.written_time.centiseconds - selected_centiseconds)
            <= tolerance_centiseconds
        ):
            continue
        if _candidate_is_exact_evidence_subsumed_by_selected(decision.candidate, candidate):
            continue
        if _candidate_is_same_occurrence_boundary_equivalent(
            decision.candidate, candidate
        ):
            continue
        if candidate.candidate_id in falsified_candidate_ids:
            continue
        # UNKNOWN ownership can make an otherwise coherent, lyric-bound opening
        # fail minimum proof.  Preserve that distant hypothesis for review when
        # every recorded blocker is an ownership/minimum-proof consequence.
        # This does not admit weak, incoherent, malformed, or diagnostic-only
        # candidates: opening geometry, identity, sequence, and support remain
        # independently required.
        if (
            rejected.evidence.ownership.status == "unknown"
            and set(rejected.rejection_reasons).issubset(ownership_only_reasons)
            and (
                rejected.evidence.onset_proof_end is not None
                or candidate_has_bound_direct_onset_evidence(candidate)
            )
            and rejected.evidence.temporal_coherence == "coherent"
            and candidate.identity_support == "supported"
            and candidate.sequence_support == "supported"
        ):
            qualifies = True
        elif (
            rejected.valid
            and abs(candidate.written_time.seconds - selected_time)
            > VALID_ALTERNATIVE_MATERIAL_SECONDS
            and "recurrence" not in _candidate_support_families(candidate)
            and candidate.identity_support == "supported"
            and candidate.sequence_support == "supported"
            and not _candidate_has_revision_bound_ownership_violation(rejected)
        ):
            # Central found this lyric-bound alternative fully valid; it lost
            # only by not dominating.  Two valid clocks over a second apart
            # mean the row's timing is not established, whatever else agrees.
            qualifies = True
        else:
            if not rejected.evidence.minimum_proof_satisfied:
                continue
            # An explicit, revision-bound ownership violation is an existing
            # structural falsification. UNKNOWN ownership, malformed violation
            # evidence, or an unbound relation remain material hypotheses.
            if _candidate_has_revision_bound_ownership_violation(rejected):
                continue
            if (
                candidate.identity_support != "supported"
                or candidate.sequence_support != "supported"
                or not candidate_has_bound_direct_onset_evidence(candidate)
            ):
                continue
            qualifies = True
        if not qualifies:
            continue
        detail = _candidate_evidence_projection(rejected)
        detail.update({
            "delta_seconds": round(candidate.written_time.seconds - selected_time, 6),
            "explicit_falsification": False,
            "explicit_falsification_reason": (
                "violation-not-revision-bound"
                if rejected.evidence.ownership.status == "violation"
                else "no-explicit-revision-bound-violation"
            ),
        })
        details.append(detail)
    return tuple(details)


def _decision_has_material_distant_timing_hypothesis(
    decision: TimingDecision,
    selected_time: float,
    *,
    falsified_candidate_ids: frozenset[str] = frozenset(),
) -> bool:
    """Whether the canonical material owner returns any alternative."""

    return bool(
        _material_distant_timing_alternatives(
            decision,
            selected_time,
            falsified_candidate_ids=falsified_candidate_ids,
        )
    )


def _candidate_has_revision_bound_ownership_violation(
    evaluation: CandidateEvaluation,
) -> bool:
    """Whether an alternative is explicitly falsified by bound tail ownership."""
    ownership = evaluation.evidence.ownership
    candidate = evaluation.candidate
    revision_bound = bool(
        ownership.status == "violation"
        and ownership.previous_tail_validity == "valid"
        and isinstance(ownership.previous_tail_evidence_revision, str)
        and bool(ownership.previous_tail_evidence_revision)
        and isinstance(ownership.relation_revision, str)
        and bool(ownership.relation_revision)
        and ownership.current_candidate_id == candidate.candidate_id
        and ownership.current_generation_revision == candidate.generation_revision
        and ownership.current_region_revision == evaluation.evidence.region_revision
        and ownership.current_centiseconds == candidate.written_time.centiseconds
    )
    if not revision_bound:
        return False
    # An external tail cannot falsify a current candidate that depends on the
    # same backend lineage.  The authority metadata is carried through the
    # ownership relation so this materiality layer cannot lose that boundary.
    if ownership.previous_tail_authority_backend_lineage is not None:
        relation = _current_tail_lineage_relation(
            candidate,
            ownership.previous_tail_authority_backend_lineage,
            ownership.previous_tail_authority_producer,
            ownership.previous_tail_authority_observation_revision,
        )
        if relation not in {"disjoint", "direct-disjoint"}:
            return False
    return True


def _final_post_central_material_conflict_entries(
    state: FinalTimingState,
    report: dict[str, object] | None = None,
) -> frozenset[int]:
    """Return entries with unresolved material alternatives in final Central state.

    The decision graph is the authority here, rather than the pre-central
    suspicious diagnostic list.  Runtime raw-timing drift may falsify only the
    exact raw-vocal timing candidates proven by the same immutable Central
    graph; it never changes selection, timestamps, or content identity.
    """
    drift = (
        _install_runtime_raw_timing_drift_falsifications(report, state)
        if isinstance(report, dict)
        else {}
    )
    return frozenset(
        decision.entry_index + 1
        for decision, _audit in zip(state.decisions, state.audits)
        if _decision_has_material_distant_timing_hypothesis(
            decision,
            decision.written_time.seconds,
            falsified_candidate_ids=drift.get(
                decision.entry_index + 1, frozenset()
            ),
        )
    )


def _precentral_candidate_gap_exceeds_half_second(
    row: object,
    selected_time: float,
) -> bool:
    """Retain review if ANY historical candidate differs by over 0.5 second."""
    if not isinstance(row, dict):
        return False
    observed = False
    values = row.get("candidate_timestamps")
    if isinstance(values, dict):
        for value in values.values():
            if isinstance(value, bool):
                continue
            try:
                timestamp = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(timestamp):
                continue
            observed = True
            if abs(timestamp - selected_time) > 0.5:
                return True
    precentral = row.get("precentral_diagnostic")
    precentral_candidates = (
        precentral.get("candidates") if isinstance(precentral, dict) else None
    )
    if isinstance(precentral_candidates, list):
        for candidate in precentral_candidates:
            if not isinstance(candidate, dict):
                continue
            value = candidate.get("time")
            if isinstance(value, bool):
                continue
            try:
                timestamp = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(timestamp):
                continue
            observed = True
            if abs(timestamp - selected_time) > 0.5:
                return True
    if not observed:
        try:
            spread = float(row.get("candidate_disagreement_seconds"))
        except (ValueError, TypeError):
            spread = 0.0
        return math.isfinite(spread) and spread > 0.5
    return False


def _central_decision_supersedes_suspicious_row(
    decision: TimingDecision,
    audit: TimingInvariantAudit,
    row: object,
    report: object = None,
) -> bool:
    """Whether final central evidence has actually resolved a pre-central review row."""
    if not audit.timing_trusted or not isinstance(row, dict):
        return False
    flags = row.get("flags")
    if not isinstance(flags, list):
        return False
    actionable = {
        str(flag) for flag in flags
        if str(flag) in ACTIONABLE_SUSPICIOUS_FLAGS
    }
    if not actionable or not actionable.issubset(CENTRAL_RESOLVABLE_SUSPICIOUS_FLAGS):
        return False
    if not decision.evidence.minimum_proof_satisfied or decision.evidence.structural_failures:
        return False
    if (
        "phonetic_anchor_disagreement" in actionable
        and not candidate_has_bound_direct_onset_evidence(decision.candidate)
    ):
        return False
    selected_time = decision.written_time.seconds
    if _precentral_candidate_gap_exceeds_half_second(row, selected_time):
        return False
    drift_falsified_ids = _runtime_raw_timing_drift_falsified_candidate_ids(
        report if isinstance(report, dict) else None,
        decision.entry_index + 1,
    )
    if "unresolved_raw_ctc_disagreement" in actionable:
        candidate_timestamps = row.get("candidate_timestamps")
        raw_time = None
        if isinstance(candidate_timestamps, dict):
            try:
                raw_time = float(candidate_timestamps.get("raw_asr"))
            except (TypeError, ValueError):
                raw_time = None
        if (
            raw_time is not None
            and math.isfinite(raw_time)
            and abs(raw_time - selected_time) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            and not _explicit_whisper_arbitration_resolves_raw_disagreement(
                report, row, selected_time
            )
            and _decision_has_material_distant_timing_hypothesis(
                decision,
                selected_time,
                falsified_candidate_ids=drift_falsified_ids,
            )
        ):
            # The historical Raw timestamp is diagnostic-only unless the final
            # decision graph still contains a genuinely material distant timing
            # hypothesis.  A material competitor must bind lyric identity and
            # sequence to revision-bound independent direct-onset evidence.  Its
            # current selection validity may still fail on ownership/structure;
            # that is not equivalent to falsifying the timing hypothesis.
            return False
    # A distant *valid* alternative remains a real disagreement. Invalid
    # candidates, runtime-falsified raw timing alternatives, or valid candidates
    # in the same canonical-centisecond cluster do not.  This consumer must use
    # the same falsification set and 0.18 s centisecond boundary as the final
    # materiality owner; otherwise a pre-central diagnostic can resurrect an
    # alternative that Central has already proven to be timing drift.
    tolerance_centiseconds = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    selected_centiseconds = decision.written_time.centiseconds
    for rejected in decision.rejected_candidates:
        if rejected.candidate.candidate_id in drift_falsified_ids:
            continue
        if not rejected.valid:
            continue
        if _candidate_is_same_occurrence_boundary_equivalent(
            decision.candidate, rejected.candidate
        ):
            continue
        if (
            abs(rejected.candidate.written_time.centiseconds - selected_centiseconds)
            > tolerance_centiseconds
        ):
            return False
    return True


def _recompute_duplicate_occurrence_offset_diagnostics(
    entries: list[LyricEntry],
    report: dict[str, object],
    state: FinalTimingState,
) -> dict[str, object]:
    """Rebind duplicate-offset diagnostics to the current final timing state.

    The diagnostic is timestamp-derived review evidence.  If a private Central
    replay changes an occurrence timestamp, retaining the pre-replay delta would
    make the suspicious owner stale.  This recomputes only that existing owner;
    it does not select or alter timestamps and it does not weaken the HuBERT
    duplicate resolver.
    """
    assignments = report.get("assignments")
    suspicious = report.get("suspicious_alignments")
    if (
        not isinstance(assignments, list)
        or not isinstance(suspicious, list)
        or len(assignments) != len(entries)
        or len(state.written_times) != len(entries)
    ):
        payload = {
            "schema_version": 1,
            "authority": "review-only",
            "status": "CONTEXT_UNAVAILABLE",
            "outlier_count": 0,
            "outliers": [],
        }
        payload["revision"] = _timing_content_digest(payload)
        return payload

    flag = "duplicate_occurrence_offset_mismatch"
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        flags = assignment.get("flags")
        if isinstance(flags, list):
            assignment["flags"] = [
                item for item in flags if str(item) != flag
            ]
        assignment.pop(flag, None)
    for row in suspicious:
        if not isinstance(row, dict):
            continue
        flags = row.get("flags")
        if isinstance(flags, list):
            row["flags"] = [item for item in flags if str(item) != flag]
        row.pop(flag, None)
        if row.get("resolution") == "independent-hubert-duplicate-occurrence-confirmation":
            row.pop("resolution", None)

    timestamps = [item.seconds for item in state.written_times]
    consensus = duplicate_lyric_offset_consensus(entries, timestamps)
    report["duplicate_lyric_offset_consensus"] = consensus
    cluster_medians: dict[int, float] = {}
    clusters = consensus.get("clusters")
    if isinstance(clusters, list):
        for cluster in clusters:
            if not isinstance(cluster, dict) or not cluster.get("available"):
                continue
            try:
                cluster_medians[int(cluster["entry_delta"])] = float(
                    cluster["median_delta_seconds"]
                )
            except (KeyError, TypeError, ValueError):
                continue

    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        normalized = normalize_match_text(entry_sung_text(entry))
        if normalized:
            groups.setdefault(normalized, []).append(index)

    risk_by_entry = {
        int(item["entry"]): item
        for item in suspicious
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    outliers: list[dict[str, object]] = []
    for indexes in groups.values():
        if len(indexes) != 2:
            continue
        first, second = indexes
        pair_median = cluster_medians.get(second - first)
        if pair_median is None:
            continue
        observed_delta = timestamps[second] - timestamps[first]
        error = observed_delta - pair_median
        if abs(error) <= 0.35:
            continue
        pair = [first, second]
        recovered = [
            index
            for index in pair
            if isinstance(assignments[index], dict)
            and (
                assignments[index].get("ctc_forward_supported_prefix_recovery")
                or assignments[index].get("ctc_crossline_initial_recovery")
            )
        ]
        flagged_index = recovered[0] if len(recovered) == 1 else first
        assignment = assignments[flagged_index]
        if not isinstance(assignment, dict):
            continue
        detail = {
            "entries": [first + 1, second + 1],
            "observed_delta_seconds": round(observed_delta, 3),
            "median_delta_seconds": round(pair_median, 3),
            "error_seconds": round(error, 3),
        }
        assignment[flag] = detail
        flags = assignment.get("flags")
        assignment["flags"] = list(dict.fromkeys([
            *(flags if isinstance(flags, list) else []),
            flag,
        ]))
        risk = risk_by_entry.get(flagged_index + 1)
        if not isinstance(risk, dict):
            risk = {
                "entry": flagged_index + 1,
                "text": entry_sung_text(entries[flagged_index]),
                "flags": [],
            }
            suspicious.append(risk)
            risk_by_entry[flagged_index + 1] = risk
        risk[flag] = copy.deepcopy(detail)
        risk_flags = risk.get("flags")
        risk["flags"] = list(dict.fromkeys([
            *(risk_flags if isinstance(risk_flags, list) else []),
            flag,
        ]))
        risk["severity"] = "high"
        risk["review_required"] = True
        outliers.append({"entry": flagged_index + 1, **detail})

    report["duplicate_occurrence_offset_outliers"] = outliers
    report["duplicate_occurrence_offset_outlier_count"] = len(outliers)
    report.pop(_RUNTIME_POSTCENTRAL_HUBERT_DUPLICATE_RESOLUTION_KEY, None)
    payload = {
        "schema_version": 1,
        "authority": "review-only",
        "status": "RECOMPUTED",
        "consensus_available": bool(consensus.get("available")),
        "outlier_count": len(outliers),
        "outliers": copy.deepcopy(outliers),
    }
    payload["revision"] = _timing_content_digest(payload)
    return payload


def _runtime_hubert_duplicate_occurrence_resolution(
    report: dict[str, object],
    state: FinalTimingState,
) -> dict[str, object]:
    """Resolve only duplicate-offset *review* when HuBERT reproduces both occurrences.

    This never changes a timestamp.  Both duplicate occurrences must have exact
    hypothesis-set-bound full-line HuBERT observations corroborating their immutable
    final timestamps, and the independent HuBERT delta must itself remain outside the
    existing 0.35 s duplicate-section tolerance.
    """
    payload: dict[str, object] = {
        "schema_version": 1,
        "authority": "review-only",
        "resolved_entries": [],
        "rows": [],
    }
    projection = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    details = (
        projection.get("cross_backend_material_conflicts")
        if isinstance(projection, dict) else None
    )
    bound_entries = {
        int(value)
        for value in (
            details.get("hubert_current_onset_hypothesis_bound_entries", [])
            if isinstance(details, dict) else []
        )
        if isinstance(value, int)
    }
    observations = _postcentral_hubert_current_onset_map(
        report.get(_RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY)
    )
    outliers = report.get("duplicate_occurrence_offset_outliers")
    if not isinstance(outliers, list) or not observations or not bound_entries:
        payload["revision"] = _timing_content_digest(payload)
        return payload
    peer_tolerance_centiseconds = int(round(
        RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0
    ))
    duplicate_tolerance_centiseconds = int(round(0.35 * 100.0))
    resolved: set[int] = set()
    rows: list[dict[str, object]] = []
    for item in outliers:
        if not isinstance(item, dict):
            continue
        flagged_entry = item.get("entry")
        pair = item.get("entries")
        if (
            not isinstance(flagged_entry, int)
            or not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(value, int) for value in pair)
        ):
            continue
        first_entry, second_entry = int(pair[0]), int(pair[1])
        if not (
            1 <= first_entry <= len(state.decisions)
            and 1 <= second_entry <= len(state.decisions)
            and first_entry in bound_entries
            and second_entry in bound_entries
        ):
            continue
        first_observation = observations.get(first_entry - 1)
        second_observation = observations.get(second_entry - 1)
        if not (
            _known_lyric_hubert_current_onset_is_valid(first_observation)
            and _known_lyric_hubert_current_onset_is_valid(second_observation)
        ):
            continue
        assert first_observation is not None and second_observation is not None
        first_final = state.decisions[first_entry - 1].written_time.centiseconds
        second_final = state.decisions[second_entry - 1].written_time.centiseconds
        first_observed = canonicalize_lrc_time(
            first_observation.onset_time
        ).centiseconds
        second_observed = canonicalize_lrc_time(
            second_observation.onset_time
        ).centiseconds
        if (
            abs(first_observed - first_final) > peer_tolerance_centiseconds
            or abs(second_observed - second_final) > peer_tolerance_centiseconds
        ):
            continue
        try:
            reported_delta_centiseconds = int(round(
                float(item["observed_delta_seconds"]) * 100.0
            ))
            median_delta_centiseconds = int(round(
                float(item["median_delta_seconds"]) * 100.0
            ))
        except (KeyError, TypeError, ValueError):
            continue
        final_delta_centiseconds = second_final - first_final
        acoustic_delta_centiseconds = second_observed - first_observed
        if reported_delta_centiseconds != final_delta_centiseconds:
            continue
        # The independent HuBERT pair must reproduce the same mismatch class; if
        # it falls back within the existing 35 cs section tolerance, retain review.
        if (
            abs(acoustic_delta_centiseconds - median_delta_centiseconds)
            <= duplicate_tolerance_centiseconds
        ):
            continue
        resolved.add(flagged_entry)
        rows.append({
            "entry": flagged_entry,
            "entries": [first_entry, second_entry],
            "first_observation_revision": first_observation.observation_revision,
            "second_observation_revision": second_observation.observation_revision,
            "final_delta_centiseconds": final_delta_centiseconds,
            "hubert_delta_centiseconds": acoustic_delta_centiseconds,
            "median_delta_centiseconds": median_delta_centiseconds,
            "resolution": "independent-hubert-reproduces-duplicate-offset-mismatch",
        })
    payload["resolved_entries"] = sorted(resolved)
    payload["rows"] = sorted(rows, key=lambda row: int(row["entry"]))
    payload["revision"] = _timing_content_digest(payload)
    return payload


def _runtime_hubert_duplicate_resolution_entries(
    report: dict[str, object],
) -> frozenset[int]:
    payload = report.get(_RUNTIME_POSTCENTRAL_HUBERT_DUPLICATE_RESOLUTION_KEY)
    if not isinstance(payload, dict) or payload.get("authority") != "review-only":
        return frozenset()
    revision = payload.get("revision")
    expected = _timing_content_digest({
        key: value for key, value in payload.items() if key != "revision"
    })
    if not isinstance(revision, str) or revision != expected:
        return frozenset()
    values = payload.get("resolved_entries")
    if not isinstance(values, list):
        return frozenset()
    return frozenset(value for value in values if isinstance(value, int) and value > 0)


def _suspicious_entry_requires_review_for_decision(
    report: dict[str, object],
    entry_number: int,
    decision: TimingDecision,
    audit: TimingInvariantAudit,
) -> bool:
    suspicious = report.get("suspicious_alignments")
    if not isinstance(suspicious, list):
        return False
    for row in suspicious:
        if not isinstance(row, dict) or row.get("entry") != entry_number:
            continue
        if not _suspicious_row_requires_review(row):
            continue
        if entry_number in _runtime_hubert_duplicate_resolution_entries(report):
            flags = row.get("flags")
            actionable = {
                str(flag) for flag in flags
                if isinstance(flags, list) and str(flag) in ACTIONABLE_SUSPICIOUS_FLAGS
            }
            if actionable == {"duplicate_occurrence_offset_mismatch"}:
                continue
        if _central_decision_supersedes_suspicious_row(decision, audit, row, report):
            continue
        return True
    return False


def _project_suspicious_alignments_to_central_state(
    report: dict[str, object],
    state: FinalTimingState,
    *,
    material_conflict_entries: frozenset[int] | None = None,
    review_only_entries: frozenset[int] | None = None,
) -> None:
    """Make diagnostic `output` fields describe the committed central output.

    Pre-central candidate scoring remains available under `precentral_diagnostic`;
    it is never allowed to masquerade as the final written timestamp.
    """
    suspicious = report.get("suspicious_alignments")
    assignments = report.get("assignments")
    if not isinstance(suspicious, list) or not isinstance(assignments, list):
        return
    if material_conflict_entries is None:
        material_conflict_entries = _final_post_central_material_conflict_entries(state, report)
    if review_only_entries is None:
        review_only_entries = frozenset()
    decisions = {decision.entry_index + 1: decision for decision in state.decisions}
    audits = {audit.entry_index + 1: audit for audit in state.audits}
    for row in suspicious:
        if not isinstance(row, dict) or not isinstance(row.get("entry"), int):
            continue
        entry = int(row["entry"])
        if not (1 <= entry <= len(assignments)):
            continue
        assignment = assignments[entry - 1]
        decision = decisions.get(entry)
        audit = audits.get(entry)
        if not isinstance(assignment, dict) or decision is None or audit is None:
            continue
        old_payload = {
            key: copy.deepcopy(row[key])
            for key in (
                "chosen_time", "confidence", "candidate_timestamps", "candidates",
                "rejected_candidates", "reasons", "penalties", "review_required", "severity",
            )
            if key in row
        }
        if old_payload and "precentral_diagnostic" not in row:
            row["precentral_diagnostic"] = old_payload
        final_time = decision.written_time.seconds
        candidate_times = row.get("candidate_timestamps")
        candidate_times = dict(candidate_times) if isinstance(candidate_times, dict) else {}
        candidate_times["output"] = final_time
        row["candidate_timestamps"] = candidate_times
        row["chosen_time"] = final_time
        row["confidence"] = round(decision.candidate.confidence, 3)
        row["canonical_written_centiseconds"] = decision.written_time.centiseconds
        row["lrc_tag"] = decision.written_time.lrc_tag
        # These are existing Central decision revisions.  Keeping them on the
        # projected diagnostic prevents a pre-central scalar from masquerading
        # as a current output-derived value.
        row["timestamp_revision"] = decision.timestamp_revision
        row["timing_evidence_revision"] = decision.timing_evidence_revision
        row["central_candidate_id"] = decision.candidate.candidate_id
        row["central_candidate_source"] = decision.candidate.source
        row["candidates"] = copy.deepcopy(assignment.get("candidates", []))
        row["rejected_candidates"] = copy.deepcopy(assignment.get("rejected_candidates", []))
        alternative_deltas = [
            abs(float(value) - final_time)
            for key, value in candidate_times.items()
            if key != "output"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ]
        if alternative_deltas:
            row["candidate_disagreement_seconds"] = round(max(alternative_deltas), 3)
        else:
            row.pop("candidate_disagreement_seconds", None)
        flags = row.get("flags")
        if (
            isinstance(flags, list)
            and (
                "unresolved_raw_ctc_disagreement" in flags
                or "unresolved_ctc_disagreement" in flags
            )
            and "resolution" not in row
        ):
            projected_severities: list[str] = []
            candidate_output = candidate_times.get("output")
            if "unresolved_raw_ctc_disagreement" in flags:
                raw_score = row.get("raw_asr_score")
                raw_time = candidate_times.get("raw_asr")
                if (
                    isinstance(raw_score, (int, float))
                    and not isinstance(raw_score, bool)
                    and math.isfinite(float(raw_score))
                    and isinstance(candidate_output, (int, float))
                    and not isinstance(candidate_output, bool)
                    and math.isfinite(float(candidate_output))
                    and isinstance(raw_time, (int, float))
                    and not isinstance(raw_time, bool)
                    and math.isfinite(float(raw_time))
                ):
                    projected_severities.append(
                        _raw_disagreement_severity(
                            float(raw_score),
                            abs(float(raw_time) - float(candidate_output)),
                        )
                    )
            if "unresolved_ctc_disagreement" in flags:
                ctc_time = candidate_times.get("ctc")
                if (
                    isinstance(candidate_output, (int, float))
                    and not isinstance(candidate_output, bool)
                    and math.isfinite(float(candidate_output))
                    and isinstance(ctc_time, (int, float))
                    and not isinstance(ctc_time, bool)
                    and math.isfinite(float(ctc_time))
                ):
                    projected_severities.append(
                        _ctc_disagreement_severity(
                            abs(float(ctc_time) - float(candidate_output))
                        )
                    )
            if projected_severities:
                projected_severity = (
                    "high" if "high" in projected_severities else "medium"
                )
                if _suspicious_row_has_independent_high_severity_owner(row):
                    projected_severity = "high"
                row["severity"] = projected_severity
        duplicate_review_resolved = False
        if entry in _runtime_hubert_duplicate_resolution_entries(report):
            flags = row.get("flags")
            actionable = {
                str(flag) for flag in flags
                if isinstance(flags, list) and str(flag) in ACTIONABLE_SUSPICIOUS_FLAGS
            }
            duplicate_review_resolved = (
                actionable == {"duplicate_occurrence_offset_mismatch"}
            )
        if duplicate_review_resolved:
            row["review_required"] = False
            row["severity"] = "resolved"
            row["resolution"] = "independent-hubert-duplicate-occurrence-confirmation"
        elif (
            entry not in review_only_entries
            and _central_decision_supersedes_suspicious_row(decision, audit, row, report)
        ):
            row["review_required"] = False
            row["severity"] = "resolved"
            row["resolution"] = "superseded-by-verified-central-decision"
        else:
            if _precentral_candidate_gap_exceeds_half_second(
                row, decision.written_time.seconds
            ):
                flags = row.get("flags")
                if isinstance(flags, list) and any(
                    str(flag) in CENTRAL_RESOLVABLE_SUSPICIOUS_FLAGS
                    for flag in flags
                ):
                    if "unresolved-precentral-disagreement" not in flags:
                        flags.append("unresolved-precentral-disagreement")
                    row.pop("resolution", None)
                    if row.get("severity") == "resolved":
                        row["severity"] = "medium"
            row["review_required"] = bool(
                _suspicious_row_requires_review(row)
                or entry in material_conflict_entries
            )
    report["suspicious_alignment_count"] = len([row for row in suspicious if isinstance(row, dict)])
    report["suspicious_alignment_severity_counts"] = _actionable_suspicious_severity_counts(report)


def compute_final_timing_self_check(
    report: dict[str, object],
    state: FinalTimingState,
    entries: list[LyricEntry],
) -> dict[str, object]:
    """Fail-closed final confidence audit. Never changes or reranks timestamps.

    F1 probes the isolated Demucs vocal stem, not the original mix.
    F2 identifies repeated text whose chosen time depends only on content identity.
    F3 flags runs of >=3 compressed or nonmonotonic consecutive lines.
    F4 flags a >1s displacement from the original CTC first token.
    """
    times = [decision.written_time.seconds for decision in state.decisions]
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(times) or len(entries) != len(times):
        raise LrcError("Final self-check must see aligned entries, assignments and decisions")
    flagged: dict[int, list[str]] = collections.defaultdict(list)
    diagnostics: dict[str, object] = {
        "schema": 1, "timestamp_mutations": 0, "vocal_status": "skipped-no-stem",
        "vocal_threshold_db": None,
    }
    vocal = vocal_stem_levels(report)
    diagnostics["vocal_status"] = vocal.status
    if vocal.status == "checked" and vocal.threshold_db is not None:
        diagnostics["vocal_threshold_db"] = round(vocal.threshold_db, 3)
        diagnostics["vocal_active_windows"] = vocal.active_windows
        for i, seconds in enumerate(times, 1):
            if vocal.absent(seconds):
                flagged[i].append("F1-vocal-absent")
    elif vocal.error is not None:
        diagnostics["vocal_error"] = vocal.error
    vocal_check_required = report.get("ctc_audio_source") == "vocal-stem"
    if vocal_check_required and diagnostics["vocal_status"] != "checked":
        for i in range(1, len(times) + 1):
            flagged[i].append("F1-vocal-check-unavailable")
    counts = collections.Counter(
        normalize_match_text(entry_sung_text(entry))
        for entry in entries
    )
    for i, (entry, decision, assignment) in enumerate(zip(entries, state.decisions, assignments), 1):
        key = normalize_match_text(entry_sung_text(entry))
        if counts[key] >= 2 and (
            "raw-vocal-independent-fusion" in decision.candidate.source
            or "content-identity" in decision.candidate.source
            or assignment.get("content_trust_source") in (
                "selected-candidate-independent-identity",
                "runtime-independent-content-identity",
            ) and decision.candidate.source.startswith("raw-")
        ):
            flagged[i].append("F2-repeated-content-identity")
        first = assignment.get("global_ctc_first_token_start", assignment.get("ctc_first_token_start"))
        if isinstance(first, (int, float)) and math.isfinite(float(first)) and abs(times[i - 1] - float(first)) > 1.0:
            flagged[i].append("F4-ctc-final-disagreement")
        sk = report.get("segment_skeleton_status")
        global_first = assignment.get("global_ctc_first_token_start")
        if (isinstance(sk, dict) and sk.get("status") == "SEGMENT_CURRENT"
            and counts[key] >= 2 and isinstance(global_first, (int, float))
            and math.isfinite(float(global_first))
            and abs(times[i - 1] - float(global_first)) > 3.0):
            flagged[i].append("skeleton_global_ctc_disagreement")
    gaps=[b-a for a,b in zip(times,times[1:])]
    # Any backward timestamp is inherently invalid, even for only two lines.
    for index, gap in enumerate(gaps, 1):
        if gap < 0:
            flagged[index].append("F3-nonmonotonic")
            flagged[index + 1].append("F3-nonmonotonic")
    positive=[d for d in gaps if d>0]
    median_gap=float(statistics.median(positive)) if positive else 0.0
    if median_gap:
        compressed = [gap <= 0 or gap < median_gap * 0.45 for gap in gaps]
        start = 0
        while start < len(compressed):
            if not compressed[start]:
                start += 1
                continue
            end=start
            while end+1<len(compressed) and compressed[end+1]:
                end+=1
            if end-start+2>=3:
                for i in range(start+1,end+3):
                    flagged[i].append("F3-compressed-or-nonmonotonic")
            start=end+1
    else:
        for i,diff in enumerate(gaps,1):
            if diff <= 0:
                flagged[i].append("F3-compressed-or-nonmonotonic")
                flagged[i+1].append("F3-compressed-or-nonmonotonic")
    result={str(k):list(dict.fromkeys(v)) for k,v in sorted(flagged.items())}
    diagnostics["flags_by_entry"]=result
    diagnostics["flagged_entries"]=sorted(flagged)
    diagnostics["flag_count_by_type"]=dict(collections.Counter(x for v in result.values() for x in v))
    diagnostics["median_interline_gap_s"]=round(median_gap,3)
    return diagnostics


def commit_final_timing_state(
    state: FinalTimingState,
    report: dict[str, object],
    generation_diagnostics: dict[str, object],
    *,
    finding_states: tuple[EntryTimingFindings, ...] | None = None,
    final_output: bool = False,
) -> FinalTimingState:
    """The only writer of final assignment timestamps and timing trust."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(state.decisions):
        raise LrcError("Final timing state does not match report assignments")
    runtime_entries = report.get(_RUNTIME_LYRIC_ENTRIES_KEY)
    occurrence_entries = (
        tuple(runtime_entries)
        if isinstance(runtime_entries, (list, tuple))
        and len(runtime_entries) == len(state.decisions)
        and all(isinstance(item, LyricEntry) for item in runtime_entries)
        else None
    )
    if finding_states is None:
        effective_state, finding_states = evaluate_final_timing_state_for_report(
            state,
            report,
            occurrence_entries=occurrence_entries,
        )
    else:
        if len(finding_states) != len(state.decisions):
            raise LrcError("Evaluated timing findings do not match timing decisions")
        effective_state = state
    material_conflict_entries = (
        _final_post_central_material_conflict_entries(effective_state, report)
        | _runtime_peer_review_only_conflicts(report)
    )
    # Only the final output audit may downgrade trust. Preview and private
    # recovery projections retain their original backend-selection semantics.
    raw_self_check = report.get("final_self_check") if final_output else None
    self_check_flags = (
        {int(index): flags for index, flags in raw_self_check.get("flags_by_entry", {}).items()}
        if isinstance(raw_self_check, dict) else {}
    )
    if final_output and (material_conflict_entries or self_check_flags):
        effective_state = FinalTimingState(
            effective_state.decisions,
            effective_state.written_times,
            tuple(
                TimingInvariantAudit(
                    entry_index=audit.entry_index,
                    timing_trusted=False,
                    status="provisional_unresolved",
                    failures=tuple(dict.fromkeys((
                        *audit.failures,
                        *(("unresolved-material-alternative",) if decision.entry_index + 1 in material_conflict_entries else ()),
                        *self_check_flags.get(decision.entry_index + 1, ()),
                    ))),
                )
                if decision.entry_index + 1 in (material_conflict_entries | self_check_flags.keys())
                else audit
                for decision, audit in zip(effective_state.decisions, effective_state.audits)
            ),
        )
    final_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    review_reason_entries: dict[str, list[int]] = collections.defaultdict(list)
    for decision, audit, raw_assignment, entry_findings in zip(
        effective_state.decisions,
        effective_state.audits,
        assignments,
        finding_states,
    ):
        if not isinstance(raw_assignment, dict):
            raise LrcError("Final timing commit encountered an invalid assignment")
        if "reasons" in raw_assignment:
            raw_assignment["precentral_reasons"] = copy.deepcopy(
                raw_assignment.pop("reasons")
            )
        if "penalties" in raw_assignment:
            raw_assignment["precentral_penalties"] = copy.deepcopy(
                raw_assignment.pop("penalties")
            )
        legacy_flags = raw_assignment.get("flags")
        flag_list = list(legacy_flags) if isinstance(legacy_flags, list) else []
        non_timing_flags = [flag for flag in flag_list if flag not in CTC_BOUNDARY_REVIEW_FLAGS]
        if entry_findings.resolved and decision.recovery is not None:
            recovery = decision.recovery
            resolved_findings = [
                {
                    "finding_id": finding.finding_id,
                    "code": finding.code,
                    "raised_revision": finding.raised_revision,
                    "evidence_scope": finding.evidence_scope,
                    "resolved_by_candidate_id": recovery.candidate_id,
                    "resolved_candidate_revision": recovery.candidate_revision,
                    "resolved_evidence_revision": recovery.evidence_revision,
                    "resolved_canonical_written_centiseconds": (
                        recovery.canonical_written_centiseconds
                    ),
                    "resolved_selection_revision": decision.selection_revision,
                    "passed_invariants": list(recovery.passed_invariants),
                    "resolution_mode": recovery.resolution_mode,
                    "evaluator_version": recovery.evaluator_version,
                }
                for finding in entry_findings.resolved
            ]
        else:
            resolved_findings = []
        unresolved_code_set = {
            finding.code for finding in entry_findings.unresolved
        }
        active_central_findings = [
            {
                "finding_id": _timing_content_digest({
                    "entry": decision.entry_index,
                    "failure": failure,
                    "selection_revision": decision.selection_revision,
                }),
                "code": failure,
                "raised_revision": decision.selection_revision,
                "evidence_scope": "active-central-evaluator",
            }
            for failure in audit.failures
            if failure not in unresolved_code_set
        ]
        active_historical_findings = [
            {
                "finding_id": finding.finding_id,
                "code": finding.code,
                "raised_revision": finding.raised_revision,
                "evidence_scope": "active-historical-boundary",
                "historical_evidence_scope": finding.evidence_scope,
            }
            for finding in entry_findings.unresolved
        ]
        active_findings = [*active_historical_findings, *active_central_findings]
        active_codes = list(dict.fromkeys(
            [str(item["code"]) for item in active_findings if isinstance(item.get("code"), str)]
        ))

        raw_assignment["timestamp"] = decision.written_time.seconds
        raw_assignment["chosen_time"] = decision.written_time.seconds
        raw_assignment["canonical_written_centiseconds"] = decision.written_time.centiseconds
        raw_assignment["lrc_tag"] = decision.written_time.lrc_tag
        raw_assignment["timestamp_status"] = audit.status
        raw_assignment["selection_revision"] = decision.selection_revision
        raw_assignment["timestamp_revision"] = decision.timestamp_revision
        raw_assignment["timing_evidence_revision"] = decision.timing_evidence_revision
        raw_assignment["selected_candidate_id"] = (
            decision.selected_candidate_id if audit.timing_trusted else None
        )
        raw_assignment["provisional_candidate_id"] = (
            decision.provisional_candidate_id or decision.candidate.candidate_id
            if not audit.timing_trusted else None
        )
        raw_assignment["candidate_source"] = decision.candidate.source
        raw_assignment["confidence"] = round(decision.candidate.confidence, 6)
        if decision.arbiter is not None:
            raw_assignment["arbiter"] = {
                "mode": "gross-rescue",
                "reverted_to_current": decision.arbiter.reverted_to_current,
                "reason": decision.arbiter.reason,
                "selected_source": decision.arbiter.selected_source,
                "shift_ms": decision.arbiter.shift_ms,
            }
        else:
            raw_assignment.pop("arbiter", None)
        raw_assignment["candidate_provenance"] = {
            "candidate_id": decision.candidate.candidate_id,
            "source": decision.candidate.source,
            "raw_candidate_time": round(decision.candidate.raw_time, 6),
            "canonical_written_centiseconds": decision.written_time.centiseconds,
            "generation_revision": decision.candidate.generation_revision,
            "identity_support": decision.candidate.identity_support,
            "sequence_support": decision.candidate.sequence_support,
            "acoustic_support": decision.candidate.acoustic_support,
            "direct_onset_support": decision.candidate.direct_onset_support,
            "direct_onset_evidence": (
                {
                    "onset_time": decision.candidate.direct_onset_evidence.onset_time,
                    "distance_seconds": decision.candidate.direct_onset_evidence.distance_seconds,
                    "source_revision": decision.candidate.direct_onset_evidence.source_revision,
                    "producer": decision.candidate.direct_onset_evidence.producer,
                    "kind": decision.candidate.direct_onset_evidence.kind,
                    "independent": decision.candidate.direct_onset_evidence.independent,
                }
                if decision.candidate.direct_onset_evidence is not None
                else None
            ),
            "occurrence_evidence": _candidate_occurrence_evidence_projection(
                decision.candidate.occurrence_evidence
            ),
            "prefix_scope": decision.candidate.prefix_scope,
            "right_edge_pileup": decision.candidate.right_edge_pileup,
            "window_truncated": decision.candidate.window_truncated,
            "acoustic_evidence": (
                {
                    "onset_time": decision.candidate.acoustic_evidence.onset_time,
                    "strength": decision.candidate.acoustic_evidence.strength,
                    "distance_seconds": decision.candidate.acoustic_evidence.distance_seconds,
                    "source_revision": decision.candidate.acoustic_evidence.source_revision,
                    "producer": decision.candidate.acoustic_evidence.producer,
                    "kind": decision.candidate.acoustic_evidence.kind,
                    "independent": decision.candidate.acoustic_evidence.independent,
                }
                if decision.candidate.acoustic_evidence is not None
                else None
            ),
        }
        raw_assignment["normalized_timing_evidence"] = _evidence_report_payload(decision.evidence)
        if any(
            item.candidate.candidate_id == decision.candidate.candidate_id
            for item in decision.rejected_candidates
        ):
            raise LrcError("Selected timing candidate also appears in rejected candidates")
        canonical_rejected = [
            {
                "candidate_id": item.candidate.candidate_id,
                "source": item.candidate.source,
                "time": item.candidate.written_time.seconds,
                "score": round(item.candidate.confidence, 6),
                "reasons": list(item.rejection_reasons),
            }
            for item in decision.rejected_candidates
        ]
        displaced_candidate = decision.replaced_candidate
        displaced_payload = (
            [{
                "candidate_id": displaced_candidate.candidate_id,
                "source": displaced_candidate.source,
                "time": displaced_candidate.written_time.seconds,
                "score": round(displaced_candidate.confidence, 6),
                "reasons": ["replaced-by-multi-model-consensus"],
                "replaced_by_consensus": True,
            }]
            if displaced_candidate is not None
            else []
        )
        raw_assignment["candidates"] = [
            {
                "candidate_id": decision.candidate.candidate_id,
                "source": decision.candidate.source,
                "time": decision.written_time.seconds,
                "score": round(decision.candidate.confidence, 6),
                "reasons": ["central-selected" if audit.timing_trusted else "central-provisional"],
            },
            *displaced_payload,
            *canonical_rejected,
        ]
        raw_assignment["rejected_candidates"] = canonical_rejected
        raw_assignment["candidate_rejections"] = [
            {
                "candidate_id": item.candidate.candidate_id,
                "source": item.candidate.source,
                "raw_candidate_time": round(item.candidate.raw_time, 6),
                "canonical_written_centiseconds": item.candidate.written_time.centiseconds,
                "reasons": list(item.rejection_reasons),
                "candidate_provenance": {
                    "generation_revision": item.candidate.generation_revision,
                    "identity_support": item.candidate.identity_support,
                    "sequence_support": item.candidate.sequence_support,
                    "acoustic_support": item.candidate.acoustic_support,
                    "direct_onset_support": item.candidate.direct_onset_support,
                    "direct_onset_evidence": (
                        {
                            "onset_time": item.candidate.direct_onset_evidence.onset_time,
                            "distance_seconds": item.candidate.direct_onset_evidence.distance_seconds,
                            "source_revision": item.candidate.direct_onset_evidence.source_revision,
                        }
                        if item.candidate.direct_onset_evidence is not None
                        else None
                    ),
                    "occurrence_evidence": _candidate_occurrence_evidence_projection(
                        item.candidate.occurrence_evidence
                    ),
                    "prefix_scope": item.candidate.prefix_scope,
                    "right_edge_pileup": item.candidate.right_edge_pileup,
                    "window_truncated": item.candidate.window_truncated,
                    "acoustic_evidence": (
                        {
                            "onset_time": item.candidate.acoustic_evidence.onset_time,
                            "strength": item.candidate.acoustic_evidence.strength,
                            "distance_seconds": item.candidate.acoustic_evidence.distance_seconds,
                            "source_revision": item.candidate.acoustic_evidence.source_revision,
                        }
                        if item.candidate.acoustic_evidence is not None
                        else None
                    ),
                },
                "evidence": _evidence_report_payload(item.evidence),
            }
            for item in decision.rejected_candidates
        ]
        raw_assignment["active_timing_findings"] = active_findings
        raw_assignment["historical_legacy_timing_findings"] = [
            {
                "finding_id": finding.finding_id,
                "code": finding.code,
                "raised_revision": finding.raised_revision,
                "evidence_scope": finding.evidence_scope,
            }
            for finding in entry_findings.historical
        ]
        raw_assignment["resolved_timing_findings"] = resolved_findings
        raw_assignment["flags"] = list(dict.fromkeys([*non_timing_flags, *active_codes]))
        raw_assignment["timing_trusted"] = audit.timing_trusted
        raw_assignment["timing_audit_trusted"] = audit.timing_trusted
        runtime_content_sources = runtime_independent_content_identity_sources_for_entry(
            report, decision.entry_index
        )
        assignment_content_trusted = assignment_content_is_trusted(raw_assignment)
        selected_candidate_content_trusted = candidate_supplies_independent_content_identity(
            decision.candidate, occurrence_entries
        )
        content_trusted = bool(
            assignment_content_trusted
            or selected_candidate_content_trusted
            or runtime_content_sources
        )
        raw_assignment["content_trusted"] = content_trusted
        if assignment_content_trusted:
            raw_assignment["content_trust_source"] = "backend-assignment"
        elif selected_candidate_content_trusted:
            raw_assignment["content_trust_source"] = "selected-candidate-independent-identity"
        elif runtime_content_sources:
            raw_assignment["content_trust_source"] = "runtime-independent-content-identity"
        else:
            raw_assignment["content_trust_source"] = "unverified"
        raw_assignment["content_trust_sources"] = list(runtime_content_sources)
        actionable_failure = any(_timing_failure_requires_review(code) for code in active_codes)
        suspicious_review = _suspicious_entry_requires_review_for_decision(
            report, decision.entry_index + 1, decision, audit
        )
        material_conflict_review = decision.entry_index + 1 in material_conflict_entries
        row_self_check = list(self_check_flags.get(decision.entry_index + 1, []))
        raw_assignment["self_check_flags"] = row_self_check
        review_reason_classes: list[str] = []
        if actionable_failure:
            review_reason_classes.append("actionable-timing-failure")
        if suspicious_review:
            review_reason_classes.append("actionable-suspicious-conflict")
        if material_conflict_review:
            review_reason_classes.append("unresolved-material-alternative")
        if self_check_flags_require_review(row_self_check):
            review_reason_classes.append("final-self-check")
        review_required = bool(review_reason_classes)
        raw_assignment["review_reason_classes"] = review_reason_classes
        for reason_class in review_reason_classes:
            review_reason_entries[reason_class].append(decision.entry_index + 1)
        raw_assignment["review_required"] = review_required
        raw_assignment["overall_trusted"] = bool(
            content_trusted and audit.timing_trusted and not review_required
        )
        verification_status = (
            "review-required"
            if review_required
            else "trusted"
            if raw_assignment["overall_trusted"]
            else "unverified"
        )
        raw_assignment["verification_status"] = verification_status
        if not audit.timing_trusted:
            raw_assignment["unresolved_reason"] = list(audit.failures) or ["no-valid-candidate"]
            unresolved_rows.append(
                {
                    "entry": decision.entry_index + 1,
                    "candidate_id": decision.candidate.candidate_id,
                    "timestamp": decision.written_time.seconds,
                    "flags": active_codes,
                    "verification_status": verification_status,
                    "review_required": review_required,
                }
            )
        else:
            raw_assignment.pop("unresolved_reason", None)
        final_rows.append(
            {
                "entry": decision.entry_index + 1,
                "timestamp": decision.written_time.seconds,
                "canonical_written_centiseconds": decision.written_time.centiseconds,
                "candidate_id": decision.candidate.candidate_id,
                "candidate_source": decision.candidate.source,
                "timestamp_status": audit.status,
                "timing_trusted": audit.timing_trusted,
            }
        )

    report["timing_state_schema_version"] = 3
    report["algorithm_source_sha256"] = validated_loaded_source_sha256()
    report["central_timing_state"] = final_rows
    report["ctc_prefix_probes"] = generation_diagnostics.get("ctc_prefix_probes", [])
    report["ctc_prefix_probe_count"] = generation_diagnostics.get("ctc_prefix_probe_count", 0)
    report["legacy_final_guard_changes_discarded"] = generation_diagnostics.get(
        "legacy_final_guard_changes_discarded", []
    )
    report["legacy_timing_diagnostics"] = {
        "status": "inactive",
        "executed": False,
        "authoritative": False,
        "final_guard_changes_discarded": generation_diagnostics.get(
            "legacy_final_guard_changes_discarded", []
        ),
        "final_guard_change_count": generation_diagnostics.get(
            "legacy_final_guard_change_count", 0
        ),
        "evidence_scope": "historical-nonauthoritative",
    }
    for key in (
        "legacy_boundary_mutation_pipeline",
        "legacy_prefix_probe_candidate_generation",
    ):
        value = generation_diagnostics.get(key)
        if isinstance(value, dict):
            report[key] = copy.deepcopy(value)
    hypothesis_layer = generation_diagnostics.get("alignment_hypothesis_layer")
    if isinstance(hypothesis_layer, dict):
        report["alignment_hypothesis_layer"] = copy.deepcopy(hypothesis_layer)
    retry_diagnostics = generation_diagnostics.get("crossline_retry")
    if isinstance(retry_diagnostics, dict):
        report["crossline_retry"] = copy.deepcopy(retry_diagnostics)
    joint_diagnostics = generation_diagnostics.get("joint_boundary_retry")
    if isinstance(joint_diagnostics, dict):
        report["joint_boundary_retry"] = copy.deepcopy(joint_diagnostics)
    local_whisper_diagnostics = generation_diagnostics.get("local_whisper_evidence")
    if isinstance(local_whisper_diagnostics, dict):
        report["local_whisper_evidence_diagnostics"] = copy.deepcopy(local_whisper_diagnostics)
    arbitration_diagnostics = generation_diagnostics.get("hypothesis_arbitration")
    if isinstance(arbitration_diagnostics, dict):
        report["hypothesis_arbitration"] = copy.deepcopy(arbitration_diagnostics)
    wide_region_diagnostics = generation_diagnostics.get("ctc_wide_region_fallback")
    if isinstance(wide_region_diagnostics, dict):
        report["ctc_wide_region_fallback"] = copy.deepcopy(wide_region_diagnostics)
    tail_diagnostics = generation_diagnostics.get("independent_tail_evidence")
    if isinstance(tail_diagnostics, list):
        report["independent_tail_evidence_diagnostics"] = {
            "authoritative_serialized": False,
            "rows": copy.deepcopy(tail_diagnostics),
            "candidate_count": sum(item is not None for item in tail_diagnostics),
        }
    authority_diagnostics = generation_diagnostics.get("independent_tail_authority")
    if isinstance(authority_diagnostics, list):
        report["independent_tail_authority_diagnostics"] = {
            "contract": "A",
            "rows": copy.deepcopy(authority_diagnostics),
            "authority_serialized": False,
        }
    fusion_counts = generation_diagnostics.get("runtime_evidence_fusion_candidate_counts")
    if isinstance(fusion_counts, dict):
        report["runtime_evidence_fusion_candidate_counts"] = copy.deepcopy(fusion_counts)
    report["ctc_final_timing_guard_count"] = 0
    report["ctc_final_timing_guard_recoveries"] = []
    report["ctc_final_timing_guard_unresolved"] = unresolved_rows
    gap_entries: dict[str, list[int]] = collections.defaultdict(list)
    for row in unresolved_rows:
        if not isinstance(row, dict):
            continue
        entry_number = row.get("entry")
        if not isinstance(entry_number, int):
            continue
        for code in row.get("flags", []):
            if isinstance(code, str):
                gap_entries[code].append(entry_number)
    report["remaining_evidence_gaps"] = {
        "timing_untrusted_count": len(unresolved_rows),
        "by_code": {key: len(value) for key, value in sorted(gap_entries.items())},
        "entries_by_code": {key: value for key, value in sorted(gap_entries.items())},
        "authority": "diagnostic-only",
    }
    report["review_reason_breakdown"] = {
        "schema_version": 1,
        "authority": "diagnostic-only",
        "counts": {
            key: len(set(value))
            for key, value in sorted(review_reason_entries.items())
        },
        "entries": {
            key: sorted(set(value))
            for key, value in sorted(review_reason_entries.items())
        },
        "overlap_allowed": True,
    }
    _project_suspicious_alignments_to_central_state(
        report,
        effective_state,
        material_conflict_entries=material_conflict_entries,
        review_only_entries=_runtime_peer_review_only_conflicts(report),
    )
    return effective_state


def prepare_backend_timing_preview(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
    args: argparse.Namespace,
    anchor_path: Path | None,
    retry_helper_cache: dict[str, tuple[str, object]] | None = None,
) -> tuple[list[float], dict[str, object], EvaluatedBackendTiming, list[dict[str, object]]]:
    """Run all candidate producers once, then return an immutable central preview."""
    if retry_helper_cache is None:
        retry_helper_cache = {}
    preview_report = copy.deepcopy(report)
    preview_timestamps = list(timestamps)
    preview_timestamps, anchor_changes = apply_anchor_hints(
        entries, preview_timestamps, preview_report, anchor_path, duration
    )
    preview_timestamps = score_line_timing_candidates(
        entries, preview_timestamps, preview_report, duration
    )
    preview_timestamps, preview_report, duplicate_changes = apply_short_duplicate_acoustic_recovery(
        audio_path, entries, preview_timestamps, preview_report, duration
    )
    validation_changes = apply_post_score_audio_validation(
        audio_path, entries, preview_timestamps, preview_report, duration
    )
    final_alignment_audio = ctc_alignment_audio_path(preview_report, audio_path)
    # S1: stage untrusted bounded vocal-passage CTC hypotheses. Only Central
    # may validate them. DP/recurrence do not claim final LRC authority.
    segment_structure_diagnostics: dict[str, object] | None = None
    if (preview_report.get("backend") == "ctc"
        and preview_report.get("ctc_audio_source") == "vocal-stem"
        and preview_report.get("segment_skeleton_status", {}).get("status")
           not in {"SEGMENT_CURRENT","FALLBACK_CTC","CTC_SELECTED"}
    ):
        from segment_production import try_stage_structural_ctc
        preview_report, segment_structure_diagnostics = try_stage_structural_ctc(
            report=preview_report,
            lyric_texts=[entry_sung_text(entry) for entry in entries],
            vocal_stem=final_alignment_audio,
            duration=duration,
            audio_api=_AUDIO_API,
            python_exe=default_ctc_python(),
            ctc_helper=Path(__file__).resolve().parent / "ctc_align.py",
            device=str(getattr(args, "whisperx_device", "auto")),
        )
    # Boundary acoustics must be measured on the isolated vocal artifact whenever
    # the CTC backend actually used one; full-mix percussion/transients are not
    # vocal-onset evidence. Producer independence is enforced separately by
    # provenance validation.
    boundary_evidence_audio = final_alignment_audio
    try:
        preview_vocal_onset_features = analyze_vocal_onsets(
            boundary_evidence_audio, duration
        )
    except LrcError:
        preview_vocal_onset_features = None
    onset_confirmation_changes = apply_vocal_onset_confirmation_hypotheses(
        boundary_evidence_audio, entries, preview_timestamps, preview_report, duration,
        features=preview_vocal_onset_features,
    )
    vocal_reentry_changes = apply_vocal_activity_reentry_hypotheses(
        boundary_evidence_audio, entries, preview_timestamps, preview_report, duration
    )
    progress_log(f"central preview collect: backend={preview_report.get('backend', 'unknown')}")
    candidate_sets, diagnostics = collect_central_timing_candidates(
        final_alignment_audio,
        entries,
        preview_timestamps,
        preview_report,
        duration,
        args,
        independent_evidence_audio_path=boundary_evidence_audio,
        retry_helper_cache=retry_helper_cache,
        vocal_onset_features=preview_vocal_onset_features,
    )
    diagnostics = {
        **diagnostics,
        "anchor_changes": anchor_changes,
        "short_duplicate_changes": duplicate_changes,
        "post_score_audio_validation_changes": validation_changes,
        "vocal_onset_confirmation_changes": onset_confirmation_changes,
        "vocal_activity_reentry_changes": vocal_reentry_changes,
        "segment_structure": segment_structure_diagnostics,
    }
    evaluated = evaluate_backend_timing(
        str(preview_report.get("backend") or "unknown"),
        candidate_sets,
        preview_report,
        diagnostics,
        timing_source_policy=str(getattr(args, "timing_source_policy", "default")),
    )
    fixed_point_rounds: list[dict[str, object]] = []
    for round_index in range(1, 5):
        assignments = preview_report.get("assignments")
        if not isinstance(assignments, list):
            break
        before_hypotheses = sum(
            len(item.get("alignment_hypotheses", []))
            for item in assignments
            if isinstance(item, dict) and isinstance(item.get("alignment_hypotheses"), list)
        )
        recurrence_changes = apply_global_acoustic_recurrence_hypotheses(
            boundary_evidence_audio, entries, preview_report, evaluated, duration
        )
        phonetic_changes = apply_phonetic_onset_fusion_hypotheses(
            boundary_evidence_audio, entries, preview_report, evaluated, duration
        )
        mutual_changes = apply_mutual_acoustic_recurrence_hypotheses(
            boundary_evidence_audio, entries, preview_report, evaluated, duration
        )
        tail_release_changes = apply_trusted_tail_release_onset_hypotheses(
            boundary_evidence_audio, entries, preview_report, evaluated, duration
        )
        after_hypotheses = sum(
            len(item.get("alignment_hypotheses", []))
            for item in assignments
            if isinstance(item, dict) and isinstance(item.get("alignment_hypotheses"), list)
        )
        added = max(0, after_hypotheses - before_hypotheses)
        round_row = {
            "round": round_index,
            "hypotheses_before": before_hypotheses,
            "hypotheses_after": after_hypotheses,
            "hypotheses_added": added,
            "global_acoustic_recurrence_changes": copy.deepcopy(recurrence_changes),
            "phonetic_onset_fusion_changes": copy.deepcopy(phonetic_changes),
            "mutual_acoustic_recurrence_changes": copy.deepcopy(mutual_changes),
            "trusted_tail_release_onset_changes": copy.deepcopy(tail_release_changes),
        }
        fixed_point_rounds.append(round_row)
        progress_log(
            f"central acoustic fixed-point round {round_index}: "
            f"new_hypotheses={added}"
        )
        if added <= 0:
            break
        progress_log(
            f"central acoustic fixed-point round {round_index}: "
            f"new_hypotheses={added}; recollecting"
        )
        candidate_sets, diagnostics = collect_central_timing_candidates(
            final_alignment_audio,

            entries,
            preview_timestamps,
            preview_report,
            duration,
            args,
            independent_evidence_audio_path=boundary_evidence_audio,
            retry_helper_cache=retry_helper_cache,
            vocal_onset_features=preview_vocal_onset_features,
        )
        diagnostics = {
            **diagnostics,
            "anchor_changes": anchor_changes,
            "short_duplicate_changes": duplicate_changes,
            "post_score_audio_validation_changes": validation_changes,
            "vocal_onset_confirmation_changes": onset_confirmation_changes,
            "vocal_activity_reentry_changes": vocal_reentry_changes,
            "alignment_fixed_point_rounds": copy.deepcopy(fixed_point_rounds),
            "segment_structure": segment_structure_diagnostics,
        }
        evaluated = evaluate_backend_timing(
            str(preview_report.get("backend") or "unknown"),
            candidate_sets,
            preview_report,
            diagnostics,
            timing_source_policy=str(getattr(args, "timing_source_policy", "default")),
        )
    arbiter_mode = str(getattr(args, "arbiter", "off"))
    if arbiter_mode != "off":
        diagnostics = bind_reviewer_validity_diagnostics(
            diagnostics, args, len(candidate_sets)
        )
        evaluated = evaluate_backend_timing(
            str(preview_report.get("backend") or "unknown"),
            candidate_sets,
            preview_report,
            diagnostics,
            arbiter_mode=arbiter_mode,
            timing_source_policy=str(getattr(args, "timing_source_policy", "default")),
        )
    preview_report["alignment_fixed_point_rounds"] = copy.deepcopy(fixed_point_rounds)
    preview_report["alignment_fixed_point_round_count"] = len(fixed_point_rounds)
    preview_report["alignment_fixed_point_authoritative_precentral_mutation"] = False
    return preview_timestamps, preview_report, evaluated, anchor_changes


def evaluate_backend_timing(
    backend: str,
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    generation_diagnostics: dict[str, object],
    *,
    worklist_order: Literal["forward", "reverse"] = "forward",
    peer_graph: PeerGraphPolicy | None = None,
    arbiter_mode: Literal["off", "gross-rescue", "reviewer-validity"] = "off",
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> EvaluatedBackendTiming:
    if arbiter_mode not in {"off", "gross-rescue", "reviewer-validity"}:
        raise ValueError(f"unsupported timing arbiter mode: {arbiter_mode}")
    if str(report.get("backend") or "") != backend:
        raise LrcError("Backend identity does not match central evaluation report")
    report_projection_revision = timing_report_evaluation_revision(report)
    tail_evidence_context = _tail_evidence_context_from_report(report)
    independent_tails = runtime_independent_tail_evidence(
        report, len(candidate_sets), tail_evidence_context
    )
    expected_tail_payload = independent_tail_evidence_payload(independent_tails)
    diagnostic_tail_payload = generation_diagnostics.get("independent_tail_evidence")
    if any(item is not None for item in independent_tails):
        if diagnostic_tail_payload != expected_tail_payload:
            raise LrcError("Independent tail evidence is not bound to generation diagnostics")
    diagnostics_json, diagnostics_integrity_revision = freeze_generation_diagnostics(
        generation_diagnostics
    )
    diagnostics_semantic_revision = generation_diagnostics_semantic_revision(
        generation_diagnostics
    )
    runtime_entries = report.get(_RUNTIME_LYRIC_ENTRIES_KEY)
    occurrence_entries = (
        tuple(runtime_entries)
        if isinstance(runtime_entries, (list, tuple))
        and len(runtime_entries) == len(candidate_sets)
        and all(isinstance(item, LyricEntry) for item in runtime_entries)
        else None
    )
    reviewer_evidence = reviewer_validity_evidence_from_diagnostics(
        generation_diagnostics, len(candidate_sets)
    )
    input_payload: dict[str, object] = {
        "backend": backend,
        "report_projection_revision": report_projection_revision,
        "generation_diagnostics_semantic_revision": diagnostics_semantic_revision,
        "candidate_sets": [
            [candidate.candidate_id for candidate in candidates]
            for candidates in candidate_sets
        ],
    }
    if peer_graph is not None:
        input_payload["peer_graph_revision"] = peer_graph.revision
    if arbiter_mode != "off":
        input_payload["arbiter_mode"] = arbiter_mode
    input_revision = _timing_content_digest(input_payload)
    base_state = build_final_timing_state(
        candidate_sets,
        worklist_order=worklist_order,
        independent_tail_evidence=independent_tails,
        tail_evidence_context=tail_evidence_context,
        peer_graph=peer_graph,
        occurrence_entries=occurrence_entries,
        arbiter_mode=arbiter_mode,
        reviewer_evidence=reviewer_evidence,
        timing_source_policy=timing_source_policy,
    )
    state, finding_states = evaluate_final_timing_state_for_report(
        base_state,
        report,
        independent_tail_evidence=independent_tails,
        tail_evidence_context=tail_evidence_context,
        peer_graph=peer_graph,
        occurrence_entries=occurrence_entries,
        reviewer_evidence=reviewer_evidence,
    )
    summary = summarize_central_timing_state(report, state)
    quality = central_evaluated_quality(report, summary)
    evaluation_payload: dict[str, object] = {
        "input_revision": input_revision,
        "generation_diagnostics_semantic_revision": diagnostics_semantic_revision,
        "decisions": [
            {
                "candidate_id": decision.candidate.candidate_id,
                "candidate_revision": decision.evidence.candidate_revision,
                "selection_revision": decision.selection_revision,
                "written_centiseconds": decision.written_time.centiseconds,
                "timing_trusted": audit.timing_trusted,
                "failures": audit.failures,
            }
            for decision, audit in zip(state.decisions, state.audits)
        ],
        "summary": {
            "timing_entries": summary.timing_entries,
            "assigned_entries": summary.assigned_entries,
            "content_trusted_entries": summary.content_trusted_entries,
            "timing_trusted_entries": summary.timing_trusted_entries,
            "overall_trusted_entries": summary.overall_trusted_entries,
            "unverified_count": summary.unverified_count,
            "review_required_count": summary.review_required_count,
        },
        "quality": quality,
    }
    if peer_graph is not None:
        evaluation_payload["peer_graph_revision"] = peer_graph.revision
    if arbiter_mode != "off":
        evaluation_payload["arbiter_mode"] = arbiter_mode
    evaluation_revision = _timing_content_digest(evaluation_payload)
    return EvaluatedBackendTiming(
        backend=backend,
        input_revision=input_revision,
        report_projection_revision=report_projection_revision,
        generation_diagnostics_revision=diagnostics_integrity_revision,
        generation_diagnostics_semantic_revision=diagnostics_semantic_revision,
        generation_diagnostics_json=diagnostics_json,
        evaluation_revision=evaluation_revision,
        candidate_sets=candidate_sets,
        state=state,
        finding_states=finding_states,
        summary=summary,
        quality=quality,
        arbiter_mode=arbiter_mode,
        timing_source_policy=timing_source_policy,
    )


def _bundle_candidate_payload(candidate: TimingCandidate) -> dict[str, object]:
    """Return a path-free, raw-transcript-free candidate identity payload."""

    def evidence_payload(value: object) -> dict[str, object] | None:
        if value is None:
            return None
        return {
            "onset_time": round(float(getattr(value, "onset_time", 0.0)), 6),
            "strength": round(float(getattr(value, "strength", 0.0)), 8)
            if hasattr(value, "strength") else None,
            "distance_seconds": round(float(getattr(value, "distance_seconds", 0.0)), 6),
            "source_revision": str(getattr(value, "source_revision", "")),
            "producer": str(getattr(value, "producer", "")),
            "kind": str(getattr(value, "kind", "")),
            "independent": bool(getattr(value, "independent", False)),
        }

    return {
        "candidate_id": candidate.candidate_id,
        "generation_revision": candidate.generation_revision,
        "entry_index": candidate.entry_index,
        "source": candidate.source,
        "raw_time": round(candidate.raw_time, 6),
        "written_centiseconds": candidate.written_time.centiseconds,
        "confidence": round(candidate.confidence, 8),
        "identity_support": candidate.identity_support,
        "sequence_support": candidate.sequence_support,
        "acoustic_support": candidate.acoustic_support,
        "direct_onset_support": candidate.direct_onset_support,
        "acoustic_strength": (
            round(candidate.acoustic_strength, 8)
            if candidate.acoustic_strength is not None else None
        ),
        "independent_content_identity": bool(candidate.independent_content_identity),
        "current": bool(candidate.current),
        "prefix_scope": bool(candidate.prefix_scope),
        "right_edge_pileup": bool(candidate.right_edge_pileup),
        "window_truncated": bool(candidate.window_truncated),
        "spans": [
            [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
            for span in candidate.spans
        ],
        "acoustic_evidence": evidence_payload(candidate.acoustic_evidence),
        "direct_onset_evidence": evidence_payload(candidate.direct_onset_evidence),
        "occurrence_evidence": _candidate_occurrence_evidence_projection(
            candidate.occurrence_evidence
        ),
    }


def _bundle_candidate_sets_payload(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> list[list[dict[str, object]]]:
    return [
        [
            _bundle_candidate_payload(candidate)
            for candidate in sorted(
                candidates,
                key=lambda item: (item.candidate_id, item.generation_revision),
            )
        ]
        for candidates in candidate_sets
    ]


def _bundle_tail_context_payload(
    context: TailEvidenceContext,
) -> dict[str, object]:
    return {
        "backend_lineage": context.backend_lineage,
        "model_identity": context.model_identity,
        "helper_revision": context.helper_revision,
        "capability_revision": context.capability_revision,
        "audio_revision": context.audio_revision,
        "transcript_revisions": list(context.transcript_revisions),
        "row_revisions": list(context.row_revisions),
        "producer_contexts": [list(item) for item in context.producer_contexts],
        "scopes": [
            {
                "entry_index": scope.entry_index,
                "producer": scope.producer,
                "backend_lineage": scope.backend_lineage,
                "model_identity": scope.model_identity,
                "helper_revision": scope.helper_revision,
                "capability_revision": scope.capability_revision,
                "audio_revision": scope.audio_revision,
                "transcript_revisions": list(scope.transcript_revisions),
                "row_revisions": list(scope.row_revisions),
                "capability_revisions": list(scope.capability_revisions),
            }
            for scope in sorted(
                context.scopes,
                key=lambda item: (item.entry_index, item.producer),
            )
        ],
    }


def _bundle_tail_binding_payload(binding: TailObservationBinding) -> dict[str, object]:
    observation = binding.observation
    provenance = observation.provenance
    return {
        "entry_index": observation.entry_index,
        "tail_start": round(observation.tail_start, 6),
        "tail_end": round(observation.tail_end, 6),
        "uncertainty_seconds": round(observation.uncertainty_seconds, 6),
        "confidence": round(observation.confidence, 8),
        "producer": observation.producer,
        "source_revision": observation.source_revision,
        "provenance": _tail_provenance_payload(provenance) if provenance else None,
        "observation_revision": provenance.observation_revision if provenance else observation.source_revision,
        "context_revision": binding.context_revision,
        "context": (
            _bundle_tail_context_payload(binding.context)
            if binding.context is not None else None
        ),
    }


_SCAFFOLD_DROP = object()


def _scaffold_key_is_private(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower()
    if normalized.startswith("_runtime"):
        return True
    return any(
        marker in normalized
        for marker in (
            "path",
            "temp",
            "credential",
            "secret",
            "password",
            "raw_transcript",
            "transcript_text",
            "transcript",
            "asr_segments",
            "raw_asr_segments",
            "reference",
            "anchor_hint",
            "prompt",
        )
    )


def _scaffold_safe_value(value: object, key: object = None) -> object:
    """Deep-copy a report scaffold while removing private/raw inference data."""
    if _scaffold_key_is_private(key):
        return _SCAFFOLD_DROP
    if isinstance(value, dict):
        result: dict[object, object] = {}
        for child_key, child_value in value.items():
            safe = _scaffold_safe_value(child_value, child_key)
            if safe is not _SCAFFOLD_DROP:
                result[child_key] = safe
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for child in value:
            safe = _scaffold_safe_value(child)
            if safe is not _SCAFFOLD_DROP:
                result.append(safe)
        return result
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Runtime dataclasses/paths must never enter the report-only scaffold.
    return _SCAFFOLD_DROP


def _central_scaffold_payload(report: dict[str, object]) -> dict[str, object] | None:
    """Deep-copy the selected report scaffold without private inference data.

    The Central evaluator consumes assignments and suspicious/history fields,
    so the peer preflight must retain those fields instead of replacing them
    with a synthetic empty report.  Runtime sidecars, paths, raw transcripts,
    and reference hints are removed before the copy is serialized.
    """

    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return None
    rows: list[dict[str, int]] = []
    for index, item in enumerate(assignments):
        if not isinstance(item, dict):
            return None
        entry = item.get("entry", index + 1)
        if not isinstance(entry, int) or entry != index + 1:
            return None
        rows.append({"position": index, "entry": entry})
    runtime_entries = report.get(_RUNTIME_LYRIC_ENTRIES_KEY)
    lyric_entry_revisions: list[str] = []
    lyric_identity_revision_value: str | None = None
    if isinstance(runtime_entries, (list, tuple)) and all(
        isinstance(item, LyricEntry) for item in runtime_entries
    ) and len(runtime_entries) == len(rows):
        lyric_entry_revisions = [
            hashlib.sha256(
                normalize_match_text(entry_sung_text(item)).encode("utf-8")
            ).hexdigest()
            for item in runtime_entries
        ]
        lyric_identity_revision_value = lyric_identity_revision(list(runtime_entries))
    try:
        copied_report = copy.deepcopy(report)
        safe_report = _scaffold_safe_value(copied_report)
        report_revision = timing_report_evaluation_revision(report)
        if not isinstance(safe_report, dict):
            return None
        json.dumps(safe_report, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (LrcError, TypeError, ValueError):
        return None
    return {
        "assignment_count": len(rows),
        "entries": rows,
        "lyric_identity_revision": lyric_identity_revision_value,
        "lyric_entry_revisions": lyric_entry_revisions,
        "report_revision": report_revision,
        "report": safe_report,
    }


def _central_scaffold_shape(scaffold: dict[str, object]) -> dict[str, object]:
    """Return the common immutable shape used to compare backend scaffolds."""
    return {
        "assignment_count": scaffold.get("assignment_count"),
        "entries": scaffold.get("entries"),
        "lyric_identity_revision": scaffold.get("lyric_identity_revision"),
        "lyric_entry_revisions": scaffold.get("lyric_entry_revisions"),
    }


def _bundle_content_observations_bound_to_scaffold(
    bundle: EvaluatedBackendBundle,
) -> tuple[tuple[BackendContentIdentityObservation, ...], int]:
    """Keep only content observations bound to the originating lyric rows."""

    try:
        scaffold = json.loads(bundle.central_scaffold_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return (), len(bundle.content_identity_observations)
    revisions = scaffold.get("lyric_entry_revisions") if isinstance(scaffold, dict) else None
    if (
        not isinstance(revisions, list)
        or len(revisions) != len(bundle.candidate_sets)
        or any(not isinstance(item, str) or not item for item in revisions)
    ):
        return (), len(bundle.content_identity_observations)
    valid: list[BackendContentIdentityObservation] = []
    diagnostic = 0
    for observation in bundle.content_identity_observations:
        if (
            0 <= observation.entry_index < len(revisions)
            and observation.backend_lineage
            and observation.lyric_revision == revisions[observation.entry_index]
            and observation.occurrence_revision
            and observation.evidence_revision
        ):
            valid.append(observation)
        else:
            diagnostic += 1
    return tuple(valid), diagnostic


def _bundle_content_identity_observations(
    report: dict[str, object],
) -> tuple[BackendContentIdentityObservation, ...]:
    producer_keys = (
        ("raw-asr-content-identity", _RUNTIME_RAW_CONTENT_IDENTITY_KEY),
        ("whisperx-row-evidence", _RUNTIME_WHISPERX_ROW_EVIDENCE_KEY),
        ("local-whisper-aligner", _RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY),
    )
    observations: list[BackendContentIdentityObservation] = []
    for producer, key in producer_keys:
        runtime = report.get(key)
        if not isinstance(runtime, dict):
            continue
        for raw_index, value in runtime.items():
            if not isinstance(value, dict):
                continue
            try:
                entry_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            evidence_revision = str(value.get("evidence_revision") or "")
            if not evidence_revision:
                continue
            actual_producer = str(value.get("producer") or producer)
            families = _lineage_families_from_labels(actual_producer, source=actual_producer)
            occurrence = value.get("occurrence_binding")
            occurrence_revision = (
                _timing_content_digest(occurrence) if occurrence is not None else ""
            )
            observations.append(
                BackendContentIdentityObservation(
                    entry_index=entry_index,
                    producer=actual_producer,
                    backend_lineage=next(iter(sorted(families)), ""),
                    lyric_revision=str(value.get("lyric_sha256") or ""),
                    occurrence_revision=occurrence_revision,
                    evidence_revision=evidence_revision,
                )
            )
    return tuple(sorted(
        observations,
        key=lambda item: (
            item.entry_index,
            item.producer,
            item.evidence_revision,
            item.backend_lineage,
        ),
    ))


def _bundle_capability_projection(
    audio_path: Path | None,
    args: argparse.Namespace | object,
    evaluated: EvaluatedBackendTiming,
    report: dict[str, object],
) -> tuple[str, str | None, str | None, str]:
    try:
        summary = _public_backend_capability_summary(
            audio_path if audio_path is not None else Path("<unavailable>"),
            args,  # type: ignore[arg-type]
            evaluated.backend,
            report,
        )
    except (AttributeError, LrcError, OSError, TypeError, ValueError):
        summary = {"backend": evaluated.backend, "capability": {}}
    capability = summary.get("capability") if isinstance(summary, dict) else {}
    if not isinstance(capability, dict):
        capability = {}
    safe_json = json.dumps(capability, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def retrieved(name: str) -> str | None:
        value = capability.get(name)
        if not isinstance(value, dict) or value.get("status") not in {"RETRIEVED", "DERIVED"}:
            return None
        item = value.get("value")
        return str(item) if isinstance(item, str) and item else None

    source_audio = retrieved("source_audio_sha256")
    if source_audio is None and audio_path is not None:
        try:
            source_audio = _path_sha256(audio_path)
        except (LrcError, OSError):
            source_audio = None
    alignment_audio = retrieved("alignment_audio_sha256")
    algorithm_source = validated_loaded_source_sha256()
    return safe_json, source_audio, alignment_audio, algorithm_source


def _capture_evaluated_backend_bundle(
    evaluated: EvaluatedBackendTiming,
    report: dict[str, object],
    *,
    audio_path: Path | None = None,
    args: argparse.Namespace | object = None,
) -> EvaluatedBackendBundle:
    candidate_payload = _bundle_candidate_sets_payload(evaluated.candidate_sets)
    candidate_sets_revision = _timing_content_digest(candidate_payload)
    scaffold = _central_scaffold_payload(report)
    scaffold_json = json.dumps(
        scaffold if scaffold is not None else {},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    context = _tail_evidence_context_from_report(report)
    tail_bindings = tuple(
        TailObservationBinding(
            observation=item,
            context=context,
            context_revision=(
                _timing_content_digest(_bundle_tail_context_payload(context))
                if context is not None and _tail_context_is_valid(context)
                else ""
            ),
        )
        for index in sorted(_runtime_tail_observation_map(report))
        for item in _runtime_tail_observation_map(report)[index]
    )
    capability_json, source_audio, alignment_audio, algorithm_source = (
        _bundle_capability_projection(audio_path, args, evaluated, report)
    )
    bundle_payload = {
        "backend": evaluated.backend,
        "candidate_sets": candidate_payload,
        "candidate_sets_revision": candidate_sets_revision,
        "report_projection_revision": evaluated.report_projection_revision,
        "central_scaffold_json": scaffold_json,
        "content_identity_observations": [
            item.__dict__ for item in _bundle_content_identity_observations(report)
        ],
        "tail_bindings": [_bundle_tail_binding_payload(item) for item in tail_bindings],
        "capability_projection_json": capability_json,
        "source_audio_revision": source_audio,
        "alignment_audio_revision": alignment_audio,
        "algorithm_source_revision": algorithm_source,
        "algorithm_revision": LRC_ALGORITHM_REVISION,
        "input_revision": evaluated.input_revision,
        "diagnostics_revision": evaluated.generation_diagnostics_revision,
        "diagnostics_semantic_revision": (
            evaluated.generation_diagnostics_semantic_revision
        ),
        "evaluation_revision": evaluated.evaluation_revision,
    }
    bundle_revision = _timing_content_digest(bundle_payload)
    bundle = EvaluatedBackendBundle(
        evaluated=evaluated,
        candidate_sets=evaluated.candidate_sets,
        candidate_sets_revision=candidate_sets_revision,
        report_projection_revision=evaluated.report_projection_revision,
        central_scaffold_json=scaffold_json,
        content_identity_observations=_bundle_content_identity_observations(report),
        tail_bindings=tail_bindings,
        capability_projection_json=capability_json,
        source_audio_revision=source_audio,
        alignment_audio_revision=alignment_audio,
        algorithm_source_revision=algorithm_source,
        algorithm_revision=LRC_ALGORITHM_REVISION,
        input_revision=evaluated.input_revision,
        diagnostics_revision=evaluated.generation_diagnostics_revision,
        diagnostics_semantic_revision=(
            evaluated.generation_diagnostics_semantic_revision
        ),
        evaluation_revision=evaluated.evaluation_revision,
        bundle_revision=bundle_revision,
    )
    _validate_evaluated_backend_bundle(bundle)
    return bundle


def _validate_evaluated_backend_bundle(
    bundle: EvaluatedBackendBundle,
) -> dict[str, int | bool]:
    if bundle.evaluated.candidate_sets != bundle.candidate_sets:
        raise LrcError("Evaluated backend bundle candidate sets changed")
    if bundle.evaluated.input_revision != bundle.input_revision:
        raise LrcError("Evaluated backend bundle input revision changed")
    if bundle.evaluated.report_projection_revision != bundle.report_projection_revision:
        raise LrcError("Evaluated backend bundle report revision changed")
    if bundle.evaluated.generation_diagnostics_revision != bundle.diagnostics_revision:
        raise LrcError("Evaluated backend bundle diagnostics revision changed")
    if (
        bundle.evaluated.generation_diagnostics_semantic_revision
        != bundle.diagnostics_semantic_revision
    ):
        raise LrcError("Evaluated backend bundle semantic diagnostics revision changed")
    if bundle.evaluated.evaluation_revision != bundle.evaluation_revision:
        raise LrcError("Evaluated backend bundle evaluation revision changed")
    candidate_payload = _bundle_candidate_sets_payload(bundle.candidate_sets)
    if _timing_content_digest(candidate_payload) != bundle.candidate_sets_revision:
        raise LrcError("Evaluated backend bundle candidate digest changed")
    try:
        scaffold = json.loads(bundle.central_scaffold_json)
        if not isinstance(scaffold, dict):
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LrcError("Evaluated backend bundle scaffold is not canonical JSON") from exc
    scaffold_report = scaffold.get("report")
    if not isinstance(scaffold_report, dict):
        raise LrcError("Evaluated backend bundle scaffold report is missing")
    if _scaffold_safe_value(scaffold_report) != scaffold_report:
        raise LrcError("Evaluated backend bundle scaffold contains private data")
    if not isinstance(scaffold.get("report_revision"), str) or not scaffold.get(
        "report_revision"
    ):
        raise LrcError("Evaluated backend bundle scaffold report revision is missing")
    scaffold_assignments = scaffold_report.get("assignments")
    if not isinstance(scaffold_assignments, list) or len(scaffold_assignments) != len(
        bundle.candidate_sets
    ):
        raise LrcError("Evaluated backend bundle scaffold assignments do not match candidates")
    lyric_revisions = scaffold.get("lyric_entry_revisions")
    scaffold_lyric_valid = bool(
        isinstance(scaffold.get("lyric_identity_revision"), str)
        and scaffold.get("lyric_identity_revision")
        and isinstance(lyric_revisions, list)
        and len(lyric_revisions) == len(bundle.candidate_sets)
        and all(isinstance(item, str) and item for item in lyric_revisions)
    )
    if _timing_content_digest(scaffold) != _timing_content_digest(
        json.loads(bundle.central_scaffold_json)
    ):
        raise LrcError("Evaluated backend bundle scaffold digest changed")
    try:
        capability = json.loads(bundle.capability_projection_json)
        if not isinstance(capability, dict):
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LrcError("Evaluated backend bundle capability projection is invalid") from exc
    keys = {str(key).lower() for key in capability}
    if any("path" in key or "credential" in key or "secret" in key for key in keys):
        raise LrcError("Evaluated backend bundle capability contains private path/secret data")
    seen: dict[tuple[str, str], dict[str, object]] = {}
    for candidates in bundle.candidate_sets:
        for candidate in candidates:
            key = (candidate.candidate_id, candidate.generation_revision)
            payload = _bundle_candidate_payload(candidate)
            previous = seen.get(key)
            if previous is not None and previous != payload:
                raise LrcError("Evaluated backend bundle candidate identity collision")
            seen[key] = payload
    authoritative_tails = 0
    diagnostic_tails = 0
    for binding in bundle.tail_bindings:
        observation = binding.observation
        if not _tail_provenance_is_valid(observation):
            diagnostic_tails += 1
            continue
        if binding.context is None or not _tail_context_is_valid(binding.context):
            diagnostic_tails += 1
            continue
        expected_context_revision = _timing_content_digest(
            _bundle_tail_context_payload(binding.context)
        )
        if binding.context_revision != expected_context_revision:
            raise LrcError("Evaluated backend bundle tail context revision changed")
        if not _tail_provenance_matches_context(observation, binding.context):
            diagnostic_tails += 1
            continue
        authoritative_tails += 1
    _bound_content, diagnostic_content = _bundle_content_observations_bound_to_scaffold(bundle)
    bundle_payload = {
        "backend": bundle.evaluated.backend,
        "candidate_sets": candidate_payload,
        "candidate_sets_revision": bundle.candidate_sets_revision,
        "report_projection_revision": bundle.report_projection_revision,
        "central_scaffold_json": bundle.central_scaffold_json,
        "content_identity_observations": [
            item.__dict__ for item in bundle.content_identity_observations
        ],
        "tail_bindings": [_bundle_tail_binding_payload(item) for item in bundle.tail_bindings],
        "capability_projection_json": bundle.capability_projection_json,
        "source_audio_revision": bundle.source_audio_revision,
        "alignment_audio_revision": bundle.alignment_audio_revision,
        "algorithm_source_revision": bundle.algorithm_source_revision,
        "algorithm_revision": bundle.algorithm_revision,
        "input_revision": bundle.input_revision,
        "diagnostics_revision": bundle.diagnostics_revision,
        "diagnostics_semantic_revision": bundle.diagnostics_semantic_revision,
        "evaluation_revision": bundle.evaluation_revision,
    }
    if _timing_content_digest(bundle_payload) != bundle.bundle_revision:
        raise LrcError("Evaluated backend bundle digest changed")
    return {
        "valid": True,
        "scaffold_lyric_valid": scaffold_lyric_valid,
        "authoritative_tail_count": authoritative_tails,
        "diagnostic_tail_count": diagnostic_tails,
        "diagnostic_content_count": int(diagnostic_content),
        "diagnostics_revision": bundle.diagnostics_revision,
        "diagnostics_semantic_revision": bundle.diagnostics_semantic_revision,
    }


def _union_peer_candidate_sets(
    bundles: tuple[EvaluatedBackendBundle, ...],
) -> tuple[tuple[tuple[TimingCandidate, ...], ...] | None, str | None]:
    if not bundles:
        return None, "no-backend-bundles"
    count = len(bundles[0].candidate_sets)
    if any(len(bundle.candidate_sets) != count for bundle in bundles):
        return None, "candidate-entry-count-mismatch"
    merged: list[dict[tuple[str, str], TimingCandidate]] = [
        {} for _ in range(count)
    ]
    for bundle in bundles:
        for entry_index, candidates in enumerate(bundle.candidate_sets):
            for candidate in candidates:
                key = (candidate.candidate_id, candidate.generation_revision)
                existing = merged[entry_index].get(key)
                if existing is not None and _bundle_candidate_payload(existing) != _bundle_candidate_payload(candidate):
                    return None, "candidate-identity-collision"
                merged[entry_index][key] = candidate
    return tuple(
        tuple(
            merged[index][key]
            for key in sorted(merged[index])
        )
        for index in range(count)
    ), None


def _union_peer_content_identity_observations(
    bundles: tuple[EvaluatedBackendBundle, ...],
) -> tuple[tuple[BackendContentIdentityObservation, ...] | None, str | None]:
    merged: dict[
        tuple[int, str, str, str, str],
        BackendContentIdentityObservation,
    ] = {}
    semantic_keys: dict[tuple[int, str, str], tuple[str, str]] = {}
    for bundle in bundles:
        bound_observations, _diagnostic_count = _bundle_content_observations_bound_to_scaffold(
            bundle
        )
        for observation in bound_observations:
            key = (
                observation.entry_index,
                observation.producer,
                observation.evidence_revision,
                observation.backend_lineage,
                observation.occurrence_revision,
            )
            semantic_key = (
                observation.entry_index,
                observation.producer,
                observation.evidence_revision,
            )
            prior_semantic = semantic_keys.get(semantic_key)
            if prior_semantic is not None and prior_semantic != (
                observation.backend_lineage,
                observation.occurrence_revision,
            ):
                return None, "content-identity-revision-collision"
            semantic_keys[semantic_key] = (
                observation.backend_lineage,
                observation.occurrence_revision,
            )
            prior = merged.get(key)
            if prior is not None and prior != observation:
                return None, "content-identity-payload-collision"
            merged[key] = observation
    return tuple(
        merged[key]
        for key in sorted(merged)
    ), None


def _peer_cluster_representative(
    evaluations: tuple[CandidateEvaluation, ...],
) -> str | None:
    """Choose an existing candidate by Central evidence semantics only."""
    valid = tuple(item for item in evaluations if item.valid)
    if not valid:
        return None
    undominated = tuple(
        item
        for item in valid
        if not any(
            other.candidate.candidate_id != item.candidate.candidate_id
            and candidate_dominates(other, item)
            for other in valid
        )
    )
    if len(undominated) == 1:
        return undominated[0].candidate.candidate_id
    equivalent = _semantically_equivalent_valid_winner(undominated)
    if equivalent is not None:
        return equivalent.candidate.candidate_id
    consensus = _bounded_independent_consensus_winner(valid)
    return consensus.candidate.candidate_id if consensus is not None else None


def _peer_cluster_evidence_state(
    evaluations: tuple[CandidateEvaluation, ...],
    selected_families: frozenset[str],
) -> Literal["independent", "same-lineage", "mixed", "unavailable"]:
    states: set[str] = set()
    for evaluation in evaluations:
        candidate = evaluation.candidate
        families = _candidate_support_families(candidate)
        if not families or not selected_families:
            states.add("unavailable")
            continue
        if families & selected_families:
            states.add("same-lineage")
            continue
        if (
            candidate.identity_support == "supported"
            and candidate.sequence_support == "supported"
            and candidate_has_bound_direct_onset_evidence(candidate)
            and evaluation.evidence.temporal_coherence == "coherent"
        ):
            states.add("independent")
        else:
            states.add("unavailable")
    if states == {"independent"}:
        return "independent"
    if states == {"same-lineage"}:
        return "same-lineage"
    if states == {"unavailable"}:
        return "unavailable"
    return "mixed"


def _peer_observation_cluster_summaries(
    decision: TimingDecision,
    selected_time: float,
    *,
    falsified_candidate_ids: frozenset[str] = frozenset(),
) -> tuple[ObservationClusterSummary, ...]:
    """Build bounded, non-authoritative clusters from the existing material owner.

    The anchor remains the earliest member of a cluster; later members never
    extend that anchor.  This is the same non-chaining association used by
    ``_bounded_independent_consensus_winner`` and keeps distant same-lineage
    hypotheses visible instead of treating them as independent votes.
    """
    selected_evaluation = CandidateEvaluation(
        candidate=decision.candidate,
        evidence=decision.evidence,
        valid=decision.status == "selected_valid",
        rejection_reasons=decision.evidence.structural_failures,
    )
    material_details = _material_distant_timing_alternatives(
        decision,
        selected_time,
        falsified_candidate_ids=falsified_candidate_ids,
    )
    material_by_id = {
        str(detail.get("candidate_id")): detail
        for detail in material_details
    }
    evaluations_by_id = {
        evaluation.candidate.candidate_id: evaluation
        for evaluation in decision.rejected_candidates
    }
    members: list[tuple[CandidateEvaluation, dict[str, object] | None, bool]] = [
        (selected_evaluation, None, True)
    ]
    for detail in material_details:
        candidate_id = str(detail.get("candidate_id") or "")
        evaluation = evaluations_by_id.get(candidate_id)
        if evaluation is None:
            raise LrcError("Peer material candidate is missing from Central decision")
        members.append((evaluation, detail, False))
    members.sort(
        key=lambda item: (
            item[0].candidate.written_time.seconds,
            item[0].candidate.candidate_id,
        )
    )
    clusters: list[list[tuple[CandidateEvaluation, dict[str, object] | None, bool]]] = []
    # Written times are canonical centiseconds.  Compare the existing 0.18 s
    # association bound in that integer domain so an exact boundary cannot be
    # split by binary floating-point representation at an exact decimal boundary.
    tolerance_centiseconds = round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100)
    for member in members:
        if (
            not clusters
            or member[0].candidate.written_time.centiseconds
            - clusters[-1][0][0].candidate.written_time.centiseconds
            > tolerance_centiseconds
        ):
            clusters.append([member])
        else:
            clusters[-1].append(member)
    # Peer conflict evidence is review-only and needs the selected producer's
    # explicit lineage even when Central intentionally carries no support vote
    # for that baseline.  Reuse the existing post-Central normalization so an
    # explicit whispercpp baseline is recognized as Whisper-family without
    # changing Central consensus, candidate admission, or selection semantics.
    selected_families = _neutral_arbitration_support_families(decision.candidate)
    summaries: list[ObservationClusterSummary] = []
    for cluster in clusters:
        evaluations = tuple(item[0] for item in cluster)
        contains_selected = any(item[2] for item in cluster)
        details = [item[1] for item in cluster if item[1] is not None]
        explicit_falsification = bool(details) and all(
            bool(detail.get("explicit_falsification")) for detail in details
        )
        relation: Literal["baseline", "distant"] = (
            "baseline" if contains_selected else "distant"
        )
        state = _peer_cluster_evidence_state(evaluations, selected_families)
        member_ids = tuple(item[0].candidate.candidate_id for item in cluster)
        generation_revisions = tuple(
            item[0].candidate.generation_revision for item in cluster
        )
        families = tuple(sorted({
            family
            for item in cluster
            for family in _candidate_support_families(item[0].candidate)
        }))
        cluster_payload = {
            "anchor_time": round(
                cluster[0][0].candidate.written_time.seconds, 6
            ),
            "member_candidate_ids": member_ids,
            "member_generation_revisions": generation_revisions,
            "support_families": families,
            "relation": relation,
            "evidence_state": state,
            "explicit_falsification": explicit_falsification,
            "material_candidate_ids": tuple(
                sorted(str(detail.get("candidate_id")) for detail in details)
            ),
        }
        summaries.append(
            ObservationClusterSummary(
                cluster_revision=_timing_content_digest(cluster_payload),
                anchor_time=cluster[0][0].candidate.written_time.seconds,
                member_candidate_ids=member_ids,
                member_generation_revisions=generation_revisions,
                support_families=families,
                representative_candidate_id=_peer_cluster_representative(evaluations),
                relation=relation,
                evidence_state=state,
                explicit_falsification=explicit_falsification,
                review_required=(
                    relation == "distant" and not explicit_falsification
                ),
            )
        )
    return tuple(summaries)


def _peer_cluster_projection(
    summary: ObservationClusterSummary,
) -> dict[str, object]:
    return {
        "cluster_revision": summary.cluster_revision,
        "anchor_time": round(summary.anchor_time, 6),
        "member_candidate_ids": list(summary.member_candidate_ids),
        "member_generation_revisions": list(summary.member_generation_revisions),
        "support_families": list(summary.support_families),
        "representative_candidate_id": summary.representative_candidate_id,
        "relation": summary.relation,
        "evidence_state": summary.evidence_state,
        "explicit_falsification": summary.explicit_falsification,
        "review_required": summary.review_required,
    }


def _peer_review_only_conflict_projection(
    peer_simulation: dict[str, object],
) -> dict[str, object]:
    """Project validated peer material clusters as review-only metadata."""
    base: dict[str, object] = {
        "authority": "review-only",
        "status": "ABSENT",
        "entries": [],
        "bundle_revisions": [],
        "context_validation": {},
    }
    if peer_simulation.get("status") != "SUCCESS" or peer_simulation.get(
        "authority"
    ) is not False:
        base["revision"] = _timing_content_digest(base)
        return base
    bundles = peer_simulation.get("bundles")
    if not isinstance(bundles, list) or not bundles or any(
        not isinstance(item, dict)
        or item.get("valid") is not True
        or item.get("scaffold_lyric_valid") is not True
        for item in bundles
    ):
        base["revision"] = _timing_content_digest(base)
        return base
    rows = peer_simulation.get("rows")
    if not isinstance(rows, list):
        base["revision"] = _timing_content_digest(base)
        return base
    entries: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("entry"), int):
            continue
        clusters = row.get("clusters")
        if not isinstance(clusters, list):
            continue
        triggered = [
            cluster for cluster in clusters
            if isinstance(cluster, dict)
            and cluster.get("relation") == "distant"
            and cluster.get("review_required") is True
            and cluster.get("explicit_falsification") is not True
        ]
        if not triggered:
            continue
        entries.append({
            "entry": row["entry"],
            "peer_selected_candidate_id": row.get("candidate_id"),
            "peer_selected_centiseconds": row.get("hypothetical_centiseconds"),
            "material_revision": row.get("material_revision"),
            "cluster_revision": row.get("cluster_revision"),
            "cluster_count": row.get("cluster_count"),
            "clusters": copy.deepcopy(triggered),
        })
    bundle_revisions = [
        item.get("bundle_revision")
        for item in bundles
        if isinstance(item.get("bundle_revision"), str)
    ]
    projection: dict[str, object] = {
        "authority": "review-only",
        "status": "RETRIEVED",
        "entries": sorted(entries, key=lambda item: int(item["entry"])),
        "bundle_revisions": sorted(bundle_revisions),
        "context_validation": copy.deepcopy(
            peer_simulation.get("context_validation", {})
        ),
        "source_revision": peer_simulation.get("simulation_revision"),
    }
    projection["revision"] = _timing_content_digest(projection)
    return projection


def _unresolved_evidence_audio_revision(
    report: dict[str, object],
) -> dict[str, object]:
    """Return the path-free runtime audio revision used by candidate evidence."""
    value = report.get("vocal_stem_sha256") or report.get("source_audio_sha256")
    if isinstance(value, str) and value:
        return {"value": value, "status": "RETRIEVED"}
    return {"value": None, "status": "ABSENT"}


def _unresolved_evidence_onset_projection(
    detail: dict[str, object] | None,
    audio_revision: dict[str, object],
) -> dict[str, object] | None:
    """Add runtime audio binding without inventing producer provenance."""
    if not isinstance(detail, dict):
        return None
    projected = copy.deepcopy(detail)
    projected["audio_revision"] = copy.deepcopy(audio_revision)
    return projected


def _unresolved_evidence_missing_categories(
    report: dict[str, object],
    assignment: dict[str, object],
    decision: TimingDecision,
    audit: TimingInvariantAudit,
    alternatives: tuple[dict[str, object], ...],
    suspicious_rows: list[dict[str, object]],
) -> tuple[str, ...]:
    """Map existing Central evidence failures to explicit projection enums.

    This is reporting metadata only.  It never feeds the Central evaluator and
    deliberately leaves the set empty when the serialized report does not prove
    which owner retained review.
    """
    categories: set[str] = set()
    candidate = decision.candidate
    evidence = decision.evidence
    flags = {
        str(item)
        for item in assignment.get("flags", [])
        if isinstance(assignment.get("flags"), list)
    }
    failures = set(audit.failures) | set(evidence.structural_failures) | flags
    if (
        assignment.get("content_trusted") is False
        or candidate.identity_support != "supported"
    ):
        categories.add("CONTENT_IDENTITY")
    if candidate.sequence_support != "supported":
        categories.add("OCCURRENCE")
    if (
        "no-strict-opening-proof" in failures
        or "tail-conflict-without-independent-current-opening" in failures
        or "insufficient-minimum-proof" in failures
        or candidate.direct_onset_support != "supported"
    ):
        categories.add("CURRENT_OPENING")
    if (
        "previous-tail-ownership-unknown" in failures
        or "previous-tail-overlap" in failures
        or evidence.ownership.previous_tail_validity in {"unknown", "invalid", None}
    ):
        categories.add("PREDECESSOR_TAIL")
    if (
        evidence.ownership.status == "unknown"
        or evidence.ownership.previous_tail_authority_backend_lineage is None
        or evidence.ownership.previous_tail_authority_observation_revision is None
    ) and alternatives:
        categories.add("OWNERSHIP_INDEPENDENCE")
    if any(not bool(item.get("explicit_falsification")) for item in alternatives):
        categories.add("ALTERNATIVE_FALSIFICATION")
    runtime_peer_entries = _runtime_peer_review_only_conflicts(report)
    if int(assignment.get("entry", 0) or 0) in runtime_peer_entries:
        categories.add("CROSS_BACKEND_ARBITRATION")
    onset_details = [
        detail.get("direct_onset_evidence")
        for detail in alternatives
        if isinstance(detail, dict)
    ] + [
        detail.get("acoustic_evidence")
        for detail in alternatives
        if isinstance(detail, dict)
    ]
    for onset in onset_details:
        if isinstance(onset, dict) and (
            not onset.get("producer")
            or not onset.get("kind")
            or not onset.get("source_revision")
            or not onset.get("independent")
        ):
            categories.add("PROVENANCE")
            break
    actionable_suspicious = {
        str(flag)
        for row in suspicious_rows
        for flag in row.get("flags", [])
        if isinstance(row, dict)
        and isinstance(row.get("flags"), list)
        and str(flag) in ACTIONABLE_SUSPICIOUS_FLAGS
    }
    if actionable_suspicious:
        categories.add("SUSPICIOUS_CONFLICT")
    return tuple(
        category
        for category in sorted(categories)
        if category in _UNRESOLVED_EVIDENCE_CATEGORY_VALUES
    )


def _same_occurrence_boundary_equivalence_projection(
    state: FinalTimingState,
) -> dict[str, object]:
    """Report distant written candidates proven to be one onset boundary.

    The projection is diagnostic-only.  Its predicate is the exact predicate
    consumed by the canonical materiality owner, so validation can distinguish
    deliberate representation deduplication from unrelated review churn.
    """
    tolerance_centiseconds = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    rows: list[dict[str, object]] = []
    for decision in state.decisions:
        selected = decision.candidate
        for rejected in decision.rejected_candidates:
            alternative = rejected.candidate
            written_delta_centiseconds = abs(
                alternative.written_time.centiseconds - selected.written_time.centiseconds
            )
            if written_delta_centiseconds <= tolerance_centiseconds:
                continue
            if not _candidate_is_same_occurrence_boundary_equivalent(
                selected, alternative
            ):
                continue
            selected_onset = selected.direct_onset_evidence
            alternative_onset = alternative.direct_onset_evidence
            selected_occurrence = selected.occurrence_evidence
            alternative_occurrence = alternative.occurrence_evidence
            if (
                selected_onset is None
                or alternative_onset is None
                or selected_occurrence is None
                or alternative_occurrence is None
            ):
                continue
            selected_onset_centiseconds = canonicalize_lrc_time(
                selected_onset.onset_time
            ).centiseconds
            alternative_onset_centiseconds = canonicalize_lrc_time(
                alternative_onset.onset_time
            ).centiseconds
            rows.append({
                "entry": decision.entry_index + 1,
                "selected_candidate_id": selected.candidate_id,
                "selected_source": selected.source,
                "selected_written_centiseconds": selected.written_time.centiseconds,
                "alternative_candidate_id": alternative.candidate_id,
                "alternative_source": alternative.source,
                "alternative_written_centiseconds": alternative.written_time.centiseconds,
                "written_delta_centiseconds": written_delta_centiseconds,
                "selected_onset_centiseconds": selected_onset_centiseconds,
                "alternative_onset_centiseconds": alternative_onset_centiseconds,
                "onset_delta_centiseconds": abs(
                    selected_onset_centiseconds - alternative_onset_centiseconds
                ),
                "entry_index": selected_occurrence.entry_index,
                "lyric_identity_revision": selected_occurrence.lyric_identity_revision,
                "lyric_sha256": selected_occurrence.lyric_sha256,
                "selected_occurrence_revision": selected_occurrence.evidence_revision,
                "alternative_occurrence_revision": alternative_occurrence.evidence_revision,
            })
    rows.sort(key=lambda item: (
        int(item["entry"]),
        str(item["selected_candidate_id"]),
        str(item["alternative_candidate_id"]),
    ))
    payload: dict[str, object] = {
        "schema_version": 1,
        "authority": False,
        "role": "materiality-dedup-diagnostic-only",
        "materiality_tolerance_centiseconds": tolerance_centiseconds,
        "equivalent_pair_count": len(rows),
        "entries": rows,
        "candidate_graph_mutated": False,
        "timestamp_mutated": False,
        "trust_promoted_directly": False,
    }
    payload["revision"] = _timing_content_digest(payload)
    return payload


def _build_unresolved_evidence_requirements_projection(
    report: dict[str, object],
    state: FinalTimingState,
) -> dict[str, object]:
    """Project every final review/unverified requirement without authority.

    The projection consumes the same final decisions, material owner, and
    observation clusters as Central.  It is intentionally not read by any
    inference or selection path.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(state.decisions):
        raise LrcError("Unresolved evidence projection does not match assignments")
    source_revision = validated_loaded_source_sha256()
    selection = report.get("candidate_selection")
    selection_binding = _timing_content_digest(selection if isinstance(selection, dict) else {})
    central_state_binding = _timing_content_digest(report.get("central_timing_state", []))
    audio_revision = _unresolved_evidence_audio_revision(report)
    rows: list[dict[str, object]] = []
    for assignment, decision, audit in zip(assignments, state.decisions, state.audits):
        if not isinstance(assignment, dict):
            continue
        if not (
            bool(assignment.get("review_required"))
            or assignment.get("verification_status") == "unverified"
        ):
            continue
        selected_evaluation = CandidateEvaluation(
            candidate=decision.candidate,
            evidence=decision.evidence,
            valid=decision.status == "selected_valid",
            rejection_reasons=decision.evidence.structural_failures,
        )
        selected = _candidate_evidence_projection(selected_evaluation)
        selected["direct_onset_evidence"] = _unresolved_evidence_onset_projection(
            selected.get("direct_onset_evidence"), audio_revision
        )
        selected["acoustic_evidence"] = _unresolved_evidence_onset_projection(
            selected.get("acoustic_evidence"), audio_revision
        )
        selected["selected_time"] = decision.written_time.seconds
        selected["selection_revision"] = decision.selection_revision
        selected["timestamp_revision"] = decision.timestamp_revision
        selected["timing_evidence_revision"] = decision.timing_evidence_revision
        selected["status"] = decision.status
        drift_falsified_ids = _runtime_raw_timing_drift_falsified_candidate_ids(
            report, int(assignment.get("entry", 0) or 0)
        )
        alternatives = _material_distant_timing_alternatives(
            decision,
            decision.written_time.seconds,
            falsified_candidate_ids=drift_falsified_ids,
        )
        projected_alternatives: list[dict[str, object]] = []
        for detail in alternatives:
            item = copy.deepcopy(detail)
            item["direct_onset_evidence"] = _unresolved_evidence_onset_projection(
                item.get("direct_onset_evidence"), audio_revision
            )
            item["acoustic_evidence"] = _unresolved_evidence_onset_projection(
                item.get("acoustic_evidence"), audio_revision
            )
            projected_alternatives.append(item)
        clusters = _peer_observation_cluster_summaries(
            decision,
            decision.written_time.seconds,
            falsified_candidate_ids=drift_falsified_ids,
        )
        projected_clusters: list[dict[str, object]] = []
        details_by_id = {
            str(item.get("candidate_id")): item
            for item in projected_alternatives
        }
        for cluster in clusters:
            projection = _peer_cluster_projection(cluster)
            projection["member_evidence"] = [
                copy.deepcopy(details_by_id[candidate_id])
                for candidate_id in cluster.member_candidate_ids
                if candidate_id in details_by_id
            ]
            projected_clusters.append(projection)
        suspicious_rows = [
            row
            for row in report.get("suspicious_alignments", [])
            if isinstance(row, dict)
            and row.get("entry") == assignment.get("entry")
        ]
        flags = [
            str(flag)
            for flag in assignment.get("flags", [])
            if isinstance(assignment.get("flags"), list)
        ]
        source_row_revision = decision.evidence.candidate_revision
        row = {
            "entry": assignment.get("entry"),
            "source_row_revision": source_row_revision,
            "candidate_report_revision": assignment.get("candidate_report_revision"),
            "selected": selected,
            "content_trusted": bool(assignment.get("content_trusted")),
            "timing_trusted": bool(assignment.get("timing_trusted")),
            "overall_trusted": bool(assignment.get("overall_trusted")),
            "review_required": bool(assignment.get("review_required")),
            "verification_status": assignment.get("verification_status"),
            "timing_failures": list(audit.failures),
            "structural_failures": list(decision.evidence.structural_failures),
            "serialized_flags": flags,
            "suspicious_actionable_flags": sorted({
                str(flag)
                for suspicious in suspicious_rows
                for flag in suspicious.get("flags", [])
                if isinstance(suspicious.get("flags"), list)
                and str(flag) in ACTIONABLE_SUSPICIOUS_FLAGS
            }),
            "suspicious_source_row_revisions": [
                item.get("source_row_revision")
                for item in suspicious_rows
                if isinstance(item.get("source_row_revision"), str)
            ],
            "selected_timing_cluster": [
                item
                for item in projected_clusters
                if item.get("relation") == "baseline"
            ],
            "material_alternatives": projected_alternatives,
            "material_clusters": [
                item
                for item in projected_clusters
                if item.get("relation") == "distant"
            ],
            "missing_proof_categories": list(
                _unresolved_evidence_missing_categories(
                    report,
                    assignment,
                    decision,
                    audit,
                    tuple(projected_alternatives),
                    suspicious_rows,
                )
            ),
            "blocking_reason_status": (
                "EXPLICIT"
                if flags or suspicious_rows or projected_alternatives
                else "UNAVAILABLE"
            ),
        }
        rows.append(row)
    revision_binding = {
        "algorithm_source_sha256": source_revision,
        "algorithm_revision": report.get("algorithm_revision", LRC_ALGORITHM_REVISION),
        "lyric_entry_revision": report.get("lyric_entry_revision"),
        "source_audio_sha256": report.get("source_audio_sha256"),
        "vocal_stem_sha256": report.get("vocal_stem_sha256"),
        "selection_revision": selection_binding,
        "central_state_revision": central_state_binding,
    }
    projection: dict[str, object] = {
        "schema_version": _UNRESOLVED_EVIDENCE_REQUIREMENTS_SCHEMA_VERSION,
        "authority": "diagnostic-only",
        "status": "RETRIEVED",
        "revision_binding": revision_binding,
        "rows": sorted(rows, key=lambda item: int(item.get("entry", 0))),
    }
    projection["projection_revision"] = _timing_content_digest(projection)
    return projection


def _strict_candidate_support_families_for_final_context(
    candidate: TimingCandidate,
) -> frozenset[str]:
    """Conservative lineage view used only to veto unsafe final-context removal.

    The production lineage helper predates the explicit ``whispercpp-current``
    source label.  Until that broader lineage repair is independently promoted,
    this stricter view may only *block* a review removal; it never grants
    independence or changes Central candidate generation/selection.
    """
    labels = [
        candidate.source,
        candidate.direct_onset_evidence.producer if candidate.direct_onset_evidence else "",
        candidate.acoustic_evidence.producer if candidate.acoustic_evidence else "",
    ]
    normalized = " ".join(str(value).lower() for value in labels if value)
    families = set(_candidate_support_families(candidate))
    if "whispercpp" in normalized or "whisper.cpp" in normalized:
        families.add("whisper-family")
    return frozenset(families)

def _peer_final_context_tail_runtime(
    bundles: tuple[EvaluatedBackendBundle, ...],
) -> tuple[dict[int, tuple[IndependentTailEvidence, ...]], tuple[TailEvidenceContext, ...]]:
    """Collect only provenance-valid peer tail observations and contexts."""
    valid_contexts: dict[str, TailEvidenceContext] = {}
    tail_rows: dict[int, list[IndependentTailEvidence]] = {}
    for bundle in bundles:
        for binding in bundle.tail_bindings:
            observation = binding.observation
            context = binding.context
            if (
                context is None
                or not _tail_context_is_valid(context)
                or not _tail_provenance_is_valid(observation)
                or not _tail_provenance_matches_context(observation, context)
            ):
                continue
            expected_context_revision = _timing_content_digest(
                _bundle_tail_context_payload(context)
            )
            if binding.context_revision != expected_context_revision:
                continue
            valid_contexts[binding.context_revision] = context
            tail_rows.setdefault(observation.entry_index, []).append(observation)
    return (
        {
            index: tuple(sorted(items, key=lambda item: (
                item.producer,
                item.source_revision,
                item.tail_start,
                item.tail_end,
            )))
            for index, items in tail_rows.items()
        },
        tuple(valid_contexts[key] for key in sorted(valid_contexts)),
    )

def _resolve_final_peer_predecessor_tail(
    previous_candidate: TimingCandidate,
    observations: tuple[IndependentTailEvidence, ...] | None,
    contexts: tuple[TailEvidenceContext, ...],
) -> TailRegionEvidence | None:
    """Resolve a final predecessor tail under the private final-context lineage map.

    Normal Central lineage semantics stay frozen.  This replay-only resolver may
    recognize the already-known ``whispercpp-current`` label as
    ``whisper-family`` so a provenance-valid CTC tail can be consumed as an
    opposite-lineage predecessor bound.  Same-lineage observations remain
    forbidden, and no candidate/timestamp selection is replayed.
    """
    strict_families = _strict_candidate_support_families_for_final_context(
        previous_candidate
    )
    intrinsic = derive_reliable_tail(previous_candidate)
    # The immutable final predecessor candidate owns its intrinsic tail when
    # available. External observations may only fill a missing tail; they
    # must not replace a valid final tail merely because they end later.
    if intrinsic.validity == "valid" and intrinsic.reliable_tail_end is not None:
        return intrinsic
    if not strict_families or not _tail_contexts_are_valid(contexts or None):
        tail = intrinsic
    else:
        selected: list[IndependentTailEvidence] = []
        for observation in observations or ():
            if observation.entry_index != previous_candidate.entry_index:
                continue
            if (
                not _tail_provenance_is_valid(observation)
                or not _tail_provenance_matches_context(observation, contexts or None)
            ):
                continue
            lineage = (
                observation.provenance.backend_lineage
                if observation.provenance is not None
                else None
            )
            tail_family = _tail_backend_family(lineage)
            if tail_family is None or tail_family in strict_families:
                continue
            selected.append(observation)
        if not selected:
            tail = intrinsic
        elif len(selected) == 1:
            tail = _bind_independent_tail_to_candidate(
                previous_candidate, selected[0]
            )
        else:
            # Multiple opposite-lineage external tails are ambiguous when the
            # final predecessor itself has no valid tail. Fail closed instead
            # of choosing whichever observation happens to end latest.
            return None
    if tail.validity != "valid" or tail.reliable_tail_end is None:
        return None
    return tail

def _bundle_candidate_lookup(
    bundles: tuple[EvaluatedBackendBundle, ...],
) -> dict[tuple[str, str], TimingCandidate] | None:
    """Return exact immutable peer candidates or fail on identity collision."""
    lookup: dict[tuple[str, str], TimingCandidate] = {}
    for bundle in bundles:
        for candidates in bundle.candidate_sets:
            for candidate in candidates:
                key = (candidate.candidate_id, candidate.generation_revision)
                existing = lookup.get(key)
                if (
                    existing is not None
                    and _bundle_candidate_payload(existing)
                    != _bundle_candidate_payload(candidate)
                ):
                    return None
                lookup[key] = candidate
    return lookup

def _postcentral_hubert_current_onset_map(
    observations: object,
) -> dict[int, IndependentCurrentOnsetEvidence]:
    """Return one valid runtime HuBERT onset observation per current entry."""
    if not isinstance(observations, tuple):
        return {}
    output: dict[int, IndependentCurrentOnsetEvidence] = {}
    for item in observations:
        if not _known_lyric_hubert_current_onset_is_valid(item):
            return {}
        if item.entry_index in output:
            return {}
        output[item.entry_index] = item
    return output


def _postcentral_hubert_row_hypothesis_set_revision(
    row: object,
    *,
    lookup: dict[tuple[str, str], TimingCandidate],
    final_decision: TimingDecision,
) -> str | None:
    """Bind one HuBERT onset observation to the exact existing peer hypothesis set.

    The current final candidate and every member of every existing distant cluster
    are revision-bound by candidate id, generation revision, and canonical time.
    Any topology change after acquisition therefore invalidates the observation for
    peer falsification instead of silently reusing stale evidence.
    """
    if not isinstance(row, dict):
        return None
    clusters = row.get("clusters")
    if not isinstance(clusters, list) or not clusters:
        return None
    bindings: dict[tuple[str, str], int] = {
        (
            final_decision.candidate.candidate_id,
            final_decision.candidate.generation_revision,
        ): final_decision.written_time.centiseconds
    }
    for cluster in clusters:
        if not isinstance(cluster, dict):
            return None
        ids = cluster.get("member_candidate_ids")
        revisions = cluster.get("member_generation_revisions")
        if (
            not isinstance(ids, list)
            or not isinstance(revisions, list)
            or not ids
            or len(ids) != len(revisions)
        ):
            return None
        for candidate_id, generation_revision in zip(ids, revisions):
            if not isinstance(candidate_id, str) or not isinstance(generation_revision, str):
                return None
            candidate = lookup.get((candidate_id, generation_revision))
            if candidate is None or candidate.entry_index != final_decision.entry_index:
                return None
            bindings[(candidate_id, generation_revision)] = (
                candidate.written_time.centiseconds
            )
    payload = {
        "mode": "existing-peer-hypothesis-set-v1",
        "entry_index": final_decision.entry_index,
        "final_candidate_id": final_decision.candidate.candidate_id,
        "final_generation_revision": final_decision.candidate.generation_revision,
        "final_selected_centiseconds": final_decision.written_time.centiseconds,
        "hypothesis_bindings": [
            [candidate_id, generation_revision, centiseconds]
            for (candidate_id, generation_revision), centiseconds
            in sorted(bindings.items())
        ],
    }
    return _timing_content_digest(payload)


def _postcentral_hubert_observation_matches_row_hypotheses(
    row: object,
    *,
    lookup: dict[tuple[str, str], TimingCandidate],
    final_decision: TimingDecision,
    observation: IndependentCurrentOnsetEvidence | None,
) -> bool:
    if not _known_lyric_hubert_current_onset_is_valid(observation):
        return False
    assert observation is not None
    expected = _postcentral_hubert_row_hypothesis_set_revision(
        row, lookup=lookup, final_decision=final_decision
    )
    return bool(
        expected
        and observation.entry_index == final_decision.entry_index
        and observation.hypothesis_set_revision == expected
    )


def _postcentral_hubert_cluster_is_falsified(
    cluster: object,
    *,
    lookup: dict[tuple[str, str], TimingCandidate],
    final_decision: TimingDecision,
    observation: IndependentCurrentOnsetEvidence | None,
) -> bool:
    """Whether one whole existing distant cluster is contradicted by HuBERT onset.

    The observation must independently corroborate the immutable final selected
    timestamp within the existing 18 cs peer association bound.  Every member of
    the distant cluster must then lie outside the existing 22 cs direct-onset
    association bound.  Partial contradictions never remove a cluster.
    """
    if (
        not _known_lyric_hubert_current_onset_is_valid(observation)
        or observation.entry_index != final_decision.candidate.entry_index
        or observation.identity_owner_candidate_id
        != final_decision.candidate.candidate_id
        or observation.identity_owner_generation_revision
        != final_decision.candidate.generation_revision
        or final_decision.status != "selected_valid"
        or final_decision.selected_candidate_id
        != final_decision.candidate.candidate_id
        or final_decision.candidate.identity_support != "supported"
        or final_decision.candidate.sequence_support != "supported"
        or not isinstance(cluster, dict)
    ):
        return False
    observation_centiseconds = canonicalize_lrc_time(
        observation.onset_time
    ).centiseconds
    peer_tolerance_centiseconds = round(
        RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100
    )
    direct_onset_tolerance_centiseconds = round(
        PREFIX_ACOUSTIC_ASSOCIATION_SECONDS * 100
    )
    if (
        abs(
            observation_centiseconds
            - final_decision.written_time.centiseconds
        )
        > peer_tolerance_centiseconds
    ):
        return False
    ids = cluster.get("member_candidate_ids")
    revisions = cluster.get("member_generation_revisions")
    if (
        not isinstance(ids, list)
        or not isinstance(revisions, list)
        or not ids
        or len(ids) != len(revisions)
    ):
        return False
    for candidate_id, generation_revision in zip(ids, revisions):
        if not isinstance(candidate_id, str) or not isinstance(generation_revision, str):
            return False
        candidate = lookup.get((candidate_id, generation_revision))
        if candidate is None or candidate.entry_index != observation.entry_index:
            return False
        families = _candidate_support_families(candidate)
        # Unknown lineage cannot be contradicted safely, and a future HuBERT-
        # sourced candidate must never be self-falsified by this producer.
        if not families or "hubert-ja-known-lyric" in families:
            return False
        if (
            abs(
                candidate.written_time.centiseconds
                - observation_centiseconds
            )
            <= direct_onset_tolerance_centiseconds
        ):
            return False
    return True


def _final_candidate_valid_under_predecessor_tail(
    final_candidate: TimingCandidate,
    previous_candidate: TimingCandidate,
    previous_tail: TailRegionEvidence,
) -> bool:
    """Only clear final ownership permits review-only peer suppression."""
    if previous_tail.validity != "valid":
        return False
    evaluation = evaluate_timing_candidate(
        final_candidate, previous_candidate, previous_tail=previous_tail
    )
    return evaluation.evidence.ownership.status == "clear"


def _postcentral_recovery_changed_entries(
    projection_seed: dict[str, object],
) -> frozenset[int] | None:
    """Return one-based rows whose final candidate changed after tail capture.

    Tail observations in backend bundles are captured before the HuBERT/peer
    recovery passes. Reusing them after a predecessor row changes candidate
    would bind pre-recovery geometry to a new final occurrence. Malformed
    recovery metadata therefore fails closed by returning None.
    """
    changed: set[int] = set()
    for key in (
        _HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY,
        _PEER_EXISTING_CANDIDATE_RECOVERY_KEY,
    ):
        payload = projection_seed.get(key)
        if payload is None:
            continue
        if not isinstance(payload, dict):
            return None
        entries = payload.get("applied_trigger_entries", [])
        if (
            not isinstance(entries, list)
            or any(type(item) is not int or item <= 0 for item in entries)
        ):
            return None
        changed.update(entries)
    return frozenset(changed)


def _postcentral_rebind_existing_peer_conflicts(
    projection_seed: dict[str, object],
    state: FinalTimingState,
    bundles: tuple[EvaluatedBackendBundle, ...],
    current_onsets: tuple[IndependentCurrentOnsetEvidence, ...] = (),
) -> dict[str, object]:
    """Re-evaluate *only existing* peer conflicts under immutable final context.

    Scope contract:
    - input is first reduced by the already-validated final-distance R2S rebind;
    - no candidate outside those surviving conflict clusters may enter replay;
    - the final selected candidate/timestamp is immutable and never reselected;
    - an already-bound VALID final predecessor tail may replay existing candidate
      ownership, while an occurrence-bound HuBERT current onset may only falsify
      a whole already-existing distant cluster;
    - malformed/ambiguous tail or onset lineage can only retain review;
    - output conflict entries must be a subset of input entries and candidate
      admission is structurally impossible.
    """
    rebound = _rebind_peer_review_only_conflicts_to_final_selection(
        projection_seed, state
    )
    details = rebound.get("cross_backend_material_conflicts")
    if not isinstance(details, dict) or details.get("authority") != "review-only":
        return rebound
    if details.get("status") != "RETRIEVED":
        return rebound
    raw_entries = details.get("entries")
    if not isinstance(raw_entries, list):
        return rebound
    lookup = _bundle_candidate_lookup(bundles)
    if lookup is None:
        return rebound
    tail_rows, tail_contexts = _peer_final_context_tail_runtime(bundles)
    onset_rows = _postcentral_hubert_current_onset_map(current_onsets)
    recovery_changed_entries = _postcentral_recovery_changed_entries(
        projection_seed
    )

    input_entries = {
        int(item["entry"])
        for item in raw_entries
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    if len(input_entries) != len(raw_entries):
        return rebound

    retained_rows: list[dict[str, object]] = []
    semantic_dropped_entries: list[int] = []
    semantic_blocked_entries: list[dict[str, object]] = []
    semantic_dropped_candidate_ids: dict[str, list[str]] = {}
    onset_falsified_entries: set[int] = set()
    onset_falsified_candidate_ids: dict[str, list[str]] = {}
    onset_hypothesis_bound_entries: set[int] = set()

    for raw_row in raw_entries:
        assert isinstance(raw_row, dict)
        entry_number = int(raw_row["entry"])
        index = entry_number - 1
        if not 0 <= index < len(state.decisions):
            return rebound
        final_decision = state.decisions[index]
        onset_observation = onset_rows.get(index)
        previous_candidate = state.decisions[index - 1].candidate if index else None
        previous_tail = (
            _resolve_final_peer_predecessor_tail(
                previous_candidate,
                (
                    tail_rows.get(index - 1)
                    if recovery_changed_entries is not None
                    and index not in recovery_changed_entries
                    else None
                ),
                tail_contexts,
            )
            if index and previous_candidate is not None
            else None
        )
        if (
            previous_candidate is not None
            and previous_tail is not None
            and not _final_candidate_valid_under_predecessor_tail(
                final_decision.candidate, previous_candidate, previous_tail
            )
        ):
            row = copy.deepcopy(raw_row)
            row["postcentral_replay_status"] = "BLOCKED"
            row["postcentral_replay_reason"] = "final-selection-invalid-under-final-predecessor-tail"
            retained_rows.append(row)
            semantic_blocked_entries.append({"entry": entry_number, "reason": "final-selection-invalid-under-final-predecessor-tail"})
            continue
        # Without a valid predecessor tail, preserve the existing row unless an
        # occurrence-bound HuBERT onset independently contradicts *every* member
        # of one or more existing distant clusters.  No candidate is re-evaluated
        # in this onset-only path; it can only delete whole upstream clusters.
        if index and previous_tail is None:
            if onset_observation is not None:
                if _postcentral_hubert_observation_matches_row_hypotheses(
                    raw_row,
                    lookup=lookup,
                    final_decision=final_decision,
                    observation=onset_observation,
                ):
                    onset_hypothesis_bound_entries.add(entry_number)
                else:
                    onset_observation = None
            raw_clusters = raw_row.get("clusters")
            if not isinstance(raw_clusters, list) or not raw_clusters:
                return rebound
            retained_onset_clusters: list[dict[str, object]] = []
            falsified_ids: set[str] = set()
            for cluster in raw_clusters:
                if _postcentral_hubert_cluster_is_falsified(
                    cluster,
                    lookup=lookup,
                    final_decision=final_decision,
                    observation=onset_observation,
                ):
                    ids = cluster.get("member_candidate_ids")
                    if isinstance(ids, list):
                        falsified_ids.update(
                            item for item in ids if isinstance(item, str)
                        )
                    continue
                if not isinstance(cluster, dict):
                    return rebound
                retained_onset_clusters.append(copy.deepcopy(cluster))
            if not falsified_ids:
                row = copy.deepcopy(raw_row)
                row["postcentral_replay_status"] = "BLOCKED"
                row["postcentral_replay_reason"] = "final-valid-predecessor-tail-unavailable"
                retained_rows.append(row)
                semantic_blocked_entries.append({
                    "entry": entry_number,
                    "reason": "final-valid-predecessor-tail-unavailable",
                })
                continue

            onset_falsified_entries.add(entry_number)
            onset_falsified_candidate_ids[str(entry_number)] = sorted(falsified_ids)
            if not retained_onset_clusters:
                semantic_dropped_entries.append(entry_number)
                semantic_dropped_candidate_ids[str(entry_number)] = sorted(falsified_ids)
                continue

            row = copy.deepcopy(raw_row)
            row["postcentral_replay_status"] = "CURRENT_ONSET_REBOUND"
            row["postcentral_replay_reason"] = (
                "postcentral-existing-cluster-hubert-current-onset-falsification"
            )
            row["hubert_current_onset_observation_revision"] = (
                onset_observation.observation_revision
                if onset_observation is not None else None
            )
            row["hubert_current_onset_falsified_candidate_ids"] = sorted(falsified_ids)
            row["clusters"] = retained_onset_clusters
            row["cluster_count"] = 1 + len(retained_onset_clusters)
            row["cluster_revision"] = _timing_content_digest({
                "entry": entry_number,
                "final_selected_candidate_id": final_decision.candidate.candidate_id,
                "final_selected_centiseconds": final_decision.written_time.centiseconds,
                "retained_cluster_revisions": [
                    item.get("cluster_revision") for item in retained_onset_clusters
                ],
                "scope": "postcentral-hubert-current-onset-existing-cluster-only",
            })
            row["material_revision"] = _timing_content_digest({
                "entry": entry_number,
                "cluster_revision": row["cluster_revision"],
                "falsified_candidate_ids": sorted(falsified_ids),
            })
            retained_rows.append(row)
            semantic_dropped_candidate_ids[str(entry_number)] = sorted(falsified_ids)
            continue

        candidate_keys: list[tuple[str, str]] = []
        raw_clusters = raw_row.get("clusters")
        if not isinstance(raw_clusters, list) or not raw_clusters:
            return rebound
        malformed = False
        for cluster in raw_clusters:
            if not isinstance(cluster, dict):
                malformed = True
                break
            ids = cluster.get("member_candidate_ids")
            revisions = cluster.get("member_generation_revisions")
            if (
                not isinstance(ids, list)
                or not isinstance(revisions, list)
                or len(ids) != len(revisions)
                or not ids
            ):
                malformed = True
                break
            for candidate_id, generation_revision in zip(ids, revisions):
                if not isinstance(candidate_id, str) or not isinstance(generation_revision, str):
                    malformed = True
                    break
                key = (candidate_id, generation_revision)
                if key not in candidate_keys:
                    candidate_keys.append(key)
            if malformed:
                break
        if malformed:
            return rebound

        candidates: list[TimingCandidate] = []
        blocked_lineage_ids: list[str] = []
        for key in candidate_keys:
            candidate = lookup.get(key)
            if candidate is None:
                return rebound
            if previous_tail is not None and previous_tail.authority_backend_lineage:
                tail_family = _tail_backend_family(
                    previous_tail.authority_backend_lineage
                )
                production_families = _candidate_support_families(candidate)
                strict_families = _strict_candidate_support_families_for_final_context(
                    candidate
                )
                # The known whispercpp label gap may only retain review.  Never
                # let an optimistic empty-family interpretation remove a row.
                if (
                    tail_family is not None
                    and tail_family in strict_families
                    and tail_family not in production_families
                ):
                    blocked_lineage_ids.append(candidate.candidate_id)
            candidates.append(candidate)
        if blocked_lineage_ids:
            row = copy.deepcopy(raw_row)
            row["postcentral_replay_status"] = "BLOCKED"
            row["postcentral_replay_reason"] = "lineage-normalization-gap"
            row["lineage_blocked_candidate_ids"] = sorted(set(blocked_lineage_ids))
            retained_rows.append(row)
            semantic_blocked_entries.append({
                "entry": entry_number,
                "reason": "lineage-normalization-gap",
                "candidate_ids": sorted(set(blocked_lineage_ids)),
            })
            continue

        evaluations = tuple(
            evaluate_timing_candidate(
                candidate,
                previous_candidate,
                previous_tail=previous_tail,
            )
            for candidate in candidates
        )
        replay_decision = replace(
            final_decision,
            rejected_candidates=evaluations,
        )
        material_details = _material_distant_timing_alternatives(
            replay_decision,
            final_decision.written_time.seconds,
        )
        allowed_ids = {candidate.candidate_id for candidate in candidates}
        material_ids = {
            str(item.get("candidate_id"))
            for item in material_details
            if isinstance(item, dict) and isinstance(item.get("candidate_id"), str)
        }
        if not material_ids.issubset(allowed_ids):
            return copy.deepcopy(projection_seed)

        summaries = _peer_observation_cluster_summaries(
            replay_decision,
            final_decision.written_time.seconds,
        )
        retained_clusters = [
            _peer_cluster_projection(summary)
            for summary in summaries
            if summary.relation == "distant"
            and summary.review_required
            and not summary.explicit_falsification
        ]
        # The current-probe planner binds to this same post-canonical-replay,
        # pre-onset cluster topology.  Validate exact provenance here, not
        # against the broader pre-replay raw row.
        if onset_observation is not None:
            hypothesis_row = copy.deepcopy(raw_row)
            hypothesis_row["clusters"] = copy.deepcopy(retained_clusters)
            if _postcentral_hubert_observation_matches_row_hypotheses(
                hypothesis_row,
                lookup=lookup,
                final_decision=final_decision,
                observation=onset_observation,
            ):
                onset_hypothesis_bound_entries.add(entry_number)
            else:
                onset_observation = None
        onset_filtered_clusters: list[dict[str, object]] = []
        onset_dropped_ids: set[str] = set()
        for cluster in retained_clusters:
            if _postcentral_hubert_cluster_is_falsified(
                cluster,
                lookup=lookup,
                final_decision=final_decision,
                observation=onset_observation,
            ):
                ids = cluster.get("member_candidate_ids")
                if isinstance(ids, list):
                    onset_dropped_ids.update(
                        item for item in ids if isinstance(item, str)
                    )
                continue
            onset_filtered_clusters.append(cluster)
        retained_clusters = onset_filtered_clusters
        if onset_dropped_ids:
            onset_falsified_entries.add(entry_number)
            onset_falsified_candidate_ids[str(entry_number)] = sorted(onset_dropped_ids)
        if not retained_clusters:
            semantic_dropped_entries.append(entry_number)
            semantic_dropped_candidate_ids[str(entry_number)] = sorted(
                allowed_ids | onset_dropped_ids
            )
            continue

        retained_member_ids = {
            str(candidate_id)
            for cluster in retained_clusters
            for candidate_id in cluster.get("member_candidate_ids", [])
            if isinstance(candidate_id, str)
        }
        dropped_ids = sorted(allowed_ids - retained_member_ids)
        row = copy.deepcopy(raw_row)
        row["postcentral_replay_status"] = "REBOUND"
        row["postcentral_replay_reason"] = "postcentral-existing-conflict-only-canonical-replay"
        row["upstream_material_revision"] = raw_row.get("material_revision")
        row["upstream_cluster_revision"] = raw_row.get("cluster_revision")
        row["final_selected_candidate_id"] = final_decision.candidate.candidate_id
        row["final_selected_centiseconds"] = final_decision.written_time.centiseconds
        row["final_predecessor_candidate_id"] = (
            previous_candidate.candidate_id if previous_candidate is not None else None
        )
        row["final_predecessor_tail_evidence_revision"] = (
            previous_tail.tail_evidence_revision if previous_tail is not None else None
        )
        row["replayed_existing_candidate_ids"] = sorted(allowed_ids)
        row["dropped_nonmaterial_or_falsified_candidate_ids"] = dropped_ids
        if onset_dropped_ids:
            row["hubert_current_onset_observation_revision"] = (
                onset_observation.observation_revision
                if onset_observation is not None else None
            )
            row["hubert_current_onset_falsified_candidate_ids"] = sorted(
                onset_dropped_ids
            )
        row["clusters"] = retained_clusters
        row["cluster_count"] = 1 + len(retained_clusters)
        row["cluster_revision"] = _timing_content_digest({
            "entry": entry_number,
            "final_selected_candidate_id": final_decision.candidate.candidate_id,
            "final_selected_centiseconds": final_decision.written_time.centiseconds,
            "retained_cluster_revisions": [
                item.get("cluster_revision") for item in retained_clusters
            ],
            "scope": "postcentral-existing-conflict-only",
        })
        row["material_revision"] = _timing_content_digest({
            "entry": entry_number,
            "cluster_revision": row["cluster_revision"],
            "candidate_ids": sorted(retained_member_ids),
        })
        retained_rows.append(row)
        if dropped_ids:
            semantic_dropped_candidate_ids[str(entry_number)] = dropped_ids

    output_entries = {
        int(item["entry"])
        for item in retained_rows
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    if not output_entries.issubset(input_entries):
        return copy.deepcopy(projection_seed)

    upstream_revision = details.get("revision")
    details["postcentral_existing_conflict_evidence_replayed"] = True
    details["postcentral_existing_conflict_scope"] = "subset-only"
    details["candidate_admission_count"] = 0
    details["added_entries"] = []
    details["semantic_input_entry_count"] = len(input_entries)
    details["semantic_retained_entry_count"] = len(retained_rows)
    details["semantic_dropped_entry_count"] = len(semantic_dropped_entries)
    details["semantic_dropped_entries"] = sorted(semantic_dropped_entries)
    details["semantic_blocked_entries"] = semantic_blocked_entries
    details["semantic_dropped_candidate_ids"] = semantic_dropped_candidate_ids
    details["hubert_current_onset_observation_count"] = len(onset_rows)
    details["hubert_current_onset_falsified_entry_count"] = len(
        onset_falsified_entries
    )
    details["hubert_current_onset_falsified_entries"] = sorted(
        onset_falsified_entries
    )
    details["hubert_current_onset_falsified_candidate_ids"] = (
        onset_falsified_candidate_ids
    )
    details["hubert_current_onset_hypothesis_bound_entries"] = sorted(
        onset_hypothesis_bound_entries
    )
    details["semantic_upstream_revision"] = upstream_revision
    details["final_context_tail_observation_rows"] = len(tail_rows)
    details["final_context_tail_context_count"] = len(tail_contexts)
    details["entries"] = sorted(retained_rows, key=lambda item: int(item["entry"]))
    details["revision"] = _timing_content_digest({
        key: value for key, value in details.items() if key != "revision"
    })
    rebound["cross_backend_material_conflicts"] = details
    return rebound

def _rebind_peer_review_only_conflicts_to_final_selection(
    projection_seed: dict[str, object],
    state: FinalTimingState,
) -> dict[str, object]:
    """Re-evaluate peer review clusters against the final selected timestamps.

    Peer simulation is intentionally non-authoritative and can select a different
    baseline cluster than the backend that is ultimately committed.  A cluster
    that was distant from the simulated peer baseline must not remain a material
    review blocker when it is within the existing 18 cs association bound of the
    immutable final selected timestamp.  This function changes review metadata
    only; it never changes a candidate, timestamp, or Central timing decision.
    """
    rebound = copy.deepcopy(projection_seed)
    details = rebound.get("cross_backend_material_conflicts")
    if not isinstance(details, dict) or details.get("authority") != "review-only":
        return rebound
    revision = details.get("revision")
    payload = dict(details)
    payload.pop("revision", None)
    if not isinstance(revision, str) or _timing_content_digest(payload) != revision:
        return rebound
    if details.get("status") != "RETRIEVED":
        return rebound
    raw_entries = details.get("entries")
    if not isinstance(raw_entries, list) or len(state.decisions) == 0:
        return rebound

    tolerance_centiseconds = round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100)
    retained_entries: list[dict[str, object]] = []
    dropped_entries: list[int] = []
    dropped_cluster_count = 0
    for raw_row in raw_entries:
        # The rebind may remove review authority only from a structurally valid
        # upstream material-conflict row.  Any malformed row stays fail-closed
        # by abandoning the rebind and preserving the original projection.
        if not isinstance(raw_row, dict) or not isinstance(raw_row.get("entry"), int):
            return copy.deepcopy(projection_seed)
        entry_number = int(raw_row["entry"])
        index = entry_number - 1
        if not 0 <= index < len(state.decisions):
            return copy.deepcopy(projection_seed)
        selected = state.decisions[index]
        selected_centiseconds = selected.written_time.centiseconds
        peer_selected_centiseconds = raw_row.get("peer_selected_centiseconds")
        peer_selected_delta_centiseconds: int | None = None
        peer_selected_baseline_conflict = False
        if isinstance(peer_selected_centiseconds, int):
            peer_selected_delta_centiseconds = abs(
                peer_selected_centiseconds - selected_centiseconds
            )
            peer_selected_baseline_conflict = (
                peer_selected_delta_centiseconds > tolerance_centiseconds
            )
        raw_clusters = raw_row.get("clusters")
        if not isinstance(raw_clusters, list) or not raw_clusters:
            return copy.deepcopy(projection_seed)
        retained_clusters: list[dict[str, object]] = []
        suppressed_clusters: list[dict[str, object]] = []
        for raw_cluster in raw_clusters:
            if not isinstance(raw_cluster, dict):
                return copy.deepcopy(projection_seed)
            # Only the exact review-only material topology produced by the peer
            # consumer is eligible for suppression.  Unknown/new cluster forms
            # remain review blockers rather than being interpreted loosely.
            if (
                raw_cluster.get("relation") != "distant"
                or raw_cluster.get("review_required") is not True
                or raw_cluster.get("explicit_falsification") is True
            ):
                retained_clusters.append(copy.deepcopy(raw_cluster))
                continue
            try:
                anchor_centiseconds = canonicalize_lrc_time(
                    float(raw_cluster.get("anchor_time"))
                ).centiseconds
            except (TypeError, ValueError):
                retained_clusters.append(copy.deepcopy(raw_cluster))
                continue
            delta_centiseconds = abs(anchor_centiseconds - selected_centiseconds)
            projected = copy.deepcopy(raw_cluster)
            projected["final_selected_delta_centiseconds"] = delta_centiseconds
            projected["final_relation"] = (
                "distant" if delta_centiseconds > tolerance_centiseconds else "baseline"
            )
            if (
                delta_centiseconds > tolerance_centiseconds
                or peer_selected_baseline_conflict
            ):
                if peer_selected_baseline_conflict:
                    projected["peer_selected_baseline_conflict"] = True
                    projected["peer_selected_delta_centiseconds"] = (
                        peer_selected_delta_centiseconds
                    )
                retained_clusters.append(projected)
            else:
                suppressed_clusters.append(projected)

        dropped_cluster_count += len(suppressed_clusters)
        if not retained_clusters:
            dropped_entries.append(entry_number)
            continue
        row = copy.deepcopy(raw_row)
        row["upstream_material_revision"] = raw_row.get("material_revision")
        row["upstream_cluster_revision"] = raw_row.get("cluster_revision")
        row["final_selected_candidate_id"] = selected.candidate.candidate_id
        row["final_selected_centiseconds"] = selected_centiseconds
        row["peer_selected_delta_centiseconds"] = peer_selected_delta_centiseconds
        row["peer_selected_baseline_conflict"] = peer_selected_baseline_conflict
        row["suppressed_nonmaterial_clusters"] = suppressed_clusters
        row["clusters"] = retained_clusters
        row["cluster_count"] = 1 + len(retained_clusters)
        row["cluster_revision"] = _timing_content_digest({
            "entry": entry_number,
            "final_selected_candidate_id": selected.candidate.candidate_id,
            "final_selected_centiseconds": selected_centiseconds,
            "retained_cluster_revisions": [
                item.get("cluster_revision") for item in retained_clusters
            ],
        })
        row["material_revision"] = _timing_content_digest({
            "entry": entry_number,
            "cluster_revision": row["cluster_revision"],
            "tolerance_centiseconds": tolerance_centiseconds,
        })
        retained_entries.append(row)

    details["upstream_revision"] = revision
    details["final_selection_materiality_rebound"] = True
    details["materiality_tolerance_centiseconds"] = tolerance_centiseconds
    details["dropped_entry_count"] = len(dropped_entries)
    details["dropped_entries"] = sorted(dropped_entries)
    details["dropped_cluster_count"] = dropped_cluster_count
    details["entries"] = sorted(
        retained_entries, key=lambda item: int(item.get("entry", 0))
    )
    details["revision"] = _timing_content_digest({
        key: value for key, value in details.items() if key != "revision"
    })
    rebound["cross_backend_material_conflicts"] = details
    return rebound


def _install_peer_review_only_conflicts(
    report: dict[str, object],
    projection_seed: dict[str, object] | None,
) -> frozenset[int]:
    """Install only validated peer review metadata into the private runtime."""
    report.pop(_RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY, None)
    report.pop(_RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICT_DETAILS_KEY, None)
    if not isinstance(projection_seed, dict):
        return frozenset()
    details = projection_seed.get("cross_backend_material_conflicts")
    if not isinstance(details, dict) or details.get("authority") != "review-only":
        return frozenset()
    revision = details.get("revision")
    payload = dict(details)
    payload.pop("revision", None)
    if not isinstance(revision, str) or _timing_content_digest(payload) != revision:
        return frozenset()
    if details.get("status") != "RETRIEVED":
        return frozenset()
    raw_entries = details.get("entries")
    if not isinstance(raw_entries, list):
        return frozenset()
    entries = frozenset(
        int(item["entry"])
        for item in raw_entries
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    )
    report[_RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY] = entries
    report[_RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICT_DETAILS_KEY] = copy.deepcopy(
        details
    )
    return entries


def _peer_policy() -> PeerGraphPolicy:
    mode = "cross_backend_neutral"
    return PeerGraphPolicy(
        mode=mode,
        revision=_timing_content_digest({
            "schema_version": _BACKEND_PEER_SIMULATION_SCHEMA_VERSION,
            "mode": mode,
        }),
    )


def _simulate_cross_backend_peer_graph(
    bundles: tuple[EvaluatedBackendBundle, ...],
    *,
    selected_backend: str | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> dict[str, object]:
    """Run one in-memory Central peer simulation, never commit its state.

    The selected backend's actual report is used as a read-only Central
    scaffold.  Nonselected reports contribute only candidate/evidence bundles;
    their diagnostics cannot become a second source of review authority.
    """

    validation: list[dict[str, object]] = []
    for bundle in bundles:
        result = _validate_evaluated_backend_bundle(bundle)
        validation.append({
            "backend": bundle.evaluated.backend,
            "bundle_revision": bundle.bundle_revision,
            **result,
        })
    candidate_sets, union_error = _union_peer_candidate_sets(bundles)
    content_observations, content_union_error = _union_peer_content_identity_observations(
        bundles
    )
    row_count = len(candidate_sets) if candidate_sets is not None else 0
    reasons: list[str] = []
    if len(bundles) < 2:
        reasons.append("insufficient-backend-peer-set")
    if any(not bool(item.get("scaffold_lyric_valid")) for item in validation):
        reasons.append("missing-or-mismatched-lyric-scaffold")
    parsed_scaffolds: dict[str, dict[str, object]] = {}
    for bundle in bundles:
        try:
            parsed = json.loads(bundle.central_scaffold_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            parsed_scaffolds[bundle.evaluated.backend] = parsed
    selected_bundle = next(
        (
            bundle for bundle in bundles
            if selected_backend is not None
            and bundle.evaluated.backend == selected_backend
        ),
        None,
    )
    if selected_bundle is None and bundles:
        selected_bundle = bundles[0]
    selected_scaffold = (
        parsed_scaffolds.get(selected_bundle.evaluated.backend)
        if selected_bundle is not None
        else None
    )
    scaffold_shape = (
        _central_scaffold_shape(selected_scaffold)
        if selected_scaffold is not None
        else None
    )
    if selected_scaffold is None:
        reasons.append("selected-central-scaffold-missing")
    elif any(
        _central_scaffold_shape(scaffold) != scaffold_shape
        for scaffold in parsed_scaffolds.values()
    ):
        reasons.append("common-central-scaffold-shape-mismatch")
    if union_error is not None:
        reasons.append(union_error)
    if content_union_error is not None:
        reasons.append(content_union_error)
    current_conflicts: list[int] = []
    if candidate_sets is not None:
        for index, candidates in enumerate(candidate_sets):
            current_keys = {
                (candidate.candidate_id, candidate.generation_revision)
                for candidate in candidates
                if candidate.current
            }
            if len(current_keys) > 1:
                current_conflicts.append(index + 1)
    valid_contexts: dict[str, TailEvidenceContext] = {}
    tail_rows: dict[int, list[IndependentTailEvidence]] = {}
    diagnostic_tail_rows: list[int] = []
    if bundles:
        for bundle in bundles:
            for binding in bundle.tail_bindings:
                observation = binding.observation
                if (
                    binding.context is None
                    or not _tail_context_is_valid(binding.context)
                    or not _tail_provenance_is_valid(observation)
                    or not _tail_provenance_matches_context(observation, binding.context)
                ):
                    diagnostic_tail_rows.append(observation.entry_index + 1)
                    continue
                context_revision = binding.context_revision
                valid_contexts[context_revision] = binding.context
                tail_rows.setdefault(observation.entry_index, []).append(observation)
    if reasons or candidate_sets is None:
        row_summary = [
            {
                "entry": index + 1,
                "status": "BLOCKED",
                "authority": False,
                "reason": ",".join(reasons) or "bundle-validation-failed",
            }
            for index in range(row_count)
        ]
        payload = {
            "schema_version": _BACKEND_PEER_SIMULATION_SCHEMA_VERSION,
            "status": "BLOCKED",
            "authority": False,
            "peer_policy": _peer_policy().revision,
            "blocked_reasons": sorted(set(reasons)) or ["bundle-validation-failed"],
            "row_count": row_count,
            "current_conflict_entries": current_conflicts,
            "context_validation": {
                "valid_context_count": len(valid_contexts),
                "diagnostic_tail_row_count": len(set(diagnostic_tail_rows)),
            },
            "content_identity_observation_count": len(content_observations or ()),
            "bundles": validation,
            "rows": row_summary,
            "material_conflicts": [],
        }
        return {**payload, "simulation_revision": _timing_content_digest(payload)}

    if selected_scaffold is None or not isinstance(
        selected_scaffold.get("report"), dict
    ):
        reasons.append("selected-central-scaffold-report-missing")
        row_count = len(candidate_sets) if candidate_sets is not None else 0
        payload = {
            "schema_version": _BACKEND_PEER_SIMULATION_SCHEMA_VERSION,
            "status": "BLOCKED",
            "authority": False,
            "peer_policy": _peer_policy().revision,
            "blocked_reasons": sorted(set(reasons)),
            "row_count": row_count,
            "current_conflict_entries": current_conflicts,
            "context_validation": {
                "valid_context_count": len(valid_contexts),
                "diagnostic_tail_row_count": len(set(diagnostic_tail_rows)),
            },
            "content_identity_observation_count": len(content_observations or ()),
            "bundles": validation,
            "rows": [
                {
                    "entry": index + 1,
                    "status": "BLOCKED",
                    "authority": False,
                    "reason": "selected-central-scaffold-report-missing",
                }
                for index in range(row_count)
            ],
            "material_conflicts": [],
        }
        return {**payload, "simulation_revision": _timing_content_digest(payload)}
    scaffold = selected_scaffold
    entries = scaffold.get("entries", [])
    synthetic_report = copy.deepcopy(scaffold["report"])
    if not isinstance(synthetic_report, dict):
        raise LrcError("Selected Central scaffold report is not an object")
    synthetic_report["peer_preflight"] = True
    independent_tails: RuntimeTailEvidence = tuple(
        tuple(tail_rows.get(index, ())) or None
        for index in range(row_count)
    )
    context_input: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = (
        tuple(valid_contexts.values()) if valid_contexts else None
    )
    policy = _peer_policy()
    try:
        state = build_final_timing_state(
            candidate_sets,
            independent_tail_evidence=independent_tails,
            tail_evidence_context=context_input,
            peer_graph=policy,
            timing_source_policy=timing_source_policy,
        )
        state, _finding_states = evaluate_final_timing_state_for_report(
            state,
            synthetic_report,
            independent_tail_evidence=independent_tails,
            tail_evidence_context=context_input,
            peer_graph=policy,
        )
        summary = summarize_central_timing_state(
            synthetic_report,
            state,
            content_identity_observations=content_observations or (),
        )
        rows: list[dict[str, object]] = []
        material_by_entry: dict[int, tuple[dict[str, object], ...]] = {}
        content_entries = {
            item.entry_index for item in (content_observations or ())
        }
        (
            content_flags,
            timing_flags,
            review_flags,
            overall_flags,
            unverified_flags,
            _material_entries,
        ) = _central_summary_row_flags(
            synthetic_report,
            state,
            content_observations or (),
        )
        for index, decision in enumerate(state.decisions):
            selected_evaluation = CandidateEvaluation(
                candidate=decision.candidate,
                evidence=decision.evidence,
                valid=decision.status == "selected_valid",
                rejection_reasons=decision.evidence.structural_failures,
            )
            selected_detail = _candidate_evidence_projection(selected_evaluation)
            selected_detail.update({
                "status": decision.status,
                "authority": False,
            })
            material_details = list(
                _material_distant_timing_alternatives(
                    decision, decision.written_time.seconds
                )
            )
            selected_families = set(
                _neutral_arbitration_support_families(decision.candidate)
            )
            for detail in material_details:
                alternative_families = set(detail.get("support_families", ()))
                detail["same_lineage"] = bool(selected_families & alternative_families)
                detail["candidate_lineage_relation"] = (
                    "overlap" if selected_families & alternative_families
                    else "disjoint" if alternative_families and selected_families
                    else "unknown"
                )
                detail["semantically_equivalent_centiseconds"] = (
                    detail.get("canonical_written_centiseconds")
                    == selected_detail.get("canonical_written_centiseconds")
                )
            if material_details:
                material_by_entry[decision.entry_index + 1] = tuple(material_details)
            cluster_summaries = _peer_observation_cluster_summaries(
                decision,
                decision.written_time.seconds,
            )
            cluster_projections = [
                _peer_cluster_projection(item) for item in cluster_summaries
            ]
            cluster_revision = _timing_content_digest(cluster_projections)
            material_revision = _timing_content_digest({
                "selected_candidate_id": decision.candidate.candidate_id,
                "alternatives": material_details,
            })
            rows.append({
                "entry": decision.entry_index + 1,
                "status": decision.status,
                "authority": False,
                "candidate_id": decision.candidate.candidate_id,
                "candidate_source": decision.candidate.source,
                "hypothetical_centiseconds": decision.written_time.centiseconds,
                "timing_trusted": bool(state.audits[index].timing_trusted),
                "content_identity_observed": decision.entry_index in content_entries,
                "current_baseline_conflict": decision.entry_index + 1 in current_conflicts,
                "content_trusted": bool(content_flags[index]),
                "review_required": bool(review_flags[index]),
                "overall_trusted": bool(overall_flags[index]),
                "unverified": bool(unverified_flags[index]),
                "selected_candidate": (
                    selected_detail if decision.status == "selected_valid" else None
                ),
                "provisional_candidate": (
                    selected_detail if decision.status == "provisional_unresolved" else None
                ),
                "material_count": len(material_details),
                "material_revision": material_revision,
                "material_alternatives": material_details,
                "cluster_count": len(cluster_summaries),
                "cluster_revision": cluster_revision,
                "clusters": cluster_projections,
            })
        material = sorted(material_by_entry)
        def summary_payload(value: CentralTrustSummary) -> dict[str, object]:
            return {
                "timing_entries": value.timing_entries,
                "assigned_entries": value.assigned_entries,
                "content_trusted_entries": value.content_trusted_entries,
                "timing_trusted_entries": value.timing_trusted_entries,
                "overall_trusted_entries": value.overall_trusted_entries,
                "unverified_count": value.unverified_count,
                "review_required_count": value.review_required_count,
                "assigned_percent": value.assigned_percent,
                "content_trusted_percent": value.content_trusted_percent,
                "timing_trusted_percent": value.timing_trusted_percent,
                "overall_trusted_percent": value.overall_trusted_percent,
                "unverified_percent": value.unverified_percent,
                "review_required_percent": value.review_required_percent,
                "review_required": value.review_required,
            }
        hypothetical_summary = summary_payload(summary)
        actual_summary = (
            selected_bundle.evaluated.summary
            if selected_bundle is not None
            else summary
        )
        actual_summary_payload = summary_payload(actual_summary)
        summary_delta = {
            key: hypothetical_summary[key] - actual_summary_payload[key]
            for key in hypothetical_summary
            if key.endswith("_entries")
            or key.endswith("_count")
            or key.endswith("_percent")
        }
        scaffold_binding = {
            "selected_backend": (
                selected_bundle.evaluated.backend
                if selected_bundle is not None
                else None
            ),
            "selected_report_revision": scaffold.get("report_revision"),
            "scaffold_shape_revision": _timing_content_digest(
                _central_scaffold_shape(scaffold)
            ),
        }
        payload = {
            "schema_version": _BACKEND_PEER_SIMULATION_SCHEMA_VERSION,
            "status": "SUCCESS",
            "authority": False,
            "peer_policy": policy.revision,
            "row_count": row_count,
            "current_conflict_entries": current_conflicts,
            "context_validation": {
                "valid_context_count": len(valid_contexts),
                "diagnostic_tail_row_count": len(set(diagnostic_tail_rows)),
            },
            "content_identity_observation_count": len(content_observations or ()),
            "bundles": validation,
            "summary": {
                **hypothetical_summary,
            },
            "actual_selected_summary": actual_summary_payload,
            "summary_delta_vs_selected": summary_delta,
            "scaffold_binding": scaffold_binding,
            "rows": rows,
            "material_conflicts": material,
        }
    except (LrcError, TypeError, ValueError) as exc:
        payload = {
            "schema_version": _BACKEND_PEER_SIMULATION_SCHEMA_VERSION,
            "status": "BLOCKED",
            "authority": False,
            "peer_policy": policy.revision,
            "blocked_reasons": [f"central-simulation-error:{type(exc).__name__}"],
            "row_count": row_count,
            "current_conflict_entries": current_conflicts,
            "context_validation": {
                "valid_context_count": len(valid_contexts),
                "diagnostic_tail_row_count": len(set(diagnostic_tail_rows)),
            },
            "content_identity_observation_count": len(content_observations or ()),
            "bundles": validation,
            "rows": [
                {
                    "entry": index + 1,
                    "status": "BLOCKED",
                    "authority": False,
                    "reason": "central-simulation-error",
                }
                for index in range(row_count)
            ],
            "material_conflicts": [],
        }
    return {**payload, "simulation_revision": _timing_content_digest(payload)}


def select_evaluated_backend(
    candidates: tuple[EvaluatedBackendTiming, ...],
) -> BackendSelection:
    if not candidates:
        raise LrcError("No centrally evaluated backend candidate is available")
    ordered = sorted(
        candidates,
        key=lambda item: (-item.quality, item.backend, item.evaluation_revision),
    )
    selected = ordered[0]
    selection_revision = _timing_content_digest({
        "selected_evaluation_revision": selected.evaluation_revision,
        "candidates": [item.evaluation_revision for item in ordered],
    })
    return BackendSelection(selected, tuple(ordered), selection_revision)


def central_backend_summary_payload(
    item: EvaluatedBackendTiming,
) -> dict[str, object]:
    return {
        "backend": item.backend,
        "metric_stage": "central-evaluated",
        "evaluation_revision": item.evaluation_revision,
        "input_revision": item.input_revision,
        "generation_diagnostics_revision": item.generation_diagnostics_revision,
        "generation_diagnostics_semantic_revision": (
            item.generation_diagnostics_semantic_revision
        ),
        "quality": round(item.quality, 3),
        "content_trusted_percent": item.summary.content_trusted_percent,
        "timing_trusted_percent": item.summary.timing_trusted_percent,
        "overall_trusted_percent": item.summary.overall_trusted_percent,
        "authority_trusted_percent": item.summary.overall_trusted_percent,
        "trusted_percent": item.summary.overall_trusted_percent,
        "unverified_percent": item.summary.unverified_percent,
        "review_required_percent": item.summary.review_required_percent,
    }


def central_selection_reason(selection: BackendSelection) -> str:
    selected = selection.selected
    alternatives = ", ".join(
        f"{item.backend}={item.quality:.1f}"
        for item in selection.candidates
        if item is not selected
    ) or "none"
    return (
        f"selected {selected.backend} from immutable central evaluation "
        f"{selected.evaluation_revision[:12]}: quality={selected.quality:.1f}, "
        f"authority_trusted={selected.summary.overall_trusted_percent:.2f}%, "
        f"timing_trusted={selected.summary.timing_trusted_percent:.2f}%, "
        f"unverified={selected.summary.unverified_count}, "
        f"review_required={selected.summary.review_required_count}; "
        f"alternatives: {alternatives}"
    )


def central_candidate_selection_payload(
    selection: BackendSelection,
) -> dict[str, object]:
    selected_payload = central_backend_summary_payload(selection.selected)
    alternatives = [
        central_backend_summary_payload(item)
        for item in selection.candidates
        if item is not selection.selected
    ]
    return {
        "status": "selected",
        "metric_stage": "central-evaluated",
        "selection_revision": selection.selection_revision,
        "selected_backend": selection.selected.backend,
        "selected_quality": round(selection.selected.quality, 3),
        "selected_reason": central_selection_reason(selection),
        "selected_evaluation_revision": selection.selected.evaluation_revision,
        "selected_input_revision": selection.selected.input_revision,
        "selected_candidate": selected_payload,
        "alternatives": alternatives,
        "candidates": [
            central_backend_summary_payload(item) for item in selection.candidates
        ],
    }


def _backend_projection_field(
    value: object,
    *,
    missing_status: Literal["ABSENT", "UNAVAILABLE"] = "UNAVAILABLE",
) -> dict[str, object]:
    """Return a public value/status pair without manufacturing provenance."""
    if value is None or value == "":
        return {"value": None, "status": missing_status}
    return {"value": value, "status": "RETRIEVED"}


def _backend_projection_derived_field(value: object) -> dict[str, object]:
    if value is None or value == "":
        return {"value": None, "status": "UNAVAILABLE"}
    return {"value": value, "status": "DERIVED"}


def _backend_projection_attr(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _backend_projection_error_code(error: object) -> str | None:
    if not error:
        return None
    text = str(error).lower()
    if "no usable whisperx candidate" in text:
        return "whisperx-unusable"
    if "alignment is untrusted" in text:
        return "alignment-untrusted"
    if "failed" in text:
        return "backend-failed"
    return "candidate-error"


def _backend_lineage_for_projection(backend: object) -> str | None:
    value = str(backend or "").strip().lower()
    if value in {"ctc", "mms-ctc"}:
        return "ctc"
    if value in {"whispercpp", "whispercpp_raw", "whisperx", "raw-asr"}:
        return "whisper-family"
    if value.startswith("local-whisper"):
        return "whisper-family"
    return None


def _public_backend_capability_summary(
    audio_path: Path,
    args: argparse.Namespace,
    backend: str,
    report: dict[str, object] | None,
) -> dict[str, object]:
    """Project retrieved backend capability without paths or runtime objects."""
    report = report if isinstance(report, dict) else {}
    try:
        source_audio_sha256 = _path_sha256(audio_path)
    except (OSError, LrcError):
        source_audio_sha256 = None

    lineage = _backend_lineage_for_projection(backend or report.get("backend"))
    fields: dict[str, dict[str, object]] = {
        "backend_lineage": _backend_projection_field(lineage),
        "source_audio_sha256": _backend_projection_field(source_audio_sha256),
        "audio_revision": _backend_projection_field(source_audio_sha256),
        "alignment_audio_sha256": _backend_projection_field(None),
        "alignment_audio_source": _backend_projection_field(None),
        "executable_sha256": _backend_projection_field(None, missing_status="ABSENT"),
        "model_sha256": _backend_projection_field(None, missing_status="ABSENT"),
        "python_sha256": _backend_projection_field(None, missing_status="ABSENT"),
        "helper_sha256": _backend_projection_field(None, missing_status="ABSENT"),
        "helper_protocol_revision": _backend_projection_field(None, missing_status="ABSENT"),
        "package_version": _backend_projection_field(None, missing_status="ABSENT"),
        "model_identity": _backend_projection_field(None, missing_status="ABSENT"),
        "model_revision": _backend_projection_field(None, missing_status="ABSENT"),
        "model_blob_sha256": _backend_projection_field(None, missing_status="ABSENT"),
        "raw_transcript_revision": _backend_projection_field(None, missing_status="ABSENT"),
        "initial_aligned_output_revision": _backend_projection_field(None, missing_status="ABSENT"),
        "forced_output_revision": _backend_projection_field(None, missing_status="ABSENT"),
        "capability_revision": _backend_projection_field(None),
    }

    if lineage == "ctc":
        seed = report.get("ctc_bounded_retry_capability_seed")
        runtime = report.get(_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY)
        source = seed if isinstance(seed, dict) else runtime
        alignment_sha = _backend_projection_attr(source, "alignment_audio_sha256")
        alignment_source = _backend_projection_attr(source, "alignment_audio_source")
        if alignment_sha is None:
            alignment_sha = report.get("vocal_stem_sha256")
        if alignment_source is None:
            alignment_source = report.get("ctc_audio_source")
        fields["alignment_audio_sha256"] = _backend_projection_field(alignment_sha)
        fields["alignment_audio_source"] = _backend_projection_field(alignment_source)
        fields["audio_revision"] = _backend_projection_field(alignment_sha or source_audio_sha256)
        for name, field_name in (
            ("helper_sha256", "helper_sha256"),
            ("helper_protocol_revision", "helper_protocol_revision"),
            ("model_identity", "model_identity"),
            ("capability_revision", "capability_revision"),
        ):
            fields[name] = _backend_projection_field(
                _backend_projection_attr(source, field_name),
                missing_status="UNAVAILABLE",
            )
    elif lineage == "whisper-family":
        provenance = report.get(_RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY)
        if provenance is not None:
            fields["executable_sha256"] = _backend_projection_field(
                _backend_projection_attr(provenance, "raw_executable_sha256")
            )
            fields["model_sha256"] = _backend_projection_field(
                _backend_projection_attr(provenance, "raw_model_sha256")
            )
            fields["python_sha256"] = _backend_projection_field(
                _backend_projection_attr(provenance, "python_sha256")
            )
            fields["helper_sha256"] = _backend_projection_field(
                _backend_projection_attr(provenance, "helper_sha256")
            )
            fields["helper_protocol_revision"] = _backend_projection_field(
                _backend_projection_attr(provenance, "helper_protocol_revision")
            )
            fields["package_version"] = _backend_projection_field(
                _backend_projection_attr(provenance, "whisperx_package_version")
            )
            fields["model_identity"] = _backend_projection_field(
                _backend_projection_attr(provenance, "align_model_id")
            )
            fields["model_revision"] = _backend_projection_field(
                _backend_projection_attr(provenance, "align_model_revision")
            )
            fields["model_blob_sha256"] = _backend_projection_field(
                _backend_projection_attr(provenance, "align_model_blob_sha256")
            )
            fields["raw_transcript_revision"] = _backend_projection_field(
                _backend_projection_attr(provenance, "raw_transcript_revision")
            )
            fields["initial_aligned_output_revision"] = _backend_projection_field(
                _backend_projection_attr(provenance, "initial_aligned_output_revision")
            )
            fields["forced_output_revision"] = _backend_projection_field(
                _backend_projection_attr(provenance, "forced_output_revision")
            )
            fields["capability_revision"] = _backend_projection_field(
                _backend_projection_attr(provenance, "capability_revision")
            )
            source_sha = _backend_projection_attr(provenance, "source_audio_sha256")
            fields["source_audio_sha256"] = _backend_projection_field(source_sha or source_audio_sha256)
            fields["audio_revision"] = _backend_projection_field(source_sha or source_audio_sha256)
            fields["alignment_audio_sha256"] = _backend_projection_field(
                _backend_projection_attr(provenance, "alignment_wav_sha256")
            )
            fields["alignment_audio_source"] = _backend_projection_field(
                "whisperx-decoded-wav"
            )
        else:
            try:
                executable_sha256, model_sha256 = _whispercpp_runtime_artifact_hashes(args) or (None, None)
            except (OSError, LrcError):
                executable_sha256, model_sha256 = None, None
            fields["executable_sha256"] = _backend_projection_field(executable_sha256)
            fields["model_sha256"] = _backend_projection_field(model_sha256)
            fields["alignment_audio_sha256"] = _backend_projection_field(source_audio_sha256)
            fields["alignment_audio_source"] = _backend_projection_field("source-audio")
            fields["capability_revision"] = _backend_projection_derived_field(
                _timing_content_digest({
                    "kind": "whispercpp-raw",
                    "backend_lineage": lineage,
                    "source_audio_sha256": source_audio_sha256,
                    "executable_sha256": executable_sha256,
                    "model_sha256": model_sha256,
                })
                if source_audio_sha256 and executable_sha256 and model_sha256
                else None
            )

    try:
        candidate_report_revision = timing_report_evaluation_revision(report)
    except (LrcError, TypeError, ValueError):
        candidate_report_revision = None
    safe_capability = {
        name: value for name, value in fields.items()
    }
    return {
        "backend": backend,
        "backend_lineage": fields["backend_lineage"],
        "capability": safe_capability,
        "candidate_report_revision": _backend_projection_field(candidate_report_revision),
        "algorithm_source_sha256": _backend_projection_field(
            validated_loaded_source_sha256()
        ),
        "algorithm_revision": _backend_projection_field(LRC_ALGORITHM_REVISION),
    }


def _public_local_capability_summary(
    audio_path: Path,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    capability = build_local_whisper_capability(audio_path, args)
    if capability is None:
        return None
    fields = {
        "backend_lineage": _backend_projection_field(capability.backend_lineage),
        "source_audio_sha256": _backend_projection_field(capability.audio_sha256),
        "audio_revision": _backend_projection_field(capability.audio_sha256),
        "alignment_audio_sha256": _backend_projection_field(capability.audio_sha256),
        "alignment_audio_source": _backend_projection_field("source-audio"),
        "executable_sha256": _backend_projection_field(capability.whisper_cli_sha256),
        "model_sha256": _backend_projection_field(capability.whisper_model_sha256),
        "python_sha256": _backend_projection_field(
            _stable_file_sha256(Path(capability.whisperx_python_path))
        ),
        "helper_sha256": _backend_projection_field(capability.whisperx_helper_sha256),
        "helper_protocol_revision": _backend_projection_field(capability.protocol_revision),
        "package_version": _backend_projection_field(None, missing_status="ABSENT"),
        "model_identity": _backend_projection_field("whisper.cpp:ggml-large-v3"),
        "model_revision": _backend_projection_field(None, missing_status="ABSENT"),
        "model_blob_sha256": _backend_projection_field(capability.whisper_model_sha256),
        "raw_transcript_revision": _backend_projection_field(None, missing_status="ABSENT"),
        "initial_aligned_output_revision": _backend_projection_field(None, missing_status="ABSENT"),
        "forced_output_revision": _backend_projection_field(None, missing_status="ABSENT"),
        "capability_revision": _backend_projection_field(capability.capability_revision),
    }
    return {
        "producer": "local-whisper-aligner",
        "capability": fields,
        "algorithm_source_sha256": _backend_projection_field(
            validated_loaded_source_sha256()
        ),
        "algorithm_revision": _backend_projection_field(LRC_ALGORITHM_REVISION),
    }


def _postcentral_bind_ctc_assignment_tail_coexistence(
    report: dict[str, object],
    entries: list[LyricEntry],
) -> None:
    """Retain valid CTC terminal tails beside already-present peer-lineage tails.

    The ordinary fallback intentionally suppresses CTC whenever *any* runtime tail
    already occupies the row.  In post-Central existing-conflict replay that is
    too coarse: a same-lineage Whisper-family tail can be unusable for a
    Whisper-family final predecessor while an already-computed CTC terminal tail
    is the required independent lineage proof.  Generate the normal CTC fallback
    observations on an isolated copy, then merge only those runtime observations
    back into this already-private post-Central report.

    Candidate sets, Central inputs, backend selection and neutral arbitration are
    untouched.  Existing runtime observations are preserved and revisions are
    deduplicated by the canonical tail-observation freeze helper.
    """
    existing = _runtime_tail_observation_map(report)
    scratch = copy.deepcopy(report)
    scratch[_RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = {}
    bind_runtime_ctc_assignment_tail_fallback(scratch, entries)
    generated = _runtime_tail_observation_map(scratch)
    for index, observations in generated.items():
        for observation in observations:
            provenance = observation.provenance
            if (
                observation.producer != "ctc-assignment-terminal-row"
                or provenance is None
                or _tail_backend_family(provenance.backend_lineage) != "ctc"
            ):
                continue
            existing.setdefault(index, []).append(observation)
    report[_RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = (
        _freeze_runtime_tail_observation_map(existing)
    )


def _postcentral_tail_blocked_peer_entries(
    report: dict[str, object],
    state: FinalTimingState,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return all tail-blocked rows and the subset a CTC tail can actually clear.

    A CTC predecessor tail cannot falsify a surviving cluster that itself
    contains CTC-family support.  Such rows may become REBOUND, but this producer
    cannot remove their final review requirement.  Keep them out of the helper
    worklist rather than spending runtime on provably non-convergent evidence.
    """
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    bundles = report.get(_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY)
    if not isinstance(seed, dict) or not (
        isinstance(bundles, tuple)
        and bundles
        and all(isinstance(item, EvaluatedBackendBundle) for item in bundles)
    ):
        return (), ()
    rebound = _postcentral_rebind_existing_peer_conflicts(seed, state, bundles)
    details = rebound.get("cross_backend_material_conflicts")
    if not isinstance(details, dict) or details.get("status") != "RETRIEVED":
        return (), ()
    rows = details.get("entries")
    if not isinstance(rows, list):
        return (), ()
    blocked: set[int] = set()
    actionable: set[int] = set()
    for row in rows:
        if not (
            isinstance(row, dict)
            and isinstance(row.get("entry"), int)
            and row.get("postcentral_replay_status") == "BLOCKED"
            and row.get("postcentral_replay_reason")
            == "final-valid-predecessor-tail-unavailable"
        ):
            continue
        entry_number = int(row["entry"])
        blocked.add(entry_number)
        clusters = row.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            continue
        all_clusters_ctc_disjoint = True
        for cluster in clusters:
            families = cluster.get("support_families") if isinstance(cluster, dict) else None
            if (
                not isinstance(families, list)
                or not families
                or any(not isinstance(item, str) or not item for item in families)
                or "ctc" in families
            ):
                all_clusters_ctc_disjoint = False
                break
        if all_clusters_ctc_disjoint:
            actionable.add(entry_number)
    return tuple(sorted(blocked)), tuple(sorted(actionable))


def _postcentral_plan_targeted_joint_ctc_tail_requests(
    context: PostcentralCtcTailProducerContext,
    blocked_entries: tuple[int, ...],
    state: FinalTimingState,
    entries: list[LyricEntry],
    duration: float,
) -> tuple[CrossLineRetryRequest, ...]:
    """Plan joint CTC probes only for already-existing tail-blocked peer rows.

    The final selected state is immutable input.  The request may observe the
    previous+current lyric pair, but its current-row candidate is never admitted
    anywhere; only a provenance-valid previous_tail_evidence can survive.
    """
    capability = context.capability
    epoch = context.epoch
    if (
        capability.capability_revision != epoch.capability_revision
        or len(state.decisions) != len(entries)
        or not math.isfinite(duration)
        or duration <= 0.0
    ):
        return ()
    duration_revision = _timing_content_digest({"duration": round(duration, 6)})
    target_set = set(blocked_entries)
    requests: list[CrossLineRetryRequest] = []
    for entry_number in sorted(target_set):
        index = entry_number - 1
        if not 0 < index < len(entries):
            continue
        previous_decision = state.decisions[index - 1]
        current_decision = state.decisions[index]
        # A CTC tail is not independent of a CTC-derived final predecessor.
        previous_families = _strict_candidate_support_families_for_final_context(
            previous_decision.candidate
        )
        if not previous_families or "ctc" in previous_families:
            continue

        current_ref = _retry_candidate_reference(current_decision.candidate)
        previous_ref = _retry_candidate_reference(previous_decision.candidate)
        next_ref = (
            _retry_candidate_reference(state.decisions[index + 1].candidate)
            if index + 1 < len(entries)
            else None
        )
        previous_tail = derive_reliable_tail(previous_decision.candidate)
        relation = current_decision.evidence.ownership
        context_indexes = (index - 1, index)
        context_candidates = (previous_decision.candidate, current_decision.candidate)
        transcripts = tuple(entry_sung_text(entries[item]) for item in context_indexes)
        transcript_digests = tuple(_timing_content_digest(text) for text in transcripts)
        row_mapping = ((index - 1, 0), (index, 1))

        previous_start = previous_decision.written_time.seconds
        current_start = current_decision.written_time.seconds
        previous_previous_start = (
            state.decisions[index - 2].written_time.seconds if index >= 2 else 0.0
        )
        spacing = max(HOP_SECONDS, current_start - previous_start)
        pre_roll = min(3.0, max(1.0, spacing * 0.45))
        lower_neighbor = previous_previous_start + HOP_SECONDS if index >= 2 else 0.0
        window_start = max(lower_neighbor, previous_start - pre_roll)
        if index + 1 < len(entries):
            next_start = state.decisions[index + 1].written_time.seconds
            window_end = min(
                duration,
                next_start + 2.0 * CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
                current_start + 14.0,
            )
        else:
            window_end = min(
                duration, current_start + min(14.0, max(7.0, spacing * 1.8))
            )
        if not 2.0 <= window_end - window_start <= 24.0:
            continue

        mapping_revision = _timing_content_digest(row_mapping)
        window_revision = _timing_content_digest({
            "mode": "postcentral-existing-conflict-joint-previous-tail-only",
            "start": round(window_start, 6),
            "end": round(window_end, 6),
            "previous_previous_start": round(previous_previous_start, 6),
            "previous": previous_ref.candidate_id,
            "current": current_ref.candidate_id,
            "next": next_ref.candidate_id if next_ref else None,
            "duration_revision": duration_revision,
            "ctc_bundle_revision": context.bundle_revision,
            "blocked_entries": list(sorted(target_set)),
        })
        payload = {
            "retry_epoch_revision": epoch.retry_epoch_revision,
            "entry_index": index,
            "ordinal": 3,
            "capability_revision": capability.capability_revision,
            "alignment_audio_revision": capability.alignment_audio_sha256,
            "current": current_ref,
            "previous": previous_ref,
            "next_bound": next_ref,
            "audio_end_seconds": round(duration, 6),
            "audio_duration_revision": duration_revision,
            "previous_tail_evidence_revision": previous_tail.tail_evidence_revision,
            "previous_tail_start": previous_tail.reliable_tail_start,
            "previous_tail_end": previous_tail.reliable_tail_end,
            "previous_tail_uncertainty": previous_tail.uncertainty_seconds,
            "previous_tail_validity": previous_tail.validity,
            "ownership_relation_revision": relation.relation_revision,
            "context_entry_indexes": context_indexes,
            "context_candidate_ids": tuple(
                item.candidate_id for item in context_candidates
            ),
            "transcript_digests": transcript_digests,
            "transcript_digest": _timing_content_digest(transcripts),
            "row_mapping": row_mapping,
            "row_mapping_revision": mapping_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "window_definition_revision": window_revision,
        }
        requests.append(CrossLineRetryRequest(
            **payload,
            request_revision=_timing_content_digest({
                **payload,
                "current": current_ref.__dict__,
                "previous": previous_ref.__dict__,
                "next_bound": next_ref.__dict__ if next_ref else None,
            }),
        ))
    return tuple(sorted(requests, key=lambda item: (item.entry_index, item.request_revision)))


def _postcentral_bundle_with_additional_tail_observations(
    base_bundle: EvaluatedBackendBundle,
    observations: tuple[IndependentTailEvidence, ...],
    context: TailEvidenceContext,
) -> EvaluatedBackendBundle:
    """Append tail-only observations to one private bundle, never candidate sets."""
    if not observations or not _tail_context_is_valid(context):
        return base_bundle
    context_revision = _timing_content_digest(_bundle_tail_context_payload(context))
    bindings = list(base_bundle.tail_bindings)
    for observation in observations:
        if (
            not _tail_provenance_is_valid(observation)
            or not _tail_provenance_matches_context(observation, context)
        ):
            continue
        bindings.append(TailObservationBinding(
            observation=observation,
            context=context,
            context_revision=context_revision,
        ))
    unique: dict[tuple[str, str], TailObservationBinding] = {}
    for binding in bindings:
        revision = (
            binding.observation.provenance.observation_revision
            if binding.observation.provenance is not None
            else binding.observation.source_revision
        )
        unique[(revision, binding.context_revision)] = binding
    tail_bindings = tuple(unique[key] for key in sorted(unique))
    if tail_bindings == base_bundle.tail_bindings:
        return base_bundle
    candidate_payload = _bundle_candidate_sets_payload(base_bundle.candidate_sets)
    bundle_payload = {
        "backend": base_bundle.evaluated.backend,
        "candidate_sets": candidate_payload,
        "candidate_sets_revision": base_bundle.candidate_sets_revision,
        "report_projection_revision": base_bundle.report_projection_revision,
        "central_scaffold_json": base_bundle.central_scaffold_json,
        "content_identity_observations": [
            item.__dict__ for item in base_bundle.content_identity_observations
        ],
        "tail_bindings": [_bundle_tail_binding_payload(item) for item in tail_bindings],
        "capability_projection_json": base_bundle.capability_projection_json,
        "source_audio_revision": base_bundle.source_audio_revision,
        "alignment_audio_revision": base_bundle.alignment_audio_revision,
        "algorithm_source_revision": base_bundle.algorithm_source_revision,
        "algorithm_revision": base_bundle.algorithm_revision,
        "input_revision": base_bundle.input_revision,
        "diagnostics_revision": base_bundle.diagnostics_revision,
        "diagnostics_semantic_revision": base_bundle.diagnostics_semantic_revision,
        "evaluation_revision": base_bundle.evaluation_revision,
    }
    enriched = replace(
        base_bundle,
        tail_bindings=tail_bindings,
        bundle_revision=_timing_content_digest(bundle_payload),
    )
    _validate_evaluated_backend_bundle(enriched)
    return enriched


def _postcentral_acquire_targeted_joint_ctc_tails(
    report: dict[str, object],
    state: FinalTimingState,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    """Acquire CTC joint predecessor tails after neutral selection is immutable."""
    context = report.get(_RUNTIME_POSTCENTRAL_CTC_TAIL_PRODUCER_CONTEXT_KEY)
    bundles = report.get(_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY)
    if not isinstance(context, PostcentralCtcTailProducerContext) or not (
        isinstance(bundles, tuple)
        and bundles
        and all(isinstance(item, EvaluatedBackendBundle) for item in bundles)
    ):
        return None
    blocked_entries, actionable_entries = _postcentral_tail_blocked_peer_entries(
        report, state
    )
    requests = _postcentral_plan_targeted_joint_ctc_tail_requests(
        context, actionable_entries, state, entries, duration
    )
    try:
        timeout_seconds = float(getattr(args, "crossline_retry_timeout", 180.0))
    except (TypeError, ValueError):
        timeout_seconds = 180.0
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        timeout_seconds = 180.0

    outcomes: list[CrossLineRetryOutcome] = []
    cache_hits = 0
    helper_invocations = 0
    if requests:
        cached_requests: list[CrossLineRetryRequest] = []
        uncached_requests: list[CrossLineRetryRequest] = []
        for request in requests:
            key = _crossline_retry_helper_cache_key(request, context.capability)
            if key in context.helper_cache:
                cached_requests.append(request)
            else:
                uncached_requests.append(request)
        for request in cached_requests:
            cache_hits += 1
            outcomes.append(invoke_crossline_retry(
                request,
                context.capability,
                entries,
                helper_cache=context.helper_cache,
                timeout_seconds=timeout_seconds,
            ))
        if uncached_requests:
            helper_invocations += len(uncached_requests)
            def run_one(request: CrossLineRetryRequest) -> tuple[CrossLineRetryOutcome, dict[str, tuple[str, object]]]:
                private_cache: dict[str, tuple[str, object]] = {}
                outcome = invoke_crossline_retry(
                    request,
                    context.capability,
                    entries,
                    helper_cache=private_cache,
                    timeout_seconds=timeout_seconds,
                )
                return outcome, private_cache
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(DEFAULT_EVIDENCE_WORKERS, len(uncached_requests)),
                thread_name_prefix="lrc-postcentral-ctc-tail",
            ) as pool:
                future_map = {
                    pool.submit(run_one, request): request
                    for request in uncached_requests
                }
                for future in concurrent.futures.as_completed(future_map):
                    outcome, private_cache = future.result()
                    context.helper_cache.update(private_cache)
                    outcomes.append(outcome)
    outcomes.sort(key=lambda item: (item.entry_index, item.request_revision))

    accepted: list[IndependentTailEvidence] = []
    covered_conflicts: list[int] = []
    discarded_current_candidates = 0
    for outcome in outcomes:
        if outcome.candidate is not None:
            discarded_current_candidates += 1
        evidence = outcome.previous_tail_evidence
        if evidence is None:
            continue
        if (
            evidence.entry_index != outcome.entry_index - 1
            or evidence.producer != "joint-ctc-boundary-previous-tail"
            or not _tail_provenance_is_valid(evidence)
            or not _tail_provenance_matches_context(evidence, context.tail_context)
        ):
            continue
        accepted.append(evidence)
        covered_conflicts.append(outcome.entry_index + 1)
    accepted_tuple = tuple({
        item.provenance.observation_revision: item
        for item in accepted
        if item.provenance is not None
    }[key] for key in sorted({
        item.provenance.observation_revision
        for item in accepted
        if item.provenance is not None
    }))

    revised_bundles: list[EvaluatedBackendBundle] = []
    bundle_changed = False
    for bundle in bundles:
        if (
            bundle.evaluated.backend == "ctc"
            and bundle.bundle_revision == context.bundle_revision
            and bundle.candidate_sets_revision == context.candidate_sets_revision
        ):
            revised = _postcentral_bundle_with_additional_tail_observations(
                bundle, accepted_tuple, context.tail_context
            )
            bundle_changed = bundle_changed or revised.bundle_revision != bundle.bundle_revision
            revised_bundles.append(revised)
        else:
            revised_bundles.append(bundle)
    if accepted_tuple and not bundle_changed:
        # Evidence that cannot be bound to its exact originating bundle is never
        # allowed to float into another context.
        accepted_tuple = ()
        covered_conflicts = []
    if bundle_changed:
        report[_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY] = tuple(revised_bundles)

    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": "postcentral-existing-conflict-targeted-joint-ctc-previous-tail-v1",
        "authority": "tail-evidence-only",
        "blocked_input_entries": list(blocked_entries),
        "ctc_tail_actionable_entries": list(actionable_entries),
        "requested_conflict_entries": [item.entry_index + 1 for item in requests],
        "request_count": len(requests),
        "outcome_count": len(outcomes),
        "valid_tail_count": len(accepted_tuple),
        "covered_conflict_entries": sorted(set(covered_conflicts)),
        "tail_entry_indexes": sorted({item.entry_index + 1 for item in accepted_tuple}),
        "current_probe_candidate_discarded_count": discarded_current_candidates,
        "candidate_admission_count": 0,
        "candidate_sets_mutated": False,
        "central_replayed": False,
        "backend_reselected": False,
        "neutral_replayed": False,
        "timestamp_mutated": False,
        "cache_hits": cache_hits,
        "helper_invocations": helper_invocations,
        "bundle_changed": bundle_changed,
        "outcomes": [_retry_diagnostic_payload(item) for item in outcomes],
    }
    payload["revision"] = _timing_content_digest(payload)
    return payload


# Post-Central third acoustic lineage: exact known-lyric Japanese HuBERT forced
# alignment.  This producer is deliberately separate from both the Whisper /
# WhisperX-Wav2Vec2 family and the MMS-CTC family already used by production.
# It can emit predecessor terminal geometry plus a runtime-only, occurrence-bound
# current-opening observation.  Neither path can admit or select a candidate.
_KNOWN_LYRIC_HUBERT_LINEAGE = "hubert-ja-known-lyric"
_KNOWN_LYRIC_HUBERT_PRODUCER = "hubert-ja-known-lyric-terminal"
_KNOWN_LYRIC_HUBERT_ONSET_PRODUCER = "hubert-ja-known-lyric-current-onset"
_KNOWN_LYRIC_HUBERT_MODEL_ID = "TKU410410103/hubert-base-japanese-asr"
_KNOWN_LYRIC_HUBERT_MODEL_REVISION = "9206637a0f07d78aa9101ff9439a9d7fe1a43778"
_KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256 = "d6f5034777c0542db6c587d57db497206d0500f7eca8fb8bc7962301f45d54d9"
_KNOWN_LYRIC_HUBERT_WHISPERX_VERSION = "3.8.6"
_KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION = "known-lyric-hubert-joint-terminal-v7"
_KNOWN_LYRIC_HUBERT_LANGUAGE = "ja-hiragana"
_KNOWN_LYRIC_HUBERT_AUDIO_LOADER = "python-wave-pcm16-mono-16k-v1"
_KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT = "embedded-viterbi-token-emission-geometry-group-confidence-hypothesis-neutral-v5"
_HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY = "hubert_known_lyric_existing_candidate_recovery"
_HUBERT_EXISTING_CANDIDATE_RECOVERY_MODE = (
    "precommit-hubert-existing-candidate-central-recovery-v1"
)
_PEER_EXISTING_CANDIDATE_RECOVERY_KEY = "independent_peer_existing_candidate_recovery"
_PEER_EXISTING_CANDIDATE_RECOVERY_MODE = (
    "precommit-independent-peer-existing-candidate-central-recovery-v1"
)
_PEER_DUPLICATE_RECOMPUTE_KEY = "postcentral_duplicate_occurrence_offset_recompute"
_HUBERT_RECOVERY_DIRECT_ONSET_KIND = "known-lyric-onset"
_KNOWN_LYRIC_HUBERT_HELPER_SOURCE = '#!/usr/bin/env python3\nfrom __future__ import annotations\nimport argparse\nimport array\nfrom dataclasses import dataclass\nimport hashlib\nimport importlib.metadata\nimport json\nimport math\nimport sys\nimport unicodedata\nimport wave\nfrom pathlib import Path\n\nEXPECTED_MODEL_ID = "__LRC_KNOWN_LYRIC_HUBERT_MODEL_ID__"\nEXPECTED_MODEL_COMMIT = "__LRC_KNOWN_LYRIC_HUBERT_MODEL_REVISION__"\nEXPECTED_MODEL_BLOB_SHA256 = "__LRC_KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256__"\nPROTOCOL_REVISION = "__LRC_KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION__"\nSAMPLE_RATE = 16000\nMIN_INPUT_SAMPLES = 400\n\n\ndef sha256_file(path: Path) -> str:\n    h = hashlib.sha256()\n    with path.open("rb") as f:\n        for block in iter(lambda: f.read(1024 * 1024), b""):\n            h.update(block)\n    return h.hexdigest()\n\n\ndef locate_model_blob() -> Path | None:\n    candidates: list[Path] = []\n    try:\n        from huggingface_hub import try_to_load_from_cache\n        value = try_to_load_from_cache(\n            EXPECTED_MODEL_ID, "model.safetensors", revision=EXPECTED_MODEL_COMMIT\n        )\n        if isinstance(value, str):\n            candidates.append(Path(value))\n    except Exception:\n        pass\n    try:\n        from huggingface_hub.constants import HF_HUB_CACHE\n        repo = EXPECTED_MODEL_ID.replace("/", "--")\n        candidates.append(\n            Path(HF_HUB_CACHE) / f"models--{repo}" / "snapshots" /\n            EXPECTED_MODEL_COMMIT / "model.safetensors"\n        )\n    except Exception:\n        pass\n    for path in candidates:\n        try:\n            if path.is_file() and sha256_file(path) == EXPECTED_MODEL_BLOB_SHA256:\n                return path\n        except OSError:\n            continue\n    return None\n\n\ndef load_pcm16_mono_16k(path: Path):\n    import torch\n    with wave.open(str(path), "rb") as wf:\n        if wf.getnchannels() != 1:\n            raise ValueError(f"wav-channel-count:{wf.getnchannels()}")\n        if wf.getframerate() != SAMPLE_RATE:\n            raise ValueError(f"wav-sample-rate:{wf.getframerate()}")\n        if wf.getsampwidth() != 2:\n            raise ValueError(f"wav-sample-width:{wf.getsampwidth()}")\n        if wf.getcomptype() != "NONE":\n            raise ValueError(f"wav-compression:{wf.getcomptype()}")\n        raw = wf.readframes(wf.getnframes())\n    values = array.array("h")\n    values.frombytes(raw)\n    if sys.byteorder != "little":\n        values.byteswap()\n    if not values:\n        raise ValueError("wav-empty")\n    return torch.tensor(values, dtype=torch.float32).div_(32768.0).unsqueeze(0)\n\n\ndef clean_reading(text: str, vocab: dict[str, int]) -> str:\n    normalized = unicodedata.normalize("NFKC", text)\n    clean: list[str] = []\n    for char in normalized:\n        if char.isspace():\n            if "|" in vocab:\n                clean.append("|")\n            continue\n        if char in vocab:\n            clean.append(char)\n            continue\n        if unicodedata.category(char).startswith("P"):\n            continue\n        raise ValueError(f"unsupported-reading-character:{ord(char):04x}")\n    return "".join(clean)\n\n\ndef resolve_device(requested: str, cuda_available: bool) -> tuple[str, str]:\n    requested_device = str(requested).strip() or "auto"\n    if requested_device.lower() == "auto":\n        resolved_device = "cuda" if cuda_available else "cpu"\n    else:\n        resolved_device = requested_device\n    return requested_device, resolved_device\n\n\ndef blank_id(processor, vocab: dict[str, int]) -> int:\n    token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)\n    if isinstance(token_id, int) and token_id >= 0:\n        return token_id\n    for token in ("<pad>", "[PAD]", "[pad]"):\n        if token in vocab:\n            return int(vocab[token])\n    raise ValueError("blank-token-unavailable")\n\n\n@dataclass(frozen=True)\nclass Point:\n    token_index: int\n    time_index: int\n    score: float\n    emitted: bool\n\n\n@dataclass(frozen=True)\nclass Segment:\n    label: str\n    start: int\n    end: int\n    score: float\n\n\ndef get_trellis(emission, tokens: list[int], blank: int):\n    import torch\n    num_frame = emission.size(0)\n    num_tokens = len(tokens)\n    if num_tokens <= 0 or num_frame < num_tokens:\n        raise ValueError("ctc-insufficient-frames")\n    trellis = torch.empty((num_frame + 1, num_tokens + 1))\n    trellis[0, 0] = 0\n    trellis[1:, 0] = torch.cumsum(emission[:, blank], 0)\n    trellis[0, -num_tokens:] = -float("inf")\n    trellis[-num_tokens:, 0] = float("inf")\n    for t in range(num_frame):\n        trellis[t + 1, 1:] = torch.maximum(\n            trellis[t, 1:] + emission[t, blank],\n            trellis[t, :-1] + emission[t, tokens],\n        )\n    return trellis\n\n\ndef backtrack(trellis, emission, tokens: list[int], blank: int):\n    j = trellis.size(1) - 1\n    t_start = int(trellis[:, j].argmax().item())\n    path: list[Point] = []\n    for t in range(t_start, 0, -1):\n        stayed = trellis[t - 1, j] + emission[t - 1, blank]\n        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]\n        use_change = bool(changed > stayed)\n        token_id = tokens[j - 1] if use_change else blank\n        prob = float(emission[t - 1, token_id].exp().item())\n        path.append(Point(j - 1, t - 1, prob, use_change))\n        if use_change:\n            j -= 1\n        if j == 0:\n            break\n    if j != 0:\n        return None\n    return path[::-1]\n\n\ndef merge_repeats(path: list[Point], transcript: str) -> list[Segment]:\n    # Geometry and support confidence are intentionally separated.  The actual\n    # token-emission frame owns the conservative character bounds, while the\n    # full contiguous CTC path group for that transcript token owns confidence.\n    # Blank-stay frames therefore remain visible as positive inter-token gaps\n    # without throwing away the support score semantics used by the frozen tail\n    # reliability policy.\n    i1 = i2 = 0\n    segments: list[Segment] = []\n    expected_index = 0\n    while i1 < len(path):\n        while i2 < len(path) and path[i1].token_index == path[i2].token_index:\n            i2 += 1\n        group = path[i1:i2]\n        emitted = [point for point in group if point.emitted]\n        if len(emitted) != 1:\n            raise ValueError("ctc-token-emission-cardinality-mismatch")\n        point = emitted[0]\n        if point.token_index != expected_index or expected_index >= len(transcript):\n            raise ValueError("ctc-token-emission-order-mismatch")\n        support_score = sum(item.score for item in group) / len(group)\n        segments.append(Segment(\n            transcript[expected_index],\n            point.time_index,\n            point.time_index + 1,\n            support_score,\n        ))\n        expected_index += 1\n        i1 = i2\n    if expected_index != len(transcript):\n        raise ValueError("ctc-token-emission-count-mismatch")\n    return segments\n\n\ndef align_chars(emission, transcript: str, vocab: dict[str, int], blank: int, start: float, end: float):\n    tokens = [int(vocab[char]) for char in transcript]\n    trellis = get_trellis(emission, tokens, blank)\n    path = backtrack(trellis, emission, tokens, blank)\n    if path is None:\n        raise ValueError("backtrack-unavailable")\n    merged = merge_repeats(path, transcript)\n    if len(merged) != len(transcript):\n        raise ValueError("incomplete-character-alignment")\n    frame_count = emission.size(0)\n    if frame_count <= 0:\n        raise ValueError("invalid-emission-frame-count")\n    ratio = (end - start) / frame_count\n    chars: list[dict[str, object]] = []\n    for item in merged:\n        char_start = float(item.start) * ratio + start\n        char_end = float(item.end) * ratio + start\n        score = float(item.score)\n        if not all(math.isfinite(v) for v in (char_start, char_end, score)):\n            raise ValueError("nonfinite-character-geometry")\n        if char_end < char_start:\n            raise ValueError("negative-character-duration")\n        chars.append({\n            "char": item.label,\n            "start": round(char_start, 6),\n            "end": round(char_end, 6),\n            "score": round(score, 8),\n        })\n    return chars, ratio\n\n\ndef main() -> int:\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--audio", required=True)\n    ap.add_argument("--request", required=True)\n    ap.add_argument("--output", required=True)\n    ap.add_argument("--device", default="cuda")\n    args = ap.parse_args()\n\n    import torch\n    from transformers import AutoModelForCTC, AutoProcessor\n\n    try:\n        transformers_version = importlib.metadata.version("transformers")\n        huggingface_hub_version = importlib.metadata.version("huggingface-hub")\n        torch_version = importlib.metadata.version("torch")\n    except importlib.metadata.PackageNotFoundError as exc:\n        raise SystemExit(f"runtime-package-unavailable:{exc.name}")\n    requested_device, resolved_device = resolve_device(\n        args.device, bool(torch.cuda.is_available())\n    )\n    if resolved_device.startswith("cuda") and not torch.cuda.is_available():\n        raise SystemExit("cuda-unavailable")\n\n    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))\n    if payload.get("protocol_revision") != PROTOCOL_REVISION:\n        raise SystemExit("protocol-revision-mismatch")\n    requests = payload.get("requests")\n    if not isinstance(requests, list):\n        raise SystemExit("malformed-request-list")\n\n    audio_t = load_pcm16_mono_16k(Path(args.audio))\n    processor = AutoProcessor.from_pretrained(\n        EXPECTED_MODEL_ID, revision=EXPECTED_MODEL_COMMIT\n    )\n    model = AutoModelForCTC.from_pretrained(\n        EXPECTED_MODEL_ID, revision=EXPECTED_MODEL_COMMIT\n    ).to(resolved_device)\n    model.eval()\n    model_blob = locate_model_blob()\n    if model_blob is None:\n        raise SystemExit("model-blob-revision-unavailable")\n\n    tokenizer = getattr(processor, "tokenizer", None)\n    if tokenizer is None or not hasattr(tokenizer, "get_vocab"):\n        raise SystemExit("tokenizer-vocab-unavailable")\n    vocab = {str(k): int(v) for k, v in tokenizer.get_vocab().items()}\n    blank = blank_id(processor, vocab)\n    vocab_revision = hashlib.sha256(\n        json.dumps(vocab, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")\n    ).hexdigest()\n    max_duration = float(audio_t.shape[1]) / SAMPLE_RATE\n    results: list[dict[str, object]] = []\n    for req in requests:\n        request_kind = str(req.get("request_kind", "pair-tail"))\n        result: dict[str, object] = {\n            "request_revision": req.get("request_revision"),\n            "conflict_entry": req.get("conflict_entry"),\n            "request_kind": request_kind,\n            "status": "invalid",\n        }\n        try:\n            start = float(req["window_start"])\n            end = float(req["window_end"])\n            if not (\n                math.isfinite(start) and math.isfinite(end)\n                and 0.0 <= start < end <= max_duration + 1e-3\n            ):\n                raise ValueError("window-invalid")\n            f1 = max(0, int(start * SAMPLE_RATE))\n            f2 = min(audio_t.shape[1], int(end * SAMPLE_RATE))\n            wav = audio_t[:, f1:f2]\n            if wav.shape[-1] < MIN_INPUT_SAMPLES:\n                wav = torch.nn.functional.pad(wav, (0, MIN_INPUT_SAMPLES - wav.shape[-1]))\n            inputs = processor(\n                wav.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"\n            )\n            input_values = inputs.input_values.to(resolved_device)\n            attention_mask = getattr(inputs, "attention_mask", None)\n            if attention_mask is not None:\n                attention_mask = attention_mask.to(resolved_device)\n            with torch.inference_mode():\n                output = model(input_values, attention_mask=attention_mask)\n                emission = torch.log_softmax(output.logits, dim=-1)[0].cpu().detach()\n\n            if request_kind == "pair-tail":\n                previous_reading = str(req["previous_reading"])\n                current_reading = str(req["current_reading"])\n                previous_clean = clean_reading(previous_reading, vocab)\n                current_clean = clean_reading(current_reading, vocab)\n                if len(previous_clean) < 2 or len(current_clean) < 2:\n                    raise ValueError("clean-reading-too-short")\n                combined = previous_clean + current_clean\n                chars, ratio = align_chars(\n                    emission, combined, vocab, blank, start, end\n                )\n                boundary = len(previous_clean)\n                previous_chars = chars[:boundary]\n                current_chars = chars[boundary:]\n                if (\n                    len(previous_chars) != len(previous_clean)\n                    or len(current_chars) != len(current_clean)\n                    or not previous_chars or not current_chars\n                ):\n                    raise ValueError("pair-boundary-mismatch")\n                if previous_chars[-1]["end"] > current_chars[0]["start"] + 0.08:\n                    raise ValueError("pair-order-overlap")\n                frame_slack = max(0.08, ratio * 4.0)\n                if current_chars[-1]["end"] >= end - frame_slack:\n                    raise ValueError("right-edge-pileup")\n                if previous_chars[0]["start"] <= start + min(0.04, frame_slack * 0.5):\n                    raise ValueError("left-edge-pileup")\n                result.update({\n                    "status": "aligned",\n                    "reason": "complete-known-lyric-hiragana-pair",\n                    "previous_clean_length": len(previous_clean),\n                    "current_clean_length": len(current_clean),\n                    "previous_chars": previous_chars,\n                    "current_chars": current_chars,\n                    "previous_first_start": previous_chars[0]["start"],\n                    "previous_last_end": previous_chars[-1]["end"],\n                    "current_first_start": current_chars[0]["start"],\n                    "current_last_end": current_chars[-1]["end"],\n                    "frame_seconds": round(ratio, 8),\n                })\n            elif request_kind == "current-neutral-onset-probe":\n                current_reading = str(req["current_reading"])\n                current_clean = clean_reading(current_reading, vocab)\n                if len(current_clean) < 2:\n                    raise ValueError("clean-reading-too-short")\n                current_chars, ratio = align_chars(\n                    emission, current_clean, vocab, blank, start, end\n                )\n                if len(current_chars) != len(current_clean) or not current_chars:\n                    raise ValueError("current-alignment-cardinality-mismatch")\n                frame_slack = max(0.08, ratio * 4.0)\n                if current_chars[-1]["end"] >= end - frame_slack:\n                    raise ValueError("right-edge-pileup")\n                if current_chars[0]["start"] <= start + min(0.04, frame_slack * 0.5):\n                    raise ValueError("left-edge-pileup")\n                result.update({\n                    "status": "aligned",\n                    "reason": "complete-known-lyric-hiragana-current-hypothesis-neutral",\n                    "current_clean_length": len(current_clean),\n                    "current_chars": current_chars,\n                    "current_first_start": current_chars[0]["start"],\n                    "current_last_end": current_chars[-1]["end"],\n                    "frame_seconds": round(ratio, 8),\n                })\n            else:\n                raise ValueError(f"unsupported-request-kind:{request_kind}")\n        except Exception as exc:\n            result["reason"] = f"{type(exc).__name__}:{exc}"\n        results.append(result)\n    output = {\n        "schema_version": 1,\n        "protocol_revision": PROTOCOL_REVISION,\n        "transformers_package_version": transformers_version,\n        "huggingface_hub_package_version": huggingface_hub_version,\n        "torch_package_version": torch_version,\n        "model_id": EXPECTED_MODEL_ID,\n        "model_revision": EXPECTED_MODEL_COMMIT,\n        "model_blob_sha256": EXPECTED_MODEL_BLOB_SHA256,\n        "vocab_revision": vocab_revision,\n        "requested_device": requested_device,\n        "device": resolved_device,\n        "language": "__LRC_KNOWN_LYRIC_HUBERT_LANGUAGE__",\n        "audio_loader": "__LRC_KNOWN_LYRIC_HUBERT_AUDIO_LOADER__",\n        "ctc_alignment": "__LRC_KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT__",\n        "results": results,\n    }\n    Path(args.output).write_text(\n        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"\n    )\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
for _known_lyric_hubert_contract_token, _known_lyric_hubert_contract_value in (
    ("__LRC_KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION__", _KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION),
    ("__LRC_KNOWN_LYRIC_HUBERT_MODEL_ID__", _KNOWN_LYRIC_HUBERT_MODEL_ID),
    ("__LRC_KNOWN_LYRIC_HUBERT_MODEL_REVISION__", _KNOWN_LYRIC_HUBERT_MODEL_REVISION),
    ("__LRC_KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256__", _KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256),
    ("__LRC_KNOWN_LYRIC_HUBERT_LANGUAGE__", _KNOWN_LYRIC_HUBERT_LANGUAGE),
    ("__LRC_KNOWN_LYRIC_HUBERT_AUDIO_LOADER__", _KNOWN_LYRIC_HUBERT_AUDIO_LOADER),
    ("__LRC_KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT__", _KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT),
):
    _KNOWN_LYRIC_HUBERT_HELPER_SOURCE = _KNOWN_LYRIC_HUBERT_HELPER_SOURCE.replace(
        _known_lyric_hubert_contract_token, _known_lyric_hubert_contract_value
    )
del _known_lyric_hubert_contract_token, _known_lyric_hubert_contract_value


def _known_lyric_hubert_helper_revision() -> str:
    return hashlib.sha256(
        _KNOWN_LYRIC_HUBERT_HELPER_SOURCE.encode("utf-8")
    ).hexdigest()


def _known_lyric_hiragana_reading(text: str) -> str | None:
    """Return a deterministic pykakasi hiragana projection of immutable lyrics.

    The reading itself is revision-bound in every request.  If the optional
    converter is unavailable or leaves an empty reading, this producer simply
    does not run; no transliteration is guessed inside the acoustic helper.
    """
    converter = optional_kakasi_converter()
    if converter is None:
        return None
    try:
        converted = converter.convert(numerals_for_alignment(unicodedata.normalize("NFKC", text)))  # type: ignore[attr-defined]
    except Exception:
        return None
    pieces: list[str] = []
    for item in converted:
        if not isinstance(item, dict):
            return None
        value = item.get("hira")
        if not isinstance(value, str):
            return None
        pieces.append(value)
    reading = unicodedata.normalize("NFKC", "".join(pieces)).strip()
    return reading or None


def _postcentral_residual_material_conflict_entries(
    report: dict[str, object],
    state: FinalTimingState,
) -> tuple[int, ...]:
    """Return peer rows still material after all already-available tail evidence."""
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    bundles = report.get(_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY)
    if not isinstance(seed, dict) or not (
        isinstance(bundles, tuple) and bundles
        and all(isinstance(item, EvaluatedBackendBundle) for item in bundles)
    ):
        return ()
    rebound = _postcentral_rebind_existing_peer_conflicts(seed, state, bundles)
    details = rebound.get("cross_backend_material_conflicts")
    rows = details.get("entries") if isinstance(details, dict) else None
    if not isinstance(rows, list):
        return ()
    return tuple(sorted({
        int(row["entry"])
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("entry"), int)
        and int(row["entry"]) > 1
        and isinstance(row.get("clusters"), list)
        and bool(row.get("clusters"))
    }))


def _postcentral_plan_known_lyric_hubert_requests(
    conflict_entries: tuple[int, ...],
    state: FinalTimingState,
    entries: list[LyricEntry],
    duration: float,
) -> tuple[dict[str, object], ...]:
    """Plan one immutable predecessor/current lyric pair per residual conflict."""
    if len(state.decisions) != len(entries) or not math.isfinite(duration) or duration <= 0.0:
        return ()
    readings = tuple(
        _known_lyric_hiragana_reading(entry_sung_text(entry))
        for entry in entries
    )
    reading_pair_counts = collections.Counter(
        (readings[index - 1], readings[index])
        for index in range(1, len(readings))
        if readings[index - 1] and readings[index]
    )
    requests: list[dict[str, object]] = []
    for entry_number in conflict_entries:
        index = entry_number - 1
        if not 0 < index < len(entries):
            continue
        previous_text = entry_sung_text(entries[index - 1])
        current_text = entry_sung_text(entries[index])
        if not normalize_match_text(previous_text) or not normalize_match_text(current_text):
            continue
        previous_reading = readings[index - 1]
        current_reading = readings[index]
        if not previous_reading or not current_reading:
            continue
        if reading_pair_counts[(previous_reading, current_reading)] != 1:
            # HuBERT sees readings, not kanji identity.  A repeated adjacent
            # reading pair cannot independently prove which lyric occurrence
            # produced the acoustic path, so it is excluded before inference.
            continue
        previous_start = state.decisions[index - 1].written_time.seconds
        current_start = state.decisions[index].written_time.seconds
        before_previous = state.decisions[index - 2].written_time.seconds if index >= 2 else 0.0
        next_start = state.decisions[index + 1].written_time.seconds if index + 1 < len(entries) else duration
        spacing = max(HOP_SECONDS, current_start - previous_start)
        lower = before_previous + HOP_SECONDS if index >= 2 else 0.0
        window_start = max(
            0.0,
            lower,
            previous_start - min(2.5, max(1.25, spacing * 0.5)),
        )
        window_end = min(duration, next_start + 1.5, current_start + 12.0)
        if not 2.0 <= window_end - window_start <= 24.0:
            continue
        transcript_revision = _timing_content_digest({
            "previous_text": previous_text,
            "current_text": current_text,
            "previous_reading": previous_reading,
            "current_reading": current_reading,
            "phonetic_system": "pykakasi-hiragana-v1",
        })
        row_payload = {
            "previous_entry_index": index - 1,
            "current_entry_index": index,
            "previous_text_revision": _timing_content_digest(previous_text),
            "current_text_revision": _timing_content_digest(current_text),
            "previous_reading_revision": _timing_content_digest(previous_reading),
            "current_reading_revision": _timing_content_digest(current_reading),
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        row_revision = _timing_content_digest(row_payload)
        request_payload: dict[str, object] = {
            "request_kind": "pair-tail",
            "conflict_entry": entry_number,
            "previous_entry_index": index - 1,
            "current_entry_index": index,
            "previous_reading": previous_reading,
            "current_reading": current_reading,
            "transcript_revision": transcript_revision,
            "row_revision": row_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        request_payload["request_revision"] = _timing_content_digest(request_payload)
        requests.append(request_payload)
    return tuple(sorted(
        requests,
        key=lambda item: (int(item["conflict_entry"]), str(item["request_revision"])),
    ))


def _hubert_neutral_probe_window_start(
    earliest_hypothesis_centiseconds: int,
) -> float:
    """Give full-line CTC fixed leading context independent of any winner."""
    return max(
        0.0,
        earliest_hypothesis_centiseconds / 100.0
        - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
    )


def _hubert_probe_window_covers_hypotheses(
    window_start: float,
    window_end: float,
    hypothesis_centiseconds: tuple[int, ...],
    direct_tolerance_centiseconds: int,
) -> bool:
    """Require the entire direct-onset neighborhood of each hypothesis."""
    if not hypothesis_centiseconds or direct_tolerance_centiseconds < 0:
        return False
    if not all(math.isfinite(value) for value in (window_start, window_end)):
        return False
    margin = direct_tolerance_centiseconds / 100.0
    return all(
        window_start <= centiseconds / 100.0 - margin + 1e-9
        and window_end >= centiseconds / 100.0 + margin - 1e-9
        for centiseconds in hypothesis_centiseconds
    )


def _hubert_probe_window_covers_request_hypotheses(
    request: dict[str, object],
    direct_tolerance_centiseconds: int,
) -> bool:
    """Fail closed on malformed/rebound requests at evidence consumption."""
    try:
        start = float(request["window_start"])
        end = float(request["window_end"])
        bindings = request["hypothesis_bindings"]
        if not isinstance(bindings, list) or not bindings:
            return False
        times = tuple(
            item[2] for item in bindings
            if isinstance(item, (list, tuple))
            and len(item) == 3
            and type(item[2]) is int
        )
        if len(times) != len(bindings):
            return False
        selected = request["final_selected_centiseconds"]
        if not isinstance(selected, int) or selected not in times:
            return False
        if min(times) != request["hypothesis_min_centiseconds"] or max(times) != request["hypothesis_max_centiseconds"]:
            return False
        expected_start = _hubert_neutral_probe_window_start(min(times))
        if abs(start - expected_start) > 1e-6:
            return False
        return _hubert_probe_window_covers_hypotheses(start, end, times, direct_tolerance_centiseconds)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _postcentral_plan_known_lyric_hubert_current_probe_requests(
    conflict_entries: tuple[int, ...],
    report: dict[str, object],
    state: FinalTimingState,
    bundles: tuple[EvaluatedBackendBundle, ...],
    entries: list[LyricEntry],
    duration: float,
) -> tuple[dict[str, object], ...]:
    """Plan hypothesis-neutral full-line HuBERT onset probes.

    The probe window contains the immutable final selected timestamp *and every
    member of every already-existing distant conflict cluster for this row*.
    No hypothesis is privileged by the crop itself.  The probe can corroborate
    the immutable final selection or fail closed; it never proposes/selects a
    timestamp and cannot admit a candidate.
    """
    if (
        len(state.decisions) != len(entries)
        or not math.isfinite(duration)
        or duration <= 0.0
        or not bundles
    ):
        return ()
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    if not isinstance(seed, dict):
        return ()
    rebound = _postcentral_rebind_existing_peer_conflicts(seed, state, bundles)
    details = rebound.get("cross_backend_material_conflicts")
    raw_rows = details.get("entries") if isinstance(details, dict) else None
    if not isinstance(raw_rows, list):
        return ()
    row_lookup = {
        int(row["entry"]): row
        for row in raw_rows
        if isinstance(row, dict) and isinstance(row.get("entry"), int)
    }
    candidate_lookup = _bundle_candidate_lookup(bundles)
    if candidate_lookup is None:
        return ()

    readings = tuple(
        _known_lyric_hiragana_reading(entry_sung_text(entry))
        for entry in entries
    )
    direct_tolerance_centiseconds = int(round(
        PREFIX_ACOUSTIC_ASSOCIATION_SECONDS * 100.0
    ))
    requests: list[dict[str, object]] = []
    for entry_number in conflict_entries:
        index = entry_number - 1
        if not 0 <= index < len(entries):
            continue
        reading = readings[index]
        row = row_lookup.get(entry_number)
        if not reading or not isinstance(row, dict):
            continue
        raw_clusters = row.get("clusters")
        if not isinstance(raw_clusters, list) or not raw_clusters:
            continue

        decision = state.decisions[index]
        hypotheses: dict[tuple[str, str], int] = {
            (
                decision.candidate.candidate_id,
                decision.candidate.generation_revision,
            ): decision.written_time.centiseconds
        }
        malformed = False
        for cluster in raw_clusters:
            if not isinstance(cluster, dict):
                malformed = True
                break
            ids = cluster.get("member_candidate_ids")
            revisions = cluster.get("member_generation_revisions")
            if (
                not isinstance(ids, list)
                or not isinstance(revisions, list)
                or not ids
                or len(ids) != len(revisions)
            ):
                malformed = True
                break
            for candidate_id, generation_revision in zip(ids, revisions):
                if not isinstance(candidate_id, str) or not isinstance(generation_revision, str):
                    malformed = True
                    break
                candidate = candidate_lookup.get((candidate_id, generation_revision))
                if candidate is None or candidate.entry_index != index:
                    malformed = True
                    break
                hypotheses[(candidate_id, generation_revision)] = (
                    candidate.written_time.centiseconds
                )
            if malformed:
                break
        if malformed or len(hypotheses) < 2:
            continue

        hypothesis_bindings = tuple(sorted(
            (
                candidate_id,
                generation_revision,
                centiseconds,
            )
            for (candidate_id, generation_revision), centiseconds in hypotheses.items()
        ))
        hypothesis_centiseconds = tuple(
            sorted({item[2] for item in hypothesis_bindings})
        )
        hypothesis_set_revision = _timing_content_digest({
            "mode": "existing-peer-hypothesis-set-v1",
            "entry_index": index,
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "hypothesis_bindings": [
                [candidate_id, generation_revision, centiseconds]
                for candidate_id, generation_revision, centiseconds
                in hypothesis_bindings
            ],
        })
        earliest_seconds = hypothesis_centiseconds[0] / 100.0
        latest_seconds = hypothesis_centiseconds[-1] / 100.0
        next_start = (
            state.decisions[index + 1].written_time.seconds
            if index + 1 < len(entries)
            else duration
        )
        # A current-row hypothesis already beyond the next immutable row cannot
        # be tested safely with a single current-line window without importing
        # next-row speech; leave that row unresolved.
        if (
            index + 1 < len(entries)
            and latest_seconds
            > next_start + PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
        ):
            continue

        window_start = _hubert_neutral_probe_window_start(
            hypothesis_centiseconds[0]
        )
        window_end = min(
            duration,
            next_start + CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
            earliest_seconds + 12.0,
        )
        if not 0.5 <= window_end - window_start <= 14.0:
            continue

        if not _hubert_probe_window_covers_hypotheses(
            window_start, window_end, hypothesis_centiseconds,
            direct_tolerance_centiseconds,
        ):
            continue
        conflict_row_revision = _timing_content_digest(row)
        transcript_revision = _timing_content_digest({
            "current_text": entry_sung_text(entries[index]),
            "current_reading": reading,
            "phonetic_system": "pykakasi-hiragana-v1",
            "probe_mode": "hypothesis-neutral-full-line",
        })
        row_payload = {
            "current_entry_index": index,
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "current_reading_revision": _timing_content_digest(reading),
            "conflict_row_revision": conflict_row_revision,
            "hypothesis_bindings": hypothesis_bindings,
            "hypothesis_set_revision": hypothesis_set_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        row_revision = _timing_content_digest(row_payload)
        request_payload: dict[str, object] = {
            "request_kind": "current-neutral-onset-probe",
            "conflict_entry": entry_number,
            "current_entry_index": index,
            "current_reading": reading,
            "transcript_revision": transcript_revision,
            "row_revision": row_revision,
            "conflict_row_revision": conflict_row_revision,
            "hypothesis_bindings": [
                [candidate_id, generation_revision, centiseconds]
                for candidate_id, generation_revision, centiseconds
                in hypothesis_bindings
            ],
            "hypothesis_set_revision": hypothesis_set_revision,
            "hypothesis_min_centiseconds": hypothesis_centiseconds[0],
            "hypothesis_max_centiseconds": hypothesis_centiseconds[-1],
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        request_payload["request_revision"] = _timing_content_digest(request_payload)
        requests.append(request_payload)
    return tuple(sorted(
        requests,
        key=lambda item: (int(item["conflict_entry"]), str(item["request_revision"])),
    ))


def _postcentral_plan_known_lyric_hubert_timing_failure_diagnostic_requests(
    report: dict[str, object],
    state: FinalTimingState,
    bundles: tuple[EvaluatedBackendBundle, ...],
    entries: list[LyricEntry],
    duration: float,
    *,
    exclude_entries: frozenset[int] = frozenset(),
) -> tuple[dict[str, object], ...]:
    """Plan evidence-only HuBERT probes for known timing-failure rows.

    These requests are deliberately outside the post-Central review authority.
    They collect the acoustic information needed to research a future timestamp
    recovery mechanism without requiring another production replay.  No result
    from this planner is converted into IndependentCurrentOnsetEvidence.
    """
    if (
        len(state.decisions) != len(entries)
        or len(state.audits) != len(entries)
        or not math.isfinite(duration)
        or duration <= 0.0
        or not bundles
    ):
        return ()
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    peer = seed.get("peer_simulation") if isinstance(seed, dict) else None
    raw_rows = peer.get("rows") if isinstance(peer, dict) else None
    if not isinstance(raw_rows, list):
        return ()
    row_lookup = {
        int(row["entry"]): row
        for row in raw_rows
        if isinstance(row, dict) and isinstance(row.get("entry"), int)
    }
    candidate_lookup = _bundle_candidate_lookup(bundles)
    if candidate_lookup is None:
        return ()
    readings = tuple(
        _known_lyric_hiragana_reading(entry_sung_text(entry))
        for entry in entries
    )
    requests: list[dict[str, object]] = []
    for audit in state.audits:
        entry_number = audit.entry_index + 1
        if entry_number in exclude_entries:
            continue
        failure_codes = tuple(
            code for code in audit.failures
            if _timing_failure_requires_review(code)
        )
        if not failure_codes:
            continue
        index = audit.entry_index
        reading = readings[index]
        row = row_lookup.get(entry_number)
        if not reading or not isinstance(row, dict):
            continue
        clusters = row.get("clusters")
        if not isinstance(clusters, list) or len(clusters) < 2:
            continue

        decision = state.decisions[index]
        hypotheses: dict[tuple[str, str], int] = {
            (
                decision.candidate.candidate_id,
                decision.candidate.generation_revision,
            ): decision.written_time.centiseconds
        }
        malformed = False
        for cluster in clusters:
            if not isinstance(cluster, dict):
                malformed = True
                break
            ids = cluster.get("member_candidate_ids")
            revisions = cluster.get("member_generation_revisions")
            if (
                not isinstance(ids, list)
                or not isinstance(revisions, list)
                or not ids
                or len(ids) != len(revisions)
            ):
                malformed = True
                break
            for candidate_id, generation_revision in zip(ids, revisions):
                if not isinstance(candidate_id, str) or not isinstance(generation_revision, str):
                    malformed = True
                    break
                candidate = candidate_lookup.get((candidate_id, generation_revision))
                if candidate is None or candidate.entry_index != index:
                    malformed = True
                    break
                hypotheses[(candidate_id, generation_revision)] = (
                    candidate.written_time.centiseconds
                )
            if malformed:
                break
        if malformed or len(hypotheses) < 2:
            continue

        hypothesis_bindings = tuple(sorted(
            (
                candidate_id,
                generation_revision,
                centiseconds,
            )
            for (candidate_id, generation_revision), centiseconds in hypotheses.items()
        ))
        hypothesis_centiseconds = tuple(
            sorted({item[2] for item in hypothesis_bindings})
        )
        earliest_seconds = hypothesis_centiseconds[0] / 100.0
        next_start = (
            state.decisions[index + 1].written_time.seconds
            if index + 1 < len(entries)
            else duration
        )
        window_start = _hubert_neutral_probe_window_start(
            hypothesis_centiseconds[0]
        )
        window_end = min(
            duration,
            next_start + CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
            earliest_seconds + 12.0,
        )
        if not 0.5 <= window_end - window_start <= 14.0:
            continue
        hypothesis_set_revision = _timing_content_digest({
            "mode": "timing-failure-diagnostic-hypothesis-set-v1",
            "entry_index": index,
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "hypothesis_bindings": [
                [candidate_id, generation_revision, centiseconds]
                for candidate_id, generation_revision, centiseconds
                in hypothesis_bindings
            ],
            "failure_codes": list(failure_codes),
        })
        transcript_revision = _timing_content_digest({
            "current_text": entry_sung_text(entries[index]),
            "current_reading": reading,
            "phonetic_system": "pykakasi-hiragana-v1",
            "probe_mode": "timing-failure-diagnostic-only",
        })
        row_revision = _timing_content_digest({
            "entry_index": index,
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "hypothesis_set_revision": hypothesis_set_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "failure_codes": list(failure_codes),
        })
        request_payload: dict[str, object] = {
            "request_kind": "current-neutral-onset-probe",
            "probe_purpose": "timing-failure-diagnostic-only",
            "authority": False,
            "conflict_entry": entry_number,
            "current_entry_index": index,
            "current_reading": reading,
            "transcript_revision": transcript_revision,
            "row_revision": row_revision,
            "hypothesis_bindings": [
                [candidate_id, generation_revision, centiseconds]
                for candidate_id, generation_revision, centiseconds
                in hypothesis_bindings
            ],
            "hypothesis_set_revision": hypothesis_set_revision,
            "hypothesis_min_centiseconds": hypothesis_centiseconds[0],
            "hypothesis_max_centiseconds": hypothesis_centiseconds[-1],
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "failure_codes": list(failure_codes),
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        request_payload["request_revision"] = _timing_content_digest(request_payload)
        requests.append(request_payload)
    return tuple(sorted(
        requests,
        key=lambda item: (int(item["conflict_entry"]), str(item["request_revision"])),
    ))


def _known_lyric_hubert_char_diagnostic_payload(
    spans_payload: object,
) -> list[dict[str, object]]:
    """Return sanitized helper geometry for report-only failure analysis."""
    if not isinstance(spans_payload, list):
        return []
    output: list[dict[str, object]] = []
    for item in spans_payload:
        if not isinstance(item, dict):
            return []
        try:
            start = float(item["start"])
            end = float(item["end"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            return []
        char = str(item.get("char", ""))
        if (
            not char
            or not all(math.isfinite(value) for value in (start, end, score))
            or end < start
        ):
            return []
        output.append({
            "char": char,
            "start": round(start, 6),
            "end": round(end, 6),
            "score": round(score, 8),
        })
    return output


def _known_lyric_hubert_dummy_candidate(
    entry_index: int,
    spans_payload: object,
    request_revision: str,
) -> TimingCandidate | None:
    """Project HuBERT char spans into the frozen TimingCandidate semantics."""
    if not isinstance(spans_payload, list) or len(spans_payload) < 2:
        return None
    spans: list[TimingTokenSpan] = []
    for item in spans_payload:
        if not isinstance(item, dict):
            return None
        try:
            start = float(item["start"])
            end = float(item["end"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (start, end, score)) or end < start:
            return None
        token = str(item.get("char", ""))
        if not token:
            return None
        spans.append(TimingTokenSpan(start, end, score, token))
    generation_revision = _timing_content_digest({
        "producer": _KNOWN_LYRIC_HUBERT_PRODUCER,
        "request_revision": request_revision,
        "spans": [
            (span.token, round(span.start, 6), round(span.end, 6), round(span.score, 8))
            for span in spans
        ],
    })
    raw_time = spans[0].start
    return TimingCandidate(
        candidate_id=_timing_content_digest({"known_lyric_hubert": generation_revision}),
        entry_index=entry_index,
        source=_KNOWN_LYRIC_HUBERT_PRODUCER,
        raw_time=raw_time,
        written_time=canonicalize_lrc_time(raw_time),
        generation_revision=generation_revision,
        spans=tuple(spans),
        confidence=float(statistics.mean(item.score for item in spans)),
        identity_support="supported",
        sequence_support="supported",
        acoustic_support="supported",
        direct_onset_support="unavailable",
        independent_content_identity=True,
    )


def _known_lyric_hubert_dummy_tail(
    predecessor_index: int,
    spans_payload: object,
    request_revision: str,
    *,
    require_valid: bool = True,
) -> TailRegionEvidence | None:
    """Reuse the frozen reliable-tail geometry policy on HuBERT char spans."""
    dummy = _known_lyric_hubert_dummy_candidate(
        predecessor_index, spans_payload, request_revision
    )
    if dummy is None:
        return None
    tail = derive_reliable_tail(dummy)
    if require_valid and (
        tail.validity != "valid"
        or tail.reliable_tail_start is None
        or tail.reliable_tail_end is None
    ):
        return None
    return tail


def _known_lyric_hubert_opening_evidence(
    current_index: int,
    spans_payload: object,
    request_revision: str,
) -> tuple[TimingCandidate, CandidateRegionEvidence] | None:
    """Return one frozen-policy current-opening proof, or fail closed."""
    dummy = _known_lyric_hubert_dummy_candidate(
        current_index, spans_payload, request_revision
    )
    if dummy is None:
        return None
    regions = analyze_candidate_regions(dummy)
    if not (
        regions.onset_proof_end is not None
        and math.isfinite(regions.onset_proof_end)
        and regions.temporal_coherence == "coherent"
        and regions.fracture_region not in {"opening", "prefix", "unknown"}
    ):
        return None
    return dummy, regions


def _known_lyric_hubert_predecessor_occurrence_status(
    previous_first: float,
    previous_last: float,
    predecessor_centiseconds: int,
    tolerance_centiseconds: int,
) -> str:
    """Bind a pair-tail path to the immutable predecessor *onset*, not its envelope."""
    if not all(math.isfinite(value) for value in (previous_first, previous_last)):
        return "geometry-unavailable"
    first_centiseconds = canonicalize_lrc_time(previous_first).centiseconds
    last_centiseconds = canonicalize_lrc_time(previous_last).centiseconds
    if abs(first_centiseconds - predecessor_centiseconds) > tolerance_centiseconds:
        return "first-token-mismatch"
    if last_centiseconds < predecessor_centiseconds - tolerance_centiseconds:
        return "envelope-mismatch"
    return "valid"
def _known_lyric_hubert_pair_tail_current_evidence_scope(
    predecessor_occurrence_status: str,
) -> str:
    """Select the narrow current-opening evidence allowed for pair-tail binding.

    A predecessor that already matches the joint pair occurrence keeps the V14
    strict final-associated onset gate.  Only a first-token mismatch may use the
    broader hypothesis-neutral boundary observation, because that is the causal
    family this recovery mechanism is intended to audit.
    """
    if predecessor_occurrence_status == "valid":
        return "strict-final-associated-onset"
    if predecessor_occurrence_status == "first-token-mismatch":
        return "hypothesis-neutral-boundary"
    return "reject"


def _known_lyric_hubert_pair_tail_occurrence_status(
    entries: list[LyricEntry],
    previous_index: int,
    current_index: int,
    previous_first: float,
    previous_last: float,
    pair_current_first: float,
    neutral_current_onset: float,
    tolerance_centiseconds: int,
) -> str:
    """Bind pair-tail geometry without depending on the old predecessor onset.

    The adjacent reading pair must be unique in the song, and the pair's current
    boundary must agree with the independently acquired hypothesis-neutral current
    opening.  That combination identifies the occurrence while leaving the
    predecessor selected timestamp free to be wrong.  Timestamp ownership does
    not move to HuBERT: this contract only authorizes predecessor-tail evidence.
    """
    if (
        not 0 <= previous_index < len(entries)
        or current_index != previous_index + 1
        or current_index >= len(entries)
    ):
        return "pair-index-mismatch"
    readings = tuple(
        _known_lyric_hiragana_reading(entry_sung_text(entry))
        for entry in entries
    )
    previous_reading = readings[previous_index]
    current_reading = readings[current_index]
    if not previous_reading or not current_reading:
        return "pair-reading-unavailable"
    pair_count = sum(
        1
        for index in range(1, len(readings))
        if readings[index - 1] == previous_reading
        and readings[index] == current_reading
    )
    if pair_count != 1:
        return "pair-reading-not-unique"
    if not all(math.isfinite(value) for value in (
        previous_first,
        previous_last,
        pair_current_first,
        neutral_current_onset,
    )):
        return "geometry-unavailable"
    if previous_last < previous_first:
        return "predecessor-geometry-invalid"
    pair_current_centiseconds = canonicalize_lrc_time(
        pair_current_first
    ).centiseconds
    neutral_current_centiseconds = canonicalize_lrc_time(
        neutral_current_onset
    ).centiseconds
    if (
        abs(pair_current_centiseconds - neutral_current_centiseconds)
        > tolerance_centiseconds
    ):
        return "current-boundary-mismatch"
    return "valid"


def _postcentral_plan_known_lyric_hubert_predecessor_only_requests(
    boundary_by_entry: Mapping[int, IndependentCurrentOnsetEvidence],
    state: FinalTimingState,
    entries: list[LyricEntry],
    duration: float,
) -> tuple[dict[str, object], ...]:
    """Plan predecessor-only tail acquisition from neutral current boundaries.

    This planner intentionally does not depend on joint-pair helper success or
    predecessor occurrence status.  Occurrence ownership comes from the already
    accepted hypothesis-neutral current boundary plus a song-global unique
    predecessor reading.  The resulting request can only feed the existing tail
    evidence owner.
    """
    if (
        len(state.decisions) != len(entries)
        or not math.isfinite(duration)
        or duration <= 0.0
    ):
        return ()
    readings = tuple(
        _known_lyric_hiragana_reading(entry_sung_text(entry))
        for entry in entries
    )
    reading_counts = collections.Counter(reading for reading in readings if reading)
    requests: list[dict[str, object]] = []
    for current_index in sorted(boundary_by_entry):
        current_boundary = boundary_by_entry[current_index]
        previous_index = current_index - 1
        if not (
            0 <= previous_index < len(entries)
            and 0 <= current_index < len(entries)
            and current_boundary.entry_index == current_index
            and _known_lyric_hubert_current_onset_is_valid(current_boundary)
        ):
            continue
        previous_reading = readings[previous_index]
        if not previous_reading or reading_counts[previous_reading] != 1:
            continue
        previous_start = state.decisions[previous_index].written_time.seconds
        current_start = state.decisions[current_index].written_time.seconds
        before_previous = (
            state.decisions[previous_index - 1].written_time.seconds
            if previous_index >= 1 else 0.0
        )
        spacing = max(HOP_SECONDS, current_start - previous_start)
        lower = before_previous + HOP_SECONDS if previous_index >= 1 else 0.0
        window_start = max(
            0.0,
            lower,
            previous_start - min(2.5, max(1.25, spacing * 0.5)),
        )
        window_end = min(duration, float(current_boundary.onset_time))
        if not 0.5 <= window_end - window_start <= 24.0:
            continue
        previous_text = entry_sung_text(entries[previous_index])
        transcript_revision = _timing_content_digest({
            "previous_text": previous_text,
            "previous_reading": previous_reading,
            "phonetic_system": "pykakasi-hiragana-v1",
            "probe_mode": "predecessor-only-neutral-current-bound-v1",
        })
        row_payload = {
            "previous_entry_index": previous_index,
            "current_entry_index": current_index,
            "previous_text_revision": _timing_content_digest(previous_text),
            "previous_reading_revision": _timing_content_digest(previous_reading),
            "current_boundary_observation_revision": current_boundary.observation_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        row_revision = _timing_content_digest(row_payload)
        request_payload: dict[str, object] = {
            # Reuse the helper's single-line forced-alignment primitive.  The
            # semantic role is bound below by transcript/row provenance and is
            # never mixed with the first-pass current-onset result lookup.
            "request_kind": "current-neutral-onset-probe",
            "evidence_role": "predecessor-only-tail",
            "conflict_entry": current_index + 1,
            "previous_entry_index": previous_index,
            "current_entry_index": current_index,
            "current_reading": previous_reading,
            "transcript_revision": transcript_revision,
            "row_revision": row_revision,
            "current_boundary_observation_revision": current_boundary.observation_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        request_payload["request_revision"] = _timing_content_digest(request_payload)
        requests.append(request_payload)
    return tuple(requests)


def _known_lyric_hubert_full_line_bridge_is_coherent(
    candidate: TimingCandidate,
    regions: CandidateRegionEvidence,
    tail: TailRegionEvidence,
) -> bool:
    """Require one occurrence-bound opening-to-terminal geometric bridge.

    V10 incorrectly equated a locally weak CTC character posterior (<0.05) with
    a whole-line temporal fracture.  The frozen region policy does not make that
    equivalence: token support breaks `_supported_run_metrics`, while a temporal
    fracture is represented independently by a >=3 s severe gap.

    V11 therefore keeps the already-frozen opening proof and terminal-tail proof
    as the two acoustic endpoints and requires that both belong to the exact same
    candidate/revision with no severe geometric fracture between them.  No score,
    gap, or materiality threshold is added or relaxed.
    """
    return bool(
        candidate.spans
        and regions.candidate_id == candidate.candidate_id
        and regions.generation_revision == candidate.generation_revision
        and regions.temporal_coherence == "coherent"
        and regions.fracture_region not in {"opening", "prefix", "unknown"}
        and not regions.severe_gap_indexes
        and tail.validity == "valid"
        and tail.previous_candidate_id == candidate.candidate_id
        and tail.previous_generation_revision == candidate.generation_revision
        and tail.previous_region_revision == regions.region_revision
        and tail.reliable_tail_start is not None
        and tail.reliable_tail_end is not None
    )


def _known_lyric_hubert_current_onset_payload(
    *,
    entry_index: int,
    onset_time: float,
    opening_proof_end: float,
    opening_region_revision: str,
    identity_owner_candidate_id: str,
    identity_owner_generation_revision: str,
    producer: str,
    backend_lineage: str,
    model_identity: str,
    helper_revision: str,
    capability_revision: str,
    audio_revision: str,
    transcript_revision: str,
    row_revision: str,
    request_revision: str,
    hypothesis_set_revision: str,
    occurrence_binding_revision: str,
) -> dict[str, object]:
    return {
        "entry_index": entry_index,
        "onset_time": round(float(onset_time), 6),
        "opening_proof_end": round(float(opening_proof_end), 6),
        "opening_region_revision": opening_region_revision,
        "identity_owner_candidate_id": identity_owner_candidate_id,
        "identity_owner_generation_revision": identity_owner_generation_revision,
        "producer": producer,
        "backend_lineage": backend_lineage,
        "model_identity": model_identity,
        "helper_revision": helper_revision,
        "capability_revision": capability_revision,
        "audio_revision": audio_revision,
        "transcript_revision": transcript_revision,
        "row_revision": row_revision,
        "request_revision": request_revision,
        "hypothesis_set_revision": hypothesis_set_revision,
        "occurrence_binding_revision": occurrence_binding_revision,
    }


def _make_known_lyric_hubert_current_onset(
    *,
    entry_index: int,
    onset_time: float,
    opening_proof_end: float,
    opening_region_revision: str,
    identity_owner_candidate_id: str,
    identity_owner_generation_revision: str,
    model_identity: str,
    helper_revision: str,
    capability_revision: str,
    audio_revision: str,
    transcript_revision: str,
    row_revision: str,
    request_revision: str,
    hypothesis_set_revision: str,
) -> IndependentCurrentOnsetEvidence | None:
    occurrence_binding_revision = _timing_content_digest({
        "mode": "existing-conflict-hypothesis-neutral-window",
        "request_revision": request_revision,
        "transcript_revision": transcript_revision,
        "row_revision": row_revision,
        "hypothesis_set_revision": hypothesis_set_revision,
        "identity_owner_candidate_id": identity_owner_candidate_id,
        "identity_owner_generation_revision": identity_owner_generation_revision,
    })
    payload = _known_lyric_hubert_current_onset_payload(
        entry_index=entry_index,
        onset_time=onset_time,
        opening_proof_end=opening_proof_end,
        opening_region_revision=opening_region_revision,
        identity_owner_candidate_id=identity_owner_candidate_id,
        identity_owner_generation_revision=identity_owner_generation_revision,
        producer=_KNOWN_LYRIC_HUBERT_ONSET_PRODUCER,
        backend_lineage=_KNOWN_LYRIC_HUBERT_LINEAGE,
        model_identity=model_identity,
        helper_revision=helper_revision,
        capability_revision=capability_revision,
        audio_revision=audio_revision,
        transcript_revision=transcript_revision,
        row_revision=row_revision,
        request_revision=request_revision,
        hypothesis_set_revision=hypothesis_set_revision,
        occurrence_binding_revision=occurrence_binding_revision,
    )
    string_fields = (
        payload["opening_region_revision"],
        payload["identity_owner_candidate_id"],
        payload["identity_owner_generation_revision"],
        payload["producer"],
        payload["backend_lineage"],
        payload["model_identity"],
        payload["helper_revision"],
        payload["capability_revision"],
        payload["audio_revision"],
        payload["transcript_revision"],
        payload["row_revision"],
        payload["request_revision"],
        payload["hypothesis_set_revision"],
        payload["occurrence_binding_revision"],
    )
    if (
        entry_index < 0
        or not math.isfinite(onset_time)
        or onset_time < 0.0
        or not math.isfinite(opening_proof_end)
        or opening_proof_end < onset_time
        or any(not isinstance(value, str) or not value.strip() for value in string_fields)
    ):
        return None
    observation_revision = _timing_content_digest(payload)
    return IndependentCurrentOnsetEvidence(
        entry_index=entry_index,
        onset_time=float(onset_time),
        opening_proof_end=float(opening_proof_end),
        opening_region_revision=opening_region_revision,
        identity_owner_candidate_id=identity_owner_candidate_id,
        identity_owner_generation_revision=identity_owner_generation_revision,
        producer=_KNOWN_LYRIC_HUBERT_ONSET_PRODUCER,
        backend_lineage=_KNOWN_LYRIC_HUBERT_LINEAGE,
        model_identity=model_identity,
        helper_revision=helper_revision,
        capability_revision=capability_revision,
        audio_revision=audio_revision,
        transcript_revision=transcript_revision,
        row_revision=row_revision,
        request_revision=request_revision,
        hypothesis_set_revision=hypothesis_set_revision,
        occurrence_binding_revision=occurrence_binding_revision,
        observation_revision=observation_revision,
    )


def _known_lyric_hubert_current_onset_is_valid(
    evidence: IndependentCurrentOnsetEvidence | None,
) -> bool:
    if not isinstance(evidence, IndependentCurrentOnsetEvidence):
        return False
    expected_occurrence_binding_revision = _timing_content_digest({
        "mode": "existing-conflict-hypothesis-neutral-window",
        "request_revision": evidence.request_revision,
        "transcript_revision": evidence.transcript_revision,
        "row_revision": evidence.row_revision,
        "hypothesis_set_revision": evidence.hypothesis_set_revision,
        "identity_owner_candidate_id": evidence.identity_owner_candidate_id,
        "identity_owner_generation_revision": evidence.identity_owner_generation_revision,
    })
    scalar_fields = (
        evidence.opening_region_revision,
        evidence.identity_owner_candidate_id,
        evidence.identity_owner_generation_revision,
        evidence.producer,
        evidence.backend_lineage,
        evidence.model_identity,
        evidence.helper_revision,
        evidence.capability_revision,
        evidence.audio_revision,
        evidence.transcript_revision,
        evidence.row_revision,
        evidence.request_revision,
        evidence.hypothesis_set_revision,
        evidence.occurrence_binding_revision,
        evidence.observation_revision,
    )
    if (
        evidence.producer != _KNOWN_LYRIC_HUBERT_ONSET_PRODUCER
        or evidence.backend_lineage != _KNOWN_LYRIC_HUBERT_LINEAGE
        or any(not isinstance(value, str) or not value.strip() for value in scalar_fields)
        or evidence.occurrence_binding_revision != expected_occurrence_binding_revision
    ):
        return False
    payload = _known_lyric_hubert_current_onset_payload(
        entry_index=evidence.entry_index,
        onset_time=evidence.onset_time,
        opening_proof_end=evidence.opening_proof_end,
        opening_region_revision=evidence.opening_region_revision,
        identity_owner_candidate_id=evidence.identity_owner_candidate_id,
        identity_owner_generation_revision=evidence.identity_owner_generation_revision,
        producer=evidence.producer,
        backend_lineage=evidence.backend_lineage,
        model_identity=evidence.model_identity,
        helper_revision=evidence.helper_revision,
        capability_revision=evidence.capability_revision,
        audio_revision=evidence.audio_revision,
        transcript_revision=evidence.transcript_revision,
        row_revision=evidence.row_revision,
        request_revision=evidence.request_revision,
        hypothesis_set_revision=evidence.hypothesis_set_revision,
        occurrence_binding_revision=evidence.occurrence_binding_revision,
    )
    return bool(
        evidence.entry_index >= 0
        and math.isfinite(evidence.onset_time)
        and evidence.onset_time >= 0.0
        and math.isfinite(evidence.opening_proof_end)
        and evidence.opening_proof_end >= evidence.onset_time
        and evidence.observation_revision == _timing_content_digest(payload)
    )


def _postcentral_acquire_known_lyric_hubert_tails(
    report: dict[str, object],
    state: FinalTimingState,
    entries: list[LyricEntry],
    duration: float,
    audio_path: Path,
    args: argparse.Namespace,
) -> dict[str, object] | None:
    """Acquire one batched third-lineage predecessor-tail observation set.

    V1.2 separates evidence ownership: immutable selected final candidates keep
    lyric identity/sequence ownership, while the hypothesis-neutral HuBERT
    current-line probe can contribute independent direct-onset corroboration
    after the frozen opening proof and exact hypothesis binding. Terminal tail
    and geometric bridge remain diagnostic only. Pair-tail evidence remains
    joint. Rows whose predecessor onset already agrees with the joint path keep
    the V14 final-associated onset gate. Only a predecessor first-token mismatch
    may use a song-unique adjacent reading pair plus the independently localized
    hypothesis-neutral current opening, because that is the timestamp relation
    this recovery family is specifically meant to audit.
    Neither branch can
    admit/select a candidate or mutate timestamps. The parent process decodes
    a deterministic 16 kHz mono PCM WAV; the helper
    uses a pinned HuBERT CTC model plus its own frozen Viterbi trellis.  Process
    diagnostics are serialized so an experimental runtime failure is actionable.
    """
    bundles = report.get(_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY)
    if not (
        isinstance(bundles, tuple) and bundles
        and all(isinstance(item, EvaluatedBackendBundle) for item in bundles)
    ):
        return None
    conflicts = _postcentral_residual_material_conflict_entries(report, state)
    pair_requests = _postcentral_plan_known_lyric_hubert_requests(
        conflicts, state, entries, duration
    )
    current_probe_requests = _postcentral_plan_known_lyric_hubert_current_probe_requests(
        conflicts, report, state, bundles, entries, duration
    )
    recovery_diagnostic_requests = (
        _postcentral_plan_known_lyric_hubert_timing_failure_diagnostic_requests(
            report,
            state,
            bundles,
            entries,
            duration,
            exclude_entries=frozenset(conflicts),
        )
    )
    requests = tuple(sorted(
        (*pair_requests, *current_probe_requests, *recovery_diagnostic_requests),
        key=lambda item: (
            int(item["conflict_entry"]),
            str(item.get("request_kind", "")),
            str(item["request_revision"]),
        ),
    ))
    python_arg = getattr(args, "whisperx_python", None)
    python_exe = (
        Path(python_arg).expanduser().resolve()
        if python_arg else default_whisperx_python()
    )
    try:
        source_audio = audio_path.resolve()
        source_audio_revision = _path_sha256(source_audio)
        python_revision = _stable_file_sha256(python_exe)
    except (OSError, LrcError) as exc:
        return {
            "schema_version": 1,
            "mode": "known-lyric-hubert-tail-current-onset-recovery-v11",
            "authority": "runtime-evidence-only",
            "tail_authority": "unique-predecessor-neutral-current-bound-existing-conflict-replay-only",
            "current_onset_authority": "final-associated-existing-distant-cluster-falsification-only",
            "boundary_observation_authority": "private-central-existing-candidate-recovery-only",
            "lineage": _KNOWN_LYRIC_HUBERT_LINEAGE,
            "status": "ENVIRONMENT_UNAVAILABLE",
            "failure": {"class": type(exc).__name__, "message": str(exc)[:2000]},
            "candidate_admission_count": 0,
            "candidate_sets_mutated": False,
            "central_replayed": False,
            "backend_reselected": False,
            "neutral_replayed": False,
            "timestamp_mutated": False,
        }
    helper_revision = _known_lyric_hubert_helper_revision()
    base_payload: dict[str, object] = {
        "schema_version": 1,
        "mode": "known-lyric-hubert-tail-current-onset-recovery-v11",
        "authority": "runtime-evidence-only",
        "tail_authority": "unique-predecessor-neutral-current-bound-existing-conflict-replay-only",
        "current_onset_authority": "final-associated-existing-distant-cluster-falsification-only",
        "boundary_observation_authority": "private-central-existing-candidate-recovery-only",
        "lineage": _KNOWN_LYRIC_HUBERT_LINEAGE,
        "model_id": _KNOWN_LYRIC_HUBERT_MODEL_ID,
        "model_revision": _KNOWN_LYRIC_HUBERT_MODEL_REVISION,
        "model_blob_sha256": _KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256,
        "helper_protocol_revision": _KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION,
        "helper_revision": helper_revision,
        "source_audio_sha256": source_audio_revision,
        "residual_conflict_entries": list(conflicts),
        "requested_conflict_entries": sorted({
            int(item["conflict_entry"]) for item in requests
        }),
        "pair_request_count": len(pair_requests),
        "current_probe_request_count": len(current_probe_requests),
        "recovery_diagnostic_request_count": len(recovery_diagnostic_requests),
        "request_count": len(requests),
        "candidate_admission_count": 0,
        "candidate_sets_mutated": False,
        "central_replayed": False,
        "backend_reselected": False,
        "neutral_replayed": False,
        "timestamp_mutated": False,
    }
    if not python_exe.is_file() or not requests:
        return {
            **base_payload,
            "valid_tail_count": 0,
            "bundle_changed": False,
            "status": "UNAVAILABLE",
        }
    try:
        alignment_wav = decode_temp_wav(source_audio)
        alignment_audio_revision = _path_sha256(alignment_wav)
    except (OSError, LrcError, subprocess.SubprocessError) as exc:
        return {
            **base_payload,
            "valid_tail_count": 0,
            "bundle_changed": False,
            "status": "AUDIO_PREP_FAILED",
            "failure": {"class": type(exc).__name__, "message": str(exc)[:4000]},
        }
    request_doc = {
        "schema_version": 1,
        "protocol_revision": _KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION,
        "requests": list(requests),
    }
    timeout_seconds = max(300.0, min(1800.0, 180.0 + 40.0 * len(requests)))
    helper_execution: dict[str, object] = {
        "python_sha256": python_revision,
        "alignment_wav_sha256": alignment_audio_revision,
        "timeout_seconds": timeout_seconds,
    }
    payload: object = None
    with tempfile.TemporaryDirectory(prefix="lrc-known-lyric-hubert-") as temp_name:
        temp_dir = Path(temp_name)
        helper_path = temp_dir / "known_lyric_hubert_tail.py"
        request_path = temp_dir / "request.json"
        output_path = temp_dir / "output.json"
        helper_path.write_text(_KNOWN_LYRIC_HUBERT_HELPER_SOURCE, encoding="utf-8")
        request_path.write_text(
            json.dumps(request_doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        command = [
            str(python_exe), str(helper_path),
            "--audio", str(alignment_wav),
            "--request", str(request_path),
            "--output", str(output_path),
            "--device", str(getattr(args, "whisperx_device", "cuda")),
        ]
        try:
            with gpu_compute_phase():
                proc = run_command(command, timeout_seconds=timeout_seconds)
            helper_execution.update({
                "returncode": int(proc.returncode),
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-8000:],
            })
        except subprocess.TimeoutExpired as exc:
            proc = None
            helper_execution.update({
                "exception_class": type(exc).__name__,
                "exception_message": str(exc)[:4000],
                "timed_out": True,
            })
        except (LrcError, OSError, subprocess.SubprocessError) as exc:
            proc = None
            helper_execution.update({
                "exception_class": type(exc).__name__,
                "exception_message": str(exc)[:4000],
                "timed_out": False,
            })
        if proc is not None and proc.returncode == 0 and output_path.is_file():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                helper_execution.update({
                    "output_parse_error_class": type(exc).__name__,
                    "output_parse_error": str(exc)[:4000],
                })
    if not isinstance(payload, dict):
        result = {
            **base_payload,
            "alignment_wav_sha256": alignment_audio_revision,
            "valid_tail_count": 0,
            "bundle_changed": False,
            "status": "HELPER_FAILED",
            "helper_execution": helper_execution,
        }
        result["revision"] = _timing_content_digest(result)
        return result
    metadata_ok = bool(
        payload.get("schema_version") == 1
        and payload.get("protocol_revision") == _KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION
        and isinstance(payload.get("transformers_package_version"), str)
        and bool(payload.get("transformers_package_version"))
        and isinstance(payload.get("huggingface_hub_package_version"), str)
        and bool(payload.get("huggingface_hub_package_version"))
        and isinstance(payload.get("torch_package_version"), str)
        and bool(payload.get("torch_package_version"))
        and payload.get("model_id") == _KNOWN_LYRIC_HUBERT_MODEL_ID
        and payload.get("model_revision") == _KNOWN_LYRIC_HUBERT_MODEL_REVISION
        and payload.get("model_blob_sha256") == _KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256
        and isinstance(payload.get("vocab_revision"), str)
        and bool(payload.get("vocab_revision"))
        and payload.get("language") == _KNOWN_LYRIC_HUBERT_LANGUAGE
        and payload.get("audio_loader") == _KNOWN_LYRIC_HUBERT_AUDIO_LOADER
        and isinstance(payload.get("requested_device"), str)
        and bool(payload.get("requested_device"))
        and isinstance(payload.get("device"), str)
        and bool(payload.get("device"))
        and payload.get("ctc_alignment") == _KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT
    )
    raw_results = payload.get("results")
    if not metadata_ok or not isinstance(raw_results, list):
        raw_results = []
    model_identity = (
        f"{_KNOWN_LYRIC_HUBERT_MODEL_ID}@{_KNOWN_LYRIC_HUBERT_MODEL_REVISION}#"
        f"{_KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256}"
    )
    capability_revision = _timing_content_digest({
        "backend_lineage": _KNOWN_LYRIC_HUBERT_LINEAGE,
        "model_identity": model_identity,
        "helper_revision": helper_revision,
        "helper_protocol_revision": _KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION,
        "python_sha256": python_revision,
        "transformers_package_version": payload.get("transformers_package_version"),
        "huggingface_hub_package_version": payload.get("huggingface_hub_package_version"),
        "torch_package_version": payload.get("torch_package_version"),
        "vocab_revision": payload.get("vocab_revision"),
        "audio_revision": alignment_audio_revision,
        "requested_device": payload.get("requested_device"),
        "device": payload.get("device"),
        "language": payload.get("language"),
        "audio_loader": payload.get("audio_loader"),
        "ctc_alignment": payload.get("ctc_alignment"),
    })
    request_lookup = {str(item["request_revision"]): item for item in requests}
    result_lookup: dict[str, dict[str, object]] = {}
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        revision = str(item.get("request_revision"))
        request = request_lookup.get(revision)
        if (
            request is None
            or item.get("conflict_entry") != request.get("conflict_entry")
            or item.get("request_kind") != request.get("request_kind")
        ):
            continue
        if revision in result_lookup:
            # Duplicate helper identity is fail-closed for the whole producer.
            result_lookup = {}
            break
        result_lookup[revision] = item

    accepted: list[IndependentTailEvidence] = []
    scopes: list[TailEvidenceScope] = []
    boundary_accepted: list[IndependentCurrentOnsetEvidence] = []
    boundary_by_entry: dict[int, IndependentCurrentOnsetEvidence] = {}
    onset_accepted: list[IndependentCurrentOnsetEvidence] = []
    onset_by_entry: dict[int, IndependentCurrentOnsetEvidence] = {}
    diagnostics: list[dict[str, object]] = []
    report.pop(_RUNTIME_POSTCENTRAL_HUBERT_BOUNDARY_OBSERVATION_KEY, None)
    report.pop(_RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY, None)
    peer_tolerance_centiseconds = int(round(
        RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0
    ))
    direct_tolerance_centiseconds = int(round(
        PREFIX_ACOUSTIC_ASSOCIATION_SECONDS * 100.0
    ))

    # First resolve only hypothesis-neutral current-line probes.  A broad joint pair path
    # never owns current onset in V10.
    for request in current_probe_requests:
        request_revision = str(request["request_revision"])
        result = result_lookup.get(request_revision)
        diagnostic: dict[str, object] = {
            "request_kind": "current-neutral-onset-probe",
            "conflict_entry": request["conflict_entry"],
            "request_revision": request_revision,
            "helper_status": result.get("status") if isinstance(result, dict) else "missing",
            "helper_reason": result.get("reason") if isinstance(result, dict) else "missing-helper-result",
            "current_onset_status": "unavailable",
        }
        if not isinstance(result, dict) or result.get("status") != "aligned":
            diagnostics.append(diagnostic)
            continue
        index = int(request["current_entry_index"])
        if not 0 <= index < len(state.decisions):
            diagnostic["current_onset_status"] = "state-entry-invalid"
            diagnostics.append(diagnostic)
            continue
        decision = state.decisions[index]
        if (
            request.get("final_candidate_id") != decision.candidate.candidate_id
            or request.get("final_generation_revision")
            != decision.candidate.generation_revision
            or request.get("final_selected_centiseconds")
            != decision.written_time.centiseconds
        ):
            diagnostic["current_onset_status"] = "final-binding-mismatch"
            diagnostics.append(diagnostic)
            continue
        if not _hubert_probe_window_covers_request_hypotheses(
            request, direct_tolerance_centiseconds
        ):
            diagnostic["current_onset_status"] = "probe-window-coverage-invalid"
            diagnostics.append(diagnostic)
            continue
        current_probe_revision = (
            f"{request_revision}:hypothesis-neutral-current-evidence"
        )
        opening = _known_lyric_hubert_opening_evidence(
            index,
            result.get("current_chars"),
            current_probe_revision,
        )
        if opening is None:
            diagnostic.update({
                "current_onset_status": "opening-validation-rejected",
                "current_char_diagnostics": _known_lyric_hubert_char_diagnostic_payload(
                    result.get("current_chars")
                ),
            })
            diagnostics.append(diagnostic)
            continue
        current_dummy, current_regions = opening

        # V12 keeps identity/sequence ownership with the immutable final
        # candidate.  HuBERT is an independent direct-onset corroborator only.
        # Provisional or identity-unbound finals cannot borrow this authority.
        if not (
            decision.status == "selected_valid"
            and decision.selected_candidate_id == decision.candidate.candidate_id
            and decision.candidate.identity_support == "supported"
            and decision.candidate.sequence_support == "supported"
        ):
            diagnostic.update({
                "current_onset_status": "final-identity-owner-unavailable",
                "current_char_diagnostics": _known_lyric_hubert_char_diagnostic_payload(
                    result.get("current_chars")
                ),
            })
            diagnostics.append(diagnostic)
            continue

        try:
            current_first = float(result.get("current_first_start"))
        except (TypeError, ValueError):
            current_first = math.nan
        if (
            not math.isfinite(current_first)
            or abs(current_dummy.raw_time - current_first) > 1e-6
        ):
            diagnostic["current_onset_status"] = "boundary-unavailable"
            diagnostics.append(diagnostic)
            continue

        observed_centiseconds = canonicalize_lrc_time(current_first).centiseconds
        final_centiseconds = decision.written_time.centiseconds
        delta_centiseconds = observed_centiseconds - final_centiseconds

        # Terminal-tail / geometric-bridge evidence remains diagnostic.  It is
        # useful for explaining the CTC path but does not own current onset.
        # Use one shared synthetic revision so the diagnostic bridge is not
        # self-invalidated by producer-local revision suffixes.
        current_line_tail = _known_lyric_hubert_dummy_tail(
            index,
            result.get("current_chars"),
            current_probe_revision,
            require_valid=False,
        )
        bridge_status = "tail-unavailable"
        if current_line_tail is not None:
            if (
                current_line_tail.validity == "valid"
                and current_line_tail.reliable_tail_start is not None
                and current_line_tail.reliable_tail_end is not None
            ):
                bridge_status = (
                    "coherent"
                    if _known_lyric_hubert_full_line_bridge_is_coherent(
                        current_dummy, current_regions, current_line_tail
                    )
                    else "incoherent"
                )
            else:
                bridge_status = f"tail-{current_line_tail.validity}"

        diagnostic.update({
            "current_first_start": round(current_first, 6),
            "current_opening_proof_end": round(
                float(current_regions.onset_proof_end), 6
            ),
            "current_opening_region_revision": current_regions.region_revision,
            "current_identity_owner_candidate_id": decision.candidate.candidate_id,
            "current_identity_owner_generation_revision": (
                decision.candidate.generation_revision
            ),
            "current_line_tail_validity": (
                current_line_tail.validity
                if current_line_tail is not None else None
            ),
            "current_line_tail_reason": (
                current_line_tail.reason
                if current_line_tail is not None else "tail-input-invalid"
            ),
            "current_line_bridge_status": bridge_status,
            "current_line_tail_start": (
                round(float(current_line_tail.reliable_tail_start), 6)
                if current_line_tail is not None
                and current_line_tail.reliable_tail_start is not None
                else None
            ),
            "current_line_tail_end": (
                round(float(current_line_tail.reliable_tail_end), 6)
                if current_line_tail is not None
                and current_line_tail.reliable_tail_end is not None
                else None
            ),
            "hypothesis_min_centiseconds": request.get(
                "hypothesis_min_centiseconds"
            ),
            "hypothesis_max_centiseconds": request.get(
                "hypothesis_max_centiseconds"
            ),
            "final_selected_centiseconds": final_centiseconds,
            "current_onset_delta_centiseconds": delta_centiseconds,
        })
        boundary_observation = _make_known_lyric_hubert_current_onset(
            entry_index=index,
            onset_time=current_first,
            opening_proof_end=float(current_regions.onset_proof_end),
            opening_region_revision=current_regions.region_revision,
            identity_owner_candidate_id=decision.candidate.candidate_id,
            identity_owner_generation_revision=decision.candidate.generation_revision,
            model_identity=model_identity,
            helper_revision=helper_revision,
            capability_revision=capability_revision,
            audio_revision=alignment_audio_revision,
            transcript_revision=str(request["transcript_revision"]),
            row_revision=str(request["row_revision"]),
            request_revision=request_revision,
            hypothesis_set_revision=str(request["hypothesis_set_revision"]),
        )
        if not _known_lyric_hubert_current_onset_is_valid(boundary_observation):
            diagnostic["current_onset_status"] = "provenance-rejected"
            diagnostics.append(diagnostic)
            continue
        if index in boundary_by_entry:
            diagnostic["current_onset_status"] = "duplicate-entry-observation"
            diagnostics.append(diagnostic)
            continue
        boundary_by_entry[index] = boundary_observation
        boundary_accepted.append(boundary_observation)
        diagnostic.update({
            "boundary_observation_status": "valid-hypothesis-neutral-opening",
            "boundary_observation_revision": boundary_observation.observation_revision,
        })

        # Final-supporting review suppression remains the exact V13 subset:
        # only observations already associated with the immutable selected
        # timestamp within the existing 18 cs materiality bound enter that path.
        if abs(delta_centiseconds) > peer_tolerance_centiseconds:
            diagnostic.update({
                "current_onset_status": "final-selected-mismatch",
                "current_char_diagnostics": _known_lyric_hubert_char_diagnostic_payload(
                    result.get("current_chars")
                ),
            })
            diagnostics.append(diagnostic)
            continue

        onset_by_entry[index] = boundary_observation
        onset_accepted.append(boundary_observation)
        diagnostic.update({
            "current_onset_status": "valid-hypothesis-neutral-direct-onset",
            "current_onset_observation_revision": boundary_observation.observation_revision,
        })
        diagnostics.append(diagnostic)

    # Independent predecessor-tail acquisition is owned by the already accepted
    # neutral current boundary plus a globally unique predecessor reading.
    # Joint pair geometry is diagnostic for its own pair-tail path and cannot
    # suppress this independent acquisition.
    predecessor_only_requests = (
        _postcentral_plan_known_lyric_hubert_predecessor_only_requests(
            boundary_by_entry,
            state,
            entries,
            duration,
        )
    )

    predecessor_only_results: dict[str, dict[str, object]] = {}
    predecessor_only_helper_execution: dict[str, object] | None = None
    if predecessor_only_requests:
        predecessor_request_doc = {
            "schema_version": 1,
            "protocol_revision": _KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION,
            "requests": predecessor_only_requests,
        }
        predecessor_timeout_seconds = max(
            300.0,
            min(1800.0, 180.0 + 40.0 * len(predecessor_only_requests)),
        )
        predecessor_only_helper_execution = {
            "python_sha256": python_revision,
            "alignment_wav_sha256": alignment_audio_revision,
            "timeout_seconds": predecessor_timeout_seconds,
        }
        predecessor_payload: object = None
        with tempfile.TemporaryDirectory(
            prefix="lrc-known-lyric-hubert-predecessor-only-"
        ) as predecessor_temp_name:
            predecessor_temp_dir = Path(predecessor_temp_name)
            predecessor_helper_path = predecessor_temp_dir / "known_lyric_hubert_tail.py"
            predecessor_request_path = predecessor_temp_dir / "request.json"
            predecessor_output_path = predecessor_temp_dir / "output.json"
            predecessor_helper_path.write_text(
                _KNOWN_LYRIC_HUBERT_HELPER_SOURCE, encoding="utf-8"
            )
            predecessor_request_path.write_text(
                json.dumps(predecessor_request_doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            predecessor_command = [
                str(python_exe), str(predecessor_helper_path),
                "--audio", str(alignment_wav),
                "--request", str(predecessor_request_path),
                "--output", str(predecessor_output_path),
                "--device", str(getattr(args, "whisperx_device", "cuda")),
            ]
            try:
                with gpu_compute_phase():
                    predecessor_proc = run_command(
                        predecessor_command,
                        timeout_seconds=predecessor_timeout_seconds,
                    )
                predecessor_only_helper_execution.update({
                    "returncode": int(predecessor_proc.returncode),
                    "stdout_tail": predecessor_proc.stdout[-4000:],
                    "stderr_tail": predecessor_proc.stderr[-8000:],
                })
            except subprocess.TimeoutExpired as exc:
                predecessor_proc = None
                predecessor_only_helper_execution.update({
                    "exception_class": type(exc).__name__,
                    "exception_message": str(exc)[:4000],
                    "timed_out": True,
                })
            except (LrcError, OSError, subprocess.SubprocessError) as exc:
                predecessor_proc = None
                predecessor_only_helper_execution.update({
                    "exception_class": type(exc).__name__,
                    "exception_message": str(exc)[:4000],
                    "timed_out": False,
                })
            if (
                predecessor_proc is not None
                and predecessor_proc.returncode == 0
                and predecessor_output_path.is_file()
            ):
                try:
                    predecessor_payload = json.loads(
                        predecessor_output_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    predecessor_only_helper_execution.update({
                        "output_parse_error_class": type(exc).__name__,
                        "output_parse_error": str(exc)[:4000],
                    })
        if isinstance(predecessor_payload, dict):
            metadata_fields = (
                "schema_version",
                "protocol_revision",
                "transformers_package_version",
                "huggingface_hub_package_version",
                "torch_package_version",
                "model_id",
                "model_revision",
                "model_blob_sha256",
                "vocab_revision",
                "requested_device",
                "device",
                "language",
                "audio_loader",
                "ctc_alignment",
            )
            predecessor_metadata_ok = all(
                predecessor_payload.get(field) == payload.get(field)
                for field in metadata_fields
            )
            raw_predecessor_results = predecessor_payload.get("results")
            if predecessor_metadata_ok and isinstance(raw_predecessor_results, list):
                predecessor_request_lookup = {
                    str(item["request_revision"]): item
                    for item in predecessor_only_requests
                }
                for item in raw_predecessor_results:
                    if not isinstance(item, dict):
                        continue
                    revision = str(item.get("request_revision"))
                    request = predecessor_request_lookup.get(revision)
                    if (
                        request is None
                        or item.get("conflict_entry") != request.get("conflict_entry")
                        or item.get("request_kind") != request.get("request_kind")
                    ):
                        continue
                    if revision in predecessor_only_results:
                        predecessor_only_results = {}
                        break
                    predecessor_only_results[revision] = item

    for request in predecessor_only_requests:
        request_revision = str(request["request_revision"])
        result = predecessor_only_results.get(request_revision)
        diagnostic: dict[str, object] = {
            "request_kind": "predecessor-only-tail",
            "conflict_entry": request["conflict_entry"],
            "request_revision": request_revision,
            "helper_status": result.get("status") if isinstance(result, dict) else "missing",
            "helper_reason": result.get("reason") if isinstance(result, dict) else "missing-helper-result",
            "tail_status": "unavailable",
        }
        if not isinstance(result, dict) or result.get("status") != "aligned":
            diagnostics.append(diagnostic)
            continue
        previous_index = int(request["previous_entry_index"])
        current_index = int(request["current_entry_index"])
        current_boundary = boundary_by_entry.get(current_index)
        if current_boundary is None:
            diagnostic["tail_status"] = "current-boundary-unavailable"
            diagnostics.append(diagnostic)
            continue
        predecessor_chars = result.get("current_chars")
        tail = _known_lyric_hubert_dummy_tail(
            previous_index,
            predecessor_chars,
            request_revision,
            require_valid=False,
        )
        diagnostic["previous_char_diagnostics"] = (
            _known_lyric_hubert_char_diagnostic_payload(predecessor_chars)
        )
        if (
            tail is None
            or tail.validity != "valid"
            or tail.reliable_tail_start is None
            or tail.reliable_tail_end is None
            or tail.confidence is None
            or tail.uncertainty_seconds is None
        ):
            diagnostic.update({
                "tail_status": "derive-reliable-tail-rejected",
                "tail_validity": tail.validity if tail is not None else None,
                "tail_reason": tail.reason if tail is not None else "tail-input-invalid",
                "terminal_span_indexes": (
                    list(tail.terminal_span_indexes) if tail is not None else []
                ),
            })
            diagnostics.append(diagnostic)
            continue
        if tail.reliable_tail_end > current_boundary.onset_time + 0.08:
            diagnostic["tail_status"] = "predecessor-current-boundary-invalid"
            diagnostics.append(diagnostic)
            continue
        provenance = _make_tail_provenance(
            entry_index=previous_index,
            tail_start=tail.reliable_tail_start,
            tail_end=tail.reliable_tail_end,
            uncertainty_seconds=tail.uncertainty_seconds,
            confidence=tail.confidence,
            producer=_KNOWN_LYRIC_HUBERT_PRODUCER,
            backend_lineage=_KNOWN_LYRIC_HUBERT_LINEAGE,
            model_identity=model_identity,
            helper_revision=helper_revision,
            capability_revision=capability_revision,
            audio_revision=alignment_audio_revision,
            transcript_revision=str(request["transcript_revision"]),
            row_revision=str(request["row_revision"]),
        )
        if provenance is None:
            diagnostic["tail_status"] = "provenance-rejected"
            diagnostics.append(diagnostic)
            continue
        evidence = IndependentTailEvidence(
            entry_index=previous_index,
            tail_start=tail.reliable_tail_start,
            tail_end=tail.reliable_tail_end,
            uncertainty_seconds=tail.uncertainty_seconds,
            confidence=tail.confidence,
            producer=_KNOWN_LYRIC_HUBERT_PRODUCER,
            source_revision=provenance.observation_revision,
            provenance=provenance,
        )
        scope = TailEvidenceScope(
            entry_index=evidence.entry_index,
            producer=_KNOWN_LYRIC_HUBERT_PRODUCER,
            backend_lineage=_KNOWN_LYRIC_HUBERT_LINEAGE,
            model_identity=model_identity,
            helper_revision=helper_revision,
            capability_revision=capability_revision,
            audio_revision=alignment_audio_revision,
            transcript_revisions=(str(request["transcript_revision"]),),
            row_revisions=(str(request["row_revision"]),),
        )
        accepted.append(evidence)
        scopes.append(scope)
        diagnostic.update({
            "tail_status": "valid-predecessor-only-neutral-current-bound",
            "predecessor_entry": evidence.entry_index + 1,
            "tail_start": round(evidence.tail_start, 6),
            "tail_end": round(evidence.tail_end, 6),
            "uncertainty_seconds": round(evidence.uncertainty_seconds, 6),
            "confidence": round(evidence.confidence, 8),
            "current_boundary_observation_revision": current_boundary.observation_revision,
            "observation_revision": provenance.observation_revision,
        })
        diagnostics.append(diagnostic)

    # Timing-failure probes are evidence collection only.  Their results are
    # intentionally never converted into runtime onset/tail authority.
    recovery_diagnostics: list[dict[str, object]] = []
    for request in recovery_diagnostic_requests:
        request_revision = str(request["request_revision"])
        result = result_lookup.get(request_revision)
        diagnostic: dict[str, object] = {
            "request_kind": "current-recovery-diagnostic",
            "probe_purpose": "timing-failure-diagnostic-only",
            "authority": False,
            "conflict_entry": request["conflict_entry"],
            "request_revision": request_revision,
            "helper_status": result.get("status") if isinstance(result, dict) else "missing",
            "helper_reason": result.get("reason") if isinstance(result, dict) else "missing-helper-result",
            "recovery_probe_status": "unavailable",
            "failure_codes": list(request.get("failure_codes", [])),
            "hypothesis_min_centiseconds": request.get("hypothesis_min_centiseconds"),
            "hypothesis_max_centiseconds": request.get("hypothesis_max_centiseconds"),
            "final_selected_centiseconds": request.get("final_selected_centiseconds"),
        }
        if not isinstance(result, dict) or result.get("status") != "aligned":
            recovery_diagnostics.append(diagnostic)
            diagnostics.append(copy.deepcopy(diagnostic))
            continue
        index = int(request["current_entry_index"])
        if not 0 <= index < len(state.decisions):
            diagnostic["recovery_probe_status"] = "state-entry-invalid"
            recovery_diagnostics.append(diagnostic)
            diagnostics.append(copy.deepcopy(diagnostic))
            continue
        decision = state.decisions[index]
        if (
            request.get("final_candidate_id") != decision.candidate.candidate_id
            or request.get("final_generation_revision")
            != decision.candidate.generation_revision
            or request.get("final_selected_centiseconds")
            != decision.written_time.centiseconds
        ):
            diagnostic["recovery_probe_status"] = "final-binding-mismatch"
            recovery_diagnostics.append(diagnostic)
            diagnostics.append(copy.deepcopy(diagnostic))
            continue
        char_diagnostics = _known_lyric_hubert_char_diagnostic_payload(
            result.get("current_chars")
        )
        diagnostic["current_char_diagnostics"] = char_diagnostics
        opening = _known_lyric_hubert_opening_evidence(
            index,
            result.get("current_chars"),
            f"{request_revision}:diagnostic-current-opening",
        )
        if opening is None:
            diagnostic["recovery_probe_status"] = "opening-validation-rejected"
            recovery_diagnostics.append(diagnostic)
            diagnostics.append(copy.deepcopy(diagnostic))
            continue
        current_dummy, current_regions = opening
        current_line_tail = _known_lyric_hubert_dummy_tail(
            index,
            result.get("current_chars"),
            f"{request_revision}:diagnostic-current-tail",
            require_valid=False,
        )
        if (
            current_line_tail is None
            or current_line_tail.validity != "valid"
            or current_line_tail.reliable_tail_start is None
            or current_line_tail.reliable_tail_end is None
        ):
            diagnostic.update({
                "recovery_probe_status": "full-line-tail-validation-rejected",
                "current_line_tail_validity": (
                    current_line_tail.validity
                    if current_line_tail is not None else None
                ),
                "current_line_tail_reason": (
                    current_line_tail.reason
                    if current_line_tail is not None else "tail-input-invalid"
                ),
            })
            recovery_diagnostics.append(diagnostic)
            diagnostics.append(copy.deepcopy(diagnostic))
            continue
        if not _known_lyric_hubert_full_line_bridge_is_coherent(
            current_dummy, current_regions, current_line_tail
        ):
            diagnostic.update({
                "recovery_probe_status": "full-line-geometric-bridge-validation-rejected",
                "current_severe_gap_indexes": list(current_regions.severe_gap_indexes),
                "current_fracture_region": current_regions.fracture_region,
            })
            recovery_diagnostics.append(diagnostic)
            diagnostics.append(copy.deepcopy(diagnostic))
            continue
        try:
            current_first = float(result.get("current_first_start"))
        except (TypeError, ValueError):
            current_first = math.nan
        if (
            not math.isfinite(current_first)
            or abs(current_dummy.raw_time - current_first) > 1e-6
        ):
            diagnostic["recovery_probe_status"] = "boundary-unavailable"
            recovery_diagnostics.append(diagnostic)
            diagnostics.append(copy.deepcopy(diagnostic))
            continue
        observed_centiseconds = canonicalize_lrc_time(current_first).centiseconds
        raw_bindings = request.get("hypothesis_bindings")
        distances: list[dict[str, object]] = []
        if isinstance(raw_bindings, list):
            for item in raw_bindings:
                if (
                    isinstance(item, list)
                    and len(item) == 3
                    and isinstance(item[0], str)
                    and isinstance(item[1], str)
                    and isinstance(item[2], int)
                ):
                    distances.append({
                        "candidate_id": item[0],
                        "generation_revision": item[1],
                        "centiseconds": item[2],
                        "delta_centiseconds": observed_centiseconds - item[2],
                    })
        distances.sort(key=lambda item: (
            abs(int(item["delta_centiseconds"])),
            str(item["candidate_id"]),
        ))
        diagnostic.update({
            "recovery_probe_status": "valid-diagnostic-only",
            "current_first_start": round(current_first, 6),
            "observed_centiseconds": observed_centiseconds,
            "current_opening_proof_end": round(
                float(current_regions.onset_proof_end), 6
            ),
            "current_line_tail_start": round(
                float(current_line_tail.reliable_tail_start), 6
            ),
            "current_line_tail_end": round(
                float(current_line_tail.reliable_tail_end), 6
            ),
            "hypothesis_distances": distances,
        })
        recovery_diagnostics.append(diagnostic)
        diagnostics.append(copy.deepcopy(diagnostic))

    # Pair requests remain tail-only.  V1.2 occurrence identity is established
    # by the already-planned song-unique adjacent reading pair plus agreement
    # between the joint pair boundary and the independent hypothesis-neutral
    # current opening.  The predecessor selected onset is diagnostic only.
    for request in pair_requests:
        request_revision = str(request["request_revision"])
        result = result_lookup.get(request_revision)
        diagnostic: dict[str, object] = {
            "request_kind": "pair-tail",
            "conflict_entry": request["conflict_entry"],
            "request_revision": request_revision,
            "helper_status": result.get("status") if isinstance(result, dict) else "missing",
            "helper_reason": result.get("reason") if isinstance(result, dict) else "missing-helper-result",
            "tail_status": "unavailable",
        }
        if not isinstance(result, dict) or result.get("status") != "aligned":
            diagnostics.append(diagnostic)
            continue
        previous_index = int(request["previous_entry_index"])
        current_index = int(request["current_entry_index"])
        if not (
            0 <= previous_index < len(state.decisions)
            and 0 <= current_index < len(state.decisions)
        ):
            diagnostic["tail_status"] = "state-entry-invalid"
            diagnostics.append(diagnostic)
            continue
        try:
            previous_first = float(result.get("previous_first_start"))
            previous_last = float(result.get("previous_last_end"))
            pair_current_first = float(result.get("current_first_start"))
        except (TypeError, ValueError):
            diagnostic["tail_status"] = "pair-geometry-unavailable"
            diagnostics.append(diagnostic)
            continue
        if not all(math.isfinite(value) for value in (
            previous_first, previous_last, pair_current_first
        )):
            diagnostic["tail_status"] = "pair-geometry-unavailable"
            diagnostics.append(diagnostic)
            continue

        predecessor_centiseconds = state.decisions[
            previous_index
        ].written_time.centiseconds
        diagnostic.update({
            "previous_first_start": round(previous_first, 6),
            "previous_last_end": round(previous_last, 6),
            "predecessor_selected_centiseconds": predecessor_centiseconds,
        })
        predecessor_occurrence_status = (
            _known_lyric_hubert_predecessor_occurrence_status(
                previous_first,
                previous_last,
                predecessor_centiseconds,
                direct_tolerance_centiseconds,
            )
        )
        diagnostic["predecessor_selected_onset_relation"] = predecessor_occurrence_status
        current_evidence_scope = _known_lyric_hubert_pair_tail_current_evidence_scope(
            predecessor_occurrence_status
        )
        diagnostic["current_evidence_scope"] = current_evidence_scope
        if current_evidence_scope == "strict-final-associated-onset":
            current_onset = onset_by_entry.get(current_index)
        elif current_evidence_scope == "hypothesis-neutral-boundary":
            current_onset = boundary_by_entry.get(current_index)
        else:
            diagnostic["tail_status"] = "predecessor-occurrence-mismatch"
            diagnostics.append(diagnostic)
            continue
        if current_onset is None:
            diagnostic["tail_status"] = "hypothesis-neutral-current-onset-unavailable"
            diagnostics.append(diagnostic)
            continue
        if not _known_lyric_hubert_current_onset_is_valid(current_onset):
            diagnostic["tail_status"] = "hypothesis-neutral-current-onset-invalid"
            diagnostics.append(diagnostic)
            continue
        if current_evidence_scope == "hypothesis-neutral-boundary":
            pair_occurrence_status = _known_lyric_hubert_pair_tail_occurrence_status(
                entries,
                previous_index,
                current_index,
                previous_first,
                previous_last,
                pair_current_first,
                current_onset.onset_time,
                direct_tolerance_centiseconds,
            )
        else:
            pair_current_centiseconds = canonicalize_lrc_time(
                pair_current_first
            ).centiseconds
            local_current_centiseconds = canonicalize_lrc_time(
                current_onset.onset_time
            ).centiseconds
            pair_occurrence_status = (
                "valid"
                if abs(pair_current_centiseconds - local_current_centiseconds)
                <= direct_tolerance_centiseconds
                else "current-boundary-mismatch"
            )
        diagnostic["pair_occurrence_status"] = pair_occurrence_status
        if pair_occurrence_status == "current-boundary-mismatch":
            diagnostic["tail_status"] = "pair-current-boundary-mismatch"
            diagnostics.append(diagnostic)
            continue
        if pair_occurrence_status != "valid":
            diagnostic["tail_status"] = "pair-occurrence-unbound"
            diagnostics.append(diagnostic)
            continue
        pair_current_centiseconds = canonicalize_lrc_time(
            pair_current_first
        ).centiseconds
        local_current_centiseconds = canonicalize_lrc_time(
            current_onset.onset_time
        ).centiseconds
        diagnostic.update({
            "pair_current_first_start": round(pair_current_first, 6),
            "final_local_current_start": round(current_onset.onset_time, 6),
            "pair_current_delta_centiseconds": (
                pair_current_centiseconds - local_current_centiseconds
            ),
        })
        tail = _known_lyric_hubert_dummy_tail(
            previous_index,
            result.get("previous_chars"),
            request_revision,
            require_valid=False,
        )
        if tail is None:
            diagnostic["tail_status"] = "tail-input-invalid"
            diagnostics.append(diagnostic)
            continue
        previous_chars = result.get("previous_chars")
        if isinstance(previous_chars, list):
            span_scores: list[float] = []
            span_gaps: list[float] = []
            previous_end: float | None = None
            for item in previous_chars:
                if not isinstance(item, dict):
                    continue
                try:
                    item_start = float(item["start"])
                    item_end = float(item["end"])
                    item_score = float(item["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(item_score):
                    span_scores.append(item_score)
                if previous_end is not None and math.isfinite(item_start):
                    span_gaps.append(max(0.0, item_start - previous_end))
                if math.isfinite(item_end):
                    previous_end = item_end
            diagnostic.update({
                "previous_span_count": len(previous_chars),
                "previous_terminal_scores": [
                    round(value, 8) for value in span_scores[-4:]
                ],
                "previous_positive_gap_count": sum(
                    1 for value in span_gaps if value > 0.0
                ),
                "previous_terminal_gaps": [
                    round(value, 6) for value in span_gaps[-4:]
                ],
            })
        if (
            tail.validity != "valid"
            or tail.reliable_tail_start is None
            or tail.reliable_tail_end is None
            or tail.confidence is None
            or tail.uncertainty_seconds is None
        ):
            diagnostic.update({
                "tail_status": "derive-reliable-tail-rejected",
                "tail_validity": tail.validity,
                "tail_reason": tail.reason,
                "terminal_span_indexes": list(tail.terminal_span_indexes),
                "local_gap_references": [
                    round(value, 6) for value in tail.local_gap_references
                ],
            })
            diagnostics.append(diagnostic)
            continue
        if tail.reliable_tail_end > current_onset.onset_time + 0.08:
            diagnostic["tail_status"] = "pair-boundary-invalid"
            diagnostics.append(diagnostic)
            continue
        provenance = _make_tail_provenance(
            entry_index=previous_index,
            tail_start=tail.reliable_tail_start,
            tail_end=tail.reliable_tail_end,
            uncertainty_seconds=tail.uncertainty_seconds,
            confidence=tail.confidence,
            producer=_KNOWN_LYRIC_HUBERT_PRODUCER,
            backend_lineage=_KNOWN_LYRIC_HUBERT_LINEAGE,
            model_identity=model_identity,
            helper_revision=helper_revision,
            capability_revision=capability_revision,
            audio_revision=alignment_audio_revision,
            transcript_revision=str(request["transcript_revision"]),
            row_revision=str(request["row_revision"]),
        )
        if provenance is None:
            diagnostic["tail_status"] = "provenance-rejected"
            diagnostics.append(diagnostic)
            continue
        evidence = IndependentTailEvidence(
            entry_index=previous_index,
            tail_start=tail.reliable_tail_start,
            tail_end=tail.reliable_tail_end,
            uncertainty_seconds=tail.uncertainty_seconds,
            confidence=tail.confidence,
            producer=_KNOWN_LYRIC_HUBERT_PRODUCER,
            source_revision=provenance.observation_revision,
            provenance=provenance,
        )
        scope = TailEvidenceScope(
            entry_index=evidence.entry_index,
            producer=_KNOWN_LYRIC_HUBERT_PRODUCER,
            backend_lineage=_KNOWN_LYRIC_HUBERT_LINEAGE,
            model_identity=model_identity,
            helper_revision=helper_revision,
            capability_revision=capability_revision,
            audio_revision=alignment_audio_revision,
            transcript_revisions=(str(request["transcript_revision"]),),
            row_revisions=(str(request["row_revision"]),),
        )
        accepted.append(evidence)
        scopes.append(scope)
        diagnostic.update({
            "tail_status": "valid-unique-pair-neutral-current-bound",
            "predecessor_entry": evidence.entry_index + 1,
            "tail_start": round(evidence.tail_start, 6),
            "tail_end": round(evidence.tail_end, 6),
            "uncertainty_seconds": round(evidence.uncertainty_seconds, 6),
            "confidence": round(evidence.confidence, 8),
            "observation_revision": provenance.observation_revision,
        })
        diagnostics.append(diagnostic)
    unique_boundary = {
        item.observation_revision: item
        for item in boundary_accepted
        if _known_lyric_hubert_current_onset_is_valid(item)
    }
    boundary_tuple = tuple(
        unique_boundary[key] for key in sorted(unique_boundary)
    )
    if boundary_tuple:
        report[_RUNTIME_POSTCENTRAL_HUBERT_BOUNDARY_OBSERVATION_KEY] = boundary_tuple
    unique_onset = {
        item.observation_revision: item
        for item in onset_accepted
        if _known_lyric_hubert_current_onset_is_valid(item)
    }
    onset_tuple = tuple(unique_onset[key] for key in sorted(unique_onset))
    if onset_tuple:
        report[_RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY] = onset_tuple
    unique_evidence = {
        item.provenance.observation_revision: item
        for item in accepted if item.provenance is not None
    }
    accepted_tuple = tuple(unique_evidence[key] for key in sorted(unique_evidence))
    scope_map = {(item.entry_index, item.producer): item for item in scopes}
    scopes_tuple = tuple(scope_map[key] for key in sorted(scope_map))
    context: TailEvidenceContext | None = None
    if accepted_tuple and scopes_tuple:
        provenances = tuple(
            item.provenance for item in accepted_tuple if item.provenance is not None
        )
        context = TailEvidenceContext(
            backend_lineage=_KNOWN_LYRIC_HUBERT_LINEAGE,
            model_identity=model_identity,
            helper_revision=helper_revision,
            capability_revision=capability_revision,
            audio_revision=alignment_audio_revision,
            transcript_revisions=tuple(sorted({p.transcript_revision for p in provenances})),
            row_revisions=tuple(sorted({p.row_revision for p in provenances})),
            scopes=scopes_tuple,
        )
    bundle_changed = False
    revised_bundles = list(bundles)
    if context is not None and _tail_context_is_valid(context):
        host_index = next(
            (i for i, item in enumerate(revised_bundles) if item.evaluated.backend == "ctc"),
            0,
        )
        revised = _postcentral_bundle_with_additional_tail_observations(
            revised_bundles[host_index], accepted_tuple, context
        )
        bundle_changed = revised.bundle_revision != revised_bundles[host_index].bundle_revision
        revised_bundles[host_index] = revised
    if bundle_changed:
        report[_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY] = tuple(revised_bundles)
    else:
        accepted_tuple = ()
    result_payload: dict[str, object] = {
        **base_payload,
        "alignment_wav_sha256": alignment_audio_revision,
        "helper_execution": helper_execution,
        "helper_requested_device": payload.get("requested_device"),
        "helper_resolved_device": payload.get("device"),
        "helper_result_count": len(raw_results),
        "pair_result_count": sum(
            1 for request in pair_requests
            if str(request["request_revision"]) in result_lookup
        ),
        "current_probe_result_count": sum(
            1 for request in current_probe_requests
            if str(request["request_revision"]) in result_lookup
        ),
        "recovery_diagnostic_result_count": sum(
            1 for request in recovery_diagnostic_requests
            if str(request["request_revision"]) in result_lookup
        ),
        "valid_boundary_observation_count": len(boundary_tuple),
        "boundary_observation_conflict_entries": sorted({
            item.entry_index + 1 for item in boundary_tuple
        }),
        "valid_current_onset_count": len(onset_tuple),
        "current_onset_conflict_entries": sorted({
            item.entry_index + 1 for item in onset_tuple
        }),
        "valid_tail_count": len(accepted_tuple),
        "covered_conflict_entries": sorted({item.entry_index + 2 for item in accepted_tuple}),
        "bundle_changed": bundle_changed,
        "status": "RETRIEVED" if metadata_ok else "INVALID_METADATA",
        "recovery_diagnostics": recovery_diagnostics,
        "diagnostics": diagnostics,
    }
    result_payload["revision"] = _timing_content_digest(result_payload)
    return result_payload

def _postcentral_tail_enriched_bundle(
    base_bundle: EvaluatedBackendBundle,
    report: dict[str, object],
    entries: list[LyricEntry],
    whisperx_evidence_candidates: list[tuple[list[float], dict[str, object], str]],
) -> EvaluatedBackendBundle:
    """Add existing runtime tail observations to a bundle without changing Central input.

    This operates on a deep copy *after* the backend preview has already been
    centrally evaluated.  Only tail_bindings are replaced; candidate_sets,
    report/input/evaluation revisions and content observations remain identical.
    The enriched bundle is runtime-only and cannot participate in candidate
    generation, backend selection, neutral arbitration, or timestamp commit.
    """
    if base_bundle.evaluated.backend != "ctc" or not whisperx_evidence_candidates:
        return base_bundle
    evidence_report = copy.deepcopy(report)
    bind_runtime_whisperx_row_evidence(
        evidence_report, entries, whisperx_evidence_candidates
    )
    # Preserve the existing fallback contract: generate CTC terminal tails only
    # for predecessor rows still lacking a distinct-backend tail observation.
    bind_runtime_ctc_assignment_tail_fallback(evidence_report, entries)
    # Post-Central only: keep a valid CTC terminal observation even when the
    # predecessor row already has a Whisper-family tail.  Final-context lineage
    # filtering decides which observation is actually independent.
    _postcentral_bind_ctc_assignment_tail_coexistence(evidence_report, entries)
    context = _tail_evidence_context_from_report(evidence_report)
    tail_bindings = tuple(
        TailObservationBinding(
            observation=item,
            context=context,
            context_revision=(
                _timing_content_digest(_bundle_tail_context_payload(context))
                if context is not None and _tail_context_is_valid(context)
                else ""
            ),
        )
        for index in sorted(_runtime_tail_observation_map(evidence_report))
        for item in _runtime_tail_observation_map(evidence_report)[index]
    )
    candidate_payload = _bundle_candidate_sets_payload(base_bundle.candidate_sets)
    bundle_payload = {
        "backend": base_bundle.evaluated.backend,
        "candidate_sets": candidate_payload,
        "candidate_sets_revision": base_bundle.candidate_sets_revision,
        "report_projection_revision": base_bundle.report_projection_revision,
        "central_scaffold_json": base_bundle.central_scaffold_json,
        "content_identity_observations": [
            item.__dict__ for item in base_bundle.content_identity_observations
        ],
        "tail_bindings": [_bundle_tail_binding_payload(item) for item in tail_bindings],
        "capability_projection_json": base_bundle.capability_projection_json,
        "source_audio_revision": base_bundle.source_audio_revision,
        "alignment_audio_revision": base_bundle.alignment_audio_revision,
        "algorithm_source_revision": base_bundle.algorithm_source_revision,
        "algorithm_revision": base_bundle.algorithm_revision,
        "input_revision": base_bundle.input_revision,
        "diagnostics_revision": base_bundle.diagnostics_revision,
        "diagnostics_semantic_revision": base_bundle.diagnostics_semantic_revision,
        "evaluation_revision": base_bundle.evaluation_revision,
    }
    enriched = replace(
        base_bundle,
        tail_bindings=tail_bindings,
        bundle_revision=_timing_content_digest(bundle_payload),
    )
    _validate_evaluated_backend_bundle(enriched)
    return enriched


def _hubert_recovery_boundary_observation_map(
    report: dict[str, object],
) -> dict[int, IndependentCurrentOnsetEvidence]:
    raw = report.get(_RUNTIME_POSTCENTRAL_HUBERT_BOUNDARY_OBSERVATION_KEY)
    if not isinstance(raw, tuple):
        return {}
    found: dict[int, IndependentCurrentOnsetEvidence] = {}
    for item in raw:
        if not _known_lyric_hubert_current_onset_is_valid(item):
            continue
        if item.entry_index in found:
            # Ambiguous duplicate producer identity is fail-closed for that row.
            found.pop(item.entry_index, None)
            continue
        found[item.entry_index] = item
    return found


def _hubert_recovery_consensus_candidate(
    *,
    entries: list[LyricEntry],
    entry_index: int,
    parent: TimingCandidate,
    observation: IndependentCurrentOnsetEvidence,
    cluster: dict[str, object],
) -> TimingCandidate | None:
    """Enrich one already-existing CTC/recurrence timestamp with HuBERT onset.

    The timestamp, token spans, lyric identity and sequence all remain owned by
    the existing parent.  HuBERT contributes only an independent known-lyric
    opening boundary.  Central must re-evaluate the derived candidate normally.
    """
    if (
        not _known_lyric_hubert_current_onset_is_valid(observation)
        or parent.entry_index != entry_index
        or parent.identity_support != "supported"
        or parent.sequence_support != "supported"
    ):
        return None
    families = _neutral_arbitration_support_families(parent)
    if "whisper-family" in families:
        return None
    if "ctc" in families:
        source = "ctc-hubert-known-lyric-boundary-consensus"
    elif "recurrence" in families:
        source = "recurrence-hubert-known-lyric-boundary-consensus"
    else:
        return None
    tolerance_cs = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    onset_cs = canonicalize_lrc_time(observation.onset_time).centiseconds
    if abs(parent.written_time.centiseconds - onset_cs) > tolerance_cs:
        return None
    artifact = {
        "mode": _HUBERT_EXISTING_CANDIDATE_RECOVERY_MODE,
        "parent_candidate_id": parent.candidate_id,
        "parent_generation_revision": parent.generation_revision,
        "parent_centiseconds": parent.written_time.centiseconds,
        "cluster_revision": cluster.get("cluster_revision"),
        "cluster_anchor_time": cluster.get("anchor_time"),
        "hubert_observation_revision": observation.observation_revision,
        "hubert_request_revision": observation.request_revision,
        "hubert_hypothesis_set_revision": observation.hypothesis_set_revision,
        "hubert_identity_owner_candidate_id": observation.identity_owner_candidate_id,
        "hubert_identity_owner_generation_revision": (
            observation.identity_owner_generation_revision
        ),
        "hubert_onset_time": round(float(observation.onset_time), 6),
    }
    return make_timing_candidate(
        entry_index=entry_index,
        entry_text=entry_sung_text(entries[entry_index]),
        source=source,
        raw_time=parent.raw_time,
        spans=parent.spans,
        confidence=parent.confidence,
        identity_support=parent.identity_support,
        sequence_support=parent.sequence_support,
        acoustic_support="unavailable",
        direct_onset_support="supported",
        direct_onset_time=float(observation.onset_time),
        direct_onset_source_artifact=artifact,
        direct_onset_evidence_producer=_KNOWN_LYRIC_HUBERT_ONSET_PRODUCER,
        direct_onset_evidence_kind=_HUBERT_RECOVERY_DIRECT_ONSET_KIND,
        direct_onset_evidence_independent=True,
        occurrence_evidence=parent.occurrence_evidence,
        independent_content_identity=False,
        current=False,
        prefix_scope=parent.prefix_scope,
        right_edge_pileup=parent.right_edge_pileup,
        window_truncated=parent.window_truncated,
        source_artifact=artifact,
    )


def _existing_peer_recovery_representative(
    *,
    row: dict[str, object],
    final_decision: TimingDecision,
    previous_decision: TimingDecision | None,
    bundle_lookup: Mapping[tuple[str, str], TimingCandidate],
    occurrence_entries: list[LyricEntry] | None,
) -> tuple[TimingCandidate | None, str]:
    """Return one existing independent peer candidate that Central can prove now.

    This is deliberately narrower than peer review.  The peer projection may
    identify many review-worthy hypotheses, but recovery is eligible only when
    the final-context rebound leaves exactly one distant independent cluster and
    its existing representative also passes the ordinary Central evaluator
    against the selected graph's own predecessor.  Runtime-only post-Central
    tail observations are not promoted into this evaluation.
    """
    if final_decision.status == "selected_valid":
        return None, "selected-boundary-not-provisional"
    if row.get("postcentral_replay_status") != "REBOUND":
        return None, "final-context-peer-rebound-unavailable"
    if (
        row.get("final_selected_candidate_id")
        != final_decision.candidate.candidate_id
        or row.get("final_selected_centiseconds")
        != final_decision.written_time.centiseconds
    ):
        return None, "final-selected-binding-mismatch"

    raw_clusters = row.get("clusters")
    if not isinstance(raw_clusters, list):
        return None, "material-alternative-unavailable"
    clusters = [
        item
        for item in raw_clusters
        if isinstance(item, dict)
        and item.get("relation") == "distant"
        and item.get("review_required") is not False
    ]
    if len(clusters) != 1:
        return None, "material-alternative-not-unique"
    cluster = clusters[0]
    if cluster.get("evidence_state") != "independent":
        return None, "cluster-evidence-not-independent"

    occurrence_state = " ".join(
        str(cluster.get(key) or "")
        for key in (
            "occurrence_state",
            "occurrence_binding_status",
            "occurrence_relation",
        )
    ).lower()
    if "ambig" in occurrence_state:
        return None, "cluster-occurrence-ambiguous"

    selected_families = _neutral_arbitration_support_families(
        final_decision.candidate
    )
    cluster_families = frozenset(
        str(item)
        for item in cluster.get("support_families", [])
        if isinstance(item, str) and item
    )
    if (
        not selected_families
        or not cluster_families
        or selected_families & cluster_families
    ):
        return None, "cluster-not-independent-from-selected-lineage"

    representative_id = cluster.get("representative_candidate_id")
    ids = cluster.get("member_candidate_ids")
    revisions = cluster.get("member_generation_revisions")
    if not (
        isinstance(representative_id, str)
        and representative_id
        and isinstance(ids, list)
        and isinstance(revisions, list)
        and len(ids) == len(revisions)
        and ids
    ):
        return None, "representative-binding-unavailable"

    exact: list[TimingCandidate] = []
    for candidate_id, generation_revision in zip(ids, revisions):
        if candidate_id != representative_id:
            continue
        if not isinstance(generation_revision, str):
            return None, "representative-binding-unavailable"
        candidate = bundle_lookup.get((representative_id, generation_revision))
        if candidate is not None:
            exact.append(candidate)
    if len(exact) != 1:
        return None, "representative-binding-unavailable"
    candidate = exact[0]
    if candidate.entry_index != final_decision.entry_index:
        return None, "representative-entry-mismatch"

    candidate_families = _candidate_support_families(candidate)
    if (
        not candidate_families
        or not (candidate_families & cluster_families)
        or candidate_families & selected_families
    ):
        return None, "representative-lineage-mismatch"
    if (
        candidate.identity_support != "supported"
        or candidate.sequence_support != "supported"
        or not candidate_has_bound_direct_onset_evidence(
            candidate, occurrence_entries
        )
    ):
        return None, "representative-lacks-independent-direct-onset"

    try:
        anchor_cs = canonicalize_lrc_time(float(cluster.get("anchor_time"))).centiseconds
    except (TypeError, ValueError):
        return None, "cluster-anchor-invalid"
    if candidate.written_time.centiseconds != anchor_cs:
        return None, "representative-anchor-mismatch"

    evaluation = evaluate_timing_candidate(
        candidate,
        previous_candidate=(
            previous_decision.candidate if previous_decision is not None else None
        ),
        occurrence_entries=occurrence_entries,
    )
    if (
        not evaluation.valid
        or not evaluation.evidence.minimum_proof_satisfied
        or evaluation.evidence.temporal_coherence != "coherent"
        or evaluation.evidence.ownership.status != "clear"
    ):
        return None, "representative-not-central-valid-in-selected-context"
    return candidate, "eligible-independent-existing-peer-candidate"


def _peer_recovery_project_selection(
    entries: list[LyricEntry],
    report: dict[str, object],
    selection: BackendSelection,
) -> tuple[dict[str, object], FinalTimingState]:
    """Privately replay one peer-recovery selection through final consumers."""
    projected = copy.deepcopy(report)
    projected_state = selection.selected.state
    postcentral_projection = _postcentral_existing_conflict_projection(
        projected, projected_state
    )
    if isinstance(postcentral_projection, dict):
        projected[_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = (
            postcentral_projection
        )
    committed = _apply_evaluated_backend_projection(selection, projected)
    _postcommit_recompute_duplicate_review(
        entries, projected, selection, committed
    )
    assert_report_timestamp_equality(projected, committed)
    update_report_confidence_metrics(projected)
    assert_report_review_contract(projected)
    return projected, committed


def _postcommit_recompute_duplicate_review(
    entries: list[LyricEntry],
    report: dict[str, object],
    selection: BackendSelection,
    committed: FinalTimingState,
) -> tuple[dict[str, object], dict[str, object]]:
    """Refresh timestamp-derived duplicate review after Central has committed.

    Central's immutable evaluation revision is checked before this stage.  The
    duplicate diagnostic is then rebound to the committed timestamps and the
    same committed state is projected once more through review-only consumers.
    """
    duplicate_recompute = _recompute_duplicate_occurrence_offset_diagnostics(
        entries, report, committed
    )
    report[_PEER_DUPLICATE_RECOMPUTE_KEY] = copy.deepcopy(duplicate_recompute)
    duplicate_resolution = _runtime_hubert_duplicate_occurrence_resolution(
        report, committed
    )
    report[_RUNTIME_POSTCENTRAL_HUBERT_DUPLICATE_RESOLUTION_KEY] = (
        duplicate_resolution
    )
    evaluated = selection.selected
    refreshed = commit_final_timing_state(
        committed,
        report,
        thaw_generation_diagnostics(evaluated),
        finding_states=evaluated.finding_states,
    )
    if refreshed is not committed:
        raise LrcError("Postcommit duplicate review changed Central timing state")
    apply_central_summary_to_report(
        report,
        summarize_central_timing_state(report, committed),
    )
    projection_seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    if isinstance(projection_seed, dict):
        report[_BACKEND_PROVENANCE_PROJECTION_KEY] = (
            _finalize_backend_provenance_projection(
                report, projection_seed, selection
            )
        )
        _validate_backend_provenance_projection(report)
    assert_report_review_contract(report)
    return duplicate_recompute, duplicate_resolution


def _peer_recovery_replay_gate(
    entries: list[LyricEntry],
    before_report: dict[str, object],
    after_report: dict[str, object],
    after_selection: BackendSelection,
    target_candidates: Mapping[int, TimingCandidate],
) -> tuple[list[int], list[int], dict[int, tuple[str, ...]]]:
    """Require an existing-peer replay to remain target-local and Central-valid."""
    targets = set(target_candidates)
    before_rows = _neutral_arbitration_final_semantic_rows(before_report)
    after_rows = _neutral_arbitration_final_semantic_rows(after_report)
    if len(before_rows) != len(after_rows):
        raise LrcError("Peer recovery projection changed assignment count")
    collateral: list[int] = []
    failed_targets: list[int] = []
    changed_fields: dict[int, tuple[str, ...]] = {}
    decisions = after_selection.selected.state.decisions
    for index, (before, after) in enumerate(zip(before_rows, after_rows)):
        fields = tuple(
            key
            for key in _NEUTRAL_ARBITRATION_FINAL_SEMANTIC_FIELDS
            if before.get(key) != after.get(key)
        )
        if fields:
            changed_fields[index + 1] = fields
        blocking_fields = tuple(
            key
            for key in fields
            if key in _NEUTRAL_ARBITRATION_HARD_COLLATERAL_FIELDS
        )
        if blocking_fields and index not in targets:
            collateral.append(index + 1)
        if index not in targets:
            continue
        expected = target_candidates[index]
        decision = decisions[index] if index < len(decisions) else None
        candidate = decision.candidate if decision is not None else None
        if (
            decision is None
            or candidate is None
            or decision.status != "selected_valid"
            or decision.selected_candidate_id != expected.candidate_id
            or candidate.candidate_id != expected.candidate_id
            or candidate.generation_revision != expected.generation_revision
            or decision.written_time.centiseconds
            != expected.written_time.centiseconds
            or after.get("timestamp_status") != "selected_valid"
            or after.get("timing_trusted") is not True
            or after.get("canonical_written_centiseconds")
            != expected.written_time.centiseconds
            or candidate.identity_support != "supported"
            or candidate.sequence_support != "supported"
            or not candidate_has_bound_direct_onset_evidence(candidate, entries)
        ):
            failed_targets.append(index + 1)
    return collateral, failed_targets, changed_fields


def _update_projection_seed_for_peer_recovery(
    seed: dict[str, object],
    selection: BackendSelection,
    recovery_payload: dict[str, object],
) -> dict[str, object]:
    rebound = copy.deepcopy(seed)
    rebound["selection"] = central_candidate_selection_payload(selection)
    rows = rebound.get("candidates")
    by_backend = {item.backend: item for item in selection.candidates}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            backend = str(row.get("backend") or "")
            evaluated = by_backend.get(backend)
            if evaluated is None:
                continue
            row["selection_status"] = (
                "selected" if evaluated is selection.selected else "nonselected"
            )
            row["evaluation"] = central_backend_summary_payload(evaluated)
            row["selection_reason"] = central_selection_reason(selection)
    rebound[_PEER_EXISTING_CANDIDATE_RECOVERY_KEY] = copy.deepcopy(
        recovery_payload
    )
    return rebound


def _precommit_independent_peer_existing_candidate_recovery(
    entries: list[LyricEntry],
    report: dict[str, object],
    selection: BackendSelection,
) -> tuple[BackendSelection, dict[str, object]]:
    """Replay one unique independent existing peer hypothesis through Central.

    Only candidates already captured in immutable backend bundles are eligible.
    No timestamp is synthesized, no whole peer graph is unioned into Central,
    and any backend flip or non-target final semantic collateral fails closed.
    """
    base_payload: dict[str, object] = {
        "schema_version": 1,
        "mode": _PEER_EXISTING_CANDIDATE_RECOVERY_MODE,
        "status": "EVALUATING",
        "candidate_source": "immutable-peer-bundle-only",
        "fabricated_timestamp_count": 0,
        "central_replayed": False,
        "backend_reselected": False,
        "trigger_count": 0,
        "trigger_entries": [],
        "applied_trigger_entries": [],
        "rejected_trigger_entries": [],
        "entries": [],
    }
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    bundles = report.get(_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY)
    if not (
        isinstance(seed, dict)
        and isinstance(bundles, tuple)
        and bundles
        and all(isinstance(item, EvaluatedBackendBundle) for item in bundles)
    ):
        base_payload["status"] = "CONTEXT_UNAVAILABLE"
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload

    rebound = _postcentral_existing_conflict_projection(
        report, selection.selected.state
    )
    details = (
        rebound.get("cross_backend_material_conflicts")
        if isinstance(rebound, dict) else None
    )
    rows = details.get("entries") if isinstance(details, dict) else None
    if not isinstance(rows, list):
        base_payload["status"] = "CONFLICT_PROJECTION_UNAVAILABLE"
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload

    lookup = _bundle_candidate_lookup(bundles)
    if lookup is None:
        base_payload["status"] = "CANDIDATE_LOOKUP_UNAVAILABLE"
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload

    selected_eval = selection.selected
    revised_sets = [list(items) for items in selected_eval.candidate_sets]
    diagnostics: list[dict[str, object]] = []
    targets: dict[int, TimingCandidate] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("entry"), int):
            continue
        entry_number = int(row["entry"])
        index = entry_number - 1
        if not 0 <= index < len(selected_eval.state.decisions):
            continue
        decision = selected_eval.state.decisions[index]
        previous = selected_eval.state.decisions[index - 1] if index else None
        candidate, reason = _existing_peer_recovery_representative(
            row=row,
            final_decision=decision,
            previous_decision=previous,
            bundle_lookup=lookup,
            occurrence_entries=entries,
        )
        diagnostic: dict[str, object] = {
            "entry": entry_number,
            "old_selected_candidate_id": decision.candidate.candidate_id,
            "old_selected_centiseconds": decision.written_time.centiseconds,
            "status": "REJECTED",
            "reason": reason,
        }
        if candidate is None:
            diagnostics.append(diagnostic)
            continue
        if any(
            item.candidate_id == candidate.candidate_id
            and item.generation_revision == candidate.generation_revision
            for item in revised_sets[index]
        ):
            diagnostic["reason"] = "candidate-already-present-in-selected-backend"
            diagnostics.append(diagnostic)
            continue
        revised_sets[index].append(candidate)
        targets[index] = candidate
        diagnostic.update({
            "status": "TRIGGER",
            "reason": reason,
            "candidate_id": candidate.candidate_id,
            "generation_revision": candidate.generation_revision,
            "candidate_source": candidate.source,
            "candidate_centiseconds": candidate.written_time.centiseconds,
        })
        diagnostics.append(diagnostic)

    base_payload["trigger_count"] = len(targets)
    base_payload["trigger_entries"] = sorted(index + 1 for index in targets)
    base_payload["entries"] = diagnostics
    if not targets:
        base_payload["status"] = "NO_SAFE_TRIGGER"
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload

    try:
        before_report, _ = _peer_recovery_project_selection(
            entries, report, selection
        )
    except (LrcError, ValueError, TypeError) as exc:
        base_payload.update({
            "status": "BASELINE_PROJECTION_FAILED",
            "failure": {"class": type(exc).__name__, "message": str(exc)[:2000]},
        })
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload

    def evaluate_subset(
        indexes: set[int],
        scope: str,
    ) -> tuple[
        BackendSelection,
        EvaluatedBackendTiming,
        list[int],
        list[int],
        dict[int, tuple[str, ...]],
        bool,
    ]:
        candidate_sets = [list(items) for items in selected_eval.candidate_sets]
        subset_targets: dict[int, TimingCandidate] = {}
        for target_index in indexes:
            candidate = targets[target_index]
            candidate_sets[target_index].append(candidate)
            subset_targets[target_index] = candidate
        generation_diagnostics = thaw_generation_diagnostics(selected_eval)
        generation_diagnostics[_PEER_EXISTING_CANDIDATE_RECOVERY_KEY] = {
            "mode": _PEER_EXISTING_CANDIDATE_RECOVERY_MODE,
            "scope": scope,
            "entries": copy.deepcopy([
                item for item in diagnostics
                if isinstance(item.get("entry"), int)
                and int(item["entry"]) - 1 in indexes
            ]),
        }
        revised_eval = evaluate_backend_timing(
            selected_eval.backend,
            tuple(tuple(items) for items in candidate_sets),
            report,
            generation_diagnostics,
            arbiter_mode=selected_eval.arbiter_mode,
            timing_source_policy=selected_eval.timing_source_policy,
        )
        revised_selection = _neutral_arbitration_selection_with_revised_eval(
            selection, selected_eval, revised_eval
        )
        backend_flip = revised_selection.selected.backend != selected_eval.backend
        if backend_flip:
            return (
                revised_selection, revised_eval, [],
                sorted(index + 1 for index in indexes), {}, True,
            )
        try:
            after_report, _ = _peer_recovery_project_selection(
                entries, report, revised_selection
            )
            collateral, failed, changed = _peer_recovery_replay_gate(
                entries,
                before_report,
                after_report,
                revised_selection,
                subset_targets,
            )
        except (LrcError, ValueError, TypeError):
            collateral = []
            failed = sorted(index + 1 for index in indexes)
            changed = {}
        return (
            revised_selection, revised_eval, collateral, failed, changed, False
        )

    (
        revised_selection,
        revised_eval,
        collateral,
        failed_targets,
        changed_fields,
        backend_flip,
    ) = evaluate_subset(set(targets), "combined")
    initial_replay = {
        "collateral_entries": list(collateral),
        "failed_target_entries": list(failed_targets),
        "backend_flip": backend_flip,
        "candidate_backend": revised_selection.selected.backend,
        "changed_final_semantic_fields": {
            str(entry): list(fields)
            for entry, fields in sorted(changed_fields.items())
        },
    }

    rejected: set[int] = set()
    isolation_rows: list[dict[str, object]] = []
    if collateral or failed_targets or backend_flip:
        safe: set[int] = set()
        for unit in _hubert_recovery_isolation_units(set(targets)):
            (
                unit_selection,
                unit_eval,
                unit_collateral,
                unit_failed,
                unit_changed,
                unit_flip,
            ) = evaluate_subset(set(unit), "dependency-component-isolation")
            unit_safe = not unit_flip and not unit_collateral and not unit_failed
            isolation_rows.append({
                "entries": sorted(index + 1 for index in unit),
                "scope": "dependency-component",
                "status": "SAFE" if unit_safe else "REJECTED",
                "collateral_entries": unit_collateral,
                "failed_target_entries": unit_failed,
                "backend_flip": unit_flip,
                "candidate_backend": unit_selection.selected.backend,
                "evaluation_revision": unit_eval.evaluation_revision,
                "selection_revision": unit_selection.selection_revision,
                "changed_final_semantic_fields": {
                    str(entry): list(fields)
                    for entry, fields in sorted(unit_changed.items())
                },
            })
            if unit_safe:
                safe.update(unit)
                continue
            if len(unit) == 1:
                rejected.update(unit)
                continue
            for index in sorted(unit):
                (
                    single_selection,
                    single_eval,
                    single_collateral,
                    single_failed,
                    single_changed,
                    single_flip,
                ) = evaluate_subset({index}, "single-trigger-isolation")
                single_safe = (
                    not single_flip
                    and not single_collateral
                    and not single_failed
                )
                isolation_rows.append({
                    "entries": [index + 1],
                    "scope": "single-trigger",
                    "status": "SAFE" if single_safe else "REJECTED",
                    "collateral_entries": single_collateral,
                    "failed_target_entries": single_failed,
                    "backend_flip": single_flip,
                    "candidate_backend": single_selection.selected.backend,
                    "evaluation_revision": single_eval.evaluation_revision,
                    "selection_revision": single_selection.selection_revision,
                    "changed_final_semantic_fields": {
                        str(entry): list(fields)
                        for entry, fields in sorted(single_changed.items())
                    },
                })
                if single_safe:
                    safe.add(index)
                else:
                    rejected.add(index)
        if not safe:
            base_payload.update({
                "status": "FAIL_CLOSED",
                "initial_combined_replay": initial_replay,
                "trigger_isolation": isolation_rows,
                "rejected_trigger_entries": sorted(index + 1 for index in targets),
            })
            base_payload["revision"] = _timing_content_digest(base_payload)
            return selection, base_payload
        (
            revised_selection,
            revised_eval,
            collateral,
            failed_targets,
            changed_fields,
            backend_flip,
        ) = evaluate_subset(safe, "safe-subset-consolidation")
        if backend_flip or collateral or failed_targets:
            base_payload.update({
                "status": "FAIL_CLOSED_SAFE_SUBSET",
                "initial_combined_replay": initial_replay,
                "trigger_isolation": isolation_rows,
                "collateral_entries": collateral,
                "failed_target_entries": failed_targets,
                "backend_flip": backend_flip,
                "rejected_trigger_entries": sorted(index + 1 for index in targets),
            })
            base_payload["revision"] = _timing_content_digest(base_payload)
            return selection, base_payload
        targets = {index: targets[index] for index in safe}

    for item in diagnostics:
        entry = item.get("entry")
        if not isinstance(entry, int):
            continue
        index = entry - 1
        if index not in targets:
            if item.get("status") == "TRIGGER":
                item["recovery_disposition"] = "rejected-private-replay"
            continue
        decision = revised_eval.state.decisions[index]
        item.update({
            "recovery_disposition": "applied",
            "new_selected_candidate_id": decision.candidate.candidate_id,
            "new_selected_source": decision.candidate.source,
            "new_selected_centiseconds": decision.written_time.centiseconds,
            "new_timing_trusted": revised_eval.state.audits[index].timing_trusted,
        })

    payload = {
        **base_payload,
        "status": "APPLIED",
        "central_replayed": True,
        "backend_reselected": False,
        "applied_trigger_entries": sorted(index + 1 for index in targets),
        "rejected_trigger_entries": sorted(
            int(item["entry"])
            for item in diagnostics
            if isinstance(item.get("entry"), int)
            and item.get("status") == "TRIGGER"
            and int(item["entry"]) - 1 not in targets
        ),
        "old_evaluation_revision": selected_eval.evaluation_revision,
        "new_evaluation_revision": revised_eval.evaluation_revision,
        "old_selection_revision": selection.selection_revision,
        "new_selection_revision": revised_selection.selection_revision,
        "initial_combined_replay": initial_replay,
        "trigger_isolation": isolation_rows,
        "entries": diagnostics,
    }
    payload["revision"] = _timing_content_digest(payload)
    projection_seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    if isinstance(projection_seed, dict):
        report[_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = (
            _update_projection_seed_for_peer_recovery(
                projection_seed, revised_selection, payload
            )
        )
    return revised_selection, payload


RECOVERY_REVIEW_MOVE_THRESHOLD_CENTISECONDS = 30


def _mark_material_recovery_moves_for_review(
    report: dict[str, object],
) -> list[dict[str, object]]:
    """Mark applied postcentral recovery moves over 0.30 s as REVIEW evidence."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        report["recovery_moved_entries"] = []
        report["recovery_moved_count"] = 0
        return []

    sources = (
        (_HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY, "hubert-existing-candidate"),
        (_PEER_EXISTING_CANDIDATE_RECOVERY_KEY, "peer-existing-candidate"),
    )
    by_entry: dict[int, dict[str, object]] = {}
    for payload_key, source_name in sources:
        payload = report.get(payload_key)
        if not isinstance(payload, dict) or payload.get("status") != "APPLIED":
            continue
        rows = payload.get("entries")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("recovery_disposition") != "applied"
                or not isinstance(row.get("entry"), int)
            ):
                continue
            entry = int(row["entry"])
            try:
                old_cs = int(row.get("old_selected_centiseconds"))
                new_cs = int(row.get("new_selected_centiseconds"))
            except (TypeError, ValueError):
                continue
            delta_cs = new_cs - old_cs
            if abs(delta_cs) <= RECOVERY_REVIEW_MOVE_THRESHOLD_CENTISECONDS:
                continue
            record = by_entry.setdefault(
                entry,
                {
                    "entry": entry,
                    "recovery_sources": [],
                    "moves": [],
                },
            )
            recovery_sources = record["recovery_sources"]
            moves = record["moves"]
            assert isinstance(recovery_sources, list)
            assert isinstance(moves, list)
            if source_name not in recovery_sources:
                recovery_sources.append(source_name)
            moves.append({
                "source": source_name,
                "old_selected_centiseconds": old_cs,
                "new_selected_centiseconds": new_cs,
                "delta_centiseconds": delta_cs,
            })

    marked: list[dict[str, object]] = []
    for entry in sorted(by_entry):
        index = entry - 1
        if not 0 <= index < len(assignments):
            continue
        assignment = assignments[index]
        if not isinstance(assignment, dict):
            continue
        record = by_entry[entry]
        moves = record["moves"]
        assert isinstance(moves, list) and moves
        largest = max(
            moves,
            key=lambda item: abs(int(item["delta_centiseconds"])),
        )
        record["old_selected_centiseconds"] = int(
            largest["old_selected_centiseconds"]
        )
        record["new_selected_centiseconds"] = int(
            largest["new_selected_centiseconds"]
        )
        record["delta_centiseconds"] = int(largest["delta_centiseconds"])
        record["threshold_centiseconds"] = (
            RECOVERY_REVIEW_MOVE_THRESHOLD_CENTISECONDS
        )
        flags = assignment.get("flags")
        assignment["flags"] = list(dict.fromkeys([
            *(flags if isinstance(flags, list) else []),
            "recovery-moved",
        ]))
        assignment["recovery_moved"] = copy.deepcopy(record)
        marked.append(copy.deepcopy(record))

    report["recovery_moved_entries"] = marked
    report["recovery_moved_count"] = len(marked)
    return marked


def _hubert_recovery_project_selection(
    report: dict[str, object],
    selection: BackendSelection,
) -> tuple[dict[str, object], FinalTimingState]:
    """Privately execute the complete post-HuBERT production consumer path."""
    projected = copy.deepcopy(report)
    projected_state = selection.selected.state
    postcentral_projection = _postcentral_existing_conflict_projection(
        projected, projected_state
    )
    if isinstance(postcentral_projection, dict):
        projected[_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = (
            postcentral_projection
        )
    duplicate_resolution = _runtime_hubert_duplicate_occurrence_resolution(
        projected, projected_state
    )
    projected[_RUNTIME_POSTCENTRAL_HUBERT_DUPLICATE_RESOLUTION_KEY] = (
        duplicate_resolution
    )
    committed = _apply_evaluated_backend_projection(selection, projected)
    assert_report_timestamp_equality(projected, committed)
    update_report_confidence_metrics(projected)
    assert_report_review_contract(projected)
    return projected, committed


def _hubert_recovery_replay_gate(
    before_report: dict[str, object],
    after_report: dict[str, object],
    after_selection: BackendSelection,
    target_indexes: frozenset[int] | set[int],
    observations: Mapping[int, IndependentCurrentOnsetEvidence],
    parent_centiseconds: Mapping[int, frozenset[int]],
) -> tuple[list[int], list[int], dict[int, tuple[str, ...]]]:
    """Require recovery plus all downstream consumers to be target-local and safe."""
    targets = set(target_indexes)
    before_rows = _neutral_arbitration_final_semantic_rows(before_report)
    after_rows = _neutral_arbitration_final_semantic_rows(after_report)
    if len(before_rows) != len(after_rows):
        raise LrcError("HuBERT recovery projection changed assignment count")
    tolerance_cs = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    collateral: list[int] = []
    failed_targets: list[int] = []
    changed_fields: dict[int, tuple[str, ...]] = {}
    decisions = after_selection.selected.state.decisions
    for index, (before, after) in enumerate(zip(before_rows, after_rows)):
        fields = tuple(
            key
            for key in _NEUTRAL_ARBITRATION_FINAL_SEMANTIC_FIELDS
            if before.get(key) != after.get(key)
        )
        if fields:
            changed_fields[index + 1] = fields
        blocking_fields = tuple(
            key for key in fields
            if key in _NEUTRAL_ARBITRATION_HARD_COLLATERAL_FIELDS
        )
        if blocking_fields and index not in targets:
            collateral.append(index + 1)
        if index not in targets:
            continue
        observation = observations.get(index)
        after_cs = after.get("canonical_written_centiseconds")
        decision = decisions[index] if index < len(decisions) else None
        candidate = decision.candidate if decision is not None else None
        direct = candidate.direct_onset_evidence if candidate is not None else None
        families = (
            _neutral_arbitration_support_families(candidate)
            if candidate is not None else frozenset()
        )
        allowed_cs = parent_centiseconds.get(index, frozenset())
        if (
            observation is None
            or not _known_lyric_hubert_current_onset_is_valid(observation)
            or after.get("timestamp_status") != "selected_valid"
            or after.get("timing_trusted") is not True
            or after.get("review_required") is not False
            or not isinstance(after_cs, int)
            or abs(
                after_cs
                - canonicalize_lrc_time(float(observation.onset_time)).centiseconds
            ) > tolerance_cs
            or after_cs not in allowed_cs
            or candidate is None
            or direct is None
            or direct.kind != _HUBERT_RECOVERY_DIRECT_ONSET_KIND
            or direct.producer != _KNOWN_LYRIC_HUBERT_ONSET_PRODUCER
            or not direct.independent
            or "hubert-ja-known-lyric" not in families
            or not ({"ctc", "recurrence"} & set(families))
            or candidate.identity_support != "supported"
            or candidate.sequence_support != "supported"
        ):
            failed_targets.append(index + 1)
    return collateral, failed_targets, changed_fields


def _update_projection_seed_for_hubert_recovery(
    seed: dict[str, object],
    selection: BackendSelection,
    recovery_payload: dict[str, object],
) -> dict[str, object]:
    rebound = copy.deepcopy(seed)
    rebound["selection"] = central_candidate_selection_payload(selection)
    rows = rebound.get("candidates")
    by_backend = {item.backend: item for item in selection.candidates}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            backend = str(row.get("backend") or "")
            evaluated = by_backend.get(backend)
            if evaluated is None:
                continue
            row["selection_status"] = (
                "selected" if evaluated is selection.selected else "nonselected"
            )
            row["evaluation"] = central_backend_summary_payload(evaluated)
            row["selection_reason"] = central_selection_reason(selection)
    rebound[_HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY] = copy.deepcopy(
        recovery_payload
    )
    return rebound


def _hubert_recovery_isolation_units(
    targeted_indexes: set[int],
) -> tuple[frozenset[int], ...]:
    """Keep adjacent targets together because Central ownership is predecessor-local."""
    if not targeted_indexes:
        return ()
    ordered = sorted(targeted_indexes)
    units: list[frozenset[int]] = []
    current: list[int] = [ordered[0]]
    for index in ordered[1:]:
        if index == current[-1] + 1:
            current.append(index)
        else:
            units.append(frozenset(current))
            current = [index]
    units.append(frozenset(current))
    return tuple(units)


def _precommit_hubert_existing_candidate_recovery(
    entries: list[LyricEntry],
    report: dict[str, object],
    selection: BackendSelection,
) -> tuple[BackendSelection, dict[str, object]]:
    """Use HuBERT only to complete proof for an existing alternative timestamp.

    No HuBERT time is admitted as a timestamp.  One existing CTC/recurrence
    parent may receive the independently bound HuBERT onset, after which the
    ordinary Central evaluator/selector and the complete downstream consumer
    path decide whether the old selected boundary can be replaced safely.
    """
    observations = _hubert_recovery_boundary_observation_map(report)
    base_payload: dict[str, object] = {
        "schema_version": 1,
        "mode": _HUBERT_EXISTING_CANDIDATE_RECOVERY_MODE,
        "status": "NO_OBSERVATION" if not observations else "EVALUATING",
        "boundary_observation_count": len(observations),
        "boundary_observation_entries": sorted(index + 1 for index in observations),
        "central_replayed": False,
        "backend_reselected": False,
        "fabricated_timestamp_count": 0,
        "candidate_timestamp_source": "existing-parent-only",
        "trigger_count": 0,
        "trigger_entries": [],
        "applied_trigger_entries": [],
        "rejected_trigger_entries": [],
        "entries": [],
    }
    if not observations:
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload

    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    bundles = report.get(_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY)
    if not (
        isinstance(seed, dict)
        and isinstance(bundles, tuple)
        and bundles
        and all(isinstance(item, EvaluatedBackendBundle) for item in bundles)
    ):
        base_payload["status"] = "CONTEXT_UNAVAILABLE"
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload

    selected_eval = selection.selected
    rebound = _postcentral_rebind_existing_peer_conflicts(
        seed, selected_eval.state, bundles, current_onsets=()
    )
    conflict_details = rebound.get("cross_backend_material_conflicts")
    rows = (
        conflict_details.get("entries")
        if isinstance(conflict_details, dict) else None
    )
    if not isinstance(rows, list):
        base_payload["status"] = "CONFLICT_PROJECTION_UNAVAILABLE"
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload

    bundle_lookup = _bundle_candidate_lookup(bundles)
    if bundle_lookup is None:
        base_payload["status"] = "CANDIDATE_LOOKUP_UNAVAILABLE"
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload
    candidate_map = _neutral_arbitration_candidate_map(selection)
    revised_sets = [list(items) for items in selected_eval.candidate_sets]
    selected_tolerance_cs = int(round(PREFIX_ACOUSTIC_ASSOCIATION_SECONDS * 100.0))
    match_tolerance_cs = int(round(RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS * 100.0))
    diagnostics: list[dict[str, object]] = []
    targeted_indexes: set[int] = set()
    parent_centiseconds: dict[int, frozenset[int]] = {}

    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("entry"), int):
            continue
        entry_number = int(row["entry"])
        index = entry_number - 1
        observation = observations.get(index)
        if observation is None or not 0 <= index < len(selected_eval.state.decisions):
            continue
        decision = selected_eval.state.decisions[index]
        diagnostic: dict[str, object] = {
            "entry": entry_number,
            "observation_revision": observation.observation_revision,
            "old_selected_candidate_id": decision.candidate.candidate_id,
            "old_selected_centiseconds": decision.written_time.centiseconds,
            "hubert_onset_centiseconds": canonicalize_lrc_time(
                observation.onset_time
            ).centiseconds,
            "status": "REJECTED",
        }
        if not _postcentral_hubert_observation_matches_row_hypotheses(
            row,
            lookup=bundle_lookup,
            final_decision=decision,
            observation=observation,
        ):
            diagnostic["reason"] = "hypothesis-provenance-mismatch"
            diagnostics.append(diagnostic)
            continue
        selected_families = _neutral_arbitration_support_families(
            decision.candidate
        )
        if not (
            decision.status == "selected_valid"
            and decision.selected_candidate_id == decision.candidate.candidate_id
            and decision.candidate.identity_support == "supported"
            and decision.candidate.sequence_support == "supported"
            and "whisper-family" in selected_families
            and "ctc" not in selected_families
            and "recurrence" not in selected_families
            and "hubert-ja-known-lyric" not in selected_families
        ):
            diagnostic["reason"] = "selected-boundary-not-whisper-only-valid"
            diagnostics.append(diagnostic)
            continue
        onset_cs = canonicalize_lrc_time(observation.onset_time).centiseconds
        selected_cs = decision.written_time.centiseconds
        if abs(onset_cs - selected_cs) <= selected_tolerance_cs:
            diagnostic["reason"] = "hubert-not-disjoint-from-selected-boundary"
            diagnostics.append(diagnostic)
            continue

        raw_clusters = row.get("clusters")
        if not isinstance(raw_clusters, list) or not raw_clusters:
            diagnostic["reason"] = "material-cluster-unavailable"
            diagnostics.append(diagnostic)
            continue
        matched: list[tuple[dict[str, object], list[TimingCandidate]]] = []
        malformed = False
        for cluster in raw_clusters:
            if not isinstance(cluster, dict):
                malformed = True
                break
            cluster_families = {
                str(item) for item in cluster.get("support_families", [])
                if isinstance(item, str)
            }
            if (
                "whisper-family" in cluster_families
                or not ({"ctc", "recurrence"} & cluster_families)
            ):
                continue
            ids = cluster.get("member_candidate_ids")
            revisions = cluster.get("member_generation_revisions")
            if (
                not isinstance(ids, list)
                or not isinstance(revisions, list)
                or len(ids) != len(revisions)
                or not ids
            ):
                malformed = True
                break
            parents: list[TimingCandidate] = []
            for candidate_id, generation_revision in zip(ids, revisions):
                if not isinstance(candidate_id, str) or not isinstance(
                    generation_revision, str
                ):
                    malformed = True
                    break
                candidate = bundle_lookup.get((candidate_id, generation_revision))
                if candidate is None or candidate.entry_index != index:
                    malformed = True
                    break
                selectable_parent = candidate_map.get(candidate.candidate_id)
                if (
                    selectable_parent is None
                    or selectable_parent.generation_revision
                    != candidate.generation_revision
                ):
                    continue
                families = _neutral_arbitration_support_families(candidate)
                if (
                    "whisper-family" in families
                    or not ({"ctc", "recurrence"} & set(families))
                    or candidate.identity_support != "supported"
                    or candidate.sequence_support != "supported"
                    or abs(candidate.written_time.centiseconds - onset_cs)
                    > match_tolerance_cs
                ):
                    continue
                parents.append(selectable_parent)
            if malformed:
                break
            if parents:
                matched.append((cluster, parents))
        if malformed:
            diagnostic["reason"] = "material-cluster-malformed"
            diagnostics.append(diagnostic)
            continue
        if len(matched) != 1:
            diagnostic["reason"] = (
                "no-unique-hubert-matched-material-cluster"
                if not matched
                else "multiple-hubert-matched-material-clusters"
            )
            diagnostics.append(diagnostic)
            continue
        cluster, parents = matched[0]

        # If Central already has a valid non-Whisper witness at the selected
        # boundary, HuBERT+alternative is a disagreement, not proof completion.
        valid_candidate_ids = _neutral_arbitration_valid_candidate_ids(
            selection, index
        )
        selected_side_independent = []
        for candidate in candidate_map.values():
            if (
                candidate.entry_index == index
                and candidate.candidate_id in valid_candidate_ids
                and abs(
                    candidate.written_time.centiseconds - selected_cs
                ) <= match_tolerance_cs
            ):
                families = _neutral_arbitration_support_families(candidate)
                if (
                    "whisper-family" not in families
                    and bool({"ctc", "recurrence"} & set(families))
                    and candidate.identity_support == "supported"
                    and candidate.sequence_support == "supported"
                ):
                    selected_side_independent.append(candidate)
        if selected_side_independent:
            diagnostic["reason"] = "selected-boundary-has-independent-nonwhisper-witness"
            diagnostics.append(diagnostic)
            continue

        consensus: list[TimingCandidate] = []
        for parent in sorted(
            {item.candidate_id: item for item in parents}.values(),
            key=lambda item: item.candidate_id,
        ):
            candidate = _hubert_recovery_consensus_candidate(
                entries=entries,
                entry_index=index,
                parent=parent,
                observation=observation,
                cluster=cluster,
            )
            if candidate is not None:
                consensus.append(candidate)
        if not consensus:
            diagnostic["reason"] = "consensus-candidate-unavailable"
            diagnostics.append(diagnostic)
            continue

        falsified = [
            candidate for candidate in revised_sets[index]
            if abs(candidate.written_time.centiseconds - selected_cs)
            <= match_tolerance_cs
            and "whisper-family" in _neutral_arbitration_support_families(candidate)
            and "ctc" not in _neutral_arbitration_support_families(candidate)
            and "recurrence" not in _neutral_arbitration_support_families(candidate)
            and "hubert-ja-known-lyric"
            not in _neutral_arbitration_support_families(candidate)
        ]
        falsified_ids = {item.candidate_id for item in falsified}
        if decision.candidate.candidate_id not in falsified_ids:
            diagnostic["reason"] = "selected-whisper-candidate-not-falsifiable"
            diagnostics.append(diagnostic)
            continue
        survivors = [
            item for item in revised_sets[index]
            if item.candidate_id not in falsified_ids
        ]
        merged = {
            item.candidate_id: item for item in [*survivors, *consensus]
        }
        revised_sets[index] = [merged[key] for key in sorted(merged)]
        targeted_indexes.add(index)
        parent_centiseconds[index] = frozenset(
            item.written_time.centiseconds for item in parents
        )
        diagnostic.update({
            "status": "TRIGGER",
            "reason": "existing-material-cluster-hubert-proof-completion",
            "cluster_revision": cluster.get("cluster_revision"),
            "cluster_anchor_time": cluster.get("anchor_time"),
            "parent_candidate_ids": sorted(item.candidate_id for item in parents),
            "parent_centiseconds": sorted(parent_centiseconds[index]),
            "admitted_consensus_candidate_ids": sorted(
                item.candidate_id for item in consensus
            ),
            "falsified_selected_candidate_ids": sorted(falsified_ids),
        })
        diagnostics.append(diagnostic)

    base_payload["trigger_count"] = len(targeted_indexes)
    base_payload["trigger_entries"] = sorted(index + 1 for index in targeted_indexes)
    base_payload["entries"] = diagnostics
    if not targeted_indexes:
        base_payload["status"] = "NO_SAFE_TRIGGER"
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload

    try:
        before_report, _before_state = _hubert_recovery_project_selection(
            report, selection
        )
    except (LrcError, ValueError, TypeError) as exc:
        base_payload.update({
            "status": "BASELINE_PROJECTION_FAILED",
            "failure": {"class": type(exc).__name__, "message": str(exc)[:2000]},
        })
        base_payload["revision"] = _timing_content_digest(base_payload)
        return selection, base_payload

    def evaluate_subset(
        indexes: set[int],
        scope: str,
    ) -> tuple[
        BackendSelection,
        EvaluatedBackendTiming,
        list[int],
        list[int],
        dict[int, tuple[str, ...]],
        bool,
    ]:
        candidate_sets = [list(items) for items in selected_eval.candidate_sets]
        for target_index in indexes:
            candidate_sets[target_index] = list(revised_sets[target_index])
        generation_diagnostics = thaw_generation_diagnostics(selected_eval)
        generation_diagnostics[_HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY] = {
            "mode": _HUBERT_EXISTING_CANDIDATE_RECOVERY_MODE,
            "scope": scope,
            "entries": copy.deepcopy([
                item for item in diagnostics
                if isinstance(item.get("entry"), int)
                and int(item["entry"]) - 1 in indexes
            ]),
        }
        revised_eval = evaluate_backend_timing(
            selected_eval.backend,
            tuple(tuple(items) for items in candidate_sets),
            report,
            generation_diagnostics,
            arbiter_mode=selected_eval.arbiter_mode,
            timing_source_policy=selected_eval.timing_source_policy,
        )
        revised_selection = _neutral_arbitration_selection_with_revised_eval(
            selection, selected_eval, revised_eval
        )
        backend_flip = revised_selection.selected.backend != selected_eval.backend
        if backend_flip:
            return (
                revised_selection,
                revised_eval,
                [],
                sorted(index + 1 for index in indexes),
                {},
                True,
            )
        try:
            after_report, _after_state = _hubert_recovery_project_selection(
                report, revised_selection
            )
            collateral, failed, changed = _hubert_recovery_replay_gate(
                before_report,
                after_report,
                revised_selection,
                indexes,
                observations,
                parent_centiseconds,
            )
        except (LrcError, ValueError, TypeError):
            collateral = []
            failed = sorted(index + 1 for index in indexes)
            changed = {}
        return (
            revised_selection,
            revised_eval,
            collateral,
            failed,
            changed,
            False,
        )

    (
        revised_selection,
        revised_eval,
        collateral,
        failed_targets,
        initial_changed_fields,
        initial_backend_flip,
    ) = evaluate_subset(set(targeted_indexes), "combined")
    initial_combined_replay = {
        "collateral_entries": list(collateral),
        "failed_target_entries": list(failed_targets),
        "backend_flip": initial_backend_flip,
        "candidate_backend": revised_selection.selected.backend,
        "changed_final_semantic_fields": {
            str(entry): list(fields)
            for entry, fields in sorted(initial_changed_fields.items())
        },
    }

    rejected_target_indexes: set[int] = set()
    isolation_rows: list[dict[str, object]] = []
    if collateral or failed_targets or initial_backend_flip:
        independently_safe: set[int] = set()
        units = _hubert_recovery_isolation_units(targeted_indexes)
        for unit in units:
            (
                unit_selection,
                unit_eval,
                unit_collateral,
                unit_failed,
                unit_changed,
                unit_flip,
            ) = evaluate_subset(set(unit), "dependency-component-isolation")
            unit_safe = not unit_flip and not unit_collateral and not unit_failed
            isolation_rows.append({
                "entries": sorted(index + 1 for index in unit),
                "scope": "dependency-component",
                "status": "SAFE" if unit_safe else "REJECTED",
                "collateral_entries": unit_collateral,
                "failed_target_entries": unit_failed,
                "backend_flip": unit_flip,
                "candidate_backend": unit_selection.selected.backend,
                "evaluation_revision": unit_eval.evaluation_revision,
                "selection_revision": unit_selection.selection_revision,
                "changed_final_semantic_fields": {
                    str(entry): list(fields)
                    for entry, fields in sorted(unit_changed.items())
                },
            })
            if unit_safe:
                independently_safe.update(unit)
                continue
            if len(unit) == 1:
                rejected_target_indexes.update(unit)
                continue
            for index in sorted(unit):
                (
                    single_selection,
                    single_eval,
                    single_collateral,
                    single_failed,
                    single_changed,
                    single_flip,
                ) = evaluate_subset({index}, "single-trigger-isolation")
                single_safe = (
                    not single_flip
                    and not single_collateral
                    and not single_failed
                )
                isolation_rows.append({
                    "entries": [index + 1],
                    "scope": "single-trigger",
                    "status": "SAFE" if single_safe else "REJECTED",
                    "collateral_entries": single_collateral,
                    "failed_target_entries": single_failed,
                    "backend_flip": single_flip,
                    "candidate_backend": single_selection.selected.backend,
                    "evaluation_revision": single_eval.evaluation_revision,
                    "selection_revision": single_selection.selection_revision,
                    "changed_final_semantic_fields": {
                        str(entry): list(fields)
                        for entry, fields in sorted(single_changed.items())
                    },
                })
                if single_safe:
                    independently_safe.add(index)
                else:
                    rejected_target_indexes.add(index)

        if not independently_safe:
            base_payload.update({
                "status": "FAIL_CLOSED",
                "initial_combined_replay": initial_combined_replay,
                "trigger_isolation": isolation_rows,
                "rejected_trigger_entries": sorted(
                    index + 1 for index in targeted_indexes
                ),
            })
            base_payload["revision"] = _timing_content_digest(base_payload)
            return selection, base_payload

        (
            revised_selection,
            revised_eval,
            collateral,
            failed_targets,
            consolidated_changed_fields,
            consolidated_backend_flip,
        ) = evaluate_subset(independently_safe, "safe-subset-consolidation")
        if consolidated_backend_flip or collateral or failed_targets:
            base_payload.update({
                "status": "FAIL_CLOSED_SAFE_SUBSET",
                "initial_combined_replay": initial_combined_replay,
                "trigger_isolation": isolation_rows,
                "collateral_entries": collateral,
                "failed_target_entries": failed_targets,
                "backend_flip": consolidated_backend_flip,
                "changed_final_semantic_fields": {
                    str(entry): list(fields)
                    for entry, fields in sorted(
                        consolidated_changed_fields.items()
                    )
                },
                "rejected_trigger_entries": sorted(
                    index + 1 for index in targeted_indexes
                ),
            })
            base_payload["revision"] = _timing_content_digest(base_payload)
            return selection, base_payload
        targeted_indexes = set(independently_safe)
    else:
        rejected_target_indexes = set()

    for item in diagnostics:
        entry = item.get("entry")
        if not isinstance(entry, int):
            continue
        index = entry - 1
        if index not in targeted_indexes:
            if item.get("status") == "TRIGGER":
                item["recovery_disposition"] = "rejected-private-replay"
            continue
        decision = revised_eval.state.decisions[index]
        item.update({
            "recovery_disposition": "applied",
            "new_selected_candidate_id": decision.candidate.candidate_id,
            "new_selected_source": decision.candidate.source,
            "new_selected_centiseconds": decision.written_time.centiseconds,
            "new_timing_trusted": revised_eval.state.audits[index].timing_trusted,
        })

    applied_diagnostics = [
        item for item in diagnostics
        if isinstance(item.get("entry"), int)
        and int(item["entry"]) - 1 in targeted_indexes
    ]
    payload = {
        **base_payload,
        "status": "APPLIED",
        "central_replayed": True,
        "backend_reselected": False,
        "fabricated_timestamp_count": 0,
        "enriched_candidate_count": sum(
            len(item.get("admitted_consensus_candidate_ids", []))
            for item in applied_diagnostics
        ),
        "falsified_selected_candidate_count": sum(
            len(item.get("falsified_selected_candidate_ids", []))
            for item in applied_diagnostics
        ),
        "applied_trigger_entries": sorted(
            index + 1 for index in targeted_indexes
        ),
        "rejected_trigger_entries": sorted(
            int(item["entry"])
            for item in diagnostics
            if isinstance(item.get("entry"), int)
            and item.get("status") == "TRIGGER"
            and int(item["entry"]) - 1 not in targeted_indexes
        ),
        "old_evaluation_revision": selected_eval.evaluation_revision,
        "new_evaluation_revision": revised_eval.evaluation_revision,
        "old_selection_revision": selection.selection_revision,
        "new_selection_revision": revised_selection.selection_revision,
        "initial_combined_replay": initial_combined_replay,
        "trigger_isolation": isolation_rows,
        "entries": diagnostics,
    }
    payload["revision"] = _timing_content_digest(payload)
    projection_seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    if isinstance(projection_seed, dict):
        report[_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = (
            _update_projection_seed_for_hubert_recovery(
                projection_seed, revised_selection, payload
            )
        )
    return revised_selection, payload


def _postcentral_existing_conflict_projection(
    report: dict[str, object],
    state: FinalTimingState,
) -> dict[str, object] | None:
    """Replay existing peer conflicts only, after neutral selection is frozen."""
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    bundles = report.get(_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY)
    if not isinstance(seed, dict):
        return None
    if not (
        isinstance(bundles, tuple)
        and bundles
        and all(isinstance(item, EvaluatedBackendBundle) for item in bundles)
    ):
        return None
    current_onsets = report.get(_RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY)
    if not isinstance(current_onsets, tuple):
        current_onsets = ()
    rebound = _postcentral_rebind_existing_peer_conflicts(
        seed, state, bundles, current_onsets=current_onsets
    )
    details = rebound.get("cross_backend_material_conflicts")
    if not isinstance(details, dict):
        return None
    # Structural safety: this stage may only remove old peer rows/candidates.
    old_details = seed.get("cross_backend_material_conflicts")
    if not isinstance(old_details, dict):
        return None
    old_entries = {
        int(item["entry"])
        for item in old_details.get("entries", [])
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    new_entries = {
        int(item["entry"])
        for item in details.get("entries", [])
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    if not new_entries.issubset(old_entries):
        return None

    def cluster_member_ids(row: object) -> set[str]:
        if not isinstance(row, dict):
            return set()
        output: set[str] = set()
        clusters = row.get("clusters")
        if not isinstance(clusters, list):
            return output
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            ids = cluster.get("member_candidate_ids")
            if isinstance(ids, list):
                output.update(item for item in ids if isinstance(item, str))
        return output

    old_rows = {
        int(item["entry"]): item
        for item in old_details.get("entries", [])
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    for row in details.get("entries", []):
        if not isinstance(row, dict) or not isinstance(row.get("entry"), int):
            return None
        old_row = old_rows.get(int(row["entry"]))
        if old_row is None or not cluster_member_ids(row).issubset(
            cluster_member_ids(old_row)
        ):
            return None
    return rebound

def _build_backend_provenance_projection_seed(
    audio_path: Path,
    args: argparse.Namespace,
    generation_candidates: list[dict[str, object]],
    preview_records: list[dict[str, object]],
    selection: BackendSelection,
    whisperx_evidence_candidates: list[tuple[list[float], dict[str, object], str]] | None = None,
) -> dict[str, object]:
    # Preview records are the last in-memory point where each backend report,
    # Central scaffold, and runtime sidecar coexist.  Capture immutable bundles
    # here before winner selection discards nonselected report objects.
    for record in preview_records:
        evaluated = record.get("evaluated")
        report = record.get("report")
        if (
            isinstance(evaluated, EvaluatedBackendTiming)
            and isinstance(report, dict)
            and not isinstance(record.get("bundle"), EvaluatedBackendBundle)
        ):
            record["bundle"] = _capture_evaluated_backend_bundle(
                evaluated,
                report,
                audio_path=audio_path,
                args=args,
            )
    bundles = tuple(
        record["bundle"]
        for record in preview_records
        if isinstance(record.get("bundle"), EvaluatedBackendBundle)
    )
    peer_simulation = _simulate_cross_backend_peer_graph(
        bundles,
        selected_backend=selection.selected.backend,
        timing_source_policy=selection.selected.timing_source_policy,
    )
    evaluated_reports = {
        id(record.get("evaluated")): record.get("report")
        for record in preview_records
        if record.get("evaluated") is not None
    }
    candidates: list[dict[str, object]] = []
    for evaluated in selection.candidates:
        report = evaluated_reports.get(id(evaluated))
        summary = _public_backend_capability_summary(
            audio_path, args, evaluated.backend, report if isinstance(report, dict) else None
        )
        summary.update({
            "usable": True,
            "error": False,
            "selection_status": "selected" if evaluated is selection.selected else "nonselected",
            "evaluation": central_backend_summary_payload(evaluated),
            "selection_reason": central_selection_reason(selection),
        })
        candidates.append(summary)

    generation: list[dict[str, object]] = []
    for candidate in generation_candidates:
        backend = str(candidate.get("backend") or "unknown")
        error = candidate.get("error")
        report = candidate.get("report")
        capability_report = report if isinstance(report, dict) else None
        if (
            backend == "whisperx"
            and error
            and whisperx_evidence_candidates
            and isinstance(whisperx_evidence_candidates[0][1], dict)
        ):
            capability_report = whisperx_evidence_candidates[0][1]
        generation.append({
            "backend": backend,
            "usable": not bool(error),
            "error": bool(error),
            "error_code": _backend_projection_error_code(error),
            "capability": _public_backend_capability_summary(
                audio_path, args, backend, capability_report
            ),
            "evidence_candidate_count": (
                len(whisperx_evidence_candidates)
                if backend == "whisperx" and whisperx_evidence_candidates
                else 0
            ),
        })

    try:
        source_audio_sha256 = _path_sha256(audio_path)
    except (OSError, LrcError):
        source_audio_sha256 = None
    local_summary = _public_local_capability_summary(audio_path, args)
    cross_backend_material_conflicts = _peer_review_only_conflict_projection(
        peer_simulation
    )
    return {
        "schema_version": _BACKEND_PROVENANCE_PROJECTION_SCHEMA_VERSION,
        "source_audio": {
            "sha256": _backend_projection_field(source_audio_sha256),
            "source": "input-audio",
        },
        "candidates": candidates,
        "generation_candidates": generation,
        "selection": central_candidate_selection_payload(selection),
        "producer_capabilities": [local_summary] if local_summary is not None else [],
        "algorithm_source_sha256": validated_loaded_source_sha256(),
        "algorithm_revision": LRC_ALGORITHM_REVISION,
        "peer_simulation": peer_simulation,
        "cross_backend_material_conflicts": cross_backend_material_conflicts,
    }


def _finalize_backend_provenance_projection(
    report: dict[str, object],
    seed: dict[str, object],
    selection: BackendSelection,
) -> dict[str, object]:
    projection = copy.deepcopy(seed)
    selected_capability_revision = None
    for candidate in projection.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        evaluation = candidate.get("evaluation")
        if isinstance(evaluation, dict) and evaluation.get("evaluation_revision") == selection.selected.evaluation_revision:
            capability = candidate.get("capability")
            if isinstance(capability, dict):
                field = capability.get("capability_revision")
                if isinstance(field, dict):
                    selected_capability_revision = field.get("value")
            break

    trusted_rows: list[dict[str, object]] = []
    assignments = report.get("assignments")
    central_state = report.get("central_timing_state")
    if (
        not isinstance(assignments, list)
        or not isinstance(central_state, list)
        or len(assignments) != len(central_state)
    ):
        raise LrcError("Backend provenance row count does not match assignments")
    if isinstance(assignments, list) and isinstance(central_state, list):
        for assignment, state in zip(assignments, central_state, strict=True):
            if not isinstance(assignment, dict) or not isinstance(state, dict):
                continue
            if not bool(assignment.get("overall_trusted")):
                continue
            provenance = assignment.get("candidate_provenance")
            evidence = assignment.get("normalized_timing_evidence")
            direct = provenance.get("direct_onset_evidence") if isinstance(provenance, dict) else None
            acoustic = provenance.get("acoustic_evidence") if isinstance(provenance, dict) else None
            source_revision = None
            if isinstance(direct, dict):
                source_revision = direct.get("source_revision")
            if source_revision is None and isinstance(acoustic, dict):
                source_revision = acoustic.get("source_revision")
            trusted_rows.append({
                "entry": assignment.get("entry"),
                "candidate_id": state.get("candidate_id"),
                "candidate_source": state.get("candidate_source"),
                "candidate_generation_revision": (
                    provenance.get("generation_revision") if isinstance(provenance, dict) else None
                ),
                "candidate_evidence_revision": (
                    evidence.get("candidate_revision") if isinstance(evidence, dict) else None
                ),
                "timing_evidence_revision": assignment.get("timing_evidence_revision"),
                "timestamp_revision": assignment.get("timestamp_revision"),
                "source_revision": source_revision,
                "occurrence_evidence": (
                    provenance.get("occurrence_evidence")
                    if isinstance(provenance, dict) else None
                ),
                "selected_backend_capability_revision": selected_capability_revision,
            })

    projection["selected_backend"] = selection.selected.backend
    projection["selected_backend_capability_revision"] = selected_capability_revision
    # The central evaluator owns this immutable report projection revision.
    # Later public metric refreshes may rebuild diagnostic booleans, so the
    # binding must remain the evaluator revision rather than a post-commit
    # re-projection of the mutable report dictionary.
    projection["report_revision_binding"] = selection.selected.report_projection_revision
    projection["central_state_binding"] = _timing_content_digest(
        report.get("central_timing_state", [])
    )
    projection["selection_binding"] = _timing_content_digest(
        report.get("candidate_selection", {})
    )
    projection["trusted_row_evidence"] = trusted_rows
    projection["projection_status"] = "RETRIEVED"
    projection.pop("projection_revision", None)
    projection["projection_revision"] = _timing_content_digest(projection)
    return projection


def validate_backend_selection(selection: BackendSelection) -> None:
    """Verify that selection provenance is the canonical immutable central result."""
    if not selection.candidates:
        raise LrcError("Backend selection has no centrally evaluated candidates")
    if not any(item is selection.selected for item in selection.candidates):
        raise LrcError("Selected backend is not an identity member of its candidate set")
    canonical = select_evaluated_backend(selection.candidates)
    if (
        len(canonical.candidates) != len(selection.candidates)
        or any(
            expected is not actual
            for expected, actual in zip(canonical.candidates, selection.candidates)
        )
        or canonical.selected is not selection.selected
        or canonical.selection_revision != selection.selection_revision
    ):
        raise LrcError("Backend selection provenance is not the canonical central selection")


def _apply_evaluated_backend_projection(
    selection: BackendSelection,
    report: dict[str, object],
) -> FinalTimingState:
    validate_backend_selection(selection)
    evaluated = selection.selected
    expected_selection_payload = central_candidate_selection_payload(selection)
    existing_selection = report.get("candidate_selection")
    if existing_selection is not None and existing_selection != expected_selection_payload:
        raise LrcError("Candidate selection provenance changed after central selection")
    current_report_revision = timing_report_evaluation_revision(report)
    if current_report_revision != evaluated.report_projection_revision:
        raise LrcError("Report evaluation input changed after central preview")
    diagnostics = thaw_generation_diagnostics(evaluated)
    expected_input_revision = _timing_content_digest({
        "backend": evaluated.backend,
        "report_projection_revision": current_report_revision,
        "generation_diagnostics_semantic_revision": (
            evaluated.generation_diagnostics_semantic_revision
        ),
        "candidate_sets": [
            [candidate.candidate_id for candidate in candidates]
            for candidates in evaluated.candidate_sets
        ],
        **(
            {"arbiter_mode": evaluated.arbiter_mode}
            if evaluated.arbiter_mode != "off"
            else {}
        ),
    })
    if expected_input_revision != evaluated.input_revision:
        raise LrcError("Backend timing input changed after central preview")
    projection_seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    if isinstance(projection_seed, dict):
        review_projection = projection_seed.get("cross_backend_material_conflicts")
        if not (
            isinstance(review_projection, dict)
            and review_projection.get("postcentral_existing_conflict_evidence_replayed") is True
        ):
            projection_seed = _rebind_peer_review_only_conflicts_to_final_selection(
                projection_seed, evaluated.state
            )
        report[_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = projection_seed
    _install_peer_review_only_conflicts(
        report,
        projection_seed if isinstance(projection_seed, dict) else None,
    )
    committed = commit_final_timing_state(
        evaluated.state,
        report,
        diagnostics,
        finding_states=evaluated.finding_states,
    )
    if committed is not evaluated.state:
        raise LrcError("Central timing commit did not reuse the selected preview state")
    report["candidate_selection"] = expected_selection_payload
    apply_central_summary_to_report(
        report,
        summarize_central_timing_state(report, committed),
    )
    # The evaluated summary may predate final suspicious-row REVIEW changes.
    # Sync only aggregate review fields here: private replays must preserve
    # their immutable Central row timing decisions until the final-output audit.
    committed_assignments = report.get("assignments", [])
    final_trusted_count = sum(
        row.get("verification_status") == "trusted"
        for row in committed_assignments if isinstance(row, dict)
    )
    final_unverified_count = sum(
        row.get("verification_status") == "unverified"
        for row in committed_assignments if isinstance(row, dict)
    )
    total = len(committed_assignments)
    report["unverified_count"] = final_unverified_count
    report["unverified_percent"] = ratio_percent(final_unverified_count, total)
    report["low_confidence_count"] = total - final_trusted_count
    report["low_confidence_percent"] = ratio_percent(total - final_trusted_count, total)
    for key in ("trusted", "overall_trusted", "authority_trusted", "matched"):
        report[key + "_entries"] = final_trusted_count
        report[key + "_percent"] = ratio_percent(final_trusted_count, total)
    finalize_review_summary(report)
    if isinstance(projection_seed, dict):
        report[_BACKEND_PROVENANCE_PROJECTION_KEY] = _finalize_backend_provenance_projection(
            report,
            projection_seed,
            selection,
        )
        _validate_backend_provenance_projection(report)
    assert_report_review_contract(report)
    return committed


def commit_evaluated_backend(
    selection: BackendSelection,
    report: dict[str, object],
) -> FinalTimingState:
    """Install one already-evaluated Central selection into the production report."""
    return _apply_evaluated_backend_projection(selection, report)


from lrc_review_tsv import format_top_review, write_review_tsv
from lrc_manual_edits import archive_manual_edits, manual_edit_reason
from lrc_publication import publish_pair, recover_pair
import lrc_stage_timings

def _publish_lrc_report(
    output_path: Path,
    report_path: Path,
    write_lrc_stage: Callable[[Path], object],
    report: dict[str, object],
) -> None:
    publish_pair(
        output_path,
        report_path,
        write_lrc_stage,
        lambda stage: write_alignment_report(stage, report),
        on_lrc_staged=lambda digest: report.__setitem__("published_lrc_sha256", digest),
    )


def report_artifact_paths(audio_path: Path, output_path: Path, report_dir: Path) -> tuple[Path, Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    identity = f"{audio_path.resolve()}\0{output_path.resolve()}".casefold().encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:10]
    stem = f"{audio_path.stem}--{digest}"
    return (
        report_dir / f"{stem}.align-report.json",
        report_dir / f"{stem}.review-audit.md",
        report_dir / f"{stem}.anchor-template.lrc",
    )


def process_audio(audio_path: Path, args: argparse.Namespace) -> Path:
    progress_log(f"processing: {audio_path.name}")
    if not audio_path.exists():
        raise LrcError(f"Audio file not found: {audio_path}")
    if not runtime_paths.is_audio_file(audio_path):
        raise LrcError(
            f"Only {', '.join(runtime_paths.AUDIO_EXTENSIONS)} audio is supported: {audio_path}"
        )

    progress_log("resolving lyric source")
    requested_timing_source = str(args.timing_source)
    lyrics_path = (
        Path(args.lyrics)
        if args.lyrics
        else find_lyrics(
            audio_path,
            include_checked=requested_timing_source == "checked",
        )
    )
    progress_log(f"lyrics source: {lyrics_path}")
    lyrics = load_lyrics(lyrics_path)
    entries = remove_generated_title_cards(lyrics.entries, audio_path)
    preflight = build_lyric_preflight(lyrics_path, lyrics, entries, audio_path)
    if args.probe:
        identity = str(audio_path.resolve()).casefold().encode("utf-8")
        suffix = hashlib.sha256(identity).hexdigest()[:10]
        output_path = DEFAULT_PROBE_DIR / f"{audio_path.stem}--{suffix}.probe.lrc"
    else:
        output_path = Path(args.output).expanduser().resolve() if args.output else audio_path.with_suffix(".lrc")
    recovery_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else DEFAULT_REPORT_DIR
    recovery_report, _, _ = report_artifact_paths(audio_path, output_path, recovery_dir)
    recover_pair(output_path, recovery_report)
    # Refuse an accidental overwrite before model inference.  Previously this
    # check lived in write_lrc(), after CTC/ASR had already consumed minutes of
    # GPU time for a probe whose destination could not be written.
    if output_path.exists() and not args.overwrite:
        raise LrcError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    edit_reason = manual_edit_reason(output_path, recovery_report)
    if edit_reason and not getattr(args, "discard_manual_edits", False):
        raise LrcError(
            f"Manual review edits detected for {output_path} ({edit_reason}); refusing to overwrite. "
            "Re-run with --discard-manual-edits to replace them (the edited LRC is archived first)."
        )
    if edit_reason:
        for archived in archive_manual_edits(output_path, recovery_report):
            progress_log(f"archived manual review artifact: {archived}")
    requested_whisper_language = args.whisper_language
    if args.whisper_language == "auto":
        args = copy.copy(args)
        args.whisper_language = infer_spoken_language(entries)
    anchor_path = find_anchor_hints(audio_path, args)
    duration = probe_duration(audio_path)
    timing_source = "checked_lrc" if requested_timing_source == "checked" else requested_timing_source
    if timing_source == "audio":
        timing_source = "heuristic"
        print(
            "WARN: --timing-source audio is now treated as experimental heuristic timing.",
            file=sys.stderr,
        )

    if requested_timing_source == "auto":
        # Independent auto inference uses ordered lyric text only. Embedded
        # timestamps, checked references, and discovered anchors are not input
        # evidence, regardless of compatibility negative flags.
        entries = strip_lyric_timestamps(entries)
        lyrics = LyricDocument(
            entries=entries,
            metadata_lines=list(lyrics.metadata_lines),
            saw_timestamps=False,
        )
        if timing_source == "auto" and default_ctc_ready() and default_whisperx_ready():
            timing_source = "backend_competition"
            progress_log("auto source: running CTC + WhisperX candidate selection")
        elif timing_source == "auto" and (default_ctc_ready() or default_whisperx_ready()):
            timing_source = "backend_competition"
            progress_log("auto source: running available backend candidate selection")
        elif timing_source == "auto":
            raise LrcError(
                "No local audio alignment backend is configured yet. "
                "Use --timing-source lyrics with a checked timestamped lyric template, "
                "install/configure whisper.cpp + WhisperX, or explicitly use "
                "--timing-source heuristic for rough experimental timing."
            )

    skipped_markers: list[dict[str, object]] = []
    timing_selection: BackendSelection | None = None
    if timing_source not in ("lyrics", "checked_lrc"):
        entries, skipped_markers = remove_instrumental_markers(entries)
        if skipped_markers:
            progress_log(f"skipped non-lyric markers: count={len(skipped_markers)}")
            preflight["transformations"].extend(
                {"kind": "instrumental-marker-removed", **item}
                for item in skipped_markers
            )
        if not entries:
            raise LrcError("No lyric text entries remain after skipping instrumental markers")

    if requested_timing_source == "checked":
        checked_hint = checked_lrc_timing_hint(audio_path, entries, lyrics_path)
        if checked_hint:
            entries, timestamps, report = checked_hint
        elif lyrics.saw_timestamps:
            timestamps = source_timestamps(entries)
            report = {
                "backend": "checked-lrc",
                "timing_provenance": (
                    "human-reviewed-bk" if lyrics_path.parent.name.casefold() == "lrc bk"
                    else "explicit-timestamped-lyrics"
                ),
                "uses_reference_timestamps": True,
                "independent_auto_benchmark_eligible": False,
                "timing_entries": len(entries),
                "assigned_entries": len(entries),
                "assigned_percent": 100.0,
                "trusted_entries": len(entries),
                "trusted_percent": 100.0,
                "matched_entries": len(entries),
                "matched_percent": 100.0,
                "review_required_count": 0,
                "review_required_percent": 0.0,
                "review_required": False,
                "note": "Used explicitly selected timestamped lyric input as assisted timing.",
            }
        else:
            raise LrcError(
                "--timing-source checked requires timestamped lyric input or a matching checked LRC"
            )
        features = None
        strategy = "checked-lrc-assisted"
    elif timing_source == "lyrics":
        timestamps = source_timestamps(entries)
        features = None
        strategy = "lyrics-source"
        report = {
            "backend": "lyrics",
            "timing_entries": len(entries),
            "assigned_entries": len(entries),
            "assigned_percent": 100.0,
            "trusted_entries": len(entries),
            "trusted_percent": 100.0,
            "matched_entries": len(entries),
            "matched_percent": 100.0,
            "low_confidence_count": 0,
            "low_confidence_percent": 0.0,
            "review_required_count": 0,
            "review_required_percent": 0.0,
            "review_required": False,
            "note": "Used timestamped lyric input as the timing source.",
        }
    elif timing_source == "checked_lrc":
        features = None
        strategy = "checked-lrc-hint"
    elif timing_source == "whispercpp":
        segments = run_whispercpp(audio_path, args)
        timestamps, report = match_whisper_segments(entries, segments, duration)
        features = None
        strategy = "whispercpp-experimental"
    elif timing_source == "ctc":
        progress_log("running explicit CTC alignment")
        timestamps, report = run_ctc_alignment(audio_path, entries, duration, args)
        progress_report_changes("CTC acoustic backtrack candidate generated", report, "ctc_acoustic_backtracks")
        features = None
        strategy = "ctc-forced-align"
    elif timing_source == "jactc":
        timestamps, report = run_japanese_ctc_alignment(audio_path, entries, duration, args)
        features = None
        strategy = "japanese-ctc-forced-align-experimental"
    elif timing_source == "whisperx":
        progress_log("running explicit WhisperX alignment")
        timestamps, report = run_whisperx_best_candidate(audio_path, entries, duration, args)
        features = None
        strategy = "whisperx-hybrid-experimental"
    elif timing_source == "backend_competition":
        timestamps, report, selected_backend, timing_selection = run_backend_competition_over_skeleton_gate(
            audio_path, entries, duration, args, anchor_path
        )
        features = None
        timing_source = selected_backend
        strategy = str(report.get("strategy") or "auto-candidate-selection")
    else:
        features = analyze_audio(audio_path, duration)
        timestamps, strategy = plan_timestamps(entries, features, args.mode)
        report = None
    if timing_source == "whisperx" and report and report.get("candidate_fusion"):
        timestamps, acoustic_changes = apply_acoustic_late_onset_refinement(audio_path, duration, timestamps, report)
        report["acoustic_refinement_count"] = len(acoustic_changes)
        report["acoustic_refinements"] = acoustic_changes
        progress_change_log("explicit WhisperX acoustic late-onset candidate generated", acoustic_changes)
    if report is None:
        report = {}
    central_timing_required = timing_source not in {"lyrics", "checked_lrc", "heuristic"}
    if not central_timing_required:
        timestamps, anchor_changes = apply_anchor_hints(
            entries, timestamps, report, anchor_path, duration
        )
        progress_change_log("anchor hints applied", anchor_changes)
        if anchor_changes:
            print(
                f"INFO: applied {len(anchor_changes)} anchor hint(s) from {anchor_path}.",
                file=sys.stderr,
            )
    final_timing_state: FinalTimingState | None = None
    neutral_probe_payload: dict[str, object] | None = None
    neutral_arbitration_payload: dict[str, object] | None = None
    if central_timing_required:
        if timing_selection is None:
            progress_log("generating and centrally evaluating timing candidates")
            timestamps, report, evaluated_timing, anchor_changes = prepare_backend_timing_preview(
                audio_path, entries, timestamps, report, duration, args, anchor_path
            )
            timing_selection = select_evaluated_backend((evaluated_timing,))
        else:
            anchor_changes = list(
                thaw_generation_diagnostics(timing_selection.selected).get("anchor_changes", [])
            )
        progress_change_log("anchor hints applied", anchor_changes)
        if anchor_changes:
            print(
                f"INFO: applied {len(anchor_changes)} anchor hint(s) from {anchor_path}.",
                file=sys.stderr,
            )
        timing_selection, neutral_probe_payload, neutral_arbitration_payload = (
            _precommit_neutral_boundary_arbitration(
                audio_path, entries, report, duration, args, timing_selection
            )
        )
        # Neutral arbitration is complete, but V14 deliberately postpones the
        # final immutability boundary until after one already-batched HuBERT
        # proof-completion replay.  Tail/current observations themselves remain
        # runtime evidence and cannot mutate timestamps directly.
        postcentral_joint_tail = _postcentral_acquire_targeted_joint_ctc_tails(
            report, timing_selection.selected.state, entries, duration, args
        )
        if isinstance(postcentral_joint_tail, dict):
            report["postcentral_targeted_joint_ctc_tail"] = postcentral_joint_tail
        # After the existing CTC-tail mechanism has exhausted what it can prove,
        # one independent Japanese HuBERT batch performs two post-Central-only
        # observations: occurrence-gated predecessor-tail geometry from joint
        # lyric pairs, and hypothesis-neutral full-current-line onset corroboration.
        # Neither observation can admit a candidate or reselect a timestamp.
        postcentral_hubert_tail = _postcentral_acquire_known_lyric_hubert_tails(
            report, timing_selection.selected.state, entries, duration, audio_path, args
        )
        if isinstance(postcentral_hubert_tail, dict):
            report["postcentral_known_lyric_hubert_tail"] = postcentral_hubert_tail

        # A non-final-associated HuBERT opening may complete the missing
        # independent-boundary proof of one *already-existing* CTC/recurrence
        # timestamp.  The replay may add only a provenance-enriched candidate at
        # that parent timestamp; full Central plus the complete downstream review
        # path must accept it without collateral before production selection is
        # revised.  No HuBERT timestamp is ever admitted directly.
        timing_selection, hubert_recovery_payload = (
            _precommit_hubert_existing_candidate_recovery(
                entries, report, timing_selection
            )
        )
        report[_HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY] = copy.deepcopy(
            hubert_recovery_payload
        )

        # Node19: an already-existing independent peer candidate may receive one
        # private, target-bounded Central replay when its own immutable evidence
        # already satisfies Central.  Run this after HuBERT proof completion so
        # predecessor corrections accepted by Central are visible to the peer
        # rebound.  This admits no new timestamp and never unions the peer graph
        # into production candidate acquisition.
        timing_selection, peer_recovery_payload = (
            _precommit_independent_peer_existing_candidate_recovery(
                entries, report, timing_selection
            )
        )
        report[_PEER_EXISTING_CANDIDATE_RECOVERY_KEY] = copy.deepcopy(
            peer_recovery_payload
        )

        # Final candidate/timestamp/backend selection is immutable from here.
        # Newly captured final-associated onset/tail evidence may only reduce
        # already-existing peer review conflicts; it cannot reselect a timestamp.
        postcentral_projection = _postcentral_existing_conflict_projection(
            report, timing_selection.selected.state
        )
        if isinstance(postcentral_projection, dict):
            report[_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = postcentral_projection
        final_timing_state = commit_evaluated_backend(timing_selection, report)
        final_timing_state = _apply_multimodel_consensus_runtime(
            final_timing_state,
            timing_selection.selected.candidate_sets,
            report,
            entries,
            args,
        )
        duplicate_recompute, duplicate_resolution = (
            _postcommit_recompute_duplicate_review(
                entries, report, timing_selection, final_timing_state
            )
        )
        if isinstance(duplicate_resolution, dict):
            report["postcentral_hubert_duplicate_occurrence_resolution"] = copy.deepcopy(
                duplicate_resolution
            )
        report["final_self_check"] = compute_final_timing_self_check(
            report, final_timing_state, entries
        )
        _mark_material_recovery_moves_for_review(report)
        final_timing_state = commit_final_timing_state(
            final_timing_state,
            report,
            thaw_generation_diagnostics(timing_selection.selected),
            finding_states=timing_selection.selected.finding_states,
            final_output=True,
        )
        assert_report_timestamp_equality(report, final_timing_state)
        update_report_confidence_metrics(report, apply_silent_risk=True)
        finalize_review_summary(report)
        projection_seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
        if isinstance(projection_seed, dict):
            report[_BACKEND_PROVENANCE_PROJECTION_KEY] = _finalize_backend_provenance_projection(
                report,
                projection_seed,
                timing_selection,
            )
            _validate_backend_provenance_projection(report)
        assert_report_review_contract(report)
    written_times = (
        final_timing_state.written_times
        if final_timing_state is not None
        else tuple(canonicalize_lrc_time(timestamp) for timestamp in timestamps)
    )
    violations = segment_postcentral_nonmonotonic(report, written_times)
    if violations and str(getattr(args, "skeleton", "ctc")) == "segment":
        # Never write a mixed or backwards draft. Return to original whole
        # backend competition, not a per-row timestamp repair.
        raise SegmentSkeletonRollback(
            f"segment-current final Central backtracks at rows {violations}"
        )
    written_times = apply_line_start_lead(
        report, written_times, resolve_start_lead_ms(getattr(args, "start_lead_ms", None))
    )
    if args.probe:
        DEFAULT_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir).expanduser().resolve() if args.report_dir else DEFAULT_REPORT_DIR
    report_path, audit_path, template_path = report_artifact_paths(audio_path, output_path, report_dir)
    metadata_lines = output_metadata_lines(lyrics.metadata_lines, timing_source)
    report.setdefault("backend", timing_source)
    if getattr(args, "option_a_rollback_reason", None):
        report["segment_skeleton_status"] = {
            "status": "FALLBACK_CTC",
            "reason": "POSTCENTRAL_NONMONOTONIC",
            "detail": str(args.option_a_rollback_reason)[:300],
            "whole_song": True,
            "BK_used": False,
        }
    report["requested_timing_source"] = args.timing_source
    report["resolved_timing_source"] = timing_source
    report.update(
        inference_mode_metadata(
            requested_timing_source,
            timing_source,
            anchor_path,
        )
    )
    report["strategy"] = strategy
    report["mode"] = timing_source
    report["heuristic_mode"] = args.mode
    report["audio_path"] = str(audio_path)
    report["lyrics_path"] = str(lyrics_path)
    report["output_path"] = str(output_path)
    report["report_path"] = str(report_path)
    report["duration_seconds"] = round(duration, 3)
    report["timing_entries"] = len(entries)
    report["display_lines"] = sum(len(entry.lines) for entry in entries)
    report["requested_whisper_language"] = requested_whisper_language
    report["resolved_whisper_language"] = args.whisper_language
    if skipped_markers:
        report["skipped_marker_entries"] = skipped_markers
        report["skipped_marker_count"] = len(skipped_markers)
    if final_timing_state is not None:
        report[_UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY] = (
            _build_unresolved_evidence_requirements_projection(
                report,
                final_timing_state,
            )
        )
        report[_SAME_OCCURRENCE_BOUNDARY_EQUIVALENCE_KEY] = (
            _same_occurrence_boundary_equivalence_projection(final_timing_state)
        )
        if neutral_probe_payload is not None:
            report[_R2T_NEUTRAL_CONFLICT_PROBE_KEY] = copy.deepcopy(neutral_probe_payload)
        if neutral_arbitration_payload is not None:
            report[_NEUTRAL_BOUNDARY_ARBITRATION_KEY] = copy.deepcopy(
                neutral_arbitration_payload
            )

    progress_quality_summary(f"final timing source={timing_source}", report)
    # Review/trust state is diagnostic metadata, never an implicit control-flow gate.
    # Match the public baseline: normal processing always writes the best current LRC;
    # only explicit strict-review CLI options may request a non-zero exit afterward.
    report["production_write_blocked"] = False
    report["production_write_block_reason"] = None
    if bool(report.get("review_required")):
        report["output_trust_status"] = "review-required"
    elif int(report.get("unverified_count", 0) or 0) > 0:
        report["output_trust_status"] = "unverified"
    else:
        report["output_trust_status"] = "trusted"
    progress_log(f"staging LRC and report pair: {output_path}")
    report.setdefault("backend", timing_source)
    report["requested_timing_source"] = args.timing_source
    report["resolved_timing_source"] = timing_source
    report["strategy"] = strategy
    report["mode"] = timing_source
    report["heuristic_mode"] = args.mode
    report["audio_path"] = str(audio_path)
    report["lyrics_path"] = str(lyrics_path)
    report["output_path"] = str(output_path)
    report["report_path"] = str(report_path)
    report["duration_seconds"] = round(duration, 3)
    report["timing_entries"] = len(entries)
    report["display_lines"] = sum(len(entry.lines) for entry in entries)
    report["requested_whisper_language"] = requested_whisper_language
    report["resolved_whisper_language"] = args.whisper_language
    if skipped_markers:
        report["skipped_marker_entries"] = skipped_markers
        report["skipped_marker_count"] = len(skipped_markers)
    raw_coverage = report.get("lyric_raw_asr_coverage")
    if isinstance(raw_coverage, dict):
        preflight["raw_asr_coverage"] = raw_coverage
        if raw_coverage.get("status") in {"LOW_COVERAGE", "ORDER_INCONSISTENT"}:
            preflight["status"] = "INPUT_SUSPECT"
            preflight["reason_codes"].append("RAW_ASR_" + str(raw_coverage["status"]))
    preflight["retained_entries"] = len(entries)
    report["lyric_preflight"] = preflight
    report["input_status"] = preflight["status"]
    apply_weak_review_policy(report)
    review_tsv_path = report_path.with_name(report_path.name.replace(".align-report.json", ".review.tsv"))
    report["review_tsv_path"] = str(review_tsv_path)
    write_review_tsv(review_tsv_path, report, entries)
    progress_log(f"publishing LRC/report pair: {report_path}")
    _publish_lrc_report(
        output_path,
        report_path,
        lambda stage: write_lrc(
            stage, audio_path, lyrics_path, entries, written_times, duration,
            True, metadata_lines, f"LRC tools {strategy}",
        ),
        report,
    )
    if not getattr(args, "no_review_summary", False):
        try:
            for line in format_top_review(review_tsv_path):
                print(line)
        except (OSError, ValueError) as error:
            print(f"REVIEW TOP 5: unavailable ({type(error).__name__})")

    if strategy.startswith("weighted-fallback"):
        print(
            "WARN: detected audio segments did not closely match lyric line count; "
            "preserved lyric order and used weighted timing with onset snapping.",
            file=sys.stderr,
        )
    if timing_source == "heuristic":
        print(
            "WARN: heuristic timing is experimental; do not treat this output as "
            "production-trusted without independent review.",
            file=sys.stderr,
        )
    if timing_source == "whispercpp":
        print(
            "WARN: whispercpp timing is experimental. Inspect the .align-report.json and "
            "run evaluate_lrc.py against a checked reference before trusting it.",
            file=sys.stderr,
        )
    if timing_source == "whisperx":
        print(
            "WARN: whisperx timing is experimental but is the current highest-accuracy "
            "local backend. Inspect the .align-report.json and verify difficult songs.",
            file=sys.stderr,
        )
    if report.get("review_required"):
        print(
            f"WARN: alignment report has {report.get('suspicious_alignment_count', 0)} suspicious line(s), "
            f"{report.get('review_required_count', 0)} review-required. "
            "Inspect suspicious_alignments before trusting output.",
            file=sys.stderr,
        )
    if strict_reason := strict_review_failure_reason(report, args):
        review_only = not strict_reason.startswith("trusted_percent=")
        audit_path = write_review_audit(output_path, report, audit_path, review_only=review_only)
        template_path = write_review_anchor_template(output_path, report, template_path, review_only=review_only)
        report["strict_failure_reason"] = strict_reason
        report["review_audit_path"] = str(audit_path)
        report["anchor_template_path"] = str(template_path)
        _publish_lrc_report(output_path, report_path, lambda q: shutil.copy2(output_path, q), report)
        raise LrcError(
            f"strict review failed for {output_path}: {strict_reason}. "
            f"Draft LRC, report, review audit, and anchor template were written; inspect {audit_path}, "
            f"{template_path}, or {report_path}."
        )
    print(
        f"OK: {output_path} ({len(entries)} timing entries, "
        f"{sum(len(entry.lines) for entry in entries)} display lines, {duration:.2f}s, "
        f"{len(features.segments) if features else 'n/a'} detected segments, "
        f"timing_source={timing_source}, heuristic_mode={args.mode}, strategy={strategy})"
    )
    print(f"REPORT: {report_path}")
    print(f"REVIEW: {review_tsv_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate rough same-folder .lrc files from FLAC plus same-name lyric text."
    )
    parser.add_argument("audio", nargs="+", help="FLAC file path(s)")
    parser.add_argument(
        "--lyrics",
        help="Explicit lyric path. Omit to auto-detect <song>.lyrics.lrc, <song>.lyrics.txt, or <song>.txt.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing .lrc files.")
    parser.add_argument("--discard-manual-edits", action="store_true", help="allow --overwrite to replace an LRC edited in the review page; the edited LRC and edit log are archived first")
    parser.add_argument(
        "--output",
        help="Explicit output .lrc path. Only valid when processing one FLAC file.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Write a disposable candidate to project outputs/probes instead of touching the same-name LRC.",
    )
    parser.add_argument(
        "--report-dir",
        help="Directory for JSON reports and strict-review audit artifacts. Default: project outputs/reports.",
    )
    parser.add_argument(
        "--start-lead-ms",
        type=int,
        default=None,
        help="write every line start this many ms earlier (default 0 = off; env LRC_START_LEAD_MS)",
    )
    parser.add_argument(
        "--no-review-summary",
        action="store_true",
        help="suppress the top-5 review list (used by R2 baseline pass)",
    )
    parser.add_argument(
        "--mode",
        choices=["energy", "even"],
        default="energy",
        help="energy: waveform/onset snapping; even: weighted rough timing only.",
    )
    parser.add_argument(
        "--timing-source",
        choices=["auto", "lyrics", "checked", "ctc", "jactc", "whisperx", "whispercpp", "heuristic", "audio"],
        default="auto",
        help=(
            "auto: use ordered lyric text only and run CTC/WhisperX independently; checked: "
            "explicitly use a checked/timestamped lyric source; lyrics: preserve timestamps "
            "from timestamped lyric input; otherwise run CTC and "
            "WhisperX candidates and select by report quality; ctc: MMS/CTC forced alignment over known lyrics; "
            "jactc: experimental Japanese Wav2Vec2 CTC candidate; "
            "whispercpp: experimental ASR matching; "
            "whisperx: experimental whisper.cpp + WhisperX hybrid alignment; "
            "heuristic/audio: rough experimental timing only."
        ),
    )
    parser.add_argument(
        "--arbiter",
        choices=["off", "gross-rescue", "reviewer-validity"],
        default="off",
        help=(
            "Final Central timing arbiter. off preserves the accepted baseline; "
            "gross-rescue applies the registered R1 valid-current preservation rule; "
            "reviewer-validity lets independent reviewer agreement satisfy only registered soft evidence gaps."
        ),
    )
    parser.add_argument(
        "--timing-source-policy", choices=("default", "ctc-first"),
        default="ctc-first",
        help="N2 production default: Whisper/RAW are diagnostic-only; explicit default restores legacy.",
    )
    parser.add_argument(
        "--reviewer-evidence",
        type=Path,
        default=None,
        help="Per-song reviewer evidence sidecar for reviewer-validity mode. Missing evidence fails closed to Central.",
    )
    parser.add_argument(
        "--skeleton", choices=("segment", "ctc"), default="segment",
        help="Option A: segment-vocal-DP-local-CTC current (whole-song fail-closed to CTC); "
             "ctc: original whole-song MMS CTC skeleton.",
    )
    parser.add_argument(
        "--segment-realign", choices=("auto", "off"), default="off",
        help="LRC 2.0: bounded CTC re-alignment of the row runs the lyric structure flags "
             "(hypotheses for Central only). Off by default until the 2.0 replay is accepted; "
             "--segment-realign auto or LRC_SEGMENT_REALIGN=auto turns it on.",
    )
    parser.add_argument("--whisper-cli", help="Path to whisper.cpp whisper-cli.exe.")
    parser.add_argument("--whisper-model", help="Path to whisper.cpp ggml model.")
    parser.add_argument(
        "--whisper-suppress-nst",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Suppress non-speech tokens in whisper.cpp decoding. Default is auto "
            "for whisperx: try normal decoding first, then retry with suppression "
            "if matching confidence is low."
        ),
    )
    parser.add_argument("--whisperx-python", help="Path to the Python executable with whisperx installed.")
    parser.add_argument(
        "--whisperx-device",
        default="auto",
        help="Device passed to WhisperX alignment, default: auto (CUDA when available, otherwise CPU).",
    )
    parser.add_argument(
        "--whisper-language",
        default="auto",
        help="Spoken language passed to whisper.cpp, default: auto from lyric text.",
    )
    parser.add_argument(
        "--vocal-onset-refine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use GPU Demucs vocal-stem onset evidence to resolve large CTC/WhisperX disagreements.",
    )
    parser.add_argument(
        "--vocal-ctc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run primary CTC forced alignment on a cached Demucs vocal stem; use --no-vocal-ctc for full-mix fallback.",
    )
    parser.add_argument(
        "--anchor-hints",
        help=(
            "Optional timestamped LRC subset with manually checked line anchors. "
            "Only an explicit path enables assisted anchor evidence; omission never auto-discovers anchors."
        ),
    )
    parser.add_argument(
        "--fail-on-review-required",
        action="store_true",
        help="Exit non-zero after writing outputs if the report contains any review-required line.",
    )
    parser.add_argument(
        "--min-trusted-percent",
        type=float,
        default=None,
        help="Exit non-zero after writing outputs unless trusted_percent is at least this value.",
    )
    parser.add_argument(
        "--strict-review",
        action="store_true",
        help=(
            "Explicit QA gate: after writing LRC/report artifacts, exit non-zero on review-required lines, "
            "collapse, or trusted_percent below 100."
        ),
    )
    parser.add_argument(
        "--crossline-retry-timeout",
        type=float,
        default=180.0,
        help=(
            "Maximum seconds for one bounded cross-line CTC helper invocation. "
            "A timeout is recorded as failed evidence and does not block LRC output. Default: 180."
        ),
    )
    parser.add_argument(
        "--model-cache", action=argparse.BooleanOptionalAction, default=True,
        help="Persist model execution outputs under outputs/cache (default: enabled).",
    )
    parser.add_argument(
        "--model-cache-only", action="store_true",
        help="Use only verified model-output cache entries; cache misses fail explicitly.",
    )
    parser.add_argument(
        "--multimodel-consensus", choices=("off", "on"), default="on",
        help="Run gated multi-model consensus on Central provisional rows (default: on).",
    )
    parser.add_argument(
        "--stage-timings-pass", choices=("standalone", "baseline", "final"), default="standalone",
        help="label recorded in report stage_timings (set by R2)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cache_mode = _apply_model_cache_flags(args)
    except LrcError as exc:
        parser.error(str(exc))
    print(f"INFO: model cache mode={cache_mode}", file=sys.stderr, flush=True)
    if args.output and len(args.audio) != 1:
        parser.error("--output can only be used with exactly one audio file")
    if args.probe and args.output:
        parser.error("--probe cannot be combined with --output")
    failures = 0
    for raw_audio in args.audio:
        temp_snapshot = _managed_temp_snapshot()
        try:
            try:
                process_audio(Path(raw_audio).expanduser().resolve(), args)
            except SegmentSkeletonRollback as retry_reason:
                # Same exact input, full 1.2-style CTC/WhisperX competition,
                # Central and reviewer recomputed. No partial row adoption.
                progress_log(f"Option A whole-song Central rollback: {retry_reason}")
                old_args = copy.copy(args)
                old_args.skeleton = "ctc"
                old_args.option_a_rollback_reason = str(retry_reason)
                process_audio(Path(raw_audio).expanduser().resolve(), old_args)
        except ModelCacheMiss as exc:
            failures += 1
            print(f"ERROR: {raw_audio}: {exc}", file=sys.stderr)
        except LrcError as exc:
            failures += 1
            print(f"ERROR: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # pragma: no cover - last-resort CLI guard
            failures += 1
            print(f"ERROR: unexpected failure for {raw_audio}: {exc}", file=sys.stderr)
        finally:
            # Do not accumulate whole-song/local-window WAVs when one CLI call
            # processes multiple songs.  Any transient Windows delete failure
            # remains registered and is retried after helper shutdown at exit.
            _cleanup_managed_temp_dirs_since(temp_snapshot)
    return 1 if failures else 0


_STAGE_TIMING_SPEC = lrc_stage_timings.STAGE_TIMING_SPEC

report_trust.bind_late_callbacks(
    _final_post_central_material_conflict_entries=_final_post_central_material_conflict_entries,
    _suspicious_entry_requires_review_for_decision=_suspicious_entry_requires_review_for_decision,
)

asr_backends.bind_late_callbacks(
    _AUDIO_API=_AUDIO_API,
)

lrc_stage_timings.install(globals(), _STAGE_TIMING_SPEC)

if __name__ == "__main__":
    raise SystemExit(main())
