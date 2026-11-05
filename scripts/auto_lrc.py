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
import central_commit
from central_commit import (
    CENTRAL_RESOLVABLE_SUSPICIOUS_FLAGS,
    RAW_TIMING_DRIFT_MAX_MEDIAN_DELTA_SECONDS,
    VALID_ALTERNATIVE_MATERIAL_SECONDS,
    _KNOWN_LYRIC_HUBERT_LINEAGE,
    _KNOWN_LYRIC_HUBERT_ONSET_PRODUCER,
    _candidate_evidence_projection,
    _candidate_has_revision_bound_ownership_violation,
    _candidate_is_exact_evidence_subsumed_by_selected,
    _candidate_is_same_occurrence_boundary_equivalent,
    _central_decision_supersedes_suspicious_row,
    _ctc_current_clock_time,
    _decision_candidate_evaluations,
    _decision_has_material_distant_timing_hypothesis,
    _evidence_report_payload,
    _explicit_whisper_arbitration_resolves_raw_disagreement,
    _final_post_central_material_conflict_entries,
    _historical_timing_findings,
    _install_runtime_raw_timing_drift_falsifications,
    _known_lyric_hubert_current_onset_is_valid,
    _known_lyric_hubert_current_onset_payload,
    _material_distant_timing_alternatives,
    _postcentral_hubert_current_onset_map,
    _precentral_candidate_gap_exceeds_half_second,
    _project_suspicious_alignments_to_central_state,
    _raw_timing_drift_falsification_details,
    _raw_vocal_occurrence_observation,
    _recompute_duplicate_occurrence_offset_diagnostics,
    _recovery_has_revision_bound_resolution,
    _runtime_hubert_duplicate_occurrence_resolution,
    _runtime_hubert_duplicate_resolution_entries,
    _runtime_raw_timing_drift_falsified_candidate_ids,
    _suspicious_entry_requires_review_for_decision,
    commit_final_timing_state,
    compute_final_timing_self_check,
    evaluate_backend_timing,
    evaluate_final_timing_state_for_report,
    prepare_backend_timing_preview,
)
from backend_bundle import (
    _BACKEND_PEER_SIMULATION_SCHEMA_VERSION,
    _HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY,
    _PEER_EXISTING_CANDIDATE_RECOVERY_KEY,
    _SCAFFOLD_DROP,
    _UNRESOLVED_EVIDENCE_REQUIREMENTS_SCHEMA_VERSION,
    _backend_lineage_for_projection,
    _backend_projection_attr,
    _backend_projection_derived_field,
    _backend_projection_error_code,
    _backend_projection_field,
    _build_unresolved_evidence_requirements_projection,
    _bundle_candidate_lookup,
    _bundle_candidate_payload,
    _bundle_candidate_sets_payload,
    _bundle_capability_projection,
    _bundle_content_identity_observations,
    _bundle_content_observations_bound_to_scaffold,
    _bundle_tail_binding_payload,
    _bundle_tail_context_payload,
    _capture_evaluated_backend_bundle,
    _central_scaffold_payload,
    _central_scaffold_shape,
    _final_candidate_valid_under_predecessor_tail,
    _install_peer_review_only_conflicts,
    _neutral_arbitration_support_families,
    _peer_cluster_evidence_state,
    _peer_cluster_projection,
    _peer_cluster_representative,
    _peer_final_context_tail_runtime,
    _peer_observation_cluster_summaries,
    _peer_policy,
    _peer_review_only_conflict_projection,
    _postcentral_bind_ctc_assignment_tail_coexistence,
    _postcentral_hubert_cluster_is_falsified,
    _postcentral_hubert_observation_matches_row_hypotheses,
    _postcentral_hubert_row_hypothesis_set_revision,
    _postcentral_rebind_existing_peer_conflicts,
    _postcentral_recovery_changed_entries,
    _public_backend_capability_summary,
    _public_local_capability_summary,
    _rebind_peer_review_only_conflicts_to_final_selection,
    _resolve_final_peer_predecessor_tail,
    _same_occurrence_boundary_equivalence_projection,
    _scaffold_key_is_private,
    _scaffold_safe_value,
    _simulate_cross_backend_peer_graph,
    _strict_candidate_support_families_for_final_context,
    _union_peer_candidate_sets,
    _union_peer_content_identity_observations,
    _unresolved_evidence_audio_revision,
    _unresolved_evidence_missing_categories,
    _unresolved_evidence_onset_projection,
    _validate_evaluated_backend_bundle,
    central_backend_summary_payload,
    central_candidate_selection_payload,
    central_selection_reason,
    select_evaluated_backend,
)
from central_peer_recovery import (
    RECOVERY_REVIEW_MOVE_THRESHOLD_CENTISECONDS,
    _BACKEND_PROVENANCE_PROJECTION_SCHEMA_VERSION,
    _HUBERT_EXISTING_CANDIDATE_RECOVERY_MODE,
    _HUBERT_RECOVERY_DIRECT_ONSET_KIND,
    _KNOWN_LYRIC_HUBERT_AUDIO_LOADER,
    _KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT,
    _KNOWN_LYRIC_HUBERT_HELPER_SOURCE,
    _KNOWN_LYRIC_HUBERT_LANGUAGE,
    _KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256,
    _KNOWN_LYRIC_HUBERT_MODEL_ID,
    _KNOWN_LYRIC_HUBERT_MODEL_REVISION,
    _KNOWN_LYRIC_HUBERT_PRODUCER,
    _KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION,
    _NEUTRAL_ARBITRATION_FINAL_SEMANTIC_FIELDS,
    _NEUTRAL_ARBITRATION_HARD_COLLATERAL_FIELDS,
    _NEUTRAL_BOUNDARY_ARBITRATION_KEY,
    _NEUTRAL_BOUNDARY_ARBITRATION_MODE,
    _PEER_DUPLICATE_RECOMPUTE_KEY,
    _PEER_EXISTING_CANDIDATE_RECOVERY_MODE,
    _R2T_NEUTRAL_CONFLICT_PROBE_KEY,
    _R2T_NEUTRAL_CONFLICT_PROBE_MODE,
    _apply_evaluated_backend_projection,
    _build_backend_provenance_projection_seed,
    _existing_peer_recovery_representative,
    _finalize_backend_provenance_projection,
    _hubert_neutral_probe_window_start,
    _hubert_probe_window_covers_hypotheses,
    _hubert_probe_window_covers_request_hypotheses,
    _hubert_recovery_boundary_observation_map,
    _hubert_recovery_consensus_candidate,
    _hubert_recovery_isolation_units,
    _hubert_recovery_project_selection,
    _hubert_recovery_replay_gate,
    _known_lyric_hiragana_reading,
    _known_lyric_hubert_char_diagnostic_payload,
    _known_lyric_hubert_dummy_candidate,
    _known_lyric_hubert_dummy_tail,
    _known_lyric_hubert_full_line_bridge_is_coherent,
    _known_lyric_hubert_helper_revision,
    _known_lyric_hubert_opening_evidence,
    _known_lyric_hubert_pair_tail_current_evidence_scope,
    _known_lyric_hubert_pair_tail_occurrence_status,
    _known_lyric_hubert_predecessor_occurrence_status,
    _make_known_lyric_hubert_current_onset,
    _mark_material_recovery_moves_for_review,
    _neutral_arbitration_candidate_map,
    _neutral_arbitration_consensus_candidate,
    _neutral_arbitration_final_semantic_rows,
    _neutral_arbitration_matching_cluster,
    _neutral_arbitration_project_selection,
    _neutral_arbitration_replay_gate,
    _neutral_arbitration_selection_with_revised_eval,
    _neutral_arbitration_valid_candidate_ids,
    _neutral_probe_trigger_rows,
    _peer_recovery_project_selection,
    _peer_recovery_replay_gate,
    _postcentral_acquire_known_lyric_hubert_tails,
    _postcentral_acquire_targeted_joint_ctc_tails,
    _postcentral_bundle_with_additional_tail_observations,
    _postcentral_existing_conflict_projection,
    _postcentral_plan_known_lyric_hubert_current_probe_requests,
    _postcentral_plan_known_lyric_hubert_predecessor_only_requests,
    _postcentral_plan_known_lyric_hubert_requests,
    _postcentral_plan_known_lyric_hubert_timing_failure_diagnostic_requests,
    _postcentral_plan_targeted_joint_ctc_tail_requests,
    _postcentral_residual_material_conflict_entries,
    _postcentral_tail_blocked_peer_entries,
    _postcentral_tail_enriched_bundle,
    _postcommit_recompute_duplicate_review,
    _precommit_hubert_existing_candidate_recovery,
    _precommit_independent_peer_existing_candidate_recovery,
    _precommit_neutral_boundary_arbitration,
    _r2t_authority_snapshot,
    _r2t_collect_final_material_targets,
    _r2t_material_cluster_rows,
    _r2t_plan_neutral_conflict_probe_requests,
    _r2t_probe_relation,
    _r2t_unresolved_projection_rows,
    _r2t_validated_peer_conflict_rows,
    _update_projection_seed_for_arbitrated_selection,
    _update_projection_seed_for_hubert_recovery,
    _update_projection_seed_for_peer_recovery,
    commit_evaluated_backend,
    run_r2t_unified_neutral_conflict_probe,
    validate_backend_selection,
)
from local_whisper_retry import (
    CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
    CROSSLINE_RETRY_MAX_WINDOW_EXPANSIONS,
    CTC_WIDE_REGION_PEER_AGREEMENT_SECONDS,
    CTC_WIDE_REGION_SUPPORT_SECONDS,
    CTC_WIDE_REGION_VETO_SECONDS,
    LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
    LOCAL_WHISPER_MAX_WINDOW_SECONDS,
    _apply_context_conditioned_whisper_hypothesis_arbitration,
    _attach_ctc_wide_region_candidates,
    _augment_candidates_with_joint_boundary_consensus,
    _augment_candidates_with_local_whisper_consensus,
    _augment_candidates_with_local_whisper_vocal_consensus,
    _augment_local_ctc_candidates_with_runtime_backend_consensus,
    _authoritative_tail_semantic_revision,
    _candidate_set_semantic_revision,
    _central_preview_generation_candidates,
    _crossline_retry_helper_cache_key,
    _crossline_retry_semantic_key,
    _freeze_runtime_tail_observation_map,
    _independent_tail_evidence_payload,
    _joint_consensus_spans,
    _joint_previous_tail_evidence,
    _local_whisper_cache_key,
    _local_whisper_execution_request_revision,
    _local_whisper_outcome_for_request,
    _local_whisper_request_execution_revision,
    _local_whisper_vocal_fusion_candidate,
    _novel_crossline_retry_requests,
    _prepare_local_whisper_batch,
    _retry_candidate_reference,
    _retry_capability_diagnostic_payload,
    _retry_diagnostic_payload,
    _retry_semantic_state_revision,
    _shift_local_assignment_to_global,
    _tail_authority_diagnostics,
    bind_ctc_generation_candidate_evidence,
    bind_runtime_ctc_assignment_tail_fallback,
    bind_runtime_local_whisper_evidence,
    bind_runtime_raw_content_identity,
    bind_runtime_whisperx_ctc_consensus,
    bind_runtime_whisperx_row_evidence,
    collect_central_timing_candidates,
    crossline_retry_request_is_current,
    execute_crossline_retry_epoch,
    execute_local_whisper_request,
    freeze_base_retry_epoch,
    independent_tail_evidence_payload,
    invoke_crossline_retry,
    merge_runtime_independent_tail_evidence,
    parse_crossline_retry_output,
    plan_crossline_retry_requests,
    plan_joint_boundary_retry_requests,
    plan_local_whisper_requests,
    record_crossline_retry_identity_hypothesis,
    run_local_whisper_requests,
    runtime_raw_ctc_consensus_for_entry,
    runtime_whisperx_ctc_consensus_for_entry,
)
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


_SAME_OCCURRENCE_BOUNDARY_EQUIVALENCE_KEY = "same_occurrence_boundary_equivalence"


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


_KNOWN_LYRIC_HUBERT_WHISPERX_VERSION = "3.8.6"
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

lrc_stage_timings.install(globals(), _STAGE_TIMING_SPEC)

asr_backends.bind_late_callbacks(
    _AUDIO_API=_AUDIO_API,
)

central_commit.bind_late_callbacks(
    _AUDIO_API=_AUDIO_API,
)

if __name__ == "__main__":
    raise SystemExit(main())
