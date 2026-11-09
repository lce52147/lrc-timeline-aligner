"""Extracted from auto_lrc.py (split phase 2): post-central recovery: hubert and peer existing-candidate recovery, joint CTC tail acquisition, duplicate review recompute."""

from __future__ import annotations

import collections
import concurrent.futures
import copy
import hashlib
import json
import math
import statistics
import subprocess
import tempfile
import unicodedata
from dataclasses import replace
from pathlib import Path
from numeral_reading import numerals_for_alignment
from local_whisper_retry import CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS, LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS, _crossline_retry_helper_cache_key, _retry_candidate_reference, _retry_diagnostic_payload, invoke_crossline_retry
from runtime_evidence_binding import bind_runtime_ctc_assignment_tail_fallback, bind_runtime_whisperx_row_evidence
from asr_backends import decode_temp_wav
from acoustic_hypotheses import assert_report_timestamp_equality
from report_trust import _RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY, _RUNTIME_POSTCENTRAL_CTC_TAIL_PRODUCER_CONTEXT_KEY, _RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY, _RUNTIME_POSTCENTRAL_HUBERT_BOUNDARY_OBSERVATION_KEY, _RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY, _RUNTIME_POSTCENTRAL_HUBERT_DUPLICATE_RESOLUTION_KEY, _timing_failure_requires_review, _validate_backend_provenance_projection, apply_central_summary_to_report, assert_report_review_contract, summarize_central_timing_state, thaw_generation_diagnostics, update_report_confidence_metrics
from timing_core import BackendSelection, CrossLineRetryRequest, EvaluatedBackendBundle, EvaluatedBackendTiming, IndependentCurrentOnsetEvidence, IndependentTailEvidence, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, PostcentralCtcTailProducerContext, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, TailEvidenceContext, TailEvidenceScope, TailObservationBinding, TimingCandidate, _candidate_support_families, _make_tail_provenance, _runtime_tail_observation_map, _tail_context_is_valid, _tail_evidence_context_from_report, _tail_provenance_is_valid, _tail_provenance_matches_context, analyze_candidate_regions, candidate_has_bound_direct_onset_evidence, derive_reliable_tail, evaluate_timing_candidate, make_timing_candidate
from lyrics_io import LyricEntry, _BACKEND_PROVENANCE_PROJECTION_KEY, entry_sung_text, normalize_match_text, optional_kakasi_converter
from audio_features import HOP_SECONDS
from runtime_base import DEFAULT_EVIDENCE_WORKERS, LrcError, _path_sha256, _stable_file_sha256, _timing_content_digest, canonicalize_lrc_time, default_whisperx_python, gpu_compute_phase, run_command
from runtime_base import TimingTokenSpan
import lrc_stage_timings
from central_commit import (
    _KNOWN_LYRIC_HUBERT_LINEAGE,
    _KNOWN_LYRIC_HUBERT_ONSET_PRODUCER,
    _known_lyric_hubert_current_onset_is_valid,
    _known_lyric_hubert_current_onset_payload,
    _recompute_duplicate_occurrence_offset_diagnostics,
    _runtime_hubert_duplicate_occurrence_resolution,
    commit_final_timing_state,
    evaluate_backend_timing,
)
import argparse
from typing import Mapping
from timing_core import CandidateRegionEvidence, CrossLineRetryOutcome, FinalTimingState, TailRegionEvidence, TimingDecision
from backend_bundle import (
    _HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY,
    _PEER_EXISTING_CANDIDATE_RECOVERY_KEY,
    _bundle_candidate_lookup,
    _bundle_candidate_sets_payload,
    _bundle_tail_binding_payload,
    _bundle_tail_context_payload,
    _neutral_arbitration_support_families,
    _postcentral_bind_ctc_assignment_tail_coexistence,
    _postcentral_hubert_observation_matches_row_hypotheses,
    _postcentral_rebind_existing_peer_conflicts,
    _strict_candidate_support_families_for_final_context,
    _validate_evaluated_backend_bundle,
    central_backend_summary_payload,
    central_candidate_selection_payload,
    central_selection_reason,
)
from central_commit_tail import (
    _apply_evaluated_backend_projection,
    _finalize_backend_provenance_projection,
    _postcentral_existing_conflict_projection,
)
from neutral_arbitration import (
    _NEUTRAL_ARBITRATION_FINAL_SEMANTIC_FIELDS,
    _NEUTRAL_ARBITRATION_HARD_COLLATERAL_FIELDS,
    _neutral_arbitration_candidate_map,
    _neutral_arbitration_final_semantic_rows,
    _neutral_arbitration_selection_with_revised_eval,
    _neutral_arbitration_valid_candidate_ids,
)
from known_lyric_hubert_planning import (
    _KNOWN_LYRIC_HUBERT_PRODUCER,
    _KNOWN_LYRIC_HUBERT_MODEL_ID,
    _KNOWN_LYRIC_HUBERT_MODEL_REVISION,
    _KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256,
    _KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION,
    _KNOWN_LYRIC_HUBERT_LANGUAGE,
    _KNOWN_LYRIC_HUBERT_AUDIO_LOADER,
    _KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT,
    _HUBERT_EXISTING_CANDIDATE_RECOVERY_MODE,
    _PEER_EXISTING_CANDIDATE_RECOVERY_MODE,
    _PEER_DUPLICATE_RECOMPUTE_KEY,
    _HUBERT_RECOVERY_DIRECT_ONSET_KIND,
    _KNOWN_LYRIC_HUBERT_HELPER_SOURCE,
    _known_lyric_hubert_helper_revision,
    _known_lyric_hiragana_reading,
    _postcentral_residual_material_conflict_entries,
    _postcentral_plan_known_lyric_hubert_requests,
    _hubert_neutral_probe_window_start,
    _hubert_probe_window_covers_hypotheses,
    _hubert_probe_window_covers_request_hypotheses,
    _postcentral_plan_known_lyric_hubert_current_probe_requests,
    _postcentral_plan_known_lyric_hubert_timing_failure_diagnostic_requests,
    _known_lyric_hubert_char_diagnostic_payload,
    _known_lyric_hubert_dummy_candidate,
    _known_lyric_hubert_dummy_tail,
    _known_lyric_hubert_opening_evidence,
    _known_lyric_hubert_predecessor_occurrence_status,
    _known_lyric_hubert_pair_tail_current_evidence_scope,
    _known_lyric_hubert_pair_tail_occurrence_status,
    _postcentral_plan_known_lyric_hubert_predecessor_only_requests,
    _known_lyric_hubert_full_line_bridge_is_coherent,
    _make_known_lyric_hubert_current_onset,
)
from postcentral_tail_acquisition import (
    _postcentral_tail_blocked_peer_entries,
    _postcentral_plan_targeted_joint_ctc_tail_requests,
    _postcentral_bundle_with_additional_tail_observations,
    _postcentral_acquire_targeted_joint_ctc_tails,
    _postcentral_acquire_known_lyric_hubert_tails,
)
from hubert_existing_candidate_recovery import (
    _hubert_recovery_boundary_observation_map,
    _hubert_recovery_consensus_candidate,
    _hubert_recovery_project_selection,
    _hubert_recovery_replay_gate,
    _update_projection_seed_for_hubert_recovery,
    _hubert_recovery_isolation_units,
    _precommit_hubert_existing_candidate_recovery,
)


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


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
