"""Extracted from auto_lrc.py (split phase 2): backend bundles: evaluated backend bundles, peer graph simulation and cross-backend conflicts, hubert tail acquisition helpers, provenance projection."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from runtime_evidence_binding import _freeze_runtime_tail_observation_map, bind_runtime_ctc_assignment_tail_fallback
from asr_backends import _whispercpp_runtime_artifact_hashes, build_local_whisper_capability
from report_trust import ACTIONABLE_SUSPICIOUS_FLAGS, _RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY, _RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICT_DETAILS_KEY, _RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY, _RUNTIME_RAW_CONTENT_IDENTITY_KEY, _RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY, _RUNTIME_WHISPERX_ROW_EVIDENCE_KEY, _UNRESOLVED_EVIDENCE_CATEGORY_VALUES, _central_summary_row_flags, _runtime_peer_review_only_conflicts, summarize_central_timing_state, timing_report_evaluation_revision
from timing_core import BackendContentIdentityObservation, BackendSelection, CandidateEvaluation, EvaluatedBackendBundle, EvaluatedBackendTiming, IndependentCurrentOnsetEvidence, IndependentTailEvidence, ObservationClusterSummary, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, PeerGraphPolicy, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, TailEvidenceContext, TailObservationBinding, TimingCandidate, _RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY, _RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY, _bind_independent_tail_to_candidate, _bounded_independent_consensus_winner, _candidate_occurrence_evidence_projection, _candidate_support_families, _lineage_families_from_labels, _runtime_tail_observation_map, _semantically_equivalent_valid_winner, _tail_backend_family, _tail_context_is_valid, _tail_contexts_are_valid, _tail_evidence_context_from_report, _tail_provenance_is_valid, _tail_provenance_matches_context, _tail_provenance_payload, build_final_timing_state, candidate_dominates, candidate_has_bound_direct_onset_evidence, derive_reliable_tail, evaluate_timing_candidate
from lyrics_io import LRC_ALGORITHM_REVISION, LyricEntry, _RUNTIME_LYRIC_ENTRIES_KEY, entry_sung_text, lyric_identity_revision, normalize_match_text
from runtime_base import LrcError, _path_sha256, _stable_file_sha256, _timing_content_digest, canonicalize_lrc_time, validated_loaded_source_sha256
from central_commit import _candidate_evidence_projection, _candidate_is_same_occurrence_boundary_equivalent, _known_lyric_hubert_current_onset_is_valid, _material_distant_timing_alternatives, _postcentral_hubert_current_onset_map, _runtime_raw_timing_drift_falsified_candidate_ids, evaluate_final_timing_state_for_report
import argparse
from typing import Literal
from timing_core import CentralTrustSummary, FinalTimingState, RuntimeTailEvidence, TailRegionEvidence, TimingDecision, TimingInvariantAudit


_UNRESOLVED_EVIDENCE_REQUIREMENTS_SCHEMA_VERSION = 1


_BACKEND_PEER_SIMULATION_SCHEMA_VERSION = 2


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


_HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY = "hubert_known_lyric_existing_candidate_recovery"


_PEER_EXISTING_CANDIDATE_RECOVERY_KEY = "independent_peer_existing_candidate_recovery"
