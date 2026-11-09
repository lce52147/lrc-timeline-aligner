"""Extracted from auto_lrc.py (split step 4): timing dataclasses, candidate/tail evidence, ownership, timing decision and final timing state."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import statistics
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Sequence
import arbiter as timing_arbiter
from multimodel_consensus import model_identity
from lyrics_io import LyricEntry, _RUNTIME_LYRIC_ENTRIES_KEY, entry_sung_text, has_ambiguous_lyric_prefix, lyric_identity_revision, normalize_match_text
from audio_features import HOP_SECONDS
from runtime_base import CanonicalWrittenTime, LrcError, _timing_content_digest, canonicalize_lrc_time, progress_log
from runtime_base import TimingTokenSpan, _timing_spans_from_value
from timing_evidence_types import (
    CTC_SEVERE_INTERNAL_GAP_SECONDS,
    PREFIX_ACOUSTIC_ASSOCIATION_SECONDS,
    TRUSTED_ALIGNMENT_SCORE,
    _RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY,
    _RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY,
    _RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY,
    _RUNTIME_WHISPERX_TAIL_SCOPES_KEY,
    RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
    _WHISPERX_HELPER_PROTOCOL_REVISION,
    EvidenceState,
    OwnershipState,
    TailValidity,
    CandidateAcousticEvidence,
    CandidateOnsetEvidence,
    CandidateOccurrenceEvidence,
    TimingCandidate,
    CandidateRegionEvidence,
    TailRegionEvidence,
    TailEvidenceProvenance,
    TailEvidenceScope,
    TailEvidenceContext,
    WhisperXRuntimeProvenance,
    IndependentTailEvidence,
    IndependentCurrentOnsetEvidence,
    TailObservationSet,
    RuntimeTailEvidence,
    OwnershipRelationEvidence,
    NormalizedTimingEvidence,
    CandidateEvaluation,
    RecoveryProvenance,
    TimingDecision,
    TimingInvariantAudit,
    FinalTimingState,
    TimingFinding,
    EntryTimingFindings,
    CentralTrustSummary,
    EvaluatedBackendTiming,
    PeerGraphPolicy,
    BackendContentIdentityObservation,
    TailObservationBinding,
    EvaluatedBackendBundle,
    ObservationClusterSummary,
    BackendSelection,
    BoundedRetryCapability,
    RetryCandidateReference,
    BaseRetryEpoch,
    PostcentralCtcTailProducerContext,
    CrossLineRetryRequest,
    CrossLineRetryOutcome,
    LocalWhisperCapability,
    LocalWhisperRequest,
    LocalWhisperOutcome,
    _CANDIDATE_OCCURRENCE_MODES,
    _CANDIDATE_OCCURRENCE_DIRECT_PRODUCERS,
    _CANDIDATE_OCCURRENCE_BINDING_KEYS,
    _GENERATION_DIAGNOSTIC_LOCATOR_KEYS,
    _generation_diagnostics_semantic_projection,
    generation_diagnostics_semantic_revision,
)
from tail_occurrence_evidence import (
    _candidate_occurrence_binding_keys_are_canonical,
    _candidate_occurrence_context_indexes,
    _canonical_occurrence_binding_json,
    _candidate_occurrence_evidence_payload,
    _candidate_occurrence_evidence_projection,
    candidate_occurrence_evidence_from_binding,
    candidate_occurrence_evidence_is_valid,
    _candidate_requires_occurrence_evidence,
    _tail_helper_revision,
    _TAIL_BACKEND_LINEAGE_FAMILIES,
    _tail_backend_family,
    _tail_provenance_payload,
    _tail_provenance_is_valid,
    _make_tail_provenance,
    _TAIL_CONTEXT_VALIDITY,
    _tail_context_is_valid,
    _tail_context_is_valid_uncached,
    _tail_provenance_matches_context,
    _tail_contexts_are_valid,
    _whisperx_runtime_provenance_payload,
    _whisperx_runtime_provenance_is_valid,
    _tail_context_revision_values,
    _tail_context_scopes_from_report,
    _tail_evidence_context_from_report,
    _SERIALIZED_SECOND_SCALE,
    _serialized_microseconds,
    _serialized_seconds_equal,
    _serialized_delta_matches,
    _finite_unit_confidence,
    _candidate_confidence,
    _entry_identity_is_unique,
    _minimal_unique_sequence_context,
    _sparse_unique_sequence_anchor_set,
    _sparse_sequence_fingerprint_is_unique,
    _occurrence_binding_has_strong_context,
    _identity_assignment_observation,
    _entry_occurrence_binding,
    _runtime_occurrence_binding_is_valid,
    runtime_independent_tail_evidence,
    _runtime_tail_observation_map,
    _runtime_tail_observation_is_current,
)
from candidate_regions import (
    make_timing_candidate,
    record_alignment_hypothesis,
    _percentile,
    _supported_run_metrics,
    STRICT_DIRECT_ONSET_KINDS,
    STRICT_DIRECT_ONSET_PRODUCERS,
    BOUNDARY_GRADE_DIRECT_ONSET_KINDS,
    candidate_has_bound_acoustic_evidence,
    candidate_has_bound_direct_onset_evidence,
    candidate_has_boundary_grade_direct_onset_evidence,
    candidate_has_independent_backend_identity,
    _ROMAJI_VOWELS,
    _opening_mora_proof_end,
    _CANDIDATE_REGIONS,
    analyze_candidate_regions,
    _analyze_candidate_regions_uncached,
    _tail_region_result,
    derive_reliable_tail,
    _bind_independent_tail_to_candidate,
    _normalize_tail_observations,
    _authoritative_tail_observations,
    resolve_reliable_tail,
    _lineage_families_from_labels,
    _candidate_support_families,
)


def classify_candidate_ownership(
    candidate: TimingCandidate,
    current_regions: CandidateRegionEvidence,
    previous_tail: TailRegionEvidence | None,
) -> OwnershipRelationEvidence:
    status: OwnershipState
    reason: str
    distance: float | None = None
    lineage_relation = _external_tail_current_lineage_relation(
        candidate, previous_tail
    )
    if previous_tail is None:
        status, reason = "clear", "first-entry"
    elif previous_tail.validity != "valid" or previous_tail.reliable_tail_end is None:
        status, reason = "unknown", previous_tail.reason
    elif lineage_relation == "overlap":
        status, reason = "unknown", "external-tail-current-lineage-overlap"
    elif lineage_relation == "unknown":
        status, reason = "unknown", "external-tail-current-lineage-unresolved"
    else:
        distance = candidate.written_time.seconds - previous_tail.reliable_tail_end
        if candidate.written_time.seconds <= previous_tail.reliable_tail_end:
            status, reason = "violation", "candidate-invades-reliable-tail"
        elif candidate.written_time.seconds <= previous_tail.reliable_tail_end + (
            previous_tail.uncertainty_seconds or HOP_SECONDS
        ):
            status, reason = "unknown", "candidate-inside-tail-release-uncertainty"
        else:
            status, reason = "clear", "candidate-after-valid-tail-release"
    payload = {
        "status": status,
        "reason": reason,
        "current_candidate_id": candidate.candidate_id,
        "current_generation_revision": candidate.generation_revision,
        "current_region_revision": current_regions.region_revision,
        "current_centiseconds": candidate.written_time.centiseconds,
        "previous_candidate_id": previous_tail.previous_candidate_id if previous_tail else None,
        "previous_generation_revision": previous_tail.previous_generation_revision if previous_tail else None,
        "previous_region_revision": previous_tail.previous_region_revision if previous_tail else None,
        "previous_tail_evidence_revision": previous_tail.tail_evidence_revision if previous_tail else None,
        "previous_tail_validity": previous_tail.validity if previous_tail else None,
        "previous_tail_authority_backend_lineage": (
            previous_tail.authority_backend_lineage if previous_tail else None
        ),
        "previous_tail_authority_producer": (
            previous_tail.authority_producer if previous_tail else None
        ),
        "previous_tail_authority_observation_revision": (
            previous_tail.authority_observation_revision if previous_tail else None
        ),
        "external_tail_current_lineage_relation": lineage_relation,
        "distance": distance,
    }
    return OwnershipRelationEvidence(
        status=status,
        relation_revision=_timing_content_digest(payload),
        current_candidate_id=candidate.candidate_id,
        current_generation_revision=candidate.generation_revision,
        current_region_revision=current_regions.region_revision,
        current_centiseconds=candidate.written_time.centiseconds,
        previous_candidate_id=previous_tail.previous_candidate_id if previous_tail else None,
        previous_generation_revision=previous_tail.previous_generation_revision if previous_tail else None,
        previous_region_revision=previous_tail.previous_region_revision if previous_tail else None,
        previous_tail_evidence_revision=previous_tail.tail_evidence_revision if previous_tail else None,
        previous_tail_validity=previous_tail.validity if previous_tail else None,
        reliable_tail_start=previous_tail.reliable_tail_start if previous_tail else None,
        reliable_tail_end=previous_tail.reliable_tail_end if previous_tail else None,
        uncertainty_seconds=previous_tail.uncertainty_seconds if previous_tail else None,
        confidence=previous_tail.confidence if previous_tail else None,
        distance_seconds=round(distance, 6) if distance is not None else None,
        reason=reason,
        previous_tail_authority_backend_lineage=(
            previous_tail.authority_backend_lineage if previous_tail else None
        ),
        previous_tail_authority_producer=(
            previous_tail.authority_producer if previous_tail else None
        ),
        previous_tail_authority_observation_revision=(
            previous_tail.authority_observation_revision if previous_tail else None
        ),
    )


def normalize_timing_candidate(
    candidate: TimingCandidate,
    previous_candidate: TimingCandidate | None = None,
    *,
    current_regions: CandidateRegionEvidence | None = None,
    previous_tail: TailRegionEvidence | None = None,
    ownership_relation: OwnershipRelationEvidence | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> NormalizedTimingEvidence:
    spans = candidate.spans
    regions = current_regions or analyze_candidate_regions(candidate)
    if previous_candidate is not None and previous_tail is None:
        previous_tail = derive_reliable_tail(previous_candidate)
    ownership = ownership_relation or classify_candidate_ownership(
        candidate, regions, previous_tail
    )
    bound_acoustic = candidate_has_bound_acoustic_evidence(candidate)
    occurrence_valid = bool(
        not _candidate_requires_occurrence_evidence(candidate)
        or candidate_occurrence_evidence_is_valid(candidate, occurrence_entries)
    )
    bound_direct = candidate_has_bound_direct_onset_evidence(
        candidate, occurrence_entries
    )
    identity_ok = candidate.identity_support == "supported"
    sequence_ok = candidate.sequence_support == "supported"
    # Independent onset evidence proves a boundary, not an arbitrarily collapsed
    # CTC path.  When token spans exist, require local continuation support unless
    # the candidate carries independent full-line content identity (recurrence).
    direct_onset_path_shape_ok = bool(
        not bound_direct
        or not spans
        or candidate.independent_content_identity
        or regions.reliable_support_count >= 3
    )
    direct_onset_path_support = bool(bound_direct and direct_onset_path_shape_ok)
    onset_ok = bool(
        direct_onset_path_shape_ok
        and (regions.onset_proof_end is not None or direct_onset_path_support)
    )
    independent_backend_identity = candidate_has_independent_backend_identity(
        candidate, occurrence_entries
    )
    coherence_ok = bool(
        regions.temporal_coherence == "coherent"
        or (
            independent_backend_identity
            and regions.temporal_coherence == "ambiguous"
        )
    )
    opening_identity_proven = bool(
        identity_ok
        and sequence_ok
        and onset_ok
        and coherence_ok
    )
    # A candidate may use its own spans to establish intrinsic identity,
    # sequence, and opening geometry, but those same spans are not independent
    # evidence against a previous-tail UNKNOWN/VIOLATION relation. Crossing a
    # tail conflict requires separately produced, revision-bound direct-onset
    # evidence.
    boundary_grade_opening = _boundary_grade_opening_can_cross_tail(
        candidate,
        previous_tail=previous_tail,
        occurrence_entries=occurrence_entries,
    )
    independent_ownership = bool(
        boundary_grade_opening and ownership.status != "clear"
    )

    failures: list[str] = []
    if not occurrence_valid:
        failures.append("occurrence-evidence-invalid")
    if candidate.right_edge_pileup:
        failures.append("right-edge-pileup")
    if candidate.window_truncated:
        failures.append("window-truncated")
    if (
        previous_candidate is not None
        and candidate.written_time.seconds
        <= previous_candidate.written_time.seconds + HOP_SECONDS
    ):
        failures.append("nonmonotonic-line-start")
    if ownership.status == "violation" and not boundary_grade_opening:
        failures.append("previous-tail-overlap")
    if ownership.status == "unknown" and not boundary_grade_opening:
        failures.append("previous-tail-ownership-unknown")
    if regions.temporal_coherence == "incoherent":
        failures.append("temporal-incoherence")
    if spans and abs(spans[0].start - candidate.written_time.seconds) > HOP_SECONDS:
        failures.append("candidate-span-revision-mismatch")
    relation_revision_ok = bool(
        ownership.current_candidate_id == candidate.candidate_id
        and ownership.current_generation_revision == candidate.generation_revision
        and ownership.current_region_revision == regions.region_revision
        and ownership.current_centiseconds == candidate.written_time.centiseconds
        and (
            previous_tail is None
            or (
                ownership.previous_candidate_id == previous_tail.previous_candidate_id
                and ownership.previous_generation_revision == previous_tail.previous_generation_revision
                and ownership.previous_region_revision == previous_tail.previous_region_revision
                and ownership.previous_tail_evidence_revision == previous_tail.tail_evidence_revision
            )
        )
    )
    if not relation_revision_ok:
        failures.append("ownership-revision-mismatch")

    ownership_ok = ownership.status == "clear" or boundary_grade_opening
    if candidate.source == "acoustic-only":
        identity_ok = False
    if candidate.source == "duplicate-projection":
        onset_ok = False
    minimum_proof = bool(
        identity_ok
        and sequence_ok
        and occurrence_valid
        and onset_ok
        and ownership_ok
        and coherence_ok
        and relation_revision_ok
    )

    revision = _timing_content_digest(
        {
            "candidate_id": candidate.candidate_id,
            "computed_for_centiseconds": candidate.written_time.centiseconds,
            "generation_revision": candidate.generation_revision,
            "region_revision": regions.region_revision,
            "ownership_relation_revision": ownership.relation_revision,
            "referenced_tail_evidence_revision": ownership.previous_tail_evidence_revision,
            "bound_acoustic": bound_acoustic,
            "bound_direct": bound_direct,
            "minimum_proof": minimum_proof,
            "failures": failures,
        }
    )
    return NormalizedTimingEvidence(
        candidate_id=candidate.candidate_id,
        candidate_revision=revision,
        computed_for_centiseconds=candidate.written_time.centiseconds,
        region_revision=regions.region_revision,
        ownership_relation_revision=ownership.relation_revision,
        referenced_tail_evidence_revision=ownership.previous_tail_evidence_revision,
        prefix_duration=regions.prefix_duration,
        max_internal_gap=regions.max_internal_gap,
        median_internal_gap=regions.median_internal_gap,
        p90_internal_gap=regions.p90_internal_gap,
        p95_internal_gap=regions.p95_internal_gap,
        token_density=regions.token_density,
        longest_supported_run=regions.longest_supported_run,
        opening_supported_run=regions.opening_supported_run,
        later_supported_run=regions.later_supported_run,
        reliable_support_count=regions.reliable_support_count,
        onset_proof_end=regions.onset_proof_end,
        island_count=(len(regions.severe_gap_indexes) + 1 if spans else 0),
        severe_gap_count=len(regions.severe_gap_indexes),
        first_severe_gap_index=regions.severe_gap_indexes[0] if regions.severe_gap_indexes else None,
        temporal_coherence=regions.temporal_coherence,
        fracture_region=regions.fracture_region,
        ownership=ownership,
        tail_distance_seconds=ownership.distance_seconds,
        ownership_independent=independent_ownership,
        structural_failures=tuple(failures),
        minimum_proof_satisfied=minimum_proof,
    )


def evaluate_timing_candidate(
    candidate: TimingCandidate,
    previous_candidate: TimingCandidate | None = None,
    *,
    previous_tail: TailRegionEvidence | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> CandidateEvaluation:
    evidence = normalize_timing_candidate(
        candidate,
        previous_candidate,
        previous_tail=previous_tail,
        occurrence_entries=occurrence_entries,
    )
    reasons = list(evidence.structural_failures)
    if not evidence.minimum_proof_satisfied:
        reasons.append("insufficient-minimum-proof")
    reasons = list(dict.fromkeys(reasons))
    return CandidateEvaluation(candidate, evidence, not reasons, tuple(reasons))


def _candidate_vector(evaluation: CandidateEvaluation) -> tuple[int, int, int, int, float]:
    ownership_rank = {"violation": 0, "unknown": 1, "clear": 2}
    temporal_rank = {"incoherent": 0, "unavailable": 1, "ambiguous": 2, "coherent": 3}
    candidate = evaluation.candidate
    opening_rank = int(
        evaluation.evidence.onset_proof_end is not None
        or candidate_has_bound_direct_onset_evidence(candidate)
    )
    return (
        opening_rank,
        int(candidate.identity_support == "supported" and candidate.sequence_support == "supported"),
        temporal_rank[evaluation.evidence.temporal_coherence],
        ownership_rank[evaluation.evidence.ownership.status],
        candidate.confidence,
    )


def candidate_dominates(left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
    if not left.valid or not right.valid:
        return left.valid and not right.valid
    left_vector = _candidate_vector(left)
    right_vector = _candidate_vector(right)
    dominates = all(a >= b for a, b in zip(left_vector, right_vector)) and any(
        a > b for a, b in zip(left_vector, right_vector)
    )
    if not dominates:
        return False
    # Producer-local confidence values are useful within one lineage, but are
    # not calibrated strongly enough to let a distant candidate from one
    # backend erase an otherwise structurally-equivalent candidate from a
    # disjoint backend.  Preserve confidence ranking everywhere else and let
    # the existing bounded-consensus owner arbitrate this cross-lineage case.
    left_has_extra_authority = (
        candidate_has_boundary_grade_direct_onset_evidence(left.candidate)
        and not candidate_has_boundary_grade_direct_onset_evidence(right.candidate)
    ) or (
        left.candidate.independent_content_identity
        and not right.candidate.independent_content_identity
    )
    if (
        left_vector[:-1] == right_vector[:-1]
        and left_vector[-1] > right_vector[-1]
        and not left_has_extra_authority
    ):
        left_families = _lineage_families_from_labels(
            left.candidate.source,
            source=left.candidate.source,
        )
        right_families = _lineage_families_from_labels(
            right.candidate.source,
            source=right.candidate.source,
        )
        distant = abs(
            left.candidate.written_time.seconds - right.candidate.written_time.seconds
        ) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
        if left_families and right_families and left_families.isdisjoint(right_families) and distant:
            return False
    return True


def _semantically_equivalent_valid_winner(
    evaluations: tuple[CandidateEvaluation, ...],
) -> CandidateEvaluation | None:
    """Select one provenance record when all valid winners mean the same LRC time.

    Multiple independent producers can legitimately prove the same canonical
    boundary.  Treating those proofs as mutually exclusive creates the
    pathological state where *more* evidence causes a provisional fallback.
    This helper never chooses between different timestamps: it only collapses
    an already-equivalent centisecond class and keeps the strongest provenance
    deterministically.
    """
    if not evaluations or any(not item.valid for item in evaluations):
        return None
    centiseconds = {item.candidate.written_time.centiseconds for item in evaluations}
    if len(centiseconds) != 1:
        return None

    def rank(item: CandidateEvaluation) -> tuple[int, int, int, int, int, int, float, str, str]:
        candidate = item.candidate
        vector = _candidate_vector(item)
        return (
            int(candidate_has_boundary_grade_direct_onset_evidence(candidate)),
            int(candidate.independent_content_identity),
            vector[0], vector[1], vector[2], vector[3], vector[4],
            candidate.source, candidate.candidate_id,
        )

    # Numeric evidence ranks are descending.  Lexical fields are only stable
    # provenance tie-breakers after timestamp/evidence equivalence is proven.
    best_rank = max(rank(item)[:-2] for item in evaluations)
    strongest = [item for item in evaluations if rank(item)[:-2] == best_rank]
    return min(strongest, key=lambda item: (item.candidate.source, item.candidate.candidate_id))


def _single_direct_onset_family(candidate: TimingCandidate) -> str | None:
    """Return a known family for the candidate's boundary-grade direct producer."""
    if not candidate_has_boundary_grade_direct_onset_evidence(candidate):
        return None
    evidence = candidate.direct_onset_evidence
    if evidence is None:
        return None
    families = _lineage_families_from_labels(evidence.producer)
    return next(iter(families)) if len(families) == 1 else None


def _current_tail_lineage_relation(
    candidate: TimingCandidate,
    authority_backend_lineage: object,
    authority_producer: object,
    authority_observation_revision: object,
) -> Literal["disjoint", "direct-disjoint", "overlap", "unknown"]:
    """Classify external-tail dependence for one current candidate.

    Broad candidate support is the default dependency boundary.  A candidate's
    existing boundary-grade direct-onset producer may establish a disjoint
    exception, but generic acoustic evidence and unknown/composite producers do
    not manufacture independence.
    """
    metadata = (
        authority_backend_lineage,
        authority_producer,
        authority_observation_revision,
    )
    if not all(isinstance(value, str) and value.strip() for value in metadata):
        return "unknown"
    tail_family = _tail_backend_family(authority_backend_lineage)
    candidate_families = _candidate_support_families(candidate)
    if tail_family is None or not candidate_families:
        return "unknown"
    if tail_family not in candidate_families:
        return "disjoint"
    direct_family = _single_direct_onset_family(candidate)
    if direct_family is not None and direct_family != tail_family:
        return "direct-disjoint"
    return "overlap"


def _external_tail_current_lineage_relation(
    candidate: TimingCandidate,
    previous_tail: TailRegionEvidence | None,
) -> Literal["intrinsic", "disjoint", "direct-disjoint", "overlap", "unknown"] | None:
    if previous_tail is None:
        return None
    metadata = (
        previous_tail.authority_backend_lineage,
        previous_tail.authority_producer,
        previous_tail.authority_observation_revision,
    )
    # No authority metadata denotes intrinsic predecessor geometry.  Only the
    # validated runtime binder can create external metadata-bearing tails.
    if all(value is None for value in metadata):
        return "intrinsic"
    relation = _current_tail_lineage_relation(candidate, *metadata)
    return relation


def _boundary_grade_opening_can_cross_tail(
    candidate: TimingCandidate,
    *,
    previous_tail: TailRegionEvidence | None = None,
    ownership: OwnershipRelationEvidence | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Whether existing direct opening evidence may cross this tail relation."""
    if not candidate_has_boundary_grade_direct_onset_evidence(
        candidate, occurrence_entries
    ):
        return False
    if previous_tail is not None:
        relation = _external_tail_current_lineage_relation(candidate, previous_tail)
    elif ownership is not None:
        metadata = (
            ownership.previous_tail_authority_backend_lineage,
            ownership.previous_tail_authority_producer,
            ownership.previous_tail_authority_observation_revision,
        )
        relation = (
            "intrinsic"
            if all(value is None for value in metadata)
            else _current_tail_lineage_relation(candidate, *metadata)
        )
    else:
        relation = None
    return relation in {None, "intrinsic", "disjoint", "direct-disjoint"}


def _bounded_independent_consensus_winner(
    evaluations: tuple[CandidateEvaluation, ...],
) -> CandidateEvaluation | None:
    """Resolve a unique tight cluster of independently supported valid candidates.

    This closes a conservative gap in the central evaluator: independent producers
    rarely quantize a real acoustic onset to the exact same centisecond. A cluster
    is therefore allowed to converge within the same tolerance already used for
    cross-backend onset consensus. No timestamp is averaged or fabricated; the
    winner is always one existing valid candidate. Competing equally-supported
    clusters remain unresolved.
    """
    valid = tuple(item for item in evaluations if item.valid)
    if len(valid) < 2:
        return None
    tolerance = RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
    ordered = tuple(sorted(
        valid,
        key=lambda item: (item.candidate.written_time.seconds, item.candidate.candidate_id),
    ))
    clusters: list[tuple[CandidateEvaluation, ...]] = []
    for left in range(len(ordered)):
        members: list[CandidateEvaluation] = []
        start = ordered[left].candidate.written_time.seconds
        for item in ordered[left:]:
            if item.candidate.written_time.seconds - start > tolerance:
                break
            members.append(item)
        if len(members) < 2:
            continue
        cluster = tuple(members)
        families = set().union(*(_candidate_support_families(item.candidate) for item in cluster))
        if len(families) < 2:
            continue
        ids = tuple(item.candidate.candidate_id for item in cluster)
        if not any(tuple(existing.candidate.candidate_id for existing in prior) == ids for prior in clusters):
            clusters.append(cluster)
    if not clusters:
        return None

    def cluster_rank(cluster: tuple[CandidateEvaluation, ...]) -> tuple[int, int, int, float]:
        families = set().union(*(_candidate_support_families(item.candidate) for item in cluster))
        boundary_families = set().union(*(
            _candidate_support_families(item.candidate)
            for item in cluster
            if candidate_has_boundary_grade_direct_onset_evidence(item.candidate)
        )) if any(
            candidate_has_boundary_grade_direct_onset_evidence(item.candidate)
            for item in cluster
        ) else set()
        identity_families = set().union(*(
            _candidate_support_families(item.candidate)
            for item in cluster
            if item.candidate.independent_content_identity
        )) if any(item.candidate.independent_content_identity for item in cluster) else set()
        return (
            len(families),
            len(boundary_families),
            len(identity_families),
            max(item.candidate.confidence for item in cluster),
        )

    best_rank = max(cluster_rank(cluster) for cluster in clusters)
    best_clusters = [cluster for cluster in clusters if cluster_rank(cluster) == best_rank]
    # Two equally strong separated clusters are genuine unresolved evidence.
    if len(best_clusters) != 1:
        return None
    cluster = best_clusters[0]
    center = statistics.median(item.candidate.written_time.seconds for item in cluster)

    def winner_rank(item: CandidateEvaluation) -> tuple[int, int, int, int, int, int, float, float, str, str]:
        vector = _candidate_vector(item)
        return (
            int(candidate_has_boundary_grade_direct_onset_evidence(item.candidate)),
            int(item.candidate.independent_content_identity),
            vector[0], vector[1], vector[2], vector[3], vector[4],
            -abs(item.candidate.written_time.seconds - center),
            item.candidate.source,
            item.candidate.candidate_id,
        )

    numeric_best = max(winner_rank(item)[:-2] for item in cluster)
    strongest = [item for item in cluster if winner_rank(item)[:-2] == numeric_best]
    return min(strongest, key=lambda item: (item.candidate.source, item.candidate.candidate_id))


def _record_current_replacement(
    current: CandidateEvaluation,
    winner: CandidateEvaluation,
    *,
    selection_revision: int,
    peer_graph: PeerGraphPolicy | None,
) -> None:
    """Opt-in observation only; never read an oracle or influence selection."""
    diagnostic_path = os.environ.get("LRC_CURRENT_REPLACEMENT_DIAGNOSTICS")
    if not diagnostic_path:
        return
    record = {
        "schema": 1,
        "entry_index": winner.candidate.entry_index,
        "selection_revision": selection_revision,
        "path": "dominance-over-valid-current" if current.valid else "selection-with-invalid-current",
        "peer_mode": peer_graph.mode if peer_graph is not None else None,
        "current": {
            "candidate_id": current.candidate.candidate_id,
            "generation_revision": current.candidate.generation_revision,
            "source": current.candidate.source,
            "current_flag": current.candidate.current,
            "written_centiseconds": current.candidate.written_time.centiseconds,
            "valid": current.valid,
            "reasons": current.rejection_reasons,
            "vector": _candidate_vector(current),
            "families": sorted(_candidate_support_families(current.candidate)),
        },
        "challenger": {
            "candidate_id": winner.candidate.candidate_id,
            "generation_revision": winner.candidate.generation_revision,
            "source": winner.candidate.source,
            "written_centiseconds": winner.candidate.written_time.centiseconds,
            "valid": winner.valid,
            "vector": _candidate_vector(winner),
            "families": sorted(_candidate_support_families(winner.candidate)),
        },
    }
    try:
        with Path(diagnostic_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        progress_log(f"current replacement diagnostics unavailable: {type(exc).__name__}")


def load_reviewer_validity_evidence(
    path: Path | None, row_count: int,
) -> tuple[dict[str, object] | None, ...]:
    empty = tuple(None for _ in range(row_count))
    if path is None or not Path(path).is_file():
        return empty
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(payload, dict):
        return empty
    reviewers = payload.get("reviewers")
    required = payload.get("required_agreements")
    offsets = payload.get("offsets")
    sample_counts = payload.get("offset_sample_counts")
    rows = payload.get("rows")
    if (
        not isinstance(reviewers, list)
        or not isinstance(required, int)
        or required < 1
        or not isinstance(offsets, dict)
        or not isinstance(sample_counts, dict)
        or not isinstance(rows, list)
    ):
        return empty
    qualified_offsets = {
        str(name): float(value)
        for name, value in offsets.items()
        if isinstance(value, (int, float))
        and isinstance(sample_counts.get(str(name)), int)
        and int(sample_counts[str(name)]) >= 8
    }
    result: list[dict[str, object] | None] = [None] * row_count
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("entry"), int):
            continue
        entry = int(row["entry"])
        if not 1 <= entry <= row_count:
            continue
        expected = row.get("expected_current_seconds")
        reviewer_times = row.get("reviewer_times")
        if not isinstance(expected, (int, float)) or not isinstance(reviewer_times, dict):
            continue
        result[entry - 1] = {
            "expected_current_seconds": float(expected),
            "reviewer_times": {
                str(name): (float(value) if isinstance(value, (int, float)) else None)
                for name, value in reviewer_times.items()
            },
            "offsets": qualified_offsets,
            "reviewers": [str(value) for value in reviewers],
            "required_agreements": int(required),
        }
    return tuple(result)


_REVIEWER_VALIDITY_DIAGNOSTICS_KEY = "reviewer_validity_evidence"


def bind_reviewer_validity_diagnostics(
    diagnostics: dict[str, object], args: argparse.Namespace, row_count: int,
) -> dict[str, object]:
    if str(getattr(args, "arbiter", "off")) != "reviewer-validity":
        return diagnostics
    rows = load_reviewer_validity_evidence(
        getattr(args, "reviewer_evidence", None), row_count
    )
    result = dict(diagnostics)
    result[_REVIEWER_VALIDITY_DIAGNOSTICS_KEY] = {
        "schema": 1,
        "rows": list(rows),
    }
    return result


def reviewer_validity_evidence_from_diagnostics(
    diagnostics: Mapping[str, object], row_count: int,
) -> tuple[dict[str, object] | None, ...]:
    payload = diagnostics.get(_REVIEWER_VALIDITY_DIAGNOSTICS_KEY)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        return tuple(None for _ in range(row_count))
    raw_rows = payload["rows"]
    if len(raw_rows) != row_count:
        return tuple(None for _ in range(row_count))
    return tuple(row if isinstance(row, dict) else None for row in raw_rows)


def _reviewer_validity_endorsement(
    candidate: TimingCandidate,
    reviewer_evidence: dict[str, object] | None,
) -> timing_arbiter.ReviewerValidityDecision | None:
    if not isinstance(reviewer_evidence, dict):
        return None
    expected = reviewer_evidence.get("expected_current_seconds")
    raw_times = reviewer_evidence.get("reviewer_times")
    raw_offsets = reviewer_evidence.get("offsets")
    raw_reviewers = reviewer_evidence.get("reviewers")
    required = reviewer_evidence.get("required_agreements")
    if (
        not isinstance(expected, (int, float))
        or not isinstance(raw_times, dict)
        or not isinstance(raw_offsets, dict)
        or not isinstance(raw_reviewers, list)
        or not isinstance(required, int)
    ):
        return None
    reviewer_times = {
        str(key): (float(value) if isinstance(value, (int, float)) else None)
        for key, value in raw_times.items()
    }
    offsets = {
        str(key): float(value)
        for key, value in raw_offsets.items()
        if isinstance(value, (int, float))
    }
    reviewers = tuple(str(value) for value in raw_reviewers)
    return timing_arbiter.decide_reviewer_validity(
        current_seconds=float(candidate.written_time.centiseconds) / 100.0,
        expected_current_seconds=float(expected),
        reviewer_times=reviewer_times,
        offsets=offsets,
        reviewers=reviewers,
        required_agreements=int(required),
    )


def _reviewer_agreement_count(
    candidate: TimingCandidate,
    reviewer_evidence: dict[str, object] | None,
) -> int | None:
    if not isinstance(reviewer_evidence, dict):
        return None
    raw_times = reviewer_evidence.get("reviewer_times")
    raw_offsets = reviewer_evidence.get("offsets")
    raw_reviewers = reviewer_evidence.get("reviewers")
    if (
        not isinstance(raw_times, dict)
        or not isinstance(raw_offsets, dict)
        or not isinstance(raw_reviewers, list)
    ):
        return None
    reviewer_times = {
        str(key): (float(value) if isinstance(value, (int, float)) else None)
        for key, value in raw_times.items()
    }
    offsets = {
        str(key): float(value)
        for key, value in raw_offsets.items()
        if isinstance(value, (int, float))
    }
    agreeing, _present = timing_arbiter.reviewer_agreements_at(
        candidate_seconds=float(candidate.written_time.centiseconds) / 100.0,
        reviewer_times=reviewer_times,
        offsets=offsets,
        reviewers=tuple(str(value) for value in raw_reviewers),
    )
    return len(agreeing)


# Whole-song CTC path collapse.  A forced path that stretches one lyric line
# over >= 12 s, or squeezes a line into <= 0.8 s, is not a plausible sung
# line; it is the signature of the path giving up on a passage (seen as a run
# of compressed lines next to one line absorbing a 16-42 s gap in Choir,
# My Dearest and ミレニア).  Diagnostic geometry only: it never makes any
# candidate valid or trusted.
CTC_COLLAPSED_LINE_MAX_SECONDS = 12.0


CTC_COLLAPSED_LINE_MIN_SECONDS = 0.8


COLLAPSED_PATH_FALLBACK_SOURCE = "ctc-local-window"


COLLAPSED_PATH_FALLBACK_MIN_SHIFT_SECONDS = 0.5


# Wide-region fallback: the radius-3 local-window replay re-aligns a broken
# passage with more sane context than the radius-1 pass.  Its row is only a
# provisional replacement for a row Central left unresolved, and only when the
# radius-2 replay (or another non-posterior candidate) lands on the same onset
# and no independent onset (recurrence, local Whisper) contradicts it.  On the
# 39-song corpus this rule moved 10-22 rows per run closer to BK and none away.
# The replay's own windows are the region: an admitted row is eligible even
# when Central's radius-3 flag on the whole-song path does not reach it (the
# replay windows run to the nearest sane anchors, past that radius).  Behind
# the flag, 726A4413 kept c04 #5, c17 #32/#34/#35 and c34 #7/#21 on their
# broken whole-song times (up to 3.8 s off) while every admitted wide time on
# an unresolved row was closer to BK.
CTC_WIDE_REGION_FALLBACK_SOURCE = "ctc-local-window-wide"


def ctc_line_extent_collapsed(first_start: float, last_end: float) -> bool:
    extent = float(last_end) - float(first_start)
    return (
        extent >= CTC_COLLAPSED_LINE_MAX_SECONDS
        or extent <= CTC_COLLAPSED_LINE_MIN_SECONDS
    )


def _current_timing_candidate(candidates: tuple[TimingCandidate, ...]) -> TimingCandidate | None:
    return next((item for item in candidates if item.current), None)


def _active_legacy_candidate(candidate: object) -> bool:
    return (
        isinstance(candidate, dict)
        and candidate.get("source") != "selected_path"
        and candidate.get("replaced_by_consensus") is not True
    )


def ctc_spans_collapsed(spans: Sequence[TimingTokenSpan]) -> bool:
    if not spans:
        return False
    return ctc_line_extent_collapsed(spans[0].start, spans[-1].end)


def ctc_collapsed_region_flags(
    row_collapsed: Sequence[bool], radius: int = 1
) -> tuple[bool, ...]:
    """A row is inside a collapsed region when it or a row within ``radius``
    collapsed (radius 1: itself or an adjacent row).

    Shared by Central's unresolved fallback and the local-window generator so
    both stages act on exactly the same rows.  Wider radii are used only by the
    posthoc local-window shadow.
    """
    flags = [bool(value) for value in row_collapsed]
    return tuple(
        any(flags[max(0, row - radius):row + radius + 1]) for row in range(len(flags))
    )


def assignment_ctc_path_collapsed(assignment: object) -> bool:
    """Same judgment as ``current_ctc_path_collapsed`` on the backend row the
    ``ctc-current`` candidate is built from."""
    if not isinstance(assignment, dict):
        return False
    return ctc_spans_collapsed(_timing_spans_from_value(assignment.get("ctc_token_spans")))


def current_ctc_path_collapsed(candidates: tuple[TimingCandidate, ...]) -> bool:
    current = _current_timing_candidate(candidates)
    if current is None:
        return False
    return ctc_spans_collapsed(current.spans)


def current_ctc_path_incoherent(candidates: tuple[TimingCandidate, ...]) -> bool:
    """The ``ctc-current`` row fails Central's own temporal-coherence check
    (a severe gap inside its opening), the same reading the evaluator rejects
    it with as ``temporal-incoherence``."""
    current = _current_timing_candidate(candidates)
    if current is None or not current.spans:
        return False
    return analyze_candidate_regions(current).temporal_coherence == "incoherent"


RAW_ORDER_ANCHOR_SOURCE = "raw-vocal-independent-fusion"


def ctc_current_out_of_raw_order(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> tuple[bool, ...]:
    """Rows whose ``ctc-current`` time contradicts the order of independent
    raw-ASR onsets: later than a later row's onset, or earlier than an
    earlier row's.

    The onsets are the raw-ASR identity rows the vocal stem confirmed
    (``raw-vocal-independent-fusion``).  They only diagnose the whole-song
    path here, exactly like collapse and incoherence; they never become a
    selectable clock.  Read before the timing-source policy strips them.
    c39 #20-#26: seven rows piled into 149-177 s while the raw onsets of #20,
    #22, #24 and #26 sit at 139, 146, 162 and 175 s; only #23's own extent
    looked collapsed.
    """
    anchors = _raw_order_anchors(candidate_sets)
    flags: list[bool] = []
    for index, candidates in enumerate(candidate_sets):
        current = _current_timing_candidate(candidates)
        flags.append(
            current is not None
            and _contradicts_raw_order(index, current.written_time.seconds, anchors)
        )
    return tuple(flags)


def _raw_order_anchors(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> tuple[tuple[int, float], ...]:
    return tuple(
        (index, candidate.written_time.seconds)
        for index, candidates in enumerate(candidate_sets)
        for candidate in candidates
        if candidate.source == RAW_ORDER_ANCHOR_SOURCE
    )


def _contradicts_raw_order(
    index: int, seconds: float, anchors: tuple[tuple[int, float], ...]
) -> bool:
    return any(
        (other > index and value <= seconds) or (other < index and value >= seconds)
        for other, value in anchors
    )


def raw_order_contradicting_candidate_ids(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> tuple[frozenset[str], ...]:
    """Per row, the candidates whose time contradicts the independent raw
    onset order (see ``ctc_current_out_of_raw_order``).  Such a time cannot be
    a valid clock: it puts the row after a later row's heard onset or before
    an earlier one's.  It may still be the provisional time when nothing
    better exists.  c39 #23: a retry of the collapsed path at 173.14 s was
    selected_valid although the raw onsets of #24-#26 sit at 162-175 s, and
    as a valid row it barred their correct wide-region replays.
    """
    anchors = _raw_order_anchors(candidate_sets)
    return tuple(
        frozenset(
            candidate.candidate_id for candidate in candidates
            if candidate.source != RAW_ORDER_ANCHOR_SOURCE
            and _contradicts_raw_order(index, candidate.written_time.seconds, anchors)
        )
        for index, candidates in enumerate(candidate_sets)
    )


def _collapsed_path_fallback(
    evaluations: tuple[CandidateEvaluation, ...],
    eligible_ids: set[str],
    current: CandidateEvaluation,
    previous_candidate: TimingCandidate | None,
    next_bound_seconds: float | None,
) -> CandidateEvaluation | None:
    """Provisional replacement for an unresolved row inside a collapsed CTC path
    (or a row whose own whole-song opening is temporally incoherent).

    Only used when Central found no valid winner.  The whole-song path is
    visibly broken around this row, so its timestamp is the least credible
    unresolved option; a known-lyric CTC re-alignment of the same passage in a
    bounded window is preferred.  The row stays provisional_unresolved.
    """
    lower = (
        previous_candidate.written_time.seconds + HOP_SECONDS
        if previous_candidate is not None
        else float("-inf")
    )
    upper = (
        next_bound_seconds - HOP_SECONDS
        if next_bound_seconds is not None
        else float("inf")
    )
    current_seconds = current.candidate.written_time.seconds
    options = [
        evaluation for evaluation in evaluations
        if evaluation.candidate.source == COLLAPSED_PATH_FALLBACK_SOURCE
        and evaluation.candidate.candidate_id in eligible_ids
        and not evaluation.candidate.window_truncated
        and not evaluation.candidate.right_edge_pileup
        and abs(evaluation.candidate.written_time.seconds - current_seconds)
        > COLLAPSED_PATH_FALLBACK_MIN_SHIFT_SECONDS
        and lower < evaluation.candidate.written_time.seconds < upper
    ]
    if not options:
        return None
    return max(
        options,
        key=lambda item: (item.candidate.confidence, item.candidate.candidate_id),
    )


CORROBORATED_REALIGNMENT_SOURCES = ("ctc-local-window", "ctc-opening-local-retry")


CORROBORATED_REALIGNMENT_AGREEMENT_SECONDS = 0.15


CORROBORATED_REALIGNMENT_RAW_ONSET_SECONDS = 0.30


def raw_onset_seconds_by_row(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Each row's independent raw-ASR onsets, read before the timing-source
    policy strips them (evidence only, never a clock)."""
    return tuple(
        tuple(
            candidate.written_time.seconds for candidate in candidates
            if candidate.source == RAW_ORDER_ANCHOR_SOURCE
        )
        for candidates in candidate_sets
    )


def _corroborated_realignment_fallback(
    evaluations: tuple[CandidateEvaluation, ...],
    eligible_ids: set[str],
    current: CandidateEvaluation,
    previous_candidate: TimingCandidate | None,
    next_bound_seconds: float | None,
    raw_onsets: tuple[float, ...],
) -> CandidateEvaluation | None:
    """Provisional replacement for an unresolved row that is not visibly
    collapsed: the window re-alignment and the opening retry are separate CTC
    passes over different audio spans; when both land on the same onset and
    the independent raw onset is there too, that onset outweighs the
    whole-song time.  The clock is still a CTC row; the row stays
    provisional_unresolved.  c20 #24: whole-song 74.57 s; local window 72.76,
    opening retry 72.77, raw onset 72.78 (BK 72.79).
    """
    if not raw_onsets:
        return None
    lower = (
        previous_candidate.written_time.seconds + HOP_SECONDS
        if previous_candidate is not None
        else float("-inf")
    )
    upper = (
        next_bound_seconds - HOP_SECONDS
        if next_bound_seconds is not None
        else float("inf")
    )
    current_seconds = current.candidate.written_time.seconds
    by_source: dict[str, list[CandidateEvaluation]] = {}
    for evaluation in evaluations:
        candidate = evaluation.candidate
        if (
            candidate.source in CORROBORATED_REALIGNMENT_SOURCES
            and candidate.candidate_id in eligible_ids
            and not candidate.window_truncated
            and not candidate.right_edge_pileup
            and lower < candidate.written_time.seconds < upper
            and abs(candidate.written_time.seconds - current_seconds)
            > COLLAPSED_PATH_FALLBACK_MIN_SHIFT_SECONDS
            and any(
                abs(candidate.written_time.seconds - onset)
                <= CORROBORATED_REALIGNMENT_RAW_ONSET_SECONDS
                for onset in raw_onsets
            )
        ):
            by_source.setdefault(candidate.source, []).append(evaluation)
    if len(by_source) < len(CORROBORATED_REALIGNMENT_SOURCES):
        return None
    options = [
        evaluation
        for evaluation in by_source[COLLAPSED_PATH_FALLBACK_SOURCE]
        if any(
            abs(evaluation.candidate.written_time.seconds - other.candidate.written_time.seconds)
            <= CORROBORATED_REALIGNMENT_AGREEMENT_SECONDS
            for source, others in by_source.items()
            if source != COLLAPSED_PATH_FALLBACK_SOURCE
            for other in others
        )
    ]
    if not options:
        return None
    return max(
        options,
        key=lambda item: (item.candidate.confidence, item.candidate.candidate_id),
    )


def _ctc_wide_region_promotions(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    decisions: list[TimingDecision],
) -> dict[int, str]:
    """Rows whose provisional time yields to their gated wide-region replay.

    Walks the finished worklist in order: an unresolved row carrying an
    admitted ``ctc-local-window-wide`` candidate takes its time when that moves
    it by more than the fallback shift and keeps it strictly between the
    (possibly already promoted) previous row and the next row.  The row stays
    provisional_unresolved; only its written clock changes.
    """
    written = [decision.written_time.seconds for decision in decisions]
    forced: dict[int, str] = {}
    for index, decision in enumerate(decisions):
        if decision.status != "provisional_unresolved":
            continue
        wide = next(
            (
                candidate for candidate in candidate_sets[index]
                if candidate.source == CTC_WIDE_REGION_FALLBACK_SOURCE
            ),
            None,
        )
        if wide is None:
            continue
        seconds = wide.written_time.seconds
        lower = written[index - 1] if index else float("-inf")
        upper = (
            decisions[index + 1].written_time.seconds
            if index + 1 < len(decisions)
            else float("inf")
        )
        if (
            abs(seconds - written[index]) > COLLAPSED_PATH_FALLBACK_MIN_SHIFT_SECONDS
            and lower < seconds < upper
        ):
            forced[index] = wide.candidate_id
            written[index] = seconds
    return forced


def select_timing_decision(
    *,
    entry_index: int,
    candidates: tuple[TimingCandidate, ...],
    previous_candidate: TimingCandidate | None,
    selection_revision: int,
    previous_tail: TailRegionEvidence | None = None,
    peer_graph: PeerGraphPolicy | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
    arbiter_mode: Literal["off", "gross-rescue", "reviewer-validity"] = "off",
    reviewer_evidence: dict[str, object] | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
    ctc_path_collapsed: bool = False,
    next_bound_seconds: float | None = None,
    forced_provisional_id: str | None = None,
    raw_order_contradicting_ids: frozenset[str] = frozenset(),
    raw_onsets: tuple[float, ...] = (),
) -> TimingDecision:
    if arbiter_mode not in {"off", "gross-rescue", "reviewer-validity"}:
        raise ValueError(f"unsupported timing arbiter mode: {arbiter_mode}")
    evaluations = tuple(
        evaluate_timing_candidate(
            candidate,
            previous_candidate,
            previous_tail=previous_tail,
            occurrence_entries=occurrence_entries,
        )
        for candidate in sorted(candidates, key=lambda item: item.candidate_id)
    )
    if raw_order_contradicting_ids:
        evaluations = tuple(
            replace(
                evaluation,
                valid=False,
                rejection_reasons=tuple(evaluation.rejection_reasons) + ("contradicts-raw-onset-order",),
            )
            if evaluation.candidate.candidate_id in raw_order_contradicting_ids
            else evaluation
            for evaluation in evaluations
        )
    if timing_source_policy not in {"default", "ctc-first"}:
        raise ValueError(f"unsupported timing-source policy: {timing_source_policy}")
    # Recurrence and the wide-region replay are evidence/fallback rows, never
    # a selectable clock in their own right.
    eligible_ids = {
        item.candidate_id for item in candidates
        if item.source != CTC_WIDE_REGION_FALLBACK_SOURCE
        and (
            (timing_source_policy == "default" and item.source != "mutual-acoustic-recurrence")
            or (item.source != "mutual-acoustic-recurrence" and "whisper" not in item.source.lower() and "raw" not in item.source.lower())
        )
    }
    if not eligible_ids:
        raise LrcError(f"ctc-first: no eligible time candidate at line {entry_index + 1}")
    fallback_current_evaluation = next(
        (
            evaluation for evaluation in evaluations
            if evaluation.candidate.current
            and evaluation.candidate.candidate_id in eligible_ids
        ),
        None,
    )
    current_evaluation = (
        None
        if peer_graph is not None
        and peer_graph.mode == "cross_backend_neutral"
        else fallback_current_evaluation
    )
    reviewer_endorsed_current = False
    reviewer_current_support: int | None = None
    if (
        arbiter_mode == "reviewer-validity"
        and current_evaluation is not None
        and not current_evaluation.valid
    ):
        reviewer_decision = _reviewer_validity_endorsement(
            current_evaluation.candidate, reviewer_evidence
        )
        if (
            reviewer_decision is not None
            and timing_arbiter.reviewer_validity_can_relax(
                current_evaluation.rejection_reasons,
                endorsed=reviewer_decision.endorsed,
            )
        ):
            current_evaluation = replace(
                current_evaluation,
                valid=True,
                rejection_reasons=(),
            )
            reviewer_endorsed_current = True
            reviewer_current_support = len(reviewer_decision.agreeing_reviewers)
            evaluations = tuple(
                current_evaluation
                if evaluation.candidate.candidate_id == current_evaluation.candidate.candidate_id
                else evaluation
                for evaluation in evaluations
            )
    valid = tuple(
        evaluation for evaluation in evaluations
        if evaluation.valid and evaluation.candidate.candidate_id in eligible_ids
    )
    winner: CandidateEvaluation | None = None
    if reviewer_endorsed_current:
        reviewer_challengers = tuple(
            evaluation for evaluation in valid
            if evaluation.candidate.candidate_id != current_evaluation.candidate.candidate_id
        )
        consensus_winner = _bounded_independent_consensus_winner(reviewer_challengers)
        if (
            consensus_winner is not None
            and abs(
                consensus_winner.candidate.written_time.seconds
                - current_evaluation.candidate.written_time.seconds
            ) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            and reviewer_current_support is not None
            and (
                challenger_reviewer_support := _reviewer_agreement_count(
                    consensus_winner.candidate, reviewer_evidence
                )
            ) is not None
            and challenger_reviewer_support > reviewer_current_support
        ):
            winner = consensus_winner
        else:
            winner = current_evaluation
    elif current_evaluation is not None and current_evaluation.valid:
        challengers = tuple(
            evaluation for evaluation in valid
            if evaluation.candidate.candidate_id != current_evaluation.candidate.candidate_id
            and candidate_dominates(evaluation, current_evaluation)
        )
        dominant_challengers = tuple(
            challenger for challenger in challengers
            if all(
                other is challenger or candidate_dominates(challenger, other)
                for other in challengers
            )
        )
        if len(dominant_challengers) == 1:
            winner = dominant_challengers[0]
        else:
            winner = current_evaluation
    elif valid:
        undominated = tuple(
            candidate for candidate in valid
            if not any(
                other.candidate.candidate_id != candidate.candidate.candidate_id
                and candidate_dominates(other, candidate)
                for other in valid
            )
        )
        if len(undominated) == 1:
            winner = undominated[0]
        else:
            winner = _semantically_equivalent_valid_winner(undominated)
            if winner is None:
                # Consensus belongs to the full valid evidence graph.  A support
                # witness may be individually Pareto-dominated by the strongest
                # member of the same tight cluster while still contributing the
                # independent lineage needed to distinguish that cluster from a
                # distant singleton.  Pruning it before consensus discards real
                # evidence.  The winner remains one existing valid candidate; no
                # timestamp is averaged or fabricated.
                winner = _bounded_independent_consensus_winner(valid)

    if (
        winner is not None
        and current_evaluation is not None
        and winner.candidate.candidate_id != current_evaluation.candidate.candidate_id
    ):
        _record_current_replacement(
            current_evaluation, winner,
            selection_revision=selection_revision, peer_graph=peer_graph,
        )

    arbiter_decision: timing_arbiter.GrossRescueDecision | None = None
    if arbiter_mode == "gross-rescue" and current_evaluation is not None:
        central_selection = winner or current_evaluation
        arbiter_decision = timing_arbiter.decide_gross_rescue(
            current_valid=current_evaluation.valid,
            current_centiseconds=current_evaluation.candidate.written_time.centiseconds,
            selected_centiseconds=central_selection.candidate.written_time.centiseconds,
            selected_source=central_selection.candidate.source,
            selected_is_current=(
                winner is not None
                and winner.candidate.candidate_id == current_evaluation.candidate.candidate_id
            ),
        )
        if (
            winner is not None
            and winner.candidate.candidate_id != current_evaluation.candidate.candidate_id
            and arbiter_decision.reverted_to_current
        ):
            winner = current_evaluation

    if winner is None:
        provisional = current_evaluation or fallback_current_evaluation or next(
            evaluation for evaluation in evaluations
            if evaluation.candidate.candidate_id in eligible_ids
        )
        if ctc_path_collapsed and provisional.candidate.current:
            collapsed_fallback = _collapsed_path_fallback(
                evaluations,
                eligible_ids,
                provisional,
                previous_candidate,
                next_bound_seconds,
            )
            if collapsed_fallback is not None:
                provisional = collapsed_fallback
        elif provisional.candidate.current:
            corroborated = _corroborated_realignment_fallback(
                evaluations,
                eligible_ids,
                provisional,
                previous_candidate,
                next_bound_seconds,
                raw_onsets,
            )
            if corroborated is not None:
                provisional = corroborated
        if forced_provisional_id is not None:
            provisional = next(
                (
                    evaluation for evaluation in evaluations
                    if evaluation.candidate.candidate_id == forced_provisional_id
                ),
                provisional,
            )
        status: Literal["selected_valid", "provisional_unresolved"] = "provisional_unresolved"
        selected_id = None
        provisional_id = provisional.candidate.candidate_id
    else:
        provisional = winner
        status = "selected_valid"
        selected_id = winner.candidate.candidate_id
        provisional_id = None

    timestamp_revision = selection_revision
    recovery: RecoveryProvenance | None = None
    if (
        winner is not None
        and not winner.candidate.current
        and current_evaluation is not None
        and not (
            peer_graph is not None
            and peer_graph.mode == "cross_backend_neutral"
        )
    ):
        recovery = RecoveryProvenance(
            candidate_id=winner.candidate.candidate_id,
            candidate_revision=winner.evidence.candidate_revision,
            evidence_revision=selection_revision,
            canonical_written_centiseconds=winner.candidate.written_time.centiseconds,
            old_selection_revision=max(0, selection_revision - 1),
            new_selection_revision=selection_revision,
            resolved_finding_ids=(),
            passed_invariants=(
                "structural-validity",
                "minimum-proof",
                "candidate-dominates-current",
                "revision-consistency",
                "canonical-time-evidence-match",
            ),
            resolution_mode="challenger-recovery",
            evaluator_version="timing-state-v1",
        )
    return TimingDecision(
        entry_index=entry_index,
        selection_revision=selection_revision,
        status=status,
        selected_candidate_id=selected_id,
        provisional_candidate_id=provisional_id,
        written_time=provisional.candidate.written_time,
        timestamp_revision=timestamp_revision,
        timing_evidence_revision=timestamp_revision,
        evidence=provisional.evidence,
        candidate=provisional.candidate,
        rejected_candidates=tuple(
            evaluation for evaluation in evaluations
            if evaluation.candidate.candidate_id != provisional.candidate.candidate_id
        ),
        recovery=recovery,
        arbiter=arbiter_decision,
    )


def global_recurrence_has_independent_corroboration(
    decision: TimingDecision,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Require a nearby, valid, different evidence lineage for recurrence trust.

    Similarity and onset from the same recurrence search are one observation.
    Rejected candidates are retained as diagnostics even under ctc-first.
    """
    if decision.candidate.source != "global-acoustic-recurrence":
        return True
    for evaluation in decision.rejected_candidates:
        witness = evaluation.candidate
        families = _candidate_support_families(witness)
        if (
            evaluation.valid
            and "recurrence" not in families
            and bool(families & {"ctc", "whisper-family", "hubert-ja-known-lyric"})
            and witness.identity_support == "supported"
            and witness.sequence_support == "supported"
            and candidate_has_bound_acoustic_evidence(witness)
            and candidate_has_bound_direct_onset_evidence(witness, occurrence_entries)
            and abs(witness.written_time.seconds - decision.written_time.seconds)
            <= RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
        ):
            return True
    return False


def audit_final_timing_decisions(
    decisions: tuple[TimingDecision, ...],
    unresolved_historical_codes: tuple[tuple[str, ...], ...] | None = None,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
    reviewer_evidence: tuple[dict[str, object] | None, ...] | None = None,
) -> tuple[TimingInvariantAudit, ...]:
    if unresolved_historical_codes is None:
        unresolved_historical_codes = tuple(() for _ in decisions)
    if len(unresolved_historical_codes) != len(decisions):
        raise ValueError("historical timing findings do not match timing decisions")
    if independent_tail_evidence is None:
        independent_tail_evidence = tuple(None for _ in decisions)
    if len(independent_tail_evidence) != len(decisions):
        raise ValueError("independent tail evidence does not match timing decisions")
    if reviewer_evidence is None:
        reviewer_evidence = tuple(None for _ in decisions)
    if len(reviewer_evidence) != len(decisions):
        raise ValueError("reviewer evidence does not match timing decisions")
    trusted = [decision.status == "selected_valid" for decision in decisions]
    failures: list[list[str]] = [[] for _ in decisions]
    for index, decision in enumerate(decisions):
        reviewer_decision = (
            _reviewer_validity_endorsement(decision.candidate, reviewer_evidence[index])
            if decision.candidate.current
            else None
        )
        reviewer_relaxation = bool(
            reviewer_decision is not None
            and reviewer_decision.endorsed
            and set(decision.evidence.structural_failures) <= {"previous-tail-ownership-unknown"}
            and decision.evidence.temporal_coherence != "incoherent"
        )
        if decision.timestamp_revision != decision.timing_evidence_revision:
            trusted[index] = False
            failures[index].append("revision-mismatch")
        if decision.evidence.computed_for_centiseconds != decision.written_time.centiseconds:
            trusted[index] = False
            failures[index].append("canonical-time-evidence-mismatch")
        hard_structural_failures = tuple(
            failure for failure in decision.evidence.structural_failures
            if not (reviewer_relaxation and failure == "previous-tail-ownership-unknown")
        )
        if hard_structural_failures:
            trusted[index] = False
            failures[index].extend(hard_structural_failures)
        strict_direct_onset = candidate_has_bound_direct_onset_evidence(
            decision.candidate, occurrence_entries
        )
        strict_opening_proof = bool(
            decision.evidence.onset_proof_end is not None or strict_direct_onset
        )
        if not strict_opening_proof and not reviewer_relaxation:
            trusted[index] = False
            failures[index].append("no-strict-opening-proof")
        boundary_grade_opening = _boundary_grade_opening_can_cross_tail(
            decision.candidate,
            ownership=decision.evidence.ownership,
            occurrence_entries=occurrence_entries,
        )
        if (
            decision.evidence.ownership.status in {"unknown", "violation"}
            and not boundary_grade_opening
            and not reviewer_relaxation
        ):
            trusted[index] = False
            failures[index].append("tail-conflict-without-independent-current-opening")
        if (
            decision.candidate.direct_onset_support == "supported"
            and not strict_direct_onset
        ):
            trusted[index] = False
            failures[index].append("unbound-or-nonindependent-direct-onset-claim")
        if (
            _candidate_requires_occurrence_evidence(decision.candidate)
            and not candidate_occurrence_evidence_is_valid(
                decision.candidate, occurrence_entries
            )
        ):
            trusted[index] = False
            failures[index].append("occurrence-evidence-invalid")
        if not decision.evidence.minimum_proof_satisfied and not reviewer_relaxation:
            trusted[index] = False
            failures[index].append("insufficient-minimum-proof")
        if not global_recurrence_has_independent_corroboration(
            decision, occurrence_entries
        ):
            trusted[index] = False
            failures[index].append("recurrence-uncorroborated")
        if unresolved_historical_codes[index]:
            trusted[index] = False
            failures[index].extend(unresolved_historical_codes[index])
        expected_tail_revision = None
        if index:
            previous = decisions[index - 1].candidate
            expected_tail_revision = resolve_reliable_tail(
                previous,
                independent_tail_evidence[index - 1],
                tail_evidence_context=tail_evidence_context,
            ).tail_evidence_revision
        if decision.evidence.referenced_tail_evidence_revision != expected_tail_revision:
            trusted[index] = False
            failures[index].append("tail-dependency-revision-mismatch")
    return tuple(
        TimingInvariantAudit(
            entry_index=decision.entry_index,
            timing_trusted=trusted[index],
            status="selected_valid" if trusted[index] else "provisional_unresolved",
            failures=tuple(dict.fromkeys(failures[index])),
        )
        for index, decision in enumerate(decisions)
    )


def build_final_timing_state(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    *,
    worklist_order: Literal["forward", "reverse"] = "forward",
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
    peer_graph: PeerGraphPolicy | None = None,
    occurrence_entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
    arbiter_mode: Literal["off", "gross-rescue", "reviewer-validity"] = "off",
    reviewer_evidence: tuple[dict[str, object] | None, ...] | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> FinalTimingState:
    if worklist_order not in {"forward", "reverse"}:
        raise ValueError(f"unsupported timing worklist order: {worklist_order}")
    out_of_raw_order = ctc_current_out_of_raw_order(candidate_sets)
    raw_order_contradicting = raw_order_contradicting_candidate_ids(candidate_sets)
    raw_onsets = raw_onset_seconds_by_row(candidate_sets)
    candidate_sets = apply_central_timing_source_policy(
        candidate_sets, timing_source_policy
    )
    if any(not candidates for candidates in candidate_sets):
        empty_index = next(index for index, candidates in enumerate(candidate_sets) if not candidates)
        raise ValueError(f"entry {empty_index + 1} has no provisional timing candidate")
    if independent_tail_evidence is None:
        independent_tail_evidence = tuple(None for _ in candidate_sets)
    if len(independent_tail_evidence) != len(candidate_sets):
        raise ValueError("independent tail evidence does not match candidate sets")
    if reviewer_evidence is None:
        reviewer_evidence = tuple(None for _ in candidate_sets)
    if len(reviewer_evidence) != len(candidate_sets):
        raise ValueError("reviewer evidence does not match candidate sets")
    # Collapse is a property of the immutable whole-song CTC baseline, so it is
    # computed once here and every replay through this owner sees the same view.
    collapsed_rows = tuple(current_ctc_path_collapsed(items) for items in candidate_sets)
    collapsed_region = ctc_collapsed_region_flags(collapsed_rows)
    # A whole-song row whose own opening is fractured is the same broken path
    # seen through its shape instead of its extent; its unresolved fallback is
    # the collapsed one.
    incoherent_rows = tuple(current_ctc_path_incoherent(items) for items in candidate_sets)

    def next_bound(row: int) -> float | None:
        # A sane next baseline row bounds the fallback so output stays ordered;
        # a collapsed next row is itself unreliable and is not a bound.
        if row + 1 >= len(candidate_sets) or collapsed_rows[row + 1]:
            return None
        following = _current_timing_candidate(candidate_sets[row + 1])
        return following.written_time.seconds if following is not None else None

    def run_worklist(forced: dict[int, str]) -> list[TimingDecision]:
        decisions: list[TimingDecision | None] = [None] * len(candidate_sets)
        pending = list(range(len(candidate_sets)))
        if worklist_order == "reverse":
            pending.reverse()
        while pending:
            index = pending.pop(0)
            if index and decisions[index - 1] is None:
                pending.append(index)
                continue
            previous_candidate = decisions[index - 1].candidate if index else None  # type: ignore[union-attr]
            previous_tail = (
                resolve_reliable_tail(
                    previous_candidate,
                    independent_tail_evidence[index - 1],
                    tail_evidence_context=tail_evidence_context,
                )
                if index and previous_candidate is not None
                else None
            )
            new_decision = select_timing_decision(
                entry_index=index,
                candidates=candidate_sets[index],
                previous_candidate=previous_candidate,
                selection_revision=index + 1,
                previous_tail=previous_tail,
                peer_graph=peer_graph,
                occurrence_entries=occurrence_entries,
                arbiter_mode=arbiter_mode,
                reviewer_evidence=reviewer_evidence[index],
                timing_source_policy=timing_source_policy,
                ctc_path_collapsed=(
                    collapsed_region[index] or incoherent_rows[index] or out_of_raw_order[index]
                ),
                next_bound_seconds=next_bound(index),
                forced_provisional_id=forced.get(index),
                raw_order_contradicting_ids=raw_order_contradicting[index],
                raw_onsets=raw_onsets[index],
            )
            old_decision = decisions[index]
            decisions[index] = new_decision
            changed = bool(
                old_decision is None
                or old_decision.candidate.candidate_id != new_decision.candidate.candidate_id
                or old_decision.evidence.candidate_revision != new_decision.evidence.candidate_revision
            )
            if changed and index + 1 < len(candidate_sets) and (index + 1) not in pending:
                pending.append(index + 1)
        return [decision for decision in decisions if decision is not None]

    decisions = run_worklist({})
    forced = _ctc_wide_region_promotions(candidate_sets, decisions)
    while forced:
        promoted = run_worklist(forced)
        # A promotion must survive its own replay: still unresolved, still the
        # forced row, still strictly ordered against the replayed neighbours.
        dropped = {
            index for index, candidate_id in forced.items()
            if promoted[index].candidate.candidate_id != candidate_id
            or (
                index
                and promoted[index - 1].written_time.centiseconds
                >= promoted[index].written_time.centiseconds
            )
            or (
                index + 1 < len(promoted)
                and promoted[index + 1].written_time.centiseconds
                <= promoted[index].written_time.centiseconds
            )
        }
        if not dropped:
            decisions = promoted
            break
        forced = {index: value for index, value in forced.items() if index not in dropped}
    frozen_decisions = tuple(decisions)
    audits = audit_final_timing_decisions(
        frozen_decisions,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        occurrence_entries=occurrence_entries,
        reviewer_evidence=reviewer_evidence,
    )
    return FinalTimingState(
        decisions=frozen_decisions,
        written_times=tuple(decision.written_time for decision in frozen_decisions),
        audits=audits,
    )


def apply_central_timing_source_policy(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    policy: Literal["default", "ctc-first"] = "default",
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Return the candidates allowed to participate in Central timing choice.

    Full generation diagnostics remain available outside the Central decision
    graph. Under ctc-first, Whisper/RAW timings are evidence-only: they may
    contribute bound row/tail diagnostics, but they cannot become timing
    alternatives, rejected material hypotheses, or REVIEW owners.
    """
    if policy == "default":
        return candidate_sets
    if policy != "ctc-first":
        raise ValueError(f"unsupported timing-source policy: {policy}")
    filtered: list[tuple[TimingCandidate, ...]] = []
    for line, candidates in enumerate(candidate_sets, 1):
        eligible = tuple(
            candidate for candidate in candidates
            if candidate.source != "mutual-acoustic-recurrence"
            and "whisper" not in candidate.source.lower()
            and "raw" not in candidate.source.lower()
        )
        if not eligible:
            raise LrcError(f"ctc-first: no eligible time candidate at line {line}")
        filtered.append(eligible)
    return tuple(filtered)
