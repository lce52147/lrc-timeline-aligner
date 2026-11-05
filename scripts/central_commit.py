"""Extracted from auto_lrc.py (split phase 2): central evaluation and commit: preview and evaluation, final timing state, projection commit, runtime hubert onset helpers."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Literal
from local_whisper_retry import collect_central_timing_candidates, independent_tail_evidence_payload
from asr_backends import ctc_alignment_audio_path
from acoustic_hypotheses import apply_global_acoustic_recurrence_hypotheses, apply_mutual_acoustic_recurrence_hypotheses, apply_phonetic_onset_fusion_hypotheses, apply_post_score_audio_validation, apply_short_duplicate_acoustic_recovery, apply_trusted_tail_release_onset_hypotheses, apply_vocal_activity_reentry_hypotheses, apply_vocal_onset_confirmation_hypotheses, duplicate_lyric_offset_consensus, score_line_timing_candidates
from report_trust import ACTIONABLE_SUSPICIOUS_FLAGS, CTC_BOUNDARY_REVIEW_FLAGS, _RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY, _RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY, _RUNTIME_POSTCENTRAL_HUBERT_DUPLICATE_RESOLUTION_KEY, _RUNTIME_RAW_TIMING_DRIFT_FALSIFICATIONS_KEY, _actionable_suspicious_severity_counts, _ctc_disagreement_severity, _raw_disagreement_severity, _runtime_peer_review_only_conflicts, _suspicious_row_has_independent_high_severity_owner, _suspicious_row_requires_review, _timing_failure_requires_review, apply_anchor_hints, assignment_content_is_trusted, candidate_supplies_independent_content_identity, central_evaluated_quality, freeze_generation_diagnostics, runtime_independent_content_identity_sources_for_entry, self_check_flags_require_review, summarize_central_timing_state, timing_report_evaluation_revision
from timing_core import CandidateEvaluation, EntryTimingFindings, EvaluatedBackendTiming, FinalTimingState, IndependentCurrentOnsetEvidence, NormalizedTimingEvidence, PeerGraphPolicy, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, RuntimeTailEvidence, TRUSTED_ALIGNMENT_SCORE, TailEvidenceContext, TimingCandidate, TimingDecision, TimingFinding, TimingInvariantAudit, _candidate_occurrence_evidence_projection, _candidate_support_families, _current_tail_lineage_relation, _tail_evidence_context_from_report, audit_final_timing_decisions, bind_reviewer_validity_diagnostics, build_final_timing_state, candidate_has_bound_direct_onset_evidence, candidate_has_boundary_grade_direct_onset_evidence, candidate_occurrence_evidence_is_valid, generation_diagnostics_semantic_revision, reviewer_validity_evidence_from_diagnostics, runtime_independent_tail_evidence
from lyrics_io import LyricEntry, _RUNTIME_LYRIC_ENTRIES_KEY, entry_sung_text, normalize_match_text
from audio_features import analyze_vocal_onsets, vocal_stem_levels
from runtime_base import LrcError, _timing_content_digest, canonicalize_lrc_time, default_ctc_python, progress_log, validated_loaded_source_sha256
from runtime_base import UnboundNamespace, bind_late_callbacks as _bind_late_callbacks
import lrc_stage_timings
import report_trust


_LATE_BOUND_CALLBACKS = (
    "_AUDIO_API",
)


# Back-edges into later modules: the owner binds the real objects once they exist.
_AUDIO_API = UnboundNamespace("_AUDIO_API")


def bind_late_callbacks(**callbacks):
    _bind_late_callbacks(globals(), _LATE_BOUND_CALLBACKS, callbacks)



RAW_TIMING_DRIFT_MAX_MEDIAN_DELTA_SECONDS = 2.0


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


# Post-Central third acoustic lineage: exact known-lyric Japanese HuBERT forced
# alignment.  This producer is deliberately separate from both the Whisper /
# WhisperX-Wav2Vec2 family and the MMS-CTC family already used by production.
# It can emit predecessor terminal geometry plus a runtime-only, occurrence-bound
# current-opening observation.  Neither path can admit or select a candidate.
_KNOWN_LYRIC_HUBERT_LINEAGE = "hubert-ja-known-lyric"


_KNOWN_LYRIC_HUBERT_ONSET_PRODUCER = "hubert-ja-known-lyric-current-onset"


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


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)

# report_trust's back-edges into this module: bound here, after this owner's
# stage-timing install, so every entry path that imports central_commit has them
# (as the final bindings) before any report_trust caller runs.
report_trust.bind_late_callbacks(
    _final_post_central_material_conflict_entries=_final_post_central_material_conflict_entries,
    _suspicious_entry_requires_review_for_decision=_suspicious_entry_requires_review_for_decision,
)
