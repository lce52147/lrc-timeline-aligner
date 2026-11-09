"""Extracted from postcentral_recovery.py (split2): pre-commit HuBERT existing-candidate recovery with its projection, replay-gate and isolation helpers."""

from __future__ import annotations

import lrc_stage_timings
import copy
from acoustic_hypotheses import assert_report_timestamp_equality
from report_trust import _RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY, _RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY, _RUNTIME_POSTCENTRAL_HUBERT_BOUNDARY_OBSERVATION_KEY, _RUNTIME_POSTCENTRAL_HUBERT_DUPLICATE_RESOLUTION_KEY, assert_report_review_contract, thaw_generation_diagnostics, update_report_confidence_metrics
from timing_core import BackendSelection, EvaluatedBackendBundle, EvaluatedBackendTiming, IndependentCurrentOnsetEvidence, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, TimingCandidate, make_timing_candidate
from lyrics_io import LyricEntry, entry_sung_text
from runtime_base import LrcError, _timing_content_digest, canonicalize_lrc_time
from central_commit import _KNOWN_LYRIC_HUBERT_ONSET_PRODUCER, _known_lyric_hubert_current_onset_is_valid, _runtime_hubert_duplicate_occurrence_resolution, evaluate_backend_timing
from typing import Mapping
from timing_core import FinalTimingState
from backend_bundle import _HUBERT_EXISTING_CANDIDATE_RECOVERY_KEY, _bundle_candidate_lookup, _neutral_arbitration_support_families, _postcentral_hubert_observation_matches_row_hypotheses, _postcentral_rebind_existing_peer_conflicts, central_backend_summary_payload, central_candidate_selection_payload, central_selection_reason
from central_commit_tail import _apply_evaluated_backend_projection, _postcentral_existing_conflict_projection
from neutral_arbitration import _NEUTRAL_ARBITRATION_FINAL_SEMANTIC_FIELDS, _NEUTRAL_ARBITRATION_HARD_COLLATERAL_FIELDS, _neutral_arbitration_candidate_map, _neutral_arbitration_final_semantic_rows, _neutral_arbitration_selection_with_revised_eval, _neutral_arbitration_valid_candidate_ids
from known_lyric_hubert_planning import (
    _HUBERT_EXISTING_CANDIDATE_RECOVERY_MODE,
    _HUBERT_RECOVERY_DIRECT_ONSET_KIND,
)


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


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
