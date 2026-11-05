"""Extracted from auto_lrc.py (split phase 2): neutral boundary arbitration: neutral conflict probes, support families and arbitration of neutral boundaries before commit."""

from __future__ import annotations

import collections
import concurrent.futures
import copy
import math
from pathlib import Path
from local_whisper_retry import LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS, LOCAL_WHISPER_MAX_WINDOW_SECONDS, _local_whisper_cache_key, _local_whisper_execution_request_revision, _prepare_local_whisper_batch, execute_local_whisper_request
from asr_backends import build_local_whisper_capability
from acoustic_hypotheses import assert_report_timestamp_equality
from report_trust import _RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY, _UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY, _validate_backend_provenance_projection, _validate_unresolved_evidence_requirements_projection, assert_report_review_contract, thaw_generation_diagnostics, update_report_confidence_metrics
from timing_core import BackendSelection, EvaluatedBackendTiming, LocalWhisperRequest, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, TimingCandidate, _CANDIDATE_OCCURRENCE_MODES, _entry_identity_is_unique, _minimal_unique_sequence_context, candidate_occurrence_evidence_from_binding, make_timing_candidate
from lyrics_io import LyricEntry, _BACKEND_PROVENANCE_PROJECTION_KEY, entry_sung_text
from runtime_base import DEFAULT_EVIDENCE_WORKERS, LrcError, _timing_content_digest, canonicalize_lrc_time, progress_log
from central_commit import evaluate_backend_timing
import argparse
from typing import Mapping
from timing_core import FinalTimingState, LocalWhisperCapability, LocalWhisperOutcome
from backend_bundle import _build_unresolved_evidence_requirements_projection, _neutral_arbitration_support_families, central_backend_summary_payload, central_candidate_selection_payload, central_selection_reason, select_evaluated_backend
from central_commit_tail import _apply_evaluated_backend_projection, _finalize_backend_provenance_projection
import lrc_stage_timings


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


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
