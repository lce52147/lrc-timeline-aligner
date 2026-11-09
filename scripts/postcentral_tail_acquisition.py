"""Extracted from postcentral_recovery.py (split2): post-central targeted joint CTC tail acquisition and known-lyric HuBERT tail acquisition."""

from __future__ import annotations

import lrc_stage_timings
import concurrent.futures
import copy
import json
import math
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from local_whisper_retry import CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS, _crossline_retry_helper_cache_key, _retry_candidate_reference, _retry_diagnostic_payload, invoke_crossline_retry
from asr_backends import decode_temp_wav
from report_trust import _RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY, _RUNTIME_POSTCENTRAL_CTC_TAIL_PRODUCER_CONTEXT_KEY, _RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY, _RUNTIME_POSTCENTRAL_HUBERT_BOUNDARY_OBSERVATION_KEY, _RUNTIME_POSTCENTRAL_HUBERT_CURRENT_ONSET_KEY
from timing_core import CrossLineRetryRequest, EvaluatedBackendBundle, IndependentCurrentOnsetEvidence, IndependentTailEvidence, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, PostcentralCtcTailProducerContext, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, TailEvidenceContext, TailEvidenceScope, TailObservationBinding, _make_tail_provenance, _tail_context_is_valid, _tail_provenance_is_valid, _tail_provenance_matches_context, derive_reliable_tail
from lyrics_io import LyricEntry, entry_sung_text
from audio_features import HOP_SECONDS
from runtime_base import DEFAULT_EVIDENCE_WORKERS, LrcError, _path_sha256, _stable_file_sha256, _timing_content_digest, canonicalize_lrc_time, default_whisperx_python, gpu_compute_phase, run_command
from central_commit import _KNOWN_LYRIC_HUBERT_LINEAGE, _known_lyric_hubert_current_onset_is_valid
import argparse
from timing_core import CrossLineRetryOutcome, FinalTimingState
from backend_bundle import _bundle_candidate_sets_payload, _bundle_tail_binding_payload, _bundle_tail_context_payload, _postcentral_rebind_existing_peer_conflicts, _strict_candidate_support_families_for_final_context, _validate_evaluated_backend_bundle
from known_lyric_hubert_planning import (
    _KNOWN_LYRIC_HUBERT_PRODUCER,
    _KNOWN_LYRIC_HUBERT_MODEL_ID,
    _KNOWN_LYRIC_HUBERT_MODEL_REVISION,
    _KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256,
    _KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION,
    _KNOWN_LYRIC_HUBERT_LANGUAGE,
    _KNOWN_LYRIC_HUBERT_AUDIO_LOADER,
    _KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT,
    _KNOWN_LYRIC_HUBERT_HELPER_SOURCE,
    _known_lyric_hubert_helper_revision,
    _postcentral_residual_material_conflict_entries,
    _postcentral_plan_known_lyric_hubert_requests,
    _hubert_probe_window_covers_request_hypotheses,
    _postcentral_plan_known_lyric_hubert_current_probe_requests,
    _postcentral_plan_known_lyric_hubert_timing_failure_diagnostic_requests,
    _known_lyric_hubert_char_diagnostic_payload,
    _known_lyric_hubert_dummy_tail,
    _known_lyric_hubert_opening_evidence,
    _known_lyric_hubert_predecessor_occurrence_status,
    _known_lyric_hubert_pair_tail_current_evidence_scope,
    _known_lyric_hubert_pair_tail_occurrence_status,
    _postcentral_plan_known_lyric_hubert_predecessor_only_requests,
    _known_lyric_hubert_full_line_bridge_is_coherent,
    _make_known_lyric_hubert_current_onset,
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


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
