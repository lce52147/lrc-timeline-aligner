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


_KNOWN_LYRIC_HUBERT_PRODUCER = "hubert-ja-known-lyric-terminal"


_KNOWN_LYRIC_HUBERT_MODEL_ID = "TKU410410103/hubert-base-japanese-asr"


_KNOWN_LYRIC_HUBERT_MODEL_REVISION = "9206637a0f07d78aa9101ff9439a9d7fe1a43778"


_KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256 = "d6f5034777c0542db6c587d57db497206d0500f7eca8fb8bc7962301f45d54d9"


_KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION = "known-lyric-hubert-joint-terminal-v7"


_KNOWN_LYRIC_HUBERT_LANGUAGE = "ja-hiragana"


_KNOWN_LYRIC_HUBERT_AUDIO_LOADER = "python-wave-pcm16-mono-16k-v1"


_KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT = "embedded-viterbi-token-emission-geometry-group-confidence-hypothesis-neutral-v5"


_HUBERT_EXISTING_CANDIDATE_RECOVERY_MODE = (
    "precommit-hubert-existing-candidate-central-recovery-v1"
)


_PEER_EXISTING_CANDIDATE_RECOVERY_MODE = (
    "precommit-independent-peer-existing-candidate-central-recovery-v1"
)


_PEER_DUPLICATE_RECOMPUTE_KEY = "postcentral_duplicate_occurrence_offset_recompute"


_HUBERT_RECOVERY_DIRECT_ONSET_KIND = "known-lyric-onset"


_KNOWN_LYRIC_HUBERT_HELPER_SOURCE = '#!/usr/bin/env python3\nfrom __future__ import annotations\nimport argparse\nimport array\nfrom dataclasses import dataclass\nimport hashlib\nimport importlib.metadata\nimport json\nimport math\nimport sys\nimport unicodedata\nimport wave\nfrom pathlib import Path\n\nEXPECTED_MODEL_ID = "__LRC_KNOWN_LYRIC_HUBERT_MODEL_ID__"\nEXPECTED_MODEL_COMMIT = "__LRC_KNOWN_LYRIC_HUBERT_MODEL_REVISION__"\nEXPECTED_MODEL_BLOB_SHA256 = "__LRC_KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256__"\nPROTOCOL_REVISION = "__LRC_KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION__"\nSAMPLE_RATE = 16000\nMIN_INPUT_SAMPLES = 400\n\n\ndef sha256_file(path: Path) -> str:\n    h = hashlib.sha256()\n    with path.open("rb") as f:\n        for block in iter(lambda: f.read(1024 * 1024), b""):\n            h.update(block)\n    return h.hexdigest()\n\n\ndef locate_model_blob() -> Path | None:\n    candidates: list[Path] = []\n    try:\n        from huggingface_hub import try_to_load_from_cache\n        value = try_to_load_from_cache(\n            EXPECTED_MODEL_ID, "model.safetensors", revision=EXPECTED_MODEL_COMMIT\n        )\n        if isinstance(value, str):\n            candidates.append(Path(value))\n    except Exception:\n        pass\n    try:\n        from huggingface_hub.constants import HF_HUB_CACHE\n        repo = EXPECTED_MODEL_ID.replace("/", "--")\n        candidates.append(\n            Path(HF_HUB_CACHE) / f"models--{repo}" / "snapshots" /\n            EXPECTED_MODEL_COMMIT / "model.safetensors"\n        )\n    except Exception:\n        pass\n    for path in candidates:\n        try:\n            if path.is_file() and sha256_file(path) == EXPECTED_MODEL_BLOB_SHA256:\n                return path\n        except OSError:\n            continue\n    return None\n\n\ndef load_pcm16_mono_16k(path: Path):\n    import torch\n    with wave.open(str(path), "rb") as wf:\n        if wf.getnchannels() != 1:\n            raise ValueError(f"wav-channel-count:{wf.getnchannels()}")\n        if wf.getframerate() != SAMPLE_RATE:\n            raise ValueError(f"wav-sample-rate:{wf.getframerate()}")\n        if wf.getsampwidth() != 2:\n            raise ValueError(f"wav-sample-width:{wf.getsampwidth()}")\n        if wf.getcomptype() != "NONE":\n            raise ValueError(f"wav-compression:{wf.getcomptype()}")\n        raw = wf.readframes(wf.getnframes())\n    values = array.array("h")\n    values.frombytes(raw)\n    if sys.byteorder != "little":\n        values.byteswap()\n    if not values:\n        raise ValueError("wav-empty")\n    return torch.tensor(values, dtype=torch.float32).div_(32768.0).unsqueeze(0)\n\n\ndef clean_reading(text: str, vocab: dict[str, int]) -> str:\n    normalized = unicodedata.normalize("NFKC", text)\n    clean: list[str] = []\n    for char in normalized:\n        if char.isspace():\n            if "|" in vocab:\n                clean.append("|")\n            continue\n        if char in vocab:\n            clean.append(char)\n            continue\n        if unicodedata.category(char).startswith("P"):\n            continue\n        raise ValueError(f"unsupported-reading-character:{ord(char):04x}")\n    return "".join(clean)\n\n\ndef resolve_device(requested: str, cuda_available: bool) -> tuple[str, str]:\n    requested_device = str(requested).strip() or "auto"\n    if requested_device.lower() == "auto":\n        resolved_device = "cuda" if cuda_available else "cpu"\n    else:\n        resolved_device = requested_device\n    return requested_device, resolved_device\n\n\ndef blank_id(processor, vocab: dict[str, int]) -> int:\n    token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)\n    if isinstance(token_id, int) and token_id >= 0:\n        return token_id\n    for token in ("<pad>", "[PAD]", "[pad]"):\n        if token in vocab:\n            return int(vocab[token])\n    raise ValueError("blank-token-unavailable")\n\n\n@dataclass(frozen=True)\nclass Point:\n    token_index: int\n    time_index: int\n    score: float\n    emitted: bool\n\n\n@dataclass(frozen=True)\nclass Segment:\n    label: str\n    start: int\n    end: int\n    score: float\n\n\ndef get_trellis(emission, tokens: list[int], blank: int):\n    import torch\n    num_frame = emission.size(0)\n    num_tokens = len(tokens)\n    if num_tokens <= 0 or num_frame < num_tokens:\n        raise ValueError("ctc-insufficient-frames")\n    trellis = torch.empty((num_frame + 1, num_tokens + 1))\n    trellis[0, 0] = 0\n    trellis[1:, 0] = torch.cumsum(emission[:, blank], 0)\n    trellis[0, -num_tokens:] = -float("inf")\n    trellis[-num_tokens:, 0] = float("inf")\n    for t in range(num_frame):\n        trellis[t + 1, 1:] = torch.maximum(\n            trellis[t, 1:] + emission[t, blank],\n            trellis[t, :-1] + emission[t, tokens],\n        )\n    return trellis\n\n\ndef backtrack(trellis, emission, tokens: list[int], blank: int):\n    j = trellis.size(1) - 1\n    t_start = int(trellis[:, j].argmax().item())\n    path: list[Point] = []\n    for t in range(t_start, 0, -1):\n        stayed = trellis[t - 1, j] + emission[t - 1, blank]\n        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]\n        use_change = bool(changed > stayed)\n        token_id = tokens[j - 1] if use_change else blank\n        prob = float(emission[t - 1, token_id].exp().item())\n        path.append(Point(j - 1, t - 1, prob, use_change))\n        if use_change:\n            j -= 1\n        if j == 0:\n            break\n    if j != 0:\n        return None\n    return path[::-1]\n\n\ndef merge_repeats(path: list[Point], transcript: str) -> list[Segment]:\n    # Geometry and support confidence are intentionally separated.  The actual\n    # token-emission frame owns the conservative character bounds, while the\n    # full contiguous CTC path group for that transcript token owns confidence.\n    # Blank-stay frames therefore remain visible as positive inter-token gaps\n    # without throwing away the support score semantics used by the frozen tail\n    # reliability policy.\n    i1 = i2 = 0\n    segments: list[Segment] = []\n    expected_index = 0\n    while i1 < len(path):\n        while i2 < len(path) and path[i1].token_index == path[i2].token_index:\n            i2 += 1\n        group = path[i1:i2]\n        emitted = [point for point in group if point.emitted]\n        if len(emitted) != 1:\n            raise ValueError("ctc-token-emission-cardinality-mismatch")\n        point = emitted[0]\n        if point.token_index != expected_index or expected_index >= len(transcript):\n            raise ValueError("ctc-token-emission-order-mismatch")\n        support_score = sum(item.score for item in group) / len(group)\n        segments.append(Segment(\n            transcript[expected_index],\n            point.time_index,\n            point.time_index + 1,\n            support_score,\n        ))\n        expected_index += 1\n        i1 = i2\n    if expected_index != len(transcript):\n        raise ValueError("ctc-token-emission-count-mismatch")\n    return segments\n\n\ndef align_chars(emission, transcript: str, vocab: dict[str, int], blank: int, start: float, end: float):\n    tokens = [int(vocab[char]) for char in transcript]\n    trellis = get_trellis(emission, tokens, blank)\n    path = backtrack(trellis, emission, tokens, blank)\n    if path is None:\n        raise ValueError("backtrack-unavailable")\n    merged = merge_repeats(path, transcript)\n    if len(merged) != len(transcript):\n        raise ValueError("incomplete-character-alignment")\n    frame_count = emission.size(0)\n    if frame_count <= 0:\n        raise ValueError("invalid-emission-frame-count")\n    ratio = (end - start) / frame_count\n    chars: list[dict[str, object]] = []\n    for item in merged:\n        char_start = float(item.start) * ratio + start\n        char_end = float(item.end) * ratio + start\n        score = float(item.score)\n        if not all(math.isfinite(v) for v in (char_start, char_end, score)):\n            raise ValueError("nonfinite-character-geometry")\n        if char_end < char_start:\n            raise ValueError("negative-character-duration")\n        chars.append({\n            "char": item.label,\n            "start": round(char_start, 6),\n            "end": round(char_end, 6),\n            "score": round(score, 8),\n        })\n    return chars, ratio\n\n\ndef main() -> int:\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--audio", required=True)\n    ap.add_argument("--request", required=True)\n    ap.add_argument("--output", required=True)\n    ap.add_argument("--device", default="cuda")\n    args = ap.parse_args()\n\n    import torch\n    from transformers import AutoModelForCTC, AutoProcessor\n\n    try:\n        transformers_version = importlib.metadata.version("transformers")\n        huggingface_hub_version = importlib.metadata.version("huggingface-hub")\n        torch_version = importlib.metadata.version("torch")\n    except importlib.metadata.PackageNotFoundError as exc:\n        raise SystemExit(f"runtime-package-unavailable:{exc.name}")\n    requested_device, resolved_device = resolve_device(\n        args.device, bool(torch.cuda.is_available())\n    )\n    if resolved_device.startswith("cuda") and not torch.cuda.is_available():\n        raise SystemExit("cuda-unavailable")\n\n    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))\n    if payload.get("protocol_revision") != PROTOCOL_REVISION:\n        raise SystemExit("protocol-revision-mismatch")\n    requests = payload.get("requests")\n    if not isinstance(requests, list):\n        raise SystemExit("malformed-request-list")\n\n    audio_t = load_pcm16_mono_16k(Path(args.audio))\n    processor = AutoProcessor.from_pretrained(\n        EXPECTED_MODEL_ID, revision=EXPECTED_MODEL_COMMIT\n    )\n    model = AutoModelForCTC.from_pretrained(\n        EXPECTED_MODEL_ID, revision=EXPECTED_MODEL_COMMIT\n    ).to(resolved_device)\n    model.eval()\n    model_blob = locate_model_blob()\n    if model_blob is None:\n        raise SystemExit("model-blob-revision-unavailable")\n\n    tokenizer = getattr(processor, "tokenizer", None)\n    if tokenizer is None or not hasattr(tokenizer, "get_vocab"):\n        raise SystemExit("tokenizer-vocab-unavailable")\n    vocab = {str(k): int(v) for k, v in tokenizer.get_vocab().items()}\n    blank = blank_id(processor, vocab)\n    vocab_revision = hashlib.sha256(\n        json.dumps(vocab, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")\n    ).hexdigest()\n    max_duration = float(audio_t.shape[1]) / SAMPLE_RATE\n    results: list[dict[str, object]] = []\n    for req in requests:\n        request_kind = str(req.get("request_kind", "pair-tail"))\n        result: dict[str, object] = {\n            "request_revision": req.get("request_revision"),\n            "conflict_entry": req.get("conflict_entry"),\n            "request_kind": request_kind,\n            "status": "invalid",\n        }\n        try:\n            start = float(req["window_start"])\n            end = float(req["window_end"])\n            if not (\n                math.isfinite(start) and math.isfinite(end)\n                and 0.0 <= start < end <= max_duration + 1e-3\n            ):\n                raise ValueError("window-invalid")\n            f1 = max(0, int(start * SAMPLE_RATE))\n            f2 = min(audio_t.shape[1], int(end * SAMPLE_RATE))\n            wav = audio_t[:, f1:f2]\n            if wav.shape[-1] < MIN_INPUT_SAMPLES:\n                wav = torch.nn.functional.pad(wav, (0, MIN_INPUT_SAMPLES - wav.shape[-1]))\n            inputs = processor(\n                wav.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"\n            )\n            input_values = inputs.input_values.to(resolved_device)\n            attention_mask = getattr(inputs, "attention_mask", None)\n            if attention_mask is not None:\n                attention_mask = attention_mask.to(resolved_device)\n            with torch.inference_mode():\n                output = model(input_values, attention_mask=attention_mask)\n                emission = torch.log_softmax(output.logits, dim=-1)[0].cpu().detach()\n\n            if request_kind == "pair-tail":\n                previous_reading = str(req["previous_reading"])\n                current_reading = str(req["current_reading"])\n                previous_clean = clean_reading(previous_reading, vocab)\n                current_clean = clean_reading(current_reading, vocab)\n                if len(previous_clean) < 2 or len(current_clean) < 2:\n                    raise ValueError("clean-reading-too-short")\n                combined = previous_clean + current_clean\n                chars, ratio = align_chars(\n                    emission, combined, vocab, blank, start, end\n                )\n                boundary = len(previous_clean)\n                previous_chars = chars[:boundary]\n                current_chars = chars[boundary:]\n                if (\n                    len(previous_chars) != len(previous_clean)\n                    or len(current_chars) != len(current_clean)\n                    or not previous_chars or not current_chars\n                ):\n                    raise ValueError("pair-boundary-mismatch")\n                if previous_chars[-1]["end"] > current_chars[0]["start"] + 0.08:\n                    raise ValueError("pair-order-overlap")\n                frame_slack = max(0.08, ratio * 4.0)\n                if current_chars[-1]["end"] >= end - frame_slack:\n                    raise ValueError("right-edge-pileup")\n                if previous_chars[0]["start"] <= start + min(0.04, frame_slack * 0.5):\n                    raise ValueError("left-edge-pileup")\n                result.update({\n                    "status": "aligned",\n                    "reason": "complete-known-lyric-hiragana-pair",\n                    "previous_clean_length": len(previous_clean),\n                    "current_clean_length": len(current_clean),\n                    "previous_chars": previous_chars,\n                    "current_chars": current_chars,\n                    "previous_first_start": previous_chars[0]["start"],\n                    "previous_last_end": previous_chars[-1]["end"],\n                    "current_first_start": current_chars[0]["start"],\n                    "current_last_end": current_chars[-1]["end"],\n                    "frame_seconds": round(ratio, 8),\n                })\n            elif request_kind == "current-neutral-onset-probe":\n                current_reading = str(req["current_reading"])\n                current_clean = clean_reading(current_reading, vocab)\n                if len(current_clean) < 2:\n                    raise ValueError("clean-reading-too-short")\n                current_chars, ratio = align_chars(\n                    emission, current_clean, vocab, blank, start, end\n                )\n                if len(current_chars) != len(current_clean) or not current_chars:\n                    raise ValueError("current-alignment-cardinality-mismatch")\n                frame_slack = max(0.08, ratio * 4.0)\n                if current_chars[-1]["end"] >= end - frame_slack:\n                    raise ValueError("right-edge-pileup")\n                if current_chars[0]["start"] <= start + min(0.04, frame_slack * 0.5):\n                    raise ValueError("left-edge-pileup")\n                result.update({\n                    "status": "aligned",\n                    "reason": "complete-known-lyric-hiragana-current-hypothesis-neutral",\n                    "current_clean_length": len(current_clean),\n                    "current_chars": current_chars,\n                    "current_first_start": current_chars[0]["start"],\n                    "current_last_end": current_chars[-1]["end"],\n                    "frame_seconds": round(ratio, 8),\n                })\n            else:\n                raise ValueError(f"unsupported-request-kind:{request_kind}")\n        except Exception as exc:\n            result["reason"] = f"{type(exc).__name__}:{exc}"\n        results.append(result)\n    output = {\n        "schema_version": 1,\n        "protocol_revision": PROTOCOL_REVISION,\n        "transformers_package_version": transformers_version,\n        "huggingface_hub_package_version": huggingface_hub_version,\n        "torch_package_version": torch_version,\n        "model_id": EXPECTED_MODEL_ID,\n        "model_revision": EXPECTED_MODEL_COMMIT,\n        "model_blob_sha256": EXPECTED_MODEL_BLOB_SHA256,\n        "vocab_revision": vocab_revision,\n        "requested_device": requested_device,\n        "device": resolved_device,\n        "language": "__LRC_KNOWN_LYRIC_HUBERT_LANGUAGE__",\n        "audio_loader": "__LRC_KNOWN_LYRIC_HUBERT_AUDIO_LOADER__",\n        "ctc_alignment": "__LRC_KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT__",\n        "results": results,\n    }\n    Path(args.output).write_text(\n        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"\n    )\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'


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


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
