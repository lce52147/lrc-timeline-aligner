"""Extracted from local_whisper_retry.py (split2): crossline retry helper invocation, output parsing, identity hypothesis recording and retry epoch execution."""

from __future__ import annotations

import concurrent.futures
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Literal
import model_cache
from ctc_refinement import ctc_right_edge_pileup_evidence
from timing_core import BaseRetryEpoch, BoundedRetryCapability, CandidateRegionEvidence, CrossLineRetryOutcome, CrossLineRetryRequest, IndependentTailEvidence, RuntimeTailEvidence, TailEvidenceContext, TailRegionEvidence, TimingCandidate, _finite_unit_confidence, _make_tail_provenance, _tail_helper_revision, analyze_candidate_regions, derive_reliable_tail, make_timing_candidate, record_alignment_hypothesis
from lyrics_io import LyricEntry, entry_sung_text
from audio_features import HOP_SECONDS
from runtime_base import LrcError, _canonical_json_digest, _ctc_model_blob_sha256, _model_cache_read, _model_cache_write, _run_model_helper_command, _timing_content_digest, default_ctc_python, progress_log
from runtime_base import TimingTokenSpan
from runtime_evidence_binding import _independent_tail_evidence_payload
from crossline_retry_planning import (
    plan_crossline_retry_requests,
    plan_joint_boundary_retry_requests,
)


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
