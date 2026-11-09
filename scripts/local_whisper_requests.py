"""Extracted from local_whisper_retry.py (split2): local whisper request planning, cache keys, runtime evidence binding, vocal-consensus and context-conditioned candidate augmentation."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Literal
from asr_backends import _occurrence_binding_time_bounds
from ctc_refinement import SEGMENT_REALIGN_SOURCE
from report_trust import _RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY, _suspicious_entry_requires_review, assignment_content_is_trusted, runtime_independent_content_identity_sources_for_entry, runtime_local_whisper_evidence_for_entry, runtime_raw_content_identity_for_entry
from timing_core import LocalWhisperCapability, LocalWhisperOutcome, LocalWhisperRequest, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, RuntimeTailEvidence, TailEvidenceContext, TimingCandidate, _candidate_support_families, _entry_identity_is_unique, _minimal_unique_sequence_context, _occurrence_binding_has_strong_context, build_final_timing_state, candidate_has_bound_direct_onset_evidence, candidate_has_boundary_grade_direct_onset_evidence, candidate_occurrence_evidence_from_binding, make_timing_candidate
from lyrics_io import LyricEntry, entry_sung_text, normalize_match_text
from audio_features import HOP_SECONDS
from runtime_base import _timing_content_digest


LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS = 1.50


LOCAL_WHISPER_MAX_WINDOW_SECONDS = 30.0


def plan_local_whisper_requests(
    report: dict[str, object],
    capability: LocalWhisperCapability | None,
    entries: list[LyricEntry],
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    duration: float,
    independent_tail_evidence: RuntimeTailEvidence | None = None,
    tail_evidence_context: TailEvidenceContext | None = None,
    timing_source_policy: Literal["default", "ctc-first"] = "default",
) -> tuple[LocalWhisperRequest, ...]:
    """Plan independent Whisper-family observations only where central evidence is incomplete."""
    if capability is None or len(entries) != len(candidate_sets):
        return ()
    if not math.isfinite(duration) or duration <= 0.0:
        return ()
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return ()
    state = build_final_timing_state(
        candidate_sets,
        independent_tail_evidence=independent_tail_evidence,
        tail_evidence_context=tail_evidence_context,
        timing_source_policy=timing_source_policy,
    )
    requests: list[LocalWhisperRequest] = []
    audio_revision = capability.audio_sha256

    for index, (decision, assignment) in enumerate(zip(state.decisions, assignments)):
        if not isinstance(assignment, dict):
            continue
        timing_need = decision.status == "provisional_unresolved"
        content_need = bool(
            not assignment_content_is_trusted(assignment)
            and not runtime_independent_content_identity_sources_for_entry(report, index)
        )
        review_need = _suspicious_entry_requires_review(report, index + 1)
        if not (timing_need or content_need or review_need):
            continue

        # Build a bounded identity context with one row of redundancy on each
        # available side.  The extra rows are not presumed trusted; they merely
        # give sparse occurrence binding a second anchor when an immediate
        # neighbor is weak.  If the expanded window would exceed the existing
        # 30 s local cap, fall back to the original minimal context.
        if _entry_identity_is_unique(entries, index):
            base_left = max(0, index - 1)
            base_right = min(len(entries) - 1, index + 1)
        else:
            fingerprint = _minimal_unique_sequence_context(entries, index, max_items=9)
            if fingerprint is None:
                continue
            base_left, base_right = fingerprint

        # Preserve the original minimal probe *and* optionally add one broader
        # sparse-context probe.  Broader context is evidence-additive only: it
        # must never replace a narrower request that may expose a cleaner direct
        # onset.  Runtime merging already ranks direct-onset evidence above
        # occurrence-only evidence for the same entry.
        context_options: list[tuple[int, int]] = [(base_left, base_right)]
        # Evaluate a small monotonic context ladder rather than betting the whole
        # observation on one segmentation-sensitive window.  Each broader probe
        # is optional evidence only; the minimal probe is always preserved.
        # Bound the ladder to at most three contexts and nine lyric rows so the
        # helper cost stays finite even on long songs.
        for expansion in (1, 2):
            expanded_left = max(0, base_left - expansion)
            expanded_right = min(len(entries) - 1, base_right + expansion)
            option = (expanded_left, expanded_right)
            if (
                expanded_left < index < expanded_right
                and expanded_right - expanded_left + 1 <= 9
                and option not in context_options
            ):
                context_options.append(option)

        seen_contexts: set[tuple[int, int, int, int]] = set()
        for option_left, option_right in context_options:
            option_left_start = state.decisions[option_left].written_time.seconds
            if option_left > 0:
                option_predecessor = state.decisions[option_left - 1].written_time.seconds
                option_window_start = max(
                    0.0,
                    option_predecessor + HOP_SECONDS,
                    option_left_start - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
                )
            else:
                option_window_start = 0.0

            option_right_start = state.decisions[option_right].written_time.seconds
            if option_right + 1 < len(entries):
                option_successor = state.decisions[option_right + 1].written_time.seconds
                option_window_end = min(
                    duration,
                    option_successor + LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
                )
            else:
                option_window_end = min(duration, option_right_start + 8.0)
            if option_window_end <= option_right_start + 0.5:
                option_window_end = min(duration, option_right_start + 4.0)
            option_length = option_window_end - option_window_start
            if (
                option_left == 0
                and option_length > LOCAL_WHISPER_MAX_WINDOW_SECONDS
            ):
                option_window_start = max(
                    0.0,
                    option_left_start - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
                )
                option_length = option_window_end - option_window_start
            if not 2.0 <= option_length <= LOCAL_WHISPER_MAX_WINDOW_SECONDS:
                continue

            context_indexes = tuple(range(option_left, option_right + 1))
            if not context_indexes or index not in context_indexes or len(context_indexes) > 9:
                continue
            semantic_key = (
                option_left,
                option_right,
                int(round(option_window_start * 1000)),
                int(round(option_window_end * 1000)),
            )
            if semantic_key in seen_contexts:
                continue
            seen_contexts.add(semantic_key)

            context_candidates = tuple(state.decisions[item].candidate for item in context_indexes)
            transcripts = tuple(entry_sung_text(entries[item]) for item in context_indexes)
            transcript_digests = tuple(_timing_content_digest(text) for text in transcripts)

            window_definition = {
                "mode": "unresolved-only-local-whisper",
                "start": round(option_window_start, 6),
                "end": round(option_window_end, 6),
                "context_indexes": list(context_indexes),
                "context_candidate_ids": [item.candidate_id for item in context_candidates],
                "timing_need": timing_need,
                "content_need": content_need,
                "review_need": review_need,
                "capability_revision": capability.capability_revision,
            }
            payload = {
                "entry_index": index,
                "capability_revision": capability.capability_revision,
                "audio_revision": audio_revision,
                "context_entry_indexes": context_indexes,
                "context_candidate_ids": tuple(item.candidate_id for item in context_candidates),
                "transcript_digests": transcript_digests,
                "transcript_digest": _timing_content_digest(transcripts),
                "window_start": round(option_window_start, 6),
                "window_end": round(option_window_end, 6),
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
    # A broken global CTC can collapse an entire verse 10-20 seconds earlier.
    # Local Whisper requests centered on Central's provisional baseline will
    # then listen to the wrong audio, even when S1 has a real bounded CTC path
    # at the correct vocal segment. Add only a capped independent ASR probe
    # around the *untrusted* S1 path. This is evidence acquisition, not timing
    # selection, and does not make S1 direct-onset/identity evidence.
    displaced_segment_rows: list[tuple[float, int, tuple[TimingCandidate, ...]]] = []
    segment_by_row: dict[int, TimingCandidate] = {}
    for index, items in enumerate(candidate_sets):
        proposals = [item for item in items
                     if item.source == "segment-constrained-ctc"
                     and item.spans and not item.window_truncated
                     and not item.right_edge_pileup]
        if not proposals:
            # V2 segment-realign proposals get the same capped independent probe.
            proposals = [item for item in items
                         if item.source == SEGMENT_REALIGN_SOURCE
                         and item.spans and not item.window_truncated
                         and not item.right_edge_pileup]
        if len(proposals) == 1:
            segment_by_row[index] = proposals[0]
            gap = abs(proposals[0].written_time.seconds - state.decisions[index].written_time.seconds)
            if gap >= 3.0 and (
                state.decisions[index].status == "provisional_unresolved"
                or not assignment_content_is_trusted(assignments[index])
                or _suspicious_entry_requires_review(report, index + 1)
            ):
                displaced_segment_rows.append((-gap, index, items))
    for _gap, index, _items in sorted(displaced_segment_rows)[:12]:
        if _entry_identity_is_unique(entries, index):
            first, last = max(0, index - 1), min(len(entries) - 1, index + 1)
        else:
            context = _minimal_unique_sequence_context(entries, index, max_items=9)
            if context is None:
                continue
            first, last = context
        context_indexes = tuple(range(first, last + 1))
        if (
            len(context_indexes) > 9
            or any(j not in segment_by_row for j in context_indexes)
        ):
            continue
        context_candidates = tuple(segment_by_row[j] for j in context_indexes)
        s1_times = tuple(item.written_time.seconds for item in context_candidates)
        if any(a >= b for a, b in zip(s1_times, s1_times[1:])):
            continue
        window_start = max(0.0, s1_times[0] - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS)
        window_end = min(duration, s1_times[-1] + max(4.0, LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS))
        if not 2.0 <= window_end - window_start <= LOCAL_WHISPER_MAX_WINDOW_SECONDS:
            continue
        transcript = tuple(entry_sung_text(entries[j]) for j in context_indexes)
        transcripts_digest = tuple(_timing_content_digest(text) for text in transcript)
        window_definition = {
            "mode": "s1-untrusted-segment-search-not-proof",
            "start": round(window_start, 6),
            "end": round(window_end, 6),
            "context_indexes": list(context_indexes),
            "context_candidate_ids": [x.candidate_id for x in context_candidates],
            "capability_revision": capability.capability_revision,
        }
        payload = {
            "entry_index": index,
            "capability_revision": capability.capability_revision,
            "audio_revision": audio_revision,
            "context_entry_indexes": context_indexes,
            "context_candidate_ids": tuple(x.candidate_id for x in context_candidates),
            "transcript_digests": transcripts_digest,
            "transcript_digest": _timing_content_digest(transcript),
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
                "transcript_digests": list(transcripts_digest),
            }),
            execution_revision=_local_whisper_execution_request_revision(payload),
        ))
    unique = {item.request_revision: item for item in requests}
    return tuple(sorted(
        unique.values(),
        key=lambda item: (item.entry_index, len(item.context_entry_indexes), item.request_revision),
    ))


def _local_whisper_execution_request_revision(
    payload: dict[str, object],
) -> str:
    """Hash only the bounded helper input; planner provenance stays diagnostic."""
    return _timing_content_digest({
        "schema": "local-whisper-execution-request-v2",
        "entry_index": payload["entry_index"],
        "capability_revision": payload["capability_revision"],
        "audio_revision": payload["audio_revision"],
        "context_entry_indexes": list(payload["context_entry_indexes"]),
        "transcript_digests": list(payload["transcript_digests"]),
        "transcript_digest": payload["transcript_digest"],
        "window_start": payload["window_start"],
        "window_end": payload["window_end"],
    })


def _local_whisper_request_execution_revision(request: LocalWhisperRequest) -> str:
    return request.execution_revision or _local_whisper_execution_request_revision(
        request.__dict__
    )


def _local_whisper_cache_key(
    request: LocalWhisperRequest,
    capability: LocalWhisperCapability,
) -> str:
    # Keep the serialized key shape stable so K2 cache entries remain reusable.
    return "local-whisper:" + _timing_content_digest({
        "request_revision": _local_whisper_request_execution_revision(request),
        "capability_revision": capability.capability_revision,
        "protocol_revision": capability.protocol_revision,
    })


def _local_whisper_outcome_for_request(
    outcome: LocalWhisperOutcome,
    request: LocalWhisperRequest,
) -> LocalWhisperOutcome:
    if outcome.request_revision == request.request_revision:
        return outcome
    base = {
        "entry_index": outcome.entry_index,
        "request_revision": request.request_revision,
        "status": outcome.status,
        "reason": outcome.reason,
        "onset_time": outcome.onset_time,
        "score": outcome.score,
        "evidence_revision": outcome.evidence_revision,
        "occurrence_binding": outcome.occurrence_binding,
        "raw_transcript_revision": outcome.raw_transcript_revision,
        "aligned_transcript_revision": outcome.aligned_transcript_revision,
    }
    return LocalWhisperOutcome(
        **base,
        outcome_revision=_timing_content_digest(base),
    )


def _shift_local_assignment_to_global(
    assignment: object,
    offset: float,
) -> dict[str, object] | None:
    if not isinstance(assignment, dict):
        return None
    shifted = copy.deepcopy(assignment)
    try:
        shifted["timestamp"] = round(float(assignment.get("timestamp")) + offset, 6)
    except (TypeError, ValueError):
        return None
    return shifted


def bind_runtime_local_whisper_evidence(
    report: dict[str, object],
    entries: list[LyricEntry],
    capability: LocalWhisperCapability | None,
    requests: tuple[LocalWhisperRequest, ...],
    outcomes: tuple[LocalWhisperOutcome, ...],
) -> None:
    if capability is None:
        return
    request_by_revision = {item.request_revision: item for item in requests}

    # Fixed-point recollection may legitimately stop requesting an entry after a
    # first-round local observation changes the candidate graph.  The observation
    # itself is immutable audio/lyric evidence and must not disappear merely
    # because the next planner round no longer needs to execute that request.
    # Seed from canonical-valid prior runtime evidence, then monotonically merge
    # any new observations.  Tampered/stale evidence fails the canonical accessor
    # and is therefore not retained.
    prior_runtime: dict[int, dict[str, object]] = {}
    previous_runtime = report.get(_RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY)
    if isinstance(previous_runtime, dict):
        for entry_index in sorted(previous_runtime):
            if not isinstance(entry_index, int) or not 0 <= entry_index < len(entries):
                continue
            retained = runtime_local_whisper_evidence_for_entry(report, entries, entry_index)
            if retained is not None:
                prior_runtime[entry_index] = retained
    retained_before_merge = len(prior_runtime)

    observations: dict[int, list[dict[str, object]]] = {
        index: [dict(value)] for index, value in prior_runtime.items()
    }
    for outcome in outcomes:
        if outcome.status != "evidence" or outcome.evidence_revision is None:
            continue
        request = request_by_revision.get(outcome.request_revision)
        if request is None or outcome.score is None:
            continue
        lyric_sha = hashlib.sha256(
            normalize_match_text(entry_sung_text(entries[outcome.entry_index])).encode("utf-8")
        ).hexdigest()
        payload = {
            "entry_index": outcome.entry_index,
            "lyric_sha256": lyric_sha,
            "onset_time": (
                round(float(outcome.onset_time), 6)
                if outcome.onset_time is not None else None
            ),
            "score": round(float(outcome.score), 6),
            "occurrence_binding": outcome.occurrence_binding,
            "request_revision": request.request_revision,
            "capability_revision": capability.capability_revision,
            "audio_revision": capability.audio_sha256,
            "window_start": round(request.window_start, 6),
            "window_end": round(request.window_end, 6),
            "raw_transcript_revision": outcome.raw_transcript_revision,
            "aligned_transcript_revision": outcome.aligned_transcript_revision,
            "producer": "local-whisper-aligner",
        }
        evidence = {
            **payload,
            "evidence_revision": _timing_content_digest(payload),
        }
        observations.setdefault(outcome.entry_index, []).append(evidence)

    composite_count = 0
    # A narrow probe may contain the clean forced-character onset while a wider
    # probe contains the sequence context needed to bind a repeated lyric.  They
    # are complementary observations from the same immutable audio/lyric source,
    # not independent votes.  Merge only when both observations are revision-
    # bound, their target times agree inside the existing association tolerance,
    # and the onset lies inside the context bracket.  The timestamp is never
    # averaged or moved: it remains the original direct onset observation.
    for entry_index, items in list(observations.items()):
        onset_items = [item for item in items if item.get("onset_time") is not None]
        context_items = [
            item for item in items
            if _occurrence_binding_has_strong_context(item.get("occurrence_binding"))
        ]
        composites: list[dict[str, object]] = []
        for onset_item in onset_items:
            try:
                onset_time = float(onset_item.get("onset_time"))
                onset_score = float(onset_item.get("score"))
            except (TypeError, ValueError):
                continue
            for context_item in context_items:
                context_binding = context_item.get("occurrence_binding")
                if context_binding == onset_item.get("occurrence_binding"):
                    continue
                if (
                    onset_item.get("lyric_sha256") != context_item.get("lyric_sha256")
                    or onset_item.get("capability_revision") != context_item.get("capability_revision")
                    or onset_item.get("audio_revision") != context_item.get("audio_revision")
                ):
                    continue
                try:
                    context_time = float(context_binding.get("timestamp"))  # type: ignore[union-attr]
                    context_score = float(context_item.get("score"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if (
                    not math.isfinite(onset_time)
                    or not math.isfinite(context_time)
                    or abs(onset_time - context_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
                ):
                    continue
                left_bound, right_bound = _occurrence_binding_time_bounds(
                    context_binding, entry_index
                )
                if left_bound is not None and onset_time <= left_bound - HOP_SECONDS:
                    continue
                if right_bound is not None and onset_time >= right_bound + HOP_SECONDS:
                    continue
                composite_payload = {
                    "entry_index": entry_index,
                    "lyric_sha256": onset_item.get("lyric_sha256"),
                    "onset_time": round(onset_time, 6),
                    "score": round(min(onset_score, context_score), 6),
                    "occurrence_binding": context_binding,
                    "request_revision": onset_item.get("request_revision"),
                    "capability_revision": onset_item.get("capability_revision"),
                    "audio_revision": onset_item.get("audio_revision"),
                    "window_start": round(float(onset_item.get("window_start")), 6),
                    "window_end": round(float(onset_item.get("window_end")), 6),
                    "raw_transcript_revision": onset_item.get("raw_transcript_revision"),
                    "aligned_transcript_revision": onset_item.get("aligned_transcript_revision"),
                    "producer": "local-whisper-aligner",
                    "composite_onset_evidence_revision": onset_item.get("evidence_revision"),
                    "composite_context_evidence_revision": context_item.get("evidence_revision"),
                }
                composite = {
                    **composite_payload,
                    "evidence_revision": _timing_content_digest(composite_payload),
                }
                composites.append(composite)
        if composites:
            unique = {item["evidence_revision"]: item for item in composites}
            observations[entry_index].extend(unique.values())
            composite_count += len(unique)

    def evidence_rank(item: dict[str, object]) -> tuple[int, int, float, str]:
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return (
            int(item.get("onset_time") is not None),
            int(_occurrence_binding_has_strong_context(item.get("occurrence_binding"))),
            score,
            str(item.get("evidence_revision") or ""),
        )

    runtime: dict[int, dict[str, object]] = {}
    for entry_index, items in observations.items():
        if not items:
            continue
        runtime[entry_index] = max(items, key=evidence_rank)

    report[_RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY] = runtime
    report["local_whisper_evidence"] = {
        "status": "runtime-evidence-generated",
        "authoritative_serialized": False,
        "protocol_revision": capability.protocol_revision,
        "candidate_count": len(runtime),
        "retained_from_prior_collect_count": retained_before_merge,
        "composite_evidence_count": composite_count,
        "entries": [index + 1 for index in sorted(runtime)],
    }


def _augment_candidates_with_local_whisper_consensus(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
) -> tuple[tuple[TimingCandidate, ...], ...]:
    """Fuse local Whisper onset/identity into existing candidates without moving time."""
    augmented = [list(items) for items in candidate_sets]
    for index, parents in enumerate(candidate_sets):
        evidence = runtime_local_whisper_evidence_for_entry(report, entries, index)
        if evidence is None:
            continue
        raw_onset_time = evidence.get("onset_time")
        if raw_onset_time is None:
            continue
        onset_time = float(raw_onset_time)
        evidence_score = float(evidence["score"])
        for parent in parents:
            families = _candidate_support_families(parent)
            # Local Whisper is one Whisper-family observation. Never use a
            # Whisper-derived parent to manufacture a second independent vote.
            if "whisper-family" in families:
                continue
            if (
                parent.identity_support != "supported"
                or parent.sequence_support != "supported"
                or parent.right_edge_pileup
                or parent.window_truncated
                or abs(parent.written_time.seconds - onset_time) > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            ):
                continue
            if "ctc" in families:
                source = "ctc-local-whisper-independent-consensus"
            elif "recurrence" in families:
                source = "recurrence-local-whisper-independent-consensus"
            else:
                source = "local-whisper-independent-consensus"
            artifact = {
                "producer": "local-whisper-aligner",
                "local_whisper_evidence_revision": evidence["evidence_revision"],
                "parent_candidate_id": parent.candidate_id,
                "parent_generation_revision": parent.generation_revision,
                "parent_source": parent.source,
                "parent_time": round(parent.raw_time, 6),
                "onset_time": round(onset_time, 6),
                "runtime_only": True,
            }
            augmented[index].append(make_timing_candidate(
                entry_index=index,
                entry_text=entry_sung_text(entries[index]),
                source=source,
                raw_time=parent.raw_time,
                spans=parent.spans,
                confidence=min(parent.confidence, evidence_score),
                identity_support="supported",
                sequence_support="supported",
                acoustic_support="unavailable",
                direct_onset_support="supported",
                direct_onset_time=onset_time,
                direct_onset_source_artifact=artifact,
                direct_onset_evidence_producer="local-whisper-aligner",
                direct_onset_evidence_kind="cross-backend-onset",
                direct_onset_evidence_independent=True,
                occurrence_evidence=candidate_occurrence_evidence_from_binding(
                    entries,
                    index,
                    producer="local-whisper-aligner",
                    occurrence_binding=evidence.get("occurrence_binding"),
                    upstream_evidence_revision=evidence.get("evidence_revision"),
                ),
                independent_content_identity=True,
                current=False,
                prefix_scope=parent.prefix_scope,
                right_edge_pileup=False,
                window_truncated=False,
                source_artifact=artifact,
            ))
    return tuple(
        tuple({item.candidate_id: item for item in items}[key] for key in sorted(
            {item.candidate_id: item for item in items}
        ))
        for items in augmented
    )


def _apply_context_conditioned_whisper_hypothesis_arbitration(
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
) -> tuple[tuple[tuple[TimingCandidate, ...], ...], tuple[dict[str, object], ...]]:
    """Invalidate weaker whole-song Whisper hypotheses with stronger bound context.

    This is *not* majority voting.  A global raw-ASR occurrence may be removed only
    when all of the following are provenance-bound facts:

    1. bounded local Whisper re-identifies the lyric inside a trusted full or
       sparse left/target/right sequence bracket and exposes either a direct aligned
       onset or an exact revision-bound independent vocal onset;
    2. an independent CTC+local-Whisper composite converges at that onset inside the
       existing 0.18 s tolerance;
    3. the global raw occurrence is only ``global-unique`` (no sequence bracket), is
       distant from the bounded onset, and is not independently corroborated by CTC
       or recurrence near its own time;
    4. the bounded observation is at least as confident as the raw identity row.

    The losing timestamp is not overwritten.  Its candidate is excluded from the
    central valid graph with an explicit diagnostic that binds both evidence
    revisions.  If any premise is absent, nothing is invalidated and fail-closed
    review behavior is preserved.
    """
    revised: list[tuple[TimingCandidate, ...]] = []
    diagnostics: list[dict[str, object]] = []
    tolerance = RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS

    for index, candidates in enumerate(candidate_sets):
        local = runtime_local_whisper_evidence_for_entry(report, entries, index)
        raw_identity = runtime_raw_content_identity_for_entry(report, entries, index)
        if not isinstance(local, dict) or not isinstance(raw_identity, dict):
            revised.append(candidates)
            continue
        local_binding = local.get("occurrence_binding")
        raw_binding = raw_identity.get("occurrence_binding")
        if not (
            _occurrence_binding_has_strong_context(local_binding)
            and isinstance(raw_binding, dict)
            and raw_binding.get("mode") == "global-unique"
        ):
            revised.append(candidates)
            continue
        try:
            local_score = float(local.get("score"))
            raw_time = float(raw_identity.get("raw_time"))
            raw_score = float(raw_identity.get("raw_score"))
        except (TypeError, ValueError):
            revised.append(candidates)
            continue

        # Independent corroboration must already exist as an actual composite;
        # raw CTC proximity or generic acoustic activity is not inferred here.
        raw_local_onset = local.get("onset_time")
        if raw_local_onset is not None:
            try:
                local_onset = float(raw_local_onset)
            except (TypeError, ValueError):
                revised.append(candidates)
                continue
            local_cluster = [
                candidate for candidate in candidates
                if candidate.source == "ctc-local-whisper-independent-consensus"
                and abs(candidate.written_time.seconds - local_onset) <= tolerance
                and candidate.identity_support == "supported"
                and candidate.sequence_support == "supported"
                and candidate_has_bound_direct_onset_evidence(candidate)
            ]
        else:
            # Occurrence-only Local Whisper can already have a boundary-grade
            # onset in the candidate graph via the independent vocal-fusion
            # producer.  Consume only an exact revision/binding match; never
            # synthesize an onset from the occurrence timestamp itself.
            local_revision = local.get("evidence_revision")
            binding_revision = (
                local_binding.get("binding_revision")
                if isinstance(local_binding, dict) else None
            )
            bound_vocal: list[tuple[TimingCandidate, float]] = []
            for candidate in candidates:
                if (
                    candidate.source != "ctc-local-whisper-vocal-independent-consensus"
                    or candidate.identity_support != "supported"
                    or candidate.sequence_support != "supported"
                    or not candidate_has_boundary_grade_direct_onset_evidence(candidate, entries)
                ):
                    continue
                occurrence = candidate.occurrence_evidence
                onset_evidence = candidate.direct_onset_evidence
                if (
                    occurrence is None
                    or onset_evidence is None
                    or occurrence.producer != "local-whisper-vocal-fusion"
                    or occurrence.upstream_evidence_revision != local_revision
                    or occurrence.binding_revision != binding_revision
                    or onset_evidence.producer != "local-whisper-vocal-fusion"
                ):
                    continue
                onset = float(onset_evidence.onset_time)
                if (
                    math.isfinite(onset)
                    and abs(candidate.written_time.seconds - onset) <= tolerance
                ):
                    bound_vocal.append((candidate, onset))
            canonical_onsets = {round(onset, 6) for _, onset in bound_vocal}
            if len(canonical_onsets) != 1:
                revised.append(candidates)
                continue
            local_onset = next(iter(canonical_onsets))
            local_cluster = [candidate for candidate, _ in bound_vocal]

        if (
            not all(math.isfinite(value) for value in (local_onset, local_score, raw_time, raw_score))
            or abs(local_onset - raw_time) <= tolerance
            or local_score < raw_score
            or not local_cluster
        ):
            revised.append(candidates)
            continue

        # If CTC or recurrence independently supports the raw cluster, it is a
        # genuine competing hypothesis and must remain review-required.
        raw_independent_support = any(
            abs(candidate.written_time.seconds - raw_time) <= tolerance
            and (
                "ctc" in _candidate_support_families(candidate)
                or "recurrence" in _candidate_support_families(candidate)
            )
            and "whisper-family" not in _candidate_support_families(candidate)
            for candidate in candidates
        )
        if raw_independent_support:
            revised.append(candidates)
            continue

        falsified = [
            candidate for candidate in candidates
            if "whisper-family" in _candidate_support_families(candidate)
            and "ctc" not in _candidate_support_families(candidate)
            and "recurrence" not in _candidate_support_families(candidate)
            and abs(candidate.written_time.seconds - raw_time) <= tolerance
            and abs(candidate.written_time.seconds - local_onset) > tolerance
        ]
        if not falsified:
            revised.append(candidates)
            continue
        falsified_ids = {candidate.candidate_id for candidate in falsified}
        survivors = tuple(candidate for candidate in candidates if candidate.candidate_id not in falsified_ids)
        if not survivors:
            revised.append(candidates)
            continue
        revised.append(survivors)
        diagnostics.append({
            "entry": index + 1,
            "reason": "bounded-sequence-context-falsifies-weaker-global-whisper-occurrence",
            "local_whisper_evidence_revision": local.get("evidence_revision"),
            "raw_content_identity_revision": raw_identity.get("evidence_revision"),
            "local_onset": round(local_onset, 6),
            "raw_identity_time": round(raw_time, 6),
            "local_score": round(local_score, 6),
            "raw_score": round(raw_score, 6),
            "context_binding_revision": local_binding.get("binding_revision"),
            "raw_binding_revision": raw_binding.get("binding_revision"),
            "corroborating_candidate_ids": sorted(candidate.candidate_id for candidate in local_cluster),
            "falsified_candidate_ids": sorted(falsified_ids),
            "max_delta_seconds": tolerance,
            "authority": "candidate-exclusion-only-central-still-commits",
        })

    return tuple(revised), tuple(diagnostics)
