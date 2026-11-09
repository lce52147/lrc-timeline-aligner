"""Extracted from timing_core.py (split2): timing candidate construction, region analysis, direct-onset predicates and reliable-tail derivation."""

from __future__ import annotations

import copy
import math
import statistics
import unicodedata
from typing import Literal
from lyrics_io import LyricEntry
from audio_features import HOP_SECONDS
from runtime_base import _timing_content_digest, canonicalize_lrc_time
from runtime_base import TimingTokenSpan
from timing_evidence_types import (
    CTC_SEVERE_INTERNAL_GAP_SECONDS,
    PREFIX_ACOUSTIC_ASSOCIATION_SECONDS,
    EvidenceState,
    TailValidity,
    CandidateAcousticEvidence,
    CandidateOnsetEvidence,
    CandidateOccurrenceEvidence,
    TimingCandidate,
    CandidateRegionEvidence,
    TailRegionEvidence,
    TailEvidenceContext,
    IndependentTailEvidence,
    TailObservationSet,
)
from tail_occurrence_evidence import (
    _candidate_occurrence_evidence_payload,
    candidate_occurrence_evidence_is_valid,
    _candidate_requires_occurrence_evidence,
    _tail_backend_family,
    _tail_provenance_payload,
    _tail_provenance_is_valid,
    _tail_provenance_matches_context,
    _tail_contexts_are_valid,
    _finite_unit_confidence,
)


def make_timing_candidate(
    *,
    entry_index: int,
    entry_text: str,
    source: str,
    raw_time: float,
    spans: tuple[TimingTokenSpan, ...] = (),
    confidence: float = 0.0,
    identity_support: EvidenceState = "unavailable",
    sequence_support: EvidenceState = "unavailable",
    acoustic_support: EvidenceState = "unavailable",
    direct_onset_support: EvidenceState = "unavailable",
    acoustic_strength: float | None = None,
    acoustic_onset_time: float | None = None,
    acoustic_source_artifact: object = None,
    acoustic_evidence_producer: str | None = None,
    acoustic_evidence_kind: str = "generic-acoustic",
    acoustic_evidence_independent: bool = False,
    direct_onset_time: float | None = None,
    direct_onset_source_artifact: object = None,
    direct_onset_evidence_producer: str | None = None,
    direct_onset_evidence_kind: str = "generic-onset",
    direct_onset_evidence_independent: bool = False,
    occurrence_evidence: CandidateOccurrenceEvidence | None = None,
    independent_content_identity: bool = False,
    current: bool = False,
    prefix_scope: bool = False,
    right_edge_pileup: bool = False,
    window_truncated: bool = False,
    source_artifact: object = None,
) -> TimingCandidate:
    """Build an order-independent candidate identity from source content."""
    written_time = canonicalize_lrc_time(raw_time)
    span_payload = [
        [round(span.start, 6), round(span.end, 6), round(span.score, 8), span.token]
        for span in spans
    ]
    acoustic_evidence: CandidateAcousticEvidence | None = None
    if acoustic_onset_time is not None and acoustic_strength is not None:
        acoustic_revision = _timing_content_digest(acoustic_source_artifact)
        acoustic_evidence = CandidateAcousticEvidence(
            onset_time=float(acoustic_onset_time),
            strength=float(acoustic_strength),
            distance_seconds=float(acoustic_onset_time) - written_time.seconds,
            source_revision=acoustic_revision,
            producer=str(acoustic_evidence_producer or ""),
            kind=str(acoustic_evidence_kind or "generic-acoustic"),
            independent=bool(acoustic_evidence_independent and acoustic_evidence_producer),
        )
    direct_onset_evidence: CandidateOnsetEvidence | None = None
    if direct_onset_time is not None and direct_onset_source_artifact is not None:
        direct_onset_evidence = CandidateOnsetEvidence(
            onset_time=float(direct_onset_time),
            distance_seconds=float(direct_onset_time) - written_time.seconds,
            source_revision=_timing_content_digest(direct_onset_source_artifact),
            producer=str(direct_onset_evidence_producer or ""),
            kind=str(direct_onset_evidence_kind or "generic-onset"),
            independent=bool(direct_onset_evidence_independent and direct_onset_evidence_producer),
        )
    # Cross-backend direct evidence must carry its immutable occurrence binding.
    # Other direct producers (notably the existing recurrence producers) already
    # have their own provenance contract and must retain that contract here.  The
    # central validators still re-check occurrence evidence for every
    # cross-backend candidate, so a caller cannot restore the old opaque
    # ``source_artifact`` trust path by omitting the binding.
    cross_backend_identity_claim = bool(
        independent_content_identity
        and direct_onset_evidence is not None
        and direct_onset_evidence.kind == "cross-backend-onset"
    )
    independent_content_identity = bool(
        independent_content_identity
        and (occurrence_evidence is not None or not cross_backend_identity_claim)
    )
    normalized_confidence = _finite_unit_confidence(confidence)
    evidence_payload = {
        "confidence": round(normalized_confidence, 8),
        "identity_support": identity_support,
        "sequence_support": sequence_support,
        "acoustic_support": acoustic_support,
        "direct_onset_support": direct_onset_support,
        "direct_onset_evidence": (
            {
                "onset_time": round(direct_onset_evidence.onset_time, 6),
                "distance_seconds": round(direct_onset_evidence.distance_seconds, 6),
                "source_revision": direct_onset_evidence.source_revision,
                "producer": direct_onset_evidence.producer,
                "kind": direct_onset_evidence.kind,
                "independent": direct_onset_evidence.independent,
            }
            if direct_onset_evidence is not None
            else None
        ),
        "independent_content_identity": bool(independent_content_identity),
        "occurrence_evidence": _candidate_occurrence_evidence_payload(
            occurrence_evidence
        ),
        "acoustic_evidence": (
            {
                "onset_time": round(acoustic_evidence.onset_time, 6),
                "strength": round(acoustic_evidence.strength, 8),
                "distance_seconds": round(acoustic_evidence.distance_seconds, 6),
                "source_revision": acoustic_evidence.source_revision,
                "producer": acoustic_evidence.producer,
                "kind": acoustic_evidence.kind,
                "independent": acoustic_evidence.independent,
            }
            if acoustic_evidence is not None
            else None
        ),
        "current": bool(current),
        "prefix_scope": bool(prefix_scope),
        "right_edge_pileup": bool(right_edge_pileup),
        "window_truncated": bool(window_truncated),
    }
    generation_revision = _timing_content_digest(
        {
            "source": source,
            "artifact": source_artifact,
            "spans": span_payload,
            "evidence": evidence_payload,
        }
    )
    candidate_id = _timing_content_digest(
        {
            "schema": 1,
            "entry": entry_index,
            "entry_digest": _timing_content_digest(unicodedata.normalize("NFKC", entry_text)),
            "source": source,
            "generation_revision": generation_revision,
            "raw_time": round(float(raw_time), 6),
            "canonical_centiseconds": written_time.centiseconds,
            "spans": span_payload,
            "evidence": evidence_payload,
        }
    )
    return TimingCandidate(
        candidate_id=candidate_id,
        entry_index=entry_index,
        source=source,
        raw_time=float(raw_time),
        written_time=written_time,
        generation_revision=generation_revision,
        spans=spans,
        confidence=normalized_confidence,
        identity_support=identity_support,
        sequence_support=sequence_support,
        acoustic_support=acoustic_support,
        direct_onset_support=direct_onset_support,
        acoustic_strength=acoustic_strength,
        acoustic_evidence=acoustic_evidence,
        direct_onset_evidence=direct_onset_evidence,
        occurrence_evidence=occurrence_evidence,
        independent_content_identity=bool(independent_content_identity),
        current=current,
        prefix_scope=prefix_scope,
        right_edge_pileup=right_edge_pileup,
        window_truncated=window_truncated,
    )


def record_alignment_hypothesis(
    assignment: dict[str, object],
    *,
    source: str,
    stage: str,
    raw_time: float,
    confidence: float = 0.0,
    spans: object = None,
    identity_support: EvidenceState = "unavailable",
    sequence_support: EvidenceState = "unavailable",
    acoustic_support: EvidenceState = "unavailable",
    direct_onset_support: EvidenceState = "unavailable",
    acoustic_strength: float | None = None,
    acoustic_onset_time: float | None = None,
    acoustic_evidence_producer: str | None = None,
    acoustic_evidence_kind: str = "generic-acoustic",
    direct_onset_time: float | None = None,
    direct_onset_evidence_producer: str | None = None,
    direct_onset_evidence_kind: str = "generic-onset",
    parent_source: str | None = None,
    parent_time: float | None = None,
    window: dict[str, object] | None = None,
    mode: str | None = None,
    right_edge_pileup: bool = False,
    window_truncated: bool = False,
) -> dict[str, object]:
    """Append a JSON-safe immutable pre-central timing hypothesis.

    Candidate generators may use mutable scratch state internally, but they must
    publish alternatives through this ledger instead of replacing the backend's
    baseline assignment timestamp.  The central evaluator is the only authority
    allowed to select/commit a final timestamp.
    """
    hypotheses = assignment.get("alignment_hypotheses")
    if not isinstance(hypotheses, list):
        hypotheses = []
        assignment["alignment_hypotheses"] = hypotheses
    spans_payload = copy.deepcopy(spans) if isinstance(spans, list) else []
    payload: dict[str, object] = {
        "source": source,
        "stage": stage,
        "raw_candidate_time": round(float(raw_time), 6),
        "confidence": round(_finite_unit_confidence(confidence), 8),
        "spans": spans_payload,
        "identity_support": identity_support,
        "sequence_support": sequence_support,
        "acoustic_support": acoustic_support,
        "direct_onset_support": direct_onset_support,
        "acoustic_strength": (
            round(float(acoustic_strength), 8) if acoustic_strength is not None else None
        ),
        "acoustic_onset_time": (
            round(float(acoustic_onset_time), 6) if acoustic_onset_time is not None else None
        ),
        "direct_onset_time": (
            round(float(direct_onset_time), 6) if direct_onset_time is not None else None
        ),
        # These labels are descriptive only.  Central collection independently
        # validates whether a producer is allowed to establish timing trust.
        "acoustic_evidence_producer": acoustic_evidence_producer,
        "acoustic_evidence_kind": acoustic_evidence_kind,
        "direct_onset_evidence_producer": direct_onset_evidence_producer,
        "direct_onset_evidence_kind": direct_onset_evidence_kind,
        "parent": (
            {
                "source": parent_source,
                "raw_candidate_time": round(float(parent_time), 6),
            }
            if parent_source is not None and parent_time is not None
            else None
        ),
        "window": copy.deepcopy(window) if isinstance(window, dict) else None,
        "mode": mode,
        "right_edge_pileup": bool(right_edge_pileup),
        "window_truncated": bool(window_truncated),
        "authoritative": False,
    }
    payload["hypothesis_id"] = _timing_content_digest(payload)
    if not any(
        isinstance(item, dict) and item.get("hypothesis_id") == payload["hypothesis_id"]
        for item in hypotheses
    ):
        hypotheses.append(payload)
    return payload


def _percentile(values: tuple[float, ...], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _supported_run_metrics(
    spans: tuple[TimingTokenSpan, ...],
    gaps: tuple[float, ...],
) -> tuple[int, int, int]:
    """Return all/opening/later runs without joining disconnected islands."""
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, span in enumerate(spans):
        disconnected = (
            index > 0 and gaps[index - 1] >= CTC_SEVERE_INTERNAL_GAP_SECONDS
        )
        if span.score < 0.05 or disconnected:
            if run_start is not None:
                runs.append((run_start, index))
                run_start = None
            if span.score >= 0.05:
                run_start = index
        elif run_start is None:
            run_start = index
    if run_start is not None:
        runs.append((run_start, len(spans)))
    longest = max((end - start for start, end in runs), default=0)
    opening = next((end - start for start, end in runs if start == 0), 0)
    later = max((end - start for start, end in runs if start > 0), default=0)
    return longest, opening, later


STRICT_DIRECT_ONSET_KINDS = frozenset({
    "phonetic-onset",
    "vocal-onset",
    "vocal-activity-reentry",
    "recurrence-onset",
    "external-reference",
    "cross-backend-onset",
    "joint-lyric-boundary",
    "known-lyric-onset",
})


STRICT_DIRECT_ONSET_PRODUCERS: dict[str, frozenset[str]] = {
    "phonetic-onset": frozenset({"phonetic-onset-fusion"}),
    "vocal-onset": frozenset({"consonant-onset-detector"}),
    "vocal-activity-reentry": frozenset({"vocal-activity-detector"}),
    "recurrence-onset": frozenset({"global-recurrence-search", "mutual-recurrence-search"}),
    "external-reference": frozenset({"external-reference"}),
    "cross-backend-onset": frozenset({
        "raw-asr-consensus",
        "raw-asr-vocal-fusion",
        "whisperx-consensus",
        "whisperx-vocal-fusion",
        "local-whisper-aligner",
        "local-whisper-vocal-fusion",
    }),
    "joint-lyric-boundary": frozenset({"joint-ctc-boundary-consensus"}),
    "known-lyric-onset": frozenset({"hubert-ja-known-lyric-current-onset"}),
}


BOUNDARY_GRADE_DIRECT_ONSET_KINDS = frozenset({
    "phonetic-onset",
    "vocal-activity-reentry",
    "recurrence-onset",
    "external-reference",
    "cross-backend-onset",
    "joint-lyric-boundary",
    "known-lyric-onset",
})


def candidate_has_bound_acoustic_evidence(candidate: TimingCandidate) -> bool:
    evidence = candidate.acoustic_evidence
    return bool(
        candidate.acoustic_support == "supported"
        and evidence is not None
        and evidence.source_revision
        and evidence.independent
        and bool(evidence.producer)
        and abs(evidence.distance_seconds) <= PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
    )


def candidate_has_bound_direct_onset_evidence(
    candidate: TimingCandidate,
    entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Return true for revision-bound onset evidence independent of the candidate.

    This establishes that an onset observation is genuine.  It does *not* imply
    that every onset kind is lyric-specific enough to defeat an UNKNOWN or
    VIOLATION previous-tail relation; use
    :func:`candidate_has_boundary_grade_direct_onset_evidence` for that.
    """
    evidence = candidate.direct_onset_evidence
    if _candidate_requires_occurrence_evidence(candidate) and not candidate_occurrence_evidence_is_valid(
        candidate, entries
    ):
        return False
    return bool(
        candidate.direct_onset_support == "supported"
        and evidence is not None
        and evidence.source_revision
        and evidence.independent
        and bool(evidence.producer)
        and evidence.kind in STRICT_DIRECT_ONSET_KINDS
        and evidence.producer in STRICT_DIRECT_ONSET_PRODUCERS.get(evidence.kind, frozenset())
        and (
            evidence.kind != "external-reference"
            or candidate.source in {"manual-anchor", "checked-lrc", "lyrics-source"}
        )
        and abs(evidence.distance_seconds) <= PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
    )


def candidate_has_boundary_grade_direct_onset_evidence(
    candidate: TimingCandidate,
    entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Whether independent onset evidence may cross a previous-tail conflict.

    A generic isolated-vocal energy peak is independent acoustic evidence, but
    it is not independently phonetic/lyric-specific.  It can support an opening
    when ownership is already CLEAR, never by itself override UNKNOWN/VIOLATION.
    """
    evidence = candidate.direct_onset_evidence
    return bool(
        candidate_has_bound_direct_onset_evidence(candidate, entries)
        and evidence is not None
        and evidence.kind in BOUNDARY_GRADE_DIRECT_ONSET_KINDS
    )


def candidate_has_independent_backend_identity(
    candidate: TimingCandidate,
    entries: list[LyricEntry] | tuple[LyricEntry, ...] | None = None,
) -> bool:
    """Whether an independent ASR/alignment backend proves this lyric row's identity."""
    evidence = candidate.direct_onset_evidence
    return bool(
        candidate.independent_content_identity
        and candidate.identity_support == "supported"
        and candidate.sequence_support == "supported"
        and evidence is not None
        and evidence.kind == "cross-backend-onset"
        and evidence.producer in {
            "raw-asr-consensus",
            "raw-asr-vocal-fusion",
            "whisperx-consensus",
            "whisperx-vocal-fusion",
            "local-whisper-aligner",
            "local-whisper-vocal-fusion",
        }
        and candidate_has_bound_direct_onset_evidence(candidate, entries)
    )


_ROMAJI_VOWELS = frozenset("aeiou")


def _opening_mora_proof_end(
    spans: tuple[TimingTokenSpan, ...],
    parsed_gaps: tuple[float, ...],
) -> float | None:
    """Return the end of a locally-supported first mora.

    Sentence timing is defined by the start of the current lyric, not by the
    release of the previous lyric.  CTC often assigns weak posteriors to short
    Japanese onset consonants (/k h s t .../) while strongly supporting the
    following vowel.  Treat a compact C(+C)+V onset as phonetic opening proof
    when the mora as a whole is supported.

    This intentionally does *not* accept an isolated high posterior with no
    token identity or a delayed vowel: it is a local phonetic-sequence test,
    not a generic score threshold.
    """
    if not spans:
        return None

    # Vowel-initial mora.  Require immediate continuation so an isolated
    # token before a multi-second fracture cannot certify the whole opening.
    first_token = (spans[0].token or "").lower()
    if (
        first_token in _ROMAJI_VOWELS
        and spans[0].score >= 0.12
        and len(spans) >= 2
        and parsed_gaps
        and parsed_gaps[0] < 0.18
        and spans[1].score >= 0.01
    ):
        return spans[0].end

    # Japanese Hepburn onsets are normally C+V or CC+V (e.g. shi/chi/tsu).
    # Search only the first three romanized characters and require a compact
    # onset; this prevents later strong vowels from back-certifying a missing
    # sentence start.
    limit = min(3, len(spans))
    vowel_index = next(
        (
            index
            for index in range(limit)
            if (spans[index].token or "").lower() in _ROMAJI_VOWELS
        ),
        None,
    )
    if vowel_index is None or vowel_index == 0:
        return None
    vowel = spans[vowel_index]
    consonants = spans[:vowel_index]
    if vowel.score < 0.12:
        return None
    if any((span.token or "").lower() in _ROMAJI_VOWELS for span in consonants):
        return None
    if max((span.score for span in consonants), default=0.0) < 0.01:
        return None
    if vowel.start - spans[0].start > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS:
        return None
    for gap_index in range(vowel_index):
        if parsed_gaps[gap_index] >= 0.18:
            return None
    # The first mora must connect to at least one following token of the same
    # lyric sequence.  This is what distinguishes a sentence opening from a
    # lone accidental posterior island.
    if vowel_index + 1 >= len(spans):
        return None
    if parsed_gaps[vowel_index] >= 0.18 or spans[vowel_index + 1].score < 0.01:
        return None
    return vowel.end


# Candidates are frozen and their region analysis depends on nothing else, yet
# Central asks for the same candidate's regions tens of thousands of times per
# song; each object's result is remembered by identity (the entry keeps the
# object alive, so its id cannot be reused while cached).
_CANDIDATE_REGIONS: dict[int, tuple[TimingCandidate, CandidateRegionEvidence]] = {}


def analyze_candidate_regions(candidate: TimingCandidate) -> CandidateRegionEvidence:
    cached = _CANDIDATE_REGIONS.get(id(candidate))
    if cached is not None and cached[0] is candidate:
        return cached[1]
    result = _analyze_candidate_regions_uncached(candidate)
    if len(_CANDIDATE_REGIONS) >= 65536:
        _CANDIDATE_REGIONS.clear()
    _CANDIDATE_REGIONS[id(candidate)] = (candidate, result)
    return result


def _analyze_candidate_regions_uncached(candidate: TimingCandidate) -> CandidateRegionEvidence:
    spans = candidate.spans
    parsed_gaps = tuple(max(0.0, right.start - left.end) for left, right in zip(spans, spans[1:]))
    positive_gaps = tuple(gap for gap in parsed_gaps if gap > 0.0)
    median_gap = float(statistics.median(positive_gaps)) if positive_gaps else None
    max_gap = max(parsed_gaps) if parsed_gaps else None
    duration = max(0.0, spans[-1].end - spans[0].start) if spans else None
    density = len(spans) / max(duration, HOP_SECONDS) if duration is not None else None
    longest_run, opening_run, later_run = _supported_run_metrics(spans, parsed_gaps)
    reliable_count = sum(span.score >= 0.08 for span in spans)
    bound_acoustic = candidate_has_bound_acoustic_evidence(candidate)
    bound_direct = candidate_has_bound_direct_onset_evidence(candidate)
    proof_end: float | None = _opening_mora_proof_end(spans, parsed_gaps)
    if proof_end is None and spans and spans[0].score >= 0.05 and opening_run >= 2:
        first_reliable = next(
            (span for span in spans[:opening_run] if span.score >= 0.08), None
        )
        if first_reliable is not None:
            proof_end = first_reliable.end
    elif (
        spans
        and spans[0].score >= 0.08
        and bound_direct
        and reliable_count >= 8
        and (max_gap is None or max_gap < CTC_SEVERE_INTERNAL_GAP_SECONDS)
        and duration is not None
        and duration >= 3.0
    ):
        # A bound independent vocal onset plus a reliable first CTC token can
        # establish the opening even when the second token is weak.  This does
        # not rescue fractured/short paths and therefore cannot validate the
        # collapsed-tail failure mode seen in late-song false positives.
        proof_end = spans[0].end
    elif (
        len(spans) >= 2
        and spans[0].score < 0.05
        and spans[1].score >= 0.08
        and (bound_acoustic or bound_direct)
        and spans[1].start - candidate.written_time.seconds
        <= PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
        and parsed_gaps[0] < CTC_SEVERE_INTERNAL_GAP_SECONDS
    ):
        proof_end = spans[1].end

    severe_gap_indexes = tuple(
        index
        for index, gap in enumerate(parsed_gaps)
        if gap >= CTC_SEVERE_INTERNAL_GAP_SECONDS
    )
    severe_gap_index: int | None = (
        severe_gap_indexes[0] if severe_gap_indexes else None
    )

    fracture_region: Literal["opening", "prefix", "middle", "tail", "unknown", "none"] = "none"
    if severe_gap_index is not None:
        left = spans[severe_gap_index]
        if proof_end is None:
            fracture_region = "unknown"
        elif left.end <= candidate.written_time.seconds:
            fracture_region = "opening"
        elif left.end < proof_end:
            fracture_region = "prefix"
        else:
            # A disconnected terminal island is not evidence that the onset itself
            # fractured.  Classify the path break after a proven opening as middle;
            # derive_reliable_tail() independently decides whether the terminal
            # island can own a cross-line boundary.  This keeps whole-row retries
            # from reintroducing the old "prefix fracture invalidates onset" bug.
            fracture_region = "middle"

    temporal: Literal["coherent", "ambiguous", "incoherent", "unavailable"]
    if not spans:
        temporal = "coherent" if bound_direct else "unavailable"
    elif candidate.prefix_scope and severe_gap_indexes:
        temporal = "incoherent"
    elif severe_gap_index is not None and fracture_region in {"opening", "prefix", "unknown"}:
        temporal = "incoherent"
    elif proof_end is None:
        temporal = "ambiguous"
    else:
        temporal = "coherent"

    region_payload = {
        "candidate_id": candidate.candidate_id,
        "generation_revision": candidate.generation_revision,
        "computed_for_centiseconds": candidate.written_time.centiseconds,
        "gaps": parsed_gaps,
        "duration": duration,
        "opening_supported_run": opening_run,
        "later_supported_run": later_run,
        "onset_proof_end": proof_end,
        "severe_gap_indexes": severe_gap_indexes,
        "temporal_coherence": temporal,
        "fracture_region": fracture_region,
        "right_edge_pileup": candidate.right_edge_pileup,
        "window_truncated": candidate.window_truncated,
    }
    return CandidateRegionEvidence(
        candidate_id=candidate.candidate_id,
        generation_revision=candidate.generation_revision,
        computed_for_centiseconds=candidate.written_time.centiseconds,
        region_revision=_timing_content_digest(region_payload),
        parsed_gaps=parsed_gaps,
        prefix_duration=round(duration, 6) if duration is not None else None,
        max_internal_gap=round(max_gap, 6) if max_gap is not None else None,
        median_internal_gap=round(median_gap, 6) if median_gap is not None else None,
        p90_internal_gap=round(_percentile(parsed_gaps, 0.90), 6) if parsed_gaps else None,
        p95_internal_gap=round(_percentile(parsed_gaps, 0.95), 6) if parsed_gaps else None,
        token_density=round(density, 6) if density is not None else None,
        longest_supported_run=longest_run,
        opening_supported_run=opening_run,
        later_supported_run=later_run,
        reliable_support_count=reliable_count,
        onset_proof_end=round(proof_end, 6) if proof_end is not None else None,
        severe_gap_indexes=severe_gap_indexes,
        temporal_coherence=temporal,
        fracture_region=fracture_region,
    )


def _tail_region_result(
    candidate: TimingCandidate,
    regions: CandidateRegionEvidence,
    validity: TailValidity,
    reason: str,
    start: float | None = None,
    end: float | None = None,
    uncertainty: float | None = None,
    confidence: float | None = None,
    terminal_span_indexes: tuple[int, ...] = (),
    local_gap_references: tuple[float, ...] = (),
) -> TailRegionEvidence:
    revision = _timing_content_digest({
        "previous_candidate_id": candidate.candidate_id,
        "previous_generation_revision": candidate.generation_revision,
        "previous_region_revision": regions.region_revision,
        "validity": validity,
        "start": start,
        "end": end,
        "uncertainty": uncertainty,
        "confidence": confidence,
        "reason": reason,
        "terminal_span_indexes": terminal_span_indexes,
        "local_gap_references": local_gap_references,
    })
    return TailRegionEvidence(
        candidate.candidate_id,
        candidate.generation_revision,
        regions.region_revision,
        revision,
        validity,
        start,
        end,
        uncertainty,
        confidence,
        reason,
        terminal_span_indexes,
        local_gap_references,
    )


def derive_reliable_tail(
    candidate: TimingCandidate,
    regions: CandidateRegionEvidence | None = None,
) -> TailRegionEvidence:
    regions = regions or analyze_candidate_regions(candidate)
    if (
        regions.candidate_id != candidate.candidate_id
        or regions.generation_revision != candidate.generation_revision
    ):
        return _tail_region_result(candidate, regions, "invalid", "tail-region-revision-mismatch")
    if candidate.right_edge_pileup or candidate.window_truncated:
        return _tail_region_result(candidate, regions, "invalid", "tail-window-invalid")
    if regions.fracture_region in {"tail", "unknown"}:
        return _tail_region_result(candidate, regions, "invalid", "tail-fracture-invalid")
    if candidate.identity_support != "supported" or candidate.sequence_support != "supported":
        return _tail_region_result(candidate, regions, "unknown", "tail-identity-or-sequence-unavailable")
    spans = candidate.spans
    attached_weak_terminal = False
    detached_low_reliability_terminal = False
    if (
        candidate.source == "ctc-local-window"
        and len(spans) >= 3
        and regions.onset_proof_end is not None
    ):
        terminal_gap = max(0.0, spans[-1].start - spans[-2].end)
        reliable_prefix_count = sum(span.score >= 0.08 for span in spans[:-1])
        if reliable_prefix_count >= 2 and spans[-2].score >= 0.08:
            if (
                spans[-1].score < 0.08
                and terminal_gap >= CTC_SEVERE_INTERNAL_GAP_SECONDS
            ):
                spans = spans[:-1]
                detached_low_reliability_terminal = True
            elif (
                spans[-1].score < 0.05
                and not regions.severe_gap_indexes
                and regions.p95_internal_gap is not None
                and terminal_gap <= max(HOP_SECONDS, float(regions.p95_internal_gap))
            ):
                attached_weak_terminal = True
    if len(spans) < 2 or (spans[-1].score < 0.05 and not attached_weak_terminal):
        return _tail_region_result(candidate, regions, "unknown", "no-terminal-supported-run")
    gaps = tuple(max(0.0, right.start - left.end) for left, right in zip(spans, spans[1:]))
    separating_gap_index = len(spans) - 2
    reference_start = separating_gap_index - 1
    while reference_start >= 0:
        if (
            gaps[reference_start] >= CTC_SEVERE_INTERNAL_GAP_SECONDS
            or spans[reference_start].score < 0.05
            or spans[reference_start + 1].score < 0.05
        ):
            break
        reference_start -= 1
    if reference_start >= 0 and gaps[reference_start] >= CTC_SEVERE_INTERNAL_GAP_SECONDS:
        return _tail_region_result(candidate, regions, "invalid", "tail-fracture-invalid")
    local_gap_references = tuple(
        gaps[index]
        for index in range(reference_start + 1, separating_gap_index)
        if gaps[index] > 0.0
    )
    if not local_gap_references:
        # A weak token immediately before an otherwise strong terminal pair can
        # cut the local reference walk even when the candidate has ample robust
        # cadence evidence elsewhere.  Fall back only to gaps whose two adjacent
        # tokens are both independently reliable, and only when the terminal pair
        # is itself reliable.  This reuses the existing reliability/cadence rules
        # rather than inventing a song-specific tail threshold.
        global_gap_references = tuple(
            gap
            for gap_index, gap in enumerate(gaps[:-1])
            if gap > 0.0
            and gap < CTC_SEVERE_INTERNAL_GAP_SECONDS
            and spans[gap_index].score >= 0.08
            and spans[gap_index + 1].score >= 0.08
        )
        if (
            spans[-2].score >= 0.08
            and spans[-1].score >= 0.08
            and global_gap_references
        ):
            local_gap_references = global_gap_references
        else:
            return _tail_region_result(
                candidate, regions, "unknown", "terminal-local-gap-reference-unavailable"
            )
    median_gap = float(statistics.median(local_gap_references))
    contiguous_limit = max(HOP_SECONDS, median_gap * 3.0)
    if gaps[separating_gap_index] >= CTC_SEVERE_INTERNAL_GAP_SECONDS:
        return _tail_region_result(candidate, regions, "invalid", "tail-fracture-invalid")
    if gaps[separating_gap_index] > contiguous_limit:
        return _tail_region_result(
            candidate,
            regions,
            "unknown",
            "detached-terminal-ghost",
            local_gap_references=local_gap_references,
        )
    start_index = len(spans) - 1
    reliable_count = 1 if spans[-1].score >= 0.08 else 0
    for index in range(len(spans) - 2, -1, -1):
        gap = max(0.0, spans[index + 1].start - spans[index].end)
        if gap > contiguous_limit or spans[index].score < 0.05:
            break
        start_index = index
        reliable_count += int(spans[index].score >= 0.08)
    tail = spans[start_index:]
    if len(tail) < 2 or reliable_count < 1:
        return _tail_region_result(candidate, regions, "unknown", "terminal-run-not-reliable")
    if any(index == start_index - 1 for index in regions.severe_gap_indexes):
        return _tail_region_result(candidate, regions, "invalid", "tail-fracture-invalid")
    tail_uncertainty = contiguous_limit
    tail_reason = "reliable-terminal-run"
    if attached_weak_terminal:
        tail_uncertainty = max(HOP_SECONDS, spans[-1].end - spans[-2].end)
        tail_reason = "local-window-attached-weak-terminal"
    elif detached_low_reliability_terminal:
        tail_reason = "local-window-detached-low-reliability-terminal-ignored"
    return _tail_region_result(
        candidate,
        regions,
        "valid",
        tail_reason,
        tail[0].start,
        tail[-1].end,
        tail_uncertainty,
        statistics.mean(span.score for span in tail),
        tuple(range(start_index, len(spans))),
        local_gap_references,
    )


def _bind_independent_tail_to_candidate(
    candidate: TimingCandidate,
    evidence: IndependentTailEvidence,
) -> TailRegionEvidence:
    """Re-bind runtime tail evidence to the selected predecessor candidate.

    The external row proves a lyric-tail upper bound, not the predecessor's
    start timestamp.  Binding the evidence to the selected predecessor keeps
    the ownership relation revision-safe without forcing that external backend
    to become the selected timing candidate.
    """
    regions = analyze_candidate_regions(candidate)
    provenance = evidence.provenance
    authority_backend_lineage = (
        provenance.backend_lineage if provenance is not None else None
    )
    authority_producer = provenance.producer if provenance is not None else None
    authority_observation_revision = (
        provenance.observation_revision if provenance is not None else None
    )
    authority_metadata_valid = bool(
        isinstance(authority_backend_lineage, str)
        and authority_backend_lineage.strip()
        and _tail_backend_family(authority_backend_lineage) is not None
        and isinstance(authority_producer, str)
        and authority_producer.strip()
        and isinstance(authority_observation_revision, str)
        and authority_observation_revision.strip()
    )
    payload = {
        "previous_candidate_id": candidate.candidate_id,
        "previous_generation_revision": candidate.generation_revision,
        "previous_region_revision": regions.region_revision,
        "external_entry_index": evidence.entry_index,
        "external_tail_start": round(evidence.tail_start, 6),
        "external_tail_end": round(evidence.tail_end, 6),
        "external_uncertainty_seconds": round(evidence.uncertainty_seconds, 6),
        "external_confidence": round(evidence.confidence, 6),
        "external_producer": evidence.producer,
        "external_source_revision": evidence.source_revision,
        "external_provenance": (
            _tail_provenance_payload(evidence.provenance)
            if evidence.provenance is not None else None
        ),
    }
    return TailRegionEvidence(
        previous_candidate_id=candidate.candidate_id,
        previous_generation_revision=candidate.generation_revision,
        previous_region_revision=regions.region_revision,
        tail_evidence_revision=_timing_content_digest(payload),
        validity="valid" if authority_metadata_valid else "unknown",
        reliable_tail_start=float(evidence.tail_start) if authority_metadata_valid else None,
        reliable_tail_end=float(evidence.tail_end) if authority_metadata_valid else None,
        uncertainty_seconds=(
            max(0.0, float(evidence.uncertainty_seconds))
            if authority_metadata_valid else None
        ),
        confidence=(
            _finite_unit_confidence(evidence.confidence)
            if authority_metadata_valid else None
        ),
        reason=(
            f"independent-{evidence.producer}-tail"
            if authority_metadata_valid
            else "independent-tail-authority-metadata-missing"
        ),
        terminal_span_indexes=(),
        local_gap_references=(),
        authority_backend_lineage=authority_backend_lineage if authority_metadata_valid else None,
        authority_producer=authority_producer if authority_metadata_valid else None,
        authority_observation_revision=(
            authority_observation_revision if authority_metadata_valid else None
        ),
    )


def _normalize_tail_observations(
    independent: IndependentTailEvidence | TailObservationSet | None,
) -> TailObservationSet:
    if independent is None:
        return ()
    if isinstance(independent, IndependentTailEvidence):
        return (independent,)
    if not isinstance(independent, tuple):
        return ()
    return tuple(
        item for item in independent
        if isinstance(item, IndependentTailEvidence)
    )


def _authoritative_tail_observations(
    candidate: TimingCandidate,
    observations: TailObservationSet,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
) -> TailObservationSet:
    if not _tail_contexts_are_valid(tail_evidence_context):
        return ()
    candidate_families = _candidate_support_families(candidate)
    if not candidate_families:
        return ()
    selected: list[IndependentTailEvidence] = []
    for observation in observations:
        if observation.entry_index != candidate.entry_index:
            continue
        if (
            not _tail_provenance_is_valid(observation)
            or not _tail_provenance_matches_context(observation, tail_evidence_context)
        ):
            continue
        evidence_family = _tail_backend_family(
            observation.provenance.backend_lineage if observation.provenance else None
        )
        if evidence_family is None or evidence_family in candidate_families:
            continue
        selected.append(observation)
    return tuple(sorted(
        selected,
        key=lambda item: (
            -(item.tail_end + max(0.0, item.uncertainty_seconds)),
            -item.tail_end,
            item.provenance.observation_revision if item.provenance else item.source_revision,
        ),
    ))


def resolve_reliable_tail(
    candidate: TimingCandidate,
    independent: IndependentTailEvidence | TailObservationSet | None = None,
    *,
    tail_evidence_context: TailEvidenceContext | tuple[TailEvidenceContext, ...] | None = None,
) -> TailRegionEvidence:
    """Return the most conservative valid predecessor-tail proof available.

    Intrinsic CTC tail geometry remains the default.  Only a provenance-valid
    observation whose backend lineage is disjoint from the selected predecessor
    candidate may replace UNKNOWN/invalid intrinsic evidence or tighten an
    already-valid release bound.  When both are valid the later
    ``tail_end + uncertainty`` wins, so fusion can never make overlap easier to
    clear than either source.  Same-lineage observations remain diagnostic.
    A missing or stale execution context makes every external observation
    diagnostic-only; intrinsic geometry is still evaluated normally.
    """
    intrinsic = derive_reliable_tail(candidate)
    observations = _authoritative_tail_observations(
        candidate,
        _normalize_tail_observations(independent),
        tail_evidence_context,
    )
    if not observations:
        return intrinsic
    external = _bind_independent_tail_to_candidate(candidate, observations[0])
    external_upper = external.reliable_tail_end + (external.uncertainty_seconds or 0.0)
    if intrinsic.validity != "valid" or intrinsic.reliable_tail_end is None:
        # A disjoint backend may resolve an unknown predecessor tail, but it
        # cannot erase reliable support that the selected predecessor itself
        # still carries beyond that backend's claimed release boundary.
        if any(
            span.score >= 0.08 and span.start > external_upper
            for span in candidate.spans
        ):
            return intrinsic
        return external
    intrinsic_upper = intrinsic.reliable_tail_end + (intrinsic.uncertainty_seconds or 0.0)
    if intrinsic_upper >= external_upper:
        return intrinsic
    return external


def _lineage_families_from_labels(
    *labels: str,
    source: str | None = None,
) -> frozenset[str]:
    """Map existing producer/source labels to coarse backend lineages."""
    normalized = " ".join(
        label.lower()
        for label in labels
        if isinstance(label, str) and label
    )
    families: set[str] = set()
    if (
        "local-whisper" in normalized
        or "whisperx" in normalized
        or "whispercpp" in normalized
        or "raw-asr" in normalized
        or (isinstance(source, str) and source.lower().startswith("raw-"))
    ):
        families.add("whisper-family")
    if "ctc" in normalized or "mms" in normalized:
        families.add("ctc")
    if "recurrence" in normalized:
        families.add("recurrence")
    if "hubert" in normalized:
        families.add("hubert-ja-known-lyric")
    return frozenset(families)


def _candidate_support_families(candidate: TimingCandidate) -> frozenset[str]:
    """Return coarse *independent* evidence lineages for consensus.

    Global raw ASR, bounded local Whisper, and WhisperX-assisted observations all
    ultimately depend on the same Whisper recognizer stack for lyric identity.
    Treating them as separate votes would manufacture independence.  They therefore
    collapse to one ``whisper-family`` lineage.  CTC and recurrence remain separate
    lineages because their evidence is produced independently.
    """
    return _lineage_families_from_labels(
        candidate.source,
        candidate.direct_onset_evidence.producer if candidate.direct_onset_evidence else "",
        candidate.acoustic_evidence.producer if candidate.acoustic_evidence else "",
        source=candidate.source,
    )
