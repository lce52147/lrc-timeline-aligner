"""Extracted from acoustic_hypotheses.py (split2): local-onset, mutual recurrence, phonetic onset fusion and trusted tail release hypothesis builders."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path
import numpy as np
from report_trust import _record_backend_participation_failure, assignment_content_is_trusted
from timing_core import EvaluatedBackendTiming, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, _finite_unit_confidence, _tail_evidence_context_from_report, analyze_candidate_regions, record_alignment_hypothesis, resolve_reliable_tail, runtime_independent_tail_evidence
from lyrics_io import LyricEntry, entry_sung_text, normalize_match_text
from audio_features import AudioFeatures, _recurrence_segment, acoustic_pitchclass_recurrence_features, acoustic_recurrence_similarity, analyze_vocal_onsets
from runtime_base import LrcError, _path_sha256, _timing_content_digest
from runtime_base import TimingTokenSpan
from vocal_onset_hypotheses import (
    _nearest_local_maximum,
)


# Phase C evidence producers deliberately use different proof shapes instead of
# weakening the central timing invariants.  These thresholds are producer
# acceptance gates; they do not alter minimum-proof, ownership, or evaluator
# trust thresholds.
MUTUAL_RECURRENCE_FEATURE_REVISION = "pitchclass-temporal-smooth-v1"


MUTUAL_RECURRENCE_MIN_SIMILARITY = 0.72


MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN = 0.12


MUTUAL_RECURRENCE_ANCHOR_RADIUS_SECONDS = 0.36


MUTUAL_RECURRENCE_PEER_RADIUS_SECONDS = 1.80


MUTUAL_RECURRENCE_DISTINCT_SECONDS = 0.75


PHONETIC_ONSET_FEATURE_REVISION = "ctc-first-token-plus-program-onset-v2"


PHONETIC_ONSET_MIN_POSTERIOR = 0.08


PHONETIC_ONSET_RADIUS_SECONDS = 0.22


PHONETIC_ONSET_MIN_STRENGTH = 0.50


PHONETIC_ONSET_MIN_PROMINENCE = 0.20


TAIL_RELEASE_FEATURE_REVISION = "trusted-tail-release-cluster-onset-v3"


TAIL_RELEASE_ONSET_MIN_STRENGTH = 0.82


TAIL_RELEASE_ONSET_MIN_PROMINENCE = 0.50


TAIL_RELEASE_CLUSTER_BREAK_SECONDS = 0.24


TAIL_RELEASE_CONTEXT_SECONDS = 0.80


def _project_recurrence_spans_from_geometry(
    spans: tuple[TimingTokenSpan, ...],
    reference_start: float,
    candidate_start: float,
    recurrence_score: float,
) -> list[dict[str, object]]:
    if not spans:
        return []
    projected: list[dict[str, object]] = []
    for span in spans:
        projected.append({
            "char": span.token,
            "start": round(candidate_start + (span.start - reference_start), 3),
            "end": round(candidate_start + (span.end - reference_start), 3),
            "score": round(min(span.score, max(0.08, recurrence_score)), 6),
        })
    return projected


def _local_onset_candidates(
    features: AudioFeatures,
    start: float,
    end: float,
    *,
    min_strength: float,
    min_prominence: float,
    min_separation: float = 0.18,
) -> list[dict[str, float]]:
    if end <= start or features.frame_times.size == 0:
        return []
    indices = np.flatnonzero(
        (features.frame_times >= start) & (features.frame_times <= end)
    )
    raw: list[dict[str, float]] = []
    for index in indices:
        strength = float(features.onset_strength[index])
        if strength < min_strength:
            continue
        left = float(features.onset_strength[index - 1]) if index > 0 else -1.0
        right = (
            float(features.onset_strength[index + 1])
            if index + 1 < features.onset_strength.size else -1.0
        )
        if strength < left or strength < right:
            continue
        time = float(features.frame_times[index])
        context = np.flatnonzero(
            (features.frame_times >= time - 0.80)
            & (features.frame_times <= time + 0.80)
        )
        baseline = float(np.median(features.onset_strength[context])) if context.size else 0.0
        prominence = strength - baseline
        if prominence < min_prominence:
            continue
        raw.append({"time": time, "strength": strength, "prominence": prominence})
    raw.sort(key=lambda item: (-item["strength"], -item["prominence"], item["time"]))
    selected: list[dict[str, float]] = []
    for item in raw:
        if all(abs(item["time"] - kept["time"]) >= min_separation for kept in selected):
            selected.append(item)
    return sorted(selected, key=lambda item: item["time"])


def _first_post_tail_cluster_onset(
    features: AudioFeatures,
    tail_end: float,
    search_end: float,
    *,
    min_strength: float,
    min_prominence: float,
) -> dict[str, float] | None:
    """Return the first strong onset that begins a new post-tail onset cluster.

    A reliable-tail boundary may still contain several high-energy sub-onsets
    from the predecessor's final syllable/instrumental accent.  Picking the
    strongest peak in the release window systematically biases late.  Instead,
    retain temporal structure: a candidate must be the first qualifying peak
    after ``tail_end`` that follows a genuine quiet separation from the prior
    qualifying peak.  If no cluster break is observable, fail closed.
    """
    context_start = max(0.0, tail_end - TAIL_RELEASE_CONTEXT_SECONDS)
    candidates = _local_onset_candidates(
        features,
        context_start,
        search_end,
        min_strength=min_strength,
        min_prominence=min_prominence,
        min_separation=0.04,
    )
    previous: dict[str, float] | None = None
    for candidate in candidates:
        if candidate["time"] < tail_end:
            previous = candidate
            continue
        if previous is not None:
            gap = candidate["time"] - previous["time"]
            if gap >= TAIL_RELEASE_CLUSTER_BREAK_SECONDS:
                return {
                    **candidate,
                    "previous_onset_time": previous["time"],
                    "cluster_gap": gap,
                    "context_start": context_start,
                }
        previous = candidate
    return None


def _self_bound_hypothesis_record(hypothesis: object) -> bool:
    if not isinstance(hypothesis, dict):
        return False
    hypothesis_id = hypothesis.get("hypothesis_id")
    if not isinstance(hypothesis_id, str) or not hypothesis_id:
        return False
    payload = copy.deepcopy(hypothesis)
    payload.pop("hypothesis_id", None)
    return _timing_content_digest(payload) == hypothesis_id


def _bound_local_occurrence_identity(assignment: dict[str, object]) -> dict[str, object] | None:
    """Return immutable whole-line local CTC identity evidence for an occurrence.

    Mutual recurrence may locate the boundary acoustically, but it must not
    bootstrap lyric identity from recurrence similarity alone.  Each side is
    therefore required to have a self-bound local CTC hypothesis that
    independently supports the full lyric identity and sequence.  In addition
    to the ordinary local-window producer, a completed bounded retry may
    contribute an *identity-only* row.  That retry row is never itself a timing
    candidate and is rejected here when its helper window was truncated or
    piled up at the right edge.  Temporal incoherence is allowed because this
    evidence establishes occurrence identity only; the recurrence candidate
    must still satisfy central onset/ownership proof separately.
    """
    hypotheses = assignment.get("alignment_hypotheses")
    if not isinstance(hypotheses, list):
        return None
    eligible: list[dict[str, object]] = []
    identity_sources = {"ctc-local-window", "ctc-opening-local-retry-identity"}
    for hypothesis in hypotheses:
        if (
            not isinstance(hypothesis, dict)
            or hypothesis.get("source") not in identity_sources
            or hypothesis.get("authoritative") is not False
            or hypothesis.get("identity_support") != "supported"
            or hypothesis.get("sequence_support") != "supported"
            or not _self_bound_hypothesis_record(hypothesis)
        ):
            continue
        if (
            hypothesis.get("source") == "ctc-opening-local-retry-identity"
            and (
                bool(hypothesis.get("window_truncated"))
                or bool(hypothesis.get("right_edge_pileup"))
            )
        ):
            continue
        spans = hypothesis.get("spans")
        if not isinstance(spans, list) or len(spans) < 4:
            continue
        try:
            raw_time = float(hypothesis.get("raw_candidate_time"))
            confidence = float(hypothesis.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(raw_time):
            continue
        eligible.append({
            "hypothesis_id": hypothesis["hypothesis_id"],
            "source": str(hypothesis.get("source")),
            "raw_candidate_time": round(raw_time, 6),
            "confidence": round(_finite_unit_confidence(confidence), 8),
            "span_digest": _timing_content_digest(spans),
        })
    if not eligible:
        return None
    return max(eligible, key=lambda row: (float(row["confidence"]), str(row["hypothesis_id"])))


def _candidate_seed_rows(assignment: dict[str, object]) -> list[dict[str, object]]:
    """Return non-authoritative phonetic seeds with explicit provenance.

    Seeds locate a broad occurrence neighborhood only.  They cannot themselves
    satisfy central timing proof and are never copied into the final timestamp
    without independent acoustic evidence.
    """
    rows: list[dict[str, object]] = []

    def semantic_rank(row: dict[str, object]) -> tuple[float, float, int, float]:
        kind_priority = {
            "ctc-local-first-token": 3,
            "ctc-local-retry-first-token": 2,
            "ctc-detached-second-token": 1,
            "ctc-first-token-posterior": 0,
        }
        return (
            float(row.get("strength", 0.0)),
            float(row.get("first_token_score", 0.0)),
            kind_priority.get(str(row.get("kind")), -1),
            -float(row.get("time", 0.0)),
        )
    hypotheses = assignment.get("alignment_hypotheses")
    if isinstance(hypotheses, list):
        for hypothesis in hypotheses:
            if (
                not isinstance(hypothesis, dict)
                or hypothesis.get("source") not in {
                    "ctc-local-window",
                    "ctc-opening-local-retry-identity",
                }
            ):
                continue
            if (
                hypothesis.get("source") == "ctc-opening-local-retry-identity"
                and (
                    bool(hypothesis.get("window_truncated"))
                    or bool(hypothesis.get("right_edge_pileup"))
                )
            ):
                continue
            spans = hypothesis.get("spans")
            if not isinstance(spans, list) or not spans or not isinstance(spans[0], dict):
                continue
            try:
                seed_time = float(spans[0].get("start"))
                first_score = float(spans[0].get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(seed_time):
                continue
            rows.append({
                "time": seed_time,
                "strength": 0.55 + min(0.44, max(0.0, first_score)),
                "kind": (
                    "ctc-local-retry-first-token"
                    if hypothesis.get("source") == "ctc-opening-local-retry-identity"
                    else "ctc-local-first-token"
                ),
                "evidence_id": str(hypothesis.get("hypothesis_id") or ""),
                "first_token_score": first_score,
            })

    legacy = assignment.get("candidates")
    if isinstance(legacy, list):
        for candidate in legacy:
            if not isinstance(candidate, dict) or candidate.get("source") != "ctc_detached_initial_token_recovery":
                continue
            try:
                seed_time = float(candidate.get("time"))
            except (TypeError, ValueError):
                continue
            rows.append({
                "time": seed_time,
                "strength": 0.82,
                "kind": "ctc-detached-second-token",
                "evidence_id": _timing_content_digest(candidate),
                "first_token_score": 0.0,
            })

    peaks = assignment.get("ctc_first_token_candidates")
    if isinstance(peaks, list):
        for peak in peaks:
            if not isinstance(peak, dict):
                continue
            try:
                seed_time = float(peak.get("time"))
                score = float(peak.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if score <= 0.0 or not math.isfinite(seed_time):
                continue
            rows.append({
                "time": seed_time,
                "strength": 0.50 + min(0.30, math.sqrt(max(0.0, score)) * 0.75),
                "kind": "ctc-first-token-posterior",
                "evidence_id": _timing_content_digest(peak),
                "first_token_score": score,
            })

    # Keep only seeds in the same broad neighborhood as a local-window result
    # when such a result exists.  This prevents a whole-song first-token echo
    # from becoming a mutual-recurrence anchor.
    local_seed_kinds = {"ctc-local-first-token", "ctc-local-retry-first-token"}
    local_times = [
        float(row["time"]) for row in rows if row.get("kind") in local_seed_kinds
    ]
    if local_times:
        local_time = max(
            (row for row in rows if row.get("kind") in local_seed_kinds),
            key=semantic_rank,
        )["time"]
        filtered = [
            row for row in rows
            if row.get("kind") == "ctc-detached-second-token"
            or row.get("kind") in local_seed_kinds
            or abs(float(row["time"]) - float(local_time)) <= 1.25
        ]
        if filtered:
            rows = filtered
    return sorted(
        rows,
        key=lambda row: (
            -semantic_rank(row)[0],
            -semantic_rank(row)[1],
            -semantic_rank(row)[2],
            -semantic_rank(row)[3],
            str(row.get("evidence_id")),
        ),
    )


def _best_mutual_recurrence_pair(
    features: np.ndarray,
    onset_features: AudioFeatures,
    left_seed: dict[str, object],
    right_seed: dict[str, object],
    *,
    left_bounds: tuple[float, float],
    right_bounds: tuple[float, float],
) -> dict[str, float] | None:
    left_seed_time = float(left_seed["time"])
    right_seed_time = float(right_seed["time"])
    left_is_anchor = float(left_seed.get("strength", 0.0)) >= float(right_seed.get("strength", 0.0))
    left_radius = MUTUAL_RECURRENCE_ANCHOR_RADIUS_SECONDS if left_is_anchor else MUTUAL_RECURRENCE_PEER_RADIUS_SECONDS
    right_radius = MUTUAL_RECURRENCE_PEER_RADIUS_SECONDS if left_is_anchor else MUTUAL_RECURRENCE_ANCHOR_RADIUS_SECONDS
    left_onsets = _local_onset_candidates(
        onset_features,
        max(left_bounds[0], left_seed_time - left_radius),
        min(left_bounds[1], left_seed_time + left_radius),
        min_strength=0.45,
        min_prominence=0.10,
    )
    right_onsets = _local_onset_candidates(
        onset_features,
        max(right_bounds[0], right_seed_time - right_radius),
        min(right_bounds[1], right_seed_time + right_radius),
        min_strength=0.45,
        min_prominence=0.10,
    )
    if not left_onsets or not right_onsets:
        return None
    candidates: list[dict[str, float]] = []
    durations = (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)
    for left in left_onsets:
        for right in right_onsets:
            for template_duration in durations:
                if left["time"] + template_duration > left_bounds[1] + 0.25:
                    continue
                if right["time"] + template_duration > right_bounds[1] + 0.25:
                    continue
                score = acoustic_recurrence_similarity(
                    _recurrence_segment(features, left["time"], left["time"] + template_duration),
                    _recurrence_segment(features, right["time"], right["time"] + template_duration),
                )
                candidates.append({
                    "score": score,
                    "left": left["time"],
                    "right": right["time"],
                    "duration": template_duration,
                    "left_strength": left["strength"],
                    "left_prominence": left["prominence"],
                    "right_strength": right["strength"],
                    "right_prominence": right["prominence"],
                })
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row["score"], row["left"], row["right"], row["duration"]))
    best = candidates[0]
    runner = next(
        (
            row for row in candidates[1:]
            if abs(row["left"] - best["left"]) >= MUTUAL_RECURRENCE_DISTINCT_SECONDS
            or abs(row["right"] - best["right"]) >= MUTUAL_RECURRENCE_DISTINCT_SECONDS
        ),
        None,
    )
    runner_score = float(runner["score"]) if runner is not None else -1.0
    best = dict(best)
    best["runner_score"] = runner_score
    best["margin"] = float(best["score"] - runner_score)
    return best


def _trusted_bounds_for_entry(
    evaluated: EvaluatedBackendTiming,
    index: int,
    duration: float,
) -> tuple[float, float]:
    trusted = [bool(audit.timing_trusted) for audit in evaluated.state.audits]
    lower = next(
        (
            evaluated.state.decisions[position].written_time.seconds + 0.20
            for position in range(index - 1, -1, -1)
            if trusted[position]
        ),
        0.0,
    )
    upper = next(
        (
            evaluated.state.decisions[position].written_time.seconds - 0.20
            for position in range(index + 1, len(trusted))
            if trusted[position]
        ),
        duration,
    )
    return max(0.0, lower), min(duration, upper)


def apply_phonetic_onset_fusion_hypotheses(
    alignment_audio_path: Path,
    entries: list[LyricEntry],
    report: dict[str, object],
    evaluated: EvaluatedBackendTiming,
    duration: float,
) -> list[dict[str, object]]:
    """Fuse a coherent CTC first-token posterior with independent audio onset.

    This is not a threshold relaxation of the existing CTC candidate.  It emits
    a span-free challenger only when the whole-line CTC path is coherent, the
    assignment content is already trusted, a first-token posterior is strong,
    and original-program audio independently places an onset within the strict
    association radius.  The baseline timestamp is never changed here.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return []
    try:
        onset_features = analyze_vocal_onsets(alignment_audio_path, duration)
        audio_sha = _path_sha256(alignment_audio_path)
    except (LrcError, OSError) as exc:
        _record_backend_participation_failure(
            report, "apply_phonetic_onset_fusion_hypotheses", exc
        )
        return []
    generated: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict) or evaluated.state.audits[index].timing_trusted:
            continue
        if not assignment_content_is_trusted(assignment):
            continue
        current = evaluated.state.decisions[index].candidate
        regions = analyze_candidate_regions(current)
        if (
            current.source not in {"ctc-current", "ctc-vocal-onset-confirmed"}
            or current.identity_support != "supported"
            or not current.spans
            or regions.temporal_coherence != "coherent"
            or current.right_edge_pileup
            or current.window_truncated
        ):
            continue
        peaks = assignment.get("ctc_first_token_candidates")
        if not isinstance(peaks, list):
            continue
        parsed: list[tuple[float, float, dict[str, object]]] = []
        for peak in peaks:
            if not isinstance(peak, dict):
                continue
            try:
                peak_time = float(peak.get("time"))
                peak_score = float(peak.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if (
                peak_score >= PHONETIC_ONSET_MIN_POSTERIOR
                and abs(peak_time - current.written_time.seconds) <= 0.50
            ):
                parsed.append((peak_score, peak_time, peak))
        if not parsed:
            continue
        peak_score, peak_time, peak_row = max(parsed, key=lambda item: (item[0], -abs(item[1] - current.written_time.seconds)))
        acoustic = _nearest_local_maximum(onset_features, peak_time, PHONETIC_ONSET_RADIUS_SECONDS)
        if acoustic is None:
            continue
        onset_time, onset_strength, onset_prominence = acoustic
        if onset_strength < PHONETIC_ONSET_MIN_STRENGTH or onset_prominence < PHONETIC_ONSET_MIN_PROMINENCE:
            continue
        if abs(onset_time - peak_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS:
            continue
        window = {
            "audio_sha256": audio_sha,
            "audio_duration_seconds": round(float(duration), 6),
            "current_candidate_id": current.candidate_id,
            "current_raw_time": round(current.raw_time, 6),
            "current_written_centiseconds": current.written_time.centiseconds,
            "current_time": round(current.written_time.seconds, 6),
            "ctc_peak_time": round(peak_time, 6),
            "ctc_peak_score": round(peak_score, 8),
            "ctc_peak_digest": _timing_content_digest(peak_row),
            "onset_time": (round(onset_time, 6) if onset_time is not None else None),
            "onset_strength": round(onset_strength, 8),
            "onset_prominence": round(onset_prominence, 8),
            "feature_revision": PHONETIC_ONSET_FEATURE_REVISION,
        }
        before = len(assignment.get("alignment_hypotheses", [])) if isinstance(assignment.get("alignment_hypotheses"), list) else 0
        hypothesis = record_alignment_hypothesis(
            assignment,
            source="ctc-phonetic-onset-fusion",
            stage="independent-phonetic-onset-fusion",
            raw_time=peak_time,
            confidence=min(1.0, 0.70 + peak_score),
            spans=[],
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=onset_strength,
            acoustic_onset_time=onset_time,
            acoustic_evidence_producer="phonetic-onset-fusion",
            acoustic_evidence_kind="phonetic-onset",
            direct_onset_time=onset_time,
            direct_onset_evidence_producer="phonetic-onset-fusion",
            direct_onset_evidence_kind="phonetic-onset",
            parent_source=current.source,
            parent_time=current.raw_time,
            window=window,
            mode="coherent-ctc-first-token-plus-independent-onset",
        )
        after = len(assignment.get("alignment_hypotheses", [])) if isinstance(assignment.get("alignment_hypotheses"), list) else before
        if after > before:
            generated.append({
                "entry": index + 1,
                "candidate": round(peak_time, 3),
                "onset": round(onset_time, 3),
                "posterior": round(peak_score, 6),
                "hypothesis_id": hypothesis.get("hypothesis_id"),
            })
    report["phonetic_onset_fusion_hypotheses"] = generated
    report["phonetic_onset_fusion_count"] = len(generated)
    report["phonetic_onset_fusion_authoritative"] = False
    report["phonetic_onset_fusion_mutates_baseline"] = False
    return generated


def apply_mutual_acoustic_recurrence_hypotheses(
    alignment_audio_path: Path,
    entries: list[LyricEntry],
    report: dict[str, object],
    evaluated: EvaluatedBackendTiming,
    duration: float,
) -> list[dict[str, object]]:
    """Resolve a deadlocked exact-duplicate pair without trusting either side.

    Both occurrences must be timing-untrusted, both must have independent CTC
    seed evidence, and a reciprocal raw-audio recurrence must pass a strong
    similarity + distinct-alternative margin.  Two hypotheses with the same
    pair id are emitted together; one cannot gain independent identity without
    the reciprocal peer row being present and provenance-valid.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return []
    try:
        features = acoustic_pitchclass_recurrence_features(alignment_audio_path)
        onset_features = analyze_vocal_onsets(alignment_audio_path, duration)
        audio_sha = _path_sha256(alignment_audio_path)
    except (LrcError, OSError) as exc:
        _record_backend_participation_failure(
            report, "apply_mutual_acoustic_recurrence_hypotheses", exc
        )
        return []
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        normalized = normalize_match_text(entry_sung_text(entry))
        if normalized:
            groups.setdefault(normalized, []).append(index)
    generated: list[dict[str, object]] = []
    for normalized, indexes in groups.items():
        untrusted = [index for index in indexes if not evaluated.state.audits[index].timing_trusted]
        trusted = [index for index in indexes if evaluated.state.audits[index].timing_trusted]
        # Trusted-reference recurrence owns mixed groups.  Mutual recurrence is
        # intentionally limited to an unambiguous two-occurrence deadlock.
        if trusted or len(indexes) != 2 or len(untrusted) != 2:
            continue
        left_index, right_index = sorted(untrusted)
        left_assignment = assignments[left_index]
        right_assignment = assignments[right_index]
        if not isinstance(left_assignment, dict) or not isinstance(right_assignment, dict):
            continue
        left_identity = _bound_local_occurrence_identity(left_assignment)
        right_identity = _bound_local_occurrence_identity(right_assignment)
        if left_identity is None or right_identity is None:
            continue
        left_seeds = _candidate_seed_rows(left_assignment)
        right_seeds = _candidate_seed_rows(right_assignment)
        if not left_seeds or not right_seeds:
            continue
        left_seed = left_seeds[0]
        right_seed = right_seeds[0]
        left_bounds = _trusted_bounds_for_entry(evaluated, left_index, duration)
        right_bounds = _trusted_bounds_for_entry(evaluated, right_index, duration)
        if not (left_bounds[0] < float(left_seed["time"]) < left_bounds[1]):
            continue
        if not (right_bounds[0] < float(right_seed["time"]) < right_bounds[1]):
            continue
        result = _best_mutual_recurrence_pair(
            features,
            onset_features,
            left_seed,
            right_seed,
            left_bounds=left_bounds,
            right_bounds=right_bounds,
        )
        if result is None:
            continue
        score = float(result["score"])
        margin = float(result["margin"])
        if score < MUTUAL_RECURRENCE_MIN_SIMILARITY or margin < MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN:
            continue
        pair_payload = {
            "audio_sha256": audio_sha,
            "feature_revision": MUTUAL_RECURRENCE_FEATURE_REVISION,
            "normalized_lyric_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            "left_entry": left_index + 1,
            "right_entry": right_index + 1,
            "left_seed": left_seed,
            "right_seed": right_seed,
            "left_identity_evidence": left_identity,
            "right_identity_evidence": right_identity,
            "left_bounds": [round(left_bounds[0], 3), round(left_bounds[1], 3)],
            "right_bounds": [round(right_bounds[0], 3), round(right_bounds[1], 3)],
            "left_candidate": round(float(result["left"]), 6),
            "right_candidate": round(float(result["right"]), 6),
            "template_duration": round(float(result["duration"]), 3),
            "similarity": round(score, 8),
            "runner_similarity": round(float(result["runner_score"]), 8),
            "distinct_margin": round(margin, 8),
        }
        pair_id = _timing_content_digest(pair_payload)
        pair_payload["pair_id"] = pair_id
        emitted: list[dict[str, object]] = []
        for target_index, peer_index, candidate_key, peer_key, strength_key, prominence_key in (
            (left_index, right_index, "left_candidate", "right_candidate", "left_strength", "left_prominence"),
            (right_index, left_index, "right_candidate", "left_candidate", "right_strength", "right_prominence"),
        ):
            assignment = assignments[target_index]
            assert isinstance(assignment, dict)
            candidate_time = float(pair_payload[candidate_key])
            onset_strength = float(result[strength_key])
            onset_prominence = float(result[prominence_key])
            window = {
                **pair_payload,
                "target_entry": target_index + 1,
                "peer_entry": peer_index + 1,
                "peer_candidate": float(pair_payload[peer_key]),
                "onset_strength": round(onset_strength, 8),
                "onset_prominence": round(onset_prominence, 8),
            }
            before = len(assignment.get("alignment_hypotheses", [])) if isinstance(assignment.get("alignment_hypotheses"), list) else 0
            hypothesis = record_alignment_hypothesis(
                assignment,
                source="mutual-acoustic-recurrence",
                stage="occurrence-level-mutual-recurrence-search",
                raw_time=candidate_time,
                confidence=min(1.0, score),
                spans=[],
                identity_support="supported",
                sequence_support="supported",
                acoustic_support="supported",
                direct_onset_support="supported",
                acoustic_strength=onset_strength,
                acoustic_onset_time=candidate_time,
                acoustic_evidence_producer="mutual-recurrence-search",
                acoustic_evidence_kind="recurrence-onset",
                direct_onset_time=candidate_time,
                direct_onset_evidence_producer="mutual-recurrence-search",
                direct_onset_evidence_kind="recurrence-onset",
                parent_source="reciprocal-untrusted-duplicate-pair",
                parent_time=float(left_seed["time"] if target_index == left_index else right_seed["time"]),
                window=window,
                mode="reciprocal-exact-duplicate-pitchclass-recurrence",
            )
            after = len(assignment.get("alignment_hypotheses", [])) if isinstance(assignment.get("alignment_hypotheses"), list) else before
            if after > before:
                emitted.append({
                    "entry": target_index + 1,
                    "peer_entry": peer_index + 1,
                    "candidate": round(candidate_time, 3),
                    "similarity": round(score, 6),
                    "distinct_margin": round(margin, 6),
                    "pair_id": pair_id,
                    "hypothesis_id": hypothesis.get("hypothesis_id"),
                })
        # A reciprocal pair is atomic.  If only one side was new because the
        # other already exists, retaining the existing peer is valid; if neither
        # row exists there is nothing to report.  Provenance validation later
        # requires both reciprocal rows in the ledger.
        generated.extend(emitted)
    report["mutual_acoustic_recurrence_hypotheses"] = generated
    report["mutual_acoustic_recurrence_count"] = len(generated)
    report["mutual_acoustic_recurrence_authoritative"] = False
    report["mutual_acoustic_recurrence_mutates_baseline"] = False
    return generated


def apply_trusted_tail_release_onset_hypotheses(
    alignment_audio_path: Path,
    entries: list[LyricEntry],
    report: dict[str, object],
    evaluated: EvaluatedBackendTiming,
    duration: float,
) -> list[dict[str, object]]:
    """Recover a content-trusted line whose current onset invades a trusted tail.

    This producer is deliberately unavailable when the predecessor tail is
    invalid/unknown.  It therefore does not resurrect the historical
    "fractured predecessor retry" behavior.  A candidate is published only
    from a strong independent onset inside the already-proven tail release
    region; central ownership still decides whether that onset can be trusted.
    """
    tail_evidence_context = _tail_evidence_context_from_report(
        report, entries=entries
    )
    independent_tail_evidence = runtime_independent_tail_evidence(
        report, len(entries), tail_evidence_context
    )
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return []
    try:
        onset_features = analyze_vocal_onsets(alignment_audio_path, duration)
        audio_sha = _path_sha256(alignment_audio_path)
    except (LrcError, OSError) as exc:
        _record_backend_participation_failure(
            report, "apply_trusted_tail_release_onset_hypotheses", exc
        )
        return []
    generated: list[dict[str, object]] = []
    for index in range(1, len(entries)):
        assignment = assignments[index]
        if not isinstance(assignment, dict) or evaluated.state.audits[index].timing_trusted:
            continue
        if not assignment_content_is_trusted(assignment):
            continue
        previous_decision = evaluated.state.decisions[index - 1]
        previous_audit = evaluated.state.audits[index - 1]
        if not previous_audit.timing_trusted:
            continue
        previous_tail = resolve_reliable_tail(
            previous_decision.candidate,
            independent_tail_evidence[index - 1]
            if independent_tail_evidence is not None
            else None,
            tail_evidence_context=tail_evidence_context,
        )
        if previous_tail.validity != "valid" or previous_tail.reliable_tail_end is None:
            continue
        current = evaluated.state.decisions[index].candidate
        if not current.current:
            continue
        if current.written_time.seconds > previous_tail.reliable_tail_end:
            # This producer only repairs an actual tail invasion.  Release-zone
            # uncertainty without overlap is handled by phonetic-onset fusion.
            continue
        uncertainty = float(previous_tail.uncertainty_seconds or 0.0)
        search_start = previous_tail.reliable_tail_end
        search_end = min(duration, search_start + max(0.65, min(1.20, uncertainty + 0.30)))
        next_trusted = next(
            (
                evaluated.state.decisions[position].written_time.seconds - 0.20
                for position in range(index + 1, len(entries))
                if evaluated.state.audits[position].timing_trusted
            ),
            duration,
        )
        search_end = min(search_end, next_trusted)
        onset = _first_post_tail_cluster_onset(
            onset_features,
            search_start,
            search_end,
            min_strength=TAIL_RELEASE_ONSET_MIN_STRENGTH,
            min_prominence=TAIL_RELEASE_ONSET_MIN_PROMINENCE,
        )
        if onset is None:
            continue
        candidate_time = float(onset["time"])
        window = {
            "audio_sha256": audio_sha,
            "feature_revision": TAIL_RELEASE_FEATURE_REVISION,
            "previous_candidate_id": previous_decision.candidate.candidate_id,
            "previous_tail_revision": previous_tail.tail_evidence_revision,
            "previous_tail_end": round(previous_tail.reliable_tail_end, 6),
            "previous_tail_uncertainty": round(uncertainty, 6),
            "search_start": round(search_start, 6),
            "search_end": round(search_end, 6),
            "onset_strength": round(float(onset["strength"]), 8),
            "onset_prominence": round(float(onset["prominence"]), 8),
            "previous_onset_time": round(float(onset["previous_onset_time"]), 6),
            "cluster_gap": round(float(onset["cluster_gap"]), 6),
            "context_start": round(float(onset["context_start"]), 6),
            "old_candidate_raw_time": round(current.raw_time, 6),
            "old_candidate_centiseconds": current.written_time.centiseconds,
            "old_candidate_time": round(current.written_time.seconds, 6),
        }
        before = len(assignment.get("alignment_hypotheses", [])) if isinstance(assignment.get("alignment_hypotheses"), list) else 0
        hypothesis = record_alignment_hypothesis(
            assignment,
            source="trusted-tail-release-onset",
            stage="trusted-tail-release-onset-search",
            raw_time=candidate_time,
            confidence=min(1.0, 0.70 + float(onset["prominence"]) * 0.30),
            spans=[],
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=float(onset["strength"]),
            acoustic_onset_time=candidate_time,
            acoustic_evidence_producer="tail-release-acoustic-search",
            acoustic_evidence_kind="generic-acoustic",
            direct_onset_time=candidate_time,
            direct_onset_evidence_producer="tail-release-acoustic-search",
            direct_onset_evidence_kind="generic-onset",
            parent_source="central-trusted-predecessor-tail",
            parent_time=previous_tail.reliable_tail_end,
            window=window,
            mode="strong-onset-inside-centrally-trusted-tail-release",
        )
        after = len(assignment.get("alignment_hypotheses", [])) if isinstance(assignment.get("alignment_hypotheses"), list) else before
        if after > before:
            generated.append({
                "entry": index + 1,
                "candidate": round(candidate_time, 3),
                "previous_entry": index,
                "previous_tail_end": round(previous_tail.reliable_tail_end, 3),
                "hypothesis_id": hypothesis.get("hypothesis_id"),
            })
    report["trusted_tail_release_onset_hypotheses"] = generated
    report["trusted_tail_release_onset_count"] = len(generated)
    report["trusted_tail_release_onset_authoritative"] = False
    report["trusted_tail_release_onset_mutates_baseline"] = False
    return generated
