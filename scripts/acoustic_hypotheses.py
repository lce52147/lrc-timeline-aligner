"""Extracted from auto_lrc.py (split step 7): vocal onset, recurrence and phonetic fusion hypotheses, provenance validators, post-score refinement."""

from __future__ import annotations

import collections
import copy
import hashlib
import math
import statistics
import time
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from ctc_refinement import _invalidate_timing_trust_after_timestamp_mutation, ctc_path_fracture_evidence, first_local_onset, has_ctc_compressed_prefix
from report_trust import COLLAPSE_SEGMENT_SCORE, _record_backend_participation_failure, assignment_content_is_trusted, assignment_has_timing, assignment_score, update_report_confidence_metrics
from timing_core import EvaluatedBackendTiming, FinalTimingState, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS, TRUSTED_ALIGNMENT_SCORE, TimingCandidate, _candidate_confidence, _finite_unit_confidence, _serialized_delta_matches, _serialized_seconds_equal, _tail_evidence_context_from_report, analyze_candidate_regions, assignment_ctc_path_collapsed, ctc_collapsed_region_flags, make_timing_candidate, record_alignment_hypothesis, resolve_reliable_tail, runtime_independent_tail_evidence
from lyrics_io import AsrSegment, LyricEntry, _is_single_cjk_or_kana_term, entry_sung_text, first_forced_char_time, has_ambiguous_lyric_prefix, has_short_duplicate_context, lyric_anchor_profile, normalize_match_text, repeated_leading_term, sequence_ratio, snap_to_onsets, weighted_rough_times
from audio_features import AudioFeatures, HOP_SECONDS, _recurrence_segment, _search_recurrence_start, acoustic_pitchclass_recurrence_features, acoustic_recurrence_features, acoustic_recurrence_similarity, analyze_audio, analyze_vocal_onsets, nearest_local_peak
from runtime_base import LrcError, _path_sha256, _timing_content_digest, canonicalize_lrc_time
from runtime_base import TimingTokenSpan, _timing_spans_from_value


def project_fun_asr_exact_positive(
    lyrics: Sequence[str],
    outputs: Sequence[Mapping[str, object]],
    material_clusters: Sequence[Mapping[str, object]] = (),
    ownership_pairs: Sequence[Mapping[str, int]] = (),
) -> dict[str, object]:
    """Project frozen Fun-ASR native spans into exact-positive observations.

    This is post-freeze reporting only.  It accepts no timing candidates or
    reference values, never scores similarity, and never treats missing text
    as negative evidence.  CTC and LLM heads from one producer are retained as
    one producer; repeated outputs are explicitly non-independent.
    """
    # The frozen producer contract is a stream of native token spans.  A stream
    # may contain several ordered lyric lines, so line identity is discovered
    # by exact contiguous normalized subsequences rather than by treating the
    # whole output as one line.
    if not isinstance(lyrics, (list, tuple)) or any(not isinstance(item, str) for item in lyrics):
        raise LrcError("Fun-ASR lyrics must be an ordered string sequence")
    normalized_lyrics = [normalize_match_text(item) for item in lyrics]
    if any(not item for item in normalized_lyrics):
        raise LrcError("Fun-ASR lyric sequence contains empty normalized text")
    if not isinstance(outputs, (list, tuple)):
        raise LrcError("Fun-ASR outputs must be a sequence")

    parsed: list[dict[str, object]] = []
    ignored_nonlexical_spans: list[dict[str, object]] = []
    for output_index, output in enumerate(outputs):
        if not isinstance(output, Mapping):
            return {"status": "ERROR", "error": "malformed Fun-ASR output"}
        producer, head, spans = output.get("producer"), output.get("head"), output.get("spans")
        if not isinstance(producer, str) or not producer or not isinstance(head, str) or not head or not isinstance(spans, (list, tuple)):
            return {"status": "ERROR", "error": "malformed Fun-ASR span contract"}
        normalized_spans: list[dict[str, object]] = []
        previous_end = -math.inf
        for span_index, span in enumerate(spans):
            if not isinstance(span, Mapping) or not isinstance(span.get("token"), str):
                return {"status": "ERROR", "error": "malformed Fun-ASR native span"}
            try:
                start, end = float(span.get("start_time")), float(span.get("end_time"))
            except (TypeError, ValueError):
                return {"status": "ERROR", "error": "malformed Fun-ASR native span"}
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < 0 or end < start or start < previous_end:
                return {"status": "ERROR", "error": "non-monotonic or overlapping Fun-ASR native spans"}
            if span["token"] == "":
                return {"status": "ERROR", "error": "empty Fun-ASR native span token"}
            token = normalize_match_text(span["token"])
            if not token:
                ignored_nonlexical_spans.append({"producer": producer, "head": head, "output_index": output_index, "span_index": span_index, "raw_token": span["token"], "start_time": start, "end_time": end})
                previous_end = end
                continue
            normalized_spans.append({"token": token, "start_time": start, "end_time": end})
            previous_end = end
        if not normalized_spans:
            parsed.append({"producer": producer, "head": head, "spans": tuple(), "hits": tuple()})
            continue
        stream = "".join(str(span["token"]) for span in normalized_spans)
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for span in normalized_spans:
            end = cursor + len(str(span["token"]))
            offsets.append((cursor, end))
            cursor = end
        hits: list[dict[str, object]] = []
        for lyric_index, target in enumerate(normalized_lyrics):
            search_at = 0
            while True:
                found = stream.find(target, search_at)
                if found < 0:
                    break
                finish = found + len(target)
                covered = tuple(index for index, (left, right) in enumerate(offsets) if left < finish and right > found)
                if covered and found == offsets[covered[0]][0] and finish == offsets[covered[-1]][1]:
                    hits.append({"lyric_index": lyric_index, "span_indexes": covered})
                search_at = found + 1
        parsed.append({"producer": producer, "head": head, "spans": tuple(normalized_spans), "hits": tuple(hits)})

    if not parsed:
        return {"status": "ABSENT", "positive_sequences": (), "material_contradictions": (), "ownership": (), "repeat_non_independent": False, "head_disagreement": False, "ignored_nonlexical_spans": tuple(ignored_nonlexical_spans)}

    unique_hits: list[dict[str, object]] = []
    for item in parsed:
        hits = item["hits"]
        for hit in hits:
            lyric_index = int(hit["lyric_index"])
            if sum(normalized_lyrics[lyric_index] == candidate for candidate in normalized_lyrics) == 1:
                matched_spans = tuple(item["spans"][index] for index in hit["span_indexes"])
                unique_hits.append({"producer": item["producer"], "head": item["head"], "lyric_index": lyric_index, "span_indexes": hit["span_indexes"], "spans": matched_spans})

    fact_groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for hit in unique_hits:
        key = (str(hit["producer"]), int(hit["lyric_index"]))
        fact_groups.setdefault(key, []).append(hit)
    positive_facts: list[dict[str, object]] = []
    for facts in fact_groups.values():
        deduped = {}
        for fact in facts:
            signature = (str(fact["producer"]), int(fact["lyric_index"]), tuple((str(span["token"]), float(span["start_time"]), float(span["end_time"])) for span in fact["spans"]))
            deduped.setdefault(signature, []).append(fact)
        if len(deduped) == 1:
            signature_facts = next(iter(deduped.values()))
            canonical = signature_facts[0]
            positive_facts.append({"producer": canonical["producer"], "lyric_index": canonical["lyric_index"], "span_indexes": canonical["span_indexes"], "spans": canonical["spans"], "heads": tuple(sorted({str(fact["head"]) for fact in signature_facts}))})

    material: list[dict[str, object]] = []
    head_disagreement = False
    requested_unresolved = False
    for cluster in material_clusters:
        if not isinstance(cluster, Mapping) or not isinstance(cluster.get("cluster_id"), str) or not cluster.get("cluster_id") or type(cluster.get("target_index")) is not int or not 0 <= int(cluster["target_index"]) < len(normalized_lyrics):
            return {"status": "ERROR", "error": "material cluster requires cluster_id and target_index"}
        try:
            anchor = float(cluster.get("anchor_time"))
        except (TypeError, ValueError):
            return {"status": "ERROR", "error": "material cluster anchor_time invalid"}
        if not math.isfinite(anchor) or anchor < 0:
            return {"status": "ERROR", "error": "material cluster anchor_time non-finite"}
        at_anchor: list[dict[str, object]] = []
        for hit in positive_facts:
            spans = hit["spans"]
            if any(float(span["start_time"]) <= anchor <= float(span["end_time"]) for span in spans):
                at_anchor.append(hit)
        identities = {(int(hit["lyric_index"]), str(hit["producer"])) for hit in at_anchor}
        if len({head for hit in at_anchor for head in hit["heads"]}) > 1 and len({int(hit["lyric_index"]) for hit in at_anchor}) > 1:
            head_disagreement = True
        non_target = [hit for hit in at_anchor if int(hit["lyric_index"]) != int(cluster["target_index"])]
        if len(identities) == 1 and len(non_target) == 1:
            material.append({"cluster_id": cluster["cluster_id"], "status": "POSITIVE", "lyric_index": non_target[0]["lyric_index"], "heads": non_target[0]["heads"], "spans": non_target[0]["spans"]})
        else:
            material.append({"cluster_id": cluster["cluster_id"], "status": "UNRESOLVED"})
            requested_unresolved = True

    ownership: list[dict[str, object]] = []
    for pair in ownership_pairs:
        if not isinstance(pair, Mapping) or type(pair.get("predecessor_index")) is not int or type(pair.get("target_index")) is not int:
            return {"status": "ERROR", "error": "malformed ownership pair contract"}
        predecessor_index, target_index = int(pair["predecessor_index"]), int(pair["target_index"])
        if not 0 <= predecessor_index < len(normalized_lyrics) or not 0 <= target_index < len(normalized_lyrics) or target_index != predecessor_index + 1:
            ownership.append({"predecessor_index": predecessor_index, "target_index": target_index, "status": "UNRESOLVED"})
            requested_unresolved = True
            continue
        pred = [hit for hit in positive_facts if int(hit["lyric_index"]) == predecessor_index]
        target = [hit for hit in positive_facts if int(hit["lyric_index"]) == target_index]
        if len(pred) != 1 or len(target) != 1 or str(pred[0]["producer"]) != str(target[0]["producer"]):
            ownership.append({"predecessor_index": predecessor_index, "target_index": target_index, "status": "UNRESOLVED"})
            requested_unresolved = True
            continue
        pred_span = pred[0]["spans"][-1]
        target_span = target[0]["spans"][0]
        status = "POSITIVE" if float(pred_span["end_time"]) <= float(target_span["start_time"]) else "UNRESOLVED"
        pair_heads = set(pred[0]["heads"]) | set(target[0]["heads"])
        for conflicting in positive_facts:
            if int(conflicting["lyric_index"]) in {predecessor_index, target_index} or str(conflicting["producer"]) != str(pred[0]["producer"]):
                continue
            conflict_spans = conflicting["spans"]
            overlap = any(
                max(float(left["start_time"]), float(right["start_time"]))
                < min(float(left["end_time"]), float(right["end_time"]))
                for left in (pred_span, target_span)
                for right in conflict_spans
            )
            if overlap:
                status = "UNRESOLVED"
                requested_unresolved = True
                if not set(conflicting["heads"]).issubset(pair_heads):
                    head_disagreement = True
        ownership.append({"predecessor_index": predecessor_index, "target_index": target_index, "status": status, **({"producer": pred[0]["producer"], "heads": tuple(sorted(pair_heads)), "predecessor_span": pred_span, "target_span": target_span} if status == "POSITIVE" else {})})
        requested_unresolved = requested_unresolved or status != "POSITIVE"

    producers = {str(item["producer"]) for item in parsed}
    top_status = "ABSENT" if not positive_facts and all(not item["spans"] for item in parsed) else ("RETRIEVED" if positive_facts and not head_disagreement and not requested_unresolved else "UNRESOLVED")
    return {"status": top_status, "positive_sequences": tuple(positive_facts), "material_contradictions": tuple(material), "ownership": tuple(ownership), "repeat_non_independent": len(parsed) > len(producers), "head_disagreement": head_disagreement, "ignored_nonlexical_spans": tuple(ignored_nonlexical_spans)}


RECURRENCE_FEATURE_REVISION = "logmel-dct-delta-v1"


RECURRENCE_MIN_SIMILARITY = 0.16


RECURRENCE_MIN_DISTINCT_MARGIN = 0.08


VOCAL_ONSET_CONFIRM_RADIUS_SECONDS = 0.18


VOCAL_ONSET_CONFIRM_MIN_STRENGTH = 0.82


VOCAL_ONSET_CONFIRM_MIN_PROMINENCE = 0.35


VOCAL_ACTIVITY_REENTRY_FEATURE_REVISION = "isolated-vocal-activity-reentry-v1"


# Re-entry evidence deliberately reuses analyze_audio()/merge_segments()'s
# conservative 450 ms split rule. A shorter dip is not enough to defeat
# UNKNOWN/VIOLATION ownership; continuous singing therefore remains unverified.
VOCAL_ACTIVITY_REENTRY_MIN_GAP_SECONDS = 0.45


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


def _candidate_spans_from_assignment(assignment: dict[str, object]) -> tuple[TimingTokenSpan, ...]:
    raw_spans = assignment.get("ctc_token_spans")
    if not isinstance(raw_spans, list):
        return ()
    spans: list[TimingTokenSpan] = []
    for raw in raw_spans:
        if not isinstance(raw, dict):
            continue
        try:
            start = float(raw.get("start"))
            end = float(raw.get("end"))
            score = _finite_unit_confidence(raw.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(start) and math.isfinite(end) and end >= start):
            continue
        spans.append(TimingTokenSpan(start, end, score, str(raw.get("char") or raw.get("token") or "")))
    return tuple(spans)


def _nearest_local_maximum(
    features: AudioFeatures,
    center: float,
    radius: float,
) -> tuple[float, float, float] | None:
    if features.onset_strength.size == 0 or features.frame_times.size == 0:
        return None
    mask = (features.frame_times >= center - radius) & (features.frame_times <= center + radius)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    best_index = int(indices[np.argmax(features.onset_strength[indices])])
    strength = float(features.onset_strength[best_index])
    context = np.flatnonzero(
        (features.frame_times >= center - 0.80) & (features.frame_times <= center + 0.80)
    )
    baseline = float(np.median(features.onset_strength[context])) if context.size else 0.0
    return float(features.frame_times[best_index]), strength, strength - baseline


def _strong_vocal_onset_near(
    features: AudioFeatures | None,
    center: float,
    *,
    require_absolute_vocal: bool = False,
) -> tuple[float, float, float] | None:
    """Return the existing production-grade vocal onset observation near a time."""
    if features is None or not math.isfinite(center):
        return None
    peak = _nearest_local_maximum(features, center, VOCAL_ONSET_CONFIRM_RADIUS_SECONDS)
    if peak is None:
        return None
    peak_time, strength, prominence = peak
    if require_absolute_vocal:
        # Vocal-only data: 0.4s power window at the detected onset, relative
        # to the high-energy active-vocal distribution (not full-mix music).
        # RMS values are absolute dBFS, unlike normalized onset strength.
        raw_levels = np.asarray(features.rms_db, dtype=np.float64)
        frame_times = np.asarray(features.frame_times, dtype=np.float64)
        usable = raw_levels[np.isfinite(raw_levels)]
        if usable.size < 8:
            return None
        active_floor = float(np.percentile(usable, 55))
        vocal_active = usable[usable >= active_floor]
        if vocal_active.size < 8:
            return None
        active_median = float(np.median(vocal_active))
        mask = (frame_times >= peak_time) & (frame_times < peak_time + 0.4)
        if not np.any(mask):
            return None
        local = np.asarray(raw_levels[mask], dtype=np.float64)
        local = local[np.isfinite(local)]
        if local.size < 3:
            return None
        local_power_db = 10.0 * math.log10(
            max(float(np.mean(np.power(10.0, local / 10.0))), 1e-20)
        )
        if local_power_db < active_median - 20.0:
            return None
    if (
        strength < VOCAL_ONSET_CONFIRM_MIN_STRENGTH
        or prominence < VOCAL_ONSET_CONFIRM_MIN_PROMINENCE
    ):
        return None
    return peak_time, strength, prominence


def _vocal_activity_reentry_for_time(
    features: AudioFeatures,
    current_time: float,
) -> dict[str, float | int] | None:
    """Return a conservative isolated-vocal activity re-entry near a boundary.

    ``analyze_audio`` merges activity separated by <=450 ms, so a surviving
    segment boundary is already evidence of a substantial non-vocal interval.
    The detector never moves a timestamp: it only certifies the existing CTC
    boundary when the isolated vocal stem independently re-enters there.
    """
    if not features.segments:
        return None
    matches: list[dict[str, float | int]] = []
    for segment_index, (segment_start, segment_end) in enumerate(features.segments):
        if abs(float(segment_start) - float(current_time)) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS:
            continue
        previous_end = (
            float(features.segments[segment_index - 1][1])
            if segment_index > 0
            else 0.0
        )
        gap = float(segment_start) - previous_end
        if gap + HOP_SECONDS * 0.51 < VOCAL_ACTIVITY_REENTRY_MIN_GAP_SECONDS:
            continue
        matches.append({
            "segment_index": segment_index,
            "segment_start": float(segment_start),
            "segment_end": float(segment_end),
            "previous_segment_end": previous_end,
            "gap_seconds": gap,
            "distance_seconds": abs(float(segment_start) - float(current_time)),
        })
    if not matches:
        return None
    return min(
        matches,
        key=lambda row: (
            float(row["distance_seconds"]),
            -float(row["gap_seconds"]),
            int(row["segment_index"]),
        ),
    )


VOCAL_ONSET_RAW_ORDER_TOLERANCE_SECONDS = 1.0


VOCAL_ONSET_RAW_ORDER_MIN_CONTRADICTIONS = 2


VOCAL_ONSET_RAW_ORDER_MIN_SCORE = 0.50


def _raw_asr_order_anchors(
    report: dict[str, object],
    entries: list[LyricEntry],
) -> dict[int, float]:
    """Rows whose recorded raw-ASR line onset can witness lyric ORDER.

    Only real (non-borrowed) raw segments recorded by
    ``flag_unresolved_raw_ctc_disagreements``; drift runs are never recorded
    there, and lines with an ambiguous lyric prefix may match a sibling line.
    """
    anchors: dict[int, float] = {}
    rows = report.get("suspicious_alignments")
    if not isinstance(rows, list):
        return anchors
    for row in rows:
        if not isinstance(row, dict) or row.get("raw_asr_borrowed") is not False:
            continue
        entry = row.get("entry")
        stamps = row.get("candidate_timestamps")
        if isinstance(entry, bool) or not isinstance(entry, int) or not isinstance(stamps, dict):
            continue
        try:
            raw_time = float(stamps.get("raw_asr"))
            raw_score = float(row.get("raw_asr_score"))
        except (TypeError, ValueError):
            continue
        index = entry - 1
        if (
            not math.isfinite(raw_time)
            or raw_score < VOCAL_ONSET_RAW_ORDER_MIN_SCORE
            or not 0 <= index < len(entries)
            or has_ambiguous_lyric_prefix(entries, index)
        ):
            continue
        anchors[index] = raw_time
    return anchors


def _raw_asr_order_contradictions(
    anchors: dict[int, float],
    index: int,
    time_seconds: float,
) -> list[int]:
    """Entries whose raw onset says ``index`` cannot start at ``time_seconds``."""
    tolerance = VOCAL_ONSET_RAW_ORDER_TOLERANCE_SECONDS
    return sorted(
        other + 1
        for other, raw_time in anchors.items()
        if (other > index and raw_time < time_seconds - tolerance)
        or (other < index and raw_time > time_seconds + tolerance)
    )


def apply_vocal_onset_confirmation_hypotheses(
    alignment_audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
    *,
    features: AudioFeatures | None = None,
) -> list[dict[str, object]]:
    """Confirm an existing coherent CTC onset with independent vocal acoustics.

    This producer never moves a timestamp.  It only publishes a challenger for
    the exact current timestamp when the current full-line CTC path is already
    internally coherent and a strong, locally-prominent vocal onset is bound to
    the written boundary.  It therefore resolves ownership uncertainty without
    converting a weak or fractured CTC path into trusted timing.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return []
    if features is None:
        try:
            features = analyze_vocal_onsets(alignment_audio_path, duration)
        except LrcError as exc:
            _record_backend_participation_failure(
                report, "apply_vocal_onset_confirmation_hypotheses", exc
            )
            return []
    audio_revision = hashlib.sha256(Path(alignment_audio_path).read_bytes()).hexdigest()
    # A coherent line next to a collapsed path, or a clock that sits after
    # (before) other rows' raw-ASR line onsets, is a path that has lost its
    # place; a vocal onset there confirms *a* line start, not this line's.
    collapsed_region = ctc_collapsed_region_flags(
        [assignment_ctc_path_collapsed(item) for item in assignments]
    )
    raw_order_anchors = _raw_asr_order_anchors(report, entries)
    generated: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            continue
        spans = _candidate_spans_from_assignment(assignment)
        if len(spans) < 4:
            continue
        try:
            current_time = float(assignment.get("timestamp", timestamps[index]) or timestamps[index])
        except (TypeError, ValueError):
            continue
        if collapsed_region[index]:
            skipped.append({"entry": index + 1, "reason": "collapsed-ctc-region"})
            continue
        contradicting = _raw_asr_order_contradictions(raw_order_anchors, index, current_time)
        if len(contradicting) >= VOCAL_ONSET_RAW_ORDER_MIN_CONTRADICTIONS:
            skipped.append({
                "entry": index + 1,
                "reason": "raw-asr-order-contradiction",
                "contradicting_entries": contradicting,
            })
            continue
        current_candidate = make_timing_candidate(
            entry_index=index,
            entry_text=entry_sung_text(entries[index]),
            source="ctc-current-geometry-probe",
            raw_time=current_time,
            spans=spans,
            confidence=_candidate_confidence(assignment),
            identity_support="supported",
            sequence_support="supported",
            current=False,
            right_edge_pileup=bool(assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(assignment.get("ctc_right_edge_truncation")),
            source_artifact={"geometry_probe": True},
        )
        regions = analyze_candidate_regions(current_candidate)
        coherent_opening = bool(
            regions.temporal_coherence == "coherent"
            and regions.onset_proof_end is not None
        )
        direct_onset_resolvable_opening = bool(
            regions.temporal_coherence == "ambiguous"
            and not regions.severe_gap_indexes
            and spans[0].score >= 0.08
            and regions.reliable_support_count >= 8
            and regions.prefix_duration is not None
            and regions.prefix_duration >= 3.0
        )
        if not (coherent_opening or direct_onset_resolvable_opening):
            continue
        peak = _nearest_local_maximum(features, current_time, VOCAL_ONSET_CONFIRM_RADIUS_SECONDS)
        if peak is None:
            continue
        peak_time, strength, prominence = peak
        if strength < VOCAL_ONSET_CONFIRM_MIN_STRENGTH or prominence < VOCAL_ONSET_CONFIRM_MIN_PROMINENCE:
            continue
        evidence_window = {
            "start": round(current_time - VOCAL_ONSET_CONFIRM_RADIUS_SECONDS, 3),
            "end": round(current_time + VOCAL_ONSET_CONFIRM_RADIUS_SECONDS, 3),
            "audio_sha256": audio_revision,
            "audio_duration_seconds": round(float(duration), 6),
            "feature_revision": "vocal-consonant-onset-v1",
            "peak_strength": round(strength, 8),
            "peak_prominence": round(prominence, 8),
        }
        hypothesis = record_alignment_hypothesis(
            assignment,
            source="ctc-vocal-onset-confirmed",
            stage="independent-vocal-onset-confirmation",
            raw_time=current_time,
            confidence=max(_candidate_confidence(assignment), strength),
            spans=assignment.get("ctc_token_spans"),
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=strength,
            acoustic_onset_time=peak_time,
            acoustic_evidence_producer="consonant-onset-detector",
            acoustic_evidence_kind="vocal-onset",
            direct_onset_time=peak_time,
            direct_onset_evidence_producer="consonant-onset-detector",
            direct_onset_evidence_kind="vocal-onset",
            parent_source=str(report.get("backend") or "unknown") + "-current",
            parent_time=current_time,
            window=evidence_window,
            mode="coherent-ctc-plus-bound-vocal-onset",
            right_edge_pileup=bool(assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(assignment.get("ctc_right_edge_truncation")),
        )
        generated.append({
            "entry": index + 1,
            "timestamp": round(current_time, 3),
            "peak_time": round(peak_time, 3),
            "strength": round(strength, 6),
            "prominence": round(prominence, 6),
            "hypothesis_id": hypothesis.get("hypothesis_id"),
        })
    report["vocal_onset_confirmation_hypotheses"] = generated
    report["vocal_onset_confirmation_count"] = len(generated)
    report["vocal_onset_confirmation_skipped"] = skipped
    report["vocal_onset_confirmation_authoritative"] = False
    report["vocal_onset_confirmation_mutates_baseline"] = False
    return generated


def apply_vocal_activity_reentry_hypotheses(
    alignment_audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> list[dict[str, object]]:
    """Publish boundary-grade evidence only for a real vocal re-entry.

    This is intentionally stricter than ``ctc-vocal-onset-confirmed``. A
    consonant/energy peak can occur inside the predecessor's held syllable, so
    generic vocal onset remains non-boundary-grade. Here the isolated vocal
    activity itself must have stopped long enough to form a separate activity
    segment and then restart at the current CTC boundary. That negative
    evidence about the predecessor tail is independent of the current CTC
    candidate and may therefore resolve UNKNOWN ownership without weakening
    the central invariant.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return []
    try:
        activity = analyze_audio(alignment_audio_path, duration)
        audio_revision = _path_sha256(alignment_audio_path)
    except (LrcError, OSError) as exc:
        _record_backend_participation_failure(
            report, "apply_vocal_activity_reentry_hypotheses", exc
        )
        return []
    generated: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            continue
        spans = _candidate_spans_from_assignment(assignment)
        if len(spans) < 4:
            continue
        try:
            current_time = float(assignment.get("timestamp", timestamps[index]) or timestamps[index])
        except (TypeError, ValueError):
            continue
        probe = make_timing_candidate(
            entry_index=index,
            entry_text=entry_sung_text(entries[index]),
            source="ctc-current-reentry-geometry-probe",
            raw_time=current_time,
            spans=spans,
            confidence=_candidate_confidence(assignment),
            identity_support="supported",
            sequence_support="supported",
            current=False,
            right_edge_pileup=bool(assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(assignment.get("ctc_right_edge_truncation")),
            source_artifact={"reentry_geometry_probe": True},
        )
        regions = analyze_candidate_regions(probe)
        geometry_ok = bool(
            regions.temporal_coherence == "coherent"
            and not regions.severe_gap_indexes
            and regions.reliable_support_count >= 3
        )
        if not geometry_ok:
            continue
        reentry = _vocal_activity_reentry_for_time(activity, current_time)
        if reentry is None:
            continue
        segment_start = float(reentry["segment_start"])
        evidence_window = {
            "audio_sha256": audio_revision,
            "audio_duration_seconds": round(float(duration), 6),
            "feature_revision": VOCAL_ACTIVITY_REENTRY_FEATURE_REVISION,
            "segment_index": int(reentry["segment_index"]),
            "segment_start": round(segment_start, 6),
            "segment_end": round(float(reentry["segment_end"]), 6),
            "previous_segment_end": round(float(reentry["previous_segment_end"]), 6),
            "gap_seconds": round(float(reentry["gap_seconds"]), 6),
            "current_time": round(current_time, 6),
        }
        hypothesis = record_alignment_hypothesis(
            assignment,
            source="ctc-vocal-activity-reentry",
            stage="independent-vocal-activity-reentry",
            raw_time=current_time,
            confidence=_candidate_confidence(assignment),
            spans=assignment.get("ctc_token_spans"),
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="unavailable",
            direct_onset_support="supported",
            direct_onset_time=segment_start,
            direct_onset_evidence_producer="vocal-activity-detector",
            direct_onset_evidence_kind="vocal-activity-reentry",
            parent_source=str(report.get("backend") or "unknown") + "-current",
            parent_time=current_time,
            window=evidence_window,
            mode="coherent-ctc-plus-isolated-vocal-reentry",
            right_edge_pileup=bool(assignment.get("ctc_right_edge_pileup")),
            window_truncated=bool(assignment.get("ctc_right_edge_truncation")),
        )
        generated.append({
            "entry": index + 1,
            "timestamp": round(current_time, 3),
            "segment_start": round(segment_start, 3),
            "gap_seconds": round(float(reentry["gap_seconds"]), 3),
            "hypothesis_id": hypothesis.get("hypothesis_id"),
        })
    report["vocal_activity_reentry_hypotheses"] = generated
    report["vocal_activity_reentry_count"] = len(generated)
    report["vocal_activity_reentry_authoritative"] = False
    report["vocal_activity_reentry_mutates_baseline"] = False
    return generated


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


def vocal_onset_confirmation_provenance_is_valid(
    hypothesis: dict[str, object],
    assignment: dict[str, object],
    evidence_audio_path: Path,
    *,
    recomputed_features: AudioFeatures | None = None,
    evidence_audio_sha256: str | None = None,
) -> bool:
    """Recompute whether a CTC + consonant-onset hypothesis is independently bound.

    The serialized source label is never sufficient.  The row must bind to the
    actual assignment CTC spans and to the audio artifact used by the onset
    producer.  This prevents a candidate timestamp from manufacturing its own
    direct-onset certificate.
    """
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if (
        hypothesis.get("source") != "ctc-vocal-onset-confirmed"
        or hypothesis.get("stage") != "independent-vocal-onset-confirmation"
        or hypothesis.get("mode") != "coherent-ctc-plus-bound-vocal-onset"
        or hypothesis.get("authoritative") is not False
        or hypothesis.get("identity_support") != "supported"
        or hypothesis.get("sequence_support") != "supported"
        or hypothesis.get("acoustic_support") != "supported"
        or hypothesis.get("direct_onset_support") != "supported"
    ):
        return False
    window = hypothesis.get("window")
    parent = hypothesis.get("parent")
    if not isinstance(window, dict) or not isinstance(parent, dict):
        return False
    if window.get("feature_revision") != "vocal-consonant-onset-v1":
        return False
    spans = hypothesis.get("spans")
    assignment_spans = assignment.get("ctc_token_spans")
    if not isinstance(spans, list) or not spans or spans != assignment_spans:
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        parent_time = float(parent.get("raw_candidate_time"))
        onset_time = float(hypothesis.get("direct_onset_time"))
        acoustic_time = float(hypothesis.get("acoustic_onset_time"))
        strength = float(hypothesis.get("acoustic_strength"))
        window_strength = float(window.get("peak_strength"))
        prominence = float(window.get("peak_prominence"))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (raw_time, parent_time, onset_time, acoustic_time, strength, window_strength, prominence)):
        return False
    if (
        not _serialized_seconds_equal(raw_time, parent_time)
        or not _serialized_seconds_equal(onset_time, acoustic_time)
        or abs(strength - window_strength) > 1e-6
        or abs(onset_time - canonicalize_lrc_time(raw_time).seconds) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
        or strength < VOCAL_ONSET_CONFIRM_MIN_STRENGTH
        or prominence < VOCAL_ONSET_CONFIRM_MIN_PROMINENCE
    ):
        return False
    try:
        audio_sha = evidence_audio_sha256 or _path_sha256(evidence_audio_path)
        evidence_duration = float(window.get("audio_duration_seconds"))
    except (OSError, LrcError, TypeError, ValueError):
        return False
    if not math.isfinite(evidence_duration) or evidence_duration <= 0.0:
        return False
    if window.get("audio_sha256") != audio_sha:
        return False
    try:
        recomputed = recomputed_features or analyze_vocal_onsets(
            evidence_audio_path, evidence_duration
        )
        peak = _nearest_local_maximum(recomputed, raw_time, VOCAL_ONSET_CONFIRM_RADIUS_SECONDS)
    except (LrcError, OSError):
        return False
    if peak is None:
        return False
    recomputed_time, recomputed_strength, recomputed_prominence = peak
    return bool(
        abs(recomputed_time - onset_time) <= HOP_SECONDS * 0.51
        and abs(recomputed_strength - strength) <= 1e-5
        and abs(recomputed_prominence - prominence) <= 1e-5
        and recomputed_strength >= VOCAL_ONSET_CONFIRM_MIN_STRENGTH
        and recomputed_prominence >= VOCAL_ONSET_CONFIRM_MIN_PROMINENCE
    )


def ctc_opening_vocal_fusion_is_valid(
    hypothesis: dict[str, object],
    assignment: dict[str, object],
    evidence_audio_path: Path,
    *,
    vocal_provenance_validated: bool = False,
    recomputed_features: AudioFeatures | None = None,
    evidence_audio_sha256: str | None = None,
) -> bool:
    """Bind lyric-specific CTC opening geometry to independent vocal acoustics.

    A generic vocal peak is not boundary-grade by itself.  It becomes a
    phonetic boundary only when the immutable baseline CTC row independently
    supplies a coherent first-mora geometry at the same canonical start.  The
    fusion never accepts local-retry geometry, fractured openings, truncation,
    or right-edge pileup.
    """
    if not vocal_provenance_validated and not vocal_onset_confirmation_provenance_is_valid(
        hypothesis, assignment, evidence_audio_path,
        recomputed_features=recomputed_features,
        evidence_audio_sha256=evidence_audio_sha256,
    ):
        return False
    try:
        assignment_time = float(assignment.get("timestamp"))
        hypothesis_time = float(hypothesis.get("raw_candidate_time"))
    except (TypeError, ValueError):
        return False
    if (
        not math.isfinite(assignment_time)
        or not math.isfinite(hypothesis_time)
        or canonicalize_lrc_time(assignment_time).centiseconds
        != canonicalize_lrc_time(hypothesis_time).centiseconds
        or bool(assignment.get("ctc_right_edge_pileup"))
        or bool(assignment.get("ctc_right_edge_truncation"))
    ):
        return False
    spans = _candidate_spans_from_assignment(assignment)
    if not spans:
        return False
    probe = make_timing_candidate(
        entry_index=0,
        entry_text="ctc-opening-geometry",
        source="ctc-current-geometry-probe",
        raw_time=assignment_time,
        spans=spans,
        confidence=_candidate_confidence(assignment),
        identity_support="supported",
        sequence_support="supported",
        current=False,
        right_edge_pileup=False,
        window_truncated=False,
        source_artifact={"geometry_only": True},
    )
    regions = analyze_candidate_regions(probe)
    return bool(
        regions.temporal_coherence == "coherent"
        and regions.onset_proof_end is not None
        and regions.fracture_region == "none"
        and not regions.severe_gap_indexes
        and abs(spans[0].start - canonicalize_lrc_time(assignment_time).seconds)
        <= HOP_SECONDS
    )


def vocal_activity_reentry_provenance_is_valid(
    hypothesis: dict[str, object],
    assignment: dict[str, object],
    evidence_audio_path: Path,
) -> bool:
    """Recompute isolated-vocal re-entry evidence from the bound audio.

    Serialized segment metadata is descriptive only. The validator must obtain
    the same activity segmentation from the actual vocal artifact before the
    direct-onset claim can become boundary-grade.
    """
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if (
        hypothesis.get("source") != "ctc-vocal-activity-reentry"
        or hypothesis.get("stage") != "independent-vocal-activity-reentry"
        or hypothesis.get("mode") != "coherent-ctc-plus-isolated-vocal-reentry"
        or hypothesis.get("authoritative") is not False
        or hypothesis.get("identity_support") != "supported"
        or hypothesis.get("sequence_support") != "supported"
        or hypothesis.get("direct_onset_support") != "supported"
    ):
        return False
    window = hypothesis.get("window")
    parent = hypothesis.get("parent")
    if not isinstance(window, dict) or not isinstance(parent, dict):
        return False
    if window.get("feature_revision") != VOCAL_ACTIVITY_REENTRY_FEATURE_REVISION:
        return False
    spans = hypothesis.get("spans")
    assignment_spans = assignment.get("ctc_token_spans")
    if not isinstance(spans, list) or not spans or spans != assignment_spans:
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        parent_time = float(parent.get("raw_candidate_time"))
        onset_time = float(hypothesis.get("direct_onset_time"))
        current_time = float(window.get("current_time"))
        segment_index = int(window.get("segment_index"))
        segment_start = float(window.get("segment_start"))
        segment_end = float(window.get("segment_end"))
        previous_segment_end = float(window.get("previous_segment_end"))
        gap_seconds = float(window.get("gap_seconds"))
        evidence_duration = float(window.get("audio_duration_seconds"))
    except (TypeError, ValueError):
        return False
    numeric = (
        raw_time, parent_time, onset_time, current_time, segment_start,
        segment_end, previous_segment_end, gap_seconds, evidence_duration,
    )
    if not all(math.isfinite(value) for value in numeric):
        return False
    if (
        not _serialized_seconds_equal(raw_time, parent_time)
        or not _serialized_seconds_equal(raw_time, current_time)
        or not _serialized_seconds_equal(onset_time, segment_start)
        or abs(segment_start - current_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
        or segment_end <= segment_start
        or gap_seconds + HOP_SECONDS * 0.51 < VOCAL_ACTIVITY_REENTRY_MIN_GAP_SECONDS
        or not _serialized_delta_matches(
            segment_start, previous_segment_end, gap_seconds, tolerance_microseconds=1
        )
        or evidence_duration <= 0.0
    ):
        return False
    try:
        if window.get("audio_sha256") != _path_sha256(evidence_audio_path):
            return False
        recomputed = analyze_audio(evidence_audio_path, evidence_duration)
        reentry = _vocal_activity_reentry_for_time(recomputed, raw_time)
    except (LrcError, OSError):
        return False
    if reentry is None:
        return False
    return bool(
        int(reentry["segment_index"]) == segment_index
        and abs(float(reentry["segment_start"]) - segment_start) <= HOP_SECONDS * 0.51
        and abs(float(reentry["segment_end"]) - segment_end) <= HOP_SECONDS * 0.51
        and abs(float(reentry["previous_segment_end"]) - previous_segment_end) <= HOP_SECONDS * 0.51
        and abs(float(reentry["gap_seconds"]) - gap_seconds) <= HOP_SECONDS * 0.51
    )


def phonetic_onset_fusion_provenance_is_valid(
    hypothesis: dict[str, object],
    assignment: dict[str, object],
    alignment_audio_path: Path,
) -> bool:
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if (
        hypothesis.get("source") != "ctc-phonetic-onset-fusion"
        or hypothesis.get("stage") != "independent-phonetic-onset-fusion"
        or hypothesis.get("mode") != "coherent-ctc-first-token-plus-independent-onset"
        or hypothesis.get("authoritative") is not False
    ):
        return False
    if (
        not assignment_content_is_trusted(assignment)
        or bool(assignment.get("ctc_right_edge_pileup"))
        or bool(assignment.get("ctc_right_edge_truncation"))
        or bool(ctc_path_fracture_evidence(assignment).get("fractured"))
    ):
        return False
    parent = hypothesis.get("parent")
    if (
        not isinstance(parent, dict)
        or parent.get("source") not in {"ctc-current", "ctc-vocal-onset-confirmed"}
    ):
        return False
    window = hypothesis.get("window")
    if not isinstance(window, dict) or window.get("feature_revision") != PHONETIC_ONSET_FEATURE_REVISION:
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        peak_time = float(window.get("ctc_peak_time"))
        peak_score = float(window.get("ctc_peak_score"))
        onset_time = float(window.get("onset_time"))
        onset_strength = float(window.get("onset_strength"))
        onset_prominence = float(window.get("onset_prominence"))
    except (TypeError, ValueError):
        return False
    try:
        current_assignment_raw = float(assignment.get("timestamp"))
        current_raw_time = float(window.get("current_raw_time"))
        current_written_centiseconds = int(window.get("current_written_centiseconds"))
        parent_time = float(parent.get("raw_candidate_time"))
        window_current_time = float(window.get("current_time"))
        top_acoustic_strength = float(hypothesis.get("acoustic_strength"))
        top_acoustic_time = float(hypothesis.get("acoustic_onset_time"))
        top_direct_time = float(hypothesis.get("direct_onset_time"))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (
        raw_time, peak_time, peak_score, onset_time, onset_strength, onset_prominence,
        current_assignment_raw, current_raw_time, parent_time, window_current_time,
        top_acoustic_strength, top_acoustic_time, top_direct_time,
    )):
        return False
    current_written = canonicalize_lrc_time(current_assignment_raw)
    if (
        not _serialized_seconds_equal(raw_time, peak_time)
        or peak_score < PHONETIC_ONSET_MIN_POSTERIOR
        or abs(peak_time - current_written.seconds) > 0.50
        or not _serialized_seconds_equal(current_raw_time, current_assignment_raw)
        or current_written_centiseconds != current_written.centiseconds
        or canonicalize_lrc_time(current_raw_time).centiseconds != current_written_centiseconds
        or not _serialized_seconds_equal(parent_time, current_raw_time)
        or not _serialized_seconds_equal(window_current_time, current_written.seconds)
        or not isinstance(window.get("current_candidate_id"), str)
        or not window.get("current_candidate_id")
    ):
        return False
    if (
        hypothesis.get("identity_support") != "supported"
        or hypothesis.get("sequence_support") != "supported"
        or hypothesis.get("acoustic_support") != "supported"
        or hypothesis.get("direct_onset_support") != "supported"
        or abs(top_acoustic_strength - onset_strength) > 1e-6
        or not _serialized_seconds_equal(top_acoustic_time, onset_time)
        or not _serialized_seconds_equal(top_direct_time, onset_time)
    ):
        return False
    if abs(onset_time - raw_time) > PREFIX_ACOUSTIC_ASSOCIATION_SECONDS:
        return False
    if onset_strength < PHONETIC_ONSET_MIN_STRENGTH or onset_prominence < PHONETIC_ONSET_MIN_PROMINENCE:
        return False
    peaks = assignment.get("ctc_first_token_candidates")
    if not isinstance(peaks, list) or not any(
        isinstance(peak, dict)
        and _timing_content_digest(peak) == window.get("ctc_peak_digest")
        and _serialized_seconds_equal(peak.get("time", -9999.0), peak_time)
        for peak in peaks
    ):
        return False
    spans = _candidate_spans_from_assignment(assignment)
    if not spans:
        return False
    current = make_timing_candidate(
        entry_index=0,
        entry_text="provenance-check",
        source="ctc-current",
        raw_time=current_assignment_raw,
        spans=spans,
        confidence=_candidate_confidence(assignment),
        identity_support="supported",
        sequence_support="supported",
        current=True,
    )
    if analyze_candidate_regions(current).temporal_coherence != "coherent":
        return False
    try:
        evidence_duration = float(window.get("audio_duration_seconds"))
        if not math.isfinite(evidence_duration) or evidence_duration <= 0.0:
            return False
        if window.get("audio_sha256") != _path_sha256(alignment_audio_path):
            return False
        recomputed = analyze_vocal_onsets(alignment_audio_path, evidence_duration)
        recomputed_peak = _nearest_local_maximum(
            recomputed, peak_time, PHONETIC_ONSET_RADIUS_SECONDS
        )
    except (OSError, LrcError, TypeError, ValueError):
        return False
    if recomputed_peak is None:
        return False
    recomputed_time, recomputed_strength, recomputed_prominence = recomputed_peak
    return bool(
        abs(recomputed_time - onset_time) <= HOP_SECONDS * 0.51
        and abs(recomputed_strength - onset_strength) <= 1e-5
        and abs(recomputed_prominence - onset_prominence) <= 1e-5
        and recomputed_strength >= PHONETIC_ONSET_MIN_STRENGTH
        and recomputed_prominence >= PHONETIC_ONSET_MIN_PROMINENCE
    )


def mutual_recurrence_hypothesis_provenance_is_valid(
    hypothesis: dict[str, object],
    entry: LyricEntry,
    alignment_audio_path: Path,
    assignments: list[object],
    entries: list[LyricEntry],
    entry_index: int,
) -> bool:
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if (
        hypothesis.get("source") != "mutual-acoustic-recurrence"
        or hypothesis.get("stage") != "occurrence-level-mutual-recurrence-search"
        or hypothesis.get("mode") != "reciprocal-exact-duplicate-pitchclass-recurrence"
        or hypothesis.get("authoritative") is not False
    ):
        return False
    window = hypothesis.get("window")
    parent = hypothesis.get("parent")
    if not isinstance(window, dict) or not isinstance(parent, dict):
        return False
    if parent.get("source") != "reciprocal-untrusted-duplicate-pair":
        return False
    if window.get("feature_revision") != MUTUAL_RECURRENCE_FEATURE_REVISION:
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        target_entry = int(window.get("target_entry"))
        peer_entry = int(window.get("peer_entry"))
        peer_candidate = float(window.get("peer_candidate"))
        score = float(window.get("similarity"))
        margin = float(window.get("distinct_margin"))
        onset_strength = float(window.get("onset_strength"))
        onset_prominence = float(window.get("onset_prominence"))
    except (TypeError, ValueError):
        return False
    if target_entry != entry_index + 1 or not (1 <= peer_entry <= len(assignments)) or peer_entry == target_entry:
        return False
    if score < MUTUAL_RECURRENCE_MIN_SIMILARITY or margin < MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN:
        return False
    if onset_strength < 0.45 or onset_prominence < 0.10:
        return False
    expected_lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entry)).encode("utf-8")
    ).hexdigest()
    if window.get("normalized_lyric_sha256") != expected_lyric_sha:
        return False
    peer_index = peer_entry - 1
    if len(entries) != len(assignments):
        return False
    peer_normalized = normalize_match_text(entry_sung_text(entries[peer_index]))
    if not peer_normalized or peer_normalized != normalize_match_text(entry_sung_text(entry)):
        return False
    if hashlib.sha256(peer_normalized.encode("utf-8")).hexdigest() != expected_lyric_sha:
        return False
    current_assignment = assignments[entry_index]
    peer_assignment = assignments[peer_index]
    if not isinstance(current_assignment, dict) or not isinstance(peer_assignment, dict):
        return False
    current_identity = _bound_local_occurrence_identity(current_assignment)
    peer_identity = _bound_local_occurrence_identity(peer_assignment)
    if current_identity is None or peer_identity is None:
        return False
    left_entry = int(window.get("left_entry", -1))
    right_entry = int(window.get("right_entry", -1))
    if {target_entry, peer_entry} != {left_entry, right_entry} or left_entry >= right_entry:
        return False
    expected_current_identity = (
        window.get("left_identity_evidence") if target_entry == left_entry
        else window.get("right_identity_evidence")
    )
    expected_peer_identity = (
        window.get("right_identity_evidence") if target_entry == left_entry
        else window.get("left_identity_evidence")
    )
    if current_identity != expected_current_identity or peer_identity != expected_peer_identity:
        return False
    try:
        expected_raw = float(window.get("left_candidate") if target_entry == left_entry else window.get("right_candidate"))
        expected_peer_candidate = float(window.get("right_candidate") if target_entry == left_entry else window.get("left_candidate"))
    except (TypeError, ValueError):
        return False
    if (
        not _serialized_seconds_equal(raw_time, expected_raw)
        or not _serialized_seconds_equal(peer_candidate, expected_peer_candidate)
    ):
        return False
    peer_hypotheses = peer_assignment.get("alignment_hypotheses")
    pair_id = window.get("pair_id")
    reciprocal = next(
        (
            row for row in peer_hypotheses
            if isinstance(row, dict)
            and row.get("source") == "mutual-acoustic-recurrence"
            and isinstance(row.get("window"), dict)
            and row["window"].get("pair_id") == pair_id
            and int(row["window"].get("peer_entry", -1)) == target_entry
        ),
        None,
    ) if isinstance(peer_hypotheses, list) else None
    if not isinstance(reciprocal, dict):
        return False
    reciprocal_window = reciprocal.get("window")
    reciprocal_parent = reciprocal.get("parent")
    if not isinstance(reciprocal_window, dict) or not isinstance(reciprocal_parent, dict):
        return False
    if (
        reciprocal.get("stage") != "occurrence-level-mutual-recurrence-search"
        or reciprocal.get("mode") != "reciprocal-exact-duplicate-pitchclass-recurrence"
        or reciprocal.get("authoritative") is not False
        or reciprocal_parent.get("source") != "reciprocal-untrusted-duplicate-pair"
        or int(reciprocal_window.get("target_entry", -1)) != peer_entry
        or int(reciprocal_window.get("peer_entry", -1)) != target_entry
    ):
        return False
    reciprocal_projection = {
        key: reciprocal_window.get(key)
        for key in (
            "audio_sha256", "feature_revision", "normalized_lyric_sha256",
            "left_entry", "right_entry", "left_seed", "right_seed",
            "left_identity_evidence", "right_identity_evidence",
            "left_bounds", "right_bounds", "left_candidate", "right_candidate",
            "template_duration", "similarity", "runner_similarity", "distinct_margin",
        )
    }
    if _timing_content_digest(reciprocal_projection) != pair_id:
        return False
    try:
        reciprocal_time = float(reciprocal.get("raw_candidate_time"))
        reciprocal_peer = float(reciprocal_window.get("peer_candidate"))
    except (TypeError, ValueError):
        return False
    if (
        not _serialized_seconds_equal(reciprocal_time, peer_candidate)
        or not _serialized_seconds_equal(reciprocal_peer, raw_time)
    ):
        return False
    # The reciprocal row must bind to an actual exact duplicate LyricEntry,
    # not merely repeat the same hash inside mutable report provenance.
    if reciprocal_window.get("normalized_lyric_sha256") != expected_lyric_sha:
        return False
    try:
        if window.get("audio_sha256") != _path_sha256(alignment_audio_path):
            return False
    except (OSError, LrcError):
        return False
    # Recompute the pair-id projection so arbitrary field edits fail closed.
    pair_projection = {
        key: window.get(key)
        for key in (
            "audio_sha256", "feature_revision", "normalized_lyric_sha256",
            "left_entry", "right_entry", "left_seed", "right_seed",
            "left_identity_evidence", "right_identity_evidence",
            "left_bounds", "right_bounds", "left_candidate", "right_candidate",
            "template_duration", "similarity", "runner_similarity", "distinct_margin",
        )
    }
    if _timing_content_digest(pair_projection) != pair_id:
        return False
    return True


def trusted_tail_release_provenance_is_valid(
    hypothesis: dict[str, object],
    alignment_audio_path: Path,
    assignments: list[object],
    entry_index: int,
) -> bool:
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if (
        hypothesis.get("source") != "trusted-tail-release-onset"
        or hypothesis.get("stage") != "trusted-tail-release-onset-search"
        or hypothesis.get("mode") != "strong-onset-inside-centrally-trusted-tail-release"
        or hypothesis.get("authoritative") is not False
    ):
        return False
    window = hypothesis.get("window")
    parent = hypothesis.get("parent")
    if not isinstance(window, dict) or not isinstance(parent, dict):
        return False
    if parent.get("source") != "central-trusted-predecessor-tail":
        return False
    if window.get("feature_revision") != TAIL_RELEASE_FEATURE_REVISION:
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        start = float(window.get("search_start"))
        end = float(window.get("search_end"))
        strength = float(window.get("onset_strength"))
        prominence = float(window.get("onset_prominence"))
        tail_end = float(window.get("previous_tail_end"))
        old_raw_time = float(window.get("old_candidate_raw_time"))
        old_centiseconds = int(window.get("old_candidate_centiseconds"))
        old_time = float(window.get("old_candidate_time"))
        previous_onset_time = float(window.get("previous_onset_time"))
        cluster_gap = float(window.get("cluster_gap"))
        context_start = float(window.get("context_start"))
    except (TypeError, ValueError):
        return False
    if not (start <= raw_time <= end and _serialized_seconds_equal(start, tail_end)):
        return False
    if old_time > tail_end:
        return False
    if previous_onset_time >= raw_time:
        return False
    if not _serialized_delta_matches(
        raw_time, previous_onset_time, cluster_gap, tolerance_microseconds=1
    ):
        return False
    if cluster_gap < TAIL_RELEASE_CLUSTER_BREAK_SECONDS:
        return False
    if context_start > tail_end or tail_end - context_start > TAIL_RELEASE_CONTEXT_SECONDS + 1e-6:
        return False
    if strength < TAIL_RELEASE_ONSET_MIN_STRENGTH or prominence < TAIL_RELEASE_ONSET_MIN_PROMINENCE:
        return False
    if not isinstance(window.get("previous_candidate_id"), str) or not isinstance(window.get("previous_tail_revision"), str):
        return False
    if entry_index <= 0 or entry_index >= len(assignments):
        return False
    current_assignment = assignments[entry_index]
    previous_assignment = assignments[entry_index - 1]
    if not isinstance(current_assignment, dict) or not isinstance(previous_assignment, dict):
        return False
    try:
        current_assignment_raw = float(current_assignment.get("timestamp"))
    except (TypeError, ValueError):
        return False
    current_written = canonicalize_lrc_time(current_assignment_raw)
    if (
        not _serialized_seconds_equal(current_assignment_raw, old_raw_time)
        or old_centiseconds != current_written.centiseconds
        or canonicalize_lrc_time(old_raw_time).centiseconds != old_centiseconds
        or not _serialized_seconds_equal(current_written.seconds, old_time)
    ):
        return False
    try:
        return window.get("audio_sha256") == _path_sha256(alignment_audio_path)
    except (OSError, LrcError):
        return False


def _trusted_recurrence_reference_geometry(
    assignment: dict[str, object],
    reference: TimingCandidate,
) -> tuple[float, tuple[TimingTokenSpan, ...], dict[str, object]] | None:
    """Return evidence-first full-line geometry for a trusted repeat reference.

    The preferred geometry is the selected central candidate itself.  A selected
    Raw/WhisperX fusion candidate may legitimately carry no token spans, however.
    In that case a self-bound local CTC identity row may provide the *template
    geometry* only when its independently observed start already agrees with the
    trusted selected boundary within the existing cross-backend consensus
    tolerance.  The CTC row is never shifted to the final timestamp, so this does
    not back-fill evidence from the answer.
    """
    if reference.spans:
        return (
            reference.written_time.seconds,
            reference.spans,
            {
                "mode": "selected-candidate-spans",
                "reference_candidate_id": reference.candidate_id,
                "reference_generation_revision": reference.generation_revision,
                "geometry_revision": _timing_content_digest([
                    {
                        "token": span.token,
                        "start": round(span.start, 6),
                        "end": round(span.end, 6),
                        "score": round(span.score, 8),
                    }
                    for span in reference.spans
                ]),
            },
        )
    hypotheses = assignment.get("alignment_hypotheses")
    if not isinstance(hypotheses, list):
        return None
    eligible: list[tuple[float, float, tuple[TimingTokenSpan, ...], dict[str, object]]] = []
    for hypothesis in hypotheses:
        if (
            not isinstance(hypothesis, dict)
            or hypothesis.get("source") not in {
                "ctc-local-window",
                "ctc-opening-local-retry-identity",
            }
            or hypothesis.get("authoritative") is not False
            or hypothesis.get("identity_support") != "supported"
            or hypothesis.get("sequence_support") != "supported"
            or bool(hypothesis.get("window_truncated"))
            or bool(hypothesis.get("right_edge_pileup"))
            or not _self_bound_hypothesis_record(hypothesis)
        ):
            continue
        spans = _timing_spans_from_value(hypothesis.get("spans"))
        if len(spans) < 4:
            continue
        try:
            raw_time = float(hypothesis.get("raw_candidate_time"))
            confidence = float(hypothesis.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(raw_time)
            or abs(raw_time - reference.written_time.seconds)
            > RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
        ):
            continue
        provenance = {
            "mode": "self-bound-local-ctc-geometry",
            "reference_candidate_id": reference.candidate_id,
            "reference_generation_revision": reference.generation_revision,
            "geometry_hypothesis_id": hypothesis.get("hypothesis_id"),
            "geometry_source": hypothesis.get("source"),
            "geometry_start": round(raw_time, 6),
            "selected_start": round(reference.written_time.seconds, 6),
            "agreement_seconds": round(abs(raw_time - reference.written_time.seconds), 6),
            "max_agreement_seconds": RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS,
            "geometry_revision": _timing_content_digest(hypothesis.get("spans")),
        }
        eligible.append((
            _finite_unit_confidence(confidence),
            -abs(raw_time - reference.written_time.seconds),
            spans,
            provenance,
        ))
    if not eligible:
        return None
    confidence, _neg_delta, spans, provenance = max(
        eligible, key=lambda item: (item[0], item[1], str(item[3].get("geometry_hypothesis_id")))
    )
    return float(provenance["geometry_start"]), spans, provenance


def apply_global_acoustic_recurrence_hypotheses(
    alignment_audio_path: Path,
    entries: list[LyricEntry],
    report: dict[str, object],
    evaluated: EvaluatedBackendTiming,
    duration: float,
) -> list[dict[str, object]]:
    """Search the whole song for exact repeated-lyric occurrences.

    A reference occurrence is eligible only when the central evaluator has
    already trusted its timing.  The search is independent of the target MMS
    timestamp and uses the nearest centrally-trusted lyric-order anchors only
    as broad bounds.  No match is emitted unless the full-line acoustic match
    is both strong and separated from the best distinct alternative.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return []
    try:
        features = acoustic_recurrence_features(alignment_audio_path)
        onset_features = analyze_vocal_onsets(alignment_audio_path, duration)
    except LrcError as exc:
        _record_backend_participation_failure(
            report, "apply_global_acoustic_recurrence_hypotheses", exc
        )
        return []
    audio_revision = hashlib.sha256(Path(alignment_audio_path).read_bytes()).hexdigest()
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        normalized = normalize_match_text(entry_sung_text(entry))
        if normalized:
            groups.setdefault(normalized, []).append(index)
    trusted = [bool(audit.timing_trusted) for audit in evaluated.state.audits]
    generated: list[dict[str, object]] = []
    for normalized, indexes in groups.items():
        if len(indexes) < 2:
            continue
        for target_index in indexes:
            if trusted[target_index]:
                continue
            reference_geometry: dict[int, tuple[float, tuple[TimingTokenSpan, ...], dict[str, object]]] = {}
            for reference_index in indexes:
                if not trusted[reference_index] or not assignment_content_is_trusted(assignments[reference_index]):
                    continue
                reference_assignment = assignments[reference_index]
                if not isinstance(reference_assignment, dict):
                    continue
                geometry = _trusted_recurrence_reference_geometry(
                    reference_assignment, evaluated.state.decisions[reference_index].candidate
                )
                if geometry is not None:
                    reference_geometry[reference_index] = geometry
            reference_indexes = sorted(reference_geometry)
            if not reference_indexes:
                continue
            reference_index = min(reference_indexes, key=lambda index: abs(index - target_index))
            reference = evaluated.state.decisions[reference_index].candidate
            template_start, template_spans, geometry_provenance = reference_geometry[reference_index]
            template_end = template_spans[-1].end
            if target_index > reference_index:
                lower = next(
                    (evaluated.state.decisions[index].written_time.seconds + 0.20
                     for index in range(target_index - 1, -1, -1) if trusted[index]),
                    template_end + 0.50,
                )
                upper = next(
                    (evaluated.state.decisions[index].written_time.seconds - 0.20
                     for index in range(target_index + 1, len(entries)) if trusted[index]),
                    duration,
                )
            else:
                lower = next(
                    (evaluated.state.decisions[index].written_time.seconds + 0.20
                     for index in range(target_index - 1, -1, -1) if trusted[index]),
                    0.0,
                )
                upper = next(
                    (evaluated.state.decisions[index].written_time.seconds - 0.20
                     for index in range(target_index + 1, len(entries)) if trusted[index]),
                    template_start - 0.50,
                )
            result = _search_recurrence_start(features, template_start, template_end, lower, upper)
            if result is None:
                continue
            score = float(result["score"])
            margin = float(result["margin"])
            candidate_start = float(result["start"])
            if score < RECURRENCE_MIN_SIMILARITY or margin < RECURRENCE_MIN_DISTINCT_MARGIN:
                continue
            peak = _nearest_local_maximum(onset_features, candidate_start, 0.22)
            if peak is None:
                continue
            peak_time, peak_strength, peak_prominence = peak
            if peak_strength < 0.55 or peak_prominence < 0.18:
                continue
            projected_spans = _project_recurrence_spans_from_geometry(
                template_spans, template_start, candidate_start, score
            )
            if len(projected_spans) < 4:
                continue
            evidence_window = {
                "search_start": round(lower, 3),
                "search_end": round(upper, 3),
                "audio_sha256": audio_revision,
                "feature_revision": RECURRENCE_FEATURE_REVISION,
                "reference_entry": reference_index + 1,
                "reference_candidate_id": reference.candidate_id,
                "reference_selected_start": round(reference.written_time.seconds, 6),
                "reference_start": round(template_start, 3),
                "reference_end": round(template_end, 3),
                "reference_geometry": geometry_provenance,
                "similarity": round(score, 8),
                "runner_similarity": result["runner_score"],
                "distinct_margin": round(margin, 8),
                "normalized_lyric_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            }
            hypothesis = record_alignment_hypothesis(
                assignments[target_index],
                source="global-acoustic-recurrence",
                stage="occurrence-level-global-recurrence-search",
                raw_time=candidate_start,
                confidence=max(0.0, min(1.0, score)),
                spans=projected_spans,
                identity_support="supported",
                sequence_support="supported",
                acoustic_support="supported",
                direct_onset_support="supported",
                acoustic_strength=peak_strength,
                acoustic_onset_time=peak_time,
                acoustic_evidence_producer="global-recurrence-search",
                acoustic_evidence_kind="recurrence-onset",
                direct_onset_time=peak_time,
                direct_onset_evidence_producer="global-recurrence-search",
                direct_onset_evidence_kind="recurrence-onset",
                parent_source="central-trusted-repeat-reference",
                parent_time=template_start,
                window=evidence_window,
                mode="full-line-acoustic-recurrence-with-trusted-reference",
            )
            generated.append({
                "entry": target_index + 1,
                "reference_entry": reference_index + 1,
                "candidate": round(candidate_start, 3),
                "similarity": round(score, 6),
                "distinct_margin": round(margin, 6),
                "peak_time": round(peak_time, 3),
                "peak_strength": round(peak_strength, 6),
                "hypothesis_id": hypothesis.get("hypothesis_id"),
            })
    report["global_acoustic_recurrence_hypotheses"] = generated
    report["global_acoustic_recurrence_count"] = len(generated)
    report["global_acoustic_recurrence_authoritative"] = False
    report["global_acoustic_recurrence_mutates_baseline"] = False
    return generated


def recurrence_hypothesis_provenance_is_valid(
    hypothesis: dict[str, object],
    entry: LyricEntry,
    alignment_audio_path: Path,
) -> bool:
    """Re-bind recurrence authority to its immutable evidence, not its source label."""
    if not _self_bound_hypothesis_record(hypothesis):
        return False
    if hypothesis.get("source") != "global-acoustic-recurrence":
        return False
    if hypothesis.get("stage") != "occurrence-level-global-recurrence-search":
        return False
    if hypothesis.get("mode") != "full-line-acoustic-recurrence-with-trusted-reference":
        return False
    if hypothesis.get("authoritative") is not False:
        return False
    parent = hypothesis.get("parent")
    window = hypothesis.get("window")
    if not isinstance(parent, dict) or parent.get("source") != "central-trusted-repeat-reference":
        return False
    if not isinstance(window, dict):
        return False
    try:
        raw_time = float(hypothesis.get("raw_candidate_time"))
        search_start = float(window.get("search_start"))
        search_end = float(window.get("search_end"))
        similarity = float(window.get("similarity"))
        margin = float(window.get("distinct_margin"))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (raw_time, search_start, search_end, similarity, margin)):
        return False
    if not (search_start <= raw_time < search_end):
        return False
    if similarity < RECURRENCE_MIN_SIMILARITY or margin < RECURRENCE_MIN_DISTINCT_MARGIN:
        return False
    if window.get("feature_revision") != RECURRENCE_FEATURE_REVISION:
        return False
    reference_candidate_id = window.get("reference_candidate_id")
    if not isinstance(reference_candidate_id, str) or not reference_candidate_id:
        return False
    expected_lyric_sha = hashlib.sha256(
        normalize_match_text(entry_sung_text(entry)).encode("utf-8")
    ).hexdigest()
    if window.get("normalized_lyric_sha256") != expected_lyric_sha:
        return False
    try:
        expected_audio_sha = _path_sha256(alignment_audio_path)
    except (OSError, LrcError):
        return False
    return bool(window.get("audio_sha256") == expected_audio_sha)


def repair_intro_hallucination(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(timestamps) < 3:
        return timestamps, []
    first_assignment = assignments[0] if isinstance(assignments[0], dict) else {}
    if assignment_score(first_assignment) >= COLLAPSE_SEGMENT_SCORE:
        return timestamps, []
    if not first_assignment.get("borrowed") and first_assignment.get("segment") is None:
        return timestamps, []
    if timestamps[0] >= 10.0:
        return timestamps, []

    anchor_index: int | None = None
    for index in range(1, min(len(timestamps), len(assignments))):
        assignment = assignments[index] if isinstance(assignments[index], dict) else {}
        if assignment_score(assignment) >= TRUSTED_ALIGNMENT_SCORE and timestamps[index] >= 20.0:
            anchor_index = index
            break
    if anchor_index is None or anchor_index < 2:
        return timestamps, []

    anchor_time = timestamps[anchor_index]
    prefix_count = anchor_index
    search_start = max(0.0, anchor_time - min(12.0, 4.0 + prefix_count * 4.0))
    search_end = max(search_start + 0.2, anchor_time - 6.0)
    features = analyze_audio(audio_path, duration)
    repaired_start = first_local_onset(features, search_start, search_end)
    if repaired_start is None:
        return timestamps, []
    if repaired_start <= timestamps[0] + 1.0:
        return timestamps, []

    repaired = list(timestamps)
    changes: list[dict[str, object]] = []
    interval = max(0.2, anchor_time - repaired_start)
    for index in range(prefix_count):
        fraction = index / max(1, prefix_count)
        new_time = repaired_start + interval * fraction
        if index > 0:
            new_time = max(new_time, repaired[index - 1] + 0.10)
        new_time = min(new_time, anchor_time - 0.10 * (prefix_count - index))
        changes.append(
            {
                "entry": index + 1,
                "from": round(repaired[index], 3),
                "to": round(new_time, 3),
                "reason": "intro-hallucination-prefix-redistribution",
                "anchor_entry": anchor_index + 1,
                "anchor_time": round(anchor_time, 3),
                "search_window": [round(search_start, 3), round(search_end, 3)],
                "lyric": entries[index].lines[0] if index < len(entries) and entries[index].lines else "",
            }
        )
        repaired[index] = new_time

    report["intro_hallucination_repair_count"] = len(changes)
    report["intro_hallucination_repairs"] = changes
    return repaired, changes


def repair_long_segment_hallucinations(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    raw_segments: list[AsrSegment],
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(timestamps) < 3:
        return timestamps, []
    features: AudioFeatures | None = None
    repaired = list(timestamps)
    changes: list[dict[str, object]] = []

    for index in range(1, min(len(repaired) - 1, len(assignments))):
        assignment = assignments[index] if isinstance(assignments[index], dict) else {}
        segment_number = assignment.get("segment")
        if not isinstance(segment_number, int):
            continue
        segment_index = segment_number - 1
        if not (0 <= segment_index < len(raw_segments)):
            continue
        segment = raw_segments[segment_index]
        segment_duration = segment.end - segment.start
        next_time = repaired[index + 1]
        score = assignment_score(assignment)
        if not (
            segment_duration >= 18.0
            and score >= 0.80
            and next_time - repaired[index] >= 12.0
            and abs(next_time - segment.end) <= 0.30
        ):
            continue
        text_length = len(normalize_match_text(entry_sung_text(entries[index])))
        lead = min(2.2, max(1.4, text_length * 0.055))
        new_time = max(repaired[index - 1] + 0.10, next_time - lead)
        if not (repaired[index] + 5.0 < new_time < next_time - 0.20):
            continue
        changes.append(
            {
                "entry": index + 1,
                "from": round(repaired[index], 3),
                "to": round(new_time, 3),
                "reason": "long-segment-pre-vocal-hallucination",
                "segment": segment_number,
                "segment_duration": round(segment_duration, 3),
                "next_anchor_time": round(next_time, 3),
                "lyric": entries[index].lines[0] if entries[index].lines else "",
            }
        )
        repaired[index] = new_time

        next_index = index + 1
        if next_index + 1 >= len(repaired):
            continue
        next_assignment = assignments[next_index] if isinstance(assignments[next_index], dict) else {}
        if assignment_score(next_assignment) < TRUSTED_ALIGNMENT_SCORE:
            continue
        if features is None:
            features = analyze_audio(audio_path, duration)
        search_start = repaired[next_index] + 1.20
        search_end = min(repaired[next_index] + 2.60, repaired[next_index + 1] - 0.10)
        onset = first_local_onset(features, search_start, search_end, threshold=0.55)
        if onset is None or not (repaired[next_index] + 0.40 < onset < repaired[next_index + 1] - 0.10):
            continue
        changes.append(
            {
                "entry": next_index + 1,
                "from": round(repaired[next_index], 3),
                "to": round(onset, 3),
                "reason": "post-long-segment-local-onset",
                "previous_repaired_entry": index + 1,
                "search_window": [round(search_start, 3), round(search_end, 3)],
                "lyric": entries[next_index].lines[0] if entries[next_index].lines else "",
            }
        )
        repaired[next_index] = onset

    if changes:
        report["long_segment_repair_count"] = len(changes)
        report["long_segment_repairs"] = changes
    return repaired, changes


def sync_scratch_assignment_timestamps(report: dict[str, object], timestamps: list[float]) -> None:
    """Synchronize mutable pre-finalization generation scratch only."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return
    for index, timestamp in enumerate(timestamps):
        if index < len(assignments) and isinstance(assignments[index], dict):
            assignments[index]["timestamp"] = round(timestamp, 3)


def assert_report_timestamp_equality(report: dict[str, object], state: FinalTimingState) -> None:
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(state.written_times):
        raise LrcError("Final report assignment count does not match immutable timing state")
    for index, (assignment, written_time, decision, audit) in enumerate(zip(
        assignments,
        state.written_times,
        state.decisions,
        state.audits,
    )):
        if not isinstance(assignment, dict):
            raise LrcError(f"Final report assignment {index + 1} is invalid")
        if assignment.get("canonical_written_centiseconds") != written_time.centiseconds:
            raise LrcError(f"Final report timestamp mismatch at entry {index + 1}")
        if assignment.get("lrc_tag") != written_time.lrc_tag:
            raise LrcError(f"Final report LRC tag mismatch at entry {index + 1}")
        if assignment.get("timestamp_status") != audit.status:
            raise LrcError(f"Final report timing status mismatch at entry {index + 1}")
        if assignment.get("timing_trusted") is not audit.timing_trusted:
            raise LrcError(f"Final report timing trust mismatch at entry {index + 1}")
        expected_selected = decision.selected_candidate_id if audit.timing_trusted else None
        if assignment.get("selected_candidate_id") != expected_selected:
            raise LrcError(f"Final report selection mismatch at entry {index + 1}")


def low_confidence_tail_start(report: dict[str, object], min_remaining: int = 4) -> int | None:
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return None
    for index in range(max(1, len(assignments) - 12), len(assignments) - min_remaining + 1):
        remaining = assignments[index:]
        low_count = sum(1 for item in remaining if assignment_score(item) < TRUSTED_ALIGNMENT_SCORE)
        if low_count < max(3, math.ceil(len(remaining) * 0.55)):
            continue
        consecutive_low = 0
        for item in remaining:
            if assignment_score(item) < TRUSTED_ALIGNMENT_SCORE:
                consecutive_low += 1
            else:
                break
        if consecutive_low >= 2:
            return index
    return None


def first_sustained_onset(
    features: AudioFeatures,
    start: float,
    end: float,
    onset_threshold: float = 0.70,
    median_rms_floor: float = -18.0,
    sustain_seconds: float = 3.0,
) -> float | None:
    if end <= start:
        return None
    frame_times = features.frame_times
    mask = (frame_times >= start) & (frame_times <= end)
    for index in np.flatnonzero(mask):
        strength = float(features.onset_strength[index])
        if strength < onset_threshold:
            continue
        time = float(frame_times[index])
        sustain_mask = (frame_times >= time) & (frame_times <= min(time + sustain_seconds, features.duration))
        if not np.any(sustain_mask):
            continue
        if float(np.median(features.rms_db[sustain_mask])) >= median_rms_floor:
            return time
    return None


def has_repeated_asr_tail(segments: list[AsrSegment], after_time: float, min_run: int = 6) -> bool:
    run_text = ""
    run_count = 0
    for segment in segments:
        if segment.start < after_time:
            continue
        normalized = normalize_match_text(segment.text)
        if len(normalized) < 8:
            run_text = ""
            run_count = 0
            continue
        if run_text and sequence_ratio(run_text, normalized) >= 0.86:
            run_count += 1
        else:
            run_text = normalized
            run_count = 1
        if run_count >= min_run:
            return True
    return False


def repair_tail_hallucination(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    segments: list[AsrSegment],
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    tail_start = low_confidence_tail_start(report)
    if tail_start is None or tail_start <= 0:
        return timestamps, []

    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return timestamps, []
    anchor_time = timestamps[tail_start - 1]
    if anchor_time <= 0.0:
        return timestamps, []
    if not has_repeated_asr_tail(segments, anchor_time + 4.0):
        report["tail_hallucination_repair"] = {
            "status": "skipped",
            "reason": "no-repeated-asr-tail",
            "candidate_start_entry": tail_start + 1,
            "anchor_entry": tail_start,
            "anchor_time": round(anchor_time, 3),
        }
        return timestamps, []

    features = analyze_audio(audio_path, duration)
    search_start = min(duration, anchor_time + 8.0)
    search_end = min(duration, anchor_time + 58.0)
    tail_time = first_sustained_onset(features, search_start, search_end)
    if tail_time is None:
        tail_time = min(duration, anchor_time + 12.0)
    if tail_time - anchor_time < 12.0:
        report["tail_hallucination_repair"] = {
            "status": "skipped",
            "reason": "tail-onset-too-close-to-anchor",
            "candidate_start_entry": tail_start + 1,
            "anchor_entry": tail_start,
            "anchor_time": round(anchor_time, 3),
            "tail_start_time": round(tail_time, 3),
        }
        return timestamps, []

    tail_end = duration
    if features.segments:
        tail_end = max(tail_time + 1.0, min(duration, features.segments[-1][1]))
    if tail_end - tail_time < max(8.0, (len(entries) - tail_start) * 1.4):
        return timestamps, []

    tail_times = weighted_rough_times(entries[tail_start:], tail_time, tail_end)
    tail_times = snap_to_onsets(tail_times, features, window_seconds=0.90, min_gap=0.35)
    repaired = list(timestamps)
    changes: list[dict[str, object]] = []
    for offset, entry_index in enumerate(range(tail_start, len(entries))):
        new_time = max(repaired[entry_index - 1] + 0.35 if entry_index > 0 else 0.0, tail_times[offset])
        old_time = repaired[entry_index]
        repaired[entry_index] = min(duration, new_time)
        if entry_index < len(assignments) and isinstance(assignments[entry_index], dict):
            assignment = assignments[entry_index]
            assignment["timestamp"] = round(repaired[entry_index], 3)
            _invalidate_timing_trust_after_timestamp_mutation(
                assignment, "tail-hallucination-acoustic"
            )
            assignment["timing_repair"] = "tail-hallucination-acoustic"
            assignment["timing_repair_source"] = "sustained-onset-weighted-tail"
        changes.append(
            {
                "entry": entry_index + 1,
                "from": round(old_time, 3),
                "to": round(repaired[entry_index], 3),
                "reason": "tail-hallucination-acoustic",
                "lyric": entries[entry_index].lines[0] if entries[entry_index].lines else "",
            }
        )

    report["tail_hallucination_repair"] = {
        "status": "applied",
        "start_entry": tail_start + 1,
        "anchor_entry": tail_start,
        "anchor_time": round(anchor_time, 3),
        "tail_start_time": round(tail_time, 3),
        "tail_end_time": round(tail_end, 3),
        "entries": len(changes),
        "method": "sustained-onset-weighted-tail",
    }
    report["tail_hallucination_repairs"] = changes
    update_report_confidence_metrics(report)
    return repaired, changes


def blend_tail_repair_with_forced(
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    forced_segments: list[AsrSegment],
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    assignments = report.get("assignments")
    if not isinstance(assignments, list):
        return timestamps, []
    refined = list(timestamps)
    changes: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict):
            continue
        if assignment.get("timing_repair") != "tail-hallucination-acoustic":
            continue
        if index >= len(forced_segments):
            continue
        forced_time = first_forced_char_time(forced_segments[index])
        current_time = refined[index]
        previous_time = refined[index - 1] if index > 0 else 0.0
        next_time = refined[index + 1] if index + 1 < len(refined) else duration
        if not (previous_time + 0.35 < forced_time < next_time - 0.35):
            continue
        lead = current_time - forced_time
        if abs(lead) <= 0.35:
            candidate = forced_time
        elif 0.35 < lead <= 6.0:
            candidate = current_time - min(1.60, lead * 0.34)
        else:
            continue
        if not (previous_time + 0.35 < candidate < next_time - 0.35):
            continue
        # Candidate producer only: never mutate the backend baseline before
        # central evaluation.  The forced-first estimate is advisory until an
        # independently proven opening/ownership relation validates it.
        changes.append(
            {
                "entry": index + 1,
                "from": round(current_time, 3),
                "to": round(candidate, 3),
                "forced_first": round(forced_time, 3),
                "reason": "tail-forced-blend",
                "lyric": entries[index].lines[0] if entries[index].lines else "",
            }
        )

    if changes:
        existing = report.get("tail_hallucination_repair")
        if isinstance(existing, dict):
            existing["method"] = "sustained-onset-weighted-tail+forced-blend"
            existing["forced_blend_count"] = len(changes)
        report["tail_forced_blend_count"] = len(changes)
        report["tail_forced_blends"] = changes
        report["tail_forced_blend_candidate_only"] = True
        report["tail_forced_blend_mutates_baseline"] = False
        update_report_confidence_metrics(report)
    return list(timestamps), changes


AUDIO_ADVISORY_REVIEW_FLAGS = {
    "unresolved_raw_ctc_disagreement",
    "candidate_disagreement",
    "long_line_disagreement",
    "raw_asr_systematic_drift",
    "phonetic_anchor_disagreement",
}


def duplicate_lyric_offset_consensus(
    entries: list[LyricEntry],
    timestamps: list[float],
) -> dict[str, object]:
    """Estimate a repeated-section offset from duplicate lyric lines."""
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        normalized = normalize_match_text(entry_sung_text(entry))
        if normalized:
            groups.setdefault(normalized, []).append(index)
    pairs: list[dict[str, object]] = []
    for normalized, indexes in groups.items():
        if len(indexes) < 2:
            continue
        # With 3+ chorus occurrences, compare consecutive occurrences only.
        # All-pairs would conflate adjacent and skip-over repeat structures.
        for ordinal, (first, second) in enumerate(zip(indexes, indexes[1:]), 1):
            pairs.append(
                {
                    "normalized": normalized,
                    "entries": [first + 1, second + 1],
                    "entry_delta": second - first,
                    "delta_seconds": round(timestamps[second] - timestamps[first], 3),
                    "occurrence_count": len(indexes),
                    "occurrence_pair": ordinal,
                }
            )

    # Preserve the historical evidence floor: fewer than four duplicate pairs
    # are not enough to establish a repeat-offset model at all.  v14.4 changes
    # only *how* sufficient evidence is partitioned, not the minimum amount of
    # duplicate evidence required.
    if len(pairs) < 4:
        return {
            "available": False,
            "reason": "fewer-than-four-duplicate-pairs",
            "mode": "lyric-index-displacement-clusters",
            "cluster_count": 0,
            "stable_cluster_count": 0,
            "clusters": [],
            "pair_count": len(pairs),
            "pairs": pairs,
        }

    # Repeated songs may contain more than one repeated section.  A single
    # global time-offset median falsely treats a legitimate second repeat
    # structure as an outlier.  Group first by immutable lyric-index
    # displacement, then estimate a robust temporal offset *inside* each
    # structurally corresponding section.
    cluster_rows: list[dict[str, object]] = []
    by_entry_delta: dict[int, list[dict[str, object]]] = collections.defaultdict(list)
    for pair in pairs:
        by_entry_delta[int(pair["entry_delta"])].append(pair)
    for entry_delta, members in sorted(by_entry_delta.items()):
        # A repeated *single lyric* is not enough to independently certify a
        # section offset, regardless of how many times that line appears.
        independent_lyrics = len({item["normalized"] for item in members})
        if len(members) < 2 or independent_lyrics < 2:
            continue
        deltas = [float(item["delta_seconds"]) for item in members]
        median_delta = float(statistics.median(deltas))
        deviations = [abs(value - median_delta) for value in deltas]
        mad = float(statistics.median(deviations))
        cluster_rows.append({
            "entry_delta": entry_delta,
            "available": mad <= 0.10,
            "median_delta_seconds": round(median_delta, 3),
            "mad_seconds": round(mad, 3),
            "pair_count": len(members),
            "independent_lyrics": independent_lyrics,
            "pairs": [dict(item) for item in members],
        })

    stable_clusters = [row for row in cluster_rows if row.get("available")]
    result: dict[str, object] = {
        "available": bool(stable_clusters),
        "mode": "lyric-index-displacement-clusters",
        "cluster_count": len(cluster_rows),
        "stable_cluster_count": len(stable_clusters),
        "clusters": cluster_rows,
        "pair_count": len(pairs),
        "pairs": pairs,
    }
    # Preserve the historical single-repeat diagnostic contract for callers
    # that only ever had one structural repeat displacement.  Multiple repeat
    # structures intentionally have no global median because such a value is
    # exactly the model error v14.4 removes.
    if len(stable_clusters) == 1:
        result["median_delta_seconds"] = stable_clusters[0]["median_delta_seconds"]
        result["mad_seconds"] = stable_clusters[0]["mad_seconds"]
    return result


def apply_post_score_audio_validation(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> list[dict[str, object]]:
    """Publish independent audio evidence without promoting pre-central trust.

    This pass never changes timestamps and never clears review/trust flags. It
    may add conservative review findings for repeated-section inconsistencies;
    positive onset evidence is emitted only as a central-evaluator hypothesis.
    """
    assignments = report.get("assignments")
    risks = report.get("suspicious_alignments")
    if (
        not isinstance(assignments, list)
        or len(assignments) != len(timestamps)
        or not isinstance(risks, list)
    ):
        return []
    # The original full mix has percussion and synthetic attacks which can
    # impersonate a lyric onset. Only use a real isolated Demucs vocal stem.
    stem_value = report.get("ctc_audio_path")
    stem_source = report.get("ctc_audio_source")
    stem = Path(stem_value) if isinstance(stem_value, str) else None
    if stem_source != "vocal-stem" or stem is None or not stem.is_file():
        report["post_score_audio_validation_status"] = "skipped-no-isolated-vocal-stem"
        report["post_score_all_line_vocal_audit"] = []
        return []
    try:
        features = analyze_audio(stem, duration)
    except (LrcError, OSError):
        report["post_score_audio_validation_status"] = "skipped-vocal-analysis-error"
        return []

    consensus = duplicate_lyric_offset_consensus(entries, timestamps)
    report["duplicate_lyric_offset_consensus"] = consensus
    cluster_medians: dict[int, float] = {}
    clusters = consensus.get("clusters")
    if isinstance(clusters, list):
        for cluster in clusters:
            if not isinstance(cluster, dict) or not cluster.get("available"):
                continue
            try:
                cluster_medians[int(cluster.get("entry_delta"))] = float(
                    cluster.get("median_delta_seconds")
                )
            except (TypeError, ValueError):
                continue

    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(entries):
        normalized = normalize_match_text(entry_sung_text(entry))
        if normalized:
            groups.setdefault(normalized, []).append(index)

    risk_by_entry = {
        int(item.get("entry")): item
        for item in risks
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    peaks: dict[int, dict[str, float] | None] = {
        index: nearest_local_peak(features, float(timestamps[index]))
        for index in range(len(timestamps))
    }

    # Analyze every lyric row, including previously trusted rows, so a
    # false-negative reviewer does not hide obvious silence. This is an
    # advisory observation; Central and final F1-F4 retain ownership.
    all_line_audit = []
    for index, timestamp in enumerate(timestamps):
        peak = peaks.get(index)
        observed = bool(
            isinstance(peak, dict)
            and abs(float(peak.get("delta_seconds",9.)))<=0.18
            and float(peak.get("strength",0.))>=0.42
        )
        all_line_audit.append({
            "entry":index+1,"acoustic_onset_nearby":observed,
            "review_required_before_audit":bool(
                isinstance(assignments[index],dict)
                and assignments[index].get("review_required")
            )
        })
    report["post_score_audio_validation_status"]="isolated-vocal-all-lines-checked"
    report["post_score_all_line_vocal_audit"]=all_line_audit
    resolved: list[dict[str, object]] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict) or not assignment.get("review_required"):
            continue
        entry_number = index + 1
        risk = risk_by_entry.get(entry_number)
        flags = set()
        assignment_flags = assignment.get("flags")
        if isinstance(assignment_flags, list):
            flags.update(str(flag) for flag in assignment_flags)
        if isinstance(risk, dict):
            risk_flags = risk.get("flags")
            if isinstance(risk_flags, list):
                flags.update(str(flag) for flag in risk_flags)

        peak = peaks.get(index)
        onset_supported = bool(
            isinstance(peak, dict)
            and abs(float(peak.get("delta_seconds", 9.0))) <= 0.18
            and float(peak.get("strength", 0.0)) >= 0.42
        )
        advisory_only = bool(flags) and flags.issubset(
            AUDIO_ADVISORY_REVIEW_FLAGS
        )

        normalized = normalize_match_text(entry_sung_text(entries[index]))
        duplicate_indexes = groups.get(normalized, [])
        duplicate_consensus_supported = False
        duplicate_detail: dict[str, object] | None = None
        if len(duplicate_indexes) >= 2 and index in duplicate_indexes:
            # S4: >2 repetitions are checked as consecutive section offsets.
            # The helper requires a stable independent multi-lyric cluster and
            # matching *real isolated-vocal* peaks for both occurrences.
            # This remains generic acoustic corroboration, NOT trusted timing.
            from duplicate_audio_consensus import consecutive_duplicate_vocal_support
            duplicate_consensus_supported, duplicate_detail = consecutive_duplicate_vocal_support(
                lyric_indexes=duplicate_indexes,
                current_index=index,
                timestamps=timestamps,
                peaks=peaks,
                stable_cluster_medians=cluster_medians,
            )

        resolution = ""
        if advisory_only and onset_supported:
            resolution = "audio-onset-confirms-selected-path"
        elif duplicate_consensus_supported and onset_supported:
            resolution = "duplicate-lyric-audio-onset-consensus"
        if not resolution:
            continue

        peak_time = float(peak.get("time")) if isinstance(peak, dict) and isinstance(peak.get("time"), (int, float)) else float(timestamps[index])
        peak_strength = float(peak.get("strength", 0.0) or 0.0) if isinstance(peak, dict) else 0.0
        spans = assignment.get("ctc_token_spans")
        record_alignment_hypothesis(
            assignment,
            source="audio-validated-current",
            stage="post-score-audio-validation",
            raw_time=float(timestamps[index]),
            confidence=max(0.90, _candidate_confidence(assignment)),
            spans=spans,
            identity_support="supported" if isinstance(spans, list) and bool(spans) else "unavailable",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="unavailable",
            acoustic_strength=peak_strength,
            acoustic_onset_time=peak_time,
            acoustic_evidence_producer="isolated-vocal-onset-validator",
            acoustic_evidence_kind="generic-acoustic",
            direct_onset_time=None,
            parent_source=str(report.get("backend") or "unknown") + "-current",
            parent_time=float(timestamps[index]),
            mode=resolution,
        )
        assignment["audio_validation_candidate_only"] = True
        assignment["audio_validation_resolution"] = resolution
        assignment["audio_validation_peak"] = peak
        if duplicate_detail is not None:
            assignment["audio_validation_duplicate_consensus"] = duplicate_detail
        if isinstance(risk, dict):
            risk["audio_validation_candidate_only"] = True
            risk["audio_validation_peak"] = peak
            if duplicate_detail is not None:
                risk["audio_validation_duplicate_consensus"] = duplicate_detail
        resolved.append(
            {
                "entry": entry_number,
                "text": entry_sung_text(entries[index]),
                "timestamp": round(float(timestamps[index]), 3),
                "resolution": resolution,
                "peak": peak,
                "duplicate_consensus": duplicate_detail,
            }
        )

    # A stable repeated section is also a false-negative detector. When one
    # duplicate pair is an outlier, flag only the occurrence that underwent an
    # automatic prefix recovery; the untouched occurrence remains the reference.
    duplicate_outliers: list[dict[str, object]] = []
    if cluster_medians:
        for normalized, indexes in groups.items():
            if len(indexes) != 2:
                continue
            first, second = indexes
            pair_median = cluster_medians.get(second - first)
            if pair_median is None:
                continue
            observed_delta = float(timestamps[second] - timestamps[first])
            error = observed_delta - pair_median
            if abs(error) <= 0.35:
                continue
            candidates = [first, second]
            recovered = [
                idx
                for idx in candidates
                if isinstance(assignments[idx], dict)
                and (
                    assignments[idx].get("ctc_forward_supported_prefix_recovery")
                    or assignments[idx].get("ctc_crossline_initial_recovery")
                )
            ]
            flagged_index = recovered[0] if len(recovered) == 1 else first
            assignment = assignments[flagged_index]
            if not isinstance(assignment, dict):
                continue
            assignment["review_required"] = True
            assignment["duplicate_occurrence_offset_mismatch"] = {
                "entries": [first + 1, second + 1],
                "observed_delta_seconds": round(observed_delta, 3),
                "median_delta_seconds": round(pair_median, 3),
                "error_seconds": round(error, 3),
            }
            flags_value = assignment.get("flags")
            assignment["flags"] = list(
                dict.fromkeys(
                    [
                        *(flags_value if isinstance(flags_value, list) else []),
                        "duplicate_occurrence_offset_mismatch",
                    ]
                )
            )
            risk = risk_by_entry.get(flagged_index + 1)
            if not isinstance(risk, dict):
                risk = {
                    "entry": flagged_index + 1,
                    "text": entry_sung_text(entries[flagged_index]),
                    "flags": [],
                    "severity": "high",
                    "review_required": True,
                }
                risks.append(risk)
                risk_by_entry[flagged_index + 1] = risk
            risk["review_required"] = True
            risk["severity"] = "high"
            risk["duplicate_occurrence_offset_mismatch"] = assignment[
                "duplicate_occurrence_offset_mismatch"
            ]
            risk_flags = risk.get("flags")
            risk["flags"] = list(
                dict.fromkeys(
                    [
                        *(risk_flags if isinstance(risk_flags, list) else []),
                        "duplicate_occurrence_offset_mismatch",
                    ]
                )
            )
            duplicate_outliers.append(
                {
                    "entry": flagged_index + 1,
                    **assignment["duplicate_occurrence_offset_mismatch"],
                }
            )

    report["post_score_audio_validation"] = resolved
    report["post_score_audio_validation_count"] = len(resolved)
    report["duplicate_occurrence_offset_outliers"] = duplicate_outliers
    report["duplicate_occurrence_offset_outlier_count"] = len(duplicate_outliers)
    report["post_score_audio_validation_authoritative"] = False
    report["post_score_audio_validation_can_promote_trust"] = False
    report["post_score_audio_validation_can_add_review"] = True
    report["suspicious_alignment_severity_counts"] = {
        "high": sum(
            1
            for item in risks
            if isinstance(item, dict) and item.get("severity") == "high"
        ),
        "medium": sum(
            1
            for item in risks
            if isinstance(item, dict) and item.get("severity") == "medium"
        ),
        "low": sum(
            1
            for item in risks
            if isinstance(item, dict) and item.get("severity") == "low"
        ),
    }
    return resolved


def score_line_timing_candidates(
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> list[float]:
    """Turn backend timestamps into an explainable per-line timing decision."""
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps):
        return timestamps
    risks = report.get("suspicious_alignments")
    risk_by_entry = {
        int(item.get("entry")): item
        for item in risks
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    } if isinstance(risks, list) else {}
    revised = list(timestamps)

    for index, entry in enumerate(entries):
        assignment = assignments[index]
        if not isinstance(assignment, dict):
            continue
        # Human-reviewed anchors are ground truth for this song. Candidate
        # scoring may describe other evidence, but must never overwrite them.
        if assignment.get("manual_anchor_hint"):
            anchor_time = float(assignment.get("timestamp", revised[index]) or revised[index])
            revised[index] = anchor_time
            assignment["chosen_time"] = round(anchor_time, 3)
            assignment["confidence"] = 1.0
            assignment["candidates"] = [
                {
                    "source": "manual_anchor",
                    "time": round(anchor_time, 3),
                    "score": 1.0,
                    "reasons": ["human-reviewed-anchor"],
                }
            ]
            assignment["rejected_candidates"] = []
            assignment["reasons"] = ["human-reviewed-anchor"]
            assignment["penalties"] = []
            assignment["flags"] = ["manual_anchor_resolved"]
            assignment["review_required"] = False
            continue
        entry_number = index + 1
        current_time = float(assignment.get("timestamp", revised[index]) or revised[index])
        previous_time = revised[index - 1] if index else 0.0
        next_time = revised[index + 1] if index + 1 < len(revised) else duration
        profile = lyric_anchor_profile(entry)
        normalized = normalize_match_text(entry_sung_text(entry))
        is_long = len(normalized) >= 18
        is_short = len(normalized) <= 4
        risk = risk_by_entry.get(entry_number, {})
        candidate_times = risk.get("candidate_timestamps") if isinstance(risk, dict) else {}
        if not isinstance(candidate_times, dict):
            candidate_times = {}

        # Local-window realignment can replace token spans after a cross-line
        # recovery was recorded. If the saved timestamp regresses to the weak
        # pre-recovery token, restore the effective supported token before
        # building the selected-path candidate.
        token_spans_for_recovery = assignment.get("ctc_token_spans")
        if assignment.get("ctc_crossline_initial_recovery") and isinstance(
            token_spans_for_recovery, list
        ):
            effective_index_for_recovery = 0
            try:
                effective_index_for_recovery = int(
                    assignment.get("ctc_effective_token_start_index", 0) or 0
                )
                recovery_evidence = assignment.get(
                    "ctc_crossline_initial_recovery_evidence"
                )
                ghost_span = (
                    float(recovery_evidence.get("ghost_span_seconds", 0.0) or 0.0)
                    if isinstance(recovery_evidence, dict)
                    else 0.0
                )
                recovered_span = token_spans_for_recovery[effective_index_for_recovery]
                recovered_time = float(recovered_span.get("start"))
                recovered_score = float(recovered_span.get("score", 0.0) or 0.0)
            except (AttributeError, IndexError, TypeError, ValueError):
                recovered_time = current_time
                recovered_score = 0.0
                ghost_span = 0.0
            if (
                effective_index_for_recovery > 0
                and ghost_span >= 0.80
                and recovered_score >= 0.08
                and recovered_time >= current_time + 0.25
                and previous_time + 0.05 < recovered_time < next_time - 0.05
            ):
                record_alignment_hypothesis(
                    assignment,
                    source="ctc-legacy-crossline-restored",
                    stage="score-line-crossline-restoration",
                    raw_time=recovered_time,
                    confidence=recovered_score,
                    spans=token_spans_for_recovery[effective_index_for_recovery:],
                    identity_support="supported",
                    sequence_support="supported",
                    parent_source="ctc-global-mms",
                    parent_time=current_time,
                    mode="legacy-crossline-restoration-candidate-only",
                )
                assignment["ctc_crossline_initial_recovery_candidate_generated"] = True

        try:
            selected_score = float(assignment.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            selected_score = 0.0
        crossline_recovery_trusted = bool(
            assignment.get("ctc_crossline_initial_recovery_trusted")
        )
        boundary_realign_trusted = bool(
            assignment.get("ctc_boundary_realign_trusted")
            or crossline_recovery_trusted
            or assignment.get("ctc_forward_supported_prefix_recovery_trusted")
        )
        selected_path_score = (
            0.98
            if boundary_realign_trusted
            else 0.55 + min(0.35, max(0.0, selected_score) * 0.35)
        )
        candidates: list[dict[str, object]] = [
            {
                "source": "selected_path",
                "time": round(current_time, 3),
                "score": round(selected_path_score, 3),
                "reasons": [
                    (
                        "multi-evidence-ctc-crossline-recovery"
                        if crossline_recovery_trusted
                        else "multi-evidence-ctc-boundary-realign"
                    )
                    if boundary_realign_trusted
                    else "backend-selected-timestamp"
                ],
            }
        ]
        raw_score = risk.get("raw_asr_score") if isinstance(risk, dict) else None
        raw_time = candidate_times.get("raw_asr", candidate_times.get("raw"))
        previous_ctc_token_end: float | None = None
        if index and isinstance(assignments[index - 1], dict):
            previous_spans = assignments[index - 1].get("ctc_token_spans")
            if isinstance(previous_spans, list):
                for span in previous_spans:
                    if not isinstance(span, dict):
                        continue
                    try:
                        token_end = float(span.get("end"))
                    except (TypeError, ValueError):
                        continue
                    previous_ctc_token_end = token_end if previous_ctc_token_end is None else max(previous_ctc_token_end, token_end)
        if isinstance(raw_time, (int, float)):
            try:
                raw_quality = float(raw_score) if raw_score is not None else 0.50
            except (TypeError, ValueError):
                raw_quality = 0.50
            candidates.append(
                {
                    "source": "raw_asr",
                    "time": round(float(raw_time), 3),
                    "score": round(0.35 + min(0.45, max(0.0, raw_quality) * 0.45), 3),
                    "reasons": ["raw-asr-lyric-match"],
                }
            )
        detached_initial_ctc_token = False
        detached_initial_ctc_token_score = 1.0
        first_ctc_token_score = 1.0
        token_spans = assignment.get("ctc_token_spans")
        effective_token_spans = token_spans
        if isinstance(token_spans, list):
            try:
                effective_start_index = int(
                    assignment.get("ctc_effective_token_start_index", 0) or 0
                )
            except (TypeError, ValueError):
                effective_start_index = 0
            effective_token_spans = token_spans[max(0, effective_start_index):]
        if isinstance(effective_token_spans, list) and len(effective_token_spans) >= 2:
            try:
                first_start = float(effective_token_spans[0].get("start"))
                second_start = float(effective_token_spans[1].get("start"))
                detached_initial_ctc_token = second_start - first_start >= 0.45
                detached_initial_ctc_token_score = float(
                    effective_token_spans[0].get("score", 1.0) or 0.0
                )
                first_ctc_token_score = detached_initial_ctc_token_score
            except (AttributeError, TypeError, ValueError):
                detached_initial_ctc_token = False
        current_boundary_gap = (
            current_time - previous_ctc_token_end
            if previous_ctc_token_end is not None and index
            else None
        )
        compressed_ctc_prefix = has_ctc_compressed_prefix(assignment, current_boundary_gap)
        if previous_ctc_token_end is not None and isinstance(raw_time, (int, float)):
            for candidate in candidates:
                if candidate.get("source") != "raw_asr":
                    continue
                if float(raw_time) <= previous_ctc_token_end + 0.03:
                    candidate["score"] = min(float(candidate["score"]), 0.10)
                    candidate.setdefault("reasons", []).append("raw-inside-previous-ctc-token-tail")
                break
        if boundary_realign_trusted and isinstance(raw_time, (int, float)):
            for candidate in candidates:
                if candidate.get("source") != "raw_asr":
                    continue
                if abs(float(raw_time) - current_time) > 0.35:
                    candidate["score"] = min(float(candidate["score"]), 0.10)
                    candidate.setdefault("reasons", []).append(
                        "cannot-override-trusted-multi-evidence-boundary-realign"
                    )
                break
        try:
            parsed_raw_score = float(raw_score) if raw_score is not None else 0.0
        except (TypeError, ValueError):
            parsed_raw_score = 0.0

        raw_candidate = next((candidate for candidate in candidates if candidate.get("source") == "raw_asr"), None)
        repeated_lyric = any(
            normalize_match_text(entry_sung_text(previous_entry)) == normalized
            for previous_entry in entries[max(0, index - 12) : index]
        )
        ambiguous_lyric_prefix = has_ambiguous_lyric_prefix(entries, index)
        short_duplicate_context = has_short_duplicate_context(entries, index)
        local_ctc_confirmed = bool(assignment.get("ctc_local_window_realign")) and first_ctc_token_score >= 0.50
        weak_local_raw_rescue = (
            isinstance(raw_candidate, dict)
            and bool(assignment.get("ctc_local_window_realign"))
            and first_ctc_token_score <= 0.12
            and parsed_raw_score >= 0.80
            and not repeated_lyric
            and not ambiguous_lyric_prefix
            and previous_time + 0.05 < float(raw_candidate["time"]) < next_time - 0.05
            and (
                previous_ctc_token_end is None
                or float(raw_candidate["time"]) > previous_ctc_token_end + 0.05
            )
        )
        if local_ctc_confirmed and isinstance(raw_candidate, dict):
            raw_candidate["score"] = 0.05
            raw_candidate.setdefault("reasons", []).append("local-vocal-ctc-confirms-first-token")
        elif weak_local_raw_rescue and isinstance(raw_candidate, dict):
            # This is deliberately narrow: local vocal CTC has a weak opening
            # posterior, raw ASR is strong and lies after the prior forced tail,
            # and the lyric is not a repeated phrase that raw matching can swap.
            raw_candidate["score"] = 0.98
            raw_candidate.setdefault("reasons", []).append("raw-resolves-weak-local-vocal-ctc")
        if detached_initial_ctc_token and parsed_raw_score >= 0.90:
            for candidate in candidates:
                if candidate.get("source") != "raw_asr":
                    continue
                candidate["score"] = 0.98
                candidate.setdefault("reasons", []).append("raw-resolves-detached-ctc-initial-token")
                break
        if (
            detached_initial_ctc_token
            and not assignment.get("ctc_crossline_initial_recovery")
            and detached_initial_ctc_token_score < 0.02
            and parsed_raw_score < 0.90
        ):
            try:
                second_token_start = float(effective_token_spans[1].get("start"))  # type: ignore[index]
            except (AttributeError, TypeError, ValueError, IndexError):
                second_token_start = current_time
            if previous_time + 0.05 < second_token_start < next_time - 0.05:
                candidates.append(
                    {
                        "source": "ctc_detached_initial_token_recovery",
                        "time": round(second_token_start, 3),
                        "score": 0.98,
                        "reasons": ["second-ctc-token-start-after-detached-initial-token"],
                    }
                )
        forced_time = candidate_times.get("whisperx_forced_first")
        if isinstance(forced_time, (int, float)):
            candidates.append(
                {
                    "source": "whisperx_forced_first",
                    "time": round(float(forced_time), 3),
                    "score": 0.58,
                    "reasons": ["forced-first-character"],
                }
            )
        peaks = assignment.get("ctc_first_token_candidates")
        nearby_peaks: list[dict[str, object]] = []
        if isinstance(peaks, list):
            # A posterior peak inside the preceding lyric's final CTC token is
            # a tail echo, not evidence for this line's opening consonant.
            # Keeping it here falsely turned an already-rejected candidate
            # into a phonetic-anchor review flag.
            peak_floor = (previous_ctc_token_end + 0.03) if previous_ctc_token_end is not None else None
            nearby_peaks = [
                peak
                for peak in peaks
                if isinstance(peak, dict)
                and isinstance(peak.get("time"), (int, float))
                and (peak_floor is None or float(peak["time"]) > peak_floor)
                and abs(float(peak["time"]) - current_time) <= 0.45
            ]
            if nearby_peaks:
                peak = max(nearby_peaks, key=lambda item: float(item.get("score", 0.0) or 0.0))
                peak_score = float(peak.get("score", 0.0) or 0.0)
                candidates.append(
                    {
                        "source": "ctc_nearby_phonetic_peak",
                        "time": round(float(peak["time"]), 3),
                        "score": round(0.48 + min(0.30, peak_score * 0.30), 3),
                        "reasons": ["phonetic-first-token-peak-near-selected"],
                    }
                )

        # CTC can skip to a later occurrence when a line begins with the same
        # visible word two or more times. A strong earlier first-token posterior
        # is then more specific evidence than the collapsed whole-line path.
        leading_term, leading_term_count = repeated_leading_term(entry_sung_text(entry))
        try:
            ctc_score = float(assignment.get("ctc_score", 1.0) or 1.0)
        except (TypeError, ValueError):
            ctc_score = 1.0
        repeated_onset_candidate: dict[str, object] | None = None
        single_cjk_repeat = _is_single_cjk_or_kana_term(leading_term)
        weak_repeated_path = bool(
            ctc_score <= 0.18 and first_ctc_token_score <= 0.05
        )
        if (
            leading_term_count >= 2
            and (weak_repeated_path or single_cjk_repeat)
            and isinstance(peaks, list)
        ):
            earlier_peaks: list[dict[str, float]] = []
            repeated_peak_floor = previous_time + 0.20
            if previous_ctc_token_end is not None:
                # A repeated-word posterior that still falls inside the prior
                # line's forced token tail is not the first occurrence of this
                # line.  This is the same cross-line leakage seen with raw ASR
                # and nearby phonetic peaks.
                repeated_peak_floor = max(repeated_peak_floor, previous_ctc_token_end + 0.03)
            for peak in peaks:
                if not isinstance(peak, dict):
                    continue
                try:
                    peak_time = float(peak.get("time"))
                    peak_score = float(peak.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if repeated_peak_floor < peak_time <= current_time - 0.45 and peak_score >= 0.005:
                    earlier_peaks.append({"time": peak_time, "score": peak_score})
            if earlier_peaks:
                strongest_score = max(peak["score"] for peak in earlier_peaks)
                credible_peaks = [
                    peak for peak in earlier_peaks
                    if peak["score"] >= max(0.005, strongest_score * 0.65)
                ]
                peak = min(credible_peaks, key=lambda item: item["time"])
                repeated_onset_candidate = {
                    "source": "ctc_repeated_leading_term_onset",
                    "time": round(peak["time"], 3),
                    # This is a deliberately narrow override: repeated visible
                    # text plus a strong earlier same-token posterior.
                    "score": (
                        0.60
                        if single_cjk_repeat and not weak_repeated_path
                        else 0.97
                    ),
                    "reasons": [
                        "repeated-leading-term-earlier-ctc-onset",
                        *(
                            ["single-cjk-repeat-review-only"]
                            if single_cjk_repeat and not weak_repeated_path
                            else []
                        ),
                        f"term={leading_term}",
                        f"repetitions={leading_term_count}",
                    ],
                }
                candidates.append(repeated_onset_candidate)

        # Reject impossible candidates before measuring disagreement. A raw-ASR
        # match from another repeated occurrence must not lower confidence for a
        # CTC path that is inside the current line's monotonic neighbor bounds.
        for candidate in candidates:
            candidate_time = float(candidate["time"])
            if not (previous_time + 0.05 < candidate_time < next_time - 0.05):
                candidate["score"] = round(max(0.0, float(candidate["score"]) - 0.35), 3)
                reasons = candidate.setdefault("reasons", [])
                if "outside-neighbor-bounds" not in reasons:
                    reasons.append("outside-neighbor-bounds")
        credible_candidates = [
            candidate
            for candidate in candidates
            if "outside-neighbor-bounds" not in candidate.get("reasons", [])
            and "raw-inside-previous-ctc-token-tail" not in candidate.get("reasons", [])
            and float(candidate.get("score", 0.0) or 0.0) >= 0.20
        ]

        penalties: list[dict[str, object]] = []
        existing_flags = assignment.get("flags")
        flags: list[str] = list(existing_flags) if isinstance(existing_flags, list) else []
        unresolved_boundary = bool(assignment.get("ctc_unresolved_boundary"))
        numeric_times = [float(candidate["time"]) for candidate in credible_candidates]
        spread = max(numeric_times) - min(numeric_times) if len(numeric_times) > 1 else 0.0
        if spread >= 0.75 and not weak_local_raw_rescue:
            penalties.append({"kind": "candidate_disagreement", "value": 0.25, "seconds": round(spread, 3)})
            flags.append("candidate_disagreement")
        if current_time <= previous_time + 0.12 and index:
            penalties.append({"kind": "previous_line_tail_attachment", "value": 0.20})
            flags.append("previous_line_tail_attachment")
        if compressed_ctc_prefix:
            # Do not let a whole-line average turn a forced monotonic handoff
            # into a seemingly strong timestamp. A compact CTC repair may
            # clear this flag; an unresolved case remains explicit review.
            penalties.append({"kind": "ctc_zero_gap_compressed_prefix", "value": 0.36})
            flags.append("ctc_zero_gap_compressed_prefix")
        if is_long and spread >= 1.25 and not weak_local_raw_rescue:
            penalties.append({"kind": "long_line_disagreement", "value": 0.18})
            flags.append("long_line_disagreement")
        if is_short and spread >= 0.45:
            penalties.append({"kind": "short_line_onset_uncertain", "value": 0.16})
            flags.append("short_line_onset_uncertain")
        timing_sources = assignment.get("timing_trusted_sources")
        timing_sources_set = set(timing_sources) if isinstance(timing_sources, list) else set()
        asr_only_trust = bool(timing_sources_set) and timing_sources_set.issubset({"whisperx", "raw"})
        if short_duplicate_context and asr_only_trust:
            penalties.append({"kind": "short_duplicate_asr_consensus", "value": 0.24})
            flags.append("short_duplicate_asr_consensus_untrusted")
        first_mora = str(profile.get("first_mora", ""))
        if first_mora and isinstance(peaks, list) and peaks:
            peak_times = [float(item["time"]) for item in nearby_peaks if isinstance(item.get("time"), (int, float))]
            if peak_times and min(abs(current_time - time) for time in peak_times) > 0.35:
                penalties.append({"kind": "phonetic_anchor_disagreement", "value": 0.12, "first_mora": first_mora})
                flags.append("phonetic_anchor_disagreement")

        ranked = sorted(candidates, key=lambda candidate: float(candidate["score"]), reverse=True)
        chosen = ranked[0]
        chosen_time = float(chosen["time"])
        total_penalty = sum(float(item["value"]) for item in penalties)
        confidence = max(0.0, min(1.0, float(chosen["score"]) - total_penalty))
        safe_to_replace = (
            chosen.get("source") != "selected_path"
            and not boundary_realign_trusted
            and float(chosen["score"]) >= 0.74
            and confidence >= 0.70
            and previous_time + 0.05 < chosen_time < next_time - 0.05
        )
        if safe_to_replace:
            assignment["precentral_candidate_recommendation"] = {
                "source": chosen.get("source"),
                "time": round(chosen_time, 3),
                "score": round(float(chosen.get("score", 0.0) or 0.0), 3),
                "confidence": round(confidence, 3),
                "authoritative": False,
            }
        # Candidate scoring is diagnostic/candidate generation only. The backend
        # selected_path remains unchanged until central evaluation commits.
        safe_to_replace = False
        chosen = next(candidate for candidate in candidates if candidate.get("source") == "selected_path")
        chosen_time = current_time
        confidence = max(
            0.0,
            min(1.0, float(chosen["score"]) - total_penalty),
        )
        isolated_low_ctc_path = (
            ctc_score < 0.03
            and not flags
            and spread < 0.75
            and not (isinstance(risk, dict) and risk.get("review_required"))
        )
        raw_resolves_detached_ctc = (
            safe_to_replace
            and chosen.get("source") == "raw_asr"
            and detached_initial_ctc_token
            and parsed_raw_score >= 0.90
        )
        raw_resolves_weak_local_ctc = (
            safe_to_replace
            and chosen.get("source") == "raw_asr"
            and weak_local_raw_rescue
        )
        review_required = bool(assignment.get("review_required"))
        if isinstance(risk, dict):
            review_required = review_required or bool(risk.get("review_required"))
        raw_rejected_by_sequence = bool(
            isinstance(raw_candidate, dict)
            and (
                "outside-neighbor-bounds" in raw_candidate.get("reasons", [])
                or "raw-inside-previous-ctc-token-tail" in raw_candidate.get("reasons", [])
            )
        )
        risk_flags = set(risk.get("flags", [])) if isinstance(risk, dict) and isinstance(risk.get("flags"), list) else set()
        raw_only_risk = bool(risk_flags) and risk_flags.issubset({
            "unresolved_raw_ctc_disagreement",
            "candidate_disagreement",
            "long_line_disagreement",
            "phonetic_anchor_disagreement",
        })
        # A lone weak CTC path remains untrusted in the report, but score alone
        # does not establish a timing defect. Require a concrete conflict before
        # escalating it to manual review.
        review_required = review_required or (
            confidence < 0.68 and not isolated_low_ctc_path
        ) or spread >= 1.25 or (short_duplicate_context and asr_only_trust) or compressed_ctc_prefix or unresolved_boundary
        if (
            not unresolved_boundary
            and (
                raw_resolves_detached_ctc
                or raw_resolves_weak_local_ctc
                or local_ctc_confirmed
                or crossline_recovery_trusted
                or (raw_rejected_by_sequence and raw_only_risk)
            )
        ):
            review_required = False
        if review_required and assignment.get("timing_trusted"):
            assignment.pop("timing_trusted", None)
            assignment.pop("timing_trusted_reason", None)
            assignment.pop("timing_trusted_sources", None)
            assignment.pop("timing_trusted_candidate_times", None)
            flags.append("timing_trust_removed_for_review")
        assignment["chosen_time"] = round(chosen_time, 3)
        assignment["confidence"] = round(confidence, 3)
        if compressed_ctc_prefix:
            assignment["score"] = min(float(assignment.get("score", 0.0) or 0.0), 0.35)
        assignment["candidates"] = candidates
        assignment["rejected_candidates"] = [candidate for candidate in candidates if candidate is not chosen]
        assignment["reasons"] = list(chosen.get("reasons", []))
        assignment["penalties"] = penalties
        assignment["flags"] = list(dict.fromkeys(flags))
        assignment["review_required"] = review_required
        assignment["phonetic_anchor"] = {
            "system": profile.get("phonetic_system"),
            "first_mora": first_mora,
            "last_mora": profile.get("last_mora"),
        }
        if is_long and (spread >= 1.25 or "long_line_disagreement" in flags):
            midpoint = max(1, len(entry_sung_text(entry)) // 2)
            assignment["split_suggestion"] = {
                "reason": "long-line-multiple-phrase-candidates",
                "suggested_after_text": entry_sung_text(entry)[:midpoint],
            }
        if review_required and isinstance(risk, dict):
            risk["chosen_time"] = round(chosen_time, 3)
            risk["confidence"] = round(confidence, 3)
            risk["candidates"] = candidates
            risk["rejected_candidates"] = assignment["rejected_candidates"]
            risk["reasons"] = assignment["reasons"]
            risk["penalties"] = penalties
            risk["flags"] = list(dict.fromkeys([*(risk.get("flags") if isinstance(risk.get("flags"), list) else []), *flags]))
            risk["review_required"] = True
        elif (
            raw_resolves_detached_ctc
            or raw_resolves_weak_local_ctc
            or local_ctc_confirmed
            or crossline_recovery_trusted
            or (raw_rejected_by_sequence and raw_only_risk)
        ) and isinstance(risk, dict):
            risk["review_required"] = False
            risk["severity"] = "resolved"
            if crossline_recovery_trusted:
                risk["resolution"] = "multi-evidence-ctc-crossline-recovery"
            elif raw_rejected_by_sequence and raw_only_risk:
                risk["resolution"] = "raw-candidate-rejected-by-monotonic-neighbor-bounds"
            elif local_ctc_confirmed:
                risk["resolution"] = "local-vocal-ctc-confirms-first-token"
            elif raw_resolves_weak_local_ctc:
                risk["resolution"] = "high-confidence-raw-resolves-weak-local-vocal-ctc"
            else:
                risk["resolution"] = "high-confidence-raw-resolves-detached-ctc-initial-token"
            prior_risk_flags = risk.get("flags")
            if isinstance(prior_risk_flags, list):
                resolution_flag = (
                    "ctc_crossline_recovery_confirmed"
                    if crossline_recovery_trusted
                    else "raw_candidate_rejected_by_sequence"
                    if raw_rejected_by_sequence and raw_only_risk
                    else "local_vocal_ctc_confirmed"
                    if local_ctc_confirmed
                    else "weak_local_vocal_ctc_resolved"
                    if raw_resolves_weak_local_ctc
                    else "detached_ctc_initial_token_resolved"
                )
                risk["flags"] = list(dict.fromkeys([*prior_risk_flags, resolution_flag]))
    # Pre-central scoring cannot normalize or rewrite backend timestamps.
    # Ordering is evaluated from immutable candidate sets by the central layer.
    if not isinstance(risks, list):
        risks = []
        report["suspicious_alignments"] = risks
    existing_entries = {
        int(item.get("entry"))
        for item in risks
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, dict) or not assignment.get("review_required"):
            continue
        entry_number = index + 1
        if entry_number in existing_entries:
            continue
        risks.append(
            {
                "entry": entry_number,
                "text": entry_sung_text(entries[index]),
                "flags": assignment.get("flags", []),
                "severity": "medium" if float(assignment.get("confidence", 0.0) or 0.0) >= 0.45 else "high",
                "review_required": True,
                "chosen_time": assignment.get("chosen_time"),
                "confidence": assignment.get("confidence"),
                "candidates": assignment.get("candidates", []),
                "rejected_candidates": assignment.get("rejected_candidates", []),
                "reasons": assignment.get("reasons", []),
                "penalties": assignment.get("penalties", []),
            }
        )
    report["suspicious_alignment_count"] = len(risks)
    report["suspicious_alignment_severity_counts"] = {
        "high": sum(1 for item in risks if isinstance(item, dict) and item.get("severity") == "high"),
        "medium": sum(1 for item in risks if isinstance(item, dict) and item.get("severity") == "medium"),
        "low": sum(1 for item in risks if isinstance(item, dict) and item.get("severity") == "low"),
    }
    update_report_confidence_metrics(report)
    return list(timestamps)


def apply_short_duplicate_acoustic_recovery(
    audio_path: Path,
    entries: list[LyricEntry],
    timestamps: list[float],
    report: dict[str, object],
    duration: float,
) -> tuple[list[float], dict[str, object], list[dict[str, object]]]:
    """Recover a short duplicate phrase that was assigned to a later occurrence."""
    assignments = report.get("assignments")
    risks = report.get("suspicious_alignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps) or not isinstance(risks, list):
        return timestamps, report, []
    try:
        features = analyze_audio(audio_path, duration)
    except LrcError:
        return timestamps, report, []

    risk_by_entry = {
        int(item.get("entry")): item
        for item in risks
        if isinstance(item, dict) and isinstance(item.get("entry"), int)
    }
    recovered = list(timestamps)
    changes: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        assignment = assignments[index]
        if not isinstance(assignment, dict) or not has_short_duplicate_context(entries, index):
            continue
        if assignment.get("ctc_boundary_realign_trusted") or assignment.get("ctc_unresolved_boundary"):
            continue
        flags = assignment.get("flags")
        flags = flags if isinstance(flags, list) else []
        if "short_duplicate_asr_consensus_untrusted" not in flags:
            continue
        try:
            current_time = float(assignment.get("timestamp", recovered[index]) or recovered[index])
        except (TypeError, ValueError):
            continue
        previous_time = recovered[index - 1] if index else 0.0
        next_time = recovered[index + 1] if index + 1 < len(recovered) else duration
        if current_time - previous_time < 3.50:
            continue
        search_start = max(previous_time + 0.40, current_time - 7.00)
        search_end = current_time - 0.45
        if search_end - search_start < 0.50:
            continue
        supporting_entries: list[int] = []
        support_times: list[float] = []
        for prior_entry in range(max(1, index - 5), index + 1):
            risk = risk_by_entry.get(prior_entry)
            if not isinstance(risk, dict):
                continue
            candidate_times = risk.get("candidate_timestamps")
            if not isinstance(candidate_times, dict):
                continue
            for key in ("raw", "raw_asr", "whisperx", "whisperx_forced_first"):
                value = candidate_times.get(key)
                if isinstance(value, (int, float)) and search_start <= float(value) <= search_end:
                    supporting_entries.append(prior_entry)
                    support_times.append(float(value))
                    break
        if not supporting_entries:
            continue
        focus_time = min(support_times)
        focused_start = max(search_start, focus_time - 0.45)
        focused_end = min(search_end, focus_time + 0.45)
        candidate = first_local_onset(features, focused_start, focused_end, threshold=0.65)
        if candidate is None:
            candidate = first_local_onset(features, focused_start, focused_end, threshold=0.55)
        if candidate is None:
            continue
        if min(abs(candidate - support_time) for support_time in support_times) > 1.20:
            continue
        shift = current_time - candidate
        if not (1.20 <= shift <= 6.80):
            continue
        if not (previous_time + 0.05 < candidate < next_time - 0.05):
            continue

        nearest_index = int(np.argmin(np.abs(features.frame_times - float(candidate)))) if len(features.frame_times) else 0
        onset_strength = (
            float(features.onset_strength[nearest_index]) if len(features.onset_strength) else 0.0
        )
        record_alignment_hypothesis(
            assignment,
            source="short-duplicate-acoustic",
            stage="short-duplicate-acoustic-recovery",
            raw_time=float(candidate),
            confidence=max(0.0, min(1.0, onset_strength)),
            spans=[],
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="unavailable",
            acoustic_strength=onset_strength,
            acoustic_onset_time=float(candidate),
            acoustic_evidence_producer="short-duplicate-full-mix-onset",
            acoustic_evidence_kind="generic-acoustic",
            direct_onset_time=None,
            parent_source=str(report.get("backend") or "unknown") + "-current",
            parent_time=current_time,
            mode="short-duplicate-acoustic-onset-with-neighbor-asr-context",
        )
        assignment["short_duplicate_acoustic_candidate_generated"] = True
        assignment["short_duplicate_acoustic_recovery_supporting_entries"] = supporting_entries
        risk = risk_by_entry.get(index + 1)
        if isinstance(risk, dict):
            candidate_times = risk.get("candidate_timestamps")
            if not isinstance(candidate_times, dict):
                candidate_times = {}
            candidate_times["acoustic_onset_candidate"] = round(candidate, 3)
            risk["candidate_timestamps"] = candidate_times
        changes.append(
            {
                "entry": index + 1,
                "from": round(current_time, 3),
                "to": round(candidate, 3),
                "shift": round(shift, 3),
                "supporting_entries": supporting_entries,
                "reason": "short-duplicate-acoustic-onset-with-neighbor-asr-context",
            }
        )
    if not changes:
        return timestamps, report, []

    report["short_duplicate_acoustic_recovery_count"] = len(changes)
    report["short_duplicate_acoustic_recoveries"] = changes
    report["short_duplicate_acoustic_recovery_authoritative"] = False
    report["short_duplicate_acoustic_recovery_mutates_baseline"] = False
    return list(timestamps), report, changes


def apply_whisperx_lyric_refinement(
    entries: list[LyricEntry],
    base_timestamps: list[float],
    report: dict[str, object],
    asr_segments: list[AsrSegment],
    forced_segments: list[AsrSegment],
    duration: float,
) -> tuple[list[float], list[dict[str, object]]]:
    """Publish forced-first alternatives without changing the backend baseline.

    Historical versions contained lyric-shape and content-specific branches that
    moved timestamps before central evaluation.  Those rules are intentionally
    removed.  A forced-alignment first-character time is now only an identity/
    sequence hint; it carries no direct-onset authority and must independently
    satisfy the central prefix/tail evidence contract before it can be selected.
    """
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(entries):
        return list(base_timestamps), []

    changes: list[dict[str, object]] = []
    limit = min(len(entries), len(base_timestamps), len(forced_segments), len(assignments))
    for index in range(limit):
        assignment = assignments[index]
        if not isinstance(assignment, dict):
            continue
        forced_time = first_forced_char_time(forced_segments[index])
        if not math.isfinite(forced_time) or not (0.0 <= forced_time <= duration):
            continue
        baseline = float(base_timestamps[index])
        if abs(forced_time - baseline) < HOP_SECONDS:
            continue
        hypothesis = record_alignment_hypothesis(
            assignment,
            source="whisperx-forced-first-advisory",
            stage="forced-alignment-advisory",
            raw_time=forced_time,
            confidence=min(0.58, _candidate_confidence(assignment)),
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="unavailable",
            direct_onset_support="unavailable",
            parent_source="whisperx-current",
            parent_time=baseline,
            mode="candidate-only-no-self-onset-proof",
        )
        changes.append(
            {
                "entry": index + 1,
                "from": round(baseline, 3),
                "candidate": round(forced_time, 3),
                "delta_seconds": round(forced_time - baseline, 3),
                "reason": "forced-first-advisory-only",
                "hypothesis_id": hypothesis.get("hypothesis_id"),
                "lyric": entry_sung_text(entries[index]),
            }
        )

    report["whisperx_refinement_candidate_only"] = True
    report["whisperx_refinement_mutates_baseline"] = False
    report["whisperx_refinement_candidates"] = copy.deepcopy(changes)
    report["whisperx_refinement_removed_special_case_rules"] = True
    return list(base_timestamps), changes


def first_late_onset(features: AudioFeatures, start: float, end: float, threshold: float = 0.55) -> float | None:
    if end <= start:
        return None
    frame_times = features.frame_times
    onset = features.onset_strength
    mask = (frame_times >= start) & (frame_times <= end)
    indices = np.flatnonzero(mask)
    for index in indices:
        strength = float(onset[index])
        if strength < threshold:
            continue
        left = float(onset[index - 1]) if index > 0 else 0.0
        right = float(onset[index + 1]) if index + 1 < len(onset) else 0.0
        if strength >= left and strength >= right:
            return float(frame_times[index])
    return None


def apply_acoustic_late_onset_refinement(
    audio_path: Path,
    duration: float,
    timestamps: list[float],
    report: dict[str, object],
) -> tuple[list[float], list[dict[str, object]]]:
    """Publish late-onset alternatives without mutating the WhisperX baseline."""
    forced_times = report.get("whisperx_forced_first_times")
    assignments = report.get("assignments")
    if not isinstance(forced_times, list) or not isinstance(assignments, list):
        return list(timestamps), []

    features = analyze_audio(audio_path, duration)
    changes: list[dict[str, object]] = []
    for index, current_time in enumerate(timestamps):
        if index >= len(forced_times) or index >= len(assignments):
            continue
        assignment = assignments[index]
        if not isinstance(assignment, dict):
            continue
        try:
            forced_time = float(forced_times[index])
        except (TypeError, ValueError):
            continue
        forced_lead = current_time - forced_time
        score = float(assignment.get("score", 0.0) or 0.0)
        if not (0.45 <= forced_lead <= 3.0):
            continue
        if score > 0.80 and forced_lead > 2.10:
            continue
        next_time = timestamps[index + 1] if index + 1 < len(timestamps) else duration
        if next_time - current_time < 1.0:
            continue
        search_start = current_time + 0.42
        search_end = min(current_time + 0.90, next_time - 0.12, duration)
        candidate = first_late_onset(features, search_start, search_end)
        if candidate is None:
            continue
        shift = candidate - current_time
        accept_shift = (
            (score <= 0.80 and shift <= 0.65)
            or (score > 0.90 and forced_lead < 0.90 and shift <= 0.75)
            or (score > 0.90 and forced_lead >= 1.60 and shift <= 0.50)
        )
        if not accept_shift:
            continue
        previous_time = timestamps[index - 1] if index > 0 else 0.0
        if not (previous_time + 0.10 < candidate < next_time - 0.10):
            continue
        onset_strength = None
        if len(features.frame_times):
            nearest = int(np.argmin(np.abs(features.frame_times - candidate)))
            onset_strength = float(features.onset_strength[nearest])
        record_alignment_hypothesis(
            assignment,
            source="whisperx-acoustic-late-onset",
            stage="explicit-whisperx-acoustic-late-onset",
            raw_time=candidate,
            confidence=score,
            identity_support="supported" if assignment_has_timing(assignment) else "unavailable",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=onset_strength,
            acoustic_onset_time=candidate,
            acoustic_evidence_producer="full-mix-late-onset",
            acoustic_evidence_kind="generic-acoustic",
            direct_onset_time=None,
            parent_source="whisperx-word",
            parent_time=current_time,
            window={"start": round(search_start, 3), "end": round(search_end, 3)},
            mode="acoustic-late-onset",
        )
        changes.append(
            {
                "entry": index + 1,
                "from": round(current_time, 3),
                "to": round(candidate, 3),
                "reason": "acoustic-late-onset-candidate",
                "forced_lead": round(forced_lead, 3),
                "score": round(score, 3),
                "authoritative": False,
            }
        )

    report["acoustic_late_onset_refinement_authoritative"] = False
    report["acoustic_late_onset_refinement_mutates_baseline"] = False
    return list(timestamps), changes
