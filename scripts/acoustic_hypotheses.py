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
from vocal_onset_hypotheses import (
    VOCAL_ONSET_CONFIRM_RADIUS_SECONDS,
    VOCAL_ONSET_CONFIRM_MIN_STRENGTH,
    VOCAL_ONSET_CONFIRM_MIN_PROMINENCE,
    VOCAL_ACTIVITY_REENTRY_FEATURE_REVISION,
    VOCAL_ACTIVITY_REENTRY_MIN_GAP_SECONDS,
    _candidate_spans_from_assignment,
    _nearest_local_maximum,
    _strong_vocal_onset_near,
    _vocal_activity_reentry_for_time,
    VOCAL_ONSET_RAW_ORDER_TOLERANCE_SECONDS,
    VOCAL_ONSET_RAW_ORDER_MIN_CONTRADICTIONS,
    VOCAL_ONSET_RAW_ORDER_MIN_SCORE,
    _raw_asr_order_anchors,
    _raw_asr_order_contradictions,
    apply_vocal_onset_confirmation_hypotheses,
    apply_vocal_activity_reentry_hypotheses,
)
from recurrence_hypotheses import (
    MUTUAL_RECURRENCE_FEATURE_REVISION,
    MUTUAL_RECURRENCE_MIN_SIMILARITY,
    MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN,
    MUTUAL_RECURRENCE_ANCHOR_RADIUS_SECONDS,
    MUTUAL_RECURRENCE_PEER_RADIUS_SECONDS,
    MUTUAL_RECURRENCE_DISTINCT_SECONDS,
    PHONETIC_ONSET_FEATURE_REVISION,
    PHONETIC_ONSET_MIN_POSTERIOR,
    PHONETIC_ONSET_RADIUS_SECONDS,
    PHONETIC_ONSET_MIN_STRENGTH,
    PHONETIC_ONSET_MIN_PROMINENCE,
    TAIL_RELEASE_FEATURE_REVISION,
    TAIL_RELEASE_ONSET_MIN_STRENGTH,
    TAIL_RELEASE_ONSET_MIN_PROMINENCE,
    TAIL_RELEASE_CLUSTER_BREAK_SECONDS,
    TAIL_RELEASE_CONTEXT_SECONDS,
    _project_recurrence_spans_from_geometry,
    _local_onset_candidates,
    _first_post_tail_cluster_onset,
    _self_bound_hypothesis_record,
    _bound_local_occurrence_identity,
    _candidate_seed_rows,
    _best_mutual_recurrence_pair,
    _trusted_bounds_for_entry,
    apply_phonetic_onset_fusion_hypotheses,
    apply_mutual_acoustic_recurrence_hypotheses,
    apply_trusted_tail_release_onset_hypotheses,
)
from hypothesis_provenance import (
    RECURRENCE_FEATURE_REVISION,
    RECURRENCE_MIN_SIMILARITY,
    RECURRENCE_MIN_DISTINCT_MARGIN,
    vocal_onset_confirmation_provenance_is_valid,
    ctc_opening_vocal_fusion_is_valid,
    vocal_activity_reentry_provenance_is_valid,
    phonetic_onset_fusion_provenance_is_valid,
    mutual_recurrence_hypothesis_provenance_is_valid,
    trusted_tail_release_provenance_is_valid,
    _trusted_recurrence_reference_geometry,
    apply_global_acoustic_recurrence_hypotheses,
    recurrence_hypothesis_provenance_is_valid,
)


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
