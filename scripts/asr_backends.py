"""Extracted from auto_lrc.py (split step 8): whisper.cpp, whisperx and CTC runners, multimodel consensus, segment text timing, capability builders."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Mapping
from ctc_confidence import ctc_alignment_confidence
import runtime_paths
import file_digest
import model_cache
from model_cache import ModelCacheMiss
from multimodel_consensus import PROTOCOL_REVISION as MULTIMODEL_PROTOCOL_REVISION, active_model_specs, build_windows, choose_consensus_overrides, consensus_report, model_disagreement_review, model_identity
from acoustic_hypotheses import apply_whisperx_lyric_refinement, blend_tail_repair_with_forced, has_repeated_asr_tail, repair_intro_hallucination, repair_long_segment_hallucinations, repair_tail_hallucination, sync_scratch_assignment_timestamps
from ctc_refinement import _BOUNDED_RETRY_MODEL_IDENTITY, annotate_ctc_boundary_evidence, apply_ctc_acoustic_backtrack
from report_trust import _RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY, _backend_participation, add_alignment_diagnostics, candidate_selection_quality, candidate_summary, report_quality, update_report_confidence_metrics
from timing_core import BoundedRetryCapability, CTC_SEVERE_INTERNAL_GAP_SECONDS, FinalTimingState, LocalWhisperCapability, RAW_ORDER_ANCHOR_SOURCE, RecoveryProvenance, TRUSTED_ALIGNMENT_SCORE, TimingCandidate, WhisperXRuntimeProvenance, _RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY, _WHISPERX_HELPER_PROTOCOL_REVISION, _occurrence_binding_has_strong_context, normalize_timing_candidate
from lyrics_io import AsrSegment, LyricEntry, compact_latin, entry_sung_text, first_forced_char_time, is_instrumental_marker, japanese_romaji, normalize_match_text, ratio_percent, sequence_ratio
from audio_features import HOP_SECONDS, SAMPLE_RATE, prepare_vocal_ctc_audio, vocal_cache_identity
from runtime_base import AlignmentEvidenceRejected, LrcError, WhisperXEvidenceBundleError, _bounded_retry_protocol_revision, _canonical_json_digest, _ctc_model_blob_sha256, _managed_temp_dir, _model_cache_key, _model_cache_only, _model_cache_path, _model_cache_read, _model_cache_write, _path_sha256, _remove_stale_output, _run_model_helper_command, _stable_file_sha256, _timing_content_digest, canonicalize_lrc_time, default_ctc_python, default_whisper_cli, default_whisper_model, default_whisperx_python, gpu_compute_phase, progress_log, run_command, whispercpp_gpu_lock
from runtime_base import UnboundNamespace, bind_late_callbacks as _bind_late_callbacks
import lrc_stage_timings


_LATE_BOUND_CALLBACKS = (
    "_AUDIO_API",
)


# Back-edges into later modules: the owner binds the real objects once they exist.
_AUDIO_API = UnboundNamespace("_AUDIO_API")


def bind_late_callbacks(**callbacks):
    _bind_late_callbacks(globals(), _LATE_BOUND_CALLBACKS, callbacks)


WHISPERCPP_LOCAL_BATCH_SIZE = 48


_RUNTIME_WHISPERX_ALIGNMENT_METADATA_KEY = "_runtime_whisperx_alignment_provenance"


_LOCAL_WHISPER_PROTOCOL_REVISION = "whispercpp-large-v3+whisperx-bounded-local-v5-batch-phase-scoped"


def _apply_multimodel_consensus_runtime(
    state: FinalTimingState,
    candidate_sets: tuple[tuple[TimingCandidate, ...], ...],
    report: dict[str, object],
    entries: list[LyricEntry],
    args: argparse.Namespace,
) -> FinalTimingState:
    """Apply only gated consensus rewrites to Central's provisional rows."""
    diagnostic: dict[str, object] = {
        "schema": 1,
        "protocol_revision": MULTIMODEL_PROTOCOL_REVISION,
        "source": "multi-model-consensus",
        "status": "skipped",
    }
    report["multimodel_consensus"] = diagnostic
    if str(getattr(args, "multimodel_consensus", "on")) == "off":
        diagnostic["reason"] = "disabled"
        return state
    if str(getattr(args, "timing_source_policy", "default")) != "ctc-first":
        diagnostic["reason"] = "timing-source-policy-not-ctc-first"
        return state
    if report.get("ctc_audio_source") != "vocal-stem":
        diagnostic["reason"] = "no-vocal-stem"
        return state
    stem_value = report.get("ctc_audio_path")
    if not isinstance(stem_value, str) or not Path(stem_value).is_file():
        diagnostic["reason"] = "vocal-stem-path-unavailable"
        return state
    stem = Path(stem_value).resolve()
    stem_sha = report.get("vocal_stem_sha256")
    if not isinstance(stem_sha, str) or not stem_sha:
        stem_sha = _path_sha256(stem)

    written_times = tuple(decision.written_time.seconds for decision in state.decisions)
    statuses = tuple(decision.status for decision in state.decisions)
    texts = tuple(entry_sung_text(entry) for entry in entries)
    windows = build_windows(written_times, statuses, texts)
    if not windows:
        diagnostic["reason"] = "no-provisional-rows"
        diagnostic["windows"] = []
        return state

    device = str(getattr(args, "whisperx_device", "auto"))
    model_specs = list(active_model_specs())
    cache_request = {
        "protocol_revision": MULTIMODEL_PROTOCOL_REVISION,
        "models": model_specs,
        "device": device,
        "windows": list(windows),
    }
    model_ids = {spec: model_identity(spec) for spec in model_specs}
    diagnostic["model_identities"] = model_ids
    unverified_models = [
        spec for spec, identity in model_ids.items()
        if identity.get("model_revision") == "UNVERIFIED"
        or identity.get("model_blob_sha256") == "UNVERIFIED"
    ]
    if unverified_models:
        diagnostic["status"] = "skipped"
        diagnostic["reason"] = "model-identity-unverified"
        diagnostic["unverified_models"] = unverified_models
        return state
    model_revision = _timing_content_digest({
        spec: identity["model_revision"] for spec, identity in model_ids.items()
    })
    model_blob_sha = _timing_content_digest({
        spec: identity["model_blob_sha256"] for spec, identity in model_ids.items()
    })
    cache_identity = model_cache.identity_multimodel_consensus(
        model_revision=model_revision,
        model_blob_sha256=model_blob_sha,
        protocol_revision=MULTIMODEL_PROTOCOL_REVISION,
        models=model_specs,
    )
    cache_key = _model_cache_key(
        "multimodel-consensus",
        str(stem_sha),
        model_revision,
        model_blob_sha,
        MULTIMODEL_PROTOCOL_REVISION,
        cache_request,
    )
    diagnostic["cache_key"] = cache_key
    diagnostic["cache_path"] = str(_model_cache_path("multimodel-consensus", cache_key))
    diagnostic["windows"] = list(windows)
    diagnostic["models"] = model_specs

    try:
        payload = _model_cache_read(
            "multimodel-consensus", cache_key, cache_identity, args
        )
        if payload is not None:
            cache_status = "cache-hit"
        else:
            helper = Path(__file__).with_name("multimodel_align.py")
            request = {**cache_request, "stem": str(stem)}
            proc = subprocess.run(
                [str(default_ctc_python()), str(helper)],
                input=json.dumps(request, ensure_ascii=False, sort_keys=True),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(900.0, float(getattr(args, "crossline_retry_timeout", 180.0))),
                check=False,
            )
            if proc.returncode != 0:
                raise LrcError(
                    f"multimodel helper failed ({proc.returncode}): "
                    f"{proc.stderr[-1200:]}"
                )
            payload = json.loads(proc.stdout)
            if not isinstance(payload, dict):
                raise LrcError("multimodel helper returned a non-object")
            if payload.get("protocol_revision") != MULTIMODEL_PROTOCOL_REVISION:
                raise LrcError("multimodel helper protocol revision mismatch")
            _model_cache_write(
                "multimodel-consensus", cache_key, cache_identity, payload, args
            )
            cache_status = "cache-filled"
    except ModelCacheMiss:
        raise
    except (LrcError, OSError, ValueError, TypeError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        diagnostic["status"] = "error"
        diagnostic["error"] = f"{type(exc).__name__}: {exc}"
        return state

    if not isinstance(payload, dict) or not isinstance(payload.get("windows"), dict):
        diagnostic["status"] = "error"
        diagnostic["error"] = "helper-payload-missing-windows"
        return state
    model_outputs: dict[int, Mapping[str, object]] = {}
    for raw_index, values in payload["windows"].items():
        if isinstance(values, dict):
            try:
                model_outputs[int(raw_index)] = values
            except (TypeError, ValueError):
                continue
    review_only = model_disagreement_review(
        written_times,
        statuses,
        model_outputs,
        windows,
    )
    diagnostic["review_only_model_disagreements"] = list(review_only.values())
    assignments = report.get("assignments")
    if isinstance(assignments, list):
        for index, evidence in review_only.items():
            if not (0 <= index < len(assignments)) or not isinstance(assignments[index], dict):
                continue
            assignment = assignments[index]
            flags = assignment.get("flags")
            assignment["flags"] = list(dict.fromkeys([
                *(flags if isinstance(flags, list) else []),
                "model-disagreement",
            ]))
            assignment["model_disagreement"] = evidence
    raw_anchors = tuple(
        (index, candidate.written_time.seconds)
        for index, candidates in enumerate(candidate_sets)
        for candidate in candidates
        if candidate.source == RAW_ORDER_ANCHOR_SOURCE
    )
    overrides = choose_consensus_overrides(
        written_times,
        statuses,
        model_outputs,
        raw_anchors,
    )
    diagnostic.update({
        "status": "no-override" if not overrides else "applied",
        "cache_status": cache_status,
        "errors": payload.get("errors", {}),
        "model_outputs": payload["windows"],
        "overrides": consensus_report(overrides),
    })
    if not overrides:
        return state

    decisions = list(state.decisions)
    for index, override in overrides.items():
        decision = decisions[index]
        if decision.status != "provisional_unresolved":
            continue
        old_candidate = decision.candidate
        written_time = canonicalize_lrc_time(override.seconds)
        candidate_id = _timing_content_digest({
            "source": "multi-model-consensus",
            "entry_index": index,
            "old_candidate_id": old_candidate.candidate_id,
            "seconds": written_time.seconds,
            "model_times": list(override.model_times),
            "support_families": list(override.support_families),
        })
        candidate = replace(
            old_candidate,
            candidate_id=candidate_id,
            source="multi-model-consensus",
            raw_time=override.seconds,
            written_time=written_time,
            generation_revision=_timing_content_digest({
                "candidate_id": candidate_id,
                "protocol_revision": MULTIMODEL_PROTOCOL_REVISION,
                "guards": list(override.guards),
            }),
            spans=(),
            current=False,
        )
        previous = decisions[index - 1].candidate if index else None
        evidence = normalize_timing_candidate(candidate, previous)
        new_selection_revision = decision.selection_revision + 1
        recovery = RecoveryProvenance(
            candidate_id=candidate_id,
            candidate_revision=evidence.candidate_revision,
            evidence_revision=decision.timing_evidence_revision + 1,
            canonical_written_centiseconds=written_time.centiseconds,
            old_selection_revision=decision.selection_revision,
            new_selection_revision=new_selection_revision,
            resolved_finding_ids=(),
            passed_invariants=tuple(
                [name for name, passed in override.guards if passed]
                + ["G6_provisional_unresolved"]
            ),
            resolution_mode="challenger-recovery",
            evaluator_version="multi-model-consensus-v1",
        )
        decisions[index] = replace(
            decision,
            selection_revision=new_selection_revision,
            provisional_candidate_id=candidate_id,
            written_time=written_time,
            timestamp_revision=decision.timestamp_revision + 1,
            timing_evidence_revision=decision.timing_evidence_revision + 1,
            evidence=evidence,
            candidate=candidate,
            recovery=recovery,
            replaced_candidate=old_candidate,
        )
    return FinalTimingState(
        decisions=tuple(decisions),
        written_times=tuple(decision.written_time for decision in decisions),
        audits=state.audits,
    )


def _hf_hub_fingerprint() -> list[list[object]]:
    """Cheap stat summary of every hub repo's snapshots and refs.

    Downloading or updating any cached model adds a snapshot or rewrites a
    ref, which changes this summary and so retires remembered model lookups.
    """
    rows: list[list[object]] = []
    for root in runtime_paths.hf_hub_cache_candidates():
        try:
            repos = sorted(item for item in root.iterdir() if item.name.startswith("models--"))
        except OSError:
            continue
        for repo in repos:
            row: list[object] = [str(repo)]
            for child in (repo / "snapshots", repo / "refs" / "main"):
                try:
                    stat = child.stat()
                    row.extend((stat.st_mtime_ns, stat.st_size))
                except OSError:
                    row.extend((None, None))
            rows.append(row)
    return rows


@lru_cache(maxsize=8)
def _whisperx_model_identity(python_exe: Path, helper: Path, language: str) -> tuple[str, str]:
    """Resolve cached WhisperX alignment weights without loading the model.

    The probe imports WhisperX in a fresh interpreter, which costs several
    seconds per process, so a resolved answer is remembered on disk for as
    long as the interpreter, helper, language and hub cache are unchanged.
    """
    try:
        python_stat = python_exe.resolve().stat()
        memo_key = _timing_content_digest({
            "python": [str(python_exe.resolve()), python_stat.st_size, python_stat.st_mtime_ns],
            "helper_sha256": _path_sha256(helper),
            "language": str(language),
            "hub": _hf_hub_fingerprint(),
            "env": [os.environ.get(name, "") for name in ("HF_HUB_CACHE", "HF_HOME", "HF_HUB_OFFLINE")],
        })
    except OSError:
        memo_key = None
    if memo_key is not None:
        stored = file_digest.memo_load("whisperx-align-model.json").get(memo_key)
        if (
            isinstance(stored, list)
            and len(stored) == 2
            and all(isinstance(value, str) and value and value != "unresolved" for value in stored)
        ):
            return str(stored[0]), str(stored[1])
    result = _probe_whisperx_model_identity(python_exe, helper, language)
    if memo_key is not None and "unresolved" not in result:
        file_digest.memo_store("whisperx-align-model.json", memo_key, list(result))
    return result


def _probe_whisperx_model_identity(python_exe: Path, helper: Path, language: str) -> tuple[str, str]:
    probe = (
        "import json; from whisperx_refine import model_provenance; "
        "print(json.dumps(model_provenance(" + repr(str(language)) + ") or {}))"
    )
    try:
        proc = subprocess.run(
            [str(python_exe), "-c", probe], cwd=str(helper.parent), check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", timeout=30.0,
        )
        value = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 else {}
        return (
            str(value.get("align_model_revision") or "unresolved"),
            str(value.get("align_model_blob_sha256") or "unresolved"),
        )
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError, IndexError):
        return "unresolved", "unresolved"


def lyric_segment_score(entry_text: str, segment_text: str) -> float:
    if not entry_text or not segment_text:
        return 0.0
    score = sequence_ratio(entry_text, segment_text)
    if entry_text in segment_text:
        score = max(score, 0.96)
    elif segment_text in entry_text and len(segment_text) >= 4:
        score = max(score, 0.72)
    return score


def suffix_char_time(entry_text: str, segment: AsrSegment) -> float | None:
    if not entry_text or not segment.chars:
        return None
    segment_text = "".join(char for char, _ in segment.chars)
    if not segment_text.endswith(entry_text):
        return None

    import difflib

    matcher = difflib.SequenceMatcher(None, entry_text, segment_text, autojunk=False)
    if matcher.ratio() < 0.75:
        return None
    blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    if not blocks or blocks[0].b <= 0:
        return None
    char_time = segment.chars[blocks[0].b][1]
    if not (segment.start - 0.10 <= char_time <= segment.end + 0.10):
        return None
    if char_time - segment.start > 1.0:
        return None
    return char_time


def segment_char_time(entry_text: str, segment: AsrSegment) -> float | None:
    if not entry_text or not segment.chars:
        return None
    segment_text = "".join(char for char, _ in segment.chars)
    exact_index = segment_text.find(entry_text)
    if exact_index >= 0:
        char_time = segment.chars[exact_index][1]
        if segment.start - 0.10 <= char_time <= segment.end + 0.10:
            return char_time
        return None
    return suffix_char_time(entry_text, segment)


def _aligned_exact_target_matches(
    entry_text: str,
    segments: list[AsrSegment],
) -> tuple[dict[str, object], ...]:
    """Return exact WhisperX character-stream matches for one immutable lyric.

    Unlike :func:`segment_char_time`, this stream may cross adjacent WhisperX
    segments.  Every returned onset is still a real forced-character timestamp;
    no interpolation or candidate-time proximity is used.  A match containing a
    severe internal time fracture is rejected rather than bridged.
    """
    target = normalize_match_text(entry_text)
    if not target:
        return ()
    stream: list[tuple[str, float, int]] = []
    for segment_index, segment in enumerate(segments):
        for char, raw_time in segment.chars:
            normalized = normalize_match_text(char)
            if not normalized:
                continue
            try:
                char_time = float(raw_time)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(char_time):
                continue
            for item in normalized:
                stream.append((item, char_time, segment_index))
    if len(stream) < len(target):
        return ()
    stream_text = "".join(item[0] for item in stream)
    matches: list[dict[str, object]] = []
    search_at = 0
    while True:
        start = stream_text.find(target, search_at)
        if start < 0:
            break
        end = start + len(target)
        observed = stream[start:end]
        times = [item[1] for item in observed]
        segment_indexes = [item[2] for item in observed]
        search_at = start + 1
        if (
            any(left > right for left, right in zip(times, times[1:]))
            or any(
                right - left >= CTC_SEVERE_INTERNAL_GAP_SECONDS
                for left, right in zip(times, times[1:])
            )
            or any(left > right for left, right in zip(segment_indexes, segment_indexes[1:]))
        ):
            continue
        payload = {
            "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "start_char_index": start,
            "end_char_index": end,
            "onset_time": round(times[0], 6),
            "last_char_time": round(times[-1], 6),
            "start_segment_index": segment_indexes[0],
            "end_segment_index": segment_indexes[-1],
            "char_count": len(observed),
        }
        matches.append({**payload, "match_revision": _timing_content_digest(payload)})
    return tuple(matches)


def _occurrence_binding_time_bounds(
    binding: object,
    entry_index: int,
) -> tuple[float | None, float | None]:
    """Return proven left/right timing bounds around a context-bound target."""
    if not isinstance(binding, dict):
        return None, None
    mode = binding.get("mode")
    if mode == "sequence-bracket":
        try:
            return float(binding.get("left_timestamp")), float(binding.get("right_timestamp"))
        except (TypeError, ValueError):
            return None, None
    if mode != "sparse-sequence-bracket":
        return None, None
    indexes = binding.get("anchor_entry_indexes")
    timestamps = binding.get("anchor_timestamps")
    if not isinstance(indexes, list) or not isinstance(timestamps, list) or len(indexes) != len(timestamps):
        return None, None
    left_values: list[tuple[int, float]] = []
    right_values: list[tuple[int, float]] = []
    for raw_index, raw_time in zip(indexes, timestamps):
        try:
            anchor_index = int(raw_index)
            anchor_time = float(raw_time)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(anchor_time):
            continue
        if anchor_index < entry_index:
            left_values.append((anchor_index, anchor_time))
        elif anchor_index > entry_index:
            right_values.append((anchor_index, anchor_time))
    left_time = max(left_values, key=lambda item: item[0])[1] if left_values else None
    right_time = min(right_values, key=lambda item: item[0])[1] if right_values else None
    return left_time, right_time


def _select_unique_aligned_target_match(
    entry_text: str,
    segments: list[AsrSegment],
    *,
    occurrence_binding: object = None,
    entry_index: int,
    preferred_segment_number: int | None = None,
) -> dict[str, object] | None:
    """Select an exact char-stream match only when evidence makes it unique.

    Context timing bounds and an already observed target segment are filters, not
    nearest-neighbour selectors.  If more than one exact match survives, timing
    evidence remains unavailable.
    """
    matches = list(_aligned_exact_target_matches(entry_text, segments))
    if not matches:
        return None
    left_time, right_time = _occurrence_binding_time_bounds(occurrence_binding, entry_index)
    if left_time is not None:
        matches = [item for item in matches if float(item["onset_time"]) > left_time - HOP_SECONDS]
    if right_time is not None:
        matches = [item for item in matches if float(item["onset_time"]) < right_time + HOP_SECONDS]
    if preferred_segment_number is not None and preferred_segment_number >= 1:
        preferred_index = preferred_segment_number - 1
        preferred = [
            item for item in matches
            if int(item["start_segment_index"]) <= preferred_index <= int(item["end_segment_index"])
        ]
        if len(preferred) == 1:
            return preferred[0]
        if preferred:
            matches = preferred
    return matches[0] if len(matches) == 1 else None


def _aligned_target_prefix_matches(
    entry_text: str,
    segments: list[AsrSegment],
) -> tuple[dict[str, object], ...]:
    """Return forced-character matches for a strong immutable lyric prefix.

    This is a target-opening proof, not a replacement for occurrence identity.
    It is consumed only after a full/sparse sequence binding has independently
    established which lyric occurrence is in the local window.  For long lines
    the first six normalized characters are required exactly; shorter lines
    require the entire normalized text.  Every onset is a real WhisperX forced
    character timestamp and internally fractured matches are rejected.
    """
    target = normalize_match_text(entry_text)
    if not target:
        return ()
    prefix_length = min(len(target), 6)
    if prefix_length < 2:
        return ()
    prefix = target[:prefix_length]
    stream: list[tuple[str, float, int]] = []
    for segment_index, segment in enumerate(segments):
        for char, raw_time in segment.chars:
            normalized = normalize_match_text(char)
            if not normalized:
                continue
            try:
                char_time = float(raw_time)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(char_time):
                continue
            for item in normalized:
                stream.append((item, char_time, segment_index))
    if len(stream) < prefix_length:
        return ()
    stream_text = "".join(item[0] for item in stream)
    matches: list[dict[str, object]] = []
    search_at = 0
    while True:
        start = stream_text.find(prefix, search_at)
        if start < 0:
            break
        end = start + prefix_length
        observed = stream[start:end]
        times = [item[1] for item in observed]
        segment_indexes = [item[2] for item in observed]
        search_at = start + 1
        if (
            any(left > right for left, right in zip(times, times[1:]))
            or any(
                right - left >= CTC_SEVERE_INTERNAL_GAP_SECONDS
                for left, right in zip(times, times[1:])
            )
            or any(left > right for left, right in zip(segment_indexes, segment_indexes[1:]))
        ):
            continue
        payload = {
            "target_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "prefix_sha256": hashlib.sha256(prefix.encode("utf-8")).hexdigest(),
            "prefix_char_count": prefix_length,
            "start_char_index": start,
            "end_char_index": end,
            "onset_time": round(times[0], 6),
            "last_char_time": round(times[-1], 6),
            "start_segment_index": segment_indexes[0],
            "end_segment_index": segment_indexes[-1],
        }
        matches.append({**payload, "match_revision": _timing_content_digest(payload)})
    return tuple(matches)


def _aligned_exact_phonetic_target_matches(
    entry_text: str,
    segments: list[AsrSegment],
) -> tuple[dict[str, object], ...]:
    """Return exact phonetic identity matches across aligned segment text.

    This is *content identity only*.  Whisper may spell the same Japanese sound
    with kana/kanji variants that defeat exact Unicode character matching even
    though the recognized phonetic sequence is exact.  We therefore compare a
    compact romanized stream, but never convert a segment start into direct-onset
    evidence.  Any later timing proof still needs real forced-character or
    independent acoustic onset evidence.
    """
    target = compact_latin(japanese_romaji(entry_text) or entry_text)
    if len(target) < 4:
        return ()

    stream: list[tuple[str, int]] = []
    segment_phonetics: list[str] = []
    for segment_index, segment in enumerate(segments):
        phonetic = compact_latin(japanese_romaji(segment.text) or segment.text)
        segment_phonetics.append(phonetic)
        for item in phonetic:
            stream.append((item, segment_index))
    if len(stream) < len(target):
        return ()

    stream_text = "".join(item[0] for item in stream)
    matches: list[dict[str, object]] = []
    search_at = 0
    while True:
        start = stream_text.find(target, search_at)
        if start < 0:
            break
        end = start + len(target)
        observed = stream[start:end]
        search_at = start + 1
        segment_indexes = [item[1] for item in observed]
        if any(left > right for left, right in zip(segment_indexes, segment_indexes[1:])):
            continue
        start_segment = segment_indexes[0]
        end_segment = segment_indexes[-1]
        fractured = False
        for left_index in range(start_segment, end_segment):
            try:
                gap = float(segments[left_index + 1].start) - float(segments[left_index].end)
            except (TypeError, ValueError, IndexError):
                fractured = True
                break
            if not math.isfinite(gap) or gap >= CTC_SEVERE_INTERNAL_GAP_SECONDS:
                fractured = True
                break
        if fractured:
            continue
        payload = {
            "target_phonetic_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "start_phonetic_index": start,
            "end_phonetic_index": end,
            "start_segment_index": start_segment,
            "end_segment_index": end_segment,
            "phonetic_char_count": len(target),
            "phonetic_score": round(lyric_segment_score(target, target), 6),
        }
        matches.append({**payload, "match_revision": _timing_content_digest(payload)})
    return tuple(matches)


def _select_unique_aligned_target_prefix_match(
    entry_text: str,
    segments: list[AsrSegment],
    *,
    occurrence_binding: object,
    entry_index: int,
    preferred_segment_number: int | None = None,
) -> dict[str, object] | None:
    """Select one context-bound forced prefix onset without nearest-time voting."""
    if not _occurrence_binding_has_strong_context(occurrence_binding):
        return None
    matches = list(_aligned_target_prefix_matches(entry_text, segments))
    if not matches:
        return None
    left_time, right_time = _occurrence_binding_time_bounds(occurrence_binding, entry_index)
    if left_time is not None:
        matches = [item for item in matches if float(item["onset_time"]) > left_time - HOP_SECONDS]
    if right_time is not None:
        matches = [item for item in matches if float(item["onset_time"]) < right_time + HOP_SECONDS]
    if preferred_segment_number is not None and preferred_segment_number >= 1:
        preferred_index = preferred_segment_number - 1
        preferred = [
            item for item in matches
            if int(item["start_segment_index"]) <= preferred_index <= int(item["end_segment_index"])
        ]
        if len(preferred) == 1:
            return preferred[0]
        if preferred:
            matches = preferred
    return matches[0] if len(matches) == 1 else None


def estimated_segment_text_time(entry_text: str, raw_entry_text: str, segment: AsrSegment) -> float | None:
    if not entry_text:
        return None
    segment_text = normalize_match_text(segment.text)
    if not segment_text:
        return None

    match_index = segment_text.find(entry_text)
    match_basis_length = len(segment_text)
    if match_index < 0:
        raw_entry_romaji = japanese_romaji(raw_entry_text) or compact_latin(raw_entry_text)
        segment_romaji = japanese_romaji(segment.text) or compact_latin(segment.text)
        if not raw_entry_romaji or not segment_romaji:
            return None
        for prefix_length in range(min(len(raw_entry_romaji), 18), 4, -1):
            prefix = raw_entry_romaji[:prefix_length]
            match_index = segment_romaji.find(prefix)
            if match_index >= 0:
                match_basis_length = len(segment_romaji)
                break
        else:
            return None

    if match_index <= 0:
        return segment.start
    segment_duration = max(0.0, segment.end - segment.start)
    if segment_duration <= 0.05 or match_basis_length <= 0:
        return None
    fraction = min(0.95, max(0.0, match_index / match_basis_length))
    estimated = segment.start + segment_duration * fraction
    if segment.start - 0.10 <= estimated <= segment.end + 0.10:
        return estimated
    return None


def decode_temp_wav(audio_path: Path) -> Path:
    temp_dir = _managed_temp_dir(prefix="lrc_tools_whispercpp_")
    wav_path = temp_dir / "input.wav"
    proc = subprocess.run(
        [
            runtime_paths.ffmpeg(),
            "-y",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise LrcError(f"ffmpeg temp WAV decode failed: {proc.stderr.strip()}")
    return wav_path


def _whispercpp_cache_identity(
    wav_path: Path,
    args: argparse.Namespace,
    *,
    suppress_nst: bool,
    cli: Path,
    model: Path,
) -> tuple[str, dict[str, object]]:
    cli_sha = _stable_file_sha256(cli)
    model_sha = _stable_file_sha256(model)
    try:
        stem_sha = _stable_file_sha256(wav_path)
    except OSError:
        # Unit callers may replace the subprocess before creating a real WAV;
        # production paths always decode a content-addressed file first.
        stem_sha = _timing_content_digest({"missing_wav_path": str(wav_path.resolve())})
    identity = model_cache.identity_raw_asr(
        stem_sha256=stem_sha,
        cli_sha256=cli_sha,
        model_sha256=model_sha,
        language=str(args.whisper_language),
        suppress_nst=bool(suppress_nst),
    )
    key = model_cache.key_from_identity("raw-asr", identity)
    return key, identity


def _whispercpp_cache_lookup(
    key: str, identity: dict[str, object], args: argparse.Namespace
) -> list[AsrSegment] | None:
    cached = _model_cache_read("raw-asr", key, identity, args)
    if isinstance(cached, list):
        try:
            return [AsrSegment(float(item["start"]), float(item["end"]), str(item["text"])) for item in cached]
        except (KeyError, TypeError, ValueError):
            if _model_cache_only(args):
                raise ModelCacheMiss(f"model-cache-invalid:raw-asr:{key}")
    return None


def _whispercpp_cache_store(
    key: str, identity: dict[str, object], segments: list[AsrSegment], args: argparse.Namespace
) -> None:
    _model_cache_write(
        "raw-asr", key, identity,
        [{"start": item.start, "end": item.end, "text": item.text} for item in segments], args,
    )


def _run_whispercpp_wav(
    wav_path: Path,
    args: argparse.Namespace,
    *,
    suppress_nst: bool,
) -> list[AsrSegment]:
    """Run whisper.cpp on an already-decoded local WAV without changing its time origin."""
    cli_arg = getattr(args, "whisper_cli", None)
    cli = Path(cli_arg).expanduser().resolve() if cli_arg else default_whisper_cli()
    model_arg = getattr(args, "whisper_model", None)
    model = Path(model_arg).expanduser().resolve() if model_arg else default_whisper_model()
    if not cli.exists():
        raise LrcError(f"whisper.cpp CLI not found: {cli}")
    if not model.exists():
        raise LrcError(f"whisper.cpp model not found: {model}")

    key, identity = _whispercpp_cache_identity(
        wav_path, args, suppress_nst=suppress_nst, cli=cli, model=model
    )
    cached_segments = _whispercpp_cache_lookup(key, identity, args)
    if cached_segments is not None:
        return cached_segments

    output_base = wav_path.with_suffix("")
    command = [
        str(cli),
        "-m", str(model),
        "-l", args.whisper_language,
        "-oj", "-ojf",
        "-of", str(output_base),
    ]
    if suppress_nst:
        command.append("-sns")
    command.append(str(wav_path))
    json_path = output_base.with_suffix(".json")
    _remove_stale_output(json_path)
    with whispercpp_gpu_lock():
        proc = run_command(command)
    if proc.returncode != 0:
        raise LrcError(f"whisper.cpp failed: {proc.stderr.strip() or proc.stdout.strip()}")

    if not json_path.exists():
        raise LrcError(f"whisper.cpp did not write JSON output: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    segments: list[AsrSegment] = []
    for item in payload.get("transcription", []):
        offsets = item.get("offsets") or {}
        start = float(offsets.get("from", 0)) / 1000.0
        end = float(offsets.get("to", 0)) / 1000.0
        text = str(item.get("text", "")).strip()
        if text:
            segments.append(AsrSegment(start=start, end=end, text=text))
    if not segments:
        raise LrcError("whisper.cpp produced no transcription segments")
    _whispercpp_cache_store(key, identity, segments, args)
    return segments


def _parse_whispercpp_json(json_path: Path) -> list[AsrSegment]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    segments: list[AsrSegment] = []
    for item in payload.get("transcription", []):
        offsets = item.get("offsets") or {}
        start = float(offsets.get("from", 0)) / 1000.0
        end = float(offsets.get("to", 0)) / 1000.0
        text = str(item.get("text", "")).strip()
        if text:
            segments.append(AsrSegment(start=start, end=end, text=text))
    if not segments:
        raise LrcError("whisper.cpp produced no transcription segments")
    return segments


@lru_cache(maxsize=8)
def _whispercpp_multifile_supported(cli_text: str, size: int, mtime_ns: int) -> bool:
    del size, mtime_ns
    try:
        proc = subprocess.run(
            [cli_text, "-h"], check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=15.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return "file0 file1" in help_text or "file0 file1 ..." in help_text


def _run_whispercpp_wav_batch(
    items: list[tuple[str, Path]],
    args: argparse.Namespace,
    *,
    suppress_nst: bool,
) -> tuple[dict[str, list[AsrSegment]], dict[str, str]]:
    """Transcribe many already-decoded WAVs while loading ggml weights once per chunk.

    Modern whisper-cli accepts file0 file1 ... and initializes one whisper context
    before iterating the files.  If the installed binary does not support this or
    a chunk fails, that chunk falls back to the legacy one-file path.
    """
    results: dict[str, list[AsrSegment]] = {}
    errors: dict[str, str] = {}
    if not items:
        return results, errors
    cli_arg = getattr(args, "whisper_cli", None)
    cli = Path(cli_arg).expanduser().resolve() if cli_arg else default_whisper_cli()
    model_arg = getattr(args, "whisper_model", None)
    model = Path(model_arg).expanduser().resolve() if model_arg else default_whisper_model()
    if not cli.is_file() or not model.is_file():
        message = "whisper.cpp CLI/model unavailable"
        return results, {key: message for key, _ in items}
    cache_entries: dict[str, tuple[str, dict[str, object]]] = {}
    pending: list[tuple[str, Path]] = []
    for item_key, item_wav in items:
        cache_key, cache_identity = _whispercpp_cache_identity(
            item_wav, args, suppress_nst=suppress_nst, cli=cli, model=model
        )
        cached_segments = _whispercpp_cache_lookup(cache_key, cache_identity, args)
        if cached_segments is not None:
            results[item_key] = cached_segments
        else:
            cache_entries[item_key] = (cache_key, cache_identity)
            pending.append((item_key, item_wav))
    if len(pending) != len(items):
        progress_log(
            f"local Whisper raw batch: model_cache_hits={len(items) - len(pending)} "
            f"misses={len(pending)}"
        )
    items = pending
    if not items:
        return results, errors
    try:
        stat = cli.stat()
        multi = _whispercpp_multifile_supported(str(cli), stat.st_size, stat.st_mtime_ns)
    except OSError:
        multi = False
    if not multi or len(items) == 1:
        for key, wav_path in items:
            try:
                results[key] = _run_whispercpp_wav(wav_path, args, suppress_nst=suppress_nst)
            except (LrcError, OSError, ValueError, json.JSONDecodeError) as exc:
                errors[key] = f"{type(exc).__name__}:{exc}"
        return results, errors

    chunks = [items[index:index + WHISPERCPP_LOCAL_BATCH_SIZE] for index in range(0, len(items), WHISPERCPP_LOCAL_BATCH_SIZE)]
    progress_log(
        f"local Whisper raw batch: files={len(items)} chunks={len(chunks)} "
        f"max_chunk={WHISPERCPP_LOCAL_BATCH_SIZE}"
    )
    for chunk_number, chunk in enumerate(chunks, start=1):
        command = [str(cli), "-m", str(model), "-l", args.whisper_language, "-oj", "-ojf"]
        if suppress_nst:
            command.append("-sns")
        for _key, wav_path in chunk:
            command.extend(["-of", str(wav_path.with_suffix(""))])
            _remove_stale_output(wav_path.with_suffix(".json"))
        command.extend(str(wav_path) for _key, wav_path in chunk)
        progress_log(
            f"local Whisper raw batch: chunk={chunk_number}/{len(chunks)} files={len(chunk)}"
        )
        try:
            with whispercpp_gpu_lock():
                proc = run_command(command)
        except (LrcError, OSError, subprocess.SubprocessError):
            proc = None
        chunk_ok = proc is not None and proc.returncode == 0
        if chunk_ok:
            for key, wav_path in chunk:
                json_path = wav_path.with_suffix(".json")
                if not json_path.is_file():
                    chunk_ok = False
                    break
        if not chunk_ok:
            progress_log(
                f"local Whisper raw batch: chunk={chunk_number} fallback=one-shot"
            )
            for key, wav_path in chunk:
                try:
                    results[key] = _run_whispercpp_wav(wav_path, args, suppress_nst=suppress_nst)
                except (LrcError, OSError, ValueError, json.JSONDecodeError) as exc:
                    errors[key] = f"{type(exc).__name__}:{exc}"
            continue
        for key, wav_path in chunk:
            try:
                results[key] = _parse_whispercpp_json(wav_path.with_suffix(".json"))
            except (LrcError, OSError, ValueError, json.JSONDecodeError) as exc:
                errors[key] = f"{type(exc).__name__}:{exc}"
                continue
            cache_key, cache_identity = cache_entries[key]
            _whispercpp_cache_store(cache_key, cache_identity, results[key], args)
    return results, errors


def run_whispercpp(audio_path: Path, args: argparse.Namespace, suppress_nst: bool | None = None) -> list[AsrSegment]:
    wav_path = decode_temp_wav(audio_path)
    use_suppress_nst = bool(args.whisper_suppress_nst) if suppress_nst is None else suppress_nst
    return _run_whispercpp_wav(wav_path, args, suppress_nst=bool(use_suppress_nst))


def decode_temp_wav_window(audio_path: Path, start: float, end: float) -> Path:
    """Decode an exact bounded audio window to local-time mono 16 kHz WAV."""
    if not all(math.isfinite(value) for value in (start, end)) or start < 0.0 or end <= start:
        raise LrcError("Invalid local Whisper audio window")
    temp_dir = _managed_temp_dir(prefix="lrc_tools_whisper_local_")
    wav_path = temp_dir / "input.wav"
    proc = subprocess.run(
        [
            runtime_paths.ffmpeg(), "-y", "-v", "error",
            "-i", str(audio_path),
            "-ss", f"{start:.6f}",
            "-t", f"{end - start:.6f}",
            "-ac", "1", "-ar", "16000",
            str(wav_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0 or not wav_path.exists():
        raise LrcError(f"ffmpeg local Whisper decode failed: {proc.stderr.strip()}")
    return wav_path


def run_whisperx_alignment(
    audio_path: Path,
    transcript: list[dict[str, object]],
    args: argparse.Namespace,
    *,
    predecoded_wav: bool = False,
) -> dict[str, object]:
    python_arg = getattr(args, "whisperx_python", None)
    python_exe = Path(python_arg).expanduser().resolve() if python_arg else default_whisperx_python()
    helper = Path(__file__).resolve().parent / "whisperx_refine.py"
    if not python_exe.exists():
        raise LrcError(
            f"WhisperX Python environment not found: {python_exe}. "
            "Create .venv-asr and install whisperx, or pass --whisperx-python."
        )
    if not helper.exists():
        raise LrcError(f"WhisperX helper not found: {helper}")

    wav_path = audio_path if predecoded_wav else decode_temp_wav(audio_path)
    temp_dir = wav_path.parent
    transcript_path = temp_dir / "transcript.json"
    output_path = temp_dir / "whisperx-aligned.json"
    transcript_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stem_sha = _stable_file_sha256(wav_path)
    helper_sha = _stable_file_sha256(helper)
    model_revision, model_blob = _whisperx_model_identity(
        python_exe, helper, str(args.whisper_language)
    )
    identity = model_cache.identity_whisperx_alignment(
        stem_sha256=stem_sha,
        model_revision=model_revision,
        model_blob_sha256=model_blob,
        helper_protocol_revision=_WHISPERX_HELPER_PROTOCOL_REVISION,
        helper_sha256=helper_sha,
        transcript=transcript,
        language=str(args.whisper_language),
        device=str(args.whisperx_device),
    )
    key = model_cache.key_from_identity("whisperx-alignment", identity)
    cached = _model_cache_read("whisperx-alignment", key, identity, args)
    if isinstance(cached, dict):
        return cached
    _remove_stale_output(output_path)
    proc = _run_model_helper_command(
        [
            str(python_exe),
            str(helper),
            "--audio",
            str(wav_path),
            "--transcript",
            str(transcript_path),
            "--output",
            str(output_path),
            "--language",
            args.whisper_language,
            "--device",
            args.whisperx_device,
        ],
        mode="whisperx",
    )
    if proc.returncode != 0:
        raise LrcError(f"WhisperX alignment failed: {proc.stderr.strip() or proc.stdout.strip()}")
    if not output_path.exists():
        raise LrcError(f"WhisperX helper did not write output: {output_path}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    metadata = payload.get(_RUNTIME_WHISPERX_ALIGNMENT_METADATA_KEY)
    if isinstance(metadata, dict):
        output_payload = {
            key: value
            for key, value in payload.items()
            if key != _RUNTIME_WHISPERX_ALIGNMENT_METADATA_KEY
        }
        payload[_RUNTIME_WHISPERX_ALIGNMENT_METADATA_KEY] = {
            **metadata,
            "alignment_wav_sha256": _path_sha256(wav_path),
            "input_transcript_revision": _timing_content_digest(transcript),
            "output_revision": _timing_content_digest(output_payload),
        }
    _model_cache_write("whisperx-alignment", key, identity, payload, args)
    return payload


def run_ctc_alignment(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object]]:
    alignment_audio, audio_source, audio_note = prepare_vocal_ctc_audio(audio_path, args)
    if getattr(args, "vocal_ctc", True) and audio_source != "vocal-stem":
        raise LrcError(
            "Vocal CTC was requested but Demucs did not produce an isolated vocal stem: "
            f"{audio_note or 'unknown failure'}. Use --no-vocal-ctc only to explicitly allow full-mix CTC."
        )
    python_exe = default_ctc_python()
    try:
        vocal_identity = vocal_cache_identity(audio_path, python_exe)
    except OSError:
        vocal_identity = {}
    try:
        vocal_stem_sha256 = _path_sha256(alignment_audio) if alignment_audio.exists() else None
    except OSError:
        vocal_stem_sha256 = None
    helper = Path(__file__).resolve().parent / "ctc_align.py"
    if not python_exe.exists():
        raise LrcError(f"CTC Python environment not found: {python_exe}")
    if not helper.exists():
        raise LrcError(f"CTC helper not found: {helper}")

    helper_sha = _stable_file_sha256(helper)
    model_blob = _ctc_model_blob_sha256()
    identity = model_cache.identity_whole_ctc(
        stem_sha256=vocal_stem_sha256,
        model_revision=_BOUNDED_RETRY_MODEL_IDENTITY,
        model_blob_sha256=model_blob,
        helper_protocol_revision=_bounded_retry_protocol_revision(helper_sha),
        helper_sha256=helper_sha,
        entries=[entry_sung_text(entry) for entry in entries],
        duration=duration,
        device=str(args.whisperx_device),
        source=audio_source,
    )
    key = model_cache.key_from_identity("whole-ctc", identity)
    cached = _model_cache_read("whole-ctc", key, identity, args)

    if isinstance(cached, dict):
        payload = cached
    else:
        with tempfile.TemporaryDirectory(prefix="lrc-ctc-") as temp_name:
            temp_dir = Path(temp_name)
            transcript_path = temp_dir / "transcript.json"
            output_path = temp_dir / "ctc-aligned.json"
            transcript_path.write_text(
                json.dumps([entry_sung_text(entry) for entry in entries], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            proc = _run_model_helper_command(
            [
                str(python_exe),
                str(helper),
                "--audio",
                str(alignment_audio),
                "--transcript",
                str(transcript_path),
                "--output",
                str(output_path),
                "--device",
                args.whisperx_device,
            ],
            mode="ctc",
        )
            if proc.returncode != 0:
                raise LrcError(f"CTC forced alignment failed: {proc.stderr.strip() or proc.stdout.strip()}")
            if not output_path.exists():
                raise LrcError(f"CTC helper did not write output: {output_path}")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        _model_cache_write("whole-ctc", key, identity, payload, args)

    rows = payload.get("entries")
    if not isinstance(rows, list) or len(rows) != len(entries):
        raise LrcError("CTC helper returned an invalid entry list")
    if any(not isinstance(row, dict) for row in rows):
        raise LrcError("CTC helper returned a non-object entry")
    timestamps: list[float] = []
    assignments: list[dict[str, object]] = []
    missing: list[int] = []
    ctc_scores: list[float] = []
    low_score_entries: list[dict[str, object]] = []
    very_low_score_entries: list[dict[str, object]] = []
    low_score_entry_numbers: set[int] = set()
    for index, row in enumerate(rows):
        start = row.get("start")
        if start is None:
            missing.append(index + 1)
            timestamp = timestamps[-1] + 0.10 if timestamps else 0.0
        else:
            timestamp = float(start)
        timestamps.append(max(0.0, min(duration, timestamp)))
        raw_ctc_score = row.get("ctc_score")
        try:
            ctc_score = float(raw_ctc_score)
        except (TypeError, ValueError):
            ctc_score = 0.0
        if start is not None:
            ctc_scores.append(ctc_score)
            if ctc_score < 0.03:
                low_score_entry_numbers.add(index + 1)
                low_score_entries.append(
                    {
                        "entry": index + 1,
                        "text": entry_sung_text(entries[index]),
                        "ctc_score": round(ctc_score, 6),
                    }
                )
            if ctc_score < 0.01:
                very_low_score_entries.append(
                    {
                        "entry": index + 1,
                        "text": entry_sung_text(entries[index]),
                        "ctc_score": round(ctc_score, 6),
                    }
                )
        # A forced path always has a position, but a weak path is only a draft
        # candidate. Do not let its existence become a fake "trusted" result.
        confidence = ctc_alignment_confidence(ctc_score)
        assignments.append(
            {
                "entry": index + 1,
                "segment": None,
                "score": confidence if start is not None else 0.0,
                "ctc_score": round(ctc_score, 6),
                "romaji": row.get("romaji"),
                "ctc_token_spans": row.get("token_spans"),
                "ctc_first_token_candidates": row.get("first_token_candidates"),
                "timestamp": round(timestamps[-1], 3),
                "timing_repair": "ctc-forced-align",
                "timing_repair_source": "torchaudio-mms-fa",
            }
        )

    for index in range(1, len(timestamps)):
        timestamps[index] = max(timestamps[index], timestamps[index - 1] + 0.10)
        assignments[index]["timestamp"] = round(timestamps[index], 3)
    annotate_ctc_boundary_evidence(assignments)

    low_score_runs: list[list[int]] = []
    run: list[int] = []
    for entry_number in range(1, len(entries) + 1):
        if entry_number in low_score_entry_numbers:
            run.append(entry_number)
        else:
            if len(run) >= 4:
                low_score_runs.append(run)
            run = []
    if len(run) >= 4:
        low_score_runs.append(run)

    collapsed_entries = {entry_number for run in low_score_runs for entry_number in run}
    # A weak individual CTC path is still recorded as low confidence, but it is
    # not evidence that its timestamp is wrong by itself. Escalate only missing
    # alignments or a consecutive low-score collapse.
    ctc_review_entries = set(missing) | collapsed_entries
    suspicious_alignments: list[dict[str, object]] = []
    for entry_number in sorted(ctc_review_entries):
        entry_index = entry_number - 1
        is_missing = entry_number in missing
        is_collapsed = entry_number in collapsed_entries
        flags: list[str] = []
        if is_missing:
            flags.append("ctc_missing_alignment")
        if entry_number in low_score_entry_numbers:
            flags.append("ctc_low_path_confidence")
        if is_collapsed:
            flags.append("ctc_confidence_collapse")
        suspicious_alignments.append(
            {
                "entry": entry_number,
                "text": entry_sung_text(entries[entry_index]),
                "flags": flags,
                "severity": "high" if is_missing or is_collapsed else "medium",
                "review_required": True,
            }
        )

    report: dict[str, object] = {
        "backend": "ctc",
        "strategy": "ctc-forced-align",
        "ctc_backend": payload.get("backend", "mms_ctc"),
        "ctc_device": payload.get("device"),
        "ctc_sample_rate": payload.get("sample_rate"),
        "ctc_emission_frames": payload.get("emission_frames"),
        "ctc_audio_source": audio_source,
        "ctc_audio_path": str(alignment_audio),
        "ctc_audio_note": audio_note,
        **vocal_identity,
        "vocal_stem_sha256": vocal_stem_sha256,
        "timing_entries": len(entries),
        "assignments": assignments,
        "ctc_missing_entries": missing,
        "ctc_missing_count": len(missing),
        "ctc_score_min": round(min(ctc_scores), 6) if ctc_scores else None,
        "ctc_score_mean": round(sum(ctc_scores) / len(ctc_scores), 6) if ctc_scores else None,
        "ctc_low_score_threshold": 0.03,
        "ctc_very_low_score_threshold": 0.01,
        "ctc_low_score_count": len(low_score_entries),
        "ctc_very_low_score_count": len(very_low_score_entries),
        "ctc_low_score_entries": low_score_entries,
        "ctc_very_low_score_entries": very_low_score_entries,
        "ctc_low_score_runs": low_score_runs,
        "ctc_confidence_collapse": bool(low_score_runs),
        "suspicious_alignment_count": len(suspicious_alignments),
        "review_required_count": len(ctc_review_entries),
        "suspicious_alignment_severity_counts": {
            "high": sum(1 for item in suspicious_alignments if item["severity"] == "high"),
            "medium": sum(1 for item in suspicious_alignments if item["severity"] == "medium"),
            "low": 0,
        },
        "suspicious_alignments": suspicious_alignments,
        "review_required": bool(ctc_review_entries),
        "note": "MMS/CTC forced alignment over the known lyric order.",
    }
    participation = _backend_participation(report)
    participation["vocal_source"] = audio_source
    if audio_source == "vocal-stem":
        participation["demucs_device"] = "cuda"
    capability_seed = create_bounded_retry_capability_seed(
        alignment_audio,
        audio_source,
        helper,
        sample_rate=int(payload.get("sample_rate") or SAMPLE_RATE),
        device=str(payload.get("device") or args.whisperx_device),
    )
    report["ctc_bounded_retry_capability_seed"] = capability_seed
    _attach_bounded_retry_runtime_capability(report, capability_seed)
    apply_ctc_acoustic_backtrack(alignment_audio, entries, timestamps, report, duration)
    report.update(inactive_legacy_timing_diagnostics())
    update_report_confidence_metrics(report)
    return timestamps, report


def run_japanese_ctc_alignment(
    audio_path: Path, entries: list[LyricEntry], duration: float, args: argparse.Namespace
) -> tuple[list[float], dict[str, object]]:
    python_exe = default_ctc_python()
    helper = Path(__file__).resolve().parent / "japanese_ctc_align.py"
    with tempfile.TemporaryDirectory(prefix="lrc-ja-ctc-") as temp_name:
        temp_dir = Path(temp_name)
        transcript_path = temp_dir / "transcript.json"
        output_path = temp_dir / "japanese-ctc.json"
        transcript_path.write_text(
            json.dumps([entry_sung_text(entry) for entry in entries], ensure_ascii=False), encoding="utf-8"
        )
        with gpu_compute_phase():
            proc = run_command(
                [str(python_exe), str(helper), "--audio", str(audio_path), "--transcript", str(transcript_path), "--output", str(output_path), "--device", args.whisperx_device]
            )
        if proc.returncode != 0 or not output_path.exists():
            raise LrcError(f"Japanese CTC alignment failed: {proc.stderr.strip() or proc.stdout.strip()}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = payload.get("entries")
    if not isinstance(rows, list) or len(rows) != len(entries):
        raise LrcError("Japanese CTC helper returned an invalid entry list")
    timestamps = [max(0.0, min(duration, float(row.get("start", 0.0)))) for row in rows if isinstance(row, dict)]
    if len(timestamps) != len(entries):
        raise LrcError("Japanese CTC helper returned malformed timestamps")
    for index in range(1, len(timestamps)):
        timestamps[index] = max(timestamps[index], timestamps[index - 1] + 0.10)
    assignments = []
    for index, row in enumerate(rows):
        assert isinstance(row, dict)
        assignments.append(
            {
                "entry": index + 1,
                "segment": None,
                "score": 0.0,
                "ja_ctc_log_score": row.get("ja_ctc_log_score"),
                "ja_ctc_token_spans": row.get("token_spans"),
                "timestamp": round(timestamps[index], 3),
                "timing_repair": "japanese-ctc-forced-align",
                "timing_repair_source": "japanese-wav2vec2-ctc",
            }
        )
    collapse_entries = payload.get("collapse_entries") if isinstance(payload.get("collapse_entries"), list) else []
    report = {
        "backend": "jactc",
        "strategy": "japanese-ctc-forced-align-experimental",
        "ja_ctc_backend": payload.get("backend"),
        "ja_ctc_model": payload.get("model"),
        "ja_ctc_device": payload.get("device"),
        "ja_ctc_collapse_detected": bool(payload.get("collapse_detected")),
        "ja_ctc_collapse_entries": collapse_entries,
        "timing_entries": len(entries),
        "assignments": assignments,
        "review_required": bool(payload.get("collapse_detected")),
        "review_required_count": len(collapse_entries),
        "note": "Experimental Japanese Wav2Vec2 CTC candidate. Never selected by auto mode.",
    }
    update_report_confidence_metrics(report)
    return timestamps, report


def whisperx_segments(payload: dict[str, object]) -> list[AsrSegment]:
    segments: list[AsrSegment] = []
    for item in payload.get("segments", []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start) or start)
        chars: list[tuple[str, float]] = []
        for char_item in item.get("chars", []):
            if not isinstance(char_item, dict) or "start" not in char_item:
                continue
            char_text = str(char_item.get("char", ""))
            char_time = float(char_item.get("start", start) or start)
            for normalized in normalize_match_text(char_text):
                chars.append((normalized, char_time))
        if not chars and end - start < 0.05:
            continue
        segments.append(AsrSegment(start=start, end=end, text=text, chars=chars))
    if not segments:
        raise LrcError("WhisperX produced no aligned segments")
    return segments


def asr_transcript(segments: list[AsrSegment]) -> list[dict[str, object]]:
    return [
        {"start": segment.start, "end": segment.end, "text": segment.text}
        for segment in segments
        if segment.text.strip()
    ]


def _whispercpp_runtime_artifact_hashes(
    args: argparse.Namespace,
) -> tuple[str, str] | None:
    cli_arg = getattr(args, "whisper_cli", None)
    cli = Path(cli_arg).expanduser().resolve() if cli_arg else default_whisper_cli()
    model_arg = getattr(args, "whisper_model", None)
    model = Path(model_arg).expanduser().resolve() if model_arg else default_whisper_model()
    try:
        if not cli.is_file() or not model.is_file():
            return None
        return _path_sha256(cli), _path_sha256(model)
    except (OSError, LrcError):
        return None


def _whisperx_alignment_metadata(
    payload: dict[str, object],
) -> dict[str, str] | None:
    value = payload.get(_RUNTIME_WHISPERX_ALIGNMENT_METADATA_KEY)
    if not isinstance(value, dict):
        return None
    fields = {
        key: value.get(key)
        for key in (
            "helper_sha256",
            "helper_protocol_revision",
            "python_sha256",
            "whisperx_package_version",
            "align_model_id",
            "align_model_revision",
            "align_model_blob_sha256",
            "alignment_wav_sha256",
            "device",
            "language",
            "input_transcript_revision",
            "output_revision",
        )
    }
    if any(not isinstance(item, str) or not item.strip() for item in fields.values()):
        return None
    return {key: str(item) for key, item in fields.items()}


def _build_whisperx_runtime_provenance(
    audio_path: Path,
    args: argparse.Namespace,
    raw_segments: list[AsrSegment],
    initial_payload: dict[str, object],
    forced_payload: dict[str, object],
) -> WhisperXRuntimeProvenance | None:
    initial = _whisperx_alignment_metadata(initial_payload)
    forced = _whisperx_alignment_metadata(forced_payload)
    artifact_hashes = _whispercpp_runtime_artifact_hashes(args)
    if initial is None or forced is None or artifact_hashes is None:
        return None
    if any(initial[key] != forced[key] for key in (
        "helper_sha256",
        "helper_protocol_revision",
        "python_sha256",
        "whisperx_package_version",
        "align_model_id",
        "align_model_revision",
        "align_model_blob_sha256",
        "alignment_wav_sha256",
        "device",
        "language",
    )):
        return None
    raw_transcript_revision = _timing_content_digest(asr_transcript(raw_segments))
    if initial["input_transcript_revision"] != raw_transcript_revision:
        return None
    try:
        source_audio_sha256 = _path_sha256(audio_path)
    except (OSError, LrcError):
        return None
    payload = {
        "backend_lineage": "whisper-family",
        "raw_executable_sha256": artifact_hashes[0],
        "raw_model_sha256": artifact_hashes[1],
        "python_sha256": initial["python_sha256"],
        "helper_sha256": initial["helper_sha256"],
        "helper_protocol_revision": initial["helper_protocol_revision"],
        "whisperx_package_version": initial["whisperx_package_version"],
        "align_model_id": initial["align_model_id"],
        "align_model_revision": initial["align_model_revision"],
        "align_model_blob_sha256": initial["align_model_blob_sha256"],
        "source_audio_sha256": source_audio_sha256,
        "alignment_wav_sha256": initial["alignment_wav_sha256"],
        "device": initial["device"],
        "language": initial["language"],
        "raw_transcript_revision": raw_transcript_revision,
        "initial_aligned_output_revision": initial["output_revision"],
        "forced_output_revision": forced["output_revision"],
    }
    return WhisperXRuntimeProvenance(
        **payload,
        capability_revision=_timing_content_digest(payload),
    )


def run_whisperx_candidate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
    suppress_nst: bool,
    *,
    predecoded_wav: Path | None = None,
    raw_segments_override: list[AsrSegment] | None = None,
) -> tuple[list[float], dict[str, object]]:
    wav_path = predecoded_wav if predecoded_wav is not None else decode_temp_wav(audio_path)
    if raw_segments_override is not None:
        if suppress_nst:
            raise LrcError("Unsuppressed raw ASR reuse cannot satisfy suppress_nst=True")
        raw_segments = list(raw_segments_override)
    else:
        raw_segments = _run_whispercpp_wav(wav_path, args, suppress_nst=suppress_nst)
    raw_transcript = asr_transcript(raw_segments)
    aligned_payload = run_whisperx_alignment(wav_path, raw_transcript, args, predecoded_wav=True)
    aligned_segments = whisperx_segments(aligned_payload)
    timestamps, report = match_whisper_segments(entries, aligned_segments, duration)
    report["backend"] = "whisperx"
    report["asr_segment_timing_source"] = "whisperx_aligned"
    report["whisperx_aligned_asr_segments"] = len(aligned_segments)
    report["whisper_suppress_nst"] = suppress_nst
    report["whisperx_device"] = aligned_payload.get("device")
    report["vocal_regions"] = []
    report["vocal_regions_available"] = False
    report["no_vocal_constraint"] = {
        "status": "placeholder",
        "enforced": False,
        "reason": "No reliable vocal-region detector is wired yet.",
    }
    report["local_window_realign"] = {
        "status": "placeholder",
        "enabled": False,
        "reason": "Disabled until a reliable vocal-region/onset constraint is available.",
    }
    if is_alignment_unusable(report):
        raise LrcError(
            "WhisperX ASR-to-lyric matching is unusable; refusing to write a fake LRC. "
            f"{unusable_alignment_reason(report)}"
        )
    timestamps, intro_changes = repair_intro_hallucination(audio_path, entries, timestamps, report, duration)
    if intro_changes:
        sync_scratch_assignment_timestamps(report, timestamps)
    timestamps, long_segment_changes = repair_long_segment_hallucinations(
        audio_path,
        entries,
        timestamps,
        report,
        aligned_segments,
        duration,
    )
    if long_segment_changes:
        sync_scratch_assignment_timestamps(report, timestamps)
    timestamps, tail_changes = repair_tail_hallucination(
        audio_path,
        entries,
        timestamps,
        report,
        aligned_segments,
        duration,
    )
    if tail_changes:
        sync_scratch_assignment_timestamps(report, timestamps)
    forced_payload = run_whisperx_alignment(
        wav_path,
        lyric_alignment_windows(entries, timestamps, duration),
        args,
        predecoded_wav=True,
    )
    forced_segments = whisperx_segments(forced_payload)
    runtime_provenance = _build_whisperx_runtime_provenance(
        audio_path,
        args,
        raw_segments,
        aligned_payload,
        forced_payload,
    )
    timestamps, refinement_changes = apply_whisperx_lyric_refinement(
        entries,
        timestamps,
        report,
        aligned_segments,
        forced_segments,
        duration,
    )
    sync_scratch_assignment_timestamps(report, timestamps)
    report["backend"] = "whisperx"
    report["strategy"] = "whisperx-hybrid-experimental"
    report["whisper_suppress_nst"] = suppress_nst
    report["whisperx_device"] = forced_payload.get("device") or aligned_payload.get("device")
    report["whisper_language"] = args.whisper_language
    report["whisperx_forced_first_times"] = [round(first_forced_char_time(segment), 3) for segment in forced_segments]
    report["whisperx_forced_rows"] = [
        {
            "segment_start": round(float(segment.start), 6),
            "segment_end": round(float(segment.end), 6),
            "first_char_time": round(float(first_forced_char_time(segment)), 6),
            "last_char_time": round(
                float(segment.chars[-1][1] if segment.chars else segment.start), 6
            ),
            "char_count": len(segment.chars),
        }
        for segment in forced_segments
    ]
    if runtime_provenance is not None:
        report[_RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY] = runtime_provenance
    report["whisperx_refinement_count"] = len(refinement_changes)
    report["whisperx_refinements"] = refinement_changes
    timestamps, tail_blend_changes = blend_tail_repair_with_forced(
        entries,
        timestamps,
        report,
        forced_segments,
        duration,
    )
    if tail_blend_changes:
        sync_scratch_assignment_timestamps(report, timestamps)
    add_alignment_diagnostics(entries, timestamps, report, forced_segments, duration)
    if is_alignment_untrusted(report):
        raise AlignmentEvidenceRejected(
            "WhisperX alignment is untrusted after diagnostics; refusing to write a fake LRC. "
            f"{untrusted_alignment_reason(report)}",
            timestamps,
            report,
        )
    return timestamps, report


def should_prefer_ctc_over_review_exploded_hybrid(
    hybrid_report: dict[str, object], ctc_report: dict[str, object]
) -> bool:
    """Reject a locally patched hybrid when it becomes much less auditable than CTC."""
    if ctc_report.get("collapse_detected"):
        return False
    try:
        ctc_missing = int(ctc_report.get("ctc_missing_count", 0) or 0)
        ctc_reviews = int(ctc_report.get("review_required_count", 0) or 0)
        hybrid_reviews = int(hybrid_report.get("review_required_count", 0) or 0)
    except (TypeError, ValueError):
        return False
    if ctc_missing:
        return False
    return hybrid_reviews >= max(8, ctc_reviews * 3 + 5)


def choose_onset_consensus_time(
    ctc_time: float,
    ctc_first_score: float,
    whisper_time: float,
    whisper_score: float,
    japanese_ctc_time: float,
) -> tuple[float | None, str | None]:
    """Choose a narrow onset rescue when CTC's first phoneme is unreliable."""
    ctc_whisper_gap = abs(ctc_time - whisper_time)
    whisper_japanese_gap = abs(whisper_time - japanese_ctc_time)
    if (
        ctc_first_score <= 0.01
        and whisper_score >= 0.80
        and 0.35 <= ctc_whisper_gap <= 0.80
        and 0.60 <= whisper_japanese_gap <= 1.00
    ):
        return whisper_time, "high-confidence-whisperx-over-weak-ctc-initial"
    if (
        ctc_first_score <= 0.01
        and whisper_score >= 0.50
        and ctc_whisper_gap >= 0.20
        and whisper_japanese_gap <= 0.15
    ):
        return (whisper_time + japanese_ctc_time) / 2.0, "whisperx-japanese-ctc-onset-consensus"
    if (
        0.05 <= ctc_first_score <= 0.10
        and whisper_score >= 0.80
        and whisper_japanese_gap <= 0.15
        and ctc_whisper_gap >= 0.20
    ):
        return (whisper_time + japanese_ctc_time) / 2.0, "whisperx-japanese-ctc-over-ctc-onset-consensus"
    return None, None


def needs_suppressed_retry(report: dict[str, object]) -> bool:
    matched_percent = float(report.get("trusted_percent", report.get("matched_percent", 0.0)) or 0.0)
    timing_entries = int(report.get("timing_entries", 0) or 0)
    low_confidence_count = int(report.get("low_confidence_count", 0) or 0)
    low_limit = max(6, math.ceil(timing_entries * 0.20))
    return matched_percent < 90.0 or low_confidence_count > low_limit


def is_alignment_unusable(report: dict[str, object]) -> bool:
    timing_entries = int(report.get("timing_entries", 0) or 0)
    matched_entries = int(report.get("trusted_entries", report.get("matched_entries", 0)) or 0)
    matched_percent = float(report.get("trusted_percent", report.get("matched_percent", 0.0)) or 0.0)
    low_confidence_count = int(report.get("low_confidence_count", 0) or 0)
    if timing_entries <= 0:
        return True
    if matched_entries == 0:
        return True
    if matched_percent < 35.0:
        return True
    if timing_entries >= 6 and low_confidence_count >= math.ceil(timing_entries * 0.80):
        return True
    return False


def is_alignment_untrusted(report: dict[str, object]) -> bool:
    if is_alignment_unusable(report):
        return True
    if report.get("collapse_detected"):
        return True
    timing_entries = int(report.get("timing_entries", 0) or 0)
    trusted_percent = float(report.get("trusted_percent", report.get("matched_percent", 0.0)) or 0.0)
    low_confidence_percent = float(report.get("low_confidence_percent", 0.0) or 0.0)
    high_risk_count = 0
    max_post_long_uncertain_run = 0
    suspicious = report.get("suspicious_alignments")
    if isinstance(suspicious, list):
        high_risk_count = sum(
            1 for item in suspicious if isinstance(item, dict) and item.get("severity") == "high"
        )
        flagged_entries: set[int] = set()
        for item in suspicious:
            if not isinstance(item, dict):
                continue
            flags = item.get("flags")
            if not isinstance(flags, list) or "post_long_segment_region_uncertain" not in flags:
                continue
            try:
                flagged_entries.add(int(item.get("entry", 0)))
            except (TypeError, ValueError):
                continue
        current_run = 0
        for entry_index in range(1, timing_entries + 1):
            if entry_index in flagged_entries:
                current_run += 1
                max_post_long_uncertain_run = max(max_post_long_uncertain_run, current_run)
            else:
                current_run = 0
        report["max_post_long_uncertain_run"] = max_post_long_uncertain_run
    if timing_entries >= 8 and trusted_percent < 85.0:
        return True
    if timing_entries >= 8 and low_confidence_percent >= 50.0:
        return True
    if timing_entries >= 8 and max_post_long_uncertain_run >= 4:
        return True
    if timing_entries >= 8 and high_risk_count >= math.ceil(timing_entries * 0.25):
        return True
    return False


def unusable_alignment_reason(report: dict[str, object]) -> str:
    return (
        f"trusted_entries={report.get('trusted_entries', report.get('matched_entries'))}, "
        f"timing_entries={report.get('timing_entries')}, "
        f"trusted_percent={report.get('trusted_percent', report.get('matched_percent'))}, "
        f"assigned_percent={report.get('assigned_percent')}, "
        f"low_confidence_count={report.get('low_confidence_count')}"
    )


def untrusted_alignment_reason(report: dict[str, object]) -> str:
    segment_runs = report.get("segment_collapse_runs")
    zero_runs = report.get("alignment_collapse_runs")
    return (
        f"trusted_entries={report.get('trusted_entries', report.get('matched_entries'))}, "
        f"timing_entries={report.get('timing_entries')}, "
        f"trusted_percent={report.get('trusted_percent', report.get('matched_percent'))}, "
        f"assigned_percent={report.get('assigned_percent')}, "
        f"low_confidence_percent={report.get('low_confidence_percent')}, "
        f"review_required_percent={report.get('review_required_percent')}, "
        f"collapse_detected={report.get('collapse_detected')}, "
        f"max_post_long_uncertain_run={report.get('max_post_long_uncertain_run')}, "
        f"segment_collapse_runs={segment_runs}, "
        f"alignment_collapse_runs={zero_runs}"
    )


def try_whisperx_candidate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
    suppress_nst: bool,
    *,
    predecoded_wav: Path | None = None,
    raw_segments_override: list[AsrSegment] | None = None,
) -> tuple[list[float], dict[str, object], str | None]:
    try:
        timestamps, report = run_whisperx_candidate(
            audio_path,
            entries,
            duration,
            args,
            suppress_nst,
            predecoded_wav=predecoded_wav,
            raw_segments_override=raw_segments_override,
        )
        return timestamps, report, None
    except AlignmentEvidenceRejected as exc:
        report = copy.deepcopy(exc.report)
        report["whisper_suppress_nst"] = suppress_nst
        report["candidate_usable"] = False
        report["candidate_rejection_reason"] = str(exc)
        return list(exc.timestamps), report, str(exc)
    except LrcError as exc:
        return [], {"whisper_suppress_nst": suppress_nst, "error": str(exc)}, str(exc)


SEGMENT_GATE_OVERRIDE_ATTR = "_segment_gate_override"


SEGMENT_GATE_STATUS_ATTR = "_segment_gate_status"


def _set_runtime_arg(args: object, name: str, value: object) -> None:
    # Synthetic caller/test arguments may be immutable objects.
    try:
        setattr(args, name, value)
    except (AttributeError, TypeError):
        pass


def try_ctc_candidate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
) -> tuple[list[float], dict[str, object], str | None]:
    try:
        timestamps, report = run_ctc_alignment(audio_path, entries, duration, args)
        skeleton_mode = str(getattr(args, "skeleton", "ctc"))
        if skeleton_mode == "segment" and str(getattr(args, "timing_source", "")) == "auto":
            from segment_skeleton_production import run_segment_skeleton
            if report.get("ctc_audio_source") == "vocal-stem":
                timestamps, report, skeleton_details = run_segment_skeleton(
                    report=report,
                    timestamps=timestamps,
                    lyric_texts=[entry_sung_text(entry) for entry in entries],
                    vocal_stem=ctc_alignment_audio_path(report, audio_path),
                    duration=duration,
                    audio_api=_AUDIO_API,
                    python_exe=default_ctc_python(),
                    ctc_helper=Path(__file__).resolve().parent / "ctc_align.py",
                    device=str(getattr(args, "whisperx_device", "auto")),
                    helper_runner=lambda command, timeout: _run_model_helper_command(
                        command, mode="ctc"
                    ),
                    romanizer=japanese_romaji,
                    raw_asr_events=getattr(args,'_skeleton_raw_asr_events',()),
                    gate_override=getattr(args, SEGMENT_GATE_OVERRIDE_ATTR, None),
                    cache_args=args,
                )
                _set_runtime_arg(args, SEGMENT_GATE_STATUS_ATTR, skeleton_details)
            else:
                report["segment_skeleton_status"] = {
                    "status": "FALLBACK_CTC",
                    "reason": "NO_ISOLATED_VOCAL_STEM",
                    "whole_song": True,
                }
        else:
            report["segment_skeleton_status"] = {
                "status": "CTC_SELECTED",
                "reason": "EXPLICIT_CTC_OR_NON_AUTO",
                "whole_song": True,
            }
        return timestamps, report, None
    except LrcError as exc:
        return [], {"backend": "ctc", "error": str(exc)}, str(exc)


def ctc_alignment_audio_path(report: dict[str, object], fallback: Path) -> Path:
    candidate = report.get("ctc_audio_path")
    if isinstance(candidate, str):
        path = Path(candidate)
        if path.exists():
            return path
    return fallback


def try_whispercpp_diagnostic_candidate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
    *,
    predecoded_wav: Path | None = None,
) -> tuple[list[float], dict[str, object], str | None, list[AsrSegment] | None]:
    raw_segments: list[AsrSegment] | None = None
    try:
        raw_segments = (
            _run_whispercpp_wav(predecoded_wav, args, suppress_nst=False)
            if predecoded_wav is not None
            else run_whispercpp(audio_path, args, suppress_nst=False)
        )
        timestamps, report = match_whisper_segments(entries, raw_segments, duration)
        report["backend"] = "whispercpp_raw"
        report["strategy"] = "whispercpp-raw-diagnostic"
        report["asr_repeated_tail_after_0s"] = has_repeated_asr_tail(raw_segments, 0.0)
        report["raw_asr_segments"] = [
            {"start": round(segment.start, 3), "end": round(segment.end, 3), "text": segment.text}
            for segment in raw_segments
        ]
        report["raw_asr_max_segment_seconds"] = round(
            max((segment.end - segment.start for segment in raw_segments), default=0.0), 3
        )
        report["diagnostic_only"] = True
        return timestamps, report, None, raw_segments
    except (LrcError, OSError, ValueError) as exc:
        if _model_cache_only(args):
            raise ModelCacheMiss(f"model-cache-miss:raw-ASR:{exc}") from exc
        return (
            [],
            {"backend": "whispercpp_raw", "diagnostic_only": True, "error": str(exc)},
            str(exc),
            raw_segments,
        )


def run_whisperx_best_candidate(
    audio_path: Path,
    entries: list[LyricEntry],
    duration: float,
    args: argparse.Namespace,
    *,
    predecoded_wav: Path | None = None,
    normal_raw_segments: list[AsrSegment] | None = None,
) -> tuple[list[float], dict[str, object]]:
    if args.whisper_suppress_nst is not None:
        try:
            explicit_suppress = bool(args.whisper_suppress_nst)
            return run_whisperx_candidate(
                audio_path,
                entries,
                duration,
                args,
                suppress_nst=explicit_suppress,
                predecoded_wav=predecoded_wav,
                raw_segments_override=(
                    normal_raw_segments if not explicit_suppress else None
                ),
            )
        except AlignmentEvidenceRejected as exc:
            raise WhisperXEvidenceBundleError(
                str(exc),
                [(list(exc.timestamps), copy.deepcopy(exc.report), str(exc))],
            ) from exc

    normal_timestamps, normal_report, normal_error = try_whisperx_candidate(
        audio_path,
        entries,
        duration,
        args,
        suppress_nst=False,
        predecoded_wav=predecoded_wav,
        raw_segments_override=normal_raw_segments,
    )
    candidates = [normal_report]

    # A failed normal candidate gets exactly one suppress-NST rescue.  The old
    # control flow then re-ran needs_suppressed_retry() *on the rescue report*
    # and could execute the identical rescue a third time.  Rescue is already
    # the alternate decode policy, so there is no second suppress policy to try.
    if normal_error:
        sns_timestamps, sns_report, sns_error = try_whisperx_candidate(
            audio_path,
            entries,
            duration,
            args,
            suppress_nst=True,
            predecoded_wav=predecoded_wav,
        )
        candidates.append(sns_report)
        if sns_error:
            evidence_candidates = [
                (normal_timestamps, normal_report, normal_error),
                (sns_timestamps, sns_report, sns_error),
            ]
            raise WhisperXEvidenceBundleError(
                "No usable WhisperX candidate; refusing to write a fake LRC. "
                + " | ".join(str(item) for item in (normal_error, sns_error) if item),
                [
                    (candidate_times, candidate_report, str(candidate_error))
                    for candidate_times, candidate_report, candidate_error in evidence_candidates
                    if candidate_error and isinstance(candidate_report.get("assignments"), list)
                ],
            )
        timestamps = list(sns_timestamps)
        report = sns_report
        report["candidate_reports"] = [
            candidate_summary(candidate, str(candidate.get("error")) if candidate.get("error") else None)
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        return timestamps, report

    timestamps = list(normal_timestamps)
    report = normal_report
    if needs_suppressed_retry(normal_report):
        sns_timestamps, sns_report, sns_error = try_whisperx_candidate(
            audio_path,
            entries,
            duration,
            args,
            suppress_nst=True,
            predecoded_wav=predecoded_wav,
        )
        candidates.append(sns_report)
        if not sns_error:
            collapse_start = low_confidence_run_start(normal_report)
            quality_gap = report_quality(sns_report) - report_quality(normal_report)
            if quality_gap >= 15.0:
                timestamps, report = list(sns_timestamps), sns_report
            elif collapse_start is not None and report_quality(sns_report) > report_quality(normal_report):
                timestamps = list(normal_timestamps[:collapse_start]) + list(sns_timestamps[collapse_start:])
                adopted_sns_entries: list[int] = []
                for change in sns_report.get("whisperx_refinements", []):
                    if not isinstance(change, dict):
                        continue
                    entry_number = change.get("entry")
                    if not isinstance(entry_number, int):
                        continue
                    entry_index = entry_number - 1
                    if not (0 <= entry_index < collapse_start):
                        continue
                    if change.get("reason") != "bounded-forced-asr-blend":
                        continue
                    shift = normal_timestamps[entry_index] - sns_timestamps[entry_index]
                    if not (0.40 <= shift <= 1.60):
                        continue
                    left_ok = entry_index == 0 or timestamps[entry_index - 1] + 0.10 < sns_timestamps[entry_index]
                    right_ok = (
                        entry_index + 1 >= len(timestamps)
                        or sns_timestamps[entry_index] < timestamps[entry_index + 1] - 0.10
                    )
                    if left_ok and right_ok:
                        timestamps[entry_index] = sns_timestamps[entry_index]
                        adopted_sns_entries.append(entry_index)
                if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
                    # R2-7: a normal/SNS seam that runs backwards is not a valid
                    # candidate; use the complete, higher-quality SNS candidate.
                    timestamps, report = list(sns_timestamps), sns_report
                else:
                    report = fused_candidate_report(
                        normal_report, sns_report, timestamps, collapse_start, adopted_sns_entries
                    )
            elif report_quality(sns_report) > report_quality(report):
                timestamps, report = list(sns_timestamps), sns_report

    report["candidate_reports"] = [
        candidate_summary(candidate, str(candidate.get("error")) if candidate.get("error") else None)
        for candidate in candidates
        if isinstance(candidate, dict)
    ]
    return timestamps, report


def choose_alignment_candidate(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    usable = [
        candidate
        for candidate in candidates
        if not candidate.get("error")
        and isinstance(candidate.get("timestamps"), list)
        and isinstance(candidate.get("report"), dict)
    ]
    if not usable:
        errors = [
            str(candidate.get("error"))
            for candidate in candidates
            if candidate.get("error")
        ]
        raise LrcError("No usable alignment candidate. " + " | ".join(errors))

    for candidate in usable:
        report = candidate["report"]
        assert isinstance(report, dict)
        candidate["quality"] = candidate_selection_quality(report)

    best = max(usable, key=lambda item: float(item.get("quality", -100.0) or -100.0))
    ctc = next((item for item in usable if item.get("backend") == "ctc"), None)
    if ctc is not None:
        ctc_quality = float(ctc.get("quality", -100.0) or -100.0)
        best_quality = float(best.get("quality", -100.0) or -100.0)
        if ctc_quality >= 80.0 and ctc_quality >= best_quality - 5.0:
            return ctc
    return best


def raw_asr_is_fallback_eligible(report: dict[str, object]) -> bool:
    """Reject raw fallback when Whisper merged a large section into one segment."""
    try:
        max_segment = float(report.get("raw_asr_max_segment_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    return 0.0 < max_segment <= 12.0


def low_confidence_run_start(report: dict[str, object], run_length: int = 4) -> int | None:
    assignments = report.get("assignments", [])
    if not isinstance(assignments, list):
        return None
    low_flags: list[bool] = []
    for item in assignments:
        if not isinstance(item, dict):
            low_flags.append(True)
            continue
        low_flags.append(item.get("segment") is None or float(item.get("score", 0.0) or 0.0) < 0.62)
    for index in range(0, max(0, len(low_flags) - run_length + 1)):
        if all(low_flags[index : index + run_length]):
            return index
    return None


def fused_candidate_report(
    normal_report: dict[str, object],
    sns_report: dict[str, object],
    timestamps: list[float],
    collapse_start: int,
    adopted_sns_entries: list[int],
) -> dict[str, object]:
    report = copy.deepcopy(sns_report)
    normal_assignments = normal_report.get("assignments", [])
    sns_assignments = sns_report.get("assignments", [])
    if isinstance(normal_assignments, list) and isinstance(sns_assignments, list):
        fused_assignments: list[object] = []
        for index in range(max(len(normal_assignments), len(sns_assignments), len(timestamps))):
            use_sns = index >= collapse_start or index in adopted_sns_entries
            source = sns_assignments if use_sns else normal_assignments
            item = copy.deepcopy(source[index]) if index < len(source) else {"entry": index + 1}
            if isinstance(item, dict) and index < len(timestamps):
                item["timestamp"] = round(timestamps[index], 3)
            fused_assignments.append(item)
        report["assignments"] = fused_assignments
    normal_forced_rows = normal_report.get("whisperx_forced_rows", [])
    sns_forced_rows = sns_report.get("whisperx_forced_rows", [])
    if isinstance(normal_forced_rows, list) and isinstance(sns_forced_rows, list):
        fused_forced_rows: list[object] = []
        for index in range(max(len(normal_forced_rows), len(sns_forced_rows), len(timestamps))):
            use_sns = index >= collapse_start or index in adopted_sns_entries
            source = sns_forced_rows if use_sns else normal_forced_rows
            fused_forced_rows.append(copy.deepcopy(source[index]) if index < len(source) else {})
        report["whisperx_forced_rows"] = fused_forced_rows
    normal_forced_times = normal_report.get("whisperx_forced_first_times", [])
    sns_forced_times = sns_report.get("whisperx_forced_first_times", [])
    if isinstance(normal_forced_times, list) and isinstance(sns_forced_times, list):
        fused_forced_times: list[object] = []
        for index in range(max(len(normal_forced_times), len(sns_forced_times), len(timestamps))):
            use_sns = index >= collapse_start or index in adopted_sns_entries
            source = sns_forced_times if use_sns else normal_forced_times
            fused_forced_times.append(copy.deepcopy(source[index]) if index < len(source) else None)
        report["whisperx_forced_first_times"] = fused_forced_times
    report["candidate_fusion"] = {
        "normal_until_entry": collapse_start,
        "sns_from_entry": collapse_start + 1,
        "adopted_sns_entries_before_tail": [entry + 1 for entry in adopted_sns_entries],
        "reason": "normal-candidate-low-confidence-run",
    }
    return report


def match_whisper_segments(entries: list[LyricEntry], segments: list[AsrSegment], duration: float) -> tuple[list[float], dict[str, object]]:
    entry_texts = [normalize_match_text(entry.lines[0] if entry.lines else entry.alignment_text) for entry in entries]
    segment_texts = [normalize_match_text(segment.text) for segment in segments]

    assignments: list[int | None] = [None] * len(entries)
    scores: list[float] = [0.0] * len(entries)
    borrowed: list[bool] = [False] * len(entries)
    last_segment = -1

    for entry_index, entry_text in enumerate(entry_texts):
        if not entry_text:
            continue
        best: tuple[float, int] | None = None
        search_start = max(0, last_segment)
        search_end = min(len(segment_texts), last_segment + 12)
        for segment_index in range(search_start, search_end):
            variant_best: tuple[float, int] | None = None
            for span in (1, 2, 3):
                if segment_index + span > len(segment_texts):
                    continue
                parts = segment_texts[segment_index : segment_index + span]
                combined = "".join(parts)
                score = lyric_segment_score(entry_text, combined)
                target_segment = segment_index
                exact_index = combined.find(entry_text)
                if exact_index >= 0:
                    cursor = 0
                    for offset, part in enumerate(parts):
                        if exact_index < cursor + len(part):
                            target_segment = segment_index + offset
                            break
                        cursor += len(part)
                candidate = (score, target_segment)
                if variant_best is None or candidate[0] > variant_best[0]:
                    variant_best = candidate
            if variant_best and (best is None or variant_best[0] > best[0]):
                best = variant_best
            if variant_best and variant_best[0] >= 0.88:
                break
        if best and best[0] >= 0.42:
            scores[entry_index] = best[0]
            assignments[entry_index] = best[1]
            last_segment = max(last_segment, best[1])

    for entry_index, segment_index in enumerate(assignments):
        if segment_index is None or segment_index <= 0:
            continue
        entry_text = entry_texts[entry_index]
        current_text = segment_texts[segment_index]
        for previous_index in range(segment_index - 1, max(-1, segment_index - 3), -1):
            previous_text = segment_texts[previous_index]
            if not previous_text:
                continue
            if segments[segment_index].start - segments[previous_index].start > 2.5:
                continue
            if sequence_ratio(previous_text, current_text) < 0.70:
                continue
            previous_score = lyric_segment_score(entry_text, previous_text)
            if previous_score >= scores[entry_index] - 0.05:
                assignments[entry_index] = previous_index
                scores[entry_index] = max(scores[entry_index], previous_score)
                segment_index = previous_index
                current_text = previous_text
                break
        if segment_index is None or segment_index <= 0:
            continue
        for previous_index in range(segment_index - 1, max(-1, segment_index - 4), -1):
            previous_text = segment_texts[previous_index]
            if not previous_text:
                continue
            if segments[segment_index].start - segments[previous_index].start > 5.0:
                continue
            if sequence_ratio(previous_text, current_text) < 0.86:
                continue
            if not entry_text.startswith(previous_text):
                continue
            assignments[entry_index] = previous_index
            scores[entry_index] = max(scores[entry_index], 0.80)
            break

    timestamps: list[float | None] = [None] * len(entries)
    for index, segment_index in enumerate(assignments):
        if segment_index is not None:
            timestamp = segments[segment_index].start
            char_time = segment_char_time(entry_texts[index], segments[segment_index])
            if char_time is not None:
                timestamp = char_time
            elif not segments[segment_index].chars:
                raw_entry_text = entries[index].lines[0] if entries[index].lines else entries[index].alignment_text
                estimated_time = estimated_segment_text_time(entry_texts[index], raw_entry_text, segments[segment_index])
                if estimated_time is not None:
                    timestamp = estimated_time
            timestamps[index] = timestamp

    # Fill missing lines by interpolation. If a missing line immediately precedes
    # a close matched segment, treat it as the first lyric in that combined ASR
    # segment instead of placing it halfway through the previous gap.
    for index, timestamp in enumerate(timestamps):
        if timestamp is not None:
            continue
        left = next((idx for idx in range(index - 1, -1, -1) if timestamps[idx] is not None), None)
        right = next((idx for idx in range(index + 1, len(timestamps)) if timestamps[idx] is not None), None)
        if (
            left is not None
            and right is not None
            and assignments[left] is not None
            and assignments[left] == assignments[right]
        ):
            assignments[index] = assignments[left]
            timestamps[index] = segments[assignments[index]].start
            continue
        if right is not None and assignments[right] is not None:
            segment_start = segments[assignments[right]].start
            if (
                (left is None or segment_start - float(timestamps[left] or 0.0) <= 4.2)
                and (left is None or segment_start > float(timestamps[left] or 0.0) + 0.4)
            ):
                timestamps[index] = segment_start
                assignments[index] = assignments[right]
                borrowed[index] = True
                continue
        if (
            left is not None
            and right is not None
            and assignments[left] is not None
            and not is_instrumental_marker(entries[index])
        ):
            previous_end = segments[assignments[left]].end
            if float(timestamps[right] or 0.0) - previous_end >= 6.0:
                entry_text = normalize_match_text(entries[index].lines[0] if entries[index].lines else entries[index].alignment_text)
                tail_offset = min(3.2, max(2.8, len(entry_text) * 0.45))
                timestamps[index] = min(
                    float(timestamps[right] or 0.0) - 0.50,
                    previous_end + tail_offset,
                )
                continue
        if left is not None and right is not None:
            span = right - left
            timestamps[index] = float(timestamps[left] or 0.0) + (float(timestamps[right] or 0.0) - float(timestamps[left] or 0.0)) * ((index - left) / span)
        elif left is not None:
            timestamps[index] = min(duration, float(timestamps[left] or 0.0) + 2.0 * (index - left))
        elif right is not None:
            timestamps[index] = max(0.0, float(timestamps[right] or 0.0) - 2.0 * (right - index))
        else:
            timestamps[index] = 0.0

    # When a borrowed line and the following matched line share one ASR segment,
    # split the segment instead of stamping both at the same start time.
    for index in range(len(timestamps) - 1):
        if borrowed[index] and assignments[index] is not None and assignments[index + 1] == assignments[index]:
            segment = segments[assignments[index]]
            segment_duration = max(0.0, segment.end - segment.start)
            timestamps[index] = segment.start
            timestamps[index + 1] = max(
                float(timestamps[index + 1] or 0.0),
                segment.start + segment_duration * 0.55,
            )

    final = [float(timestamp or 0.0) for timestamp in timestamps]
    for index in range(1, len(final) - 1):
        segment_index = assignments[index]
        next_segment_index = assignments[index + 1]
        if (
            segment_index is None
            or next_segment_index is None
            or segment_index == next_segment_index
            or scores[index] < 0.60
        ):
            continue
        current_segment = segments[segment_index]
        next_time = final[index + 1]
        gap_after_segment = next_time - current_segment.end
        gap_after_previous = final[index] - final[index - 1]
        current_duration = current_segment.end - current_segment.start
        current_text = segment_texts[segment_index]
        entry_text = normalize_match_text(entries[index].lines[0] if entries[index].lines else entries[index].alignment_text)
        direct_segment_match = (
            bool(entry_text)
            and (
                entry_text in current_text
                or current_text in entry_text
                or lyric_segment_score(entry_text, current_text) >= 0.82
            )
        )
        if (
            gap_after_segment < 6.0
            or next_time - final[index] < 8.0
            or gap_after_previous < 4.5
            or current_duration < 2.5
            or direct_segment_match
        ):
            continue
        lead_time = min(2.8, max(1.4, 1.0 + len(entry_text) * 0.10))
        shifted = max(final[index - 1] + 0.10, next_time - lead_time)
        if final[index] < shifted < next_time - 0.30:
            final[index] = shifted

    group_start = 0
    while group_start < len(final):
        segment_index = assignments[group_start]
        group_end = group_start + 1
        while group_end < len(final) and assignments[group_end] == segment_index:
            group_end += 1
        group_size = group_end - group_start
        if segment_index is not None and group_size > 1:
            segment = segments[segment_index]
            next_segment_start: float | None = None
            for later_index in range(group_end, len(final)):
                later_segment = assignments[later_index]
                if later_segment is not None and later_segment != segment_index:
                    next_segment_start = segments[later_segment].start
                    break
            segment_span = max(0.5, segment.end - segment.start)
            if next_segment_start is not None and 0.5 < next_segment_start - segment.start <= 18.0:
                interval_end = next_segment_start - 0.10
            else:
                interval_end = segment.start + max(segment_span, group_size * 3.5)
                if next_segment_start is not None:
                    interval_end = min(interval_end, next_segment_start - 0.10)
            interval_end = max(interval_end, segment.start + 0.20 * group_size)
            interval = interval_end - segment.start
            for offset, entry_index in enumerate(range(group_start, group_end)):
                if group_size == 2:
                    fraction = offset / group_size
                else:
                    fraction = offset / max(1.0, group_size - 0.55)
                distributed = segment.start + interval * fraction
                final[entry_index] = max(final[entry_index], distributed)
        group_start = group_end

    for index in range(1, len(final)):
        final[index] = max(final[index], final[index - 1] + 0.10)
    for index in range(len(final)):
        final[index] = max(0.0, min(duration, final[index]))

    assigned = sum(1 for assignment in assignments if assignment is not None)
    trusted = sum(
        1
        for assignment, score in zip(assignments, scores)
        if assignment is not None and score >= TRUSTED_ALIGNMENT_SCORE
    )
    low_confidence = [
        {
            "entry": index + 1,
            "score": round(scores[index], 3),
            "lyric": entries[index].lines[0] if entries[index].lines else "",
            "asr_segment": assignments[index] + 1 if assignments[index] is not None else None,
        }
        for index in range(len(entries))
        if assignments[index] is None or scores[index] < 0.62
    ]
    raw_review_entries = [
        {
            "entry": index + 1,
            "text": entry_sung_text(entries[index]),
            "flags": [
                "raw_asr_missing_alignment" if assignments[index] is None else "raw_asr_low_match_score",
                *( ["raw_asr_borrowed_segment"] if borrowed[index] else [] ),
            ],
            "severity": "high" if assignments[index] is None or scores[index] <= 0.0 else "medium",
            "review_required": True,
        }
        for index in range(len(entries))
        if assignments[index] is None or scores[index] < TRUSTED_ALIGNMENT_SCORE or borrowed[index]
    ]
    report: dict[str, object] = {
        "backend": "whispercpp",
        "asr_segments": len(segments),
        "timing_entries": len(entries),
        "assigned_entries": assigned,
        "assigned_percent": ratio_percent(assigned, len(entries)),
        "trusted_entries": trusted,
        "trusted_percent": ratio_percent(trusted, len(entries)),
        "matched_entries": trusted,
        "matched_percent": ratio_percent(trusted, len(entries)),
        "low_confidence_entries": low_confidence,
        "low_confidence_count": len(low_confidence),
        "low_confidence_percent": ratio_percent(len(low_confidence), len(entries)),
        "review_required_count": len(raw_review_entries),
        "review_required_percent": ratio_percent(len(raw_review_entries), len(entries)),
        "review_required": bool(raw_review_entries),
        "suspicious_alignment_count": len(raw_review_entries),
        "suspicious_alignment_severity_counts": {
            "high": sum(1 for item in raw_review_entries if item["severity"] == "high"),
            "medium": sum(1 for item in raw_review_entries if item["severity"] == "medium"),
            "low": 0,
        },
        "suspicious_alignments": raw_review_entries,
        "assignments": [
            {
                "entry": index + 1,
                "segment": assignments[index] + 1 if assignments[index] is not None else None,
                "score": round(scores[index], 3),
                "borrowed": borrowed[index],
                "timestamp": round(final[index], 3),
            }
            for index in range(len(entries))
        ],
        "note": "Experimental ASR-to-lyric matching. Review low-confidence entries before trusting output.",
    }
    return final, report


def lyric_alignment_windows(entries: list[LyricEntry], timestamps: list[float], duration: float) -> list[dict[str, object]]:
    transcript: list[dict[str, object]] = []
    for index, (entry, timestamp) in enumerate(zip(entries, timestamps)):
        previous_time = timestamps[index - 1] if index > 0 else 0.0
        next_time = timestamps[index + 1] if index + 1 < len(timestamps) else min(duration, timestamp + 4.0)
        entry_text = normalize_match_text(entry_sung_text(entry))
        gap_from_previous = timestamp - previous_time if index > 0 else timestamp
        lookback = 4.2 if len(entry_text) <= 3 and gap_from_previous >= 4.0 else 1.2
        start = max(0.0, min(timestamp - lookback, (previous_time + timestamp) / 2 if index > 0 else timestamp - lookback))
        end = min(duration, max(next_time + 0.5, timestamp + 1.2))
        transcript.append({"start": start, "end": end, "text": entry_sung_text(entry)})
    return transcript


def build_local_whisper_capability(
    audio_path: Path,
    args: argparse.Namespace,
) -> LocalWhisperCapability | None:
    """Build live provenance for unresolved-only Whisper-family observations.

    This authority is reconstructed from the actual current executable/model/helper
    files and source audio.  It is never inferred from serialized report metadata.
    """
    whisper_cli_arg = getattr(args, "whisper_cli", None)
    whisper_model_arg = getattr(args, "whisper_model", None)
    whisperx_python_arg = getattr(args, "whisperx_python", None)
    cli = Path(whisper_cli_arg).expanduser().resolve() if whisper_cli_arg else default_whisper_cli()
    model = Path(whisper_model_arg).expanduser().resolve() if whisper_model_arg else default_whisper_model()
    python_exe = Path(whisperx_python_arg).expanduser().resolve() if whisperx_python_arg else default_whisperx_python()
    helper = Path(__file__).resolve().parent / "whisperx_refine.py"
    try:
        source = audio_path.resolve()
    except OSError:
        return None
    if not all(path.is_file() for path in (source, cli, model, python_exe, helper)):
        return None
    try:
        payload = {
            "kind": "whispercpp-whisperx-local",
            "backend_lineage": "whisper-family",
            "audio_path": str(source),
            "audio_sha256": _stable_file_sha256(source),
            "whisper_cli_path": str(cli),
            "whisper_cli_sha256": _stable_file_sha256(cli),
            "whisper_model_path": str(model),
            "whisper_model_sha256": _stable_file_sha256(model),
            "whisperx_python_path": str(python_exe),
            "whisperx_helper_path": str(helper),
            "whisperx_helper_sha256": _stable_file_sha256(helper),
            "language": str(getattr(args, "whisper_language", "auto") or "auto"),
            "device": str(getattr(args, "whisperx_device", "auto") or "auto"),
            "protocol_revision": _LOCAL_WHISPER_PROTOCOL_REVISION,
        }
    except (OSError, LrcError):
        return None
    return LocalWhisperCapability(
        **payload,
        capability_revision=_canonical_json_digest(
            _local_whisper_capability_semantic_payload(payload)
        ),
    )


def _bounded_retry_capability_semantic_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return the path-free semantic identity for one CTC capability."""
    return {
        key: value
        for key, value in payload.items()
        if key not in {"helper_path", "alignment_audio_path", "capability_revision"}
    }


def _local_whisper_capability_semantic_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return the path-free semantic identity for local Whisper execution."""
    return {
        key: value
        for key, value in payload.items()
        if key not in {
            "audio_path",
            "whisper_cli_path",
            "whisper_model_path",
            "whisperx_python_path",
            "whisperx_helper_path",
            "capability_revision",
        }
    }


def create_bounded_retry_capability_seed(
    alignment_audio: Path,
    audio_source: str,
    helper: Path,
    *,
    sample_rate: int,
    device: str,
) -> dict[str, object]:
    """Create provenance only after the MMS known-lyric generation succeeded."""
    helper = helper.resolve()
    alignment_audio = alignment_audio.resolve()
    helper_sha = _path_sha256(helper)
    protocol_revision = _bounded_retry_protocol_revision(helper_sha)
    payload: dict[str, object] = {
        "kind": "mms-known-lyric-ctc",
        "backend_lineage": "ctc",
        "helper_path": str(helper),
        "helper_sha256": helper_sha,
        "helper_protocol_revision": protocol_revision,
        "alignment_audio_path": str(alignment_audio),
        "alignment_audio_sha256": _path_sha256(alignment_audio),
        "alignment_audio_source": audio_source,
        "sample_rate": sample_rate,
        "model_identity": _BOUNDED_RETRY_MODEL_IDENTITY,
        "device": device,
    }
    return {
        **payload,
        "capability_revision": _canonical_json_digest(
            _bounded_retry_capability_semantic_payload(payload)
        ),
    }


def _attach_bounded_retry_runtime_capability(
    report: dict[str, object],
    seed: dict[str, object],
) -> None:
    """Attach non-serializable in-process authority after a real CTC generation.

    The serialized seed is diagnostic provenance only.  Retry permission must not
    be reconstructible from report metadata loaded or fabricated later.
    """
    try:
        capability = BoundedRetryCapability(**seed)  # type: ignore[arg-type]
    except TypeError as exc:
        raise LrcError(f"Invalid bounded-retry capability seed: {exc}") from exc
    report[_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY] = capability


def carry_bounded_retry_capability_seed(
    target_report: dict[str, object],
    ctc_report: dict[str, object],
) -> None:
    seed = ctc_report.get("ctc_bounded_retry_capability_seed")
    if not isinstance(seed, dict):
        return
    target_report["ctc_bounded_retry_capability_seed"] = copy.deepcopy(seed)
    authority = ctc_report.get(_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY)
    if isinstance(authority, BoundedRetryCapability):
        target_report[_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY] = authority
    for key in (
        "ctc_backend", "ctc_device", "ctc_sample_rate", "ctc_audio_source", "ctc_audio_path"
    ):
        target_report[key] = ctc_report.get(key)


def build_bounded_retry_capability(
    report: dict[str, object],
    audio_path: Path,
    args: argparse.Namespace,
) -> BoundedRetryCapability | None:
    """Verify a generation-carried capability; metadata alone is never enough."""
    if str(report.get("backend") or "") not in {"ctc", "hybrid"}:
        return None
    seed = report.get("ctc_bounded_retry_capability_seed")
    authority = report.get(_RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY)
    if not isinstance(seed, dict) or not isinstance(authority, BoundedRetryCapability):
        return None
    seed_payload = _bounded_retry_capability_semantic_payload(seed)
    try:
        if seed.get("capability_revision") != _canonical_json_digest(seed_payload):
            return None
    except LrcError:
        return None
    helper = Path(__file__).resolve().parent / "ctc_align.py"
    if seed.get("kind") != "mms-known-lyric-ctc" or seed.get("backend_lineage") != "ctc":
        return None
    try:
        effective_audio = audio_path.resolve()
        configured_path = Path(str(seed.get("alignment_audio_path"))).resolve()
        seeded_helper = Path(str(seed.get("helper_path"))).resolve()
    except (OSError, TypeError, ValueError):
        return None
    if (
        configured_path != effective_audio
        or seeded_helper != helper.resolve()
        or not helper.is_file()
        or not effective_audio.is_file()
    ):
        return None
    try:
        sample_rate = int(seed.get("sample_rate") or 0)
    except (TypeError, ValueError):
        return None
    helper_sha = _path_sha256(helper)
    if (
        sample_rate != SAMPLE_RATE
        or seed.get("helper_sha256") != helper_sha
        or seed.get("helper_protocol_revision") != _bounded_retry_protocol_revision(helper_sha)
        or seed.get("alignment_audio_sha256") != _path_sha256(effective_audio)
        or seed.get("model_identity") != _BOUNDED_RETRY_MODEL_IDENTITY
        or not str(seed.get("device") or "").strip()
    ):
        return None
    source = str(seed.get("alignment_audio_source") or "")
    if not source:
        return None
    try:
        seeded_capability = BoundedRetryCapability(**seed)  # type: ignore[arg-type]
    except TypeError:
        return None
    if authority != seeded_capability:
        return None
    return authority


def inactive_legacy_timing_diagnostics() -> dict[str, object]:
    return {
        "legacy_boundary_mutation_pipeline": {
            "status": "inactive",
            "executed": False,
            "authoritative": False,
            "inactive_stages": ["zero-gap", "crossline-initial", "forward-prefix", "weak-prefix"],
            "raw_existing_evidence_preserved": True,
            "evidence_scope": "historical-nonauthoritative",
        },
        "legacy_prefix_probe_candidate_generation": {
            "status": "inactive",
            "executed": False,
            "authoritative": False,
            "active_candidate_count": 0,
            "raw_existing_evidence_preserved": True,
            "evidence_scope": "historical-nonauthoritative",
        },
    }


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
