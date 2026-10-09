"""Production S1 structural CTC candidate producer, without timing authority.

The only side effect is a temporary transcript/output directory for bounded
GPU CTC jobs. It never writes LRC, never reads checked/BK clocks, and never
promotes an onset/occurrence claim on Central's behalf.
"""
from __future__ import annotations

import copy
import json
import math
from collections import OrderedDict
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from segment_anchor_inference import (
    find_unambiguous_repeated_section,
    infer_repeated_passage_anchors,
    qualify_structural_repair,
)
from segment_central_bridge import stage_segment_constrained_ctc
from segment_recurrence import vocal_cepstral_features
from segment_structure import assign_lyrics_to_vocal_passages, merge_vocal_passages


# Bounded process-local reuse for baseline/final Central previews. The key
# includes source file metadata, lyrics, priors, and helper identity; no BK.
# Cached spans remain just untrusted hypotheses and pass bridge checks again.
_RUNTIME_S1_CACHE: OrderedDict[tuple[object, ...], tuple[list[dict[str, object]], dict[str, object]]] = OrderedDict()


def _block(status: str, **context: object) -> dict[str, object]:
    return {"status": status, "admitted_by_central": False, "candidate_kind": "segment-constrained-ctc", **context}


def try_stage_structural_ctc(
    *,
    report: dict[str, object],
    lyric_texts: list[str],
    vocal_stem: Path,
    duration: float,
    audio_api: object,
    python_exe: Path,
    ctc_helper: Path,
    device: str,
    max_bundles: int = 16,
    helper_runner: Callable[..., object] = subprocess.run,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return a cloned in-memory report with bounded hypotheses, or unchanged.

    A full song of genuine CTC spans is necessary even when only one repeat
    is wrong, because the Central selector must see an ordered candidate
    sequence. Weak CTC rows remain weak; this producer never provides independent
    lyric-identity evidence.
    """
    if report.get("backend") != "ctc" or report.get("ctc_audio_source") != "vocal-stem":
        return report, _block("BLOCKED_NOT_ISOLATED_VOCAL_CTC")
    assignments = report.get("assignments")
    if not (
        isinstance(assignments, list)
        and len(assignments) == len(lyric_texts) >= 6
        and math.isfinite(duration) and 20.0 <= duration <= 600.0
    ):
        return report, _block("BLOCKED_INVALID_AUDIO_OR_LYRICS")
    section, reason = find_unambiguous_repeated_section(lyric_texts)
    if section is None:
        return report, _block(reason)
    if not vocal_stem.is_file():
        return report, _block("BLOCKED_VOCAL_STEM_MISSING")
    if not python_exe.is_file() or not ctc_helper.is_file():
        return report, _block("BLOCKED_CTC_HELPER_MISSING")
    # Existing global CTC times are only weak positional priors and may not
    # identify which repeated phrase produced the observed path.
    priors: list[float | None] = []
    for row in assignments:
        if not isinstance(row, dict):
            return report, _block("BLOCKED_ASSIGNMENT_SHAPE")
        source = row.get("ctc_first_token_start")
        if not isinstance(source, (int, float)) or not math.isfinite(float(source)):
            source = row.get("timestamp")
        priors.append(float(source) if isinstance(source, (int,float)) and math.isfinite(float(source)) else None)
    try:
        stem_stat = vocal_stem.stat()
        helper_stat = ctc_helper.stat()
        interpreter_stat = python_exe.stat()
        cache_key: tuple[object, ...] = (
            "segment-s1-vocal-local-ctc-v2",
            str(vocal_stem.resolve()), stem_stat.st_size, stem_stat.st_mtime_ns,
            str(ctc_helper.resolve()), helper_stat.st_size, helper_stat.st_mtime_ns,
            str(python_exe.resolve()), interpreter_stat.st_size, interpreter_stat.st_mtime_ns,
            float(duration), tuple(lyric_texts), tuple(priors), str(device),
            max_bundles, id(audio_api), id(helper_runner),
        )
        cached = _RUNTIME_S1_CACHE.get(cache_key)
        if cached is not None:
            frozen_rows, frozen_diag = cached
            staged, status = stage_segment_constrained_ctc(
                report, copy.deepcopy(frozen_rows), entry_count=len(lyric_texts),
                source_revision=str(frozen_diag["source_revision"]),
            )
            if status == "STAGED_FOR_CENTRAL_NO_AUTHORITY":
                _RUNTIME_S1_CACHE.move_to_end(cache_key)
                return staged, {**copy.deepcopy(frozen_diag), "runtime_cache_hit": True}
            _RUNTIME_S1_CACHE.pop(cache_key, None)
        audio = audio_api.decode_audio(vocal_stem)
        frames = audio_api.frame_audio(audio)
        features = audio_api.analyze_audio(vocal_stem, duration)
        passages = merge_vocal_passages(features.segments, duration=duration)
        timbre, levels = vocal_cepstral_features(frames, audio_api.SAMPLE_RATE)
        inferred = infer_repeated_passage_anchors(
            lyric_texts, passages, timbre, levels, hop_seconds=audio_api.HOP_SECONDS,
        )
        if inferred.status != "RESEARCH_UNVERIFIED_OCCURRENCE":
            return report, _block(inferred.status, anchor_count=0)
        lengths = [max(1, len(text)) for text in lyric_texts]
        dp = assign_lyrics_to_vocal_passages(
            [(p.start, p.end) for p in passages],
            duration=duration,
            lyric_lengths=lengths,
            ctc_priors=priors,
            anchored_bundles=inferred.anchors,
        )
        if dp.status != "UNVERIFIED_OCCURRENCE":
            return report, _block(dp.status)
        qualified = qualify_structural_repair(inferred.anchors, dp.entry_times, priors)
        if qualified.status != "RESEARCH_TRIGGER_UNVERIFIED_OCCURRENCE":
            return report, _block(qualified.status, medians_s=qualified.medians_s)
        if not 1 <= len(dp.bundles) <= max_bundles:
            return report, _block("BLOCKED_GPU_BUNDLE_BUDGET", bundles=len(dp.bundles))
        input_rows: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="lrc13-structural-ctc-") as temporary:
            folder = Path(temporary)
            for index, bundle in enumerate(dp.bundles):
                vocal = dp.passages[bundle.passage_index]
                transcript = folder / f"passage_{index:02d}.lyrics.json"
                output = folder / f"passage_{index:02d}.ctc.json"
                transcript.write_text(
                    json.dumps(lyric_texts[bundle.start_row - 1:bundle.end_row],
                               ensure_ascii=False),
                    encoding="utf-8",
                )
                command = [
                    str(python_exe), str(ctc_helper),
                    "--audio", str(vocal_stem),
                    "--transcript", str(transcript),
                    "--output", str(output),
                    "--device", device,
                    "--start", str(round(max(0., vocal.start - 0.3), 3)),
                    "--end", str(round(min(duration, vocal.end + 0.3), 3)),
                ]
                result = helper_runner(command, capture_output=True, text=True,
                                       errors="replace", timeout=180)
                if result.returncode != 0 or not output.is_file():
                    return report, _block(
                        "BLOCKED_LOCAL_CTC_FAILED", passage=index,
                        error=str(getattr(result,"stderr",""))[-240:],
                    )
                payload = json.loads(output.read_text(encoding="utf-8"))
                values = payload.get("entries")
                if not isinstance(values, list) or len(values) != bundle.end_row - bundle.start_row + 1:
                    return report, _block("BLOCKED_LOCAL_CTC_ROW_COUNT", passage=index)
                for j, row in enumerate(values):
                    if not isinstance(row, dict):
                        return report, _block("BLOCKED_LOCAL_CTC_ROW_SHAPE")
                    input_rows.append({
                        "row": bundle.start_row + j,
                        "time": row.get("start"),
                        "ctc_score": row.get("ctc_score"),
                        "token_spans": row.get("token_spans"),
                        "passage": bundle.passage_index,
                    })
            # Test low-support lyrics at neighboring passage edges using
            # alternate genuine CUDA paths. No reference clocks and no
            # duplicated lines are permitted to enter Central.
            from segment_boundary_search import compete_ctc_adjacent_boundaries
            refined, boundary_diag = compete_ctc_adjacent_boundaries(
                input_rows, dp.bundles, dp.passages, lyric_texts, inferred.anchors,
                stem=vocal_stem, duration=duration, python_exe=python_exe,
                helper=ctc_helper, device=device, temp_dir=folder,
                runner=helper_runner,
            )
            if refined is None:
                return report, _block(str(boundary_diag["status"]))
            input_rows = refined
        # The digest binds ONLY evidence/source conditions, not any asserted
        # occurrence or independent onset. Central must still verify candidates.
        import hashlib
        source_revision = hashlib.sha256(json.dumps({
            "source": "segment-vad-recurrence-dp-local-mms-v1",
            "anchors": inferred.anchors,
            "windows": [(round(b.start, 3), round(b.end, 3)) for b in dp.passages],
            "rows": input_rows,
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        staged, status = stage_segment_constrained_ctc(
            report, input_rows, entry_count=len(lyric_texts),
            source_revision=source_revision,
        )
        if status != "STAGED_FOR_CENTRAL_NO_AUTHORITY":
            return report, _block(status)
        diagnostics = {
            "status": status,
            "candidate_kind": "segment-constrained-ctc",
            "admitted_by_central": False,
            "bundles": len(dp.bundles),
            "rows": len(input_rows),
            "anchors": inferred.anchors,
            "acoustic_pair_score": inferred.evidence[0].similarity if inferred.evidence else None,
            "median_offset_s": qualified.medians_s,
            "source_revision": source_revision,
            "boundary_competition": boundary_diag,
            "runtime_only": True,
            "runtime_cache_hit": False,
        }
        _RUNTIME_S1_CACHE[cache_key] = (copy.deepcopy(input_rows), copy.deepcopy(diagnostics))
        _RUNTIME_S1_CACHE.move_to_end(cache_key)
        while len(_RUNTIME_S1_CACHE) > 2:
            _RUNTIME_S1_CACHE.popitem(last=False)
        return staged, diagnostics
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyError, TypeError) as exc:
        return report, _block("BLOCKED_S1_PRODUCER_EXCEPTION",
                              exception_type=type(exc).__name__,
                              detail=str(exc)[:240])