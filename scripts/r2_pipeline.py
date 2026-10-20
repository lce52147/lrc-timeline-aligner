from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import runtime_paths  # noqa: E402

PROJECT = runtime_paths.project_root()
REVIEWERS_DIR = SCRIPTS / "reviewers"
AUTO_LRC = SCRIPTS / "auto_lrc.py"
DEFAULT_WORK_DIR = PROJECT / "outputs" / "reviewer-work"
REVIEWERS = ("HUBP", "WX", "XLSR")
MIN_OFFSET_SAMPLES = 8

import auto_lrc  # noqa: E402
import model_cache  # noqa: E402
from lrc_manual_edits import manual_edit_reason  # noqa: E402
from lrc_publication import journal_path  # noqa: E402
import reviewer_layer  # noqa: E402
import reviewer_independent_offset as vocal_offset  # noqa: E402

TRUST_REVIEWERS = reviewer_layer.SHIPPED_REVIEWERS


class MissingCurrentCandidateError(ValueError):
    pass


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    forbidden = set('<>:"/\\|?*')
    cleaned = "".join("_" if char in forbidden else char for char in value).strip().rstrip(".")
    return cleaned or "song"


def reviewer_work_directory(base: Path, audio_path: Path) -> Path:
    """Keep reviewer artifacts from identically named songs independent."""
    canonical_audio = str(audio_path.expanduser().resolve()).casefold()
    suffix = hashlib.sha256(canonical_audio.encode("utf-8")).hexdigest()[:12]
    return base / f"{_safe_name(audio_path.stem)}--{suffix}"


def _current_candidate(assignment: Mapping[str, object]) -> Mapping[str, object]:
    candidates = assignment.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("baseline assignment has no candidate list")
    currents = [
        item
        for item in candidates
        if isinstance(item, Mapping) and item.get("source") == "ctc-current"
    ]
    if len(currents) != 1:
        raise MissingCurrentCandidateError(
            "baseline assignment must contain exactly one ctc-current candidate"
        )
    return currents[0]


def build_reviewer_payload(report: Mapping[str, object]) -> dict[str, object]:
    lyrics_value = report.get("lyrics_path")
    vocal_value = report.get("ctc_audio_path")
    audio_value = report.get("audio_path")
    assignments = report.get("assignments")
    if not isinstance(lyrics_value, str) or not isinstance(vocal_value, str):
        raise ValueError("baseline report is missing lyric or vocal-audio provenance")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("baseline report has no assignments")

    lyrics_path = Path(lyrics_value)
    vocal_path = Path(vocal_value)
    audio_path = (
        Path(audio_value)
        if isinstance(audio_value, str)
        else lyrics_path.with_suffix(".flac")
    )
    if not lyrics_path.is_file():
        raise FileNotFoundError(f"lyric source unavailable: {lyrics_path}")
    if not vocal_path.is_file():
        raise FileNotFoundError(f"vocal source unavailable: {vocal_path}")

    document = auto_lrc.load_lyrics(lyrics_path)
    entries = auto_lrc.remove_generated_title_cards(document.entries, audio_path)
    entries, _skipped = auto_lrc.remove_instrumental_markers(entries)
    if len(entries) != len(assignments):
        raise ValueError(
            f"baseline lyric/report row mismatch: {len(entries)} != {len(assignments)}"
        )

    rows: list[dict[str, object]] = []
    for expected_entry, (entry, assignment) in enumerate(
        zip(entries, assignments, strict=True), start=1
    ):
        if not isinstance(assignment, Mapping) or assignment.get("entry") != expected_entry:
            raise ValueError(f"invalid baseline assignment at entry {expected_entry}")
        current = _current_candidate(assignment)
        cur_time = _finite(current.get("time"))
        fin_time = _finite(assignment.get("timestamp"))
        mms_time = _finite(assignment.get("ctc_first_token_start"))
        if cur_time is None or fin_time is None or mms_time is None:
            raise ValueError(f"baseline canonical timing missing at entry {expected_entry}")
        rows.append(
            {
                "entry": expected_entry,
                "lines": list(entry.lines),
                "text": auto_lrc.entry_sung_text(entry),
                "sources": {
                    "MMS": {"time": mms_time},
                    "CUR": {"time": cur_time, "role": "baseline"},
                    "FIN": {"time": fin_time, "role": "proposal"},
                },
            }
        )

    language_code = auto_lrc.infer_spoken_language(entries)
    language = {"ja": "japanese", "en": "english", "zh": "chinese"}.get(
        language_code, language_code
    )
    song_id = _safe_name(audio_path.stem)
    song = {
        "id": song_id,
        "title": audio_path.stem,
        "language": language,
        "audio_path": str(audio_path),
        "vocal_audio_path": str(vocal_path),
        "vocal_audio_sha256": _file_digest(vocal_path),
        "row_count": len(rows),
        "rows": rows,
    }
    return {
        "schema": 1,
        "stage": "reviewer-production-single",
        "partition": "production-single",
        "status": "BASE_EXTRACTED",
        "song_count": 1,
        "row_count": len(rows),
        "sources": {
            "MMS": {"status": "READY_FROM_BASELINE_REPORT"},
            "CUR": {"status": "READY_FROM_BASELINE_REPORT"},
            "FIN": {"status": "READY_FROM_BASELINE_REPORT"},
            "HUBP": {"status": "PENDING"},
            "WX": {"status": "PENDING"},
            "XLSR": {"status": "PENDING"},
        },
        "songs": [song],
    }


def _source_time(row: Mapping[str, object], name: str) -> float | None:
    sources = row.get("sources")
    item = sources.get(name) if isinstance(sources, Mapping) else None
    return _finite(item.get("time")) if isinstance(item, Mapping) else None


def _hubp_time(hubp_rows: Mapping[str, object], song_id: str, entry: int) -> float | None:
    item = hubp_rows.get(f"{song_id}::{entry}")
    if not isinstance(item, Mapping) or item.get("status") != "OK":
        return None
    return _finite(item.get("time"))


def build_r2_sidecar(
    reviewer_payload: Mapping[str, object], hubp_payload: Mapping[str, object]
) -> dict[str, object]:
    songs = reviewer_payload.get("songs")
    if not isinstance(songs, list) or len(songs) != 1 or not isinstance(songs[0], Mapping):
        raise ValueError("R2 production sidecar requires exactly one song")
    song = songs[0]
    song_id = str(song.get("id", "song"))
    rows = song.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("reviewer payload has no rows")
    raw_hubp_rows = hubp_payload.get("rows")
    hubp_rows = raw_hubp_rows if isinstance(raw_hubp_rows, Mapping) else {}

    row_values: list[dict[str, object]] = []
    current_times: list[float | None] = []
    reviewer_series: dict[str, list[float | None]] = {name: [] for name in REVIEWERS}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("entry"), int):
            raise ValueError("invalid reviewer row")
        entry = int(row["entry"])
        current = _source_time(row, "CUR")
        if current is None:
            raise ValueError(f"reviewer row missing CUR: {entry}")
        times = {
            "HUBP": _hubp_time(hubp_rows, song_id, entry),
            "WX": _source_time(row, "WX"),
            "XLSR": _source_time(row, "XLSR"),
        }
        current_times.append(current)
        for name in REVIEWERS:
            reviewer_series[name].append(times[name])
        row_values.append(
            {
                "entry": entry,
                "expected_current_seconds": current,
                "reviewer_times": times,
            }
        )

    # S2 independent calibration: NEVER derive reviewer offsets from CUR,
    # FIN, Central's chosen time, or manual BK. Acoustic onsets are a separate
    # witness from the Demucs-isolated vocal audio.
    offsets: dict[str, float] = {}
    sample_counts: dict[str, int] = {name: 0 for name in REVIEWERS}
    offset_status: dict[str, str] = {name: "VOCAL_STEM_UNAVAILABLE" for name in REVIEWERS}
    vocal_path_value = song.get("vocal_audio_path")
    vocal_path = Path(vocal_path_value) if isinstance(vocal_path_value, str) else None
    if vocal_path is not None and vocal_path.is_file():
        try:
            vocal_duration = auto_lrc.probe_duration(vocal_path)
            features = auto_lrc.analyze_vocal_onsets(vocal_path, vocal_duration)
            peaks = vocal_offset.anchor_peaks_from_vocal_features(
                features.frame_times, features.rms_db, features.onset_strength
            )
            for name in REVIEWERS:
                observation = vocal_offset.calibrate_reviewer_vocal_offset(
                    reviewer_series[name], peaks, min_samples=MIN_OFFSET_SAMPLES
                )
                sample_counts[name] = observation.sample_count
                offset_status[name] = observation.status
                if observation.offset_seconds is not None:
                    offsets[name] = observation.offset_seconds
        except (OSError, ValueError, auto_lrc.LrcError) as exc:
            offset_status = {
                name: f"VOCAL_ANALYSIS_FAILED:{type(exc).__name__}" for name in REVIEWERS
            }

    return {
        "schema": 1,
        "policy": "R2b",
        "reviewers": list(REVIEWERS),
        "required_agreements": 1,
        "offsets": offsets,
        "offset_sample_counts": sample_counts,
        "offset_calibration_method": "independent-isolated-vocal-onsets-v1",
        "offset_calibration_status": offset_status,
        "offset_calibration_uses_central": False,
        "rows": row_values,
    }


def build_reviewer_trust_report(
    reviewer_payload: Mapping[str, object],
    hubp_payload: Mapping[str, object],
    final_times: Sequence[float],
    *,
    profile: str | None = None,
) -> dict[str, object]:
    config = reviewer_layer.resolve_trust_profile(profile)
    songs = reviewer_payload.get("songs")
    if not isinstance(songs, list) or len(songs) != 1 or not isinstance(songs[0], Mapping):
        raise ValueError("reviewer trust report requires exactly one song")
    song = songs[0]
    rows = song.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("reviewer payload has no rows")
    if len(rows) != len(final_times):
        raise ValueError(f"reviewer/final row mismatch: {len(rows)} != {len(final_times)}")

    language = str(song.get("language", "unknown"))
    song_id = str(song.get("id", "song"))
    raw_hubp_rows = hubp_payload.get("rows")
    hubp_rows = raw_hubp_rows if isinstance(raw_hubp_rows, Mapping) else {}

    normalized_final: list[float] = []
    reviewer_rows: list[dict[str, float | None]] = []
    reviewer_series: dict[str, list[float | None]] = {name: [] for name in TRUST_REVIEWERS}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("entry"), int):
            raise ValueError("invalid reviewer row")
        final_time = _finite(final_times[index])
        if final_time is None:
            raise ValueError(f"invalid final R2 timestamp at entry {index + 1}")
        entry = int(row["entry"])
        times = {
            "HUBP": _hubp_time(hubp_rows, song_id, entry),
            "WX": _source_time(row, "WX"),
            "XLSR": _source_time(row, "XLSR"),
            "HUB": _source_time(row, "HUB"),
        }
        normalized_final.append(final_time)
        reviewer_rows.append(times)
        for name in TRUST_REVIEWERS:
            reviewer_series[name].append(times[name])

    # Reviewer-trust is not allowed to calibrate on its OWN final output.
    # Reuse the same independent audio-onset-only estimator as R2b.
    offsets: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    offset_status: dict[str, str] = {}
    vocal_value = song.get("vocal_audio_path")
    vocal_file = Path(vocal_value) if isinstance(vocal_value, str) else None
    peaks: tuple[float, ...] = ()
    if vocal_file is not None and vocal_file.is_file():
        try:
            features = auto_lrc.analyze_vocal_onsets(vocal_file, auto_lrc.probe_duration(vocal_file))
            peaks = vocal_offset.anchor_peaks_from_vocal_features(
                features.frame_times, features.rms_db, features.onset_strength
            )
        except (OSError, ValueError, auto_lrc.LrcError):
            peaks = ()
    for name in TRUST_REVIEWERS:
        offset = vocal_offset.calibrate_reviewer_vocal_offset(
            reviewer_series[name], peaks, min_samples=MIN_OFFSET_SAMPLES
        )
        sample_counts[name] = offset.sample_count
        offset_status[name] = offset.status
        if offset.offset_seconds is not None:
            offsets[name] = offset.offset_seconds

    output_rows: list[dict[str, object]] = []
    trusted_count = 0
    for row, final_time, times in zip(rows, normalized_final, reviewer_rows, strict=True):
        decision = reviewer_layer.classify_row(
            final_time=final_time,
            reviewer_times=times,
            offsets=offsets,
            reviewers=TRUST_REVIEWERS,
        )
        trusted = bool(
            language == "japanese"
            and reviewer_layer.trusted_by_profile(decision, profile=config.name)
        )
        if trusted:
            trusted_count += 1
        if not config.enabled:
            reason = config.reason
        elif language != "japanese":
            reason = "non-japanese-fail-closed"
        elif decision.missing_reviewers:
            reason = "missing-required-reviewer"
        elif trusted:
            reason = "profile-rule-met"
        else:
            reason = "profile-rule-not-met"
        output_rows.append(
            {
                "entry": int(row["entry"]),
                "final_time": final_time,
                "reviewer_trusted": trusted,
                "reason": reason,
                "label": decision.label,
                "present_count": decision.present_count,
                "agree_count": decision.agree_count,
                "missing_reviewers": list(decision.missing_reviewers),
                "agreeing_reviewers": list(decision.agreeing_reviewers),
            }
        )

    return {
        "schema": 1,
        "profile": config.name,
        "selected_rule_id": config.selected_rule_id,
        "profile_enabled": config.enabled,
        "language": language,
        "reviewers": list(TRUST_REVIEWERS),
        "offsets": offsets,
        "offset_sample_counts": sample_counts,
        "offset_calibration_status": offset_status,
        "offset_calibration_method": "independent-isolated-vocal-onsets-v1",
        "offset_calibration_uses_final_output": False,
        "trusted_count": trusted_count,
        "review_count": len(output_rows) - trusted_count,
        "rows": output_rows,
    }


def final_times_from_report(report_path: Path) -> list[float]:
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, Mapping):
        raise ValueError("final report must be a JSON object")
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise ValueError("final report has no assignments")
    result: list[float] = []
    for index, assignment in enumerate(assignments, start=1):
        if not isinstance(assignment, Mapping):
            raise ValueError(f"invalid final assignment at entry {index}")
        value = _finite(assignment.get("timestamp"))
        if value is None:
            raise ValueError(f"final assignment missing timestamp at entry {index}")
        result.append(value)
    return result


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_message(stage: int, message: str) -> str:
    if stage not in (1, 2, 3, 4):
        raise ValueError(f"invalid R2 stage: {stage}")
    return f"[{stage}/4] {message}"


def _format_elapsed_seconds(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(float(elapsed_seconds) + 0.5))
    minutes, seconds = divmod(total_seconds, 60)
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def stage_done_message(stage: int, elapsed_seconds: float) -> str:
    return stage_message(stage, f"完成，用時 {_format_elapsed_seconds(elapsed_seconds)}")


def build_lyric_review_messages(
    report: Mapping[str, object], reviewer_payload: Mapping[str, object], *, limit: int = 5
) -> list[str]:
    review_required = report.get("review_required_count")
    low_confidence = report.get("low_confidence_count")
    review_count = int(review_required) if isinstance(review_required, int) else 0
    low_count = int(low_confidence) if isinstance(low_confidence, int) else 0
    messages = [
        f"歌詞檢查：Review required {review_count} 行；Low confidence {low_count} 行。",
        "人工檢查線索（WA 校準：ctc_score 最低 5 行；內部 confidence 不代表準確率）：",
    ]

    songs = reviewer_payload.get("songs")
    song = songs[0] if isinstance(songs, list) and len(songs) == 1 else None
    rows = song.get("rows") if isinstance(song, Mapping) else None
    text_by_entry: dict[int, str] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            entry = row.get("entry")
            if isinstance(entry, int):
                text_by_entry[entry] = str(row.get("text", ""))

    assignments = report.get("assignments")
    scored: list[tuple[float, int, float | None]] = []
    if isinstance(assignments, list):
        for assignment in assignments:
            if not isinstance(assignment, Mapping):
                continue
            entry = assignment.get("entry")
            score = _finite(assignment.get("ctc_score"))
            if not isinstance(entry, int) or score is None:
                continue
            scored.append((score, entry, _finite(assignment.get("ctc_first_token_score"))))
    scored.sort(key=lambda item: (item[0], item[1]))

    for score, entry, first_score in scored[: max(0, int(limit))]:
        first = "" if first_score is None else f" ctc_first_token_score={first_score:.6f}"
        text = text_by_entry.get(entry, "")[:30]
        messages.append(f"#{entry} ctc_score={score:.6f}{first} {text}")
    return messages


def build_sparse_reviewer_notice(
    reviewer_payload: Mapping[str, object], hubp_payload: Mapping[str, object]
) -> str | None:
    songs = reviewer_payload.get("songs")
    if not isinstance(songs, list) or len(songs) != 1 or not isinstance(songs[0], Mapping):
        return None
    song = songs[0]
    song_id = str(song.get("id", "song"))
    rows = song.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    raw_hubp_rows = hubp_payload.get("rows")
    hubp_rows = raw_hubp_rows if isinstance(raw_hubp_rows, Mapping) else {}

    reviewers_with_values: set[str] = set()
    rows_with_values = 0
    try:
        sidecar = build_r2_sidecar(reviewer_payload, hubp_payload)
    except ValueError:
        sidecar = None

    if isinstance(sidecar, Mapping):
        raw_offsets = sidecar.get("offsets")
        qualified = {
            str(name)
            for name in REVIEWERS
            if isinstance(raw_offsets, Mapping) and _finite(raw_offsets.get(name)) is not None
        }
        reviewers_with_values.update(qualified)
        sidecar_rows = sidecar.get("rows")
        if isinstance(sidecar_rows, list):
            for row in sidecar_rows:
                if not isinstance(row, Mapping):
                    continue
                reviewer_times = row.get("reviewer_times")
                if not isinstance(reviewer_times, Mapping):
                    continue
                if any(_finite(reviewer_times.get(name)) is not None for name in qualified):
                    rows_with_values += 1
    else:
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            entry = row.get("entry")
            if not isinstance(entry, int):
                continue
            values = {
                "HUBP": _hubp_time(hubp_rows, song_id, entry),
                "WX": _source_time(row, "WX"),
                "XLSR": _source_time(row, "XLSR"),
            }
            present = [name for name, value in values.items() if value is not None]
            reviewers_with_values.update(present)
            if present:
                rows_with_values += 1

    row_count = len(rows)
    if len(reviewers_with_values) >= 2 or rows_with_values / row_count >= 0.5:
        return None
    return (
        "此歌曲的 reviewer 證據不足（日文專用模型）：R2 背書與可信標記對此歌曲幾乎不起作用；"
        "Trusted timing 等百分比是內部指標，不代表準確度"
        f"（有值 reviewer {len(reviewers_with_values)}/3；有值行 {rows_with_values}/{row_count}）。"
    )

EXIT_TIMEOUT = 124
EXIT_LAUNCH_ERROR = 127
EXIT_STRICT_REVIEWERS = 4
DEFAULT_SUBPROCESS_TIMEOUT_SECONDS = 7200.0


def subprocess_timeout_seconds(environ: Mapping[str, str] | None = None) -> float | None:
    """``LRC_SUBPROCESS_TIMEOUT_SECONDS`` (default 7200); ``0`` disables the limit."""
    raw = (environ if environ is not None else os.environ).get("LRC_SUBPROCESS_TIMEOUT_SECONDS", "")
    if not str(raw).strip():
        return DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
    return value if value > 0 else None


def _popen_group_kwargs() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def kill_process_tree(process: subprocess.Popen[object]) -> None:
    """Kill ``process`` and everything it spawned (taskkill /T on Windows, the
    session/process group elsewhere)."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        else:
            import signal
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    finally:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def _run_with_timeout(
    command: Sequence[str],
    *,
    timeout: float | None,
    log=None,
    env: Mapping[str, str] | None = None,
) -> int:
    """Run ``command`` to completion, returning its exit code, ``EXIT_TIMEOUT``
    (124) after killing the whole tree on timeout, or ``EXIT_LAUNCH_ERROR``
    (127) when it could not be started."""
    def note(text: str) -> None:
        if log is not None:
            log.write(text + "\n")
            log.flush()
    try:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT,
            env=dict(env) if env is not None else None,
            stdout=log if log is not None else None,
            stderr=subprocess.STDOUT if log is not None else None,
            **_popen_group_kwargs(),
        )
    except OSError as exc:
        note(f"launch-error: {type(exc).__name__}: {exc}")
        return EXIT_LAUNCH_ERROR
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(process)
        note(f"timeout: killed pid {process.pid} after {timeout}s")
        return EXIT_TIMEOUT
    return int(process.returncode)


def _run_logged(command: Sequence[str], log_path: Path, *, timeout: float | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if timeout is None:
        timeout = subprocess_timeout_seconds()
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n$ " + json.dumps(list(command), ensure_ascii=False) + "\n")
        log.flush()
        return _run_with_timeout(command, timeout=timeout, log=log, env=env)


def build_reviewer_commands(
    *,
    data_path: Path,
    plan_path: Path,
    hubp_path: Path,
    song_id: str,
    python_path: Path,
    device: str,
    include_hub: bool = False,
) -> list[tuple[str, list[str]]]:
    commands = [
        (
            "HUBP_PLAN",
            [
                str(python_path), str(REVIEWERS_DIR / "build_hubp_plan.py"),
                "--data", str(data_path), "--output", str(plan_path), "--song-id", song_id,
            ],
        ),
        (
            "HUBP",
            [
                str(python_path), str(REVIEWERS_DIR / "run_hubp.py"),
                "--plan", str(plan_path), "--output", str(hubp_path), "--device", device,
            ],
        ),
        (
            "WX",
            [
                str(python_path), str(REVIEWERS_DIR / "run_whisperx.py"),
                "--data", str(data_path), "--song-id", song_id, "--whisperx-device", device,
            ],
        ),
        (
            "XLSR",
            [
                str(python_path), str(REVIEWERS_DIR / "run_xlsr.py"),
                "--data", str(data_path), "--song-id", song_id, "--device", device,
            ],
        ),
    ]
    if include_hub:
        commands.append(
            (
                "HUB",
                [
                    str(python_path), str(REVIEWERS_DIR / "run_hub.py"),
                    "--data", str(data_path), "--song-id", song_id, "--device", device,
                ],
            )
        )
    return commands


REVIEWER_STATUSES = (
    "OK", "FAILED", "MISSING_UNSUPPORTED", "SKIPPED_DEPENDENCY", "TIMEOUT", "LAUNCH_ERROR",
)
REVIEWER_ACCEPTABLE_STATUSES = frozenset({"OK", "MISSING_UNSUPPORTED", "SKIPPED_DEPENDENCY"})


def reviewer_step_status(name: str, exit_code: int, payload: Mapping[str, object] | None = None) -> str:
    if exit_code == 0:
        return "OK"
    if exit_code == EXIT_TIMEOUT:
        return "TIMEOUT"
    if exit_code == EXIT_LAUNCH_ERROR:
        return "LAUNCH_ERROR"
    if name == "HUB" and exit_code == 2 and isinstance(payload, Mapping):
        songs = payload.get("songs")
        if isinstance(songs, list) and songs:
            hub_states = [
                song.get("HUB")
                for song in songs
                if isinstance(song, Mapping)
            ]
            if hub_states and all(
                isinstance(state, Mapping) and state.get("status") == "UNSUPPORTED_TEXT"
                for state in hub_states
            ):
                return "MISSING_UNSUPPORTED"
    return "FAILED"


REVIEWER_SIDECAR_CACHE_KIND = "r2-reviewer-sidecar"
_CACHEABLE_REVIEWER_STATUSES = frozenset({"OK", "MISSING_UNSUPPORTED"})


def _sha256_or_missing(path: Path) -> str:
    try:
        return auto_lrc._stable_file_sha256(Path(path))
    except OSError:
        return "missing"


def reviewers_source_sha256(reviewers_dir: Path = REVIEWERS_DIR) -> str:
    """sha256 over the reviewer scripts, sorted by file name."""
    digest = hashlib.sha256()
    for path in sorted(Path(reviewers_dir).glob("*.py"), key=lambda item: item.name):
        if path.name.startswith("test_"):
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def reviewer_sidecar_identity(
    payload: Mapping[str, object],
    *,
    device: str,
    include_hub: bool,
    reviewers_dir: Path = REVIEWERS_DIR,
) -> dict[str, object]:
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    return model_cache.identity_r2_reviewer_sidecar(
        payload_sha256=hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
        reviewers_sha256=reviewers_source_sha256(reviewers_dir),
        whisper_cli_sha256=_sha256_or_missing(runtime_paths.whisper_cli()),
        whisper_model_sha256=_sha256_or_missing(runtime_paths.whisper_model()),
        device=str(device),
        include_hub=bool(include_hub),
    )


def _reviewer_statuses_cacheable(statuses: Mapping[str, object]) -> bool:
    if not statuses:
        return False
    for state in statuses.values():
        status = state.get("status") if isinstance(state, Mapping) else None
        if status not in _CACHEABLE_REVIEWER_STATUSES:
            return False
    return True


def _reviewer_cache_payload_complete(cached: object) -> bool:
    """A cached sidecar entry must carry every file stage 4 re-reads.

    Entries written before reviewer_data was cached are treated as misses so a
    fill run recomputes and overwrites them (cache-only mode then reports the
    miss instead of failing later in stage 4).
    """
    if not isinstance(cached, Mapping):
        return False
    if not isinstance(cached.get("sidecar"), str) or not isinstance(cached.get("statuses"), Mapping):
        return False
    if not isinstance(cached.get("reviewer_data"), str):
        return False
    hubp = cached.get("hubp_raw")
    return hubp is None or isinstance(hubp, str)


def acquire_reviewer_evidence(
    report_path: Path,
    output_path: Path,
    work_dir: Path,
    *,
    reviewer_python: Path | None = None,
    device: str = "cuda",
    include_hub: bool = False,
    cache_policy: model_cache.ModelCachePolicy | None = None,
    cache_root: Path | None = None,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("baseline report must be a JSON object")
    payload = build_reviewer_payload(report)
    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = work_dir / "reviewer-data.json"
    policy = cache_policy if cache_policy is not None else model_cache.policy_from_env_and_args(None)
    cache_identity = reviewer_sidecar_identity(payload, device=device, include_hub=include_hub)
    cache_key = model_cache.key_from_identity(REVIEWER_SIDECAR_CACHE_KIND, cache_identity)
    cached = model_cache.read(REVIEWER_SIDECAR_CACHE_KIND, cache_key, cache_identity, policy, cache_root)
    if _reviewer_cache_payload_complete(cached):
        # Files are stored as their exact text so a replay writes the same
        # bytes the fill run wrote (the envelope itself is key-sorted). Stage 4
        # re-reads reviewer-data.json / hubp-raw.json, so they are restored too.
        sidecar_text = str(cached["sidecar"])
        sidecar = json.loads(sidecar_text)
        statuses = {
            name: {**dict(state), "source": "cache"} if isinstance(state, Mapping) else state
            for name, state in cached["statuses"].items()
        }
        _write_json(work_dir / "reviewer-status.json", {"schema": 1, "steps": statuses, "source": "cache"})
        data_path.write_text(str(cached["reviewer_data"]), encoding="utf-8")
        hubp_cached = cached.get("hubp_raw")
        hubp_restore = work_dir / "hubp-raw.json"
        if isinstance(hubp_cached, str):
            hubp_restore.write_text(hubp_cached, encoding="utf-8")
        else:
            hubp_restore.unlink(missing_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(sidecar_text, encoding="utf-8")
        return {
            "evidence": str(output_path),
            "statuses": statuses,
            "hub_requested": include_hub,
            "qualified_reviewers": sorted(sidecar["offsets"]),
            "offset_sample_counts": sidecar["offset_sample_counts"],
            "cache": {"status": "hit", "kind": REVIEWER_SIDECAR_CACHE_KIND, "key": cache_key},
        }
    if policy.cache_only:
        raise model_cache.ModelCacheMiss(f"model-cache-miss:{REVIEWER_SIDECAR_CACHE_KIND}:{cache_key}")
    plan_path = work_dir / "hubp-plan.json"
    hubp_path = work_dir / "hubp-raw.json"
    log_path = work_dir / "reviewers.log"
    status_path = work_dir / "reviewer-status.json"
    status_path.unlink(missing_ok=True)
    # HUBP resumes any song previously marked OK, even when its lyrics changed.
    # Invalidate the complete HUBP plan/output pair on every acquisition.
    for stale_path in (plan_path, hubp_path):
        stale_path.unlink(missing_ok=True)
    _write_json(data_path, payload)

    python_path = reviewer_python or runtime_paths.venv_python()
    song_id = str(payload["songs"][0]["id"])
    statuses: dict[str, object] = {}
    commands = build_reviewer_commands(
        data_path=data_path,
        plan_path=plan_path,
        hubp_path=hubp_path,
        song_id=song_id,
        python_path=python_path,
        device=device,
        include_hub=include_hub,
    )
    plan_ok = True
    for name, command in commands:
        if name == "HUBP" and not plan_ok:
            statuses[name] = {"status": "SKIPPED_DEPENDENCY"}
            continue
        started = time.perf_counter()
        try:
            code = _run_logged(command, log_path)
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            statuses[name] = {
                "status": f"error:{type(exc).__name__}",
                "exit_code": None,
                "seconds": round(max(0.0, float(elapsed)), 6),
            }
            _write_json(status_path, {"schema": 1, "steps": statuses})
            raise
        finally:
            elapsed = time.perf_counter() - started
        status_payload: Mapping[str, object] | None = None
        if name == "HUB" and data_path.is_file():
            loaded = json.loads(data_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, Mapping):
                status_payload = loaded
        statuses[name] = {
            "status": reviewer_step_status(name, code, status_payload),
            "exit_code": code,
            "seconds": round(max(0.0, float(elapsed)), 6),
        }
        if name == "HUBP_PLAN":
            plan_ok = code == 0
        _write_json(status_path, {"schema": 1, "steps": statuses})

    reviewer_payload = json.loads(data_path.read_text(encoding="utf-8-sig"))
    hubp_payload: dict[str, object] = {}
    if hubp_path.is_file():
        loaded_hubp = json.loads(hubp_path.read_text(encoding="utf-8-sig"))
        if isinstance(loaded_hubp, dict):
            hubp_payload = loaded_hubp
    sidecar = build_r2_sidecar(reviewer_payload, hubp_payload)
    _write_json(output_path, sidecar)
    cache_state = "not-written"
    if _reviewer_statuses_cacheable(statuses):
        written = model_cache.write(
            REVIEWER_SIDECAR_CACHE_KIND, cache_key, cache_identity,
            {
                "sidecar": output_path.read_text(encoding="utf-8"),
                "statuses": statuses,
                "reviewer_data": data_path.read_text(encoding="utf-8"),
                "hubp_raw": hubp_path.read_text(encoding="utf-8") if hubp_path.is_file() else None,
            },
            policy, cache_root,
        )
        cache_state = "written" if written else "disabled"
    return {
        "evidence": str(output_path),
        "statuses": statuses,
        "hub_requested": include_hub,
        "qualified_reviewers": sorted(sidecar["offsets"]),
        "offset_sample_counts": sidecar["offset_sample_counts"],
        "cache": {"status": cache_state, "kind": REVIEWER_SIDECAR_CACHE_KIND, "key": cache_key},
    }


def _strip_options(
    args: Sequence[str], *, one_value: set[str], flags: set[str]
) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(args):
        value = str(args[index])
        matched_value = next(
            (name for name in one_value if value == name or value.startswith(name + "=")),
            None,
        )
        if matched_value is not None:
            if value == matched_value:
                index += 2
            else:
                index += 1
            continue
        if value in flags:
            index += 1
            continue
        result.append(value)
        index += 1
    return result


def prepare_baseline_args(
    backend_args: Sequence[str], *, output: Path, report_dir: Path
) -> list[str]:
    cleaned = _strip_options(
        backend_args,
        one_value={
            "--arbiter", "--reviewer-evidence", "--output", "--report-dir",
            "--min-trusted-percent", "--stage-timings-pass",
        },
        flags={"--strict-review", "--fail-on-review-required", "--discard-manual-edits"},
    )
    cleaned.extend(["--stage-timings-pass", "baseline"])
    cleaned.append("--no-review-summary")
    cleaned.extend(
        [
            "--arbiter", "off",
            "--output", str(output),
            "--report-dir", str(report_dir),
            "--overwrite",
        ]
    )
    return cleaned


def prepare_final_args(backend_args: Sequence[str], evidence_path: Path) -> list[str]:
    cleaned = _strip_options(
        backend_args,
        one_value={"--arbiter", "--reviewer-evidence", "--stage-timings-pass"},
        flags=set(),
    )
    cleaned.extend(["--stage-timings-pass", "final"])
    cleaned.extend(
        [
            "--arbiter", "reviewer-validity",
            "--reviewer-evidence", str(evidence_path),
        ]
    )
    return cleaned


def prepare_final_fallback_args(backend_args: Sequence[str], evidence_path: Path) -> list[str]:
    return _strip_options(
        prepare_final_args(backend_args, evidence_path),
        one_value={"--reviewer-evidence"},
        flags=set(),
    )


def _run_backend(args: Sequence[str], *, timeout: float | None = None) -> int:
    if timeout is None:
        timeout = subprocess_timeout_seconds()
    return _run_with_timeout([sys.executable, str(AUTO_LRC), *args], timeout=timeout)


def reviewer_status_rows(statuses: Mapping[str, object], log_path: Path | None) -> list[dict[str, object]]:
    """Flatten the per-step status mapping into report rows
    ``{name, exit_code, status, log_path, elapsed_s}``."""
    rows: list[dict[str, object]] = []
    for name, state in statuses.items():
        if not isinstance(state, Mapping):
            continue
        rows.append({
            "name": str(name),
            "exit_code": state.get("exit_code"),
            "status": str(state.get("status")),
            "log_path": str(log_path) if log_path is not None else None,
            "elapsed_s": state.get("seconds"),
        })
    return rows


def reviewer_status_source(statuses: Mapping[str, object]) -> str:
    states = [state for state in statuses.values() if isinstance(state, Mapping)]
    if states and all(state.get("source") == "cache" for state in states):
        return "cache"
    return "r2"


def strict_reviewers_exit_code(rows: Sequence[Mapping[str, object]]) -> int:
    """``--strict-reviewers``: 4 when any step is outside the acceptable set."""
    for row in rows:
        if str(row.get("status")) not in REVIEWER_ACCEPTABLE_STATUSES:
            return EXIT_STRICT_REVIEWERS
    return 0


def _rewrite_report(path: Path, mutate) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("final report must be a JSON object")
    mutate(report)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def attach_reviewer_status(path: Path, rows: Sequence[Mapping[str, object]], source: str) -> dict[str, object]:
    """Add ``report["reviewer_status"]`` to an already written final report.

    The sidecar text is cached byte-exact (W3b), so the status block lives on
    the report rather than inside the sidecar."""
    def mutate(report: dict[str, object]) -> None:
        report["reviewer_status"] = {"source": source, "steps": [dict(row) for row in rows]}
    return _rewrite_report(path, mutate)


def _mark_r2_skipped_report(
    path: Path, reason: str, reviewer_rows: Sequence[Mapping[str, object]] = ()
) -> None:
    def mutate(report: dict[str, object]) -> None:
        report["r2"] = {
            "skipped": True,
            "reason": str(reason),
            "arbiter": "reviewer-validity",
            "reviewer_evidence": None,
        }
        report["reviewer_status"] = {"source": "r2-fallback", "steps": [dict(row) for row in reviewer_rows]}
    _rewrite_report(path, mutate)


def _load_reviewer_status_rows(run_dir: Path) -> list[dict[str, object]]:
    status_path = run_dir / "reviewers" / "reviewer-status.json"
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return []
    steps = payload.get("steps") if isinstance(payload, Mapping) else None
    if not isinstance(steps, Mapping):
        return []
    return reviewer_status_rows(steps, run_dir / "reviewers" / "reviewers.log")


def _report_stage_total(path: Path | None) -> float | None:
    if path is None or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        timing = value.get("stage_timings") if isinstance(value, Mapping) else None
        total = timing.get("total_seconds") if isinstance(timing, Mapping) else None
        return float(total) if total is not None else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_r2_timings(
    run_dir: Path,
    stages: list[dict[str, object]],
    reviewers: Mapping[str, object] | None,
    baseline_report: Path | None,
    final_report: Path | None,
    total_seconds: float,
) -> None:
    try:
        baseline_total = _report_stage_total(baseline_report)
        final_total = _report_stage_total(final_report)
        stage_seconds = {int(row["stage"]): row.get("seconds") for row in stages}
        payload = {
            "schema": 1,
            "stages": stages,
            "reviewers": dict(reviewers) if isinstance(reviewers, Mapping) else None,
            "baseline_report": {
                "path": str(baseline_report) if baseline_report is not None else None,
                "stage_timings_total_seconds": baseline_total,
            },
            "final_report": {
                "path": str(final_report) if final_report is not None else None,
                "stage_timings_total_seconds": final_total,
            },
            "process_overhead_seconds": {
                "baseline": (
                    float(stage_seconds[1]) - baseline_total
                    if stage_seconds.get(1) is not None and baseline_total is not None else None
                ),
                "final": (
                    float(stage_seconds[3]) - final_total
                    if stage_seconds.get(3) is not None and final_total is not None else None
                ),
            },
            "total_seconds": round(max(0.0, float(total_seconds)), 6),
        }
        _write_json(run_dir / "r2-timings.json", payload)
    except Exception as exc:
        print(f"WARN: r2 timings unavailable ({type(exc).__name__})", file=sys.stderr, flush=True)


def _print_r2_timings(stages: list[dict[str, object]], reviewers: Mapping[str, object] | None, total: float) -> None:
    try:
        by_stage = {int(row["stage"]): row for row in stages}
        def seconds(stage):
            value = by_stage.get(stage, {}).get("seconds")
            return float(value) if value is not None else 0.0
        reviewer_bits = []
        for name, value in (reviewers or {}).items():
            if isinstance(value, Mapping) and value.get("seconds") is not None:
                reviewer_bits.append(f"{name} {float(value['seconds']):.1f}s")
        detail = ", ".join(reviewer_bits)
        print(
            f"R2 TIMINGS total={total:.1f}s | 1 baseline {seconds(1):.1f}s | "
            f"2 reviewers {seconds(2):.1f}s ({detail}) | 3 final {seconds(3):.1f}s | "
            f"4 trust {seconds(4):.1f}s",
            flush=True,
        )
    except Exception as exc:
        print(f"WARN: r2 timings unavailable ({type(exc).__name__})", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one-song R2 reviewer acquisition and final LRC generation."
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--reviewer-python", type=Path)
    parser.add_argument("--reviewer-device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--include-hub-reviewer", action="store_true")
    parser.add_argument(
        "--strict-reviewers",
        action="store_true",
        help="exit 4 (after writing the LRC and report) when any reviewer step is not OK / MISSING_UNSUPPORTED / SKIPPED_DEPENDENCY",
    )
    parser.add_argument(
        "--reviewer-profile",
        choices=tuple(reviewer_layer.TRUST_PROFILES),
        default=reviewer_layer.DEFAULT_TRUST_PROFILE,
        help="Auxiliary reviewer trust label profile; does not change R2b timing selection.",
    )
    parser.add_argument("backend_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    backend_args = list(args.backend_args)
    if backend_args and backend_args[0] == "--":
        backend_args = backend_args[1:]
    if not backend_args:
        parser.error("pass auto_lrc arguments after --")

    parsed = auto_lrc.build_parser().parse_args(backend_args)
    if len(parsed.audio) != 1:
        parser.error("R2 production pipeline processes exactly one song per invocation")
    if parsed.probe:
        parser.error("R2 production pipeline does not support --probe")
    audio_path = Path(parsed.audio[0]).expanduser().resolve()
    final_output = Path(parsed.output).expanduser().resolve() if parsed.output else audio_path.with_suffix(".lrc")
    final_report_dir = Path(parsed.report_dir).expanduser().resolve() if parsed.report_dir else runtime_paths.reports_dir()
    final_report_path, _audit_path, _template_path = auto_lrc.report_artifact_paths(audio_path, final_output, final_report_dir)
    if parsed.overwrite and final_output.exists():
        edit_reason = manual_edit_reason(final_output, final_report_path)
        if edit_reason and not getattr(parsed, "discard_manual_edits", False) and not journal_path(final_report_path).exists():
            print(
                f"ERROR: Manual review edits detected for {final_output} ({edit_reason}); refusing to overwrite. "
                "Re-run with --discard-manual-edits to replace them (the edited LRC and edit log are archived first).",
                file=sys.stderr,
            )
            return 2
    run_dir = reviewer_work_directory(Path(args.work_dir).expanduser().resolve(), audio_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reviewers" / "reviewer-status.json").unlink(missing_ok=True)
    baseline_output = run_dir / "baseline.lrc"
    baseline_report_dir = run_dir / "baseline-reports"
    evidence_path = run_dir / "reviewer-evidence.json"
    stages: list[dict[str, object]] = [
        {"stage": stage, "name": name, "seconds": None, "status": "not-run"}
        for stage, name in ((1, "baseline"), (2, "reviewers"), (3, "final"), (4, "trust"))
    ]
    reviewers: Mapping[str, object] | None = None
    baseline_report_path: Path | None = None
    started_total = time.perf_counter()
    active_stage = 0
    try:
        baseline_args = prepare_baseline_args(backend_args, output=baseline_output, report_dir=baseline_report_dir)
        active_stage = 1
        stage_started = time.perf_counter()
        print(stage_message(1, "R2: generating Central baseline"), flush=True)
        baseline_code = _run_backend(baseline_args)
        stages[0].update({"seconds": round(time.perf_counter() - stage_started, 6), "status": "ok" if baseline_code == 0 else "failed"})
        print(stage_done_message(1, stages[0]["seconds"]), flush=True)
        active_stage = 0
        if baseline_code != 0:
            return baseline_code
        baseline_report_path, _audit, _template = auto_lrc.report_artifact_paths(audio_path, baseline_output.resolve(), baseline_report_dir.resolve())
        if not baseline_report_path.is_file():
            print(f"ERROR: baseline report missing: {baseline_report_path}", file=sys.stderr)
            return 1

        active_stage = 2
        stage_started = time.perf_counter()
        print(stage_message(2, "R2: acquiring independent reviewer evidence"), flush=True)
        try:
            summary = acquire_reviewer_evidence(
                report_path=baseline_report_path,
                output_path=evidence_path,
                work_dir=run_dir / "reviewers",
                reviewer_python=args.reviewer_python,
                device=args.reviewer_device,
                include_hub=args.include_hub_reviewer,
            )
        except model_cache.ModelCacheMiss as exc:
            stages[1].update({"seconds": round(time.perf_counter() - stage_started, 6), "status": "cache-miss"})
            active_stage = 0
            print(f"ERROR: R2 reviewer evidence missing in cache-only mode: {exc}", file=sys.stderr, flush=True)
            return 1
        except MissingCurrentCandidateError as exc:
            stages[1].update({"seconds": round(time.perf_counter() - stage_started, 6), "status": "skipped"})
            active_stage = 0
            print(f"WARNING: R2 skipped for this song: {exc}", file=sys.stderr, flush=True)
            active_stage = 3
            fallback_started = time.perf_counter()
            fallback_code = _run_backend(
                prepare_final_fallback_args(backend_args, evidence_path)
            )
            stages[2].update(
                {
                    "seconds": round(time.perf_counter() - fallback_started, 6),
                    "status": "ok-no-reviewer" if fallback_code == 0 else "failed",
                }
            )
            active_stage = 0
            if fallback_code != 0:
                return fallback_code
            if not final_report_path.is_file():
                print(f"ERROR: final report missing: {final_report_path}", file=sys.stderr)
                return 1
            try:
                _mark_r2_skipped_report(final_report_path, str(exc), _load_reviewer_status_rows(run_dir))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as marker_exc:
                print(f"ERROR: unable to mark R2 skipped report: {marker_exc}", file=sys.stderr)
                return 1
            stages[3].update({"seconds": 0.0, "status": "skipped"})
            return 3
        reviewers = summary.get("statuses", {}) if isinstance(summary, Mapping) else {}
        stages[1].update({"seconds": round(time.perf_counter() - stage_started, 6), "status": "ok"})
        print(stage_done_message(2, stages[1]["seconds"]), flush=True)
        active_stage = 0
        print(stage_message(2, "R2: reviewer acquisition " + json.dumps(summary, ensure_ascii=False, sort_keys=True)), flush=True)

        active_stage = 3
        stage_started = time.perf_counter()
        print(stage_message(3, "R2: generating final reviewer-validity output"), flush=True)
        final_code = _run_backend(prepare_final_args(backend_args, evidence_path))
        stages[2].update({"seconds": round(time.perf_counter() - stage_started, 6), "status": "ok" if final_code == 0 else "failed"})
        print(stage_done_message(3, stages[2]["seconds"]), flush=True)
        active_stage = 0
        if final_code != 0:
            return final_code
        if not final_report_path.is_file():
            print(f"ERROR: final report missing: {final_report_path}", file=sys.stderr)
            return 1

        active_stage = 4
        stage_started = time.perf_counter()
        print(stage_message(4, "R2: building reviewer trust summary"), flush=True)
        reviewer_data_path = run_dir / "reviewers" / "reviewer-data.json"
        hubp_path = run_dir / "reviewers" / "hubp-raw.json"
        reviewer_payload = json.loads(reviewer_data_path.read_text(encoding="utf-8-sig"))
        hubp_payload: dict[str, object] = {}
        if hubp_path.is_file():
            loaded_hubp = json.loads(hubp_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded_hubp, dict):
                hubp_payload = loaded_hubp
        reviewer_rows = reviewer_status_rows(reviewers, run_dir / "reviewers" / "reviewers.log")
        final_report = attach_reviewer_status(final_report_path, reviewer_rows, reviewer_status_source(reviewers))
        trust_report = build_reviewer_trust_report(reviewer_payload, hubp_payload, final_times_from_report(final_report_path), profile=args.reviewer_profile)
        trust_path = run_dir / "reviewer-trust.json"
        _write_json(trust_path, trust_report)
        stages[3].update({"seconds": round(time.perf_counter() - stage_started, 6), "status": "ok"})
        print(stage_done_message(4, stages[3]["seconds"]), flush=True)
        active_stage = 0
        print(stage_message(4, "R2: reviewer trust " + json.dumps({"profile": trust_report["profile"], "trusted_count": trust_report["trusted_count"], "review_count": trust_report["review_count"], "path": str(trust_path)}, ensure_ascii=False, sort_keys=True)), flush=True)
        for message in build_lyric_review_messages(final_report, reviewer_payload):
            print(message, flush=True)
        sparse_notice = build_sparse_reviewer_notice(reviewer_payload, hubp_payload)
        if sparse_notice:
            print(sparse_notice, flush=True)
        if args.strict_reviewers:
            strict_code = strict_reviewers_exit_code(reviewer_rows)
            if strict_code:
                bad = [f"{row['name']}={row['status']}" for row in reviewer_rows if str(row.get("status")) not in REVIEWER_ACCEPTABLE_STATUSES]
                print(f"ERROR: --strict-reviewers: reviewer steps not acceptable: {', '.join(bad)}", file=sys.stderr, flush=True)
                return strict_code
        return 0
    except BaseException as exc:
        if 1 <= active_stage <= 4:
            row = stages[active_stage - 1]
            if row["status"] == "not-run":
                row.update(
                    {
                        "seconds": round(time.perf_counter() - stage_started, 6),
                        "status": f"error:{type(exc).__name__}",
                    }
                )
        raise
    finally:
        total = time.perf_counter() - started_total
        try:
            status_payload = json.loads((run_dir / "reviewers" / "reviewer-status.json").read_text(encoding="utf-8-sig"))
            reviewers = status_payload.get("steps") if isinstance(status_payload, Mapping) and isinstance(status_payload.get("steps"), Mapping) else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            reviewers = None
        _write_r2_timings(run_dir, stages, reviewers, baseline_report_path, final_report_path, total)
        if not getattr(parsed, "no_review_summary", False):
            _print_r2_timings(stages, reviewers, total)


if __name__ == "__main__":
    raise SystemExit(main())
