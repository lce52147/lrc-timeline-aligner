from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
REVIEWERS_DIR = SCRIPTS / "reviewers"
AUTO_LRC = SCRIPTS / "auto_lrc.py"
DEFAULT_WORK_DIR = PROJECT / "outputs" / "reviewer-work"
REVIEWERS = ("HUBP", "WX", "XLSR")
MIN_OFFSET_SAMPLES = 8

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc  # noqa: E402
import reviewer_layer  # noqa: E402

TRUST_REVIEWERS = reviewer_layer.SHIPPED_REVIEWERS


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
        raise ValueError("baseline assignment must contain exactly one ctc-current candidate")
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
    if not lyrics_path.is_file():
        raise FileNotFoundError(f"lyric source unavailable: {lyrics_path}")
    if not vocal_path.is_file():
        raise FileNotFoundError(f"vocal source unavailable: {vocal_path}")

    document = auto_lrc.load_lyrics(lyrics_path)
    entries, _skipped = auto_lrc.remove_instrumental_markers(document.entries)
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
                    "CUR": {"time": cur_time},
                    "FIN": {"time": fin_time},
                },
            }
        )

    language_code = auto_lrc.infer_spoken_language(entries)
    language = {"ja": "japanese", "en": "english", "zh": "chinese"}.get(
        language_code, language_code
    )
    audio_path = Path(audio_value) if isinstance(audio_value, str) else lyrics_path.with_suffix(".flac")
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

    offsets: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    for name in REVIEWERS:
        paired_current: list[float | None] = []
        paired_reviewer: list[float | None] = []
        count = 0
        for current, reviewer in zip(current_times, reviewer_series[name], strict=True):
            if current is None or reviewer is None:
                continue
            if abs(float(reviewer) - float(current)) >= reviewer_layer.OFFSET_SAMPLE_LIMIT_SECONDS:
                continue
            paired_current.append(current)
            paired_reviewer.append(reviewer)
            count += 1
        sample_counts[name] = count
        if count >= MIN_OFFSET_SAMPLES:
            offsets[name] = reviewer_layer.estimate_offset_seconds(
                paired_current, paired_reviewer
            )

    return {
        "schema": 1,
        "policy": "R2b",
        "reviewers": list(REVIEWERS),
        "required_agreements": 1,
        "offsets": offsets,
        "offset_sample_counts": sample_counts,
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

    offsets: dict[str, float] = {}
    sample_counts: dict[str, int] = {}
    for name in TRUST_REVIEWERS:
        count = sum(
            1
            for final_time, reviewer_time in zip(
                normalized_final, reviewer_series[name], strict=True
            )
            if reviewer_time is not None
            and abs(float(reviewer_time) - float(final_time))
            < reviewer_layer.OFFSET_SAMPLE_LIMIT_SECONDS
        )
        sample_counts[name] = count
        if count >= MIN_OFFSET_SAMPLES:
            offsets[name] = reviewer_layer.estimate_offset_seconds(
                normalized_final, reviewer_series[name]
            )

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


def _run_logged(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write("\n$ " + json.dumps(list(command), ensure_ascii=False) + "\n")
        log.flush()
        try:
            result = subprocess.run(
                list(command),
                cwd=PROJECT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            return int(result.returncode)
        except OSError as exc:
            log.write(f"launch-error: {type(exc).__name__}: {exc}\n")
            return 127


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


def reviewer_step_status(name: str, exit_code: int, payload: Mapping[str, object] | None = None) -> str:
    if exit_code == 0:
        return "OK"
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


def acquire_reviewer_evidence(
    report_path: Path,
    output_path: Path,
    work_dir: Path,
    *,
    reviewer_python: Path | None = None,
    device: str = "cuda",
    include_hub: bool = False,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict):
        raise ValueError("baseline report must be a JSON object")
    payload = build_reviewer_payload(report)
    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = work_dir / "reviewer-data.json"
    plan_path = work_dir / "hubp-plan.json"
    hubp_path = work_dir / "hubp-raw.json"
    log_path = work_dir / "reviewers.log"
    status_path = work_dir / "reviewer-status.json"
    _write_json(data_path, payload)

    python_path = reviewer_python or auto_lrc.default_whisperx_python()
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
        code = _run_logged(command, log_path)
        status_payload: Mapping[str, object] | None = None
        if name == "HUB" and data_path.is_file():
            loaded = json.loads(data_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, Mapping):
                status_payload = loaded
        statuses[name] = {
            "status": reviewer_step_status(name, code, status_payload),
            "exit_code": code,
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
    return {
        "evidence": str(output_path),
        "statuses": statuses,
        "hub_requested": include_hub,
        "qualified_reviewers": sorted(sidecar["offsets"]),
        "offset_sample_counts": sidecar["offset_sample_counts"],
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
            "--min-trusted-percent",
        },
        flags={"--strict-review", "--fail-on-review-required"},
    )
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
        one_value={"--arbiter", "--reviewer-evidence"},
        flags=set(),
    )
    cleaned.extend(
        [
            "--arbiter", "reviewer-validity",
            "--reviewer-evidence", str(evidence_path),
        ]
    )
    return cleaned


def _run_backend(args: Sequence[str]) -> int:
    result = subprocess.run([sys.executable, str(AUTO_LRC), *args], cwd=PROJECT, check=False)
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one-song R2 reviewer acquisition and final LRC generation."
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--reviewer-python", type=Path)
    parser.add_argument("--reviewer-device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--include-hub-reviewer", action="store_true")
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
    run_dir = Path(args.work_dir).expanduser().resolve() / _safe_name(audio_path.stem)
    baseline_output = run_dir / "baseline.lrc"
    baseline_report_dir = run_dir / "baseline-reports"
    evidence_path = run_dir / "reviewer-evidence.json"

    baseline_args = prepare_baseline_args(
        backend_args, output=baseline_output, report_dir=baseline_report_dir
    )
    print("R2: generating Central baseline", flush=True)
    baseline_code = _run_backend(baseline_args)
    if baseline_code != 0:
        return baseline_code
    report_path, _audit, _template = auto_lrc.report_artifact_paths(
        audio_path, baseline_output.resolve(), baseline_report_dir.resolve()
    )
    if not report_path.is_file():
        print(f"ERROR: baseline report missing: {report_path}", file=sys.stderr)
        return 1

    print("R2: acquiring independent reviewer evidence", flush=True)
    summary = acquire_reviewer_evidence(
        report_path,
        evidence_path,
        run_dir / "reviewers",
        reviewer_python=args.reviewer_python,
        device=args.reviewer_device,
        include_hub=args.include_hub_reviewer,
    )
    print(
        "R2: reviewer acquisition "
        + json.dumps(summary, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    print("R2: generating final reviewer-validity output", flush=True)
    final_code = _run_backend(prepare_final_args(backend_args, evidence_path))
    if final_code != 0:
        return final_code

    final_output = (
        Path(parsed.output).expanduser().resolve()
        if parsed.output
        else audio_path.with_suffix(".lrc")
    )
    final_report_dir = (
        Path(parsed.report_dir).expanduser().resolve()
        if parsed.report_dir
        else auto_lrc.DEFAULT_REPORT_DIR
    )
    final_report_path, _audit_path, _template_path = auto_lrc.report_artifact_paths(
        audio_path, final_output, final_report_dir
    )
    if not final_report_path.is_file():
        print(f"ERROR: final report missing: {final_report_path}", file=sys.stderr)
        return 1

    reviewer_data_path = run_dir / "reviewers" / "reviewer-data.json"
    hubp_path = run_dir / "reviewers" / "hubp-raw.json"
    reviewer_payload = json.loads(reviewer_data_path.read_text(encoding="utf-8-sig"))
    hubp_payload: dict[str, object] = {}
    if hubp_path.is_file():
        loaded_hubp = json.loads(hubp_path.read_text(encoding="utf-8-sig"))
        if isinstance(loaded_hubp, dict):
            hubp_payload = loaded_hubp
    trust_report = build_reviewer_trust_report(
        reviewer_payload,
        hubp_payload,
        final_times_from_report(final_report_path),
        profile=args.reviewer_profile,
    )
    trust_path = run_dir / "reviewer-trust.json"
    _write_json(trust_path, trust_report)
    print(
        "R2: reviewer trust "
        + json.dumps(
            {
                "profile": trust_report["profile"],
                "trusted_count": trust_report["trusted_count"],
                "review_count": trust_report["review_count"],
                "path": str(trust_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
