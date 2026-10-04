from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
REVIEW = PROJECT / "_review"
SCRIPTS = PROJECT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import auto_lrc  # noqa: E402
from common import select_songs  # noqa: E402

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows_from_match_report(
    report: dict[str, object], timestamps: list[float]
) -> list[dict[str, object]]:
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != len(timestamps):
        raise ValueError("whisper assignment/timestamp shape mismatch")
    rows: list[dict[str, object]] = []
    for expected, (assignment, timestamp) in enumerate(zip(assignments, timestamps), start=1):
        if not isinstance(assignment, dict) or assignment.get("entry") != expected:
            raise ValueError(f"invalid whisper assignment at entry {expected}")
        raw_score = assignment.get("score")
        score = float(raw_score) if isinstance(raw_score, (int, float)) else 0.0
        explicit_trust = assignment.get("timing_trusted")
        trusted = (
            bool(explicit_trust)
            if isinstance(explicit_trust, bool)
            else bool(
                assignment.get("segment") is not None
                and score >= auto_lrc.TRUSTED_ALIGNMENT_SCORE
                and not assignment.get("borrowed")
            )
        )
        rows.append({
            "time": round(float(timestamp), 6),
            "trusted": trusted,
            "score": round(score, 6),
            "segment": assignment.get("segment"),
            "borrowed": bool(assignment.get("borrowed")),
        })
    return rows


def song_has_complete_whisper(song: dict[str, object]) -> bool:
    rows = song.get("rows")
    asr_state = song.get("ASR")
    wx_state = song.get("WX")
    if not isinstance(rows, list) or not rows:
        return False
    if not isinstance(asr_state, dict) or asr_state.get("status") not in {"OK", "FAILED"}:
        return False
    if not isinstance(wx_state, dict) or wx_state.get("status") not in {
        "OK", "REJECTED_BY_TRUST_GATE", "FAILED"
    }:
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        sources = row.get("sources")
        if not isinstance(sources, dict):
            return False
        for key in ("ASR", "WX"):
            measured = sources.get(key)
            if not isinstance(measured, dict) or "time" not in measured:
                return False
    return True


def make_args(entries: list[auto_lrc.LyricEntry]):
    args = auto_lrc.build_parser().parse_args(["stage1-placeholder.flac"])
    args.whisper_language = auto_lrc.infer_spoken_language(entries)
    args.whisperx_device = "auto"
    args.whisper_suppress_nst = False
    return args


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--song-id", action="append", default=[])
    result.add_argument("--whisperx-device", choices=("cuda", "cpu", "auto"), default="cuda")
    return result


def save(payload: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def failure_rows(count: int, reason: str) -> list[dict[str, object]]:
    return [
        {"time": None, "trusted": False, "score": None, "reason": reason[:1000]}
        for _ in range(count)
    ]


def main() -> int:
    cli_args = parser().parse_args()
    payload = json.loads(cli_args.data.read_text(encoding="utf-8-sig"))
    songs = select_songs(payload, cli_args.song_id)
    if not isinstance(payload.get("sources"), dict):
        payload["sources"] = {}
    cli = auto_lrc.default_whisper_cli()
    model = auto_lrc.default_whisper_model()
    wx_python = auto_lrc.default_whisperx_python()
    if not cli.is_file() or not model.is_file() or not wx_python.is_file():
        raise RuntimeError("current Whisper/WhisperX runtime is incomplete")
    existing = sum(song_has_complete_whisper(song) for song in songs)
    common = {
        "status": "RUNNING",
        "shared_audio": "accepted S03 vocal stem decoded once to 16 kHz mono WAV",
        "songs_complete": existing,
        "songs_failed": 0,
        "whisper_cli_sha256": sha256_file(cli),
        "whisper_model_sha256": sha256_file(model),
    }
    payload["sources"]["ASR"] = {
        **common,
        "mode": "whisper.cpp raw ASR + current match_whisper_segments",
    }
    payload["sources"]["WX"] = {
        **common,
        "mode": "current run_whisperx_candidate 5-step path; shared raw ASR and shared vocal WAV",
        "whisperx_python_sha256": sha256_file(wx_python),
        "helper_sha256": sha256_file(SCRIPTS / "whisperx_refine.py"),
        "selected_song_ids": [str(song["id"]) for song in songs],
    }
    save(payload, cli_args.data)

    complete = existing
    asr_failures: list[dict[str, object]] = []
    wx_failures: list[dict[str, object]] = []
    total = len(songs)
    for song_index, song in enumerate(songs, start=1):
        if song_has_complete_whisper(song):
            print(f"[{song_index}/{total}] WHISPER SKIP", flush=True)
            continue
        rows = song["rows"]
        entries = [auto_lrc.LyricEntry(list(row["lines"])) for row in rows]
        args = make_args(entries)
        args.whisperx_device = cli_args.whisperx_device
        vocal_path = Path(song["vocal_audio_path"])
        duration = auto_lrc.probe_duration(vocal_path)
        snapshot = auto_lrc._managed_temp_snapshot()
        try:
            wav_path = auto_lrc.decode_temp_wav(vocal_path)
            raw_segments = auto_lrc._run_whispercpp_wav(
                wav_path, args, suppress_nst=False
            )
            try:
                asr_times, asr_report = auto_lrc.match_whisper_segments(
                    entries, raw_segments, duration
                )
                asr_rows = rows_from_match_report(asr_report, asr_times)
                song["ASR"] = {
                    "status": "OK",
                    "trusted_percent": asr_report.get("trusted_percent"),
                    "review_required_percent": asr_report.get("review_required_percent"),
                    "asr_segments": len(raw_segments),
                }
            except Exception as exc:
                asr_rows = failure_rows(len(rows), f"{type(exc).__name__}:{exc}")
                song["ASR"] = {
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}:{exc}"[:2000],
                }
                asr_failures.append({
                    "id": song["id"],
                    "error_class": type(exc).__name__,
                    "error": str(exc)[:2000],
                })
            for row, measured in zip(rows, asr_rows):
                row["sources"]["ASR"] = measured

            wx_status = "OK"
            try:
                wx_times, wx_report = auto_lrc.run_whisperx_candidate(
                    vocal_path,
                    entries,
                    duration,
                    args,
                    suppress_nst=False,
                    predecoded_wav=wav_path,
                    raw_segments_override=raw_segments,
                )
            except auto_lrc.AlignmentEvidenceRejected as exc:
                wx_status = "REJECTED_BY_TRUST_GATE"
                wx_times = list(exc.timestamps)
                wx_report = copy.deepcopy(exc.report)
            except Exception as exc:
                wx_times = []
                wx_report = {}
                wx_status = "FAILED"
                wx_failures.append({
                    "id": song["id"],
                    "error_class": type(exc).__name__,
                    "error": str(exc)[:2000],
                })
                song["WX"] = {
                    "status": "FAILED",
                    "error": f"{type(exc).__name__}:{exc}"[:2000],
                }
            if wx_status != "FAILED":
                try:
                    wx_rows = rows_from_match_report(wx_report, wx_times)
                    song["WX"] = {
                        "status": wx_status,
                        "trusted_percent": wx_report.get("trusted_percent"),
                        "review_required_percent": wx_report.get("review_required_percent"),
                        "whisperx_refinement_count": wx_report.get("whisperx_refinement_count"),
                    }
                except Exception as exc:
                    wx_rows = failure_rows(len(rows), f"{type(exc).__name__}:{exc}")
                    wx_status = "FAILED"
                    song["WX"] = {
                        "status": "FAILED",
                        "error": f"{type(exc).__name__}:{exc}"[:2000],
                    }
                    wx_failures.append({
                        "id": song["id"],
                        "error_class": type(exc).__name__,
                        "error": str(exc)[:2000],
                    })
            else:
                wx_rows = failure_rows(len(rows), song["WX"]["error"])
            for row, measured in zip(rows, wx_rows):
                row["sources"]["WX"] = measured

            complete += 1
            payload["sources"]["ASR"]["songs_complete"] = complete
            payload["sources"]["WX"]["songs_complete"] = complete
            payload["sources"]["ASR"]["songs_failed"] = len(asr_failures)
            payload["sources"]["WX"]["songs_failed"] = len(wx_failures)
            save(payload, cli_args.data)
            print(
                f"[{song_index}/{total}] WHISPER ASR={song['ASR']['status']} WX={song['WX']['status']}",
                flush=True,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}:{exc}"
            asr_failures.append({
                "id": song["id"], "error_class": type(exc).__name__, "error": str(exc)[:2000]
            })
            wx_failures.append({
                "id": song["id"], "error_class": type(exc).__name__, "error": str(exc)[:2000]
            })
            for row in rows:
                row["sources"]["ASR"] = {
                    "time": None, "trusted": False, "score": None, "reason": reason[:1000]
                }
                row["sources"]["WX"] = {
                    "time": None, "trusted": False, "score": None, "reason": reason[:1000]
                }
            song["ASR"] = {"status": "FAILED", "error": reason[:2000]}
            song["WX"] = {"status": "FAILED", "error": reason[:2000]}
            complete += 1
            payload["sources"]["ASR"]["songs_complete"] = complete
            payload["sources"]["WX"]["songs_complete"] = complete
            payload["sources"]["ASR"]["songs_failed"] = len(asr_failures)
            payload["sources"]["WX"]["songs_failed"] = len(wx_failures)
            save(payload, cli_args.data)
            print(f"[{song_index}/{total}] WHISPER FAILED class={type(exc).__name__}", flush=True)
        finally:
            auto_lrc._cleanup_managed_temp_dirs_since(snapshot)

    payload["sources"]["ASR"]["status"] = "READY"
    payload["sources"]["WX"]["status"] = "READY"
    payload["sources"]["ASR"]["failures"] = asr_failures
    payload["sources"]["WX"]["failures"] = wx_failures
    payload["status"] = "ACQUISITION_COMPLETE"
    save(payload, cli_args.data)
    print(json.dumps({
        "songs_complete": complete,
        "ASR_failures": len(asr_failures),
        "WX_failures": len(wx_failures),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
