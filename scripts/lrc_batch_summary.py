"""Read-only summary of completed per-song LRC production artifacts."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
import runtime_paths  # noqa: E402
DEFAULT_REPORT_DIR = runtime_paths.reports_dir()
DEFAULT_WORK_DIR = REPO / "outputs" / "reviewer-work"
REVIEW_SERVER = Path(__file__).resolve().with_name("review_server.py")
_SONG_FIELDS = (
    "index", "audio", "name", "status", "lrc_exists", "report_path",
    "review_tsv_path", "seconds", "seconds_source",
    "stage_timings_total_seconds", "r2_total_seconds", "review_rows",
    "untrusted_rows", "model_disagreement_rows", "max_severity_score",
    "reviewers_ok", "reviewers_total", "reviewers_source", "notes",
)


def _reviewer_counts(report):
    """(ok, total, source) from ``report["reviewer_status"]`` or (None, None, None)."""
    block = report.get("reviewer_status")
    if not isinstance(block, dict):
        return None, None, None
    steps = block.get("steps")
    if not isinstance(steps, list):
        return None, None, block.get("source")
    acceptable = {"OK", "MISSING_UNSUPPORTED", "SKIPPED_DEPENDENCY"}
    total = sum(1 for step in steps if isinstance(step, dict))
    ok = sum(1 for step in steps if isinstance(step, dict) and str(step.get("status")) in acceptable)
    return ok, total, block.get("source")


def report_path(audio, output, report_dir):
    audio = Path(audio)
    output = Path(output)
    identity = f"{audio.resolve()}\0{output.resolve()}".casefold().encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:10]
    return Path(report_dir) / f"{audio.stem}--{digest}.align-report.json"


def _safe_name(value):
    forbidden = set('<>:"/\\|?*')
    cleaned = "".join("_" if char in forbidden else char for char in value).strip().rstrip(".")
    return cleaned or "song"


def r2_run_dir(work_dir, audio):
    audio = Path(audio)
    canonical = str(audio.expanduser().resolve()).casefold()
    suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return Path(work_dir) / f"{_safe_name(audio.stem)}--{suffix}"


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) <= sys.float_info.max else None


def _rounded(value):
    number = _number(value)
    return round(number, 3) if number is not None and number >= 0 else None


def _canonical(value):
    return str(Path(value).expanduser().resolve()).casefold()


def _paths_equal(left, right):
    if _canonical(left) == _canonical(right):
        return True
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _since_parts(since):
    if since is None:
        return None, None
    if isinstance(since, datetime.datetime):
        return since.timestamp(), since.isoformat()
    if isinstance(since, (int, float)) and not isinstance(since, bool):
        stamp = float(since)
        return stamp, datetime.datetime.fromtimestamp(stamp).isoformat()
    parsed = datetime.datetime.fromisoformat(str(since))
    return parsed.timestamp(), parsed.isoformat()


def _read_tsv(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 2 or not lines[0].startswith("#"):
        raise ValueError("invalid review TSV header")
    metadata = {}
    for field in lines[0][1:].strip().split("\t"):
        key, separator, value = field.partition("=")
        if separator:
            metadata[key.strip()] = value.strip()
    review_rows = int(metadata["review_rows"])
    untrusted_rows = int(metadata["untrusted_rows"])
    rows = list(csv.DictReader(lines[1:], delimiter="\t", quoting=csv.QUOTE_NONE))
    required = {"severity_reasons", "severity_score"}
    if not required.issubset(rows[0].keys() if rows else lines[1].split("\t")):
        raise ValueError("invalid review TSV columns")
    disagreements = sum("model-disagreement" in row["severity_reasons"] for row in rows)
    scores = []
    for row in rows:
        score = _number(row["severity_score"])
        if score is None:
            raise ValueError("invalid severity score")
        scores.append(int(score) if score.is_integer() else score)
    return review_rows, untrusted_rows, disagreements, max(scores, default=0)


def _record(audio, index, report_dir, work_dir, since_stamp, since_text):
    audio = Path(audio).expanduser().resolve()
    output = audio.with_suffix(".lrc")
    artifact = report_path(audio, output, report_dir)
    record = {
        "index": index,
        "audio": str(audio),
        "name": audio.name,
        "status": "no-report",
        "lrc_exists": output.is_file(),
        "report_path": str(artifact),
        "review_tsv_path": None,
        "seconds": None,
        "seconds_source": None,
        "stage_timings_total_seconds": None,
        "r2_total_seconds": None,
        "review_rows": None,
        "untrusted_rows": None,
        "model_disagreement_rows": None,
        "max_severity_score": None,
        "reviewers_ok": None,
        "reviewers_total": None,
        "reviewers_source": None,
        "notes": [],
        "_work_dir": str(Path(work_dir).expanduser().resolve()),
        "_since": since_text,
    }
    if not artifact.is_file():
        return record

    try:
        report = json.loads(artifact.read_text(encoding="utf-8-sig"))
        if not isinstance(report, dict):
            raise ValueError("report is not an object")
        if not isinstance(report.get("audio_path"), str) or not _paths_equal(report["audio_path"], audio):
            raise ValueError("audio_path mismatch")
    except Exception as exc:
        record["status"] = "unreadable"
        record["notes"].append(f"report:{type(exc).__name__}")
        return record

    if not record["lrc_exists"]:
        record["status"] = "lrc-missing"
    elif since_stamp is not None and artifact.stat().st_mtime < since_stamp:
        record["status"] = "stale"
    else:
        record["status"] = "ok"
    timings = report.get("stage_timings")
    if isinstance(timings, dict):
        record["stage_timings_total_seconds"] = _rounded(timings.get("total_seconds"))
    record["reviewers_ok"], record["reviewers_total"], record["reviewers_source"] = _reviewer_counts(report)

    raw_tsv = report.get("review_tsv_path")
    if isinstance(raw_tsv, str) and raw_tsv:
        tsv_path = Path(raw_tsv).expanduser()
        if not tsv_path.is_absolute():
            tsv_path = artifact.parent / tsv_path
        record["review_tsv_path"] = str(tsv_path.resolve())
        try:
            metrics = _read_tsv(tsv_path)
            (
                record["review_rows"], record["untrusted_rows"],
                record["model_disagreement_rows"], record["max_severity_score"],
            ) = metrics
        except Exception as exc:
            record["notes"].append(f"review-tsv:{type(exc).__name__}")
    else:
        record["notes"].append("review-tsv:missing-path")

    r2_path = r2_run_dir(work_dir, audio) / "r2-timings.json"
    if r2_path.is_file() and not (since_stamp is not None and r2_path.stat().st_mtime < since_stamp):
        try:
            r2 = json.loads(r2_path.read_text(encoding="utf-8-sig"))
            final = r2.get("final_report") if isinstance(r2, dict) else None
            final_path = final.get("path") if isinstance(final, dict) else None
            if isinstance(final_path, str) and _paths_equal(final_path, artifact):
                record["r2_total_seconds"] = _rounded(r2.get("total_seconds"))
            else:
                record["notes"].append("r2-timings:report-mismatch")
        except Exception as exc:
            record["notes"].append(f"r2-timings:{type(exc).__name__}")

    if record["r2_total_seconds"] is not None:
        record["seconds"] = record["r2_total_seconds"]
        record["seconds_source"] = "r2-timings"
    elif record["stage_timings_total_seconds"] is not None:
        record["seconds"] = record["stage_timings_total_seconds"]
        record["seconds_source"] = "stage-timings"
    return record


def collect(audios, *, report_dir, work_dir, since=None):
    since_stamp, since_text = _since_parts(since)
    records = []
    for index, audio in enumerate(audios, start=1):
        try:
            records.append(_record(audio, index, report_dir, work_dir, since_stamp, since_text))
        except Exception as exc:
            path = Path(audio).expanduser().resolve()
            records.append({
                "index": index, "audio": str(path), "name": path.name,
                "status": "unreadable", "lrc_exists": path.with_suffix(".lrc").is_file(),
                "report_path": str(report_path(path, path.with_suffix(".lrc"), report_dir)),
                "review_tsv_path": None, "seconds": None, "seconds_source": None,
                "stage_timings_total_seconds": None, "r2_total_seconds": None,
                "review_rows": None, "untrusted_rows": None,
                "model_disagreement_rows": None, "max_severity_score": None,
                "notes": [f"collect:{type(exc).__name__}"],
                "_work_dir": str(Path(work_dir).expanduser().resolve()), "_since": since_text,
            })
    return records


def _duration(seconds):
    minutes = int(max(0.0, float(seconds)) // 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def format_eta(ok_seconds, songs_left):
    values = [float(value) for value in ok_seconds if _number(value) is not None]
    if not values:
        return "n/a"
    median = statistics.median(values)
    median_minutes, median_seconds = divmod(int(median + 0.5), 60)
    return (
        f"~{_duration(median * max(0, songs_left))} "
        f"(median {median_minutes}m{median_seconds:02d}s x {songs_left} left, n={len(values)})"
    )


def summarize(records, *, report_dir, review_server_path, python_exe, targets, now=None):
    records = list(records)
    counts = {
        "ok": sum(row.get("status") == "ok" for row in records),
        "stale": sum(row.get("status") == "stale" for row in records),
        "lrc_missing": sum(row.get("status") == "lrc-missing" for row in records),
        "no_report": sum(row.get("status") == "no-report" for row in records),
        "unreadable": sum(row.get("status") == "unreadable" for row in records),
    }
    attention = sorted(
        (
            row for row in records
            if row.get("status") == "ok" and isinstance(row.get("review_rows"), int) and row["review_rows"] > 0
        ),
        key=lambda row: (
            -row["review_rows"],
            -(row.get("model_disagreement_rows") or 0),
            -(row.get("max_severity_score") or 0),
            str(row.get("name", "")).casefold(),
        ),
    )[:5]
    target_map = {_canonical(target): str(Path(target).expanduser().resolve()) for target in targets}
    attention_paths = [target_map.get(_canonical(row["audio"]), row["audio"]) for row in attention]
    known = [row["seconds"] for row in records if row.get("seconds") is not None]
    stamp = now or datetime.datetime.now().astimezone()
    if isinstance(stamp, (int, float)):
        stamp = datetime.datetime.fromtimestamp(stamp).astimezone()
    work_dir = records[0].get("_work_dir") if records else str(DEFAULT_WORK_DIR.resolve())
    since = records[0].get("_since") if records else None
    songs = [{key: row.get(key) for key in _SONG_FIELDS} for row in records]
    attention_json = [{key: row.get(key) for key in _SONG_FIELDS} for row in attention]
    return {
        "schema": 1,
        "kind": "lrc-batch-summary",
        "generated_at": stamp.isoformat(),
        "since": since,
        "report_dir": str(Path(report_dir).expanduser().resolve()),
        "work_dir": work_dir,
        "n_songs": len(records),
        "counts": counts,
        "total_known_seconds": round(sum(known), 3),
        "total_review_rows": sum(row.get("review_rows") or 0 for row in records),
        "total_untrusted_rows": sum(row.get("untrusted_rows") or 0 for row in records),
        "total_reviewers_ok": sum(row.get("reviewers_ok") or 0 for row in records),
        "total_reviewers": sum(row.get("reviewers_total") or 0 for row in records),
        "songs": songs,
        "attention": attention_json,
        "review_command": [
            str(python_exe), str(Path(review_server_path).expanduser().resolve()),
            "--report-dir", str(Path(report_dir).expanduser().resolve()), *attention_paths,
        ],
    }


def _quote(value):
    return f'"{str(value)}"'


def format_console(summary):
    counts = summary["counts"]
    lines = [
        f"BATCH SUMMARY: {summary['n_songs']} songs | ok={counts['ok']} stale={counts['stale']} "
        f"lrc-missing={counts['lrc_missing']} no-report={counts['no_report']} unreadable={counts['unreadable']} | "
        f"time {_duration(summary['total_known_seconds'])} "
        f"(known for {sum(row.get('seconds') is not None for row in summary['songs'])}/{summary['n_songs']}) | "
        f"review rows={summary['total_review_rows']} (untrusted={summary['total_untrusted_rows']})"
        + (
            f" | reviewers OK={summary.get('total_reviewers_ok', 0)}/{summary.get('total_reviewers', 0)}"
            if summary.get("total_reviewers") else ""
        )
    ]
    lines.extend(
        f"  REVIEWERS {row['name']}: {row['reviewers_ok']}/{row['reviewers_total']} OK ({row.get('reviewers_source')})"
        for row in summary["songs"]
        if isinstance(row.get("reviewers_total"), int) and row.get("reviewers_ok") != row.get("reviewers_total")
    )
    lines.extend(
        f"  MISSING {row['name']}: {row['status']}"
        for row in summary["songs"] if row["status"] != "ok"
    )
    lines.extend(
        f"  ATTENTION #{index} {row['name']}: review={row['review_rows']} "
        f"model-disagreement={row['model_disagreement_rows']} max-score={row['max_severity_score']}"
        for index, row in enumerate(summary["attention"], start=1)
    )
    if summary["attention"]:
        command = summary["review_command"]
        lines.append(
            "REVIEW PAGE: " + " ".join(
                [_quote(command[0]), _quote(command[1]), "--report-dir", _quote(command[3])]
                + [_quote(value) for value in command[4:]]
            )
        )
    else:
        lines.append("REVIEW PAGE: none (no songs need review)")
    lines.append(
        "BATCH JSON: disabled" if summary.get("_json_disabled")
        else f"BATCH JSON: {summary.get('_json_path', 'disabled')}"
    )
    return lines


def write_json(summary, report_dir, *, output=None, now=None, pid=None):
    report_dir = Path(report_dir).expanduser().resolve()
    if output is None and not report_dir.is_dir():
        raise FileNotFoundError(report_dir)
    stamp = now or datetime.datetime.now()
    if isinstance(stamp, (int, float)):
        stamp = datetime.datetime.fromtimestamp(stamp)
    target = Path(output).expanduser().resolve() if output else report_dir / (
        f"batch-{stamp.strftime('%Y%m%d-%H%M%S')}-{os.getpid() if pid is None else pid}.batch-summary.json"
    )
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    payload = {key: value for key, value in summary.items() if not key.startswith("_")}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        summary["_json_path"] = str(target)
        return target
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _parse_since(value):
    lowered = value.casefold()
    if lowered.endswith(("m", "h")):
        try:
            amount = float(lowered[:-1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("--since must be an ISO datetime or <number>m/<number>h") from exc
        if amount < 0:
            raise argparse.ArgumentTypeError("--since duration must be non-negative")
        seconds = amount * (60 if lowered.endswith("m") else 3600)
        return datetime.datetime.fromtimestamp(time.time() - seconds).astimezone()
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--since must be an ISO datetime or <number>m/<number>h") from exc


def _expand_inputs(paths):
    audios = []
    seen = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            candidates = sorted(
                (child for child in path.iterdir() if child.is_file() and runtime_paths.is_audio_file(child)),
                key=lambda child: child.name.casefold(),
            )
        elif path.is_file() and runtime_paths.is_audio_file(path):
            candidates = [path]
        else:
            print(f"WARN: skipped non-FLAC {path}", file=sys.stderr)
            candidates = []
        for candidate in candidates:
            resolved = candidate.resolve()
            key = str(resolved).casefold()
            if key not in seen:
                seen.add(key)
                audios.append(resolved)
    return audios


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize existing per-song LRC reports without modifying them.",
        epilog="Default output is each FLAC with .lrc; wrapper -Output runs are out of scope.",
    )
    parser.add_argument("paths", metavar="PATH", nargs="*")
    parser.add_argument(
        "--paths-file", type=Path,
        help="UTF-8 file with one queued FLAC path per line (used by align-lrc.ps1).",
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--since", type=_parse_since)
    parser.add_argument("--no-json", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = list(args.paths)
    if args.paths_file is not None:
        try:
            listed = args.paths_file.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            print(f"ERROR: cannot read --paths-file: {exc}", file=sys.stderr)
            return 2
        paths.extend(line.strip() for line in listed if line.strip())
    elif not paths:
        parser.error("give PATH arguments or --paths-file")
    try:
        audios = _expand_inputs(paths)
        if not audios:
            print("ERROR: no FLAC inputs", file=sys.stderr)
            return 2
        report_dir = args.report_dir.expanduser().resolve()
        work_dir = args.work_dir.expanduser().resolve()
        records = collect(audios, report_dir=report_dir, work_dir=work_dir, since=args.since)
        summary = summarize(
            records, report_dir=report_dir, review_server_path=REVIEW_SERVER,
            python_exe=sys.executable, targets=audios,
        )
        disabled = args.no_json or os.environ.get("LRC_BATCH_SUMMARY") == "0"
        summary["_json_disabled"] = disabled
        bookkeeping_failed = False
        if not disabled and not report_dir.is_dir() and args.json_out is None:
            summary["_json_disabled"] = True
            print(f"WARN: report directory does not exist; batch JSON skipped: {report_dir}", file=sys.stderr)
        elif not disabled:
            if not report_dir.is_dir():
                print(f"WARN: report directory does not exist: {report_dir}", file=sys.stderr)
            try:
                write_json(summary, report_dir, output=args.json_out)
            except Exception as exc:
                summary["_json_disabled"] = True
                bookkeeping_failed = True
                print(f"WARN: batch summary failed ({type(exc).__name__})", file=sys.stderr)
        for line in format_console(summary):
            print(line)
        return 0 if not bookkeeping_failed and summary["counts"]["ok"] == summary["n_songs"] else 1
    except Exception as exc:
        print(f"WARN: batch summary failed ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
