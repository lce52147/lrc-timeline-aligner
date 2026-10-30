#!/usr/bin/env python3
"""Move already-written LRC line starts earlier by the pipeline's start lead (BK).

Dry run by default: per song, how many lines would move and by how much, with the
DF_checklist.md status. PASS/CHECKED songs, songs whose report already carries a start
lead, and every line EvanOG set by hand (review decision "change", a review-edits entry, or a
tag that differs from the report's written tag) are skipped. ``--apply`` needs
``--backup-dir``; originals are copied there first and files are replaced atomically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_lrc  # noqa: E402

DEFAULT_REPORTS = Path(__file__).resolve().parent.parent / "outputs" / "reports"
DEFAULT_CHECKLIST = Path(r"D:\Users\Administrator\Music\LRC tools checked references\DF_checklist.md")
SKIP_STATUSES = {"PASS", "CHECKED"}
TAG_RE = re.compile(r"^\[(\d{1,3}):(\d{2})\.(\d{2})\]")


def song_key(text: str) -> str:
    text = text[:-4] if text.lower().endswith(".lrc") else text
    text = re.sub(r"--[0-9a-f]{6,}.*$", "", text)
    text = re.sub(r"^\s*\(?\d{1,3}\)?[\s._-]+", "", text)
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE).casefold()


def load_checklist(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0].isdigit():
            result.setdefault(song_key(cells[3]), cells[2])
    return result


def _tag_cs(line: str) -> int | None:
    match = TAG_RE.match(line)
    if match is None:
        return None
    return (int(match.group(1)) * 60 + int(match.group(2))) * 100 + int(match.group(3))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def manual_entries(report_path: Path) -> set[int]:
    stem = report_path.name.replace(".align-report.json", "")
    entries: set[int] = set()
    for row in _read_jsonl(report_path.with_name(stem + ".review-decisions.jsonl")):
        if row.get("decision") == "change" and isinstance(row.get("entry"), int):
            entries.add(row["entry"])
    edits = report_path.with_name(stem + ".review-edits.json")
    if edits.exists():
        try:
            data = json.loads(edits.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
        for item in data.get("edits", []) if isinstance(data, dict) else []:
            if isinstance(item, dict) and isinstance(item.get("entry"), int):
                entries.add(item["entry"])
    return entries


def plan_song(report_path: Path, checklist: dict[str, str], lead_ms: int) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output = Path(str(report.get("output_path") or ""))
    status = checklist.get(song_key(output.stem or report_path.name), "未審")
    row = {"report": report_path, "lrc": output, "song": output.stem, "df_status": status,
           "lines": 0, "moves": [], "plan": [], "skipped_manual": 0, "action": "plan"}
    if status in SKIP_STATUSES:
        row["action"] = f"skip: DF {status}"
    elif report.get("start_lead_applied"):
        row["action"] = "skip: lead already applied"
    elif not output.exists():
        row["action"] = "skip: LRC missing"
    assignments = report.get("assignments")
    if row["action"] == "plan" and (
        not isinstance(assignments, list) or not assignments
        or any("canonical_written_centiseconds" not in a for a in assignments)
    ):
        row["action"] = "skip: report has no written times"
    if row["action"] != "plan":
        return row
    lrc_lines = output.read_text(encoding="utf-8-sig").splitlines()
    timed = [index for index, line in enumerate(lrc_lines) if _tag_cs(line) is not None]
    lyric_index = timed[1:] if timed and _tag_cs(lrc_lines[timed[0]]) == 0 else timed
    tags = [_tag_cs(lrc_lines[index]) for index in lyric_index]
    written = tuple(
        auto_lrc.canonicalize_lrc_time(a["canonical_written_centiseconds"] / 100) for a in assignments
    )
    shifted = auto_lrc.apply_line_start_lead(json.loads(json.dumps(report)), written, lead_ms)
    manual = manual_entries(report_path)
    cursor = 0
    for number, (old, new) in enumerate(zip(written, shifted), 1):
        if cursor >= len(tags):
            row["skipped_manual"] += 1
            continue
        current = tags[cursor]
        indexes = []
        while cursor < len(tags) and tags[cursor] == current:
            indexes.append(lyric_index[cursor])
            cursor += 1
        # a tag that is not the report's written tag was edited by hand (or the files are out of sync)
        if current != old.centiseconds or number in manual:
            row["skipped_manual"] += 1
            continue
        if new.centiseconds != old.centiseconds:
            row["plan"].append((number, indexes, new.centiseconds))
            row["moves"].append((old.centiseconds - new.centiseconds) * 10)
    row["lines"] = len(written)
    row.update(lrc_lines=lrc_lines, shifted=shifted, written=written, report_data=report)
    if not row["moves"]:
        row["action"] = "skip: nothing to move"
    return row


def _atomic_write(path: Path, data: bytes) -> None:
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def apply_song(row: dict, backup_dir: Path) -> None:
    lrc, report_path = row["lrc"], row["report"]
    backup_dir.mkdir(parents=True, exist_ok=True)
    for source in (lrc, report_path):
        shutil.copy2(source, backup_dir / f"{hashlib.sha1(str(source).encode()).hexdigest()[:10]}__{source.name}")
    lines = list(row["lrc_lines"])
    report = row["report_data"]
    for number, indexes, new_cs in row["plan"]:
        new = row["shifted"][number - 1]
        for index in indexes:
            lines[index] = f"[{new.lrc_tag}]" + lines[index].split("]", 1)[1]
        assignment = report["assignments"][number - 1]
        assignment["start_lead_ms"] = (row["written"][number - 1].centiseconds - new_cs) * 10
        assignment["chosen_time"] = new.seconds
        assignment["canonical_written_centiseconds"] = new.centiseconds
        assignment["lrc_tag"] = new.lrc_tag
    raw = lrc.read_bytes()
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    data = bom + ("\n".join(lines) + "\n").encode("utf-8")
    report["start_lead_applied"] = True
    report["start_lead_retrofit"] = True
    report["published_lrc_sha256"] = hashlib.sha256(data).hexdigest()
    _atomic_write(lrc, data)
    _atomic_write(report_path, json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    parser.add_argument("--lead-ms", type=int, default=100)
    parser.add_argument("--apply", action="store_true", help="write the changes (needs --backup-dir)")
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args(argv)
    if args.apply and args.backup_dir is None:
        parser.error("--apply requires --backup-dir")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checklist = load_checklist(args.checklist)
    rows = [plan_song(path, checklist, args.lead_ms) for path in sorted(args.reports_dir.glob("*.align-report.json"))]
    summary: dict[str, list[int]] = {}
    for row in rows:
        moved = row["moves"]
        entry = summary.setdefault(row["df_status"], [0, 0, 0])
        entry[2] += 1
        if moved:
            entry[0] += 1
            entry[1] += len(moved)
            note = (f"move {len(moved)}/{row['lines']} lines, median {statistics.median(moved):.0f} ms, "
                    f"hand-set/out-of-sync skipped {row['skipped_manual']}")
        else:
            note = row["action"]
        print(f"[{row['df_status']}] {row['song']}: {note}")
    print("\nSUMMARY (DF status: songs that would move / lines that would move / songs listed)")
    for status, (songs, lines, listed) in sorted(summary.items()):
        print(f"  {status}: {songs} songs, {lines} lines (of {listed} listed)")
    print(f"  TOTAL: {sum(v[0] for v in summary.values())} songs, {sum(v[1] for v in summary.values())} lines")
    if args.apply:
        for row in rows:
            if row["moves"]:
                apply_song(row, args.backup_dir)
        print("applied; originals in", args.backup_dir)
    else:
        print("dry run: nothing written (use --apply --backup-dir DIR)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
