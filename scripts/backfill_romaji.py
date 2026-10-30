"""Add additive `romaji_display` to existing align reports (dry run unless --apply).

    python scripts/backfill_romaji.py --report-dir outputs/reports --backup-dir <dir> [--apply]

Each changed report is backed up first and rewritten atomically.  The rows come
from the same payload the review page uses, so the stored romaji matches the
lyric line currently shown (`romaji_display_text`).  Reports whose rows do not
line up one-to-one with their assignments are skipped and listed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lrc_reading
import review_server


def backfill_report(report_path: Path, apply: bool, backup_dir: Path | None) -> str:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assignments = report.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        return "skip:no-assignments"
    if all(isinstance(a, dict) and a.get("romaji_display") for a in assignments if isinstance(a, dict)):
        return "skip:already"
    payload = review_server._song_payload(report_path, report)
    rows = payload.get("rows") or []
    if len(rows) != len(assignments):
        return f"skip:rows {len(rows)} != assignments {len(assignments)}"
    changed = 0
    for assignment, row in zip(assignments, rows):
        lyrics = row.get("lyrics") or []
        line = lyrics[0] if lyrics else ""
        if isinstance(assignment, dict) and lrc_reading.has_japanese(line) and "romaji_display" not in assignment:
            value = lrc_reading.romanize_line(line)
            if value:
                assignment["romaji_display"] = value
                assignment["romaji_display_text"] = line
                changed += 1
    if not changed:
        return "skip:nothing"
    report["romaji_display_system"] = lrc_reading.reading_system()
    if apply:
        if backup_dir is not None:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report_path, backup_dir / report_path.name)
        temporary = report_path.with_name(report_path.name + ".tmp-romaji")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, report_path)
    return f"{'wrote' if apply else 'would-write'}:{changed}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.apply and args.backup_dir is None:
        parser.error("--apply requires --backup-dir")
    counts: dict[str, int] = {}
    for path in sorted(args.report_dir.glob("*.align-report.json")):
        try:
            result = backfill_report(path, args.apply, args.backup_dir)
        except Exception as exc:  # keep going; report the failure
            result = f"error:{type(exc).__name__}:{exc}"
        key = result.split(":")[0] + (":" + result.split(":")[1] if result.startswith("skip") else "")
        counts[key] = counts.get(key, 0) + 1
        if result.startswith(("error", "skip:rows")):
            print(path.name, result)
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
