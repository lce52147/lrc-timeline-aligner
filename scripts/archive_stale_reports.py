"""Move alignment reports whose LRC no longer exists out of the live reports folder.

A report is "stale" when its ``output_path`` (the same-name .lrc) is missing, so the
review page can only show it read-only as ``lrc-missing``. Old test runs leave such
reports behind. By default this only lists them; ``--apply`` moves each report and
its sidecars (``.review.tsv``, ``.auto.lrc``, ``.review-edits.json``) into the
archive folder. Nothing is deleted and reports with a live LRC are never touched.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime_paths  # noqa: E402

REPORT_SUFFIX = ".align-report.json"
SIDECAR_SUFFIXES = (".review.tsv", ".auto.lrc", ".review-edits.json")


def report_is_stale(report_path: Path) -> tuple[bool, str]:
    """Return (stale, reason). Unreadable reports are left alone and reported."""
    try:
        data = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        return False, f"unreadable: {error}"
    output = data.get("output_path") if isinstance(data, dict) else None
    if not isinstance(output, str) or not output:
        return False, "no output_path"
    if Path(output).expanduser().is_file():
        return False, "lrc present"
    return True, "lrc-missing"


def sidecars_of(report_path: Path) -> list[Path]:
    stem = report_path.name[: -len(REPORT_SUFFIX)]
    return [report_path.with_name(stem + suffix) for suffix in SIDECAR_SUFFIXES]


def plan(report_dir: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    stale: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for report in sorted(report_dir.glob("*" + REPORT_SUFFIX), key=lambda item: item.name.casefold()):
        is_stale, reason = report_is_stale(report)
        if is_stale:
            stale.append(report)
        elif reason != "lrc present":
            skipped.append((report, reason))
    return stale, skipped


def archive(stale: list[Path], archive_dir: Path) -> list[Path]:
    moved: list[Path] = []
    archive_dir.mkdir(parents=True, exist_ok=True)
    for report in stale:
        for path in [report, *sidecars_of(report)]:
            if not path.is_file():
                continue
            target = archive_dir / path.name
            if target.exists():
                raise FileExistsError(f"refusing to overwrite {target}")
            shutil.move(str(path), str(target))
            moved.append(target)
    return moved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-dir", type=Path, default=None, help="default: the runtime reports folder")
    parser.add_argument("--archive-dir", type=Path, default=None, help="default: <report-dir>/_archive/stale_<date>")
    parser.add_argument("--apply", action="store_true", help="move the files; without it only list them")
    args = parser.parse_args(argv)
    report_dir = (args.report_dir or runtime_paths.reports_dir()).expanduser().resolve()
    if not report_dir.is_dir():
        print(f"ERROR: report dir not found: {report_dir}")
        return 2
    archive_dir = args.archive_dir or report_dir / "_archive" / ("stale_" + _dt.date.today().isoformat())
    stale, skipped = plan(report_dir)
    for report in stale:
        print(f"STALE {report.name}")
    for report, reason in skipped:
        print(f"SKIP  {report.name}: {reason}")
    print(f"{len(stale)} stale report(s) in {report_dir}")
    if not stale:
        return 0
    if not args.apply:
        print(f"dry run; add --apply to move them to {archive_dir}")
        return 0
    moved = archive(stale, Path(archive_dir).expanduser().resolve())
    print(f"moved {len(moved)} file(s) to {archive_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
