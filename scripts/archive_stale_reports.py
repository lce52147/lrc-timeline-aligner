"""Move alignment reports whose audio file no longer exists out of the live reports folder.

A report is "stale" only when the audio file it points to (``audio_path``) is gone: the song was deleted or moved away,
so nothing can be reviewed or exported from it any more. A song whose LRC was exported (the LRC is archived to ``checked/``,
the audio is still there) is finished work, not stale, and a report with a missing LRC but a present audio is left alone.

By default this only lists the stale reports; ``--apply`` moves each report together with every file that shares its stem
(``.review.tsv``, ``.auto.lrc``, ``.review-edits.json``, ``.review-decisions.jsonl``, ``.orphaned-*``, ``.discarded-*``,
the romaji overrides, the harmony candidate sidecar, ...) into the archive folder. A name that already exists there gets a
``.dupN`` suffix instead of stopping the run. Nothing is deleted.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runtime_paths  # noqa: E402

REPORT_SUFFIX = ".align-report.json"
HARMONY_SIDECAR_SUFFIX = ".harmony_cands.json"


def report_is_stale(report_path: Path) -> tuple[bool, str]:
    """Return (stale, reason). Unreadable reports and reports with no audio path are left alone and reported."""
    try:
        data = json.loads(report_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        return False, f"unreadable: {error}"
    audio = data.get("audio_path") if isinstance(data, dict) else None
    if not isinstance(audio, str) or not audio:
        return False, "no audio_path"
    if Path(audio).expanduser().is_file():
        return False, "audio present"
    return True, "audio-missing"


def _audio_stem(report_path: Path) -> str | None:
    try:
        audio = json.loads(report_path.read_text(encoding="utf-8-sig")).get("audio_path")
    except (OSError, ValueError, AttributeError):
        return None
    return Path(audio).stem.casefold() if isinstance(audio, str) and audio else None


def live_audio_stems(report_dir: Path, stale: list[Path]) -> set[str]:
    """The audio stems of the reports that stay: the harmony sidecar is named after the audio, so a stale report whose audio
    has the same file name as a song still in use (a song moved to a new folder and aligned again) does not own it."""
    leaving = {path.resolve() for path in stale}
    stems = set()
    for report in report_dir.glob("*" + REPORT_SUFFIX):
        if report.resolve() not in leaving:
            stem = _audio_stem(report)
            if stem:
                stems.add(stem)
    return stems


def sidecars_of(report_path: Path, live_stems: frozenset[str] | set[str] = frozenset()) -> list[Path]:
    """Every file beside the report that belongs to it: same stem followed by a dot, plus the harmony sidecar named after the audio
    (unless a report that stays uses an audio file of the same name: ``live_stems``)."""
    stem = report_path.name[: -len(REPORT_SUFFIX)]
    prefix = stem + "."
    found = [path for path in report_path.parent.iterdir()
             if path.is_file() and path != report_path and path.name.startswith(prefix)]
    try:
        audio = json.loads(report_path.read_text(encoding="utf-8-sig")).get("audio_path")
    except (OSError, ValueError, AttributeError):
        audio = None
    if isinstance(audio, str) and audio and Path(audio).stem.casefold() not in live_stems:
        harmony = report_path.with_name(Path(audio).stem + HARMONY_SIDECAR_SUFFIX)
        if harmony.is_file() and harmony not in found:
            found.append(harmony)
    return sorted(found, key=lambda path: path.name.casefold())


def plan(report_dir: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    stale: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for report in sorted(report_dir.glob("*" + REPORT_SUFFIX), key=lambda item: item.name.casefold()):
        is_stale, reason = report_is_stale(report)
        if is_stale:
            stale.append(report)
        elif reason != "audio present":
            skipped.append((report, reason))
    return stale, skipped


def _numbered(name: str, number: int) -> str:
    """``a.review.tsv`` -> ``a.review.dup2.tsv``: the extension stays, so the file keeps its kind."""
    dot = name.rfind(".")
    return f"{name}.dup{number}" if dot <= 0 else f"{name[:dot]}.dup{number}{name[dot:]}"


def archive_report(report: Path, archive_dir: Path, live_stems: frozenset[str] | set[str] = frozenset()) -> tuple[list[Path], int, list[str]]:
    """Move one report and its files; (moved targets, the .dupN used or 0, failures). One number serves the whole set."""
    files = [report, *sidecars_of(report, live_stems)]
    number = 0
    while any((archive_dir / (path.name if not number else _numbered(path.name, number))).exists() for path in files):
        number += 1 if number else 2
    moved: list[Path] = []
    failures: list[str] = []
    for path in files:
        target = archive_dir / (path.name if not number else _numbered(path.name, number))
        try:
            shutil.move(str(path), str(target))
        except OSError as error:
            failures.append(f"{path.name}: {error}")
            continue
        moved.append(target)
    return moved, number, failures


def archive(stale: list[Path], archive_dir: Path) -> list[Path]:
    moved, _renamed, _failures = archive_many(stale, archive_dir)
    return moved


def archive_many(stale: list[Path], archive_dir: Path) -> tuple[list[Path], int, list[str]]:
    """Move every stale report; a collision renames, a failed move is recorded, and the rest still go."""
    moved: list[Path] = []
    renamed = 0
    failures: list[str] = []
    archive_dir.mkdir(parents=True, exist_ok=True)
    live_stems = live_audio_stems(stale[0].parent, stale) if stale else set()
    for report in stale:
        done, number, failed = archive_report(report, archive_dir, live_stems)
        moved.extend(done)
        renamed += 1 if number else 0
        failures.extend(failed)
    return moved, renamed, failures


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
    moved, renamed, failures = archive_many(stale, Path(archive_dir).expanduser().resolve())
    kinds = collections.Counter(_kind(path.name) for path in moved)
    print(f"moved {len(moved)} file(s) of {len(stale)} report(s) to {archive_dir}"
          + (f"; {renamed} report(s) renamed with .dupN because the target name already existed" if renamed else ""))
    print("by kind: " + ", ".join(f"{kind} x{count}" for kind, count in sorted(kinds.items())))
    for failure in failures:
        print(f"FAILED {failure}")
    return 1 if failures else 0


def _kind(name: str) -> str:
    """The file's kind for the summary: what follows the report stem's ``--<hash>``, else its last two dotted parts."""
    match = re.search(r"--[0-9a-f]{6,}(\.[^\/]*)$", name)
    kind = match.group(1) if match else "." + ".".join(name.rsplit(".", 2)[-2:])
    return re.sub(r"\d{8}T\d{6}Z(-\d+)?", "<stamp>", kind)


if __name__ == "__main__":
    raise SystemExit(main())
