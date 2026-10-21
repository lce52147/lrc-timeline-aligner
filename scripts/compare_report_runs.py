"""Compare two alignment runs (folders of *.align-report.json) against checked LRC references.

Reads the final per-line times straight from each report's ``assignments[*].lrc_tag``,
so it needs no LRC files from the runs and no bundle layout. Reports are paired by
file name (fallback: same name before the ``--<hash>`` suffix). The checked reference
for a report is the ``*.lrc`` under ``--bk`` whose file name equals the report's
``output_path`` file name; when several match, the one whose parent folder name equals
the report's audio folder name wins, otherwise the song is marked ambiguous and skipped.

Writes ``summary.json`` and ``summary.md`` into ``--out`` and prints the markdown.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPORT_SUFFIX = ".align-report.json"
TAG_RE = re.compile(r"^\[?(\d{2,}):(\d{2})\.(\d{2})\]?")  # report lrc_tag values carry no brackets
BUCKETS = ((10, "<=0.10s"), (20, "<=0.20s"), (50, "<=0.50s"), (100, "<=1.00s"), (None, ">1.00s"))


def tag_cs(value: object) -> int | None:
    match = TAG_RE.match(str(value or ""))
    if not match:
        return None
    return int(match.group(1)) * 6000 + int(match.group(2)) * 100 + int(match.group(3))


def status_of(row: dict) -> str:
    if row.get("review_required") or row.get("verification_status") == "review-required":
        return "REVIEW"
    if row.get("verification_status") == "trusted" and row.get("overall_trusted") is not False:
        return "TRUSTED"
    return "UNTRUSTED"


def load_reports(folder: Path) -> dict[str, dict]:
    found = {}
    for path in sorted(folder.glob("*" + REPORT_SUFFIX)):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("assignments"), list):
            found[path.name] = data
    return found


def base_name(name: str) -> str:
    stem = name[: -len(REPORT_SUFFIX)] if name.endswith(REPORT_SUFFIX) else name
    return re.sub(r"--[0-9a-f]{6,}$", "", stem)


def pair_reports(ref: dict[str, dict], run: dict[str, dict]) -> list[tuple[str, dict | None, dict]]:
    by_base = {}
    for name, data in ref.items():
        by_base.setdefault(base_name(name), []).append((name, data))
    pairs = []
    for name, data in run.items():
        partner = ref.get(name)
        if partner is None:
            candidates = by_base.get(base_name(name), [])
            partner = candidates[0][1] if len(candidates) == 1 else None
        pairs.append((name, partner, data))
    return pairs


def times_of(report: dict) -> list[int | None]:
    return [tag_cs(row.get("lrc_tag")) if isinstance(row, dict) else None for row in report["assignments"]]


def lrc_times(path: Path) -> list[int]:
    """Timestamps of a checked LRC, one per sung line.

    Checked references are bilingual: the translation repeats the original line's
    timestamp on the next line, so consecutive lines with the same tag count once.
    """
    times: list[int] = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        cs = tag_cs(line)
        if cs is None or (times and times[-1] == cs):
            continue
        times.append(cs)
    return times


def loose_key(name: str) -> str:
    """Case/punctuation-insensitive key for matching a BK file name."""
    stem = name[:-4] if name.lower().endswith(".lrc") else name
    return re.sub(r"[\W_]+", "", stem, flags=re.UNICODE).casefold()


def reference_names(report: dict, report_name: str) -> list[str]:
    """Candidate BK file names, most specific first: audio stem, report base name, output name."""
    names = []
    audio = report.get("audio_path")
    if isinstance(audio, str) and audio:
        names.append(Path(audio).stem + ".lrc")
    names.append(base_name(report_name) + ".lrc")
    output = report.get("output_path")
    if isinstance(output, str) and output:
        names.append(Path(output).name)
    return list(dict.fromkeys(names))


def find_reference(report: dict, report_name: str, bk_index: dict[str, list[Path]]) -> tuple[Path | None, str]:
    expected = len(report["assignments"])
    audio = report.get("audio_path")
    folder = Path(audio).parent.name if isinstance(audio, str) and audio else ""
    loose_index: dict[str, list[Path]] = {}
    for paths in bk_index.values():
        for path in paths:
            loose_index.setdefault(loose_key(path.name), []).append(path)
    for name in reference_names(report, report_name):
        candidates = bk_index.get(name) or loose_index.get(loose_key(name), [])
        if not candidates:
            key = loose_key(name)
            partial = [path for k, paths in loose_index.items() for path in paths if len(key) >= 6 and (key in k or k in key)]
            candidates = partial if len(partial) == 1 else []
        if not candidates:
            continue
        if len(candidates) > 1:
            narrowed = [path for path in candidates if path.parent.name == folder]
            if not narrowed:
                narrowed = [path for path in candidates if len(lrc_times(path)) == expected]
            if len(narrowed) != 1:
                return None, "ambiguous-bk-match:" + name
            candidates = narrowed
        return candidates[0], "ok"
    return None, "no-bk-match:" + "|".join(reference_names(report, report_name))


def bucket_counts(deltas: list[int]) -> dict[str, int]:
    counts = {label: 0 for _, label in BUCKETS}
    for delta in deltas:
        for limit, label in BUCKETS:
            if limit is None or delta <= limit:
                counts[label] += 1
                break
    return counts


def score(report: dict, reference: list[int]) -> dict | None:
    times = times_of(report)
    if len(times) != len(reference) or any(t is None for t in times):
        return None
    statuses = [status_of(row) for row in report["assignments"]]
    deltas = [abs(t - r) for t, r in zip(times, reference)]
    return {
        "lines": len(deltas),
        "buckets": bucket_counts(deltas),
        "over_1s": sum(d > 100 for d in deltas),
        "trusted_over_1s": sum(d > 100 for d, s in zip(deltas, statuses) if s == "TRUSTED"),
        "within_20cs": sum(d <= 20 for d in deltas),
        "status": {key: statuses.count(key) for key in ("REVIEW", "UNTRUSTED", "TRUSTED")},
        "deltas": deltas,
    }


def add_totals(total: dict, part: dict) -> None:
    total["lines"] += part["lines"]
    total["over_1s"] += part["over_1s"]
    total["trusted_over_1s"] += part["trusted_over_1s"]
    total["within_20cs"] += part["within_20cs"]
    for key, value in part["buckets"].items():
        total["buckets"][key] = total["buckets"].get(key, 0) + value
    for key, value in part["status"].items():
        total["status"][key] = total["status"].get(key, 0) + value


def empty_totals() -> dict:
    return {"songs": 0, "lines": 0, "over_1s": 0, "trusted_over_1s": 0, "within_20cs": 0, "buckets": {}, "status": {}}


def compare(ref_dir: Path, run_dir: Path, bk_dir: Path) -> dict:
    bk_index: dict[str, list[Path]] = {}
    for path in bk_dir.rglob("*.lrc"):
        bk_index.setdefault(path.name, []).append(path)
    ref_reports, run_reports = load_reports(ref_dir), load_reports(run_dir)
    totals = {"ref": empty_totals(), "run": empty_totals()}
    songs, skipped, moved = [], [], []
    for name, ref_report, run_report in pair_reports(ref_reports, run_reports):
        reference, why = find_reference(run_report, name, bk_index)
        if reference is None:
            skipped.append({"report": name, "reason": why})
            continue
        reference_times = lrc_times(reference)
        run_score = score(run_report, reference_times)
        ref_score = score(ref_report, reference_times) if ref_report is not None else None
        if run_score is None:
            unparsed = sum(t is None for t in times_of(run_report))
            why = (f"unparsable-lrc-tag ({unparsed} rows)" if unparsed
                   else f"line-count-mismatch-with-bk (report {len(run_report['assignments'])} vs bk {len(reference_times)}: {reference})")
            skipped.append({"report": name, "reason": why})
            continue
        entry = {"report": name, "bk": str(reference), "run": {k: v for k, v in run_score.items() if k != "deltas"}}
        totals["run"]["songs"] += 1
        add_totals(totals["run"], run_score)
        if ref_score is not None:
            entry["ref"] = {k: v for k, v in ref_score.items() if k != "deltas"}
            totals["ref"]["songs"] += 1
            add_totals(totals["ref"], ref_score)
            ref_times, run_times = times_of(ref_report), times_of(run_report)
            for index, (a, b, bk) in enumerate(zip(ref_times, run_times, reference_times), 1):
                if a is not None and b is not None and a != b:
                    moved.append({
                        "report": name, "entry": index, "ref_cs": a, "run_cs": b, "shift_cs": b - a,
                        "ref_err_cs": abs(a - bk), "run_err_cs": abs(b - bk),
                    })
        else:
            entry["ref"] = None
            skipped.append({"report": name, "reason": "no-ref-report (scored run only)"})
        songs.append(entry)
    moved.sort(key=lambda item: -abs(item["shift_cs"]))
    better = sum(item["run_err_cs"] < item["ref_err_cs"] for item in moved)
    worse = sum(item["run_err_cs"] > item["ref_err_cs"] for item in moved)
    return {
        "ref_dir": str(ref_dir), "run_dir": str(run_dir), "bk_dir": str(bk_dir),
        "totals": totals, "songs": songs, "skipped": skipped,
        "moved_lines": moved, "moved_songs": len({item["report"] for item in moved}),
        "moved_over_20cs": sum(abs(item["shift_cs"]) > 20 for item in moved),
        "moved_better": better, "moved_worse": worse,
    }


def markdown(result: dict) -> str:
    ref, run = result["totals"]["ref"], result["totals"]["run"]
    lines = ["| metric | ref | run |", "|---|---:|---:|"]
    lines.append(f"| songs scored | {ref['songs']} | {run['songs']} |")
    lines.append(f"| lines | {ref['lines']} | {run['lines']} |")
    lines.append(f"| lines within 0.20 s of BK | {ref['within_20cs']} | {run['within_20cs']} |")
    for _, label in BUCKETS:
        lines.append(f"| {label} | {ref['buckets'].get(label, 0)} | {run['buckets'].get(label, 0)} |")
    lines.append(f"| lines off by more than 1 s | {ref['over_1s']} | {run['over_1s']} |")
    lines.append(f"| TRUSTED lines off by more than 1 s | {ref['trusted_over_1s']} | {run['trusted_over_1s']} |")
    status = lambda t: " / ".join(str(t["status"].get(k, 0)) for k in ("REVIEW", "UNTRUSTED", "TRUSTED"))
    lines.append(f"| REVIEW / UNTRUSTED / TRUSTED | {status(ref)} | {status(run)} |")
    lines.append("")
    lines.append(
        f"Lines that moved between ref and run: {len(result['moved_lines'])} in {result['moved_songs']} song(s); "
        f"{result['moved_over_20cs']} moved by more than 0.20 s; closer to BK: {result['moved_better']}, further: {result['moved_worse']}."
    )
    if result["moved_lines"]:
        lines.append("")
        lines.append("| report | entry | shift s | ref err s | run err s |")
        lines.append("|---|---:|---:|---:|---:|")
        for item in result["moved_lines"][:15]:
            lines.append(
                f"| {item['report']} | {item['entry']} | {item['shift_cs'] / 100:+.2f} | "
                f"{item['ref_err_cs'] / 100:.2f} | {item['run_err_cs'] / 100:.2f} |"
            )
    if result["skipped"]:
        lines.append("")
        lines.append("Skipped: " + "; ".join(f"{item['report']} ({item['reason']})" for item in result["skipped"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", type=Path, required=True, help="baseline run: folder of *.align-report.json")
    parser.add_argument("--run", type=Path, required=True, help="candidate run: folder of *.align-report.json")
    parser.add_argument("--bk", type=Path, required=True, help="checked reference LRC folder (searched recursively)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(args.ref, args.run, args.bk)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    text = markdown(result)
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
