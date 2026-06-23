#!/usr/bin/env python3
"""Compare a generated LRC against a checked reference LRC."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


TIMESTAMP_RE = re.compile(r"^\[(\d{1,3}):(\d{2})(?:\.(\d{1,3}))?\](.*)$")
META_RE = re.compile(r"^\[[A-Za-z]+:")


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_stdio()


@dataclass
class Entry:
    time_cs: int
    lines: list[str]


def parse_time_cs(match: re.Match[str]) -> int:
    fraction = match.group(3) or "0"
    centiseconds = int((fraction + "00")[:2])
    return ((int(match.group(1)) * 60) + int(match.group(2))) * 100 + centiseconds


def parse_lrc(path: Path) -> tuple[list[str], list[Entry]]:
    metadata: list[str] = []
    entries: list[Entry] = []
    last_time: int | None = None

    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        if META_RE.match(line):
            metadata.append(line)
            continue

        match = TIMESTAMP_RE.match(line)
        if not match:
            continue
        time_cs = parse_time_cs(match)
        text = match.group(4)
        if entries and time_cs == last_time:
            entries[-1].lines.append(text)
        else:
            entries.append(Entry(time_cs=time_cs, lines=[text]))
        last_time = time_cs

    return metadata, entries


def is_marker_entry(entry: Entry) -> bool:
    if not entry.lines:
        return True
    text = entry.lines[0].strip()
    marker_text = text.strip("()[]{}").strip().lower()
    marker_words = ("instrumental", "intro", "interlude", "outro", "間奏", "イントロ", "アウトロ")
    return text == "♪" or text == "" or (text.startswith("(") and any(word in marker_text for word in marker_words))


def is_generated_title_card(entry: Entry) -> bool:
    """Recognize the tool's fixed zero-time title card, not a real lyric."""
    return entry.time_cs == 0 and len(entry.lines) == 1 and " - " in entry.lines[0]


def summarize(reference: Path, generated: Path, ignore_markers: bool = False) -> dict[str, object]:
    ref_meta, ref_entries = parse_lrc(reference)
    gen_meta, gen_entries = parse_lrc(generated)
    ref_entries = [entry for entry in ref_entries if not is_generated_title_card(entry)]
    gen_entries = [entry for entry in gen_entries if not is_generated_title_card(entry)]
    if ignore_markers:
        ref_entries = [entry for entry in ref_entries if not is_marker_entry(entry)]
        gen_entries = [entry for entry in gen_entries if not is_marker_entry(entry)]
    pair_count = min(len(ref_entries), len(gen_entries))
    deltas = [
        gen_entries[idx].time_cs - ref_entries[idx].time_cs
        for idx in range(pair_count)
    ]
    abs_deltas = [abs(delta) for delta in deltas]
    text_mismatches = [
        idx + 1
        for idx in range(pair_count)
        if ref_entries[idx].lines != gen_entries[idx].lines
    ]

    def within(limit_cs: int) -> int:
        return sum(1 for delta in abs_deltas if delta <= limit_cs)

    denominator = len(ref_entries) if ref_entries else 1
    return {
        "reference": str(reference),
        "generated": str(generated),
        "metadata_equal": ref_meta == gen_meta,
        "reference_entries": len(ref_entries),
        "generated_entries": len(gen_entries),
        "entry_count_match": len(ref_entries) == len(gen_entries),
        "reference_display_lines": sum(len(entry.lines) for entry in ref_entries),
        "generated_display_lines": sum(len(entry.lines) for entry in gen_entries),
        "text_mismatches": len(text_mismatches) + abs(len(ref_entries) - len(gen_entries)),
        "text_mismatch_indices": text_mismatches[:25],
        "max_abs_delta_cs": max(abs_deltas) if abs_deltas else None,
        "mean_abs_delta_cs": round(sum(abs_deltas) / len(abs_deltas), 2) if abs_deltas else None,
        "within_10cs": within(10),
        "within_25cs": within(25),
        "within_50cs": within(50),
        "within_100cs": within(100),
        "within_10cs_percent": round(within(10) * 100 / denominator, 2),
        "within_25cs_percent": round(within(25) * 100 / denominator, 2),
        "within_50cs_percent": round(within(50) * 100 / denominator, 2),
        "within_100cs_percent": round(within(100) * 100 / denominator, 2),
        "nonzero_time_diffs": sum(1 for delta in deltas if delta != 0)
        + abs(len(ref_entries) - len(gen_entries)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated LRC timing against a reference LRC.")
    parser.add_argument("reference", help="Checked reference LRC")
    parser.add_argument("generated", help="Generated LRC to evaluate")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--ignore-markers",
        action="store_true",
        help="Ignore non-lyric marker entries such as ♪, (Intro), (Interlude), and (Outro).",
    )
    parser.add_argument(
        "--require-within-50cs",
        type=float,
        default=None,
        help="Fail unless this percent of entries are within +/-0.50s.",
    )
    parser.add_argument(
        "--require-within-100cs",
        type=float,
        default=None,
        help="Fail unless this percent of entries are within +/-1.00s.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = summarize(Path(args.reference).resolve(), Path(args.generated).resolve(), args.ignore_markers)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"entries: {result['generated_entries']} / {result['reference_entries']}")
        print(f"display lines: {result['generated_display_lines']} / {result['reference_display_lines']}")
        print(f"text mismatches: {result['text_mismatches']}")
        print(f"metadata equal: {result['metadata_equal']}")
        print(f"max abs delta: {result['max_abs_delta_cs']} cs")
        print(f"mean abs delta: {result['mean_abs_delta_cs']} cs")
        print(f"within +/-0.10s: {result['within_10cs_percent']}%")
        print(f"within +/-0.25s: {result['within_25cs_percent']}%")
        print(f"within +/-0.50s: {result['within_50cs_percent']}%")
        print(f"within +/-1.00s: {result['within_100cs_percent']}%")

    failed = False
    if args.require_within_50cs is not None:
        failed = failed or result["within_50cs_percent"] < args.require_within_50cs
    if args.require_within_100cs is not None:
        failed = failed or result["within_100cs_percent"] < args.require_within_100cs
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
