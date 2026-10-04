#!/usr/bin/env python3
"""Compare a generated LRC against a checked reference LRC."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
import unicodedata
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


def normalized_sung_text(entry: Entry) -> str:
    if not entry.lines:
        return ""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", entry.lines[0])).casefold()


def entry_detail(index: int, entry: Entry) -> dict[str, object]:
    return {
        "index": index + 1,
        "time_cs": entry.time_cs,
        "text": entry.lines[0] if entry.lines else "",
        "display_lines": list(entry.lines),
    }


def align_entries_by_text(
    reference: list[Entry], generated: list[Entry]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Align exact normalized sung-text rows while preserving sequence order."""
    ref_keys = [normalized_sung_text(entry) for entry in reference]
    gen_keys = [normalized_sung_text(entry) for entry in generated]
    matcher = difflib.SequenceMatcher(a=ref_keys, b=gen_keys, autojunk=False)
    pairs: list[tuple[int, int]] = []
    matched_ref: set[int] = set()
    matched_gen: set[int] = set()
    for match in matcher.get_matching_blocks():
        for offset in range(match.size):
            ref_index = match.a + offset
            gen_index = match.b + offset
            pairs.append((ref_index, gen_index))
            matched_ref.add(ref_index)
            matched_gen.add(gen_index)
    unmatched_ref = [index for index in range(len(reference)) if index not in matched_ref]
    unmatched_gen = [index for index in range(len(generated)) if index not in matched_gen]
    return pairs, unmatched_ref, unmatched_gen


def summarize(reference: Path, generated: Path, ignore_markers: bool = False) -> dict[str, object]:
    ref_meta, ref_entries = parse_lrc(reference)
    gen_meta, gen_entries = parse_lrc(generated)
    ref_entries = [entry for entry in ref_entries if not is_generated_title_card(entry)]
    gen_entries = [entry for entry in gen_entries if not is_generated_title_card(entry)]
    # Non-lyric marker rows never own a lyric timing comparison.  Keep the
    # compatibility argument because older callers still pass it explicitly.
    ref_entries = [entry for entry in ref_entries if not is_marker_entry(entry)]
    gen_entries = [entry for entry in gen_entries if not is_marker_entry(entry)]
    pairs, unmatched_ref, unmatched_gen = align_entries_by_text(ref_entries, gen_entries)
    deltas = [
        gen_entries[gen_index].time_cs - ref_entries[ref_index].time_cs
        for ref_index, gen_index in pairs
    ]
    abs_deltas = [abs(delta) for delta in deltas]
    text_mismatches = [
        (ref_index + 1, gen_index + 1)
        for ref_index, gen_index in pairs
        if ref_entries[ref_index].lines != gen_entries[gen_index].lines
    ]

    def within(limit_cs: int) -> int:
        return sum(1 for delta in abs_deltas if delta <= limit_cs)

    legacy_denominator = len(pairs) if pairs else 1
    tier_denominator = len(pairs) if pairs else 1
    correct_le_30ms = sum(1 for delta in abs_deltas if delta <= 3)
    acceptable_30_to_50ms = sum(1 for delta in abs_deltas if 3 < delta <= 5)
    wrong_gt_50ms = sum(1 for delta in abs_deltas if delta > 5)
    abs_delta_ms = [delta * 10 for delta in abs_deltas]
    aligned_pairs = [
        {
            "reference_index": ref_index + 1,
            "generated_index": gen_index + 1,
            "text": ref_entries[ref_index].lines[0] if ref_entries[ref_index].lines else "",
            "reference_time_cs": ref_entries[ref_index].time_cs,
            "generated_time_cs": gen_entries[gen_index].time_cs,
            "delta_ms": deltas[pair_index] * 10,
            "abs_delta_ms": abs_deltas[pair_index] * 10,
        }
        for pair_index, (ref_index, gen_index) in enumerate(pairs)
    ]
    return {
        "reference": str(reference),
        "generated": str(generated),
        "metadata_equal": ref_meta == gen_meta,
        "reference_entries": len(ref_entries),
        "generated_entries": len(gen_entries),
        "entry_count_match": len(ref_entries) == len(gen_entries),
        "reference_display_lines": sum(len(entry.lines) for entry in ref_entries),
        "generated_display_lines": sum(len(entry.lines) for entry in gen_entries),
        "text_mismatches": len(text_mismatches) + len(unmatched_ref) + len(unmatched_gen),
        "text_mismatch_indices": [item[0] for item in text_mismatches[:25]],
        "text_mismatch_pairs": [
            {"reference_index": ref_index, "generated_index": gen_index}
            for ref_index, gen_index in text_mismatches[:25]
        ],
        "timing_compared_entries": len(pairs),
        "aligned_pairs": aligned_pairs,
        "unmatched_reference_count": len(unmatched_ref),
        "unmatched_generated_count": len(unmatched_gen),
        "unmatched_reference_entries": [
            entry_detail(index, ref_entries[index]) for index in unmatched_ref
        ],
        "unmatched_generated_entries": [
            entry_detail(index, gen_entries[index]) for index in unmatched_gen
        ],
        "max_abs_delta_cs": max(abs_deltas) if abs_deltas else None,
        "mean_abs_delta_cs": round(sum(abs_deltas) / len(abs_deltas), 2) if abs_deltas else None,
        "max_abs_delta_ms": max(abs_delta_ms) if abs_delta_ms else None,
        "median_abs_delta_ms": (
            round(float(statistics.median(abs_delta_ms)), 2) if abs_delta_ms else None
        ),
        "mae_ms": round(sum(abs_delta_ms) / len(abs_delta_ms), 2) if abs_delta_ms else None,
        "correct_le_30ms": correct_le_30ms,
        "acceptable_30_to_50ms": acceptable_30_to_50ms,
        "wrong_gt_50ms": wrong_gt_50ms,
        "correct_le_30ms_percent": round(correct_le_30ms * 100 / tier_denominator, 2),
        "acceptable_30_to_50ms_percent": round(
            acceptable_30_to_50ms * 100 / tier_denominator, 2
        ),
        "wrong_gt_50ms_percent": round(wrong_gt_50ms * 100 / tier_denominator, 2),
        "within_10cs": within(10),
        "within_25cs": within(25),
        "within_50cs": within(50),
        "within_100cs": within(100),
        "within_10cs_percent": round(within(10) * 100 / legacy_denominator, 2),
        "within_25cs_percent": round(within(25) * 100 / legacy_denominator, 2),
        "within_50cs_percent": round(within(50) * 100 / legacy_denominator, 2),
        "within_100cs_percent": round(within(100) * 100 / legacy_denominator, 2),
        "legacy_timing_metrics_note": (
            "Legacy centisecond fields are retained for compatibility: "
            "within_10cs means <=100 ms, within_25cs <=250 ms, "
            "within_50cs <=500 ms, within_100cs <=1000 ms. "
            "Percentages use text-aligned timing pairs as the denominator."
        ),
        "nonzero_time_diffs": sum(1 for delta in deltas if delta != 0),
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
        print(f"timing compared: {result['timing_compared_entries']}")
        print(f"display lines: {result['generated_display_lines']} / {result['reference_display_lines']}")
        print(f"text mismatches: {result['text_mismatches']}")
        print(f"metadata equal: {result['metadata_equal']}")
        print(f"correct <=30 ms: {result['correct_le_30ms_percent']}%")
        print(f"acceptable 30-50 ms: {result['acceptable_30_to_50ms_percent']}%")
        print(f"wrong >50 ms: {result['wrong_gt_50ms_percent']}%")
        print(f"median abs delta: {result['median_abs_delta_ms']} ms")
        print(f"MAE: {result['mae_ms']} ms")
        print(f"max abs delta: {result['max_abs_delta_ms']} ms")

    failed = False
    if args.require_within_50cs is not None:
        failed = failed or result["within_50cs_percent"] < args.require_within_50cs
    if args.require_within_100cs is not None:
        failed = failed or result["within_100cs_percent"] < args.require_within_100cs
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
