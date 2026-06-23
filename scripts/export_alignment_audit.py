#!/usr/bin/env python3
"""Export a compact per-line audit table from an LRC and its align report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from evaluate_lrc import Entry, parse_lrc


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_stdio()


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def default_report_path(lrc_path: Path) -> Path:
    sibling = lrc_path.with_suffix(".align-report.json")
    if sibling.exists():
        return sibling

    report_dir = Path(__file__).resolve().parents[1] / "outputs" / "reports"
    target_lrc = lrc_path.expanduser().resolve()
    if report_dir.exists():
        for candidate in sorted(report_dir.glob("*.align-report.json"), key=lambda path: path.stat().st_mtime, reverse=True):
            try:
                report = load_report(candidate)
            except (OSError, json.JSONDecodeError):
                continue
            output_path = report.get("output_path")
            if isinstance(output_path, str) and Path(output_path).expanduser().resolve() == target_lrc:
                return candidate
    return sibling


def format_time(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    value = float(seconds)
    minutes = int(value // 60)
    sec = value - minutes * 60
    return f"{minutes:02d}:{sec:05.2f}"


def entry_time(entry: Entry) -> str:
    return f"{entry.time_cs // 6000:02d}:{(entry.time_cs % 6000) / 100:05.2f}"


def flatten_lines(entry: Entry) -> str:
    return " / ".join(line.strip() for line in entry.lines if line.strip())


def by_entry(items: object) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        entry = item.get("entry")
        if isinstance(entry, int):
            result[entry] = item
    return result


def max_candidate_spread(candidate_timestamps: object) -> float | None:
    if not isinstance(candidate_timestamps, dict):
        return None
    values: list[float] = []
    for value in candidate_timestamps.values():
        if isinstance(value, (int, float)):
            values.append(float(value))
    if len(values) < 2:
        return None
    return max(values) - min(values)


def joined_flags(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    flags = item.get("flags")
    if not isinstance(flags, list):
        return ""
    return ";".join(str(flag) for flag in flags)


def joined_list(value: object) -> str:
    if not isinstance(value, list):
        return ""
    return ";".join(str(item) for item in value)


def candidate_value(candidate_timestamps: object, key: str) -> str:
    if not isinstance(candidate_timestamps, dict):
        return ""
    value = candidate_timestamps.get(key)
    if not isinstance(value, (int, float)):
        return ""
    return format_time(float(value))


def format_ctc_token_spans(value: object, limit: int = 6) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        char = str(item.get("char", ""))
        start = item.get("start")
        score = item.get("score")
        if not char or not isinstance(start, (int, float)):
            continue
        score_text = f"/{float(score):.3f}" if isinstance(score, (int, float)) else ""
        parts.append(f"{char}@{format_time(float(start))}{score_text}")
    return ";".join(parts)


def format_ctc_token_candidates(value: object, limit: int = 4) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        time_value = item.get("time")
        score = item.get("score")
        if not isinstance(time_value, (int, float)):
            continue
        score_text = f"/{float(score):.3f}" if isinstance(score, (int, float)) else ""
        parts.append(f"{format_time(float(time_value))}{score_text}")
    return ";".join(parts)


def format_penalties(value: object) -> str:
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "penalty"))
        amount = item.get("value")
        parts.append(f"{kind}:{float(amount):.2f}" if isinstance(amount, (int, float)) else kind)
    return ";".join(parts)


def row_for_entry(
    index: int,
    entry: Entry,
    assignments: dict[int, dict[str, Any]],
    suspicious: dict[int, dict[str, Any]],
) -> dict[str, str]:
    entry_number = index + 1
    assignment = assignments.get(entry_number, {})
    risk = suspicious.get(entry_number)
    candidates = risk.get("candidate_timestamps") if risk else assignment.get("timing_trusted_candidate_times")
    ctc_token_spans = assignment.get("ctc_token_spans") or (risk.get("ctc_token_spans") if risk else None)
    ctc_first_token_candidates = assignment.get("ctc_first_token_candidates") or (
        risk.get("ctc_first_token_candidates") if risk else None
    )
    spread = max_candidate_spread(candidates)
    score = assignment.get("score")
    ctc_score = assignment.get("ctc_score")
    timing_trusted = bool(assignment.get("timing_trusted"))
    trust_reason = str(assignment.get("timing_trusted_reason", "")) if timing_trusted else ""
    trust_sources = joined_list(assignment.get("timing_trusted_sources")) if timing_trusted else ""
    chosen_time = assignment.get("chosen_time")
    confidence = assignment.get("confidence")
    split_suggestion = assignment.get("split_suggestion")
    flags = joined_flags(risk)
    if timing_trusted:
        flags = ";".join(part for part in [flags, "timing_trusted"] if part)
    return {
        "entry": str(entry_number),
        "time": entry_time(entry),
        "review": "yes" if risk and risk.get("review_required") else "no",
        "severity": str(risk.get("severity", "")) if risk else "",
        "score": f"{float(score):.3f}" if isinstance(score, (int, float)) else "",
        "ctc_score": f"{float(ctc_score):.6f}" if isinstance(ctc_score, (int, float)) else "",
        "spread": f"{spread:.3f}" if spread is not None else "",
        "output": candidate_value(candidates, "output") or candidate_value(candidates, "selected") or entry_time(entry),
        "whisperx": candidate_value(candidates, "whisperx"),
        "ctc": candidate_value(candidates, "ctc"),
        "raw": candidate_value(candidates, "raw"),
        "forced_first": candidate_value(candidates, "whisperx_forced_first"),
        "chosen_time": format_time(chosen_time) if isinstance(chosen_time, (int, float)) else entry_time(entry),
        "confidence": f"{float(confidence):.3f}" if isinstance(confidence, (int, float)) else "",
        "decision_reasons": joined_list(assignment.get("reasons")),
        "decision_penalties": format_penalties(assignment.get("penalties")),
        "split_suggestion": str(split_suggestion.get("suggested_after_text", "")) if isinstance(split_suggestion, dict) else "",
        "timing_trusted": "yes" if timing_trusted else "no",
        "timing_trust_reason": trust_reason,
        "timing_trust_sources": trust_sources,
        "ctc_tokens": format_ctc_token_spans(ctc_token_spans),
        "ctc_first_token_candidates": format_ctc_token_candidates(ctc_first_token_candidates),
        "flags": flags,
        "text": flatten_lines(entry),
    }


def build_rows(lrc_path: Path, report: dict[str, Any], review_only: bool = False) -> list[dict[str, str]]:
    _, entries = parse_lrc(lrc_path)
    assignments = by_entry(report.get("assignments"))
    suspicious = by_entry(report.get("suspicious_alignments"))
    rows = [row_for_entry(index, entry, assignments, suspicious) for index, entry in enumerate(entries)]
    if review_only:
        rows = [row for row in rows if row["review"] == "yes"]
    return rows


def write_csv(rows: list[dict[str, str]], output: Path | None) -> None:
    fieldnames = [
        "entry",
        "time",
        "review",
        "severity",
        "score",
        "ctc_score",
        "spread",
        "output",
        "whisperx",
        "ctc",
        "raw",
        "forced_first",
        "chosen_time",
        "confidence",
        "decision_reasons",
        "decision_penalties",
        "split_suggestion",
        "timing_trusted",
        "timing_trust_reason",
        "timing_trust_sources",
        "ctc_tokens",
        "ctc_first_token_candidates",
        "flags",
        "text",
    ]
    if output:
        handle = output.open("w", encoding="utf-8-sig", newline="")
        close = True
    else:
        handle = sys.stdout
        close = False
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if close:
            handle.close()


def write_markdown(rows: list[dict[str, str]], report: dict[str, Any], output: Path | None) -> None:
    lines = [
        "# Alignment Audit",
        "",
        f"- backend: `{report.get('backend', '')}`",
        f"- strategy: `{report.get('strategy', '')}`",
        f"- timing entries: `{report.get('timing_entries', '')}`",
        f"- trusted percent: `{report.get('trusted_percent', report.get('matched_percent', ''))}`",
        f"- timing-trusted entries: `{report.get('timing_trusted_entries', '')}`",
        f"- review required: `{report.get('review_required_count', '')}`",
        "",
        "| # | time | chosen | confidence | review | trust | severity | spread | score | candidates | text |",
        "|---:|---:|---:|---:|:---:|---|:---:|---:|---:|---|---|",
    ]
    for row in rows:
        candidates = ", ".join(
            part
            for part in [
                f"out {row['output']}" if row["output"] else "",
                f"wx {row['whisperx']}" if row["whisperx"] else "",
                f"ctc {row['ctc']}" if row["ctc"] else "",
                f"raw {row['raw']}" if row["raw"] else "",
                f"ff {row['forced_first']}" if row["forced_first"] else "",
                f"ctc-tok {row['ctc_tokens']}" if row["ctc_tokens"] else "",
                f"ctc-first {row['ctc_first_token_candidates']}" if row["ctc_first_token_candidates"] else "",
            ]
            if part
        )
        text = row["text"].replace("|", "\\|")
        trust_parts = [
            "trusted" if row["timing_trusted"] == "yes" else "",
            row["timing_trust_reason"],
            row["timing_trust_sources"],
        ]
        trust = "<br>".join(part for part in trust_parts if part)
        flags = f"<br>{row['flags']}" if row["flags"] else ""
        lines.append(
            f"| {row['entry']} | {row['time']} | {row['chosen_time']} | {row['confidence']} | {row['review']} | {trust} | {row['severity']} | "
            f"{row['spread']} | {row['score']} | {candidates}{flags} | {text} |"
        )
    content = "\n".join(lines) + "\n"
    if output:
        output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


def write_anchor_template(rows: list[dict[str, str]], output: Path) -> None:
    lines = [
        "# Review these timestamps manually, then rename/copy to .anchors.lrc only after checking.",
        "# This file is intentionally named .anchor-template.lrc so it is not auto-applied.",
        "# Lines starting with # are ignored when this is copied to .anchors.lrc.",
    ]
    for row in rows:
        text = row["text"].split(" / ", 1)[0].strip()
        if not text:
            continue
        lines.append(f"# entry={row['entry']}")
        lines.append(f"[{row['time']}]{text}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a readable LRC alignment audit table.")
    parser.add_argument("lrc", help="Generated LRC file")
    parser.add_argument(
        "--report",
        default=None,
        help="Alignment report path. Defaults to a matching report in project outputs/reports, then legacy sibling path.",
    )
    parser.add_argument("--format", choices=("md", "csv"), default="md")
    parser.add_argument("--output", default=None, help="Output path. Defaults to stdout.")
    parser.add_argument("--review-only", action="store_true", help="Only include review-required rows.")
    parser.add_argument(
        "--anchor-template-output",
        default=None,
        help="Write a timestamped LRC subset template for manual review. This does not auto-apply anchors.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lrc_path = Path(args.lrc)
    report_path = Path(args.report) if args.report else default_report_path(lrc_path)
    output = Path(args.output) if args.output else None
    report = load_report(report_path)
    rows = build_rows(lrc_path, report, args.review_only)
    if args.anchor_template_output:
        write_anchor_template(rows, Path(args.anchor_template_output))
    if args.format == "csv":
        write_csv(rows, output)
    else:
        write_markdown(rows, report, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
