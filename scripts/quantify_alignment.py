#!/usr/bin/env python3
"""Build local quantitative alignment summaries from checked LRC references.

This script is intentionally local-data driven: it never needs bundled audio,
lyrics, or generated LRC fixtures.  It matches checked reference LRCs against
alignment reports in ``outputs/reports`` and can optionally regenerate fresh
candidate LRCs into ``outputs/quantitative/generated`` before evaluating.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from evaluate_lrc import summarize


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCES_DIR = Path(r"D:\Users\Administrator\Music\LRC tools checked references")
DEFAULT_REPORTS_DIR = PROJECT / "outputs" / "reports"
DEFAULT_OUTPUT_DIR = PROJECT / "outputs" / "quantitative"


@dataclass(frozen=True)
class MatchedCase:
    key: str
    title: str
    category: str
    reference: Path
    report_path: Path
    report: dict[str, object]


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_stdio()


def normalize_key(value: str) -> str:
    stem = Path(value).stem
    if "--" in stem:
        stem = stem.split("--", 1)[0]
    stem = re.sub(r"\.checked$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s+", "", stem)
    return stem.casefold()


def safe_stem(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip().strip(".")
    return cleaned or "case"


def infer_category(title: str) -> str:
    if any(token in title for token in ("ツユ", "雨", "過去", "終点", "バス", "朧月夜")):
        return "high-density / complex Japanese"
    if any(token in title for token in ("MyGO", "影色舞", "春日影")):
        return "band / dense mix"
    if any(token in title for token in ("鸣潮", "OST", "Original Soundtrack")):
        return "OST / sparse vocal"
    if any(token in title for token in ("Elegy", "Reveries")):
        return "standard / English vocal"
    return "standard / Japanese vocal"


def read_category_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("--category-map must point to a JSON object")
    return {normalize_key(str(key)): str(value) for key, value in payload.items()}


def load_reports(reports_dir: Path) -> dict[str, tuple[Path, dict[str, object]]]:
    reports: dict[str, tuple[Path, dict[str, object]]] = {}
    for path in sorted(reports_dir.glob("*.align-report.json"), key=lambda item: item.stat().st_mtime):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        keys = {normalize_key(path.name)}
        output_path = report.get("output_path")
        audio_path = report.get("audio_path")
        if isinstance(output_path, str):
            keys.add(normalize_key(Path(output_path).name))
        if isinstance(audio_path, str):
            keys.add(normalize_key(Path(audio_path).name))
        for key in keys:
            reports[key] = (path, report)
    return reports


def checked_references(references_dir: Path) -> Iterable[Path]:
    return sorted(references_dir.glob("*.lrc"), key=lambda path: path.name.casefold())


def match_cases(references_dir: Path, reports_dir: Path, category_map: dict[str, str]) -> list[MatchedCase]:
    reports = load_reports(reports_dir)
    cases: list[MatchedCase] = []
    for reference in checked_references(references_dir):
        key = normalize_key(reference.name)
        matched = reports.get(key)
        if matched is None:
            continue
        report_path, report = matched
        title = re.sub(r"\.checked$", "", reference.stem, flags=re.IGNORECASE)
        category = category_map.get(key, infer_category(title))
        cases.append(MatchedCase(key, title, category, reference, report_path, report))
    return cases


def report_for_output(reports_dir: Path, output_path: Path) -> tuple[Path | None, dict[str, object]]:
    target = output_path.resolve()
    newest: tuple[float, Path, dict[str, object]] | None = None
    for path in reports_dir.glob("*.align-report.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stored = report.get("output_path")
        if not isinstance(stored, str):
            continue
        try:
            if Path(stored).resolve() != target:
                continue
        except OSError:
            continue
        current = (path.stat().st_mtime, path, report)
        if newest is None or current[0] > newest[0]:
            newest = current
    if newest is None:
        return None, {}
    return newest[1], newest[2]


def regenerate_case(case: MatchedCase, output_dir: Path, reports_dir: Path) -> tuple[Path | None, Path | None, dict[str, object], str]:
    audio_value = case.report.get("audio_path")
    lyrics_value = case.report.get("lyrics_path")
    if not isinstance(audio_value, str) or not isinstance(lyrics_value, str):
        return None, None, {}, "missing audio_path or lyrics_path in report"
    audio = Path(audio_value)
    lyrics = Path(lyrics_value)
    if not audio.exists():
        return None, None, {}, f"missing audio: {audio}"
    if not lyrics.exists():
        return None, None, {}, f"missing lyrics: {lyrics}"

    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated = generated_dir / f"{safe_stem(case.title)}.generated.lrc"
    command = [
        sys.executable,
        str(PROJECT / "scripts" / "auto_lrc.py"),
        str(audio),
        "--lyrics",
        str(lyrics),
        "--timing-source",
        str(case.report.get("requested_timing_source") or "auto"),
        "--output",
        str(generated),
        "--overwrite",
        "--no-checked-lrc-hint",
        "--no-anchor-hints",
    ]
    if lyrics.suffix.lower() == ".lrc":
        command.append("--ignore-lyric-timestamps")
    proc = subprocess.run(command, cwd=PROJECT, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        return generated, None, {}, f"regenerate failed: exit {proc.returncode}"
    fresh_report_path, fresh_report = report_for_output(reports_dir, generated)
    return generated, fresh_report_path, fresh_report, ""


def existing_output(case: MatchedCase, use_reference_as_final: bool) -> tuple[Path | None, dict[str, object], str, str]:
    if use_reference_as_final:
        return case.reference, case.report, "", "checked-reference-as-final-output"
    output = case.report.get("output_path")
    if not isinstance(output, str):
        return None, {}, "missing output_path in report", "existing-report-output"
    path = Path(output)
    if not path.exists():
        return path, {}, f"missing generated output: {path}", "existing-report-output"
    return path, case.report, "", "existing-report-output"


def selected_backend(report: dict[str, object]) -> str:
    selection = report.get("candidate_selection")
    if isinstance(selection, dict):
        value = selection.get("selected_backend")
        if isinstance(value, str):
            return value
    value = report.get("backend")
    return str(value) if value is not None else ""


def metric_float(report: dict[str, object], key: str) -> float | None:
    value = report.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def evaluate_row(
    case: MatchedCase,
    generated: Path | None,
    report_path: Path | None,
    report: dict[str, object],
    mode: str,
    error: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "case": case.title,
        "category": case.category,
        "mode": mode,
        "status": "SKIP" if error else "OK",
        "error": error,
        "reference": str(case.reference),
        "generated": str(generated) if generated is not None else "",
        "report": str(report_path or case.report_path),
        "backend": report.get("backend", ""),
        "selected_backend": selected_backend(report),
        "ctc_audio_source": report.get("ctc_audio_source", ""),
        "ctc_device": report.get("ctc_device", report.get("whisperx_device", "")),
        "trusted_percent": metric_float(report, "trusted_percent"),
        "assigned_percent": metric_float(report, "assigned_percent"),
        "low_confidence_percent": metric_float(report, "low_confidence_percent"),
        "review_required_percent": metric_float(report, "review_required_percent"),
        "review_required_count": report.get("review_required_count", ""),
    }
    if error or generated is None or not generated.exists():
        return row
    try:
        summary = summarize(case.reference, generated)
    except Exception as exc:  # noqa: BLE001 - keep batch summaries resilient.
        row["status"] = "ERROR"
        row["error"] = str(exc)
        return row
    row.update(
        {
            "reference_entries": summary["reference_entries"],
            "generated_entries": summary["generated_entries"],
            "text_mismatches": summary["text_mismatches"],
            "max_abs_delta_cs": summary["max_abs_delta_cs"],
            "mean_abs_delta_cs": summary["mean_abs_delta_cs"],
            "within_10cs_percent": summary["within_10cs_percent"],
            "within_25cs_percent": summary["within_25cs_percent"],
            "within_50cs_percent": summary["within_50cs_percent"],
            "within_100cs_percent": summary["within_100cs_percent"],
        }
    )
    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    keys = [
        "case",
        "category",
        "mode",
        "status",
        "backend",
        "selected_backend",
        "ctc_audio_source",
        "ctc_device",
        "reference_entries",
        "generated_entries",
        "text_mismatches",
        "max_abs_delta_cs",
        "mean_abs_delta_cs",
        "within_10cs_percent",
        "within_25cs_percent",
        "within_50cs_percent",
        "within_100cs_percent",
        "trusted_percent",
        "assigned_percent",
        "low_confidence_percent",
        "review_required_percent",
        "review_required_count",
        "error",
        "reference",
        "generated",
        "report",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def weighted_percent(rows: list[dict[str, object]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for row in rows:
        try:
            entries = int(row.get("reference_entries") or row.get("generated_entries") or 0)
            value = float(row[key])
        except (TypeError, ValueError, KeyError):
            continue
        if entries <= 0:
            continue
        numerator += value * entries
        denominator += entries
    return round(numerator / denominator, 2) if denominator else None


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    ok_rows = [row for row in rows if row.get("status") == "OK"]
    by_category: dict[str, list[dict[str, object]]] = {}
    for row in ok_rows:
        by_category.setdefault(str(row["category"]), []).append(row)
    lines = [
        "# Quantitative Alignment Summary",
        "",
        "Generated from local checked references and alignment reports. Audio, lyrics, generated LRCs, and reports are not committed.",
        "",
        "| Category | Songs | Entries | <=0.10s | <=0.25s | <=0.50s | Trusted | Review required | Max error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, category_rows in sorted(by_category.items()):
        entries = sum(int(row.get("reference_entries") or 0) for row in category_rows)
        max_error = max((int(row.get("max_abs_delta_cs") or 0) for row in category_rows), default=0)
        lines.append(
            "| "
            + " | ".join(
                [
                    category,
                    str(len(category_rows)),
                    str(entries),
                    f"{weighted_percent(category_rows, 'within_10cs_percent') or 0:.2f}%",
                    f"{weighted_percent(category_rows, 'within_25cs_percent') or 0:.2f}%",
                    f"{weighted_percent(category_rows, 'within_50cs_percent') or 0:.2f}%",
                    f"{weighted_percent(category_rows, 'trusted_percent') or 0:.2f}%",
                    f"{weighted_percent(category_rows, 'review_required_percent') or 0:.2f}%",
                    f"{max_error / 100:.2f}s",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Song Results",
            "",
            "| Song | Category | Backend | Entries | <=0.25s | Max error | Trusted | Review count |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ok_rows:
        max_error = row.get("max_abs_delta_cs")
        max_error_text = "" if max_error in (None, "") else f"{int(max_error) / 100:.2f}s"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]).replace("|", "\\|"),
                    str(row["category"]),
                    str(row.get("selected_backend") or row.get("backend") or ""),
                    str(row.get("reference_entries") or ""),
                    f"{float(row.get('within_25cs_percent') or 0):.2f}%",
                    max_error_text,
                    f"{float(row.get('trusted_percent') or 0):.2f}%",
                    str(row.get("review_required_count") or 0),
                ]
            )
            + " |"
        )
    skipped = [row for row in rows if row.get("status") != "OK"]
    if skipped:
        lines.extend(["", "## Skipped / Errors", ""])
        for row in skipped:
            lines.append(f"- {row.get('case')}: {row.get('error')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize checked LRC alignment quality across local songs.")
    parser.add_argument("--references-dir", type=Path, default=DEFAULT_REFERENCES_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--category-map", type=Path, default=None, help="Optional JSON mapping from song title/stem to category.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate fresh outputs before evaluating.")
    parser.add_argument(
        "--use-reference-as-final-output",
        action="store_true",
        help="If the original generated output was moved away, evaluate the checked reference as the final reviewed output.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit matched cases, useful for smoke tests.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    category_map = read_category_map(args.category_map)
    cases = match_cases(args.references_dir, args.reports_dir, category_map)
    if args.limit is not None:
        cases = cases[: args.limit]
    rows: list[dict[str, object]] = []
    for case in cases:
        if args.regenerate:
            generated, report_path, report, error = regenerate_case(case, args.output_dir, args.reports_dir)
            rows.append(evaluate_row(case, generated, report_path, report, "regenerated", error))
        else:
            generated, report, error, mode = existing_output(case, args.use_reference_as_final_output)
            rows.append(evaluate_row(case, generated, case.report_path, report, mode, error))
    csv_path = args.output_dir / ("regenerated-summary.csv" if args.regenerate else "existing-summary.csv")
    md_path = args.output_dir / ("regenerated-summary.md" if args.regenerate else "existing-summary.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    ok = sum(1 for row in rows if row.get("status") == "OK")
    skipped = len(rows) - ok
    print(f"matched_cases={len(cases)} ok={ok} skipped_or_errors={skipped}")
    print(f"csv={csv_path}")
    print(f"markdown={md_path}")
    return 1 if ok == 0 and rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
