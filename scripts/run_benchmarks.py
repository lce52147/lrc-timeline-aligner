#!/usr/bin/env python3
"""Run the checked LRC timing regression gates for LRC tools."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from evaluate_lrc import summarize


PROJECT = Path(__file__).resolve().parents[1]
MUSIC = Path(os.environ.get("LRC_TOOLS_MUSIC_DIR", r"D:\MusicLibrary"))


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_stdio()


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    reference: Path
    generated: Path
    audio: Path | None = None
    lyrics: Path | None = None
    timing_source: str = "auto"
    ignore_markers: bool = False
    require_within_50cs: float = 100.0
    require_text_mismatches: int = 0
    require_max_abs_delta_cs: int | None = None
    require_report_device: str | None = None


def report_path(lrc_path: Path) -> Path:
    return lrc_path.with_suffix(".align-report.json")


def load_report(lrc_path: Path) -> dict[str, object]:
    path = report_path(lrc_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_cases() -> list[BenchmarkCase]:
    rain_ref = MUSIC / "01. 雨模様.lrc"
    utopia_ref = MUSIC / "11.ユートピア.lrc"
    oyasumi_ref = MUSIC / "01.おやすみモノクローム.lrc"
    rain_audio = MUSIC / "Music" / "TUYU" / "ツユ - 雨模様" / "01. 雨模様.flac"
    utopia_audio = MUSIC / "Music" / "Islet" / "magic" / "11.ユートピア.flac"
    oyasumi_audio = (
        MUSIC
        / "Music"
        / "Dreamin’ Her - 僕は、彼女の夢を見る。- Original Soundtrack"
        / "01.おやすみモノクローム.flac"
    )
    scripts = PROJECT / "scripts"
    return [
        BenchmarkCase(
            name="rain audio-only gpu",
            reference=rain_ref,
            generated=scripts / "rain_audio_gpu10.lrc",
            audio=rain_audio,
            lyrics=scripts / "rain_untimed.lyrics.txt",
            timing_source="whisperx",
            require_report_device="cuda",
        ),
        BenchmarkCase(
            name="utopia audio-only gpu",
            reference=utopia_ref,
            generated=scripts / "utopia_audio_gpu10.lrc",
            audio=utopia_audio,
            lyrics=scripts / "utopia_untimed.lyrics.txt",
            timing_source="whisperx",
            ignore_markers=True,
            require_report_device="cuda",
        ),
        BenchmarkCase(
            name="oyasumi audio-only gpu",
            reference=oyasumi_ref,
            generated=scripts / "oyasumi_audio_gpu10.lrc",
            audio=oyasumi_audio,
            lyrics=scripts / "oyasumi_monochrome_untimed.lyrics.txt",
            timing_source="whisperx",
            ignore_markers=True,
            require_report_device="cuda",
        ),
        BenchmarkCase(
            name="rain checked-lrc hint",
            reference=rain_ref,
            generated=scripts / "rain_auto_checked_hint.lrc",
            audio=rain_audio,
            lyrics=scripts / "rain_untimed.lyrics.txt",
            timing_source="auto",
            require_max_abs_delta_cs=0,
        ),
        BenchmarkCase(
            name="utopia checked-lrc hint",
            reference=utopia_ref,
            generated=scripts / "utopia_auto_checked_hint.lrc",
            audio=utopia_audio,
            lyrics=scripts / "utopia_untimed.lyrics.txt",
            timing_source="auto",
            ignore_markers=True,
            require_max_abs_delta_cs=0,
        ),
        BenchmarkCase(
            name="oyasumi checked-lrc hint",
            reference=oyasumi_ref,
            generated=scripts / "oyasumi_auto_checked_hint.lrc",
            audio=oyasumi_audio,
            lyrics=scripts / "oyasumi_monochrome_untimed.lyrics.txt",
            timing_source="auto",
            ignore_markers=True,
            require_max_abs_delta_cs=0,
        ),
    ]


def regenerate_case(case: BenchmarkCase) -> None:
    if case.audio is None or case.lyrics is None:
        raise RuntimeError(f"{case.name} does not define regeneration inputs")
    command = [
        sys.executable,
        str(PROJECT / "scripts" / "auto_lrc.py"),
        str(case.audio),
        "--lyrics",
        str(case.lyrics),
        "--timing-source",
        case.timing_source,
        "--output",
        str(case.generated),
        "--overwrite",
    ]
    print(f"GENERATE {case.name}: {case.generated.name}")
    proc = subprocess.run(command, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"regeneration failed for {case.name}: exit {proc.returncode}")


def evaluate_case(case: BenchmarkCase) -> tuple[bool, dict[str, object], list[str]]:
    result = summarize(case.reference, case.generated, case.ignore_markers)
    failures: list[str] = []
    if result["text_mismatches"] != case.require_text_mismatches:
        failures.append(f"text mismatches={result['text_mismatches']}")
    if result["within_50cs_percent"] < case.require_within_50cs:
        failures.append(f"within +/-0.50s={result['within_50cs_percent']}%")
    if case.require_max_abs_delta_cs is not None and result["max_abs_delta_cs"] != case.require_max_abs_delta_cs:
        failures.append(f"max delta={result['max_abs_delta_cs']} cs")
    if case.require_report_device is not None:
        report = load_report(case.generated)
        if report.get("whisperx_device") != case.require_report_device:
            failures.append(f"report device={report.get('whisperx_device')!r}")
    return not failures, result, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run checked LRC benchmark gates.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate benchmark LRC outputs with the current pipeline before evaluating.",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only run the three audio-only GPU cases.",
    )
    parser.add_argument(
        "--checked-hint-only",
        action="store_true",
        help="Only run the three checked-LRC hint cases.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    failed = False
    cases = benchmark_cases()
    if args.audio_only:
        cases = [case for case in cases if "audio-only" in case.name]
    if args.checked_hint_only:
        cases = [case for case in cases if "checked-lrc hint" in case.name]

    for case in cases:
        if args.regenerate:
            regenerate_case(case)
        ok, result, failures = evaluate_case(case)
        status = "PASS" if ok else "FAIL"
        print(
            f"{status} {case.name}: "
            f"entries {result['generated_entries']}/{result['reference_entries']}, "
            f"text mismatches {result['text_mismatches']}, "
            f"within +/-0.50s {result['within_50cs_percent']}%, "
            f"max {result['max_abs_delta_cs']} cs"
        )
        if failures:
            print("  " + "; ".join(failures))
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
