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
    reference: Path | None
    generated: Path
    audio: Path | None = None
    lyrics: Path | None = None
    timing_source: str = "auto"
    no_checked_lrc_hint: bool = False
    no_anchor_hints: bool = False
    ignore_markers: bool = False
    require_within_25cs: float | None = None
    require_within_50cs: float = 100.0
    require_text_mismatches: int = 0
    require_max_abs_delta_cs: int | None = None
    require_max_abs_delta_cs_at_most: int | None = None
    require_backend: str | None = None
    require_selected_backend: str | None = None
    require_report_device: str | None = None
    require_trusted_percent: float | None = None
    require_review_required_count: int | None = None
    require_review_required_entries: tuple[int, ...] | None = None
    require_ctc_local_fusion_count: int | None = None
    strict_review: bool = False
    ignore_lyric_timestamps: bool = False


def report_path(lrc_path: Path) -> Path:
    return lrc_path.with_suffix(".align-report.json")


def load_report(lrc_path: Path) -> dict[str, object]:
    legacy_path = report_path(lrc_path)
    if legacy_path.exists():
        return json.loads(legacy_path.read_text(encoding="utf-8"))
    # Reports are deliberately routed away from a user's music folders. Match
    # the output path stored in each report instead of guessing from its name.
    target = lrc_path.resolve()
    for path in (PROJECT / "outputs" / "reports").glob("*.align-report.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            output = report.get("output_path")
            if isinstance(output, str) and Path(output).resolve() == target:
                return report
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def parse_required_entries(value: str) -> tuple[int, ...]:
    entries: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped:
            entries.append(int(stripped))
    return tuple(sorted(entries))


def required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for --private-audit-only")
    return Path(value)


def required_env_int(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is required for --private-audit-only")
    return int(value)


def required_env_entries(name: str) -> tuple[int, ...]:
    value = os.environ.get(name)
    if value is None:
        if os.environ.get("LRC_TOOLS_PRIVATE_AUDIT_REVIEW_COUNT") == "0":
            return ()
        raise RuntimeError(f"{name} is required for --private-audit-only")
    if not value.strip():
        return ()
    return parse_required_entries(value)


def optional_env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return float(value)


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
            name="utopia ctc forced alignment",
            reference=utopia_ref,
            generated=scripts / "utopia_ctc_benchmark.lrc",
            audio=utopia_audio,
            lyrics=scripts / "utopia_untimed.lyrics.txt",
            timing_source="ctc",
            ignore_markers=True,
            require_within_25cs=100.0,
            require_backend="ctc",
            require_report_device="cuda",
        ),
        BenchmarkCase(
            name="oyasumi ctc forced alignment",
            reference=oyasumi_ref,
            generated=scripts / "oyasumi_ctc_benchmark.lrc",
            audio=oyasumi_audio,
            lyrics=scripts / "oyasumi_monochrome_untimed.lyrics.txt",
            timing_source="ctc",
            ignore_markers=True,
            require_within_25cs=100.0,
            require_backend="ctc",
            require_report_device="cuda",
        ),
        BenchmarkCase(
            name="rain auto backend selection",
            reference=rain_ref,
            generated=scripts / "rain_auto_selection_benchmark.lrc",
            audio=rain_audio,
            lyrics=scripts / "rain_untimed.lyrics.txt",
            timing_source="auto",
            no_checked_lrc_hint=True,
            require_within_25cs=100.0,
            require_backend="hybrid",
            require_selected_backend="hybrid",
            require_report_device="cuda",
        ),
        BenchmarkCase(
            name="utopia auto backend selection",
            reference=utopia_ref,
            generated=scripts / "utopia_auto_selection_benchmark.lrc",
            audio=utopia_audio,
            lyrics=scripts / "utopia_untimed.lyrics.txt",
            timing_source="auto",
            no_checked_lrc_hint=True,
            ignore_markers=True,
            require_within_25cs=100.0,
            require_backend="ctc",
            require_selected_backend="ctc",
            require_report_device="cuda",
        ),
        BenchmarkCase(
            name="oyasumi auto backend selection",
            reference=oyasumi_ref,
            generated=scripts / "oyasumi_auto_selection_benchmark.lrc",
            audio=oyasumi_audio,
            lyrics=scripts / "oyasumi_monochrome_untimed.lyrics.txt",
            timing_source="auto",
            no_checked_lrc_hint=True,
            ignore_markers=True,
            require_within_25cs=100.0,
            require_backend="ctc",
            require_selected_backend="ctc",
            require_report_device="cuda",
        ),
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


def local_regression_cases() -> list[BenchmarkCase]:
    """Private checked songs that are expected to exist only on the local box.

    Checked references must live outside the source-audio folder. In
    particular, an adjacent ``Song.lrc`` is the drag/drop output path and must
    never become its own benchmark oracle.
    """
    magic = MUSIC / "Music" / "Islet" / "magic"
    scripts = PROJECT / "scripts"
    checked_root = Path(
        os.environ.get(
            "LRC_TOOLS_CHECKED_REFERENCE_DIR",
            str(MUSIC / "LRC tools checked references"),
        )
    )
    cases = [
        BenchmarkCase(
            name="hakobune auto local regression",
            reference=magic / "10.方舟.lrc",
            generated=scripts / "hakobune_auto_local_regression.lrc",
            audio=magic / "10.方舟.flac",
            lyrics=magic / "10.方舟.txt",
            timing_source="auto",
            no_checked_lrc_hint=True,
            no_anchor_hints=True,
            require_within_25cs=90.0,
            require_within_50cs=95.0,
            require_max_abs_delta_cs_at_most=60,
            require_backend="ctc",
            require_selected_backend="ctc",
            require_report_device="cuda",
            require_trusted_percent=100.0,
            require_review_required_count=0,
            strict_review=True,
        ),
    ]
    rainfall_checked = checked_root / "09. ツユ - レインフォール.checked.lrc"
    rainfall_dir = MUSIC / "Music" / "TUYU" / "ツユ – アンダーメンタリティ (2023.06.21)[Hi-Res FLAC]"
    if rainfall_checked.exists():
        cases.append(
            BenchmarkCase(
                name="rainfall human-checked local regression",
                reference=rainfall_checked,
                generated=scripts / "rainfall_auto_local_regression.lrc",
                audio=rainfall_dir / "09. ツユ - レインフォール.flac",
                lyrics=rainfall_checked,
                timing_source="auto",
                no_checked_lrc_hint=True,
                no_anchor_hints=True,
                ignore_lyric_timestamps=True,
                require_within_25cs=100.0,
                require_max_abs_delta_cs=0,
                require_backend="ctc",
                require_selected_backend="ctc",
                require_report_device="cuda",
            )
        )
    else:
        print(
            "SKIP rainfall human-checked local regression: expected independent checked reference "
            f"{rainfall_checked}"
        )
    kashiyo_checked = checked_root / "04.可惜夜.checked.lrc"
    if kashiyo_checked.exists():
        cases.insert(
            0,
            BenchmarkCase(
                name="kashiyo auto local regression",
                reference=kashiyo_checked,
                generated=scripts / "kashiyo_auto_local_regression.lrc",
                audio=magic / "04.可惜夜.flac",
                lyrics=magic / "04.可惜夜.txt",
                timing_source="auto",
                no_checked_lrc_hint=True,
                no_anchor_hints=True,
                require_within_25cs=100.0,
                require_max_abs_delta_cs=0,
                require_backend="hybrid",
                require_selected_backend="hybrid",
                require_report_device="cuda",
                require_trusted_percent=100.0,
                require_review_required_count=0,
                require_ctc_local_fusion_count=3,
                strict_review=True,
            ),
        )
    else:
        print(
            "SKIP kashiyo auto local regression: expected independent checked reference "
            f"{kashiyo_checked}"
        )
    return cases


def existing_local_regression_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for case in local_regression_cases():
        required_paths = [case.reference, case.audio, case.lyrics]
        missing = [path for path in required_paths if path is not None and not path.exists()]
        if missing:
            print(f"SKIP {case.name}: missing local file(s): {', '.join(str(path) for path in missing)}")
            continue
        cases.append(case)
    return cases


def private_audit_cases() -> list[BenchmarkCase]:
    scripts = PROJECT / "scripts"
    name = os.environ.get("LRC_TOOLS_PRIVATE_AUDIT_NAME", "private risk audit")
    return [
        BenchmarkCase(
            name=name,
            reference=None,
            generated=Path(os.environ.get("LRC_TOOLS_PRIVATE_AUDIT_OUTPUT", scripts / "private_auto_audit.lrc")),
            audio=required_env_path("LRC_TOOLS_PRIVATE_AUDIT_AUDIO"),
            lyrics=required_env_path("LRC_TOOLS_PRIVATE_AUDIT_LYRICS"),
            timing_source="auto",
            no_checked_lrc_hint=True,
            no_anchor_hints=True,
            require_backend=os.environ.get("LRC_TOOLS_PRIVATE_AUDIT_BACKEND", "hybrid"),
            require_selected_backend=os.environ.get("LRC_TOOLS_PRIVATE_AUDIT_SELECTED_BACKEND", "hybrid"),
            require_report_device=os.environ.get("LRC_TOOLS_PRIVATE_AUDIT_DEVICE", "cuda"),
            require_trusted_percent=optional_env_float("LRC_TOOLS_PRIVATE_AUDIT_TRUSTED_PERCENT"),
            require_review_required_count=required_env_int("LRC_TOOLS_PRIVATE_AUDIT_REVIEW_COUNT"),
            require_review_required_entries=required_env_entries("LRC_TOOLS_PRIVATE_AUDIT_REVIEW_ENTRIES"),
            require_ctc_local_fusion_count=required_env_int("LRC_TOOLS_PRIVATE_AUDIT_FUSION_COUNT"),
            strict_review=os.environ.get("LRC_TOOLS_PRIVATE_AUDIT_STRICT", "1") != "0",
        )
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
    if case.no_checked_lrc_hint:
        command.append("--no-checked-lrc-hint")
    if case.no_anchor_hints:
        command.append("--no-anchor-hints")
    if case.ignore_lyric_timestamps:
        command.append("--ignore-lyric-timestamps")
    if case.strict_review:
        command.append("--strict-review")
    print(f"GENERATE {case.name}: {case.generated.name}")
    proc = subprocess.run(command, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"regeneration failed for {case.name}: exit {proc.returncode}")


def evaluate_case(case: BenchmarkCase) -> tuple[bool, dict[str, object], list[str]]:
    if case.reference is None:
        result: dict[str, object] = {
            "reference": None,
            "generated": str(case.generated),
            "reference_entries": None,
            "generated_entries": None,
            "text_mismatches": None,
            "within_50cs_percent": None,
            "max_abs_delta_cs": None,
        }
    else:
        result = summarize(case.reference, case.generated, case.ignore_markers)
    failures: list[str] = []
    if case.reference is not None and result["text_mismatches"] != case.require_text_mismatches:
        failures.append(f"text mismatches={result['text_mismatches']}")
    if (
        case.reference is not None
        and case.require_within_25cs is not None
        and result["within_25cs_percent"] < case.require_within_25cs
    ):
        failures.append(f"within +/-0.25s={result['within_25cs_percent']}%")
    if case.reference is not None and result["within_50cs_percent"] < case.require_within_50cs:
        failures.append(f"within +/-0.50s={result['within_50cs_percent']}%")
    if (
        case.reference is not None
        and case.require_max_abs_delta_cs is not None
        and result["max_abs_delta_cs"] != case.require_max_abs_delta_cs
    ):
        failures.append(f"max delta={result['max_abs_delta_cs']} cs")
    if (
        case.reference is not None
        and case.require_max_abs_delta_cs_at_most is not None
        and result["max_abs_delta_cs"] > case.require_max_abs_delta_cs_at_most
    ):
        failures.append(f"max delta={result['max_abs_delta_cs']} cs")
    report = load_report(case.generated)
    if case.require_backend is not None:
        if report.get("backend") != case.require_backend:
            failures.append(f"report backend={report.get('backend')!r}")
    if case.require_selected_backend is not None:
        selection = report.get("candidate_selection")
        selected_backend = selection.get("selected_backend") if isinstance(selection, dict) else None
        if selected_backend != case.require_selected_backend:
            failures.append(f"selected backend={selected_backend!r}")
    if case.require_report_device is not None:
        device = report.get("ctc_device") if report.get("backend") == "ctc" else report.get("whisperx_device")
        if device != case.require_report_device:
            failures.append(f"report device={device!r}")
    if case.require_trusted_percent is not None:
        try:
            trusted_percent = float(report.get("trusted_percent", report.get("matched_percent", 0.0)) or 0.0)
        except (TypeError, ValueError):
            trusted_percent = 0.0
        if trusted_percent < case.require_trusted_percent:
            failures.append(f"trusted percent={trusted_percent}")
    if case.require_review_required_count is not None:
        review_count = report.get("review_required_count")
        if review_count != case.require_review_required_count:
            failures.append(f"review required count={review_count!r}")
    if case.require_review_required_entries is not None:
        suspicious = report.get("suspicious_alignments")
        review_entry_numbers: list[int] = []
        if isinstance(suspicious, list):
            for item in suspicious:
                if isinstance(item, dict) and item.get("review_required") and isinstance(item.get("entry"), int):
                    review_entry_numbers.append(int(item["entry"]))
        review_entries = tuple(sorted(review_entry_numbers))
        if review_entries != case.require_review_required_entries:
            failures.append(f"review entries={review_entries!r}")
    if case.require_ctc_local_fusion_count is not None:
        fusion_count = report.get("ctc_local_fusion_count")
        if fusion_count != case.require_ctc_local_fusion_count:
            failures.append(f"ctc local fusion count={fusion_count!r}")
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
    parser.add_argument(
        "--ctc-only",
        action="store_true",
        help="Only run the CTC forced-alignment backend cases.",
    )
    parser.add_argument(
        "--auto-selection-only",
        action="store_true",
        help="Only run the auto CTC/WhisperX backend-selection cases.",
    )
    parser.add_argument(
        "--private-audit-only",
        action="store_true",
        help="Only run local private audit cases that do not have public reference LRCs.",
    )
    parser.add_argument(
        "--local-regression-only",
        action="store_true",
        help="Only run local private checked-song regression cases when their files exist.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    failed = False
    if args.private_audit_only and args.local_regression_only:
        raise RuntimeError("--private-audit-only and --local-regression-only cannot be combined")
    if args.private_audit_only:
        cases = private_audit_cases()
    elif args.local_regression_only:
        cases = existing_local_regression_cases()
    else:
        cases = benchmark_cases()
    if args.audio_only:
        cases = [case for case in cases if "audio-only" in case.name]
    if args.checked_hint_only:
        cases = [case for case in cases if "checked-lrc hint" in case.name]
    if args.ctc_only:
        cases = [case for case in cases if "ctc forced alignment" in case.name]
    if args.auto_selection_only:
        cases = [case for case in cases if "auto backend selection" in case.name]

    for case in cases:
        if args.regenerate:
            regenerate_case(case)
        ok, result, failures = evaluate_case(case)
        status = "PASS" if ok else "FAIL"
        if case.reference is None:
            report = load_report(case.generated)
            print(
                f"{status} {case.name}: "
                f"backend {report.get('backend')!r}, "
                f"trusted {report.get('trusted_percent')}, "
                f"review-required {report.get('review_required_count')}, "
                f"fusion {report.get('ctc_local_fusion_count')}"
            )
        else:
            print(
                f"{status} {case.name}: "
                f"entries {result['generated_entries']}/{result['reference_entries']}, "
                f"text mismatches {result['text_mismatches']}, "
                f"within +/-0.25s {result['within_25cs_percent']}%, "
                f"within +/-0.50s {result['within_50cs_percent']}%, "
                f"max {result['max_abs_delta_cs']} cs"
            )
        if failures:
            print("  " + "; ".join(failures))
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
