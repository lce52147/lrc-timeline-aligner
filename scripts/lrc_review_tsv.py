"""Non-destructive review queue sidecar for LRC production reports."""
from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _fast_singing(assignments: list[object]) -> str:
    measured = 0
    for previous, current in zip(assignments, assignments[1:]):
        if not isinstance(previous, dict) or not isinstance(current, dict):
            continue
        start = _number(previous.get("ctc_first_token_start"))
        end = _number(current.get("ctc_first_token_start"))
        romanized = str(previous.get("romaji") or "")
        vowel_groups = len(re.findall(r"[aeiou]+", romanized, flags=re.IGNORECASE))
        if start is None or end is None or vowel_groups == 0 or end <= start:
            continue
        measured += 1
        if (end - start) / vowel_groups < 0.35:
            return "TRUE"
    return "FALSE" if measured else "UNKNOWN"


def _clock(value: object) -> str:
    seconds = _number(value)
    if seconds is None or seconds < 0:
        return "UNKNOWN"
    centiseconds = int(round(seconds * 100))
    minutes, rem = divmod(centiseconds, 6000)
    return f"{minutes:02d}:{rem // 100:02d}.{rem % 100:02d}"


def write_review_tsv(
    path: Path, report: dict[str, object], entries: list[Any]
) -> Path:
    """Write every REVIEW/UNTRUSTED lyric line, even outside strict mode."""
    assignments = report.get("assignments")
    rows = assignments if isinstance(assignments, list) else []
    if len(rows) != len(entries):
        if not rows:
            rows = [{"entry": i + 1, "verification_status": "unverified"} for i in range(len(entries))]
        else:
            raise ValueError("Review sidecar cannot silently drop or merge lyric entries")
    review_count = sum(
        bool(isinstance(row, dict) and (
            row.get("review_required") or row.get("verification_status") == "review-required"
        )) for row in rows
    )
    percent = round(100.0 * review_count / len(rows), 2) if rows else 0.0
    preflight = report.get("lyric_preflight")
    preflight_status = (
        str(preflight.get("status") or "NOT_RUN")
        if isinstance(preflight, dict) else "NOT_RUN"
    )
    lines = [
        f"# fast_singing={_fast_singing(rows)}\treview_percent={percent}\tpreflight={preflight_status}",
        "entry\ttime\tlyric\tstatus\treason_codes",
    ]
    for index, (row, entry) in enumerate(zip(rows, entries), start=1):
        if not isinstance(row, dict):
            raise ValueError("Review sidecar assignment is malformed")
        review = bool(row.get("review_required") or row.get("verification_status") == "review-required")
        trusted = row.get("verification_status") == "trusted" and row.get("overall_trusted") is not False
        if trusted and not review:
            continue
        status = "REVIEW" if review else "UNTRUSTED"
        text = " / ".join(str(part) for part in entry.lines)
        text = re.sub(r"[\t\r\n]+", " ", text)
        reason_codes = row.get("final_trust_reason_codes")
        if isinstance(reason_codes, list):
            reasons = [str(code) for code in reason_codes]
        else:
            reasons = [str(code) for code in row.get("flags", [])] if isinstance(row.get("flags"), list) else []
        if not reasons:
            reasons = ["review-required" if review else "unverified"]
        lines.append(
            f"{index}\t{_clock(row.get('timestamp'))}\t{text}\t{status}\t{','.join(reasons)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path
