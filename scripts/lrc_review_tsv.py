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


def _optional_clock(value: object) -> str:
    return "" if _number(value) is None else _clock(value)


def _clean_field(value: object) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")


def _reason_code(value: object) -> str:
    if isinstance(value, dict):
        if "code" in value:
            return _reason_code(value["code"])
        if "reason" in value:
            return _reason_code(value["reason"])
    return str(value)


def reason_codes_for(row: object, status: object) -> list[str]:
    """Return stable, display-ready reason codes without changing classification."""
    data = row if isinstance(row, dict) else {}
    final_codes = data.get("final_trust_reason_codes")
    if isinstance(final_codes, list) and final_codes:
        values = final_codes
    else:
        flags = data.get("flags")
        if isinstance(flags, list) and flags:
            values = flags
        else:
            legacy = data.get("reason_codes")
            values = [legacy] if legacy not in (None, "") and not isinstance(legacy, list) else (
                legacy if isinstance(legacy, list) else []
            )
    reasons = [_reason_code(value) for value in values]
    reasons = [reason for reason in reasons if reason != ""]
    if not reasons:
        reasons = ["review-required" if str(status) == "REVIEW" else "unverified"]
    return reasons


def _flatten_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_strings(item))
        return result
    if isinstance(value, dict):
        result = []
        for key in ("code", "name", "reason", "type", "flag", "class", "finding"):
            if key in value:
                result.extend(_flatten_strings(value[key]))
        return result
    return []


def _candidate_times(value: object) -> list[float]:
    result: list[float] = []
    if isinstance(value, list):
        for item in value:
            result.extend(_candidate_times(item))
    elif isinstance(value, dict):
        for key in ("time", "timestamp", "chosen_time", "candidate_time", "raw_candidate_time", "start", "onset"):
            number = _number(value.get(key))
            if number is not None:
                result.append(number)
        for key in ("candidates", "alignment_hypotheses", "hypotheses", "candidate"):
            result.extend(_candidate_times(value.get(key)))
    return result


def _distinct_sorted(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        if not result or abs(value - result[-1]) > 1e-6:
            result.append(value)
    return result


def _central_rows(report: dict[str, object]) -> dict[int, dict[str, object]]:
    rows = report.get("central_timing_state")
    if not isinstance(rows, list):
        return {}
    result: dict[int, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            entry = int(row["entry"])
        except (KeyError, TypeError, ValueError):
            continue
        result[entry] = row
    return result


def severity_features(report: dict[str, object]) -> dict[int, dict[str, object]]:
    """Build immutable review ordering features from serialized report fields."""
    assignments = report.get("assignments")
    rows = assignments if isinstance(assignments, list) else []
    central = _central_rows(report)
    consensus = report.get("multimodel_consensus")
    disagreement_entries: set[int] = set()
    if isinstance(consensus, dict):
        disagreements = consensus.get("review_only_model_disagreements")
        if isinstance(disagreements, list):
            for item in disagreements:
                if isinstance(item, dict):
                    try:
                        disagreement_entries.add(int(item["entry"]))
                    except (KeyError, TypeError, ValueError):
                        pass

    result: dict[int, dict[str, object]] = {}
    for assignment in rows:
        if not isinstance(assignment, dict):
            continue
        try:
            entry = int(assignment["entry"])
        except (KeyError, TypeError, ValueError):
            continue
        strings: list[str] = []
        for field in (
            "flags", "review_reason_classes", "active_timing_findings",
            "resolved_timing_findings", "final_trust_reason_codes", "self_check_flags",
        ):
            strings.extend(_flatten_strings(assignment.get(field)))
        lowered = {value.lower() for value in strings}
        central_row = central.get(entry, {})
        review = bool(
            assignment.get("review_required")
            or assignment.get("verification_status") == "review-required"
        )
        status = "REVIEW" if review else "UNTRUSTED"
        timestamp_status = central_row.get("timestamp_status") or assignment.get("timestamp_status")
        provisional = str(timestamp_status or "").lower().startswith("provisional") or any(
            "provisional" in value for value in lowered
        )
        model_disagreement = entry in disagreement_entries or bool(
            {"model-disagreement", "model_disagreement"} & lowered
        )
        candidate_values = _distinct_sorted(
            _candidate_times(assignment.get("candidates"))
            + _candidate_times(assignment.get("alignment_hypotheses"))
        )
        central_cs = _number(central_row.get("canonical_written_centiseconds"))
        assignment_cs = _number(assignment.get("canonical_written_centiseconds"))
        lrc_time = (
            central_cs / 100.0 if central_cs is not None
            else assignment_cs / 100.0 if assignment_cs is not None
            else _number(assignment.get("timestamp"))
        )
        result[entry] = {
            "status": status,
            "timestamp_status": timestamp_status,
            "lrc_time": lrc_time,
            "candidate_times": candidate_values,
            "candidate_spread_s": (
                candidate_values[-1] - candidate_values[0]
                if len(candidate_values) >= 2 else None
            ),
            "provisional": provisional,
            "model_disagreement": model_disagreement,
            "severity_score": 4 * int(review) + 2 * int(provisional) + int(model_disagreement),
        }
    for entry, feature in result.items():
        previous = result.get(entry - 1, {}).get("lrc_time")
        following = result.get(entry + 1, {}).get("lrc_time")
        feature["prev_time"] = previous
        feature["next_time"] = following
    return result


def rank_rows(report: dict[str, object], rows_listed: list[object]) -> list[object]:
    features = severity_features(report)
    indexed: list[tuple[object, int, int]] = []
    for order, row in enumerate(rows_listed):
        entry_value = row.get("entry") if isinstance(row, dict) else row
        try:
            entry = int(entry_value)
        except (TypeError, ValueError):
            indexed.append((row, 10**9, order))
            continue
        score = int(features.get(entry, {}).get("severity_score", 0))
        indexed.append((row, -score, entry))
    indexed.sort(key=lambda item: (item[1], item[2]))
    return [item[0] for item in indexed]


def _severity_reasons(feature: dict[str, object]) -> str:
    reasons = [str(feature.get("status", "UNTRUSTED")).lower().replace("_", "-")]
    if feature.get("provisional"):
        reasons.append("provisional")
    if feature.get("model_disagreement"):
        reasons.append("model-disagreement")
    return ",".join(reasons)


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
    listed = []
    for row, entry in zip(rows, entries):
        if not isinstance(row, dict):
            raise ValueError("Review sidecar assignment is malformed")
        review = bool(row.get("review_required") or row.get("verification_status") == "review-required")
        trusted = row.get("verification_status") == "trusted" and row.get("overall_trusted") is not False
        if trusted and not review:
            continue
        listed.append((row, entry))
    feature_report = dict(report)
    feature_report["assignments"] = rows
    ranked_rows = rank_rows(feature_report, [row for row, _ in listed])
    entry_by_id = {id(row): entry for row, entry in listed}
    features = severity_features(feature_report)
    review_rows = sum(
        int(features.get(int(row["entry"]), {}).get("status") == "REVIEW")
        for row in ranked_rows if isinstance(row, dict)
    )
    untrusted_rows = len(ranked_rows) - review_rows
    lines = [
        f"# fast_singing={_fast_singing(rows)}\treview_percent={percent}\tpreflight={preflight_status}"
        f"\treview_rows={review_rows}\tuntrusted_rows={untrusted_rows}\tseverity_rule=lrc14-pmd-v1",
        "entry\ttime\tlyric\tstatus\treason_codes\tseverity_rank\tseverity_score\tseverity_reasons\ttimestamp_status\tlrc_time\tprev_time\tnext_time\tcandidate_times\tcandidate_spread_s",
    ]
    for severity_rank, row in enumerate(ranked_rows, start=1):
        if not isinstance(row, dict):
            raise ValueError("Review sidecar assignment is malformed")
        entry = entry_by_id[id(row)]
        feature = features.get(int(row["entry"]), {})
        status = str(feature.get("status", "UNTRUSTED"))
        text = " / ".join(str(part) for part in entry.lines)
        reasons = reason_codes_for(row, status)
        candidate_values = feature.get("candidate_times", [])
        candidate_text = ",".join(_clock(value) for value in candidate_values[:6])
        spread = feature.get("candidate_spread_s")
        lines.append(
            "\t".join(_clean_field(value) for value in (
                row.get("entry"),
                _clock(row.get("timestamp")),
                text,
                status,
                ",".join(reasons),
                severity_rank,
                feature.get("severity_score", 0),
                _severity_reasons(feature),
                feature.get("timestamp_status") or "",
                _optional_clock(feature.get("lrc_time")),
                _optional_clock(feature.get("prev_time")),
                _optional_clock(feature.get("next_time")),
                candidate_text,
                f"{float(spread):.2f}" if isinstance(spread, (int, float)) else "",
            ))
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def format_top_review(path_or_rows: Path | str | list[dict[str, object]], n: int = 5) -> list[str]:
    if isinstance(path_or_rows, (Path, str)):
        raw_lines = Path(path_or_rows).read_text(encoding="utf-8-sig").splitlines()
        if len(raw_lines) < 2:
            return ["REVIEW TOP 5: none"]
        headers = raw_lines[1].split("\t")
        rows = [dict(zip(headers, line.split("\t"))) for line in raw_lines[2:] if line]
    else:
        rows = path_or_rows
    review_rows = [row for row in rows if str(row.get("status", "")) == "REVIEW"]
    if not rows:
        return ["REVIEW TOP 5: none"]
    lines = [f"REVIEW TOP 5 (of {len(rows)} listed, {len(review_rows)} review):"]
    for row in review_rows[:n]:
        lyric = _clean_field(row.get("lyric", ""))[:40]
        lines.append(
            f"  #{row.get('severity_rank', '')} entry {row.get('entry', '')} "
            f"{row.get('lrc_time', '')} score={row.get('severity_score', '')} "
            f"{row.get('severity_reasons', '')} | {lyric}"
        )
    return lines
