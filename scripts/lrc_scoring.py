"""Official LRC 1.3 release posthoc scoring (BK is scoring-only).

This module scores immutable completed P3R/P4 manifests and fails closed on
lexical, assignment, or trust alignment errors. REVIEW means mandatory listening;
UNVERIFIED is separately reported, never counted as REVIEW or TRUSTED.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import json

import auto_lrc

EPS = 1e-9
CLASSES = ("trusted", "review-required", "unverified")


class ScoringError(ValueError):
    pass


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parsed_rows(path: Path, audio_path: Path):
    raw = auto_lrc.load_lyrics(path).entries
    untitled = auto_lrc.remove_generated_title_cards(raw, audio_path)
    clean, _ = auto_lrc.remove_instrumental_markers(untitled)
    return [(auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(x)), x.source_time_cs / 100.0) for x in clean]


def reorder_assignments_by_timestamp(clean, selected, path: Path):
    """Correct explicit timestamp-sorted output without losing row trust identity.

    The report retains lyric input order; output LRC may be sorted by timestamp.
    Require an identical multiset of canonical centisecond timestamps and
    preserve original report order within equal-time buckets.
    """
    expected = [e.source_time_cs for e in clean]
    original = [int(round(float(a["timestamp"]) * 100)) for a in selected]
    if original == expected:
        return selected
    if Counter(original) != Counter(expected):
        raise ScoringError(f"REPORT_TIME_MULTISET_MISMATCH {path}")
    bins = defaultdict(list)
    for a, t in zip(selected, original, strict=True):
        bins[t].append(a)
    return [bins[t].pop(0) for t in expected]


def rows_and_trust(path: Path, audio_path: Path, report: dict):
    raw = auto_lrc.load_lyrics(path).entries
    untitled = auto_lrc.remove_generated_title_cards(raw, audio_path)
    clean, skipped = auto_lrc.remove_instrumental_markers(untitled)
    assignment = report.get("assignments")
    central = report.get("central_timing_state")
    if not isinstance(assignment, list) or not isinstance(central, list) or len(assignment) != len(central):
        raise ScoringError(f"REPORT_ASSIGNMENT_CENTRAL_SKEW {path}")
    if len(assignment) == len(clean):
        selected = assignment
    elif len(assignment) == len(untitled) and skipped:
        selected = [a for a, e in zip(assignment, untitled, strict=True) if not auto_lrc.is_instrumental_marker(e)]
    else:
        raise ScoringError(f"REPORT_ASSIGNMENT_COUNT {path}: {len(assignment)} != {len(clean)}")
    if len(selected) != len(clean):
        raise ScoringError(f"REPORT_FILTER_COUNT {path}")
    selected = reorder_assignments_by_timestamp(clean, selected, path)
    output = []
    for line, (e, a) in enumerate(zip(clean, selected, strict=True), 1):
        if not isinstance(a, dict):
            raise ScoringError(f"REPORT_ROW_INVALID {path}:{line}")
        if abs(float(a.get("timestamp", -1e5)) - e.source_time_cs / 100.0) > 0.021:
            raise ScoringError(f"REPORT_CLOCK_SKEW {path}:{line}")
        status = a.get("verification_status")
        if status not in CLASSES or bool(a.get("review_required")) != (status == "review-required"):
            raise ScoringError(f"REVIEW_STATUS_CONTRADICTION {path}:{line}")
        if type(a.get("timing_trusted")) is not bool:
            raise ScoringError(f"MISSING_TIMING_TRUST {path}:{line}")
        if bool(a.get("overall_trusted")) != (status == "trusted"):
            raise ScoringError(f"OVERALL_TRUST_CONTRADICTION {path}:{line}")
        if a.get("timing_audit_trusted") is not a.get("timing_trusted"):
            raise ScoringError(f"TIMING_TRUST_CONTRADICTION {path}:{line}")
        output.append({
            "text": auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(e)),
            "time": e.source_time_cs / 100.0,
            "status": status, "trusted": bool(a["timing_trusted"]),
            "reasons": sorted(a.get("final_trust_reason_codes") or []),
            "flags": sorted(a.get("flags") or []),
            "source": a.get("candidate_source"),
        })
    return output


def align_gold(rows, gold, name):
    if len(rows) != len(gold):
        raise ScoringError(f"BK_COUNT_MISMATCH {name}: {len(rows)} vs {len(gold)}")
    generated = [r["text"] for r in rows]
    expected = [g[0] for g in gold]
    if generated == expected:
        return gold, False
    if Counter(generated) != Counter(expected):
        raise ScoringError(f"BK_TEXT_MISMATCH {name}")
    pool = defaultdict(list)
    for entry in gold:
        pool[entry[0]].append(entry)
    counts = Counter()
    remapped = []
    for text in generated:
        remapped.append(pool[text][counts[text]])
        counts[text] += 1
    return remapped, True


def count_metrics(rows):
    if not rows:
        raise ScoringError("EMPTY_CORPUS")
    total = len(rows)
    status = Counter(r["status"] for r in rows)
    if set(status) - set(CLASSES):
        raise ScoringError("UNEXPECTED_STATUS")
    def errors(r):
        return abs(r["time"] - r["bk"])
    def subset(pred):
        return [r for r in rows if pred(r)]
    def buckets(data):
        return {
            "rows": len(data),
            "PCO200": sum(errors(r) <= .2 + EPS for r in data),
            "gt05": sum(errors(r) > .5 + EPS for r in data),
            "gt1": sum(errors(r) > 1. + EPS for r in data),
        }
    review = subset(lambda r: r["status"] == "review-required")
    trusted = subset(lambda r: r["status"] == "trusted")
    unverified = subset(lambda r: r["status"] == "unverified")
    no_review = subset(lambda r: r["status"] != "review-required")
    reasons = Counter(reason for r in review for reason in r["reasons"])
    metrics = {
        "all": buckets(rows),
        "REVIEW": buckets(review), "TRUSTED": buckets(trusted),
        "UNVERIFIED": buckets(unverified), "NO_REVIEW": buckets(no_review),
        "review_request_ratio": len(review) / total,
        "trusted_ratio": len(trusted) / total,
        "unverified_ratio": len(unverified) / total,
        "nontrusted_ratio": (len(review) + len(unverified)) / total,
        "review_reason_counts": dict(reasons.most_common()),
        "no_review_gt1_rows": [small_row(r) for r in no_review if errors(r) > 1. + EPS],
        "trusted_gt1_rows": [small_row(r) for r in trusted if errors(r) > 1. + EPS],
        "review_gt05_precision": sum(errors(r) > .5 + EPS for r in review) / len(review) if review else None,
    }
    return metrics


def small_row(r):
    return {k: r.get(k) for k in ("song", "case", "line", "text", "time", "bk", "status", "trusted", "source", "reasons")}


def load_manifest(
    folder: Path,
    expected_phase: str,
    expected_commit: str,
    *,
    expected_source_sha256: str | None = None,
):
    m = read_json(folder / "MANIFEST.json")
    if m.get("phase") != expected_phase or m.get("count") != 39 or m.get("total") != 39:
        raise ScoringError(f"MANIFEST_NOT_COMPLETE {folder}")
    if m.get("validation_of_commit") != expected_commit:
        raise ScoringError(f"MANIFEST_WRONG_COMMIT {folder}")
    if m.get("auto_lrc_sha256") and expected_source_sha256 is None:
        raise ScoringError(f"MANIFEST_SOURCE_PIN_REQUIRED {folder}")
    if (
        expected_source_sha256 is not None
        and m.get("auto_lrc_sha256") != expected_source_sha256
    ):
        raise ScoringError(f"MANIFEST_WRONG_SOURCE {folder}")
    entries = m.get("cases")
    if not isinstance(entries, list) or len(entries) != 39:
        raise ScoringError(f"MANIFEST_WRONG_COUNT {folder}")
    by_index = {int(c["index"]): c for c in entries}
    if set(by_index) != set(range(1, 40)) or not all(c.get("status") == "PRODUCED" and c.get("exit") == 0 for c in entries):
        raise ScoringError(f"MANIFEST_FAILURE {folder}")
    return by_index


def score_phase(
    folder: Path,
    corpus: list,
    *,
    name: str,
    expected_phase: str,
    expected_commit: str,
    expected_source_sha256: str | None = None,
):
    cases = load_manifest(
        folder,
        expected_phase,
        expected_commit,
        expected_source_sha256=expected_source_sha256,
    )
    rows = []
    remaps = []
    for index, item in enumerate(corpus, 1):
        info = cases[index]
        audio = Path(item["audio_path"])
        candidate = Path(info["candidate"])
        if not candidate.is_file() or not candidate.is_relative_to(folder):
            raise ScoringError(f"CANDIDATE_MISSING_OR_UNEXPECTED {candidate}")
        reportdir = candidate.parent / "reports"
        report_files = list(reportdir.glob("*.align-report.json"))
        if len(report_files) != 1:
            raise ScoringError(f"REPORT_MISSING_OR_MULTIPLE {candidate}")
        report = read_json(report_files[0])
        if len(report.get("central_timing_state", [])) != len(report.get("assignments", [])):
            raise ScoringError(f"TRUST_ROW_MISMATCH {candidate}")
        for a, c in zip(report["assignments"], report["central_timing_state"], strict=True):
            if a.get("timing_trusted") is not c.get("timing_trusted"):
                raise ScoringError(f"TRUST_CENTRAL_MISMATCH {candidate}:{a.get('entry')}")
        parsed = rows_and_trust(candidate, audio, report)
        gold = parsed_rows(Path(item["human_BK"]), audio)
        aligned, remapped = align_gold(parsed, gold, info["song"])
        if remapped:
            remaps.append({"case": index, "song": info["song"], "kind": "BK_ORDER_REMAP"})
        for lineno, (row, (gt_text, gt_time)) in enumerate(zip(parsed, aligned, strict=True), 1):
            if row["text"] != gt_text:
                raise ScoringError(f"TEXT_ALIGNMENT {candidate}:{lineno}")
            rows.append({**row, "bk": gt_time, "song": info["song"], "case": index, "line": lineno})
    # 752e6c7e: remove_generated_title_cards also recognizes "Title - Artist"
    # cards (c07, c10). Candidate and BK both go through that function, so
    # every run, old or new, now scores the same 1487 sung rows.
    if len(rows) != 1487:
        raise ScoringError(f"DENOMINATOR_CHANGED {name}: {len(rows)} != 1487")
    return {
        "label": name,
        "commit": expected_commit,
        "auto_lrc_sha256": expected_source_sha256,
        "metrics": count_metrics(rows),
        "bk_order_remaps": remaps,
        "rows": rows,
    }


def compare_phases(old, new):
    previous = {(r["case"], r["line"]): r for r in old["rows"]}
    updated = {(r["case"], r["line"]): r for r in new["rows"]}
    if previous.keys() != updated.keys():
        raise ScoringError("PHASE_LINESET_DIFFERENCE")
    changes, status_changes, reasons_changes, new_gt1, fixed_gt1 = [], [], [], [], []
    for key in sorted(previous):
        a, b = previous[key], updated[key]
        if (a["text"], a["bk"]) != (b["text"], b["bk"]):
            raise ScoringError(f"PHASE_LEXICAL_BK_DIFFERENCE {key}")
        base = {"case": a["case"], "line": a["line"], "song": a["song"], "text": a["text"]}
        if abs(b["time"] - a["time"]) > .005:
            changes.append({**base, "old_time": a["time"], "new_time": b["time"], "delta_s": round(b["time"] - a["time"], 3),
                            "bk_time": a["bk"], "old_error_s": round(abs(a["time"] - a["bk"]), 3),
                            "new_error_s": round(abs(b["time"] - b["bk"]), 3), "old_source": a["source"], "new_source": b["source"],
                            "old_status": a["status"], "new_status": b["status"]})
        if a["status"] != b["status"]:
            status_changes.append({**base, "from": a["status"], "to": b["status"],
                                   "previous_reasons": a["reasons"], "current_reasons": b["reasons"]})
        if a["reasons"] != b["reasons"]:
            reasons_changes.append({**base, "old": a["reasons"], "new": b["reasons"]})
        if abs(a["time"] - a["bk"]) <= 1. + EPS and abs(b["time"] - b["bk"]) > 1. + EPS:
            new_gt1.append(base)
        if abs(a["time"] - a["bk"]) > 1. + EPS and abs(b["time"] - b["bk"]) <= 1. + EPS:
            fixed_gt1.append(base)
    return {"timestamp_changes": changes, "trust_status_changes": status_changes,
            "trust_reason_changes": reasons_changes,
            "new_gt1": new_gt1, "fixed_gt1": fixed_gt1}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    base = root / "_review"
    parser.add_argument("--corpus", type=Path, default=base / "LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json")
    parser.add_argument("--p3r", type=Path, default=base / "LRC13_FULL_AUDIT_P3R_CTC_POLICY_39_20261012")
    parser.add_argument("--p4", type=Path, default=base / "LRC13_FULL_AUDIT_P4_39_20261012")
    parser.add_argument("--p4-phase", default="P4_39_PRODUCED")
    parser.add_argument("--p4-commit", default="f4c9c2cdb333e5ab8601ea6e4774db8e71df8fbb")
    parser.add_argument("--p4-source-sha256")
    parser.add_argument("--native-n2", type=Path, default=base / "LRC13_LEAN_N2_SCORE_20261011.json")
    parser.add_argument("--output", type=Path, default=base / "LRC13_P5_OFFICIAL_SCORE_20261013.json")
    args = parser.parse_args(argv)
    corpus = read_json(args.corpus)["corpus"]
    if len(corpus) != 39:
        raise ScoringError("EXPECTED_39_SONGS")
    a = score_phase(args.p3r, corpus, name="P3R", expected_phase="P3_39_PRODUCED", expected_commit="5e36f09c")
    b = score_phase(
        args.p4,
        corpus,
        name="P4",
        expected_phase=args.p4_phase,
        expected_commit=args.p4_commit,
        expected_source_sha256=args.p4_source_sha256,
    )
    n2 = read_json(args.native_n2)
    historic = n2["counts"]["N2"]
    transition = compare_phases(a, b)
    old, now = a["metrics"], b["metrics"]
    out = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "P5_SCORER_COMPLETE_P4_ACCEPTANCE_BLOCKED" if transition["timestamp_changes"] else "P5_SCORER_COMPLETE",
        "input_BK_timing_used_for_inference": False,
        "phase_counts": {"P3R": 39, "P4": 39, "rows_each": len(a["rows"])},
        "P3R": {
            "commit": a["commit"],
            "auto_lrc_sha256": a["auto_lrc_sha256"],
            "metrics": old,
            "bk_order_remaps": a["bk_order_remaps"],
        },
        "P4": {
            "commit": b["commit"],
            "auto_lrc_sha256": b["auto_lrc_sha256"],
            "metrics": now,
            "bk_order_remaps": b["bk_order_remaps"],
        },
        "N2_native_historical": historic,
        "N2_same_semantics_recomputed": None,
        "N2_comparability": "N2 1487-row legacy/native trust semantics; P3R/P4 1489 rows with two title-card rows and final_line_trust. No same-denominator trust comparison is claimed.",
        "delta_P4_minus_N2_native": {key: now["all"][key] - historic[key] for key in ("PCO200", "gt1", "gt05")},
        "delta_P4_minus_P3R": {key: now["all"][key] - old["all"][key] for key in ("PCO200", "gt1", "gt05")},
        "main_historical_Decision_A": {"total_gt1": 117, "total_gt05": 146, "review_ratio": .416, "trusted_silent_gt1": 5, "trusted_silent_gt05": 19},
        "parser_exceptions": [
            "06.ビビッド title-only first line retained",
            "Ever be my love title-only first line retained",
            "朧月夜 #8/#9 BK order remap (same-text occurrence mapping)",
        ],
        "changes": transition,
        "acceptance": {
            "P3R_decision_A": "PREVIOUSLY_ACCEPTED",
            "P4_identity": "BLOCKED" if transition["timestamp_changes"] else "ZERO_DELTA",
            "P5_freeze": "NOT_AUTHORIZED_PENDING_P4",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = args.output.with_suffix(".md")
    lines = [
        "# LRC 1.3 P5 official scorer — preliminary release review", "",
        f"Status: **{out['status']}**. BK timing was used exclusively for posthoc scoring.", "",
        "| Metric | Historic N2 | P3R | P4 |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (("PCO@0.2s", "PCO200"), ("Errors >0.5s", "gt05"), ("Errors >1.0s", "gt1")):
        lines.append(f"| {label} | {historic[key]} | {old['all'][key]} | {now['all'][key]} |")
    lines.extend(["", "Historical N2 has different lyric cardinality and trust denominator; its review percentage is not directly comparable.",
                  "", f"P3R REVIEW {old['REVIEW']['rows']}/{old['all']['rows']} ({old['review_request_ratio']:.2%}); "
                  f"P4 REVIEW {now['REVIEW']['rows']}/{now['all']['rows']} ({now['review_request_ratio']:.2%}).",
                  f"P4 TRUSTED {now['TRUSTED']['rows']}; UNVERIFIED {now['UNVERIFIED']['rows']}; "
                  f"no-REVIEW >1s {now['NO_REVIEW']['gt1']}; trusted-only >1s {now['TRUSTED']['gt1']}.",
                  f"P4 timestamp changes vs P3R: {len(transition['timestamp_changes'])} rows; "
                  f"new >1s errors {len(transition['new_gt1'])}, fixed >1s errors {len(transition['fixed_gt1'])}.",
                  "", "## Parser / scorer deviations",
                  *[f"- {x}" for x in out["parser_exceptions"]],
                  "", "## Timestamp changes (all)",
                  *[f"- {r['song']} #{r['line']}: {r['old_time']:.2f} -> {r['new_time']:.2f} ({r['delta_s']:+.2f}s), "
                    f"BK error {r['old_error_s']:.2f} -> {r['new_error_s']:.2f}, {r['old_source']} -> {r['new_source']}"
                    for r in sorted(transition["timestamp_changes"], key=lambda x: -abs(x["delta_s"]))],
                  "", f"Trust/status transitions: {len(transition['trust_status_changes'])}; "
                  f"trust-reason transitions: {len(transition['trust_reason_changes'])}.",
                  "", "P4 timing identity is not accepted. No freeze tag or automatic release approval is produced by this scorer.",
                  ""])
    md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": out["status"], "phase_counts": out["phase_counts"],
        "P3R": old["all"], "P4": now["all"],
        "review": now["REVIEW"]["rows"], "trusted": now["TRUSTED"]["rows"],
        "unverified": now["UNVERIFIED"]["rows"],
        "changed_rows": len(transition["timestamp_changes"]), "output": str(args.output)
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
