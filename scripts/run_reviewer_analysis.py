from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from evaluate_lrc import Entry, align_entries_by_text, is_generated_title_card, is_marker_entry, parse_lrc
import reviewer_layer


PROJECT = Path(__file__).resolve().parents[1]
REVIEW = PROJECT / "_review"
REVIEWER_ORDER = ("HUBP", "HUB", "WX", "XLSR")
ACCURACY_SOURCES = ("CUR", "HUBP", "HUB", "XLSR", "WX")
TOLERANCE_SECONDS = 0.050


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def filtered_entries(path: Path):
    _meta, entries = parse_lrc(path)
    return [entry for entry in entries if not is_generated_title_card(entry) and not is_marker_entry(entry)]


def split_members_for_partition(
    split: Mapping[str, object], partition: str
) -> dict[str, dict[str, object]]:
    members = split.get(partition)
    if partition not in {"dev", "acceptance"} or not isinstance(members, list):
        raise ValueError(f"split has no valid partition: {partition}")
    if not all(isinstance(member, dict) and "id" in member for member in members):
        raise ValueError(f"split partition has invalid member: {partition}")
    return {str(member["id"]): member for member in members}


def final_entries_from_manifest(
    manifest_path: Path, split_members: Mapping[str, dict[str, object]]
) -> dict[str, list[Entry]]:
    payload = load_json(manifest_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), dict):
        raise ValueError(f"invalid manifest: {manifest_path}")
    result: dict[str, list[Entry]] = {}
    for song_id, member in split_members.items():
        state = payload["cases"].get(song_id)
        if not isinstance(state, dict) or state.get("status") != "OK":
            raise ValueError(f"manifest missing OK case: {song_id}")
        output = Path(str(state["output"]))
        entries = filtered_entries(output)
        result[song_id] = entries
        expected = int(member.get("row_count", len(entries))) if member.get("row_count") is not None else len(entries)
        if len(entries) != expected:
            # split.json does not currently carry row_count; producer validation below is authoritative.
            pass
    return result


def build_rows(
    *,
    producer_path: Path,
    hubp_path: Path,
    split_path: Path,
    partition: str,
    final_source: str,
    final_manifest: Path | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    producer = load_json(producer_path)
    hubp = load_json(hubp_path)
    split = load_json(split_path)
    if not isinstance(producer, dict) or not isinstance(producer.get("songs"), list):
        raise ValueError("producer times has unexpected schema")
    if not isinstance(hubp, dict) or not isinstance(hubp.get("rows"), dict):
        raise ValueError("HUBP raw has unexpected schema")
    if not isinstance(split, dict):
        raise ValueError("split has unexpected schema")

    producer_songs = {str(song["id"]): song for song in producer["songs"] if isinstance(song, dict)}
    split_members = split_members_for_partition(split, partition)
    if set(producer_songs) != set(split_members):
        raise ValueError(f"producer/split {partition} song IDs differ")

    manifest_entries = (
        final_entries_from_manifest(final_manifest, split_members) if final_manifest is not None else None
    )
    hubp_rows = hubp["rows"]
    rows: list[dict[str, object]] = []
    for song_id in split_members:
        song = producer_songs[song_id]
        member = split_members[song_id]
        producer_rows = song.get("rows")
        if not isinstance(producer_rows, list):
            raise ValueError(f"producer rows missing: {song_id}")
        reference_entries = filtered_entries(Path(str(member["reference"])))
        producer_entries: list[Entry] = []
        for zero_index, row in enumerate(producer_rows):
            if not isinstance(row, dict) or not isinstance(row.get("sources"), dict):
                raise ValueError(f"invalid producer row {song_id}::{zero_index + 1}")
            lines = row.get("lines")
            if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
                raise ValueError(f"producer row text missing: {song_id}::{zero_index + 1}")
            producer_entries.append(Entry(time_cs=0, lines=list(lines)))

        producer_pairs, unmatched_reference, _unmatched_producer = align_entries_by_text(
            reference_entries, producer_entries
        )
        if unmatched_reference:
            raise ValueError(
                f"reference rows missing from producer for {song_id}: {len(unmatched_reference)}"
            )
        producer_index_by_reference = {reference_index: producer_index for reference_index, producer_index in producer_pairs}

        final_index_by_reference: dict[int, int] | None = None
        if manifest_entries is not None:
            final_pairs, unmatched_reference_final, _unmatched_final = align_entries_by_text(
                reference_entries, manifest_entries[song_id]
            )
            if unmatched_reference_final:
                raise ValueError(
                    f"reference rows missing from manifest output for {song_id}: {len(unmatched_reference_final)}"
                )
            final_index_by_reference = {
                reference_index: final_index for reference_index, final_index in final_pairs
            }

        for reference_index, reference_entry in enumerate(reference_entries):
            producer_index = producer_index_by_reference[reference_index]
            row = producer_rows[producer_index]
            entry = int(row.get("entry", producer_index + 1))
            sources = row["sources"]
            if manifest_entries is not None and final_index_by_reference is not None:
                final_entry = manifest_entries[song_id][final_index_by_reference[reference_index]]
                final_time = float(final_entry.time_cs) / 100.0
            else:
                source = sources.get(final_source)
                if not isinstance(source, dict) or not isinstance(source.get("time"), (int, float)):
                    raise ValueError(f"missing final source {final_source}: {song_id}::{entry}")
                final_time = float(source["time"])

            def source_time(name: str) -> float | None:
                item = sources.get(name)
                if isinstance(item, dict) and isinstance(item.get("time"), (int, float)):
                    value = float(item["time"])
                    return value if math.isfinite(value) else None
                return None

            hubp_item = hubp_rows.get(f"{song_id}::{entry}")
            hubp_time = (
                float(hubp_item["time"])
                if isinstance(hubp_item, dict)
                and hubp_item.get("status") == "OK"
                and isinstance(hubp_item.get("time"), (int, float))
                else None
            )
            rows.append(
                {
                    "song_id": song_id,
                    "title": str(song.get("title", song_id)),
                    "entry": entry,
                    "final_time": final_time,
                    "reference_time": float(reference_entry.time_cs) / 100.0,
                    "times": {
                        "CUR": source_time("CUR"),
                        "HUBP": hubp_time,
                        "HUB": source_time("HUB"),
                        "WX": source_time("WX"),
                        "XLSR": source_time("XLSR"),
                    },
                }
            )
    metadata = {
        "producer_path": str(producer_path),
        "hubp_path": str(hubp_path),
        "split_path": str(split_path),
        "partition": partition,
        "final_source": final_source if final_manifest is None else "manifest",
        "final_manifest": str(final_manifest) if final_manifest else None,
        "song_count": len(split_members),
        "row_count": len(rows),
    }
    return rows, metadata


def reviewer_combinations() -> list[tuple[str, ...]]:
    base = ("HUBP", "HUB", "WX")
    combos: list[tuple[str, ...]] = []
    for size in range(1, len(base) + 1):
        combos.extend(itertools.combinations(base, size))
    for size in range(0, len(base) + 1):
        for subset in itertools.combinations(base, size):
            combos.append((*subset, "XLSR"))
    return combos


def compute_offsets(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_song: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_song[str(row["song_id"])].append(row)
    global_offsets: dict[str, float] = {}
    per_song: dict[str, dict[str, float]] = {}
    for reviewer in REVIEWER_ORDER:
        global_offsets[reviewer] = reviewer_layer.estimate_offset_seconds(
            [float(row["final_time"]) for row in rows],
            [row["times"].get(reviewer) for row in rows],  # type: ignore[index,union-attr]
        )
    for song_id, song_rows in by_song.items():
        per_song[song_id] = {}
        for reviewer in REVIEWER_ORDER:
            per_song[song_id][reviewer] = reviewer_layer.estimate_offset_seconds(
                [float(row["final_time"]) for row in song_rows],
                [row["times"].get(reviewer) for row in song_rows],  # type: ignore[index,union-attr]
            )
    return {"global": global_offsets, "per_song": per_song}


def offsets_for_row(offsets: Mapping[str, object], row: Mapping[str, object], mode: str) -> dict[str, float]:
    if mode == "global":
        return dict(offsets["global"])  # type: ignore[arg-type]
    if mode == "per_song":
        per_song = offsets["per_song"]
        return dict(per_song[str(row["song_id"])])  # type: ignore[index,arg-type]
    if mode == "none":
        return {name: 0.0 for name in REVIEWER_ORDER}
    raise ValueError(f"unsupported offset mode: {mode}")


def error_gt50(row: Mapping[str, object]) -> bool:
    return abs(float(row["final_time"]) - float(row["reference_time"])) > TOLERANCE_SECONDS + 1e-12


def classify_rows(
    rows: Sequence[dict[str, object]],
    offsets: Mapping[str, object],
    combo: tuple[str, ...],
    mode: str,
) -> list[tuple[dict[str, object], reviewer_layer.ReviewerRowDecision]]:
    result = []
    for row in rows:
        row_offsets = offsets_for_row(offsets, row, mode)
        decision = reviewer_layer.classify_row(
            final_time=float(row["final_time"]),
            reviewer_times=row["times"],  # type: ignore[arg-type]
            offsets=row_offsets,
            reviewers=combo,
        )
        result.append((row, decision))
    return result


def aggregate_point(
    classified: Sequence[tuple[dict[str, object], reviewer_layer.ReviewerRowDecision]],
    *,
    rule: str,
    view: str,
) -> dict[str, object]:
    expected = len(classified[0][1].reviewers) if classified else 0
    if view == "common":
        eligible = [(row, decision) for row, decision in classified if decision.present_count == expected]
    elif view == "full":
        eligible = list(classified)
    else:
        raise ValueError(view)
    trusted: list[tuple[dict[str, object], reviewer_layer.ReviewerRowDecision]] = []
    for row, decision in eligible:
        if reviewer_layer.trusted_by_rule(decision, rule=rule, view=view):  # type: ignore[arg-type]
            trusted.append((row, decision))
    review = [(row, decision) for row, decision in eligible if (row, decision) not in trusted]
    trusted_errors = sum(error_gt50(row) for row, _ in trusted)
    all_errors = sum(error_gt50(row) for row, _ in eligible)
    review_errors = sum(error_gt50(row) for row, _ in review)
    denom = len(eligible)
    trusted_count = len(trusted)
    review_count = len(review)
    return {
        "denominator": denom,
        "trusted_count": trusted_count,
        "coverage_percent": round(100.0 * trusted_count / denom, 4) if denom else None,
        "trusted_error_count": trusted_errors,
        "trusted_error_percent": round(100.0 * trusted_errors / trusted_count, 4) if trusted_count else None,
        "review_count": review_count,
        "review_percent": round(100.0 * review_count / denom, 4) if denom else None,
        "all_error_count": all_errors,
        "review_error_count": review_errors,
        "error_recall_percent": round(100.0 * review_errors / all_errors, 4) if all_errors else None,
        "review_precision_percent": round(100.0 * review_errors / review_count, 4) if review_count else None,
    }


def aggregate_label_thresholds(
    classified: Sequence[tuple[dict[str, object], reviewer_layer.ReviewerRowDecision]],
    *,
    view: str,
) -> dict[str, object]:
    expected = len(classified[0][1].reviewers) if classified else 0
    if view == "common":
        eligible = [(row, decision) for row, decision in classified if decision.present_count == expected]
    elif view == "full":
        eligible = list(classified)
    else:
        raise ValueError(view)
    ranks = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}

    def effective_rank(decision: reviewer_layer.ReviewerRowDecision) -> int:
        if view == "common":
            return ranks[decision.label]
        if decision.present_count != expected:
            return 0
        if expected > 0 and decision.agree_count == expected:
            return 3
        if decision.agree_count >= 2:
            return 2
        if decision.agree_count == 1:
            return 1
        return 0

    output: dict[str, object] = {}
    for threshold_name, threshold_rank in ((">=L3", 3), (">=L2", 2), (">=L1", 1)):
        trusted = [
            (row, decision)
            for row, decision in eligible
            if effective_rank(decision) >= threshold_rank
        ]
        trusted_ids = {(str(row["song_id"]), int(row["entry"])) for row, _ in trusted}
        review = [
            (row, decision)
            for row, decision in eligible
            if (str(row["song_id"]), int(row["entry"])) not in trusted_ids
        ]
        trusted_errors = sum(error_gt50(row) for row, _ in trusted)
        all_errors = sum(error_gt50(row) for row, _ in eligible)
        review_errors = sum(error_gt50(row) for row, _ in review)
        output[threshold_name] = {
            "denominator": len(eligible),
            "trusted_count": len(trusted),
            "coverage_percent": round(100.0 * len(trusted) / len(eligible), 4) if eligible else None,
            "trusted_error_count": trusted_errors,
            "trusted_error_percent": round(100.0 * trusted_errors / len(trusted), 4) if trusted else None,
            "review_count": len(review),
            "error_recall_percent": round(100.0 * review_errors / all_errors, 4) if all_errors else None,
            "review_precision_percent": round(100.0 * review_errors / len(review), 4) if review else None,
        }
    l0 = [(row, decision) for row, decision in eligible if effective_rank(decision) == 0]
    l0_errors = sum(error_gt50(row) for row, _ in l0)
    output["L0"] = {
        "count": len(l0),
        "error_count": l0_errors,
        "error_percent": round(100.0 * l0_errors / len(l0), 4) if l0 else None,
    }
    return output


def source_accuracy(
    rows: Sequence[dict[str, object]], offsets: Mapping[str, object], mode: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for source in (*ACCURACY_SOURCES, "FINAL"):
        available = 0
        correct = 0
        for row in rows:
            if source == "FINAL":
                value = float(row["final_time"])
            else:
                raw = row["times"].get(source)  # type: ignore[index,union-attr]
                if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
                    continue
                value = float(raw)
                if source in REVIEWER_ORDER:
                    value -= offsets_for_row(offsets, row, mode).get(source, 0.0)
            available += 1
            if abs(value - float(row["reference_time"])) <= TOLERANCE_SECONDS + 1e-12:
                correct += 1
        result[source] = {
            "correct_le50": correct,
            "denominator": available,
            "percent": round(100.0 * correct / available, 4) if available else None,
        }
    return result


def offset_effects(rows: Sequence[dict[str, object]], offsets: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for reviewer in REVIEWER_ORDER:
        entry: dict[str, object] = {}
        for mode in ("none", "global", "per_song"):
            present = 0
            agree = 0
            for row in rows:
                raw = row["times"].get(reviewer)  # type: ignore[index,union-attr]
                if not isinstance(raw, (int, float)):
                    continue
                present += 1
                corrected = float(raw) - offsets_for_row(offsets, row, mode).get(reviewer, 0.0)
                if abs(corrected - float(row["final_time"])) <= TOLERANCE_SECONDS + 1e-12:
                    agree += 1
            entry[mode] = {
                "available": present,
                "agree": agree,
                "agree_percent_of_available": round(100.0 * agree / present, 4) if present else None,
            }
        result[reviewer] = entry
    return result


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def bootstrap_ci(
    classified: Sequence[tuple[dict[str, object], reviewer_layer.ReviewerRowDecision]],
    *,
    rule: str,
    view: str,
    samples: int,
    seed: int,
) -> dict[str, object]:
    by_song: dict[str, list[tuple[dict[str, object], reviewer_layer.ReviewerRowDecision]]] = defaultdict(list)
    for item in classified:
        by_song[str(item[0]["song_id"])].append(item)
    song_ids = sorted(by_song)
    rng = random.Random(seed)
    coverage_values: list[float] = []
    error_values: list[float] = []
    for _ in range(samples):
        sample_rows: list[tuple[dict[str, object], reviewer_layer.ReviewerRowDecision]] = []
        for _slot in song_ids:
            picked = rng.choice(song_ids)
            sample_rows.extend(by_song[picked])
        metric = aggregate_point(sample_rows, rule=rule, view=view)
        coverage = metric["coverage_percent"]
        error = metric["trusted_error_percent"]
        if isinstance(coverage, (int, float)):
            coverage_values.append(float(coverage))
        if isinstance(error, (int, float)):
            error_values.append(float(error))
    return {
        "samples": samples,
        "coverage_percent_95ci": [percentile(coverage_values, 0.025), percentile(coverage_values, 0.975)],
        "trusted_error_percent_95ci": [percentile(error_values, 0.025), percentile(error_values, 0.975)],
    }


def per_song_metrics(
    classified: Sequence[tuple[dict[str, object], reviewer_layer.ReviewerRowDecision]],
    *,
    rule: str,
    view: str,
) -> dict[str, object]:
    by_song: dict[str, list[tuple[dict[str, object], reviewer_layer.ReviewerRowDecision]]] = defaultdict(list)
    for item in classified:
        by_song[str(item[0]["song_id"])].append(item)
    return {
        song_id: aggregate_point(items, rule=rule, view=view)
        for song_id, items in sorted(by_song.items())
    }


def combo_key(combo: Iterable[str]) -> str:
    return "+".join(combo)


def pareto_frontier(points: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    valid = [
        point
        for point in points
        if isinstance(point.get("coverage_percent"), (int, float))
        and isinstance(point.get("trusted_error_percent"), (int, float))
    ]
    frontier = []
    for point in valid:
        coverage = float(point["coverage_percent"])
        error = float(point["trusted_error_percent"])
        dominated = False
        for other in valid:
            if other is point:
                continue
            other_coverage = float(other["coverage_percent"])
            other_error = float(other["trusted_error_percent"])
            if (
                other_coverage >= coverage
                and other_error <= error
                and (other_coverage > coverage or other_error < error)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: (float(point["trusted_error_percent"]), -float(point["coverage_percent"])))


def render_markdown(payload: Mapping[str, object]) -> str:
    lines = [
        f"# Reviewer analysis — {payload['name']}",
        "",
        f"Rows: {payload['metadata']['row_count']} / songs: {payload['metadata']['song_count']}",  # type: ignore[index]
        "",
        "## Offsets and agreement",
        "",
        "| Reviewer | Global offset ms | Available | Raw agree % | Global agree % | Per-song agree % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    offsets = payload["offsets"]  # type: ignore[assignment]
    effects = payload["offset_effects"]  # type: ignore[assignment]
    for reviewer in REVIEWER_ORDER:
        global_ms = 1000.0 * float(offsets["global"][reviewer])  # type: ignore[index]
        e = effects[reviewer]  # type: ignore[index]
        lines.append(
            f"| {reviewer} | {global_ms:.1f} | {e['none']['available']} | {e['none']['agree_percent_of_available']} | {e['global']['agree_percent_of_available']} | {e['per_song']['agree_percent_of_available']} |"
        )
    lines.extend(
        [
            "",
            "## Full-view Pareto frontier (per-song offset)",
            "",
            "| Reviewers | Rule | Coverage % | Trusted error % | Review % | Coverage 95% CI | Error 95% CI |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    for point in payload["pareto_frontier"]:  # type: ignore[union-attr]
        ci = point["bootstrap_95ci"]
        lines.append(
            f"| {point['combo']} | {point['rule']} | {point['coverage_percent']} | {point['trusted_error_percent']} | {point['review_percent']} | {ci['coverage_percent_95ci']} | {ci['trusted_error_percent_95ci']} |"
        )
    lines.extend(["", "## Risk thresholds", "", "| Max trusted error | Best point | Coverage % | Trusted error % | Review % |", "|---:|---|---:|---:|---:|"])
    for threshold, point in payload["threshold_best"].items():  # type: ignore[union-attr]
        if point is None:
            lines.append(f"| {threshold}% | none | — | — | — |")
        else:
            lines.append(
                f"| {threshold}% | {point['combo']} / {point['rule']} | {point['coverage_percent']} | {point['trusted_error_percent']} | {point['review_percent']} |"
            )
    lines.extend(["", "## Proposal-source accuracy", "", "| Offset mode | Source | <=50 | Denominator | Accuracy % |", "|---|---|---:|---:|---:|"])
    for mode in ("per_song", "global"):
        for source, metric in payload["proposal_accuracy"][mode].items():  # type: ignore[index,union-attr]
            lines.append(
                f"| {mode} | {source} | {metric['correct_le50']} | {metric['denominator']} | {metric['percent']} |"
            )
    comparison = payload.get("claude_approx_comparison")
    if isinstance(comparison, list) and comparison:
        lines.extend(["", "## Claude approximate cross-check", "", "| Point | Mode | Coverage actual/approx | Error actual/approx |", "|---|---|---|---|"])
        for item in comparison:
            lines.append(
                f"| {item['point']} | {item['offset_mode']} | {item['coverage_actual']} / {item['coverage_approx']} | {item['error_actual']} / {item['error_approx']} |"
            )
    return "\n".join(lines) + "\n"


def output_payload_for_partition(payload: Mapping[str, object], partition: str) -> dict[str, object]:
    result = copy.deepcopy(dict(payload))
    if partition != "acceptance":
        return result

    result.pop("sidecar_rows", None)
    offsets = result.get("offsets")
    if isinstance(offsets, dict):
        offsets.pop("per_song", None)

    analysis = result.get("analysis")
    if isinstance(analysis, dict):
        for mode_result in analysis.values():
            if not isinstance(mode_result, dict):
                continue
            for combo_result in mode_result.values():
                if not isinstance(combo_result, dict):
                    continue
                rules = combo_result.get("rules")
                if not isinstance(rules, dict):
                    continue
                for rule_result in rules.values():
                    if not isinstance(rule_result, dict):
                        continue
                    for metric in rule_result.values():
                        if isinstance(metric, dict):
                            metric.pop("per_song", None)

    result["acceptance_privacy"] = "aggregate_only"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--producer-times", default=str(REVIEW / "PRODUCER_TIMES_20261002.json"))
    parser.add_argument("--hubp", default=str(REVIEW / "hfa" / "HUBP_RAW_20261002.json"))
    parser.add_argument("--split", default=str(REVIEW / "split.json"))
    parser.add_argument("--partition", choices=("dev", "acceptance"), default="dev")
    parser.add_argument("--final-source", default="CUR")
    parser.add_argument("--final-manifest")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    rows, metadata = build_rows(
        producer_path=Path(args.producer_times),
        hubp_path=Path(args.hubp),
        split_path=Path(args.split),
        partition=args.partition,
        final_source=args.final_source,
        final_manifest=Path(args.final_manifest) if args.final_manifest else None,
    )
    offsets = compute_offsets(rows)
    analysis: dict[str, object] = {}
    bootstrap_points: list[dict[str, object]] = []
    sidecar_rows: dict[str, object] = {}
    for mode in ("per_song", "global"):
        mode_result: dict[str, object] = {}
        for combo in reviewer_combinations():
            key = combo_key(combo)
            classified = classify_rows(rows, offsets, combo, mode)
            rules = ["all"] + (["at-least-2"] if len(combo) >= 3 else [])
            combo_result: dict[str, object] = {"reviewers": list(combo), "rules": {}, "labels": {}}
            for view in ("common", "full"):
                combo_result["labels"][view] = aggregate_label_thresholds(classified, view=view)  # type: ignore[index]
            for rule in rules:
                rule_result: dict[str, object] = {}
                for view in ("common", "full"):
                    metric = aggregate_point(classified, rule=rule, view=view)
                    metric["per_song"] = per_song_metrics(classified, rule=rule, view=view)
                    if view == "full" and mode == "per_song":
                        metric["bootstrap_95ci"] = bootstrap_ci(
                            classified,
                            rule=rule,
                            view=view,
                            samples=args.bootstrap_samples,
                            seed=20261002 + len(bootstrap_points),
                        )
                        bootstrap_points.append(
                            {
                                "combo": key,
                                "rule": rule,
                                "coverage_percent": metric["coverage_percent"],
                                "trusted_error_percent": metric["trusted_error_percent"],
                                "review_percent": metric["review_percent"],
                                "bootstrap_95ci": metric["bootstrap_95ci"],
                            }
                        )
                    rule_result[view] = metric
                combo_result["rules"][rule] = rule_result  # type: ignore[index]
            mode_result[key] = combo_result

            if mode == "per_song":
                for row, decision in classified:
                    row_key = f"{row['song_id']}::{row['entry']}"
                    sidecar = sidecar_rows.setdefault(
                        row_key,
                        {
                            "song_id": row["song_id"],
                            "entry": row["entry"],
                            "final_time": row["final_time"],
                            "reviewers": row["times"],
                            "labels": {},
                        },
                    )
                    sidecar["labels"][key] = {  # type: ignore[index]
                        "label": decision.label,
                        "present_count": decision.present_count,
                        "agree_count": decision.agree_count,
                        "missing": list(decision.missing_reviewers),
                        "agreeing": list(decision.agreeing_reviewers),
                    }
        analysis[mode] = mode_result

    frontier = pareto_frontier(bootstrap_points)
    threshold_best: dict[str, object] = {}
    for threshold in (2.0, 5.0, 10.0):
        candidates = [
            point
            for point in frontier
            if isinstance(point.get("trusted_error_percent"), (int, float))
            and float(point["trusted_error_percent"]) <= threshold
        ]
        threshold_best[str(int(threshold))] = (
            max(candidates, key=lambda point: (float(point["coverage_percent"]), -float(point["trusted_error_percent"])))
            if candidates
            else None
        )

    approx_specs = [
        (("HUBP", "WX"), "all", 25.6, 4.6),
        (("HUBP", "WX", "XLSR"), "all", 15.6, 1.9),
        (("HUBP", "WX", "XLSR"), "at-least-2", 47.9, 7.4),
        (("HUBP",), "all", 74.8, 9.5),
    ]
    comparisons: list[dict[str, object]] = []
    if args.final_manifest is None and args.final_source == "CUR":
        for combo, rule, approx_cov, approx_err in approx_specs:
            key = combo_key(combo)
            for mode in ("per_song", "global"):
                metric = analysis[mode][key]["rules"][rule]["full"]  # type: ignore[index]
                comparisons.append(
                    {
                        "point": f"{key}/{rule}",
                        "offset_mode": mode,
                        "coverage_actual": metric["coverage_percent"],
                        "coverage_approx": approx_cov,
                        "coverage_delta_pp": round(float(metric["coverage_percent"]) - approx_cov, 4),
                        "error_actual": metric["trusted_error_percent"],
                        "error_approx": approx_err,
                        "error_delta_pp": (
                            round(float(metric["trusted_error_percent"]) - approx_err, 4)
                            if isinstance(metric["trusted_error_percent"], (int, float))
                            else None
                        ),
                    }
                )

    payload = {
        "schema": 1,
        "name": args.name,
        "metadata": metadata,
        "registered_reviewers": list(REVIEWER_ORDER),
        "registered_combinations": [list(combo) for combo in reviewer_combinations()],
        "tolerance_ms": 50,
        "offset_sample_limit_ms": 500,
        "offsets": offsets,
        "offset_effects": offset_effects(rows, offsets),
        "proposal_accuracy": {
            "per_song": source_accuracy(rows, offsets, "per_song"),
            "global": source_accuracy(rows, offsets, "global"),
        },
        "analysis": analysis,
        "pareto_frontier": frontier,
        "threshold_best": threshold_best,
        "claude_approx_comparison": comparisons,
        "sidecar_rows": sidecar_rows,
    }
    output_payload = output_payload_for_partition(payload, args.partition)
    output_json = Path(args.output_json) if args.output_json else REVIEW / f"{args.name}.json"
    output_md = Path(args.output_md) if args.output_md else REVIEW / f"{args.name}.md"
    output_json.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(output_payload), encoding="utf-8")
    print(json.dumps({"json": str(output_json), "md": str(output_md), "rows": len(rows), "frontier": len(frontier)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
