from __future__ import annotations

import hashlib
import http.server
import math
import json
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import os
import pathlib
import re
import secrets
import threading
import uuid
import urllib.parse
import webbrowser
from datetime import datetime, timezone

import lrc_review_tsv
from lrc_manual_edits import review_artifact_paths

_EXCLUDED_DIR_NAMES = ("lrc tools checked references", "lrc bk")
_TAG_RE = re.compile(rb"^\[(\d{2,}):(\d{2})\.(\d{2})\]")
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_MAX_BODY = 256 * 1024


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_compare(left, right):
    try:
        return secrets.compare_digest(left, right)
    except TypeError:
        return False


def _round_cs(value):
    try:
        return int((Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _tag_cs(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _round_cs(value)
    match = re.fullmatch(r"\[?(\d+):(\d{2})\.(\d{2})\]?", str(value or ""))
    if not match:
        return None
    return int(match.group(1)) * 6000 + int(match.group(2)) * 100 + int(match.group(3))


def _format_tag(cs):
    minutes, remainder = divmod(int(cs), 6000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"[{minutes:02d}:{seconds:02d}.{centiseconds:02d}]".encode("ascii")


def _split_lrc(data):
    bom = data.startswith(b"\xef\xbb\xbf")
    body = data[3:] if bom else data
    trailing = body.endswith(b"\n")
    lines = body.split(b"\n")
    if trailing:
        lines = lines[:-1]
    return bom, trailing, lines


def _join_lrc(bom, trailing, lines):
    body = b"\n".join(lines) + (b"\n" if trailing else b"")
    return (b"\xef\xbb\xbf" if bom else b"") + body


def _line_tag(line):
    match = _TAG_RE.match(line)
    if not match:
        return None
    return int(match.group(1)) * 6000 + int(match.group(2)) * 100 + int(match.group(3))


def _tag_tail(line):
    match = _TAG_RE.match(line)
    return line[match.end():] if match else line


def _timing_mapping(data, display_lines, assignments):
    _bom, _trailing, lines = _split_lrc(data)
    try:
        display_lines = int(display_lines)
    except (TypeError, ValueError):
        return None, lines, "display-lines-mismatch"
    if display_lines < 0 or display_lines > len(lines):
        return None, lines, "display-lines-mismatch"
    start = len(lines) - display_lines
    region = lines[start:]
    wanted = []
    for expected_entry, assignment in enumerate(assignments, 1):
        if not isinstance(assignment, dict):
            return None, lines, "malformed-assignment"
        try:
            if int(assignment.get("entry", -1)) != expected_entry:
                return None, lines, "entry-index-mismatch"
        except (TypeError, ValueError):
            return None, lines, "entry-index-mismatch"
        wanted.append(_tag_cs(assignment.get("lrc_tag")))
    if any(tag is None for tag in wanted):
        return None, lines, "malformed-assignment"
    tags = [_line_tag(line) for line in region]
    if any(tag is None for tag in tags):
        return None, lines, "untimed-region"
    paths = {(0, 0): [()]}
    for position in range(len(region) + 1):
        for index in range(len(wanted)):
            options = paths.get((position, index), [])
            if not options:
                continue
            for end in range(position + 1, len(region) + 1):
                if any(tags[offset] != wanted[index] for offset in range(position, end)):
                    break
                key = (end, index + 1)
                additions = [path + (tuple(range(start + position, start + end)),) for path in options]
                paths[key] = (paths.get(key, []) + additions)[:2]
    matches = paths.get((len(region), len(wanted)), [])
    if len(matches) != 1:
        return None, lines, "ambiguous-timing-region" if matches else "timing-entry-count-mismatch"
    return list(matches[0]), lines, None


def _path_is_excluded(path):
    if not isinstance(path, (str, pathlib.Path)):
        return False
    parts = {part.casefold() for part in pathlib.Path(path).parts}
    return any(name in parts for name in _EXCLUDED_DIR_NAMES)


def _load_report(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError):
        return None
    return data if isinstance(data, dict) else None


def _report_files(targets, report_dir):
    found = []
    excluded = []
    targets = list(targets or [])
    if not targets:
        targets = [report_dir]
    for raw in targets:
        path = pathlib.Path(raw).expanduser().resolve()
        if path.is_file() and path.name.endswith(".align-report.json"):
            candidates = [path]
        elif path.is_file():
            candidates = []
            target = str(path).casefold()
            for report in report_dir.glob("*.align-report.json"):
                data = _load_report(report)
                if data is None:
                    if report not in excluded:
                        excluded.append(report)
                    continue
                if str(data.get("audio_path", "")).casefold() == target:
                    candidates.append(report)
        elif path.is_dir():
            local_reports = sorted(path.rglob("*.align-report.json"))
            if local_reports:
                candidates = local_reports
            else:
                prefix = str(path).casefold().rstrip("\\/") + "\\"
                candidates = []
                for report in report_dir.glob("*.align-report.json"):
                    data = _load_report(report)
                    if data is None:
                        if report not in excluded:
                            excluded.append(report)
                        continue
                    if str(data.get("audio_path", "")).casefold().startswith(prefix):
                        candidates.append(report)
        else:
            candidates = []
        for report in candidates:
            data = _load_report(report)
            if data is None:
                if report not in excluded:
                    excluded.append(report)
                continue
            if not isinstance(data.get("audio_path"), str):
                if report not in excluded:
                    excluded.append(report)
                continue
            paths = (report, data.get("audio_path", ""), data.get("output_path", ""))
            if not report.is_file():
                continue
            if any(_path_is_excluded(item) for item in paths):
                if report not in excluded:
                    excluded.append(report)
            elif report not in found:
                found.append(report)
    return sorted(found, key=lambda item: item.name.casefold()), sorted(excluded, key=lambda item: item.name.casefold())


def _report_rows(report_path, report):
    tsv_path = report_path.with_name(report_path.name.replace(".align-report.json", ".review.tsv"))
    if tsv_path.is_file():
        try:
            lines = tsv_path.read_text(encoding="utf-8-sig").splitlines()
            if len(lines) >= 2 and "severity_rank" in lines[1].split("\t"):
                header = lines[1].split("\t")
                rows = []
                for line in lines[2:]:
                    values = line.split("\t")
                    if len(values) != len(header):
                        continue
                    row = dict(zip(header, values))
                    try:
                        row["entry"] = int(row["entry"])
                    except (KeyError, ValueError):
                        continue
                    rows.append(row)
                return rows
        except (OSError, UnicodeError):
            pass
    assignments = report.get("assignments") if isinstance(report, dict) else []
    assignments = assignments if isinstance(assignments, list) else []
    features = lrc_review_tsv.severity_features(report)
    selected = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        entry = assignment.get("entry")
        feature = features.get(int(entry), {}) if isinstance(entry, int) or str(entry).isdigit() else {}
        if assignment.get("review_required") or assignment.get("verification_status") != "trusted":
            selected.append(assignment)
    return lrc_review_tsv.rank_rows(report, selected)


def _candidate_time(value):
    if not isinstance(value, dict):
        return None
    for key in ("time", "timestamp", "chosen_time", "candidate_time", "raw_candidate_time", "start", "onset"):
        number = _number(value.get(key))
        if number is not None:
            return number
    return None


def _seconds_cs(value):
    number = _number(value)
    return None if number is None else _round_cs(number)


def _candidate_rows(assignment, current_cs, auto_cs):
    raw = []
    dropped_count = 0

    def add(value, source, score, rejected=False, required=False, already_cs=False):
        number = _number(value)
        cs = int(number) if already_cs and number is not None else _seconds_cs(value)
        if cs is None:
            return False
        raw.append((cs, source, score, rejected, required))
        return True

    if auto_cs is not None:
        add(auto_cs, "auto", 10000.0, required=True, already_cs=True)
    if current_cs is not None:
        add(current_cs, "current", 9999.0, required=True, already_cs=True)
    candidates = assignment.get("candidates") if isinstance(assignment.get("candidates"), list) else []
    for item in candidates:
        value = _candidate_time(item)
        if not isinstance(item, dict) or not add(
            value, str(item.get("source", "candidate")), _number(item.get("score")) or 0.0
        ):
            continue
    precentral = assignment.get("precentral_candidate_recommendation")
    if isinstance(precentral, dict):
        value = _candidate_time(precentral)
        add(
            value,
            str(precentral.get("source", "precentral-recommendation")),
            _number(precentral.get("score")) or 0.0,
        )
    first = _number(assignment.get("ctc_first_token_start"))
    if first is not None:
        add(first, "ctc-first-token", _number(assignment.get("ctc_first_token_score")) or 0.0)
    rejected = assignment.get("rejected_candidates")
    for item in rejected if isinstance(rejected, list) else []:
        value = _candidate_time(item)
        if not isinstance(item, dict) or not add(
            value, str(item.get("source", "rejected")), _number(item.get("score")) or 0.0, True
        ):
            continue
    merged = []
    for cs, source, score, rejected_flag, required in sorted(raw, key=lambda item: (-item[2], item[0], item[1])):
        match = next((item for item in merged if abs(item["cs"] - cs) <= 5), None)
        diagnostic_only = source in {"raw_asr", "precentral-recommendation"}
        if match is None:
            merged.append({"cs": cs, "label": source, "sources": [source], "score": score, "rejected": rejected_flag, "diagnostic_only": diagnostic_only, "required": required})
        else:
            if source not in match["sources"]:
                match["sources"].append(source)
            match["rejected"] = match["rejected"] and rejected_flag
            match["diagnostic_only"] = match["diagnostic_only"] and diagnostic_only
            match["required"] = match["required"] or required
    required_rows = [item for item in merged if item["required"]]
    other_rows = sorted(
        (item for item in merged if not item["required"]),
        key=lambda item: (item["rejected"], -item["score"], item["cs"], item["label"]),
    )
    ordered = required_rows + other_rows
    visible = ordered[:9]
    dropped_count += max(0, len(ordered) - len(visible))
    visible.sort(key=lambda item: item["cs"])
    for item in visible:
        item.pop("required", None)
    return visible, dropped_count


def _read_log(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _trust_state(report, output_path, auto_path, edits_path, mapping, current_data, reference_data):
    auto_exists = auto_path.is_file()
    edits_exists = edits_path.is_file()
    if auto_exists != edits_exists:
        return True, "incomplete-review-artifacts"
    if not auto_exists:
        published = report.get("published_lrc_sha256")
        if isinstance(published, str) and _sha256_bytes(current_data) != published:
            return True, "lrc-differs-from-published"
        if not isinstance(published, str):
            assignments = report.get("assignments") if isinstance(report.get("assignments"), list) else []
            _bom, _trailing, lines = _split_lrc(current_data)
            for line_indexes, assignment in zip(mapping, assignments):
                expected = _tag_cs(assignment.get("lrc_tag"))
                if any(_line_tag(lines[index]) != expected for index in line_indexes):
                    return True, "lrc-tag-mismatch"
        return False, None
    log = _read_log(edits_path)
    if not isinstance(log, dict) or log.get("auto_lrc_sha256") != _sha256_file(auto_path):
        return True, "review-log-mismatch"
    _cbom, _ctrailing, current_lines = _split_lrc(current_data)
    _abom, _atrailing, auto_lines = _split_lrc(reference_data)
    if len(current_lines) != len(auto_lines) or _cbom != _abom or _ctrailing != _atrailing:
        return True, "lrc-layout-mismatch"
    for current, reference in zip(current_lines, auto_lines):
        if _line_tag(current) is None:
            if current != reference:
                return True, "lrc-non-timing-content-changed"
        elif _tag_tail(current) != _tag_tail(reference):
            return True, "lrc-lyric-content-changed"
    return False, None


def _path_is_file(value):
    if not isinstance(value, (str, pathlib.Path)) or not value:
        return False
    try:
        return pathlib.Path(value).is_file()
    except (OSError, ValueError):
        return False


def _song_payload(report_path, report):
    assignments = report.get("assignments") if isinstance(report.get("assignments"), list) else []
    output_value = report.get("output_path")
    output_path = pathlib.Path(output_value).expanduser() if isinstance(output_value, str) and output_value else None
    auto_path, edits_path = review_artifact_paths(report_path)
    current_data = output_path.read_bytes() if output_path and output_path.is_file() else b""
    reference_data = auto_path.read_bytes() if auto_path.is_file() else current_data
    mapping = None
    mapping_reason = None
    if output_path is None or not output_path.is_file():
        mapping_reason = "lrc-missing"
    else:
        timing_entries = report.get("timing_entries")
        if timing_entries is not None and (not isinstance(timing_entries, int) or timing_entries != len(assignments)):
            mapping_reason = "timing-entry-count-mismatch"
        else:
            mapping, _reference_lines, mapping_reason = _timing_mapping(
                reference_data, report.get("display_lines"), assignments
            )
    _cbom, _ctrailing, current_lines = _split_lrc(current_data)
    readonly = mapping_reason is not None
    reason = mapping_reason
    rows = []
    if mapping is not None:
        trust_readonly, trust_reason = _trust_state(
            report, output_path, auto_path, edits_path, mapping, current_data, reference_data
        )
        if trust_readonly:
            readonly, reason = True, trust_reason
        _abom, _atrailing, auto_lines = _split_lrc(reference_data)
        features = lrc_review_tsv.severity_features(report)
        tsv_rows = _report_rows(report_path, report)
        listed = {
            int(row.get("entry")): row for row in tsv_rows
            if isinstance(row, dict) and str(row.get("entry", "")).isdigit()
        }

        def line_time(lines, index):
            return _line_tag(lines[index]) if 0 <= index < len(lines) else None

        def line_lyrics(lines, indexes):
            return [
                _tag_tail(lines[index]).rstrip(b"\r").decode("utf-8", "replace")
                for index in indexes if 0 <= index < len(lines)
            ]

        def adjacent(index):
            if not (0 <= index < len(mapping)):
                return None
            indexes = mapping[index]
            return {
                "cs": line_time(current_lines, indexes[0]),
                "lyrics": line_lyrics(current_lines, indexes),
                "lyric": "\n".join(line_lyrics(current_lines, indexes)),
            }

        for index, line_indexes in enumerate(mapping):
            assignment = assignments[index]
            feature = features.get(index + 1, {})
            current_cs = line_time(current_lines, line_indexes[0])
            auto_cs = line_time(auto_lines, line_indexes[0])
            lyrics = line_lyrics(current_lines, line_indexes)
            candidates, dropped_count = _candidate_rows(assignment, current_cs, auto_cs)
            row = {
                "entry": index + 1,
                "lyrics": lyrics,
                "lyric": "\n".join(lyrics),
                "current_cs": current_cs,
                "auto_cs": auto_cs,
                "current": current_cs / 100.0 if current_cs is not None else None,
                "auto": auto_cs / 100.0 if auto_cs is not None else None,
                "previous": adjacent(index - 1),
                "next": adjacent(index + 1),
                "review": listed.get(index + 1, {}).get("status", feature.get("status", "UNTRUSTED")),
                "severity_rank": listed.get(index + 1, {}).get("severity_rank"),
                "severity_score": feature.get("severity_score", 0),
                "candidate_spread": feature.get("candidate_spread_s"),
                "reason": listed.get(index + 1, {}).get("reason_codes", ""),
                "candidates": candidates,
                "dropped_count": dropped_count,
            }
            rows.append(row)
    return {
        "report": str(report_path),
        "song": report_path.name.replace(".align-report.json", ""),
        "rows": rows,
        "review_rows": sum(row.get("review") == "REVIEW" for row in rows),
        "read_only": bool(readonly),
        "read_only_reason": reason,
        "audio_path": str(report.get("audio_path", "")),
        "ctc_audio_path": str(report.get("ctc_audio_path", "")),
        "has_flac": _path_is_file(report.get("audio_path")),
        "has_stem": _path_is_file(report.get("ctc_audio_path")),
        "output_path": str(output_path) if output_path else "",
        "base_sha256": _sha256_bytes(current_data),
        "duration_seconds": _number(report.get("duration_seconds")) or 0.0,
        "r2": report.get("r2"),
        "report_data": report,
        "output": output_path,
        "auto": auto_path,
        "edits": edits_path,
    }


def _song_stub(report_path, error):
    auto_path, edits_path = review_artifact_paths(report_path)
    return {
        "report": str(report_path),
        "song": report_path.name.replace(".align-report.json", ""),
        "load_error": f"{type(error).__name__}: {error}",
        "rows": [],
        "review_rows": 0,
        "read_only": True,
        "read_only_reason": "load-error",
        "audio_path": "",
        "ctc_audio_path": "",
        "has_flac": False,
        "has_stem": False,
        "output_path": "",
        "base_sha256": "",
        "duration_seconds": 0.0,
        "r2": None,
        "report_data": {},
        "output": None,
        "auto": auto_path,
        "edits": edits_path,
    }


def _json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _write_json_stage(path, payload):
    stage = path.with_name(path.name + "." + uuid.uuid4().hex + ".stage")
    stage.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(stage, path)


def _save_song(song, body):
    if song["read_only"]:
        return 403, {"error": song["read_only_reason"] or "read-only"}
    if set(body) != {"base_sha256", "edits"} or not isinstance(body.get("edits"), list) or not isinstance(body.get("base_sha256"), str):
        return 400, {"error": "body must contain base_sha256 and edits"}
    output = song["output"]
    if output is None or not output.is_file():
        return 403, {"error": "lrc-missing"}
    current = output.read_bytes()
    if _sha256_bytes(current) != body["base_sha256"]:
        return 409, {"error": "stale base_sha256"}
    assignments = song["report_data"].get("assignments") if isinstance(song["report_data"].get("assignments"), list) else []
    reference = song["auto"].read_bytes() if song["auto"].is_file() else current
    timing_entries = song["report_data"].get("timing_entries")
    if timing_entries is not None and (not isinstance(timing_entries, int) or timing_entries != len(assignments)):
        mapping, mapping_reason = None, "timing-entry-count-mismatch"
    else:
        mapping, _reference_lines, mapping_reason = _timing_mapping(reference, song["report_data"].get("display_lines"), assignments)
    _bom, _trailing, lines = _split_lrc(current)
    if mapping_reason:
        return 403, {"error": mapping_reason}
    edits = []
    seen = set()
    for edit in body["edits"]:
        if not isinstance(edit, dict) or set(edit) != {"entry", "cs", "source", "nudge_cs"}:
            return 400, {"error": "invalid edit"}
        try:
            entry, cs, nudge = int(edit["entry"]), int(edit["cs"]), int(edit["nudge_cs"])
        except (TypeError, ValueError):
            return 400, {"error": "invalid edit"}
        if entry in seen or entry < 1 or entry > len(mapping) or not isinstance(edit["source"], str) or len(edit["source"]) > 64:
            return 400, {"error": "invalid edit"}
        if cs < 0 or cs > int((_number(song["report_data"].get("duration_seconds")) or 0) * 100 + 100):
            return 400, {"error": "timestamp out of range"}
        seen.add(entry)
        current_cs = _line_tag(lines[mapping[entry - 1][0]])
        if cs != current_cs:
            edits.append({"entry": entry, "cs": cs, "source": edit["source"], "nudge_cs": nudge, "before_cs": current_cs})
    if not edits:
        return 200, _song_payload(pathlib.Path(song["report"]), song["report_data"])
    auto = song["auto"]
    edits_path = song["edits"]
    created_auto = False
    if not auto.exists():
        try:
            with auto.open("xb") as handle:
                handle.write(current)
                handle.flush()
                os.fsync(handle.fileno())
            created_auto = True
            _write_json_stage(edits_path, {
                "schema": 1, "report": pathlib.Path(song["report"]).name,
                "lrc_name": output.name, "auto_lrc_sha256": _sha256_file(auto), "edits": []
            })
        except Exception:
            if created_auto:
                auto.unlink(missing_ok=True)
            return 500, {"error": "cannot create review baseline"}
    bom, trailing, lines = _split_lrc(current)
    for edit in edits:
        tag = _format_tag(edit["cs"])
        for line_index in mapping[edit["entry"] - 1]:
            match = _TAG_RE.match(lines[line_index])
            lines[line_index] = tag + lines[line_index][match.end():]
    updated = _join_lrc(bom, trailing, lines)
    stage = output.with_name(output.name + "." + uuid.uuid4().hex + ".stage")
    stage.write_bytes(updated)
    with stage.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    replaced = False
    for attempt in range(3):
        try:
            os.replace(stage, output)
            replaced = True
            break
        except PermissionError:
            if attempt < 2:
                threading.Event().wait(0.2)
    if not replaced:
        stage.unlink(missing_ok=True)
        return 423, {"error": "LRC file is locked (close the player), then Save again"}
    payload = _read_log(edits_path) or {
        "schema": 1, "report": pathlib.Path(song["report"]).name,
        "lrc_name": output.name, "auto_lrc_sha256": _sha256_file(auto), "edits": []
    }
    payload.setdefault("schema", 1)
    payload.setdefault("auto_lrc_sha256", _sha256_file(auto))
    payload.setdefault("edits", [])
    payload["report"] = pathlib.Path(song["report"]).name
    payload["lrc_name"] = output.name
    after_sha = _sha256_file(output)
    auto_lines = _split_lrc(auto.read_bytes())[2]
    payload["edits"].extend({
        "utc": datetime.now(timezone.utc).isoformat(),
        "entry": item["entry"], "auto_cs": _line_tag(auto_lines[mapping[item["entry"] - 1][0]]),
        "before_cs": item["before_cs"], "after_cs": item["cs"], "source": item["source"],
        "nudge_cs": item["nudge_cs"], "lrc_sha256_after": after_sha
    } for item in edits)
    payload["last_lrc_sha256"] = after_sha
    try:
        _write_json_stage(edits_path, payload)
    except OSError:
        return 500, {"error": "LRC saved, edit log not written"}
    return 200, _song_payload(pathlib.Path(song["report"]), song["report_data"])


class ReviewStore:
    def __init__(self, report_dir, targets, read_only=False):
        self.report_dir = pathlib.Path(report_dir).expanduser().resolve()
        self.targets = list(targets)
        self.read_only = bool(read_only)
        self.lock = threading.RLock()
        self.songs = []
        self.excluded = []
        self.reload()

    def reload(self):
        self.songs = []
        reports, self.excluded = _report_files(self.targets, self.report_dir)
        for path in reports:
            try:
                report = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(report, dict):
                    self.songs.append(_song_payload(path, report))
                else:
                    raise ValueError("report must be a JSON object")
            except Exception as exc:
                self.songs.append(_song_stub(path, exc))

    def song(self, index):
        with self.lock:
            if index < 0 or index >= len(self.songs):
                return None
            current = self.songs[index]
            report_path = pathlib.Path(current["report"])
            try:
                report = json.loads(report_path.read_text(encoding="utf-8-sig"))
                if isinstance(report, dict):
                    current = _song_payload(report_path, report)
                    self.songs[index] = current
            except Exception as exc:
                current = _song_stub(report_path, exc)
                self.songs[index] = current
            return current


def _public_song(song, server_read_only=False):
    payload = {key: value for key, value in song.items() if key not in {"report_data", "output", "auto", "edits"}}
    payload["server_read_only"] = bool(server_read_only)
    return payload


class ReviewHandler(http.server.BaseHTTPRequestHandler):
    server_version = "LRCReview/1"

    def log_message(self, *_args):
        return

    def _host_ok(self):
        return self.headers.get("Host", "") == f"127.0.0.1:{self.server.server_port}"

    def _token_header_ok(self):
        return _safe_compare(self.headers.get("X-Review-Token", ""), self.server.token)

    def _token_query_ok(self, query):
        values = urllib.parse.parse_qs(query).get("t", [])
        return bool(values) and _safe_compare(values[0], self.server.token)

    def _send(self, status, data=b"", content_type="application/octet-stream", extra=None, head=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head and data:
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    def _json(self, status, payload):
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def _check_host(self):
        if not self._host_ok():
            self._send(403, b"forbidden\n", "text/plain; charset=utf-8")
            return False
        return True

    def do_GET(self):
        if not self._check_host():
            return
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/":
            try:
                data = (pathlib.Path(__file__).parent / "review_page" / "index.html").read_bytes()
            except OSError:
                self._send(404, b"not found\n", "text/plain; charset=utf-8")
                return
            self._send(200, data, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/songs":
            if not self._token_header_ok():
                self._send(403, b"forbidden\n", "text/plain; charset=utf-8")
                return
            with self.server.store.lock:
                payload = [{"index": index, "song": song["song"], "rows": len(song["rows"]), "review_rows": song["review_rows"], "read_only": song["read_only"], "read_only_reason": song["read_only_reason"], "load_error": song.get("load_error"), "server_read_only": self.server.store.read_only} for index, song in enumerate(self.server.store.songs)]
            self._json(200, payload)
            return
        match = re.fullmatch(r"/api/song/(\d+)", parsed.path)
        if match:
            if not self._token_header_ok():
                self._send(403, b"forbidden\n", "text/plain; charset=utf-8")
                return
            song = self.server.store.song(int(match.group(1)))
            if song is None:
                self._json(404, {"error": "not found"})
            else:
                self._json(200, _public_song(song, self.server.store.read_only))
            return
        match = re.fullmatch(r"/audio/(\d+)/(flac|stem)", parsed.path)
        if match:
            self._serve_audio(int(match.group(1)), match.group(2), parsed.query, False)
            return
        self._send(404, b"not found\n", "text/plain; charset=utf-8")

    def do_HEAD(self):
        if not self._check_host():
            return
        parsed = urllib.parse.urlsplit(self.path)
        match = re.fullmatch(r"/audio/(\d+)/(flac|stem)", parsed.path)
        if match:
            self._serve_audio(int(match.group(1)), match.group(2), parsed.query, True)
            return
        self._send(405, b"method not allowed\n", "text/plain; charset=utf-8")

    def do_POST(self):
        if not self._check_host():
            return
        parsed = urllib.parse.urlsplit(self.path)
        match = re.fullmatch(r"/api/song/(\d+)/save", parsed.path)
        if not match:
            self._send(404, b"not found\n", "text/plain; charset=utf-8")
            return
        if not self._token_header_ok() or self.headers.get("Origin", "") != f"http://127.0.0.1:{self.server.server_port}":
            self._send(403, b"forbidden\n", "text/plain; charset=utf-8")
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            self._send(415, b"unsupported media type\n", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 0 or length > _MAX_BODY:
            self._send(413, b"request too large\n", "text/plain; charset=utf-8")
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeError, ValueError):
            self._send(400, b"invalid json\n", "text/plain; charset=utf-8")
            return
        if not isinstance(body, dict):
            self._send(400, b"invalid json\n", "text/plain; charset=utf-8")
            return
        with self.server.store.lock:
            song = self.server.store.song(int(match.group(1)))
            if song is None or self.server.store.read_only:
                self._json(403, {"error": "read-only"})
                return
            status, result = _save_song(song, body)
            self._json(status, _public_song(result, self.server.store.read_only) if status == 200 else result)

    def _serve_audio(self, index, kind, query, head):
        if not self._token_query_ok(query):
            self._send(403, b"forbidden\n", "text/plain; charset=utf-8", head=head)
            return
        song = self.server.store.song(index)
        if song is None:
            self._send(404, b"not found\n", "text/plain; charset=utf-8", head=head)
            return
        raw = pathlib.Path(song["audio_path"] if kind == "flac" else song["ctc_audio_path"])
        if not raw.is_file():
            self._send(404, b"not found\n", "text/plain; charset=utf-8", head=head)
            return
        try:
            size = raw.stat().st_size
            range_value = self.headers.get("Range", "")
            match = _RANGE_RE.fullmatch(range_value)
            start, end, status = 0, size - 1, 200
            extra = {"Accept-Ranges": "bytes"}
            if match:
                left, right = match.groups()
                if not left and not right:
                    match = None
                elif not left:
                    length = int(right)
                    if size == 0 or length == 0:
                        self._send(416, b"", "application/octet-stream", {"Content-Range": f"bytes */{size}"}, head=head)
                        return
                    start = max(0, size - length)
                else:
                    start = int(left)
                    if start >= size:
                        self._send(416, b"", "application/octet-stream", {"Content-Range": f"bytes */{size}"}, head=head)
                        return
                    if right:
                        candidate_end = int(right)
                        if candidate_end < start:
                            match = None
                        else:
                            end = min(candidate_end, size - 1)
                    else:
                        end = size - 1
                if match:
                    status = 206
                    extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            if status == 200:
                start, end = 0, size - 1
            length = max(0, end - start + 1)
            suffix = raw.suffix.casefold()
            mime = "audio/flac" if suffix == ".flac" else "audio/mpeg" if suffix == ".mp3" else "audio/wav" if suffix == ".wav" else "application/octet-stream"
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            if not head and length:
                with raw.open("rb") as handle:
                    handle.seek(start)
                    remaining = length
                    while remaining:
                        block = handle.read(min(65536, remaining))
                        if not block:
                            break
                        try:
                            self.wfile.write(block)
                        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                            break
                        remaining -= len(block)
        except (OSError, ConnectionResetError, BrokenPipeError):
            return



def run_server(targets, report_dir=None, port=0, no_browser=False, read_only=False, dump_summary=False):
    report_dir = pathlib.Path(report_dir).expanduser().resolve() if report_dir else pathlib.Path(__file__).resolve().parents[1] / "outputs" / "reports"
    store = ReviewStore(report_dir, targets, read_only=read_only)
    if dump_summary:
        for song in store.songs:
            share = sum(len({item.get("cs") for item in row.get("candidates", [])}) >= 2 for row in song["rows"])
            print(json.dumps({"song": song["song"], "rows": len(song["rows"]), "review_rows": song["review_rows"], "read_only": song["read_only"], "reason": song["read_only_reason"], "has_flac": _path_is_file(song["audio_path"]), "has_stem": _path_is_file(song["ctc_audio_path"]), "share_rows": share}, ensure_ascii=True))
        return None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", int(port)), ReviewHandler)
    server.store = store
    server.token = secrets.token_urlsafe(32)
    url = f"http://127.0.0.1:{server.server_port}/#t={urllib.parse.quote(server.token)}"
    print(json.dumps({"url": url, "token": server.token, "songs": len(store.songs), "excluded": len(store.excluded),
                      "excluded_files": [path.name for path in store.excluded]}, ensure_ascii=False), flush=True)
    if not no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return server


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*")
    parser.add_argument("--report-dir", type=pathlib.Path)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--dump-summary", action="store_true")
    args = parser.parse_args()
    return run_server(args.targets, args.report_dir, args.port, args.no_browser, args.read_only, args.dump_summary)


if __name__ == "__main__":
    main()
