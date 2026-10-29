from __future__ import annotations

import hashlib
import http.server
import math
import json
import struct
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from collections import OrderedDict
import os
import pathlib
import re
import secrets
import socket
import subprocess
import threading
import time
import uuid
import urllib.parse
import webbrowser
import sys
import wave
from datetime import datetime, timezone

import lrc_review_tsv
import lyrics_tag_export
import review_presentation
import runtime_paths
from lrc_manual_edits import review_artifact_paths

_EXCLUDED_DIR_NAMES = ("lrc bk",)
_TAG_RE = re.compile(rb"^\[(\d{2,}):(\d{2})\.(\d{2})\]")
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_MAX_BODY = 256 * 1024
_PEAK_CACHE = OrderedDict()
_PEAK_CACHE_LOCK = threading.Lock()
_PEAK_CACHE_LIMIT = 256


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


def _read_stem_peaks(path, from_cs, to_cs, bins):
    path = pathlib.Path(path).expanduser().resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, from_cs, to_cs, bins)
    with _PEAK_CACHE_LOCK:
        cached = _PEAK_CACHE.get(key)
        if cached is not None:
            _PEAK_CACHE.move_to_end(key)
            return cached
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16000 or handle.getcomptype() != "NONE":
                raise ValueError("stem must be 16 kHz mono signed-16 PCM")
            start_sample = math.floor(from_cs * 160)
            end_sample = math.floor(to_cs * 160)
            total_samples = end_sample - start_sample
            handle.setpos(min(start_sample, handle.getnframes()))
            data = handle.readframes(max(0, total_samples))
    except (wave.Error, EOFError, struct.error) as exc:
        raise ValueError("stem must be 16 kHz mono signed-16 PCM") from exc
    sample_count = len(data) // 2
    samples = struct.unpack("<" + "h" * sample_count, data) if sample_count else ()
    peaks = []
    for index in range(bins):
        left = (total_samples * index) // bins
        right = (total_samples * (index + 1)) // bins
        values = samples[left:right] if left < sample_count else ()
        if not values:
            peaks.append([0.0, 0.0])
            continue
        peaks.append([
            round(max(-1.0, min(1.0, min(values) / 32768.0)), 3),
            round(max(-1.0, min(1.0, max(values) / 32768.0)), 3),
        ])
    payload = {
        "from_cs": from_cs,
        "to_cs": to_cs,
        "bins": bins,
        "sample_rate": 16000,
        "peaks": peaks,
    }
    with _PEAK_CACHE_LOCK:
        _PEAK_CACHE[key] = payload
        _PEAK_CACHE.move_to_end(key)
        while len(_PEAK_CACHE) > _PEAK_CACHE_LIMIT:
            _PEAK_CACHE.popitem(last=False)
    return payload


def _parse_peaks_query(query):
    values = urllib.parse.parse_qs(query)
    try:
        from_cs = int(values["from"][0])
        to_cs = int(values["to"][0])
        bins = int(values["bins"][0])
    except (KeyError, IndexError, TypeError, ValueError):
        raise ValueError("from, to and bins are required integers")
    if from_cs < 0 or to_cs <= from_cs or to_cs - from_cs > 600 or bins <= 0 or bins > 600:
        raise ValueError("invalid peak range")
    return from_cs, to_cs, bins


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


_LEAD_MARK_RE = re.compile(rb"^\*+[ \t]*(?=\[\d{2,}:\d{2}\.\d{2}\])")
_TRAIL_MARK_RE = re.compile(rb"[ \t]*\*+(?=\r?$)")


def _line_marks(line):
    """(leading, trailing) user marks on a timed line: ``*[00:01.00]text``, optionally also ``text*``.

    A trailing ``*`` on its own is lyric text (footnote, censor symbol), never a mark.
    """
    lead = _LEAD_MARK_RE.match(line)
    if not lead:
        return False, False
    rest = line[lead.end():]
    match = _TAG_RE.match(rest)
    if not match:
        return False, False
    return True, bool(_TRAIL_MARK_RE.search(rest[match.end():]))


def _strip_line_marks(line):
    lead = _LEAD_MARK_RE.match(line)
    if not lead:
        return line
    rest = line[lead.end():]
    match = _TAG_RE.match(rest)
    if not match:
        return line
    return rest[:match.end()] + _TRAIL_MARK_RE.sub(b"", rest[match.end():])


def _strip_marks(data):
    """The LRC bytes without the reviewer's ``*`` marks, and the set of line indexes that carried one."""
    bom, trailing, lines = _split_lrc(data)
    marked = {index for index, line in enumerate(lines) if any(_line_marks(line))}
    if not marked:
        return data, marked
    cleaned = [_strip_line_marks(line) if index in marked else line for index, line in enumerate(lines)]
    return _join_lrc(bom, trailing, cleaned), marked


def _content_lines(lines):
    """The lines up to the last non-blank one; trailing blank lines are never mapped (and never rewritten)."""
    end = len(lines)
    while end and not lines[end - 1].strip():
        end -= 1
    return lines[:end]


def _song_mapping(report, assignments, reference_raw):
    """(mapping, reason) for the reference LRC.

    The published tag sequence must match the display region exactly once; then the tag structure alone fixes
    which lines belong to which entry, whatever the lyric text now says (stale or not). Anything else yields
    the specific reason and no mapping, so nothing is ever guessed and saved.
    """
    clean = _strip_marks(reference_raw)[0]
    mapping, _lines, reason = _timing_mapping(clean, report.get("display_lines"), assignments)
    if reason is not None:
        return None, reason
    timing_entries = report.get("timing_entries")
    if timing_entries is not None and (not isinstance(timing_entries, int) or timing_entries != len(assignments)):
        return None, "timing-entry-count-mismatch"
    return mapping, None


def _timing_mapping(data, display_lines, assignments):
    _bom, _trailing, all_lines = _split_lrc(data)
    lines = _content_lines(all_lines)
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


def _find_beside(names, folders):
    for folder in folders:
        if folder is None:
            continue
        for name in names:
            if not name:
                continue
            candidate = folder / name
            if _path_is_file(candidate):
                return candidate
    return None


def _relocated(report, anchor=None):
    """Bind a report to the FLAC/LRC where the user is reviewing, not where it was aligned.

    ``anchor`` is the folder the user opened (dragged folder, or the dragged file's folder);
    files there win. Otherwise a missing FLAC/LRC is looked up beside its partner.
    """
    audio_value = report.get("audio_path")
    output_value = report.get("output_path")
    audio = pathlib.Path(audio_value) if isinstance(audio_value, str) and audio_value else None
    output = pathlib.Path(output_value) if isinstance(output_value, str) and output_value else None
    anchor = pathlib.Path(anchor) if anchor else None
    audio_names = [audio.name if audio else None]
    audio_names += [(output.stem + ext) if output else None for ext in runtime_paths.AUDIO_EXTENSIONS]
    output_names = [output.name if output else None, (audio.stem + ".lrc") if audio else None]
    new_audio = _find_beside(audio_names, [anchor]) or (audio if _path_is_file(audio) else None)
    new_output = _find_beside(output_names, [anchor]) or (output if _path_is_file(output) else None)
    if new_audio is None:
        new_audio = _find_beside(audio_names, [new_output.parent if new_output else None])
    if new_output is None:
        new_output = _find_beside(output_names, [new_audio.parent if new_audio else None])
    changed = {}
    if new_audio is not None and audio is not None and new_audio != audio:
        changed["audio_path"] = str(new_audio)
    if new_output is not None and output is not None and new_output != output:
        changed["output_path"] = str(new_output)
    if not changed:
        return report
    moved = dict(report)
    moved.update(changed)
    moved["review_relocated_from"] = {key: report.get(key) for key in changed}
    return moved


def _names_of(data):
    names = set()
    for key in ("audio_path", "output_path"):
        value = data.get(key)
        if isinstance(value, str) and value:
            names.add(pathlib.PureWindowsPath(value).name.casefold())
            names.add(pathlib.Path(value).name.casefold())
    return names


def _newest(reports):
    if len(reports) <= 1:
        return reports
    return [max(reports, key=lambda item: item.stat().st_mtime_ns)]


class _ReportIndex:
    """Every report under ``report_dir`` parsed once per load, however many targets ask about it.

    Each FLAC/LRC target used to re-glob the report directory and re-parse every report, so a
    180-song review took 180 x 180 JSON loads. The index lists the directory once, parses each
    report at most once (``cache`` is shared with ``ReviewStore.reload`` so the page build does
    not parse it again) and remembers the casefolded paths and names the matching needs.
    """

    def __init__(self, report_dir, cache=None):
        self.report_dir = report_dir
        self.cache = cache if cache is not None else {}
        self._listing = None
        self._keys = {}

    def load(self, report):
        if report not in self.cache:
            self.cache[report] = _load_report(report)
        return self.cache[report]

    def listing(self):
        if self._listing is None:
            self._listing = list(self.report_dir.glob("*.align-report.json"))
        return self._listing

    def entries(self):
        """``(report, data, audio_casefold, output_casefold, names)`` for every parseable report."""
        for report in self.listing():
            data = self.load(report)
            if data is None:
                yield report, None, None, None, None
                continue
            keys = self._keys.get(report)
            if keys is None:
                keys = (
                    str(data.get("audio_path", "")).casefold(),
                    str(data.get("output_path", "")).casefold(),
                    _names_of(data),
                )
                self._keys[report] = keys
            yield (report, data) + keys


def _report_files(targets, report_dir, *, scan_report_dir=True, anchors=None, cache=None):
    found = []
    excluded = []
    index = _ReportIndex(report_dir, cache)
    targets = list(targets or [])
    if not targets and scan_report_dir:
        targets = [report_dir]
    for raw in targets:
        path = pathlib.Path(raw).expanduser().resolve()
        anchor = None
        if path.is_file() and path.name.endswith(".align-report.json"):
            candidates = [path]
        elif path.is_file():
            candidates = []
            by_name = []
            target = str(path).casefold()
            target_name = path.name.casefold()
            for report, data, audio_key, output_key, names in index.entries():
                if data is None:
                    if report not in excluded:
                        excluded.append(report)
                    continue
                if target in (audio_key, output_key):
                    candidates.append(report)
                elif target_name in names:
                    by_name.append(report)
            # A song moved after alignment is still the same song: match it by file name.
            candidates = candidates or _newest(by_name)
            anchor = path.parent
        elif path.is_dir():
            # Archived reports (archive_stale_reports.py) live under _archive/ and must stay hidden.
            local_reports = sorted(
                item for item in path.rglob("*.align-report.json")
                if not any(part.casefold() == "_archive" for part in item.relative_to(path).parts[:-1])
            )
            anchor = None
            if local_reports:
                candidates = local_reports
            else:
                anchor = path
                prefix = str(path).casefold().rstrip("\\/") + "\\"
                here = {item.name.casefold() for item in path.iterdir() if item.is_file()}
                candidates = []
                by_name = {}
                for report, data, audio_key, _output_key, _names in index.entries():
                    if data is None:
                        if report not in excluded:
                            excluded.append(report)
                        continue
                    if audio_key.startswith(prefix):
                        candidates.append(report)
                    else:
                        # Songs moved into this folder after alignment: match by file name.
                        audio_value = data.get("audio_path")
                        audio_name = pathlib.PureWindowsPath(audio_value).name.casefold() if isinstance(audio_value, str) else ""
                        if audio_name and audio_name in here:
                            by_name.setdefault(audio_name, []).append(report)
                for group in by_name.values():
                    candidates.extend(_newest(group))
        else:
            candidates = []
            anchor = None
        if not (path.is_file() and not path.name.endswith(".align-report.json")) and not path.is_dir():
            anchor = None
        for report in candidates:
            data = index.load(report)
            if data is None:
                if report not in excluded:
                    excluded.append(report)
                continue
            if not isinstance(data.get("audio_path"), str):
                if report not in excluded:
                    excluded.append(report)
                continue
            data = _relocated(data, anchor)
            paths = (report, data.get("audio_path", ""), data.get("output_path", ""))
            if not report.is_file():
                continue
            if any(_path_is_excluded(item) for item in paths):
                if report not in excluded:
                    excluded.append(report)
            elif report not in found:
                found.append(report)
                if anchors is not None and anchor is not None:
                    anchors[report] = anchor
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
    diagnostic_sources = {"raw_asr", "precentral-recommendation"}

    def add(value, source, score, rejected=False, required=False, already_cs=False):
        number = _number(value)
        cs = int(number) if already_cs and number is not None else _seconds_cs(value)
        if cs is None:
            return False
        raw.append({
            "cs": cs,
            "source": str(source),
            "score": float(score),
            "rejected": bool(rejected),
            "required": bool(required),
            "diagnostic": str(source) in diagnostic_sources,
        })
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
    # Alignment hypotheses are the local-window replays Central weighed but did not select; on the
    # 36 checked songs one of them sat within 0.2 s of the checked time on a line the page could
    # not otherwise offer, at no extra options on correct lines.
    hypotheses = assignment.get("alignment_hypotheses") if isinstance(assignment.get("alignment_hypotheses"), list) else []
    for item in hypotheses:
        if isinstance(item, dict):
            add(_candidate_time(item), str(item.get("source", "hypothesis")), _number(item.get("confidence")) or 0.0)
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
    groups = []
    for member in sorted(raw, key=lambda item: (-item["score"], item["cs"], item["source"])):
        group = next((item for item in groups if abs(item["anchor_cs"] - member["cs"]) <= 5), None)
        if group is None:
            groups.append({"anchor_cs": member["cs"], "members": [member]})
        else:
            group["members"].append(member)

    merged = []
    for group in groups:
        members = group["members"]
        real_members = [member for member in members if not member["diagnostic"]]
        selected = max(real_members or members, key=lambda item: (item["score"], -item["cs"], item["source"]))
        timestamps = sorted({
            member["cs"]
            for member in raw
            if not member["required"]
            and not member["diagnostic"]
            and member["cs"] != selected["cs"]
            and abs(member["cs"] - selected["cs"]) <= 15
        })
        span_values = [selected["cs"], *timestamps]
        merged.append({
            "cs": selected["cs"] if real_members else group["anchor_cs"],
            "label": selected["source"],
            "sources": list(dict.fromkeys(member["source"] for member in members)),
            "score": max(member["score"] for member in members),
            "rejected": all(member["rejected"] for member in members),
            "diagnostic_only": all(member["diagnostic"] for member in members),
            "required": any(member["required"] for member in members),
            "members": [dict(member) for member in members],
            "agree_count": len(timestamps),
            "agree_span_cs": max(span_values) - min(span_values) if timestamps else 0,
        })
    required_rows = [item for item in merged if item["required"]]
    other_rows = sorted(
        (item for item in merged if not item["required"]),
        key=lambda item: (item["rejected"], -item["score"], item["cs"], item["label"]),
    )
    ordered = required_rows + other_rows
    visible = ordered[:9]
    extra = ordered[9:]
    visible.sort(key=lambda item: item["cs"])
    extra.sort(key=lambda item: item["cs"])
    return visible, extra


def _read_log(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _trust_state(report, output_path, auto_path, edits_path, mapping, current_data, reference_data, current_raw=None):
    auto_exists = auto_path.is_file()
    edits_exists = edits_path.is_file()
    if auto_exists != edits_exists:
        return True, "incomplete-review-artifacts"
    if not auto_exists:
        published = report.get("published_lrc_sha256")
        if isinstance(published, str) and _sha256_bytes(current_data if current_raw is None else current_raw) != published:
            # Hand-edited LRC (e.g. `*` marks): the mapping already matched it line by line, so show it, badged.
            return False, "lrc-differs-from-published"
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


def _lrc_entry_groups(current_data):
    """Entries read from the LRC file itself: consecutive tag lines sharing one timestamp are one entry.

    That is how this tool writes original + translation pairs; a lone tag line is a one-line entry. Header
    lines ([ti:] ...), blank lines and a title card at 00:00.00 are not entries. Returns line-index groups.
    """
    _bom, _trailing, all_lines = _split_lrc(current_data)
    lines = _content_lines(all_lines)
    groups = []
    previous = None
    for index, line in enumerate(lines):
        cs = _line_tag(line)
        if cs is None:
            previous = None
            continue
        if groups and previous == cs and groups[-1][-1] == index - 1:
            groups[-1].append(index)
        else:
            groups.append([index])
        previous = cs
    if groups and _is_title_group(lines, groups[0]):
        groups = groups[1:]
    return groups


_HEADER_RE = re.compile(rb"^\[(?:ti|ar|al|au|by|offset|length|re|ve):", re.I)


def _is_title_group(lines, group):
    """The tool's title card: the single first tagged line at 00:00.00, in a file with a [ti:]/[ar:]/[al:] header or an
    ``Artist - Title`` text (evaluate_lrc.is_generated_title_card). Any other 00:00.00 line is a real entry."""
    if len(group) != 1 or _line_tag(lines[group[0]]) != 0:
        return False
    if any(_line_tag(line) is not None for line in lines[:group[0]]):
        return False
    if any(_HEADER_RE.match(line) for line in lines[:group[0]]):
        return True
    return b" - " in _tag_tail(lines[group[0]])


def _line_key(lines, indexes):
    """Stable identity of an entry: first line index plus a hash of its text (the timestamp is not part of it)."""
    texts = b"\n".join(_tag_tail(_strip_line_marks(lines[index])).rstrip(b"\r") for index in indexes)
    return f"{indexes[0]}:{hashlib.sha1(texts).hexdigest()}"


def _link_assignments(groups, current_lines, assignments):
    """Report assignment index (or None) per LRC-derived entry.

    Position when the counts agree and all but a few tags still match (a few retimed lines); otherwise the
    next unused assignment with the same tag, in order. Entries without a match simply carry no report data.
    """
    tags = [_line_tag(current_lines[group[0]]) for group in groups]
    wanted = [_tag_cs(assignment.get("lrc_tag")) if isinstance(assignment, dict) else None for assignment in assignments]
    if len(groups) == len(assignments) and sum(a != b for a, b in zip(tags, wanted)) <= max(1, len(groups) // 10):
        return list(range(len(groups)))
    links = []
    cursor = 0
    for tag in tags:
        found = next((number for number in range(cursor, len(wanted)) if wanted[number] == tag), None)
        if found is not None:
            cursor = found + 1
        links.append(found)
    return links


def _resolve_entries(report_path, report, output_path, current_raw):
    """(mapping, links, model, reason, manual_edited, manual_reason) for the LRC beside the audio.

    The report mapping is used when it matches the file uniquely and the review artifacts agree; otherwise the
    entries are built from the LRC file (the file is the source of truth). ``reason`` is only ever
    ``lrc-unparsable`` (no timestamped entry at all).
    """
    assignments = report.get("assignments") if isinstance(report.get("assignments"), list) else []
    auto_path, edits_path = review_artifact_paths(report_path)
    current_data, marked_lines = _strip_marks(current_raw)
    reference_raw = auto_path.read_bytes() if auto_path.is_file() else current_raw
    reference_data = _strip_marks(reference_raw)[0]
    mapping, _reason = _song_mapping(report, assignments, reference_raw)
    if mapping is not None:
        trust_readonly, trust_reason = _trust_state(
            report, output_path, auto_path, edits_path, mapping, current_data, reference_data, current_raw
        )
        if not trust_readonly:
            manual_reason = trust_reason or ("user-marks" if marked_lines else None)
            return mapping, list(range(len(mapping))), "report", None, bool(manual_reason), manual_reason
    groups = _lrc_entry_groups(current_data)
    if not groups:
        return None, [], "lrc", "lrc-unparsable", False, None
    current_lines = _split_lrc(current_data)[2]
    links = _link_assignments(groups, current_lines, assignments)
    return groups, links, "lrc", None, bool(marked_lines), "user-marks" if marked_lines else None


def _path_is_file(value):
    if not isinstance(value, (str, pathlib.Path)) or not value:
        return False
    try:
        return pathlib.Path(value).is_file()
    except (OSError, ValueError):
        return False


def _lyrics_embedded(audio_value):
    """True when any of the song's own audio files carries embedded lyrics (its .lrc was archived after an export)."""
    if not isinstance(audio_value, str) or not audio_value:
        return False
    try:
        found = lyrics_tag_export.candidates(pathlib.Path(audio_value))
    except OSError:
        return False
    for candidate in found:
        try:
            if lyrics_tag_export.read_lyrics(candidate):
                return True
        except (lyrics_tag_export.ExportError, OSError):
            continue
    return False


def _song_payload(report_path, report, anchor=None):
    report = _relocated(report, anchor)
    assignments = report.get("assignments") if isinstance(report.get("assignments"), list) else []
    output_value = report.get("output_path")
    output_path = pathlib.Path(output_value).expanduser() if isinstance(output_value, str) and output_value else None
    auto_path, edits_path = review_artifact_paths(report_path)
    current_raw = output_path.read_bytes() if output_path and output_path.is_file() else b""
    current_data, marked_lines = _strip_marks(current_raw)
    reference_raw = auto_path.read_bytes() if auto_path.is_file() else current_raw
    reference_data = _strip_marks(reference_raw)[0]
    mapping = None
    links = []
    entry_model = None
    mapping_reason = None
    manual_edited = False
    manual_reason = None
    if output_path is None or not output_path.is_file():
        mapping_reason = "lrc-exported" if _lyrics_embedded(report.get("audio_path")) else "lrc-missing"
    else:
        mapping, links, entry_model, mapping_reason, manual_edited, manual_reason = _resolve_entries(
            report_path, report, output_path, current_raw
        )
    _cbom, _ctrailing, current_lines = _split_lrc(current_data)
    readonly = mapping_reason is not None
    reason = mapping_reason
    rows = []
    if mapping is not None:
        _abom, _atrailing, auto_lines = _split_lrc(reference_data)
        if len(auto_lines) != len(current_lines):
            auto_lines = []
        assigned = [assignments[link] if link is not None else {} for link in links]
        features = lrc_review_tsv.severity_features(report)
        whisperx_rows = report.get("whisperx_row_evidence")
        whisperx_entries = whisperx_rows.get("entries") if isinstance(whisperx_rows, dict) else None
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

        # Neighbour fit uses the LRC as it is now, so a corrected neighbour counts.
        written = [line_time(current_lines, indexes[0]) for indexes in mapping]
        speech_rate = review_presentation.song_speech_rate(assigned, written)

        for index, line_indexes in enumerate(mapping):
            assignment = assigned[index]
            link = links[index]
            feature = features.get(link + 1, {}) if link is not None else {}
            current_cs = line_time(current_lines, line_indexes[0])
            fit = review_presentation.line_fit(
                assignment, current_cs,
                assigned[index - 1] if index else None,
                written[index - 1] if index else None,
                written[index + 1] if index + 1 < len(written) else None,
                speech_rate,
            )
            auto_cs = line_time(auto_lines, line_indexes[0])
            lyrics = line_lyrics(current_lines, line_indexes)
            candidates, extra_candidates = _candidate_rows(assignment, current_cs, auto_cs)
            user_marked = any(line in marked_lines for line in line_indexes)
            listed_row = listed.get(link + 1, {}) if link is not None else {}
            status = listed_row.get("status", feature.get("status", "UNTRUSTED"))
            reason_codes = lrc_review_tsv.reason_codes_for(
                assignment,
                status,
            )
            options = review_presentation.present_options(candidates, extra_candidates, current_cs)
            alternatives = sum(option.get("kind") != "keep" for option in options)
            risk_tier, risk_basis = (
                review_presentation.classify_risk(
                    assignment, reason_codes, _tag_cs(assignment.get("lrc_tag")),
                    whisperx_entries if isinstance(whisperx_entries, list) else (), fit,
                ) if status == "REVIEW" else (None, None)
            )
            reason_text = review_presentation.reason_sentence(reason_codes, options_count=alternatives)
            if link is None:
                reason_codes, reason_text = ["no-alignment-data"], "無對齊資料"
            if user_marked:
                status, risk_tier, risk_basis = "REVIEW", "high", "user-mark"
                reason_codes = [*reason_codes, "user-mark"]
                reason_text = "你在 LRC 裡用 * 標記過這句。" + ("" if alternatives else "報告裡沒有這句的候選時間。")
            row = {
                "entry": index + 1,
                "line_key": _line_key(current_lines, line_indexes),
                "options": options,
                "reason_text": reason_text,
                "user_marked": user_marked,
                "lyrics": lyrics,
                "lyric": "\n".join(lyrics),
                "current_cs": current_cs,
                "auto_cs": auto_cs,
                "current": current_cs / 100.0 if current_cs is not None else None,
                "auto": auto_cs / 100.0 if auto_cs is not None else None,
                "previous": adjacent(index - 1),
                "next": adjacent(index + 1),
                "review": status,
                "risk": risk_tier,
                "risk_basis": risk_basis,
                "fit": fit,
                "severity_rank": listed_row.get("severity_rank"),
                "severity_score": feature.get("severity_score", 0),
                "candidate_spread": feature.get("candidate_spread_s"),
                "reason_codes": reason_codes,
                "reason": ",".join(reason_codes),
                "candidates": candidates,
                "extra_candidates": extra_candidates,
                "dropped_count": len(extra_candidates),
            }
            rows.append(row)
    return {
        "report": str(report_path),
        "song": report_path.name.replace(".align-report.json", ""),
        "rows": rows,
        "review_rows": sum(row.get("review") == "REVIEW" for row in rows),
        "read_only": bool(readonly),
        "read_only_reason": reason,
        "entry_model": entry_model,
        "manual_edited": bool(manual_edited),
        "manual_edit_reason": manual_reason,
        "marked_rows": sum(bool(row.get("user_marked")) for row in rows),
        "audio_path": str(report.get("audio_path", "")),
        "ctc_audio_path": str(report.get("ctc_audio_path", "")),
        "has_flac": _path_is_file(report.get("audio_path")),
        "has_stem": _path_is_file(report.get("ctc_audio_path")),
        "output_path": str(output_path) if output_path else "",
        "base_sha256": _sha256_bytes(current_raw),
        "duration_seconds": _number(report.get("duration_seconds")) or 0.0,
        "song_risk": review_presentation.song_risk(report),
        "r2": report.get("r2"),
        "report_data": report,
        "output": output_path,
        "auto": auto_path,
        "edits": edits_path,
        "anchor": str(anchor) if anchor else None,
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


def _stage_bytes(path, data):
    stage = path.with_name(path.name + "." + uuid.uuid4().hex + ".stage")
    try:
        with stage.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        stage.unlink(missing_ok=True)
        raise
    return stage


def _write_json_stage(path, payload):
    stage = _stage_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    try:
        os.replace(stage, path)
    except Exception:
        stage.unlink(missing_ok=True)
        raise


_DECISIONS = ("keep", "change")


def _decisions_path(song):
    edits_path = pathlib.Path(song["edits"])
    return edits_path.with_name(edits_path.name[:-len(".review-edits.json")] + ".review-decisions.jsonl")


def _record_decision(song, body):
    """Append one reviewer verdict (keep or change) so thresholds can later be tuned on real outcomes."""
    if song["read_only"]:
        return 403, {"error": song["read_only_reason"] or "read-only"}
    if set(body) - {"entry", "decision", "cs", "source"} or body.get("decision") not in _DECISIONS:
        return 400, {"error": "body must contain entry and decision (keep or change)"}
    try:
        entry = int(body.get("entry"))
        cs = None if body.get("cs") is None else int(body["cs"])
    except (TypeError, ValueError):
        return 400, {"error": "invalid decision"}
    row = next((item for item in song.get("rows", []) if item.get("entry") == entry), None)
    if row is None:
        return 400, {"error": "unknown entry"}
    source = body.get("source")
    record = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "entry": entry,
        "decision": body["decision"],
        "chosen_cs": cs if cs is not None else row.get("current_cs"),
        "current_cs": row.get("current_cs"),
        "auto_cs": row.get("auto_cs"),
        "review": row.get("review"),
        "risk": row.get("risk"),
        "risk_basis": row.get("risk_basis"),
        "reason_codes": list(row.get("reason_codes") or []),
        "severity_rank": row.get("severity_rank"),
        "candidate_spread_s": row.get("candidate_spread"),
        "source": source[:64] if isinstance(source, str) else None,
        "options": [{"kind": option.get("kind"), "label": option.get("label"), "cs": option.get("cs")} for option in row.get("options") or []],
    }
    path = _decisions_path(song)
    try:
        with open(path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        return 500, {"error": f"cannot record decision: {exc}"}
    return 200, {"recorded": path.name}


def _export_song(song, body, report_dir, store=None):
    """Embed the saved .lrc into the song's own audio file (check=true only describes the targets)."""
    if song["read_only"]:
        return 403, {"error": song["read_only_reason"] or "read-only"}
    if set(body) - {"audio", "check"} or not isinstance(body.get("audio", ""), str):
        return 400, {"error": "body must contain audio (and optionally check)"}
    output = song["output"]
    if output is None or not output.is_file():
        return 403, {"error": "lrc-missing"}
    audio_value = song.get("audio_path") or ""
    if not audio_value or not _path_is_file(audio_value):
        return 403, {"error": "找不到這首歌的音檔"}
    targets = lyrics_tag_export.candidates(pathlib.Path(audio_value))
    try:
        if body.get("check"):
            lrc_lines = lyrics_tag_export._line_count(lyrics_tag_export.lrc_text(output.read_bytes()))
            return 200, {"targets": [lyrics_tag_export.info(item) for item in targets], "lrc_lines": lrc_lines}
        chosen = next((item for item in targets if item.name == body.get("audio")), None)
        if chosen is None:
            return 400, {"error": "audio is not one of this song's audio files"}
        work_dir = report_dir.parent if report_dir.name.casefold() == "reports" else report_dir
        started = time.monotonic()
        timings = {}
        def publish(**fields):
            if store is not None:
                store.update_export_status(**fields)
        publish(song=song.get("song"), step="snapshot_before", started=started, active=True)
        archive = {}
        try:
            result = lyrics_tag_export.export(chosen, output.read_bytes(), work_dir, timings=timings,
                                              on_step=lambda name: publish(step=name or "archive"))
            result["seconds"] = round(time.monotonic() - started, 1)
            step = time.perf_counter()
            publish(step="archive")
            archive = result["archive"] = lyrics_tag_export.archive_lrc(output, work_dir, chosen.name)
            timings["archive"] = round((time.perf_counter() - step) * 1000)
            result["timings"] = timings
            result["_log"] = (work_dir, song.get("song"), archive)
            return 200, result
        except lyrics_tag_export.ExportError as exc:
            _export_log(work_dir, song.get("song"), "failed", archive, str(exc), timings, started)
            publish(active=False)
            return 422, {"error": str(exc)}
        except Exception as exc:  # a failed backup copy, an unreadable LRC, a broken archive step: logged and answered, not dropped
            _export_log(work_dir, song.get("song"), "failed", archive, f"{type(exc).__name__}: {exc}", timings, started)
            publish(active=False)
            return 500, {"error": f"匯出失敗：{exc}"}
    except lyrics_tag_export.ExportError as exc:
        return 422, {"error": str(exc)}
    except Exception as exc:
        return 500, {"error": f"匯出失敗：{exc}"}


def _export_log(work_dir, song_name, result, archive, error, timings, started):
    """One line per export in <outputs>/export.log: timestamp, song, result, archive path, error, step timings."""
    archive = archive if isinstance(archive, dict) else {}
    where = archive.get("to") or archive.get("error") or archive.get("mode") or ""
    steps = " ".join(f"{key}={value}" for key, value in timings.items())
    fields = [datetime.now().astimezone().isoformat(timespec="seconds"), str(song_name or ""), result, str(where), error, steps]
    line = "\t".join(" ".join(str(field).split()) for field in fields)
    try:
        with open(pathlib.Path(work_dir) / "export.log", "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def _free_slot(cs, prev, nxt, neighbours, limit):
    """Nearest free time for a line whose time equals a neighbour's: between its neighbours when there is room (order kept),
    otherwise just beyond the farther neighbour, away from the nearer one."""
    toward_up = cs == prev
    low, high = (prev if prev is not None else 0), (nxt if nxt is not None else limit + 1)
    room = [value for value in range(max(low, 0) + 1, min(high, limit + 1)) if value not in neighbours]
    if room:
        return min(room, key=lambda value: (abs(value - cs), 0 if (value > cs) == toward_up else 1))
    far = next((value for value in (prev, nxt) if value is not None and value != cs), cs)
    down = far < cs or (far == cs and not toward_up)
    for direction in (down, not down):
        value = far - 1 if direction else far + 1
        while 1 <= value <= limit and value in neighbours:
            value += -1 if direction else 1
        if 1 <= value <= limit:
            return value
    return cs


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
    mapping, _links, model, mapping_reason, _manual, _manual_reason = _resolve_entries(
        pathlib.Path(song["report"]), song["report_data"], output, current
    )
    _bom, _trailing, lines = _split_lrc(current)
    if mapping_reason:
        return 403, {"error": mapping_reason}
    requested = []
    seen = set()
    for edit in body["edits"]:
        if not isinstance(edit, dict) or not {"entry", "cs", "source", "nudge_cs"} <= set(edit) <= {"entry", "cs", "source", "nudge_cs", "line_key"}:
            return 400, {"error": "invalid edit"}
        try:
            entry, cs, nudge = int(edit["entry"]), int(edit["cs"]), int(edit["nudge_cs"])
        except (TypeError, ValueError):
            return 400, {"error": "invalid edit"}
        if entry in seen or entry < 1 or entry > len(mapping) or not isinstance(edit["source"], str) or len(edit["source"]) > 64:
            return 400, {"error": "invalid edit"}
        if "line_key" in edit and not isinstance(edit["line_key"], str):
            return 400, {"error": "invalid edit"}
        if cs < 0 or cs > int((_number(song["report_data"].get("duration_seconds")) or 0) * 100 + 100):
            return 400, {"error": "timestamp out of range"}
        if "line_key" in edit and edit["line_key"] != _line_key(lines, mapping[entry - 1]):
            return 422, {"error": f"第 {entry} 句對應的句子已變動（檔案在別處被改過），未儲存", "entry": entry}
        seen.add(entry)
        requested.append({"entry": entry, "cs": cs, "source": edit["source"], "nudge_cs": nudge})
    final = {}
    for position in range(len(mapping)):
        final[position + 1] = _line_tag(_strip_line_marks(lines[mapping[position][0]]))
    before = dict(final)
    adjusted = []
    limit = int((_number(song["report_data"].get("duration_seconds")) or 0) * 100 + 100)
    if model == "lrc":
        # Entries are tag groups here: a save must never give a line the neighbour's time (the groups would merge).
        # Every requested value is applied first, so a neighbour in the same save counts at its new time.
        for edit in requested:
            edit["asked"] = edit["cs"]
            edit["cs"] = max(edit["cs"], 1)
            final[edit["entry"]] = edit["cs"]
    for edit in sorted(requested, key=lambda item: item["entry"]):
        entry = edit["entry"]
        cs = edit["cs"]
        if model == "lrc":
            prev, nxt = final.get(entry - 1), final.get(entry + 1)
            neighbours = {value for value in (prev, nxt) if value is not None}
            if cs in neighbours:
                found = _free_slot(cs, prev, nxt, neighbours, limit)
                item = {"entry": entry, "requested_cs": edit["asked"], "cs": found}
                cs = found
                adjusted.append(item)
            elif cs != edit["asked"]:
                item = {"entry": entry, "requested_cs": edit["asked"], "cs": cs}
                adjusted.append(item)
            else:
                item = None
            if item is not None and ((prev is not None and cs < prev) or (nxt is not None and cs > nxt)):
                item["crossed"] = True
        final[entry] = cs
        edit["cs"] = cs
        edit.pop("asked", None)
    edits = [
        {**edit, "before_cs": before[edit["entry"]]}
        for edit in sorted(requested, key=lambda item: item["entry"]) if edit["cs"] != before[edit["entry"]]
    ]
    if not edits:
        unchanged = _song_payload(pathlib.Path(song["report"]), song["report_data"])
        if adjusted:
            unchanged["adjusted_cs"] = adjusted
        return 200, unchanged
    auto = song["auto"]
    edits_path = song["edits"]
    auto_bytes = auto.read_bytes() if auto.is_file() else current
    bom, trailing, lines = _split_lrc(current)
    for edit in edits:
        tag = _format_tag(edit["cs"])
        for line_index in mapping[edit["entry"] - 1]:
            # A line he fixes here is resolved, so its `*` mark goes; every untouched line keeps its bytes.
            clean = _strip_line_marks(lines[line_index])
            lines[line_index] = tag + clean[_TAG_RE.match(clean).end():]
    updated = _join_lrc(bom, trailing, lines)
    after_sha = _sha256_bytes(updated)
    payload = _read_log(edits_path) or {
        "schema": 1, "report": pathlib.Path(song["report"]).name,
        "lrc_name": output.name, "auto_lrc_sha256": _sha256_bytes(auto_bytes), "edits": []
    }
    payload.setdefault("schema", 1)
    payload.setdefault("auto_lrc_sha256", _sha256_bytes(auto_bytes))
    payload.setdefault("edits", [])
    payload["report"] = pathlib.Path(song["report"]).name
    payload["lrc_name"] = output.name
    auto_lines = _split_lrc(_strip_marks(auto_bytes)[0])[2]
    if len(auto_lines) != len(lines):
        auto_lines = []

    def auto_time(entry):
        index = mapping[entry - 1][0]
        return _line_tag(auto_lines[index]) if index < len(auto_lines) else None

    payload["edits"].extend({
        "utc": datetime.now(timezone.utc).isoformat(),
        "entry": item["entry"], "auto_cs": auto_time(item["entry"]),
        "before_cs": item["before_cs"], "after_cs": item["cs"], "source": item["source"],
        "nudge_cs": item["nudge_cs"], "lrc_sha256_after": after_sha
    } for item in edits)
    payload["last_lrc_sha256"] = after_sha

    output_stage = None
    auto_stage = None
    log_stage = None
    auto_published = False
    try:
        output_stage = _stage_bytes(output, updated)
        if not auto.is_file():
            auto_stage = _stage_bytes(auto, current)
        log_stage = _stage_bytes(
            edits_path,
            (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
    except Exception:
        for stage_path in (output_stage, auto_stage, log_stage):
            if stage_path is not None:
                stage_path.unlink(missing_ok=True)
        return 500, {"error": "cannot prepare review save"}

    if auto_stage is not None:
        try:
            os.replace(auto_stage, auto)
            auto_published = True
            auto_stage = None
        except Exception:
            auto_stage.unlink(missing_ok=True)
            log_stage.unlink(missing_ok=True)
            output_stage.unlink(missing_ok=True)
            return 500, {"error": "cannot create review baseline"}

    replaced = False
    for attempt in range(3):
        try:
            os.replace(output_stage, output)
            replaced = True
            break
        except PermissionError:
            if attempt < 2:
                threading.Event().wait(0.2)
        except OSError:
            output_stage.unlink(missing_ok=True)
            log_stage.unlink(missing_ok=True)
            if auto_published:
                auto.unlink(missing_ok=True)
            return 500, {"error": "cannot publish LRC"}
    if not replaced:
        output_stage.unlink(missing_ok=True)
        log_stage.unlink(missing_ok=True)
        if auto_published:
            auto.unlink(missing_ok=True)
        return 423, {"error": "LRC 檔案被其他程式鎖住（請關閉播放器）再按儲存 (LRC file is locked; close the player, then Save again)"}
    try:
        os.replace(log_stage, edits_path)
    except OSError:
        log_stage.unlink(missing_ok=True)
        return 500, {"error": "LRC saved, edit log not written", "lrc_saved": True}
    result = _song_payload(pathlib.Path(song["report"]), song["report_data"])
    if adjusted:
        result["adjusted_cs"] = adjusted
    return 200, result


# ---------------------------------------------------------------------------
# Import panel: find lyrics beside a FLAC, write pasted lyrics, run alignment
# jobs through align-lrc.ps1 and expose their state. Alignment itself is not
# touched; this is the same command a drag onto Align LRC.bat would run.

_LYRIC_SUFFIXES = (".lyrics.lrc", ".lyrics.txt", ".txt", ".lyrics")  # same order as auto_lrc.find_lyrics


def _find_lyrics(audio):
    audio = pathlib.Path(audio)
    for suffix in _LYRIC_SUFFIXES:
        candidate = audio.with_name(audio.stem + suffix)
        if candidate.is_file():
            return candidate
    return None


def _sibling_audio(lyric_base):
    """First existing audio file named ``lyric_base`` + a supported extension."""
    return next((lyric_base.with_name(lyric_base.name + ext) for ext in runtime_paths.AUDIO_EXTENSIONS
                 if lyric_base.with_name(lyric_base.name + ext).is_file()), None)


def _expand_audio(paths):
    """Expand folders to their top-level audio files; keep audio files; drop the rest."""
    audios, skipped = [], []
    seen = set()
    for raw in paths or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = pathlib.Path(raw.strip().strip('"')).expanduser()
        if path.is_dir():
            found = sorted((c for c in path.iterdir() if c.is_file() and runtime_paths.is_audio_file(c)),
                           key=lambda c: c.name.casefold())
            if not found:
                skipped.append({"path": str(path), "reason": "no-flac-in-folder"})
        elif path.is_file() and runtime_paths.is_audio_file(path):
            found = [path]
        elif path.is_file():
            # A dropped lyric file stands for the audio beside it.
            sibling = next((s for s in (_sibling_audio(path.with_name(path.name[: -len(sfx)])) for sfx in _LYRIC_SUFFIXES
                                        if path.name.casefold().endswith(sfx)) if s), None)
            if sibling is None and path.suffix.casefold() == ".lrc":
                sibling = _sibling_audio(path.with_name(path.stem))
            found = [sibling] if sibling else []
            if not found:
                skipped.append({"path": str(path), "reason": "not-a-flac"})
        else:
            found = []
            skipped.append({"path": str(path), "reason": "not-found"})
        for audio in found:
            key = str(audio.resolve()).casefold()
            if key not in seen:
                seen.add(key)
                audios.append(audio)
    return audios, skipped


def _inspect_imports(paths):
    audios, skipped = _expand_audio(paths)
    items = []
    for audio in audios:
        lyrics = _find_lyrics(audio)
        items.append({
            "audio": str(audio),
            "name": audio.stem,
            "lyrics": str(lyrics) if lyrics else None,
            "existing_lrc": audio.with_suffix(".lrc").is_file(),
        })
    return {"items": items, "skipped": skipped}


def _write_lyrics(audio, text=None, source=None):
    """Save pasted lyrics (or copy a chosen lyric file) as <song>.lyrics.txt beside the FLAC."""
    audio = pathlib.Path(audio)
    if not audio.is_file() or not runtime_paths.is_audio_file(audio):
        return 400, {"error": "audio not found"}
    if _find_lyrics(audio) is not None:
        return 409, {"error": "lyrics already exist", "lyrics": str(_find_lyrics(audio))}
    if source:
        try:
            text = pathlib.Path(source).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            return 400, {"error": f"cannot read lyrics: {exc}"}
    if not isinstance(text, str) or not text.strip():
        return 400, {"error": "empty lyrics"}
    target = audio.with_name(audio.stem + ".lyrics.txt")
    try:
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text.replace("\r\n", "\n").rstrip("\n") + "\n")
    except OSError as exc:
        return 500, {"error": f"cannot write lyrics: {exc}"}
    return 200, {"lyrics": str(target)}


_PICK_SCRIPT = r"""
import json, sys
try:
    import tkinter
    from tkinter import filedialog
except Exception as exc:
    print(json.dumps({"error": "no-tkinter", "detail": str(exc)})); sys.exit(0)
kind = sys.argv[1]
root = tkinter.Tk(); root.withdraw(); root.attributes("-topmost", True)
if kind == "folder":
    picked = filedialog.askdirectory(title="選擇含 FLAC 的資料夾")
    paths = [picked] if picked else []
elif kind == "lyrics":
    paths = list(filedialog.askopenfilenames(title="選擇歌詞", filetypes=[("Lyrics", "*.txt *.lrc"), ("All", "*.*")]))
else:
    paths = list(filedialog.askopenfilenames(title="選擇音檔", filetypes=[("Audio", "*.flac *.m4a"), ("All", "*.*")]))
root.destroy()
print(json.dumps({"paths": paths}))
"""


def _pick_paths(kind):
    """Open the native file dialog on this PC and return the chosen paths."""
    try:
        result = subprocess.run([sys.executable, "-c", _PICK_SCRIPT, kind], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=600)
        payload = json.loads(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else {}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return 501, {"error": "dialog-unavailable", "detail": str(exc)}
    if payload.get("error"):
        return 501, {"error": "dialog-unavailable", "detail": payload.get("detail", payload["error"])}
    return 200, {"paths": payload.get("paths", [])}


def _align_command(audio, overwrite, report_dir):
    """The command a drag onto Align LRC.bat would run for one FLAC."""
    wrapper = runtime_paths.project_root() / "align-lrc.ps1"
    command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(wrapper),
               "-Paths", str(audio), "-ReportDirectory", str(report_dir)]
    if overwrite:
        command.append("-Overwrite")
    return command


_STAGE_RE = re.compile(r"^(LRC tools:.*|\[\d+/\d+\].*|.*reviewer.*)$", re.IGNORECASE)


def _stage_of(line):
    line = line.strip()
    if not line or not _STAGE_RE.match(line):
        return None
    return line[:120]


def _failure_reason(log_path, exit_code):
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in reversed(lines):
        if line.strip().upper().startswith("ERROR"):
            return line.strip()[:300]
    return f"exit {exit_code}"


def _jobs_dir(report_dir):
    report_dir = pathlib.Path(report_dir)
    if report_dir.name.casefold() == "reports":
        return report_dir.parent / "jobs"
    return report_dir / "_jobs"


class JobRunner:
    """Runs alignment jobs one song at a time and remembers their state on disk."""

    def __init__(self, store, report_dir):
        self.store = store
        self.report_dir = pathlib.Path(report_dir)
        self.jobs_dir = _jobs_dir(self.report_dir)
        self.lock = threading.RLock()
        self.jobs = []
        self.current = None  # (job id, item index, Popen)
        self.cancelled = set()
        self._worker = None
        self._load_saved()

    # -- persistence -------------------------------------------------------
    def _load_saved(self):
        try:
            files = sorted(self.jobs_dir.glob("*.json"))
        except OSError:
            files = []
        for path in files[-20:]:
            try:
                job = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                continue
            if isinstance(job, dict) and isinstance(job.get("items"), list):
                for item in job["items"]:
                    if item.get("status") in ("queued", "running"):
                        item["status"] = "interrupted"  # the server was restarted; nothing auto-resumes
                        item["error"] = "server restarted"
                self.jobs.append(job)

    def _save(self, job):
        try:
            self.jobs_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.jobs_dir / (job["id"] + ".json.tmp")
            tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.jobs_dir / (job["id"] + ".json"))
        except OSError:
            pass

    # -- public ------------------------------------------------------------
    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.jobs))

    def queued_audio(self):
        with self.lock:
            return {item["audio"].casefold() for job in self.jobs for item in job["items"] if item["status"] in ("queued", "running")}

    def enqueue(self, items):
        prepared, rejected = [], []
        busy = self.queued_audio()
        for raw in items or []:
            if not isinstance(raw, dict):
                continue
            audio = pathlib.Path(str(raw.get("audio", ""))).expanduser()
            if not audio.is_file() or not runtime_paths.is_audio_file(audio):
                rejected.append({"audio": str(audio), "reason": "audio not found"})
                continue
            if str(audio).casefold() in busy:
                rejected.append({"audio": str(audio), "reason": "already queued"})
                continue
            if _find_lyrics(audio) is None:
                rejected.append({"audio": str(audio), "reason": "缺歌詞"})
                continue
            busy.add(str(audio).casefold())
            prepared.append({"audio": str(audio), "name": audio.stem, "overwrite": bool(raw.get("overwrite")),
                             "status": "queued", "stage": "等待中", "exit_code": None, "error": None,
                             "report": None, "log": None, "started": None, "finished": None})
        if not prepared:
            return 400, {"error": "nothing to run", "rejected": rejected}
        job = {"id": datetime.now().strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(2),
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "items": prepared, "rejected": rejected}
        with self.lock:
            self.jobs.append(job)
            self._save(job)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run_loop, name="lrc-import-jobs", daemon=True)
                self._worker.start()
        return 200, {"job": job["id"], "items": len(prepared), "rejected": rejected}

    def cancel(self, job_id):
        with self.lock:
            job = next((j for j in self.jobs if j["id"] == job_id), None)
            if job is None:
                return 404, {"error": "no such job"}
            self.cancelled.add(job_id)
            for item in job["items"]:
                if item["status"] == "queued":
                    item["status"] = "cancelled"
                    item["stage"] = "已取消"
            proc = self.current[2] if self.current and self.current[0] == job_id else None
            self._save(job)
        if proc is not None:
            try:
                proc.terminate()
            except OSError:
                pass
        return 200, {"job": job_id}

    # -- worker ------------------------------------------------------------
    def _next_item(self):
        with self.lock:
            for job in self.jobs:
                for index, item in enumerate(job["items"]):
                    if item["status"] == "queued":
                        return job, index
        return None, None

    def _run_loop(self):
        while True:
            job, index = self._next_item()
            if job is None:
                with self.lock:
                    self._worker = None
                return
            self._run_item(job, index)

    def _run_item(self, job, index):
        item = job["items"][index]
        audio = pathlib.Path(item["audio"])
        log_dir = self.jobs_dir / job["id"]
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        log_path = log_dir / (audio.stem + ".log")
        started = time.time()
        with self.lock:
            item.update({"status": "running", "stage": "對齊中", "log": str(log_path),
                         "started": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            self._save(job)
        env = dict(os.environ, LRC_NO_REVIEW="1", LRC_NO_PAUSE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        command = _align_command(audio, item["overwrite"], self.report_dir)
        exit_code = None
        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as log:
                proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace", env=env, cwd=str(runtime_paths.project_root()))
                with self.lock:
                    self.current = (job["id"], index, proc)
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    stage = _stage_of(line)
                    if stage:
                        with self.lock:
                            item["stage"] = stage
                exit_code = proc.wait()
        except OSError as exc:
            exit_code = -1
            try:
                log_path.write_text(f"ERROR: cannot start alignment: {exc}\n", encoding="utf-8")
            except OSError:
                pass
        with self.lock:
            self.current = None
            item["exit_code"] = exit_code
            item["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if job["id"] in self.cancelled and item["status"] == "running" and exit_code != 0:
                item.update({"status": "cancelled", "stage": "已取消"})
            else:
                report = self._report_for(audio, started) if exit_code == 0 else None
                if exit_code == 0 and report is not None:
                    item.update({"status": "done", "stage": "完成", "report": str(report)})
                else:
                    item.update({"status": "failed", "stage": "失敗",
                                 "error": _failure_reason(log_path, exit_code) if exit_code != 0 else "report not written"})
            self._save(job)
        if item["status"] == "done":
            self.store.add_target(audio)

    def _report_for(self, audio, started):
        target = str(audio).casefold()
        best = None
        try:
            candidates = list(self.report_dir.glob("*.align-report.json"))
        except OSError:
            candidates = []
        for report in candidates:
            try:
                if report.stat().st_mtime < started - 5:
                    continue
                data = _load_report(report)
            except OSError:
                continue
            if data is not None and str(data.get("audio_path", "")).casefold() == target:
                if best is None or report.stat().st_mtime > best.stat().st_mtime:
                    best = report
        return best


_OWN_LOGS = {"export.log", "export-log.jsonl"}  # written by the server itself; never a reason to refresh


class ReviewStore:
    def __init__(self, report_dir, targets, read_only=False, scan_report_dir=True):
        self.report_dir = pathlib.Path(report_dir).expanduser().resolve()
        self.targets = list(targets)
        self.scan_report_dir = bool(scan_report_dir)
        self.read_only = bool(read_only)
        self.lock = threading.RLock()
        self.songs = []
        self.excluded = []
        self.export_status = {}
        self._status_lock = threading.Lock()  # guards export_status only, never the store lock: /api/status must answer during an export
        self.reload_progress = {"active": False, "done": 0, "total": 0}
        self.file_stats = {}
        self.generation = 0
        self.reload()
        self._remember()

    def _stats(self):
        """Every file the review depends on -> (size, mtime_ns); None marks a missing or unreadable one."""
        seen = {}

        def stat(path):
            try:
                info = path.stat()
                seen[str(path)] = (info.st_size, info.st_mtime_ns)
            except (OSError, ValueError):
                seen[str(path)] = None

        def listing(folder, suffixes=None):
            try:
                for child in sorted(folder.iterdir(), key=lambda item: item.name):
                    if child.is_file() and (suffixes is None or child.name.endswith(suffixes)) and child.name not in _OWN_LOGS:
                        stat(child)
            except OSError:
                seen[str(folder)] = "unreadable"

        listing(self.report_dir, (".align-report.json", ".review.tsv"))
        folders = set()
        for raw in self.targets:
            path = pathlib.Path(raw).expanduser()
            if path.is_dir():
                folders.add(path)
            else:
                stat(path)
        for song in self.songs:
            for key in ("audio_path", "output_path", "ctc_audio_path"):
                value = song.get(key)
                if value:
                    stat(pathlib.Path(value))
                    folders.add(pathlib.Path(value).parent)
        for folder in sorted(folders, key=str):
            listing(folder)
        return seen

    def _signature(self):
        """Cheap fingerprint of every file the review depends on (names, sizes, mtimes)."""
        return hashlib.sha256(repr(sorted(self._stats().items(), key=lambda item: item[0])).encode("utf-8", "replace")).hexdigest()

    def _remember(self):
        self.file_stats = self._stats()
        self.version = self._digest(self.file_stats)

    @staticmethod
    def _digest(stats):
        return hashlib.sha256(repr(sorted(stats.items(), key=lambda item: item[0])).encode("utf-8", "replace")).hexdigest()

    def is_current(self):
        with self.lock:
            return self._signature() == self.version

    def add_target(self, path):
        """A song aligned from the import panel joins the review list without a restart."""
        with self.lock:
            value = str(path)
            if not self.scan_report_dir and value.casefold() not in {str(t).casefold() for t in self.targets}:
                self.targets.append(value)
            self.reload()
            self._remember()
            self.generation += 1

    def _owners(self):
        """File path -> set of indexes of the songs that own it (its report, LRC, audio or CTC audio); songs may share a file."""
        owners = {}
        for index, song in enumerate(self.songs):
            for key in ("report", "audio_path", "output_path", "ctc_audio_path"):
                if song.get(key):
                    owners.setdefault(str(song[key]), set()).add(index)
        return owners

    def refresh_if_changed(self):
        """Refresh what changed on disk: one song when only its own files moved, everything when reports came or went."""
        with self.lock:
            stats = self._stats()
            if stats == self.file_stats:
                return self.generation
            changed = {key for key in stats.keys() | self.file_stats.keys() if stats.get(key, "-") != self.file_stats.get(key, "-")}
            owners = self._owners()
            if any(key not in owners for key in changed):
                self.reload()
            else:
                for index in sorted(set().union(*(owners[key] for key in changed))):
                    self.song(index)
            self._remember()
            self.generation += 1
            return self.generation

    def refresh_song(self, index, own_extra=()):
        """Re-read one song after our own change (an export) and adopt the new file stats of that song's own files
        (plus ``own_extra``, e.g. the archived LRC) as the baseline, so the /api/version poll does not reload for them.
        Every other file keeps its old baseline: a change made meanwhile is picked up by the next refresh_if_changed."""
        with self.lock:
            own = set(map(str, own_extra))
            own.update(self._own_files(index))
            self.song(index)
            own.update(self._own_files(index))
            # Songs sharing any of these files (e.g. one FLAC) see the export too: refresh every owner.
            owners = self._owners()
            co_owners = set().union(*(owners.get(key, set()) for key in own)) - {index}
            for other in sorted(co_owners):
                self.song(other)
            for other in co_owners:
                own.update(self._own_files(other))
            stats = self._stats()
            # Anything else that moved on disk since the baseline is not ours: the export answer says so, so the page refreshes.
            self.other_changes = any(stats.get(key, "-") != self.file_stats.get(key, "-") for key in stats.keys() | self.file_stats.keys() if key not in own)
            for key in own:
                if key in stats:
                    self.file_stats[key] = stats[key]
                else:
                    self.file_stats.pop(key, None)
            self.version = self._digest(self.file_stats)
            self.generation += 1
            return self.generation

    def _own_files(self, index):
        song = self.songs[index] if 0 <= index < len(self.songs) else {}
        return {str(song[key]) for key in ("report", "audio_path", "output_path", "ctc_audio_path") if song.get(key)}

    def update_export_status(self, **fields):
        with self._status_lock:
            self.export_status = {**self.export_status, **fields}  # replaced, never mutated in place

    def status(self):
        """Progress for a page that is waiting on the store lock; takes only the small status lock."""
        with self._status_lock:
            export = dict(self.export_status)
        if export.get("active") and export.get("started") is not None:
            export["elapsed_ms"] = round((time.monotonic() - export["started"]) * 1000)
        export.pop("started", None)
        return {"export": export, "reload": dict(self.reload_progress), "generation": self.generation}

    def reload(self):
        self.reload_progress = {"active": True, "done": 0, "total": 0}
        try:
            self._reload()
        finally:
            self.reload_progress = {"active": False, "done": self.reload_progress["total"], "total": self.reload_progress["total"]}

    def _reload(self):
        self.songs = []
        anchors = {}
        parsed = {}
        reports, self.excluded = _report_files(
            self.targets, self.report_dir, scan_report_dir=self.scan_report_dir, anchors=anchors, cache=parsed
        )
        self.reload_progress["total"] = len(reports)
        for path in reports:
            try:
                # Discovery already parsed this report; only re-read it if that parse failed.
                report = parsed.get(path)
                if not isinstance(report, dict):
                    report = json.loads(path.read_text(encoding="utf-8-sig"))
                if isinstance(report, dict):
                    self.songs.append(_song_payload(path, report, anchors.get(path)))
                else:
                    raise ValueError("report must be a JSON object")
            except Exception as exc:
                self.songs.append(_song_stub(path, exc))
            self.reload_progress["done"] = len(self.songs)

    def song(self, index):
        with self.lock:
            if index < 0 or index >= len(self.songs):
                return None
            current = self.songs[index]
            report_path = pathlib.Path(current["report"])
            try:
                report = json.loads(report_path.read_text(encoding="utf-8-sig"))
                if isinstance(report, dict):
                    current = _song_payload(report_path, report, current.get("anchor"))
                    self.songs[index] = current
            except Exception as exc:
                current = _song_stub(report_path, exc)
                self.songs[index] = current
            return current

    def cached_song(self, index):
        with self.lock:
            if index < 0 or index >= len(self.songs):
                return None
            return self.songs[index]


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
                payload = [{"index": index, "song": song["song"], "report": song["report"], "rows": len(song["rows"]), "review_rows": song["review_rows"], "read_only": song["read_only"], "read_only_reason": song["read_only_reason"], "entry_model": song.get("entry_model"), "manual_edited": song.get("manual_edited", False), "load_error": song.get("load_error"), "song_risk": song.get("song_risk"), "server_read_only": self.server.store.read_only} for index, song in enumerate(self.server.store.songs)]
            self._json(200, payload)
            return
        if parsed.path == "/api/version":
            if not self._token_header_ok():
                self._send(403, b"forbidden\n", "text/plain; charset=utf-8")
                return
            self._json(200, {"generation": self.server.store.refresh_if_changed(), "port_notice": getattr(self.server, "port_notice", "")})
            return
        if parsed.path == "/api/status":
            if not self._token_header_ok():
                self._send(403, b"forbidden\n", "text/plain; charset=utf-8")
                return
            self._json(200, self.server.store.status())  # lock-free: answers while a reload or export holds the store lock
            return
        if parsed.path == "/api/jobs":
            if not self._token_header_ok():
                self._send(403, b"forbidden\n", "text/plain; charset=utf-8")
                return
            runner = getattr(self.server, "jobs", None)
            self._json(200, {"import_enabled": runner is not None and not self.server.store.read_only,
                             "jobs": runner.snapshot() if runner is not None else []})
            return
        if parsed.path == "/api/excluded":
            if not self._token_header_ok():
                self._send(403, b"forbidden\n", "text/plain; charset=utf-8")
                return
            with self.server.store.lock:
                payload = {"excluded_files": [path.name for path in self.server.store.excluded]}
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
        match = re.fullmatch(r"/api/song/(\d+)/peaks", parsed.path)
        if match:
            self._serve_peaks(int(match.group(1)), parsed.query)
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

    _IMPORT_POSTS = ("/api/pick", "/api/import/inspect", "/api/import/lyrics", "/api/jobs")

    def do_POST(self):
        if not self._check_host():
            return
        parsed = urllib.parse.urlsplit(self.path)
        match = re.fullmatch(r"/api/song/(\d+)/(save|decision|export)", parsed.path)
        cancel = re.fullmatch(r"/api/jobs/([A-Za-z0-9_]+)/cancel", parsed.path)
        if not match and not cancel and parsed.path not in self._IMPORT_POSTS:
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
        if not match:
            self._handle_import(parsed.path, cancel, body)
            return
        with self.server.store.lock:
            song = self.server.store.song(int(match.group(1)))
            if song is None or self.server.store.read_only:
                self._json(403, {"error": "read-only"})
                return
            if match.group(2) == "decision":
                self._json(*_record_decision(song, body))
                return
            if match.group(2) == "export":
                store = self.server.store
                try:
                    status, result = _export_song(song, body, store.report_dir, store)
                except BaseException:
                    store.update_export_status(active=False)
                    raise
                if status == 200 and not body.get("check"):
                    work_dir, song_name, archive = result.pop("_log")
                    store.update_export_status(step="store_refresh")
                    step = time.perf_counter()
                    store.refresh_song(int(match.group(1)), [archive["to"]] if archive.get("to") else [])  # this song only, never a full reload
                    result["timings"]["store_refresh"] = round((time.perf_counter() - step) * 1000)
                    result["timings"]["total"] = round((time.monotonic() - (store.export_status.get("started") or time.monotonic())) * 1000)
                    result["generation"] = store.generation
                    if getattr(store, "other_changes", False):
                        result["other_changes"] = True
                    _export_log(work_dir, song_name, "ok", archive, "", result["timings"], 0)
                    store.update_export_status(active=False)
                self._json(status, result)
                return
            status, result = _save_song(song, body)
            self._json(status, _public_song(result, self.server.store.read_only) if status == 200 else result)

    def _handle_import(self, path, cancel, body):
        runner = getattr(self.server, "jobs", None)
        if runner is None or self.server.store.read_only:
            self._json(403, {"error": "import disabled"})
            return
        if path == "/api/pick":
            status, result = _pick_paths(str(body.get("kind", "audio")))
        elif path == "/api/import/inspect":
            status, result = 200, _inspect_imports(body.get("paths"))
        elif path == "/api/import/lyrics":
            status, result = _write_lyrics(str(body.get("audio", "")), body.get("text"), body.get("source"))
        elif path == "/api/jobs":
            status, result = runner.enqueue(body.get("items"))
        else:
            status, result = runner.cancel(cancel.group(1))
        self._json(status, result)

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
            mime = "audio/flac" if suffix == ".flac" else "audio/mp4" if suffix == ".m4a" else "audio/mpeg" if suffix == ".mp3" else "audio/wav" if suffix == ".wav" else "application/octet-stream"
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

    def _serve_peaks(self, index, query):
        if not self._token_query_ok(query):
            self._send(403, b"forbidden\n", "text/plain; charset=utf-8")
            return
        try:
            from_cs, to_cs, bins = _parse_peaks_query(query)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return
        song = self.server.store.cached_song(index)
        if song is None:
            self._json(404, {"error": "not found"})
            return
        path = pathlib.Path(song.get("ctc_audio_path", "")).expanduser()
        if not path.is_file():
            self._json(404, {"error": "stem not found"})
            return
        try:
            payload = _read_stem_peaks(path, from_cs, to_cs, bins)
        except ValueError as exc:
            self._json(415, {"error": str(exc)})
            return
        except OSError:
            self._json(404, {"error": "stem not found"})
            return
        self._json(200, payload)



def _read_targets_file(path):
    try:
        lines = pathlib.Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        print(f"WARN: review targets file unavailable: {path} ({exc})", file=sys.stderr)
        return []
    targets = []
    seen = set()
    for line in lines:
        value = line.strip()
        if not value:
            continue
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            targets.append(value)
    return targets


def _warn_missing_targets(targets):
    seen = set()
    for raw in targets:
        value = str(raw).strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        try:
            exists = pathlib.Path(value).expanduser().exists()
        except (OSError, ValueError):
            exists = False
        if not exists:
            print(f"WARN: review target not found: {value}", file=sys.stderr)


DEFAULT_REVIEW_PORT = 58400
PORT_FALLBACK_SPAN = 20


def default_port():
    """The review page keeps its drafts in localStorage, which is per origin: a stable port is what lets them survive a restart."""
    try:
        return int(os.environ.get("LRC_REVIEW_PORT", "") or DEFAULT_REVIEW_PORT)
    except ValueError:
        return DEFAULT_REVIEW_PORT


class ReviewHTTPServer(http.server.ThreadingHTTPServer):
    """A busy port must really fail: on Windows SO_REUSEADDR lets a second server bind a port the first still answers on."""
    allow_reuse_address = False

    def server_bind(self):
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def port_busy_notice(wanted, used):
    return f"{wanted} 已被佔用（可能有另一個 Review LRC 還開著），改用 {used}；此埠的暫存草稿不會跨埠號"


def bind_server(port, handler=None):
    """Bind 127.0.0.1:port; when it is busy, try the next PORT_FALLBACK_SPAN ports, then an OS-chosen one. Port 0 (tests) keeps the OS-chosen port.
    The returned server carries port_notice when it is not on the port that was asked for."""
    handler = handler or ReviewHandler
    port = int(port)
    if port == 0:
        server = ReviewHTTPServer(("127.0.0.1", 0), handler)
        server.port_notice = ""
        return server
    for candidate in range(port, port + PORT_FALLBACK_SPAN + 1):
        try:
            server = ReviewHTTPServer(("127.0.0.1", candidate), handler)
        except OSError:
            continue
        server.port_notice = port_busy_notice(port, candidate) if candidate != port else ""
        return server
    server = ReviewHTTPServer(("127.0.0.1", 0), handler)
    server.port_notice = port_busy_notice(port, server.server_port)
    return server


def run_server(targets, report_dir=None, port=0, no_browser=False, read_only=False, dump_summary=False, targets_file=None):
    report_dir = pathlib.Path(report_dir).expanduser().resolve() if report_dir else runtime_paths.reports_dir()
    explicit_targets = bool(targets) or targets_file is not None
    target_list = list(targets or [])
    if targets_file is not None:
        target_list.extend(_read_targets_file(targets_file))
    _warn_missing_targets(target_list)
    store = ReviewStore(report_dir, target_list, read_only=read_only, scan_report_dir=not explicit_targets)
    if dump_summary:
        for song in store.songs:
            share = sum(len({item.get("cs") for item in row.get("candidates", [])}) >= 2 for row in song["rows"])
            print(json.dumps({"song": song["song"], "rows": len(song["rows"]), "review_rows": song["review_rows"], "read_only": song["read_only"], "reason": song["read_only_reason"], "entry_model": song.get("entry_model"), "has_flac": _path_is_file(song["audio_path"]), "has_stem": _path_is_file(song["ctc_audio_path"]), "share_rows": share}, ensure_ascii=True))
        return None
    server = bind_server(port)
    server.store = store
    server.jobs = None if (read_only or os.environ.get("LRC_NO_REVIEW") == "1") else JobRunner(store, report_dir)
    server.token = secrets.token_urlsafe(32)
    if server.port_notice:
        print(server.port_notice, file=sys.stderr, flush=True)
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
    parser.add_argument("--port", type=int, default=default_port())
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--dump-summary", action="store_true")
    parser.add_argument("--targets-file", type=pathlib.Path)
    args = parser.parse_args()
    return run_server(args.targets, args.report_dir, args.port, args.no_browser, args.read_only, args.dump_summary, args.targets_file)


if __name__ == "__main__":
    main()
