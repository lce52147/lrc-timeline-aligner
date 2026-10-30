"""Source furigana for every formal report, and the ASR check of the romaji that is shown.

`fetch`  UtaTen ruby (what is SUNG, 時間 -> とき) matched line by line onto each song's lyric lines, written as
         `<report stem>.furigana.json` (scripts/lrc_reading.py reads it) and cached under --cache.
`audit`  per song: lines / from furigana / dictionary / analyzer / flagged by ASR, the flagged list, and how many
         CTC alignment targets would differ from the furigana reading; writes `.romaji-asr.json` sidecars and
         `romaji_audit.md`.  Uses only cached whisper.cpp output (outputs/cache/model-outputs/raw-asr).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lyricsee"))

import lrc_reading  # noqa: E402

REPORTS = ROOT / "outputs" / "reports"
RAW_ASR = ROOT / "outputs" / "cache" / "model-outputs" / "raw-asr"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.8",
    "Accept-Encoding": "identity",
    "Referer": "https://utaten.com/",
}
_LRC_LINE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
MIN_PAGE_MATCH = 0.6
# A line is flagged when the ASR heard something close to it but not the same: far enough that a word was
# probably sung differently (>= MIN), close enough that this is the same line and not a missed or hallucinated
# one (<= MAX), and the best-matching ASR window is about as long as the reading.
ASR_FLAG_DISTANCE = 0.2
ASR_FLAG_MAX_DISTANCE = 0.6
ASR_MIN_COVERAGE = 0.7
ASR_PAD_SECONDS = 4.0


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = lrc_reading._to_katakana(text)
    return re.sub(r"[^\w]|_", "", text)


def report_stem(path: pathlib.Path) -> str:
    return path.name.replace(".align-report.json", "")


def formal_reports():
    return sorted(REPORTS.glob("*.align-report.json"))


def song_lrc(report: dict) -> pathlib.Path | None:
    out = pathlib.Path(report.get("output_path") or "")
    audio = pathlib.Path(report.get("audio_path") or "")
    for cand in (out, audio.with_suffix(".lrc") if audio.name else None):
        if cand and cand.name and cand.is_file():
            return cand
    return None


def lyric_rows(lrc: pathlib.Path):
    """[(seconds, original line)]: the first line of each timestamp group (the rest is the translation)."""
    rows = []
    last = None
    for raw in lrc.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = _LRC_LINE.match(raw.strip())
        if not match:
            continue
        seconds = int(match.group(1)) * 60 + float(match.group(2))
        text = match.group(3).strip()
        if not text:
            continue
        if last is not None and abs(seconds - last) < 1e-6:
            continue
        last = seconds
        rows.append((seconds, text))
    return rows


def titles_for(report: dict, stem: str):
    out = []
    audio = pathlib.Path(report.get("audio_path") or "")
    try:
        import mutagen

        tags = mutagen.File(str(audio), easy=True) if audio.is_file() else None
        if tags and tags.get("title"):
            out.append(str(tags["title"][0]))
    except Exception:
        pass
    base = re.sub(r"^\s*\d+[\s.\-_]*", "", audio.stem or stem.split("--")[0])
    if " - " in base:
        base = base.rsplit(" - ", 1)[1]
    out.append(base)
    more = []
    for title in out:
        short = re.sub(r"\s*[\(（\[【〈《].*?[\)）\]】〉》]\s*", " ", title).strip()
        short = re.split(r"\s+(?:-|―|〜|~)\s*", short)[0].strip()
        more.append(short)
    seen, result = set(), []
    for title in out + more:
        if title and title not in seen:
            seen.add(title)
            result.append(title)
    return result[:3]


def http_get(url: str, delay: float, retries: int = 2) -> str:
    last = None
    for attempt in range(retries + 1):
        time.sleep(delay)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=25) as response:
                data = response.read().decode("utf-8", errors="replace")
            if data:
                return data
            last = "empty body"
        except Exception as error:
            last = str(error)
            if "404" in last:
                break
        time.sleep(3 * (attempt + 1))
    raise OSError(last or "fetch failed")


def match_page(rows, page_lines):
    """{our line: segments} for lines equal (after normalization) to a page line; plus the match ratio."""
    by_text = {}
    for segments in page_lines:
        by_text.setdefault(norm("".join(surface for surface, _ in segments)), segments)
    found = {}
    wanted = 0
    for _, text in rows:
        if not lrc_reading.has_japanese(text) or not re.search("[ぁ-ゟァ-ヿ㐀-鿿]", text):
            continue
        wanted += 1
        key = norm(text)
        if key in by_text:
            found[lrc_reading.line_key(text)] = [list(item) for item in by_text[key]]
    return found, (len(found) / wanted if wanted else 0.0), wanted


def fetch_song(path: pathlib.Path, cache: pathlib.Path, delay: float, force: bool = False):
    from utaten_source import parse_utaten_ruby, parse_utaten_search, utaten_search_url

    report = json.loads(path.read_text(encoding="utf-8"))
    stem = report_stem(path)
    target = cache / (stem + ".furigana.json")
    if target.is_file() and not force:
        return json.loads(target.read_text(encoding="utf-8")).get("status", "cached")
    lrc = song_lrc(report)
    result = {"stem": stem, "source": "utaten", "status": "", "url": "", "matched": 0, "total": 0, "lines": {}}
    if lrc is None:
        result["status"] = "no-lrc"
    else:
        rows = lyric_rows(lrc)
        head = [norm(text)[:8] for _, text in rows[:6] if len(norm(text)) >= 4]
        best = (0.0, {}, "", 0)
        tried = 0
        error = ""
        for title in titles_for(report, stem):
            try:
                hits = parse_utaten_search(http_get(utaten_search_url(title), delay))
            except OSError as exc:
                error = str(exc)
                continue
            ranked = sorted(hits, key=lambda hit: -any(prefix and prefix in norm(hit.opening) for prefix in head))
            for hit in ranked[:3]:
                if tried >= 4:
                    break
                tried += 1
                try:
                    lines = parse_utaten_ruby(http_get(hit.url, delay))
                except (OSError, LookupError) as exc:
                    error = str(exc)
                    continue
                found, ratio, wanted = match_page(rows, lines)
                if ratio > best[0]:
                    best = (ratio, found, hit.url, wanted)
                if ratio >= 0.85:
                    break
            if best[0] >= 0.85:
                break
        ratio, found, url, wanted = best
        result["total"] = wanted
        if ratio >= MIN_PAGE_MATCH:
            result.update(status="ok", url=url, matched=len(found), lines=found)
        else:
            result["status"] = "no-match" if not error or tried else "error: " + error
            result["best_ratio"] = round(ratio, 3)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result["status"]


def cmd_fetch(args):
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    reports = formal_reports()
    log = open(cache / "fetch.log", "a", encoding="utf-8")
    for index, path in enumerate(reports, 1):
        if args.limit and index > args.limit:
            break
        try:
            status = fetch_song(path, cache, args.delay, args.force)
        except Exception as error:  # keep going: one bad song must not stop the run
            status = "exception: " + str(error)[:120]
        log.write(f"{index}/{len(reports)}\t{report_stem(path)}\t{status}\n")
        log.flush()
    print("fetch done")


def install_sidecars(cache: pathlib.Path, target: pathlib.Path):
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in cache.glob("*.furigana.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "ok" and data.get("lines"):
            (target / path.name).write_text(json.dumps({"source": data["source"], "url": data["url"], "lines": data["lines"]}, ensure_ascii=False, indent=1), encoding="utf-8")
            count += 1
    return count


# ---- ASR check -----------------------------------------------------------------------------------------

def build_asr_index(cache_path: pathlib.Path):
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    index: dict = {}
    pattern = re.compile(rb'"audio_sha256":\s*"([0-9a-f]{64})"')
    for path in RAW_ASR.glob("*.json"):
        try:
            with open(path, "rb") as handle:
                head = handle.read(1500)
        except OSError:
            continue
        match = pattern.search(head)
        if match and b'"language": "ja"' in head:
            index.setdefault(match.group(1).decode(), []).append(path.name)
    cache_path.write_text(json.dumps(index), encoding="utf-8")
    return index


def asr_segments(report_path: pathlib.Path, index: dict):
    """whisper.cpp segments cached for this song.  The cache is keyed by the decoded 16 kHz WAV, whose hash the
    report keeps in its tail-evidence provenance (audio_revision), so any hash of the report that is a cache key counts."""
    text = report_path.read_text(encoding="utf-8")
    names = [name for sha in dict.fromkeys(re.findall(r"[0-9a-f]{64}", text)) for name in index.get(sha, [])]
    segments = []
    for name in names[:2]:
        try:
            payload = json.loads((RAW_ASR / name).read_text(encoding="utf-8")).get("payload")
        except (OSError, ValueError):
            continue
        for item in payload if isinstance(payload, list) else []:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                segments.append((float(item.get("start") or 0), float(item.get("end") or 0), item["text"]))
        if segments:
            break
    return segments


def semi_global(needle: str, hay: str):
    """(edit distance, start, end) of the substring of ``hay`` closest to ``needle`` (free start and end in hay)."""
    n, m = len(needle), len(hay)
    prev = [0] * (m + 1)
    starts = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        cur_start = [starts[0]] + [0] * m
        for j in range(1, m + 1):
            best = prev[j - 1] + (needle[i - 1] != hay[j - 1])
            start = starts[j - 1]
            if prev[j] + 1 < best:
                best, start = prev[j] + 1, starts[j]
            if cur[j - 1] + 1 < best:
                best, start = cur[j - 1] + 1, cur_start[j - 1]
            cur[j], cur_start[j] = best, start
        prev, starts = cur, cur_start
    j = min(range(m + 1), key=lambda k: prev[k])
    return prev[j], starts[j], j


def asr_check(text: str, segments_kana, lo: float, hi: float, reading: str):
    """(distance/len, heard kana window, coverage) of ``reading`` against the ASR text overlapping [lo, hi]."""
    heard = "".join(kana for start, end, kana in segments_kana if end > lo and start < hi)
    if not heard or not reading:
        return None
    distance, start, end = semi_global(reading, heard)
    window = heard[start:end]
    return distance / len(reading), window, len(window) / len(reading)


def is_flagged(distance: float, coverage: float) -> bool:
    return ASR_FLAG_DISTANCE <= distance <= ASR_FLAG_MAX_DISTANCE and coverage >= ASR_MIN_COVERAGE


def cmd_audit(args):
    cache = pathlib.Path(args.cache)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = build_asr_index(out_dir / "raw_asr_index.json")
    sidecar_dir = lrc_reading.furigana_dir()
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    totals = dict(songs=0, jp_songs=0, lines=0, kanji_lines=0, furigana=0, partial=0, dictionary=0, auto=0, kana_only=0, flagged=0,
                  asr_checked=0, ctc_diff=0, ctc_total=0, no_lrc=0, no_page=0, no_asr=0)
    song_rows, flagged_rows, ctc_rows = [], [], []
    for path in formal_reports():
        report = json.loads(path.read_text(encoding="utf-8"))
        stem = report_stem(path)
        totals["songs"] += 1
        lrc = song_lrc(report)
        if lrc is None:
            totals["no_lrc"] += 1
            continue
        # rows with kana only (a kanji-only line has no translation to tell it from); the 00:00 "artist - title" header is no lyric
        rows = [(t, x) for t, x in lyric_rows(lrc) if re.search("[ぁ-ゟァ-ヿ]", x) and not (t < 0.01 and " - " in x)]
        if not rows:
            continue
        totals["jp_songs"] += 1
        fetched = {}
        fpath = cache / (stem + ".furigana.json")
        if fpath.is_file():
            data = json.loads(fpath.read_text(encoding="utf-8"))
            fetched = {lrc_reading.line_key(k): lrc_reading.clean_segments(v) for k, v in (data.get("lines") or {}).items()} if data.get("status") == "ok" else {}
        if not fetched:
            totals["no_page"] += 1
        segs = asr_segments(path, index)
        if not segs:
            totals["no_asr"] += 1
        seg_kana = [(a, b, lrc_reading.reading_kana(t, use_dictionary=False)) for a, b, t in segs]
        stat = dict(stem=stem, lines=0, furigana=0, partial=0, dictionary=0, auto=0, kana_only=0, flagged=0, ctc_diff=0, asr="yes" if segs else "no", page="yes" if fetched else "no")
        flags = {}
        for number, (seconds, text) in enumerate(rows):
            key = lrc_reading.line_key(text)
            kanji = lrc_reading.line_has_kanji(text)
            stat["lines"] += 1
            totals["lines"] += 1
            segments = fetched.get(key)
            if not kanji:
                stat["kana_only"] += 1
                totals["kana_only"] += 1
                continue
            totals["kanji_lines"] += 1
            source = lrc_reading.reading_source(text, fetched)
            if source == "furigana":
                stat["furigana"] += 1
                totals["furigana"] += 1
            else:
                if segments and any(r for _, r in segments):
                    stat["partial"] += 1
                    totals["partial"] += 1
                stat["dictionary" if source == "dict" else "auto"] += 1
                totals["dictionary" if source == "dict" else "auto"] += 1
            if segments and any(r for _, r in segments):
                totals["ctc_total"] += 1
                a = lrc_reading.romanize_compact(text, use_dictionary=False)
                b = lrc_reading.romanize_compact(text, use_dictionary=False, furigana=segments)
                if a != b:
                    stat["ctc_diff"] += 1
                    totals["ctc_diff"] += 1
                    ctc_rows.append((stem, number + 1, text, a, b))
            if segs:
                reading = lrc_reading.reading_kana(text, segments, True)
                nxt = rows[number + 1][0] if number + 1 < len(rows) else seconds + 8
                check = asr_check(text, seg_kana, seconds - ASR_PAD_SECONDS, min(nxt, seconds + 14) + ASR_PAD_SECONDS, reading)
                if check:
                    totals["asr_checked"] += 1
                    distance, window, coverage = check
                    if is_flagged(distance, coverage):
                        suggest = lrc_reading.kana_to_romaji(window)
                        current = lrc_reading.romanize_line(text, furigana=segments)
                        flags[key] = {"heard": window, "suggest": suggest, "distance": round(distance, 2)}
                        stat["flagged"] += 1
                        totals["flagged"] += 1
                        flagged_rows.append((stem, number + 1, text, current, window, suggest, round(distance, 2), source))
        side = lrc_reading.furigana_path(stem, ".romaji-asr.json")
        if flags:
            side.write_text(json.dumps({"flags": flags}, ensure_ascii=False, indent=1), encoding="utf-8")
        elif side.exists():
            side.unlink()
        song_rows.append(stat)
    installed = install_sidecars(cache, sidecar_dir)
    write_markdown(out_dir / "romaji_audit.md", totals, song_rows, flagged_rows, ctc_rows, installed)
    print(json.dumps(totals))


def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def write_markdown(path, totals, song_rows, flagged_rows, ctc_rows, installed):
    out = ["# romaji audit (BO)", ""]
    out += [
        f"- formal reports: {totals['songs']} (Japanese: {totals['jp_songs']}, no LRC found: {totals['no_lrc']})",
        f"- Japanese lines: {totals['lines']}; with kanji: {totals['kanji_lines']}; kana-only (reading is the text itself): {totals['kana_only']}",
        f"- kanji lines read from source furigana (標 歌詞): {totals['furigana']} = {pct(totals['furigana'], totals['kanji_lines'])} of kanji lines "
        f"(partial furigana, some kanji uncovered: {totals['partial']})",
        f"- kanji lines from dictionary: {totals['dictionary'] - 0}, analyzer: {totals['auto']} (these are the lines without full furigana; dictionary count includes partial)",
        f"- every Japanese line carries a source label: kanji lines manual/歌詞/字典/自動, kana-only lines 自動 (unambiguous)",
        f"- songs with a furigana page: {totals['jp_songs'] - totals['no_page']}/{totals['jp_songs']}; sidecars installed: {installed}",
        f"- ASR check (cached whisper.cpp only; songs without cached ASR: {totals['no_asr']}): kanji lines checked {totals['asr_checked']}, flagged 「讀音可能不同」: {totals['flagged']} "
        f"(rule: kana edit distance of the reading to its best-matching ASR window between {ASR_FLAG_DISTANCE} and {ASR_FLAG_MAX_DISTANCE} with window length >= {ASR_MIN_COVERAGE} of the reading, ASR searched +-{ASR_PAD_SECONDS:g}s around the line; "
        f"further away means the ASR missed the line (not flagged); ASR writes kanji by its own guess, so this is a hint, not proof)",
        f"- alignment impact (flag OFF, nothing changed): of {totals['ctc_total']} lines that have furigana, {totals['ctc_diff']} would get a different CTC target than the analyzer reading",
        "",
        "## per song",
        "",
        "| song | lines | furigana | dict | analyzer | kana-only | flagged | CTC differs | page | asr |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in song_rows:
        out.append(f"| {s['stem']} | {s['lines']} | {s['furigana']} | {s['dictionary']} | {s['auto']} | {s['kana_only']} | {s['flagged']} | {s['ctc_diff']} | {s['page']} | {s['asr']} |")
    out += ["", "## flagged lines (song, line no, current romaji, ASR-heard kana, suggested romaji)", "",
            "| song | line | lyric | current romaji | ASR heard | suggested romaji | dist | source |", "|---|---|---|---|---|---|---|---|"]
    for stem, number, text, current, heard, suggest, dist, source in sorted(flagged_rows, key=lambda r: (r[0], r[1])):
        out.append(f"| {stem} | {number} | {text} | {current} | {heard} | {suggest} | {dist} | {source} |")
    out += ["", "## CTC target differences (analyzer vs furigana reading; alignment still uses the analyzer)", "",
            "| song | line | lyric | analyzer target | furigana target |", "|---|---|---|---|---|"]
    for stem, number, text, a, b in ctc_rows[:400]:
        out.append(f"| {stem} | {number} | {text} | {a} | {b} |")
    if len(ctc_rows) > 400:
        out.append(f"| … | | {len(ctc_rows) - 400} more | | |")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--cache", default=r"F:\_RuntimePrompt\LRC15\BO\furigana")
    fetch.add_argument("--delay", type=float, default=1.5)
    fetch.add_argument("--limit", type=int, default=0)
    fetch.add_argument("--force", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("--cache", default=r"F:\_RuntimePrompt\LRC15\BO\furigana")
    audit.add_argument("--out", default=r"F:\_RuntimePrompt\LRC15\BO")
    args = parser.parse_args(argv)
    {"fetch": cmd_fetch, "audit": cmd_audit}[args.cmd](args)


if __name__ == "__main__":
    main()
