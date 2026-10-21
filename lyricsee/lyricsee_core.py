from __future__ import annotations

import html as html_module
import difflib
import unicodedata
import hashlib
import gzip
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, NamedTuple
from urllib.error import HTTPError
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen


class TrackIdentity(NamedTuple):
    title: str
    artist: str = ""
    album: str = ""
    duration: float | None = None


class TimedLine(NamedTuple):
    milliseconds: int
    text: str


class LyricPair(NamedTuple):
    original: str
    translation: str | None = None


class LyricCandidate:
    def __init__(
        self,
        *,
        source: str,
        url: str,
        title: str,
        artist: str,
        lines: list[LyricPair],
        discovered_urls: list[str] | None = None,
        google_search: str | None = None,
        translator: str | None = None,
    ) -> None:
        self.source = source
        self.url = url
        self.title = title
        self.artist = artist
        self.lines = lines
        self.discovered_urls = list(discovered_urls or [])
        self.google_search = google_search
        self.translator = translator


_LRC_TIMESTAMP = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")
_METADATA_PREFIXES = (
    "作词",
    "作詞",
    "作曲",
    "编曲",
    "編曲",
    "制作",
    "製作",
    "composer",
    "lyricist",
    "arranger",
)
_CREDIT_LINE_RE = re.compile(
    r"^(?:詞|词|曲|詞曲|词曲|作詞|作词|作曲|編曲|编曲|"
    r"作詞者|作词者|作曲者|原唱|翻唱|演唱|歌手|歌唱|"
    r"vocal|vocals|lyrics\s+by|music\s+by|written\s+by|"
    r"composer|lyricist|arranger|translator|translation)\s*[:：]\s*\S",
    flags=re.IGNORECASE,
)


def _is_credit_line(text: str) -> bool:
    line = text.strip()
    return line.casefold().startswith(_METADATA_PREFIXES) or bool(_CREDIT_LINE_RE.match(line))


def identity_from_filename(path: Path) -> TrackIdentity:
    stem = path.stem.strip()
    if " - " in stem:
        artist, title = (part.strip() for part in stem.split(" - ", 1))
        if artist and title:
            return TrackIdentity(title=title, artist=artist)
    return TrackIdentity(title=stem)


def _fraction_to_milliseconds(value: str | None) -> int:
    if not value:
        return 0
    if len(value) == 1:
        return int(value) * 100
    if len(value) == 2:
        return int(value) * 10
    return int(value[:3])


def parse_lrc(text: str) -> list[TimedLine]:
    result: list[TimedLine] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        matches = list(_LRC_TIMESTAMP.finditer(raw_line))
        if not matches:
            continue
        lyric = _LRC_TIMESTAMP.sub("", raw_line).strip()
        if not lyric or _is_credit_line(lyric):
            continue
        for match in matches:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            milliseconds = (minutes * 60 + seconds) * 1000
            milliseconds += _fraction_to_milliseconds(match.group(3))
            result.append(TimedLine(milliseconds, lyric))
    result.sort(key=lambda row: row.milliseconds)
    return result


def pair_timed_lyrics(
    original: Iterable[TimedLine],
    translated: Iterable[TimedLine],
) -> list[LyricPair]:
    translations: dict[int, str] = {}
    for row in translated:
        translations.setdefault(row.milliseconds, row.text)
    return [
        LyricPair(row.text, translations.get(row.milliseconds))
        for row in original
    ]


def translation_coverage(lines: Iterable[LyricPair]) -> float:
    materialized = list(lines)
    if not materialized:
        return 0.0
    translated = sum(bool(row.translation and row.translation.strip()) for row in materialized)
    return translated / len(materialized)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


_GOOGLE_HOSTS = {
    "google.com",
    "www.google.com",
    "accounts.google.com",
    "support.google.com",
    "maps.google.com",
}


def _google_target(href: str) -> str | None:
    href = html_module.unescape(href)
    if href.startswith("/url?"):
        query = parse_qs(urlparse(href).query)
        values = query.get("q") or query.get("url")
        href = unquote(values[0]) if values else ""
    if not href.startswith(("http://", "https://")):
        return None
    parsed = urlparse(href)
    host = parsed.hostname.casefold() if parsed.hostname else ""
    if host in _GOOGLE_HOSTS or host.endswith(".google.com"):
        return None
    return href


def parse_google_results(document: str) -> list[str]:
    parser = _LinkParser()
    parser.feed(document)
    results: list[str] = []
    seen: set[str] = set()
    for href in parser.hrefs:
        target = _google_target(href)
        if not target or target in seen:
            continue
        seen.add(target)
        results.append(target)
    return results


class _VisibleTextParser(HTMLParser):
    _BREAK_TAGS = {
        "br",
        "p",
        "div",
        "li",
        "article",
        "section",
        "h1",
        "h2",
        "h3",
        "h4",
        "tr",
    }
    _SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if lowered in self._SKIP_TAGS:
            self.skip_depth += 1
        elif lowered in self._BREAK_TAGS and not self.skip_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in self._SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif lowered in self._BREAK_TAGS and not self.skip_depth:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)


class _PageTitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_title = False
        self.in_heading = False
        self.title_parts: list[str] = []
        self.heading_parts: list[str] = []
        self.meta_title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.casefold()
        if lowered == "title":
            self.in_title = True
        elif lowered in {"h1", "h2"} and not self.heading_parts:
            self.in_heading = True
        elif lowered == "meta":
            attributes = {key.casefold(): value or "" for key, value in attrs}
            key = (attributes.get("property") or attributes.get("name") or "").casefold()
            if key in {"og:title", "twitter:title"} and attributes.get("content"):
                self.meta_title = attributes["content"]

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered == "title":
            self.in_title = False
        elif lowered in {"h1", "h2"}:
            self.in_heading = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        elif self.in_heading:
            self.heading_parts.append(data)

    def value(self) -> str:
        for value in (self.meta_title, "".join(self.title_parts), "".join(self.heading_parts)):
            value = re.sub(r"\s+", " ", html_module.unescape(value)).strip()
            if value:
                return value
        return ""


_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]{3,}")
_NON_LYRIC_WEB_RE = re.compile(
    r"https?://|www\.|©|all rights reserved|"
    r"巴哈姆特|最近閱覽|我的聊天列表|相關創作|開啟\s*app|下載\s*now!?\s*app|"
    r"gnn新聞|acg資料庫|電玩瘋|google站內查詢|新增加歌曲|漢字顯示振假名|歌詞跟隨|"
    r"唱歌學日語|marumaru|重覆這句|重複這句|"
    r"【(?:中日|日中)?歌詞(?:翻譯)?】|【歌詞翻譯】|"
    r"(?:^|[\s|｜・])(?:pv|mv|唄|作詞|作曲|編曲|母帶工程師|母带工程师|"
    r"混音工程師|混音工程师|錄音工程師|录音工程师|小提琴|和聲|和声|"
    r"mastering engineer|mixing engineer)\s*[:：]?|"
    r"\b(?:64|96|128|192|256)\s*kbps\b",
    flags=re.IGNORECASE,
)
_NON_LYRIC_WEB_EXACT = {
    "好友",
    "首頁",
    "會員",
    "追蹤",
    "相關創作",
    "開啟 app",
    "top",
    "end",
    "the end",
    "undefined",
    "純音樂，請欣賞",
    "純音樂,請欣賞",
    "纯音乐，请欣赏",
    "纯音乐,请欣赏",
}
_SOFT_WEB_HELPERS = {"重覆這句", "重複這句"}
_NON_LYRIC_SYMBOLS_RE = re.compile(r"^[♪♫♬♩\s]+$")


def _looks_like_original(text: str) -> bool:
    if _KANA_RE.search(text):
        return True
    if not _LATIN_RE.search(text):
        return False
    return len(_HAN_RE.findall(text)) <= 4


def _looks_like_chinese_translation(text: str) -> bool:
    return bool(_HAN_RE.search(text)) and not bool(_KANA_RE.search(text))


def _looks_like_non_lyric_web_text(text: str) -> bool:
    stripped = text.strip()
    return (
        stripped.casefold() in _NON_LYRIC_WEB_EXACT
        or bool(_NON_LYRIC_WEB_RE.search(stripped))
        or bool(_NON_LYRIC_SYMBOLS_RE.fullmatch(stripped))
        or stripped.startswith("#")
    )


def _sanitize_candidate_lines(
    lines: Iterable[LyricPair], identity: TrackIdentity
) -> list[LyricPair]:
    title = _normalized(identity.title)
    artists = {
        _normalized(value)
        for value in re.split(r"\s*[,、/&／]\s*", identity.artist)
        if value.strip()
    }
    title_artist_rows = {
        combined
        for artist in artists
        for combined in (title + artist, artist + title)
        if title and artist
    }
    result: list[LyricPair] = []
    for row in lines:
        original = row.original.strip()
        if not original or _is_credit_line(original) or _looks_like_non_lyric_web_text(original):
            continue
        if title_artist_rows and _normalized(original) in title_artist_rows:
            continue
        # Some sources insert a mistyped "song - artist" caption after credit
        # lines; exact string matching misses these false lyrics.
        if len(result) < 4 and identity.artist and re.search(r"\s[-–—]\s", original):
            caption_title, caption_artist = re.split(r"\s[-–—]\s", original, maxsplit=1)
            if any(artist and artist in _normalized(caption_artist) for artist in artists):
                if difflib.SequenceMatcher(
                    None, _normalized(caption_title), title, autojunk=False
                ).ratio() >= 0.75:
                    continue
        translation = (row.translation or "").strip()
        if translation and (_looks_like_non_lyric_web_text(translation)
                            or _is_credit_line(translation)):
            translation = ""
        result.append(LyricPair(original, translation or None))
    return result


def _page_title(document: str) -> str:
    parser = _PageTitleParser()
    parser.feed(document)
    return parser.value()


def _page_matches_identity(identity: TrackIdentity, page_title: str) -> bool:
    if not page_title:
        return True
    actual = _normalized(page_title)
    requested = _normalized(identity.title)
    if requested and requested in actual:
        return True

    slash_parts = [
        _normalized(value)
        for value in re.split(r"\s*[/／]\s*", identity.title)
        if value.strip()
    ]
    if len(slash_parts) > 1 and all(
        len(value) >= 3 and value in actual for value in slash_parts
    ):
        return True

    without_feat = re.sub(
        r"\s*[（(]\s*(?:feat\.?|ft\.?)\s*[^)）]*[)）]\s*",
        "",
        identity.title,
        flags=re.IGNORECASE,
    )
    base = _normalized(without_feat)
    if base and base != requested and len(base) >= 3 and base in actual:
        artists = [
            _normalized(value)
            for value in re.split(r"\s*[,、/&／]\s*", identity.artist)
            if value.strip()
        ]
        return any(len(value) >= 2 and value in actual for value in artists)
    return False


def extract_bilingual_pairs(document: str) -> list[LyricPair]:
    parser = _VisibleTextParser()
    parser.feed(document)
    raw = "".join(parser.parts)
    lines = [
        re.sub(r"\s+", " ", html_module.unescape(line)).strip()
        for line in raw.splitlines()
    ]
    lines = [line for line in lines if 0 < len(line) <= 300]

    runs: list[list[LyricPair]] = []
    current: list[LyricPair] = []

    def finish_run() -> None:
        nonlocal current
        if current:
            runs.append(current)
            current = []

    index = 0
    while index + 1 < len(lines):
        original = lines[index]
        translated = lines[index + 1]
        if _looks_like_original(original) and _looks_like_chinese_translation(translated):
            original_meta = _looks_like_non_lyric_web_text(original)
            translated_meta = _looks_like_non_lyric_web_text(translated)
            if original_meta or translated_meta:
                if (
                    original.strip() not in _SOFT_WEB_HELPERS
                    and translated.strip() not in _SOFT_WEB_HELPERS
                ):
                    finish_run()
            else:
                pair = LyricPair(original, translated)
                if not current or current[-1] != pair:
                    current.append(pair)
            index += 2
            continue
        if _looks_like_non_lyric_web_text(original):
            finish_run()
        index += 1
    finish_run()
    return max(runs, key=len) if runs else []


def choose_candidate(
    candidates: Iterable[LyricCandidate],
    *,
    require_translation: bool,
) -> LyricCandidate:
    materialized = list(candidates)
    if require_translation:
        materialized = [
            candidate
            for candidate in materialized
            if any(row.translation for row in candidate.lines)
        ]
    if not materialized:
        kind = "translated lyrics" if require_translation else "lyrics"
        raise LookupError(f"No candidate supplied {kind}")

    def score(candidate: LyricCandidate) -> tuple[int, float, int, int]:
        translated = sum(bool(row.translation) for row in candidate.lines)
        coverage = translation_coverage(candidate.lines)
        near_complete = int(coverage >= 0.95)
        return near_complete, coverage, translated, len(candidate.lines)

    return max(materialized, key=score)


def _single_line(value: str) -> str:
    return re.sub(r"[\r\n]+", " ", value).strip()


def _to_traditional_chinese(value: str) -> str:
    value = value.replace("\x00", "")
    if not value:
        return value
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        flags = 0x04000000  # LCMAP_TRADITIONAL_CHINESE
        size = kernel32.LCMapStringEx("zh-TW", flags, value, -1, None, 0, None, None, 0)
        if not size:
            return value
        buffer = ctypes.create_unicode_buffer(size)
        if not kernel32.LCMapStringEx(
            "zh-TW", flags, value, -1, buffer, size, None, None, 0
        ):
            return value
        return buffer.value
    except (AttributeError, OSError):
        return value


def render_lyrics(lines: Iterable[LyricPair]) -> str:
    rendered: list[str] = []
    for row in lines:
        # Two ASCII spaces is the file-format delimiter; normalize runs
        # *inside* both fields so the downstream parser cannot split a
        # Japanese lyric in the middle of its original text.
        original = re.sub(r" {2,}", " ", _single_line(row.original))
        translation = _to_traditional_chinese(
            re.sub(r" {2,}", " ", _single_line(row.translation or ""))
        )
        if not original:
            continue
        rendered.append(f"{original}  {translation}" if translation else original)
    if not rendered:
        raise ValueError("No lyric lines to write")
    return "\n".join(rendered) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def write_result(
    audio_path: Path,
    candidate: LyricCandidate,
    *,
    output_root: Path,
    overwrite: bool,
    translation_method: str = "",
    status: str = "",
    translation_error: str = "",
    human_approved: bool = False,
    oogg_endpoint: str = "",
) -> tuple[Path, Path]:
    audio_path = audio_path.resolve()
    output_root = Path(output_root)
    txt_dir = output_root / "txt"
    report_dir = output_root / "report"
    output = txt_dir / f"{audio_path.stem}.lyrics.txt"
    provenance = report_dir / f"{audio_path.stem}.lyrics.sources.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing lyrics: {output}")
    if provenance.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing provenance: {provenance}")

    lyrics_text = render_lyrics(candidate.lines)
    payload = {
        "txt_sha256": hashlib.sha256(lyrics_text.encode("utf-8")).hexdigest(),
        "verification": getattr(candidate, "verification", None),
        "selected_source": candidate.source,
        "source_url": candidate.url,
        "title": candidate.title,
        "artist": candidate.artist,
        "translator": candidate.translator,
        "line_count": len(candidate.lines),
        "translated_line_count": sum(bool(row.translation) for row in candidate.lines),
        "translation_coverage": translation_coverage(candidate.lines),
        "google_search_url": candidate.google_search,
        "google_candidates": candidate.discovered_urls,
        "translation_method": translation_method,
        "human_approved": human_approved,
        "status": status,
        "translation_error": translation_error,
        "oogg_endpoint": oogg_endpoint,
    }
    provenance_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(output, lyrics_text)
    _atomic_write_text(provenance, provenance_text)
    return output, provenance


class CollectionResult(NamedTuple):
    candidates: list[LyricCandidate]
    google_urls: list[str]
    errors: list[str]


def _normalized(value: str) -> str:
    return re.sub(r"[^\w]+", "", value, flags=re.UNICODE).casefold()


def _metadata_score(
    identity: TrackIdentity,
    *,
    title: str,
    artist: str,
    duration: float | None,
) -> tuple[int, int, float]:
    requested_title = _normalized(identity.title)
    candidate_title = _normalized(title)
    requested_artist = _normalized(identity.artist)
    candidate_artist = _normalized(artist)
    title_score = 2 if requested_title and requested_title == candidate_title else int(
        requested_title in candidate_title or candidate_title in requested_title
    )
    artist_score = 2 if requested_artist and requested_artist == candidate_artist else int(
        bool(requested_artist)
        and (requested_artist in candidate_artist or candidate_artist in requested_artist)
    )
    duration_penalty = (
        abs(identity.duration - duration)
        if identity.duration is not None and duration is not None
        else 9999.0
    )
    return title_score, artist_score, -duration_penalty


def _plain_lyric_pairs(text: str) -> list[LyricPair]:
    result: list[LyricPair] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip()
        if line and not _is_credit_line(line):
            result.append(LyricPair(line))
    return result


def fetch_lrclib_candidate(identity: TrackIdentity, client) -> LyricCandidate:
    params = {"track_name": identity.title}
    if identity.artist:
        params["artist_name"] = identity.artist
    payload = client.get_json("https://lrclib.net/api/search", params=params)
    records = payload if isinstance(payload, list) else []
    records = [record for record in records if not record.get("instrumental")]
    if not records:
        raise LookupError("LRCLIB returned no song matches")

    def record_score(record):
        duration = record.get("duration")
        return _metadata_score(
            identity,
            title=str(record.get("trackName") or record.get("name") or ""),
            artist=str(record.get("artistName") or ""),
            duration=float(duration) if duration is not None else None,
        )

    record = max(records, key=record_score)
    matched_title, matched_artist, _ = record_score(record)
    record_duration = record.get("duration")
    if matched_title < 1 or (identity.artist and matched_artist < 1):
        raise LookupError("LRCLIB best match failed title/artist metadata check")
    if (identity.duration is not None and record_duration is not None
            and abs(identity.duration - float(record_duration)) > 15.0):
        raise LookupError("LRCLIB best match failed audio duration check")
    synced = record.get("syncedLyrics") or ""
    if synced:
        lines = [LyricPair(row.text) for row in parse_lrc(synced)]
    else:
        lines = _plain_lyric_pairs(record.get("plainLyrics") or "")
    if not lines:
        raise LookupError("LRCLIB match has no usable original lyrics")
    record_id = record.get("id")
    return LyricCandidate(
        source="lrclib",
        url=f"https://lrclib.net/api/get/{record_id}",
        title=str(record.get("trackName") or record.get("name") or identity.title),
        artist=str(record.get("artistName") or identity.artist),
        lines=lines,
    )


def google_search_url(identity: TrackIdentity) -> str:
    from urllib.parse import urlencode

    query = " ".join(
        part for part in (identity.title, identity.artist, "歌詞 中文翻譯") if part
    )
    return "https://www.google.com/search?" + urlencode({"q": query})


def fetch_google_candidates(
    identity: TrackIdentity,
    client,
    *,
    max_pages: int = 5,
) -> tuple[list[str], list[LyricCandidate]]:
    query = " ".join(
        part for part in (identity.title, identity.artist, "歌詞 中文翻譯") if part
    )
    search_document = client.get_text(
        "https://www.google.com/search",
        params={"q": query, "udm": "14"},
    )
    urls = parse_google_results(search_document)
    if not urls and hasattr(client, "google_search_urls"):
        urls = list(client.google_search_urls(identity))
    candidates: list[LyricCandidate] = []
    for url in urls[:max_pages]:
        try:
            document = client.get_text(url)
            page_title = _page_title(document)
            if not _page_matches_identity(identity, page_title):
                continue
            pairs = extract_bilingual_pairs(document)
        except Exception:
            continue
        if len(pairs) < 2:
            continue
        host = urlparse(url).hostname or "page"
        candidates.append(
            LyricCandidate(
                source=f"google:{host}",
                url=url,
                title=page_title or identity.title,
                artist="",
                lines=pairs,
            )
        )
    return urls, candidates


def collect_candidates(
    identity: TrackIdentity,
    client,
    *,
    lrclib_fetcher=fetch_lrclib_candidate,
    google_fetcher=fetch_google_candidates,
) -> CollectionResult:
    candidates: list[LyricCandidate] = []
    errors: list[str] = []

    try:
        candidates.append(lrclib_fetcher(identity, client))
    except Exception as exc:
        errors.append(f"lrclib: {exc}")

    google_urls: list[str] = []
    try:
        google_urls, google_candidates = google_fetcher(identity, client)
        candidates.extend(google_candidates)
        if not google_urls:
            errors.append("google: search_blocked")
    except Exception as exc:
        errors.append(f"google: {exc}")

    for candidate in candidates:
        candidate.lines = _sanitize_candidate_lines(candidate.lines, identity)
    candidates = [candidate for candidate in candidates if candidate.lines]

    # Cross-source check: a blog/page can supply valid-looking two-column rows
    # from an entirely different song. Keep discovery URLs, but never trust
    # those rows when independently retrieved structured originals disagree.
    structured = [
        item for item in candidates
        if item.source == "lrclib" and len(item.lines) >= 8
    ]
    safe_candidates: list[LyricCandidate] = []
    for candidate in candidates:
        if not candidate.source.startswith("google:"):
            safe_candidates.append(candidate)
            continue
        host = (urlparse(candidate.url).hostname or "").lower()
        supported = host in {
            "home.gamer.com.tw", "forum.gamer.com.tw", "www.marumaru-x.com",
            "marumaru-x.com",
        }
        if not supported and not structured:
            errors.append(f"unverified lyric page host rejected: {host}")
            continue
        if len(candidate.lines) < (4 if structured else 8):
            errors.append(f"short unverified lyric page rejected: {host}")
            continue
        if structured:
            originals = {_normalized(row.original) for item in structured for row in item.lines}
            page_lines = [_normalized(row.original) for row in candidate.lines]
            overlap = sum(bool(line) and line in originals for line in page_lines) / len(page_lines)
            if overlap < 0.55:
                errors.append(f"web original lyrics disagree with structured sources: {host}")
                continue
        safe_candidates.append(candidate)
    candidates = safe_candidates

    search_url = google_search_url(identity)
    for candidate in candidates:
        candidate.discovered_urls = list(google_urls)
        candidate.google_search = search_url
    return CollectionResult(candidates, google_urls, errors)


def _translation_messages(
    title: str,
    numbered_lines: list[tuple[str, str]],
    target_ids: list[str],
) -> list[dict[str, str]]:
    context = [{"id": line_id, "text": text} for line_id, text in numbered_lines]
    system = (
        "你是日文、英文歌曲的繁體中文逐句譯者，不負責尋找或創作原文。"
        "輸入中的 original 都是已蒐集的歌詞；禁止補寫、猜測或改動任何原文。"
        "把每行完整忠實譯成自然流暢的繁體中文。"
        "尤其檢查否定、條件、主詞受詞、時態、口語省略、比喻及本句與下一句的界線；"
        "不可自行添加原文沒有的人物、事件、情緒或歌詞。"
        "日文『報われる』是得到回報、『スクラップ』依情境是廢料等，"
        "遇到有歧義的詞必須先按整首上下文判斷；重複句前後保持一致。"
        "檢查中文沒有重字、病句或不通順詞組；不照搬網頁翻譯或網頁導覽文字。"
        "full_song_context 只供理解上下文；只翻譯 translate_only_ids 指定的行。"
        "如果真的無法有根據地翻譯某行，text 請輸出 __UNSURE__，"
        "嚴禁杜撰。每個指定 id 恰好一次，不得合併、拆分、省略或增加其他 id。"
        "每個 text 必須是單行，只回傳 JSON 物件，格式為 "
        '{"translations":[{"id":"L001","text":"..."}]}。'
    )
    user = json.dumps(
        {
            "title": title,
            "target": "繁體中文",
            "full_song_context": context,
            "translate_only_ids": target_ids,
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_model_json(payload: dict) -> dict:
    try:
        content = str(payload["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Local model response did not contain message content") from exc
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.I | re.S)
    if fenced_blocks:
        try:
            result = json.loads(fenced_blocks[-1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Local model's final fenced block was not valid JSON: {exc}") from exc
        if not isinstance(result, dict):
            raise ValueError("Local model response must be a JSON object")
        return result

    content = content.strip()
    decoder = json.JSONDecoder()
    objects: list[object] = []
    position = 0
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position >= len(content):
            break
        try:
            result, position = decoder.raw_decode(content, position)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Local model response was not valid JSON: {exc}") from exc
        objects.append(result)
    if not objects:
        raise ValueError("Local model response did not contain JSON")
    if any(item != objects[0] for item in objects[1:]):
        raise ValueError("Local model returned different JSON objects in one response")
    result = objects[0]
    if not isinstance(result, dict):
        raise ValueError("Local model response must be a JSON object")
    return result


def _validate_translation_rows(result: dict, expected_ids: list[str]) -> dict[str, str]:
    rows = result.get("translations")
    if not isinstance(rows, list):
        raise ValueError("Local model response is missing translations list")
    translated: dict[str, str] = {}
    observed_order: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Local model returned a non-object translation row")
        line_id = str(row.get("id", ""))
        text = str(row.get("text", "")).strip()
        if line_id in translated:
            raise ValueError(f"duplicate id {line_id}")
        if "\n" in text or "\r" in text:
            raise ValueError(f"translation {line_id} must stay on one line")
        if "__UNSURE__" in text.upper():
            raise ValueError(f"translation {line_id} is marked unverified by OOGG")
        translated[line_id] = text
        observed_order.append(line_id)

    missing = [line_id for line_id in expected_ids if not translated.get(line_id)]
    extra = [line_id for line_id in translated if line_id not in expected_ids]
    if missing:
        raise ValueError(f"missing/empty ids: {','.join(missing)}")
    if extra:
        raise ValueError(f"extra ids: {','.join(extra)}")
    if observed_order != expected_ids:
        raise ValueError("translation id order differs")
    return translated


def translate_with_local_llm(
    lines: Iterable[str],
    *,
    title: str,
    endpoint: str,
    model: str,
    post_json,
    batch_size: int = 12,
) -> list[LyricPair]:
    source_lines = [_single_line(line) for line in lines if _single_line(line)]
    if not source_lines:
        raise ValueError("No original lyric lines supplied for translation")
    if batch_size < 1 or batch_size > 12:
        raise ValueError("batch_size must be between 1 and 12")

    numbered = [(f"L{index:03d}", text) for index, text in enumerate(source_lines, start=1)]
    all_translations: dict[str, str] = {}
    chat_url = endpoint.rstrip("/") + "/v1/chat/completions"
    for start in range(0, len(numbered), batch_size):
        chunk = numbered[start : start + batch_size]
        expected_ids = [line_id for line_id, _ in chunk]
        request = {
            "model": model,
            "temperature": 0,
            "max_tokens": 1536,
            "messages": _translation_messages(title, numbered, expected_ids),
        }
        last_error: ValueError | None = None
        for _attempt in range(2):
            payload = post_json(chat_url, request, 300)
            try:
                result = _parse_model_json(payload)
                batch_translations = _validate_translation_rows(result, expected_ids)
            except ValueError as exc:
                last_error = exc
                continue
            all_translations.update(batch_translations)
            break
        else:
            assert last_error is not None
            batch_number = start // batch_size + 1
            raise ValueError(f"batch {batch_number}: {last_error}") from last_error

    return [LyricPair(text, all_translations[line_id]) for line_id, text in numbered]


def review_translations_with_local_llm(
    lines: Iterable[LyricPair],
    *,
    title: str,
    endpoint: str,
    model: str,
    post_json,
    batch_size: int = 12,
) -> list[LyricPair]:
    """Second independent *prompt* pass, retaining original rows exactly.

    The same local model is not an independent evidence source; its review
    is a quality-assistance step, NOT semantic certification.
    """
    source = list(lines)
    if not source:
        raise ValueError("No translation rows to review")
    if batch_size < 1 or batch_size > 12:
        raise ValueError("review batch_size must be between 1 and 12")
    if any(not row.original.strip() or not (row.translation or "").strip() for row in source):
        raise ValueError("Review requires complete one-to-one translated rows")
    numbered = [
        {"id": f"L{i:03d}", "original": row.original, "translation": row.translation}
        for i, row in enumerate(source, 1)
    ]
    review_system = (
        "你是歌曲中日翻譯的第二輪校對者。這次不需要重新翻譯全曲。"
        "只修正可根據日文或英文原文確認的錯譯、否定與主客體顛倒、漏譯、"
        "上一行譯文錯掛下一行、中文重字及明顯病句。"
        "原翻譯沒錯的行必須原封不動輸出；不得美化造成意思變化。"
        "所有 original 只供閱讀，禁止更動、刪除、續寫或猜歌詞。"
        "以完整 full_song_context 判斷語境，只處理 review_only_ids。"
        "每個 id 須輸出一次、順序相同。若無法依據原文確認，"
        "text 輸出 __UNSURE__，不能猜。"
        "僅輸出 JSON："
        '{"translations":[{"id":"L001","text":"已校對的繁體中文"}]}'
    )
    translated: dict[str, str] = {}
    chat_url = endpoint.rstrip("/") + "/v1/chat/completions"
    for start in range(0, len(numbered), batch_size):
        chunk = numbered[start : start + batch_size]
        expected_ids = [row["id"] for row in chunk]
        review_user = json.dumps(
            {
                "title": title,
                "full_song_context": numbered,
                "review_only_ids": expected_ids,
            },
            ensure_ascii=False,
        )
        request = {
            "model": model, "temperature": 0, "max_tokens": 1600,
            "messages": [
                {"role": "system", "content": review_system},
                {"role": "user", "content": review_user},
            ],
        }
        last_error = None
        for _attempt in range(2):
            payload = post_json(chat_url, request, 300)
            try:
                verified = _validate_translation_rows(
                    _parse_model_json(payload), expected_ids
                )
            except ValueError as exc:
                last_error = exc
                continue
            translated.update(verified)
            break
        else:
            raise ValueError(
                f"Review batch {start // batch_size + 1}: {last_error}"
            ) from last_error
    return [
        LyricPair(row.original, translated[f"L{i:03d}"])
        for i, row in enumerate(source, 1)
    ]


SUPPORTED_AUDIO_EXTENSIONS = {
    ".flac",
    ".mp3",
    ".m4a",
    ".wav",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".aiff",
    ".alac",
}


def probe_audio(path: Path, *, runner=subprocess.run) -> TrackIdentity:
    fallback = identity_from_filename(path)
    args = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:format_tags=title,artist,album",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = runner(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except (FileNotFoundError, OSError):
        return fallback
    if completed.returncode != 0:
        return fallback
    try:
        payload = json.loads(completed.stdout)
        format_data = payload.get("format") or {}
        tags = {str(key).casefold(): value for key, value in (format_data.get("tags") or {}).items()}
        duration_value = format_data.get("duration")
        duration = float(duration_value) if duration_value not in (None, "") else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback
    return TrackIdentity(
        title=str(tags.get("title") or fallback.title).strip(),
        artist=str(tags.get("artist") or fallback.artist).strip(),
        album=str(tags.get("album") or "").strip(),
        duration=duration,
    )


class HttpClient:
    def __init__(self, *, user_agent: str = "Lyricsee/0.1 (local LRC tools helper)") -> None:
        self.user_agent = user_agent
        self._last_request = 0.0

    def _throttle(self) -> None:
        remaining = 0.25 - (time.monotonic() - self._last_request)
        if remaining > 0:
            time.sleep(remaining)

    def _request(
        self,
        url: str,
        *,
        params: dict | None = None,
        payload: dict | None = None,
        timeout: float = 30,
    ) -> str:
        if params:
            separator = "&" if "?" in url else "?"
            url = url + separator + urlencode(params)
        body = None
        headers = {"User-Agent": self.user_agent, "Accept": "application/json,text/html;q=0.9,*/*;q=0.8"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=headers, method="POST" if body else "GET")
        for attempt in range(2):
            self._throttle()
            try:
                with urlopen(request, timeout=timeout) as response:
                    raw = response.read(8 * 1024 * 1024)
                    if (response.headers.get("Content-Encoding") or "").casefold() == "gzip":
                        raw = gzip.decompress(raw)
                    charset = response.headers.get_content_charset() or "utf-8"
                    self._last_request = time.monotonic()
                    return raw.decode(charset, errors="replace")
            except HTTPError as exc:
                self._last_request = time.monotonic()
                if attempt == 0 and exc.code in (429, 503):
                    retry_after = exc.headers.get("Retry-After", "1")
                    try:
                        time.sleep(max(0.0, float(retry_after)))
                    except ValueError:
                        time.sleep(1.0)
                    continue
                raise
        raise RuntimeError("HTTP request retry loop ended unexpectedly")

    def get_text(self, url: str, params: dict | None = None) -> str:
        return self._request(url, params=params)

    def get_json(
        self,
        url: str,
        params: dict | None = None,
        timeout: float = 30,
    ) -> dict | list:
        return json.loads(self._request(url, params=params, timeout=timeout))

    def post_json(self, url: str, payload: dict, timeout: float) -> dict:
        result = json.loads(self._request(url, payload=payload, timeout=timeout))
        if not isinstance(result, dict):
            raise ValueError("JSON response was not an object")
        return result

    def google_search_urls(self, identity: TrackIdentity) -> list[str]:
        configured = os.environ.get("LYRICSEE_CHROME", "").strip()
        executable: str | None = None
        if configured and Path(configured).is_file():
            executable = configured
        else:
            default = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
            if default.is_file():
                executable = str(default)
            else:
                executable = shutil.which("chrome")
        if not executable:
            return []

        search_url = google_search_url(identity)
        try:
            with tempfile.TemporaryDirectory(prefix="lyricsee-chrome-") as profile:
                completed = subprocess.run(
                    [
                        executable,
                        "--headless=new",
                        f"--user-data-dir={profile}",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-gpu",
                        "--dump-dom",
                        search_url,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=False,
                )
                if completed.returncode != 0:
                    return []
                document = (completed.stdout or b"").decode("utf-8", errors="replace")
                return parse_google_results(document)
        except (OSError, subprocess.TimeoutExpired):
            return []
