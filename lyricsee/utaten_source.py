"""UtaTen (うたてん) as a second publisher-licensed source of Japanese lyrics.

Title search ``/search?title=<keyword>`` lists a ``table.searchResult`` with
the song link, artist and 歌い出し; the song page keeps the lyric in
``div.lyricBody div.hiragana`` with furigana as ``<span class="rt">`` that
must be dropped.  Row matching and page fetching are shared with uta-net in
``lyric_sites``.
"""
from __future__ import annotations

import html as html_module
import re
from urllib.parse import urlencode

try:
    from lyricsee_core import _is_credit_line
    from lyric_sites import (
        AmbiguousHits, HEADERS, LyricSite, SiteHit, SitePage, fetch_site_candidate,
        resolve_ambiguous_hits as _resolve,
    )
except ModuleNotFoundError:  # pragma: no cover - package import style
    from lyricsee.lyricsee_core import _is_credit_line
    from lyricsee.lyric_sites import (
        AmbiguousHits, HEADERS, LyricSite, SiteHit, SitePage, fetch_site_candidate,
        resolve_ambiguous_hits as _resolve,
    )

SOURCE = "utaten"
HOST = "utaten.com"
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TITLE_RE = re.compile(r'<p class="searchResult__title">.*?<a href="(/lyric/[^"]+)"[^>]*>(.*?)</a>', re.S)
_ARTIST_RE = re.compile(r'<td class="searchResult__artist">.*?<a href="/artist/[^"]+"[^>]*>(.*?)</a>', re.S)
_OPENING_RE = re.compile(r'<td class="lyricList__beginning">\s*<a[^>]*>(.*?)</a>', re.S)
_WRITER_RE = re.compile(r"<p>(作詞|作曲)：.*?<a href=\"/songWriter/[^\"]+\"[^>]*>(.*?)</a>", re.S)
_HIRAGANA_RE = re.compile(r'<div class="hiragana"[^>]*>(.*?)</div>', re.S)
_RT_RE = re.compile(r'<span class="rt">.*?</span>', re.S)
_PAGE_TITLE_RE = re.compile(r"^(?P<title>.+?)\s+歌詞\s+(?P<artist>.+?)\s+ふりがな付\s*-\s*うたてん$")
_TITLE_TAG_RE = re.compile(r"<title>(.*?)</title>", re.S)
_NOISE_RE = re.compile(r"うたてん|UtaTen|歌詞検索|ふりがな付", re.IGNORECASE)


def _clean(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", "", fragment)
    return re.sub(r"\s+", " ", html_module.unescape(text)).strip()


def utaten_search_url(keyword: str) -> str:
    return f"https://{HOST}/search?" + urlencode(
        {"sort": "popular_sort_asc", "title": keyword, "show_artists": 1}
    )


def parse_utaten_search(document: str) -> list[SiteHit]:
    if 'class="noItem"' in document:
        return []
    hits: list[SiteHit] = []
    seen: set[str] = set()
    for row in _ROW_RE.findall(document):
        title = _TITLE_RE.search(row)
        if not title:
            continue
        url = f"https://{HOST}" + title.group(1)
        if url in seen:
            continue
        seen.add(url)
        artist = _ARTIST_RE.search(row)
        opening = _OPENING_RE.search(row)
        writers = {kind: _clean(name) for kind, name in _WRITER_RE.findall(row)}
        hits.append(
            SiteHit(
                url,
                _clean(title.group(2)),
                _clean(artist.group(1)) if artist else "",
                _clean(opening.group(1)) if opening else "",
                writers.get("作詞", ""),
                writers.get("作曲", ""),
            )
        )
    return hits


def parse_utaten_page(document: str) -> SitePage:
    block = _HIRAGANA_RE.search(document)
    if not block:
        raise LookupError("utaten page has no hiragana lyric block")
    body = _RT_RE.sub("", block.group(1))
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    lines: list[str] = []
    for part in html_module.unescape(re.sub(r"<[^>]+>", "", body)).split("\n"):
        text = re.sub(r"[ \t　]+", " ", part).strip()
        if not text or _NOISE_RE.search(text) or _is_credit_line(text):
            continue
        lines.append(text)
    title = artist = ""
    tag = _TITLE_TAG_RE.search(document)
    if tag:
        match = _PAGE_TITLE_RE.match(_clean(tag.group(1)))
        if match:
            title, artist = match.group("title"), match.group("artist")
    return SitePage(title, artist, lines)


def utaten_artist_search_url(artist_name: str, title: str = "") -> str:
    return f"https://{HOST}/search?" + urlencode(
        {"sort": "popular_sort_asc", "artist_name": artist_name, "title": title, "show_artists": 1}
    )


def utaten_artist_hits(client, artist_name: str, identity=None) -> list[SiteHit]:
    """Song rows credited to ``artist_name``: the artist's most popular songs, then the artist narrowed by each title alias."""
    queries = [utaten_artist_search_url(artist_name)]
    for alias in getattr(identity, "aliases", ())[:3]:
        queries.append(utaten_artist_search_url(artist_name, alias))
    hits: list[SiteHit] = []
    seen: set[str] = set()
    for url in queries:
        for hit in parse_utaten_search(client.get_text(url, headers=HEADERS, retries=2)):
            if hit.url not in seen:
                seen.add(hit.url)
                hits.append(hit)
    return hits


UTATEN = LyricSite(
    SOURCE, "publisher", utaten_search_url, parse_utaten_search, parse_utaten_page,
    artist_hits=utaten_artist_hits,
)


def fetch_utaten_candidate(identity, client):
    return fetch_site_candidate(UTATEN, identity, client)


def resolve_ambiguous_hits(identity, client, hits, reference_lines):
    return _resolve(UTATEN, identity, client, hits, reference_lines)


__all__ = [
    "AmbiguousHits", "UTATEN", "fetch_utaten_candidate", "parse_utaten_page",
    "parse_utaten_search", "resolve_ambiguous_hits", "utaten_search_url",
]
