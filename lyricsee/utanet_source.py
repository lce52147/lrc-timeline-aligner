"""uta-net (歌ネット) as the primary, publisher-licensed source of Japanese lyrics.

Flow: title search (``/search/?target=so&type=in&keyword=``) → pick the row
whose title *and* artist match the recording → fetch ``/song/<id>/`` → take
the text of ``div#kashi_area`` only, drop page furniture.  A row whose title
merely contains the requested title, or whose artist is unknown, is never
silently accepted: that is how "Call you" became "I Call Your Name" and
"花がら" became "花がらもようの便箋に" in the 2026-10-24 audit.
"""
from __future__ import annotations

import html as html_module
import re
from html.parser import HTMLParser
from typing import NamedTuple
from urllib.parse import urlencode

try:
    from lyricsee_core import SEARCH_USER_AGENT, _is_credit_line
    from lyric_sites import (
        AmbiguousHits, LyricSite, SiteHit, fetch_site_candidate,
        resolve_ambiguous_hits as _resolve,
    )
except ModuleNotFoundError:  # pragma: no cover - package import style
    from lyricsee.lyricsee_core import SEARCH_USER_AGENT, _is_credit_line
    from lyricsee.lyric_sites import (
        AmbiguousHits, LyricSite, SiteHit, fetch_site_candidate,
        resolve_ambiguous_hits as _resolve,
    )

SOURCE = "uta-net"
HOST = "www.uta-net.com"
SEARCH_URL = f"https://{HOST}/search/"
SONG_URL = f"https://{HOST}/song/{{song_id}}/"
HEADERS = {
    "User-Agent": SEARCH_USER_AGENT,
    "Accept-Language": "ja,en-US;q=0.8,zh-TW;q=0.6",
}
_SONG_HREF_RE = re.compile(r"^(?:https?://(?:www\.)?uta-net\.com)?/song/(\d+)/?$")
_DESCRIPTION_RE = re.compile(r"^(?P<artist>.+?)の「(?P<title>.+)」歌詞ページです")
_OG_TITLE_RE = re.compile(r"^「(?P<title>.+)/(?P<artist>[^/]+)」の歌詞")
_PAGE_TITLE_RE = re.compile(r"^(?P<text>.+?)\s*歌詞\s*-\s*歌ネット$")
_UTANET_NOISE_RE = re.compile(
    r"マイ歌ネット|歌ネット|この歌詞を|このアーティストを|歌詞ページ|コトバのキモチ|"
    r"^※|^\(C\)|^©|lyrics powered by|^歌詞$",
    flags=re.IGNORECASE,
)


class UtanetHit(NamedTuple):
    song_id: str
    title: str
    artist: str
    lyricist: str = ""
    composer: str = ""
    opening: str = ""

    @property
    def url(self) -> str:
        return SONG_URL.format(song_id=self.song_id)


class UtanetSong(NamedTuple):
    title: str
    artist: str
    lines: list[str]


def utanet_search_url(keyword: str, *, match: str = "in") -> str:
    return SEARCH_URL + "?" + urlencode({"target": "so", "type": match, "keyword": keyword})


class _SearchParser(HTMLParser):
    """Rows of the 曲名 search table: song link, artist, lyricist, composer, 歌い出し."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[UtanetHit] = []
        self._row: dict[str, str] | None = None
        self._capture: str | None = None
        self._in_song_anchor = False
        self._span_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        attributes = {key.casefold(): value or "" for key, value in attrs}
        if tag == "tr":
            self._row = {"song_id": "", "title": "", "artist": "", "lyricist": "", "composer": "", "opening": ""}
            self._capture = None
            self._in_song_anchor = False
            return
        if self._row is None:
            return
        if tag == "a":
            href = attributes.get("href", "")
            match = _SONG_HREF_RE.match(href)
            if match and not self._row["song_id"]:
                self._row["song_id"] = match.group(1)
                self._in_song_anchor = True
                self._span_depth = 0
                self._capture = "title"
            elif "/artist/" in href:
                self._capture = "artist"
            elif "/lyricist/" in href:
                self._capture = "lyricist"
            elif "/composer/" in href:
                self._capture = "composer"
            else:
                self._capture = None
            return
        if tag == "span":
            classes = attributes.get("class", "").split()
            if "pc-utaidashi" in classes:
                self._capture = "opening"
            elif self._in_song_anchor:
                self._span_depth += 1
                # Inside the song link only the title span is the title; the
                # mobile-only "utaidashi" span repeats the artist.
                self._capture = "title" if "songlist-title" in classes else None

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self._row is None:
            return
        if tag == "tr":
            if self._row["song_id"] and self._row["title"].strip():
                self.hits.append(
                    UtanetHit(
                        self._row["song_id"],
                        _clean(self._row["title"]),
                        _clean(self._row["artist"]),
                        _clean(self._row["lyricist"]),
                        _clean(self._row["composer"]),
                        _clean(self._row["opening"]),
                    )
                )
            self._row = None
            self._capture = None
        elif tag == "a":
            self._in_song_anchor = False
            self._capture = None
        elif tag == "span" and self._in_song_anchor and self._span_depth:
            self._span_depth -= 1
            self._capture = None

    def handle_data(self, data: str) -> None:
        if self._row is not None and self._capture:
            if self._capture == "title" and self._in_song_anchor and self._span_depth == 0:
                # Text directly inside <a> (older markup without spans).
                if not self._row["title"].strip():
                    self._row["title"] += data
                return
            self._row[self._capture] += data


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", html_module.unescape(value)).strip()


def parse_utanet_search(document: str) -> list[UtanetHit]:
    parser = _SearchParser()
    parser.feed(document)
    seen: set[str] = set()
    hits: list[UtanetHit] = []
    for hit in parser.hits:
        if hit.song_id in seen:
            continue
        seen.add(hit.song_id)
        hits.append(hit)
    return hits


class _SongParser(HTMLParser):
    """Collect ``div#kashi_area`` text (``<br>`` = line break) and page metadata."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.meta: dict[str, str] = {}
        self.title_parts: list[str] = []
        self._depth = 0
        self._done = False
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        attributes = {key.casefold(): value or "" for key, value in attrs}
        if tag == "meta":
            key = (attributes.get("property") or attributes.get("name") or "").casefold()
            if key in {"description", "og:title"} and attributes.get("content"):
                self.meta.setdefault(key, attributes["content"])
            return
        if tag == "title":
            self._in_title = True
            return
        if self._done:
            return
        if self._depth:
            if tag == "br":
                self.parts.append("\n")
            elif tag == "div":
                self._depth += 1
            elif tag in {"p", "li"}:
                self.parts.append("\n")
        elif tag == "div" and attributes.get("id") == "kashi_area":
            self._depth = 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "title":
            self._in_title = False
        if self._done or not self._depth:
            return
        if tag == "div":
            self._depth -= 1
            if not self._depth:
                self._done = True
        elif tag in {"p", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._depth and not self._done:
            self.parts.append(data)


def _page_identity(parser: _SongParser) -> tuple[str, str]:
    description = parser.meta.get("description", "")
    match = _DESCRIPTION_RE.match(html_module.unescape(description).strip())
    if match:
        return _clean(match.group("title")), _clean(match.group("artist"))
    og_title = parser.meta.get("og:title", "")
    match = _OG_TITLE_RE.match(html_module.unescape(og_title).strip())
    if match:
        return _clean(match.group("title")), _clean(match.group("artist"))
    page_title = _clean("".join(parser.title_parts))
    match = _PAGE_TITLE_RE.match(page_title)
    return (match.group("text") if match else page_title), ""


def _lyric_lines(raw: str) -> list[str]:
    lines: list[str] = []
    for part in html_module.unescape(raw).replace("\r", "").split("\n"):
        text = re.sub(r"[ \t　]+", " ", part).strip()
        if not text:
            continue
        # kashi_area holds only the lyric; the broad page-noise regex used for
        # blogs is not applied here because it would also drop lyric lines
        # that begin with 唄 or contain "MV".
        if _UTANET_NOISE_RE.search(text) or _is_credit_line(text):
            continue
        lines.append(text)
    return lines


def parse_utanet_song(document: str) -> UtanetSong:
    parser = _SongParser()
    parser.feed(document)
    title, artist = _page_identity(parser)
    if not parser.parts:
        raise LookupError("uta-net page has no kashi_area lyric block")
    return UtanetSong(title, artist, _lyric_lines("".join(parser.parts)))


# Backwards-compatible name: exact-title rows without artist evidence.
UtanetAmbiguous = AmbiguousHits


def _site_hits(document: str) -> list[SiteHit]:
    return [
        SiteHit(hit.url, hit.title, hit.artist, hit.opening, hit.lyricist, hit.composer)
        for hit in parse_utanet_search(document)
    ]


UTANET = LyricSite(SOURCE, "publisher", utanet_search_url, _site_hits, parse_utanet_song, HEADERS)


def fetch_utanet_candidate(identity, client):
    """Search uta-net for the recording and return its lyrics as a candidate.

    Raises ``LookupError`` naming the rejected rows when no row matches both
    title and artist, or ``UtanetAmbiguous`` (carrying the rows) when rows
    have the exact title but no artist evidence; ``resolve_ambiguous_hits``
    settles those against an independent text.
    """
    return fetch_site_candidate(UTANET, identity, client)


def resolve_ambiguous_hits(identity, client, hits, reference_lines):
    return _resolve(UTANET, identity, client, hits, reference_lines)
