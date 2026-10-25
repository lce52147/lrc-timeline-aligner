"""J-Lyric (j-lyric.net) as the third publisher-licensed source of Japanese lyrics.

Exact-title search ``/search.php?kt=<keyword>&ct=1`` lists ``div.bdy`` blocks
with the song link, 歌：artist and a 歌詞：snippet; the song page keeps the
lyric in ``p#Lyric`` (``<br>`` line breaks, optional ``<rt>`` furigana).  Row
matching and page fetching are shared with uta-net and UtaTen in
``lyric_sites``.
"""
from __future__ import annotations

import html as html_module
import re
from urllib.parse import urlencode

try:
    from lyricsee_core import _is_credit_line
    from lyric_sites import (
        AmbiguousHits, LyricSite, SiteHit, SitePage, fetch_site_candidate,
        resolve_ambiguous_hits as _resolve,
    )
except ModuleNotFoundError:  # pragma: no cover - package import style
    from lyricsee.lyricsee_core import _is_credit_line
    from lyricsee.lyric_sites import (
        AmbiguousHits, LyricSite, SiteHit, SitePage, fetch_site_candidate,
        resolve_ambiguous_hits as _resolve,
    )

SOURCE = "j-lyric"
HOST = "j-lyric.net"
_BLOCK_RE = re.compile(r'<div class="bdy">(.*?)</div>', re.S)
_TITLE_RE = re.compile(r'<p class="mid">\s*(?:♪\s*)?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_ARTIST_RE = re.compile(r"歌：\s*(?:<a[^>]*>)?(.*?)(?:</a>)?\s*</p>", re.S)
_OPENING_RE = re.compile(r"歌詞：(.*?)</p>", re.S)
_LYRIC_RE = re.compile(r'<p id="Lyric"[^>]*>(.*?)</p>', re.S)
_RT_RE = re.compile(r"<rt>.*?</rt>", re.S)
_DESCRIPTION_RE = re.compile(r'<meta name="description" content="(.*?)"', re.S)
_PAGE_DESCRIPTION_RE = re.compile(r"^(?P<artist>.+?)の「(?P<title>.+)」歌詞ページ")
_TITLE_TAG_RE = re.compile(r"<title>(.*?)</title>", re.S)
_PAGE_TITLE_RE = re.compile(r"^(?P<title>.+?)／(?P<artist>.+?)\s+歌詞$")
_NOISE_RE = re.compile(r"J-Lyric|歌詞検索|著作権保護", re.IGNORECASE)


def _clean(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", "", fragment)
    return re.sub(r"\s+", " ", html_module.unescape(text)).strip()


def jlyric_search_url(keyword: str) -> str:
    # ct=1: the title must match exactly, which avoids paging through
    # hundreds of partial matches for a short title.
    return f"https://{HOST}/search.php?" + urlencode(
        {"kt": keyword, "ct": 1, "ka": "", "ca": 2, "kl": "", "cl": 2}
    )


def parse_jlyric_search(document: str) -> list[SiteHit]:
    hits: list[SiteHit] = []
    seen: set[str] = set()
    for block in _BLOCK_RE.findall(document):
        title = _TITLE_RE.search(block)
        if not title or "/artist/" not in title.group(1) or not title.group(1).endswith(".html"):
            continue
        url = title.group(1)
        if url.startswith("/"):
            url = f"https://{HOST}" + url
        if url in seen:
            continue
        seen.add(url)
        artist = _ARTIST_RE.search(block)
        opening = _OPENING_RE.search(block)
        hits.append(
            SiteHit(
                url,
                _clean(title.group(2)),
                _clean(artist.group(1)) if artist else "",
                _clean(opening.group(1)).rstrip(" .…") if opening else "",
            )
        )
    return hits


def parse_jlyric_page(document: str) -> SitePage:
    block = _LYRIC_RE.search(document)
    if not block:
        raise LookupError("j-lyric page has no Lyric block")
    body = _RT_RE.sub("", block.group(1))
    body = re.sub(r"<br\s*/?>", "\n", body, flags=re.I)
    lines: list[str] = []
    for part in html_module.unescape(re.sub(r"<[^>]+>", "", body)).split("\n"):
        text = re.sub(r"[ \t　]+", " ", part).strip()
        if not text or _NOISE_RE.search(text) or _is_credit_line(text):
            continue
        lines.append(text)
    title = artist = ""
    description = _DESCRIPTION_RE.search(document)
    if description:
        match = _PAGE_DESCRIPTION_RE.match(html_module.unescape(description.group(1)).strip())
        if match:
            title, artist = _clean(match.group("title")), _clean(match.group("artist"))
    if not title:
        tag = _TITLE_TAG_RE.search(document)
        if tag:
            match = _PAGE_TITLE_RE.match(_clean(tag.group(1)))
            if match:
                title, artist = match.group("title"), match.group("artist")
    return SitePage(title, artist, lines)


JLYRIC = LyricSite(SOURCE, "publisher", jlyric_search_url, parse_jlyric_search, parse_jlyric_page)


def fetch_jlyric_candidate(identity, client):
    return fetch_site_candidate(JLYRIC, identity, client)


def resolve_ambiguous_hits(identity, client, hits, reference_lines):
    return _resolve(JLYRIC, identity, client, hits, reference_lines)


__all__ = [
    "AmbiguousHits", "JLYRIC", "fetch_jlyric_candidate", "jlyric_search_url",
    "parse_jlyric_page", "parse_jlyric_search", "resolve_ambiguous_hits",
]
