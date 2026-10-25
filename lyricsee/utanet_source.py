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
from urllib.error import HTTPError
from urllib.parse import urlencode

try:
    from lyricsee_core import (
        LyricCandidate,
        LyricPair,
        SEARCH_USER_AGENT,
        TrackIdentity,
        _is_credit_line,
        _looks_like_non_lyric_web_text,
        artist_match_level,
        title_match_level,
        title_search_keywords,
    )
except ModuleNotFoundError:  # pragma: no cover - package import style
    from lyricsee.lyricsee_core import (
        LyricCandidate,
        LyricPair,
        SEARCH_USER_AGENT,
        TrackIdentity,
        _is_credit_line,
        _looks_like_non_lyric_web_text,
        artist_match_level,
        title_match_level,
        title_search_keywords,
    )

SOURCE = "uta-net"
HOST = "www.uta-net.com"
SEARCH_URL = f"https://{HOST}/search/"
SONG_URL = f"https://{HOST}/song/{{song_id}}/"
HEADERS = {
    "User-Agent": SEARCH_USER_AGENT,
    "Accept-Language": "ja,en-US;q=0.8,zh-TW;q=0.6",
}
MAX_SEARCHES = 4
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
        if _UTANET_NOISE_RE.search(text) or _is_credit_line(text) or _looks_like_non_lyric_web_text(text):
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


class UtanetAmbiguous(LookupError):
    """Rows carry the exact title but none names the artist.

    ``hits`` are the (score, row) pairs; ``resolve_ambiguous_hits`` can pick
    the row whose 歌い出し matches an independently fetched lyric.
    """

    def __init__(self, message: str, hits: list) -> None:
        super().__init__(message)
        self.hits = hits


class HitScore(NamedTuple):
    title: int   # 0 none, 1 contains, 2 same after stripping feat/brackets, 3 identical
    artist: int  # 0 none, 1 partial or album/folder hint, 2 identical


def score_hit(identity: TrackIdentity, hit: UtanetHit) -> HitScore:
    return HitScore(
        title_match_level(identity, hit.title),
        artist_match_level(identity, hit.artist),
    )


def _strong(score: HitScore) -> bool:
    return score.title >= 2 and score.artist >= 1


def _describe(hit: UtanetHit, score: HitScore) -> str:
    return f"{hit.title} / {hit.artist or '?'} (title={score.title}, artist={score.artist})"


def fetch_utanet_candidate(identity: TrackIdentity, client) -> LyricCandidate:
    """Search uta-net for the recording and return its lyrics as a candidate.

    Raises ``LookupError`` with the rejected rows when no row matches both
    title and artist.  A row with the exact title but no artist evidence is
    returned flagged ``identity.strong == False`` only when it is the single
    such row; the policy layer admits it solely through agreement with an
    independent source.
    """
    keywords = title_search_keywords(identity)
    rejected: list[str] = []
    weak: list[tuple[HitScore, UtanetHit]] = []
    searched: list[str] = []
    chosen: tuple[HitScore, UtanetHit] | None = None
    search_url = ""
    for keyword in keywords[:MAX_SEARCHES]:
        url = utanet_search_url(keyword)
        searched.append(keyword)
        try:
            document = client.get_text(url, headers=HEADERS, retries=2)
        except HTTPError as exc:
            if exc.code == 404:
                continue  # uta-net answers 404 for a keyword without matches
            raise
        hits = parse_utanet_search(document)
        scored = sorted(
            ((score_hit(identity, hit), hit) for hit in hits),
            key=lambda item: (item[0].title, item[0].artist),
            reverse=True,
        )
        strong = [item for item in scored if _strong(item[0])]
        if strong:
            chosen = strong[0]
            search_url = url
            break
        for score, hit in scored:
            if score.title >= 2:
                weak.append((score, hit))
            if score.title >= 1 and len(rejected) < 6:
                rejected.append(_describe(hit, score))
    if chosen is None:
        exact_weak = list({hit.song_id: (score, hit) for score, hit in weak if score.title >= 2}.values())
        detail = "; ".join(rejected) if rejected else "no row with this title"
        message = f"uta-net: no row matched title and artist for {' | '.join(searched)}; rejected: {detail}"
        if exact_weak:
            raise UtanetAmbiguous(message, exact_weak)
        raise LookupError(message)

    return fetch_utanet_song_candidate(identity, client, chosen, search_url=search_url, rejected=rejected, searched=searched)


def opening_matches(opening: str, reference_lines: list[str], *, min_ratio: float = 0.6) -> float:
    """Similarity of a search row's 歌い出し to the start of a reference lyric
    (notation folded); 0.0 when either side is empty."""
    try:
        from lyric_compare import canonical_text
    except ModuleNotFoundError:  # pragma: no cover - package import style
        from lyricsee.lyric_compare import canonical_text
    import difflib

    head = canonical_text(opening)[:160]
    reference = canonical_text(" ".join(reference_lines))[:160]
    if len(head) < 12 or len(reference) < 12:
        return 0.0
    size = min(len(head), len(reference))
    ratio = difflib.SequenceMatcher(None, head[:size], reference[:size], autojunk=False).ratio()
    return ratio if ratio >= min_ratio else 0.0


def resolve_ambiguous_hits(
    identity: TrackIdentity,
    client,
    hits: list,
    reference_lines: list[str] | None,
) -> LyricCandidate:
    """Pick among exact-title rows without artist evidence.

    With an independent reference (LRCLIB) the row whose opening text matches
    it is fetched; the policy then still requires full-text agreement.
    Without a reference only a single row is fetched, flagged weak.
    """
    if reference_lines:
        scored = sorted(
            ((opening_matches(hit.opening, reference_lines), score, hit) for score, hit in hits),
            key=lambda item: item[0],
            reverse=True,
        )
        best_ratio, score, hit = scored[0]
        if best_ratio <= 0.0:
            raise LookupError(
                "uta-net: exact-title rows do not match the reference lyric: "
                + "; ".join(_describe(h, s) for s, h in hits[:6])
            )
        candidate = fetch_utanet_song_candidate(identity, client, (score, hit), search_url="", rejected=[], searched=[])
        candidate.meta["identity"]["opening_matched_reference"] = round(best_ratio, 3)
        return candidate
    if len(hits) == 1:
        return fetch_utanet_song_candidate(identity, client, hits[0], search_url="", rejected=[], searched=[])
    raise LookupError(
        "uta-net: several rows carry this title and none names the artist: "
        + "; ".join(_describe(h, s) for s, h in hits[:6])
    )


def fetch_utanet_song_candidate(
    identity: TrackIdentity,
    client,
    chosen: tuple,
    *,
    search_url: str,
    rejected: list[str],
    searched: list[str],
) -> LyricCandidate:
    score, hit = chosen
    document = client.get_text(hit.url, headers=HEADERS, retries=2)
    song = parse_utanet_song(document)
    page_title = song.title or hit.title
    page_artist = song.artist or hit.artist
    page_score = HitScore(
        title_match_level(identity, page_title),
        artist_match_level(identity, page_artist),
    )
    if page_score.title < 2 and score.title >= 2:
        raise LookupError(f"uta-net: song page title {page_title!r} does not match the search row {hit.title!r}")
    if not song.lines:
        raise LookupError("uta-net: song page has no lyric lines")
    final = HitScore(max(score.title, page_score.title), max(score.artist, page_score.artist))
    return LyricCandidate(
        source=SOURCE,
        url=hit.url,
        title=page_title,
        artist=page_artist,
        lines=[LyricPair(line) for line in song.lines],
        search_url=search_url,
        search_engine="uta-net",
        meta={
            "trust": "publisher",
            "identity": {
                "title_level": final.title,
                "artist_level": final.artist,
                "strong": _strong(final),
                "keywords": searched,
            },
            "lyricist": hit.lyricist,
            "composer": hit.composer,
            "rejected": rejected,
        },
    )
