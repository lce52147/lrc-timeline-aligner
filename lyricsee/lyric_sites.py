"""Shared flow for publisher-licensed lyric sites (uta-net, UtaTen).

Each site supplies three pure functions (search URL, search-result parser,
song-page parser); this module does the matching: search by title, accept
only a row whose title *and* artist name the recording, fetch that one song
page, and report rows that carry the exact title without artist evidence so
the caller can resolve them against an independently fetched text.
"""
from __future__ import annotations

import difflib
from typing import Callable, NamedTuple
from urllib.error import HTTPError

try:
    from lyricsee_core import (
        LyricCandidate,
        LyricPair,
        SEARCH_USER_AGENT,
        TrackIdentity,
        artist_match_level,
        title_match_level,
        title_search_keywords,
    )
    from lyric_compare import canonical_text
except ModuleNotFoundError:  # pragma: no cover - package import style
    from lyricsee.lyricsee_core import (
        LyricCandidate,
        LyricPair,
        SEARCH_USER_AGENT,
        TrackIdentity,
        artist_match_level,
        title_match_level,
        title_search_keywords,
    )
    from lyricsee.lyric_compare import canonical_text

MAX_SEARCHES = 4
HEADERS = {
    "User-Agent": SEARCH_USER_AGENT,
    "Accept-Language": "ja,en-US;q=0.8,zh-TW;q=0.6",
}


class SiteHit(NamedTuple):
    url: str
    title: str
    artist: str
    opening: str = ""      # 歌い出し shown in the result row, if any
    lyricist: str = ""
    composer: str = ""


class SitePage(NamedTuple):
    title: str
    artist: str
    lines: list[str]


class LyricSite(NamedTuple):
    name: str                                   # candidate.source
    trust: str                                  # "publisher"
    search_url: Callable[[str], str]
    parse_search: Callable[[str], list[SiteHit]]
    parse_page: Callable[[str], SitePage]
    headers: dict = HEADERS


class HitScore(NamedTuple):
    title: int   # 0 none, 1 contains, 2 same after stripping feat/brackets, 3 identical
    artist: int  # 0 none, 1 partial or album/folder hint, 2 identical


class AmbiguousHits(LookupError):
    """Rows carry the exact title but none names the artist.

    ``hits`` are the (score, row) pairs; ``resolve_ambiguous_hits`` can pick
    the row whose 歌い出し matches an independently fetched lyric.
    """

    def __init__(self, message: str, hits: list, site: LyricSite) -> None:
        super().__init__(message)
        self.hits = hits
        self.site = site


def score_hit(identity: TrackIdentity, hit: SiteHit) -> HitScore:
    return HitScore(title_match_level(identity, hit.title), artist_match_level(identity, hit.artist))


def is_strong(score: HitScore) -> bool:
    return score.title >= 2 and score.artist >= 1


def describe_hit(hit: SiteHit, score: HitScore) -> str:
    return f"{hit.title} / {hit.artist or '?'} (title={score.title}, artist={score.artist})"


def fetch_site_candidate(site: LyricSite, identity: TrackIdentity, client) -> LyricCandidate:
    """Search the site for the recording and return its lyrics as a candidate.

    Raises ``LookupError`` naming the rejected rows when no row matches both
    title and artist, or ``AmbiguousHits`` when exact-title rows exist whose
    artist could not be confirmed.
    """
    keywords = title_search_keywords(identity)
    rejected: list[str] = []
    weak: dict[str, tuple[HitScore, SiteHit]] = {}
    searched: list[str] = []
    chosen: tuple[HitScore, SiteHit] | None = None
    search_url = ""
    for keyword in keywords[:MAX_SEARCHES]:
        url = site.search_url(keyword)
        searched.append(keyword)
        try:
            document = client.get_text(url, headers=site.headers, retries=2)
        except HTTPError as exc:
            if exc.code == 404:
                continue  # a keyword without matches
            raise
        scored = sorted(
            ((score_hit(identity, hit), hit) for hit in site.parse_search(document)),
            key=lambda item: (item[0].title, item[0].artist),
            reverse=True,
        )
        strong = [item for item in scored if is_strong(item[0])]
        if strong:
            chosen = strong[0]
            search_url = url
            break
        for score, hit in scored:
            if score.title >= 2:
                weak.setdefault(hit.url, (score, hit))
            if score.title >= 1 and len(rejected) < 6:
                rejected.append(describe_hit(hit, score))
    if chosen is None:
        detail = "; ".join(rejected) if rejected else "no row with this title"
        message = f"{site.name}: no row matched title and artist for {' | '.join(searched)}; rejected: {detail}"
        if weak:
            raise AmbiguousHits(message, list(weak.values()), site)
        raise LookupError(message)
    return fetch_song_candidate(site, identity, client, chosen, search_url=search_url, rejected=rejected, searched=searched)


def opening_matches(opening: str, reference_lines: list[str], *, min_ratio: float = 0.6) -> float:
    """Similarity of a result row's 歌い出し to the start of a reference lyric
    (notation folded); 0.0 when either side is too short or unlike."""
    head = canonical_text(opening)[:160]
    reference = canonical_text(" ".join(reference_lines))[:160]
    if len(head) < 12 or len(reference) < 12:
        return 0.0
    size = min(len(head), len(reference))
    ratio = difflib.SequenceMatcher(None, head[:size], reference[:size], autojunk=False).ratio()
    return ratio if ratio >= min_ratio else 0.0


def resolve_ambiguous_hits(
    site: LyricSite,
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
                f"{site.name}: exact-title rows do not match the reference lyric: "
                + "; ".join(describe_hit(h, s) for s, h in hits[:6])
            )
        candidate = fetch_song_candidate(site, identity, client, (score, hit), search_url="", rejected=[], searched=[])
        candidate.meta["identity"]["opening_matched_reference"] = round(best_ratio, 3)
        return candidate
    if len(hits) == 1:
        return fetch_song_candidate(site, identity, client, hits[0], search_url="", rejected=[], searched=[])
    raise LookupError(
        f"{site.name}: several rows carry this title and none names the artist: "
        + "; ".join(describe_hit(h, s) for s, h in hits[:6])
    )


def fetch_song_candidate(
    site: LyricSite,
    identity: TrackIdentity,
    client,
    chosen: tuple,
    *,
    search_url: str,
    rejected: list[str],
    searched: list[str],
) -> LyricCandidate:
    score, hit = chosen
    document = client.get_text(hit.url, headers=site.headers, retries=2)
    page = site.parse_page(document)
    page_title = page.title or hit.title
    page_artist = page.artist or hit.artist
    page_score = HitScore(title_match_level(identity, page_title), artist_match_level(identity, page_artist))
    if page_score.title < 2 and score.title >= 2:
        raise LookupError(f"{site.name}: song page title {page_title!r} does not match the search row {hit.title!r}")
    if not page.lines:
        raise LookupError(f"{site.name}: song page has no lyric lines")
    final = HitScore(max(score.title, page_score.title), max(score.artist, page_score.artist))
    return LyricCandidate(
        source=site.name,
        url=hit.url,
        title=page_title,
        artist=page_artist,
        lines=[LyricPair(line) for line in page.lines],
        search_url=search_url,
        search_engine=site.name,
        meta={
            "trust": site.trust,
            "identity": {
                "title_level": final.title,
                "artist_level": final.artist,
                "strong": is_strong(final),
                "keywords": searched,
            },
            "lyricist": hit.lyricist,
            "composer": hit.composer,
            "rejected": rejected,
        },
    )
