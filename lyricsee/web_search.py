"""Candidate search over plain HTTP: several engines, several query variants.

A person googles the song title; lyricsee used to send one narrow query
("<title> <artist> 歌詞 中文翻譯") to Google (a JavaScript wall for HTTP
clients) and then to DuckDuckGo HTML (a bot challenge after a couple of
queries).  This module sends a prioritised list of queries, tries the
engines in turn, remembers which engine blocked this run (no retry storm),
keeps about one request per two seconds per host, and logs every query.
"""
from __future__ import annotations

import base64
import html as html_module
import json
import re
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

# music.163.com is never searched, queried or fetched.
BANNED_HOSTS = ("music.163.com",)

HOST_INTERVAL = 2.0
# After a block (CAPTCHA, JS wall, 429) an engine rests for the whole run.
ENGINE_ORDER = ("google", "duckduckgo", "yahoojp", "bing")
MAX_QUERIES = 16
# A blocked engine rests this long (seconds) and is then tried once more; Google, which
# always answers HTTP clients with a JavaScript wall, rests for the whole run.
BLOCK_COOLDOWN = 180.0
BLOCK_COOLDOWN_MAX = 900.0  # each repeated block doubles the rest, up to this
# Yahoo! JAPAN answers 429 to a burst of about fifty requests at 2 s spacing; it gets more room.
ENGINE_INTERVAL = {"yahoojp": 8.0}
# An engine that times out this many requests in a row is dropped for the run.
ENGINE_MAX_TIMEOUTS = 2

SITE_QUERIES = (
    "home.gamer.com.tw",
    "mojim.com",
    "uta-net.com",
    "utaten.com",
    "j-lyric.net",
    "lyrical-nonsense.com",
    "vocaloidlyrics.fandom.com",
)

# Result hosts that never carry lyrics text; skipping them saves page fetches.
_SKIP_RESULT_HOSTS = (
    "youtube.com", "youtu.be", "spotify.com", "music.apple.com", "apple.com", "amazon.", "facebook.com",
    "twitter.com", "x.com", "instagram.com", "tiktok.com", "awa.fm", "line.me", "wikipedia.org",
)
_ENGINE_HOST_MARKERS = {
    "google": ("google.",),
    "duckduckgo": ("duckduckgo.com",),
    "yahoojp": ("yahoo.co.jp", "yimg.jp", "lycorp.co.jp", "lycbiz.com", "yahoo-net.jp", "yahoo.com", "yahooapis.jp"),
    "bing": ("bing.com", "microsoft.com", "msn.com", "live.com"),
}

_TRACK_NUMBER_RE = re.compile(r"^\s*[\(\[（【]?\d{1,3}[\)\]）】]?\s*[.\-_・)]?\s*")
_BRACKETS_RE = re.compile(r"\s*[\(（\[【][^\)）\]】]*[\)）\]】]\s*")
_FEAT_RE = re.compile(r"\s*[\(（\[]?\s*(?:feat\.?|ft\.?|featuring)\s+[^\)）\]]*[\)）\]]?\s*$", re.IGNORECASE)


def result_priority(url: str) -> int:
    """Smaller sorts first: pages that publish lyrics or translated lyric posts before the rest."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    if host == "home.gamer.com.tw":
        return 0
    if host.endswith(("mojim.com", "uta-net.com", "utaten.com", "j-lyric.net", "lyrical-nonsense.com", "fandom.com")):
        return 1
    return 2


def not_a_lyric_page(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").casefold()
    return host.endswith("uta-net.com") and parsed.path.startswith(("/movie/", "/artist/", "/user/"))


def host_banned(url: str) -> bool:
    host = (urlparse(url).hostname or "").casefold()
    return any(host == banned or host.endswith("." + banned) for banned in BANNED_HOSTS)


# -- queries ---------------------------------------------------------------

def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_number(title: str) -> str:
    if re.match(r"^\s*[\(\[（【]?\d{1,3}[\)\]）】]?\s*[.\-_・)]\s*\S", title):
        return _TRACK_NUMBER_RE.sub("", title)
    return title


def typed_title(title: str) -> str:
    """The title as a person would type it: no track number, (feat ...) or brackets."""
    plain = _squash(_FEAT_RE.sub("", _strip_number(title)))
    return _squash(_BRACKETS_RE.sub(" ", plain)) or plain or _squash(title)


def title_forms_for_search(title: str) -> list[str]:
    """The title as typed, without track number / brackets / (feat ...),
    and in half-width and full-width spellings; deduped, most literal first."""
    forms: list[str] = []

    def add(text: str) -> None:
        text = _squash(text)
        if text and text not in forms:
            forms.append(text)

    add(title)
    add(_strip_number(title))
    add(typed_title(title))
    for form in list(forms):
        add(unicodedata.normalize("NFKC", form))
    for form in list(forms):
        add("".join(chr(ord(ch) + 0xFEE0) if "!" <= ch <= "~" else ch for ch in form))
    return forms


def clean_artist(artist: str) -> str:
    first = re.split(r"\s*[/／,，、;；]\s*", _BRACKETS_RE.sub(" ", artist or ""))[0]
    return _squash(first)


def query_variants(title: str, artist: str = "", *, limit: int = MAX_QUERIES) -> list[str]:
    """Queries in priority order, deduped, capped at ``limit``."""
    forms = title_forms_for_search(title)
    typed = typed_title(title)
    who = clean_artist(artist)
    queries: list[str] = []

    def add(query: str) -> None:
        query = _squash(query)
        if query and query not in queries:
            queries.append(query)

    add(f'"{typed}" 歌詞')
    if who:
        add(f"{typed} {who} 歌詞")
    add(f"{typed} 歌詞 翻譯")
    add(f"{typed} 中文歌詞")
    add(f"{typed} lyrics")
    # The query lyricsee has always sent: finds the 歌詞中文翻譯 posts.
    if who:
        add(f"{typed} {who} 歌詞 中文翻譯")
    add(f"{typed} 歌詞 中文翻譯")
    for site in SITE_QUERIES:
        add(f"site:{site} {typed}")
    # Spelling variants of the title for the two plain queries.
    for form in forms:
        if form == typed:
            continue
        add(f'"{form}" 歌詞')
        if who:
            add(f"{form} {who} 歌詞")
    return queries[:limit]


# -- parsing ---------------------------------------------------------------

class _Anchors(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.casefold() == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag.casefold() == "a" and self._href is not None:
            self.items.append((self._href, _squash("".join(self._text))))
            self._href = None


def _engine_host(host: str, engine: str) -> bool:
    return any(marker in host for marker in _ENGINE_HOST_MARKERS.get(engine, ()))


def _clean_result_urls(urls, engine: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if not url.startswith(("http://", "https://")) or url in seen or host_banned(url) or not_a_lyric_page(url):
            continue
        host = (urlparse(url).hostname or "").casefold()
        if _engine_host(host, engine) or any(host == s or host.endswith("." + s) or s in host for s in _SKIP_RESULT_HOSTS):
            continue
        seen.add(url)
        out.append(url)
    return out


def decode_bing_redirect(href: str) -> str:
    """Bing wraps results in /ck/a?...&u=a1<base64url of the target>."""
    href = html_module.unescape(href)
    values = parse_qs(urlparse(href).query).get("u")
    if values and values[0].startswith("a1"):
        payload = values[0][2:]
        payload += "=" * (-len(payload) % 4)
        try:
            return base64.urlsafe_b64decode(payload).decode("utf-8", "replace")
        except Exception:
            return ""
    return href


def parse_bing_results(document: str, *, must_contain: str = "") -> list[str]:
    """Organic result links.  Bing answers a query it finds nothing for with
    unrelated results; ``must_contain`` (the title, lowercased, no spaces)
    drops every block whose visible text does not mention it."""
    urls: list[str] = []
    for block in re.findall(r'<li class="b_algo".*?</li>', document, re.S):
        if must_contain:
            visible = re.sub(r"\s+", "", html_module.unescape(re.sub(r"<[^>]+>", "", block))).casefold()
            if must_contain not in visible:
                continue
        match = re.search(r'<a[^>]+href="([^"]+)"', block)
        if match:
            urls.append(decode_bing_redirect(match.group(1)))
    return _clean_result_urls(urls, "bing")


def parse_yahoojp_results(document: str) -> list[str]:
    parser = _Anchors()
    parser.feed(document)
    return _clean_result_urls([html_module.unescape(href) for href, _ in parser.items], "yahoojp")


def classify_block(engine: str, document: str, urls: list[str]) -> str | None:
    """'captcha' when the engine answered with a challenge or JS wall,
    'empty' when it answered normally with nothing, None when there are results."""
    low = document.lower()
    if engine == "google":
        if "enablejs" in low or "/sorry/" in low or "unusual traffic" in low or "captcha" in low:
            return "captcha"
    elif engine == "duckduckgo":
        if "anomaly-modal" in low or "challenge-form" in low or "bots use duckduckgo" in low:
            return "captcha"
    elif engine == "yahoojp":
        if "captcha" in low or "認証" in document[:5000] and not urls:
            return "captcha"
    elif engine == "bing":
        if "captcha" in low or "/turing/" in low:
            return "captcha"
    return None if urls else "empty"


# -- the session -----------------------------------------------------------

def _engine_request(engine: str, query: str) -> tuple[str, dict, dict]:
    headers = {"User-Agent": "", "Accept-Language": "zh-TW,ja;q=0.8"}
    if engine == "google":
        return "https://www.google.com/search", {"q": query, "udm": "14"}, headers
    if engine == "duckduckgo":
        return "https://html.duckduckgo.com/html/", {"q": query}, headers
    if engine == "yahoojp":
        return "https://search.yahoo.co.jp/search", {"p": query}, headers
    return "https://www.bing.com/search", {"q": query, "setlang": "ja"}, headers


def _title_probe(query: str) -> str:
    """The quoted/leading title of a query, as Bing relevance text."""
    match = re.search(r'"([^"]+)"', query)
    text = match.group(1) if match else re.sub(r"^site:\S+\s+", "", query)
    text = re.split(r"\s+(?:歌詞|歌词|lyrics|中文歌詞)\b", text)[0]
    return re.sub(r"\s+", "", text).casefold()


class SearchSession:
    """Per-run search state: blocked engines, throttling, the query log."""

    def __init__(self, client, *, engines=ENGINE_ORDER, interval: float | None = None,
                 log_path: str | Path | None = None, user_agent: str | None = None, sleep=time.sleep,
                 clock=time.monotonic) -> None:
        self.client = client
        self.engines = tuple(engines)
        # Real HTTP clients wait HOST_INTERVAL between requests to one host;
        # test doubles (no search_interval attribute) do not wait.
        self.interval = interval if interval is not None else getattr(client, "search_interval", 0.0)
        self.log_path = Path(log_path) if log_path else None
        self.user_agent = user_agent
        self.sleep = sleep
        self.clock = clock
        self.blocked: dict[str, str] = {}
        self.timeouts: dict[str, int] = {}
        self.blocked_at: dict[str, float] = {}
        self.block_count: dict[str, int] = {}
        self.records: list[dict] = []
        self.errors: dict[str, str] = {}
        self.last_url = ""
        self.context = ""
        self._last_by_host: dict[str, float] = {}

    def _throttle(self, host: str, engine: str = "") -> None:
        interval = max(self.interval, ENGINE_INTERVAL.get(engine, 0.0)) if self.interval else 0.0
        wait = interval - (self.clock() - self._last_by_host.get(host, -1e9))
        if wait > 0:
            self.sleep(wait)
        self._last_by_host[host] = self.clock()

    def _log(self, record: dict) -> None:
        record["context"] = self.context
        self.records.append(record)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def search(self, engine: str, query: str) -> list[str]:
        """One request.  Returns result URLs; marks the engine blocked on a challenge."""
        url, params, headers = _engine_request(engine, query)
        headers["User-Agent"] = self.user_agent or _default_ua()
        host = urlparse(url).hostname or ""
        record = {"engine": engine, "query": query, "url": url + "?" + urlencode(params)}
        self.last_url = record["url"]
        self._throttle(host, engine)
        status: object = 200
        document = ""
        try:
            document = self.client.get_text(url, params=params, headers=headers, retries=1)
        except Exception as exc:  # noqa: BLE001 - every failure is data here
            status = getattr(exc, "code", None) or type(exc).__name__
            record.update(status=status, outcome="error", detail=str(exc)[:120], results=[])
            self.errors[engine] = str(exc)
            if status in (429, 403):
                self._block(engine, f"error:{status}")
            elif "timed out" in str(exc) or status in ("TimeoutError", "URLError"):
                self.timeouts[engine] = self.timeouts.get(engine, 0) + 1
                if self.timeouts[engine] >= ENGINE_MAX_TIMEOUTS:
                    self._block(engine, "timeouts")
            self._log(record)
            return []
        self.timeouts[engine] = 0
        if engine == "bing":
            urls = parse_bing_results(document, must_contain=_title_probe(query))
        elif engine == "yahoojp":
            urls = parse_yahoojp_results(document)
        elif engine == "duckduckgo":
            urls = _clean_result_urls(_ddg_urls(document), engine)
        else:
            urls = _clean_result_urls(_google_urls(document), engine)
        outcome = classify_block(engine, document, urls)
        if outcome == "captcha":
            self._block(engine, "captcha")
        record.update(status=status, outcome=outcome or "ok", bytes=len(document), results=urls)
        self._log(record)
        return urls

    def _block(self, engine: str, reason: str) -> None:
        self.blocked[engine] = reason
        self.blocked_at[engine] = self.clock()
        self.block_count[engine] = self.block_count.get(engine, 0) + 1

    def _cooldown(self, engine: str) -> float:
        return min(BLOCK_COOLDOWN * 2 ** (self.block_count.get(engine, 1) - 1), BLOCK_COOLDOWN_MAX)

    def available(self) -> list[str]:
        for engine in list(self.blocked):
            if engine != "google" and self.clock() - self.blocked_at.get(engine, 0.0) >= self._cooldown(engine):
                del self.blocked[engine]
                self.timeouts[engine] = ENGINE_MAX_TIMEOUTS - 1
        return [engine for engine in self.engines if engine not in self.blocked]

    def find_urls(self, query: str) -> tuple[list[str], str]:
        """Engines in order until one returns results; blocked ones are skipped."""
        for engine in self.available():
            urls = self.search(engine, query)
            if urls:
                return urls, engine
        return [], "none"


def _default_ua() -> str:
    try:
        from lyricsee_core import SEARCH_USER_AGENT
    except ModuleNotFoundError:  # pragma: no cover - package import style
        from lyricsee.lyricsee_core import SEARCH_USER_AGENT
    return SEARCH_USER_AGENT


def _ddg_urls(document: str) -> list[str]:
    try:
        from lyricsee_core import parse_duckduckgo_results
    except ModuleNotFoundError:  # pragma: no cover - package import style
        from lyricsee.lyricsee_core import parse_duckduckgo_results
    return parse_duckduckgo_results(document)


def _google_urls(document: str) -> list[str]:
    try:
        from lyricsee_core import parse_google_results
    except ModuleNotFoundError:  # pragma: no cover - package import style
        from lyricsee.lyricsee_core import parse_google_results
    return parse_google_results(document)


def session_for(client, **kwargs) -> SearchSession:
    """One SearchSession per HTTP client, so a blocked engine stays blocked for the run."""
    session = getattr(client, "_lyricsee_web_search", None)
    if session is None:
        session = SearchSession(client, **kwargs)
        try:
            setattr(client, "_lyricsee_web_search", session)
        except (AttributeError, TypeError):
            pass
    return session
