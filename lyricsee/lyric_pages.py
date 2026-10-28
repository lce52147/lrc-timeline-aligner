"""Lyric rows from pages the two-column extractor misses: timestamped LRC blocks (kugeci and similar),
lyric containers (genius ``data-lyrics-container``), and a plain run of short lines.

Everything here works on text, so the same functions serve the static HTML fetched over plain HTTP and the
rendered ``innerText`` read through the DevTools tab (cdp_search.CdpGoogle.fetch_page).  Callers decide
whether a page is about the song; these functions only return rows."""
from __future__ import annotations

import html as html_module
import re
from html.parser import HTMLParser

try:
    from lyricsee_core import (  # type: ignore
        LyricPair, _KANA_RE, _HAN_RE, _is_credit_line, _looks_like_non_lyric_web_text, _looks_like_original,
    )
except ModuleNotFoundError:  # pragma: no cover - package import style
    from lyricsee.lyricsee_core import (
        LyricPair, _KANA_RE, _HAN_RE, _is_credit_line, _looks_like_non_lyric_web_text, _looks_like_original,
    )

_TIMED_RE = re.compile(r"^\[(\d{1,3}):(\d{1,2}(?:[.:]\d{1,3})?)\]\s*(.*)$")
_SECTION_RE = re.compile(r"^\[[^\]\d][^\]]{0,40}\]$|^\([^)]{1,30}\)$")
_PAGE_NOISE_RE = re.compile(
    r"contributors?\b|embed\b|you might also like|read more|translations?\b|lyrics$|歌詞$|歌词$|^作[词詞曲]|^编曲|^編曲",
    re.IGNORECASE,
)
_NO_LYRICS_RE = re.compile(
    r"do not have these lyrics|no lyrics (?:found|available)|歌詞はまだ登録|歌詞が見つかりません|暂无歌词|暫無歌詞|尚無歌詞",
    re.IGNORECASE,
)
MAX_ROW = 90        # a lyric row; longer lines are prose
MIN_RUN = 8


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", html_module.unescape(line)).strip()


_STAMP_ANYWHERE_RE = re.compile(r"(\[\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?\])")


def _lines(text: str) -> list[str]:
    # a timestamp always starts a row, also when the page puts other text right in front of it
    text = _STAMP_ANYWHERE_RE.sub(r"\n\1", text.replace("\r", "\n"))
    return [line for line in (_clean(raw) for raw in text.split("\n")) if line]


def _seconds(minutes: str, rest: str) -> float:
    return int(minutes) * 60 + float(rest.replace(":", "."))


def timed_blocks(lines: list[str]) -> list[list[tuple[float, str]]]:
    """Runs of ``[mm:ss.xx]text`` lines; a new run starts when the clock goes back (second language block)."""
    blocks: list[list[tuple[float, str]]] = []
    current: list[tuple[float, str]] = []
    for line in lines:
        match = _TIMED_RE.match(line)
        if not match:
            continue
        stamp = _seconds(match.group(1), match.group(2))
        text = match.group(3).strip()
        if current and stamp < current[-1][0] - 0.5:
            blocks.append(current)
            current = []
        if text:
            current.append((stamp, text))
    if current:
        blocks.append(current)
    return [block for block in blocks if len(block) >= 2]


def _original_share(block: list[tuple[float, str]]) -> float:
    return sum(_looks_like_original(text) for _, text in block) / max(1, len(block))


def timed_pairs(lines: list[str]) -> list:
    """Original (+ translation, paired by timestamp) rows from LRC-looking lines."""
    blocks = timed_blocks(lines)
    if not blocks:
        return []
    first = blocks[0]
    if len(blocks) == 1:
        return [LyricPair(text) for _, text in first]
    # the block with the most original-language rows is the original; the next Chinese-only block translates it
    original = max(blocks, key=lambda block: (_original_share(block), len(block)))
    others = [block for block in blocks if block is not original and _original_share(block) < 0.5]
    if not others or _original_share(original) < 0.5:
        return [LyricPair(text) for _, text in first]
    translation = {round(stamp, 1): text for stamp, text in others[0]}
    rows = [(stamp, text, translation.get(round(stamp, 1))) for stamp, text in original]
    matched = [index for index, row in enumerate(rows) if row[2]]
    if len(matched) >= 2:
        # rows before the first translated one are the credit header ("作词/作曲/..." with a clock)
        rows = rows[matched[0]:]
    return [LyricPair(text, translated) for _, text, translated in rows]


def _is_furniture_word(line: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{0,14}", line))


def run_of_lyric_lines(lines: list[str]) -> list:
    """Longest run of short, lyric-looking lines (no menus, credits, section tags); [] when under MIN_RUN."""
    best: list[str] = []
    current: list[str] = []
    for line in lines:
        keep = (
            0 < len(line) <= MAX_ROW
            and not _TIMED_RE.match(line)
            and not _SECTION_RE.match(line)
            and not _PAGE_NOISE_RE.search(line)
            and not _is_credit_line(line)
            and not _looks_like_non_lyric_web_text(line)
            and (_KANA_RE.search(line) or _HAN_RE.search(line) or re.search(r"[A-Za-z]{2,}", line))
        )
        if keep:
            current.append(line)
            continue
        if len(current) > len(best):
            best = current
        current = []
    if len(current) > len(best):
        best = current
    # menu words ("Login", "Artist") sit right next to the lyric block: trim single-word edges
    while best and _is_furniture_word(best[0]):
        best = best[1:]
    while best and _is_furniture_word(best[-1]):
        best = best[:-1]
    return [LyricPair(line) for line in best] if len(best) >= MIN_RUN else []


def container_rows(containers: list[str]) -> list:
    """Rows from the text of lyric containers (genius): section tags and page furniture dropped."""
    best: list = []
    for text in containers:
        rows = [LyricPair(line) for line in _lines(text)
                if not _SECTION_RE.match(line) and not _PAGE_NOISE_RE.search(line) and len(line) <= 300]
        if len(rows) > len(best):
            best = rows
    return best


def drop_title_rows(rows: list, title: str) -> list:
    """A page repeats the song title (often "title (feat. x)") above the lyric; drop it from the first rows."""
    key = re.sub(r"\W+", "", title).casefold()
    if not key:
        return rows
    start = 0
    while start < min(2, len(rows)):
        text = re.sub(r"\W+", "", rows[start].original).casefold()
        if key in text and len(text) <= len(key) + 14:
            start += 1
        else:
            break
    # "title" is often followed by the artist's name alone on a line
    if start and start < len(rows) and re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{1,24}", rows[start].original.strip()):
        start += 1
    return rows[start:]


def rows_from_text(text: str, containers: list[str] | None = None) -> tuple[list, str]:
    """Best rows for a page's visible text; returns (rows, how) with how in timed/container/run/''."""
    lines = _lines(text)
    options = []
    timed = timed_pairs(lines)
    if len(timed) >= MIN_RUN:
        options.append((timed, "timed"))
    contained = container_rows(containers or [])
    if len(contained) >= MIN_RUN:
        options.append((contained, "container"))
    if not options and not _NO_LYRICS_RE.search(text):
        run = run_of_lyric_lines(lines)
        if run and len({row.original for row in run}) >= 0.6 * len(run):
            options.append((run, "run"))
    if not options:
        return [], ""
    # a precise source (timestamps, lyric container) wins over a longer but looser one
    return options[0]


class _ContainerParser(HTMLParser):
    """Text of ``data-lyrics-container`` blocks in static HTML; <br> is a line break."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.blocks: list[list[str]] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        names = {key for key, _ in attrs}
        if tag == "div" and "data-lyrics-container" in names and not self.depth:
            self.depth = 1
            self.blocks.append([])
        elif self.depth:
            if tag == "div":
                self.depth += 1
            elif tag == "br":
                self.blocks[-1].append("\n")
            elif tag in ("script", "style"):
                self._skip += 1

    def handle_endtag(self, tag):
        if self.depth:
            if tag == "div":
                self.depth -= 1
            elif tag in ("script", "style") and self._skip:
                self._skip -= 1

    def handle_data(self, data):
        if self.depth and not self._skip:
            self.blocks[-1].append(data)


def rows_from_document(document: str) -> tuple[list, str]:
    """Rows from static HTML: genius lyric containers first, then timestamped blocks, then a plain run."""
    from_html = _ContainerParser()
    from_html.feed(document)
    containers = ["".join(block) for block in from_html.blocks]
    try:
        from lyricsee_core import _VisibleTextParser  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - package import style
        from lyricsee.lyricsee_core import _VisibleTextParser
    parser = _VisibleTextParser()
    parser.feed(document)
    return rows_from_text("".join(parser.parts), containers)
