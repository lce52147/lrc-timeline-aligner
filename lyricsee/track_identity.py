"""Who a recording is, from what the file itself says: its tags and its album folder.

Every FLAC carries artist and album tags and sits in an album folder; searching with the bare file stem
("01. Islet - 52Hz") and then asking a person who sang it throws that away.  This module turns the tags
(title, artist, album artist, album, composer, lyricist), the file stem and the folder names into

* title aliases: the tag title, the stem without track number and artist prefix/suffix, the titles quoted
  in the folder name (ED「ラブソング」), the two halves of "直到下次再见（Dasvidaniya）";
* artist names: the tag artist split into singers, character and voice actor ("Teresa Wagner (CV: ...)"),
  plus the singer and CV names the folder name spells (テレサ・ワーグナー(CV.石見舞菜香));
* work titles: the 「...」/《...》 contents of the folder that are not the song (TVアニメ「多田くんは恋をしない」);
* gate tokens: the strings a web page must name (any of them) to count as this recording's page.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

MAX_ALIASES = 6
MAX_ARTISTS = 6

_QUOTED_RE = re.compile(r"[「『《](.+?)[」』》]")
# A quoted name right after one of these is the song (ED「ラブソング」), otherwise the work (TVアニメ「...」).
_SONG_ROLE_RE = re.compile(
    r"(?:\bED|\bOP|\bIN|挿入歌|主題歌|エンディング|オープニング|テーマ|ソング|イメージソング|キャラクターソング|劇中歌)\s*[:：]?\s*$",
    re.IGNORECASE,
)
_BRACKET_TAG_RE = re.compile(r"\[[^\]]*\]")
_FORMAT_TAIL_RE = re.compile(
    r"\s*[\(\[（]?\s*(?:flac|ape|cue|mp3|wav|png|bk|scans?|hi-?res|web|eac|\d+\s*k?hz|\d+\s*bit)\b.*$",
    re.IGNORECASE,
)
_RIP_WORDS = frozenset({"flac", "web", "ape", "cue", "mp3", "wav", "png", "bk", "scans", "eac", "hires", "cd", "ost"})
_AFTER_QUOTE_RE =re.compile(r"^\s*[-－–—―／/]\s*(.+)$")
_NAME_SPLIT_RE = re.compile(
    r"\s*(?:,|，|、|/|／|&|＆|×|;|；|\bfeat\.?\s|\bft\.?\s|\bfeaturing\s|\bwith\s)\s*", re.IGNORECASE
)
_CV_RE = re.compile(r"[（(]\s*(?:cv|c\.v\.|voice|声)?\s*[.:：\-－]?\s*([^（()）]+?)\s*[)）]", re.IGNORECASE)
_PAREN_RE = re.compile(r"\s*[（(]([^（()）]*)[)）]\s*")
_TITLE_ARTIST_SEP_RE = re.compile(r"\s+[-–—―]\s+|\s*—\s*")
_DISC_FOLDER_RE = re.compile(r"^(?:disc|cd|vol)\b|\bvol\.?\s*\d|^original\s+soundtrack\b", re.IGNORECASE)
_KANA_RE = re.compile(r"[぀-ヿ]")
_HAN_RE = re.compile(r"[㐀-鿿]")
_LATIN_DIACRITIC_RE = re.compile(r"[À-ɏ]")


def norm(value: str) -> str:
    """Comparison key (same folding as lyricsee_core._normalized): NFKC, diacritics dropped, no punctuation."""
    folded = unicodedata.normalize("NFKC", value or "")
    # accented Latin letters only: NFD on the whole string would split kana dakuten
    folded = _LATIN_DIACRITIC_RE.sub(
        lambda m: "".join(ch for ch in unicodedata.normalize("NFD", m.group()) if not unicodedata.combining(ch)),
        folded,
    )
    return re.sub(r"[^\w]+", "", folded, flags=re.UNICODE).casefold()


def _dedupe(values, *, key=norm, limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        k = key(value)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(value)
        if limit and len(out) >= limit:
            break
    return out


def split_names(value: str) -> list[str]:
    """Display names in a credit string; the voice actor of a CV credit is a name of its own.

    "Teresa Wagner (CV: Manaka Iwami)" -> ["Teresa Wagner", "Manaka Iwami"];
    "鸣潮先约电台/Crywolf/Cxssidy" -> the three of them.
    """
    names: list[str] = []
    for raw in _NAME_SPLIT_RE.split(value or ""):
        raw = raw.strip()
        if not raw:
            continue
        outer = _CV_RE.sub(" ", raw).strip()
        inner = [match.strip() for match in _CV_RE.findall(raw)]
        for piece in [outer, *inner]:
            if len(norm(piece)) >= 2:
                names.append(piece)
    return _dedupe(names)


def folder_info(name: str) -> dict:
    """{'work_titles', 'song_titles', 'artist_names'} read from an album folder name.

    "[180523] TVアニメ「多田くんは恋をしない」ED「ラブソング」-テレサ・ワーグナー(CV.石見舞菜香) FLAC+BK"
    -> work 多田くんは恋をしない, song ラブソング, artists テレサ・ワーグナー and 石見舞菜香.
    """
    text = _BRACKET_TAG_RE.sub(" ", name or "")
    works: list[str] = []
    songs: list[str] = []
    matches = list(_QUOTED_RE.finditer(text))
    for match in matches:
        before = text[max(0, match.start() - 10):match.start()]
        (songs if _SONG_ROLE_RE.search(before) else works).append(match.group(1).strip())
    artists: list[str] = []
    if matches:
        tail = _AFTER_QUOTE_RE.match(text[matches[-1].end():])
        if tail:
            artists = split_names(_FORMAT_TAIL_RE.sub("", tail.group(1)).strip())
    return {
        "work_titles": _dedupe(works),
        "song_titles": _dedupe(songs),
        "artist_names": artists,
    }


def folder_context(path: Path) -> dict:
    """folder_info of the album folder, plus the work titles quoted in its parent when the album folder is
    only a disc / "Original Soundtrack Vol.1" subfolder."""
    info = folder_info(path.parent.name)
    grand = path.parent.parent.name if path.parent.parent != path.parent else ""
    if grand and not info["work_titles"] and (_DISC_FOLDER_RE.search(path.parent.name) or not info["song_titles"]):
        upper = folder_info(grand)
        info["work_titles"] = _dedupe(info["work_titles"] + upper["work_titles"])
    return info


def _stem_title_parts(stem_title: str, artists: list[str]) -> list[str]:
    """The stem with a leading "artist - " or trailing " - artists" removed."""
    parts = [part.strip() for part in _TITLE_ARTIST_SEP_RE.split(stem_title) if part.strip()]
    if len(parts) < 2:
        return []
    artist_keys = [norm(name) for name in artists if len(norm(name)) >= 2]

    def is_artist(part: str) -> bool:
        pieces = split_names(part) or [part]
        return all(
            any(key == norm(piece) or (len(key) >= 3 and len(norm(piece)) >= 3 and (key in norm(piece) or norm(piece) in key))
                for key in artist_keys)
            for piece in pieces
        )

    kept = [part for part in parts if not is_artist(part)]
    return kept if kept and len(kept) < len(parts) else []


def _typed(title: str) -> str:
    """Title without (feat ...) and bracketed additions, as a person would type it."""
    plain = re.sub(r"\s*[\(（\[]?\s*(?:feat\.?|ft\.?|featuring)\s+[^\)）\]]*[\)）\]]?\s*$", "", title, flags=re.IGNORECASE)
    plain = re.sub(r"\s*[\(（\[【][^\)）\]】]*[\)）\]】]\s*", " ", plain)
    return re.sub(r"\s+", " ", plain).strip() or title.strip()


def title_aliases(tag_title: str, stem_title: str, artists: list[str], song_titles: list[str]) -> list[str]:
    """Spellings of the song's title that a web page may carry, most literal first."""
    candidates: list[str] = []
    if tag_title:
        candidates += [tag_title, _typed(tag_title)]
    for part in _stem_title_parts(stem_title, artists):
        candidates += [part, _typed(part)]
    if stem_title and not _stem_title_parts(stem_title, artists):
        candidates += [stem_title, _typed(stem_title)]
    candidates += song_titles
    # "直到下次再见（Dasvidaniya）": the title and the reading/translation in brackets are each findable
    for value in list(candidates):
        for inner in _PAREN_RE.findall(value):
            if len(norm(inner)) >= 3 and not re.match(r"\s*(?:feat\.?|ft\.?)\b", inner, re.IGNORECASE):
                candidates.append(inner)
    return _dedupe(candidates, limit=MAX_ALIASES)


def build_identity_fields(
    path: Path,
    *,
    title: str,
    artist: str,
    album: str = "",
    albumartist: str = "",
) -> dict:
    """aliases / artist_names / work_titles for TrackIdentity, from tags + stem + album folder."""
    folder = folder_context(path)
    names = _dedupe(
        split_names(artist) + split_names(albumartist) + folder["artist_names"],
        limit=MAX_ARTISTS,
    )
    stem = path.stem.strip()
    stem_title = re.sub(r"^\s*(?:[\(\[（【]?\d{1,3}(?:[-.]\d{1,3})?[\)\]）】]?\s*[.\-_ ]+\s*)", "", stem).strip() or stem
    aliases = title_aliases(title, stem_title, names, folder["song_titles"])
    return {
        "aliases": tuple(aliases),
        "artist_names": tuple(names),
        "work_titles": tuple(folder["work_titles"]),
    }


def gate_tokens(identity) -> list[str]:
    """Normalized strings of which any one, found on a page, confirms the page is about this recording:
    artist / singer / CV names, the album, the folder's work titles and the pieces of the album folder.
    Never the song's own title (many songs share one)."""
    alias_keys = [norm(identity.title)] + [norm(alias) for alias in getattr(identity, "aliases", ())]
    alias_keys = [key for key in alias_keys if key]

    def is_title(token: str) -> bool:
        return any(token == key or (len(key) >= 3 and key in token) for key in alias_keys)

    tokens: list[str] = []

    def add(value: str, *, minimum: int = 2) -> None:
        token = norm(value)
        if len(token) >= minimum and token not in tokens and not is_title(token):
            tokens.append(token)

    for name in split_names(identity.artist) + list(getattr(identity, "artist_names", ())):
        add(name)
    add(getattr(identity, "albumartist", ""))
    add(identity.album, minimum=3)
    for work in getattr(identity, "work_titles", ()):
        add(work, minimum=3)
    for hint in identity.hints:
        # "[WEB-FLAC]", "(2023.06.21)[Hi-Res FLAC]", "FLAC+BK" describe the rip, not the music
        hint = _FORMAT_TAIL_RE.sub("", _BRACKET_TAG_RE.sub(" ", hint)).strip()
        for piece in re.split(r"\s[-–—]\s|[-–—―/／、,，]", hint):
            if norm(piece) not in _RIP_WORDS:
                add(piece, minimum=3)
    return tokens


def identity_evidence(identity, text: str) -> list[str]:
    """The gate tokens ``text`` (a page title + body) names; empty means the page names nothing of this recording."""
    page = norm(text)
    return [token for token in gate_tokens(identity) if token in page]


def looks_chinese(identity) -> bool:
    """A Han-only title without any kana in the title, artist or album: a Chinese song, search 歌词."""
    texts = [identity.title, identity.artist, identity.album, *getattr(identity, "aliases", ())]
    if any(_KANA_RE.search(text or "") for text in texts):
        return False
    return any(_HAN_RE.search(text or "") for text in texts)
