"""Map lyricsee's parenthesized-block flags onto Review page rows.

lyricsee lists lyrics-file line numbers of parenthesized blocks that may not be
sung (``paren_flags`` in the song's sources.json, or a ``<stem>.paren_flags.json``
sidecar beside the audio).  The Review page only shows them: no tier, ordering,
review-required or alignment logic reads this.  Rows are matched to lyrics
lines by text, in order (difflib), like the rest of the page maps lyrics.
Missing, stale (the lyrics file's hash differs) or unreadable flag data
gives no marks.

Where the flags are looked up (first file whose hash matches the lyrics wins):
  1. ``<audio stem>.paren_flags.json`` beside the audio (optional)
  2. ``report/<audio stem>.sources.json`` of each lyricsee output root: roots in
     ``LRC_PAREN_FLAG_ROOTS`` (os.pathsep separated), ``LYRICSEE_OUTPUT_ROOT``,
     lyricsee's default root (the lyricsee folder next to this scripts folder),
     and the parent of a ``txt`` folder the lyrics file sits in
The lyrics file is the text file beside the audio, never the .lrc.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import pathlib
import unicodedata

LYRIC_SUFFIXES = (".lyrics.txt", ".txt", ".lyrics")  # auto_lrc.find_lyrics order without the timed .lyrics.lrc: flags refer to text lyrics
CLASSES = {"official-only": "官方歌詞有、聽得到的版本沒有", "unconfirmed": "官方歌詞括號區塊，沒有可比對的來源"}


def _key(text):
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _original(line):
    return line.split("  ", 1)[0]


def _text_hash(raw):
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()  # same as lyricsee/paren_flags.text_sha256


def output_roots(extra=()):
    """lyricsee output roots to look in (see module docstring)."""
    roots = [pathlib.Path(p) for p in extra]
    for var in ("LRC_PAREN_FLAG_ROOTS", "LYRICSEE_OUTPUT_ROOT"):
        roots += [pathlib.Path(p) for p in os.environ.get(var, "").split(os.pathsep) if p.strip()]
    roots.append(pathlib.Path(__file__).resolve().parent.parent / "lyricsee")
    return roots


def _load_flags(audio, lyrics_path, lyrics_bytes, extra_roots=()):
    audio = pathlib.Path(audio)
    sha = _text_hash(lyrics_bytes)
    candidates = [audio.with_name(audio.stem + ".paren_flags.json")]
    roots = output_roots(extra_roots)
    if lyrics_path.parent.name.casefold() == "txt":
        roots.append(lyrics_path.parent.parent)
    for root in roots:
        candidates.append(root / "report" / f"{audio.stem}.sources.json")
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        flags = payload.get("paren_flags") if isinstance(payload, dict) else None
        if not isinstance(flags, dict):
            continue
        if (flags.get("txt_sha256") or payload.get("txt_sha256")) == sha:
            return flags
    return None


def row_marks(audio, row_lyrics, extra_roots=()):
    """``{row index: class}`` for rows whose lyrics sit in a flagged block.

    ``row_lyrics`` is one list of LRC lyric lines per review row, in order.
    """
    try:
        audio = pathlib.Path(audio)
        lyrics_path = next((audio.with_name(audio.stem + s) for s in LYRIC_SUFFIXES
                            if audio.with_name(audio.stem + s).is_file()), None)
        if lyrics_path is None:
            return {}
        raw = lyrics_path.read_bytes()
        flags = _load_flags(audio, lyrics_path, raw, extra_roots)
        if not flags or not flags.get("blocks"):
            return {}
        file_lines = [_original(line) for line in raw.decode("utf-8-sig").splitlines()]
        wanted = {}
        for block in flags["blocks"]:
            if block.get("class") in CLASSES:
                for number in block.get("lines") or []:
                    if isinstance(number, int) and 1 <= number <= len(file_lines):
                        wanted[number - 1] = block["class"]
        if not wanted:
            return {}
        flat = [(index, _key(text)) for index, lines in enumerate(row_lyrics) for text in lines]
        matcher = difflib.SequenceMatcher(None, [_key(line) for line in file_lines], [key for _, key in flat], autojunk=False)
        marks = {}
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                continue
            for offset in range(i2 - i1):
                klass = wanted.get(i1 + offset)
                if klass:
                    marks.setdefault(flat[j1 + offset][0], klass)
        return marks
    except (OSError, ValueError, UnicodeError, TypeError, AttributeError):
        return {}
