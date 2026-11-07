"""Flag parenthesized lyric blocks that may not be sung.

Official lyric sites print backing vocals and unrecorded lines in parentheses,
sometimes as a block of several lines.  Those lines are information only: they
are never removed or changed, only listed in ``paren_flags`` of the song's
sources.json so the Review page can mark them quietly.

A block is a run of lines from an opening ``(`` / ``（`` at the start of a line
to the closing ``)`` / ``）`` that ends a line (a single fully parenthesized
line is a block of one).  Parentheses inside an otherwise normal line are not
a block.  An opening paren that never closes within MAX_BLOCK_LINES lines is
reported as an ``unclosed`` block that ends at its own line.

Classes:
  official-only  an independent timed source (LRCLIB) exists and lacks most of
                 the block's lines ("lacks": fewer than half of the block's
                 countable lines are found in that source)
  unconfirmed    no such source to compare
  (a block the timed source has is not flagged)
"""
from __future__ import annotations

try:
    from lyric_compare import canonical_text, is_filler
except ModuleNotFoundError:  # pragma: no cover - package import style
    from lyricsee.lyric_compare import canonical_text, is_filler

VERSION = 1
OPEN = "(（"
CLOSE = ")）"
MAX_BLOCK_LINES = 8
LACK_BELOW = 0.5
MIN_CONTAINED = 6
TIMED_SOURCES = ("lrclib",)
_FRAGMENT_KINDS = {"missing_in_b", "word"}


def original_of(line: str) -> str:
    """The lyric part of a lyrics.txt line (translation follows two spaces)."""
    return line.split("  ", 1)[0]


def _scan(text: str, depth: int) -> tuple[int, str, int]:
    """Walk one line. Returns (depth after, text outside parens, closes at depth 0)."""
    outside: list[str] = []
    zero_closes = 0
    for ch in text:
        if ch in OPEN:
            depth += 1
        elif ch in CLOSE:
            if depth > 0:
                depth -= 1
                if depth == 0:
                    zero_closes += 1
        elif depth == 0:
            outside.append(ch)
    return depth, "".join(outside), zero_closes


def _blank(outside: str) -> bool:
    return not canonical_text(outside, drop_parens=False) and not outside.strip(" \t　")


def detect_blocks(originals: list[str]) -> list[dict]:
    """Parenthesized blocks as ``{start, end, unclosed}`` (1-based line numbers)."""
    blocks: list[dict] = []
    i = 0
    n = len(originals)
    while i < n:
        text = originals[i].strip()
        if not text or text[0] not in OPEN:
            i += 1
            continue
        depth, outside, _ = _scan(text, 0)
        if depth == 0:
            if _blank(outside):
                blocks.append({"start": i + 1, "end": i + 1, "unclosed": False})
            i += 1
            continue
        end = None
        for j in range(i + 1, min(n, i + MAX_BLOCK_LINES)):
            depth, outside, _ = _scan(originals[j], depth)
            if depth == 0:
                end = j if _blank(outside) else -1
                break
        if end is None:
            blocks.append({"start": i + 1, "end": i + 1, "unclosed": True})
            i += 1
        elif end == -1:
            i += 1
        else:
            blocks.append({"start": i + 1, "end": end + 1, "unclosed": False})
            i = end + 1
    return blocks


def _lacks_by_text(block_lines: list[str], timed_lines: list[str]) -> bool:
    timed = [canonical_text(line, drop_parens=False) for line in timed_lines]
    countable = [
        canon
        for canon in (canonical_text(line, drop_parens=False) for line in block_lines)
        if not is_filler(canon, min_length=1)
    ]
    if not countable:
        return False
    # A short line must equal a timed line (a bare "hey" sits inside many others);
    # a longer one may also sit inside a timed line that merged it with its neighbour.
    present = sum(1 for canon in countable if canon in timed or (len(canon) >= MIN_CONTAINED and any(canon in t for t in timed)))
    return present / len(countable) < LACK_BELOW


def _lacks_by_diff_rows(numbers: list[int], originals: list[str], rows: list[dict]) -> bool:
    """Same test from a saved lrclib cross-source entry (its diff_lines)."""
    absent: set[int] = set()
    for row in rows:
        if row.get("kind") not in _FRAGMENT_KINDS:
            continue
        number = row.get("line")
        if number not in numbers:
            continue
        fragment = str(row.get("text") or "").rstrip("…")
        canon = canonical_text(fragment, drop_parens=False)
        if canon and canon in canonical_text(originals[number - 1], drop_parens=False):
            absent.add(number)
    countable = [
        n for n in numbers
        if not is_filler(canonical_text(originals[n - 1], drop_parens=False), min_length=1)
    ]
    if not countable:
        return False
    return 1 - len(absent & set(countable)) / len(countable) < LACK_BELOW


def compute_flags(
    originals: list[str],
    *,
    timed_lines: list[str] | None = None,
    cross_sources: list[dict] | None = None,
) -> dict:
    """The ``paren_flags`` payload for a lyrics file.

    ``timed_lines`` are the lyric lines of an independent timed source (live
    run).  Without them, a saved lrclib entry of ``cross_sources`` is used
    (offline re-flagging); with neither, every block is ``unconfirmed``.
    """
    saved = next(
        (c for c in (cross_sources or []) if c.get("source") in TIMED_SOURCES and c.get("independent", True)),
        None,
    )
    basis = "none"
    if timed_lines:
        basis = "timed-lines"
    elif saved is not None:
        basis = "saved-diff"
    flagged: list[dict] = []
    for block in detect_blocks(originals):
        numbers = list(range(block["start"], block["end"] + 1))
        texts = [originals[n - 1] for n in numbers]
        if basis == "timed-lines":
            klass = "official-only" if _lacks_by_text(texts, timed_lines or []) else None
        elif basis == "saved-diff":
            rows = saved.get("diff_lines") or []
            klass = "official-only" if _lacks_by_diff_rows(numbers, originals, rows) else None
        else:
            klass = "unconfirmed"
        if klass:
            flagged.append({
                "start": block["start"],
                "end": block["end"],
                "lines": numbers,
                "class": klass,
                "unclosed": block["unclosed"],
            })
    return {
        "version": VERSION,
        "basis": basis,
        "blocks": flagged,
        "block_count": len(flagged),
        "line_count": sum(len(b["lines"]) for b in flagged),
    }


def flags_for_lyrics_text(lyrics_text: str, **kwargs) -> dict:
    originals = [original_of(line) for line in lyrics_text.splitlines()]
    return compute_flags(originals, **kwargs)


def _fetch_timed_lines(audio) -> list[str] | None:
    """Lyric lines of the LRCLIB match for ``audio`` (one request); None when there is none."""
    from pathlib import Path

    try:
        from lyricsee_core import HttpClient, fetch_lrclib_candidate, probe_audio
    except ModuleNotFoundError:  # pragma: no cover - package import style
        from lyricsee.lyricsee_core import HttpClient, fetch_lrclib_candidate, probe_audio
    try:
        candidate = fetch_lrclib_candidate(probe_audio(Path(audio)), HttpClient())
    except Exception:
        return None
    return [row.original for row in candidate.lines if row.original.strip()]


def reflag_existing(output_root, stem: str, sidecar_dir=None, audio=None) -> dict:
    """Add paren_flags to an existing song output from its lyrics file and saved sources.json.

    Saved diff rows only list real differences, so a block whose lines the timed
    source lacks can read as a mere notation difference; give ``audio`` to compare
    against the LRCLIB lines themselves (one request) as a new run does.  With
    ``sidecar_dir`` also writes ``<audio stem>.paren_flags.json`` there, the file the
    Review page reads when it sits beside the audio.
    """
    import json
    from pathlib import Path

    root = Path(output_root)
    txt = root / "txt" / f"{stem}.lyrics.txt"
    prov = root / "report" / f"{stem}.sources.json"
    payload = json.loads(prov.read_text(encoding="utf-8"))
    cross = (payload.get("verification") or {}).get("cross_sources") or []
    timed = _fetch_timed_lines(audio) if audio else None
    flags = flags_for_lyrics_text(txt.read_text(encoding="utf-8"), timed_lines=timed, cross_sources=cross)
    payload["paren_flags"] = flags
    prov.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if sidecar_dir is not None:
        side = {"txt_sha256": payload.get("txt_sha256", ""), "paren_flags": flags}
        name = Path(audio).stem if audio else stem
        (Path(sidecar_dir) / f"{name}.paren_flags.json").write_text(
            json.dumps(side, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return flags


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Add paren_flags to existing lyricsee outputs.")
    parser.add_argument("output_root")
    parser.add_argument("stems", nargs="+")
    parser.add_argument("--audio-dir", help="fetch each song's LRCLIB lines (audio whose name ends with the stem) instead of using saved diffs")
    parser.add_argument("--sidecar-dir", help="also write <audio stem>.paren_flags.json here (beside the audio)")
    args = parser.parse_args(argv)
    for stem in args.stems:
        audio = None
        if args.audio_dir:
            audio = next((p for p in sorted(Path(args.audio_dir).iterdir())
                          if p.stem.endswith(stem) and p.suffix.casefold() in {".flac", ".mp3", ".m4a", ".wav", ".ogg", ".opus"}), None)
        flags = reflag_existing(args.output_root, stem, args.sidecar_dir, audio)
        print(json.dumps({"stem": stem, **{k: flags[k] for k in ("basis", "block_count", "line_count")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
