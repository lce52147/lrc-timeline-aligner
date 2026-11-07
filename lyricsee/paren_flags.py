"""Flag parenthesized lyric blocks that may not be sung.

Official lyric sites print backing vocals and unrecorded lines in parentheses,
sometimes as a block of several lines.  Those lines are information only: they
are never removed or changed, only listed in ``paren_flags`` of the song's
sources.json so the Review page can mark them quietly.

A block is a run of lines from an opening ``(`` / ``（`` at the start of a line
to the closing ``)`` / ``）`` that ends a line (a single fully parenthesized
line is a block of one).  Parentheses inside an otherwise normal line are not
a block, and neither is a block whose opening line also carries text outside
the parens.  An opening paren that never closes within MAX_BLOCK_LINES lines is
reported as an ``unclosed`` block that ends at its own line.

Classes:
  official-only  an independent timed source (LRCLIB) exists and lacks most of
                 the block's lines ("lacks": fewer than half of the block's
                 countable lines are found in that source; a block of nothing
                 but fillers such as "la la la" is judged on those lines)
  unconfirmed    no such source to compare, or the saved comparison cannot decide
  (a block the timed source has is not flagged)

``paren_flags["txt_sha256"]`` is the hash of the lyrics text the line numbers
refer to; the Review page ignores the flags when the lyrics file it reads
hashes differently.
"""
from __future__ import annotations

import hashlib

try:
    from lyric_compare import canonical_text, is_filler
except ModuleNotFoundError:  # pragma: no cover - package import style
    from lyricsee.lyric_compare import canonical_text, is_filler

VERSION = 2
OPEN = "(（"
CLOSE = ")）"
MAX_BLOCK_LINES = 8
LACK_BELOW = 0.5
MIN_CONTAINED = 6
FETCH_GAP_SECONDS = 1.0
TIMED_SOURCES = ("lrclib",)
AUDIO_SUFFIXES = {".flac", ".mp3", ".m4a", ".wav", ".ogg", ".opus"}
_FRAGMENT_KINDS = {"missing_in_b", "word"}
_last_fetch = [0.0]


def original_of(line: str) -> str:
    """The lyric part of a lyrics.txt line (translation follows two spaces)."""
    return line.split("  ", 1)[0]


def text_sha256(raw: bytes) -> str:
    """Hash of lyrics text bytes, leading UTF-8 BOM dropped and line endings normalised so a BOM or CRLF copy matches."""
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()


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
            else:
                outside.append(ch)  # a stray close paren is ordinary text
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
        if not _blank(outside):
            i += 1
            continue
        if depth == 0:
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
    canons = [canonical_text(line, drop_parens=False) for line in block_lines]
    countable = [canon for canon in canons if not is_filler(canon, min_length=1)]
    if not countable:
        # Backing fillers ("la la la", "oh oh") are exactly what a timed source may leave out.
        countable = [canon for canon in canons if canon]
    if not countable:
        return False
    # A short line must equal a timed line (a bare "hey" sits inside many others);
    # a longer one may also sit inside a timed line that merged it with its neighbour.
    present = sum(1 for canon in countable if canon in timed or (len(canon) >= MIN_CONTAINED and any(canon in t for t in timed)))
    return present / len(countable) < LACK_BELOW


def _lacks_by_diff_rows(numbers: list[int], originals: list[str], rows: list[dict]) -> bool:
    """Same test from a saved lrclib cross-source entry (its diff_lines).

    Only lines that appear as a diff row are known to be missing; a saved entry
    never lists a whole parenthesized line as missing (the comparer files it
    under notation), so "no row" cannot tell "present" from "not compared".
    """
    absent: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") not in _FRAGMENT_KINDS:
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
    ] or [n for n in numbers if canonical_text(originals[n - 1], drop_parens=False)]
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
    run, or fetched when re-flagging).  Without them, a saved lrclib entry of
    ``cross_sources`` is used (offline re-flagging): a block it shows as lacking is
    ``official-only``, any other block is ``unconfirmed`` because the saved diff
    cannot say whether the source has it.  With neither, every block is ``unconfirmed``.
    """
    saved = next(
        (c for c in (cross_sources or []) if isinstance(c, dict) and c.get("source") in TIMED_SOURCES and c.get("independent", True)),
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
            klass = "official-only" if _lacks_by_diff_rows(numbers, originals, rows) else "unconfirmed"
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
    flags = compute_flags(originals, **kwargs)
    flags["txt_sha256"] = text_sha256(lyrics_text.encode("utf-8"))
    return flags


def _identity(audio, payload: dict):
    """What to search LRCLIB with: the audio's tags, else the title/artist lyricsee saved."""
    from pathlib import Path

    try:
        from lyricsee_core import TrackIdentity, probe_audio
    except ModuleNotFoundError:  # pragma: no cover - package import style
        from lyricsee.lyricsee_core import TrackIdentity, probe_audio
    if audio:
        return probe_audio(Path(audio))
    title = str(payload.get("title") or "").strip()
    return TrackIdentity(title=title, artist=str(payload.get("artist") or "").strip()) if title else None


def _fetch_timed_lines(identity, *, strict: bool = False) -> list[str] | None:
    """Lyric lines of the LRCLIB match for ``identity`` (one request, at least FETCH_GAP_SECONDS after the last); None when there is none.

    A failed request (network, HTTP) also gives None, unless ``strict``: then it raises, so the caller can tell a failure from "no match"."""
    import time

    try:
        from lyricsee_core import HttpClient, fetch_lrclib_candidate
    except ModuleNotFoundError:  # pragma: no cover - package import style
        from lyricsee.lyricsee_core import HttpClient, fetch_lrclib_candidate
    if identity is None:
        return None
    wait = _last_fetch[0] + FETCH_GAP_SECONDS - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_fetch[0] = time.monotonic()
    try:
        candidate = fetch_lrclib_candidate(identity, HttpClient())
    except LookupError:
        return None
    except Exception:
        if strict:
            raise
        return None
    return [row.original for row in candidate.lines if row.original.strip()]


def _atomic_write(path, text: str) -> None:
    import os
    from pathlib import Path

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()  # still there only when the write or replace failed
        except FileNotFoundError:
            pass


def _check_stem(stem: str) -> None:
    from pathlib import Path

    if not stem or "/" in stem or "\\" in stem or ":" in stem or stem in {".", ".."} or Path(stem).is_absolute() or Path(stem).drive:
        raise ValueError(f"not a song stem: {stem!r}")


def reflag_existing(output_root, stem: str, sidecar_dir=None, audio=None, *, lrclib: bool = True) -> dict:
    """Add paren_flags to an existing song output from its lyrics file and saved sources.json.

    The flags are computed from the bytes of ``txt/<stem>.lyrics.txt`` read now and
    carry the hash of exactly those bytes.  By default the song's LRCLIB lines are
    fetched (one request, only when the lyrics contain a parenthesized block; the
    audio's tags are used when ``audio`` is given, else the title/artist in
    sources.json).  ``lrclib=False``, or a failed fetch, falls back to the saved diff
    rows of sources.json, which cannot decide most blocks (they become ``unconfirmed``).
    With ``sidecar_dir`` also writes ``<stem>.paren_flags.json`` there, a file the
    Review page reads when it sits beside the audio.
    """
    import json
    from pathlib import Path

    _check_stem(stem)
    root = Path(output_root)
    txt = root / "txt" / f"{stem}.lyrics.txt"
    prov = root / "report" / f"{stem}.sources.json"
    payload = json.loads(prov.read_text(encoding="utf-8"))
    cross = (payload.get("verification") or {}).get("cross_sources") or []
    raw = txt.read_bytes()
    text = raw.decode("utf-8-sig")
    timed = None
    if lrclib and detect_blocks([original_of(line) for line in text.splitlines()]):
        timed = _fetch_timed_lines(_identity(audio, payload))
    flags = flags_for_lyrics_text(text, timed_lines=timed, cross_sources=cross)
    flags["txt_sha256"] = text_sha256(raw)
    payload["paren_flags"] = flags
    _atomic_write(prov, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if sidecar_dir is not None:
        side = {"txt_sha256": flags["txt_sha256"], "paren_flags": flags}
        _atomic_write(Path(sidecar_dir) / f"{stem}.paren_flags.json", json.dumps(side, ensure_ascii=False, indent=2) + "\n")
    return flags


def find_audio(audio_dir, stem: str):
    """The one audio file in ``audio_dir`` whose stem is exactly ``stem``; ValueError for none or several."""
    from pathlib import Path

    matches = [p for p in sorted(Path(audio_dir).iterdir()) if p.stem == stem and p.suffix.casefold() in AUDIO_SUFFIXES]
    if len(matches) != 1:
        raise ValueError(f"{len(matches)} audio files named {stem!r} in {audio_dir}")
    return matches[0]


LYRIC_SUFFIXES = (".lyrics.txt", ".txt", ".lyrics")  # auto_lrc text order; flags refer to text lyrics, never the timed .lyrics.lrc
MAX_SONGS = 600


def lyrics_beside(audio):
    """The lyrics text file beside ``audio`` (first of LYRIC_SUFFIXES), or None."""
    return next((audio.with_name(audio.stem + s) for s in LYRIC_SUFFIXES if audio.with_name(audio.stem + s).is_file()), None)


def _hidden_runner(*args, **kwargs):
    import subprocess

    kwargs.setdefault("creationflags", getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return subprocess.run(*args, **kwargs)


def _probe(audio):
    try:
        from lyricsee_core import probe_audio
    except ModuleNotFoundError:  # pragma: no cover - package import style
        from lyricsee.lyricsee_core import probe_audio
    return probe_audio(audio, runner=_hidden_runner)


def _merge_out_root(target, flags: dict, body: str) -> None:
    """Write ``flags`` into ``target``: merged into an existing lyricsee sources.json (its other keys kept), else the skeleton ``body``."""
    import json

    if target.exists():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if not isinstance(payload, dict):
            raise ValueError("existing out-root file is not a JSON object; left untouched")
        payload["paren_flags"] = flags
        body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write(target, body)


def flag_audio_dir(audio_dir, *, recursive=False, sidecar_dir=None, out_root=None, offline=False,
                   dry_run=False, max_songs=MAX_SONGS, fetch=None, identity_of=None) -> dict:
    """Flag songs straight from the lyrics text beside each audio file.

    A song with no parenthesized block makes no LRCLIB request and writes nothing.
    Results go to ``sidecar_dir`` (``<stem>.paren_flags.json``, sub-folders mirrored)
    and/or ``out_root`` (``report/<stem>.sources.json``); nothing is written next to
    the audio unless ``sidecar_dir`` is that folder.  ``dry_run`` writes no flag files.
    Returns the summary (no lyric text in it).
    """
    import json
    from pathlib import Path

    root = Path(audio_dir)
    fetch = fetch or (lambda identity: _fetch_timed_lines(identity, strict=True))
    identity_of = identity_of or _probe
    audios = sorted((p for p in (root.rglob("*") if recursive else root.iterdir()) if p.is_file() and p.suffix.casefold() in AUDIO_SUFFIXES),
                    key=lambda p: str(p).casefold())
    summary = {"scanned": 0, "with_blocks": 0, "lines_by_class": {}, "songs": [], "errors": [], "truncated": False,
               "fetched": 0, "no_match": 0, "fetch_failed": 0}
    seen_stems: set[str] = set()
    seen_songs: set[tuple[Path, str]] = set()  # (folder, stem): FLAC/MP3 twins share lyrics and sidecar
    for audio in audios:
        lyrics = lyrics_beside(audio)
        if lyrics is None:
            continue
        song_key = (audio.relative_to(root).parent, audio.stem)
        if song_key in seen_songs:
            continue  # twin of a song already handled this run
        if summary["scanned"] >= max_songs:
            summary["truncated"] = True
            break
        seen_songs.add(song_key)
        summary["scanned"] += 1
        rel = audio.relative_to(root)
        try:
            raw = lyrics.read_bytes()
            text = raw.decode("utf-8-sig")
            if not detect_blocks([original_of(line) for line in text.splitlines()]):
                continue
            timed = None
            if not offline:
                try:
                    timed = fetch(identity_of(audio))
                except Exception:
                    timed = None  # a failed fetch leaves the blocks unconfirmed
                    summary["fetch_failed"] += 1
                else:
                    summary["fetched" if timed else "no_match"] += 1
            flags = flags_for_lyrics_text(text, timed_lines=timed)
            flags["txt_sha256"] = text_sha256(raw)
            body = json.dumps({"version": VERSION, "txt_sha256": flags["txt_sha256"], "paren_flags": flags}, ensure_ascii=False, indent=2) + "\n"
            if not dry_run:
                if sidecar_dir is not None:
                    target = Path(sidecar_dir) / rel.parent / f"{audio.stem}.paren_flags.json"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_write(target, body)
                if out_root is not None:
                    if audio.stem in seen_stems:
                        summary["errors"].append(f"{rel}: stem already written to out-root, skipped")
                    else:
                        (Path(out_root) / "report").mkdir(parents=True, exist_ok=True)
                        _merge_out_root(Path(out_root) / "report" / f"{audio.stem}.sources.json", flags, body)
            seen_stems.add(audio.stem)
        except (OSError, ValueError) as exc:
            summary["errors"].append(f"{rel}: {type(exc).__name__}")
            continue
        summary["with_blocks"] += 1
        line_numbers = []
        for block in flags["blocks"]:
            summary["lines_by_class"][block["class"]] = summary["lines_by_class"].get(block["class"], 0) + len(block["lines"])
            line_numbers += block["lines"]
        if flags["blocks"]:
            summary["songs"].append({"path": str(rel), "basis": flags["basis"], "lines": line_numbers,
                                     "classes": sorted({b["class"] for b in flags["blocks"]})})
    return summary


def format_summary(summary: dict, *, dry_run: bool) -> str:
    out = ["# paren_flags census" + (" (dry run)" if dry_run else ""), "",
           f"- songs scanned (audio with lyrics beside it): {summary['scanned']}",
           f"- songs with parenthesized blocks: {summary['with_blocks']}",
           f"- songs flagged: {len(summary['songs'])}"]
    if summary["truncated"]:
        out.append(f"- STOPPED at the {summary['scanned']}-song limit; more songs exist")
    out.append("- flagged lines by class: " + (", ".join(f"{k} {v}" for k, v in sorted(summary["lines_by_class"].items())) or "none"))
    out.append(f"- LRCLIB: matched {summary['fetched']}, no match {summary['no_match']}, fetch failed {summary['fetch_failed']}")
    out.append(f"- errors: {len(summary['errors'])}")
    out += ["", "## Flagged songs", ""]
    for song in summary["songs"]:
        out.append(f"- {song['path']} [{', '.join(song['classes'])}; {song['basis']}] lines {', '.join(map(str, song['lines']))}")
    if summary["errors"]:
        out += ["", "## Errors", ""] + [f"- {e}" for e in summary["errors"]]
    return "\n".join(out) + "\n"


def _lower_priority() -> None:
    try:
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL
    except Exception:  # pragma: no cover - not Windows
        pass


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Add paren_flags to existing lyricsee outputs, or flag songs from the lyrics beside their audio (--from-audio-dir).")
    parser.add_argument("output_root", nargs="?")
    parser.add_argument("stems", nargs="*")
    parser.add_argument("--from-audio-dir", help="flag each audio file here from the lyrics text beside it")
    parser.add_argument("--recursive", action="store_true", help="with --from-audio-dir: include sub-folders")
    parser.add_argument("--out-root", help="with --from-audio-dir: write report/<stem>.sources.json-style files here (and SUMMARY.md)")
    parser.add_argument("--dry-run", action="store_true", help="with --from-audio-dir: write no flag files")
    parser.add_argument("--max-songs", type=int, default=MAX_SONGS, help="with --from-audio-dir: stop after this many songs with lyrics")
    parser.add_argument("--audio-dir", help="read each song's tags from the audio file named exactly <stem> here (else title/artist from sources.json)")
    parser.add_argument("--sidecar-dir", help="also write <stem>.paren_flags.json here (beside the audio)")
    parser.add_argument("--offline", action="store_true", help="do not fetch LRCLIB; use the saved diff rows (most blocks stay unconfirmed)")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors="backslashreplace")
        sys.stderr.reconfigure(errors="backslashreplace")
    except (AttributeError, OSError):
        pass
    if args.from_audio_dir:
        if not (args.sidecar_dir or args.out_root):
            parser.error("--from-audio-dir needs --sidecar-dir or --out-root")
        if not args.dry_run:
            _lower_priority()
        summary = flag_audio_dir(args.from_audio_dir, recursive=args.recursive, sidecar_dir=args.sidecar_dir, out_root=args.out_root,
                                 offline=args.offline, dry_run=args.dry_run, max_songs=args.max_songs)
        text = format_summary(summary, dry_run=args.dry_run)
        if args.out_root:
            from pathlib import Path

            Path(args.out_root).mkdir(parents=True, exist_ok=True)
            _atomic_write(Path(args.out_root) / "SUMMARY.md", text)
        print(text)
        return 1 if summary["errors"] else 0
    if not args.output_root or not args.stems:
        parser.error("output_root and stems are required without --from-audio-dir")
    failed = 0
    for stem in args.stems:
        try:
            audio = find_audio(args.audio_dir, stem) if args.audio_dir else None
            flags = reflag_existing(args.output_root, stem, args.sidecar_dir, audio, lrclib=not args.offline)
        except (ValueError, OSError) as exc:
            failed += 1
            print(json.dumps({"stem": stem, "error": str(exc)}, ensure_ascii=False))
            continue
        print(json.dumps({"stem": stem, **{k: flags[k] for k in ("basis", "block_count", "line_count")}}, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
