"""Lyric-translation posts: a page that carries the original and a human Chinese translation line by line.

A Bahamut 【歌詞翻譯】 post for one song is a complete lyric by construction, so it is accepted on its
title and shape rather than on the chars-per-second snippet rule, and its human translation is written
instead of a machine one. OOGG only fills the lines the post did not translate.

  python translation_posts.py apply --out OUT [--stem STEM ...] [--oogg]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from candidate_review import load_candidates_metadata
from candidate_text import AGREE_SIM, char_similarity, fold

MIN_POST_LINES = 6          # a dedicated post shorter than this is not trusted as a whole song
MIN_POST_TRANSLATED = 0.7   # share of the post's rows that carry a translation
ALIGN_SIM = 0.8             # folded similarity for pairing an accepted line with a post row
WRONG_SONG_ASR = 0.13       # a real transcript scoring at most this against the post means another song
MIN_JAPANESE_CHARS = 30     # fewer than this in the transcript: the ASR heard no singing

_POST_MARKER_RE = re.compile(r"歌詞翻譯|歌词翻译|歌詞中譯|歌词中译|中日歌詞|日中歌詞|中日歌词|歌詞對譯|歌詞翻译|歌词翻譯")
_AUTHOR_RE = re.compile(r"-\s*([^-]+?)的創作\s*-")
_WRAPPERS = (("【", "】"), ("「", "」"), ("『", "』"), ("（", "）"), ("(", ")"), ("〔", "〕"), ("[", "]"), ("《", "》"))
_JAPANESE_RE = re.compile(r"[぀-ヿ一-鿿]")
_ASR_CREDIT_RE = re.compile(
    r"字幕|ご視聴|チャンネル登録|amara|subtitles?|subscribe|thanks? for watching|ありがとうございました"
    r"|请不吝|點贊|点赞",
    re.IGNORECASE,
)


def unwrap_translation(text: str) -> str:
    """Drop the bracket pair a post puts around a translation line: 【…】, 「…」, （…）."""
    text = text.strip()
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in _WRAPPERS:
            if text.startswith(left) and text.endswith(right) and left not in text[1:-1] and right not in text[1:-1]:
                text = text[1:-1].strip()
                changed = True
    return text


def split_inline_translation(line: str) -> tuple[str, str] | None:
    """'original【中文】' / 'original「中文」' / 'original（中文）' on one line -> (original, translation)."""
    line = line.strip()
    for left, right in _WRAPPERS:
        if not line.endswith(right) or left not in line[:-1]:
            continue
        head, _, tail = line[:-1].rpartition(left)
        head, tail = head.strip(), tail.strip()
        if len(head) >= 2 and len(tail) >= 4 and re.search(r"[぀-ヿ]", head) and re.search(r"[㐀-鿿]", tail) \
                and not re.search(r"[぀-ヿ]", tail):
            return head, tail
    return None


def post_author(page_title: str) -> str:
    match = _AUTHOR_RE.search(page_title or "")
    return match.group(1).strip() if match else ""


def is_translation_post_title(page_title: str, identity) -> bool:
    """The page title announces a lyric translation and names this song."""
    from lyricsee_core import _page_matches_identity

    if not page_title or not _POST_MARKER_RE.search(page_title):
        return False
    return bool(identity.title) and _page_matches_identity(identity, page_title)


def clean_pairs(pairs) -> list[dict]:
    """[{'original', 'translation'}] with the wrapper brackets removed; rows without an original dropped."""
    rows = []
    for pair in pairs:
        original = (pair.original if hasattr(pair, "original") else pair["original"]).strip()
        translation = pair.translation if hasattr(pair, "translation") else pair.get("translation")
        if original:
            rows.append({"original": original, "translation": unwrap_translation(translation or "")})
    return rows


def post_is_complete(rows: list[dict]) -> bool:
    translated = sum(bool(row["translation"]) for row in rows)
    return len(rows) >= MIN_POST_LINES and translated >= MIN_POST_TRANSLATED * len(rows)


def transcript_reliable(text: str) -> bool:
    """False when the ASR output is not singing: too little Japanese, repeated subtitle credits, a loop."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    body = [line for line in lines if not _ASR_CREDIT_RE.search(line)]
    if len(body) < 0.5 * len(lines):
        return False
    if len(set(body)) < 0.5 * len(body) and len(body) >= 6:
        return False
    return len(_JAPANESE_RE.findall("".join(body))) >= MIN_JAPANESE_CHARS


def align_translations(lines: list[str], rows: list[dict]) -> list[str | None]:
    """Human translation per original line, matched by folded text (exact first, then >= ALIGN_SIM); None if the post has none."""
    folded_rows = [fold(row["original"]) for row in rows]
    used: set[int] = set()
    result: list[str | None] = []
    for line in lines:
        key = fold(line)
        choice = None
        if key:
            for i, other in enumerate(folded_rows):
                if i not in used and other == key and rows[i]["translation"]:
                    choice = i
                    break
            if choice is None:
                best = ALIGN_SIM
                for i, other in enumerate(folded_rows):
                    if i in used or not rows[i]["translation"]:
                        continue
                    sim = char_similarity(key, other)
                    if sim >= best:
                        best, choice = sim, i
        if choice is None:
            # a chorus repeated in the song but written once in the post reuses that row
            for i, other in enumerate(folded_rows):
                if other == key and key and rows[i]["translation"]:
                    choice = i
                    break
        if choice is None:
            result.append(None)
        else:
            used.add(choice)
            result.append(rows[choice]["translation"])
    return result


# --- reading the saved post candidates --------------------------------------

def load_post_rows(out: Path, stem: str, idx: int) -> list[dict]:
    path = out / "candidates" / stem / f"candidate_{idx}.pairs.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def find_translation_post(out: Path, stem: str, original_lines: list[str]) -> dict | None:
    """Best saved translation post whose original agrees with these lines (sim >= AGREE_SIM)."""
    metadata = load_candidates_metadata(out, stem) or []
    mine = fold("".join(original_lines))
    best = None
    for idx, item in enumerate(metadata):
        if not item.get("translation_post") or item.get("rejected_by_user"):
            continue
        rows = load_post_rows(out, stem, idx)
        if not post_is_complete(rows):
            continue
        sim = char_similarity(mine, fold("".join(row["original"] for row in rows)))
        if sim < AGREE_SIM:
            continue
        aligned = align_translations(original_lines, rows)
        covered = sum(t is not None for t in aligned)
        if best is None or (covered, sim) > (best["covered"], best["sim"]):
            best = {
                "idx": idx, "url": item.get("url", ""), "author": item.get("page_author", ""),
                "title": item.get("page_title", ""), "rows": rows, "aligned": aligned,
                "covered": covered, "sim": round(sim, 4),
            }
    return best


# --- apply: write the human translation for a stem --------------------------

def original_column(line: str) -> str:
    """The original half of an 'original  translation' line (two ASCII spaces are the delimiter)."""
    return line.split("  ", 1)[0].strip()


def apply_post_translation(out: Path, stem: str, *, oogg_fill=None) -> dict:
    """Rewrite txt/<stem>.lyrics.txt with the human translation of a matching post.

    oogg_fill(lines) -> list[str] translates the lines the post did not cover. Returns a status dict;
    status 'no_post' leaves the file untouched.
    """
    from lyricsee_core import LyricPair, render_lyrics

    txt = out / "txt" / f"{stem}.lyrics.txt"
    if not txt.exists():
        return {"stem": stem, "status": "no_original"}
    lines = [original_column(line) for line in txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    post = find_translation_post(out, stem, lines)
    if not post:
        return {"stem": stem, "status": "no_post"}
    translations = list(post["aligned"])
    missing = [i for i, value in enumerate(translations) if value is None]
    method = "page_human"
    if missing:
        if oogg_fill is None:
            return {"stem": stem, "status": "uncovered_lines", "missing": len(missing), "url": post["url"]}
        filled = oogg_fill([lines[i] for i in missing])
        for i, value in zip(missing, filled):
            translations[i] = value
        method = "page_human+oogg_fill"
    text = render_lyrics([LyricPair(o, t) for o, t in zip(lines, translations)])
    txt.write_text(text, encoding="utf-8", newline="\n")
    sources = out / "report" / f"{stem}.sources.json"
    payload = json.loads(sources.read_text(encoding="utf-8")) if sources.exists() else {}
    payload.update({
        "translation_method": method,
        "translation_source_url": post["url"],
        "translation_author": post["author"],
        "translation_page_title": post["title"],
        "translation_original_agree_sim": post["sim"],
        "translation_lines_from_page": post["covered"],
        "translation_lines_oogg": len(missing),
    })
    sources.parent.mkdir(parents=True, exist_ok=True)
    sources.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "stem": stem, "status": "applied", "translation_method": method, "url": post["url"],
        "author": post["author"], "lines": len(lines), "from_page": post["covered"], "oogg": len(missing),
    }


def cmd_apply(args: argparse.Namespace) -> int:
    out = Path(args.out)
    stems = args.stem or sorted(p.name[: -len(".lyrics.txt")] for p in (out / "txt").glob("*.lyrics.txt"))
    oogg_fill = None
    if args.oogg:
        from lyricsee import DEFAULT_OOGG_ENDPOINTS, _select_oogg_endpoint
        from lyricsee_core import HttpClient
        from translation_oogg import translate_oogg

        target = _select_oogg_endpoint(HttpClient(), DEFAULT_OOGG_ENDPOINTS)
        if target:
            def oogg_fill(rows, _t=target):  # noqa: E306
                return translate_oogg(rows, title="", endpoint=_t.endpoint, model=_t.model, review=False)
    results = [apply_post_translation(out, stem, oogg_fill=oogg_fill) for stem in stems]
    for result in results:
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    apply = sub.add_parser("apply")
    apply.add_argument("--out", required=True)
    apply.add_argument("--stem", action="append")
    apply.add_argument("--oogg", action="store_true")
    apply.set_defaults(func=cmd_apply)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
