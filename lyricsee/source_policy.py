"""Fail-closed source policy for lyricsee.

Order of preference for the original lyrics: the publisher-licensed sites
(uta-net, then UtaTen) first, then LRCLIB and lyric/translation pages as
backups.  A result counts
as **verified** only when

* two independent sources agree on the lyric text (differences in notation
  such as 二人/ふたり, ラララ fills or punctuation do not count; one wrong
  word does), or
* the text comes from a trusted publisher source whose row matched the
  recording's title *and* artist, and the content passes sanity checks
  (enough distinct lines, plausible amount of text for the audio duration,
  no page furniture).

Anything else that still names the right song is written as **unverified**
and is never reported as correct.  Rows that fail the title/artist gate
(a different song with a similar title) are rejected outright.  Translations
are only borrowed from pages whose originals match the verified text; no
translation is ever invented from search snippets.
"""
from __future__ import annotations

import hashlib
import unicodedata
from urllib.parse import urlparse

try:
    from lyricsee_core import (
        LyricCandidate, LyricPair, TrackIdentity,
        _normalized, _looks_like_non_lyric_web_text,
        artist_match_level, title_match_level,
    )
    from lyric_compare import canonical_text, compare_lyrics, diff_lines_for_report, is_filler
except ModuleNotFoundError:  # pragma: no cover - package import style
    from lyricsee.lyricsee_core import (
        LyricCandidate, LyricPair, TrackIdentity,
        _normalized, _looks_like_non_lyric_web_text,
        artist_match_level, title_match_level,
    )
    from lyricsee.lyric_compare import canonical_text, compare_lyrics, diff_lines_for_report, is_filler

VERSION = "2026-10-25-utanet-primary-cross-verified"
PUBLISHER_SOURCES = {"uta-net", "utaten"}
WEB_HOSTS = {
    "home.gamer.com.tw", "forum.gamer.com.tw",
    "marumaru-x.com", "www.marumaru-x.com",
}
MIN_LINES = 8
MIN_DISTINCT_LINES = 6
MIN_DISTINCT_RATIO = 0.3
MIN_WEB_AGREEMENT = 0.82
MIN_AGREEMENT_COVERAGE = 0.75
# Calibrated on 146 uta-net references of the DF playlist (2026-10-25):
# chars/second 0.89–5.2 (median 2.0), lines/minute 3.2–28.4.  The bounds
# below only reject junk (13 lines of 止まる/ここで/はあ for a 3-minute song)
# or a page that is clearly not one song's lyrics.
MIN_CHARS_PER_SECOND = 0.45
MAX_CHARS_PER_SECOND = 8.0
MIN_LINES_PER_MINUTE = 2.0
MAX_LINES_PER_MINUTE = 45.0
MAX_LINE_LENGTH = 200


def _orig_norm(candidate: LyricCandidate) -> list[str]:
    return [_normalized(row.original) for row in candidate.lines if row.original.strip()]


def _distinct_lines(candidate: LyricCandidate) -> int:
    return len(set(_orig_norm(candidate)))


def _originals(candidate: LyricCandidate) -> list[str]:
    return [row.original for row in candidate.lines if row.original.strip()]


def source_family(candidate: LyricCandidate) -> str:
    """Independence key: two candidates from the same family never confirm
    each other (two pages of one blog host are one source)."""
    if candidate.source.startswith("google:"):
        return (urlparse(candidate.url).hostname or candidate.source).lower()
    return candidate.source


def trust_class(candidate: LyricCandidate) -> str:
    if candidate.meta.get("trust"):
        return str(candidate.meta["trust"])
    if candidate.source in PUBLISHER_SOURCES:
        return "publisher"
    if candidate.source.startswith("google:"):
        return "web"
    return "community"


def content_sanity(candidate: LyricCandidate, identity: TrackIdentity) -> dict:
    """Is this plausibly the complete lyric of one song for this recording?"""
    originals = _originals(candidate)
    distinct = _distinct_lines(candidate)
    chars = sum(
        1 for line in originals for ch in unicodedata.normalize("NFKC", line) if ch.isalnum()
    )
    reasons: list[str] = []
    if len(originals) < MIN_LINES:
        reasons.append(f"only {len(originals)} lines")
    if distinct < MIN_DISTINCT_LINES:
        reasons.append(f"only {distinct} distinct lines")
    ratio = distinct / len(originals) if originals else 0.0
    if originals and ratio < MIN_DISTINCT_RATIO:
        reasons.append(f"distinct ratio {ratio:.2f}")
    if any(len(line) > MAX_LINE_LENGTH for line in originals):
        reasons.append("a line longer than a lyric line")
    noise = sum(1 for line in originals if _looks_like_non_lyric_web_text(line))
    if originals and noise / len(originals) > 0.2:
        reasons.append(f"{noise} page-furniture lines")
    filler = sum(1 for line in originals if is_filler(canonical_text(line), min_length=1))
    if originals and (len(originals) - filler) < MIN_DISTINCT_LINES:
        reasons.append("mostly vocal fills")
    metrics = {
        "lines": len(originals),
        "distinct_lines": distinct,
        "distinct_ratio": round(ratio, 3),
        "chars": chars,
    }
    duration = identity.duration
    if duration and duration > 0:
        chars_per_second = chars / duration
        lines_per_minute = len(originals) / duration * 60
        metrics["chars_per_second"] = round(chars_per_second, 3)
        metrics["lines_per_minute"] = round(lines_per_minute, 2)
        if chars_per_second < MIN_CHARS_PER_SECOND:
            reasons.append(f"too little text for {duration:.0f}s of audio ({chars} chars)")
        elif chars_per_second > MAX_CHARS_PER_SECOND:
            reasons.append(f"too much text for {duration:.0f}s of audio ({chars} chars)")
        if lines_per_minute < MIN_LINES_PER_MINUTE or lines_per_minute > MAX_LINES_PER_MINUTE:
            reasons.append(f"{lines_per_minute:.1f} lines per minute")
    return {"ok": not reasons, "reasons": reasons, **metrics}


def candidate_identity(identity: TrackIdentity, candidate: LyricCandidate) -> dict:
    """How well the candidate's own title/artist name the recording."""
    recorded = candidate.meta.get("identity") if isinstance(candidate.meta.get("identity"), dict) else {}
    if recorded and "strong" in recorded:
        return {
            "title_level": int(recorded.get("title_level", 0)),
            "artist_level": int(recorded.get("artist_level", 0)),
            "strong": bool(recorded["strong"]),
        }
    title_level = title_match_level(identity, candidate.title or "")
    artist_level = artist_match_level(identity, candidate.artist or "")
    if candidate.source.startswith("google:"):
        # Pages were already matched against the recording title when they
        # were fetched; their artist is unknown.  They can confirm a
        # structured source or pair with another host, never stand alone.
        return {"title_level": max(title_level, 1), "artist_level": artist_level, "strong": False}
    strong = title_level >= 1 and (artist_level >= 1 or not identity.artist)
    return {"title_level": title_level, "artist_level": artist_level, "strong": strong}


def _trusted_web(identity: TrackIdentity, candidate: LyricCandidate) -> bool:
    if not candidate.source.startswith("google:"):
        return False
    if (urlparse(candidate.url).hostname or "").lower() not in WEB_HOSTS:
        return False
    if not candidate.title:
        return False
    return (
        len(candidate.lines) >= MIN_LINES
        and _distinct_lines(candidate) >= MIN_DISTINCT_LINES
        and all(
            not _looks_like_non_lyric_web_text(row.original)
            for row in candidate.lines
        )
    )


class Assessment:
    def __init__(self, candidate: LyricCandidate, identity: dict, sanity: dict, admitted: bool, reason: str) -> None:
        self.candidate = candidate
        self.identity = identity
        self.sanity = sanity
        self.admitted = admitted
        self.reason = reason

    @property
    def strong(self) -> bool:
        return bool(self.identity.get("strong"))

    @property
    def trust(self) -> str:
        return trust_class(self.candidate)

    @property
    def family(self) -> str:
        return source_family(self.candidate)

    def describe(self) -> dict:
        return {
            "source": self.candidate.source,
            "url": self.candidate.url,
            "title": self.candidate.title,
            "artist": self.candidate.artist,
            "trust": self.trust,
            "identity": self.identity,
            "sanity": self.sanity,
            "admitted": self.admitted,
            "reason": self.reason,
        }


def assess_candidates(identity: TrackIdentity, candidates: list[LyricCandidate]) -> list[Assessment]:
    assessed: list[Assessment] = []
    for candidate in candidates:
        if not candidate.lines:
            continue
        ident = candidate_identity(identity, candidate)
        sanity = content_sanity(candidate, identity)
        reason = ""
        is_page = candidate.source.startswith("google:")
        if is_page and not _trusted_web(identity, candidate):
            reason = "untrusted lyric page"
        elif ident["title_level"] < (1 if is_page else 2):
            # "Call you" inside "I Call Your Name" is not the same song.
            reason = "title does not match"
        elif not sanity["ok"]:
            reason = "content check failed: " + "; ".join(sanity["reasons"])
        admitted = not reason
        if admitted and not ident["strong"] and not candidate.source.startswith("google:"):
            reason = "artist not confirmed; needs an agreeing independent source"
        assessed.append(Assessment(candidate, ident, sanity, admitted, reason))
    return assessed


SOURCE_PREFERENCE = {"uta-net": 0, "utaten": 1, "lrclib": 2}


def _primary_rank(item: Assessment) -> tuple:
    trust_order = {"publisher": 0, "community": 1, "web": 2}
    return (
        trust_order.get(item.trust, 3),
        0 if item.strong else 1,
        SOURCE_PREFERENCE.get(item.candidate.source, 9),
        -len(item.candidate.lines),
    )


def _cross_entry(primary: Assessment, other: Assessment, diff) -> dict:
    return {
        "source": other.candidate.source,
        "url": other.candidate.url,
        "independent": other.family != primary.family,
        "identity_strong": other.strong,
        **diff.summary(),
        "diff_lines": diff_lines_for_report(diff),
    }


def choose_verified_original(
    identity: TrackIdentity,
    candidates: list[LyricCandidate],
) -> tuple[LyricCandidate, dict]:
    """Pick the original lyrics and say how far they are verified.

    Returns ``(candidate, evidence)``; ``evidence["verification_status"]`` is
    ``"verified"`` or ``"unverified"``.  Raises ``LookupError`` when no
    candidate even names the right song.
    """
    assessed = assess_candidates(identity, list(candidates))
    considered = [item.describe() for item in assessed]
    admitted = sorted((item for item in assessed if item.admitted), key=_primary_rank)
    if not admitted:
        rejected = "; ".join(
            f"{item.candidate.source}: {item.reason}" for item in assessed if item.reason
        )
        raise LookupError(
            "No usable source for this recording"
            + (f" ({rejected})" if rejected else "")
            + ". Refusing an unverified or wrong-song TXT."
        )

    def evidence_for(primary: Assessment, status: str, verified_by: str, cross: list[dict], agreeing: list[Assessment]) -> dict:
        used = [primary.candidate.source] + [item.candidate.source for item in agreeing]
        independent = [entry for entry in cross if entry["independent"]]
        strong_entries = [entry for entry in independent if entry["identity_strong"]]
        pool = strong_entries or independent
        best = min(pool, key=lambda entry: (entry["word_diff_lines"], -entry["similarity"])) if pool else None
        return {
            "policy": VERSION,
            "verification_status": status,
            "original_verified_by": verified_by,
            "canonical_original_url": primary.candidate.url,
            "primary_source": primary.candidate.source,
            "primary_trust": primary.trust,
            "sources_used": used,
            "sources_considered": considered,
            "cross_sources": cross,
            "diff_line_count": best["real_diff_lines"] if best else 0,
            "word_diff_line_count": best["word_diff_lines"] if best else 0,
            "notation_diff_line_count": best["notation_diff_lines"] if best else 0,
            "cross_source_conflict": bool(best) and best["similarity"] < 0.5,
            "identity": primary.identity,
            "sanity": primary.sanity,
        }

    for primary in admitted:
        cross: list[dict] = []
        agreeing: list[Assessment] = []
        for other in admitted:
            if other is primary or other.family == primary.family:
                continue
            diff = compare_lyrics(_originals(primary.candidate), _originals(other.candidate))
            cross.append(_cross_entry(primary, other, diff))
            if diff.agrees(min_coverage=MIN_AGREEMENT_COVERAGE):
                agreeing.append(other)
        # Two independent sources agree.  Normally one of them was matched to
        # the recording by title and artist; two lyric pages that both carry
        # the exact title may also confirm each other, but not against a
        # title+artist+duration-matched source that disagrees.
        if agreeing:
            strong_support = primary.strong or any(item.strong for item in agreeing)
            exact_titles = primary.identity["title_level"] >= 2 and any(
                item.identity["title_level"] >= 2 for item in agreeing
            )
            conflicts_with_strong = any(
                entry["identity_strong"] and not entry["agrees"] and entry["similarity"] < 0.5
                for entry in cross
            )
            if strong_support or (exact_titles and not conflicts_with_strong):
                evidence = evidence_for(primary, "verified", "two_source_agreement", cross, agreeing)
                evidence["artist_confirmed"] = strong_support
                return primary.candidate, evidence
        if primary.trust == "publisher" and primary.strong and primary.sanity["ok"]:
            evidence = evidence_for(primary, "verified", "trusted_source_sanity", cross, [])
            evidence["artist_confirmed"] = True
            return primary.candidate, evidence

    # Nothing verified: keep the best strongly identified source, flagged.
    for primary in admitted:
        if primary.strong:
            cross = []
            for other in admitted:
                if other is primary or other.family == primary.family:
                    continue
                diff = compare_lyrics(_originals(primary.candidate), _originals(other.candidate))
                cross.append(_cross_entry(primary, other, diff))
            return primary.candidate, evidence_for(primary, "unverified", "single_source", cross, [])
    rejected = "; ".join(f"{item.candidate.source}: {item.reason}" for item in assessed if item.reason)
    raise LookupError(
        "Cannot verify original lyrics: no source matched both title and artist, "
        "and no two independent sources agree"
        + (f" ({rejected})" if rejected else "")
        + ". Refusing LLM reconstruction or an unverified TXT."
    )


def original_overlap(a: LyricCandidate, b: LyricCandidate) -> float:
    """Coverage of canonical a by independent source b (unique lyric lines,
    notation folded)."""
    a_lines = {canonical_text(row.original) for row in a.lines if row.original.strip()}
    b_lines = {canonical_text(row.original) for row in b.lines if row.original.strip()}
    a_lines.discard("")
    b_lines.discard("")
    if not a_lines or not b_lines:
        return 0.0
    return len(a_lines & b_lines) / len(a_lines)


def choose_verified_web_translation(
    identity: TrackIdentity,
    original: LyricCandidate,
    candidates: list[LyricCandidate],
) -> LyricCandidate:
    pages = []
    for candidate in candidates:
        if not _trusted_web(identity, candidate):
            continue
        overlap = original_overlap(original, candidate)
        if overlap < MIN_WEB_AGREEMENT:
            continue
        # Pair page rows to the verified original by line alignment so a
        # notation variant on the page (きみ for 君) still carries its
        # translation; the original text itself is never taken from the page.
        page_rows = [row for row in candidate.lines if row.original.strip()]
        diff = compare_lyrics([row.original for row in original.lines], [row.original for row in page_rows])
        aligned = {a_line: b_line for a_line, b_line in diff.pairs}
        mapped = {}
        for row in page_rows:
            if row.translation and row.translation.strip():
                mapped.setdefault(canonical_text(row.original), row.translation.strip())
        rows = []
        for number, row in enumerate(original.lines, start=1):
            translation = None
            if number in aligned:
                translation = (page_rows[aligned[number] - 1].translation or "").strip() or None
            if translation is None:
                translation = mapped.get(canonical_text(row.original))
            rows.append(LyricPair(row.original, translation))
        translations = sum(bool(row.translation) for row in rows)
        if translations < max(6, int(len(rows) * 0.85)):
            continue
        pages.append((overlap, translations / len(rows), candidate, rows))
    if not pages:
        raise LookupError(
            "No independently verified existing human translation. "
            "Keep the verified original in the separate no-LLM rebuild staging; "
            "do not fabricate a Chinese translation."
        )
    _, _, page, pairs = max(pages, key=lambda item: (item[0], item[1]))
    return LyricCandidate(
        source=page.source, url=page.url, title=identity.title,
        artist=identity.artist, lines=pairs, translator=page.translator,
        discovered_urls=page.discovered_urls, search_url=page.search_url,
        search_engine=page.search_engine,
    )


def validate_output(
    canonical: LyricCandidate,
    candidate: LyricCandidate,
    *,
    require_translation: bool,
) -> None:
    if len(candidate.lines) != len(canonical.lines) or len(candidate.lines) < MIN_LINES:
        raise ValueError("Output lost or invented lyric rows")
    for idx, (original_row, final_row) in enumerate(
        zip(canonical.lines, candidate.lines), 1
    ):
        if not final_row.original.strip():
            raise ValueError(f"Empty original lyric at row {idx}")
        if final_row.original != original_row.original:
            raise ValueError(f"Original lyric changed at row {idx}")
        translation = (final_row.translation or "").strip()
        if not translation:
            if require_translation:
                raise ValueError(f"Untranslated row {idx}")
            continue
        if _looks_like_non_lyric_web_text(translation):
            raise ValueError(f"Page UI / non-lyric translation at row {idx}")
        if "�" in translation or "\x00" in translation:
            raise ValueError(f"Corrupted translation at row {idx}")
        if len(translation) > 350:
            raise ValueError(f"Translation unexpectedly long at row {idx}")


def attach_verification(
    candidate: LyricCandidate,
    evidence: dict,
    *,
    translation_method: str,
) -> LyricCandidate:
    status = str(evidence.get("verification_status") or "verified")
    candidate.verification = {
        **evidence,
        "verification_status": status,
        "translation_method": translation_method,
        "source_verified": status == "verified",
        # Source correctness still does NOT establish semantic translation
        # accuracy or timing, and cannot be auto-admitted to final LRC.
        "translation_semantics_human_checked": False,
        "lrc_timing_verified": False,
    }
    return candidate


def verified_provenance_is_current(provenance: dict, output_bytes: bytes) -> bool:
    verify = provenance.get("verification") or {}
    return (
        verify.get("policy") == VERSION
        and verify.get("source_verified") is True
        and provenance.get("txt_sha256") ==
            hashlib.sha256(output_bytes).hexdigest()
    )
