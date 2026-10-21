"""Fail-closed source policy for lyricsee (2026-10-08).

Prefer LRCLIB original lyrics verified against audio metadata. Only borrow
published translation rows from matched lyrics pages, or keep verified
originals separately when no translation exists.
Do not create translations by guessing from search snippets.
"""
from __future__ import annotations

import hashlib
import re
from urllib.parse import urlparse

try:
    from lyricsee_core import (
        LyricCandidate, LyricPair, TrackIdentity,
        _metadata_score, _normalized, _looks_like_non_lyric_web_text,
    )
except ModuleNotFoundError:
    from lyricsee.lyricsee_core import (
        LyricCandidate, LyricPair, TrackIdentity,
        _metadata_score, _normalized, _looks_like_non_lyric_web_text,
    )

VERSION = "2026-10-08-google-lrclib-oogg-strict-v1"
WEB_HOSTS = {
    "home.gamer.com.tw", "forum.gamer.com.tw",
    "marumaru-x.com", "www.marumaru-x.com",
}
MIN_LINES = 8
MIN_WEB_AGREEMENT = 0.82
MIN_SOURCE_INDEPENDENCE = 0.85


def _orig_norm(candidate: LyricCandidate) -> list[str]:
    return [_normalized(row.original) for row in candidate.lines if row.original.strip()]


def _distinct_lines(candidate: LyricCandidate) -> int:
    return len(set(_orig_norm(candidate)))


def _metadata_ok(identity: TrackIdentity, candidate: LyricCandidate) -> bool:
    score = _metadata_score(
        identity, title=candidate.title,
        artist=candidate.artist or identity.artist,
        duration=None,
    )
    return score[0] >= 1 and (not identity.artist or score[1] >= 1)


def _trusted_web(identity: TrackIdentity, candidate: LyricCandidate) -> bool:
    if not candidate.source.startswith("google:"):
        return False
    if (urlparse(candidate.url).hostname or "").lower() not in WEB_HOSTS:
        return False
    if not candidate.title or not _metadata_ok(identity, candidate):
        return False
    return (
        len(candidate.lines) >= MIN_LINES
        and _distinct_lines(candidate) >= 6
        and all(
            not _looks_like_non_lyric_web_text(row.original)
            for row in candidate.lines
        )
    )


def original_overlap(a: LyricCandidate, b: LyricCandidate) -> float:
    """Coverage of canonical a by independent source b (unique lyric lines)."""
    a_lines = set(_orig_norm(a))
    b_lines = set(_orig_norm(b))
    if not a_lines or not b_lines:
        return 0.0
    return len(a_lines & b_lines) / len(a_lines)


def choose_verified_original(
    identity: TrackIdentity,
    candidates: list[LyricCandidate],
) -> tuple[LyricCandidate, dict]:
    lrclib = [
        item for item in candidates
        if item.source == "lrclib"
        and _metadata_ok(identity, item)
        and len(item.lines) >= MIN_LINES
        and _distinct_lines(item) >= 6
    ]
    websites = [item for item in candidates if _trusted_web(identity, item)]

    if lrclib:
        canonical = max(lrclib, key=lambda c: len(c.lines))
        agreement = [
            {
                "url": web.url,
                "canonical_original_coverage": round(original_overlap(canonical, web), 3),
            }
            for web in websites
        ]
        return canonical, {
            "policy": VERSION,
            "original_verified_by": "lrclib_title_artist_duration",
            "canonical_original_url": canonical.url,
            "cross_sources": agreement,
        }

    # When a metadata-checked structured original is unavailable, require two
    # separately published sources and substantial line-by-line agreement.
    for primary in websites:
        for secondary in websites:
            if primary is secondary:
                continue
            host_a = (urlparse(primary.url).hostname or "").lower()
            host_b = (urlparse(secondary.url).hostname or "").lower()
            if host_a == host_b:
                continue
            a_to_b = original_overlap(primary, secondary)
            b_to_a = original_overlap(secondary, primary)
            if (
                a_to_b >= MIN_SOURCE_INDEPENDENCE
                and b_to_a >= MIN_SOURCE_INDEPENDENCE
            ):
                return primary, {
                    "policy": VERSION,
                    "original_verified_by": "independent_web_original_agreement",
                    "canonical_original_url": primary.url,
                    "cross_sources": [{
                        "url": secondary.url,
                        "canonical_original_coverage": round(a_to_b, 3),
                        "reverse_coverage": round(b_to_a, 3),
                    }],
                }
    raise LookupError(
        "Cannot verify original lyrics: need a matching LRCLIB track "
        "or two independently published lyric pages with strong agreement. "
        "Refusing LLM reconstruction or an unverified TXT."
    )


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
        mapped = {}
        for row in candidate.lines:
            if row.translation and row.translation.strip():
                mapped.setdefault(_normalized(row.original), row.translation.strip())
        rows = [
            LyricPair(row.original, mapped.get(_normalized(row.original)))
            for row in original.lines
        ]
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
        discovered_urls=page.discovered_urls, google_search=page.google_search,
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
        if "\ufffd" in translation or "\x00" in translation:
            raise ValueError(f"Corrupted translation at row {idx}")
        if len(translation) > 350:
            raise ValueError(f"Translation unexpectedly long at row {idx}")


def attach_verification(
    candidate: LyricCandidate,
    evidence: dict,
    *,
    translation_method: str,
) -> LyricCandidate:
    candidate.verification = {
        **evidence,
        "translation_method": translation_method,
        "source_verified": True,
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
