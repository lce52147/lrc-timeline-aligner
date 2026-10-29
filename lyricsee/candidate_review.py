"""Candidate review system for unverified lyrics."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

from candidate_text import (
    AGREE_SIM, char_similarity, clean_lyric_lines, clean_original_lines, fold, has_inline_furigana,
)
from lyricsee_core import LyricCandidate, traditionalize_lyrics_text


class CandidateInfo(NamedTuple):
    """Metadata about a candidate for review."""
    url: str
    host: str
    line_count: int
    char_count: int
    title_match: str
    artist_match: str
    duration_sanity: str
    reason: str


def extract_host(url: str) -> str:
    """Extract hostname from URL."""
    return (urlparse(url).hostname or "").lower()


def compute_similarity(lines1: list[str], lines2: list[str]) -> float:
    """Character-level similarity of two candidates, after cleaning to original-language lyric lines."""
    text1 = fold("".join(clean_lyric_lines(lines1)))
    text2 = fold("".join(clean_lyric_lines(lines2)))
    return round(char_similarity(text1, text2), 4)


def save_candidates(
    output_root: Path,
    stem: str,
    candidates: list[LyricCandidate],
    rejection_reason: str | None = None,
    identity=None,
) -> None:
    """Save candidates to output directory structure.

    Args:
        output_root: Root output directory
        stem: Audio file stem
        candidates: List of lyric candidates to save
        rejection_reason: Reason why no candidate was auto-accepted
        identity: the track; with it a page titled as a lyric translation of this song is flagged
            translation_post and its original/translation rows are saved as candidate_<n>.pairs.json
    """
    from translation_posts import clean_pairs, is_translation_post_title, post_author

    candidates_dir = output_root / "candidates" / stem
    candidates_dir.mkdir(parents=True, exist_ok=True)

    # Extract original lyrics for each candidate
    originals_by_idx: dict[int, list[str]] = {}
    for idx, candidate in enumerate(candidates):
        originals_by_idx[idx] = [row.original for row in candidate.lines]

    # Compute candidate metadata
    metadata: list[dict] = []
    for idx, candidate in enumerate(candidates):
        host = extract_host(candidate.url)
        originals = originals_by_idx[idx]

        # Compute similarity to all other candidates
        similarities: dict[int, float] = {}
        for other_idx, other_lines in originals_by_idx.items():
            if other_idx != idx:
                similarities[other_idx] = compute_similarity(originals, other_lines)

        # Determine match flags
        title_match = "exact" if candidate.title and candidate.title.strip() else "none"
        artist_match = "exact" if candidate.artist and candidate.artist.strip() else "none"

        # Duration sanity based on character count
        char_count = sum(len(line) for line in originals)
        duration_sanity = f"{len(originals)} lines, {char_count} chars"

        info = {
            "url": candidate.url,
            "host": host,
            "source": candidate.source,
            "line_count": len(originals),
            "char_count": char_count,
            "title_match": title_match,
            "artist_match": artist_match,
            "duration_sanity": duration_sanity,
            "similarity_to_others": similarities,
            "reason": rejection_reason or "no_matching_verified_source",
        }
        rows = clean_pairs(candidate.lines)
        if candidate.meta.get("identity_evidence") is not None:
            info["identity_evidence"] = candidate.meta["identity_evidence"]
        if identity is not None and any(row["translation"] for row in rows)                and is_translation_post_title(candidate.title, identity):
            info["translation_post"] = True
            info["page_title"] = candidate.title
            info["page_author"] = post_author(candidate.title)
            info["artist_confirmed"] = candidate.meta.get("artist_confirmed")
            (candidates_dir / f"candidate_{idx}.pairs.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        metadata.append(info)

        # Save original lyrics to text file
        lyrics_file = candidates_dir / f"candidate_{idx}.txt"
        lyrics_text = "\n".join(originals)
        lyrics_file.write_text(lyrics_text, encoding="utf-8")

    # Save candidates.json
    candidates_file = candidates_dir / "candidates.json"
    candidates_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_candidates_metadata(output_root: Path, stem: str) -> list[dict] | None:
    """Load candidates metadata for a stem."""
    candidates_file = output_root / "candidates" / stem / "candidates.json"
    if not candidates_file.exists():
        return None

    try:
        return json.loads(candidates_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def record_review_decision(
    output_root: Path,
    stem: str,
    decision: str,
    candidate_idx: int | None = None,
    execute: bool = True,
) -> None:
    """Record a review decision to the decisions log.

    Args:
        output_root: Root output directory
        stem: Audio file stem
        decision: "accept" or "reject"
        candidate_idx: Index of accepted candidate (for accept) or None (for reject)
        execute: If True, execute the decision (save/reject files); if False, just log
    """
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    decisions_file = report_dir / "review_decisions.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "stem": stem,
        "decision": decision,
        "candidate_idx": candidate_idx,
    }

    with open(decisions_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if execute:
        if decision == "accept" and candidate_idx is not None:
            accept_candidate(output_root, stem, candidate_idx)
        elif decision == "reject":
            reject_candidate(output_root, stem)


def cleaned_candidate_lines(candidates_dir: Path, candidate_idx: int, count: int) -> tuple[list[str], dict]:
    """Cleaned original lines of one candidate, preferring an agreeing candidate over inline furigana.

    furigana in cleaning: none | replaced_by_agreeing_candidate | unresolved (kept, to be fixed by hand or another source).

    Returns (lines, cleaning) where cleaning records what was done, for the sources.json provenance.
    """
    def read(idx: int) -> list[str]:
        path = candidates_dir / f"candidate_{idx}.txt"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    lines = clean_original_lines(read(candidate_idx))
    cleaning: dict = {"furigana": "none", "furigana_from_idx": None}
    if not has_inline_furigana(lines):
        return lines, cleaning
    mine = fold("".join(lines))
    others = []
    for idx in range(count):
        if idx == candidate_idx:
            continue
        other = clean_original_lines(read(idx))
        sim = char_similarity(mine, fold("".join(other)))
        if sim >= AGREE_SIM and other and not has_inline_furigana(other):
            others.append((sim, idx, other))
    if others:
        sim, idx, other = max(others)
        cleaning.update(furigana="replaced_by_agreeing_candidate", furigana_from_idx=idx, furigana_agree_sim=round(sim, 4))
        return other, cleaning
    # no agreeing candidate: a reading cannot be told from okurigana by rule (e.g. a kanji run whose
    # reading ends in the okurigana), so the text is kept as is and flagged rather than half-stripped
    cleaning["furigana"] = "unresolved"
    return lines, cleaning


def accept_candidate(
    output_root: Path,
    stem: str,
    candidate_idx: int,
    *,
    verification_status: str = "human_approved",
    verified_by: str = "review_page",
    extra: dict | None = None,
    lyrics_lines: list[str] | None = None,
) -> None:
    """Accept a candidate and save it as an approved original.

    The saved text is the cleaned original (see cleaned_candidate_lines), not the raw page text.

    Args:
        output_root: Root output directory
        stem: Audio file stem
        candidate_idx: Index of the accepted candidate
        verification_status / verified_by: provenance labels (auto_consensus for the auto-pick)
        extra: more provenance fields (scores) merged into the sources.json payload
        lyrics_lines: already-cleaned lines to save instead of this candidate's (caller states why in extra)
    """
    candidates_dir = output_root / "candidates" / stem
    txt_dir = output_root / "txt"
    report_dir = output_root / "report"

    # Load candidate metadata
    metadata_file = candidates_dir / "candidates.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"No candidates found for {stem}")

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if candidate_idx >= len(metadata):
        raise ValueError(f"Candidate index {candidate_idx} out of range")

    candidate_meta = metadata[candidate_idx]

    # Read the lyrics file
    lyrics_file = candidates_dir / f"candidate_{candidate_idx}.txt"
    if not lyrics_file.exists():
        raise FileNotFoundError(f"Candidate file not found: {lyrics_file}")

    if lyrics_lines is None:
        lyrics_lines, cleaning = cleaned_candidate_lines(candidates_dir, candidate_idx, len(metadata))
        extra = {**(extra or {}), "cleaning": cleaning}
    lyrics_text = traditionalize_lyrics_text("\n".join(lyrics_lines), original_only=True)
    if not lyrics_text.endswith("\n"):
        lyrics_text += "\n"

    # Create provenance entry
    txt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save the lyrics text
    output_lyrics = txt_dir / f"{stem}.lyrics.txt"
    output_lyrics.write_text(lyrics_text, encoding="utf-8")

    # Save provenance
    import hashlib
    payload = {
        "txt_sha256": hashlib.sha256(lyrics_text.encode("utf-8")).hexdigest(),
        "verification_status": verification_status,
        "verified_by": verified_by,
        "sources_used": [candidate_meta.get("host", "unknown")],
        "selected_source": candidate_meta.get("source", ""),
        "source_url": candidate_meta.get("url", ""),
        "human_approved": verification_status == "human_approved",
        "status": "needs_review",  # Mark as needs translation
        "line_count": candidate_meta.get("line_count", 0),
    }
    if extra:
        payload.update(extra)
    provenance_file = report_dir / f"{stem}.sources.json"
    provenance_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reject_candidate(output_root: Path, stem: str) -> None:
    """Mark a candidate as rejected by user.

    Args:
        output_root: Root output directory
        stem: Audio file stem
    """
    candidates_dir = output_root / "candidates" / stem
    candidates_metadata_file = candidates_dir / "candidates.json"

    if candidates_metadata_file.exists():
        metadata = json.loads(candidates_metadata_file.read_text(encoding="utf-8"))
        for item in metadata:
            item["rejected_by_user"] = True
        candidates_metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_review_decisions(output_root: Path) -> dict[str, dict]:
    """Load all review decisions for stems.

    Returns:
        Dictionary mapping stem to decision info
    """
    decisions_file = output_root / "report" / "review_decisions.jsonl"
    decisions: dict[str, dict] = {}

    if not decisions_file.exists():
        return decisions

    try:
        for line in decisions_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            stem = entry.get("stem")
            if stem:
                decisions[stem] = entry
    except Exception:
        pass

    return decisions
