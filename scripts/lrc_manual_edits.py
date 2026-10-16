"""Detect and archive files edited through the local review page."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil

_REPORT_SUFFIX = ".align-report.json"

def review_artifact_paths(report_path: Path) -> tuple[Path, Path]:
    if not report_path.name.endswith(_REPORT_SUFFIX):
        raise ValueError(f"Unsupported alignment report path: {report_path}")
    stem = report_path.name[:-len(_REPORT_SUFFIX)]
    return (
        report_path.with_name(stem + ".auto.lrc"),
        report_path.with_name(stem + ".review-edits.json"),
    )

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _empty_skeleton_matches(output_path: Path, edits_path: Path, auto_path: Path) -> bool:
    if not (edits_path.is_file() and auto_path.is_file()):
        return False
    try:
        payload = json.loads(edits_path.read_text(encoding="utf-8-sig"))
        return (
            isinstance(payload, dict)
            and payload.get("edits") == []
            and payload.get("auto_lrc_sha256") == sha256_file(output_path)
        )
    except (OSError, TypeError, ValueError):
        return False

def manual_edit_reason(output_path: Path, report_path: Path) -> str | None:
    if not output_path.is_file():
        return None
    auto_path, edits_path = review_artifact_paths(report_path)
    if _empty_skeleton_matches(output_path, edits_path, auto_path):
        return None
    if auto_path.exists() or edits_path.exists():
        return "review-edits-present"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        published = report.get("published_lrc_sha256") if isinstance(report, dict) else None
        if isinstance(published, str) and sha256_file(output_path) != published:
            return "lrc-differs-from-published"
    except (OSError, TypeError, ValueError):
        pass
    return None

def archive_manual_edits(output_path: Path, report_path: Path) -> list[Path]:
    auto_path, edits_path = review_artifact_paths(report_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = report_path.name[:-len(_REPORT_SUFFIX)]
    archived: list[Path] = []
    lrc_archive = output_path.with_name(stem + f".discarded-{stamp}.lrc")
    shutil.copy2(output_path, lrc_archive)
    archived.append(lrc_archive)
    for source, suffix in ((auto_path, ".auto.lrc"), (edits_path, ".review-edits.json")):
        if source.exists():
            destination = report_path.with_name(stem + f".discarded-{stamp}" + suffix)
            os.replace(source, destination)
            archived.append(destination)
    return archived
