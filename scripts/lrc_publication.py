"""Recoverable two-file publication of an LRC and its alignment report."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from typing import Callable


class PublicationError(RuntimeError):
    pass


# A virus scanner or the search indexer can hold a just-renamed file for a moment: deleting the journal is retried with a short backoff.
JOURNAL_REMOVE_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)


def journal_path(report_path: Path) -> Path:
    return report_path.with_name(report_path.name + ".txn.json")


def _remove_journal(journal: Path) -> bool:
    """Delete ``journal``, retrying a locked file; False when it still cannot be deleted (the caller leaves it)."""
    for delay in (*JOURNAL_REMOVE_DELAYS, None):
        try:
            journal.unlink(missing_ok=True)
            return True
        except OSError:
            if delay is None:
                return False
            time.sleep(delay)
    return False


def _write_journal(journal: Path, manifest: dict, ident: str) -> None:
    """Atomically replace the journal with ``manifest``, retrying a locked file."""
    tmp_journal = journal.with_name(journal.name + "." + ident + ".tmp")
    tmp_journal.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        for delay in (*JOURNAL_REMOVE_DELAYS, None):
            try:
                os.replace(tmp_journal, journal)
                return
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)
    finally:
        tmp_journal.unlink(missing_ok=True)


def recover_pair(lrc_path: Path, report_path: Path) -> bool:
    journal = journal_path(report_path)
    if not journal.exists():
        return False
    try:
        info = json.loads(journal.read_text(encoding="utf-8"))
        if info.get("lrc") != str(lrc_path.resolve()) or info.get("report") != str(report_path.resolve()):
            raise PublicationError("Journal belongs to another output pair")
        if info.get("committed"):
            # Both files were published and only the journal's deletion failed: nothing to undo, so no rollback.
            for backup in (info.get("backups") or {}).values():
                if backup:
                    Path(backup).unlink(missing_ok=True)
            if not _remove_journal(journal):
                print(f"WARN: published pair is complete but its journal cannot be deleted yet: {journal}", file=sys.stderr)
            return True
        for name, target in (("lrc", lrc_path), ("report", report_path)):
            backup = Path(info["backups"][name]) if info["backups"][name] else None
            if backup is not None:
                if not backup.is_file():
                    raise PublicationError("Required recovery backup missing")
                restore = backup.with_name(backup.name + ".restoring")
                shutil.copy2(backup, restore)
                os.replace(restore, target)
            else:
                target.unlink(missing_ok=True)
        for stage in info.get("stages", {}).values():
            Path(stage).unlink(missing_ok=True)
        journal.unlink()
        for backup in info["backups"].values():
            if backup:
                Path(backup).unlink(missing_ok=True)
        return True
    except Exception as exc:
        raise PublicationError(f"Cannot recover incomplete output pair: {exc}") from exc

def publish_pair(
    lrc_path: Path,
    report_path: Path,
    write_lrc: Callable[[Path], object],
    write_report: Callable[[Path], object],
    *,
    replace: Callable[[Path, Path], object] = os.replace,
    on_lrc_staged: Callable[[str], object] | None = None,
) -> None:
    recover_pair(lrc_path, report_path)
    lrc_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # 12 hex digits are plenty to keep concurrent stages apart and keep staged paths short on Windows.
    ident = uuid.uuid4().hex[:12]
    stages = {
        "lrc": lrc_path.with_name(lrc_path.name + "." + ident + ".stage"),
        "report": report_path.with_name(report_path.name + "." + ident + ".stage"),
    }
    backups: dict[str, Path | None] = {"lrc": None, "report": None}
    journal = journal_path(report_path)
    journal_written = False
    try:
        write_lrc(stages["lrc"])
        if on_lrc_staged is not None and stages["lrc"].is_file():
            digest = hashlib.sha256(stages["lrc"].read_bytes()).hexdigest()
            on_lrc_staged(digest)
        write_report(stages["report"])
        if not stages["lrc"].is_file() or stages["lrc"].stat().st_size == 0:
            raise PublicationError("LRC stage was not written")
        if not stages["report"].is_file() or stages["report"].stat().st_size == 0:
            raise PublicationError("Report stage was not written")
        data = json.loads(stages["report"].read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise PublicationError("Report stage must be a JSON object")
        for name, target in (("lrc", lrc_path), ("report", report_path)):
            if target.exists():
                backup = target.with_name(target.name + "." + ident + ".backup")
                shutil.copy2(target, backup)
                backups[name] = backup
        manifest = {
            "schema": 1,
            "lrc": str(lrc_path.resolve()),
            "report": str(report_path.resolve()),
            "backups": {name: str(path) if path else None for name, path in backups.items()},
            "stages": {name: str(path) for name, path in stages.items()},
        }
        _write_journal(journal, manifest, ident)
        journal_written = True
        replace(stages["lrc"], lrc_path)
        replace(stages["report"], report_path)
        # Both files are in place. Record that first, so a journal that cannot be deleted never rolls a finished publication back.
        _write_journal(journal, {**manifest, "committed": True}, ident)
        journal_written = False
        if not _remove_journal(journal):
            print(f"WARN: published pair is complete but its journal cannot be deleted yet: {journal}", file=sys.stderr)
    except Exception:
        if journal_written:
            recover_pair(lrc_path, report_path)
        raise
    finally:
        if not journal_written:
            for stage in stages.values():
                stage.unlink(missing_ok=True)
            for backup in backups.values():
                if backup:
                    backup.unlink(missing_ok=True)
