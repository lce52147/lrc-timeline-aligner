"""Recoverable two-file publication of an LRC and its alignment report."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Callable


class PublicationError(RuntimeError):
    pass


def journal_path(report_path: Path) -> Path:
    return report_path.with_name(report_path.name + ".txn.json")


def recover_pair(lrc_path: Path, report_path: Path) -> bool:
    journal = journal_path(report_path)
    if not journal.exists():
        return False
    try:
        info = json.loads(journal.read_text(encoding="utf-8"))
        if info.get("lrc") != str(lrc_path.resolve()) or info.get("report") != str(report_path.resolve()):
            raise PublicationError("Journal belongs to another output pair")
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
        tmp_journal = journal.with_name(journal.name + "." + ident + ".tmp")
        tmp_journal.write_text(json.dumps(manifest), encoding="utf-8")
        os.replace(tmp_journal, journal)
        journal_written = True
        replace(stages["lrc"], lrc_path)
        replace(stages["report"], report_path)
        journal.unlink()
        journal_written = False
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