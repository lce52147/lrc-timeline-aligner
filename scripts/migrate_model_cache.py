#!/usr/bin/env python3
"""Migrate a schema-1 model-output cache directory to schema 2 (LRC 1.5 W3).

Usage::

    python scripts/migrate_model_cache.py --src DIR --dst DIR [--dry-run]
    python scripts/migrate_model_cache.py --src DIR --in-place [--dry-run]

``--in-place`` migrates a cache where it is (the v1.5 formal-install step):
schema-2 entries are written next to the schema-1 ones; where the migrated
entry lands on the schema-1 file's own path, the original is first copied to
``_schema1_backup/<kind>/`` under the cache root, so nothing is lost.

Only ``raw-asr`` entries get a recomputed key: their request drops the
absolute ``cli`` path in favour of ``cli_sha256`` (recovered from the
``whispercpp-cli:`` prefix of ``identity.model_revision``).  Every other kind
keeps the key stored in its envelope; ``local-whisper`` additionally rewrites
``identity.model_revision`` to ``whispercpp-model:SHA``.  Payloads are copied
byte-for-byte as JSON values.  The source directory is never written.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import model_cache  # noqa: E402

LEGACY_SCHEMA = 1


class MigrationSkip(Exception):
    pass


def migrate_identity(kind: str, identity: dict) -> tuple[dict, bool]:
    """Return ``(new_identity, key_must_be_recomputed)``."""
    identity = json.loads(json.dumps(identity))  # deep copy
    if kind == "raw-asr":
        request = identity.get("request")
        if not isinstance(request, dict):
            raise MigrationSkip("raw-asr-request-missing")
        model_revision = str(identity.get("model_revision", ""))
        prefix = "whispercpp-cli:"
        if not model_revision.startswith(prefix):
            raise MigrationSkip("raw-asr-model-revision-without-cli-sha")
        cli_sha = model_revision[len(prefix):]
        if "cli_sha256" in request and "cli" not in request:
            return identity, False
        request.pop("cli", None)
        request["cli_sha256"] = cli_sha
        identity["request"] = request
        return identity, True
    if kind == "local-whisper":
        blob = identity.get("model_blob_sha256")
        if not isinstance(blob, str) or not blob:
            raise MigrationSkip("local-whisper-model-blob-missing")
        identity["model_revision"] = f"whispercpp-model:{blob}"
        return identity, False
    return identity, False


def migrate_envelope(envelope: dict) -> tuple[dict, bool]:
    if not isinstance(envelope, dict):
        raise MigrationSkip("envelope-not-object")
    schema = envelope.get("schema")
    if schema == model_cache.MODEL_CACHE_SCHEMA:
        raise MigrationSkip("already-schema-2")
    if schema != LEGACY_SCHEMA:
        raise MigrationSkip(f"unknown-schema:{schema}")
    kind = envelope.get("kind")
    if not isinstance(kind, str) or kind not in model_cache.ALL_KINDS:
        raise MigrationSkip(f"unknown-kind:{kind}")
    for field in ("key", "identity", "payload"):
        if field not in envelope:
            raise MigrationSkip(f"missing-field:{field}")
    if not isinstance(envelope["identity"], dict):
        raise MigrationSkip("identity-not-object")
    identity, recompute = migrate_identity(kind, envelope["identity"])
    key = model_cache.key_from_identity(kind, identity) if recompute else str(envelope["key"])
    try:
        model_cache.assert_portable_identity(identity)
    except model_cache.NonPortableIdentity as exc:
        raise MigrationSkip(f"identity-not-portable:{exc}") from exc
    return {
        "schema": model_cache.MODEL_CACHE_SCHEMA,
        "kind": kind,
        "key": key,
        "identity": identity,
        "payload": envelope["payload"],
    }, recompute


BACKUP_DIRNAME = "_schema1_backup"


def migrate_directory(src: Path, dst: Path, dry_run: bool = False, *, in_place: bool = False) -> dict:
    src = Path(src).resolve()
    dst = Path(dst).resolve()
    if src == dst and not in_place:
        raise SystemExit("--src and --dst must differ (use --in-place to migrate a cache where it is)")
    if in_place and src != dst:
        raise SystemExit("--in-place requires --dst to equal --src (or be omitted)")
    backup_root = dst / BACKUP_DIRNAME
    report: dict[str, dict] = {}
    backed_up = 0
    written_keys: dict[tuple[str, str], str] = {}

    def entry(kind: str) -> dict:
        return report.setdefault(kind, {
            "read": 0, "written": 0, "skipped": 0, "skip_reasons": defaultdict(int),
            "key_recomputed": kind == "raw-asr", "key_collisions": 0,
        })

    for path in sorted(src.rglob("*.json")):
        if path.name.endswith(".tmp"):
            continue
        if in_place and backup_root in path.parents:
            continue
        kind_guess = path.parent.name
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            row = entry(kind_guess)
            row["read"] += 1
            row["skipped"] += 1
            row["skip_reasons"][f"unreadable:{type(exc).__name__}"] += 1
            continue
        kind = envelope.get("kind") if isinstance(envelope, dict) else None
        row = entry(kind if isinstance(kind, str) else kind_guess)
        row["read"] += 1
        try:
            migrated, _ = migrate_envelope(envelope)
        except MigrationSkip as exc:
            row["skipped"] += 1
            row["skip_reasons"][str(exc)] += 1
            continue
        target = model_cache.cache_path(migrated["kind"], migrated["key"], dst)
        ident = (migrated["kind"], migrated["key"])
        if ident in written_keys and written_keys[ident] != str(path):
            row["key_collisions"] += 1
            row["skipped"] += 1
            row["skip_reasons"]["key-collision"] += 1
            continue
        if target.exists():
            if not (in_place and target.resolve() == path.resolve()):
                row["skipped"] += 1
                row["skip_reasons"]["dst-exists"] += 1
                continue
            # In place and the schema-2 entry lands on the schema-1 file: keep
            # the original under the backup folder before rewriting.
            backup = backup_root / path.relative_to(src)
            if not dry_run:
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(path, backup)
            backed_up += 1
        written_keys[ident] = str(path)
        if not dry_run:
            model_cache.write_envelope(target, migrated)
        row["written"] += 1

    for row in report.values():
        row["skip_reasons"] = dict(row["skip_reasons"])
    summary = {
        "src": str(src),
        "dst": str(dst),
        "dry_run": bool(dry_run),
        "in_place": bool(in_place),
        "schema1_backed_up": backed_up,
        "backup_dir": str(backup_root) if in_place else None,
        "schema_from": LEGACY_SCHEMA,
        "schema_to": model_cache.MODEL_CACHE_SCHEMA,
        "kinds": report,
        "total_read": sum(r["read"] for r in report.values()),
        "total_written": sum(r["written"] for r in report.values()),
        "total_skipped": sum(r["skipped"] for r in report.values()),
        "total_key_collisions": sum(r["key_collisions"] for r in report.values()),
    }
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "migration_report.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true", help="migrate --src where it is (schema-1 originals copied to _schema1_backup/)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    dst = args.dst if args.dst is not None else (args.src if args.in_place else None)
    if dst is None:
        parser.error("--dst is required unless --in-place is given")
    summary = migrate_directory(args.src, dst, dry_run=args.dry_run, in_place=args.in_place)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["total_key_collisions"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
