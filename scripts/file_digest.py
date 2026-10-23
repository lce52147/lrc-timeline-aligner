"""SHA-256 of files, remembered while each file's stat identity is unchanged.

A digest is reused only while the file's path, size, mtime, ctime and inode
are all unchanged; any rewrite changes mtime, so the bytes are hashed again.
Files of 64 MiB or more (model weights) also keep their digest on disk under
the model cache root, so each process no longer re-reads multi-GB weights just
to name them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

import runtime_paths

_MEMO: dict[tuple[str, int, int, int, int], str] = {}
DISK_MIN_BYTES = 64 * 1024 * 1024
_DISK_NAME = "file-sha256.json"
_LOCK = threading.Lock()
_HEX64 = re.compile(r"[0-9a-f]{64}")


def hash_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def memo_path(name: str) -> Path:
    return runtime_paths.cache_root() / "runtime-identity" / name


def memo_load(name: str) -> dict[str, object]:
    try:
        payload = json.loads(memo_path(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def memo_store(name: str, key: str, value: object) -> None:
    path = memo_path(name)
    with _LOCK:
        try:
            payload = memo_load(name)
            payload[key] = value
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            temp.write_text(json.dumps(payload, sort_keys=True, indent=0), encoding="utf-8")
            os.replace(temp, path)
        except OSError:
            pass


def sha256(path: Path) -> str:
    try:
        stat = os.stat(path)
    except OSError:
        return hash_bytes(path)
    identity = (os.path.abspath(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
    digest = _MEMO.get(identity)
    if digest is not None:
        return digest
    large = stat.st_size >= DISK_MIN_BYTES
    disk_key = "|".join(str(part) for part in identity)
    if large:
        stored = memo_load(_DISK_NAME).get(disk_key)
        if isinstance(stored, str) and _HEX64.fullmatch(stored):
            digest = stored
    if digest is None:
        digest = hash_bytes(Path(path))
        if large:
            memo_store(_DISK_NAME, disk_key, digest)
    _MEMO[identity] = digest
    return digest
