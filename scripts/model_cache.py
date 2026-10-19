"""Persistent model-output cache (LRC 1.5 W3).

One module owns the cache envelope, the key algorithm, the cache-only policy
and the portability rule for cache identities, so that ``auto_lrc.py``,
``segment_skeleton_production.py`` and ``r2_pipeline.py`` share one
implementation instead of three copies.

Envelope (``MODEL_CACHE_SCHEMA = 2``)::

    {"schema": 2, "kind": ..., "key": ..., "identity": {...}, "payload": ...}

Key algorithm (frozen, ``MODEL_CACHE_KEY_REVISION = 1``): the key of the
four identity-keyed kinds is ``sha256`` of the canonical JSON of
``{"schema": 1, "kind", "vocal_stem_sha256", "model_revision",
"model_blob_sha256", "helper_protocol_revision", "request"}``.  The ``schema``
field inside the digest is the *key revision*, not the envelope schema, so a
new envelope schema does not move cache files.  Changing the key algorithm
requires updating ``test_fixtures/model_cache_golden_v2.json``.

Identities never contain filesystem paths (``assert_portable_identity``) so a
cache directory can be copied between machines and worktrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import runtime_paths

MODEL_CACHE_SCHEMA = 2
MODEL_CACHE_KEY_REVISION = 1

ENV_MODEL_CACHE = "LRC_MODEL_CACHE"
ENV_MODEL_CACHE_ONLY = "LRC_MODEL_CACHE_ONLY"

_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}

#: Kinds whose key is a pure function of the identity (``key_from_identity``).
IDENTITY_KEYED_KINDS = frozenset({
    "raw-asr",
    "whisperx-alignment",
    "whole-ctc",
    "local-window-ctc",
})
#: Kinds whose key is computed by a kind-specific function elsewhere; their
#: envelopes are reused as-is by the migration tool.
EXTERNALLY_KEYED_KINDS = frozenset({
    "multimodel-consensus",
    "local-whisper",
    "crossline-joint-ctc",
})
#: Kinds keyed by a digest of the whole identity (``key_from_identity``).
WHOLE_IDENTITY_KEYED_KINDS = frozenset({
    "r2-reviewer-sidecar",
})
ALL_KINDS = IDENTITY_KEYED_KINDS | EXTERNALLY_KEYED_KINDS | WHOLE_IDENTITY_KEYED_KINDS

#: Identity keys whose string values are lyric text and may legitimately look
#: like paths (``A:/``, leading slash) without binding the cache to a machine.
PORTABLE_IDENTITY_TEXT_KEYS = frozenset({
    "transcript", "entries", "lyric_texts", "text", "lines", "lyrics",
})

_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"\\\\"),
    re.compile(r"^/"),
)


class ModelCacheMiss(BaseException):
    """The requested cache-only execution has no verified model output.

    Deliberately not an LrcError/Exception: evidence collectors degrade on
    ordinary helper failures, and a cache-only replay must never turn a miss
    into "evidence unavailable" and still publish an LRC.
    """


class NonPortableIdentity(ValueError):
    """A cache identity contains a filesystem path."""


def content_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# policy
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCachePolicy:
    enabled: bool = True
    read_only: bool = False

    @property
    def cache_only(self) -> bool:
        return bool(self.enabled and self.read_only)

    @property
    def mode(self) -> str:
        if not self.enabled:
            return "off"
        return "cache-only" if self.read_only else "read-write"


def _env_flag_off(name: str, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(name, "1")).lower() in _FALSE_VALUES


def _env_flag_on(name: str, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(name, "0")).lower() in _TRUE_VALUES


def cache_only_active(environ: Mapping[str, str] | None = None) -> bool:
    return _env_flag_on(ENV_MODEL_CACHE_ONLY, environ)


def policy_from_env_and_args(args: object | None = None, environ: Mapping[str, str] | None = None) -> ModelCachePolicy:
    """Effective policy: environment narrows first, then CLI flags."""
    enabled = not _env_flag_off(ENV_MODEL_CACHE, environ)
    if args is not None and not getattr(args, "model_cache", True):
        enabled = False
    read_only = cache_only_active(environ)
    if args is not None and getattr(args, "model_cache_only", False):
        read_only = True
    return ModelCachePolicy(enabled=enabled, read_only=read_only)


def policy_from_args(args: object | None) -> ModelCachePolicy:
    if isinstance(args, ModelCachePolicy):
        return args
    return policy_from_env_and_args(args)


# --------------------------------------------------------------------------
# keys and paths
# --------------------------------------------------------------------------

def model_cache_key(
    kind: str,
    stem_sha256: str,
    model_revision: str,
    model_blob_sha256: str,
    helper_protocol_revision: str,
    request: object,
) -> str:
    """The 1.4 ``_model_cache_key`` algorithm, with the key revision frozen."""
    return content_digest({
        "schema": MODEL_CACHE_KEY_REVISION,
        "kind": kind,
        "vocal_stem_sha256": stem_sha256,
        "model_revision": model_revision,
        "model_blob_sha256": model_blob_sha256,
        "helper_protocol_revision": helper_protocol_revision,
        "request": request,
    })


def key_from_identity(kind: str, identity: Mapping[str, object]) -> str:
    if kind in IDENTITY_KEYED_KINDS:
        return model_cache_key(
            kind,
            str(identity["vocal_stem_sha256"]),
            str(identity["model_revision"]),
            str(identity["model_blob_sha256"]),
            str(identity["helper_protocol_revision"]),
            identity["request"],
        )
    if kind in WHOLE_IDENTITY_KEYED_KINDS:
        return content_digest({
            "schema": MODEL_CACHE_KEY_REVISION,
            "kind": kind,
            "identity": dict(identity),
        })
    raise ValueError(f"model-cache-kind-not-identity-keyed:{kind}")


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def cache_path(kind: str, key: str, root: Path | None = None) -> Path:
    base = Path(root) if root is not None else runtime_paths.model_cache_dir()
    return base / _safe_component(kind) / f"{_safe_component(key)}.json"


# --------------------------------------------------------------------------
# portability
# --------------------------------------------------------------------------

def _looks_like_path(value: str, project_root_text: str) -> bool:
    for pattern in _PATH_PATTERNS:
        if pattern.search(value):
            return True
    return bool(project_root_text) and project_root_text in value


def _walk_strings(value: object, parent_key: str | None, out: list[tuple[str | None, str]]) -> None:
    if isinstance(value, str):
        out.append((parent_key, value))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in PORTABLE_IDENTITY_TEXT_KEYS:
                continue
            _walk_strings(item, str(key), out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_strings(item, parent_key, out)


def assert_portable_identity(identity: Mapping[str, object]) -> None:
    """Raise ``NonPortableIdentity`` if any non-lyric string looks like a path."""
    if not isinstance(identity, Mapping):
        raise NonPortableIdentity("identity-not-a-mapping")
    strings: list[tuple[str | None, str]] = []
    _walk_strings(identity, None, strings)
    root_text = str(runtime_paths.project_root())
    for key, text in strings:
        if _looks_like_path(text, root_text):
            raise NonPortableIdentity(f"identity-contains-path:{key}:{text[:80]}")


# --------------------------------------------------------------------------
# read / write
# --------------------------------------------------------------------------

def _load_envelope(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read(
    kind: str,
    key: str,
    identity: Mapping[str, object],
    policy: ModelCachePolicy | object | None,
    root: Path | None = None,
) -> object | None:
    """Return the cached payload, ``None`` on a miss, or raise ``ModelCacheMiss``.

    ``policy`` may be a ``ModelCachePolicy`` or an argparse-style namespace
    (``model_cache`` / ``model_cache_only`` attributes).
    """
    effective = policy_from_args(policy)
    if not effective.enabled:
        return None
    assert_portable_identity(identity)
    path = cache_path(kind, key, root)
    try:
        envelope = _load_envelope(path)
        if (
            not isinstance(envelope, dict)
            or envelope.get("schema") != MODEL_CACHE_SCHEMA
            or envelope.get("kind") != kind
            or envelope.get("key") != key
            or envelope.get("identity") != dict(identity)
            or "payload" not in envelope
        ):
            raise ValueError("identity-mismatch")
        return envelope["payload"]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        if effective.cache_only:
            raise ModelCacheMiss(f"model-cache-miss:{kind}:{key}")
        return None


def write_envelope(path: Path, envelope: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def write(
    kind: str,
    key: str,
    identity: Mapping[str, object],
    payload: object,
    policy: ModelCachePolicy | object | None,
    root: Path | None = None,
) -> bool:
    """Write an envelope atomically; returns ``False`` when the policy forbids it."""
    effective = policy_from_args(policy)
    if not effective.enabled or effective.read_only:
        return False
    assert_portable_identity(identity)
    write_envelope(cache_path(kind, key, root), {
        "schema": MODEL_CACHE_SCHEMA,
        "kind": kind,
        "key": key,
        "identity": dict(identity),
        "payload": payload,
    })
    return True


# --------------------------------------------------------------------------
# identity builders (one per kind; callable with fake inputs in tests)
# --------------------------------------------------------------------------

def identity_raw_asr(
    *, stem_sha256: str, cli_sha256: str, model_sha256: str, language: str, suppress_nst: bool,
) -> dict[str, object]:
    return {
        "vocal_stem_sha256": stem_sha256,
        "model_revision": f"whispercpp-cli:{cli_sha256}",
        "model_blob_sha256": model_sha256,
        "helper_protocol_revision": "whispercpp-json-transcription-v1",
        "request": {
            "audio_sha256": stem_sha256,
            "language": str(language),
            "suppress_nst": bool(suppress_nst),
            "cli_sha256": cli_sha256,
        },
    }


def identity_whisperx_alignment(
    *, stem_sha256: str, model_revision: str, model_blob_sha256: str,
    helper_protocol_revision: str, helper_sha256: str, transcript: object,
    language: str, device: str,
) -> dict[str, object]:
    return {
        "vocal_stem_sha256": stem_sha256,
        "model_revision": model_revision,
        "model_blob_sha256": model_blob_sha256,
        "helper_protocol_revision": helper_protocol_revision,
        "request": {
            "audio_sha256": stem_sha256,
            "transcript": transcript,
            "language": str(language),
            "device": str(device),
        },
        "helper_sha256": helper_sha256,
    }


def identity_whole_ctc(
    *, stem_sha256: str, model_revision: str, model_blob_sha256: str,
    helper_protocol_revision: str, helper_sha256: str, entries: list[str],
    duration: float, device: str, source: str,
) -> dict[str, object]:
    return {
        "vocal_stem_sha256": stem_sha256,
        "model_revision": model_revision,
        "model_blob_sha256": model_blob_sha256,
        "helper_protocol_revision": helper_protocol_revision,
        "request": {
            "entries": list(entries),
            "duration": round(float(duration), 6),
            "device": str(device),
            "source": str(source),
        },
        "helper_sha256": helper_sha256,
    }


def identity_local_window_ctc(
    *, stem_sha256: str, model_revision: str, model_blob_sha256: str,
    helper_protocol_revision: str, helper_sha256: str, request: Mapping[str, object],
) -> dict[str, object]:
    return {
        "vocal_stem_sha256": stem_sha256,
        "model_revision": model_revision,
        "model_blob_sha256": model_blob_sha256,
        "helper_protocol_revision": helper_protocol_revision,
        "request": dict(request),
        "helper_sha256": helper_sha256,
    }


def identity_multimodel_consensus(
    *, model_revision: str, model_blob_sha256: str, protocol_revision: str, models: list[str],
) -> dict[str, object]:
    return {
        "model_revision": model_revision,
        "model_blob_sha256": model_blob_sha256,
        "helper_protocol_revision": protocol_revision,
        "models": list(models),
    }


def identity_local_whisper(
    *, audio_sha256: str, whisper_model_sha256: str, protocol_revision: str, request_revision: str,
) -> dict[str, object]:
    return {
        "vocal_stem_sha256": audio_sha256,
        "model_revision": f"whispercpp-model:{whisper_model_sha256}",
        "model_blob_sha256": whisper_model_sha256,
        "helper_protocol_revision": protocol_revision,
        "request": request_revision,
    }


def identity_crossline_joint_ctc(
    *, alignment_audio_sha256: str, model_identity: str, model_blob_sha256: str,
    helper_protocol_revision: str, transcript_digest: str, window_start: object,
    window_end: object, context_entry_indexes: list[int],
) -> dict[str, object]:
    return {
        "vocal_stem_sha256": alignment_audio_sha256,
        "model_revision": model_identity,
        "model_blob_sha256": model_blob_sha256,
        "helper_protocol_revision": helper_protocol_revision,
        "request": {
            "transcript_digest": transcript_digest,
            "window_start": window_start,
            "window_end": window_end,
            "context_entry_indexes": list(context_entry_indexes),
        },
    }


def identity_r2_reviewer_sidecar(
    *, payload_sha256: str, reviewers_sha256: str, whisper_cli_sha256: str,
    whisper_model_sha256: str, device: str, include_hub: bool,
) -> dict[str, object]:
    return {
        "payload_sha256": payload_sha256,
        "reviewers_sha256": reviewers_sha256,
        "whisper_cli_sha256": whisper_cli_sha256,
        "whisper_model_sha256": whisper_model_sha256,
        "device": str(device),
        "include_hub": bool(include_hub),
    }


__all__ = [
    "MODEL_CACHE_SCHEMA", "MODEL_CACHE_KEY_REVISION", "ModelCacheMiss", "NonPortableIdentity",
    "ModelCachePolicy", "policy_from_env_and_args", "policy_from_args", "cache_only_active",
    "model_cache_key", "key_from_identity", "cache_path", "assert_portable_identity",
    "read", "write", "write_envelope", "content_digest",
    "IDENTITY_KEYED_KINDS", "EXTERNALLY_KEYED_KINDS", "WHOLE_IDENTITY_KEYED_KINDS", "ALL_KINDS",
    "PORTABLE_IDENTITY_TEXT_KEYS",
]
