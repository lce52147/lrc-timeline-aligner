"""Extracted from auto_lrc.py (split step 1): constants, exceptions, paths/hashes, temp dirs, persistent helper pools, GPU coordination, model-cache policy."""

from __future__ import annotations

import argparse
import atexit
import collections
import contextlib
import copy
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping
import runtime_paths
import file_digest
import model_cache
from model_cache import ModelCacheMiss
import lrc_stage_timings


_ALGORITHM_SOURCE_MODULES = (
    "runtime_base",
    "audio_features",
    "lyrics_io",
    "timing_core",
    "report_trust",
    "ctc_refinement",
    "acoustic_hypotheses",
    "asr_backends",
    "local_whisper_retry",
    "central_peer_recovery",
    "neutral_arbitration",
    "central_commit",
    "backend_bundle",
    "postcentral_recovery",
    "runtime_evidence_binding",
)


def _algorithm_source_files() -> list[Path]:
    """Source files whose bytes define the loaded algorithm (missing ones skipped)."""
    here = Path(__file__).resolve().parent
    paths = [here / "auto_lrc.py"]
    paths.extend(here / f"{name}.py" for name in _ALGORITHM_SOURCE_MODULES)
    return sorted(path for path in paths if path.is_file())


def _algorithm_source_sha256() -> str:
    digest = hashlib.sha256()
    for path in _algorithm_source_files():
        digest.update(path.name.encode("utf-8") + b"\x00")
        digest.update(path.read_bytes() + b"\x00")
    return digest.hexdigest()


def algorithm_source_text() -> str:
    """Concatenated algorithm source (for source-inspection tests)."""
    parts = []
    for path in _algorithm_source_files():
        text = path.read_text(encoding="utf-8")
        if parts:
            text = text.replace("from __future__ import annotations\n", "", 1)
        parts.append(text)
    return "\n".join(parts)


_LOADED_SOURCE_SHA256 = _algorithm_source_sha256()


DEFAULT_MODEL_CACHE_DIR = runtime_paths.model_cache_dir()


def _evidence_worker_count() -> int:
    """Parallel evidence workers (resident model copies per model family).

    Each worker is a separate helper process holding its own copy of the CTC or
    WhisperX model, so two workers double model VRAM/RAM during parallel
    phases.  Default 1; ``LRC_EVIDENCE_WORKERS`` (1-4) opts into more.
    Execution policy only: Central semantics and evidence payloads are the same
    for any worker count.
    """
    raw = os.environ.get("LRC_EVIDENCE_WORKERS", "").strip()
    try:
        value = int(raw) if raw else 1
    except ValueError:
        value = 1
    return max(1, min(value, 4))


DEFAULT_EVIDENCE_WORKERS = _evidence_worker_count()


_BOUNDED_RETRY_HELPER_PROTOCOL = "json-known-lyric-rows-global-window-v1+persistent-host-v1"


PROGRESS_START = time.monotonic()


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


configure_stdio()


def progress_log(message: str) -> None:
    if os.environ.get("LRC_TOOLS_PROGRESS", "1").lower() in {"0", "false", "no", "off"}:
        return
    elapsed = max(0, int(time.monotonic() - PROGRESS_START))
    minutes, seconds = divmod(elapsed, 60)
    print(f"[{minutes:02d}:{seconds:02d}] {message}", file=sys.stderr, flush=True)


class _PersistentHelperClient:
    """One long-lived helper Python process with exactly one in-flight request."""

    def __init__(self, python_exe: Path, helper: Path, mode: str, worker_script: Path):
        self.python_exe = python_exe
        self.helper = helper
        self.mode = mode
        self.worker_script = worker_script
        self.lock = threading.Lock()
        self.responses: queue.Queue[object] = queue.Queue()
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=32)
        self.request_counter = 0
        self.process = subprocess.Popen(
            [str(python_exe), str(worker_script), "--mode", mode, "--helper", str(helper), "--parent-pid", str(os.getpid())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert self.process.stdout is not None and self.process.stderr is not None
        threading.Thread(target=self._stdout_reader, daemon=True).start()
        threading.Thread(target=self._stderr_reader, daemon=True).start()
        try:
            ready = self.responses.get(timeout=45.0)
        except queue.Empty as exc:
            self.close()
            raise LrcError(f"persistent {mode} helper startup timeout") from exc
        if not isinstance(ready, dict) or ready.get("type") != "ready":
            self.close()
            raise LrcError(f"persistent {mode} helper startup failed: {ready!r}")

    def _stdout_reader(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.responses.put(json.loads(line))
                except json.JSONDecodeError:
                    self.responses.put({"type": "protocol-error", "error": "invalid-worker-json", "raw": line[-1024:]})
        finally:
            self.responses.put({"type": "eof"})

    def _stderr_reader(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr_tail.append(line.rstrip())

    def is_alive(self) -> bool:
        return self.process.poll() is None

    def run(self, argv: list[str], timeout_seconds: float | None = None) -> subprocess.CompletedProcess[str]:
        with self.lock:
            if not self.is_alive() or self.process.stdin is None:
                raise LrcError(f"persistent {self.mode} helper is not running")
            self.request_counter += 1
            request_id = f"{os.getpid()}-{id(self)}-{self.request_counter}"
            self.process.stdin.write(json.dumps({"type": "run", "request_id": request_id, "argv": argv}, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            try:
                response = self.responses.get(timeout=timeout_seconds)
            except queue.Empty as exc:
                self.close()
                raise subprocess.TimeoutExpired([str(self.python_exe), str(self.helper), *argv], timeout_seconds) from exc
            if not isinstance(response, dict) or response.get("type") != "result" or response.get("request_id") != request_id:
                raise LrcError(f"persistent {self.mode} helper protocol failure: {response!r}")
            return subprocess.CompletedProcess(
                [str(self.python_exe), str(self.helper), *argv],
                int(response.get("returncode", 1)),
                str(response.get("stdout", "")),
                str(response.get("stderr", "")),
            )

    def park(self, timeout_seconds: float = 30.0) -> dict[str, object]:
        """Move cached model weights to CPU while keeping this helper process alive."""
        with self.lock:
            if not self.is_alive() or self.process.stdin is None:
                raise LrcError(f"persistent {self.mode} helper is not running")
            self.request_counter += 1
            request_id = f"{os.getpid()}-{id(self)}-{self.request_counter}"
            self.process.stdin.write(json.dumps({"type": "park", "request_id": request_id}, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            try:
                response = self.responses.get(timeout=timeout_seconds)
            except queue.Empty as exc:
                self.close()
                raise LrcError(f"persistent {self.mode} helper park timeout") from exc
            if not isinstance(response, dict) or response.get("type") != "parked" or response.get("request_id") != request_id:
                raise LrcError(f"persistent {self.mode} helper park protocol failure: {response!r}")
            return response

    def close(self) -> None:
        proc = getattr(self, "process", None)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write('{"type":"shutdown"}\n')
                    proc.stdin.flush()
            except (OSError, ValueError):
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
        # The reader threads do not own these descriptors. Python can emit
        # ResourceWarning for both Popen pipes when the worker exits during
        # failed startup or a retry timeout.
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except (OSError, ValueError):
                pass


class _PersistentHelperPool:
    def __init__(self, python_exe: Path, helper: Path, mode: str, worker_script: Path, size: int):
        self.python_exe = python_exe
        self.helper = helper
        self.mode = mode
        self.worker_script = worker_script
        self.size = max(1, min(int(size), 4))
        # LIFO keeps serial helper calls on the same warm client.  The old FIFO
        # rotated a purely serial Global WhisperX sequence across worker 0/1,
        # cold-loading a second model before any parallel work existed.  Push in
        # reverse so worker 0 remains the first/primary client.
        self.available: queue.LifoQueue[int] = queue.LifoQueue()
        for index in reversed(range(self.size)):
            self.available.put(index)
        self.clients: dict[int, _PersistentHelperClient] = {}
        self.clients_lock = threading.Lock()

    def _client(self, index: int) -> _PersistentHelperClient:
        with self.clients_lock:
            client = self.clients.get(index)
            if client is None or not client.is_alive():
                if client is not None:
                    client.close()
                client = _PersistentHelperClient(self.python_exe, self.helper, self.mode, self.worker_script)
                self.clients[index] = client
            return client

    def run(self, argv: list[str], timeout_seconds: float | None) -> subprocess.CompletedProcess[str]:
        index = self.available.get()
        try:
            return self._client(index).run(argv, timeout_seconds)
        finally:
            self.available.put(index)

    def park(self, timeout_seconds: float = 30.0) -> dict[str, int]:
        """Park only already-created clients; never instantiate work just to park it."""
        with self.clients_lock:
            clients = list(self.clients.values())
        parked_models = 0
        parked_clients = 0
        for client in clients:
            if not client.is_alive():
                continue
            response = client.park(timeout_seconds)
            parked_clients += 1
            parked_models += int(response.get("parked_models", 0) or 0)
        return {"clients": parked_clients, "models": parked_models}

    def close(self) -> None:
        with self.clients_lock:
            clients = list(self.clients.values())
            self.clients.clear()
        for client in clients:
            client.close()


_PERSISTENT_HELPER_POOLS: dict[tuple[str, str, str, str], _PersistentHelperPool] = {}


_PERSISTENT_HELPER_POOLS_LOCK = threading.Lock()


_PERSISTENT_HELPER_DISABLED: set[tuple[str, str, str, str]] = set()


_PERSISTENT_HELPER_LOGGED: set[tuple[str, str, str, str]] = set()


_MANAGED_TEMP_DIRS: set[Path] = set()


_MANAGED_TEMP_DIRS_LOCK = threading.Lock()


def _managed_temp_dir(*, prefix: str) -> Path:
    """Create a process-owned temp directory tracked until it is safely removable."""
    path = Path(tempfile.mkdtemp(prefix=prefix))
    with _MANAGED_TEMP_DIRS_LOCK:
        _MANAGED_TEMP_DIRS.add(path)
    return path


def _managed_temp_snapshot() -> frozenset[Path]:
    with _MANAGED_TEMP_DIRS_LOCK:
        return frozenset(_MANAGED_TEMP_DIRS)


def _cleanup_managed_temp_dirs_since(snapshot: frozenset[Path] | None = None) -> int:
    """Remove managed dirs created after *snapshot*; retain failures for exit retry."""
    with _MANAGED_TEMP_DIRS_LOCK:
        targets = [
            path for path in _MANAGED_TEMP_DIRS
            if snapshot is None or path not in snapshot
        ]
    removed = 0
    for path in targets:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except OSError:
            # A transient Windows handle may still be closing.  Keep ownership
            # registered so the process-exit cleanup gets another chance.
            continue
        with _MANAGED_TEMP_DIRS_LOCK:
            _MANAGED_TEMP_DIRS.discard(path)
        removed += 1
    return removed


def _remove_stale_output(path: Path) -> None:
    """Require an external producer to create this invocation's output afresh."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _persistent_helper_worker_path() -> Path:
    override = os.environ.get("LRC_PERSISTENT_HELPER_WORKER", "").strip()
    return Path(override).expanduser().resolve() if override else (Path(__file__).resolve().parent / "persistent_helper_worker.py")


def _close_persistent_helper_pools() -> None:
    with _PERSISTENT_HELPER_POOLS_LOCK:
        pools = list(_PERSISTENT_HELPER_POOLS.values())
        _PERSISTENT_HELPER_POOLS.clear()
    for pool in pools:
        pool.close()
    _cleanup_managed_temp_dirs_since()


def _close_persistent_helper_mode(
    mode: Literal["ctc", "whisperx"],
    *,
    reason: str,
) -> int:
    """Release one resident model family at an explicit GPU phase boundary.

    Persistence is phase-scoped on 12 GB GPUs: a model remains loaded while its
    bounded work is active, then the helper process is terminated before the
    next heavyweight model family starts. Process exit is the CUDA/VRAM release
    boundary and does not alter evidence payloads or Central semantics.
    """
    with _PERSISTENT_HELPER_POOLS_LOCK:
        selected = [
            (key, pool)
            for key, pool in _PERSISTENT_HELPER_POOLS.items()
            if key[2] == mode
        ]
        for key, _pool in selected:
            _PERSISTENT_HELPER_POOLS.pop(key, None)
            _PERSISTENT_HELPER_LOGGED.discard(key)
    for _key, pool in selected:
        pool.close()
    if selected:
        progress_log(
            f"persistent {mode} helper phase release: pools={len(selected)} "
            f"reason={reason}"
        )
    return len(selected)


def _park_persistent_helper_mode(
    mode: Literal["ctc", "whisperx"],
    *,
    reason: str,
) -> int:
    """Release CUDA residency without discarding a warm helper/model cache.

    Any protocol/model-move failure fails open to the previous phase-release
    behavior: the affected pool is removed and terminated so correctness never
    depends on parking support.
    """
    with _PERSISTENT_HELPER_POOLS_LOCK:
        selected = [
            (key, pool)
            for key, pool in _PERSISTENT_HELPER_POOLS.items()
            if key[2] == mode
        ]
    if not selected:
        return 0

    parked_pools = 0
    parked_clients = 0
    parked_models = 0
    failed: list[tuple[tuple[str, str, str, str], _PersistentHelperPool, Exception]] = []
    for key, pool in selected:
        try:
            stats = pool.park()
            parked_pools += 1
            parked_clients += int(stats.get("clients", 0) or 0)
            parked_models += int(stats.get("models", 0) or 0)
        except (LrcError, OSError, ValueError, subprocess.SubprocessError) as exc:
            failed.append((key, pool, exc))

    for key, pool, exc in failed:
        with _PERSISTENT_HELPER_POOLS_LOCK:
            current = _PERSISTENT_HELPER_POOLS.get(key)
            if current is pool:
                _PERSISTENT_HELPER_POOLS.pop(key, None)
                _PERSISTENT_HELPER_LOGGED.discard(key)
        pool.close()
        progress_log(
            f"persistent {mode} helper park failed; phase release fallback: "
            f"{type(exc).__name__} reason={reason}"
        )

    if parked_pools:
        progress_log(
            f"persistent {mode} helper parked: pools={parked_pools} "
            f"clients={parked_clients} models={parked_models} reason={reason}"
        )
    return parked_pools


atexit.register(_close_persistent_helper_pools)


def _run_model_helper_command(
    command: list[str],
    *,
    mode: Literal["ctc", "whisperx"],
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run an unchanged helper through a resident model host, fail-open to one-shot."""
    with gpu_compute_phase():
        return _run_model_helper_command_now(command, mode=mode, timeout_seconds=timeout_seconds)


def _run_model_helper_command_now(
    command: list[str],
    *,
    mode: Literal["ctc", "whisperx"],
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    if len(command) < 2:
        return run_command(command, timeout_seconds=timeout_seconds)
    python_token = str(command[0])
    resolved_python = shutil.which(python_token) if not Path(python_token).is_absolute() else python_token
    python_exe = Path(resolved_python or python_token).expanduser().resolve()
    helper = Path(command[1]).expanduser().resolve()
    worker_script = _persistent_helper_worker_path()
    key = (str(python_exe), str(helper), mode, str(worker_script))
    if not worker_script.is_file() or key in _PERSISTENT_HELPER_DISABLED:
        return run_command(command, timeout_seconds=timeout_seconds)
    try:
        with _PERSISTENT_HELPER_POOLS_LOCK:
            pool = _PERSISTENT_HELPER_POOLS.get(key)
            if pool is None:
                pool = _PersistentHelperPool(
                    python_exe, helper, mode, worker_script, DEFAULT_EVIDENCE_WORKERS
                )
                _PERSISTENT_HELPER_POOLS[key] = pool
        result = pool.run([str(item) for item in command[2:]], timeout_seconds)
        if key not in _PERSISTENT_HELPER_LOGGED:
            _PERSISTENT_HELPER_LOGGED.add(key)
            progress_log(f"persistent {mode} helper active: resident model cache")
        return result
    except subprocess.TimeoutExpired:
        raise
    except (LrcError, OSError, ValueError, subprocess.SubprocessError) as exc:
        _PERSISTENT_HELPER_DISABLED.add(key)
        with _PERSISTENT_HELPER_POOLS_LOCK:
            pool = _PERSISTENT_HELPER_POOLS.pop(key, None)
        if pool is not None:
            pool.close()
        progress_log(f"persistent {mode} helper unavailable; one-shot fallback: {type(exc).__name__}")
        return run_command(command, timeout_seconds=timeout_seconds)


def progress_change_summary(changes: object) -> str:
    if not isinstance(changes, list) or not changes:
        return "count=0"
    entries: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            continue
        entry = change.get("entry")
        if isinstance(entry, int):
            entries.append(str(entry))
        elif isinstance(change.get("entries"), list):
            numeric_entries = [item for item in change["entries"] if isinstance(item, int)]
            if numeric_entries:
                if len(numeric_entries) == 1:
                    entries.append(str(numeric_entries[0]))
                elif numeric_entries == list(range(numeric_entries[0], numeric_entries[-1] + 1)):
                    entries.append(f"{numeric_entries[0]}-{numeric_entries[-1]}")
                else:
                    entries.extend(str(item) for item in numeric_entries)
        elif isinstance(change.get("start_entry"), int) and isinstance(change.get("end_entry"), int):
            entries.append(f"{change['start_entry']}-{change['end_entry']}")
    unique_entries = list(dict.fromkeys(entries))
    suffix = f", entries={','.join(unique_entries[:12])}" if unique_entries else ""
    if len(unique_entries) > 12:
        suffix += ",..."
    return f"count={len(changes)}{suffix}"


def progress_change_log(label: str, changes: object) -> None:
    progress_log(f"{label}: {progress_change_summary(changes)}")


def progress_report_changes(label: str, report: dict[str, object], key: str) -> None:
    progress_change_log(label, report.get(key))


@dataclass(frozen=True)
class CanonicalWrittenTime:
    centiseconds: int
    seconds: float
    lrc_tag: str


def canonicalize_lrc_time(seconds: float) -> CanonicalWrittenTime:
    """Return the exact centisecond value that will be serialized to LRC."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    whole = int(rest)
    centiseconds = int(round((rest - whole) * 100))
    if centiseconds >= 100:
        whole += 1
        centiseconds -= 100
    if whole >= 60:
        minutes += 1
        whole -= 60
    total_centiseconds = (minutes * 60 + whole) * 100 + centiseconds
    return CanonicalWrittenTime(
        centiseconds=total_centiseconds,
        seconds=total_centiseconds / 100.0,
        lrc_tag=f"{minutes:02d}:{whole:02d}.{centiseconds:02d}",
    )


def _timing_content_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class LrcError(RuntimeError):
    pass


def _model_cache_only_active() -> bool:
    return model_cache.cache_only_active()


def _refuse_model_launch_in_cache_only(site: str) -> None:
    """Every GPU model launch passes gpu_compute_phase or whispercpp_gpu_lock."""
    if _model_cache_only_active():
        raise ModelCacheMiss(f"model-cache-only:model-launch-refused:{site}")


def _apply_model_cache_flags(args: argparse.Namespace) -> str:
    """CLI flags only narrow the environment; an inherited cache-only stays on."""
    if not getattr(args, "model_cache", True):
        os.environ["LRC_MODEL_CACHE"] = "0"
    if getattr(args, "model_cache_only", False):
        os.environ["LRC_MODEL_CACHE_ONLY"] = "1"
    enabled = os.environ.get("LRC_MODEL_CACHE", "1").lower() not in {"0", "false", "no", "off"}
    if _model_cache_only_active():
        if not enabled:
            raise LrcError("--model-cache-only requires the model cache to be enabled")
        return "cache-only"
    return "read-write" if enabled else "off"


def _model_cache_policy(args: argparse.Namespace | None) -> model_cache.ModelCachePolicy:
    return model_cache.policy_from_args(args)


def _model_cache_enabled(args: argparse.Namespace) -> bool:
    return _model_cache_policy(args).enabled


def _model_cache_only(args: argparse.Namespace) -> bool:
    return _model_cache_policy(args).cache_only


def _model_cache_key(
    kind: str,
    stem_sha256: str,
    model_revision: str,
    model_blob_sha256: str,
    helper_protocol_revision: str,
    request: object,
) -> str:
    return model_cache.model_cache_key(
        kind, stem_sha256, model_revision, model_blob_sha256, helper_protocol_revision, request
    )


def _model_cache_path(kind: str, key: str) -> Path:
    return model_cache.cache_path(kind, key, DEFAULT_MODEL_CACHE_DIR)


def _model_cache_read(
    kind: str,
    key: str,
    identity: dict[str, object],
    args: argparse.Namespace,
) -> object | None:
    return model_cache.read(kind, key, identity, _model_cache_policy(args), DEFAULT_MODEL_CACHE_DIR)


def _model_cache_write(
    kind: str,
    key: str,
    identity: dict[str, object],
    payload: object,
    args: argparse.Namespace,
) -> None:
    model_cache.write(kind, key, identity, payload, _model_cache_policy(args), DEFAULT_MODEL_CACHE_DIR)


def _ctc_model_blob_sha256() -> str:
    candidates: list[Path] = []
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        root = Path(torch_home)
        candidates.extend((root / "hub" / "checkpoints" / "model.pt", root / "checkpoints" / "model.pt"))
    candidates.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "model.pt")
    for candidate in candidates:
        try:
            if candidate.is_file():
                return _stable_file_sha256(candidate)
        except OSError:
            continue
    return "unresolved"


class SegmentSkeletonRollback(LrcError):
    """Full Central rejected this segment-current sequence; retry whole old run."""


def segment_postcentral_nonmonotonic(
    report: dict[str, object],
    written_times: tuple[CanonicalWrittenTime, ...],
) -> tuple[int, ...]:
    """Only completed segment skeletons need a CTC retry.

    This does not change a single Central decision, time, or proof state.
    """
    status = report.get("segment_skeleton_status")
    if not isinstance(status, dict) or status.get("status") != "SEGMENT_CURRENT":
        return ()
    return tuple(
        index+1 for index in range(1, len(written_times))
        if written_times[index].centiseconds < written_times[index-1].centiseconds
    )


def validated_loaded_source_sha256() -> str:
    """Bind evidence to the code loaded on process startup, fail on mid-run edit.

    A source file digest computed only at report-writing time falsely claims
    that changed Python source was the code the running process executed.
    """
    try:
        on_disk = _algorithm_source_sha256()
    except OSError as exc:
        raise LrcError(f"Cannot verify running algorithm source: {exc}") from exc
    if on_disk != _LOADED_SOURCE_SHA256:
        raise LrcError(
            "Algorithm source changed since this process loaded; "
            "abort mixed-version provenance and restart with frozen source"
        )
    return _LOADED_SOURCE_SHA256


class AlignmentEvidenceRejected(LrcError):
    """A backend result is globally unusable but still contains line evidence."""

    def __init__(self, message: str, timestamps: list[float], report: dict[str, object]):
        super().__init__(message)
        self.timestamps = list(timestamps)
        self.report = copy.deepcopy(report)


class WhisperXEvidenceBundleError(LrcError):
    """All WhisperX candidates failed globally; preserve their per-line evidence."""

    def __init__(
        self,
        message: str,
        evidence_candidates: list[tuple[list[float], dict[str, object], str]],
    ):
        super().__init__(message)
        self.evidence_candidates = [
            (list(timestamps), copy.deepcopy(report), str(error))
            for timestamps, report, error in evidence_candidates
            if isinstance(report, dict) and isinstance(report.get("assignments"), list)
        ]


def run_command(
    args: list[str],
    *,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise LrcError(f"Command not found: {args[0]}") from exc


def default_whisper_cli() -> Path:
    return runtime_paths.whisper_cli()


def default_whisper_model() -> Path:
    return runtime_paths.whisper_model()


def default_whisperx_python() -> Path:
    return runtime_paths.venv_python()


def default_ctc_python() -> Path:
    return default_whisperx_python()


def default_whisperx_ready() -> bool:
    return default_whisper_cli().exists() and default_whisper_model().exists() and default_whisperx_python().exists()


def _effective_whisperx_python(args: argparse.Namespace) -> Path:
    python_arg = getattr(args, "whisperx_python", None)
    return Path(python_arg).expanduser().resolve() if python_arg else default_whisperx_python()


def _whisperx_python_available(args: argparse.Namespace) -> bool:
    python_exe = _effective_whisperx_python(args)
    if python_exe.is_file():
        return True
    if _model_cache_only(args):
        raise ModelCacheMiss("model-cache-miss:whisperx-python-unavailable:" + str(python_exe))
    progress_log(
        "WARNING: WhisperX Python unavailable: "
        + str(python_exe)
        + "; fallback=skip-whisperx-candidate"
    )
    return False


def _effective_raw_asr_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    cli_arg = getattr(args, "whisper_cli", None)
    model_arg = getattr(args, "whisper_model", None)
    cli = Path(cli_arg).expanduser().resolve() if cli_arg else default_whisper_cli()
    model = Path(model_arg).expanduser().resolve() if model_arg else default_whisper_model()
    return cli, model


def _raw_asr_backend_status(args: argparse.Namespace) -> dict[str, object]:
    cli, model = _effective_raw_asr_paths(args)
    missing: list[str] = []
    if not cli.is_file():
        missing.append(f"whisper-cli:{cli}")
    if not model.is_file():
        missing.append(f"whisper-model:{model}")
    available = not missing
    return {
        "available": available,
        "attempted": False,
        "failure_kind": None if available else "backend-unavailable",
        "missing_components": missing,
        "paths": {"whisper_cli": str(cli), "whisper_model": str(model)},
        "fallback": None if available else "empty-raw-asr-events",
    }


def _raw_asr_warning(status: dict[str, object]) -> str:
    failure_kind = str(status.get("failure_kind") or "backend-unavailable")
    missing = status.get("missing_components")
    details = ", ".join(str(item) for item in missing) if isinstance(missing, list) and missing else str(status.get("error") or "unknown error")
    return f"raw ASR unavailable ({failure_kind}): {details}; fallback=empty-raw-asr-events"


def _attach_raw_asr_status(report: dict[str, object], status: dict[str, object]) -> None:
    report["raw_asr_backend"] = copy.deepcopy(status)
    if status.get("failure_kind"):
        report["raw_asr_warning"] = _raw_asr_warning(status)


def default_ctc_ready() -> bool:
    return default_ctc_python().exists() and (Path(__file__).resolve().parent / "ctc_align.py").exists()


DEFAULT_LOCK_TIMEOUT_SECONDS = 1800.0


def _lock_timeout_seconds(environ: Mapping[str, str] | None = None) -> float | None:
    """``LRC_LOCK_TIMEOUT_SECONDS`` (default 1800); ``0`` waits forever."""
    raw = (environ if environ is not None else os.environ).get("LRC_LOCK_TIMEOUT_SECONDS", "")
    if not str(raw).strip():
        return DEFAULT_LOCK_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_LOCK_TIMEOUT_SECONDS
    return value if value > 0 else None


@contextlib.contextmanager
def _exclusive_file_lock(path: Path, *, timeout_seconds: float | None, busy_message: str):
    """Exclusive lock across *processes*, including Windows.

    The lock file persists, but OS byte-range locks are automatically released
    on process termination, even after a hard crash. Never delete a live lock
    file: replacing an inode could permit simultaneous holders.
    ``timeout_seconds=None`` waits indefinitely; a timeout raises
    ``LrcError("lock-timeout:PATH")`` so the song fails like any other error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b", buffering=0) as handle:
        if handle.tell() == 0:
            handle.write(b"0")
        handle.seek(0)
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        if os.name == "nt":
            import msvcrt
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if deadline is not None and time.monotonic() >= deadline:
                        handle.close()
                        raise LrcError(f"lock-timeout:{path} ({busy_message})") from exc
                    time.sleep(0.1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if deadline is not None and time.monotonic() >= deadline:
                        handle.close()
                        raise LrcError(f"lock-timeout:{path} ({busy_message})") from exc
                    time.sleep(0.1)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _try_file_lock(path: Path):
    """Non-blocking ``_exclusive_file_lock``: yields True while this handle holds
    the lock, or False at once when another holder has it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b", buffering=0) as handle:
        if handle.tell() == 0:
            handle.write(b"0")
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                yield False
                return
            try:
                yield True
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def demucs_cache_key_lock(cache_dir: Path, *, timeout_seconds: float = 300.0):
    """Serialize Demucs cache writers across processes."""
    with _exclusive_file_lock(
        cache_dir / ".demucs_writer.lock",
        timeout_seconds=timeout_seconds,
        busy_message="Demucs cache lock busy",
    ):
        yield


WHISPERCPP_GPU_LOCK_PATH = runtime_paths.gpu_lock_dir() / ".whispercpp_gpu.lock"


@contextlib.contextmanager
def whispercpp_gpu_lock():
    """GPU phase for one whisper.cpp run: nothing else resident, one song at a time.

    whisper.cpp large-v3 is the largest model (~4.1 GB sampled).  The global
    WhisperX phase used to start it while the WhisperX model was still resident
    (~2.5 GB), giving the 8.5 GB per-song peak.  So this song's CTC/WhisperX
    helpers are parked first (weights move to CPU and return unchanged on their
    next request), and a cross-process lock lets only one song run whisper.cpp
    at a time, so two songs can share a 12 GB GPU.  Other songs' GPU work
    yields too (``gpu_compute_phase``): the run starts only after their
    in-flight helper request ends, and they park and wait until it is over.
    Scheduling only: inputs, outputs and evidence are unchanged.
    ``LRC_WHISPERCPP_GPU_LOCK=0`` disables the cross-process coordination
    (parking this song's helpers stays).
    """
    _refuse_model_launch_in_cache_only("whispercpp")
    _park_persistent_helper_mode("whisperx", reason="before-whispercpp")
    _park_persistent_helper_mode("ctc", reason="before-whispercpp")
    if not _gpu_coordination_enabled():
        yield
        return
    with _exclusive_file_lock(
        WHISPERCPP_GPU_LOCK_PATH,
        timeout_seconds=_lock_timeout_seconds(),
        busy_message="whisper.cpp GPU lock busy",
    ):
        _WHISPERCPP_HELD_HERE.set()
        try:
            _wait_for_other_gpu_compute()
            yield
        finally:
            _WHISPERCPP_HELD_HERE.clear()


# One lock file per process that has CTC/WhisperX/HuBERT/Demucs work on the
# GPU; whisper.cpp waits until no other process holds its file.
GPU_COMPUTE_LOCK_DIR = runtime_paths.gpu_lock_dir() / ".gpu_compute"


_WHISPERCPP_HELD_HERE = threading.Event()


_GPU_COMPUTE_STATE: dict[str, object] = {"depth": 0, "stack": None}


_GPU_COMPUTE_STATE_LOCK = threading.Lock()


def _gpu_coordination_enabled() -> bool:
    return os.environ.get("LRC_WHISPERCPP_GPU_LOCK", "1").strip() != "0"


def _other_process_runs_whispercpp() -> bool:
    if _WHISPERCPP_HELD_HERE.is_set():
        return False
    with _try_file_lock(WHISPERCPP_GPU_LOCK_PATH) as free:
        return not free


def _other_process_runs_whispercpp_confirmed() -> bool:
    """Two observations 50 ms apart: a peer's own probe holds the lock file for
    a moment, which must not read as a whisper.cpp run (it would park and
    reload models for nothing)."""
    if not _other_process_runs_whispercpp():
        return False
    time.sleep(0.05)
    return _other_process_runs_whispercpp()


WHISPERCPP_PARK_WATCH_SECONDS = 0.5


_PARK_WATCHER: dict[str, object] = {"thread": None}


_PARK_WATCHER_LOCK = threading.Lock()


_PARK_WATCHER_STOP = threading.Event()


def _whispercpp_park_watch() -> None:
    parked = False
    while not _PARK_WATCHER_STOP.wait(WHISPERCPP_PARK_WATCH_SECONDS):
        if not _gpu_coordination_enabled():
            continue
        try:
            busy = _other_process_runs_whispercpp_confirmed()
        except OSError:
            continue
        if not busy:
            parked = False
            continue
        with _GPU_COMPUTE_STATE_LOCK:
            in_flight = int(_GPU_COMPUTE_STATE["depth"]) > 0
        # An in-flight request ends before that run may start (it waits on
        # this process's marker); the next request then parks and waits.
        if not parked and not in_flight:
            _park_persistent_helper_mode("whisperx", reason="other-song-whispercpp")
            _park_persistent_helper_mode("ctc", reason="other-song-whispercpp")
            parked = True


def _ensure_whispercpp_park_watcher() -> None:
    """Resident helpers left loaded between requests (this song busy on CPU)
    are parked as soon as another song's whisper.cpp run starts, not only at
    this song's next GPU request."""
    with _PARK_WATCHER_LOCK:
        thread = _PARK_WATCHER["thread"]
        if isinstance(thread, threading.Thread) and thread.is_alive():
            return
        _PARK_WATCHER_STOP.clear()
        thread = threading.Thread(target=_whispercpp_park_watch, name="whispercpp-park-watch", daemon=True)
        _PARK_WATCHER["thread"] = thread
        thread.start()


def _wait_for_other_gpu_compute() -> None:
    own = f"{os.getpid()}.lock"
    while True:
        busy = False
        if GPU_COMPUTE_LOCK_DIR.is_dir():
            for path in sorted(GPU_COMPUTE_LOCK_DIR.glob("*.lock")):
                if path.name == own:
                    continue
                with _try_file_lock(path) as free:
                    if not free:
                        busy = True
                        break
        if not busy:
            return
        time.sleep(0.1)


def _enter_gpu_compute() -> None:
    with _GPU_COMPUTE_STATE_LOCK:
        depth = int(_GPU_COMPUTE_STATE["depth"])
        if depth == 0:
            stack = contextlib.ExitStack()
            stack.enter_context(
                _exclusive_file_lock(
                    GPU_COMPUTE_LOCK_DIR / f"{os.getpid()}.lock",
                    timeout_seconds=_lock_timeout_seconds(),
                    busy_message="GPU compute marker busy",
                )
            )
            _GPU_COMPUTE_STATE["stack"] = stack
        _GPU_COMPUTE_STATE["depth"] = depth + 1


def _leave_gpu_compute() -> None:
    with _GPU_COMPUTE_STATE_LOCK:
        depth = int(_GPU_COMPUTE_STATE["depth"]) - 1
        _GPU_COMPUTE_STATE["depth"] = depth
        if depth == 0:
            stack = _GPU_COMPUTE_STATE["stack"]
            _GPU_COMPUTE_STATE["stack"] = None
            if isinstance(stack, contextlib.ExitStack):
                stack.close()


@contextlib.contextmanager
def gpu_compute_phase():
    """GPU work of this song (CTC, WhisperX, HuBERT, Demucs) yields to another
    song's whisper.cpp run.

    While that run lasts this song parks its resident helpers and starts no new
    GPU work; while this work lasts the other song's whisper.cpp waits.  The
    peak is then one whisper.cpp plus parked contexts, not whisper.cpp plus the
    other song's live models.  Scheduling only: inputs, outputs and evidence
    are unchanged.
    """
    _refuse_model_launch_in_cache_only("gpu-compute-phase")
    if not _gpu_coordination_enabled():
        yield
        return
    _ensure_whispercpp_park_watcher()
    parked = False
    while True:
        if _other_process_runs_whispercpp_confirmed():
            if not parked:
                _park_persistent_helper_mode("whisperx", reason="other-song-whispercpp")
                _park_persistent_helper_mode("ctc", reason="other-song-whispercpp")
                parked = True
            time.sleep(0.2)
            continue
        _enter_gpu_compute()
        if _other_process_runs_whispercpp():
            # A whisper.cpp run took the lock between the check and the
            # marker; it goes first.
            _leave_gpu_compute()
            continue
        break
    try:
        yield
    finally:
        try:
            _park_before_whispercpp_handoff()
        finally:
            _leave_gpu_compute()


def _park_before_whispercpp_handoff() -> None:
    """A whisper.cpp run of another song waits on this process's marker; it may
    start the moment the marker is released.  The helpers still hold this
    request's working memory then, and the watcher parks them only up to
    ~0.5 s later, so both briefly share the GPU.  The last in-flight request
    parks them before releasing the marker instead."""
    with _GPU_COMPUTE_STATE_LOCK:
        last = int(_GPU_COMPUTE_STATE["depth"]) == 1
    if not last or not _other_process_runs_whispercpp_confirmed():
        return
    _park_persistent_helper_mode("whisperx", reason="other-song-whispercpp-handoff")
    _park_persistent_helper_mode("ctc", reason="other-song-whispercpp-handoff")


def _path_sha256(path: Path) -> str:
    return file_digest.sha256(path)


def _stable_file_sha256(path: Path) -> str:
    return _path_sha256(path)


def _canonical_json_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LrcError(f"Value is not finite canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _bounded_retry_protocol_revision(helper_sha256: str) -> str:
    return _canonical_json_digest({
        "helper_sha256": helper_sha256,
        "protocol": _BOUNDED_RETRY_HELPER_PROTOCOL,
        "required_fields": ["entries", "entry", "text", "romaji", "tokens", "token_spans"],
    })


@dataclass(frozen=True)
class TimingTokenSpan:
    start: float
    end: float
    score: float
    token: str = ""


def _timing_spans_from_value(value: object) -> tuple[TimingTokenSpan, ...]:
    if not isinstance(value, list):
        return ()
    spans: list[TimingTokenSpan] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        spans.append(TimingTokenSpan(start, end, score, str(item.get("char") or "")))
    return tuple(sorted(spans, key=lambda span: (span.start, span.end)))


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
