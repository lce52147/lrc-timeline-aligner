"""Single source of truth for project-relative runtime paths (LRC 1.5 W2).

Every path that the pipeline derives from the repository location or from the
environment is resolved here, so that ``auto_lrc.py``, ``r2_pipeline.py``,
``doctor.py``, ``multimodel_consensus.py`` and the helper scripts agree on one
rule per path.

Environment overrides (each is passed through ``expandvars``, ``expanduser``
and ``resolve``):

``LRC_VENV_PYTHON``     interpreter for WhisperX / CTC helpers and reviewers
``LRC_WHISPER_CLI``     whisper.cpp ``whisper-cli`` executable
``LRC_WHISPER_MODEL``   whisper.cpp GGML model file
``LRC_FFMPEG``          ffmpeg executable (default: ``ffmpeg`` on PATH)
``HF_HUB_CACHE`` / ``HF_HOME``  Hugging Face hub cache (see ``hf_hub_cache``)
``LRC_MODEL_CACHE_DIR`` overrides ``cache_root()`` (``outputs/cache``); the
                        model-output, vocal and GPU-lock directories follow it
``LRC_REPORT_DIR``      default alignment report directory

With no environment set, every function returns exactly the path the 1.4
code computed from ``Path(__file__).resolve().parents[1]``, so cache
identities derived from ``str(path)`` are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

ENV_VENV_PYTHON = "LRC_VENV_PYTHON"
ENV_WHISPER_CLI = "LRC_WHISPER_CLI"
ENV_WHISPER_MODEL = "LRC_WHISPER_MODEL"
ENV_FFMPEG = "LRC_FFMPEG"
ENV_HF_HUB_CACHE = "HF_HUB_CACHE"
ENV_HF_HOME = "HF_HOME"
ENV_MODEL_CACHE_DIR = "LRC_MODEL_CACHE_DIR"
ENV_REPORT_DIR = "LRC_REPORT_DIR"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, environ: "os._Environ[str] | dict[str, str] | None" = None) -> Path | None:
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    return Path(os.path.expandvars(raw)).expanduser().resolve()


def project_root() -> Path:
    return _PROJECT_ROOT


def venv_python() -> Path:
    override = _env_path(ENV_VENV_PYTHON)
    if override is not None:
        return override
    return _PROJECT_ROOT / ".venv-asr" / "Scripts" / "python.exe"


def whisper_cli() -> Path:
    override = _env_path(ENV_WHISPER_CLI)
    if override is not None:
        return override
    return _PROJECT_ROOT / "tools" / "whisper.cpp" / "Release" / "whisper-cli.exe"


def whisper_model() -> Path:
    override = _env_path(ENV_WHISPER_MODEL)
    if override is not None:
        return override
    return _PROJECT_ROOT / "models" / "whisper.cpp" / "ggml-large-v3.bin"


def ffmpeg() -> str:
    """ffmpeg executable: an absolute override, otherwise ``"ffmpeg"`` on PATH."""
    override = _env_path(ENV_FFMPEG)
    if override is not None:
        return str(override)
    return "ffmpeg"


def hf_hub_cache(environ: "os._Environ[str] | dict[str, str] | None" = None) -> Path:
    """The Hugging Face hub cache directory.

    ``HF_HUB_CACHE`` is used as given; otherwise ``HF_HOME/hub``; otherwise
    ``~/.cache/huggingface/hub``.  A value that already ends in ``hub`` is not
    suffixed again, so ``.../huggingface`` and ``.../huggingface/hub`` map to
    the same directory.
    """
    env = os.environ if environ is None else environ
    hub = _env_path(ENV_HF_HUB_CACHE, env)
    if hub is not None:
        return hub
    home = _env_path(ENV_HF_HOME, env)
    if home is not None:
        return home if home.name == "hub" else home / "hub"
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def hf_hub_cache_candidates() -> list[Path]:
    """``hf_hub_cache()`` first, then the user default, without duplicates."""
    candidates = [hf_hub_cache(), (Path.home() / ".cache" / "huggingface" / "hub").resolve()]
    return list(dict.fromkeys(candidates))


def cache_root() -> Path:
    override = _env_path(ENV_MODEL_CACHE_DIR)
    if override is not None:
        return override
    return _PROJECT_ROOT / "outputs" / "cache"


def model_cache_dir() -> Path:
    return cache_root() / "model-outputs"


def vocal_cache_dir() -> Path:
    return cache_root() / "vocals"


def gpu_lock_dir() -> Path:
    """Directory holding ``.whispercpp_gpu.lock`` and ``.gpu_compute/``.

    Follows ``cache_root()`` so that a run pointed at another cache does not
    contend with (or depend on) the formal install's lock files.
    """
    return cache_root()


def reports_dir() -> Path:
    override = _env_path(ENV_REPORT_DIR)
    if override is not None:
        return override
    return _PROJECT_ROOT / "outputs" / "reports"


def probes_dir() -> Path:
    return _PROJECT_ROOT / "outputs" / "probes"


def _is_junction(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        if os.name == "nt" and path.exists():
            # Junctions report as reparse points without being symlinks.
            attrs = os.lstat(path).st_file_attributes  # type: ignore[attr-defined]
            return bool(attrs & 0x400)
    except OSError:
        return False
    return False


def describe() -> list[dict[str, object]]:
    """Rows for ``doctor.py``: name, value, source, exists, junction."""
    rows: list[dict[str, object]] = []
    specs: Iterable[tuple[str, object, str | None]] = (
        ("project_root", project_root(), None),
        ("venv_python", venv_python(), ENV_VENV_PYTHON),
        ("whisper_cli", whisper_cli(), ENV_WHISPER_CLI),
        ("whisper_model", whisper_model(), ENV_WHISPER_MODEL),
        ("ffmpeg", ffmpeg(), ENV_FFMPEG),
        ("hf_hub_cache", hf_hub_cache(), ENV_HF_HUB_CACHE),
        ("cache_root", cache_root(), ENV_MODEL_CACHE_DIR),
        ("model_cache_dir", model_cache_dir(), ENV_MODEL_CACHE_DIR),
        ("vocal_cache_dir", vocal_cache_dir(), ENV_MODEL_CACHE_DIR),
        ("gpu_lock_dir", gpu_lock_dir(), ENV_MODEL_CACHE_DIR),
        ("reports_dir", reports_dir(), ENV_REPORT_DIR),
    )
    for name, value, env_name in specs:
        source = "default"
        if name == "hf_hub_cache":
            if os.environ.get(ENV_HF_HUB_CACHE, "").strip():
                source = f"env:{ENV_HF_HUB_CACHE}"
            elif os.environ.get(ENV_HF_HOME, "").strip():
                source = f"env:{ENV_HF_HOME}"
        elif env_name and os.environ.get(env_name, "").strip():
            source = f"env:{env_name}"
        path = Path(str(value))
        exists = path.exists() if os.path.isabs(str(value)) else bool(__import__("shutil").which(str(value)))
        rows.append({
            "name": name,
            "value": str(value),
            "source": source,
            "exists": bool(exists),
            "junction": _is_junction(path) if os.path.isabs(str(value)) else False,
        })
    return rows
