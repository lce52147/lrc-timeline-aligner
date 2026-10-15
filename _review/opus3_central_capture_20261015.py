"""Capture Central's real inputs for one corpus song from a cache-only replay.

usage (from the repo root):
  python -X utf8 _review/opus3_central_capture_20261015.py <case index> <out dir>

Runs the production command of the full-39 runner in-process with
--model-cache-only (no model launch possible; a cache miss fails the song) and
pickles every build_final_timing_state call that is the reviewer-validity
commit or carries a ctc-local-window-wide row: (candidate_sets, args, kwargs,
written seconds).  The last call is the commit.
`opus3_central_replay_20261015.py` then re-runs Central on those inputs with
the current code in about a second per song.  BK is only the lyric text input
here, exactly as in the runner; its timestamps are used later only to score.
"""
import json
import os
import pickle
import sys
from pathlib import Path

V = Path(__file__).resolve().parent
ROOT = V.parent
# LRC_CAPTURE_SCRIPTS runs another checkout's scripts (a worktree) against this
# folder's caches, so a dev branch reuses the filled model cache.
SCRIPTS = Path(os.environ.get("LRC_CAPTURE_SCRIPTS", ROOT / "scripts"))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(V))

index = int(sys.argv[1])
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
corpus = json.loads((V / "LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json").read_text(encoding="utf8"))["corpus"]
item = corpus[index - 1]
os.environ.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf8", "LRC_EVIDENCE_WORKERS": "1"})

import auto_lrc  # noqa: E402,F401  (bind SCRIPTS' copy before the runner module imports one)
from run_demucs_37_regression_20261008 import build  # noqa: E402

sidecar = {row["title"]: row["sidecar"] for row in build()}.get(item.get("title", item["id"]))

import auto_lrc as L  # noqa: E402

if L.PROJECT_ROOT != ROOT:
    cache = ROOT / "outputs" / "cache"
    L.DEFAULT_VOCAL_CACHE_DIR = cache / "vocals"
    L.DEFAULT_MODEL_CACHE_DIR = cache / "model-outputs"
    L.WHISPERCPP_GPU_LOCK_PATH = cache / ".whispercpp_gpu.lock"
    L.GPU_COMPUTE_LOCK_DIR = cache / ".gpu_compute"
    # tools, models and the ASR venv live only in the main folder
    L.default_whisper_cli = lambda: ROOT / "tools" / "whisper.cpp" / "Release" / "whisper-cli.exe"
    L.default_whisper_model = lambda: ROOT / "models" / "whisper.cpp" / "ggml-large-v3.bin"
    L.default_whisperx_python = lambda: ROOT / ".venv-asr" / "Scripts" / "python.exe"

original = L.build_final_timing_state
calls = []


def capturing(candidate_sets, *args, **kwargs):
    state = original(candidate_sets, *args, **kwargs)
    if kwargs.get("arbiter_mode") == "reviewer-validity" or any(
        c.source == L.CTC_WIDE_REGION_FALLBACK_SOURCE for items in candidate_sets for c in items
    ):
        calls.append((candidate_sets, args, kwargs, [d.written_time.seconds for d in state.decisions]))
    return state


L.build_final_timing_state = capturing
argv = [
    item["audio_path"], "--lyrics", item["human_BK"],
    "--timing-source", "auto", "--timing-source-policy", "ctc-first",
    "--skeleton", "segment",
    "--output", str(out / "candidate.lrc"), "--report-dir", str(out / "reports"),
    "--arbiter", "reviewer-validity",
]
# LRC_CAPTURE_FILL=1: a generation change needs new model requests; let them run
# (and be cached) instead of failing the song on a cache miss.
if os.environ.get("LRC_CAPTURE_FILL") != "1":
    argv.append("--model-cache-only")
if sidecar is not None:
    argv += ["--reviewer-evidence", str(sidecar)]
code = L.main(argv)
with (out / "central_capture.pkl").open("wb") as handle:
    pickle.dump({"index": index, "exit": code, "calls": calls}, handle)
print("CAPTURE", index, "exit", code, "wide_calls", len(calls))
