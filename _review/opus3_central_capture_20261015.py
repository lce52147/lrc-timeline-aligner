"""Capture Central's real inputs for one corpus song from a cache-only replay.

usage (from the repo root):
  python -X utf8 _review/opus3_central_capture_20261015.py <case index> <out dir>

Runs the production command of the full-39 runner in-process with
--model-cache-only (no model launch possible; a cache miss fails the song) and
pickles every build_final_timing_state call whose candidate sets carry a
ctc-local-window-wide row: (candidate_sets, args, kwargs, written seconds).
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(V))

index = int(sys.argv[1])
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
corpus = json.loads((V / "LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json").read_text(encoding="utf8"))["corpus"]
item = corpus[index - 1]
os.environ.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf8", "LRC_EVIDENCE_WORKERS": "1"})

from run_demucs_37_regression_20261008 import build  # noqa: E402

sidecar = {row["title"]: row["sidecar"] for row in build()}.get(item.get("title", item["id"]))

import auto_lrc as L  # noqa: E402

original = L.build_final_timing_state
calls = []


def capturing(candidate_sets, *args, **kwargs):
    state = original(candidate_sets, *args, **kwargs)
    if any(c.source == L.CTC_WIDE_REGION_FALLBACK_SOURCE for items in candidate_sets for c in items):
        calls.append((candidate_sets, args, kwargs, [d.written_time.seconds for d in state.decisions]))
    return state


L.build_final_timing_state = capturing
argv = [
    item["audio_path"], "--lyrics", item["human_BK"],
    "--timing-source", "auto", "--timing-source-policy", "ctc-first",
    "--skeleton", "segment",
    "--output", str(out / "candidate.lrc"), "--report-dir", str(out / "reports"),
    "--arbiter", "reviewer-validity", "--model-cache-only",
]
if sidecar is not None:
    argv += ["--reviewer-evidence", str(sidecar)]
code = L.main(argv)
with (out / "central_capture.pkl").open("wb") as handle:
    pickle.dump({"index": index, "exit": code, "calls": calls}, handle)
print("CAPTURE", index, "exit", code, "wide_calls", len(calls))
