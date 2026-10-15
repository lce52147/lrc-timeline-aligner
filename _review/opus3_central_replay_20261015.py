"""Re-run Central on captured inputs with the current scripts/auto_lrc.py.

usage (repo root, or a worktree root that has the same _review capture dir):
  python -X utf8 _review/opus3_central_replay_20261015.py <capture root> [case ...]

For every captured wide-region call of each case: replays build_final_timing_state
and prints the rows whose written time changed against the captured run, with
the BK error before/after (BK read here only to score).  Ends with a tally.
Point SCRIPTS at another checkout with env LRC_REPLAY_SCRIPTS.
"""
import json
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

V = Path(__file__).resolve().parent
scripts = Path(os.environ.get("LRC_REPLAY_SCRIPTS", V.parent / "scripts"))
sys.path.insert(0, str(scripts))
import auto_lrc as L  # noqa: E402
import lrc_scoring as S  # noqa: E402

root = Path(sys.argv[1])
wanted = {int(x) for x in sys.argv[2:]}
corpus = json.loads((V / "LRC13_S5_39_CORPUS_PREFLIGHT_20261010.json").read_text(encoding="utf8"))["corpus"]
TAG = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")


def bk_times(path: str) -> list[float]:
    times = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        match = TAG.match(line.strip())
        if match and line.strip()[match.end():].strip():
            times.append(int(match.group(1)) * 60 + float(match.group(2)))
    return times


tally = Counter()
for case_dir in sorted(root.iterdir()):
    if not case_dir.name.isdigit() or (wanted and int(case_dir.name) not in wanted):
        continue
    capture = case_dir / "central_capture.pkl"
    if not capture.is_file():
        continue
    data = pickle.loads(capture.read_bytes())
    index = data["index"]
    if not data["calls"]:
        print(f"c{index}: no wide-region call captured")
        continue
    item = corpus[index - 1]
    bk = [t for _, t in S.parsed_rows(Path(item["human_BK"]), Path(item["audio_path"]))]
    candidate_sets, args, kwargs, before = data["calls"][-1]
    state = L.build_final_timing_state(candidate_sets, *args, **kwargs)
    after = [d.written_time.seconds for d in state.decisions]
    if len(bk) != len(after):
        print(f"c{index}: BK rows {len(bk)} != Central rows {len(after)}; scoring skipped")
        bk = None
    for row, (old, new) in enumerate(zip(before, after)):
        if abs(old - new) < 0.005:
            continue
        if bk is None:
            print(f"c{index} #{row + 1} {old:.2f}->{new:.2f}")
            continue
        e0, e1 = abs(old - bk[row]), abs(new - bk[row])
        verdict = "BETTER" if e1 + 0.05 < e0 else ("WORSE" if e1 > e0 + 0.05 else "same")
        tally[verdict] += 1
        if max(e0, e1) > 1:
            tally[(verdict, ">1s")] += 1
        print(f"c{index} #{row + 1} {old:.2f}->{new:.2f} err {old - bk[row]:+.2f}->{new - bk[row]:+.2f} {verdict}")
    if os.environ.get("LRC_REPLAY_REGION") and bk is not None:
        collapsed = [L.current_ctc_path_collapsed(items) for items in candidate_sets]
        wide_flags = L.ctc_collapsed_region_flags(collapsed, radius=L.CTC_WIDE_REGION_RADIUS)
        for row, decision in enumerate(state.decisions):
            wide = next((c.written_time.seconds for c in candidate_sets[row] if c.source == L.CTC_WIDE_REGION_FALLBACK_SOURCE), None)
            if not wide_flags[row] and wide is None:
                continue
            w = f" wide={wide:.2f}({wide - bk[row]:+.2f})" if wide is not None else ""
            print(f"   R c{index} #{row + 1} {'C' if collapsed[row] else ' '}{'W' if wide_flags[row] else ' '} {decision.status[:11]:11s} {decision.candidate.source[:24]:24s} t={after[row]:.2f} err={after[row] - bk[row]:+.2f}{w}")
    order_breaks = sum(1 for a, b in zip(after, after[1:]) if b <= a)
    if order_breaks:
        print(f"c{index}: {order_breaks} order break(s) in replayed output")
        tally["order_breaks"] += order_breaks
print(dict(tally))
