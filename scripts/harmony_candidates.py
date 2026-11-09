"""Harmony candidates for parenthesized (faint-voice) lyric lines: sidecar generator and reader (WEAK1).

``<song stem>.harmony_cands.json`` holds, per paren *lyric* line, one extra start time found by forced
alignment on the song's harmony stem (HARM2 "W_h").  The Review page only shows it as a chip and an
audio toggle: nothing here changes an LRC time, and a missing or stale sidecar gives no chips.

Sidecar: ``{"schema": 1, "lrc_sha256", "lines_total", "harmony_audio", "lines": [{"entry", "start_cs", "method": "W_h", "window": [a, b]}]}``
``entry`` is the 1-based timed entry of the LRC (title card excluded, same numbering as the Review rows).

Generate (separation stays an offline step; the stems come from HARM1b's karaoke split, ``run_sep``
in H\\venv on the htdemucs vocals -> ``harmony.flac``), then, with the formal venv (torch + torchaudio):
    python -X utf8 scripts/harmony_candidates.py <audio> --lrc <lrc> --report <align-report.json> --stems <dir> --out <sidecar>

W_h: a run of consecutive paren lines is aligned together with MMS_FA on the harmony stem inside the
window [previous non-paren start - 0.5 s, next non-paren start + 0.5 s]; cue lines (no explicit lyric,
EvanOG: 沒有明確歌詞都屬於純提示) are not aligned and get no candidate.
Where the sidecar is looked up (first file whose lrc hash matches wins): beside the audio, then in each of
``LRC_HARMONY_CAND_ROOTS`` (os.pathsep separated) and the report's folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sys

SCHEMA = 1
PAD = 0.5
OPEN, CLOSE = "(（", ")）"
SUFFIX = ".harmony_cands.json"
AUDIO_SUFFIXES = {".flac", ".wav", ".m4a", ".mp3", ".ogg", ".opus"}

# ---- cue classifier: same rules as HARM2-CUES (h2_cues.py) ----
KW = r"(間奏|前奏|後奏|間奏中|イントロ|アウトロ|インスト(ゥルメンタル)?|インタールード|Interlude|Instrumental|Inst\.?|Intro|Outro|Music|BGM|Repeat|リピート|繰り返し|くりかえし|Chorus|コーラス|Solo|ソロ|Rap|ラップ|台詞|セリフ|語り|ナレーション|Narration|Spoken|Dialogue|Bridge|ブリッジ|SE|効果音|Guitar|ギター|Piano|ピアノ|Drum|ドラム|Verse|Hook|Fade\s*out|Fade\s*in|Silence|無音|休止|Pause)"
REP = r"([×xX✕＊*]\s*\d+|\d+\s*[×xX✕回]|[×✕])"
LABEL = r"^[^\s:：()（）]{1,14}\s*[:：]"
SYM = r"[\W_\d]"


def classify(text):
    """``cue`` / ``unsure`` / ``lyric`` for one paren line."""
    s = re.sub(r"\s+", "", text)
    if not s:
        return "cue"
    s = re.sub(r"[\[\]【】()（）「」『』〈〉<>]", "", s)
    hit = False
    s2 = re.sub(REP, "", s)
    hit |= s2 != s
    s = s2
    m = re.match(LABEL, s)
    if m:
        s = s[m.end():]
        hit = True
    s2 = re.sub(KW, "", s, flags=re.I)
    hit |= s2 != s
    s = re.sub(SYM, "", s2, flags=re.U)
    if not s:
        return "cue"
    return "unsure" if hit else "lyric"


def paren_lines(texts, maxn=8):
    """0-based indexes of lines inside a parenthesized block (a block opens a line and closes within ``maxn`` lines)."""
    out, i = set(), 0
    stripped = [re.sub(r"\s+", "", t) for t in texts]
    while i < len(stripped):
        if stripped[i] and stripped[i][0] in OPEN:
            for j in range(i, min(i + maxn, len(stripped))):
                if stripped[j] and stripped[j][-1] in CLOSE:
                    out.update(range(i, j + 1))
                    i = j
                    break
        i += 1
    return out


def paren_runs(is_paren):
    """``[(first, last)]`` runs of consecutive True flags."""
    runs, i = [], 0
    while i < len(is_paren):
        if is_paren[i]:
            j = i
            while j + 1 < len(is_paren) and is_paren[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def run_window(starts, anchor, first, last, duration, pad=PAD):
    """(a, b) seconds: the nearest ``anchor`` (non-paren, non-marker) line before ``first`` / after ``last`` +- ``pad``, clipped to the song."""
    prev = next((starts[k] for k in range(first - 1, -1, -1) if anchor[k]), None)
    nxt = next((starts[k] for k in range(last + 1, len(starts)) if anchor[k]), None)
    a = max(0.0, (prev if prev is not None else 0.0) - pad)
    b = min(duration, (nxt if nxt is not None else duration) + pad)
    return a, b


def plan(entries_text, starts, duration, romaji, markers=None, include_unsure=False):
    """Runs to align: ``[{"first", "last", "window": [a, b], "lines": [0-based lyric line indexes]}]``.

    ``entries_text`` / ``starts`` (seconds) / ``romaji`` have one item per LRC entry; ``markers`` flags marker entries.
    """
    markers = markers or [False] * len(entries_text)
    inside = paren_lines(entries_text)
    flags = [i in inside for i in range(len(entries_text))]
    anchor = [not flags[i] and not markers[i] for i in range(len(entries_text))]
    allowed = ("lyric", "unsure") if include_unsure else ("lyric",)
    out = []
    for first, last in paren_runs(flags):
        lines = [k for k in range(first, last + 1)
                 if not markers[k] and classify(entries_text[k]) in allowed and (romaji[k] or "").strip()]
        if not lines:
            continue
        a, b = run_window(starts, anchor, first, last, duration)
        out.append({"first": first, "last": last, "window": [round(a, 3), round(b, 3)], "lines": lines})
    return out


def lrc_sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _align(harmony_path, runs, romaji, threads=4):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import torchaudio.functional as AF
    torch.set_num_threads(threads)
    sr_model = 16000
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).eval()
    tok = bundle.get_tokenizer()

    def ids_for(text):
        words = re.sub(r"[^a-z']+", " ", (text or "").lower()).split()
        return [t for w in tok(words) for t in w]

    x, sr = sf.read(str(harmony_path), dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x
    wave = AF.resample(torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0), sr, sr_model)[0]
    found = {}
    for run in runs:
        a, b = run["window"]
        ids = [ids_for(romaji[k]) for k in run["lines"]]
        flat = [t for line in ids for t in line]
        if not flat or any(not line for line in ids):
            continue
        crop = wave[int(a * sr_model):int(b * sr_model)]
        with torch.inference_mode():
            emission, _ = model(crop.unsqueeze(0))
            emission = torch.log_softmax(emission.float(), dim=-1)
            if emission.shape[1] < len(flat) * 2 + 1:
                continue
            try:
                alignment, scores = AF.forced_align(emission, torch.tensor([flat], dtype=torch.int32), blank=0)
            except Exception:
                continue
        spans = AF.merge_tokens(alignment[0], scores[0].exp())
        if len(spans) != len(flat):
            continue
        seconds_per_frame = (crop.shape[0] / sr_model) / emission.shape[1]
        at = 0
        for k, line in zip(run["lines"], ids):
            found[k] = round(a + spans[at].start * seconds_per_frame, 3)
            at += len(line)
    return found


def generate(audio, lrc, report, stems, out, include_unsure=False):
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import evaluate_lrc as ev

    lrc = pathlib.Path(lrc)
    _meta, entries = ev.parse_lrc(lrc)
    entries = [e for e in entries if not ev.is_generated_title_card(e)]
    data = json.loads(pathlib.Path(report).read_text(encoding="utf-8"))
    assignments = data.get("assignments") or []
    if len(assignments) != len(entries):
        raise SystemExit(f"report has {len(assignments)} assignments but the LRC has {len(entries)} entries")
    texts = [e.lines[0] if e.lines else "" for e in entries]
    starts = [e.time_cs / 100.0 for e in entries]
    romaji = [(a.get("romaji") or "") for a in assignments]
    markers = [ev.is_marker_entry(e) for e in entries]
    duration = float(data.get("duration_seconds") or (max(starts) + 10.0))
    runs = plan(texts, starts, duration, romaji, markers, include_unsure)
    harmony = pathlib.Path(stems) / "harmony.flac"
    if not harmony.is_file():
        raise SystemExit(f"no harmony stem: {harmony}")
    found = _align(harmony, runs, romaji) if runs else {}
    window_of = {k: run["window"] for run in runs for k in run["lines"]}
    out = pathlib.Path(out)
    try:
        stored = os.path.relpath(harmony, out.parent)
    except ValueError:
        stored = str(harmony)
    payload = {
        "schema": SCHEMA,
        "lrc_sha256": lrc_sha256(lrc),
        "lines_total": len(entries),
        "harmony_audio": stored,
        "lines": [{"entry": k + 1, "start_cs": int(round(t * 100)), "method": "W_h", "window": window_of[k]}
                  for k, t in sorted(found.items())],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return payload


# ---- reader (used by review_server) ----

def sidecar_roots(extra=()):
    roots = [pathlib.Path(p) for p in extra]
    roots += [pathlib.Path(p) for p in os.environ.get("LRC_HARMONY_CAND_ROOTS", "").split(os.pathsep) if p.strip()]
    return roots


def load(audio, lrc_hashes, rows_total, extra_roots=()):
    """``{"harmony_audio": path, "lines": {entry: start_cs}}`` or None.

    ``lrc_hashes``: sha256 values the sidecar may match (the LRC now, and its first-save baseline).  A hash
    mismatch, an entry count that differs from ``rows_total`` or a missing harmony file gives None.
    """
    try:
        audio = pathlib.Path(audio)
        candidates = [audio.with_name(audio.stem + SUFFIX)] + [root / (audio.stem + SUFFIX) for root in sidecar_roots(extra_roots)]
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
                continue
            if payload.get("lrc_sha256") not in set(lrc_hashes) or payload.get("lines_total") != rows_total:
                continue
            harmony = payload.get("harmony_audio")
            if not isinstance(harmony, str) or not harmony:
                continue
            harmony_path = pathlib.Path(harmony)
            if not harmony_path.is_absolute():
                harmony_path = path.parent / harmony_path
            if not harmony_path.is_file() or harmony_path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            lines = {}
            for item in payload.get("lines") or []:
                if (isinstance(item, dict) and isinstance(item.get("entry"), int) and isinstance(item.get("start_cs"), int)
                        and not isinstance(item.get("entry"), bool) and 1 <= item["entry"] <= rows_total and item["start_cs"] >= 0):
                    lines[item["entry"]] = item["start_cs"]
            if lines:
                return {"harmony_audio": str(harmony_path), "lines": lines}
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return None


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio")
    parser.add_argument("--lrc", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--stems", required=True, help="folder with harmony.flac")
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-unsure", action="store_true", help="also align lines the cue rules cannot decide")
    args = parser.parse_args(argv)
    payload = generate(args.audio, args.lrc, args.report, args.stems, args.out, args.include_unsure)
    print(json.dumps({"out": args.out, "candidates": len(payload["lines"])}))


if __name__ == "__main__":
    main()
