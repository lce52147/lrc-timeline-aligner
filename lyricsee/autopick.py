"""Pick the right lyric candidate without a human: cross-source agreement + ASR match.

For every song left in needs_review: clean each saved candidate to the original
language, drop snippets, score each against a Whisper transcript of the audio,
and auto-accept only when the decision rule below is met (provenance
verification = auto_consensus). Everything else stays needs_review.

  python autopick.py run --out OUT --audio-map audio.json [--stem STEM ...]
  python autopick.py calibrate --txt-dir DIR --audio-map audio.json --cache-dir DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from candidate_review import accept_candidate, load_candidates_metadata
from candidate_text import AGREE_SIM, asr_match, char_similarity, clean_lyric_lines, fold
from translation_posts import (
    MIN_POST_LINES, WRONG_SONG_ASR, apply_post_translation, load_post_rows, post_is_complete, transcript_reliable,
)

# --- thresholds, calibrated on 12 verified songs (README "Auto-pick"): correct lyrics scored
# 0.25-0.62 against their own audio (2.6-6.4 cleaned chars/s), wrong songs at most 0.13 ------
MIN_CHARS_PER_SEC = 1.0     # cleaned candidate shorter than this per audio second is a snippet
MIN_CHARS_ABS = 60          # ... or shorter than this in absolute terms
ASR_FLOOR = 0.20            # best candidate must reach this ASR score
ASR_MARGIN = 0.10           # ... and beat the next *different* candidate by this
ASR_STRICT_FLOOR = 0.40     # without a second agreeing source the ASR score must reach this
# AGREE_SIM (candidate_text): candidate-to-candidate similarity that counts as "same lyric"

DEFAULT_WHISPER_LANGUAGE = "auto"


def is_snippet(cleaned_chars: int, duration: float | None) -> bool:
    need = MIN_CHARS_ABS
    if duration:
        need = max(need, MIN_CHARS_PER_SEC * duration)
    return cleaned_chars < need


# --- ASR transcript ---------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def whisper_paths() -> tuple[Path, Path]:
    scripts = str(_repo_root() / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import runtime_paths  # type: ignore

    return runtime_paths.whisper_cli(), runtime_paths.whisper_model()


MIN_WAV_BYTES = 1024


def _ffmpeg_wav16k(source: Path, wav: Path, run=subprocess.run) -> None:
    run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
        check=True, capture_output=True,
    )


def _wav_is_empty(wav: Path) -> bool:
    return not wav.exists() or wav.stat().st_size < MIN_WAV_BYTES


def decode_to_wav16k(audio: Path, wav: Path, native_wav: Path, run=subprocess.run) -> str:
    """16 kHz mono WAV of ``audio``; returns the decoder that produced it ('ffmpeg' or 'libsndfile').

    ffmpeg exits 0 with a 44-byte-header WAV and no samples for a FLAC whose first frames are damaged (the
    demuxer reads the stream start, finds no frame, and reports EOF; the audio is not path- or argument-related).
    libFLAC, behind soundfile, resynchronises, so an empty ffmpeg result is retried through it: decode to a
    native-rate WAV in the work dir, then let ffmpeg resample that. The source file is only read.
    """
    _ffmpeg_wav16k(audio, wav, run)
    if not _wav_is_empty(wav):
        return "ffmpeg"
    try:
        import soundfile
    except ImportError as exc:
        raise RuntimeError("audio_undecodable: ffmpeg wrote an empty WAV and soundfile is not installed") from exc
    try:
        samples, rate = soundfile.read(str(audio), dtype="int16", always_2d=True)
        soundfile.write(str(native_wav), samples, rate, subtype="PCM_16", format="WAV")
        _ffmpeg_wav16k(native_wav, wav, run)
    except Exception as exc:
        raise RuntimeError(f"audio_undecodable: ffmpeg wrote an empty WAV and libsndfile failed ({exc})") from exc
    finally:
        native_wav.unlink(missing_ok=True)
    if _wav_is_empty(wav):
        raise RuntimeError("audio_undecodable: no decoder produced samples")
    return "libsndfile"


def transcribe(audio: Path, cache_dir: Path, work_dir: Path | None = None) -> dict:
    """Whisper (whisper.cpp large-v3, the LRC tools path) transcript of the audio, cached under cache_dir.

    Nothing is written beside the audio: the WAV decode and JSON go to work_dir.
    """
    stat = audio.stat()
    key = hashlib.sha256(f"{audio}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")).hexdigest()[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    cli, model = whisper_paths()
    own_tmp = None
    if work_dir is None:
        own_tmp = tempfile.TemporaryDirectory(prefix="autopick_asr_")
        work_dir = Path(own_tmp.name)
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / f"{key}.wav"
    try:
        decode_to_wav16k(audio, wav, work_dir / f"{key}.native.wav")
        base = work_dir / key
        proc = subprocess.run(
            [str(cli), "-m", str(model), "-l", DEFAULT_WHISPER_LANGUAGE, "-oj", "-of", str(base), str(wav)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        out_json = base.with_suffix(".json")
        if proc.returncode != 0 or not out_json.exists():
            raise RuntimeError(f"whisper-cli failed ({proc.returncode}): {proc.stderr.strip()[-400:]}")
        payload = json.loads(out_json.read_text(encoding="utf-8", errors="replace"))
    finally:
        wav.unlink(missing_ok=True)
        if own_tmp:
            own_tmp.cleanup()
    segments = [
        str(item.get("text", "")).strip()
        for item in payload.get("transcription", [])
        if str(item.get("text", "")).strip()
    ]
    result = {
        "audio": str(audio),
        "language": (payload.get("result") or {}).get("language", ""),
        "text": "\n".join(segments),
    }
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def audio_duration(audio: Path) -> float | None:
    from lyricsee_core import probe_audio

    return probe_audio(audio).duration


# --- scoring + decision -----------------------------------------------------

def score_candidates(
    raw_lines_by_idx: dict[int, list[str]],
    urls: dict[int, str],
    transcript_text: str,
    duration: float | None,
) -> list[dict]:
    """One entry per candidate: cleaned/folded text, snippet flag, ASR scores, similarity to the others."""
    transcript = fold(transcript_text)
    entries: list[dict] = []
    for idx, raw in raw_lines_by_idx.items():
        cleaned = clean_lyric_lines(raw)
        text = fold("".join(cleaned))
        entries.append({
            "idx": idx,
            "url": urls.get(idx, ""),
            "cleaned_lines": len(cleaned),
            "cleaned_chars": len(text),
            "folded": text,
            "snippet": is_snippet(len(text), duration),
            "asr": asr_match(text, transcript),
        })
    for entry in entries:
        entry["sim"] = {
            other["idx"]: round(char_similarity(entry["folded"], other["folded"]), 4)
            for other in entries if other is not entry
        }
    return entries


def decide(entries: list[dict], asr_reliable: bool = True) -> dict:
    """Auto-accept rule. Returns {'decision': 'accept'|'review', 'reason', 'idx'?, ...}.

    A translation post (dedicated 【歌詞翻譯】 page for this song, see translation_posts) is complete by
    construction: it is accepted on its title and shape without the snippet rule, and the ASR score only
    vetoes it when the transcript is real singing that clearly belongs to another song. The page must also
    name the artist (or the album/folder), unless the ASR itself reaches ASR_FLOOR.
    """
    verdict = _decide_by_asr(entries)
    if verdict["decision"] == "accept":
        return verdict
    posts = [e for e in entries if e.get("translation_post") and e.get("post_complete")]
    if not posts:
        return verdict
    best = max(posts, key=lambda e: (e["asr"]["score"], e["cleaned_chars"]))
    score = best["asr"]["score"]
    info = {"idx": best["idx"], "asr_score": score, "asr_reliable": asr_reliable}
    if asr_reliable and score < WRONG_SONG_ASR:
        return {"decision": "review", "reason": "translation_post_contradicted_by_asr", **info}
    # same-titled posts exist for other artists' songs: the page has to name the artist (or the album the
    # file sits in), unless a real transcript confirms the words
    if not best.get("artist_confirmed") and not (asr_reliable and score >= ASR_FLOOR):
        return {"decision": "review", "reason": "translation_post_artist_unconfirmed", **info}
    return {"decision": "accept", "reason": "translation_post_title_match", **info}


def _decide_by_asr(entries: list[dict]) -> dict:
    eligible = [e for e in entries if not e["snippet"]]
    if not eligible:
        return {"decision": "review", "reason": "no_full_length_candidate"}
    ranked = sorted(eligible, key=lambda e: e["asr"]["score"], reverse=True)
    best = ranked[0]
    score = best["asr"]["score"]
    info = {"idx": best["idx"], "asr_score": score}
    if score < ASR_FLOOR:
        return {"decision": "review", "reason": "asr_below_floor", **info}
    # candidates that agree with the best are the same lyric, not rivals
    rivals = [e for e in ranked[1:] if best["sim"].get(e["idx"], 0.0) < AGREE_SIM]
    next_score = max((e["asr"]["score"] for e in rivals), default=0.0)
    info["next_different_score"] = next_score
    if score - next_score < ASR_MARGIN:
        return {"decision": "review", "reason": "asr_margin_too_small", **info}
    agreeing = [
        e for e in eligible
        if e is not best and e["url"] != best["url"]
        and best["sim"].get(e["idx"], 0.0) >= AGREE_SIM and e["asr"]["score"] >= ASR_FLOOR
    ]
    if agreeing:
        info["agreeing_idx"] = [e["idx"] for e in agreeing]
        return {"decision": "accept", "reason": "asr_best_and_sources_agree", **info}
    if score >= ASR_STRICT_FLOOR:
        return {"decision": "accept", "reason": "asr_above_strict_floor", **info}
    return {"decision": "review", "reason": "no_independent_agreement", **info}


# --- short tracks -----------------------------------------------------------

INSTRUMENTAL_MAX_SECONDS = 100.0   # a track shorter than this with no lyric page naming it is likely an interlude


def instrumental_evidence(duration: float | None, entries: list[dict], transcript_text: str) -> dict | None:
    """Evidence that a short track is an instrumental / interlude, or None.

    Only a hint for a human, never a fact: the track is under INSTRUMENTAL_MAX_SECONDS and no saved candidate is
    a full-length text that the audio supports (ASR score >= ASR_FLOOR).  Whisper's own text on such a track is
    recorded so a reader can see whether any singing was heard.
    """
    if not duration or duration >= INSTRUMENTAL_MAX_SECONDS:
        return None
    if any(not e["snippet"] and e["asr"]["score"] >= ASR_FLOOR for e in entries):
        return None
    heard = len(fold(transcript_text))
    return {
        "duration": round(duration, 1),
        "candidates": len(entries),
        "full_length_candidates": sum(1 for e in entries if not e["snippet"]),
        "best_asr_score": max((e["asr"]["score"] for e in entries), default=None),
        "transcript_chars": heard,
        "transcript_chars_per_sec": round(heard / duration, 2),
        "transcript_reliable": transcript_reliable(transcript_text),
        "note": "short track and no lyric page for it that the audio supports; not proof",
    }


# --- per-song run -----------------------------------------------------------

def autopick_song(
    output_root: Path,
    stem: str,
    transcript_text: str,
    duration: float | None,
    *,
    write: bool = True,
) -> dict:
    """Score the saved candidates of one stem, update candidates.json, accept if the rule says so."""
    metadata = load_candidates_metadata(output_root, stem)
    if not metadata:
        evidence = instrumental_evidence(duration, [], transcript_text)
        if evidence:
            return {"stem": stem, "duration": duration, "decision": "review",
                    "reason": "likely_instrumental_or_interlude", "likely_instrumental_or_interlude": evidence}
        return {"stem": stem, "decision": "skip", "reason": "no_candidates"}
    cand_dir = output_root / "candidates" / stem
    raw: dict[int, list[str]] = {}
    for idx in range(len(metadata)):
        path = cand_dir / f"candidate_{idx}.txt"
        raw[idx] = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    urls = {idx: str(item.get("url", "")) for idx, item in enumerate(metadata)}
    entries = score_candidates(raw, urls, transcript_text, duration)
    for entry in entries:
        entry["translation_post"] = bool(metadata[entry["idx"]].get("translation_post"))
        entry["artist_confirmed"] = bool(metadata[entry["idx"]].get("artist_confirmed"))
        entry["post_complete"] = entry["translation_post"] and entry["cleaned_lines"] >= MIN_POST_LINES             and post_is_complete(load_post_rows(output_root, stem, entry["idx"]))
    verdict = decide(entries, asr_reliable=transcript_reliable(transcript_text))
    if verdict["decision"] == "review":
        evidence = instrumental_evidence(duration, entries, transcript_text)
        if evidence:
            evidence["decided_otherwise_as"] = verdict["reason"]
            verdict = {**verdict, "reason": "likely_instrumental_or_interlude", "likely_instrumental_or_interlude": evidence}

    for entry in entries:
        item = metadata[entry["idx"]]
        item["cleaned_lines"] = entry["cleaned_lines"]
        item["cleaned_chars"] = entry["cleaned_chars"]
        item["snippet"] = entry["snippet"]
        item["similarity_to_others"] = {str(k): v for k, v in entry["sim"].items()}
        item["asr"] = entry["asr"]
        if entry["translation_post"]:
            item["post_complete"] = entry["post_complete"]
    if write:
        (cand_dir / "candidates.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if verdict["decision"] == "accept":
            best = next(e for e in entries if e["idx"] == verdict["idx"])
            accept_candidate(
                output_root, stem, verdict["idx"],
                verification_status="auto_consensus",
                verified_by="autopick",
                extra={
                    "autopick": {
                        "reason": verdict["reason"],
                        "asr": best["asr"],
                        "next_different_score": verdict.get("next_different_score"),
                        "agreeing_idx": verdict.get("agreeing_idx", []),
                        "similarity_to_others": {str(k): v for k, v in best["sim"].items()},
                        "thresholds": thresholds(),
                    }
                },
            )
            # a post that agrees with the accepted original supplies the human translation, whichever
            # source the original came from; the lines it lacks are filled by `translation_posts apply --oogg`
            result = apply_post_translation(output_root, stem)
            if result["status"] != "no_post":
                verdict["translation"] = result
    return {
        "stem": stem,
        "duration": duration,
        **verdict,
        "candidates": [
            {k: e[k] for k in ("idx", "url", "cleaned_lines", "cleaned_chars", "snippet", "asr", "sim")}
            for e in entries
        ],
        "translation_posts": [e["idx"] for e in entries if e["translation_post"]],
    }


OFF_VOCAL_REASON = "no_lyrics_found_after_tag_and_google_search"


def mark_off_vocal(output_root: Path, stem: str, *, queries: int = 0, detail: str = "") -> Path:
    """Record that tag-based and Google searching found no lyric page: no lyrics file, no LRC, and
    ``cmd_run`` leaves the stem alone until ``--include-off-vocal`` (EvanOG: no lyrics means off vocal)."""
    path = output_root / "off_vocal" / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"stem": stem, "status": "off_vocal", "reason": OFF_VOCAL_REASON, "google_cdp_queries": queries,
              "detail": detail, "marked_at": datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def is_off_vocal(output_root: Path, stem: str) -> bool:
    return (output_root / "off_vocal" / f"{stem}.json").is_file()


def thresholds() -> dict:
    return {
        "MIN_CHARS_PER_SEC": MIN_CHARS_PER_SEC, "MIN_CHARS_ABS": MIN_CHARS_ABS,
        "ASR_FLOOR": ASR_FLOOR, "ASR_MARGIN": ASR_MARGIN,
        "ASR_STRICT_FLOOR": ASR_STRICT_FLOOR, "AGREE_SIM": AGREE_SIM,
        "MIN_POST_LINES": MIN_POST_LINES, "WRONG_SONG_ASR": WRONG_SONG_ASR,
    }


# --- CLI --------------------------------------------------------------------

def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    audio_map = json.loads(Path(args.audio_map).read_text(encoding="utf-8-sig"))
    cache_dir = Path(args.cache_dir) if args.cache_dir else out.parent / "asr_cache"
    stems = args.stem or sorted(p.name for p in (out / "candidates").iterdir() if p.is_dir())
    results: list[dict] = []
    for stem in stems:
        if (out / "txt" / f"{stem}.lyrics.txt").exists():
            _log(f"{stem}: already has lyrics, skipped")
            continue
        if is_off_vocal(out, stem) and not args.include_off_vocal:
            _log(f"{stem}: off_vocal, skipped")
            continue
        metadata = load_candidates_metadata(out, stem) or []
        if metadata and all(item.get("rejected_by_user") for item in metadata) and not args.include_rejected:
            _log(f"{stem}: rejected by user, skipped")
            continue
        audio = Path(audio_map.get(stem, ""))
        if not audio.is_file():
            _log(f"{stem}: audio not found ({audio})")
            results.append({"stem": stem, "decision": "review", "reason": "audio_not_found"})
            continue
        try:
            transcript = transcribe(audio, cache_dir, out.parent / "asr_work")
            result = autopick_song(out, stem, transcript["text"], audio_duration(audio))
        except Exception as exc:  # one bad song must not stop the batch
            _log(f"{stem}: ERROR {exc}")
            results.append({"stem": stem, "decision": "review", "reason": f"error: {exc}"})
            continue
        _log(f"{stem}: {result['decision']} ({result['reason']}) idx={result.get('idx')} asr={result.get('asr_score')}")
        results.append(result)
    report = out / "report" / f"autopick_{datetime.now():%Y%m%d_%H%M%S}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps({"thresholds": thresholds(), "songs": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"report: {report}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """ASR score of verified lyrics against their own audio (positive) and other songs' audio (negative)."""
    audio_map = json.loads(Path(args.audio_map).read_text(encoding="utf-8-sig"))
    cache_dir = Path(args.cache_dir)
    rows: list[dict] = []
    transcripts: dict[str, str] = {}
    for txt in sorted(Path(args.txt_dir).glob("*.lyrics.txt"))[: args.limit]:
        stem = txt.name[: -len(".lyrics.txt")]
        audio = Path(audio_map.get(stem, ""))
        if not audio.is_file():
            continue
        _log(f"{stem}: transcribing")
        transcripts[stem] = fold(transcribe(audio, cache_dir, cache_dir.parent / "asr_work")["text"])
        text = fold("".join(clean_lyric_lines(txt.read_text(encoding="utf-8").splitlines())))
        duration = audio_duration(audio)
        rows.append({
            "stem": stem, "text": text, "duration": duration, "chars": len(text),
            "chars_per_sec": round(len(text) / duration, 3) if duration else None,
            "own": asr_match(text, transcripts[stem]),
        })
    for row in rows:
        row["wrong"] = max(
            (asr_match(row["text"], t)["score"] for s, t in transcripts.items() if s != row["stem"]),
            default=0.0,
        )
        del row["text"]
    Path(args.result).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        _log(f"{row['stem']}: own={row['own']['score']} wrong_max={row['wrong']} cps={row['chars_per_sec']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--audio-map", required=True)
    run.add_argument("--cache-dir")
    run.add_argument("--stem", action="append")
    run.add_argument("--include-rejected", action="store_true")
    run.add_argument("--include-off-vocal", action="store_true", help="also retry stems marked off_vocal")
    run.set_defaults(func=cmd_run)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--txt-dir", required=True)
    cal.add_argument("--audio-map", required=True)
    cal.add_argument("--cache-dir", required=True)
    cal.add_argument("--result", required=True)
    cal.add_argument("--limit", type=int, default=12)
    cal.set_defaults(func=cmd_calibrate)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
