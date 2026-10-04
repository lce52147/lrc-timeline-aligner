from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "_review"
SCRIPTS = ROOT / "scripts"
MODEL = ROOT / "models" / "hfa" / "v0.0.7" / "1218_hfa_model_new_dict"

sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import select_songs  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--song-id", action="append", default=[])
    return result


def load_dictionary(path: Path) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        word, phones = raw.split("\t", 1)
        result[word.strip()] = tuple(phones.strip().split())
    return result


def syllabify(romaji: str, dictionary: dict[str, tuple[str, ...]]) -> list[str] | None:
    normal_keys = tuple(
        sorted(
            (key for key in dictionary if key not in {"SP", "AP", "cl"}),
            key=lambda item: (-len(item), item),
        )
    )

    @functools.lru_cache(maxsize=None)
    def solve(index: int) -> tuple[str, ...] | None:
        if index == len(romaji):
            return ()
        # Pykakasi writes sokuon as a doubled consonant.  HFA's Japanese
        # dictionary represents that closure explicitly as `cl`.
        if (
            index + 1 < len(romaji)
            and romaji[index] == romaji[index + 1]
            and romaji[index] not in "aeioun"
            and "cl" in dictionary
        ):
            tail = solve(index + 1)
            if tail is not None:
                return ("cl",) + tail
        for key in normal_keys:
            if romaji.startswith(key, index):
                tail = solve(index + len(key))
                if tail is not None:
                    return (key,) + tail
        return None

    value = solve(0)
    return list(value) if value is not None else None


def finite_time(row: dict[str, object], source: str) -> float | None:
    sources = row.get("sources")
    item = sources.get(source) if isinstance(sources, dict) else None
    value = item.get("time") if isinstance(item, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def main() -> int:
    args = parser().parse_args()
    import ctc_align

    payload = json.loads(args.data.read_text(encoding="utf-8-sig"))
    selected = select_songs(payload, args.song_id)
    dictionary = load_dictionary(MODEL / "japanese_dict_full.txt")
    converter = ctc_align.build_romanizer()
    tagger = ctc_align.build_unidic_tagger()
    songs: list[dict[str, object]] = []
    supported = 0
    unsupported = 0

    for song_index, song in enumerate(selected, start=1):
        planned_rows: list[dict[str, object]] = []
        rows = song["rows"]
        for index, row in enumerate(rows):
            cur = finite_time(row, "CUR")
            fin = finite_time(row, "FIN")
            if cur is None or fin is None:
                planned_rows.append({
                    "entry": row["entry"],
                    "status": "UNSUPPORTED",
                    "reason": "missing-cur-or-fin",
                })
                unsupported += 1
                continue
            start = max(0.0, min(cur, fin) - 1.5)
            if index + 1 < len(rows):
                next_cur = finite_time(rows[index + 1], "CUR")
                next_fin = finite_time(rows[index + 1], "FIN")
                if next_cur is None or next_fin is None:
                    end = cur + 8.0
                else:
                    end = max(next_cur, next_fin) + 0.5
            else:
                end = cur + 8.0
            end = min(end, start + 30.0)
            if end <= start + 0.05:
                planned_rows.append({
                    "entry": row["entry"],
                    "status": "UNSUPPORTED",
                    "reason": "invalid-window",
                    "window_start": round(start, 6),
                    "window_end": round(end, 6),
                })
                unsupported += 1
                continue
            romaji, romanization_source = ctc_align.transcript_romanize(
                converter, tagger, str(row["text"])
            )
            syllables = syllabify(romaji, dictionary) if romaji else None
            if not syllables:
                planned_rows.append({
                    "entry": row["entry"],
                    "status": "UNSUPPORTED",
                    "reason": "japanese-dictionary-oov",
                    "romaji": romaji,
                    "romanization_source": romanization_source,
                    "window_start": round(start, 6),
                    "window_end": round(end, 6),
                })
                unsupported += 1
                continue
            planned_rows.append({
                "entry": row["entry"],
                "status": "READY",
                "text": row["text"],
                "romaji": romaji,
                "romanization_source": romanization_source,
                "syllables": syllables,
                "lab": " ".join(syllables),
                "window_start": round(start, 6),
                "window_end": round(end, 6),
            })
            supported += 1
        songs.append({
            "song_index": song_index,
            "id": song["id"],
            "title": song["title"],
            "language": song.get("language"),
            "vocal_audio_path": song["vocal_audio_path"],
            "vocal_audio_sha256": song["vocal_audio_sha256"],
            "rows": planned_rows,
        })

    output = {
        "schema": 1,
        "source": "HubertFA-v0.0.7",
        "source_commit": "521281fe52dcedb8f970ca200dc4f00fc2869b08",
        "model_sha256": "4722c13e1d5c1d740ec69c9121f5ec37a88d534aba7637bafb480208be170183",
        "dictionary": "japanese_dict_full.txt",
        "window_policy": "[min(CUR_i,FIN_i)-1.5,max(CUR_i+1,FIN_i+1)+0.5], cap30s; final CUR_last+8",
        "supported_rows": supported,
        "unsupported_rows": unsupported,
        "songs": songs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"supported": supported, "unsupported": unsupported, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
