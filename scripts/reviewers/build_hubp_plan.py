from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "_review"
MODEL = ROOT / "models" / "hubert-phoneme-ctc"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import select_songs  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--song-id", action="append", default=[])
    return result


def main() -> int:
    args = parser().parse_args()
    import pyopenjtalk

    payload = json.loads(args.data.read_text(encoding="utf-8-sig"))
    selected = select_songs(payload, args.song_id)
    vocab = {str(k): int(v) for k, v in json.loads((MODEL / "vocab.json").read_text(encoding="utf-8")).items()}
    songs: list[dict[str, object]] = []
    supported = 0
    unsupported = 0
    for song_index, song in enumerate(selected, start=1):
        planned_rows: list[dict[str, object]] = []
        if song.get("language") != "japanese":
            for row in song["rows"]:
                planned_rows.append({"entry": row["entry"], "status": "UNSUPPORTED", "reason": "unsupported-language"})
                unsupported += 1
        else:
            for row in song["rows"]:
                try:
                    raw = pyopenjtalk.g2p(str(row["text"]), kana=False)
                    tokens = [item for item in raw.split() if item and item not in {"pau", "sil"}]
                    unknown = sorted({item for item in tokens if item not in vocab})
                    if not tokens:
                        raise ValueError("empty-phoneme-sequence")
                    if unknown:
                        raise ValueError("unknown-phoneme:" + ",".join(unknown))
                    planned_rows.append({
                        "entry": row["entry"], "status": "READY", "text": row["text"],
                        "phonemes": tokens, "token_ids": [vocab[item] for item in tokens],
                    })
                    supported += 1
                except Exception as exc:
                    planned_rows.append({
                        "entry": row["entry"], "status": "UNSUPPORTED",
                        "reason": f"{type(exc).__name__}:{exc}"[:1000],
                    })
                    unsupported += 1
        songs.append({
            "song_index": song_index, "id": song["id"], "title": song["title"],
            "language": song.get("language"), "vocal_audio_path": song["vocal_audio_path"],
            "vocal_audio_sha256": song["vocal_audio_sha256"], "rows": planned_rows,
        })
    result = {
        "schema": 1,
        "source": "prj-beatrice/japanese-hubert-base-phoneme-ctc",
        "model_revision": "1ec4eb3c45b2a1cafb7c477d447df34ca03070f2",
        "model_sha256": "f958c23c16a54d9cca9fcc6e453e4f9054b47bcfe3c1d9313cbeb17975483caf",
        "g2p": "pyopenjtalk-plus==0.4.1.post3",
        "supported_rows": supported,
        "unsupported_rows": unsupported,
        "songs": songs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"supported": supported, "unsupported": unsupported, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
