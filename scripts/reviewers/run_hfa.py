from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "_review"
HFA_SOURCE = ROOT / "tools" / "HubertFA"
MODEL = ROOT / "models" / "hfa" / "v0.0.7" / "1218_hfa_model_new_dict"

sys.path.insert(0, str(HFA_SOURCE))
from onnx_infer import InferenceOnnx  # noqa: E402
from tools.g2p import DictionaryG2P  # noqa: E402


class CUDAInferenceOnnx(InferenceOnnx):
    @staticmethod
    def create_session(onnx_path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(onnx_path), options, providers=["CUDAExecutionProvider"])


class CPUInferenceOnnx(InferenceOnnx):
    @staticmethod
    def create_session(onnx_path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_audio(path: Path, sample_rate: int = 44_100) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sample_rate),
            "-f", "s16le", "-",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace")[-2000:])
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def load_output(plan: dict[str, object], output_path: Path) -> dict[str, object]:
    if output_path.exists():
        current = json.loads(output_path.read_text(encoding="utf-8-sig"))
        if current.get("model_sha256") != plan.get("model_sha256"):
            raise RuntimeError("existing HFA output model identity mismatch")
        return current
    return {
        "schema": 1,
        "source": "HFA",
        "status": "RUNNING",
        "provider": "CPUExecutionProvider",
        "source_commit": plan["source_commit"],
        "model_sha256": plan["model_sha256"],
        "sample_rate": 44100,
        "hop_size": 441,
        "rows": {},
        "failures": [],
    }


def save(output: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_lexical_start(words) -> tuple[float | None, str | None]:
    for word in words:
        for phoneme in word.phonemes:
            text = str(phoneme.text)
            if text not in {"AP", "SP", ""}:
                return float(phoneme.start), text
    return None, None


def load_dataset_with_training_prefix_rules(inference, wav: Path, lab_text: str) -> None:
    """Build one HFA item using the same silent-prefix rule as training.

    v0.0.7's inference DictionaryG2P prefixes every non-SP phone with the
    language, which turns the model's global silent token `cl` into the
    nonexistent `ja/cl`.  The training binarizer keeps all configured silent
    phonemes unprefixed.  Reproduce that general rule here without changing
    third-party source or the dictionary.
    """
    raw_g2p = DictionaryG2P(None, MODEL / "japanese_dict_full.txt")
    ph_seq, word_seq, ph_idx_to_word_idx = raw_g2p(lab_text)
    silent = {str(value) for value in inference.vocab.get("silent_phonemes", [])}
    prefixed = [
        phone if phone in silent or "/" in phone else f"ja/{phone}"
        for phone in ph_seq
    ]
    inference.dataset.append((wav, prefixed, word_seq, ph_idx_to_word_idx))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--temp-dir", type=Path, default=REVIEW / "hfa" / "tmp_hfa")
    result.add_argument("--max-rows", type=int, default=0)
    result.add_argument("--retry-failed", action="store_true")
    result.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    return result


def main() -> int:
    args = parser().parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    output = load_output(plan, args.output)
    rows_out: dict[str, object] = output["rows"]  # type: ignore[assignment]
    model_path = MODEL / "model.onnx"
    if sha256_file(model_path) != str(plan["model_sha256"]):
        raise RuntimeError("HFA model hash mismatch")

    inference_cls = CUDAInferenceOnnx if args.provider == "cuda" else CPUInferenceOnnx
    inference = inference_cls(model_path)
    inference.load_config()
    inference.init_decoder()
    inference.load_model()
    providers = inference.model.get_providers()
    expected_provider = "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    if not providers or providers[0] != expected_provider:
        raise RuntimeError(f"unexpected HFA providers: {providers}")
    output["provider"] = expected_provider
    output["provider_chain"] = providers

    processed_this_run = 0
    total = len(plan["songs"])
    args.temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        for song in plan["songs"]:
            audio: np.ndarray | None = None
            for row in song["rows"]:
                key = f"{song['id']}::{row['entry']}"
                if key in rows_out and not (
                    args.retry_failed and rows_out[key].get("status") == "FAILED"
                ):
                    continue
                if row["status"] != "READY":
                    rows_out[key] = {
                        "song_id": song["id"], "entry": row["entry"], "time": None,
                        "status": "MISSING", "reason": row.get("reason", "unsupported"),
                    }
                    save(output, args.output)
                    continue
                if args.max_rows and processed_this_run >= args.max_rows:
                    output["status"] = "PARTIAL"
                    save(output, args.output)
                    print(json.dumps({"status": "PARTIAL", "processed": processed_this_run}))
                    return 0
                if audio is None:
                    audio_path = Path(str(song["vocal_audio_path"]))
                    if sha256_file(audio_path) != str(song["vocal_audio_sha256"]):
                        raise RuntimeError(f"vocal hash mismatch: {song['id']}")
                    audio = decode_audio(audio_path)
                start = float(row["window_start"])
                end = float(row["window_end"])
                left = max(0, int(round(start * 44100)))
                right = min(len(audio), int(round(end * 44100)))
                if right <= left:
                    rows_out[key] = {
                        "song_id": song["id"], "entry": row["entry"], "time": None,
                        "status": "MISSING", "reason": "empty-audio-window",
                    }
                    save(output, args.output)
                    continue
                work = args.temp_dir / f"s{int(song['song_index']):02d}_e{int(row['entry']):03d}"
                if work.exists():
                    shutil.rmtree(work)
                work.mkdir(parents=True)
                wav = work / "row.wav"
                lab = work / "row.lab"
                sf.write(wav, audio[left:right], 44100, subtype="PCM_16")
                lab.write_text(str(row["lab"]), encoding="utf-8")
                try:
                    inference.dataset.clear()
                    inference.predictions.clear()
                    load_dataset_with_training_prefix_rules(inference, wav, str(row["lab"]))
                    if len(inference.dataset) != 1:
                        raise RuntimeError(f"HFA dataset cardinality {len(inference.dataset)}")
                    inference.infer(non_lexical_phonemes="AP", pad_times=1, pad_length=5)
                    if len(inference.predictions) != 1:
                        raise RuntimeError(f"HFA prediction cardinality {len(inference.predictions)}")
                    _, _, words = inference.predictions[0]
                    rel_start, first_phone = first_lexical_start(words)
                    if rel_start is None or not math.isfinite(rel_start):
                        raise RuntimeError("no-non-ap-sp-phoneme")
                    absolute = start + rel_start
                    rows_out[key] = {
                        "song_id": song["id"], "entry": row["entry"], "time": round(absolute, 6),
                        "status": "OK", "window_start": start, "window_end": end,
                        "relative_start": round(rel_start, 6), "first_phoneme": first_phone,
                        "romaji": row["romaji"], "romanization_source": row["romanization_source"],
                    }
                except Exception as exc:
                    rows_out[key] = {
                        "song_id": song["id"], "entry": row["entry"], "time": None,
                        "status": "FAILED", "reason": f"{type(exc).__name__}:{exc}"[:2000],
                    }
                    output["failures"].append({"song_id": song["id"], "entry": row["entry"], "error": rows_out[key]["reason"]})
                finally:
                    shutil.rmtree(work, ignore_errors=True)
                processed_this_run += 1
                save(output, args.output)
                print(f"HFA {song['song_index']}/{total} entry={row['entry']} {rows_out[key]['status']}", flush=True)
    finally:
        shutil.rmtree(args.temp_dir, ignore_errors=True)

    ready = sum(1 for item in rows_out.values() if item.get("status") == "OK")
    missing = sum(1 for item in rows_out.values() if item.get("status") == "MISSING")
    failed = sum(1 for item in rows_out.values() if item.get("status") == "FAILED")
    output["status"] = "COMPLETE"
    output["summary"] = {"ok": ready, "missing": missing, "failed": failed, "total": len(rows_out)}
    output["failures"] = [
        {"song_id": item.get("song_id"), "entry": item.get("entry"), "error": item.get("reason")}
        for item in rows_out.values()
        if item.get("status") == "FAILED"
    ]
    save(output, args.output)
    print(json.dumps(output["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
