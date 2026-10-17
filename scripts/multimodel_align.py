"""One-process bounded-window multi-model CTC alignment helper.

The JSON protocol is intentionally path-and-request based so the caller can
persist its complete response under the normal model-output cache. Models are
loaded one at a time; failures for one model remain explicit in
the response and never become an invented timestamp.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Callable

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctc_align as C  # noqa: E402
from multimodel_consensus import (  # noqa: E402
    MODEL_SPECS,
    PROTOCOL_REVISION,
    hf_model_load_options,
    model_identity,
)
SAMPLE_RATE = 16_000
DEFAULT_MODELS = MODEL_SPECS


def _native(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKC", str(text))
        if ch.isalnum()
    ).lower()


def _roman(text: str) -> str:
    converter = C.build_romanizer()
    tagger = C.build_unidic_tagger()
    return C.transcript_romanize(converter, tagger, str(text))[0]


def _first_token(
    log_probs: torch.Tensor,
    line_tokens: list[list[int]],
    blank: int,
    seconds_per_frame: float,
    start: float,
) -> float | None:
    flattened = [token for part in line_tokens for token in part]
    target = line_tokens[-1] if line_tokens else []
    if not flattened or not target or log_probs.shape[0] < len(flattened):
        return None
    target_tensor = torch.tensor(
        [flattened], dtype=torch.int32, device=log_probs.device
    )
    alignment, scores = torchaudio.functional.forced_align(
        log_probs[None], target_tensor, blank=blank
    )
    spans = torchaudio.functional.merge_tokens(
        alignment[0], scores[0].exp()
    )
    first = len(flattened) - len(target)
    if first >= len(spans):
        return None
    return round(start + spans[first].start * seconds_per_frame, 3)


def _window(row: dict[str, Any]) -> tuple[float, float]:
    start = max(0.0, float(row["start"]))
    end = float(row["end"])
    if end <= start:
        raise ValueError("window end must be after start")
    return start, end


def _lines(row: dict[str, Any]) -> list[str]:
    texts = row.get("texts")
    if not isinstance(texts, list) or not all(isinstance(x, str) for x in texts):
        raise ValueError("window texts must be a list of strings")
    target = int(row["target_offset"])
    if not (0 <= target < len(texts)):
        raise ValueError("window target_offset is outside texts")
    return texts[:target + 1]


class HFCtc:
    def __init__(self, name: str, text_mode: str, device: str) -> None:
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForCTC,
            AutoTokenizer,
            Wav2Vec2Processor,
        )

        model_options = hf_model_load_options(name)
        common_options = {
            "revision": model_options["revision"],
            "local_files_only": True,
        }
        self.processor = Wav2Vec2Processor(
            feature_extractor=AutoFeatureExtractor.from_pretrained(name, **common_options),
            tokenizer=AutoTokenizer.from_pretrained(name, **common_options),
        )
        self.model = (
            AutoModelForCTC.from_pretrained(name, **model_options)
            .to(device)
            .eval()
        )
        self.device = device
        self.text_mode = text_mode
        self.vocab = self.processor.tokenizer.get_vocab()
        self.blank = int(self.processor.tokenizer.pad_token_id or 0)
        self.lower = any(
            key.isalpha() and key.islower() for key in self.vocab
        )

    def _tokens(self, text: str) -> list[int]:
        normalized = _roman(text) if self.text_mode == "roman" else _native(text)
        if not self.lower:
            normalized = normalized.upper()
        return [
            int(self.vocab[char])
            for char in normalized
            if char in self.vocab
        ]

    def __call__(self, row: dict[str, Any]) -> float | None:
        start, end = _window(row)
        audio = C.decode_audio(Path(str(row["stem"])), start, end)
        values = audio.squeeze(0).numpy()
        inputs = self.processor(
            values, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        with torch.inference_mode():
            logits = self.model(
                inputs.input_values.to(self.device)
            ).logits[0]
        lines = [_lines(row)[0]] if len(_lines(row)) == 1 else _lines(row)
        tokens = [self._tokens(text) for text in lines]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return _first_token(
            log_probs,
            tokens,
            self.blank,
            (end - start) / log_probs.shape[0],
            start,
        )


class Mms1b(HFCtc):
    def __init__(self, lang: str, text_mode: str, device: str) -> None:
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        name = "facebook/mms-1b-all"
        model_options = hf_model_load_options(name)
        common_options = {
            "revision": model_options["revision"],
            "local_files_only": True,
        }
        self.processor = AutoProcessor.from_pretrained(
            name, target_lang=lang, **common_options
        )
        self.model = (
            Wav2Vec2ForCTC.from_pretrained(
                name, target_lang=lang, ignore_mismatched_sizes=True, **model_options
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.text_mode = text_mode
        self.vocab = self.processor.tokenizer.get_vocab()
        self.blank = int(self.processor.tokenizer.pad_token_id or 0)
        self.lower = any(
            key.isalpha() and key.islower() for key in self.vocab
        )


class TorchaudioEnglish:
    def __init__(self, device: str) -> None:
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
        self.model = bundle.get_model().to(device).eval()
        self.labels = bundle.get_labels()
        self.index = {char: index for index, char in enumerate(self.labels)}
        self.device = device

    def __call__(self, row: dict[str, Any]) -> float | None:
        start, end = _window(row)
        audio = C.decode_audio(Path(str(row["stem"])), start, end).to(self.device)
        with torch.inference_mode():
            emissions, _ = self.model(audio)
        log_probs = torch.log_softmax(emissions[0].float(), dim=-1)
        tokens = [
            [
                self.index[char]
                for char in _roman(text).upper()
                if char in self.index
            ]
            for text in _lines(row)
        ]
        return _first_token(
            log_probs,
            tokens,
            0,
            (end - start) / log_probs.shape[0],
            start,
        )


def _mms(row: dict[str, Any], device: str) -> float | None:
    start, end = _window(row)
    with tempfile.TemporaryDirectory(prefix="lrc-multimodel-") as temp:
        transcript = Path(temp) / "transcript.json"
        output = Path(temp) / "output.json"
        transcript.write_text(
            json.dumps(_lines(row), ensure_ascii=False),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "ctc_align.py"),
            "--audio", str(row["stem"]),
            "--transcript", str(transcript),
            "--output", str(output),
            "--device", device,
            "--start", f"{start:.3f}",
            "--end", f"{end:.3f}",
        ]
        proc = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if proc.returncode or not output.is_file():
            return None
        payload = json.loads(output.read_text(encoding="utf-8"))
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            return None
        value = entries[-1].get("start")
        return round(float(value), 3) if isinstance(value, (int, float)) else None


def _load(spec: str, device: str) -> Callable[[dict[str, Any]], float | None]:
    if spec == "mms":
        return lambda row: _mms(row, device)
    if spec.startswith("mms1b:"):
        _, lang, text_mode = spec.split(":", 2)
        return Mms1b(lang, text_mode, device)
    if spec.startswith("en-ctc:torchaudio:"):
        return TorchaudioEnglish(device)
    if spec.startswith("en-ctc:"):
        _, name, text_mode = spec.split(":", 2)
        return HFCtc(name, text_mode, device)
    if spec.startswith("ja-ctc:"):
        _, name, text_mode = spec.split(":", 2)
        return HFCtc(name, text_mode, device)
    raise ValueError(f"unsupported multimodel spec: {spec}")


def _model_identity(spec: str) -> dict[str, str]:
    return model_identity(spec)

def run_request(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("protocol_revision", PROTOCOL_REVISION) != PROTOCOL_REVISION:
        raise ValueError("unsupported multimodel protocol revision")
    stem = Path(str(request["stem"])).expanduser().resolve()
    if not stem.is_file():
        raise FileNotFoundError(stem)
    device = str(request.get("device") or "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_models = request.get("models") or list(DEFAULT_MODELS)
    models = [str(item) for item in raw_models]
    raw_windows = request.get("windows")
    if not isinstance(raw_windows, list):
        raise ValueError("windows must be a list")
    windows = []
    for raw in raw_windows:
        if not isinstance(raw, dict):
            raise ValueError("window must be an object")
        windows.append({**raw, "stem": str(stem)})
    errors: dict[str, str] = {}
    outputs: dict[str, dict[str, float | None]] = {
        str(int(window["entry_index"])): {spec: None for spec in models}
        for window in windows
    }
    for spec in models:
        model: Callable[[dict[str, Any]], float | None] | None = None
        try:
            model = _load(spec, device)
            for window in windows:
                entry_index = str(int(window["entry_index"]))
                try:
                    outputs[entry_index][spec] = model(window)
                except Exception as exc:
                    errors.setdefault(spec, f"run:{type(exc).__name__}:{exc}")
        except Exception as exc:
            errors[spec] = f"load:{type(exc).__name__}:{exc}"
        finally:
            del model
            gc.collect()
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "device": device,
        "models": models,
        "model_identities": {spec: _model_identity(spec) for spec in models},
        "windows": outputs,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    source = (
        args.request.read_text(encoding="utf-8")
        if args.request is not None
        else sys.stdin.read()
    )
    request = json.loads(source)
    response = run_request(request)
    sys.stdout.write(json.dumps(response, ensure_ascii=False, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


