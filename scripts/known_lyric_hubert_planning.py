"""Extracted from postcentral_recovery.py (split2): known-lyric HuBERT contract constants and helper source, request planning, dummy candidate/tail and current-onset builders."""

from __future__ import annotations

import collections
import hashlib
import math
import statistics
import unicodedata
from numeral_reading import numerals_for_alignment
from local_whisper_retry import CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS, LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS
from report_trust import _RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY, _RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY, _timing_failure_requires_review
from timing_core import EvaluatedBackendBundle, IndependentCurrentOnsetEvidence, PREFIX_ACOUSTIC_ASSOCIATION_SECONDS, TimingCandidate, analyze_candidate_regions, derive_reliable_tail
from lyrics_io import LyricEntry, entry_sung_text, normalize_match_text, optional_kakasi_converter
from audio_features import HOP_SECONDS
from runtime_base import _timing_content_digest, canonicalize_lrc_time
from runtime_base import TimingTokenSpan
from central_commit import _KNOWN_LYRIC_HUBERT_LINEAGE, _KNOWN_LYRIC_HUBERT_ONSET_PRODUCER, _known_lyric_hubert_current_onset_is_valid, _known_lyric_hubert_current_onset_payload
from typing import Mapping
from timing_core import CandidateRegionEvidence, FinalTimingState, TailRegionEvidence
from backend_bundle import _bundle_candidate_lookup, _postcentral_rebind_existing_peer_conflicts


_KNOWN_LYRIC_HUBERT_PRODUCER = "hubert-ja-known-lyric-terminal"


_KNOWN_LYRIC_HUBERT_MODEL_ID = "TKU410410103/hubert-base-japanese-asr"


_KNOWN_LYRIC_HUBERT_MODEL_REVISION = "9206637a0f07d78aa9101ff9439a9d7fe1a43778"


_KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256 = "d6f5034777c0542db6c587d57db497206d0500f7eca8fb8bc7962301f45d54d9"


_KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION = "known-lyric-hubert-joint-terminal-v7"


_KNOWN_LYRIC_HUBERT_LANGUAGE = "ja-hiragana"


_KNOWN_LYRIC_HUBERT_AUDIO_LOADER = "python-wave-pcm16-mono-16k-v1"


_KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT = "embedded-viterbi-token-emission-geometry-group-confidence-hypothesis-neutral-v5"


_HUBERT_EXISTING_CANDIDATE_RECOVERY_MODE = (
    "precommit-hubert-existing-candidate-central-recovery-v1"
)


_PEER_EXISTING_CANDIDATE_RECOVERY_MODE = (
    "precommit-independent-peer-existing-candidate-central-recovery-v1"
)


_PEER_DUPLICATE_RECOMPUTE_KEY = "postcentral_duplicate_occurrence_offset_recompute"


_HUBERT_RECOVERY_DIRECT_ONSET_KIND = "known-lyric-onset"


_KNOWN_LYRIC_HUBERT_HELPER_SOURCE = '#!/usr/bin/env python3\nfrom __future__ import annotations\nimport argparse\nimport array\nfrom dataclasses import dataclass\nimport hashlib\nimport importlib.metadata\nimport json\nimport math\nimport sys\nimport unicodedata\nimport wave\nfrom pathlib import Path\n\nEXPECTED_MODEL_ID = "__LRC_KNOWN_LYRIC_HUBERT_MODEL_ID__"\nEXPECTED_MODEL_COMMIT = "__LRC_KNOWN_LYRIC_HUBERT_MODEL_REVISION__"\nEXPECTED_MODEL_BLOB_SHA256 = "__LRC_KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256__"\nPROTOCOL_REVISION = "__LRC_KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION__"\nSAMPLE_RATE = 16000\nMIN_INPUT_SAMPLES = 400\n\n\ndef sha256_file(path: Path) -> str:\n    h = hashlib.sha256()\n    with path.open("rb") as f:\n        for block in iter(lambda: f.read(1024 * 1024), b""):\n            h.update(block)\n    return h.hexdigest()\n\n\ndef locate_model_blob() -> Path | None:\n    candidates: list[Path] = []\n    try:\n        from huggingface_hub import try_to_load_from_cache\n        value = try_to_load_from_cache(\n            EXPECTED_MODEL_ID, "model.safetensors", revision=EXPECTED_MODEL_COMMIT\n        )\n        if isinstance(value, str):\n            candidates.append(Path(value))\n    except Exception:\n        pass\n    try:\n        from huggingface_hub.constants import HF_HUB_CACHE\n        repo = EXPECTED_MODEL_ID.replace("/", "--")\n        candidates.append(\n            Path(HF_HUB_CACHE) / f"models--{repo}" / "snapshots" /\n            EXPECTED_MODEL_COMMIT / "model.safetensors"\n        )\n    except Exception:\n        pass\n    for path in candidates:\n        try:\n            if path.is_file() and sha256_file(path) == EXPECTED_MODEL_BLOB_SHA256:\n                return path\n        except OSError:\n            continue\n    return None\n\n\ndef load_pcm16_mono_16k(path: Path):\n    import torch\n    with wave.open(str(path), "rb") as wf:\n        if wf.getnchannels() != 1:\n            raise ValueError(f"wav-channel-count:{wf.getnchannels()}")\n        if wf.getframerate() != SAMPLE_RATE:\n            raise ValueError(f"wav-sample-rate:{wf.getframerate()}")\n        if wf.getsampwidth() != 2:\n            raise ValueError(f"wav-sample-width:{wf.getsampwidth()}")\n        if wf.getcomptype() != "NONE":\n            raise ValueError(f"wav-compression:{wf.getcomptype()}")\n        raw = wf.readframes(wf.getnframes())\n    values = array.array("h")\n    values.frombytes(raw)\n    if sys.byteorder != "little":\n        values.byteswap()\n    if not values:\n        raise ValueError("wav-empty")\n    return torch.tensor(values, dtype=torch.float32).div_(32768.0).unsqueeze(0)\n\n\ndef clean_reading(text: str, vocab: dict[str, int]) -> str:\n    normalized = unicodedata.normalize("NFKC", text)\n    clean: list[str] = []\n    for char in normalized:\n        if char.isspace():\n            if "|" in vocab:\n                clean.append("|")\n            continue\n        if char in vocab:\n            clean.append(char)\n            continue\n        if unicodedata.category(char).startswith("P"):\n            continue\n        raise ValueError(f"unsupported-reading-character:{ord(char):04x}")\n    return "".join(clean)\n\n\ndef resolve_device(requested: str, cuda_available: bool) -> tuple[str, str]:\n    requested_device = str(requested).strip() or "auto"\n    if requested_device.lower() == "auto":\n        resolved_device = "cuda" if cuda_available else "cpu"\n    else:\n        resolved_device = requested_device\n    return requested_device, resolved_device\n\n\ndef blank_id(processor, vocab: dict[str, int]) -> int:\n    token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)\n    if isinstance(token_id, int) and token_id >= 0:\n        return token_id\n    for token in ("<pad>", "[PAD]", "[pad]"):\n        if token in vocab:\n            return int(vocab[token])\n    raise ValueError("blank-token-unavailable")\n\n\n@dataclass(frozen=True)\nclass Point:\n    token_index: int\n    time_index: int\n    score: float\n    emitted: bool\n\n\n@dataclass(frozen=True)\nclass Segment:\n    label: str\n    start: int\n    end: int\n    score: float\n\n\ndef get_trellis(emission, tokens: list[int], blank: int):\n    import torch\n    num_frame = emission.size(0)\n    num_tokens = len(tokens)\n    if num_tokens <= 0 or num_frame < num_tokens:\n        raise ValueError("ctc-insufficient-frames")\n    trellis = torch.empty((num_frame + 1, num_tokens + 1))\n    trellis[0, 0] = 0\n    trellis[1:, 0] = torch.cumsum(emission[:, blank], 0)\n    trellis[0, -num_tokens:] = -float("inf")\n    trellis[-num_tokens:, 0] = float("inf")\n    for t in range(num_frame):\n        trellis[t + 1, 1:] = torch.maximum(\n            trellis[t, 1:] + emission[t, blank],\n            trellis[t, :-1] + emission[t, tokens],\n        )\n    return trellis\n\n\ndef backtrack(trellis, emission, tokens: list[int], blank: int):\n    j = trellis.size(1) - 1\n    t_start = int(trellis[:, j].argmax().item())\n    path: list[Point] = []\n    for t in range(t_start, 0, -1):\n        stayed = trellis[t - 1, j] + emission[t - 1, blank]\n        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]\n        use_change = bool(changed > stayed)\n        token_id = tokens[j - 1] if use_change else blank\n        prob = float(emission[t - 1, token_id].exp().item())\n        path.append(Point(j - 1, t - 1, prob, use_change))\n        if use_change:\n            j -= 1\n        if j == 0:\n            break\n    if j != 0:\n        return None\n    return path[::-1]\n\n\ndef merge_repeats(path: list[Point], transcript: str) -> list[Segment]:\n    # Geometry and support confidence are intentionally separated.  The actual\n    # token-emission frame owns the conservative character bounds, while the\n    # full contiguous CTC path group for that transcript token owns confidence.\n    # Blank-stay frames therefore remain visible as positive inter-token gaps\n    # without throwing away the support score semantics used by the frozen tail\n    # reliability policy.\n    i1 = i2 = 0\n    segments: list[Segment] = []\n    expected_index = 0\n    while i1 < len(path):\n        while i2 < len(path) and path[i1].token_index == path[i2].token_index:\n            i2 += 1\n        group = path[i1:i2]\n        emitted = [point for point in group if point.emitted]\n        if len(emitted) != 1:\n            raise ValueError("ctc-token-emission-cardinality-mismatch")\n        point = emitted[0]\n        if point.token_index != expected_index or expected_index >= len(transcript):\n            raise ValueError("ctc-token-emission-order-mismatch")\n        support_score = sum(item.score for item in group) / len(group)\n        segments.append(Segment(\n            transcript[expected_index],\n            point.time_index,\n            point.time_index + 1,\n            support_score,\n        ))\n        expected_index += 1\n        i1 = i2\n    if expected_index != len(transcript):\n        raise ValueError("ctc-token-emission-count-mismatch")\n    return segments\n\n\ndef align_chars(emission, transcript: str, vocab: dict[str, int], blank: int, start: float, end: float):\n    tokens = [int(vocab[char]) for char in transcript]\n    trellis = get_trellis(emission, tokens, blank)\n    path = backtrack(trellis, emission, tokens, blank)\n    if path is None:\n        raise ValueError("backtrack-unavailable")\n    merged = merge_repeats(path, transcript)\n    if len(merged) != len(transcript):\n        raise ValueError("incomplete-character-alignment")\n    frame_count = emission.size(0)\n    if frame_count <= 0:\n        raise ValueError("invalid-emission-frame-count")\n    ratio = (end - start) / frame_count\n    chars: list[dict[str, object]] = []\n    for item in merged:\n        char_start = float(item.start) * ratio + start\n        char_end = float(item.end) * ratio + start\n        score = float(item.score)\n        if not all(math.isfinite(v) for v in (char_start, char_end, score)):\n            raise ValueError("nonfinite-character-geometry")\n        if char_end < char_start:\n            raise ValueError("negative-character-duration")\n        chars.append({\n            "char": item.label,\n            "start": round(char_start, 6),\n            "end": round(char_end, 6),\n            "score": round(score, 8),\n        })\n    return chars, ratio\n\n\ndef main() -> int:\n    ap = argparse.ArgumentParser()\n    ap.add_argument("--audio", required=True)\n    ap.add_argument("--request", required=True)\n    ap.add_argument("--output", required=True)\n    ap.add_argument("--device", default="cuda")\n    args = ap.parse_args()\n\n    import torch\n    from transformers import AutoModelForCTC, AutoProcessor\n\n    try:\n        transformers_version = importlib.metadata.version("transformers")\n        huggingface_hub_version = importlib.metadata.version("huggingface-hub")\n        torch_version = importlib.metadata.version("torch")\n    except importlib.metadata.PackageNotFoundError as exc:\n        raise SystemExit(f"runtime-package-unavailable:{exc.name}")\n    requested_device, resolved_device = resolve_device(\n        args.device, bool(torch.cuda.is_available())\n    )\n    if resolved_device.startswith("cuda") and not torch.cuda.is_available():\n        raise SystemExit("cuda-unavailable")\n\n    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))\n    if payload.get("protocol_revision") != PROTOCOL_REVISION:\n        raise SystemExit("protocol-revision-mismatch")\n    requests = payload.get("requests")\n    if not isinstance(requests, list):\n        raise SystemExit("malformed-request-list")\n\n    audio_t = load_pcm16_mono_16k(Path(args.audio))\n    processor = AutoProcessor.from_pretrained(\n        EXPECTED_MODEL_ID, revision=EXPECTED_MODEL_COMMIT\n    )\n    model = AutoModelForCTC.from_pretrained(\n        EXPECTED_MODEL_ID, revision=EXPECTED_MODEL_COMMIT\n    ).to(resolved_device)\n    model.eval()\n    model_blob = locate_model_blob()\n    if model_blob is None:\n        raise SystemExit("model-blob-revision-unavailable")\n\n    tokenizer = getattr(processor, "tokenizer", None)\n    if tokenizer is None or not hasattr(tokenizer, "get_vocab"):\n        raise SystemExit("tokenizer-vocab-unavailable")\n    vocab = {str(k): int(v) for k, v in tokenizer.get_vocab().items()}\n    blank = blank_id(processor, vocab)\n    vocab_revision = hashlib.sha256(\n        json.dumps(vocab, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")\n    ).hexdigest()\n    max_duration = float(audio_t.shape[1]) / SAMPLE_RATE\n    results: list[dict[str, object]] = []\n    for req in requests:\n        request_kind = str(req.get("request_kind", "pair-tail"))\n        result: dict[str, object] = {\n            "request_revision": req.get("request_revision"),\n            "conflict_entry": req.get("conflict_entry"),\n            "request_kind": request_kind,\n            "status": "invalid",\n        }\n        try:\n            start = float(req["window_start"])\n            end = float(req["window_end"])\n            if not (\n                math.isfinite(start) and math.isfinite(end)\n                and 0.0 <= start < end <= max_duration + 1e-3\n            ):\n                raise ValueError("window-invalid")\n            f1 = max(0, int(start * SAMPLE_RATE))\n            f2 = min(audio_t.shape[1], int(end * SAMPLE_RATE))\n            wav = audio_t[:, f1:f2]\n            if wav.shape[-1] < MIN_INPUT_SAMPLES:\n                wav = torch.nn.functional.pad(wav, (0, MIN_INPUT_SAMPLES - wav.shape[-1]))\n            inputs = processor(\n                wav.squeeze(0).numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt"\n            )\n            input_values = inputs.input_values.to(resolved_device)\n            attention_mask = getattr(inputs, "attention_mask", None)\n            if attention_mask is not None:\n                attention_mask = attention_mask.to(resolved_device)\n            with torch.inference_mode():\n                output = model(input_values, attention_mask=attention_mask)\n                emission = torch.log_softmax(output.logits, dim=-1)[0].cpu().detach()\n\n            if request_kind == "pair-tail":\n                previous_reading = str(req["previous_reading"])\n                current_reading = str(req["current_reading"])\n                previous_clean = clean_reading(previous_reading, vocab)\n                current_clean = clean_reading(current_reading, vocab)\n                if len(previous_clean) < 2 or len(current_clean) < 2:\n                    raise ValueError("clean-reading-too-short")\n                combined = previous_clean + current_clean\n                chars, ratio = align_chars(\n                    emission, combined, vocab, blank, start, end\n                )\n                boundary = len(previous_clean)\n                previous_chars = chars[:boundary]\n                current_chars = chars[boundary:]\n                if (\n                    len(previous_chars) != len(previous_clean)\n                    or len(current_chars) != len(current_clean)\n                    or not previous_chars or not current_chars\n                ):\n                    raise ValueError("pair-boundary-mismatch")\n                if previous_chars[-1]["end"] > current_chars[0]["start"] + 0.08:\n                    raise ValueError("pair-order-overlap")\n                frame_slack = max(0.08, ratio * 4.0)\n                if current_chars[-1]["end"] >= end - frame_slack:\n                    raise ValueError("right-edge-pileup")\n                if previous_chars[0]["start"] <= start + min(0.04, frame_slack * 0.5):\n                    raise ValueError("left-edge-pileup")\n                result.update({\n                    "status": "aligned",\n                    "reason": "complete-known-lyric-hiragana-pair",\n                    "previous_clean_length": len(previous_clean),\n                    "current_clean_length": len(current_clean),\n                    "previous_chars": previous_chars,\n                    "current_chars": current_chars,\n                    "previous_first_start": previous_chars[0]["start"],\n                    "previous_last_end": previous_chars[-1]["end"],\n                    "current_first_start": current_chars[0]["start"],\n                    "current_last_end": current_chars[-1]["end"],\n                    "frame_seconds": round(ratio, 8),\n                })\n            elif request_kind == "current-neutral-onset-probe":\n                current_reading = str(req["current_reading"])\n                current_clean = clean_reading(current_reading, vocab)\n                if len(current_clean) < 2:\n                    raise ValueError("clean-reading-too-short")\n                current_chars, ratio = align_chars(\n                    emission, current_clean, vocab, blank, start, end\n                )\n                if len(current_chars) != len(current_clean) or not current_chars:\n                    raise ValueError("current-alignment-cardinality-mismatch")\n                frame_slack = max(0.08, ratio * 4.0)\n                if current_chars[-1]["end"] >= end - frame_slack:\n                    raise ValueError("right-edge-pileup")\n                if current_chars[0]["start"] <= start + min(0.04, frame_slack * 0.5):\n                    raise ValueError("left-edge-pileup")\n                result.update({\n                    "status": "aligned",\n                    "reason": "complete-known-lyric-hiragana-current-hypothesis-neutral",\n                    "current_clean_length": len(current_clean),\n                    "current_chars": current_chars,\n                    "current_first_start": current_chars[0]["start"],\n                    "current_last_end": current_chars[-1]["end"],\n                    "frame_seconds": round(ratio, 8),\n                })\n            else:\n                raise ValueError(f"unsupported-request-kind:{request_kind}")\n        except Exception as exc:\n            result["reason"] = f"{type(exc).__name__}:{exc}"\n        results.append(result)\n    output = {\n        "schema_version": 1,\n        "protocol_revision": PROTOCOL_REVISION,\n        "transformers_package_version": transformers_version,\n        "huggingface_hub_package_version": huggingface_hub_version,\n        "torch_package_version": torch_version,\n        "model_id": EXPECTED_MODEL_ID,\n        "model_revision": EXPECTED_MODEL_COMMIT,\n        "model_blob_sha256": EXPECTED_MODEL_BLOB_SHA256,\n        "vocab_revision": vocab_revision,\n        "requested_device": requested_device,\n        "device": resolved_device,\n        "language": "__LRC_KNOWN_LYRIC_HUBERT_LANGUAGE__",\n        "audio_loader": "__LRC_KNOWN_LYRIC_HUBERT_AUDIO_LOADER__",\n        "ctc_alignment": "__LRC_KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT__",\n        "results": results,\n    }\n    Path(args.output).write_text(\n        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"\n    )\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'

for _known_lyric_hubert_contract_token, _known_lyric_hubert_contract_value in (
    ("__LRC_KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION__", _KNOWN_LYRIC_HUBERT_PROTOCOL_REVISION),
    ("__LRC_KNOWN_LYRIC_HUBERT_MODEL_ID__", _KNOWN_LYRIC_HUBERT_MODEL_ID),
    ("__LRC_KNOWN_LYRIC_HUBERT_MODEL_REVISION__", _KNOWN_LYRIC_HUBERT_MODEL_REVISION),
    ("__LRC_KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256__", _KNOWN_LYRIC_HUBERT_MODEL_BLOB_SHA256),
    ("__LRC_KNOWN_LYRIC_HUBERT_LANGUAGE__", _KNOWN_LYRIC_HUBERT_LANGUAGE),
    ("__LRC_KNOWN_LYRIC_HUBERT_AUDIO_LOADER__", _KNOWN_LYRIC_HUBERT_AUDIO_LOADER),
    ("__LRC_KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT__", _KNOWN_LYRIC_HUBERT_CTC_ALIGNMENT),
):
    _KNOWN_LYRIC_HUBERT_HELPER_SOURCE = _KNOWN_LYRIC_HUBERT_HELPER_SOURCE.replace(
        _known_lyric_hubert_contract_token, _known_lyric_hubert_contract_value
    )

del _known_lyric_hubert_contract_token, _known_lyric_hubert_contract_value


def _known_lyric_hubert_helper_revision() -> str:
    return hashlib.sha256(
        _KNOWN_LYRIC_HUBERT_HELPER_SOURCE.encode("utf-8")
    ).hexdigest()


def _known_lyric_hiragana_reading(text: str) -> str | None:
    """Return a deterministic pykakasi hiragana projection of immutable lyrics.

    The reading itself is revision-bound in every request.  If the optional
    converter is unavailable or leaves an empty reading, this producer simply
    does not run; no transliteration is guessed inside the acoustic helper.
    """
    converter = optional_kakasi_converter()
    if converter is None:
        return None
    try:
        converted = converter.convert(numerals_for_alignment(unicodedata.normalize("NFKC", text)))  # type: ignore[attr-defined]
    except Exception:
        return None
    pieces: list[str] = []
    for item in converted:
        if not isinstance(item, dict):
            return None
        value = item.get("hira")
        if not isinstance(value, str):
            return None
        pieces.append(value)
    reading = unicodedata.normalize("NFKC", "".join(pieces)).strip()
    return reading or None


def _postcentral_residual_material_conflict_entries(
    report: dict[str, object],
    state: FinalTimingState,
) -> tuple[int, ...]:
    """Return peer rows still material after all already-available tail evidence."""
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    bundles = report.get(_RUNTIME_POSTCENTRAL_EVIDENCE_BUNDLES_KEY)
    if not isinstance(seed, dict) or not (
        isinstance(bundles, tuple) and bundles
        and all(isinstance(item, EvaluatedBackendBundle) for item in bundles)
    ):
        return ()
    rebound = _postcentral_rebind_existing_peer_conflicts(seed, state, bundles)
    details = rebound.get("cross_backend_material_conflicts")
    rows = details.get("entries") if isinstance(details, dict) else None
    if not isinstance(rows, list):
        return ()
    return tuple(sorted({
        int(row["entry"])
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("entry"), int)
        and int(row["entry"]) > 1
        and isinstance(row.get("clusters"), list)
        and bool(row.get("clusters"))
    }))


def _postcentral_plan_known_lyric_hubert_requests(
    conflict_entries: tuple[int, ...],
    state: FinalTimingState,
    entries: list[LyricEntry],
    duration: float,
) -> tuple[dict[str, object], ...]:
    """Plan one immutable predecessor/current lyric pair per residual conflict."""
    if len(state.decisions) != len(entries) or not math.isfinite(duration) or duration <= 0.0:
        return ()
    readings = tuple(
        _known_lyric_hiragana_reading(entry_sung_text(entry))
        for entry in entries
    )
    reading_pair_counts = collections.Counter(
        (readings[index - 1], readings[index])
        for index in range(1, len(readings))
        if readings[index - 1] and readings[index]
    )
    requests: list[dict[str, object]] = []
    for entry_number in conflict_entries:
        index = entry_number - 1
        if not 0 < index < len(entries):
            continue
        previous_text = entry_sung_text(entries[index - 1])
        current_text = entry_sung_text(entries[index])
        if not normalize_match_text(previous_text) or not normalize_match_text(current_text):
            continue
        previous_reading = readings[index - 1]
        current_reading = readings[index]
        if not previous_reading or not current_reading:
            continue
        if reading_pair_counts[(previous_reading, current_reading)] != 1:
            # HuBERT sees readings, not kanji identity.  A repeated adjacent
            # reading pair cannot independently prove which lyric occurrence
            # produced the acoustic path, so it is excluded before inference.
            continue
        previous_start = state.decisions[index - 1].written_time.seconds
        current_start = state.decisions[index].written_time.seconds
        before_previous = state.decisions[index - 2].written_time.seconds if index >= 2 else 0.0
        next_start = state.decisions[index + 1].written_time.seconds if index + 1 < len(entries) else duration
        spacing = max(HOP_SECONDS, current_start - previous_start)
        lower = before_previous + HOP_SECONDS if index >= 2 else 0.0
        window_start = max(
            0.0,
            lower,
            previous_start - min(2.5, max(1.25, spacing * 0.5)),
        )
        window_end = min(duration, next_start + 1.5, current_start + 12.0)
        if not 2.0 <= window_end - window_start <= 24.0:
            continue
        transcript_revision = _timing_content_digest({
            "previous_text": previous_text,
            "current_text": current_text,
            "previous_reading": previous_reading,
            "current_reading": current_reading,
            "phonetic_system": "pykakasi-hiragana-v1",
        })
        row_payload = {
            "previous_entry_index": index - 1,
            "current_entry_index": index,
            "previous_text_revision": _timing_content_digest(previous_text),
            "current_text_revision": _timing_content_digest(current_text),
            "previous_reading_revision": _timing_content_digest(previous_reading),
            "current_reading_revision": _timing_content_digest(current_reading),
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        row_revision = _timing_content_digest(row_payload)
        request_payload: dict[str, object] = {
            "request_kind": "pair-tail",
            "conflict_entry": entry_number,
            "previous_entry_index": index - 1,
            "current_entry_index": index,
            "previous_reading": previous_reading,
            "current_reading": current_reading,
            "transcript_revision": transcript_revision,
            "row_revision": row_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        request_payload["request_revision"] = _timing_content_digest(request_payload)
        requests.append(request_payload)
    return tuple(sorted(
        requests,
        key=lambda item: (int(item["conflict_entry"]), str(item["request_revision"])),
    ))


def _hubert_neutral_probe_window_start(
    earliest_hypothesis_centiseconds: int,
) -> float:
    """Give full-line CTC fixed leading context independent of any winner."""
    return max(
        0.0,
        earliest_hypothesis_centiseconds / 100.0
        - LOCAL_WHISPER_CONTEXT_MARGIN_SECONDS,
    )


def _hubert_probe_window_covers_hypotheses(
    window_start: float,
    window_end: float,
    hypothesis_centiseconds: tuple[int, ...],
    direct_tolerance_centiseconds: int,
) -> bool:
    """Require the entire direct-onset neighborhood of each hypothesis."""
    if not hypothesis_centiseconds or direct_tolerance_centiseconds < 0:
        return False
    if not all(math.isfinite(value) for value in (window_start, window_end)):
        return False
    margin = direct_tolerance_centiseconds / 100.0
    return all(
        window_start <= centiseconds / 100.0 - margin + 1e-9
        and window_end >= centiseconds / 100.0 + margin - 1e-9
        for centiseconds in hypothesis_centiseconds
    )


def _hubert_probe_window_covers_request_hypotheses(
    request: dict[str, object],
    direct_tolerance_centiseconds: int,
) -> bool:
    """Fail closed on malformed/rebound requests at evidence consumption."""
    try:
        start = float(request["window_start"])
        end = float(request["window_end"])
        bindings = request["hypothesis_bindings"]
        if not isinstance(bindings, list) or not bindings:
            return False
        times = tuple(
            item[2] for item in bindings
            if isinstance(item, (list, tuple))
            and len(item) == 3
            and type(item[2]) is int
        )
        if len(times) != len(bindings):
            return False
        selected = request["final_selected_centiseconds"]
        if not isinstance(selected, int) or selected not in times:
            return False
        if min(times) != request["hypothesis_min_centiseconds"] or max(times) != request["hypothesis_max_centiseconds"]:
            return False
        expected_start = _hubert_neutral_probe_window_start(min(times))
        if abs(start - expected_start) > 1e-6:
            return False
        return _hubert_probe_window_covers_hypotheses(start, end, times, direct_tolerance_centiseconds)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _postcentral_plan_known_lyric_hubert_current_probe_requests(
    conflict_entries: tuple[int, ...],
    report: dict[str, object],
    state: FinalTimingState,
    bundles: tuple[EvaluatedBackendBundle, ...],
    entries: list[LyricEntry],
    duration: float,
) -> tuple[dict[str, object], ...]:
    """Plan hypothesis-neutral full-line HuBERT onset probes.

    The probe window contains the immutable final selected timestamp *and every
    member of every already-existing distant conflict cluster for this row*.
    No hypothesis is privileged by the crop itself.  The probe can corroborate
    the immutable final selection or fail closed; it never proposes/selects a
    timestamp and cannot admit a candidate.
    """
    if (
        len(state.decisions) != len(entries)
        or not math.isfinite(duration)
        or duration <= 0.0
        or not bundles
    ):
        return ()
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    if not isinstance(seed, dict):
        return ()
    rebound = _postcentral_rebind_existing_peer_conflicts(seed, state, bundles)
    details = rebound.get("cross_backend_material_conflicts")
    raw_rows = details.get("entries") if isinstance(details, dict) else None
    if not isinstance(raw_rows, list):
        return ()
    row_lookup = {
        int(row["entry"]): row
        for row in raw_rows
        if isinstance(row, dict) and isinstance(row.get("entry"), int)
    }
    candidate_lookup = _bundle_candidate_lookup(bundles)
    if candidate_lookup is None:
        return ()

    readings = tuple(
        _known_lyric_hiragana_reading(entry_sung_text(entry))
        for entry in entries
    )
    direct_tolerance_centiseconds = int(round(
        PREFIX_ACOUSTIC_ASSOCIATION_SECONDS * 100.0
    ))
    requests: list[dict[str, object]] = []
    for entry_number in conflict_entries:
        index = entry_number - 1
        if not 0 <= index < len(entries):
            continue
        reading = readings[index]
        row = row_lookup.get(entry_number)
        if not reading or not isinstance(row, dict):
            continue
        raw_clusters = row.get("clusters")
        if not isinstance(raw_clusters, list) or not raw_clusters:
            continue

        decision = state.decisions[index]
        hypotheses: dict[tuple[str, str], int] = {
            (
                decision.candidate.candidate_id,
                decision.candidate.generation_revision,
            ): decision.written_time.centiseconds
        }
        malformed = False
        for cluster in raw_clusters:
            if not isinstance(cluster, dict):
                malformed = True
                break
            ids = cluster.get("member_candidate_ids")
            revisions = cluster.get("member_generation_revisions")
            if (
                not isinstance(ids, list)
                or not isinstance(revisions, list)
                or not ids
                or len(ids) != len(revisions)
            ):
                malformed = True
                break
            for candidate_id, generation_revision in zip(ids, revisions):
                if not isinstance(candidate_id, str) or not isinstance(generation_revision, str):
                    malformed = True
                    break
                candidate = candidate_lookup.get((candidate_id, generation_revision))
                if candidate is None or candidate.entry_index != index:
                    malformed = True
                    break
                hypotheses[(candidate_id, generation_revision)] = (
                    candidate.written_time.centiseconds
                )
            if malformed:
                break
        if malformed or len(hypotheses) < 2:
            continue

        hypothesis_bindings = tuple(sorted(
            (
                candidate_id,
                generation_revision,
                centiseconds,
            )
            for (candidate_id, generation_revision), centiseconds in hypotheses.items()
        ))
        hypothesis_centiseconds = tuple(
            sorted({item[2] for item in hypothesis_bindings})
        )
        hypothesis_set_revision = _timing_content_digest({
            "mode": "existing-peer-hypothesis-set-v1",
            "entry_index": index,
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "hypothesis_bindings": [
                [candidate_id, generation_revision, centiseconds]
                for candidate_id, generation_revision, centiseconds
                in hypothesis_bindings
            ],
        })
        earliest_seconds = hypothesis_centiseconds[0] / 100.0
        latest_seconds = hypothesis_centiseconds[-1] / 100.0
        next_start = (
            state.decisions[index + 1].written_time.seconds
            if index + 1 < len(entries)
            else duration
        )
        # A current-row hypothesis already beyond the next immutable row cannot
        # be tested safely with a single current-line window without importing
        # next-row speech; leave that row unresolved.
        if (
            index + 1 < len(entries)
            and latest_seconds
            > next_start + PREFIX_ACOUSTIC_ASSOCIATION_SECONDS
        ):
            continue

        window_start = _hubert_neutral_probe_window_start(
            hypothesis_centiseconds[0]
        )
        window_end = min(
            duration,
            next_start + CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
            earliest_seconds + 12.0,
        )
        if not 0.5 <= window_end - window_start <= 14.0:
            continue

        if not _hubert_probe_window_covers_hypotheses(
            window_start, window_end, hypothesis_centiseconds,
            direct_tolerance_centiseconds,
        ):
            continue
        conflict_row_revision = _timing_content_digest(row)
        transcript_revision = _timing_content_digest({
            "current_text": entry_sung_text(entries[index]),
            "current_reading": reading,
            "phonetic_system": "pykakasi-hiragana-v1",
            "probe_mode": "hypothesis-neutral-full-line",
        })
        row_payload = {
            "current_entry_index": index,
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "current_reading_revision": _timing_content_digest(reading),
            "conflict_row_revision": conflict_row_revision,
            "hypothesis_bindings": hypothesis_bindings,
            "hypothesis_set_revision": hypothesis_set_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        row_revision = _timing_content_digest(row_payload)
        request_payload: dict[str, object] = {
            "request_kind": "current-neutral-onset-probe",
            "conflict_entry": entry_number,
            "current_entry_index": index,
            "current_reading": reading,
            "transcript_revision": transcript_revision,
            "row_revision": row_revision,
            "conflict_row_revision": conflict_row_revision,
            "hypothesis_bindings": [
                [candidate_id, generation_revision, centiseconds]
                for candidate_id, generation_revision, centiseconds
                in hypothesis_bindings
            ],
            "hypothesis_set_revision": hypothesis_set_revision,
            "hypothesis_min_centiseconds": hypothesis_centiseconds[0],
            "hypothesis_max_centiseconds": hypothesis_centiseconds[-1],
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        request_payload["request_revision"] = _timing_content_digest(request_payload)
        requests.append(request_payload)
    return tuple(sorted(
        requests,
        key=lambda item: (int(item["conflict_entry"]), str(item["request_revision"])),
    ))


def _postcentral_plan_known_lyric_hubert_timing_failure_diagnostic_requests(
    report: dict[str, object],
    state: FinalTimingState,
    bundles: tuple[EvaluatedBackendBundle, ...],
    entries: list[LyricEntry],
    duration: float,
    *,
    exclude_entries: frozenset[int] = frozenset(),
) -> tuple[dict[str, object], ...]:
    """Plan evidence-only HuBERT probes for known timing-failure rows.

    These requests are deliberately outside the post-Central review authority.
    They collect the acoustic information needed to research a future timestamp
    recovery mechanism without requiring another production replay.  No result
    from this planner is converted into IndependentCurrentOnsetEvidence.
    """
    if (
        len(state.decisions) != len(entries)
        or len(state.audits) != len(entries)
        or not math.isfinite(duration)
        or duration <= 0.0
        or not bundles
    ):
        return ()
    seed = report.get(_RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY)
    peer = seed.get("peer_simulation") if isinstance(seed, dict) else None
    raw_rows = peer.get("rows") if isinstance(peer, dict) else None
    if not isinstance(raw_rows, list):
        return ()
    row_lookup = {
        int(row["entry"]): row
        for row in raw_rows
        if isinstance(row, dict) and isinstance(row.get("entry"), int)
    }
    candidate_lookup = _bundle_candidate_lookup(bundles)
    if candidate_lookup is None:
        return ()
    readings = tuple(
        _known_lyric_hiragana_reading(entry_sung_text(entry))
        for entry in entries
    )
    requests: list[dict[str, object]] = []
    for audit in state.audits:
        entry_number = audit.entry_index + 1
        if entry_number in exclude_entries:
            continue
        failure_codes = tuple(
            code for code in audit.failures
            if _timing_failure_requires_review(code)
        )
        if not failure_codes:
            continue
        index = audit.entry_index
        reading = readings[index]
        row = row_lookup.get(entry_number)
        if not reading or not isinstance(row, dict):
            continue
        clusters = row.get("clusters")
        if not isinstance(clusters, list) or len(clusters) < 2:
            continue

        decision = state.decisions[index]
        hypotheses: dict[tuple[str, str], int] = {
            (
                decision.candidate.candidate_id,
                decision.candidate.generation_revision,
            ): decision.written_time.centiseconds
        }
        malformed = False
        for cluster in clusters:
            if not isinstance(cluster, dict):
                malformed = True
                break
            ids = cluster.get("member_candidate_ids")
            revisions = cluster.get("member_generation_revisions")
            if (
                not isinstance(ids, list)
                or not isinstance(revisions, list)
                or not ids
                or len(ids) != len(revisions)
            ):
                malformed = True
                break
            for candidate_id, generation_revision in zip(ids, revisions):
                if not isinstance(candidate_id, str) or not isinstance(generation_revision, str):
                    malformed = True
                    break
                candidate = candidate_lookup.get((candidate_id, generation_revision))
                if candidate is None or candidate.entry_index != index:
                    malformed = True
                    break
                hypotheses[(candidate_id, generation_revision)] = (
                    candidate.written_time.centiseconds
                )
            if malformed:
                break
        if malformed or len(hypotheses) < 2:
            continue

        hypothesis_bindings = tuple(sorted(
            (
                candidate_id,
                generation_revision,
                centiseconds,
            )
            for (candidate_id, generation_revision), centiseconds in hypotheses.items()
        ))
        hypothesis_centiseconds = tuple(
            sorted({item[2] for item in hypothesis_bindings})
        )
        earliest_seconds = hypothesis_centiseconds[0] / 100.0
        next_start = (
            state.decisions[index + 1].written_time.seconds
            if index + 1 < len(entries)
            else duration
        )
        window_start = _hubert_neutral_probe_window_start(
            hypothesis_centiseconds[0]
        )
        window_end = min(
            duration,
            next_start + CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
            earliest_seconds + 12.0,
        )
        if not 0.5 <= window_end - window_start <= 14.0:
            continue
        hypothesis_set_revision = _timing_content_digest({
            "mode": "timing-failure-diagnostic-hypothesis-set-v1",
            "entry_index": index,
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "hypothesis_bindings": [
                [candidate_id, generation_revision, centiseconds]
                for candidate_id, generation_revision, centiseconds
                in hypothesis_bindings
            ],
            "failure_codes": list(failure_codes),
        })
        transcript_revision = _timing_content_digest({
            "current_text": entry_sung_text(entries[index]),
            "current_reading": reading,
            "phonetic_system": "pykakasi-hiragana-v1",
            "probe_mode": "timing-failure-diagnostic-only",
        })
        row_revision = _timing_content_digest({
            "entry_index": index,
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "hypothesis_set_revision": hypothesis_set_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
            "failure_codes": list(failure_codes),
        })
        request_payload: dict[str, object] = {
            "request_kind": "current-neutral-onset-probe",
            "probe_purpose": "timing-failure-diagnostic-only",
            "authority": False,
            "conflict_entry": entry_number,
            "current_entry_index": index,
            "current_reading": reading,
            "transcript_revision": transcript_revision,
            "row_revision": row_revision,
            "hypothesis_bindings": [
                [candidate_id, generation_revision, centiseconds]
                for candidate_id, generation_revision, centiseconds
                in hypothesis_bindings
            ],
            "hypothesis_set_revision": hypothesis_set_revision,
            "hypothesis_min_centiseconds": hypothesis_centiseconds[0],
            "hypothesis_max_centiseconds": hypothesis_centiseconds[-1],
            "final_candidate_id": decision.candidate.candidate_id,
            "final_generation_revision": decision.candidate.generation_revision,
            "final_selected_centiseconds": decision.written_time.centiseconds,
            "failure_codes": list(failure_codes),
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        request_payload["request_revision"] = _timing_content_digest(request_payload)
        requests.append(request_payload)
    return tuple(sorted(
        requests,
        key=lambda item: (int(item["conflict_entry"]), str(item["request_revision"])),
    ))


def _known_lyric_hubert_char_diagnostic_payload(
    spans_payload: object,
) -> list[dict[str, object]]:
    """Return sanitized helper geometry for report-only failure analysis."""
    if not isinstance(spans_payload, list):
        return []
    output: list[dict[str, object]] = []
    for item in spans_payload:
        if not isinstance(item, dict):
            return []
        try:
            start = float(item["start"])
            end = float(item["end"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            return []
        char = str(item.get("char", ""))
        if (
            not char
            or not all(math.isfinite(value) for value in (start, end, score))
            or end < start
        ):
            return []
        output.append({
            "char": char,
            "start": round(start, 6),
            "end": round(end, 6),
            "score": round(score, 8),
        })
    return output


def _known_lyric_hubert_dummy_candidate(
    entry_index: int,
    spans_payload: object,
    request_revision: str,
) -> TimingCandidate | None:
    """Project HuBERT char spans into the frozen TimingCandidate semantics."""
    if not isinstance(spans_payload, list) or len(spans_payload) < 2:
        return None
    spans: list[TimingTokenSpan] = []
    for item in spans_payload:
        if not isinstance(item, dict):
            return None
        try:
            start = float(item["start"])
            end = float(item["end"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (start, end, score)) or end < start:
            return None
        token = str(item.get("char", ""))
        if not token:
            return None
        spans.append(TimingTokenSpan(start, end, score, token))
    generation_revision = _timing_content_digest({
        "producer": _KNOWN_LYRIC_HUBERT_PRODUCER,
        "request_revision": request_revision,
        "spans": [
            (span.token, round(span.start, 6), round(span.end, 6), round(span.score, 8))
            for span in spans
        ],
    })
    raw_time = spans[0].start
    return TimingCandidate(
        candidate_id=_timing_content_digest({"known_lyric_hubert": generation_revision}),
        entry_index=entry_index,
        source=_KNOWN_LYRIC_HUBERT_PRODUCER,
        raw_time=raw_time,
        written_time=canonicalize_lrc_time(raw_time),
        generation_revision=generation_revision,
        spans=tuple(spans),
        confidence=float(statistics.mean(item.score for item in spans)),
        identity_support="supported",
        sequence_support="supported",
        acoustic_support="supported",
        direct_onset_support="unavailable",
        independent_content_identity=True,
    )


def _known_lyric_hubert_dummy_tail(
    predecessor_index: int,
    spans_payload: object,
    request_revision: str,
    *,
    require_valid: bool = True,
) -> TailRegionEvidence | None:
    """Reuse the frozen reliable-tail geometry policy on HuBERT char spans."""
    dummy = _known_lyric_hubert_dummy_candidate(
        predecessor_index, spans_payload, request_revision
    )
    if dummy is None:
        return None
    tail = derive_reliable_tail(dummy)
    if require_valid and (
        tail.validity != "valid"
        or tail.reliable_tail_start is None
        or tail.reliable_tail_end is None
    ):
        return None
    return tail


def _known_lyric_hubert_opening_evidence(
    current_index: int,
    spans_payload: object,
    request_revision: str,
) -> tuple[TimingCandidate, CandidateRegionEvidence] | None:
    """Return one frozen-policy current-opening proof, or fail closed."""
    dummy = _known_lyric_hubert_dummy_candidate(
        current_index, spans_payload, request_revision
    )
    if dummy is None:
        return None
    regions = analyze_candidate_regions(dummy)
    if not (
        regions.onset_proof_end is not None
        and math.isfinite(regions.onset_proof_end)
        and regions.temporal_coherence == "coherent"
        and regions.fracture_region not in {"opening", "prefix", "unknown"}
    ):
        return None
    return dummy, regions


def _known_lyric_hubert_predecessor_occurrence_status(
    previous_first: float,
    previous_last: float,
    predecessor_centiseconds: int,
    tolerance_centiseconds: int,
) -> str:
    """Bind a pair-tail path to the immutable predecessor *onset*, not its envelope."""
    if not all(math.isfinite(value) for value in (previous_first, previous_last)):
        return "geometry-unavailable"
    first_centiseconds = canonicalize_lrc_time(previous_first).centiseconds
    last_centiseconds = canonicalize_lrc_time(previous_last).centiseconds
    if abs(first_centiseconds - predecessor_centiseconds) > tolerance_centiseconds:
        return "first-token-mismatch"
    if last_centiseconds < predecessor_centiseconds - tolerance_centiseconds:
        return "envelope-mismatch"
    return "valid"


def _known_lyric_hubert_pair_tail_current_evidence_scope(
    predecessor_occurrence_status: str,
) -> str:
    """Select the narrow current-opening evidence allowed for pair-tail binding.

    A predecessor that already matches the joint pair occurrence keeps the V14
    strict final-associated onset gate.  Only a first-token mismatch may use the
    broader hypothesis-neutral boundary observation, because that is the causal
    family this recovery mechanism is intended to audit.
    """
    if predecessor_occurrence_status == "valid":
        return "strict-final-associated-onset"
    if predecessor_occurrence_status == "first-token-mismatch":
        return "hypothesis-neutral-boundary"
    return "reject"


def _known_lyric_hubert_pair_tail_occurrence_status(
    entries: list[LyricEntry],
    previous_index: int,
    current_index: int,
    previous_first: float,
    previous_last: float,
    pair_current_first: float,
    neutral_current_onset: float,
    tolerance_centiseconds: int,
) -> str:
    """Bind pair-tail geometry without depending on the old predecessor onset.

    The adjacent reading pair must be unique in the song, and the pair's current
    boundary must agree with the independently acquired hypothesis-neutral current
    opening.  That combination identifies the occurrence while leaving the
    predecessor selected timestamp free to be wrong.  Timestamp ownership does
    not move to HuBERT: this contract only authorizes predecessor-tail evidence.
    """
    if (
        not 0 <= previous_index < len(entries)
        or current_index != previous_index + 1
        or current_index >= len(entries)
    ):
        return "pair-index-mismatch"
    readings = tuple(
        _known_lyric_hiragana_reading(entry_sung_text(entry))
        for entry in entries
    )
    previous_reading = readings[previous_index]
    current_reading = readings[current_index]
    if not previous_reading or not current_reading:
        return "pair-reading-unavailable"
    pair_count = sum(
        1
        for index in range(1, len(readings))
        if readings[index - 1] == previous_reading
        and readings[index] == current_reading
    )
    if pair_count != 1:
        return "pair-reading-not-unique"
    if not all(math.isfinite(value) for value in (
        previous_first,
        previous_last,
        pair_current_first,
        neutral_current_onset,
    )):
        return "geometry-unavailable"
    if previous_last < previous_first:
        return "predecessor-geometry-invalid"
    pair_current_centiseconds = canonicalize_lrc_time(
        pair_current_first
    ).centiseconds
    neutral_current_centiseconds = canonicalize_lrc_time(
        neutral_current_onset
    ).centiseconds
    if (
        abs(pair_current_centiseconds - neutral_current_centiseconds)
        > tolerance_centiseconds
    ):
        return "current-boundary-mismatch"
    return "valid"


def _postcentral_plan_known_lyric_hubert_predecessor_only_requests(
    boundary_by_entry: Mapping[int, IndependentCurrentOnsetEvidence],
    state: FinalTimingState,
    entries: list[LyricEntry],
    duration: float,
) -> tuple[dict[str, object], ...]:
    """Plan predecessor-only tail acquisition from neutral current boundaries.

    This planner intentionally does not depend on joint-pair helper success or
    predecessor occurrence status.  Occurrence ownership comes from the already
    accepted hypothesis-neutral current boundary plus a song-global unique
    predecessor reading.  The resulting request can only feed the existing tail
    evidence owner.
    """
    if (
        len(state.decisions) != len(entries)
        or not math.isfinite(duration)
        or duration <= 0.0
    ):
        return ()
    readings = tuple(
        _known_lyric_hiragana_reading(entry_sung_text(entry))
        for entry in entries
    )
    reading_counts = collections.Counter(reading for reading in readings if reading)
    requests: list[dict[str, object]] = []
    for current_index in sorted(boundary_by_entry):
        current_boundary = boundary_by_entry[current_index]
        previous_index = current_index - 1
        if not (
            0 <= previous_index < len(entries)
            and 0 <= current_index < len(entries)
            and current_boundary.entry_index == current_index
            and _known_lyric_hubert_current_onset_is_valid(current_boundary)
        ):
            continue
        previous_reading = readings[previous_index]
        if not previous_reading or reading_counts[previous_reading] != 1:
            continue
        previous_start = state.decisions[previous_index].written_time.seconds
        current_start = state.decisions[current_index].written_time.seconds
        before_previous = (
            state.decisions[previous_index - 1].written_time.seconds
            if previous_index >= 1 else 0.0
        )
        spacing = max(HOP_SECONDS, current_start - previous_start)
        lower = before_previous + HOP_SECONDS if previous_index >= 1 else 0.0
        window_start = max(
            0.0,
            lower,
            previous_start - min(2.5, max(1.25, spacing * 0.5)),
        )
        window_end = min(duration, float(current_boundary.onset_time))
        if not 0.5 <= window_end - window_start <= 24.0:
            continue
        previous_text = entry_sung_text(entries[previous_index])
        transcript_revision = _timing_content_digest({
            "previous_text": previous_text,
            "previous_reading": previous_reading,
            "phonetic_system": "pykakasi-hiragana-v1",
            "probe_mode": "predecessor-only-neutral-current-bound-v1",
        })
        row_payload = {
            "previous_entry_index": previous_index,
            "current_entry_index": current_index,
            "previous_text_revision": _timing_content_digest(previous_text),
            "previous_reading_revision": _timing_content_digest(previous_reading),
            "current_boundary_observation_revision": current_boundary.observation_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        row_revision = _timing_content_digest(row_payload)
        request_payload: dict[str, object] = {
            # Reuse the helper's single-line forced-alignment primitive.  The
            # semantic role is bound below by transcript/row provenance and is
            # never mixed with the first-pass current-onset result lookup.
            "request_kind": "current-neutral-onset-probe",
            "evidence_role": "predecessor-only-tail",
            "conflict_entry": current_index + 1,
            "previous_entry_index": previous_index,
            "current_entry_index": current_index,
            "current_reading": previous_reading,
            "transcript_revision": transcript_revision,
            "row_revision": row_revision,
            "current_boundary_observation_revision": current_boundary.observation_revision,
            "window_start": round(window_start, 6),
            "window_end": round(window_end, 6),
        }
        request_payload["request_revision"] = _timing_content_digest(request_payload)
        requests.append(request_payload)
    return tuple(requests)


def _known_lyric_hubert_full_line_bridge_is_coherent(
    candidate: TimingCandidate,
    regions: CandidateRegionEvidence,
    tail: TailRegionEvidence,
) -> bool:
    """Require one occurrence-bound opening-to-terminal geometric bridge.

    V10 incorrectly equated a locally weak CTC character posterior (<0.05) with
    a whole-line temporal fracture.  The frozen region policy does not make that
    equivalence: token support breaks `_supported_run_metrics`, while a temporal
    fracture is represented independently by a >=3 s severe gap.

    V11 therefore keeps the already-frozen opening proof and terminal-tail proof
    as the two acoustic endpoints and requires that both belong to the exact same
    candidate/revision with no severe geometric fracture between them.  No score,
    gap, or materiality threshold is added or relaxed.
    """
    return bool(
        candidate.spans
        and regions.candidate_id == candidate.candidate_id
        and regions.generation_revision == candidate.generation_revision
        and regions.temporal_coherence == "coherent"
        and regions.fracture_region not in {"opening", "prefix", "unknown"}
        and not regions.severe_gap_indexes
        and tail.validity == "valid"
        and tail.previous_candidate_id == candidate.candidate_id
        and tail.previous_generation_revision == candidate.generation_revision
        and tail.previous_region_revision == regions.region_revision
        and tail.reliable_tail_start is not None
        and tail.reliable_tail_end is not None
    )


def _make_known_lyric_hubert_current_onset(
    *,
    entry_index: int,
    onset_time: float,
    opening_proof_end: float,
    opening_region_revision: str,
    identity_owner_candidate_id: str,
    identity_owner_generation_revision: str,
    model_identity: str,
    helper_revision: str,
    capability_revision: str,
    audio_revision: str,
    transcript_revision: str,
    row_revision: str,
    request_revision: str,
    hypothesis_set_revision: str,
) -> IndependentCurrentOnsetEvidence | None:
    occurrence_binding_revision = _timing_content_digest({
        "mode": "existing-conflict-hypothesis-neutral-window",
        "request_revision": request_revision,
        "transcript_revision": transcript_revision,
        "row_revision": row_revision,
        "hypothesis_set_revision": hypothesis_set_revision,
        "identity_owner_candidate_id": identity_owner_candidate_id,
        "identity_owner_generation_revision": identity_owner_generation_revision,
    })
    payload = _known_lyric_hubert_current_onset_payload(
        entry_index=entry_index,
        onset_time=onset_time,
        opening_proof_end=opening_proof_end,
        opening_region_revision=opening_region_revision,
        identity_owner_candidate_id=identity_owner_candidate_id,
        identity_owner_generation_revision=identity_owner_generation_revision,
        producer=_KNOWN_LYRIC_HUBERT_ONSET_PRODUCER,
        backend_lineage=_KNOWN_LYRIC_HUBERT_LINEAGE,
        model_identity=model_identity,
        helper_revision=helper_revision,
        capability_revision=capability_revision,
        audio_revision=audio_revision,
        transcript_revision=transcript_revision,
        row_revision=row_revision,
        request_revision=request_revision,
        hypothesis_set_revision=hypothesis_set_revision,
        occurrence_binding_revision=occurrence_binding_revision,
    )
    string_fields = (
        payload["opening_region_revision"],
        payload["identity_owner_candidate_id"],
        payload["identity_owner_generation_revision"],
        payload["producer"],
        payload["backend_lineage"],
        payload["model_identity"],
        payload["helper_revision"],
        payload["capability_revision"],
        payload["audio_revision"],
        payload["transcript_revision"],
        payload["row_revision"],
        payload["request_revision"],
        payload["hypothesis_set_revision"],
        payload["occurrence_binding_revision"],
    )
    if (
        entry_index < 0
        or not math.isfinite(onset_time)
        or onset_time < 0.0
        or not math.isfinite(opening_proof_end)
        or opening_proof_end < onset_time
        or any(not isinstance(value, str) or not value.strip() for value in string_fields)
    ):
        return None
    observation_revision = _timing_content_digest(payload)
    return IndependentCurrentOnsetEvidence(
        entry_index=entry_index,
        onset_time=float(onset_time),
        opening_proof_end=float(opening_proof_end),
        opening_region_revision=opening_region_revision,
        identity_owner_candidate_id=identity_owner_candidate_id,
        identity_owner_generation_revision=identity_owner_generation_revision,
        producer=_KNOWN_LYRIC_HUBERT_ONSET_PRODUCER,
        backend_lineage=_KNOWN_LYRIC_HUBERT_LINEAGE,
        model_identity=model_identity,
        helper_revision=helper_revision,
        capability_revision=capability_revision,
        audio_revision=audio_revision,
        transcript_revision=transcript_revision,
        row_revision=row_revision,
        request_revision=request_revision,
        hypothesis_set_revision=hypothesis_set_revision,
        occurrence_binding_revision=occurrence_binding_revision,
        observation_revision=observation_revision,
    )
