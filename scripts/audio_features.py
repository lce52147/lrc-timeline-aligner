"""Extracted from auto_lrc.py (split step 2): audio decode and features, vocal onsets, demucs, recurrence acoustics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import numpy as np
import runtime_paths
from runtime_base import LrcError, _path_sha256, _timing_content_digest, default_ctc_python, demucs_cache_key_lock, gpu_compute_phase, run_command
import lrc_stage_timings


SAMPLE_RATE = 16_000


HOP_SECONDS = 0.02


FRAME_SECONDS = 0.06


DEFAULT_VOCAL_CACHE_DIR = runtime_paths.vocal_cache_dir()


VOCAL_SEPARATOR_PROTOCOL_REVISION = "demucs-two-stems-vocals-mp3-v3-shifts0"


@dataclass
class AudioFeatures:
    duration: float
    frame_times: np.ndarray
    rms_db: np.ndarray
    onset_strength: np.ndarray
    segments: list[tuple[float, float]]


_AUDIO_FEATURE_CACHE_MAX_ENTRIES = 24


_audio_feature_cache: dict[tuple[object, ...], object] = {}


def _audio_feature_cache_key(
    kind: str,
    audio_path: Path,
    duration: float | None = None,
) -> tuple[object, ...] | None:
    """Return an exact-content key for deterministic heavy audio analysis.

    The key carries the file's content digest, so reuse is possible only while
    the same input bytes (and, where relevant, exact duration) are presented
    again; ``_path_sha256`` re-hashes whenever the file's stat identity changes.
    If content identity cannot be established, fail open to the legacy uncached
    analysis path rather than changing call semantics.
    """
    try:
        resolved = audio_path.resolve()
        digest = _path_sha256(resolved)
    except (OSError, LrcError):
        return None
    return (kind, str(resolved), digest, duration)


def _audio_feature_cache_store(key: tuple[object, ...], value: object) -> None:
    if key in _audio_feature_cache:
        _audio_feature_cache[key] = value
        return
    if len(_audio_feature_cache) >= _AUDIO_FEATURE_CACHE_MAX_ENTRIES:
        _audio_feature_cache.pop(next(iter(_audio_feature_cache)))
    _audio_feature_cache[key] = value


def _clone_audio_features(features: AudioFeatures) -> AudioFeatures:
    # Keep cache storage isolated from downstream callers even though current
    # production consumers are read-only.  This preserves call-level semantics
    # if a future consumer mutates an array or the segments list in-place.
    return AudioFeatures(
        duration=features.duration,
        frame_times=features.frame_times.copy(),
        rms_db=features.rms_db.copy(),
        onset_strength=features.onset_strength.copy(),
        segments=list(features.segments),
    )


def probe_duration(audio_path: Path) -> float:
    proc = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(audio_path),
        ]
    )
    if proc.returncode != 0:
        raise LrcError(f"ffprobe failed for {audio_path}: {proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout)
        duration = float(payload["format"]["duration"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise LrcError(f"Could not read duration from ffprobe output for {audio_path}") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise LrcError(f"Invalid audio duration for {audio_path}: {duration}")
    return duration


def decode_audio(audio_path: Path) -> np.ndarray:
    proc = subprocess.run(
        [
            runtime_paths.ffmpeg(),
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-f",
            "s16le",
            "-",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise LrcError(f"ffmpeg decode failed for {audio_path}: {stderr}")
    if not proc.stdout:
        raise LrcError(f"ffmpeg decoded no audio for {audio_path}")
    audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size < SAMPLE_RATE:
        raise LrcError(f"Audio is too short to analyze: {audio_path}")
    return audio


def frame_audio(audio: np.ndarray) -> np.ndarray:
    frame_length = max(128, int(round(FRAME_SECONDS * SAMPLE_RATE)))
    hop = max(64, int(round(HOP_SECONDS * SAMPLE_RATE)))
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    frame_count = 1 + (audio.size - frame_length) // hop
    shape = (frame_count, frame_length)
    strides = (audio.strides[0] * hop, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)


def smooth(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size == 0:
        return values
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(values, kernel, mode="same")


def normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def merge_segments(
    active: np.ndarray,
    frame_times: np.ndarray,
    duration: float,
    min_gap: float = 0.45,
    min_length: float = 0.70,
) -> list[tuple[float, float]]:
    raw: list[tuple[float, float]] = []
    start_idx: int | None = None
    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            raw.append((float(frame_times[start_idx]), float(frame_times[idx - 1] + HOP_SECONDS)))
            start_idx = None
    if start_idx is not None:
        raw.append((float(frame_times[start_idx]), duration))

    if not raw:
        return []

    merged: list[tuple[float, float]] = [raw[0]]
    for start, end in raw[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= min_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return [(s, e) for s, e in merged if e - s >= min_length]


def analyze_audio(audio_path: Path, duration: float) -> AudioFeatures:
    cache_key = _audio_feature_cache_key("analyze-audio-v1", audio_path, duration)
    cached = _audio_feature_cache.get(cache_key) if cache_key is not None else None
    if isinstance(cached, AudioFeatures):
        return _clone_audio_features(cached)

    audio = decode_audio(audio_path)
    frames = frame_audio(audio)
    hop = max(64, int(round(HOP_SECONDS * SAMPLE_RATE)))
    frame_times = np.arange(frames.shape[0], dtype=np.float32) * (hop / SAMPLE_RATE)

    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-9)
    rms_norm = normalize(smooth(rms_db, 5))

    window = np.hanning(frames.shape[1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    spectrum = spectrum / (np.sum(spectrum, axis=1, keepdims=True) + 1e-9)
    flux = np.zeros(spectrum.shape[0], dtype=np.float32)
    flux[1:] = np.maximum(spectrum[1:] - spectrum[:-1], 0.0).sum(axis=1)
    flux = smooth(normalize(flux), 3)

    energy_rise = np.zeros_like(rms_norm)
    energy_rise[1:] = np.maximum(rms_norm[1:] - rms_norm[:-1], 0.0)
    onset_strength = normalize((0.70 * flux) + (0.30 * normalize(energy_rise)))

    threshold = max(0.18, float(np.percentile(rms_norm, 55)))
    active = smooth((rms_norm > threshold).astype(np.float32), 9) > 0.28
    segments = merge_segments(active, frame_times, duration)

    result = AudioFeatures(
        duration=duration,
        frame_times=frame_times,
        rms_db=rms_db,
        onset_strength=onset_strength,
        segments=segments,
    )
    if cache_key is not None:
        _audio_feature_cache_store(cache_key, _clone_audio_features(result))
    return result


def analyze_vocal_onsets(audio_path: Path, duration: float) -> AudioFeatures:
    """Measure consonant-sensitive onsets from an isolated vocal stem."""
    cache_key = _audio_feature_cache_key("analyze-vocal-onsets-v1", audio_path, duration)
    cached = _audio_feature_cache.get(cache_key) if cache_key is not None else None
    if isinstance(cached, AudioFeatures):
        return _clone_audio_features(cached)

    audio = decode_audio(audio_path)
    frames = frame_audio(audio)
    hop = max(64, int(round(HOP_SECONDS * SAMPLE_RATE)))
    frame_times = np.arange(frames.shape[0], dtype=np.float32) * (hop / SAMPLE_RATE)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-9)

    window = np.hanning(frames.shape[1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32) + 1e-9
    frequencies = np.fft.rfftfreq(frames.shape[1], 1.0 / SAMPLE_RATE)
    consonant_band = (frequencies >= 1200.0) & (frequencies <= 6500.0)
    log_band = np.log(spectrum[:, consonant_band])
    flux = np.zeros(frames.shape[0], dtype=np.float32)
    flux[1:] = np.maximum(log_band[1:] - log_band[:-1], 0.0).mean(axis=1)
    energy_rise = np.zeros_like(rms)
    energy_rise[1:] = np.maximum(rms[1:] - rms[:-1], 0.0)
    onset_strength = normalize((0.75 * normalize(flux)) + (0.25 * normalize(energy_rise)))
    result = AudioFeatures(duration, frame_times, rms_db, onset_strength, [])
    if cache_key is not None:
        _audio_feature_cache_store(cache_key, _clone_audio_features(result))
    return result


RECURRENCE_MIN_TEMPLATE_SECONDS = 3.0


RECURRENCE_MAX_TEMPLATE_SECONDS = 18.0


def _mel_filterbank(n_fft: int, n_mels: int = 40) -> np.ndarray:
    def hz_to_mel(value: float) -> float:
        return 2595.0 * math.log10(1.0 + value / 700.0)
    def mel_to_hz(value: float) -> float:
        return 700.0 * (10.0 ** (value / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(80.0), hz_to_mel(7600.0), n_mels + 2)
    hz_points = np.array([mel_to_hz(float(value)) for value in mel_points])
    bins = np.floor((n_fft + 1) * hz_points / SAMPLE_RATE).astype(int)
    max_bin = n_fft // 2
    bank = np.zeros((n_mels, max_bin + 1), dtype=np.float32)
    for index in range(n_mels):
        left = max(0, int(bins[index]))
        center = max(left + 1, int(bins[index + 1]))
        right = max(center + 1, int(bins[index + 2]))
        center = min(center, max_bin)
        right = min(right, max_bin + 1)
        if center > left:
            bank[index, left:center] = (
                np.arange(left, center, dtype=np.float32) - float(left)
            ) / float(center - left)
        if right > center:
            bank[index, center:right] = (
                float(right) - np.arange(center, right, dtype=np.float32)
            ) / float(right - center)
    return bank


def acoustic_recurrence_features(audio_path: Path) -> np.ndarray:
    """Return deterministic MFCC-like full-song features using only NumPy."""
    cache_key = _audio_feature_cache_key("acoustic-recurrence-features-v1", audio_path)
    cached = _audio_feature_cache.get(cache_key) if cache_key is not None else None
    if isinstance(cached, np.ndarray):
        return cached.copy()

    audio = decode_audio(audio_path)
    n_fft = 1024
    hop = int(round(HOP_SECONDS * SAMPLE_RATE))
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))
    frame_count = 1 + (audio.size - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        np.ascontiguousarray(audio),
        shape=(frame_count, n_fft),
        strides=(audio.strides[0] * hop, audio.strides[0]),
    )
    window = np.hanning(n_fft).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32) ** 2
    log_mel = np.log(np.maximum(power @ _mel_filterbank(n_fft).T, 1e-8)).T
    mel_count = log_mel.shape[0]
    coeff_count = 20
    mel_index = np.arange(mel_count, dtype=np.float32) + 0.5
    coeff_index = np.arange(coeff_count, dtype=np.float32)[:, None]
    dct = np.cos((math.pi / mel_count) * coeff_index * mel_index[None, :]).astype(np.float32)
    cepstral = (dct @ log_mel)[1:]
    delta = np.zeros_like(cepstral)
    if cepstral.shape[1] >= 3:
        delta[:, 1:-1] = 0.5 * (cepstral[:, 2:] - cepstral[:, :-2])
    result = np.vstack([cepstral, delta]).astype(np.float32)
    if cache_key is not None:
        _audio_feature_cache_store(cache_key, result.copy())
    return result


def _recurrence_segment(features: np.ndarray, start: float, end: float) -> np.ndarray:
    start_frame = max(0, int(round(start / HOP_SECONDS)))
    end_frame = min(features.shape[1], int(round(end / HOP_SECONDS)))
    return features[:, start_frame:end_frame]


def acoustic_recurrence_similarity(left: np.ndarray, right: np.ndarray) -> float:
    frame_count = min(left.shape[1], right.shape[1])
    if frame_count < 8:
        return -1.0
    left = left[:, :frame_count].astype(np.float64, copy=True)
    right = right[:, :frame_count].astype(np.float64, copy=True)
    left = (left - left.mean(axis=1, keepdims=True)) / (left.std(axis=1, keepdims=True) + 1e-6)
    right = (right - right.mean(axis=1, keepdims=True)) / (right.std(axis=1, keepdims=True) + 1e-6)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return -1.0
    return float(np.sum(left * right) / denominator)


def _search_recurrence_start(
    features: np.ndarray,
    template_start: float,
    template_end: float,
    search_start: float,
    search_end: float,
) -> dict[str, float] | None:
    duration = template_end - template_start
    if not (RECURRENCE_MIN_TEMPLATE_SECONDS <= duration <= RECURRENCE_MAX_TEMPLATE_SECONDS):
        return None
    if search_end - search_start < duration + 0.20:
        return None
    template = _recurrence_segment(features, template_start, template_end)
    if template.shape[1] < 8:
        return None
    coarse: list[tuple[float, float]] = []
    step = 0.10
    start = search_start
    latest = search_end - duration
    while start <= latest + 1e-9:
        segment = _recurrence_segment(features, start, start + duration)
        coarse.append((acoustic_recurrence_similarity(template, segment), start))
        start += step
    if not coarse:
        return None
    coarse.sort(reverse=True)
    coarse_best_score, coarse_best_start = coarse[0]
    refine_start = max(search_start, coarse_best_start - 0.16)
    refine_end = min(latest, coarse_best_start + 0.16)
    refined: list[tuple[float, float]] = []
    start = refine_start
    while start <= refine_end + 1e-9:
        segment = _recurrence_segment(features, start, start + duration)
        refined.append((acoustic_recurrence_similarity(template, segment), start))
        start += HOP_SECONDS
    refined.sort(reverse=True)
    best_score, best_start = refined[0] if refined else (coarse_best_score, coarse_best_start)
    separation = max(2.0, duration * 0.30)
    distinct = next(
        ((score, candidate_start) for score, candidate_start in coarse if abs(candidate_start - best_start) >= separation),
        None,
    )
    runner_score = float(distinct[0]) if distinct is not None else -1.0
    return {
        "start": round(float(best_start), 3),
        "score": round(float(best_score), 8),
        "runner_score": round(runner_score, 8),
        "margin": round(float(best_score - runner_score), 8),
        "template_duration": round(duration, 3),
    }


def acoustic_pitchclass_recurrence_features(audio_path: Path) -> np.ndarray:
    """Return a dependency-free pitch-class recurrence representation.

    This representation is intentionally independent from the MMS posterior
    path.  Short-time spectral energy is accumulated by pitch class, then
    temporally smoothed to suppress transient/percussive dominance.  A delta
    stream preserves phrase-shape changes.  It is used only as recurrence
    evidence; it never writes a timestamp directly.
    """
    audio = decode_audio(audio_path)
    n_fft = 2048
    hop = int(round(HOP_SECONDS * SAMPLE_RATE))
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))
    frame_count = 1 + (audio.size - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        np.ascontiguousarray(audio),
        shape=(frame_count, n_fft),
        strides=(audio.strides[0] * hop, audio.strides[0]),
    )
    window = np.hanning(n_fft).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32) ** 2
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    valid = (frequencies >= 80.0) & (frequencies <= 5000.0)
    valid_frequencies = frequencies[valid]
    midi = 69.0 + 12.0 * np.log2(np.maximum(valid_frequencies, 1e-6) / 440.0)
    pitch_classes = np.mod(np.rint(midi).astype(np.int16), 12)
    compressed = np.log1p(power[:, valid] * 1000.0)
    chroma = np.zeros((12, frame_count), dtype=np.float32)
    for pitch_class in range(12):
        bins = pitch_classes == pitch_class
        if np.any(bins):
            chroma[pitch_class] = compressed[:, bins].sum(axis=1)

    # ~300 ms temporal mean: long enough to attenuate drum transients while
    # keeping sung phrase contours.  Implemented with a cumulative sum so the
    # production path remains NumPy-only.
    smoothing_frames = 15
    pad = smoothing_frames // 2
    padded = np.pad(chroma, ((0, 0), (pad, pad)), mode="edge")
    cumulative = np.cumsum(
        np.pad(padded, ((0, 0), (1, 0)), mode="constant"), axis=1, dtype=np.float64
    )
    smoothed = (
        (cumulative[:, smoothing_frames:] - cumulative[:, :-smoothing_frames])
        / float(smoothing_frames)
    ).astype(np.float32)
    delta = np.zeros_like(smoothed)
    if smoothed.shape[1] >= 3:
        delta[:, 1:-1] = 0.5 * (smoothed[:, 2:] - smoothed[:, :-2])
    return np.vstack([smoothed, delta]).astype(np.float32)


def _stage_demucs_input(audio_path: Path, staging_dir: Path, tag: str) -> Path:
    # Deliberately preserve the original bytes and vocal_cache_identity.
    suffix = audio_path.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        raise OSError(f"Unsupported Demucs input extension: {suffix!r}")
    safe_tag = re.sub(r"[^A-Za-z0-9_-]", "_", tag)[:40] or "source"
    staged = staging_dir / f"audio_{safe_tag}{suffix}"
    shutil.copyfile(audio_path, staged)
    return staged


def vocal_onset_features(
    audio_path: Path, duration: float, args: argparse.Namespace
) -> tuple[AudioFeatures | None, str | None]:
    if not args.vocal_onset_refine:
        return None, "disabled"
    python_exe = default_ctc_python()
    if not python_exe.exists():
        return None, "missing-asr-python"
    with tempfile.TemporaryDirectory(prefix="lrc-vocals-") as temp_name:
        temp_dir = Path(temp_name)
        try:
            safe_audio = _stage_demucs_input(audio_path, temp_dir, "onset")
        except OSError as exc:
            return None, f"demucs-input-stage-failed: {exc}"
        command = [
            str(python_exe),
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "--mp3",
            "-d",
            "cuda",
            "-o",
            str(temp_dir),
            str(safe_audio),
        ]
        with gpu_compute_phase():
            proc = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().replace("\n", " ")
            return None, f"demucs-failed: {detail[-1000:]}"
        stems = list(temp_dir.rglob("vocals.mp3"))
        if not stems:
            return None, "demucs-did-not-write-vocals"
        return analyze_vocal_onsets(stems[0], duration), None


@lru_cache(maxsize=4)
def _demucs_separator_identity(python_exe_text: str) -> tuple[str, str]:
    """Return the runtime Demucs version and deterministic separator revision."""
    demucs_version = "unknown"
    try:
        proc = run_command(
            [
                python_exe_text,
                "-X",
                "utf8",
                "-c",
                (
                    "import importlib.metadata as m; "
                    "print(m.version('demucs'))"
                ),
            ],
            timeout_seconds=15.0,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            demucs_version = proc.stdout.strip().splitlines()[-1].strip()
    except (LrcError, OSError, subprocess.SubprocessError):
        pass
    payload = {
        "protocol_revision": VOCAL_SEPARATOR_PROTOCOL_REVISION,
        "module": "demucs",
        "demucs_version": demucs_version,
        "two_stems": "vocals",
        "format": "mp3",
        "device": "cuda",
    }
    return demucs_version, _timing_content_digest(payload)


def vocal_cache_identity(audio_path: Path, python_exe: Path) -> dict[str, object]:
    """Content-address the vocal stem; filesystem timestamps are diagnostic only."""
    stat = audio_path.stat()
    source_sha256 = _path_sha256(audio_path)
    demucs_version, separator_revision = _demucs_separator_identity(str(python_exe))
    key_payload = {
        "source_audio_sha256": source_sha256,
        "separator_revision": separator_revision,
    }
    return {
        "source_audio_sha256": source_sha256,
        "source_audio_size": int(stat.st_size),
        "source_audio_mtime_ns": int(stat.st_mtime_ns),
        "vocal_separator_protocol_revision": VOCAL_SEPARATOR_PROTOCOL_REVISION,
        "vocal_separator_demucs_version": demucs_version,
        "vocal_separator_revision": separator_revision,
        "vocal_cache_key": hashlib.sha256(
            json.dumps(
                key_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16],
    }


def _demucs_vocal_stem_valid(path: Path, source_audio: Path) -> bool:
    """Reject truncated/corrupt audio even when a cache file is non-empty."""
    try:
        if path.stat().st_size < 4096:
            return False
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries",
                "stream=codec_name,sample_rate,channels:format=duration",
                "-of", "json", str(path),
            ],
            check=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=12,
        )
        if proc.returncode != 0:
            return False
        probe = json.loads(proc.stdout)
        streams = probe.get("streams") or []
        if len(streams) != 1:
            return False
        stream = streams[0]
        if (
            str(stream.get("codec_name", "")).lower() != "mp3"
            or int(stream.get("sample_rate", 0)) < 8000
            or int(stream.get("channels", 0)) < 1
        ):
            return False
        duration = float(probe["format"]["duration"])
        if not math.isfinite(duration) or duration < 1:
            return False
        source_duration = probe_duration(source_audio)
        if abs(duration - source_duration) > max(2.5, source_duration * 0.03):
            return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            OverflowError, LrcError, subprocess.SubprocessError):
        return False


def _demucs_write_error(cache_dir: Path, detail: str) -> str:
    """Preserve full stderr under the song's own content-addressed cache."""
    log_file = cache_dir / "demucs_last_error.log"
    try:
        log_file.write_text(detail, encoding="utf-8")
        return str(log_file)
    except OSError:
        return "unavailable"


def prepare_vocal_ctc_audio(audio_path: Path, args: argparse.Namespace) -> tuple[Path, str, str | None]:
    """Build or reuse a validated content-addressed isolated vocal stem.

    The temporary output folder is unique per invocation and cannot select an
    earlier Demucs run's result from the shared cache tree.
    """
    if not getattr(args, "vocal_ctc", True):
        return audio_path, "mix", "disabled"
    python_exe = default_ctc_python()
    if not python_exe.exists():
        return audio_path, "mix", "missing-asr-python"
    try:
        identity = vocal_cache_identity(audio_path, python_exe)
    except OSError:
        return audio_path, "mix", "audio-stat-or-hash-failed"
    key = str(identity["vocal_cache_key"])
    cache_dir = DEFAULT_VOCAL_CACHE_DIR / key
    cached = cache_dir / "vocals.mp3"
    if cached.is_file() and _demucs_vocal_stem_valid(cached, audio_path):
        return cached, "vocal-stem", "cache-hit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with demucs_cache_key_lock(cache_dir):
            if cached.is_file() and _demucs_vocal_stem_valid(cached, audio_path):
                return cached, "vocal-stem", "cache-hit"
            demucs_env = os.environ.copy()
            demucs_env["PYTHONUTF8"] = "1"
            demucs_env["PYTHONIOENCODING"] = "utf-8"
            with tempfile.TemporaryDirectory(prefix="lrc-demucs-run-", dir=cache_dir) as temp_name:
                run_dir = Path(temp_name)
                staged_input = _stage_demucs_input(audio_path, run_dir, key)
                current_output = run_dir / "out"
                with gpu_compute_phase():
                    proc = subprocess.run(
                        [
                            str(python_exe), "-X", "utf8", "-m", "demucs",
                            "--two-stems", "vocals", "--mp3", "--shifts", "0",
                            "-d", "cuda", "-o", str(current_output), str(staged_input),
                        ],
                        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding="utf-8", errors="replace", env=demucs_env,
                    )
                if proc.returncode:
                    detail = (proc.stderr or proc.stdout or "").strip()
                    logfile = _demucs_write_error(cache_dir, detail)
                    tail = " ".join(detail[-1300:].splitlines())
                    return audio_path, "mix", f"demucs-failed: {tail} [full-log: {logfile}]"
                new_stems = list(current_output.rglob("vocals.mp3"))
                if len(new_stems) != 1:
                    return audio_path, "mix", f"demucs-invalid-output-count={len(new_stems)}"
                if not _demucs_vocal_stem_valid(new_stems[0], audio_path):
                    return audio_path, "mix", "demucs-invalid-generated-vocal-stem"
                # The output file and the cache are on the same volume, so replace
                # makes a single atomic publication instead of a partial cache.
                os.replace(new_stems[0], cached)
                (cache_dir / "demucs_last_error.log").unlink(missing_ok=True)
    except OSError as exc:
        return audio_path, "mix", f"demucs-input-or-cache-io-failed: {type(exc).__name__}: {exc}"
    return cached, "vocal-stem", "generated"


def nearest_onset_evidence(features: AudioFeatures, timestamp: float) -> tuple[float, float] | None:
    mask = (features.frame_times >= timestamp - 0.35) & (features.frame_times <= timestamp + 0.35)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    peak = max(indices, key=lambda index: float(features.onset_strength[index]))
    return float(features.frame_times[peak]), float(features.onset_strength[peak])


FINAL_SELF_CHECK_WINDOW_SECONDS = 0.40


FINAL_SELF_CHECK_BELOW_MEDIAN_DB = 20.0


@dataclass(frozen=True)
class VocalStemLevels:
    """Whether the singer is audible at a time, measured on the Demucs vocal stem.

    The one judgment behind both the final self-check's F1 flag and the
    local-window rule that a re-aligned line cannot start in vocal silence.
    """

    status: str
    threshold_db: float | None = None
    active_windows: int = 0
    error: str | None = None
    samples: np.ndarray | None = field(default=None, repr=False, compare=False)

    def window_db(self, seconds: float) -> float | None:
        if self.samples is None:
            return None
        window_samples = int(round(SAMPLE_RATE * FINAL_SELF_CHECK_WINDOW_SECONDS))
        start = int(round(seconds * SAMPLE_RATE))
        if start < 0 or start + window_samples > len(self.samples):
            return None
        x = self.samples[start:start + window_samples].astype(np.float64)
        return 10.0 * math.log10(max(float(np.mean(x * x)), 1e-20))

    def absent(self, seconds: float) -> bool:
        if self.status != "checked" or self.threshold_db is None:
            return False
        db = self.window_db(seconds)
        return db is not None and db < self.threshold_db


def vocal_stem_levels(report: dict[str, object]) -> VocalStemLevels:
    vocal_file = report.get("ctc_audio_path")
    if not (
        report.get("ctc_audio_source") == "vocal-stem"
        and isinstance(vocal_file, str)
        and Path(vocal_file).is_file()
    ):
        return VocalStemLevels("skipped-no-stem")
    try:
        stem = Path(vocal_file)
        samples = decode_audio(stem)
        probe = VocalStemLevels("probing", samples=samples)
        active = analyze_audio(stem, len(samples) / SAMPLE_RATE).segments
        levels = [
            value
            for seg_start, seg_end in active
            for t in np.arange(
                seg_start, seg_end - FINAL_SELF_CHECK_WINDOW_SECONDS + 1e-6, 0.2
            )
            for value in (probe.window_db(float(t)),)
            if value is not None
        ]
        if len(levels) < 8:
            return VocalStemLevels("skipped-insufficient-active-windows")
        threshold = float(statistics.median(levels)) - FINAL_SELF_CHECK_BELOW_MEDIAN_DB
        return VocalStemLevels("checked", threshold, len(levels), None, samples)
    except (LrcError, OSError, ValueError, subprocess.SubprocessError) as exc:
        return VocalStemLevels("skipped-decoding-error", error=str(exc))


def nearest_local_peak(
    features: AudioFeatures,
    center: float,
    radius: float = 0.18,
    minimum_strength: float = 0.42,
) -> dict[str, float] | None:
    """Return the closest strong local onset peak around a selected timestamp."""
    start = max(0.0, center - radius)
    end = min(features.duration, center + radius)
    indices = np.flatnonzero(
        (features.frame_times >= start) & (features.frame_times <= end)
    )
    candidates: list[tuple[float, float]] = []
    for index in indices:
        strength = float(features.onset_strength[index])
        if strength < minimum_strength:
            continue
        left = float(features.onset_strength[index - 1]) if index > 0 else 0.0
        right = (
            float(features.onset_strength[index + 1])
            if index + 1 < len(features.onset_strength)
            else 0.0
        )
        if strength + 1e-6 < left or strength + 1e-6 < right:
            continue
        candidates.append((float(features.frame_times[index]), strength))
    if not candidates:
        return None
    peak_time, strength = min(
        candidates,
        key=lambda item: (abs(item[0] - center), -item[1]),
    )
    return {
        "time": round(peak_time, 3),
        "strength": round(strength, 3),
        "delta_seconds": round(peak_time - center, 3),
    }


lrc_stage_timings.install(globals(), lrc_stage_timings.STAGE_TIMING_SPEC, owner=True)
