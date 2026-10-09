"""Research-only vocal recurrence evidence for LRC 1.3 S1.

Produces non-authoritative acoustic matches; no BK timestamps, Central-selected
clocks, or reviewer offsets enter retrieval. A similar sound is NOT independent
proof of lyric identity. Central must still validate any later candidate.
Lightweight runtime: NumPy only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AcousticRepeatMatch:
    time: float
    similarity: float


@dataclass(frozen=True)
class AcousticRepeatScan:
    status: str
    template_duration: float
    matches: tuple[AcousticRepeatMatch, ...]


def vocal_cepstral_features(frames: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Compact per-frame timbre descriptors and absolute vocal loudness (dB)."""
    samples = np.asarray(frames, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] < 64 or sample_rate < 8000:
        raise ValueError("expected [frames, window_samples] and valid sample rate")
    nfft = samples.shape[1]
    frequencies = np.fft.rfftfreq(nfft, 1 / sample_rate)
    edges = np.geomspace(160, min(5500, sample_rate * .45), 37)
    spectrum = np.abs(np.fft.rfft(
        samples * np.hanning(nfft).astype(np.float32), axis=1,
    )).astype(np.float32)
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        selection = (frequencies >= low) & (frequencies < high)
        bands.append(np.log1p(spectrum[:, selection].sum(axis=1)))
    log_bands = np.asarray(bands, dtype=np.float32).T
    # DCT-II without SciPy: discard global level, preserve 15 timbral coefficients.
    positions = np.arange(36, dtype=np.float32) + .5
    orders = np.arange(1, 16, dtype=np.float32)
    basis = np.cos((np.pi / 36.0) * orders[:, None] * positions[None, :])
    descriptors = log_bands @ basis.T
    descriptors = (descriptors - descriptors.mean(axis=0)) / (
        descriptors.std(axis=0) + 1e-4
    )
    descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-4)
    rms = np.sqrt(np.mean(samples * samples, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-9)
    return descriptors.astype(np.float32), rms_db.astype(np.float32)


def scan_vocal_recurrences(
    features: np.ndarray,
    rms_db: np.ndarray,
    *,
    hop_seconds: float,
    template_start: float,
    template_end: float,
    min_similarity: float = .60,
    min_spacing_seconds: float = 3.0,
    min_active_fraction: float = .65,
    max_matches: int = 40,
) -> AcousticRepeatScan:
    """Whole-stem normalized timbre autocorrelation, with a vocal activity gate.

    Does not use the alignment's other timestamps. The template is an explicit
    acoustic interval, normally obtained from an already independently
    qualified verse. Output times are untrusted hypotheses, not lyric owners.
    """
    x = np.asarray(features, dtype=np.float32)
    db = np.asarray(rms_db, dtype=np.float32)
    if x.ndim != 2 or len(x) != len(db) or not len(x) or x.shape[1] < 2:
        return AcousticRepeatScan("BLOCKED_INVALID_FEATURES", 0., ())
    if not math.isfinite(hop_seconds) or hop_seconds <= 0:
        raise ValueError("invalid frame hop")
    start = int(round(template_start / hop_seconds))
    end = int(round(template_end / hop_seconds))
    n = end - start
    duration = n * hop_seconds
    if n < 50 or end > len(x) or start < 0 or n >= len(x):
        return AcousticRepeatScan("BLOCKED_TEMPLATE_BOUNDS", duration, ())
    if not .01 < min_similarity <= 1 or min_spacing_seconds < 0 or not 0 < min_active_fraction <= 1:
        raise ValueError("invalid matching thresholds")
    active = db[np.isfinite(db) & (db > -60)]
    if len(active) < 12:
        return AcousticRepeatScan("BLOCKED_NO_VOCAL_ACTIVITY", duration, ())
    vocal_floor = float(np.median(active)) - 20.
    window_flags = (db >= vocal_floor).astype(np.float32)
    count = len(x) - n + 1
    active_counts = np.convolve(window_flags, np.ones(n, dtype=np.float32), mode="valid") / n
    if active_counts[start] < min_active_fraction:
        return AcousticRepeatScan("BLOCKED_SILENT_REFERENCE", duration, ())
    # FFT correlation across timbre dimensions, no quadratic sliding-window materialization.
    fft_len = 1 << (len(x) + n - 2).bit_length()
    total = np.zeros(count, dtype=np.float64)
    template = x[start:end, :]
    for j in range(x.shape[1]):
        full = np.fft.irfft(
            np.fft.rfft(x[:, j], n=fft_len)
            * np.fft.rfft(template[::-1, j], n=fft_len),
            n=fft_len,
        )
        total += full[n - 1:len(x)] / n
    acceptable = (total >= min_similarity) & (active_counts >= min_active_fraction)
    # Local maxima; keep strongest nonoverlapping candidates first.
    peaks = np.flatnonzero(
        (total[1:-1] >= total[:-2]) & (total[1:-1] > total[2:])
    ) + 1
    peaks = sorted((int(i) for i in peaks if acceptable[i]), key=lambda i: (-total[i], i))
    selected: list[int] = []
    separation = int(math.ceil(min_spacing_seconds / hop_seconds))
    for i in peaks:
        if all(abs(i - previous) >= separation for previous in selected):
            selected.append(i)
        if len(selected) >= max_matches:
            break
    results = tuple(
        AcousticRepeatMatch(round(i * hop_seconds, 3), round(float(total[i]), 4))
        for i in sorted(selected)
    )
    return AcousticRepeatScan("RESEARCH_UNVERIFIED_OCCURRENCE", duration, results)


def unique_repeat_anchor(
    scan: AcousticRepeatScan,
    *,
    earliest: float,
    latest: float,
    min_similarity: float = .60,
    exclude_time: float | None = None,
    exclude_radius: float = 1.0,
) -> AcousticRepeatMatch | None:
    """Fail closed if one line's independently bounded slot has 0 or 2+ matches."""
    if scan.status != "RESEARCH_UNVERIFIED_OCCURRENCE" or not earliest < latest:
        return None
    qualified = [
        p for p in scan.matches
        if earliest <= p.time <= latest
        and p.similarity >= min_similarity
        and (exclude_time is None or abs(p.time - exclude_time) > exclude_radius)
    ]
    return qualified[0] if len(qualified) == 1 else None
