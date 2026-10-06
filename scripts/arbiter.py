from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


GROSS_RESCUE_THRESHOLD_MS = 500
REVIEWER_VALIDITY_TOLERANCE_SECONDS = 0.050
REVIEWER_CURRENT_BINDING_TOLERANCE_SECONDS = 0.010
REVIEWER_VALIDITY_SOFT_FAILURES = frozenset({
    "insufficient-minimum-proof",
    "previous-tail-ownership-unknown",
    "central-provisional",
})

ASR_FUSION_SOURCES = frozenset({
    "raw-vocal-independent-fusion",
    "whisperx-vocal-independent-fusion",
    "local-whisper-vocal-independent-fusion",
    "ctc-whisperx-independent-consensus",
    "ctc-local-whisper-independent-consensus",
    "ctc-local-raw-independent-consensus",
})


@dataclass(frozen=True)
class GrossRescueDecision:
    use_current: bool
    reverted_to_current: bool
    reason: str
    selected_source: str
    shift_ms: int


@dataclass(frozen=True)
class ReviewerValidityDecision:
    endorsed: bool
    reason: str
    agreeing_reviewers: tuple[str, ...]
    present_reviewers: tuple[str, ...]


def reviewer_agreements_at(
    *,
    candidate_seconds: float,
    reviewer_times: Mapping[str, float | None],
    offsets: Mapping[str, float],
    reviewers: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    present: list[str] = []
    agreeing: list[str] = []
    for reviewer in reviewers:
        name = str(reviewer)
        raw = reviewer_times.get(name)
        if raw is None or name not in offsets:
            continue
        value = float(raw)
        if not math.isfinite(value):
            continue
        present.append(name)
        corrected = value - float(offsets[name])
        if abs(corrected - float(candidate_seconds)) <= REVIEWER_VALIDITY_TOLERANCE_SECONDS + 1e-12:
            agreeing.append(name)
    return tuple(agreeing), tuple(present)


def decide_reviewer_validity(
    *,
    current_seconds: float,
    expected_current_seconds: float | None,
    reviewer_times: Mapping[str, float | None],
    offsets: Mapping[str, float],
    reviewers: Sequence[str],
    required_agreements: int,
) -> ReviewerValidityDecision:
    if expected_current_seconds is None or abs(float(current_seconds) - float(expected_current_seconds)) > REVIEWER_CURRENT_BINDING_TOLERANCE_SECONDS + 1e-12:
        return ReviewerValidityDecision(False, "current-binding-mismatch", (), ())
    agreeing, present = reviewer_agreements_at(
        candidate_seconds=float(current_seconds),
        reviewer_times=reviewer_times,
        offsets=offsets,
        reviewers=reviewers,
    )
    endorsed = len(agreeing) >= int(required_agreements)
    return ReviewerValidityDecision(
        endorsed,
        "reviewer-endorsed" if endorsed else "insufficient-reviewer-agreement",
        tuple(agreeing),
        tuple(present),
    )


def reviewer_validity_can_relax(
    rejection_reasons: Sequence[str], *, endorsed: bool
) -> bool:
    if not endorsed:
        return False
    reasons = {str(reason) for reason in rejection_reasons}
    return bool(reasons) and reasons <= REVIEWER_VALIDITY_SOFT_FAILURES


def decide_gross_rescue(
    *,
    current_valid: bool,
    current_centiseconds: int,
    selected_centiseconds: int,
    selected_source: str,
    selected_is_current: bool = False,
) -> GrossRescueDecision:
    """Apply the registered R1 gross-rescue rule to one Central decision."""
    shift_ms = abs(int(selected_centiseconds) - int(current_centiseconds)) * 10
    if not current_valid:
        return GrossRescueDecision(
            False, False, "current-invalid", str(selected_source), shift_ms
        )
    if selected_is_current:
        return GrossRescueDecision(
            True, False, "central-selected-current", str(selected_source), shift_ms
        )
    if shift_ms <= GROSS_RESCUE_THRESHOLD_MS:
        return GrossRescueDecision(
            True, True, "within-gross-rescue-threshold", str(selected_source), shift_ms
        )
    if str(selected_source) in ASR_FUSION_SOURCES:
        return GrossRescueDecision(
            True, True, "registered-asr-fusion", str(selected_source), shift_ms
        )
    return GrossRescueDecision(
        False, False, "gross-rescue-eligible", str(selected_source), shift_ms
    )
