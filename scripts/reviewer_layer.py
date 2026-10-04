from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Literal, Mapping, Sequence


DEFAULT_TOLERANCE_SECONDS = 0.050
OFFSET_SAMPLE_LIMIT_SECONDS = 0.500


TrustProfileName = Literal["none", "strict", "balanced", "loose"]


@dataclass(frozen=True)
class ReviewerTrustProfile:
    name: TrustProfileName
    tau_percent: float | None
    selected_rule_id: str
    enabled: bool
    reason: str


# These profiles govern only the auxiliary reviewer trust label.  The default
# remains disabled, strict fails closed, and balanced/loose use the fixed
# shipped four-reviewer policies.  They do not alter R2b timestamp selection or
# --arbiter behavior.
TRUST_PROFILES: dict[TrustProfileName, ReviewerTrustProfile] = {
    "none": ReviewerTrustProfile(
        name="none",
        tau_percent=None,
        selected_rule_id="ALL_REVIEW",
        enabled=False,
        reason="trust-labeling-disabled",
    ),
    "strict": ReviewerTrustProfile(
        name="strict",
        tau_percent=2.0,
        selected_rule_id="ALL_REVIEW",
        enabled=False,
        reason="cv-target-not-met",
    ),
    "balanced": ReviewerTrustProfile(
        name="balanced",
        tau_percent=5.0,
        selected_rule_id="HUBP+WX+XLSR+HUB/>=3",
        enabled=True,
        reason="fixed-shipped-reviewer-rule",
    ),
    "loose": ReviewerTrustProfile(
        name="loose",
        tau_percent=10.0,
        selected_rule_id="HUBP+WX+XLSR+HUB/>=2",
        enabled=True,
        reason="fixed-shipped-reviewer-rule",
    ),
}
DEFAULT_TRUST_PROFILE: TrustProfileName = "none"
SHIPPED_REVIEWERS = ("HUBP", "WX", "XLSR", "HUB")
_PROFILE_RULES = {
    "HUBP+WX+XLSR+HUB/>=3": "at-least-3",
    "HUBP+WX+XLSR+HUB/>=2": "at-least-2",
}


@dataclass(frozen=True)
class ReviewerRowDecision:
    reviewers: tuple[str, ...]
    present_count: int
    agree_count: int
    missing_reviewers: tuple[str, ...]
    agreeing_reviewers: tuple[str, ...]
    corrected_times: dict[str, float | None]
    label: Literal["L0", "L1", "L2", "L3"]


def resolve_trust_profile(name: str | None = None) -> ReviewerTrustProfile:
    key = DEFAULT_TRUST_PROFILE if name is None else str(name)
    try:
        return TRUST_PROFILES[key]  # type: ignore[index]
    except KeyError as exc:
        choices = ", ".join(TRUST_PROFILES)
        raise ValueError(f"unsupported reviewer trust profile: {key}; choose one of {choices}") from exc


def trusted_by_profile(
    decision: ReviewerRowDecision,
    *,
    profile: str | None = None,
) -> bool:
    config = resolve_trust_profile(profile)
    if not config.enabled or config.selected_rule_id == "ALL_REVIEW":
        return False
    rule = _PROFILE_RULES.get(config.selected_rule_id)
    if rule is None:
        raise RuntimeError(
            f"enabled reviewer trust rule is not implemented: {config.selected_rule_id}"
        )
    if len(decision.reviewers) != len(SHIPPED_REVIEWERS) or set(decision.reviewers) != set(SHIPPED_REVIEWERS):
        return False
    return trusted_by_rule(decision, rule=rule, view="full")


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def estimate_offset_seconds(
    final_times: Sequence[float | None],
    reviewer_times: Sequence[float | None],
    *,
    sample_limit_seconds: float = OFFSET_SAMPLE_LIMIT_SECONDS,
) -> float:
    if len(final_times) != len(reviewer_times):
        raise ValueError("final/reviewer time lengths differ")
    diffs = [
        float(reviewer) - float(final)
        for final, reviewer in zip(final_times, reviewer_times, strict=True)
        if _finite(final)
        and _finite(reviewer)
        and abs(float(reviewer) - float(final)) < sample_limit_seconds
    ]
    return float(statistics.median(diffs)) if diffs else 0.0


def classify_row(
    *,
    final_time: float,
    reviewer_times: Mapping[str, float | None],
    offsets: Mapping[str, float],
    reviewers: Sequence[str],
    tolerance_seconds: float = DEFAULT_TOLERANCE_SECONDS,
) -> ReviewerRowDecision:
    names = tuple(str(name) for name in reviewers)
    corrected: dict[str, float | None] = {}
    present: list[str] = []
    agreeing: list[str] = []
    missing: list[str] = []
    for name in names:
        raw = reviewer_times.get(name)
        if not _finite(raw) or name not in offsets or not _finite(offsets.get(name)):
            corrected[name] = None
            missing.append(name)
            continue
        value = float(raw) - float(offsets[name])
        corrected[name] = value
        present.append(name)
        if abs(value - float(final_time)) <= tolerance_seconds + 1e-12:
            agreeing.append(name)

    agree_count = len(agreeing)
    present_count = len(present)
    if present_count > 0 and agree_count == present_count:
        label: Literal["L0", "L1", "L2", "L3"] = "L3"
    elif agree_count >= 2:
        label = "L2"
    elif agree_count == 1:
        label = "L1"
    else:
        label = "L0"
    return ReviewerRowDecision(
        reviewers=names,
        present_count=present_count,
        agree_count=agree_count,
        missing_reviewers=tuple(missing),
        agreeing_reviewers=tuple(agreeing),
        corrected_times=corrected,
        label=label,
    )


def trusted_by_rule(
    decision: ReviewerRowDecision,
    *,
    rule: Literal["all", "at-least-2", "at-least-3"],
    view: Literal["common", "full"],
) -> bool:
    if view not in {"common", "full"}:
        raise ValueError(f"unsupported reviewer view: {view}")
    expected = len(decision.reviewers)
    if decision.present_count != expected:
        return False
    if rule == "all":
        return decision.agree_count == expected
    if rule == "at-least-2":
        if expected < 3:
            raise ValueError("at-least-2 rule is only registered for reviewer sets of size >= 3")
        return decision.agree_count >= 2
    if rule == "at-least-3":
        if expected < 4:
            raise ValueError("at-least-3 rule is only registered for reviewer sets of size >= 4")
        return decision.agree_count >= 3
    raise ValueError(f"unsupported reviewer rule: {rule}")
