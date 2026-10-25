"""Reader-facing presentation of review rows: at most three timing options and one plain sentence.

Pure functions, no I/O. ``review_server`` calls them when it builds a song payload; the
review page shows their output and keeps the raw candidates behind a "details" fold.
"""
from __future__ import annotations

from typing import Any

# Candidates closer than this to one another are one option (centiseconds).
OPTION_CLUSTER_CS = 25
# Candidates this close to the current timestamp are folded into "keep".
KEEP_FOLD_CS = 15
MAX_ALTERNATIVES = 2

# Highest-priority group first; the first group with a matching code decides the sentence.
REASON_SENTENCES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("F1-vocal-absent",), "這段可能沒有人聲，請確認是否該刪或移到下一句。"),
    (("F3-compressed-or-nonmonotonic", "F3-nonmonotonic", "nonmonotonic-line-start", "temporal-incoherence",
      "ctc_zero_gap_compressed_prefix"), "這句的時間順序怪怪的，請聽前後兩句。"),
    (("previous-tail-overlap", "previous-tail-ownership-unknown", "tail-conflict-without-independent-current-opening",
      "no-strict-opening-proof"), "這句可能提早開唱（跟上一句尾巴黏在一起），請聽開頭。"),
    (("unresolved-material-alternative", "candidate_disagreement", "unresolved_raw_ctc_disagreement",
      "phonetic_anchor_disagreement", "model-disagreement", "unresolved-precentral-disagreement",
      "F4-ctc-final-disagreement", "ctc_confidence_collapse", "group_placement_unsupported"),
     "幾個偵測結果不一致，請聽並挑一個。"),
    (("silent-risk",), "這句之後有一段長空白，請確認結束位置沒有抓錯。"),
    (("recurrence-uncorroborated", "duplicate_occurrence_offset_mismatch"), "這句歌詞重複出現，可能對到別段，請確認是這一次。"),
    (("insufficient-minimum-proof", "timing-evidence-unverified", "unverified", "review-required"),
     "系統證據不足，請聽一下確認。"),
)
DEFAULT_SENTENCE = "需要人工確認。"
AGREEMENT_SENTENCE = "候選一致，只需確認。"


def _code_text(value: object) -> str:
    if isinstance(value, dict):
        for key in ("code", "reason", "name"):
            if isinstance(value.get(key), str):
                return value[key]
        return ""
    return str(value or "")


def reason_sentence(codes: object, *, options_count: int | None = None) -> str:
    """One plain sentence for a row's reason codes (priority order, first hit wins)."""
    values = codes if isinstance(codes, list) else ([] if codes is None else [codes])
    texts = [_code_text(value).strip() for value in values]
    texts = [text for text in texts if text]
    for keys, sentence in REASON_SENTENCES:
        for text in texts:
            if any(text == key or text.startswith(key + ":") for key in keys):
                if options_count == 0 and sentence == REASON_SENTENCES[-1][1]:
                    return AGREEMENT_SENTENCE
                return sentence
    if texts:
        return DEFAULT_SENTENCE
    return AGREEMENT_SENTENCE if options_count == 0 else DEFAULT_SENTENCE


def _int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def present_options(candidates: object, extra_candidates: object, current_cs: object) -> list[dict[str, Any]]:
    """Collapse the candidate rows into at most two alternatives plus "keep".

    Returns options sorted by time. Each has ``cs``, ``kind`` (earlier/keep/later), ``label``
    (A/B for alternatives, 維持 for keep), ``delta_s`` relative to current, ``support`` (how many
    raw candidates agree), ``score`` and ``sources``.
    """
    current = _int(current_cs)
    pool: list[dict[str, Any]] = []
    for group in (candidates, extra_candidates):
        for item in group if isinstance(group, list) else []:
            if not isinstance(item, dict) or item.get("rejected") or item.get("diagnostic_only"):
                continue
            cs = _int(item.get("cs"))
            if cs is None:
                continue
            if current is not None and abs(cs - current) <= KEEP_FOLD_CS:
                continue
            members = item.get("members") if isinstance(item.get("members"), list) else []
            distinct = {_int(m.get("cs")) for m in members if isinstance(m, dict) and not m.get("diagnostic") and not m.get("rejected")}
            distinct.discard(None)
            pool.append({
                "cs": cs,
                "score": float(item.get("score") or 0.0),
                "support": max(1, len(distinct)),
                "sources": [str(s) for s in (item.get("sources") or [item.get("label") or "candidate"])],
            })
    clusters: list[dict[str, Any]] = []
    for member in sorted(pool, key=lambda item: (-item["score"], item["cs"])):
        home = next((c for c in clusters if abs(c["cs"] - member["cs"]) <= OPTION_CLUSTER_CS), None)
        if home is None:
            clusters.append({**member, "sources": list(member["sources"])})
        else:
            home["support"] += member["support"]
            home["score"] = max(home["score"], member["score"])
            for source in member["sources"]:
                if source not in home["sources"]:
                    home["sources"].append(source)
    ranked = sorted(clusters, key=lambda c: (-c["score"], -c["support"], c["cs"]))[:MAX_ALTERNATIVES]
    options: list[dict[str, Any]] = []
    for cluster in sorted(ranked, key=lambda c: c["cs"]):
        delta = (cluster["cs"] - current) if current is not None else 0
        options.append({
            "cs": cluster["cs"],
            "kind": "earlier" if delta < 0 else "later",
            "label": "",
            "delta_s": round(delta / 100.0, 2),
            "support": cluster["support"],
            "score": cluster["score"],
            "sources": cluster["sources"],
        })
    for index, option in enumerate(options):
        option["label"] = "AB"[index]
    if current is not None:
        keep = {"cs": current, "kind": "keep", "label": "維持", "delta_s": 0.0, "support": 0, "score": 0.0, "sources": ["current"]}
        options.append(keep)
        options.sort(key=lambda o: o["cs"])
    return options


# Review risk tiers (R3 study, 36 checked songs): "high" lines held 38 of the 45 REVIEW lines
# off by more than 0.5 s, "low" lines held 2 of them, so the page lists high first and folds low.
RISK_AGREE_TOLERANCE_CS = 30
RISK_AGREE_MIN_SOURCES = 5
RISK_COMPETITOR_DISTANCE_CS = 50
RISK_COMPETITOR_MAX_SCORE = 0.7
RISK_MANY_CODES = 6


def _seconds_to_cs(value: object) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return round(number * 100)


# Fit between neighbours (R5 study, same 36 songs): a "normal" line whose written time leaves
# room for the previous line to finish, fits before the next line and whose CTC path is at most
# twice its expected length joins "low". The expected length of a line is its romaji length times
# the song's speech rate. At least one neighbour must sit within 2 s of the expected boundary:
# with both neighbours far away a whole block shifted by several seconds still "fits" (two such
# lines in the real data). With all of this, 47 normal lines move to "low" (44 within 0.2 s of
# the checked time, 3 within 0.5 s, none further off), and leave-one-song-out picks the same rule
# every time with the same result. Without the neighbour condition it was 53 lines, 50 within 0.2 s.
FIT_SLACK_MIN_S = -0.5
FIT_PIN_MAX_S = 2.0
FIT_SPAN_RATIO_MAX = 2.0
FIT_RATE_MIN_LINES = 5
FIT_RATE_MAX_GAP_S = 4.0
FIT_RATE_MIN_CHARS = 4
FIT_DEFAULT_RATE_S_PER_CHAR = 0.12


def _romaji_length(assignment: object) -> int:
    romaji = assignment.get("romaji") if isinstance(assignment, dict) else None
    return len(romaji) if isinstance(romaji, str) else 0


def _token_span_cs(assignment: object) -> int | None:
    spans = assignment.get("ctc_token_spans") if isinstance(assignment, dict) else None
    if not isinstance(spans, list):
        return None
    starts = [_seconds_to_cs(s.get("start")) for s in spans if isinstance(s, dict)]
    ends = [_seconds_to_cs(s.get("end")) for s in spans if isinstance(s, dict)]
    starts = [v for v in starts if v is not None]
    ends = [v for v in ends if v is not None]
    if not starts or not ends:
        return None
    return max(ends) - min(starts)


def _line_trusted(assignment: object) -> bool:
    return (
        isinstance(assignment, dict)
        and assignment.get("verification_status") == "trusted"
        and assignment.get("overall_trusted") is not False
    )


def song_speech_rate(assignments: object, written_cs: object) -> float:
    """Seconds sung per romaji character, median over the song's trusted lines.

    ``written_cs`` holds one written time (centiseconds, or None) per assignment. Gaps of
    4 s or more are instrumental breaks, not singing, and lines under four characters are
    too short to time; with fewer than five usable trusted lines every line counts, and a
    song with none falls back to 0.12 s per character.
    """
    rows = assignments if isinstance(assignments, list) else []
    times = list(written_cs) if isinstance(written_cs, (list, tuple)) else []

    def rates(trusted_only: bool) -> list[float]:
        values: list[float] = []
        for index, assignment in enumerate(rows):
            if index + 1 >= len(times) or index >= len(times):
                break
            if trusted_only and not _line_trusted(assignment):
                continue
            current, following = _int(times[index]), _int(times[index + 1])
            chars = _romaji_length(assignment)
            if current is None or following is None or chars < FIT_RATE_MIN_CHARS:
                continue
            gap = following - current
            if 0 < gap < FIT_RATE_MAX_GAP_S * 100:
                values.append(gap / 100.0 / chars)
        return values

    values = rates(True)
    if len(values) < FIT_RATE_MIN_LINES:
        values = rates(False)
    if not values:
        return FIT_DEFAULT_RATE_S_PER_CHAR
    values.sort()
    return values[len(values) // 2]


def line_fit(
    assignment: object,
    written_cs: object,
    previous: object,
    previous_cs: object,
    following_cs: object,
    rate: float,
) -> dict[str, Any] | None:
    """How well a line's written time fits between its neighbours and its own CTC path.

    Returns ``slack_prev_s`` (room left after the previous line's expected end; None for the
    first line), ``slack_next_s`` (room before the next line; None for the last line),
    ``span_ratio`` (CTC token span over expected length) and ``fits`` (both slacks above
    FIT_SLACK_MIN_S, the smaller one at most FIT_PIN_MAX_S, span ratio at most
    FIT_SPAN_RATIO_MAX). None when the line has no romaji, no written time or no previous
    line to measure against.
    """
    current = _int(written_cs)
    chars = _romaji_length(assignment)
    previous_time = _int(previous_cs)
    previous_chars = _romaji_length(previous)
    if current is None or chars <= 0 or previous_time is None or previous_chars <= 0:
        return None
    expected = chars * rate * 100.0
    slack_prev = (current - (previous_time + previous_chars * rate * 100.0)) / 100.0
    following = _int(following_cs)
    slack_next = (following - (current + expected)) / 100.0 if following is not None else None
    span = _token_span_cs(assignment)
    span_ratio = span / expected if span is not None and expected > 0 else None
    pinned = min(value for value in (slack_prev, slack_next) if value is not None) <= FIT_PIN_MAX_S
    fits = bool(
        slack_prev > FIT_SLACK_MIN_S
        and (slack_next is None or slack_next > FIT_SLACK_MIN_S)
        and pinned
        and span_ratio is not None
        and span_ratio <= FIT_SPAN_RATIO_MAX
    )
    return {
        "slack_prev_s": round(slack_prev, 2),
        "slack_next_s": round(slack_next, 2) if slack_next is not None else None,
        "span_ratio": round(span_ratio, 2) if span_ratio is not None else None,
        "fits": fits,
    }


def classify_risk(
    assignment: dict,
    codes: object,
    written_cs: object,
    whisperx_entries: object = (),
    fit: object = None,
) -> tuple[str, str]:
    """Return (tier, basis) for one REVIEW line.

    Tiers are "high", "normal" or "low". low: at least five candidate sources (accepted
    candidates and alignment hypotheses, rejected ones excluded) sit within 0.3 s of the
    written time, and no accepted candidate more than 0.5 s away scores above 0.7 (basis
    "agreement"); or the line is not high and ``fit`` (from ``line_fit``) says it fits between
    its neighbours (basis "fit"). high: the line carries model-disagreement ("model-disagreement"),
    or more than six reason codes without WhisperX row evidence ("many-codes"). Everything
    else is normal ("default").
    """
    written = _int(written_cs)
    code_set = {str(code) for code in codes} if isinstance(codes, (list, tuple, set)) else set()
    if written is not None:
        agreeing: set[str] = set()
        competitor = 0.0
        for item in assignment.get("candidates") or []:
            if not isinstance(item, dict):
                continue
            cs = _seconds_to_cs(item.get("time"))
            if cs is None:
                continue
            if abs(cs - written) <= RISK_AGREE_TOLERANCE_CS:
                agreeing.add(str(item.get("source")))
            elif abs(cs - written) > RISK_COMPETITOR_DISTANCE_CS:
                try:
                    competitor = max(competitor, float(item.get("score") or 0.0))
                except (TypeError, ValueError):
                    pass
        for item in assignment.get("alignment_hypotheses") or []:
            if not isinstance(item, dict):
                continue
            cs = _seconds_to_cs(item.get("raw_candidate_time"))
            if cs is not None and abs(cs - written) <= RISK_AGREE_TOLERANCE_CS:
                agreeing.add(str(item.get("source")))
        if len(agreeing) >= RISK_AGREE_MIN_SOURCES and competitor <= RISK_COMPETITOR_MAX_SCORE:
            return "low", "agreement"
    try:
        entries = {int(value) for value in whisperx_entries or ()}
    except (TypeError, ValueError):
        entries = set()
    has_whisperx = _int(assignment.get("entry")) in entries
    if "model-disagreement" in code_set:
        return "high", "model-disagreement"
    if len(code_set) > RISK_MANY_CODES and not has_whisperx:
        return "high", "many-codes"
    if isinstance(fit, dict) and fit.get("fits"):
        return "low", "fit"
    return "normal", "default"


SONG_RISK_REVIEW_SHARE = 0.5
SONG_RISK_MIN_ROWS = 8
SONG_RISK_ANCHOR_SHARE = 0.20


def song_risk(report: object) -> dict[str, Any]:
    """Whole-song risk class for the review page (LRC 2.0 P4, the Choir class).

    ``"choir"`` when the song gave the structural map almost no anchored rows (the
    segment-realign stage reported ``weak-anchors`` with under 20 % of the rows
    anchored: no corroboration for most lines, as with choirs, doubled vocals and
    heavy effects) or when at least half of its lines (eight or more) need review.  Such a song is listened
    to as a whole rather than line by line; nothing in the alignment changes.
    """
    rows = report.get("assignments") if isinstance(report, dict) else None
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    review_rows = sum(1 for row in rows if row.get("review_required"))
    share = review_rows / len(rows) if rows else 0.0
    stage = report.get("segment_realign") if isinstance(report, dict) else None
    stage = stage if isinstance(stage, dict) else {}
    structure = stage.get("structure") if isinstance(stage.get("structure"), dict) else {}
    reasons: list[str] = []
    anchors = structure.get("anchors")
    anchor_rows = structure.get("rows")
    if (
        stage.get("status") == "weak-anchors"
        and isinstance(anchors, int)
        and isinstance(anchor_rows, int)
        and anchor_rows > 0
        and anchors / anchor_rows < SONG_RISK_ANCHOR_SHARE
    ):
        reasons.append("weak-anchors")
    if len(rows) >= SONG_RISK_MIN_ROWS and share >= SONG_RISK_REVIEW_SHARE:
        reasons.append("review-share")
    return {
        "class": "choir" if reasons else None,
        "reasons": reasons,
        "rows": len(rows),
        "review_rows": review_rows,
        "review_share": round(share, 3),
        "anchors": anchors,
        "anchor_rows": anchor_rows,
        "segment_realign": stage.get("status"),
    }


def review_risk(
    assignment: dict,
    codes: object,
    written_cs: object,
    whisperx_entries: object = (),
    fit: object = None,
) -> str:
    """Return "high", "normal" or "low" for one REVIEW line (see ``classify_risk``)."""
    return classify_risk(assignment, codes, written_cs, whisperx_entries, fit)[0]
