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
