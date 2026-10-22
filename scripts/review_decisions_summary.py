"""Read-only tally of reviewer verdicts: how often each review reason code ends in keep vs change."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys


def load_decisions(report_dir: Path) -> list[dict]:
    """Last verdict per (song, entry); a later change overrides an earlier keep and vice versa."""
    latest: dict[tuple[str, int], dict] = {}
    for path in sorted(report_dir.glob("*.review-decisions.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("decision") in ("keep", "change"):
                latest[(path.name, int(record.get("entry", 0)))] = record
    return list(latest.values())


def summarize(records: list[dict]) -> dict:
    by_code: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    by_status: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for record in records:
        by_status[str(record.get("review"))][record["decision"]] += 1
        for code in record.get("reason_codes") or ["(none)"]:
            by_code[code][record["decision"]] += 1
    return {"decisions": len(records), "by_status": by_status, "by_code": by_code}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "reports")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args(argv)
    summary = summarize(load_decisions(args.report_dir))
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    print(f"decisions: {summary['decisions']}")
    for status, counts in sorted(summary["by_status"].items()):
        print(f"  {status}: keep {counts['keep']}, change {counts['change']}")
    print(f"  {'reason code':<48} {'keep':>5} {'change':>7} {'change%':>7}")
    rows = sorted(summary["by_code"].items(), key=lambda item: -sum(item[1].values()))
    for code, counts in rows:
        total = counts["keep"] + counts["change"]
        print(f"  {code:<48} {counts['keep']:5d} {counts['change']:7d} {100 * counts['change'] / total:7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
