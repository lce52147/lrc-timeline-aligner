"""Compare two align-report JSON files key by key (LRC 1.5 W4).

Usage: report_diff.py BEFORE.json AFTER.json [--allow-changed KEY,KEY] [--json]

Path-typed keys are never compared by value: the keys listed in
``auto_lrc._GENERATION_DIAGNOSTIC_LOCATOR_KEYS``, ``review_tsv_path``,
``multimodel_consensus.cache_path`` and any sub-dictionary named ``paths``.
Exit status is 0 when AFTER only adds keys, 1 when a key was removed or a
value changed without being listed in ``--allow-changed``, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:  # auto_lrc is heavy; fall back to a frozen copy of the locator keys.
    from auto_lrc import _GENERATION_DIAGNOSTIC_LOCATOR_KEYS as _LOCATOR_KEYS
except Exception:  # pragma: no cover - import failure only
    _LOCATOR_KEYS = frozenset({
        "helper_path", "alignment_audio_path", "ctc_audio_path", "audio_path",
        "output_path", "report_path", "lyrics_path", "whisper_cli_path",
        "whisper_model_path", "whisperx_python_path", "whisperx_helper_path",
        "temp_path", "temp_paths", "temp_dir", "cache_path", "cache_dir",
        "cache_root", "cache_file",
    })

PATH_KEYS = frozenset(_LOCATOR_KEYS) | {"review_tsv_path"}
PATH_DOTTED = frozenset({"multimodel_consensus.cache_path"})
PATH_SUBTREES = frozenset({"paths"})


def _is_path_key(path: tuple[str, ...]) -> bool:
    if not path:
        return False
    leaf = path[-1]
    if leaf in PATH_KEYS or leaf in PATH_SUBTREES:
        return True
    dotted = ".".join(path)
    return any(dotted == item or dotted.endswith("." + item) for item in PATH_DOTTED)


def diff_reports(
    before: object, after: object, *, prefix: tuple[str, ...] = ()
) -> dict[str, list[str]]:
    """Return {"added": [...], "removed": [...], "changed": [...]} dotted keys."""
    result: dict[str, list[str]] = {"added": [], "removed": [], "changed": []}
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after), key=str):
            path = prefix + (str(key),)
            if _is_path_key(path):
                continue
            if key not in before:
                result["added"].append(".".join(path))
            elif key not in after:
                result["removed"].append(".".join(path))
            else:
                sub = diff_reports(before[key], after[key], prefix=path)
                for name in result:
                    result[name].extend(sub[name])
        return result
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        for index, (left, right) in enumerate(zip(before, after)):
            sub = diff_reports(left, right, prefix=prefix + (f"[{index}]",))
            for name in result:
                result[name].extend(sub[name])
        return result
    if before != after or type(before) is not type(after):
        result["changed"].append(".".join(prefix) or "<root>")
    return result


def _strip_index(key: str) -> str:
    return ".".join(part for part in key.split(".") if not (part.startswith("[") and part.endswith("]")))


def unexpected_changes(diff: dict[str, list[str]], allow_changed: Iterable[str]) -> list[str]:
    allowed = {item.strip() for item in allow_changed if item.strip()}
    bad = list(diff["removed"])
    for key in diff["changed"]:
        plain = _strip_index(key)
        if key in allowed or plain in allowed or plain.rsplit(".", 1)[-1] in allowed:
            continue
        bad.append(key)
    return bad


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--allow-changed", default="", help="comma-separated keys whose value may change")
    parser.add_argument("--json", action="store_true", help="print the diff as JSON")
    args = parser.parse_args(argv)
    try:
        before = _load(args.before)
        after = _load(args.after)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    diff = diff_reports(before, after)
    bad = unexpected_changes(diff, args.allow_changed.split(","))
    if args.json:
        print(json.dumps({**diff, "unexpected": bad}, ensure_ascii=False, indent=2))
    else:
        for name in ("added", "removed", "changed"):
            for key in diff[name]:
                print(f"{name}: {key}")
        print(f"summary: added={len(diff['added'])} removed={len(diff['removed'])} changed={len(diff['changed'])} unexpected={len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
