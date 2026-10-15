"""Compare a full-39 GPU production manifest with its cache-only replay.

The GPU manifest is produced by the existing production runner.  This small
validator deliberately treats absent inputs as SKIP so a missing cache cannot
be reported as a passing replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = ("PCO@0.2", ">0.5 s", ">1 s", "TRUSTED>1 s")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return value


def _rows(manifest: dict[str, Any]) -> list[Any]:
    for key in ("rows", "results", "songs"):
        value = manifest.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("manifest has no rows/results/songs list")


def _metrics(manifest: dict[str, Any]) -> dict[str, int]:
    metrics = manifest.get("metrics")
    if isinstance(metrics, dict):
        aliases = {
            "PCO@0.2": ("PCO@0.2", "pco02", "pco_02"),
            ">0.5 s": (">0.5 s", "gt05", "gt_05"),
            ">1 s": (">1 s", "gt1", "gt_1"),
            "TRUSTED>1 s": ("TRUSTED>1 s", "trusted_gt1", "trusted_gt_1"),
        }
        out: dict[str, int] = {}
        for name, keys in aliases.items():
            for key in keys:
                if key in metrics:
                    out[name] = int(metrics[key])
                    break
        if len(out) == len(METRIC_KEYS):
            return out
    raise ValueError("manifest metrics must contain PCO@0.2, >0.5 s, >1 s, TRUSTED>1 s")


def compare(gpu: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    gpu_rows = _rows(gpu)
    replay_rows = _rows(replay)
    if gpu_rows != replay_rows:
        raise AssertionError(f"row mismatch: gpu={len(gpu_rows)} replay={len(replay_rows)}")
    gpu_metrics = _metrics(gpu)
    replay_metrics = _metrics(replay)
    if gpu_metrics != replay_metrics:
        raise AssertionError(f"metric mismatch: gpu={gpu_metrics} replay={replay_metrics}")
    return {"rows": len(gpu_rows), "metrics": replay_metrics, "candidate_only_regression": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-manifest", type=Path, required=True)
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    missing = [str(path) for path in (args.gpu_manifest, args.replay_manifest) if not path.is_file()]
    if not args.cache_dir.is_dir() or not any(args.cache_dir.rglob("*.json")):
        missing.append(f"cache:{args.cache_dir}")
    if missing:
        print("SKIP: missing " + ", ".join(missing))
        return 2
    try:
        result = compare(_read(args.gpu_manifest), _read(args.replay_manifest))
    except (OSError, ValueError, AssertionError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
