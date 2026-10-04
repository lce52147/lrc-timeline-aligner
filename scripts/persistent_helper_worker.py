#!/usr/bin/env python3
"""Persistent model-helper host for LRC Tools 1.1.1.

The original ctc_align.py / whisperx_refine.py helper is executed unchanged for
one request at a time via runpy.  Only heavyweight model construction is cached
inside this long-lived Python process.  IPC is one JSON object per line.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import runpy
import sys
import traceback
from pathlib import Path
from typing import Any


def _json_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    try:
        return json.dumps([args, sorted(kwargs.items())], ensure_ascii=False, sort_keys=True, default=repr)
    except Exception:
        return repr((args, sorted(kwargs.items())))


class _ModelCacheController:
    """Own model-cache lifecycle without changing helper inference semantics."""

    def __init__(self, mode: str, model_cache: dict[str, object]):
        self.mode = mode
        self.model_cache = model_cache

    def _models(self) -> list[object]:
        models: list[object] = []
        for value in self.model_cache.values():
            model = value[0] if self.mode == "whisperx" and isinstance(value, tuple) and value else value
            if model is not None and hasattr(model, "to"):
                models.append(model)
        return models

    def park(self) -> dict[str, object]:
        """Move cached model weights off CUDA while keeping the process/cache alive."""
        moved = 0
        for model in self._models():
            model.to("cpu")  # type: ignore[attr-defined]
            moved += 1
        gc.collect()
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
        except Exception:
            # Parking the model itself is authoritative; allocator cleanup is best-effort.
            pass
        return {"mode": self.mode, "parked_models": moved, "cache_entries": len(self.model_cache)}


def _requested_device(args: tuple[Any, ...], kwargs: dict[str, Any]) -> object | None:
    if "device" in kwargs:
        return kwargs.get("device")
    # whisperx.load_align_model(language_code, device, model_name=..., model_dir=...)
    return args[1] if len(args) >= 2 else None


def _install_whisperx_cache() -> _ModelCacheController:
    import whisperx  # type: ignore

    original = whisperx.load_align_model
    cache: dict[str, object] = {}

    def cached_load_align_model(*args: Any, **kwargs: Any):
        key = _json_key(args, kwargs)
        if key not in cache:
            cache[key] = original(*args, **kwargs)
        else:
            cached = cache[key]
            if isinstance(cached, tuple) and cached and hasattr(cached[0], "to"):
                device = _requested_device(args, kwargs)
                if device is not None:
                    cached[0].to(device)  # type: ignore[attr-defined]
        return cache[key]

    whisperx.load_align_model = cached_load_align_model  # type: ignore[assignment]
    return _ModelCacheController("whisperx", cache)


def _install_ctc_cache() -> _ModelCacheController:
    import torchaudio  # type: ignore

    original_bundle = torchaudio.pipelines.MMS_FA
    model_cache: dict[str, object] = {}
    aux_cache: dict[str, object] = {}

    class _BundleProxy:
        def __getattr__(self, name: str) -> object:
            return getattr(original_bundle, name)

        def get_model(self, *args: Any, **kwargs: Any):
            key = _json_key(args, kwargs)
            if key not in model_cache:
                model_cache[key] = original_bundle.get_model(*args, **kwargs)
            return model_cache[key]

        def get_tokenizer(self, *args: Any, **kwargs: Any):
            key = "tokenizer:" + _json_key(args, kwargs)
            if key not in aux_cache:
                aux_cache[key] = original_bundle.get_tokenizer(*args, **kwargs)
            return aux_cache[key]

        def get_aligner(self, *args: Any, **kwargs: Any):
            key = "aligner:" + _json_key(args, kwargs)
            if key not in aux_cache:
                aux_cache[key] = original_bundle.get_aligner(*args, **kwargs)
            return aux_cache[key]

    torchaudio.pipelines.MMS_FA = _BundleProxy()  # type: ignore[assignment]
    return _ModelCacheController("ctc", model_cache)


def _run_helper(helper: Path, argv: list[str]) -> dict[str, object]:
    old_argv = sys.argv[:]
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    code = 0
    try:
        sys.argv = [str(helper), *argv]
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
            try:
                runpy.run_path(str(helper), run_name="__main__")
            except SystemExit as exc:
                if exc.code is None:
                    code = 0
                elif isinstance(exc.code, int):
                    code = int(exc.code)
                else:
                    code = 1
                    print(str(exc.code), file=sys.stderr)
    except BaseException:
        code = 1
        traceback.print_exc(file=captured_stderr)
    finally:
        sys.argv = old_argv
    return {
        "type": "result",
        "returncode": code,
        "stdout": captured_stdout.getvalue()[-32768:],
        "stderr": captured_stderr.getvalue()[-32768:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ctc", "whisperx"), required=True)
    parser.add_argument("--helper", required=True)
    args = parser.parse_args()
    helper = Path(args.helper).resolve()
    if not helper.is_file():
        print(json.dumps({"type": "fatal", "error": "helper-not-found"}), flush=True)
        return 2

    try:
        if args.mode == "ctc":
            cache_controller = _install_ctc_cache()
        else:
            cache_controller = _install_whisperx_cache()
    except BaseException as exc:
        print(json.dumps({"type": "fatal", "error": f"init:{type(exc).__name__}:{exc}"}, ensure_ascii=False), flush=True)
        return 3

    print(json.dumps({"type": "ready", "mode": args.mode}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"type": "protocol-error", "error": "invalid-json"}), flush=True)
            continue
        request_type = request.get("type")
        if request_type == "shutdown":
            print(json.dumps({"type": "bye"}), flush=True)
            return 0
        if request_type == "park":
            request_id = request.get("request_id")
            try:
                stats = cache_controller.park()
                print(json.dumps({"type": "parked", "request_id": request_id, **stats}, ensure_ascii=False, separators=(",", ":")), flush=True)
            except BaseException as exc:
                print(json.dumps({"type": "park-error", "request_id": request_id, "error": f"{type(exc).__name__}:{exc}"}, ensure_ascii=False), flush=True)
            continue
        argv = request.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            print(json.dumps({"type": "protocol-error", "error": "invalid-argv"}), flush=True)
            continue
        result = _run_helper(helper, argv)
        result["request_id"] = request.get("request_id")
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
