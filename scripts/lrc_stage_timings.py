"""Failure transparent wall-clock instrumentation for LRC production stages."""

import contextlib
import functools
import json
import os
import sys
import threading
import time


_CLOCK = time.perf_counter
_LOCK = threading.RLock()
_ACTIVE = None
_TLS = threading.local()
_PENDING = {}
_INSTALL = {"wrapped": [], "missing": []}
_PROGRESS_START = None


def set_clock(fn):
    global _CLOCK
    _CLOCK = time.perf_counter if fn is None else fn


def _round(value):
    try:
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            return 0.0
        return round(max(0.0, value), 6)
    except Exception:
        return 0.0


def _enabled():
    return os.environ.get("LRC_STAGE_TIMINGS", "1") != "0"


def _state_payload(state):
    try:
        total = _round(_CLOCK() - state["t0"])
    except Exception:
        total = 0.0
    stages = state["stages"]
    top = sum(float(row.get("seconds", 0.0)) for row in stages if row.get("parent") is None)
    worker = {}
    for name, row in state["worker_stages"].items():
        worker[name] = {
            "calls": int(row.get("calls", 0)),
            "seconds": _round(row.get("seconds", 0.0)),
            "errors": int(row.get("errors", 0)),
        }
    return {
        "schema": 1,
        "clock": "perf_counter",
        "pass": state["pass"],
        "attempts": int(state.get("attempts", 1)),
        "startup_seconds": state.get("startup_seconds"),
        "total_seconds": total,
        "unaccounted_seconds": _round(max(0.0, total - top)),
        "stages": stages,
        "stages_truncated": int(state.get("stages_truncated", 0)),
        "worker_stages": worker,
        "model_cache": state["model_cache"],
        "vocal_stem": list(state["vocal_stem"]),
    }


def current_payload():
    state = _ACTIVE
    if state is None:
        return None
    try:
        payload = state.get("payload")
        if payload is None:
            payload = _state_payload(state)
            state["payload"] = payload
        else:
            fresh = _state_payload(state)
            payload.clear()
            payload.update(fresh)
        return payload
    except Exception:
        return None


@contextlib.contextmanager
def session(pass_label, audio_key, *, state=None, attempt=1):
    global _ACTIVE
    previous = _ACTIVE
    if state is None:
        try:
            startup_seconds = _round(time.monotonic() - _PROGRESS_START) if _PROGRESS_START is not None else None
        except Exception:
            startup_seconds = None
        state = {
            "pass": str(pass_label),
            "audio_key": str(audio_key),
            "t0": _CLOCK(),
            "attempts": int(attempt),
            "startup_seconds": startup_seconds,
            "owner_thread": threading.get_ident(),
            "stages": [],
            "stages_truncated": 0,
            "worker_stages": {},
            "model_cache": {},
            "vocal_stem": [],
            "payload": None,
            "pending": False,
        }
    else:
        state["attempts"] = int(attempt)
        state["owner_thread"] = threading.get_ident()
        state["pending"] = False
    _ACTIVE = state
    _TLS.stack = []
    try:
        yield state
    finally:
        try:
            payload = _state_payload(state)
            state["payload"] = payload
            state.update(payload)
        except Exception:
            pass
        _ACTIVE = previous
        _TLS.stack = []


def _append_stage(state, row):
    try:
        with _LOCK:
            for existing in state["stages"]:
                if all(existing.get(key) == row.get(key) for key in ("name", "parent", "depth", "attempt")):
                    existing["calls"] += row["calls"]
                    existing["seconds"] = _round(existing["seconds"] + row["seconds"])
                    existing["errors"] += row["errors"]
                    if row["status"] != "ok":
                        existing["status"] = row["status"]
                    return
            if len(state["stages"]) >= 200:
                state["stages_truncated"] = int(state.get("stages_truncated", 0)) + 1
            else:
                state["stages"].append(row)
    except Exception:
        pass


def _record_worker(state, name, started, error=False):
    try:
        elapsed = _round(_CLOCK() - started)
        with _LOCK:
            row = state["worker_stages"].setdefault(name, {"calls": 0, "seconds": 0.0, "errors": 0})
            row["calls"] += 1
            row["seconds"] += elapsed
            if error:
                row["errors"] += 1
    except Exception:
        pass


def _kind(args, kwargs):
    try:
        if args:
            return str(args[0])
        return str(kwargs.get("kind", "unknown"))
    except Exception:
        return "unknown"


def _cache_count(state, kind, field):
    try:
        with _LOCK:
            row = state["model_cache"].setdefault(kind, {"hit": 0, "miss": 0, "fill": 0})
            row[field] = int(row.get(field, 0)) + 1
    except Exception:
        pass


def _is_miss(exc):
    return exc.__class__.__name__ == "ModelCacheMiss"


def _call(fn, args, kwargs, name, *, cache_kind=None, function_name=None):
    state = _ACTIVE
    if state is None or not _enabled():
        return fn(*args, **kwargs)
    is_cache = cache_kind in {"read", "write"}
    if is_cache:
        kind = _kind(args, kwargs)
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            if cache_kind == "read" and _is_miss(exc):
                _cache_count(state, kind, "miss")
            raise
        else:
            if cache_kind == "read":
                _cache_count(state, kind, "hit" if result is not None else "miss")
            else:
                _cache_count(state, kind, "fill")
            return result
    if threading.get_ident() != state.get("owner_thread"):
        try:
            started = _CLOCK()
        except Exception:
            return fn(*args, **kwargs)
        failed = False
        try:
            return fn(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            _record_worker(state, name, started, failed)
    try:
        stack = getattr(_TLS, "stack", [])
        parent = stack[-1]["name"] if stack else None
        row = {
            "name": name,
            "parent": parent,
            "depth": len(stack),
            "attempt": int(state.get("attempts", 1)),
            "start_s": _round(_CLOCK() - state["t0"]),
            "seconds": 0.0,
            "calls": 1,
            "errors": 0,
            "status": "ok",
        }
        stack.append(row)
        _TLS.stack = stack
        started = _CLOCK()
    except Exception:
        return fn(*args, **kwargs)
    try:
        result = fn(*args, **kwargs)
    except BaseException as exc:
        if row is not None:
            try:
                row["status"] = f"error:{type(exc).__name__}"
                row["errors"] = 1
            except Exception:
                pass
        raise
    finally:
        if row is not None:
            try:
                stack.pop()
                row["seconds"] = _round(_CLOCK() - started)
                _append_stage(state, row)
            except Exception:
                pass
    if function_name == "prepare_vocal_ctc_audio":
        try:
            status = result[2] if isinstance(result, tuple) and len(result) >= 3 else None
            if status:
                with _LOCK:
                    state["vocal_stem"].append(str(status)[:80])
        except Exception:
            pass
    return result


def timed(name, fn, *, cache_kind=None, function_name=None):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return _call(fn, args, kwargs, name, cache_kind=cache_kind, function_name=function_name)
    wrapper.__stage_timing_wrapper__ = True
    wrapper.__stage_timing_label__ = name
    return wrapper


def _audio_key(args, kwargs):
    try:
        value = args[0] if args else kwargs.get("audio_path", "")
        return str(value)
    except Exception:
        return ""


def _args_value(args, key, default=None):
    try:
        return getattr(args, key, default)
    except Exception:
        return default


def _print_summary(state, status):
    try:
        if not _enabled():
            return
        payload = _state_payload(state)
        top = [row for row in payload["stages"] if row.get("parent") is None]
        top.sort(key=lambda row: float(row.get("seconds", 0.0)), reverse=True)
        shown = top[:6]
        detail = " | ".join(f"{row['name']} {row['seconds']:.1f}s" for row in shown)
        other = sum(float(row.get("seconds", 0.0)) for row in top[6:])
        cache = payload["model_cache"]
        hit = sum(int(row.get("hit", 0)) for row in cache.values())
        miss = sum(int(row.get("miss", 0)) for row in cache.values())
        fill = sum(int(row.get("fill", 0)) for row in cache.values())
        stem = payload["vocal_stem"][-1] if payload["vocal_stem"] else "n/a"
        suffix = (detail + " | ") if detail else ""
        print(f"TIMINGS pass={payload['pass']} status={status} total={payload['total_seconds']:.1f}s | {suffix}other {other:.1f}s | cache hit={hit} miss={miss} fill={fill} | stem={stem}", flush=True)
    except Exception:
        pass


def _song_wrapper(fn, args, kwargs, name):
    try:
        arg_object = args[1] if len(args) > 1 else kwargs.get("args")
        pass_label = _args_value(arg_object, "stage_timings_pass", "standalone")
        rollback_reason = _args_value(arg_object, "option_a_rollback_reason")
        no_summary = bool(_args_value(arg_object, "no_review_summary", False))
        audio_key = _audio_key(args, kwargs)
        retry = bool(rollback_reason)
        with _LOCK:
            dropped = bool(_PENDING) and (not retry or audio_key not in _PENDING or len(_PENDING) > 1)
            state = _PENDING.pop(audio_key, None) if retry else None
            _PENDING.clear()
        if dropped:
            print("WARN: stage timings: dropped pending session", file=sys.stderr, flush=True)
        context = session(pass_label, audio_key, state=state, attempt=2 if retry else 1)
    except Exception:
        return fn(*args, **kwargs)
    try:
        active = context.__enter__()
    except Exception:
        return fn(*args, **kwargs)
    try:
        value = fn(*args, **kwargs)
    except BaseException as exc:
        pending = exc.__class__.__name__ == "SegmentSkeletonRollback" and not retry
        if pending:
            try:
                active["pending"] = True
                with _LOCK:
                    _PENDING[audio_key] = active
            except Exception:
                pass
        elif not no_summary:
            _print_summary(active, f"error:{type(exc).__name__}")
        try:
            context.__exit__(*sys.exc_info())
        except Exception:
            pass
        raise
    else:
        if not no_summary:
            _print_summary(active, "ok")
        try:
            context.__exit__(None, None, None)
        except Exception:
            pass
        return value


def _install_special(name, label, fn, role):
    if role == "song":
        wrapper = functools.wraps(fn)(lambda *a, **kw: _song_wrapper(fn, a, kw, label))
        wrapper.__stage_timing_wrapper__ = True
        wrapper.__stage_timing_label__ = label
        return wrapper
    if name == "_publish_lrc_report":
        @functools.wraps(fn)
        def publish_wrapper(*args, **kwargs):
            if _ACTIVE is not None and _enabled():
                try:
                    report = args[3] if len(args) > 3 else kwargs.get("report")
                    payload = current_payload()
                    if isinstance(report, dict) and payload is not None and "stage_timings" not in report:
                        items = list(report.items())
                        report.clear()
                        for key, value in items:
                            if key == "published_lrc_sha256":
                                report["stage_timings"] = payload
                            report[key] = value
                        if "stage_timings" not in report:
                            report["stage_timings"] = payload
                except Exception:
                    pass
            return _call(fn, args, kwargs, label, function_name=name)
        publish_wrapper.__stage_timing_wrapper__ = True
        return publish_wrapper
    if name == "write_alignment_report":
        @functools.wraps(fn)
        def report_wrapper(*args, **kwargs):
            if _ACTIVE is not None and _enabled():
                try:
                    payload = current_payload()
                    if payload is not None:
                        payload.clear()
                        payload.update(_state_payload(_ACTIVE))
                except Exception:
                    pass
            return _call(fn, args, kwargs, label, function_name=name)
        report_wrapper.__stage_timing_wrapper__ = True
        return report_wrapper
    cache_kind = "read" if name == "_model_cache_read" else "write" if name == "_model_cache_write" else None
    return timed(label, fn, cache_kind=cache_kind, function_name=name)


def install(namespace, spec):
    global _PROGRESS_START
    try:
        _PROGRESS_START = namespace.get("PROGRESS_START")
    except Exception:
        _PROGRESS_START = None
    _INSTALL["wrapped"] = []
    _INSTALL["missing"] = []
    for row in spec:
        try:
            name, label, role = row[:3]
            fn = namespace.get(name)
            if fn is None:
                _INSTALL["missing"].append(name)
                continue
            if getattr(fn, "__stage_timing_wrapper__", False):
                _INSTALL["wrapped"].append(name)
                continue
            namespace[name] = _install_special(name, label, fn, role)
            _INSTALL["wrapped"].append(name)
        except Exception:
            try:
                _INSTALL["missing"].append(row[0])
            except Exception:
                pass


def install_report():
    return {"wrapped": list(_INSTALL["wrapped"]), "missing": list(_INSTALL["missing"])}
