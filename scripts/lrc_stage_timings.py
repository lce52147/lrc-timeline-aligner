"""Failure transparent wall-clock instrumentation for LRC production stages."""

import contextlib
import functools
import json
import os
import sys
import threading
import time


_CLOCK = time.perf_counter
_MAIN_THREAD = threading.main_thread().ident
_LOCK = threading.RLock()
_ACTIVE = None
_TLS = threading.local()
_PENDING = {}
_INSTALL = {"wrapped": [], "missing": []}


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
        "total_seconds": total,
        "unaccounted_seconds": _round(max(0.0, total - top)),
        "stages": stages,
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
        state = {
            "pass": str(pass_label),
            "audio_key": str(audio_key),
            "t0": _CLOCK(),
            "attempts": int(attempt),
            "stages": [],
            "worker_stages": {},
            "model_cache": {},
            "vocal_stem": [],
            "payload": None,
            "pending": False,
        }
    else:
        state["attempts"] = int(attempt)
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


def _call(fn, args, kwargs, name, *, cache_kind=None, role=None):
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
    if threading.get_ident() != _MAIN_THREAD:
        started = _CLOCK()
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
            "status": "ok",
        }
        stack.append(row)
        _TLS.stack = stack
    except Exception:
        row = None
        stack = []
    started = _CLOCK()
    try:
        result = fn(*args, **kwargs)
    except BaseException as exc:
        if row is not None:
            try:
                row["status"] = f"error:{type(exc).__name__}"
            except Exception:
                pass
        raise
    finally:
        if row is not None:
            try:
                row["seconds"] = _round(_CLOCK() - started)
                stack.pop()
                _append_stage(state, row)
            except Exception:
                pass
    if name == "prepare_vocal_ctc_audio":
        try:
            status = result[2] if isinstance(result, tuple) and len(result) >= 3 else None
            if status:
                state["vocal_stem"].append(str(status)[:80])
        except Exception:
            pass
    return result


def timed(name, fn, *, cache_kind=None, role=None):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return _call(fn, args, kwargs, name, cache_kind=cache_kind, role=role)
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
        detail = " | ".join(f"{row['name']} {row['seconds']:.6f}s" for row in shown)
        other = sum(float(row.get("seconds", 0.0)) for row in top[6:])
        cache = payload["model_cache"]
        hit = sum(int(row.get("hit", 0)) for row in cache.values())
        miss = sum(int(row.get("miss", 0)) for row in cache.values())
        fill = sum(int(row.get("fill", 0)) for row in cache.values())
        stem = payload["vocal_stem"][-1] if payload["vocal_stem"] else "n/a"
        suffix = (detail + " | ") if detail else ""
        print(f"TIMINGS pass={payload['pass']} status={status} total={payload['total_seconds']:.6f}s | {suffix}other {other:.6f}s | cache hit={hit} miss={miss} fill={fill} | stem={stem}", flush=True)
    except Exception:
        pass


def _song_wrapper(fn, args, kwargs, name):
    arg_object = args[1] if len(args) > 1 else kwargs.get("args")
    pass_label = _args_value(arg_object, "stage_timings_pass", "standalone")
    state = _PENDING.pop(_audio_key(args, kwargs), None)
    retry = state is not None and bool(_args_value(arg_object, "option_a_rollback_reason"))
    no_summary = bool(_args_value(args[1] if len(args) > 1 else kwargs.get("args"), "no_review_summary", False))
    holder = {"value": None, "error": None, "pending": False}
    context = session(pass_label, _audio_key(args, kwargs), state=state, attempt=2 if retry else 1)
    try:
        with context as active:
            try:
                holder["value"] = _call(fn, args, kwargs, name)
            except BaseException as exc:
                holder["error"] = exc
                if exc.__class__.__name__ == "SegmentSkeletonRollback" and not retry:
                    active["pending"] = True
                    _PENDING[_audio_key(args, kwargs)] = active
                    holder["pending"] = True
            if not holder["pending"]:
                status = "ok" if holder["error"] is None else f"error:{type(holder['error']).__name__}"
                if not no_summary:
                    _print_summary(active, status)
    finally:
        if holder["error"] is not None:
            raise holder["error"]
    return holder["value"]


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
            return _call(fn, args, kwargs, label)
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
            return _call(fn, args, kwargs, label)
        report_wrapper.__stage_timing_wrapper__ = True
        return report_wrapper
    return timed(label, fn, cache_kind=role if role in {"read", "write"} else None)


def install(namespace, spec):
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
