"""Patch helper for tests while auto_lrc.py is split into owner modules.

A name moved out of auto_lrc is bound in its owner module, in the auto_lrc
facade and in every module that imported it.  patch() replaces it in each of
those bindings with one shared replacement, so callers behave as they did when
all of them shared one namespace.
"""
from __future__ import annotations

import contextlib
import sys
from unittest import mock

import runtime_base

SPLIT_MODULES = ("auto_lrc",) + runtime_base._ALGORITHM_SOURCE_MODULES


class _Patcher:
    def __init__(self, name, new, kwargs):
        self._name = name
        self._new = mock.MagicMock(**kwargs) if new is mock.DEFAULT else new
        self._stack = None

    def start(self):
        stack = contextlib.ExitStack()
        bound = False
        for module_name in SPLIT_MODULES:
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, self._name):
                stack.enter_context(mock.patch.object(module, self._name, self._new))
                bound = True
        if not bound:
            raise AttributeError(f"no split module binds {self._name!r}")
        self._stack = stack
        return self._new

    def stop(self):
        if self._stack is not None:
            self._stack.close()
            self._stack = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def patch(name, new=mock.DEFAULT, **kwargs):
    return _Patcher(name, new, kwargs)


# Active assign() calls: name -> list of {module_name: previous binding}.  An entry
# is dropped by its restore callable, or by assigning the name back to the
# original objects; assert_clean() fails while any entry is left over.
_ACTIVE: dict[str, list[dict[str, object]]] = {}


def assign(name, value):
    """Direct ``auto_lrc.name = value`` equivalent for every module binding name.

    Returns a callable that puts every binding back.  Restoring one binding (for
    example only ``auto_lrc.name``) leaves the owner modules on the stub, and
    assert_clean() reports it.
    """
    previous = {}
    for module_name in SPLIT_MODULES:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, name):
            previous[module_name] = getattr(module, name)
    if not previous:
        raise AttributeError(f"no split module binds {name!r}")
    entries = _ACTIVE.setdefault(name, [])
    for index in range(len(entries) - 1, -1, -1):
        if all(value is old for old in entries[index].values()):
            del entries[index:]  # assigning the originals back is a restore
            break
    else:
        entries.append(previous)
    if not entries:
        del _ACTIVE[name]
    for module_name in previous:
        setattr(sys.modules[module_name], name, value)

    def restore():
        for module_name, old in previous.items():
            setattr(sys.modules[module_name], name, old)
        remaining = _ACTIVE.get(name, [])
        if previous in remaining:
            remaining.remove(previous)
        if not remaining:
            _ACTIVE.pop(name, None)

    return restore


def assert_clean():
    """Raise if an assign() was never undone through its restore callable."""
    if _ACTIVE:
        names = sorted(_ACTIVE)
        _ACTIVE.clear()
        raise AssertionError(f"split_patch.assign never restored for: {', '.join(names)}")
