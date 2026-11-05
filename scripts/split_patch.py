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


def assign(name, value):
    """Direct ``auto_lrc.name = value`` equivalent for every module binding name."""
    bound = False
    for module_name in SPLIT_MODULES:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, name):
            setattr(module, name, value)
            bound = True
    if not bound:
        raise AttributeError(f"no split module binds {name!r}")
