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

SPLIT_MODULES = (
    "auto_lrc",
    "runtime_base",
    "audio_features",
    "lyrics_io",
    "timing_core",
    "report_trust",
    "ctc_refinement",
    "acoustic_hypotheses",
    "asr_backends",
    "local_whisper_retry",
    "central_peer_recovery",
    "neutral_arbitration",
    "central_commit",
    "backend_bundle",
    "postcentral_recovery",
    "runtime_evidence_binding",
)


class _Patcher:
    def __init__(self, name, new, kwargs):
        self._name = name
        self._new = mock.MagicMock(**kwargs) if new is mock.DEFAULT else new
        self._stack = None

    def start(self):
        stack = contextlib.ExitStack()
        for module_name in SPLIT_MODULES:
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, self._name):
                stack.enter_context(mock.patch.object(module, self._name, self._new))
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
    for module_name in SPLIT_MODULES:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, name):
            setattr(module, name, value)
