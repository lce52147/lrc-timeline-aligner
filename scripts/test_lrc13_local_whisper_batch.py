from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import auto_lrc


def request(entry: int, start: float) -> SimpleNamespace:
    return SimpleNamespace(
        entry_index=entry,
        window_start=start,
        window_end=start + 20.0,
        request_revision=f"window-{start:g}",
    )


class LocalWhisperBatchTests(unittest.TestCase):
    def run_epoch(self, requests: tuple[SimpleNamespace, ...]):
        prepared_calls: list[list[tuple[str, int, object]]] = []
        executed: list[tuple[int, object]] = []

        def fake_prepare(work, capability, args):  # type: ignore[no-untyped-def]
            prepared_calls.append(list(work))
            return {key: (["segments-for-" + key], {}) for key, _ordinal, _request in work}, {}

        def fake_execute(req, capability, entries, args, **kwargs):  # type: ignore[no-untyped-def]
            executed.append((req.entry_index, kwargs.get("prepared")))
            return SimpleNamespace(
                entry_index=req.entry_index,
                request_revision=req.request_revision,
                status="evidence",
                reason="fake",
            )

        stats = {"cache_hits": 0, "cache_misses": 0, "helper_invocations": 0}
        with mock.patch.object(auto_lrc, "DEFAULT_EVIDENCE_WORKERS", 1), \
                mock.patch.object(auto_lrc, "_prepare_local_whisper_batch", fake_prepare), \
                mock.patch.object(auto_lrc, "execute_local_whisper_request", fake_execute), \
                mock.patch.object(auto_lrc, "_local_whisper_cache_key",
                                  lambda req, cap: f"w{req.window_start:g}"):
            outcomes = auto_lrc.run_local_whisper_requests(
                requests, object(), [], object(), cache={}, stats=stats,
            )
        return outcomes, prepared_calls, executed, stats

    def test_one_worker_still_prepares_the_whole_epoch_in_one_batch(self) -> None:
        # Entries 1 and 3 ask for the same window, so it is aligned once.
        requests = (request(0, 0.0), request(1, 10.0), request(2, 0.0), request(3, 30.0))
        outcomes, prepared_calls, executed, stats = self.run_epoch(requests)

        self.assertEqual(len(prepared_calls), 1)
        self.assertEqual([key for key, _o, _r in prepared_calls[0]], ["w0", "w10", "w30"])
        self.assertEqual(
            executed,
            [(0, (["segments-for-w0"], {})), (1, (["segments-for-w10"], {})), (3, (["segments-for-w30"], {}))],
        )
        # Request order is preserved and the shared window's outcome is reused.
        self.assertEqual([item.entry_index for item in outcomes], [0, 1, 0, 3])
        self.assertEqual(stats["cache_misses"], 3)
        self.assertEqual(stats["cache_hits"], 1)

    def test_single_request_runs_directly(self) -> None:
        outcomes, prepared_calls, executed, _stats = self.run_epoch((request(5, 40.0),))
        self.assertEqual(prepared_calls, [])
        self.assertEqual(executed, [(5, None)])
        self.assertEqual([item.entry_index for item in outcomes], [5])


if __name__ == "__main__":
    unittest.main()
