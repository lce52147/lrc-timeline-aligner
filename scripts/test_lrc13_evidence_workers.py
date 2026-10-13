from __future__ import annotations

import os
import unittest
from unittest import mock

import auto_lrc


class EvidenceWorkerCountTests(unittest.TestCase):
    def count(self, value: str | None) -> int:
        env = {k: v for k, v in os.environ.items() if k != "LRC_EVIDENCE_WORKERS"}
        if value is not None:
            env["LRC_EVIDENCE_WORKERS"] = value
        with mock.patch.dict(os.environ, env, clear=True):
            return auto_lrc._evidence_worker_count()

    def test_default_is_one_resident_copy_per_model_family(self) -> None:
        self.assertEqual(self.count(None), 1)
        self.assertEqual(self.count(""), 1)

    def test_environment_opts_into_more_workers_within_bounds(self) -> None:
        self.assertEqual(self.count("2"), 2)
        self.assertEqual(self.count("9"), 4)
        self.assertEqual(self.count("0"), 1)
        self.assertEqual(self.count("two"), 1)

    def test_helper_pool_follows_the_worker_count(self) -> None:
        pool = auto_lrc._PersistentHelperPool(
            auto_lrc.Path("python"), auto_lrc.Path("helper.py"), "ctc",
            auto_lrc.Path("worker.py"), auto_lrc.DEFAULT_EVIDENCE_WORKERS,
        )
        self.assertEqual(pool.size, auto_lrc.DEFAULT_EVIDENCE_WORKERS)


if __name__ == "__main__":
    unittest.main()
