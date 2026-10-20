"""Tests for the r2 reviewer-sidecar cache (LRC 1.5 W3b)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import model_cache  # noqa: E402
import r2_pipeline  # noqa: E402
from test_r2_pipeline import R2PipelineTests  # noqa: E402

_CLEAN_ENV = {"LRC_MODEL_CACHE": "1", "LRC_MODEL_CACHE_ONLY": "0"}


class ReviewerSidecarCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache_root = self.root / "cache"
        self.lyrics = self.root / "song.lyrics.txt"
        self.vocal = self.root / "vocals.wav"
        self.lyrics.write_text("\n".join(f"歌詞{n}" for n in range(1, 9)) + "\n", encoding="utf-8")
        self.vocal.write_bytes(b"fake audio")
        self.report_path = self.root / "baseline.json"
        self._write_report()
        env = mock.patch.dict(os.environ, _CLEAN_ENV)
        env.start()
        self.addCleanup(env.stop)

    def _write_report(self, lyric_text: str | None = None) -> None:
        if lyric_text is not None:
            self.lyrics.write_text(lyric_text, encoding="utf-8")
        report = R2PipelineTests()._baseline_report(self.lyrics, self.vocal)
        self.report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")

    def _acquire(self, policy: model_cache.ModelCachePolicy, run_code: int = 0, work: str = "work"):
        calls: list[list[str]] = []

        def fake_run(command, log):
            calls.append(list(command))
            return run_code

        with mock.patch.object(r2_pipeline, "_run_logged", side_effect=fake_run):
            summary = r2_pipeline.acquire_reviewer_evidence(
                self.report_path, self.root / f"{work}-evidence.json", self.root / work,
                cache_policy=policy, cache_root=self.cache_root,
            )
        return summary, calls

    def test_identity_has_no_paths(self) -> None:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        identity = r2_pipeline.reviewer_sidecar_identity(
            r2_pipeline.build_reviewer_payload(report), device="cuda", include_hub=False,
        )
        model_cache.assert_portable_identity(identity)
        self.assertEqual(set(identity), {
            "payload_sha256", "reviewers_sha256", "whisper_cli_sha256",
            "whisper_model_sha256", "device", "include_hub",
        })

    def test_fill_then_hit_without_running_reviewers(self) -> None:
        rw = model_cache.ModelCachePolicy(enabled=True, read_only=False)
        first, calls = self._acquire(rw)
        self.assertTrue(calls, "reviewers must run on a miss")
        self.assertEqual(first["cache"]["status"], "written")
        key = first["cache"]["key"]
        self.assertTrue(model_cache.cache_path(r2_pipeline.REVIEWER_SIDECAR_CACHE_KIND, key, self.cache_root).is_file())

        second, calls2 = self._acquire(rw, work="replay")
        self.assertEqual(calls2, [], "a hit must not invoke _run_logged")
        self.assertEqual(second["cache"]["status"], "hit")
        self.assertEqual(second["qualified_reviewers"], first["qualified_reviewers"])
        self.assertTrue(all(state["source"] == "cache" for state in second["statuses"].values()))
        first_sidecar = (self.root / "work-evidence.json").read_bytes()
        second_sidecar = (self.root / "replay-evidence.json").read_bytes()
        self.assertEqual(first_sidecar, second_sidecar)
        # Stage 4 of the pipeline re-reads reviewer-data.json from the work dir.
        self.assertEqual(
            (self.root / "work" / "reviewer-data.json").read_bytes(),
            (self.root / "replay" / "reviewer-data.json").read_bytes(),
        )

        third, calls3 = self._acquire(model_cache.ModelCachePolicy(enabled=True, read_only=True), work="cacheonly")
        self.assertEqual(calls3, [])
        self.assertEqual(third["cache"]["status"], "hit")

    def test_cache_only_miss_raises(self) -> None:
        cache_only = model_cache.ModelCachePolicy(enabled=True, read_only=True)
        with mock.patch.object(r2_pipeline, "_run_logged", side_effect=AssertionError("launched")):
            with self.assertRaises(model_cache.ModelCacheMiss):
                r2_pipeline.acquire_reviewer_evidence(
                    self.report_path, self.root / "evidence.json", self.root / "work",
                    cache_policy=cache_only, cache_root=self.cache_root,
                )

    def test_failed_reviewer_is_not_cached(self) -> None:
        rw = model_cache.ModelCachePolicy(enabled=True, read_only=False)
        summary, calls = self._acquire(rw, run_code=1)
        self.assertTrue(calls)
        self.assertEqual(summary["cache"]["status"], "not-written")
        self.assertFalse(self.cache_root.exists())
        with self.assertRaises(model_cache.ModelCacheMiss):
            self._acquire(model_cache.ModelCachePolicy(enabled=True, read_only=True), work="again")

    def test_lyric_change_changes_key(self) -> None:
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        payload = r2_pipeline.build_reviewer_payload(report)
        key_a = model_cache.key_from_identity(
            r2_pipeline.REVIEWER_SIDECAR_CACHE_KIND,
            r2_pipeline.reviewer_sidecar_identity(payload, device="cuda", include_hub=False),
        )
        self._write_report("\n".join(f"歌詞{n}" for n in range(1, 8)) + "\n歌詞X\n")
        payload_b = r2_pipeline.build_reviewer_payload(json.loads(self.report_path.read_text(encoding="utf-8")))
        key_b = model_cache.key_from_identity(
            r2_pipeline.REVIEWER_SIDECAR_CACHE_KIND,
            r2_pipeline.reviewer_sidecar_identity(payload_b, device="cuda", include_hub=False),
        )
        self.assertNotEqual(key_a, key_b)
        key_hub = model_cache.key_from_identity(
            r2_pipeline.REVIEWER_SIDECAR_CACHE_KIND,
            r2_pipeline.reviewer_sidecar_identity(payload, device="cuda", include_hub=True),
        )
        self.assertNotEqual(key_a, key_hub)

    def test_disabled_policy_never_touches_cache(self) -> None:
        summary, calls = self._acquire(model_cache.ModelCachePolicy(enabled=False))
        self.assertTrue(calls)
        self.assertEqual(summary["cache"]["status"], "disabled")
        self.assertFalse(self.cache_root.exists())

    def test_legacy_payload_without_reviewer_data_is_a_miss(self) -> None:
        rw = model_cache.ModelCachePolicy(enabled=True, read_only=False)
        first, _ = self._acquire(rw)
        key = first["cache"]["key"]
        path = model_cache.cache_path(r2_pipeline.REVIEWER_SIDECAR_CACHE_KIND, key, self.cache_root)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["payload"].pop("reviewer_data")
        envelope["payload"].pop("hubp_raw", None)
        path.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True), encoding="utf-8")

        with self.assertRaises(model_cache.ModelCacheMiss):
            self._acquire(model_cache.ModelCachePolicy(enabled=True, read_only=True), work="cacheonly")
        second, calls = self._acquire(rw, work="refill")
        self.assertTrue(calls, "a legacy entry must be recomputed")
        self.assertEqual(second["cache"]["status"], "written")
        self.assertTrue((self.root / "refill" / "reviewer-data.json").is_file())


if __name__ == "__main__":
    unittest.main()
