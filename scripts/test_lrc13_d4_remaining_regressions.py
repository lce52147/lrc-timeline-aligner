from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import auto_lrc
from test_core_logic import V144JointBoundaryGraphRegressionTests


class HybridWindowBoundaryRegressionTests(unittest.TestCase):
    def test_short_right_boundary_skips_window_instead_of_crossing_next_line(self) -> None:
        entries = [auto_lrc.LyricEntry([f"line {index}"]) for index in range(4)]
        timestamps = [10.0, 10.2, 10.4, 10.7]
        ctc_report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "ctc_token_spans": [{"score": 0.9}]},
                {"entry": 2, "timestamp": 10.2, "ctc_token_spans": [{"score": 0.9}]},
                {"entry": 3, "timestamp": 10.4, "ctc_token_spans": [{"score": 0.9}]},
                {"entry": 4, "timestamp": 10.7, "ctc_token_spans": [{"score": 0.9}]},
            ]
        }
        whisperx_report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0},
                {"entry": 2, "timestamp": 8.0},
                {"entry": 3, "timestamp": 10.4},
                {"entry": 4, "timestamp": 10.7},
            ]
        }
        args = SimpleNamespace(whisperx_device="cuda")
        with mock.patch.object(auto_lrc, "_run_model_helper_command") as helper:
            helper.return_value = subprocess.CompletedProcess([], 1, "", "")
            refined, _report, changes = auto_lrc.apply_ctc_hybrid_window_realign(
                Path("dummy.wav"),
                entries,
                list(timestamps),
                ctc_report,
                whisperx_report,
                20.0,
                args,
            )
        self.assertEqual(refined, timestamps)
        self.assertEqual(changes, [])
        helper.assert_not_called()


class WhisperxSpliceOrderRegressionTests(unittest.TestCase):
    def test_nonmonotonic_normal_sns_seam_uses_complete_better_candidate(self) -> None:
        normal_times = [1.0, 10.0, 20.0, 30.0]
        sns_times = [1.0, 5.0, 9.0, 12.0]
        normal_report = {
            "assignments": [
                {"entry": index + 1, "timestamp": value, "score": 0.9}
                for index, value in enumerate(normal_times)
            ]
        }
        sns_report = {
            "assignments": [
                {"entry": index + 1, "timestamp": value, "score": 0.9}
                for index, value in enumerate(sns_times)
            ]
        }

        def quality(report: dict[str, object]) -> float:
            return 75.0 if report is sns_report else 70.0

        args = SimpleNamespace(whisper_suppress_nst=None)
        with (
            mock.patch.object(
                auto_lrc,
                "try_whisperx_candidate",
                side_effect=[
                    (normal_times, normal_report, None),
                    (sns_times, sns_report, None),
                ],
            ),
            mock.patch.object(auto_lrc, "needs_suppressed_retry", return_value=True),
            mock.patch.object(auto_lrc, "low_confidence_run_start", return_value=2),
            mock.patch.object(auto_lrc, "report_quality", side_effect=quality),
        ):
            timestamps, report = auto_lrc.run_whisperx_best_candidate(
                Path("dummy.flac"), [], 40.0, args
            )

        self.assertEqual(timestamps, sns_times)
        self.assertEqual(report["assignments"], sns_report["assignments"])


class CrosslineTransientCacheRegressionTests(unittest.TestCase):
    def test_timeout_is_not_persisted_in_helper_cache(self) -> None:
        fixture = V144JointBoundaryGraphRegressionTests()
        entries, _candidate_sets, _report, capability, _epoch, request = fixture.joint_request()
        helper_cache: dict[str, tuple[str, object]] = {}

        with mock.patch.object(
            auto_lrc,
            "_run_model_helper_command",
            side_effect=subprocess.TimeoutExpired(cmd=["ctc"], timeout=0.01),
        ):
            outcome = auto_lrc.invoke_crossline_retry(
                request,
                capability,
                entries,
                helper_cache=helper_cache,
                timeout_seconds=0.01,
            )

        self.assertEqual(outcome.status, "error")
        self.assertEqual(outcome.reason, "helper-timeout")
        self.assertEqual(helper_cache, {})

    def test_successful_dict_payload_is_cached_without_hashing_payload(self) -> None:
        # Regression: the first R3-6 fix tested ``result in {tuples}``, which
        # hashes the payload dict and crashed every real crossline retry.
        fixture = V144JointBoundaryGraphRegressionTests()
        entries, _candidate_sets, _report, capability, _epoch, request = fixture.joint_request()
        helper_cache: dict[str, tuple[str, object]] = {}

        def fake_helper(command, **_kwargs):
            output = Path(command[command.index("--output") + 1])
            output.write_text('{"entries": []}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(auto_lrc, "_run_model_helper_command", side_effect=fake_helper):
            outcome = auto_lrc.invoke_crossline_retry(
                request,
                capability,
                entries,
                helper_cache=helper_cache,
                timeout_seconds=0.01,
            )

        self.assertEqual(outcome.status, "invalid")
        self.assertEqual(list(helper_cache.values()), [("payload", {"entries": []})])


if __name__ == "__main__":
    unittest.main()
