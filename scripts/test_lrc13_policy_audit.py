"""Regression contracts for reachable N2 timing policy paths."""
from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path
import unittest

import auto_lrc


class TimingPolicyRegressionTests(unittest.TestCase):
    def test_ctc_first_is_default_but_legacy_still_explicit(self) -> None:
        parser = auto_lrc.build_parser()
        self.assertEqual(
            parser.parse_args(["song.flac"]).timing_source_policy, "ctc-first"
        )
        self.assertEqual(
            parser.parse_args(["--timing-source-policy", "default", "song.flac"]).timing_source_policy,
            "default",
        )

    def test_evaluation_retains_actual_policy(self) -> None:
        self.assertIn(
            "timing_source_policy",
            {field.name for field in fields(auto_lrc.EvaluatedBackendTiming)},
        )

    def test_all_reachable_private_replays_inherit_selected_policy(self) -> None:
        tree = ast.parse(Path(auto_lrc.__file__).read_text(encoding="utf-8"))
        replay_calls = [
            call for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "evaluate_backend_timing"
            and (17000 <= call.lineno <= 18000 or 32000 <= call.lineno <= 34000)
        ]
        self.assertEqual(len(replay_calls), 5, "New replay calls must be reviewed")
        for call in replay_calls:
            with self.subTest(line=call.lineno):
                value = next(
                    (kw.value for kw in call.keywords if kw.arg == "timing_source_policy"),
                    None,
                )
                self.assertIsNotNone(value, "Private replay may silently use legacy policy")
                self.assertIsInstance(value, ast.Attribute)
                self.assertEqual(value.attr, "timing_source_policy")
                self.assertIsInstance(value.value, ast.Name)
                self.assertEqual(value.value.id, "selected_eval")


if __name__ == "__main__":
    unittest.main()
