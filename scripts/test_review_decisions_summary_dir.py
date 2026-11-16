"""review_decisions_summary reads LRC_REPORT_DIR when no --report-dir is given."""
from __future__ import annotations

import contextlib
import io
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

import review_decisions_summary


class DecisionsSummaryReportDirTests(unittest.TestCase):
    def _summary(self, argv, env):
        out = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False), contextlib.redirect_stdout(out):
            self.assertEqual(review_decisions_summary.main(argv), 0)
        return out.getvalue()

    def test_default_follows_lrc_report_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = {"decision": "keep", "entry": 1, "review": "flagged", "reason_codes": ["x"]}
            (Path(tmp) / "s--1.review-decisions.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            summary = json.loads(self._summary(["--json"], {"LRC_REPORT_DIR": tmp}))
            self.assertEqual(summary["decisions"], 1)

    def test_explicit_report_dir_wins_over_the_environment(self):
        with tempfile.TemporaryDirectory() as env_dir, tempfile.TemporaryDirectory() as arg_dir:
            record = {"decision": "change", "entry": 2, "review": "flagged", "reason_codes": ["y"]}
            (Path(arg_dir) / "s--1.review-decisions.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            summary = json.loads(self._summary(["--json", "--report-dir", arg_dir], {"LRC_REPORT_DIR": env_dir}))
            self.assertEqual(summary["decisions"], 1)

    def test_without_the_variable_the_runtime_default_is_used(self):
        env = {k: v for k, v in os.environ.items() if k != "LRC_REPORT_DIR"}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                review_decisions_summary, "load_decisions", return_value=[]) as load:
            with contextlib.redirect_stdout(io.StringIO()):
                review_decisions_summary.main([])
        self.assertEqual(Path(load.call_args.args[0]).parts[-2:], ("outputs", "reports"))


if __name__ == "__main__":
    unittest.main()
