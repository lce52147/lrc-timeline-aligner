import argparse
import unittest
from pathlib import Path

import auto_lrc


class CentralZipContractTests(unittest.TestCase):
    def test_summary_rejects_decision_audit_length_mismatch(self):
        state = auto_lrc.FinalTimingState(
            decisions=(),
            written_times=(),
            audits=(auto_lrc.TimingInvariantAudit(0, False, "provisional_unresolved", ()),),
        )
        with self.assertRaisesRegex(auto_lrc.LrcError, "decision"):
            auto_lrc._central_summary_row_flags({"assignments": [{}]}, state)

    def test_candidate_collector_rejects_short_scratch_clock_as_contract_error(self):
        entries = [auto_lrc.LyricEntry(["first lyric"])]
        report = {"assignments": [{"timestamp": 1.0}]}
        with self.assertRaisesRegex(auto_lrc.LrcError, "scratch"):
            auto_lrc.collect_central_timing_candidates(
                Path("unused.wav"),
                entries,
                [],
                report,
                2.0,
                argparse.Namespace(),
            )


if __name__ == "__main__":
    unittest.main()
