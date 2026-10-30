from __future__ import annotations

import os
import unittest
from unittest import mock

import auto_lrc
from auto_lrc import apply_line_start_lead, canonicalize_lrc_time, resolve_start_lead_ms


def _report(starts_s, ends_s=None):
    rows = []
    for index, start in enumerate(starts_s, 1):
        cw = canonicalize_lrc_time(start)
        row = {
            "entry": index,
            "chosen_time": cw.seconds,
            "timestamp": cw.seconds,
            "canonical_written_centiseconds": cw.centiseconds,
            "lrc_tag": cw.lrc_tag,
            "candidate_timestamps": {"output": cw.seconds},
            "candidate_provenance": {"canonical_written_centiseconds": cw.centiseconds},
        }
        if ends_s and ends_s[index - 1] is not None:
            row["ctc_token_spans"] = [{"char": "a", "start": start, "end": ends_s[index - 1], "score": 1.0}]
        rows.append(row)
    return {
        "assignments": rows,
        "central_timing_state": [
            {"entry": r["entry"], "timestamp": r["timestamp"],
             "canonical_written_centiseconds": r["canonical_written_centiseconds"]}
            for r in rows
        ],
    }


def _apply(starts, ends=None, lead=100):
    report = _report(starts, ends)
    times = tuple(canonicalize_lrc_time(s) for s in starts)
    return report, apply_line_start_lead(report, times, lead)


class StartLeadTests(unittest.TestCase):
    def test_plain_lead_moves_every_line_100ms(self):
        report, out = _apply([10.0, 20.0, 30.0], [12.0, 22.0, 32.0])
        self.assertEqual([t.centiseconds for t in out], [990, 1990, 2990])
        self.assertEqual([r["start_lead_ms"] for r in report["assignments"]], [100, 100, 100])

    def test_clamps_at_previous_line_end(self):
        report, out = _apply([10.0, 12.05], [12.0, 14.0])
        self.assertEqual(out[1].centiseconds, 1205 - 5)
        self.assertEqual(report["assignments"][1]["start_lead_ms"], 50)

    def test_no_move_when_previous_end_after_start(self):
        _report_, out = _apply([10.0, 12.0], [10.5, 14.0])
        self.assertEqual(out[1].centiseconds, 1190)
        _report_, out = _apply([10.0, 11.0], [11.5, 14.0])
        self.assertEqual(out[1].centiseconds, 1100)

    def test_without_end_uses_previous_tag_plus_one_cs(self):
        report, out = _apply([10.0, 10.0])
        self.assertEqual(out[0].centiseconds, 990)
        self.assertEqual(out[1].centiseconds, 990 + 1)

    def test_zero_floor(self):
        report, out = _apply([0.04, 5.0], [1.0, 6.0])
        self.assertEqual(out[0].centiseconds, 0)
        self.assertEqual(report["assignments"][0]["start_lead_ms"], 40)

    def test_report_matches_lrc_tag_everywhere(self):
        report, out = _apply([10.0, 20.0], [12.0, 22.0])
        for index, written in enumerate(out):
            row = report["assignments"][index]
            self.assertEqual(row["lrc_tag"], written.lrc_tag)
            self.assertEqual(row["canonical_written_centiseconds"], written.centiseconds)
            self.assertEqual(row["chosen_time"], written.seconds)
            self.assertEqual(row["candidate_timestamps"]["output"], written.seconds)
            self.assertEqual(row["candidate_provenance"]["canonical_written_centiseconds"], written.centiseconds)
            central = report["central_timing_state"][index]
            self.assertEqual(central["canonical_written_centiseconds"], written.centiseconds)
            self.assertEqual(central["timestamp"], written.seconds)
        self.assertTrue(report["start_lead_applied"])

    def test_lead_zero_is_identity(self):
        report, out = _apply([10.0, 20.0], [12.0, 22.0], lead=0)
        self.assertEqual([t.centiseconds for t in out], [1000, 2000])
        self.assertFalse(report["start_lead_applied"])

    def test_resolve_cli_env_default(self):
        with mock.patch.dict(os.environ, {"LRC_START_LEAD_MS": "60"}):
            self.assertEqual(resolve_start_lead_ms(None), 60)
            self.assertEqual(resolve_start_lead_ms(30), 30)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LRC_START_LEAD_MS", None)
            self.assertEqual(resolve_start_lead_ms(None), 100)
            self.assertEqual(resolve_start_lead_ms(0), 0)


class ReviewCandidateLeadTests(unittest.TestCase):
    def test_candidates_shift_with_the_applied_lead(self):
        import review_server

        assignment = {
            "start_lead_ms": 100,
            "candidates": [{"time": 10.0, "source": "ctc-current", "score": 0.9}],
        }
        # written tag is 9.90 (10.00 - lead): the candidate must read 9.90 too, i.e. match "current".
        candidates, _extra = review_server._candidate_rows(assignment, 990, 990)
        self.assertEqual({row["cs"] for row in candidates}, {990})
        legacy = {"candidates": [{"time": 10.0, "source": "ctc-current", "score": 0.9}]}
        candidates, _extra = review_server._candidate_rows(legacy, 1000, 1000)
        self.assertEqual({row["cs"] for row in candidates}, {1000})


if __name__ == "__main__":
    unittest.main()
