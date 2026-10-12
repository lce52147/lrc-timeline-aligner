from __future__ import annotations

import unittest
from unittest.mock import patch

import auto_lrc


class CentralSourceValidationTests(unittest.TestCase):
    def test_source_validation_lrcerror_fails_closed(self) -> None:
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_state_schema_version": 3,
            "candidate_selection": {
                "status": "selected",
                "metric_stage": "central-evaluated",
                "selected_backend": "ctc",
            },
            "assignments": [{}],
            "central_timing_state": [{}],
        }
        with patch.object(
            auto_lrc,
            "validated_loaded_source_sha256",
            side_effect=auto_lrc.LrcError("source changed"),
        ):
            self.assertFalse(auto_lrc.report_has_verified_central_commit(report))


if __name__ == "__main__":
    unittest.main()
