from __future__ import annotations

import unittest

import auto_lrc


class WhisperLineageTests(unittest.TestCase):
    def test_whispercpp_current_is_whisper_family(self) -> None:
        self.assertEqual(
            auto_lrc._lineage_families_from_labels(
                "whispercpp-current",
                source="whispercpp-current",
            ),
            frozenset({"whisper-family"}),
        )


if __name__ == "__main__":
    unittest.main()
