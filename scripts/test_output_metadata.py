import unittest

import auto_lrc


class OutputMetadataTests(unittest.TestCase):
    SOURCE = ["[ar:Artist]", "[al:Album]", "[ti:Title]", "[offset:+500]", "[length:03:10.00]", "[re:OtherTool]", "[by:someone]"]

    def test_retimed_output_drops_source_clock_and_generator_tags(self) -> None:
        for timing_source in ("ctc", "backend_competition", "auto", "whisperx", "heuristic"):
            kept = auto_lrc.output_metadata_lines(self.SOURCE, timing_source)
            self.assertEqual(kept, ["[ar:Artist]", "[al:Album]", "[ti:Title]"], timing_source)

    def test_source_timed_output_round_trips_every_tag(self) -> None:
        for timing_source in ("lyrics", "checked"):
            self.assertEqual(auto_lrc.output_metadata_lines(self.SOURCE, timing_source), self.SOURCE)

    def test_empty_metadata(self) -> None:
        self.assertEqual(auto_lrc.output_metadata_lines(None, "ctc"), [])


if __name__ == "__main__":
    unittest.main()
