import inspect
import unittest

import auto_lrc


class ProducerContractTests(unittest.TestCase):
    def test_phonetic_fusion_serializes_supported_direct_onset(self) -> None:
        source = inspect.getsource(auto_lrc.apply_phonetic_onset_fusion_hypotheses)
        self.assertIn('direct_onset_support="supported"', source)
        self.assertNotIn('direct_onset_support="unavailable"', source)

    def test_trusted_tail_release_is_diagnostic_only_not_central_candidate(self) -> None:
        source = inspect.getsource(auto_lrc.collect_central_timing_candidates)
        branch = source.split('elif source == "trusted-tail-release-onset":', 1)[1]
        branch = branch.split('spans_value =', 1)[0]
        self.assertIn("trusted-tail-release-diagnostic-only", branch)
        self.assertIn("continue", branch)


if __name__ == "__main__":
    unittest.main()
