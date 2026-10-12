import inspect
import unittest

import auto_lrc
import quantify_alignment
import run_benchmarks


class DeadCliCleanupTests(unittest.TestCase):
    def test_noop_compatibility_flags_are_removed(self) -> None:
        parser = auto_lrc.build_parser()
        for option in (
            "--allow-untrusted-draft",
            "--ignore-lyric-timestamps",
            "--no-checked-lrc-hint",
            "--no-anchor-hints",
        ):
            self.assertNotIn(option, parser._option_string_actions)

    def test_anchor_hint_lookup_has_no_dead_negative_branch(self) -> None:
        source = inspect.getsource(auto_lrc.find_anchor_hints)
        self.assertNotIn("no_anchor_hints", source)

    def test_harnesses_do_not_emit_removed_flags(self) -> None:
        benchmark_source = inspect.getsource(run_benchmarks)
        quantify_source = inspect.getsource(quantify_alignment)
        for option in (
            "--ignore-lyric-timestamps",
            "--no-checked-lrc-hint",
            "--no-anchor-hints",
        ):
            self.assertNotIn(option, benchmark_source)
            self.assertNotIn(option, quantify_source)


if __name__ == "__main__":
    unittest.main()
