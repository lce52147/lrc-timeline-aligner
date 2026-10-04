from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from evaluate_lrc import main as evaluate_main
from evaluate_lrc import summarize
from quantify_alignment import DEFAULT_REFERENCES_DIR, PROJECT, MatchedCase, evaluate_row, write_markdown
from run_benchmarks import BenchmarkCase, evaluate_case


class EvaluateLrcTextAlignmentTests(unittest.TestCase):
    def summarize_text(self, reference: str, generated: str) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "reference.lrc"
            gen = root / "generated.lrc"
            ref.write_text(reference, encoding="utf-8")
            gen.write_text(generated, encoding="utf-8")
            return summarize(ref, gen)

    def test_extra_generated_line_does_not_shift_later_timing_pairs(self) -> None:
        result = self.summarize_text(
            "[00:01.00]A\n[00:02.00]B\n[00:03.00]C\n",
            "[00:01.01]A\n[00:01.50]EXTRA\n[00:02.02]B\n[00:03.03]C\n",
        )

        self.assertEqual(result["timing_compared_entries"], 3)
        self.assertEqual(result["unmatched_reference_count"], 0)
        self.assertEqual(result["unmatched_generated_count"], 1)
        self.assertEqual(result["unmatched_generated_entries"][0]["text"], "EXTRA")
        self.assertEqual(result["max_abs_delta_ms"], 30)

    def test_missing_generated_line_is_reported_without_shifting_later_pair(self) -> None:
        result = self.summarize_text(
            "[00:01.00]A\n[00:02.00]B\n[00:03.00]C\n",
            "[00:01.01]A\n[00:03.02]C\n",
        )

        self.assertEqual(result["timing_compared_entries"], 2)
        self.assertEqual(result["unmatched_reference_count"], 1)
        self.assertEqual(result["unmatched_reference_entries"][0]["text"], "B")
        self.assertEqual(result["unmatched_generated_count"], 0)
        self.assertEqual(result["max_abs_delta_ms"], 20)

    def test_marker_entries_are_excluded_from_alignment_and_denominator(self) -> None:
        result = self.summarize_text(
            "[00:01.00]A\n[00:01.50]♪\n[00:02.00]B\n",
            "[00:01.01]A\n[00:02.02]B\n",
        )

        self.assertEqual(result["reference_entries"], 2)
        self.assertEqual(result["generated_entries"], 2)
        self.assertEqual(result["timing_compared_entries"], 2)
        self.assertEqual(result["unmatched_reference_count"], 0)
        self.assertEqual(result["unmatched_generated_count"], 0)

    def test_bilingual_display_grouping_difference_keeps_sung_line_pairing(self) -> None:
        result = self.summarize_text(
            "[00:01.00]雨が降る\n[00:01.00]雨落下來\n[00:02.00]明日へ\n[00:02.00]前往明天\n",
            "[00:01.02]雨が降る\n[00:02.03]明日へ\n",
        )

        self.assertEqual(result["timing_compared_entries"], 2)
        self.assertEqual(result["unmatched_reference_count"], 0)
        self.assertEqual(result["unmatched_generated_count"], 0)
        self.assertEqual(result["max_abs_delta_ms"], 30)
        self.assertEqual(result["text_mismatches"], 2)


class EvaluateLrcMillisecondMetricsTests(unittest.TestCase):
    def test_reports_30_50_ms_tiers_with_matched_pair_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "reference.lrc"
            gen = root / "generated.lrc"
            ref.write_text(
                "[00:01.00]A\n[00:02.00]B\n[00:03.00]C\n[00:04.00]D\n",
                encoding="utf-8",
            )
            gen.write_text(
                "[00:01.03]A\n[00:02.04]B\n[00:03.05]C\n[00:04.06]D\n",
                encoding="utf-8",
            )

            result = summarize(ref, gen)

        self.assertEqual(result["timing_compared_entries"], 4)
        self.assertEqual(result["correct_le_30ms"], 1)
        self.assertEqual(result["acceptable_30_to_50ms"], 2)
        self.assertEqual(result["wrong_gt_50ms"], 1)
        self.assertEqual(result["correct_le_30ms_percent"], 25.0)
        self.assertEqual(result["acceptable_30_to_50ms_percent"], 50.0)
        self.assertEqual(result["wrong_gt_50ms_percent"], 25.0)
        self.assertEqual(result["median_abs_delta_ms"], 45.0)
        self.assertEqual(result["mae_ms"], 45.0)
        self.assertEqual(result["max_abs_delta_ms"], 60)
        self.assertIn("100 ms", result["legacy_timing_metrics_note"])

    def test_cli_primary_output_uses_current_30_50ms_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "reference.lrc"
            gen = root / "generated.lrc"
            ref.write_text("[00:01.00]A\n[00:02.00]B\n", encoding="utf-8")
            gen.write_text("[00:01.03]A\n[00:02.06]B\n", encoding="utf-8")
            output = StringIO()
            with redirect_stdout(output):
                exit_code = evaluate_main([str(ref), str(gen)])

        text = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("correct <=30 ms: 50.0%", text)
        self.assertIn("acceptable 30-50 ms: 0.0%", text)
        self.assertIn("wrong >50 ms: 50.0%", text)
        self.assertIn("median abs delta: 45.0 ms", text)
        self.assertIn("MAE: 45.0 ms", text)
        self.assertIn("max abs delta: 60 ms", text)
        self.assertNotIn("within +/-0.25s", text)


class BatchConsumerMetricTests(unittest.TestCase):
    def test_quantitative_default_reference_path_is_public_and_project_relative(self) -> None:
        self.assertEqual(DEFAULT_REFERENCES_DIR.relative_to(PROJECT), Path("checked-references"))

    def test_quantitative_row_exposes_current_millisecond_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "song.lrc"
            gen = root / "generated.lrc"
            report_path = root / "song.align-report.json"
            ref.write_text("[00:01.00]A\n[00:02.00]B\n", encoding="utf-8")
            gen.write_text("[00:01.03]A\n[00:02.06]B\n", encoding="utf-8")
            report_path.write_text("{}", encoding="utf-8")
            case = MatchedCase("song", "song", "test", ref, report_path, {})

            row = evaluate_row(case, gen, report_path, {}, "existing", "")

        self.assertEqual(row["timing_compared_entries"], 2)
        self.assertEqual(row["correct_le_30ms_percent"], 50.0)
        self.assertEqual(row["wrong_gt_50ms_percent"], 50.0)
        self.assertEqual(row["max_abs_delta_ms"], 60)

    def test_benchmark_can_gate_wrong_over_50ms_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "song.lrc"
            gen = root / "generated.lrc"
            report = gen.with_suffix(".align-report.json")
            ref.write_text("[00:01.00]A\n", encoding="utf-8")
            gen.write_text("[00:01.06]A\n", encoding="utf-8")
            report.write_text("{}", encoding="utf-8")
            case = BenchmarkCase(
                name="50ms gate",
                reference=ref,
                generated=gen,
                require_within_50cs=0.0,
                require_wrong_gt_50ms=0,
            )

            ok, result, failures = evaluate_case(case)

        self.assertFalse(ok)
        self.assertEqual(result["wrong_gt_50ms"], 1)
        self.assertTrue(any("wrong >50 ms=1" in failure for failure in failures))

    def test_markdown_summary_leads_with_current_30_50ms_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.md"
            write_markdown(
                [
                    {
                        "status": "OK",
                        "case": "song",
                        "category": "test",
                        "selected_backend": "ctc",
                        "reference_entries": 4,
                        "timing_compared_entries": 4,
                        "correct_le_30ms_percent": 25.0,
                        "acceptable_30_to_50ms_percent": 50.0,
                        "wrong_gt_50ms_percent": 25.0,
                        "median_abs_delta_ms": 40.0,
                        "mae_ms": 42.5,
                        "max_abs_delta_ms": 60,
                        "trusted_percent": 75.0,
                        "review_required_percent": 25.0,
                        "review_required_count": 1,
                    }
                ],
                output,
            )
            text = output.read_text(encoding="utf-8")

        self.assertIn("<=30 ms", text)
        self.assertIn("30-50 ms", text)
        self.assertIn(">50 ms", text)
        self.assertIn("Median", text)
        self.assertIn("MAE", text)
        self.assertIn("60 ms", text)
        self.assertNotIn("<=0.25s", text)


if __name__ == "__main__":
    unittest.main()
