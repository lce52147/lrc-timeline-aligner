#!/usr/bin/env python3
"""Public-safe unit tests for alignment trust and anchor matching logic."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import auto_lrc
from auto_lrc import (
    AnchorHint,
    AudioFeatures,
    LrcError,
    LyricEntry,
    apply_ctc_acoustic_backtrack,
    apply_ctc_crossline_initial_recovery,
    apply_ctc_weak_prefix_recovery,
    apply_ctc_local_fusion_to_whisperx,
    apply_ctc_micro_refinement_to_whisperx,
    apply_vocal_onset_tiebreak,
    apply_whisperx_acoustic_boundary_refinement,
    annotate_ctc_boundary_evidence,
    build_parser,
    choose_alignment_candidate,
    choose_onset_consensus_time,
    flag_unresolved_raw_ctc_disagreements,
    has_ambiguous_lyric_prefix,
    match_anchor_entries,
    load_lyrics,
    refresh_ctc_confidence_diagnostics,
    raw_asr_is_fallback_eligible,
    remove_generated_title_cards,
    score_line_timing_candidates,
    should_accept_zero_gap_boundary_realign,
    should_prefer_ctc_over_review_exploded_hybrid,
    split_group_text,
    update_report_confidence_metrics,
)
from evaluate_lrc import summarize
from export_alignment_audit import build_rows, write_markdown


class AnchorHintTests(unittest.TestCase):
    def test_onset_consensus_prefers_two_model_agreement_over_weak_ctc_initial(self) -> None:
        candidate, reason = choose_onset_consensus_time(23.29, 0.002, 23.03, 0.52, 23.14)
        self.assertAlmostEqual(candidate or 0.0, 23.085, places=3)
        self.assertEqual(reason, "whisperx-japanese-ctc-onset-consensus")
        candidate, reason = choose_onset_consensus_time(202.27, 0.06, 202.525, 0.96, 202.539)
        self.assertAlmostEqual(candidate or 0.0, 202.532, places=3)
        self.assertEqual(reason, "whisperx-japanese-ctc-over-ctc-onset-consensus")
        candidate, reason = choose_onset_consensus_time(123.876, 0.009, 123.222, 0.889, 124.112)
        self.assertAlmostEqual(candidate or 0.0, 123.222, places=3)
        self.assertEqual(reason, "high-confidence-whisperx-over-weak-ctc-initial")

    def test_untimed_adjacent_translation_lines_share_one_timing_entry(self) -> None:
        with TemporaryDirectory() as temp_name:
            lyrics = Path(temp_name) / "lyrics.txt"
            lyrics.write_text(
                "春を待つ\n等待春天\nI'll stay here\n我會留在這裡\n",
                encoding="utf-8",
            )

            document = load_lyrics(lyrics)

        self.assertEqual(len(document.entries), 2)
        self.assertEqual(document.entries[0].lines, ["春を待つ", "等待春天"])
        self.assertEqual(document.entries[1].lines, ["I'll stay here", "我會留在這裡"])

    def test_untimed_same_language_lines_are_not_merged(self) -> None:
        with TemporaryDirectory() as temp_name:
            lyrics = Path(temp_name) / "lyrics.txt"
            lyrics.write_text("春を待つ\nあなたを待つ\n", encoding="utf-8")

            document = load_lyrics(lyrics)

        self.assertEqual(len(document.entries), 2)

    def test_evaluator_ignores_only_the_generated_zero_time_title_card(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            reference = root / "reference.lrc"
            generated = root / "generated.lrc"
            reference.write_text("[00:01.00]first lyric\n", encoding="utf-8")
            generated.write_text("[00:00.00]Artist - Title\n[00:01.00]first lyric\n", encoding="utf-8")

            result = summarize(reference, generated)

        self.assertEqual(result["reference_entries"], 1)
        self.assertEqual(result["generated_entries"], 1)
        self.assertEqual(result["text_mismatches"], 0)
        self.assertEqual(result["max_abs_delta_cs"], 0)

    def test_double_space_separates_bilingual_display_lines(self) -> None:
        self.assertEqual(
            split_group_text("夜に浮かぶ月を仰いで  遠くに浮かぶ月", preserve_single=True),
            ["夜に浮かぶ月を仰いで", "遠くに浮かぶ月"],
        )

    def test_single_spaces_remain_part_of_one_lyric_line(self) -> None:
        self.assertEqual(
            split_group_text("その瞳を その声を 憶えている", preserve_single=True),
            ["その瞳を その声を 憶えている"],
        )

    def test_generated_title_card_is_not_reused_as_alignment_lyric(self) -> None:
        original_labels = auto_lrc.audio_track_labels
        auto_lrc.audio_track_labels = lambda _path: ("Tayori", "可惜夜")  # type: ignore[assignment]
        try:
            entries = [
                LyricEntry(["Tayori - 可惜夜"], 0),
                LyricEntry(["ずっと遠くに感じていた"], 1890),
            ]

            kept = remove_generated_title_cards(entries, Path("dummy.flac"))

            self.assertEqual([entry.lines[0] for entry in kept], ["ずっと遠くに感じていた"])
        finally:
            auto_lrc.audio_track_labels = original_labels  # type: ignore[assignment]

    def test_untimed_source_title_header_is_not_reused_as_alignment_lyric(self) -> None:
        original_labels = auto_lrc.audio_track_labels
        auto_lrc.audio_track_labels = lambda _path: ("", "Song Title")  # type: ignore[assignment]
        try:
            entries = [
                LyricEntry(["Song Title", "Song translation"]),
                LyricEntry(["actual lyric"], None),
            ]

            kept = remove_generated_title_cards(entries, Path("dummy.flac"))

            self.assertEqual([entry.lines[0] for entry in kept], ["actual lyric"])
        finally:
            auto_lrc.audio_track_labels = original_labels  # type: ignore[assignment]

    def test_probe_option_parses_without_output_path(self) -> None:
        args = build_parser().parse_args(["song.flac", "--probe"])

        self.assertTrue(args.probe)
        self.assertIsNone(args.output)

    def test_entry_number_disambiguates_repeated_lyrics(self) -> None:
        entries = [
            LyricEntry(["repeat"]),
            LyricEntry(["middle"]),
            LyricEntry(["repeat"]),
        ]

        matches = match_anchor_entries(entries, [AnchorHint(3, LyricEntry(["repeat"], 1234))])

        self.assertEqual([(index + 1, anchor.source_time_cs) for index, anchor in matches], [(3, 1234)])

    def test_entry_number_rejects_text_mismatch(self) -> None:
        entries = [
            LyricEntry(["repeat"]),
            LyricEntry(["middle"]),
            LyricEntry(["repeat"]),
        ]

        with self.assertRaisesRegex(LrcError, "text mismatch"):
            match_anchor_entries(entries, [AnchorHint(2, LyricEntry(["repeat"], 1234))])


class TimingTrustTests(unittest.TestCase):
    def test_shared_long_prefix_disables_raw_asr_as_unique_evidence(self) -> None:
        entries = [
            LyricEntry(["消えてしまいたい生涯なんてもんに愛を望んだって"]),
            LyricEntry(["消えてしまいたい生涯なんてもんに温もり望んだって"]),
        ]

        self.assertTrue(has_ambiguous_lyric_prefix(entries, 0))
        self.assertTrue(has_ambiguous_lyric_prefix(entries, 1))

    def test_raw_systematic_drift_is_one_section_review_not_many_line_failures(self) -> None:
        timestamps = [10.0, 20.0, 30.0, 40.0, 50.0]
        report: dict[str, object] = {
            "timing_entries": len(timestamps),
            "assignments": [{"timestamp": timestamp, "score": 0.9} for timestamp in timestamps],
            "suspicious_alignments": [],
        }
        raw_report: dict[str, object] = {
            "assignments": [
                {"timestamp": 9.05, "score": 0.92},
                {"timestamp": 19.10, "score": 0.91},
                {"timestamp": 29.00, "score": 0.95},
                {"timestamp": 39.08, "score": 0.89},
                {"timestamp": 49.02, "score": 0.94},
            ]
        }

        flag_unresolved_raw_ctc_disagreements(timestamps, report, raw_report)

        self.assertEqual(report["review_required_count"], 1)
        self.assertEqual(len(report["raw_asr_systematic_drift_runs"]), 1)  # type: ignore[arg-type]
        risk = report["suspicious_alignments"][0]  # type: ignore[index]
        self.assertIn("raw_asr_systematic_drift", risk["flags"])  # type: ignore[index]
        self.assertEqual(risk["raw_asr_systematic_drift"]["entries"], [1, 2, 3, 4, 5])  # type: ignore[index]

    def test_zero_score_raw_assignment_is_not_alignment_evidence(self) -> None:
        timestamps = [10.0, 20.0]
        report: dict[str, object] = {
            "timing_entries": len(timestamps),
            "assignments": [{"timestamp": timestamp, "score": 0.9} for timestamp in timestamps],
            "suspicious_alignments": [],
        }
        raw_report: dict[str, object] = {
            "assignments": [
                {"timestamp": 2.0, "score": 0.0},
                {"timestamp": 18.9, "score": 0.0},
            ]
        }

        flag_unresolved_raw_ctc_disagreements(timestamps, report, raw_report)

        self.assertEqual(report["review_required_count"], 0)
        self.assertEqual(report["raw_asr_zero_score_rejections"], 2)

    def test_raw_disagreement_compares_each_entry_to_its_own_ctc_time(self) -> None:
        timestamps = [10.0, 20.0]
        report: dict[str, object] = {
            "timing_entries": len(timestamps),
            "assignments": [{"timestamp": timestamp, "score": 0.9} for timestamp in timestamps],
            "suspicious_alignments": [],
        }
        raw_report: dict[str, object] = {
            "assignments": [
                {"timestamp": 10.05, "score": 0.95},
                {"timestamp": 20.70, "score": 0.95},
            ]
        }

        flag_unresolved_raw_ctc_disagreements(timestamps, report, raw_report)

        self.assertEqual(report["review_required_count"], 1)
        risk = report["suspicious_alignments"][0]  # type: ignore[index]
        self.assertEqual(risk["entry"], 2)  # type: ignore[index]
        self.assertEqual(risk["candidate_timestamps"]["output"], 20.0)  # type: ignore[index]

    def test_local_window_uses_first_line_raw_anchor_to_exclude_instrumental_intro(self) -> None:
        # The exact CTC subprocess is covered in integration runs.  This
        # regression protects the boundary choice used before that subprocess.
        raw_anchor = 3.76
        local_start = max(0.0, raw_anchor - 1.00)

        self.assertAlmostEqual(local_start, 2.76, places=3)

    def test_review_exploded_hybrid_prefers_clean_ctc_sequence(self) -> None:
        self.assertTrue(
            should_prefer_ctc_over_review_exploded_hybrid(
                {"review_required_count": 44},
                {"ctc_missing_count": 0, "review_required_count": 3, "collapse_detected": False},
            )
        )

    def test_review_guard_keeps_collapsed_or_missing_ctc_out_of_contention(self) -> None:
        self.assertFalse(
            should_prefer_ctc_over_review_exploded_hybrid(
                {"review_required_count": 44},
                {"ctc_missing_count": 1, "review_required_count": 0, "collapse_detected": False},
            )
        )
        self.assertFalse(
            should_prefer_ctc_over_review_exploded_hybrid(
                {"review_required_count": 44},
                {"ctc_missing_count": 0, "review_required_count": 0, "collapse_detected": True},
            )
        )

    def test_candidate_scoring_rejects_raw_time_inside_the_previous_ctc_tail(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["target"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9, "ctc_token_spans": [{"end": 10.8}]},
                {
                    "entry": 2,
                    "timestamp": 11.2,
                    "score": 0.9,
                    "ctc_token_spans": [{"start": 11.2}, {"start": 11.3}],
                },
                {"entry": 3, "timestamp": 15.0, "score": 0.9},
            ],
            "suspicious_alignments": [
                {"entry": 2, "candidate_timestamps": {"raw_asr": 10.5}, "raw_asr_score": 0.96}
            ],
        }

        result = score_line_timing_candidates(entries, [10.0, 11.2, 15.0], report, 20.0)

        self.assertAlmostEqual(result[1], 11.2, places=3)
        raw = next(item for item in report["assignments"][1]["candidates"] if item["source"] == "raw_asr")  # type: ignore[index]
        self.assertIn("raw-inside-previous-ctc-token-tail", raw["reasons"])

    def test_ctc_boundary_evidence_marks_a_clear_interline_gap(self) -> None:
        assignments: list[object] = [
            {"ctc_token_spans": [{"char": "a", "start": 42.0, "end": 48.60, "score": 0.7}]},
            {"ctc_token_spans": [{"char": "d", "start": 49.82, "end": 49.84, "score": 0.95}]},
        ]

        annotate_ctc_boundary_evidence(assignments)

        target = assignments[1]
        self.assertTrue(target["ctc_clear_boundary"])  # type: ignore[index]
        self.assertAlmostEqual(target["ctc_boundary_gap_seconds"], 1.22, places=3)  # type: ignore[index]

    def test_ctc_boundary_evidence_rejects_noisy_nearby_transition(self) -> None:
        assignments: list[object] = [
            {"ctc_token_spans": [{"char": "a", "start": 42.0, "end": 48.60, "score": 0.7}]},
            {"ctc_token_spans": [{"char": "d", "start": 48.68, "end": 48.70, "score": 0.95}]},
        ]

        annotate_ctc_boundary_evidence(assignments)

        self.assertFalse(assignments[1]["ctc_clear_boundary"])  # type: ignore[index]

    def test_candidate_scoring_selects_bounded_higher_evidence_raw_candidate(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["target"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                {"entry": 2, "segment": 2, "score": 0.35, "timestamp": 20.0},
                {"entry": 3, "segment": 3, "score": 0.96, "timestamp": 25.0},
            ],
            "suspicious_alignments": [
                {
                    "entry": 2,
                    "review_required": False,
                    "candidate_timestamps": {"raw_asr": 20.15},
                    "raw_asr_score": 0.96,
                }
            ],
        }

        result = score_line_timing_candidates(entries, [10.0, 20.0, 25.0], report, 30.0)

        self.assertAlmostEqual(result[1], 20.15, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertEqual(assignment["chosen_time"], 20.15)
        self.assertGreaterEqual(assignment["confidence"], 0.70)
        self.assertTrue(assignment["candidates"])
        self.assertTrue(assignment["rejected_candidates"])

    def test_candidate_scoring_marks_long_line_disagreement_with_split_suggestion(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["abcdefghijklmnopqrst"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                {"entry": 2, "segment": 2, "score": 0.90, "timestamp": 20.0},
                {"entry": 3, "segment": 3, "score": 0.96, "timestamp": 28.0},
            ],
            "suspicious_alignments": [
                {
                    "entry": 2,
                    "review_required": False,
                    "candidate_timestamps": {"whisperx_forced_first": 17.5},
                }
            ],
        }

        score_line_timing_candidates(entries, [10.0, 20.0, 28.0], report, 32.0)

        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertTrue(assignment["review_required"])
        self.assertIn("long_line_disagreement", assignment["flags"])
        self.assertIn("split_suggestion", assignment)

    def test_candidate_scoring_recovers_repeated_leading_term_onset(self) -> None:
        entries = [
            LyricEntry(["前の行"]),
            LyricEntry(["人生 人生 人生とか"]),
            LyricEntry(["次の行"]),
        ]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "score": 0.9, "timestamp": 171.4},
                {
                    "entry": 2,
                    "score": 0.7,
                    "timestamp": 174.266,
                    "ctc_score": 0.073687,
                    "ctc_token_spans": [
                        {"char": "x", "start": 174.266, "score": 0.02},
                        {"char": "y", "start": 174.366, "score": 0.20},
                    ],
                    "ctc_first_token_candidates": [
                        {"time": 172.770, "score": 0.034785},
                        {"time": 174.091, "score": 0.039345},
                    ],
                },
                {"entry": 3, "score": 0.9, "timestamp": 175.929},
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(entries, [171.4, 174.266, 175.929], report, 180.0)

        self.assertAlmostEqual(result[1], 172.770, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertEqual(assignment["chosen_time"], 172.770)
        self.assertIn("ctc_repeated_leading_term_onset", [item["source"] for item in assignment["candidates"]])  # type: ignore[index]

    def test_repeated_leading_term_recovery_rejects_prior_ctc_tail_peak(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["again again target"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 10.0,
                    "ctc_token_spans": [{"char": "e", "end": 14.9, "score": 0.8}],
                },
                {
                    "entry": 2,
                    "timestamp": 16.0,
                    "ctc_score": 0.05,
                    "ctc_token_spans": [{"char": "a", "start": 16.0, "score": 0.2}],
                    "ctc_first_token_candidates": [{"time": 14.8, "score": 0.08}],
                },
                {"entry": 3, "timestamp": 20.0},
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(entries, [10.0, 16.0, 20.0], report, 25.0)

        self.assertEqual(result[1], 16.0)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertNotIn("ctc_repeated_leading_term_onset", [item["source"] for item in assignment["candidates"]])  # type: ignore[index]

    def test_repeated_leading_term_keeps_usable_forced_initial(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["again again target"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {
                    "entry": 2,
                    "timestamp": 16.0,
                    "score": 0.7,
                    "ctc_score": 0.07,
                    "ctc_token_spans": [{"char": "a", "start": 16.0, "score": 0.12}],
                    "ctc_first_token_candidates": [{"time": 14.4, "score": 0.03}],
                },
                {"entry": 3, "timestamp": 20.0, "score": 0.9},
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(entries, [10.0, 16.0, 20.0], report, 25.0)

        self.assertAlmostEqual(result[1], 16.0, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertNotIn("ctc_repeated_leading_term_onset", [item["source"] for item in assignment["candidates"]])  # type: ignore[index]

    def test_candidate_scoring_never_overwrites_manual_anchor(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["人生 人生 人生とか"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "score": 0.9, "timestamp": 171.4},
                {
                    "entry": 2,
                    "score": 1.0,
                    "timestamp": 172.8,
                    "manual_anchor_hint": True,
                    "ctc_score": 0.01,
                    "ctc_first_token_candidates": [{"time": 172.77, "score": 0.03}],
                },
                {"entry": 3, "score": 0.9, "timestamp": 175.9},
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(entries, [171.4, 172.8, 175.9], report, 180.0)

        self.assertAlmostEqual(result[1], 172.8, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertEqual(assignment["reasons"], ["human-reviewed-anchor"])
        self.assertEqual(assignment["confidence"], 1.0)

    def test_candidate_scoring_ignores_peak_inside_previous_ctc_tail(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["kana"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 10.0,
                    "ctc_token_spans": [{"char": "e", "end": 15.00, "score": 0.8}],
                },
                {
                    "entry": 2,
                    "timestamp": 15.35,
                    "ctc_score": 0.05,
                    "ctc_token_spans": [{"char": "k", "start": 15.35, "score": 0.01}],
                    "ctc_first_token_candidates": [{"time": 14.95, "score": 0.2}],
                },
                {"entry": 3, "timestamp": 20.0},
            ],
            "suspicious_alignments": [],
        }

        score_line_timing_candidates(entries, [10.0, 15.35, 20.0], report, 25.0)

        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertNotIn("phonetic_anchor_disagreement", assignment["flags"])
        self.assertNotIn("ctc_nearby_phonetic_peak", [item["source"] for item in assignment["candidates"]])  # type: ignore[index]

    def test_isolated_low_ctc_scores_remain_untrusted_without_forcing_review(self) -> None:
        entries = [LyricEntry(["one"]), LyricEntry(["two"]), LyricEntry(["three"])]
        report: dict[str, object] = {
            "ctc_missing_entries": [],
            "assignments": [
                {"entry": 1, "timestamp": 1.0, "ctc_score": 0.02},
                {"entry": 2, "timestamp": 2.0, "ctc_score": 0.02},
                {"entry": 3, "timestamp": 3.0, "ctc_score": 0.02},
            ],
        }

        refresh_ctc_confidence_diagnostics(entries, report)

        self.assertEqual(report["ctc_low_score_count"], 3)
        self.assertEqual(report["review_required_count"], 0)
        self.assertFalse(report["review_required"])
        self.assertEqual(report["low_confidence_count"], 3)

        score_line_timing_candidates(entries, [1.0, 2.0, 3.0], report, 4.0)
        self.assertEqual(report["review_required_count"], 0)
        self.assertFalse(any(item["review_required"] for item in report["assignments"]))  # type: ignore[index]

    def test_four_consecutive_low_ctc_scores_are_a_review_collapse(self) -> None:
        entries = [LyricEntry([f"line {index}"]) for index in range(4)]
        report: dict[str, object] = {
            "ctc_missing_entries": [],
            "assignments": [
                {"entry": index + 1, "timestamp": float(index + 1), "ctc_score": 0.02}
                for index in range(4)
            ],
        }

        refresh_ctc_confidence_diagnostics(entries, report)

        self.assertTrue(report["collapse_detected"])
        self.assertEqual(report["review_required_count"], 4)
        self.assertEqual(report["suspicious_alignment_severity_counts"]["high"], 4)  # type: ignore[index]

    def test_raw_candidate_beats_conflicted_ctc_candidate(self) -> None:
        ctc_report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 22,
            "ctc_missing_count": 0,
            "ctc_low_score_count": 0,
            "ctc_very_low_score_count": 0,
            "review_required_count": 16,
        }
        raw_report: dict[str, object] = {
            "backend": "whispercpp",
            "timing_entries": 22,
            "trusted_percent": 90.91,
            "low_confidence_count": 2,
            "review_required_count": 2,
        }

        selected = choose_alignment_candidate(
            [
                {"backend": "ctc", "timestamps": [1.0], "report": ctc_report},
                {"backend": "whispercpp", "timestamps": [1.0], "report": raw_report},
            ]
        )

        self.assertEqual(selected["backend"], "whispercpp")

    def test_raw_fallback_rejects_abnormally_long_asr_segment(self) -> None:
        self.assertFalse(raw_asr_is_fallback_eligible({"raw_asr_max_segment_seconds": 29.98}))
        self.assertTrue(raw_asr_is_fallback_eligible({"raw_asr_max_segment_seconds": 4.2}))

    def test_high_confidence_raw_replaces_detached_ctc_initial_token(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["target"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "score": 0.9, "timestamp": 34.87},
                {
                    "entry": 2,
                    "score": 0.9,
                    "timestamp": 36.396,
                    "ctc_score": 0.454,
                    "ctc_token_spans": [
                        {"char": "i", "start": 36.396, "end": 36.416},
                        {"char": "k", "start": 37.258, "end": 37.278},
                    ],
                },
                {"entry": 3, "score": 0.9, "timestamp": 38.34},
            ],
            "suspicious_alignments": [
                {
                    "entry": 2,
                    "review_required": True,
                    "candidate_timestamps": {"raw_asr": 37.0},
                    "raw_asr_score": 0.96,
                }
            ],
        }

        result = score_line_timing_candidates(entries, [34.87, 36.396, 38.34], report, 42.0)

        self.assertAlmostEqual(result[1], 37.0, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertFalse(assignment["review_required"])
        self.assertEqual(assignment["reasons"], ["raw-asr-lyric-match", "raw-resolves-detached-ctc-initial-token"])
        self.assertFalse(report["suspicious_alignments"][0]["review_required"])  # type: ignore[index]

    def test_detached_ctc_initial_token_uses_second_token_without_raw_support(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["target"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "score": 0.9, "timestamp": 102.686},
                {
                    "entry": 2,
                    "score": 0.7,
                    "timestamp": 111.947,
                    "ctc_score": 0.031,
                    "ctc_token_spans": [
                        {"char": "t", "start": 111.947, "end": 111.967, "score": 0.008},
                        {"char": "s", "start": 113.807, "end": 113.827},
                    ],
                },
                {"entry": 3, "score": 0.9, "timestamp": 121.167},
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(entries, [102.686, 111.947, 121.167], report, 130.0)

        self.assertAlmostEqual(result[1], 113.807, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertEqual(assignment["reasons"], ["second-ctc-token-start-after-detached-initial-token"])

    def test_detached_high_score_ctc_initial_token_is_not_skipped(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["Hi"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "score": 0.9, "timestamp": 10.0},
                {
                    "entry": 2,
                    "score": 0.9,
                    "timestamp": 18.322,
                    "ctc_score": 0.2,
                    "ctc_token_spans": [
                        {"char": "h", "start": 18.322, "end": 18.342, "score": 0.8},
                        {"char": "i", "start": 19.662, "end": 19.682, "score": 0.1},
                    ],
                },
                {"entry": 3, "score": 0.9, "timestamp": 23.64},
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(entries, [10.0, 18.322, 23.64], report, 30.0)

        self.assertAlmostEqual(result[1], 18.322, places=3)

    def test_timing_trusted_assignment_counts_as_trusted_without_changing_score(self) -> None:
        report: dict[str, object] = {
            "timing_entries": 2,
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                {
                    "entry": 2,
                    "segment": 2,
                    "score": 0.48,
                    "timestamp": 20.0,
                    "timing_trusted": True,
                },
            ],
            "low_confidence_entries": [{"entry": 2, "score": 0.48, "lyric": "low text score"}],
            "review_required_count": 0,
        }

        update_report_confidence_metrics(report)

        self.assertEqual(report["trusted_entries"], 2)
        self.assertEqual(report["trusted_percent"], 100.0)
        self.assertEqual(report["low_confidence_count"], 0)
        self.assertEqual(report["timing_trusted_entries"], 1)
        self.assertEqual(report["assignments"][1]["score"], 0.48)  # type: ignore[index]

    def test_ctc_raw_consensus_resolves_local_fusion_review(self) -> None:
        whisperx_report: dict[str, object] = {
            "backend": "whisperx",
            "strategy": "whisperx-hybrid-experimental",
            "timing_entries": 2,
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                {"entry": 2, "segment": 2, "score": 0.55, "timestamp": 20.0},
            ],
            "low_confidence_entries": [{"entry": 2, "score": 0.55, "lyric": "second"}],
            "suspicious_alignments": [],
            "review_required_count": 0,
        }
        ctc_report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "ctc_score": 0.1},
                {
                    "entry": 2,
                    "timestamp": 22.5,
                    "ctc_score": 0.1,
                    "ctc_token_spans": [{"char": "s", "start": 22.5, "end": 22.6, "score": 0.8}],
                },
            ],
        }
        raw_report: dict[str, object] = {
            "backend": "whispercpp_raw",
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.9, "timestamp": 10.0},
                {"entry": 2, "segment": 2, "score": 0.76, "timestamp": 22.3},
            ],
        }

        timestamps, report, changes = apply_ctc_local_fusion_to_whisperx(
            [10.0, 20.0],
            whisperx_report,
            ctc_report,
            duration=30.0,
            raw_report=raw_report,
        )

        self.assertEqual(timestamps, [10.0, 22.5])
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["reason"], "ctc-raw-consensus-over-whisperx")
        self.assertTrue(changes[0]["fusion_trusted"])
        self.assertEqual(changes[0]["ctc_token_spans"][0]["char"], "s")
        self.assertEqual(report["review_required_count"], 0)
        self.assertEqual(report["trusted_percent"], 100.0)
        self.assertEqual(report["low_confidence_count"], 0)
        assignments = report["assignments"]  # type: ignore[assignment]
        self.assertTrue(assignments[1]["timing_trusted"])  # type: ignore[index]
        self.assertEqual(assignments[1]["score"], 0.55)  # type: ignore[index]
        self.assertEqual(assignments[1]["ctc_token_spans"][0]["start"], 22.5)  # type: ignore[index]

    def test_ctc_micro_refinement_resolves_review_when_candidates_are_close(self) -> None:
        whisperx_report: dict[str, object] = {
            "backend": "whisperx",
            "strategy": "unit-test",
            "timing_entries": 3,
            "assignments": [
                {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                {"entry": 2, "segment": 2, "score": 0.96, "timestamp": 20.4},
                {"entry": 3, "segment": 3, "score": 0.96, "timestamp": 25.0},
            ],
            "suspicious_alignments": [
                {
                    "entry": 2,
                    "flags": ["close_neighbor_onset_uncertain"],
                    "severity": "medium",
                    "review_required": True,
                    "candidate_timestamps": {"output": 20.4},
                }
            ],
            "review_required_count": 1,
            "review_required": True,
        }
        ctc_report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "ctc_score": 0.4},
                {
                    "entry": 2,
                    "timestamp": 20.05,
                    "ctc_score": 0.12,
                    "ctc_token_spans": [{"char": "a", "start": 20.05, "end": 20.07, "score": 0.2}],
                },
                {"entry": 3, "timestamp": 25.0, "ctc_score": 0.4},
            ],
        }

        timestamps, report, changes = apply_ctc_micro_refinement_to_whisperx(
            [10.0, 20.4, 25.0],
            whisperx_report,
            ctc_report,
            duration=30.0,
        )

        self.assertEqual(timestamps, [10.0, 20.05, 25.0])
        self.assertEqual(len(changes), 1)
        self.assertEqual(report["review_required_count"], 0)
        assignments = report["assignments"]  # type: ignore[assignment]
        self.assertTrue(assignments[1]["timing_trusted"])  # type: ignore[index]
        self.assertEqual(assignments[1]["ctc_token_spans"][0]["char"], "a")  # type: ignore[index]
        suspicious = report["suspicious_alignments"]  # type: ignore[assignment]
        self.assertEqual(suspicious[0]["severity"], "resolved")  # type: ignore[index]
        self.assertFalse(suspicious[0]["review_required"])  # type: ignore[index]


class CtcAcousticBacktrackTests(unittest.TestCase):
    def test_crossline_initial_recovery_rejects_a_prior_line_tail_token(self) -> None:
        timestamps = [39.46, 43.74, 52.91]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 39.46,
                    "ctc_token_spans": [{"char": "y", "start": 43.70, "end": 43.72, "score": 0.3}],
                },
                {
                    "entry": 2,
                    "timestamp": 43.74,
                    "ctc_token_spans": [
                        {"char": "i", "start": 43.74, "end": 43.76, "score": 0.19},
                        {"char": "l", "start": 46.32, "end": 46.34, "score": 0.2},
                    ],
                },
                {"entry": 3, "timestamp": 52.91, "ctc_token_spans": []},
            ]
        }

        recovered, updated, changes = apply_ctc_crossline_initial_recovery(timestamps, report, 60.0)

        self.assertAlmostEqual(recovered[1], 46.32, places=3)
        self.assertEqual(len(changes), 1)
        self.assertTrue(updated["assignments"][1]["ctc_crossline_initial_recovery"])  # type: ignore[index]

    def test_crossline_initial_recovery_keeps_a_normal_initial_cluster(self) -> None:
        timestamps = [39.46, 46.15, 52.91]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 39.46,
                    "ctc_token_spans": [{"char": "y", "start": 43.70, "end": 43.72, "score": 0.3}],
                },
                {
                    "entry": 2,
                    "timestamp": 46.15,
                    "ctc_token_spans": [
                        {"char": "i", "start": 46.15, "end": 46.17, "score": 0.2},
                        {"char": "l", "start": 46.32, "end": 46.34, "score": 0.2},
                    ],
                },
                {"entry": 3, "timestamp": 52.91, "ctc_token_spans": []},
            ]
        }

        recovered, _, changes = apply_ctc_crossline_initial_recovery(timestamps, report, 60.0)

        self.assertEqual(recovered, timestamps)
        self.assertEqual(changes, [])

    def test_weak_ctc_prefix_recovers_strong_earlier_first_token(self) -> None:
        timestamps = [195.0, 202.16, 207.03]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 195.0, "ctc_score": 0.5},
                {
                    "entry": 2,
                    "timestamp": 202.16,
                    "ctc_score": 0.08,
                    "ctc_token_spans": [{"char": "h", "start": 202.16, "score": 0.01}],
                    "ctc_first_token_candidates": [
                        {"time": 200.09, "score": 0.19},
                        {"time": 201.00, "score": 0.03},
                    ],
                },
                {"entry": 3, "timestamp": 207.03, "ctc_score": 0.5},
            ]
        }

        recovered, updated, changes = apply_ctc_weak_prefix_recovery(timestamps, report, 220.0)

        self.assertAlmostEqual(recovered[1], 200.09, places=3)
        self.assertEqual(len(changes), 1)
        self.assertTrue(updated["assignments"][1]["ctc_weak_prefix_recovery"])  # type: ignore[index]

    def test_weak_prefix_does_not_use_an_unconvincing_peak(self) -> None:
        timestamps = [100.0, 105.0, 110.0]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 100.0, "ctc_score": 0.5},
                {
                    "entry": 2,
                    "timestamp": 105.0,
                    "ctc_score": 0.1,
                    "ctc_token_spans": [{"char": "h", "start": 105.0, "score": 0.01}],
                    "ctc_first_token_candidates": [{"time": 103.5, "score": 0.04}],
                },
                {"entry": 3, "timestamp": 110.0, "ctc_score": 0.5},
            ]
        }

        recovered, _, changes = apply_ctc_weak_prefix_recovery(timestamps, report, 120.0)

        self.assertEqual(recovered, timestamps)
        self.assertEqual(changes, [])

    def test_weak_prefix_never_reuses_the_previous_line_tail(self) -> None:
        timestamps = [195.0, 202.16, 207.03]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 195.0,
                    "ctc_score": 0.5,
                    "ctc_token_spans": [{"char": "a", "end": 200.45, "score": 0.8}],
                },
                {
                    "entry": 2,
                    "timestamp": 202.16,
                    "ctc_score": 0.08,
                    "ctc_token_spans": [{"char": "h", "start": 202.16, "score": 0.01}],
                    "ctc_first_token_candidates": [{"time": 200.09, "score": 0.19}],
                },
                {"entry": 3, "timestamp": 207.03, "ctc_score": 0.5},
            ]
        }

        recovered, _, changes = apply_ctc_weak_prefix_recovery(timestamps, report, 220.0)

        self.assertEqual(recovered, timestamps)
        self.assertEqual(changes, [])

    def test_backtracks_low_confidence_r_initial_only(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 19.4))[0][0]] = 1.0
        features = AudioFeatures(
            duration=30.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda audio_path, duration: features  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["理"]), LyricEntry(["next"])]
            timestamps = [10.0, 20.0, 25.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.1,
                        "romaji": "ri",
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 30.0)

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 19.4, places=3)
            self.assertEqual(report["ctc_acoustic_backtrack_count"], 1)
            self.assertTrue(report["assignments"][1]["ctc_acoustic_backtrack"])  # type: ignore[index]

            entries = [LyricEntry(["previous"]), LyricEntry(["その"]), LyricEntry(["next"])]
            timestamps = [10.0, 20.0, 25.0]
            report = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.1,
                        "romaji": "sono",
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 30.0)

            self.assertEqual(changes, [])
            self.assertEqual(timestamps[1], 20.0)
            self.assertEqual(report["ctc_acoustic_backtrack_count"], 0)
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]


class VocalOnsetTiebreakTests(unittest.TestCase):
    def test_zero_gap_boundary_realign_requires_a_stronger_late_local_path(self) -> None:
        self.assertTrue(should_accept_zero_gap_boundary_realign(0.0, 0.20, 10.0, 10.75, 0.24))
        self.assertFalse(should_accept_zero_gap_boundary_realign(0.12, 0.20, 10.0, 10.75, 0.24))
        self.assertFalse(should_accept_zero_gap_boundary_realign(0.0, 0.20, 10.0, 10.20, 0.24))
        self.assertFalse(should_accept_zero_gap_boundary_realign(0.0, 0.20, 10.0, 10.75, 0.20))

    def test_prefers_ctc_only_when_vocal_onset_evidence_is_clear(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.02), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 18.8))[0][0]] = 0.95
        onset_strength[np.where(np.isclose(frame_times, 20.2))[0][0]] = 0.40
        features = AudioFeatures(30.0, frame_times, np.full_like(frame_times, -20.0), onset_strength, [])
        original_features = auto_lrc.vocal_onset_features
        auto_lrc.vocal_onset_features = lambda audio, duration, args: (features, None)  # type: ignore[assignment]
        try:
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0},
                    {"entry": 2, "timestamp": 20.0},
                    {"entry": 3, "timestamp": 25.0},
                ]
            }
            ctc_report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0},
                    {"entry": 2, "timestamp": 18.8},
                    {"entry": 3, "timestamp": 25.0},
                ]
            }
            timestamps, result, changes = apply_vocal_onset_tiebreak(
                [10.0, 20.0, 25.0], report, ctc_report, Path("dummy.flac"), 30.0, object()  # type: ignore[arg-type]
            )
            self.assertEqual(timestamps[0], 10.0)
            self.assertAlmostEqual(timestamps[1], 18.8, places=3)
            self.assertEqual(timestamps[2], 25.0)
            self.assertEqual(changes[0]["reason"], "demucs-vocal-onset-prefers-ctc")
            self.assertEqual(result["vocal_onset_refinement"]["change_count"], 1)  # type: ignore[index]
        finally:
            auto_lrc.vocal_onset_features = original_features  # type: ignore[assignment]

    def test_backtracks_from_ctc_first_token_candidate_for_weak_prefix(self) -> None:
        features = AudioFeatures(
            duration=40.0,
            frame_times=np.round(np.arange(0.0, 40.0, 0.1), 3).astype(np.float32),
            rms_db=np.full(400, -20.0, dtype=np.float32),
            onset_strength=np.zeros(400, dtype=np.float32),
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda audio_path, duration: features  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["target"]), LyricEntry(["next"])]
            timestamps = [10.0, 20.0, 30.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.12,
                        "romaji": "mou",
                        "ctc_first_token_candidates": [
                            {"time": 19.7, "score": 0.10},
                            {"time": 18.6, "score": 0.008},
                        ],
                    },
                    {"entry": 3, "timestamp": 30.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 18.6, places=3)
            self.assertEqual(changes[0]["mode"], "ctc-first-token-posterior")
            self.assertEqual(report["assignments"][1]["ctc_acoustic_backtrack_mode"], "ctc-first-token-posterior")  # type: ignore[index]
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_rejects_weak_t_prefix_first_token_candidate(self) -> None:
        features = AudioFeatures(
            duration=40.0,
            frame_times=np.round(np.arange(0.0, 40.0, 0.1), 3).astype(np.float32),
            rms_db=np.full(400, -20.0, dtype=np.float32),
            onset_strength=np.zeros(400, dtype=np.float32),
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda audio_path, duration: features  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["target"]), LyricEntry(["next"])]
            timestamps = [10.0, 20.0, 30.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 10.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.12,
                        "romaji": "tomosu",
                        "ctc_first_token_candidates": [
                            {"time": 18.6, "score": 0.009},
                            {"time": 19.7, "score": 0.10},
                        ],
                    },
                    {"entry": 3, "timestamp": 30.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(changes, [])
            self.assertEqual(timestamps[1], 20.0)
            self.assertEqual(report["ctc_acoustic_backtrack_count"], 0)
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_short_low_ctc_snap_uses_nearby_acoustic_onset(self) -> None:
        frame_times = np.round(np.arange(0.0, 40.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 19.6))[0][0]] = 1.0
        features = AudioFeatures(
            duration=40.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda audio_path, duration: features  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["target"]), LyricEntry(["next"])]
            timestamps = [15.0, 20.0, 25.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 15.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.04,
                        "romaji": "sen",
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 19.6, places=3)
            self.assertEqual(changes[0]["mode"], "ctc-short-acoustic-onset")
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_short_acoustic_snap_keeps_credible_ctc_start_when_backtrack_harms_pacing(self) -> None:
        frame_times = np.round(np.arange(0.0, 40.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 19.6))[0][0]] = 1.0
        features = AudioFeatures(
            duration=40.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda audio_path, duration: features  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous phrase"]), LyricEntry(["target phrase"]), LyricEntry(["next phrase"])]
            timestamps = [15.0, 20.0, 25.0]
            report: dict[str, object] = {
                "assignments": [
                    {"entry": 1, "timestamp": 15.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.04,
                        "romaji": "sen",
                        "ctc_token_spans": [{"char": "s", "start": 20.0, "score": 0.08}],
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(changes, [])
            self.assertEqual(timestamps[1], 20.0)
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]


class WhisperxAcousticBoundaryTests(unittest.TestCase):
    def test_candidate_disagreement_backtracks_to_local_onset(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 19.5))[0][0]] = 1.0
        features = AudioFeatures(
            duration=30.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda audio_path, duration: features  # type: ignore[assignment]
        try:
            report: dict[str, object] = {
                "timing_entries": 3,
                "assignments": [
                    {"entry": 1, "score": 0.96, "timestamp": 10.0},
                    {"entry": 2, "score": 0.96, "timestamp": 20.0},
                    {"entry": 3, "score": 0.96, "timestamp": 25.0},
                ],
                "suspicious_alignments": [
                    {
                        "entry": 2,
                        "flags": ["candidate_disagreement"],
                        "severity": "low",
                        "review_required": False,
                        "candidate_timestamps": {"output": 20.0},
                    }
                ],
                "review_required_count": 0,
            }

            timestamps, refined_report, changes = apply_whisperx_acoustic_boundary_refinement(
                Path("dummy.flac"),
                [10.0, 20.0, 25.0],
                report,
                30.0,
            )

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 19.5, places=3)
            assignments = refined_report["assignments"]  # type: ignore[assignment]
            self.assertTrue(assignments[1]["timing_trusted"])  # type: ignore[index]
            suspicious = refined_report["suspicious_alignments"]  # type: ignore[assignment]
            self.assertEqual(suspicious[0]["severity"], "resolved")  # type: ignore[index]
            self.assertFalse(suspicious[0]["review_required"])  # type: ignore[index]
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_close_neighbor_can_snap_forward_to_next_strong_onset(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 20.5))[0][0]] = 1.0
        features = AudioFeatures(
            duration=30.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda audio_path, duration: features  # type: ignore[assignment]
        try:
            report: dict[str, object] = {
                "timing_entries": 3,
                "assignments": [
                    {"entry": 1, "score": 0.96, "timestamp": 15.0},
                    {"entry": 2, "score": 0.96, "timestamp": 20.0},
                    {"entry": 3, "score": 0.96, "timestamp": 25.0},
                ],
                "suspicious_alignments": [
                    {
                        "entry": 2,
                        "text": "long enough lyric line",
                        "flags": ["close_neighbor_onset_uncertain"],
                        "severity": "medium",
                        "review_required": True,
                        "candidate_timestamps": {"output": 20.0},
                    }
                ],
                "review_required_count": 1,
            }

            timestamps, refined_report, changes = apply_whisperx_acoustic_boundary_refinement(
                Path("dummy.flac"),
                [15.0, 20.0, 25.0],
                report,
                30.0,
            )

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 20.5, places=3)
            self.assertEqual(changes[0]["reason"], "lead-in-acoustic-forward-onset")
            self.assertEqual(refined_report["review_required_count"], 0)
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]


class AuditExportTests(unittest.TestCase):
    def test_audit_rows_expose_timing_trust_reason_and_sources(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lrc_path = tmp_path / "song.lrc"
            lrc_path.write_text("[00:10.00]first\n[00:22.50]second\n", encoding="utf-8")
            report: dict[str, object] = {
                "backend": "hybrid",
                "strategy": "unit-test",
                "timing_entries": 2,
                "trusted_percent": 100.0,
                "timing_trusted_entries": 1,
                "review_required_count": 0,
                "assignments": [
                    {"entry": 1, "segment": 1, "score": 0.96, "timestamp": 10.0},
                    {
                        "entry": 2,
                        "segment": 2,
                        "score": 0.55,
                        "timestamp": 22.5,
                        "chosen_time": 22.5,
                        "confidence": 0.81,
                        "reasons": ["raw-asr-lyric-match"],
                        "penalties": [{"kind": "candidate_disagreement", "value": 0.12}],
                        "split_suggestion": {"suggested_after_text": "second"},
                        "timing_trusted": True,
                        "timing_trusted_reason": "multi-backend-time-consensus",
                        "timing_trusted_sources": ["ctc", "raw"],
                        "ctc_token_spans": [{"char": "s", "start": 22.5, "end": 22.6, "score": 0.8}],
                        "timing_trusted_candidate_times": {
                            "selected": 22.5,
                            "whisperx": 20.0,
                            "ctc": 22.5,
                            "raw": 22.3,
                        },
                    },
                ],
                "suspicious_alignments": [],
            }

            rows = build_rows(lrc_path, report)
            output = tmp_path / "audit.md"
            write_markdown(rows, report, output)
            content = output.read_text(encoding="utf-8")

        self.assertEqual(rows[1]["timing_trusted"], "yes")
        self.assertEqual(rows[1]["timing_trust_sources"], "ctc;raw")
        self.assertEqual(rows[1]["ctc_tokens"], "s@00:22.50/0.800")
        self.assertIn("multi-backend-time-consensus", content)
        self.assertIn("ctc;raw", content)
        self.assertIn("ctc-tok s@00:22.50/0.800", content)
        self.assertIn("0.810", content)
        self.assertEqual(rows[1]["chosen_time"], "00:22.50")
        self.assertEqual(rows[1]["decision_penalties"], "candidate_disagreement:0.12")
        self.assertEqual(rows[1]["split_suggestion"], "second")
        self.assertIn("timing_trusted", content)


if __name__ == "__main__":
    unittest.main()
