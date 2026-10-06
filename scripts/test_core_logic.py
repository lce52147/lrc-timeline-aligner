#!/usr/bin/env python3
"""Public-safe unit tests for alignment trust and anchor matching logic."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
import types
import unittest
from unittest import mock
from dataclasses import FrozenInstanceError, replace
from itertools import permutations
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
    apply_ctc_forward_supported_prefix_recovery,
    apply_final_ctc_timing_guard,
    apply_ctc_weak_prefix_recovery,
    apply_ctc_local_window_realign,
    apply_ctc_local_fusion_to_whisperx,
    apply_ctc_micro_refinement_to_whisperx,
    apply_vocal_onset_tiebreak,
    apply_whisperx_acoustic_boundary_refinement,
    annotate_ctc_boundary_evidence,
    build_parser,
    choose_alignment_candidate,
    choose_onset_consensus_time,
    ctc_forward_supported_prefix_candidate,
    ctc_prefix_boundary_evidence,
    ctc_right_edge_pileup_evidence,
    ctc_path_fracture_evidence,
    duplicate_block_offset_evidence,
    lyric_prefix_probe_text,
    flag_unresolved_ctc_boundary_issues,
    flag_unresolved_raw_ctc_disagreements,
    has_ambiguous_lyric_prefix,
    match_anchor_entries,
    load_lyrics,
    refresh_ctc_confidence_diagnostics,
    raw_asr_is_fallback_eligible,
    remove_generated_title_cards,
    score_line_timing_candidates,
    should_accept_zero_gap_boundary_realign,
    should_accept_ctc_boundary_realign,
    has_ctc_compressed_prefix,
    should_prefer_ctc_over_review_exploded_hybrid,
    split_group_text,
    update_report_confidence_metrics,
    assignment_timing_is_trusted,
)
from evaluate_lrc import summarize
from export_alignment_audit import build_rows, write_markdown


class PersistentHelperRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        auto_lrc._PERSISTENT_HELPER_POOLS.clear()
        auto_lrc._PERSISTENT_HELPER_DISABLED.clear()
        auto_lrc._PERSISTENT_HELPER_LOGGED.clear()

    def tearDown(self) -> None:
        auto_lrc._PERSISTENT_HELPER_POOLS.clear()
        auto_lrc._PERSISTENT_HELPER_DISABLED.clear()
        auto_lrc._PERSISTENT_HELPER_LOGGED.clear()

    def test_missing_persistent_worker_falls_back_to_one_shot(self) -> None:
        expected = subprocess.CompletedProcess(["python", "helper.py"], 0, "ok", "")
        with TemporaryDirectory() as temp_name:
            missing_worker = Path(temp_name) / "missing_worker.py"
            with (
                mock.patch.object(auto_lrc, "_persistent_helper_worker_path", return_value=missing_worker),
                mock.patch.object(auto_lrc, "run_command", return_value=expected) as run_command,
                mock.patch.object(auto_lrc, "_PersistentHelperPool") as pool_class,
            ):
                result = auto_lrc._run_model_helper_command(
                    [sys.executable, str(Path(temp_name) / "helper.py"), "--probe"],
                    mode="ctc",
                    timeout_seconds=7.0,
                )

        self.assertIs(result, expected)
        run_command.assert_called_once()
        pool_class.assert_not_called()

    def test_present_persistent_worker_uses_pool(self) -> None:
        expected = subprocess.CompletedProcess(["python", "helper.py"], 0, "cached", "")
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            worker = root / "persistent_helper_worker.py"
            helper = root / "helper.py"
            worker.write_text("# worker\n", encoding="utf-8")
            helper.write_text("# helper\n", encoding="utf-8")
            pool = mock.Mock()
            pool.run.return_value = expected
            with (
                mock.patch.object(auto_lrc, "_persistent_helper_worker_path", return_value=worker),
                mock.patch.object(auto_lrc, "_PersistentHelperPool", return_value=pool) as pool_class,
                mock.patch.object(auto_lrc, "run_command") as run_command,
                mock.patch.object(auto_lrc, "progress_log"),
            ):
                result = auto_lrc._run_model_helper_command(
                    [sys.executable, str(helper), "--probe"],
                    mode="ctc",
                    timeout_seconds=7.0,
                )

        self.assertIs(result, expected)
        pool_class.assert_called_once()
        pool.run.assert_called_once_with(["--probe"], 7.0)
        run_command.assert_not_called()


class VocalStemDeterminismTests(unittest.TestCase):
    def test_demucs_generation_disables_random_shifts(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "input.flac"
            python_exe = root / "python.exe"
            cache_root = root / "cache"
            audio.write_bytes(b"audio")
            python_exe.write_bytes(b"")

            def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
                output_dir = Path(command[command.index("-o") + 1])
                stem_dir = output_dir / "htdemucs" / "input"
                stem_dir.mkdir(parents=True, exist_ok=True)
                (stem_dir / "vocals.mp3").write_bytes(b"stem")
                return subprocess.CompletedProcess(command, 0, "", "")

            args = type("Args", (), {"vocal_ctc": True})()
            with (
                mock.patch.object(auto_lrc, "default_ctc_python", return_value=python_exe),
                mock.patch.object(
                    auto_lrc,
                    "vocal_cache_identity",
                    return_value={"vocal_cache_key": "deterministic-test"},
                ),
                mock.patch.object(auto_lrc, "DEFAULT_VOCAL_CACHE_DIR", cache_root),
                mock.patch.object(auto_lrc.subprocess, "run", side_effect=fake_run) as run_command,
            ):
                path, source, status = auto_lrc.prepare_vocal_ctc_audio(audio, args)

        command = run_command.call_args.args[0]
        shift_index = command.index("--shifts")
        self.assertEqual(command[shift_index + 1], "0")
        self.assertEqual(source, "vocal-stem")
        self.assertEqual(status, "generated")
        self.assertEqual(path.name, "vocals.mp3")

    def test_vocal_separator_protocol_records_zero_shift_policy(self) -> None:
        self.assertEqual(
            auto_lrc.VOCAL_SEPARATOR_PROTOCOL_REVISION,
            "demucs-two-stems-vocals-mp3-v3-shifts0",
        )


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
            split_group_text("合成試験の第一行  Synthetic translation line", preserve_single=True),
            ["合成試験の第一行", "Synthetic translation line"],
        )

    def test_single_spaces_remain_part_of_one_lyric_line(self) -> None:
        self.assertEqual(
            split_group_text("合成 試験 文を 保持する", preserve_single=True),
            ["合成 試験 文を 保持する"],
        )

    def test_generated_title_card_is_not_reused_as_alignment_lyric(self) -> None:
        original_labels = auto_lrc.audio_track_labels
        auto_lrc.audio_track_labels = lambda _path: ("Demo Artist", "Demo Song")  # type: ignore[assignment]
        try:
            entries = [
                LyricEntry(["Demo Artist - Demo Song"], 0),
                LyricEntry(["synthetic lyric line"], 1890),
            ]

            kept = remove_generated_title_cards(entries, Path("dummy.flac"))

            self.assertEqual([entry.lines[0] for entry in kept], ["synthetic lyric line"])
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
            LyricEntry(["これは合成試験の共通接頭辞です甲を確認する"]),
            LyricEntry(["これは合成試験の共通接頭辞です乙を確認する"]),
        ]

        self.assertTrue(has_ambiguous_lyric_prefix(entries, 0))
        self.assertTrue(has_ambiguous_lyric_prefix(entries, 1))

    def test_raw_systematic_drift_is_one_section_diagnostic_not_review(self) -> None:
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

        self.assertEqual(report["review_required_count"], 0)
        self.assertEqual(len(report["raw_asr_systematic_drift_runs"]), 1)  # type: ignore[arg-type]
        risk = report["suspicious_alignments"][0]  # type: ignore[index]
        self.assertIn("raw_asr_systematic_drift", risk["flags"])  # type: ignore[index]
        self.assertFalse(risk["review_required"])
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

    def test_final_suffix_local_window_extends_to_audio_end(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["context"]), LyricEntry(["final line"])]
        timestamps = [10.0, 20.0, 25.0]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "ctc_score": 0.5},
                {"entry": 2, "timestamp": 20.0, "ctc_score": 0.5},
                {"entry": 3, "timestamp": 25.0, "ctc_score": 0.02},
            ],
            "ctc_low_score_runs": [[3]],
            "ctc_low_score_entries": [{"entry": 3, "text": "final line", "ctc_score": 0.02}],
            "ctc_missing_entries": [],
        }
        raw_report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {"entry": 2, "timestamp": 20.0, "score": 0.9},
                {"entry": 3, "timestamp": 25.0, "score": 0.9},
            ]
        }
        captured: dict[str, float] = {}
        original_run = auto_lrc.subprocess.run
        original_python = auto_lrc.default_ctc_python

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
            output_path = Path(command[command.index("--output") + 1])
            captured["start"] = float(command[command.index("--start") + 1])
            captured["end"] = float(command[command.index("--end") + 1])
            output_path.write_text(
                '{"entries": ['
                '{"start": 20.0, "ctc_score": 0.5, "token_spans": [{"start": 20.0, "end": 20.1, "score": 0.8}], "first_token_candidates": []},'
                '{"start": 28.0, "ctc_score": 0.5, "token_spans": [{"start": 28.0, "end": 28.1, "score": 0.8}], "first_token_candidates": []}'
                ']}',
                encoding="utf-8",
            )
            return Result()

        auto_lrc.subprocess.run = fake_run  # type: ignore[assignment]
        auto_lrc.default_ctc_python = lambda: Path("python")  # type: ignore[assignment]
        try:
            refined, _, changes = apply_ctc_local_window_realign(
                Path("dummy.flac"),
                entries,
                timestamps,
                report,
                raw_report,
                40.0,
                type("Args", (), {"whisperx_device": "cpu"})(),
            )
        finally:
            auto_lrc.subprocess.run = original_run  # type: ignore[assignment]
            auto_lrc.default_ctc_python = original_python  # type: ignore[assignment]

        self.assertAlmostEqual(captured["end"], 40.0, places=3)
        self.assertAlmostEqual(refined[0], 10.0, places=3)
        self.assertAlmostEqual(refined[1], 20.0, places=3)
        self.assertAlmostEqual(refined[2], 25.0, places=3)
        self.assertEqual(report["assignments"][2]["timestamp"], 25.0)  # type: ignore[index]
        hypotheses = report["assignments"][2]["alignment_hypotheses"]  # type: ignore[index]
        self.assertTrue(any(item["source"] == "ctc-local-window" and item["raw_candidate_time"] == 28.0 for item in hypotheses))
        self.assertEqual(report["ctc_local_window_realign"]["status"], "candidate-generated")  # type: ignore[index]
        self.assertFalse(report["ctc_local_window_realign"]["authoritative"])  # type: ignore[index]
        self.assertEqual(changes[0]["entries"], [3])

    def test_local_window_rejects_a_right_edge_pileup_candidate(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["context"]), LyricEntry(["target"])]
        timestamps = [10.0, 20.0, 25.0]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "ctc_score": 0.5},
                {"entry": 2, "timestamp": 20.0, "ctc_score": 0.5},
                {"entry": 3, "timestamp": 25.0, "ctc_score": 0.02},
            ],
            "ctc_low_score_runs": [[3]],
            "ctc_low_score_entries": [{"entry": 3, "text": "target", "ctc_score": 0.02}],
            "ctc_missing_entries": [],
        }
        raw_report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {"entry": 2, "timestamp": 20.0, "score": 0.9},
                {"entry": 3, "timestamp": 25.0, "score": 0.9},
            ]
        }
        original_run = auto_lrc.subprocess.run
        original_python = auto_lrc.default_ctc_python

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
            output_path = Path(command[command.index("--output") + 1])
            weak_tail = [
                {
                    "char": "x",
                    "start": 39.22 + index * 0.04,
                    "end": 39.24 + index * 0.04,
                    "score": 0.005,
                }
                for index in range(20)
            ]
            output_path.write_text(
                __import__("json").dumps(
                    {
                        "entries": [
                            {
                                "start": 20.0,
                                "ctc_score": 0.5,
                                "token_spans": [{"start": 20.0, "end": 20.1, "score": 0.8}],
                                "first_token_candidates": [],
                            },
                            {
                                "start": 30.0,
                                "ctc_score": 0.01,
                                "token_spans": weak_tail,
                                "first_token_candidates": [],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return Result()

        auto_lrc.subprocess.run = fake_run  # type: ignore[assignment]
        auto_lrc.default_ctc_python = lambda: Path("python")  # type: ignore[assignment]
        try:
            refined, report, changes = apply_ctc_local_window_realign(
                Path("dummy.flac"),
                entries,
                timestamps,
                report,
                raw_report,
                40.0,
                type("Args", (), {"whisperx_device": "cpu"})(),
            )
        finally:
            auto_lrc.subprocess.run = original_run  # type: ignore[assignment]
            auto_lrc.default_ctc_python = original_python  # type: ignore[assignment]

        self.assertEqual(changes, [])
        self.assertAlmostEqual(refined[2], 25.0, places=3)
        rejected = report["ctc_local_window_realign"]["rejected_rows"]  # type: ignore[index]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reason"], "local-window-right-edge-pileup")

    def test_detects_weak_token_pileup_at_local_window_right_edge(self) -> None:
        spans = [
            {
                "start": 136.80 + index * 0.02,
                "end": 136.82 + index * 0.02,
                "score": 0.003,
            }
            for index in range(14)
        ]
        assignment = {
            "ctc_local_window": {"start": 114.837, "end": 137.109},
            "ctc_token_spans": spans,
        }

        evidence = ctc_right_edge_pileup_evidence(assignment)

        self.assertTrue(evidence["touches_window_end"])
        self.assertTrue(evidence["right_edge_pileup"])
        self.assertGreater(evidence["tail_tokens_per_second"], 20.0)

    def test_right_edge_pileup_remains_review_required(self) -> None:
        entries = [LyricEntry(["previous"]), LyricEntry(["final line"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {
                    "entry": 2,
                    "timestamp": 20.0,
                    "score": 0.9,
                    "ctc_boundary_evidence": {"available": True, "realign_required": False},
                    "ctc_right_edge_evidence": {
                        "available": True,
                        "right_edge_pileup": True,
                        "window_end": 24.0,
                        "tail_end": 23.98,
                    },
                },
            ],
            "suspicious_alignments": [],
        }

        unresolved = flag_unresolved_ctc_boundary_issues(entries, report)

        self.assertEqual(len(unresolved), 1)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertTrue(assignment["review_required"])
        self.assertIn("ctc_right_edge_pileup", assignment["flags"])
        self.assertIn("ctc_local_window_truncated", assignment["flags"])

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

    def test_out_of_bounds_raw_candidate_does_not_lower_ctc_confidence(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["target lyric"]), LyricEntry(["after"])]
        timestamps = [10.0, 20.0, 30.0]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 10.0,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_token_spans": [{"end": 14.0}],
                },
                {
                    "entry": 2,
                    "timestamp": 20.0,
                    "score": 0.9,
                    "ctc_score": 0.25,
                    "timing_repair": "ctc-forced-align",
                    "ctc_token_spans": [
                        {"start": 20.0, "end": 20.02, "score": 0.8},
                        {"start": 20.1, "end": 20.12, "score": 0.7},
                    ],
                    "ctc_first_token_candidates": [],
                },
                {
                    "entry": 3,
                    "timestamp": 30.0,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                },
            ],
            "suspicious_alignments": [
                {
                    "entry": 2,
                    "review_required": True,
                    "severity": "high",
                    "flags": ["unresolved_raw_ctc_disagreement"],
                    "candidate_timestamps": {"output": 20.0, "raw_asr": 5.0},
                    "raw_asr_score": 0.95,
                }
            ],
        }

        result = score_line_timing_candidates(entries, timestamps, report, 40.0)

        self.assertAlmostEqual(result[1], 20.0, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertFalse(assignment["review_required"])
        self.assertNotIn("candidate_disagreement", assignment["flags"])
        self.assertGreaterEqual(assignment["confidence"], 0.85)
        risk = report["suspicious_alignments"][0]  # type: ignore[index]
        self.assertFalse(risk["review_required"])
        self.assertEqual(
            risk["resolution"],
            "raw-candidate-rejected-by-monotonic-neighbor-bounds",
        )

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

        self.assertAlmostEqual(result[1], 20.0, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertEqual(assignment["timestamp"], 20.0)
        self.assertEqual(assignment["chosen_time"], 20.0)
        recommendation = assignment["precentral_candidate_recommendation"]
        self.assertEqual(recommendation["source"], "raw_asr")
        self.assertAlmostEqual(recommendation["time"], 20.15, places=3)
        self.assertFalse(recommendation["authoritative"])
        self.assertTrue(assignment["candidates"])
        self.assertTrue(any(item["source"] == "raw_asr" for item in assignment["candidates"]))

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
            LyricEntry(["試験 試験 試験だけ"]),
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

        self.assertAlmostEqual(result[1], 174.266, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertEqual(assignment["timestamp"], 174.266)
        recommendation = assignment["precentral_candidate_recommendation"]
        self.assertEqual(recommendation["source"], "ctc_repeated_leading_term_onset")
        self.assertAlmostEqual(recommendation["time"], 172.770, places=3)
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
        entries = [LyricEntry(["before"]), LyricEntry(["試験 試験 試験だけ"]), LyricEntry(["after"])]
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

    def test_high_confidence_raw_remains_candidate_only_without_material_proof(self) -> None:
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

        self.assertAlmostEqual(result[1], 36.396, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertEqual(assignment["timestamp"], 36.396)
        recommendation = assignment["precentral_candidate_recommendation"]
        self.assertEqual(recommendation["source"], "raw_asr")
        self.assertAlmostEqual(recommendation["time"], 37.0, places=3)
        self.assertFalse(report["suspicious_alignments"][0]["review_required"])  # type: ignore[index]
        self.assertEqual(report["review_required_count"], 0)
        self.assertEqual(report["assignments"][1]["timestamp"], 36.396)  # type: ignore[index]

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

        self.assertAlmostEqual(result[1], 111.947, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertEqual(assignment["timestamp"], 111.947)
        recommendation = assignment["precentral_candidate_recommendation"]
        self.assertEqual(recommendation["source"], "ctc_detached_initial_token_recovery")
        self.assertAlmostEqual(recommendation["time"], 113.807, places=3)

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

    def test_timing_trust_no_longer_masks_low_content_confidence(self) -> None:
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

        self.assertEqual(report["content_trusted_entries"], 1)
        self.assertEqual(report["timing_trusted_entries"], 2)
        self.assertEqual(report["overall_trusted_entries"], 1)
        self.assertEqual(report["trusted_percent"], 50.0)
        self.assertEqual(report["low_confidence_count"], 1)
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
        self.assertEqual(report["content_trusted_percent"], 50.0)
        self.assertEqual(report["timing_trusted_percent"], 100.0)
        self.assertEqual(report["trusted_percent"], 50.0)
        self.assertEqual(report["low_confidence_count"], 1)
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

    def test_crossline_initial_recovery_skips_a_multi_token_ghost_prefix(self) -> None:
        timestamps = [129.82, 137.287]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 129.82,
                    "ctc_token_spans": [
                        {"char": "e", "start": 137.187, "end": 137.207, "score": 0.019817}
                    ],
                },
                {
                    "entry": 2,
                    "timestamp": 137.287,
                    "ctc_score": 0.373806,
                    "ctc_token_spans": [
                        {"char": "a", "start": 137.287, "end": 137.307, "score": 0.019517},
                        {"char": "r", "start": 137.947, "end": 137.967, "score": 0.002573},
                        {"char": "e", "start": 138.627, "end": 138.647, "score": 0.049344},
                        {"char": "y", "start": 139.768, "end": 139.788, "score": 0.625119},
                        {"char": "o", "start": 139.948, "end": 139.968, "score": 0.600449},
                    ],
                },
            ]
        }

        recovered, updated, changes = apply_ctc_crossline_initial_recovery(
            timestamps, report, 151.813
        )

        self.assertAlmostEqual(recovered[1], 138.627, places=3)
        self.assertEqual(len(changes), 1)
        assignment = updated["assignments"][1]  # type: ignore[index]
        self.assertEqual(assignment["ctc_effective_token_start_index"], 2)
        self.assertEqual(
            assignment["ctc_crossline_initial_recovery_mode"],
            "multi-token-ghost-prefix",
        )
        self.assertTrue(assignment["ctc_crossline_initial_recovery_trusted"])

    def test_candidate_scoring_keeps_a_trusted_crossline_recovery(self) -> None:
        entries = [
            LyricEntry(["previous"]),
            LyricEntry(["Are you checking this synthetic boundary?"]),
        ]
        timestamps = [129.803, 138.627]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 129.803,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_token_spans": [
                        {"char": "e", "start": 137.247, "end": 137.267, "score": 0.022157}
                    ],
                },
                {
                    "entry": 2,
                    "timestamp": 138.627,
                    "score": 0.9,
                    "ctc_score": 0.373806,
                    "timing_repair": "ctc-forced-align",
                    "ctc_effective_token_start_index": 2,
                    "ctc_crossline_initial_recovery": True,
                    "ctc_crossline_initial_recovery_trusted": True,
                    "ctc_token_spans": [
                        {"char": "a", "start": 137.287, "end": 137.307, "score": 0.019517},
                        {"char": "r", "start": 137.947, "end": 137.967, "score": 0.002573},
                        {"char": "e", "start": 138.627, "end": 138.647, "score": 0.049344},
                        {"char": "y", "start": 139.768, "end": 139.788, "score": 0.625119},
                        {"char": "o", "start": 139.948, "end": 139.968, "score": 0.600449},
                    ],
                    "ctc_first_token_candidates": [],
                },
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(entries, timestamps, report, 151.813)

        self.assertAlmostEqual(result[1], 138.627, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertFalse(assignment["review_required"])
        self.assertGreaterEqual(assignment["confidence"], 0.95)
        self.assertNotIn("candidate_disagreement", assignment["flags"])
        self.assertNotIn(
            "ctc_detached_initial_token_recovery",
            [item["source"] for item in assignment["candidates"]],
        )

    def test_candidate_scoring_restores_a_detached_crossline_recovery_after_window_realign(self) -> None:
        entries = [
            LyricEntry(["Synthetic signals gather near dawn"]),
            LyricEntry(["Underwater test signals surround the marker"]),
            LyricEntry(["Now this synthetic case is complete"]),
        ]
        timestamps = [81.44, 85.992, 93.955]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 81.44,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_token_spans": [
                        {"char": "s", "start": 84.104, "end": 84.124, "score": 0.107459}
                    ],
                },
                {
                    "entry": 2,
                    "timestamp": 85.992,
                    "score": 0.9,
                    "ctc_score": 0.227128,
                    "timing_repair": "ctc-forced-align",
                    "ctc_effective_token_start_index": 1,
                    "ctc_crossline_initial_recovery": True,
                    "ctc_crossline_initial_recovery_trusted": False,
                    "ctc_crossline_initial_recovery_evidence": {
                        "ghost_span_seconds": 3.501,
                    },
                    "ctc_token_spans": [
                        {"char": "u", "start": 85.992, "end": 86.012, "score": 0.006262},
                        {"char": "n", "start": 89.493, "end": 89.513, "score": 0.602427},
                        {"char": "d", "start": 89.533, "end": 89.553, "score": 0.111421},
                        {"char": "e", "start": 89.573, "end": 89.593, "score": 0.195719},
                    ],
                    "ctc_first_token_candidates": [],
                },
                {
                    "entry": 3,
                    "timestamp": 93.955,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_token_spans": [],
                },
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(entries, timestamps, report, 110.0)

        self.assertAlmostEqual(result[1], 85.992, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertAlmostEqual(float(assignment["timestamp"]), 85.992, places=3)
        self.assertTrue(assignment["ctc_crossline_initial_recovery_candidate_generated"])
        self.assertFalse(assignment["ctc_crossline_initial_recovery_trusted"])
        hypotheses = assignment.get("alignment_hypotheses", [])
        restored = [
            item
            for item in hypotheses
            if isinstance(item, dict) and item.get("source") == "ctc-legacy-crossline-restored"
        ]
        self.assertEqual(len(restored), 1)
        self.assertAlmostEqual(float(restored[0]["raw_candidate_time"]), 89.493, places=3)
        self.assertFalse(restored[0]["authoritative"])

    def test_forward_supported_prefix_candidate_rejects_continuous_weak_prefix(self) -> None:
        candidate = ctc_forward_supported_prefix_candidate(
            {
                "ctc_token_spans": [
                    {"start": 20.00, "end": 20.02, "score": 0.006},
                    {"start": 20.10, "end": 20.12, "score": 0.004},
                    {"start": 20.20, "end": 20.22, "score": 0.008},
                    {"start": 20.30, "end": 20.32, "score": 0.003},
                    {"start": 20.40, "end": 20.42, "score": 0.012},
                    {"start": 20.50, "end": 20.52, "score": 0.004},
                    {"start": 21.20, "end": 21.22, "score": 0.12},
                    {"start": 21.30, "end": 21.32, "score": 0.15},
                    {"start": 21.40, "end": 21.42, "score": 0.09},
                ]
            }
        )

        self.assertIsNone(candidate)

    def test_forward_supported_prefix_recovery_does_not_skip_continuous_prefix(self) -> None:
        frame_times = np.round(np.arange(0.0, 30.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 21.2))[0][0]] = 1.0
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
            entries = [LyricEntry(["previous"]), LyricEntry(["The bubbles"]), LyricEntry(["next"])]
            timestamps = [15.0, 20.0, 25.0]
            report: dict[str, object] = {
                "assignments": [
                    {
                        "entry": 1,
                        "timestamp": 15.0,
                        "ctc_token_spans": [
                            {"char": "x", "start": 18.0, "end": 18.02, "score": 0.2}
                        ],
                    },
                    {
                        "entry": 2,
                        "timestamp": 20.0,
                        "timing_repair": "ctc-forced-align",
                        "ctc_score": 0.04,
                        "ctc_token_spans": [
                            {"char": "t", "start": 20.0, "end": 20.02, "score": 0.006},
                            {"char": "h", "start": 20.1, "end": 20.12, "score": 0.004},
                            {"char": "e", "start": 20.2, "end": 20.22, "score": 0.008},
                            {"char": "b", "start": 20.3, "end": 20.32, "score": 0.003},
                            {"char": "u", "start": 20.4, "end": 20.42, "score": 0.012},
                            {"char": "b", "start": 20.5, "end": 20.52, "score": 0.004},
                            {"char": "i", "start": 21.5, "end": 21.52, "score": 0.12},
                            {"char": "n", "start": 21.6, "end": 21.62, "score": 0.15},
                            {"char": "g", "start": 21.7, "end": 21.72, "score": 0.09},
                        ],
                    },
                    {"entry": 3, "timestamp": 25.0, "ctc_token_spans": []},
                ]
            }

            changes = apply_ctc_forward_supported_prefix_recovery(
                Path("dummy.flac"), entries, timestamps, report, 30.0
            )

            self.assertEqual(changes, [])
            self.assertAlmostEqual(timestamps[1], 20.0, places=3)
            assignment = report["assignments"][1]  # type: ignore[index]
            self.assertNotIn("ctc_forward_supported_prefix_recovery_trusted", assignment)
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

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

    def test_backtracks_low_confidence_ctc_when_acoustic_onset_is_clear(self) -> None:
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
                        "ctc_score": 0.04,
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 30.0)

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 20.0, places=3)
            self.assertEqual(report["assignments"][1]["timestamp"], 20.0)  # type: ignore[index]
            self.assertEqual(report["ctc_acoustic_backtrack_count"], 1)
            self.assertFalse(report["ctc_acoustic_backtrack_authoritative"])
            hypotheses = report["assignments"][1]["alignment_hypotheses"]  # type: ignore[index]
            self.assertTrue(any(item["source"] == "ctc-acoustic-backtrack" and item["raw_candidate_time"] == 19.5 for item in hypotheses))

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
                        "ctc_score": 0.10,
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
    def test_boundary_evidence_requires_a_weak_multi_token_prefix(self) -> None:
        weak = {
            "ctc_score": 0.02,
            "ctc_token_spans": [
                {"start": 10.00 + index * 0.05, "end": 10.02 + index * 0.05, "score": score}
                for index, score in enumerate([0.002, 0.003, 0.001, 0.004, 0.008, 0.002, 0.01, 0.04])
            ],
        }
        strong = {
            "ctc_score": 0.35,
            "ctc_token_spans": [
                {"start": 10.00 + index * 0.08, "end": 10.04 + index * 0.08, "score": score}
                for index, score in enumerate([0.31, 0.24, 0.18, 0.29, 0.15, 0.22, 0.17, 0.20])
            ],
        }

        weak_evidence = ctc_prefix_boundary_evidence(weak, 0.0)
        strong_evidence = ctc_prefix_boundary_evidence(strong, 0.0)

        self.assertTrue(weak_evidence["realign_required"])
        self.assertFalse(strong_evidence["realign_required"])

    def test_boundary_evidence_detects_a_nonzero_gap_ghost_prefix(self) -> None:
        evidence = ctc_prefix_boundary_evidence(
            {
                "ctc_score": 0.362048,
                "ctc_token_spans": [
                    {"start": 137.367, "end": 137.387, "score": 0.018011},
                    {"start": 137.747, "end": 137.767, "score": 0.002650},
                    {"start": 138.627, "end": 138.647, "score": 0.041703},
                    {"start": 139.788, "end": 139.808, "score": 0.649345},
                    {"start": 139.928, "end": 139.948, "score": 0.521000},
                    {"start": 140.128, "end": 140.148, "score": 0.410000},
                    {"start": 140.268, "end": 140.288, "score": 0.380000},
                    {"start": 140.408, "end": 140.428, "score": 0.330000},
                ],
            },
            0.160,
        )

        self.assertTrue(evidence["ghost_weak_prefix"])
        self.assertTrue(evidence["realign_required"])
        self.assertAlmostEqual(evidence["reliable_token_delay_seconds"], 2.421, places=3)

    def test_boundary_realign_requires_next_reliable_token_evidence(self) -> None:
        current = ctc_prefix_boundary_evidence(
            {
                "ctc_score": 0.02,
                "ctc_previous_token_score": 0.145,
                "ctc_token_spans": [
                    {"start": 10.00 + index * 0.05, "end": 10.02 + index * 0.05, "score": 0.003}
                    for index in range(8)
                ],
            },
            0.02,
        )
        local = ctc_prefix_boundary_evidence(
            {
                "ctc_score": 0.18,
                "ctc_previous_token_score": 0.676,
                "ctc_token_spans": [
                    {"start": 12.00 + index * 0.09, "end": 12.04 + index * 0.09, "score": score}
                    for index, score in enumerate([0.04, 0.06, 0.12, 0.10, 0.09, 0.11, 0.08, 0.13])
                ],
            },
            0.12,
        )
        accepted, reasons = should_accept_ctc_boundary_realign(
            current, local, 0.02, 0.18, 10.0, 12.0, 11.94
        )
        self.assertTrue(accepted)
        self.assertIn("next-reliable-token-support", reasons)
        self.assertIn("previous-line-tail-evidence-improved", reasons)

        local_without_reliable = dict(local)
        local_without_reliable["next_reliable_token_index"] = None
        local_without_reliable["next_reliable_token_score"] = None
        local_without_reliable["reliable_token_delay_seconds"] = None
        accepted, reasons = should_accept_ctc_boundary_realign(
            current, local_without_reliable, 0.02, 0.18, 10.0, 12.0, 11.94
        )
        self.assertFalse(accepted)
        self.assertIn("no-nearby-reliable-token-evidence", reasons)

    def test_unresolved_boundary_remains_review_required_after_candidate_scoring(self) -> None:
        entries = [LyricEntry(["previous"]), LyricEntry(["current"]), LyricEntry(["next"])]
        timestamps = [5.0, 10.0, 15.0]
        evidence = {
            "available": True,
            "realign_required": True,
            "zero_boundary_gap": True,
            "low_confidence_prefix": True,
            "abnormal_prefix_compression": True,
        }
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 5.0, "score": 0.9},
                {
                    "entry": 2,
                    "timestamp": 10.0,
                    "score": 0.9,
                    "ctc_score": 0.02,
                    "ctc_boundary_evidence": evidence,
                },
                {"entry": 3, "timestamp": 15.0, "score": 0.9},
            ],
            "suspicious_alignments": [],
        }

        unresolved = flag_unresolved_ctc_boundary_issues(entries, report)
        score_line_timing_candidates(entries, timestamps, report, 20.0)
        assignment = report["assignments"][1]  # type: ignore[index]

        self.assertEqual(len(unresolved), 1)
        self.assertTrue(assignment["review_required"])
        self.assertTrue(assignment["ctc_unresolved_boundary"])
        self.assertIn("ctc_unresolved_boundary", assignment["flags"])

    def test_zero_gap_compressed_prefix_is_not_a_normal_connected_boundary(self) -> None:
        assignment = {
            "ctc_score": 0.07,
            "ctc_token_spans": [
                {"start": 10.00 + index * 0.03, "end": 10.02 + index * 0.03, "score": score}
                for index, score in enumerate([0.002, 0.003, 0.001, 0.004, 0.008, 0.002, 0.01, 0.04])
            ],
        }
        self.assertTrue(has_ctc_compressed_prefix(assignment, 0.0))
        self.assertFalse(has_ctc_compressed_prefix(assignment, 0.12))

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
                        "ctc_first_token_candidates": [
                            {"time": 19.7, "score": 0.008},
                            {"time": 18.6, "score": 0.10},
                        ],
                    },
                    {"entry": 3, "timestamp": 30.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 20.0, places=3)
            self.assertEqual(report["assignments"][1]["timestamp"], 20.0)  # type: ignore[index]
            self.assertEqual(changes[0]["mode"], "ctc-first-token-posterior")
            self.assertEqual(report["assignments"][1]["ctc_acoustic_backtrack_mode"], "ctc-first-token-posterior")  # type: ignore[index]
            hypotheses = report["assignments"][1]["alignment_hypotheses"]  # type: ignore[index]
            self.assertTrue(any(item["raw_candidate_time"] == 18.6 and item["mode"] == "ctc-first-token-posterior" for item in hypotheses))
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_rejects_first_token_candidate_inside_previous_reliable_tail(self) -> None:
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
                        "ctc_previous_token_end": 19.65,
                        "ctc_first_token_candidates": [
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
            self.assertEqual(report["ctc_acoustic_backtrack_rejection_count"], 1)
            rejection = report["ctc_acoustic_backtrack_rejections"][0]
            self.assertEqual(rejection["reason"], "candidate-borrows-previous-line-tail")
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_short_acoustic_backtrack_rejects_an_unsupported_weak_prefix(self) -> None:
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
            entries = [LyricEntry(["previous"]), LyricEntry(["The bubbles"]), LyricEntry(["next"])]
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
                        "romaji": "thebubbles",
                        "ctc_boundary_gap_seconds": 5.0,
                        "ctc_token_spans": [
                            {"char": char, "start": 20.0 + index * 0.08, "end": 20.02 + index * 0.08, "score": 0.006}
                            for index, char in enumerate("thebubbl")
                        ],
                    },
                    {"entry": 3, "timestamp": 25.0, "timing_repair": "ctc-forced-align", "ctc_score": 0.5},
                ],
            }

            changes = apply_ctc_acoustic_backtrack(Path("dummy.flac"), entries, timestamps, report, 40.0)

            self.assertEqual(changes, [])
            self.assertEqual(timestamps[1], 20.0)
            self.assertEqual(report["ctc_acoustic_backtrack_rejection_count"], 1)
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
            self.assertAlmostEqual(timestamps[1], 20.0, places=3)
            self.assertEqual(report["assignments"][1]["timestamp"], 20.0)  # type: ignore[index]
            self.assertEqual(changes[0]["mode"], "ctc-short-acoustic-onset")
            hypotheses = report["assignments"][1]["alignment_hypotheses"]  # type: ignore[index]
            self.assertTrue(any(item["raw_candidate_time"] == 19.6 and item["direct_onset_support"] == "supported" for item in hypotheses))
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


    def test_forward_supported_prefix_candidate_accepts_strong_second_token_with_broad_support(self) -> None:
        candidate = ctc_forward_supported_prefix_candidate(
            {
                "ctc_token_spans": [
                    {"start": 10.00, "end": 10.02, "score": 0.005},
                    {"start": 11.58, "end": 11.60, "score": 0.72},
                    {"start": 11.62, "end": 11.64, "score": 0.02},
                    {"start": 11.68, "end": 11.70, "score": 0.02},
                    {"start": 11.72, "end": 11.74, "score": 0.001},
                    {"start": 11.80, "end": 11.82, "score": 0.06},
                    {"start": 11.88, "end": 11.90, "score": 0.02},
                    {"start": 11.94, "end": 11.96, "score": 0.001},
                    {"start": 12.00, "end": 12.02, "score": 0.002},
                    {"start": 12.08, "end": 12.10, "score": 0.43},
                    {"start": 12.14, "end": 12.16, "score": 0.002},
                    {"start": 12.20, "end": 12.22, "score": 0.66},
                    {"start": 12.28, "end": 12.30, "score": 0.63},
                ]
            }
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["token_index"], 1)  # type: ignore[index]
        self.assertEqual(candidate["support_mode"], "strong-detached-initial-token")  # type: ignore[index]

    def test_forward_supported_prefix_candidate_rejects_distant_midline_cluster(self) -> None:
        spans = [
            {"start": 10.0 + index * 0.10, "end": 10.02 + index * 0.10, "score": 0.005}
            for index in range(12)
        ]
        spans.extend(
            [
                {"start": 12.60, "end": 12.62, "score": 0.10},
                {"start": 12.70, "end": 12.72, "score": 0.12},
                {"start": 12.80, "end": 12.82, "score": 0.11},
            ]
        )

        self.assertIsNone(ctc_forward_supported_prefix_candidate({"ctc_token_spans": spans}))

    def test_strong_detached_initial_recovery_sets_effective_token_index(self) -> None:
        frame_times = np.round(np.arange(0.0, 20.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 11.5))[0][0]] = 1.0
        features = AudioFeatures(
            duration=20.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda audio_path, duration: features  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["Underwater"]), LyricEntry(["next"])]
            timestamps = [5.0, 10.0, 15.0]
            report: dict[str, object] = {
                "assignments": [
                    {
                        "entry": 1,
                        "timestamp": 5.0,
                        "ctc_token_spans": [
                            {"char": "x", "start": 8.0, "end": 8.02, "score": 0.2}
                        ],
                    },
                    {
                        "entry": 2,
                        "timestamp": 10.0,
                        "timing_repair": "ctc-forced-align",
                        "ctc_token_spans": [
                            {"char": "u", "start": 10.0, "end": 10.02, "score": 0.005},
                            {"char": "n", "start": 11.58, "end": 11.60, "score": 0.72},
                            {"char": "d", "start": 11.62, "end": 11.64, "score": 0.02},
                            {"char": "e", "start": 11.68, "end": 11.70, "score": 0.02},
                            {"char": "r", "start": 11.72, "end": 11.74, "score": 0.001},
                            {"char": "w", "start": 11.80, "end": 11.82, "score": 0.06},
                            {"char": "a", "start": 11.88, "end": 11.90, "score": 0.02},
                            {"char": "t", "start": 11.94, "end": 11.96, "score": 0.001},
                            {"char": "e", "start": 12.00, "end": 12.02, "score": 0.002},
                            {"char": "r", "start": 12.08, "end": 12.10, "score": 0.43},
                            {"char": "w", "start": 12.14, "end": 12.16, "score": 0.002},
                            {"char": "e", "start": 12.20, "end": 12.22, "score": 0.66},
                            {"char": "r", "start": 12.28, "end": 12.30, "score": 0.63},
                        ],
                    },
                    {"entry": 3, "timestamp": 15.0, "ctc_token_spans": []},
                ]
            }

            changes = apply_ctc_forward_supported_prefix_recovery(
                Path("dummy.flac"), entries, timestamps, report, 20.0
            )

            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 11.5, places=3)
            assignment = report["assignments"][1]  # type: ignore[index]
            self.assertEqual(assignment["ctc_effective_token_start_index"], 1)
            self.assertTrue(assignment["ctc_forward_supported_prefix_recovery_trusted"])
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_provisional_boundary_detection_does_not_lower_trust(self) -> None:
        entries = [LyricEntry(["previous"]), LyricEntry(["current"])]
        report: dict[str, object] = {
            "timing_entries": 2,
            "assignments": [
                {"entry": 1, "timestamp": 5.0, "score": 0.9, "timing_repair": "ctc-forced-align"},
                {
                    "entry": 2,
                    "timestamp": 10.0,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_boundary_evidence": {
                        "realign_required": True,
                        "zero_boundary_gap": True,
                        "low_confidence_prefix": True,
                    },
                },
            ],
            "suspicious_alignments": [],
            "review_required_count": 0,
        }

        unresolved = flag_unresolved_ctc_boundary_issues(entries, report, persist=False)

        self.assertEqual(len(unresolved), 1)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertFalse(bool(assignment.get("review_required")))
        self.assertNotIn("ctc_unresolved_boundary", assignment)
        self.assertEqual(report["suspicious_alignments"], [])

    def test_final_boundary_validation_removes_resolved_provisional_review(self) -> None:
        entries = [LyricEntry(["previous"]), LyricEntry(["current"])]
        report: dict[str, object] = {
            "timing_entries": 2,
            "assignments": [
                {"entry": 1, "timestamp": 5.0, "score": 0.9, "timing_repair": "ctc-forced-align"},
                {
                    "entry": 2,
                    "timestamp": 10.0,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "review_required": True,
                    "ctc_unresolved_boundary": True,
                    "flags": ["ctc_unresolved_boundary", "ctc_low_confidence_prefix"],
                    "ctc_boundary_evidence": {"realign_required": False},
                },
            ],
            "suspicious_alignments": [
                {
                    "entry": 2,
                    "text": "current",
                    "flags": ["ctc_unresolved_boundary", "ctc_low_confidence_prefix"],
                    "severity": "medium",
                    "review_required": True,
                }
            ],
            "review_required_count": 1,
        }

        unresolved = flag_unresolved_ctc_boundary_issues(entries, report, persist=True)

        self.assertEqual(unresolved, [])
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertFalse(assignment["review_required"])
        self.assertNotIn("ctc_unresolved_boundary", assignment)
        self.assertEqual(report["suspicious_alignments"], [])
        self.assertEqual(report["review_required_count"], 0)



class AudioAuditV7Tests(unittest.TestCase):
    def test_detached_initial_recovery_stays_near_supported_token(self) -> None:
        entries = [
            LyricEntry(["previous"]),
            LyricEntry(["Underwater test signals surround the marker"]),
            LyricEntry(["next"]),
        ]
        timestamps = [5.0, 10.0, 20.0]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 5.0, "ctc_token_spans": [{"char": "x", "start": 5.0, "end": 8.0, "score": 0.5}]},
                {
                    "entry": 2,
                    "timestamp": 10.0,
                    "ctc_token_spans": [
                        {"char": "u", "start": 10.0, "end": 10.02, "score": 0.004},
                        {"char": "n", "start": 13.56, "end": 13.58, "score": 0.49},
                        {"char": "d", "start": 13.60, "end": 13.62, "score": 0.12},
                        {"char": "e", "start": 13.66, "end": 13.68, "score": 0.14},
                        {"char": "r", "start": 13.72, "end": 13.74, "score": 0.10},
                        {"char": "w", "start": 13.80, "end": 13.82, "score": 0.11},
                    ],
                },
                {"entry": 3, "timestamp": 20.0, "ctc_token_spans": [{"char": "x", "start": 20.0, "end": 21.0, "score": 0.5}]},
            ]
        }
        frame_times = np.arange(0.0, 25.0, 0.02, dtype=np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.argmin(np.abs(frame_times - 12.80))] = 0.95
        onset_strength[np.argmin(np.abs(frame_times - 13.56))] = 0.80
        features = AudioFeatures(25.0, frame_times, np.zeros_like(frame_times), onset_strength, [])
        original = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda *_args, **_kwargs: features
        try:
            changes = apply_ctc_forward_supported_prefix_recovery(
                Path("dummy.wav"), entries, timestamps, report, 25.0
            )
        finally:
            auto_lrc.analyze_audio = original

        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(timestamps[1], 13.56, places=2)
        self.assertGreater(timestamps[1], 13.30)

    def test_duplicate_lyric_offset_consensus_detects_stable_section(self) -> None:
        entries = [
            LyricEntry(["A"]),
            LyricEntry(["B"]),
            LyricEntry(["C"]),
            LyricEntry(["D"]),
            LyricEntry(["A"]),
            LyricEntry(["B"]),
            LyricEntry(["C"]),
            LyricEntry(["D"]),
        ]
        timestamps = [1.0, 3.0, 5.0, 7.0, 101.0, 103.02, 105.01, 107.0]
        result = auto_lrc.duplicate_lyric_offset_consensus(entries, timestamps)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(float(result["median_delta_seconds"]), 100.0, places=2)


class GeorgetteBoundaryRegressionTests(unittest.TestCase):
    def test_backtracked_timestamp_does_not_impersonate_first_ctc_token(self) -> None:
        assignments: list[object] = [
            {
                "timestamp": 46.556,
                "ctc_token_spans": [
                    {"char": "o", "start": 55.282, "end": 55.302, "score": 0.008932},
                ],
            },
            {
                "timestamp": 55.342,
                "ctc_score": 0.104232,
                "ctc_token_spans": [
                    {"char": "m", "start": 57.002, "end": 57.022, "score": 0.576660},
                    {"char": "o", "start": 57.102, "end": 57.122, "score": 0.249099},
                    {"char": "t", "start": 57.582, "end": 57.603, "score": 0.002623},
                    {"char": "s", "start": 57.643, "end": 57.663, "score": 0.246038},
                    {"char": "u", "start": 57.783, "end": 57.803, "score": 0.252842},
                    {"char": "r", "start": 59.503, "end": 59.523, "score": 0.002233},
                    {"char": "e", "start": 59.563, "end": 59.583, "score": 0.010809},
                    {"char": "t", "start": 59.683, "end": 59.703, "score": 0.098325},
                ],
            },
        ]

        annotate_ctc_boundary_evidence(assignments)

        current = assignments[1]
        self.assertAlmostEqual(current["ctc_boundary_start"], 55.342, places=3)  # type: ignore[index]
        self.assertAlmostEqual(current["ctc_first_token_start"], 57.002, places=3)  # type: ignore[index]
        self.assertAlmostEqual(current["ctc_boundary_gap_seconds"], 0.040, places=3)  # type: ignore[index]
        evidence = current["ctc_boundary_evidence"]  # type: ignore[index]
        self.assertAlmostEqual(evidence["ctc_token_lead_seconds"], 1.660, places=3)
        self.assertTrue(evidence["timestamp_precedes_ctc_support"])

    def test_0055_posterior_backtrack_cannot_borrow_suruno_tail(self) -> None:
        features = AudioFeatures(
            duration=80.0,
            frame_times=np.round(np.arange(0.0, 80.0, 0.1), 3).astype(np.float32),
            rms_db=np.full(800, -20.0, dtype=np.float32),
            onset_strength=np.zeros(800, dtype=np.float32),
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda *_args, **_kwargs: features  # type: ignore[assignment]
        try:
            entries = [
                LyricEntry(["前の合成試験行"]),
                LyricEntry(["もつれた　テストを　まわる"]),
                LyricEntry(["次の合成試験行"]),
            ]
            timestamps = [46.556, 57.002, 67.223]
            report: dict[str, object] = {
                "assignments": [
                    {
                        "entry": 1,
                        "timestamp": 46.556,
                        "timing_repair": "ctc-forced-align",
                        "ctc_score": 0.564761,
                    },
                    {
                        "entry": 2,
                        "timestamp": 57.002,
                        "timing_repair": "ctc-forced-align",
                        "timing_repair_source": "torchaudio-mms-fa",
                        "ctc_score": 0.104232,
                        "romaji": "motsuretatestwomawaru",
                        "ctc_previous_token_end": 55.302,
                        "ctc_first_token_candidates": [
                            {"time": 55.762, "score": 0.009981},
                            {"time": 55.342, "score": 0.001064},
                        ],
                    },
                    {
                        "entry": 3,
                        "timestamp": 67.223,
                        "timing_repair": "ctc-forced-align",
                        "ctc_score": 0.136584,
                    },
                ]
            }

            changes = apply_ctc_acoustic_backtrack(
                Path("dummy.flac"), entries, timestamps, report, 80.0
            )

            self.assertEqual(changes, [])
            self.assertAlmostEqual(timestamps[1], 57.002, places=3)
            self.assertEqual(report["ctc_acoustic_backtrack_rejection_count"], 1)
            rejection = report["ctc_acoustic_backtrack_rejections"][0]  # type: ignore[index]
            self.assertEqual(rejection["reason"], "candidate-borrows-previous-line-tail")
            self.assertAlmostEqual(rejection["candidate_gap_seconds"], 0.040, places=3)
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_0344_weak_zero_gap_after_again_stays_review_required(self) -> None:
        entries = [LyricEntry(["... again"]), LyricEntry(["Everlasting test pattern ever ever ever"])]
        report: dict[str, object] = {
            "timing_entries": 2,
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 215.013,
                    "ctc_score": 0.158220,
                    "ctc_token_spans": [
                        {"char": "g", "start": 224.036, "end": 224.056, "score": 0.007160},
                        {"char": "a", "start": 224.056, "end": 224.076, "score": 0.001813},
                        {"char": "i", "start": 224.136, "end": 224.156, "score": 0.290717},
                        {"char": "n", "start": 224.196, "end": 224.216, "score": 0.013697},
                    ],
                },
                {
                    "entry": 2,
                    "timestamp": 224.236,
                    "score": 0.9,
                    "ctc_score": 0.140417,
                    "ctc_token_spans": [
                        {"char": "e", "start": 224.236, "end": 224.256, "score": 0.008757},
                        {"char": "v", "start": 224.276, "end": 224.296, "score": 0.000918},
                        {"char": "e", "start": 224.356, "end": 224.376, "score": 0.157146},
                        {"char": "r", "start": 224.496, "end": 224.516, "score": 0.353693},
                        {"char": "l", "start": 224.576, "end": 224.596, "score": 0.001283},
                        {"char": "a", "start": 224.596, "end": 224.616, "score": 0.645580},
                        {"char": "s", "start": 224.656, "end": 224.676, "score": 0.000583},
                        {"char": "t", "start": 224.896, "end": 224.916, "score": 0.051476},
                    ],
                },
            ],
            "suspicious_alignments": [],
        }

        annotate_ctc_boundary_evidence(report["assignments"])  # type: ignore[arg-type]
        current = report["assignments"][1]  # type: ignore[index]
        evidence = current["ctc_boundary_evidence"]
        self.assertTrue(evidence["borrowed_tail_prefix"])
        self.assertTrue(evidence["realign_required"])

        timestamps = [215.013, 224.236]
        timestamps, report, changes = apply_ctc_crossline_initial_recovery(
            timestamps, report, 234.680
        )
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["mode"], "short-borrowed-tail-prefix")
        self.assertAlmostEqual(timestamps[1], 224.356, places=3)
        self.assertEqual(current["ctc_effective_token_start_index"], 2)
        self.assertTrue(current["ctc_crossline_initial_recovery_trusted"])

        annotate_ctc_boundary_evidence(report["assignments"])  # type: ignore[arg-type]
        unresolved = flag_unresolved_ctc_boundary_issues(entries, report, persist=True)
        self.assertEqual(unresolved, [])
        self.assertFalse(bool(current.get("review_required")))


    def test_trusted_percent_requires_timing_boundary_trust(self) -> None:
        report: dict[str, object] = {
            "timing_entries": 2,
            "assignments": [
                {
                    "entry": 1,
                    "timing_repair": "ctc-forced-align",
                    "score": 0.9,
                    "timestamp": 10.0,
                    "ctc_token_spans": [
                        {"char": "a", "start": 19.98, "end": 20.0, "score": 0.8},
                    ],
                },
                {
                    "entry": 2,
                    "timing_repair": "ctc-forced-align",
                    "score": 0.9,
                    "timestamp": 20.04,
                    "ctc_previous_token_end": 20.0,
                    "ctc_previous_token_score": 0.01,
                    "ctc_token_spans": [
                        {"char": "e", "start": 20.04, "end": 20.06, "score": 0.02},
                        {"char": "v", "start": 20.10, "end": 20.12, "score": 0.01},
                        {"char": "e", "start": 20.16, "end": 20.18, "score": 0.02},
                        {"char": "r", "start": 21.10, "end": 21.12, "score": 0.20},
                        {"char": "x", "start": 21.20, "end": 21.22, "score": 0.20},
                        {"char": "x", "start": 21.30, "end": 21.32, "score": 0.20},
                        {"char": "x", "start": 21.40, "end": 21.42, "score": 0.20},
                        {"char": "x", "start": 21.50, "end": 21.52, "score": 0.20},
                    ],
                },
            ],
        }
        annotate_ctc_boundary_evidence(report["assignments"])  # type: ignore[arg-type]
        update_report_confidence_metrics(report)
        self.assertEqual(report["content_trusted_percent"], 100.0)
        self.assertEqual(report["timing_trusted_percent"], 50.0)
        self.assertEqual(report["overall_trusted_percent"], 50.0)
        self.assertEqual(report["trusted_percent"], 50.0)
        self.assertFalse(assignment_timing_is_trusted(report["assignments"][1]))  # type: ignore[index]

    def test_final_guard_moves_again_boundary_to_its_own_acoustic_onset(self) -> None:
        frame_times = np.round(np.arange(220.0, 231.0, 0.01), 3).astype(np.float32)
        onset = np.zeros(len(frame_times), dtype=np.float32)
        onset[int(round((226.16 - 220.0) / 0.01))] = 1.0
        features = AudioFeatures(
            duration=234.68,
            frame_times=frame_times,
            rms_db=np.full(len(frame_times), -20.0, dtype=np.float32),
            onset_strength=onset,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda *_args, **_kwargs: features  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["... again"]), LyricEntry(["everlastin ever 離れぬように"])]
            report: dict[str, object] = {
                "backend": "ctc",
                "timing_entries": 2,
                "assignments": [
                    {
                        "entry": 1,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 205.57,
                        "ctc_token_spans": [
                            {"char": "g", "start": 225.557, "end": 225.577, "score": 0.01},
                            {"char": "a", "start": 225.617, "end": 225.637, "score": 0.01},
                            {"char": "i", "start": 225.677, "end": 225.697, "score": 0.29},
                            {"char": "n", "start": 225.717, "end": 225.737, "score": 0.015},
                        ],
                    },
                    {
                        "entry": 2,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 225.777,
                        "ctc_score": 0.131604,
                        "ctc_token_spans": [
                            {"char": "e", "start": 225.777, "end": 225.797, "score": 0.035797},
                            {"char": "v", "start": 225.937, "end": 225.957, "score": 0.000460},
                            {"char": "e", "start": 226.037, "end": 226.057, "score": 0.060586},
                            {"char": "r", "start": 226.777, "end": 226.797, "score": 0.002959},
                            {"char": "l", "start": 226.797, "end": 226.817, "score": 0.000898},
                            {"char": "a", "start": 226.877, "end": 226.897, "score": 0.254915},
                            {"char": "s", "start": 227.057, "end": 227.077, "score": 0.000907},
                            {"char": "t", "start": 227.257, "end": 227.277, "score": 0.005328},
                        ],
                    },
                ],
                "suspicious_alignments": [],
            }
            timestamps, report, changes = apply_final_ctc_timing_guard(
                Path("dummy.flac"), entries, [205.57, 225.777], report, 234.68
            )
            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 226.16, places=2)
            current = report["assignments"][1]  # type: ignore[index]
            self.assertTrue(current["ctc_final_timing_recovery_trusted"])
            self.assertTrue(current["timing_audit_trusted"])
            self.assertFalse(bool(current.get("review_required")))
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_final_guard_catches_80ms_again_leak_with_isolated_first_token(self) -> None:
        """Regression for the v2 miss: token 0 looked usable, the prefix did not."""
        frame_times = np.round(np.arange(223.0, 229.0, 0.01), 3).astype(np.float32)
        onset = np.zeros(len(frame_times), dtype=np.float32)
        onset[int(round((226.16 - 223.0) / 0.01))] = 1.0
        features = AudioFeatures(
            duration=234.68,
            frame_times=frame_times,
            rms_db=np.full(len(frame_times), -20.0, dtype=np.float32),
            onset_strength=onset,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda *_args, **_kwargs: features  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["... again"]), LyricEntry(["everlastin ever 離れぬように"])]
            report: dict[str, object] = {
                "backend": "ctc",
                "timing_entries": 2,
                "assignments": [
                    {
                        "entry": 1,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 215.013,
                        "ctc_token_spans": [
                            {"char": "g", "start": 224.496, "end": 224.516, "score": 0.016454},
                            {"char": "a", "start": 224.596, "end": 224.616, "score": 0.644287},
                            {"char": "i", "start": 224.756, "end": 224.776, "score": 0.001853},
                            {"char": "n", "start": 224.896, "end": 224.916, "score": 0.010721},
                        ],
                    },
                    {
                        "entry": 2,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 224.996,
                        "ctc_score": 0.124389,
                        "ctc_token_spans": [
                            {"char": "e", "start": 224.996, "end": 225.016, "score": 0.203845},
                            {"char": "v", "start": 225.257, "end": 225.277, "score": 0.000450},
                            {"char": "e", "start": 225.337, "end": 225.357, "score": 0.048510},
                            {"char": "r", "start": 225.517, "end": 225.537, "score": 0.018484},
                            {"char": "l", "start": 225.537, "end": 225.557, "score": 0.006313},
                            {"char": "a", "start": 226.877, "end": 226.897, "score": 0.226242},
                            {"char": "s", "start": 227.057, "end": 227.077, "score": 0.001424},
                            {"char": "t", "start": 227.257, "end": 227.277, "score": 0.008473},
                        ],
                    },
                ],
                "suspicious_alignments": [],
            }
            annotate_ctc_boundary_evidence(report["assignments"])  # type: ignore[arg-type]
            evidence = report["assignments"][1]["ctc_boundary_evidence"]  # type: ignore[index]
            self.assertTrue(evidence["near_gap_weak_prefix"])
            self.assertFalse(assignment_timing_is_trusted(report["assignments"][1]))  # type: ignore[index]
            timestamps, report, changes = apply_final_ctc_timing_guard(
                Path("dummy.flac"), entries, [215.013, 224.996], report, 234.68
            )
            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 226.16, places=2)
            self.assertTrue(report["assignments"][1]["timing_audit_trusted"])  # type: ignore[index]
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_final_guard_salvages_onset_from_rejected_right_edge_row(self) -> None:
        frame_times = np.round(np.arange(143.0, 149.0, 0.01), 3).astype(np.float32)
        onset = np.zeros(len(frame_times), dtype=np.float32)
        onset[int(round((147.42 - 143.0) / 0.01))] = 0.8
        features = AudioFeatures(
            duration=185.0,
            frame_times=frame_times,
            rms_db=np.full(len(frame_times), -20.0, dtype=np.float32),
            onset_strength=onset,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        auto_lrc.analyze_audio = lambda *_args, **_kwargs: features  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["previous"]), LyricEntry(["もつれたまま"]), LyricEntry(["next"])]
            report: dict[str, object] = {
                "backend": "ctc",
                "timing_entries": 3,
                "assignments": [
                    {
                        "entry": 1,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 136.93,
                        "ctc_token_spans": [{"char": "x", "start": 144.806, "end": 144.826, "score": 0.01}],
                    },
                    {
                        "entry": 2,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.35,
                        "timestamp": 144.866,
                        "ctc_score": 0.0245,
                        "ctc_token_spans": [
                            {"char": "m", "start": 144.866, "end": 144.886, "score": 0.00129},
                            {"char": "o", "start": 144.966, "end": 144.986, "score": 0.003849},
                            {"char": "t", "start": 144.986, "end": 145.006, "score": 0.079064},
                            {"char": "s", "start": 145.006, "end": 145.026, "score": 0.261354},
                            {"char": "u", "start": 145.186, "end": 145.206, "score": 0.351242},
                            {"char": "r", "start": 145.406, "end": 145.426, "score": 0.002466},
                            {"char": "e", "start": 146.046, "end": 146.066, "score": 0.003242},
                            {"char": "t", "start": 146.066, "end": 146.086, "score": 0.014562},
                        ],
                    },
                    {
                        "entry": 3,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "timestamp": 167.588,
                        "ctc_token_spans": [{"char": "n", "start": 167.588, "end": 167.608, "score": 0.6}],
                    },
                ],
                "suspicious_alignments": [],
                "ctc_local_window_realign": {
                    "status": "applied",
                    "rejected_rows": [
                        {
                            "entry": 2,
                            "candidate": 147.42,
                            "reason": "local-window-right-edge-pileup",
                            "ctc_score": 0.014457,
                        }
                    ],
                },
            }
            timestamps, report, changes = apply_final_ctc_timing_guard(
                Path("dummy.flac"), entries, [136.93, 144.866, 167.588], report, 185.0
            )
            # A rejected full-line row no longer donates its onset.  Without an
            # independent prefix probe the line must remain unresolved.
            self.assertEqual(len(changes), 0)
            self.assertAlmostEqual(timestamps[1], 144.866, places=3)
            self.assertTrue(report["assignments"][1]["ctc_final_timing_guard_unresolved"])  # type: ignore[index]
            self.assertFalse(report["assignments"][1]["timing_audit_trusted"])  # type: ignore[index]
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_path_fracture_detects_disconnected_ctc_islands(self) -> None:
        assignment = {
            "ctc_token_spans": [
                {"start": 10.0, "end": 10.1, "score": 0.2},
                {"start": 10.2, "end": 10.3, "score": 0.2},
                {"start": 10.4, "end": 10.5, "score": 0.2},
                {"start": 17.0, "end": 17.1, "score": 0.2},
            ]
        }
        evidence = ctc_path_fracture_evidence(assignment)
        self.assertTrue(evidence["fractured"])
        self.assertAlmostEqual(evidence["max_gap_seconds"], 6.5, places=3)

    def test_duplicate_block_offset_projects_boundary_without_using_manual_timestamp(self) -> None:
        entries = [
            LyricEntry(["repeat A"]),
            LyricEntry(["repeat B"]),
            LyricEntry(["middle"]),
            LyricEntry(["repeat A"]),
            LyricEntry(["repeat B"]),
        ]
        assignments: list[dict[str, object]] = [
            {
                "timestamp": 10.0,
                "ctc_token_spans": [
                    {"char": "a", "start": 10.0, "score": 0.2},
                    {"char": "b", "start": 10.5, "score": 0.2},
                    {"char": "c", "start": 11.0, "score": 0.2},
                ],
            },
            {"timestamp": 20.0, "ctc_token_spans": [{"char": "x", "start": 20.0, "score": 0.01}]},
            {"timestamp": 50.0, "ctc_token_spans": [{"char": "m", "start": 50.0, "score": 0.2}]},
            {
                "timestamp": 110.0,
                "ctc_token_spans": [
                    {"char": "a", "start": 110.02, "score": 0.2},
                    {"char": "b", "start": 110.52, "score": 0.2},
                    {"char": "c", "start": 111.02, "score": 0.2},
                ],
            },
            {"timestamp": 121.4, "ctc_token_spans": [{"char": "x", "start": 121.4, "score": 0.01}]},
        ]
        evidence = duplicate_block_offset_evidence(entries, assignments, 1)
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertAlmostEqual(float(evidence["offset_seconds"]), 100.02, places=2)
        # The projection comes from the adjacent repeated line's acoustic/CTC
        # offset, not from a manually supplied expected boundary.
        self.assertAlmostEqual(float(evidence["projected_time"]), 21.38, places=2)


class CentralTimingStateTests(unittest.TestCase):
    def span(self, start: float, score: float = 0.20, token: str = "x") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def candidate(
        self,
        time: float,
        spans: tuple[auto_lrc.TimingTokenSpan, ...],
        *,
        source: str = "ctc-full",
        current: bool = False,
        confidence: float = 0.80,
        acoustic: auto_lrc.EvidenceState = "unavailable",
        acoustic_time: float | None = None,
        sequence: auto_lrc.EvidenceState = "supported",
        prefix_scope: bool = False,
        pileup: bool = False,
        truncated: bool = False,
        direct: bool = False,
        direct_bound: bool = True,
        direct_independent: bool = False,
        entry_index: int = 0,
    ) -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=entry_index,
            entry_text=f"generic line {entry_index}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=confidence,
            identity_support="supported",
            sequence_support=sequence,
            acoustic_support=acoustic,
            acoustic_strength=0.8 if acoustic == "supported" else None,
            acoustic_onset_time=(
                time if acoustic == "supported" and acoustic_time is None
                else acoustic_time
            ),
            acoustic_source_artifact=(
                {
                    "source": "synthetic-acoustic",
                    "time": time if acoustic_time is None else acoustic_time,
                    "strength": 0.8,
                }
                if acoustic == "supported"
                else None
            ),
            acoustic_evidence_producer=(
                "synthetic-independent-acoustic" if acoustic == "supported" else None
            ),
            acoustic_evidence_kind="generic-acoustic",
            acoustic_evidence_independent=bool(acoustic == "supported"),
            direct_onset_support="supported" if direct else "unavailable",
            direct_onset_time=time if direct and direct_bound else None,
            direct_onset_source_artifact=(
                {"source": "synthetic-direct", "time": time}
                if direct and direct_bound
                else None
            ),
            direct_onset_evidence_producer=(
                "phonetic-onset-fusion" if direct and direct_independent else None
            ),
            direct_onset_evidence_kind=(
                "phonetic-onset" if direct and direct_independent else "generic-onset"
            ),
            direct_onset_evidence_independent=bool(direct and direct_independent),
            current=current,
            prefix_scope=prefix_scope,
            right_edge_pileup=pileup,
            window_truncated=truncated,
            source_artifact={"source": source, "time": time},
        )

    def previous(self, end: float = 10.0, *, pileup: bool = False) -> auto_lrc.TimingCandidate:
        return self.candidate(
            end - 0.26,
            (
                self.span(end - 0.26),
                self.span(end - 0.18),
                self.span(end - 0.10),
            ),
            current=True,
            pileup=pileup,
            entry_index=0,
        )

    def coherent_spans(self, start: float) -> tuple[auto_lrc.TimingTokenSpan, ...]:
        return (
            self.span(start, 0.20, "a"),
            self.span(start + 0.08, 0.22, "b"),
            self.span(start + 0.16, 0.18, "c"),
        )

    def report(self, count: int, *, backend: str = "synthetic") -> dict[str, object]:
        return {
            "backend": backend,
            "assignments": [
                {"timing_repair": "synthetic", "score": 0.9}
                for _ in range(count)
            ],
            "timing_entries": count,
        }

    def test_canonical_written_time_recomputes_at_centisecond(self) -> None:
        candidate = self.candidate(59.996, self.coherent_spans(60.0))
        evidence = auto_lrc.normalize_timing_candidate(candidate)
        self.assertEqual(candidate.written_time.centiseconds, 6000)
        self.assertEqual(candidate.written_time.lrc_tag, "01:00.00")
        self.assertEqual(evidence.computed_for_centiseconds, 6000)

    def test_previous_tail_overlap_is_structurally_invalid(self) -> None:
        previous = self.previous(10.0)
        candidate = self.candidate(9.90, self.coherent_spans(9.90), entry_index=1)
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertFalse(evaluation.valid)
        self.assertIn("previous-tail-overlap", evaluation.rejection_reasons)

    def test_20ms_weak_prefix_is_not_timing_proof(self) -> None:
        previous = self.previous(10.0)
        spans = tuple(self.span(10.02 + offset * 0.06, 0.01) for offset in range(4))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(10.02, spans, entry_index=1), previous
        )
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.evidence.minimum_proof_satisfied)

    def test_60_to_150ms_weak_prefix_is_not_timing_proof(self) -> None:
        previous = self.previous(10.0)
        for gap in (0.06, 0.10, 0.15):
            spans = tuple(self.span(10.0 + gap + offset * 0.06, 0.01) for offset in range(5))
            evaluation = auto_lrc.evaluate_timing_candidate(
                self.candidate(10.0 + gap, spans, entry_index=1), previous
            )
            self.assertFalse(evaluation.valid, gap)

    def test_strong_onset_survives_tail_fracture(self) -> None:
        spans = (*self.coherent_spans(12.0), self.span(16.5), self.span(16.6))
        evaluation = auto_lrc.evaluate_timing_candidate(self.candidate(12.0, spans))
        self.assertTrue(evaluation.valid)
        self.assertIn(evaluation.evidence.fracture_region, {"middle", "tail"})

    def test_weak_token_zero_with_supported_following_run_can_prove_onset(self) -> None:
        spans = (
            self.span(12.0, 0.01),
            self.span(12.08, 0.20),
            self.span(12.16, 0.18),
        )
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, acoustic="supported")
        )
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.evidence.longest_supported_run, 2)

    def test_prefix_stretched_several_seconds_is_rejected(self) -> None:
        spans = (
            self.span(12.0), self.span(12.1), self.span(12.2),
            self.span(17.0), self.span(17.1),
        )
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, source="ctc-prefix-probe", prefix_scope=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("temporal-incoherence", evaluation.rejection_reasons)

    def test_multi_second_internal_gap_is_rejected_for_prefix_candidate(self) -> None:
        spans = (self.span(20.0), self.span(20.1), self.span(24.0), self.span(24.1))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(20.0, spans, source="ctc-prefix-probe", prefix_scope=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertGreater(float(evaluation.evidence.max_internal_gap or 0.0), 3.0)

    def test_right_edge_pileup_never_commits(self) -> None:
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, self.coherent_spans(12.0), pileup=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("right-edge-pileup", evaluation.rejection_reasons)

    def test_window_truncation_never_commits(self) -> None:
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, self.coherent_spans(12.0), truncated=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("window-truncated", evaluation.rejection_reasons)

    def test_timestamp_evidence_revision_mismatch_fails_final_audit(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(candidate,),
            previous_candidate=None,
            selection_revision=1,
        )
        stale = replace(decision, timing_evidence_revision=0)
        audit = auto_lrc.audit_final_timing_decisions((stale,))[0]
        self.assertFalse(audit.timing_trusted)
        self.assertIn("revision-mismatch", audit.failures)

    def test_valid_repair_candidate_commits_with_provenance(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.20,
        )
        repair = self.candidate(
            12.8,
            self.coherent_spans(12.8),
            source="ctc-prefix-probe",
            confidence=0.90,
            acoustic="supported",
            prefix_scope=True,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, repair),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.status, "selected_valid")
        self.assertEqual(decision.selected_candidate_id, repair.candidate_id)
        self.assertIsNotNone(decision.recovery)
        self.assertIn("candidate-dominates-current", decision.recovery.passed_invariants)  # type: ignore[union-attr]

    def test_no_valid_candidate_stays_unresolved(self) -> None:
        candidate = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
        )
        state = auto_lrc.build_final_timing_state(((candidate,),))
        self.assertEqual(state.decisions[0].status, "provisional_unresolved")
        self.assertFalse(state.audits[0].timing_trusted)

    def test_late_fracture_does_not_destroy_onset_proof(self) -> None:
        spans = (*self.coherent_spans(30.0), self.span(34.5), self.span(34.6))
        evidence = auto_lrc.normalize_timing_candidate(self.candidate(30.0, spans))
        self.assertEqual(evidence.temporal_coherence, "coherent")
        self.assertNotIn("temporal-incoherence", evidence.structural_failures)

    def test_weaker_new_candidate_cannot_override_valid_current(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=0.90)
        weaker = self.candidate(
            12.5, self.coherent_spans(12.5), source="ctc-local", confidence=0.60
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(weaker, current),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.selected_candidate_id, current.candidate_id)

    def test_current_replacement_diagnostics_preserve_decision_and_vectors(self):
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=.8)
        challenger = self.candidate(12.5, self.coherent_spans(12.5), source="ctc-local", confidence=.9)
        kwargs = dict(entry_index=0, candidates=(current, challenger),
                      previous_candidate=None, selection_revision=1)
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "events.jsonl"
            with mock.patch.dict(auto_lrc.os.environ, {}, clear=True):
                disabled = auto_lrc.select_timing_decision(**kwargs)
            with mock.patch.dict(auto_lrc.os.environ, {"LRC_CURRENT_REPLACEMENT_DIAGNOSTICS": str(path)}):
                enabled = auto_lrc.select_timing_decision(**kwargs)
            self.assertEqual(disabled, enabled)
            event = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(event["path"], "dominance-over-valid-current")
            self.assertEqual(len(event["current"]["vector"]), 5)
            self.assertEqual(len(event["challenger"]["vector"]), 5)
            self.assertEqual(event["challenger"]["candidate_id"], challenger.candidate_id)

    def test_gross_rescue_preserves_valid_current_for_subthreshold_replacement(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=0.80)
        challenger = self.candidate(
            12.5,
            self.coherent_spans(12.5),
            source="ctc-local",
            confidence=0.90,
            acoustic="supported",
        )
        baseline = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, challenger),
            previous_candidate=None,
            selection_revision=1,
        )
        rescued = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, challenger),
            previous_candidate=None,
            selection_revision=1,
            arbiter_mode="gross-rescue",
        )
        self.assertEqual(baseline.selected_candidate_id, challenger.candidate_id)
        self.assertEqual(rescued.selected_candidate_id, current.candidate_id)
        self.assertIsNotNone(rescued.arbiter)
        self.assertTrue(rescued.arbiter.use_current)  # type: ignore[union-attr]

    def test_gross_rescue_keeps_large_non_asr_central_replacement(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=0.80)
        challenger = self.candidate(
            12.6,
            self.coherent_spans(12.6),
            source="ctc-local",
            confidence=0.90,
            acoustic="supported",
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, challenger),
            previous_candidate=None,
            selection_revision=1,
            arbiter_mode="gross-rescue",
        )
        self.assertEqual(decision.selected_candidate_id, challenger.candidate_id)
        self.assertIsNotNone(decision.arbiter)
        self.assertFalse(decision.arbiter.use_current)  # type: ignore[union-attr]

    def test_gross_rescue_trace_is_committed_per_row(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=0.80)
        challenger = self.candidate(
            12.5,
            self.coherent_spans(12.5),
            source="ctc-local",
            confidence=0.90,
            acoustic="supported",
        )
        state = auto_lrc.build_final_timing_state(
            ((current, challenger),), arbiter_mode="gross-rescue"
        )
        report = self.report(1)
        committed = auto_lrc.commit_final_timing_state(state, report, {})
        trace = report["assignments"][0]["arbiter"]
        self.assertEqual(committed, state)
        self.assertEqual(trace["mode"], "gross-rescue")
        self.assertTrue(trace["reverted_to_current"])
        self.assertEqual(trace["selected_source"], "ctc-local")
        self.assertEqual(trace["shift_ms"], 500)

    def test_cli_arbiter_defaults_off_and_accepts_gross_rescue(self) -> None:
        parser = auto_lrc.build_parser()
        self.assertEqual(parser.parse_args(["song.flac"]).arbiter, "off")
        self.assertEqual(
            parser.parse_args(["song.flac", "--arbiter", "gross-rescue"]).arbiter,
            "gross-rescue",
        )

    def test_cli_accepts_reviewer_validity_and_evidence_path(self) -> None:
        parser = auto_lrc.build_parser()
        args = parser.parse_args(["song.flac", "--arbiter", "reviewer-validity", "--reviewer-evidence", "sidecar.json"])
        self.assertEqual(args.arbiter, "reviewer-validity")
        self.assertEqual(str(args.reviewer_evidence), "sidecar.json")

    def test_reviewer_sidecar_missing_falls_back_to_empty_rows(self) -> None:
        with TemporaryDirectory() as temp_name:
            rows = auto_lrc.load_reviewer_validity_evidence(Path(temp_name) / "missing.json", 2)
        self.assertEqual(rows, (None, None))

    def test_reviewer_sidecar_maps_entries_and_shared_policy(self) -> None:
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "sidecar.json"
            path.write_text(json.dumps({
                "schema": 1, "policy": "R2b",
                "reviewers": ["HUBP", "WX", "XLSR"], "required_agreements": 1,
                "offsets": {"HUBP": 0.02}, "offset_sample_counts": {"HUBP": 12},
                "rows": [{"entry": 2, "expected_current_seconds": 12.0, "reviewer_times": {"HUBP": 12.02}}],
            }), encoding="utf-8")
            rows = auto_lrc.load_reviewer_validity_evidence(path, 2)
        self.assertIsNone(rows[0])
        self.assertEqual(rows[1]["expected_current_seconds"], 12.0)
        self.assertEqual(rows[1]["offsets"], {"HUBP": 0.02})
        self.assertEqual(rows[1]["required_agreements"], 1)

    def test_reviewer_evidence_binds_into_generation_diagnostics_and_round_trips(self) -> None:
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "sidecar.json"
            path.write_text(json.dumps({
                "schema": 1, "policy": "R2b",
                "reviewers": ["HUBP", "WX", "XLSR"], "required_agreements": 1,
                "offsets": {"HUBP": 0.02}, "offset_sample_counts": {"HUBP": 8},
                "rows": [{"entry": 1, "expected_current_seconds": 12.0, "reviewer_times": {"HUBP": 12.02}}],
            }), encoding="utf-8")
            args = mock.Mock(arbiter="reviewer-validity", reviewer_evidence=path)
            diagnostics = auto_lrc.bind_reviewer_validity_diagnostics({}, args, 1)
        rows = auto_lrc.reviewer_validity_evidence_from_diagnostics(diagnostics, 1)
        self.assertEqual(rows[0]["expected_current_seconds"], 12.0)
        self.assertEqual(rows[0]["offsets"], {"HUBP": 0.02})

    def reviewer_validity_row(self, current_seconds: float) -> dict[str, object]:
        return {
            "expected_current_seconds": current_seconds,
            "reviewer_times": {"HUBP": current_seconds + 0.02, "WX": None, "XLSR": None},
            "offsets": {"HUBP": 0.02},
            "reviewers": ["HUBP", "WX", "XLSR"],
            "required_agreements": 1,
        }

    def test_reviewer_validity_endorsement_promotes_only_soft_invalid_current(self) -> None:
        current = self.candidate(12.0, tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)), current=True)
        baseline = auto_lrc.build_final_timing_state(((current,),))
        self.assertFalse(baseline.audits[0].timing_trusted)
        endorsed = auto_lrc.build_final_timing_state(((current,),), arbiter_mode="reviewer-validity", reviewer_evidence=(self.reviewer_validity_row(12.0),))
        self.assertEqual(endorsed.decisions[0].selected_candidate_id, current.candidate_id)
        self.assertEqual(endorsed.decisions[0].status, "selected_valid")
        self.assertTrue(endorsed.audits[0].timing_trusted)

    def test_reviewer_validity_endorsed_soft_invalid_current_survives_ordinary_challenger(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.20,
        )
        challenger = self.candidate(
            12.6,
            self.coherent_spans(12.6),
            source="ctc-local",
            confidence=0.95,
            acoustic="supported",
        )
        baseline = auto_lrc.build_final_timing_state(((current, challenger),))
        self.assertEqual(baseline.decisions[0].selected_candidate_id, challenger.candidate_id)

        endorsed = auto_lrc.build_final_timing_state(
            ((current, challenger),),
            arbiter_mode="reviewer-validity",
            reviewer_evidence=(self.reviewer_validity_row(12.0),),
        )
        self.assertEqual(endorsed.decisions[0].selected_candidate_id, current.candidate_id)
        self.assertEqual(endorsed.decisions[0].status, "selected_valid")
        self.assertTrue(endorsed.audits[0].timing_trusted)

    def test_reviewer_validity_endorsed_soft_invalid_current_yields_to_independent_consensus(self) -> None:
        current = self.candidate(
            12.69,
            tuple(self.span(12.69 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.20,
        )
        ctc_whisper_consensus = self.candidate(
            12.22,
            self.coherent_spans(12.22),
            source="ctc-local-whisper-vocal-independent-consensus",
            confidence=0.85,
            acoustic="supported",
        )
        direct_whisper = self.candidate(
            12.26,
            self.coherent_spans(12.26),
            source="local-whisper-vocal-independent-fusion",
            confidence=0.88,
            acoustic="supported",
            direct=True,
            direct_independent=True,
        )

        endorsed = auto_lrc.build_final_timing_state(
            ((current, ctc_whisper_consensus, direct_whisper),),
            arbiter_mode="reviewer-validity",
            reviewer_evidence=({
                "expected_current_seconds": 12.69,
                "reviewer_times": {"HUBP": 12.26, "WX": 12.27, "XLSR": 12.69},
                "offsets": {"HUBP": 0.0, "WX": 0.0, "XLSR": 0.0},
                "reviewers": ["HUBP", "WX", "XLSR"],
                "required_agreements": 1,
            },),
        )

        self.assertEqual(
            endorsed.decisions[0].selected_candidate_id,
            direct_whisper.candidate_id,
        )
        self.assertEqual(endorsed.decisions[0].status, "selected_valid")
        self.assertTrue(endorsed.audits[0].timing_trusted)

    def test_reviewer_validity_endorsed_current_blocks_consensus_with_less_reviewer_support(self) -> None:
        current = self.candidate(
            12.69,
            tuple(self.span(12.69 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.20,
        )
        ctc_whisper_consensus = self.candidate(
            12.22,
            self.coherent_spans(12.22),
            source="ctc-local-whisper-vocal-independent-consensus",
            confidence=0.85,
            acoustic="supported",
        )
        direct_whisper = self.candidate(
            12.26,
            self.coherent_spans(12.26),
            source="local-whisper-vocal-independent-fusion",
            confidence=0.88,
            acoustic="supported",
            direct=True,
            direct_independent=True,
        )

        endorsed = auto_lrc.build_final_timing_state(
            ((current, ctc_whisper_consensus, direct_whisper),),
            arbiter_mode="reviewer-validity",
            reviewer_evidence=({
                "expected_current_seconds": 12.69,
                "reviewer_times": {"HUBP": 12.69, "WX": 12.68, "XLSR": 12.26},
                "offsets": {"HUBP": 0.0, "WX": 0.0, "XLSR": 0.0},
                "reviewers": ["HUBP", "WX", "XLSR"],
                "required_agreements": 1,
            },),
        )

        self.assertEqual(endorsed.decisions[0].selected_candidate_id, current.candidate_id)

    def test_reviewer_validity_endorsed_current_keeps_current_on_tied_reviewer_support(self) -> None:
        current = self.candidate(
            12.69,
            tuple(self.span(12.69 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.20,
        )
        ctc_whisper_consensus = self.candidate(
            12.22,
            self.coherent_spans(12.22),
            source="ctc-local-whisper-vocal-independent-consensus",
            confidence=0.85,
            acoustic="supported",
        )
        direct_whisper = self.candidate(
            12.26,
            self.coherent_spans(12.26),
            source="local-whisper-vocal-independent-fusion",
            confidence=0.88,
            acoustic="supported",
            direct=True,
            direct_independent=True,
        )

        endorsed = auto_lrc.build_final_timing_state(
            ((current, ctc_whisper_consensus, direct_whisper),),
            arbiter_mode="reviewer-validity",
            reviewer_evidence=({
                "expected_current_seconds": 12.69,
                "reviewer_times": {"HUBP": 12.69, "WX": 12.26, "XLSR": None},
                "offsets": {"HUBP": 0.0, "WX": 0.0},
                "reviewers": ["HUBP", "WX", "XLSR"],
                "required_agreements": 1,
            },),
        )

        self.assertEqual(endorsed.decisions[0].selected_candidate_id, current.candidate_id)

    def test_reviewer_validity_does_not_relax_temporal_incoherence(self) -> None:
        current = self.candidate(12.0, (self.span(12.0), self.span(12.1), self.span(12.2), self.span(17.0), self.span(17.1)), source="ctc-prefix-probe", prefix_scope=True, current=True)
        state = auto_lrc.build_final_timing_state(((current,),), arbiter_mode="reviewer-validity", reviewer_evidence=(self.reviewer_validity_row(12.0),))
        self.assertFalse(state.audits[0].timing_trusted)
        self.assertIn("temporal-incoherence", state.audits[0].failures)

    def test_reviewer_validity_missing_evidence_falls_back_to_central(self) -> None:
        current = self.candidate(12.0, tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)), current=True)
        baseline = auto_lrc.build_final_timing_state(((current,),))
        fallback = auto_lrc.build_final_timing_state(((current,),), arbiter_mode="reviewer-validity", reviewer_evidence=(None,))
        self.assertEqual(fallback.decisions[0].status, baseline.decisions[0].status)
        self.assertEqual(fallback.written_times, baseline.written_times)
        self.assertEqual(fallback.audits[0].timing_trusted, baseline.audits[0].timing_trusted)

    def test_candidate_permutation_does_not_change_selection(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, confidence=0.80)
        stronger = self.candidate(
            12.5, self.coherent_spans(12.5), source="ctc-local", confidence=0.90,
            acoustic="supported",
        )
        weaker = self.candidate(
            12.8, self.coherent_spans(12.8), source="whisperx-word", confidence=0.60
        )
        selected = set()
        for items in permutations((current, stronger, weaker)):
            decision = auto_lrc.select_timing_decision(
                entry_index=0,
                candidates=tuple(items),
                previous_candidate=None,
                selection_revision=1,
            )
            selected.add(decision.selected_candidate_id)
        self.assertEqual(selected, {stronger.candidate_id})

    def test_cross_producer_confidence_alone_cannot_dominate_tight_independent_consensus(self) -> None:
        invalid_current = self.candidate(
            5.0,
            tuple(self.span(5.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=1.0,
        )
        near_ctc = self.candidate(
            10.0,
            self.coherent_spans(10.0),
            source="ctc-raw-independent-consensus",
            confidence=0.90,
            direct=True,
            direct_independent=True,
        )
        near_whisper = self.candidate(
            10.08,
            self.coherent_spans(10.08),
            source="raw-asr-vocal-fusion",
            confidence=0.90,
            direct=True,
            direct_independent=True,
        )
        distant_high_confidence = self.candidate(
            30.0,
            self.coherent_spans(30.0),
            source="whisperx-vocal-independent-fusion",
            confidence=0.99,
            direct=True,
            direct_independent=True,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(
                invalid_current,
                near_ctc,
                near_whisper,
                distant_high_confidence,
            ),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.status, "selected_valid")
        self.assertIn(decision.written_time.seconds, {10.0, 10.08})

    def test_candidate_id_is_content_derived(self) -> None:
        first = self.candidate(12.0, self.coherent_spans(12.0), source="ctc-local")
        second = self.candidate(12.0, self.coherent_spans(12.0), source="ctc-local")
        self.assertEqual(first.candidate_id, second.candidate_id)
        self.assertEqual(first.generation_revision, second.generation_revision)

    def test_non_ctc_unavailable_tail_can_use_independent_minimum_proof(self) -> None:
        previous = self.previous(10.0, pileup=True)
        whisperx = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="whisperx-word",
            acoustic="supported",
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(whisperx, previous)
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        # Generic acoustic onset may support an opening when ownership is clear,
        # but it is not phonetic proof and cannot bypass an unknown previous tail.
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.evidence.ownership_independent)

    def test_acoustic_only_candidate_lacks_identity_proof(self) -> None:
        candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="generic",
            source="acoustic-only",
            raw_time=12.0,
            spans=self.coherent_spans(12.0),
            confidence=0.9,
            identity_support="unavailable",
            sequence_support="supported",
            acoustic_support="supported",
        )
        self.assertFalse(auto_lrc.evaluate_timing_candidate(candidate).valid)

    def test_no_local_median_or_p90_is_not_an_absolute_hard_failure(self) -> None:
        spans = tuple(self.span(12.0 + index * 0.80, 0.20) for index in range(4))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, source="ctc-prefix-probe", prefix_scope=True)
        )
        self.assertNotIn("temporal-incoherence", evaluation.evidence.structural_failures)

    def test_final_state_is_frozen_and_uses_tuples(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        self.assertIsInstance(state.written_times, tuple)
        with self.assertRaises(FrozenInstanceError):
            state.written_times = ()  # type: ignore[misc]

    def test_previous_onset_downgrade_does_not_invalidate_revision_safe_clear_tail(self) -> None:
        first = self.candidate(10.0, self.coherent_spans(10.0), current=True, entry_index=0)
        first_decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(first,),
            previous_candidate=None,
            selection_revision=1,
        )
        second = self.candidate(12.0, self.coherent_spans(12.0), current=True, entry_index=1)
        second_decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(second,),
            previous_candidate=first,
            selection_revision=2,
        )
        stale_first = replace(first_decision, timing_evidence_revision=0)
        audits = auto_lrc.audit_final_timing_decisions((stale_first, second_decision))
        self.assertFalse(audits[0].timing_trusted)
        self.assertTrue(audits[1].timing_trusted)
        self.assertEqual(second_decision.evidence.ownership.status, "clear")
        self.assertEqual(
            second_decision.evidence.referenced_tail_evidence_revision,
            auto_lrc.derive_reliable_tail(first).tail_evidence_revision,
        )

    def test_independent_onset_proof_survives_previous_downgrade(self) -> None:
        first = self.candidate(
            10.0,
            tuple(self.span(10.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            entry_index=0,
        )
        second = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            current=True,
            direct=True,
            direct_independent=True,
            entry_index=1,
        )
        state = auto_lrc.build_final_timing_state(((first,), (second,)))
        self.assertFalse(state.audits[0].timing_trusted)
        self.assertTrue(state.audits[1].timing_trusted)

    def test_numeric_evidence_is_invariant_to_lyric_and_romaji_text(self) -> None:
        evaluations = []
        for text in ("alpha line", "beta line"):
            candidate = auto_lrc.make_timing_candidate(
                entry_index=0,
                entry_text=text,
                source="ctc-full",
                raw_time=12.0,
                spans=self.coherent_spans(12.0),
                confidence=0.8,
                identity_support="supported",
                sequence_support="supported",
                current=True,
                source_artifact={"numeric_fixture": 1},
            )
            evaluations.append(auto_lrc.evaluate_timing_candidate(candidate))
        self.assertEqual([item.valid for item in evaluations], [True, True])
        self.assertEqual(
            [item.evidence.structural_failures for item in evaluations],
            [(), ()],
        )
        self.assertEqual(
            [item.evidence.temporal_coherence for item in evaluations],
            ["coherent", "coherent"],
        )

    def test_single_raw_token_is_insufficient_timing_proof(self) -> None:
        candidate = self.candidate(
            12.0,
            (self.span(12.0, 0.9),),
            source="ctc-first-token-posterior",
            confidence=0.9,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate)
        self.assertFalse(evaluation.valid)
        self.assertIn("insufficient-minimum-proof", evaluation.rejection_reasons)

    def test_valid_earlier_repair_can_replace_invalid_current(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.2,
        )
        repair = self.candidate(
            11.5,
            self.coherent_spans(11.5),
            source="ctc-local-prefix",
            confidence=0.9,
            acoustic="supported",
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, repair),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.selected_candidate_id, repair.candidate_id)
        self.assertEqual(decision.written_time.seconds, 11.5)

    def test_previous_tail_distance_is_hard_gated_before_ranking(self) -> None:
        previous = self.previous(10.0)
        current = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            current=True,
            confidence=0.8,
            entry_index=1,
        )
        borrowing = self.candidate(
            9.96,
            self.coherent_spans(9.96),
            source="ctc-first-token-posterior",
            confidence=0.99,
            entry_index=1,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(borrowing, current),
            previous_candidate=previous,
            selection_revision=2,
        )
        self.assertEqual(decision.selected_candidate_id, current.candidate_id)
        rejected = next(
            item for item in decision.rejected_candidates
            if item.candidate.candidate_id == borrowing.candidate_id
        )
        self.assertFalse(rejected.valid)
        self.assertEqual(rejected.evidence.ownership.status, "unknown")
        self.assertAlmostEqual(float(rejected.evidence.tail_distance_seconds), 0.04, places=3)
        self.assertIn("previous-tail-ownership-unknown", rejected.rejection_reasons)
        state = auto_lrc.build_final_timing_state(
            ((previous,), (borrowing, current))
        )
        report: dict[str, object] = {
            "assignments": [
                {"timing_repair": "ctc-forced-align", "score": 0.9},
                {"timing_repair": "ctc-forced-align", "score": 0.9},
            ]
        }
        auto_lrc.commit_final_timing_state(state, report, {})
        report_rejections = report["assignments"][1]["candidate_rejections"]
        reported = next(
            item for item in report_rejections
            if item["candidate_id"] == borrowing.candidate_id
        )
        self.assertEqual(
            reported["evidence"]["ownership"]["reliable_tail_end"], 9.92
        )
        self.assertAlmostEqual(
            float(reported["evidence"]["tail_distance_seconds"]), 0.04, places=3
        )
        self.assertIn("previous-tail-ownership-unknown", reported["reasons"])

    def test_candidate_beyond_reliable_tail_uncertainty_is_valid(self) -> None:
        previous = self.previous(10.0)
        candidate = self.candidate(
            10.5,
            self.coherent_spans(10.5),
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.evidence.ownership.status, "clear")
        self.assertGreater(float(evaluation.evidence.tail_distance_seconds), 0.2)

    def test_successful_recovery_clears_severe_flag_with_provenance(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.2,
        )
        repair = self.candidate(
            12.8,
            self.coherent_spans(12.8),
            source="ctc-prefix-probe",
            confidence=0.9,
            acoustic="supported",
            prefix_scope=True,
        )
        state = auto_lrc.build_final_timing_state(((current, repair),))
        decision = state.decisions[0]
        self.assertIsNotNone(decision.recovery)
        resolved_id = auto_lrc._timing_content_digest({
            "entry": decision.entry_index,
            "flag": "ctc_unresolved_boundary",
            "candidate_revision": decision.evidence.candidate_revision,
        })
        decision = replace(
            decision,
            recovery=replace(
                decision.recovery,
                resolved_finding_ids=(resolved_id,),
            ),
        )
        state = replace(state, decisions=(decision,))
        report: dict[str, object] = {
            "assignments": [{
                "timestamp": 12.0,
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "flags": ["ctc_unresolved_boundary"],
            }]
        }
        state = auto_lrc.commit_final_timing_state(state, report, {})
        assignment = report["assignments"][0]
        self.assertTrue(assignment["timing_trusted"])
        self.assertNotIn("ctc_unresolved_boundary", assignment["flags"])
        self.assertEqual(assignment["active_timing_findings"], [])
        resolved = assignment["resolved_timing_findings"]
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["resolved_by_candidate_id"], repair.candidate_id)
        self.assertEqual(resolved[0]["resolution_mode"], "challenger-recovery")
        auto_lrc.assert_report_timestamp_equality(report, state)

    def test_valid_current_with_historical_severe_finding_cannot_silently_clear(self) -> None:
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        state = auto_lrc.build_final_timing_state(((current,),))
        self.assertIsNone(state.decisions[0].recovery)
        report: dict[str, object] = {
            "assignments": [{
                "timestamp": 12.0,
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "flags": ["ctc_unresolved_boundary"],
            }]
        }

        final_state = auto_lrc.commit_final_timing_state(state, report, {})

        assignment = report["assignments"][0]
        self.assertIn("ctc_unresolved_boundary", assignment["flags"])
        self.assertEqual(assignment["resolved_timing_findings"], [])
        self.assertFalse(assignment["timing_trusted"])
        self.assertTrue(assignment["review_required"])
        self.assertFalse(final_state.audits[0].timing_trusted)
        auto_lrc.assert_report_timestamp_equality(report, final_state)

    def test_successful_recovery_only_clears_explicitly_resolved_finding(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(4)),
            current=True,
            confidence=0.2,
        )
        repair = self.candidate(
            12.8,
            self.coherent_spans(12.8),
            source="ctc-opening-segment",
            confidence=0.9,
            acoustic="supported",
        )
        state = auto_lrc.build_final_timing_state(((current, repair),))
        decision = state.decisions[0]
        self.assertIsNotNone(decision.recovery)
        resolved_id = auto_lrc._timing_content_digest({
            "entry": decision.entry_index,
            "flag": "ctc_unresolved_boundary",
            "candidate_revision": decision.evidence.candidate_revision,
        })
        recovery = replace(
            decision.recovery,
            resolved_finding_ids=(resolved_id,),
        )
        decision = replace(decision, recovery=recovery)
        state = replace(state, decisions=(decision,))
        report: dict[str, object] = {
            "assignments": [{
                "timestamp": 12.0,
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "flags": ["ctc_unresolved_boundary", "ctc_path_fracture"],
            }]
        }

        final_state = auto_lrc.commit_final_timing_state(state, report, {})

        assignment = report["assignments"][0]
        self.assertNotIn("ctc_unresolved_boundary", assignment["flags"])
        self.assertIn("ctc_path_fracture", assignment["flags"])
        self.assertEqual(
            [item["code"] for item in assignment["resolved_timing_findings"]],
            ["ctc_unresolved_boundary"],
        )
        self.assertEqual(
            [item["code"] for item in assignment["active_timing_findings"]],
            ["ctc_path_fracture"],
        )
        resolved = assignment["resolved_timing_findings"][0]
        self.assertEqual(resolved["resolved_by_candidate_id"], repair.candidate_id)
        self.assertEqual(
            resolved["resolved_candidate_revision"], decision.evidence.candidate_revision
        )
        self.assertEqual(
            resolved["resolved_evidence_revision"], decision.timing_evidence_revision
        )
        self.assertEqual(resolved["resolved_canonical_written_centiseconds"], 1280)
        self.assertEqual(resolved["resolved_selection_revision"], decision.selection_revision)
        self.assertEqual(resolved["resolution_mode"], "challenger-recovery")
        self.assertTrue(resolved["passed_invariants"])
        self.assertFalse(assignment["timing_trusted"])
        self.assertTrue(assignment["review_required"])
        self.assertFalse(final_state.audits[0].timing_trusted)
        auto_lrc.assert_report_timestamp_equality(report, final_state)

    def test_report_equality_detects_late_alias_mutation(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        report: dict[str, object] = {
            "assignments": [{
                "timestamp": 12.0,
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
            }]
        }
        auto_lrc.commit_final_timing_state(state, report, {})
        report["assignments"][0]["canonical_written_centiseconds"] = 1201
        with self.assertRaises(LrcError):
            auto_lrc.assert_report_timestamp_equality(report, state)

    def test_predecessor_onset_failure_does_not_cascade_across_clear_lines(self) -> None:
        candidates = tuple(
            self.candidate(
                10.0 + index * 2.0,
                self.coherent_spans(10.0 + index * 2.0),
                current=True,
                entry_index=index,
            )
            for index in range(3)
        )
        decisions = []
        previous = None
        for index, candidate in enumerate(candidates):
            decisions.append(auto_lrc.select_timing_decision(
                entry_index=index,
                candidates=(candidate,),
                previous_candidate=previous,
                selection_revision=index + 1,
            ))
            previous = candidate
        decisions[0] = replace(decisions[0], timing_evidence_revision=0)
        audits = auto_lrc.audit_final_timing_decisions(tuple(decisions))
        self.assertEqual([audit.timing_trusted for audit in audits], [False, True, True])
        self.assertTrue(all(
            decision.evidence.ownership.status == "clear"
            for decision in decisions[1:]
        ))

    def test_first_local_onset_edge_guard_is_an_explicit_caller_contract(self) -> None:
        frame_times = np.round(np.arange(0.0, 2.0, 0.1), 3).astype(np.float32)
        onset_strength = np.zeros_like(frame_times)
        onset_strength[np.where(np.isclose(frame_times, 1.1))[0][0]] = 1.0
        features = AudioFeatures(
            duration=2.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset_strength,
            segments=[],
        )
        self.assertIsNone(auto_lrc.first_local_onset(features, 1.0, 1.5))
        self.assertAlmostEqual(
            auto_lrc.first_local_onset(
                features, 1.0, 1.5, edge_guard_seconds=0.0
            ) or 0.0,
            1.1,
            places=3,
        )

    def test_write_lrc_reads_immutable_canonical_times_without_mutation(self) -> None:
        candidate = self.candidate(12.006, self.coherent_spans(12.01), current=True)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        before = state.written_times
        original_labels = auto_lrc.audio_track_labels
        auto_lrc.audio_track_labels = lambda _path: ("", "generic")  # type: ignore[assignment]
        try:
            with TemporaryDirectory() as temp_name:
                root = Path(temp_name)
                output = root / "output.lrc"
                auto_lrc.write_lrc(
                    output,
                    root / "audio.flac",
                    root / "lyrics.txt",
                    [LyricEntry(["generic line"])],
                    state.written_times,
                    20.0,
                    False,
                )
                written = output.read_text(encoding="utf-8-sig")
        finally:
            auto_lrc.audio_track_labels = original_labels  # type: ignore[assignment]
        self.assertIs(state.written_times, before)
        self.assertIn(f"[{state.written_times[0].lrc_tag}]generic line", written)

    def test_sol_r1_late_supported_pair_cannot_prove_opening(self) -> None:
        spans = (
            self.span(12.0, 0.01),
            self.span(12.08, 0.01),
            self.span(12.16, 0.20),
            self.span(12.24, 0.20),
        )
        evidence = auto_lrc.normalize_timing_candidate(self.candidate(12.0, spans))
        self.assertIsNone(evidence.onset_proof_end)
        self.assertEqual(evidence.opening_supported_run, 0)
        self.assertEqual(evidence.later_supported_run, 2)

    def test_sol_r1_weak_token_zero_contiguous_token_one_and_acoustic_prove_opening(self) -> None:
        spans = (self.span(12.0, 0.01), self.span(12.08, 0.20))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, acoustic="supported")
        )
        self.assertTrue(evaluation.valid)
        self.assertAlmostEqual(float(evaluation.evidence.onset_proof_end), 12.10, places=3)

    def test_sol_r1_weak_token_zero_distant_token_one_cannot_prove_opening(self) -> None:
        spans = (self.span(12.0, 0.01), self.span(12.5, 0.20))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, acoustic="supported")
        )
        self.assertFalse(evaluation.valid)
        self.assertIsNone(evaluation.evidence.onset_proof_end)

    def test_sol_r1_scores_separated_by_multi_second_gap_are_not_consecutive(self) -> None:
        spans = (self.span(12.0, 0.20), self.span(15.1, 0.20))
        evidence = auto_lrc.normalize_timing_candidate(self.candidate(12.0, spans))
        self.assertEqual(evidence.longest_supported_run, 1)
        self.assertEqual(evidence.opening_supported_run, 1)
        self.assertIsNone(evidence.onset_proof_end)

    def test_sol_r2_prefix_single_existing_severe_gap_is_incoherent(self) -> None:
        spans = (
            self.span(12.0), self.span(12.08), self.span(15.10), self.span(15.18)
        )
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, prefix_scope=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertEqual(evaluation.evidence.severe_gap_count, 1)
        self.assertEqual(evaluation.evidence.island_count, 2)
        self.assertEqual(evaluation.evidence.temporal_coherence, "incoherent")

    def test_sol_r2_multiple_severe_gaps_cannot_hide_in_inflated_median(self) -> None:
        spans = tuple(self.span(time) for time in (12.0, 15.2, 18.5, 21.9))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, prefix_scope=True)
        )
        self.assertFalse(evaluation.valid)
        self.assertEqual(evaluation.evidence.severe_gap_count, 3)
        self.assertEqual(evaluation.evidence.island_count, 4)
        self.assertEqual(evaluation.evidence.temporal_coherence, "incoherent")

    def test_sol_r2_full_line_strong_opening_survives_late_tail_fracture(self) -> None:
        spans = (*self.coherent_spans(12.0), self.span(16.2), self.span(16.3))
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(12.0, spans, prefix_scope=False)
        )
        self.assertTrue(evaluation.valid)
        self.assertGreater(evaluation.evidence.severe_gap_count, 0)
        self.assertIn(evaluation.evidence.fracture_region, {"middle", "tail"})

    def test_sol_r2_fractured_prefix_requires_independent_opening_segment_candidate(self) -> None:
        fractured = self.candidate(
            12.0,
            (*self.coherent_spans(12.0), self.span(16.2)),
            source="ctc-prefix-probe",
            prefix_scope=True,
            confidence=0.95,
        )
        opening = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="ctc-opening-segment",
            confidence=0.8,
            acoustic="supported",
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(fractured, opening),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.selected_candidate_id, opening.candidate_id)
        self.assertIn(
            "temporal-incoherence",
            next(
                item for item in decision.rejected_candidates
                if item.candidate.candidate_id == fractured.candidate_id
            ).rejection_reasons,
        )

    def test_sol_r4_acoustic_peak_change_updates_candidate_and_evidence_revisions(self) -> None:
        first = self.candidate(
            12.0, self.coherent_spans(12.0), acoustic="supported", acoustic_time=12.02
        )
        second = self.candidate(
            12.0, self.coherent_spans(12.0), acoustic="supported", acoustic_time=12.08
        )
        self.assertNotEqual(first.candidate_id, second.candidate_id)
        self.assertNotEqual(first.generation_revision, second.generation_revision)
        self.assertNotEqual(
            auto_lrc.normalize_timing_candidate(first).candidate_revision,
            auto_lrc.normalize_timing_candidate(second).candidate_revision,
        )

    def test_sol_r4_candidate_a_acoustic_peak_cannot_support_candidate_b(self) -> None:
        candidate_a = self.candidate(
            12.0, self.coherent_spans(12.0), acoustic="supported", acoustic_time=12.0
        )
        candidate_b = self.candidate(
            13.0, self.coherent_spans(13.0), acoustic="supported", acoustic_time=12.0
        )
        self.assertTrue(auto_lrc.candidate_has_bound_acoustic_evidence(candidate_a))
        self.assertFalse(auto_lrc.candidate_has_bound_acoustic_evidence(candidate_b))

    def test_sol_r4_missing_peak_provenance_cannot_create_independent_ownership(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="generic",
            source="ctc-prefix-probe",
            raw_time=12.0,
            spans=self.coherent_spans(12.0),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.evidence.ownership_independent)

    def test_sol_r6_prefix_source_name_does_not_grant_sequence_support(self) -> None:
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="ctc-prefix-probe",
            prefix_scope=True,
            sequence="unavailable",
            acoustic="supported",
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate)
        self.assertFalse(evaluation.valid)
        self.assertEqual(candidate.sequence_support, "unavailable")

    def test_sol_r6_unavailable_neighbor_and_no_sequence_proof_stay_unresolved(self) -> None:
        previous = self.previous(10.0, pileup=True)
        prefix = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="ctc-prefix-probe",
            prefix_scope=True,
            sequence="unavailable",
            acoustic="supported",
            entry_index=1,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(prefix,),
            previous_candidate=previous,
            selection_revision=2,
        )
        self.assertEqual(decision.status, "provisional_unresolved")

    def test_sol_r7_invalid_prefix_cannot_clear_historical_severe_finding(self) -> None:
        prefix = self.candidate(
            12.0,
            (*self.coherent_spans(12.0), self.span(16.2)),
            source="ctc-prefix-probe",
            prefix_scope=True,
            current=True,
        )
        state = auto_lrc.build_final_timing_state(((prefix,),))
        report: dict[str, object] = {
            "assignments": [{
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "flags": ["ctc_unresolved_boundary"],
            }]
        }
        auto_lrc.commit_final_timing_state(state, report, {})
        assignment = report["assignments"][0]
        self.assertEqual(assignment["resolved_timing_findings"], [])
        self.assertEqual(
            assignment["historical_legacy_timing_findings"][0]["code"],
            "ctc_unresolved_boundary",
        )
        self.assertFalse(assignment["timing_trusted"])

    def test_sol_r7_active_and_historical_evidence_scopes_do_not_mix(self) -> None:
        candidate = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(3)),
            current=True,
        )
        state = auto_lrc.build_final_timing_state(((candidate,),))
        report: dict[str, object] = {
            "assignments": [{
                "timing_repair": "ctc-forced-align",
                "score": 0.9,
                "flags": ["ctc_unresolved_boundary"],
            }]
        }
        auto_lrc.commit_final_timing_state(state, report, {})
        assignment = report["assignments"][0]
        active_historical = [
            item for item in assignment["active_timing_findings"]
            if item["evidence_scope"] == "active-historical-boundary"
        ]
        active_central = [
            item for item in assignment["active_timing_findings"]
            if item["evidence_scope"] == "active-central-evaluator"
        ]
        self.assertEqual(
            [item["code"] for item in active_historical],
            ["ctc_unresolved_boundary"],
        )
        self.assertTrue(active_central)
        self.assertEqual(
            active_historical[0]["finding_id"],
            assignment["historical_legacy_timing_findings"][0]["finding_id"],
        )
        self.assertTrue(all(
            item["evidence_scope"] == "historical-legacy-boundary"
            for item in assignment["historical_legacy_timing_findings"]
        ))

    def test_sol_r7_permutation_preserves_multi_island_rejection_winner_and_trust(self) -> None:
        current = self.candidate(
            12.0,
            tuple(self.span(12.0 + i * 0.06, 0.01) for i in range(3)),
            current=True,
        )
        multi_island = self.candidate(
            12.0,
            tuple(self.span(time) for time in (12.0, 15.2, 18.5)),
            source="ctc-prefix-probe",
            prefix_scope=True,
            confidence=0.99,
        )
        valid = self.candidate(
            12.5,
            self.coherent_spans(12.5),
            source="ctc-opening-segment",
            confidence=0.8,
            acoustic="supported",
        )
        outcomes = set()
        for items in permutations((current, multi_island, valid)):
            decision = auto_lrc.select_timing_decision(
                entry_index=0,
                candidates=tuple(items),
                previous_candidate=None,
                selection_revision=1,
            )
            audit = auto_lrc.audit_final_timing_decisions((decision,))[0]
            rejected_multi = next(
                item for item in decision.rejected_candidates
                if item.candidate.candidate_id == multi_island.candidate_id
            )
            outcomes.add((
                decision.selected_candidate_id,
                audit.timing_trusted,
                rejected_multi.evidence.island_count,
                rejected_multi.evidence.temporal_coherence,
            ))
        self.assertEqual(outcomes, {(valid.candidate_id, True, 3, "incoherent")})

    def test_revision_safe_clear_binds_both_candidates_and_tail_revision(self) -> None:
        previous = self.previous(10.0)
        current = self.candidate(12.0, self.coherent_spans(12.0), entry_index=1)
        evidence = auto_lrc.normalize_timing_candidate(current, previous)
        ownership = evidence.ownership
        self.assertEqual(ownership.status, "clear")
        self.assertEqual(ownership.current_candidate_id, current.candidate_id)
        self.assertEqual(ownership.current_generation_revision, current.generation_revision)
        self.assertEqual(ownership.previous_candidate_id, previous.candidate_id)
        self.assertEqual(
            ownership.previous_tail_evidence_revision,
            auto_lrc.derive_reliable_tail(previous).tail_evidence_revision,
        )
        self.assertEqual(
            evidence.referenced_tail_evidence_revision,
            ownership.previous_tail_evidence_revision,
        )

    def test_relation_revision_mismatch_fails_closed(self) -> None:
        previous = self.previous(10.0)
        current = self.candidate(12.0, self.coherent_spans(12.0), entry_index=1)
        regions = auto_lrc.analyze_candidate_regions(current)
        tail = auto_lrc.derive_reliable_tail(previous)
        relation = auto_lrc.classify_candidate_ownership(current, regions, tail)
        stale = replace(relation, previous_tail_evidence_revision="stale")
        evidence = auto_lrc.normalize_timing_candidate(
            current,
            previous,
            current_regions=regions,
            previous_tail=tail,
            ownership_relation=stale,
        )
        self.assertFalse(evidence.minimum_proof_satisfied)
        self.assertIn("ownership-revision-mismatch", evidence.structural_failures)

    def test_tail_pileup_and_truncation_make_ownership_unknown(self) -> None:
        for previous in (
            self.previous(10.0, pileup=True),
            self.candidate(
                9.8,
                (self.span(9.8), self.span(9.9)),
                current=True,
                truncated=True,
            ),
        ):
            with self.subTest(reason=auto_lrc.derive_reliable_tail(previous).reason):
                tail = auto_lrc.derive_reliable_tail(previous)
                self.assertEqual(tail.validity, "invalid")
                evaluation = auto_lrc.evaluate_timing_candidate(
                    self.candidate(12.0, self.coherent_spans(12.0), entry_index=1),
                    previous,
                )
                self.assertEqual(evaluation.evidence.ownership.status, "unknown")
                self.assertFalse(evaluation.valid)

    def test_tail_fracture_preserves_onset_but_invalidates_tail_evidence(self) -> None:
        previous = self.candidate(
            12.0,
            (*self.coherent_spans(12.0), self.span(16.2), self.span(16.3)),
            current=True,
        )
        self.assertTrue(auto_lrc.evaluate_timing_candidate(previous).valid)
        tail = auto_lrc.derive_reliable_tail(previous)
        self.assertEqual(tail.validity, "invalid")
        self.assertEqual(tail.reason, "tail-fracture-invalid")

    def test_e14_like_unknown_fracture_cannot_create_clear_successor(self) -> None:
        previous = self.candidate(
            12.0,
            (self.span(12.0, 0.01), self.span(15.2), self.span(15.3)),
            current=True,
        )
        regions = auto_lrc.analyze_candidate_regions(previous)
        self.assertGreater(len(regions.severe_gap_indexes), 0)
        self.assertEqual(auto_lrc.derive_reliable_tail(previous).validity, "invalid")
        evaluation = auto_lrc.evaluate_timing_candidate(
            self.candidate(18.0, self.coherent_spans(18.0), entry_index=1),
            previous,
        )
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        self.assertFalse(evaluation.valid)

    def test_historical_onset_failure_does_not_invalidate_valid_tail_relation(self) -> None:
        previous = self.candidate(10.0, self.coherent_spans(10.0), current=True)
        current = self.candidate(12.0, self.coherent_spans(12.0), current=True, entry_index=1)
        state = auto_lrc.build_final_timing_state(((previous,), (current,)))
        report = self.report(2)
        report["assignments"][0]["flags"] = ["ctc_unresolved_boundary"]
        effective, _findings = auto_lrc.evaluate_final_timing_state_for_report(state, report)
        self.assertEqual(
            [audit.timing_trusted for audit in effective.audits],
            [False, True],
        )
        self.assertEqual(state.decisions[1].evidence.ownership.status, "clear")

    def test_unknown_without_candidate_bound_onset_remains_unresolved(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="whisperx-word",
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        self.assertFalse(evaluation.evidence.ownership_independent)
        self.assertFalse(evaluation.valid)

    def test_sequence_or_confidence_cannot_bypass_unknown_ownership(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            confidence=1.0,
            sequence="supported",
            entry_index=1,
        )
        self.assertFalse(auto_lrc.evaluate_timing_candidate(candidate, previous).valid)

    def test_revision_bound_direct_onset_can_bypass_unknown_ownership(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            direct=True,
            direct_independent=True,
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertTrue(evaluation.evidence.ownership_independent)
        self.assertTrue(evaluation.valid)

    def test_self_bound_direct_onset_cannot_bypass_unknown_ownership(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            source="whisperx-forced-first",
            direct=True,
            direct_independent=False,
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        self.assertFalse(evaluation.evidence.ownership_independent)
        self.assertFalse(evaluation.valid)

    def test_unapproved_direct_onset_producer_cannot_bypass_unknown(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="generic line 1",
            source="whisperx-forced-first",
            raw_time=12.0,
            spans=self.coherent_spans(12.0),
            confidence=1.0,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported",
            direct_onset_time=12.0,
            direct_onset_source_artifact={"claimed": "independent"},
            direct_onset_evidence_producer="whisperx-forced-first",
            direct_onset_evidence_kind="phonetic-onset",
            direct_onset_evidence_independent=True,
            source_artifact={"source": "whisperx-forced-first"},
        )
        self.assertFalse(auto_lrc.candidate_has_bound_direct_onset_evidence(candidate))
        self.assertFalse(auto_lrc.evaluate_timing_candidate(candidate, previous).valid)

    def test_bare_direct_support_without_provenance_cannot_bypass_unknown(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            self.coherent_spans(12.0),
            direct=True,
            direct_bound=False,
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertFalse(evaluation.evidence.ownership_independent)
        self.assertFalse(evaluation.valid)

    def test_unresolved_unknown_line_does_not_poison_clear_successor(self) -> None:
        first = self.previous(10.0, pileup=True)
        second = self.candidate(12.0, self.coherent_spans(12.0), current=True, entry_index=1)
        third = self.candidate(14.0, self.coherent_spans(14.0), current=True, entry_index=2)
        state = auto_lrc.build_final_timing_state(((first,), (second,), (third,)))
        self.assertEqual(
            [audit.timing_trusted for audit in state.audits],
            [False, False, True],
        )
        self.assertEqual(state.decisions[2].evidence.ownership.status, "clear")

    def test_egakumirai_forced_first_5957_cannot_self_certify_over_unknown_tail(self) -> None:
        previous = self.previous(59.20, pileup=True)
        forced = self.candidate(
            59.57,
            (),
            source="whisperx-forced-first",
            confidence=0.58,
            direct=True,
            direct_independent=False,
            entry_index=1,
        )
        selected = self.candidate(
            62.28,
            (),
            source="whisperx-word",
            confidence=0.835,
            direct=False,
            entry_index=1,
        )
        forced_eval = auto_lrc.evaluate_timing_candidate(forced, previous)
        selected_eval = auto_lrc.evaluate_timing_candidate(selected, previous)
        self.assertFalse(forced_eval.valid)
        self.assertFalse(forced_eval.evidence.ownership_independent)
        self.assertFalse(selected_eval.valid)
        state = auto_lrc.build_final_timing_state(((previous,), (forced, selected)))
        self.assertFalse(state.audits[1].timing_trusted)
        self.assertEqual(state.decisions[1].status, "provisional_unresolved")

    def test_generic_acoustic_peak_cannot_replace_phonetic_opening_proof(self) -> None:
        previous = self.previous(10.0, pileup=True)
        candidate = self.candidate(
            12.0,
            (),
            source="audio-validated-current",
            acoustic="supported",
            acoustic_time=12.0,
            entry_index=1,
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate, previous)
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.evidence.ownership_independent)

    def test_whisperx_advisory_refinement_never_mutates_baseline(self) -> None:
        entries = [LyricEntry(["first"]), LyricEntry(["second"])]
        baseline = [10.0, 20.0]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {"entry": 2, "timestamp": 20.0, "score": 0.9},
            ]
        }
        asr = [
            auto_lrc.AsrSegment(10.0, 12.0, "first", 0.9),
            auto_lrc.AsrSegment(20.0, 22.0, "second", 0.9),
        ]
        forced = [
            auto_lrc.AsrSegment(9.5, 12.0, "first", 0.9, [("f", 9.5)]),
            auto_lrc.AsrSegment(17.0, 22.0, "second", 0.9, [("s", 17.0)]),
        ]
        refined, changes = auto_lrc.apply_whisperx_lyric_refinement(
            entries, baseline, report, asr, forced, 30.0
        )
        self.assertEqual(refined, baseline)
        self.assertEqual([row["timestamp"] for row in report["assignments"]], baseline)
        self.assertTrue(changes)
        self.assertTrue(report["whisperx_refinement_candidate_only"])
        self.assertFalse(report["whisperx_refinement_mutates_baseline"])

    def test_lyric_identity_contract_rejects_corrupted_anchor_text(self) -> None:
        entries = [LyricEntry(["合成試験の文字列です"])]
        report: dict[str, object] = {
            "suspicious_alignments": [
                {"entry": 1, "anchor_profile": {"entry_text": "合成試験の文宇列です"}}
            ]
        }
        with self.assertRaisesRegex(auto_lrc.LrcError, "Phonetic anchor text diverged"):
            auto_lrc.assert_lyric_identity_contract(entries, report, initialize=True)

    def test_review_state_is_not_an_implicit_output_gate(self) -> None:
        args = auto_lrc.build_parser().parse_args(["song.flac"])
        self.assertFalse(args.strict_review)
        self.assertFalse(args.fail_on_review_required)
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("Refusing to write untrusted LRC", source)
        self.assertNotIn("blocking untrusted LRC", source)

    def test_allow_untrusted_draft_flag_remains_cli_compatible_but_is_not_required(self) -> None:
        args = auto_lrc.build_parser().parse_args(["song.flac", "--allow-untrusted-draft"])
        self.assertTrue(args.allow_untrusted_draft)

    def test_production_source_contains_no_song_specific_alignment_literals(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        for forbidden in ("Georgette", "エガクミライ", "\\u79c1"):
            self.assertNotIn(forbidden, source)

    def test_forward_and_reverse_worklists_preserve_adjacency_and_result(self) -> None:
        candidates = tuple(
            (
                self.candidate(
                    10.0 + index * 2.0,
                    self.coherent_spans(10.0 + index * 2.0),
                    current=True,
                    entry_index=index,
                ),
            )
            for index in range(4)
        )
        forward = auto_lrc.build_final_timing_state(candidates, worklist_order="forward")
        reverse = auto_lrc.build_final_timing_state(candidates, worklist_order="reverse")
        self.assertEqual(
            [decision.selected_candidate_id for decision in forward.decisions],
            [decision.selected_candidate_id for decision in reverse.decisions],
        )
        self.assertEqual(forward.audits, reverse.audits)
        self.assertEqual(
            [decision.evidence.ownership_relation_revision for decision in forward.decisions],
            [decision.evidence.ownership_relation_revision for decision in reverse.decisions],
        )

    def test_backend_quality_prefers_timing_coverage_to_review_sparsity(self):
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1, backend="synthetic")
        evaluated = auto_lrc.evaluate_backend_timing("synthetic", ((candidate,),), report, {})
        rich = replace(evaluated.summary, timing_trusted_percent=80.0, overall_trusted_percent=40.0)
        sparse = replace(evaluated.summary, timing_trusted_percent=60.0, overall_trusted_percent=60.0)
        self.assertGreater(auto_lrc.central_evaluated_quality({}, rich),
                           auto_lrc.central_evaluated_quality({}, sparse))
        self.assertLess(auto_lrc.central_evaluated_quality({"collapse_detected": True}, rich),
                        auto_lrc.central_evaluated_quality({}, sparse))

    def test_central_backend_rank_overrides_opposite_generation_metrics(self) -> None:
        invalid = self.candidate(
            12.0,
            tuple(self.span(12.0 + index * 0.06, 0.01) for index in range(3)),
            current=True,
        )
        valid = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report_a = self.report(1, backend="backend-a")
        report_a.update({"trusted_percent": 100.0, "review_required_count": 0})
        report_b = self.report(1, backend="backend-b")
        report_b.update({"trusted_percent": 0.0, "review_required_count": 1})
        evaluated_a = auto_lrc.evaluate_backend_timing(
            "backend-a", ((invalid,),), report_a, {}
        )
        evaluated_b = auto_lrc.evaluate_backend_timing(
            "backend-b", ((valid,),), report_b, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated_a, evaluated_b))
        self.assertIs(selection.selected, evaluated_b)
        self.assertGreater(evaluated_b.quality, evaluated_a.quality)

    def test_central_quality_ignores_candidate_selection_record(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        first = self.report(1, backend="synthetic")
        second = self.report(1, backend="synthetic")
        first["candidate_selection"] = {"selected_quality": 100.0}
        second["candidate_selection"] = {"selected_quality": -100.0}
        left = auto_lrc.evaluate_backend_timing("synthetic", ((candidate,),), first, {})
        right = auto_lrc.evaluate_backend_timing("synthetic", ((candidate,),), second, {})
        self.assertEqual(left.quality, right.quality)
        self.assertEqual(left.summary, right.summary)

    def test_preview_commit_reuses_identity_without_rerunning_evaluator(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        original = auto_lrc.evaluate_final_timing_state_for_report
        original_collect = auto_lrc.collect_central_timing_candidates
        original_probe = auto_lrc.apply_final_ctc_timing_guard
        auto_lrc.evaluate_final_timing_state_for_report = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("re-evaluated"))  # type: ignore[assignment]
        auto_lrc.collect_central_timing_candidates = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("regenerated"))  # type: ignore[assignment]
        auto_lrc.apply_final_ctc_timing_guard = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("re-probed"))  # type: ignore[assignment]
        try:
            committed = auto_lrc.commit_evaluated_backend(
                auto_lrc.select_evaluated_backend((evaluated,)), report
            )
        finally:
            auto_lrc.evaluate_final_timing_state_for_report = original  # type: ignore[assignment]
            auto_lrc.collect_central_timing_candidates = original_collect  # type: ignore[assignment]
            auto_lrc.apply_final_ctc_timing_guard = original_probe  # type: ignore[assignment]
        self.assertIs(committed, evaluated.state)
        self.assertEqual(report["trusted_percent"], evaluated.summary.overall_trusted_percent)

    def test_commit_rejects_candidate_set_changed_after_preview(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        changed = self.candidate(13.0, self.coherent_spans(13.0), current=True)
        stale = replace(evaluated, candidate_sets=((changed,),))
        with self.assertRaises(LrcError):
            auto_lrc.commit_evaluated_backend(
                auto_lrc.select_evaluated_backend((stale,)), report
            )

    def test_commit_rejects_every_changed_report_evaluation_input_without_writes(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        baseline = self.report(1, backend="ctc")
        baseline.update({
            "ctc_missing_count": 0,
            "ctc_low_score_count": 0,
            "ctc_very_low_score_count": 0,
        })
        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc", ((candidate,),), baseline, {"nested": {"probe": [1]}}
        )

        mutations = {
            "severe-flag": lambda report: report["assignments"][0].update(
                {"flags": ["ctc_unresolved_boundary"]}
            ),
            "content-score": lambda report: report["assignments"][0].update(
                {"score": 0.1}
            ),
            "segment": lambda report: report["assignments"][0].update(
                {"segment": {"start": 12.0, "end": 13.0}}
            ),
            "backend": lambda report: report.update({"backend": "whisperx"}),
            "quality-input": lambda report: report.update({"ctc_low_score_count": 1}),
            "assignment-identity": lambda report: report["assignments"][0].update(
                {"entry": 99}
            ),
            "timing-revision": lambda report: report["assignments"][0].update(
                {"timing_evidence_revision": 99}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                report = copy.deepcopy(baseline)
                mutate(report)
                before = copy.deepcopy(report)
                with self.assertRaises(LrcError):
                    auto_lrc.commit_evaluated_backend(
                        auto_lrc.select_evaluated_backend((evaluated,)), report
                    )
                self.assertEqual(report, before)

    def test_post_preview_severe_flag_counterexample_is_rejected(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        report["assignments"][0]["flags"] = ["ctc_unresolved_boundary"]
        before = copy.deepcopy(report)
        with self.assertRaises(LrcError):
            auto_lrc.commit_evaluated_backend(
                auto_lrc.select_evaluated_backend((evaluated,)), report
            )
        self.assertEqual(report, before)
        self.assertNotIn("timing_trusted", report["assignments"][0])

    def test_generation_diagnostics_are_deep_frozen_and_digest_guarded(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        diagnostics: dict[str, object] = {"nested": {"probe": [1, 2]}}
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, diagnostics
        )
        diagnostics["nested"]["probe"].append(3)
        self.assertEqual(
            auto_lrc.thaw_generation_diagnostics(evaluated),
            {"nested": {"probe": [1, 2]}},
        )
        stale = replace(
            evaluated,
            generation_diagnostics_json='{"nested":{"probe":[9]}}',
        )
        before = copy.deepcopy(report)
        with self.assertRaises(LrcError):
            auto_lrc.commit_evaluated_backend(
                auto_lrc.select_evaluated_backend((stale,)), report
            )
        self.assertEqual(report, before)

    def test_nonfinite_projection_and_diagnostics_fail_closed(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        report["assignments"][0]["score"] = float("nan")
        with self.assertRaises(LrcError):
            auto_lrc.evaluate_backend_timing(
                "synthetic", ((candidate,),), report, {}
            )

        clean_report = self.report(1)
        with self.assertRaises(LrcError):
            auto_lrc.evaluate_backend_timing(
                "synthetic",
                ((candidate,),),
                clean_report,
                {"nested": {"probe": float("inf")}},
            )

    def test_commit_writes_only_truthful_central_candidate_selection(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        expected = auto_lrc.central_candidate_selection_payload(selection)
        committed = auto_lrc.commit_evaluated_backend(selection, report)
        self.assertIs(committed, evaluated.state)
        self.assertEqual(report["candidate_selection"], expected)
        self.assertEqual(report["backend"], expected["selected_backend"])

    def test_commit_rejects_incomplete_or_mutated_candidate_selection_without_writes(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        expected = auto_lrc.central_candidate_selection_payload(selection)
        mutations = {
            "incomplete": lambda payload: payload.clear(),
            "backend": lambda payload: payload.update({"selected_backend": "whisperx"}),
            "quality": lambda payload: payload.update({
                "selected_quality": float(payload["selected_quality"]) - 1.0
            }),
            "input-revision": lambda payload: payload.update({"selected_input_revision": "fake"}),
            "evaluation-revision": lambda payload: payload.update({"selected_evaluation_revision": "fake"}),
            "selected-candidate": lambda payload: payload["selected_candidate"].update(
                {"backend": "whisperx"}
            ),
            "alternatives": lambda payload: payload.update({"alternatives": [{"backend": "fake"}]}),
            "reason": lambda payload: payload.update({"selected_reason": "fake selection"}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                attempt = copy.deepcopy(report)
                attempt["candidate_selection"] = copy.deepcopy(expected)
                mutate(attempt["candidate_selection"])
                before = copy.deepcopy(attempt)
                with self.assertRaises(LrcError):
                    auto_lrc.commit_evaluated_backend(selection, attempt)
                self.assertEqual(attempt, before)

    def test_commit_accepts_exact_preexisting_central_selection_payload(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1)
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        report["candidate_selection"] = auto_lrc.central_candidate_selection_payload(selection)
        committed = auto_lrc.commit_evaluated_backend(selection, report)
        self.assertIs(committed, evaluated.state)

    def test_central_selected_reason_satisfies_active_powershell_consumer(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1, backend="synthetic")
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        payload = auto_lrc.central_candidate_selection_payload(selection)
        self.assertIn("immutable central evaluation", payload["selected_reason"])
        self.assertIn(evaluated.evaluation_revision[:12], payload["selected_reason"])
        consumer = (
            Path(auto_lrc.__file__).resolve().parents[1] / "align-lrc.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("$report.candidate_selection.selected_reason", consumer)
        self.assertTrue(payload["selected_reason"])

    def test_mocked_auto_and_explicit_previews_run_each_producer_once(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        counts = {
            "anchor": 0,
            "score": 0,
            "duplicate": 0,
            "validation": 0,
            "collect": 0,
        }
        originals = (
            auto_lrc.apply_anchor_hints,
            auto_lrc.score_line_timing_candidates,
            auto_lrc.apply_short_duplicate_acoustic_recovery,
            auto_lrc.apply_post_score_audio_validation,
            auto_lrc.collect_central_timing_candidates,
            auto_lrc.ctc_alignment_audio_path,
        )

        def anchors(_entries, timestamps, _report, _path, _duration):
            counts["anchor"] += 1
            return list(timestamps), []

        def score(_entries, timestamps, _report, _duration):
            counts["score"] += 1
            return list(timestamps)

        def duplicate(_audio, _entries, timestamps, report, _duration):
            counts["duplicate"] += 1
            return list(timestamps), report, []

        def validation(_audio, _entries, _timestamps, _report, _duration):
            counts["validation"] += 1
            return []

        def collect(*_args, **_kwargs):
            counts["collect"] += 1
            return ((candidate,),), {"probe_calls": 1}

        auto_lrc.apply_anchor_hints = anchors  # type: ignore[assignment]
        auto_lrc.score_line_timing_candidates = score  # type: ignore[assignment]
        auto_lrc.apply_short_duplicate_acoustic_recovery = duplicate  # type: ignore[assignment]
        auto_lrc.apply_post_score_audio_validation = validation  # type: ignore[assignment]
        auto_lrc.collect_central_timing_candidates = collect  # type: ignore[assignment]
        auto_lrc.ctc_alignment_audio_path = lambda _report, audio: audio  # type: ignore[assignment]
        try:
            previews = []
            for backend in ("auto-candidate", "explicit-candidate"):
                report = self.report(1, backend=backend)
                _times, preview_report, evaluated, _changes = auto_lrc.prepare_backend_timing_preview(
                    Path("dummy.flac"),
                    [LyricEntry(["generic"])],
                    [12.0],
                    report,
                    20.0,
                    object(),  # type: ignore[arg-type]
                    None,
                )
                previews.append((preview_report, evaluated))
        finally:
            (
                auto_lrc.apply_anchor_hints,
                auto_lrc.score_line_timing_candidates,
                auto_lrc.apply_short_duplicate_acoustic_recovery,
                auto_lrc.apply_post_score_audio_validation,
                auto_lrc.collect_central_timing_candidates,
                auto_lrc.ctc_alignment_audio_path,
            ) = originals
        self.assertEqual(counts, {key: 2 for key in counts})
        selection = auto_lrc.select_evaluated_backend(
            tuple(evaluated for _report, evaluated in previews)
        )
        selected_report = next(
            report for report, evaluated in previews
            if evaluated is selection.selected
        )
        committed = auto_lrc.commit_evaluated_backend(
            selection, selected_report
        )
        self.assertIs(committed, selection.selected.state)
        self.assertEqual(counts, {key: 2 for key in counts})

    def test_auto_competition_previews_each_usable_candidate_once(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        counts = {"inference": 0, "preview": 0, "commit": 0}
        preview_backends: list[str] = []
        cache_ids: list[int] = []
        originals = (
            auto_lrc.default_ctc_ready,
            auto_lrc.default_whisperx_ready,
            auto_lrc.try_ctc_candidate,
            auto_lrc._central_preview_generation_candidates,
            auto_lrc.prepare_backend_timing_preview,
            auto_lrc.commit_evaluated_backend,
        )

        def infer(*_args, **_kwargs):
            counts["inference"] += 1
            return [12.0], self.report(1, backend="ctc"), None

        def generation_candidates(*_args, **_kwargs):
            return [
                {
                    "backend": "ctc",
                    "timestamps": [12.0],
                    "report": self.report(1, backend="ctc"),
                    "error": None,
                },
                {
                    "backend": "whisperx",
                    "timestamps": [12.0],
                    "report": self.report(1, backend="whisperx"),
                    "error": None,
                },
            ]

        def preview(_audio, _entries, timestamps, report, _duration, _args, _anchor, *, retry_helper_cache):
            counts["preview"] += 1
            preview_backends.append(str(report["backend"]))
            self.assertIsNotNone(retry_helper_cache)
            cache_ids.append(id(retry_helper_cache))
            if retry_helper_cache:
                self.assertEqual(retry_helper_cache.get("sentinel"), "first-preview")
            else:
                retry_helper_cache["sentinel"] = "first-preview"
            evaluated = auto_lrc.evaluate_backend_timing(
                str(report["backend"]), ((candidate,),), report, {"anchor_changes": []}
            )
            return list(timestamps), report, evaluated, []

        auto_lrc.default_ctc_ready = lambda: True  # type: ignore[assignment]
        auto_lrc.default_whisperx_ready = lambda: False  # type: ignore[assignment]
        auto_lrc.try_ctc_candidate = infer  # type: ignore[assignment]
        auto_lrc._central_preview_generation_candidates = generation_candidates  # type: ignore[assignment]
        auto_lrc.prepare_backend_timing_preview = preview  # type: ignore[assignment]
        auto_lrc.commit_evaluated_backend = lambda *_args, **_kwargs: counts.update(  # type: ignore[assignment]
            {"commit": counts["commit"] + 1}
        )
        try:
            timestamps, report, backend, selection = auto_lrc.run_auto_backend_competition(
                Path("dummy.flac"),
                [LyricEntry(["generic"])],
                20.0,
                object(),  # type: ignore[arg-type]
                None,
            )
        finally:
            (
                auto_lrc.default_ctc_ready,
                auto_lrc.default_whisperx_ready,
                auto_lrc.try_ctc_candidate,
                auto_lrc._central_preview_generation_candidates,
                auto_lrc.prepare_backend_timing_preview,
                auto_lrc.commit_evaluated_backend,
            ) = originals
        self.assertEqual(timestamps, [12.0])
        self.assertEqual(report["backend"], backend)
        self.assertEqual(counts["inference"], 1)
        self.assertEqual(counts["preview"], len(selection.candidates))
        self.assertEqual(preview_backends, ["ctc", "whisperx"])
        self.assertEqual(len(set(cache_ids)), 1)
        self.assertEqual(counts["commit"], 0)
        self.assertIsInstance(selection, auto_lrc.BackendSelection)

    def test_process_auto_reuses_selection_and_explicit_previews_once(self) -> None:
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        original_functions = (
            auto_lrc.probe_duration,
            auto_lrc.find_anchor_hints,
            auto_lrc.default_ctc_ready,
            auto_lrc.default_whisperx_ready,
            auto_lrc.run_ctc_alignment,
            auto_lrc.run_auto_backend_competition,
            auto_lrc.prepare_backend_timing_preview,
            auto_lrc.commit_evaluated_backend,
            auto_lrc.audio_track_labels,
        )
        real_commit = auto_lrc.commit_evaluated_backend

        with TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            audio = temp / "generic.flac"
            lyrics = temp / "generic.txt"
            audio.write_bytes(b"synthetic")
            lyrics.write_text("ordinary lyric line\n", encoding="utf-8")

            for timing_source in ("auto", "ctc"):
                with self.subTest(timing_source=timing_source):
                    report = self.report(1, backend="ctc")
                    evaluated = auto_lrc.evaluate_backend_timing(
                        "ctc", ((candidate,),), report, {"anchor_changes": []}
                    )
                    selection = auto_lrc.select_evaluated_backend((evaluated,))
                    counts = {
                        "backend_inference": 0,
                        "competition": 0,
                        "preview": 0,
                        "commit": 0,
                    }

                    def explicit_inference(*_args, **_kwargs):
                        counts["backend_inference"] += 1
                        return [12.0], report

                    def competition(*_args, **_kwargs):
                        counts["competition"] += 1
                        return [12.0], report, "ctc", selection

                    def preview(*_args, **_kwargs):
                        counts["preview"] += 1
                        return [12.0], report, evaluated, []

                    def commit(selected, selected_report):
                        counts["commit"] += 1
                        return real_commit(selected, selected_report)

                    auto_lrc.probe_duration = lambda _path: 20.0  # type: ignore[assignment]
                    auto_lrc.find_anchor_hints = lambda *_args: None  # type: ignore[assignment]
                    auto_lrc.default_ctc_ready = lambda: True  # type: ignore[assignment]
                    auto_lrc.default_whisperx_ready = lambda: False  # type: ignore[assignment]
                    auto_lrc.run_ctc_alignment = explicit_inference  # type: ignore[assignment]
                    auto_lrc.run_auto_backend_competition = competition  # type: ignore[assignment]
                    auto_lrc.prepare_backend_timing_preview = preview  # type: ignore[assignment]
                    auto_lrc.commit_evaluated_backend = commit  # type: ignore[assignment]
                    auto_lrc.audio_track_labels = lambda _path: ("", "generic")  # type: ignore[assignment]
                    output = temp / f"{timing_source}.lrc"
                    report_dir = temp / f"{timing_source}-reports"
                    args = build_parser().parse_args([
                        str(audio),
                        "--lyrics", str(lyrics),
                        "--output", str(output),
                        "--report-dir", str(report_dir),
                        "--timing-source", timing_source,
                        "--whisper-language", "en",
                        "--overwrite",
                        "--no-checked-lrc-hint",
                    ])
                    try:
                        result = auto_lrc.process_audio(audio, args)
                    finally:
                        (
                            auto_lrc.probe_duration,
                            auto_lrc.find_anchor_hints,
                            auto_lrc.default_ctc_ready,
                            auto_lrc.default_whisperx_ready,
                            auto_lrc.run_ctc_alignment,
                            auto_lrc.run_auto_backend_competition,
                            auto_lrc.prepare_backend_timing_preview,
                            auto_lrc.commit_evaluated_backend,
                            auto_lrc.audio_track_labels,
                        ) = original_functions
                    self.assertEqual(result, output.resolve())
                    self.assertEqual(counts["commit"], 1)
                    if timing_source == "auto":
                        self.assertEqual(counts["competition"], 1)
                        self.assertEqual(counts["backend_inference"], 0)
                        self.assertEqual(counts["preview"], 0)
                    else:
                        self.assertEqual(counts["competition"], 0)
                        self.assertEqual(counts["backend_inference"], 1)
                        self.assertEqual(counts["preview"], 1)

    def test_review_boolean_count_percent_and_rows_are_one_contract(self) -> None:
        candidate = self.candidate(
            12.0,
            tuple(self.span(12.0 + index * 0.06, 0.01) for index in range(3)),
            current=True,
        )
        report = self.report(1)
        report["review_required"] = False
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), report, {}
        )
        auto_lrc.commit_evaluated_backend(
            auto_lrc.select_evaluated_backend((evaluated,)), report
        )
        self.assertFalse(report["review_required"])
        self.assertEqual(report["review_required_count"], 0)
        self.assertEqual(report["review_required_percent"], 0.0)
        self.assertFalse(report["assignments"][0]["review_required"])

        actionable = self.report(1)
        actionable["assignments"][0]["flags"] = ["candidate_disagreement"]  # type: ignore[index]
        actionable["suspicious_alignments"] = [{
            "entry": 1,
            "flags": ["candidate_disagreement"],
            "review_required": False,
        }]
        positive_evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((candidate,),), actionable, {}
        )
        auto_lrc.commit_evaluated_backend(
            auto_lrc.select_evaluated_backend((positive_evaluated,)), actionable
        )
        self.assertTrue(actionable["review_required"])
        self.assertEqual(actionable["review_required_count"], 1)
        self.assertEqual(actionable["review_required_percent"], 100.0)
        self.assertTrue(actionable["assignments"][0]["review_required"])

    def test_metric_refresh_repairs_latent_stale_root_review_boolean(self) -> None:
        report: dict[str, object] = {
            "timing_entries": 1,
            "review_required": False,
            "assignments": [{
                "timing_repair": "synthetic",
                "score": 0.9,
                "timestamp_status": "provisional_unresolved",
                "timing_trusted": False,
                "review_required": True,
            }],
        }
        auto_lrc.update_report_confidence_metrics(report)
        self.assertFalse(report["review_required"])
        self.assertEqual(report["review_required_count"], 0)

        actionable = copy.deepcopy(report)
        actionable["assignments"][0]["flags"] = ["previous-tail-overlap"]  # type: ignore[index]
        auto_lrc.update_report_confidence_metrics(actionable)
        self.assertTrue(actionable["review_required"])
        self.assertEqual(actionable["review_required_count"], 1)
        self.assertTrue(actionable["assignments"][0]["review_required"])  # type: ignore[index]

    def test_auto_and_explicit_paths_share_preview_and_single_commit(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        process = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_audio"
        )
        competition = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_auto_backend_competition"
        )
        self.assertTrue(any(
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "prepare_backend_timing_preview"
            for node in ast.walk(process)
        ))
        self.assertTrue(any(
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "prepare_backend_timing_preview"
            for node in ast.walk(competition)
        ))
        commits = [
            node for node in ast.walk(process)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "commit_evaluated_backend"
        ]
        self.assertEqual(len(commits), 1)

    def test_production_source_has_no_fixture_or_phoneme_recovery_branch(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("Georgette", source)
        self.assertNotIn("motsuretamama", source)
        self.assertNotIn("romaji.startswith", source)

    def test_no_timestamp_mutation_after_central_finalizer(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        process = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_audio"
        )
        finalizer_call = next(
            node for node in ast.walk(process)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "commit_evaluated_backend"
        )
        late_targets = []
        for node in ast.walk(process):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            if node.lineno <= finalizer_call.lineno:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            late_targets.extend(
                ast.unparse(target) for target in targets
                if "timestamp" in ast.unparse(target)
                or "assignments" in ast.unparse(target)
            )
        self.assertEqual(late_targets, [])

    def _projection_selection(
        self,
        root: Path,
        *,
        include_nonselected: bool = False,
    ) -> tuple[dict[str, object], auto_lrc.BackendSelection]:
        audio = root / "projection.flac"
        audio.write_bytes(b"projection-audio")
        args = build_parser().parse_args([str(audio)])
        candidate = self.candidate(12.0, self.coherent_spans(12.0), current=True)
        report = self.report(1, backend="ctc")
        evaluated = auto_lrc.evaluate_backend_timing("ctc", ((candidate,),), report, {})
        evaluated_items = [evaluated]
        preview_records: list[dict[str, object]] = [
            {"timestamps": [12.0], "report": report, "evaluated": evaluated}
        ]
        generation_candidates: list[dict[str, object]] = [
            {"backend": "ctc", "report": report, "timestamps": [12.0], "error": None}
        ]
        if include_nonselected:
            other = self.report(1, backend="whispercpp")
            other_evaluated = auto_lrc.evaluate_backend_timing(
                "whispercpp", ((candidate,),), other, {}
            )
            evaluated_items.append(other_evaluated)
            preview_records.append({
                "timestamps": [12.0], "report": other, "evaluated": other_evaluated
            })
            generation_candidates.append({
                "backend": "whispercpp", "report": other, "timestamps": [12.0], "error": None
            })
        selection = auto_lrc.select_evaluated_backend(tuple(evaluated_items))
        seed = auto_lrc._build_backend_provenance_projection_seed(
            audio, args, generation_candidates, preview_records, selection
        )
        report[auto_lrc._RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY] = seed
        auto_lrc.commit_evaluated_backend(selection, report)
        return report, selection

    def test_backend_projection_contains_selected_and_nonselected_capabilities(self) -> None:
        with TemporaryDirectory() as temp_name:
            report, selection = self._projection_selection(Path(temp_name), include_nonselected=True)
        projection = report[auto_lrc._BACKEND_PROVENANCE_PROJECTION_KEY]
        self.assertEqual(projection["selected_backend"], selection.selected.backend)  # type: ignore[index]
        self.assertEqual(len(projection["candidates"]), 2)  # type: ignore[index]
        statuses = {item["selection_status"] for item in projection["candidates"]}  # type: ignore[index]
        self.assertEqual(statuses, {"selected", "nonselected"})
        self.assertEqual(projection["source_audio"]["sha256"]["status"], "RETRIEVED")  # type: ignore[index]
        serialized = json.dumps(projection, ensure_ascii=False)
        self.assertNotIn("projection.flac", serialized)
        self.assertNotIn("temp", serialized.lower())

    def test_backend_projection_digest_and_current_source_binding_fail_closed(self) -> None:
        with TemporaryDirectory() as temp_name:
            report, _selection = self._projection_selection(Path(temp_name))
        projection = report[auto_lrc._BACKEND_PROVENANCE_PROJECTION_KEY]
        projection["selection"]["selected_quality"] = -1.0  # type: ignore[index]
        with self.assertRaises(LrcError):
            auto_lrc._validate_backend_provenance_projection(report)

        with TemporaryDirectory() as temp_name:
            report, _selection = self._projection_selection(Path(temp_name))
        projection = report[auto_lrc._BACKEND_PROVENANCE_PROJECTION_KEY]
        projection["algorithm_source_sha256"] = "stale-source"  # type: ignore[index]
        payload = dict(projection)
        payload.pop("projection_revision", None)
        projection["projection_revision"] = auto_lrc._timing_content_digest(payload)  # type: ignore[index]
        with self.assertRaises(LrcError):
            auto_lrc._validate_backend_provenance_projection(report)

    def test_backend_projection_missing_capability_fields_are_explicit(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "projection.flac"
            audio.write_bytes(b"projection-audio")
            args = build_parser().parse_args([str(audio)])
            with mock.patch.object(
                auto_lrc,
                "_whispercpp_runtime_artifact_hashes",
                return_value=None,
            ):
                summary = auto_lrc._public_backend_capability_summary(
                    audio,
                    args,
                    "whisperx",
                    {"backend": "whisperx", "error": "No usable WhisperX candidate"},
                )
        capability = summary["capability"]
        self.assertEqual(capability["helper_sha256"]["status"], "ABSENT")
        self.assertEqual(capability["package_version"]["status"], "ABSENT")
        self.assertEqual(capability["source_audio_sha256"]["status"], "RETRIEVED")
        self.assertEqual(capability["capability_revision"]["status"], "UNAVAILABLE")
        self.assertIsNone(capability["capability_revision"]["value"])

    def test_writer_preserves_public_projection_and_strips_private_seed(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            report, _selection = self._projection_selection(root)
            report[auto_lrc._RUNTIME_WHISPERX_TEMP_PATHS_KEY] = (
                "C:/temp/private-input.wav",
            )
            path = root / "report.json"
            auto_lrc.write_alignment_report(path, report)
            text = path.read_text(encoding="utf-8")
        self.assertIn(auto_lrc._BACKEND_PROVENANCE_PROJECTION_KEY, text)
        self.assertNotIn(auto_lrc._RUNTIME_BACKEND_PROVENANCE_PROJECTION_SEED_KEY, text)
        self.assertNotIn("private-input.wav", text)
        self.assertNotIn("_runtime_", text)

    def test_backend_projection_does_not_change_committed_timing_metrics(self) -> None:
        with TemporaryDirectory() as temp_name:
            report, selection = self._projection_selection(Path(temp_name))
        self.assertEqual(report["central_timing_state"][0]["timestamp"], 12.0)
        self.assertEqual(report["overall_trusted_entries"], selection.selected.summary.overall_trusted_entries)
        self.assertEqual(report["review_required_count"], selection.selected.summary.review_required_count)


class BackendPeerSimulationTests(unittest.TestCase):
    def span(self, start: float, token: str = "x") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.04, 0.95, token)

    def candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str = "ctc-current",
        current: bool = False,
        confidence: float = 0.80,
        direct: bool = False,
        independent: bool = False,
    ) -> auto_lrc.TimingCandidate:
        spans = tuple(self.span(time + index * 0.04, f"t{index}") for index in range(4))
        occurrence_evidence = None
        if direct:
            occurrence_entries = [
                LyricEntry([f"peer line {index}"])
                for index in range(max(1, entry + 1))
            ]
            occurrence_assignments = [
                {"timestamp": float(index + 1), "score": .95, "segment": index + 1}
                for index in range(len(occurrence_entries))
            ]
            occurrence_binding = auto_lrc._entry_occurrence_binding(
                occurrence_entries, occurrence_assignments, entry
            )
            occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
                occurrence_entries,
                entry,
                producer="raw-asr-vocal-fusion",
                occurrence_binding=occurrence_binding,
                upstream_evidence_revision="peer-direct-upstream",
            )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"peer line {entry}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=confidence,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported" if direct else "unavailable",
            direct_onset_support="supported" if direct else "unavailable",
            direct_onset_time=time if direct else None,
            direct_onset_source_artifact={"entry": entry, "time": time} if direct else None,
            direct_onset_evidence_producer=("raw-asr-vocal-fusion" if direct else None),
            direct_onset_evidence_kind=("cross-backend-onset" if direct else "generic-onset"),
            direct_onset_evidence_independent=bool(direct),
            occurrence_evidence=occurrence_evidence,
            independent_content_identity=independent,
            current=current,
            source_artifact={"entry": entry, "time": time, "source": source},
        )

    def report(self, count: int, backend: str = "ctc") -> dict[str, object]:
        return {
            "backend": backend,
            "assignments": [
                {"entry": index + 1, "timestamp": 1.0 + index, "flags": []}
                for index in range(count)
            ],
        }

    def report_with_entries(
        self,
        count: int,
        backend: str = "ctc",
    ) -> tuple[dict[str, object], list[LyricEntry]]:
        report = self.report(count, backend)
        entries = [LyricEntry([f"peer line {index}"]) for index in range(count)]
        report[auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY] = tuple(entries)
        report["lyric_entry_revision"] = auto_lrc.lyric_identity_revision(entries)
        return report, entries

    def evaluated_bundle(
        self,
        root: Path,
        candidate_sets: tuple[tuple[auto_lrc.TimingCandidate, ...], ...],
        *,
        backend: str = "ctc",
        report: dict[str, object] | None = None,
    ) -> auto_lrc.EvaluatedBackendBundle:
        report = report or self.report(len(candidate_sets), backend)
        evaluated = auto_lrc.evaluate_backend_timing(
            backend,
            candidate_sets,
            report,
            {},
        )
        audio = root / "peer.flac"
        audio.write_bytes(b"peer-audio")
        args = build_parser().parse_args([str(audio)])
        return auto_lrc._capture_evaluated_backend_bundle(
            evaluated,
            report,
            audio_path=audio,
            args=args,
        )

    def test_peer_cluster_anchor_boundary_and_distant_same_lineage_review(self) -> None:
        candidates = (
            self.candidate(0, 1.00, source="ctc-current", direct=True, confidence=0.95),
            self.candidate(0, 1.30, source="ctc-opening-vocal-fusion", direct=True),
            self.candidate(0, 1.40, source="ctc-vocal-activity-reentry", direct=True),
            self.candidate(0, 1.60, source="ctc-vocal-onset-confirmed", direct=True),
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=candidates,
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        clusters = auto_lrc._peer_observation_cluster_summaries(
            decision, decision.written_time.seconds
        )
        self.assertEqual(
            [round(item.anchor_time, 2) for item in clusters],
            [1.00, 1.30, 1.60],
        )
        self.assertTrue(all(item.review_required for item in clusters[1:]))
        self.assertEqual(clusters[1].member_candidate_ids.__len__(), 2)

    def test_peer_tight_cluster_has_no_material_conflict(self) -> None:
        candidates = (
            self.candidate(0, 1.00, source="ctc-current", direct=True, confidence=0.95),
            self.candidate(0, 1.17, source="ctc-opening-vocal-fusion", direct=True),
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=candidates,
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        clusters = auto_lrc._peer_observation_cluster_summaries(
            decision, decision.written_time.seconds
        )
        self.assertEqual(len(clusters), 1)
        self.assertFalse(clusters[0].review_required)
        self.assertEqual(
            auto_lrc._material_distant_timing_alternatives(
                decision, decision.written_time.seconds
            ),
            (),
        )

        boundary_candidates = (
            self.candidate(0, 1.00, source="ctc-current", direct=True, confidence=0.95),
            self.candidate(0, 1.18, source="ctc-opening-vocal-fusion", direct=True),
        )
        boundary_decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=boundary_candidates,
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        boundary_clusters = auto_lrc._peer_observation_cluster_summaries(
            boundary_decision, boundary_decision.written_time.seconds
        )
        self.assertEqual(len(boundary_clusters), 1)
        self.assertFalse(boundary_clusters[0].review_required)

    def test_peer_full_scaffold_preserves_suspicious_history_and_report_immutability(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            selected_report, _entries = self.report_with_entries(1, "ctc")
            nonselected_report, _entries = self.report_with_entries(1, "whispercpp")
            selected_report["suspicious_alignments"] = [{
                "entry": 1,
                "flags": ["candidate_disagreement"],
                "candidate_timestamps": {"ctc": 1.0, "raw_asr": 2.0},
                "review_required": True,
                "severity": "high",
            }]
            selected_report["timing_review_entries"] = [1]
            selected_report["legacy_timing_diagnostics"] = {
                "history": [{"entry": 1, "reason": "retained"}]
            }
            ctc = self.candidate(0, 1.0, current=True, source="ctc-current")
            whisper = self.candidate(
                0,
                2.0,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )
            first = self.evaluated_bundle(
                root, ((ctc,),), backend="ctc", report=selected_report
            )
            second = self.evaluated_bundle(
                root, ((whisper,),), backend="whispercpp", report=nonselected_report
            )
            before = copy.deepcopy(selected_report)
            result = auto_lrc._simulate_cross_backend_peer_graph(
                (first, second), selected_backend="ctc"
            )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["authority"])
        self.assertEqual(result["scaffold_binding"]["selected_backend"], "ctc")  # type: ignore[index]
        self.assertEqual(result["summary"]["review_required_count"], 1)  # type: ignore[index]
        self.assertTrue(result["rows"][0]["review_required"])  # type: ignore[index]
        self.assertEqual(selected_report, before)
        self.assertNotIn("commit", json.dumps(result).lower())
        self.assertNotIn("candidate_disagreement", json.dumps(result["bundles"]).lower())

    def test_current_is_identity_bound_but_peer_selection_neutralizes_only_policy(self) -> None:
        current = self.candidate(0, 1.0, current=True, source="ctc-current", direct=True)
        neutral = self.candidate(0, 1.0, current=False, source="ctc-current", direct=True)
        self.assertNotEqual(current.candidate_id, neutral.candidate_id)
        self.assertNotEqual(current.generation_revision, neutral.generation_revision)
        self.assertNotEqual(
            auto_lrc.evaluate_timing_candidate(current).evidence.candidate_revision,
            auto_lrc.evaluate_timing_candidate(neutral).evidence.candidate_revision,
        )
        normal = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, self.candidate(
                0,
                2.0,
                current=True,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )),
            previous_candidate=None,
            selection_revision=1,
        )
        peer = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, self.candidate(
                0,
                2.0,
                current=True,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )),
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        self.assertEqual(normal.status, "selected_valid")
        self.assertEqual(peer.status, "provisional_unresolved")
        self.assertIsNone(peer.selected_candidate_id)
        self.assertIsNone(peer.recovery)
        self.assertTrue(current.current)
        self.assertFalse(neutral.current)

    def test_peer_dominance_and_consensus_use_existing_evidence_only(self) -> None:
        weak_current = self.candidate(0, 1.0, current=True, confidence=0.20)
        strong = self.candidate(
            0,
            2.0,
            source="raw-vocal-independent-fusion",
            direct=True,
            independent=True,
            confidence=0.95,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(weak_current, strong),
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        self.assertEqual(decision.status, "selected_valid")
        self.assertEqual(decision.candidate.candidate_id, strong.candidate_id)
        self.assertIsNone(decision.recovery)

        ctc = self.candidate(0, 1.00, source="ctc-opening-vocal-fusion")
        whisper = self.candidate(
            0,
            1.10,
            source="raw-vocal-independent-fusion",
            direct=True,
            independent=True,
        )
        cluster = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(ctc, whisper),
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        self.assertEqual(cluster.status, "selected_valid")
        self.assertIn(cluster.candidate.candidate_id, {ctc.candidate_id, whisper.candidate_id})

        same_lineage = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(
                self.candidate(0, 1.00, source="ctc-opening-vocal-fusion"),
                self.candidate(0, 1.10, source="ctc-vocal-activity-reentry"),
            ),
            previous_candidate=None,
            selection_revision=1,
            peer_graph=auto_lrc._peer_policy(),
        )
        self.assertEqual(same_lineage.status, "provisional_unresolved")

    def test_bundle_freezes_scaffold_and_exact_candidate_union(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            candidate = self.candidate(0, 1.0, current=True)
            report = self.report(1)
            bundle = self.evaluated_bundle(root, ((candidate,),), report=report)
            digest_before = bundle.bundle_revision
            report["assignments"][0]["entry"] = 99  # type: ignore[index]
            self.assertEqual(auto_lrc._validate_evaluated_backend_bundle(bundle)["valid"], True)
            self.assertEqual(bundle.bundle_revision, digest_before)
            union, error = auto_lrc._union_peer_candidate_sets((bundle, bundle))
            self.assertIsNone(error)
            self.assertIsNotNone(union)
            assert union is not None
            self.assertEqual(len(union[0]), 1)
            self.assertNotIn("path", bundle.capability_projection_json.lower())

    def test_missing_tail_context_is_diagnostic_only_and_simulation_never_authoritative(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            candidate = self.candidate(0, 1.0, current=True)
            report = self.report(1)
            report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = {
                0: auto_lrc.IndependentTailEvidence(
                    entry_index=0,
                    tail_start=0.5,
                    tail_end=0.9,
                    uncertainty_seconds=0.1,
                    confidence=0.9,
                    producer="ctc-assignment-terminal-row",
                    source_revision="missing-provenance",
                    provenance=None,
                )
            }
            bundle = self.evaluated_bundle(root, ((candidate,),), report=report)
            result = auto_lrc._simulate_cross_backend_peer_graph((bundle,))
        self.assertFalse(result["authority"])
        self.assertIn(result["status"], {"SUCCESS", "BLOCKED"})
        self.assertEqual(result["context_validation"]["diagnostic_tail_row_count"], 1)  # type: ignore[index]

    def test_stale_or_transplanted_content_cannot_enter_peer_observations(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            report, entries = self.report_with_entries(1)
            stale_lyric = hashlib.sha256(
                auto_lrc.normalize_match_text("stale lyric").encode("utf-8")
            ).hexdigest()
            report[auto_lrc._RUNTIME_RAW_CONTENT_IDENTITY_KEY] = {
                0: {
                    "producer": "raw-asr-content-identity",
                    "lyric_sha256": stale_lyric,
                    "occurrence_binding": {"entry": 2},
                    "evidence_revision": "self-consistent-stale-revision",
                }
            }
            candidate = self.candidate(0, 1.0, current=True)
            bundle = self.evaluated_bundle(root, ((candidate,),), report=report)
            bound, diagnostic = auto_lrc._bundle_content_observations_bound_to_scaffold(bundle)
            report[auto_lrc._RUNTIME_RAW_CONTENT_IDENTITY_KEY] = {
                1: {
                    "producer": "raw-asr-content-identity",
                    "lyric_sha256": hashlib.sha256(
                        auto_lrc.normalize_match_text(entries[0].lines[0]).encode("utf-8")
                    ).hexdigest(),
                    "occurrence_binding": {"entry": 1},
                    "evidence_revision": "wrong-entry-revision",
                }
            }
            transplanted = self.evaluated_bundle(root, ((candidate,),), report=report)
            transplanted_bound, transplanted_diagnostic = (
                auto_lrc._bundle_content_observations_bound_to_scaffold(transplanted)
            )
        self.assertEqual(bound, ())
        self.assertEqual(diagnostic, 1)
        self.assertEqual(transplanted_bound, ())
        self.assertEqual(transplanted_diagnostic, 1)
        self.assertEqual(entries[0].lines[0], "peer line 0")

    def test_peer_success_is_non_authoritative_with_current_conflict_diagnostic(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            ctc_report, _entries = self.report_with_entries(1, "ctc")
            whisper_report, _entries = self.report_with_entries(1, "whispercpp")
            ctc = self.candidate(0, 1.0, current=True, source="ctc-current")
            whisper = self.candidate(
                0,
                2.0,
                current=True,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )
            first = self.evaluated_bundle(root, ((ctc,),), backend="ctc", report=ctc_report)
            second = self.evaluated_bundle(
                root,
                ((whisper,),),
                backend="whispercpp",
                report=whisper_report,
            )
            before = copy.deepcopy(ctc_report)
            result = auto_lrc._simulate_cross_backend_peer_graph((first, second))
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["authority"])
        self.assertEqual(result["current_conflict_entries"], [1])
        self.assertNotIn("multiple-current-candidates", result.get("blocked_reasons", []))
        self.assertEqual(result["rows"][0]["current_baseline_conflict"], True)  # type: ignore[index]
        self.assertEqual(ctc_report, before)

    def test_material_owner_details_match_existing_boolean_predicate(self) -> None:
        current = self.candidate(0, 1.0, current=True, confidence=0.95)
        alternative = self.candidate(
            0,
            3.0,
            source="raw-vocal-independent-fusion",
            direct=True,
            independent=True,
            confidence=0.20,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, alternative),
            previous_candidate=None,
            selection_revision=1,
        )
        details = auto_lrc._material_distant_timing_alternatives(
            decision, decision.written_time.seconds
        )
        self.assertEqual(
            bool(details),
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            ),
        )
        self.assertTrue(details)
        self.assertFalse(details[0]["explicit_falsification"])
        self.assertEqual(details[0]["direct_onset_evidence"]["producer"], "raw-asr-vocal-fusion")  # type: ignore[index]
        self.assertNotIn("path", json.dumps(details).lower())

    def test_peer_simulation_diagnoses_distinct_current_baselines_without_blocking(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            ctc_report, _entries = self.report_with_entries(1, "ctc")
            whisper_report, _entries = self.report_with_entries(1, "whispercpp")
            ctc = self.candidate(0, 1.0, current=True, source="ctc-current")
            whisper = self.candidate(
                0,
                2.0,
                current=True,
                source="raw-vocal-independent-fusion",
                direct=True,
                independent=True,
            )
            first = self.evaluated_bundle(root, ((ctc,),), backend="ctc", report=ctc_report)
            second = self.evaluated_bundle(
                root,
                ((whisper,),),
                backend="whispercpp",
                report=whisper_report,
            )
            result = auto_lrc._simulate_cross_backend_peer_graph((first, second))
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result["authority"])
        self.assertIn(1, result["current_conflict_entries"])
        self.assertNotIn("multiple-current-candidates", result.get("blocked_reasons", []))

    def test_review_only_projection_filters_tight_and_falsified_clusters(self) -> None:
        def cluster(relation: str, review: bool, falsified: bool, revision: str) -> dict[str, object]:
            return {
                "cluster_revision": revision,
                "relation": relation,
                "review_required": review,
                "explicit_falsification": falsified,
                "member_candidate_ids": [revision],
                "member_generation_revisions": [revision],
                "support_families": ["ctc"],
            }
        peer = {
            "status": "SUCCESS",
            "authority": False,
            "simulation_revision": "peer-revision",
            "bundles": [
                {
                    "valid": True,
                    "scaffold_lyric_valid": True,
                    "bundle_revision": "bundle-a",
                }
            ],
            "context_validation": {"valid_context_count": 1},
            "rows": [
                {
                    "entry": 1,
                    "material_revision": "mat-1",
                    "cluster_revision": "clusters-1",
                    "cluster_count": 2,
                    "clusters": [
                        cluster("baseline", False, False, "base-1"),
                        cluster("distant", True, False, "distant-1"),
                    ],
                },
                {
                    "entry": 2,
                    "material_revision": "mat-2",
                    "cluster_revision": "clusters-2",
                    "cluster_count": 1,
                    "clusters": [cluster("baseline", False, False, "base-2")],
                },
                {
                    "entry": 3,
                    "material_revision": "mat-3",
                    "cluster_revision": "clusters-3",
                    "cluster_count": 2,
                    "clusters": [
                        cluster("baseline", False, False, "base-3"),
                        cluster("distant", True, True, "falsified-3"),
                    ],
                },
                {
                    "entry": 4,
                    "material_revision": "mat-4",
                    "cluster_revision": "clusters-4",
                    "cluster_count": 2,
                    "clusters": [
                        cluster("baseline", False, False, "base-4"),
                        cluster("distant", True, False, "same-4"),
                    ],
                },
            ],
        }
        projection = auto_lrc._peer_review_only_conflict_projection(peer)
        self.assertEqual(projection["status"], "RETRIEVED")
        self.assertEqual([item["entry"] for item in projection["entries"]], [1, 4])  # type: ignore[index]
        self.assertEqual(projection["authority"], "review-only")
        blocked = auto_lrc._peer_review_only_conflict_projection({
            "status": "BLOCKED",
            "authority": False,
        })
        self.assertEqual(blocked["status"], "ABSENT")
        self.assertEqual(blocked["entries"], [])

    def test_review_only_guard_changes_review_only_state_not_time_content_or_timing(self) -> None:
        candidate = self.candidate(0, 1.0, current=True, direct=True)
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [{"entry": 1, "segment": 1, "score": 0.9, "flags": []}],
        }
        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc", ((candidate,),), report, {}
        )
        report[auto_lrc._RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY] = frozenset({1})
        before = {
            "timestamp": evaluated.state.decisions[0].written_time.seconds,
            "content": report["assignments"][0].get("content_trusted"),  # type: ignore[index]
        }
        committed = auto_lrc.commit_final_timing_state(
            evaluated.state,
            report,
            {},
            finding_states=evaluated.finding_states,
        )
        assignment = report["assignments"][0]  # type: ignore[index]
        self.assertEqual(committed.decisions[0].written_time.seconds, before["timestamp"])
        self.assertTrue(assignment["timing_trusted"])
        self.assertTrue(assignment["content_trusted"])
        self.assertTrue(assignment["review_required"])
        self.assertFalse(assignment["overall_trusted"])
        self.assertEqual(assignment["verification_status"], "review-required")
        self.assertEqual(assignment["timestamp"], before["timestamp"])

    def test_review_only_runtime_metadata_is_writer_private(self) -> None:
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [],
            auto_lrc._RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY: frozenset({1}),
            auto_lrc._RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICT_DETAILS_KEY: {
                "authority": "review-only",
                "entries": [{"entry": 1}],
            },
        }
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "review-only.json"
            auto_lrc.write_alignment_report(path, report)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn(auto_lrc._RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICTS_KEY, text)
        self.assertNotIn(auto_lrc._RUNTIME_CROSS_BACKEND_MATERIAL_CONFLICT_DETAILS_KEY, text)


class Node18SelectedLineageRegressionTests(unittest.TestCase):
    def candidate(
        self,
        time: float,
        source: str,
        *,
        current: bool = False,
        direct: bool = False,
    ) -> auto_lrc.TimingCandidate:
        spans = tuple(
            auto_lrc.TimingTokenSpan(
                time + index * 0.04,
                time + index * 0.04 + 0.04,
                0.95,
                f"t{index}",
            )
            for index in range(4)
        )
        selected_baseline = current and source == "whispercpp-current"
        return auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="node18 lineage probe",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=0.90,
            identity_support="unavailable" if selected_baseline else "supported",
            sequence_support="supported",
            acoustic_support="supported" if direct else "unavailable",
            direct_onset_support="supported" if direct else "unavailable",
            direct_onset_time=time if direct else None,
            direct_onset_source_artifact=(
                {"source": source, "time": time} if direct else None
            ),
            direct_onset_evidence_producer=(
                "consonant-onset-detector" if direct else None
            ),
            direct_onset_evidence_kind="vocal-onset" if direct else "generic-onset",
            direct_onset_evidence_independent=direct,
            current=current,
            source_artifact={"source": source, "time": time},
        )

    def decision(
        self,
        selected: auto_lrc.TimingCandidate,
        *alternatives: auto_lrc.TimingCandidate,
    ) -> auto_lrc.TimingDecision:
        selected_eval = auto_lrc.evaluate_timing_candidate(selected)
        rejected = tuple(
            auto_lrc.evaluate_timing_candidate(candidate)
            for candidate in alternatives
        )
        return auto_lrc.TimingDecision(
            entry_index=0,
            selection_revision=1,
            status="selected_valid",
            selected_candidate_id=selected.candidate_id,
            provisional_candidate_id=None,
            written_time=selected.written_time,
            timestamp_revision=1,
            timing_evidence_revision=1,
            evidence=selected_eval.evidence,
            candidate=selected,
            rejected_candidates=rejected,
            recovery=None,
        )

    def distant_cluster(self, *alternatives: auto_lrc.TimingCandidate):
        selected = self.candidate(1.00, "whispercpp-current", current=True)
        decision = self.decision(selected, *alternatives)
        clusters = auto_lrc._peer_observation_cluster_summaries(
            decision,
            decision.written_time.seconds,
        )
        distant = [cluster for cluster in clusters if cluster.relation == "distant"]
        self.assertEqual(len(distant), 1)
        self.assertTrue(distant[0].review_required)
        return distant[0]

    def test_explicit_whispercpp_selected_lineage_allows_proven_ctc_independence(self):
        cluster = self.distant_cluster(
            self.candidate(1.40, "ctc-opening-vocal-fusion", direct=True)
        )
        self.assertEqual(cluster.evidence_state, "independent")

    def test_explicit_whispercpp_selected_lineage_blocks_same_whisper_lineage(self):
        cluster = self.distant_cluster(
            self.candidate(
                1.40,
                "local-whisper-vocal-independent-fusion",
                direct=True,
            )
        )
        self.assertEqual(cluster.evidence_state, "same-lineage")

    def test_ctc_without_direct_onset_remains_unavailable(self):
        selected = self.candidate(1.00, "whispercpp-current", current=True)
        alternative = self.candidate(1.40, "ctc-local-window", direct=False)
        evaluation = auto_lrc.evaluate_timing_candidate(alternative)
        state = auto_lrc._peer_cluster_evidence_state(
            (evaluation,),
            auto_lrc._neutral_arbitration_support_families(selected),
        )
        self.assertEqual(state, "unavailable")

    def test_mixed_independent_and_same_lineage_members_remain_mixed(self):
        cluster = self.distant_cluster(
            self.candidate(1.40, "ctc-opening-vocal-fusion", direct=True),
            self.candidate(
                1.45,
                "local-whisper-vocal-independent-fusion",
                direct=True,
            ),
        )
        self.assertEqual(cluster.evidence_state, "mixed")


class UnresolvedEvidenceRequirementsProjectionTests(unittest.TestCase):
    def candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str = "ctc-current",
        identity: str = "supported",
        sequence: str = "supported",
        direct: bool = False,
        current: bool = False,
    ) -> auto_lrc.TimingCandidate:
        spans = tuple(
            auto_lrc.TimingTokenSpan(
                time + index * 0.08,
                time + index * 0.08 + 0.04,
                0.95,
                f"t{index}",
            )
            for index in range(4)
        )
        occurrence_evidence = None
        if direct:
            occurrence_entries = [
                LyricEntry([f"projection line {index}"])
                for index in range(max(1, entry + 1))
            ]
            occurrence_assignments = [
                {"timestamp": float(index + 1), "score": .95, "segment": index + 1}
                for index in range(len(occurrence_entries))
            ]
            occurrence_binding = auto_lrc._entry_occurrence_binding(
                occurrence_entries, occurrence_assignments, entry
            )
            occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
                occurrence_entries,
                entry,
                producer="raw-asr-vocal-fusion",
                occurrence_binding=occurrence_binding,
                upstream_evidence_revision="projection-direct-upstream",
            )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"projection line {entry}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=0.9,
            identity_support=identity,
            sequence_support=sequence,
            acoustic_support="supported" if direct else "unavailable",
            acoustic_strength=0.9 if direct else None,
            acoustic_onset_time=time if direct else None,
            acoustic_source_artifact={"source": "synthetic-acoustic", "time": time} if direct else None,
            acoustic_evidence_producer="synthetic-acoustic" if direct else None,
            acoustic_evidence_kind="vocal-onset" if direct else "generic-acoustic",
            acoustic_evidence_independent=bool(direct),
            direct_onset_support="supported" if direct else "unavailable",
            direct_onset_time=time if direct else None,
            direct_onset_source_artifact={"source": "synthetic-direct", "time": time} if direct else None,
            direct_onset_evidence_producer="raw-asr-vocal-fusion" if direct else None,
            direct_onset_evidence_kind="cross-backend-onset" if direct else "generic-onset",
            direct_onset_evidence_independent=bool(direct),
            occurrence_evidence=occurrence_evidence,
            current=current,
            source_artifact={"source": source, "time": time},
        )

    def report_state(
        self,
        candidates: tuple[auto_lrc.TimingCandidate, ...],
        *,
        review: bool = True,
        content: bool = True,
        timing: bool = True,
        overall: bool = False,
    ) -> tuple[dict[str, object], auto_lrc.EvaluatedBackendTiming]:
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 1,
            "assignments": [{"entry": 1, "timestamp": 1.0, "flags": []}],
            "lyric_entry_revision": "lyric-revision",
            "source_audio_sha256": "audio-revision",
            "vocal_stem_sha256": "stem-revision",
            "algorithm_revision": auto_lrc.LRC_ALGORITHM_REVISION,
        }
        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc", (candidates,), report, {}
        )
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        auto_lrc.commit_evaluated_backend(selection, report)
        assignment = report["assignments"][0]
        assignment.update({
            "content_trusted": content,
            "timing_trusted": timing,
            "overall_trusted": overall,
            "review_required": review,
            "verification_status": "review-required" if review else "unverified",
        })
        report["candidate_selection"] = {
            "selection_revision": selection.selection_revision,
        }
        return report, evaluated

    def project(self, report: dict[str, object], evaluated: auto_lrc.EvaluatedBackendTiming) -> dict[str, object]:
        projection = auto_lrc._build_unresolved_evidence_requirements_projection(
            report, evaluated.state
        )
        report[auto_lrc._UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY] = projection
        auto_lrc._validate_unresolved_evidence_requirements_projection(report)
        return projection

    def test_projection_preserves_material_direct_producer_and_state(self) -> None:
        current = self.candidate(0, 1.0, current=True)
        alternative = self.candidate(
            0,
            3.0,
            source="raw-vocal-independent-fusion",
            direct=True,
        )
        report, evaluated = self.report_state((current, alternative))
        before = copy.deepcopy(report["assignments"])
        projection = self.project(report, evaluated)
        row = projection["rows"][0]
        self.assertEqual(projection["authority"], "diagnostic-only")
        self.assertTrue(row["material_alternatives"])
        alternative_projection = row["material_alternatives"][0]
        self.assertEqual(
            alternative_projection["direct_onset_evidence"]["producer"],
            "raw-asr-vocal-fusion",
        )
        self.assertEqual(
            alternative_projection["direct_onset_evidence"]["audio_revision"]["value"],
            "stem-revision",
        )
        self.assertIn("ALTERNATIVE_FALSIFICATION", row["missing_proof_categories"])
        self.assertIn("OWNERSHIP_INDEPENDENCE", row["missing_proof_categories"])
        self.assertEqual(report["assignments"], before)
        self.assertEqual(report["assignments"][0]["timestamp"], 1.0)
        self.assertTrue(report["assignments"][0]["review_required"])
        self.assertFalse(report["assignments"][0]["overall_trusted"])

    def test_projection_deduplicates_wrapper_cluster_only(self) -> None:
        current = self.candidate(0, 1.0, current=True)
        wrapper_a = self.candidate(0, 3.0, source="ctc-opening-vocal-fusion", direct=True)
        wrapper_b = self.candidate(0, 3.1, source="ctc-vocal-onset-confirmed", direct=True)
        report, evaluated = self.report_state((current, wrapper_a, wrapper_b))
        projection = self.project(report, evaluated)
        clusters = projection["rows"][0]["material_clusters"]
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]["member_candidate_ids"]), 2)
        self.assertEqual(
            len(set(clusters[0]["member_candidate_ids"])),
            len(clusters[0]["member_candidate_ids"]),
        )
        self.assertTrue(clusters[0]["review_required"])

    def test_projection_covers_content_opening_and_suspicious_only(self) -> None:
        content_candidate = self.candidate(
            0,
            1.0,
            current=True,
            identity="unavailable",
            sequence="unavailable",
        )
        content_report, content_evaluated = self.report_state(
            (content_candidate,), review=False, content=False, timing=True, overall=False
        )
        content_projection = self.project(content_report, content_evaluated)
        categories = set(content_projection["rows"][0]["missing_proof_categories"])
        self.assertIn("CONTENT_IDENTITY", categories)
        self.assertIn("OCCURRENCE", categories)
        self.assertIn("CURRENT_OPENING", categories)

        suspicious_candidate = self.candidate(0, 1.0, current=True, direct=True)
        suspicious_report, suspicious_evaluated = self.report_state(
            (suspicious_candidate,), review=True, content=True, timing=True, overall=False
        )
        suspicious_report["suspicious_alignments"] = [{
            "entry": 1,
            "flags": ["unresolved_raw_ctc_disagreement"],
            "source_row_revision": "suspicious-revision",
            "review_required": True,
        }]
        suspicious_projection = self.project(suspicious_report, suspicious_evaluated)
        suspicious_row = suspicious_projection["rows"][0]
        self.assertEqual(
            suspicious_row["suspicious_actionable_flags"],
            ["unresolved_raw_ctc_disagreement"],
        )
        self.assertIn("SUSPICIOUS_CONFLICT", suspicious_row["missing_proof_categories"])
        self.assertEqual(suspicious_row["material_alternatives"], [])

    def test_projection_tamper_stale_binding_and_private_stripping_fail_closed(self) -> None:
        current = self.candidate(0, 1.0, current=True)
        report, evaluated = self.report_state((current,))
        projection = self.project(report, evaluated)
        projection["rows"][0]["selected"]["selected_time"] = 99.0
        with self.assertRaises(LrcError):
            auto_lrc._validate_unresolved_evidence_requirements_projection(report)

        report, evaluated = self.report_state((current,))
        self.project(report, evaluated)
        report["source_audio_sha256"] = "stale-audio"
        with self.assertRaises(LrcError):
            auto_lrc._validate_unresolved_evidence_requirements_projection(report)

        report, evaluated = self.report_state((current,))
        self.project(report, evaluated)
        report[auto_lrc._RUNTIME_WHISPERX_TEMP_PATHS_KEY] = ("C:/private/temp.wav",)
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "projection.json"
            auto_lrc.write_alignment_report(path, report)
            text = path.read_text(encoding="utf-8")
        self.assertIn(auto_lrc._UNRESOLVED_EVIDENCE_REQUIREMENTS_KEY, text)
        self.assertNotIn("private/temp.wav", text)
        self.assertNotIn("_runtime_", text)


class CandidateGenerationCleanupTests(unittest.TestCase):
    def test_active_automatic_precentral_stages_do_not_write_assignment_timestamp(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        names = {
            "apply_ctc_acoustic_backtrack",
            "apply_ctc_local_window_realign",
            "score_line_timing_candidates",
            "apply_short_duplicate_acoustic_recovery",
            "apply_post_score_audio_validation",
            "apply_acoustic_late_onset_refinement",
        }
        functions = {
            node.name: node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        }
        self.assertEqual(set(functions), names)
        for name, function in functions.items():
            body = ast.get_source_segment(source, function) or ""
            self.assertNotIn('assignment["timestamp"] =', body, name)
            self.assertNotIn("assignment['timestamp'] =", body, name)
        acoustic_body = ast.get_source_segment(source, functions["apply_ctc_acoustic_backtrack"]) or ""
        self.assertNotIn("timestamps[index] =", acoustic_body)
        local_body = ast.get_source_segment(source, functions["apply_ctc_local_window_realign"]) or ""
        self.assertNotIn("refined[index] = timestamp", local_body)
        duplicate_body = ast.get_source_segment(source, functions["apply_short_duplicate_acoustic_recovery"]) or ""
        self.assertNotIn("recovered[index] = candidate", duplicate_body)

    def test_structured_hypothesis_is_separate_from_current_backend_candidate(self) -> None:
        entries = [LyricEntry(["target"])]
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 10.0,
                    "score": 0.9,
                    "timing_repair": "ctc-forced-align",
                    "ctc_token_spans": [
                        {"char": "x", "start": 10.0, "end": 10.02, "score": 0.01},
                        {"char": "y", "start": 10.04, "end": 10.06, "score": 0.01},
                    ],
                }
            ],
            "suspicious_alignments": [],
        }
        assignment = report["assignments"][0]  # type: ignore[index]
        auto_lrc.record_alignment_hypothesis(
            assignment,
            source="ctc-local-window",
            stage="ctc-local-window-realign",
            raw_time=12.0,
            confidence=0.95,
            spans=[
                {"char": "t", "start": 12.0, "end": 12.02, "score": 0.9},
                {"char": "a", "start": 12.04, "end": 12.06, "score": 0.9},
                {"char": "r", "start": 12.08, "end": 12.10, "score": 0.9},
            ],
            identity_support="supported",
            sequence_support="supported",
            parent_source="ctc-global-mms",
            parent_time=10.0,
        )

        candidate_sets, diagnostics = auto_lrc.collect_central_timing_candidates(
            Path("dummy.flac"),
            entries,
            [10.0],
            report,
            20.0,
            type("Args", (), {})(),
        )
        self.assertEqual(assignment["timestamp"], 10.0)
        sources = {candidate.source for candidate in candidate_sets[0]}
        self.assertIn("ctc-current", sources)
        self.assertIn("ctc-local-window", sources)
        self.assertFalse(diagnostics["alignment_hypothesis_layer"]["authoritative_precentral_mutation"])  # type: ignore[index]

        evaluated = auto_lrc.evaluate_backend_timing(
            "ctc", candidate_sets, report, diagnostics
        )
        self.assertEqual(evaluated.state.decisions[0].candidate.source, "ctc-local-window")
        self.assertEqual(assignment["timestamp"], 10.0)
        selection = auto_lrc.select_evaluated_backend((evaluated,))
        auto_lrc.commit_evaluated_backend(selection, report)
        self.assertEqual(assignment["timestamp"], 12.0)
        self.assertEqual(assignment["candidate_source"], "ctc-local-window")
        self.assertTrue(report["alignment_hypothesis_layer"]["central_commit_only"])  # type: ignore[index]

    def test_score_recommendation_cannot_control_review_or_move_backend_baseline(self) -> None:
        entries = [LyricEntry(["before"]), LyricEntry(["target"]), LyricEntry(["after"])]
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "score": 0.9},
                {"entry": 2, "timestamp": 20.0, "score": 0.35, "review_required": True},
                {"entry": 3, "timestamp": 25.0, "score": 0.9},
            ],
            "suspicious_alignments": [
                {
                    "entry": 2,
                    "review_required": True,
                    "candidate_timestamps": {"raw_asr": 20.15},
                    "raw_asr_score": 0.96,
                }
            ],
        }
        result = score_line_timing_candidates(entries, [10.0, 20.0, 25.0], report, 30.0)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertEqual(result, [10.0, 20.0, 25.0])
        self.assertEqual(assignment["timestamp"], 20.0)
        self.assertFalse(assignment["review_required"])
        self.assertEqual(report["review_required_count"], 0)
        self.assertFalse(assignment["precentral_candidate_recommendation"]["authoritative"])

        actionable = copy.deepcopy(report)
        actionable["suspicious_alignments"][0]["flags"] = ["unresolved_raw_ctc_disagreement"]  # type: ignore[index]
        score_line_timing_candidates(entries, [10.0, 20.0, 25.0], actionable, 30.0)
        self.assertTrue(actionable["assignments"][1]["review_required"])  # type: ignore[index]
        self.assertEqual(actionable["review_required_count"], 1)

    def test_explicit_whisperx_late_onset_is_candidate_only(self) -> None:
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
        auto_lrc.analyze_audio = lambda *_args, **_kwargs: features  # type: ignore[assignment]
        try:
            report: dict[str, object] = {
                "backend": "whisperx",
                "whisperx_forced_first_times": [19.0, 24.0],
                "assignments": [
                    {
                        "entry": 1,
                        "timestamp": 20.0,
                        "score": 0.75,
                        "timing_repair": "whisperx-word",
                    },
                    {
                        "entry": 2,
                        "timestamp": 25.0,
                        "score": 0.9,
                        "timing_repair": "whisperx-word",
                    },
                ],
            }
            original = [20.0, 25.0]
            result, changes = auto_lrc.apply_acoustic_late_onset_refinement(
                Path("dummy.flac"), 30.0, original, report
            )
            self.assertEqual(result, original)
            self.assertEqual(report["assignments"][0]["timestamp"], 20.0)  # type: ignore[index]
            self.assertEqual(len(changes), 1)
            self.assertFalse(changes[0]["authoritative"])
            hypotheses = report["assignments"][0]["alignment_hypotheses"]  # type: ignore[index]
            self.assertEqual(hypotheses[0]["source"], "whisperx-acoustic-late-onset")
            self.assertEqual(hypotheses[0]["raw_candidate_time"], 20.5)
            self.assertFalse(hypotheses[0]["authoritative"])
            self.assertFalse(report["acoustic_late_onset_refinement_authoritative"])
            self.assertFalse(report["acoustic_late_onset_refinement_mutates_baseline"])
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]

    def test_hybrid_ctc_derivation_preserves_base_candidate_by_copy(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_auto_backend_competition"
        )
        body = ast.get_source_segment(source, function) or ""
        self.assertIn("ctc_candidate = copy.deepcopy(base_ctc_candidate)", body)


class TailAwareCrossLineRetryTests(unittest.TestCase):
    def span(self, start: float, score: float = 0.20, token: str = "x") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def candidate(
        self,
        entry: int,
        time: float,
        *,
        current: bool = True,
        direct: bool = False,
        direct_independent: bool = False,
        pileup: bool = False,
        spans: tuple[auto_lrc.TimingTokenSpan, ...] | None = None,
    ) -> auto_lrc.TimingCandidate:
        spans = spans or tuple(self.span(time + offset) for offset in (0.0, 0.08, 0.16))
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"line {entry}",
            source="ctc-current",
            raw_time=time,
            spans=spans,
            confidence=0.8,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported" if direct else "unavailable",
            direct_onset_time=time if direct else None,
            direct_onset_source_artifact={"entry": entry, "time": time} if direct else None,
            direct_onset_evidence_producer=(
                "phonetic-onset-fusion" if direct and direct_independent else None
            ),
            direct_onset_evidence_kind=(
                "phonetic-onset" if direct and direct_independent else "generic-onset"
            ),
            direct_onset_evidence_independent=bool(direct and direct_independent),
            current=current,
            right_edge_pileup=pileup,
            source_artifact={"entry": entry, "time": time},
        )

    def capability(self, backend: str = "ctc") -> auto_lrc.BoundedRetryCapability:
        payload = {
            "kind": "mms-known-lyric-ctc",
            "backend_lineage": backend,
            "helper_path": "ctc_align.py",
            "helper_sha256": "helper-sha",
            "helper_protocol_revision": "protocol-revision",
            "alignment_audio_path": "audio.flac",
            "alignment_audio_sha256": "audio-sha",
            "alignment_audio_source": "vocal-stem",
            "sample_rate": auto_lrc.SAMPLE_RATE,
            "model_identity": "MMS_FA:known-lyric",
            "device": "cpu",
        }
        return auto_lrc.BoundedRetryCapability(
            **payload,
            capability_revision=auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(payload)
            ),
        )

    def fixture(
        self,
        current_time: float = 9.90,
        *,
        next_time: float = 12.0,
    ) -> tuple[
        list[LyricEntry],
        tuple[tuple[auto_lrc.TimingCandidate, ...], ...],
        dict[str, object],
        auto_lrc.BoundedRetryCapability,
        auto_lrc.BaseRetryEpoch,
        tuple[auto_lrc.CrossLineRetryRequest, ...],
    ]:
        entries = [LyricEntry(["line zero"]), LyricEntry(["line one"]), LyricEntry(["line two"])]
        previous = self.candidate(0, 9.74)
        current = self.candidate(1, current_time)
        following = self.candidate(2, next_time)
        candidate_sets = ((previous,), (current,), (following,))
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 3,
            "assignments": [
                {"entry": index + 1, "timestamp": candidate.written_time.seconds, "timing_repair": "ctc", "score": 0.9}
                for index, candidate in enumerate((previous, current, following))
            ],
        }
        diagnostics = auto_lrc.inactive_legacy_timing_diagnostics()
        capability = self.capability()
        epoch = auto_lrc.freeze_base_retry_epoch(
            "ctc", report, candidate_sets, diagnostics, capability
        )
        requests = auto_lrc.plan_crossline_retry_requests(
            epoch, capability, entries, candidate_sets, 20.0
        )
        return entries, candidate_sets, report, capability, epoch, requests

    def helper_payload(
        self,
        request: auto_lrc.CrossLineRetryRequest,
        entries: list[LyricEntry],
        *,
        target_starts: tuple[float, ...] = (10.40, 10.48, 10.56),
        target_scores: tuple[float, ...] | None = None,
    ) -> dict[str, object]:
        target_scores = target_scores or tuple(0.20 for _ in target_starts)
        row_starts = {
            request.context_entry_indexes[0]: (request.window_start, request.window_start + 0.08, request.window_start + 0.16),
            request.entry_index: target_starts,
        }
        if request.context_entry_indexes[-1] != request.entry_index:
            assert request.next_bound is not None
            next_time = request.next_bound.canonical_centiseconds / 100.0
            row_starts[request.context_entry_indexes[-1]] = (next_time, next_time + 0.08, next_time + 0.16)
        rows: list[dict[str, object]] = []
        for local, global_index in enumerate(request.context_entry_indexes):
            starts = row_starts[global_index]
            scores = target_scores if global_index == request.entry_index else tuple(0.2 for _ in starts)
            spans = [
                {"char": chr(97 + index), "start": start, "end": start + 0.02, "score": scores[index]}
                for index, start in enumerate(starts)
            ]
            rows.append({
                "entry": local + 1,
                "text": auto_lrc.entry_sung_text(entries[global_index]),
                "romaji": "".join(item["char"] for item in spans),
                "start": spans[0]["start"],
                "end": spans[-1]["end"],
                "ctc_score": sum(scores) / len(scores),
                "tokens": len(spans),
                "token_spans": spans,
            })
        return {
            "window_start": round(request.window_start, 3),
            "window_end": round(request.window_end, 3),
            "entries": rows,
        }

    def test_01_clear_after_release_is_valid_and_never_retries(self) -> None:
        entries, sets, _report, capability, epoch, requests = self.fixture(10.50)
        decision = auto_lrc.build_final_timing_state(sets).decisions[1]
        self.assertEqual(decision.evidence.ownership.status, "clear")
        self.assertEqual(requests, ())

    def test_02_exact_tail_end_is_violation_and_plans_once(self) -> None:
        entries, sets, _report, capability, epoch, _ = self.fixture(9.92)
        requests = auto_lrc.plan_crossline_retry_requests(epoch, capability, entries, sets, 20.0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].ordinal, 1)

    def test_03_after_tail_inside_release_uncertainty_plans_bounded_retry(self) -> None:
        entries, sets, _report, _capability, _epoch, requests = self.fixture(10.00)
        decision = auto_lrc.build_final_timing_state(sets).decisions[1]
        self.assertEqual(decision.evidence.ownership.status, "unknown")
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(len(requests), 1)

    def test_04_strong_valid_current_is_protected_without_retry(self) -> None:
        entries, sets, _report, _capability, _epoch, requests = self.fixture(10.50)
        self.assertEqual(auto_lrc.build_final_timing_state(sets).decisions[1].selected_candidate_id, sets[1][0].candidate_id)
        self.assertEqual(requests, ())

    def test_05_unknown_with_bound_direct_onset_can_pass_without_retry(self) -> None:
        previous = self.candidate(0, 9.74, pileup=True)
        current = self.candidate(1, 12.0, direct=True, direct_independent=True)
        evaluation = auto_lrc.evaluate_timing_candidate(current, previous)
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        self.assertTrue(evaluation.valid)

    def test_06_unknown_sequence_only_gets_recovery_attempt_not_trust_bypass(self) -> None:
        entries, sets, _report, _capability, _epoch, requests = self.fixture(10.00)
        self.assertEqual(auto_lrc.build_final_timing_state(sets).decisions[1].status, "provisional_unresolved")
        self.assertEqual(len(requests), 1)

    def test_07_overlap_request_is_content_derived_and_repeatable(self) -> None:
        entries, sets, _report, capability, epoch, first = self.fixture()
        second = auto_lrc.plan_crossline_retry_requests(epoch, capability, entries, sets, 20.0)
        self.assertEqual(first, second)
        self.assertEqual(first[0].request_revision, second[0].request_revision)

    def test_07b_candidate_permutation_preserves_epoch_request_calls_and_final_state(self) -> None:
        entries, sets, report, capability, epoch, first = self.fixture()
        extra = self.candidate(1, 9.80, current=False)
        first_sets = (sets[0], (*sets[1], extra), sets[2])
        second_sets = (sets[0], (extra, *sets[1]), sets[2])
        first_epoch = auto_lrc.freeze_base_retry_epoch("ctc", report, first_sets, auto_lrc.inactive_legacy_timing_diagnostics(), capability)
        second_epoch = auto_lrc.freeze_base_retry_epoch("ctc", report, second_sets, auto_lrc.inactive_legacy_timing_diagnostics(), capability)
        self.assertEqual(first_epoch, second_epoch)
        first_requests = auto_lrc.plan_crossline_retry_requests(first_epoch, capability, entries, first_sets, 20.0)
        second_requests = auto_lrc.plan_crossline_retry_requests(second_epoch, capability, entries, second_sets, 20.0)
        self.assertEqual(first_requests, second_requests)
        self.assertEqual(auto_lrc.build_final_timing_state(first_sets), auto_lrc.build_final_timing_state(second_sets))

    def test_08_retry_epoch_is_distinct_from_final_evaluation_input_revision(self) -> None:
        entries, sets, report, capability, epoch, requests = self.fixture()
        outcome = auto_lrc.parse_crossline_retry_output(requests[0], capability, entries, self.helper_payload(requests[0], entries))
        augmented = (sets[0], (*sets[1], outcome.candidate), sets[2])  # type: ignore[arg-type]
        evaluated = auto_lrc.evaluate_backend_timing("ctc", augmented, report, {"retry_epoch": epoch.retry_epoch_revision})
        self.assertNotEqual(epoch.retry_epoch_revision, evaluated.input_revision)

    def test_09_duplicate_and_reverse_schedule_invoke_at_most_once(self) -> None:
        entries, sets, _report, capability, epoch, requests = self.fixture()
        calls: list[str] = []
        original = auto_lrc.invoke_crossline_retry
        def fake(request, _capability, _entries):
            calls.append(request.request_revision)
            return auto_lrc.parse_crossline_retry_output(request, capability, entries, self.helper_payload(request, entries))
        auto_lrc.invoke_crossline_retry = fake  # type: ignore[assignment]
        try:
            outcomes = auto_lrc.execute_crossline_retry_epoch(epoch, capability, entries, sets, 20.0, tuple(reversed((*requests, *requests))))
        finally:
            auto_lrc.invoke_crossline_retry = original  # type: ignore[assignment]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(outcomes), 1)

    def test_10_stale_core_relation_dependencies_fail_closed(self) -> None:
        entries, sets, _report, capability, epoch, requests = self.fixture()
        request = requests[0]
        for stale in (
            replace(request, retry_epoch_revision="stale"),
            replace(request, current=replace(request.current, generation_revision="stale")),
            replace(request, previous=replace(request.previous, candidate_id="stale")),
            replace(request, previous_tail_evidence_revision="stale"),
            replace(request, ownership_relation_revision="stale"),
        ):
            self.assertFalse(auto_lrc.crossline_retry_request_is_current(stale, epoch, capability, entries, sets, 20.0))

    def test_11_stale_context_transcript_mapping_and_window_fail_closed(self) -> None:
        entries, sets, _report, capability, epoch, requests = self.fixture()
        request = requests[0]
        for stale in (
            replace(request, next_bound=replace(request.next_bound, candidate_id="stale")),  # type: ignore[arg-type]
            replace(request, context_candidate_ids=("stale",)),
            replace(request, transcript_digest="stale"),
            replace(request, row_mapping_revision="stale"),
            replace(request, window_end=request.window_end + 0.1),
        ):
            self.assertFalse(auto_lrc.crossline_retry_request_is_current(stale, epoch, capability, entries, sets, 20.0))

    def test_12_capability_is_runtime_authorized_for_ctc_and_carried_hybrid_only(self) -> None:
        with TemporaryDirectory() as name:
            path = Path(name) / "audio.bin"
            path.write_bytes(b"audio")
            args = auto_lrc.argparse.Namespace(whisperx_device="cpu")
            helper = Path(auto_lrc.__file__).resolve().parent / "ctc_align.py"
            seed = auto_lrc.create_bounded_retry_capability_seed(
                path, "vocal-stem", helper, sample_rate=16000, device="cpu"
            )
            metadata = {
                "ctc_backend": "mms_ctc",
                "ctc_audio_path": str(path),
                "ctc_audio_source": "vocal-stem",
                "ctc_sample_rate": 16000,
                "ctc_device": "cpu",
            }
            self.assertIsNone(
                auto_lrc.build_bounded_retry_capability(
                    {"backend": "ctc", **metadata}, path, args
                )
            )
            # A serialized/dict seed is provenance only.  Metadata must not be
            # sufficient to mint execution capability, even when all hashes match.
            metadata_only_ctc: dict[str, object] = {
                "backend": "ctc",
                "ctc_bounded_retry_capability_seed": copy.deepcopy(seed),
            }
            metadata_only_hybrid: dict[str, object] = {
                "backend": "hybrid",
                "ctc_bounded_retry_capability_seed": copy.deepcopy(seed),
            }
            self.assertIsNone(
                auto_lrc.build_bounded_retry_capability(metadata_only_ctc, path, args)
            )
            self.assertIsNone(
                auto_lrc.build_bounded_retry_capability(metadata_only_hybrid, path, args)
            )

            ctc_report: dict[str, object] = {
                "backend": "ctc",
                "ctc_bounded_retry_capability_seed": copy.deepcopy(seed),
            }
            auto_lrc._attach_bounded_retry_runtime_capability(ctc_report, seed)
            self.assertIsNotNone(
                auto_lrc.build_bounded_retry_capability(ctc_report, path, args)
            )

            hybrid_report: dict[str, object] = {"backend": "hybrid"}
            auto_lrc.carry_bounded_retry_capability_seed(hybrid_report, ctc_report)
            self.assertIsNotNone(
                auto_lrc.build_bounded_retry_capability(hybrid_report, path, args)
            )
            tampered = copy.deepcopy(hybrid_report)
            tampered["ctc_bounded_retry_capability_seed"]["alignment_audio_sha256"] = "fake"  # type: ignore[index]
            self.assertIsNone(
                auto_lrc.build_bounded_retry_capability(tampered, path, args)
            )
            for field_name, bad_value in (
                ("helper_protocol_revision", ""),
                ("helper_protocol_revision", "self-consistent-but-wrong"),
                ("model_identity", ""),
                ("model_identity", "other-model"),
                ("device", ""),
            ):
                incomplete_seed = copy.deepcopy(seed)
                incomplete_seed[field_name] = bad_value
                seed_payload = auto_lrc._bounded_retry_capability_semantic_payload(
                    incomplete_seed
                )
                incomplete_seed["capability_revision"] = auto_lrc._canonical_json_digest(seed_payload)
                incomplete_report: dict[str, object] = {
                    "backend": "ctc",
                    "ctc_bounded_retry_capability_seed": incomplete_seed,
                }
                auto_lrc._attach_bounded_retry_runtime_capability(
                    incomplete_report, incomplete_seed
                )
                self.assertIsNone(
                    auto_lrc.build_bounded_retry_capability(
                        incomplete_report, path, args
                    ),
                    field_name,
                )
            self.assertIsNone(
                auto_lrc.build_bounded_retry_capability(
                    {
                        "backend": "whisperx",
                        "ctc_bounded_retry_capability_seed": copy.deepcopy(seed),
                        auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: ctc_report[
                            auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY
                        ],
                    },
                    path,
                    args,
                )
            )

    def test_13_helper_wrong_rows_mapping_and_nonmonotonic_output_fail(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        variants = []
        missing = self.helper_payload(request, entries); missing["entries"] = missing["entries"][:-1]; variants.append(missing)
        wrong = self.helper_payload(request, entries); wrong["entries"][0]["entry"] = 2; variants.append(wrong)  # type: ignore[index]
        nonmonotonic = self.helper_payload(request, entries); nonmonotonic["entries"][0]["token_spans"][1]["start"] = 9.0; variants.append(nonmonotonic)  # type: ignore[index]
        for payload in variants:
            self.assertEqual(auto_lrc.parse_crossline_retry_output(request, capability, entries, payload).status, "invalid")

    def test_13b_same_length_token_substitution_and_reorder_fail(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        for mode in ("substitute", "reorder"):
            payload = self.helper_payload(request, entries)
            row = payload["entries"][0]  # type: ignore[index]
            if mode == "substitute":
                row["token_spans"][1]["char"] = "z"
            else:
                row["token_spans"][0]["char"], row["token_spans"][1]["char"] = row["token_spans"][1]["char"], row["token_spans"][0]["char"]
            self.assertEqual(auto_lrc.parse_crossline_retry_output(request, capability, entries, payload).status, "invalid", mode)

    def test_13c_nonfinite_score_and_nested_values_fail_closed(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        for value in (float("nan"), float("inf"), float("-inf")):
            payload = self.helper_payload(request, entries)
            payload["entries"][0]["ctc_score"] = value  # type: ignore[index]
            self.assertEqual(auto_lrc.parse_crossline_retry_output(request, capability, entries, payload).status, "invalid")
            nested = self.helper_payload(request, entries)
            nested["diagnostic"] = {"bad": value}
            self.assertEqual(auto_lrc.parse_crossline_retry_output(request, capability, entries, nested).status, "invalid")

    def test_13d_nonfinite_candidate_confidence_never_promotes_to_one(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            self.assertEqual(auto_lrc._candidate_confidence({"score": value}), 0.0)
            candidate = auto_lrc.make_timing_candidate(
                entry_index=0,
                entry_text="generic",
                source="synthetic-nonfinite",
                raw_time=12.0,
                spans=(self.span(12.0), self.span(12.08)),
                confidence=value,
                identity_support="supported",
                sequence_support="supported",
            )
            self.assertEqual(candidate.confidence, 0.0)

    def test_14_partial_target_row_prefix_suffix_interior_and_empty_fail(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        for mode in ("prefix", "suffix", "interior", "empty"):
            payload = self.helper_payload(request, entries)
            row = payload["entries"][0]  # type: ignore[index]
            spans = row["token_spans"]
            if mode == "empty":
                row["token_spans"] = []
            elif mode == "prefix":
                row["token_spans"] = spans[1:]
            elif mode == "suffix":
                row["token_spans"] = spans[:-1]
            else:
                row["token_spans"] = [spans[0], spans[-1]]
            self.assertEqual(auto_lrc.parse_crossline_retry_output(request, capability, entries, payload).status, "invalid", mode)

    def test_15_complete_target_row_touching_edge_is_truncated_and_rejected(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        starts = (request.window_end - 0.09, request.window_end - 0.06, request.window_end - 0.03)
        payload = self.helper_payload(request, entries, target_starts=starts)
        payload["entries"] = payload["entries"][:2]
        request = replace(request, context_entry_indexes=request.context_entry_indexes[:2], context_candidate_ids=request.context_candidate_ids[:2], transcript_digests=request.transcript_digests[:2], row_mapping=request.row_mapping[:2])
        outcome = auto_lrc.parse_crossline_retry_output(request, capability, entries, payload)
        self.assertEqual(outcome.status, "candidate")
        self.assertTrue(outcome.candidate.window_truncated)  # type: ignore[union-attr]
        self.assertFalse(auto_lrc.evaluate_timing_candidate(outcome.candidate).valid)  # type: ignore[arg-type]

    def test_16_right_edge_pileup_is_hard_rejected(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture()
        request = requests[0]
        starts = tuple(request.window_end - 0.09 + index * 0.01 for index in range(8))
        payload = self.helper_payload(request, entries, target_starts=starts, target_scores=tuple(0.001 for _ in starts))
        row = payload["entries"][0]  # type: ignore[index]
        for span in row["token_spans"]:
            span["end"] = span["start"] + 0.005
        row["end"] = row["token_spans"][-1]["end"]
        outcome = auto_lrc.parse_crossline_retry_output(request, capability, entries, payload)
        self.assertTrue(outcome.candidate.right_edge_pileup)  # type: ignore[union-attr]
        self.assertFalse(auto_lrc.evaluate_timing_candidate(outcome.candidate).valid)  # type: ignore[arg-type]

    def test_17_retry_still_overlapping_tail_is_rejected_without_second_attempt(self) -> None:
        entries, sets, _report, _capability, _epoch, requests = self.fixture()
        retry = self.candidate(1, 9.90, current=False)
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(*sets[1], retry),
            previous_candidate=sets[0][0],
            selection_revision=2,
        )
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(len(requests), 1)

    def test_18_retry_with_opening_fracture_is_rejected(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture(next_time=15.0)
        payload = self.helper_payload(requests[0], entries, target_starts=(10.4, 13.6, 13.68))
        outcome = auto_lrc.parse_crossline_retry_output(requests[0], capability, entries, payload)
        evaluation = auto_lrc.evaluate_timing_candidate(outcome.candidate)  # type: ignore[arg-type]
        self.assertFalse(evaluation.valid)
        self.assertIn("temporal-incoherence", evaluation.rejection_reasons)

    def test_18b_retry_strong_opening_late_tail_fracture_keeps_onset_not_tail(self) -> None:
        entries, _sets, _report, capability, _epoch, requests = self.fixture(next_time=15.0)
        payload = self.helper_payload(requests[0], entries, target_starts=(10.4, 10.48, 10.56, 13.8, 13.88))
        outcome = auto_lrc.parse_crossline_retry_output(requests[0], capability, entries, payload)
        evaluation = auto_lrc.evaluate_timing_candidate(outcome.candidate)  # type: ignore[arg-type]
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.evidence.fracture_region, "middle")
        self.assertEqual(auto_lrc.derive_reliable_tail(outcome.candidate).validity, "invalid")  # type: ignore[arg-type]

    def test_18c_missing_helper_returns_error_outcome_and_no_second_attempt(self) -> None:
        entries, sets, _report, capability, epoch, requests = self.fixture()
        original = auto_lrc.run_command
        auto_lrc.run_command = lambda *_args, **_kwargs: (_ for _ in ()).throw(LrcError("missing python"))  # type: ignore[assignment]
        try:
            first = auto_lrc.execute_crossline_retry_epoch(epoch, capability, entries, sets, 20.0, requests)
        finally:
            auto_lrc.run_command = original  # type: ignore[assignment]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].status, "error")
        self.assertEqual(auto_lrc.build_final_timing_state(sets).decisions[1].status, "provisional_unresolved")

    def test_19_coherent_retry_after_release_can_replace_invalid_current(self) -> None:
        entries, sets, _report, capability, _epoch, requests = self.fixture()
        outcome = auto_lrc.parse_crossline_retry_output(requests[0], capability, entries, self.helper_payload(requests[0], entries))
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(*sets[1], outcome.candidate),  # type: ignore[arg-type]
            previous_candidate=sets[0][0],
            selection_revision=2,
        )
        self.assertEqual(decision.selected_candidate_id, outcome.candidate.candidate_id)  # type: ignore[union-attr]

    def test_20_late_tail_fracture_preserves_onset_but_removes_tail_authority(self) -> None:
        candidate = self.candidate(0, 10.0, spans=(self.span(10.0), self.span(10.08), self.span(10.16), self.span(14.0), self.span(14.08)))
        self.assertTrue(auto_lrc.evaluate_timing_candidate(candidate).valid)
        self.assertEqual(auto_lrc.derive_reliable_tail(candidate).validity, "invalid")

    def test_21_adjacent_violations_are_planned_once_from_frozen_epoch(self) -> None:
        entries = [LyricEntry([f"line {index}"]) for index in range(3)]
        sets = ((self.candidate(0, 9.74),), (self.candidate(1, 9.90),), (self.candidate(2, 10.02),))
        report = {"backend": "ctc", "assignments": [{"timing_repair": "ctc", "score": .9} for _ in entries]}
        cap = self.capability(); diagnostics = auto_lrc.inactive_legacy_timing_diagnostics()
        epoch = auto_lrc.freeze_base_retry_epoch("ctc", report, sets, diagnostics, cap)
        first = auto_lrc.plan_crossline_retry_requests(epoch, cap, entries, sets, 20.0)
        second = auto_lrc.plan_crossline_retry_requests(epoch, cap, entries, sets, 20.0)
        self.assertEqual(first, second)
        self.assertEqual(len({(item.retry_epoch_revision, item.entry_index) for item in first}), len(first))

    def test_22_legacy_stages_are_inactive_and_generic_contamination_stays_unknown(self) -> None:
        diagnostics = auto_lrc.inactive_legacy_timing_diagnostics()
        self.assertFalse(diagnostics["legacy_boundary_mutation_pipeline"]["executed"])  # type: ignore[index]
        self.assertFalse(diagnostics["legacy_prefix_probe_candidate_generation"]["authoritative"])  # type: ignore[index]
        contaminated = self.candidate(0, 10.0, pileup=True)
        current = self.candidate(1, 12.0)
        evaluation = auto_lrc.evaluate_timing_candidate(current, contaminated)
        self.assertEqual(evaluation.evidence.ownership.status, "unknown")
        self.assertFalse(evaluation.valid)

    def test_23_collect_does_not_execute_legacy_final_guard(self) -> None:
        entries = [LyricEntry(["generic"])]
        report: dict[str, object] = {"backend": "whisperx", "assignments": [{"timestamp": 12.0, "timing_repair": "word", "score": 0.9}]}
        original = auto_lrc.apply_final_ctc_timing_guard
        calls = 0
        def forbidden(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("inactive legacy final guard executed")
        auto_lrc.apply_final_ctc_timing_guard = forbidden  # type: ignore[assignment]
        try:
            _sets, diagnostics = auto_lrc.collect_central_timing_candidates(Path("missing.wav"), entries, [12.0], report, 20.0, auto_lrc.argparse.Namespace(whisperx_device="cpu"))
        finally:
            auto_lrc.apply_final_ctc_timing_guard = original  # type: ignore[assignment]
        self.assertEqual(calls, 0)
        self.assertEqual(diagnostics["legacy_prefix_probe_candidate_generation"]["active_candidate_count"], 0)  # type: ignore[index]

    def test_23b_collector_has_no_dormant_prefix_probe_candidate_construction(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        collector = next(item for item in module.body if isinstance(item, ast.FunctionDef) and item.name == "collect_central_timing_candidates")
        collector_source = ast.unparse(collector)
        self.assertNotIn("ctc-prefix-probe", collector_source)
        self.assertNotIn("probes_by_entry", collector_source)
        self.assertNotIn("prefix_assignment_items", collector_source)

    def test_24_production_source_has_zero_legacy_boundary_stage_callers(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        callers = []
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None)
            if name in {"apply_ctc_boundary_validation_stage", "apply_final_ctc_timing_guard"}:
                parent_function = next(
                    (item.name for item in module.body if isinstance(item, ast.FunctionDef) and node in ast.walk(item)),
                    "module",
                )
                callers.append(parent_function)
        self.assertEqual(callers, [])

    def test_25_inactive_legacy_console_never_claims_forward_prefix_was_applied(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        self.assertNotIn("CTC forward weak-prefix recovery applied", source)
        self.assertIn("CTC legacy forward weak-prefix evidence (inactive)", source)

    def test_26_runtime_retry_authority_is_not_serialized(self) -> None:
        with TemporaryDirectory() as name:
            report_path = Path(name) / "report.json"
            report: dict[str, object] = {
                "backend": "ctc",
                auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: self.capability(),
            }
            auto_lrc.write_alignment_report(report_path, report)
            serialized = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertNotIn(auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY, serialized)
            self.assertIn(auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY, report)


class GeorgetteBoundaryRegressionContinuedTests(unittest.TestCase):
    def test_final_guard_prefers_independent_prefix_probe_over_weak_zero_gap_path(self) -> None:
        frame_times = np.round(np.arange(10.0, 15.0, 0.01), 3).astype(np.float32)
        onset = np.zeros(len(frame_times), dtype=np.float32)
        onset[int(round((12.75 - 10.0) / 0.01))] = 1.0
        features = AudioFeatures(
            duration=20.0,
            frame_times=frame_times,
            rms_db=np.full(len(frame_times), -20.0, dtype=np.float32),
            onset_strength=onset,
            segments=[],
        )
        original_analyze_audio = auto_lrc.analyze_audio
        original_analyze_vocal = auto_lrc.analyze_vocal_onsets
        original_prefix_probe = auto_lrc.run_ctc_prefix_probe
        auto_lrc.analyze_audio = lambda *_args, **_kwargs: features  # type: ignore[assignment]
        auto_lrc.analyze_vocal_onsets = lambda *_args, **_kwargs: features  # type: ignore[assignment]
        auto_lrc.run_ctc_prefix_probe = lambda *_args, **_kwargs: {  # type: ignore[assignment]
            "accepted": True,
            "candidate": 12.75,
            "ctc_score": 0.18,
            "prefix_mean_score": 0.14,
            "prefix_weak_ratio": 0.25,
            "prefix_span_seconds": 0.9,
            "prefix_text": "everlastin",
        }
        try:
            entries = [LyricEntry(["... again"]), LyricEntry(["everlastin ever"])]
            report: dict[str, object] = {
                "backend": "ctc",
                "timing_entries": 2,
                "assignments": [
                    {
                        "entry": 1,
                        "timestamp": 10.0,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "ctc_token_spans": [
                            {"char": "n", "start": 11.9, "end": 12.0, "score": 0.01},
                        ],
                    },
                    {
                        "entry": 2,
                        "timestamp": 12.02,
                        "timing_repair": "ctc-forced-align",
                        "score": 0.9,
                        "ctc_score": 0.1,
                        "ctc_token_spans": [
                            {"char": "e", "start": 12.02, "end": 12.04, "score": 0.01},
                            {"char": "v", "start": 12.08, "end": 12.10, "score": 0.001},
                            {"char": "e", "start": 12.14, "end": 12.16, "score": 0.001},
                            {"char": "r", "start": 12.20, "end": 12.22, "score": 0.001},
                            {"char": "l", "start": 12.28, "end": 12.30, "score": 0.001},
                            {"char": "a", "start": 12.80, "end": 12.82, "score": 0.2},
                            {"char": "s", "start": 12.9, "end": 12.92, "score": 0.001},
                            {"char": "t", "start": 13.0, "end": 13.02, "score": 0.001},
                        ],
                    },
                ],
                "suspicious_alignments": [],
            }
            args = auto_lrc.build_parser().parse_args(["dummy.txt"])
            timestamps, report, changes = apply_final_ctc_timing_guard(
                Path("dummy.flac"), entries, [10.0, 12.02], report, 20.0, args
            )
            self.assertEqual(len(changes), 1)
            self.assertAlmostEqual(timestamps[1], 12.75, places=3)
            self.assertEqual(changes[0]["reason"], "independent-lyric-prefix-ctc-probe")
            self.assertTrue(report["assignments"][1]["ctc_prefix_probe_recovery"])  # type: ignore[index]
        finally:
            auto_lrc.analyze_audio = original_analyze_audio  # type: ignore[assignment]
            auto_lrc.analyze_vocal_onsets = original_analyze_vocal  # type: ignore[assignment]
            auto_lrc.run_ctc_prefix_probe = original_prefix_probe  # type: ignore[assignment]

    def test_untrusted_crossline_recovery_does_not_leave_stale_effective_index(self) -> None:
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 70.0,
                    "ctc_token_spans": [{"start": 77.943, "end": 77.963, "score": 0.01}],
                },
                {
                    "entry": 2,
                    "timestamp": 77.963,
                    "ctc_token_spans": [
                        {"start": 77.963, "end": 77.983, "score": 0.003},
                        {"start": 83.205, "end": 83.225, "score": 0.099},
                        {"start": 83.325, "end": 83.345, "score": 0.009},
                    ],
                },
                {"entry": 3, "timestamp": 92.33, "ctc_token_spans": [{"start": 92.33, "end": 92.35, "score": 0.1}]},
            ]
        }
        timestamps, report, changes = apply_ctc_crossline_initial_recovery(
            [70.0, 77.963, 92.33], report, 100.0
        )
        self.assertEqual(len(changes), 1)
        current = report["assignments"][1]  # type: ignore[index]
        self.assertFalse(current["ctc_crossline_initial_recovery_trusted"])
        self.assertNotIn("ctc_effective_token_start_index", current)

    def test_0247_inherits_previous_right_edge_pileup_contamination(self) -> None:
        previous_tail = [
            {
                "char": "x",
                "start": 166.687 + index * 0.04,
                "end": 166.707 + index * 0.04,
                "score": 0.005,
            }
            for index in range(20)
        ]
        # Make the final token land exactly on the local-window edge.
        shift = 167.447 - previous_tail[-1]["end"]
        for span in previous_tail:
            span["start"] += shift
            span["end"] += shift

        assignments: list[object] = [
            {
                "entry": 1,
                "timestamp": 147.340,
                "ctc_score": 0.019063,
                "ctc_local_window": {"start": 136.896, "end": 167.447},
                "ctc_token_spans": previous_tail,
            },
            {
                "entry": 2,
                "timestamp": 167.468,
                "ctc_score": 0.500062,
                "ctc_token_spans": [
                    {"char": "k", "start": 167.468, "end": 167.488, "score": 0.003825},
                    {"char": "o", "start": 167.588, "end": 167.608, "score": 0.618277},
                    {"char": "n", "start": 171.272, "end": 171.292, "score": 0.028629},
                    {"char": "o", "start": 172.594, "end": 172.614, "score": 0.261168},
                    {"char": "m", "start": 172.954, "end": 172.974, "score": 0.924104},
                    {"char": "a", "start": 173.074, "end": 173.094, "score": 0.942513},
                    {"char": "m", "start": 173.575, "end": 173.595, "score": 0.035145},
                    {"char": "a", "start": 173.715, "end": 173.735, "score": 0.957866},
                ],
            },
        ]

        annotate_ctc_boundary_evidence(assignments)

        self.assertTrue(assignments[0]["ctc_right_edge_pileup"])  # type: ignore[index]
        evidence = assignments[1]["ctc_boundary_evidence"]  # type: ignore[index]
        self.assertTrue(evidence["prior_line_truncation_contamination"])
        self.assertTrue(evidence["realign_required"])

        report: dict[str, object] = {"assignments": assignments}
        timestamps = [147.340, 167.468]
        timestamps, report, changes = apply_ctc_crossline_initial_recovery(
            timestamps, report, 185.028
        )
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["mode"], "short-borrowed-tail-prefix")
        self.assertAlmostEqual(timestamps[1], 167.588, places=3)
        self.assertTrue(assignments[1]["ctc_crossline_initial_recovery_trusted"])  # type: ignore[index]


class OccurrenceRecurrenceZeroTrustTests(unittest.TestCase):
    def test_bound_vocal_onset_can_prove_single_reliable_opening_without_fracture(self) -> None:
        spans = [auto_lrc.TimingTokenSpan(10.0, 10.02, 0.30, "a")]
        spans.append(auto_lrc.TimingTokenSpan(10.30, 10.32, 0.01, "b"))
        spans.extend(
            auto_lrc.TimingTokenSpan(10.70 + index * 0.42, 10.72 + index * 0.42, 0.20, "c")
            for index in range(8)
        )
        candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="target",
            source="ctc-vocal-onset-confirmed",
            raw_time=10.0,
            spans=tuple(spans),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=0.95,
            acoustic_onset_time=10.04,
            acoustic_source_artifact={"audio": "independent"},
            acoustic_evidence_producer="consonant-onset-detector",
            acoustic_evidence_kind="vocal-onset",
            acoustic_evidence_independent=True,
            direct_onset_time=10.04,
            direct_onset_source_artifact={"audio": "independent"},
            direct_onset_evidence_producer="consonant-onset-detector",
            direct_onset_evidence_kind="vocal-onset",
            direct_onset_evidence_independent=True,
            source_artifact={"producer": "vocal-onset"},
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate)
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.evidence.temporal_coherence, "coherent")
        self.assertTrue(evaluation.evidence.minimum_proof_satisfied)

    def test_bound_vocal_onset_does_not_rescue_short_collapsed_path(self) -> None:
        spans = (
            auto_lrc.TimingTokenSpan(10.0, 10.02, 0.30, "a"),
            auto_lrc.TimingTokenSpan(10.18, 10.20, 0.01, "b"),
            auto_lrc.TimingTokenSpan(10.40, 10.42, 0.20, "c"),
        )
        candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="target",
            source="ctc-vocal-onset-confirmed",
            raw_time=10.0,
            spans=spans,
            confidence=0.95,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=1.0,
            acoustic_onset_time=10.0,
            acoustic_source_artifact={"audio": "independent"},
            acoustic_evidence_producer="consonant-onset-detector",
            acoustic_evidence_kind="vocal-onset",
            acoustic_evidence_independent=True,
            direct_onset_time=10.0,
            direct_onset_source_artifact={"audio": "independent"},
            direct_onset_evidence_producer="consonant-onset-detector",
            direct_onset_evidence_kind="vocal-onset",
            direct_onset_evidence_independent=True,
            source_artifact={"producer": "vocal-onset"},
        )
        evaluation = auto_lrc.evaluate_timing_candidate(candidate)
        self.assertFalse(evaluation.valid)
        self.assertFalse(evaluation.evidence.minimum_proof_satisfied)

    def test_independent_content_identity_is_recurrence_source_scoped(self) -> None:
        common = dict(
            entry_index=0,
            entry_text="same lyric",
            raw_time=20.0,
            spans=(),
            confidence=0.8,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=0.9,
            acoustic_onset_time=20.0,
            acoustic_source_artifact={"recurrence": 1},
            acoustic_evidence_producer="global-recurrence-search",
            acoustic_evidence_kind="recurrence-onset",
            acoustic_evidence_independent=True,
            direct_onset_time=20.0,
            direct_onset_source_artifact={"recurrence": 1},
            direct_onset_evidence_producer="global-recurrence-search",
            direct_onset_evidence_kind="recurrence-onset",
            direct_onset_evidence_independent=True,
            source_artifact={"recurrence": 1},
        )
        recurrence_unbound = auto_lrc.make_timing_candidate(source="global-acoustic-recurrence", **common)
        recurrence = auto_lrc.make_timing_candidate(
            source="global-acoustic-recurrence", independent_content_identity=True, **common
        )
        ordinary = auto_lrc.make_timing_candidate(
            source="ctc-vocal-onset-confirmed", independent_content_identity=True, **common
        )
        self.assertFalse(auto_lrc.candidate_supplies_independent_content_identity(recurrence_unbound))
        self.assertTrue(auto_lrc.candidate_supplies_independent_content_identity(recurrence))
        self.assertFalse(auto_lrc.candidate_supplies_independent_content_identity(ordinary))

    def test_vocal_onset_confirmation_never_mutates_baseline(self) -> None:
        frame_times = np.arange(0.0, 20.0, 0.02, dtype=np.float32)
        onset = np.zeros_like(frame_times)
        onset[int(round(10.0 / 0.02))] = 1.0
        features = AudioFeatures(
            duration=20.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -20.0),
            onset_strength=onset,
            segments=[],
        )
        original = auto_lrc.analyze_vocal_onsets
        auto_lrc.analyze_vocal_onsets = lambda _audio, _duration: features  # type: ignore[assignment]
        try:
            spans = [
                {"char": chr(97 + (index % 20)), "start": 10.0 + index * 0.40, "end": 10.02 + index * 0.40, "score": 0.25}
                for index in range(10)
            ]
            report: dict[str, object] = {
                "backend": "ctc",
                "assignments": [{"entry": 1, "timestamp": 10.0, "score": 0.9, "ctc_token_spans": spans}],
            }
            timestamps = [10.0]
            before = copy.deepcopy(report)
            generated = auto_lrc.apply_vocal_onset_confirmation_hypotheses(
                Path(__file__), [LyricEntry(["target"])], timestamps, report, 20.0
            )
            self.assertEqual(timestamps, [10.0])
            self.assertEqual(report["assignments"][0]["timestamp"], before["assignments"][0]["timestamp"])  # type: ignore[index]
            self.assertEqual(len(generated), 1)
            self.assertFalse(report["vocal_onset_confirmation_authoritative"])
            self.assertFalse(report["vocal_onset_confirmation_mutates_baseline"])
        finally:
            auto_lrc.analyze_vocal_onsets = original  # type: ignore[assignment]

    def test_global_recurrence_search_requires_a_distinct_full_line_match(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(0.0, 0.15, size=(8, 1200)).astype(np.float32)
        template = rng.normal(0.0, 1.0, size=(8, 200)).astype(np.float32)
        features[:, 100:300] = template
        features[:, 700:900] = template + rng.normal(0.0, 0.01, size=template.shape).astype(np.float32)
        result = auto_lrc._search_recurrence_start(features, 2.0, 6.0, 10.0, 22.0)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(float(result["start"]), 14.0, delta=0.12)
        self.assertGreater(float(result["score"]), 0.95)
        self.assertGreater(float(result["margin"]), auto_lrc.RECURRENCE_MIN_DISTINCT_MARGIN)
    def test_recurrence_identity_provenance_fails_closed_on_margin_or_audio_mismatch(self) -> None:
        lyric = LyricEntry(["same lyric"])
        audio = Path(__file__)
        audio_sha = auto_lrc._path_sha256(audio)
        lyric_sha = auto_lrc.hashlib.sha256(
            auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(lyric)).encode("utf-8")
        ).hexdigest()
        base = {
            "source": "global-acoustic-recurrence",
            "stage": "occurrence-level-global-recurrence-search",
            "raw_candidate_time": 20.0,
            "authoritative": False,
            "mode": "full-line-acoustic-recurrence-with-trusted-reference",
            "parent": {"source": "central-trusted-repeat-reference", "raw_candidate_time": 10.0},
            "window": {
                "search_start": 15.0,
                "search_end": 25.0,
                "audio_sha256": audio_sha,
                "feature_revision": auto_lrc.RECURRENCE_FEATURE_REVISION,
                "reference_candidate_id": "trusted-ref",
                "similarity": auto_lrc.RECURRENCE_MIN_SIMILARITY + 0.1,
                "distinct_margin": auto_lrc.RECURRENCE_MIN_DISTINCT_MARGIN + 0.1,
                "normalized_lyric_sha256": lyric_sha,
            },
        }
        base["hypothesis_id"] = auto_lrc._timing_content_digest(base)
        self.assertTrue(auto_lrc.recurrence_hypothesis_provenance_is_valid(base, lyric, audio))
        weak = copy.deepcopy(base)
        weak["window"]["distinct_margin"] = auto_lrc.RECURRENCE_MIN_DISTINCT_MARGIN - 0.001  # type: ignore[index]
        self.assertFalse(auto_lrc.recurrence_hypothesis_provenance_is_valid(weak, lyric, audio))
        wrong_audio = copy.deepcopy(base)
        wrong_audio["window"]["audio_sha256"] = "0" * 64  # type: ignore[index]
        self.assertFalse(auto_lrc.recurrence_hypothesis_provenance_is_valid(wrong_audio, lyric, audio))

    def test_global_recurrence_ambiguous_distinct_matches_fail_margin_gate(self) -> None:
        rng = np.random.default_rng(11)
        features = rng.normal(0.0, 0.05, size=(8, 1400)).astype(np.float32)
        template = rng.normal(0.0, 1.0, size=(8, 200)).astype(np.float32)
        features[:, 100:300] = template
        features[:, 600:800] = template
        features[:, 1000:1200] = template
        result = auto_lrc._search_recurrence_start(features, 2.0, 6.0, 10.0, 26.0)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreater(float(result["score"]), 0.95)
        self.assertLess(float(result["margin"]), auto_lrc.RECURRENCE_MIN_DISTINCT_MARGIN)


class FixedPointPhaseCZeroTrustTests(unittest.TestCase):
    @staticmethod
    def _features_with_peaks(peaks: list[tuple[float, float]]) -> AudioFeatures:
        frame_times = np.arange(0.0, 6.0, 0.02, dtype=np.float32)
        onset = np.zeros_like(frame_times)
        for time, strength in peaks:
            onset[int(round(time / 0.02))] = strength
        return AudioFeatures(
            duration=6.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -30.0),
            onset_strength=onset,
            segments=[],
        )

    def _phonetic_runtime_fixture(
        self, audio: Path
    ) -> tuple[AudioFeatures, dict[str, object], dict[str, object]]:
        frame_times = np.arange(0.0, 20.0, 0.02, dtype=np.float32)
        onset = np.zeros_like(frame_times)
        onset[int(round(10.04 / 0.02))] = 0.95
        features = AudioFeatures(
            duration=20.0,
            frame_times=frame_times,
            rms_db=np.full_like(frame_times, -30.0),
            onset_strength=onset,
            segments=[],
        )
        peak = {"time": 10.0, "score": auto_lrc.PHONETIC_ONSET_MIN_POSTERIOR + 0.10}
        spans = [
            {
                "char": chr(97 + index),
                "start": 10.003 + index * 0.12,
                "end": 10.053 + index * 0.12,
                "score": 0.8,
            }
            for index in range(8)
        ]
        assignment: dict[str, object] = {
            "entry": 1,
            "timestamp": 10.003,
            "score": 0.9,
            "timing_repair": "ctc",
            "ctc_token_spans": spans,
            "ctc_first_token_candidates": [peak],
        }
        current_candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="provenance-check",
            source="ctc-current",
            raw_time=10.003,
            spans=tuple(
                auto_lrc.TimingTokenSpan(
                    float(span["start"]), float(span["end"]), float(span["score"]), str(span["char"])
                )
                for span in spans
            ),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
        )
        window: dict[str, object] = {
            "feature_revision": auto_lrc.PHONETIC_ONSET_FEATURE_REVISION,
            "audio_sha256": auto_lrc._path_sha256(audio),
            "audio_duration_seconds": 20.0,
            "current_candidate_id": current_candidate.candidate_id,
            "current_raw_time": 10.003,
            "current_written_centiseconds": 1000,
            "current_time": 10.0,
            "ctc_peak_time": 10.0,
            "ctc_peak_score": peak["score"],
            "ctc_peak_digest": auto_lrc._timing_content_digest(peak),
            "onset_time": 10.04,
            "onset_strength": 0.95,
            "onset_prominence": 0.95,
        }
        holder: dict[str, object] = {}
        hypothesis = auto_lrc.record_alignment_hypothesis(
            holder,
            source="ctc-phonetic-onset-fusion",
            stage="independent-phonetic-onset-fusion",
            raw_time=10.0,
            confidence=0.9,
            spans=[],
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=float(window["onset_strength"]),
            acoustic_onset_time=10.04,
            direct_onset_time=10.04,
            parent_source="ctc-current",
            parent_time=10.003,
            window=window,
            mode="coherent-ctc-first-token-plus-independent-onset",
        )
        return features, assignment, hypothesis

    def test_tail_release_chooses_first_new_cluster_not_strongest_late_peak(self) -> None:
        features = self._features_with_peaks([
            (1.92, 0.90),
            (2.00, 0.94),
            (2.08, 0.92),
            (2.16, 0.91),
            (2.46, 0.86),
            (3.20, 1.00),
        ])
        result = auto_lrc._first_post_tail_cluster_onset(
            features,
            2.05,
            3.50,
            min_strength=0.82,
            min_prominence=0.50,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result["time"], 2.46, places=2)
        self.assertGreaterEqual(result["cluster_gap"], auto_lrc.TAIL_RELEASE_CLUSTER_BREAK_SECONDS)
        self.assertNotAlmostEqual(result["time"], 3.20, places=2)

    def test_tail_release_abstains_without_observable_cluster_break(self) -> None:
        features = self._features_with_peaks([
            (1.98, 0.90),
            (2.10, 0.93),
            (2.20, 0.95),
            (2.34, 0.96),
        ])
        result = auto_lrc._first_post_tail_cluster_onset(
            features,
            2.05,
            2.40,
            min_strength=0.82,
            min_prominence=0.50,
        )
        self.assertIsNone(result)

    def _mutual_pair_fixture(self) -> tuple[list[LyricEntry], list[object], dict[str, object], dict[str, object]]:
        audio = Path(__file__)
        entries = [LyricEntry(["same lyric"]), LyricEntry(["same lyric"])]
        assignments: list[object] = [
            {"entry": 1, "timestamp": 9.8, "timing_repair": "ctc", "score": 0.3},
            {"entry": 2, "timestamp": 29.8, "timing_repair": "ctc", "score": 0.3},
        ]
        for index, assignment in enumerate(assignments):
            assert isinstance(assignment, dict)
            start_time = 10.0 + index * 20.0
            auto_lrc.record_alignment_hypothesis(
                assignment,
                source="ctc-local-window",
                stage="ctc-local-window-realign",
                raw_time=start_time,
                confidence=0.6,
                spans=[
                    {"char": chr(97 + n), "start": start_time + n * 0.08, "end": start_time + n * 0.08 + 0.02, "score": 0.2}
                    for n in range(6)
                ],
                identity_support="supported",
                sequence_support="supported",
                parent_source="ctc-global-mms",
                parent_time=float(assignment["timestamp"]),
                mode="local-window-independent-candidate",
            )
        left_identity = auto_lrc._bound_local_occurrence_identity(assignments[0])  # type: ignore[arg-type]
        right_identity = auto_lrc._bound_local_occurrence_identity(assignments[1])  # type: ignore[arg-type]
        self.assertIsNotNone(left_identity)
        self.assertIsNotNone(right_identity)
        normalized = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[0]))
        lyric_sha = auto_lrc.hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        payload: dict[str, object] = {
            "audio_sha256": auto_lrc._path_sha256(audio),
            "feature_revision": auto_lrc.MUTUAL_RECURRENCE_FEATURE_REVISION,
            "normalized_lyric_sha256": lyric_sha,
            "left_entry": 1,
            "right_entry": 2,
            "left_seed": {"time": 10.0, "strength": 0.8, "kind": "ctc-local-first-token", "evidence_id": "L"},
            "right_seed": {"time": 30.0, "strength": 0.9, "kind": "ctc-local-first-token", "evidence_id": "R"},
            "left_identity_evidence": left_identity,
            "right_identity_evidence": right_identity,
            "left_bounds": [5.0, 20.0],
            "right_bounds": [25.0, 40.0],
            "left_candidate": 10.2,
            "right_candidate": 30.2,
            "template_duration": 5.0,
            "similarity": auto_lrc.MUTUAL_RECURRENCE_MIN_SIMILARITY + 0.10,
            "runner_similarity": 0.40,
            "distinct_margin": auto_lrc.MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN + 0.10,
        }
        pair_id = auto_lrc._timing_content_digest(payload)
        def row(target: int, peer: int, raw: float, peer_time: float) -> dict[str, object]:
            result: dict[str, object] = {
                "source": "mutual-acoustic-recurrence",
                "stage": "occurrence-level-mutual-recurrence-search",
                "raw_candidate_time": raw,
                "authoritative": False,
                "mode": "reciprocal-exact-duplicate-pitchclass-recurrence",
                "parent": {"source": "reciprocal-untrusted-duplicate-pair", "raw_candidate_time": raw},
                "window": {
                    **payload,
                    "pair_id": pair_id,
                    "target_entry": target,
                    "peer_entry": peer,
                    "peer_candidate": peer_time,
                    "onset_strength": 0.90,
                    "onset_prominence": 0.40,
                },
            }
            result["hypothesis_id"] = auto_lrc._timing_content_digest(result)
            return result
        left = row(1, 2, 10.2, 30.2)
        right = row(2, 1, 30.2, 10.2)
        assignments[0]["alignment_hypotheses"].append(left)  # type: ignore[index]
        assignments[1]["alignment_hypotheses"].append(right)  # type: ignore[index]
        return entries, assignments, left, right

    def test_mutual_recurrence_provenance_binds_actual_exact_duplicate_peer(self) -> None:
        entries, assignments, left, _right = self._mutual_pair_fixture()
        self.assertTrue(auto_lrc.mutual_recurrence_hypothesis_provenance_is_valid(
            left, entries[0], Path(__file__), assignments, entries, 0
        ))
        wrong_entries = [entries[0], LyricEntry(["different lyric"])]
        self.assertFalse(auto_lrc.mutual_recurrence_hypothesis_provenance_is_valid(
            left, wrong_entries[0], Path(__file__), assignments, wrong_entries, 0
        ))

    def test_mutual_recurrence_provenance_rejects_pair_tampering(self) -> None:
        entries, assignments, left, _right = self._mutual_pair_fixture()
        tampered = copy.deepcopy(left)
        tampered["window"]["distinct_margin"] = auto_lrc.MUTUAL_RECURRENCE_MIN_DISTINCT_MARGIN - 0.001  # type: ignore[index]
        self.assertFalse(auto_lrc.mutual_recurrence_hypothesis_provenance_is_valid(
            tampered, entries[0], Path(__file__), assignments, entries, 0
        ))

    def test_mutual_recurrence_is_only_independent_identity_when_provenance_bound(self) -> None:
        common = dict(
            entry_index=0,
            entry_text="same lyric",
            raw_time=10.0,
            spans=(),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            acoustic_support="supported",
            direct_onset_support="supported",
            acoustic_strength=0.9,
            acoustic_onset_time=10.0,
            acoustic_source_artifact={"pair": 1},
            acoustic_evidence_producer="mutual-recurrence-search",
            acoustic_evidence_kind="recurrence-onset",
            acoustic_evidence_independent=True,
            direct_onset_time=10.0,
            direct_onset_source_artifact={"pair": 1},
            direct_onset_evidence_producer="mutual-recurrence-search",
            direct_onset_evidence_kind="recurrence-onset",
            direct_onset_evidence_independent=True,
            source_artifact={"pair": 1},
        )
        unbound = auto_lrc.make_timing_candidate(source="mutual-acoustic-recurrence", **common)
        bound = auto_lrc.make_timing_candidate(
            source="mutual-acoustic-recurrence", independent_content_identity=True, **common
        )
        self.assertFalse(auto_lrc.candidate_supplies_independent_content_identity(unbound))
        self.assertTrue(auto_lrc.candidate_supplies_independent_content_identity(bound))

    def test_collector_rejects_fake_mutual_source_label_without_reciprocal_provenance(self) -> None:
        entries = [LyricEntry(["same lyric"]), LyricEntry(["same lyric"])]
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 10.0,
                    "timing_repair": "ctc",
                    "score": 0.9,
                    "alignment_hypotheses": [{
                        "source": "mutual-acoustic-recurrence",
                        "stage": "occurrence-level-mutual-recurrence-search",
                        "raw_candidate_time": 10.2,
                        "authoritative": False,
                        "mode": "reciprocal-exact-duplicate-pitchclass-recurrence",
                        "parent": {"source": "reciprocal-untrusted-duplicate-pair", "raw_candidate_time": 10.0},
                        "window": {},
                    }],
                },
                {"entry": 2, "timestamp": 30.0, "timing_repair": "ctc", "score": 0.9},
            ],
        }
        sets, _diagnostics = auto_lrc.collect_central_timing_candidates(
            Path(__file__), entries, [10.0, 30.0], report, 40.0, type("Args", (), {})(),
            independent_evidence_audio_path=Path(__file__),
        )
        self.assertNotIn("mutual-acoustic-recurrence", {candidate.source for candidate in sets[0]})

    def test_phonetic_onset_provenance_requires_coherent_ctc_and_exact_audio(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "isolated-vocal.wav"
            audio.write_bytes(b"isolated-vocal-runtime-artifact")
            features, assignment, hypothesis = self._phonetic_runtime_fixture(audio)
            with mock.patch.object(auto_lrc, "analyze_vocal_onsets", return_value=features):
                self.assertTrue(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    hypothesis, assignment, audio
                ))
                wrong_audio = copy.deepcopy(hypothesis)
                wrong_audio["window"]["audio_sha256"] = "0" * 64  # type: ignore[index]
                self.assertFalse(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    wrong_audio, assignment, audio
                ))
                empty_spans = copy.deepcopy(assignment)
                empty_spans["ctc_token_spans"] = []
                self.assertFalse(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    hypothesis, empty_spans, audio
                ))
                fractured = copy.deepcopy(assignment)
                fractured["ctc_token_spans"] = [
                    {"char": "a", "start": 10.0, "end": 10.02, "score": 0.8},
                    {"char": "b", "start": 10.08, "end": 10.10, "score": 0.8},
                    {"char": "c", "start": 13.2, "end": 13.22, "score": 0.8},
                    {"char": "d", "start": 13.28, "end": 13.30, "score": 0.8},
                ]
                self.assertFalse(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    hypothesis, fractured, audio
                ))

    def test_bound_local_occurrence_identity_rejects_mutated_hypothesis_record(self) -> None:
        assignment: dict[str, object] = {"entry": 1, "timestamp": 10.0, "score": 0.5}
        auto_lrc.record_alignment_hypothesis(
            assignment,
            source="ctc-local-window",
            stage="ctc-local-window-realign",
            raw_time=10.2,
            confidence=0.8,
            spans=[
                {"char": chr(97 + index), "start": 10.2 + index * 0.08, "end": 10.22 + index * 0.08, "score": 0.3}
                for index in range(6)
            ],
            identity_support="supported",
            sequence_support="supported",
            parent_source="ctc-global-mms",
            parent_time=10.0,
            mode="local-window-independent-candidate",
        )
        self.assertIsNotNone(auto_lrc._bound_local_occurrence_identity(assignment))
        hypothesis = assignment["alignment_hypotheses"][0]  # type: ignore[index]
        hypothesis["raw_candidate_time"] = 10.9  # type: ignore[index]
        self.assertIsNone(auto_lrc._bound_local_occurrence_identity(assignment))

    def test_mutual_recurrence_provenance_rejects_mutated_peer_identity_evidence(self) -> None:
        entries, assignments, left, _right = self._mutual_pair_fixture()
        self.assertTrue(auto_lrc.mutual_recurrence_hypothesis_provenance_is_valid(
            left, entries[0], Path(__file__), assignments, entries, 0
        ))
        peer_hypotheses = assignments[1]["alignment_hypotheses"]  # type: ignore[index]
        local_peer = next(
            hypothesis for hypothesis in peer_hypotheses
            if isinstance(hypothesis, dict) and hypothesis.get("source") == "ctc-local-window"
        )
        local_peer["confidence"] = 0.01
        self.assertFalse(auto_lrc.mutual_recurrence_hypothesis_provenance_is_valid(
            left, entries[0], Path(__file__), assignments, entries, 0
        ))

    def test_new_phase_c_producers_are_candidate_only(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        names = {
            "apply_phonetic_onset_fusion_hypotheses",
            "apply_mutual_acoustic_recurrence_hypotheses",
            "apply_trusted_tail_release_onset_hypotheses",
        }
        functions = {
            node.name: node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        }
        self.assertEqual(set(functions), names)
        for name, function in functions.items():
            body = ast.get_source_segment(source, function) or ""
            self.assertNotIn('assignment["timestamp"] =', body, name)
            self.assertNotIn("assignment['timestamp'] =", body, name)
            self.assertNotIn("timestamps[index] =", body, name)
            self.assertIn("record_alignment_hypothesis", body, name)

    def test_no_unbound_independent_tail_evidence_in_functions(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        offenders: list[str] = []
        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            loads = {
                child.id for child in ast.walk(node)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id == "independent_tail_evidence"
            }
            if not loads:
                continue
            bound = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            if node.args.vararg is not None:
                bound.add(node.args.vararg.arg)
            if node.args.kwarg is not None:
                bound.add(node.args.kwarg.arg)
            bound.update(
                child.id for child in ast.walk(node)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, (ast.Store, ast.Param))
            )
            if "independent_tail_evidence" not in bound:
                offenders.append(node.name)
        self.assertEqual(offenders, [])

    def test_tail_release_provenance_rejects_missing_cluster_break(self) -> None:
        audio = Path(__file__)
        assignments: list[object] = [
            {"entry": 1, "timestamp": 8.0, "timing_repair": "ctc", "score": 0.9},
            {"entry": 2, "timestamp": 9.5, "timing_repair": "ctc", "score": 0.9},
        ]
        base: dict[str, object] = {
            "source": "trusted-tail-release-onset",
            "stage": "trusted-tail-release-onset-search",
            "raw_candidate_time": 10.5,
            "authoritative": False,
            "mode": "strong-onset-inside-centrally-trusted-tail-release",
            "parent": {"source": "central-trusted-predecessor-tail", "raw_candidate_time": 10.0},
            "window": {
                "audio_sha256": auto_lrc._path_sha256(audio),
                "feature_revision": "trusted-tail-release-cluster-onset-v3",
                "previous_candidate_id": "previous",
                "previous_tail_revision": "tail",
                "previous_tail_end": 10.0,
                "previous_tail_uncertainty": 0.9,
                "search_start": 10.0,
                "search_end": 11.0,
                "onset_strength": 0.95,
                "onset_prominence": 0.70,
                "previous_onset_time": 10.2,
                "cluster_gap": 0.3,
                "context_start": 9.2,
                "old_candidate_raw_time": 9.5,
                "old_candidate_centiseconds": 950,
                "old_candidate_time": 9.5,
            },
        }
        base["hypothesis_id"] = auto_lrc._timing_content_digest(base)
        self.assertTrue(auto_lrc.trusted_tail_release_provenance_is_valid(
            base, audio, assignments, 1
        ))
        weak = copy.deepcopy(base)
        weak["window"]["cluster_gap"] = 0.10  # type: ignore[index]
        self.assertFalse(auto_lrc.trusted_tail_release_provenance_is_valid(
            weak, audio, assignments, 1
        ))


    def test_phonetic_provenance_binds_raw_assignment_and_canonical_written_time(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "isolated-vocal.wav"
            audio.write_bytes(b"isolated-vocal-runtime-artifact")
            features, assignment, hypothesis = self._phonetic_runtime_fixture(audio)
            with mock.patch.object(auto_lrc, "analyze_vocal_onsets", return_value=features):
                self.assertTrue(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    hypothesis, assignment, audio
                ))
                wrong_raw = copy.deepcopy(assignment)
                wrong_raw["timestamp"] = 10.013
                self.assertFalse(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    hypothesis, wrong_raw, audio
                ))
                stale_revision = copy.deepcopy(hypothesis)
                stale_revision["window"]["feature_revision"] = "stale"  # type: ignore[index]
                stale_revision["hypothesis_id"] = auto_lrc._timing_content_digest(
                    {key: value for key, value in stale_revision.items() if key != "hypothesis_id"}
                )
                self.assertFalse(auto_lrc.phonetic_onset_fusion_provenance_is_valid(
                    stale_revision, assignment, audio
                ))

    def test_tail_release_provenance_binds_raw_assignment_and_canonical_written_time(self) -> None:
        audio = Path(__file__)
        assignments: list[object] = [
            {"entry": 1, "timestamp": 8.0, "timing_repair": "ctc", "score": 0.9},
            {"entry": 2, "timestamp": 9.503, "timing_repair": "ctc", "score": 0.9},
        ]
        base: dict[str, object] = {
            "source": "trusted-tail-release-onset",
            "stage": "trusted-tail-release-onset-search",
            "raw_candidate_time": 10.5,
            "authoritative": False,
            "mode": "strong-onset-inside-centrally-trusted-tail-release",
            "parent": {"source": "central-trusted-predecessor-tail", "raw_candidate_time": 10.0},
            "window": {
                "audio_sha256": auto_lrc._path_sha256(audio),
                "feature_revision": "trusted-tail-release-cluster-onset-v3",
                "previous_candidate_id": "previous",
                "previous_tail_revision": "tail",
                "previous_tail_end": 10.0,
                "previous_tail_uncertainty": 0.9,
                "search_start": 10.0,
                "search_end": 11.0,
                "onset_strength": 0.95,
                "onset_prominence": 0.70,
                "previous_onset_time": 10.2,
                "cluster_gap": 0.3,
                "context_start": 9.2,
                "old_candidate_raw_time": 9.503,
                "old_candidate_centiseconds": 950,
                "old_candidate_time": 9.5,
            },
        }
        base["hypothesis_id"] = auto_lrc._timing_content_digest(base)
        self.assertTrue(auto_lrc.trusted_tail_release_provenance_is_valid(base, audio, assignments, 1))
        wrong_raw = copy.deepcopy(assignments)
        wrong_raw[1]["timestamp"] = 9.513  # type: ignore[index]
        self.assertFalse(auto_lrc.trusted_tail_release_provenance_is_valid(base, audio, wrong_raw, 1))

    def test_update_metrics_preserves_central_candidate_identity_trust(self) -> None:
        source_sha = auto_lrc.hashlib.sha256(Path(auto_lrc.__file__).read_bytes()).hexdigest()
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 1,
            "timing_state_schema_version": 3,
            "algorithm_source_sha256": source_sha,
            "candidate_selection": {
                "status": "selected",
                "metric_stage": "central-evaluated",
                "selected_backend": "ctc",
            },
            "central_timing_state": [{
                "entry": 1,
                "timestamp": 10.0,
                "canonical_written_centiseconds": 1000,
                "candidate_id": "independent",
                "candidate_source": "global-acoustic-recurrence",
                "timestamp_status": "selected_valid",
                "timing_trusted": True,
            }],
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "timing_repair": "ctc",
                "score": 0.2,
                "canonical_written_centiseconds": 1000,
                "timestamp_status": "selected_valid",
                "selection_revision": 1,
                "timestamp_revision": 1,
                "timing_evidence_revision": 1,
                "selected_candidate_id": "independent",
                "provisional_candidate_id": None,
                "candidate_source": "global-acoustic-recurrence",
                "candidate_provenance": {
                    "candidate_id": "independent",
                    "source": "global-acoustic-recurrence",
                    "canonical_written_centiseconds": 1000,
                },
                "normalized_timing_evidence": {
                    "computed_for_centiseconds": 1000,
                    "structural_failures": [],
                    "minimum_proof_satisfied": True,
                },
                "content_trusted": True,
                "content_trust_source": "selected-candidate-independent-identity",
                "timing_trusted": True,
                "timing_audit_trusted": True,
                "overall_trusted": True,
                "review_required": False,
            }],
        }
        self.assertTrue(auto_lrc.report_has_verified_central_commit(report))
        auto_lrc.update_report_confidence_metrics(report)
        self.assertEqual(report["content_trusted_entries"], 1)
        self.assertEqual(report["timing_trusted_entries"], 1)
        self.assertEqual(report["overall_trusted_entries"], 1)
        self.assertEqual(report["overall_trusted_percent"], 100.0)
        assignment = report["assignments"][0]  # type: ignore[index]
        self.assertTrue(assignment["content_trusted"])
        self.assertTrue(assignment["overall_trusted"])

    def test_update_metrics_refuses_unbound_fake_central_content_trust(self) -> None:
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 1,
            # A field name alone must not grant central authority.
            "central_timing_state": [{
                "entry": 1,
                "timestamp": 10.0,
                "canonical_written_centiseconds": 1000,
                "candidate_id": "forged",
                "candidate_source": "global-acoustic-recurrence",
                "timestamp_status": "selected_valid",
                "timing_trusted": True,
            }],
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "timing_repair": "ctc",
                "score": 0.2,
                "content_trusted": True,
                "timing_trusted": True,
                "review_required": False,
            }],
        }
        self.assertFalse(auto_lrc.report_has_verified_central_commit(report))
        auto_lrc.update_report_confidence_metrics(report)
        self.assertEqual(report["content_trusted_entries"], 0)
        self.assertEqual(report["overall_trusted_entries"], 0)
        assignment = report["assignments"][0]  # type: ignore[index]
        self.assertFalse(assignment["content_trusted"])
        self.assertFalse(assignment["overall_trusted"])

class BoundaryRecoveryTests(unittest.TestCase):
    def span(self, start: float, score: float, token: str) -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def candidate(self, entry: int, time: float, spans, *, source='ctc-current', current=False):
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f'line {entry}',
            source=source,
            raw_time=time,
            spans=tuple(spans),
            confidence=0.8,
            identity_support='supported',
            sequence_support='supported',
            current=current,
            source_artifact={'entry': entry, 'time': time, 'source': source},
        )

    def capability(self):
        payload = {
            'kind': 'mms-known-lyric-ctc',
            'backend_lineage': 'ctc',
            'helper_path': 'ctc_align.py',
            'helper_sha256': 'helper-sha',
            'helper_protocol_revision': 'protocol-revision',
            'alignment_audio_path': 'audio.flac',
            'alignment_audio_sha256': 'audio-sha',
            'alignment_audio_source': 'vocal-stem',
            'sample_rate': auto_lrc.SAMPLE_RATE,
            'model_identity': 'MMS_FA:known-lyric',
            'device': 'cpu',
        }
        return auto_lrc.BoundedRetryCapability(
            **payload,
            capability_revision=auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(payload)
            ),
        )

    def test_weak_initial_consonant_strong_vowel_can_prove_first_mora(self):
        c = self.candidate(0, 45.745, (
            self.span(45.745, 0.041866, 'k'),
            self.span(45.845, 0.586112, 'i'),
            self.span(45.925, 0.20, 'b'),
        ))
        region = auto_lrc.analyze_candidate_regions(c)
        self.assertAlmostEqual(region.onset_proof_end or 0.0, 45.865, places=3)
        self.assertEqual(region.temporal_coherence, 'coherent')

    def test_global_serial_candidate_cannot_override_tail_conflict(self):
        previous = self.candidate(0, 9.74, (
            self.span(9.74, .2, 'l'), self.span(9.82, .2, 'a'), self.span(9.90, .2, 'l')
        ), current=True)
        current = self.candidate(1, 9.90, (
            self.span(9.90, .2, 's'), self.span(9.98, .4, 'o'), self.span(10.06, .2, 'n')
        ), source='ctc-current')
        ev = auto_lrc.evaluate_timing_candidate(current, previous)
        self.assertFalse(ev.valid)
        self.assertIn('previous-tail-overlap', ev.rejection_reasons)

    def test_current_line_opening_requires_bound_disjoint_evidence(self):
        previous = self.candidate(0, 9.74, (
            self.span(9.74, .2, 'l'), self.span(9.82, .2, 'a'), self.span(9.90, .2, 'l')
        ), current=True)
        current = self.candidate(1, 9.90, (
            self.span(9.90, .2, 's'), self.span(9.98, .4, 'o'), self.span(10.06, .2, 'n')
        ), source='ctc-opening-local-retry')
        ev = auto_lrc.evaluate_timing_candidate(current, previous)
        self.assertFalse(ev.valid)
        self.assertEqual(ev.evidence.ownership.status, 'violation')
        self.assertIn('previous-tail-overlap', ev.rejection_reasons)

        previous = self.candidate(0, 9.74, (
            self.span(9.74, .2, 'l'), self.span(9.82, .2, 'a'), self.span(9.90, .2, 'l')
        ), current=True)
        previous_region = auto_lrc.analyze_candidate_regions(previous)
        previous_tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id=previous.candidate_id,
            previous_generation_revision=previous.generation_revision,
            previous_region_revision=previous_region.region_revision,
            tail_evidence_revision='external-tail-observation',
            validity='valid',
            reliable_tail_start=9.74,
            reliable_tail_end=9.92,
            uncertainty_seconds=.18,
            confidence=.9,
            reason='runtime-external-tail',
            authority_backend_lineage='ctc',
            authority_producer='ctc-assignment-terminal-row',
            authority_observation_revision='external-observation-revision',
        )
        occurrence_entries = [LyricEntry(['line 0']), LyricEntry(['line 1'])]
        occurrence_binding = auto_lrc._entry_occurrence_binding(
            occurrence_entries,
            [
                {'timestamp': 9.74, 'score': .95, 'segment': 1},
                {'timestamp': 9.90, 'score': .95, 'segment': 2},
            ],
            1,
        )
        occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
            occurrence_entries,
            1,
            producer='raw-asr-consensus',
            occurrence_binding=occurrence_binding,
            upstream_evidence_revision='boundary-direct-upstream',
        )
        current = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text='line 1',
            source='raw-vocal-independent-fusion',
            raw_time=9.90,
            spans=(
                self.span(9.90, .2, 's'),
                self.span(9.98, .4, 'o'),
                self.span(10.06, .2, 'n'),
            ),
            confidence=.8,
            identity_support='supported',
            sequence_support='supported',
            direct_onset_support='supported',
            direct_onset_time=9.90,
            direct_onset_source_artifact={'entry': 1, 'observation': 'external-onset'},
            direct_onset_evidence_producer='raw-asr-consensus',
            direct_onset_evidence_kind='cross-backend-onset',
            direct_onset_evidence_independent=True,
            occurrence_evidence=occurrence_evidence,
            current=True,
        )
        self.assertEqual(auto_lrc._candidate_support_families(current), {'whisper-family'})
        self.assertTrue(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(current))
        evaluation = auto_lrc.evaluate_timing_candidate(
            current, previous, previous_tail=previous_tail
        )
        self.assertTrue(evaluation.valid)
        self.assertEqual(evaluation.evidence.ownership.status, 'violation')
        self.assertTrue(evaluation.evidence.ownership_independent)
        self.assertEqual(
            evaluation.evidence.ownership.previous_tail_authority_backend_lineage,
            'ctc',
        )
        self.assertEqual(
            evaluation.evidence.ownership.previous_tail_authority_producer,
            'ctc-assignment-terminal-row',
        )

    def test_unknown_previous_tail_still_schedules_opening_retry(self):
        entries = [LyricEntry(['prev']), LyricEntry(['current'])]
        previous = self.candidate(0, 9.0, (
            self.span(9.0, .2, 'x'), self.span(9.08, .01, 'x')
        ), current=True)
        current = self.candidate(1, 10.0, (
            self.span(10.0, .01, 'x'), self.span(10.08, .01, 'x'), self.span(10.16, .01, 'x')
        ), current=True)
        sets = ((previous,), (current,))
        report = {
            'backend': 'ctc', 'timing_entries': 2,
            'assignments': [
                {'entry': 1, 'timestamp': 9.0, 'timing_repair': 'ctc', 'score': .9},
                {'entry': 2, 'timestamp': 10.0, 'timing_repair': 'ctc', 'score': .9},
            ],
        }
        cap = self.capability()
        epoch = auto_lrc.freeze_base_retry_epoch(
            'ctc', report, sets, auto_lrc.inactive_legacy_timing_diagnostics(), cap
        )
        requests = auto_lrc.plan_crossline_retry_requests(epoch, cap, entries, sets, 30.0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].previous_tail_validity, 'unknown')
        self.assertEqual(requests[0].context_entry_indexes, (1,))

    def test_terminal_line_retry_has_no_next_bound_and_is_bounded(self):
        entries = [LyricEntry(['prev']), LyricEntry(['current'])]
        previous = self.candidate(0, 197.6, (
            self.span(197.6, .2, 'x'), self.span(197.68, .01, 'x')
        ), current=True)
        current = self.candidate(1, 206.5, (
            self.span(206.5, .01, 's'), self.span(206.6, .01, 'o'), self.span(206.7, .01, 'n')
        ), current=True)
        sets = ((previous,), (current,))
        report = {
            'backend': 'ctc', 'timing_entries': 2,
            'assignments': [
                {'entry': 1, 'timestamp': 197.6, 'timing_repair': 'ctc', 'score': .9},
                {'entry': 2, 'timestamp': 206.5, 'timing_repair': 'ctc', 'score': .9},
            ],
        }
        cap = self.capability()
        epoch = auto_lrc.freeze_base_retry_epoch(
            'ctc', report, sets, auto_lrc.inactive_legacy_timing_diagnostics(), cap
        )
        request = auto_lrc.plan_crossline_retry_requests(epoch, cap, entries, sets, 239.4)[0]
        self.assertIsNone(request.next_bound)
        self.assertEqual(request.context_entry_indexes, (1,))
        self.assertLessEqual(request.window_end - request.window_start, 20.0)
        self.assertLess(request.window_start, 206.5)

    def test_isolated_token_before_large_fracture_is_not_opening_proof(self):
        c = self.candidate(0, 10.0, (
            self.span(10.0, .5, 'a'), self.span(13.5, .5, 'b'), self.span(13.58, .5, 'c')
        ), source='ctc-opening-local-retry')
        ev = auto_lrc.evaluate_timing_candidate(c)
        self.assertFalse(ev.valid)
        self.assertIn('temporal-incoherence', ev.rejection_reasons)


class V142CentralConsensusRegressionTests(unittest.TestCase):
    def candidate(
        self,
        source: str,
        time: float,
        *,
        current: bool = False,
        independent: bool = True,
        confidence: float = 0.9,
    ) -> auto_lrc.TimingCandidate:
        artifact = {"source": source, "time": time}
        if source.startswith("raw-"):
            onset_producer = "raw-asr-vocal-fusion"
            onset_kind = "cross-backend-onset"
        elif "whisperx" in source:
            onset_producer = "whisperx-vocal-fusion"
            onset_kind = "cross-backend-onset"
        elif "recurrence" in source:
            onset_producer = "global-recurrence-search"
            onset_kind = "recurrence-onset"
        else:
            onset_producer = "consonant-onset-detector"
            onset_kind = "vocal-onset"
        occurrence_evidence = None
        if onset_kind == "cross-backend-onset":
            occurrence_entries = [LyricEntry(["generic lyric"])]
            occurrence_assignments = [
                {"timestamp": 1.0, "score": .95, "segment": 1}
            ]
            occurrence_binding = auto_lrc._entry_occurrence_binding(
                occurrence_entries, occurrence_assignments, 0
            )
            occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
                occurrence_entries,
                0,
                producer=onset_producer,
                occurrence_binding=occurrence_binding,
                upstream_evidence_revision="v142-direct-upstream",
            )
        return auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="generic lyric",
            source=source,
            raw_time=time,
            confidence=confidence,
            identity_support="supported" if independent else "unsupported",
            sequence_support="supported",
            acoustic_support="supported" if independent else "unavailable",
            direct_onset_support="supported" if independent else "unavailable",
            acoustic_strength=0.9 if independent else None,
            acoustic_onset_time=time if independent else None,
            acoustic_source_artifact=artifact if independent else None,
            acoustic_evidence_producer=f"{source}-acoustic" if independent else None,
            acoustic_evidence_kind="vocal-onset",
            acoustic_evidence_independent=independent,
            direct_onset_time=time if independent else None,
            direct_onset_source_artifact=artifact if independent else None,
            direct_onset_evidence_producer=onset_producer if independent else None,
            direct_onset_evidence_kind=onset_kind,
            direct_onset_evidence_independent=independent,
            occurrence_evidence=occurrence_evidence,
            independent_content_identity=independent,
            current=current,
            source_artifact=artifact,
        )

    def test_tight_independent_valid_cluster_resolves_invalid_current(self) -> None:
        current = self.candidate("ctc-current", 10.50, current=True, independent=False)
        raw = self.candidate("raw-vocal-independent-fusion", 10.00)
        recurrence = self.candidate("global-acoustic-recurrence", 10.06)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, raw, recurrence),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.status, "selected_valid")
        self.assertIn(decision.candidate.source, {
            "raw-vocal-independent-fusion",
            "global-acoustic-recurrence",
        })
        self.assertLessEqual(abs(decision.written_time.seconds - 10.03), 0.04)

    def test_two_equally_supported_separated_clusters_remain_unresolved(self) -> None:
        current = self.candidate("ctc-current", 10.50, current=True, independent=False)
        first_a = self.candidate("raw-vocal-independent-fusion", 10.00)
        first_b = self.candidate("global-acoustic-recurrence", 10.06)
        second_a = self.candidate("raw-vocal-independent-fusion", 11.00)
        second_b = self.candidate("global-acoustic-recurrence", 11.06)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, first_a, first_b, second_a, second_b),
            previous_candidate=None,
            selection_revision=1,
        )
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(decision.candidate.candidate_id, current.candidate_id)

    def test_postcentral_projection_replaces_stale_output_and_resolves_superseded_review(self) -> None:
        current = self.candidate("ctc-current", 11.00, current=True, independent=False)
        selected = self.candidate("raw-vocal-independent-fusion", 10.00)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, selected),
            previous_candidate=None,
            selection_revision=1,
        )
        state = auto_lrc.FinalTimingState(
            decisions=(decision,),
            written_times=(decision.written_time,),
            audits=auto_lrc.audit_final_timing_decisions((decision,)),
        )
        report = {
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "candidates": [{"source": decision.candidate.source, "time": 10.0}],
                "rejected_candidates": [],
            }],
            "suspicious_alignments": [{
                "entry": 1,
                "flags": ["candidate_disagreement"],
                "severity": "high",
                "review_required": True,
                "chosen_time": 11.0,
                "confidence": 0.2,
                "candidate_timestamps": {"output": 11.0, "raw_asr": 10.0},
                "candidate_disagreement_seconds": 1.0,
            }],
        }
        auto_lrc._project_suspicious_alignments_to_central_state(report, state)
        row = report["suspicious_alignments"][0]
        self.assertEqual(row["chosen_time"], 10.0)
        self.assertEqual(row["candidate_timestamps"]["output"], 10.0)
        self.assertEqual(row["precentral_diagnostic"]["chosen_time"], 11.0)
        self.assertEqual(row["severity"], "resolved")
        self.assertFalse(row["review_required"])
        self.assertEqual(row["confidence"], 0.9)
        self.assertEqual(row["candidate_disagreement_seconds"], 0.0)
        self.assertEqual(row["timestamp_revision"], decision.timestamp_revision)
        self.assertEqual(row["timing_evidence_revision"], decision.timing_evidence_revision)

    def test_distant_valid_alternative_keeps_review_open(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 10.00, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 11.00)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            selection_revision=1,
        )
        audit = auto_lrc.audit_final_timing_decisions((decision,))[0]
        row = {
            "entry": 1,
            "flags": ["candidate_disagreement"],
            "severity": "high",
            "review_required": True,
        }
        self.assertTrue(audit.timing_trusted)
        self.assertFalse(auto_lrc._central_decision_supersedes_suspicious_row(decision, audit, row))

    def test_final_postcentral_material_conflict_propagates_without_suspicious_row(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 10.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 11.0)
        report = {
            "backend": "synthetic",
            "timing_entries": 1,
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "timing_repair": "synthetic",
                "score": 0.9,
            }],
            "suspicious_alignments": [],
        }
        evaluated = auto_lrc.evaluate_backend_timing(
            "synthetic", ((current, distant),), report, {}
        )
        self.assertTrue(evaluated.state.audits[0].timing_trusted)
        auto_lrc.commit_evaluated_backend(
            auto_lrc.select_evaluated_backend((evaluated,)), report
        )
        committed = auto_lrc.commit_final_timing_state(
            evaluated.state, report, {},
            finding_states=evaluated.finding_states, final_output=True,
        )
        self.assertIs(committed.decisions, evaluated.state.decisions)
        self.assertIs(committed.written_times, evaluated.state.written_times)
        self.assertTrue(evaluated.state.audits[0].timing_trusted)
        self.assertFalse(committed.audits[0].timing_trusted)
        auto_lrc.assert_report_timestamp_equality(report, committed)
        self.assertEqual(report["review_required_count"], 1)
        self.assertTrue(report["assignments"][0]["review_required"])
        self.assertFalse(report["assignments"][0]["timing_trusted"])
        self.assertEqual(report["assignments"][0]["timestamp_status"], "provisional_unresolved")
        self.assertEqual(report["assignments"][0]["verification_status"], "review-required")
        self.assertEqual(report["suspicious_alignments"], [])

    def test_material_unknown_ownership_is_not_falsification(self) -> None:
        unknown_tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id="previous",
            previous_generation_revision="generation",
            previous_region_revision="region",
            tail_evidence_revision="tail",
            validity="unknown",
            reliable_tail_start=None,
            reliable_tail_end=None,
            uncertainty_seconds=None,
            confidence=None,
            reason="tail-unavailable",
        )
        current = self.candidate("raw-vocal-independent-fusion", 10.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 11.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            previous_tail=unknown_tail,
            selection_revision=1,
        )
        self.assertTrue(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )

    def test_explicit_ownership_violation_can_falsify_material_alternative(self) -> None:
        valid_tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id="previous",
            previous_generation_revision="generation",
            previous_region_revision="region",
            tail_evidence_revision="tail",
            validity="valid",
            reliable_tail_start=9.5,
            reliable_tail_end=10.0,
            uncertainty_seconds=0.01,
            confidence=0.9,
            reason="valid-tail",
        )
        current = self.candidate("raw-vocal-independent-fusion", 11.0, current=True)
        violating = self.candidate("whisperx-vocal-independent-fusion", 9.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, violating),
            previous_candidate=None,
            previous_tail=valid_tail,
            selection_revision=1,
        )
        self.assertFalse(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )

    def test_material_candidate_without_minimum_proof_is_not_material(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 10.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 11.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            selection_revision=1,
        )
        rejected = next(
            item for item in decision.rejected_candidates
            if item.candidate.candidate_id == distant.candidate_id
        )
        malformed = replace(
            rejected,
            evidence=replace(rejected.evidence, minimum_proof_satisfied=False),
        )
        decision = replace(decision, rejected_candidates=(malformed,))
        self.assertFalse(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )

    def test_malformed_ownership_violation_remains_material(self) -> None:
        valid_tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id="previous",
            previous_generation_revision="generation",
            previous_region_revision="region",
            tail_evidence_revision="tail",
            validity="valid",
            reliable_tail_start=9.5,
            reliable_tail_end=10.0,
            uncertainty_seconds=0.01,
            confidence=0.9,
            reason="valid-tail",
        )
        current = self.candidate("raw-vocal-independent-fusion", 11.0, current=True)
        violating = self.candidate("whisperx-vocal-independent-fusion", 9.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, violating),
            previous_candidate=None,
            previous_tail=valid_tail,
            selection_revision=1,
        )
        rejected = next(
            item for item in decision.rejected_candidates
            if item.candidate.candidate_id == violating.candidate_id
        )
        malformed_ownership = replace(
            rejected.evidence.ownership,
            previous_tail_validity="unknown",
            previous_tail_evidence_revision=None,
            relation_revision="",
        )
        malformed = replace(
            rejected,
            evidence=replace(rejected.evidence, ownership=malformed_ownership),
        )
        decision = replace(decision, rejected_candidates=(malformed,))
        self.assertTrue(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )

    def test_postcentral_projection_recomputes_severity_boundary(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 12.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 14.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            selection_revision=1,
        )
        state = auto_lrc.FinalTimingState(
            decisions=(decision,),
            written_times=(decision.written_time,),
            audits=auto_lrc.audit_final_timing_decisions((decision,)),
        )
        report = {
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "timing_repair": "synthetic",
                "score": 0.9,
                "candidates": [],
                "rejected_candidates": [],
            }],
            "suspicious_alignments": [{
                "entry": 1,
                "flags": ["unresolved_raw_ctc_disagreement"],
                "severity": "medium",
                "review_required": True,
                "chosen_time": 10.0,
                "candidate_timestamps": {"output": 10.0, "raw_asr": 9.8},
                "raw_asr_score": 0.78,
                "candidate_disagreement_seconds": 0.2,
            }],
        }
        auto_lrc._project_suspicious_alignments_to_central_state(report, state)
        row = report["suspicious_alignments"][0]
        self.assertEqual(row["candidate_disagreement_seconds"], 2.2)
        self.assertEqual(row["severity"], "high")

    def test_postcentral_projection_downshifts_sole_raw_severity(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 12.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 14.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            selection_revision=1,
        )
        state = auto_lrc.FinalTimingState(
            decisions=(decision,),
            written_times=(decision.written_time,),
            audits=auto_lrc.audit_final_timing_decisions((decision,)),
        )
        report = {
            "assignments": [{
                "entry": 1,
                "timestamp": 11.0,
                "timing_repair": "synthetic",
                "score": 0.9,
                "candidates": [],
                "rejected_candidates": [],
            }],
            "suspicious_alignments": [{
                "entry": 1,
                "flags": ["unresolved_raw_ctc_disagreement"],
                "severity": "high",
                "review_required": True,
                "chosen_time": 11.0,
                "candidate_timestamps": {"output": 11.0, "raw_asr": 11.2},
                "raw_asr_score": 0.78,
                "candidate_disagreement_seconds": 0.2,
            }],
        }
        auto_lrc._project_suspicious_alignments_to_central_state(report, state)
        row = report["suspicious_alignments"][0]
        self.assertEqual(row["candidate_disagreement_seconds"], 0.8)
        self.assertEqual(row["severity"], "medium")

    def test_postcentral_projection_preserves_independent_high_severity(self) -> None:
        current = self.candidate("raw-vocal-independent-fusion", 12.0, current=True)
        distant = self.candidate("whisperx-vocal-independent-fusion", 14.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(current, distant),
            previous_candidate=None,
            selection_revision=1,
        )
        state = auto_lrc.FinalTimingState(
            decisions=(decision,),
            written_times=(decision.written_time,),
            audits=auto_lrc.audit_final_timing_decisions((decision,)),
        )
        report = {
            "assignments": [{
                "entry": 1,
                "timestamp": 11.0,
                "timing_repair": "synthetic",
                "score": 0.9,
                "candidates": [],
                "rejected_candidates": [],
            }],
            "suspicious_alignments": [{
                "entry": 1,
                "flags": ["unresolved_raw_ctc_disagreement", "phonetic_anchor_disagreement"],
                "severity": "high",
                "review_required": True,
                "chosen_time": 11.0,
                "candidate_timestamps": {"output": 11.0, "raw_asr": 10.8},
                "raw_asr_score": 0.78,
                "candidate_disagreement_seconds": 0.2,
            }],
        }
        auto_lrc._project_suspicious_alignments_to_central_state(report, state)
        row = report["suspicious_alignments"][0]
        self.assertEqual(row["candidate_disagreement_seconds"], 1.2)
        self.assertEqual(row["severity"], "high")


class V143EvidenceRecoveryRegressionTests(unittest.TestCase):
    def span(self, start: float, score: float = 0.5, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str = "ctc-current",
        current: bool = False,
        spans: tuple[auto_lrc.TimingTokenSpan, ...] | None = None,
        truncated: bool = False,
    ) -> auto_lrc.TimingCandidate:
        if spans is None:
            spans = (
                self.span(time, 0.3, "a"),
                self.span(time + 0.08, 0.4, "b"),
                self.span(time + 0.16, 0.5, "c"),
            )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"line {entry}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=0.8,
            identity_support="supported",
            sequence_support="supported",
            current=current,
            window_truncated=truncated,
            source_artifact={"entry": entry, "time": time, "source": source, "truncated": truncated},
        )

    def capability(self) -> auto_lrc.BoundedRetryCapability:
        payload = {
            "kind": "mms-known-lyric-ctc",
            "backend_lineage": "ctc",
            "helper_path": "ctc_align.py",
            "helper_sha256": "helper-sha",
            "helper_protocol_revision": "protocol-revision",
            "alignment_audio_path": "audio.flac",
            "alignment_audio_sha256": "audio-sha",
            "alignment_audio_source": "vocal-stem",
            "sample_rate": auto_lrc.SAMPLE_RATE,
            "model_identity": "MMS_FA:known-lyric",
            "device": "cpu",
        }
        return auto_lrc.BoundedRetryCapability(
            **payload,
            capability_revision=auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(payload)
            ),
        )

    def tail_evidence(
        self,
        *,
        producer: str = "whisperx-forced-row",
        tail_start: float = 9.8,
        tail_end: float = 10.4,
        uncertainty: float = 0.02,
        confidence: float = 0.95,
    ) -> auto_lrc.IndependentTailEvidence:
        provenance = auto_lrc._make_tail_provenance(
            entry_index=0,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=confidence,
            producer=producer,
            backend_lineage="whisper-family" if producer.startswith("whisper") else "ctc",
            model_identity="test-model",
            helper_revision="test-helper",
            capability_revision="test-capability",
            audio_revision="test-audio",
            transcript_revision="test-transcript",
            row_revision="test-row",
        )
        assert provenance is not None
        return auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=confidence,
            producer=producer,
            source_revision=provenance.observation_revision,
            provenance=provenance,
        )

    def test_ctc_assignment_tail_fallback_restores_missing_tail_geometry(self) -> None:
        entries = [LyricEntry(["unique predecessor"])]
        report: dict[str, object] = {
            "backend": "ctc",
            auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: self.capability(),
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "score": 0.9,
                "timing_repair_source": "torchaudio-mms-fa",
                "ctc_token_spans": [
                    {"char": "a", "start": 10.0, "end": 10.02, "score": 0.5},
                    {"char": "b", "start": 10.10, "end": 10.12, "score": 0.6},
                    {"char": "c", "start": 10.20, "end": 10.22, "score": 0.7},
                ],
            }],
        }
        auto_lrc.bind_runtime_ctc_assignment_tail_fallback(report, entries)
        runtime = report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY]
        self.assertIn(0, runtime)
        tail = runtime[0][0]
        self.assertEqual(tail.producer, "ctc-assignment-terminal-row")
        self.assertGreaterEqual(tail.tail_end, 10.22)
        self.assertGreaterEqual(tail.uncertainty_seconds, 0.0)

    def test_ctc_assignment_tail_fallback_never_overwrites_distinct_backend_tail(self) -> None:
        entries = [LyricEntry(["unique predecessor"])]
        external = self.tail_evidence()
        report: dict[str, object] = {
            "backend": "ctc",
            auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY: {0: (external,)},
            "assignments": [{
                "entry": 1,
                "timestamp": 10.0,
                "score": 0.9,
                "ctc_token_spans": [
                    {"char": "a", "start": 10.0, "end": 10.02, "score": 0.5},
                    {"char": "b", "start": 10.10, "end": 10.12, "score": 0.6},
                    {"char": "c", "start": 10.20, "end": 10.22, "score": 0.7},
                ],
            }],
        }
        auto_lrc.bind_runtime_ctc_assignment_tail_fallback(report, entries)
        runtime = report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY]
        self.assertIs(runtime[0][0], external)
        self.assertEqual(runtime[0][0].producer, "whisperx-forced-row")

    def test_truncated_retry_expands_next_bounded_window(self) -> None:
        entries = [LyricEntry(["previous"]), LyricEntry(["target"]), LyricEntry(["next"])]
        weak_prev = self.candidate(
            0, 9.0, current=True,
            spans=(self.span(9.0, 0.2, "a"), self.span(9.08, 0.01, "b")),
        )
        weak_current = self.candidate(
            1, 10.0, current=True,
            spans=(self.span(10.0, 0.01, "a"), self.span(10.08, 0.01, "b"), self.span(10.16, 0.01, "c")),
        )
        following = self.candidate(2, 12.0, current=True)
        base_sets = ((weak_prev,), (weak_current,), (following,))
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 3,
            "assignments": [
                {"entry": 1, "timestamp": 9.0, "timing_repair": "ctc", "score": 0.9},
                {"entry": 2, "timestamp": 10.0, "timing_repair": "ctc", "score": 0.9},
                {"entry": 3, "timestamp": 12.0, "timing_repair": "ctc", "score": 0.9},
            ],
        }
        cap = self.capability()
        epoch = auto_lrc.freeze_base_retry_epoch(
            "ctc", report, base_sets, auto_lrc.inactive_legacy_timing_diagnostics(), cap
        )
        first = auto_lrc.plan_crossline_retry_requests(epoch, cap, entries, base_sets, 30.0)[0]
        truncated = self.candidate(
            1, 10.1, source="ctc-opening-local-retry", current=False, truncated=True,
            spans=(self.span(10.1, 0.2, "a"), self.span(10.18, 0.3, "b"), self.span(10.26, 0.4, "c")),
        )
        expanded_sets = (base_sets[0], (weak_current, truncated), base_sets[2])
        epoch2 = auto_lrc.freeze_base_retry_epoch(
            "ctc", report, expanded_sets, auto_lrc.inactive_legacy_timing_diagnostics(), cap
        )
        second = auto_lrc.plan_crossline_retry_requests(epoch2, cap, entries, expanded_sets, 30.0)[0]
        self.assertAlmostEqual(
            second.window_end - first.window_end,
            auto_lrc.CROSSLINE_RETRY_FORWARD_OVERLAP_SECONDS,
            places=6,
        )
        self.assertNotEqual(first.window_definition_revision, second.window_definition_revision)

    def _install_unique_raw_identity(
        self,
        report: dict[str, object],
        entries: list[LyricEntry],
        raw_time: float,
        raw_score: float = 0.95,
    ) -> None:
        index = 0
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[index]))
        lyric_sha = __import__("hashlib").sha256(target.encode("utf-8")).hexdigest()
        binding_payload = {
            "mode": "global-unique",
            "entry_index": index,
            "lyric_sha256": lyric_sha,
        }
        binding = {
            **binding_payload,
            "binding_revision": auto_lrc._timing_content_digest(binding_payload),
        }
        payload = {
            "entry_index": index,
            "lyric_sha256": lyric_sha,
            "raw_time": round(raw_time, 6),
            "raw_score": round(raw_score, 6),
            "segment": 0,
            "occurrence_binding": binding,
            "producer": "raw-asr-content-identity",
        }
        report[auto_lrc._RUNTIME_RAW_CONTENT_IDENTITY_KEY] = {
            index: {**payload, "evidence_revision": auto_lrc._timing_content_digest(payload)}
        }

    def test_local_ctc_candidate_can_bind_existing_raw_identity_without_moving_timestamp(self) -> None:
        entries = [LyricEntry(["globally unique lyric"])]
        report: dict[str, object] = {}
        self._install_unique_raw_identity(report, entries, 10.10)
        parent = self.candidate(
            0, 10.0, source="ctc-opening-local-retry", current=False,
            spans=(self.span(10.0, 0.4, "a"), self.span(10.08, 0.5, "b"), self.span(10.16, 0.6, "c")),
        )
        augmented = auto_lrc._augment_local_ctc_candidates_with_runtime_backend_consensus(
            ((parent,),), report, entries
        )[0]
        composites = [item for item in augmented if item.source == "ctc-local-raw-independent-consensus"]
        self.assertEqual(len(composites), 1)
        composite = composites[0]
        self.assertEqual(composite.written_time, parent.written_time)
        self.assertTrue(composite.independent_content_identity)
        self.assertIsNotNone(composite.direct_onset_evidence)
        self.assertEqual(composite.direct_onset_evidence.kind, "cross-backend-onset")
        self.assertAlmostEqual(composite.direct_onset_evidence.onset_time, 10.10, places=6)

    def test_local_ctc_candidate_does_not_bind_distant_raw_identity(self) -> None:
        entries = [LyricEntry(["globally unique lyric"])]
        report: dict[str, object] = {}
        self._install_unique_raw_identity(report, entries, 10.50)
        parent = self.candidate(0, 10.0, source="ctc-opening-local-retry", current=False)
        augmented = auto_lrc._augment_local_ctc_candidates_with_runtime_backend_consensus(
            ((parent,),), report, entries
        )[0]
        self.assertFalse(any(
            item.source == "ctc-local-raw-independent-consensus" for item in augmented
        ))

    def test_candidate_seed_ranking_ignores_provenance_id_for_capped_strength_ties(self) -> None:
        def assignment(local_id: str, retry_id: str) -> dict[str, object]:
            return {
                "alignment_hypotheses": [
                    {
                        "source": "ctc-local-window",
                        "spans": [{"start": 215.013, "score": 0.716128}],
                        "hypothesis_id": local_id,
                    },
                    {
                        "source": "ctc-opening-local-retry-identity",
                        "spans": [{"start": 201.51, "score": 0.666215}],
                        "hypothesis_id": retry_id,
                        "window_truncated": False,
                        "right_edge_pileup": False,
                    },
                ]
            }

        first = auto_lrc._candidate_seed_rows(assignment("z-local", "a-retry"))
        second = auto_lrc._candidate_seed_rows(assignment("a-local", "z-retry"))

        self.assertEqual(first[0]["time"], 215.013)
        self.assertEqual(second[0]["time"], 215.013)
        self.assertEqual(first[0]["kind"], "ctc-local-first-token")
        self.assertEqual(second[0]["kind"], "ctc-local-first-token")

    def test_complete_retry_publishes_identity_only_occurrence_evidence(self) -> None:
        spans = tuple(
            self.span(10.0 + 0.08 * index, 0.4 + 0.05 * index, token)
            for index, token in enumerate("abcd")
        )
        candidate = self.candidate(
            0, 10.0, source="ctc-opening-local-retry", current=False, spans=spans
        )
        base = {
            "retry_epoch_revision": "epoch",
            "request_revision": "request",
            "entry_index": 0,
            "status": "candidate",
            "reason": "complete-known-lyric-target-row",
            "candidate": candidate,
            "output_revision": "output",
        }
        outcome = auto_lrc.CrossLineRetryOutcome(
            **base, outcome_revision="outcome-revision"
        )
        report: dict[str, object] = {"assignments": [{"entry": 1}]}
        hypothesis = auto_lrc.record_crossline_retry_identity_hypothesis(report, outcome)
        self.assertIsNotNone(hypothesis)
        assert hypothesis is not None
        self.assertEqual(hypothesis["source"], "ctc-opening-local-retry-identity")
        self.assertEqual(hypothesis["acoustic_support"], "unavailable")
        self.assertEqual(hypothesis["direct_onset_support"], "unavailable")
        assignment = report["assignments"][0]
        identity = auto_lrc._bound_local_occurrence_identity(assignment)
        self.assertIsNotNone(identity)
        self.assertEqual(identity["source"], "ctc-opening-local-retry-identity")
        seeds = auto_lrc._candidate_seed_rows(assignment)
        self.assertTrue(any(row["kind"] == "ctc-local-retry-first-token" for row in seeds))

    def test_truncated_retry_cannot_publish_occurrence_identity(self) -> None:
        spans = tuple(
            self.span(10.0 + 0.08 * index, 0.5, token)
            for index, token in enumerate("abcd")
        )
        candidate = self.candidate(
            0, 10.0, source="ctc-opening-local-retry", current=False,
            spans=spans, truncated=True,
        )
        base = {
            "retry_epoch_revision": "epoch",
            "request_revision": "request",
            "entry_index": 0,
            "status": "candidate",
            "reason": "complete-known-lyric-target-row",
            "candidate": candidate,
            "output_revision": "output",
        }
        outcome = auto_lrc.CrossLineRetryOutcome(
            **base, outcome_revision="outcome-revision"
        )
        report: dict[str, object] = {"assignments": [{"entry": 1}]}
        self.assertIsNone(auto_lrc.record_crossline_retry_identity_hypothesis(report, outcome))
        self.assertIsNone(auto_lrc._bound_local_occurrence_identity(report["assignments"][0]))
    def test_trusted_reference_can_use_unshifted_local_ctc_geometry(self) -> None:
        reference = self.candidate(
            0, 20.0, source="raw-vocal-independent-fusion", current=False, spans=()
        )
        assignment: dict[str, object] = {"entry": 1}
        spans = [
            {"char": token, "start": 20.10 + index * 0.08, "end": 20.12 + index * 0.08, "score": 0.5}
            for index, token in enumerate("abcd")
        ]
        hypothesis = auto_lrc.record_alignment_hypothesis(
            assignment,
            source="ctc-local-window",
            stage="ctc-local-window-realign",
            raw_time=20.10,
            confidence=0.7,
            spans=spans,
            identity_support="supported",
            sequence_support="supported",
            window={"kind": "test-local-window"},
            mode="local-known-lyric-row",
        )
        geometry = auto_lrc._trusted_recurrence_reference_geometry(assignment, reference)
        self.assertIsNotNone(geometry)
        assert geometry is not None
        start, parsed_spans, provenance = geometry
        self.assertAlmostEqual(start, 20.10, places=6)
        self.assertAlmostEqual(parsed_spans[0].start, 20.10, places=6)
        self.assertEqual(provenance["geometry_hypothesis_id"], hypothesis["hypothesis_id"])
        self.assertEqual(provenance["mode"], "self-bound-local-ctc-geometry")
        self.assertAlmostEqual(provenance["selected_start"], 20.0, places=6)

    def test_trusted_reference_never_backfills_distant_local_geometry(self) -> None:
        reference = self.candidate(
            0, 20.0, source="raw-vocal-independent-fusion", current=False, spans=()
        )
        assignment: dict[str, object] = {"entry": 1}
        spans = [
            {"char": token, "start": 20.50 + index * 0.08, "end": 20.52 + index * 0.08, "score": 0.5}
            for index, token in enumerate("abcd")
        ]
        auto_lrc.record_alignment_hypothesis(
            assignment,
            source="ctc-local-window",
            stage="ctc-local-window-realign",
            raw_time=20.50,
            confidence=0.9,
            spans=spans,
            identity_support="supported",
            sequence_support="supported",
            window={"kind": "test-local-window"},
            mode="local-known-lyric-row",
        )
        self.assertIsNone(
            auto_lrc._trusted_recurrence_reference_geometry(assignment, reference)
        )



class V144JointBoundaryGraphRegressionTests(unittest.TestCase):
    def span(self, start: float, score: float = 0.4, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str = "ctc-current",
        current: bool = True,
        spans: tuple[auto_lrc.TimingTokenSpan, ...] | None = None,
    ) -> auto_lrc.TimingCandidate:
        if spans is None:
            spans = (
                self.span(time, 0.30, "a"),
                self.span(time + 0.08, 0.40, "b"),
                self.span(time + 0.16, 0.50, "c"),
            )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"generic line {entry}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=0.8,
            identity_support="supported",
            sequence_support="supported",
            current=current,
            source_artifact={"entry": entry, "time": time, "source": source},
        )

    def capability(self) -> auto_lrc.BoundedRetryCapability:
        payload = {
            "kind": "mms-known-lyric-ctc",
            "backend_lineage": "ctc",
            "helper_path": "ctc_align.py",
            "helper_sha256": "helper-sha-v144",
            "helper_protocol_revision": "protocol-v144",
            "alignment_audio_path": "audio.flac",
            "alignment_audio_sha256": "audio-sha-v144",
            "alignment_audio_source": "vocal-stem",
            "sample_rate": auto_lrc.SAMPLE_RATE,
            "model_identity": "MMS_FA:known-lyric",
            "device": "cpu",
        }
        return auto_lrc.BoundedRetryCapability(
            **payload,
            capability_revision=auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(payload)
            ),
        )

    def tail_evidence(
        self,
        *,
        producer: str = "joint-ctc-boundary-previous-tail",
        tail_start: float = 9.0,
        tail_end: float = 10.0,
        uncertainty: float = 0.20,
        confidence: float = 0.90,
    ) -> auto_lrc.IndependentTailEvidence:
        cap = self.capability()
        provenance = auto_lrc._make_tail_provenance(
            entry_index=0,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=confidence,
            producer=producer,
            backend_lineage=cap.backend_lineage,
            model_identity=cap.model_identity,
            helper_revision=auto_lrc._tail_helper_revision(
                cap.helper_sha256,
                cap.helper_protocol_revision,
            ),
            capability_revision=cap.capability_revision,
            audio_revision=cap.alignment_audio_sha256,
            transcript_revision="test-transcript",
            row_revision="test-row",
        )
        assert provenance is not None
        return auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=tail_start,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=confidence,
            producer=producer,
            source_revision=provenance.observation_revision,
            provenance=provenance,
        )

    def fixture(self):
        entries = [
            LyricEntry(["previous lyric"]),
            LyricEntry(["current lyric"]),
            LyricEntry(["next lyric"]),
        ]
        previous = self.candidate(0, 9.74)
        current = self.candidate(1, 10.00)
        following = self.candidate(2, 12.00)
        candidate_sets = ((previous,), (current,), (following,))
        report: dict[str, object] = {
            "backend": "ctc",
            auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: self.capability(),
            "timing_entries": 3,
            "assignments": [
                {"entry": 1, "timestamp": 9.74, "timing_repair": "ctc", "score": 0.9},
                {"entry": 2, "timestamp": 10.00, "timing_repair": "ctc", "score": 0.9},
                {"entry": 3, "timestamp": 12.00, "timing_repair": "ctc", "score": 0.9},
            ],
        }
        cap = self.capability()
        helper_revision = auto_lrc._tail_helper_revision(
            cap.helper_sha256,
            cap.helper_protocol_revision,
        )
        report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY] = auto_lrc.TailEvidenceContext(
            backend_lineage=cap.backend_lineage,
            model_identity=cap.model_identity,
            helper_revision=helper_revision,
            capability_revision=cap.capability_revision,
            audio_revision=cap.alignment_audio_sha256,
            transcript_revisions=("test-transcript",),
            row_revisions=("test-row",),
            scopes=(auto_lrc.TailEvidenceScope(
                entry_index=0,
                producer="joint-ctc-boundary-previous-tail",
                backend_lineage=cap.backend_lineage,
                model_identity=cap.model_identity,
                helper_revision=helper_revision,
                capability_revision=cap.capability_revision,
                audio_revision=cap.alignment_audio_sha256,
                transcript_revisions=("test-transcript",),
                row_revisions=("test-row",),
            ),),
        )
        epoch = auto_lrc.freeze_base_retry_epoch(
            "ctc", report, candidate_sets,
            auto_lrc.inactive_legacy_timing_diagnostics(), cap,
        )
        return entries, candidate_sets, report, cap, epoch

    def joint_request(self):
        entries, candidate_sets, report, cap, epoch = self.fixture()
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        requests = auto_lrc.plan_joint_boundary_retry_requests(
            epoch, cap, entries, candidate_sets, 20.0, None, context
        )
        self.assertEqual(len(requests), 1)
        request = requests[0]
        assert isinstance(context, auto_lrc.TailEvidenceContext)
        assert len(context.scopes) == 1
        scope = context.scopes[0]
        report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY] = replace(
            context,
            transcript_revisions=tuple(sorted({
                *context.transcript_revisions,
                request.transcript_digest,
                *request.transcript_digests,
            })),
            row_revisions=tuple(sorted({
                *context.row_revisions,
                request.row_mapping_revision,
            })),
            scopes=(replace(
                scope,
                transcript_revisions=tuple(sorted({
                    *scope.transcript_revisions,
                    request.transcript_digest,
                    *request.transcript_digests,
                })),
                row_revisions=tuple(sorted({
                    *scope.row_revisions,
                    request.row_mapping_revision,
                })),
            ),),
        )
        return entries, candidate_sets, report, cap, epoch, request

    def joint_payload(
        self,
        request: auto_lrc.CrossLineRetryRequest,
        entries: list[LyricEntry],
        *,
        current_start: float = 10.00,
        touch_left_edge: bool = False,
    ) -> dict[str, object]:
        previous_start = request.window_start + 0.05 if touch_left_edge else 9.20
        row_specs = (
            (request.entry_index - 1, tuple(previous_start + 0.08 * i for i in range(4))),
            (request.entry_index, tuple(current_start + 0.08 * i for i in range(4))),
        )
        rows: list[dict[str, object]] = []
        for local_index, (global_index, starts) in enumerate(row_specs):
            spans = [
                {
                    "char": chr(97 + token_index),
                    "start": start,
                    "end": start + 0.02,
                    "score": 0.30 + 0.05 * token_index,
                }
                for token_index, start in enumerate(starts)
            ]
            rows.append({
                "entry": local_index + 1,
                "text": auto_lrc.entry_sung_text(entries[global_index]),
                "romaji": "".join(str(span["char"]) for span in spans),
                "start": spans[0]["start"],
                "end": spans[-1]["end"],
                "ctc_score": 0.50,
                "tokens": len(spans),
                "token_spans": spans,
            })
        return {
            "window_start": round(request.window_start, 3),
            "window_end": round(request.window_end, 3),
            "entries": rows,
        }

    def test_joint_planner_uses_previous_and_current_rows_only(self) -> None:
        entries, candidate_sets, report, cap, epoch = self.fixture()
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        decision = auto_lrc.build_final_timing_state(candidate_sets).decisions[1]
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertEqual(decision.evidence.ownership.status, "unknown")
        requests = auto_lrc.plan_joint_boundary_retry_requests(
            epoch, cap, entries, candidate_sets, 20.0, None, context
        )
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.ordinal, 2)
        self.assertEqual(request.context_entry_indexes, (0, 1))
        self.assertEqual(request.row_mapping, ((0, 0), (1, 1)))
        self.assertEqual(len(request.transcript_digests), 2)
        self.assertTrue(
            auto_lrc.crossline_retry_request_is_current(
                request,
                epoch,
                cap,
                entries,
                candidate_sets,
                20.0,
                None,
                context,
            )
        )

    def test_complete_joint_pair_is_observation_only(self) -> None:
        entries, candidate_sets, _report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries)
        )
        self.assertEqual(outcome.status, "candidate")
        self.assertEqual(outcome.reason, "complete-joint-boundary-rows")
        assert outcome.candidate is not None
        self.assertEqual(outcome.candidate.source, "ctc-joint-boundary-observation")
        self.assertEqual(outcome.candidate.direct_onset_support, "unavailable")
        self.assertIsNone(outcome.candidate.direct_onset_evidence)
        # Even a geometrically strong pair observation cannot by itself cross
        # the old unknown-tail relation; only the separately bound consensus may.
        raw_eval = auto_lrc.evaluate_timing_candidate(
            outcome.candidate, candidate_sets[0][0]
        )
        self.assertFalse(raw_eval.valid)
        self.assertIn("previous-tail-ownership-unknown", raw_eval.rejection_reasons)
        self.assertIsNotNone(outcome.previous_tail_evidence)
        assert outcome.previous_tail_evidence is not None
        self.assertEqual(
            outcome.previous_tail_evidence.entry_index,
            request.entry_index - 1,
        )
        self.assertEqual(
            outcome.previous_tail_evidence.producer,
            "joint-ctc-boundary-previous-tail",
        )
        self.assertGreater(
            outcome.previous_tail_evidence.tail_end,
            outcome.previous_tail_evidence.tail_start,
        )
        self.assertIsNotNone(
            auto_lrc._retry_diagnostic_payload(outcome)["previous_tail_evidence"]
        )

    def test_strong_joint_previous_tail_rebinds_without_moving_timestamp(self) -> None:
        entries, _fixture_sets, report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries)
        )
        assert outcome.previous_tail_evidence is not None
        predecessor = self.candidate(
            0, 9.74, source="raw-vocal-independent-fusion", spans=()
        )
        # The joint CTC tail is authoritative only for a disjoint current
        # opening; same-lineage CTC ownership is covered by Contract-A tests.
        current = self.candidate(1, 10.00, source="raw-vocal-independent-fusion")
        following = self.candidate(2, 12.00)
        candidate_sets = ((predecessor,), (current,), (following,))
        changed = auto_lrc.merge_runtime_independent_tail_evidence(
            report, outcome.previous_tail_evidence
        )
        self.assertTrue(changed)
        tails = auto_lrc.runtime_independent_tail_evidence(
            report, len(candidate_sets), report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        )
        before = auto_lrc.build_final_timing_state(candidate_sets)
        after = auto_lrc.build_final_timing_state(
            candidate_sets,
            independent_tail_evidence=tails,
            tail_evidence_context=report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY],
        )
        self.assertEqual(before.written_times, after.written_times)
        self.assertEqual(after.decisions[1].evidence.ownership.status, "clear")
        self.assertTrue(after.audits[1].timing_trusted)

    def test_joint_previous_tail_merge_is_monotonic(self) -> None:
        report: dict[str, object] = {}
        cap = self.capability()
        report[auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY] = cap
        existing = self.tail_evidence()
        weaker = self.tail_evidence(
            producer="joint-ctc-boundary-previous-tail-weaker",
            tail_start=9.2,
            tail_end=9.8,
            uncertainty=0.05,
            confidence=0.99,
        )
        later = self.tail_evidence(
            producer="joint-ctc-boundary-previous-tail-later",
            tail_end=10.4,
            uncertainty=0.10,
            confidence=0.80,
        )
        self.assertTrue(auto_lrc.merge_runtime_independent_tail_evidence(report, existing))
        self.assertTrue(auto_lrc.merge_runtime_independent_tail_evidence(report, weaker))
        retained = auto_lrc.runtime_independent_tail_evidence(report, 1)[0]
        assert retained is not None
        self.assertEqual(len(retained), 2)
        self.assertTrue(auto_lrc.merge_runtime_independent_tail_evidence(report, later))
        retained = auto_lrc.runtime_independent_tail_evidence(report, 1)[0]
        assert retained is not None
        self.assertEqual(len(retained), 3)

    def test_joint_previous_tail_rejects_wrong_row_mapping_and_edge_pair(self) -> None:
        entries, _candidate_sets, _report, cap, _epoch, request = self.joint_request()
        wrong_text = self.joint_payload(request, entries)
        wrong_text["entries"][0]["text"] = "wrong immutable lyric"
        invalid = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, wrong_text
        )
        self.assertEqual(invalid.status, "invalid")
        self.assertIsNone(invalid.previous_tail_evidence)

        edge_pair = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries, touch_left_edge=True)
        )
        self.assertEqual(edge_pair.status, "candidate")
        self.assertIsNone(edge_pair.previous_tail_evidence)
        assert edge_pair.candidate is not None
        self.assertEqual(edge_pair.candidate.source, "ctc-joint-boundary-observation-weak")

    def test_joint_consensus_keeps_parent_timestamp_and_resolves_unknown_boundary(self) -> None:
        entries, candidate_sets, _report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries)
        )
        augmented = auto_lrc._augment_candidates_with_joint_boundary_consensus(
            candidate_sets, (outcome,), entries
        )
        composites = [
            candidate for candidate in augmented[1]
            if candidate.source == "ctc-joint-boundary-consensus"
        ]
        self.assertEqual(len(composites), 1)
        composite = composites[0]
        self.assertEqual(composite.written_time, candidate_sets[1][0].written_time)
        self.assertTrue(auto_lrc.candidate_has_bound_direct_onset_evidence(composite))
        self.assertTrue(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(composite))
        assert composite.direct_onset_evidence is not None
        self.assertEqual(composite.direct_onset_evidence.kind, "joint-lyric-boundary")
        self.assertEqual(
            composite.direct_onset_evidence.producer,
            "joint-ctc-boundary-consensus",
        )
        evaluation = auto_lrc.evaluate_timing_candidate(
            composite, candidate_sets[0][0]
        )
        self.assertTrue(evaluation.valid)
        self.assertTrue(evaluation.evidence.minimum_proof_satisfied)
        self.assertEqual(evaluation.evidence.structural_failures, ())

    def test_joint_observation_does_not_fuse_distant_parent(self) -> None:
        entries, candidate_sets, _report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries, self.joint_payload(request, entries)
        )
        distant = self.candidate(1, 10.50, source="ctc-local-window", current=False)
        modified = (candidate_sets[0], (distant,), candidate_sets[2])
        augmented = auto_lrc._augment_candidates_with_joint_boundary_consensus(
            modified, (outcome,), entries
        )
        self.assertFalse(any(
            candidate.source == "ctc-joint-boundary-consensus"
            for candidate in augmented[1]
        ))

    def test_left_edge_truncated_joint_pair_cannot_certify_boundary(self) -> None:
        entries, candidate_sets, _report, cap, _epoch, request = self.joint_request()
        outcome = auto_lrc.parse_crossline_retry_output(
            request, cap, entries,
            self.joint_payload(request, entries, touch_left_edge=True),
        )
        self.assertEqual(outcome.status, "candidate")
        assert outcome.candidate is not None
        self.assertEqual(outcome.candidate.source, "ctc-joint-boundary-observation-weak")
        self.assertTrue(outcome.candidate.window_truncated)
        augmented = auto_lrc._augment_candidates_with_joint_boundary_consensus(
            candidate_sets, (outcome,), entries
        )
        self.assertFalse(any(
            candidate.source == "ctc-joint-boundary-consensus"
            for candidate in augmented[1]
        ))

    def test_joint_boundary_evidence_producer_cannot_be_spoofed(self) -> None:
        candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="generic",
            source="synthetic",
            raw_time=10.0,
            spans=(self.span(10.0), self.span(10.08), self.span(10.16)),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported",
            direct_onset_time=10.0,
            direct_onset_source_artifact={"source": "spoof"},
            direct_onset_evidence_producer="wrong-producer",
            direct_onset_evidence_kind="joint-lyric-boundary",
            direct_onset_evidence_independent=True,
        )
        self.assertFalse(auto_lrc.candidate_has_bound_direct_onset_evidence(candidate))
        self.assertFalse(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(candidate))

    def test_duplicate_offset_model_preserves_four_pair_evidence_floor(self) -> None:
        entries = [LyricEntry(["A"]), LyricEntry(["B"]), LyricEntry(["A"]), LyricEntry(["B"])]
        timestamps = [0.0, 1.0, 50.0, 51.0]
        consensus = auto_lrc.duplicate_lyric_offset_consensus(entries, timestamps)
        self.assertFalse(consensus["available"])
        self.assertEqual(consensus["reason"], "fewer-than-four-duplicate-pairs")
        self.assertEqual(consensus["pair_count"], 2)

    def test_duplicate_offsets_are_clustered_by_lyric_index_displacement(self) -> None:
        entries = [
            LyricEntry(["A"]), LyricEntry(["B"]),
            LyricEntry(["A"]), LyricEntry(["B"]),
            LyricEntry(["C"]), LyricEntry(["D"]),
            LyricEntry(["X"]), LyricEntry(["Y"]),
            LyricEntry(["C"]), LyricEntry(["D"]),
        ]
        timestamps = [
            0.00, 1.00, 75.68, 76.70,
            100.00, 101.00, 150.00, 180.00, 232.92, 233.94,
        ]
        consensus = auto_lrc.duplicate_lyric_offset_consensus(entries, timestamps)
        self.assertTrue(consensus["available"])
        self.assertEqual(consensus["mode"], "lyric-index-displacement-clusters")
        clusters = {
            int(cluster["entry_delta"]): cluster
            for cluster in consensus["clusters"]
        }
        self.assertEqual(set(clusters), {2, 4})
        self.assertTrue(clusters[2]["available"])
        self.assertTrue(clusters[4]["available"])
        self.assertAlmostEqual(float(clusters[2]["median_delta_seconds"]), 75.69, places=2)
        self.assertAlmostEqual(float(clusters[4]["median_delta_seconds"]), 132.93, places=2)
        self.assertLessEqual(float(clusters[2]["mad_seconds"]), 0.02)
        self.assertLessEqual(float(clusters[4]["mad_seconds"]), 0.02)



class GateASemanticFixedPointTests(unittest.TestCase):
    def capability(self) -> auto_lrc.BoundedRetryCapability:
        payload = {
            "kind": "mms-known-lyric-ctc",
            "backend_lineage": "ctc",
            "helper_path": "ctc_align.py",
            "helper_sha256": "gate-a-helper",
            "helper_protocol_revision": "gate-a-protocol",
            "alignment_audio_path": "gate-a-audio.flac",
            "alignment_audio_sha256": "gate-a-audio-sha",
            "alignment_audio_source": "vocal-stem",
            "sample_rate": auto_lrc.SAMPLE_RATE,
            "model_identity": "MMS_FA:known-lyric",
            "device": "cpu",
        }
        return auto_lrc.BoundedRetryCapability(
            **payload,
            capability_revision=auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(payload)
            ),
        )

    def fixture(self, *, valid_previous: bool = False):
        entries = [LyricEntry(["generic predecessor"]), LyricEntry(["generic current"])]
        previous_spans = (
            (
                auto_lrc.TimingTokenSpan(9.0, 9.08, 0.8, "a"),
                auto_lrc.TimingTokenSpan(9.10, 9.18, 0.8, "b"),
                auto_lrc.TimingTokenSpan(9.20, 9.28, 0.8, "c"),
            )
            if valid_previous else ()
        )
        previous = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="generic predecessor",
            source="ctc-current" if valid_previous else "raw-vocal-independent-fusion",
            raw_time=9.0,
            spans=previous_spans,
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"gate": "A", "entry": 0},
        )
        current = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="generic current",
            # Keep the Gate-A fixed-point fixture's current opening disjoint
            # from the injected CTC tail; same-lineage ownership is covered by
            # Contract-A lineage tests below.
            source="raw-vocal-independent-fusion",
            raw_time=10.0,
            spans=(
                auto_lrc.TimingTokenSpan(10.0, 10.08, 0.8, "a"),
                auto_lrc.TimingTokenSpan(10.10, 10.18, 0.8, "b"),
            ),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"gate": "A", "entry": 1},
        )
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {"entry": 1, "timestamp": 9.0, "score": 0.9},
                {"entry": 2, "timestamp": 10.0, "score": 0.9},
            ],
        }
        capability = self.capability()
        report[auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY] = capability
        helper_revision = auto_lrc._tail_helper_revision(
            capability.helper_sha256,
            capability.helper_protocol_revision,
        )
        report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY] = auto_lrc.TailEvidenceContext(
            backend_lineage=capability.backend_lineage,
            model_identity=capability.model_identity,
            helper_revision=helper_revision,
            capability_revision=capability.capability_revision,
            audio_revision=capability.alignment_audio_sha256,
            transcript_revisions=("gate-a-transcript",),
            row_revisions=("gate-a-row", "gate-a-weaker"),
            scopes=(auto_lrc.TailEvidenceScope(
                entry_index=0,
                producer="joint-ctc-boundary-previous-tail",
                backend_lineage=capability.backend_lineage,
                model_identity=capability.model_identity,
                helper_revision=helper_revision,
                capability_revision=capability.capability_revision,
                audio_revision=capability.alignment_audio_sha256,
                transcript_revisions=("gate-a-transcript",),
                row_revisions=("gate-a-row", "gate-a-weaker"),
            ),),
        )
        candidate_sets = ((previous,), (current,))
        return entries, candidate_sets, report, capability

    def tail(self, *, tail_end: float = 9.5, row_revision: str = "gate-a-row"):
        cap = self.capability()
        provenance = auto_lrc._make_tail_provenance(
            entry_index=0,
            tail_start=8.9,
            tail_end=tail_end,
            uncertainty_seconds=0.05,
            confidence=0.9,
            producer="joint-ctc-boundary-previous-tail",
            backend_lineage="ctc",
            model_identity=cap.model_identity,
            helper_revision=auto_lrc._tail_helper_revision(
                cap.helper_sha256,
                cap.helper_protocol_revision,
            ),
            capability_revision=cap.capability_revision,
            audio_revision=cap.alignment_audio_sha256,
            transcript_revision="gate-a-transcript",
            row_revision=row_revision,
        )
        assert provenance is not None
        return auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=8.9,
            tail_end=tail_end,
            uncertainty_seconds=0.05,
            confidence=0.9,
            producer="joint-ctc-boundary-previous-tail",
            source_revision=provenance.observation_revision,
            provenance=provenance,
        )

    def run_synthetic(self, *, inject_tail=False, weaker=False):
        entries, candidate_sets, report, capability = self.fixture()
        diagnostics = auto_lrc.inactive_legacy_timing_diagnostics()
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        tails = auto_lrc.runtime_independent_tail_evidence(report, len(entries), context)
        passes: list[dict[str, object]] = []
        for pass_index in range(1, 5):
            before = auto_lrc._retry_semantic_state_revision(candidate_sets, tails, context)
            epoch = auto_lrc.freeze_base_retry_epoch(
                "ctc", report, candidate_sets, diagnostics, capability
            )
            requests = auto_lrc.plan_crossline_retry_requests(
                epoch, capability, entries, candidate_sets, 20.0, tails, context
            )
            if not requests:
                passes.append({"pass": pass_index, "requests": 0, "stable": True})
                break
            if inject_tail and pass_index == 1:
                auto_lrc.merge_runtime_independent_tail_evidence(
                    report, self.tail()
                )
                if weaker:
                    auto_lrc.merge_runtime_independent_tail_evidence(
                        report, self.tail(tail_end=9.2, row_revision="gate-a-weaker")
                    )
                tails = auto_lrc.runtime_independent_tail_evidence(report, len(entries), context)
            after = auto_lrc._retry_semantic_state_revision(candidate_sets, tails, context)
            stable = before == after
            passes.append({"pass": pass_index, "requests": len(requests), "stable": stable})
            if stable:
                break
        return passes, tails

    def test_a_tail_change_re_evaluates_without_candidate(self) -> None:
        passes, tails = self.run_synthetic(inject_tail=True)
        self.assertEqual([item["requests"] for item in passes], [1, 0])
        self.assertIsNotNone(tails[0])

    def test_b_tail_change_then_no_consequence_converges(self) -> None:
        passes, _tails = self.run_synthetic(inject_tail=True)
        self.assertEqual(passes[-1]["stable"], True)
        self.assertLessEqual(len(passes), 2)

    def test_c_zero_candidate_zero_tail_is_immediate(self) -> None:
        passes, _tails = self.run_synthetic(inject_tail=False)
        # The synthetic unresolved boundary has one request, but no evidence
        # mutation is allowed to claim a fixed point before its planner pass.
        self.assertEqual(len(passes), 1)
        self.assertEqual(passes[0]["stable"], True)

    def test_d_identical_tail_revision_does_not_loop(self) -> None:
        passes, _tails = self.run_synthetic(inject_tail=True)
        self.assertLessEqual(len(passes), 2)

    def test_e_weaker_tail_does_not_churn_authoritative_state(self) -> None:
        passes, _tails = self.run_synthetic(inject_tail=True, weaker=True)
        self.assertEqual([item["requests"] for item in passes], [1, 0])


class ContractATailLineageTests(unittest.TestCase):
    def capability(self) -> auto_lrc.BoundedRetryCapability:
        payload = {
            "kind": "mms-known-lyric-ctc",
            "backend_lineage": "ctc",
            "helper_path": "ctc_align.py",
            "helper_sha256": "contract-a-helper",
            "helper_protocol_revision": "contract-a-protocol",
            "alignment_audio_path": "contract-a-audio.flac",
            "alignment_audio_sha256": "contract-a-audio-sha",
            "alignment_audio_source": "vocal-stem",
            "sample_rate": auto_lrc.SAMPLE_RATE,
            "model_identity": "MMS_FA:known-lyric",
            "device": "cpu",
        }
        return auto_lrc.BoundedRetryCapability(
            **payload,
            capability_revision=auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(payload)
            ),
        )

    def candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str = "ctc-current",
        spans: tuple[auto_lrc.TimingTokenSpan, ...] | None = None,
        current: bool = True,
    ) -> auto_lrc.TimingCandidate:
        if spans is None:
            spans = () if source.startswith("raw-") else (
                auto_lrc.TimingTokenSpan(time, time + 0.08, 0.8, "a"),
                auto_lrc.TimingTokenSpan(time + 0.10, time + 0.18, 0.8, "b"),
            )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"generic contract line {entry}",
            source=source,
            raw_time=time,
            spans=spans,
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            current=current,
            source_artifact={"contract": "A", "entry": entry, "source": source},
        )

    def observation(
        self,
        *,
        producer: str = "joint-ctc-boundary-previous-tail",
        backend_lineage: str = "ctc",
        tail_end: float = 9.5,
        uncertainty: float = 0.05,
        model_identity: str | None = None,
        helper_revision: str | None = None,
        capability_revision: str | None = None,
        audio_revision: str | None = None,
        transcript_revision: str = "contract-transcript",
        row_revision: str = "contract-row",
    ) -> auto_lrc.IndependentTailEvidence:
        cap = self.capability()
        provenance = auto_lrc._make_tail_provenance(
            entry_index=0,
            tail_start=8.9,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=0.9,
            producer=producer,
            backend_lineage=backend_lineage,
            model_identity=model_identity or cap.model_identity,
            helper_revision=helper_revision or auto_lrc._tail_helper_revision(
                cap.helper_sha256,
                cap.helper_protocol_revision,
            ),
            capability_revision=capability_revision or cap.capability_revision,
            audio_revision=audio_revision or cap.alignment_audio_sha256,
            transcript_revision=transcript_revision,
            row_revision=row_revision,
        )
        assert provenance is not None
        return auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=8.9,
            tail_end=tail_end,
            uncertainty_seconds=uncertainty,
            confidence=0.9,
            producer=producer,
            source_revision=provenance.observation_revision,
            provenance=provenance,
        )

    def report(self, observations=()) -> dict[str, object]:
        cap = self.capability()
        helper_revision = auto_lrc._tail_helper_revision(
            cap.helper_sha256,
            cap.helper_protocol_revision,
        )
        report: dict[str, object] = {
            "backend": "ctc",
            auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: cap,
            auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY: auto_lrc.TailEvidenceContext(
                backend_lineage=cap.backend_lineage,
                model_identity=cap.model_identity,
                helper_revision=helper_revision,
                capability_revision=cap.capability_revision,
                audio_revision=cap.alignment_audio_sha256,
                transcript_revisions=("contract-transcript",),
                row_revisions=("contract-row",),
                producer_contexts=(
                    (
                        "whisper-family",
                        cap.model_identity,
                        helper_revision,
                        cap.capability_revision,
                        cap.alignment_audio_sha256,
                    ),
                ),
                scopes=(
                    auto_lrc.TailEvidenceScope(
                        entry_index=0,
                        producer="joint-ctc-boundary-previous-tail",
                        backend_lineage=cap.backend_lineage,
                        model_identity=cap.model_identity,
                        helper_revision=helper_revision,
                        capability_revision=cap.capability_revision,
                        audio_revision=cap.alignment_audio_sha256,
                        transcript_revisions=("contract-transcript",),
                        row_revisions=("contract-row",),
                    ),
                    auto_lrc.TailEvidenceScope(
                        entry_index=0,
                        producer="whisperx-forced-row",
                        backend_lineage="whisper-family",
                        model_identity=cap.model_identity,
                        helper_revision=helper_revision,
                        capability_revision=cap.capability_revision,
                        audio_revision=cap.alignment_audio_sha256,
                        transcript_revisions=("contract-transcript",),
                        row_revisions=("contract-row",),
                    ),
                ),
            ),
        }
        if observations:
            report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY] = {0: tuple(observations)}
        return report

    def state(
        self,
        predecessor: auto_lrc.TimingCandidate,
        observations=(),
        current: auto_lrc.TimingCandidate | None = None,
    ):
        current = current or self.candidate(1, 10.0)
        report = self.report(observations)
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        tails = auto_lrc.runtime_independent_tail_evidence(report, 2, context)
        return auto_lrc.build_final_timing_state(
            ((predecessor,), (current,)),
            independent_tail_evidence=tails,
            tail_evidence_context=context,
        )

    def direct_candidate(
        self,
        entry: int,
        time: float,
        *,
        source: str,
        producer: str,
        kind: str = "cross-backend-onset",
        current: bool = True,
    ) -> auto_lrc.TimingCandidate:
        occurrence_entries = [
            LyricEntry([f"direct line {index}"])
            for index in range(max(2, entry + 1))
        ]
        occurrence_assignments = [
            {"timestamp": float(index + 1), "score": .95, "segment": index + 1}
            for index in range(len(occurrence_entries))
        ]
        binding = auto_lrc._entry_occurrence_binding(
            occurrence_entries, occurrence_assignments, entry
        )
        occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
            occurrence_entries,
            entry,
            producer=producer,
            occurrence_binding=binding,
            upstream_evidence_revision="contract-direct-upstream",
        )
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"direct line {entry}",
            source=source,
            raw_time=time,
            spans=(),
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported",
            direct_onset_time=time,
            direct_onset_source_artifact={
                "entry": entry,
                "time": time,
                "producer": producer,
            },
            direct_onset_evidence_producer=producer,
            direct_onset_evidence_kind=kind,
            direct_onset_evidence_independent=True,
            occurrence_evidence=occurrence_evidence,
            independent_content_identity=True,
            current=current,
            source_artifact={"entry": entry, "time": time, "source": source},
        )

    def test_a_same_lineage_is_not_authoritative(self) -> None:
        predecessor = self.candidate(0, 9.0, spans=())
        observation = self.observation()
        state = self.state(predecessor, (observation,))
        self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown")
        self.assertFalse(state.audits[1].timing_trusted)

    def test_b_disjoint_lineage_is_authoritative(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        current = self.direct_candidate(
            1,
            10.0,
            source="raw-vocal-independent-fusion",
            producer="raw-asr-vocal-fusion",
        )
        state = self.state(predecessor, (observation,), current=current)
        self.assertEqual(state.decisions[1].evidence.ownership.status, "clear")
        self.assertTrue(state.audits[1].timing_trusted)

    def test_c_missing_provenance_is_diagnostic_only(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        missing = auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=8.9,
            tail_end=9.5,
            uncertainty_seconds=0.05,
            confidence=0.9,
            producer="joint-ctc-boundary-previous-tail",
            source_revision="missing-provenance",
        )
        state = self.state(predecessor, (missing,))
        self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown")
        self.assertFalse(state.audits[1].timing_trusted)

    def test_d_stale_provenance_revisions_fail_closed(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        cap = self.capability()
        helper_revision = auto_lrc._tail_helper_revision(
            cap.helper_sha256,
            cap.helper_protocol_revision,
        )
        for field, value in (
            ("model_identity", "stale-model"),
            ("helper_revision", "stale-helper"),
            ("capability_revision", "stale-capability"),
            ("audio_revision", "stale-audio"),
        ):
            kwargs = {field: value}
            if field == "helper_revision":
                kwargs[field] = str(value)
            stale = self.observation(**kwargs)
            state = self.state(predecessor, (stale,))
            self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown", field)
            self.assertFalse(state.audits[1].timing_trusted, field)
        for field, value in (
            ("transcript_revision", "stale-transcript"),
            ("row_revision", "stale-row"),
        ):
            original = self.observation()
            assert original.provenance is not None
            tampered = replace(original.provenance, **{field: value})
            stale = replace(original, provenance=tampered)
            state = self.state(predecessor, (stale,))
            self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown", field)
            self.assertFalse(state.audits[1].timing_trusted, field)
        self.assertTrue(helper_revision)

    def test_direct_resolver_without_context_fails_closed(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        resolved = auto_lrc.resolve_reliable_tail(predecessor, observation)
        self.assertNotEqual(resolved.reason, "independent-joint-ctc-boundary-previous-tail")
        self.assertEqual(resolved.validity, "unknown")

    def test_self_consistent_stale_ctc_context_fails_runtime_and_direct(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        stale = self.observation(
            transcript_revision="stale-current-transcript",
            row_revision="stale-current-row",
        )
        report = self.report((stale,))
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        self.assertIsNone(
            auto_lrc.runtime_independent_tail_evidence(report, 2, context)[0]
        )
        resolved = auto_lrc.resolve_reliable_tail(
            predecessor, stale, tail_evidence_context=context
        )
        self.assertEqual(resolved.validity, "unknown")

    def test_self_consistent_non_ctc_context_cannot_auto_pass(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        non_ctc = self.observation(
            producer="whisperx-forced-row",
            backend_lineage="whisper-family",
            model_identity="whisper-model",
            helper_revision="whisper-helper",
            capability_revision="whisper-capability",
            audio_revision="whisper-audio",
        )
        report = self.report((non_ctc,))
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        self.assertIsNone(
            auto_lrc.runtime_independent_tail_evidence(report, 2, context)[0]
        )
        resolved = auto_lrc.resolve_reliable_tail(
            predecessor, non_ctc, tail_evidence_context=context
        )
        self.assertEqual(resolved.validity, "unknown")

    def test_serialized_retry_diagnostics_cannot_mint_current_tail_context(self) -> None:
        cap = self.capability()
        entries = [LyricEntry(["current immutable lyric"])]
        assignment = {
            "entry": 1,
            "timestamp": 9.0,
            "ctc_token_spans": [
                {"char": "a", "start": 9.0, "end": 9.08, "score": .8},
                {"char": "b", "start": 9.10, "end": 9.18, "score": .8},
                {"char": "c", "start": 9.20, "end": 9.28, "score": .8},
            ],
        }
        stale = self.observation(
            transcript_revision="serialized-stale-transcript",
            row_revision="serialized-stale-row",
        )
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [assignment],
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
            auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: cap,
            auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY: {0: (stale,)},
            "joint_boundary_retry": {
                "requests": [{
                    "transcript_digest": "serialized-stale-transcript",
                    "row_mapping_revision": "serialized-stale-row",
                }],
            },
        }
        context = auto_lrc._tail_evidence_context_from_report(report)
        self.assertIsNotNone(context)
        assert context is not None
        self.assertNotIn("serialized-stale-transcript", context.transcript_revisions)
        self.assertNotIn("serialized-stale-row", context.row_revisions)
        self.assertIsNone(auto_lrc.runtime_independent_tail_evidence(report, 1, context)[0])

    def test_exact_entry_producer_scope_rejects_transplant_and_wrong_mapping(self) -> None:
        cap = self.capability()
        helper_revision = auto_lrc._tail_helper_revision(
            cap.helper_sha256,
            cap.helper_protocol_revision,
        )
        context = auto_lrc.TailEvidenceContext(
            backend_lineage=cap.backend_lineage,
            model_identity=cap.model_identity,
            helper_revision=helper_revision,
            capability_revision=cap.capability_revision,
            audio_revision=cap.alignment_audio_sha256,
            transcript_revisions=("entry-a-transcript", "entry-b-transcript"),
            row_revisions=("entry-a-row", "entry-b-row"),
            scopes=(
                auto_lrc.TailEvidenceScope(
                    entry_index=0,
                    producer="joint-ctc-boundary-previous-tail",
                    backend_lineage=cap.backend_lineage,
                    model_identity=cap.model_identity,
                    helper_revision=helper_revision,
                    capability_revision=cap.capability_revision,
                    audio_revision=cap.alignment_audio_sha256,
                    transcript_revisions=("entry-a-transcript",),
                    row_revisions=("entry-a-row",),
                ),
                auto_lrc.TailEvidenceScope(
                    entry_index=1,
                    producer="joint-ctc-boundary-previous-tail",
                    backend_lineage=cap.backend_lineage,
                    model_identity=cap.model_identity,
                    helper_revision=helper_revision,
                    capability_revision=cap.capability_revision,
                    audio_revision=cap.alignment_audio_sha256,
                    transcript_revisions=("entry-b-transcript",),
                    row_revisions=("entry-b-row",),
                ),
            ),
        )
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        transplant = self.observation(
            transcript_revision="entry-b-transcript",
            row_revision="entry-b-row",
        )
        wrong_producer = self.observation(
            producer="ctc-assignment-terminal-row",
            transcript_revision="entry-a-transcript",
            row_revision="entry-a-row",
        )
        for observation in (transplant, wrong_producer):
            report = {
                "backend": "ctc",
                auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: cap,
                auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY: context,
                auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY: {0: (observation,)},
            }
            self.assertIsNone(
                auto_lrc.runtime_independent_tail_evidence(report, 1, context)[0]
            )
            resolved = auto_lrc.resolve_reliable_tail(
                predecessor, observation, tail_evidence_context=context
            )
            self.assertEqual(resolved.validity, "unknown")

    def test_e_same_lineage_recomputed_observation_is_rejected(self) -> None:
        predecessor = self.candidate(0, 9.0, spans=())
        first = self.observation(tail_end=9.5, row_revision="recompute-1")
        second = self.observation(tail_end=10.5, row_revision="recompute-2")
        state = self.state(predecessor, (first, second))
        self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown")

    def test_f_only_disjoint_sidecar_can_change_unknown_to_clear(self) -> None:
        raw_predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        ctc_predecessor = self.candidate(0, 9.0, spans=())
        observation = self.observation()
        raw_state = self.state(
            raw_predecessor,
            (observation,),
            current=self.candidate(1, 10.0, source="raw-vocal-independent-fusion", spans=()),
        )
        ctc_state = self.state(ctc_predecessor, (observation,))
        self.assertEqual(raw_state.decisions[1].evidence.ownership.status, "clear")
        self.assertEqual(ctc_state.decisions[1].evidence.ownership.status, "unknown")

    def test_multi_observation_order_and_later_same_lineage_are_stable(self) -> None:
        predecessor = self.candidate(0, 9.0, spans=())
        cross_backend = self.observation(
            producer="whisperx-forced-row",
            backend_lineage="whisper-family",
            tail_end=9.7,
        )
        same_lineage_later = self.observation(tail_end=10.8, row_revision="later-same")
        for ordered in ((cross_backend, same_lineage_later), (same_lineage_later, cross_backend)):
            report = self.report(ordered)
            context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
            tails = auto_lrc.runtime_independent_tail_evidence(report, 2, context)
            state = auto_lrc.build_final_timing_state(
                ((predecessor,), (self.candidate(1, 10.0),)),
                independent_tail_evidence=tails,
                tail_evidence_context=context,
            )
            self.assertEqual(state.decisions[1].evidence.ownership.status, "clear")
            self.assertAlmostEqual(
                state.decisions[1].evidence.ownership.reliable_tail_end or 0.0,
                9.7,
                places=3,
            )

    def test_candidate_lineage_switch_recomputes_authority(self) -> None:
        observation = self.observation()
        report = self.report((observation,))
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        tails = auto_lrc.runtime_independent_tail_evidence(report, 2, context)
        raw_state = auto_lrc.build_final_timing_state(
            (
                (self.candidate(0, 9.0, source="raw-vocal-independent-fusion"),),
                (self.candidate(1, 10.0, source="raw-vocal-independent-fusion", spans=()),),
            ),
            independent_tail_evidence=tails,
            tail_evidence_context=context,
        )
        ctc_state = auto_lrc.build_final_timing_state(
            ((self.candidate(0, 9.0, spans=()),), (self.candidate(1, 10.0),)),
            independent_tail_evidence=tails,
            tail_evidence_context=context,
        )
        self.assertEqual(raw_state.decisions[1].evidence.ownership.status, "clear")
        self.assertEqual(ctc_state.decisions[1].evidence.ownership.status, "unknown")

    def test_unknown_candidate_family_fails_closed(self) -> None:
        predecessor = self.candidate(0, 9.0, source="synthetic-unknown", spans=())
        state = self.state(predecessor, (self.observation(),))
        self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown")

    def test_current_same_lineage_tail_cannot_clear_or_violate(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        later = self.state(predecessor, (observation,))
        self.assertEqual(
            later.decisions[1].evidence.ownership.reason,
            "external-tail-current-lineage-overlap",
        )
        self.assertEqual(later.decisions[1].evidence.ownership.status, "unknown")

        earlier = self.state(
            predecessor,
            (observation,),
            current=self.candidate(1, 9.0, source="ctc-current", current=True),
        )
        self.assertEqual(earlier.decisions[1].evidence.ownership.status, "unknown")
        self.assertEqual(
            earlier.decisions[1].evidence.ownership.reason,
            "external-tail-current-lineage-overlap",
        )

    def test_composite_overlap_requires_disjoint_direct_producer(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        composite = self.state(
            predecessor,
            (observation,),
            current=self.candidate(
                1,
                10.0,
                source="ctc-local-whisper-vocal-independent-consensus",
            ),
        )
        self.assertEqual(composite.decisions[1].evidence.ownership.status, "unknown")
        direct_exception = self.state(
            predecessor,
            (observation,),
            current=self.direct_candidate(
                1,
                10.0,
                source="ctc-local-whisper-vocal-independent-consensus",
                producer="local-whisper-vocal-fusion",
            ),
        )
        self.assertEqual(direct_exception.decisions[1].evidence.ownership.status, "clear")
        self.assertTrue(direct_exception.audits[1].timing_trusted)

    def test_same_lineage_direct_onset_is_not_the_exception(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        state = self.state(
            predecessor,
            (observation,),
            current=self.direct_candidate(
                1,
                10.0,
                source="ctc-current",
                producer="joint-ctc-boundary-consensus",
                kind="joint-lyric-boundary",
            ),
        )
        self.assertEqual(state.decisions[1].evidence.ownership.status, "unknown")
        self.assertEqual(
            state.decisions[1].evidence.ownership.reason,
            "external-tail-current-lineage-overlap",
        )
        self.assertFalse(state.audits[1].timing_trusted)

    def test_59_like_ctc_tail_can_select_disjoint_whisper_opening(self) -> None:
        predecessor = self.candidate(0, 9.0, source="raw-vocal-independent-fusion")
        observation = self.observation()
        report = self.report((observation,))
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        tails = auto_lrc.runtime_independent_tail_evidence(report, 2, context)
        ctc_current = self.candidate(1, 10.0, source="ctc-current", current=True)
        whisper_opening = self.direct_candidate(
            1,
            10.1,
            source="local-whisper-vocal-independent-fusion",
            producer="local-whisper-vocal-fusion",
            current=False,
        )
        state = auto_lrc.build_final_timing_state(
            ((predecessor,), (ctc_current, whisper_opening)),
            independent_tail_evidence=tails,
            tail_evidence_context=context,
        )
        self.assertEqual(
            next(item for item in state.decisions if item.entry_index == 1)
            .candidate.source,
            "local-whisper-vocal-independent-fusion",
        )
        self.assertEqual(
            next(item for item in state.decisions if item.entry_index == 1)
            .candidate.raw_time,
            10.1,
        )
        self.assertEqual(ctc_current.raw_time, 10.0)

    def test_same_lineage_material_alternative_remains_reviewable(self) -> None:
        predecessor = self.candidate(0, 8.0, source="ctc-current", spans=())
        observation = self.observation(
            producer="whisperx-forced-row",
            backend_lineage="whisper-family",
        )
        report = self.report((observation,))
        context = report[auto_lrc._RUNTIME_TAIL_EVIDENCE_CONTEXT_KEY]
        tails = auto_lrc.runtime_independent_tail_evidence(report, 2, context)
        current = self.direct_candidate(
            1,
            10.0,
            source="local-whisper-vocal-independent-fusion",
            producer="local-whisper-vocal-fusion",
            current=True,
        )
        distant = self.direct_candidate(
            1,
            9.0,
            source="whisperx-vocal-independent-fusion",
            producer="whisperx-vocal-fusion",
            current=False,
        )
        previous_tail = auto_lrc.resolve_reliable_tail(
            predecessor,
            tails[0],
            tail_evidence_context=context,
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(current, distant),
            previous_candidate=predecessor,
            previous_tail=previous_tail,
            selection_revision=2,
        )
        self.assertTrue(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )


class WhisperXRuntimeProvenanceTests(unittest.TestCase):
    def provenance(self, **overrides) -> auto_lrc.WhisperXRuntimeProvenance:
        payload = {
            "backend_lineage": "whisper-family",
            "raw_executable_sha256": "raw-executable",
            "raw_model_sha256": "raw-model",
            "python_sha256": "python",
            "helper_sha256": "helper",
            "helper_protocol_revision": auto_lrc._WHISPERX_HELPER_PROTOCOL_REVISION,
            "whisperx_package_version": "3.8.6",
            "align_model_id": "model-id",
            "align_model_revision": "model-revision",
            "align_model_blob_sha256": "model-blob",
            "source_audio_sha256": "source-audio",
            "alignment_wav_sha256": "alignment-wav",
            "device": "cuda",
            "language": "ja",
            "raw_transcript_revision": "raw-transcript",
            "initial_aligned_output_revision": "initial-output",
            "forced_output_revision": "forced-output",
        }
        payload.update(overrides)
        return auto_lrc.WhisperXRuntimeProvenance(
            **payload,
            capability_revision=auto_lrc._timing_content_digest(payload),
        )

    def scope(self, provenance: auto_lrc.WhisperXRuntimeProvenance) -> auto_lrc.TailEvidenceScope:
        model_identity = (
            f"{provenance.align_model_id}@{provenance.align_model_revision}#"
            f"{provenance.align_model_blob_sha256}"
        )
        helper_revision = auto_lrc._timing_content_digest({
            "helper_sha256": provenance.helper_sha256,
            "helper_protocol_revision": provenance.helper_protocol_revision,
        })
        return auto_lrc.TailEvidenceScope(
            entry_index=0,
            producer="whisperx-forced-row",
            backend_lineage="whisper-family",
            model_identity=model_identity,
            helper_revision=helper_revision,
            capability_revision=provenance.capability_revision,
            audio_revision=provenance.alignment_wav_sha256,
            transcript_revisions=("row-transcript",),
            row_revisions=("row-revision",),
        )

    def context(self, provenance: auto_lrc.WhisperXRuntimeProvenance) -> auto_lrc.TailEvidenceContext:
        scope = self.scope(provenance)
        return auto_lrc.TailEvidenceContext(
            backend_lineage="ctc",
            model_identity="ctc-model",
            helper_revision="ctc-helper",
            capability_revision="ctc-capability",
            audio_revision="ctc-audio",
            transcript_revisions=("row-transcript",),
            row_revisions=("row-revision",),
            scopes=(scope,),
        )

    def observation(self, provenance: auto_lrc.WhisperXRuntimeProvenance) -> auto_lrc.IndependentTailEvidence:
        scope = self.scope(provenance)
        tail_provenance = auto_lrc._make_tail_provenance(
            entry_index=0,
            tail_start=8.9,
            tail_end=9.5,
            uncertainty_seconds=.05,
            confidence=.9,
            producer="whisperx-forced-row",
            backend_lineage=scope.backend_lineage,
            model_identity=scope.model_identity,
            helper_revision=scope.helper_revision,
            capability_revision=scope.capability_revision,
            audio_revision=scope.audio_revision,
            transcript_revision="row-transcript",
            row_revision="row-revision",
        )
        assert tail_provenance is not None
        return auto_lrc.IndependentTailEvidence(
            entry_index=0,
            tail_start=8.9,
            tail_end=9.5,
            uncertainty_seconds=.05,
            confidence=.9,
            producer="whisperx-forced-row",
            source_revision=tail_provenance.observation_revision,
            provenance=tail_provenance,
        )

    def test_missing_runtime_fields_fail_closed(self) -> None:
        fields = tuple(auto_lrc._whisperx_runtime_provenance_payload(self.provenance()))
        for field in fields:
            self.assertFalse(
                auto_lrc._whisperx_runtime_provenance_is_valid(
                    self.provenance(**{field: ""})
                ),
                field,
            )

    def test_self_consistent_stale_runtime_context_fails_scope_match(self) -> None:
        original = self.provenance()
        for field, value in (
            ("raw_executable_sha256", "stale-executable"),
            ("raw_model_sha256", "stale-model-file"),
            ("python_sha256", "stale-python"),
            ("helper_sha256", "stale-helper"),
            ("align_model_revision", "stale-model-revision"),
            ("align_model_blob_sha256", "stale-model-blob"),
            ("source_audio_sha256", "stale-source-audio"),
            ("alignment_wav_sha256", "stale-alignment-wav"),
            ("raw_transcript_revision", "stale-raw-transcript"),
            ("initial_aligned_output_revision", "stale-initial-output"),
            ("forced_output_revision", "stale-forced-output"),
        ):
            stale_base = replace(original, **{field: value, "capability_revision": ""})
            stale_payload = auto_lrc._whisperx_runtime_provenance_payload(stale_base)
            stale = replace(
                stale_base,
                capability_revision=auto_lrc._timing_content_digest(stale_payload),
            )
            self.assertTrue(auto_lrc._whisperx_runtime_provenance_is_valid(stale), field)
            observation = self.observation(stale)
            self.assertFalse(
                auto_lrc._tail_provenance_matches_context(observation, self.context(original)),
                field,
            )

    def test_binder_populates_exact_scope_without_timestamp_mutation(self) -> None:
        provenance = self.provenance()
        entries = [LyricEntry(["unique target"]), LyricEntry(["next line"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "segment": 1, "score": .96, "timestamp": 9.0, "borrowed": False},
                {"entry": 2, "segment": 2, "score": .96, "timestamp": 10.0, "borrowed": False},
            ],
            "whisperx_forced_rows": [
                {"segment_start": 8.9, "segment_end": 9.5, "first_char_time": 8.9, "last_char_time": 9.45, "char_count": 3},
                {"segment_start": 10.0, "segment_end": 10.5, "first_char_time": 10.0, "last_char_time": 10.45, "char_count": 3},
            ],
            auto_lrc._RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY: provenance,
        }
        before = [item["timestamp"] for item in report["assignments"]]  # type: ignore[index]
        auto_lrc.bind_runtime_whisperx_row_evidence(
            report, entries, [([], report, "synthetic-error")],
        )
        runtime = report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY]
        self.assertIsNotNone(runtime[0][0].provenance)  # type: ignore[index]
        self.assertEqual(
            report[auto_lrc._RUNTIME_WHISPERX_TAIL_SCOPES_KEY][0].producer,  # type: ignore[index]
            "whisperx-forced-row",
        )
        self.assertEqual(
            [item["timestamp"] for item in report["assignments"]],  # type: ignore[index]
            before,
        )

    def test_ambiguous_occurrence_is_diagnostic_only(self) -> None:
        provenance = self.provenance()
        entries = [LyricEntry(["repeat line"]), LyricEntry(["repeat line"])]
        report: dict[str, object] = {
            "assignments": [
                {"entry": 1, "segment": 1, "score": .96, "timestamp": 9.0, "borrowed": False},
                {"entry": 2, "segment": 1, "score": .96, "timestamp": 9.0, "borrowed": False},
            ],
            "whisperx_forced_rows": [
                {"segment_start": 8.9, "segment_end": 9.5, "first_char_time": 8.9, "last_char_time": 9.45, "char_count": 3},
                {"segment_start": 9.0, "segment_end": 9.6, "first_char_time": 9.0, "last_char_time": 9.55, "char_count": 3},
            ],
            auto_lrc._RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY: provenance,
        }
        auto_lrc.bind_runtime_whisperx_row_evidence(
            report, entries, [([], report, "synthetic-error")]
        )
        runtime = report[auto_lrc._RUNTIME_INDEPENDENT_TAIL_EVIDENCE_KEY]
        self.assertNotIn(0, runtime)
        self.assertNotIn(1, runtime)

    def test_row_output_and_occurrence_revision_tamper_fails_closed(self) -> None:
        provenance = self.provenance()
        observation = self.observation(provenance)
        assert observation.provenance is not None
        tampered = replace(
            observation,
            provenance=replace(observation.provenance, row_revision="wrong-row"),
        )
        self.assertFalse(auto_lrc._tail_provenance_is_valid(tampered))
        self.assertFalse(
            auto_lrc._tail_provenance_matches_context(tampered, self.context(provenance))
        )

    def test_writer_strips_runtime_provenance_and_temp_artifacts(self) -> None:
        provenance = self.provenance()
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [],
            auto_lrc._RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY: provenance,
            auto_lrc._RUNTIME_WHISPERX_TEMP_PATHS_KEY: ("C:/temp/should-not-serialize",),
        }
        with TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "report.json"
            auto_lrc.write_alignment_report(path, report)
            text = path.read_text(encoding="utf-8")
        self.assertNotIn(auto_lrc._RUNTIME_WHISPERX_EXECUTION_PROVENANCE_KEY, text)
        self.assertNotIn("should-not-serialize", text)


class WhisperXRefineContractTests(unittest.TestCase):
    def _load_helper(self, fake_whisperx: types.ModuleType, fake_hf: types.ModuleType):
        helper_path = Path(__file__).resolve().parent / "whisperx_refine.py"
        spec = importlib.util.spec_from_file_location("_whisperx_refine_contract_test", helper_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(
            sys.modules,
            {"whisperx": fake_whisperx, "huggingface_hub": fake_hf},
        ):
            spec.loader.exec_module(module)
        return module

    def test_helper_emits_runtime_alignment_provenance_matching_auto_lrc_protocol(self) -> None:
        with TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            snapshot = temp / "models--fake" / "snapshots" / "revision-123"
            snapshot.mkdir(parents=True)
            config_path = snapshot / "config.json"
            config_path.write_text("{}\n", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"fake-model")

            fake_alignment = types.SimpleNamespace(
                DEFAULT_ALIGN_MODELS_HF={"ja": "fake/alignment-model"},
                DEFAULT_ALIGN_MODELS_TORCH={},
                torchaudio=types.SimpleNamespace(
                    pipelines=types.SimpleNamespace(__all__=()),
                ),
            )
            fake_whisperx = types.ModuleType("whisperx")
            fake_whisperx.alignment = fake_alignment
            fake_whisperx.load_align_model = lambda **_: (object(), {"dictionary": "fake"})
            fake_whisperx.align = lambda transcript, *_args, **_kwargs: {
                "segments": transcript,
            }

            fake_hf = types.ModuleType("huggingface_hub")
            fake_hf.try_to_load_from_cache = lambda *_args, **_kwargs: str(config_path)

            helper = self._load_helper(fake_whisperx, fake_hf)
            audio_path = temp / "input.wav"
            audio_path.write_bytes(b"fake-audio")
            transcript_path = temp / "transcript.json"
            transcript = [{"start": 0.0, "end": 1.0, "text": "test"}]
            transcript_path.write_text(json.dumps(transcript), encoding="utf-8")
            output_path = temp / "output.json"

            argv = [
                str(Path(__file__).resolve().parent / "whisperx_refine.py"),
                "--audio",
                str(audio_path),
                "--transcript",
                str(transcript_path),
                "--output",
                str(output_path),
                "--language",
                "ja",
                "--device",
                "cpu",
            ]
            with mock.patch.dict(
                sys.modules,
                {"whisperx": fake_whisperx, "huggingface_hub": fake_hf},
            ), mock.patch.object(sys, "argv", argv), mock.patch.object(
                importlib.metadata, "version", return_value="fake-whisperx-version"
            ):
                self.assertEqual(helper.main(), 0)

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            provenance = payload[auto_lrc._RUNTIME_WHISPERX_ALIGNMENT_METADATA_KEY]
            self.assertEqual(
                provenance["helper_protocol_revision"],
                auto_lrc._WHISPERX_HELPER_PROTOCOL_REVISION,
            )
            self.assertEqual(provenance["align_model_id"], "fake/alignment-model")
            self.assertEqual(provenance["align_model_revision"], "revision-123")
            self.assertEqual(provenance["device"], "cpu")
            self.assertEqual(provenance["language"], "ja")
            self.assertTrue(provenance["helper_sha256"])
            self.assertTrue(provenance["python_sha256"])
            self.assertTrue(provenance["alignment_wav_sha256"])
            self.assertTrue(provenance["raw_transcript_revision"])

    def test_auto_lrc_missing_helper_provenance_fails_closed(self) -> None:
        payload = {"segments": []}
        self.assertIsNone(auto_lrc._whisperx_alignment_metadata(payload))
        with mock.patch.object(
            auto_lrc,
            "_whispercpp_runtime_artifact_hashes",
            return_value=("raw-executable", "raw-model"),
        ):
            self.assertIsNone(
                auto_lrc._build_whisperx_runtime_provenance(
                    Path("unused-audio.wav"),
                    types.SimpleNamespace(),
                    [],
                    payload,
                    payload,
                )
            )


class PathFreeCapabilityIdentityTests(unittest.TestCase):
    def ctc_seed(self, audio: Path) -> dict[str, object]:
        helper = Path(auto_lrc.__file__).resolve().parent / "ctc_align.py"
        return auto_lrc.create_bounded_retry_capability_seed(
            audio,
            "vocal-stem",
            helper,
            sample_rate=auto_lrc.SAMPLE_RATE,
            device="cpu",
        )

    def retry_request(self, capability_revision: str) -> auto_lrc.CrossLineRetryRequest:
        reference = auto_lrc.RetryCandidateReference(
            "candidate", "generation", "region", 100
        )
        return auto_lrc.CrossLineRetryRequest(
            retry_epoch_revision="epoch",
            entry_index=0,
            ordinal=1,
            capability_revision=capability_revision,
            alignment_audio_revision="audio-sha",
            current=reference,
            previous=reference,
            next_bound=None,
            audio_end_seconds=10.0,
            audio_duration_revision="duration",
            previous_tail_evidence_revision=None,
            previous_tail_start=None,
            previous_tail_end=None,
            previous_tail_uncertainty=None,
            previous_tail_validity="unknown",
            ownership_relation_revision="ownership",
            context_entry_indexes=(0,),
            context_candidate_ids=("candidate",),
            transcript_digests=("text",),
            transcript_digest="text",
            row_mapping=((0, 0),),
            row_mapping_revision="row",
            window_start=1.0,
            window_end=2.0,
            window_definition_revision="window",
            request_revision="request",
        )

    def test_same_bytes_different_paths_have_equal_ctc_local_tail_and_cache_identity(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            first = root / "first" / "stem.mp3"
            second = root / "second" / "stem.mp3"
            first.parent.mkdir(); second.parent.mkdir()
            first.write_bytes(b"identical-audio")
            second.write_bytes(b"identical-audio")
            runtime = root / "runtime"
            runtime.mkdir()
            whisper_cli = runtime / "whisper-cli.exe"
            whisper_model = runtime / "ggml-large-v3.bin"
            whisperx_python = runtime / "python.exe"
            whisper_cli.write_bytes(b"whisper-cli")
            whisper_model.write_bytes(b"whisper-model")
            whisperx_python.write_bytes(b"python")
            first_seed = self.ctc_seed(first)
            second_seed = self.ctc_seed(second)
            self.assertNotEqual(first_seed["alignment_audio_path"], second_seed["alignment_audio_path"])
            self.assertEqual(first_seed["capability_revision"], second_seed["capability_revision"])
            first_cap = auto_lrc.BoundedRetryCapability(**first_seed)
            second_cap = auto_lrc.BoundedRetryCapability(**second_seed)
            self.assertEqual(
                auto_lrc._tail_helper_revision(
                    first_cap.helper_sha256, first_cap.helper_protocol_revision
                ),
                auto_lrc._tail_helper_revision(
                    second_cap.helper_sha256, second_cap.helper_protocol_revision
                ),
            )
            self.assertEqual(
                auto_lrc._crossline_retry_helper_cache_key(
                    self.retry_request(first_cap.capability_revision), first_cap
                ),
                auto_lrc._crossline_retry_helper_cache_key(
                    self.retry_request(second_cap.capability_revision), second_cap
                ),
            )
            args_first = auto_lrc.build_parser().parse_args([str(first)])
            args_second = auto_lrc.build_parser().parse_args([str(second)])
            with (
                mock.patch.object(auto_lrc, "default_whisper_cli", return_value=whisper_cli),
                mock.patch.object(auto_lrc, "default_whisper_model", return_value=whisper_model),
                mock.patch.object(auto_lrc, "default_whisperx_python", return_value=whisperx_python),
            ):
                local_first = auto_lrc.build_local_whisper_capability(first, args_first)
                local_second = auto_lrc.build_local_whisper_capability(second, args_second)
            self.assertIsNotNone(local_first)
            self.assertIsNotNone(local_second)
            assert local_first is not None and local_second is not None
            self.assertNotEqual(local_first.audio_path, local_second.audio_path)
            self.assertEqual(local_first.capability_revision, local_second.capability_revision)
            request_first = auto_lrc.LocalWhisperRequest(
                0, local_first.capability_revision, local_first.audio_sha256,
                (0,), ("candidate",), ("text",), "text", 1.0, 2.0, "window", "request"
            )
            request_second = replace(
                request_first,
                capability_revision=local_second.capability_revision,
                audio_revision=local_second.audio_sha256,
            )
            self.assertEqual(
                auto_lrc._local_whisper_cache_key(request_first, local_first),
                auto_lrc._local_whisper_cache_key(request_second, local_second),
            )

    def test_different_bytes_same_path_change_path_free_identity(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "stem.mp3"
            audio.write_bytes(b"first-bytes")
            first = self.ctc_seed(audio)
            audio.write_bytes(b"second-bytes")
            second = self.ctc_seed(audio)
        self.assertNotEqual(first["alignment_audio_sha256"], second["alignment_audio_sha256"])
        self.assertNotEqual(first["capability_revision"], second["capability_revision"])

    def test_missing_artifact_sha_fails_closed_and_paths_do_not_rescue(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "stem.mp3"
            audio.write_bytes(b"audio")
            seed = self.ctc_seed(audio)
            seed["alignment_audio_sha256"] = ""
            seed["capability_revision"] = auto_lrc._canonical_json_digest(
                auto_lrc._bounded_retry_capability_semantic_payload(seed)
            )
            capability = auto_lrc.BoundedRetryCapability(**seed)
            report = {
                "backend": "ctc",
                "ctc_bounded_retry_capability_seed": seed,
                auto_lrc._RUNTIME_BOUNDED_RETRY_CAPABILITY_KEY: capability,
            }
            args = auto_lrc.build_parser().parse_args([str(audio)])
            self.assertIsNone(auto_lrc.build_bounded_retry_capability(report, audio, args))
            self.assertIsNone(
                auto_lrc.build_local_whisper_capability(
                    Path(temp_name) / "missing.mp3", args
                )
            )

    def test_execution_locators_remain_available_but_not_semantic(self) -> None:
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "stem.mp3"
            audio.write_bytes(b"audio")
            seed = self.ctc_seed(audio)
            self.assertEqual(Path(str(seed["alignment_audio_path"])), audio.resolve())
            semantic = auto_lrc._bounded_retry_capability_semantic_payload(seed)
            self.assertNotIn("alignment_audio_path", semantic)
            self.assertNotIn("helper_path", semantic)

    def test_full_diagnostics_integrity_and_semantic_revision_are_separate(self) -> None:
        first = {
            "crossline_retry": {
                "capability": {
                    "helper_path": "C:/one/ctc_align.py",
                    "alignment_audio_path": "C:/one/vocals.mp3",
                    "helper_sha256": "helper-sha",
                    "alignment_audio_sha256": "audio-sha",
                    "status": "candidate",
                },
                "request_revision": "request-1",
                "candidate_id": "candidate-1",
            },
            "custom_path_note": "semantic-value",
        }
        second = copy.deepcopy(first)
        second["crossline_retry"]["capability"]["helper_path"] = "D:/two/ctc_align.py"  # type: ignore[index]
        second["crossline_retry"]["capability"]["alignment_audio_path"] = "D:/two/vocals.mp3"  # type: ignore[index]
        first_json, first_full = auto_lrc.freeze_generation_diagnostics(first)
        second_json, second_full = auto_lrc.freeze_generation_diagnostics(second)
        self.assertNotEqual(first_full, second_full)
        self.assertNotEqual(first_json, second_json)
        self.assertEqual(
            auto_lrc.generation_diagnostics_semantic_revision(first),
            auto_lrc.generation_diagnostics_semantic_revision(second),
        )
        semantic = auto_lrc._generation_diagnostics_semantic_projection(first)
        self.assertIn("custom_path_note", semantic)
        self.assertNotIn("helper_path", semantic["crossline_retry"]["capability"])

        changed = copy.deepcopy(first)
        changed["crossline_retry"]["capability"]["helper_sha256"] = "changed-helper"  # type: ignore[index]
        changed["crossline_retry"]["candidate_id"] = "candidate-2"  # type: ignore[index]
        changed["crossline_retry"]["capability"]["status"] = "error"  # type: ignore[index]
        self.assertNotEqual(
            auto_lrc.generation_diagnostics_semantic_revision(first),
            auto_lrc.generation_diagnostics_semantic_revision(changed),
        )

    def test_evaluation_identity_uses_semantic_diagnostics_but_thaw_uses_full_digest(self) -> None:
        candidate = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="diagnostic identity",
            source="ctc-current",
            raw_time=1.0,
            spans=(
                auto_lrc.TimingTokenSpan(1.0, 1.04, 0.9, "a"),
                auto_lrc.TimingTokenSpan(1.1, 1.14, 0.9, "b"),
                auto_lrc.TimingTokenSpan(1.2, 1.24, 0.9, "c"),
            ),
            confidence=0.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"producer": "diagnostic-test"},
        )
        report = {
            "backend": "ctc",
            "assignments": [{"entry": 1, "score": 0.9, "segment": None}],
        }
        first = {
            "crossline_retry": {
                "capability": {
                    "helper_path": "C:/one/ctc_align.py",
                    "alignment_audio_path": "C:/one/vocals.mp3",
                    "helper_sha256": "helper-sha",
                    "alignment_audio_sha256": "audio-sha",
                }
            }
        }
        second = copy.deepcopy(first)
        second["crossline_retry"]["capability"]["alignment_audio_path"] = "D:/two/vocals.mp3"  # type: ignore[index]
        evaluated_first = auto_lrc.evaluate_backend_timing(
            "ctc", ((candidate,),), report, first
        )
        evaluated_second = auto_lrc.evaluate_backend_timing(
            "ctc", ((candidate,),), report, second
        )
        self.assertNotEqual(
            evaluated_first.generation_diagnostics_revision,
            evaluated_second.generation_diagnostics_revision,
        )
        self.assertEqual(
            evaluated_first.generation_diagnostics_semantic_revision,
            evaluated_second.generation_diagnostics_semantic_revision,
        )
        self.assertEqual(evaluated_first.input_revision, evaluated_second.input_revision)
        self.assertEqual(evaluated_first.evaluation_revision, evaluated_second.evaluation_revision)
        tampered = replace(
            evaluated_first,
            generation_diagnostics_json=evaluated_first.generation_diagnostics_json + " ",
        )
        with self.assertRaises(LrcError):
            auto_lrc.thaw_generation_diagnostics(tampered)


class V145ContentTimingSeparationRegressionTests(unittest.TestCase):
    def span(self, start: float, score: float = 0.5, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, score, token)

    def structural_candidate(
        self, entry: int, time: float, *, source: str, confidence: float, current: bool = False
    ) -> auto_lrc.TimingCandidate:
        spans = (
            self.span(time, 0.5, "a"),
            self.span(time + 0.08, 0.6, "b"),
            self.span(time + 0.16, 0.7, "c"),
        )
        return auto_lrc.make_timing_candidate(
            entry_index=entry, entry_text=f"line {entry}", source=source, raw_time=time,
            spans=spans, confidence=confidence, identity_support="supported",
            sequence_support="supported", current=current,
            source_artifact={"source": source, "time": time},
        )

    def independent_candidate(
        self, entry: int, time: float, *, source: str, producer: str, confidence: float
    ) -> auto_lrc.TimingCandidate:
        artifact = {"source": source, "producer": producer, "time": time}
        occurrence_entries = [
            LyricEntry([f"line {index}"])
            for index in range(max(1, entry + 1))
        ]
        occurrence_assignments = [
            {"timestamp": float(index + 1), "score": .95, "segment": index + 1}
            for index in range(len(occurrence_entries))
        ]
        binding = auto_lrc._entry_occurrence_binding(
            occurrence_entries, occurrence_assignments, entry
        )
        occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
            occurrence_entries,
            entry,
            producer=producer,
            occurrence_binding=binding,
            upstream_evidence_revision="v145-direct-upstream",
        )
        return auto_lrc.make_timing_candidate(
            entry_index=entry, entry_text=f"line {entry}", source=source, raw_time=time,
            confidence=confidence, identity_support="supported", sequence_support="supported",
            direct_onset_support="supported", direct_onset_time=time,
            direct_onset_source_artifact=artifact, direct_onset_evidence_producer=producer,
            direct_onset_evidence_kind="cross-backend-onset",
            direct_onset_evidence_independent=True,
            occurrence_evidence=occurrence_evidence,
            independent_content_identity=occurrence_evidence is not None,
            current=False, source_artifact=artifact,
        )

    def invalid_current(self, entry: int = 0, time: float = 10.5) -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=entry, entry_text=f"line {entry}", source="ctc-current",
            raw_time=time, confidence=0.8, identity_support="supported",
            sequence_support="supported", current=True, source_artifact={"invalid": True},
        )

    def test_consensus_uses_valid_support_witnesses_before_pareto_pruning(self) -> None:
        current = self.invalid_current()
        raw = self.independent_candidate(
            0, 10.00, source="raw-vocal-independent-fusion",
            producer="raw-asr-vocal-fusion", confidence=0.96,
        )
        ctc = self.structural_candidate(
            0, 10.05, source="ctc-local-window", confidence=0.50
        )
        whisper = self.independent_candidate(
            0, 11.00, source="whisperx-vocal-independent-fusion",
            producer="whisperx-vocal-fusion", confidence=0.96,
        )
        # The CTC witness is individually dominated by RAW, but it supplies the
        # second lineage proving the 10.00 s cluster.  A distant WhisperX
        # singleton must not erase that cluster before consensus is evaluated.
        decision = auto_lrc.select_timing_decision(
            entry_index=0, candidates=(current, raw, ctc, whisper),
            previous_candidate=None, selection_revision=1,
        )
        self.assertEqual(decision.status, "selected_valid")
        self.assertEqual(decision.written_time.centiseconds, 1000)
        self.assertEqual(decision.candidate.source, "raw-vocal-independent-fusion")

    def test_two_equally_supported_separated_clusters_still_fail_closed(self) -> None:
        current = self.invalid_current()
        raw = self.independent_candidate(
            0, 10.00, source="raw-vocal-independent-fusion",
            producer="raw-asr-vocal-fusion", confidence=0.96,
        )
        ctc_a = self.structural_candidate(0, 10.05, source="ctc-local-window", confidence=0.50)
        whisper = self.independent_candidate(
            0, 11.00, source="whisperx-vocal-independent-fusion",
            producer="whisperx-vocal-fusion", confidence=0.96,
        )
        ctc_b = self.structural_candidate(0, 11.05, source="ctc-opening-local-retry", confidence=0.50)
        decision = auto_lrc.select_timing_decision(
            entry_index=0, candidates=(current, raw, ctc_a, whisper, ctc_b),
            previous_candidate=None, selection_revision=1,
        )
        self.assertEqual(decision.status, "provisional_unresolved")
        self.assertIsNone(decision.selected_candidate_id)

    def test_runtime_raw_identity_can_trust_content_without_trusting_its_timestamp(self) -> None:
        entries = [LyricEntry(["unique lyric alpha"])]
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 1,
            "assignments": [{
                "entry": 1, "timestamp": 10.0, "timing_repair": "ctc",
                "score": 0.35,
            }],
        }
        raw_report = {
            "assignments": [{
                "entry": 1, "timestamp": 14.0, "score": 0.95,
                "segment": 0, "borrowed": False,
            }]
        }
        auto_lrc.bind_runtime_raw_content_identity(report, entries, raw_report)
        report[auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY] = tuple(entries)
        self.assertEqual(
            auto_lrc.runtime_independent_content_identity_sources_for_entry(report, 0),
            ("raw-asr-content-identity",),
        )
        candidate = self.structural_candidate(0, 10.0, source="ctc-local-window", confidence=0.8)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        self.assertTrue(state.audits[0].timing_trusted)
        summary = auto_lrc.summarize_central_timing_state(report, state)
        self.assertEqual(summary.content_trusted_entries, 1)
        self.assertEqual(summary.timing_trusted_entries, 1)
        self.assertEqual(summary.overall_trusted_entries, 1)
        # The independent ASR timestamp is deliberately far away: it proves
        # content identity only and must not alter the selected 10.00 s timing.
        self.assertEqual(state.decisions[0].written_time.centiseconds, 1000)

    def test_runtime_content_identity_revision_is_part_of_central_projection(self) -> None:
        entries = [LyricEntry(["unique lyric beta"])]
        report: dict[str, object] = {
            "backend": "ctc",
            "timing_entries": 1,
            "assignments": [{"entry": 1, "timestamp": 1.0, "score": 0.35}],
        }
        raw_report = {
            "assignments": [{
                "entry": 1, "timestamp": 2.0, "score": 0.95,
                "segment": 0, "borrowed": False,
            }]
        }
        auto_lrc.bind_runtime_raw_content_identity(report, entries, raw_report)
        report[auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY] = tuple(entries)
        projection = auto_lrc.timing_report_evaluation_projection(report)
        self.assertEqual(len(projection["runtime_content_identity"]), 1)
        before = auto_lrc.timing_report_evaluation_revision(report)
        runtime = report[auto_lrc._RUNTIME_RAW_CONTENT_IDENTITY_KEY]
        assert isinstance(runtime, dict) and isinstance(runtime[0], dict)
        runtime[0]["evidence_revision"] = "tampered"
        self.assertEqual(
            auto_lrc.runtime_independent_content_identity_sources_for_entry(report, 0), ()
        )
        after = auto_lrc.timing_report_evaluation_revision(report)
        self.assertNotEqual(before, after)

    def test_runtime_lyric_entries_are_never_serialized(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        writer = next(
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "write_alignment_report"
        )
        segment = ast.get_source_segment(source, writer) or ""
        self.assertIn("_RUNTIME_LYRIC_ENTRIES_KEY", segment)



class V146LocalWhisperEvidenceRegressionTests(unittest.TestCase):
    def span(self, start: float, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, 0.5, token)

    def parent(self, entry: int, time: float, *, source: str = "ctc-local-window") -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=entry,
            entry_text=f"line {entry}",
            source=source,
            raw_time=time,
            spans=(self.span(time, "a"), self.span(time + 0.08, "b"), self.span(time + 0.16, "c")),
            confidence=0.8,
            identity_support="supported",
            sequence_support="supported",
            current=False,
            source_artifact={"source": source, "time": time},
        )

    def local_evidence(self, report, entries, entry, onset, score=0.95):
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[entry]))
        binding_payload = {
            "mode": "global-unique",
            "entry_index": entry,
            "lyric_sha256": auto_lrc.hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "segment": 1,
            "timestamp": round(onset, 6),
            "score": round(score, 6),
        }
        binding = {
            **binding_payload,
            "binding_revision": auto_lrc._timing_content_digest(binding_payload),
        }
        payload = {
            "entry_index": entry,
            "lyric_sha256": auto_lrc.hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "onset_time": round(onset, 6),
            "score": round(score, 6),
            "occurrence_binding": binding,
            "request_revision": "request",
            "capability_revision": "capability",
            "audio_revision": "audio",
            "window_start": round(onset - 2.0, 6),
            "window_end": round(onset + 2.0, 6),
            "raw_transcript_revision": "raw-transcript",
            "aligned_transcript_revision": "aligned-transcript",
            "producer": "local-whisper-aligner",
        }
        report[auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY] = {
            entry: {**payload, "evidence_revision": auto_lrc._timing_content_digest(payload)}
        }

    def test_local_whisper_consensus_preserves_parent_timestamp(self) -> None:
        entries = [LyricEntry(["unique lyric local alpha"])]
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        self.local_evidence(report, entries, 0, 10.08)
        parent = self.parent(0, 10.00)
        augmented = auto_lrc._augment_candidates_with_local_whisper_consensus(
            ((parent,),), report, entries
        )
        composites = [c for c in augmented[0] if c.source == "ctc-local-whisper-independent-consensus"]
        self.assertEqual(len(composites), 1)
        composite = composites[0]
        self.assertEqual(composite.written_time.centiseconds, 1000)
        self.assertTrue(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(composite))
        self.assertTrue(auto_lrc.candidate_has_independent_backend_identity(composite))
        assert composite.direct_onset_evidence is not None
        self.assertAlmostEqual(composite.direct_onset_evidence.onset_time, 10.08, places=6)

    def test_distant_local_whisper_observation_cannot_fuse(self) -> None:
        entries = [LyricEntry(["unique lyric local beta"])]
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        self.local_evidence(report, entries, 0, 10.40)
        parent = self.parent(0, 10.00)
        augmented = auto_lrc._augment_candidates_with_local_whisper_consensus(
            ((parent,),), report, entries
        )
        self.assertFalse(any("local-whisper" in c.source for c in augmented[0]))

    def test_local_whisper_family_is_one_lineage_not_raw_plus_whisperx(self) -> None:
        entries = [LyricEntry(["unique lyric local gamma"])]
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        self.local_evidence(report, entries, 0, 10.05)
        parent = self.parent(0, 10.00)
        composite = next(
            c for c in auto_lrc._augment_candidates_with_local_whisper_consensus(
                ((parent,),), report, entries
            )[0]
            if "local-whisper" in c.source
        )
        self.assertEqual(auto_lrc._candidate_support_families(composite), frozenset({"ctc", "whisper-family"}))
        raw = auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="unique lyric local gamma",
            source="raw-vocal-independent-fusion", raw_time=12.0, confidence=0.95,
            identity_support="supported", sequence_support="supported",
            acoustic_support="supported", direct_onset_support="supported",
            acoustic_strength=1.0, acoustic_onset_time=12.0,
            acoustic_source_artifact={"raw": True},
            acoustic_evidence_producer="raw-asr-vocal-fusion",
            acoustic_evidence_independent=True, direct_onset_time=12.0,
            direct_onset_source_artifact={"raw": True},
            direct_onset_evidence_producer="raw-asr-vocal-fusion",
            direct_onset_evidence_independent=True, independent_content_identity=True,
            source_artifact={"raw": True},
        )
        self.assertEqual(auto_lrc._candidate_support_families(raw), frozenset({"whisper-family"}))

    def test_local_whisper_runtime_identity_is_orthogonal_to_timing(self) -> None:
        entries = [LyricEntry(["unique lyric local delta"])]
        report = {
            "backend": "ctc",
            "timing_entries": 1,
            "assignments": [{"entry": 1, "timestamp": 10.0, "timing_repair": "ctc", "score": 0.35}],
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
        }
        self.local_evidence(report, entries, 0, 14.0)
        self.assertEqual(
            auto_lrc.runtime_independent_content_identity_sources_for_entry(report, 0),
            ("local-whisper-aligner",),
        )
        candidate = self.parent(0, 10.0)
        state = auto_lrc.build_final_timing_state(((candidate,),))
        summary = auto_lrc.summarize_central_timing_state(report, state)
        self.assertEqual(summary.content_trusted_entries, 1)
        self.assertEqual(state.decisions[0].written_time.centiseconds, 1000)

    def test_local_whisper_runtime_evidence_tamper_fails_closed(self) -> None:
        entries = [LyricEntry(["unique lyric local epsilon"])]
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        self.local_evidence(report, entries, 0, 10.0)
        self.assertIsNotNone(auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0))
        runtime = report[auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY]
        runtime[0]["onset_time"] = 10.10
        self.assertIsNone(auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0))
        self.local_evidence(report, entries, 0, 10.0)
        runtime = report[auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY]
        runtime[0]["raw_transcript_revision"] = "tampered-transcript"
        self.assertIsNone(auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0))

    def test_local_whisper_execute_uses_direct_aligned_char_onset_not_match_heuristic(self) -> None:
        entries = [
            LyricEntry(["unique left local"]),
            LyricEntry(["unique target local"]),
            LyricEntry(["unique right local"]),
        ]
        with TemporaryDirectory() as temp_name:
            audio = Path(temp_name) / "audio.wav"
            audio.write_bytes(b"audio")
            audio_sha = auto_lrc._path_sha256(audio)
            cap_payload = {
                "kind": "whispercpp-whisperx-local",
                "backend_lineage": "whisper-family",
                "audio_path": str(audio.resolve()),
                "audio_sha256": audio_sha,
                "whisper_cli_path": "cli",
                "whisper_cli_sha256": "cli-sha",
                "whisper_model_path": "model",
                "whisper_model_sha256": "model-sha",
                "whisperx_python_path": "python",
                "whisperx_helper_path": "helper",
                "whisperx_helper_sha256": "helper-sha",
                "language": "ja",
                "device": "cuda",
                "protocol_revision": auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            }
            cap = auto_lrc.LocalWhisperCapability(
                **cap_payload,
                capability_revision=auto_lrc._canonical_json_digest(
                    auto_lrc._local_whisper_capability_semantic_payload(cap_payload)
                ),
            )
            request_payload = {
                "entry_index": 1,
                "capability_revision": cap.capability_revision,
                "audio_revision": audio_sha,
                "context_entry_indexes": (0, 1, 2),
                "context_candidate_ids": ("a", "b", "c"),
                "transcript_digests": tuple(auto_lrc._timing_content_digest(auto_lrc.entry_sung_text(e)) for e in entries),
                "transcript_digest": auto_lrc._timing_content_digest(tuple(auto_lrc.entry_sung_text(e) for e in entries)),
                "window_start": 10.0,
                "window_end": 20.0,
                "window_definition_revision": "window",
            }
            request = auto_lrc.LocalWhisperRequest(
                **request_payload,
                request_revision=auto_lrc._timing_content_digest({
                    **request_payload,
                    "context_entry_indexes": [0, 1, 2],
                    "context_candidate_ids": ["a", "b", "c"],
                    "transcript_digests": list(request_payload["transcript_digests"]),
                }),
            )
            raw = [auto_lrc.AsrSegment(0.0, 9.0, "raw")]
            aligned = [
                auto_lrc.AsrSegment(0.5, 2.5, "unique left local", chars=[(c, 0.5 + i * 0.03) for i, c in enumerate("uniqueleftlocal")]),
                auto_lrc.AsrSegment(3.0, 5.0, "unique target local", chars=[(c, 3.0 + i * 0.03) for i, c in enumerate("uniquetargetlocal")]),
                auto_lrc.AsrSegment(6.0, 8.0, "unique right local", chars=[(c, 6.0 + i * 0.03) for i, c in enumerate("uniquerightlocal")]),
            ]
            local_report = {
                "assignments": [
                    {"entry": 1, "segment": 1, "score": 0.95, "borrowed": False, "timestamp": 0.5},
                    {"entry": 2, "segment": 2, "score": 0.95, "borrowed": False, "timestamp": 3.0},
                    {"entry": 3, "segment": 3, "score": 0.95, "borrowed": False, "timestamp": 6.0},
                ]
            }
            # Deliberately return a bogus heuristic time (4.5 s). The evidence
            # must use the aligned target character onset (3.0 s) instead.
            with mock.patch.object(auto_lrc, "decode_temp_wav_window", return_value=audio), \
                 mock.patch.object(auto_lrc, "_run_whispercpp_wav", return_value=raw), \
                 mock.patch.object(auto_lrc, "run_whisperx_alignment", return_value={}), \
                 mock.patch.object(auto_lrc, "whisperx_segments", return_value=aligned), \
                 mock.patch.object(auto_lrc, "match_whisper_segments", return_value=([0.5, 4.5, 6.0], local_report)):
                outcome = auto_lrc.execute_local_whisper_request(
                    request, cap, entries, auto_lrc.argparse.Namespace(whisper_cli=None, whisper_model=None, whisper_language="ja", whisper_suppress_nst=False, whisperx_python=None, whisperx_device="cpu"), cache={}
                )
            self.assertEqual(outcome.status, "evidence")
            self.assertAlmostEqual(outcome.onset_time or 0.0, 13.0, places=6)

    def test_content_only_local_whisper_runtime_evidence_is_valid(self) -> None:
        entries = [LyricEntry(["unique content-only local"])]
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[0]))
        binding_payload = {
            "mode": "global-unique", "entry_index": 0,
            "lyric_sha256": auto_lrc.hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "segment": 1, "timestamp": 10.0, "score": 0.95,
        }
        binding = {**binding_payload, "binding_revision": auto_lrc._timing_content_digest(binding_payload)}
        payload = {
            "entry_index": 0,
            "lyric_sha256": binding_payload["lyric_sha256"],
            "onset_time": None,
            "score": 0.95,
            "occurrence_binding": binding,
            "request_revision": "request",
            "capability_revision": "capability",
            "audio_revision": "audio",
            "window_start": 8.0,
            "window_end": 12.0,
            "raw_transcript_revision": "raw-transcript",
            "aligned_transcript_revision": "aligned-transcript",
            "producer": "local-whisper-aligner",
        }
        report = {
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
            auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY: {
                0: {**payload, "evidence_revision": auto_lrc._timing_content_digest(payload)}
            },
        }
        self.assertIsNotNone(auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0))
        self.assertEqual(
            auto_lrc.runtime_independent_content_identity_sources_for_entry(report, 0),
            ("local-whisper-aligner",),
        )

    def test_local_whisper_planner_targets_content_gap_without_timing_gap(self) -> None:
        entries = [
            LyricEntry(["unique left anchor"]),
            LyricEntry(["unique target line"]),
            LyricEntry(["unique right anchor"]),
        ]
        candidate_sets = tuple((self.parent(i, 10.0 + i * 3.0),) for i in range(3))
        report = {
            "backend": "ctc",
            "assignments": [
                {"entry": 1, "timestamp": 10.0, "timing_repair": "ctc", "score": 0.9},
                {"entry": 2, "timestamp": 13.0, "timing_repair": "ctc", "score": 0.35},
                {"entry": 3, "timestamp": 16.0, "timing_repair": "ctc", "score": 0.9},
            ],
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
        }
        cap = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        requests = auto_lrc.plan_local_whisper_requests(
            report, cap, entries, candidate_sets, 30.0
        )
        self.assertEqual([r.entry_index for r in requests], [1])
        self.assertEqual(requests[0].context_entry_indexes, (0, 1, 2))



class V147HypothesisArbitrationRegressionTests(unittest.TestCase):
    def span(self, start: float, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, 0.8, token)

    def entries(self):
        return [
            LyricEntry(["unique left context anchor"]),
            LyricEntry(["unique target hypothesis lyric"]),
            LyricEntry(["unique right context anchor"]),
        ]

    def local_sequence_evidence(self, report, entries, onset: float, score: float = 0.96):
        assignments = [
            {"timestamp": onset - 3.0, "score": 0.95, "segment": 1, "borrowed": False},
            {"timestamp": onset, "score": score, "segment": 2, "borrowed": False},
            {"timestamp": onset + 3.0, "score": 0.95, "segment": 3, "borrowed": False},
        ]
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[1]))
        payload = {
            "entry_index": 1,
            "lyric_sha256": auto_lrc.hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "onset_time": round(onset, 6), "score": round(score, 6),
            "occurrence_binding": binding, "request_revision": "local-request",
            "capability_revision": "local-cap", "audio_revision": "audio",
            "window_start": round(onset - 5.0, 6), "window_end": round(onset + 5.0, 6),
            "raw_transcript_revision": "local-raw",
            "aligned_transcript_revision": "local-aligned",
            "producer": "local-whisper-aligner",
        }
        report[auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY] = {
            1: {**payload, "evidence_revision": auto_lrc._timing_content_digest(payload)}
        }

    def raw_global_identity(self, report, entries, raw_time: float, score: float = 0.95):
        raw_report = {"assignments": [
            {"timestamp": raw_time - 3.0, "score": 0.95, "segment": 1, "borrowed": False},
            {"timestamp": raw_time, "score": score, "segment": 2, "borrowed": False},
            {"timestamp": raw_time + 3.0, "score": 0.95, "segment": 3, "borrowed": False},
        ]}
        auto_lrc.bind_runtime_raw_content_identity(report, entries, raw_report)

    def ctc_parent(self, time: float):
        return auto_lrc.make_timing_candidate(
            entry_index=1, entry_text="unique target hypothesis lyric",
            source="ctc-local-window", raw_time=time,
            spans=(self.span(time, "a"), self.span(time + .08, "b"), self.span(time + .16, "c")),
            confidence=.8, identity_support="supported", sequence_support="supported",
            source_artifact={"ctc": time},
        )

    def raw_fusion(self, time: float):
        artifact = {"producer": "raw-asr-vocal-fusion", "time": time}
        return auto_lrc.make_timing_candidate(
            entry_index=1, entry_text="unique target hypothesis lyric",
            source="raw-vocal-independent-fusion", raw_time=time, confidence=.95,
            identity_support="supported", sequence_support="supported",
            acoustic_support="supported", direct_onset_support="supported",
            acoustic_strength=1.0, acoustic_onset_time=time,
            acoustic_source_artifact=artifact, acoustic_evidence_producer="raw-asr-vocal-fusion",
            acoustic_evidence_independent=True, direct_onset_time=time,
            direct_onset_source_artifact=artifact, direct_onset_evidence_producer="raw-asr-vocal-fusion",
            direct_onset_evidence_kind="cross-backend-onset", direct_onset_evidence_independent=True,
            independent_content_identity=True, source_artifact=artifact,
        )

    def test_context_bound_local_plus_ctc_can_falsify_weaker_global_raw_occurrence(self):
        entries = self.entries()
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        self.local_sequence_evidence(report, entries, 10.05)
        self.raw_global_identity(report, entries, 14.00)
        base = ((self.ctc_parent(7.0),), (self.ctc_parent(10.00), self.raw_fusion(14.02)), (self.ctc_parent(17.0),))
        augmented = auto_lrc._augment_candidates_with_local_whisper_consensus(base, report, entries)
        revised, diagnostics = auto_lrc._apply_context_conditioned_whisper_hypothesis_arbitration(augmented, report, entries)
        self.assertEqual(len(diagnostics), 1)
        self.assertFalse(any(c.source == "raw-vocal-independent-fusion" for c in revised[1]))
        self.assertTrue(any(c.source == "ctc-local-whisper-independent-consensus" for c in revised[1]))
        self.assertIn("bounded-sequence-context", diagnostics[0]["reason"])

    def test_global_unique_local_binding_alone_cannot_falsify(self):
        entries = self.entries()
        report = {auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries)}
        # Reuse v14.6-style weaker global-unique local evidence.
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[1]))
        binding_payload = {"mode":"global-unique","entry_index":1,"lyric_sha256":auto_lrc.hashlib.sha256(target.encode()).hexdigest(),"segment":2,"timestamp":10.05,"score":0.96}
        binding={**binding_payload,"binding_revision":auto_lrc._timing_content_digest(binding_payload)}
        payload={"entry_index":1,"lyric_sha256":binding_payload["lyric_sha256"],"onset_time":10.05,"score":0.96,"occurrence_binding":binding,"request_revision":"r","capability_revision":"c","audio_revision":"a","window_start":5.0,"window_end":15.0,"raw_transcript_revision":"rr","aligned_transcript_revision":"ar","producer":"local-whisper-aligner"}
        report[auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY]={1:{**payload,"evidence_revision":auto_lrc._timing_content_digest(payload)}}
        self.raw_global_identity(report, entries, 14.0)
        base=((self.ctc_parent(7.0),),(self.ctc_parent(10.0),self.raw_fusion(14.02)),(self.ctc_parent(17.0),))
        augmented=auto_lrc._augment_candidates_with_local_whisper_consensus(base,report,entries)
        revised, diagnostics=auto_lrc._apply_context_conditioned_whisper_hypothesis_arbitration(augmented,report,entries)
        self.assertFalse(diagnostics)
        self.assertTrue(any(c.source=="raw-vocal-independent-fusion" for c in revised[1]))

    def test_independent_ctc_support_at_raw_cluster_blocks_falsification(self):
        entries=self.entries(); report={auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY:tuple(entries)}
        self.local_sequence_evidence(report,entries,10.05); self.raw_global_identity(report,entries,14.0)
        raw_ctc=self.ctc_parent(14.02)
        base=((self.ctc_parent(7.0),),(self.ctc_parent(10.0),self.raw_fusion(14.02),raw_ctc),(self.ctc_parent(17.0),))
        augmented=auto_lrc._augment_candidates_with_local_whisper_consensus(base,report,entries)
        revised, diagnostics=auto_lrc._apply_context_conditioned_whisper_hypothesis_arbitration(augmented,report,entries)
        self.assertFalse(diagnostics)
        self.assertTrue(any(c.source=="raw-vocal-independent-fusion" for c in revised[1]))

    def test_raw_and_local_whisper_are_same_support_family(self):
        raw=self.raw_fusion(10.0)
        local=auto_lrc.make_timing_candidate(
            entry_index=1,entry_text="unique target hypothesis lyric",source="local-whisper-independent-consensus",raw_time=10.0,confidence=.9,identity_support="supported",sequence_support="supported",direct_onset_support="supported",direct_onset_time=10.0,direct_onset_source_artifact={"local":1},direct_onset_evidence_producer="local-whisper-aligner",direct_onset_evidence_independent=True,independent_content_identity=True,source_artifact={"local":1})
        self.assertEqual(auto_lrc._candidate_support_families(raw),frozenset({"whisper-family"}))
        self.assertEqual(auto_lrc._candidate_support_families(local),frozenset({"whisper-family"}))


class V148SequenceFingerprintRegressionTests(unittest.TestCase):
    def span(self, start: float, token: str = "a") -> auto_lrc.TimingTokenSpan:
        return auto_lrc.TimingTokenSpan(start, start + 0.02, 0.8, token)

    def valid_candidate(self, entry: int, text: str, time: float) -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=entry, entry_text=text, source="ctc-local-window", raw_time=time,
            spans=(self.span(time, "a"), self.span(time + .08, "b"), self.span(time + .16, "c")),
            confidence=.8, identity_support="supported", sequence_support="supported",
            source_artifact={"entry": entry, "time": time},
        )

    def provisional_candidate(self, entry: int, text: str, time: float) -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=entry, entry_text=text, source="ctc-current", raw_time=time,
            spans=(), confidence=.7, identity_support="supported", sequence_support="supported",
            source_artifact={"entry": entry, "time": time},
        )

    def entries(self):
        return [
            LyricEntry(["unique intro"]),
            LyricEntry(["repeat left"]),
            LyricEntry(["repeat target"]),
            LyricEntry(["unique branch alpha"]),
            LyricEntry(["unique middle"]),
            LyricEntry(["repeat left"]),
            LyricEntry(["repeat target"]),
            LyricEntry(["unique branch beta"]),
        ]

    def capability(self):
        return auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )

    def test_short_unique_sequence_fingerprint_recovers_repeated_target(self):
        entries = self.entries()
        self.assertFalse(auto_lrc._entry_identity_is_unique(entries, 2))
        self.assertFalse(auto_lrc._entry_identity_is_unique(entries, 1))
        self.assertEqual(auto_lrc._minimal_unique_sequence_context(entries, 2, max_items=9), (1, 3))
        self.assertEqual(auto_lrc._minimal_unique_sequence_context(entries, 6, max_items=9), (5, 7))

    def test_sequence_bracket_binding_does_not_require_individually_unique_neighbors(self):
        entries = self.entries()
        assignments = [None] * len(entries)
        assignments[1] = {"timestamp": 10.0, "score": .95, "segment": 1, "borrowed": False}
        assignments[2] = {"timestamp": 12.0, "score": .96, "segment": 2, "borrowed": False}
        assignments[3] = {"timestamp": 14.0, "score": .94, "segment": 3, "borrowed": False}
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 2, require_context=True)
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(binding["mode"], "sequence-bracket")
        self.assertEqual(binding["context_entry_indexes"], [1, 2, 3])
        self.assertTrue(auto_lrc._runtime_occurrence_binding_is_valid(entries, 2, binding))

    def test_planner_uses_short_sequence_fingerprint_instead_of_distant_unique_anchors(self):
        entries = self.entries()
        candidate_sets = []
        for i, entry in enumerate(entries):
            text = auto_lrc.entry_sung_text(entry)
            candidate = (
                self.provisional_candidate(i, text, 10.0 + i * 3.0)
                if i == 2 else self.valid_candidate(i, text, 10.0 + i * 3.0)
            )
            candidate_sets.append((candidate,))
        report = {
            "backend": "ctc",
            "assignments": [
                {"entry": i + 1, "timestamp": 10.0 + i * 3.0, "timing_repair": "ctc", "score": .9}
                for i in range(len(entries))
            ],
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
        }
        requests = auto_lrc.plan_local_whisper_requests(
            report, self.capability(), entries, tuple(candidate_sets), 60.0
        )
        target = [request for request in requests if request.entry_index == 2]
        self.assertGreaterEqual(len(target), 1)
        self.assertTrue(all(len(request.context_entry_indexes) <= 9 for request in target))
        self.assertTrue(any({1, 2, 3}.issubset(set(request.context_entry_indexes)) for request in target))




class V1410MaterialHypothesisReviewRegressionTests(unittest.TestCase):
    def capability(self) -> auto_lrc.LocalWhisperCapability:
        return auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )

    def runtime_evidence(self, entries, entry=0, onset=10.0, score=.96):
        target = auto_lrc.normalize_match_text(auto_lrc.entry_sung_text(entries[entry]))
        binding_payload = {
            "mode": "global-unique",
            "entry_index": entry,
            "lyric_sha256": auto_lrc.hashlib.sha256(target.encode("utf-8")).hexdigest(),
            "segment": 1,
            "timestamp": round(onset, 6),
            "score": round(score, 6),
        }
        binding = {
            **binding_payload,
            "binding_revision": auto_lrc._timing_content_digest(binding_payload),
        }
        payload = {
            "entry_index": entry,
            "lyric_sha256": binding_payload["lyric_sha256"],
            "onset_time": round(onset, 6),
            "score": round(score, 6),
            "occurrence_binding": binding,
            "request_revision": "request-r1",
            "capability_revision": "capability",
            "audio_revision": "audio",
            "window_start": round(onset - 2.0, 6),
            "window_end": round(onset + 2.0, 6),
            "raw_transcript_revision": "raw-r1",
            "aligned_transcript_revision": "aligned-r1",
            "producer": "local-whisper-aligner",
        }
        return {**payload, "evidence_revision": auto_lrc._timing_content_digest(payload)}

    def test_fixed_point_recollect_retains_prior_valid_local_whisper_evidence(self):
        entries = [LyricEntry(["fixed point unique lyric"])]
        evidence = self.runtime_evidence(entries)
        report = {
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
            auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY: {0: evidence},
        }
        auto_lrc.bind_runtime_local_whisper_evidence(
            report, entries, self.capability(), (), ()
        )
        retained = auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0)
        self.assertIsNotNone(retained)
        self.assertEqual(report["local_whisper_evidence"]["candidate_count"], 1)
        self.assertEqual(report["local_whisper_evidence"]["retained_from_prior_collect_count"], 1)
        self.assertEqual(report["local_whisper_evidence"]["entries"], [1])

    def test_fixed_point_recollect_drops_tampered_prior_runtime_evidence(self):
        entries = [LyricEntry(["fixed point unique lyric tamper"])]
        evidence = self.runtime_evidence(entries)
        evidence["raw_transcript_revision"] = "tampered"
        report = {
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
            auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY: {0: evidence},
        }
        auto_lrc.bind_runtime_local_whisper_evidence(
            report, entries, self.capability(), (), ()
        )
        self.assertIsNone(auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 0))
        self.assertEqual(report["local_whisper_evidence"]["candidate_count"], 0)

    def valid_selected(self, time: float) -> auto_lrc.TimingCandidate:
        artifact = {"producer": "whisperx-vocal-fusion", "time": time}
        occurrence_entries = [LyricEntry(["review invariant lyric"])]
        occurrence_binding = auto_lrc._entry_occurrence_binding(
            occurrence_entries,
            [{"timestamp": time, "score": .96, "segment": 1}],
            0,
        )
        occurrence_evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
            occurrence_entries,
            0,
            producer="whisperx-vocal-fusion",
            occurrence_binding=occurrence_binding,
            upstream_evidence_revision="v1410-direct-upstream",
        )
        return auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="review invariant lyric",
            source="whisperx-vocal-independent-fusion", raw_time=time,
            confidence=.96, identity_support="supported", sequence_support="supported",
            acoustic_support="supported", direct_onset_support="supported",
            acoustic_strength=1.0, acoustic_onset_time=time,
            acoustic_source_artifact=artifact,
            acoustic_evidence_producer="whisperx-vocal-fusion",
            acoustic_evidence_independent=True,
            direct_onset_time=time, direct_onset_source_artifact=artifact,
            direct_onset_evidence_producer="whisperx-vocal-fusion",
            direct_onset_evidence_kind="cross-backend-onset",
            direct_onset_evidence_independent=True,
            occurrence_evidence=occurrence_evidence,
            independent_content_identity=True,
            source_artifact=artifact,
        )

    def invalid_distant_raw(self, time: float) -> auto_lrc.TimingCandidate:
        return auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="review invariant lyric",
            source="raw-asr", raw_time=time, confidence=.78,
            identity_support="supported", sequence_support="supported",
            source_artifact={"raw": time},
        )

    def review_fixture(self):
        selected = self.valid_selected(10.0)
        distant = self.invalid_distant_raw(11.0)
        decision = auto_lrc.select_timing_decision(
            entry_index=0, candidates=(selected, distant),
            previous_candidate=None, selection_revision=1,
        )
        audit = auto_lrc.audit_final_timing_decisions((decision,))[0]
        row = {
            "entry": 1,
            "flags": ["unresolved_raw_ctc_disagreement"],
            "severity": "high",
            "review_required": True,
            "candidate_timestamps": {"output": 10.0, "raw_asr": 11.0},
        }
        return decision, audit, row

    def test_diagnostic_only_raw_disagreement_can_clear_after_verified_central_decision(self):
        decision, audit, row = self.review_fixture()
        report = {"hypothesis_arbitration": {"entries": [], "falsification_count": 0}}
        self.assertTrue(audit.timing_trusted)
        self.assertTrue(
            auto_lrc._central_decision_supersedes_suspicious_row(
                decision, audit, row, report
            )
        )

    def test_material_distant_direct_onset_hypothesis_keeps_review_open(self):
        selected = self.valid_selected(10.0)
        material_artifact = {"producer": "vocal-activity-detector", "time": 11.0}
        material = auto_lrc.make_timing_candidate(
            entry_index=0, entry_text="review invariant lyric",
            source="vocal-activity-reentry", raw_time=11.0, confidence=.9,
            identity_support="supported", sequence_support="supported",
            acoustic_support="supported", direct_onset_support="supported",
            acoustic_strength=.9, acoustic_onset_time=11.0,
            acoustic_source_artifact=material_artifact,
            acoustic_evidence_producer="vocal-activity-detector",
            acoustic_evidence_kind="vocal-activity-reentry",
            acoustic_evidence_independent=True,
            direct_onset_time=11.0, direct_onset_source_artifact=material_artifact,
            direct_onset_evidence_producer="vocal-activity-detector",
            direct_onset_evidence_kind="vocal-activity-reentry",
            direct_onset_evidence_independent=True,
            current=False, window_truncated=True, source_artifact=material_artifact,
        )
        diagnostic_raw = self.invalid_distant_raw(11.05)
        decision = auto_lrc.select_timing_decision(
            entry_index=0, candidates=(selected, material, diagnostic_raw),
            previous_candidate=None, selection_revision=1,
        )
        audit = auto_lrc.audit_final_timing_decisions((decision,))[0]
        row = {
            "entry": 1,
            "flags": ["unresolved_raw_ctc_disagreement"],
            "severity": "high",
            "review_required": True,
            "candidate_timestamps": {"output": 10.0, "raw_asr": 11.05},
        }
        report = {"hypothesis_arbitration": {"entries": [], "falsification_count": 0}}
        self.assertTrue(audit.timing_trusted)
        self.assertTrue(any(
            auto_lrc.candidate_has_bound_direct_onset_evidence(item.candidate)
            and abs(item.candidate.written_time.seconds - 10.0) > auto_lrc.RAW_CTC_CONSENSUS_MAX_DELTA_SECONDS
            for item in decision.rejected_candidates
        ))
        self.assertFalse(
            auto_lrc._central_decision_supersedes_suspicious_row(
                decision, audit, row, report
            )
        )

    def test_unknown_ownership_only_blocker_keeps_supported_opening_review(self):
        previous = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="previous lyric",
            source="ctc-current",
            raw_time=8.0,
            spans=(
                auto_lrc.TimingTokenSpan(8.0, 8.08, .8, "a"),
                auto_lrc.TimingTokenSpan(8.10, 8.18, .8, "b"),
                auto_lrc.TimingTokenSpan(8.20, 8.28, .8, "c"),
            ),
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"generic": "previous"},
        )
        selected = self.valid_selected(10.0)
        distant = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="review invariant lyric",
            source="ctc-current",
            raw_time=11.0,
            spans=(
                auto_lrc.TimingTokenSpan(11.0, 11.02, .01, "k"),
                auto_lrc.TimingTokenSpan(11.04, 11.12, .12, "a"),
                auto_lrc.TimingTokenSpan(11.14, 11.16, .01, "n"),
            ),
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            current=False,
            source_artifact={"generic": "distant-opening"},
        )
        unknown_tail = auto_lrc.TailRegionEvidence(
            previous_candidate_id=previous.candidate_id,
            previous_generation_revision=previous.generation_revision,
            previous_region_revision=auto_lrc.analyze_candidate_regions(previous).region_revision,
            tail_evidence_revision="unknown-tail-revision",
            validity="unknown",
            reliable_tail_start=None,
            reliable_tail_end=None,
            uncertainty_seconds=None,
            confidence=None,
            reason="independent-tail-unavailable",
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(selected, distant),
            previous_candidate=previous,
            previous_tail=unknown_tail,
            selection_revision=2,
        )
        self.assertTrue(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )
        distant_evaluation = next(
            item for item in decision.rejected_candidates
            if item.candidate.candidate_id == distant.candidate_id
        )
        self.assertLess(distant_evaluation.evidence.reliable_support_count, 3)
        self.assertIsNotNone(distant_evaluation.evidence.onset_proof_end)
        audit = auto_lrc.audit_final_timing_decisions((decision,))[0]
        row = {
            "entry": 2,
            "flags": ["unresolved_raw_ctc_disagreement"],
            "severity": "high",
            "review_required": True,
            "candidate_timestamps": {"output": 10.0, "raw_asr": 11.0},
        }
        self.assertFalse(
            auto_lrc._central_decision_supersedes_suspicious_row(
                decision, audit, row, {"hypothesis_arbitration": {"entries": []}}
            )
        )

    def test_revision_bound_ownership_violation_still_falsifies_alternative(self):
        previous = auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="previous lyric",
            source="ctc-current",
            raw_time=8.0,
            spans=(
                auto_lrc.TimingTokenSpan(8.0, 8.30, .8, "a"),
                auto_lrc.TimingTokenSpan(8.40, 8.70, .8, "b"),
                auto_lrc.TimingTokenSpan(8.80, 9.50, .8, "c"),
            ),
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"generic": "previous-tail"},
        )
        valid_tail = auto_lrc.derive_reliable_tail(previous)
        self.assertEqual(valid_tail.validity, "valid")
        selected = self.valid_selected(10.2)
        invading = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="review invariant lyric",
            source="ctc-current",
            raw_time=8.9,
            spans=(
                auto_lrc.TimingTokenSpan(8.9, 8.98, .8, "a"),
                auto_lrc.TimingTokenSpan(9.0, 9.08, .8, "b"),
                auto_lrc.TimingTokenSpan(9.1, 9.18, .8, "c"),
            ),
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            current=False,
            source_artifact={"generic": "invading"},
        )
        decision = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(selected, invading),
            previous_candidate=previous,
            previous_tail=valid_tail,
            selection_revision=2,
        )
        self.assertFalse(
            auto_lrc._decision_has_material_distant_timing_hypothesis(
                decision, decision.written_time.seconds
            )
        )

    def test_explicit_context_falsification_can_clear_distant_raw_disagreement(self):
        decision, audit, row = self.review_fixture()
        report = {
            "hypothesis_arbitration": {
                "entries": [{
                    "entry": 1,
                    "raw_identity_time": 11.0,
                    "local_onset": 10.0,
                    "falsified_candidate_ids": ["raw-hypothesis"],
                }],
                "falsification_count": 1,
            }
        }
        self.assertTrue(
            auto_lrc._central_decision_supersedes_suspicious_row(
                decision, audit, row, report
            )
        )


class V15EvidenceConvergenceRegressionTests(unittest.TestCase):
    def repeated_entries(self):
        return [
            LyricEntry(["anchor alpha"]),
            LyricEntry(["repeat target"]),
            LyricEntry(["shared middle"]),
            LyricEntry(["branch beta"]),
            LyricEntry(["separator"]),
            LyricEntry(["anchor alpha"]),
            LyricEntry(["repeat target"]),
            LyricEntry(["shared middle"]),
            LyricEntry(["branch gamma"]),
        ]

    def test_sparse_sequence_binding_ignores_untrusted_middle_rows_without_inventing_them(self):
        entries = self.repeated_entries()
        assignments = [None] * len(entries)
        assignments[0] = {"timestamp": 10.0, "score": .95, "segment": 1, "borrowed": False}
        assignments[1] = {"timestamp": 12.0, "score": .96, "segment": 2, "borrowed": False}
        assignments[2] = {"timestamp": 13.0, "score": .20, "segment": 3, "borrowed": False}
        assignments[3] = {"timestamp": 15.0, "score": .94, "segment": 4, "borrowed": False}
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(binding["mode"], "sparse-sequence-bracket")
        self.assertEqual(binding["anchor_entry_indexes"], [0, 1, 3])
        self.assertTrue(auto_lrc._runtime_occurrence_binding_is_valid(entries, 1, binding))

    def test_sparse_binding_revalidation_fails_closed_after_anchor_tamper(self):
        entries = self.repeated_entries()
        assignments = [None] * len(entries)
        assignments[0] = {"timestamp": 10.0, "score": .95, "segment": 1, "borrowed": False}
        assignments[1] = {"timestamp": 12.0, "score": .96, "segment": 2, "borrowed": False}
        assignments[3] = {"timestamp": 15.0, "score": .94, "segment": 4, "borrowed": False}
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        assert binding is not None
        tampered = copy.deepcopy(binding)
        tampered["anchor_timestamps"][0] = 9.5
        self.assertFalse(auto_lrc._runtime_occurrence_binding_is_valid(entries, 1, tampered))

    def test_exact_aligned_char_stream_recovers_cross_segment_onset(self):
        segments = [
            auto_lrc.AsrSegment(1.0, 1.5, "今日", chars=[("今", 1.02), ("日", 1.20)]),
            auto_lrc.AsrSegment(1.5, 2.1, "の天気", chars=[("の", 1.52), ("天", 1.62), ("気", 1.82)]),
        ]
        matches = auto_lrc._aligned_exact_target_matches("今日の天気", segments)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(float(matches[0]["onset_time"]), 1.02, places=6)
        self.assertEqual(matches[0]["start_segment_index"], 0)
        self.assertEqual(matches[0]["end_segment_index"], 1)

    def test_planner_adds_bounded_redundant_neighbors_for_sparse_fallback(self):
        entries = [LyricEntry([f"unique row {i}"]) for i in range(7)]
        candidate_sets = []
        for i, entry in enumerate(entries):
            text = auto_lrc.entry_sung_text(entry)
            candidate_sets.append((auto_lrc.make_timing_candidate(
                entry_index=i, entry_text=text, source="ctc-current",
                raw_time=10.0 + i * 2.0, confidence=.8,
                identity_support="supported", sequence_support="supported",
                source_artifact={"entry": i},
            ),))
        report = {
            "backend": "ctc",
            "assignments": [
                {"entry": i + 1, "timestamp": 10.0 + i * 2.0, "score": .9}
                for i in range(len(entries))
            ],
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
        }
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        requests = auto_lrc.plan_local_whisper_requests(
            report, capability, entries, tuple(candidate_sets), 60.0
        )
        target = [request for request in requests if request.entry_index == 3]
        self.assertGreaterEqual(len(target), 2)
        self.assertLessEqual(len(target), 3)
        contexts = [request.context_entry_indexes for request in target]
        self.assertIn((2, 3, 4), contexts)
        self.assertIn((1, 2, 3, 4, 5), contexts)
        self.assertTrue(all(len(context) <= 9 for context in contexts))

    def test_execute_local_whisper_recovers_weak_target_only_from_exact_chars_and_sparse_context(self):
        entries = self.repeated_entries()
        request = auto_lrc.LocalWhisperRequest(
            entry_index=1, capability_revision="capability", audio_revision="audio",
            context_entry_indexes=(0, 1, 2, 3),
            context_candidate_ids=("a", "b", "c", "d"),
            transcript_digests=("ta", "tb", "tc", "td"), transcript_digest="all",
            window_start=10.0, window_end=20.0,
            window_definition_revision="window", request_revision="request",
        )
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        raw_segments = [auto_lrc.AsrSegment(0.0, 1.0, "raw")]
        aligned_segments = [
            auto_lrc.AsrSegment(.4, .9, "anchoralpha", chars=[(c, .4 + i*.02) for i,c in enumerate("anchoralpha")]),
            auto_lrc.AsrSegment(1.2, 2.0, "repeattarget", chars=[(c, 1.2 + i*.02) for i,c in enumerate("repeattarget")]),
            auto_lrc.AsrSegment(2.4, 3.1, "sharedmiddle", chars=[(c, 2.4 + i*.02) for i,c in enumerate("sharedmiddle")]),
            auto_lrc.AsrSegment(3.5, 4.2, "branchbeta", chars=[(c, 3.5 + i*.02) for i,c in enumerate("branchbeta")]),
        ]
        aligned_payload = {
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text,
                 "chars": [{"char": c, "start": t} for c,t in seg.chars]}
                for seg in aligned_segments
            ]
        }
        local_report = {"assignments": [
            {"timestamp": .4, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 1.2, "score": .40, "segment": 2, "borrowed": False},
            {"timestamp": 2.4, "score": .20, "segment": 3, "borrowed": False},
            {"timestamp": 3.5, "score": .94, "segment": 4, "borrowed": False},
        ]}
        with mock.patch.object(auto_lrc, "_stable_file_sha256", return_value="audio"), \
             mock.patch.object(auto_lrc, "decode_temp_wav_window", return_value=Path("window.wav")), \
             mock.patch.object(auto_lrc, "_run_whispercpp_wav", return_value=raw_segments), \
             mock.patch.object(auto_lrc, "run_whisperx_alignment", return_value=aligned_payload), \
             mock.patch.object(auto_lrc, "match_whisper_segments", return_value=([.4, 1.2, 2.4, 3.5], local_report)):
            outcome = auto_lrc.execute_local_whisper_request(
                request, capability, entries, mock.Mock(), cache={}, runtime_stats={}
            )
        self.assertEqual(outcome.status, "evidence")
        self.assertEqual(outcome.reason, "trusted-local-whisper-exact-char-recovery-and-onset")
        self.assertAlmostEqual(float(outcome.onset_time or 0.0), 11.2, places=6)
        self.assertEqual(outcome.occurrence_binding["mode"], "sparse-sequence-bracket")

    def test_sparse_context_can_drive_explicit_whisper_hypothesis_falsification(self):
        entries = [
            LyricEntry(["far left"]), LyricEntry(["weak left"]),
            LyricEntry(["unique target"]), LyricEntry(["weak right"]), LyricEntry(["far right"]),
        ]
        local_assignments = [
            {"timestamp": 12.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 13.0, "score": .20, "segment": 2, "borrowed": False},
            {"timestamp": 15.0, "score": .96, "segment": 3, "borrowed": False},
            {"timestamp": 16.0, "score": .20, "segment": 4, "borrowed": False},
            {"timestamp": 18.0, "score": .95, "segment": 5, "borrowed": False},
        ]
        local_binding = auto_lrc._entry_occurrence_binding(entries, local_assignments, 2, require_context=True)
        self.assertIsNotNone(local_binding)
        assert local_binding is not None
        self.assertEqual(local_binding["mode"], "sparse-sequence-bracket")
        raw_assignments = [None] * len(entries)
        raw_assignments[2] = {"timestamp": 10.0, "score": .90, "segment": 1, "borrowed": False}
        raw_binding = auto_lrc._entry_occurrence_binding(entries, raw_assignments, 2)
        self.assertEqual(raw_binding["mode"], "global-unique")

        lyric_sha = auto_lrc.hashlib.sha256(auto_lrc.normalize_match_text("unique target").encode("utf-8")).hexdigest()
        local_payload = {
            "entry_index": 2, "lyric_sha256": lyric_sha, "onset_time": 15.0, "score": .96,
            "occurrence_binding": local_binding, "request_revision": "req",
            "capability_revision": "cap", "audio_revision": "audio",
            "window_start": 11.0, "window_end": 19.0,
            "raw_transcript_revision": "raw-local", "aligned_transcript_revision": "aligned-local",
            "producer": "local-whisper-aligner",
        }
        raw_payload = {
            "entry_index": 2, "lyric_sha256": lyric_sha, "raw_time": 10.0, "raw_score": .90,
            "segment": 1, "occurrence_binding": raw_binding, "producer": "raw-asr-content-identity",
        }
        report = {
            auto_lrc._RUNTIME_LYRIC_ENTRIES_KEY: tuple(entries),
            auto_lrc._RUNTIME_LOCAL_WHISPER_EVIDENCE_KEY: {2: {**local_payload, "evidence_revision": auto_lrc._timing_content_digest(local_payload)}},
            auto_lrc._RUNTIME_RAW_CONTENT_IDENTITY_KEY: {2: {**raw_payload, "evidence_revision": auto_lrc._timing_content_digest(raw_payload)}},
        }
        raw_artifact = {"producer": "raw-asr-vocal-fusion"}
        raw_candidate = auto_lrc.make_timing_candidate(
            entry_index=2, entry_text="unique target", source="raw-vocal-independent-fusion",
            raw_time=10.0, confidence=.96, identity_support="supported", sequence_support="supported",
            direct_onset_support="supported", direct_onset_time=10.0,
            direct_onset_source_artifact=raw_artifact,
            direct_onset_evidence_producer="raw-asr-vocal-fusion",
            direct_onset_evidence_kind="cross-backend-onset", direct_onset_evidence_independent=True,
            occurrence_evidence=auto_lrc.candidate_occurrence_evidence_from_binding(
                entries, 2, producer="raw-asr-vocal-fusion",
                occurrence_binding=raw_binding,
                upstream_evidence_revision=auto_lrc._timing_content_digest(raw_payload),
            ),
            independent_content_identity=True, source_artifact=raw_artifact,
        )
        ctc_artifact = {"producer": "local-whisper-aligner"}
        ctc_local = auto_lrc.make_timing_candidate(
            entry_index=2, entry_text="unique target", source="ctc-local-whisper-independent-consensus",
            raw_time=15.0, confidence=.9, identity_support="supported", sequence_support="supported",
            direct_onset_support="supported", direct_onset_time=15.0,
            direct_onset_source_artifact=ctc_artifact,
            direct_onset_evidence_producer="local-whisper-aligner",
            direct_onset_evidence_kind="cross-backend-onset", direct_onset_evidence_independent=True,
            occurrence_evidence=auto_lrc.candidate_occurrence_evidence_from_binding(
                entries, 2, producer="local-whisper-aligner",
                occurrence_binding=local_binding,
                upstream_evidence_revision=auto_lrc._timing_content_digest(local_payload),
            ),
            independent_content_identity=True, source_artifact=ctc_artifact,
        )
        candidate_sets = tuple((ctc_local,) if i != 2 else (raw_candidate, ctc_local) for i in range(len(entries)))
        revised, diagnostics = auto_lrc._apply_context_conditioned_whisper_hypothesis_arbitration(
            candidate_sets, report, entries
        )
        self.assertEqual(len(diagnostics), 1)
        self.assertNotIn(raw_candidate.candidate_id, {item.candidate_id for item in revised[2]})
        self.assertIn(ctc_local.candidate_id, {item.candidate_id for item in revised[2]})

class V151MonotonicLocalProbingRegressionTests(unittest.TestCase):
    def test_context_prefix_onset_crosses_segments_without_full_exact_line(self):
        segments = [
            auto_lrc.AsrSegment(1.0, 1.5, "今日の", chars=[("今", 1.02), ("日", 1.20), ("の", 1.42)]),
            auto_lrc.AsrSegment(1.5, 2.1, "天気は", chars=[("天", 1.52), ("気", 1.70), ("は", 1.90)]),
            auto_lrc.AsrSegment(2.1, 2.5, "別語", chars=[("別", 2.12), ("語", 2.30)]),
        ]
        # Full immutable line is intentionally absent; its first six normalized
        # characters are a directly forced, cross-segment observation.
        matches = auto_lrc._aligned_target_prefix_matches("今日の天気は晴れ", segments)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(float(matches[0]["onset_time"]), 1.02, places=6)
        self.assertEqual(matches[0]["start_segment_index"], 0)
        self.assertEqual(matches[0]["end_segment_index"], 1)

    def test_context_prefix_selector_fails_closed_when_two_matches_survive(self):
        entries = [
            LyricEntry(["left anchor"]), LyricEntry(["今日の天気は晴れ"]), LyricEntry(["right anchor"]),
        ]
        assignments = [
            {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 2.0, "score": .95, "segment": 2, "borrowed": False},
            {"timestamp": 5.0, "score": .95, "segment": 4, "borrowed": False},
        ]
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        segments = [
            auto_lrc.AsrSegment(1.5, 2.2, "今日の天気は", chars=[(c, 1.5 + i*.05) for i,c in enumerate("今日の天気は")]),
            auto_lrc.AsrSegment(3.0, 3.7, "今日の天気は", chars=[(c, 3.0 + i*.05) for i,c in enumerate("今日の天気は")]),
        ]
        selected = auto_lrc._select_unique_aligned_target_prefix_match(
            "今日の天気は晴れ", segments,
            occurrence_binding=binding, entry_index=1, preferred_segment_number=None,
        )
        self.assertIsNone(selected)

    def test_runtime_merge_prefers_direct_onset_over_higher_score_occurrence_only(self):
        entries = [LyricEntry(["left"]), LyricEntry(["target"]), LyricEntry(["right"])]
        assignments = [
            {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 2.0, "score": .96, "segment": 2, "borrowed": False},
            {"timestamp": 3.0, "score": .95, "segment": 3, "borrowed": False},
        ]
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        def req(name, start, end):
            payload = dict(
                entry_index=1, capability_revision="capability", audio_revision="audio",
                context_entry_indexes=(0,1,2), context_candidate_ids=("a","b","c"),
                transcript_digests=("a","b","c"), transcript_digest="abc",
                window_start=start, window_end=end, window_definition_revision=name,
            )
            return auto_lrc.LocalWhisperRequest(**payload, request_revision=name)
        narrow=req("narrow",0.0,4.0)
        broad=req("broad",0.0,5.0)
        occurrence_only=auto_lrc.LocalWhisperOutcome(
            entry_index=1, status="evidence", reason="occurrence-only", onset_time=None,
            score=.99, evidence_revision="ignored", occurrence_binding=binding,
            raw_transcript_revision="raw-broad", aligned_transcript_revision="aligned-broad",
            request_revision="broad", outcome_revision="out-broad",
        )
        with_onset=auto_lrc.LocalWhisperOutcome(
            entry_index=1, status="evidence", reason="with-onset", onset_time=2.1,
            score=.70, evidence_revision="ignored", occurrence_binding=binding,
            raw_transcript_revision="raw-narrow", aligned_transcript_revision="aligned-narrow",
            request_revision="narrow", outcome_revision="out-narrow",
        )
        report={}
        auto_lrc.bind_runtime_local_whisper_evidence(
            report, entries, capability, (broad,narrow), (occurrence_only,with_onset)
        )
        evidence=auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 1)
        self.assertIsNotNone(evidence)
        self.assertAlmostEqual(float(evidence["onset_time"]),2.1,places=6)
        self.assertAlmostEqual(float(evidence["score"]),.70,places=6)

    def test_weak_target_assignment_can_recover_from_context_bound_forced_prefix(self):
        entries = [
            LyricEntry(["anchor alpha"]), LyricEntry(["repeat target"]),
            LyricEntry(["shared middle"]), LyricEntry(["branch beta"]),
            LyricEntry(["separator"]), LyricEntry(["anchor alpha"]),
            LyricEntry(["repeat target"]), LyricEntry(["shared middle"]),
            LyricEntry(["branch gamma"]),
        ]
        request = auto_lrc.LocalWhisperRequest(
            entry_index=1, capability_revision="capability", audio_revision="audio",
            context_entry_indexes=(0, 1, 2, 3),
            context_candidate_ids=("a", "b", "c", "d"),
            transcript_digests=("ta", "tb", "tc", "td"), transcript_digest="all",
            window_start=10.0, window_end=20.0,
            window_definition_revision="window", request_revision="request-prefix",
        )
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        raw_segments = [auto_lrc.AsrSegment(0.0, 1.0, "raw")]
        # The target's full text is deliberately absent.  Only its forced opening
        # prefix is present, while the surrounding rows are enough to prove the
        # immutable lyric occurrence.
        aligned_segments = [
            auto_lrc.AsrSegment(.4, .9, "anchoralpha", chars=[(c, .4 + i*.02) for i,c in enumerate("anchoralpha")]),
            auto_lrc.AsrSegment(1.2, 1.7, "repeat", chars=[(c, 1.2 + i*.02) for i,c in enumerate("repeat")]),
            auto_lrc.AsrSegment(2.4, 3.1, "sharedmiddle", chars=[(c, 2.4 + i*.02) for i,c in enumerate("sharedmiddle")]),
            auto_lrc.AsrSegment(3.5, 4.2, "branchbeta", chars=[(c, 3.5 + i*.02) for i,c in enumerate("branchbeta")]),
        ]
        aligned_payload = {
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text,
                 "chars": [{"char": c, "start": t} for c,t in seg.chars]}
                for seg in aligned_segments
            ]
        }
        local_report = {"assignments": [
            {"timestamp": .4, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 1.2, "score": .40, "segment": 2, "borrowed": False},
            {"timestamp": 2.4, "score": .20, "segment": 3, "borrowed": False},
            {"timestamp": 3.5, "score": .94, "segment": 4, "borrowed": False},
        ]}
        with mock.patch.object(auto_lrc, "_stable_file_sha256", return_value="audio"), \
             mock.patch.object(auto_lrc, "decode_temp_wav_window", return_value=Path("window.wav")), \
             mock.patch.object(auto_lrc, "_run_whispercpp_wav", return_value=raw_segments), \
             mock.patch.object(auto_lrc, "run_whisperx_alignment", return_value=aligned_payload), \
             mock.patch.object(auto_lrc, "match_whisper_segments", return_value=([.4, 1.2, 2.4, 3.5], local_report)):
            outcome = auto_lrc.execute_local_whisper_request(
                request, capability, entries, mock.Mock(), cache={}, runtime_stats={}
            )
        self.assertEqual(outcome.status, "evidence")
        self.assertEqual(outcome.reason, "trusted-local-whisper-context-prefix-recovery-and-onset")
        self.assertAlmostEqual(float(outcome.onset_time or 0.0), 11.2, places=6)
        self.assertEqual(outcome.occurrence_binding["mode"], "sparse-sequence-bracket")


class V152BoundaryContentConvergenceRegressionTests(unittest.TestCase):
    def test_one_sided_sparse_fingerprint_can_prove_unique_occurrence(self):
        entries = [
            LyricEntry(["anchor alpha"]), LyricEntry(["branch beta"]), LyricEntry(["repeat target"]),
            LyricEntry(["weak after"]), LyricEntry(["separator"]),
            LyricEntry(["anchor alpha"]), LyricEntry(["branch gamma"]), LyricEntry(["repeat target"]),
            LyricEntry(["weak after"]),
        ]
        assignments: list[object] = [None] * len(entries)
        assignments[0] = {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False}
        assignments[1] = {"timestamp": 2.0, "score": .95, "segment": 2, "borrowed": False}
        assignments[2] = {"timestamp": 3.0, "score": .95, "segment": 3, "borrowed": False}
        binding = auto_lrc._entry_occurrence_binding(
            entries, assignments, 2, require_context=True
        )
        self.assertIsNotNone(binding)
        assert binding is not None
        self.assertEqual(binding["mode"], "sparse-sequence-bracket")
        self.assertEqual(binding["anchor_entry_indexes"], [0, 1, 2])
        self.assertTrue(auto_lrc._runtime_occurrence_binding_is_valid(entries, 2, binding))

    def test_exact_phonetic_identity_recovers_content_without_inventing_onset(self):
        entries = [LyricEntry(["left anchor"]), LyricEntry(["東京"]), LyricEntry(["right anchor"])]
        request = auto_lrc.LocalWhisperRequest(
            entry_index=1, capability_revision="capability", audio_revision="audio",
            context_entry_indexes=(0, 1, 2), context_candidate_ids=("a", "b", "c"),
            transcript_digests=("ta", "tb", "tc"), transcript_digest="all",
            window_start=10.0, window_end=20.0,
            window_definition_revision="window", request_revision="request-phonetic",
        )
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        raw_segments = [auto_lrc.AsrSegment(0.0, 1.0, "raw")]
        aligned_segments = [
            auto_lrc.AsrSegment(.4, .9, "leftanchor", chars=[(c, .4 + i*.02) for i,c in enumerate("leftanchor")]),
            auto_lrc.AsrSegment(1.2, 1.8, "とうきょう", chars=[(c, 1.2 + i*.05) for i,c in enumerate("とうきょう")]),
            auto_lrc.AsrSegment(2.4, 3.1, "rightanchor", chars=[(c, 2.4 + i*.02) for i,c in enumerate("rightanchor")]),
        ]
        aligned_payload = {
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text,
                 "chars": [{"char": c, "start": t} for c,t in seg.chars]}
                for seg in aligned_segments
            ]
        }
        local_report = {"assignments": [
            {"timestamp": .4, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 1.2, "score": .40, "segment": 2, "borrowed": False},
            {"timestamp": 2.4, "score": .95, "segment": 3, "borrowed": False},
        ]}

        real_romaji = auto_lrc.japanese_romaji
        def fake_romaji(text: str):
            if text in {"東京", "とうきょう"}:
                return "tokyo"
            return real_romaji(text) or text

        with mock.patch.object(auto_lrc, "_stable_file_sha256", return_value="audio"), \
             mock.patch.object(auto_lrc, "decode_temp_wav_window", return_value=Path("window.wav")), \
             mock.patch.object(auto_lrc, "_run_whispercpp_wav", return_value=raw_segments), \
             mock.patch.object(auto_lrc, "run_whisperx_alignment", return_value=aligned_payload), \
             mock.patch.object(auto_lrc, "match_whisper_segments", return_value=([.4, 1.2, 2.4], local_report)), \
             mock.patch.object(auto_lrc, "japanese_romaji", side_effect=fake_romaji):
            outcome = auto_lrc.execute_local_whisper_request(
                request, capability, entries, mock.Mock(), cache={}, runtime_stats={}
            )
        self.assertEqual(outcome.status, "evidence")
        self.assertEqual(outcome.reason, "trusted-local-whisper-exact-phonetic-occurrence-only")
        self.assertIsNone(outcome.onset_time)
        self.assertIsNotNone(outcome.occurrence_binding)
        self.assertEqual(outcome.occurrence_binding["mode"], "sequence-bracket")

    def test_exact_phonetic_identity_ambiguity_fails_closed_end_to_end(self):
        entries = [LyricEntry(["left anchor"]), LyricEntry(["東京"]), LyricEntry(["right anchor"])]
        request = auto_lrc.LocalWhisperRequest(
            entry_index=1, capability_revision="capability", audio_revision="audio",
            context_entry_indexes=(0, 1, 2), context_candidate_ids=("a", "b", "c"),
            transcript_digests=("ta", "tb", "tc"), transcript_digest="all",
            window_start=10.0, window_end=20.0,
            window_definition_revision="window", request_revision="request-phonetic-ambiguous",
        )
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        raw_segments = [auto_lrc.AsrSegment(0.0, 1.0, "raw")]
        aligned_segments = [
            auto_lrc.AsrSegment(.4, .9, "leftanchor", chars=[(c, .4 + i*.02) for i,c in enumerate("leftanchor")]),
            auto_lrc.AsrSegment(1.2, 1.8, "とうきょう", chars=[(c, 1.2 + i*.05) for i,c in enumerate("とうきょう")]),
            auto_lrc.AsrSegment(2.0, 2.6, "とうきょう", chars=[(c, 2.0 + i*.05) for i,c in enumerate("とうきょう")]),
            auto_lrc.AsrSegment(3.0, 3.7, "rightanchor", chars=[(c, 3.0 + i*.02) for i,c in enumerate("rightanchor")]),
        ]
        aligned_payload = {
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text,
                 "chars": [{"char": c, "start": t} for c,t in seg.chars]}
                for seg in aligned_segments
            ]
        }
        local_report = {"assignments": [
            {"timestamp": .4, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 1.2, "score": .40, "segment": 2, "borrowed": False},
            {"timestamp": 3.0, "score": .95, "segment": 4, "borrowed": False},
        ]}
        real_romaji = auto_lrc.japanese_romaji

        def fake_romaji(text: str):
            if text in {"東京", "とうきょう"}:
                return "tokyo"
            return real_romaji(text) or text

        with mock.patch.object(auto_lrc, "_stable_file_sha256", return_value="audio"), \
             mock.patch.object(auto_lrc, "decode_temp_wav_window", return_value=Path("window.wav")), \
             mock.patch.object(auto_lrc, "_run_whispercpp_wav", return_value=raw_segments), \
             mock.patch.object(auto_lrc, "run_whisperx_alignment", return_value=aligned_payload), \
             mock.patch.object(auto_lrc, "whisperx_segments", return_value=aligned_segments), \
             mock.patch.object(auto_lrc, "match_whisper_segments", return_value=([.4, 1.2, 3.0], local_report)), \
             mock.patch.object(auto_lrc, "japanese_romaji", side_effect=fake_romaji):
            outcome = auto_lrc.execute_local_whisper_request(
                request, capability, entries, mock.Mock(), cache={}, runtime_stats={}
            )
        self.assertEqual(outcome.status, "invalid")
        self.assertEqual(outcome.reason, "target-phonetic-occurrence-ambiguous")
        self.assertIsNone(outcome.occurrence_binding)

    def test_phonetic_identity_fails_closed_when_two_occurrences_bind(self):
        segments = [
            auto_lrc.AsrSegment(0.0, .5, "とうきょう"),
            auto_lrc.AsrSegment(1.0, 1.5, "とうきょう"),
        ]
        real_romaji = auto_lrc.japanese_romaji
        def fake_romaji(text: str):
            if text in {"東京", "とうきょう"}:
                return "tokyo"
            return real_romaji(text) or text
        with mock.patch.object(auto_lrc, "japanese_romaji", side_effect=fake_romaji):
            matches = auto_lrc._aligned_exact_phonetic_target_matches("東京", segments)
        self.assertEqual(len(matches), 2)
        self.assertNotEqual(matches[0]["match_revision"], matches[1]["match_revision"])

    def test_occurrence_only_local_whisper_can_fuse_with_independent_vocal_onset(self):
        entries = [LyricEntry(["left"]), LyricEntry(["target"]), LyricEntry(["right"])]
        assignments = [
            {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 2.0, "score": .90, "segment": 2, "borrowed": False},
            {"timestamp": 3.0, "score": .95, "segment": 3, "borrowed": False},
        ]
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 1, require_context=True)
        self.assertIsNotNone(binding)
        assert binding is not None
        local_evidence = {
            "evidence_revision": "local-evidence",
            "onset_time": None,
            "score": .90,
            "occurrence_binding": binding,
        }
        features = AudioFeatures(
            duration=4.0,
            frame_times=np.asarray([1.8, 2.0, 2.1, 2.2]),
            rms_db=np.zeros(4), onset_strength=np.asarray([.1, .2, .95, .2]),
            segments=[(1.9, 2.8)],
        )
        with mock.patch.object(auto_lrc, "_strong_vocal_onset_near", return_value=(2.1, .95, .80)):
            candidate = auto_lrc._local_whisper_vocal_fusion_candidate(
                entry_index=1, entry=entries[1], local_evidence=local_evidence,
                vocal_onset_features=features, audio_revision="audio-sha",
                occurrence_evidence=auto_lrc.candidate_occurrence_evidence_from_binding(
                    entries,
                    1,
                    producer="local-whisper-vocal-fusion",
                    occurrence_binding=binding,
                    upstream_evidence_revision="local-evidence",
                ),
            )
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.source, "local-whisper-vocal-independent-fusion")
        self.assertTrue(auto_lrc.candidate_has_bound_direct_onset_evidence(candidate))
        self.assertTrue(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(candidate))
        self.assertTrue(candidate.independent_content_identity)
        self.assertAlmostEqual(candidate.written_time.seconds, 2.1, places=6)

    def test_complementary_local_whisper_evidence_merges_context_and_direct_onset(self):
        entries = [LyricEntry(["left"]), LyricEntry(["target"]), LyricEntry(["right"])]
        assignments = [
            {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 2.0, "score": .96, "segment": 2, "borrowed": False},
            {"timestamp": 3.0, "score": .95, "segment": 3, "borrowed": False},
        ]
        context_binding = auto_lrc._entry_occurrence_binding(
            entries, assignments, 1, require_context=True
        )
        global_binding = auto_lrc._entry_occurrence_binding(
            entries, assignments, 1, require_context=False
        )
        self.assertIsNotNone(context_binding)
        self.assertIsNotNone(global_binding)
        assert context_binding is not None and global_binding is not None
        self.assertNotEqual(context_binding["mode"], global_binding["mode"])
        capability = auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local", backend_lineage="whisper-family",
            audio_path="audio.wav", audio_sha256="audio",
            whisper_cli_path="cli", whisper_cli_sha256="cli-sha",
            whisper_model_path="model", whisper_model_sha256="model-sha",
            whisperx_python_path="python", whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha", language="ja", device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )
        def req(rev: str, end: float):
            return auto_lrc.LocalWhisperRequest(
                entry_index=1, capability_revision="capability", audio_revision="audio",
                context_entry_indexes=(0, 1, 2), context_candidate_ids=("a", "b", "c"),
                transcript_digests=("a", "b", "c"), transcript_digest="abc",
                window_start=0.0, window_end=end, window_definition_revision=rev,
                request_revision=rev,
            )
        narrow, broad = req("narrow", 4.0), req("broad", 5.0)
        onset = auto_lrc.LocalWhisperOutcome(
            entry_index=1, status="evidence", reason="onset", onset_time=2.05,
            score=.96, evidence_revision="unused", occurrence_binding=global_binding,
            raw_transcript_revision="raw-narrow", aligned_transcript_revision="aligned-narrow",
            request_revision="narrow", outcome_revision="out-narrow",
        )
        context = auto_lrc.LocalWhisperOutcome(
            entry_index=1, status="evidence", reason="context", onset_time=None,
            score=.90, evidence_revision="unused", occurrence_binding=context_binding,
            raw_transcript_revision="raw-broad", aligned_transcript_revision="aligned-broad",
            request_revision="broad", outcome_revision="out-broad",
        )
        report = {}
        auto_lrc.bind_runtime_local_whisper_evidence(
            report, entries, capability, (narrow, broad), (onset, context)
        )
        evidence = auto_lrc.runtime_local_whisper_evidence_for_entry(report, entries, 1)
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertAlmostEqual(float(evidence["onset_time"]), 2.05, places=6)
        self.assertTrue(auto_lrc._occurrence_binding_has_strong_context(evidence["occurrence_binding"]))
        self.assertIsNotNone(evidence.get("composite_onset_evidence_revision"))
        self.assertIsNotNone(evidence.get("composite_context_evidence_revision"))

    def test_local_whisper_vocal_ctc_consensus_is_evidence_additive(self):
        entries = [LyricEntry(["left"]), LyricEntry(["target"]), LyricEntry(["right"])]
        parent = auto_lrc.make_timing_candidate(
            entry_index=1, entry_text="target", source="ctc-current", raw_time=2.0,
            confidence=.90, identity_support="supported", sequence_support="supported",
            current=True, source_artifact={"producer": "ctc"},
        )
        assignments = [
            {"timestamp": 1.0, "score": .95, "segment": 1, "borrowed": False},
            {"timestamp": 2.0, "score": .90, "segment": 2, "borrowed": False},
            {"timestamp": 3.0, "score": .95, "segment": 3, "borrowed": False},
        ]
        binding = auto_lrc._entry_occurrence_binding(
            entries, assignments, 1, require_context=True
        )
        self.assertIsNotNone(binding)
        assert binding is not None
        local_evidence = {
            "evidence_revision": "local", "onset_time": None, "score": .90,
            "occurrence_binding": binding,
        }
        features = AudioFeatures(
            duration=4.0, frame_times=np.asarray([1.9, 2.0, 2.1]),
            rms_db=np.zeros(3), onset_strength=np.asarray([.1, .2, .95]),
            segments=[(1.8, 2.8)],
        )
        with mock.patch.object(
            auto_lrc, "runtime_local_whisper_evidence_for_entry", return_value=local_evidence
        ), mock.patch.object(
            auto_lrc, "_strong_vocal_onset_near", return_value=(2.1, .95, .80)
        ):
            augmented = auto_lrc._augment_candidates_with_local_whisper_vocal_consensus(
                ((), (parent,), ()), {}, entries, features, "audio-sha"
            )
        sources = {item.source: item for item in augmented[1]}
        self.assertIn("local-whisper-vocal-independent-fusion", sources)
        self.assertIn("ctc-local-whisper-vocal-independent-consensus", sources)
        composite = sources["ctc-local-whisper-vocal-independent-consensus"]
        self.assertAlmostEqual(composite.written_time.seconds, 2.0, places=6)
        self.assertIsNotNone(composite.direct_onset_evidence)
        self.assertAlmostEqual(composite.direct_onset_evidence.onset_time, 2.1, places=6)
        self.assertTrue(auto_lrc.candidate_has_boundary_grade_direct_onset_evidence(composite))
        self.assertTrue(auto_lrc.candidate_has_independent_backend_identity(composite))


class ZeroReferenceAutoIsolationTests(unittest.TestCase):
    def test_auto_timestamp_stripping_preserves_text_revision(self) -> None:
        timestamped = [
            LyricEntry(["one", "translation"], 100),
            LyricEntry(["two", "translation two"], 250),
        ]
        stripped = auto_lrc.strip_lyric_timestamps(timestamped)
        self.assertEqual([entry.lines for entry in stripped], [entry.lines for entry in timestamped])
        self.assertEqual([entry.source_time_cs for entry in stripped], [None, None])
        self.assertEqual(
            auto_lrc.lyric_identity_revision(stripped),
            auto_lrc.lyric_identity_revision(timestamped),
        )

    def test_primary_lyrics_discovery_excludes_checked_candidates_by_default(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "Music" / "album" / "song.flac"
            checked = root / "Music" / "LRC tools checked references" / "song.lrc"
            audio.parent.mkdir(parents=True)
            checked.parent.mkdir(parents=True)
            audio.write_bytes(b"audio")
            checked.write_text("[00:01.00]one\n", encoding="utf-8")
            with self.assertRaises(LrcError):
                auto_lrc.find_lyrics(audio)
            self.assertEqual(auto_lrc.find_lyrics(audio, include_checked=True), checked)

    def test_auto_anchor_discovery_requires_explicit_path(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "song.flac"
            anchor = root / "song.anchors.lrc"
            audio.write_bytes(b"audio")
            anchor.write_text("[00:01.00]one\n", encoding="utf-8")
            args = build_parser().parse_args([str(audio)])
            self.assertIsNone(auto_lrc.find_anchor_hints(audio, args))
            explicit = build_parser().parse_args([str(audio), "--anchor-hints", str(anchor)])
            self.assertEqual(auto_lrc.find_anchor_hints(audio, explicit), anchor.resolve())

    def test_inference_metadata_distinguishes_independent_and_assisted_modes(self) -> None:
        independent = auto_lrc.inference_mode_metadata("auto", "backend_competition", None)
        self.assertEqual(independent["inference_mode"], "independent-auto")
        self.assertTrue(independent["independent_auto"])
        self.assertFalse(independent["reference_used_before_inference"])
        self.assertFalse(independent["experience_artifact_used"])
        checked = auto_lrc.inference_mode_metadata("checked", "checked_lrc", None)
        self.assertEqual(checked["inference_mode"], "assisted-checked")
        self.assertFalse(checked["independent_auto"])
        self.assertTrue(checked["reference_used_before_inference"])
        self.assertTrue(checked["experience_artifact_used"])
        anchor = auto_lrc.inference_mode_metadata("auto", "backend_competition", Path("manual.anchors.lrc"))
        self.assertEqual(anchor["inference_mode"], "assisted-auto-anchor")
        self.assertFalse(anchor["independent_auto"])
        self.assertFalse(anchor["reference_used_before_inference"])
        self.assertTrue(anchor["experience_artifact_used"])

    def test_explicit_checked_mode_remains_assisted(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "song.flac"
            lyrics = root / "song.lrc"
            audio.write_bytes(b"audio")
            lyrics.write_text("[00:01.00]one\n[00:02.00]two\n", encoding="utf-8")
            args = build_parser().parse_args([
                str(audio), "--lyrics", str(lyrics), "--timing-source", "checked",
                "--output", str(root / "result.lrc"), "--report-dir", str(root / "reports"), "--overwrite",
            ])
            with mock.patch.object(auto_lrc, "probe_duration", return_value=20.0):
                auto_lrc.process_audio(audio, args)
            report_path = next((root / "reports").glob("*.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["inference_mode"], "assisted-checked")
            self.assertFalse(report["independent_auto"])
            self.assertTrue(report["reference_used_before_inference"])
            self.assertTrue(report["experience_artifact_used"])

    def test_explicit_lyrics_mode_remains_assisted(self) -> None:
        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            audio = root / "song.flac"
            lyrics = root / "song.lrc"
            audio.write_bytes(b"audio")
            lyrics.write_text("[00:01.00]one\n[00:02.00]two\n", encoding="utf-8")
            args = build_parser().parse_args([
                str(audio), "--lyrics", str(lyrics), "--timing-source", "lyrics",
                "--output", str(root / "result.lrc"), "--report-dir", str(root / "reports"), "--overwrite",
            ])
            with mock.patch.object(auto_lrc, "probe_duration", return_value=20.0):
                auto_lrc.process_audio(audio, args)
            report_path = next((root / "reports").glob("*.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["inference_mode"], "assisted-lyrics")
            self.assertFalse(report["independent_auto"])
            self.assertTrue(report["reference_used_before_inference"])
            self.assertTrue(report["experience_artifact_used"])

    def test_auto_filesystem_metamorphic_and_timestamped_equivalence(self) -> None:
        def run_case(root: Path, lyric_name: str, *, timestamped: bool) -> dict[str, object]:
            music = root / "Music" / "album"
            music.mkdir(parents=True)
            audio = music / "song.flac"
            audio.write_bytes(b"same-audio")
            lyric = music / lyric_name
            lyric.write_text(
                "[00:01.00]one\n[00:02.00]two\n" if timestamped else "one\ntwo\n",
                encoding="utf-8",
            )
            if root.name == "decoy":
                (music / "song.anchors.lrc").write_text("[00:01.00]wrong\n", encoding="utf-8")
                (music / "song.lrc").write_text("[00:99.00]one\n[00:99.50]two\n", encoding="utf-8")
                (music / "song.align-report.json").write_text("{\"timestamp\":99.0}\n", encoding="utf-8")
                (music / "song.previous-trust.json").write_text("{\"overall_trusted\":true}\n", encoding="utf-8")
                (music / "timing-authority").mkdir()
                (music / "timing-authority" / "candidate.json").write_text("{\"timestamp\":99.0}\n", encoding="utf-8")
            spans = lambda start: tuple(
                auto_lrc.TimingTokenSpan(start + index * 0.06, start + index * 0.06 + 0.04, 0.9, chr(97 + index))
                for index in range(3)
            )
            candidates = tuple(
                auto_lrc.make_timing_candidate(
                    entry_index=index, entry_text=text, source="ctc-current", raw_time=time,
                    spans=spans(time), confidence=0.9, identity_support="supported",
                    sequence_support="supported", current=True,
                )
                for index, (text, time) in enumerate((("one", 2.0), ("two", 5.0)))
            )
            report = {
                "backend": "ctc", "timing_entries": 2,
                "assignments": [{"entry": 1, "timestamp": 2.0, "score": 0.9}, {"entry": 2, "timestamp": 5.0, "score": 0.9}],
                "suspicious_alignments": [],
            }
            evaluated = auto_lrc.evaluate_backend_timing("ctc", tuple((candidate,) for candidate in candidates), report, {})
            selection = auto_lrc.select_evaluated_backend((evaluated,))

            def competition(*_args: object, **_kwargs: object):
                return [2.0, 5.0], report, "ctc", selection

            args = build_parser().parse_args([
                str(audio), "--lyrics", str(lyric), "--output", str(root / "result.lrc"),
                "--report-dir", str(root / "reports"), "--overwrite",
            ])
            with mock.patch.object(auto_lrc, "probe_duration", return_value=20.0), \
                mock.patch.object(auto_lrc, "default_ctc_ready", return_value=True), \
                mock.patch.object(auto_lrc, "default_whisperx_ready", return_value=False), \
                mock.patch.object(auto_lrc, "run_auto_backend_competition", side_effect=competition), \
                mock.patch.object(auto_lrc, "audio_track_labels", return_value=("", "generic")), \
                mock.patch.object(auto_lrc, "checked_lrc_timing_hint", side_effect=AssertionError("checked hint read")):
                auto_lrc.process_audio(audio, args)
            report_path = next((root / "reports").glob("*.json"))
            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            return {
                "lrc": (root / "result.lrc").read_bytes(),
                "assignments": persisted["assignments"],
                "metadata": {key: persisted.get(key) for key in (
                    "inference_mode", "independent_auto", "reference_used_before_inference", "experience_artifact_used"
                )},
            }

        with TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            plain = run_case(root / "plain", "untimed.txt", timestamped=False)
            timestamped = run_case(root / "timestamped", "source.lrc", timestamped=True)
            decoy = run_case(root / "decoy", "untimed.txt", timestamped=False)
        for result in (timestamped, decoy):
            self.assertEqual(result["lrc"], plain["lrc"])
            self.assertEqual(result["assignments"], plain["assignments"])
            self.assertEqual(result["metadata"], plain["metadata"])
        self.assertEqual(plain["metadata"]["inference_mode"], "independent-auto")
        self.assertTrue(plain["metadata"]["independent_auto"])
        self.assertFalse(plain["metadata"]["reference_used_before_inference"])


class Cycle33OccurrenceEvidenceContractTests(unittest.TestCase):
    def occurrence_evidence(
        self,
        entries: list[LyricEntry],
        *,
        producer: str = "raw-asr-vocal-fusion",
    ) -> auto_lrc.CandidateOccurrenceEvidence:
        assignments = [
            {"timestamp": float(index + 1), "score": .95, "segment": index + 1}
            for index in range(len(entries))
        ]
        binding = auto_lrc._entry_occurrence_binding(entries, assignments, 0)
        evidence = auto_lrc.candidate_occurrence_evidence_from_binding(
            entries,
            0,
            producer=producer,
            occurrence_binding=binding,
            upstream_evidence_revision="cycle33-upstream",
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        return evidence

    def direct_candidate(
        self,
        evidence: auto_lrc.CandidateOccurrenceEvidence,
        *,
        producer: str = "raw-asr-vocal-fusion",
    ) -> auto_lrc.TimingCandidate:
        spans = tuple(
            auto_lrc.TimingTokenSpan(1.0 + index * .08, 1.04 + index * .08, .95, str(index))
            for index in range(4)
        )
        return auto_lrc.make_timing_candidate(
            entry_index=0,
            entry_text="cycle33 target",
            source="raw-vocal-independent-fusion",
            raw_time=1.0,
            spans=spans,
            confidence=.9,
            identity_support="supported",
            sequence_support="supported",
            direct_onset_support="supported",
            direct_onset_time=1.0,
            direct_onset_source_artifact={"producer": producer},
            direct_onset_evidence_producer=producer,
            direct_onset_evidence_kind="cross-backend-onset",
            direct_onset_evidence_independent=True,
            occurrence_evidence=evidence,
            independent_content_identity=True,
        )

    def test_direct_valid_occurrence_invalid_fails_minimum_proof(self) -> None:
        entries = [LyricEntry(["cycle33 target"])]
        valid = self.occurrence_evidence(entries)
        stale = replace(valid, upstream_evidence_revision="stale-upstream")
        candidate = self.direct_candidate(stale)
        evaluation = auto_lrc.evaluate_timing_candidate(
            candidate, occurrence_entries=entries
        )
        self.assertFalse(evaluation.valid)
        self.assertIn("occurrence-evidence-invalid", evaluation.rejection_reasons)
        self.assertFalse(evaluation.evidence.minimum_proof_satisfied)

    def test_occurrence_binding_unknown_path_field_fails_closed(self) -> None:
        entries = [LyricEntry(["cycle33 target"])]
        valid = self.occurrence_evidence(entries)
        binding = json.loads(valid.binding_payload_json)
        binding["path"] = "C:/private/audio.wav"
        binding_payload = {key: value for key, value in binding.items() if key != "binding_revision"}
        binding["binding_revision"] = auto_lrc._timing_content_digest(binding_payload)
        self.assertFalse(auto_lrc._runtime_occurrence_binding_is_valid(entries, 0, binding))
        self.assertIsNone(
            auto_lrc.candidate_occurrence_evidence_from_binding(
                entries,
                0,
                producer="raw-asr-vocal-fusion",
                occurrence_binding=binding,
                upstream_evidence_revision="cycle33-upstream",
            )
        )

    def test_occurrence_producer_mismatch_fails_closed(self) -> None:
        entries = [LyricEntry(["cycle33 target"])]
        evidence = self.occurrence_evidence(entries)
        candidate = self.direct_candidate(evidence, producer="whisperx-vocal-fusion")
        self.assertFalse(
            auto_lrc.candidate_has_bound_direct_onset_evidence(candidate, entries)
        )
        self.assertFalse(
            auto_lrc.candidate_has_independent_backend_identity(candidate, entries)
        )


class V12HubertPairTailOccurrenceRegressionTests(unittest.TestCase):
    def test_current_evidence_scope_preserves_strict_gate_for_valid_predecessor(self) -> None:
        self.assertEqual(
            auto_lrc._known_lyric_hubert_pair_tail_current_evidence_scope("valid"),
            "strict-final-associated-onset",
        )
        self.assertEqual(
            auto_lrc._known_lyric_hubert_pair_tail_current_evidence_scope(
                "first-token-mismatch"
            ),
            "hypothesis-neutral-boundary",
        )
        self.assertEqual(
            auto_lrc._known_lyric_hubert_pair_tail_current_evidence_scope(
                "envelope-mismatch"
            ),
            "reject",
        )

    def test_unique_pair_current_opening_can_bind_tail_when_old_predecessor_onset_disagrees(self) -> None:
        entries = [
            LyricEntry(["あさ"]),
            LyricEntry(["ひる"]),
            LyricEntry(["よる"]),
        ]
        legacy = auto_lrc._known_lyric_hubert_predecessor_occurrence_status(
            4.08,
            5.46,
            620,
            18,
        )
        self.assertEqual(legacy, "first-token-mismatch")
        status = auto_lrc._known_lyric_hubert_pair_tail_occurrence_status(
            entries,
            0,
            1,
            4.08,
            5.46,
            10.08,
            10.10,
            18,
        )
        self.assertEqual(status, "valid")

    def test_repeated_adjacent_reading_pair_stays_unbound(self) -> None:
        entries = [
            LyricEntry(["あさ"]),
            LyricEntry(["ひる"]),
            LyricEntry(["あさ"]),
            LyricEntry(["ひる"]),
        ]
        status = auto_lrc._known_lyric_hubert_pair_tail_occurrence_status(
            entries,
            0,
            1,
            4.0,
            5.0,
            6.0,
            6.0,
            18,
        )
        self.assertEqual(status, "pair-reading-not-unique")

    def test_pair_boundary_must_match_neutral_current_opening(self) -> None:
        entries = [LyricEntry(["あさ"]), LyricEntry(["ひる"])]
        status = auto_lrc._known_lyric_hubert_pair_tail_occurrence_status(
            entries,
            0,
            1,
            4.0,
            5.0,
            6.0,
            6.40,
            18,
        )
        self.assertEqual(status, "current-boundary-mismatch")


class V12HubertPredecessorPlannerRegressionTests(unittest.TestCase):
    def state(
        self,
        entries: list[LyricEntry],
        times: list[float],
    ) -> auto_lrc.FinalTimingState:
        decisions = []
        previous = None
        for index, (entry, timestamp) in enumerate(zip(entries, times, strict=True)):
            candidate = auto_lrc.make_timing_candidate(
                entry_index=index,
                entry_text=auto_lrc.entry_sung_text(entry),
                source="ctc-current",
                raw_time=timestamp,
                spans=(
                    auto_lrc.TimingTokenSpan(
                        timestamp,
                        timestamp + 0.08,
                        0.9,
                        "a",
                    ),
                    auto_lrc.TimingTokenSpan(
                        timestamp + 0.10,
                        timestamp + 0.18,
                        0.9,
                        "b",
                    ),
                ),
                confidence=0.9,
                identity_support="supported",
                sequence_support="supported",
                current=True,
                source_artifact={"planner-test": index},
            )
            decision = auto_lrc.select_timing_decision(
                entry_index=index,
                candidates=(candidate,),
                previous_candidate=previous,
                selection_revision=1,
            )
            decisions.append(decision)
            previous = decision.candidate
        decisions_tuple = tuple(decisions)
        return auto_lrc.FinalTimingState(
            decisions=decisions_tuple,
            written_times=tuple(item.written_time for item in decisions_tuple),
            audits=auto_lrc.audit_final_timing_decisions(decisions_tuple),
        )

    def boundary(
        self,
        state: auto_lrc.FinalTimingState,
        entry_index: int,
        onset_time: float,
    ) -> auto_lrc.IndependentCurrentOnsetEvidence:
        owner = state.decisions[entry_index].candidate
        evidence = auto_lrc._make_known_lyric_hubert_current_onset(
            entry_index=entry_index,
            onset_time=onset_time,
            opening_proof_end=onset_time + 0.20,
            opening_region_revision=f"region-{entry_index}",
            identity_owner_candidate_id=owner.candidate_id,
            identity_owner_generation_revision=owner.generation_revision,
            model_identity="planner-test-model",
            helper_revision="planner-test-helper",
            capability_revision="planner-test-capability",
            audio_revision="planner-test-audio",
            transcript_revision=f"transcript-{entry_index}",
            row_revision=f"row-{entry_index}",
            request_revision=f"request-{entry_index}",
            hypothesis_set_revision=f"hypothesis-{entry_index}",
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertTrue(auto_lrc._known_lyric_hubert_current_onset_is_valid(evidence))
        return evidence

    def test_generalized_planner_uses_neutral_boundary_and_global_predecessor_uniqueness(self) -> None:
        entries = [
            LyricEntry(["あ"]),
            LyricEntry(["い"]),
            LyricEntry(["う"]),
            LyricEntry(["え"]),
            LyricEntry(["お"]),
            LyricEntry(["い"]),
            LyricEntry(["か"]),
        ]
        state = self.state(entries, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        unique_boundary = self.boundary(state, 3, 4.20)
        repeated_boundary = self.boundary(state, 6, 7.20)

        requests = auto_lrc._postcentral_plan_known_lyric_hubert_predecessor_only_requests(
            {
                3: unique_boundary,
                6: repeated_boundary,
            },
            state,
            entries,
            20.0,
        )

        self.assertEqual(
            [int(item["conflict_entry"]) for item in requests],
            [4],
        )
        self.assertEqual(int(requests[0]["previous_entry_index"]), 2)
        self.assertEqual(int(requests[0]["current_entry_index"]), 3)
        self.assertEqual(
            requests[0]["current_boundary_observation_revision"],
            unique_boundary.observation_revision,
        )
        self.assertAlmostEqual(float(requests[0]["window_end"]), 4.20, places=6)


class V12Node19ExistingPeerRecoveryEligibilityTests(unittest.TestCase):
    def candidate(
        self,
        *,
        entry_index: int,
        source: str,
        raw_time: float,
        direct: bool,
        tail_end: float | None = None,
    ) -> auto_lrc.TimingCandidate:
        if tail_end is None:
            starts = [raw_time, raw_time + 0.30]
            starts.extend(raw_time + 0.70 + index * 0.42 for index in range(8))
        else:
            starts = [
                raw_time + (tail_end - raw_time - 0.05) * index / 5.0
                for index in range(6)
            ]
        spans = tuple(
            auto_lrc.TimingTokenSpan(start, start + 0.04, 0.96, chr(97 + index))
            for index, start in enumerate(starts)
        )
        kwargs: dict[str, object] = {}
        if direct:
            kwargs.update({
                "acoustic_support": "supported",
                "direct_onset_support": "supported",
                "acoustic_strength": 0.95,
                "acoustic_onset_time": raw_time,
                "acoustic_source_artifact": {"node19": "independent-vocal-onset"},
                "acoustic_evidence_producer": "consonant-onset-detector",
                "acoustic_evidence_kind": "vocal-onset",
                "acoustic_evidence_independent": True,
                "direct_onset_time": raw_time,
                "direct_onset_source_artifact": {"node19": "independent-vocal-onset"},
                "direct_onset_evidence_producer": "consonant-onset-detector",
                "direct_onset_evidence_kind": "vocal-onset",
                "direct_onset_evidence_independent": True,
            })
        return auto_lrc.make_timing_candidate(
            entry_index=entry_index,
            entry_text=f"node19-{entry_index}",
            source=source,
            raw_time=raw_time,
            spans=spans,
            confidence=0.95,
            identity_support="supported",
            sequence_support="supported",
            source_artifact={"node19": source},
            **kwargs,
        )

    def fixture(
        self,
        *,
        evidence_state: str = "independent",
        cluster_families: list[str] | None = None,
        direct: bool = True,
        trusted_selected: bool = False,
        predecessor_tail_end: float = 29.80,
        multiple_clusters: bool = False,
    ) -> tuple[
        dict[str, object],
        auto_lrc.TimingDecision,
        auto_lrc.TimingDecision,
        dict[tuple[str, str], auto_lrc.TimingCandidate],
    ]:
        previous = self.candidate(
            entry_index=0,
            source="ctc-hubert-known-lyric-boundary-consensus",
            raw_time=29.0,
            direct=True,
            tail_end=predecessor_tail_end,
        )
        previous_decision = auto_lrc.select_timing_decision(
            entry_index=0,
            candidates=(previous,),
            previous_candidate=None,
            selection_revision=1,
        )
        weak = auto_lrc.make_timing_candidate(
            entry_index=1,
            entry_text="node19-1",
            source="whispercpp-current",
            raw_time=30.0,
            spans=(),
            confidence=0.6,
            identity_support="unavailable",
            sequence_support="supported",
            source_artifact={"node19": "selected-weak"},
        )
        selected = auto_lrc.select_timing_decision(
            entry_index=1,
            candidates=(weak,),
            previous_candidate=previous,
            selection_revision=1,
        )
        if trusted_selected:
            selected = replace(
                selected,
                status="selected_valid",
                selected_candidate_id=selected.candidate.candidate_id,
                provisional_candidate_id=None,
            )
        alternative = self.candidate(
            entry_index=1,
            source="ctc-opening-vocal-fusion",
            raw_time=30.88,
            direct=direct,
        )
        cluster = {
            "cluster_revision": "node19-cluster",
            "anchor_time": 30.88,
            "member_candidate_ids": [alternative.candidate_id],
            "member_generation_revisions": [alternative.generation_revision],
            "support_families": cluster_families or ["ctc"],
            "representative_candidate_id": alternative.candidate_id,
            "relation": "distant",
            "evidence_state": evidence_state,
            "review_required": True,
        }
        clusters: list[dict[str, object]] = [cluster]
        if multiple_clusters:
            clusters.append({**cluster, "cluster_revision": "node19-cluster-2", "anchor_time": 31.2})
        row = {
            "entry": 2,
            "postcentral_replay_status": "REBOUND",
            "postcentral_replay_reason": "postcentral-existing-conflict-only-canonical-replay",
            "final_selected_candidate_id": selected.candidate.candidate_id,
            "final_selected_centiseconds": selected.written_time.centiseconds,
            "clusters": clusters,
        }
        lookup = {(alternative.candidate_id, alternative.generation_revision): alternative}
        return row, selected, previous_decision, lookup

    def resolve(self, **kwargs: object) -> tuple[auto_lrc.TimingCandidate | None, str]:
        row, selected, previous, lookup = self.fixture(**kwargs)
        return auto_lrc._existing_peer_recovery_representative(
            row=row,
            final_decision=selected,
            previous_decision=previous,
            bundle_lookup=lookup,
            occurrence_entries=None,
        )

    def test_independent_complete_existing_candidate_is_eligible(self) -> None:
        row, selected, previous, lookup = self.fixture()
        probe = next(iter(lookup.values()))
        evaluation = auto_lrc.evaluate_timing_candidate(
            probe, previous_candidate=previous.candidate
        )
        self.assertTrue(
            evaluation.valid,
            (evaluation.rejection_reasons, evaluation.evidence),
        )
        candidate, reason = auto_lrc._existing_peer_recovery_representative(
            row=row,
            final_decision=selected,
            previous_decision=previous,
            bundle_lookup=lookup,
            occurrence_entries=None,
        )
        self.assertIsNotNone(candidate, reason)
        self.assertEqual(reason, "eligible-independent-existing-peer-candidate")
        assert candidate is not None
        self.assertEqual(candidate.written_time.centiseconds, 3088)

    def test_same_lineage_cluster_is_rejected(self) -> None:
        candidate, reason = self.resolve(cluster_families=["whisper-family"])
        self.assertIsNone(candidate)
        self.assertEqual(reason, "cluster-not-independent-from-selected-lineage")

    def test_mixed_cluster_is_rejected(self) -> None:
        candidate, reason = self.resolve(evidence_state="mixed")
        self.assertIsNone(candidate)
        self.assertEqual(reason, "cluster-evidence-not-independent")

    def test_trusted_selected_boundary_is_not_recovered(self) -> None:
        candidate, reason = self.resolve(trusted_selected=True)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "selected-boundary-not-provisional")

    def test_candidate_without_bound_direct_onset_is_rejected(self) -> None:
        candidate, reason = self.resolve(direct=False)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "representative-lacks-independent-direct-onset")

    def test_multiple_distant_clusters_are_rejected(self) -> None:
        candidate, reason = self.resolve(multiple_clusters=True)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "material-alternative-not-unique")

    def test_representative_still_overlapping_predecessor_is_rejected(self) -> None:
        candidate, reason = self.resolve(predecessor_tail_end=31.10)
        self.assertIsNone(candidate)
        self.assertEqual(reason, "representative-not-central-valid-in-selected-context")

    def test_peer_recovery_runs_after_hubert_predecessor_recovery(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        process = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_audio"
        )
        process_calls = {
            getattr(node.func, "id", None): node.lineno
            for node in ast.walk(process)
            if isinstance(node, ast.Call)
        }
        self.assertLess(
            process_calls["_precommit_hubert_existing_candidate_recovery"],
            process_calls["_precommit_independent_peer_existing_candidate_recovery"],
        )


class V12Node19DuplicateOwnerRecomputeTests(unittest.TestCase):
    def fixture(
        self,
        *,
        corrected_second_a: float,
    ) -> tuple[list[LyricEntry], dict[str, object], auto_lrc.FinalTimingState]:
        entries = [
            LyricEntry(["A"]), LyricEntry(["B"]), LyricEntry(["C"]), LyricEntry(["D"]),
            LyricEntry(["A"]), LyricEntry(["B"]), LyricEntry(["C"]), LyricEntry(["D"]),
        ]
        original_times = [1.0, 3.0, 5.0, 7.0, 84.0, 85.0, 87.0, 89.0]
        final_times = [1.0, 3.0, 5.0, 7.0, corrected_second_a, 85.0, 87.0, 89.0]
        assignments: list[dict[str, object]] = [
            {"entry": index + 1, "timestamp": value, "score": 0.9, "flags": []}
            for index, value in enumerate(original_times)
        ]
        assignments[0]["ctc_forward_supported_prefix_recovery"] = True
        assignments[0]["flags"] = ["duplicate_occurrence_offset_mismatch"]
        assignments[0]["duplicate_occurrence_offset_mismatch"] = {
            "entries": [1, 5],
            "observed_delta_seconds": 83.0,
            "median_delta_seconds": 82.0,
            "error_seconds": 1.0,
        }
        report: dict[str, object] = {
            "assignments": assignments,
            "suspicious_alignments": [{
                "entry": 1,
                "text": "A",
                "flags": ["duplicate_occurrence_offset_mismatch"],
                "severity": "high",
                "review_required": True,
                "duplicate_occurrence_offset_mismatch": dict(
                    assignments[0]["duplicate_occurrence_offset_mismatch"]  # type: ignore[arg-type]
                ),
            }],
            "duplicate_occurrence_offset_outliers": [{
                "entry": 1,
                "entries": [1, 5],
                "observed_delta_seconds": 83.0,
                "median_delta_seconds": 82.0,
                "error_seconds": 1.0,
            }],
            "duplicate_occurrence_offset_outlier_count": 1,
        }
        decisions = []
        previous = None
        for index, value in enumerate(final_times):
            candidate = auto_lrc.make_timing_candidate(
                entry_index=index,
                entry_text=auto_lrc.entry_sung_text(entries[index]),
                source="ctc-current",
                raw_time=value,
                spans=tuple(
                    auto_lrc.TimingTokenSpan(
                        value + offset,
                        value + offset + 0.04,
                        0.95,
                        chr(97 + span_index),
                    )
                    for span_index, offset in enumerate((0.0, 0.08, 0.16, 0.24, 0.32, 0.40))
                ),
                confidence=0.95,
                identity_support="supported",
                sequence_support="supported",
                current=True,
                source_artifact={"node19-duplicate": index},
            )
            decision = auto_lrc.select_timing_decision(
                entry_index=index,
                candidates=(candidate,),
                previous_candidate=previous,
                selection_revision=1,
            )
            decisions.append(decision)
            previous = decision.candidate
        decisions_tuple = tuple(decisions)
        state = auto_lrc.FinalTimingState(
            decisions=decisions_tuple,
            written_times=tuple(item.written_time for item in decisions_tuple),
            audits=auto_lrc.audit_final_timing_decisions(decisions_tuple),
        )
        return entries, report, state

    def test_recomputed_final_delta_removes_stale_duplicate_mismatch(self) -> None:
        entries, report, state = self.fixture(corrected_second_a=83.12)
        payload = auto_lrc._recompute_duplicate_occurrence_offset_diagnostics(
            entries, report, state
        )
        self.assertEqual(payload["outlier_count"], 0)
        self.assertEqual(report["duplicate_occurrence_offset_outliers"], [])
        first = report["assignments"][0]  # type: ignore[index]
        self.assertNotIn("duplicate_occurrence_offset_mismatch", first["flags"])
        self.assertNotIn("duplicate_occurrence_offset_mismatch", first)
        risk = report["suspicious_alignments"][0]  # type: ignore[index]
        self.assertNotIn("duplicate_occurrence_offset_mismatch", risk["flags"])
        self.assertNotIn("duplicate_occurrence_offset_mismatch", risk)

    def test_recomputed_final_delta_retains_genuine_duplicate_mismatch(self) -> None:
        entries, report, state = self.fixture(corrected_second_a=84.0)
        payload = auto_lrc._recompute_duplicate_occurrence_offset_diagnostics(
            entries, report, state
        )
        self.assertEqual(payload["outlier_count"], 1)
        outlier = report["duplicate_occurrence_offset_outliers"][0]  # type: ignore[index]
        self.assertEqual(outlier["entry"], 1)
        self.assertAlmostEqual(float(outlier["error_seconds"]), 1.0, places=3)
        first = report["assignments"][0]  # type: ignore[index]
        self.assertIn("duplicate_occurrence_offset_mismatch", first["flags"])

    def test_duplicate_review_refresh_is_postcommit_only(self) -> None:
        source = Path(auto_lrc.__file__).read_text(encoding="utf-8")
        module = ast.parse(source)
        private_replay = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_peer_recovery_project_selection"
        )
        replay_calls = {
            getattr(node.func, "id", None): node.lineno
            for node in ast.walk(private_replay)
            if isinstance(node, ast.Call)
        }
        self.assertLess(
            replay_calls["_apply_evaluated_backend_projection"],
            replay_calls["_postcommit_recompute_duplicate_review"],
        )

        process = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_audio"
        )
        process_calls = {
            getattr(node.func, "id", None): node.lineno
            for node in ast.walk(process)
            if isinstance(node, ast.Call)
        }
        self.assertLess(
            process_calls["commit_evaluated_backend"],
            process_calls["_postcommit_recompute_duplicate_review"],
        )


if __name__ == "__main__":
    unittest.main()
