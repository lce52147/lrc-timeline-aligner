#!/usr/bin/env python3
"""Anchor hints, timing trust and persistent-helper routing tests."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
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
import split_patch
import audio_features
import central_commit
import ctc_refinement
import local_whisper_retry
import lyrics_io
import runtime_base
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
from multimodel_consensus import model_disagreement_review


class MultimodelReviewOnlyTests(unittest.TestCase):
    def test_displaced_consensus_baseline_cannot_reenter_central(self) -> None:
        self.assertFalse(auto_lrc._active_legacy_candidate({
            "source": "ctc-current",
            "replaced_by_consensus": True,
        }))
        self.assertTrue(auto_lrc._active_legacy_candidate({"source": "ctc-current"}))

    def test_mms_disagreement_is_review_only(self) -> None:
        result = model_disagreement_review(
            written_times=(10.0,),
            statuses=("selected_valid",),
            model_outputs={0: {"mms": 12.2}},
            windows=({"entry_index": 0, "start": 9.0, "end": 13.0},),
        )
        self.assertEqual(result[0]["reason"], "model-disagreement")
        self.assertIn("mms", result[0]["triggers"])

    def test_three_family_cluster_is_reviewed_and_window_is_required(self) -> None:
        values = {
            "mms": 12.2,
            "mms1b:eng:roman": 12.21,
            "ja-ctc:test": 12.19,
        }
        result = model_disagreement_review(
            written_times=(10.0, 20.0),
            statuses=("selected_valid", "selected_valid"),
            model_outputs={0: values, 1: values},
            windows=({"entry_index": 0, "start": 9.0, "end": 13.0},),
        )
        self.assertIn("cluster", result[0]["triggers"])
        self.assertNotIn(1, result)

    def test_model_disagreement_flag_is_actionable_review(self) -> None:
        report = {
            "timing_entries": 1,
            "assignments": [{
                "timestamp": 10.0,
                "timing_trusted": True,
                "content_trusted": True,
                "flags": ["model-disagreement"],
            }],
        }
        update_report_confidence_metrics(report)
        self.assertTrue(report["assignments"][0]["review_required"])
        self.assertIn("model-disagreement", report["assignments"][0]["final_trust_reason_codes"])

    def test_silent_risk_rule_is_review_only(self) -> None:
        report = {
            "timing_entries": 2,
            "assignments": [
                {
                    "timestamp": 10.0,
                    "confidence": 0.8,
                    "timing_trusted": True,
                    "content_trusted": True,
                    "flags": [],
                },
                {
                    "timestamp": 30.0,
                    "confidence": 0.9,
                    "timing_trusted": True,
                    "content_trusted": True,
                    "flags": [],
                },
            ],
        }
        update_report_confidence_metrics(report, apply_silent_risk=True)
        flagged = report["assignments"][0]
        untouched = report["assignments"][1]
        self.assertIn(auto_lrc.SILENT_RISK_REVIEW_FLAG, flagged["flags"])
        self.assertTrue(flagged["review_required"])
        self.assertIn(auto_lrc.SILENT_RISK_REVIEW_FLAG, flagged["final_trust_reason_codes"])
        self.assertEqual(flagged["timestamp"], 10.0)
        self.assertFalse(untouched["review_required"])


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
                mock.patch.object(runtime_base, "_persistent_helper_worker_path", return_value=missing_worker),
                split_patch.patch("run_command", return_value=expected) as run_command,
                mock.patch.object(runtime_base, "_PersistentHelperPool") as pool_class,
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
                mock.patch.object(runtime_base, "_persistent_helper_worker_path", return_value=worker),
                mock.patch.object(runtime_base, "_PersistentHelperPool", return_value=pool) as pool_class,
                split_patch.patch("run_command") as run_command,
                split_patch.patch("progress_log"),
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
                split_patch.patch("default_ctc_python", return_value=python_exe),
                split_patch.patch("vocal_cache_identity",
                    return_value={"vocal_cache_key": "deterministic-test"},
                ),
                mock.patch.object(audio_features, "DEFAULT_VOCAL_CACHE_DIR", cache_root),
                mock.patch.object(auto_lrc.subprocess, "run", side_effect=fake_run) as run_command,
                mock.patch.object(audio_features, "_demucs_vocal_stem_valid", return_value=True),
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
        lyrics_io.audio_track_labels = lambda _path: ("Demo Artist", "Demo Song")  # type: ignore[assignment]
        try:
            entries = [
                LyricEntry(["Demo Artist - Demo Song"], 0),
                LyricEntry(["synthetic lyric line"], 1890),
            ]

            kept = remove_generated_title_cards(entries, Path("dummy.flac"))

            self.assertEqual([entry.lines[0] for entry in kept], ["synthetic lyric line"])
        finally:
            lyrics_io.audio_track_labels = original_labels  # type: ignore[assignment]

    def test_generated_title_card_match_is_ascii_case_insensitive(self) -> None:
        original_labels = auto_lrc.audio_track_labels
        lyrics_io.audio_track_labels = lambda _path: ("MyGO!!!!!", "影色舞")  # type: ignore[assignment]
        try:
            entries = [
                LyricEntry(["Mygo!!!!! - 影色舞"], 0),
                LyricEntry(["synthetic lyric line"], 1890),
            ]

            kept = remove_generated_title_cards(entries, Path("dummy.flac"))

            self.assertEqual([entry.lines[0] for entry in kept], ["synthetic lyric line"])
        finally:
            lyrics_io.audio_track_labels = original_labels  # type: ignore[assignment]

    def test_title_artist_order_card_is_not_aligned_as_lyric(self) -> None:
        # c10: "Ever be my love - 山田タマル" at 0.00 was aligned as lyric #1.
        original_labels = auto_lrc.audio_track_labels
        lyrics_io.audio_track_labels = lambda _path: ("山田タマル", "Ever be my love")  # type: ignore[assignment]
        try:
            entries = [
                LyricEntry(["Ever be my love - 山田タマル"], 0),
                LyricEntry(["synthetic lyric line"], 2100),
            ]

            kept = remove_generated_title_cards(entries, Path("dummy.flac"))

            self.assertEqual([entry.lines[0] for entry in kept], ["synthetic lyric line"])
        finally:
            lyrics_io.audio_track_labels = original_labels  # type: ignore[assignment]

    def test_card_shaped_lines_that_do_not_match_tags_are_kept(self) -> None:
        original_labels = auto_lrc.audio_track_labels
        lyrics_io.audio_track_labels = lambda _path: ("Demo Artist", "Demo Song")  # type: ignore[assignment]
        try:
            entries = [
                LyricEntry(["Demo Song - Someone Else"], 0),
                LyricEntry(["Demo Artist - Demo Song"], 150),
                LyricEntry(["Demo Song - Demo Artist", "translation"], 0),
            ]

            kept = remove_generated_title_cards(entries, Path("dummy.flac"))

            self.assertEqual(len(kept), 3)
        finally:
            lyrics_io.audio_track_labels = original_labels  # type: ignore[assignment]

    def test_title_only_tags_never_match_a_reversed_card(self) -> None:
        original_labels = auto_lrc.audio_track_labels
        lyrics_io.audio_track_labels = lambda _path: ("", "Demo Song")  # type: ignore[assignment]
        try:
            entries = [LyricEntry(["Demo Song - anything"], 0)]

            kept = remove_generated_title_cards(entries, Path("dummy.flac"))

            self.assertEqual(len(kept), 1)
        finally:
            lyrics_io.audio_track_labels = original_labels  # type: ignore[assignment]

    def test_untimed_source_title_is_preserved_as_potential_sung_lyric(self) -> None:
        original_labels = auto_lrc.audio_track_labels
        lyrics_io.audio_track_labels = lambda _path: ("", "Song Title")  # type: ignore[assignment]
        try:
            entries = [
                LyricEntry(["Song Title", "Song translation"]),
                LyricEntry(["actual lyric"], None),
            ]

            kept = remove_generated_title_cards(entries, Path("dummy.flac"))

            self.assertEqual([entry.lines[0] for entry in kept], ["Song Title", "actual lyric"])
        finally:
            lyrics_io.audio_track_labels = original_labels  # type: ignore[assignment]

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
        split_patch.assign("default_ctc_python", lambda: Path("python"))  # type: ignore[assignment]
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
            split_patch.assign("default_ctc_python", original_python)  # type: ignore[assignment]

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
        split_patch.assign("default_ctc_python", lambda: Path("python"))  # type: ignore[assignment]
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
            split_patch.assign("default_ctc_python", original_python)  # type: ignore[assignment]

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
        selected_path = next(
            item for item in assignment["candidates"]
            if item["source"] == "selected_path"
        )
        self.assertEqual(assignment["penalties"], [])
        self.assertAlmostEqual(
            assignment["confidence"], float(selected_path["score"]), places=3
        )

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

    def test_repeated_single_cjk_term_with_earlier_peak_requires_review(self) -> None:
        entries = [LyricEntry(["前の行"]), LyricEntry(["雨 雨"]), LyricEntry(["次の行"])]
        report: dict[str, object] = {
            "assignments": [
                {
                    "entry": 1,
                    "timestamp": 254.94,
                    "score": 0.9,
                    "ctc_token_spans": [
                        {"char": "x", "start": 255.0, "end": 256.46, "score": 0.66},
                    ],
                },
                {
                    "entry": 2,
                    "timestamp": 261.701,
                    "score": 0.9,
                    "ctc_score": 0.462238,
                    "ctc_token_spans": [
                        {"char": "a", "start": 261.701, "end": 261.721, "score": 0.714446},
                        {"char": "m", "start": 262.021, "end": 262.041, "score": 0.572837},
                        {"char": "e", "start": 262.181, "end": 262.201, "score": 0.556741},
                        {"char": "u", "start": 263.521, "end": 263.541, "score": 0.004929},
                    ],
                    "ctc_first_token_candidates": [
                        {"time": 259.421, "score": 0.033324},
                    ],
                },
                {"entry": 3, "timestamp": 265.861, "score": 0.9},
            ],
            "suspicious_alignments": [],
        }

        result = score_line_timing_candidates(
            entries, [254.94, 261.701, 265.861], report, 286.0
        )

        self.assertAlmostEqual(result[1], 261.701, places=3)
        assignment = report["assignments"][1]  # type: ignore[index]
        self.assertAlmostEqual(assignment["timestamp"], 261.701, places=3)  # type: ignore[arg-type]
        self.assertTrue(assignment["review_required"])
        self.assertIn("candidate_disagreement", assignment["flags"])  # type: ignore[operator]
        repeated = [
            item for item in assignment["candidates"]  # type: ignore[index]
            if item["source"] == "ctc_repeated_leading_term_onset"
        ]
        self.assertEqual(len(repeated), 1)
        self.assertAlmostEqual(repeated[0]["time"], 259.421, places=3)
        self.assertNotIn("precentral_candidate_recommendation", assignment)

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


    def test_ctc_micro_refinement_preserves_actionable_review(self) -> None:
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
                    "flags": ["candidate_disagreement"],
                    "severity": "high",
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
                    "ctc_token_spans": [
                        {"char": "a", "start": 20.05, "end": 20.07, "score": 0.2}
                    ],
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
        assignments = report["assignments"]  # type: ignore[assignment]
        self.assertFalse(assignments[1]["timing_trusted"])  # type: ignore[index]
        suspicious = report["suspicious_alignments"]  # type: ignore[assignment]
        self.assertEqual(suspicious[0]["severity"], "high")  # type: ignore[index]
        self.assertTrue(suspicious[0]["review_required"])  # type: ignore[index]
        self.assertEqual(report["review_required_count"], 1)


def tearDownModule() -> None:
    split_patch.assert_clean()


if __name__ == "__main__":
    unittest.main()
