from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import r2_pipeline


class R2PipelineTests(unittest.TestCase):
    def _baseline_report(self, lyric_path: Path, vocal_path: Path, rows: int = 8) -> dict[str, object]:
        assignments = []
        for entry in range(1, rows + 1):
            current = float(entry)
            assignments.append(
                {
                    "entry": entry,
                    "timestamp": current + 0.02,
                    "ctc_first_token_start": current - 0.01,
                    "candidates": [
                        {
                            "source": "ctc-current",
                            "time": current,
                            "score": 0.9,
                            "reasons": [],
                        }
                    ],
                }
            )
        return {
            "lyrics_path": str(lyric_path),
            "audio_path": str(lyric_path.with_suffix(".flac")),
            "ctc_audio_path": str(vocal_path),
            "assignments": assignments,
        }

    def test_build_reviewer_payload_uses_baseline_current_and_lyric_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lyrics = root / "song.lyrics.txt"
            vocal = root / "vocals.wav"
            lyrics.write_text("\n".join(f"かな{entry}" for entry in range(1, 9)) + "\n", encoding="utf-8")
            vocal.write_bytes(b"fake")
            report = self._baseline_report(lyrics, vocal)

            payload = r2_pipeline.build_reviewer_payload(report)

        self.assertEqual(payload["partition"], "production-single")
        self.assertEqual(payload["song_count"], 1)
        song = payload["songs"][0]
        self.assertEqual(song["language"], "japanese")
        self.assertEqual(song["row_count"], 8)
        self.assertEqual(song["rows"][0]["text"], "かな1")
        self.assertEqual(song["rows"][0]["sources"]["CUR"]["time"], 1.0)
        self.assertEqual(song["rows"][0]["sources"]["FIN"]["time"], 1.02)
        self.assertEqual(song["rows"][0]["sources"]["MMS"]["time"], 0.99)

    def test_build_r2_sidecar_qualifies_offset_only_with_eight_nearby_samples(self) -> None:
        rows = []
        hubp_rows: dict[str, object] = {}
        for entry in range(1, 9):
            current = float(entry)
            rows.append(
                {
                    "entry": entry,
                    "sources": {
                        "CUR": {"time": current},
                        "WX": {"time": current + 0.10},
                        "XLSR": {"time": current + 0.20},
                    },
                }
            )
            hubp_rows[f"song::{entry}"] = {"status": "OK", "time": current + 0.05}
        payload = {"songs": [{"id": "song", "rows": rows}]}

        sidecar = r2_pipeline.build_r2_sidecar(payload, {"rows": hubp_rows})

        self.assertEqual(sidecar["policy"], "R2b")
        self.assertEqual(sidecar["required_agreements"], 1)
        self.assertEqual(sidecar["offset_sample_counts"], {"HUBP": 8, "WX": 8, "XLSR": 8})
        self.assertAlmostEqual(sidecar["offsets"]["HUBP"], 0.05, places=6)
        self.assertAlmostEqual(sidecar["offsets"]["WX"], 0.10, places=6)
        self.assertAlmostEqual(sidecar["offsets"]["XLSR"], 0.20, places=6)

        payload["songs"][0]["rows"] = rows[:7]
        limited_hubp = {"rows": {key: value for key, value in hubp_rows.items() if not key.endswith("::8")}}
        limited = r2_pipeline.build_r2_sidecar(payload, limited_hubp)
        self.assertEqual(limited["offset_sample_counts"], {"HUBP": 7, "WX": 7, "XLSR": 7})
        self.assertEqual(limited["offsets"], {})

    def test_optional_hub_command_is_absent_by_default_and_added_only_when_requested(self) -> None:
        args = dict(
            data_path=Path("work/reviewer-data.json"),
            plan_path=Path("work/hubp-plan.json"),
            hubp_path=Path("work/hubp-raw.json"),
            song_id="song",
            python_path=Path("python.exe"),
            device="cuda",
        )
        default = r2_pipeline.build_reviewer_commands(**args)
        opted_in = r2_pipeline.build_reviewer_commands(**args, include_hub=True)
        self.assertNotIn("HUB", [name for name, _command in default])
        self.assertIn("HUB", [name for name, _command in opted_in])

    def test_hub_unsupported_status_is_missing_not_failed(self) -> None:
        payload = {
            "songs": [{"HUB": {"status": "UNSUPPORTED_TEXT", "reason": "reading-unavailable"}}]
        }
        self.assertEqual(r2_pipeline.reviewer_step_status("HUB", 2, payload), "MISSING_UNSUPPORTED")
        self.assertEqual(r2_pipeline.reviewer_step_status("HUB", 1, payload), "FAILED")

    def _trust_inputs(self, *, language: str = "japanese", include_hub: bool = True):
        rows = []
        hubp_rows: dict[str, object] = {}
        final_times = []
        for entry in range(1, 9):
            final_time = float(entry)
            sources = {
                "WX": {"time": final_time + 0.01},
                "XLSR": {"time": final_time + 0.02},
            }
            if include_hub:
                sources["HUB"] = {"time": final_time + 0.03}
            rows.append({"entry": entry, "sources": sources})
            hubp_rows[f"song::{entry}"] = {"status": "OK", "time": final_time + 0.04}
            final_times.append(final_time)
        return (
            {"songs": [{"id": "song", "language": language, "rows": rows}]},
            {"rows": hubp_rows},
            final_times,
        )

    def test_reviewer_profile_default_none_marks_all_rows_review(self) -> None:
        payload, hubp, final_times = self._trust_inputs()
        result = r2_pipeline.build_reviewer_trust_report(payload, hubp, final_times)
        self.assertEqual(result["profile"], "none")
        self.assertEqual(result["trusted_count"], 0)
        self.assertEqual(result["review_count"], 8)

    def test_balanced_and_loose_profiles_use_final_r2_time(self) -> None:
        payload, hubp, final_times = self._trust_inputs()
        balanced = r2_pipeline.build_reviewer_trust_report(
            payload, hubp, final_times, profile="balanced"
        )
        loose = r2_pipeline.build_reviewer_trust_report(
            payload, hubp, final_times, profile="loose"
        )
        self.assertEqual(balanced["trusted_count"], 8)
        self.assertEqual(loose["trusted_count"], 8)
        self.assertTrue(all(row["reviewer_trusted"] for row in balanced["rows"]))

    def test_reviewer_profiles_fail_closed_for_non_japanese_or_missing_hub(self) -> None:
        english, hubp, final_times = self._trust_inputs(language="english")
        missing_hub, hubp2, final_times2 = self._trust_inputs(include_hub=False)
        for payload, raw_hubp, times in (
            (english, hubp, final_times),
            (missing_hub, hubp2, final_times2),
        ):
            with self.subTest(language=payload["songs"][0]["language"]):
                balanced = r2_pipeline.build_reviewer_trust_report(
                    payload, raw_hubp, times, profile="balanced"
                )
                loose = r2_pipeline.build_reviewer_trust_report(
                    payload, raw_hubp, times, profile="loose"
                )
                self.assertEqual(balanced["trusted_count"], 0)
                self.assertEqual(loose["trusted_count"], 0)
                self.assertEqual(balanced["review_count"], 8)
                self.assertEqual(loose["review_count"], 8)

    def test_stage_messages_keep_existing_text_and_add_progress_and_elapsed(self) -> None:
        self.assertEqual(
            r2_pipeline.stage_message(1, "R2: generating Central baseline"),
            "[1/4] R2: generating Central baseline",
        )
        self.assertEqual(r2_pipeline.stage_done_message(1, 192.4), "[1/4] 完成，用時 3m12s")
        self.assertEqual(r2_pipeline.stage_done_message(2, 9.6), "[2/4] 完成，用時 10s")

    def test_lyric_review_messages_use_calibrated_lowest_five_without_auto_mismatch_warning(self) -> None:
        report = {
            "review_required_count": 4,
            "low_confidence_count": 6,
            "assignments": [
                {
                    "entry": entry,
                    "ctc_score": score,
                    "ctc_first_token_score": score + 0.01,
                }
                for entry, score in enumerate((0.8, 0.1, 0.6, 0.2, 0.4, 0.3), start=1)
            ],
        }
        reviewer_payload = {
            "songs": [
                {
                    "rows": [
                        {"entry": entry, "text": f"line-{entry}-" + "x" * 40}
                        for entry in range(1, 7)
                    ]
                }
            ]
        }

        lines = r2_pipeline.build_lyric_review_messages(report, reviewer_payload)
        combined = "\n".join(lines)

        self.assertIn("Review required 4 行", combined)
        self.assertIn("Low confidence 6 行", combined)
        self.assertNotIn("歌詞可能與歌聲不一致", combined)
        self.assertEqual([line.split()[0] for line in lines[2:]], ["#2", "#4", "#6", "#5", "#3"])
        self.assertNotIn("x" * 31, combined)

    def test_sparse_reviewer_notice_triggers_below_two_reviewers_and_below_half_rows(self) -> None:
        rows = []
        for entry in range(1, 11):
            sources = {}
            if entry <= 4:
                sources["XLSR"] = {"time": float(entry)}
            rows.append({"entry": entry, "sources": sources})
        payload = {"songs": [{"id": "song", "rows": rows}]}

        notice = r2_pipeline.build_sparse_reviewer_notice(payload, {})

        self.assertIsNotNone(notice)
        self.assertIn("reviewer 證據不足（日文專用模型）", notice)
        self.assertIn("Trusted timing", notice)
        self.assertIn("1/3", notice)
        self.assertIn("4/10", notice)

    def test_sparse_reviewer_notice_ignores_raw_values_that_cannot_qualify_for_r2(self) -> None:
        rows = []
        for entry in range(1, 22):
            current = float(entry * 10)
            rows.append(
                {
                    "entry": entry,
                    "sources": {
                        "CUR": {"time": current},
                        "XLSR": {"time": current + 1.0},
                    },
                }
            )
        payload = {"songs": [{"id": "song", "rows": rows}]}

        notice = r2_pipeline.build_sparse_reviewer_notice(payload, {})

        self.assertIsNotNone(notice)
        self.assertIn("0/3", notice)
        self.assertIn("0/21", notice)

    def test_sparse_reviewer_notice_does_not_trigger_at_two_reviewers_or_half_rows(self) -> None:
        two_reviewer_rows = []
        for entry in range(1, 11):
            sources = {}
            if entry <= 4:
                sources["WX"] = {"time": float(entry)}
            if entry == 1:
                sources["XLSR"] = {"time": float(entry) + 0.1}
            two_reviewer_rows.append({"entry": entry, "sources": sources})
        two_reviewers = {"songs": [{"id": "song", "rows": two_reviewer_rows}]}
        self.assertIsNone(r2_pipeline.build_sparse_reviewer_notice(two_reviewers, {}))

        half_rows = []
        for entry in range(1, 11):
            sources = {"XLSR": {"time": float(entry)}} if entry <= 5 else {}
            half_rows.append({"entry": entry, "sources": sources})
        half_covered = {"songs": [{"id": "song", "rows": half_rows}]}
        self.assertIsNone(r2_pipeline.build_sparse_reviewer_notice(half_covered, {}))
    def test_prepare_backend_args_for_baseline_removes_review_gate_and_overrides_paths(self) -> None:
        raw = [
            "--timing-source", "auto",
            "--arbiter", "reviewer-validity",
            "--reviewer-evidence", "old.json",
            "--strict-review",
            "--min-trusted-percent", "100",
            "--output", "final.lrc",
            "--report-dir", "final-reports",
            "song.flac",
        ]
        baseline = r2_pipeline.prepare_baseline_args(
            raw,
            output=Path("scratch/baseline.lrc"),
            report_dir=Path("scratch/reports"),
        )
        joined = " ".join(baseline)
        self.assertNotIn("old.json", joined)
        self.assertNotIn("--strict-review", baseline)
        self.assertNotIn("--min-trusted-percent", baseline)
        self.assertEqual(baseline[-7:], [
            "--arbiter", "off",
            "--output", str(Path("scratch/baseline.lrc")),
            "--report-dir", str(Path("scratch/reports")),
            "--overwrite",
        ])

    def test_prepare_final_args_binds_generated_evidence(self) -> None:
        final = r2_pipeline.prepare_final_args(
            ["--timing-source", "auto", "--arbiter", "off", "song.flac"],
            Path("reviewers/song.json"),
        )
        self.assertEqual(
            final[-4:],
            ["--arbiter", "reviewer-validity", "--reviewer-evidence", str(Path("reviewers/song.json"))],
        )


if __name__ == "__main__":
    unittest.main()
