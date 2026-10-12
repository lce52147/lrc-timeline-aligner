from __future__ import annotations

import unittest

import auto_lrc


class LongIntroLocalWhisperTests(unittest.TestCase):
    @staticmethod
    def capability() -> auto_lrc.LocalWhisperCapability:
        return auto_lrc.LocalWhisperCapability(
            kind="whispercpp-whisperx-local",
            backend_lineage="whisper-family",
            audio_path="vocal.wav",
            audio_sha256="stem-sha",
            whisper_cli_path="cli",
            whisper_cli_sha256="cli-sha",
            whisper_model_path="model",
            whisper_model_sha256="model-sha",
            whisperx_python_path="python",
            whisperx_helper_path="helper",
            whisperx_helper_sha256="helper-sha",
            language="ja",
            device="cuda",
            protocol_revision=auto_lrc._LOCAL_WHISPER_PROTOCOL_REVISION,
            capability_revision="capability",
        )

    @staticmethod
    def candidate(index: int, seconds: float) -> auto_lrc.TimingCandidate:
        spans = tuple(
            auto_lrc.TimingTokenSpan(
                seconds + offset * 0.05,
                seconds + offset * 0.05 + 0.03,
                0.6,
                token,
            )
            for offset, token in enumerate("abc")
        )
        return auto_lrc.make_timing_candidate(
            entry_index=index,
            entry_text=f"unique long intro line {index}",
            source="ctc-current",
            raw_time=seconds,
            spans=spans,
            confidence=0.6,
            identity_support="supported",
            sequence_support="supported",
            current=True,
            source_artifact={"entry": index},
        )

    def test_long_intro_does_not_drop_first_two_local_whisper_requests(self) -> None:
        entries = [
            auto_lrc.LyricEntry(["intro alpha"]),
            auto_lrc.LyricEntry(["intro beta"]),
            auto_lrc.LyricEntry(["intro gamma"]),
        ]
        times = (40.0, 43.0, 46.0)
        candidate_sets = tuple(
            (self.candidate(index, seconds),)
            for index, seconds in enumerate(times)
        )
        report: dict[str, object] = {
            "backend": "ctc",
            "assignments": [
                {
                    "entry": index + 1,
                    "timestamp": seconds,
                    "score": 0.5,
                    "segment": index + 1,
                }
                for index, seconds in enumerate(times)
            ],
        }

        requests = auto_lrc.plan_local_whisper_requests(
            report,
            self.capability(),
            entries,
            candidate_sets,
            duration=70.0,
        )

        by_entry = {request.entry_index: request for request in requests}
        self.assertIn(0, by_entry)
        self.assertIn(1, by_entry)
        self.assertGreaterEqual(by_entry[0].window_start, 38.5)
        self.assertGreaterEqual(by_entry[1].window_start, 38.5)
        self.assertLessEqual(
            by_entry[0].window_end - by_entry[0].window_start,
            auto_lrc.LOCAL_WHISPER_MAX_WINDOW_SECONDS,
        )
        self.assertLessEqual(
            by_entry[1].window_end - by_entry[1].window_start,
            auto_lrc.LOCAL_WHISPER_MAX_WINDOW_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()