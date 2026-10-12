import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import auto_lrc


class CtcHelperRowContractTests(unittest.TestCase):
    def test_non_object_ctc_helper_row_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            audio = temp / "song.wav"
            audio.write_bytes(b"not-real-audio-but-hashed")
            entries = [auto_lrc.LyricEntry(["a"]), auto_lrc.LyricEntry(["b"])]
            args = argparse.Namespace(
                vocal_ctc=False,
                whisperx_device="cpu",
            )

            def fake_helper(command, *, mode):
                out = Path(command[command.index("--output") + 1])
                out.write_text(
                    json.dumps({
                        "entries": [
                            {"start": 1.0, "ctc_score": 0.9, "token_spans": []},
                            "bad-row",
                        ],
                        "backend": "mms_ctc",
                    }),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            with (
                mock.patch.object(
                    auto_lrc,
                    "prepare_vocal_ctc_audio",
                    return_value=(audio, "full-mix", None),
                ),
                mock.patch.object(
                    auto_lrc,
                    "default_ctc_python",
                    return_value=Path(sys.executable),
                ),
                mock.patch.object(
                    auto_lrc,
                    "vocal_cache_identity",
                    return_value={},
                ),
                mock.patch.object(
                    auto_lrc,
                    "_run_model_helper_command",
                    side_effect=fake_helper,
                ),
            ):
                with self.assertRaisesRegex(
                    auto_lrc.LrcError,
                    "non-object entry",
                ):
                    auto_lrc.run_ctc_alignment(audio, entries, 10.0, args)


if __name__ == "__main__":
    unittest.main()
