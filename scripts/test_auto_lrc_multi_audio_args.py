"""auto_lrc refuses --lyrics / --anchor-hints together with several audio files."""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import auto_lrc


class AutoLrcMultiAudioTests(unittest.TestCase):
    def _run(self, argv):
        err = io.StringIO()
        with mock.patch.object(auto_lrc, "process_audio") as process, contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                auto_lrc.main(argv)
        return caught.exception.code, err.getvalue(), process

    def test_lyrics_with_several_audio_files_is_an_argparse_error(self):
        code, err, process = self._run(["a.flac", "b.flac", "--lyrics", "a.txt"])
        self.assertEqual(code, 2)
        self.assertIn("--lyrics 只能搭配一首音檔", err)
        process.assert_not_called()

    def test_anchor_hints_with_several_audio_files_is_an_argparse_error(self):
        code, err, process = self._run(["a.flac", "b.flac", "--anchor-hints", "h.txt"])
        self.assertEqual(code, 2)
        self.assertIn("--anchor-hints 只能搭配一首音檔", err)
        process.assert_not_called()

    def test_one_audio_file_with_lyrics_still_runs(self):
        with mock.patch.object(auto_lrc, "process_audio") as process, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(auto_lrc.main(["a.flac", "--lyrics", "a.txt", "--anchor-hints", "h.txt"]), 0)
        process.assert_called_once()

    def test_several_audio_files_without_per_song_inputs_still_run(self):
        with mock.patch.object(auto_lrc, "process_audio") as process, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(auto_lrc.main(["a.flac", "b.flac"]), 0)
        self.assertEqual(process.call_count, 2)


if __name__ == "__main__":
    unittest.main()
