import hashlib
import json
import pathlib
import tempfile
import unittest

import paren_flag_marks as marks

LYRICS = "alpha one\n(back one\nback two)\nalpha two\n(solo)\nalpha one\n"


def _flags(path, klass="official-only"):
    return {"version": 1, "blocks": [
        {"start": 2, "end": 3, "lines": [2, 3], "class": klass, "unclosed": False},
        {"start": 5, "end": 5, "lines": [5], "class": "unconfirmed", "unclosed": False},
    ]}


class RowMarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.audio = self.dir / "song.flac"
        self.audio.write_bytes(b"")
        (self.dir / "song.lyrics.txt").write_bytes(LYRICS.encode("utf-8"))
        self.sha = hashlib.sha256(LYRICS.encode("utf-8")).hexdigest()
        self.rows = [["alpha one"], ["(back one"], ["back two)"], ["alpha two"], ["(solo)"], ["alpha one"]]

    def tearDown(self):
        self.tmp.cleanup()

    def sidecar(self, sha=None):
        (self.dir / "song.paren_flags.json").write_text(
            json.dumps({"txt_sha256": sha or self.sha, "paren_flags": _flags(None)}), encoding="utf-8")

    def test_marks_flagged_rows_only(self):
        self.sidecar()
        self.assertEqual(marks.row_marks(self.audio, self.rows), {1: "official-only", 2: "official-only", 4: "unconfirmed"})

    def test_repeated_text_outside_block_is_not_marked(self):
        self.sidecar()
        self.assertNotIn(0, marks.row_marks(self.audio, self.rows))
        self.assertNotIn(5, marks.row_marks(self.audio, self.rows))

    def test_no_flag_data_means_no_marks(self):
        self.assertEqual(marks.row_marks(self.audio, self.rows), {})

    def test_stale_lyrics_hash_means_no_marks(self):
        self.sidecar(sha="0" * 64)
        self.assertEqual(marks.row_marks(self.audio, self.rows), {})

    def test_garbage_flag_file_means_no_marks(self):
        (self.dir / "song.paren_flags.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(marks.row_marks(self.audio, self.rows), {})

    def test_reads_sources_json_next_to_txt_dir(self):
        out = self.dir / "out"
        (out / "txt").mkdir(parents=True)
        (out / "report").mkdir()
        (out / "txt" / "song.lyrics.txt").write_bytes(LYRICS.encode("utf-8"))
        (out / "report" / "song.sources.json").write_text(
            json.dumps({"txt_sha256": self.sha, "paren_flags": _flags(None)}), encoding="utf-8")
        (self.dir / "song.lyrics.txt").unlink()
        audio = out / "txt" / "song.flac"
        self.assertEqual(marks.row_marks(audio, self.rows), {1: "official-only", 2: "official-only", 4: "unconfirmed"})


if __name__ == "__main__":
    unittest.main()
