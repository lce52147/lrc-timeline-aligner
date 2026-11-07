import hashlib
import json
import os
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


class LookupTests(RowMarkTests):
    """Flags for a song whose audio and lyrics sit in a plain music folder."""

    def source_json(self, root, **extra):
        (root / "report").mkdir(parents=True, exist_ok=True)
        flags = dict(_flags(None), **extra)
        (root / "report" / "song.sources.json").write_text(json.dumps({"paren_flags": flags}), encoding="utf-8")

    def test_finds_flags_in_extra_output_root_without_a_sidecar(self):
        out = self.dir / "elsewhere" / "lyricsee_out"
        self.source_json(out, txt_sha256=self.sha)
        self.assertEqual(marks.row_marks(self.audio, self.rows), {})
        self.assertEqual(marks.row_marks(self.audio, self.rows, extra_roots=[out]),
                         {1: "official-only", 2: "official-only", 4: "unconfirmed"})
        self.assertFalse((self.dir / "song.paren_flags.json").exists())

    def test_env_root_and_top_level_hash_fallback(self):
        out = self.dir / "env_out"
        self.source_json(out)
        (out / "report" / "song.sources.json").write_text(
            json.dumps({"txt_sha256": self.sha, "paren_flags": _flags(None)}), encoding="utf-8")
        old = os.environ.get("LRC_PAREN_FLAG_ROOTS")
        os.environ["LRC_PAREN_FLAG_ROOTS"] = str(out)
        try:
            self.assertEqual(set(marks.row_marks(self.audio, self.rows)), {1, 2, 4})
        finally:
            if old is None:
                os.environ.pop("LRC_PAREN_FLAG_ROOTS")
            else:
                os.environ["LRC_PAREN_FLAG_ROOTS"] = old

    def test_matches_by_stem_then_hash_decides(self):
        out = self.dir / "o"
        self.source_json(out, txt_sha256="0" * 64)
        self.assertEqual(marks.row_marks(self.audio, self.rows, extra_roots=[out]), {})
        other = self.dir / "other.flac"
        other.write_bytes(b"")
        (self.dir / "other.lyrics.txt").write_bytes(LYRICS.encode("utf-8"))
        self.source_json(out, txt_sha256=self.sha)
        self.assertEqual(marks.row_marks(other, self.rows, extra_roots=[out]), {})  # no report for that stem

    def test_edited_lyrics_beside_audio_never_label_a_wrong_row(self):
        out = self.dir / "o"
        self.source_json(out, txt_sha256=self.sha)
        edited = "inserted line\n" + LYRICS
        (self.dir / "song.lyrics.txt").write_bytes(edited.encode("utf-8"))
        self.assertEqual(marks.row_marks(self.audio, self.rows, extra_roots=[out]), {})

    def test_flags_hash_is_the_one_compared_not_the_stale_top_level(self):
        out = self.dir / "o"
        (out / "report").mkdir(parents=True)
        (out / "report" / "song.sources.json").write_text(
            json.dumps({"txt_sha256": "0" * 64, "paren_flags": dict(_flags(None), txt_sha256=self.sha)}), encoding="utf-8")
        self.assertEqual(set(marks.row_marks(self.audio, self.rows, extra_roots=[out])), {1, 2, 4})

    def test_crlf_copy_beside_audio_still_matches(self):
        out = self.dir / "o"
        self.source_json(out, txt_sha256=self.sha)
        (self.dir / "song.lyrics.txt").write_bytes(LYRICS.replace("\n", "\r\n").encode("utf-8"))
        self.assertEqual(set(marks.row_marks(self.audio, self.rows, extra_roots=[out])), {1, 2, 4})

    def test_lyrics_lrc_is_never_the_lyrics_file(self):
        out = self.dir / "o"
        self.source_json(out, txt_sha256=self.sha)
        (self.dir / "song.lyrics.lrc").write_text("[00:01.00]alpha one\n", encoding="utf-8")
        self.assertEqual(set(marks.row_marks(self.audio, self.rows, extra_roots=[out])), {1, 2, 4})
        (self.dir / "song.lyrics.txt").unlink()
        self.assertEqual(marks.row_marks(self.audio, self.rows, extra_roots=[out]), {})

    def test_plain_txt_is_second_choice(self):
        out = self.dir / "o"
        self.source_json(out, txt_sha256=self.sha)
        (self.dir / "song.lyrics.txt").rename(self.dir / "song.txt")
        self.assertEqual(set(marks.row_marks(self.audio, self.rows, extra_roots=[out])), {1, 2, 4})


if __name__ == "__main__":
    unittest.main()
