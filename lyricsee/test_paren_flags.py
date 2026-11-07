import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import paren_flags as pf
from lyric_compare import compare_lyrics, diff_lines_for_report


class DetectTests(unittest.TestCase):
    def blocks(self, lines):
        return [(b["start"], b["end"], b["unclosed"]) for b in pf.detect_blocks(lines)]

    def test_single_line(self):
        self.assertEqual(self.blocks(["abc", "(I want to call)", "def"]), [(2, 2, False)])

    def test_multi_line(self):
        self.assertEqual(self.blocks(["abc", "(one", "two", "three)", "def"]), [(2, 4, False)])

    def test_inline_not_flagged(self):
        self.assertEqual(self.blocks(["abc (inner) def", "(lead) tail", "tail (end)"]), [])

    def test_full_width(self):
        self.assertEqual(self.blocks(["（ひとつ", "ふたつ）", "（みっつ）"]), [(1, 2, False), (3, 3, False)])

    def test_unclosed_ends_at_its_own_line(self):
        self.assertEqual(self.blocks(["(open", "a", "b", "c", "d", "e", "f", "g", "h", "i"]), [(1, 1, True)])

    def test_plain_middle_line_stays_in_block(self):
        self.assertEqual(self.blocks(["(open", "plain", "tail)"]), [(1, 3, False)])

    def test_text_on_opening_line_outside_parens_is_not_a_block(self):
        self.assertEqual(self.blocks(["(Hey) yeah (oh", "yeah)"]), [])

    def test_stray_close_paren_counts_as_outside_text(self):
        self.assertEqual(self.blocks(["(verse) )"]), [])

    def test_translation_split(self):
        self.assertEqual(pf.original_of("(abc)  翻譯"), "(abc)")


class ClassifyTests(unittest.TestCase):
    LINES = ["alpha beta", "(gamma delta)", "(one two", "three four)", "omega"]

    def test_unconfirmed_without_timed_source(self):
        flags = pf.compute_flags(self.LINES)
        self.assertEqual([b["class"] for b in flags["blocks"]], ["unconfirmed", "unconfirmed"])
        self.assertEqual(flags["basis"], "none")
        self.assertEqual(flags["line_count"], 3)

    def test_official_only_when_timed_lacks_block(self):
        flags = pf.compute_flags(self.LINES, timed_lines=["alpha beta", "omega"])
        self.assertEqual([(b["start"], b["end"], b["class"]) for b in flags["blocks"]],
                         [(2, 2, "official-only"), (3, 4, "official-only")])

    def test_filler_only_block_lacking_in_timed_is_official_only(self):
        lines = ["alpha beta", "(la la la)", "(oh oh)"]
        flags = pf.compute_flags(lines, timed_lines=["alpha beta"])
        self.assertEqual([(b["start"], b["class"]) for b in flags["blocks"]], [(2, "official-only"), (3, "official-only")])
        self.assertEqual(pf.compute_flags(lines, timed_lines=["alpha beta", "la la la", "oh oh"])["blocks"], [])

    def test_not_flagged_when_timed_has_block(self):
        flags = pf.compute_flags(self.LINES, timed_lines=["alpha beta", "gamma delta", "one two", "three four", "omega"])
        self.assertEqual(flags["blocks"], [])

    def test_mostly_absent_counts_as_lacking(self):
        lines = ["(a1 a1", "b2 b2", "c3 c3)"]
        self.assertEqual(pf.compute_flags(lines, timed_lines=["a1 a1"])["blocks"][0]["class"], "official-only")
        self.assertEqual(pf.compute_flags(lines, timed_lines=["a1 a1", "b2 b2"])["blocks"], [])

    def test_short_block_line_must_equal_a_timed_line(self):
        lines = ["(kiss)", "plain words here"]
        self.assertEqual(pf.compute_flags(lines, timed_lines=["kissing now", "plain words here"])["blocks"][0]["class"], "official-only")
        self.assertEqual(pf.compute_flags(lines, timed_lines=["Kiss!", "plain words here"])["blocks"], [])

    def test_saved_diff_rows(self):
        cross = [{"source": "lrclib", "independent": True, "diff_lines": [
            {"kind": "missing_in_b", "line": 2, "text": "(gamma delta)"},
            {"kind": "word", "line": 3, "text": "one two"},
            {"kind": "missing_in_b", "line": 1, "text": "unrelated"},
        ]}]
        flags = pf.compute_flags(self.LINES, cross_sources=cross)
        self.assertEqual(flags["basis"], "saved-diff")
        self.assertEqual([(b["start"], b["class"]) for b in flags["blocks"]], [(2, "official-only"), (3, "unconfirmed")])

    def test_saved_diff_from_real_comparer_never_drops_a_block(self):
        official = ["alpha beta gamma", "(I want to call you)", "delta epsilon zeta", "(one two three", "four five six)", "eta theta iota"]
        lrclib = ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
        rows = diff_lines_for_report(compare_lyrics(official, lrclib))
        cross = [{"source": "lrclib", "independent": True, "diff_lines": rows}]
        saved = pf.compute_flags(official, cross_sources=cross)
        self.assertEqual(saved["basis"], "saved-diff")
        self.assertEqual([(b["start"], b["end"]) for b in saved["blocks"]], [(2, 2), (4, 5)])  # undecidable here, never silently dropped
        self.assertTrue(all(b["class"] in ("official-only", "unconfirmed") for b in saved["blocks"]))
        timed = pf.compute_flags(official, timed_lines=lrclib)
        self.assertEqual([(b["start"], b["class"]) for b in timed["blocks"]], [(2, "official-only"), (4, "official-only")])

    def test_reflag_existing_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "txt").mkdir()
            (root / "report").mkdir()
            (root / "txt" / "s.lyrics.txt").write_text("a b\n(c d)\n", encoding="utf-8")
            (root / "report" / "s.sources.json").write_text(json.dumps({"txt_sha256": "x"}), encoding="utf-8")
            flags = pf.reflag_existing(root, "s", lrclib=False)
            saved = json.loads((root / "report" / "s.sources.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["paren_flags"], flags)
            self.assertEqual(flags["txt_sha256"], hashlib.sha256(b"a b\n(c d)\n").hexdigest())

    def test_reflag_hashes_the_txt_it_read_not_the_old_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            music = root / "music"
            for d in ("txt", "report", "music"):
                (root / d).mkdir()
            original = "a line\nb line\n(c line)\nd line\ne line\n"
            edited = "a line\nb line\nd line\n(e line)\n"
            (music / "s.lyrics.txt").write_text(original, encoding="utf-8", newline="\n")
            (root / "txt" / "s.lyrics.txt").write_text(edited, encoding="utf-8", newline="\n")
            old = hashlib.sha256(original.encode("utf-8")).hexdigest()
            (root / "report" / "s.sources.json").write_text(json.dumps({"txt_sha256": old}), encoding="utf-8")
            flags = pf.reflag_existing(root, "s", sidecar_dir=music, lrclib=False)
            self.assertEqual(flags["txt_sha256"], hashlib.sha256(edited.encode("utf-8")).hexdigest())
            self.assertNotEqual(flags["txt_sha256"], old)
            side = json.loads((music / "s.paren_flags.json").read_text(encoding="utf-8"))
            self.assertEqual(side["txt_sha256"], flags["txt_sha256"])

    def test_reflag_rejects_stems_with_path_separators(self):
        for bad in ("..\\x", "a/b", ".."):
            with self.assertRaises(ValueError):
                pf.reflag_existing(".", bad)

    def test_find_audio_exact_stem_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "01 I Love.flac").write_bytes(b"")
            (d / "Love.flac").write_bytes(b"")
            self.assertEqual(pf.find_audio(d, "Love").name, "Love.flac")
            with self.assertRaises(ValueError):
                pf.find_audio(d, "Missing")
            (d / "Love.mp3").write_bytes(b"")
            with self.assertRaises(ValueError):
                pf.find_audio(d, "Love")

    def test_reflag_fetches_only_songs_with_blocks(self):
        calls = []
        orig = pf._fetch_timed_lines
        pf._fetch_timed_lines = lambda identity: calls.append(identity) or ["a b"]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "txt").mkdir()
                (root / "report").mkdir()
                for stem, body in (("plain", "a b\n"), ("paren", "a b\n(c d)\n")):
                    (root / "txt" / f"{stem}.lyrics.txt").write_text(body, encoding="utf-8")
                    (root / "report" / f"{stem}.sources.json").write_text(json.dumps({"title": "T"}), encoding="utf-8")
                pf.reflag_existing(root, "plain")
                self.assertEqual(calls, [])
                flags = pf.reflag_existing(root, "paren")
                self.assertEqual(len(calls), 1)
                self.assertEqual((flags["basis"], flags["blocks"][0]["class"]), ("timed-lines", "official-only"))
        finally:
            pf._fetch_timed_lines = orig

    def test_write_result_survives_a_flag_error(self):
        import lyricsee_core as core
        orig = core.flags_for_lyrics_text
        core.flags_for_lyrics_text = lambda *a, **k: 1 / 0
        try:
            with tempfile.TemporaryDirectory() as tmp:
                audio = Path(tmp) / "s.flac"
                audio.write_bytes(b"")
                cand = core.LyricCandidate(source="site", url="http://x", title="t", artist="a", lines=[core.LyricPair("a b")])
                with self.assertLogs(level="ERROR"):
                    out, prov = core.write_result(audio, cand, output_root=Path(tmp) / "out", overwrite=False)
                self.assertTrue(out.is_file())
                self.assertIsNone(json.loads(prov.read_text(encoding="utf-8"))["paren_flags"])
        finally:
            core.flags_for_lyrics_text = orig


if __name__ == "__main__":
    unittest.main()
