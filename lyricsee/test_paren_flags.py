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
        for bad in ("..\\x", "a/b", "..", "C:evil", "c:\\evil", "a:b", "\\\\host\\share", "/abs"):
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


    def test_text_sha256_ignores_bom_and_crlf(self):
        plain = hashlib.sha256(b"a b\n(c d)\n").hexdigest()
        self.assertEqual(pf.text_sha256(b"\xef\xbb\xbfa b\r\n(c d)\r\n"), plain)
        self.assertEqual(pf.VERSION, 2)

    def test_reflag_hash_matches_for_a_bom_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for d in ("txt", "report"):
                (root / d).mkdir()
            (root / "txt" / "s.lyrics.txt").write_bytes(b"\xef\xbb\xbfa b\n(c d)\n")
            (root / "report" / "s.sources.json").write_text("{}", encoding="utf-8")
            flags = pf.reflag_existing(root, "s", lrclib=False)
            self.assertEqual(flags["txt_sha256"], hashlib.sha256(b"a b\n(c d)\n").hexdigest())


class AudioDirTests(unittest.TestCase):
    PLAIN = "a b\nc d\n"
    PAREN = "a b\n(c d)\ne f\n"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.music = self.root / "music"
        (self.music / "sub").mkdir(parents=True)
        self.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def song(self, rel, body, suffix=".lyrics.txt", data=None):
        audio = self.music / rel
        audio.write_bytes(b"")
        if body is not None:
            audio.with_name(audio.stem + suffix).write_bytes(data if data is not None else body.encode("utf-8"))
        return audio

    def run_mode(self, **kwargs):
        def fetch(identity):
            self.calls.append(identity)
            return ["a b", "e f"]
        return pf.flag_audio_dir(self.music, fetch=fetch, identity_of=lambda audio: audio.stem, **kwargs)

    def test_only_songs_with_blocks_fetch_and_write(self):
        self.song("plain.flac", self.PLAIN)
        self.song("paren.flac", self.PAREN)
        self.song("nolyrics.flac", None)
        side = self.root / "side"
        side.mkdir()
        out = self.run_mode(sidecar_dir=side)
        self.assertEqual((out["scanned"], out["with_blocks"]), (2, 1))
        self.assertEqual(self.calls, ["paren"])
        self.assertEqual(out["lines_by_class"], {"official-only": 1})
        self.assertEqual(sorted(p.name for p in side.iterdir()), ["paren.paren_flags.json"])
        saved = json.loads((side / "paren.paren_flags.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["version"], 2)
        self.assertEqual(saved["txt_sha256"], hashlib.sha256(self.PAREN.encode("utf-8")).hexdigest())
        self.assertEqual(saved["paren_flags"]["txt_sha256"], saved["txt_sha256"])
        self.assertEqual(out["songs"][0]["lines"], [2])

    def test_flags_come_from_the_text_beside_the_audio_in_suffix_order(self):
        self.song("s.flac", self.PAREN, suffix=".txt")
        (self.music / "s.lyrics.lrc").write_text("[00:01.00](x)\n", encoding="utf-8")
        out = self.run_mode(offline=True, out_root=self.root / "o")
        self.assertEqual(out["with_blocks"], 1)
        self.assertEqual(self.calls, [])
        self.assertEqual(out["songs"][0]["classes"], ["unconfirmed"])
        (self.music / "s.lyrics.txt").write_text("a b\n", encoding="utf-8")  # higher priority, no block
        self.assertEqual(self.run_mode(offline=True)["with_blocks"], 0)

    def test_dry_run_writes_nothing(self):
        self.song("paren.flac", self.PAREN)
        out = self.run_mode(dry_run=True, out_root=self.root / "o", sidecar_dir=self.root / "side")
        self.assertEqual(out["with_blocks"], 1)
        self.assertFalse((self.root / "o").exists())
        self.assertFalse((self.root / "side").exists())
        self.assertEqual(sorted(p.name for p in self.music.iterdir() if p.is_file()), ["paren.flac", "paren.lyrics.txt"])

    def test_recursive_mirrors_sub_folders_and_non_recursive_skips_them(self):
        self.song("sub/deep.flac", self.PAREN)
        side = self.root / "side"
        self.assertEqual(self.run_mode(sidecar_dir=side)["scanned"], 0)
        self.assertEqual(self.run_mode(sidecar_dir=side, recursive=True)["with_blocks"], 1)
        self.assertTrue((side / "sub" / "deep.paren_flags.json").is_file())

    def test_out_root_file_is_reader_compatible_and_bom_hash_matches(self):
        self.song("s.flac", None)
        (self.music / "s.lyrics.txt").write_bytes(b"\xef\xbb\xbf" + self.PAREN.replace("\n", "\r\n").encode("utf-8"))
        out = self.root / "o"
        self.run_mode(out_root=out)
        saved = json.loads((out / "report" / "s.sources.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["paren_flags"]["txt_sha256"], hashlib.sha256(self.PAREN.encode("utf-8")).hexdigest())
        self.assertTrue((out / "SUMMARY.md").exists() is False)  # summary is written by main, not the function

    def test_failed_fetch_is_unconfirmed(self):
        self.song("s.flac", self.PAREN)

        def boom(identity):
            raise OSError("down")
        out = pf.flag_audio_dir(self.music, fetch=boom, identity_of=lambda a: a, out_root=self.root / "o")
        self.assertEqual(out["songs"][0]["classes"], ["unconfirmed"])

    def test_song_limit_stops_and_says_so(self):
        for n in range(3):
            self.song(f"s{n}.flac", self.PAREN)
        out = self.run_mode(offline=True, max_songs=2)
        self.assertEqual((out["scanned"], out["truncated"]), (2, True))

    def test_main_writes_summary_and_requires_a_target(self):
        self.song("s.flac", self.PAREN)
        out = self.root / "o"
        orig = pf._fetch_timed_lines
        pf._fetch_timed_lines = lambda identity: None
        try:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                pf.main(["--from-audio-dir", str(self.music)])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(pf.main(["--from-audio-dir", str(self.music), "--out-root", str(out), "--dry-run", "--offline"]), 0)
        finally:
            pf._fetch_timed_lines = orig
        summary = (out / "SUMMARY.md").read_text(encoding="utf-8")
        self.assertIn("songs with parenthesized blocks: 1", summary)
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["SUMMARY.md"])


if __name__ == "__main__":
    unittest.main()
