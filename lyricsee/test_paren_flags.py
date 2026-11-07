import json
import tempfile
import unittest
from pathlib import Path

import paren_flags as pf


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
        self.assertEqual([(b["start"], b["class"]) for b in flags["blocks"]], [(2, "official-only")])

    def test_reflag_existing_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "txt").mkdir()
            (root / "report").mkdir()
            (root / "txt" / "s.lyrics.txt").write_text("a b\n(c d)\n", encoding="utf-8")
            (root / "report" / "s.sources.json").write_text(json.dumps({"txt_sha256": "x"}), encoding="utf-8")
            flags = pf.reflag_existing(root, "s")
            saved = json.loads((root / "report" / "s.sources.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["paren_flags"], flags)
            self.assertEqual(saved["txt_sha256"], "x")


if __name__ == "__main__":
    unittest.main()
