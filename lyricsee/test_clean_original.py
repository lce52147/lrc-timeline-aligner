"""Tests for the cleaned original lyrics written on accept: header/annotation stripping and the furigana preference."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from candidate_review import accept_candidate, cleaned_candidate_lines
from candidate_text import (
    clean_original_lines, has_inline_furigana, strip_annotations,
)

BODY = ["朝の光が差し込んで", "僕らは歩き出した", "", "夜明けの空に手を伸ばす", "遠い約束を信じて"]
# every kanji word is followed by its own reading, as on pages that print ruby inline
RUBY = ["朝あさの光ひかりが差さし込こんで", "夜明よあけの空そらに手てを伸のばす", "遠とおい約束やくそくを信しんじて", "感情かんじょうが溢あふれる"]


def write_candidates(root: Path, stem: str, texts: list[list[str]]) -> None:
    folder = root / "candidates" / stem
    folder.mkdir(parents=True)
    meta = []
    for idx, lines in enumerate(texts):
        (folder / f"candidate_{idx}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        meta.append({"url": f"https://example.com/{idx}", "host": f"h{idx}.example.com", "source": f"s{idx}", "line_count": len(lines)})
    (folder / "candidates.json").write_text(json.dumps(meta), encoding="utf-8")


class StripPageNoiseTests(unittest.TestCase):
    def test_drops_title_header_and_credit_lines(self):
        raw = ["詞遠(Shion) - 月夜(Moon) 中日歌詞", "Lyrics : 誰か", "作詞・作曲：誰か", "作曲：誰か", *BODY]
        self.assertEqual(clean_original_lines(raw), BODY)

    def test_strips_annotation_marks_and_footnote_lines(self):
        raw = ["朝の光が差し込んで 註1", "僕らは歩き出した（注2）", "夜明けの空に ※意訳", "註1：補足の説明", "※ほかの説明", "遠い約束を信じて"]
        self.assertEqual(
            clean_original_lines(raw),
            ["朝の光が差し込んで", "僕らは歩き出した", "夜明けの空に", "遠い約束を信じて"],
        )
        self.assertEqual(strip_annotations("言葉 註12"), "言葉")

    def test_drops_chinese_translation_and_romaji_keeps_layout(self):
        raw = []
        for line in ["朝の光が差し込んで", "僕らは歩き出した"]:
            raw += [line, "早晨的陽光照射進來了", "asa no hikari ga"]
        raw.insert(2, "")
        self.assertEqual(clean_original_lines(raw), ["朝の光が差し込んで", "", "僕らは歩き出した"])

    def test_keeps_short_kanji_only_japanese_line(self):
        self.assertEqual(clean_original_lines(["朝の光が差し込んで", "永遠", "僕らは歩き出した"]), ["朝の光が差し込んで", "永遠", "僕らは歩き出した"])

    def test_no_leading_or_trailing_blank_and_blank_runs_collapse(self):
        self.assertEqual(clean_original_lines(["", "朝の光が", "", "", "夜の空に", ""]), ["朝の光が", "", "夜の空に"])


class FuriganaTests(unittest.TestCase):
    def test_detects_inline_furigana_not_plain_text(self):
        self.assertTrue(has_inline_furigana(RUBY))
        self.assertFalse(has_inline_furigana(BODY))


class AcceptWritesCleanedTextTests(unittest.TestCase):
    def test_accept_writes_cleaned_original_not_raw_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_candidates(root, "Song", [["Title - 曲 中日歌詞", "Lyrics : 誰か", *BODY, "註1：補足"]])
            accept_candidate(root, "Song", 0)
            self.assertEqual((root / "txt" / "Song.lyrics.txt").read_text(encoding="utf-8"), "\n".join(BODY) + "\n")
            prov = json.loads((root / "report" / "Song.sources.json").read_text(encoding="utf-8"))
            self.assertEqual(prov["cleaning"]["furigana"], "none")

    def test_agreeing_candidate_without_furigana_replaces_furigana_text(self):
        plain = ["朝の光が差し込んで", "夜明けの空に手を伸ばす", "遠い約束を信じて", "感情が溢れる"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_candidates(root, "Song", [RUBY, plain])
            accept_candidate(root, "Song", 0)
            self.assertEqual((root / "txt" / "Song.lyrics.txt").read_text(encoding="utf-8"), "\n".join(plain) + "\n")
            prov = json.loads((root / "report" / "Song.sources.json").read_text(encoding="utf-8"))
            self.assertEqual(prov["cleaning"]["furigana"], "replaced_by_agreeing_candidate")
            self.assertEqual(prov["cleaning"]["furigana_from_idx"], 1)

    def test_unrelated_candidate_is_not_used_for_furigana(self):
        other = ["さよならの朝に花が咲く", "遠い空の向こう側へ", "夢の続きを探している", "雨上がりの虹を見たくて"]
        with tempfile.TemporaryDirectory() as tmp:
            lines, cleaning = (lambda r: (write_candidates(r, "S", [RUBY, other]), cleaned_candidate_lines(r / "candidates" / "S", 0, 2))[1])(Path(tmp))
            self.assertNotEqual(cleaning["furigana"], "replaced_by_agreeing_candidate")
            self.assertFalse(any("さよなら" in line for line in lines))

    def test_furigana_without_agreeing_candidate_is_kept_and_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_candidates(root, "S", [RUBY])
            lines, cleaning = cleaned_candidate_lines(root / "candidates" / "S", 0, 1)
            self.assertEqual(lines, RUBY)
            self.assertEqual(cleaning["furigana"], "unresolved")


if __name__ == "__main__":
    unittest.main()
