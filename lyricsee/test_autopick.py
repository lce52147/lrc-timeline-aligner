"""Tests for candidate cleaning, character similarity and the auto-pick decision rule (ASR is mocked)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import autopick
from candidate_review import compute_similarity, save_candidates
from candidate_text import asr_match, char_similarity, clean_lyric_lines, detect_language, fold
from lyricsee_core import LyricCandidate, LyricPair

SONG_A = [
    "枯れてしまって消えない恋を",
    "僕は捨てられないまま",
    "一人部屋で書き連ねた",
    "月夜が満ちて僕ら",
    "恋をしていたんだ",
    "瞬きはしないで",
    "独りはもう散々なんだ",
    "季節が回るその隣で",
    "笑う君はかわいくて",
    "風の香り空の高さ",
] * 3
SONG_B = [
    "さよならの朝に花が咲く",
    "遠い空の向こう側へ",
    "夢の続きを探している",
    "雨上がりの虹を見たくて",
    "ひとりきりで歩き出した",
    "あなたの声が聞こえたよ",
    "星降る夜に手を伸ばして",
    "届かない願いを抱いて",
] * 3


def mixed(lines: list[str]) -> list[str]:
    """Candidate as the sites give it: header, Japanese line + Chinese translation + romaji."""
    out = ["作詞：誰か", "歌詞 - 中日歌詞", "[Chorus]", ""]
    for line in lines:
        out += [line, "這是中文翻譯的一行", "kore wa romaji no ichigyou"]
    return out


class CleaningTests(unittest.TestCase):
    def test_drops_translation_romaji_headers_and_blanks(self):
        cleaned = clean_lyric_lines(mixed(["月夜が満ちて僕ら", "恋をしていたんだ"]))
        self.assertEqual(cleaned, ["月夜が満ちて僕ら", "恋をしていたんだ"])

    def test_english_song_keeps_latin_lines_only(self):
        lines = ["Hold me in the dark", "在黑暗中抱緊我", "Never let me go", "永遠不要放開我"]
        self.assertEqual(detect_language(lines), "en")
        self.assertEqual(clean_lyric_lines(lines), ["Hold me in the dark", "Never let me go"])

    def test_fold_normalises_width_kana_case_and_punctuation(self):
        self.assertEqual(fold("ＡＢｃ・カタカナ ー！"), fold("abcかたかな"))
        self.assertEqual(fold("「ほんとう」、だね。"), "ほんとうだね")


class SimilarityTests(unittest.TestCase):
    def test_same_song_with_and_without_translation_is_high(self):
        self.assertGreater(compute_similarity(mixed(SONG_A), SONG_A), 0.95)

    def test_different_songs_are_low(self):
        self.assertLess(compute_similarity(SONG_A, SONG_B), 0.2)

    def test_chorus_written_once_still_agrees_with_full(self):
        once = list(dict.fromkeys(SONG_A))
        self.assertGreater(char_similarity(fold("".join(once)), fold("".join(SONG_A))), 0.5)

    def test_save_candidates_stores_cleaned_similarity(self):
        def cand(lines, url):
            return LyricCandidate(
                url=url, source="t", title="t", artist="a",
                lines=[LyricPair(original=x, translation="") for x in lines],
            )

        with tempfile.TemporaryDirectory() as tmp:
            save_candidates(Path(tmp), "s", [cand(mixed(SONG_A), "https://a/1"), cand(SONG_A, "https://b/2")])
            meta = json.loads((Path(tmp) / "candidates" / "s" / "candidates.json").read_text(encoding="utf-8"))
            self.assertGreater(meta[0]["similarity_to_others"]["1"], 0.95)


class AsrMatchTests(unittest.TestCase):
    def test_right_song_beats_wrong_song(self):
        transcript = fold("".join(SONG_A))
        right = asr_match(fold("".join(SONG_A)), transcript)["score"]
        wrong = asr_match(fold("".join(SONG_B)), transcript)["score"]
        self.assertGreater(right, 0.95)
        self.assertLess(wrong, 0.2)

    def test_snippet_has_low_coverage(self):
        transcript = fold("".join(SONG_A))
        self.assertLess(asr_match(fold("".join(SONG_A[:2])), transcript)["coverage"], 0.4)

    def test_empty_inputs_score_zero(self):
        self.assertEqual(asr_match("", "abc")["score"], 0.0)

    def test_snippet_rule_scales_with_duration(self):
        self.assertTrue(autopick.is_snippet(200, 240.0))
        self.assertFalse(autopick.is_snippet(300, 240.0))
        self.assertTrue(autopick.is_snippet(30, None))


def entry(idx, url, score, sim=None, snippet=False):
    return {"idx": idx, "url": url, "snippet": snippet, "asr": {"score": score}, "sim": sim or {}}


class DecisionTests(unittest.TestCase):
    def test_accept_when_best_and_second_source_agrees(self):
        verdict = autopick.decide([
            entry(0, "u0", 0.80, {1: 0.9, 2: 0.1}),
            entry(1, "u1", 0.75, {0: 0.9, 2: 0.1}),
            entry(2, "u2", 0.30, {0: 0.1, 1: 0.1}),
        ])
        self.assertEqual((verdict["decision"], verdict["idx"]), ("accept", 0))
        self.assertEqual(verdict["agreeing_idx"], [1])

    def test_agreeing_copy_is_not_a_rival_for_the_margin(self):
        verdict = autopick.decide([
            entry(0, "u0", 0.80, {1: 0.9}),
            entry(1, "u1", 0.78, {0: 0.9}),
        ])
        self.assertEqual(verdict["decision"], "accept")

    def test_single_candidate_needs_strict_floor(self):
        mid = (autopick.ASR_FLOOR + autopick.ASR_STRICT_FLOOR) / 2
        self.assertEqual(autopick.decide([entry(0, "u0", mid)])["reason"], "no_independent_agreement")
        high = autopick.decide([entry(0, "u0", autopick.ASR_STRICT_FLOOR + 0.05)])
        self.assertEqual((high["decision"], high["reason"]), ("accept", "asr_above_strict_floor"))

    def test_below_floor_is_review(self):
        verdict = autopick.decide([entry(0, "u0", autopick.ASR_FLOOR - 0.05)])
        self.assertEqual(verdict["reason"], "asr_below_floor")

    def test_close_rival_blocks_even_a_high_score(self):
        verdict = autopick.decide([
            entry(0, "u0", 0.85, {1: 0.1}),
            entry(1, "u1", 0.80, {0: 0.1}),
        ])
        self.assertEqual((verdict["decision"], verdict["reason"]), ("review", "asr_margin_too_small"))

    def test_snippets_are_ignored(self):
        verdict = autopick.decide([
            entry(0, "u0", 0.95, snippet=True),
            entry(1, "u1", 0.60, {2: 0.9}),
            entry(2, "u2", 0.58, {1: 0.9}),
        ])
        self.assertEqual((verdict["decision"], verdict["idx"]), ("accept", 1))
        only_snippets = autopick.decide([entry(0, "u0", 0.95, snippet=True)])
        self.assertEqual(only_snippets["reason"], "no_full_length_candidate")

    def test_same_url_does_not_count_as_independent(self):
        verdict = autopick.decide([
            entry(0, "same", 0.30, {1: 0.99}),
            entry(1, "same", 0.30, {0: 0.99}),
        ])
        self.assertEqual(verdict["decision"], "review")


class EndToEndTests(unittest.TestCase):
    def _write(self, root: Path, stem: str, candidates: dict[int, list[str]]):
        cdir = root / "candidates" / stem
        cdir.mkdir(parents=True)
        meta = []
        for idx, lines in candidates.items():
            (cdir / f"candidate_{idx}.txt").write_text("\n".join(lines), encoding="utf-8")
            meta.append({"url": f"https://h{idx}/x", "host": f"h{idx}", "source": "s", "line_count": len(lines)})
        (cdir / "candidates.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_accepts_and_writes_auto_consensus_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "song", {0: mixed(SONG_A), 1: SONG_A, 2: SONG_B})
            result = autopick.autopick_song(root, "song", "\n".join(SONG_A), duration=120.0)
            self.assertEqual((result["decision"], result["idx"]), ("accept", 0))
            prov = json.loads((root / "report" / "song.sources.json").read_text(encoding="utf-8"))
            self.assertEqual(prov["verification_status"], "auto_consensus")
            self.assertFalse(prov["human_approved"])
            self.assertIn("asr", prov["autopick"])
            self.assertTrue((root / "txt" / "song.lyrics.txt").read_text(encoding="utf-8").endswith("\n"))
            meta = json.loads((root / "candidates" / "song" / "candidates.json").read_text(encoding="utf-8"))
            self.assertIn("asr", meta[2])
            self.assertLess(meta[2]["asr"]["score"], 0.3)

    def test_wrong_song_transcript_leaves_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "song", {0: SONG_A, 1: SONG_A})
            result = autopick.autopick_song(root, "song", "\n".join(SONG_B), duration=120.0)
            self.assertEqual(result["decision"], "review")
            self.assertFalse((root / "txt" / "song.lyrics.txt").exists())


if __name__ == "__main__":
    unittest.main()
