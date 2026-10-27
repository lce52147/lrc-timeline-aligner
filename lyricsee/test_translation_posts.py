"""Tests for lyric-translation posts: recognising them, accepting them past the snippet rule, writing the human translation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import autopick
import translation_posts as tp
from candidate_review import save_candidates
from lyricsee_core import LyricCandidate, LyricPair, TrackIdentity, extract_bilingual_pairs

# synthetic song, not a real lyric
ORIG = ["あおいそらをみあげて", "しろいくもがながれる", "とおいまちのひかりが", "ゆれているよまどべで", "きみのなまえをよんだ", "こえはかぜにとけた", "ねむるまえのうたを"]
TRANS = ["仰望著蔚藍的天空", "白雲緩緩流動", "遠方城鎮的燈火", "在窗邊搖曳著", "我呼喚了你的名字", "聲音融入了風中", "入睡之前的歌"]
TITLE = "【歌詞翻譯】「そらのうた」- テスト歌手 - tester的創作 - 巴哈姆特"
IDENT = TrackIdentity(title="そらのうた")


def post_html(wrap=("【", "】"), inline=False) -> str:
    body = ""
    for o, t in zip(ORIG, TRANS):
        if inline:
            body += f"<p>{o}{wrap[0]}{t}{wrap[1]}</p>"
        else:
            body += f"<p>{o}</p><p>{wrap[0]}{t}{wrap[1]}</p>"
    return f"<html><head><title>{TITLE}</title></head><body>{body}</body></html>"


def post_candidate(title=TITLE, orig=ORIG, trans=TRANS):
    lines = [LyricPair(o, f"【{t}】") for o, t in zip(orig, trans)]
    return LyricCandidate(source="google:home.gamer.com.tw", url="https://home.gamer.com.tw/artwork.php?sn=1",
                          title=title, artist="", lines=lines)


class ParseTests(unittest.TestCase):
    def test_unwrap_translation_brackets(self):
        self.assertEqual(tp.unwrap_translation("【吶　如果永恆真的存在】"), "吶　如果永恆真的存在")
        self.assertEqual(tp.unwrap_translation("「你好」"), "你好")
        self.assertEqual(tp.unwrap_translation("（你好）"), "你好")
        self.assertEqual(tp.unwrap_translation("【甲】和【乙】"), "【甲】和【乙】")
        self.assertEqual(tp.unwrap_translation("沒有括號"), "沒有括號")

    def test_split_inline_translation_styles(self):
        for left, right in (("【", "】"), ("「", "」"), ("（", "）"), ("(", ")")):
            got = tp.split_inline_translation(f"あおいそら{left}藍色的天空{right}")
            self.assertEqual(got, ("あおいそら", "藍色的天空"))
        self.assertIsNone(tp.split_inline_translation("あおいそら（ねぇ）"))
        self.assertIsNone(tp.split_inline_translation("藍色的天空【藍】"))

    def test_extract_pairs_from_interleaved_post_then_unwrap(self):
        rows = tp.clean_pairs(extract_bilingual_pairs(post_html()))
        self.assertEqual(len(rows), len(ORIG))
        self.assertEqual(rows[0], {"original": ORIG[0], "translation": TRANS[0]})

    def test_extract_pairs_from_inline_bracket_post(self):
        for wrap in (("【", "】"), ("「", "」"), ("（", "）")):
            rows = tp.clean_pairs(extract_bilingual_pairs(post_html(wrap=wrap, inline=True)))
            self.assertEqual([r["translation"] for r in rows], TRANS)
            self.assertEqual([r["original"] for r in rows], ORIG)

    def test_title_recognition(self):
        self.assertTrue(tp.is_translation_post_title(TITLE, IDENT))
        self.assertFalse(tp.is_translation_post_title("そらのうた 歌詞 - uta-net", IDENT))  # no translation marker
        self.assertFalse(tp.is_translation_post_title("【歌詞翻譯】「べつのうた」 - 巴哈姆特", IDENT))  # other song
        self.assertFalse(tp.is_translation_post_title("", IDENT))
        self.assertEqual(tp.post_author(TITLE), "tester")

    def test_post_shape(self):
        rows = [{"original": o, "translation": t} for o, t in zip(ORIG, TRANS)]
        self.assertTrue(tp.post_is_complete(rows))
        self.assertFalse(tp.post_is_complete(rows[:5]))
        self.assertFalse(tp.post_is_complete([{**r, "translation": ""} for r in rows]))

    def test_transcript_reliability(self):
        self.assertTrue(tp.transcript_reliable("\n".join(ORIG)))
        self.assertFalse(tp.transcript_reliable("한글자막 by 한효정\n" * 9))
        self.assertFalse(tp.transcript_reliable("ご視聴ありがとうございました\nチャンネル登録お願いします"))
        self.assertFalse(tp.transcript_reliable("あい\nう"))  # < 30 Japanese chars
        self.assertFalse(tp.transcript_reliable(""))

    def test_align_exact_fuzzy_and_missing(self):
        rows = [{"original": o, "translation": t} for o, t in zip(ORIG, TRANS)]
        lines = [ORIG[1], ORIG[0][:-1] + "てて", "ぜんぜんちがう一行"]
        got = tp.align_translations(lines, rows)
        self.assertEqual(got[0], TRANS[1])
        self.assertEqual(got[1], TRANS[0])
        self.assertIsNone(got[2])


class SaveAndAcceptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, cands, identity=IDENT):
        save_candidates(self.out, "song", cands, "no_original", identity=identity)

    def test_save_flags_post_and_stores_rows(self):
        self.save([post_candidate()])
        meta = json.loads((self.out / "candidates/song/candidates.json").read_text(encoding="utf-8"))[0]
        self.assertTrue(meta["translation_post"])
        self.assertEqual(meta["page_author"], "tester")
        rows = json.loads((self.out / "candidates/song/candidate_0.pairs.json").read_text(encoding="utf-8"))
        self.assertEqual(rows[0]["translation"], TRANS[0])  # wrapper removed

    def test_save_without_identity_or_for_other_song_is_not_a_post(self):
        self.save([post_candidate()], identity=None)
        meta = json.loads((self.out / "candidates/song/candidates.json").read_text(encoding="utf-8"))[0]
        self.assertNotIn("translation_post", meta)
        self.save([post_candidate()], identity=TrackIdentity(title="べつのうた"))
        meta = json.loads((self.out / "candidates/song/candidates.json").read_text(encoding="utf-8"))[0]
        self.assertNotIn("translation_post", meta)

    def test_few_line_post_accepted_despite_snippet_rule_when_asr_heard_nothing(self):
        self.save([post_candidate()])
        # 7 short lines in a 249 s song: below 1.0 chars/s, so the ASR rule alone calls it a snippet
        result = autopick.autopick_song(self.out, "song", "한글자막 by 한효정\n" * 9, 249.0)
        self.assertEqual((result["decision"], result["reason"]), ("accept", "translation_post_title_match"))
        txt = (self.out / "txt/song.lyrics.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(txt), len(ORIG))
        self.assertTrue(txt[0].startswith(ORIG[0] + "  "))
        sources = json.loads((self.out / "report/song.sources.json").read_text(encoding="utf-8"))
        self.assertEqual(sources["translation_method"], "page_human")
        self.assertEqual(sources["translation_author"], "tester")
        self.assertEqual(sources["translation_source_url"], "https://home.gamer.com.tw/artwork.php?sn=1")

    def test_real_transcript_of_another_song_vetoes_the_post(self):
        self.save([post_candidate()])
        other = "\n".join(["さよならのあさにはながさく", "とおいそらのむこうがわへ", "ゆめのつづきをさがしている",
                           "あめあがりのにじをみたくて", "ひとりきりであるきだした", "あなたのこえがきこえたよ"])
        result = autopick.autopick_song(self.out, "song", other, 249.0)
        self.assertEqual((result["decision"], result["reason"]), ("review", "translation_post_contradicted_by_asr"))
        self.assertFalse((self.out / "txt/song.lyrics.txt").exists())

    def test_matching_transcript_accepts_the_post(self):
        self.save([post_candidate()])
        result = autopick.autopick_song(self.out, "song", "\n".join(ORIG), 249.0)
        self.assertEqual(result["decision"], "accept")

    def test_short_post_is_not_accepted(self):
        self.save([post_candidate(orig=ORIG[:5], trans=TRANS[:5])])
        result = autopick.autopick_song(self.out, "song", "한글자막 by 한효정\n" * 9, 249.0)
        self.assertEqual(result["decision"], "review")

    def test_non_post_snippet_is_still_a_snippet(self):
        cand = LyricCandidate(source="google:x.example", url="https://x.example/1", title="歌詞 そらのうた",
                              artist="", lines=[LyricPair(o, t) for o, t in zip(ORIG, TRANS)])
        self.save([cand])
        result = autopick.autopick_song(self.out, "song", "\n".join(ORIG), 249.0)
        self.assertEqual((result["decision"], result["reason"]), ("review", "no_full_length_candidate"))


class PreferPostTranslationTests(unittest.TestCase):
    def test_post_translation_replaces_machine_translation_for_agreeing_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            other = LyricCandidate(source="uta-net", url="https://www.uta-net.com/song/1/", title="そらのうた",
                                   artist="", lines=[LyricPair(o) for o in ORIG])
            save_candidates(out, "song", [other, post_candidate()], "no_original", identity=IDENT)
            (out / "txt").mkdir()
            # original already accepted from uta-net and machine-translated by OOGG
            (out / "txt/song.lyrics.txt").write_text(
                "\n".join(f"{o}  機翻{i}" for i, o in enumerate(ORIG)) + "\n", encoding="utf-8")
            result = tp.apply_post_translation(out, "song")
            self.assertEqual(result["status"], "applied")
            self.assertEqual(result["translation_method"], "page_human")
            first = (out / "txt/song.lyrics.txt").read_text(encoding="utf-8").splitlines()[0]
            self.assertTrue(first.startswith(ORIG[0] + "  "))
            self.assertNotIn("機翻", first)

    def test_unrelated_post_is_not_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_candidates(out, "song", [post_candidate()], "no_original", identity=IDENT)
            (out / "txt").mkdir()
            different = ["さよならのあさにはなが", "とおいそらのむこうがわ", "ゆめのつづきをさがして", "あめあがりのにじを",
                         "ひとりきりであるきだ", "あなたのこえがきこえ", "ほしふるよるにてをのば"]
            (out / "txt/song.lyrics.txt").write_text("\n".join(different) + "\n", encoding="utf-8")
            self.assertEqual(tp.apply_post_translation(out, "song")["status"], "no_post")

    def test_lines_the_post_lacks_are_filled_by_oogg_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_candidates(out, "song", [post_candidate()], "no_original", identity=IDENT)
            (out / "txt").mkdir()
            lines = ORIG + ["あたらしいいちぎょう"]
            (out / "txt/song.lyrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(tp.apply_post_translation(out, "song")["status"], "uncovered_lines")
            result = tp.apply_post_translation(out, "song", oogg_fill=lambda rows: ["機翻一行"] * len(rows))
            self.assertEqual((result["from_page"], result["oogg"]), (len(ORIG), 1))
            self.assertEqual(result["translation_method"], "page_human+oogg_fill")


if __name__ == "__main__":
    unittest.main()
