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


def post_candidate(title=TITLE, orig=ORIG, trans=TRANS, artist_confirmed=True):
    lines = [LyricPair(o, f"【{t}】") for o, t in zip(orig, trans)]
    return LyricCandidate(source="google:home.gamer.com.tw", url="https://home.gamer.com.tw/artwork.php?sn=1",
                          title=title, artist="", lines=lines, meta={"artist_confirmed": artist_confirmed})


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

    def test_post_rows_triple_layout_drops_romaji(self):
        body = "".join(f"<p>{o}</p><p>romaji {i} line here</p><p>{t}</p>" for i, (o, t) in enumerate(zip(ORIG, TRANS)))
        rows = tp.extract_post_rows(f"<html><body><p>作詞：誰か</p>{body}<p>#歌詞翻譯#テスト</p><p>留言</p></body></html>")
        self.assertEqual([(r.original, r.translation) for r in rows], list(zip(ORIG, TRANS)))

    def test_post_rows_stanza_blocks_align_by_line(self):
        body = "".join(f"<p>{ORIG[i]}</p><p>{ORIG[i + 1]}</p><p>{TRANS[i]}</p><p>{TRANS[i + 1]}</p>" for i in (0, 2, 4))
        rows = tp.extract_post_rows(f"<html><body><p>歌：テスト</p>{body}<p>謝謝大家觀看這篇翻譯貼文</p></body></html>")
        self.assertEqual([(r.original, r.translation) for r in rows], list(zip(ORIG[:6], TRANS[:6])))

    def test_post_rows_uneven_stanza_left_for_oogg_and_note_not_a_row(self):
        body = (f"<p>{ORIG[0]}</p><p>{ORIG[1]}</p><p>{TRANS[0]}</p>"          # 2 originals, 1 translation line
                f"<p>{ORIG[2]}</p><p>{TRANS[2]}</p>"
                f"<p>{ORIG[3]}</p><p>{TRANS[3]}</p>")
        note = "標題不翻譯的原因是「クラムボン」這個詞到現在仍然沒有明確的定義所以這邊暫且先保留不翻譯，希望大家多多包涵與指教喔"
        rows = tp.extract_post_rows(f"<html><body>{body}<p>{note}</p></body></html>")
        self.assertEqual([(r.original, r.translation) for r in rows],
                         [(ORIG[0], None), (ORIG[1], None), (ORIG[2], TRANS[2]), (ORIG[3], TRANS[3])])

    def test_romaji_only_post_is_not_complete(self):
        rows = [{"original": f"kyou mo shinzou ni ame wo {i}", "translation": t} for i, t in enumerate(TRANS)]
        self.assertFalse(tp.post_is_complete(rows))

    def test_title_recognition(self):
        self.assertTrue(tp.is_translation_post_title(TITLE, IDENT))
        self.assertFalse(tp.is_translation_post_title("そらのうた 歌詞 - uta-net", IDENT))  # no translation marker
        self.assertFalse(tp.is_translation_post_title("【歌詞翻譯】「べつのうた」 - 巴哈姆特", IDENT))  # other song
        self.assertFalse(tp.is_translation_post_title("", IDENT))
        for marker in ("【歌詞中文翻譯】そらのうた／テスト", "【知声】そらのうた【中、日、羅歌詞】", "そらのうた 中日歌詞", "【歌词翻译】そらのうた"):
            self.assertTrue(tp.is_translation_post_title(marker, IDENT), marker)
        # a song-sharing post that merely mentions Chinese-Japanese lyrics is not a translation post
        self.assertFalse(tp.is_translation_post_title("【音樂分享】そらのうた(三首附中日文歌詞)", IDENT))
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

    def test_post_for_another_artist_is_not_accepted_without_a_real_transcript(self):
        self.save([post_candidate(artist_confirmed=False)])
        result = autopick.autopick_song(self.out, "song", "한글자막 by 한효정\n" * 9, 249.0)
        self.assertEqual((result["decision"], result["reason"]), ("review", "translation_post_artist_unconfirmed"))

    def test_unconfirmed_artist_is_fine_when_a_real_transcript_matches(self):
        self.save([post_candidate(artist_confirmed=False)])
        result = autopick.autopick_song(self.out, "song", "\n".join(ORIG), 249.0)
        self.assertEqual(result["decision"], "accept")

    def test_names_artist_uses_artist_and_folder_hints_not_the_title(self):
        ident = TrackIdentity(title="そらのうた", artist="テスト歌手", hints=("そらのうた", "Some Album - Original Soundtrack"))
        self.assertTrue(tp.names_artist(ident, "【歌詞翻譯】そらのうた - テスト歌手"))
        self.assertTrue(tp.names_artist(ident, "そらのうた  some album ED"))
        self.assertFalse(tp.names_artist(ident, "【知声】そらのうた 唄：知声"))

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


class EnglishTitledPostTests(unittest.TestCase):
    """A post titled with the song's English name only: its body still names the tag title and artist."""

    IDENT_ARTIST = TrackIdentity(title="そらのうた", artist="テスト歌手")
    ENGLISH_TITLE = "【歌詞翻譯】《テスト劇》收錄曲 - Song Of The Sky (中/日/羅馬拼音) - tester的創作 - 巴哈姆特"

    class Client:
        def __init__(self, html):
            self.html = html

        def get_text(self, url, params=None, headers=None, retries=None):
            return self.html

    def page(self, title, head):
        body = "".join(f"<p>{o}</p><p>【{t}】</p>" for o, t in zip(ORIG, TRANS))
        return f"<html><head><title>{title}</title></head><body><p>{head}</p>{body}</body></html>"

    def candidate(self, html):
        import lyricsee_core as core
        return core._page_candidate(self.IDENT_ARTIST, self.Client(html), "https://home.gamer.com.tw/artwork.php?sn=2", "t", "")

    def test_body_naming_title_and_artist_makes_the_post_match(self):
        candidate = self.candidate(self.page(self.ENGLISH_TITLE, "そらのうた / テスト歌手"))
        self.assertIsNotNone(candidate)
        import lyricsee_core as core
        rows = core._sanitize_candidate_lines(candidate.lines, self.IDENT_ARTIST)
        self.assertEqual([row.original for row in rows], ORIG)
        # the header line that names the song joins the first stanza, which then loses its page translation
        self.assertGreaterEqual(sum(1 for row in rows if row.translation), len(ORIG) - 1)
        self.assertTrue(candidate.meta["artist_confirmed"])

    def test_body_without_the_artist_is_still_rejected(self):
        self.assertIsNone(self.candidate(self.page(self.ENGLISH_TITLE, "そらのうた / べつの歌手")))

    def test_body_without_the_title_is_still_rejected(self):
        self.assertIsNone(self.candidate(self.page(self.ENGLISH_TITLE, "テスト歌手")))

    def test_title_without_a_translation_marker_is_still_rejected(self):
        self.assertIsNone(self.candidate(self.page("Song Of The Sky - 巴哈姆特", "そらのうた / テスト歌手")))

    def test_title_check_with_body_text(self):
        ident = self.IDENT_ARTIST
        self.assertFalse(tp.is_translation_post_title(self.ENGLISH_TITLE, ident))
        self.assertTrue(tp.is_translation_post_title(self.ENGLISH_TITLE, ident, "そらのうた テスト歌手"))


if __name__ == "__main__":
    unittest.main()
