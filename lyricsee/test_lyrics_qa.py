"""Tests for the lyric QA gate. Synthetic text only: no real lyrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lyrics_qa as qa
import translation_posts as tp
from candidate_text import clean_lyric_lines, clean_original_lines
from lyricsee_core import LyricCandidate, LyricPair, write_result
from translation_oogg import translate_oogg, translate_oogg_line

ORIG = ["あおいそらをみあげて", "しろいくもがながれる", "とおいまちのひかりが", "ゆれているよまどべで",
        "きみのなまえをよんだ", "こえはかぜにとけた", "ねむるまえのうたを", "あしたはきっとはれる"]
TRANS = ["仰望著蔚藍的天空", "白雲緩緩流動著", "遠方城鎮的燈火", "在窗邊搖曳著", "我呼喚了你的名字", "聲音融入了風中",
         "入睡之前的歌", "明天一定會放晴"]
NOTE = "【例詞】，指一個很長的說明。我認為它與前面的用字相當契合，此外，我在整張專輯皆選用【舊夢】作為翻譯。原因是第一首的後半。"


def rows(originals=ORIG, translations=TRANS):
    return list(zip(originals, translations))


def found_at(issues, kind):
    return [issue.line for issue in issues if issue.kind == kind]


class CheckTests(unittest.TestCase):
    def test_clean_lyric_passes(self):
        self.assertEqual(qa.check(rows()), [])

    def test_note_leak_chinese_commentary_in_original_column(self):
        found = qa.check(rows(ORIG[:3] + [NOTE] + ORIG[3:], TRANS[:3] + ["說明"] + TRANS[3:]))
        self.assertEqual(found_at(found, "note_leak"), [4])

    def test_note_leak_track_list_row(self):
        found = qa.check(rows(ORIG + ["2.あるきょく"], TRANS + ["2.某曲"]))
        self.assertEqual(found_at(found, "note_leak"), [9])

    def test_note_leak_commentary_in_translation_column(self):
        found = qa.check(rows(ORIG[:2] + ["ふつうのいちぎょう"] + ORIG[2:], TRANS[:2] + [NOTE] + TRANS[2:]))
        self.assertEqual(found_at(found, "note_leak"), [3])

    def test_japanese_line_and_digits_are_not_notes(self):
        self.assertFalse(qa.is_note_line("あなたの愛を信じてる"))
        self.assertFalse(qa.is_track_row("1 2 3 でいこう"))

    def test_bleed_translation_repeats_previous_line(self):
        translations = list(TRANS)
        translations[3] = TRANS[2] + "的光芒"     # 6 chars of the previous translation
        found = [i for i in qa.check(rows(translations=translations)) if i.kind == "bleed"]
        self.assertEqual([(i.line, i.strong) for i in found], [(4, True)])

    def test_bleed_identical_translation_for_different_originals_is_strong(self):
        translations = list(TRANS)
        translations[2] = TRANS[1]
        found = [i for i in qa.check(rows(translations=translations)) if i.kind == "bleed"]
        self.assertEqual([(i.line, i.strong) for i in found], [(3, True)])

    def test_repeated_chorus_is_not_bleed(self):
        found = qa.check(rows(ORIG[:2] + ORIG[:2] + ORIG[2:], TRANS[:2] + TRANS[:2] + TRANS[2:]))
        self.assertEqual(found_at(found, "bleed"), [])

    def test_order_flags_lines_appended_after_the_reference_end(self):
        found = qa.check(rows(ORIG + ["みしらぬさいごのいちぎょう"], TRANS + ["未知的最後一行"]), reference_original=ORIG)
        self.assertEqual(found_at(found, "order"), [9])

    def test_order_flags_a_moved_block(self):
        moved = ORIG[:2] + ORIG[5:8] + ORIG[2:5]
        found = qa.check(rows(moved, TRANS), reference_original=ORIG)
        self.assertTrue(found_at(found, "order"))

    def test_order_ignores_a_reference_that_explains_few_lines(self):
        other = ["ぜんぜんちがううた", "べつのかしのれつ", "なにもかさならない"]
        self.assertEqual(found_at(qa.check(rows(), reference_original=other), "order"), [])

    def test_jp_variant_chinese_form_in_a_kana_line(self):
        originals = list(ORIG)
        originals[1] = "ひとつに繫ごう"
        reference = list(ORIG)
        reference[1] = "ひとつに繋ごう"
        found = qa.check(rows(originals), reference_original=reference)
        self.assertEqual(found_at(found, "jp_variant"), [2])

    def test_jp_variant_ignores_a_plain_japanese_kanji_line(self):
        originals = list(ORIG)
        originals[1] = "ひとつに繋ごう"
        self.assertEqual(found_at(qa.check(rows(originals)), "jp_variant"), [])

    def test_untranslated_empty_equal_and_left_over_kana(self):
        translations = list(TRANS)
        translations[0] = ""
        translations[1] = ORIG[1]
        translations[2] = "とおいまちのひかりが光"
        self.assertEqual(found_at(qa.check(rows(translations=translations)), "untranslated"), [1, 2, 3])

    def test_untranslated_english_left_in_translation(self):
        originals = ORIG[:2] + ["Hold me close tonight"] + ORIG[2:]
        translations = TRANS[:2] + ["Hold me close tonight"] + TRANS[2:]
        self.assertEqual(found_at(qa.check(rows(originals, translations)), "untranslated"), [3])

    def test_original_only_lyric_is_not_judged_for_translation(self):
        self.assertEqual(qa.check([(o, "") for o in ORIG], translated=False), [])

    def test_ratio_flags_a_translation_far_longer_than_the_songs_own(self):
        translations = list(TRANS)
        translations[4] = "我呼喚了你的名字" * 6
        self.assertEqual(found_at(qa.check(rows(translations=translations)), "ratio"), [5])


class GateTests(unittest.TestCase):
    def test_gate_drops_note_rows_and_never_keeps_them(self):
        pairs = [LyricPair(o, t) for o, t in rows(ORIG + [NOTE, "5.あるきょく"], TRANS + ["說明", "5.某曲"])]
        kept, issues, dropped = qa.gate(pairs)
        self.assertEqual([p.original for p in kept], ORIG)
        self.assertEqual(sorted(i.line for i in dropped), [9, 10])
        self.assertEqual(issues, [])

    def test_gate_retranslates_a_bleed_line_alone_and_rechecks(self):
        translations = list(TRANS)
        translations[2] = TRANS[1]
        pairs = [LyricPair(o, t) for o, t in rows(translations=translations)]
        asked: list[str] = []

        def retranslate(line):
            asked.append(line)
            return TRANS[2]

        kept, issues, _ = qa.gate(pairs, retranslate=retranslate)
        self.assertEqual(asked, [ORIG[2]])
        self.assertEqual(kept[2].translation, TRANS[2])
        self.assertEqual(issues, [])

    def test_gate_keeps_and_flags_a_line_whose_retranslation_fails(self):
        translations = list(TRANS)
        translations[2] = TRANS[1]
        pairs = [LyricPair(o, t) for o, t in rows(translations=translations)]

        def retranslate(line):
            raise ValueError("model down")

        kept, issues, _ = qa.gate(pairs, retranslate=retranslate)
        self.assertEqual(kept[2].translation, TRANS[1])
        self.assertEqual([(i.kind, i.line) for i in issues], [("bleed", 3)])

    def test_gate_leaves_a_weak_bleed_of_a_human_translation_alone(self):
        translations = list(TRANS)
        translations[3] = "遠方城鎮的"
        pairs = [LyricPair(o, t) for o, t in rows(translations=translations)]
        kept, _, _ = qa.gate(pairs, retranslate=lambda line: "機翻", repair_weak=False)
        self.assertEqual(kept[3].translation, "遠方城鎮的")


class WriteResultGateTests(unittest.TestCase):
    def write(self, lines, method="oogg_draft", status="oogg_draft", **kwargs):
        candidate = LyricCandidate(source="s", url="u", title="t", artist="a", lines=lines)
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "song.flac"
            audio.write_bytes(b"")
            output, provenance = write_result(audio, candidate, output_root=Path(tmp) / "out", overwrite=False,
                                              translation_method=method, status=status, **kwargs)
            return output.read_text(encoding="utf-8"), json.loads(provenance.read_text(encoding="utf-8"))

    def test_clean_song_keeps_its_status(self):
        _, payload = self.write([LyricPair(o, t) for o, t in rows()])
        self.assertEqual(payload["status"], "oogg_draft")
        self.assertNotIn("qa_issues", payload)

    def test_note_row_is_not_written_and_song_needs_review(self):
        lines = [LyricPair(o, t) for o, t in rows(ORIG + [NOTE], TRANS + ["說明"])]
        text, payload = self.write(lines)
        self.assertNotIn("我認為", text)
        self.assertEqual(payload["status"], "needs_review")
        self.assertEqual(payload["status_before_qa"], "oogg_draft")
        self.assertEqual(payload["line_count"], len(ORIG))
        self.assertEqual(payload["qa_dropped_note_lines"][0]["kind"], "note_leak")

    def test_bleed_is_repaired_through_the_retranslate_callback(self):
        translations = list(TRANS)
        translations[2] = TRANS[1]
        text, payload = self.write([LyricPair(o, t) for o, t in rows(translations=translations)],
                                   qa_retranslate=lambda line: TRANS[2])
        self.assertEqual(payload["status"], "oogg_draft")
        self.assertIn(TRANS[2], text.splitlines()[2])

    def test_remaining_issue_is_recorded_not_shipped_silently(self):
        translations = list(TRANS)
        translations[2] = TRANS[1]
        _, payload = self.write([LyricPair(o, t) for o, t in rows(translations=translations)])
        self.assertEqual(payload["status"], "needs_review")
        self.assertEqual([(i["kind"], i["line"]) for i in payload["qa_issues"]], [("bleed", 3)])

    def test_original_only_song_is_not_judged_for_translation(self):
        _, payload = self.write([LyricPair(o) for o in ORIG], method="none", status="original_only")
        self.assertEqual(payload["status"], "original_only")


class ExtractionTests(unittest.TestCase):
    def test_candidate_cleaning_drops_track_rows_and_commentary_with_quoted_kana(self):
        raw = ORIG[:4] + [NOTE] + ORIG[4:] + ["2.あるきょく", "5.べつのきょく"]
        self.assertEqual(clean_original_lines(raw), ORIG)
        self.assertEqual(clean_lyric_lines(raw, "ja"), ORIG)

    def test_extract_post_rows_ignores_notes_and_track_lists(self):
        body = "".join(f"<p>{o}</p><p>{t}</p>" for o, t in zip(ORIG, TRANS))
        html = f"<html><body>{body}<p>{NOTE}</p><p>2.あるきょく</p><p>2.某曲</p></body></html>"
        found = tp.extract_post_rows(html)
        self.assertEqual([(row.original, row.translation) for row in found], list(zip(ORIG, TRANS)))

    def test_stanza_post_with_trailing_commentary_pairs_by_stanza(self):
        body = "".join(f"<p>{o}</p>" for o in ORIG) + "".join(f"<p>{t}</p>" for t in TRANS)
        html = f"<html><body>{body}<p>{NOTE}</p></body></html>"
        found = tp.extract_post_rows(html)
        self.assertEqual([(row.original, row.translation) for row in found], list(zip(ORIG, TRANS)))


class OoggLineTests(unittest.TestCase):
    def test_line_translation_prompt_carries_no_neighbours(self):
        sent = []

        def post_json(url, request, timeout):
            sent.append(request)
            return {"choices": [{"message": {"content": json.dumps({"text": "一行"}, ensure_ascii=False)}}]}

        self.assertEqual(translate_oogg_line("あおいそら", title="t", endpoint="http://x", model="m", post_json=post_json), "一行")
        user = json.loads(sent[0]["messages"][1]["content"])
        self.assertEqual(list(user), ["translate_this_line_only"])

    def test_translate_oogg_repairs_a_bleed_row_without_context(self):
        batch = [LyricPair(o, t) for o, t in rows()]
        batch[2] = LyricPair(ORIG[2], TRANS[1])    # the model translated the previous line again
        with patch("lyricsee_core.translate_with_local_llm", return_value=batch), \
                patch("translation_oogg.translate_oogg_line", return_value=TRANS[2]) as single:
            result = translate_oogg(ORIG, title="t", model="m", endpoint="http://x", review=False)
        single.assert_called_once()
        self.assertEqual(result[2], TRANS[2])


if __name__ == "__main__":
    unittest.main()
