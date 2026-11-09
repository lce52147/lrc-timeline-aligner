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
        found = qa.check(rows(ORIG[:3] + [NOTE] + ORIG[3:], TRANS[:3] + [""] + TRANS[3:]))
        self.assertEqual(found_at(found, "note_leak"), [4])

    def test_note_leak_track_list_row(self):
        found = qa.check(rows(ORIG + ["2.あるきょく", "3.べつのきょく"], TRANS + ["2.某曲", "3.另一曲"]))
        self.assertEqual(found_at(found, "note_leak"), [9, 10])

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
        pairs = [LyricPair(o, t) for o, t in rows(ORIG + [NOTE, "5.あるきょく", "6.べつのきょく"], TRANS + ["", "5.某曲", "6.另一曲"])]
        kept, issues, dropped = qa.gate(pairs)
        self.assertEqual([p.original for p in kept], ORIG)
        self.assertEqual(sorted(i.line for i in dropped), [9, 10, 11])
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


class BleedRepairKeepsOldTests(unittest.TestCase):
    def pairs(self):
        translations = list(TRANS)
        translations[2] = TRANS[1]
        return [LyricPair(o, t) for o, t in rows(translations=translations)]

    def test_new_text_that_still_bleeds_is_dropped_and_the_old_text_stays(self):
        kept, issues, _ = qa.gate(self.pairs(), retranslate=lambda line: TRANS[1])
        self.assertEqual(kept[2].translation, TRANS[1])
        self.assertEqual([(i.kind, i.line) for i in issues], [("bleed", 3)])

    def test_new_text_that_bleeds_into_the_next_row_is_not_clearly_better(self):
        kept, issues, _ = qa.gate(self.pairs(), retranslate=lambda line: TRANS[3])
        self.assertEqual(kept[2].translation, TRANS[1])
        self.assertEqual([(i.kind, i.line) for i in issues], [("bleed", 3)])

    def test_a_fragment_much_shorter_than_expected_is_not_clearly_better(self):
        kept, _, _ = qa.gate(self.pairs(), retranslate=lambda line: "燈")
        self.assertEqual(kept[2].translation, TRANS[1])

    def test_latin_or_cyrillic_garbage_is_not_clearly_better(self):
        kept, _, _ = qa.gate(self.pairs(), retranslate=lambda line: "遠方тюmin的燈火")
        self.assertEqual(kept[2].translation, TRANS[1])
        kept, _, _ = qa.gate(self.pairs(), retranslate=lambda line: "遠方城鎮 the lights")
        self.assertEqual(kept[2].translation, TRANS[1])

    def test_a_weak_bleed_is_report_only_even_with_repair_weak(self):
        translations = list(TRANS)
        translations[3] = "遠方城鎮的呢呢呢呢呢呢呢呢呢"
        pairs = [LyricPair(o, t) for o, t in rows(translations=translations)]
        asked = []
        kept, _, _ = qa.gate(pairs, retranslate=lambda line: asked.append(line) or "在窗邊搖曳著", repair_weak=True)
        self.assertEqual(asked, [])
        self.assertEqual(kept[3].translation, translations[3])

    def test_context_retranslator_gets_neighbours(self):
        seen = {}

        def retranslate(line, context=((), ())):
            seen["context"] = context
            return TRANS[2]
        retranslate.accepts_context = True
        kept, issues, _ = qa.gate(self.pairs(), retranslate=retranslate)
        self.assertEqual(seen["context"], (ORIG[0:2], ORIG[3:5]))
        self.assertEqual(kept[2].translation, TRANS[2])
        self.assertEqual(issues, [])


class ScriptGarbageTests(unittest.TestCase):
    def test_cyrillic_hangul_and_mixed_tokens_fail(self):
        for bad in ("我тюmin你的名字", "한글混入", "我呼喚了ты", "mix안ed"):
            translations = list(TRANS)
            translations[4] = bad
            self.assertEqual(found_at(qa.check(rows(translations=translations)), "script_garbage"), [5], bad)

    def test_clean_chinese_and_a_latin_name_pass(self):
        translations = list(TRANS)
        translations[4] = "我呼喚了 Mujica 的名字"
        self.assertEqual(found_at(qa.check(rows(translations=translations)), "script_garbage"), [])

    def test_a_script_the_original_itself_carries_is_not_garbage(self):
        originals = list(ORIG)
        originals[4] = "きみのなまえтест"
        translations = list(TRANS)
        translations[4] = "你的名字тест"
        self.assertEqual(found_at(qa.check(rows(originals, translations)), "script_garbage"), [])


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
        lines = [LyricPair(o, t) for o, t in rows(ORIG + [NOTE], TRANS + [""])]
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
        html = f"<html><body>{body}<p>{NOTE}</p><p>2.あるきょく</p><p>3.べつのきょく</p><p>2.某曲</p></body></html>"
        found = tp.extract_post_rows(html)
        self.assertEqual([(row.original, row.translation) for row in found], list(zip(ORIG, TRANS)))

    def test_stanza_post_with_trailing_commentary_pairs_by_stanza(self):
        body = "".join(f"<p>{o}</p>" for o in ORIG) + "".join(f"<p>{t}</p>" for t in TRANS)
        html = f"<html><body>{body}<p>{NOTE}</p></body></html>"
        found = tp.extract_post_rows(html)
        self.assertEqual([(row.original, row.translation) for row in found], list(zip(ORIG, TRANS)))


class Fix1Tests(unittest.TestCase):
    """TQ2-FIX1: the filters must not drop real lyric lines."""

    NUMBERED = ["1.こどもあつかいしないでください", "2.やさしくしてごまかすのはもっとだめ"]
    NUMBERED_TRANS = ["1.請不要把我當成小孩對待", "2.用溫柔來敷衍更是不行"]

    def body_with_numbered(self):
        return ORIG[:4] + self.NUMBERED + ORIG[4:], TRANS[:4] + self.NUMBERED_TRANS + TRANS[4:]

    def test_numbered_lyric_lines_inside_the_song_body_are_kept(self):
        originals, translations = self.body_with_numbered()
        self.assertEqual(found_at(qa.check(rows(originals, translations)), "note_leak"), [])
        kept, issues, dropped = qa.gate([LyricPair(o, t) for o, t in rows(originals, translations)])
        self.assertEqual([p.original for p in kept], originals)
        self.assertEqual(dropped, [])
        self.assertEqual(clean_original_lines(originals), originals)
        self.assertEqual(clean_lyric_lines(originals, "ja"), originals)

    def test_numbered_lines_in_a_post_body_stay_rows(self):
        originals, translations = self.body_with_numbered()
        body = "".join(f"<p>{o}</p><p>{t}</p>" for o, t in zip(originals, translations))
        found = tp.extract_post_rows(f"<html><body>{body}</body></html>")
        self.assertEqual([row.original for row in found], originals)

    def test_a_real_tail_track_list_is_dropped_everywhere(self):
        tail = ["1.あるきょく", "2.べつのきょく", "3.さいごのきょく"]
        found = qa.check(rows(ORIG + tail, TRANS + ["1.甲", "2.乙", "3.丙"]))
        self.assertEqual(found_at(found, "note_leak"), [9, 10, 11])
        self.assertEqual(clean_original_lines(ORIG + tail), ORIG)
        self.assertEqual(clean_lyric_lines(ORIG + tail, "ja"), ORIG)
        body = "".join(f"<p>{o}</p><p>{t}</p>" for o, t in zip(ORIG, TRANS)) + "".join(f"<p>{t}</p>" for t in tail)
        self.assertEqual(len(tp.extract_post_rows(f"<html><body>{body}</body></html>")), len(ORIG))

    def test_a_tail_run_that_the_reference_explains_is_kept(self):
        tail = ["1.こどもあつかいしないでください", "2.やさしくしてごまかすのはもっとだめ"]
        reference = ORIG + [t[2:] for t in tail]
        found = qa.check(rows(ORIG + tail, TRANS + self.NUMBERED_TRANS), reference_original=reference)
        self.assertEqual(found_at(found, "note_leak"), [])

    def test_a_lone_numbered_row_is_not_a_track_row(self):
        self.assertEqual(found_at(qa.check(rows(ORIG + ["2.あるきょく"], TRANS + ["2.某曲"])), "note_leak"), [])

    def test_japanese_lyric_with_a_weak_marker_is_kept(self):
        line31 = "補充したいこころのすきまをあなたでうめてほしいのいまもずっとね"
        self.assertEqual(len(line31), 31)
        self.assertFalse(qa.is_note_line(line31))
        self.assertEqual(clean_lyric_lines([line31] + ORIG, "ja")[0], line31)
        kept, _, dropped = qa.gate([LyricPair(o, "譯") for o in [line31] + ORIG])
        self.assertEqual(dropped, [])

    def test_japanese_kanji_chain_with_ja_common_function_kanji_is_not_a_note(self):
        for line in ("過去現在未来永劫の果てに咲く花を見つけた", "完全無欠過去現在未来永劫輪廻転生の果て"):
            self.assertFalse(qa.is_note_line(line), line)
            self.assertEqual(clean_lyric_lines([line] + ORIG, "ja")[0], line)
            kept, issues, dropped = qa.gate([LyricPair(o, "譯") for o in [line] + ORIG])
            self.assertEqual(dropped, [])
            self.assertNotIn("note_in_original_keep_translation", [i.kind for i in issues])

    def test_chinese_prose_in_a_japanese_song_is_still_a_note(self):
        self.assertTrue(qa.is_note_line("但是同時也是日本古搖籃曲的句子是這樣的說明"))

    def test_chinese_translation_with_yi_ge_ren_keeps_its_stanza_aligned(self):
        trans = list(TRANS[:4])
        trans[1] = "一個人在這裡等著你回來的那一天一個人看著星空慢慢地想起你的樣子"
        self.assertEqual(len(trans[1]), 31)
        self.assertFalse(qa.is_translation_note(trans[1]))
        body = "".join(f"<p>{o}</p>" for o in ORIG[:4]) + "".join(f"<p>{t}</p>" for t in trans)
        found = tp.extract_post_rows(f"<html><body>{body}</body></html>")
        self.assertEqual([(r.original, r.translation) for r in found], list(zip(ORIG[:4], trans)))

    def test_note_markers_need_a_commentary_frame(self):
        self.assertFalse(qa.is_translation_note("每個人都在補充著各自的聽起來很遙遠的考量與意象，直到天亮為止才停下腳步來"))
        self.assertTrue(qa.is_translation_note("2. レガシー看到大家都翻譯成遺產，也許是動畫翻譯就說遺產，但我還是去查了這個詞有何意義。"))

    def test_two_bracket_pairs_alone_are_not_a_note(self):
        self.assertFalse(qa.is_translation_note("【甲】說了一些話，而【乙】只是靜靜地聽著，夜風吹過了整個冷清的街道盡頭"))

    def test_chinese_song_is_never_judged_by_note_rules(self):
        long_zh = "我們一起走過的那些年，" * 9
        self.assertFalse(qa.is_note_line(long_zh, "zh"))

    def test_repeated_tail_chorus_is_not_an_order_issue(self):
        reference = ORIG[:5]
        originals = ORIG[:5] + ORIG[2:4]
        found = qa.check(rows(originals, TRANS[:5] + TRANS[2:4]), reference_original=reference)
        self.assertEqual(found_at(found, "order"), [])
        unknown = qa.check(rows(ORIG[:5] + ["まったくべつのさいごのいちぎょう"], TRANS[:6]), reference_original=reference)
        self.assertEqual(found_at(unknown, "order"), [6])

    def test_standard_jis_kanji_in_a_kana_line_is_not_a_variant(self):
        originals = list(ORIG)
        originals[1] = "頰をつたうなみだ"
        self.assertEqual(found_at(qa.check(rows(originals)), "jp_variant"), [])

    def test_chinese_only_kanji_in_a_kana_line_is_a_variant(self):
        originals = list(ORIG)
        originals[1] = "ひとつに繫ごう"
        self.assertEqual(found_at(qa.check(rows(originals)), "jp_variant"), [2])
        self.assertTrue(all(k != v for k, v in qa._VARIANT_TO_JP.items()))

    def test_chinese_prose_in_the_original_column_by_shape(self):
        note = "但是ねんねんころり 同時也是日本古搖籃曲的句子"
        found = qa.check(rows(ORIG[:3] + [note] + ORIG[3:], TRANS[:3] + [note.replace(" ", "")] + TRANS[3:]))
        self.assertEqual(found_at(found, "note_leak"), [4])

    def test_japanese_dictionary_explanation_and_numbered_note(self):
        original = "えぐみは漢字で「蘞味」と表し、苦味に近い舌触りがざらざらする状態のことをいいます。"
        translation = "2. レガシー看到大家都翻譯成遺產，也許是動畫翻譯就說遺產，但我還是去查了這個詞有何意義。"
        found = qa.check(rows(ORIG[:3] + [original] + ORIG[3:], TRANS[:3] + [translation] + TRANS[3:]))
        self.assertEqual(found_at(found, "note_leak"), [4])

    def test_original_note_with_a_real_translation_is_flagged_not_dropped(self):
        note = "但是ねんねんころり 同時也是日本古搖籃曲的句子，這裡是補充說明"
        originals = ORIG[:3] + [note] + ORIG[3:]
        translations = TRANS[:3] + ["如回憶般，追尋著遠方的光"] + TRANS[3:]
        found = qa.check(rows(originals, translations))
        self.assertEqual(found_at(found, "note_in_original_keep_translation"), [4])
        self.assertEqual(found_at(found, "note_leak"), [])
        kept, _, dropped = qa.gate([LyricPair(o, t) for o, t in rows(originals, translations)])
        self.assertEqual(len(kept), len(originals))
        self.assertEqual(dropped, [])

    def test_original_note_quoting_the_lyric_translation_keeps_the_row(self):
        note = "【憑弔】，指對著遺跡追思故人。我認為它與【辿る】都是相當契合的，相較於直譯的【如回憶般，追尋著流逝的日子】更有意象與文學性的展現。"
        originals = ORIG[:3] + [note] + ORIG[3:]
        translations = TRANS[:3] + ["如回憶般，追尋著流逝的日子"] + TRANS[3:]
        found = qa.check(rows(originals, translations))
        self.assertEqual(found_at(found, "note_in_original_keep_translation"), [4])
        self.assertEqual(found_at(found, "note_leak"), [])

    def test_dictionary_entry_with_a_usage_example_is_a_note(self):
        self.assertTrue(qa.is_note_line("絶頂（ぜっちょう）上りつめた一番高い所。「山の―」。最高の程度。「人気の―」"))

    def test_english_hook_kept_elsewhere_is_not_untranslated(self):
        originals = ORIG[:2] + ["Hold me close tonight", "Never let me go away"] + ORIG[2:]
        translations = TRANS[:2] + ["Hold me close tonight", "Never let me go away"] + TRANS[2:]
        self.assertEqual(found_at(qa.check(rows(originals, translations)), "untranslated"), [])

    def test_stale_qa_state_is_cleared_when_the_file_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "txt").mkdir()
            (out / "report").mkdir()
            (out / "txt" / "s.lyrics.txt").write_text("\n".join(f"{o}  {t}" for o, t in rows()), encoding="utf-8")
            sources = out / "report" / "s.sources.json"
            sources.write_text(json.dumps({"status": "needs_review", "status_before_qa": "page_human",
                                           "qa_issues": [{"kind": "bleed", "line": 2}],
                                           "qa_dropped_note_lines": [{"kind": "note_leak", "line": 9}]}), encoding="utf-8")
            post = {"aligned": list(TRANS), "url": "u", "author": "a", "title": "t", "sim": 1.0, "covered": len(ORIG)}
            with patch.object(tp, "find_translation_post", return_value=post):
                result = tp.apply_post_translation(out, "s")
            payload = json.loads(sources.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "applied")
        self.assertEqual(payload["status"], "page_human")
        for key in ("qa_issues", "qa_dropped_note_lines", "status_before_qa"):
            self.assertNotIn(key, payload)


class OoggLineTests(unittest.TestCase):
    def test_line_translation_prompt_carries_no_neighbours(self):
        sent = []

        def post_json(url, request, timeout):
            sent.append(request)
            return {"choices": [{"message": {"content": json.dumps({"text": "一行"}, ensure_ascii=False)}}]}

        self.assertEqual(translate_oogg_line("あおいそら", title="t", endpoint="http://x", model="m", post_json=post_json), "一行")
        user = json.loads(sent[0]["messages"][1]["content"])
        self.assertEqual(list(user), ["translate_this_line_only"])

    def test_translate_oogg_leaves_bleed_repair_to_the_gate(self):
        batch = [LyricPair(o, t) for o, t in rows()]
        batch[2] = LyricPair(ORIG[2], TRANS[1])    # the model translated the previous line again
        with patch("lyricsee_core.translate_with_local_llm", return_value=batch), \
                patch("translation_oogg.translate_oogg_line", return_value=TRANS[2]) as single:
            result = translate_oogg(ORIG, title="t", model="m", endpoint="http://x", review=False)
        single.assert_not_called()
        self.assertEqual(result[2], TRANS[1])


if __name__ == "__main__":
    unittest.main()
