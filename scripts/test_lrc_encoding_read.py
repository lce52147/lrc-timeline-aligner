"""Review page reads LRC files in legacy encodings and millisecond tags; undecodable files say so instead of showing no rows."""
from __future__ import annotations

import json
import os
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import review_server
import test_lrc14_review_server as _fixture_module

GBK_TEXT = "[ti:歌词 (《穿越边缘》)]\r\n[00:01.543]这是第一句\r\n[00:03.210]爱你一万年\r\n[00:05.000]third\r\n"


SIMPLIFIED = "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方又如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工话儿者向情部正名定女问力机给等几很业最间新什"
TRADITIONAL = "的一是不了人我在有他這中大來上國個到說們為子和你地出道也時年得就那要下以生會自著去之過家學對可她裡後小麼心多天而能好都然沒日於起還發成事只作當想看文無開手十用主行方又前所本見經頭面公同三已老從動兩長知民樣現分將外但身些與高意進把法此實回二理美點月明其種聲全工話兒者向情部正名定女問力機給等幾很業最間新什"


def _detect_rate(chars, codec, length, trials=300, seed=20261112):
    rnd = random.Random(seed + length)
    hits = 0
    for _ in range(trials):
        text = "[00:01.00]" + "".join(rnd.choice(chars) for _ in range(length)) + "\n"
        hits += review_server._decode_lrc(text.encode(codec)) == (text, "gb18030" if codec == "gb18030" else "cp950")
    return hits / trials


class DecodeTests(unittest.TestCase):
    def test_utf8_and_bom_pass_through(self) -> None:
        raw = "[00:01.00]你好\n".encode("utf-8")
        self.assertEqual(review_server._canonical_lrc(raw), raw)
        bommed = b"\xef\xbb\xbf" + raw
        self.assertEqual(review_server._canonical_lrc(bommed), bommed)

    def test_utf16_with_bom(self) -> None:
        raw = "[00:01.00]你好\r\n".encode("utf-16")
        self.assertEqual(review_server._canonical_lrc(raw), "[00:01.00]你好\r\n".encode("utf-8"))

    def test_gb18030(self) -> None:
        text, encoding = review_server._decode_lrc(GBK_TEXT.encode("gb18030"))
        self.assertEqual((text, encoding), (GBK_TEXT, "gb18030"))

    def test_big5(self) -> None:
        text = "[00:01.00]我愛你，這是歌詞\n"
        self.assertEqual(review_server._decode_lrc(text.encode("cp950")), (text, "cp950"))

    def test_gbk_big5_detection_rates(self) -> None:
        floors = {2: 0.58, 4: 0.85, 8: 0.97, 16: 1.0, 64: 1.0}
        for length, floor in floors.items():
            gbk = _detect_rate(SIMPLIFIED, "gb18030", length)
            big5 = _detect_rate(TRADITIONAL, "cp950", length)
            print(f"detect length={length} gbk={gbk:.3f} big5={big5:.3f}")
            self.assertGreaterEqual(gbk, floor, f"GBK at {length} chars")
            self.assertEqual(big5, 1.0, f"Big5 at {length} chars")

    def test_two_char_ambiguous_sample_is_flagged_uncertain(self) -> None:
        raw = "[00:01.00]爱你\n".encode("gb18030")  # the same bytes are valid Big5
        self.assertTrue(review_server.lrc_encoding_uncertain(raw))
        _text, encoding, uncertain = review_server._decode_lrc_checked(raw)
        self.assertEqual(encoding, "cp950")  # still decodes, never blocks
        self.assertTrue(uncertain)

    def test_clear_cases_are_not_flagged(self) -> None:
        self.assertFalse(review_server.lrc_encoding_uncertain(GBK_TEXT.encode("gb18030")))
        self.assertFalse(review_server.lrc_encoding_uncertain("[00:01.00]我愛你，這是歌詞，請你相信我\n".encode("cp950")))
        self.assertFalse(review_server.lrc_encoding_uncertain("[00:01.00]你好\n".encode("utf-8")))

    def test_undecodable_raises(self) -> None:
        with self.assertRaises(review_server.LrcDecodeError):
            review_server._canonical_lrc(b"[00:01.00]\x81\x20\xff\xff\xfe\n")
        with self.assertRaises(review_server.LrcDecodeError):
            review_server._canonical_lrc(b"\xff\xfe\x00")  # UTF-16 BOM, odd byte count

    def test_shift_jis_kana_is_read_as_cp932(self) -> None:
        text = "[00:01.00]これはテストです\n"
        self.assertEqual(review_server._decode_lrc_checked(text.encode("cp932")), (text, "cp932", False))
        mixed = "[00:01.00]ひらがなとカタカナ、漢字\n"
        self.assertEqual(review_server._decode_lrc_checked(mixed.encode("cp932")), (mixed, "cp932", False))

    def test_long_vowel_mark_counts_as_kana(self) -> None:
        text = "[00:01.00]ラーメン\n"
        self.assertEqual(review_server._decode_lrc_checked(text.encode("cp932")), (text, "cp932", False))
        self.assertEqual(review_server._fullwidth_kana("ー"), 1)

    def test_shift_jis_with_almost_no_kana_is_flagged_uncertain_but_not_re_decoded(self) -> None:
        raw = "[00:01.00]東京の夜\n".encode("cp932")
        text, encoding, uncertain = review_server._decode_lrc_checked(raw)
        self.assertEqual((encoding in ("gb18030", "cp950"), uncertain), (True, True))
        self.assertEqual(text, raw.decode(encoding))

    def test_existing_gbk_and_big5_samples_stay_certain(self) -> None:
        for text, codec in ((GBK_TEXT, "gb18030"), ("[00:01.00]我愛你，這是歌詞，請你相信我\n", "cp950"), (TRADITIONAL[:60], "cp950"), (SIMPLIFIED[:60], "gb18030")):
            self.assertFalse(review_server.lrc_encoding_uncertain(("[00:01.00]" + text if not text.startswith("[") else text).encode(codec)), codec)

    def test_one_digit_seconds_at_line_start_are_not_a_tag(self) -> None:
        self.assertEqual(review_server._canonical_lrc(b"[1:2]verse\n[1:02]x\n"), b"[1:2]verse\n[01:02.00]x\n")

    def test_shift_jis_does_not_steal_chinese(self) -> None:
        self.assertEqual(review_server._decode_lrc(GBK_TEXT.encode("gb18030"))[1], "gb18030")
        self.assertEqual(review_server._decode_lrc("[00:01.00]我愛你，這是歌詞，請你相信我\n".encode("cp950"))[1], "cp950")

    def test_millisecond_tags_become_centiseconds(self) -> None:
        raw = b"[00:01.543]a\n*[10:00.004]b*\n[00:59.996]c\n[00:01.50]kept\n"
        self.assertEqual(review_server._canonical_lrc(raw), b"[00:01.54]a\n*[10:00.00]b*\n[01:00.00]c\n[00:01.50]kept\n")


    def test_odd_tag_shapes_become_centiseconds(self) -> None:
        raw = b"[0:05.12]a\n[00:07]b\n[00:09.5]c\n*[1:02.5]d*\n[00:61]kept\n"
        self.assertEqual(review_server._canonical_lrc(raw), b"[00:05.12]a\n[00:07.00]b\n[00:09.50]c\n*[01:02.50]d*\n[00:61]kept\n")

    def test_canonical_bytes_do_not_change(self) -> None:
        raw = "[ti:x]\r\n[00:01.00]一\r\n[100:00.05]二\r\n*[02:03.45]三*\r\n[00:05.00]a [00:06.00]b\r\n".encode("utf-8")
        self.assertEqual(review_server._canonical_lrc(raw), raw)


class ReviewReadsLegacyLrcTests(unittest.TestCase):
    _start = _fixture_module.ReviewServerTests._start
    _request = _fixture_module.ReviewServerTests._request
    _json_request = _fixture_module.ReviewServerTests._json_request
    _detail = _fixture_module.ReviewServerTests._detail

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {"LRC_EXPORT_ARCHIVE_DIR": "off"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._start(False)
        for sidecar in self.root.glob("song.auto.lrc"):
            sidecar.unlink()

    def tearDown(self) -> None:
        _fixture_module.ReviewServerTests.tearDown(self)

    def test_gbk_millisecond_lrc_shows_rows_and_saves_utf8_with_backup(self) -> None:
        original = GBK_TEXT.encode("gb18030")
        self.output.write_bytes(original)
        self.store.refresh_if_changed()
        detail = self._detail()
        self.assertFalse(detail["read_only"], detail["read_only_reason"])
        self.assertEqual([row["lyric"] for row in detail["rows"]], ["这是第一句", "爱你一万年", "third"])
        self.assertIs(detail["encoding_uncertain"], False)
        body = {"base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 160, "source": "ear", "nudge_cs": 6}]}
        status, _headers, data = self._json_request("POST", "/api/song/0/save", body)
        self.assertEqual(status, 200, data)
        saved = self.output.read_bytes()
        text = saved.decode("utf-8")  # strict: the file on disk is UTF-8 now
        self.assertIn("[00:01.60]这是第一句", text)
        self.assertIn("[00:03.21]爱你一万年", text)
        backups = list(self.root.glob("song.lrc.orig-*.bak"))
        self.assertEqual([path.read_bytes() for path in backups], [original])
        again = self._detail()
        self.assertEqual([row["lyric"] for row in again["rows"]], ["这是第一句", "爱你一万年", "third"])
        status, _headers, _data = self._json_request(
            "POST", "/api/song/0/save",
            {"base_sha256": again["base_sha256"], "edits": [{"entry": 3, "cs": 700, "source": "ear", "nudge_cs": 0}]})
        self.assertEqual(status, 200)
        self.assertEqual(len(list(self.root.glob("song.lrc.orig-*.bak"))), 1)  # a UTF-8 file is not backed up again

    def test_shift_jis_lrc_shows_rows_and_saves_utf8_with_backup(self) -> None:
        text = "[ti:テストのうた]\r\n[00:01.543]これはテストです\r\n[00:03.210]もういちどテスト\r\n[00:05.000]third\r\n"
        original = text.encode("cp932")
        self.output.write_bytes(original)
        self.store.refresh_if_changed()
        detail = self._detail()
        self.assertFalse(detail["read_only"], detail["read_only_reason"])
        self.assertEqual([row["lyric"] for row in detail["rows"]], ["これはテストです", "もういちどテスト", "third"])
        self.assertIs(detail["encoding_uncertain"], False)
        body = {"base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 160, "source": "ear", "nudge_cs": 6}]}
        status, _headers, data = self._json_request("POST", "/api/song/0/save", body)
        self.assertEqual(status, 200, data)
        saved = self.output.read_bytes().decode("utf-8")
        self.assertIn("[ti:テストのうた]", saved)
        self.assertIn("[00:01.60]これはテストです", saved)
        self.assertEqual([path.read_bytes() for path in self.root.glob("song.lrc.orig-*.bak")], [original])

    def test_odd_tag_lrc_shows_every_row_and_saves_canonical_with_backup(self) -> None:
        original = b"[0:01.50]first\n[00:03]second\n[00:05.5]third\n[00:07.00]fourth\n"
        self.output.write_bytes(original)
        self.store.refresh_if_changed()
        detail = self._detail()
        self.assertFalse(detail["read_only"], detail["read_only_reason"])
        self.assertEqual([row["lyric"] for row in detail["rows"]], ["first", "second", "third", "fourth"])
        body = {"base_sha256": detail["base_sha256"], "edits": [{"entry": 2, "cs": 320, "source": "ear", "nudge_cs": 0}]}
        status, _headers, data = self._json_request("POST", "/api/song/0/save", body)
        self.assertEqual(status, 200, data)
        self.assertEqual(self.output.read_bytes(), b"[00:01.50]first\n[00:03.20]second\n[00:05.50]third\n[00:07.00]fourth\n")
        self.assertEqual([path.read_bytes() for path in self.root.glob("song.lrc.orig-*.bak")], [original])

    def test_two_char_ambiguous_lrc_sets_the_payload_flag(self) -> None:
        self.output.write_bytes("[00:01.00]爱你\n[00:02.00]x\n".encode("gb18030"))
        self.store.refresh_if_changed()
        detail = self._detail()
        self.assertFalse(detail["read_only"], detail["read_only_reason"])
        self.assertIs(detail["encoding_uncertain"], True)

    def test_undecodable_lrc_is_read_only_with_a_reason(self) -> None:
        self.output.write_bytes(b"[00:01.00]\x81\x20\xff\xff\xfe\n[00:02.00]x\n")
        self.store.refresh_if_changed()
        detail = self._detail()
        self.assertTrue(detail["read_only"])
        self.assertEqual(detail["read_only_reason"], "lrc-undecodable")
        self.assertEqual(detail["rows"], [])
        self.assertEqual(self._json_request("POST", "/api/song/0/save", {"base_sha256": detail["base_sha256"], "edits": []})[0], 403)


class GbkJapaneseTieTests(unittest.TestCase):
    """Japanese saved as GBK on a short file: both Big5 and GB18030 read it without a stray character, and it must not come out as certain Big5."""
    JAPANESE_TIE = ("[00:01.00]夜が明けるまで一緒にいよう\n", "[00:01.00]私の名前を呼んでくれた\n[00:02.00]遠い空の向こうに行こう\n",
                    "[00:01.00]君が笑うと世界が変わる\n")
    TOO_LITTLE = ("[00:01.00]ありがとう\n", "[00:01.00]さよなら\n")

    def _read(self, text, codec):
        return review_server._decode_lrc_checked(text.encode(codec))

    def _tie(self, raw):
        try:
            return review_server._legacy_score(raw.decode("cp950")) == review_server._legacy_score(raw.decode("gb18030"))
        except UnicodeDecodeError:
            return False

    def test_the_samples_include_ties_the_old_rule_got_wrong(self) -> None:
        # (a sample Big5 cannot read at all is read as GB18030 by every version)
        self.assertGreaterEqual(sum(self._tie(text.encode("gbk")) for text in self.JAPANESE_TIE), 2)
        for text in self.TOO_LITTLE:
            self.assertTrue(self._tie(text.encode("gbk")), text)

    def test_clearly_japanese_short_files_read_as_gb18030(self) -> None:
        for text in self.JAPANESE_TIE:
            self.assertEqual(self._read(text, "gbk"), (text, "gb18030", False))

    def test_a_trace_of_kana_is_a_refusal_not_a_certain_big5(self) -> None:
        for text in self.TOO_LITTLE:
            self.assertTrue(self._read(text, "gbk")[2], text)

    def test_one_or_two_kana_on_a_short_gbk_file_is_a_refusal_never_a_certain_big5(self) -> None:
        # claude-10's four: 1-2 lines whose GB18030 reading has one or two kana (the Big5 reading is a few rare characters)
        for lines in (["もう一度"], ["君が好き"], ["花火", "空へ"], ["花火", "君が好き"]):
            text = "".join("[00:%02d.00]%s\n" % (number, line) for number, line in enumerate(lines))
            got = self._read(text, "gbk")
            self.assertTrue(got[1] == "gb18030" or got[2], (lines, got[1:]))
            self.assertNotEqual((got[1], got[2]), ("cp950", False), lines)

    def test_short_big5_files_are_still_certain_and_right(self) -> None:
        for lines in (["我愛你"], ["回家"], ["我愛你", "回家"], ["一個人", "你的心"]):
            text = "".join("[00:%02d.00]%s\n" % (number, line) for number, line in enumerate(lines))
            self.assertEqual(self._read(text, "cp950"), (text, "cp950", False), lines)

    def test_kana_in_a_header_tag_is_not_evidence_of_japanese(self) -> None:
        # a Big5 title whose bytes read as one kana in GB18030, on a file with no timed line
        raw = "[ti:獵れ锭垫 BLUE BUD]\n[re:LRC tools ctc-forced-align]\n[length:03:59.00]\n".encode("gbk")
        self.assertEqual(review_server._decode_lrc_checked(raw)[1:], ("cp950", False))

    def test_real_big5_chinese_is_unchanged(self) -> None:
        for text in ("[00:01.00]我們一起走過的日子\n[00:02.00]你是我心中最美的夢\n[00:03.00]了解你的心情\n", "[00:01.00]我愛你\n[00:02.00]一個人\n"):
            self.assertEqual(self._read(text, "cp950")[:2], (text, "cp950"))

    def test_a_repeated_short_chinese_phrase_is_not_read_as_japanese(self) -> None:
        # as Big5 these bytes read in GB18030 as six kana of two kinds
        got = self._read("[00:37.66]反正 反正 反正\n", "cp950")
        self.assertNotEqual(got[1], "gb18030")
        self.assertTrue(got[2])

    def test_a_refused_file_is_refused_by_the_lyric_reader(self) -> None:
        import tempfile
        from text_encoding import LyricEncodingError, read_lyric_text
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "short.lrc"
            path.write_bytes(self.TOO_LITTLE[0].encode("gbk"))
            with self.assertRaises(LyricEncodingError):
                read_lyric_text(path)
            path.write_bytes(self.JAPANESE_TIE[1].encode("gbk"))
            self.assertEqual(read_lyric_text(path), (self.JAPANESE_TIE[1], "gb18030"))

if __name__ == "__main__":
    unittest.main()
