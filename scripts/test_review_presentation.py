from __future__ import annotations

import json
import pathlib
import sys
import unittest

SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import review_presentation  # noqa: E402
import review_server  # noqa: E402


def cand(cs, score, sources=("ctc-first-token-posterior",), agree=0, **extra):
    return {"cs": cs, "score": score, "sources": list(sources), "agree_count": agree, "rejected": False, "diagnostic_only": False, **extra}


class PresentOptionsTests(unittest.TestCase):
    def test_nine_close_candidates_become_two_alternatives_plus_keep(self):
        # 01 - フロム entry 10 as shown on the 1.5 page: 8 posterior candidates between -2.48s and -0.98s.
        cs_list = [6856, 6888, 6912, 6930, 6970, 6982, 7006, 7104]
        candidates = [cand(cs, 0.9 - i * 0.01, agree=(1 if cs in (6888, 6970, 6982) else 0)) for i, cs in enumerate(cs_list)]
        options = review_presentation.present_options(candidates, [], 7104)
        self.assertEqual([o["kind"] for o in options], ["earlier", "earlier", "keep"])
        self.assertEqual([o["label"] for o in options], ["A", "B", "維持"])
        self.assertTrue(all(o["cs"] < 7104 for o in options if o["kind"] != "keep"))
        self.assertGreaterEqual(options[1]["cs"] - options[0]["cs"], review_presentation.OPTION_CLUSTER_CS)
        self.assertEqual(options[-1], {"cs": 7104, "kind": "keep", "label": "維持", "delta_s": 0.0, "support": 0, "score": 0.0, "sources": ["current"]})

    def test_candidates_within_fold_distance_merge_into_keep(self):
        options = review_presentation.present_options([cand(1005, 0.9), cand(1012, 0.8), cand(990, 0.7)], [], 1000)
        self.assertEqual(options, [{"cs": 1000, "kind": "keep", "label": "維持", "delta_s": 0.0, "support": 0, "score": 0.0, "sources": ["current"]}])

    def test_rejected_and_diagnostic_candidates_are_ignored_and_support_sums(self):
        members = [{"cs": 700, "source": "a"}, {"cs": 704, "source": "b"}, {"cs": 704, "source": "c"}, {"cs": 690, "source": "raw_asr", "diagnostic": True}]
        candidates = [cand(500, 0.9, rejected=True), cand(600, 0.5, diagnostic_only=True), cand(700, 0.6, members=members), cand(715, 0.4, sources=("ctc-local-window",))]
        options = review_presentation.present_options(candidates, [cand(1400, 0.7)], 1000)
        alternatives = [o for o in options if o["kind"] != "keep"]
        self.assertEqual([o["cs"] for o in alternatives], [700, 1400])
        self.assertEqual(alternatives[0]["support"], 3)
        self.assertEqual(alternatives[0]["sources"], ["ctc-first-token-posterior", "ctc-local-window"])
        self.assertEqual(alternatives[0]["delta_s"], -3.0)
        self.assertEqual(alternatives[1]["kind"], "later")

    def test_no_current_timestamp_gives_alternatives_only(self):
        options = review_presentation.present_options([cand(700, 0.6)], [], None)
        self.assertEqual([o["kind"] for o in options], ["later"])

    def test_malformed_input_is_safe(self):
        self.assertEqual(review_presentation.present_options(None, "x", "bad"), [])
        self.assertEqual(review_presentation.present_options([{"cs": None}, "junk"], [], 100)[0]["kind"], "keep")


class ReasonSentenceTests(unittest.TestCase):
    def test_priority_order_first_hit_wins(self):
        codes = ["insufficient-minimum-proof", "previous-tail-ownership-unknown", "unresolved-material-alternative"]
        self.assertEqual(review_presentation.reason_sentence(codes), "這句可能提早開唱（跟上一句尾巴黏在一起），請聽開頭。")
        self.assertEqual(review_presentation.reason_sentence(["F1-vocal-absent", "previous-tail-overlap"]), "這段可能沒有人聲，請確認是否該刪或移到下一句。")
        self.assertEqual(review_presentation.reason_sentence(["silent-risk:selected-confidence-le-0.8-and-next-gap-ge-19"]), "這句之後有一段長空白，請確認結束位置沒有抓錯。")
        self.assertEqual(review_presentation.reason_sentence([{"code": "candidate_disagreement"}]), "幾個偵測結果不一致，請聽並挑一個。")

    def test_every_b1_code_maps_to_a_sentence(self):
        fixture = json.loads((SCRIPTS / "test_fixtures" / "reason_codes_b1_39.json").read_text(encoding="utf-8"))
        for code in fixture["codes"]:
            self.assertNotEqual(review_presentation.reason_sentence([code]), review_presentation.DEFAULT_SENTENCE, code)

    def test_unknown_and_empty_codes(self):
        self.assertEqual(review_presentation.reason_sentence(["mystery"]), "需要人工確認。")
        self.assertEqual(review_presentation.reason_sentence([]), "需要人工確認。")
        self.assertEqual(review_presentation.reason_sentence([], options_count=0), "候選一致，只需確認。")
        self.assertEqual(review_presentation.reason_sentence(["insufficient-minimum-proof"], options_count=0), "候選一致，只需確認。")
        self.assertEqual(review_presentation.reason_sentence(["insufficient-minimum-proof"], options_count=2), "系統證據不足，請聽一下確認。")


class PayloadWiringTests(unittest.TestCase):
    def test_song_payload_rows_carry_options_and_sentence(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            audio = root / "song.flac"; audio.write_bytes(b"fLaC")
            lrc = root / "song.lrc"; lrc.write_bytes(b"[00:01.00]one\n[00:05.00]two\n")
            report = root / "song.align-report.json"
            data = {
                "audio_path": str(audio), "output_path": str(lrc), "display_lines": 2, "timing_entries": 2,
                "assignments": [
                    {"entry": 1, "lrc_tag": "[00:01.00]", "final_trust_reason_codes": ["previous-tail-ownership-unknown"], "verification_status": "review-required",
                     "candidates": [{"time": 0.2, "source": "ctc-first-token-posterior", "score": 0.9}, {"time": 0.3, "source": "ctc-local-window", "score": 0.8}]},
                    {"entry": 2, "lrc_tag": "[00:05.00]", "final_trust_reason_codes": [], "verification_status": "verified", "candidates": []},
                ],
            }
            report.write_text(json.dumps(data), encoding="utf-8")
            payload = review_server._song_payload(report, data)
            first, second = payload["rows"]
            self.assertEqual(first["reason_text"], "這句可能提早開唱（跟上一句尾巴黏在一起），請聽開頭。")
            self.assertEqual([o["kind"] for o in first["options"]], ["earlier", "keep"])
            self.assertEqual(first["options"][0]["cs"], 20)
            self.assertEqual(first["options"][0]["support"], 2)
            self.assertEqual([o["kind"] for o in second["options"]], ["keep"])
            self.assertIn(second["reason_text"], ("候選一致，只需確認。", "需要人工確認。"))
            # legacy fields stay for the details fold
            self.assertIn("candidates", first); self.assertIn("reason_codes", first)


if __name__ == "__main__":
    unittest.main()
