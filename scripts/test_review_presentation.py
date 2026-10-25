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
            self.assertEqual(first["risk"], "normal")
            self.assertIsNone(second["risk"])


    def test_review_risk_tiers(self):
        cands = [{"time": 10.0 + i * 0.05, "source": "s%d" % i, "score": 0.8} for i in range(5)]
        base = {"entry": 3, "candidates": cands}
        self.assertEqual(review_presentation.review_risk(base, ["insufficient-minimum-proof"], 1000), "low")
        # A strong competitor more than 0.5 s away blocks "low".
        rival = dict(base, candidates=cands + [{"time": 12.0, "source": "x", "score": 0.9}])
        self.assertEqual(review_presentation.review_risk(rival, ["insufficient-minimum-proof"], 1000), "normal")
        # Rejected candidates do not count as agreement; hypotheses do.
        few = {"entry": 3, "candidates": cands[:3], "rejected_candidates": cands[3:]}
        self.assertEqual(review_presentation.review_risk(few, [], 1000), "normal")
        hyp = dict(few, alignment_hypotheses=[{"source": "h1", "raw_candidate_time": 10.1}, {"source": "h2", "raw_candidate_time": 9.9}])
        self.assertEqual(review_presentation.review_risk(hyp, [], 1000), "low")
        self.assertEqual(review_presentation.review_risk(few, ["model-disagreement"], 1000), "high")
        many = ["c%d" % i for i in range(7)]
        self.assertEqual(review_presentation.review_risk(few, many, 1000), "high")
        self.assertEqual(review_presentation.review_risk(few, many, 1000, [3]), "normal")

    def test_song_speech_rate_uses_trusted_lines_and_skips_breaks(self):
        trusted = {"verification_status": "trusted", "romaji": "abcdefghij"}  # 10 chars
        rows = [dict(trusted) for _ in range(6)] + [{"verification_status": "review-required", "romaji": "abcdefghij"}]
        # six trusted lines 1.0 s apart (0.1 s/char), the last gap is a 10 s instrumental break
        times = [0, 100, 200, 300, 400, 500, 1500]
        self.assertAlmostEqual(review_presentation.song_speech_rate(rows, times), 0.1)
        # too few trusted lines: every line counts
        rows[0]["verification_status"] = "review-required"; rows[1]["verification_status"] = "review-required"
        times = [0, 200, 400, 600, 800, 1000, 1200]
        self.assertAlmostEqual(review_presentation.song_speech_rate(rows, times), 0.2)
        # nothing usable: default
        self.assertEqual(review_presentation.song_speech_rate([{"romaji": "abc"}], [0]), review_presentation.FIT_DEFAULT_RATE_S_PER_CHAR)
        self.assertEqual(review_presentation.song_speech_rate(None, None), review_presentation.FIT_DEFAULT_RATE_S_PER_CHAR)

    def test_line_fit_measures_slack_and_span(self):
        rate = 0.1
        spans = lambda start, end: [{"start": start, "end": end, "score": 0.5}]
        prev = {"romaji": "abcdefghij"}  # expected 1.0 s
        line = {"romaji": "abcdefghijklmnopqrst", "ctc_token_spans": spans(10.0, 11.8)}  # expected 2.0 s, span 1.8 s
        fit = review_presentation.line_fit(line, 1000, prev, 850, 1300, rate)
        self.assertEqual(fit, {"slack_prev_s": 0.5, "slack_next_s": 1.0, "span_ratio": 0.9, "fits": True})
        # previous line could not have finished (0.9 s short)
        self.assertFalse(review_presentation.line_fit(line, 1000, prev, 1090, 1300, rate)["fits"])
        # does not fit before the next line
        self.assertFalse(review_presentation.line_fit(line, 1000, prev, 850, 1140, rate)["fits"])
        # last line: no next, still fits
        self.assertTrue(review_presentation.line_fit(line, 1000, prev, 850, None, rate)["fits"])
        # both neighbours far away: a shifted block would still fit, so this is not a fit
        loose = review_presentation.line_fit(line, 1000, prev, 650, 1500, rate)
        self.assertEqual((loose["slack_prev_s"], loose["slack_next_s"], loose["fits"]), (2.5, 3.0, False))
        self.assertTrue(review_presentation.line_fit(line, 1000, prev, 650, 1400, rate)["fits"], "one close neighbour pins the line")
        self.assertFalse(review_presentation.line_fit(line, 1000, prev, 650, None, rate)["fits"], "a last line far from its predecessor is not pinned")
        # absurd CTC span
        stretched = dict(line, ctc_token_spans=spans(10.0, 15.0))
        self.assertFalse(review_presentation.line_fit(stretched, 1000, prev, 850, 1300, rate)["fits"])
        # no span at all: cannot fit
        self.assertFalse(review_presentation.line_fit({"romaji": "abcdefghij"}, 1000, prev, 850, 1300, rate)["fits"])
        # first line or missing data: None
        self.assertIsNone(review_presentation.line_fit(line, 1000, None, None, 1300, rate))
        self.assertIsNone(review_presentation.line_fit({"romaji": ""}, 1000, prev, 850, 1300, rate))
        self.assertIsNone(review_presentation.line_fit(line, None, prev, 850, 1300, rate))

    def test_fit_demotes_normal_lines_only(self):
        few = {"entry": 3, "candidates": [{"time": 10.0, "source": "a", "score": 0.8}]}
        fits = {"slack_prev_s": 0.5, "slack_next_s": 0.5, "span_ratio": 1.0, "fits": True}
        unfit = dict(fits, fits=False)
        self.assertEqual(review_presentation.classify_risk(few, ["insufficient-minimum-proof"], 1000), ("normal", "default"))
        self.assertEqual(review_presentation.classify_risk(few, ["insufficient-minimum-proof"], 1000, (), fits), ("low", "fit"))
        self.assertEqual(review_presentation.classify_risk(few, ["insufficient-minimum-proof"], 1000, (), unfit), ("normal", "default"))
        self.assertEqual(review_presentation.review_risk(few, ["insufficient-minimum-proof"], 1000, (), fits), "low")
        # high stays high whatever the fit says
        self.assertEqual(review_presentation.classify_risk(few, ["model-disagreement"], 1000, (), fits), ("high", "model-disagreement"))
        many = ["c%d" % i for i in range(7)]
        self.assertEqual(review_presentation.classify_risk(few, many, 1000, (), fits), ("high", "many-codes"))
        self.assertEqual(review_presentation.classify_risk(few, many, 1000, [3], fits), ("low", "fit"))
        # agreement keeps its own basis
        cands = [{"time": 10.0 + i * 0.05, "source": "s%d" % i, "score": 0.8} for i in range(5)]
        self.assertEqual(review_presentation.classify_risk({"entry": 3, "candidates": cands}, [], 1000, (), unfit), ("low", "agreement"))

class SongRiskTests(unittest.TestCase):
    def rows(self, n, review):
        return [{"entry": i + 1, "review_required": i < review} for i in range(n)]

    def test_half_of_the_lines_in_review_is_the_choir_class(self):
        risk = review_presentation.song_risk({"assignments": self.rows(44, 37)})
        self.assertEqual(risk["class"], "choir")
        self.assertEqual(risk["reasons"], ["review-share"])
        self.assertEqual((risk["rows"], risk["review_rows"]), (44, 37))

    def test_weak_anchors_from_the_segment_stage_is_the_choir_class(self):
        report = {"assignments": self.rows(40, 10), "segment_realign": {"status": "weak-anchors", "structure": {"anchors": 7}}}
        risk = review_presentation.song_risk(report)
        self.assertEqual((risk["class"], risk["reasons"], risk["anchors"]), ("choir", ["weak-anchors"], 7))

    def test_ordinary_song_has_no_class(self):
        risk = review_presentation.song_risk({"assignments": self.rows(43, 19), "segment_realign": {"status": "no-runs"}})
        self.assertIsNone(risk["class"])
        self.assertEqual(risk["review_share"], 0.442)
        # too few rows for the share rule
        self.assertIsNone(review_presentation.song_risk({"assignments": self.rows(6, 5)})["class"])
        self.assertIsNone(review_presentation.song_risk({})["class"])


if __name__ == "__main__":
    unittest.main()
