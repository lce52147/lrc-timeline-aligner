import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

import translation_gemini as tg

FAKE_KEY = "AIza" + "X" * 35      # synthetic, not a real key


def reply(rows):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps({"rows": rows}, ensure_ascii=False)}]}}]}


def echo_rows(request_body, texts=None):
    user = json.loads(request_body["contents"][0]["parts"][0]["text"])
    ids = user["translate_ids"]
    return reply([{"id": i, "text": (texts or {}).get(i, f"譯{i}")} for i in ids])


def http_error(code):
    return HTTPError("https://example.invalid/x?key=" + FAKE_KEY, code, "err", {}, io.BytesIO(b""))


class Fake:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []
        self.headers = []

    def __call__(self, url, body, headers, timeout):
        self.requests.append(body)
        self.headers.append(headers)
        return self.handler(len(self.requests), body, headers)


def client(handler, keys=(("google:a", FAKE_KEY), ("google:b", "AIza" + "Y" * 35))):
    fake = Fake(handler)
    return tg.GeminiClient(keys=list(keys), post=fake, sleep=lambda s: None), fake


class TranslateGeminiTests(unittest.TestCase):
    def test_one_request_per_short_song_with_ordered_ids(self):
        c, fake = client(lambda n, body, h: echo_rows(body))
        out = tg.translate_gemini(["a", "b", "c"], title="T", artist="A", client=c)
        self.assertEqual(out, ["譯L001", "譯L002", "譯L003"])
        self.assertEqual(len(fake.requests), 1)
        user = json.loads(fake.requests[0]["contents"][0]["parts"][0]["text"])
        self.assertEqual(user["translate_ids"], ["L001", "L002", "L003"])
        self.assertIn("繁體中文", fake.requests[0]["systemInstruction"]["parts"][0]["text"])

    def test_chunks_of_40_carry_whole_song_context(self):
        lines = [f"line{n}" for n in range(85)]
        c, fake = client(lambda n, body, h: echo_rows(body))
        out = tg.translate_gemini(lines, client=c)
        self.assertEqual(len(out), 85)
        self.assertEqual(len(fake.requests), 3)
        user = json.loads(fake.requests[1]["contents"][0]["parts"][0]["text"])
        self.assertEqual(len(user["whole_song_for_meaning_only"]), 85)
        self.assertEqual(len(user["translate_ids"]), 40)

    def test_id_mismatch_retries_with_smaller_chunk(self):
        def handler(n, body, h):
            user = json.loads(body["contents"][0]["parts"][0]["text"])
            ids = user["translate_ids"]
            if len(ids) > 2:        # a big request comes back shifted
                return reply([{"id": i, "text": "x"} for i in ids[1:]] + [{"id": "L999", "text": "y"}])
            return echo_rows(body)
        c, fake = client(handler)
        out = tg.translate_gemini(["a", "b", "c", "d"], client=c)
        self.assertEqual(out, ["譯L001", "譯L002", "譯L003", "譯L004"])
        self.assertGreater(len(fake.requests), 1)

    def test_empty_or_multiline_row_is_rejected(self):
        for bad in ("", "  ", "a\nb"):
            with self.assertRaises(ValueError):
                tg.validate_rows(["L001"], {"rows": [{"id": "L001", "text": bad}]})
        with self.assertRaises(ValueError):
            tg.validate_rows(["L001", "L002"], {"rows": [{"id": "L002", "text": "x"}, {"id": "L001", "text": "y"}]})

    def test_single_row_that_never_validates_raises_masked_error(self):
        c, _ = client(lambda n, body, h: reply([{"id": "L001", "text": ""}]))
        with self.assertRaises(tg.GeminiError) as ctx:
            tg.translate_gemini(["a"], client=c)
        self.assertNotIn(FAKE_KEY, str(ctx.exception))


class RotationTests(unittest.TestCase):
    def test_429_rotates_to_next_profile(self):
        def handler(n, body, headers):
            if headers["x-goog-api-key"] == FAKE_KEY:
                raise http_error(429)
            return echo_rows(body)
        c, fake = client(handler)
        self.assertEqual(tg.translate_gemini(["a"], client=c), ["譯L001"])
        self.assertEqual(c.rotations, 1)
        self.assertEqual(c.profile(), "google:b")
        # the next request starts on the profile that worked
        tg.translate_gemini(["a"], client=c)
        self.assertEqual(fake.headers[-1]["x-goog-api-key"], "AIza" + "Y" * 35)

    def test_all_profiles_limited_raises_without_key(self):
        c, _ = client(lambda n, body, h: (_ for _ in ()).throw(http_error(429)))
        with self.assertRaises(tg.GeminiError) as ctx:
            tg.translate_gemini(["a"], client=c)
        self.assertNotIn(FAKE_KEY, str(ctx.exception))
        self.assertNotIn("AIza", str(ctx.exception))
        self.assertEqual(c.rotations, 1)

    def test_key_goes_in_header_never_in_url_or_body(self):
        seen = []

        def handler(n, body, headers):
            seen.append(json.dumps(body))
            return echo_rows(body)
        c, fake = client(handler)
        tg.translate_gemini(["a"], client=c)
        self.assertNotIn(FAKE_KEY, seen[0])


class FallbackTests(unittest.TestCase):
    def test_auto_falls_back_to_oogg_when_gemini_fails(self):
        c, _ = client(lambda n, body, h: (_ for _ in ()).throw(http_error(500)))
        result = tg.translate_auto(["a", "b"], gemini=c, oogg=lambda rows: (["甲", "乙"], "oogg-model"))
        self.assertEqual((result.lines, result.backend, result.model), (["甲", "乙"], "oogg", "oogg-model"))

    def test_auto_prefers_gemini(self):
        c, _ = client(lambda n, body, h: echo_rows(body))
        called = []
        result = tg.translate_auto(["a"], gemini=c, oogg=lambda rows: called.append(1) or (["x"], "m"))
        self.assertEqual(result.backend, "gemini")
        self.assertEqual(result.model, tg.DEFAULT_MODEL)
        self.assertFalse(called)

    def test_translator_gemini_never_uses_oogg_and_oogg_never_uses_gemini(self):
        c, fake = client(lambda n, body, h: (_ for _ in ()).throw(http_error(500)))
        with self.assertRaises(RuntimeError):
            tg.translate_auto(["a"], translator="gemini", gemini=c, oogg=lambda rows: (["x"], "m"))
        c2, fake2 = client(lambda n, body, h: echo_rows(body))
        result = tg.translate_auto(["a"], translator="oogg", gemini=c2, oogg=lambda rows: (["x"], "m"))
        self.assertEqual(result.backend, "oogg")
        self.assertEqual(fake2.requests, [])

    def test_both_fail_message_has_no_key(self):
        c, _ = client(lambda n, body, h: (_ for _ in ()).throw(http_error(500)))

        def oogg(rows):
            raise RuntimeError("boom " + FAKE_KEY)
        with self.assertRaises(RuntimeError) as ctx:
            tg.translate_auto(["a"], gemini=c, oogg=oogg)
        self.assertNotIn(FAKE_KEY, str(ctx.exception))
        self.assertIn("gemini", str(ctx.exception))
        self.assertIn("oogg", str(ctx.exception))


class LineRetryTests(unittest.TestCase):
    def test_single_line_with_context_only_neighbours(self):
        def handler(n, body, h):
            return {"candidates": [{"content": {"parts": [{"text": json.dumps({"text": "只此行"})}]}}]}
        c, fake = client(handler)
        out = tg.line_retranslator(c, "T")("行", context=(["前"], ["後"]))
        self.assertEqual(out, "只此行")
        user = json.loads(fake.requests[0]["contents"][0]["parts"][0]["text"])
        self.assertEqual(user["context_before"], ["前"])
        self.assertEqual(user["context_after"], ["後"])
        self.assertIn("不要翻譯", fake.requests[0]["systemInstruction"]["parts"][0]["text"])

    def test_gate_passes_neighbours_to_a_context_retranslator(self):
        from lyricsee_core import LyricPair
        from lyrics_qa import gate
        import test_lyrics_qa as base
        translations = list(base.TRANS)
        translations[2] = base.TRANS[1]
        pairs = [LyricPair(o, t) for o, t in base.rows(translations=translations)]
        seen = {}

        def retr(line, context=((), ())):
            seen["ctx"] = context
            return base.TRANS[2]
        retr.accepts_context = True
        gate(pairs, retranslate=retr)
        self.assertEqual(seen["ctx"], (list(base.ORIG[0:2]), list(base.ORIG[3:5])))

class KeyLoadingTests(unittest.TestCase):
    def test_env_key_wins_and_splits(self):
        keys = tg.load_keys(env={"GEMINI_API_KEY": "k1, k2"}, openclaw_home=Path("nonexistent"))
        self.assertEqual([k for _, k in keys], ["k1", "k2"])

    def test_reads_google_profiles_from_store_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            db = home / "agents" / "main" / "agent" / "openclaw-agent.sqlite"
            db.parent.mkdir(parents=True)
            conn = sqlite3.connect(db)
            conn.execute("create table auth_profile_store (store_key text primary key, store_json text, updated_at int)")
            store = {"profiles": {"google:leprechaun": {"provider": "google", "type": "token", "token": FAKE_KEY},
                                  "openai:x": {"provider": "openai", "token": "nope"}}}
            conn.execute("insert into auth_profile_store values ('primary', ?, 0)", (json.dumps(store),))
            conn.commit()
            conn.close()
            before = db.read_bytes()
            keys = tg.load_keys(env={}, openclaw_home=home)
            self.assertEqual(keys, [("google:leprechaun", FAKE_KEY)])
            self.assertEqual(db.read_bytes(), before)

    def test_no_key_raises_clear_error_without_secrets(self):
        c = tg.GeminiClient(keys=[], post=lambda *a: {})
        with self.assertRaises(tg.GeminiError):
            c.generate("s", {})

    def test_mask_hides_key_shaped_text(self):
        self.assertNotIn(FAKE_KEY, tg.mask(f"url?key={FAKE_KEY}"))
        self.assertNotIn("zzz-secret", tg.mask("a zzz-secret b", ["zzz-secret"]))


if __name__ == "__main__":
    unittest.main()
