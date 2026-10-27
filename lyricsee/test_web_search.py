import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lyricsee_core as core  # noqa: E402
import web_search as ws  # noqa: E402


def bing_page(items):
    blocks = []
    for target, text in items:
        token = "a1" + base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
        blocks.append(
            f'<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&amp;&amp;p=x&amp;u={token}&amp;ntb=1">{text}</a></h2></li>'
        )
    return "<html><ol>" + "".join(blocks) + "</ol></html>"


ROWS = "<p>" + "<br>".join(f"原文の行{i}<br>譯文第{i}行" for i in range(10)) + "</p>"
FULL_PAGE = f"<html><head><title>Song 歌詞</title></head><body>{ROWS}</body></html>"
SHORT_PAGE = "<html><head><title>Song 歌詞</title></head><body><p>短<br>い</p></body></html>"


class QueryVariantTests(unittest.TestCase):
    def test_priority_order_and_sites(self):
        queries = ws.query_variants("Song", "Artist")
        self.assertEqual(queries[:5], ['"Song" 歌詞', "Song Artist 歌詞", "Song 歌詞 翻譯", "Song 中文歌詞", "Song lyrics"])
        self.assertEqual(queries[5], "Song Artist 歌詞 中文翻譯")
        self.assertEqual(queries[6], "Song 歌詞 中文翻譯")
        self.assertEqual(queries[7], "site:home.gamer.com.tw Song")
        self.assertEqual(ws.query_variants("Song", "")[4:6], ["Song 歌詞 中文翻譯", "site:home.gamer.com.tw Song"])
        for site in ws.SITE_QUERIES:
            self.assertIn(f"site:{site} Song", queries)

    def test_deduped_and_capped(self):
        queries = ws.query_variants("Song", "Artist", limit=6)
        self.assertEqual(len(queries), 6)
        everything = ws.query_variants("Ｌｏｖｅ Song", "Artist", limit=100)
        self.assertEqual(len(set(everything)), len(everything))

    def test_no_artist_skips_artist_query(self):
        self.assertFalse(any("Artist" in q for q in ws.query_variants("Song", "")))

    def test_title_forms_strip_number_brackets_feat_and_widths(self):
        forms = ws.title_forms_for_search("03. 白日夢 (feat. 倚水)")
        self.assertIn("白日夢", forms)
        self.assertEqual(forms[0], "03. 白日夢 (feat. 倚水)")
        forms = ws.title_forms_for_search("Ｌｏｖｅ　Song！")
        self.assertIn("Love Song!", forms)
        self.assertIn("Ｌｏｖｅ Ｓｏｎｇ！", forms)

    def test_typed_title_has_no_track_number_and_artist_is_cleaned(self):
        queries = ws.query_variants("01. 流転", "Islet (CV: Someone) / Other")
        self.assertEqual(queries[0], '"流転" 歌詞')
        self.assertEqual(queries[1], "流転 Islet 歌詞")

    def test_lyric_hosts_sort_first_and_uta_net_video_pages_are_dropped(self):
        urls = ["https://blog.example/a", "https://www.uta-net.com/song/1/", "https://home.gamer.com.tw/artwork.php?sn=1",
                "https://www.uta-net.com/movie/1/"]
        kept = ws._clean_result_urls(urls, "yahoojp")
        self.assertNotIn("https://www.uta-net.com/movie/1/", kept)
        self.assertEqual(sorted(kept, key=ws.result_priority)[0], "https://home.gamer.com.tw/artwork.php?sn=1")

    def test_music_163_is_never_queried_or_returned(self):
        self.assertFalse(any("163" in q for q in ws.query_variants("Song", "Artist")))
        self.assertTrue(ws.host_banned("https://music.163.com/song?id=1"))
        urls = ws._clean_result_urls(["https://music.163.com/song?id=1", "https://lyrics.example/a"], "bing")
        self.assertEqual(urls, ["https://lyrics.example/a"])


class ParserTests(unittest.TestCase):
    def test_bing_redirects_are_decoded_and_filtered_by_title(self):
        page = bing_page([("https://lyrics.example/song", "Song 歌詞"), ("https://travel.example/", "台東 民宿")])
        self.assertEqual(ws.parse_bing_results(page), ["https://lyrics.example/song", "https://travel.example/"])
        self.assertEqual(ws.parse_bing_results(page, must_contain="song"), ["https://lyrics.example/song"])

    def test_yahoo_drops_engine_and_social_links(self):
        page = (
            '<a href="https://login.yahoo.co.jp/x">l</a><a href="https://www.youtube.com/watch?v=1">y</a>'
            '<a href="https://home.gamer.com.tw/artwork.php?sn=1">g</a><a href="https://www.uta-net.com/song/1/">u</a>'
            '<a href="https://www.lycorp.co.jp/ja/company/terms/">t</a><a href="/relative">r</a>'
        )
        self.assertEqual(
            ws.parse_yahoojp_results(page),
            ["https://home.gamer.com.tw/artwork.php?sn=1", "https://www.uta-net.com/song/1/"],
        )

    def test_block_classification(self):
        self.assertEqual(ws.classify_block("google", "<noscript>enablejs</noscript>", []), "captcha")
        self.assertEqual(ws.classify_block("duckduckgo", '<div class="anomaly-modal__mask">', []), "captcha")
        self.assertEqual(ws.classify_block("bing", "<html>plain</html>", []), "empty")
        self.assertIsNone(ws.classify_block("bing", "<html>plain</html>", ["https://a.example/"]))


class FakeClient:
    def __init__(self, handler):
        self.handler = handler
        self.requests = []

    def get_text(self, url, params=None, headers=None, retries=None):
        self.requests.append((url, dict(params or {})))
        return self.handler(url, params or {})


class SessionTests(unittest.TestCase):
    def test_captcha_marks_engine_blocked_for_the_run(self):
        def handler(url, params):
            if "google.com" in url:
                return "<noscript>enablejs</noscript>"
            if "duckduckgo" in url:
                return '<div class="anomaly-modal__mask"></div>'
            if "yahoo" in url:
                return '<a href="https://lyrics.example/a">a</a>'
            raise AssertionError(url)

        client = FakeClient(handler)
        session = ws.SearchSession(client)
        for query in ("q1", "q2", "q3"):
            self.assertEqual(session.find_urls(query), (["https://lyrics.example/a"], "yahoojp"))
        urls = [url for url, _ in client.requests]
        self.assertEqual(sum("google.com" in u for u in urls), 1)
        self.assertEqual(sum("duckduckgo" in u for u in urls), 1)
        self.assertEqual(session.blocked, {"google": "captcha", "duckduckgo": "captcha"})
        self.assertEqual([r["outcome"] for r in session.records if r["engine"] == "google"], ["captcha"])

    def test_block_cooldown_expires_except_for_google(self):
        now = [0.0]
        session = ws.SearchSession(FakeClient(lambda url, params: ""), clock=lambda: now[0])
        session._block("google", "captcha")
        session._block("yahoojp", "error:429")
        self.assertEqual(session.available(), ["duckduckgo", "bing"])
        now[0] = ws.BLOCK_COOLDOWN + 1
        self.assertEqual(session.available(), ["duckduckgo", "yahoojp", "bing"])
        self.assertIn("google", session.blocked)
        session._block("yahoojp", "error:429")
        now[0] += ws.BLOCK_COOLDOWN + 1
        self.assertNotIn("yahoojp", session.available())  # second block rests twice as long
        now[0] += ws.BLOCK_COOLDOWN
        self.assertIn("yahoojp", session.available())

    def test_rate_limit_waits_between_requests_to_one_host(self):
        now = [100.0]
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        client = FakeClient(lambda url, params: "<html></html>")
        session = ws.SearchSession(client, engines=("bing",), interval=2.0, sleep=sleep, clock=lambda: now[0])
        session.search("bing", "a")
        session.search("bing", "b")
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 2.0)

    def test_429_blocks_engine_and_timeouts_block_after_two(self):
        calls = {"bing": 0, "yahoojp": 0}

        def handler(url, params):
            if "bing" in url:
                calls["bing"] += 1
                raise HTTPError(url, 429, "slow down", {}, None)
            calls["yahoojp"] += 1
            raise URLError("timed out")

        session = ws.SearchSession(FakeClient(handler), engines=("bing", "yahoojp"))
        for query in ("a", "b", "c", "d"):
            self.assertEqual(session.find_urls(query), ([], "none"))
        self.assertEqual(calls, {"bing": 1, "yahoojp": 2})
        self.assertEqual(session.blocked, {"bing": "error:429", "yahoojp": "timeouts"})

    def test_every_query_is_logged_to_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "q.jsonl"
            client = FakeClient(lambda url, params: '<a href="https://lyrics.example/a">a</a>')
            session = ws.SearchSession(client, engines=("yahoojp",), log_path=log)
            session.context = "Song"
            session.find_urls("hello")
            record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((record["engine"], record["query"], record["status"], record["outcome"]), ("yahoojp", "hello", 200, "ok"))
        self.assertEqual(record["results"], ["https://lyrics.example/a"])
        self.assertEqual(record["context"], "Song")


def page_handler(search_pages):
    def handler(url, params):
        if "search.yahoo.co.jp" in url:
            return search_pages.pop(0) if search_pages else "<html></html>"
        if "google.com" in url:
            return "<noscript>enablejs</noscript>"
        if "duckduckgo" in url:
            return '<div class="anomaly-modal__mask"></div>'
        if "bing.com" in url:
            return "<html></html>"
        if url.startswith(("https://a.example/", "https://b.example/")):
            return FULL_PAGE
        if url.startswith("https://c.example/"):
            return SHORT_PAGE
        raise AssertionError(url)

    return handler


class FetchTests(unittest.TestCase):
    def test_stops_once_two_full_length_pages_agree_after_the_gamer_query(self):
        client = FakeClient(page_handler(['<a href="https://a.example/1">a</a><a href="https://b.example/1">b</a>']))
        result = core.fetch_google_candidates(core.TrackIdentity("Song", "Artist"), client)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(result.search_engine, "yahoojp")
        queries = [params["p"] for url, params in client.requests if "search.yahoo" in url]
        self.assertEqual(len(queries), 8)
        self.assertEqual(queries[-1], "site:home.gamer.com.tw Song")
        self.assertIn("google: blocked (captcha)", result.errors)

    def test_keeps_searching_when_pages_do_not_agree_and_dedupes_urls(self):
        pages = [
            '<a href="https://c.example/1">c</a>',
            '<a href="https://c.example/1">c</a><a href="https://a.example/1">a</a>',
            '<a href="https://b.example/2">b</a>',
        ]
        client = FakeClient(page_handler(pages))
        result = core.fetch_google_candidates(core.TrackIdentity("Song", "Artist"), client)
        self.assertEqual(len([u for u, _ in client.requests if u.startswith("https://c.example/")]), 1)
        self.assertEqual([c.url for c in result.candidates], ["https://a.example/1", "https://b.example/2"])
        queries = [params["p"] for url, params in client.requests if "search.yahoo" in url]
        self.assertEqual(queries[:3], ['"Song" 歌詞', "Song Artist 歌詞", "Song 歌詞 翻譯"])

    def test_publisher_pages_agreeing_do_not_stop_before_a_translation_post(self):
        def cand(name, post=False):
            return core.LyricCandidate(
                source=f"google:{name}", url=f"https://{name}.example/", title="t", artist="",
                lines=[core.LyricPair(f"原文の行{i}", f"譯{i}") for i in range(10)],
                meta={"artist_confirmed": True} if post else {},
            )

        self.assertTrue(core._agreeing_full_length([cand("a"), cand("b")]))
        self.assertFalse(core._agreeing_full_length([cand("a"), cand("b")], need_post=True))
        self.assertTrue(core._agreeing_full_length([cand("a"), cand("b", post=True)], need_post=True))
        self.assertFalse(core._agreeing_full_length([cand("a", post=True)], need_post=True))

    def test_banned_host_result_is_not_fetched(self):
        pages = ['<a href="https://music.163.com/song?id=1">n</a><a href="https://a.example/1">a</a>']
        client = FakeClient(page_handler(pages))
        core.fetch_google_candidates(core.TrackIdentity("Song", "Artist"), client)
        self.assertFalse(any("163.com" in url for url, _ in client.requests))


if __name__ == "__main__":
    unittest.main()
