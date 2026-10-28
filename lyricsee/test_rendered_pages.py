import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import autopick  # noqa: E402
import cdp_search as cdp  # noqa: E402
import lyric_pages as lp  # noqa: E402
import lyricsee_core as core  # noqa: E402
import web_search as ws  # noqa: E402
from test_cdp_search import FakeChrome, make_cdp  # noqa: E402

ORIG = [f"Line number {i} of the song" for i in range(12)]
TRANS = [f"第{i}句翻译" for i in range(12)]


def timed(texts, offset=0.0):
    return [f"[{int((i * 7 + offset) // 60):02d}:{(i * 7 + offset) % 60:06.3f}]{text}" for i, text in enumerate(texts)]


class TimedBlockTests(unittest.TestCase):
    def test_original_and_translation_blocks_pair_by_timestamp_and_drop_the_credit_header(self):
        credits = ["[00:00.100]出品 someone", "[00:00.500]演唱 someone", "[00:01.000]作词 someone"]
        lines = credits + timed(ORIG, 3.0) + timed(TRANS, 3.0)
        rows = lp.timed_pairs(lines)
        self.assertEqual([r.original for r in rows], ORIG)
        self.assertEqual([r.translation for r in rows], TRANS)

    def test_single_block_gives_originals_only(self):
        rows = lp.timed_pairs(timed(ORIG))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r.translation is None for r in rows))

    def test_untimed_text_gives_nothing(self):
        self.assertEqual(lp.timed_pairs(ORIG), [])


class ContainerTests(unittest.TestCase):
    def test_genius_container_html_becomes_rows_without_section_tags(self):
        body = "<br/>".join(["[Verse 1]"] + ORIG[:6] + ["[Chorus]"] + ORIG[6:])
        html = f'<html><body><div class="x"><div data-lyrics-container="true">{body}</div></div><div>menu</div></body></html>'
        rows, how = lp.rows_from_document(html)
        self.assertEqual(how, "container")
        self.assertEqual([r.original for r in rows], ORIG)

    def test_rendered_container_text_is_preferred_over_body_text(self):
        rows, how = lp.rows_from_text("menu\nmenu", ["\n".join(ORIG)])
        self.assertEqual((how, len(rows)), ("container", 12))


class RunTests(unittest.TestCase):
    def test_navigation_like_text_gives_no_run(self):
        self.assertEqual(lp.run_of_lyric_lines(["Home", "Top", "Login", "ok", "Songs", "Menu"] * 3), [])

    def test_page_that_says_it_has_no_lyrics_gives_no_run(self):
        text = "We currently do not have these lyrics.\n" + "\n".join(ORIG)
        self.assertEqual(lp.rows_from_text(text), ([], ""))

    def test_run_is_found_between_furniture(self):
        text = "Login\nRegister\n" + "\n".join(ORIG) + "\nhttps://example.com/x\nfooter"
        rows, how = lp.rows_from_text(text)
        self.assertEqual((how, len(rows)), ("run", 12))

    def test_title_and_artist_rows_are_dropped_from_the_top(self):
        rows = [lp.LyricPair(t) for t in ["Song (feat. x)", "Laplacian"] + ORIG]
        self.assertEqual([r.original for r in lp.drop_title_rows(rows, "Song")], ORIG)
        rows = [lp.LyricPair(t) for t in ORIG]
        self.assertEqual(len(lp.drop_title_rows(rows, "Song")), 12)


class FetchPageTests(unittest.TestCase):
    def page(self, **extra):
        page = {"url": "https://lyrics.example/a", "ready": "complete", "title": "Song - Artist", "text": "\n".join(ORIG),
                "containers": []}
        page.update(extra)
        return page

    def test_uses_the_one_background_tab_and_only_allowed_methods(self):
        chrome = FakeChrome()
        chrome.page = self.page()
        backend = make_cdp(chrome)
        first = backend.fetch_page("https://lyrics.example/a")
        backend.fetch_page("https://lyrics.example/b")
        backend.close()
        self.assertEqual(first["text"], "\n".join(ORIG))
        self.assertEqual(chrome.methods().count("Target.createTarget"), 1)
        self.assertEqual(chrome.methods().count("Target.closeTarget"), 1)
        self.assertEqual(backend.pages_loaded, 2)
        allowed = {"Target.createTarget", "Target.attachToTarget", "Page.navigate", "Runtime.evaluate", "Target.closeTarget"}
        self.assertLessEqual(set(chrome.methods()), allowed)
        navigated = [m["params"]["url"] for m in chrome.sent if m["method"] == "Page.navigate"]
        self.assertEqual(navigated, ["https://lyrics.example/a", "https://lyrics.example/b"])

    def test_loads_are_spaced_by_the_search_interval(self):
        now = [100.0]

        def sleep(seconds):
            now[0] += seconds

        chrome = FakeChrome()
        chrome.page = self.page()
        backend = cdp.CdpGoogle(connect=lambda url: chrome, http_json=lambda url: {"webSocketDebuggerUrl": "ws://x"},
                                sleep=sleep, clock=lambda: now[0], interval=5.0)
        backend.fetch_page("https://lyrics.example/a")
        before = now[0]
        backend.fetch_page("https://lyrics.example/b")
        self.assertGreaterEqual(now[0] - before, 5.0)

    def test_challenge_page_returns_no_text_and_is_not_solved(self):
        chrome = FakeChrome()
        chrome.page = self.page(title="Just a moment...", text="Checking your browser")
        backend = make_cdp(chrome)
        page = backend.fetch_page("https://lyrics.example/a")
        self.assertTrue(page["blocked"])
        self.assertEqual((page["text"], page["containers"]), ("", []))

    def test_a_google_challenge_does_not_stop_page_loads(self):
        chrome = FakeChrome()
        chrome.page = self.page()
        backend = make_cdp(chrome)
        backend.blocked = True
        self.assertEqual(backend.fetch_page("https://lyrics.example/a")["text"], "\n".join(ORIG))

    def test_unreachable_port_raises_unavailable(self):
        def refuse(url):
            raise OSError("refused")

        backend = cdp.CdpGoogle(http_json=refuse, sleep=lambda s: None, interval=0.0)
        with self.assertRaises(cdp.CdpUnavailable):
            backend.fetch_page("https://lyrics.example/a")


class RenderingCdp:
    """Backend double: fetch_page answers from a dict of rendered pages."""

    def __init__(self, pages):
        self.pages = pages
        self.loaded = []

    def search(self, query):
        return []

    def fetch_page(self, url):
        self.loaded.append(url)
        page = self.pages.get(url) or {"title": "", "text": "", "containers": []}
        return {"url": url, "blocked": False, **page}

    def close(self):
        pass


class Client:
    def __init__(self, pages):
        self.pages = pages

    def get_text(self, url, params=None, headers=None, retries=None):
        if url not in self.pages:
            raise OSError("403")
        return self.pages[url]


IDENTITY = core.TrackIdentity("Song", "Artist")
STUB = "<html><head><title>Song lyrics</title></head><body><p>Artist</p><p>a few words</p></body></html>"
UNNAMED_STUB = "<html><head><title>Song lyrics</title></head><body><p>a few words</p></body></html>"
RENDERED = {"title": "Song lyrics - Artist", "text": "Artist\n" + "\n".join(ORIG), "containers": []}


class RenderedCandidateTests(unittest.TestCase):
    def candidate(self, pages, rendered, url="https://lyrics.example/a"):
        client = Client(pages)
        session = ws.session_for(client)
        backend = RenderingCdp(rendered)
        session.enable_cdp(backend)
        return core._page_candidate(IDENTITY, client, url, "test", ""), backend, session

    def test_short_static_page_naming_the_artist_is_rendered(self):
        candidate, backend, session = self.candidate({"https://lyrics.example/a": STUB}, {"https://lyrics.example/a": RENDERED})
        self.assertEqual(len(candidate.lines), 12)
        self.assertEqual(candidate.meta["rows_from"], "rendered_run")
        self.assertIn("artist", [e.lower() for e in candidate.meta["identity_evidence"]])
        self.assertEqual(backend.loaded, ["https://lyrics.example/a"])
        self.assertEqual([r["engine"] for r in session.records], ["cdp_page"])

    def test_page_that_does_not_name_the_artist_is_not_rendered(self):
        candidate, backend, _ = self.candidate({"https://lyrics.example/a": UNNAMED_STUB}, {"https://lyrics.example/a": RENDERED})
        self.assertIsNone(candidate)
        self.assertEqual(backend.loaded, [])

    def test_refused_page_is_rendered_and_rows_need_the_artist_in_the_rendered_text(self):
        pages = {"https://lyrics.example/a": {"title": "Song lyrics", "text": "\n".join(ORIG), "containers": []}}
        candidate, backend, _ = self.candidate({}, pages)
        self.assertIsNone(candidate)            # rendered text does not name the artist
        self.assertEqual(backend.loaded, ["https://lyrics.example/a"])
        candidate, _, _ = self.candidate({}, {"https://lyrics.example/a": RENDERED})
        self.assertEqual(len(candidate.lines), 12)

    def test_full_static_page_is_not_rendered(self):
        rows = "<br>".join(f"原文の行{i}<br>譯文第{i}行" for i in range(10))
        page = f"<html><head><title>Song</title></head><body>Artist {rows}</body></html>"
        candidate, backend, _ = self.candidate({"https://lyrics.example/a": page}, {})
        self.assertEqual(len(candidate.lines), 10)
        self.assertEqual(backend.loaded, [])

    def test_timed_static_page_is_found_without_a_browser(self):
        body = "<br>".join(timed(ORIG, 3.0) + timed(TRANS, 3.0))
        page = f"<html><head><title>Song - Artist</title></head><body>{body}</body></html>"
        client = Client({"https://lyrics.example/t": page})
        candidate = core._page_candidate(IDENTITY, client, "https://lyrics.example/t", "test", "")
        self.assertEqual(candidate.meta["rows_from"], "static_timed")
        self.assertEqual(candidate.lines[0].translation, TRANS[0])

    def test_six_rendered_loads_per_song(self):
        client = Client({})
        session = ws.session_for(client)
        backend = RenderingCdp({})
        session.enable_cdp(backend)
        for i in range(9):
            core._page_candidate(IDENTITY, client, f"https://lyrics.example/{i}", "test", "")
        self.assertEqual(len(backend.loaded), cdp.RENDER_MAX_LOADS)

    def test_backend_failure_does_not_stop_the_song(self):
        class Broken(RenderingCdp):
            def fetch_page(self, url):
                raise RuntimeError("boom")

        client = Client({"https://lyrics.example/a": STUB})
        session = ws.session_for(client)
        session.enable_cdp(Broken({}))
        self.assertIsNone(core._page_candidate(IDENTITY, client, "https://lyrics.example/a", "test", ""))
        self.assertEqual(session.records[-1]["status"], "error")


class SparseOverrideTests(unittest.TestCase):
    def entry(self, coverage, precision, chars=313):
        return {"asr": {"coverage": coverage, "precision": precision, "score": 0.7}, "cleaned_chars": chars}

    def test_text_that_explains_the_singing_and_names_the_artist_is_no_snippet(self):
        self.assertTrue(autopick.sparse_but_complete(self.entry(0.79, 0.71), ["yuki"]))

    def test_needs_identity_evidence_and_good_asr(self):
        self.assertFalse(autopick.sparse_but_complete(self.entry(0.79, 0.71), []))
        self.assertFalse(autopick.sparse_but_complete(self.entry(0.30, 0.90), ["yuki"]))
        self.assertFalse(autopick.sparse_but_complete(self.entry(0.90, 0.30), ["yuki"]))
        self.assertFalse(autopick.sparse_but_complete(self.entry(0.90, 0.90, chars=20), ["yuki"]))


if __name__ == "__main__":
    unittest.main()
