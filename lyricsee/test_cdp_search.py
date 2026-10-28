import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import autopick  # noqa: E402
import cdp_search as cdp  # noqa: E402
import lyricsee_core as core  # noqa: E402
import web_search as ws  # noqa: E402

ROWS = "<p>" + "<br>".join(f"原文の行{i}<br>譯文第{i}行" for i in range(10)) + "</p>"
NAMED_PAGE = f"<html><head><title>Song 歌詞</title></head><body>Artist {ROWS}</body></html>"
ANON_PAGE = f"<html><head><title>Song 歌詞</title></head><body>{ROWS}</body></html>"


class FakeChrome:
    """Stands in for the DevTools socket: records every command and answers like Chrome."""

    def __init__(self, links=None, url="https://www.google.com/search?q=x", title="x - Google", text="", ready="complete"):
        self.sent = []
        self.replies = []
        self.page = {"url": url, "ready": ready, "title": title, "text": text, "links": links or []}
        self.closed = False

    def send(self, text):
        message = json.loads(text)
        self.sent.append(message)
        method = message["method"]
        result = {}
        if method == "Target.createTarget":
            result = {"targetId": "OWN-TAB"}
        elif method == "Target.attachToTarget":
            result = {"sessionId": "S1"}
        elif method == "Runtime.evaluate":
            result = {"result": {"value": json.dumps(self.page)}}
        self.replies.append(json.dumps({"id": message["id"], "result": result}))

    def recv(self):
        return self.replies.pop(0)

    def close(self):
        self.closed = True

    def methods(self):
        return [m["method"] for m in self.sent]


def make_cdp(chrome, **kwargs):
    kwargs.setdefault("interval", 0.0)
    return cdp.CdpGoogle(connect=lambda url: chrome,
                         http_json=lambda url: {"webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/x"},
                         sleep=lambda s: None, **kwargs)


class CdpGoogleTests(unittest.TestCase):
    def test_one_background_tab_is_opened_used_and_closed(self):
        chrome = FakeChrome(links=["https://a.example/1", "https://www.google.com/x"])
        backend = make_cdp(chrome)
        self.assertEqual(backend.search("q1")[0], "https://a.example/1")
        backend.search("q2")
        backend.close()
        create = [m for m in chrome.sent if m["method"] == "Target.createTarget"]
        self.assertEqual(len(create), 1)
        self.assertTrue(create[0]["params"]["background"])
        close = [m for m in chrome.sent if m["method"] == "Target.closeTarget"]
        self.assertEqual([m["params"]["targetId"] for m in close], ["OWN-TAB"])
        self.assertEqual((backend.opened_targets, backend.closed_targets), (["OWN-TAB"], ["OWN-TAB"]))
        self.assertTrue(chrome.closed)

    def test_only_allowed_methods_are_used(self):
        chrome = FakeChrome(links=["https://a.example/1"])
        backend = make_cdp(chrome)
        backend.search("q")
        backend.close()
        allowed = {"Target.createTarget", "Target.attachToTarget", "Page.navigate", "Runtime.evaluate", "Target.closeTarget"}
        self.assertLessEqual(set(chrome.methods()), allowed)
        navigate = [m for m in chrome.sent if m["method"] == "Page.navigate"][0]
        self.assertEqual(navigate["sessionId"], "S1")
        self.assertTrue(navigate["params"]["url"].startswith("https://www.google.com/search?q="))
        self.assertTrue(navigate["params"]["url"].endswith("&hl=ja"))

    def test_close_is_idempotent_and_safe_without_a_tab(self):
        chrome = FakeChrome()
        backend = make_cdp(chrome)
        backend.close()
        backend.search("q")
        backend.close()
        backend.close()
        self.assertEqual(chrome.methods().count("Target.closeTarget"), 1)

    def test_captcha_page_stops_the_backend(self):
        chrome = FakeChrome(url="https://www.google.com/sorry/index?continue=x")
        backend = make_cdp(chrome)
        with self.assertRaises(cdp.CdpBlocked):
            backend.search("q")
        with self.assertRaises(cdp.CdpBlocked):
            backend.search("q2")
        self.assertEqual(chrome.methods().count("Page.navigate"), 1)
        backend.close()
        self.assertEqual(backend.closed_targets, ["OWN-TAB"])

    def test_consent_and_traffic_pages_are_blocks(self):
        self.assertTrue(cdp.is_blocked_page("https://consent.google.com/m?x", "", ""))
        self.assertTrue(cdp.is_blocked_page("https://www.google.com/search", "", "Our systems have detected unusual traffic"))
        self.assertFalse(cdp.is_blocked_page("https://www.google.com/search?q=a", "a - Google", "results"))

    def test_unreachable_port_is_unavailable_and_never_launches_chrome(self):
        def refuse(url):
            raise OSError("refused")

        backend = cdp.CdpGoogle(http_json=refuse, sleep=lambda s: None, interval=0.0)
        with self.assertRaises(cdp.CdpUnavailable):
            backend.search("q")

    def test_queries_are_spaced(self):
        now = [100.0]
        waits = []

        def sleep(seconds):
            waits.append(seconds)
            now[0] += seconds

        chrome = FakeChrome(links=["https://a.example/1"])
        backend = cdp.CdpGoogle(connect=lambda url: chrome, http_json=lambda url: {"webSocketDebuggerUrl": "ws://x"},
                                sleep=sleep, clock=lambda: now[0], interval=5.0)
        backend.search("q1")
        backend.search("q2")
        self.assertTrue(any(abs(w - 5.0) < 1e-6 for w in waits))


class Client:
    def __init__(self, pages):
        self.pages = pages
        self.requests = []

    def get_text(self, url, params=None, headers=None, retries=None):
        self.requests.append(url)
        if url in self.pages:
            return self.pages[url]
        return "<html></html>"


class FakeCdp:
    def __init__(self, results, blocked_after=None):
        self.results = results
        self.queries = []
        self.closed = False
        self.blocked_after = blocked_after

    def search(self, query):
        self.queries.append(query)
        if self.blocked_after is not None and len(self.queries) > self.blocked_after:
            raise cdp.CdpBlocked("sorry")
        return self.results

    def close(self):
        self.closed = True


class SessionIntegrationTests(unittest.TestCase):
    identity = core.TrackIdentity("Song", "Artist")

    def run_fetch(self, pages, backend):
        client = Client(pages)
        session = ws.session_for(client)
        if backend is not None:
            session.enable_cdp(backend)
        return core.fetch_google_candidates(self.identity, client), session, client

    def test_disabled_by_default(self):
        result, session, client = self.run_fetch({}, None)
        self.assertIsNone(session.cdp)
        self.assertEqual(result.candidates, [])

    def test_cli_flag_turns_the_backend_on_for_new_sessions(self):
        class Enabled(Client):
            pass

        saved = ws.CDP_DEFAULT
        ws.CDP_DEFAULT = True
        try:
            session = ws.session_for(Enabled({}))
        finally:
            ws.CDP_DEFAULT = saved
        self.assertIsInstance(session.cdp, cdp.CdpGoogle)

    def test_runs_at_most_eight_queries_and_logs_them(self):
        backend = FakeCdp([])
        result, session, client = self.run_fetch({}, backend)
        self.assertLessEqual(len(backend.queries), cdp.CDP_MAX_QUERIES)
        self.assertGreater(len(backend.queries), 0)
        logged = [r for r in session.records if r["engine"] == "google_cdp"]
        self.assertEqual(len(logged), len(backend.queries))
        self.assertEqual({r["status"] for r in logged}, {"ok"})

    def test_stops_when_an_identity_confirmed_full_length_page_turns_up(self):
        backend = FakeCdp(["https://pub.example/1"])
        result, session, client = self.run_fetch({"https://pub.example/1": NAMED_PAGE}, backend)
        self.assertEqual(len(backend.queries), 1)
        self.assertEqual([c.search_engine for c in result.candidates], ["google_cdp"])
        self.assertEqual(result.search_engine, "google_cdp")

    def test_unnamed_page_does_not_count_as_confirmed(self):
        backend = FakeCdp(["https://pub.example/1"])
        self.run_fetch({"https://pub.example/1": ANON_PAGE}, backend)
        self.assertGreater(len(backend.queries), 1)

    def test_captcha_stops_cdp_for_the_run_and_is_reported(self):
        backend = FakeCdp([], blocked_after=0)
        result, session, client = self.run_fetch({}, backend)
        self.assertEqual(len(backend.queries), 1)
        self.assertEqual(session.blocked["google_cdp"], "captcha")
        self.assertIn("google_cdp: blocked (captcha)", result.errors)
        self.assertEqual(session.cdp_search("again"), [])
        self.assertEqual(len(backend.queries), 1)

    def test_banned_host_from_cdp_results_is_never_fetched(self):
        backend = FakeCdp(["https://music.163.com/song?id=1"])
        result, session, client = self.run_fetch({}, backend)
        self.assertFalse(any("163.com" in url for url in client.requests))

    def test_unavailable_port_blocks_backend(self):
        class Down:
            def search(self, query):
                raise cdp.CdpUnavailable("port 9222 not answering")

            def close(self):
                pass

        result, session, client = self.run_fetch({}, Down())
        self.assertEqual(session.blocked["google_cdp"], "unavailable")


class OffVocalTests(unittest.TestCase):
    def test_marker_is_written_and_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertFalse(autopick.is_off_vocal(out, "01 Song"))
            path = autopick.mark_off_vocal(out, "01 Song", queries=8)
            self.assertTrue(autopick.is_off_vocal(out, "01 Song"))
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "off_vocal")
            self.assertEqual(record["reason"], "no_lyrics_found_after_tag_and_google_search")
            self.assertEqual(list(out.glob("txt/*")), [])

    def test_run_skips_off_vocal_stems_unless_asked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            (out / "candidates" / "01 Song").mkdir(parents=True)
            autopick.mark_off_vocal(out, "01 Song")
            audio_map = Path(tmp) / "map.json"
            audio_map.write_text("{}", encoding="utf-8")
            args = autopick.argparse.Namespace(out=str(out), audio_map=str(audio_map), cache_dir=None, stem=None,
                                               include_rejected=False, include_off_vocal=False)
            self.assertEqual(autopick.cmd_run(args), 0)
            report = next((out / "report").glob("autopick_*.json"))
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["songs"], [])
            report.unlink()
            args.include_off_vocal = True
            self.assertEqual(autopick.cmd_run(args), 0)
            report = next((out / "report").glob("autopick_*.json"))
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["songs"][0]["reason"], "audio_not_found")


if __name__ == "__main__":
    unittest.main()
