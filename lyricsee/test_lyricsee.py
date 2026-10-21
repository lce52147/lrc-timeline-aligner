from __future__ import annotations

import importlib.util
import gzip
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from unittest import mock


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "lyricsee_core.py"


def load_core():
    if not CORE_PATH.exists():
        raise AssertionError("lyricsee_core.py has not been implemented")
    spec = importlib.util.spec_from_file_location("lyricsee_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise AssertionError("could not load lyricsee_core.py")
    spec.loader.exec_module(module)
    return module


class LyricseeCoreTests(unittest.TestCase):
    def test_filename_identity_uses_artist_dash_title(self):
        core = load_core()
        identity = core.identity_from_filename(Path("MyGO!!!!! - 影色舞.flac"))
        self.assertEqual(identity.artist, "MyGO!!!!!")
        self.assertEqual(identity.title, "影色舞")

    def test_parse_and_pair_lrc_keeps_untranslated_original_rows(self):
        core = load_core()
        original = core.parse_lrc("[00:01.00]原文一\n[00:02.50]原文二\n")
        translated = core.parse_lrc("[00:01.00]翻譯一\n")
        pairs = core.pair_timed_lyrics(original, translated)
        self.assertEqual(
            [(row.original, row.translation) for row in pairs],
            [("原文一", "翻譯一"), ("原文二", None)],
        )
        self.assertAlmostEqual(core.translation_coverage(pairs), 0.5)

    def test_credit_rows_are_excluded_from_all_input_paths(self):
        core = load_core()
        credits = "词：Diggy-MO'\n詞：Diggy-MO'\n曲：長谷川大介"
        source = "\n".join(f"[00:{i:02d}.00]{line}" for i, line in enumerate(credits.splitlines(), 1))
        source += "\n[00:04.00]失えばいい"
        self.assertEqual([line.text for line in core.parse_lrc(source)], ["失えばいい"])
        self.assertEqual(
            [row.original for row in core._plain_lyric_pairs(credits + "\n失えばいい")],
            ["失えばいい"],
        )
        identity = core.TrackIdentity(title="Example", artist="Example Artist")
        rows = [core.LyricPair(line, line) for line in credits.splitlines()]
        rows.append(core.LyricPair("失えばいい", "不畏失去"))
        self.assertEqual(
            [row.original for row in core._sanitize_candidate_lines(rows, identity)],
            ["失えばいい"],
        )

    def test_parse_google_results_extracts_external_links_once(self):
        core = load_core()
        html = """
        <a href="/url?q=https%3A%2F%2Fexample.com%2Flyrics%3Fa%3D1&amp;sa=U">A</a>
        <a href="https://example.com/lyrics?a=1">duplicate</a>
        <a href="https://www.google.com/search?q=ignored">Google</a>
        <a href="https://another.example/song">B</a>
        """
        self.assertEqual(
            core.parse_google_results(html),
            ["https://example.com/lyrics?a=1", "https://another.example/song"],
        )

    def test_extract_bilingual_pairs_accepts_adjacent_japanese_and_chinese_lines(self):
        core = load_core()
        html = """
        <html><body><main>
          <p>君の声を聞かせて<br>讓我聽見你的聲音<br>
          夜を越えて行こう<br>跨越黑夜前行</p>
        </main></body></html>
        """
        pairs = core.extract_bilingual_pairs(html)
        self.assertEqual(
            [(row.original, row.translation) for row in pairs],
            [
                ("君の声を聞かせて", "讓我聽見你的聲音"),
                ("夜を越えて行こう", "跨越黑夜前行"),
            ],
        )

    def test_extract_bilingual_pairs_discards_page_chrome_credits_and_related_links(self):
        core = load_core()
        html = """
        <html><head><title>Song - Artist - 巴哈姆特</title></head><body>
          <div>巴哈姆特 Now! - 我的聊天列表</div><div>好友</div>
          <div>PV：someone</div><div>唄：Singer</div>
          <p>君の声を聞かせて<br>讓我聽見你的聲音<br>
          夜を越えて行こう<br>跨越黑夜前行<br>
          光を探して<br>尋找那道光</p>
          <div>david110442</div><div>追蹤</div>
          <div>【歌詞翻譯】別の歌</div><div>相關創作</div>
        </body></html>
        """
        pairs = core.extract_bilingual_pairs(html)
        self.assertEqual(
            [(row.original, row.translation) for row in pairs],
            [
                ("君の声を聞かせて", "讓我聽見你的聲音"),
                ("夜を越えて行こう", "跨越黑夜前行"),
                ("光を探して", "尋找那道光"),
            ],
        )

    def test_extract_bilingual_pairs_ignores_marumaru_repeat_helper_rows(self):
        core = load_core()
        html = """
        <html><body><p>
          私が私の事を愛して<br>我愛著我自己<br>
          What's wrong with me loving myself?<br>重覆這句<br>
          何が悪いの?<br>有什麼不行?<br>
          Is it jealousy?<br>重覆這句<br>
          届きませんね<br>都傳不進我的耳裡呢
        </p></body></html>
        """
        pairs = core.extract_bilingual_pairs(html)
        self.assertEqual(
            [(row.original, row.translation) for row in pairs],
            [
                ("私が私の事を愛して", "我愛著我自己"),
                ("何が悪いの?", "有什麼不行?"),
                ("届きませんね", "都傳不進我的耳裡呢"),
            ],
        )

    def test_extract_bilingual_pairs_drops_engineering_credits(self):
        core = load_core()
        html = """
        <p>
          Mastering Engineer: raku<br>母帶工程師：raku<br>
          君の声<br>你的聲音<br>
          夜を越える<br>跨越黑夜
        </p>
        """
        pairs = core.extract_bilingual_pairs(html)
        self.assertEqual(
            [(row.original, row.translation) for row in pairs],
            [("君の声", "你的聲音"), ("夜を越える", "跨越黑夜")],
        )

    def test_extract_bilingual_pairs_rejects_chinese_commentary_with_incidental_latin(self):
        core = load_core()
        html = """
        <p>
          君の声<br>你的聲音<br>
          夜を越える<br>跨越黑夜<br>
          這部是 MELLOW 個人風格最濃厚的遊戲，最後也不是 HAPPY END<br>
          如果你能接受主角團會死人，或許能從遊戲中得到什麼<br>
        </p>
        """
        pairs = core.extract_bilingual_pairs(html)
        self.assertEqual(
            [(row.original, row.translation) for row in pairs],
            [("君の声", "你的聲音"), ("夜を越える", "跨越黑夜")],
        )

    def test_choose_web_candidate_prefers_more_translated_rows(self):
        core = load_core()
        weaker = core.LyricCandidate(
            source="lrclib",
            url="https://lrclib.net/api/get/1",
            title="Song",
            artist="Artist",
            lines=[core.LyricPair("a", "甲"), core.LyricPair("b", None)],
        )
        stronger = core.LyricCandidate(
            source="google:https://example.com",
            url="https://example.com",
            title="Song",
            artist="Artist",
            lines=[core.LyricPair("a", "甲"), core.LyricPair("b", "乙")],
        )
        self.assertIs(core.choose_candidate([weaker, stronger], require_translation=True), stronger)

    def test_choose_web_candidate_prefers_more_complete_translation_regardless_of_source_type(self):
        core = load_core()
        structured = core.LyricCandidate(
            source="lrclib",
            url="https://lrclib.net/api/get/1",
            title="Song",
            artist="Artist",
            lines=[core.LyricPair(f"line {index}", f"translation {index}") for index in range(54)],
        )
        generic_page = core.LyricCandidate(
            source="google:blog.example",
            url="https://blog.example/song",
            title="Song",
            artist="Artist",
            lines=[core.LyricPair(f"line {index}", f"translation {index}") for index in range(86)],
        )
        self.assertIs(
            core.choose_candidate([generic_page, structured], require_translation=True),
            generic_page,
        )

    def test_render_lyrics_uses_two_ascii_spaces_between_columns(self):
        core = load_core()
        rendered = core.render_lyrics(
            [core.LyricPair("原文一", "翻譯一"), core.LyricPair("原文二", None)]
        )
        self.assertEqual(rendered, "原文一  翻譯一\n原文二\n")

    def test_render_lyrics_converts_translation_to_traditional_chinese(self):
        core = load_core()
        rendered = core.render_lyrics(
            [core.LyricPair("君の声", "还能像镜子对面的你那样\x00发问")]
        )
        self.assertEqual(rendered, "君の声  還能像鏡子對面的你那樣發問\n")

    def test_collect_candidates_sanitizes_structured_metadata_without_dropping_lyrics(self):
        core = load_core()

        def lrclib(identity, client):
            return core.LyricCandidate(
                source="lrclib",
                url="https://lrclib.net/api/get/1",
                title="無路矢",
                artist="MyGO!!!!!",
                lines=[
                    core.LyricPair("無路矢 - MyGO!!!!!"),
                    core.LyricPair("本当は僕だけが違う星から", "只有我从截然不同的星球而来"),
                    core.LyricPair("(yeah yeah yeah yeah yeah)"),
                    core.LyricPair("..."),
                    core.LyricPair("The Way"),
                    core.LyricPair("♪"),
                    core.LyricPair("母带工程师 : raku"),
                    core.LyricPair("录音工程师 : Shunsuke Miyazawa"),
                    core.LyricPair("纯音乐，请欣赏"),
                    core.LyricPair("END"),
                    core.LyricPair("undefined"),
                ],
            )

        result = core.collect_candidates(
            core.TrackIdentity("無路矢", "MyGO!!!!!"),
            object(),
            lrclib_fetcher=lrclib,
            google_fetcher=lambda identity, client: ([], []),
        )
        self.assertEqual(
            [row.original for row in result.candidates[0].lines],
            ["本当は僕だけが違う星から", "(yeah yeah yeah yeah yeah)", "...", "The Way"],
        )

    def test_write_result_refuses_to_overwrite_existing_lyrics(self):
        core = load_core()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "song.flac"
            audio.write_bytes(b"fake")
            output = root / "txt" / "song.lyrics.txt"
            output.parent.mkdir()
            output.write_text("existing\n", encoding="utf-8")
            candidate = core.LyricCandidate(
                source="test",
                url="https://example.com",
                title="song",
                artist="artist",
                lines=[core.LyricPair("original", "translation")],
            )
            with self.assertRaises(FileExistsError):
                core.write_result(audio, candidate, output_root=root, overwrite=False)
            self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")

    def test_atomic_write_preserves_existing_file_when_replace_fails(self):
        core = load_core()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "song.lyrics.txt"
            output.write_text("existing\n", encoding="utf-8")

            with mock.patch.object(core.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    core._atomic_write_text(output, "replacement\n")

            self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")
            self.assertEqual(list(root.glob("*.tmp")), [])
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_write_result_records_sources_without_lyrics_in_json(self):
        core = load_core()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "song.flac"
            audio.write_bytes(b"fake")
            candidate = core.LyricCandidate(
                source="lrclib",
                url="https://lrclib.net/api/get/1",
                title="song",
                artist="artist",
                lines=[core.LyricPair("original", "translation")],
                discovered_urls=["https://example.com/translation"],
                search_url="https://html.duckduckgo.com/html/?q=song",
                search_engine="duckduckgo",
            )
            output, provenance = core.write_result(audio, candidate, output_root=root, overwrite=False)
            self.assertEqual(output, root / "txt" / "song.lyrics.txt")
            self.assertEqual(provenance, root / "report" / "song.lyrics.sources.json")
            self.assertFalse((root / "song.lyrics.txt").exists())
            self.assertFalse((root / "song.lyrics.sources.json").exists())
            self.assertEqual(output.read_text(encoding="utf-8"), "original  translation\n")
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            self.assertEqual(payload["selected_source"], "lrclib")
            self.assertEqual(payload["source_url"], "https://lrclib.net/api/get/1")
            self.assertEqual(payload["google_candidates"], ["https://example.com/translation"])
            self.assertEqual(payload["search_url"], "https://html.duckduckgo.com/html/?q=song")
            self.assertEqual(payload["search_engine"], "duckduckgo")
            self.assertNotIn("lyrics", payload)

    def test_lrclib_candidate_supplies_original_when_translation_is_unavailable(self):
        core = load_core()

        class Client:
            def get_json(self, url, params=None):
                return [
                    {
                        "id": 7,
                        "trackName": "Song",
                        "artistName": "Artist",
                        "albumName": "Album",
                        "duration": 120.0,
                        "plainLyrics": "first line\nsecond line",
                        "syncedLyrics": None,
                    }
                ]

        candidate = core.fetch_lrclib_candidate(
            core.TrackIdentity("Song", "Artist", duration=120.0), Client()
        )
        self.assertEqual(candidate.source, "lrclib")
        self.assertEqual([row.original for row in candidate.lines], ["first line", "second line"])
        self.assertEqual(core.translation_coverage(candidate.lines), 0.0)

    def test_parse_duckduckgo_results_decodes_redirects_dedupes_and_preserves_order(self):
        core = load_core()
        document = """
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fhome.gamer.com.tw%2Fartwork.php%3Fsn%3D5820258&amp;rut=x">a</a>
        <a class="result__a" href="https://marumaru-x.com/japanese-song/play-y7okwq04oq">b</a>
        <a class="result__a" href="https://marumaru-x.com/japanese-song/play-y7okwq04oq">dup</a>
        <a class="other" href="https://ignored.example/song">ignored</a>
        """
        self.assertEqual(
            core.parse_duckduckgo_results(document),
            [
                "https://home.gamer.com.tw/artwork.php?sn=5820258",
                "https://marumaru-x.com/japanese-song/play-y7okwq04oq",
            ],
        )

    def test_google_discovery_uses_duckduckgo_before_headless_chrome(self):
        core = load_core()

        class Client:
            user_agent = "test-agent"

            def __init__(self):
                self.headless_calls = 0
                self.google_headers = None
                self.ddg_headers = None

            def get_text(self, url, params=None, headers=None):
                if "google.com/search" in url:
                    self.google_headers = headers
                    return '<a href="https://support.google.com/websearch">blocked</a>'
                if "duckduckgo.com/html" in url:
                    self.ddg_headers = headers
                    return '<a class="result__a" href="https://lyrics.example/song">result</a>'
                return "<html><head><title>Song - Lyrics</title></head><body><p>君の声<br>你的聲音<br>夜を越える<br>跨越黑夜</p></body></html>"

            def google_search_urls(self, identity):
                self.headless_calls += 1
                return ["https://headless.example/should-not-run"]

        client = Client()
        result = core.fetch_google_candidates(core.TrackIdentity("Song", "Artist"), client)
        self.assertEqual(result.urls, ["https://lyrics.example/song"])
        self.assertEqual(result.search_engine, "duckduckgo")
        self.assertTrue(result.search_url.startswith("https://html.duckduckgo.com/html/?q="))
        self.assertEqual(client.headless_calls, 0)
        self.assertTrue(client.google_headers["User-Agent"].startswith("Mozilla/5.0"))
        self.assertTrue(client.ddg_headers["User-Agent"].startswith("Mozilla/5.0"))
        self.assertEqual(client.google_headers["Accept-Language"], "zh-TW,ja;q=0.8")
        self.assertEqual(client.ddg_headers["Accept-Language"], "zh-TW,ja;q=0.8")
        self.assertEqual(len(result.candidates), 1)

    def test_collect_candidates_marks_search_blocked_only_after_all_search_layers_are_empty(self):
        core = load_core()

        class Client:
            user_agent = "test-agent"

            def get_text(self, url, params=None, headers=None):
                if "google.com/search" in url:
                    return "<html></html>"
                if "duckduckgo.com/html" in url:
                    raise RuntimeError("ddg down")
                raise AssertionError(url)

            def google_search_urls(self, identity):
                return []

        result = core.collect_candidates(
            core.TrackIdentity("Song", "Artist"),
            Client(),
            lrclib_fetcher=lambda identity, client: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        self.assertIn("duckduckgo: ddg down", result.errors)
        self.assertIn("google: search_blocked", result.errors)
        self.assertEqual(result.search_engine, "none")
        self.assertEqual(result.search_url, "")

    def test_google_discovery_searches_translation_and_extracts_supported_pages(self):
        core = load_core()

        class Client:
            def __init__(self):
                self.search_params = None
                self.search_headers = None

            def get_text(self, url, params=None, headers=None):
                if "google.com/search" in url:
                    self.search_params = params
                    self.search_headers = headers
                    return '<a href="https://lyrics.example/song">result</a>'
                return "<p>君の声<br>你的聲音<br>夜を越える<br>跨越黑夜</p>"

        client = Client()
        urls, candidates = core.fetch_google_candidates(
            core.TrackIdentity("Song", "Artist"), client
        )
        self.assertIn("中文翻譯", client.search_params["q"])
        self.assertTrue(client.search_headers["User-Agent"].startswith("Mozilla/5.0"))
        self.assertEqual(client.search_headers["Accept-Language"], "zh-TW,ja;q=0.8")
        self.assertEqual(urls, ["https://lyrics.example/song"])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].lines[0], core.LyricPair("君の声", "你的聲音"))

    def test_google_discovery_rejects_page_for_a_different_song(self):
        core = load_core()

        class Client:
            def get_text(self, url, params=None, headers=None):
                if "google.com/search" in url:
                    return '<a href="https://lyrics.example/wrong-song">result</a>'
                return """
                <html><head><title>ワンダー - 倚水 | Lyrics</title></head><body><p>
                君の声<br>你的聲音<br>夜を越える<br>跨越黑夜
                </p></body></html>
                """

        urls, candidates = core.fetch_google_candidates(
            core.TrackIdentity("52Hz", "Islet"), Client()
        )
        self.assertEqual(urls, ["https://lyrics.example/wrong-song"])
        self.assertEqual(candidates, [])

    def test_page_identity_accepts_feat_suffix_and_slash_title_variants(self):
        core = load_core()
        self.assertTrue(
            core._page_matches_identity(
                core.TrackIdentity("白日夢 (feat. 倚水)", "Islet, 倚水"),
                "【倚水】白日夢【中、日、羅歌詞】 - 巴哈姆特",
            )
        )
        self.assertTrue(
            core._page_matches_identity(
                core.TrackIdentity("夢乃ゆき / Blue Skies", "MELLOW"),
                "【GalGame cover】スカイコード OP -『Blue Skies』By 夢乃ゆき",
            )
        )
        self.assertFalse(
            core._page_matches_identity(
                core.TrackIdentity("夢乃ゆき / Blue Skies", "MELLOW"),
                "Lenka - Blue Skies：歌詞+中文翻譯",
            )
        )

    def test_google_candidate_records_observed_page_title_instead_of_target_metadata(self):
        core = load_core()

        class Client:
            def get_text(self, url, params=None, headers=None):
                if "google.com/search" in url:
                    return '<a href="https://lyrics.example/song">result</a>'
                return """
                <html><head><meta property="og:title" content="Song - Lyrics Site"></head><body><p>
                君の声<br>你的聲音<br>夜を越える<br>跨越黑夜
                </p></body></html>
                """

        _, candidates = core.fetch_google_candidates(core.TrackIdentity("Song", "Artist"), Client())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].title, "Song - Lyrics Site")
        self.assertEqual(candidates[0].artist, "")

    def test_google_search_uses_isolated_headless_chrome_and_cleans_profile(self):
        core = load_core()
        calls = []
        profiles = []

        def runner(args, **kwargs):
            calls.append((args, kwargs))
            profile_arg = next(arg for arg in args if arg.startswith("--user-data-dir="))
            profile = Path(profile_arg.split("=", 1)[1])
            profiles.append(profile)
            self.assertTrue(profile.is_dir())
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=b'<a href="https://lyrics.example/song">result</a>',
            )

        client = core.HttpClient()
        with mock.patch.dict(core.os.environ, {"LYRICSEE_CHROME": r"C:\fake\chrome.exe"}, clear=False), mock.patch.object(
            core.Path, "is_file", return_value=True
        ), mock.patch.object(core.subprocess, "run", side_effect=runner):
            urls = client.google_search_urls(core.TrackIdentity("Song", "Artist"))

        self.assertEqual(urls, ["https://lyrics.example/song"])
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertIn("--headless=new", args)
        self.assertIn("--dump-dom", args)
        self.assertTrue(any(arg.startswith("--user-data-dir=") for arg in args))
        self.assertIs(kwargs["stdout"], subprocess.PIPE)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)
        self.assertEqual(kwargs["timeout"], 30)
        self.assertFalse(kwargs["check"])
        joined = "\n".join(args)
        for forbidden in (
            "--remote-debugging-port",
            "--remote-debugging-address",
            "chrome-use",
            r"Google\Chrome\User Data",
            "DCTEST",
        ):
            self.assertNotIn(forbidden, joined)
        self.assertEqual(len(profiles), 1)
        self.assertFalse(profiles[0].exists())

    def test_google_search_returns_empty_without_chrome(self):
        core = load_core()
        client = core.HttpClient()
        with mock.patch.dict(core.os.environ, {}, clear=True), mock.patch.object(
            core.Path, "is_file", return_value=False
        ), mock.patch.object(core.shutil, "which", return_value=None), mock.patch.object(
            core.subprocess, "run"
        ) as runner:
            self.assertEqual(client.google_search_urls(core.TrackIdentity("Song", "Artist")), [])
        runner.assert_not_called()

    def test_google_search_returns_empty_on_timeout_or_failed_chrome(self):
        core = load_core()
        client = core.HttpClient()
        identity = core.TrackIdentity("Song", "Artist")
        common = [
            mock.patch.dict(core.os.environ, {"LYRICSEE_CHROME": r"C:\fake\chrome.exe"}, clear=False),
            mock.patch.object(core.Path, "is_file", return_value=True),
        ]
        with common[0], common[1], mock.patch.object(
            core.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="chrome", timeout=30),
        ):
            self.assertEqual(client.google_search_urls(identity), [])
        with mock.patch.dict(core.os.environ, {"LYRICSEE_CHROME": r"C:\fake\chrome.exe"}, clear=False), mock.patch.object(
            core.Path, "is_file", return_value=True
        ), mock.patch.object(
            core.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(["chrome"], 1, stdout=b"ignored"),
        ):
            self.assertEqual(client.google_search_urls(identity), [])

    def test_collect_candidates_marks_google_search_blocked_when_no_urls(self):
        core = load_core()

        def lrclib(identity, client):
            raise RuntimeError("offline")

        result = core.collect_candidates(
            core.TrackIdentity("Song", "Artist"),
            object(),
            lrclib_fetcher=lrclib,
            google_fetcher=lambda identity, client: ([], []),
        )
        self.assertIn("google: search_blocked", result.errors)

    def test_http_client_decompresses_gzip_response_before_decoding(self):
        core = load_core()

        class Headers:
            def get(self, key, default=None):
                return "gzip" if key.casefold() == "content-encoding" else default

            def get_content_charset(self):
                return "utf-8"

        class Response:
            headers = Headers()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, limit):
                return gzip.compress("君の声  你的聲音".encode("utf-8"))

        with mock.patch.object(core, "urlopen", return_value=Response()):
            self.assertEqual(
                core.HttpClient().get_text("https://example.com"),
                "君の声  你的聲音",
            )

    def test_http_client_retries_transient_503_four_attempts(self):
        core = load_core()

        class Headers:
            def get(self, key, default=None):
                return default

            def get_content_charset(self):
                return "utf-8"

        class Response:
            headers = Headers()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, limit):
                return b"ok"

        failures = [HTTPError("https://example.com", 503, "busy", {}, None) for _ in range(2)]
        with mock.patch.object(core, "urlopen", side_effect=failures + [Response()]) as urlopen, mock.patch.object(
            core.time, "sleep"
        ) as sleep:
            self.assertEqual(core.HttpClient().get_text("https://example.com"), "ok")
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_args_list, [mock.call(1.0), mock.call(2.0)])

    def test_http_client_raises_after_four_transient_503_attempts(self):
        core = load_core()
        failures = [HTTPError("https://example.com", 503, "busy", {}, None) for _ in range(4)]
        with mock.patch.object(core, "urlopen", side_effect=failures) as urlopen, mock.patch.object(
            core.time, "sleep"
        ) as sleep, self.assertRaises(HTTPError):
            core.HttpClient().get_text("https://example.com")
        self.assertEqual(urlopen.call_count, 4)
        self.assertEqual(sleep.call_args_list, [mock.call(1.0), mock.call(2.0), mock.call(4.0)])

    def test_collect_candidates_marks_lrclib_http_503_unavailable(self):
        core = load_core()

        def lrclib(identity, client):
            raise HTTPError("https://lrclib.net/api/get/1", 503, "busy", {}, None)

        result = core.collect_candidates(
            core.TrackIdentity("Song", "Artist"),
            object(),
            lrclib_fetcher=lrclib,
            google_fetcher=lambda identity, client: ([], []),
        )
        self.assertTrue(result.lrclib_unavailable)
        self.assertIn("lrclib: source_unavailable (HTTP 503)", result.errors)

    def test_collect_candidates_does_not_mark_lrclib_lookup_error_unavailable(self):
        core = load_core()
        result = core.collect_candidates(
            core.TrackIdentity("Song", "Artist"),
            object(),
            lrclib_fetcher=lambda identity, client: (_ for _ in ()).throw(LookupError("no match")),
            google_fetcher=lambda identity, client: ([], []),
        )
        self.assertFalse(result.lrclib_unavailable)
        self.assertIn("lrclib: no match", result.errors)

    def test_collect_candidates_attempts_google_even_when_structured_sources_succeed(self):
        core = load_core()
        calls = []

        def lrclib(identity, client):
            calls.append("lrclib")
            return core.LyricCandidate(
                source="lrclib",
                url="l",
                title=identity.title,
                artist=identity.artist,
                lines=[core.LyricPair("original")],
            )

        def google(identity, client):
            calls.append("google")
            return ["https://example.com"], []

        result = core.collect_candidates(
            core.TrackIdentity("Song", "Artist"),
            object(),
            lrclib_fetcher=lrclib,
            google_fetcher=google,
        )
        self.assertEqual(calls, ["lrclib", "google"])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.google_urls, ["https://example.com"])
        self.assertEqual(result.errors, [])

    def test_local_llm_translation_batches_twelve_and_preserves_order(self):
        core = load_core()
        calls = []

        def post_json(url, payload, timeout):
            calls.append((url, payload, timeout))
            user_payload = json.loads(payload["messages"][1]["content"])
            rows = [
                {"id": line_id, "text": f"譯文{line_id}"}
                for line_id in user_payload["translate_only_ids"]
            ]
            return {
                "choices": [
                    {"message": {"content": json.dumps({"translations": rows}, ensure_ascii=False)}}
                ]
            }

        pairs = core.translate_with_local_llm(
            [f"source {index}" for index in range(1, 14)],
            title="Song",
            endpoint="http://127.0.0.1:6969",
            model="local-model",
            post_json=post_json,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "http://127.0.0.1:6969/v1/chat/completions")
        self.assertNotIn("response_format", calls[0][1])
        self.assertEqual(pairs[0], core.LyricPair("source 1", "譯文L001"))
        self.assertEqual(pairs[-1], core.LyricPair("source 13", "譯文L013"))

    def test_local_llm_translation_rejects_missing_or_duplicate_ids(self):
        core = load_core()

        def post_json(url, payload, timeout):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"translations":['
                            '{"id":"L001","text":"甲"},'
                            '{"id":"L001","text":"乙"}]}'
                        }
                    }
                ]
            }

        with self.assertRaisesRegex(ValueError, "duplicate id L001"):
            core.translate_with_local_llm(
                ["one", "two"],
                title="Song",
                endpoint="http://127.0.0.1:6969",
                model="local-model",
                post_json=post_json,
            )

    def test_local_llm_translation_rejects_multiline_text(self):
        core = load_core()

        def post_json(url, payload, timeout):
            content = json.dumps(
                {"translations": [{"id": "L001", "text": "第一行\n第二行"}]},
                ensure_ascii=False,
            )
            return {"choices": [{"message": {"content": content}}]}

        with self.assertRaisesRegex(ValueError, "must stay on one line"):
            core.translate_with_local_llm(
                ["one"],
                title="Song",
                endpoint="http://127.0.0.1:6969",
                model="local-model",
                post_json=post_json,
            )

    def test_local_llm_translation_retries_one_invalid_batch(self):
        core = load_core()
        calls = 0

        def post_json(url, payload, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"choices": [{"message": {"content": '{"translations":[]}'}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"translations":[{"id":"L001","text":"譯文"}]}'
                        }
                    }
                ]
            }

        result = core.translate_with_local_llm(
            ["source"],
            title="Song",
            endpoint="http://127.0.0.1:6969",
            model="local-model",
            post_json=post_json,
        )
        self.assertEqual(calls, 2)
        self.assertEqual(result, [core.LyricPair("source", "譯文")])

    def test_model_json_accepts_identical_repeated_fenced_objects(self):
        core = load_core()
        object_text = '{"translations":[{"id":"L001","text":"譯文"}]}'
        content = f"```json\n{object_text}\n```\n```json\n{object_text}\n```"
        parsed = core._parse_model_json(
            {"choices": [{"message": {"content": content}}]}
        )
        self.assertEqual(parsed["translations"][0]["id"], "L001")

    def test_model_json_uses_last_fenced_object_after_self_correction(self):
        core = load_core()
        first = '{"translations":[{"id":"L001","text":"甲"}]}'
        second = '{"translations":[{"id":"L001","text":"乙"}]}'
        content = (
            f"```json\n{first}\n```\n"
            "I included an incorrect row and will correct it.\n"
            f"```json\n{second}\n```"
        )
        parsed = core._parse_model_json(
            {"choices": [{"message": {"content": content}}]}
        )
        self.assertEqual(parsed["translations"][0]["text"], "乙")

    def test_probe_audio_prefers_embedded_metadata(self):
        core = load_core()

        def runner(args, **kwargs):
            self.assertEqual(args[0], "ffprobe")
            payload = {
                "format": {
                    "duration": "209.453",
                    "tags": {"title": "影色舞", "artist": "MyGO!!!!!", "album": "迷跡波"},
                }
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

        identity = core.probe_audio(Path("wrong filename.flac"), runner=runner)
        self.assertEqual(identity.title, "影色舞")
        self.assertEqual(identity.artist, "MyGO!!!!!")
        self.assertEqual(identity.album, "迷跡波")
        self.assertAlmostEqual(identity.duration, 209.453)

if __name__ == "__main__":
    unittest.main()
