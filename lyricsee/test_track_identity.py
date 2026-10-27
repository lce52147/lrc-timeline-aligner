"""Identity from the file's tags and album folder: aliases, artist names, gate, queries, artist-page lookup,
the empty-WAV fallback and the short-track hint.  Synthetic names and markup only."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import autopick
import jlyric_source
import track_identity as ti
import utanet_source
import utaten_source
import web_search
from lyric_sites import fetch_site_candidate, pick_artists
from lyricsee_core import TrackIdentity, _page_matches_identity, probe_audio, title_match_level
from translation_posts import names_artist

FOLDER = "[180523] TVアニメ「多田くんは恋をしない」ED「ラブソング」-テレサ・ワーグナー(CV.石見舞菜香) FLAC+BK"


def love_song_identity() -> TrackIdentity:
    path = Path("D:/Music") / "多田くんは恋をしない" / FOLDER / "01 Love Song.flac"
    fields = ti.build_identity_fields(
        path, title="Love Song", artist="Teresa Wagner (CV: Manaka Iwami)", album="Love Song",
        albumartist="Teresa Wagner (CV: Manaka Iwami)",
    )
    return TrackIdentity(
        "Love Song", "Teresa Wagner (CV: Manaka Iwami)", "Love Song", 319.0,
        ("Love Song", "Love Song", FOLDER), albumartist="Teresa Wagner (CV: Manaka Iwami)", **fields,
    )


class SplitAndFolderTests(unittest.TestCase):
    def test_cv_credit_gives_character_and_voice_actor(self):
        self.assertEqual(ti.split_names("Teresa Wagner (CV: Manaka Iwami)"), ["Teresa Wagner", "Manaka Iwami"])
        self.assertEqual(ti.split_names("エンドオブザワールド(CV:青山 吉能)"), ["エンドオブザワールド", "青山 吉能"])
        self.assertEqual(ti.split_names("倉科明日香(CV-澤田なつ)"), ["倉科明日香", "澤田なつ"])

    def test_multi_artist_string_is_split(self):
        self.assertEqual(ti.split_names("鸣潮先约电台/Crywolf/Cxssidy"), ["鸣潮先约电台", "Crywolf", "Cxssidy"])
        self.assertEqual(ti.split_names("Ann、Bob, Cat"), ["Ann", "Bob", "Cat"])

    def test_folder_name_gives_work_song_and_artists(self):
        info = ti.folder_info(FOLDER)
        self.assertEqual(info["work_titles"], ["多田くんは恋をしない"])
        self.assertEqual(info["song_titles"], ["ラブソング"])
        self.assertEqual(info["artist_names"], ["テレサ・ワーグナー", "石見舞菜香"])

    def test_folder_with_op_theme_and_slash_artist(self):
        info = ti.folder_info("[250704] TVアニメ「雨と君と」OP「雨と」／鈴木真海子 FLAC+MP3")
        self.assertEqual((info["work_titles"], info["song_titles"], info["artist_names"]),
                         (["雨と君と"], ["雨と"], ["鈴木真海子"]))

    def test_disc_subfolder_takes_the_work_from_its_parent(self):
        path = Path("D:/Music/MYGO/[240417] TVアニメ「Some Show!!」オリジナルサウンドトラック FLAC/Original Soundtrack Vol.1/30. Song.flac")
        self.assertEqual(ti.folder_context(path)["work_titles"], ["Some Show!!"])


class AliasTests(unittest.TestCase):
    def test_folder_song_title_becomes_an_alias(self):
        self.assertEqual(love_song_identity().aliases, ("Love Song", "ラブソング"))

    def test_stem_loses_artist_prefix_and_suffix(self):
        self.assertEqual(ti.title_aliases("", "Islet — 52Hz", ["Islet"], []), ["52Hz"])
        self.assertEqual(ti.title_aliases("", "ツユ - 雨宿り", ["ツユ"], []), ["雨宿り"])
        self.assertEqual(
            ti.title_aliases("", "Song Name - Band A,Singer B", ["Band A", "Singer B"], []), ["Song Name"]
        )

    def test_title_with_artist_word_is_kept_whole(self):
        self.assertEqual(ti.title_aliases("Lia's Song", "Lia's Song", ["Lia"], []), ["Lia's Song"])

    def test_bracketed_reading_is_an_alias(self):
        aliases = ti.title_aliases("直到下次再见（Dasvidaniya）", "直到下次再见（Dasvidaniya）", [], [])
        self.assertIn("直到下次再见", aliases)
        self.assertIn("Dasvidaniya", aliases)

    def test_feat_tail_is_not_an_alias(self):
        aliases = ti.title_aliases("白日夢 (feat. 倚水)", "白日夢 (feat. 倚水)", [], [])
        self.assertEqual(aliases, ["白日夢 (feat. 倚水)", "白日夢"])

    def test_aliases_widen_title_matching(self):
        identity = love_song_identity()
        self.assertTrue(_page_matches_identity(identity, "ラブソング 歌詞 - 誰か"))
        self.assertEqual(title_match_level(identity, "ラブソング"), 2)
        bare = TrackIdentity("Love Song", "Teresa Wagner")
        self.assertFalse(_page_matches_identity(bare, "ラブソング 歌詞 - 誰か"))


class GateTests(unittest.TestCase):
    def test_folder_spelled_names_confirm_a_page(self):
        identity = love_song_identity()
        self.assertTrue(names_artist(identity, "ラブソング 歌詞 テレサ・ワーグナー"))
        self.assertTrue(names_artist(identity, "石見舞菜香さんが歌う"))
        self.assertTrue(names_artist(identity, "TVアニメ『多田くんは恋をしない』エンディング"))

    def test_same_title_by_another_artist_is_not_confirmed(self):
        identity = love_song_identity()
        self.assertFalse(names_artist(identity, "ラブソング 歌詞 - 別のバンド 作詞 誰か アルバム 何か"))
        self.assertEqual(ti.identity_evidence(identity, "ラブソング 歌詞 - 別のバンド"), [])

    def test_rip_words_never_confirm(self):
        identity = TrackIdentity("流転", "Islet", "ASTER", 85.0, ("流転", "ASTER", "Islet — ASTER [WEB-FLAC]"))
        tokens = ti.gate_tokens(identity)
        self.assertNotIn("flac", tokens)
        self.assertNotIn("web", tokens)
        self.assertFalse(names_artist(identity, "free FLAC download of 流転"))
        self.assertTrue(names_artist(identity, "Isletの流転"))

    def test_the_songs_own_title_is_not_evidence(self):
        identity = love_song_identity()
        self.assertNotIn("lovesong", ti.gate_tokens(identity))
        self.assertNotIn("ラブソング", ti.gate_tokens(identity))

    def test_work_title_confirms(self):
        identity = TrackIdentity(
            "人間になりたいうた", "CRYCHIC", "Soundtrack Vol.1", 140.0, (),
            work_titles=("BanG Dream! It's MyGO!!!!!",),
        )
        self.assertEqual(ti.identity_evidence(identity, "別の作品の挿入歌"), [])
        self.assertIn("bangdreamitsmygo", ti.identity_evidence(identity, "BanG Dream! It's MyGO!!!!! 挿入歌"))


class QueryTests(unittest.TestCase):
    def test_plain_identity_keeps_the_old_queries(self):
        identity = TrackIdentity("Song", "Artist")
        self.assertEqual(web_search.identity_query_variants(identity), web_search.query_variants("Song", "Artist"))

    def test_queries_use_aliases_and_every_name(self):
        queries = web_search.identity_query_variants(love_song_identity())
        for expected in (
            '"Love Song" Teresa Wagner 歌詞',
            '"ラブソング" テレサ・ワーグナー 歌詞',
            '"ラブソング" 石見舞菜香 歌詞',
            '"Love Song" Manaka Iwami 歌詞',
        ):
            self.assertIn(expected, queries)
        self.assertEqual(len(queries), len(set(queries)))
        self.assertLessEqual(len(queries), web_search.MAX_IDENTITY_QUERIES)
        self.assertIn("site:uta-net.com ラブソング", queries)
        self.assertTrue(queries.index('"ラブソング" テレサ・ワーグナー 歌詞') < queries.index("site:uta-net.com Love Song"))

    def test_album_query_and_chinese_query(self):
        identity = TrackIdentity(
            "若能触及群星", "鸣潮先约电台、开心蛙蛙", "Some Album", 186.0, (),
            aliases=("若能触及群星", "Unwritten"), artist_names=("鸣潮先约电台", "开心蛙蛙"),
        )
        queries = web_search.identity_query_variants(identity)
        self.assertIn("若能触及群星 Some Album 歌詞", queries)
        self.assertIn("若能触及群星 歌词", queries)
        japanese = TrackIdentity("夜明けの片隅で", "Yuki", "A", 200.0, (), aliases=("夜明けの片隅で",), artist_names=("Yuki",))
        self.assertNotIn("夜明けの片隅で 歌词", web_search.identity_query_variants(japanese))

    def test_short_track_gets_a_short_query_list(self):
        names = ("Islet",)
        short = TrackIdentity("52Hz", "Islet", "CYANIDE", 85.0, (), aliases=("52Hz",), artist_names=names)
        long_ = short._replace(duration=200.0)
        self.assertEqual(len(web_search.identity_query_variants(short)), web_search.SHORT_TRACK_QUERIES)
        self.assertGreater(len(web_search.identity_query_variants(long_)), web_search.SHORT_TRACK_QUERIES)


class ProbeTests(unittest.TestCase):
    def test_probe_reads_album_artist_composer_and_lyricist(self):
        payload = {"format": {"duration": "85.0", "tags": {
            "TITLE": "雨宿り", "ARTIST": "ツユ", "album_artist": "ツユ", "ALBUM": "アンダーメンタリティ",
            "composer": "miro", "LYRICIST": "ぷす",
        }}}
        runner = lambda *a, **k: subprocess.CompletedProcess(a[0], 0, json.dumps(payload), "")
        path = Path("D:/Music/TUYU/ツユ – アンダーメンタリティ (2023.06.21)[Hi-Res FLAC]/07. ツユ - 雨宿り.flac")
        identity = probe_audio(path, runner=runner)
        self.assertEqual((identity.albumartist, identity.composer, identity.lyricist), ("ツユ", "miro", "ぷす"))
        self.assertEqual(identity.aliases, ("雨宿り",))
        self.assertEqual(identity.artist_names, ("ツユ",))



UTANET_ARTISTS = (
    '<table><tbody>'
    '<tr><td><a class="d-block" href="/artist/111/"><span class="fw-bold">ミトカツ</span><br><span>歌詞：3</span></a></td></tr>'
    '<tr><td><a class="d-block" href="/artist/222/"><span class="fw-bold">テレサ・ワーグナー</span><br><span>歌詞：1</span></a></td></tr>'
    '</tbody></table><a href="/artist/999/">サイドバー</a>'
)
UTANET_ARTIST_PAGE = (
    '<table><tbody><tr><td><a href="/song/5001/"><span class="fw-bold songlist-title">ラブソング</span></a></td>'
    '<td><a href="/artist/222/">テレサ・ワーグナー</a></td><td><a href="/lyricist/1/">作詞者</a></td>'
    '<td><a href="/composer/2/">作曲者</a></td><td></td><td><span class="pc-utaidashi">はじまりのことば</span></td></tr>'
    '<tr><td><a href="/song/5002/"><span class="fw-bold songlist-title">別の曲</span></a></td>'
    '<td><a href="/artist/222/">テレサ・ワーグナー</a></td></tr></tbody></table>'
)
UTANET_SONG = (
    '<html><head><meta name="description" content="テレサ・ワーグナーの「ラブソング」歌詞ページです。"></head>'
    '<body><div id="kashi_area">一行目の言葉<br>二行目の言葉<br>三行目の言葉</div></body></html>'
)
JLYRIC_ARTISTS = (
    '<div class="bdy"><p class="mid"><a class="artist" href="/artist/a111/" title="テレサ・ワーグナー 歌詞リスト">'
    'テレサ・ワーグナー</a></p><p class="sml">収録数：1曲</p></div>'
)
JLYRIC_ARTIST_PAGE = (
    '<div id="ly1" class="bdy"><input type="hidden" value="1"><p class="ttl"><a href="/artist/a111/l0001.html" '
    'title="テレサ・ワーグナー ラブソング 歌詞">ラブソング</a></p><p class="sml">作詞：甲 作曲：乙</p>'
    '<p class="sml">歌詞：はじまりのことば ...</p></div>'
)


class SiteClient:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.urls: list[str] = []

    def get_text(self, url, params=None, headers=None, retries=1, timeout=20):
        self.urls.append(url)
        for key, body in self.pages.items():
            if key in url:
                return body
        return ""


class ArtistPageTests(unittest.TestCase):
    def test_pick_artists_prefers_the_exact_name(self):
        rows = [("1", "ミトカツ"), ("2", "テレサ"), ("3", "テレサ・ワーグナー")]
        self.assertEqual(pick_artists(rows, "テレサ"), ["2", "3"])
        self.assertEqual(pick_artists(rows, "ミト"), [])

    def test_utanet_artist_search_and_page_parsers(self):
        self.assertEqual(utanet_source.parse_utanet_artists(UTANET_ARTISTS),
                         [("111", "ミトカツ"), ("222", "テレサ・ワーグナー")])
        titles = [hit.title for hit in utanet_source.parse_utanet_search(UTANET_ARTIST_PAGE)]
        self.assertEqual(titles, ["ラブソング", "別の曲"])

    def test_jlyric_artist_search_and_page_parsers(self):
        self.assertEqual(jlyric_source.parse_jlyric_artists(JLYRIC_ARTISTS), [("/artist/a111", "テレサ・ワーグナー")])
        hits = jlyric_source.parse_jlyric_artist_page(JLYRIC_ARTIST_PAGE, "テレサ・ワーグナー")
        self.assertEqual((hits[0].title, hits[0].artist, hits[0].lyricist, hits[0].composer, hits[0].url),
                         ("ラブソング", "テレサ・ワーグナー", "甲", "乙", "https://j-lyric.net/artist/a111/l0001.html"))

    def test_utaten_artist_url_carries_artist_and_title(self):
        url = utaten_source.utaten_artist_search_url("テレサ", "ラブソング")
        self.assertIn("artist_name=", url)
        self.assertIn("title=", url)

    def test_artist_page_finds_the_song_the_title_search_missed(self):
        identity = love_song_identity()
        client = SiteClient({
            "target=so": "<html></html>",              # no row for any title keyword
            "target=art": UTANET_ARTISTS,
            "/artist/222/": UTANET_ARTIST_PAGE,
            "/song/5001/": UTANET_SONG,
        })
        candidate = fetch_site_candidate(utanet_source.UTANET, identity, client)
        self.assertEqual(candidate.url, "https://www.uta-net.com/song/5001/")
        self.assertEqual(len(candidate.lines), 3)
        self.assertTrue(any("/artist/222/" in url for url in client.urls))

    def test_artist_page_row_with_wrong_title_is_rejected(self):
        identity = love_song_identity()._replace(aliases=("Love Song",))
        client = SiteClient({"target=so": "<html></html>", "target=art": UTANET_ARTISTS,
                             "/artist/222/": UTANET_ARTIST_PAGE.replace("ラブソング", "違う題")})
        with self.assertRaises(LookupError):
            fetch_site_candidate(utanet_source.UTANET, identity, client)

    def test_jlyric_artist_lookup_end_to_end(self):
        identity = love_song_identity()
        song = (
            '<meta name="description" content="テレサ・ワーグナーの「ラブソング」歌詞ページです。">'
            '<p id="Lyric">一行目の言葉<br>二行目の言葉</p>'
        )
        client = SiteClient({"ca=1": JLYRIC_ARTISTS, "ct=1": "<html></html>", "/artist/a111/l0001.html": song,
                             "/artist/a111/": JLYRIC_ARTIST_PAGE})
        candidate = fetch_site_candidate(jlyric_source.JLYRIC, identity, client)
        self.assertEqual(candidate.url, "https://j-lyric.net/artist/a111/l0001.html")


class DecodeFallbackTests(unittest.TestCase):
    """ffmpeg exits 0 with a header-only WAV for a FLAC whose first frames are damaged; libsndfile resyncs."""

    def _run_factory(self, calls: list):
        def run(args, **kwargs):
            calls.append(list(args))
            wav = Path(args[-1])
            source = Path(args[args.index("-i") + 1])
            wav.write_bytes(b"RIFF" + b"\0" * 40 if source.suffix == ".flac" else b"RIFF" + b"\0" * 5000)
            return subprocess.CompletedProcess(args, 0, b"", b"")
        return run

    def test_good_decode_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            calls: list = []
            run = lambda args, **kw: (Path(args[-1]).write_bytes(b"\0" * 4000), subprocess.CompletedProcess(args, 0))[1]
            audio = tmp / "a.flac"
            audio.write_bytes(b"fLaC")
            self.assertEqual(autopick.decode_to_wav16k(audio, tmp / "o.wav", tmp / "n.wav", run), "ffmpeg")

    @unittest.skipUnless(__import__("importlib").util.find_spec("soundfile"), "soundfile not installed")
    def test_empty_wav_retries_through_libsndfile(self):
        import numpy as np
        import soundfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            audio = tmp / "01 Love Song ｢ラブソング｣ (CV.声優) [x].flac"   # the odd path of the original failure
            soundfile.write(str(audio), (np.sin(np.arange(22050) / 7) * 9000).astype("int16"), 44100, format="FLAC")
            calls: list = []
            used = autopick.decode_to_wav16k(audio, tmp / "o.wav", tmp / "n.wav", self._run_factory(calls))
            self.assertEqual(used, "libsndfile")
            self.assertEqual(len(calls), 2)                       # ffmpeg on the FLAC, then ffmpeg on the native WAV
            self.assertTrue(calls[1][calls[1].index("-i") + 1].endswith("n.wav"))
            self.assertGreaterEqual((tmp / "o.wav").stat().st_size, autopick.MIN_WAV_BYTES)
            self.assertFalse((tmp / "n.wav").exists())            # temp decode removed; source untouched

    def test_undecodable_audio_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            audio = tmp / "bad.flac"
            audio.write_bytes(b"not audio at all")
            with self.assertRaises(RuntimeError) as ctx:
                autopick.decode_to_wav16k(audio, tmp / "o.wav", tmp / "n.wav", self._run_factory([]))
            self.assertIn("audio_undecodable", str(ctx.exception))


class InstrumentalTests(unittest.TestCase):
    def entry(self, score, snippet=False):
        return {"snippet": snippet, "asr": {"score": score}}

    def test_short_track_without_supported_candidate_is_flagged(self):
        evidence = autopick.instrumental_evidence(66.6, [self.entry(0.02)], "")
        self.assertEqual(evidence["duration"], 66.6)
        self.assertEqual(evidence["transcript_chars"], 0)
        self.assertIn("not proof", evidence["note"])

    def test_long_track_or_supported_candidate_is_not_flagged(self):
        self.assertIsNone(autopick.instrumental_evidence(200.0, [], ""))
        self.assertIsNone(autopick.instrumental_evidence(66.6, [self.entry(0.5)], "x"))
        self.assertIsNone(autopick.instrumental_evidence(None, [], ""))

    def test_no_candidates_on_a_short_track_is_a_review_with_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = autopick.autopick_song(Path(tmp), "52Hz", "", 85.0)
        self.assertEqual((result["decision"], result["reason"]), ("review", "likely_instrumental_or_interlude"))
        self.assertIn("likely_instrumental_or_interlude", result)

    def test_no_candidates_on_a_long_track_stays_a_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = autopick.autopick_song(Path(tmp), "Song", "", 240.0)
        self.assertEqual((result["decision"], result["reason"]), ("skip", "no_candidates"))


if __name__ == "__main__":
    unittest.main()
