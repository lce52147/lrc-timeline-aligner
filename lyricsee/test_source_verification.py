"""uta-net primary source, notation-aware comparison and cross-source verification."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

import lyricsee as cli
import lyricsee_core as core
import source_policy
from lyric_compare import classify_segment, compare_lyrics, canonical_text, is_filler
from utanet_source import (
    UtanetAmbiguous, fetch_utanet_candidate, parse_utanet_search, parse_utanet_song,
    resolve_ambiguous_hits, utanet_search_url,
)


def _row(song_id, title, artist, opening="歌い出し"):
    return (
        '<tr class="border-bottom"><td class="sp-w-100 pt-0 pt-lg-2">'
        f'<a href="/song/{song_id}/" class="py-2 py-lg-0"><span class="fw-bold songlist-title">{title}</span>'
        '<span class="d-inline d-lg-none ps-1 icon-new-senko-crown"><img src="crown.png" alt="GOLD LYLIC"></span>'
        f'<span class="d-block d-lg-none utaidashi">{artist}</span></a>'
        '<a class="d-none d-lg-inline pe-1 ps-1 icon-new-senko-crown" href="https://www.uta-net.com/ranking/total/"><img src="crown.png"></a></td>'
        f'<td class="sp-none fw-bold"><a href="/artist/1/">{artist}</a></td>'
        '<td class="sp-none fw-bold"><a href="/lyricist/2/">作詞者</a></td>'
        '<td class="sp-none fw-bold"><a href="/composer/3/">作曲者</a></td>'
        '<td class="sp-none fw-bold"><a href="/arranger/4/">編曲者</a></td>'
        f'<td class="sp-none fw-bold"><span class="d-block pc-utaidashi">{opening}</span></td></tr>'
    )


def search_page(rows):
    return (
        '<html><head><title>検索 歌詞リスト - 歌ネット</title></head><body>'
        '<div class="row songlist-table-block p-3"><h2><span class="search_keyword">x</span></h2>'
        '<table class="table table-sm table-borderless songlist-table th-col-5"><thead><tr><th>曲名</th></tr></thead>'
        '<tbody class="songlist-table-body">' + "".join(rows) + '</tbody></table></div>'
        '<ol><li><div class="songdata"><a href="/song/399983/">シンデレラ</a><br><a href="/artist/18526/">Mrs. GREEN APPLE</a></div></li></ol>'
        '</body></html>'
    )


def song_page(title, artist, lines):
    body = "<br />".join(lines)
    return (
        '<!DOCTYPE html><html lang="ja"><head><meta charset="utf-8" />'
        f'<title>{artist} {title} 歌詞 - 歌ネット</title>'
        f'<meta property="og:title" content="「{title}/{artist}」の歌詞 って「イイネ！」" />'
        f'<meta name="description" content="{artist}の「{title}」歌詞ページです。作詞:誰か,作曲:誰か。 歌ネットは無料の歌詞検索サービスです。" />'
        '</head><body><div class="row pt-4 kashi" id="kashi-box"><div class="col-12" itemprop="lyrics">'
        f'<div id="kashi_area" itemprop="text">{body}</div>'
        '<div class="d-none d-md-block text-end"><a href="/renew/myutanet/song_book.php?mode=add&amp;ID=1">この歌詞をマイ歌ネットに登録&nbsp;&gt;</a><br />'
        '<a href="/renew/myutanet/artist.php?mode=add">このアーティストをマイ歌ネットに登録&nbsp;&gt;</a><br /></div>'
        '</div></div><p>コトバのキモチ投稿フォーム</p></body></html>'
    )


UTANET_LINES = [
    "願った　勇敢な選択の", "未来を今照らしてく", "", "利己的な感情　ぶつけあっちゃ(まだ)見えないね",
    "逃げて来た分の距離を埋める(ほら)旅に出よう", "", "最果てなんて　見られなくても",
    "命を燃やして今を生きていくだけだから", "後悔は要らない　手を取り合って(暗い過去だって)",
    "受け止めて進んでいく", "示された答え選ぶより", "新しい夢を目指すから(ずっと)",
    "眩しい道を歩いていこう", "二人しかない可能性を(ほら)見つけよう", "不甲斐なくて泣いた　冷たい風の夜",
]
LRCLIB_LINES = [
    "願った 勇敢な選択の", "未来を今照らしてく", "利己的な感情 ぶつけあっちゃ 見えないね",
    "逃げて来た分の距離を埋める 旅に出よう", "最果てなんて 見られなくても",
    "命を燃やして今を生きていくだけだから", "後悔はいらない 手を取り合って 暗い過去だって",
    "受け止めて進んでいく", "示された答え選ぶより", "新しい夢を目指すから ずっと",
    "眩しい道を歩いていこう", "ふたりしかない可能性を 見つけよう", "不甲斐なくて泣いた 冷たい風の夜",
]


class FakeClient:
    """Routes URLs to canned documents; records every request."""

    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def get_text(self, url, params=None, headers=None, retries=1, timeout=20):
        self.requests.append(url)
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise HTTPError(url, 404, "Not Found", {}, None)


class LyricCompareTests(unittest.TestCase):
    def test_spelling_differences_are_notation_not_errors(self):
        self.assertEqual(classify_segment("二人", "ふたり"), "notation:kana_kanji")
        self.assertEqual(classify_segment("行", "ゆ"), "notation:kana_kanji")
        self.assertEqual(classify_segment("1", "ひと"), "notation:kana_kanji")
        self.assertEqual(classify_segment("ちゅ", "chu"), "notation:transliteration")
        self.assertEqual(classify_segment("2", "二"), "notation:numeral")
        self.assertEqual(classify_segment("onetwothree", "123"), "notation:numeral")
        self.assertEqual(classify_segment("ahahah", ""), "notation:filler")
        self.assertEqual(canonical_text("ヴァイオリン　人々！ Ō"), canonical_text("ゔぁいおりん人人 o"))
        self.assertEqual(canonical_text("繫いだ手"), canonical_text("繋いだ手"))
        self.assertEqual(canonical_text("あぁ"), canonical_text("ああ"))

    def test_real_word_differences_stay_real(self):
        self.assertEqual(classify_segment("黒", "暗"), "real")
        self.assertEqual(classify_segment("後悔", "不甲斐"), "real")
        self.assertEqual(classify_segment("は", "ば"), "real")
        self.assertEqual(classify_segment("を", "も"), "real")
        # kana reading that cannot be the kanji spelling (に must survive)
        self.assertEqual(classify_segment("此処に", "ここで"), "real")
        self.assertEqual(classify_segment("答えどーるの眼", "かげさま"), "real")
        # a dropped particle is not a vocal fill
        self.assertEqual(classify_segment("は", ""), "real")
        self.assertFalse(is_filler("ほら"))
        self.assertTrue(is_filler("ららら"))
        self.assertTrue(is_filler("yeahyeahyeah"))

    def test_alive_mishearing_is_a_word_diff_with_line_numbers(self):
        wrong = list(LRCLIB_LINES)
        wrong[6] = "後悔はいらない 手を取り合って 黒い過去だって"
        wrong[12] = "後悔なくて泣いた 冷たい風の夜"
        diff = compare_lyrics(UTANET_LINES, wrong)
        self.assertEqual(diff.word_diffs, 2)
        self.assertFalse(diff.agrees())
        kinds = [(item.kind, item.a_line, item.a_text, item.b_text) for item in diff.real_diffs]
        self.assertIn(("word", 9, "暗", "黒"), kinds)
        self.assertIn(("word", 15, "不甲斐", "後悔"), kinds)
        # Everything else (parentheses, kana spelling, spacing) is notation.
        self.assertTrue(all(item.kind.startswith("notation:") for item in diff.notation_diffs))

    def test_parenthesised_backing_vocals_and_fills_do_not_break_agreement(self):
        diff = compare_lyrics(UTANET_LINES, LRCLIB_LINES + ["ラララ ラララ", "Ah ah"])
        self.assertEqual(diff.word_diffs, 0)
        self.assertEqual(diff.real_diffs, [])
        self.assertTrue(diff.agrees())
        self.assertGreaterEqual(diff.coverage_a, 0.99)
        self.assertEqual(len(diff.pairs), 13)
        notation_kinds = {item.kind for item in diff.notation_diffs}
        self.assertIn("notation:filler", notation_kinds)
        self.assertTrue(any("parenthesised" in kind for kind in notation_kinds))
        self.assertTrue(any("kana_kanji" in kind for kind in notation_kinds))

    def test_missing_lyric_line_is_real_but_tolerated_up_to_coverage(self):
        shorter = LRCLIB_LINES[:-1]
        diff = compare_lyrics(UTANET_LINES, shorter)
        self.assertEqual(diff.word_diffs, 0)
        self.assertEqual([item.kind for item in diff.real_diffs], ["missing_in_b"])
        self.assertEqual(diff.real_diffs[0].a_line, 15)
        self.assertTrue(diff.agrees())           # 12 of 13 lines still covered
        half = LRCLIB_LINES[:6]
        self.assertFalse(compare_lyrics(UTANET_LINES, half).agrees())

    def test_different_song_never_agrees(self):
        other = [f"全く別の歌の{i}行目です" for i in range(13)]
        diff = compare_lyrics(UTANET_LINES, other)
        self.assertFalse(diff.agrees())
        self.assertLess(diff.similarity, 0.5)
        self.assertEqual(diff.matched_lines, 0)


class UtanetParserTests(unittest.TestCase):
    def test_search_rows_parse_title_artist_and_opening(self):
        page = search_page([
            _row("242417", "ふゆびより", "佐々木恵梨", "額に感じる澄んだ空気　吐く息が弾む"),
            _row("1", "I Call Your Name", "Shogo Hamada &amp; The J.S. Inspirations"),
        ])
        hits = parse_utanet_search(page)
        self.assertEqual([(h.song_id, h.title, h.artist) for h in hits], [
            ("242417", "ふゆびより", "佐々木恵梨"),
            ("1", "I Call Your Name", "Shogo Hamada & The J.S. Inspirations"),
        ])
        self.assertEqual(hits[0].url, "https://www.uta-net.com/song/242417/")
        self.assertEqual(hits[0].lyricist, "作詞者")
        self.assertTrue(hits[0].opening.startswith("額に感じる"))
        # ranking links outside the table are not hits
        self.assertNotIn("399983", [h.song_id for h in hits])
        self.assertEqual(
            utanet_search_url("ふゆびより"),
            "https://www.uta-net.com/search/?target=so&type=in&keyword=%E3%81%B5%E3%82%86%E3%81%B3%E3%82%88%E3%82%8A",
        )

    def test_song_page_keeps_kashi_area_lines_only(self):
        song = parse_utanet_song(song_page("ALIVE", "ClariS", UTANET_LINES))
        self.assertEqual((song.title, song.artist), ("ALIVE", "ClariS"))
        self.assertEqual(song.lines[0], "願った 勇敢な選択の")
        self.assertEqual(len(song.lines), 13)
        self.assertFalse(any("歌ネット" in line or "コトバ" in line for line in song.lines))
        self.assertIn("後悔は要らない 手を取り合って(暗い過去だって)", song.lines)

    def test_song_page_without_kashi_area_raises(self):
        with self.assertRaisesRegex(LookupError, "kashi_area"):
            parse_utanet_song("<html><head><title>x</title></head><body><div>nothing</div></body></html>")


class UtanetFetchTests(unittest.TestCase):
    def test_similar_title_of_another_artist_is_rejected(self):
        client = FakeClient({
            "keyword=Call+you": search_page([_row("387865", "I Call Your Name", "Shogo Hamada &amp; The J.S. Inspirations")]),
        })
        identity = core.TrackIdentity("Call you", "山田タマル", duration=200.0)
        with self.assertRaisesRegex(LookupError, "I Call Your Name"):
            fetch_utanet_candidate(identity, client)
        self.assertFalse(any("/song/" in url for url in client.requests))

    def test_exact_title_and_artist_row_wins_over_lookalikes(self):
        client = FakeClient({
            "keyword=ALIVE": search_page([
                _row("10", "Stay Alive", "ClariS"),
                _row("11", "ALIVE", "Someone Else"),
                _row("321583", "ALIVE", "ClariS"),
            ]),
            "/song/321583/": song_page("ALIVE", "ClariS", UTANET_LINES),
        })
        identity = core.TrackIdentity("ALIVE", "ClariS", duration=240.0)
        candidate = fetch_utanet_candidate(identity, client)
        self.assertEqual(candidate.source, "uta-net")
        self.assertEqual(candidate.url, "https://www.uta-net.com/song/321583/")
        self.assertEqual(candidate.title, "ALIVE")
        self.assertEqual(len(candidate.lines), 13)
        self.assertTrue(candidate.meta["identity"]["strong"])
        self.assertEqual(candidate.meta["trust"], "publisher")
        self.assertEqual(candidate.search_engine, "uta-net")

    def test_slash_title_searches_japanese_part_first_and_folder_names_the_artist(self):
        client = FakeClient({
            "keyword=%E3%81%B5%E3%82%86%E3%81%B3%E3%82%88%E3%82%8A": search_page([_row("242417", "ふゆびより", "佐々木恵梨")]),
            "/song/242417/": song_page("ふゆびより", "佐々木恵梨", UTANET_LINES),
        })
        identity = core.TrackIdentity(
            "Fuyubiyori/ふゆびより", "Sasaki Eri", duration=230.0,
            hints=("ふゆびより", "", "[180124] TVアニメ「ゆるキャン△」ED「ふゆびより」-佐々木恵梨 FLAC"),
        )
        candidate = fetch_utanet_candidate(identity, client)
        self.assertEqual(candidate.url, "https://www.uta-net.com/song/242417/")
        self.assertEqual(client.requests[0], utanet_search_url("ふゆびより"))
        self.assertEqual(candidate.meta["identity"]["artist_level"], 1)
        self.assertTrue(candidate.meta["identity"]["strong"])

    def test_404_search_moves_to_the_next_keyword(self):
        client = FakeClient({
            "keyword=Rhapsody": search_page([_row("5", "Mas?uerade Rhapsody Re?uest", "Ave Mujica")]),
            "/song/5/": song_page("Mas?uerade Rhapsody Re?uest", "Ave Mujica", UTANET_LINES),
        })
        identity = core.TrackIdentity("Mas?uerade Rhapsody Re?uest", "Ave Mujica", duration=240.0)
        candidate = fetch_utanet_candidate(identity, client)
        self.assertEqual(candidate.url, "https://www.uta-net.com/song/5/")
        self.assertEqual(len([url for url in client.requests if "/search/" in url]), 2)

    def test_exact_title_without_artist_evidence_is_ambiguous_then_weak(self):
        client = FakeClient({
            "keyword=R.I.P.": search_page([_row("301240", "R.I.P.", "鋭児")]),
            "/song/301240/": song_page("R.I.P.", "鋭児", UTANET_LINES),
        })
        identity = core.TrackIdentity("R.I.P.", "3OH!3", duration=200.0)
        with self.assertRaises(UtanetAmbiguous) as caught:
            fetch_utanet_candidate(identity, client)
        self.assertEqual([hit.url for _, hit in caught.exception.hits], ["https://www.uta-net.com/song/301240/"])
        # Without any reference a single row is fetched, flagged weak.
        candidate = resolve_ambiguous_hits(identity, client, caught.exception.hits, None)
        self.assertFalse(candidate.meta["identity"]["strong"])
        self.assertEqual(candidate.meta["identity"]["artist_level"], 0)
        # A reference lyric of a different song rejects the row.
        english = [f"english line number {i} of another song" for i in range(12)]
        with self.assertRaisesRegex(LookupError, "do not match the reference"):
            resolve_ambiguous_hits(identity, client, caught.exception.hits, english)

    def test_reference_lyric_picks_among_several_exact_title_rows(self):
        other_opening = "全く別の歌の歌い出しです　別の言葉が続きます　別の言葉が続きます"
        konayuki_opening = "　".join(line for line in UTANET_LINES[:6] if line)
        client = FakeClient({
            "keyword=": search_page([
                _row("1", "粉雪", "浅沼晋太郎", other_opening),
                _row("2", "粉雪", "レミオロメン", konayuki_opening),
                _row("3", "粉雪", "INSPi", other_opening),
            ]),
            "/song/2/": song_page("粉雪", "レミオロメン", UTANET_LINES),
        })
        identity = core.TrackIdentity("KONAYUKI", "Remioromen", duration=325.0, hints=("粉雪",))
        with self.assertRaises(UtanetAmbiguous):
            fetch_utanet_candidate(identity, client)
        result = core.collect_candidates(
            identity, client,
            lrclib_fetcher=lambda i, c: core.LyricCandidate(
                source="lrclib", url="https://lrclib.net/api/get/9", title="KONAYUKI", artist="Remioromen",
                lines=[core.LyricPair(line) for line in LRCLIB_LINES]),
            google_fetcher=lambda i, c: ([], []),
        )
        self.assertEqual([c.source for c in result.candidates], ["uta-net", "lrclib"])
        self.assertEqual(result.candidates[0].url, "https://www.uta-net.com/song/2/")
        self.assertFalse(result.candidates[0].meta["identity"]["strong"])
        self.assertGreater(result.candidates[0].meta["identity"]["opening_matched_reference"], 0.6)
        self.assertFalse(any("/song/1/" in url or "/song/3/" in url for url in client.requests))
        chosen, evidence = source_policy.choose_verified_original(identity, result.candidates)
        self.assertEqual(chosen.source, "uta-net")
        self.assertEqual(evidence["verification_status"], "verified")
        self.assertEqual(evidence["original_verified_by"], "two_source_agreement")

    def test_several_exact_titles_without_artist_or_reference_are_rejected(self):
        client = FakeClient({
            "keyword=": search_page([_row("1", "顔", "くるり"), _row("2", "顔", "誰か")]),
        })
        identity = core.TrackIdentity("顔", "女王蜂")
        with self.assertRaises(UtanetAmbiguous) as caught:
            fetch_utanet_candidate(identity, client)
        with self.assertRaisesRegex(LookupError, "several rows carry this title"):
            resolve_ambiguous_hits(identity, client, caught.exception.hits, None)
        result = core.collect_candidates(
            identity, client,
            lrclib_fetcher=lambda i, c: (_ for _ in ()).throw(LookupError("no match")),
            google_fetcher=lambda i, c: ([], []),
        )
        self.assertEqual(result.candidates, [])
        self.assertTrue(any("several rows carry this title" in item for item in result.errors))

    def test_server_error_propagates_as_unavailable(self):
        client = FakeClient({"keyword=": HTTPError("u", 503, "busy", {}, None)})
        result = core.collect_candidates(
            core.TrackIdentity("Song", "Artist"), client,
            lrclib_fetcher=lambda i, c: (_ for _ in ()).throw(LookupError("no match")),
            google_fetcher=lambda i, c: ([], []),
        )
        self.assertTrue(result.utanet_unavailable)
        self.assertTrue(any(item.startswith("uta-net: source_unavailable") for item in result.errors))


class IdentityMatchTests(unittest.TestCase):
    def test_audit_wrong_song_rows_fail_the_title_or_artist_gate(self):
        cases = [
            (core.TrackIdentity("花がら", "tayori"), "花がらもようの便箋に", "シモンズ"),
            (core.TrackIdentity("Wishing", "レム(CV：水瀬いのり)"), "WISHING ON", "RUN＆GUN"),
            (core.TrackIdentity("シオン (feat. 倚水)", "Islet"), "オラシオンのテーマ ～共に歩こう～", "林明日香"),
            (core.TrackIdentity("swim (Cover)", "MyGO!!!!!"), "City Swimmer", "財津和夫"),
            (core.TrackIdentity("アイドル", "YOASOBI"), "全開！満開！アイドル道！", ""),
            (core.TrackIdentity("星になる (feat. 倚水)", "Islet"), "君が星になるまで", "AKB48"),
            (core.TrackIdentity("顔", "女王蜂"), "朝顔", "くるり"),
            (core.TrackIdentity("メメント", "Islet"), "恍惚のメメント", "SERRA"),
            (core.TrackIdentity("風のたより", "Islet"), "風のたより", "村下孝蔵"),
        ]
        for identity, title, artist in cases:
            with self.subTest(title=title):
                strong = core.title_match_level(identity, title) >= 2 and core.artist_match_level(identity, artist) >= 1
                self.assertFalse(strong)

    def test_spelling_variants_of_the_same_song_pass(self):
        cases = [
            (core.TrackIdentity("ぼうやの夢よ", "エミリア(CV：高橋李依)"), "ぼうやの夢よ", "エミリア(高橋李依)"),
            (core.TrackIdentity("可愛くてごめん", "HoneyWorks feat. ちゅーたん(CV-早見沙織)"), "可愛くてごめん feat. ちゅーたん(早見沙織)", "HoneyWorks"),
            (core.TrackIdentity("Release My Soul <feat.Aimee Blackschleger>", "Aimee Blackschleger"), "Release My Soul", "Aimee Blackschleger"),
            (core.TrackIdentity("春日影", "MyGO!!!!!"), "春日影 (MyGO!!!!! ver.)", "MyGO!!!!!"),
            (core.TrackIdentity("Mas?uerade Rhapsody Re?uest", "Ave Mujica"), "Mas？uerade Rhapsody Re？uest", "Ave Mujica"),
            (core.TrackIdentity("ふたりごと", "TRUE & 茅原実里"), "ふたりごと", "TRUE&茅原実里"),
            (core.TrackIdentity("たより", "上田 麗奈"), "たより", "上田麗奈"),
            (core.TrackIdentity("KONAYUKI", "Remioromen", hints=("粉雪", "", "Remioromen 粉雪")), "粉雪", "レミオロメン"),
        ]
        for identity, title, artist in cases:
            with self.subTest(title=title):
                self.assertGreaterEqual(core.title_match_level(identity, title), 2)
        self.assertEqual(core.artist_match_level(cases[0][0], "エミリア(高橋李依)"), 2)
        self.assertEqual(core.artist_match_level(cases[-1][0], "レミオロメン"), 0)
        self.assertEqual(core.artist_parts("麻枝准 X やなぎなぎ"), ["麻枝准", "やなぎなぎ"])

    def test_identity_hints_come_from_file_name_album_and_folder(self):
        hints = core.identity_hints(Path("/music/[180124] ED「ふゆびより」-佐々木恵梨 FLAC/01 ふゆびより.flac"), album="ふゆびより")
        self.assertEqual(hints[0], "ふゆびより")
        self.assertIn("[180124] ED「ふゆびより」-佐々木恵梨 FLAC", hints)
        self.assertEqual(core.strip_track_number("3-12. Lasting Moment"), "Lasting Moment")
        self.assertEqual(core.strip_track_number("001-ゼロセンチメートル"), "ゼロセンチメートル")
        self.assertEqual(core.strip_track_number("3OH!3"), "3OH!3")
        self.assertEqual(core.identity_from_filename(Path("01 HoneyWorks feat. ちゅーたん - 可愛くてごめん.flac")).artist, "HoneyWorks feat. ちゅーたん")
        self.assertEqual(core.identity_from_filename(Path("01. ALIVE.flac")).title, "ALIVE")


def _utanet(lines=UTANET_LINES, *, strong=True, artist_level=2):
    parsed = parse_utanet_song(song_page("ALIVE", "ClariS", lines)).lines
    return core.LyricCandidate(
        source="uta-net", url="https://www.uta-net.com/song/321583/", title="ALIVE", artist="ClariS",
        lines=[core.LyricPair(line) for line in parsed],
        meta={"trust": "publisher", "identity": {"title_level": 3, "artist_level": artist_level, "strong": strong}},
    )


def _lrclib(lines=LRCLIB_LINES):
    return core.LyricCandidate(
        source="lrclib", url="https://lrclib.net/api/get/22231409", title="ALIVE", artist="ClariS",
        lines=[core.LyricPair(line) for line in lines],
    )


class VerificationPolicyTests(unittest.TestCase):
    identity = core.TrackIdentity("ALIVE", "ClariS", duration=240.0)

    def test_publisher_and_lrclib_agreeing_is_verified_by_two_sources(self):
        chosen, evidence = source_policy.choose_verified_original(self.identity, [_lrclib(), _utanet()])
        self.assertEqual(chosen.source, "uta-net")
        self.assertEqual(evidence["verification_status"], "verified")
        self.assertEqual(evidence["original_verified_by"], "two_source_agreement")
        self.assertEqual(evidence["sources_used"], ["uta-net", "lrclib"])
        self.assertEqual(evidence["word_diff_line_count"], 0)
        self.assertTrue(evidence["artist_confirmed"])
        self.assertEqual(evidence["cross_sources"][0]["source"], "lrclib")
        self.assertTrue(evidence["cross_sources"][0]["agrees"])

    def test_publisher_alone_with_real_word_difference_is_verified_by_trust_and_flags_the_line(self):
        wrong = list(LRCLIB_LINES)
        wrong[6] = "後悔はいらない 手を取り合って 黒い過去だって"
        chosen, evidence = source_policy.choose_verified_original(self.identity, [_lrclib(wrong), _utanet()])
        self.assertEqual(chosen.source, "uta-net")
        self.assertEqual(evidence["verification_status"], "verified")
        self.assertEqual(evidence["original_verified_by"], "trusted_source_sanity")
        self.assertEqual(evidence["sources_used"], ["uta-net"])
        self.assertEqual(evidence["word_diff_line_count"], 1)
        self.assertEqual(evidence["diff_line_count"], 1)
        diff_lines = evidence["cross_sources"][0]["diff_lines"]
        # line numbers count the output lines (blank stanza breaks are not written)
        self.assertEqual((diff_lines[0]["line"], diff_lines[0]["text"], diff_lines[0]["other_text"]), (7, "暗", "黒"))
        self.assertFalse(evidence["cross_source_conflict"])

    def test_junk_lrclib_fails_content_check_and_does_not_block_publisher(self):
        junk = ["止まる", "ここで", "はあ", "ああ", "止まる", "ここで", "はあ", "ああ", "止まる", "ここで", "La la la", "はあ", "ああ"]
        chosen, evidence = source_policy.choose_verified_original(self.identity, [_lrclib(junk), _utanet()])
        self.assertEqual(chosen.source, "uta-net")
        self.assertEqual(evidence["verification_status"], "verified")
        considered = {item["source"]: item for item in evidence["sources_considered"]}
        self.assertFalse(considered["lrclib"]["admitted"])
        self.assertIn("content check failed", considered["lrclib"]["reason"])
        self.assertEqual(evidence["cross_sources"], [])

    def test_weak_publisher_row_is_verified_only_through_agreement(self):
        chosen, evidence = source_policy.choose_verified_original(
            self.identity, [_lrclib(), _utanet(strong=False, artist_level=0)]
        )
        self.assertEqual(chosen.source, "uta-net")
        self.assertEqual(evidence["verification_status"], "verified")
        self.assertEqual(evidence["original_verified_by"], "two_source_agreement")

        other = [f"別の曲の{i}行目の歌詞です" for i in range(13)]
        chosen, evidence = source_policy.choose_verified_original(
            self.identity, [_lrclib(), _utanet(other, strong=False, artist_level=0)]
        )
        self.assertEqual(chosen.source, "lrclib")
        self.assertEqual(evidence["verification_status"], "unverified")
        self.assertEqual(evidence["original_verified_by"], "single_source")
        considered = {item["source"]: item for item in evidence["sources_considered"]}
        self.assertIn("artist not confirmed", considered["uta-net"]["reason"])

    def test_wrong_song_is_never_admitted(self):
        wrong = core.LyricCandidate(
            source="uta-net", url="https://www.uta-net.com/song/387865/", title="I Call Your Name",
            artist="Shogo Hamada", lines=[core.LyricPair(f"line {i} of another song") for i in range(20)],
            meta={"trust": "publisher", "identity": {"title_level": 1, "artist_level": 0, "strong": False}},
        )
        with self.assertRaisesRegex(LookupError, "No usable source.*title does not match"):
            source_policy.choose_verified_original(core.TrackIdentity("Call you", "山田タマル", duration=200.0), [wrong])

    def test_publisher_text_too_short_for_the_audio_is_not_trusted_alone(self):
        short = _utanet(UTANET_LINES[:9])   # 8 lines, ~90 characters
        identity = core.TrackIdentity("ALIVE", "ClariS", duration=600.0)
        with self.assertRaisesRegex(LookupError, "No usable source.*too little text"):
            source_policy.choose_verified_original(identity, [short])
        sanity = source_policy.content_sanity(short, identity)
        self.assertFalse(sanity["ok"])
        self.assertTrue(any("too little text" in reason for reason in sanity["reasons"]))

    def test_content_sanity_accepts_calibrated_range(self):
        sanity = source_policy.content_sanity(_utanet(), self.identity)
        self.assertTrue(sanity["ok"], sanity)
        self.assertEqual(sanity["lines"], 13)
        self.assertIn("chars_per_second", sanity)


class CliVerificationTests(unittest.TestCase):
    def _run(self, root, candidates, identity=None):
        audio = root / "ALIVE.flac"
        audio.write_bytes(b"")
        identity = identity or core.TrackIdentity("ALIVE", "ClariS", duration=240.0)
        collected = core.CollectionResult(candidates, [], [])
        config = cli.Config(root / "bk", root, cli.DEFAULT_OOGG_ENDPOINTS)
        with (
            mock.patch.object(cli, "probe_embedded_lyrics", return_value=cli.EmbeddedInfo("ClariS", "ALIVE", False, 0, 0)),
            mock.patch.object(cli, "probe_audio", return_value=identity),
            mock.patch.object(cli, "collect_candidates", return_value=collected),
        ):
            return cli._process_track(1, audio, config=config, no_translate=True, replace_existing=False, client=object())

    def test_verified_row_records_sources_and_diffs_in_run_table_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self._run(root, [_lrclib(), _utanet()])
            self.assertEqual(row["status"], "original_only")
            self.assertEqual(row["verification"], "verified")
            self.assertEqual(row["verified_by"], "two_source_agreement")
            self.assertEqual(row["sources"], "uta-net+lrclib")
            self.assertEqual(row["diff_lines"], 0)
            self.assertEqual(row["original_source_url"], "https://www.uta-net.com/song/321583/")
            payload = json.loads(Path(row["output_json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["verification_status"], "verified")
            self.assertEqual(payload["sources_used"], ["uta-net", "lrclib"])
            self.assertEqual(payload["verification"]["original_verified_by"], "two_source_agreement")
            self.assertTrue(payload["verification"]["source_verified"])
            text = Path(row["output_txt"]).read_text(encoding="utf-8")
            self.assertTrue(text.startswith("願った 勇敢な選択の\n"))
            self.assertNotIn("lyrics", payload)
            status, _, replace = cli._inspect_existing(root / "ALIVE.flac", output_root=root, overwrite=False, no_translate=True)
            self.assertEqual(status, "existing")
            self.assertFalse(replace)
            rendered = cli.render_markdown([row])
            self.assertIn("# Verification counts", rendered)
            self.assertIn("- verified (two_source_agreement): 1", rendered)
            self.assertIn("| verification | verified_by | sources | diff_lines | notation_diffs |", rendered)

    def test_single_source_row_is_written_as_unverified_and_redone_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self._run(root, [_lrclib()])
            self.assertEqual(row["status"], "original_only")
            self.assertEqual(row["verification"], "unverified")
            self.assertEqual(row["verified_by"], "single_source")
            self.assertEqual(row["sources"], "lrclib")
            payload = json.loads(Path(row["output_json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["verification_status"], "unverified")
            self.assertFalse(payload["verification"]["source_verified"])
            status, _, replace = cli._inspect_existing(root / "ALIVE.flac", output_root=root, overwrite=False, no_translate=True)
            self.assertIsNone(status)
            self.assertTrue(replace)
            existing = cli._existing_row(2, root / "ALIVE.flac", root, "legacy_unverified", payload)
            self.assertEqual(existing["verification"], "unverified")

    def test_wrong_song_only_gives_no_original_with_the_rejection_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrong = core.LyricCandidate(
                source="uta-net", url="https://www.uta-net.com/song/387865/", title="I Call Your Name",
                artist="Shogo Hamada", lines=[core.LyricPair(f"line {i}") for i in range(20)],
                meta={"trust": "publisher", "identity": {"title_level": 1, "artist_level": 0, "strong": False}},
            )
            row = self._run(root, [wrong], identity=core.TrackIdentity("Call you", "山田タマル", duration=200.0))
            self.assertEqual(row["status"], "no_original")
            self.assertEqual(row["verification"], "none")
            self.assertIn("title does not match", row["error"])
            self.assertEqual(row["output_txt"], "")


if __name__ == "__main__":
    unittest.main()


UTATEN_SEARCH = '''<html><body><section class="contentBox"><div class="contentBox__body">
<table class="searchResult artistLyricList">
  <tr><th class="searchResult__head">楽曲・タイトル</th><th class="searchResult__artist">アーティスト</th><th>歌詞・歌い出し</th></tr>
  <tr>
    <td><p class="searchResult__title">
          <a href="/lyric/mi24011216/">
          革命前線        </a>
    </p></td>
    <td class="searchResult__artist"><p>
          <a href="/artist/34415/">
          ツユ        </a></p>
      <div class="searchResult__lyricist"><p>作詞：<span class="songWriters"><a href="/songWriter/48503/">ぷす</a></span></p>
      <p>作曲：<span class="songWriters"><a href="/songWriter/48503/">ぷす</a></span></p></div>
    </td>
    <td class="lyricList__beginning"><a href="/lyric/mi24011216/">
          あの日の覚悟も 全部溶けてしまったんだ  ほら欲に塗れた 見苦しい悪魔            </a></td>
  </tr>
  <tr>
    <td><p class="searchResult__title"><a href="/lyric/xx1/">革命前線 (Remix)</a></p></td>
    <td class="searchResult__artist"><p><a href="/artist/1/">誰か</a></p></td>
    <td class="lyricList__beginning"><a href="/lyric/xx1/">別の歌い出し</a></td>
  </tr>
</table></div></section></body></html>'''

UTATEN_PAGE = '''<html><head><title>革命前線 歌詞 ツユ ふりがな付 - うたてん</title></head><body>
<h2 class="newLyricTitle__main">革命前線 <span class="newLyricTitle_afterTxt">歌詞</span></h2>
<div style="margin-top: 0px" class="lyricBody " oncontextmenu="return false">
  <div class="medium">
    <div class="hiragana" >
      あの<span class="ruby"><span class="rb">日</span><span class="rt">ひ</span></span>の<span class="ruby"><span class="rb">覚悟</span><span class="rt">かくご</span></span>も<br />
<span class="ruby"><span class="rb">全部</span><span class="rt">ぜんぶ</span></span><span class="ruby"><span class="rb">溶</span><span class="rt">と</span></span>けてしまったんだ<br />
<br />
ほら<span class="ruby"><span class="rb">欲</span><span class="rt">よく</span></span>に<span class="ruby"><span class="rb">塗</span><span class="rt">まみ</span></span>れた<br />
そうさ <span class="ruby"><span class="rb">君</span><span class="rt">きみ</span></span>だって<br />
    </div>
    <div class="romaji" style="display: none;">ano hi no kakugo mo<br /></div>
  </div>
</div></body></html>'''


class UtatenSourceTests(unittest.TestCase):
    def test_search_rows_and_no_result_page(self):
        from utaten_source import parse_utaten_search, utaten_search_url
        hits = parse_utaten_search(UTATEN_SEARCH)
        self.assertEqual([(h.url, h.title, h.artist) for h in hits], [
            ("https://utaten.com/lyric/mi24011216/", "革命前線", "ツユ"),
            ("https://utaten.com/lyric/xx1/", "革命前線 (Remix)", "誰か"),
        ])
        self.assertTrue(hits[0].opening.startswith("あの日の覚悟も"))
        self.assertEqual(hits[0].lyricist, "ぷす")
        self.assertEqual(parse_utaten_search('<p class="noItem">ごめんなさい。</p><table class="newestList"><tr><td><a href="/lyric/z/">x</a></td></tr></table>'), [])
        self.assertIn("title=%E9%9D%A9%E5%91%BD%E5%89%8D%E7%B7%9A", utaten_search_url("革命前線"))

    def test_page_drops_furigana_and_keeps_lines(self):
        from utaten_source import parse_utaten_page
        page = parse_utaten_page(UTATEN_PAGE)
        self.assertEqual((page.title, page.artist), ("革命前線", "ツユ"))
        self.assertEqual(page.lines, ["あの日の覚悟も", "全部溶けてしまったんだ", "ほら欲に塗れた", "そうさ 君だって"])
        with self.assertRaisesRegex(LookupError, "hiragana"):
            parse_utaten_page("<html><body>nothing</body></html>")

    def test_fetch_picks_exact_title_and_artist_row(self):
        from utaten_source import fetch_utaten_candidate
        client = FakeClient({"utaten.com/search?": UTATEN_SEARCH, "/lyric/mi24011216/": UTATEN_PAGE})
        candidate = fetch_utaten_candidate(core.TrackIdentity("革命前線", "ツユ", duration=200.0), client)
        self.assertEqual(candidate.source, "utaten")
        self.assertEqual(candidate.url, "https://utaten.com/lyric/mi24011216/")
        self.assertTrue(candidate.meta["identity"]["strong"])
        self.assertEqual(candidate.meta["trust"], "publisher")
        self.assertEqual(len(candidate.lines), 4)

    def test_two_publisher_sites_agreeing_verify_without_lrclib(self):
        utanet = _utanet()
        utaten = core.LyricCandidate(
            source="utaten", url="https://utaten.com/lyric/x/", title="ALIVE", artist="ClariS",
            lines=[core.LyricPair(line) for line in LRCLIB_LINES],
            meta={"trust": "publisher", "identity": {"title_level": 3, "artist_level": 2, "strong": True}},
        )
        chosen, evidence = source_policy.choose_verified_original(core.TrackIdentity("ALIVE", "ClariS", duration=240.0), [utaten, utanet])
        self.assertEqual(chosen.source, "uta-net")
        self.assertEqual(evidence["original_verified_by"], "two_source_agreement")
        self.assertEqual(evidence["sources_used"], ["uta-net", "utaten"])

    def test_collect_resolves_ambiguous_rows_for_each_site(self):
        from lyric_sites import AmbiguousHits, HitScore, SiteHit
        from utaten_source import UTATEN
        hits = [(HitScore(2, 0), SiteHit("https://utaten.com/lyric/a/", "粉雪", "レミオロメン", "　".join(l for l in UTANET_LINES[:6] if l)))]
        client = FakeClient({"/lyric/a/": UTATEN_PAGE.replace("革命前線", "粉雪").replace("ツユ", "レミオロメン")})

        def utaten(identity, c):
            raise AmbiguousHits("utaten: ambiguous", hits, UTATEN)

        result = core.collect_candidates(
            core.TrackIdentity("KONAYUKI", "Remioromen", duration=325.0, hints=("粉雪",)), client,
            utanet_fetcher=lambda i, c: (_ for _ in ()).throw(LookupError("no row")),
            utaten_fetcher=utaten,
            lrclib_fetcher=lambda i, c: core.LyricCandidate(
                source="lrclib", url="l", title="KONAYUKI", artist="Remioromen",
                lines=[core.LyricPair(line) for line in LRCLIB_LINES]),
            google_fetcher=lambda i, c: ([], []),
        )
        self.assertEqual([c.source for c in result.candidates], ["utaten", "lrclib"])
        self.assertFalse(result.candidates[0].meta["identity"]["strong"])
        self.assertIn("opening_matched_reference", result.candidates[0].meta["identity"])


JLYRIC_SEARCH = '''<div class="cap"><h2>検索結果.</h2></div><div class="bdy"><p class="mid"><a href="/artist/a0651f8/l06473f.html" title="Ave Mujica 顔 歌詞">顔</a></p><p class="sml">歌：<a href="/artist/a0651f8/" title="Ave Mujica 歌詞リスト">Ave Mujica</a></p><p class="sml">歌詞：事を/事を拗らせてるの/せてるのアルレッキーノ　コルセットを ...</p></div><div class="bdy"><p class="mid"><a href="/artist/a000001/l000001.html" title="くるり 朝顔 歌詞">朝顔</a></p><p class="sml">歌：<a href="/artist/a000001/" title="くるり 歌詞リスト">くるり</a></p><p class="sml">歌詞：別の歌い出し ...</p></div><div id="pager"><a class="sel" href="/search.php?p=1">1</a></div>'''

JLYRIC_PAGE = '''<!DOCTYPE html><html lang="ja"><head><meta charset="UTF-8"><title>顔／Ave Mujica 歌詞</title><meta name="description" content="Ave Mujicaの「顔」歌詞ページ。「顔」は、作詞：Diggy-MO&#039;、作曲：木下龍平(SUPA LOVE)・Diggy-MO&#039;です。"></head><body>
<div class="cap"><h2>「顔」歌詞</h2></div><div class="lbdy"><p class="sml">歌：<a href="/artist/a0651f8/" title="Ave Mujica 歌詞リスト">Ave Mujica</a></p><p class="sml">作詞：Diggy-MO&#039;　作曲：木下龍平(SUPA LOVE)・Diggy-MO&#039;</p><p id="Lyric">願った　勇敢な<ruby>選択<rt>せんたく</rt></ruby>の<br>未来を今照らしてく<br><br>利己的な感情　ぶつけあっちゃ(まだ)見えないね<br></p><div class="fnt">文字サイズ：</div></div>
<div class="cap"><h2>お願い</h2></div><div class="bdy"><p class="ss">歌詞検索J-Lyric.netでは、無料で歌詞の検索・閲覧サービスを提供しておりますが</p></div></body></html>'''


class JlyricSourceTests(unittest.TestCase):
    def test_search_rows_use_exact_title_mode_and_relative_links(self):
        from jlyric_source import jlyric_search_url, parse_jlyric_search
        self.assertIn("ct=1", jlyric_search_url("顔"))
        hits = parse_jlyric_search(JLYRIC_SEARCH)
        self.assertEqual([(h.url, h.title, h.artist) for h in hits], [
            ("https://j-lyric.net/artist/a0651f8/l06473f.html", "顔", "Ave Mujica"),
            ("https://j-lyric.net/artist/a000001/l000001.html", "朝顔", "くるり"),
        ])
        self.assertTrue(hits[0].opening.startswith("事を/事を"))

    def test_page_parses_lyric_block_title_and_artist(self):
        from jlyric_source import parse_jlyric_page
        page = parse_jlyric_page(JLYRIC_PAGE)
        self.assertEqual((page.title, page.artist), ("顔", "Ave Mujica"))
        self.assertEqual(page.lines, ["願った 勇敢な選択の", "未来を今照らしてく", "利己的な感情 ぶつけあっちゃ(まだ)見えないね"])
        with self.assertRaisesRegex(LookupError, "Lyric"):
            parse_jlyric_page("<html><body>nothing</body></html>")

    def test_fetch_rejects_lookalike_title_and_accepts_the_exact_row(self):
        from jlyric_source import fetch_jlyric_candidate
        client = FakeClient({"search.php?": JLYRIC_SEARCH, "/l06473f.html": JLYRIC_PAGE})
        candidate = fetch_jlyric_candidate(core.TrackIdentity("顔", "Ave Mujica", duration=240.0), client)
        self.assertEqual(candidate.source, "j-lyric")
        self.assertEqual(candidate.url, "https://j-lyric.net/artist/a0651f8/l06473f.html")
        self.assertTrue(candidate.meta["identity"]["strong"])
        self.assertFalse(any("l000001" in url for url in client.requests))

    def test_accented_title_matches_plain_spelling(self):
        identity = core.TrackIdentity("天球のMúsica", "Ave Mujica")
        self.assertEqual(core.title_match_level(identity, "天球のMusica"), 3)
        self.assertIn("天球のMusica", core.title_search_keywords(identity))
        self.assertEqual(core._normalized("がぎぐ Música"), "がぎぐmusica")


class SearchDegradationTests(unittest.TestCase):
    def test_duckduckgo_is_skipped_after_repeated_timeouts_in_one_run(self):
        from urllib.error import URLError

        class Client:
            def __init__(self):
                self.ddg_calls = 0

            def get_text(self, url, params=None, headers=None, retries=1, timeout=20):
                if "google.com/search" in url:
                    return "<html></html>"
                if "duckduckgo.com" in url:
                    self.ddg_calls += 1
                    raise URLError("timed out")
                raise AssertionError(url)

        client = Client()
        for _ in range(4):
            result = core.fetch_google_candidates(core.TrackIdentity("Song", "Artist"), client)
        self.assertEqual(client.ddg_calls, core.DUCKDUCKGO_MAX_FAILURES)
        self.assertTrue(any("skipped after repeated timeouts" in item for item in result.errors))
        self.assertEqual(result.search_engine, "none")
