from __future__ import annotations
import unittest
from test_lyricsee import load_core


class SourceIntegrity20261008(unittest.TestCase):
    def setUp(self):
        self.core=load_core()

    def test_render_bilingual_field_with_internal_double_spaces_is_unambiguous(self):
        core=self.core
        output=core.render_lyrics([
            core.LyricPair("嘲笑う世界  闇を暴け","嘲笑著的世界  揭露黑暗")
        ])
        self.assertEqual(output,"嘲笑う世界 闇を暴け  嘲笑著的世界 揭露黑暗\n")
        self.assertEqual(output.count("  "),1)

    def test_lrclib_partial_title_or_artist_does_not_silently_pass(self):
        core=self.core
        class Client:
            def get_json(self,url,params=None):
                return [{"id":4,"trackName":"A Promise TV Size","artistName":"Wrong Artist",
                         "duration":100.,"plainLyrics":"hello\nworld"}]
        with self.assertRaisesRegex(LookupError,"metadata"):
            core.fetch_lrclib_candidate(core.TrackIdentity(
                "A Promise","結城アイラ",duration=240.0),Client())

    def test_lrclib_rejects_duration_mismatch_even_when_name_matches(self):
        core=self.core
        class Client:
            def get_json(self,url,params=None):
                return [{"id":4,"trackName":"A Promise","artistName":"結城アイラ",
                         "duration":89.,"plainLyrics":"hello\nworld"}]
        with self.assertRaisesRegex(LookupError,"duration"):
            core.fetch_lrclib_candidate(core.TrackIdentity(
                "A Promise","結城アイラ",duration=240.0),Client())

    def test_mistyped_title_caption_and_credit_are_removed(self):
        core=self.core
        identity=core.TrackIdentity("素晴らしき世界 でも どこにもない場所","Ave Mujica")
        pairs=[
            core.LyricPair("素晴らしき世界でもどにもない場所 - Ave Mujica",
                           "素晴らしき世界 (Utopia) - Ave Mujica"),
            core.LyricPair("词：Diggy-MO'","詞：Diggy-MO'"),
            core.LyricPair("曲：長谷川大介","曲：長谷川大介"),
            core.LyricPair("失えばいい","不畏失去")]
        self.assertEqual(core._sanitize_candidate_lines(pairs,identity),
                         [core.LyricPair("失えばいい","不畏失去")])

    def test_google_different_song_rows_blocked_by_structured_originals(self):
        core=self.core
        ref=[core.LyricPair(f"原文{i}の声",f"譯文{i}") for i in range(10)]
        wrong=[core.LyricPair(f"他の世界{i}",f"錯譯{i}") for i in range(12)]
        result=core.collect_candidates(
            core.TrackIdentity("影色舞","MyGO!!!!!"),object(),
            lrclib_fetcher=lambda identity,client:core.LyricCandidate(
                source="lrclib",url="https://lrclib.net/api/get/1",
                title=identity.title,artist=identity.artist,lines=ref),
            google_fetcher=lambda identity,client:(
                ["https://home.gamer.com.tw/artwork.php?sn=1"],
                [core.LyricCandidate(
                    source="google:home.gamer.com.tw",
                    url="https://home.gamer.com.tw/artwork.php?sn=1",
                    title="影色舞",artist="",lines=wrong)]),
        )
        self.assertEqual([x.source for x in result.candidates],["lrclib"])
        self.assertTrue(any("disagree" in x for x in result.errors))

    def test_untrusted_site_without_second_source_is_not_admitted(self):
        core=self.core
        fake=core.LyricCandidate(
            source="google:open.spotify.com",url="https://open.spotify.com/track/1",
            title="Song",artist="Artist",
            lines=[core.LyricPair(f"metadata about album{i}",f"中文{i}") for i in range(30)])
        result=core.collect_candidates(
            core.TrackIdentity("Song","Artist"),object(),
            lrclib_fetcher=lambda i,c:(_ for _ in ()).throw(LookupError()),
            google_fetcher=lambda i,c:([fake.url],[fake]))
        self.assertEqual(result.candidates,[])
        self.assertTrue(any("host rejected" in x for x in result.errors))


if __name__=="__main__": unittest.main()
