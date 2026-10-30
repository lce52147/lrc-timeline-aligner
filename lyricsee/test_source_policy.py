from __future__ import annotations
import hashlib
import json
import unittest
from lyricsee_core import (
    TrackIdentity, LyricPair, LyricCandidate, collect_candidates,
    review_translations_with_local_llm,
)
from source_policy import (
    VERSION, choose_verified_original, choose_verified_web_translation,
    validate_output, attach_verification, verified_provenance_is_current,
)


def candidate(source, url, rows, *, title="Sample", artist="Singer", meta=None):
    return LyricCandidate(
        source=source, url=url, title=title, artist=artist, lines=rows, meta=meta,
    )


class SourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.identity=TrackIdentity("Sample","Singer",duration=201.0)
        self.rows=[
            LyricPair(f"君の声と星が光る夜に{i}") for i in range(12)
        ]
        self.ref=candidate(
            "lrclib","https://lrclib.net/api/get/123",self.rows,
        )

    def test_lrclib_alone_is_kept_but_unverified(self):
        orig,evidence=choose_verified_original(self.identity,[self.ref])
        self.assertIs(orig,self.ref)
        self.assertEqual(evidence["verification_status"],"unverified")
        self.assertEqual(evidence["original_verified_by"],"single_source")
        self.assertEqual(evidence["canonical_original_url"],self.ref.url)
        self.assertEqual(evidence["sources_used"],["lrclib"])

    def test_page_without_independent_original_rejected(self):
        web=candidate(
            "google:home.gamer.com.tw",
            "https://home.gamer.com.tw/artwork.php?sn=1",
            [LyricPair(r.original,f"中文譯文{i}") for i,r in enumerate(self.rows)],
        )
        with self.assertRaisesRegex(LookupError,"Cannot verify original"):
            choose_verified_original(self.identity,[web])

    def test_wrong_song_web_translation_not_accepted(self):
        web=candidate(
            "google:home.gamer.com.tw",
            "https://home.gamer.com.tw/artwork.php?sn=1",
            [LyricPair(f"違う作品{i}の歌",f"中文{i}") for i in range(20)],
        )
        orig,evidence=choose_verified_original(self.identity,[self.ref,web])
        self.assertIs(orig,self.ref)
        self.assertEqual(evidence["verification_status"],"unverified")
        with self.assertRaisesRegex(LookupError,"No independently verified"):
            choose_verified_web_translation(self.identity,orig,[self.ref,web])

    def test_valid_web_translation_has_canonical_original_order(self):
        # The page writes one line in kana, adds a ラララ fill and keeps the
        # rest: notation differences only, so the two sources agree.
        page_rows=[LyricPair(r.original,f"第{i}行的譯文") for i,r in enumerate(self.rows)]
        page_rows[3]=LyricPair(page_rows[3].original.replace("君","きみ"),page_rows[3].translation)
        page_rows.insert(6,LyricPair("ラララ ラララ","啦啦啦"))
        web=candidate(
            "google:home.gamer.com.tw",
            "https://home.gamer.com.tw/artwork.php?sn=1",
            page_rows,
        )
        original,evidence=choose_verified_original(self.identity,[self.ref,web])
        self.assertEqual(evidence["verification_status"],"verified")
        self.assertEqual(evidence["original_verified_by"],"two_source_agreement")
        output=choose_verified_web_translation(self.identity,original,[web])
        validate_output(original, output, require_translation=True)
        self.assertEqual(
            [x.original for x in output.lines],
            [x.original for x in self.rows],
        )

    def test_two_independent_web_hosts_can_verify_original(self):
        a=candidate(
            "google:home.gamer.com.tw",
            "https://home.gamer.com.tw/artwork.php?sn=1",
            [LyricPair(r.original,"譯文") for r in self.rows],
        )
        b=candidate(
            "google:www.marumaru-x.com",
            "https://www.marumaru-x.com/japanese-song/play-1",
            [LyricPair(r.original,"另一譯文") for r in self.rows],
        )
        result,evidence=choose_verified_original(self.identity,[a,b])
        self.assertIs(result,a)
        self.assertEqual(evidence["verification_status"],"verified")
        self.assertEqual(evidence["original_verified_by"],"two_source_agreement")
        self.assertFalse(evidence["artist_confirmed"])

    def test_same_host_pages_are_one_source(self):
        a=candidate(
            "google:home.gamer.com.tw",
            "https://home.gamer.com.tw/artwork.php?sn=1",
            [LyricPair(r.original,"譯文") for r in self.rows],
        )
        b=candidate(
            "google:home.gamer.com.tw",
            "https://home.gamer.com.tw/artwork.php?sn=2",
            [LyricPair(r.original,"另一譯文") for r in self.rows],
        )
        with self.assertRaisesRegex(LookupError,"Cannot verify original"):
            choose_verified_original(self.identity,[a,b])

    def test_failed_legacy_or_tampered_provenance_does_not_pass(self):
        payload=b"original  translation\n"
        old={"selected_source":"legacy","txt_sha256":hashlib.sha256(payload).hexdigest()}
        self.assertFalse(verified_provenance_is_current(old,payload))
        web=candidate(
            "google:home.gamer.com.tw",
            "https://home.gamer.com.tw/artwork.php?sn=1",
            [LyricPair(r.original,"譯文") for r in self.rows],
        )
        original,proof=choose_verified_original(self.identity,[self.ref,web])
        out=candidate("local-llm:OOGG",original.url,
                      [LyricPair(x.original,"合格的中文") for x in original.lines])
        attach_verification(out,proof,translation_method="local_llm")
        metadata={
            "selected_source":out.source,
            "verification":out.verification,
            "txt_sha256":hashlib.sha256(payload).hexdigest(),
        }
        self.assertTrue(verified_provenance_is_current(metadata,payload))
        self.assertFalse(verified_provenance_is_current(metadata,payload+b"tamper"))
        self.assertFalse(
            out.verification["translation_semantics_human_checked"]
        )

    def test_unverified_provenance_is_never_current(self):
        payload=b"original\n"
        original,proof=choose_verified_original(self.identity,[self.ref])
        out=candidate("original-only",original.url,[LyricPair(x.original) for x in original.lines])
        attach_verification(out,proof,translation_method="none")
        self.assertFalse(out.verification["source_verified"])
        self.assertEqual(out.verification["verification_status"],"unverified")
        metadata={"verification":out.verification,"txt_sha256":hashlib.sha256(payload).hexdigest()}
        self.assertFalse(verified_provenance_is_current(metadata,payload))

    def test_validate_output_allows_original_only_without_mutation(self):
        original,_=choose_verified_original(self.identity,[self.ref])
        output=candidate(
            "original-only",original.url,
            [LyricPair(row.original) for row in original.lines],
        )
        validate_output(original, output, require_translation=False)
        output.lines[0]=LyricPair("changed")
        with self.assertRaisesRegex(ValueError,"Original lyric changed"):
            validate_output(original, output, require_translation=False)

    def test_review_fixes_but_does_not_change_original(self):
        original=[LyricPair("私はダメ 油断してると","我真沒用，只要一掉以為輕"),
                  LyricPair("私も知らない私のことに","連連我自己都不知道的自我")]
        def send(url,payload,timeout):
            user=json.loads(payload["messages"][1]["content"])
            ids=user["review_only_ids"]
            rows=[{"id":ids[0],"text":"我可不行，一不小心鬆懈就會"},
                  {"id":ids[1],"text":"關於連我自己都不知道的自己"}]
            return {"choices":[{"message":{"content":json.dumps({"translations":rows},ensure_ascii=False)}}]}
        reviewed=review_translations_with_local_llm(
            original,title="A Promise",endpoint="http://localhost:6969",
            model="OOGG",post_json=send,
        )
        self.assertEqual(reviewed[0].original,original[0].original)
        self.assertEqual(reviewed[1].original,original[1].original)
        self.assertNotIn("連連",reviewed[1].translation)


class DroppedShortLineTests(unittest.TestCase):
    """A publisher page that lost a short line is completed from the agreeing source."""

    def setUp(self):
        self.identity=TrackIdentity("Sample","Singer",duration=200.0)
        self.full=[LyricPair(f"君の声と星が光る夜に{i}") for i in range(14)]
        self.full.insert(0,LyricPair("空へ"))
        self.full.insert(8,LyricPair("空へ"))
        self.full.append(LyricPair("空は青く"))

    def _pair(self, drop):
        short=[row for i,row in enumerate(self.full) if i not in drop]
        primary=candidate("uta-net","https://www.uta-net.com/song/1/",short)
        other=candidate("lrclib","https://lrclib.net/api/get/1",list(self.full))
        return primary,other

    def _texts(self, cand):
        return [row.original for row in cand.lines]

    def test_dropped_closing_line_is_restored(self):
        primary,other=self._pair({len(self.full)-1})
        chosen,evidence=choose_verified_original(self.identity,[primary,other])
        self.assertEqual(self._texts(chosen),self._texts(other))
        self.assertEqual(evidence["verification_status"],"verified")
        self.assertEqual(evidence["primary_source"],"uta-net")
        added=evidence["lines_added_from_cross_source"]
        self.assertEqual([a["text"] for a in added],["空は青く"])

    def test_dropped_repeated_refrain_is_restored_in_place(self):
        primary,other=self._pair({8})
        chosen,_=choose_verified_original(self.identity,[primary,other])
        self.assertEqual(self._texts(chosen),self._texts(other))

    def test_dropped_opening_line_is_restored(self):
        primary,other=self._pair({0})
        chosen,_=choose_verified_original(self.identity,[primary,other])
        self.assertEqual(self._texts(chosen),self._texts(other))

    def test_complete_primary_is_returned_untouched(self):
        primary=candidate("uta-net","https://www.uta-net.com/song/1/",list(self.full))
        other=candidate("lrclib","https://lrclib.net/api/get/1",list(self.full))
        chosen,evidence=choose_verified_original(self.identity,[primary,other])
        self.assertIs(chosen,primary)
        self.assertNotIn("lines_added_from_cross_source",evidence)

    def test_web_page_commentary_is_never_added(self):
        primary=candidate("uta-net","https://www.uta-net.com/song/1/",list(self.full))
        web=candidate("google:home.gamer.com.tw","https://home.gamer.com.tw/artwork.php?sn=1",
                      list(self.full)+[LyricPair("ピアノ：誰か")])
        chosen,evidence=choose_verified_original(self.identity,[primary,web])
        self.assertIs(chosen,primary)
        self.assertNotIn("lines_added_from_cross_source",evidence)

    def test_source_with_many_extra_lines_is_not_merged(self):
        primary,other=self._pair({1,3,5,7})
        chosen,evidence=choose_verified_original(self.identity,[primary,other])
        self.assertIs(chosen,primary)
        self.assertNotIn("lines_added_from_cross_source",evidence)


if __name__ == "__main__":
    unittest.main()
