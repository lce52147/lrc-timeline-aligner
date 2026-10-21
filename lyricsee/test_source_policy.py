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


def candidate(source, url, rows, *, title="Sample", artist="Singer"):
    return LyricCandidate(
        source=source, url=url, title=title, artist=artist, lines=rows,
    )


class SourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.identity=TrackIdentity("Sample","Singer",duration=201.0)
        self.rows=[
            LyricPair(f"君の声と星が光る{i}") for i in range(12)
        ]
        self.ref=candidate(
            "lrclib","https://lrclib.net/api/get/123",self.rows,
        )

    def test_lrclib_canonical_original_with_evidence(self):
        orig,evidence=choose_verified_original(self.identity,[self.ref])
        self.assertIs(orig,self.ref)
        self.assertEqual(evidence["original_verified_by"],"lrclib_title_artist_duration")
        self.assertEqual(evidence["canonical_original_url"],self.ref.url)

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
        orig,_=choose_verified_original(self.identity,[self.ref,web])
        with self.assertRaisesRegex(LookupError,"No independently verified"):
            choose_verified_web_translation(self.identity,orig,[self.ref,web])

    def test_valid_web_translation_has_canonical_original_order(self):
        web=candidate(
            "google:home.gamer.com.tw",
            "https://home.gamer.com.tw/artwork.php?sn=1",
            [LyricPair(r.original,f"第{i}行的譯文") for i,r in
             reversed(list(enumerate(self.rows)))],
        )
        original,_=choose_verified_original(self.identity,[self.ref,web])
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
        self.assertEqual(
            evidence["original_verified_by"],
            "independent_web_original_agreement",
        )

    def test_failed_legacy_or_tampered_provenance_does_not_pass(self):
        payload=b"original  translation\n"
        old={"selected_source":"legacy","txt_sha256":hashlib.sha256(payload).hexdigest()}
        self.assertFalse(verified_provenance_is_current(old,payload))
        original,proof=choose_verified_original(self.identity,[self.ref])
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


if __name__ == "__main__":
    unittest.main()
