"""Near phonetic agreement is review context, NEVER new Central authority."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from segment_phonetic_audit import audit_japanese_asr_phonetics


def converter(x: str) -> str:
    mapping={
        '今日を称えて':'kyouwotonaete','今日を抱えて':'kyouwokakaete',
        'ただ舞って舞う':'tadamatsutemau',
        'ただ待ってます':'tadamatsutemasu',
        '何も知らない':'nanimoshiranai',
        '別の言葉':'betsunokotoba',
        '音が似てる':'onoganiteru',
        'そっくり':'sokkuri'
    }
    return mapping.get(x,x)


class PhoneticReviewOnlyTests(unittest.TestCase):
    def check(self,target,raw,others=()):
        return audit_japanese_asr_phonetics(target_lyric=target,raw_asr_text=raw,
           other_lyric_texts=others,to_romaji=converter)
    def test_asr_misread_cannot_be_reclassified_as_identity(self):
        x=self.check('今日を称えて','今日を抱えて')
        self.assertEqual(x.status,'BLOCKED_INDEPENDENT_ASR_PHONETIC_MISMATCH')
        self.assertFalse(x.independent_occurrence_proven)
        self.assertFalse(x.direct_onset_proven)
    def test_very_similar_wrong_lyric_only_produces_review_target(self):
        x=self.check('ただ舞って舞う','ただ待ってます')
        self.assertEqual(x.status,'REVIEW_ONLY_PHONETICALLY_SIMILAR')
        self.assertGreater(x.match_similarity,.90)
        self.assertFalse(x.independent_occurrence_proven)
    def test_phonetic_homophone_ambiguous(self):
        x=self.check('ただ舞って舞う','ただ待ってます',('ただ舞って舞う',))
        self.assertEqual(x.status,'BLOCKED_SAME_PHONETIC_LYRIC_EXISTS')
    def test_near_competitor_blocks(self):
        x=self.check('ただ舞って舞う','ただ待ってます',('ただ待ってます',))
        self.assertEqual(x.status,'BLOCKED_PHONETIC_NEAREST_LYRIC_AMBIGUOUS')
    def test_missing_phonetic_converter_fails(self):
        x=audit_japanese_asr_phonetics(target_lyric='これが本当',
          raw_asr_text='何も知らない',other_lyric_texts=[],
          to_romaji=lambda _:None)
        self.assertEqual(x.status,'BLOCKED_PHONETIC_PROFILE')
    def test_short_sound_not_evidence(self):
        x=self.check('そっくり','そっくり')
        self.assertEqual(x.status,'BLOCKED_PHONETIC_PROFILE')


if __name__=='__main__':unittest.main()