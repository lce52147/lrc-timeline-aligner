"""Explicit human-corrected BK timing, never implicit auto inference."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import auto_lrc as a


class CheckedBkReferenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp=TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.music=Path(self._tmp.name)/'Music'
        self.album=self.music/'Album'
        self.album.mkdir(parents=True)
        self.audio=self.album/'Song.flac'
        self.audio.touch()
        self.ref_dir=self.music/'LRC tools checked references'/'LRC bk'
        self.ref_dir.mkdir(parents=True)
        self.bk=self.ref_dir/'Song.lrc'
        self.bk.write_text(
            '[ti:Song]\n[00:00.00]Artist - Song\n'
            '[00:11.25]first singing line\n[00:22.75]second singing line\n',
            encoding='utf-8',
        )
        self.untimed=self.album/'Song.lyrics.txt'
        self.untimed.write_text('first singing line\nsecond singing line\n',encoding='utf-8')
        self.entries=[a.LyricEntry(['first singing line']),
                      a.LyricEntry(['second singing line'])]
    def test_checked_scans_bk_without_implicit_auto_scan(self):
        got=a.checked_lrc_candidates(self.audio)
        self.assertIn(self.bk,got)
        self.assertEqual(a.find_lyrics(self.audio,include_checked=False),self.untimed)
        self.untimed.unlink()
        with self.assertRaises(a.LrcError):
            a.find_lyrics(self.audio,include_checked=False)
        self.assertEqual(a.find_lyrics(self.audio,include_checked=True),self.bk)
    def test_opt_in_accepts_exact_lyrics_with_metadata_card(self):
        with patch.object(a,'audio_track_labels',return_value=('Artist','Song')):
            result=a.checked_lrc_timing_hint(self.audio,self.entries,self.untimed)
        self.assertIsNotNone(result)
        lyrics,times,report=result
        self.assertEqual(times,[11.25,22.75])
        self.assertEqual(report['timing_provenance'],'human-reviewed-bk')
        self.assertTrue(report['uses_reference_timestamps'])
        self.assertFalse(report['independent_auto_benchmark_eligible'])
        self.assertEqual(len(lyrics),2)
    def test_bad_lyric_order_must_fail_closed(self):
        with patch.object(a,'audio_track_labels',return_value=('Artist','Song')):
            result=a.checked_lrc_timing_hint(
                self.audio,list(reversed(self.entries)),self.untimed)
        self.assertIsNone(result)
    def test_refusing_self_as_secondary_checked_source(self):
        # If source lyrics are already the BK file, the outer checked branch
        # must honor its explicit timestamps rather than comparing it to itself.
        with patch.object(a,'audio_track_labels',return_value=('Artist','Song')):
            result=a.checked_lrc_timing_hint(self.audio,self.entries,self.bk)
        self.assertIsNone(result)
    def test_oogg_rana_bk_provenance_not_a_sung_line(self):
        original=self.bk.read_text(encoding='utf-8')
        self.bk.write_text('[oogg_rana]\n'+original,encoding='utf-8')
        raw=a.load_lyrics(self.bk)
        self.assertTrue(raw.saw_timestamps)
        self.assertIn('[oogg_rana]',raw.metadata_lines)
        self.assertEqual(len(raw.entries),3)
        with patch.object(a,'audio_track_labels',return_value=('Artist','Song')):
            hint=a.checked_lrc_timing_hint(self.audio,self.entries,self.untimed)
        self.assertIsNotNone(hint)
        self.assertEqual(hint[1],[11.25,22.75])

    def test_legacy_timestamped_oogg_provenance_not_sung(self):
        marker=a.LyricEntry(['[oogg_rana]'],1648)
        real=a.LyricEntry(['first singing line'],1736)
        kept,skipped=a.remove_instrumental_markers([marker,real])
        self.assertEqual(len(kept),1)
        self.assertEqual(a.entry_sung_text(kept[0]),'first singing line')
        self.assertEqual(len(skipped),1)

    def test_unknown_bracketed_lyric_not_silently_dropped(self):
        self.bk.write_text('[madeup_hook]\n[00:01.00]Song lyric\n',encoding='utf-8')
        raw=a.load_lyrics(self.bk)
        self.assertEqual(a.entry_sung_text(raw.entries[0]),'[madeup_hook]')

    def test_metadata_card_not_stripped_if_real_lyric(self):
        # The first zero-timestamp row must resemble the actual title.
        self.bk.write_text('[00:00.00]singing at zero\n'
                           '[00:11.25]first singing line\n'
                           '[00:22.75]second singing line\n',encoding='utf-8')
        with patch.object(a,'audio_track_labels',return_value=('Artist','Song')):
            result=a.checked_lrc_timing_hint(self.audio,self.entries,self.untimed)
        self.assertIsNone(result)


if __name__=='__main__':
    unittest.main()