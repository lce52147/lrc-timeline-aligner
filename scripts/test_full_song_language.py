import unittest
import auto_lrc as l

class WholeSongLanguageTests(unittest.TestCase):
 def test_english_opening_japanese_verse(self):
    e=[l.LyricEntry(['English lyric and long sentence']) for _ in range(12)]
    e.extend(l.LyricEntry(['あの日の夢から目覚めて']) for _ in range(28))
    self.assertEqual(l.infer_spoken_language(e),'ja')
 def test_japanese_opening_english_verse(self):
    e=[l.LyricEntry(['夢の途中で']) for _ in range(12)]
    e.extend(l.LyricEntry(['The journey has finally begun']) for _ in range(28))
    self.assertEqual(l.infer_spoken_language(e),'en')
 def test_full_english(self):
    self.assertEqual(l.infer_spoken_language([l.LyricEntry(['I will never forget'])]),'en')
 def test_full_chinese(self):
    self.assertEqual(l.infer_spoken_language([l.LyricEntry(['在遠方的光芒'])]),'zh')
 def test_no_lyrics(self):
    self.assertEqual(l.infer_spoken_language([]),'ja')
if __name__=='__main__':unittest.main()
