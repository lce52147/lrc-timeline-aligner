# s2tw round trip over the user's own lyrics

Method: `roundtrip.py` (kept outside the repo) takes every report in `outputs/reports` whose LRC exists and decodes confidently, keeps the lyric lines without kana, converts them with OpenCC `t2s` and then with `zh_traditional.convert_text`, and compares the result with the original line character by character. Only character pairs and counts are recorded, never lyric text.

Corpus: 169 songs, 6628 lines, 66860 Han characters (the kana-free lines of all songs; only 4 songs, 137 lines, have no kana at all and those round trip with 0 differences before and after). 50 reports had no LRC and were skipped.

Categories: `wrong` = the other character has a different meaning (乾/幹, 後/后, 髮/發, 隻/只, 幾/几, 裡/里, 鬆/松, 捲/卷); `variant` = both are standard Traditional; `japanese` = the source line had a Japanese or simplified glyph (国 独 声 ...) that is not Traditional input.

| | stock OpenCC s2tw | zh_traditional |
|---|---|---|
| lines with a difference | 108 | 24 |
| character differences | 117 | 26 |
| wrong meaning | 26 | 1 |
| variant | 80 | 14 |
| Japanese / not Traditional input | 11 | 11 |

The one remaining wrong pair is 髮 -> 發 inside a proper name written with 后+发; no context rule can tell it from 发现.

Corrections are derived from this same corpus (see `PHRASE_OVERRIDES` and `VARIANT_PREFERENCE` in `scripts/zh_traditional.py`), so the numbers above are a fit to the user's lyrics, not a held-out measurement.

## Stock OpenCC s2tw (pair, count, uses of the original character in the corpus, uses of the produced character)
~~~
songs 169 skipped 50 lines 6628 han_chars 66860 lines_with_diff 108
pairs 117 {'variant': 80, 'wrong': 26, 'japanese': 11}
台>臺 U+53F0>U+81FA x17 | orig_uses 17 back_uses 3
濕>溼 U+6FD5>U+6EBC x11 | orig_uses 11 back_uses 0
只>隻 U+53EA>U+96BB x8 | orig_uses 253 back_uses 13
游>遊 U+6E38>U+904A x7 | orig_uses 16 back_uses 4
背>揹 U+80CC>U+63F9 x4 | orig_uses 29 back_uses 1
幾>几 U+5E7E>U+51E0 x4 | orig_uses 38 back_uses 0
捲>卷 U+6372>U+5377 x4 | orig_uses 6 back_uses 0
畫>劃 U+756B>U+5283 x4 | orig_uses 16 back_uses 9
注>註 U+6CE8>U+8A3B x4 | orig_uses 25 back_uses 0
弦>絃 U+5F26>U+7D43 x4 | orig_uses 4 back_uses 0
裏>裡 U+88CF>U+88E1 x4 | orig_uses 4 back_uses 230
污>汙 U+6C61>U+6C59 x3 | orig_uses 3 back_uses 2
裡>里 U+88E1>U+91CC x3 | orig_uses 230 back_uses 0
乾>幹 U+4E7E>U+5E79 x3 | orig_uses 16 back_uses 0
周>週 U+5468>U+9031 x3 | orig_uses 10 back_uses 1
当>當 U+5F53>U+7576 x2 | orig_uses 2 back_uses 90
了>瞭 U+4E86>U+77AD x2 | orig_uses 705 back_uses 2
念>唸 U+5FF5>U+5538 x1 | orig_uses 50 back_uses 0
髮>發 U+9AEE>U+767C x1 | orig_uses 7 back_uses 97
回>迴 U+56DE>U+8FF4 x1 | orig_uses 181 back_uses 25
淒>悽 U+6DD2>U+60BD x1 | orig_uses 2 back_uses 0
顔>顏 U+9854>U+984F x1 | orig_uses 1 back_uses 17
装>裝 U+88C5>U+88DD x1 | orig_uses 1 back_uses 28
發>髮 U+767C>U+9AEE x1 | orig_uses 97 back_uses 7
蘇>甦 U+8607>U+7526 x1 | orig_uses 1 back_uses 3
嘆>歎 U+5606>U+6B4E x1 | orig_uses 12 back_uses 0
沈>沉 U+6C88>U+6C89 x1 | orig_uses 2 back_uses 44
礼>禮 U+793C>U+79AE x1 | orig_uses 1 back_uses 8
後>后 U+5F8C>U+540E x1 | orig_uses 118 back_uses 0
欲>慾 U+6B32>U+617E x1 | orig_uses 8 back_uses 6
乾>干 U+4E7E>U+5E72 x1 | orig_uses 16 back_uses 1
雲>云 U+96F2>U+4E91 x1 | orig_uses 29 back_uses 0
儘>盡 U+5118>U+76E1 x1 | orig_uses 5 back_uses 62
燻>薰 U+71FB>U+85B0 x1 | orig_uses 1 back_uses 0
潜>潛 U+6F5C>U+6F5B x1 | orig_uses 1 back_uses 4
讚>贊 U+8B9A>U+8D0A x1 | orig_uses 3 back_uses 0
独>獨 U+72EC>U+7368 x1 | orig_uses 1 back_uses 56
会>會 U+4F1A>U+6703 x1 | orig_uses 1 back_uses 403
嚮>向 U+56AE>U+5411 x1 | orig_uses 4 back_uses 124
毀>燬 U+6BC0>U+71EC x1 | orig_uses 8 back_uses 0
声>聲 U+58F0>U+8072 x1 | orig_uses 1 back_uses 142
胡>衚 U+80E1>U+885A x1 | orig_uses 2 back_uses 0
同>衕 U+540C>U+8855 x1 | orig_uses 135 back_uses 0
托>託 U+6258>U+8A17 x1 | orig_uses 6 back_uses 24
堕>墮 U+5815>U+58AE x1 | orig_uses 1 back_uses 2
国>國 U+56FD>U+570B x1 | orig_uses 1 back_uses 2
盡>儘 U+76E1>U+5118 x1 | orig_uses 62 back_uses 5

~~~

## zh_traditional
~~~
songs 169 skipped 50 lines 6628 han_chars 66860 lines_with_diff 24
pairs 26 {'wrong': 1, 'japanese': 11, 'variant': 14}
裏>裡 U+88CF>U+88E1 x4 | orig_uses 4 back_uses 230
臺>台 U+81FA>U+53F0 x3 | orig_uses 3 back_uses 17
当>當 U+5F53>U+7576 x2 | orig_uses 2 back_uses 90
汙>污 U+6C59>U+6C61 x2 | orig_uses 2 back_uses 3
髮>發 U+9AEE>U+767C x1 | orig_uses 7 back_uses 97
顔>顏 U+9854>U+984F x1 | orig_uses 1 back_uses 17
装>裝 U+88C5>U+88DD x1 | orig_uses 1 back_uses 28
蘇>甦 U+8607>U+7526 x1 | orig_uses 1 back_uses 3
沈>沉 U+6C88>U+6C89 x1 | orig_uses 2 back_uses 44
礼>禮 U+793C>U+79AE x1 | orig_uses 1 back_uses 8
欲>慾 U+6B32>U+617E x1 | orig_uses 8 back_uses 6
儘>盡 U+5118>U+76E1 x1 | orig_uses 5 back_uses 62
潜>潛 U+6F5C>U+6F5B x1 | orig_uses 1 back_uses 4
独>獨 U+72EC>U+7368 x1 | orig_uses 1 back_uses 56
会>會 U+4F1A>U+6703 x1 | orig_uses 1 back_uses 403
声>聲 U+58F0>U+8072 x1 | orig_uses 1 back_uses 142
托>託 U+6258>U+8A17 x1 | orig_uses 6 back_uses 24
堕>墮 U+5815>U+58AE x1 | orig_uses 1 back_uses 2
国>國 U+56FD>U+570B x1 | orig_uses 1 back_uses 2

~~~
