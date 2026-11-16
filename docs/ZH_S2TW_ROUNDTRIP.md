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

## Word-level audit of the character preference (BH2-FIX9)

`VARIANT_PREFERENCE` rewrites a character wherever the conversion produced it, so a character belongs there only when its two forms are variants of each other in every word OpenCC produces it in. The audit lists every `STPhrases` / `STCharacters` entry whose s2tw output holds the character (`audit_variants.py`, kept outside the repo like `roundtrip.py`; OpenCC dictionaries of the installed package) and judges the preferred writing against Taiwan usage (教育部《重編國語辭典修訂本》/《異體字字典》 as far as known; where it could not be judged, OpenCC's own writing is kept).

| OpenCC writes | preferred | words that produce it | verdict |
|---|---|---|---|
| 註 | 注 | 56 phrases | **removed from the character table.** 註冊 註解 註釋 註明 註銷 備註 附註 批註 標註 腳註 註生娘娘 are written 註 in Taiwan; only 注定 / 命中注定 / 命運注定 / 五百姻緣天注定 (the user writes 注定 25 / 註定 0) are word rules now |
| 嚮 | 向 | 14 phrases | **removed.** 嚮導 嚮導公司/員/犬 嚮往 心嚮往之 嚮應 嚮慕 嚮邇 嚮晦 are OpenCC's (MOE) writing. Exceptions that write 向: 女生外向, 面向对象的技术, 面向对象语言 (the mainland term has no 嚮 in Taiwan), 向明 (it matches inside 迈向明天) |
| 臺 | 台 | 856 phrases + the character 台 | kept. 台 is the common variant of 臺 in every word below (Taiwan names and organisations 263, broadcast stations 60, other 533: 舞台 平台 讲台 后台 阳台 灯台 and classical or proper-name words). No word needs an exception: the user's own count is 台 17 / 臺 3 |
| 溼 | 濕 | 14 phrases (干湿 风湿症 湿度表 ...) | kept, 濕 is the standard form in all of them; no exception |
| 汙 | 污 | 9 phrases (污蔑 脏污 奸污 ...) | kept, 污 is in common use in all of them; no exception |
| 燬 | 毀 | 10 phrases | kept for 烧毁 焚毁 炸毁 销毁 熔毁 禁毁 禁毁书. **Exceptions (OpenCC's 燬 kept): 毁炎, 毁犀 (classical phrases), 四库禁毁书丛刋 (a book series title)** |
| 絃 | 弦 | 32 phrases (和弦 琴弦 心弦 管弦 弦乐 ...) | kept, 弦 is the standard form in all of them; no exception |
| 悽 | 淒 | 25 phrases (凄凉-type words: 凄惨 凄美 凄楚 ...) | kept, 淒 is in common use in all of them; no exception |
| 歎 | 嘆 | 19 phrases (叹为观止 赞叹 惊叹 咏叹 ...) | kept, 嘆 is in common use in all of them; no exception |
| 唸 | 念 | 34 phrases (念书 念经 默念 叨念 ...) | kept, 念 is in common use in all of them; no exception |

Two more things the audit found and fixed:
- Removing 註 / 嚮 from the character table exposed OpenCC phrases that match across a word boundary: `关注明天` took 注明 (-> 關註明天), `迈向明天` took 向明 (-> 邁嚮明天; three lines of the user's lyrics). The common words that end in 注 or 向 (关注 专注 倾注 ... 方向 走向 迈向 ...) are listed in `PHRASE_OVERRIDES` so that the phrase cannot take their last character.
- `干了` was a phrase override to 乾了 (a mistake for 他干了一件大事). It is gone; "dried" is only listed where it is meant (干了就没事, 喉咙干了, 眼泪干了 ...). The segmentation score has a third key, the length of the longest phrase (干了这杯酒 is the four-character phrase, not 干了 + 杯酒), and `头发现在` is a phrase of its own.

Round trip after the change (same corpus, same method as above): lines with a difference 24, character differences 26, **wrong meaning 1, variant 14**, Japanese / not Traditional input 11, i.e. the same 26 pairs as before (`diff` of the two outputs is empty). A comparison of the old and the new converter over all 49051 `STPhrases` keys: 65 differ, all of them 註 -> 注 words (52), 嚮 -> 向 words (9), the three 燬 words and 干了; over the 7174 kana-free lines of the corpus (after `t2s`): 0 differ.

