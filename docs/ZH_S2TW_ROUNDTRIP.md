# s2tw: stock OpenCC plus guarded rules

`scripts/zh_traditional.py` converts Simplified to Traditional (Taiwan) for the review page. **With every rule table empty its output is character for character the stock OpenCC `s2tw` output**; each difference from stock comes from one rule that has a reason and a positive and a negative example, all run by `scripts/test_zh_convert.py`.

## Engine

The same algorithm as `opencc-python-reimplemented`'s `s2tw` (read from the installed package), run on our own tables: split at separators; in each clause take the longest phrase of `STPhrases`, the leftmost one among equally long ones, and convert the two sides the same way; characters no phrase covers go through `STCharacters`; then `TWVariants`. There is no scoring, no forward / backward comparison and no priority: a rule word is an ordinary dictionary word, so it only wins by being longer or further left. `convert_marked` still returns the marks (position and options of every character `STCharacters` maps to several forms) for the page.

## The four kinds of rule

1. `ENTRIES`: a word added to or replaced in the dictionary. Kept in two lists: `CORRECTIONS` (stock is wrong) and `PREFERENCES` (both forms are standard; the user's lyrics use this one).
2. `DROPPED_PHRASES`: a stock phrase that is not used, because it swallows the first character of the next word (一起干杯 is 一起+干杯, stock's 一起干 hides 干杯).
3. `CONTEXT_RULES`: a regular expression over the Simplified clause that fixes a span and its Traditional form where only the surrounding text can tell (a fixed span is a boundary: no phrase crosses it). A clause ends at a punctuation mark, so a rule cannot look at a colon or a quotation mark behind it.
4. `VARIANT_PREFERENCE`: a character-level variant choice applied to the characters the conversion produced (never to Traditional text already in the line).

Totals: 21 corrections, 26 preferences, 10 dropped phrases, 16 context rules, 8 variant characters.

## Rules and their reasons

Every entry lists the look-alike that must not change (negative example); the context rules' and dropped phrases' positive and negative sentences are in the table `CONTEXT_RULES` / `DROPPED_NOTES` and run by `test_zh_convert.py`.

### a1. Corrections of stock OpenCC (21)

| word | written | reason | negative example |
|---|---|---|---|
| 一直 | 一直 | not a stock phrase: 直发 (直髮) swallows the 发 of 一直\|发现 / 一直\|发光 | 直发 -> 直髮 |
| 一头 | 一頭 | not a stock phrase: 头发 (頭髮) swallows the 发 of 一头\|发光 | 长头发 -> 長頭髮 |
| 风后来 | 風後來 | 风后 is the name 風后 (kept); 风+后来 is wind + 'later' | 风后 -> 風后 |
| 风后面 | 風後面 | 风后 is the name 風后 (kept); 风+后面 is wind + 'behind' | 风后 -> 風后 |
| 回归线 | 回歸線 | Taiwan writes 回歸線 (tropic); stock gives 迴歸線 | 线性回归 -> 線性迴歸 |
| 干了啥 | 幹了啥 | stock's phrase 天干 (heavenly stem) swallows the 干 of 今天\|干了啥 | 池塘干了啥都不剩 -> 池塘乾了啥都不剩 |
| 书签名 | 書簽名 | 书签名 = signing a book (签名), not the bookmark 书签 of the preference below | 书签 -> 書籤 |
| 了如列眉 | 瞭如列眉 | idiom: stock only has 了如指掌 and the dropped 了如 phrase leaves 了如列眉 unconverted (0 / 1 in the Taiwan corpus) | 我明白了如何做 -> 我明白了如何做 |
| 周几 | 週幾 | week + how many ('which day of the week'): Taiwan writes 週幾; stock writes 周幾 (9 / 0 in the Taiwan corpus) | 周末 -> 週末 |
| 游标 | 游標 | cursor / vernier: 游標; stock writes 遊標 (12 / 0 in the Taiwan corpus) | 旅游 -> 旅遊 |
| 控制台 | 控制台 | console: Taiwan writes 控制台; stock's 控制檯 is the table-like 檯 (24 / 0 in the Taiwan corpus) | 台灯 -> 檯燈 |
| 老么 | 老么 | 么 is the Traditional character in 老么 / 么儿 / 么妹 / 么弟 / 么女 (youngest child); 么 after 什 怎 这 那 多 要 甚 is 麼 | 什么 -> 什麼 |
| 么儿 | 么兒 | youngest son: 么 is Traditional here | 什么 -> 什麼 |
| 么妹 | 么妹 | youngest sister: 么 is Traditional here | 什么 -> 什麼 |
| 么弟 | 么弟 | youngest brother: 么 is Traditional here | 什么 -> 什麼 |
| 么女 | 么女 | youngest daughter: 么 is Traditional here | 什么 -> 什麼 |
| 面向对象的技术 | 面向對象的技術 | Taiwan does not write 面嚮; stock turns 向 into 嚮 here | 心向往之 -> 心嚮往之 |
| 面向对象语言 | 面向對象語言 | Taiwan does not write 面嚮; stock turns 向 into 嚮 here | 心向往之 -> 心嚮往之 |
| 毁炎 | 燬炎 | classical words keep 燬 (the character preference to 毀 does not apply) | 烧毁 -> 燒毀 |
| 毁犀 | 燬犀 | classical words keep 燬 (the character preference to 毀 does not apply) | 烧毁 -> 燒毀 |
| 四库禁毁书丛刋 | 四庫禁燬書叢刋 | a book series title written with 燬 | 烧毁 -> 燒毀 |

### a2. Preferences (both forms standard; the user's lyrics use this one) (26)

| word | written | reason | negative example |
|---|---|---|---|
| 糊里糊涂 | 糊裡糊塗 | both are standard (the dictionary headword itself is 糊里糊塗); the lyrics write 糊裡糊塗 | 糊里 -> 糊裡 |
| 克制 | 克制 | both are standard (剋制 for 相剋 senses); the lyrics write 克制 (restrain), stock gives 剋制 | 克星 -> 剋星 |
| 女生外向 | 女生外向 | both are standard (外嚮 / 外向); the lyrics write 外向, stock turns 向 into 嚮 here | 心向往之 -> 心嚮往之 |
| 计划 | 計畫 | both are standard; 计划 -> 計畫 4 / 計劃 0 in the lyrics | 合计划拨 -> 合計劃撥 |
| 了解 | 了解 | both are standard; 了解 4 / 瞭解 1 | 了如指掌 -> 瞭如指掌 |
| 点赞 | 點讚 | both are standard; 點讚 1 / 點贊 0 | 赞美 -> 讚美 |
| 胡同 | 胡同 | both are standard; 胡同 1 / 衚衕 0 | 胡子 -> 鬍子 |
| 死胡同 | 死胡同 | both are standard; 死胡同 1 / 死衚衕 0 | 胡子 -> 鬍子 |
| 尽可能 | 盡可能 | both are standard; 盡可能 1 / 儘可能 0 | 尽管 -> 儘管 |
| 游动 | 游動 | swimming, not touring: 游動 7 / 遊動 0 | 旅游动态 -> 旅遊動態 |
| 游过 | 游過 | swam across, not toured: 游過 / 遊過 | 旅游过程 -> 旅遊過程 |
| 游移 | 游移 | drifting, not touring: 游移 / 遊移 | 旅游移民 -> 旅遊移民 |
| 背包 | 背包 | both are standard; 背包 29 / 揹包 1 | 背叛 -> 背叛 |
| 背负 | 背負 | both are standard; 背負 / 揹負 | 背叛 -> 背叛 |
| 回归 | 回歸 | both are standard; 回歸 1 / 迴歸 0 in the lyrics | 线性回归 -> 線性迴歸 |
| 回归分析 | 迴歸分析 | carve-out of 回归: statistics writes 迴歸 | 回归线 -> 回歸線 |
| 线性回归 | 線性迴歸 | carve-out of 回归: statistics writes 迴歸 | 回归线 -> 回歸線 |
| 逻辑回归 | 邏輯迴歸 | carve-out of 回归: statistics writes 迴歸 | 回归线 -> 回歸線 |
| 回归模型 | 迴歸模型 | carve-out of 回归: statistics writes 迴歸 | 回归线 -> 回歸線 |
| 回归方程 | 迴歸方程 | carve-out of 回归: statistics writes 迴歸 | 回归线 -> 回歸線 |
| 熏染 | 燻染 | both are standard; 燻染 1 / 薰染 0 | 熏陶 -> 薰陶 |
| 周而复始 | 周而復始 | both are standard; 周而復始 3 / 週而復始 0 | 周末 -> 週末 |
| 注定 | 注定 | both are standard; 注定 25 / 註定 0 in the lyrics (註冊 註解 備註 are 註 in Taiwan and stay stock) | 注册 -> 註冊 |
| 命中注定 | 命中注定 | both are standard; 注定 25 / 註定 0 in the lyrics | 注册 -> 註冊 |
| 命运注定 | 命運注定 | both are standard; 注定 25 / 註定 0 in the lyrics | 注册 -> 註冊 |
| 五百姻缘天注定 | 五百姻緣天注定 | both are standard; 注定 25 / 註定 0 in the lyrics | 注册 -> 註冊 |

### b. Dropped stock phrases (10)

| phrase | reason (positive -> result; negative -> result) |
|---|---|
| 一起干 | 一起干杯 -> 一起乾杯; 一起干活 -> 一起幹活 |
| 了如 | 我明白了如何做 -> 我明白了如何做; 了如指掌 -> 瞭如指掌 |
| 先尝 | 先尝试 -> 先嘗試; 尝试 -> 嘗試 |
| 如干 | 倒不如干脆 -> 倒不如乾脆; 若干 -> 若干 |
| 并产 | 并产生 -> 並產生; 合并 -> 合併 |
| 并包 | 并包含 -> 並包含; 合并 -> 合併 |
| 心向 | 内心向往 -> 內心嚮往; 心向党 -> 心向黨 |
| 有云 | 唯有云雀 -> 唯有雲雀; 今天有云 -> 今天有雲 |
| 短几 | 短短几秒 -> 短短幾秒; 茶几 -> 茶几 |
| 被发 | 被发送 -> 被發送; 被发现 -> 被發現 |

### c. Context rules (16)

| name | why |
|---|---|
| sea | 海里 after a numeral is the nautical mile (stays 海里), and so is a clause that is just 海里 (a headword); anywhere else it is 海+里 'in the sea' -> 海裡 |
| dry-drink | a cup / wine / soup / bowl / bottle (or 一口 一饮) within four characters before 干了 is a drink finished: 乾了 |
| dry-verb | 喝干 晾干 晒干 流干 烧干 熬干 风干, and 汤 / 碗 / 喉咙 / 嘴 / 口 / 泪 / 眼泪 / 衣服 + 干了, are dried: 乾了 (stock writes 熬幹了, 喉嚨幹了, 衣服幹了) |
| dry-toast | 干了这杯 / 干了杯 / 干了就没事: the drink or the thirst is gone: 乾了 |
| dry-place | a river / pond / well / spring / water word right before 干了 is dried up (熬干了 is boiled dry): 乾了 |
| zhi-demonstrative | stock reads 这只 as 這隻 (measure word); before 有 能 会 想 要 the user's lyrics mean 'only' (這只), unless the clause has 的 + an animal noun (這隻會飛的鳥) |
| zhi-adverb | 只 right after 有 别 是 and before 有 能 会 想 要 是 在 is 'only' (没有只要 别只是 总是只有), 隻 when the clause has 的 + an animal noun (是只有毒的蛇); stock's phrases 有只 / 别只 / 是只 always read it as 隻. Anywhere else stock is kept: 只能 / 只有 are stock phrases, and a rule that fixed their 只 would hide the next word (只能\|干呕) |
| said-yun | 有云 right after 人 子 书 诗 易 经 谚 语 传 史 》 is the classical 'it is said' (古人有云, 孔子有云, 兵书有云); the dropped stock phrase would otherwise turn it into weather 有雲 (the clause ends at the colon, so only the word before can tell) |
| loose-hair | 被发 not followed by 送 现 生 布 表 展 起 觉 动 明 挥 达 言 is 'with loose hair' (被髮, stock): 被发乱首, 被发而为狂; the stock phrase is dropped because it swallows 被\|发送 / 被\|发现 |
| wind-after | 风后 after 的 了 着 吹 ... is 'after the wind' (風後); alone it is the name 風后 (stock) |
| dry-compound | 没干透 / 没干净: dried (乾); stock's phrase 没干 -> 沒幹 swallows the 干 |
| curl | 卷卷 + 而 的 地 着 起 成 了 = curling (捲捲而上, 捲捲的头发); 卷卷 alone, or before another word, is 卷卷 (volumes, 惓惓: stock); after 试 长 画 书 手 经 压 考 答 the first 卷 belongs to the word before (试卷\|卷面) |
| pine | after 青 古 老 孤 苍 劲 乔 雪 the 松 is a pine tree and the 开 starts the next word (青松\|开花); stock's phrase 松开 -> 鬆開 hides it |
| tour | after 旅 漫 导 周 巡 夜 远 出 畅 郊 the 游 is 遊 (tour) and 动 过 移 starts the next word; the preference 游动/游过/游移 (swim) would take it |
| scrape | after 胡 须 脸 毛 腿 头 眉 the 刮 is shave / scrape (刮); stock's phrase 刮了 is the wind 颳了 |
| surname-yu | the surname 余 at the start of a clause (or after 姓 叫 是 找 ...) before a title stays 余; stock writes 餘 |

### d. Variant preference (8)

臺->台, 溼->濕, 汙->污, 燬->毀, 絃->弦, 悽->淒, 歎->嘆, 唸->念


## Measurements (final HEAD)

* **Empty rules = stock** (`tools/g1_all.py`, rules emptied): every `STPhrases` word (49051), the 530 sentences of the adversarial set, the 7174 kana-free lyric lines of the user's corpus (after `t2s`): 0 differences from `OpenCC("s2tw")`. `test_zh_convert.py` repeats it for all `STPhrases` words plus a fixed sample.
* **Adversarial set** (530 sentences, `run_cases.py`): stock-right-new-wrong 0, both wrong 0, stock-wrong-new-right 59.
* **User's lyrics round trip** (169 songs, 6628 lines, 66860 Han characters): wrong meaning 1, variant 14, Japanese 11, same output as before this change.
* **Taiwan dictionary corpus round trip** (MOE 《重編國語辭典修訂本》 as published in g0v/moedict-data `dict-revised.json.xz`: 656935 lines of headwords, definitions and examples, 8681563 Han characters; Traditional -> `tw2s` -> ours and stock `s2tw` -> compared with the source; test data only, not in the repo): positions where stock is right and ours wrong 6789, of which 6781 come from the *preference* rules (臺->台 4648 and the like: the dictionary is Taiwan-official and the user's own lyrics use the other form) and **8 from correction rules**, all of them one case: 有云 in classical quotations that no word before it identifies (古人有云 is handled; 署名有云臨川, 各有云設 are not). Positions where ours is right and stock wrong: 2075. Both wrong: 48620 (mostly 出->齣 in 〇出, 卷 sections, 分佈, names; they need a dictionary, not a rule).

| correction / context / drop rule | lines | helped positions | hurt positions | neutral lines |
|---|---|---|---|---|
| drop:有云 | 26 | 18 | 8 | 0 |
| ctx:sea | 25 | 25 | 0 | 0 |
| drop:并产 | 16 | 16 | 0 | 0 |
| ctx:zhi-adverb | 12 | 12 | 0 | 0 |
| drop:并包 | 10 | 5 | 0 | 6 |
| drop:先尝 | 9 | 9 | 0 | 0 |
| ctx:dry-compound | 5 | 5 | 0 | 0 |
| entry:游标 | 4 | 4 | 0 | 0 |
| drop:短几 | 3 | 3 | 0 | 0 |
| entry:一直 | 2 | 2 | 0 | 0 |
| drop:心向 | 2 | 2 | 0 | 0 |
| entry:回归线 | 2 | 3 | 0 | 0 |
| entry:控制台 | 2 | 0 | 0 | 2 |
| ctx:zhi-demonstrative | 2 | 2 | 0 | 0 |
| drop:如干 | 2 | 3 | 0 | 0 |
| entry:书签名 | 1 | 1 | 0 | 0 |
| drop:了如 | 1 | 1 | 0 | 0 |
| ctx:surname-yu | 1 | 1 | 0 | 0 |

Preference rules (both forms standard in Taiwan; the user's lyrics decide; hurt counts are expected):

| rule | lines | helped positions | hurt positions | neutral lines |
|---|---|---|---|---|
| variant:臺 | 4291 | 43 | 4648 | 36 |
| variant:汙 | 871 | 5 | 900 | 0 |
| variant:溼 | 711 | 131 | 616 | 3 |
| variant:絃 | 454 | 397 | 87 | 2 |
| variant:悽 | 403 | 260 | 199 | 6 |
| entry:计划 | 339 | 268 | 75 | 0 |
| variant:唸 | 260 | 224 | 55 | 2 |
| entry:了解 | 255 | 216 | 37 | 2 |
| variant:歎 | 158 | 106 | 54 | 0 |
| variant:燬 | 69 | 45 | 25 | 0 |
| entry:背负 | 56 | 56 | 0 | 0 |
| entry:注定 | 36 | 19 | 16 | 1 |
| entry:糊里糊涂 | 34 | 26 | 8 | 0 |
| entry:命中注定 | 29 | 20 | 9 | 0 |
| entry:回归 | 28 | 27 | 3 | 0 |
| entry:胡同 | 22 | 28 | 22 | 0 |
| entry:周而复始 | 21 | 18 | 3 | 0 |
| entry:克制 | 15 | 13 | 2 | 0 |
| entry:游动 | 14 | 12 | 2 | 0 |
| entry:尽可能 | 11 | 9 | 2 | 0 |
| entry:背包 | 9 | 10 | 0 | 0 |
| entry:熏染 | 9 | 0 | 0 | 9 |
| entry:游移 | 7 | 7 | 0 | 0 |
| entry:游过 | 5 | 3 | 2 | 0 |
| entry:女生外向 | 4 | 1 | 3 | 0 |
| entry:死胡同 | 4 | 8 | 0 | 0 |
| entry:五百姻缘天注定 | 2 | 0 | 2 | 0 |

## Known single-sentence undecidables

* 有云: classical "it is said" (有云：「…」) and weather 有雲 are written the same way once the colon is cut off; only a word before it (人 子 书 诗 易 经 谚 语 传 史) is used.
* 船只能 (a boat can only) vs 船只 + 能 (vessels): stock 船隻能 is kept.
* 髮 -> 發 inside a proper name written with 后+发 in the lyrics (the one remaining wrong meaning of the lyrics round trip): no rule can tell it from 发现.
* The page still marks every character with several Traditional forms, so the user can pick by hand.

## Round trip tables (BH2-FIX6 / FIX9, same corpus and method)

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

## Notes carried over from the BH2-FIX9 audit

The `注 / 註` and `向 / 嚮` words are not in the character table (rules above and the stock words win); the three 燬 exceptions (毁炎 毁犀 四库禁毁书丛刋) are entries.
