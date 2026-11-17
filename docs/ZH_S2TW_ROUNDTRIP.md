# s2tw: stock OpenCC plus guarded rules

`scripts/zh_traditional.py` converts Simplified to Traditional (Taiwan) for the review page. **With every rule table empty its output is character for character the stock OpenCC `s2tw` output**; each difference from stock comes from one rule that has a reason and a positive and a negative example, all run by `scripts/test_zh_convert.py`.

## Engine

The same algorithm as `opencc-python-reimplemented`'s `s2tw` (read from the installed package), run on our own tables: split at separators; in each clause take the longest phrase of `STPhrases`, the leftmost one among equally long ones, and convert the two sides the same way; characters no phrase covers go through `STCharacters`; then `TWVariants`. There is no scoring, no forward / backward comparison and no priority: a rule word is an ordinary dictionary word, so it only wins by being longer or further left. `convert_marked` still returns the marks (position and options of every character `STCharacters` maps to several forms) for the page.

## The kinds of rule (BH2-FIX11A: closed contexts only; BH2-FIX13A: whole lines, preferences keep the stock segmentation)

The default is stock. A difference from stock is allowed in three forms only:

1. **Literal word** (`ENTRIES`): a fixed string longer than the stock word it replaces; the longest match wins, so its reach is the string itself. Two lists: `CORRECTIONS` (stock is wrong) and `PREFERENCES` (both forms are standard; the user's lyrics use this one).
2. **Closed context rule** (`CONTEXT_RULES`): a regular expression whose left and right sides are both lists of literal words. Not allowed: a gap (`.{m,n}`), a "not followed by" guard as the only protection, a one-character left or right context that can be the tail or head of another word without a word boundary, a noun list that decides a default. A rule that a reviewer breaks once is deleted, not patched.
3. **Preference** (`VARIANT_PREFERENCE`, and the `PREFERENCES` entries): either spelling is standard, the user's own lyrics decide.

`DROPPED_PHRASES` (a stock phrase that is not used) stays only where the Taiwan dictionary corpus has no sentence that needs the stock phrase (victims 0); every other stock phrase that hides a word is covered by a longer literal word instead (短短几, 一起干杯, 被发送 ...).

4. **Whole line** (`WHOLE_LINES`, BH2-FIX13A): a table of a whole simplified line to its whole Traditional line, applied only when the line (white space at both ends aside) equals the key. The same words inside a longer line, after a comma or with one more character in front go the normal way. Used where a lyric line stands alone and no literal word can be written for it without breaking on a longer sentence.

**A preference never takes a character away from the stock reading (BH2-FIX13A).** A `PREFERENCES` word that stock also has (计划, 了解, 回归, 注定 ...) only replaces the value: the segmentation is stock's. Where, with the word left out, a stock phrase crosses the border of the word (预计|划分, 估计|划算) the word keeps the stock value (預計劃分, 估計劃算); 计划 on its own is still 計畫. A literal phrase that stock does not have (点赞, 游动着 ...) is left out of the segmentation where a stock phrase crosses its border (点赞同 is 點贊同, 旅游动着 is 旅遊動著). Classes of the 30 preferences: 17 value replacements (糊里糊涂 克制 女生外向 计划 了解 胡同 死胡同 尽可能 背包 背负 回归 熏染 周而复始 注定 命中注定 命运注定 五百姻缘天注定) and 13 literal phrases (鱼类在游动 正游动在 正游动其中 交织游动吗 游动着 游动著 游过群星之间 点赞, and the five 回归 carve-outs 回归分析 线性回归 逻辑回归 回归模型 回归方程); every one has a border example in `scripts/test_zh_fix13a.py`.

Totals: 25 corrections, 30 preferences, 3 whole lines, 6 dropped phrases, 1 context rule, 8 variant characters.

## Rules and their reasons

Every entry lists the look-alike that must not change (negative example); `scripts/test_zh_rules_r3.py` has for every non-preference word and every context rule two sentences where it fires and two look-alikes; `scripts/test_zh_edges_r4.py` (also `cases_edges.tsv`) attacks both ends of every non-preference word with 3 left and 3 right extensions (hand-written Traditional) where it must not (the look-alikes use a different word with the same ending or beginning).

### a1. Corrections of stock OpenCC (25)

| word | written | reason | negative example |
|---|---|---|---|
| 一直发现着 | 一直發現著 | stock's 直发 (直髮) swallows the 发 of 一直\|发现; the word ends in 着 so 一直发现在 (hair + now) stays stock | 梳着一直发 -> 梳著一直髮 |
| 一直发现著 | 一直發現著 | the same word written with 著, which is what a Traditional-to-Simplified round trip leaves | 梳着一直发 -> 梳著一直髮 |
| 短短几 | 短短幾 | stock's phrase 短几 (a low table) swallows 短短\|几秒 | 一张短几 -> 一張短几 |
| 一起干杯 | 一起乾杯 | stock's phrase 一起干 (一起幹) swallows 一起\|干杯 | 一起干活 -> 一起幹活 |
| 没干透吧 | 沒乾透吧 | stock's phrase 没干 (沒幹) swallows the 干 of 没\|干透 (not dry yet); the word ends in 吧 so 没干透露 stays stock | 他没干坏事 -> 他沒幹壞事 |
| 这杯干了 | 這杯乾了 | this cup / drink is finished (stock 幹了); the word is the whole string, a 这杯 elsewhere stays stock | 他把这件事干了吧 -> 他把這件事幹了吧 |
| 若是只有 | 若是只有 | 若是 + 只有 = 'if only'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 总是只有 | 總是只有 | 总是 + 只有 = 'always only'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 总是只在 | 總是只在 | 总是 + 只在 = 'always only in'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 还是只想 | 還是只想 | 还是 + 只想 = 'still only want'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 没有只要 | 沒有只要 | 没有 + 只要 = 'without just asking'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 别只是 | 別只是 | 别 + 只是 = 'don't just'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 卷卷而上 | 捲捲而上 | curling upward (捲捲而上); stock leaves 卷卷 | 问卷卷成了筒状 -> 問卷捲成了筒狀 |
| 湿润的风后 | 濕潤的風後 | wet air after the wind (風後); 风后 alone stays the name 風后 (stock) | 黄帝的大臣风后 -> 黃帝的大臣風后 |
| 闪烁的星海里 | 閃爍的星海裡 | in the sea of stars (stock keeps 海里, the nautical mile); the word starts with 闪烁的 so 十二星海里 is not touched | 多少海里才到岸 -> 多少海里才到岸 |
| 被发送 | 被發送 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发送 | 被发乱首 -> 被髮亂首 |
| 被发生 | 被發生 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发生 | 被发乱首 -> 被髮亂首 |
| 被发布 | 被發布 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发布 | 被发乱首 -> 被髮亂首 |
| 被发表 | 被發表 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发表 | 被发乱首 -> 被髮亂首 |
| 了如列眉 | 瞭如列眉 | idiom: stock only has 了如指掌 and the dropped 了如 phrase leaves 了如列眉 unconverted (0 / 1 in the Taiwan corpus) | 我明白了如何做 -> 我明白了如何做 |
| 唯有云雀的声音 | 唯有雲雀的聲音 | 唯有\|云雀 (a lark); stock's classical phrase 有云 ('it is said') would hide it; the word is the whole phrase so 有云雀跃 stays stock | 古人有云雀跃不已 -> 古人有云雀躍不已 |
| 不是只有一次 | 不是只有一次 | 不是 + 只有 + 一次 = 'not only once' (stock reads 是只 as 是隻) | 那不是只有斑点的狗吗 -> 那不是隻有斑點的狗嗎 |
| 这杯酒我干了 | 這杯酒我乾了 | 'I drain this wine': stock 幹了; the word is the whole phrase | 卖酒我干了十年 -> 賣酒我幹了十年 |
| 如果喉咙干了 | 如果喉嚨乾了 | if the throat is dry (stock 喉嚨幹了); the word is the whole phrase, so 喉咙干了坏事 stays stock | 捂着喉咙干了坏事 -> 捂著喉嚨幹了壞事 |
| 四库禁毁书丛刋 | 四庫禁燬書叢刋 | a book series title written with 燬 (the variant preference 毀 does not apply) | 烧毁 -> 燒毀 |

### a2. Preferences (both forms standard; the user's lyrics use this one) (30)

| word | written | reason | negative example |
|---|---|---|---|
| 鱼类在游动 | 魚類在游動 | both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics | 旅游动态 -> 旅遊動態 |
| 正游动在 | 正游動在 | both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics | 我们正游动物园 -> 我們正遊動物園 |
| 正游动其中 | 正游動其中 | both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics | 我们正游动物园 -> 我們正遊動物園 |
| 交织游动吗 | 交織游動嗎 | both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics | 我们交织游动物园 -> 我們交織遊動物園 |
| 游动着 | 游動著 | both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics | 旅游动态 -> 旅遊動態 |
| 游动著 | 游動著 | both are standard; 游動 in the lyrics, stock 遊動; a phrase from the lyrics (著 is what the Traditional-to-Simplified round trip leaves) | 旅游动态 -> 旅遊動態 |
| 游过群星之间 | 游過群星之間 | both are standard; 游過 (swam) in the lyrics, stock 遊過; a phrase from the lyrics | 旅游过程 -> 旅遊過程 |
| 糊里糊涂 | 糊裡糊塗 | both are standard (the dictionary headword itself is 糊里糊塗); the lyrics write 糊裡糊塗 | 糊里 -> 糊裡 |
| 克制 | 克制 | both are standard (剋制 for 相剋 senses); the lyrics write 克制 (restrain), stock gives 剋制 | 克星 -> 剋星 |
| 女生外向 | 女生外向 | both are standard (外嚮 / 外向); the lyrics write 外向, stock turns 向 into 嚮 here | 心向往之 -> 心嚮往之 |
| 计划 | 計畫 | both are standard; 计划 -> 計畫 4 / 計劃 0 in the lyrics | 合计划拨 -> 合計劃撥 |
| 了解 | 了解 | both are standard; 了解 4 / 瞭解 1 | 了如指掌 -> 瞭如指掌 |
| 点赞 | 點讚 | both are standard; 點讚 1 / 點贊 0 | 赞美 -> 讚美 |
| 胡同 | 胡同 | both are standard; 胡同 1 / 衚衕 0 | 胡子 -> 鬍子 |
| 死胡同 | 死胡同 | both are standard; 死胡同 1 / 死衚衕 0 | 胡子 -> 鬍子 |
| 尽可能 | 盡可能 | both are standard; 盡可能 1 / 儘可能 0 | 尽管 -> 儘管 |
| 背包 | 背包 | both are standard; 背包 29 / 揹包 1 | 背叛 -> 背叛 |
| 背负 | 背負 | both are standard; 背負 / 揹負 | 背叛 -> 背叛 |
| 回归 | 回歸 | both are standard; 回歸 1 / 迴歸 0 in the lyrics | 线性回归 -> 線性迴歸 |
| 回归分析 | 迴歸分析 | carve-out of 回归: statistics writes 迴歸 | 香港回归 -> 香港回歸 |
| 线性回归 | 線性迴歸 | carve-out of 回归: statistics writes 迴歸 | 香港回归 -> 香港回歸 |
| 逻辑回归 | 邏輯迴歸 | carve-out of 回归: statistics writes 迴歸 | 香港回归 -> 香港回歸 |
| 回归模型 | 迴歸模型 | carve-out of 回归: statistics writes 迴歸 | 香港回归 -> 香港回歸 |
| 回归方程 | 迴歸方程 | carve-out of 回归: statistics writes 迴歸 | 香港回归 -> 香港回歸 |
| 熏染 | 燻染 | both are standard; 燻染 1 / 薰染 0 | 熏陶 -> 薰陶 |
| 周而复始 | 周而復始 | both are standard; 周而復始 3 / 週而復始 0 | 周末 -> 週末 |
| 注定 | 注定 | both are standard; 注定 25 / 註定 0 in the lyrics (註冊 註解 備註 are 註 in Taiwan and stay stock) | 注册 -> 註冊 |
| 命中注定 | 命中注定 | both are standard; 注定 25 / 註定 0 in the lyrics | 注册 -> 註冊 |
| 命运注定 | 命運注定 | both are standard; 注定 25 / 註定 0 in the lyrics | 注册 -> 註冊 |
| 五百姻缘天注定 | 五百姻緣天注定 | both are standard; 注定 25 / 註定 0 in the lyrics | 注册 -> 註冊 |

### a3. Whole lines (3)

| Simplified line | Traditional | Reason | Same words in a longer line |
|---|---|---|---|
| 干了就没事了！ | 乾了就沒事了！ | a lyric line that stands alone (stock 幹了); a literal word fails 这份作业，干了就没事了 | 这份作业，干了就没事了！ -> 這份作業，幹了就沒事了！ |
| 干了就没事了 | 乾了就沒事了 | the same line without its closing mark | 这份作业，干了就没事了 -> 這份作業，幹了就沒事了 |
| 海里有各种鱼类在游动 | 海裡有各種魚類在游動 | a lyric line that stands alone (stock keeps 海里, the nautical mile) | 方圆十海里有各种鱼类在游动 -> 方圓十海里有各種魚類在游動 |

All of convert_text, convert_marked (marks at the same positions), song_candidates, convert_line / convert_header and so the page's one-click conversion take this path; the tests clear the table with the other rule tables (empty = stock).

### b. Dropped stock phrases (6)

| phrase | reason (positive -> result; negative -> result) |
|---|---|
| 了如 | 我明白了如何做 -> 我明白了如何做; 了如指掌 -> 瞭如指掌 |
| 先尝 | 先尝试 -> 先嘗試; 尝试 -> 嘗試 |
| 如干 | 倒不如干脆 -> 倒不如乾脆; 若干 -> 若干 |
| 并产 | 并产生 -> 並產生; 合并 -> 合併 |
| 并包 | 并包含 -> 並包含; 合并 -> 合併 |
| 心向 | 内心向往 -> 內心嚮往; 心向党 -> 心向黨 |

### c. Context rules (1)

| name | why |
|---|---|
| surname-yu | the surname 余 at the start of a clause (or right after 姓 叫 找 问 请) and right before a title that ends the word stays 余; stock writes 餘 |

### d. Variant preference (8)

臺->台, 溼->濕, 汙->污, 燬->毀, 絃->弦, 悽->淒, 歎->嘆, 唸->念

## Removed in BH2-FIX12A (REVIEW4 claude-10: 17 of 74 new sentences were stock right / ours wrong; termination clause of FIX11A)

* Context rules `tour`, `dry-throat`, `toast-start` deleted (not patched): `tour` took the 周 of 下周游动物园, `dry-throat` and `toast-start` fired on a clause that only starts with the same characters. No replacement rule.
* Entries deleted because their edges beat stock words: 酒我干了 (卖酒我干了十年), 海里的鱼 and 海里有各种 (十海里的鱼), 周几 (老周几点来, wins a same-length leftmost tie against 几点), 不是只有 (那不是只有斑点的狗), 新书签名 (新书签名贵), 风后来 (黄帝命风后来此), 没干透 (他没干透露机密的事, replaced by 没干透吧), 有云雀 (古人有云雀跃, replaced by 唯有云雀的声音), 星海里 (replaced by 闪烁的星海里), 一直发现 (梳着一直发现在很流行, replaced by 一直发现着 / 著), 不是只有 (replaced by 不是只有一次), and every other word that no lyric line, G5 or G6 needs: 一直发X (17 more), 一头发光/亮, 风后面, 回归线, 今天/昨天/明天干了啥, 图书签名, 没干净/燥, 熬干了, 嘴巴干了, 在/大/到/入海里, 有云X and 天上/天空/今天/明天有云, 古松开, 被发X (24 of the 28; kept 被发送 被发生 被发布 被发表), 游标 and 游标卡尺, 控制台, 周几, 面向对象的技术/语言, 毁炎, 毁犀, 老么 么儿 么妹 么弟 么女 (a 2-character word that wins a same-length tie against the word to its right).
* Preference `游过` (guesses the sense: 我游过很多国家 is 遊過) and `游动` / `游移` (游动物园 against 动物) deleted; the lyric phrases that need them are literal preferences: 游过群星之间, 游动着 / 游动著, 正游动在, 正游动其中, 交织游动吗, 鱼类在游动.
* Added (longer, never shorter): 如果喉咙干了, 这杯酒我干了 (the full phrase of 酒我干了), 唯有云雀的声音, 不是只有一次, 闪烁的星海里, 没干透吧, 一直发现着 / 一直发现著.
* Kept with its own attacks because deleting it made the variant preference hurt stock: 四库禁毁书丛刋 (燬 -> 毀). The entry 统计划分 was removed in BH2-FIX13A: the preference rule above now gives 統計劃分 by itself.

## Removed in BH2-FIX11A (REVIEW4: 42 + 22 sentences where stock was right and the rules were wrong)

`sea` (海里), `dry-drink` / `dry-verb` / `dry-toast` / `dry-place` / `dry-compound` (干了), `zhi-demonstrative` / `zhi-adverb` (这只), `said-yun` (有云), `loose-hair` (被发), `wind-after` (风后), `curl` (卷卷), `pine` (松开), `scrape` (刮), the stock-phrase drops 一起干 / 短几 / 有云 / 被发, and the entries 一直 / 一头 / 干了啥 / 书签名 (each swallowed a longer stock word). What the user's lyrics or the dictionary corpus needed from them is a literal word now; the rest is stock.

## Known trade-offs (BH2-FIX13A)

The review sets of RV1d and RV2d (`s2tw_cases3.tsv`, 40 and 60 sentences) were written to break the literal words: each sentence below is a deliberately built sentence where the characters of a literal word sit next to a word of the other reading (千|总是|只有斑点的猎犬的主人, 居丧者皆被|发送|终). Any literal word can be broken that way, so these words are **not** deleted for such a sentence. The decision (2026-11-17): the user's lyrics come first; a literal word stays when the Taiwan dictionary corpus has no sentence it hurts (stock right, ours wrong: 0), the five everyday adversarial sets show no regression against stock, and the lyrics round trip holds. One sentence per word that breaks it (simplified -> ours, stock or the hand-written reading in brackets):

| Word | Sentence | Ours | Reading intended |
|---|---|---|---|
| 一起干杯 | 我们三个一起干杯子批发的生意 | 乾杯子 | 幹 (do the cup business) |
| 若是只有 | 若是只有毒的蜘蛛就别碰 | 只有 | 隻 |
| 还是只想 | 它还是只想家的小猫 | 只想 | 隻 |
| 不是只有一次 | 那不是只有一次生育机会的熊猫吗 | 只有 | 隻 |
| 总是只有 | 千总是只有斑点的猎犬的主人 | 只有 | 隻 (千总 is a rank) |
| 总是只在 | 千总是只在军营养大的犬 | 只在 | 隻 |
| 没有只要 | 窝里没有只要破壳的小鸡 | 只要 | 隻 |
| 别只是 | 这笼饲料送给别只是否合适 | 只是 | 隻 |
| 这杯干了 | 你对这杯干了什么 | 乾了 | 幹了 |
| 这杯酒我干了 | 这杯酒我干了什么手脚你也知道 | 乾了 | 幹了 |
| 如果喉咙干了 | 如果喉咙干了坏事就要受罚 | 乾了 | 幹了 |
| 没干透吧 | 大扫除没干透吧嗒响的门轴 | 沒乾透 | 沒幹 |
| 短短几 | 大殿内摆着一张短短几案 | 短短幾案 | 几案 (a low table) |
| 卷卷而上 | 侍从将画卷卷而上呈 | 畫捲捲而上 | 画卷|卷而上 |
| 一直发现着 / 一直发现著 | 她梳着一直发现着迷人的光泽 | 發現 | 直髮 |
| 闪烁的星海里 | 航向那颗闪烁的星海里数尚远 | 星海裡 | 海里 (nautical mile) |
| 被发送 / 被发生 / 被发布 | 居丧者皆被发送终 | 被發送 | 被髮 (loose hair; classical) |

What decides, in short: a word stays when it is longer than the stock word it replaces, was needed by the user's lyrics or by a Taiwan-corpus sentence, and the sentence that breaks it is a made-up one whose natural reading is the word. The measures are: MOE corpus victims 0 (outside the preferences), the five everyday sets 0 regressions, the 156 both-ends attacks of `test_zh_edges_r4.py`, the lyrics round trip. Preferences are never judged by such sentences once they keep the stock segmentation (above).

Whole lines (above) are the narrowest rule: only a line equal to the key; there is no longer sentence they can break.

## Known stock limitations (not regressions: stock is wrong too, a single sentence cannot tell)

* 这只 / 那只 + 有 能 会 想 要: stock reads the measure word (這隻); the adverb 這只 (这只有五分钟) is written the same, so stock stays except for the literal 总是只有 / 若是只有 / 总是只在 / 还是只想 / 没有只要 / 别只是 / 不是只有一次.
* 干了 as 乾了 (a drink or a river finished) vs 幹了 (did): stock 幹; only the literal 这杯干了, 这杯酒我干了 and 如果喉咙干了 are 乾了, and the whole line 干了就没事了.
* 有云 (classical 'it is said' vs weather): stock 有云 (classical) except 唯有云雀的声音.
* 海里 (nautical mile vs in the sea): stock 海里 except 闪烁的星海里 and the whole line 海里有各种鱼类在游动; 在海里 / 大海里 / 到海里 are 海里 again.
* 老么 / 么儿 / 么妹 / 么弟 / 么女 (youngest child): stock writes 麼; the kana gate still does not call them simplified.
* 旅游动态 is right (旅游 wins the leftmost tie) but 漫游过去 / 导游过来 are 漫游過 / 導游過 in stock; 周几 is 周幾; 回归线 is 迴歸線; 游标 is 遊標; 控制台 is 控制檯; 面向对象 is 面嚮對象; 今天干了啥 leaves 干 unconverted; 被发 + a verb other than 送 生 布 表 is 被髮 in stock when stock has no longer phrase; 新书签名 / 书签名 is 書籤名.
* 髮 -> 發 inside a proper name written with 后+发 in the lyrics (the one remaining wrong meaning of the lyrics round trip besides the two lines above).
* The page still marks every character with several Traditional forms, so the user can pick by hand.

## Measurements (final HEAD, BH2-FIX13A where marked; the rest as in FIX12A)

* **Empty rules = stock** (`tools/g1_all.py`): `STPhrases` 49051 words, the 530 adversarial sentences, 7174 lyric lines: 0 differences.
* **Adversarial sets** (`tools/run_cases.py`; stock right and ours wrong / both wrong / stock wrong and ours right): `cases.tsv` 530: 0 / 35 / 24 (FIX11A: 0 / 22 / 37); REVIEW4 gemini set 100: 0 / 27 / 2 (0 / 18 / 11); REVIEW4 claude set 112: 0 / 37 / 9 (0 / 26 / 20); REVIEW4 claude-10 set 74: 0 / 28 / 10 (17 regressions before); REVIEW5 gemini set 100: 0 / 45 / 12 (0 / 8 / 49 on the FIX11A HEAD by the reviewer). Fixed went down because the rules and entries that were deleted are stock now; no case became worse than stock.
* **Both-ends attacks** (`scripts/test_zh_edges_r4.py`, `cases_edges.tsv`): every non-preference word has 3 left and 3 right extensions with hand-written Traditional text: 156 sentences, 0 wrong.
* **User's lyrics round trip** (169 songs, 6628 lines; BH2-FIX13A): wrong meaning 1, variant 14, Japanese 11 (FIX12A: wrong 3, the two stand-alone lines now come back through the whole-line table).
* **Taiwan dictionary corpus** (MOE `dict-revised.json.xz`; `tools/g4.py`; BH2-FIX13A): 656935 lines, 8681563 Han characters: stock right and ours wrong 6765, of which not explained by preference rules **0**; ours right and stock wrong 1957; both wrong 48738 (`out13/g4_13a.txt`).
* Speed: 7174 lines, FIX9 0.56 s, now 0.40 s (BH2-FIX13A).
* **Adversarial sets, BH2-FIX13A** (regression / still wrong / fixed): 530: 0 / 38 / 21; RV2d 100: 0 / 27 / 2; RV1d 112: 0 / 38 / 8; RV1d2 74: 0 / 28 / 10; RV2d2 100: 0 / 45 / 12. Fixed went down by four (胡同学, 胡同志, 耗尽可能, 了解了 were right only because the preference crossed a stock word; they are stock now). RV1d cases3 (40): 6 regressions (9 before; R24 and R26 are right); RV2d cases3 (60): 47 (unchanged: the made-up literal-word sentences above).

| correction / drop rule | lines | helped positions | hurt positions | neutral lines |
|---|---|---|---|---|
| drop:并产 | 16 | 16 | 0 | 0 |
| drop:并包 | 10 | 5 | 0 | 6 |
| drop:先尝 | 9 | 9 | 0 | 0 |
| drop:心向 | 2 | 2 | 0 | 0 |
| drop:如干 | 2 | 3 | 0 | 0 |
| drop:了如 | 1 | 1 | 0 | 0 |
| entry:短短几 | 1 | 1 | 0 | 0 |

All other corrections have no hit in the corpus; the preference rows (marked `*` in `out/g4_12a.txt`) are the user's spellings and are expected to differ from the Taiwan dictionary.

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
