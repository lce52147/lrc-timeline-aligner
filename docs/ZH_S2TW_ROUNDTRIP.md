# s2tw: stock OpenCC plus guarded rules

`scripts/zh_traditional.py` converts Simplified to Traditional (Taiwan) for the review page. **With every rule table empty its output is character for character the stock OpenCC `s2tw` output**; each difference from stock comes from one rule that has a reason and a positive and a negative example, all run by `scripts/test_zh_convert.py`.

## Engine

The same algorithm as `opencc-python-reimplemented`'s `s2tw` (read from the installed package), run on our own tables: split at separators; in each clause take the longest phrase of `STPhrases`, the leftmost one among equally long ones, and convert the two sides the same way; characters no phrase covers go through `STCharacters`; then `TWVariants`. There is no scoring, no forward / backward comparison and no priority: a rule word is an ordinary dictionary word, so it only wins by being longer or further left. `convert_marked` still returns the marks (position and options of every character `STCharacters` maps to several forms) for the page.

## The four kinds of rule (BH2-FIX11A: closed contexts only)

The default is stock. A difference from stock is allowed in three forms only:

1. **Literal word** (`ENTRIES`): a fixed string longer than the stock word it replaces; the longest match wins, so its reach is the string itself. Two lists: `CORRECTIONS` (stock is wrong) and `PREFERENCES` (both forms are standard; the user's lyrics use this one).
2. **Closed context rule** (`CONTEXT_RULES`): a regular expression whose left and right sides are both lists of literal words. Not allowed: a gap (`.{m,n}`), a "not followed by" guard as the only protection, a one-character left or right context that can be the tail or head of another word without a word boundary, a noun list that decides a default. A rule that a reviewer breaks once is deleted, not patched.
3. **Preference** (`VARIANT_PREFERENCE`, and the `PREFERENCES` entries): either spelling is standard, the user's own lyrics decide.

`DROPPED_PHRASES` (a stock phrase that is not used) stays only where the Taiwan dictionary corpus has no sentence that needs the stock phrase (victims 0); every other stock phrase that hides a word is covered by a longer literal word instead (短短几, 一起干杯, 被发送 ...).

Totals: 113 corrections, 26 preferences, 6 dropped phrases, 4 context rules, 8 variant characters.

## Rules and their reasons

Every entry lists the look-alike that must not change (negative example); `scripts/test_zh_rules_r3.py` has for every non-preference word and every context rule two sentences where it fires and two look-alikes where it must not (the look-alikes use a different word with the same ending or beginning).

### a1. Corrections of stock OpenCC (113)

| word | written | reason | negative example |
|---|---|---|---|
| 一直发现 | 一直發現 | stock's 直发 (直髮) swallows the 发 of 一直\|发现 | 梳着一直发 -> 梳著一直髮 |
| 一直发光 | 一直發光 | stock's 直发 (直髮) swallows the 发 of 一直\|发光 | 梳着一直发 -> 梳著一直髮 |
| 一直发生 | 一直發生 | stock's 直发 (直髮) swallows the 发 of 一直\|发生 | 梳着一直发 -> 梳著一直髮 |
| 一直发展 | 一直發展 | stock's 直发 (直髮) swallows the 发 of 一直\|发展 | 梳着一直发 -> 梳著一直髮 |
| 一直发出 | 一直發出 | stock's 直发 (直髮) swallows the 发 of 一直\|发出 | 梳着一直发 -> 梳著一直髮 |
| 一直发愁 | 一直發愁 | stock's 直发 (直髮) swallows the 发 of 一直\|发愁 | 梳着一直发 -> 梳著一直髮 |
| 一直发呆 | 一直發呆 | stock's 直发 (直髮) swallows the 发 of 一直\|发呆 | 梳着一直发 -> 梳著一直髮 |
| 一直发抖 | 一直發抖 | stock's 直发 (直髮) swallows the 发 of 一直\|发抖 | 梳着一直发 -> 梳著一直髮 |
| 一直发烧 | 一直發燒 | stock's 直发 (直髮) swallows the 发 of 一直\|发烧 | 梳着一直发 -> 梳著一直髮 |
| 一直发笑 | 一直發笑 | stock's 直发 (直髮) swallows the 发 of 一直\|发笑 | 梳着一直发 -> 梳著一直髮 |
| 一直发亮 | 一直發亮 | stock's 直发 (直髮) swallows the 发 of 一直\|发亮 | 梳着一直发 -> 梳著一直髮 |
| 一直发怒 | 一直發怒 | stock's 直发 (直髮) swallows the 发 of 一直\|发怒 | 梳着一直发 -> 梳著一直髮 |
| 一直发火 | 一直發火 | stock's 直发 (直髮) swallows the 发 of 一直\|发火 | 梳着一直发 -> 梳著一直髮 |
| 一直发狂 | 一直發狂 | stock's 直发 (直髮) swallows the 发 of 一直\|发狂 | 梳着一直发 -> 梳著一直髮 |
| 一直发挥 | 一直發揮 | stock's 直发 (直髮) swallows the 发 of 一直\|发挥 | 梳着一直发 -> 梳著一直髮 |
| 一直发布 | 一直發布 | stock's 直发 (直髮) swallows the 发 of 一直\|发布 | 梳着一直发 -> 梳著一直髮 |
| 一直发表 | 一直發表 | stock's 直发 (直髮) swallows the 发 of 一直\|发表 | 梳着一直发 -> 梳著一直髮 |
| 一直发誓 | 一直發誓 | stock's phrase 直发 (直髮) swallows the 发 of 一直\|发誓 | 梳着一直发 -> 梳著一直髮 |
| 一头发光 | 一頭發光 | stock's 头发 (頭髮) swallows the 发 of 一头\|发光 | 长头发 -> 長頭髮 |
| 一头发亮 | 一頭發亮 | stock's 头发 (頭髮) swallows the 发 of 一头\|发亮 | 长头发 -> 長頭髮 |
| 风后来 | 風後來 | 风后 is the name 風后 (kept); 风+后来 is wind + 'later' | 风后 -> 風后 |
| 风后面 | 風後面 | 风后 is the name 風后 (kept); 风+后面 is wind + 'behind' | 风后 -> 風后 |
| 回归线 | 回歸線 | Taiwan writes 回歸線 (tropic); stock gives 迴歸線 | 线性回归 -> 線性迴歸 |
| 今天干了啥 | 今天幹了啥 | stock's phrase 天干 (heavenly stem) swallows the 干 of 今天\|干了啥 (stock leaves 干 unconverted) | 吃完饼干了啥都不想做 -> 吃完餅乾了啥都不想做 |
| 昨天干了啥 | 昨天幹了啥 | stock's phrase 天干 swallows the 干 of 昨天\|干了啥 | 吃完饼干了啥都不想做 -> 吃完餅乾了啥都不想做 |
| 明天干了啥 | 明天幹了啥 | stock's phrase 天干 swallows the 干 of 明天\|干了啥 | 吃完饼干了啥都不想做 -> 吃完餅乾了啥都不想做 |
| 新书签名 | 新書簽名 | 新书签名 = a new book signing (书\|签名), not the bookmark 书签 | 书签名称 -> 書籤名稱 |
| 图书签名 | 圖書簽名 | 图书签名 = signing of a book (书\|签名), not the bookmark 书签 | 书签名称 -> 書籤名稱 |
| 短短几 | 短短幾 | stock's phrase 短几 (a low table) swallows 短短\|几秒 | 一张短几 -> 一張短几 |
| 一起干杯 | 一起乾杯 | stock's phrase 一起干 (一起幹) swallows 一起\|干杯 | 一起干活 -> 一起幹活 |
| 统计划分 | 統計劃分 | stock reads 统计\|划分; the preference 计划 -> 計畫 would take the 计划 across the two words | 计划 -> 計畫 |
| 没干透 | 沒乾透 | stock's phrase 没干 (沒幹) swallows the 干 of 没\|干透 (not dry yet) | 他没干坏事 -> 他沒幹壞事 |
| 没干净 | 沒乾淨 | stock's phrase 没干 (沒幹) swallows the 干 of 没\|干净 (not clean) | 他没干坏事 -> 他沒幹壞事 |
| 没干燥 | 沒乾燥 | stock's phrase 没干 (沒幹) swallows the 干 of 没\|干燥 (not dry) | 他没干坏事 -> 他沒幹壞事 |
| 这杯干了 | 這杯乾了 | this cup / drink is finished (stock 幹了); the word is the whole string, a 这杯 elsewhere stays stock | 他把这件事干了吧 -> 他把這件事幹了吧 |
| 酒我干了 | 酒我乾了 | 'I drain this wine': 酒 + 我 + 干了 (stock 幹了) | 他干了一件大事 -> 他幹了一件大事 |
| 熬干了 | 熬乾了 | boiled dry (stock 熬幹了) | 他干了坏事 -> 他幹了壞事 |
| 嘴巴干了 | 嘴巴乾了 | a dry mouth (stock 嘴巴幹了) | 他干了坏事 -> 他幹了壞事 |
| 在海里 | 在海裡 | in the sea (stock leaves 在海里: 海里 is the nautical mile in stock); 在 + 海里 is never a distance | 航行了二十海里 -> 航行了二十海里 |
| 大海里 | 大海裡 | in the ocean (stock 大海里); a number before 海里 is never 大 | 还有半海里 -> 還有半海里 |
| 到海里 | 到海裡 | to the sea (stock 到海里) | 多少海里才到岸 -> 多少海里才到岸 |
| 入海里 | 入海裡 | into the sea (stock 入海里) | 每海里收费十元 -> 每海里收費十元 |
| 有云层 | 有雲層 | weather: a layer of cloud (stock keeps the classical phrase 有云 = 'it is said') | 古人有云 -> 古人有云 |
| 有云雾 | 有雲霧 | weather: cloud and mist (stock keeps the classical phrase 有云) | 古人有云 -> 古人有云 |
| 有云彩 | 有雲彩 | weather: clouds (stock keeps the classical phrase 有云) | 古人有云 -> 古人有云 |
| 有云朵 | 有雲朵 | weather: clouds (stock keeps the classical phrase 有云) | 古人有云 -> 古人有云 |
| 有云纹 | 有雲紋 | a cloud pattern (stock keeps the classical phrase 有云) | 古人有云 -> 古人有云 |
| 有云形 | 有雲形 | a cloud shape (stock keeps the classical phrase 有云) | 古人有云 -> 古人有云 |
| 有云状 | 有雲狀 | a cloud-like shape (stock keeps the classical phrase 有云) | 古人有云 -> 古人有云 |
| 有云雷 | 有雲雷 | clouds and thunder (stock keeps the classical phrase 有云) | 古人有云 -> 古人有云 |
| 有云雀 | 有雲雀 | 有\|云雀 (a lark); stock's classical phrase 有云 would hide it | 古人有云 -> 古人有云 |
| 不是只有 | 不是只有 | 不是 + 只有 = 'not only' (stock reads 是只 as 是隻); stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 若是只有 | 若是只有 | 若是 + 只有 = 'if only'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 总是只有 | 總是只有 | 总是 + 只有 = 'always only'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 总是只在 | 總是只在 | 总是 + 只在 = 'always only in'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 还是只想 | 還是只想 | 还是 + 只想 = 'still only want'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 没有只要 | 沒有只要 | 没有 + 只要 = 'without just asking'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 别只是 | 別只是 | 别 + 只是 = 'don't just'; stock's phrase 有只 / 是只 / 别只 reads 只 as the measure word 隻, here it is the adverb | 院子里有只会说话的八哥 -> 院子裡有隻會說話的八哥 |
| 卷卷而上 | 捲捲而上 | curling upward (捲捲而上); stock leaves 卷卷 | 问卷卷成了筒状 -> 問卷捲成了筒狀 |
| 湿润的风后 | 濕潤的風後 | wet air after the wind (風後); 风后 alone stays the name 風后 (stock) | 黄帝的大臣风后 -> 黃帝的大臣風后 |
| 海里有各种 | 海裡有各種 | in the sea there are all kinds (stock leaves 海里, the nautical mile, at the start of a clause) | 十二海里有一座岛 -> 十二海里有一座島 |
| 星海里 | 星海裡 | in the sea of stars (stock 星海里) | 多少海里才到岸 -> 多少海里才到岸 |
| 海里的鱼 | 海裡的魚 | the fish in the sea (stock 海里的魚) | 每小时航行十二海里的速度 -> 每小時航行十二海里的速度 |
| 古松开 | 古松開 | an old pine + 开 (古松开始变黄); stock's phrase 松开 (鬆開) would hide the 开 | 鞋带又松开了 -> 鞋帶又鬆開了 |
| 被发送 | 被發送 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发送 | 被发乱首 -> 被髮亂首 |
| 被发生 | 被發生 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发生 | 被发乱首 -> 被髮亂首 |
| 被发布 | 被發布 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发布 | 被发乱首 -> 被髮亂首 |
| 被发表 | 被發表 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发表 | 被发乱首 -> 被髮亂首 |
| 被发展 | 被發展 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发展 | 被发乱首 -> 被髮亂首 |
| 被发动 | 被發動 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发动 | 被发乱首 -> 被髮亂首 |
| 被发明 | 被發明 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发明 | 被发乱首 -> 被髮亂首 |
| 被发挥 | 被發揮 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发挥 | 被发乱首 -> 被髮亂首 |
| 被发出 | 被發出 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发出 | 被发乱首 -> 被髮亂首 |
| 被发回 | 被發回 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发回 | 被发乱首 -> 被髮亂首 |
| 被发下 | 被發下 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发下 | 被发乱首 -> 被髮亂首 |
| 被发给 | 被發給 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发给 | 被发乱首 -> 被髮亂首 |
| 被发到 | 被發到 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发到 | 被发乱首 -> 被髮亂首 |
| 被发往 | 被發往 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发往 | 被发乱首 -> 被髮亂首 |
| 被发掘 | 被發掘 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发掘 | 被发乱首 -> 被髮亂首 |
| 被发配 | 被發配 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发配 | 被发乱首 -> 被髮亂首 |
| 被发育 | 被發育 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发育 | 被发乱首 -> 被髮亂首 |
| 被发票 | 被發票 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发票 | 被发乱首 -> 被髮亂首 |
| 被发放 | 被發放 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发放 | 被发乱首 -> 被髮亂首 |
| 被发射 | 被發射 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发射 | 被发乱首 -> 被髮亂首 |
| 被发起 | 被發起 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发起 | 被发乱首 -> 被髮亂首 |
| 被发言 | 被發言 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发言 | 被发乱首 -> 被髮亂首 |
| 被发达 | 被發達 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发达 | 被发乱首 -> 被髮亂首 |
| 被发扬 | 被發揚 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发扬 | 被发乱首 -> 被髮亂首 |
| 被发来 | 被發來 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发来 | 被发乱首 -> 被髮亂首 |
| 被发过 | 被發過 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发过 | 被发乱首 -> 被髮亂首 |
| 被发去 | 被發去 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发去 | 被发乱首 -> 被髮亂首 |
| 被发走 | 被發走 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发走 | 被发乱首 -> 被髮亂首 |
| 被发了 | 被發了 | stock reads 被发 as 被髮 (loose hair); here it is 被\|发了 | 被发乱首 -> 被髮亂首 |
| 游标卡尺 | 游標卡尺 | vernier caliper: 游標; stock's longer phrase gives 遊標卡尺 | 旅游 -> 旅遊 |
| 兼容并包 | 兼容並包 | idiom written 並包 in Taiwan (stock 幷包) | 并包含 -> 並包含 |
| 天上有云 | 天上有雲 | weather: there are clouds in the sky (stock keeps the classical phrase 有云 = 'it is said') | 古人有云 -> 古人有云 |
| 天空有云 | 天空有雲 | weather: there are clouds in the sky (stock keeps the classical phrase 有云) | 古人有云 -> 古人有云 |
| 今天有云 | 今天有雲 | weather: clouds today (stock keeps the classical phrase 有云) | 古人有云 -> 古人有云 |
| 明天有云 | 明天有雲 | weather: clouds tomorrow (stock keeps the classical phrase 有云) | 古人有云 -> 古人有云 |
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

### b. Dropped stock phrases (6)

| phrase | reason (positive -> result; negative -> result) |
|---|---|
| 了如 | 我明白了如何做 -> 我明白了如何做; 了如指掌 -> 瞭如指掌 |
| 先尝 | 先尝试 -> 先嘗試; 尝试 -> 嘗試 |
| 如干 | 倒不如干脆 -> 倒不如乾脆; 若干 -> 若干 |
| 并产 | 并产生 -> 並產生; 合并 -> 合併 |
| 并包 | 并包含 -> 並包含; 合并 -> 合併 |
| 心向 | 内心向往 -> 內心嚮往; 心向党 -> 心向黨 |

### c. Context rules (4)

| name | fires | must not fire | why |
|---|---|---|---|
| tour | 漫游过去的事 -> 漫遊過去的事; 导游过来接我们 -> 導遊過來接我們 | 他游过了河 -> 他游過了河; 鱼游动着 -> 魚游動著 | 旅游 漫游 导游 周游 巡游 畅游 before 动 过 移 is the tourism word (遊) and 动 过 移 starts the next word; the preference 游动/游过/游移 (swim) would take the 游 |
| dry-throat | 如果喉咙干了，就去喝水 -> 如果喉嚨乾了，就去喝水; 喉咙干了就喝水 -> 喉嚨乾了就喝水 | 捂住喉咙干了坏事 -> 捂住喉嚨幹了壞事; 他喉咙干了三杯 -> 他喉嚨幹了三杯 | 喉咙 + 干了 that ends the clause (or is followed by a particle: 的 了 就 都 也 ...) is a dry throat: 乾了 (stock 喉嚨幹了) |
| toast-start | 干了就没事了 -> 乾了就沒事了 | 这活儿干了就没事了 -> 這活兒幹了就沒事了; 他干了就没事 -> 他幹了就沒事 | a clause that starts with 干了就没事 is the toast 'drink up and it is fine': 乾了 (stock 幹了); after any other word it is 'do it and it is fine' (幹了, stock) |
| surname-yu | 余先生在家 -> 余先生在家; 我找余老师 -> 我找余老師 | 剩余先生产 -> 剩餘先生產; 多余老师的话 -> 多餘老師的話 | the surname 余 at the start of a clause (or right after 姓 叫 找 问 请) and right before a title that ends the word stays 余; stock writes 餘 |

### d. Variant preference (8)

臺->台, 溼->濕, 汙->污, 燬->毀, 絃->弦, 悽->淒, 歎->嘆, 唸->念


## Removed in BH2-FIX11A (REVIEW4: 42 + 22 sentences where stock was right and the rules were wrong)

`sea` (海里), `dry-drink` / `dry-verb` / `dry-toast` / `dry-place` / `dry-compound` (干了), `zhi-demonstrative` / `zhi-adverb` (这只), `said-yun` (有云), `loose-hair` (被发), `wind-after` (风后), `curl` (卷卷), `pine` (松开), `scrape` (刮), the stock-phrase drops 一起干 / 短几 / 有云 / 被发, and the entries 一直 / 一头 / 干了啥 / 书签名 (each swallowed a longer stock word). What the user's lyrics or the dictionary corpus needed from them is a literal word now; the rest is stock.

## Known stock limitations (not regressions: stock is wrong too, a single sentence cannot tell)

* 这只 / 那只 + 有 能 会 想 要: stock reads the measure word (這隻); the adverb 這只 (这只有五分钟) is written the same, so stock stays except for the literal 总是只有 / 不是只有 / 若是只有 / 还是只想 / 没有只要 / 别只是.
* 干了 as 乾了 (a drink or a river finished) vs 幹了 (did): stock 幹; only the literal words 把这杯干了 / 酒我干了 / 熬干了 / 嘴巴干了 / the clause-start 干了就没事 and 喉咙干了 at a clause end are fixed. Sentences such as 河水都干了, 池塘干了啥都不剩, 把汤干了吧 stay 幹.
* 有云 (classical 'it is said' vs weather): stock 有云 (classical) except the literal weather words (有云层 天上有云 今天有云 ...).
* 海里 (nautical mile vs in the sea): stock 海里 except 在海里 大海里 到海里 入海里 星海里 海里有各种 海里的鱼.
* 卷卷 / 刮 / 风后 / 古松开 variants and the 游 of 夜游过: stock.
* 髮 -> 發 inside a proper name written with 后+发 in the lyrics (the one remaining wrong meaning of the lyrics round trip).
* The page still marks every character with several Traditional forms, so the user can pick by hand.

## Measurements (final HEAD)

* **Empty rules = stock** (`tools/g1_all.py`, rules emptied): every `STPhrases` word (49051), the 530 sentences of the adversarial set, the 7174 kana-free lyric lines of the user's corpus (after `t2s`): 0 differences from `OpenCC("s2tw")`; `test_zh_convert.py` repeats it for all `STPhrases` words plus a fixed sample.
* **Adversarial sets** (`tools/run_cases.py`, stock right and ours wrong / both wrong / stock wrong and ours right): `cases.tsv` 530 sentences 0 / 22 / 37; REVIEW4 set 1 (100 sentences) 0 / 18 / 11; REVIEW4 set 2 (112 sentences) 0 / 26 / 20; self-made set (85 sentences, `cases_self2.tsv`) 0 / 0 / 21 (9 more are listed as uncertain). The both-wrong sentences are the known limitations below.
* **User's lyrics round trip** (169 songs, 6628 lines, 66860 Han characters): wrong meaning 1, variant 14, Japanese 11; 0 lines differ from FIX9 (`tools/lyrics_diff.py` over 7174 lines). Converting the 7174 lines: FIX9 0.56 s, now 0.43 s.
* **Taiwan dictionary corpus round trip** (MOE 《重編國語辭典修訂本》 from g0v/moedict-data `dict-revised.json.xz`: 656935 lines, 8681563 Han characters; Traditional -> `tw2s` -> ours and stock `s2tw` -> compared with the source; test data only, not in the repo): positions where stock is right and ours wrong 6781, **0 of them outside the preference rules** (臺->台 4648 and the like: the dictionary is Taiwan-official, the user's own lyrics use the other form); ours right and stock wrong 2041; both wrong 48654 (出->齣 in 〇出, 卷 sections, 分佈, names: they need a dictionary, not a rule). Hit table of the non-preference rules (final HEAD, `tools/g4.py`):

| correction / context / drop rule | lines | helped positions | hurt positions | neutral lines |
|---|---|---|---|---|
| drop:并产 | 16 | 16 | 0 | 0 |
| drop:并包 | 10 | 5 | 0 | 6 |
| drop:先尝 | 9 | 9 | 0 | 0 |
| entry:在海里 | 6 | 6 | 0 | 0 |
| entry:兼容并包 | 5 | 0 | 0 | 5 |
| entry:大海里 | 5 | 5 | 0 | 0 |
| entry:游标 | 4 | 4 | 0 | 0 |
| entry:有云形 | 3 | 3 | 0 | 0 |
| entry:到海里 | 2 | 2 | 0 | 0 |
| entry:被发掘 | 2 | 2 | 0 | 0 |
| drop:心向 | 2 | 2 | 0 | 0 |
| entry:回归线 | 2 | 3 | 0 | 0 |
| entry:控制台 | 2 | 0 | 0 | 2 |
| entry:入海里 | 2 | 2 | 0 | 0 |
| entry:有云雾 | 2 | 2 | 0 | 0 |
| entry:没干净 | 2 | 2 | 0 | 0 |
| drop:如干 | 2 | 3 | 0 | 0 |
| entry:有云纹 | 2 | 2 | 0 | 0 |
| entry:一直发抖 | 1 | 1 | 0 | 0 |
| entry:被发配 | 1 | 1 | 0 | 0 |
| entry:被发票 | 1 | 1 | 0 | 0 |
| drop:了如 | 1 | 1 | 0 | 0 |
| entry:游标卡尺 | 1 | 1 | 0 | 0 |
| entry:被发育 | 1 | 1 | 0 | 0 |
| entry:有云雷 | 1 | 1 | 0 | 0 |
| entry:有云状 | 1 | 1 | 0 | 0 |
| entry:短短几 | 1 | 1 | 0 | 0 |
| entry:有云彩 | 1 | 1 | 0 | 0 |


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
