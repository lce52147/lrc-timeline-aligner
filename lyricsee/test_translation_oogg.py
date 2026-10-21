from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from translation_oogg import (
    translate_deepl, _validate_texts,
)
import translate_rebuilt_df_20261008 as pipeline


class ExternalTranslatorTests(unittest.TestCase):
    def test_deepl_uses_traditional_and_preserves_order(self):
        traffic=[]
        def transport(url,body,key,timeout):
            traffic.append((url,body))
            self.assertEqual(body["target_lang"],"ZH-HANT")
            self.assertEqual(body["split_sentences"],"0")
            self.assertTrue(url.startswith("https://api-free.deepl.com/"))
            return {"translations":[{"text":"繁體："+v} for v in body["text"]]}
        texts=["今日は星が光る","明日は会える","今日は星が光る"]
        result=translate_deepl(texts,key="MOCK:fx",
             context="完整原文上下文",batch_size=2,transport=transport)
        self.assertEqual(result,["繁體："+s for s in texts])
        self.assertEqual(sum(len(x[1]["text"]) for x in traffic),2)
        self.assertEqual(len(traffic),1)

    def test_invalid_translation_row_count_is_rejected(self):
        def transport(url,body,key,timeout):
            return {"translations":[{"text":"有誤"}]}
        with self.assertRaisesRegex(RuntimeError,"number"):
            translate_deepl(["一","二"],key="MOCK:fx",transport=transport)

    def test_deepl_key_missing_fails_before_network(self):
        with patch.dict("os.environ",{"DEEPL_API_KEY":""}):
            with self.assertRaisesRegex(RuntimeError,"DEEPL_API_KEY"):
                translate_deepl(["日文原文"])

    def test_text_output_rejects_newlines(self):
        with self.assertRaisesRegex(ValueError,"multiline"):
            _validate_texts(["歌詞原文"],["翻譯\n第二行"])

    def test_original_only_staging_never_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            src=root/"100__Sample.lyrics.txt"
            orig=[f"君がいる世界{i}" for i in range(12)]
            src.write_text("\n".join(orig)+"\n",encoding="utf-8")
            digest=hashlib.sha256(src.read_bytes()).hexdigest()
            row={
                "index":100,"stem":"Sample","status":"ORIGINAL_VERIFIED_NEEDS_TRANSLATION",
                "txt_path":str(src),"txt_sha256":digest,"original_rows":len(orig),
                "original_evidence":{"policy":pipeline.VERSION},
                "original_source":"lrclib",
                "original_source_url":"https://lrclib.net/api/get/10",
                "title":"Sample","artist":"Singer","audio":"C:/Music/Sample.flac",
            }
            def fake_deepl(lines,context):
                return ["月光與你"+str(i) for i in range(len(lines))]
            with patch.object(pipeline,"TARGET",root/"stage"):
                result=pipeline.translate_one(row,"deepl",deepl=fake_deepl)
                self.assertEqual(result["status"],"TRANSLATED_STAGED_QA_PENDING")
                self.assertEqual(src.read_text(encoding="utf-8"),"\n".join(orig)+"\n")
                output=Path(result["txt_path"]).read_text(encoding="utf-8")
                self.assertEqual(len(output.splitlines()),len(orig))
                evidence=json.loads(Path(root/"stage/deepl/sources/100__Sample.json").read_text(encoding="utf-8"))
                self.assertEqual(evidence["translation_method"],"deepl-api-zh-hant")
                self.assertFalse(evidence["human_translation_qa"])
                self.assertFalse(evidence["lrc_admitted"])

    def test_source_integrity_rejects_modified_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/"test.txt"
            source.write_text("bad",encoding="utf-8")
            r={"index":1,"stem":"bad","status":"ORIGINAL_VERIFIED_NEEDS_TRANSLATION",
               "txt_path":str(source),"txt_sha256":"wrong",
               "original_evidence":{"policy":pipeline.VERSION}}
            attempted=[]
            def fake_deepl(lines,context):attempted.append(1);return []
            result=pipeline.translate_one(r,"deepl",deepl=fake_deepl)
            self.assertEqual(result["status"],"FAILED_SAFE")
            self.assertEqual(attempted,[])

    def test_oogg_cold_wake_503_retries_then_second_pass(self):
        from urllib.error import HTTPError
        from lyricsee_core import LyricPair
        from translation_oogg import translate_oogg
        source=["声が響く","君が笑う"]
        attempts=[]
        def post_json(url,payload,timeout):
            attempts.append(1)
            if len(attempts)==1:
                raise HTTPError(url,503,"Model waking",{},None)
            return {"choices":[{"message":{"content":"{}"}}]}
        def generate(lines,*,post_json,**kwargs):
            post_json("http://localhost/v1/chat/completions",{},300)
            return [LyricPair(x,"初稿"+str(i)) for i,x in enumerate(lines)]
        def review(lines,*,post_json,**kwargs):
            post_json("http://localhost/v1/chat/completions",{},300)
            return [LyricPair(x.original,"校對後"+str(i)) for i,x in enumerate(lines)]
        with patch("lyricsee_core.HttpClient.get_json",
                   return_value={"data":[{"id":"OOGG-test"}]}), \
             patch("lyricsee_core.translate_with_local_llm",side_effect=generate), \
             patch("lyricsee_core.review_translations_with_local_llm",side_effect=review), \
             patch("time.sleep") as sleep:
            result=translate_oogg(source,title="歌",post_json=post_json)
        self.assertEqual(result,["校對後0","校對後1"])
        self.assertEqual(len(attempts),3)
        sleep.assert_called_once_with(2)


    def test_oogg_compact_single_row_fallback_preserves_exact_order(self):
        from translation_oogg import translate_oogg
        original=["夜が染める","君と歩く"]
        requests=[]
        def remote(url,payload,timeout):
            req=json.loads(payload["messages"][1]["content"])
            requests.append(req)
            self.assertNotIn("full_song_context",req)
            self.assertNotIn("translate_only_ids",req)
            text=req["translate_this_line_only"]
            return {"choices":[{"message":{"content":
                json.dumps({"text":"翻譯："+text},ensure_ascii=False)}}]}
        with patch("lyricsee_core.HttpClient.get_json",
                   return_value={"data":[{"id":"GGOO-test"}]}), \
             patch("lyricsee_core.translate_with_local_llm",
                   side_effect=ValueError("extra ids: L009")):
            got=translate_oogg(original,title="Sample",endpoint="http://100.87.97.121:6969",
                               post_json=remote,review=False)
        self.assertEqual(got,["翻譯："+s for s in original])
        self.assertEqual(len(requests),len(original))

    def test_oogg_compact_minimal_recovery_after_context_refusal(self):
        from translation_oogg import translate_oogg
        attempts=[]
        def remote(url,payload,timeout):
            content=json.loads(payload["messages"][1]["content"])
            attempts.append(content)
            translated=("救贖嗎？" if "context_only" not in content else "__UNSURE__")
            return {"choices":[{"message":{"content":
                json.dumps({"text":translated},ensure_ascii=False)}}]}
        with patch("lyricsee_core.HttpClient.get_json",
                   return_value={"data":[{"id":"GGOO-test"}]}), \
             patch("lyricsee_core.translate_with_local_llm",
                   side_effect=ValueError("extra ids")):
            got=translate_oogg(["(Memento mori) 救済なの？"],
                               title="Ave Mujica",post_json=remote,review=False)
        self.assertEqual(got,["救贖嗎？"])
        self.assertEqual(sum("context_only" in x for x in attempts),3)
        self.assertEqual(sum("context_only" not in x for x in attempts),1)

    def test_oogg_compact_rejects_extra_response_fields(self):
        from translation_oogg import translate_oogg
        def remote(url,payload,timeout):
            return {"choices":[{"message":{"content":json.dumps(
                {"text":"翻譯","also":"unverified_extra"})}}]}
        with patch("lyricsee_core.HttpClient.get_json",
                   return_value={"data":[{"id":"GGOO-test"}]}), \
             patch("lyricsee_core.translate_with_local_llm",
                   side_effect=ValueError("extra ids")):
            with self.assertRaisesRegex(RuntimeError,"compact"):
                translate_oogg(["花が咲く"],title="Sample",
                               post_json=remote,review=False)


if __name__=="__main__":
    unittest.main()
