from __future__ import annotations
import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch

from lyricsee_core import LyricPair
from translation_oogg import _validate_texts, translate_oogg


class OoggTranslatorTests(unittest.TestCase):
    def test_text_output_rejects_newlines(self):
        with self.assertRaisesRegex(ValueError, "multiline"):
            _validate_texts(["歌詞原文"], ["翻譯\n第二行"])

    def test_oogg_cold_wake_503_retries_then_second_pass(self):
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
        original=["夜が染める","君と歩く"]
        requests=[]
        def remote(url,payload,timeout):
            req=json.loads(payload["messages"][1]["content"])
            requests.append(req)
            text=req["translate_this_line_only"]
            return {"choices":[{"message":{"content":
                json.dumps({"text":"翻譯："+text},ensure_ascii=False)}}]}
        with patch("lyricsee_core.HttpClient.get_json",
                   return_value={"data":[{"id":"GGOO-test"}]}), \
             patch("lyricsee_core.translate_with_local_llm",
                   side_effect=ValueError("extra ids: L009")):
            got=translate_oogg(original,title="Sample",
                               endpoint="http://100.87.97.121:6969",
                               post_json=remote,review=False)
        self.assertEqual(got,["翻譯："+s for s in original])
        self.assertEqual(len(requests),len(original))

    def test_oogg_compact_minimal_recovery_after_context_refusal(self):
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
