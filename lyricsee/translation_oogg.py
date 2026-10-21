"""OOGG translation backend. Never fetch lyrics or create original text."""
from __future__ import annotations
import json
from urllib.error import HTTPError, URLError

def _validate_texts(original: list[str], translated: list[str]) -> list[str]:
    if len(original) != len(translated):
        raise ValueError(f"Translator changed lyric row count: {len(original)} -> {len(translated)}")
    valid=[]
    for i,(source,out) in enumerate(zip(original,translated),1):
        if not isinstance(out,str) or not out.strip():
            raise ValueError(f"Empty translation at line {i}")
        out=out.strip()
        if any(c in out for c in ("\n","\r","\x00","\ufffd")):
            raise ValueError(f"Invalid multiline/control translation at line {i}")
        if len(out) > 450 or len(out)>max(120,len(source)*7):
            raise ValueError(f"Abnormally verbose translation at line {i}")
        valid.append(out)
    return valid

def translate_oogg(original: list[str], *,
                   title: str,
                   endpoint: str="http://127.0.0.1:6969",
                   model: str | None=None,
                   post_json=None,
                   review: bool=True) -> list[str]:
    from lyricsee_core import (
        HttpClient, translate_with_local_llm,
        review_translations_with_local_llm,
    )
    import time
    http=HttpClient()
    if post_json is None: post_json=http.post_json
    if model is None:
        payload=http.get_json(endpoint.rstrip("/")+"/v1/models")
        models=payload.get("data") if isinstance(payload,dict) else None
        if not models or not models[0].get("id"):
            raise RuntimeError("OOGG endpoint has no available model")
        model=str(models[0]["id"])

    def post_with_wake_retry(url, request, timeout):
        # llama-server --sleep-idle-seconds may return 503 during wake-up.
        # This is not evidence that source lyrics or model output are wrong.
        for attempt in range(4):
            try:
                return post_json(url, request, timeout)
            except HTTPError as exc:
                if exc.code != 503 or attempt == 3:
                    raise
                time.sleep((2, 3, 5)[attempt])
        raise RuntimeError("Unreachable OOGG retry state")

    # Keep strict JSON row-ID validation. If the model returns an extra
    # row, missing ID or server reset, regenerate in *smaller* chunks.
    # Do not silently shift a Chinese line onto the wrong original.
    pairs=None
    errors=[]
    for chunk_size in (4, 2, 1):
        try:
            pairs=translate_with_local_llm(
                original,title=title,endpoint=endpoint,
                model=model,post_json=post_with_wake_retry,
                batch_size=chunk_size,
            )
            break
        except (ValueError, HTTPError, URLError, ConnectionResetError) as exc:
            errors.append(f"{chunk_size}-row: {type(exc).__name__}: {str(exc)[:110]}")
    if pairs is None:
        # Some small models repeat full_song_context IDs even with a one-row
        # target (e.g. L009 when requesting only L008). Final rescue mode
        # presents one target row with no numbered full-song context, so an
        # out-of-scope ID cannot accidentally be assigned to a source line.
        from lyricsee_core import LyricPair, _parse_model_json
        translated=[]
        try:
            for index,source in enumerate(original):
                neighbors=original[max(0,index-2):index]+original[index+1:index+3]
                request={
                    "model":model,"temperature":0,"max_tokens":360,
                    "response_format":{"type":"json_object"},
                    "messages":[
                        {"role":"system","content":(
                            "將指定的一句日文或英文歌詞忠實翻譯為繁體中文。"
                            "上下文只供理解；禁止翻譯上下文或改寫指定原文。"
                            "輸出唯一 JSON 物件，只有 text 一個欄位："
                            '{"text":"一行繁體中文翻譯"}。'
                            "不得有換行、其他行、其他物件、行號或註解。"
                            "不能確認含義時輸出 __UNSURE__。"
                        )},
                        {"role":"user","content":json.dumps({
                            "title":title,"context_only":neighbors,
                            "translate_this_line_only":source,
                        },ensure_ascii=False)},
                    ],
                }
                checked=None
                for _ in range(3):
                    response=post_with_wake_retry(
                        endpoint.rstrip("/")+"/v1/chat/completions",
                        request,300,
                    )
                    try:
                        result=_parse_model_json(response)
                    except ValueError:
                        # A malformed single-line JSON response can be
                        # regenerated; never accept an unparsed raw string.
                        continue
                    if set(result)!={"text"} or not isinstance(result["text"],str):
                        continue
                    try:
                        checked=_validate_texts([source],[result["text"]])[0]
                    except ValueError:
                        continue
                    if "__UNSURE__" in checked.upper():
                        checked=None
                        continue
                    break
                if checked is None:
                    # Context sometimes triggers unrelated continuation for
                    # parenthetical or mixed-language lyric rows. Retry only
                    # the literal source with an intentionally minimal prompt.
                    minimal={
                        "model":model,"temperature":0,"max_tokens":300,
                        "response_format":{"type":"json_object"},
                        "messages":[
                            {"role":"system","content":(
                                '只翻譯這一句日文或英文為繁體中文。'
                                '只輸出JSON：{"text":"翻譯"}。'
                                '不得輸出其他行、解釋、行號或原文。')},
                            {"role":"user","content":json.dumps(
                                {"translate_this_line_only":source},
                                ensure_ascii=False)},
                        ],
                    }
                    for _ in range(3):
                        response=post_with_wake_retry(
                            endpoint.rstrip("/")+"/v1/chat/completions",
                            minimal,300,
                        )
                        try:
                            parsed=_parse_model_json(response)
                            if set(parsed)!={"text"} or not isinstance(parsed["text"],str):
                                continue
                            value=_validate_texts([source],[parsed["text"]])[0]
                        except ValueError:
                            continue
                        if "__UNSURE__" in value.upper():
                            continue
                        checked=value
                        break
                if checked is None:
                    raise ValueError(f"compact row {index+1}: no valid one-line translation")
                translated.append(LyricPair(source,checked))
            pairs=translated
        except Exception as exc:
            raise RuntimeError(
                "OOGG failed both strict batch and compact single-row fallback: "
                +" | ".join(errors)+" | "+
                f"{type(exc).__name__}: {str(exc)[:130]}"
            ) from exc
    if [r.original for r in pairs]!=original:
        raise RuntimeError("LLM unexpectedly changed original lyric rows")
    if not review:
        return _validate_texts(original,[r.translation or "" for r in pairs])
    reviewed=review_translations_with_local_llm(
        pairs, title=title, endpoint=endpoint, model=model,
        post_json=post_with_wake_retry, batch_size=4,
    )
    if [r.original for r in reviewed]!=original:
        raise RuntimeError("OOGG QA unexpectedly changed original lyric rows")
    return _validate_texts(original,[r.translation or "" for r in reviewed])
