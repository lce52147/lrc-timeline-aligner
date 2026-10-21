"""Optional translation backends. Never fetch lyrics or create original text.

DeepL: HTTPS POST /v2/translate, exact 1-to-1 output, ZH-HANT. Key ONLY
from DEEPL_API_KEY environment variable and NEVER written to any report.
OOGG: pre-existing OpenAI-compatible HTTP endpoint, explicit opt-in.
"""
from __future__ import annotations
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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

def _deepl_http(url: str, body: dict, api_key: str, timeout: float) -> dict:
    payload=json.dumps(body,ensure_ascii=False,separators=(",",":")).encode("utf-8")
    if len(payload)>128*1024:
        raise ValueError("DeepL request exceeds 128 KiB limit")
    req=Request(url,
        data=payload,
        headers={
            "Authorization":"DeepL-Auth-Key "+api_key,
            "Content-Type":"application/json",
            "User-Agent":"Lyricsee-external-translation/1.0",
        },
        method="POST")
    try:
        with urlopen(req,timeout=timeout) as r:
            if r.status!=200:raise RuntimeError("DeepL request failed")
            return json.loads(r.read(500000).decode("utf-8"))
    except HTTPError as e:
        labels={400:"invalid-request",403:"invalid-key-or-plan",429:"rate-limited",456:"quota-exceeded"}
        raise RuntimeError("DeepL API "+labels.get(e.code,f"HTTP-{e.code}")) from None
    except URLError as e:
        raise RuntimeError("DeepL network connection failed") from e

def translate_deepl(original: list[str], *,
                    key: str | None=None,
                    source_lang: str | None=None,
                    context: str="",
                    batch_size: int=25,
                    transport=None) -> list[str]:
    if not original: raise ValueError("No lyric originals to translate")
    if not 1<=batch_size<=50:raise ValueError("DeepL batch size must be 1..50")
    if any(not isinstance(line,str) or not line.strip() for line in original):
        raise ValueError("Original contains blank or non-text lyric")
    key = key if key is not None else os.environ.get("DEEPL_API_KEY","")
    if not key:raise RuntimeError("DEEPL_API_KEY missing; no API calls made")
    # User explicitly prohibits paid translation requests: no paid API endpoint.
    if not key.endswith(":fx"):
        raise RuntimeError("Paid DeepL API key refused; use free MyMemory or local OOGG")
    url="https://api-free.deepl.com/v2/translate"
    http=transport or _deepl_http
    # Deduplicate identical refrain lines while retaining original order.
    unique=list(dict.fromkeys(original))
    table={}
    short_context=context[:7000]
    for start in range(0,len(unique),batch_size):
        chunk=unique[start:start+batch_size]
        body={
            "text":chunk,
            "target_lang":"ZH-HANT",
            "split_sentences":"0",
            "preserve_formatting":True,
        }
        if source_lang:body["source_lang"]=source_lang
        if short_context:body["context"]=short_context
        result=http(url,body,key,40.0)
        if not isinstance(result,dict) or not isinstance(result.get("translations"),list):
            raise RuntimeError("DeepL returned malformed translation response")
        rows=result["translations"]
        if len(rows)!=len(chunk):
            raise RuntimeError("DeepL changed the number of translated lines")
        outputs=[item.get("text") if isinstance(item,dict) else None for item in rows]
        checked=_validate_texts(chunk,outputs)
        table.update(zip(chunk,checked))
    return _validate_texts(original,[table[v] for v in original])

def translate_oogg(original: list[str], *,
                   title: str,
                   endpoint: str="http://127.0.0.1:6969",
                   post_json=None,
                   review: bool=True) -> list[str]:
    from lyricsee_core import (
        HttpClient, translate_with_local_llm,
        review_translations_with_local_llm,
    )
    import time
    http=HttpClient()
    if post_json is None: post_json=http.post_json
    payload=http.get_json(endpoint.rstrip("/")+"/v1/models")
    models=payload.get("data") if isinstance(payload,dict) else None
    if not models or not models[0].get("id"):
        raise RuntimeError("OOGG endpoint has no available model")
    model=models[0]["id"]

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


def _mymemory_http(text: str, *, timeout: float = 15.0) -> dict:
    """Free documented MyMemory GET API. Never use a paid service or hidden key."""
    from urllib.parse import urlencode
    url = "https://api.mymemory.translated.net/get?" + urlencode({
        "q": text,
        "langpair": "ja|zh-TW",
    })
    req = Request(url, headers={
        "User-Agent": "Lyricsee/1.0 (voluntary anonymous free API)",
        "Accept": "application/json",
    })
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read(150000).decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"MyMemory HTTP {exc.code}; no automatic paid fallback") from None
    except URLError as exc:
        raise RuntimeError("MyMemory unavailable; no automatic paid fallback") from exc


def translate_mymemory(
    original: list[str],
    *,
    daily_budget: int = 4500,
    usage_path=None,
    transport=None,
) -> list[str]:
    """No-key free translator with local UTC-day budget and repeated-line cache.

    Default budget is below the published 5000 char/day anonymous IP allowance.
    Each returned lyric remains machine-translated / pending human QA.
    """
    from datetime import datetime, timezone
    from html import unescape
    from pathlib import Path
    from urllib.parse import quote
    if not original or any(not isinstance(line, str) or not line.strip() for line in original):
        raise ValueError("MyMemory requires nonempty source lyric lines")
    if not 1 <= daily_budget <= 5000:
        raise ValueError("Anonymous free limit cannot exceed 5000 characters per day")
    unique = list(dict.fromkeys(original))
    if any(len(line.encode("utf-8")) > 480 for line in unique):
        raise ValueError("A lyric line exceeds the MyMemory free query byte limit")
    date = datetime.now(timezone.utc).date().isoformat()
    ledger = Path(usage_path) if usage_path is not None else (
        Path(__file__).resolve().parent / "report" / "mymemory_free_usage.json"
    )
    used = 0
    if ledger.is_file():
        record = json.loads(ledger.read_text(encoding="utf-8-sig"))
        if record.get("utc_day") == date:
            used = int(record.get("sent_characters", 0))
    requested = sum(len(line) for line in unique)
    if used + requested > daily_budget:
        raise RuntimeError(
            f"MyMemory anonymous free daily budget reached: "
            f"{used}+{requested}>{daily_budget} characters; choose OOGG"
        )
    send = transport or _mymemory_http
    lookup = {}
    for line in unique:
        # Debit before each request to avoid retrying a paid-for or failed
        # request after a network interruption. This is a conservative cap.
        used += len(line)
        from lyricsee_core import _atomic_write_text
        _atomic_write_text(
            ledger,
            json.dumps(
                {"utc_day": date, "sent_characters": used,
                 "local_daily_budget": daily_budget, "provider": "MyMemory anonymous"},
                ensure_ascii=False, indent=2,
            ) + "\n",
        )
        response = send(line)
        if not isinstance(response, dict) or response.get("responseStatus") != 200:
            raise RuntimeError("MyMemory rejected or rate-limited free request")
        data = response.get("responseData") or {}
        text = unescape(str(data.get("translatedText") or "")).strip()
        if "MYMEMORY WARNING" in text.upper() or "QUERY LENGTH LIMIT" in text.upper():
            raise RuntimeError("MyMemory returned a quota/warning message")
        lookup[line] = _validate_texts([line], [text])[0]
    return _validate_texts(original, [lookup[v] for v in original])
