"""Gemini translation backend (OpenClaw's google provider) with OOGG as the fallback. Never fetches lyrics.

The API key is read at run time only: env GEMINI_API_KEY (comma separated for several keys), otherwise the
OpenClaw auth store (read-only sqlite, profiles ``google:*``). A key is never printed, logged or put into an
exception text; every message that leaves this module goes through ``mask``.
"""
from __future__ import annotations
import json
import os
import re
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Callable, NamedTuple
from urllib.error import HTTPError, URLError

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
OPENCLAW_HOME = Path.home() / ".openclaw"
MAX_CHUNK = 40
_KEY_SHAPE = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")

SYSTEM_PROMPT = (
    "你是歌詞翻譯員。把歌曲的每一行歌詞忠實翻譯成繁體中文（臺灣用字）。"
    "整首歌只供你理解語意；每個輸出列只能翻譯它自己的那一行原文，不得翻譯、重複或合併相鄰的行。"
    "不要加註解、說明、行號或原文；英文與羅馬拼音的人名、專有名詞照原樣保留；輸出一律為繁體中文。"
    "只輸出 JSON：{\"rows\":[{\"id\":\"L001\",\"text\":\"翻譯\"}]}，id 必須與要求翻譯的 id 完全相同、順序相同，"
    "text 不得為空、不得含換行。"
)
LINE_PROMPT = (
    "你是歌詞翻譯員。只翻譯 translate_this_line_only 這一行為繁體中文（臺灣用字）。"
    "context_before 與 context_after 是前後的歌詞，只供理解語意，絕對不要翻譯它們，也不要重複它們的意思。"
    "不要加註解、說明或原文；英文與羅馬拼音的人名照原樣保留。"
    "只輸出 JSON：{\"text\":\"翻譯\"}，text 不得為空、不得含換行。"
)


class GeminiError(RuntimeError):
    """Any Gemini failure. The message is already masked."""


class TranslationResult(NamedTuple):
    lines: list[str]
    backend: str
    model: str


def mask(text: object, keys: tuple[str, ...] | list[str] = ()) -> str:
    out = str(text)
    for key in keys:
        if key:
            out = out.replace(key, "***")
    return _KEY_SHAPE.sub("***", out)


def _store_tokens(path: Path) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    try:
        conn = sqlite3.connect("file:" + path.as_posix() + "?mode=ro", uri=True)
    except sqlite3.Error:
        return found
    try:
        for table in ("auth_profile_store", "auth_profile_stores"):
            try:
                rows = conn.execute(f"select store_json from {table}").fetchall()
            except sqlite3.Error:
                continue
            for (blob,) in rows:
                try:
                    profiles = (json.loads(blob) or {}).get("profiles") or {}
                except ValueError:
                    continue
                for name, profile in profiles.items():
                    if isinstance(profile, dict) and profile.get("provider") == "google" and profile.get("token"):
                        found.append((str(name), str(profile["token"])))
    finally:
        conn.close()
    return found


def load_keys(*, env: dict | None = None, openclaw_home: Path | None = None) -> list[tuple[str, str]]:
    """[(profile name, key)] in rotation order. Env wins; the OpenClaw store is only read."""
    env = os.environ if env is None else env
    raw = (env.get("GEMINI_API_KEY") or "").strip()
    if raw:
        return [(f"env-{i}", k.strip()) for i, k in enumerate(raw.split(","), 1) if k.strip()]
    home = Path(openclaw_home) if openclaw_home else OPENCLAW_HOME
    seen: dict[str, str] = {}
    paths = [home / "agents" / "main" / "agent" / "openclaw-agent.sqlite", home / "state" / "openclaw.sqlite"]
    paths += sorted((home / "agents").glob("*/agent/openclaw-agent.sqlite"))
    for path in paths:
        if path.is_file():
            for name, token in _store_tokens(path):
                seen.setdefault(token, name)
    return sorted(((name, token) for token, name in seen.items()), key=lambda item: item[0])


def _http_post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class GeminiClient:
    """One generateContent call at a time; 429/quota rotates to the next profile. Counters feed the report."""

    def __init__(self, *, keys: list[tuple[str, str]] | None = None, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, post: Callable | None = None,
                 sleep: Callable[[float], None] = time.sleep, timeout: float = 120):
        self.keys = list(keys) if keys is not None else load_keys()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._post = post or _http_post
        self._sleep = sleep
        self.timeout = timeout
        self.index = 0
        self.calls = 0
        self.errors = 0
        self.rotations = 0

    def _secrets(self) -> list[str]:
        return [key for _, key in self.keys]

    def profile(self) -> str:
        return self.keys[self.index][0] if self.keys else ""

    def generate(self, system: str, user: dict) -> str:
        if not self.keys:
            raise GeminiError("no Gemini API key (set GEMINI_API_KEY or sign the google profile in OpenClaw)")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(user, ensure_ascii=False)}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        url = f"{self.base_url}/models/{self.model}:generateContent"
        tried_keys = 0
        transient = 0
        while True:
            self.calls += 1
            try:
                payload = self._post(url, body, {"x-goog-api-key": self.keys[self.index][1]}, self.timeout)
            except HTTPError as exc:
                self.errors += 1
                if exc.code in (429, 403):
                    tried_keys += 1
                    if tried_keys >= len(self.keys):
                        raise GeminiError(f"Gemini quota/rate limit on all {len(self.keys)} profile(s) (HTTP {exc.code})") from None
                    self.index = (self.index + 1) % len(self.keys)
                    self.rotations += 1
                    continue
                if exc.code >= 500 and transient < 2:
                    transient += 1
                    self._sleep(2 * transient)
                    continue
                raise GeminiError(f"Gemini HTTP {exc.code}") from None
            except (URLError, TimeoutError, ConnectionError, OSError) as exc:
                self.errors += 1
                if transient < 2:
                    transient += 1
                    self._sleep(2 * transient)
                    continue
                raise GeminiError(f"Gemini unreachable: {type(exc).__name__}") from None
            try:
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                self.errors += 1
                raise GeminiError("Gemini returned no candidate text") from None
            return str(text)


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        value = json.loads(text)
    except ValueError:
        raise ValueError("Gemini output is not JSON") from None
    if not isinstance(value, dict):
        raise ValueError("Gemini output is not a JSON object")
    return value


def _clean(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"empty translation at {where}")
    value = value.strip()
    if any(c in value for c in ("\n", "\r", "\x00", "�")):
        raise ValueError(f"multiline/control translation at {where}")
    return value


def validate_rows(ids: list[str], parsed: dict) -> list[str]:
    """The same IDs in the same order, no empty row, no newline. Anything else raises ValueError."""
    rows = parsed.get("rows")
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError(f"row count {len(rows) if isinstance(rows, list) else 'n/a'} != {len(ids)}")
    out = []
    for want, row in zip(ids, rows):
        if not isinstance(row, dict) or row.get("id") != want:
            raise ValueError(f"row id mismatch at {want}")
        out.append(_clean(row.get("text"), want))
    return out


def _translate_chunk(client: GeminiClient, lines: list[str], ids: list[str], start: int, end: int,
                     title: str, artist: str) -> list[str]:
    """ids[start:end] translated; a mismatch retries with half the chunk (down to one row)."""
    request = {
        "title": title, "artist": artist,
        "whole_song_for_meaning_only": [{"id": i, "text": t} for i, t in zip(ids, lines)],
        "translate_ids": ids[start:end],
    }
    try:
        return validate_rows(ids[start:end], _parse(client.generate(SYSTEM_PROMPT, request)))
    except ValueError as exc:
        if end - start <= 1:
            raise GeminiError(f"row {ids[start]} failed validation: {mask(exc, client._secrets())}") from None
        mid = (start + end) // 2
        return (_translate_chunk(client, lines, ids, start, mid, title, artist)
                + _translate_chunk(client, lines, ids, mid, end, title, artist))


def translate_gemini(original: list[str], *, title: str = "", artist: str = "", model: str = DEFAULT_MODEL,
                     client: GeminiClient | None = None) -> list[str]:
    lines = [str(x) for x in original]
    if not lines:
        return []
    client = client or GeminiClient(model=model)
    ids = [f"L{n:03d}" for n in range(1, len(lines) + 1)]
    out: list[str] = []
    for start in range(0, len(lines), MAX_CHUNK):
        out += _translate_chunk(client, lines, ids, start, min(start + MAX_CHUNK, len(lines)), title, artist)
    return out


def translate_gemini_line(line: str, *, before: list[str] = (), after: list[str] = (), title: str = "",
                          client: GeminiClient | None = None, model: str = DEFAULT_MODEL) -> str:
    """One line, neighbours given as 'context, do not translate'. One retry on a bad answer."""
    client = client or GeminiClient(model=model)
    request = {"title": title, "context_before": list(before), "context_after": list(after),
               "translate_this_line_only": line}
    last = "no answer"
    for _ in range(2):
        try:
            parsed = _parse(client.generate(LINE_PROMPT, request))
            if set(parsed) != {"text"}:
                raise ValueError("unexpected fields")
            return _clean(parsed["text"], "line")
        except ValueError as exc:
            last = mask(exc, client._secrets())
    raise GeminiError(f"one-line translation failed: {last}")


def line_retranslator(client: GeminiClient, title: str = "") -> Callable:
    """The QA gate's re-ask: Gemini for that single line, neighbours as context only."""
    def retranslate(line: str, context: tuple[list[str], list[str]] = ((), ())) -> str:
        return translate_gemini_line(line, before=list(context[0]), after=list(context[1]), title=title, client=client)
    retranslate.accepts_context = True
    return retranslate


def translate_auto(original: list[str], *, title: str = "", artist: str = "", translator: str = "auto",
                   gemini: GeminiClient | None = None, oogg: Callable[[list[str]], tuple[list[str], str]] | None = None,
                   ) -> TranslationResult:
    """Gemini first, then OOGG. ``oogg(lines) -> (translations, model)`` picks whichever host is up.

    translator: 'gemini' (no fallback), 'oogg' (no Gemini) or 'auto'. Raises RuntimeError when nothing worked;
    the message names each backend's failure, masked.
    """
    failures: list[str] = []
    if translator in ("auto", "gemini"):
        try:
            client = gemini or GeminiClient()
            return TranslationResult(translate_gemini(original, title=title, artist=artist, client=client),
                                     "gemini", client.model)
        except Exception as exc:
            failures.append(f"gemini: {mask(exc, (gemini._secrets() if gemini else ()))[:200]}")
    if translator in ("auto", "oogg") and oogg is not None:
        try:
            lines, model = oogg(list(original))
            return TranslationResult(lines, "oogg", model)
        except Exception as exc:
            failures.append(f"oogg: {type(exc).__name__}: {mask(exc)[:200]}")
    elif translator in ("auto", "oogg"):
        failures.append("oogg: no host")
    raise RuntimeError("; ".join(failures) or "no translator selected")
