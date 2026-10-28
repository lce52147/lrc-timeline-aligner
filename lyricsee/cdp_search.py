"""Google result links through an already-running Chrome (DevTools port), as the last search fallback.

Plain HTTP search first (web_search); only when that finds no full-length, identity-confirmed page may this
backend ask Google for the result list.  It touches the browser the narrow way EvanOG authorised:

* ``Target.createTarget`` with ``background: true`` -> ONE new tab for the whole run, never activated;
* that tab navigates to the Google results page and the result links are read from its DOM;
* the tab is closed at the end, also after errors.

Other tabs are never listed, attached, focused, navigated or closed.  Chrome is never started, restarted or
killed: when the port does not answer the backend is simply unavailable.  A CAPTCHA / consent page stops the
backend for the run; it is not solved.  Result pages are fetched by the normal HTTP fetchers.
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import socket
import struct
import time
import urllib.request
from urllib.parse import quote, urlparse

CDP_PORT = 9222
CDP_MAX_QUERIES = 8
CDP_INTERVAL = 5.0   # seconds between two Google queries
NAV_TIMEOUT = 25.0

# Anchors of the organic result list; Google's own links are dropped by the caller.
_LINKS_JS = (
    "JSON.stringify({url: location.href, ready: document.readyState, title: document.title,"
    " text: (document.body ? document.body.innerText : '').slice(0, 600),"
    " links: Array.from(document.querySelectorAll('#search a[href], #rso a[href]')).map(a => a.href)})"
)
# Rendered result page of a lyric site: the visible text plus the text of the known lyric containers.
RENDER_MAX_LOADS = 6        # page loads per song
RENDER_SETTLE = 1.0         # seconds between two reads while the page keeps growing
_PAGE_JS = (
    "JSON.stringify({url: location.href, ready: document.readyState, title: document.title,"
    " text: (document.body ? document.body.innerText : '').slice(0, 200000),"
    " containers: Array.from(document.querySelectorAll('[data-lyrics-container], .lyrics, #lyrics, .lyric-text,"
    " .lyricbody, .entry-content, article')).map(e => e.innerText).filter(t => t && t.length > 20)"
    ".sort((a, b) => b.length - a.length).slice(0, 3)})"
)
_PAGE_BLOCK_MARKERS = ("just a moment", "verify you are human", "checking your browser", "access denied",
                       "attention required", "enable javascript and cookies")
_BLOCK_MARKERS = ("unusual traffic", "recaptcha", "我们的系统检测到", "通常と異なるトラフィック", "before you continue")


class CdpUnavailable(RuntimeError):
    """The DevTools port does not answer (Chrome not running with it)."""


class CdpBlocked(RuntimeError):
    """Google answered with a CAPTCHA / consent page."""


class _WebSocket:
    """Just enough RFC 6455 client for DevTools: text frames, masked sends, fragmented replies."""

    def __init__(self, url: str, timeout: float = 15.0) -> None:
        parsed = urlparse(url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port or 80), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        path = parsed.path + (("?" + parsed.query) if parsed.query else "")
        self.sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {parsed.hostname}:{parsed.port}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise CdpUnavailable("websocket handshake closed")
            head += chunk
        header, _, self._buffer = head.partition(b"\r\n\r\n")
        if b" 101 " not in header.split(b"\r\n", 1)[0]:
            raise CdpUnavailable("websocket handshake refused")

    def _read(self, count: int) -> bytes:
        while len(self._buffer) < count:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise CdpUnavailable("websocket closed")
            self._buffer += chunk
        data, self._buffer = self._buffer[:count], self._buffer[count:]
        return data

    def send(self, text: str) -> None:
        payload = text.encode("utf-8")
        mask = os.urandom(4)
        length = len(payload)
        head = bytes([0x81])
        if length < 126:
            head += bytes([0x80 | length])
        elif length < 65536:
            head += bytes([0x80 | 126]) + struct.pack(">H", length)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", length)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(head + mask + masked)

    def recv(self) -> str:
        message = b""
        while True:
            first, second = self._read(2)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(length)
            opcode = first & 0x0F
            if opcode == 0x8:
                raise CdpUnavailable("websocket closed by Chrome")
            if opcode in (0x9, 0xA):
                continue
            message += payload
            if first & 0x80:
                return message.decode("utf-8", "replace")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def _http_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def google_search_page(query: str) -> str:
    return f"https://www.google.com/search?q={quote(query)}&hl=ja"


def is_blocked_page(url: str, title: str, text: str) -> bool:
    host = urlparse(url).hostname or ""
    if "/sorry/" in url or host.startswith("consent.") or "consent.google" in host:
        return True
    lowered = (title + " " + text).lower()
    return any(marker.lower() in lowered for marker in _BLOCK_MARKERS)


class CdpGoogle:
    """One background tab in the running Chrome; ``search`` returns the organic result URLs."""

    def __init__(self, port: int = CDP_PORT, *, connect=_WebSocket, http_json=_http_json, sleep=time.sleep,
                 clock=time.monotonic, interval: float = CDP_INTERVAL, nav_timeout: float = NAV_TIMEOUT) -> None:
        self.port = port
        self._connect = connect
        self._http_json = http_json
        self.sleep = sleep
        self.clock = clock
        self.interval = interval
        self.nav_timeout = nav_timeout
        self.ws = None
        self.target_id = ""
        self.session_id = ""
        self.blocked = False
        self.opened_targets: list[str] = []
        self.closed_targets: list[str] = []
        self.queries_sent = 0
        self.pages_loaded = 0
        self._next_id = 0
        self._last_query = float("-inf")

    # -- protocol ------------------------------------------------------------
    def _call(self, method: str, params: dict | None = None, *, session: bool = False) -> dict:
        self._next_id += 1
        message = {"id": self._next_id, "method": method, "params": params or {}}
        if session:
            message["sessionId"] = self.session_id
        self.ws.send(json.dumps(message))
        while True:
            reply = json.loads(self.ws.recv())
            if reply.get("id") == self._next_id:
                if "error" in reply:
                    raise CdpUnavailable(f"{method}: {reply['error']}")
                return reply.get("result", {})

    def open(self) -> None:
        """Connect to the browser endpoint and open the one background tab.  Raises CdpUnavailable."""
        if self.target_id:
            return
        try:
            version = self._http_json(f"http://127.0.0.1:{self.port}/json/version")
            self.ws = self._connect(version["webSocketDebuggerUrl"])
        except CdpUnavailable:
            raise
        except Exception as exc:
            raise CdpUnavailable(f"port {self.port} not answering: {exc}") from exc
        try:
            created = self._call("Target.createTarget", {"url": "about:blank", "background": True})
            self.target_id = created["targetId"]
            self.opened_targets.append(self.target_id)
            atexit.register(self.close)
            attached = self._call("Target.attachToTarget", {"targetId": self.target_id, "flatten": True})
            self.session_id = attached["sessionId"]
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Close only the tab this object opened; safe to call repeatedly."""
        try:
            if self.ws is not None and self.target_id and self.target_id not in self.closed_targets:
                try:
                    self._call("Target.closeTarget", {"targetId": self.target_id})
                except Exception:
                    pass
                self.closed_targets.append(self.target_id)
        finally:
            if self.ws is not None:
                self.ws.close()
            self.ws = None
            self.target_id = ""
            self.session_id = ""

    # -- rendered pages --------------------------------------------------------
    def fetch_page(self, url: str) -> dict:
        """Load ``url`` in the run's own tab and return {url, title, text, containers} (rendered text).

        Same tab and spacing as ``search``; a challenge page comes back with empty text and ``blocked`` set,
        it is not solved.  A Google challenge seen by ``search`` does not stop page loads.  Raises CdpUnavailable."""
        wait = self._last_query + self.interval - self.clock()
        if wait > 0:
            self.sleep(wait)
        self.open()
        self._last_query = self.clock()
        self.pages_loaded += 1
        self._call("Page.navigate", {"url": url}, session=True)
        deadline = self.clock() + self.nav_timeout
        page: dict = {}
        previous = -1
        while True:
            raw = self._call("Runtime.evaluate", {"expression": _PAGE_JS, "returnByValue": True}, session=True)
            page = json.loads(raw.get("result", {}).get("value") or "{}")
            size = len(page.get("text") or "")
            loaded = page.get("ready") == "complete" and (page.get("url") or "").startswith("http")
            if loaded and size == previous:     # complete and no longer growing (client-rendered pages)
                break
            previous = size if loaded else -1
            if self.clock() > deadline:
                break
            self.sleep(RENDER_SETTLE)
        title, text = page.get("title") or "", page.get("text") or ""
        lowered = (title + " " + text[:400]).lower()
        blocked = any(marker in lowered for marker in _PAGE_BLOCK_MARKERS) and len(text) < 2000
        return {"url": page.get("url") or url, "title": title, "text": "" if blocked else text,
                "containers": [] if blocked else list(page.get("containers") or []), "blocked": blocked}

    # -- search --------------------------------------------------------------
    def search(self, query: str) -> list[str]:
        """Result URLs for ``query``.  Raises CdpUnavailable, or CdpBlocked (backend then stays stopped)."""
        if self.blocked:
            raise CdpBlocked("Google challenge seen earlier in this run")
        wait = self._last_query + self.interval - self.clock()
        if wait > 0:
            self.sleep(wait)
        self.open()
        self._last_query = self.clock()
        self.queries_sent += 1
        self._call("Page.navigate", {"url": google_search_page(query)}, session=True)
        deadline = self.clock() + self.nav_timeout
        page: dict = {}
        while True:
            raw = self._call("Runtime.evaluate", {"expression": _LINKS_JS, "returnByValue": True}, session=True)
            page = json.loads(raw.get("result", {}).get("value") or "{}")
            if page.get("ready") == "complete" and "google." in (page.get("url") or ""):
                break
            if self.clock() > deadline:
                break
            self.sleep(0.5)
        if is_blocked_page(page.get("url", ""), page.get("title", ""), page.get("text", "")):
            self.blocked = True
            raise CdpBlocked(page.get("url", ""))
        return [link for link in page.get("links", []) if link.startswith("http")]
