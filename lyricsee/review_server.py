"""Web server for reviewing candidate lyrics."""

from __future__ import annotations

import html
import http.server
import json
import os
import socketserver
import threading
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

from candidate_review import load_candidates_metadata, load_review_decisions, record_review_decision, accept_candidate, reject_candidate


class ReviewServerConfig(NamedTuple):
    output_root: Path
    host: str = "127.0.0.1"
    port: int = 0  # Auto-select free port


def find_free_port() -> int:
    """Find a free port."""
    with socketserver.TCPServer(("127.0.0.1", 0), socketserver.BaseRequestHandler) as s:
        return s.server_address[1]


class ReviewHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the review server."""

    output_root: Path = Path()

    def do_GET(self) -> None:
        """Handle GET requests."""
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)

        if path == "/" or path == "/index.html":
            self.serve_review_page()
        elif path == "/api/songs":
            self.serve_songs_api()
        elif path == "/api/candidates":
            stem = query.get("stem", [""])[0]
            self.serve_candidates_api(stem)
        else:
            self.send_error(404, "Not found")

    def do_POST(self) -> None:
        """Handle POST requests."""
        path = urlparse(self.path).path

        if path == "/api/decide":
            self.handle_decision()
        else:
            self.send_error(404, "Not found")

    def serve_review_page(self) -> None:
        """Serve the main review page."""
        html_content = self.generate_review_html()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html_content.encode("utf-8")))
        self.end_headers()
        self.wfile.write(html_content.encode("utf-8"))

    def serve_songs_api(self) -> None:
        """Serve list of songs needing review."""
        candidates_dir = self.output_root / "candidates"
        songs = []

        if candidates_dir.exists():
            for stem_dir in candidates_dir.iterdir():
                if stem_dir.is_dir():
                    metadata = load_candidates_metadata(self.output_root, stem_dir.name)
                    if metadata:
                        songs.append({"stem": stem_dir.name, "candidate_count": len(metadata)})

        self.send_json_response({"songs": songs})

    def serve_candidates_api(self, stem: str) -> None:
        """Serve candidates for a song."""
        metadata = load_candidates_metadata(self.output_root, stem)
        if not metadata:
            self.send_error(404, "Song not found")
            return

        candidates = []
        candidates_dir = self.output_root / "candidates" / stem

        for idx, meta in enumerate(metadata):
            lyric_file = candidates_dir / f"candidate_{idx}.txt"
            lyrics = ""
            if lyric_file.exists():
                lyrics = lyric_file.read_text(encoding="utf-8")

            candidates.append({
                "idx": idx,
                "url": meta.get("url"),
                "host": meta.get("host"),
                "source": meta.get("source"),
                "line_count": meta.get("line_count", 0),
                "char_count": meta.get("char_count", 0),
                "title_match": meta.get("title_match"),
                "artist_match": meta.get("artist_match"),
                "lyrics": lyrics[:500],  # Preview only
                "similarities": meta.get("similarity_to_others", {}),
            })

        self.send_json_response({"stem": stem, "candidates": candidates})

    def handle_decision(self) -> None:
        """Handle a review decision."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8"))
            stem = data.get("stem")
            decision = data.get("decision")  # "accept" or "reject"
            candidate_idx = data.get("candidate_idx")

            if not stem or not decision:
                self.send_error(400, "Missing stem or decision")
                return

            if decision not in ("accept", "reject"):
                self.send_error(400, f"Invalid decision: {decision}")
                return

            record_review_decision(self.output_root, stem, decision, candidate_idx)
            self.send_json_response({"status": "ok"})
        except Exception as e:
            self.send_error(400, f"Error: {str(e)}")

    def send_json_response(self, data: dict) -> None:
        """Send a JSON response."""
        response = json.dumps(data, ensure_ascii=False, indent=2)
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(response.encode("utf-8")))
        self.end_headers()
        self.wfile.write(response.encode("utf-8"))

    def generate_review_html(self) -> str:
        """Generate the review page HTML (Traditional Chinese)."""
        return """<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>歌詞候選審核</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f5f5; color: #333; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #1a1a1a; margin-bottom: 30px; }
        .song-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 20px; }
        .song-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .song-title { font-size: 18px; font-weight: bold; margin-bottom: 15px; }
        .candidates-list { margin-top: 15px; }
        .candidate { padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px; background: #fafafa; }
        .candidate-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .host-badge { background: #007bff; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        .match-info { font-size: 12px; color: #666; margin-bottom: 8px; }
        .candidate-actions { display: flex; gap: 10px; }
        .btn { padding: 8px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .btn-accept { background: #28a745; color: white; }
        .btn-accept:hover { background: #218838; }
        .btn-reject { background: #dc3545; color: white; }
        .btn-reject:hover { background: #c82333; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .song-status { font-size: 12px; color: #999; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>歌詞候選審核</h1>
        <div id="loading">載入中...</div>
        <div class="song-grid" id="songs-container"></div>
    </div>
    <script>
        async function loadSongs() {
            try {
                const res = await fetch('/api/songs');
                const data = await res.json();
                document.getElementById('loading').style.display = 'none';
                renderSongs(data.songs);
            } catch (e) {
                document.getElementById('loading').textContent = '載入失敗: ' + e.message;
            }
        }

        function renderSongs(songs) {
            const container = document.getElementById('songs-container');
            if (!songs || songs.length === 0) {
                container.innerHTML = '<p>沒有需要審核的歌曲</p>';
                return;
            }

            songs.forEach(song => {
                const card = document.createElement('div');
                card.className = 'song-card';
                card.innerHTML = `<div class="song-title">${song.stem}</div>
                    <div class="song-status">${song.candidate_count} 個候選</div>
                    <div class="candidates-list" id="cand-${song.stem}">載入中...</div>`;
                container.appendChild(card);
                loadCandidates(song.stem);
            });
        }

        async function loadCandidates(stem) {
            try {
                const res = await fetch('/api/candidates?stem=' + encodeURIComponent(stem));
                const data = await res.json();
                renderCandidates(stem, data.candidates);
            } catch (e) {
                document.getElementById('cand-' + stem).textContent = '載入失敗: ' + e.message;
            }
        }

        function renderCandidates(stem, candidates) {
            const container = document.getElementById('cand-' + stem);
            container.innerHTML = '';

            candidates.forEach(c => {
                const el = document.createElement('div');
                el.className = 'candidate';
                el.innerHTML = `<div class="candidate-header">
                    <span class="host-badge">${c.host}</span>
                    <span>${c.line_count} 行</span>
                </div>
                <div class="match-info">
                    來源: ${c.source}<br>
                    標題匹配: ${c.title_match} | 藝人匹配: ${c.artist_match}
                </div>
                <div class="candidate-actions">
                    <button class="btn btn-accept" onclick="acceptCandidate('${stem}', ${c.idx})">採用</button>
                    <button class="btn btn-reject" onclick="rejectCandidate('${stem}')">都不對</button>
                </div>`;
                container.appendChild(el);
            });
        }

        async function acceptCandidate(stem, idx) {
            await submitDecision(stem, 'accept', idx);
        }

        async function rejectCandidate(stem) {
            await submitDecision(stem, 'reject', null);
        }

        async function submitDecision(stem, decision, candidateIdx) {
            try {
                const res = await fetch('/api/decide', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({stem, decision, candidate_idx: candidateIdx})
                });
                if (res.ok) {
                    alert('決定已記錄');
                    location.reload();
                } else {
                    alert('錯誤: ' + res.statusText);
                }
            } catch (e) {
                alert('錯誤: ' + e.message);
            }
        }

        loadSongs();
    </script>
</body>
</html>"""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress logging."""
        pass


def start_review_server(config: ReviewServerConfig) -> tuple[str, threading.Thread | None]:
    """Start the review server and return its URL.

    Returns:
        Tuple of (URL, server_thread) or just URL if auto_port is False
    """
    port = config.port if config.port else find_free_port()
    ReviewHandler.output_root = config.output_root

    try:
        server = socketserver.TCPServer((config.host, port), ReviewHandler)
        url = f"http://{config.host}:{port}"

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        return url, server
    except Exception as e:
        raise RuntimeError(f"Failed to start review server: {e}")
