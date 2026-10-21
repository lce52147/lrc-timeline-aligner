from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import NamedTuple

try:
    from lyricsee_core import (
        HttpClient,
        LyricCandidate,
        choose_candidate,
        collect_candidates,
        google_search_url,
        probe_audio,
        translate_with_local_llm,
        review_translations_with_local_llm,
        translation_coverage,
        write_result,
    )
except ModuleNotFoundError:
    from lyricsee.lyricsee_core import (
        HttpClient,
        LyricCandidate,
        choose_candidate,
        collect_candidates,
        google_search_url,
        probe_audio,
        translate_with_local_llm,
        review_translations_with_local_llm,
        translation_coverage,
        write_result,
    )


try:
    from source_policy import (
        choose_verified_original, choose_verified_web_translation,
        validate_output, attach_verification, verified_provenance_is_current,
    )
except ModuleNotFoundError:
    from lyricsee.source_policy import (
        choose_verified_original, choose_verified_web_translation,
        validate_output, attach_verification, verified_provenance_is_current,
    )


HERE = Path(__file__).resolve().parent
REPORT_DIR = HERE / "report"
TXT_DIR = HERE / "txt"


class EmbeddedInfo(NamedTuple):
    artist: str
    title: str
    has_lyrics: bool
    timed_lines: int
    lyrics_chars: int


def playlist_paths(playlist: Path) -> list[Path]:
    playlist = Path(playlist)
    paths: list[Path] = []
    for raw in playlist.read_text(encoding="utf-8-sig").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        path = Path(raw.replace("/", "\\"))
        paths.append(path if path.is_absolute() else playlist.parent / path)
    return paths


def probe_embedded_lyrics(path: Path, *, runner=subprocess.run) -> EmbeddedInfo:
    args = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format_tags=title,artist,lyrics,unsyncedlyrics",
        "-of",
        "json",
        str(path),
    ]
    completed = runner(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({completed.returncode})")
    payload = json.loads(completed.stdout)
    tags = {
        str(key).casefold(): value
        for key, value in ((payload.get("format") or {}).get("tags") or {}).items()
    }
    lyrics = str(tags.get("lyrics") or tags.get("unsyncedlyrics") or "").strip()
    timed_lines = len(re.findall(r"(?m)^\s*\[\d{1,3}:\d{2}(?:[.:]\d{1,3})?\]", lyrics))
    return EmbeddedInfo(
        artist=str(tags.get("artist") or "").strip(),
        title=str(tags.get("title") or path.stem).strip(),
        has_lyrics=bool(lyrics),
        timed_lines=timed_lines,
        lyrics_chars=len(lyrics),
    )


def _md(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_markdown(rows: list[dict], *, playlist: Path) -> str:
    counts = Counter(str(row.get("status") or "") for row in rows)
    lines = [
        f"# {playlist.stem} LyricSee comparison",
        "",
        f"- Playlist: `{playlist}`",
        f"- Entries recorded: {len(rows)}",
        f"- Embedded LYRICS skipped: {counts.get('embedded_lyrics', 0)}",
        f"- Newly generated: {counts.get('generated', 0)}",
        f"- Existing LyricSee output reused: {counts.get('existing_output', 0)}",
        f"- Duplicate playlist entries: {counts.get('duplicate_playlist_entry', 0)}",
        f"- Failed: {counts.get('failed', 0)}",
        "",
        "Embedded LYRICS entries are preserved as references. By default, generation uses independently checked sources and existing online translations without any model calls. Missing verified translations fail closed; local LLM use requires explicit --mode llm.",
        "",
        "| # | Track | Embedded ref lines | Status | Source | Generated lines | Coverage | Output / error |",
        "|---:|---|---:|---|---|---:|---:|---|",
    ]
    for row in rows:
        coverage = row.get("coverage")
        coverage_text = "" if coverage is None else f"{float(coverage) * 100:.1f}%"
        track = f"{row.get('artist', '')} - {row.get('title', '')}".strip(" -")
        detail = row.get("output") or row.get("error") or ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("index", "")),
                    _md(track),
                    str(row.get("reference_lines", 0)),
                    _md(row.get("status", "")),
                    _md(row.get("source", "")),
                    str(row.get("line_count", 0)),
                    coverage_text,
                    _md(detail),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _model_id(client: HttpClient, endpoint: str) -> str:
    payload = client.get_json(endpoint.rstrip("/") + "/v1/models")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not data or not isinstance(data, list) or not data[0].get("id"):
        raise RuntimeError("The local endpoint returned no model")
    return str(data[0]["id"])


def _record_from_provenance(index: int, info: EmbeddedInfo, provenance: Path, output: Path) -> dict:
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    return {
        "index": index,
        "artist": info.artist or str(payload.get("artist") or ""),
        "title": info.title or str(payload.get("title") or output.stem),
        "status": "existing_output",
        "reference_lines": 0,
        "source": str(payload.get("selected_source") or ""),
        "line_count": int(payload.get("line_count") or 0),
        "coverage": payload.get("translation_coverage"),
        "output": str(output.relative_to(HERE)),
        "error": "",
    }


def _save_reports(playlist: Path, rows: list[dict]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state = REPORT_DIR / f"{playlist.stem}_batch.json"
    comparison = REPORT_DIR / f"{playlist.stem}_comparison.md"
    state.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comparison.write_text(render_markdown(rows, playlist=playlist), encoding="utf-8")
    return state, comparison


def run_batch(
    playlist: Path,
    *,
    endpoint: str = "http://127.0.0.1:6969",
    limit: int | None = None,
    mode: str = "web",
) -> list[dict]:
    if mode not in ("web", "llm"):
        raise ValueError("mode must be web or llm")
    paths = playlist_paths(playlist)
    if limit is not None:
        paths = paths[:limit]

    client = HttpClient()
    model: str | None = None
    rows: list[dict] = []
    seen: dict[str, int] = {}

    for index, path in enumerate(paths, start=1):
        key = str(path.resolve()).casefold()
        if key in seen:
            rows.append(
                {
                    "index": index,
                    "artist": "",
                    "title": path.stem,
                    "status": "duplicate_playlist_entry",
                    "reference_lines": 0,
                    "source": "",
                    "line_count": 0,
                    "coverage": None,
                    "output": "",
                    "error": f"same path as entry {seen[key]}",
                }
            )
            _save_reports(playlist, rows)
            print(f"{index}/{len(paths)} duplicate")
            continue
        seen[key] = index

        try:
            info = probe_embedded_lyrics(path)
            if info.has_lyrics:
                rows.append(
                    {
                        "index": index,
                        "artist": info.artist,
                        "title": info.title,
                        "status": "embedded_lyrics",
                        "reference_lines": info.timed_lines,
                        "source": "embedded:LYRICS",
                        "line_count": 0,
                        "coverage": None,
                        "output": "",
                        "error": "",
                    }
                )
                _save_reports(playlist, rows)
                print(f"{index}/{len(paths)} embedded")
                continue

            output = TXT_DIR / f"{path.stem}.lyrics.txt"
            provenance = REPORT_DIR / f"{path.stem}.lyrics.sources.json"
            if output.exists() or provenance.exists():
                if output.exists() and provenance.exists():
                    previous = json.loads(provenance.read_text(encoding="utf-8"))
                    verified = verified_provenance_is_current(
                        previous, output.read_bytes()
                    )
                    row = _record_from_provenance(index, info, provenance, output)
                    row["status"] = "verified_output" if verified else "legacy_unverified"
                    if not verified:
                        row["error"] = (
                            "Old source is not independently verified by the "
                            "NetEase-free policy; existing TXT was left unchanged"
                        )
                    rows.append(row)
                    _save_reports(playlist, rows)
                    print(f"{index}/{len(paths)} {row['status']}")
                    continue
                raise RuntimeError("partial existing LyricSee output; refusing ambiguous overwrite")

            identity = probe_audio(path)
            collected = collect_candidates(identity, client)
            if not collected.candidates:
                detail = "; ".join(collected.errors) or "no usable lyric source"
                raise LookupError(detail)

            original, evidence = choose_verified_original(
                identity, collected.candidates
            )
            if mode == "web":
                selected = choose_verified_web_translation(
                    identity, original, collected.candidates
                )
                method = "existing_web_translation_source_matched"
            else:
                if model is None:
                    model = _model_id(client, endpoint)
                translated = translate_with_local_llm(
                    [line.original for line in original.lines],
                    title=identity.title, endpoint=endpoint,
                    model=model, post_json=client.post_json,
                )
                translated = review_translations_with_local_llm(
                    translated, title=identity.title, endpoint=endpoint,
                    model=model, post_json=client.post_json,
                )
                selected = LyricCandidate(
                    source=f"local-llm:{model}", url=original.url,
                    title=identity.title, artist=identity.artist,
                    lines=translated,
                    discovered_urls=collected.google_urls,
                    google_search=google_search_url(identity),
                    translator=model,
                )
                method = "local_llm_verified_original_reviewed"
            validate_output(original, selected)
            attach_verification(
                selected, evidence, translation_method=method
            )

            selected.discovered_urls = list(collected.google_urls)
            selected.google_search = selected.google_search or google_search_url(identity)
            output, provenance = write_result(path, selected, overwrite=False)
            rows.append(
                {
                    "index": index,
                    "artist": identity.artist or selected.artist,
                    "title": identity.title or selected.title,
                    "status": "generated",
                    "reference_lines": 0,
                    "source": selected.source,
                    "line_count": len(selected.lines),
                    "coverage": translation_coverage(selected.lines),
                    "output": str(output.relative_to(HERE)),
                    "error": "",
                }
            )
            print(f"{index}/{len(paths)} generated")
        except Exception as exc:
            rows.append(
                {
                    "index": index,
                    "artist": "",
                    "title": path.stem,
                    "status": "failed",
                    "reference_lines": 0,
                    "source": "",
                    "line_count": 0,
                    "coverage": None,
                    "output": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"{index}/{len(paths)} failed")
        _save_reports(playlist, rows)

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch LyricSee processing for an M3U8 playlist.")
    parser.add_argument("playlist", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:6969")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=("web", "llm"), default="web",
                        help="web is deterministic/no-LLM default; llm requires explicit opt-in")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = run_batch(args.playlist, endpoint=args.endpoint, limit=args.limit, mode=args.mode)
    failures = sum(row["status"] == "failed" for row in rows)
    print(f"done={len(rows)} failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
