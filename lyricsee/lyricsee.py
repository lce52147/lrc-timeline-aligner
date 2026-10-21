from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import NamedTuple
from urllib.error import HTTPError, URLError

from lyricsee_core import (
    HttpClient,
    LyricCandidate,
    LyricPair,
    SUPPORTED_AUDIO_EXTENSIONS,
    _atomic_write_text,
    collect_candidates,
    google_search_url,
    probe_audio,
    write_result,
)
from source_policy import (
    attach_verification,
    choose_verified_original,
    choose_verified_web_translation,
    validate_output,
    verified_provenance_is_current,
)
from translation_oogg import translate_oogg


HERE = Path(__file__).resolve().parent
DEFAULT_BK_DIR = Path(r"D:\Users\Administrator\Music\LRC tools checked references\LRC bk")
DEFAULT_OOGG_ENDPOINTS = (
    "http://127.0.0.1:6969",
    "http://100.87.97.121:6969",
)
RUN_FIELDS = (
    "index",
    "audio",
    "stem",
    "status",
    "translation_method",
    "translation_error",
    "oogg_endpoint",
    "line_count",
    "original_source_url",
    "google_search",
    "embedded_lyrics_lines",
    "error",
    "output_txt",
    "output_json",
)


class EmbeddedInfo(NamedTuple):
    artist: str
    title: str
    has_lyrics: bool
    timed_lines: int
    lyrics_chars: int


class Config(NamedTuple):
    bk_dir: Path
    output_root: Path
    oogg_endpoints: tuple[str, ...]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find verified lyrics for audio files or M3U8 playlists and optionally translate them."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="audio|playlist.m3u8",
        help="Audio path(s) and/or .m3u8 playlist path(s)",
    )
    parser.add_argument(
        "--bk-dir",
        help="Checked-reference LRC directory (CLI > LYRICSEE_BK_DIR > built-in default)",
    )
    parser.add_argument(
        "--output-root",
        help="Output root containing txt/ and report/ (CLI > LYRICSEE_OUTPUT_ROOT > script directory)",
    )
    parser.add_argument(
        "--endpoint",
        help=(
            "Force one OOGG endpoint; otherwise LYRICSEE_OOGG_ENDPOINT overrides "
            "automatic local-then-Sister probing"
        ),
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Write verified original lyrics only; do not call OOGG",
    )
    parser.add_argument("--overwrite", action="store_true", help="Redo existing and legacy outputs")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Process at most N tracks that reach audio probing; BK/existing/duplicates do not count",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only evaluate BK/existing/duplicate skips; no ffprobe, HTTP, or file writes",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> Config:
    bk_value = args.bk_dir or os.environ.get("LYRICSEE_BK_DIR")
    output_value = args.output_root or os.environ.get("LYRICSEE_OUTPUT_ROOT")
    endpoint = args.endpoint or os.environ.get("LYRICSEE_OOGG_ENDPOINT")
    oogg_endpoints = (
        (endpoint.rstrip("/"),)
        if endpoint
        else DEFAULT_OOGG_ENDPOINTS
    )
    return Config(
        bk_dir=Path(bk_value).expanduser() if bk_value else DEFAULT_BK_DIR,
        output_root=Path(output_value).expanduser() if output_value else HERE,
        oogg_endpoints=oogg_endpoints,
    )


def playlist_paths(playlist: Path) -> list[Path]:
    playlist = Path(playlist).expanduser()
    paths: list[Path] = []
    for raw in playlist.read_text(encoding="utf-8-sig").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        path = Path(raw.replace("/", "\\"))
        paths.append(path if path.is_absolute() else playlist.parent / path)
    return paths


def probe_embedded_lyrics(path: Path, *, runner=subprocess.run) -> EmbeddedInfo:
    completed = runner(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=title,artist,lyrics,unsyncedlyrics",
            "-of",
            "json",
            str(path),
        ],
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


def _expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in values:
        path = Path(raw).expanduser()
        if path.suffix.casefold() == ".m3u8":
            paths.extend(playlist_paths(path))
        else:
            paths.append(path)
    return paths


def _bk_stems(bk_dir: Path) -> set[str]:
    if not bk_dir.is_dir():
        return set()
    return {path.stem.casefold() for path in bk_dir.glob("*.lrc") if path.is_file()}


def _output_paths(output_root: Path, audio_path: Path) -> tuple[Path, Path]:
    return (
        output_root / "txt" / f"{audio_path.stem}.lyrics.txt",
        output_root / "report" / f"{audio_path.stem}.lyrics.sources.json",
    )


def _translation_method(payload: dict) -> str:
    return str(
        payload.get("translation_method")
        or (payload.get("verification") or {}).get("translation_method")
        or ""
    )


def _inspect_existing(
    audio_path: Path,
    *,
    output_root: Path,
    overwrite: bool,
    no_translate: bool,
) -> tuple[str | None, dict, bool]:
    output, provenance = _output_paths(output_root, audio_path)
    if not output.exists() and not provenance.exists():
        return None, {}, False

    payload: dict = {}
    if provenance.is_file():
        try:
            payload = json.loads(provenance.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    if overwrite:
        return None, payload, True

    if output.is_file() and provenance.is_file():
        try:
            current = verified_provenance_is_current(payload, output.read_bytes())
        except Exception:
            current = False
        if current:
            if _translation_method(payload) == "none" and not no_translate:
                return None, payload, True
            return "existing", payload, False

    return "legacy_unverified", payload, False


def _base_row(index: int, audio_path: Path, **updates: object) -> dict:
    row = {
        "index": index,
        "audio": str(audio_path),
        "stem": audio_path.stem,
        "status": "",
        "translation_method": "",
        "translation_error": "",
        "oogg_endpoint": "",
        "line_count": 0,
        "original_source_url": "",
        "google_search": "",
        "embedded_lyrics_lines": 0,
        "error": "",
        "output_txt": "",
        "output_json": "",
    }
    row.update(updates)
    return row


def _existing_row(index: int, audio_path: Path, output_root: Path, status: str, payload: dict) -> dict:
    output, provenance = _output_paths(output_root, audio_path)
    return _base_row(
        index,
        audio_path,
        status=status,
        translation_method=_translation_method(payload),
        translation_error=str(payload.get("translation_error") or ""),
        oogg_endpoint=str(payload.get("oogg_endpoint") or ""),
        line_count=int(payload.get("line_count") or 0),
        original_source_url=str(payload.get("source_url") or ""),
        google_search=str(payload.get("google_search_url") or ""),
        error="" if status == "existing" else "existing output is not current verified provenance",
        output_txt=str(output) if output.exists() else "",
        output_json=str(provenance) if provenance.exists() else "",
    )


def _oogg_error(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        url = str(getattr(exc, "url", "") or getattr(exc, "filename", ""))
        if url.rstrip("/").endswith("/v1/models"):
            return "oogg_unreachable"
        return f"oogg_http_{exc.code}"
    if isinstance(exc, (URLError, ConnectionError, TimeoutError)):
        return "oogg_unreachable"
    return f"{type(exc).__name__}: {str(exc)[:240]}"


def _select_oogg_endpoint(client: HttpClient, endpoints: tuple[str, ...]) -> str | None:
    for endpoint in endpoints:
        try:
            payload = client.get_json(endpoint.rstrip("/") + "/v1/models")
            models = payload.get("data") if isinstance(payload, dict) else None
            if (
                isinstance(models, list)
                and models
                and isinstance(models[0], dict)
                and models[0].get("id")
            ):
                return endpoint.rstrip("/")
        except Exception:
            continue
    return None


def _process_track(
    index: int,
    audio_path: Path,
    *,
    config: Config,
    no_translate: bool,
    replace_existing: bool,
    client: HttpClient,
) -> dict:
    embedded_lines = 0
    google = ""
    original_url = ""
    try:
        audio_path = audio_path.expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if audio_path.suffix.casefold() not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise ValueError(f"Unsupported audio extension {audio_path.suffix}; expected: {supported}")

        try:
            embedded_lines = probe_embedded_lyrics(audio_path).timed_lines
        except Exception:
            embedded_lines = 0

        identity = probe_audio(audio_path)
        google = google_search_url(identity)
        collected = collect_candidates(identity, client)
        try:
            original, evidence = choose_verified_original(identity, collected.candidates)
        except Exception as exc:
            details = list(collected.errors)
            details.append(f"{type(exc).__name__}: {exc}")
            return _base_row(
                index,
                audio_path,
                status="no_original",
                google_search=google,
                embedded_lyrics_lines=embedded_lines,
                error="; ".join(item for item in details if item),
            )

        original_url = original.url
        translation_method = "none"
        translation_error = ""
        oogg_endpoint = ""
        status = "original_only"

        if no_translate:
            selected = LyricCandidate(
                source=original.source,
                url=original.url,
                title=identity.title,
                artist=identity.artist,
                lines=[LyricPair(row.original) for row in original.lines],
                discovered_urls=list(collected.google_urls),
                google_search=google,
                translator=None,
            )
            validate_output(original, selected, require_translation=False)
            attach_verification(selected, evidence, translation_method="none")
        else:
            try:
                selected = choose_verified_web_translation(identity, original, collected.candidates)
                validate_output(original, selected, require_translation=True)
                translation_method = "web"
                status = "web"
                attach_verification(selected, evidence, translation_method="web")
            except Exception:
                oogg_endpoint = _select_oogg_endpoint(client, config.oogg_endpoints) or ""
                try:
                    if not oogg_endpoint:
                        raise ConnectionError("no configured OOGG endpoint is online")
                    translated = translate_oogg(
                        [row.original for row in original.lines],
                        title=identity.title,
                        endpoint=oogg_endpoint,
                        review=False,
                    )
                    selected = LyricCandidate(
                        source=f"oogg:{oogg_endpoint}",
                        url=original.url,
                        title=identity.title,
                        artist=identity.artist,
                        lines=[
                            LyricPair(row.original, translated[index])
                            for index, row in enumerate(original.lines)
                        ],
                        discovered_urls=list(collected.google_urls),
                        google_search=google,
                        translator="OOGG",
                    )
                    validate_output(original, selected, require_translation=True)
                    translation_method = "oogg_draft"
                    status = "oogg_draft"
                    attach_verification(selected, evidence, translation_method="oogg_draft")
                except Exception as exc:
                    translation_error = _oogg_error(exc)
                    selected = LyricCandidate(
                        source=original.source,
                        url=original.url,
                        title=identity.title,
                        artist=identity.artist,
                        lines=[LyricPair(row.original) for row in original.lines],
                        discovered_urls=list(collected.google_urls),
                        google_search=google,
                        translator=None,
                    )
                    validate_output(original, selected, require_translation=False)
                    attach_verification(selected, evidence, translation_method="none")

        selected.discovered_urls = list(collected.google_urls)
        selected.google_search = google
        output, provenance = write_result(
            audio_path,
            selected,
            output_root=config.output_root,
            overwrite=replace_existing,
            translation_method=translation_method,
            status=status,
            translation_error=translation_error,
            human_approved=False,
            oogg_endpoint=oogg_endpoint,
        )
        return _base_row(
            index,
            audio_path,
            status=status,
            translation_method=translation_method,
            translation_error=translation_error,
            oogg_endpoint=oogg_endpoint,
            line_count=len(selected.lines),
            original_source_url=original_url,
            google_search=google,
            embedded_lyrics_lines=embedded_lines,
            output_txt=str(output),
            output_json=str(provenance),
        )
    except Exception as exc:
        return _base_row(
            index,
            audio_path,
            status="failed",
            original_source_url=original_url,
            google_search=google,
            embedded_lyrics_lines=embedded_lines,
            error=f"{type(exc).__name__}: {exc}",
        )


def _md(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_markdown(rows: list[dict]) -> str:
    counts = Counter(str(row.get("status") or "") for row in rows)
    lines = ["# Status counts", ""]
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "| index | audio | stem | status | translation_method | translation_error | oogg_endpoint | line_count | original_source_url | google_search | embedded_lyrics_lines | error | output_txt | output_json |",
            "|---:|---|---|---|---|---|---|---:|---|---|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_md(row.get(field, "")) for field in RUN_FIELDS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _save_run_reports(output_root: Path, rows: list[dict]) -> tuple[Path, Path]:
    runs_dir = output_root / "report" / "runs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = runs_dir / f"{stamp}_run.json"
    md_path = runs_dir / f"{stamp}_run.md"
    suffix = 1
    while json_path.exists() or md_path.exists():
        json_path = runs_dir / f"{stamp}_{suffix:02d}_run.json"
        md_path = runs_dir / f"{stamp}_{suffix:02d}_run.md"
        suffix += 1
    _atomic_write_text(json_path, json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(md_path, render_markdown(rows))
    return json_path, md_path


def run(args: argparse.Namespace) -> tuple[list[dict], tuple[Path, Path] | None]:
    config = resolve_config(args)
    tracks = _expand_inputs(args.inputs)
    bk_stems = _bk_stems(config.bk_dir)
    rows: list[dict] = []
    seen: dict[str, int] = {}
    eligible = 0
    client: HttpClient | None = None

    for index, raw_path in enumerate(tracks, start=1):
        path = raw_path.expanduser()
        key = str(path.resolve()).casefold()
        if key in seen:
            rows.append(
                _base_row(
                    index,
                    path,
                    status="duplicate",
                    error=f"same path as entry {seen[key]}",
                )
            )
            continue
        seen[key] = index

        if path.stem.casefold() in bk_stems:
            rows.append(_base_row(index, path, status="skipped_bk"))
            continue

        existing_status, payload, replace_existing = _inspect_existing(
            path,
            output_root=config.output_root,
            overwrite=args.overwrite,
            no_translate=args.no_translate,
        )
        if existing_status:
            rows.append(_existing_row(index, path, config.output_root, existing_status, payload))
            continue

        if args.limit is not None and eligible >= args.limit:
            break
        eligible += 1

        if args.dry_run:
            rows.append(_base_row(index, path, status="pending"))
            continue

        if client is None:
            client = HttpClient()
        row = _process_track(
            index,
            path,
            config=config,
            no_translate=args.no_translate,
            replace_existing=replace_existing,
            client=client,
        )
        rows.append(row)
        print(f"{index}/{len(tracks)} {row['status']} {path}")

    if args.dry_run:
        print(render_markdown(rows), end="")
        return rows, None

    reports = _save_run_reports(config.output_root, rows)
    print(f"Run JSON: {reports[0]}")
    print(f"Run Markdown: {reports[1]}")
    return rows, reports


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    rows, _ = run(args)
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
