from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lyricsee_core import (
    HttpClient,
    LyricCandidate,
    LyricPair,
    SUPPORTED_AUDIO_EXTENSIONS,
    choose_candidate,
    collect_candidates,
    google_search_url,
    probe_audio,
    translate_with_local_llm,
    review_translations_with_local_llm,
    translation_coverage,
    write_result,
)
from source_policy import (
    choose_verified_original, choose_verified_web_translation,
    validate_output, attach_verification,
)
from translation_oogg import translate_deepl, translate_mymemory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find original lyrics plus a web or local-LLM translation."
    )
    parser.add_argument("audio", nargs="+", help="Audio file path(s)")
    parser.add_argument(
        "--mode",
        choices=("web", "mymemory", "deepl", "llm"),
        default="web",
        help="web (default): existing verified translation; "
             "mymemory: limited no-key free translation; "
             "deepl: only pre-existing free API key; "
             "llm: optional local 6969 inference",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:6969",
        help="OpenAI-compatible local endpoint used by llm mode",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    return parser


def _local_model(client: HttpClient, endpoint: str) -> str:
    payload = client.get_json(endpoint.rstrip("/") + "/v1/models")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not data or not isinstance(data, list) or not data[0].get("id"):
        raise RuntimeError("The local endpoint returned no model")
    return str(data[0]["id"])


def process_audio(
    audio_path: Path,
    *,
    mode: str,
    endpoint: str,
    overwrite: bool,
    client: HttpClient,
) -> tuple[Path, Path]:
    audio_path = audio_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.casefold() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio extension {audio_path.suffix}; expected: {supported}")

    identity = probe_audio(audio_path)
    print(f"Track: {identity.artist + ' - ' if identity.artist else ''}{identity.title}")
    if identity.duration is not None:
        print(f"Duration: {identity.duration:.1f} seconds")
    print(f"Google: {google_search_url(identity)}")

    collected = collect_candidates(identity, client)
    for error in collected.errors:
        print(f"Source warning: {error}", file=sys.stderr)
    if not collected.candidates:
        raise LookupError(
            "No lyric source returned usable text. Open the Google URL above and verify metadata."
        )

    original, evidence = choose_verified_original(identity, collected.candidates)
    if mode == "web":
        selected = choose_verified_web_translation(
            identity, original, collected.candidates
        )
        method = "existing_web_translation_source_matched"
    elif mode == "mymemory":
        translated = translate_mymemory([row.original for row in original.lines])
        selected = LyricCandidate(
            source="external:mymemory",
            url=original.url, title=identity.title, artist=identity.artist,
            lines=[LyricPair(row.original, zh) for row, zh in zip(original.lines, translated)],
            discovered_urls=collected.google_urls,
            google_search=google_search_url(identity),
            translator="MyMemory anonymous free",
        )
        method = "mymemory-ja-zh-tw-free"
    elif mode == "deepl":
        translated = translate_deepl(
            [row.original for row in original.lines],
            context="\n".join(row.original for row in original.lines),
        )
        selected = LyricCandidate(
            source="external:deepl",
            url=original.url, title=identity.title, artist=identity.artist,
            lines=[LyricPair(a.original, zh) for a, zh in zip(original.lines, translated)],
            discovered_urls=collected.google_urls,
            google_search=google_search_url(identity),
            translator="DeepL ZH-HANT",
        )
        method = "deepl-api-zh-hant"
    else:
        model = _local_model(client, endpoint)
        translated = translate_with_local_llm(
            [row.original for row in original.lines],
            title=identity.title,
            endpoint=endpoint,
            model=model,
            post_json=client.post_json,
        )
        translated = review_translations_with_local_llm(
            translated, title=identity.title, endpoint=endpoint,
            model=model, post_json=client.post_json,
        )
        selected = LyricCandidate(
            source=f"local-llm:{model}",
            url=original.url,
            title=identity.title,
            artist=identity.artist,
            lines=translated,
            discovered_urls=collected.google_urls,
            google_search=google_search_url(identity),
            translator=model,
        )
        method = "local_llm_verified_original_reviewed"
    validate_output(original, selected)
    attach_verification(selected, evidence, translation_method=method)

    selected.discovered_urls = list(collected.google_urls)
    output, provenance = write_result(audio_path, selected, overwrite=overwrite)
    translated_count = sum(bool(row.translation) for row in selected.lines)
    print(f"Selected source: {selected.source}")
    print(
        f"Translation coverage: {translated_count}/{len(selected.lines)} "
        f"({translation_coverage(selected.lines) * 100:.1f}%)"
    )
    print(f"Lyrics: {output}")
    print(f"Sources: {provenance}")
    return output, provenance


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    client = HttpClient()
    failures = 0
    for raw_path in args.audio:
        try:
            process_audio(
                Path(raw_path),
                mode=args.mode,
                endpoint=args.endpoint,
                overwrite=args.overwrite,
                client=client,
            )
        except Exception as exc:
            failures += 1
            print(f"FAILED: {raw_path}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
