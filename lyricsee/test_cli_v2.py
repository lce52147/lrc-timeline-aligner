from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from urllib.error import URLError
from unittest import mock

import lyricsee as cli
import lyricsee_core as core
import source_policy


class _EndpointClient:
    def __init__(self, online: set[str]):
        self.online = {item.rstrip("/") for item in online}
        self.calls: list[tuple[str, float | None]] = []

    def get_json(self, url, params=None, timeout=None):
        self.calls.append((url, timeout))
        endpoint = url.removesuffix("/v1/models").rstrip("/")
        if endpoint not in self.online:
            raise ConnectionError(endpoint)
        return {"data": [{"id": "oogg"}]}


class LyricseeV2CliTests(unittest.TestCase):
    def _canonical(self):
        return core.LyricCandidate(
            source="lrclib",
            url="https://lrclib.net/api/get/1",
            title="Song",
            artist="Artist",
            lines=[core.LyricPair(f"line {i}") for i in range(8)],
        )

    def _process_with_patches(self, root: Path, *, endpoint: str | None, no_translate: bool, translated=None):
        audio = root / "Song.flac"
        audio.write_bytes(b"")
        original = self._canonical()
        collected = core.CollectionResult([original], [], [])
        translated = translated or [f"譯文{i}" for i in range(8)]
        config = cli.Config(root / "bk", root, cli.DEFAULT_OOGG_ENDPOINTS)
        evidence = {
            "policy": source_policy.VERSION,
            "original_verified_by": "test",
            "canonical_original_url": original.url,
            "cross_sources": [],
        }
        target = cli.OoggTarget(endpoint, "selected-model") if endpoint else None
        with (
            mock.patch.object(cli, "probe_embedded_lyrics", return_value=cli.EmbeddedInfo("Artist", "Song", False, 0, 0)),
            mock.patch.object(cli, "probe_audio", return_value=core.TrackIdentity("Song", "Artist")),
            mock.patch.object(cli, "collect_candidates", return_value=collected),
            mock.patch.object(cli, "choose_verified_original", return_value=(original, evidence)),
            mock.patch.object(cli, "choose_verified_web_translation", side_effect=LookupError("no web translation")) as web,
            mock.patch.object(cli, "_select_oogg_endpoint", return_value=target) as select,
            mock.patch.object(cli, "translate_oogg", return_value=translated) as translate,
        ):
            row = cli._process_track(
                1,
                audio,
                config=config,
                no_translate=no_translate,
                replace_existing=False,
                client=object(),
            )
        return row, web, select, translate

    def test_run_markdown_includes_search_engine_and_search_url_columns(self):
        row = cli._base_row(
            1,
            Path("Song.flac"),
            status="original_only",
            search_engine="duckduckgo",
            search_url="https://html.duckduckgo.com/html/?q=Song",
        )
        rendered = cli.render_markdown([row])
        self.assertIn("| search_engine | search_url |", rendered.splitlines()[4])
        self.assertIn("duckduckgo", rendered)
        self.assertNotIn("google_search", rendered.splitlines()[4])

    def test_playlist_paths_resolve_relative_entries_and_preserve_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            playlist = root / "list.m3u8"
            playlist.write_text(
                "#EXTM3U\nMusic\\A.flac\n#EXTINF:1,B\nMusic\\B.flac\n",
                encoding="utf-8",
            )
            self.assertEqual(
                cli.playlist_paths(playlist),
                [root / "Music" / "A.flac", root / "Music" / "B.flac"],
            )

    def test_probe_embedded_lyrics_finds_case_insensitive_lyrics_tag(self):
        def runner(args, **kwargs):
            payload = {
                "format": {
                    "tags": {
                        "ARTIST": "Artist",
                        "TITLE": "Song",
                        "LYRICS": "[00:01.00]line",
                    }
                }
            }
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")

        info = cli.probe_embedded_lyrics(Path("song.flac"), runner=runner)
        self.assertTrue(info.has_lyrics)
        self.assertEqual(info.artist, "Artist")
        self.assertEqual(info.title, "Song")
        self.assertEqual(info.timed_lines, 1)

    def test_oogg_auto_probe_prefers_local_when_both_are_online(self):
        client = _EndpointClient(set(cli.DEFAULT_OOGG_ENDPOINTS))
        selected = cli._select_oogg_endpoint(client, cli.DEFAULT_OOGG_ENDPOINTS)
        self.assertEqual(selected.endpoint, "http://127.0.0.1:6969")
        self.assertEqual(selected.model, "oogg")
        self.assertEqual(client.calls, [("http://127.0.0.1:6969/v1/models", 5)])

    def test_oogg_auto_probe_uses_sister_when_local_is_offline(self):
        client = _EndpointClient({"http://100.87.97.121:6969"})
        selected = cli._select_oogg_endpoint(client, cli.DEFAULT_OOGG_ENDPOINTS)
        self.assertEqual(selected.endpoint, "http://100.87.97.121:6969")
        self.assertEqual(selected.model, "oogg")
        self.assertEqual(
            client.calls,
            [
                ("http://127.0.0.1:6969/v1/models", 5),
                ("http://100.87.97.121:6969/v1/models", 5),
            ],
        )

    def test_oogg_probe_is_cached_for_shared_run_client(self):
        client = _EndpointClient({"http://100.87.97.121:6969"})
        first = cli._select_oogg_endpoint(client, cli.DEFAULT_OOGG_ENDPOINTS)
        second = cli._select_oogg_endpoint(client, cli.DEFAULT_OOGG_ENDPOINTS)
        self.assertEqual(second, first)
        self.assertEqual(
            client.calls,
            [
                ("http://127.0.0.1:6969/v1/models", 5),
                ("http://100.87.97.121:6969/v1/models", 5),
            ],
        )

    def test_explicit_endpoint_disables_default_fallbacks(self):
        with mock.patch.dict(os.environ, {"LYRICSEE_OOGG_ENDPOINT": "http://env:6969"}):
            args = cli.build_parser().parse_args(["--endpoint", "http://forced:6969", "Song.flac"])
            self.assertEqual(cli.resolve_config(args).oogg_endpoints, ("http://forced:6969",))

    def test_environment_endpoint_disables_default_fallbacks(self):
        with mock.patch.dict(
            os.environ,
            {
                "LYRICSEE_OOGG_ENDPOINTS": "",
                "LYRICSEE_OOGG_ENDPOINT": "http://env:6969",
            },
        ):
            args = cli.build_parser().parse_args(["Song.flac"])
            self.assertEqual(cli.resolve_config(args).oogg_endpoints, ("http://env:6969",))

    def test_environment_endpoint_list_overrides_defaults(self):
        with mock.patch.dict(
            os.environ,
            {
                "LYRICSEE_OOGG_ENDPOINTS": "http://one:6969 ; http://two:6969/",
                "LYRICSEE_OOGG_ENDPOINT": "",
            },
        ):
            args = cli.build_parser().parse_args(["Song.flac"])
            self.assertEqual(
                cli.resolve_config(args).oogg_endpoints,
                ("http://one:6969", "http://two:6969"),
            )

    def test_bk_skip_does_not_construct_http_client_or_process_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bk = root / "bk"
            bk.mkdir()
            (bk / "Song.lrc").write_text("[00:01]x\n", encoding="utf-8")
            args = cli.build_parser().parse_args([
                "--bk-dir", str(bk),
                "--output-root", str(root / "out"),
                str(root / "Song.flac"),
            ])
            with (
                mock.patch.object(cli, "HttpClient", side_effect=AssertionError("HTTP client constructed")),
                mock.patch.object(cli, "_process_track", side_effect=AssertionError("track processed")),
            ):
                rows, reports = cli.run(args)
            self.assertEqual([row["status"] for row in rows], ["skipped_bk"])
            self.assertTrue(reports[0].is_file())
            self.assertTrue(reports[1].is_file())

    def test_missing_bk_dir_warns_on_stderr(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-bk"
            with redirect_stderr(StringIO()) as stderr:
                self.assertEqual(cli._bk_stems(missing), set())
            self.assertEqual(stderr.getvalue(), f"BK dir not found: {missing}\n")

    def test_m3u8_duplicate_is_reported_without_second_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            playlist = root / "list.m3u8"
            playlist.write_text("Song.flac\nSong.flac\n", encoding="utf-8")
            args = cli.build_parser().parse_args(["--dry-run", str(playlist)])
            rows, reports = cli.run(args)
            self.assertEqual([row["status"] for row in rows], ["pending", "duplicate"])
            self.assertIsNone(reports)

    def test_web_unavailable_uses_selected_oogg_and_persists_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row, web, select, translate = self._process_with_patches(
                root,
                endpoint="http://100.87.97.121:6969",
                no_translate=False,
            )
            self.assertEqual(row["status"], "oogg_draft")
            self.assertEqual(row["translation_method"], "oogg_draft")
            self.assertEqual(row["oogg_endpoint"], "http://100.87.97.121:6969")
            web.assert_called_once()
            select.assert_called_once()
            translate.assert_called_once()
            self.assertEqual(translate.call_args.kwargs["model"], "selected-model")
            payload = json.loads(Path(row["output_json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["oogg_endpoint"], "http://100.87.97.121:6969")

    def test_oogg_unreachable_switches_to_second_endpoint(self):
        wrapped = RuntimeError("wrapped connection failure")
        wrapped.__cause__ = URLError("local down")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Song.flac"
            audio.write_bytes(b"")
            original = self._canonical()
            collected = core.CollectionResult([original], [], [])
            config = cli.Config(root / "bk", root, cli.DEFAULT_OOGG_ENDPOINTS)
            evidence = {
                "policy": source_policy.VERSION,
                "original_verified_by": "test",
                "canonical_original_url": original.url,
                "cross_sources": [],
            }
            local = cli.OoggTarget(cli.DEFAULT_OOGG_ENDPOINTS[0], "local")
            sister = cli.OoggTarget(cli.DEFAULT_OOGG_ENDPOINTS[1], "sister")
            with (
                mock.patch.object(cli, "probe_embedded_lyrics", return_value=cli.EmbeddedInfo("Artist", "Song", False, 0, 0)),
                mock.patch.object(cli, "probe_audio", return_value=core.TrackIdentity("Song", "Artist")),
                mock.patch.object(cli, "collect_candidates", return_value=collected),
                mock.patch.object(cli, "choose_verified_original", return_value=(original, evidence)),
                mock.patch.object(cli, "choose_verified_web_translation", side_effect=LookupError("no web translation")),
                mock.patch.object(cli, "_select_oogg_endpoint", side_effect=[local, sister]) as select,
                mock.patch.object(
                    cli,
                    "translate_oogg",
                    side_effect=[
                        wrapped,
                        [f"譯文{i}" for i in range(8)],
                    ],
                ) as translate,
            ):
                row = cli._process_track(
                    1,
                    audio,
                    config=config,
                    no_translate=False,
                    replace_existing=False,
                    client=core.HttpClient(),
                )
            self.assertEqual(row["status"], "oogg_draft")
            self.assertEqual(row["oogg_endpoint"], cli.DEFAULT_OOGG_ENDPOINTS[1])
            self.assertEqual(select.call_count, 2)
            self.assertEqual([call.kwargs["endpoint"] for call in translate.call_args_list], list(cli.DEFAULT_OOGG_ENDPOINTS))

    def test_oogg_disables_after_three_unreachable_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self._canonical()
            collected = core.CollectionResult([original], [], [])
            config = cli.Config(root / "bk", root, cli.DEFAULT_OOGG_ENDPOINTS)
            evidence = {
                "policy": source_policy.VERSION,
                "original_verified_by": "test",
                "canonical_original_url": original.url,
                "cross_sources": [],
            }
            target = cli.OoggTarget(cli.DEFAULT_OOGG_ENDPOINTS[0], "local")
            state = {"consecutive_failures": 0, "disabled": False}
            with (
                mock.patch.object(cli, "probe_embedded_lyrics", return_value=cli.EmbeddedInfo("Artist", "Song", False, 0, 0)),
                mock.patch.object(cli, "probe_audio", return_value=core.TrackIdentity("Song", "Artist")),
                mock.patch.object(cli, "collect_candidates", return_value=collected),
                mock.patch.object(cli, "choose_verified_original", return_value=(original, evidence)),
                mock.patch.object(cli, "choose_verified_web_translation", side_effect=LookupError("no web translation")),
                mock.patch.object(cli, "_select_oogg_endpoint", side_effect=[target] * 6) as select,
                mock.patch.object(cli, "translate_oogg", side_effect=URLError("offline")) as translate,
                redirect_stderr(StringIO()) as stderr,
            ):
                rows = []
                for index in range(1, 5):
                    audio = root / f"Song{index}.flac"
                    audio.write_bytes(b"")
                    rows.append(
                        cli._process_track(
                            index,
                            audio,
                            config=config,
                            no_translate=False,
                            replace_existing=False,
                            client=core.HttpClient(),
                            oogg_state=state,
                        )
                    )
            self.assertEqual([row["status"] for row in rows], ["original_only"] * 4)
            self.assertEqual([row["translation_error"] for row in rows], ["oogg_unreachable"] * 4)
            self.assertEqual(translate.call_count, 3)
            self.assertEqual(select.call_count, 6)
            self.assertIn("OOGG offline after 3 consecutive failures", stderr.getvalue())

    def test_oogg_disabled_reprobes_on_tenth_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Song10.flac"
            audio.write_bytes(b"")
            original = self._canonical()
            collected = core.CollectionResult([original], [], [])
            config = cli.Config(root / "bk", root, cli.DEFAULT_OOGG_ENDPOINTS)
            evidence = {
                "policy": source_policy.VERSION,
                "original_verified_by": "test",
                "canonical_original_url": original.url,
                "cross_sources": [],
            }
            target = cli.OoggTarget(cli.DEFAULT_OOGG_ENDPOINTS[1], "sister")
            state = {"consecutive_failures": 3, "disabled": True}
            with (
                mock.patch.object(cli, "probe_embedded_lyrics", return_value=cli.EmbeddedInfo("Artist", "Song", False, 0, 0)),
                mock.patch.object(cli, "probe_audio", return_value=core.TrackIdentity("Song", "Artist")),
                mock.patch.object(cli, "collect_candidates", return_value=collected),
                mock.patch.object(cli, "choose_verified_original", return_value=(original, evidence)),
                mock.patch.object(cli, "choose_verified_web_translation", side_effect=LookupError("no web translation")),
                mock.patch.object(cli, "_select_oogg_endpoint", side_effect=[target, target]) as select,
                mock.patch.object(cli, "translate_oogg", return_value=[f"譯文{i}" for i in range(8)]) as translate,
                redirect_stderr(StringIO()) as stderr,
            ):
                row = cli._process_track(
                    10,
                    audio,
                    config=config,
                    no_translate=False,
                    replace_existing=False,
                    client=core.HttpClient(),
                    oogg_state=state,
                )
            self.assertEqual(row["status"], "oogg_draft")
            self.assertEqual(row["oogg_endpoint"], target.endpoint)
            self.assertEqual(state, {"consecutive_failures": 0, "disabled": False})
            self.assertEqual(select.call_count, 2)
            translate.assert_called_once()
            self.assertIn(f"OOGG back online: {target.endpoint}", stderr.getvalue())

    def test_lrclib_unavailable_is_reported_as_source_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Song.flac"
            audio.write_bytes(b"")
            config = cli.Config(root / "bk", root, cli.DEFAULT_OOGG_ENDPOINTS)
            collected = core.CollectionResult([], [], ["lrclib: source_unavailable (HTTP 503)"], "none", "", True)
            with (
                mock.patch.object(cli, "probe_embedded_lyrics", return_value=cli.EmbeddedInfo("Artist", "Song", False, 0, 0)),
                mock.patch.object(cli, "probe_audio", return_value=core.TrackIdentity("Song", "Artist")),
                mock.patch.object(cli, "collect_candidates", return_value=collected),
                mock.patch.object(cli, "collect_all_candidates_for_review", return_value=[]),
                mock.patch.object(cli, "choose_verified_original", side_effect=LookupError("Cannot verify original")),
            ):
                row = cli._process_track(
                    1,
                    audio,
                    config=config,
                    no_translate=False,
                    replace_existing=False,
                    client=core.HttpClient(),
                )
            self.assertEqual(row["status"], "source_unavailable")
            self.assertIn("source_unavailable (HTTP 503)", row["error"])

    def test_both_oogg_endpoints_offline_falls_back_to_original_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            row, _, select, translate = self._process_with_patches(
                Path(tmp),
                endpoint=None,
                no_translate=False,
            )
            self.assertEqual(row["status"], "original_only")
            self.assertEqual(row["translation_method"], "none")
            self.assertEqual(row["translation_error"], "oogg_unreachable")
            self.assertEqual(row["oogg_endpoint"], "")
            select.assert_called_once()
            translate.assert_not_called()

    def test_no_translate_never_checks_web_or_oogg(self):
        with tempfile.TemporaryDirectory() as tmp:
            row, web, select, translate = self._process_with_patches(
                Path(tmp),
                endpoint="http://127.0.0.1:6969",
                no_translate=True,
            )
            self.assertEqual(row["status"], "original_only")
            self.assertEqual(row["translation_method"], "none")
            web.assert_not_called()
            select.assert_not_called()
            translate.assert_not_called()

    def test_current_original_only_is_existing_only_when_no_translate_is_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Song.flac"
            audio.write_bytes(b"")
            candidate = self._canonical()
            candidate.verification = {
                "policy": source_policy.VERSION,
                "source_verified": True,
            }
            core.write_result(
                audio,
                candidate,
                output_root=root,
                overwrite=False,
                translation_method="none",
                status="original_only",
            )
            status, _, replace = cli._inspect_existing(
                audio,
                output_root=root,
                overwrite=False,
                no_translate=True,
            )
            self.assertEqual(status, "existing")
            self.assertFalse(replace)
            status, _, replace = cli._inspect_existing(
                audio,
                output_root=root,
                overwrite=False,
                no_translate=False,
            )
            self.assertIsNone(status)
            self.assertTrue(replace)

    def test_legacy_pair_is_renamed_and_judged_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Song.flac"
            audio.write_bytes(b"")
            legacy_output = root / "txt" / "Song.txt"
            legacy_provenance = root / "report" / "Song.legacy.sources.json"
            legacy_output.parent.mkdir(parents=True)
            legacy_provenance.parent.mkdir(parents=True)
            lyrics = "line 1\n"
            legacy_output.write_bytes(lyrics.encode("utf-8"))
            legacy_provenance.write_text(
                json.dumps(
                    {
                        "txt_sha256": hashlib.sha256(lyrics.encode("utf-8")).hexdigest(),
                        "verification": {
                            "policy": source_policy.VERSION,
                            "source_verified": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with redirect_stderr(StringIO()) as stderr:
                status, _, replace = cli._inspect_existing(
                    audio,
                    output_root=root,
                    overwrite=False,
                    no_translate=False,
                )
            self.assertEqual(status, "existing")
            self.assertFalse(replace)
            self.assertEqual(stderr.getvalue(), "renamed legacy output: Song\n")
            self.assertTrue((root / "txt" / "Song.lyrics.txt").is_file())
            self.assertTrue((root / "report" / "Song.sources.json").is_file())
            self.assertFalse(legacy_output.exists())
            self.assertFalse(legacy_provenance.exists())

    def test_dry_run_reports_legacy_pair_without_renaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "Song.flac"
            legacy_output = root / "txt" / "Song.txt"
            legacy_provenance = root / "report" / "Song.legacy.sources.json"
            legacy_output.parent.mkdir(parents=True)
            legacy_provenance.parent.mkdir(parents=True)
            legacy_output.write_text("legacy\n", encoding="utf-8")
            legacy_provenance.write_text("{}", encoding="utf-8")
            with redirect_stderr(StringIO()) as stderr:
                status, _, replace = cli._inspect_existing(
                    audio,
                    output_root=root,
                    overwrite=False,
                    no_translate=False,
                    dry_run=True,
                )
            self.assertEqual(status, "legacy_unverified")
            self.assertFalse(replace)
            self.assertEqual(stderr.getvalue(), "would rename legacy output: Song\n")
            self.assertTrue(legacy_output.is_file())
            self.assertTrue(legacy_provenance.is_file())
            self.assertFalse((root / "txt" / "Song.lyrics.txt").exists())
            self.assertFalse((root / "report" / "Song.sources.json").exists())

    def test_dry_run_does_not_construct_http_client_process_or_write_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "out"
            args = cli.build_parser().parse_args([
                "--dry-run",
                "--output-root", str(output_root),
                str(Path(tmp) / "Song.flac"),
            ])
            with (
                mock.patch.object(cli, "HttpClient", side_effect=AssertionError("HTTP client constructed")),
                mock.patch.object(cli, "_process_track", side_effect=AssertionError("track processed")),
                mock.patch.object(cli, "_save_run_reports", side_effect=AssertionError("report written")),
            ):
                rows, reports = cli.run(args)
            self.assertEqual([row["status"] for row in rows], ["pending"])
            self.assertIsNone(reports)
            self.assertFalse((output_root / "report").exists())

    def test_run_persists_incremental_reports_after_each_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bk = root / "bk"
            bk.mkdir()
            audio_paths = [root / "Song1.flac", root / "Song2.flac"]
            for audio in audio_paths:
                audio.write_bytes(b"")
            output_root = root / "out"
            playlist = root / "list.m3u8"
            playlist.write_text("Song1.flac\nSong2.flac\n", encoding="utf-8")
            args = cli.build_parser().parse_args([
                "--bk-dir", str(bk),
                "--output-root", str(output_root),
                str(playlist),
            ])
            snapshots = []
            original_write = cli._write_run_reports

            def tracked_write(reports, rows):
                original_write(reports, rows)
                snapshots.append(json.loads(reports[0].read_text(encoding="utf-8")))

            def fake_process(index, path, *, config, no_translate, replace_existing, client):
                return cli._base_row(index, path, status="original_only")

            with (
                mock.patch.object(cli, "HttpClient", return_value=object()),
                mock.patch.object(cli, "_process_track", side_effect=fake_process),
                mock.patch.object(cli, "_write_run_reports", side_effect=tracked_write),
            ):
                rows, reports = cli.run(args)

            self.assertEqual([len(snapshot) for snapshot in snapshots], [0, 1, 2, 2])
            self.assertEqual(len(json.loads(reports[0].read_text(encoding="utf-8"))), 2)
            self.assertEqual(len(rows), 2)

    def test_output_root_is_passed_to_track_processing_and_run_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "chosen"
            seen = []
            args = cli.build_parser().parse_args([
                "--output-root", str(root),
                str(Path(tmp) / "Song.flac"),
            ])

            def fake_process(index, path, *, config, no_translate, replace_existing, client):
                seen.append(config.output_root)
                return cli._base_row(index, path, status="original_only")

            with (
                mock.patch.object(cli, "HttpClient", return_value=object()),
                mock.patch.object(cli, "_process_track", side_effect=fake_process),
            ):
                _, reports = cli.run(args)
            self.assertEqual(seen, [root])
            self.assertEqual(reports[0].parent, root / "report" / "runs")
            self.assertEqual(reports[1].parent, root / "report" / "runs")

    def test_removed_mode_option_is_rejected(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
            cli.build_parser().parse_args(["--mode", "llm", "Song.flac"])
        self.assertEqual(caught.exception.code, 2)

    def test_launcher_has_no_mode_switch(self):
        text = (Path(__file__).resolve().parent / "lyricsee.ps1").read_text(encoding="utf-8")
        self.assertNotIn("$Mode", text)
        self.assertNotIn("-Mode", text)

    def test_core_has_no_netease_or_output_root_global(self):
        text = (Path(__file__).resolve().parent / "lyricsee_core.py").read_text(encoding="utf-8")
        self.assertNotIn("netease", text.casefold())
        self.assertNotIn("OUTPUT_ROOT", text)


if __name__ == "__main__":
    unittest.main()
