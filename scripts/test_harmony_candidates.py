"""WEAK1: harmony candidates for paren lyric lines (generator logic, sidecar reader, Review server wiring)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import harmony_candidates as hc
import review_server
import test_lrc14_review_server as _fixture_module

INDEX_HTML = (SCRIPTS / "review_page" / "index.html").read_text(encoding="utf-8")


class CueTests(unittest.TestCase):
    def test_cue_lines_have_no_explicit_lyric(self):
        for text in ("(間奏)", "（イントロ）", "(×2)", "(Chorus)", "(♪)", "()", "（コーラス：ギター）"):
            self.assertEqual(hc.classify(text), "cue", text)

    def test_vocalisation_and_words_are_lyrics(self):
        for text in ("(ララララ)", "(Oh oh)", "（うん）", "(ほら 見て)"):
            self.assertEqual(hc.classify(text), "lyric", text)

    def test_keyword_with_text_is_unsure(self):
        self.assertEqual(hc.classify("(コーラス：ララ)"), "unsure")

    def test_paren_block_spans_lines_until_it_closes(self):
        texts = ["a", "(b", "c", "d)", "e", "(f)"]
        self.assertEqual(hc.paren_lines(texts), {1, 2, 3, 5})

    def test_unclosed_paren_is_not_a_block(self):
        self.assertEqual(hc.paren_lines(["a", "(b"] + ["c"] * 10), set())


class WindowTests(unittest.TestCase):
    def test_window_is_neighbour_baselines_plus_minus_half_second(self):
        starts = [10.0, 20.0, 21.0, 30.0]
        anchor = [True, False, False, True]
        self.assertEqual(hc.run_window(starts, anchor, 1, 2, 100.0), (9.5, 30.5))

    def test_window_clips_to_the_song(self):
        starts = [0.2, 5.0, 9.9]
        anchor = [True, False, True]
        self.assertEqual(hc.run_window(starts, anchor, 1, 1, 10.0), (0.0, 10.0))

    def test_run_at_the_edges_uses_song_bounds(self):
        starts = [1.0, 2.0, 3.0]
        anchor = [False, True, True]
        self.assertEqual(hc.run_window(starts, anchor, 0, 0, 50.0), (0.0, 2.5))
        anchor = [True, True, False]
        self.assertEqual(hc.run_window(starts, anchor, 2, 2, 50.0), (1.5, 50.0))

    def test_plan_aligns_runs_together_and_skips_cues_and_markers(self):
        texts = ["a", "(ララ", "オオ)", "(間奏)", "b", "(うん)", "c"]
        starts = [10.0, 20.0, 21.0, 22.0, 30.0, 40.0, 50.0]
        romaji = ["a", "rara", "oo", "kansou", "b", "un", "c"]
        runs = hc.plan(texts, starts, 60.0, romaji)
        self.assertEqual([(r["first"], r["last"], r["lines"], r["window"]) for r in runs], [
            (1, 3, [1, 2], [9.5, 30.5]),
            (5, 5, [5], [29.5, 50.5]),
        ])

    def test_plan_drops_lines_without_romaji_and_all_cue_runs(self):
        texts = ["a", "(間奏)", "b", "(うん)", "c"]
        runs = hc.plan(texts, [1.0, 2.0, 3.0, 4.0, 5.0], 9.0, ["a", "", "b", "", "c"])
        self.assertEqual(runs, [])

    def test_marker_entries_are_not_window_anchors(self):
        texts = ["a", "(間奏)", "(うん)", "b"]
        runs = hc.plan(texts, [10.0, 12.0, 14.0, 30.0], 60.0, ["a", "x", "un", "b"], markers=[False, True, False, False])
        self.assertEqual(runs[0]["window"], [9.5, 30.5])


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.audio = self.dir / "song.flac"
        self.audio.write_bytes(b"x")
        (self.dir / "harm.flac").write_bytes(b"h")

    def write(self, **over):
        payload = {"schema": 1, "lrc_sha256": "aa", "lines_total": 5, "harmony_audio": "harm.flac",
                   "lines": [{"entry": 3, "start_cs": 347, "method": "W_h", "window": [1.0, 5.0]}]}
        payload.update(over)
        (self.dir / "song.harmony_cands.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_matching_sidecar_loads(self):
        self.write()
        found = hc.load(self.audio, {"aa"}, 5)
        self.assertEqual(found, {"harmony_audio": str(self.dir / "harm.flac"), "lines": {3: 347}})

    def test_baseline_hash_also_matches(self):
        self.write()
        self.assertIsNotNone(hc.load(self.audio, {"zz", "aa"}, 5))

    def test_hash_mismatch_is_ignored(self):
        self.write()
        self.assertIsNone(hc.load(self.audio, {"bb"}, 5))

    def test_entry_count_mismatch_is_ignored(self):
        self.write()
        self.assertIsNone(hc.load(self.audio, {"aa"}, 4))

    def test_missing_harmony_file_and_bad_data_are_ignored(self):
        self.write(harmony_audio="nope.flac")
        self.assertIsNone(hc.load(self.audio, {"aa"}, 5))
        self.write(schema=2)
        self.assertIsNone(hc.load(self.audio, {"aa"}, 5))
        (self.dir / "song.harmony_cands.json").write_text("{not json", encoding="utf-8")
        self.assertIsNone(hc.load(self.audio, {"aa"}, 5))

    def test_out_of_range_and_non_integer_lines_are_dropped(self):
        self.write(lines=[{"entry": 9, "start_cs": 1}, {"entry": True, "start_cs": 1}, {"entry": 2, "start_cs": "x"}, {"entry": 2, "start_cs": 100}])
        self.assertEqual(hc.load(self.audio, {"aa"}, 5)["lines"], {2: 100})

    def test_no_sidecar_gives_none(self):
        self.assertIsNone(hc.load(self.audio, {"aa"}, 5))

    def test_root_from_environment(self):
        other = self.dir / "roots"
        other.mkdir()
        (other / "harm.flac").write_bytes(b"h")
        self.write()
        (self.dir / "song.harmony_cands.json").replace(other / "song.harmony_cands.json")
        with mock.patch.dict(os.environ, {"LRC_HARMONY_CAND_ROOTS": str(other)}):
            self.assertIsNotNone(hc.load(self.audio, {"aa"}, 5))


class ServerTests(unittest.TestCase):
    _start = _fixture_module.ReviewServerTests._start
    _request = _fixture_module.ReviewServerTests._request
    _json_request = _fixture_module.ReviewServerTests._json_request
    _detail = _fixture_module.ReviewServerTests._detail

    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"LRC_EXPORT_ARCHIVE_DIR": "off"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self._start(False)

    def tearDown(self):
        _fixture_module.ReviewServerTests.tearDown(self)

    def _reload(self):
        # a save with no edits reloads the song (and so reads the sidecar) without touching the LRC
        detail = self._detail()
        status, _h, _d = self._json_request("POST", "/api/song/0/save", {"base_sha256": detail["base_sha256"], "edits": []})
        self.assertEqual(status, 200)
        return self._detail()

    def _sidecar(self, rows_total=None, sha=None):
        (self.root / "harm.flac").write_bytes(b"harmony-audio")
        rows_total = rows_total or len(self._detail()["rows"])
        payload = {"schema": 1, "lrc_sha256": sha or hashlib.sha256(self.output.read_bytes()).hexdigest(), "lines_total": rows_total,
                   "harmony_audio": "harm.flac", "lines": [{"entry": 3, "start_cs": 347, "method": "W_h", "window": [1.0, 5.0]}]}
        (self.root / "song.harmony_cands.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_no_sidecar_page_is_as_before(self):
        detail = self._reload()
        self.assertFalse(detail["has_harmony"])
        self.assertTrue(all("harmony_cand_cs" not in row for row in detail["rows"]))
        self.assertEqual(self._request("GET", f"/audio/0/harm?t={self.server.token}")[0], 404)

    def test_candidate_chip_data_and_harmony_audio(self):
        total = len(self._detail()["rows"])
        self._sidecar(total)
        before = self.output.read_bytes()
        detail = self._reload()
        self.assertTrue(detail["has_harmony"])
        by_entry = {row["entry"]: row.get("harmony_cand_cs") for row in detail["rows"]}
        self.assertEqual(by_entry[3], 347)
        self.assertEqual([e for e, cs in by_entry.items() if cs is not None], [3])
        status, _h, data = self._request("GET", f"/audio/0/harm?t={self.server.token}")
        self.assertEqual((status, data), (200, b"harmony-audio"))
        self.assertEqual(self.output.read_bytes(), before)        # showing / playing a candidate never edits the LRC
        self.assertEqual([row["current_cs"] for row in detail["rows"]][2], 300)

    def test_hash_mismatch_gives_no_candidates(self):
        self._sidecar(sha="0" * 64)
        detail = self._reload()
        self.assertFalse(detail["has_harmony"])
        self.assertTrue(all("harmony_cand_cs" not in row for row in detail["rows"]))

    def test_candidates_survive_an_edit_save_through_the_baseline_hash(self):
        self._sidecar()
        detail = self._reload()
        status, _h, _d = self._json_request("POST", "/api/song/0/save", {
            "base_sha256": detail["base_sha256"], "edits": [{"entry": 1, "cs": 125, "source": "ear", "nudge_cs": 25}]})
        self.assertEqual(status, 200)
        self.assertTrue(self._detail()["has_harmony"])


class PageTests(unittest.TestCase):
    def test_toggle_is_hidden_until_the_song_has_a_harmony_stem(self):
        self.assertIn('id="harm-toggle" type="button" hidden', INDEX_HTML)
        self.assertIn("harmToggle.hidden = !(shownSong && shownSong.has_harmony)", INDEX_HTML)

    def test_chip_only_for_songs_with_harmony_and_never_changes_time(self):
        self.assertIn("(song.has_harmony ? harmChip(row) : '')", INDEX_HTML)
        chip = next(line for line in INDEX_HTML.splitlines() if "'.harm-chip'" in line)
        self.assertIn("playAround(item, row.harmony_cand_cs, 'harm')", chip)
        for forbidden in ("setEdit", "pickOption", "applyCandidate", "nudge("):
            self.assertNotIn(forbidden, chip)


if __name__ == "__main__":
    unittest.main()
