"""Segment-level realignment: lyric sections, passages and the section mapper."""
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import segment_realign as sr  # noqa: E402

VERSE1 = ["asa no hikari ga", "mado wo tataku", "kimi no koe ga", "kikoeta ki ga shita"]
CHORUS = ["kono te wo nobashite", "tooku made tonde yuku", "futari no yume"]
VERSE2 = ["yoru ga kite mo", "hoshi wa kienai", "zutto soba ni", "ite hoshii dake"]
BRIDGE = ["kaze ga fuku"]
LYRICS = VERSE1 + CHORUS + VERSE2 + CHORUS + BRIDGE + CHORUS


class LyricStructureTests(unittest.TestCase):
    def test_dice_bigram(self):
        self.assertEqual(sr.dice_bigram("abcd", "abcd"), 1.0)
        self.assertEqual(sr.dice_bigram("abcd", "wxyz"), 0.0)
        self.assertEqual(sr.dice_bigram("", "abc"), 0.0)
        self.assertGreater(sr.dice_bigram("kono te wo nobashite", "kono te wo nobashita"), 0.8)

    def test_repeated_blocks_find_the_chorus_three_times(self):
        sim = sr.line_similarity(LYRICS)
        runs = sr.repeated_blocks(sim)
        starts = {(i, j, length) for i, j, length in runs}
        self.assertIn((4, 11, 3), starts)
        self.assertIn((4, 15, 3), starts)
        self.assertIn((11, 15, 3), starts)
        families = sr.chorus_families(sim)
        self.assertEqual(families, [(3, (4, 11, 15))])

    def test_sections_cover_every_line_in_order(self):
        sections = sr.lyric_sections(LYRICS)
        kinds = [(s.kind, s.start_row, s.end_row, s.occurrence) for s in sections]
        self.assertEqual(kinds, [
            ("verse", 0, 4, None), ("chorus", 4, 7, 0), ("verse", 7, 11, None),
            ("chorus", 11, 14, 1), ("verse", 14, 15, None), ("chorus", 15, 18, 2),
        ])
        self.assertTrue(all(a.end_row == b.start_row for a, b in zip(sections, sections[1:])))

    def test_no_repeats_is_one_verse(self):
        sections = sr.lyric_sections(VERSE1 + VERSE2)
        self.assertEqual([(s.kind, s.start_row, s.end_row) for s in sections], [("verse", 0, 8)])
        self.assertEqual(sr.lyric_sections([]), ())

    def test_nested_repeat_falls_under_the_longer_block(self):
        tail = CHORUS[1:]
        lyrics = VERSE1 + CHORUS + VERSE2 + CHORUS + tail
        families = sr.chorus_families(sr.line_similarity(lyrics))
        self.assertEqual(families[0], (3, (4, 11)))
        # the two-line tail at the end repeats the chorus tail but overlaps nothing accepted
        self.assertTrue(all(len(f[1]) >= 2 for f in families))

    def test_sections_from_a_precomputed_similarity_matrix(self):
        sim = sr.line_similarity(LYRICS)
        self.assertEqual(sr.lyric_sections(sim=sim), sr.lyric_sections(LYRICS))


class PassageTests(unittest.TestCase):
    def test_merge_across_breaths_keep_breaks(self):
        segments = [(1.0, 3.0), (3.8, 6.0), (6.9, 9.5), (12.0, 20.0), (20.6, 21.0), (40.0, 41.5)]
        passages = sr.vocal_passages(segments, duration=60.0)
        self.assertEqual([(p.start, p.end) for p in passages], [(1.0, 9.5), (12.0, 21.0)])
        self.assertEqual(sr.vocal_passages([], duration=60.0), ())


def song_passages():
    # verse 10 s, break, chorus 8 s, break, verse 10 s, break, chorus 8 s, break, bridge 3 s, break, chorus 8 s
    return [(5.0, 15.0), (18.0, 26.0), (30.0, 40.0), (43.0, 51.0), (54.0, 57.0), (59.0, 67.0)]


class MappingTests(unittest.TestCase):
    def test_sections_land_on_their_passages(self):
        plan = sr.plan_segment_realign(LYRICS, song_passages(), duration=70.0)
        self.assertEqual(plan.status, "READY")
        runs = [(p.first_passage, p.last_passage) for p in plan.placements]
        self.assertEqual(runs, [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])
        self.assertEqual(len(plan.windows), len(LYRICS))
        chorus_rows = [w for w in plan.windows if 11 <= w.row < 14]
        self.assertTrue(all(43.0 - 0.4 <= w.start and w.end <= 51.0 + 0.4 for w in chorus_rows))
        self.assertEqual([w.row for w in plan.windows], list(range(len(LYRICS))))
        proposed = [w.proposed for w in plan.windows]
        self.assertEqual(proposed, sorted(proposed))
        self.assertEqual(plan.diagnostics["skipped_passages"], 0)

    def test_instrumental_hum_passage_is_skipped(self):
        # a hummed passage sits between chorus 1 and verse 2, two seconds clear of both
        passages = [(5.0, 15.0), (18.0, 26.0), (28.0, 30.5), (32.5, 42.5), (45.5, 53.5), (56.5, 59.5), (61.5, 69.5)]
        plan = sr.plan_segment_realign(LYRICS, passages, duration=75.0)
        self.assertEqual(plan.status, "READY")
        self.assertEqual(len(plan.passages), 7, "the hum is a passage of its own")
        # The hum is either skipped or absorbed into a neighbour's window; it never takes a section of its own.
        self.assertTrue(all(p.first_passage != 2 or p.last_passage != 2 for p in plan.placements))
        verse2 = next(p for p in plan.placements if p.section.start_row == 7)
        self.assertEqual((verse2.start, verse2.end), (32.5, 42.5))
        chorus2 = next(p for p in plan.placements if p.section.start_row == 11)
        self.assertEqual((chorus2.start, chorus2.end), (45.5, 53.5))

    def test_chorus_similarity_picks_the_matching_passage(self):
        # Two equally plausible passages for chorus 2: timbre says the second one.
        lyrics = VERSE1 + CHORUS + VERSE2 + CHORUS
        passages = [(5.0, 15.0), (18.0, 26.0), (30.0, 40.0), (43.0, 51.0), (54.0, 62.0)]
        no_sim = sr.plan_segment_realign(lyrics, passages, duration=70.0)
        sim = [[0.0] * 5 for _ in range(5)]
        sim[1][4] = sim[4][1] = 0.9   # chorus 1 sounds like passage 4
        sim[1][3] = sim[3][1] = 0.2
        with_sim = sr.plan_segment_realign(lyrics, passages, duration=70.0, pair_similarity=sim)
        chorus2 = next(p for p in with_sim.placements if p.section.start_row == 11)
        self.assertEqual((chorus2.first_passage, chorus2.last_passage), (4, 4))
        self.assertEqual(with_sim.status, no_sim.status, "READY")

    def test_section_may_span_two_passages(self):
        # verse 1 is split by a breath gap longer than the passage gap
        passages = [(5.0, 9.0), (10.8, 15.0), (18.0, 26.0), (30.0, 40.0), (43.0, 51.0), (54.0, 57.0), (59.0, 67.0)]
        plan = sr.plan_segment_realign(LYRICS, passages, duration=70.0)
        self.assertEqual(plan.status, "READY")
        verse1 = plan.placements[0]
        self.assertEqual((verse1.first_passage, verse1.last_passage), (0, 1))
        rows = [w for w in plan.windows if w.row < 4]
        self.assertTrue(all(w.start <= w.proposed <= w.end for w in rows))

    def test_prior_is_weak(self):
        # wildly wrong priors (everything at 60 s) do not drag the first verse off its passage
        plan = sr.plan_segment_realign(LYRICS, song_passages(), duration=70.0, priors=[60.0] * len(LYRICS))
        self.assertEqual(plan.placements[0].first_passage, 0)

    def test_blocked_without_passages(self):
        plan = sr.plan_segment_realign(LYRICS, [], duration=70.0)
        self.assertEqual(plan.status, "BLOCKED")
        self.assertEqual(plan.windows, ())


# --- row-level structural map (V2 P2) -------------------------------------------------

# verse 1 (4 rows), chorus (3 rows, gaps 3 s and 8 s), verse 2, chorus, bridge, chorus
TIMES = [10.0, 14.0, 18.0, 22.0, 30.0, 33.0, 41.0, 55.0, 59.0, 63.0, 67.0, 80.0, 83.0, 91.0, 100.0, 110.0, 113.0, 121.0]
SEGMENTS = [(10.0, 26.0), (30.0, 47.0), (55.0, 71.0), (80.0, 97.0), (100.0, 108.0), (110.0, 127.0)]
DURATION = 130.0


def observations(times, review=(), chars=20):
    return [sr.RowObservation(t, chars, "REVIEW" if i in review else "TRUSTED") for i, t in enumerate(times)]


class StructuralRowMapTests(unittest.TestCase):
    def test_consistent_song_is_left_alone(self):
        result = sr.structural_row_map(observations(TIMES, review={5, 9, 16}), SEGMENTS, duration=DURATION,
                                       sim=sr.line_similarity(LYRICS))
        self.assertEqual(result.status, "READY")
        self.assertEqual([r.row for r in result.rows if r.flagged], [])
        self.assertEqual(result.runs, ())
        self.assertTrue(all(abs(r.proposed - t) <= 0.3 for r, t in zip(result.rows, TIMES)))
        self.assertGreater(result.diagnostics["sibling_pairs"], 0)

    def test_chorus_line_sung_in_the_next_occurrence_is_pulled_back(self):
        times = list(TIMES)
        times[13] = 125.0            # last line of chorus 2 placed 34 s late, after the bridge
        result = sr.structural_row_map(observations(times, review={13}), SEGMENTS, duration=DURATION,
                                       sim=sr.line_similarity(LYRICS))
        self.assertEqual(result.status, "READY")
        self.assertEqual([r.row for r in result.rows if r.flagged], [13])
        self.assertEqual(result.rows[12].gap_model, "sibling")   # the gap into row 13 is judged by the sibling chorus
        self.assertAlmostEqual(result.rows[13].proposed, 91.0, delta=0.5)
        (run,) = result.runs
        self.assertEqual((run.first_row, run.last_row), (13, 13))
        self.assertLessEqual(run.start, 91.0)
        self.assertGreaterEqual(run.end, 91.0)
        self.assertAlmostEqual(run.start, 83.0, delta=0.11)
        self.assertAlmostEqual(run.end, 100.0, delta=0.11)

    def test_backwards_row_is_moved_forward(self):
        times = list(TIMES)
        times[5] = 29.0              # middle chorus line placed before the chorus starts
        result = sr.structural_row_map(observations(times, review={5}), SEGMENTS, duration=DURATION,
                                       sim=sr.line_similarity(LYRICS))
        self.assertEqual([r.row for r in result.rows if r.flagged], [5])
        self.assertGreater(result.rows[5].proposed, 30.0)
        self.assertAlmostEqual(result.rows[5].proposed, 33.0, delta=0.5)

    def test_anchor_rows_do_not_move_far(self):
        times = list(TIMES)
        times[8] = 61.0              # a trusted row off by 2 s stays where it is reported
        result = sr.structural_row_map(observations(times), SEGMENTS, duration=DURATION, sim=sr.line_similarity(LYRICS))
        self.assertEqual([r.row for r in result.rows if r.flagged], [])

    def test_too_few_anchors_blocks(self):
        result = sr.structural_row_map(observations(TIMES, review=set(range(len(TIMES)))), SEGMENTS, duration=DURATION)
        self.assertEqual(result.status, "BLOCKED")
        self.assertEqual(result.reason, "too-few-anchors")

    def test_long_run_window_is_narrowed(self):
        times = list(TIMES)
        times[13] = 125.0
        obs = observations(times, review={13})
        # no anchor between the chorus and the end: the run would span to the song's end
        obs = obs[:14] + [sr.RowObservation(t, 20, "REVIEW") for t in times[14:]]
        result = sr.structural_row_map(obs, SEGMENTS, duration=DURATION, sim=sr.line_similarity(LYRICS))
        self.assertEqual(result.status, "READY")
        for run in result.runs:
            self.assertLessEqual(run.end - run.start, sr.RUN_WINDOW_MAX_S + 0.2)


if __name__ == "__main__":
    unittest.main()
