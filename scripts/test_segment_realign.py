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


if __name__ == "__main__":
    unittest.main()
