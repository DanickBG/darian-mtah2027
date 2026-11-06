"""Synthetic physical-error and exact segment-decoding checks; no real match files."""

import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from starter.robust_possession import (
    RichTracking, cached_tracking, decode_duration, distances_to_teams, perturb, read_epts,
)
from starter.sequence_possession import decode


class RobustPossessionTests(unittest.TestCase):
    def tracking(self, n=200):
        home = np.tile([[10, 20], [15, 25]], (n, 1, 1)).astype(np.float32)
        away = np.tile([[30, 20], [40, 25]], (n, 1, 1)).astype(np.float32)
        ball = np.tile([11.0, 20.0], (n, 1))
        players = (home, away)
        return RichTracking(ball, distances_to_teams(ball, players), np.tile([1, 2, 0.5], (n, 1)),
                            [(1, 0, n - 1)], 25.0, (105.0, 68.0), players)

    def test_segment_decoder_matches_exhaustive_minimum_duration_search(self):
        p = np.random.default_rng(7).dirichlet([2, 2, 2], size=8)
        fps, cost, minimum = 10, 0.1, 2
        labels = decode_duration(p, [(1, 0, 7)], fps, cost, minimum / fps, step=1)
        def valid(path):
            bounds = [0] + [index for index in range(1, len(path)) if path[index] != path[index - 1]] + [len(path)]
            return all(right - left >= minimum for left, right in zip(bounds[:-1], bounds[1:]))
        def objective(path):
            return sum(-np.log(p[index, state]) / fps for index, state in enumerate(path)) + cost * sum(
                path[index] != path[index - 1] for index in range(1, len(path)))
        possible = (path for path in itertools.product(range(3), repeat=8) if valid(path))
        best = min(possible, key=objective)
        self.assertTrue(valid(labels))
        self.assertAlmostEqual(objective(labels), objective(best))

    def test_no_minimum_and_step_one_matches_viterbi_objective(self):
        p = np.random.default_rng(6).dirichlet([2, 3, 1], size=100)
        old = decode(p, [(1, 0, 99)], 25, 0.25)
        new = decode_duration(p, [(1, 0, 99)], 25, 0.25, 0, 1)
        np.testing.assert_array_equal(old, new)

    def test_partial_final_group_respects_minimum_in_actual_frames(self):
        p = np.tile([0.99, 0.005, 0.005], (14, 1))
        p[5:] = [0.005, 0.99, 0.005]
        labels = decode_duration(p, [(1, 0, 13)], 10, 0, 1.0, 5)
        # A 5-frame first segment followed by a 9-frame final segment is not legal.
        self.assertEqual(len(set(labels)), 1)

    def test_short_touch_is_suppressed_but_sustained_change_is_retained(self):
        p = np.tile([0.999, 0.0005, 0.0005], (40, 1))
        p[10:12] = [0.01, 0.98, 0.01]
        short = decode_duration(p, [(1, 0, 39)], 10, 0, 0.4, 1)
        self.assertTrue(np.all(short == 0))
        p[20:28] = [0.01, 0.98, 0.01]
        sustained = decode_duration(p, [(1, 0, 39)], 10, 0, 0.4, 1)
        self.assertTrue(np.all(sustained[20:28] == 1))

    def test_gap_is_resolved_from_context_and_half_memory_is_separate(self):
        p = np.tile([0.98, 0.01, 0.01], (100, 1))
        p[20:30] = 1 / 3
        p[60:] = [0.01, 0.98, 0.01]
        labels = decode_duration(p, [(1, 5, 39), (2, 60, 89)], 25, 100, 0.4, 5)
        self.assertTrue(np.all(labels[5:40] == 0))
        self.assertTrue(np.all(labels[60:90] == 1))
        self.assertTrue(np.all(labels[40:60] == 3))

    def test_half_shorter_than_minimum_still_gets_one_valid_segment(self):
        p = np.tile([0.01, 0.98, 0.01], (3, 1))
        np.testing.assert_array_equal(decode_duration(p, [(1, 0, 2)], 25, 1, 10, 5), [1, 1, 1])

    def test_decoder_home_away_symmetry(self):
        p = np.random.default_rng(9).dirichlet([2, 2, 1], size=90)
        labels = decode_duration(p, [(1, 0, 89)], 25, 0.1, 0.4, 5)
        mirrored = decode_duration(p[:, [1, 0, 2]], [(1, 0, 89)], 25, 0.1, 0.4, 5)
        np.testing.assert_array_equal(np.array([1, 0, 2])[labels], mirrored)

    def test_invalid_settings_and_probabilities_are_rejected(self):
        p = np.full((10, 3), 1 / 3)
        for values in ({'step': 0}, {'step': True}, {'minimum_s': -1}, {'switch_cost': float('nan')}):
            with self.assertRaises(ValueError):
                decode_duration(p, [(1, 0, 9)], 25, **({'switch_cost': 1} | values))
        with self.assertRaises(ValueError):
            decode_duration(np.zeros((10, 3)), [(1, 0, 9)], 25, 1)

    def test_geometric_distances_and_missing_teams(self):
        tracking = self.tracking()
        np.testing.assert_allclose(tracking.distances[0], [1, 19])
        tracking.players[0][0] = np.nan
        distance = distances_to_teams(tracking.ball, tracking.players)
        self.assertTrue(np.isnan(distance[0, 0]))
        self.assertEqual(distance[0, 1], 19)

    def test_corruption_is_seeded_and_preserves_inputs(self):
        tracking = self.tracking(n=500)
        original = tracking.ball.copy()
        first = perturb(tracking, 5, 'combined')
        second = perturb(tracking, 5, 'combined')
        np.testing.assert_array_equal(first.ball, second.ball)
        np.testing.assert_array_equal(first.distances, second.distances)
        np.testing.assert_array_equal(tracking.ball, original)
        self.assertTrue(np.any(np.isnan(first.ball)))
        self.assertTrue(np.any(np.isfinite(first.ball)))
        self.assertGreater(np.nanmedian(first.distances[:, 0]), 0.1)

    def test_position_noise_recomputes_nearest_distance_instead_of_changing_labels(self):
        tracking = self.tracking(n=500)
        first = perturb(tracking, 5, 'position')
        self.assertGreater(np.nanstd(first.distances[:, 0]), 0.1)
        self.assertFalse(np.array_equal(first.distances, tracking.distances))
        np.testing.assert_array_equal(first.movement, tracking.movement)

    def test_clean_profile_is_a_copy_and_has_no_corruption(self):
        tracking = self.tracking()
        clean = perturb(tracking, 1, 'clean')
        np.testing.assert_array_equal(clean.ball, tracking.ball)
        clean.ball[0] = 0
        self.assertFalse(np.array_equal(clean.ball, tracking.ball))

    def test_tracking_cache_invalidation(self):
        tracking = self.tracking(n=20)
        calls = []
        def build():
            calls.append(True)
            return tracking
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cache.npz'
            cached_tracking({'version': 1}, build, path)
            restored = cached_tracking({'version': 1}, build, path)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(restored.players[0], tracking.players[0])
            cached_tracking({'version': 2}, build, path)
            self.assertEqual(len(calls), 2)

    def test_epts_two_and_three_field_layouts(self):
        metadata = '<main><Team id="h"/><Team id="a"/><Player teamId="h"/><Player teamId="a"/></main>'
        with tempfile.TemporaryDirectory() as folder:
            meta, raw = Path(folder) / 'metadata.xml', Path(folder) / 'tracking.txt'
            meta.write_text(metadata)
            for has_speed, line in ((False, '1:0.1,0.2;0.9,0.8:0.1,0.3\n'),
                                    (True, '1:0.1,0.2,1.0;0.9,0.8,2.0:0.1,0.3,0.0\n')):
                raw.write_text(line)
                tracking = read_epts(meta, raw, 1, [(1, 0, 0)], 25, (100, 60), has_speed)
                np.testing.assert_allclose(tracking.ball[0], [10, 18])
                self.assertAlmostEqual(tracking.distances[0, 0], 6, places=5)

    def test_epts_layout_tracks_substitutions_and_changed_column_order(self):
        metadata = '''<main><Team id="h"/><Team id="a"/>
        <Player id="one" teamId="h"/><Player id="two" teamId="a"/><Player id="sub" teamId="h"/>
        <PlayerChannels>
        <PlayerChannel id="one_x" playerId="one" channelId="x"/><PlayerChannel id="one_y" playerId="one" channelId="y"/>
        <PlayerChannel id="two_x" playerId="two" channelId="x"/><PlayerChannel id="two_y" playerId="two" channelId="y"/>
        <PlayerChannel id="sub_x" playerId="sub" channelId="x"/><PlayerChannel id="sub_y" playerId="sub" channelId="y"/>
        </PlayerChannels>
        <DataFormatSpecification startFrame="1" endFrame="1"><SplitRegister separator=";">
        <SplitRegister><PlayerChannelRef playerChannelId="one_x"/><PlayerChannelRef playerChannelId="one_y"/></SplitRegister>
        <SplitRegister><PlayerChannelRef playerChannelId="two_x"/><PlayerChannelRef playerChannelId="two_y"/></SplitRegister>
        </SplitRegister></DataFormatSpecification>
        <DataFormatSpecification startFrame="2" endFrame="2"><SplitRegister separator=";">
        <SplitRegister><PlayerChannelRef playerChannelId="two_x"/><PlayerChannelRef playerChannelId="two_y"/></SplitRegister>
        <SplitRegister><PlayerChannelRef playerChannelId="sub_x"/><PlayerChannelRef playerChannelId="sub_y"/></SplitRegister>
        </SplitRegister></DataFormatSpecification></main>'''
        with tempfile.TemporaryDirectory() as folder:
            meta, raw = Path(folder) / 'meta.xml', Path(folder) / 'raw.txt'
            meta.write_text(metadata)
            raw.write_text('1:0.1,0.2;0.9,0.8:0.1,0.3\n2:0.9,0.8;0.2,0.2:0.2,0.3\n')
            tracking = read_epts(meta, raw, 2, [(1, 0, 1)], 25, (100, 60), False)
            self.assertTrue(np.all(np.isnan(tracking.players[0][1, 0])))
            np.testing.assert_allclose(tracking.players[0][1, 1], [20, 12])
            self.assertAlmostEqual(tracking.distances[1, 0], 6, places=5)


if __name__ == '__main__':
    unittest.main()
