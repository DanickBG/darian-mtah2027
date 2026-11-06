"""Synthetic sequence/model-input checks; no competition or external match files used."""

import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from starter.sequence_possession import (
    Tracking, active_mask, decode, emission_probabilities, feature_matrix,
    mirrored_features, neighbour_age, rolling_mean, sample_reference,
)


class SequenceTests(unittest.TestCase):
    def tracking(self, n=120, blocks=None):
        ball = np.column_stack((10 + np.arange(n) * 0.1, np.full(n, 20.0)))
        return Tracking(ball, np.tile([0.5, 5.0], (n, 1)), np.tile([1.0, 2.0, 0.4], (n, 1)),
                        blocks or [(1, 0, n - 1)])

    def test_viterbi_matches_exhaustive_path_search(self):
        rng = np.random.default_rng(10)
        probability = rng.uniform(0.1, 1, (5, 3))
        probability /= probability.sum(axis=1, keepdims=True)
        cost, fps = 0.1, 10
        decoded = decode(probability, [(1, 0, 4)], fps, cost)
        def objective(path):
            return sum(-np.log(probability[index, state]) / fps for index, state in enumerate(path)) + cost * sum(
                path[index] != path[index - 1] for index in range(1, len(path)))
        best = min(itertools.product(range(3), repeat=5), key=objective)
        self.assertAlmostEqual(objective(decoded), objective(best))

    def test_offline_context_resolves_an_observation_gap(self):
        probabilities = np.tile([0.98, 0.01, 0.01], (100, 1))
        probabilities[30:70] = [1 / 3] * 3
        labels = decode(probabilities, [(1, 0, 99)], 25, 0.25)
        self.assertTrue(np.all(labels == 0))

    def test_short_opponent_spike_does_not_invent_possession(self):
        probabilities = np.tile([0.98, 0.01, 0.01], (100, 1))
        probabilities[50] = [0.01, 0.98, 0.01]
        self.assertTrue(np.all(decode(probabilities, [(1, 0, 99)], 25, 0.25) == 0))

    def test_sustained_control_change_and_stoppage_are_retained(self):
        probabilities = np.tile([0.98, 0.01, 0.01], (150, 1))
        probabilities[50:100] = [0.01, 0.98, 0.01]
        probabilities[100:] = [0.01, 0.01, 0.98]
        expected = np.repeat([0, 1, 2], 50)
        np.testing.assert_array_equal(decode(probabilities, [(1, 0, 149)], 25, 0.25), expected)

    def test_half_reset_and_ungraded_frames(self):
        probabilities = np.tile([0.98, 0.01, 0.01], (100, 1))
        probabilities[60:] = [0.01, 0.98, 0.01]
        labels = decode(probabilities, [(1, 10, 39), (2, 60, 89)], 25, 1000)
        self.assertTrue(np.all(labels[10:40] == 0))
        self.assertTrue(np.all(labels[60:90] == 1))
        self.assertTrue(np.all(labels[40:60] == 3))

    def test_decoder_is_home_away_symmetric(self):
        probability = np.random.default_rng(5).dirichlet([2, 3, 1], 80)
        original = decode(probability, [(1, 0, 79)], 25, 0.1)
        mirrored = decode(probability[:, [1, 0, 2]], [(1, 0, 79)], 25, 0.1)
        np.testing.assert_array_equal(np.array([1, 0, 2])[original], mirrored)

    def test_costs_are_consistent_across_frame_rates(self):
        probability = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.9, 0.05, 0.05]])
        base = decode(probability, [(1, 0, 2)], 1, 2)
        dense = decode(np.repeat(probability, 10, axis=0), [(1, 0, 29)], 10, 2)
        np.testing.assert_array_equal(np.repeat(base, 10), dense)

    def test_invalid_decoder_inputs_rejected(self):
        for values in (np.zeros((5, 3)), np.full((5, 3), np.nan), np.ones((5, 2))):
            with self.assertRaises(ValueError):
                decode(values, [(1, 0, 4)], 25, 1)
        with self.assertRaises(ValueError):
            decode(np.full((5, 3), 1 / 3), [(1, 0, 3), (2, 3, 4)], 25, 1)

    def test_features_never_cross_half_boundary(self):
        tracking = self.tracking(blocks=[(1, 0, 59), (2, 60, 119)])
        first, names = feature_matrix(tracking)
        tracking.ball[60:] = np.nan
        tracking.distances[60:] = np.nan
        tracking.movement[60:] = 10
        second, _ = feature_matrix(tracking)
        np.testing.assert_array_equal(first[:60], second[:60])
        self.assertGreater(first[59, names.index('away_next_control_age')], np.log1p(29))

    def test_mirrored_features_equal_features_from_swapped_teams(self):
        tracking = self.tracking()
        original, names = feature_matrix(tracking)
        swapped = Tracking(tracking.ball, tracking.distances[:, ::-1], tracking.movement, tracking.blocks)
        actual, actual_names = feature_matrix(swapped)
        self.assertEqual(names, actual_names)
        np.testing.assert_allclose(mirrored_features(original, names), actual, equal_nan=True)
        np.testing.assert_array_equal(mirrored_features(mirrored_features(original, names), names), original)

    def test_dropout_is_reproducible_and_does_not_modify_input(self):
        tracking = self.tracking(n=500)
        original = tracking.ball.copy()
        first, names = feature_matrix(tracking, dropout_seed=7)
        second, _ = feature_matrix(tracking, dropout_seed=7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(tracking.ball, original)
        visible = first[:, names.index('ball_visible')]
        self.assertTrue(np.any(visible == 0))
        self.assertTrue(np.any(visible == 1))


    def test_missing_ball_has_no_fabricated_distance_or_speed(self):
        tracking = self.tracking()
        tracking.ball[30:60] = np.nan
        matrix, names = feature_matrix(tracking)
        self.assertTrue(np.isnan(matrix[40, names.index('home_distance')]))
        self.assertTrue(np.isnan(matrix[40, names.index('ball_speed')]))
        self.assertEqual(matrix[40, names.index('ball_visible')], 0)

    def test_teleport_rejected_without_becoming_extreme_control_feature(self):
        tracking = self.tracking()
        tracking.ball[50] = [500, 500]
        matrix, names = feature_matrix(tracking)
        self.assertEqual(matrix[50, names.index('ball_visible')], 0)
        self.assertTrue(np.isnan(matrix[50, names.index('home_distance')]))

    def test_finite_rolling_mean_and_neighbour_ages(self):
        np.testing.assert_allclose(rolling_mean([1, np.nan, 3], 1), [1, 2, 3])
        previous, following = neighbour_age(np.array([False, True, False, False, True]), 1)
        np.testing.assert_array_equal(previous[1:], [0, 1, 2, 0])
        np.testing.assert_array_equal(following[:4], [1, 0, 2, 1])

    def test_reference_uses_recovery_priority_not_team_name_on_ties(self):
        tracking = self.tracking(n=50)
        # Construct OTHER-match fixture events only; no actual match labels.
        rows = [
            ['Away', 'PASS', '', 1, 1, 10],
            ['Away', 'BALL LOST', '', 1, 11, 15],
            ['Home', 'RECOVERY', '', 1, 11, 11],
            ['Home', 'BALL OUT', '', 1, 20, 25],
            ['Away', 'SET PIECE', 'THROW IN', 1, 30, 30],
            ['Away', 'PASS', '', 1, 30, 40],
        ]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'data/Sample_Game_1/Sample_Game_1_RawEventsData.csv'
            target.parent.mkdir(parents=True)
            pd.DataFrame(rows, columns=['Team', 'Type', 'Subtype', 'Period', 'Start Frame', 'End Frame']).to_csv(target, index=False)
            reference = sample_reference(1, tracking, Path(folder))
        self.assertEqual(reference[9], 1)
        self.assertEqual(reference[10], 0)
        self.assertTrue(np.all(reference[24:29] == 2))
        self.assertEqual(reference[29], 1)
        self.assertTrue(np.all(reference[40:] == 3))

    def test_model_predictions_are_symmetrised_and_normalised(self):
        class Model:
            classes_ = np.array([0, 1, 2])
            def predict_proba(self, matrix):
                return np.tile([0.7, 0.2, 0.1], (len(matrix), 1))
        probability = emission_probabilities(Model(), np.zeros((4, 1)), ['distance_advantage'])
        np.testing.assert_allclose(probability, np.tile([0.45, 0.45, 0.1], (4, 1)))
        np.testing.assert_allclose(probability.sum(axis=1), 1)


if __name__ == '__main__':
    unittest.main()
