"""Self-contained tests that do not need an API key or downloaded Kaggle files."""

import unittest

import numpy as np
import pandas as pd

from madness_lab.data import (
    FEATURE_COLUMNS,
    _parse_seed,
    build_matchups,
    reversed_features,
    team_regular_season_features,
)
from madness_lab.model import (
    forward_season_validation,
    make_model,
    summarize_backtest,
)


def demo_data():
    regular_rows, tournament_rows, seed_rows = [], [], []
    teams = (100, 200, 300, 400)
    for season in range(2016, 2025):
        for rank, team in enumerate(teams, start=1):
            seed_rows.append(
                {"Season": season, "TeamID": team, "Seed": f"W{rank:02d}"}
            )
        for a, b in ((100, 200), (100, 300), (400, 200), (300, 400), (100, 400)):
            # Include a mix of favorites and upsets, without using future outcomes
            # to construct model features.
            higher_wins = (season + a + b) % 4 != 0
            winner, loser = (min(a, b), max(a, b)) if higher_wins else (max(a, b), min(a, b))
            regular_rows.append(
                {"Season": season, "WTeamID": winner, "LTeamID": loser,
                 "WScore": 79, "LScore": 71}
            )
        results = [(100, 200), (400, 300), (100, 400)] if season % 2 == 0 else [
            (200, 100), (300, 400), (100, 300)
        ]
        for winner, loser in results:
            tournament_rows.append(
                {"Season": season, "WTeamID": winner, "LTeamID": loser,
                 "WScore": 74, "LScore": 69}
            )
    return (
        pd.DataFrame(regular_rows),
        pd.DataFrame(tournament_rows),
        pd.DataFrame(seed_rows),
    )


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regular, cls.tournament, cls.seeds = demo_data()
        cls.matchups = build_matchups(cls.regular, cls.tournament, cls.seeds)

    def test_pre_game_team_aggregation(self):
        team = team_regular_season_features(self.regular)
        self.assertEqual(len(team), 4 * 9)
        self.assertEqual(len(self.matchups), len(self.tournament))
        self.assertFalse(self.matchups[list(FEATURE_COLUMNS)].isna().any().any())

    def test_seeds(self):
        self.assertEqual(_parse_seed("W01"), 1)
        self.assertEqual(_parse_seed("X16b"), 16)
        with self.assertRaises(ValueError):
            _parse_seed("not-a-seed")

    def test_target_really_is_winner_binary(self):
        self.assertEqual(set(self.matchups.team_a_won.unique()), {0, 1})
        expected = (self.matchups.WTeamID == self.matchups.TeamAID).astype(int)
        np.testing.assert_array_equal(self.matchups.team_a_won, expected)

    def test_swapping_team_order_reverses_probability(self):
        X = self.matchups[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        y = self.matchups.team_a_won.to_numpy(dtype=int)
        model = make_model().fit(X, y)
        p_a = model.predict_proba(X)[:, 1]
        p_b = model.predict_proba(reversed_features(X))[:, 1]
        np.testing.assert_allclose(p_a + p_b, 1, atol=1e-10)

    def test_future_seasons_never_used_to_train(self):
        result = forward_season_validation(
            self.matchups, min_train_seasons=4, last_n_seasons=2
        )
        self.assertEqual(sorted(result.test_season.unique().tolist()), [2023, 2024])
        self.assertEqual(set(result.model), {"seed_only", "seed_plus_regular_season"})
        self.assertEqual(len(result), 4)
        self.assertTrue(result["brier"].between(0, 1).all())
        self.assertEqual(len(summarize_backtest(result)), 2)

    def test_fail_fast_missing_regular_season(self):
        regular = self.regular[self.regular.Season != 2024]
        with self.assertRaisesRegex(ValueError, "no regular-season data"):
            build_matchups(regular, self.tournament, self.seeds)


if __name__ == "__main__":
    unittest.main()
