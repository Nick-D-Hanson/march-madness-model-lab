"""Chronological evaluation without tournament outcome or future-season leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLUMNS


def make_model():
    # No intercept and no mean centering: reversing teams gives 1 - original p.
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(fit_intercept=False, max_iter=2000, C=1.0),
    )


def score_probabilities(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, probability >= 0.5)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
    }


def forward_season_validation(
    matchups: pd.DataFrame,
    min_train_seasons: int = 5,
    last_n_seasons: int | None = None,
) -> pd.DataFrame:
    """For every test season, train only on games from earlier seasons."""
    if min_train_seasons < 2:
        raise ValueError("Use at least two training seasons")
    for name in ("Season", "team_a_won", *FEATURE_COLUMNS):
        if name not in matchups.columns:
            raise ValueError(f"Matchups missing column {name}")
    seasons = sorted(int(x) for x in matchups["Season"].unique())
    eligible = seasons[min_train_seasons:]
    if last_n_seasons is not None:
        if last_n_seasons < 1:
            raise ValueError("last_n_seasons must be positive")
        eligible = eligible[-last_n_seasons:]
    if not eligible:
        raise ValueError("Not enough historical seasons to evaluate")

    rows: list[dict] = []
    for season in eligible:
        train = matchups.loc[matchups.Season < season]
        test = matchups.loc[matchups.Season == season]
        y_train = train["team_a_won"].to_numpy(dtype=int)
        y_test = test["team_a_won"].to_numpy(dtype=int)
        if len(np.unique(y_train)) != 2:
            raise ValueError(f"Training seasons before {season} need both win classes")
        variants = {
            "seed_only": ("seed_advantage",),
            "seed_plus_regular_season": FEATURE_COLUMNS,
        }
        for variant, columns in variants.items():
            model = make_model()
            model.fit(train[list(columns)], y_train)
            probability = model.predict_proba(test[list(columns)])[:, 1]
            rows.append(
                {"test_season": season, "model": variant, "games": len(test)}
                | score_probabilities(y_test, probability)
            )
    return pd.DataFrame(rows).sort_values(["test_season", "model"]).reset_index(drop=True)


def summarize_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Per-season mean prevents a season with more games dominating the report."""
    return results.groupby("model", as_index=False).agg(
        seasons=("test_season", "nunique"),
        mean_accuracy=("accuracy", "mean"),
        mean_brier=("brier", "mean"),
        mean_log_loss=("log_loss", "mean"),
    )
