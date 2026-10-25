"""Build matchup features using only regular-season data and pre-tournament seeds."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


RESULT_COLUMNS = ("Season", "WTeamID", "LTeamID", "WScore", "LScore")
SEED_COLUMNS = ("Season", "TeamID", "Seed")
PROFILE_COLUMNS = ("win_rate", "avg_margin", "points_for", "points_against")
FEATURE_COLUMNS = ("seed_advantage",) + tuple(f"{c}_diff" for c in PROFILE_COLUMNS)


def _require_columns(frame: pd.DataFrame, required: tuple[str, ...], source: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def load_kaggle_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read 2026 Machine Learning Mania men's dataset CSV names (no API credentials)."""
    folder = Path(data_dir)
    files = (
        "MRegularSeasonCompactResults.csv",
        "MNCAATourneyCompactResults.csv",
        "MNCAATourneySeeds.csv",
    )
    if any(not (folder / name).is_file() for name in files):
        missing = [name for name in files if not (folder / name).is_file()]
        raise FileNotFoundError(f"Missing Kaggle CSV files in {folder}: {missing}")
    regular, tournament, seeds = (pd.read_csv(folder / name) for name in files)
    _require_columns(regular, RESULT_COLUMNS, files[0])
    _require_columns(tournament, RESULT_COLUMNS, files[1])
    _require_columns(seeds, SEED_COLUMNS, files[2])
    return regular, tournament, seeds


def team_regular_season_features(regular: pd.DataFrame) -> pd.DataFrame:
    """Summarize team performance using games completed before tournament play."""
    _require_columns(regular, RESULT_COLUMNS, "regular")
    winners = regular.assign(
        TeamID=regular["WTeamID"], won=1,
        margin=regular["WScore"] - regular["LScore"],
        scored=regular["WScore"], allowed=regular["LScore"],
    )[["Season", "TeamID", "won", "margin", "scored", "allowed"]]
    losers = regular.assign(
        TeamID=regular["LTeamID"], won=0,
        margin=regular["LScore"] - regular["WScore"],
        scored=regular["LScore"], allowed=regular["WScore"],
    )[["Season", "TeamID", "won", "margin", "scored", "allowed"]]
    both = pd.concat([winners, losers], ignore_index=True)
    out = both.groupby(["Season", "TeamID"], as_index=False).agg(
        games=("won", "size"),
        win_rate=("won", "mean"),
        avg_margin=("margin", "mean"),
        points_for=("scored", "mean"),
        points_against=("allowed", "mean"),
    )
    if not np.isfinite(out[list(PROFILE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Regular-season features contain non-finite values")
    return out


def _parse_seed(seed: str) -> int:
    """Parse W01 / X16a / X16b codes without interpreting region as strength."""
    match = re.fullmatch(r"[WXYZ](\d{2})[ab]?", str(seed))
    if not match:
        raise ValueError(f"Unexpected seed format: {seed!r}")
    return int(match.group(1))


def tournament_team_features(
    regular: pd.DataFrame, seeds: pd.DataFrame
) -> pd.DataFrame:
    _require_columns(seeds, SEED_COLUMNS, "seeds")
    seed_rows = seeds[["Season", "TeamID", "Seed"]].copy()
    seed_rows["seed_number"] = seed_rows["Seed"].map(_parse_seed)
    if seed_rows.duplicated(["Season", "TeamID"]).any():
        raise ValueError("Seed table has duplicate season/team pairs")
    combined = seed_rows.merge(
        team_regular_season_features(regular),
        on=["Season", "TeamID"],
        how="left",
        validate="one_to_one",
    )
    if combined[list(PROFILE_COLUMNS)].isna().any().any():
        raise ValueError("Some tournament-seeded teams have no regular-season data")
    return combined


def build_matchups(
    regular: pd.DataFrame, tournament: pd.DataFrame, seeds: pd.DataFrame
) -> pd.DataFrame:
    """One record per historical game; target is whether lower-ID team A won."""
    _require_columns(tournament, RESULT_COLUMNS, "tournament")
    if tournament.empty:
        raise ValueError("No tournament results supplied")
    games = tournament[["Season", "WTeamID", "LTeamID"]].copy()
    games["TeamAID"] = games[["WTeamID", "LTeamID"]].min(axis=1)
    games["TeamBID"] = games[["WTeamID", "LTeamID"]].max(axis=1)
    if (games["TeamAID"] == games["TeamBID"]).any():
        raise ValueError("Game has identical teams")
    games["team_a_won"] = (games["WTeamID"] == games["TeamAID"]).astype(int)
    features = tournament_team_features(regular, seeds)
    field_map = {"seed_number": "seed"} | {name: name for name in PROFILE_COLUMNS}
    for side in ("A", "B"):
        rename = {"TeamID": f"Team{side}ID"} | {
            key: f"{side}_{value}" for key, value in field_map.items()
        }
        cols = ["Season", "TeamID"] + list(field_map)
        games = games.merge(
            features[cols].rename(columns=rename),
            on=["Season", f"Team{side}ID"],
            how="left",
            validate="many_to_one",
        )
    games["seed_advantage"] = games["B_seed"] - games["A_seed"]
    for name in PROFILE_COLUMNS:
        games[f"{name}_diff"] = games[f"A_{name}"] - games[f"B_{name}"]
    if games[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Missing game features: inspect season/team joins")
    return games.sort_values(["Season", "TeamAID", "TeamBID"]).reset_index(drop=True)


def reversed_features(features: np.ndarray) -> np.ndarray:
    """Swapping opponents negates every difference feature."""
    return -np.asarray(features, dtype=float)
