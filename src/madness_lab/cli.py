"""Command-line, reproducible evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import build_matchups, load_kaggle_data
from .model import forward_season_validation, summarize_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest March Madness win probabilities.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--min-train-seasons", type=int, default=5)
    parser.add_argument("--last-n-seasons", type=int)
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    regular, tournament, seeds = load_kaggle_data(args.data_dir)
    matchups = build_matchups(regular, tournament, seeds)
    results = forward_season_validation(
        matchups, min_train_seasons=args.min_train_seasons,
        last_n_seasons=args.last_n_seasons,
    )
    summary = summarize_backtest(results)
    report = {
        "method": "Train only on seasons strictly earlier than each held-out tournament",
        "summary": summary.to_dict(orient="records"),
        "by_season": results.to_dict(orient="records"),
    }
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
