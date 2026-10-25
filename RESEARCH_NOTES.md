# Research notes: what changed from the original exploration

The early notebook was a useful exploration of Spark, team statistics, and
classification. It cannot, however, support the claim that 82–84% of NCAA
tournament matchups were correctly predicted.

## Problems identified in the original notebook

1. **Wrong target.** `RESULT` is set to the winning team's `SEED`; the
   algorithms classify winning-seed values instead of correctly picking a
   winner between two unknown opponents.
2. **Winner-first dataset.** The final joined dataframe has winner-side and
   loser-side columns by construction. This encodes hindsight in the input
   representation and cannot be applied to two unknown opponents as written.
3. **Non-temporal split.** A random 70/30 split across games/seasons is not an
   honest simulation of predicting the next March tournament.
4. **Preprocessing leakage.** Indexers and scalers were fit before the split.
5. **Unstable game-pair construction.** Assigning pairs based on global row
   numbers after sorting by round does not explicitly enforce per-season,
   per-game uniqueness.
6. **Non-portable execution.** Local/Colab/Drive paths, inline installations,
   unpinned environments, and an undefined `training_features` variable make
   the original notebook hard to reproduce.

## Replacement approach

- Get actual winner/loser IDs from tournament results, but canonicalize team
  ordering by TeamID **before** assigning the target.
- Compute pre-tournament team aggregates solely from regular-season games.
- Represent every feature as team A minus team B; use no-intercept logistic
  regression for coherent team-order reversal.
- Hold out each complete tournament year, training only on prior years.
- Compare to a fair seed-only logistic regression trained with the same split.
- Judge probabilities (Brier score/log loss), not accuracy alone.

## Limitations and next experiments

- These simple aggregates do not adjust for opponent strength, injuries, roster
  changes, or schedule quality. Add an Elo-style or opponent-adjusted feature
  set as a *separate*, backtested experiment.
- This version predicts historical men's matchups, not an automatically filled
  future bracket. A later CLI can score arbitrary team pairs using the same
  fitted model and pre-tournament feature snapshot.
- Point estimates and model comparisons are not yet reported until a complete
  historical dataset is downloaded and the evaluation has actually been run.
- A model may be well calibrated without yielding a high bracket win rate.
  Probability evaluation and bracket-selection strategy are distinct tasks.
