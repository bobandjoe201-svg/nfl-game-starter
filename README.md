# NFL Game Model — data starter v0.1

This is a **first-stage feature generator**, not a calibrated prediction or profit model. It reuses your `nflreadpy` approach and starts a new NFL game-level workflow for **point spreads and totals**.

## Run on your computer

Requires Python 3.10+ and internet access:

```bash
pip install nflreadpy pandas pyarrow
python nfl_game_data_starter.py --start-season 2022 --end-season 2026 --outdir output
```

Outputs:
- `output/pregame_features.csv`: strictly earlier-week rolling four-game offense/defense features, one row per regular-season NFL game. **No final scores or lines.**
- `output/historical_labels.csv`: final game scores and nflverse's **historical closing** spreads/totals and odds (where available). Store separately and join only when training/evaluating.

Features include offense EPA/play, success rate, pass rate, explosive play rate, plus corresponding defensive figures allowed. It also preserves schedule rest days if present. The first week will have missing team history: that's expected, not zero. Later we should add previous-season priors.

## Research integrity

1. **Do not use closing odds as if available before kickoff or at the selected decision time.** Acquire separate bookmaker-specific, timestamped historical odds for genuine ROI backtests.
2. The public datasets may be revised, so keep raw downloads and a hash/version for each official experiment; the current script does not yet implement raw snapshot preservation.
3. The script excludes the target week entirely from all rolling stats. For live use, freeze a pre-kickoff snapshot and exclude data for any game not yet completed.
4. This starter does not incorporate injuries, confirmed QBs, weather forecasts, or player participation. These need separate as-of data feeds and timestamp joins.
5. Avoid claiming any predictive ability until the model is trained on earlier seasons, tuned on later validation seasons, and measured on an untouched holdout set.
6. The features are preliminary; run diagnostics and check data schema on your environment before using them in any model.

## Next steps

- [ ] Verify the downloaded data and count duplicate game IDs, missing rows and coverage.
- [ ] Add opponent-adjusted EPA, QB quality, situation-neutral pace, run/pass efficiency, and previous-season priors.
- [ ] Design `odds_snapshots` table with sportsbook, market, selection, point, price, `retrieved_at_utc`, `commence_time_utc`.
- [ ] Design `availability_snapshots` for injuries/starting QB with corresponding `known_at_utc`.
- [ ] Build a market-implied baseline and a separate score-margin / game-total model; calibrate cover and total probabilities.
- [ ] Chronological walk-forward backtest, then forward-test every prediction with append-only records.
