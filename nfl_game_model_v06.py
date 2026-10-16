"""NFL Game Model v0.6 — market-aware *closing-line research experiment*.

WHAT THIS TESTS
---------------
For each of spreads (home winning margin) and game totals, compare:
  1. closing_book      : archived sportsbook closing line, NO statistical model
  2. independent_v01   : original, independently trained v0.1 Ridge forecast
  3. market_base       : closing line + learned residual from v0.1 feature set
  4. market_all        : closing line + learned residual from ALL v0.2-v0.4 groups

The residual target is (actual outcome - historical closing line). A correction
of zero therefore reproduces the bookmaker estimate exactly. This is deliberately
NOT an EV, cover-probability, ROI, or wagering recommendation system.

CAUTIONS
--------
* Both 2024 and 2025 results have been repeatedly inspected in earlier experiments.
  All results are exploratory diagnostics, NOT proof of predictive superiority.
* Historical closing lines need not have been available at any earlier decision
  timestamp, and might reflect different books; these calculations are NOT a
  viable historical betting simulation. NO future-game forecasts are generated.
* "Starting QB" statistics are a last-game primary passer proxy, not verified
  pregame announced starters. No current injuries/weather are incorporated.
* The pairing bootstrap intervals are descriptive, not multiple-test adjusted,
  and do not account for all schedule-dependent correlations.

SETUP
-----
Keep all earlier files intact. Put this script alongside nfl_game_model_v01.py
and nfl_game_model_v05.py, then run:
    python nfl_game_model_v06.py

Needs the SAME input files v0.5 already used:
    output/v04/pregame_features_v04.csv
    output/v04/feature_manifest_v04.json
    output/historical_labels.csv
    output/model_v01/holdout_2025_predictions.csv

Writes new timestamped output/model_v06/run_*/ files, never overwrites earlier
outputs. Uses the v0.5 input validation, fixed 2024/2025 chronological splits,
and original v0.1 Ridge construction without modifying earlier scripts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from nfl_game_model_v01 import build_model
from nfl_game_model_v05 import (
    KEY, TARGETS, GROUPS, load_inputs, completed, variants,
    fit_models, forecast, sha256, verify_raw_reproduces_v01,
)

VERSION = "nfl-game-v0.6-market-aware-closing-line-research"
SEASON_SPLITS = [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]
MARKETS = {"margin": ("home_margin", "spread_line"),
           "total": ("actual_total", "total_line")}
APPROACHES = ("closing_book", "independent_v01", "market_base", "market_all")
DRAWS = 2000
SEED = 20261015


def _check_numerical(data: pd.DataFrame, description: str) -> None:
    """Catch malformed odds or outcomes before fitting any model."""
    for col in ("home_margin", "actual_total", "spread_line", "total_line"):
        if not np.isfinite(pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite {col} in {description}")


def _model_predictions(train: pd.DataFrame, test: pd.DataFrame,
                       predictors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit on completed older seasons only. Never pass closing lines to v0.1."""
    models = fit_models(train, predictors)
    return forecast(models, test, predictors)


def _corrected_predictions(train: pd.DataFrame, test: pd.DataFrame,
                           predictors: list[str]) -> dict[str, np.ndarray]:
    """Fit Ridge on outcome minus closing line; predict correction for test.

    A market column is included as an input, allowing an empirical response to
    favourite/underdog or high/low total, as well as statistical covariates.
    This makes historical closing lines available *at training and evaluation*,
    which is appropriate ONLY for the retrospective research experiment.
    """
    result = {}
    for market, (target, book) in MARKETS.items():
        columns = [book] + predictors
        # The original v0.1 pipeline: training-only median imputer, scaler,
        # fixed Ridge(alpha=10). Identical algorithm across the two variants.
        model = build_model()
        residual = train[target] - train[book]
        model.fit(train[columns], residual)
        correction = model.predict(test[columns])
        predicted = test[book].to_numpy(dtype=float) + correction
        result[f"{market}_correction"] = np.asarray(correction, dtype=float)
        result[f"{market}_prediction"] = np.asarray(predicted, dtype=float)
    return result


def _metrics(detail: pd.DataFrame, year: int, training: tuple[int, ...],
             approach: str) -> list[dict]:
    output = []
    for market, (target, book) in MARKETS.items():
        predictions = detail[f"{approach}__{market}_pred"].to_numpy(dtype=float)
        truth = detail[target].to_numpy(dtype=float)
        error = predictions - truth
        correction = predictions - detail[book].to_numpy(dtype=float)
        output.append({
            "test_season": year,
            "training_seasons": ",".join(map(str, training)),
            "approach": approach,
            "market": market,
            "training_games": -1,  # filled in evaluation function
            "test_games": len(detail),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "signed_error": float(error.mean()),
            "mean_abs_market_correction": float(np.abs(correction).mean()),
            "mean_market_correction": float(correction.mean()),
        })
    return output


def evaluate(data: pd.DataFrame, train_seasons: tuple[int, ...],
             year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = completed(data, train_seasons)
    test = completed(data, (year,))
    if max(train_seasons) >= year:
        raise ValueError("Training season must precede evaluated season")
    _check_numerical(train, "training")
    _check_numerical(test, "evaluation")
    # Frozen feature definitions. Both correction variants share the same game
    # sample and the same algorithm. No tuning or feature selection here.
    options = variants()
    base = options["raw"]
    full_key = "+".join(GROUPS.keys())
    full = options[full_key]

    detail = test[KEY + ["gameday", "home_team", "away_team", "home_margin",
                         "actual_total", "spread_line", "total_line"]].copy()
    detail["closing_book__margin_pred"] = test["spread_line"].to_numpy(dtype=float)
    detail["closing_book__total_pred"] = test["total_line"].to_numpy(dtype=float)

    independent_margin, independent_total = _model_predictions(train, test, base)
    detail["independent_v01__margin_pred"] = independent_margin
    detail["independent_v01__total_pred"] = independent_total

    for approach, predictors in (("market_base", base), ("market_all", full)):
        correction = _corrected_predictions(train, test, predictors)
        for market in MARKETS:
            detail[f"{approach}__{market}_pred"] = correction[f"{market}_prediction"]
            detail[f"{approach}__{market}_correction"] = correction[f"{market}_correction"]

    records: list[dict] = []
    for approach in APPROACHES:
        rec = _metrics(detail, year, train_seasons, approach)
        for row in rec:
            row["training_games"] = len(train)
        records.extend(rec)
        for market, (truth, _) in MARKETS.items():
            detail[f"{approach}__{market}_abs_err"] = (
                detail[f"{approach}__{market}_pred"] - detail[truth]
            ).abs()
    for market, (_, book) in MARKETS.items():
        for approach in ("market_base", "market_all"):
            predicted = detail[f"{approach}__{market}_pred"].to_numpy()
            reconstructed = (detail[book].to_numpy() +
                             detail[f"{approach}__{market}_correction"].to_numpy())
            if not np.allclose(predicted, reconstructed, atol=1e-10):
                raise AssertionError("Correction does not reconstruct prediction")
    # Stable order, no ranking by evaluated error.
    return detail.sort_values(KEY).reset_index(drop=True), pd.DataFrame(records)


def descriptive_intervals(detail: pd.DataFrame, season: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + season)
    n = len(detail)
    sample_idx = rng.integers(0, n, size=(DRAWS, n))
    rows = []
    for market in MARKETS:
        book_errors = detail[f"closing_book__{market}_abs_err"].to_numpy(dtype=float)
        for approach in APPROACHES[1:]:
            candidate_errors = detail[f"{approach}__{market}_abs_err"].to_numpy(dtype=float)
            diffs = candidate_errors - book_errors
            boot_means = diffs[sample_idx].mean(axis=1)
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            rows.append({
                "season": season,
                "market": market,
                "approach": approach,
                "delta_mae_vs_closing_book": float(diffs.mean()),
                "descriptive_bootstrap_ci_low": float(ci_low),
                "descriptive_bootstrap_ci_high": float(ci_high),
                "warning": "Explored seasons; naive game bootstrap; not multiple-testing adjusted",
            })
    return pd.DataFrame(rows)


def print_summary(metrics: pd.DataFrame, max_diff: float, output: Path) -> None:
    print(f"Model: {VERSION}")
    print(f"PASS: independent 2025 v0.1 reproduction (maximum error {max_diff:.2e})")
    print("Book input: HISTORICAL CLOSING spread/total, not timestamped wager odds")
    for season in (2024, 2025):
        print(f"\n{season} EXPLORED DIAGNOSTIC — not a clean holdout")
        print("  Approach               margin MAE  total MAE   delta margin  delta total")
        sub = metrics.loc[metrics.test_season.eq(season)]
        ref_margin = float(sub.loc[(sub.approach == "closing_book") &
                                   (sub.market == "margin"), "mae"].iloc[0])
        ref_total = float(sub.loc[(sub.approach == "closing_book") &
                                  (sub.market == "total"), "mae"].iloc[0])
        for approach in APPROACHES:
            margin_mae = float(sub.loc[(sub.approach == approach) &
                                       (sub.market == "margin"), "mae"].iloc[0])
            total_mae = float(sub.loc[(sub.approach == approach) &
                                      (sub.market == "total"), "mae"].iloc[0])
            print(f"  {approach:23s} {margin_mae:9.3f}  {total_mae:9.3f}   "
                  f"{margin_mae - ref_margin:+12.3f}  {total_mae - ref_total:+11.3f}")
    print(f"\nSaved diagnostic reports: {output}")
    print("NO 2026 future predictions: historical closing lines aren't valid as-of prices.")
    print("NO wager recommendations, calibrated probabilities, expected value, or ROI.")
    print("Do not choose a winner using previously explored 2024/2025 outcomes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=Path("output/v04/pregame_features_v04.csv"))
    parser.add_argument("--labels", type=Path,
                        default=Path("output/historical_labels.csv"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("output/v04/feature_manifest_v04.json"))
    parser.add_argument("--v01-holdout", type=Path,
                        default=Path("output/model_v01/holdout_2025_predictions.csv"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("output/model_v06"))
    args = parser.parse_args()
    for source in (args.features, args.labels, args.manifest, args.v01_holdout):
        if not source.exists():
            parser.error(f"Required input missing: {source}")
    if any(args.outdir.resolve() == src.resolve().parent for src in
           (args.features, args.labels, args.manifest, args.v01_holdout)):
        parser.error("Do not use an input directory as --outdir")

    # Reuse exact one-to-one checks and validated team identity/hash checks from v05.
    data = load_inputs(args.features, args.labels, args.manifest, variants())
    if data["v04_features_ready"].isna().any():
        raise ValueError("Unrecognized v0.4 feature-ready state")

    details = {}
    all_metrics = []
    all_ci = []
    for seasons, year in SEASON_SPLITS:
        detail, metrics = evaluate(data, seasons, year)
        details[year] = detail
        all_metrics.append(metrics)
        all_ci.append(descriptive_intervals(detail, year))

    # Verify old v0.1 predictions on EXACT game keys and BOTH outcomes.
    old = details[2025].rename(columns={
        "independent_v01__margin_pred": "raw__margin_pred",
        "independent_v01__total_pred": "raw__total_pred",
    })
    diff = verify_raw_reproduces_v01(old, args.v01_holdout)
    metrics = pd.concat(all_metrics, ignore_index=True)
    intervals = pd.concat(all_ci, ignore_index=True)

    output = args.outdir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    output.mkdir(parents=True, exist_ok=False)
    for year, detail in details.items():
        detail.to_csv(output / f"diagnostic_predictions_{year}.csv", index=False)
    metrics.to_csv(output / "model_metrics.csv", index=False)
    intervals.to_csv(output / "paired_deltas_vs_market.csv", index=False)

    manifest = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Can v0.1 features / all feature groups predict residual closing-line error?",
        "approaches": list(APPROACHES),
        "markets": MARKETS,
        "training_test_splits": [
            {"train": list(tr), "diagnostic_year": test} for tr, test in SEASON_SPLITS
        ],
        "feature_groups_market_all": list(GROUPS),
        "feature_proxies": {"QB": "last-game primary passer, NOT verified starting QB"},
        "model": "v0.1 build_model() (median impute, scale, Ridge alpha=10)",
        "book_source": "nflverse archived closing line, non-book-specific as-of timestamp",
        "results_are_exploratory": True,
        "no_forward_predictions": True,
        "no_betting_odds_ev_or_roi": True,
        "raw_v01_reproduction_max_error": diff,
        "input_sha256": {
            "v04_features": sha256(args.features),
            "labels": sha256(args.labels),
            "v04_manifest": sha256(args.manifest),
            "v01_2025_holdout": sha256(args.v01_holdout),
            "v01_script": sha256(Path(__file__).with_name("nfl_game_model_v01.py")),
            "v05_script": sha256(Path(__file__).with_name("nfl_game_model_v05.py")),
            "v06_script": sha256(Path(__file__)),
        },
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print_summary(metrics, diff, output)


if __name__ == "__main__":
    main()
