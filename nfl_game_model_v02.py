"""NFL Game Model v0.2: controlled raw-EPA vs opponent-adjusted EPA comparison.

NEW standalone script. It imports the exact v0.1 predictors and model factory,
without modifying v0.1 code or outputs.

Run from the project directory:
    python nfl_game_model_v02.py

Inputs (already generated on your computer):
    output/v02/pregame_features_v02.csv
    output/v02/feature_manifest_v02.json
    output/historical_labels.csv
    output/model_v01/holdout_2025_predictions.csv  (optional cross-check)

Two variants, identical training games and Ridge settings:
    RAW      = the exact v0.1 predictor list
    ADJUSTED = RAW plus four opponent-adjusted EPA columns

Experiment A: train 2022-2023, validate on 2024.
Experiment B: train 2022-2024, diagnose on 2025 (ALREADY SEEN; not an
              independent holdout for choosing a model).
Research-only forecasts: optionally calculate ungraded predictions for
              remaining 2026 games in --forecast-week (default 6).

No injury data, kickoff timestamps, sportsbook-specific as-of prices,
calibrated probabilities, bet selections, or ROI calculations are provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from nfl_game_model_v01 import PREDICTORS as RAW_PREDICTORS, build_model

KEY = ["season", "week", "game_id"]
TARGETS = ["home_margin", "actual_total"]
ADJ_COLUMNS = [
    "home_off_epa_adj", "home_def_epa_adj", "away_off_epa_adj", "away_def_epa_adj"
]
PREDICTORS_BY_VARIANT = {
    "raw": list(RAW_PREDICTORS),
    "adjusted": list(RAW_PREDICTORS) + ADJ_COLUMNS,
}
MODEL_VERSION = "nfl-game-v0.2-controlled-comparison"
BOOTSTRAP_SEED = 20261015


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual.to_numpy() - predicted.to_numpy()) ** 2)))


def read_inputs(features_path: Path, labels_path: Path, manifest_path: Path) -> pd.DataFrame:
    for path in (features_path, labels_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "nfl-game-features-v0.2":
        raise ValueError("Unexpected feature manifest version")
    checks = (
        ("v02_features_sha256", features_path),
        ("base_labels_sha256", labels_path),
    )
    for key, path in checks:
        expected = manifest.get(key)
        if not expected or expected != sha256(path):
            raise ValueError(
                f"Snapshot mismatch for {path}. Rerun nfl_game_features_v02.py "
                "using your current base CSVs, then rerun this script."
            )

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    needed_features = set(KEY + ["home_team", "away_team", "gameday", "v02_features_ready"])
    needed_features.update(PREDICTORS_BY_VARIANT["adjusted"])
    needed_labels = set(KEY + ["home_score", "away_score", *TARGETS, "spread_line", "total_line"])
    if not needed_features.issubset(features.columns):
        raise ValueError(f"Missing features: {sorted(needed_features - set(features.columns))}")
    if not needed_labels.issubset(labels.columns):
        raise ValueError(f"Missing labels: {sorted(needed_labels - set(labels.columns))}")
    if features.duplicated(KEY).any() or labels.duplicated(KEY).any():
        raise ValueError("Duplicate game keys in features or labels")

    # Guard against pandas interpreting strings such as 'False' as truthy.
    status = features["v02_features_ready"].astype(str).str.lower()
    if not status.isin(["true", "false"]).all():
        raise ValueError("Unrecognized values in v02_features_ready")
    features["v02_features_ready"] = status.eq("true")

    merged = features.merge(
        labels[KEY + ["home_score", "away_score", *TARGETS, "spread_line", "total_line"]],
        on=KEY, how="left", indicator=True, validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Feature games missing in the labels CSV")
    merged = merged.drop(columns="_merge")
    required_feature_cols = PREDICTORS_BY_VARIANT["adjusted"]
    incomplete_ready = merged["v02_features_ready"] & merged[required_feature_cols].isna().any(axis=1)
    if incomplete_ready.any():
        example = merged.loc[incomplete_ready, "game_id"].iloc[0]
        raise ValueError(f"Ready game has missing predictors: {example}")
    return merged


def completed_rows(df: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    rows = df.loc[df["season"].isin(years) & df["v02_features_ready"]].copy()
    if rows.empty:
        raise ValueError(f"No ready rows for years {years}")
    if rows[["home_score", "away_score", *TARGETS]].isna().any().any():
        raise ValueError(f"Training/evaluation years {years} contain unfinished games")
    return rows.sort_values(KEY).reset_index(drop=True)


def fitted_models(training: pd.DataFrame, variant: str) -> dict[str, object]:
    # build_model() is imported unchanged from v0.1: median imputation,
    # standard scaling, and Ridge(alpha=10.0).
    x = training[PREDICTORS_BY_VARIANT[variant]]
    fitted = {}
    for target in TARGETS:
        model = build_model()
        model.fit(x, training[target])
        fitted[target] = model
    return fitted


def predict(models: dict[str, object], rows: pd.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
    x = rows[PREDICTORS_BY_VARIANT[variant]]
    return (models["home_margin"].predict(x), models["actual_total"].predict(x))


def run_experiment(
    df: pd.DataFrame, train_years: tuple[int, ...], test_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = completed_rows(df, train_years)
    test = completed_rows(df, (test_year,))
    # Both models operate on the exact same rows; no sample selection advantage.
    detail = test[KEY + ["home_team", "away_team", "gameday", *TARGETS, "spread_line", "total_line"]].copy()
    detail["experiment_train_years"] = ",".join(map(str, train_years))
    detail["experiment_test_year"] = test_year

    reports = []
    for variant in PREDICTORS_BY_VARIANT:
        models = fitted_models(train, variant)
        margin, total = predict(models, test, variant)
        detail[f"{variant}_pred_home_margin"] = margin
        detail[f"{variant}_pred_total"] = total
        actual_margin = detail["home_margin"]
        actual_total = detail["actual_total"]
        detail[f"{variant}_margin_abs_error"] = np.abs(actual_margin - margin)
        detail[f"{variant}_total_abs_error"] = np.abs(actual_total - total)
        row = {
            "test_season": test_year,
            "train_seasons": ",".join(map(str, train_years)),
            "variant": variant,
            "training_games": len(train),
            "test_games": len(test),
            "margin_mae": mean_absolute_error(actual_margin, margin),
            "margin_rmse": rmse(actual_margin, pd.Series(margin)),
            "total_mae": mean_absolute_error(actual_total, total),
            "total_rmse": rmse(actual_total, pd.Series(total)),
            "margin_signed_error": float(np.mean(margin - actual_margin)),
            "total_signed_error": float(np.mean(total - actual_total)),
        }
        available_spreads = detail["spread_line"].notna()
        available_totals = detail["total_line"].notna()
        row["closing_spread_games"] = int(available_spreads.sum())
        row["closing_total_games"] = int(available_totals.sum())
        row["closing_spread_mae"] = (
            float(np.mean(np.abs(detail.loc[available_spreads, "home_margin"] - detail.loc[available_spreads, "spread_line"])))
            if available_spreads.any() else float("nan")
        )
        row["closing_total_mae"] = (
            float(np.mean(np.abs(detail.loc[available_totals, "actual_total"] - detail.loc[available_totals, "total_line"])))
            if available_totals.any() else float("nan")
        )
        if available_spreads.any():
            book_err = np.abs(detail.loc[available_spreads, "home_margin"] - detail.loc[available_spreads, "spread_line"])
            model_err = detail.loc[available_spreads, f"{variant}_margin_abs_error"]
            row["margin_more_accurate_than_closing_count"] = int((model_err < book_err).sum())
        else:
            row["margin_more_accurate_than_closing_count"] = 0
        if available_totals.any():
            book_err = np.abs(detail.loc[available_totals, "actual_total"] - detail.loc[available_totals, "total_line"])
            model_err = detail.loc[available_totals, f"{variant}_total_abs_error"]
            row["total_more_accurate_than_closing_count"] = int((model_err < book_err).sum())
        else:
            row["total_more_accurate_than_closing_count"] = 0
        reports.append(row)
    return detail, pd.DataFrame(reports)


def paired_bootstrap_ci(raw_errors: np.ndarray, adj_errors: np.ndarray, seed: int) -> tuple[float, float]:
    """95% percentile CI of (adjusted MAE - raw MAE) for a fixed test season.

    Approximate game-level resampling; games are not perfectly independent.
    Used for context, not a formal betting-profitability test.
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(adj_errors) - np.asarray(raw_errors)
    indices = rng.integers(0, len(diffs), size=(2000, len(diffs)))
    boot_means = diffs[indices].mean(axis=1)
    return tuple(np.percentile(boot_means, [2.5, 97.5]).tolist())


def crosscheck_v01(detail_2025: pd.DataFrame, v01_csv: Path) -> str:
    if not v01_csv.exists():
        return f"v0.1 cross-check skipped (file not present: {v01_csv})"
    old = pd.read_csv(v01_csv)
    needed = KEY + ["pred_home_margin", "pred_total"]
    if not set(needed).issubset(old.columns) or old.duplicated(KEY).any():
        raise ValueError("Unexpected v0.1 holdout schema or duplicate games")
    comparison = detail_2025[KEY + ["raw_pred_home_margin", "raw_pred_total"]].merge(
        old[needed], on=KEY, validate="one_to_one", how="outer", indicator=True
    )
    if not comparison["_merge"].eq("both").all():
        raise ValueError("v0.1 holdout covers different games than v0.2")
    max_margin_diff = float(np.max(np.abs(comparison.raw_pred_home_margin - comparison.pred_home_margin)))
    max_total_diff = float(np.max(np.abs(comparison.raw_pred_total - comparison.pred_total)))
    if max(max_margin_diff, max_total_diff) > 1e-6:
        raise ValueError(
            "Raw baseline did not reproduce v0.1 predictions. "
            f"Max margin difference {max_margin_diff:.6g}, total {max_total_diff:.6g}; "
            "investigate input/data/version changes BEFORE comparing v0.2."
        )
    return ("PASS: raw 2025 forecasts reproduce v0.1 predictions "
            f"(max difference: {max(max_margin_diff, max_total_diff):.2e})")


def research_forecasts(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    """Ungraded research output; NOT proof odds were available before kickoff."""
    rows = df.loc[
        (df.season == year) & (df.week == week) & df.v02_features_ready
        & df.home_score.isna() & df.away_score.isna()
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    # Never use a target season as part of its own model's training set.
    training = completed_rows(df, tuple(y for y in (2022, 2023, 2024, 2025) if y < year))
    result = rows[KEY + ["home_team", "away_team", "gameday"]].copy()
    for variant in PREDICTORS_BY_VARIANT:
        models = fitted_models(training, variant)
        margin, total = predict(models, rows, variant)
        result[f"{variant}_pred_home_margin"] = margin
        result[f"{variant}_pred_total"] = total
        result[f"{variant}_pred_home_points"] = (margin + total) / 2.0
        result[f"{variant}_pred_away_points"] = (total - margin) / 2.0
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["research_only"] = True
    result["note"] = "No kickoff timestamps or as-of sportsbook prices; not a verified pregame bet"
    return result.sort_values(["gameday", "game_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("output/v02/pregame_features_v02.csv"))
    parser.add_argument("--labels", type=Path, default=Path("output/historical_labels.csv"))
    parser.add_argument("--feature-manifest", type=Path, default=Path("output/v02/feature_manifest_v02.json"))
    parser.add_argument("--v01-holdout", type=Path, default=Path("output/model_v01/holdout_2025_predictions.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("output/model_v02"))
    parser.add_argument("--forecast-week", type=int, default=6)
    args = parser.parse_args()
    if args.forecast_week < 1 or args.forecast_week > 18:
        parser.error("--forecast-week must be between 1 and 18")
    if args.outdir.resolve() == args.labels.resolve().parent or args.outdir.resolve() == args.features.resolve().parent:
        parser.error("Use a separate output directory; do not overwrite data or v0.1 outputs")

    df = read_inputs(args.features, args.labels, args.feature_manifest)
    splits = [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]
    comparison_files = {}
    summaries = []
    print(f"Model: {MODEL_VERSION}")
    print("Model factory and RAW predictors imported unchanged from v0.1")
    for training_years, test_year in splits:
        detailed, report = run_experiment(df, training_years, test_year)
        comparison_files[test_year] = detailed
        summaries.append(report)

    verification = crosscheck_v01(comparison_files[2025], args.v01_holdout)
    print(verification)
    summary = pd.concat(summaries, ignore_index=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_folder = args.outdir / f"run_{run_id}"
    run_folder.mkdir(parents=True, exist_ok=False)

    comparison_ci = []
    for year, detailed in comparison_files.items():
        detailed.round(6).to_csv(run_folder / f"comparison_{year}.csv", index=False)
        ci_margin = paired_bootstrap_ci(detailed.raw_margin_abs_error.to_numpy(),
                                        detailed.adjusted_margin_abs_error.to_numpy(), BOOTSTRAP_SEED + year)
        ci_total = paired_bootstrap_ci(detailed.raw_total_abs_error.to_numpy(),
                                       detailed.adjusted_total_abs_error.to_numpy(), BOOTSTRAP_SEED + year + 100)
        comparison_ci.append({
            "test_season": year, "test_games": len(detailed),
            "adjusted_minus_raw_margin_mae": float((detailed.adjusted_margin_abs_error - detailed.raw_margin_abs_error).mean()),
            "margin_difference_ci95_low": ci_margin[0], "margin_difference_ci95_high": ci_margin[1],
            "adjusted_minus_raw_total_mae": float((detailed.adjusted_total_abs_error - detailed.raw_total_abs_error).mean()),
            "total_difference_ci95_low": ci_total[0], "total_difference_ci95_high": ci_total[1],
        })
    summary.round(6).to_csv(run_folder / "model_metrics.csv", index=False)
    difference = pd.DataFrame(comparison_ci)
    difference.round(6).to_csv(run_folder / "paired_model_differences.csv", index=False)

    forecast = research_forecasts(df, year=2026, week=args.forecast_week)
    if not forecast.empty:
        forecast.round(6).to_csv(run_folder / f"week{args.forecast_week}_2026_research_forecasts.csv", index=False)

    manifest = {
        "model_version": MODEL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_2024_validation": [2022, 2023],
        "training_2025_diagnostic": [2022, 2023, 2024],
        "training_2026_research": [2022, 2023, 2024, 2025],
        "evaluation_2024": "validation (model design only)",
        "evaluation_2025": "previously examined; diagnostic only, not untouched",
        "model_factory": "nfl_game_model_v01.build_model (median impute, standardize, Ridge alpha=10)",
        "features_raw": list(RAW_PREDICTORS),
        "features_adjusted": list(PREDICTORS_BY_VARIANT["adjusted"]),
        "features_sha256": sha256(args.features),
        "labels_sha256": sha256(args.labels),
        "v01_script_sha256": sha256(Path(__file__).with_name("nfl_game_model_v01.py")),
        "feature_manifest_sha256": sha256(args.feature_manifest),
        "v01_crosscheck": verification,
        "output_folder": str(run_folder),
        "warning": "Closing sportsbook lines are retrospective benchmarks; no placed bets, EV or ROI are inferred.",
    }
    (run_folder / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for year in (2024, 2025):
        print(f"\n{year} {'VALIDATION' if year == 2024 else 'PREVIOUSLY SEEN DIAGNOSTIC'}")
        each = summary.loc[summary.test_season == year].set_index("variant")
        for label in ("raw", "adjusted"):
            row = each.loc[label]
            print(f"  {label:8s}: n={int(row.test_games)} | margin MAE={row.margin_mae:.3f}"
                  f" | total MAE={row.total_mae:.3f}")
        delta = difference.loc[difference.test_season == year].iloc[0]
        print(f"  ADJUSTED - RAW MAE: margin={delta.adjusted_minus_raw_margin_mae:+.3f}"
              f" (95% bootstrap CI {delta.margin_difference_ci95_low:+.3f} to {delta.margin_difference_ci95_high:+.3f})"
              f", total={delta.adjusted_minus_raw_total_mae:+.3f}"
              f" (95% CI {delta.total_difference_ci95_low:+.3f} to {delta.total_difference_ci95_high:+.3f})")
        print(f"  Book closing MAE: margin={each.loc['raw', 'closing_spread_mae']:.3f}"
              f" | total={each.loc['raw', 'closing_total_mae']:.3f}")
    print(f"\nFiles saved to: {run_folder}")
    print(f"2026 Week {args.forecast_week} ungraded research forecasts: {len(forecast)}")
    print("No bet suggestions, calibrated probabilities, or ROI calculations are made.")
    print("Negative (ADJUSTED - RAW) MAE means adjusted features HELPED on that test set.")


if __name__ == "__main__":
    main()
