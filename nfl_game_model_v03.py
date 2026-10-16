"""NFL Game Model v0.3: controlled pace / play-volume experiment.

NEW, independent model runner. Does NOT modify v0.1/v0.2 code or results.
Run from project root after generating v0.3 features:

    python nfl_game_model_v03.py

Comparison:
  raw  = EXACT v0.1 predictors + Ridge training pipeline (same games).
  pace = raw predictors + four pregame pace/volume features.

Tests: train 2022-2023, check 2024; train 2022-2024, diagnose 2025.
Both are already viewed/explored; neither is a pristine new holdout.
Research forecasts for 2026 Week 6 only if currently unplayed in input labels.

Missing pace is imputed using TRAINING-set medians within the same Ridge
pipeline. No closing sportsbook lines are model inputs; they're benchmarks only.
No injury/QB updates, calibrated cover probabilities, EV, bets, or ROI.
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
from nfl_game_features_v03 import FEATURES as PACE_PREDICTORS

VERSION = "nfl-game-v0.3-pace-controlled"
KEY = ["season", "week", "game_id"]
TARGETS = ["home_margin", "actual_total"]
VARIANTS = {"raw": list(RAW_PREDICTORS),
            "pace": list(RAW_PREDICTORS) + list(PACE_PREDICTORS)}
BOOT_SEED = 20261015


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(features_file: Path, labels_file: Path, manifest_file: Path) -> pd.DataFrame:
    for p in (features_file, labels_file, manifest_file):
        if not p.is_file():
            raise FileNotFoundError(f"Missing model input: {p}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("version") != "nfl-game-features-v0.3-pace-volume":
        raise ValueError("Unexpected v0.3 feature manifest version")
    for field, file in (("v03_features_sha256", features_file),
                        ("base_labels_sha256", labels_file)):
        if sha256(file) != manifest.get(field):
            raise ValueError(f"Data snapshot mismatch: {file}")
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)
    needed_f = set(KEY + ["home_team", "away_team", "gameday", "v03_features_ready"] + VARIANTS["pace"])
    needed_l = set(KEY + ["home_score", "away_score", *TARGETS, "spread_line", "total_line"])
    if not needed_f.issubset(features.columns):
        raise ValueError(f"Missing features: {sorted(needed_f - set(features.columns))}")
    if not needed_l.issubset(labels.columns):
        raise ValueError(f"Missing result columns: {sorted(needed_l - set(labels.columns))}")
    if features.duplicated(KEY).any() or labels.duplicated(KEY).any():
        raise ValueError("Duplicate feature or label game keys")
    status = features["v03_features_ready"].astype(str).str.lower()
    if not status.isin(["true", "false"]).all():
        raise ValueError("Unrecognized v03_features_ready")
    features["v03_features_ready"] = status.eq("true")
    df = features.merge(
        labels[KEY + ["home_score", "away_score", *TARGETS, "spread_line", "total_line"]],
        on=KEY, how="left", validate="one_to_one", indicator=True)
    if not df["_merge"].eq("both").all():
        raise ValueError("Missing labels for some games")
    df = df.drop(columns="_merge")
    if df.loc[df["v03_features_ready"], RAW_PREDICTORS].isna().any().any():
        raise ValueError("Ready rows missing raw features")
    return df


def finished(df: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    result = df.loc[df.season.isin(years) & df.v03_features_ready].sort_values(KEY).reset_index(drop=True)
    if result.empty or result[["home_score", "away_score"] + TARGETS].isna().any().any():
        raise ValueError(f"Incomplete training/test sample for {years}")
    return result


def fit(training: pd.DataFrame, variant: str) -> dict[str, object]:
    predictors = VARIANTS[variant]
    models = {}
    for y in TARGETS:
        m = build_model()  # exact factory used in v0.1, with training-only median imputation
        m.fit(training[predictors], training[y])
        models[y] = m
    return models


def predict(models: dict[str, object], games: pd.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
    X = games[VARIANTS[variant]]
    return models["home_margin"].predict(X), models["actual_total"].predict(X)


def compare_season(df: pd.DataFrame, train_years: tuple[int, ...], test_year: int):
    train, test = finished(df, train_years), finished(df, (test_year,))
    detail = test[KEY + ["home_team", "away_team", "gameday", *TARGETS, "spread_line", "total_line"]].copy()
    records = []
    for variant in VARIANTS:
        models = fit(train, variant)
        margin, total = predict(models, test, variant)
        detail[f"{variant}_pred_home_margin"] = margin
        detail[f"{variant}_pred_total"] = total
        detail[f"{variant}_margin_error"] = abs(test.home_margin.to_numpy() - margin)
        detail[f"{variant}_total_error"] = abs(test.actual_total.to_numpy() - total)
        records.append({"season": test_year, "train_years": ",".join(map(str, train_years)),
                        "variant": variant, "training_games": len(train), "test_games": len(test),
                        "margin_mae": mean_absolute_error(test.home_margin, margin),
                        "total_mae": mean_absolute_error(test.actual_total, total),
                        "margin_signed_error": float(np.mean(margin - test.home_margin.to_numpy())),
                        "total_signed_error": float(np.mean(total - test.actual_total.to_numpy()))})
    closing = {
        "closing_spread_mae": float((detail.home_margin - detail.spread_line).abs().mean()),
        "closing_total_mae": float((detail.actual_total - detail.total_line).abs().mean()),
    }
    for r in records:
        r.update(closing)
    return detail, pd.DataFrame(records)


def bootstrap_delta(old_abs: np.ndarray, new_abs: np.ndarray, seed: int) -> tuple[float, float, float]:
    """Paired game-level percentile CI; descriptive, not a profit guarantee."""
    differences = np.asarray(new_abs, dtype=float) - np.asarray(old_abs, dtype=float)
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(differences), size=(2000, len(differences)))
    lower, upper = np.percentile(differences[ids].mean(axis=1), [2.5, 97.5])
    return float(differences.mean()), float(lower), float(upper)


def crosscheck_raw(detail_2025: pd.DataFrame, baseline_path: Path) -> str:
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Expected v0.1 cross-check not found: {baseline_path}")
    old = pd.read_csv(baseline_path)
    if old.duplicated(KEY).any():
        raise ValueError("Duplicate v0.1 baseline games")
    needed = set(KEY + ["pred_home_margin", "pred_total"])
    if not needed.issubset(old.columns):
        raise ValueError("v0.1 baseline CSV missing expected columns")
    both = detail_2025[KEY + ["raw_pred_home_margin", "raw_pred_total"]].merge(
        old[KEY + ["pred_home_margin", "pred_total"]],
        on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not both["_merge"].eq("both").all():
        raise ValueError("Changed test-game universe; raw v0.3 cannot be compared fairly")
    maximum = float(max((both.raw_pred_home_margin - both.pred_home_margin).abs().max(),
                        (both.raw_pred_total - both.pred_total).abs().max()))
    if maximum > 1e-6:
        raise ValueError(f"v0.3 raw predictions do NOT reproduce v0.1 (max difference={maximum})")
    return f"PASS: raw 2025 predictions reproduce v0.1 (maximum absolute difference {maximum:.2e})"


def ungraded_forecasts(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    sample = df.loc[(df.season == year) & (df.week == week) & df.v03_features_ready
                    & df.home_score.isna() & df.away_score.isna()].copy()
    if sample.empty:
        return pd.DataFrame()
    historical = finished(df, tuple(x for x in (2022, 2023, 2024, 2025) if x < year))
    results = sample[KEY + ["home_team", "away_team", "gameday"]].copy()
    for variant in VARIANTS:
        models = fit(historical, variant)
        margin, total = predict(models, sample, variant)
        results[f"{variant}_pred_home_margin"] = margin
        results[f"{variant}_pred_total"] = total
    results["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    results["research_only"] = True
    results["note"] = "No verified kickoff timestamps or as-of odds; not an executable wager"
    return results.sort_values(["gameday", "game_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("output/v03/pregame_features_v03.csv"))
    parser.add_argument("--labels", type=Path, default=Path("output/historical_labels.csv"))
    parser.add_argument("--feature-manifest", type=Path, default=Path("output/v03/feature_manifest_v03.json"))
    parser.add_argument("--v01-holdout", type=Path, default=Path("output/model_v01/holdout_2025_predictions.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("output/model_v03"))
    parser.add_argument("--forecast-week", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.forecast_week <= 18:
        parser.error("--forecast-week must be between 1 and 18")
    if any(args.outdir.resolve() == source.resolve().parent for source in (args.features, args.labels, args.feature_manifest)):
        parser.error("Choose a separate model output directory")
    data = load_data(args.features, args.labels, args.feature_manifest)
    all_reports, comparisons, ci_rows = [], {}, []
    for train, season in [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]:
        details, report = compare_season(data, train, season)
        comparisons[season] = details
        all_reports.append(report)
        margin = bootstrap_delta(details.raw_margin_error, details.pace_margin_error, BOOT_SEED + season)
        total = bootstrap_delta(details.raw_total_error, details.pace_total_error, BOOT_SEED + season + 100)
        ci_rows.append({"test_year": season, "pace_minus_raw_margin_mae": margin[0],
                        "margin_ci95_low": margin[1], "margin_ci95_high": margin[2],
                        "pace_minus_raw_total_mae": total[0],
                        "total_ci95_low": total[1], "total_ci95_high": total[2]})
    verification = crosscheck_raw(comparisons[2025], args.v01_holdout)
    summary = pd.concat(all_reports, ignore_index=True)
    diffs = pd.DataFrame(ci_rows)
    forecasts = ungraded_forecasts(data, 2026, args.forecast_week)
    folder = args.outdir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    folder.mkdir(parents=True, exist_ok=False)
    for year, detail in comparisons.items():
        detail.round(6).to_csv(folder / f"comparison_{year}.csv", index=False)
    summary.round(6).to_csv(folder / "model_metrics.csv", index=False)
    diffs.round(6).to_csv(folder / "paired_differences.csv", index=False)
    if not forecasts.empty:
        forecasts.round(6).to_csv(folder / f"week{args.forecast_week}_2026_research_forecasts.csv", index=False)
    meta = {
        "model_version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v01_crosscheck": verification,
        "raw_predictors": VARIANTS["raw"], "pace_predictors": VARIANTS["pace"],
        "model_factory": "nfl_game_model_v01.build_model: train-only median imputation, scaled Ridge alpha=10",
        "2024_role": "explored validation; not pristine holdout",
        "2025_role": "previously examined diagnostic; not pristine holdout",
        "2026_role": "forward research only if captured before kickoff",
        "features_sha256": sha256(args.features), "labels_sha256": sha256(args.labels),
        "feature_manifest_sha256": sha256(args.feature_manifest),
        "baseline_script_sha256": sha256(Path(__file__).with_name("nfl_game_model_v01.py")),
        "no_ROI_claims": True,
    }
    (folder / "model_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Model: {VERSION}")
    print(verification)
    for season in (2024, 2025):
        r = summary.loc[summary.season == season].set_index("variant")
        d = diffs.loc[diffs.test_year == season].iloc[0]
        print(f"\n{season} {'EXPLORED VALIDATION' if season == 2024 else 'PREVIOUSLY SEEN DIAGNOSTIC'}")
        for label in ("raw", "pace"):
            row = r.loc[label]
            print(f"  {label:5s}: n={int(row.test_games)} | margin MAE={row.margin_mae:.3f} | total MAE={row.total_mae:.3f}")
        print(f"  PACE - RAW MAE: margin={d.pace_minus_raw_margin_mae:+.3f} "
              f"(95% CI {d.margin_ci95_low:+.3f} to {d.margin_ci95_high:+.3f}), "
              f"total={d.pace_minus_raw_total_mae:+.3f} "
              f"(95% CI {d.total_ci95_low:+.3f} to {d.total_ci95_high:+.3f})")
        print(f"  Closing sportsbook MAE: margin={r.loc['raw','closing_spread_mae']:.3f}, "
              f"total={r.loc['raw','closing_total_mae']:.3f}")
    print(f"\nSaved evaluation: {folder}")
    print(f"2026 Week {args.forecast_week} ungraded research forecasts: {len(forecasts)}")
    print("No expected-value, bet selection, or ROI calculations are made.")


if __name__ == "__main__":
    main()
