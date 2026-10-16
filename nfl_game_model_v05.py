"""NFL Game Model v0.5 — exhaustive, *predeclared* feature-group combinations.

Adds NO new statistics and modifies NO previous files. Uses v0.4's combined
feature table, which already contains all feature groups from v0.1–v0.4.
For three optional feature groups (opponent adjustment, pace, QB proxy), tests
all 2**3 == 8 combinations on IDENTICAL game samples and training splits.

Run from the VSCode project directory after v0.4 feature collection:
    python nfl_game_model_v05.py

Inputs (defaults):
    output/v04/pregame_features_v04.csv
    output/v04/feature_manifest_v04.json
    output/historical_labels.csv
    output/model_v01/holdout_2025_predictions.csv

Outputs: a new timestamped directory in output/model_v05 with full per-game
predictions, model_metrics.csv, paired_deltas_vs_raw.csv, ranks, provenance,
and optionally *ungraded, research-only* 2026 Week 6 predictions.

IMPORTANT: 2024 and 2025 have been explored before; every result here is
DIAGNOSTIC ONLY, not an independent estimate of achievable betting ROI.
No winner is automatically selected. Comparing many combinations multiplies
the risk of chance findings. Closing lines are ONLY accuracy benchmarks;
they are not guaranteed bettable at a previous decision time. The QB group
uses a most-recent-game primary passer PROXY, NOT an announced starter.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from nfl_game_model_v01 import PREDICTORS as BASE_PREDICTORS, build_model
from nfl_game_features_v03 import FEATURES as PACE_PREDICTORS
from nfl_game_features_v04 import OUT_COLS as QB_PREDICTORS, VERSION as FEATURE_VERSION

VERSION = "nfl-game-v0.5-feature-combinations"
KEY = ["season", "week", "game_id"]
TARGETS = ("home_margin", "actual_total")
# These are exactly the four added v0.2 variables (in addition to raw EPA).
ADJUSTED_PREDICTORS = [
    "home_off_epa_adj", "home_def_epa_adj",
    "away_off_epa_adj", "away_def_epa_adj",
]
GROUPS = {
    "adjusted": ADJUSTED_PREDICTORS,
    "pace": list(PACE_PREDICTORS),
    "qb_proxy": list(QB_PREDICTORS),
}
BOOTSTRAP_DRAWS = 2000
SEED = 20261015


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variants() -> dict[str, list[str]]:
    """Generate full factorial: raw, singles, pairs, and all groups."""
    options = {"raw": list(BASE_PREDICTORS)}
    group_names = list(GROUPS)
    for n in range(1, len(group_names) + 1):
        for chosen in itertools.combinations(group_names, n):
            name = "+".join(chosen)
            predictors = list(BASE_PREDICTORS)
            for group in chosen:
                predictors.extend(GROUPS[group])
            if len(set(predictors)) != len(predictors):
                raise ValueError(f"Duplicate predictor found for {name}")
            options[name] = predictors
    if len(options) != 2 ** len(GROUPS):
        raise AssertionError("Did not generate all feature group combinations")
    return options


def load_inputs(features_path: Path, labels_path: Path, manifest_path: Path,
                choices: dict[str, list[str]]) -> pd.DataFrame:
    for path in (features_path, labels_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != FEATURE_VERSION:
        raise ValueError("Feature manifest is not from the v0.4 QB proxy builder")
    for field, path in (("v04_features_sha256", features_path),
                        ("base_labels_sha256", labels_path)):
        actual = sha256(path)
        if actual != manifest.get(field):
            raise ValueError(f"v0.4 manifest/hash mismatch ({field}) for {path}")

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    expected_feat = set(KEY + ["home_team", "away_team", "gameday", "v04_features_ready"])
    for predictors in choices.values():
        expected_feat.update(predictors)
    expected_label = set(KEY + ["home_team", "away_team", "home_score", "away_score",
                                *TARGETS, "spread_line", "total_line"])
    if not expected_feat.issubset(features.columns):
        raise ValueError(f"Missing v0.4 feature columns: {sorted(expected_feat - set(features.columns))}")
    if not expected_label.issubset(labels.columns):
        raise ValueError(f"Missing result/closing line columns: {sorted(expected_label - set(labels.columns))}")
    if features.duplicated(KEY).any() or labels.duplicated(KEY).any():
        raise ValueError("Duplicate game IDs in feature or labels data")
    readiness = features["v04_features_ready"].astype(str).str.lower()
    if not readiness.isin(("true", "false")).all():
        raise ValueError("Unrecognized v04_features_ready flag")
    features["v04_features_ready"] = readiness.eq("true")
    joined = features.merge(labels[KEY + ["home_team", "away_team", "home_score",
                                            "away_score", *TARGETS, "spread_line", "total_line"]],
                            on=KEY, how="left", validate="one_to_one", indicator=True,
                            suffixes=("", "_label"))
    if not joined["_merge"].eq("both").all():
        raise ValueError("v0.4 features don't align one-to-one with labels")
    if not (joined["home_team"] == joined["home_team_label"]).all() or not (
            joined["away_team"] == joined["away_team_label"]).all():
        raise ValueError("Team identity mismatch between features and results")
    joined.drop(columns=["_merge", "home_team_label", "away_team_label"], inplace=True)
    if joined.loc[joined.v04_features_ready, list(BASE_PREDICTORS)].isna().any().any():
        raise ValueError("Ready rows missing original v0.1 features")
    if joined.loc[joined.v04_features_ready, ADJUSTED_PREDICTORS].isna().any().any():
        raise ValueError("Ready rows missing opponent-adjusted EPA")
    # Missing QB CPOE / neutral pace is permitted; v0.1's training-only
    # imputer handles it. We compare every model on exactly the same games.
    return joined


def completed(data: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    rows = data.loc[data.v04_features_ready & data.season.isin(years)].copy()
    rows = rows.sort_values(KEY).reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No valid model rows for {years}")
    if rows[["home_score", "away_score", *TARGETS]].isna().any().any():
        raise ValueError(f"Incomplete games in required years {years}")
    if rows[["spread_line", "total_line"]].isna().any().any():
        raise ValueError("No closing-line benchmark for one or more completed games")
    return rows


def fit_models(train: pd.DataFrame, predictors: list[str]) -> dict[str, object]:
    models = {}
    for target in TARGETS:
        model = build_model()  # EXACT v0.1 pipeline, including Ridge alpha=10
        model.fit(train[predictors], train[target])
        models[target] = model
    return models


def forecast(models: dict[str, object], sample: pd.DataFrame,
             predictors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = sample[predictors]
    return (models["home_margin"].predict(X),
            models["actual_total"].predict(X))


def evaluate_season(data: pd.DataFrame, choices: dict[str, list[str]],
                    train_years: tuple[int, ...], test_year: int
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    training, test = completed(data, train_years), completed(data, (test_year,))
    detail = test[KEY + ["home_team", "away_team", "gameday", *TARGETS,
                         "spread_line", "total_line"]].copy()
    book_margin_mae = float((test.home_margin - test.spread_line).abs().mean())
    book_total_mae = float((test.actual_total - test.total_line).abs().mean())
    records = []
    for name, predictors in choices.items():
        models = fit_models(training, predictors)
        margin, total = forecast(models, test, predictors)
        detail[f"{name}__margin_pred"] = margin
        detail[f"{name}__total_pred"] = total
        detail[f"{name}__margin_abs_err"] = abs(test.home_margin.to_numpy() - margin)
        detail[f"{name}__total_abs_err"] = abs(test.actual_total.to_numpy() - total)
        records.append({
            "season": test_year, "training_seasons": ",".join(map(str, train_years)),
            "variant": name, "training_games": len(training), "test_games": len(test),
            "predictor_count": len(predictors),
            "margin_mae": mean_absolute_error(test.home_margin, margin),
            "total_mae": mean_absolute_error(test.actual_total, total),
            "closing_spread_mae": book_margin_mae,
            "closing_total_mae": book_total_mae,
        })
    return detail, pd.DataFrame(records)


def verify_raw_reproduces_v01(detail_2025: pd.DataFrame, old_path: Path) -> float:
    if not old_path.is_file():
        raise FileNotFoundError(f"Missing baseline for reproducibility check: {old_path}")
    prior = pd.read_csv(old_path)
    required = set(KEY + ["pred_home_margin", "pred_total"])
    if not required.issubset(prior.columns):
        raise ValueError("v0.1 prediction file doesn't contain expected keys/forecasts")
    if prior.duplicated(KEY).any():
        raise ValueError("Duplicate v0.1 prediction keys")
    joined = detail_2025[KEY + ["raw__margin_pred", "raw__total_pred"]].merge(
        prior[KEY + ["pred_home_margin", "pred_total"]],
        how="outer", on=KEY, validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Raw model and v0.1 tested on different 2025 games")
    max_diff = max(float((joined.raw__margin_pred - joined.pred_home_margin).abs().max()),
                   float((joined.raw__total_pred - joined.pred_total).abs().max()))
    if not np.isfinite(max_diff) or max_diff > 1e-6:
        raise ValueError(f"Baseline predictions changed! Max difference {max_diff}")
    return max_diff


def paired_intervals(detail: pd.DataFrame, choices: dict[str, list[str]],
                     season: int) -> pd.DataFrame:
    """Unadjusted, descriptive paired bootstrap intervals; NOT selection evidence."""
    rng = np.random.default_rng(SEED + season)
    n = len(detail)
    resample = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    records = []
    for name in choices:
        for market in ("margin", "total"):
            delta = (detail[f"{name}__{market}_abs_err"] -
                     detail[f"raw__{market}_abs_err"]).to_numpy(dtype=float)
            means = delta[resample].mean(axis=1)
            low, high = np.percentile(means, [2.5, 97.5])
            records.append({"season": season, "variant": name, "market": market,
                            "delta_mae_vs_raw": float(delta.mean()),
                            "descriptive_ci_low": float(low),
                            "descriptive_ci_high": float(high),
                            "note": "Explored data; unadjusted for multiple comparisons"})
    return pd.DataFrame(records)


def ungraded_research_forecast(data: pd.DataFrame, choices: dict[str, list[str]],
                               year: int, week: int) -> pd.DataFrame:
    sample = data.loc[data.v04_features_ready & data.season.eq(year) &
                      data.week.eq(week) & data.home_score.isna() &
                      data.away_score.isna()].copy()
    if sample.empty:
        return pd.DataFrame()
    # Only previous COMPLETED seasons train this exploratory forecast.
    training_years = tuple(y for y in (2022, 2023, 2024, 2025) if y < year)
    training = completed(data, training_years)
    result = sample[KEY + ["home_team", "away_team", "gameday"]].copy()
    for name, predictors in choices.items():
        models = fit_models(training, predictors)
        margin, total = forecast(models, sample, predictors)
        result[f"{name}__margin_pred"] = margin
        result[f"{name}__total_pred"] = total
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["research_only"] = True
    result["verified_pregame_timestamp"] = False
    result["note"] = ("QB previous-game passer proxy; no kickoff timestamps, "
                      "verified starter, as-of odds, or ROI")
    return result.sort_values(["gameday", "game_id"])


def save_report(output_dir: Path, choices: dict[str, list[str]],
                year_details: dict[int, pd.DataFrame],
                metrics: pd.DataFrame, intervals: pd.DataFrame,
                research: pd.DataFrame,
                features_path: Path, labels_path: Path,
                manifest_path: Path, baseline_path: Path,
                max_raw_error: float, forecast_year: int, forecast_week: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for year, detail in year_details.items():
        detail.to_csv(output_dir / f"predictions_{year}_diagnostic.csv", index=False)
    metrics.to_csv(output_dir / "combination_metrics.csv", index=False)
    intervals.to_csv(output_dir / "paired_deltas_vs_raw.csv", index=False)
    ranks = metrics.copy()
    ranks["rank_margin_by_year"] = ranks.groupby("season")["margin_mae"].rank(method="min")
    ranks["rank_total_by_year"] = ranks.groupby("season")["total_mae"].rank(method="min")
    ranks.sort_values(["season", "rank_margin_by_year", "variant"]).to_csv(
        output_dir / "diagnostic_ranks_not_selection.csv", index=False)
    if not research.empty:
        research.to_csv(output_dir / f"week{forecast_week}_{forecast_year}_research_only.csv", index=False)
    metadata = {
        "model_version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_groups": GROUPS,
        "predictors_by_variant": choices,
        "reproducibility": {"v01_2025_max_abs_difference": max_raw_error},
        "data_roles": {"2024": "explored diagnostic", "2025": "explored diagnostic",
                       "2026": "future prospective validation ONLY if forecast locked before real kickoff"},
        "multiple_testing_warning": ("All 8 combinations explored on previously seen seasons; "
                                     "individual bootstrapped CIs are NOT adjusted for selection/multiple tests."),
        "forecast_year": forecast_year, "forecast_week": forecast_week,
        "QB_limitation": "Previous-game passer proxy; not verified starter",
        "book_limitation": "Closing lines only; NOT bettable price snapshots",
        "no_EV_or_ROI": True,
        "input_sha256": {
            "features_v04": sha256(features_path), "labels": sha256(labels_path),
            "features_manifest_v04": sha256(manifest_path),
            "v01_predictions_2025": sha256(baseline_path),
            "v05_model_script": sha256(Path(__file__)),
            "v01_model_script": sha256(Path(__file__).with_name("nfl_game_model_v01.py")),
        },
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("output/v04/pregame_features_v04.csv"))
    parser.add_argument("--labels", type=Path, default=Path("output/historical_labels.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("output/v04/feature_manifest_v04.json"))
    parser.add_argument("--v01-holdout", type=Path,
                        default=Path("output/model_v01/holdout_2025_predictions.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("output/model_v05"))
    parser.add_argument("--forecast-year", type=int, default=2026)
    parser.add_argument("--forecast-week", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.forecast_week <= 18:
        parser.error("--forecast-week must be 1..18")
    sources = (args.features, args.labels, args.manifest, args.v01_holdout)
    if any(args.outdir.resolve() == src.resolve().parent for src in sources):
        parser.error("Do not write outputs into a source-data directory")
    choices = variants()
    data = load_inputs(args.features, args.labels, args.manifest, choices)

    by_year: dict[int, pd.DataFrame] = {}
    all_metrics = []
    all_intervals = []
    for training_years, year in [((2022, 2023), 2024),
                                 ((2022, 2023, 2024), 2025)]:
        detail, summary = evaluate_season(data, choices, training_years, year)
        by_year[year] = detail
        all_metrics.append(summary)
        all_intervals.append(paired_intervals(detail, choices, year))
    same_as_v01 = verify_raw_reproduces_v01(by_year[2025], args.v01_holdout)
    metrics = pd.concat(all_metrics, ignore_index=True)
    intervals = pd.concat(all_intervals, ignore_index=True)
    research = ungraded_research_forecast(data, choices,
                                         args.forecast_year, args.forecast_week)
    target = args.outdir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    save_report(target, choices, by_year, metrics, intervals, research,
                args.features, args.labels, args.manifest, args.v01_holdout,
                same_as_v01, args.forecast_year, args.forecast_week)
    print(f"Model: {VERSION}")
    print(f"PASS: RAW v0.1 reproduction on 2025, maximum error {same_as_v01:.2e}")
    print(f"Feature combinations tested: {len(choices)} (3 groups, full factorial)")
    print("Groups: adjusted EPA, offensive pace/volume, historical QB proxy")
    for year in (2024, 2025):
        results = metrics.loc[metrics.season.eq(year)].sort_values("total_mae")
        print(f"\n{year} EXPLORED DIAGNOSTIC — not an untouched holdout")
        print("  (lower MAE is better; ranks shown for diagnosis, NOT selection)")
        print("  variant                     margin MAE  total MAE  delta margin  delta total")
        baseline = results.loc[results.variant.eq("raw")].iloc[0]
        for _, row in results.iterrows():
            print(f"  {row.variant:27s} {row.margin_mae:10.3f} {row.total_mae:10.3f} "
                  f"{row.margin_mae - baseline.margin_mae:+12.3f} "
                  f"{row.total_mae - baseline.total_mae:+12.3f}")
        print(f"  Closing sportsbook MAE: spread={baseline.closing_spread_mae:.3f}, "
              f"total={baseline.closing_total_mae:.3f}")
    print(f"\nSaved reports: {target}")
    print(f"Ungraded {args.forecast_year} Week {args.forecast_week} RESEARCH forecasts: {len(research)}")
    print("None is classified as a pregame wager; no EV, calibrated probabilities or ROI.")
    print("CAUTION: 8 combinations on already explored years cannot validate a profitable edge.")


if __name__ == "__main__":
    main()
