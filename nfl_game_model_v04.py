"""NFL Game Model v0.4 — controlled QB-history PROXY experiment.

NEW independent runner. Does not modify v0.1/v0.2/v0.3 scripts or outputs.
Run from the project root after the v0.4 feature builder:
    python nfl_game_model_v04.py

Compare the EXACT v0.1 raw Ridge predictors and pipeline with those SAME
features plus 10 QB-history proxy statistics. Same games / same train splits.
2024 and 2025 have already been inspected in previous experiments: diagnostic
only, NOT untouched test samples. The models do not contain verified pregame
starting lineups; they proxy QB from the most recent prior game's primary
passer. They do not account for a late QB change or injury announcement.

Saves per-game comparisons, summary MAE, paired bootstrap descriptive
intervals, and ungraded 2026 forecasts only for unfinished games whose
features are ready. No EV, ROI, cover probabilities, or bets.
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
from nfl_game_features_v04 import VERSION as FEATURE_VERSION, OUT_COLS as QB_PREDICTORS

VERSION = "nfl-game-v0.4-qb-proxy-controlled"
KEY = ["season", "week", "game_id"]
TARGETS = ["home_margin", "actual_total"]
VARIANTS = {"raw": list(RAW_PREDICTORS),
            "qb_proxy": list(RAW_PREDICTORS) + list(QB_PREDICTORS)}
BOOT_SEED = 20261015


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(features_file: Path, labels_file: Path, manifest_file: Path) -> pd.DataFrame:
    for path in (features_file, labels_file, manifest_file):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("version") != FEATURE_VERSION:
        raise ValueError("Expected the v0.4 QB proxy feature manifest")
    for key, path in (("v04_features_sha256", features_file),
                      ("base_labels_sha256", labels_file)):
        if sha256(path) != manifest.get(key):
            raise ValueError(f"Changed or mismatched file: {path}")
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)
    required_feat = set(KEY + ["home_team", "away_team", "gameday", "v04_features_ready"]
                        + VARIANTS["qb_proxy"])
    required_label = set(KEY + ["home_score", "away_score", *TARGETS,
                                "spread_line", "total_line"])
    if not required_feat.issubset(features.columns):
        raise ValueError(f"Missing feature columns: {sorted(required_feat-set(features.columns))}")
    if not required_label.issubset(labels.columns):
        raise ValueError(f"Missing label columns: {sorted(required_label-set(labels.columns))}")
    if features.duplicated(KEY).any() or labels.duplicated(KEY).any():
        raise ValueError("Duplicate game keys")
    ready = features.v04_features_ready.astype(str).str.lower()
    if not ready.isin(["true", "false"]).all():
        raise ValueError("Unrecognized v04_features_ready value")
    features["v04_features_ready"] = ready.eq("true")
    df = features.merge(labels[KEY + ["home_score", "away_score", *TARGETS,
                                      "spread_line", "total_line"]],
                        on=KEY, how="left", validate="one_to_one", indicator=True)
    if not df["_merge"].eq("both").all():
        raise ValueError("Feature games have no matching labels")
    df = df.drop(columns="_merge")
    if df.loc[df.v04_features_ready, RAW_PREDICTORS].isna().any().any():
        raise ValueError("Missing raw predictors in a ready game")
    return df


def finished(data: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    subset = data.loc[data.season.isin(years) & data.v04_features_ready].sort_values(KEY).reset_index(drop=True)
    if subset.empty or subset[["home_score", "away_score", *TARGETS]].isna().any().any():
        raise ValueError(f"Missing/incomplete finished games for {years}")
    return subset


def train(training: pd.DataFrame, variant: str):
    predictors = VARIANTS[variant]
    models = {}
    for target in TARGETS:
        model = build_model()  # unchanged v0.1 factory, median imputation fitted only on training
        model.fit(training[predictors], training[target])
        models[target] = model
    return models


def predict(models, games: pd.DataFrame, variant: str):
    x = games[VARIANTS[variant]]
    return models["home_margin"].predict(x), models["actual_total"].predict(x)


def test_season(data: pd.DataFrame, train_years: tuple[int, ...], test_year: int):
    train_data, test = finished(data, train_years), finished(data, (test_year,))
    result = test[KEY + ["home_team", "away_team", "gameday", *TARGETS,
                         "spread_line", "total_line"]].copy()
    summaries = []
    for variant in VARIANTS:
        m = train(train_data, variant)
        margin, total = predict(m, test, variant)
        result[f"{variant}_pred_home_margin"] = margin
        result[f"{variant}_pred_total"] = total
        result[f"{variant}_margin_abs_error"] = abs(test.home_margin.to_numpy() - margin)
        result[f"{variant}_total_abs_error"] = abs(test.actual_total.to_numpy() - total)
        summaries.append({"season": test_year, "train_years": ",".join(map(str, train_years)),
                          "variant": variant, "train_games": len(train_data),
                          "test_games": len(test),
                          "margin_mae": mean_absolute_error(test.home_margin, margin),
                          "total_mae": mean_absolute_error(test.actual_total, total)})
    book_margin = float((result.home_margin - result.spread_line).abs().mean())
    book_total = float((result.actual_total - result.total_line).abs().mean())
    for row in summaries:
        row["closing_spread_mae"] = book_margin
        row["closing_total_mae"] = book_total
    return result, pd.DataFrame(summaries)


def bootstrap_delta(a, b, seed: int):
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(diff), size=(2000, len(diff)))
    lo, hi = np.quantile(diff[index].mean(axis=1), [0.025, 0.975])
    return float(np.mean(diff)), float(lo), float(hi)


def verify_v01_baseline(detail_2025: pd.DataFrame, baseline_path: Path) -> str:
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Cannot verify raw reproducibility; missing {baseline_path}")
    old = pd.read_csv(baseline_path)
    if old.duplicated(KEY).any():
        raise ValueError("Duplicate v0.1 baseline game key")
    required = KEY + ["pred_home_margin", "pred_total"]
    if not set(required).issubset(old.columns):
        raise ValueError("v0.1 baseline missing forecast columns")
    compare = detail_2025[KEY + ["raw_pred_home_margin", "raw_pred_total"]].merge(
        old[required], on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not compare["_merge"].eq("both").all():
        raise ValueError("v0.1 and v0.4 are using different historical games")
    error = max(float((compare.raw_pred_home_margin - compare.pred_home_margin).abs().max()),
                float((compare.raw_pred_total - compare.pred_total).abs().max()))
    if not np.isfinite(error) or error > 1e-6:
        raise ValueError(f"Baseline model not reproduced: maximum error={error}")
    return f"PASS: RAW 2025 reproduces v0.1, max difference {error:.2e}"


def forecasts(data: pd.DataFrame, year: int, week: int):
    upcoming = data.loc[(data.season == year) & (data.week == week) & data.v04_features_ready
                        & data.home_score.isna() & data.away_score.isna()].copy()
    if upcoming.empty:
        return pd.DataFrame()
    hist = finished(data, (2022, 2023, 2024, 2025))
    output = upcoming[KEY + ["home_team", "away_team", "gameday"]].copy()
    for variant in VARIANTS:
        model = train(hist, variant)
        m, t = predict(model, upcoming, variant)
        output[f"{variant}_pred_home_margin"] = m
        output[f"{variant}_pred_total"] = t
    output["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    output["research_only"] = True
    output["qb_id_is_confirmed"] = False
    output["note"] = "Last-game passer proxy; NOT confirmed pregame starter; no as-of sportsbook odds"
    return output.sort_values(["gameday", "game_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("output/v04/pregame_features_v04.csv"))
    parser.add_argument("--labels", type=Path, default=Path("output/historical_labels.csv"))
    parser.add_argument("--feature-manifest", type=Path, default=Path("output/v04/feature_manifest_v04.json"))
    parser.add_argument("--v01-holdout", type=Path, default=Path("output/model_v01/holdout_2025_predictions.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("output/model_v04"))
    parser.add_argument("--forecast-week", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.forecast_week <= 18:
        parser.error("--forecast-week must be from 1 to 18")
    if any(args.outdir.resolve() == source.resolve().parent for source in
           (args.features, args.labels, args.feature_manifest, args.v01_holdout)):
        parser.error("Do not use the source folder as the model output folder")
    data = load_data(args.features, args.labels, args.feature_manifest)
    all_summaries, by_year, deltas = [], {}, []
    for years, year in [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]:
        detail, summary = test_season(data, years, year)
        by_year[year] = detail
        all_summaries.append(summary)
        margin = bootstrap_delta(detail.raw_margin_abs_error, detail.qb_proxy_margin_abs_error,
                                 BOOT_SEED + year)
        total = bootstrap_delta(detail.raw_total_abs_error, detail.qb_proxy_total_abs_error,
                                BOOT_SEED + year + 100)
        deltas.append({"season": year, "qb_minus_raw_margin_mae": margin[0],
                       "margin_ci_low": margin[1], "margin_ci_high": margin[2],
                       "qb_minus_raw_total_mae": total[0],
                       "total_ci_low": total[1], "total_ci_high": total[2]})
    baseline_message = verify_v01_baseline(by_year[2025], args.v01_holdout)
    metrics = pd.concat(all_summaries, ignore_index=True)
    intervals = pd.DataFrame(deltas)
    live = forecasts(data, 2026, args.forecast_week)
    output_dir = args.outdir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    for year, detail in by_year.items():
        detail.round(6).to_csv(output_dir / f"comparison_{year}.csv", index=False)
    metrics.round(6).to_csv(output_dir / "model_metrics.csv", index=False)
    intervals.round(6).to_csv(output_dir / "paired_differences.csv", index=False)
    if not live.empty:
        live.round(6).to_csv(output_dir / f"week{args.forecast_week}_2026_research_forecasts.csv", index=False)
    metadata = {
        "version": VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_predictors": VARIANTS["raw"], "qb_proxy_predictors": VARIANTS["qb_proxy"],
        "model_factory": "unchanged nfl_game_model_v01.build_model; train-only imputation, scaled Ridge alpha=10",
        "qb_identity": "Primary passer in most recent PRIOR team game; NOT verified starter",
        "2024_2025": "Previously inspected: diagnostic only, NOT independent validation",
        "2026": "Forward test only if forecast saved before actual kickoff",
        "features_sha256": sha256(args.features),
        "labels_sha256": sha256(args.labels),
        "features_manifest_sha256": sha256(args.feature_manifest),
        "baseline_check": baseline_message,
        "no_betting_or_ROI_claims": True,
    }
    (output_dir / "model_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Model: {VERSION}")
    print(baseline_message)
    for year in (2024, 2025):
        rows = metrics.loc[metrics.season.eq(year)].set_index("variant")
        diff = intervals.loc[intervals.season.eq(year)].iloc[0]
        print(f"\n{year} PREVIOUSLY EXPLORED (diagnostic, NOT clean holdout)")
        for variant in ("raw", "qb_proxy"):
            row = rows.loc[variant]
            print(f"  {variant:8s}: n={int(row.test_games)} | margin MAE={row.margin_mae:.3f} | total MAE={row.total_mae:.3f}")
        print(f"  QB - RAW MAE: margin={diff.qb_minus_raw_margin_mae:+.3f} "
              f"(95% CI {diff.margin_ci_low:+.3f} to {diff.margin_ci_high:+.3f}), "
              f"total={diff.qb_minus_raw_total_mae:+.3f} "
              f"(95% CI {diff.total_ci_low:+.3f} to {diff.total_ci_high:+.3f})")
        print(f"  Sportsbook close: margin={rows.loc['raw', 'closing_spread_mae']:.3f}, "
              f"total={rows.loc['raw', 'closing_total_mae']:.3f}")
    print(f"\nSaved results: {output_dir}")
    print(f"2026 Week {args.forecast_week} unfinished-game research forecasts: {len(live)}")
    print("QB is historical last-game passer PROXY; not a verified announced starter.")
    print("No calibrated probabilities, live sportsbook odds, EV or ROI calculations.")


if __name__ == "__main__":
    main()
