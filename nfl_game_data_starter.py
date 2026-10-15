"""NFL game-level, pregame feature builder (research starter, NOT a betting model).

Usage:
    pip install nflreadpy pandas pyarrow
    python nfl_game_data_starter.py --start-season 2022 --end-season 2026 --outdir output

Produces:
    pregame_features.csv  : one record per regular season game; no final results or betting lines
    historical_labels.csv : final scores and historical *closing* betting lines, for evaluation only

IMPORTANT:
  * Each game's features are computed using games from STRICTLY EARLIER WEEKS
    in the SAME season.  Week 1 features are deliberately missing.
  * Historical schedule odds are generally closing lines, not proof of odds
    available at a specific earlier timestamp. Do NOT claim executable ROI
    from these lines. For that, add timestamped book-specific odds snapshots.
  * No injury, weather, lineup or projection data are included in this starter.
  * nflverse upstream historical data may be revised; snapshot raw inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

FEATURES = [
    "off_epa_play",
    "off_success_rate",
    "off_pass_rate",
    "off_explosive_rate",
    "def_epa_allowed",
    "def_success_allowed",
    "def_explosive_allowed",
]


def _num_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


def load_raw(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Downloading nflverse PBP and schedules for {seasons} ...")
    pbp = nfl.load_pbp(seasons).to_pandas()
    schedule = nfl.load_schedules(seasons).to_pandas()
    return pbp, schedule


def compute_weekly_team_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Summarize regular-season offensive and defensive efficiency by team/week."""
    required = {"season", "week", "game_id", "posteam", "defteam", "epa"}
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing columns: {sorted(missing)}")
    plays = pbp.copy()
    if "season_type" in plays.columns:
        plays = plays.loc[plays["season_type"].eq("REG")].copy()
    if "play_type" in plays.columns:
        plays = plays.loc[plays["play_type"].isin(["pass", "run"])].copy()
    else:
        # Fallback for dataset versions without a play_type column.
        plays = plays.loc[(_num_col(plays, "rush") == 1) | (_num_col(plays, "pass") == 1)].copy()
    plays = plays.dropna(subset=["posteam", "defteam", "epa"])
    plays["epa"] = pd.to_numeric(plays["epa"], errors="coerce")
    plays = plays.dropna(subset=["epa"])
    plays["success_clean"] = _num_col(plays, "success").fillna(0)
    plays["is_pass"] = _num_col(plays, "pass").fillna(0)
    plays["explosive"] = (
        ((_num_col(plays, "pass") == 1) & (_num_col(plays, "yards_gained") >= 15)) |
        ((_num_col(plays, "rush") == 1) & (_num_col(plays, "yards_gained") >= 10))
    ).astype(float)

    group_off = ["season", "week", "game_id", "posteam"]
    off = (plays.groupby(group_off, as_index=False)
           .agg(off_epa_play=("epa", "mean"),
                off_success_rate=("success_clean", "mean"),
                off_pass_rate=("is_pass", "mean"),
                off_explosive_rate=("explosive", "mean"),
                off_plays=("epa", "size"))
           .rename(columns={"posteam": "team"}))
    group_def = ["season", "week", "game_id", "defteam"]
    defense = (plays.groupby(group_def, as_index=False)
               .agg(def_epa_allowed=("epa", "mean"),
                    def_success_allowed=("success_clean", "mean"),
                    def_explosive_allowed=("explosive", "mean"))
               .rename(columns={"defteam": "team"}))
    weekly = off.merge(defense, on=["season", "week", "game_id", "team"], how="outer", validate="one_to_one")
    return weekly.sort_values(["season", "team", "week", "game_id"]).reset_index(drop=True)


def build_features(schedules: pd.DataFrame, weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = schedules.copy()
    required = {"season", "week", "game_id", "home_team", "away_team"}
    missing = required - set(schedule.columns)
    if missing:
        raise ValueError(f"Schedule missing columns: {sorted(missing)}")
    if "game_type" in schedule:
        schedule = schedule.loc[schedule["game_type"].eq("REG")].copy()
    schedule = schedule.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    # Snapshots are indexed by prior week, never by the row's own week.
    # This avoids accidentally incorporating a team's target-game statistics.
    snapshot_rows = []
    for (season, team), games in weekly.groupby(["season", "team"], sort=False):
        games = games.sort_values(["week", "game_id"])
        season_end = int(schedule.loc[schedule["season"].eq(season), "week"].max())
        for target_week in range(1, season_end + 1):
            past = games.loc[games["week"] < target_week].tail(4)
            row = {"season": season, "week": target_week, "team": team,
                   "prior_games": len(past)}
            for metric in FEATURES:
                row[metric] = float(past[metric].mean()) if len(past) else float("nan")
            snapshot_rows.append(row)
    snaps = pd.DataFrame(snapshot_rows)
    if snaps.empty:
        raise ValueError("No PBP team-week history; check season/data availability.")
    home = snaps.add_prefix("home_").rename(columns={"home_season": "season", "home_week": "week", "home_team": "home_team"})
    away = snaps.add_prefix("away_").rename(columns={"away_season": "season", "away_week": "week", "away_team": "away_team"})
    base = ["season", "week", "game_id", "home_team", "away_team"]
    if "gameday" in schedule:
        base.append("gameday")
    for rest in ["home_rest", "away_rest"]:
        if rest in schedule:
            base.append(rest)
    features = schedule[base].merge(home, on=["season", "week", "home_team"], how="left", validate="many_to_one")
    features = features.merge(away, on=["season", "week", "away_team"], how="left", validate="many_to_one")
    # Do not mix future results/closing lines into the model's feature table.
    label_columns = base[:5] + [c for c in ["home_score", "away_score", "spread_line", "total_line", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds"] if c in schedule.columns]
    labels = schedule[label_columns].copy()
    labels["home_margin"] = labels["home_score"] - labels["away_score"] if {"home_score", "away_score"}.issubset(labels.columns) else float("nan")
    labels["actual_total"] = labels["home_score"] + labels["away_score"] if {"home_score", "away_score"}.issubset(labels.columns) else float("nan")
    return features, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--outdir", type=Path, default=Path("output"))
    args = parser.parse_args()
    if args.end_season < args.start_season:
        parser.error("--end-season cannot be earlier than --start-season")
    seasons = list(range(args.start_season, args.end_season + 1))
    args.outdir.mkdir(parents=True, exist_ok=True)
    pbp, schedules = load_raw(seasons)
    print(f"PBP rows: {len(pbp):,}; schedule rows: {len(schedules):,}")
    weekly = compute_weekly_team_stats(pbp)
    features, labels = build_features(schedules, weekly)
    feature_path = args.outdir / "pregame_features.csv"
    label_path = args.outdir / "historical_labels.csv"
    features.to_csv(feature_path, index=False)
    labels.to_csv(label_path, index=False)
    print(f"Wrote {len(features):,} game rows: {feature_path}")
    print(f"Wrote results/closing-line labels: {label_path}")
    print("NEXT: add historical as-of injury/QB/odds snapshots and trained models. No bets are produced.")


if __name__ == "__main__":
    main()
