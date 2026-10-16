"""NFL Game Engine v0.2: opponent-adjusted EPA feature builder.

Standalone NEW module; does not edit v0.1 scripts or output files.
Run from the same directory as nfl_game_data_starter.py:

    python nfl_game_features_v02.py

Reads:  output/pregame_features.csv, output/historical_labels.csv
Fetches: nflverse play-by-play via nflreadpy (same data source as v0.1)
Writes: output/v02/pregame_features_v02.csv
        output/v02/weekly_team_stats_v02.csv
        output/v02/opponent_adjustment_audit_v02.csv
        output/v02/feature_manifest_v02.json

Definition:
  For a team's last four completed games strictly before target week, adjust
  each observed EPA/play using the opponent's performance in its other recent
  games, all strictly before target week. Exclude the matchup itself from the
  opponent reference (leave-one-matchup-out) and shrink the reference toward
  0.0 EPA/play with an equivalent of four prior games.

  off_epa_adj = mean(game_off_epa - opponent_def_reference)
  def_epa_adj = mean(game_def_epa_allowed - opponent_off_reference)

Lower defensive EPA (adjusted or raw) is better. Neither historic target-game
results nor future-week data enter any target-week statistic.

This is a feature experiment, NOT a trained model or betting recommendation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

from nfl_game_data_starter import compute_weekly_team_stats

GAME_KEY = ["season", "week", "game_id"]
ADJUSTED = ["home_off_epa_adj", "home_def_epa_adj", "away_off_epa_adj", "away_def_epa_adj"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_weekly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Attach each played game's opponent, requiring exactly two NFL teams."""
    required = {"season", "week", "game_id", "team", "off_epa_play", "def_epa_allowed"}
    missing = required - set(weekly.columns)
    if missing:
        raise ValueError(f"Missing weekly stats columns: {sorted(missing)}")
    keys = ["season", "week", "game_id"]
    sizes = weekly.groupby(keys, dropna=False)["team"].nunique()
    if (sizes != 2).any():
        raise ValueError("Played game must contain exactly two distinct teams; inspect PBP groupings")
    if weekly.duplicated(keys + ["team"]).any():
        raise ValueError("Duplicate team-game in PBP summary")

    opponent = weekly[keys + ["team"]].rename(columns={"team": "opponent"})
    merged = weekly.merge(opponent, on=keys, how="left", validate="many_to_many")
    merged = merged.loc[merged["team"] != merged["opponent"]].copy()
    if len(merged) != len(weekly) or merged["opponent"].isna().any():
        raise ValueError("Could not map every observed team-game to one opponent")
    return merged.sort_values(["season", "team", "week", "game_id"]).reset_index(drop=True)


def complete_week_by_season(labels: pd.DataFrame) -> dict[int, int]:
    """Find last consecutively completed REG week; a partial week is not complete."""
    required = {"season", "week", "home_score", "away_score"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    result = {}
    for season, rows in labels.groupby("season"):
        last_complete = 0
        for week in sorted(rows.week.unique()):
            if int(week) != last_complete + 1:
                break
            this_week = rows.loc[rows.week == week]
            if this_week[["home_score", "away_score"]].isna().any().any():
                break
            last_complete = int(week)
        result[int(season)] = last_complete
    return result


def build_adjusted_features(
    base: pd.DataFrame,
    labels: pd.DataFrame,
    weekly: pd.DataFrame,
    window: int = 4,
    shrink_games: int = 4,
) -> pd.DataFrame:
    """Produce four new feature columns; never use results for predictors.

    Labels determine only whether an entire prior week has finished. They are
    *not* used to calculate any EPA value.
    """
    if window <= 0 or shrink_games < 0:
        raise ValueError("window must be positive and shrink_games cannot be negative")
    needed = set(GAME_KEY + ["home_team", "away_team", "home_prior_games", "away_prior_games"])
    if not needed.issubset(base.columns):
        raise ValueError(f"Missing v0.1 feature columns: {sorted(needed - set(base.columns))}")
    if base.duplicated(GAME_KEY).any():
        raise ValueError("Duplicate games in base features")
    if labels.duplicated(GAME_KEY).any():
        raise ValueError("Duplicate games in labels")
    if not base[GAME_KEY].merge(labels[GAME_KEY], on=GAME_KEY, how="left", indicator=True)["_merge"].eq("both").all():
        raise ValueError("A v0.1 feature game has no matching label game")

    max_complete = complete_week_by_season(labels)
    # Indexed team histories; group includes only actually PLAYED PBP games.
    histories: dict[tuple[int, str], list[dict]] = {}
    for (season, team), grp in weekly.groupby(["season", "team"]):
        histories[(int(season), str(team))] = grp.sort_values(["week", "game_id"]).to_dict("records")

    cache: dict[tuple[int, int, str], tuple[float, float, int]] = {}

    def for_team(season: int, week: int, team: str) -> tuple[float, float, int]:
        key = (season, week, team)
        if key in cache:
            return cache[key]
        played = [g for g in histories.get((season, team), []) if int(g["week"]) < week][-window:]
        if not played:
            answer = (float("nan"), float("nan"), 0)
            cache[key] = answer
            return answer

        adjusted_off = []
        adjusted_def = []
        for game in played:
            opponent = str(game["opponent"])
            # As-of target week; omit the head-to-head game to avoid feeding
            # the team's own performance back into the opponent adjustment.
            other_games = [g for g in histories.get((season, opponent), [])
                           if int(g["week"]) < week and g["game_id"] != game["game_id"]][-window:]
            n = len(other_games)
            # n/(n+shrink_games) * sample mean = sum/(n+shrink_games).
            # If n=0, fallback 0; if shrink_games=0, divide by n (nonzero).
            opponent_def = (sum(float(g["def_epa_allowed"]) for g in other_games) /
                            (n + shrink_games)) if n else 0.0
            opponent_off = (sum(float(g["off_epa_play"]) for g in other_games) /
                            (n + shrink_games)) if n else 0.0
            adjusted_off.append(float(game["off_epa_play"]) - opponent_def)
            adjusted_def.append(float(game["def_epa_allowed"]) - opponent_off)
        answer = (sum(adjusted_off) / len(adjusted_off),
                  sum(adjusted_def) / len(adjusted_def), len(played))
        cache[key] = answer
        return answer

    out = base.copy()
    # 'ready' identifies weeks with all earlier results known at download time.
    # For incomplete 2026 season, week 6 is ready when weeks 1-5 are complete,
    # whereas weeks 7+ are not yet valid prospective snapshots.
    out["v02_features_ready"] = False
    for col in ADJUSTED:
        out[col] = float("nan")

    for idx, row in out.iterrows():
        season, week = int(row["season"]), int(row["week"])
        if week > max_complete.get(season, 0) + 1:
            continue
        h_off, h_def, h_n = for_team(season, week, str(row["home_team"]))
        a_off, a_def, a_n = for_team(season, week, str(row["away_team"]))
        if h_n != int(row["home_prior_games"]) or a_n != int(row["away_prior_games"]):
            raise ValueError(f"Prior-game count mismatch: {row['game_id']}; check PBP source revisions")
        out.loc[idx, ADJUSTED] = [h_off, h_def, a_off, a_def]
        # Week 1 lacks prior-game features by design.
        out.loc[idx, "v02_features_ready"] = h_n > 0 and a_n > 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("output/pregame_features.csv"))
    parser.add_argument("--labels", type=Path, default=Path("output/historical_labels.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("output/v02"))
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--shrink-games", type=int, default=4)
    args = parser.parse_args()
    if not args.features.exists() or not args.labels.exists():
        parser.error("Missing v0.1 CSV(s); run nfl_game_data_starter.py first")
    if args.window <= 0 or args.shrink_games < 0:
        parser.error("--window must be positive; --shrink-games cannot be negative")
    if args.outdir.resolve() == args.features.resolve().parent or args.outdir.resolve() == args.labels.resolve().parent:
        parser.error("Use a separate v0.2 output directory to protect v0.1 files")

    base = pd.read_csv(args.features)
    labels = pd.read_csv(args.labels)
    years = sorted(int(y) for y in base["season"].unique())
    print(f"Downloading NFL PBP for {years} (v0.1 scripts remain unchanged) ...", flush=True)
    pbp = nfl.load_pbp(years).to_pandas()
    weekly = prepare_weekly(compute_weekly_team_stats(pbp))
    print(f"PBP rows: {len(pbp):,}; observed team-game rows: {len(weekly):,}", flush=True)
    result = build_adjusted_features(base, labels, weekly, args.window, args.shrink_games)

    args.outdir.mkdir(parents=True, exist_ok=True)
    feature_file = args.outdir / "pregame_features_v02.csv"
    weekly_file = args.outdir / "weekly_team_stats_v02.csv"
    audit_file = args.outdir / "opponent_adjustment_audit_v02.csv"
    manifest_file = args.outdir / "feature_manifest_v02.json"
    result.to_csv(feature_file, index=False)
    weekly.to_csv(weekly_file, index=False)
    audit_cols = GAME_KEY + ["home_team", "away_team", "home_off_epa_play", "home_off_epa_adj",
                              "home_def_epa_allowed", "home_def_epa_adj", "away_off_epa_play", "away_off_epa_adj",
                              "away_def_epa_allowed", "away_def_epa_adj", "v02_features_ready"]
    audit = result.loc[result["v02_features_ready"], audit_cols]
    audit.to_csv(audit_file, index=False)
    manifest = {
        "version": "nfl-game-features-v0.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": years,
        "window": args.window,
        "shrink_games": args.shrink_games,
        "base_features_sha256": sha256(args.features),
        "base_labels_sha256": sha256(args.labels),
        "weekly_stats_sha256": sha256(weekly_file),
        "v02_features_sha256": sha256(feature_file),
        "v02_ready_game_rows": int(result["v02_features_ready"].sum()),
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(result):,} game rows: {feature_file}")
    print(f"Ready for feature research: {int(result.v02_features_ready.sum()):,}")
    print(f"Wrote audit: {audit_file}")
    print(f"Wrote source snapshot: {weekly_file}")
    print("This is a feature builder only; no trained model, wagers or ROI.")


if __name__ == "__main__":
    main()
