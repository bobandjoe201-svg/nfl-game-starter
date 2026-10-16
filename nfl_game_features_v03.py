"""NFL Game Engine v0.3: offensive pace / play-volume feature experiment.

NEW script: does NOT edit existing v0.1 or v0.2 scripts or outputs.

    python nfl_game_features_v03.py

Reads immutable v0.2 snapshots:
  output/v02/pregame_features_v02.csv
  output/v02/weekly_team_stats_v02.csv
  output/v02/feature_manifest_v02.json
  output/historical_labels.csv

Downloads nflverse PBP using nflreadpy for clock-based metrics. To reproduce a
prior run without downloading current/revised PBP again, use:
  --pace-snapshot output/v03/weekly_pace_stats_v03.csv 

Outputs in output/v03/ (never overwrites v0.1/v0.2):
  pregame_features_v03.csv
  weekly_pace_stats_v03.csv
  pace_feature_audit_v03.csv
  feature_manifest_v03.json

PACE DEFINITIONS:
 * off_plays_pg: mean offensive pass/run plays, per game, across last four
   actually played team games, sourced from v0.2's frozen weekly stats.
 * neutral_sec_per_snap: mean seconds between CONSECUTIVE pass/run plays by
   the same offense on the same drive and in the same quarter, filtered to:
   quarter <=3; pre-play absolute score differential <=8; not the final 2
   minutes of first half; and elapsed time of 8..60 seconds. The statistic
   is first calculated per team-game, then averaged across the last four.
   It is an approximate, observable pace proxy, not true possession-level
   or NFL official 'seconds per play'.

All rolling features use only weeks STRICTLY before the predicted game's week.
Missing pace is left missing for model-pipeline training-only imputation.
Does not make ROI, EV, or bet recommendations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["home_off_plays_pg", "away_off_plays_pg",
            "home_neutral_sec_per_snap", "away_neutral_sec_per_snap"]
KEY = ["season", "week", "game_id"]
TEAM_GAME = KEY + ["team"]
VERSION = "nfl-game-features-v0.3-pace-volume"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_inputs(features_file: Path, weekly_file: Path, labels_file: Path,
                  manifest_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = (features_file, weekly_file, labels_file, manifest_file)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("version") != "nfl-game-features-v0.2":
        raise ValueError("Incorrect v0.2 feature manifest")
    for field, path in (("v02_features_sha256", features_file),
                        ("weekly_stats_sha256", weekly_file),
                        ("base_labels_sha256", labels_file)):
        if sha256(path) != manifest.get(field):
            raise ValueError(f"v0.2 snapshot/hash mismatch: {path}. Preserve/refresh matching inputs first.")
    base = pd.read_csv(features_file)
    weekly = pd.read_csv(weekly_file)
    labels = pd.read_csv(labels_file)
    for frame, keys, name in ((base, KEY, "features"),
                              (weekly, TEAM_GAME, "weekly"),
                              (labels, KEY, "labels")):
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate {name} keys")
    required = set(KEY + ["home_team", "away_team", "home_prior_games", "away_prior_games", "v02_features_ready"])
    if not required.issubset(base.columns):
        raise ValueError(f"Missing v0.2 feature columns: {sorted(required - set(base.columns))}")
    required_weekly = set(TEAM_GAME + ["off_plays"])
    if not required_weekly.issubset(weekly.columns):
        raise ValueError(f"Missing weekly columns: {sorted(required_weekly - set(weekly.columns))}")
    required_label = set(KEY + ["home_score", "away_score"])
    if not required_label.issubset(labels.columns):
        raise ValueError(f"Missing label columns: {sorted(required_label - set(labels.columns))}")
    if not base[KEY].merge(labels[KEY], on=KEY, how="left", indicator=True)["_merge"].eq("both").all():
        raise ValueError("Feature games without labels")
    if not base["v02_features_ready"].astype(str).str.lower().isin(["true", "false"]).all():
        raise ValueError("Unrecognized v02_features_ready values")
    base["v02_features_ready"] = base["v02_features_ready"].astype(str).str.lower().eq("true")
    return base, weekly, labels


def summarize_neutral_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """Create one team-game neutral-pace summary using chronological PBP clocks.

    Keeps original play_id ordering. Shift is WITHIN each game/drive/offense,
    so we never calculate clock intervals across a change of possession.
    """
    required = set(KEY + ["posteam", "drive", "qtr", "play_id",
                          "game_seconds_remaining", "score_differential", "play_type"])
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"PBP has no columns required for neutral pace: {sorted(missing)}")
    plays = pbp
    if "season_type" in plays.columns:
        plays = plays.loc[plays["season_type"].eq("REG")]
    plays = plays.loc[plays["play_type"].isin(["pass", "run"])].copy()
    for col in ("drive", "qtr", "play_id", "game_seconds_remaining", "score_differential"):
        plays[col] = pd.to_numeric(plays[col], errors="coerce")
    plays = plays.dropna(subset=["posteam", "drive", "qtr", "play_id",
                                      "game_seconds_remaining", "score_differential"])
    plays = plays.sort_values(["season", "week", "game_id", "drive", "posteam", "play_id"])
    within_drive = ["season", "week", "game_id", "drive", "posteam"]
    grp = plays.groupby(within_drive, sort=False)
    plays["next_clock"] = grp["game_seconds_remaining"].shift(-1)
    plays["next_qtr"] = grp["qtr"].shift(-1)
    plays["elapsed"] = plays["game_seconds_remaining"] - plays["next_clock"]
    # Quarter 2 has overall game clock 1800..2700; 2-min warning = 1920.
    before_q2_two_min = ~((plays["qtr"] == 2) & (plays["game_seconds_remaining"] <= 1920))
    usable = (
        (plays["qtr"] <= 3)
        & (plays["next_qtr"] == plays["qtr"])
        & (plays["score_differential"].abs() <= 8)
        & before_q2_two_min
        & plays["elapsed"].between(8, 60, inclusive="both")
    )
    relevant = plays.loc[usable]
    out = (relevant.groupby(["season", "week", "game_id", "posteam"], as_index=False)
           .agg(neutral_sec_per_snap=("elapsed", "mean"),
                neutral_pace_pairs=("elapsed", "size"))
           .rename(columns={"posteam": "team"}))
    return out.sort_values(TEAM_GAME).reset_index(drop=True)


def last_complete_week(labels: pd.DataFrame) -> dict[int, int]:
    """Do not ready a distant future week using today's earlier-week data."""
    result = {}
    for season, group in labels.groupby("season"):
        last = 0
        for week in sorted(group["week"].unique()):
            if int(week) != last + 1:
                break
            games = group.loc[group["week"] == week]
            if games[["home_score", "away_score"]].isna().any().any():
                break
            last = int(week)
        result[int(season)] = last
    return result


def attach_pace_features(base: pd.DataFrame, weekly: pd.DataFrame,
                         labels: pd.DataFrame, pace: pd.DataFrame,
                         window: int = 4) -> pd.DataFrame:
    if window != 4:
        raise ValueError("Only window=4 is supported to match the fixed v0.1/v0.2 baseline")
    if pace.duplicated(TEAM_GAME).any():
        raise ValueError("Duplicate team-game pace records")
    weekly = weekly.merge(pace[TEAM_GAME + ["neutral_sec_per_snap", "neutral_pace_pairs"]],
                          on=TEAM_GAME, how="left", validate="one_to_one")
    histories = {(int(year), str(team)): g.sort_values(["week", "game_id"]).copy()
                 for (year, team), g in weekly.groupby(["season", "team"])}
    max_complete = last_complete_week(labels)
    output = base.copy()
    output["v03_features_ready"] = False
    for col in FEATURES:
        output[col] = np.nan

    cache: dict[tuple[int, int, str], tuple[float, float, int]] = {}

    def prior_summary(season: int, week: int, team: str) -> tuple[float, float, int]:
        key = (season, week, team)
        if key in cache:
            return cache[key]
        hist = histories.get((season, team))
        if hist is None:
            result = (np.nan, np.nan, 0)
        else:
            prior = hist.loc[hist["week"] < week].tail(window)
            # Match the v0.1 per-game average calculation (not play-weighted).
            off_plays = float(prior["off_plays"].mean()) if len(prior) else np.nan
            secs = float(prior["neutral_sec_per_snap"].mean()) if prior["neutral_sec_per_snap"].notna().any() else np.nan
            result = (off_plays, secs, len(prior))
        cache[key] = result
        return result

    for i, row in output.iterrows():
        season, week = int(row["season"]), int(row["week"])
        if week > max_complete.get(season, 0) + 1:
            continue
        home = prior_summary(season, week, str(row["home_team"]))
        away = prior_summary(season, week, str(row["away_team"]))
        if home[2] != int(row["home_prior_games"]) or away[2] != int(row["away_prior_games"]):
            raise ValueError(f"Prior-game count mismatch: {row['game_id']}")
        output.loc[i, FEATURES] = [home[0], away[0], home[1], away[1]]
        # Same game sample as v0.1/v0.2; missing pace is allowed and only
        # imputed from TRAINING values by sklearn's pipeline.
        output.loc[i, "v03_features_ready"] = bool(row["v02_features_ready"])
    if int(output["v03_features_ready"].sum()) != int(base["v02_features_ready"].sum()):
        raise ValueError("v0.3 changed which games are ready; abort controlled comparison")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("output/v02/pregame_features_v02.csv"))
    parser.add_argument("--weekly", type=Path, default=Path("output/v02/weekly_team_stats_v02.csv"))
    parser.add_argument("--labels", type=Path, default=Path("output/historical_labels.csv"))
    parser.add_argument("--v02-manifest", type=Path, default=Path("output/v02/feature_manifest_v02.json"))
    parser.add_argument("--pace-snapshot", type=Path, default=None,
                        help="Optional existing pace CSV to rerun without downloading PBP")
    parser.add_argument("--outdir", type=Path, default=Path("output/v03"))
    args = parser.parse_args()
    protected = [args.features, args.weekly, args.labels, args.v02_manifest]
    if any(args.outdir.resolve() == path.resolve().parent for path in protected):
        parser.error("Use a separate v0.3 output directory; do not overwrite older versions")

    base, weekly, labels = verify_inputs(args.features, args.weekly, args.labels, args.v02_manifest)
    if args.pace_snapshot:
        print(f"Loading archived weekly pace snapshot: {args.pace_snapshot}", flush=True)
        pace = pd.read_csv(args.pace_snapshot)
    else:
        import nflreadpy as nfl
        years = sorted(base["season"].astype(int).unique().tolist())
        print(f"Downloading NFL PBP {years} to calculate neutral pace ...", flush=True)
        pbp = nfl.load_pbp(years).to_pandas()
        pace = summarize_neutral_pace(pbp)
        print(f"Downloaded PBP rows: {len(pbp):,}; neutral team-games: {len(pace):,}", flush=True)
    required_pace = set(TEAM_GAME + ["neutral_sec_per_snap", "neutral_pace_pairs"])
    if not required_pace.issubset(pace.columns):
        raise ValueError(f"Pace snapshot is missing: {sorted(required_pace - set(pace.columns))}")
    if not set(map(tuple, pace[TEAM_GAME].to_records(index=False))).issubset(
            set(map(tuple, weekly[TEAM_GAME].to_records(index=False)))):
        raise ValueError("Pace data includes team-games outside the v0.2 frozen weekly snapshot")

    enriched = attach_pace_features(base, weekly, labels, pace)
    ready = enriched.loc[enriched["v03_features_ready"]]
    if ready.empty:
        raise ValueError("No ready feature rows")
    if ready[["home_off_plays_pg", "away_off_plays_pg"]].isna().any().any():
        raise ValueError("A ready game is missing offensive play volume; investigate source")
    pace_missing = ready[["home_neutral_sec_per_snap", "away_neutral_sec_per_snap"]].isna().sum()
    print(f"Ready games: {len(ready):,}; missing neutral-pace snapshots: {pace_missing.to_dict()}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    features_file = args.outdir / "pregame_features_v03.csv"
    pace_file = args.outdir / "weekly_pace_stats_v03.csv"
    audit_file = args.outdir / "pace_feature_audit_v03.csv"
    manifest_file = args.outdir / "feature_manifest_v03.json"
    # Refuse to silently overwrite a snapshot from another run.
    if any(p.exists() for p in (features_file, pace_file, audit_file, manifest_file)):
        raise FileExistsError(f"v0.3 output files already exist under {args.outdir}. Use --outdir for a new run.")
    enriched.to_csv(features_file, index=False)
    pace.to_csv(pace_file, index=False)
    audit = ready[KEY + ["home_team", "away_team", "v03_features_ready"] + FEATURES].copy()
    audit.to_csv(audit_file, index=False)
    manifest = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window": 4,
        "pace_definition": "same-drive neutral scrimmage-play clock intervals, quarter<=3, score_diff<=8, 8..60 seconds",
        "pace_source": str(args.pace_snapshot) if args.pace_snapshot else "fresh nflreadpy PBP",
        "v02_features_sha256": sha256(args.features),
        "v02_weekly_sha256": sha256(args.weekly),
        "base_labels_sha256": sha256(args.labels),
        "v02_manifest_sha256": sha256(args.v02_manifest),
        "weekly_pace_sha256": sha256(pace_file),
        "v03_features_sha256": sha256(features_file),
        "v03_ready_rows": len(ready),
        "missing_neutral_pace_features_ready": pace_missing.to_dict(),
        "note": "Existing v0.1/v0.2 outputs are unchanged; missing pace imputed from training data only.",
    }
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote v0.3 features: {features_file}")
    print(f"Wrote raw pace audit snapshot: {pace_file}")
    print(f"Wrote audit: {audit_file}")
    print("This is a feature experiment, not a betting model or ROI test.")


if __name__ == "__main__":
    main()
