"""NFL Game Engine v0.4 — quarterback performance *proxy* experiment.

NEW file. Does not modify any previous script or dataset.

Run from the project root after v0.3:
    python nfl_game_features_v04.py

Optional exact replay of the previously archived PBP-derived QB snapshot:
    python nfl_game_features_v04.py --qb-snapshot output/v04/weekly_qb_stats_v04.csv --outdir output/v04_replay

Reads:
    output/v03/pregame_features_v03.csv
    output/v03/feature_manifest_v03.json
    output/historical_labels.csv

Writes, once, to output/v04/:
    pregame_features_v04.csv
    weekly_qb_stats_v04.csv
    qb_feature_audit_v04.csv
    feature_manifest_v04.json

METHOD / LIMITATION
-------------------
For each team at a target game, identify its most recent prior game's
PRIMARY PASSER (highest QB dropback count). This is a deterministic,
point-in-time-compatible PROXY for the expected starting QB, NOT proof of
who was announced to start that game. It may be wrong following injuries,
benchings, bye weeks, trades, or QB changes.

For that identified QB, aggregate only THEIR prior appearances in the same
season, strictly BEFORE the target week, across up to four QB games:
    EPA per dropback, pass-attempt-weighted CPOE, sack rate, dropbacks.
Also flag whether that primary-passer identity differs from the team's
preceding game's primary passer. That is an observable past switch, NOT a
confirmed future starter change.

No current target-game passer data, final scores, or betting odds are used
as model predictors. All missing QB performance remains NaN for train-only
imputation by the existing model pipeline. The last completed-week cutoff
from v0.3 prevents creating distant-future predictions in unfinished seasons.

Real confirmed starter modeling requires time-stamped *pregame* starting-QB
snapshots and should be a separate experiment later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "nfl-game-features-v0.4-qb-proxy"
KEY = ["season", "week", "game_id"]
QB_KEY = KEY + ["team", "qb_id"]
OUT_COLS = [
    "home_qb_epa_dropback", "away_qb_epa_dropback",
    "home_qb_cpoe", "away_qb_cpoe",
    "home_qb_sack_rate", "away_qb_sack_rate",
    "home_qb_prior_dropbacks", "away_qb_prior_dropbacks",
    "home_qb_recent_switch", "away_qb_recent_switch",
]
AUDIT_COLS = ["home_qb_proxy_id", "away_qb_proxy_id",
              "home_qb_proxy_source_week", "away_qb_proxy_source_week"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_base(features_path: Path, labels_path: Path, manifest_path: Path):
    for path in (features_path, labels_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite file: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != "nfl-game-features-v0.3-pace-volume":
        raise ValueError("Expected the original v0.3 feature manifest")
    for field, path in (("v03_features_sha256", features_path),
                        ("base_labels_sha256", labels_path)):
        if sha256(path) != manifest.get(field):
            raise ValueError(f"Input snapshot hash mismatch: {path}")
    base = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    needed_base = set(KEY + ["home_team", "away_team", "v03_features_ready",
                             "home_prior_games", "away_prior_games"])
    needed_labels = set(KEY + ["home_score", "away_score"])
    if not needed_base.issubset(base.columns):
        raise ValueError(f"Missing v0.3 feature columns: {sorted(needed_base - set(base.columns))}")
    if not needed_labels.issubset(labels.columns):
        raise ValueError(f"Missing labels columns: {sorted(needed_labels - set(labels.columns))}")
    if base.duplicated(KEY).any() or labels.duplicated(KEY).any():
        raise ValueError("Duplicate game key in features or labels")
    if not base[KEY].merge(labels[KEY], on=KEY, how="left", indicator=True)["_merge"].eq("both").all():
        raise ValueError("Some feature rows have no labels")
    flag = base["v03_features_ready"].astype(str).str.lower()
    if not flag.isin(["true", "false"]).all():
        raise ValueError("Unrecognized v03_features_ready values")
    base["v03_features_ready"] = flag.eq("true")
    return base, labels


def weekly_qb_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """One per QB per team-game, built ONLY from observed *past* PBP.

    QB dropback includes pass plays and QB sacks when nflverse marks them
    as qb_dropback=1. CPOE is aggregated only where a pass attempt and a
    nonmissing CPOE value are recorded. All QB IDs are nflverse IDs.
    """
    required = set(KEY + ["posteam", "passer_player_id", "qb_dropback", "epa",
                          "pass_attempt", "sack", "cpoe"])
    if not required.issubset(pbp.columns):
        raise ValueError(f"Missing QB PBP fields: {sorted(required - set(pbp.columns))}")
    p = pbp
    if "season_type" in p.columns:
        p = p.loc[p["season_type"].eq("REG")]
    p = p.loc[p["posteam"].notna() & p["passer_player_id"].notna()].copy()
    for col in ("qb_dropback", "epa", "pass_attempt", "sack", "cpoe"):
        p[col] = pd.to_numeric(p[col], errors="coerce")
    p = p.loc[p["qb_dropback"].eq(1) & p["epa"].notna()].copy()
    if p.empty:
        raise ValueError("No identifiable QB dropbacks in PBP")
    p["is_att"] = p["pass_attempt"].eq(1).astype(float)
    p["sack_clean"] = p["sack"].fillna(0).clip(0, 1)
    p["cpoe_valid"] = p["cpoe"].where(p["is_att"].eq(1))
    p["cpoe_sum"] = p["cpoe_valid"].fillna(0)
    p["cpoe_n"] = p["cpoe_valid"].notna().astype(int)
    weekly = (p.groupby(KEY + ["posteam", "passer_player_id"], as_index=False)
              .agg(dropbacks=("epa", "size"), epa_sum=("epa", "sum"),
                   sacks=("sack_clean", "sum"),
                   attempts=("is_att", "sum"),
                   cpoe_sum=("cpoe_sum", "sum"), cpoe_n=("cpoe_n", "sum"))
              .rename(columns={"posteam": "team", "passer_player_id": "qb_id"}))
    weekly["qb_id"] = weekly["qb_id"].astype(str)
    if weekly.duplicated(QB_KEY).any():
        raise ValueError("Duplicate weekly QB key")
    return weekly.sort_values(QB_KEY).reset_index(drop=True)


def add_qb_features(base: pd.DataFrame, weekly: pd.DataFrame,
                    lookback: int = 4) -> pd.DataFrame:
    """Target-week features from previous-week team primary passer only."""
    if lookback != 4:
        raise ValueError("For controlled comparison lookback must be four QB appearances")
    required = set(QB_KEY + ["dropbacks", "epa_sum", "sacks", "cpoe_sum", "cpoe_n"])
    if not required.issubset(weekly.columns):
        raise ValueError(f"Missing weekly QB fields: {sorted(required-set(weekly.columns))}")
    if weekly.duplicated(QB_KEY).any():
        raise ValueError("Duplicate QB-team-game keys")
    qb = weekly.copy()
    qb["qb_id"] = qb["qb_id"].astype(str)
    for col in ("dropbacks", "epa_sum", "sacks", "cpoe_sum", "cpoe_n"):
        qb[col] = pd.to_numeric(qb[col], errors="raise")
    if qb["dropbacks"].le(0).any():
        raise ValueError("QB record with no dropbacks")

    # Identify historical PRIMARY PASSER *per team-game* using past performance.
    # Deterministic tie break on QB ID; do NOT use same-week QB as predictor.
    primaries = (qb.sort_values(["season", "week", "game_id", "team", "dropbacks", "qb_id"],
                                ascending=[True, True, True, True, False, True])
                 .drop_duplicates(KEY + ["team"])
                 .sort_values(["season", "team", "week", "game_id"]))
    team_hists = {(int(s), str(t)): grp.copy() for (s, t), grp in
                  primaries.groupby(["season", "team"])}
    # Same QB may have moved teams; stats for that QB across teams only from
    # the SAME season. Never look forward into the target week.
    qb_hists = {(int(s), str(q)): grp.sort_values(["week", "game_id"]).copy()
                for (s, q), grp in qb.groupby(["season", "qb_id"])}

    out = base.copy()
    out["v04_features_ready"] = out["v03_features_ready"].copy()
    for col in OUT_COLS:
        out[col] = np.nan
    for col in AUDIT_COLS:
        out[col] = np.nan if col.endswith("week") else ""
    cache = {}

    def for_team(season, week, team):
        k = (season, week, team)
        if k in cache:
            return cache[k]
        hist = team_hists.get((season, team))
        past = hist.loc[hist["week"] < week].tail(2) if hist is not None else pd.DataFrame()
        if past.empty:
            ans = (np.nan, np.nan, np.nan, np.nan, np.nan, "", np.nan)
        else:
            most_recent = past.iloc[-1]
            selected_id = str(most_recent.qb_id)
            switched = float(len(past) > 1 and str(past.iloc[-2].qb_id) != selected_id)
            qb_history = qb_hists.get((season, selected_id))
            prior = qb_history.loc[qb_history["week"] < week].tail(lookback) if qb_history is not None else pd.DataFrame()
            dropbacks = float(prior["dropbacks"].sum())
            attempts_with_cpoe = float(prior["cpoe_n"].sum())
            ans = (
                float(prior["epa_sum"].sum() / dropbacks) if dropbacks else np.nan,
                float(prior["cpoe_sum"].sum() / attempts_with_cpoe) if attempts_with_cpoe else np.nan,
                float(prior["sacks"].sum() / dropbacks) if dropbacks else np.nan,
                dropbacks,
                switched,
                selected_id,
                float(most_recent.week),
            )
        cache[k] = ans
        return ans

    for idx, row in out.iterrows():
        if not bool(row["v04_features_ready"]):
            continue
        year, week = int(row["season"]), int(row["week"])
        home = for_team(year, week, str(row["home_team"]))
        away = for_team(year, week, str(row["away_team"]))
        out.loc[idx, OUT_COLS] = [home[0], away[0], home[1], away[1],
                                  home[2], away[2], home[3], away[3],
                                  home[4], away[4]]
        out.loc[idx, "home_qb_proxy_id"] = home[5]
        out.loc[idx, "away_qb_proxy_id"] = away[5]
        out.loc[idx, "home_qb_proxy_source_week"] = home[6]
        out.loc[idx, "away_qb_proxy_source_week"] = away[6]
        # In a ready week 2+ there must have been a prior team QB.
        if not home[5] or not away[5]:
            raise ValueError(f"No prior primary passer for ready game {row['game_id']}")
        if home[6] >= week or away[6] >= week:
            raise ValueError(f"FUTURE LEAKAGE: QB proxy source week not strictly prior: {row['game_id']}")
    if not (out["v04_features_ready"].values == base["v03_features_ready"].values).all():
        raise ValueError("Controlled sample changed")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=Path("output/v03/pregame_features_v03.csv"))
    parser.add_argument("--labels", type=Path, default=Path("output/historical_labels.csv"))
    parser.add_argument("--v03-manifest", type=Path, default=Path("output/v03/feature_manifest_v03.json"))
    parser.add_argument("--qb-snapshot", type=Path, default=None,
                        help="Archived weekly_qb_stats_v04.csv to avoid a new PBP download")
    parser.add_argument("--outdir", type=Path, default=Path("output/v04"))
    args = parser.parse_args()
    if any(args.outdir.resolve() == p.resolve().parent for p in (args.features, args.labels, args.v03_manifest)):
        parser.error("Choose separate v0.4 output directory; older versions are protected")
    base, labels = read_base(args.features, args.labels, args.v03_manifest)
    if args.qb_snapshot:
        qb_weekly = pd.read_csv(args.qb_snapshot, dtype={"qb_id": str})
        print(f"Loaded archived QB data: {args.qb_snapshot}")
    else:
        import nflreadpy as nfl
        years = sorted(base["season"].astype(int).unique().tolist())
        print(f"Downloading PBP for QB stats, seasons {years} ...", flush=True)
        pbp = nfl.load_pbp(years).to_pandas()
        qb_weekly = weekly_qb_from_pbp(pbp)
        print(f"PBP rows: {len(pbp):,}; QB-game observations: {len(qb_weekly):,}", flush=True)
    # Validate exact game set: nothing beyond v0.3 labels, including PBP
    # mistakenly classified as an extra season/game.
    allowed = set(map(tuple, labels[KEY].itertuples(index=False, name=None)))
    present = set(map(tuple, qb_weekly[KEY].itertuples(index=False, name=None)))
    if not present.issubset(allowed):
        raise ValueError("QB snapshot includes a game outside historical labels")
    out = add_qb_features(base, qb_weekly)
    ready = out.loc[out.v04_features_ready]
    if ready.empty:
        raise ValueError("No ready records")
    print(f"Ready games: {len(ready):,}; missing QB CPOE: "
          f"{ready[['home_qb_cpoe','away_qb_cpoe']].isna().sum().to_dict()}")
    paths = {
        "features": args.outdir / "pregame_features_v04.csv",
        "weekly": args.outdir / "weekly_qb_stats_v04.csv",
        "audit": args.outdir / "qb_feature_audit_v04.csv",
        "manifest": args.outdir / "feature_manifest_v04.json",
    }
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("v0.4 outputs already exist. Use --outdir for a new run, never overwrite existing snapshots.")
    args.outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths["features"], index=False)
    qb_weekly.to_csv(paths["weekly"], index=False)
    ready[KEY + ["home_team", "away_team"] + AUDIT_COLS + OUT_COLS].to_csv(paths["audit"], index=False)
    manifest = {
        "version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "assumed_starter": "Most recent prior team game primary passer; NOT an announced expected starter",
        "qb_stat_history": "Prior games same QB same season, up to 4 most recent, week strictly before target",
        "v03_features_sha256": sha256(args.features),
        "v03_manifest_sha256": sha256(args.v03_manifest),
        "base_labels_sha256": sha256(args.labels),
        "qb_weekly_sha256": sha256(paths["weekly"]),
        "v04_features_sha256": sha256(paths["features"]),
        "ready_rows": len(ready),
        "missing_qb_cpoe_ready": ready[["home_qb_cpoe", "away_qb_cpoe"]].isna().sum().to_dict(),
        "no_ROI_claims": True,
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for label, path in paths.items():
        print(f"Saved {label}: {path}")
    print("Research features only. The QB ID is a historical proxy, NOT a verified starting lineup.")


if __name__ == "__main__":
    main()
