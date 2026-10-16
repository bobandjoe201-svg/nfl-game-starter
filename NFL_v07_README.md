# NFL Game Engine v0.7 — Prospective Odds Tracker

**Status:** Research/paper tracking only. No wagers are placed. No edge, calibrated probabilities, or positive expected ROI has been established by prior versions.

This new script leaves `nfl_game_model_v01.py` through `nfl_game_model_v06.py` untouched. It uses the Python standard library; your existing `.venv` is sufficient. All new data goes under `output/tracker_v07/` by default.

## The important distinction

- **Model prediction time** comes from the original v0.1 forecast CSV; importing it later does *not* change the original prediction timestamp.
- **Odds capture time** is the actual UTC time a new API response was received, or the time a manual/offline CSV/JSON was imported. Importing an old screenshot/JSON now **does not** magically make it a contemporaneous quote.
- **Market update time**, when provided by the source, is kept separately.
- The tracker refuses to capture already-started games and refuses retrospective paper-pick creation after kickoff. It does not claim any captured price was personally executable at a specific account or stake.
- **Sportsbook close** is not assumed. The report only gives an approximate *last captured pregame line* if a matching bookmaker/side was snapped within 90 minutes of kickoff; this is not a certified closing line and excludes changing American odds prices.
- Observing a forecast difference from an offered line does **not** establish a positive expected value. A separate calibrated probability and validation project is needed.

## Step 1 — Add the new file and initialize

Put `nfl_game_tracker_v07.py` alongside your existing v0.1–v0.6 scripts in `C:\Scripts\GitHub\nfl-game-starter`.

```powershell
python nfl_game_tracker_v07.py init
```

## Step 2 — Freeze your existing 2026 Week 6 v0.1 forecast

This reads an already-created forecast CSV. It does not retrain or regenerate models, even if some games have already started. **This is deliberate:** original generated timestamps must be preserved.

If the folder contains only one original v0.1 forecast:

```powershell
python nfl_game_tracker_v07.py freeze --predictions-dir output/model_v01
```

If it contains multiple, select the actual prior-generated file explicitly, using PowerShell to find filenames:

```powershell
Get-ChildItem output/model_v01/week*_2026_predictions_*.csv
python nfl_game_tracker_v07.py freeze --predictions "output/model_v01/YOUR_ACTUAL_FILENAME.csv"
```

**Freeze before quotes whenever possible.** The script associates games by normalized teams and Toronto-local kickoff date, and only associates odds with predictions created before the quote timestamp. If a game cannot be matched, its quote is preserved but cannot be used to record a paper pick yet.

## Step 3 — Capture real NFL bookmaker odds

### Option A: The Odds API, Canadian books (recommended for automation)

The specific provider is **https://the-odds-api.com/** (notice the hyphens). Its v4 endpoint supports `americanfootball_nfl`, `spreads`, `totals`, `oddsFormat=american`, and region `ca`. Check account eligibility and actual sportsbook availability: Canada/Ontario bookmaker coverage does not guarantee the exact line or stake is available to you.

Obtain **your own** key from the provider. Do not put it in the source code, in a CSV, or in ChatGPT. In the current PowerShell terminal:

```powershell
$env:THE_ODDS_API_KEY = Read-Host 'The Odds API key'
python nfl_game_tracker_v07.py capture --source api --region ca
```

This takes **one provider request** for both spreads and totals in Canada; per the provider's API documentation, requesting two markets in one region usually consumes **two quota credits**. Free-plan coverage, bookmakers, and quotas may change; check the provider's account dashboard before relying on it. The script prints the returned quota headers without displaying your API key.

You may restrict to specific supported provider bookmaker IDs:

```powershell
python nfl_game_tracker_v07.py capture --source api --region ca --bookmakers betmgm_ca_on,betano_ca_on
```

To collect prices at several moments in the week, manually repeat `capture`; every call writes a new immutable snapshot. No automatic monitoring or scheduler is installed.

### Option B: Manually entered offers (no API subscription)

Copy `nfl_v07_manual_odds_template.csv` to a *different* CSV and fill in the values **at the time you observe them**. `home_team` and `away_team` accept common NFL short codes. `kickoff_at_utc` needs a UTC time with `Z` (not a date only). The signed spread `line` is from the chosen team side: home -3.5 or away +3.5. For totals, use market `total`, selection `over` or `under`, and the posted total line (e.g. 44.5). Prices must be American odds like `-110` or `+100`; do not use `1.91` decimal odds.

```powershell
python nfl_game_tracker_v07.py capture --source csv --input my_observed_odds.csv
```

Manual odds are classified as *self-reported and unverified*. Never backfill an old quote and interpret its import time as when it was available.

### Option C: Offline The Odds API response JSON

```powershell
python nfl_game_tracker_v07.py capture --source json --input existing_api_response.json
```

Useful for parser testing, but **not a verified live snapshot**. Import time is recorded accurately.

## Step 4 — Inspect prices versus the frozen forecast

```powershell
python nfl_game_tracker_v07.py report
python nfl_game_tracker_v07.py quotes --limit 30
python nfl_game_tracker_v07.py export --out output/tracker_v07/research_export_001.csv
```

The `GAP` column reports raw point disagreement (home margin from model minus home margin implied by spread, or model total minus book total). It is **not** expected value, a win probability, or a bet recommendation. Every captured side and bookmaker is exported, not just picks.

## Step 5 — Optional pregame paper-only observation

You manually choose an **actual captured quote ID** and a **predeclared strategy tag** before kickoff. This is optional, and recording no paper selections is perfectly valid:

```powershell
python nfl_game_tracker_v07.py paper --quote-id q_ACTUAL_QUOTE_ID --strategy-tag exploratory-gap --stake-units 1
```

By default, the selected quote must have been captured within the preceding **120 minutes** and the original prediction must predate the snapshot. Paper units are hypothetical. The program never executes bets, generates automated selections, or claims that any threshold is profitable.

## Step 6 — After games: settle paper records

Regenerate the base schedule/labels *after* results become available (using your existing normal data workflow). Pass the updated labels file:

```powershell
python nfl_game_tracker_v07.py grade --labels output/historical_labels.csv
python nfl_game_tracker_v07.py report
```

Grades are computed from the **specific selected side's quoted line and odds**, accounting for pushes. The score source itself is not automatically verified by v0.7. Corrected results are appended as new grade rows; previous records remain available in SQLite. Sample paper ROI is descriptive, not evidence of sustainable expected returns.

## Files and safeguards

`output/tracker_v07/tracker.sqlite` stores frozen forecasts, bookmaker/selection prices, optional paper observations, and settlement history. `output/tracker_v07/raw/` stores all raw snapshot imports (API responses are saved without the key). The tables are append-only through SQLite triggers; using SQL to manually alter the database defeats the intended audit.

**Back up this directory.** The project's existing `.gitignore` excludes `output/`, so Git will generally **not** back up your new tracker data.

There is no live odds API call, automated wager, payment, or sportsbook account connection until *you* choose to run an explicit `capture` command with your own key.

## Research limitations before a real filter study

- A historical winning filter chosen after seeing the outcomes is heavily vulnerable to multiple-testing and look-ahead bias.
- Markets differ across bookmakers and between bet decision time and close; provider odds are not guaranteed executable at your location, account, or desired stake.
- Forecast CSV timestamps show when the script generated results, but do not independently certify that its underlying features were built strictly pre-kickoff. Preserve input hashes and feature build times.
- We still need calibrated cover/over probabilities, validation by chronological unseen weeks, realistic prices, and appropriately uncertainty-adjusted estimates before claiming positive EV.
- With roughly 3–4 paper selections per NFL week, robust statistical validation will take many games and should account for correlation within weeks.

## CLI reference

```powershell
python nfl_game_tracker_v07.py --help
python nfl_game_tracker_v07.py capture --help
python nfl_game_tracker_v07.py quotes --game-id 2026_06_SEA_DEN
```
