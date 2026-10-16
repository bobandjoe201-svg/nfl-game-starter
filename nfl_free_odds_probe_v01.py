"""Read-only ESPN NFL odds coverage probe (unofficial public endpoints).

Does NOT modify the NFL tracker or place bets. No API key required.
Usage: python nfl_free_odds_probe_v01.py --season 2026 --week 6 --max-games 3
The URLs are unofficial and may change or stop working at any time.
"""
import argparse
import json
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
CORE = "https://sports.core.api.espn.com"


def get_json(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 NFL-Odds-Coverage-Check/0.1", "Accept": "application/json"})
    with urlopen(req, timeout=12) as reply:
        return json.load(reply)


def get_item(row):
    if not isinstance(row, dict):
        return {}
    if "overUnder" in row or "details" in row or "homeTeamOdds" in row:
        return row
    link = row.get("$ref", "")
    if link and urlparse(link).hostname in {"sports.core.api.espn.com", "site.api.espn.com"}:
        try:
            return get_json(link.replace("http://", "https://", 1))
        except (HTTPError, URLError, ValueError, TimeoutError) as exc:
            print(f"     Referenced odds unavailable: {exc}")
    return row


def formatted(value):
    return "MISSING" if value is None or value == "" else str(value)


def teams(event):
    competitors = (event.get("competitions") or [{}])[0].get("competitors") or []
    by_home = {x.get("homeAway"): x.get("team", {}).get("abbreviation", "?") for x in competitors}
    return by_home.get("away", "?"), by_home.get("home", "?")


def main():
    ap = argparse.ArgumentParser(description="Read-only ESPN NFL odds coverage test; no sign-up, no tracker changes")
    ap.add_argument("--season", type=int, default=datetime.now(timezone.utc).year)
    ap.add_argument("--week", type=int, default=6)
    ap.add_argument("--max-games", type=int, default=3, help="Test at most this many upcoming games")
    args = ap.parse_args()
    if not 1 <= args.max_games <= 18:
        ap.error("--max-games must be 1 through 18")

    url = SCOREBOARD + "?" + urlencode({"season": args.season, "seasontype": 2, "week": args.week, "limit": 30})
    print(f"ESPN free odds COVERAGE PROBE | NFL {args.season} week {args.week}")
    print("No API key. Read-only. ESPN is not a verified Ontario-executable betting quote.")
    try:
        feed = get_json(url)
    except (HTTPError, URLError, ValueError, TimeoutError) as exc:
        print(f"ERROR fetching scoreboard: {exc}")
        print("ESPN may be inaccessible or its undocumented endpoint may have changed.")
        return 1
    now = datetime.now(timezone.utc)
    upcoming = []
    for event in feed.get("events") or []:
        date = event.get("date", "")
        try:
            kickoff = datetime.fromisoformat(date.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if kickoff <= now or event.get("status", {}).get("type", {}).get("completed"):
            continue
        upcoming.append((kickoff, event))
    upcoming.sort(key=lambda item: item[0])
    print(f"Scoreboard: {len(feed.get('events') or [])} events; {len(upcoming)} not yet started.")
    if not upcoming:
        print("No future games in this week; choose a future week, or inspect the scoreboard source.")
        return 0
    complete = 0
    with_odds = 0
    checked = 0
    for kickoff, event in upcoming[:args.max_games]:
        gameid = event.get("id", "")
        away, home = teams(event)
        print(f"\n{away} @ {home} | {kickoff.isoformat()} | ESPN event {gameid}")
        if not gameid:
            print("  No event ID, cannot request odds")
            continue
        try:
            data = get_json(f"{CORE}/v2/sports/football/leagues/nfl/events/{gameid}/competitions/{gameid}/odds?limit=20")
        except (HTTPError, URLError, ValueError, TimeoutError) as exc:
            print(f"  Odds endpoint failed: {exc}")
            continue
        items = data.get("items", []) if isinstance(data, dict) else []
        print(f"  Odds entries: {len(items)}")
        checked += 1
        if items:
            with_odds += 1
        valid = False
        for raw in items[:8]:
            offer = get_item(raw)
            source = offer.get("provider") or {}
            if isinstance(source, dict):
                source = source.get("name") or source.get("id") or "Unknown"
            hs = offer.get("homeTeamOdds") or {}
            a = offer.get("awayTeamOdds") or {}
            fields = (offer.get("overUnder"), hs.get("spreadOdds"), a.get("spreadOdds"), offer.get("overOdds"), offer.get("underOdds"))
            if all(v is not None for v in fields) and all(isinstance(v, (int, float)) and (v <= -100 or v >= 100) for v in fields[1:]):
                valid = True
            print(f"  [{source}] {formatted(offer.get('details'))} | total {formatted(offer.get('overUnder'))}")
            print(f"     spread odds: home {formatted(hs.get('spreadOdds'))}, away {formatted(a.get('spreadOdds'))}; total odds: over {formatted(offer.get('overOdds'))}, under {formatted(offer.get('underOdds'))}")
        complete += int(valid)
        if not items:
            print("  No posted odds. Not proof the sportsbook has no prices.")
    print(f"\nSUMMARY: tested {checked} odds endpoints; {with_odds} returned odds; {complete} had a row with both spread-side and total-side American prices.")
    print("These are coverage checks only. Quote timestamps, freshness and Ontario availability are NOT verified.")
    print("Do not import these responses into v0.7 as if they were executable wagers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
