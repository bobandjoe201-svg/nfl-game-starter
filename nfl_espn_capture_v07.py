"""Save unofficial ESPN-reported NFL spreads/totals to the existing v0.7 audit tracker.

No paid account, no ESPN authentication, no betting, no modifications to v0.1-v0.7.
This is a RESEARCH odds source, NOT a verified Ontario-executable DraftKings quote.
ESPN's undocumented endpoints can change or cease working without warning.

Preview without changing the tracker:
  python nfl_espn_capture_v07.py --season 2026 --week 6

Explicitly capture a new append-only research snapshot:
  python nfl_espn_capture_v07.py --season 2026 --week 6 --capture
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import nfl_game_tracker_v07 as tracker

SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard'
CORE = 'https://sports.core.api.espn.com'
ALLOWED_HOSTS = {'site.api.espn.com', 'sports.core.api.espn.com'}
# ESPN short codes are not identical to nflverse codes in the tracker.
ESPN_CODES = {'LAR': 'LA', 'WSH': 'WAS', 'JAC': 'JAX'}
USER_AGENT = 'NFL-Research-ESPN-Odds-Capture/0.1'


def fetch_json(url):
    parts = urlparse(url)
    if parts.scheme != 'https' or parts.hostname not in ALLOWED_HOSTS:
        raise ValueError('Refusing an untrusted ESPN reference URL')
    req = Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
    with urlopen(req, timeout=14) as resp:
        body = resp.read(5_000_001)
        if len(body) > 5_000_000:
            raise ValueError('ESPN response exceeds 5 MB safety limit')
    return json.loads(body)


def unwrap_offer(item):
    if not isinstance(item, dict):
        raise ValueError('Unexpected odds item type')
    if any(k in item for k in ('homeTeamOdds', 'spread', 'overUnder')):
        return item
    ref = item.get('$ref')
    if not ref or not isinstance(ref, str):
        raise ValueError('Odds item has no usable fields or reference')
    # ESPN occasionally returns http $ref links to an HTTPS-capable host.
    ref = ref.replace('http://', 'https://', 1)
    return fetch_json(ref)


def espn_code(abbr):
    code = ESPN_CODES.get(str(abbr).upper().strip(), str(abbr).upper().strip())
    return tracker.team_code(code)


def event_teams(event):
    competitions = event.get('competitions') or []
    if len(competitions) != 1:
        raise ValueError('Expected exactly one NFL competition')
    comps = competitions[0].get('competitors') or []
    sides = {}
    for c in comps:
        side = c.get('homeAway')
        if side in ('home', 'away'):
            sides[side] = espn_code((c.get('team') or {}).get('abbreviation'))
    if not sides.get('home') or not sides.get('away') or sides['home'] == sides['away']:
        raise ValueError('Unrecognized ESPN home/away team')
    return sides['home'], sides['away']


def number(x, *, min_val, max_val):
    if isinstance(x, bool):
        raise ValueError('Boolean is not a valid numeric price or line')
    value = float(x)
    if not math.isfinite(value) or not min_val <= value <= max_val:
        raise ValueError(f'Out-of-range number {x!r}')
    return value


def parse_offer(offer, event_id, kickoff, home, away, conn=None):
    if not isinstance(offer, dict):
        raise ValueError('Non-object sportsbook odds entry')
    provider = offer.get('provider') or {}
    if isinstance(provider, dict):
        provider_name = str(provider.get('name') or provider.get('id') or '').strip()
    else:
        provider_name = str(provider).strip()
    if not provider_name:
        raise ValueError('Missing sportsbook identity')
    provider_key = re.sub(r'[^a-z0-9]+', '_', provider_name.casefold()).strip('_')
    if not provider_key:
        raise ValueError('Invalid sportsbook identity')

    # ESPN's `spread` field is the HOME team's handicap, not the favorite's
    # generic sign from the `details` label. For example, SEA -2.5 at DEN
    # means `spread` is +2.5 (Denver), and SEA's handicap is -2.5.
    home_handicap = number(offer.get('spread'), min_val=-60, max_val=60)
    if not float(home_handicap * 2).is_integer():
        raise ValueError('Unsupported spread increment (expected half points)')
    total = number(offer.get('overUnder'), min_val=10, max_val=100)
    if not float(total * 2).is_integer():
        raise ValueError('Unsupported total increment (expected half points)')
    h = offer.get('homeTeamOdds') or {}
    a = offer.get('awayTeamOdds') or {}
    home_price = tracker.numeric_price(h.get('spreadOdds'))
    away_price = tracker.numeric_price(a.get('spreadOdds'))
    over_price = tracker.numeric_price(offer.get('overOdds'))
    under_price = tracker.numeric_price(offer.get('underOdds'))

    if home_handicap != 0:
        # Guard against a reversed signed spread without trusting `details`.
        if isinstance(h.get('favorite'), bool) and h['favorite'] != (home_handicap < 0):
            raise ValueError('ESPN home spread/favorite sign conflict')
        if isinstance(a.get('favorite'), bool) and a['favorite'] != (home_handicap > 0):
            raise ValueError('ESPN away spread/favorite sign conflict')

    # ESPN references do not assert available Ontario bet offers or source
    # update times. No fabricated market_updated_at timestamp is inserted.
    base = dict(event_id=str(event_id), kickoff_at=tracker.iso(kickoff),
                home_team=home, away_team=away,
                game_id=tracker.prediction_matches(conn, home, away, kickoff) if conn else None,
                bookmaker_key='espn_reported_' + provider_key,
                bookmaker_title='ESPN-reported ' + provider_name + ' (research; Ontario unverified)',
                market_updated_at=None)
    return [
        dict(base, market='spread', selection='home', line=home_handicap, american_price=home_price),
        dict(base, market='spread', selection='away', line=-home_handicap, american_price=away_price),
        dict(base, market='total', selection='over', line=total, american_price=over_price),
        dict(base, market='total', selection='under', line=total, american_price=under_price),
    ]


def collect(season, week, max_games, conn=None):
    run_started = tracker.now_utc()
    scoreboard_url = SCOREBOARD + '?' + urlencode({'season': season, 'seasontype': 2, 'week': week, 'limit': 30})
    scoreboard = fetch_json(scoreboard_url)
    events = scoreboard.get('events') or []
    candidate = []
    skipped = 0
    for ev in events:
        try:
            kickoff = tracker.utc_parse(ev['date'])
            status = (ev.get('status') or {}).get('type') or {}
            if kickoff <= run_started or status.get('completed') or status.get('state') in ('in', 'post'):
                skipped += 1
                continue
            home, away = event_teams(ev)
            event_id = str(ev['id'])
            if not event_id.isdigit():
                raise ValueError('Invalid ESPN event ID')
            candidate.append((kickoff, event_id, home, away))
        except (KeyError, ValueError, TypeError) as exc:
            print(f'  SKIP invalid ESPN event: {exc}')
            skipped += 1
    candidate.sort()
    if len(candidate) > max_games:
        print(f'  WARNING: only {max_games} of {len(candidate)} pregame events will be checked.')
    candidate = candidate[:max_games]

    raw = {'feed': 'ESPN unofficial public NFL API',
           'research_only': True,
           'ontario_executable_verified': False,
           'source_market_updated_at_verified': False,
           'retrieval_started_at_utc': tracker.iso(run_started),
           'scoreboard_url': scoreboard_url, 'scoreboard': scoreboard,
           'event_odds': []}
    rows = []
    errors = 0
    for kickoff, event_id, home, away in candidate:
        odds_url = f'{CORE}/v2/sports/football/leagues/nfl/events/{event_id}/competitions/{event_id}/odds?limit=20'
        record = {'event_id': event_id, 'home_team': home, 'away_team': away,
                  'kickoff_at_utc': tracker.iso(kickoff), 'source_url': odds_url}
        try:
            payload = fetch_json(odds_url)
            record['raw_odds_response'] = payload
            for item in payload.get('items') or []:
                try:
                    offer = unwrap_offer(item)
                    record.setdefault('resolved_offers', []).append(offer)
                    offers = parse_offer(offer, event_id, kickoff, home, away, conn)
                    rows.extend(offers)
                except (ValueError, TypeError, KeyError, HTTPError, URLError, TimeoutError) as exc:
                    errors += 1
                    record.setdefault('parse_errors', []).append(str(exc))
            record['retrieved_at_utc'] = tracker.iso(tracker.now_utc())
        except (ValueError, TypeError, KeyError, HTTPError, URLError, TimeoutError) as exc:
            errors += 1
            record['retrieval_error'] = str(exc)
            print(f'  WARN {away} @ {home}: odds fetch failed: {type(exc).__name__}')
        raw['event_odds'].append(record)

    observed_at = tracker.now_utc()  # conservative: after all HTTP requests
    raw['retrieval_completed_at_utc'] = tracker.iso(observed_at)
    pregame = [r for r in rows if tracker.utc_parse(r['kickoff_at']) > observed_at]
    skipped_after_fetch = len(rows) - len(pregame)
    return pregame, raw, observed_at, errors, skipped, skipped_after_fetch, len(candidate)


def preview(rows, season, week, observed_at, errors, skipped, skip_after, checked):
    print(f'ESPN NFL FREE ODDS | {season} Week {week}')
    print('Research only. Unverified source freshness and Ontario bookmaker availability.')
    print(f'Observation completed: {tracker.iso(observed_at)} UTC')
    print('                         Home spread     Away spread        Over        Under')
    by_game = {}
    for row in rows:
        key = (row['event_id'], row['away_team'], row['home_team'], row['bookmaker_key'])
        by_game.setdefault(key, {})[(row['market'], row['selection'])] = row
    for (event_id, away, home, book), r in sorted(by_game.items()):
        if len(r) != 4:
            continue
        h, a = r['spread','home'], r['spread','away']
        ov, un = r['total','over'], r['total','under']
        print(f'{away:>3} @ {home:<3} ({event_id}) {h["line"]:+6.1f} {h["american_price"]:+5d}'
              f'   {a["line"]:+6.1f} {a["american_price"]:+5d}'
              f'   {ov["line"]:5.1f} {ov["american_price"]:+5d}'
              f'   {un["line"]:5.1f} {un["american_price"]:+5d}')
    print(f'SUMMARY: events checked={checked}; complete quotes={len(rows)}; '
          f'games with priced markets={len(set(r["event_id"] for r in rows))}; '
          f'errors={errors}; started/skipped={skipped}; crossed kickoff during fetch={skip_after}.')
    print('All quotes are ESPN-reported research prices, not certified live offers.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--week', type=int, required=True)
    ap.add_argument('--max-games', type=int, default=18)
    ap.add_argument('--db', type=Path, default=tracker.DEFAULT_DB)
    ap.add_argument('--capture', action='store_true', help='Explicitly append a research-only snapshot to v0.7')
    args = ap.parse_args()
    if not 2020 <= args.season <= 2100 or not 1 <= args.week <= 18 or not 1 <= args.max_games <= 18:
        ap.error('Season must be 2020–2100; week and max-games must be 1–18')
    conn = None
    try:
        if args.capture:
            conn = tracker.connect(args.db)
        elif args.db.exists():
            conn = sqlite3.connect(f'{args.db.resolve().as_uri()}?mode=ro', uri=True)
            conn.row_factory = sqlite3.Row
        rows, raw, observed_at, errors, skipped, skipped_after, checked = collect(
            args.season, args.week, args.max_games, conn)
        preview(rows, args.season, args.week, observed_at, errors, skipped, skipped_after, checked)
        if not rows:
            print('No complete pregame offers. No snapshot created.', file=sys.stderr)
            return 1
        if args.capture:
            tracker.save_snapshot(SimpleNamespace(db=args.db), conn, json.dumps(raw).encode('utf-8'),
                                  rows, 'espn_unofficial_research',
                                  f'ESPN reference prices NFL {args.season} Week {args.week}; '
                                  'NOT Ontario verified or quote-freshness certified',
                                  errors, skipped + skipped_after, observed_at)
            print('View with: python nfl_game_tracker_v07.py quotes --limit 60')
        else:
            print('PREVIEW ONLY: No tracker data changed. Add --capture to append a research snapshot.')
        return 0
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, sqlite3.Error,
            json.JSONDecodeError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    raise SystemExit(main())
