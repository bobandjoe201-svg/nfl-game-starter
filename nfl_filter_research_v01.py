"""NFL game markets: READ-ONLY, PRE-RESULTS filter/disagreement research v0.1.

Runs on a CSV exported by nfl_game_tracker_v07.py. Never accesses the
tracker database, never makes an HTTP request, and never places bets.

Example (from the existing project directory):
    python nfl_filter_research_v01.py --input output/tracker_v07/week6_first_comparison.csv

Exports a timestamped, non-overwriting research report under
    output/filter_research_v01/

All proposed bands (2, 4, 6 points) are EXPLORATORY, NOT learned or
validated against outcomes. NFL v0.1 has not established betting value.
ESPN-reported quotes have unverified freshness and Ontario availability.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

VERSION = 'nfl-filter-research-v0.1'
DEFAULT_INPUT = Path('output/tracker_v07/week6_first_comparison.csv')
DEFAULT_OUTDIR = Path('output/filter_research_v01')
THRESHOLDS = (2.0, 4.0, 6.0)
REQUIRED = {
    'quote_id', 'snapshot_id', 'source', 'captured_at', 'kickoff_at',
    'home_team', 'away_team', 'game_id', 'bookmaker_key', 'bookmaker_title',
    'market', 'selection', 'line', 'american_price',
    'model_version', 'pred_generated_at', 'pred_home_margin', 'pred_total',
    'model_gap_pts',
}
FOUR_SIDES = {('spread', 'home'), ('spread', 'away'),
              ('total', 'over'), ('total', 'under')}


def parse_utc(value: str, field: str) -> datetime:
    if not value or not value.strip():
        raise ValueError(f'Missing {field}')
    try:
        stamp = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError(f'Bad {field}: {value!r}') from exc
    if stamp.tzinfo is None:
        raise ValueError(f'{field} missing UTC offset: {value!r}')
    return stamp.astimezone(timezone.utc)


def finite_float(raw: str, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid {field}: {raw!r}') from exc
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite {field}: {raw!r}')
    return value


def american_price(raw: str) -> int:
    number = finite_float(raw, 'american_price')
    if not number.is_integer() or abs(number) < 100 or abs(number) > 100000:
        raise ValueError(f'Unsupported American price: {raw!r}')
    return int(number)


def implied_break_even(price: int) -> float:
    return 100 / (100 + price) if price > 0 else -price / (100 - price)


def almost_equal(a: float, b: float, eps: float = 0.0001) -> bool:
    return abs(a - b) <= eps


def load_csv(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Input missing required columns: {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError('CSV is empty')
    seen_ids = set()
    for line_num, row in enumerate(rows, start=2):
        if not row['quote_id'] or row['quote_id'] in seen_ids:
            raise ValueError(f'Blank/repeated quote_id at CSV row {line_num}: {row["quote_id"]!r}')
        seen_ids.add(row['quote_id'])
        if not all(row.get(k) for k in ('game_id', 'snapshot_id', 'bookmaker_key', 'home_team', 'away_team')):
            raise ValueError(f'Incomplete identity fields on CSV row {line_num}')
        row['_captured'] = parse_utc(row['captured_at'], 'captured_at')
        row['_kickoff'] = parse_utc(row['kickoff_at'], 'kickoff_at')
        row['_pred_time'] = parse_utc(row['pred_generated_at'], 'pred_generated_at')
        row['_line'] = finite_float(row['line'], 'line')
        row['_price'] = american_price(row['american_price'])
        row['_pred_home'] = finite_float(row['pred_home_margin'], 'pred_home_margin')
        row['_pred_total'] = finite_float(row['pred_total'], 'pred_total')
        row['_gap'] = finite_float(row['model_gap_pts'], 'model_gap_pts')
        if (row['market'], row['selection']) not in FOUR_SIDES:
            raise ValueError(f'Unsupported market/side at CSV row {line_num}')
        if row['_pred_time'] > row['_captured']:
            raise ValueError(f'Prediction after quote observed on row {line_num}')
        if row['_captured'] >= row['_kickoff']:
            raise ValueError(f'Quote was not observed before kickoff on row {line_num}')
        if row['_pred_time'] >= row['_kickoff']:
            raise ValueError(f'Forecast generated after kickoff on row {line_num}')
        if row['market'] == 'total' and row['_line'] <= 0:
            raise ValueError(f'Nonpositive total on row {line_num}')
    return rows


def select_latest_complete(rows: list[dict]) -> tuple[list[list[dict]], int]:
    """Latest complete observation for each GAME+BOOK, no mixing snapshots."""
    candidates = defaultdict(list)
    for row in rows:
        key = (row['game_id'], row['bookmaker_key'], row['snapshot_id'])
        candidates[key].append(row)
    by_game_book = defaultdict(list)
    for (game, book, snapshot), group in candidates.items():
        by_game_book[(game, book)].append((snapshot, group))
    selected = []
    older_groups = 0
    for game_book, options in sorted(by_game_book.items()):
        options.sort(key=lambda item: (max(r['_captured'] for r in item[1]), item[0]))
        snapshot, latest = options[-1]
        older_groups += len(options) - 1
        sides = [(r['market'], r['selection']) for r in latest]
        if set(sides) != FOUR_SIDES or len(sides) != 4:
            raise ValueError(f'Newest snapshot for {game_book} is not a full four-sided quote: '
                             f'{snapshot} {sides}. Refusing to silently fall back to an older snapshot.')
        selected.append(latest)
    return selected, older_groups


def canonicalize(group: list[dict]) -> list[dict]:
    by = {(r['market'], r['selection']): r for r in group}
    h, a, o, u = (by[('spread','home')], by[('spread','away')],
                   by[('total','over')], by[('total','under')])
    fields = ('snapshot_id','game_id','home_team','away_team','bookmaker_key',
              'bookmaker_title','captured_at','kickoff_at','model_version',
              'pred_generated_at','source')
    for field in fields:
        if len({r[field] for r in group}) != 1:
            raise ValueError(f'Conflicting {field} in {h["game_id"]} {h["snapshot_id"]}')
    for field in ('_pred_home','_pred_total'):
        if max(r[field] for r in group) - min(r[field] for r in group) > .00001:
            raise ValueError(f'Model forecast disagreement within one snapshot: {h["game_id"]}')
    if not almost_equal(h['_line'] + a['_line'], 0.0):
        raise ValueError(f'Spread home/away lines not complementary: {h["game_id"]}')
    if not almost_equal(o['_line'], u['_line']):
        raise ValueError(f'Over/under lines not equal: {h["game_id"]}')
    if not (-60 <= h['_line'] <= 60 and 10 <= o['_line'] <= 100):
        raise ValueError(f'Implausible spreads/totals: {h["game_id"]}')

    # For the HOME selection, the sportsbook-implied home margin is -spread.
    # A HOME -3.5 quote corresponds to sportsbook home margin +3.5.
    market_home_margin = -h['_line']
    gap_spread = h['_pred_home'] - market_home_margin
    gap_total = o['_pred_total'] - o['_line']
    for q, expected in ((h,gap_spread),(a,gap_spread),(o,gap_total),(u,gap_total)):
        if not almost_equal(q['_gap'], expected, .0011):
            raise ValueError(f'Export model_gap_pts does not agree with recomputed gap '
                             f'for {q["game_id"]}, {q["market"]}/{q["selection"]}: '
                             f'{q["_gap"]} vs {expected}')

    common = {
        'game_id':h['game_id'], 'matchup':f'{h["away_team"]} @ {h["home_team"]}',
        'home_team':h['home_team'], 'away_team':h['away_team'],
        'bookmaker':h['bookmaker_title'], 'bookmaker_key':h['bookmaker_key'],
        'source':h['source'], 'snapshot_id':h['snapshot_id'],
        'quote_observed_at_utc':h['captured_at'],
        'kickoff_at_utc':h['kickoff_at'],
        'model_version':h['model_version'],
        'forecast_generated_at_utc':h['pred_generated_at'],
        'forecast_age_hours_at_observation':round((h['_captured']-h['_pred_time']).total_seconds()/3600,2),
        'source_market_updated_at_known':bool(all(r.get('market_updated_at') for r in group)),
        'verified_ontario_offer':False,
        'home_spread':h['_line'], 'home_spread_price':h['_price'],
        'away_spread':a['_line'], 'away_spread_price':a['_price'],
        'total_line':o['_line'], 'over_price':o['_price'], 'under_price':u['_price'],
        'model_home_margin':h['_pred_home'], 'market_home_margin':market_home_margin,
        'model_total':h['_pred_total'],
    }
    result = []
    for market, gap, side, price in (
        ('spread',gap_spread,'home' if gap_spread > 0 else 'away' if gap_spread < 0 else 'none',
          h['_price'] if gap_spread > 0 else a['_price'] if gap_spread < 0 else None),
        ('total',gap_total,'over' if gap_total > 0 else 'under' if gap_total < 0 else 'none',
          o['_price'] if gap_total > 0 else u['_price'] if gap_total < 0 else None),
    ):
        row = dict(common, market=market, model_minus_market_points=round(gap, 3),
                   abs_disagreement_points=round(abs(gap), 3),
                   model_lean=side, lean_side_american_price=price,
                   lean_price_break_even_win_pct=round(100*implied_break_even(price),2) if price else '',
                   gap_ge_2=int(abs(gap)>=THRESHOLDS[0]),
                   gap_ge_4=int(abs(gap)>=THRESHOLDS[1]),
                   gap_ge_6=int(abs(gap)>=THRESHOLDS[2]),
                   is_paper_bet=False, estimated_ev='', calibrated_probability='')
        result.append(row)
    return result


def write_csv(path:Path, rows:list[dict]):
    if not rows:
        raise ValueError('Nothing to write')
    with path.open('x', newline='', encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)


def sha(path:Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_top(rows:list[dict], market:str, n:int=7):
    print(f'\n{market.upper()} DISAGREEMENTS (highest first; NOT betting recommendations)')
    print(f'  {"Game":<12} {"Market":>8} {"Model":>8} {"Gap":>8} {"Lean":>6} {"Odds":>6}')
    for r in sorted((r for r in rows if r['market']==market),
                    key=lambda r:(-r['abs_disagreement_points'],r['game_id'],r['bookmaker_key']))[:n]:
        book_val=r['market_home_margin'] if market=='spread' else r['total_line']
        model_val=r['model_home_margin'] if market=='spread' else r['model_total']
        price=f'{r["lean_side_american_price"]:+d}' if isinstance(r['lean_side_american_price'],int) else '—'
        print(f'  {r["matchup"]:<12} {book_val:>+8.1f} {model_val:>+8.1f} '
              f'{r["model_minus_market_points"]:>+8.2f} {r["model_lean"]:>6} {price:>6}')
    if market=='spread':
        print('  Market/Model are HOME winning margins; positive = home team favoured.')
    else:
        print('  Market/Model are full-game points; positive gap means model leans OVER.')


def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description='Read-only NFL model-versus-market exploratory filter analysis.')
    ap.add_argument('--input',type=Path,default=DEFAULT_INPUT,help='Export CSV created by tracker v0.7')
    ap.add_argument('--outdir',type=Path,default=DEFAULT_OUTDIR,help='Independent output folder; tracker not touched')
    args=ap.parse_args(argv)
    try:
        rows=load_csv(args.input)
        selected, older = select_latest_complete(rows)
        analysis=[]
        for group in selected:
            analysis.extend(canonicalize(group))
        analysis.sort(key=lambda r:(r['game_id'],r['bookmaker_key'],r['market']))
        n_games=len({r['game_id'] for r in analysis})
        n_books=len({r['bookmaker_key'] for r in analysis})
        if not analysis:
            raise ValueError('No valid complete quotes')
        # This study tests 1 observation per game/book/market, NOT per-side quotes.
        cohort_rows=[]
        for market in ('spread','total'):
            for side in (('home','away') if market=='spread' else ('over','under')):
                market_side=[r for r in analysis if r['market']==market and r['model_lean']==side]
                for t in THRESHOLDS:
                    matching=[r for r in market_side if r['abs_disagreement_points']>=t]
                    cohort_rows.append({
                        'market':market,'lean':side,'min_abs_gap_pts':t,
                        'candidate_game_book_observations':len(matching),
                        'total_game_book_observations':len([r for r in analysis if r['market']==market]),
                        'game_ids':' | '.join(r['game_id'] for r in matching),
                        'pre_results_exploratory_only':True, 'roi_not_estimated':True,
                    })
        now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        run=args.outdir/f'run_{now}'
        run.mkdir(parents=True,exist_ok=False)
        detailed=run/'model_market_disagreements.csv'
        cohorts=run/'exploratory_filter_cohorts.csv'
        manifest=run/'analysis_manifest.json'
        write_csv(detailed,analysis)
        write_csv(cohorts,cohort_rows)
        meta={
            'version':VERSION,'run_timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'source_csv_path':str(args.input), 'source_csv_sha256':sha(args.input),
            'raw_quote_rows':len(rows), 'unique_raw_snapshots':len({r['snapshot_id'] for r in rows}),
            'older_game_book_snapshot_groups_excluded':older,
            'complete_selected_game_book_groups':len(selected),
            'unique_games':n_games,'unique_books':n_books,'market_rows':len(analysis),
            'source_quote_updated_at_missing_groups':sum(not r['source_market_updated_at_known'] for r in analysis)//2,
            'thresholds_points':list(THRESHOLDS),
            'threshold_note':'Exploratory prospective candidate definitions; chosen after observing this Week 6 export, not previously validated.',
            'snapshot_rule':'Latest complete snapshot by game and bookmaker; fail if the latest snapshot is incomplete; no mixing sides/snapshots.',
            'time_rule':'Both forecast and observation strictly pregame, forecast no later than captured quote.',
            'market_rule':'One observation per game/book/market; compare MODEL to market, no postgame selection.',
            'spread_sign':'Book home margin = -home spread; gap = model home margin - market home margin.',
            'total_sign':'Gap = model total - posted total; positive=OVER, negative=UNDER.',
            'price_rule':'Lean-side American price and conditional break-even rate shown for context only; not an EV calculation.',
            'outcomes_consulted':False, 'probabilities_calibrated':False,'roi_estimated':False,
            'quote_quality':'ESPN research observations; bookmaker update time / Ontario bettability unverified.',
            'comparison_csv_sha256':sha(detailed), 'cohort_csv_sha256':sha(cohorts),
        }
        with manifest.open('x',encoding='utf-8') as f:
            json.dump(meta,f,indent=2);f.write('\n')
        print(f'NFL FILTER RESEARCH | {VERSION} | observational, NO wagers')
        print(f'Input: {args.input} | SHA256: {meta["source_csv_sha256"][:16]}…')
        print(f'Input: {len(rows)} individual price rows across {meta["unique_raw_snapshots"]} snapshots.')
        print(f'Latest complete selection: {len(selected)} game/book groups, {n_games} unique games, '
              f'{n_books} bookmakers; {older} older group(s) excluded.')
        print(f'Analyzed {len(analysis)} unique game/book/market cases (spread and total).')
        print(f'Quote-update time missing for {meta["source_quote_updated_at_missing_groups"]} of {len(selected)} selected groups.')
        format_top(analysis,'spread',7)
        format_top(analysis,'total',7)
        print('\nEXPLORATORY COHORT COUNTS (overlapping thresholds; NOT independent bets)')
        for t in THRESHOLDS:
            filtered=[r for r in analysis if r['abs_disagreement_points']>=t]
            sp=sum(r['market']=='spread' for r in filtered)
            to=sum(r['market']=='total' for r in filtered)
            over=sum(r['market']=='total' and r['model_lean']=='over' for r in filtered)
            under=sum(r['market']=='total' and r['model_lean']=='under' for r in filtered)
            print(f'  |gap| >= {t:g} pts: spread={sp}, total={to} (over={over}, under={under})')
        print(f'\nReports saved in: {run}')
        print('  model_market_disagreements.csv (one spread + one total row per game/book)')
        print('  exploratory_filter_cohorts.csv (prespecified display thresholds, by direction)')
        print('  analysis_manifest.json (input hash, selection rules, reproducibility)')
        print('NO EV or ROI: no calibrated win probabilities, no results, no verified Ontario wagers.')
        print('Thresholds are exploratory and cannot establish that disagreement causes an edge.')
        return 0
    except (OSError,ValueError,csv.Error) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
