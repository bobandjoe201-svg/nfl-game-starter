"""NFL Game Tracker v0.7 — prospective odds, forecasts, and paper-bet audit.

Requires Python 3.10+ standard library only. No real wagers, suggestions,
calibrated probabilities, or claimed ROI edge. Never edits v0.1–v0.6.

Usage: python nfl_game_tracker_v07.py --help
See NFL_v07_README.md before first live capture.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from uuid import uuid4
from zoneinfo import ZoneInfo

VERSION = 'nfl-tracker-v0.7.0'
DEFAULT_DB = Path('output/tracker_v07/tracker.sqlite')
TZ = ZoneInfo('America/Toronto')
API_URL = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/'
TEAM_NAMES = {
    'ARI':'Arizona Cardinals','ATL':'Atlanta Falcons','BAL':'Baltimore Ravens',
    'BUF':'Buffalo Bills','CAR':'Carolina Panthers','CHI':'Chicago Bears',
    'CIN':'Cincinnati Bengals','CLE':'Cleveland Browns','DAL':'Dallas Cowboys',
    'DEN':'Denver Broncos','DET':'Detroit Lions','GB':'Green Bay Packers',
    'HOU':'Houston Texans','IND':'Indianapolis Colts','JAX':'Jacksonville Jaguars',
    'KC':'Kansas City Chiefs','LV':'Las Vegas Raiders','LAC':'Los Angeles Chargers',
    'LA':'Los Angeles Rams','MIA':'Miami Dolphins','MIN':'Minnesota Vikings',
    'NE':'New England Patriots','NO':'New Orleans Saints','NYG':'New York Giants',
    'NYJ':'New York Jets','PHI':'Philadelphia Eagles','PIT':'Pittsburgh Steelers',
    'SF':'San Francisco 49ers','SEA':'Seattle Seahawks','TB':'Tampa Bay Buccaneers',
    'TEN':'Tennessee Titans','WAS':'Washington Commanders',
}
ALIASES = {v.casefold():k for k,v in TEAM_NAMES.items()}
ALIASES.update({k.casefold():k for k in TEAM_NAMES})
ALIASES.update({
    'los angeles rams':'LA','la rams':'LA','st louis rams':'LA',
    'los angeles chargers':'LAC','la chargers':'LAC',
    'jacksonville jaguars':'JAX','washington football team':'WAS',
    'washington redskins':'WAS','oakland raiders':'LV',
    'san francisco forty niners':'SF',
})


def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')


def utc_parse(value):
    if not value or not isinstance(value,str):
        raise ValueError('Missing required timezone-aware ISO-8601 timestamp')
    if re.fullmatch(r'\d{8}T\d{12}Z', value):  # v0.1 timestamp incl microseconds
        return datetime.strptime(value,'%Y%m%dT%H%M%S%fZ').replace(tzinfo=timezone.utc)
    d = datetime.fromisoformat(value.replace('Z','+00:00'))
    if d.tzinfo is None or d.utcoffset() is None:
        raise ValueError(f'UTC offset required in timestamp {value!r}')
    return d.astimezone(timezone.utc)


def team_code(name):
    return ALIASES.get(str(name).strip().casefold())


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def connect(dbpath):
    dbpath.parent.mkdir(parents=True,exist_ok=True)
    con = sqlite3.connect(dbpath)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA foreign_keys=ON')
    con.execute('PRAGMA busy_timeout=10000')
    con.executescript('''
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS imports (
      import_id TEXT PRIMARY KEY, frozen_at TEXT NOT NULL,
      source_filename TEXT NOT NULL, file_sha256 TEXT NOT NULL UNIQUE,
      rows_count INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS predictions (
      import_id TEXT NOT NULL REFERENCES imports(import_id),
      game_id TEXT NOT NULL, season INTEGER NOT NULL, week INTEGER NOT NULL,
      gameday TEXT NOT NULL, home_team TEXT NOT NULL, away_team TEXT NOT NULL,
      generated_at TEXT NOT NULL, model_version TEXT NOT NULL,
      pred_home_margin REAL NOT NULL, pred_total REAL NOT NULL,
      PRIMARY KEY(import_id,game_id)
    );
    CREATE TABLE IF NOT EXISTS snapshots (
      snapshot_id TEXT PRIMARY KEY, captured_at TEXT NOT NULL,
      origin TEXT NOT NULL, description TEXT NOT NULL,
      raw_sha256 TEXT NOT NULL, raw_path TEXT NOT NULL,
      quote_count INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS quotes (
      quote_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
      captured_at TEXT NOT NULL, event_id TEXT NOT NULL, kickoff_at TEXT NOT NULL,
      home_team TEXT NOT NULL, away_team TEXT NOT NULL, game_id TEXT,
      bookmaker_key TEXT NOT NULL, bookmaker_title TEXT NOT NULL,
      market TEXT NOT NULL, selection TEXT NOT NULL, line REAL NOT NULL,
      american_price INTEGER NOT NULL, market_updated_at TEXT,
      UNIQUE(snapshot_id,event_id,bookmaker_key,market,selection,line,american_price)
    );
    CREATE TABLE IF NOT EXISTS paper_picks (
      pick_id TEXT PRIMARY KEY, quote_id TEXT NOT NULL REFERENCES quotes(quote_id),
      import_id TEXT NOT NULL, game_id TEXT NOT NULL,
      recorded_at TEXT NOT NULL, strategy_tag TEXT NOT NULL, notes TEXT NOT NULL,
      stake_units REAL NOT NULL, UNIQUE(quote_id,strategy_tag)
    );
    CREATE TABLE IF NOT EXISTS grades (
      grade_id TEXT PRIMARY KEY, pick_id TEXT NOT NULL REFERENCES paper_picks(pick_id),
      graded_at TEXT NOT NULL, home_score INTEGER NOT NULL, away_score INTEGER NOT NULL,
      result TEXT NOT NULL, net_units REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_quotes_game ON quotes(game_id,bookmaker_key,market,selection,captured_at);
    CREATE INDEX IF NOT EXISTS ix_predictions_week ON predictions(season,week,home_team,away_team);
    ''')
    existing = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if existing and existing['value'] != '7':
        raise ValueError('Refusing to use database with a different schema version')
    con.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version','7')")
    # Audit records cannot be silently modified. Corrections to grades are new grade rows.
    for tab in ('imports','predictions','snapshots','quotes','paper_picks','grades'):
        con.execute(f'''CREATE TRIGGER IF NOT EXISTS {tab}_no_update
          BEFORE UPDATE ON {tab} BEGIN SELECT RAISE(ABORT,'append-only audit: update forbidden'); END''')
        con.execute(f'''CREATE TRIGGER IF NOT EXISTS {tab}_no_delete
          BEFORE DELETE ON {tab} BEGIN SELECT RAISE(ABORT,'append-only audit: delete forbidden'); END''')
    con.commit()
    return con


def cmd_init(args, con):
    print(f'Initialized {args.db} ({VERSION}). No historical odds or wagers have been generated.')


def single_prediction_file(args):
    if args.predictions:
        path=Path(args.predictions)
        if not path.is_file(): raise ValueError(f'Forecast file missing: {path}')
        return path
    directory=Path(args.predictions_dir)
    matches=sorted(directory.glob('week*_2026_predictions_*.csv'))
    if len(matches)!=1:
        raise ValueError(f'Found {len(matches)} eligible files in {directory}; use --predictions PATH to choose exactly one')
    return matches[0]


def cmd_freeze(args, con):
    path=single_prediction_file(args)
    raw=path.read_bytes(); hashvalue=sha_bytes(raw)
    if con.execute('SELECT 1 FROM imports WHERE file_sha256=?',(hashvalue,)).fetchone():
        print('Already frozen: exact file checksum is present. No changes made.'); return
    with path.open(newline='',encoding='utf-8-sig') as fp:
        reader=csv.DictReader(fp)
        required={'game_id','season','week','home_team','away_team','gameday',
                  'pred_home_margin','pred_total','model_version','generated_at_utc'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f'Prediction CSV missing {sorted(required-set(reader.fieldnames or []))}')
        rows=list(reader)
    if not rows: raise ValueError('Prediction CSV is empty')
    now=now_utc(); records=[]; seen=set()
    for r in rows:
        game_id=r['game_id'].strip()
        if game_id in seen: raise ValueError('Duplicate forecast game ID')
        seen.add(game_id)
        home=team_code(r['home_team']); away=team_code(r['away_team'])
        if not home or not away or home==away: raise ValueError(f'Unrecognized teams for {game_id}')
        generated=utc_parse(r['generated_at_utc'])
        if generated>now+timedelta(minutes=2): raise ValueError('Forecast appears to come from the future')
        day=date.fromisoformat(r['gameday'])
        season=int(r['season']); week=int(r['week'])
        if not re.fullmatch(r'\d{4}_\d{2}_[A-Z]+_[A-Z]+',game_id):
            raise ValueError(f'Unexpected game ID format: {game_id}')
        if game_id!=f'{season}_{week:02d}_{away}_{home}':
            raise ValueError(f'Game ID disagrees with home/away columns: {game_id}')
        margin=float(r['pred_home_margin']); total=float(r['pred_total'])
        if not -70<=margin<=70 or not 0<=total<=120: raise ValueError(f'Implausible projection in {game_id}')
        records.append((game_id,season,week,day.isoformat(),home,away,iso(generated),
                        r['model_version'].strip(),margin,total))
    ident='f_'+hashvalue[:20]
    with con:
        con.execute('INSERT INTO imports VALUES (?,?,?,?,?)',
                    (ident,iso(now),path.name,hashvalue,len(records)))
        con.executemany('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                        [(ident,*r) for r in records])
    print(f'Frozen {len(records)} existing forecasts from {path.name} as {ident}')
    print('Forecast generated_at timestamps preserved. Freezing does not turn an old forecast into a live one.')


def prediction_matches(con, home,away,kickoff):
    # No ambiguous matches; convert UTC kickoff to local football calendar day.
    localday=kickoff.astimezone(TZ).date().isoformat()
    matches=con.execute('''SELECT DISTINCT game_id FROM predictions
                  WHERE home_team=? AND away_team=? AND gameday=?''',
                  (home,away,localday)).fetchall()
    return matches[0]['game_id'] if len(matches)==1 else None


def numeric_price(raw):
    if isinstance(raw,bool): raise ValueError('Invalid American odds')
    val=float(raw)
    if not val.is_integer() or (-100 < val < 100):
        raise ValueError(f'American odds must be an integer with magnitude >= 100: {raw}')
    return int(val)


def normalize_api(payload, con, captured_at):
    if not isinstance(payload,list): raise ValueError('Expected The Odds API array of events')
    normalized=[]; omissions=0; inplay=0
    for event in payload:
        home=team_code(event.get('home_team',''))
        away=team_code(event.get('away_team',''))
        if not home or not away or home==away:
            omissions+=1; continue
        kickoff=utc_parse(event.get('commence_time',''))
        if kickoff<=captured_at: inplay+=1; continue
        game_id=prediction_matches(con,home,away,kickoff)
        for book in event.get('bookmakers',[]):
            book_key=str(book.get('key','')).strip()
            if not book_key: continue
            for market in book.get('markets',[]):
                kind=market.get('key')
                if kind not in ('spreads','totals'): continue
                market_update=market.get('last_update')
                try:
                    updated=iso(utc_parse(market_update)) if market_update else None
                except (TypeError,ValueError): updated=None
                for outcome in market.get('outcomes',[]):
                    name=str(outcome.get('name','')).strip()
                    if kind=='spreads':
                        selection='home' if team_code(name)==home else ('away' if team_code(name)==away else None)
                    else:
                        selection=name.lower() if name.lower() in ('over','under') else None
                    if not selection: omissions+=1; continue
                    try:
                        point=float(outcome['point']); price=numeric_price(outcome['price'])
                    except (KeyError,TypeError,ValueError): omissions+=1; continue
                    if not -80<=point<=150: omissions+=1; continue
                    normalized.append(dict(event_id=str(event.get('id','')),kickoff_at=iso(kickoff),
                        home_team=home,away_team=away,game_id=game_id,
                        bookmaker_key=book_key,bookmaker_title=str(book.get('title',book_key)),
                        market='spread' if kind=='spreads' else 'total',selection=selection,
                        line=point,american_price=price,market_updated_at=updated))
    return normalized,omissions,inplay


def normalize_manual(rows, con, captured_at):
    required={'event_id','kickoff_at_utc','home_team','away_team','bookmaker_key',
              'market','selection','line','american_price'}
    normalized=[]; omissions=0; inplay=0
    for line_no,r in enumerate(rows,2):
        if not required.issubset(r): raise ValueError('Manual CSV missing columns: '+', '.join(sorted(required-set(r))))
        if not any(str(v or '').strip() for v in r.values()): continue
        home=team_code(r['home_team']); away=team_code(r['away_team'])
        if not home or not away or home==away: raise ValueError(f'Unknown team at CSV line {line_no}')
        kickoff=utc_parse(r['kickoff_at_utc'])
        if kickoff<=captured_at: inplay+=1; continue
        market=str(r['market']).strip().lower(); selection=str(r['selection']).strip().lower()
        if market not in ('spread','total') or selection not in (('home','away') if market=='spread' else ('over','under')):
            raise ValueError(f'Invalid market/selection at CSV line {line_no}')
        point=float(r['line']); price=numeric_price(r['american_price'])
        if not -80<=point<=150: raise ValueError(f'Implausible point at line {line_no}')
        book=str(r['bookmaker_key']).strip()
        if not book: raise ValueError(f'Missing bookmaker at CSV line {line_no}')
        normalized.append(dict(event_id=str(r['event_id']).strip(),kickoff_at=iso(kickoff),
            home_team=home,away_team=away,game_id=prediction_matches(con,home,away,kickoff),
            bookmaker_key=book,bookmaker_title=str(r.get('bookmaker_title') or book),
            market=market,selection=selection,line=point,american_price=price,
            market_updated_at=None))
    return normalized,omissions,inplay


def save_snapshot(args, con, payload_bytes, rows, origin, description, omissions, inplay, capture_at):
    ident='s_'+uuid4().hex[:20]
    rawdir=args.db.parent/'raw'; rawdir.mkdir(parents=True,exist_ok=True)
    final=rawdir/(ident+'.json')
    # Do not silently modify existing raw archives; no API key is persisted.
    archive={'tracker_version':VERSION,'captured_at_utc':iso(capture_at),
             'source':origin,'description':description,'raw_data':json.loads(payload_bytes)}
    contents=(json.dumps(archive,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    with final.open('xb') as f: f.write(contents)
    seen=set(); data=[]
    for r in rows:
        if not r['event_id']: continue
        key=(r['event_id'],r['bookmaker_key'],r['market'],r['selection'],r['line'],r['american_price'])
        if key in seen: continue
        seen.add(key)
        data.append(('q_'+uuid4().hex[:20],ident,iso(capture_at),r['event_id'],
            r['kickoff_at'],r['home_team'],r['away_team'],r['game_id'],r['bookmaker_key'],
            r['bookmaker_title'],r['market'],r['selection'],r['line'],r['american_price'],r['market_updated_at']))
    try:
        with con:
            con.execute('INSERT INTO snapshots VALUES (?,?,?,?,?,?,?)',
                        (ident,iso(capture_at),origin,description,sha_bytes(contents),str(final),len(data)))
            con.executemany('INSERT INTO quotes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',data)
    except Exception:
        final.unlink(missing_ok=True)
        raise
    print(f'Snapshot {ident}: {len(data)} pregame spread/total price quotes captured at {iso(capture_at)}')
    print(f'  Source: {origin} | matched games with frozen predictions: {len(set(r[7] for r in data if r[7]))}')
    if inplay: print(f'  SAFETY: excluded {inplay} games already kicked off')
    if omissions: print(f'  Other skipped/unusable entries: {omissions}')
    print(f'  Immutable raw archive: {final}')
    if origin!='odds_api_live':
        print('  CAUTION: imported/manual data are NOT independently verified as a live offer.')
    if not data: print('  No pregame prices recorded; check your source, markets, bookmaker coverage or kickoff times.')


def cmd_capture(args,con):
    if args.source=='api':
        key=os.getenv('THE_ODDS_API_KEY','').strip()
        if not key: raise ValueError('THE_ODDS_API_KEY environment variable is missing. See README.')
        params={'apiKey':key,'regions':args.region,'markets':'spreads,totals','oddsFormat':'american','dateFormat':'iso'}
        if args.bookmakers: params['bookmakers']=args.bookmakers
        req=Request(API_URL+'?'+urlencode(params),headers={'User-Agent':'NFL-Research-Tracker-v07/1.0','Accept':'application/json'})
        try:
            with urlopen(req,timeout=25) as response:
                response_bytes=response.read(5_000_000)
                quota={k:response.headers.get(k,'?') for k in ['x-requests-remaining','x-requests-used','x-requests-last']}
        except HTTPError as exc:
            # Do NOT print the exception's URL: it contains the secret API key.
            raise ValueError(f'Odds API returned HTTP {exc.code}; check key, plan, region, and quota') from None
        except (URLError,TimeoutError) as exc:
            raise ValueError(f'Cannot reach odds API ({type(exc).__name__}); no snapshot recorded') from None
        captured=now_utc(); payload=json.loads(response_bytes)
        rows,omissions,inplay=normalize_api(payload,con,captured)
        save_snapshot(args,con,response_bytes,rows,'odds_api_live',
                      f'NFL region={args.region} bookmakers={args.bookmakers or "all"}',omissions,inplay,captured)
        print('  API quota headers:',', '.join(f'{k}={v}' for k,v in quota.items()))
    elif args.source=='json':
        if not args.input: raise ValueError('--input PATH required for --source json')
        source=Path(args.input); raw=source.read_bytes(); captured=now_utc()
        payload=json.loads(raw); rows,omissions,inplay=normalize_api(payload,con,captured)
        save_snapshot(args,con,raw,rows,'api_json_offline_import',source.name,omissions,inplay,captured)
    else:
        if not args.input: raise ValueError('--input PATH required for --source csv')
        source=Path(args.input)
        with source.open(newline='',encoding='utf-8-sig') as f:
            records=list(csv.DictReader(f))
        captured=now_utc()
        rows,omissions,inplay=normalize_manual(records,con,captured)
        blob=json.dumps(records,ensure_ascii=False,sort_keys=True).encode()
        save_snapshot(args,con,blob,rows,'manual_csv_import',source.name,omissions,inplay,captured)


def latest_prediction(con, game_id, cutoff):
    # Only forecasts generated no later than the contemporaneous price snapshot.
    return con.execute('''SELECT * FROM predictions WHERE game_id=? AND generated_at<=?
     ORDER BY generated_at DESC, import_id DESC LIMIT 1''',(game_id,cutoff)).fetchone()


def disagreement(pred,q):
    if not pred: return None
    if q['market']=='total':
        return float(pred['pred_total'])-float(q['line'])
    # The model's home winning margin versus bookmaker's home winning margin.
    book_home_margin = -float(q['line']) if q['selection']=='home' else float(q['line'])
    return float(pred['pred_home_margin'])-book_home_margin


def cmd_quotes(args,con):
    sql='SELECT * FROM quotes WHERE 1=1'; params=[]
    if args.game_id: sql+=' AND game_id=?';params.append(args.game_id)
    if args.bookmaker: sql+=' AND bookmaker_key=?';params.append(args.bookmaker)
    if args.market: sql+=' AND market=?';params.append(args.market)
    sql+=' ORDER BY captured_at DESC, kickoff_at,bookmaker_key,market LIMIT ?';params.append(args.limit)
    rows=con.execute(sql,params).fetchall()
    if not rows:
        print('No matching quotes. Freeze a forecast and capture pregame prices first.'); return
    print('QUOTE_ID                GAME                BOOK           SIDE     LINE   ODDS  GAP    CAPTURED UTC')
    for q in rows:
        pred=latest_prediction(con,q['game_id'],q['captured_at']) if q['game_id'] else None
        delta=disagreement(pred,q)
        gap=f'{delta:+.2f}' if delta is not None else 'N/A'
        line=f'{q["line"]:+.1f}' if q['market']=='spread' else f'{q["line"]:.1f}'
        side=q['selection'] if q['market']=='spread' else q['selection'][0].upper()
        print(f'{q["quote_id"]:<23} {str(q["game_id"] or "unmatched"):<19} {q["bookmaker_key"][:14]:<14} '
              f'{side:<6} {line:>6} {q["american_price"]:>5} {gap:>6} {q["captured_at"]}')
    print('GAP = model-minus-market estimate in points, NOT a calibrated probability or expected return.')


def cmd_paper(args,con):
    q=con.execute('SELECT * FROM quotes WHERE quote_id=?',(args.quote_id,)).fetchone()
    if not q: raise ValueError('Unknown quote ID. Run quotes to see captured IDs.')
    if not q['game_id']: raise ValueError('Unmatched game: cannot create auditable paper pick without frozen forecast')
    now=now_utc(); kickoff=utc_parse(q['kickoff_at']); captured=utc_parse(q['captured_at'])
    if now>=kickoff: raise ValueError('Kickoff already occurred: cannot retrospectively record a pregame paper pick')
    if captured>=kickoff: raise ValueError('Quote was captured after kickoff')
    age=(now-captured).total_seconds()/60
    if age < -1 or age>args.max_age_minutes:
        raise ValueError(f'Quote is {age:.1f} minutes old; recapture before recording a paper pick')
    if args.stake_units<=0 or args.stake_units>10: raise ValueError('Stake must be >0 and <=10 paper units')
    if not re.fullmatch(r'[a-z][a-z0-9_-]{1,50}',args.strategy_tag):
        raise ValueError('Strategy tag must be lowercase letters/digits/hyphen/underscore, 2-51 characters')
    pred=latest_prediction(con,q['game_id'],q['captured_at'])
    if not pred:
        raise ValueError('No frozen forecast generated before the quote was captured; no retroactive linking')
    # User manually initiates an exploration-only paper pick; no algorithm selects bets.
    ident='p_'+uuid4().hex[:20]
    with con:
        con.execute('INSERT INTO paper_picks VALUES (?,?,?,?,?,?,?,?)',
                    (ident,q['quote_id'],pred['import_id'],q['game_id'],iso(now),
                     args.strategy_tag,args.notes,args.stake_units))
    print(f'RECORDED PAPER-ONLY selection {ident} | {q["game_id"]} | {q["market"]} {q["selection"]} {q["line"]:+g} ({q["american_price"]:+d})')
    print(f'  Source {q["bookmaker_key"]}, observed {q["captured_at"]}; decision {iso(now)}')
    print('  NO calibrated probability, EV calculation, or actual wager was made.')


def profit_units(stake,american,result):
    if result=='win': return stake*(american/100 if american>0 else 100/abs(american))
    if result=='push': return 0.0
    return -stake


def latest_grades(con):
    rows=con.execute('''SELECT g.* FROM grades g
      WHERE g.rowid=(SELECT MAX(g2.rowid) FROM grades g2 WHERE g2.pick_id=g.pick_id)''').fetchall()
    return {r['pick_id']:r for r in rows}


def cmd_grade(args,con):
    path=Path(args.labels)
    with path.open(newline='',encoding='utf-8-sig') as fp:
        reader=csv.DictReader(fp)
        required={'game_id','home_score','away_score'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Label CSV requires game_id, home_score and away_score columns')
        scores={}
        for row in reader:
            if not row['home_score'] or not row['away_score']: continue
            game=row['game_id']
            if game in scores: raise ValueError(f'Duplicate result for {game}')
            hs=float(row['home_score']);aw=float(row['away_score'])
            if not (hs.is_integer() and aw.is_integer() and 0<=hs<=100 and 0<=aw<=100):
                raise ValueError('Invalid NFL game scores')
            scores[game]=(int(hs),int(aw))
    previous=latest_grades(con); counted=0; correction=0; pending=0
    picks=con.execute('''SELECT p.*, q.market,q.selection,q.line,q.american_price,q.kickoff_at
                         FROM paper_picks p JOIN quotes q ON p.quote_id=q.quote_id''').fetchall()
    now=now_utc()
    with con:
        for p in picks:
            result=scores.get(p['game_id'])
            if not result or utc_parse(p['kickoff_at'])>=now:
                pending+=1; continue
            h,a=result
            if p['market']=='spread':
                winning_margin=(h-a if p['selection']=='home' else a-h)+float(p['line'])
            elif p['selection']=='over': winning_margin=(h+a)-float(p['line'])
            else: winning_margin=float(p['line'])-(h+a)
            status='win' if winning_margin>1e-9 else ('loss' if winning_margin<-1e-9 else 'push')
            net=profit_units(float(p['stake_units']),int(p['american_price']),status)
            old=previous.get(p['pick_id'])
            if old:
                if (old['home_score'],old['away_score'],old['result'])==(h,a,status): continue
                correction+=1
            con.execute('INSERT INTO grades VALUES (?,?,?,?,?,?,?)',
                        ('g_'+uuid4().hex[:20],p['pick_id'],iso(now),h,a,status,net))
            counted+=1
    print(f'Graded {counted} new or corrected paper selections; {pending} lack final available results.')
    if correction: print(f'  {correction} result correction(s) appended; previous grades remain audited.')
    print('PAPER-ONLY realized outcomes; no evidence of expected positive ROI without prospective validation.')


def captured_close_proxy(con,q,max_minutes):
    kick=utc_parse(q['kickoff_at']); start=kick-timedelta(minutes=max_minutes)
    rows=con.execute('''SELECT line,american_price,captured_at FROM quotes
        WHERE game_id=? AND bookmaker_key=? AND market=? AND selection=?
          AND captured_at>=? AND captured_at<? AND captured_at>=?
        ORDER BY captured_at DESC LIMIT 1''',
        (q['game_id'],q['bookmaker_key'],q['market'],q['selection'],iso(start),iso(kick),q['captured_at'])).fetchall()
    if not rows: return None
    r=rows[0]
    if q['market']=='spread': clv=float(q['line'])-float(r['line'])
    elif q['selection']=='over': clv=float(r['line'])-float(q['line'])
    else: clv=float(q['line'])-float(r['line'])
    return clv, r


def cmd_report(args,con):
    total_imports=con.execute('SELECT COUNT(*) FROM imports').fetchone()[0]
    total_preds=con.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    captures=con.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0]
    quotes=con.execute('SELECT COUNT(*) FROM quotes').fetchone()[0]
    matches=con.execute('SELECT COUNT(DISTINCT game_id) FROM quotes WHERE game_id IS NOT NULL').fetchone()[0]
    print(f'{VERSION} | database {args.db}')
    print(f'Frozen files={total_imports} | forecasts={total_preds} | odds snapshots={captures} | individual prices={quotes} | matched games={matches}')
    grades=latest_grades(con)
    picks=con.execute('''SELECT p.*,q.* FROM paper_picks p JOIN quotes q ON p.quote_id=q.quote_id
                         ORDER BY p.recorded_at''').fetchall()
    if not picks:
        print('No paper picks. This is a tracking foundation, not a bet-generation system.'); return
    done=[(p,grades[p['pick_id']]) for p in picks if p['pick_id'] in grades]
    invested=sum(float(p['stake_units']) for p,g in done)
    net=sum(float(g['net_units']) for p,g in done)
    wins=sum(g['result']=='win' for p,g in done)
    losses=sum(g['result']=='loss' for p,g in done)
    pushes=sum(g['result']=='push' for p,g in done)
    print(f'PAPER picks={len(picks)} | graded={len(done)} | pending={len(picks)-len(done)}')
    if done:
        print(f'PAPER record {wins}-{losses}-{pushes} | staked={invested:.2f} units | net={net:+.3f} | sample ROI={100*net/invested:+.2f}%')
    vals=[]
    for p in picks:
        cp=captured_close_proxy(con,p,args.clv_window_minutes)
        if cp: vals.append(cp[0])
    print(f'Pregame closing PROXY coverage: {len(vals)}/{len(picks)} paper picks within {args.clv_window_minutes} minutes of kickoff')
    if vals: print(f'  Mean line-movement proxy: {sum(vals)/len(vals):+.3f} points (ignores changing odds prices)')
    print('CAUTION: small paper samples, stale/manual odds, and last-snapshot proxies do not establish tradable profit or closing-line value.')


def cmd_export(args,con):
    path=Path(args.out)
    if path.exists(): raise ValueError(f'Output exists: {path}; choose a new filename (exports never overwrite)')
    path.parent.mkdir(parents=True,exist_ok=True)
    rows=con.execute('SELECT * FROM quotes ORDER BY captured_at,game_id,bookmaker_key,market').fetchall()
    cols=['quote_id','snapshot_id','source','captured_at','event_id','kickoff_at',
          'home_team','away_team','game_id','bookmaker_key','bookmaker_title',
          'market','selection','line','american_price','market_updated_at',
          'model_version','pred_generated_at','pred_home_margin','pred_total','model_gap_pts']
    with path.open('x',encoding='utf-8',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=cols);writer.writeheader()
        for q in rows:
            pred=latest_prediction(con,q['game_id'],q['captured_at']) if q['game_id'] else None
            source=con.execute('SELECT origin FROM snapshots WHERE snapshot_id=?',(q['snapshot_id'],)).fetchone()[0]
            r={k:q[k] for k in cols if k in q.keys()}
            r['source']=source
            if pred:
                r.update(model_version=pred['model_version'],pred_generated_at=pred['generated_at'],
                         pred_home_margin=pred['pred_home_margin'],pred_total=pred['pred_total'],
                         model_gap_pts=disagreement(pred,q))
            writer.writerow(r)
    print(f'Exported {len(rows)} quote rows to {path} (includes provenance; not a winning-bet shortlist).')


def make_parser():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--db',type=Path,default=DEFAULT_DB,help='New append-only tracker database location')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('init',help='Initialize tracker database without fetching anything')
    p=sub.add_parser('freeze',help='Freeze existing v0.1 forecast file without changing it')
    p.add_argument('--predictions',help='Exact existing forecast CSV file')
    p.add_argument('--predictions-dir',default='output/model_v01',help='Autoselect only if directory contains exactly one matching 2026 forecast')
    p=sub.add_parser('capture',help='Capture one time-stamped bookmaker-price snapshot')
    p.add_argument('--source',choices=['api','json','csv'],default='api')
    p.add_argument('--input',help='Existing offline API JSON or manually recorded CSV')
    p.add_argument('--region',default='ca',help='The Odds API bookmaker region; ca for Canada/Ontario')
    p.add_argument('--bookmakers',help='Optional comma-separated provider bookmaker keys')
    p=sub.add_parser('quotes',help='List recently captured prices and their quote IDs')
    p.add_argument('--game-id');p.add_argument('--bookmaker');p.add_argument('--market',choices=['spread','total']);p.add_argument('--limit',type=int,default=30)
    p=sub.add_parser('paper',help='Manually log one exploratory paper-only selection before kickoff')
    p.add_argument('--quote-id',required=True)
    p.add_argument('--strategy-tag',required=True)
    p.add_argument('--notes',default='')
    p.add_argument('--stake-units',type=float,default=1.0)
    p.add_argument('--max-age-minutes',type=int,default=120)
    p=sub.add_parser('grade',help='Grade paper picks from updated historical_labels.csv, never placed wagers')
    p.add_argument('--labels',default='output/historical_labels.csv')
    p=sub.add_parser('report',help='Audit storage and summarize paper-only results')
    p.add_argument('--clv-window-minutes',type=int,default=90)
    p=sub.add_parser('export',help='Export captured quotes with contemporaneous model info')
    p.add_argument('--out',required=True)
    return parser


def main():
    parser=make_parser();args=parser.parse_args()
    try:
        con=connect(args.db)
        funcs={'init':cmd_init,'freeze':cmd_freeze,'capture':cmd_capture,
               'quotes':cmd_quotes,'paper':cmd_paper,'grade':cmd_grade,
               'report':cmd_report,'export':cmd_export}
        funcs[args.command](args,con)
    except (ValueError,sqlite3.Error,FileNotFoundError,OSError,KeyError,json.JSONDecodeError) as ex:
        print(f'ERROR: {ex}',file=sys.stderr)
        sys.exit(1)


if __name__=='__main__': main()
