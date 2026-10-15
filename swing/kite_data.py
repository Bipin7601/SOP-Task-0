"""Download daily candles from the Kite Connect historical API.

Needs the KITE_API_KEY and KITE_API_SECRET environment variables (set them as environment secrets,
never in code). Kite tokens last one trading day, so log in once per day:

    python -m swing.kite_data login                 prints the login URL, then asks for the redirect URL
    python -m swing.kite_data download              all symbols in data/universe/nifty500.csv + NIFTY 50
    python -m swing.kite_data download --symbols INFY TCS

Re-running `download` only fetches bars newer than what is already on disk.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .data import DATA_DIR, read_universe

TOKEN_FILE = DATA_DIR / ".kite_token"
CHUNK_DAYS = 1900          # Kite allows up to 2000 days of daily candles per request
REQUEST_GAP_S = 0.35       # stay under the 3 requests/second historical-data limit
HISTORY_START = date(2010, 1, 1)


def _client():
    from kiteconnect import KiteConnect

    kite = KiteConnect(api_key=os.environ["KITE_API_KEY"])
    return kite


def login(redirect_or_token: str | None = None) -> None:
    kite = _client()
    if not redirect_or_token:
        print("Open this URL, log in, then paste the full address you land on:\n")
        print(kite.login_url())
        redirect_or_token = input("\nRedirect URL or request_token: ").strip()
    if redirect_or_token.startswith("http"):
        redirect_or_token = parse_qs(urlparse(redirect_or_token).query)["request_token"][0]
    session = kite.generate_session(redirect_or_token, api_secret=os.environ["KITE_API_SECRET"])
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(f"{date.today().isoformat()}\n{session['access_token']}\n")
    TOKEN_FILE.chmod(0o600)
    print(f"Logged in as {session.get('user_id')}. Token saved for today.")


def authed_client():
    if not TOKEN_FILE.exists():
        raise SystemExit("Not logged in. Run `python -m swing.kite_data login` first.")
    day, token = TOKEN_FILE.read_text().split()
    if day != date.today().isoformat():
        print("Warning: token is from an earlier day and has probably expired; log in again if requests fail.")
    kite = _client()
    kite.set_access_token(token)
    return kite


def _fetch(kite, token: int, start: date, end: date) -> pd.DataFrame:
    frames = []
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=CHUNK_DAYS), end)
        rows = kite.historical_data(token, datetime.combine(cur, datetime.min.time()),
                                    datetime.combine(stop, datetime.max.time()), "day")
        if rows:
            frames.append(pd.DataFrame(rows))
        cur = stop + timedelta(days=1)
        time.sleep(REQUEST_GAP_S)
    if not frames:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.date
    return df[["date", "open", "high", "low", "close", "volume"]]


def download(symbols: list[str], index_symbol: str = "NIFTY 50") -> None:
    kite = authed_client()
    tokens = {i["tradingsymbol"]: i["instrument_token"] for i in kite.instruments("NSE")}
    out = DATA_DIR / "daily"
    out.mkdir(parents=True, exist_ok=True)
    today = date.today()
    missing = []
    for n, sym in enumerate([index_symbol] + symbols, 1):
        # Stocks moved to the trade-to-trade segment are listed as e.g. HFCL-BE; same history, saved as HFCL
        tok = next((tokens[s] for s in (sym, f"{sym}-BE", f"{sym}-BZ") if s in tokens), None)
        if tok is None:
            missing.append(sym)
            continue
        path = out / f"{sym}.csv"
        start = HISTORY_START
        old = None
        if path.exists():
            old = pd.read_csv(path)
            if len(old):
                start = pd.to_datetime(old["date"]).max().date() + timedelta(days=1)
        if start > today:
            continue
        new = _fetch(kite, tok, start, today)
        df = new if old is None else pd.concat([old, new.astype({"date": str})])
        df.drop_duplicates("date", keep="last").to_csv(path, index=False)
        print(f"[{n}/{len(symbols) + 1}] {sym}: {len(new)} new bars")
    if missing:
        print(f"Not found on NSE (renamed or delisted?): {', '.join(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="kite_data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("login")
    lg.add_argument("redirect", nargs="?", help="redirect URL or request_token (prompted if omitted)")
    dl = sub.add_parser("download")
    dl.add_argument("--symbols", nargs="*", help="default: data/universe/nifty500.csv")
    args = ap.parse_args()
    if args.cmd == "login":
        login(args.redirect)
    else:
        download(args.symbols or read_universe())


if __name__ == "__main__":
    main()
