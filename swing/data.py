"""Load daily OHLCV CSVs into wide panels.

On-disk layout (data/ is git-ignored):
    data/daily/<SYMBOL>.csv      columns: date,open,high,low,close,volume
    data/daily/NIFTY 50.csv      the market-filter index, same columns
    data/universe/nifty500.csv   NSE's constituent list (needs a `Symbol` column)

Prices must be adjusted for splits and bonuses; Kite's historical API already does this.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class Panel:
    """Wide frames sharing one date index (the index's trading calendar)."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    index_close: pd.Series

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


def read_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates("date").set_index("date").sort_index()
    return clean_ohlcv(df[list(FIELDS)].astype(float))


MAX_DAILY_RATIO = 10.0   # no NSE stock moves 10x (or to a tenth) in one session; such a jump is bad data
MAX_GAP_DAYS = 60        # a longer silence means the bars before it belong to another security (recycled token)
MIN_SESSION_COVERAGE = 0.5  # index dates on which fewer listed stocks than this traded are special sessions


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bars with non-positive prices, and everything before an impossible one-day jump or a long gap.

    Some vendor histories carry placeholder prices before a listing or relisting (e.g. 0.15 for months,
    then 553), or another security's bars years before the real listing (e.g. SBICARD trading in 2015).
    Left in, they count as history and make moving averages, 52-week lows and relative strength
    meaningless for the first year after the stock really starts trading.
    """
    px = ["open", "high", "low", "close"]
    df = df[(df[px] > 0).all(axis=1)]
    ratio = df["close"] / df["close"].shift(1)
    gap = df.index.to_series().diff().dt.days
    breaks = ratio.index[(ratio > MAX_DAILY_RATIO) | (ratio < 1 / MAX_DAILY_RATIO) | (gap > MAX_GAP_DAYS)]
    if len(breaks):
        df = df.loc[breaks[-1]:]
    return df


def read_universe(path: Path | None = None) -> list[str]:
    path = path or DATA_DIR / "universe" / "nifty500.csv"
    df = pd.read_csv(path)
    col = next(c for c in df.columns if c.strip().lower() == "symbol")
    return sorted(df[col].astype(str).str.strip().unique())


def build_panel(frames: dict[str, pd.DataFrame], index_frame: pd.DataFrame) -> Panel:
    """Align per-symbol frames onto the index calendar. Missing bars stay NaN (not forward-filled),
    so a stock that did not trade can never be entered or exited on that day.

    Index dates on which most listed stocks have no bar (Muhurat and other special sessions) are dropped:
    one such date would leave a hole in hundreds of stocks and blank every 200-day average for 200 days."""
    cal = index_frame.index
    if frames:
        has_bar = pd.DataFrame({s: pd.Series(True, index=df.index) for s, df in frames.items()}).reindex(cal)
        listed = has_bar.ffill().notna() & has_bar.bfill().notna()
        coverage = has_bar.fillna(False).sum(axis=1) / listed.sum(axis=1).clip(lower=1)
        cal = cal[(coverage >= MIN_SESSION_COVERAGE).to_numpy()]
        index_frame = index_frame.loc[cal]
    wide = {f: pd.DataFrame({s: df[f] for s, df in frames.items()}).reindex(cal) for f in FIELDS}
    return Panel(**wide, index_close=index_frame["close"])


def load_panel(
    data_dir: Path = DATA_DIR,
    index_symbol: str = "NIFTY 50",
    symbols: list[str] | None = None,
) -> Panel:
    daily = data_dir / "daily"
    index_frame = read_ohlcv(daily / f"{index_symbol}.csv")
    if symbols is None:
        symbols = [p.stem for p in sorted(daily.glob("*.csv")) if p.stem != index_symbol]
    frames = {}
    for s in symbols:
        p = daily / f"{s}.csv"
        if p.exists():
            frames[s] = read_ohlcv(p)
    if not frames:
        raise FileNotFoundError(f"No stock CSVs found in {daily}. Run `python -m swing.kite_data download` first.")
    return build_panel(frames, index_frame)
