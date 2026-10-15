"""Vectorised indicators on wide frames (index = dates, columns = symbols).

Every value at row t uses only bars up to and including t — no look-ahead.
"""

from __future__ import annotations

import pandas as pd


def sma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=n).mean()


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    prev_close = close.shift(1)
    a = high - low
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return a.where(a >= b, b).where(lambda t: t >= c, c)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Wilder's ATR."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rs_score(close: pd.DataFrame) -> pd.DataFrame:
    """IBD-style relative strength: 40% weight on the last quarter, 20% on each of the three before."""
    r63 = close / close.shift(63)
    r126 = close / close.shift(126)
    r189 = close / close.shift(189)
    r252 = close / close.shift(252)
    return 0.4 * r63 + 0.2 * r126 + 0.2 * r189 + 0.2 * r252


def cross_sectional_percentile(x: pd.DataFrame, mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rank each row 0-100 across columns. Cells outside `mask` are excluded from the ranking."""
    if mask is not None:
        x = x.where(mask)
    return x.rank(axis=1, pct=True) * 100
