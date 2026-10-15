"""Synthetic market data for tests. These prices are made up — they test the code, not the strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swing import config
from swing.data import Panel, build_panel


def ohlcv_from_close(close: np.ndarray, dates: pd.DatetimeIndex, vol: float = 1e6, wiggle: float = 0.005) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": np.full(len(close), vol)}, index=dates)


def make_panel(stock_closes: dict[str, np.ndarray], index_close: np.ndarray | None = None,
               vols: dict[str, np.ndarray] | None = None) -> Panel:
    n = len(next(iter(stock_closes.values())))
    dates = pd.bdate_range("2015-01-01", periods=n)
    if index_close is None:
        index_close = np.linspace(100, 200, n)  # steady uptrend: market filter on
    frames = {}
    for s, c in stock_closes.items():
        df = ohlcv_from_close(c, dates)
        if vols and s in vols:
            df["volume"] = vols[s]
        frames[s] = df
    return build_panel(frames, ohlcv_from_close(index_close, dates))


def breakout_series(n: int = 400, base_start: int = 300, base_len: int = 20, jump_at: int = 320,
                    after: str = "up") -> tuple[np.ndarray, np.ndarray]:
    """Uptrend, a flat tight base, a high-volume breakout at `jump_at`, then a move up or down."""
    c = np.empty(n)
    c[:base_start] = np.linspace(100, 300, base_start)
    rng = np.random.default_rng(0)
    c[base_start:jump_at] = 300 + rng.uniform(-2, 2, jump_at - base_start)
    c[jump_at] = 312
    rest = n - jump_at - 1
    c[jump_at + 1:] = np.linspace(314, 400, rest) if after == "up" else np.linspace(310, 250, rest)
    vol = np.full(n, 1e6)
    vol[jump_at] = 3e6
    return c, vol


@pytest.fixture
def cfg():
    # Loose universe filters so small synthetic prices/volumes qualify; one stock means RS rank is trivial.
    return config.load(overrides={
        "universe": {"min_avg_traded_value_cr": 0, "min_history_days": 250},
        "trend_template": {"min_rs_percentile": 0},
        "setup": {"base_days": 15, "max_atr_contraction": 10},
    })
