"""Turn a price panel into daily breakout signals.

A signal on date t is decided with bars up to and including t's close only. The simulator
acts on it from t+1 onwards, which is exactly what the evening scan does live.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import indicators as ind
from .data import Panel


@dataclass
class Signals:
    breakout: pd.DataFrame        # bool: all conditions met at t's close
    base_high: pd.DataFrame       # breakout level (highest high of the base, excluding t)
    base_low: pd.DataFrame        # lowest low of the base, excluding t
    rs_pct: pd.DataFrame          # relative-strength percentile, used to rank competing signals
    atr14: pd.DataFrame           # for trailing stops
    regime_ok: pd.Series          # bool: market filter passes at t's close
    eligible: pd.DataFrame        # bool: liquid, priced and old enough to trade


def compute(p: Panel, cfg) -> Signals:
    close, high, low, vol = p.close, p.high, p.low, p.volume
    u, tt, st = cfg.universe, cfg.trend_template, cfg.setup

    # Market regime
    idx = p.index_close
    regime_ok = (idx > idx.rolling(cfg.market_filter.ma_days, min_periods=cfg.market_filter.ma_days).mean())

    # Tradable universe
    traded_value_cr = (close * vol).rolling(20, min_periods=20).mean() / 1e7
    history = close.notna().cumsum()
    eligible = (close >= u.min_price) & (traded_value_cr >= u.min_avg_traded_value_cr) & (history >= u.min_history_days)

    # Trend template
    sma50, sma150, sma200 = ind.sma(close, 50), ind.sma(close, 150), ind.sma(close, 200)
    low52 = low.rolling(252, min_periods=200).min()
    high52 = high.rolling(252, min_periods=200).max()
    trend = (sma200 > sma200.shift(tt.sma200_rising_days))
    if tt.require_ma_stack:
        trend &= (close > sma50) & (sma50 > sma150) & (sma150 > sma200)
    trend &= close >= low52 * (1 + tt.min_above_52w_low_pct / 100)
    trend &= close >= high52 * (1 - tt.max_below_52w_high_pct / 100)

    rs_pct = ind.cross_sectional_percentile(ind.rs_score(close), mask=eligible)
    trend &= rs_pct >= tt.min_rs_percentile

    # Base: the `base_days` bars before t (t itself is the breakout bar)
    base_high = high.shift(1).rolling(st.base_days, min_periods=st.base_days).max()
    base_low = low.shift(1).rolling(st.base_days, min_periods=st.base_days).min()
    depth_pct = (base_high - base_low) / base_high * 100
    atr10 = ind.atr(high, low, close, 10).shift(1)
    atr50 = ind.atr(high, low, close, 50).shift(1)
    tight = (depth_pct <= st.max_base_depth_pct) & (atr10 / atr50 <= st.max_atr_contraction)

    avg_vol50 = vol.shift(1).rolling(50, min_periods=50).mean()
    if st.entry_mode == "next_open":
        # Confirmed breakout: closed above the base on heavy volume
        trigger = (close > base_high) & (vol >= avg_vol50 * st.breakout_volume_mult)
        level_high, level_low = base_high, base_low
    else:
        # stop_trigger: the stock is coiled at t's close, below its base high; the order sits above it.
        # The base for tomorrow's breakout includes today's bar.
        level_high = high.rolling(st.base_days, min_periods=st.base_days).max()
        level_low = low.rolling(st.base_days, min_periods=st.base_days).min()
        depth_pct = (level_high - level_low) / level_high * 100
        tight = (depth_pct <= st.max_base_depth_pct) & (ind.atr(high, low, close, 10) / ind.atr(high, low, close, 50) <= st.max_atr_contraction)
        trigger = close >= level_high * 0.97

    breakout = (eligible & trend & tight & trigger).fillna(False)
    breakout = breakout & regime_ok.reindex(close.index).fillna(False).values[:, None]

    return Signals(
        breakout=breakout.astype(bool),
        base_high=level_high,
        base_low=level_low,
        rs_pct=rs_pct,
        atr14=ind.atr(high, low, close, 14),
        regime_ok=regime_ok.reindex(close.index).fillna(False),
        eligible=eligible.fillna(False),
    )
