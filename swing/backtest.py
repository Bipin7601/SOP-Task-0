"""Day-by-day portfolio simulator.

Each day, in this order:
  1. Exits for positions held from earlier days (time exit at the open, then stop, then target).
  2. Entries for orders created from yesterday's signals.
  3. Mark-to-market, drawdown check.
  4. New orders from today's signals for tomorrow.

Conservative assumptions, so the backtest errs on the side of looking worse than reality:
  * a gap through a stop or a target fills at the open, not at the level;
  * if one bar touches both stop and target, the stop is assumed to have hit first;
  * no target is credited on the entry day;
  * every fill pays slippage on top of Zerodha's charges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from . import costs
from .data import Panel
from .signals import Signals


@dataclass
class Order:
    symbol: str
    signal_date: pd.Timestamp
    ref_price: float        # price the order was sized from (signal close, or trigger level)
    trigger: float          # stop_trigger mode: buy-stop level
    stop: float
    qty: int


@dataclass
class Position:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    qty: int
    initial_stop: float
    stop: float
    target: float
    entry_cost: float
    bars_held: int = 0
    highest_close: float = 0.0
    at_breakeven: bool = False


@dataclass
class Trade:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    initial_stop: float
    reason: str
    costs: float
    pnl: float              # net of all costs
    r_multiple: float
    bars_held: int


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    halted_on: pd.Timestamp | None
    first_breach_on: pd.Timestamp | None
    config: object = field(repr=False, default=None)


def _val(frame: pd.DataFrame, i: int, sym: str) -> float:
    return frame.iat[i, frame.columns.get_loc(sym)]


def run(p: Panel, sig: Signals, cfg, start: str | None = None, end: str | None = None) -> Result:
    acct, ex, st, c = cfg.account, cfg.exits, cfg.setup, cfg.costs
    dates = p.dates
    lo = dates.searchsorted(pd.Timestamp(start)) if start else 0
    hi = dates.searchsorted(pd.Timestamp(end), side="right") if end else len(dates)

    cash = float(acct.starting_capital)
    positions: dict[str, Position] = {}
    orders: list[Order] = []
    trades: list[Trade] = []
    equity_curve = []
    peak = cash
    halted_on = first_breach_on = None
    last_close: dict[str, float] = {}

    def close_position(pos: Position, i: int, price: float, reason: str) -> None:
        nonlocal cash
        fill = costs.slip(price, c, "sell")
        value = fill * pos.qty
        sc = costs.sell_cost(value, c)
        cash += value - sc
        pnl = value - sc - (pos.entry_price * pos.qty + pos.entry_cost)
        risk = (pos.entry_price - pos.initial_stop) * pos.qty
        trades.append(Trade(
            symbol=pos.symbol, signal_date=pos.signal_date, entry_date=pos.entry_date,
            exit_date=dates[i], entry_price=pos.entry_price, exit_price=fill, qty=pos.qty,
            initial_stop=pos.initial_stop, reason=reason, costs=pos.entry_cost + sc, pnl=pnl,
            r_multiple=pnl / risk if risk > 0 else 0.0, bars_held=pos.bars_held,
        ))
        del positions[pos.symbol]

    for i in range(lo, hi):
        d = dates[i]

        # 1. Exits for positions entered on earlier days
        for pos in list(positions.values()):
            o, h, l = (_val(p.open, i, pos.symbol), _val(p.high, i, pos.symbol), _val(p.low, i, pos.symbol))
            if math.isnan(o):
                continue  # no bar (suspension/holiday for this stock): nothing can happen
            if pos.bars_held >= ex.max_hold_days:
                close_position(pos, i, o, "time")
            elif o <= pos.stop:
                close_position(pos, i, o, "stop_gap")
            elif l <= pos.stop:
                close_position(pos, i, pos.stop, "stop")
            elif o >= pos.target:
                close_position(pos, i, o, "target_gap")
            elif h >= pos.target:
                close_position(pos, i, pos.target, "target")

        # 2. Entries from yesterday's orders (already ranked best-first)
        for od in orders:
            if len(positions) >= acct.max_positions or od.symbol in positions:
                continue
            o, h, l, cl = (_val(p.open, i, od.symbol), _val(p.high, i, od.symbol),
                           _val(p.low, i, od.symbol), _val(p.close, i, od.symbol))
            if math.isnan(o):
                continue
            max_fill = od.ref_price * (1 + st.max_gap_pct / 100)
            if st.entry_mode == "next_open":
                price = o if o <= max_fill else None
            else:
                if o >= od.trigger:
                    price = o if o <= max_fill else None
                elif h >= od.trigger:
                    price = od.trigger
                else:
                    price = None
            if price is None or price <= od.stop:
                continue
            fill = costs.slip(price, c, "buy")
            qty = od.qty
            bc = costs.buy_cost(fill * qty, c)
            if fill * qty + bc > cash:
                qty = int((cash * 0.995) // fill)
                if qty < 1:
                    continue
                bc = costs.buy_cost(fill * qty, c)
            cash -= fill * qty + bc
            r = fill - od.stop
            pos = Position(
                symbol=od.symbol, signal_date=od.signal_date, entry_date=d, entry_price=fill, qty=qty,
                initial_stop=od.stop, stop=od.stop, target=fill + ex.target_r * r, entry_cost=bc,
                highest_close=cl if not math.isnan(cl) else fill,
            )
            positions[od.symbol] = pos
            # Entry-day stop check. next_open buys at the open, so the whole bar's low counts.
            # stop_trigger fills intraday with unknown ordering, so only a close below the stop counts.
            if st.entry_mode == "next_open" and l <= od.stop:
                close_position(pos, i, od.stop, "stop")
            elif st.entry_mode != "next_open" and cl <= od.stop:
                close_position(pos, i, od.stop, "stop")
        orders = []

        # 3. End-of-day bookkeeping: stops ratchet up, mark to market
        for pos in positions.values():
            cl = _val(p.close, i, pos.symbol)
            if math.isnan(cl):
                continue
            last_close[pos.symbol] = cl
            pos.bars_held += 1
            pos.highest_close = max(pos.highest_close, cl)
            r = pos.entry_price - pos.initial_stop
            if not pos.at_breakeven and cl >= pos.entry_price + ex.breakeven_at_r * r:
                pos.at_breakeven = True
                pos.stop = max(pos.stop, pos.entry_price)
            if pos.at_breakeven:
                a = _val(sig.atr14, i, pos.symbol)
                if not math.isnan(a):
                    pos.stop = max(pos.stop, pos.highest_close - ex.trail_atr_mult * a)

        equity = cash + sum(pos.qty * last_close.get(pos.symbol, pos.entry_price) for pos in positions.values())
        equity_curve.append((d, equity))
        peak = max(peak, equity)
        if (peak - equity) / peak * 100 >= acct.max_drawdown_halt_pct:
            first_breach_on = first_breach_on or d
            if acct.halt_on_drawdown and halted_on is None:
                halted_on = d

        # 4. Orders for tomorrow
        if halted_on is not None or not sig.regime_ok.iat[i]:
            continue
        slots = acct.max_positions - len(positions)
        if slots <= 0:
            continue
        row = sig.breakout.iloc[i]
        cands = [s for s in row.index[row.values] if s not in positions]
        cands.sort(key=lambda s: -_val(sig.rs_pct, i, s))
        for s in cands:
            if len(orders) >= slots:
                break
            ref = _val(p.close, i, s) if st.entry_mode == "next_open" else _val(sig.base_high, i, s) * 1.001
            stop = _val(sig.base_low, i, s) * (1 - ex.stop_buffer_pct / 100)
            stop_pct = (ref - stop) / ref * 100
            if stop_pct > ex.max_stop_pct:
                continue
            if stop_pct < ex.min_stop_pct:
                stop = ref * (1 - ex.min_stop_pct / 100)
            risk_amt = equity * acct.risk_per_trade_pct / 100
            qty = int(risk_amt // (ref - stop))
            qty = min(qty, int(equity * acct.max_position_pct / 100 // ref))
            if qty < 1:
                continue
            orders.append(Order(symbol=s, signal_date=d, ref_price=ref, trigger=ref, stop=stop, qty=qty))

    # Close anything still open at the last close, so every position is counted
    for pos in list(positions.values()):
        close_position(pos, hi - 1, last_close.get(pos.symbol, pos.entry_price), "end")
    if equity_curve:
        equity_curve[-1] = (equity_curve[-1][0], cash)

    eq = pd.Series(dict(equity_curve), name="equity", dtype=float)
    tr = pd.DataFrame([asdict(t) for t in trades])
    return Result(equity=eq, trades=tr, halted_on=halted_on, first_breach_on=first_breach_on, config=cfg)
