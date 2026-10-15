import numpy as np
import pandas as pd
import pytest

from swing import backtest, config, costs, metrics, signals
from swing.data import build_panel

from conftest import breakout_series, make_panel, ohlcv_from_close


def test_zerodha_delivery_costs(cfg):
    c = cfg.costs
    # 10,000 buy: STT 10 + stamp 1.5 + exchange/SEBI 0.317 + GST on those 0.057
    assert costs.buy_cost(10_000, c) == pytest.approx(11.874, abs=0.01)
    # 10,000 sell: STT 10 + exchange/SEBI 0.317 + GST 0.057 + DP 15.34
    assert costs.sell_cost(10_000, c) == pytest.approx(25.714, abs=0.01)
    assert costs.slip(100, c, "buy") == pytest.approx(100.1)
    assert costs.slip(100, c, "sell") == pytest.approx(99.9)


def test_breakout_fires_on_breakout_day_only(cfg):
    close, vol = breakout_series()
    p = make_panel({"AAA": close}, vols={"AAA": vol})
    sig = signals.compute(p, cfg)
    fired = np.flatnonzero(sig.breakout["AAA"].to_numpy())
    assert 320 in fired
    assert not any(300 <= f < 320 for f in fired)  # nothing inside the base


def test_signals_have_no_lookahead(cfg):
    close, vol = breakout_series()
    full = make_panel({"AAA": close}, vols={"AAA": vol})
    cut = make_panel({"AAA": close[:321]}, vols={"AAA": vol[:321]})
    a = signals.compute(full, cfg).breakout["AAA"].iloc[:321]
    b = signals.compute(cut, cfg).breakout["AAA"]
    pd.testing.assert_series_equal(a, b)


def test_volume_confirmation_required(cfg):
    close, vol = breakout_series()
    vol[:] = 1e6  # breakout on ordinary volume
    p = make_panel({"AAA": close}, vols={"AAA": vol})
    assert not signals.compute(p, cfg).breakout["AAA"].iloc[320]


def test_market_filter_blocks_trades(cfg):
    close, vol = breakout_series()
    falling_index = np.linspace(200, 100, len(close))
    p = make_panel({"AAA": close}, index_close=falling_index, vols={"AAA": vol})
    res = backtest.run(p, signals.compute(p, cfg), cfg)
    assert res.trades.empty


def test_winning_trade_enters_next_open_and_profits(cfg):
    close, vol = breakout_series(after="up")
    p = make_panel({"AAA": close}, vols={"AAA": vol})
    res = backtest.run(p, signals.compute(p, cfg), cfg)
    t = res.trades.iloc[0]
    assert t.entry_date == p.dates[321]
    assert t.entry_price == pytest.approx(p.open["AAA"].iloc[321] * 1.001)
    assert t.pnl > 0 and t.r_multiple > 1


def test_losing_trade_risks_about_one_and_a_half_percent(cfg):
    close, vol = breakout_series(after="down")
    p = make_panel({"AAA": close}, vols={"AAA": vol})
    res = backtest.run(p, signals.compute(p, cfg), cfg)
    t = res.trades.iloc[0]
    assert t.reason in ("stop", "stop_gap")
    loss_pct = -t.pnl / cfg.account.starting_capital * 100
    # 1.5% planned risk, plus costs and slippage, minus share rounding
    assert 0.8 < loss_pct < 2.2


def test_gap_through_stop_fills_at_open(cfg):
    close, vol = breakout_series(after="up")
    close = close.copy()
    close[325:] = 200.0
    p = make_panel({"AAA": close}, vols={"AAA": vol})
    for f in ("open", "high", "low"):
        getattr(p, f).iloc[325:, 0] = 200.0  # opens far below the stop
    res = backtest.run(p, signals.compute(p, cfg), cfg)
    t = res.trades.iloc[0]
    assert t.reason == "stop_gap"
    assert t.exit_price < t.initial_stop
    assert t.r_multiple < -1  # gaps can lose more than 1R; the backtest must show it


def test_max_positions_respected(cfg):
    close, vol = breakout_series(after="up")
    stocks = {f"S{k}": close * (1 + k / 100) for k in range(6)}
    vols = {s: vol for s in stocks}
    p = make_panel(stocks, vols=vols)
    res = backtest.run(p, signals.compute(p, cfg), cfg)
    assert 0 < len(res.trades) <= cfg.account.max_positions


def test_drawdown_halt_stops_new_trades():
    cfg = config.load(overrides={
        "universe": {"min_avg_traded_value_cr": 0, "min_history_days": 250},
        "trend_template": {"min_rs_percentile": 0},
        "setup": {"max_atr_contraction": 10},
        "account": {"max_drawdown_halt_pct": 0.5, "halt_on_drawdown": True},
    })
    close, vol = breakout_series(after="down")
    p = make_panel({"AAA": close}, vols={"AAA": vol})
    res = backtest.run(p, signals.compute(p, cfg), cfg)
    assert res.halted_on is not None


def test_drawdown_breach_recorded_without_halting():
    cfg = config.load(overrides={
        "universe": {"min_avg_traded_value_cr": 0, "min_history_days": 250},
        "trend_template": {"min_rs_percentile": 0},
        "setup": {"max_atr_contraction": 10},
        "account": {"max_drawdown_halt_pct": 0.5, "halt_on_drawdown": False},
    })
    close, vol = breakout_series(after="down")
    p = make_panel({"AAA": close}, vols={"AAA": vol})
    res = backtest.run(p, signals.compute(p, cfg), cfg)
    assert res.first_breach_on is not None and res.halted_on is None


def test_missing_bars_are_not_traded(cfg):
    close, vol = breakout_series(after="up")
    p = make_panel({"AAA": close}, vols={"AAA": vol})
    for f in ("open", "high", "low", "close", "volume"):
        getattr(p, f).iloc[321, 0] = np.nan  # suspended the day after the signal
    res = backtest.run(p, signals.compute(p, cfg), cfg)
    assert res.trades.empty or res.trades.iloc[0].entry_date != p.dates[321]


def test_metrics_and_monte_carlo():
    eq = pd.Series([100, 110, 99, 120], index=pd.bdate_range("2020-01-01", periods=4), dtype=float)
    assert metrics.max_drawdown_pct(eq) == pytest.approx(10.0)
    mc = metrics.monte_carlo(np.array([2.5, -1, -1, 2.5, -1]), risk_pct=1.5, runs=200)
    assert mc["worst_max_dd_pct"] >= mc["median_max_dd_pct"] > 0
    assert metrics.longest_losing_streak(pd.Series([1, -1, -2, 0, 3, -1])) == 3


def test_promotion_check(cfg):
    good = {"trades": 40, "profit_factor": 1.6, "expectancy_r": 0.3, "max_drawdown_pct": 12}
    bad = dict(good, profit_factor=1.1)
    assert metrics.promotion_check(good, cfg.promotion)["PASS"]
    assert not metrics.promotion_check(bad, cfg.promotion)["PASS"]


def test_stop_trigger_mode_runs(cfg):
    cfg2 = config.load(overrides={
        "universe": {"min_avg_traded_value_cr": 0, "min_history_days": 250},
        "trend_template": {"min_rs_percentile": 0},
        "setup": {"entry_mode": "stop_trigger", "max_atr_contraction": 10},
    })
    close, vol = breakout_series(after="up")
    p = make_panel({"AAA": close}, vols={"AAA": vol})
    res = backtest.run(p, signals.compute(p, cfg2), cfg2)
    assert len(res.trades) >= 1
    for t in res.trades.itertuples():
        k = p.dates.get_loc(t.entry_date)
        # bought at or above the prior base high, never inside the base
        assert t.entry_price >= p.high["AAA"].iloc[k - cfg2.setup.base_days:k].max()
