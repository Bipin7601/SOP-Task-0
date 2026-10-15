"""Command line entry point.

    python -m swing.cli backtest              full period + in-sample / out-of-sample split
    python -m swing.cli scan                  today's candidates from the latest data (for paper/live runs)
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import backtest, config, data, metrics, signals

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def _fmt(stats: dict) -> str:
    keys = ["start", "end", "end_equity", "cagr_pct", "max_drawdown_pct", "sharpe", "trades", "win_rate_pct",
            "avg_win_r", "avg_loss_r", "expectancy_r", "profit_factor", "avg_bars_held", "total_costs",
            "longest_losing_streak"]
    lines = []
    for k in keys:
        if k in stats:
            v = stats[k]
            lines.append(f"| {k} | {v:,.2f} |" if isinstance(v, float) else f"| {k} | {v} |")
    return "| metric | value |\n|---|---|\n" + "\n".join(lines)


def cmd_backtest(args) -> None:
    cfg = config.load(args.config)
    panel = data.load_panel(Path(args.data_dir), index_symbol=cfg.market_filter.index_symbol)
    sig = signals.compute(panel, cfg)
    b = cfg.backtest

    periods = {
        "full": (b.start, b.end),
        "in_sample": (b.start, b.in_sample_end),
        "out_of_sample": (str(pd.Timestamp(b.in_sample_end) + pd.Timedelta(days=1)), b.end),
    }
    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Backtest report", "",
          f"Universe: {len(panel.symbols)} stocks. Entry mode: `{cfg.setup.entry_mode}`. "
          f"Risk per trade: {cfg.account.risk_per_trade_pct}%. Max positions: {cfg.account.max_positions}.", "",
          "Caveat: the universe is today's constituents, so delisted stocks are missing (survivorship bias).", ""]
    summary = {}
    for name, (s, e) in periods.items():
        res = backtest.run(panel, sig, cfg, s, e)
        stats = metrics.summarize(res.equity, res.trades)
        bench = panel.index_close.loc[s:e]
        stats["benchmark_cagr_pct"] = metrics.cagr_pct(bench)
        stats["benchmark_max_drawdown_pct"] = metrics.max_drawdown_pct(bench)
        stats["drawdown_limit_first_hit"] = res.first_breach_on.date().isoformat() if res.first_breach_on else None
        stats["halted_on"] = res.halted_on.date().isoformat() if res.halted_on else None
        if len(res.trades):
            stats["monte_carlo"] = metrics.monte_carlo(res.trades.r_multiple.to_numpy(),
                                                       cfg.account.risk_per_trade_pct, b.monte_carlo_runs)
        summary[name] = stats
        res.trades.to_csv(out_dir / f"trades_{name}.csv", index=False)
        res.equity.to_csv(out_dir / f"equity_{name}.csv")

        md += [f"## {name.replace('_', ' ').title()} ({s} → {e})", "", _fmt(stats), "",
               f"Nifty 50 buy-and-hold: CAGR {stats['benchmark_cagr_pct']:.2f}%, "
               f"max drawdown {stats['benchmark_max_drawdown_pct']:.2f}%.", "",
               f"10% drawdown limit first hit: {stats['drawdown_limit_first_hit'] or 'never'}"
               f"{' (trading halted)' if res.halted_on else ''}.", ""]
        if len(res.equity):
            yr = metrics.yearly_returns_pct(res.equity)
            md += ["| year | return % |", "|---|---|"] + [f"| {y} | {v:.2f} |" for y, v in yr.items()] + [""]
        if "monte_carlo" in stats:
            mc = stats["monte_carlo"]
            md += [f"Monte Carlo ({b.monte_carlo_runs} reshuffles): median max drawdown "
                   f"{mc['median_max_dd_pct']:.1f}%, 95th percentile {mc['p95_max_dd_pct']:.1f}%.", ""]

    verdict = metrics.promotion_check(summary["out_of_sample"], cfg.promotion)
    md += ["## Go / no-go for paper trading (judged on out-of-sample only)", ""]
    md += [f"- {'✅' if ok else '❌'} {k}" for k, ok in verdict.items() if k != "PASS"]
    md += ["", f"**Verdict: {'PASS — move to paper trading' if verdict['PASS'] else 'FAIL — do not trade this version'}**"]

    (out_dir / "report.md").write_text("\n".join(md))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n".join(md))
    print(f"\nSaved to {out_dir}")


def cmd_scan(args) -> None:
    """Signals at the latest close, sized for the given equity. Used by the evening run."""
    cfg = config.load(args.config)
    panel = data.load_panel(Path(args.data_dir), index_symbol=cfg.market_filter.index_symbol)
    sig = signals.compute(panel, cfg)
    i = len(panel.dates) - 1
    d = panel.dates[i]
    equity = args.equity or cfg.account.starting_capital
    print(f"Scan for {d.date()} | market filter: {'PASS' if sig.regime_ok.iat[i] else 'FAIL — no new trades'}")
    if not sig.regime_ok.iat[i]:
        return
    row = sig.breakout.iloc[i]
    cands = sorted(row.index[row.values], key=lambda s: -sig.rs_pct.at[d, s])
    rows = []
    for s in cands:
        ref = panel.close.at[d, s] if cfg.setup.entry_mode == "next_open" else sig.base_high.at[d, s] * 1.001
        stop = sig.base_low.at[d, s] * (1 - cfg.exits.stop_buffer_pct / 100)
        stop_pct = (ref - stop) / ref * 100
        if stop_pct > cfg.exits.max_stop_pct:
            continue
        if stop_pct < cfg.exits.min_stop_pct:
            stop = ref * (1 - cfg.exits.min_stop_pct / 100)
        qty = int(equity * cfg.account.risk_per_trade_pct / 100 // (ref - stop))
        qty = min(qty, int(equity * cfg.account.max_position_pct / 100 // ref))
        if qty < 1 or math.isnan(qty):
            continue
        rows.append({"symbol": s, "rs_pct": round(sig.rs_pct.at[d, s], 1), "entry_ref": round(ref, 2),
                     "stop": round(stop, 2), "target": round(ref + cfg.exits.target_r * (ref - stop), 2),
                     "qty": qty, "risk_inr": round(qty * (ref - stop), 2)})
    print(pd.DataFrame(rows).to_string(index=False) if rows else "No candidates today.")


def main() -> None:
    ap = argparse.ArgumentParser(prog="swing")
    ap.add_argument("--config", default=str(config.DEFAULT_PATH))
    ap.add_argument("--data-dir", default=str(data.DATA_DIR))
    ap.add_argument("--out", default=str(REPORTS), help="where backtest reports are written")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backtest").set_defaults(fn=cmd_backtest)
    sc = sub.add_parser("scan")
    sc.add_argument("--equity", type=float, default=None)
    sc.set_defaults(fn=cmd_scan)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
