"""Performance statistics, Monte Carlo drawdown risk, and the go/no-go check."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float(((peak - equity) / peak).max() * 100)


def cagr_pct(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float(((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100)


def sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    if r.std() == 0 or r.empty:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def longest_losing_streak(pnl: pd.Series) -> int:
    best = cur = 0
    for x in pnl:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


def summarize(equity: pd.Series, trades: pd.DataFrame) -> dict:
    out = {
        "start": equity.index[0].date().isoformat() if len(equity) else None,
        "end": equity.index[-1].date().isoformat() if len(equity) else None,
        "start_equity": float(equity.iloc[0]) if len(equity) else 0.0,
        "end_equity": float(equity.iloc[-1]) if len(equity) else 0.0,
        "cagr_pct": cagr_pct(equity),
        "max_drawdown_pct": max_drawdown_pct(equity),
        "sharpe": sharpe(equity),
        "trades": int(len(trades)),
    }
    if len(trades):
        wins, losses = trades[trades.pnl > 0], trades[trades.pnl <= 0]
        gross_loss = -losses.pnl.sum()
        out.update({
            "win_rate_pct": float(len(wins) / len(trades) * 100),
            "avg_win_r": float(wins.r_multiple.mean()) if len(wins) else 0.0,
            "avg_loss_r": float(losses.r_multiple.mean()) if len(losses) else 0.0,
            "expectancy_r": float(trades.r_multiple.mean()),
            "profit_factor": float(wins.pnl.sum() / gross_loss) if gross_loss > 0 else float("inf"),
            "avg_bars_held": float(trades.bars_held.mean()),
            "total_costs": float(trades.costs.sum()),
            "longest_losing_streak": longest_losing_streak(trades.sort_values("exit_date").pnl),
            "exit_reasons": trades.reason.value_counts().to_dict(),
        })
    return out


def yearly_returns_pct(equity: pd.Series) -> pd.Series:
    year_end = equity.groupby(equity.index.year).last()
    prev = year_end.shift(1)
    prev.iloc[0] = equity.iloc[0]
    return (year_end / prev - 1) * 100


def monte_carlo(r_multiples: np.ndarray, risk_pct: float, runs: int = 2000, seed: int = 7) -> dict:
    """Reshuffle the order of trade outcomes to see how bad drawdowns could plausibly get.

    Each trade compounds equity by r * risk_pct. Positions overlap in reality, so this is a
    simplification — it answers "what if the same trades had come in a different order?"
    """
    if len(r_multiples) == 0:
        return {}
    rng = np.random.default_rng(seed)
    dds, finals = np.empty(runs), np.empty(runs)
    for k in range(runs):
        seq = rng.permutation(r_multiples)
        eq = np.cumprod(1 + seq * risk_pct / 100)
        eq = np.concatenate([[1.0], eq])
        peak = np.maximum.accumulate(eq)
        dds[k] = ((peak - eq) / peak).max() * 100
        finals[k] = eq[-1]
    return {
        "median_max_dd_pct": float(np.median(dds)),
        "p95_max_dd_pct": float(np.percentile(dds, 95)),
        "worst_max_dd_pct": float(dds.max()),
        "p05_final_multiple": float(np.percentile(finals, 5)),
        "median_final_multiple": float(np.median(finals)),
    }


def promotion_check(stats: dict, promo) -> dict:
    """Pass/fail for each go-live criterion in config `promotion`."""
    checks = {
        f"trades >= {promo.min_trades}": stats.get("trades", 0) >= promo.min_trades,
        f"profit factor >= {promo.min_profit_factor}": stats.get("profit_factor", 0) >= promo.min_profit_factor,
        f"expectancy >= {promo.min_expectancy_r}R": stats.get("expectancy_r", -1) >= promo.min_expectancy_r,
        f"max drawdown <= {promo.max_drawdown_pct}%": stats.get("max_drawdown_pct", 100) <= promo.max_drawdown_pct,
    }
    checks["PASS"] = all(checks.values())
    return checks
